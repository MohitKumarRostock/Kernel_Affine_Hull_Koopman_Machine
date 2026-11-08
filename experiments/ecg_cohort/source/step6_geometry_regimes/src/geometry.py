"""Native input-only KAHM geometry families for the Step 6 regime study.

Only ``fit_family`` fits anything. Importing this module reads no files and
creates no data. ``initialize_runtime`` verifies a pinned source manifest and
loads its native helpers; it never reads historical inputs or result arrays.

The maximal local cap is fitted by the native state-regime routine. Lower
caps refit the same native autoencoder on its exact retained reference
columns, so KMeans, singleton augmentation and reference selection are shared.
The control randomly regroups that same reference multiset, preserving each
state's reference count, before calling the same native autoencoder.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import numbers
import os
from pathlib import Path
import pickle
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "OMP_THREAD_LIMIT"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import numpy as np
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT.parent / "step1_geometry_pilot" / "vendor" / "dimension_certificates"
MANIFEST_SHA256 = "333071e4a72ac07777853ad61ab5fb53408687cd354016bb125e0dd2c011d0a7"
_NATIVE_FILES = {
    "kahm_dimension_policy": "runtime/policy/kahm_dimension_policy.py",
    **{name: "runtime/risk/" + name + ".py" for name in (
        "dimension_policy_bridge", "distance_matrix", "dim_reduce",
        "kernel_regularized_least_squares", "kxx_matrix", "kxa_matrix",
        "autoencoder", "autoencoder_filtering_extended", "parallel_autoencoders",
        "combine_multiple_autoencoders_extended", "kernel_affine_hull_koopman_machines",
    )},
}
_RUNTIME = None
_THREAD_LIMITER = None
_ENCODER_ARRAYS = ("y_data_n_subspace", "PC", "B", "y_data", "sqrt_weights_matrix")


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.shape).encode("ascii"))
    h.update(value.dtype.str.encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def source_hashes():
    """Verify and return only pinned native source dependencies, relative to parent.

    The source manifest's own digest is fixed in this module. No historical
    configuration, selected-forecast record, data array or model is opened.
    """
    manifest_path = VENDOR / "runtime_sources.json"
    actual = _file_digest(manifest_path)
    if actual != MANIFEST_SHA256:
        raise RuntimeError("Pinned native source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = {str(manifest_path.relative_to(ROOT.parent)): actual}
    for name in _NATIVE_FILES.values():
        path = VENDOR / name
        digest = _file_digest(path)
        if digest != manifest["files"][name]["sha256"]:
            raise RuntimeError("Pinned native source changed: " + name)
        checked[str(path.relative_to(ROOT.parent))] = digest
    return checked


def _check_imports():
    result = {}
    for name, relative in _NATIVE_FILES.items():
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        filename = getattr(loaded, "__file__", None)
        expected = (VENDOR / relative).resolve()
        if filename is None or Path(filename).resolve() != expected:
            raise ImportError("Foreign native module " + name + "; use an isolated worker")
        result[name] = str(expected.relative_to(ROOT.parent))
    return result


def initialize_runtime():
    """Verify sources and imports, force one thread, return native callables.

    Call explicitly before timing. Normal query calls use the already loaded
    runtime and do not repeat source hashing or inspect any data dependencies.
    """
    global _RUNTIME, _THREAD_LIMITER
    verified = source_hashes()
    _check_imports()
    if _RUNTIME is None:
        for relative in ("runtime/risk", "runtime/policy"):
            path = str(VENDOR / relative)
            if path not in sys.path:
                sys.path.insert(0, path)
        modules = {name: importlib.import_module(name) for name in _NATIVE_FILES}
        threadpoolctl = importlib.import_module("threadpoolctl")
        _THREAD_LIMITER = threadpoolctl.threadpool_limits(limits=1)
        _RUNTIME = {
            "native": modules["kernel_affine_hull_koopman_machines"],
            "autoencoder": modules["autoencoder"].autoencoder,
            "threadpoolctl": threadpoolctl,
        }
    _RUNTIME["source_hashes"] = verified
    _RUNTIME["loaded_modules"] = _check_imports()
    return _RUNTIME


def _runtime():
    return initialize_runtime() if _RUNTIME is None else _RUNTIME


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(name + " must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(name + " must be >= " + str(minimum))
    return value


def _matrix(value, name="inputs"):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError(name + " must be a nonempty finite sample-by-coordinate matrix")
    return value


def _readonly(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class InputScaler:
    """Actual MinMax01 or population-SD scaling fitted on U, never on labels.

    Every nonconstant MinMax01 training coordinate spans exactly [0,1]. No
    nonconstant range floor is used. Constant training axes always map to zero,
    including held-out values. Held-out inputs are never clipped.
    """
    mode: str
    offset: np.ndarray
    divisor: np.ndarray
    constant_mask: np.ndarray
    training_rows: int

    @classmethod
    def fit(cls, U, mode):
        U = _matrix(U, "unlabeled inputs")
        if mode not in ("minmax01", "standard"):
            raise ValueError("Geometry scaling must be minmax01 or standard")
        lower, upper = U.min(axis=0), U.max(axis=0)
        constant = upper == lower
        if mode == "minmax01":
            offset, divisor = lower, upper - lower
        else:
            offset, divisor = U.mean(axis=0), U.std(axis=0, ddof=0)
        divisor = np.where(constant, 1.0, divisor)
        if (not np.isfinite(offset).all() or not np.isfinite(divisor).all()
                or np.any(divisor <= 0)):
            raise FloatingPointError("Invalid training-only geometry scale")
        return cls(mode, _readonly(offset), _readonly(divisor),
                   _readonly(constant, bool), int(len(U)))

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if (X.ndim not in (2, 3) or min(X.shape) < 1
                or X.shape[-1] != len(self.offset) or not np.isfinite(X).all()):
            raise ValueError("Query shape or finite-value mismatch with fitted geometry scaler")
        result = (X - self.offset) / self.divisor
        result[..., self.constant_mask] = 0.0
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite scaled geometry query")
        return result

    def metadata(self):
        return {"mode": self.mode, "offset": self.offset.tolist(),
                "divisor": self.divisor.tolist(),
                "constant_mask": self.constant_mask.tolist(),
                "training_rows": self.training_rows, "heldout_clipping": False,
                "constant_policy": "zero for training and heldout",
                "nonconstant_floor": None,
                "standard_ddof": 0 if self.mode == "standard" else None}


@dataclass(frozen=True)
class GeometryBank:
    """One fixed scaler/cap/family at its stored retained block layout.

    ``blocks`` contains native in-memory classifier dictionaries. Development
    banks retain audits and fit indices. ``compact`` removes those and removes
    all unused later blocks and native diagnostic-only encoder fields.
    """
    scaler: InputScaler
    blocks: tuple
    block_sizes: tuple
    requested_counts: tuple
    cap: int
    kind: str
    seed: int
    tau: float = 1e-6
    fit_indices: np.ndarray | None = None
    audit: dict | None = None

    @property
    def n_features_in_(self):
        return len(self.scaler.offset)

    @property
    def n_features_out_(self):
        return sum(self.block_sizes)

    @property
    def block_slices_(self):
        ends = np.cumsum((0,) + self.block_sizes)
        return [slice(int(a), int(b)) for a, b in zip(ends[:-1], ends[1:])]

    @property
    def subspace_dim(self):
        return self.cap

    def metadata(self):
        return {"kind": self.kind, "cap": self.cap, "seed": self.seed,
                "ambient_dimension": self.n_features_in_,
                "requested_cluster_counts": list(self.requested_counts),
                "actual_cluster_counts": list(self.block_sizes),
                "coordinate_dimension": self.n_features_out_,
                "block_count": len(self.blocks),
                "maximum_affine_dimension": self.n_features_out_ - len(self.blocks),
                "tau": self.tau, "scaler": self.scaler.metadata(),
                "fit_indices": None if self.fit_indices is None else self.fit_indices.tolist(),
                "audit": copy.deepcopy(self.audit)}


def _encoder_copy(encoder, keep_audit=True):
    result = {key: _readonly(encoder[key]) for key in _ENCODER_ARRAYS}
    result["kerneltype"] = str(encoder["kerneltype"])
    if keep_audit:
        result["dimension_audit"] = copy.deepcopy(encoder["dimension_audit"])
    return result


def _model_copy(model, keep_audit=True, copy_arrays=True):
    classifiers = []
    for group in model["classifier"]:
        group = group if isinstance(group, (list, tuple)) else [group]
        if len(group) != 1:
            raise AssertionError("Expected one native AE per state with Nb=max_references")
        if copy_arrays:
            encoder = _encoder_copy(group[0], keep_audit)
        else:
            encoder = {key: group[0][key] for key in _ENCODER_ARRAYS}
            encoder["kerneltype"] = group[0]["kerneltype"]
            if keep_audit:
                encoder["dimension_audit"] = copy.deepcopy(group[0]["dimension_audit"])
        classifiers.append((encoder,))
    centers = model["cluster_centers"]
    return {"classifier": tuple(classifiers),
            "cluster_centers": _readonly(centers) if copy_arrays else centers,
            "n_clusters": int(model["n_clusters"]), "classifier_dir": None}


def _rank_records(blocks, cap, dimension):
    records = []
    for block_index, model in enumerate(blocks):
        for state_index, group in enumerate(model["classifier"]):
            encoder = group[0]
            audit = copy.deepcopy(encoder["dimension_audit"])
            refs, basis = encoder["y_data"], encoder["PC"]
            if (audit["effective_rank"] != basis.shape[1]
                    or audit["ambient_dimension"] != dimension
                    or audit["local_reference_count"] != refs.shape[1]
                    or audit["requested_cap"] != cap):
                raise AssertionError("Native rank audit does not match stored model")
            audit.update(block_index=block_index, state_index=state_index,
                         reference_sha256=array_digest(refs))
            records.append(audit)
    return records


def _reference_multiset_digest(references):
    pool = np.ascontiguousarray(np.concatenate(references, axis=1).T)
    hashes = sorted(hashlib.sha256(row.tobytes()).hexdigest() for row in pool)
    return hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()


def _refit_blocks(template_blocks, reference_groups, cap):
    fit = _runtime()["autoencoder"]
    result = []
    for template, groups in zip(template_blocks, reference_groups):
        classifiers = []
        for reference in groups:
            # Native autoencoder itself performs the original float32 rounding.
            classifiers.append((_encoder_copy(fit(reference, subspace_dim=cap)),))
        result.append({"classifier": tuple(classifiers),
                       "cluster_centers": template["cluster_centers"],
                       "n_clusters": int(template["n_clusters"]), "classifier_dir": None})
    return tuple(result)


def _config(config, dimension):
    cfg = config.get("geometry", config)
    scalings = tuple(cfg.get("scalings", ("minmax01", "standard")))
    if (not scalings or len(scalings) != len(set(scalings))
            or any(s not in ("minmax01", "standard") for s in scalings)):
        raise ValueError("Invalid geometry scaler roster")
    caps = cfg.get("caps", {"4": [1, 2, 4], "20": [2, 4, 20]})
    if isinstance(caps, dict):
        caps = caps.get(str(dimension), caps.get(dimension))
        if caps is None:
            raise ValueError("No cap roster for ambient dimension")
    requested_caps = tuple(_integer(c, "cap") for c in caps)
    if not requested_caps or len(requested_caps) != len(set(requested_caps)):
        raise ValueError("Caps must be a nonempty distinct list")
    cap_aliases = {str(c): min(c, dimension) for c in requested_caps}
    canonical_caps = tuple(sorted(set(cap_aliases.values())))
    counts = tuple(_integer(c, "cluster count") for c in
                   cfg.get("cluster_counts", (2, 4, 8, 16, 32)))
    if not counts:
        raise ValueError("Cluster layout must be nonempty")
    max_fit = cfg.get("max_fit", cfg.get("max_geometry_inputs", 512))
    max_fit = None if max_fit is None else _integer(max_fit, "max_fit")
    max_references = _integer(cfg.get("max_references", cfg.get("max_train_per_cluster", 24)),
                              "max_references", 2)
    tau = float(cfg.get("tau", 1e-6))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    return {"scalings": scalings, "requested_caps": requested_caps,
            "caps": canonical_caps, "cap_aliases": cap_aliases,
            "cluster_counts": counts, "max_fit": max_fit,
            "max_references": max_references, "tau": tau}


def fit_family(U, config, seed):
    """Fit learned and reference-regrouped native banks using U alone.

    Return ``{'learned': {(scaler, cap): bank}, 'regrouped': {...},
    'metadata': JSON_safe_records}``. Caps in keys are ambient-canonicalized;
    requested aliases are documented in metadata. The seed is an unsigned
    integer; native block seeds use ``seed % 2**31 + 104729*(block+1)``.
    One fixed regrouping per scaler/block is shared across every local cap.
    """
    U = _matrix(U, "unlabeled inputs")
    seed = _integer(seed, "seed", 0)
    cfg = _config(config, U.shape[1])
    runtime = _runtime()
    base_seed = seed % (2 ** 31)
    counts, max_fit = cfg["cluster_counts"], cfg["max_fit"]
    if len(U) < max(counts) or (max_fit is not None and max_fit < max(counts)):
        raise ValueError("Too few geometry fit rows for requested state layout")
    if max_fit is not None and len(U) > max_fit:
        fit_indices = np.sort(np.random.default_rng(base_seed).choice(len(U), max_fit, replace=False))
    else:
        fit_indices = np.arange(len(U), dtype=np.int64)
    fit_indices = _readonly(fit_indices, np.int64)
    maximum_cap = max(cfg["caps"])
    result = {"learned": {}, "regrouped": {}}
    metadata = {"schema": "step6-native-geometry-family-v1", "seed": seed,
                "native_base_seed": base_seed, "unlabeled_shape": list(U.shape),
                "unlabeled_sha256": array_digest(U), "fit_indices": fit_indices.tolist(),
                "fit_indices_sha256": array_digest(fit_indices),
                "config": {key: list(value) if isinstance(value, tuple) else value
                           for key, value in cfg.items()},
                "source_sha256": dict(runtime["source_hashes"]),
                "forecast_labels_used": False, "maximal_partition_fit_count": 0,
                "native_options": {"kmeans_kind": "full", "model_dtype": "float64",
                    "reference_rounding_before_native_pca": "float32",
                    "singleton_strategy": "augment", "singleton_aux_mix": 0.05,
                    "Nb_equals_max_references": True,
                    "lower_caps": "native autoencoder refit on exact maximal-cap references",
                    "regrouped_control": "same retained reference multiset and per-state counts"},
                "native_kmeans_block_fit_count": 0,
                "native_local_ae_fit_count": 0, "scalings": [], "banks": []}
    family_start = time.perf_counter()
    for scale_index, scaling in enumerate(cfg["scalings"]):
        scale_start = time.perf_counter()
        scaler = InputScaler.fit(U, scaling)
        scaled = scaler.transform(U)
        scaler_seconds = time.perf_counter() - scale_start
        train = scaled[fit_indices]
        partition_start = time.perf_counter()
        maximal_blocks, block_times = [], []
        for block_index, count in enumerate(counts):
            block_seed = base_seed + 104729 * (block_index + 1)
            if block_seed >= 2 ** 32:
                raise ValueError("Native block seed exceeds sklearn's uint32 range")
            block_start = time.perf_counter()
            original = runtime["native"].train_kahm_state_regime_abstraction(
                train.T, n_clusters=count, subspace_dim=maximum_cap,
                Nb=cfg["max_references"], random_state=block_seed, verbose=False,
                kmeans_kind="full", max_train_per_cluster=cfg["max_references"],
                model_dtype="float64", save_ae_to_disk=False,
                singleton_strategy="augment", singleton_aux_mix=0.05)
            maximal_blocks.append(_model_copy(original))
            block_times.append(time.perf_counter() - block_start)
        maximal_blocks = tuple(maximal_blocks)
        partition_seconds = time.perf_counter() - partition_start
        block_sizes = tuple(int(block["n_clusters"]) for block in maximal_blocks)
        learned_references = [[group[0]["y_data"] for group in block["classifier"]]
                              for block in maximal_blocks]
        regrouped_references, regroup_templates, regroup_records = [], [], []
        regroup_start = time.perf_counter()
        for block_index, (block, groups) in enumerate(zip(maximal_blocks, learned_references)):
            pool = np.concatenate(groups, axis=1)
            entropy = [seed, 0x52454736, scale_index, block_index]
            permutation = np.random.default_rng(np.random.SeedSequence(entropy)).permutation(pool.shape[1])
            shuffled, start = pool[:, permutation], 0
            new_groups = []
            for group in groups:
                stop = start + group.shape[1]
                new_groups.append(_readonly(shuffled[:, start:stop]))
                start = stop
            if _reference_multiset_digest(groups) != _reference_multiset_digest(new_groups):
                raise AssertionError("Regrouping changed the retained reference multiset")
            regrouped_references.append(new_groups)
            regroup_templates.append({"n_clusters": block["n_clusters"],
                "cluster_centers": _readonly(np.column_stack([g.mean(axis=1) for g in new_groups]))})
            regroup_records.append({"block_index": block_index,
                "seed_entropy": entropy, "permutation": permutation.tolist(),
                "permutation_sha256": array_digest(permutation),
                "reference_counts": [int(g.shape[1]) for g in groups],
                "source_reference_sequence_sha256": array_digest(pool),
                "regrouped_reference_sequence_sha256": array_digest(shuffled),
                "reference_multiset_sha256": _reference_multiset_digest(groups)})
        regroup_seconds = time.perf_counter() - regroup_start
        metadata["maximal_partition_fit_count"] += 1
        metadata["native_kmeans_block_fit_count"] += len(counts)
        metadata["native_local_ae_fit_count"] += sum(block_sizes)
        metadata["scalings"].append({"scaling": scaling, "scaler": scaler.metadata(),
            "unlabeled_min": U.min(axis=0).tolist(), "unlabeled_max": U.max(axis=0).tolist(),
            "scaler_fit_seconds": scaler_seconds,
            "maximal_native_partition_seconds": partition_seconds,
            "maximal_native_block_seconds": block_times,
            "regrouping_preparation_seconds": regroup_seconds,
            "actual_block_sizes": list(block_sizes), "regrouping": regroup_records})
        for cap in cfg["caps"]:
            for kind in ("learned", "regrouped"):
                start = time.perf_counter()
                if kind == "learned" and cap == maximum_cap:
                    blocks = maximal_blocks
                    fit_seconds = partition_seconds
                    reused = True
                else:
                    blocks = _refit_blocks(
                        maximal_blocks if kind == "learned" else regroup_templates,
                        learned_references if kind == "learned" else regrouped_references, cap)
                    fit_seconds = time.perf_counter() - start
                    metadata["native_local_ae_fit_count"] += sum(block_sizes)
                    reused = False
                ranks = _rank_records(blocks, cap, U.shape[1])
                audit = {"local_models": ranks,
                    "reference_counts": [[int(g[0]["y_data"].shape[1]) for g in b["classifier"]]
                                         for b in blocks],
                    "native_fit_seconds": fit_seconds,
                    "reused_maximal_cap_fit": reused,
                    "conditioning": "learned reference selection" if kind == "regrouped" else None}
                bank = GeometryBank(scaler, blocks, block_sizes, counts, cap, kind, seed,
                                    cfg["tau"], fit_indices, audit)
                result[kind][(scaling, cap)] = bank
                metadata["banks"].append({"scaling": scaling, "cap": cap, "kind": kind,
                                          **bank.metadata()})
    metadata["total_fit_seconds"] = time.perf_counter() - family_start
    metadata["timing_note"] = ("Total includes scaling, native partitions, cap refits, regrouping and audits; "
        "the maximal cap's native_fit_seconds reuses its partition time and is not additive twice.")
    result["metadata"] = metadata
    return result


def score(bank, X):
    """Evaluate pinned folding scores once per state, preserving 2D/3D row axes."""
    if not isinstance(bank, GeometryBank):
        raise TypeError("score requires a GeometryBank")
    scaled = bank.scaler.transform(X)
    shape = scaled.shape[:-1]
    flat = scaled.reshape(-1, scaled.shape[-1])
    native = _runtime()["native"]
    values = np.concatenate([native.kahm_folding_distance_matrix(
        block, flat.T, n_jobs=1, batch_size=1000, show_progress=False).T
        for block in bank.blocks], axis=1)
    if (values.shape != (len(flat), bank.n_features_out_)
            or not np.isfinite(values).all()
            or np.any(values < -1e-12) or np.any(values > 1 + 1e-12)):
        raise FloatingPointError("Native folding scores violate finite/range/shape contract")
    return values.reshape(shape + (bank.n_features_out_,))


def associations(scores, block_sizes, omega, prefix=None, tau=1e-6):
    """Normalize cached scores per block; prefix is the number of retained blocks."""
    scores = np.asarray(scores, dtype=np.float64)
    sizes = tuple(_integer(n, "block size") for n in block_sizes)
    prefix = len(sizes) if prefix is None else _integer(prefix, "prefix")
    if (scores.ndim not in (2, 3) or min(scores.shape) < 1 or not sizes
            or scores.shape[-1] != sum(sizes) or prefix > len(sizes)
            or not np.isfinite(scores).all()):
        raise ValueError("Cached folding-score layout does not match block sizes")
    omega, tau = float(omega), float(tau)
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(tau) or tau <= 0:
        raise ValueError("omega and tau must be finite and positive")
    pieces, start = [], 0
    for size in sizes[:prefix]:
        stop = start + size
        logits = omega * np.log(1.0 - np.clip(scores[..., start:stop], 0, 1) + tau)
        pieces.append(np.exp(logits - logsumexp(logits, axis=-1, keepdims=True)))
        start = stop
    result = np.concatenate(pieces, axis=-1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite native associations")
    return result


def compact(bank, prefix):
    """Retain only selected blocks and query-required native fields, without fits."""
    if not isinstance(bank, GeometryBank):
        raise TypeError("compact requires a GeometryBank")
    prefix = _integer(prefix, "prefix")
    if prefix > len(bank.blocks):
        raise ValueError("Prefix exceeds the fitted layout")
    blocks = tuple(_model_copy(block, keep_audit=False, copy_arrays=False)
                   for block in bank.blocks[:prefix])
    return GeometryBank(bank.scaler, blocks, bank.block_sizes[:prefix],
                        bank.requested_counts[:prefix], bank.cap, bank.kind,
                        bank.seed, bank.tau, None, None)


@dataclass(frozen=True)
class GeometryMap:
    """Configured compact geometry map; normal query calls perform no file I/O."""
    bank: GeometryBank
    omega: float

    def __post_init__(self):
        if not isinstance(self.bank, GeometryBank):
            raise TypeError("GeometryMap requires a GeometryBank")
        if not np.isfinite(self.omega) or self.omega <= 0:
            raise ValueError("omega must be finite and positive")

    @property
    def n_features_in_(self):
        return self.bank.n_features_in_

    @property
    def n_features_out_(self):
        return self.bank.n_features_out_

    def transform(self, X):
        return associations(score(self.bank, X), self.bank.block_sizes,
                            self.omega, tau=self.bank.tau)


def geometry_payload(bank):
    """Serialize a local geometry bank/map with protocol5; no training arrays.

    Use ``compact`` first for deployed storage. Development banks intentionally
    contain their rank/provenance audits and are not lean deployment payloads.
    """
    if not isinstance(bank, (GeometryBank, GeometryMap)):
        raise TypeError("Expected a local GeometryBank or GeometryMap")
    return pickle.dumps(bank, protocol=5)
