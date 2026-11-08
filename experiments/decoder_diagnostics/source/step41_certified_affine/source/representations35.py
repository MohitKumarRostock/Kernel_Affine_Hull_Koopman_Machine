"""Frozen Step35 representation factory; callers supply training inputs only.

The 18-map roster is fixed here. There is deliberately no target argument and
no selection based on calibration or test inputs. Input rows must already be
chronologically ordered. The factory is not a canonical-real-arithmetic
certificate: its outputs are the operational binary64 representation values.

Native sources are copied byte-for-byte under inputs/native. Only the retained
source files are imported; historical scientific arrays are never consulted.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib.util
import json
import numbers
import os
from pathlib import Path
import pickle
import sys
import time

for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT"):
    os.environ[_thread_name] = "1"
sys.dont_write_bytecode = True

import numpy as np
from scipy.spatial.distance import cdist, pdist


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = (ROOT / "inputs/native/step6_geometry_regimes/src/geometry.py")
GEOMETRY_SHA256 = "7c664cf4e85f34341753d48f47ae790c5c71aa33af120b5276e20a063355beeb"
_GEOMETRY_MODULE = "_step35_pinned_geometry"
_GEOMETRY = None

DEFAULT_CONFIG = {
    "dimension": 24,
    "scalings": ["minmax01", "standard"],
    "nystrom_widths": [8, 24],
    "kahm_caps": [4, 12, 20],
    "kahm_prefixes": [1, 2],
    "cluster_counts": [8, 16],
    "max_fit": 512,
    "max_references": 24,
    "tau": 1e-6,
    "omega": 2.0,
    "probe_count": 32,
    "nystrom_eigen_floor": 1e-12,
    "temporal_selection": "equal_row_stratum_midpoints",
    "metric_normalization": "reciprocal_median_positive_pair_distance",
}


def _geometry():
    """Load the pinned wrapper under a stable, pickle-compatible module name."""
    global _GEOMETRY
    if _GEOMETRY is None:
        digest = hashlib.sha256(GEOMETRY_PATH.read_bytes()).hexdigest()
        if digest != GEOMETRY_SHA256:
            raise RuntimeError("Step35 pinned geometry wrapper digest mismatch")
        existing = sys.modules.get(_GEOMETRY_MODULE)
        if existing is not None:
            if Path(existing.__file__).resolve() != GEOMETRY_PATH.resolve():
                raise ImportError("Foreign Step35 geometry module; use an isolated worker")
            _GEOMETRY = existing
        else:
            spec = importlib.util.spec_from_file_location(_GEOMETRY_MODULE, GEOMETRY_PATH)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_GEOMETRY_MODULE] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(_GEOMETRY_MODULE, None)
                raise
            _GEOMETRY = module
    return _GEOMETRY


def initialize_runtime():
    """Verify native source hashes and initialize one-thread native execution."""
    return _geometry().initialize_runtime()


def _matrix(X, name):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError(name + " must be a finite, nonempty rows-by-coordinate matrix")
    return X


def _readonly(X):
    result = np.array(X, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _config(config):
    supplied = {} if config is None else config
    if not isinstance(supplied, dict):
        raise TypeError("config must be a dictionary or None")
    if "representations" in supplied:
        supplied = supplied["representations"]
        if not isinstance(supplied, dict):
            raise TypeError("config['representations'] must be a dictionary")
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("Unknown representation settings: " + ", ".join(sorted(unknown)))
    effective = copy.deepcopy(DEFAULT_CONFIG)
    effective.update(copy.deepcopy(supplied))
    if effective != DEFAULT_CONFIG:
        raise ValueError("Step35's 18-map roster/settings are fixed; configuration changed")
    return effective


def temporal_indices(n_rows, count):
    """One deterministic midpoint per equal chronological row stratum.

    For j=0,...,count-1 the index is floor((2*j+1)*n_rows/(2*count)).
    Landmark rosters for widths 8 and 24 are constructed separately.
    """
    if (isinstance(n_rows, bool) or isinstance(count, bool)
            or not isinstance(n_rows, numbers.Integral)
            or not isinstance(count, numbers.Integral)
            or count < 1 or n_rows < count):
        raise ValueError("Temporal selection needs integer n_rows >= count >= 1")
    result = ((2 * np.arange(count, dtype=np.int64) + 1) * int(n_rows)) // (2 * int(count))
    if len(np.unique(result)) != count:
        raise AssertionError("Temporal selection produced repeated indices")
    return result


def _positive_distance_median(X, squared=False):
    distances = pdist(X, metric="sqeuclidean" if squared else "euclidean")
    if not np.isfinite(distances).all():
        raise FloatingPointError("Nonfinite training-probe distances")
    positive = distances[distances > 0]
    if not len(positive):
        return 1.0, True, 0
    median = float(np.median(positive))
    if not np.isfinite(median) or median <= 0:
        raise FloatingPointError("Invalid positive training-probe distance median")
    return median, False, int(len(positive))


@dataclass(frozen=True)
class NystromCore35:
    scaler: object
    landmarks: np.ndarray
    projection: np.ndarray
    gamma: np.ndarray

    def transform(self, X):
        scaled = self.scaler.transform(X)
        squared = cdist(scaled, self.landmarks, metric="sqeuclidean")
        result = np.exp(-float(self.gamma) * squared) @ self.projection
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite operational Nystrom features")
        return result


@dataclass(frozen=True)
class Representation35:
    name: str
    family: str
    scaling: str
    requested_width: int
    core: object
    metric_scale: np.ndarray

    @property
    def n_features_in_(self):
        if self.family == "raw":
            return len(self.core.offset)
        if self.family == "nystrom":
            return len(self.core.scaler.offset)
        if self.family == "kahm":
            return self.core.n_features_in_
        raise ValueError("Unknown representation family")

    @property
    def n_features_out_(self):
        if self.family == "raw":
            return self.n_features_in_
        if self.family == "nystrom":
            return self.core.projection.shape[1]
        if self.family == "kahm":
            return self.core.n_features_out_
        raise ValueError("Unknown representation family")


def _unscaled_transform(representation, X):
    if not isinstance(representation, Representation35):
        raise TypeError("Expected a Representation35 map")
    X = _matrix(X, "query inputs")
    if X.shape[1] != representation.n_features_in_:
        raise ValueError("Query dimension does not match the frozen representation")
    result = representation.core.transform(X)
    if (result.shape != (len(X), representation.n_features_out_)
            or not np.isfinite(result).all()):
        raise FloatingPointError("Representation violated its finite/shape contract")
    return result


def transform_one(representation, X):
    """Query a frozen map, including its frozen training-only metric scalar."""
    result = _unscaled_transform(representation, X) * float(representation.metric_scale)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite metric-normalized representation")
    return result


def serialize_map(representation):
    """Trusted local pickle payload; contains all deployed transform parameters."""
    if not isinstance(representation, Representation35):
        raise TypeError("Expected a Representation35 map")
    return pickle.dumps(representation, protocol=5)


def deserialize_map(payload):
    """Reload a trusted local Step35 payload after registering pinned classes."""
    _geometry()
    result = pickle.loads(payload)
    if not isinstance(result, Representation35):
        raise TypeError("Payload does not contain a Step35 representation")
    return result


def payload_bytes(representation):
    """Serialized payload size, including metadata and Python pickle overhead."""
    return len(serialize_map(representation))


def numeric_bytes(representation):
    """Stored numeric bytes, including the encoder and feature metric scalar.

    Count each retained ndarray object once by identity, at its stored dtype;
    Python float/int scalars use 8 bytes, bool uses 1, complex uses 16, and NumPy
    scalars use their dtype width. Shared container objects are visited once.
    Strings, object headers and code are excluded here but pickle overhead is
    included in payload_bytes. This counts one independently deployed map;
    summing maps describes independent deployments, not an optimized shared bank.
    """
    if not isinstance(representation, Representation35):
        raise TypeError("Expected a Representation35 map")
    seen = set()

    def count(value):
        if value is None or isinstance(value, (str, bytes, Path)):
            return 0
        if isinstance(value, np.generic):
            return int(value.dtype.itemsize)
        if isinstance(value, bool):
            return 1
        if isinstance(value, numbers.Integral):
            return 8
        if isinstance(value, numbers.Real):
            return 8
        if isinstance(value, numbers.Complex):
            return 16
        ident = id(value)
        if ident in seen:
            return 0
        seen.add(ident)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object arrays are not deployable numeric buffers")
            return int(value.nbytes)
        if is_dataclass(value):
            return sum(count(getattr(value, field.name)) for field in fields(value))
        if isinstance(value, dict):
            return sum(count(key) + count(item) for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return sum(count(item) for item in value)
        raise TypeError("Unaccounted deployed object: " + type(value).__name__)

    return int(count(representation))


def _freeze_metric(name, family, scaling, width, core, probes):
    temporary = Representation35(name, family, scaling, width, core, _readonly(1.0))
    unscaled = _unscaled_transform(temporary, probes)
    median, fallback, positive_count = _positive_distance_median(unscaled)
    scalar = 1.0 if fallback else 1.0 / median
    if not np.isfinite(scalar) or scalar <= 0:
        raise FloatingPointError("Invalid frozen feature-metric scalar")
    result = Representation35(name, family, scaling, width, core, _readonly(scalar))
    return result, {
        "metric_scale": scalar,
        "unscaled_probe_distance_median": median,
        "metric_scale_fallback_one": fallback,
        "positive_probe_pair_count": positive_count,
        "metric_scaling_is_training_only": True,
        "metric_rule": DEFAULT_CONFIG["metric_normalization"],
    }


def fit_representations(Xtrain, config, seed):
    """Fit the complete ordered 18-map roster and return (maps, metadata).

    Xtrain is the chronological training feature matrix; labels are never an
    argument. Every scaler, landmark, PCA/encoder, bandwidth and metric scalar
    is fixed from these inputs. The caller must seal the scientific protocol
    before supplying actual public data. Artificial preflights are separate.

    Failures propagate: the factory never omits or refits a failed setting.
    Native actual state counts/ranks are logged even if requested counts differ.
    """
    started = time.perf_counter()
    cfg = _config(config)
    Xtrain = _matrix(Xtrain, "training inputs")
    if Xtrain.shape[1] != cfg["dimension"] or len(Xtrain) < cfg["probe_count"]:
        raise ValueError("Step35 needs at least 32 training rows and exactly 24 input coordinates")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    runtime_started = time.perf_counter()
    native_runtime = initialize_runtime()
    runtime_seconds = time.perf_counter() - runtime_started
    geometry = _geometry()
    probe_indices = temporal_indices(len(Xtrain), cfg["probe_count"])
    probes = Xtrain[probe_indices]
    maps = OrderedDict()
    metadata = {
        "schema": "step35-frozen-representations-v1",
        "input_role": "caller_supplied_chronological_training_features_only",
        "forecast_labels_used": False,
        "config": cfg,
        "seed": seed,
        "training_shape": list(Xtrain.shape),
        "training_sha256": geometry.array_digest(Xtrain),
        "probe_indices": probe_indices.tolist(),
        "probe_sha256": geometry.array_digest(probes),
        "probe_rule": "floor((2*j+1)*n_training_rows/(2*32)), j=0,...,31",
        "factory_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(
            cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "geometry_wrapper_sha256": GEOMETRY_SHA256,
        "native_source_sha256": dict(native_runtime["source_hashes"]),
        "runtime_initialization_seconds": runtime_seconds,
        "scalers": {},
        "nystrom_bandwidths": {},
        "maps": [],
        "no_heldout_input_or_label_used": True,
        "numeric_byte_convention": numeric_bytes.__doc__.strip(),
        "timing_scope": (
            "factory_total_seconds is one physical wall-clock total including verification, "
            "all fits, training-probe queries, compaction and metadata/payload accounting. "
            "native_fit_family_wall_seconds includes both learned and unused regrouped fits. "
            "Per-map construction seconds exclude shared scaler/native-family fitting. "
            "Native per-bank fit times are diagnostic and must not be summed as disjoint costs."
        ),
    }

    def add_map(representation, extra, construction_started):
        if representation.name in maps:
            raise AssertionError("Repeated fixed representation name")
        maps[representation.name] = representation
        payload = serialize_map(representation)
        metadata["maps"].append({
            "name": representation.name,
            "family": representation.family,
            "scaling": representation.scaling,
            "requested_width": int(representation.requested_width),
            "actual_width": int(representation.n_features_out_),
            "numeric_bytes": numeric_bytes(representation),
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "construction_seconds_excluding_shared_fits": time.perf_counter() - construction_started,
            **extra,
        })

    scalers = {}
    for scaling in cfg["scalings"]:
        scaler_started = time.perf_counter()
        scaler = geometry.InputScaler.fit(Xtrain, scaling)
        scaler_seconds = time.perf_counter() - scaler_started
        scalers[scaling] = scaler
        metadata["scalers"][scaling] = {**scaler.metadata(), "fit_seconds": scaler_seconds}
        construction_started = time.perf_counter()
        representation, metric = _freeze_metric(
            "raw_" + scaling, "raw", scaling, cfg["dimension"], scaler, probes)
        add_map(representation, metric, construction_started)

    for scaling in cfg["scalings"]:
        scaler = scalers[scaling]
        bandwidth_started = time.perf_counter()
        scaled_probes = scaler.transform(probes)
        squared_median, fallback, positive_count = _positive_distance_median(scaled_probes, squared=True)
        gamma = 1.0 if fallback else 1.0 / squared_median
        if not np.isfinite(gamma) or gamma <= 0:
            raise FloatingPointError("Invalid frozen Nystrom bandwidth")
        metadata["nystrom_bandwidths"][scaling] = {
            "gamma": gamma, "positive_squared_probe_distance_median": squared_median,
            "fallback_one": fallback, "positive_probe_pair_count": positive_count,
            "construction_seconds": time.perf_counter() - bandwidth_started,
        }
        for width in cfg["nystrom_widths"]:
            construction_started = time.perf_counter()
            indices = temporal_indices(len(Xtrain), width)
            landmarks = scaler.transform(Xtrain[indices])
            gram = np.exp(-gamma * cdist(landmarks, landmarks, metric="sqeuclidean"))
            gram = (gram + gram.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            clipped = np.maximum(eigenvalues, cfg["nystrom_eigen_floor"])
            projection = eigenvectors / np.sqrt(clipped)[None, :]
            core = NystromCore35(scaler, _readonly(landmarks), _readonly(projection), _readonly(gamma))
            name = "nystrom_" + scaling + "_w" + str(width)
            representation, metric = _freeze_metric(name, "nystrom", scaling, width, core, probes)
            add_map(representation, {
                **metric, "gamma": gamma, "landmark_indices": indices.tolist(),
                "landmark_sha256": geometry.array_digest(landmarks),
                "eigen_floor": cfg["nystrom_eigen_floor"],
                "eigenvalues_below_floor": int(np.sum(eigenvalues < cfg["nystrom_eigen_floor"])),
                "gram_eigenvalue_minimum": float(eigenvalues.min()),
                "gram_eigenvalue_maximum": float(eigenvalues.max()),
            }, construction_started)

    native_config = {
        "scalings": cfg["scalings"], "caps": cfg["kahm_caps"],
        "cluster_counts": cfg["cluster_counts"], "max_fit": cfg["max_fit"],
        "max_references": cfg["max_references"], "tau": cfg["tau"],
    }
    native_started = time.perf_counter()
    family = geometry.fit_family(Xtrain, native_config, seed)
    metadata["native_fit_family_wall_seconds"] = time.perf_counter() - native_started
    metadata["native_fit_family"] = family["metadata"]
    metadata["native_cost_accounting"] = {
        "learned_banks_fitted": len(family["learned"]),
        "regrouped_banks_fitted": len(family["regrouped"]),
        "regrouped_banks_used_for_deployment": False,
        "unused_regrouped_fitting_included_in_physical_factory_cost": True,
        "native_kmeans_block_fit_count": family["metadata"]["native_kmeans_block_fit_count"],
        "native_local_ae_fit_count": family["metadata"]["native_local_ae_fit_count"],
    }
    for scaling in cfg["scalings"]:
        for cap in cfg["kahm_caps"]:
            bank = family["learned"][(scaling, cap)]
            for prefix in cfg["kahm_prefixes"]:
                construction_started = time.perf_counter()
                compact_bank = geometry.compact(bank, prefix)
                core = geometry.GeometryMap(compact_bank, cfg["omega"])
                width = sum(cfg["cluster_counts"][:prefix])
                name = "kahm_" + scaling + "_cap" + str(cap) + "_p" + str(prefix)
                representation, metric = _freeze_metric(name, "kahm", scaling, width, core, probes)
                add_map(representation, {
                    **metric, "cap": cap, "prefix": prefix, "omega": cfg["omega"],
                    "tau": cfg["tau"],
                    "requested_block_sizes": cfg["cluster_counts"][:prefix],
                    "actual_block_sizes": list(compact_bank.block_sizes),
                    "all_local_models": copy.deepcopy([
                        record for record in bank.audit["local_models"]
                        if record["block_index"] < prefix]),
                    "native_fit_cost_is_shared_family_cost": True,
                }, construction_started)
    if len(maps) != 18:
        raise AssertionError("The fixed Step35 roster must contain exactly 18 maps")
    metadata["map_order"] = list(maps)
    metadata["factory_total_seconds"] = time.perf_counter() - started
    metadata["scientific_scope"] = (
        "Operational floating-point features. This factory does not supply a canonical "
        "real-arithmetic KAHM enclosure or a population forecast-risk guarantee."
    )
    return maps, metadata
