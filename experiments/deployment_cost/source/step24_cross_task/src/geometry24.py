"""Pinned input-only KAHM geometry and operational Step 24 representations.

Importing this file does not read benchmark data or fit a model. ``fit_banks``
accepts only a supplied 512-row reference feature matrix; the caller owns the
development-access gate. All exported query payloads contain plain Python
containers, strings, NumPy arrays/scalars and no fitted estimator or file path.

The operational score path is inherited unchanged, including 1,000-row native
batching. This module makes no canonical-real numerical-enclosure claim.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "OMP_THREAD_LIMIT"):
    os.environ[_name] = "1"

import numpy as np
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
STEP23 = ROOT.parent / "step23_cross_task"
GEOMETRY_SOURCE = STEP23 / "inherited_source/step6_geometry_regimes/src/geometry.py"
EXPECTED_PROTOCOL_SHA256 = "e7109b924bc0899aa0d97c9091828c29068bd0f2b67714e754e59d594df588ef"
BLOCK_SIZES = (2, 4, 8, 16)
PREFIXES = (2, 6, 14, 30)
SCALINGS = ("minmax01", "standard")
CAPS = (2, 8, 20)
TAU = 1e-6
NATIVE_BATCH_SIZE = 1000
ENCODER_ARRAYS = ("y_data_n_subspace", "PC", "B", "y_data", "sqrt_weights_matrix")
_GEOMETRY = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_provenance() -> dict[str, str]:
    """Verify only scientific source/specification bytes, never a data archive."""
    if _sha(STEP23 / "PROTOCOL.json") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("The sealed Step 23 protocol changed")
    manifest = json.loads((STEP23 / "protocol/inherited_source_manifest23.json").read_text())
    result = {"src/geometry24.py": _sha(Path(__file__).resolve()),
              "../step23_cross_task/PROTOCOL.json": EXPECTED_PROTOCOL_SHA256,
              "../step23_cross_task/protocol/inherited_source_manifest23.json":
                  _sha(STEP23 / "protocol/inherited_source_manifest23.json")}
    for record in manifest["files"]:
        path = STEP23 / record["package_path"]
        actual = _sha(path)
        if actual != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise RuntimeError("Pinned inherited geometry source changed: " + record["package_path"])
        result["../step23_cross_task/" + record["package_path"]] = actual
    return result


def initialize_runtime():
    """Load the verified inherited source once and limit native libraries to one thread."""
    global _GEOMETRY
    if _GEOMETRY is None:
        source_provenance()
        name = "step24_pinned_geometry_source"
        spec = importlib.util.spec_from_file_location(name, GEOMETRY_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.initialize_runtime()
        _GEOMETRY = module
    return _GEOMETRY


def bank_key(scaling: str, cap: int) -> str:
    if scaling not in SCALINGS or isinstance(cap, (bool, np.bool_)) or int(cap) not in CAPS or int(cap) != cap:
        raise ValueError("Unknown fixed geometry scaling/cap")
    return f"{scaling}_{int(cap)}"


def blocks_for_prefix(prefix: int) -> tuple[int, ...]:
    if isinstance(prefix, (bool, np.bool_)) or not isinstance(prefix, Integral) or int(prefix) not in PREFIXES:
        raise ValueError("State prefix must be one of 2, 6, 14, 30")
    return BLOCK_SIZES[:PREFIXES.index(int(prefix)) + 1]


def _matrix(value, name="feature matrix") -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError(name + " must be a nonempty finite sample-by-coordinate matrix")
    return np.ascontiguousarray(x)


def _readonly(value, dtype=None):
    a = np.asarray(value, dtype=dtype)
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def _export_native_bank(bank) -> dict:
    """Drop fit audits/indices, while retaining all native query arrays and centers."""
    blocks = []
    for block in bank.blocks:
        classifiers = []
        for group in block["classifier"]:
            if len(group) != 1:
                raise ValueError("Expected exactly one native autoencoder per state")
            ae = group[0]
            arrays = {key: _readonly(ae[key]) for key in ENCODER_ARRAYS}
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ArithmeticError("Nonfinite native encoder parameter")
            arrays["kerneltype"] = str(ae["kerneltype"])
            classifiers.append((arrays,))
        blocks.append({"classifier": tuple(classifiers),
                       "cluster_centers": _readonly(block["cluster_centers"]),
                       "n_clusters": np.int64(block["n_clusters"]),
                       "classifier_dir": None})
    return {"kind": "kahm_encoder",
            "scaler": {"offset": _readonly(bank.scaler.offset),
                       "divisor": _readonly(bank.scaler.divisor),
                       "constant_mask": _readonly(bank.scaler.constant_mask)},
            "blocks": tuple(blocks),
            "block_sizes": _readonly(bank.block_sizes, np.int64),
            "score_batch_size": np.int64(NATIVE_BATCH_SIZE)}


def fit_banks(U, seed: int):
    """Fit the six frozen learned banks from reference features only.

    Returns (banks, evidence), with string keys such as ``minmax01_2``. Banks
    already contain pure deployable 30-coordinate encoder payloads. Evidence
    retains requested/realized ranks and the inherited unused-regrouping costs.
    This function never opens an input file or accepts a forecast target.
    """
    U = _matrix(U, "reference features")
    if U.shape[0] != 512:
        raise ValueError("The fixed study requires exactly 512 reference feature rows")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("Geometry seed must be a nonnegative integer")
    G = initialize_runtime()
    start = time.perf_counter()
    config = {"scalings": list(SCALINGS), "caps": list(CAPS),
              "cluster_counts": list(BLOCK_SIZES), "max_fit": 512,
              "max_references": 24, "tau": TAU}
    native = G.fit_family(U, config, int(seed))
    banks = {}
    for scaling in SCALINGS:
        for cap in CAPS:
            # Protocol tasks have ambient dimension >= 20; this also preserves
            # declared cap aliases for manufactured lower-dimensional inputs.
            native_bank = native["learned"][(scaling, min(cap, U.shape[1]))]
            if tuple(native_bank.block_sizes) != BLOCK_SIZES:
                raise RuntimeError("Actual native blocks differ from the frozen 2/4/8/16 roster")
            banks[bank_key(scaling, cap)] = _export_native_bank(native_bank)
    metadata = native["metadata"]
    unused = [b for b in metadata["banks"] if b["kind"] == "regrouped"]
    learned = [b for b in metadata["banks"] if b["kind"] == "learned"]
    evidence = {"schema": "step24-native-geometry-fit-v1",
                "reference_rows": int(len(U)), "ambient_dimension": int(U.shape[1]),
                "reference_feature_sha256": G.array_digest(U),
                "forecast_targets_used": False,
                "learned_bank_keys": list(banks),
                "native_metadata": metadata,
                "native_geometry_wall_seconds": time.perf_counter() - start,
                "unused_regrouped_bank_count": len(unused),
                "unused_regrouped_local_autoencoder_fits": sum(sum(b["actual_cluster_counts"]) for b in unused),
                "unused_regrouped_refit_seconds": sum(b["audit"]["native_fit_seconds"] for b in unused),
                "unused_regrouping_preparation_seconds": sum(s["regrouping_preparation_seconds"] for s in metadata["scalings"]),
                "learned_local_autoencoder_fits": sum(sum(b["actual_cluster_counts"]) for b in learned),
                "native_total_local_autoencoder_fits": metadata["native_local_ae_fit_count"],
                "unused_regrouped_controls_evaluated": False,
                "source_sha256": source_provenance()}
    # Drop the unused fitted objects here. Only their explicit fit evidence is
    # retained; they cannot become candidates or a free encoder in deployment.
    return banks, evidence


def compact_bank(bank: dict, prefix: int) -> dict:
    """Select a complete initial block prefix without refitting or copying data."""
    widths = blocks_for_prefix(prefix)
    if bank.get("kind") != "kahm_encoder":
        raise ValueError("Expected a pure KAHM encoder payload")
    actual = tuple(map(int, bank["block_sizes"]))
    if actual[:len(widths)] != widths or len(actual) < len(widths):
        raise ValueError("Requested prefix exceeds or mismatches the retained bank")
    if len(bank["blocks"]) != len(actual):
        raise ValueError("Encoder block metadata mismatch")
    return {"kind": "kahm_encoder", "scaler": bank["scaler"],
            "blocks": tuple(bank["blocks"][:len(widths)]),
            "block_sizes": _readonly(widths, np.int64),
            "score_batch_size": np.int64(bank["score_batch_size"])}


def scale_input(scaler: dict, X) -> np.ndarray:
    X = _matrix(X)
    offset = np.asarray(scaler["offset"])
    divisor = np.asarray(scaler["divisor"])
    constant = np.asarray(scaler["constant_mask"], dtype=bool)
    if (offset.shape != (X.shape[1],) or divisor.shape != offset.shape
            or constant.shape != offset.shape or np.any(divisor <= 0)):
        raise ValueError("Input scaler dimension/divisor mismatch")
    result = (X - offset) / divisor
    result[:, constant] = 0.0
    if not np.isfinite(result).all():
        raise ArithmeticError("Nonfinite scaled reference/query features")
    return np.ascontiguousarray(result)


def score_compact(payload: dict, X) -> np.ndarray:
    """Compute native folding scores from raw rows, including input scaling."""
    if payload.get("kind") != "kahm_encoder":
        raise ValueError("Expected a pure KAHM encoder payload")
    widths = tuple(map(int, payload["block_sizes"]))
    if widths not in tuple(BLOCK_SIZES[:i] for i in range(1, 5)):
        raise ValueError("Invalid retained block roster")
    if len(payload["blocks"]) != len(widths) or int(payload["score_batch_size"]) != NATIVE_BATCH_SIZE:
        raise ValueError("Native block layout or fixed batching changed")
    scaled = scale_input(payload["scaler"], X)
    G = initialize_runtime()
    native = G._runtime()["native"]
    result = np.concatenate([
        native.kahm_folding_distance_matrix(block, scaled.T, n_jobs=1,
            batch_size=int(payload["score_batch_size"]), show_progress=False).T
        for block in payload["blocks"]], axis=1)
    if (result.shape != (len(scaled), sum(widths)) or not np.isfinite(result).all()
            or np.any(result < -1e-12) or np.any(result > 1 + 1e-12)):
        raise ArithmeticError("Native folding score violated its shape/finite/range contract")
    return np.ascontiguousarray(result)


def score_bank(bank: dict, X, prefix: int = 30) -> np.ndarray:
    return score_compact(compact_bank(bank, prefix), X)


def _widths(block_sizes) -> tuple[int, ...]:
    result = tuple(block_sizes)
    if (not result or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                          or int(v) not in BLOCK_SIZES for v in result)):
        raise ValueError("Representations require block widths in {2,4,8,16}")
    return tuple(map(int, result))


def walsh_contrasts(width: int) -> np.ndarray:
    width = _widths((width,))[0]
    h = np.ones((1, 1), dtype=np.int8)
    while len(h) < width:
        h = np.block([[h, h], [h, -h]]).astype(np.int8, copy=False)
    # An h[1:] view retains the unused constant row in memory, while NumPy
    # pickling would serialize only the visible rows. Own the exact retained
    # array so deployment storage is invariant under save/load.
    return _readonly(np.array(h[1:], dtype=np.int8, order="C", copy=True))


def _positive_scalar(value, name, lower=0.0, upper=math.inf) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= lower or result > upper:
        raise ValueError(name + " lies outside its permitted range")
    return result


def view_payload(view: str, block_sizes, omega=None, tau: float = TAU) -> dict:
    """Return all retained numerical parameters of one representation.

    CS omits unused tau/omega; LR0 omits unused omega. Walsh arrays are retained
    explicitly and therefore included by a recursive numeric inventory.
    """
    if view not in {"P", "CS", "LR0", "LRM", "LW"}:
        raise ValueError("Unknown frozen representation")
    widths = _widths(block_sizes)
    result = {"kind": "kahm_view", "view": view,
              "block_sizes": _readonly(widths, np.int64)}
    if view != "CS":
        result["tau"] = np.float64(_positive_scalar(tau, "tau", 0.0, 1.0))
    if view in {"P", "LRM", "LW"}:
        result["omega"] = np.float64(_positive_scalar(omega, "omega", 0.0, 128.0))
    elif omega is not None:
        _positive_scalar(omega, "omega", 0.0, 128.0)
    if view in {"LR0", "LRM"}:
        result["walsh"] = tuple(walsh_contrasts(j) for j in widths)
    return result


def _direct_contrasts(v: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Fixed sign-separated binary64 summation; uniform blocks give exact zero."""
    width = v.shape[-1]
    cols = []
    for signs in H:
        positive = np.sum(v[..., signs > 0], axis=-1, dtype=np.float64)
        negative = np.sum(v[..., signs < 0], axis=-1, dtype=np.float64)
        cols.append((positive - negative) / width)
    return np.stack(cols, axis=-1)


def transform_scores(scores, payload: dict) -> np.ndarray:
    """Compute the declared operational view without a probability floor."""
    scores = np.asarray(scores, dtype=np.float64)
    if payload.get("kind") != "kahm_view":
        raise ValueError("Expected a representation payload")
    widths = _widths(payload["block_sizes"])
    if (scores.ndim not in (2, 3) or min(scores.shape) < 1
            or scores.shape[-1] != sum(widths) or not np.isfinite(scores).all()):
        raise ValueError("Finite score shape does not match representation blocks")
    view = payload["view"]
    if view not in {"P", "CS", "LR0", "LRM", "LW"}:
        raise ValueError("Unknown representation")
    c = np.clip(scores, 0.0, 1.0)
    if view == "CS":
        return np.ascontiguousarray(c)
    tau = _positive_scalar(payload["tau"], "tau", 0.0, 1.0)
    omega = (None if view == "LR0" else
             _positive_scalar(payload["omega"], "omega", 0.0, 128.0))
    parts = []
    start = 0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for index, width in enumerate(widths):
            block = c[..., start:start + width]
            v = np.log(1.0 - block + tau)
            if view == "P":
                eta = omega * v
                piece = np.exp(eta - logsumexp(eta, axis=-1, keepdims=True))
            elif view == "LW":
                piece = omega * v
            else:
                H = np.asarray(payload["walsh"][index])
                if H.shape != (width - 1, width):
                    raise ValueError("Walsh payload shape changed")
                r = _direct_contrasts(v, H)
                mass = (np.zeros(v.shape[:-1] + (1,), dtype=np.float64) if view == "LR0"
                        else logsumexp(omega * v, axis=-1, keepdims=True))
                piece = np.concatenate((r, mass), axis=-1)
            parts.append(piece)
            start += width
    result = np.ascontiguousarray(np.concatenate(parts, axis=-1))
    if result.shape != scores.shape or not np.isfinite(result).all():
        raise ArithmeticError("Representation produced nonfinite or misplaced coordinates")
    return result


def representation_diagnostics(scores, payload: dict, features=None) -> dict:
    """Descriptive rounding/clipping measurements; no target sufficiency claim."""
    scores = np.asarray(scores, dtype=np.float64)
    actual = transform_scores(scores, payload)
    if features is not None:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != actual.shape or not np.array_equal(features, actual):
            raise ArithmeticError("Cached representation differs from the fixed operational transform")
    F = actual
    c = np.clip(scores, 0.0, 1.0)
    widths = _widths(payload["block_sizes"])
    view = payload["view"]
    out = {"view": view, "rows": int(np.prod(scores.shape[:-1])),
           "dimension": int(scores.shape[-1]), "block_sizes": list(widths),
           "score_min": float(scores.min()), "score_max": float(scores.max()),
           "scores_below_zero": int(np.count_nonzero(scores < 0)),
           "scores_above_one": int(np.count_nonzero(scores > 1)),
           "clipped_at_zero": int(np.count_nonzero(c == 0)),
           "clipped_at_one": int(np.count_nonzero(c == 1)),
           "feature_min": float(F.min()), "feature_max": float(F.max()),
           "zero_feature_count": int(np.count_nonzero(F == 0)),
           "canonical_enclosure_claim": False}
    if view == "P":
        positive = F[F > 0]
        errors = []
        start = 0
        for width in widths:
            errors.append(float(np.max(np.abs(F[..., start:start + width].sum(axis=-1) - 1))))
            start += width
        out.update(zero_probability_count=int(np.count_nonzero(F == 0)),
                   rows_with_zero_probability=int(np.count_nonzero(np.any(F == 0, axis=-1))),
                   minimum_positive_probability=(float(positive.min()) if len(positive) else None),
                   max_probability_sum_error=max(errors))
    if view in {"LRM", "LW"}:
        tau = float(payload["tau"])
        omega = float(payload["omega"])
        recovered = []
        start = 0
        for index, width in enumerate(widths):
            block = F[..., start:start + width]
            if view == "LW":
                v = block / omega
            else:
                r, mass = block[..., :-1], block[..., -1:]
                z = r @ np.asarray(payload["walsh"][index], dtype=np.float64)
                v = z + (mass - logsumexp(omega * z, axis=-1, keepdims=True)) / omega
            recovered.append(1.0 + tau - np.exp(v))
            start += width
        out["max_clipped_score_reconstruction_error"] = float(np.max(np.abs(np.concatenate(recovered, axis=-1) - c)))
    return out


def paired_coordinate_diagnostics(scores, block_sizes, omega, tau=TAU) -> dict:
    """Record matched exact-information controls at their operational precision."""
    widths = _widths(block_sizes)
    views = {name: transform_scores(scores, view_payload(name, widths, omega, tau))
             for name in ("P", "CS", "LR0", "LRM", "LW")}
    starts = np.cumsum((0,) + widths)
    r_equal = True
    reconstructed = []
    for i, width in enumerate(widths):
        lo, hi = int(starts[i]), int(starts[i + 1])
        r0 = views["LR0"][..., lo:hi - 1]
        rm = views["LRM"][..., lo:hi - 1]
        r_equal = r_equal and np.array_equal(r0, rm)
        z = r0 @ np.asarray(walsh_contrasts(width), dtype=np.float64)
        eta = float(omega) * z
        reconstructed.append(np.exp(eta - logsumexp(eta, axis=-1, keepdims=True)))
    restored_p = np.concatenate(reconstructed, axis=-1)
    if not r_equal:
        raise ArithmeticError("Matched LR0/LRM contrast coordinates differ")
    return {"shared_LR0_LRM_contrasts_bit_identical": True,
            "P_zero_probability_count": int(np.count_nonzero(views["P"] == 0)),
            "max_P_from_LR0_reconstruction_error": float(np.max(np.abs(restored_p - views["P"]))),
            "max_CS_from_LRM_reconstruction_error": representation_diagnostics(
                scores, view_payload("LRM", widths, omega, tau), views["LRM"])["max_clipped_score_reconstruction_error"],
            "max_CS_from_LW_reconstruction_error": representation_diagnostics(
                scores, view_payload("LW", widths, omega, tau), views["LW"])["max_clipped_score_reconstruction_error"],
            "scope": "Operational rounding diagnostics only; no pure-coordinate or information-loss inference."}

