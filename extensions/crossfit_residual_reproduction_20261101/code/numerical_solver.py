"""Scoped numerical amendment for the inherited centered ridge solver.

The original runtime function is always attempted first.  Successful calls
return its original result unchanged.  Only ``numpy.linalg.LinAlgError``
activates an independently checked SciPy ``gesvd`` decomposition of the same
centered design.  The ridge objective, coefficient formula, intercept, rank
tolerance, penalties, and output fields are unchanged.

Public API: ``install(runtime)``, ``clear_events()``, and ``events()``.
Installation replaces only that runtime's ``_ridge_path``.  It never replaces
``numpy.linalg.svd`` and does not affect the geometry's other decompositions.
Events cover every attempted fallback, including attempts that fail closed.
Ordinary successful original calls do not produce fallback events.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from functools import wraps
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.linalg


POLICY_ID = "numpy-first-gesvd-on-linalgerror-v1"
EVENT_SCHEMA = "crossfit-ridge-svd-fallback-event-v1"
TOLERANCES = {
    "relative_reconstruction_error": 1e-10,
    "left_orthogonality_frobenius": 1e-8,
    "right_orthogonality_frobenius": 1e-8,
    "relative_normal_equation_residual": 1e-8,
}

_EVENTS: list[dict[str, Any]] = []
_INSTALLATIONS: dict[int, dict[str, Any]] = {}


def clear_events():
    """Start an empty per-process fallback ledger; do not change installation."""
    _EVENTS.clear()


def events():
    """Return a detached JSON-safe ledger of recovered and failed attempts."""
    return copy.deepcopy(_EVENTS)


def _file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _array_hash(value):
    """Use the inherited array-hash convention, including dtype and shape."""
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def _exception(error):
    return {"type": type(error).__name__, "message": str(error)}


def _context():
    """Collect names already present in callers without reading target arrays."""
    frames = []
    preferred_prefix = None
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        while frame is not None and len(frames) < 12:
            if frame.f_code.co_filename != __file__:
                row = {"function": frame.f_code.co_name,
                       "file": frame.f_code.co_filename, "line": int(frame.f_lineno)}
                for name in ("prefix", "tag", "mode", "kind", "pair", "scaler", "cap",
                             "key", "fold", "seed", "task_id"):
                    value = frame.f_locals.get(name)
                    if isinstance(value, (str, int, float, bool)):
                        if isinstance(value, float) and not math.isfinite(value):
                            continue
                        row[name] = value
                if frame.f_code.co_name == "_select_ridge" and preferred_prefix is None:
                    value = frame.f_locals.get("prefix")
                    if isinstance(value, str):
                        preferred_prefix = value
                task = frame.f_locals.get("task")
                if isinstance(task, dict) and isinstance(task.get("task_id"), str):
                    row["task_id"] = task["task_id"]
                frames.append(row)
            frame = frame.f_back
    finally:
        del frame
    return {"ridge_prefix": preferred_prefix, "frames": frames}


def _finite(array, name, checks):
    result = bool(np.isfinite(array).all())
    checks[name] = result
    if not result:
        raise FloatingPointError("Nonfinite numerical fallback " + name)


def _checked_metric(value, name, checks):
    value = float(value)
    checks[name] = value if math.isfinite(value) else None
    if not math.isfinite(value) or value > TOLERANCES[name]:
        raise FloatingPointError("Numerical fallback failed " + name)
    return value


def _relative_norm(numerator, denominator):
    numerator, denominator = float(np.linalg.norm(numerator)), float(np.linalg.norm(denominator))
    # If both mathematical quantities are exactly zero, the identity holds.
    if numerator == 0.0 and denominator == 0.0:
        return 0.0
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def _fallback(runtime, x, y, alphas, event):
    """Repeat original ridge arithmetic, changing only the SVD driver."""
    checks = event["checks"]
    x, y = runtime._matrix(x, "ridge features"), runtime._matrix(y, "ridge targets")
    if len(x) != len(y):
        raise ValueError("Ridge feature and target rows differ")
    event["features_shape"] = list(x.shape)
    event["targets_shape"] = list(y.shape)
    event["features_dtype"] = str(x.dtype)
    event["targets_dtype"] = str(y.dtype)
    _finite(x, "features_finite", checks)
    _finite(y, "targets_finite", checks)
    mean, ymean = x.mean(axis=0), y.mean(axis=0)
    centered, response = x - mean, y - ymean
    event["centered_array_sha256"] = _array_hash(centered)
    event["response_array_sha256"] = _array_hash(response)
    _finite(mean, "feature_mean_finite", checks)
    _finite(ymean, "target_mean_finite", checks)
    _finite(centered, "centered_finite", checks)
    _finite(response, "response_finite", checks)
    alphas = list(alphas)
    event["alphas"] = []
    for alpha in alphas:
        alpha = float(alpha)
        event["alphas"].append(alpha if math.isfinite(alpha) else None)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge penalties must be positive and finite")

    u, singular, vt = scipy.linalg.svd(centered, full_matrices=False,
                                       lapack_driver="gesvd", check_finite=True, overwrite_a=False)
    width = min(centered.shape)
    checks["factor_shapes_valid"] = (u.shape == (len(x), width) and singular.shape == (width,)
                                      and vt.shape == (width, x.shape[1]))
    if not checks["factor_shapes_valid"]:
        raise FloatingPointError("Numerical fallback returned invalid factor shapes")
    for value, name in ((u, "u_finite"), (singular, "singular_values_finite"), (vt, "vt_finite")):
        _finite(value, name, checks)
    checks["singular_values_nonnegative_and_descending"] = bool(
        np.all(singular >= 0) and np.all(np.diff(singular) <= 0))
    if not checks["singular_values_nonnegative_and_descending"]:
        raise FloatingPointError("Numerical fallback returned invalid singular values")
    _checked_metric(_relative_norm((u * singular) @ vt - centered, centered),
                    "relative_reconstruction_error", checks)
    _checked_metric(np.linalg.norm(u.T @ u - np.eye(width)),
                    "left_orthogonality_frobenius", checks)
    _checked_metric(np.linalg.norm(vt @ vt.T - np.eye(width)),
                    "right_orthogonality_frobenius", checks)

    # These statements reproduce the inherited implementation and field order.
    projected = u.T @ response
    rank_tolerance = max(centered.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.)
    rank = int(np.count_nonzero(singular > rank_tolerance))
    response_rms = np.sqrt(np.mean(response * response, axis=0))
    _finite(projected, "projected_response_finite", checks)
    _finite(response_rms, "response_rms_finite", checks)
    event["feature_rank"] = rank
    event["rank_tolerance"] = float(rank_tolerance)
    event["largest_singular_value"] = float(singular[0]) if len(singular) else None
    event["smallest_singular_value"] = float(singular[-1]) if len(singular) else None
    output = []
    rhs = centered.T @ response
    _finite(rhs, "normal_equation_rhs_finite", checks)
    for alpha in alphas:
        alpha = float(alpha)
        coefficient = vt.T @ ((singular / (singular ** 2 + len(x) * alpha))[:, None] * projected)
        if not np.isfinite(coefficient).all():
            raise FloatingPointError("Nonfinite centered-SVD ridge coefficient")
        row = {"alpha": alpha, "coefficient_finite": True}
        event["ridge_checks"].append(row)
        normal_residual = centered.T @ (centered @ coefficient) + len(x) * alpha * coefficient - rhs
        _finite(normal_residual, "normal_equation_residual_finite", row)
        _checked_metric(_relative_norm(normal_residual, rhs),
                        "relative_normal_equation_residual", row)
        output.append({"mean": mean, "ymean": ymean, "coef": coefficient,
                       "alpha_by_horizon": np.full(y.shape[1], alpha),
                       "response_rms": response_rms, "feature_rank": rank,
                       "feature_width": x.shape[1], "training_rows": len(x)})
    return output


def install(runtime):
    """Install the policy uniformly for every caller of this runtime's ridge.

    Returns immutable descriptive data, not a fitted object.  Installation is
    idempotent and preserves the event ledger.  Clear it explicitly at the
    start of each fit task.  Replacing an already installed function by some
    other function is rejected rather than silently composing amendments.
    """
    key = id(runtime)
    if key in _INSTALLATIONS:
        installed = _INSTALLATIONS[key]
        if installed["runtime"] is not runtime or runtime._ridge_path is not installed["wrapper"]:
            raise RuntimeError("Installed ridge function was replaced")
        return copy.deepcopy(installed["provenance"])
    original = runtime._ridge_path
    if getattr(original, "_numerical_solver_policy", None) is not None:
        raise RuntimeError("A numerical ridge amendment is already installed by another module instance")
    runtime_path = getattr(runtime, "__file__", None)
    original_path = inspect.getsourcefile(original)
    provenance = {
        "policy_id": POLICY_ID, "scope": "runtime._ridge_path only",
        "primary": "original NumPy SVD centered ridge",
        "fallback": "scipy.linalg.svd(full_matrices=False, lapack_driver='gesvd', check_finite=True, overwrite_a=False)",
        "trigger": "numpy.linalg.LinAlgError from the original ridge path only",
        "adapter_file": str(Path(__file__).resolve()), "adapter_sha256": _file_hash(__file__),
        "runtime_file": str(runtime_path) if runtime_path else None,
        "runtime_sha256": _file_hash(runtime_path) if runtime_path else None,
        "original_function_file": original_path,
        "original_function_file_sha256": _file_hash(original_path) if original_path else None,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "tolerances": dict(TOLERANCES), "event_schema": EVENT_SCHEMA,
        "array_hash_convention": "SHA256(sorted JSON dtype/shape + NUL + C-order bytes)",
    }

    @wraps(original)
    def amended(x, y, alphas):
        try:
            return original(x, y, alphas)
        except np.linalg.LinAlgError as primary_error:
            event = {
                "schema": EVENT_SCHEMA, "policy_id": POLICY_ID, "event_index": len(_EVENTS) + 1,
                "utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "recovered": False,
                "triggering_exception": _exception(primary_error), "fallback_exception": None,
                "context": _context(), "solver_order": ["original_numpy_svd", "scipy_gesvd"],
                "provenance": copy.deepcopy(provenance), "tolerances": dict(TOLERANCES),
                "checks": {}, "ridge_checks": [],
            }
            try:
                result = _fallback(runtime, x, y, alphas, event)
            except BaseException as fallback_error:
                event["fallback_exception"] = _exception(fallback_error)
                raise
            else:
                event["status"] = "recovered"
                event["recovered"] = True
                return result
            finally:
                _EVENTS.append(event)

    amended._numerical_solver_policy = POLICY_ID
    runtime._ridge_path = amended
    _INSTALLATIONS[key] = {"runtime": runtime, "original": original,
                           "wrapper": amended, "provenance": provenance}
    return copy.deepcopy(provenance)
