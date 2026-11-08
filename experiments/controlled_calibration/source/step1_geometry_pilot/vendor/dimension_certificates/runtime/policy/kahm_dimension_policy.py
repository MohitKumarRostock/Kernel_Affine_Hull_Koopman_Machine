"""Shared KAHM subspace policy and transparent native PCA rank accounting.

The native policy requests min(cap, ambient dimension, local sample count - 1).
The default cap is 20.  The historical normal-departure restriction is an
explicitly named replay policy, never an implicit native default.

This module does not choose a policy from prediction outcomes.  Callers retain
their reference storage convention (the pinned native implementation rounds to
float32 before calling PCA).  The supplied pinned ``dim_reduce`` function is
the sole source of the returned PCA basis and projected coordinates.  No extra
eigendecomposition is performed for logging.  The total positive-eigenvalue
count is recorded exactly only when the returned rank identifies it; otherwise
the record supplies a lower bound and leaves the total unknown.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

DEFAULT_KAHM_CAP = 20
NATIVE_POLICY = "native"
LEGACY_NORMAL_DEPARTURE_POLICY = "legacy_normal_departure"
POLICIES = (NATIVE_POLICY, LEGACY_NORMAL_DEPARTURE_POLICY)
SCHEMA = "kahm-dimension-policy-v1"
POLICY_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a coerced numeric value")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def validate_dimension_cap(cap) -> int:
    """Validate an explicit native/default or deliberate ablation cap."""
    return _integer(cap, "cap", 1)


def dimension_request(ambient_dimension, reference_count, *,
                      cap=DEFAULT_KAHM_CAP, policy=NATIVE_POLICY) -> dict:
    """Return a JSON-safe request for one actual local reference matrix.

    ``reference_count`` is the number of references supplied to this local
    autoencoder/PCA, after any original sampling or singleton augmentation.
    A one-reference request is recorded as rank zero; fitting it requires an
    explicit existing fallback or raises a clear unsupported-input error.
    """
    d = _integer(ambient_dimension, "ambient_dimension", 1)
    n = _integer(reference_count, "reference_count", 1)
    cap = validate_dimension_cap(cap)
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    ambient_limit = d if policy == NATIVE_POLICY else d - 1
    limits = {"requested_cap": cap, "ambient_dimension": ambient_limit,
              "local_sample_rank": n - 1}
    initial = min(limits.values())
    return {"schema": SCHEMA, "policy_source_sha256": POLICY_SOURCE_SHA256,
            "policy": policy, "requested_cap": cap,
            "ambient_dimension": d, "local_reference_count": n,
            "initial_admissible_rank": initial, "initial_limits": limits,
            "initial_limit_reasons": [key for key, value in limits.items()
                                      if value == initial],
            "normal_direction_reserved": policy == LEGACY_NORMAL_DEPARTURE_POLICY}


def finalize_dimension_audit(request: dict, effective_rank, *,
                             reasons: Iterable[str] = (), pca_trace=()) -> dict:
    """Attach the realized P.shape[1] to a validated local request."""
    if not isinstance(request, dict):
        raise TypeError("request must be the dictionary from dimension_request")
    expected = dimension_request(request.get("ambient_dimension"),
                                 request.get("local_reference_count"),
                                 cap=request.get("requested_cap"),
                                 policy=request.get("policy"))
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("Dimension request was modified or is inconsistent")
    rank = _integer(effective_rank, "effective_rank", 0)
    if rank > expected["initial_admissible_rank"]:
        raise ValueError("Actual basis rank exceeds the initial admissible rank")
    if isinstance(reasons, str):
        raise TypeError("reasons must be an iterable of nonempty strings")
    reasons = list(reasons)
    if any(not isinstance(x, str) or not x for x in reasons):
        raise ValueError("Dimension reduction reasons must be nonempty strings")
    if rank < expected["initial_admissible_rank"] and not reasons:
        raise ValueError("An effective-rank reduction requires an explicit reason")
    result = copy.deepcopy(expected)
    result.update(effective_rank=rank,
                  rank_reduction_reasons=list(dict.fromkeys(reasons)),
                  pca_trace=copy.deepcopy(list(pca_trace)))
    return result


def fit_native_pca(reference_columns, *, dim_reduce: Callable,
                   cap=DEFAULT_KAHM_CAP, policy=NATIVE_POLICY,
                   min_range=1e-3, allow_rank_zero=False):
    """Return (projected, basis, audit) using the pinned native PCA/range loop.

    References have shape (ambient dimension, local reference count).  Native
    shrinkage stops at requested rank one even if its range remains below the
    threshold; ``range_threshold_satisfied`` records that outcome explicitly.

    ``allow_rank_zero=True`` opts into the forecasting wrapper's pre-existing
    zero-rank behavior for singleton/constant references.  It is not native
    success, and the audit labels the failure and fallback.  All other failures
    propagate.  Risk/native callers leave this option false.
    """
    refs = np.asarray(reference_columns)
    if refs.ndim != 2 or min(refs.shape) == 0:
        raise ValueError("References must be a nonempty d-by-N matrix")
    if refs.dtype.kind not in "fiu" or not np.isfinite(refs).all():
        raise ValueError("References must contain finite real numeric values")
    if not callable(dim_reduce):
        raise TypeError("dim_reduce must be the pinned callable")
    if isinstance(min_range, (bool, np.bool_)) or not isinstance(min_range, Real):
        raise TypeError("min_range must be a finite positive real value")
    min_range = float(min_range)
    if not math.isfinite(min_range) or min_range <= 0:
        raise ValueError("min_range must be finite and positive")
    if not isinstance(allow_rank_zero, (bool, np.bool_)):
        raise TypeError("allow_rank_zero must be a boolean")
    allow_rank_zero = bool(allow_rank_zero)
    request = dimension_request(refs.shape[0], refs.shape[1], cap=cap, policy=policy)
    rank = request["initial_admissible_rank"]
    common = {"reference_dtype": str(refs.dtype), "pca_working_dtype": "float64",
              "pca_eigenvalue_threshold": float(np.finfo(float).eps),
              "range_threshold": min_range,
              "rank_zero_fallback_allowed": allow_rank_zero,
              "diagnostic_eigendecomposition_count": 0}
    trace, steps = [], 0
    positive = None
    positive_lower_bound = 0

    def unsupported(message, reason):
        record = finalize_dimension_audit(request, 0, reasons=[reason], pca_trace=trace)
        record.update(common, empirical_positive_eigenvalue_count=positive,
                      empirical_positive_eigenvalue_count_exact=positive is not None,
                      empirical_positive_eigenvalue_count_lower_bound=positive_lower_bound,
                      range_reduction_steps=steps, final_requested_rank=rank,
                      native_pca_calls=len(trace) + int(message == "No eigenvalues > eps"),
                      range_threshold_satisfied=None,
                      range_floor_reached=False, fallback_applied=allow_rank_zero,
                      native_failure=message,
                      status="inherited_zero_rank_fallback" if allow_rank_zero else "unsupported_native_input")
        if allow_rank_zero:
            return np.empty((0, refs.shape[1])), np.empty((refs.shape[0], 0)), record
        error = ValueError(message)
        error.dimension_audit = record
        raise error

    if rank == 0:
        return unsupported("No positive admissible local subspace dimension",
                           "insufficient_local_references" if refs.shape[1] == 1
                           else "legacy_normal_dimension_zero")
    while True:
        try:
            projected, basis = dim_reduce(refs, rank)
        except ValueError as error:
            if str(error) != "No eigenvalues > eps":
                raise
            positive = 0
            return unsupported(str(error), "no_positive_empirical_variance")
        projected, basis = np.asarray(projected), np.asarray(basis)
        if (basis.ndim != 2 or basis.shape[0] != refs.shape[0]
                or not 1 <= basis.shape[1] <= rank
                or projected.shape != (basis.shape[1], refs.shape[1])
                or not np.isfinite(projected).all() or not np.isfinite(basis).all()):
            raise ValueError("Pinned PCA callback returned an invalid projection")
        positive_lower_bound = max(positive_lower_bound, int(basis.shape[1]))
        if basis.shape[1] < rank or basis.shape[1] == refs.shape[0]:
            positive = int(basis.shape[1])
        ranges = projected.max(axis=1) - projected.min(axis=1)
        smallest = float(np.min(ranges))
        trace.append({"requested_rank": rank, "returned_rank": int(basis.shape[1]),
                      "minimum_projected_range": smallest,
                      "range_threshold_satisfied": smallest >= min_range})
        if smallest >= min_range or rank <= 1:
            break
        rank -= 1
        steps += 1
    reasons = []
    if trace[0]["returned_rank"] < request["initial_admissible_rank"]:
        reasons.append("positive_eigenvalue_limit")
    if steps:
        reasons.append("projected_range_reduction")
    floor_reached = rank <= 1 and smallest < min_range
    if floor_reached:
        reasons.append("range_threshold_unmet_at_rank_one")
    record = finalize_dimension_audit(request, int(basis.shape[1]),
                                      reasons=reasons, pca_trace=trace)
    record.update(common, empirical_positive_eigenvalue_count=positive,
                  empirical_positive_eigenvalue_count_exact=positive is not None,
                  empirical_positive_eigenvalue_count_lower_bound=positive_lower_bound,
                  range_reduction_steps=steps, final_requested_rank=rank,
                  native_pca_calls=len(trace),
                  range_threshold_satisfied=smallest >= min_range,
                  range_floor_reached=floor_reached, fallback_applied=False,
                  native_failure=None, status="fitted")
    return projected, basis, record

