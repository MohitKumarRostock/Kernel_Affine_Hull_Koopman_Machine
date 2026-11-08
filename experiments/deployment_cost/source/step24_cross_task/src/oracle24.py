"""Exact per-history HMM diagnostics with rigorously enclosed sample means.

This module consumes arrays supplied by the caller; it never generates data,
opens a benchmark file, or selects a predictor. The production caller must
enforce the panel fit seal before supplying held-out histories. Manufactured
inputs can exercise the same pure evaluator before that seal.

Each oracle quantity and conditional regret is an exact rational for one
history. We bracket each such value in binary64 first, then add the dyadic
endpoints exactly. We never form a giant common denominator by summing the
arbitrary rational posterior quantities across benchmark histories.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from numbers import Integral
from pathlib import Path
import time

import numpy as np

import common24 as C
import data24 as D


def _mean_enclosure(nominal: np.ndarray, lower: np.ndarray,
                    upper: np.ndarray) -> dict:
    """Enclose the arithmetic mean of independently valid per-row boxes."""
    nominal = np.asarray(nominal, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if (nominal.ndim != 1 or not len(nominal) or lower.shape != nominal.shape
            or upper.shape != nominal.shape or not np.isfinite(nominal).all()
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower > nominal) or np.any(nominal > upper)):
        raise ValueError("Finite, nonempty, ordered per-row enclosures are required")
    n = len(nominal)
    low_exact = C.exact_sum_dyadics(lower) / n
    high_exact = C.exact_sum_dyadics(upper) / n
    nominal_exact = C.exact_sum_dyadics(nominal) / n
    return {
        "mean": float(nominal_exact),
        "lower": C.bracket(low_exact)["lower"],
        "upper": C.bracket(high_exact)["upper"],
        "lower_exact": low_exact,
        "upper_exact": high_exact,
        "mean_of_binary64_representatives_exact": nominal_exact,
        "rows": n,
        "max_per_row_enclosure_width": float(np.max(upper - lower)),
        "scope": "Arithmetic average over the supplied finite history sample, not a population expectation or confidence interval",
    }


def _validated_inputs(X, Y, horizons: Sequence[int], predictions: Mapping):
    h = D.hmm_module()
    x_raw, y_raw = np.asarray(X), np.asarray(Y)
    if x_raw.dtype.kind == "b" or y_raw.dtype.kind == "b":
        raise ValueError("Boolean arrays are not raw HMM observations")
    x, y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    horizons = tuple(horizons)
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(isinstance(k, bool) or not isinstance(k, Integral)
                   or int(k) not in h.HORIZONS for k in horizons)):
        raise ValueError("Distinct declared HMM horizons are required")
    horizons = tuple(int(k) for k in horizons)
    if x.ndim != 2 or x.shape[1] != h.HISTORY_LENGTH or not len(x):
        raise ValueError("The oracle requires a nonempty matrix of exactly 32 raw observations")
    if y.shape != (len(x), len(horizons)):
        raise ValueError("Target columns must match the supplied horizon order")
    if (not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isin(x, h.SYMBOL_VALUES).all()
            or not np.isin(y, h.SYMBOL_VALUES).all()):
        raise ValueError("Raw oracle observations/targets must use the exact four-symbol alphabet")
    if not isinstance(predictions, Mapping) or set(predictions) != {str(k) for k in horizons}:
        raise ValueError("Prediction horizon keys must exactly match the supplied horizons")
    checked = {}
    for horizon in horizons:
        models = predictions[str(horizon)]
        if not isinstance(models, Mapping) or any(not isinstance(k, str) or not k for k in models):
            raise ValueError("Each horizon requires a candidate-id to forecast-array mapping")
        checked[horizon] = {}
        for cid in sorted(models):
            p = np.asarray(models[cid], dtype=np.float64)
            if p.shape != (len(x),) or not np.isfinite(p).all() or np.any((p < -1.) | (p > 1.)):
                raise ValueError("Every HMM forecast must be finite, scalar, and clipped to [-1,1]")
            checked[horizon][cid] = p
    return x, y, horizons, checked


def _allocate_triple(arrays: dict, prefix: str, n: int):
    for suffix in ("", "_lower", "_upper"):
        arrays[prefix + suffix] = np.empty(n, dtype=np.float64)
    return arrays[prefix], arrays[prefix + "_lower"], arrays[prefix + "_upper"]


def _put_exact(triple, row: int, value: Fraction, h) -> None:
    enclosed = h.binary64_bracket(value)
    triple[0][row] = enclosed["nominal"]
    triple[1][row] = enclosed["lower"]
    triple[2][row] = enclosed["upper"]


def evaluate_oracle(X, Y, horizons, predictions):
    """Return ``(JSON-safe summary, flat numeric arrays)`` for a fixed panel.

    ``predictions`` is ``{str(h): {candidate_id: shape-(n,) forecasts}}``.
    Every oracle conditions on X alone, using the pinned generative law. Y is
    used only for realized risks after all oracle quantities are computed.

    The summary has ``horizons[str(h)]`` records with
    ``oracle_realized_risk``, ``conditional_variance_mean``,
    ``oracle_conditional_mean``, and ``models[cid]``. A model record contains
    ``conditional_regret_mean``, ``realized_risk``, and
    ``realized_mse_minus_binary64_oracle``. Model array prefixes use stable
    numeric slots to avoid imposing path-like restrictions on candidate IDs.
    """
    started = time.perf_counter_ns()
    x, y, horizons, model_predictions = _validated_inputs(X, Y, horizons, predictions)
    h = D.hmm_module()
    n = len(x)
    arrays, triples, model_prefixes = {}, {}, {}
    for horizon in horizons:
        oracle_prefix = f"h{horizon}_oracle_mean"
        variance_prefix = f"h{horizon}_conditional_variance"
        triples[(horizon, "mean")] = _allocate_triple(arrays, oracle_prefix, n)
        triples[(horizon, "variance")] = _allocate_triple(arrays, variance_prefix, n)
        for slot, cid in enumerate(model_predictions[horizon]):
            prefix = f"h{horizon}_model{slot:03d}_conditional_regret"
            triples[(horizon, cid, "regret")] = _allocate_triple(arrays, prefix, n)
            model_prefixes[(horizon, cid)] = prefix
    for row, raw_history in enumerate(x):
        symbols = h.symbols_from_values(raw_history)
        for horizon in horizons:
            oracle = h.oracle_from_symbols(symbols, horizon)
            if oracle.variance < 0:
                raise ArithmeticError("Exact HMM conditional variance became negative")
            _put_exact(triples[(horizon, "mean")], row, oracle.mean, h)
            _put_exact(triples[(horizon, "variance")], row, oracle.variance, h)
            for cid, p in model_predictions[horizon].items():
                regret = (Fraction.from_float(float(p[row])) - oracle.mean) ** 2
                _put_exact(triples[(horizon, cid, "regret")], row, regret, h)

    results = {}
    for column, horizon in enumerate(horizons):
        mu = triples[(horizon, "mean")]
        variance = triples[(horizon, "variance")]
        oracle_realized = C.risk_record(y[:, column], mu[0])
        models = {}
        for cid, p in model_predictions[horizon].items():
            actual_risk = C.risk_record(y[:, column], p)
            difference = actual_risk["mse_exact"] - oracle_realized["mse_exact"]
            models[cid] = {
                "conditional_regret_mean": _mean_enclosure(*triples[(horizon, cid, "regret")]),
                "realized_risk": actual_risk,
                "realized_mse_minus_binary64_oracle": {
                    "value": float(difference), "exact": difference,
                    "enclosure": C.bracket(difference),
                    "interpretation": "Realized sample risk difference; this need not equal conditional regret and can be negative",
                },
                "array_prefix": model_prefixes[(horizon, cid)],
            }
        results[str(horizon)] = {
            "oracle_realized_risk": oracle_realized,
            "oracle_realized_risk_definition": "Actual squared loss of nearest-binary64 exact conditional means against supplied realized labels",
            "oracle_conditional_mean": _mean_enclosure(*mu),
            "conditional_variance_mean": _mean_enclosure(*variance),
            "oracle_mean_array_prefix": f"h{horizon}_oracle_mean",
            "conditional_variance_array_prefix": f"h{horizon}_conditional_variance",
            "models": models,
        }
    summary = {
        "schema": "step24-exact-history-oracle-evaluation-v1",
        "rows": n, "history_length": 32, "horizons_order": list(horizons),
        "conditioning": "Exactly the 32 supplied raw observations, stationary prior and known fixed HMM parameters; no hidden state, prehistory, future observation or model forecast",
        "oracle_used_for_selection": False,
        "arithmetic": "Exact rational per-history means, variances and conditional regrets; each value independently bracketed in binary64, then dyadic endpoints summed exactly and divided by the history count",
        "no_sum_of_arbitrary_posterior_denominators": True,
        "statistical_scope": "Finite supplied-history sample averages. Enclosures concern numerical arithmetic, not sampling uncertainty or population expectations.",
        "information_loss_identified": False,
        "horizons": results,
        "elapsed_nanoseconds": int(time.perf_counter_ns() - started),
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(C.BASE)): C.sha(Path(__file__).resolve()),
            str((D.STEP23 / "controlled/hmm23.py").relative_to(C.BASE)): C.sha(D.STEP23 / "controlled/hmm23.py"),
            str(Path(C.__file__).resolve().relative_to(C.BASE)): C.sha(Path(C.__file__).resolve()),
            str(Path(D.__file__).resolve().relative_to(C.BASE)): C.sha(Path(D.__file__).resolve()),
        },
    }
    return C.jsonable(summary), arrays
