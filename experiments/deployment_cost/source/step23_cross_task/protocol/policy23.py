"""Pure Step23 selectors and exact observed-utility screen.

No files, datasets, generators, fitted estimators, or clocks are accessed here.
The caller supplies the complete frozen task roster, including its unfitted
baselines. Candidate records use the template keys ``id``, ``index``, ``group``
plus ``validation_mse`` (stored nominal binary64) and
``complete_numeric_bytes`` (nonnegative integer). Unfitted baselines have group
``UNFITTED`` and frozen indices in the protocol's declared baseline order.

This small implementation intentionally also accepts manufactured rosters;
the separate roster audit must enforce the task's actual candidate counts.
Matched diagnostic candidates may be present but never enter these selectors.
Returned records are shallow copies; input records are not modified.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
import math
from numbers import Integral, Real


ALLOWANCE = Fraction(11, 10)
COMPARATOR_GROUPS = frozenset({"RAW", "NYSTROM", "UNFITTED"})
ALLOWED_GROUPS = COMPARATOR_GROUPS | {"P", "CS", "MATCHED_DIAGNOSTIC"}


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _nominal_mse(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("validation_mse must be a finite nonnegative binary64 number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("validation_mse is not finite in binary64") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("validation_mse must be finite and nonnegative")
    return result


def _exact_nonnegative(value: object, name: str) -> Fraction:
    """Do not silently reinterpret a rounded float as an exact risk/latency."""
    if isinstance(value, bool) or not isinstance(value, (Fraction, Integral)):
        raise ValueError(f"{name} must be an exact Fraction or integer")
    result = Fraction(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _candidate_record(candidate: Mapping) -> dict:
    if not isinstance(candidate, Mapping):
        raise ValueError("Each candidate must be a mapping")
    result = dict(candidate)
    if not isinstance(result.get("id"), str) or not result["id"]:
        raise ValueError("Candidate id must be a nonempty string")
    result["index"] = _nonnegative_integer(result["index"], "candidate index")
    if result.get("group") not in ALLOWED_GROUPS:
        raise ValueError("Unknown candidate group; no automatic comparator fallback")
    result["validation_mse"] = _nominal_mse(result["validation_mse"])
    result["complete_numeric_bytes"] = _nonnegative_integer(
        result["complete_numeric_bytes"], "complete_numeric_bytes")
    return result


def select_candidates(candidates: Iterable[Mapping]) -> dict:
    """Return the six declared selections and an exact validation cap.

    Accuracy ranking uses (nominal validation MSE, frozen candidate index).
    Qualification compares the exact binary64 MSE value with11/10 times the
    exact binary64 comparator MSE; it never rounds the multiplier to1.1.
    Cost ranking uses (complete bytes, nominal validation MSE, candidate index).

    An empty P/CS/comparator *menu* is an invalid run, not a fallback case.
    A nonempty P or CS menu with no qualifying entry has cost selection None.
    """
    records = [_candidate_record(candidate) for candidate in candidates]
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate candidate id")
    if len({row["index"] for row in records}) != len(records):
        raise ValueError("Duplicate frozen candidate index")
    groups = {
        "C": [row for row in records if row["group"] in COMPARATOR_GROUPS],
        "P": [row for row in records if row["group"] == "P"],
        "CS": [row for row in records if row["group"] == "CS"],
    }
    if any(not group for group in groups.values()):
        raise ValueError("The complete comparator, P, and CS menus are required")
    accuracy = {
        name: min(group, key=lambda row: (row["validation_mse"], row["index"]))
        for name, group in groups.items()
    }
    cap = ALLOWANCE * Fraction.from_float(accuracy["C"]["validation_mse"])
    result = {name + "_accuracy": row for name, row in accuracy.items()}
    result["validation_cap_exact"] = cap
    result["qualifying_counts"] = {}
    for name, group in groups.items():
        qualified = [row for row in group if Fraction.from_float(row["validation_mse"]) <= cap]
        result["qualifying_counts"][name] = len(qualified)
        result[name + "_cost"] = (
            min(qualified, key=lambda row: (row["complete_numeric_bytes"],
                                            row["validation_mse"], row["index"]))
            if qualified else None
        )
    # With a nonnegative comparator risk, its accuracy winner must qualify.
    assert result["C_cost"] is not None
    return result


def _observed_record(record: Mapping) -> dict:
    if not isinstance(record, Mapping):
        raise ValueError("Observed model record must be a mapping")
    result = dict(record)
    if not isinstance(result.get("id"), str) or not result["id"]:
        raise ValueError("Observed model id must be a nonempty string")
    result["test_mse_exact"] = _exact_nonnegative(result["test_mse_exact"], "test_mse_exact")
    result["complete_numeric_bytes"] = _nonnegative_integer(
        result["complete_numeric_bytes"], "complete_numeric_bytes")
    result["median_warm_batch_nanoseconds_per_row"] = _exact_nonnegative(
        result["median_warm_batch_nanoseconds_per_row"],
        "median_warm_batch_nanoseconds_per_row")
    return result


def observed_utility_gate(candidate: Mapping | None,
                          comparator_accuracy: Mapping,
                          comparator_cost: Mapping) -> dict:
    """Screen the frozen P_cost or CS_cost against the frozen references.

    Each observed record carries exact ``test_mse_exact`` and
    ``median_warm_batch_nanoseconds_per_row`` (Fraction/integer), complete bytes,
    id, and group. For a nine-repeat512-row timing panel, construct the latency
    as Fraction(median_total_nanoseconds,512), without rounding.

    None means no validation-qualified candidate and always fails the screen.
    A failing C_cost test risk is reported but never replaces either reference.
    Risk and latency boundaries are inclusive; the byte boundary is strict.
    """
    accuracy_reference = _observed_record(comparator_accuracy)
    cost_reference = _observed_record(comparator_cost)
    if (accuracy_reference.get("group") not in COMPARATOR_GROUPS
            or cost_reference.get("group") not in COMPARATOR_GROUPS):
        raise ValueError("Both frozen references must be declared comparators")
    cap = ALLOWANCE * accuracy_reference["test_mse_exact"]
    result = {
        "candidate_id": None,
        "candidate_group": None,
        "accuracy_reference_id": accuracy_reference["id"],
        "cost_reference_id": cost_reference["id"],
        "accuracy_reference_test_mse_exact": accuracy_reference["test_mse_exact"],
        "cost_reference_test_mse_exact": cost_reference["test_mse_exact"],
        "cost_reference_numeric_bytes": cost_reference["complete_numeric_bytes"],
        "cost_reference_latency_exact": cost_reference["median_warm_batch_nanoseconds_per_row"],
        "test_accuracy_cap_exact": cap,
        "cost_reference_accuracy_pass": cost_reference["test_mse_exact"] <= cap,
        "no_qualifying_candidate": candidate is None,
        "accuracy_pass": False,
        "memory_pass": False,
        "latency_pass": False,
        "accuracy_memory_pass": False,
        "all_three_pass": False,
    }
    if candidate is None:
        result["reason"] = "No validation-qualified candidate; no fallback"
        return result
    model = _observed_record(candidate)
    if model.get("group") not in {"P", "CS"}:
        raise ValueError("Only the declared P_cost or CS_cost may enter this screen")
    accuracy_pass = model["test_mse_exact"] <= cap
    memory_pass = model["complete_numeric_bytes"] < cost_reference["complete_numeric_bytes"]
    latency_pass = (model["median_warm_batch_nanoseconds_per_row"]
                    <= cost_reference["median_warm_batch_nanoseconds_per_row"])
    result.update({
        "candidate_id": model["id"],
        "candidate_group": model["group"],
        "candidate_test_mse_exact": model["test_mse_exact"],
        "candidate_numeric_bytes": model["complete_numeric_bytes"],
        "candidate_latency_exact": model["median_warm_batch_nanoseconds_per_row"],
        "accuracy_margin_exact": cap - model["test_mse_exact"],
        "memory_saving_bytes": cost_reference["complete_numeric_bytes"] - model["complete_numeric_bytes"],
        "latency_margin_exact": (cost_reference["median_warm_batch_nanoseconds_per_row"]
                                   - model["median_warm_batch_nanoseconds_per_row"]),
        "accuracy_pass": accuracy_pass,
        "memory_pass": memory_pass,
        "latency_pass": latency_pass,
        "accuracy_memory_pass": accuracy_pass and memory_pass,
        "all_three_pass": accuracy_pass and memory_pass and latency_pass,
        "reason": "Observed fixed-reference operational screen; no population claim",
    })
    return result
