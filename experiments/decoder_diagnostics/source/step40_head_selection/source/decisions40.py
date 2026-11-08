"""Exact interval consequences for Step40's fixed mixed comparator collection.

The affine comparator is the stored, fitted rational coefficient payload.  The
QP comparator is the midpoint extension of the unique exact anchor optimizer
for its already fixed training problem.  This module does not fit either head,
compute a feature map, open data, or verify an upstream fitting certificate.
Its inputs must come from the separately checked prediction/loss pipeline.

These elementary interval deductions are a decision-utility endpoint, not a
new safe hyperparameter-search theorem.  Every candidate is fitted at the
predeclared numerical budget; no computational saving is asserted.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction as F
import hashlib
import json


SCHEMA = "step40-mixed-comparator-decision-certificate-v1"
PAIRED_SCHEMA = "step40-same-map-head-comparisons-v1"
SCOPE = "FIXED_AFFINE_AND_EXACT_QP"
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
HEAD_FAMILIES = ("affine", "qp")


def _q(value):
    """Read declared exact rationals; never silently reinterpret a float."""
    if isinstance(value, bool) or not isinstance(value, (str, int, F)):
        raise ValueError("Decision inputs must use exact rational strings or integers")
    return F(value)


def _bounds(record, name="interval", *, nonnegative=False):
    if not isinstance(record, Mapping):
        raise ValueError(name + " must be an interval object")
    try:
        lo, hi = _q(record["lower"]), _q(record["upper"])
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(name + " lacks valid rational endpoints") from exc
    if lo > hi or (nonnegative and lo < 0):
        raise ValueError(name + " has invalid ordered endpoints")
    if "width" in record and _q(record["width"]) != hi - lo:
        raise ValueError(name + " has inconsistent width")
    return lo, hi


def box(lo, hi):
    lo, hi = _q(lo), _q(hi)
    if lo > hi:
        raise ValueError("Reversed interval")
    return {"lower": str(lo), "upper": str(hi), "width": str(hi - lo)}


def _digest(value):
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _scope():
    return {
        "comparator_scope": SCOPE,
        "assurance": ASSURANCE,
        "affine_comparator": "EXACT_STORED_FITTED_COEFFICIENT_PAYLOAD",
        "qp_comparator": "MIDPOINT_EXTENSION_OF_UNIQUE_EXACT_TRAINING_ANCHOR_OPTIMUM",
        "same_fixed_maps_training_problems_and_development_labels": True,
        "affine_training_optimizer_certified": False,
        "native_transform_error_enclosed": False,
        "future_response_coverage_claim": False,
        "population_risk_claim": False,
        "computational_savings_claim": False,
        "all_candidates_fitted_at_predeclared_budget": True,
    }


def _comparator_rows(rows, expected_maps, operational_risk_key, qp_optimizer_risk_key):
    """Validate and project source rows to the exact mixed comparison inputs."""
    if isinstance(expected_maps, (str, bytes)) or not isinstance(expected_maps, Sequence):
        raise ValueError("Expected maps must be an explicit nonempty sequence")
    names = list(expected_maps)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Expected map roster is empty, duplicated, or malformed")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("Candidate rows must be a sequence")
    if len(rows) != 2 * len(names):
        raise ValueError("Exactly one affine and one QP head per declared map are required")
    lookup, ids = {}, set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Candidate must be an object")
        cid, name, recipe = row.get("candidate_id"), row.get("map"), row.get("recipe")
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("Candidate IDs must be unique nonempty strings")
        if name not in names:
            raise ValueError("Candidate uses a map outside the fixed roster")
        if recipe not in ("affine", "qp_regularized"):
            raise ValueError("Only the two declared certified head recipes can be compared")
        family = "affine" if recipe == "affine" else "qp"
        if (name, family) in lookup:
            raise ValueError("Duplicate head family for a fixed map")
        if operational_risk_key not in row:
            raise ValueError("Operational risk is missing")
        op_lo, op_hi = _bounds(row[operational_risk_key], "operational risk", nonnegative=True)
        if family == "affine":
            if row.get("ridge_lambda") is not None or row.get("extension_weight") is not None:
                raise ValueError("Affine selector rows cannot carry QP hyperparameters")
            if row.get(qp_optimizer_risk_key) is not None:
                raise ValueError("No exact affine-training-optimizer risk is supported")
            if op_lo != op_hi:
                raise ValueError("Stored affine arithmetic risk must be an exact point")
            lo, hi = op_lo, op_hi
            kind = "FIXED_AFFINE_PAYLOAD"
            risk_source = operational_risk_key
        else:
            if _q(row.get("ridge_lambda")) != 0 or _q(row.get("extension_weight")) != F(1, 2):
                raise ValueError("Step40 QP must use lambda zero and midpoint extension")
            if qp_optimizer_risk_key not in row or row[qp_optimizer_risk_key] is None:
                raise ValueError("Exact-QP optimizer risk enclosure is required")
            lo, hi = _bounds(row[qp_optimizer_risk_key], "QP optimizer risk", nonnegative=True)
            if not lo <= op_lo <= op_hi <= hi:
                raise ValueError("Expanded QP risk must contain the operational risk interval")
            kind = "EXACT_QP_ANCHOR_OPTIMUM_MIDPOINT_EXTENSION"
            risk_source = qp_optimizer_risk_key
        ids.add(cid)
        lookup[(name, family)] = {
            "candidate_id": cid,
            "map": name,
            "head_family": family,
            "comparator_kind": kind,
            "operational_risk": box(op_lo, op_hi),
            "mixed_comparator_risk": box(lo, hi),
            "risk_source": risk_source,
        }
    # Explicit map order, then affine/QP, makes the input projection canonical.
    for name in names:
        if any((name, family) not in lookup for family in HEAD_FAMILIES):
            raise ValueError("A declared map lacks a head family")
    return [lookup[(name, family)] for name in names for family in HEAD_FAMILIES]


def _paired_rows(comparators, expected_maps):
    lookup = {(row["map"], row["head_family"]): row for row in comparators}
    result = []
    for name in expected_maps:
        affine, qp = lookup[(name, "affine")], lookup[(name, "qp")]
        alo, ahi = _bounds(affine["mixed_comparator_risk"])
        qlo, qhi = _bounds(qp["mixed_comparator_risk"])
        opalo, opahi = _bounds(affine["operational_risk"])
        opqlo, opqhi = _bounds(qp["operational_risk"])
        lo, hi = qlo - ahi, qhi - alo
        result.append({
            "map": name,
            "affine_candidate_id": affine["candidate_id"],
            "qp_candidate_id": qp["candidate_id"],
            "difference_definition": "QP_MINUS_AFFINE_PHYSICAL_MSE",
            "mixed_difference": box(lo, hi),
            "operational_difference": box(opqlo - opahi, opqhi - opalo),
            "status": ("AFFINE_STRICTLY_LOWER_RISK" if lo > 0 else
                       "QP_STRICTLY_LOWER_RISK" if hi < 0 else
                       "NO_STRICT_SEPARATION"),
        })
    return result


def paired_head_comparisons(rows, *, expected_maps,
                            operational_risk_key="pooled_physical_risk",
                            qp_optimizer_risk_key="pooled_optimizer_risk",
                            cohort_role="development"):
    """Same-map head-recipe contrasts, optionally for predeclared test reporting.

    Positive QP-minus-affine differences favor the stored affine predictor.
    A test comparison is descriptive and cannot be used to select a head.
    """
    if cohort_role not in ("development", "test"):
        raise ValueError("Only declared development or test reporting is supported")
    comparators = _comparator_rows(rows, expected_maps, operational_risk_key, qp_optimizer_risk_key)
    scope = _scope()
    # The comparison cohort, including its fixed labels, is specified below;
    # test predictions have already been sealed before their own label reveal.
    scope.pop("same_fixed_maps_training_problems_and_development_labels")
    scope["same_fixed_maps_training_problems_and_comparison_labels"] = True
    return {
        "schema": PAIRED_SCHEMA,
        **scope,
        "cohort_role": cohort_role,
        "selection_performed_by_this_function": False,
        "test_reselection_allowed": False,
        "map_count": len(expected_maps),
        "source_comparator_rows_digest": _digest(comparators),
        "comparator_rows": comparators,
        "pairs": _paired_rows(comparators, expected_maps),
        "attribution_scope": "PAIRED_HEAD_RECIPES_ON_THE_SAME_STORED_MAP_NOT_PURE_CAPACITY_EFFECT",
    }


def mixed_decision_certificate(rows, selected_id, *, expected_maps):
    """Enclose mixed-family model-selection decisions on fixed development data.

    Operational selection has already happened and uses the implemented heads'
    pooled arithmetic risks.  This function does not change that selection.
    It asks which decisions remain established when only QP anchor training
    is idealized to its exact optimizer, keeping each fitted affine head fixed.
    """
    comparators = _comparator_rows(rows, expected_maps, "pooled_physical_risk", "pooled_optimizer_risk")
    lookup = {row["candidate_id"]: row for row in comparators}
    if selected_id not in lookup:
        raise ValueError("Operational selected ID is outside the fixed certified collection")
    incumbent = min(comparators, key=lambda row: (_bounds(row["mixed_comparator_risk"])[1], row["candidate_id"]))
    threshold = _bounds(incumbent["mixed_comparator_risk"])[1]
    survivors, excluded = [], []
    for row in comparators:
        lo, _ = _bounds(row["mixed_comparator_risk"])
        if lo <= threshold:
            survivors.append(row["candidate_id"])
        else:
            excluded.append({"candidate_id": row["candidate_id"],
                             "strict_exclusion_margin": str(lo - threshold)})
    if not survivors:
        raise ArithmeticError("A valid finite collection must retain at least one minimizer candidate")
    winner = lookup[selected_id]
    others = [row for row in comparators if row["candidate_id"] != selected_id]
    nearest = min(others, key=lambda row: (_bounds(row["mixed_comparator_risk"])[0], row["candidate_id"]))
    nearest_lower = _bounds(nearest["mixed_comparator_risk"])[0]
    winner_upper = _bounds(winner["mixed_comparator_risk"])[1]
    margin = nearest_lower - winner_upper
    family_intervals = {}
    for family in HEAD_FAMILIES:
        entries = [row for row in comparators if row["head_family"] == family]
        family_intervals[family] = box(
            min(_bounds(row["mixed_comparator_risk"])[0] for row in entries),
            min(_bounds(row["mixed_comparator_risk"])[1] for row in entries),
        )
    alo, ahi = _bounds(family_intervals["affine"])
    qlo, qhi = _bounds(family_intervals["qp"])
    family_difference = box(qlo - ahi, qhi - alo)
    family_status = ("AFFINE_FAMILY_STRICTLY_LOWER_RISK" if qlo > ahi else
                     "QP_FAMILY_STRICTLY_LOWER_RISK" if qhi < alo else
                     "NO_STRICT_SEPARATION")
    unique_survivor = survivors[0] if len(survivors) == 1 else None
    return {
        "schema": SCHEMA,
        **_scope(),
        "cohort_role": "development",
        "source_comparator_rows_digest": _digest(comparators),
        "comparator_rows": comparators,
        "candidate_count": len(comparators),
        "map_count": len(expected_maps),
        "operational_selected_id": selected_id,
        "operational_selection_changed": False,
        "test_labels_used": False,
        "candidate_exclusion": {
            "criterion": "EXCLUDE_IF_LOWER_STRICTLY_EXCEEDS_MINIMUM_UPPER",
            "incumbent_upper_candidate_id": incumbent["candidate_id"],
            "minimum_upper": str(threshold),
            "survivor_ids": survivors,
            "survivor_count": len(survivors),
            "excluded": excluded,
            "excluded_count": len(excluded),
            "excluded_fraction": str(F(len(excluded), len(comparators))),
            "unique_survivor_id": unique_survivor,
            "unique_survivor_agrees_with_operational_selection": unique_survivor == selected_id if unique_survivor else None,
            "excluded_candidates_cannot_minimize_mixed_comparator_risk": True,
            "equality_is_retained": True,
            "fitting_work_avoided": False,
        },
        "selected_winner": {
            "selected_id": selected_id,
            "selected_mixed_comparator_risk": winner["mixed_comparator_risk"],
            "nearest_competitor_id": nearest["candidate_id"],
            "competitor_minimum_lower": str(nearest_lower),
            "strict_margin_lower": str(margin),
            "status": "CERTIFIED_UNIQUE_MIXED_COMPARATOR_WINNER" if margin > 0 else "UNRESOLVED",
        },
        "head_family_resolution": {
            "family_minimum_risks": family_intervals,
            "difference_definition": "MIN_QP_MINUS_MIN_FIXED_AFFINE_PHYSICAL_MSE",
            "difference": family_difference,
            "status": family_status,
            "family_minimum_interval_rule": "[min_c lower_c, min_c upper_c] WITHIN EACH FAMILY",
        },
        "same_map_comparisons": _paired_rows(comparators, expected_maps),
        "attribution_scope": "PAIRED_HEAD_RECIPES_ON_THE_SAME_STORED_MAP_NOT_PURE_CAPACITY_EFFECT",
        "utility_scope": "POST_FIT_DECISION_RESOLUTION_AT_THE_PREDECLARED_NUMERICAL_BUDGET",
    }
