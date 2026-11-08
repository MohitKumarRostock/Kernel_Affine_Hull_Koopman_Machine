"""Exact regularized anchor fits and weighted Lipschitz extensions for Step 39.

The module opens no files or datasets.  A numerical solution is only a proposal;
the inherited exact primal/dual witnesses enclose the effective-target problem.
An exact affine conversion then encloses the declared regularized objective on
the original training targets.  Extension weights are separate from the inner
repair interpolation coefficient named ``theta`` in the inherited witness.
"""
from __future__ import annotations

from fractions import Fraction as F
from math import isqrt
import time

import numpy as np

import learners38 as old


ASSURANCE = old.ASSURANCE
FIT_SCHEMA = "step39-regularized-training-fit-v1"
HEAD_SCHEMA = "step39-weighted-extension-head-v1"
PREDICTION_SCHEMA = "step39-weighted-head-prediction-v1"
REFINEMENT_SCHEMA = "step39-exact-optimum-prediction-enclosure-v1"
RIDGE_GRID = ("0", "1/4", "1")
EXTENSION_GRID = ("0", "1/2", "1")
KAPPA = "2"

canonical = old.canonical
canonical_bytes = old.canonical_bytes
digest = old.digest
content_digest = old.ff.digest
box = old.box
clip = old.clip
features_exact = old.features_exact
query_distances = old.query_distances
fit_affine = old.fit_affine
baseline_prediction = old.baseline_prediction
score = old.score


def sqrt_enclosure(value, *, bits=80):
    """Enclose sqrt of a nonnegative rational with exact dyadic arithmetic."""
    value = F(value)
    if value < 0 or type(bits) is not int or bits < 1:
        raise ValueError("A nonnegative radicand and positive integer precision are required")
    denominator = 1 << bits
    scaled_numerator = value.numerator << (2 * bits)
    floor = isqrt(scaled_numerator // value.denominator)
    exact = floor * floor * value.denominator == scaled_numerator
    lower = F(floor, denominator)
    upper = F(floor + (not exact), denominator)
    if not lower * lower <= value <= upper * upper:
        raise ArithmeticError("Square-root enclosure failed")
    return canonical({
        "bits": bits,
        "floor_scaled": str(floor),
        "exact": exact,
        "interval": box(lower, upper),
        "lower_square": lower * lower,
        "upper_square": upper * upper,
    })


def fit_regularized(z, y, *, ids, feature_id, ridge_lambda, kappa="2",
                    role="train", input_binding=None, settings=None):
    """Fit J(v)=mean((v-y)^2)+lambda*mean((v-mean(y))^2).

    For t=(y+lambda*mean(y))/(1+lambda), J=(1+lambda)*R_t+c.
    The inner fit is the unchanged Step 38 effective-target witness.  Its lower
    bound is never represented as a lower bound for unregularized original-y
    risk.  Original-y risks of every retained feasible candidate are separate.
    """
    started = time.perf_counter()
    original = list(map(F, y))
    lam = F(ridge_lambda)
    if not original or lam < 0:
        raise ValueError("A nonempty training target vector and nonnegative lambda are required")
    center = sum(original, F(0)) / len(original)
    scale = 1 + lam
    effective = [(value + lam * center) / scale for value in original]
    variance = sum(((value - center) ** 2 for value in original), F(0)) / len(original)
    constant = lam * variance / scale
    core = old.fit_values(
        z, effective, ids=ids, feature_id=feature_id, kappa=kappa, role=role,
        input_binding=input_binding, settings=settings,
    )
    risks = {}
    for candidate in core["primal_candidates"]:
        values = list(map(F, candidate["values"]))
        original_mse = sum(((v - yv) ** 2 for v, yv in zip(values, original)), F(0)) / len(original)
        regularizer_mse = sum(((v - center) ** 2 for v in values), F(0)) / len(original)
        objective = original_mse + lam * regularizer_mse
        effective_mse = F(candidate["mean"])
        if objective != scale * effective_mse + constant:
            raise ArithmeticError("Exact regularized-objective conversion failed")
        risks[candidate["source"]] = {
            "original_target_mse": original_mse,
            "regularizer_mse": regularizer_mse,
            "regularized_objective": objective,
            "effective_target_mse": effective_mse,
        }
    lower = scale * F(core["class_risk"]["lower"]) + constant
    upper = scale * F(core["class_risk"]["upper"]) + constant
    if not 0 <= lower <= upper:
        raise ArithmeticError("Regularized objective bracket is inconsistent")
    squared_upper = len(original) * (upper - lower) / scale
    square_root = sqrt_enclosure(squared_upper, bits=80)
    prediction_bound = {
        "schema": "step39-optimization-to-prediction-bound-v1",
        "squared_upper": squared_upper,
        "sqrt_enclosure": square_root,
        "uniform_upper": square_root["interval"]["upper"],
        "formula": "sqrt(n*(J_upper-J_lower)/(1+ridge_lambda))",
        "same_map_anchors_objective_kappa_range_and_extension_weight_required": True,
        "target": "canonical_weighted_extension_of_unique_exact_regularized_training_optimum",
        "future_response_coverage_claim": False,
        "population_risk_claim": False,
    }
    return canonical({
        "schema": FIT_SCHEMA,
        "role": role,
        "assurance": ASSURANCE,
        "feature_id": core["feature_id"],
        "class_id": core["class_id"],
        "feature_digest": core["feature_digest"],
        "n": len(original),
        "dimension": core["dimension"],
        "ids": list(ids),
        "kappa": F(kappa),
        "output_interval": ["0", "1"],
        "ridge_lambda": lam,
        "regularization_center": center,
        "regularization_center_rule": "exact_mean_of_original_training_targets",
        "original_target_digest": content_digest(original),
        "effective_target_digest": content_digest(effective),
        "effective_targets": effective,
        "objective_scale": scale,
        "objective_additive_constant": constant,
        "original_target_variance": variance,
        "objective_formula": "mean((v-y)^2)+ridge_lambda*mean((v-mean(y))^2)",
        "effective_target_formula": "(y+ridge_lambda*mean(y))/(1+ridge_lambda)",
        "regularized_objective": box(lower, upper),
        "regularized_gap_target": scale * F(core["gap_target_mean"]),
        "gap_resolved": core["gap_resolved"],
        "original_target_risks": risks,
        "selected_qp_source": core["selected_qp_source"],
        "selected_envelope_source": core["selected_envelope_source"],
        "optimizer_prediction_bound": prediction_bound,
        "core_fit": core,
        "core_fit_digest": digest(core),
        "input_binding": input_binding or {},
        "head_promotion_allowed": role == "train",
        "original_targets_clipped_in_loss": False,
        "regularized_lower_is_not_original_mse_lower": True,
        "native_transform_error_enclosed": False,
        "future_response_coverage_claim": False,
        "population_risk_claim": False,
        "timing": {
            "core_fit_seconds": core["timing"]["total_seconds"],
            "wrapper_total_seconds": time.perf_counter() - started,
        },
    })


def make_extension(fit, z, *, anchor_indices, recipe, extension_weight,
                   fit_path, fit_sha256):
    """Promote a training-only fit to one fixed weighted global extension."""
    if fit.get("schema") != FIT_SCHEMA or fit["role"] != "train" or not fit["head_promotion_allowed"]:
        raise ValueError("Only a verified-scope training witness can become a forecasting head")
    weight = F(extension_weight)
    lam = F(fit["ridge_lambda"])
    if recipe not in ("qp_regularized", "envelope_all") or not 0 <= weight <= 1:
        raise ValueError("Unknown recipe or extension weight outside [0,1]")
    if recipe == "envelope_all" and (lam != 0 or weight != F(1, 2)):
        raise ValueError("The envelope control is unregularized and uses the midpoint extension")
    z = np.asarray(z, dtype=np.float64)
    if content_digest(features_exact(z)) != fit["feature_digest"]:
        raise ValueError("Head geometry differs from fitted geometry")
    if len(anchor_indices) != len(z) or len(set(anchor_indices)) != len(z):
        raise ValueError("Invalid anchor columns")
    if fit_sha256 != digest(fit):
        raise ValueError("Training fit hash does not bind the supplied canonical witness")
    key = "selected_envelope_source" if recipe == "envelope_all" else "selected_qp_source"
    candidate = next(c for c in fit["core_fit"]["primal_candidates"] if c["source"] == fit[key])
    risks = fit["original_target_risks"][candidate["source"]]
    return canonical({
        "schema": HEAD_SCHEMA,
        "kind": "weighted_lipschitz_extension",
        "recipe": recipe,
        "feature_id": fit["feature_id"],
        "class_id": fit["class_id"],
        "dimension": fit["dimension"],
        "kappa": fit["kappa"],
        "output_interval": fit["output_interval"],
        "assurance": ASSURANCE,
        "ridge_lambda": lam,
        "extension_weight": weight,
        "anchor_ids": fit["ids"],
        "anchor_indices": list(map(int, anchor_indices)),
        "anchor_features": features_exact(z),
        "anchor_values": candidate["values"],
        "training_fit_path": fit_path,
        "training_fit_sha256": fit_sha256,
        "training_fit_role": "train",
        "selected_training_source": candidate["source"],
        "training_original_risk": risks["original_target_mse"],
        "training_regularized_objective": risks["regularized_objective"],
        "training_effective_risk": risks["effective_target_mse"],
        "optimizer_prediction_bound": fit["optimizer_prediction_bound"] if recipe == "qp_regularized" else None,
        "optimizer_prediction_bound_valid_for_this_head": recipe == "qp_regularized",
        "formula": "clip_I((1-extension_weight)*max_i(v_i-kappa*d(z,a_i))+extension_weight*min_i(v_i+kappa*d(z,a_i)))",
        "clipping_order": "after_weighted_raw_envelope_mixture",
        "future_response_coverage_claim": False,
        "native_transform_error_enclosed": False,
    })


def head_identity(head):
    return "head39:" + content_digest(head) if head.get("schema") == HEAD_SCHEMA else old.head_identity(head)


def evaluate_head(head, z, *, ids, distance=None):
    """Enclose one fixed weighted head without access to response labels."""
    if head.get("schema") != HEAD_SCHEMA:
        return old.evaluate_head(head, z, ids=ids, distance=distance)
    qz = features_exact(z)
    kap = F(head["kappa"])
    weight = F(head["extension_weight"])
    if head["kind"] != "weighted_lipschitz_extension" or not 0 <= weight <= 1:
        raise ValueError("Unknown head or invalid extension weight")
    if len(qz) != len(ids) or any(len(v) != head["dimension"] for v in qz):
        raise ValueError("Head/query roster mismatch")
    if head["class_id"] != old.ff.class_identity(head["feature_id"], kap, ("0", "1")):
        raise ValueError("Class identity mismatch")
    if distance is None or distance["feature_id"] != head["feature_id"] or distance["query_ids"] != list(ids):
        raise ValueError("Unbound query distances")
    if distance["query_feature_digest"] != content_digest(qz):
        raise ValueError("Query distance feature mismatch")
    lookup = {name: i for i, name in enumerate(distance["anchor_ids"])}
    columns = [lookup[name] for name in head["anchor_ids"]]
    if columns != head["anchor_indices"]:
        raise ValueError("Anchor columns mismatch")
    if columns == list(range(distance["n_anchor"])) and content_digest(head["anchor_features"]) != distance["anchor_feature_digest"]:
        raise ValueError("Anchor geometry mismatch")
    values = list(map(F, head["anchor_values"]))
    if len(values) != len(columns) or any(not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid anchor values")
    exact = set(distance["exact_pair_indices"])
    denominator = 1 << distance["bits"]
    rows = []
    for i, name in enumerate(ids):
        lower_lo = []
        lower_hi = []
        upper_lo = []
        upper_hi = []
        for j, value in zip(columns, values):
            index = i * distance["n_anchor"] + j
            floor = int(distance["floor_scaled"][index])
            dlo = F(floor, denominator)
            dhi = F(floor + (index not in exact), denominator)
            lower_lo.append(value - kap * dhi)
            lower_hi.append(value - kap * dlo)
            upper_lo.append(value + kap * dlo)
            upper_hi.append(value + kap * dhi)
        elo, ehi = max(lower_lo), max(lower_hi)
        ulo, uhi = min(upper_lo), min(upper_hi)
        rlo = (1 - weight) * elo + weight * ulo
        rhi = (1 - weight) * ehi + weight * uhi
        plo, phi = clip(rlo), clip(rhi)
        if not 0 <= plo <= phi <= 1 or phi - plo > kap / denominator:
            raise ArithmeticError("Invalid weighted prediction enclosure")
        rows.append({
            "query_id": name,
            "lower_envelope": box(elo, ehi),
            "upper_envelope": box(ulo, uhi),
            "raw_prediction": box(rlo, rhi),
            "prediction": box(plo, phi),
        })
    return canonical({
        "schema": PREDICTION_SCHEMA,
        "feature_id": head["feature_id"],
        "class_id": head["class_id"],
        "head_id": head_identity(head),
        "recipe": head["recipe"],
        "ridge_lambda": head["ridge_lambda"],
        "extension_weight": head["extension_weight"],
        "query_ids": list(ids),
        "query_feature_digest": content_digest(qz),
        "n": len(qz),
        "distance_digest": digest(distance),
        "anchor_columns": columns,
        "rows": rows,
        "predictions": [r["prediction"] for r in rows],
        "enclosure_kind": "fixed_fitted_head_arithmetic",
        "targets_accessed": False,
        "assurance": ASSURANCE,
        "future_response_coverage_claim": False,
        "native_transform_error_enclosed": False,
    })


def expand_optimizer_predictions(prediction, head, fit):
    """Enclose the same-weight extension of the exact regularized optimum.

    This deterministic expansion concerns the optimizer for this exact training
    problem.  It is not a prediction interval for an unknown future response.
    It is available only for the retained QP candidate whose objective is the
    upper endpoint used in the gap, not for the simpler envelope control.
    """
    if head.get("schema") != HEAD_SCHEMA or head.get("recipe") != "qp_regularized":
        raise ValueError("Optimizer expansion is only declared for QP heads")
    if fit.get("schema") != FIT_SCHEMA or fit["role"] != "train":
        raise ValueError("Optimizer expansion requires a training-only objective")
    if head["training_fit_sha256"] != digest(fit) or head["optimizer_prediction_bound"] != fit["optimizer_prediction_bound"]:
        raise ValueError("Optimizer expansion has an unbound fit or bound")
    if head["selected_training_source"] != fit["selected_qp_source"]:
        raise ValueError("Head values do not use the objective upper-bound candidate")
    if prediction.get("schema") != PREDICTION_SCHEMA or prediction["head_id"] != head_identity(head):
        raise ValueError("Optimizer expansion has an unbound source prediction")
    delta = F(fit["optimizer_prediction_bound"]["uniform_upper"])
    if delta < 0:
        raise ValueError("Negative optimizer discrepancy bound")
    intervals = []
    for interval in prediction["predictions"]:
        lower, upper = F(interval["lower"]), F(interval["upper"])
        if not 0 <= lower <= upper <= 1:
            raise ValueError("Source prediction interval is invalid")
        intervals.append(box(max(F(0), lower - delta), min(F(1), upper + delta)))
    return canonical({
        "schema": REFINEMENT_SCHEMA,
        "feature_id": head["feature_id"],
        "class_id": head["class_id"],
        "head_id": head_identity(head),
        "query_ids": prediction["query_ids"],
        "query_feature_digest": prediction["query_feature_digest"],
        "n": prediction["n"],
        "source_prediction_digest": digest(prediction),
        "training_fit_sha256": head["training_fit_sha256"],
        "optimizer_prediction_bound_digest": digest(fit["optimizer_prediction_bound"]),
        "optimizer_uniform_upper": delta,
        "predictions": intervals,
        "enclosure_kind": "same_weight_extension_of_unique_exact_regularized_training_optimum",
        "intersected_with_output_interval": ["0", "1"],
        "targets_accessed": False,
        "assurance": ASSURANCE,
        "future_response_coverage_claim": False,
        "native_transform_error_enclosed": False,
        "population_risk_claim": False,
    })
