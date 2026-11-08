"""Matched ridge heads for the Step 42 *diagnostic* factorial study.

The binary64 base solve has the same operation order as the inherited
``affine41.fit_affine``: center the stored feature matrix and rounded targets,
solve with n*lambda, and fit an unpenalized intercept.  Its coefficient and
intercept values are then retained as exact dyadic rationals.

Each base is shared by the eight projection/recentering/clipping variants:

* P=1 contracts a coefficient vector exceeding kappa using the inherited
  80-bit upward dyadic square-root rule.  P=0 leaves it unchanged even when
  its norm exceeds kappa; such a head does NOT inherit that norm guarantee.
* R=1 recomputes b=mean_exact(y)-beta_variant dot mean_exact(z).  R=0 retains
  the original binary64 fitted intercept, including after active projection.
  With inactive projection, R isolates rational-versus-binary64 recentering.
* C=1 clips normalized output to [0,1].  C=0 leaves it unbounded; that head
  does NOT inherit the original range guarantee.

The integer square-root construction below is the elementary construction
used in Step 41's frozen finite_feature35.sqrt_enclosure (SHA-256
7a6d9478bb03293eb42f133a61ac6a31471c3ce95fcf819304f54e3d5635fc18).
This module is self-contained and never imports or reads Step 41 files.
It reads no data, fits no representation, and asserts no training optimality,
native-transform enclosure, future-label coverage, or population guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any

import numpy as np


F = Fraction
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
BASE_SCHEMA = "step42-ridge-base-v1"
HEAD_SCHEMA = "step42-stored-feature-affine-head-v1"
VARIANT_SCHEMA = "step42-affine-variant-v1"


class HeadError(ValueError):
    """Invalid explicit diagnostic head or training input."""


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON for records already containing explicit rational strings."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeadError(message)


def _fraction(value: Any, name: str) -> Fraction:
    if isinstance(value, (bool, np.bool_)):
        raise HeadError(name + " must not be Boolean")
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        _require(math.isfinite(numeric), name + " must be finite")
        return F.from_float(numeric)
    if isinstance(value, np.integer):
        value = int(value)
    _require(isinstance(value, (F, str, int)), name + " must be an explicit rational")
    try:
        return F(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise HeadError(name + " is not a valid rational") from exc


def _finite_float(value: Fraction, name: str) -> float:
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise HeadError(name + " overflows binary64") from exc
    _require(math.isfinite(numeric), name + " is not finite in binary64")
    return numeric


def _matrix(value: Any) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise HeadError("Training features must be a binary64 matrix") from exc
    _require(matrix.ndim == 2 and min(matrix.shape) > 0,
             "Training features require positive row and feature counts")
    _require(bool(np.isfinite(matrix).all()), "Training features must be finite")
    return matrix


def _vector(value: Any, dimension: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise HeadError("Query must be a binary64 feature vector") from exc
    _require(vector.ndim == 1 and len(vector) == dimension,
             "Query dimension differs from the fitted head")
    _require(bool(np.isfinite(vector).all()), "Query features must be finite")
    return vector


def _sqrt_upper(value: Fraction, bits: int) -> Fraction:
    """Tight upward dyadic sqrt, derived entirely with integer arithmetic."""
    _require(type(bits) is int and 1 <= bits <= 512, "Invalid projection precision")
    _require(value >= 0, "Squared norm must be nonnegative")
    denominator = 1 << bits
    root = math.isqrt((value.numerator << (2 * bits)) // value.denominator)
    lower = F(root, denominator)
    upper = lower if lower * lower == value else F(root + 1, denominator)
    _require(lower * lower <= value <= upper * upper, "Invalid sqrt enclosure")
    return upper


def fit_base(z: Any, y: Any, *, feature_id: str,
             ridge_lambda: Any = "1/1000000",
             training_binding: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit one training-only base ridge head, shared by all eight variants.

    ``y`` holds exact normalized training targets.  Lambda is the coefficient
    of ||beta||^2 when the data term is MEAN squared error.  Intercept fitting
    is unpenalized.  The retained floating solution is not claimed to solve
    the corresponding exact rational normal equations exactly.
    """
    z = _matrix(z)
    _require(isinstance(feature_id, str) and bool(feature_id), "Feature ID is required")
    lam = _fraction(ridge_lambda, "Ridge lambda")
    _require(lam > 0, "Ridge lambda must be positive")
    float_lambda = _finite_float(lam, "Ridge lambda")
    _require(float_lambda > 0, "Ridge lambda underflows binary64")
    try:
        ys = [_fraction(v, "Training target") for v in y]
    except TypeError as exc:
        raise HeadError("Training targets must be an iterable roster") from exc
    _require(len(ys) == len(z), "Training target count differs from features")
    yf = np.asarray([_finite_float(v, "Training target") for v in ys])
    # Keep these operations in the same order as inherited affine41.fit_affine.
    xc = z - z.mean(axis=0)
    yc = yf - yf.mean()
    beta = np.linalg.solve(xc.T @ xc + len(z) * float_lambda * np.eye(z.shape[1]),
                           xc.T @ yc)
    intercept = float(yf.mean() - z.mean(axis=0) @ beta)
    _require(bool(np.isfinite(beta).all()) and math.isfinite(intercept),
             "Base solve produced a nonfinite coefficient or intercept")
    exact_beta = [F.from_float(float(v)) for v in beta]
    exact_intercept = F.from_float(intercept)
    center = [sum((F.from_float(float(v)) for v in z[:, j]), F(0)) / len(z)
              for j in range(z.shape[1])]
    binding = {} if training_binding is None else training_binding
    _require(isinstance(binding, dict), "Training binding must be a dictionary")
    # Copy JSON metadata so later caller mutations cannot alter the base record.
    binding = json.loads(canonical_bytes(binding))
    record = {
        "schema": BASE_SCHEMA,
        "feature_id": feature_id,
        "ridge_lambda": str(lam),
        "n": len(z),
        "dimension": int(z.shape[1]),
        "coefficients": [str(v) for v in exact_beta],
        "intercept": str(exact_intercept),
        "coefficient_squared_norm": str(sum((v * v for v in exact_beta), F(0))),
        "center_features": [str(v) for v in center],
        "center_target": str(sum(ys, F(0)) / len(ys)),
        "penalty_multiplier": str(len(z) * lam),
        "solver": "numpy.linalg.solve",
        "objective": "mean_squared_error_plus_lambda_coefficient_norm_squared",
        "intercept_penalty": "none",
        "centering": "binary64_unweighted_feature_and_rounded_target_means",
        "training_optimality_claim": False,
        "training_binding": binding,
    }
    record["base_id"] = "base42:" + digest(record)
    return record


def _check_base(base: dict[str, Any]) -> tuple[list[Fraction], Fraction,
                                             list[Fraction], Fraction]:
    _require(isinstance(base, dict) and base.get("schema") == BASE_SCHEMA,
             "Unknown base schema")
    body = {k: v for k, v in base.items() if k != "base_id"}
    _require(base.get("base_id") == "base42:" + digest(body), "Base identity differs")
    n, dimension = base.get("n"), base.get("dimension")
    _require(type(n) is int and n > 0 and type(dimension) is int and dimension > 0,
             "Invalid base shape")
    _require(isinstance(base.get("feature_id"), str) and bool(base["feature_id"]),
             "Invalid base feature ID")
    _require(isinstance(base.get("coefficients"), list)
             and len(base["coefficients"]) == dimension, "Invalid base coefficients")
    _require(isinstance(base.get("center_features"), list)
             and len(base["center_features"]) == dimension, "Invalid training center")
    beta = [_fraction(v, "Base coefficient") for v in base["coefficients"]]
    intercept = _fraction(base["intercept"], "Base intercept")
    center = [_fraction(v, "Feature center") for v in base["center_features"]]
    center_y = _fraction(base["center_target"], "Target center")
    _require(sum((v * v for v in beta), F(0))
             == _fraction(base["coefficient_squared_norm"], "Base squared norm"),
             "Base squared norm differs from its coefficients")
    lam = _fraction(base["ridge_lambda"], "Base lambda")
    _require(lam > 0 and _fraction(base["penalty_multiplier"], "Penalty multiplier") == n * lam,
             "Base penalty does not use n*lambda")
    return beta, intercept, center, center_y


def make_variant(base: dict[str, Any], *, project: bool, recenter: bool,
                 clip: bool, kappa: Any = "2", bits: int = 80) -> dict[str, Any]:
    """Apply the independent P/R/C switches without refitting the base head."""
    _require(all(type(v) is bool for v in (project, recenter, clip)),
             "All factorial switches must be Boolean")
    _require(type(bits) is int and 1 <= bits <= 512, "Invalid projection precision")
    original_beta, original_intercept, center, center_y = _check_base(base)
    kap = _fraction(kappa, "Kappa")
    _require(kap >= 0, "Kappa must be nonnegative")
    source_squared = sum((v * v for v in original_beta), F(0))
    factor = F(1)
    if project and source_squared > kap * kap:
        factor = kap / _sqrt_upper(source_squared, bits)
    beta = [factor * v for v in original_beta]
    intercept = (center_y - sum((v * x for v, x in zip(beta, center)), F(0))
                 if recenter else original_intercept)
    squared = sum((v * v for v in beta), F(0))
    if project:
        _require(squared <= kap * kap, "Projection failed the exact coefficient bound")
    center_prediction = intercept + sum((v * x for v, x in zip(beta, center)), F(0))
    head = {
        "schema": HEAD_SCHEMA,
        "kind": "clipped_affine" if clip else "affine",
        "feature_id": base["feature_id"],
        "dimension": base["dimension"],
        "coefficients": [str(v) for v in beta],
        "intercept": str(intercept),
        "clipping": clip,
        "output_interval": ["0", "1"] if clip else None,
        "assurance": ASSURANCE,
    }
    return {
        "schema": VARIANT_SCHEMA,
        "base_id": base["base_id"],
        "feature_id": base["feature_id"],
        "ridge_lambda": base["ridge_lambda"],
        "switches": {"projection": project, "recenter": recenter, "clipping": clip},
        "kappa": str(kap),
        "projection_bits": bits,
        "head": head,
        "head_id": "head42:" + digest(head),
        "transformation": {
            "source_squared_norm": str(source_squared),
            "coefficient_squared_norm": str(squared),
            "projection_factor": str(factor),
            "projection_active": factor != 1,
            "source_intercept": str(original_intercept),
            "recenter_delta": str(intercept - original_intercept),
            "training_mean_prediction_before_clipping": str(center_prediction),
            "center_target": str(center_y),
        },
        "structural_claims": {
            "norm_within_kappa": squared <= kap * kap,
            "normalized_range": ["0", "1"] if clip else None,
            "training_mean_affine_matches_target": center_prediction == center_y,
            "native_transform_error_enclosed": False,
            "population_risk_claim": False,
        },
    }


@dataclass(frozen=True)
class PreparedPrediction:
    """Cached runtime arithmetic; the JSON variant remains the persisted record."""
    head_id: str
    dimension: int
    coefficients: tuple[Fraction, ...]
    intercept: Fraction
    clipping: bool
    target_min: Fraction
    span: Fraction
    coefficient_squared_norm: Fraction
    norm_within_kappa: bool
    projection_active: bool
    recenter_delta: Fraction
    float_coefficients: np.ndarray
    float_intercept: float
    float_target_min: float
    float_span: float


def prepare_prediction(variant: dict[str, Any], *, target_min: Any,
                       span: Any) -> PreparedPrediction:
    """Validate a diagnostic record and cache its actual binary64 evaluation.

    This is the producer's runtime preparation, not an independent certificate
    verifier.  A separate study auditor checks the base/variant derivation.
    """
    _require(isinstance(variant, dict) and variant.get("schema") == VARIANT_SCHEMA,
             "Unknown variant schema")
    head = variant.get("head")
    _require(isinstance(head, dict) and head.get("schema") == HEAD_SCHEMA,
             "Unknown head schema")
    _require(variant.get("head_id") == "head42:" + digest(head), "Head identity differs")
    _require(head.get("assurance") == ASSURANCE, "Unsupported assurance scope")
    _require(head.get("feature_id") == variant.get("feature_id"), "Feature identities differ")
    dimension = head.get("dimension")
    _require(type(dimension) is int and dimension > 0, "Invalid head dimension")
    _require(isinstance(head.get("coefficients"), list)
             and len(head["coefficients"]) == dimension, "Head coefficient shape differs")
    clipping = head.get("clipping")
    _require(type(clipping) is bool, "Clipping flag must be Boolean")
    _require(head.get("kind") == ("clipped_affine" if clipping else "affine"),
             "Head kind differs from clipping flag")
    _require(head.get("output_interval") == (["0", "1"] if clipping else None),
             "Output range differs from clipping flag")
    switches = variant.get("switches", {})
    _require(set(switches) == {"projection", "recenter", "clipping"}
             and all(type(v) is bool for v in switches.values())
             and switches["clipping"] == clipping, "Variant switches differ")
    beta = tuple(_fraction(v, "Coefficient") for v in head["coefficients"])
    intercept = _fraction(head["intercept"], "Intercept")
    squared = sum((v * v for v in beta), F(0))
    kap = _fraction(variant["kappa"], "Kappa")
    _require(kap >= 0, "Kappa must be nonnegative")
    transformation = variant["transformation"]
    claims = variant["structural_claims"]
    _require(_fraction(transformation["coefficient_squared_norm"], "Squared norm") == squared,
             "Recorded squared norm differs from coefficients")
    _require(type(claims.get("norm_within_kappa")) is bool
             and claims["norm_within_kappa"] == (squared <= kap * kap),
             "Incorrect coefficient-norm claim")
    _require(not switches["projection"] or squared <= kap * kap,
             "Projected head exceeds kappa")
    _require(claims.get("normalized_range") == head["output_interval"],
             "Incorrect range claim")
    _require(claims.get("native_transform_error_enclosed") is False
             and claims.get("population_risk_claim") is False,
             "Unsupported structural claim")
    factor = _fraction(transformation["projection_factor"], "Projection factor")
    _require(0 <= factor <= 1, "Projection factor is not a contraction")
    _require(type(transformation["projection_active"]) is bool
             and transformation["projection_active"] == (factor != 1),
             "Projection activity differs from factor")
    _require(switches["projection"] or factor == 1, "Disabled projection altered coefficients")
    _require(squared == factor * factor * _fraction(transformation["source_squared_norm"],
                                                    "Source squared norm"),
             "Projection norm accounting differs")
    source_intercept = _fraction(transformation["source_intercept"], "Source intercept")
    recenter_delta = _fraction(transformation["recenter_delta"], "Recenter delta")
    _require(intercept - source_intercept == recenter_delta, "Recenter delta differs")
    _require(switches["recenter"] or recenter_delta == 0, "Disabled recentering changed intercept")
    center_pred = _fraction(transformation["training_mean_prediction_before_clipping"], "Mean prediction")
    center_target = _fraction(transformation["center_target"], "Mean target")
    _require(type(claims.get("training_mean_affine_matches_target")) is bool
             and claims["training_mean_affine_matches_target"] == (center_pred == center_target),
             "Incorrect recentering claim")
    _require(not switches["recenter"] or center_pred == center_target,
             "Recentered head does not preserve its declared training mean")
    minimum, scale = _fraction(target_min, "Target minimum"), _fraction(span, "Target span")
    _require(scale > 0, "Target span must be positive")
    beta_float = np.asarray([_finite_float(v, "Coefficient") for v in beta], dtype=np.float64)
    beta_float.setflags(write=False)
    float_span = _finite_float(scale, "Target span")
    _require(float_span > 0, "Target span underflows binary64")
    return PreparedPrediction(
        head_id=variant["head_id"], dimension=dimension, coefficients=beta,
        intercept=intercept, clipping=clipping, target_min=minimum, span=scale,
        coefficient_squared_norm=squared, norm_within_kappa=squared <= kap * kap,
        projection_active=transformation["projection_active"], recenter_delta=recenter_delta,
        float_coefficients=beta_float, float_intercept=_finite_float(intercept, "Intercept"),
        float_target_min=_finite_float(minimum, "Target minimum"), float_span=float_span,
    )


def _float_components(prepared: PreparedPrediction, features_binary64: Any
                      ) -> tuple[float, float, float]:
    vector = _vector(features_binary64, prepared.dimension)
    # This one-query np.dot/add/clip/inverse order is byte-identical to Step 41.
    raw = float(np.dot(prepared.float_coefficients, vector)) + prepared.float_intercept
    _require(math.isfinite(raw), "Affine intermediate is nonfinite")
    normalized = min(1.0, max(0.0, raw)) if prepared.clipping else raw
    physical = prepared.float_target_min + prepared.float_span * normalized
    _require(math.isfinite(physical), "Physical prediction is nonfinite")
    return raw, normalized, float(physical)


def predict_float(prepared: PreparedPrediction, features_binary64: Any) -> float:
    return _float_components(prepared, features_binary64)[2]


def predict_details(prepared: PreparedPrediction, features_binary64: Any) -> dict[str, Any]:
    """JSON record of the actual operational path and clipping activation."""
    raw, normalized, physical = _float_components(prepared, features_binary64)
    active = prepared.clipping and normalized != raw
    return {
        "raw_normalized_hex": raw.hex(),
        "normalized_hex": normalized.hex(),
        "reported_physical_hex": physical.hex(),
        "clipping_active": active,
        "clipping_side": ("below" if raw < 0 else "above") if active else "none",
        "raw_outside_unit_interval": raw < 0 or raw > 1,
    }


def _exact_components(prepared: PreparedPrediction, features_binary64: Any
                      ) -> tuple[Fraction, Fraction, Fraction]:
    vector = _vector(features_binary64, prepared.dimension)
    raw = prepared.intercept + sum(
        (v * F.from_float(float(x)) for v, x in zip(prepared.coefficients, vector)), F(0))
    normalized = min(F(1), max(F(0), raw)) if prepared.clipping else raw
    return raw, normalized, prepared.target_min + prepared.span * normalized


def exact_prediction(prepared: PreparedPrediction, features_binary64: Any) -> Fraction:
    return _exact_components(prepared, features_binary64)[2]


def exact_details(prepared: PreparedPrediction, features_binary64: Any) -> dict[str, Any]:
    raw, normalized, physical = _exact_components(prepared, features_binary64)
    active = prepared.clipping and normalized != raw
    return {
        "raw_normalized": str(raw),
        "normalized": str(normalized),
        "exact_physical": str(physical),
        "clipping_active": active,
        "clipping_side": ("below" if raw < 0 else "above") if active else "none",
        "raw_outside_unit_interval": raw < 0 or raw > 1,
    }


def as_legacy_head(variant: dict[str, Any]) -> dict[str, Any]:
    """Bridge an actually bounded clipped head to the unchanged Step 41 checker.

    The old class is exactly kappa=2 and normalized range [0,1].  Merely
    selecting a projection switch does not authorize that claim: the exact
    retained coefficient norm is checked here, and unbounded outputs reject.
    The original identity convention omits the trailing JSON newline used by
    Step 42.  No new guarantee is attached to a head outside the old class.
    """
    prepared = prepare_prediction(variant, target_min="0", span="1")
    _require(prepared.clipping, "Legacy bridge requires [0,1] clipping")
    _require(prepared.coefficient_squared_norm <= 4,
             "Legacy bridge requires exact coefficient norm at most 2")
    feature_id = variant["head"]["feature_id"]
    descriptor = {
        "schema": "step35-stored-feature-decoder-class-v1",
        "feature_id": feature_id,
        "metric": "Euclidean_distance_on_supplied_feature_coordinates",
        "decoder_class": "all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space",
        "kappa": "2",
        "output_interval": ["0", "1"],
        "assurance": ASSURANCE,
    }
    identity_bytes = json.dumps(descriptor, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True, allow_nan=False).encode()
    return {
        "schema": "step35-stored-feature-head-v1",
        "kind": "clipped_affine",
        "feature_id": feature_id,
        "class_id": "class35:" + hashlib.sha256(identity_bytes).hexdigest(),
        "dimension": prepared.dimension,
        "kappa": "2",
        "output_interval": ["0", "1"],
        "assurance": ASSURANCE,
        "coefficients": [str(v) for v in prepared.coefficients],
        "intercept": str(prepared.intercept),
    }
