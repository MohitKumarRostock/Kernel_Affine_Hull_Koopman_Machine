"""Compact arithmetic certificates for one fixed clipped affine predictor.

The reference is the inherited *rational* coefficient payload applied to the
supplied binary64 feature vector, followed by exact clipping and inverse target
normalization.  The fast deployment path evaluates rounded binary64 parameters.
It is accepted only when its independently checked absolute discrepancy is at
most ``span * 2**-40``.  No training optimum or ideal feature-transform value is
certified.  No response labels, data loaders, fitting routines or files are used.

The producer computes with Fraction operations.  The per-query verifier instead
uses an integer dot product with a common coefficient denominator and a common
binary exponent for the query.  Static validation is separately amortizable.
The checker has the trusted deployment, expected query ID, feature vector and
actual reported value as inputs; a packet is not its own trust anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
import struct
from typing import Any

import numpy as np


DEPLOYMENT_SCHEMA = "step41-affine-deployment-v1"
CERTIFICATE_SCHEMA = "step41-affine-query-certificate-v1"
VERIFICATION_SCHEMA = "step41-affine-query-verification-v1"
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
NORMALIZED_ERROR_BUDGET = "1/1099511627776"  # exactly 2**-40
SCOPE = {
    "assurance": ASSURANCE,
    "reference": "FIXED_STORED_RATIONAL_AFFINE_HEAD",
    "affine_training_optimum_certified": False,
    "native_transform_error_enclosed": False,
    "raw_input_robustness_certified": False,
    "future_response_coverage_claim": False,
    "population_risk_claim": False,
    "cryptographic_authenticity_claim": False,
}
_HEAD_KEYS = {
    "schema", "kind", "feature_id", "class_id", "dimension", "kappa",
    "output_interval", "assurance", "coefficients", "intercept",
}
_DEPLOYMENT_KEYS = {
    "schema", "head", "head_id", "candidate_id", "target_min", "span",
    "map_sha256", "map_bytes", "normalized_error_budget", "scope",
    "deployment_id",
}
_CERTIFICATE_KEYS = {
    "schema", "deployment_id", "query_id", "feature_sha256",
    "reported_prediction_hex", "reference_interval_hex", "absolute_error_upper_hex",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CertificateError(ValueError):
    """A static premise, binding, arithmetic claim or accuracy budget failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificateError(message)


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes; objects here contain explicit rational strings."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _keys(value: Any, expected: set[str], name: str) -> None:
    _require(type(value) is dict and set(value) == expected,
             name + " fields differ from the frozen schema")


def _text(value: Any, name: str) -> str:
    _require(type(value) is str and bool(value), name + " must be nonempty text")
    return value


def _fraction(value: Any, name: str) -> Fraction:
    _require(type(value) is str, name + " must be an explicit rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise CertificateError(name + " is not a rational") from error
    _require(str(result) == value, name + " is not in canonical rational form")
    return result


def _binary64(value: Any, name: str) -> float:
    _require(type(value) is float or isinstance(value, np.float64),
             name + " must be binary64, not a decimal string or integer")
    value = float(value)
    _require(math.isfinite(value), name + " must be finite")
    return value


def _hex_float(value: Any, name: str) -> float:
    _require(type(value) is str, name + " must be a canonical binary64 hex string")
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError) as error:
        raise CertificateError(name + " is not binary64 hex") from error
    _require(math.isfinite(number) and number.hex() == value,
             name + " must be finite canonical binary64 hex")
    return number


def _features(features: Any, dimension: int | None = None) -> np.ndarray:
    if isinstance(features, np.ndarray):
        _require(features.dtype.kind == "f" and features.dtype.itemsize == 8,
                 "Features must be stored binary64 values")
        vector = np.asarray(features, dtype=np.float64)
    else:
        _require(isinstance(features, (list, tuple)), "Features must be a binary64 vector")
        _require(all(type(x) is float or isinstance(x, np.float64) for x in features),
                 "Feature entries must be binary64 values")
        vector = np.asarray(features, dtype=np.float64)
    _require(vector.ndim == 1 and len(vector) > 0, "Features must be a nonempty vector")
    _require(dimension is None or len(vector) == dimension, "Feature dimension differs")
    _require(bool(np.isfinite(vector).all()), "Nonfinite feature value")
    return vector


def feature_payload_bytes(features_binary64: Any) -> bytes:
    """Bound query representation: a dimension header and big-endian doubles.

    The fixed format tag and 32-bit dimension add 29 bytes to the 8*d coordinate
    bytes.  The actual length, rather than this explanatory count, is reported.
    Signed zeros retain different byte identities, although their rational
    meanings are equal.
    """
    vector = _features(features_binary64)
    return (b"step41-feature-vector-v1\0" + struct.pack(">I", len(vector))
            + vector.astype(">f8", copy=False).tobytes(order="C"))


def feature_digest(features_binary64: Any) -> str:
    return hashlib.sha256(feature_payload_bytes(features_binary64)).hexdigest()


def _class_id(head: dict[str, Any]) -> str:
    descriptor = {
        "schema": "step35-stored-feature-decoder-class-v1",
        "feature_id": head["feature_id"],
        "metric": "Euclidean_distance_on_supplied_feature_coordinates",
        "decoder_class": "all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space",
        "kappa": head["kappa"],
        "output_interval": head["output_interval"],
        "assurance": ASSURANCE,
    }
    return "class35:" + _digest(descriptor)


@dataclass(frozen=True)
class PreparedDeployment:
    """Trusted immutable arithmetic context established by static validation."""

    deployment_id: str
    candidate_id: str
    feature_id: str
    head_id: str
    dimension: int
    coefficients: tuple[Fraction, ...]
    intercept: Fraction
    target_min: Fraction
    span: Fraction
    normalized_error_budget: Fraction
    coefficient_denominator: int
    integer_coefficients: tuple[int, ...]
    integer_intercept: int
    float_coefficients: np.ndarray
    float_intercept: float
    float_target_min: float
    float_span: float
    deployment_bytes: bytes
    map_sha256: str
    map_bytes: int


def make_deployment(head: dict[str, Any], *, target_min: Any, span: Any,
                    map_sha256: str, map_bytes: int, candidate_id: str) -> dict[str, Any]:
    """Bind a fitted inherited head and normalization to a map's byte identity."""
    # JSON round-trip prevents sharing mutable nested dictionaries with fitting.
    head_copy = json.loads(canonical_bytes(head))
    payload = {
        "schema": DEPLOYMENT_SCHEMA,
        "head": head_copy,
        "head_id": "head35:" + _digest(head_copy),
        "candidate_id": candidate_id,
        "target_min": str(Fraction(target_min)),
        "span": str(Fraction(span)),
        "map_sha256": map_sha256,
        "map_bytes": map_bytes,
        "normalized_error_budget": NORMALIZED_ERROR_BUDGET,
        "scope": dict(SCOPE),
    }
    payload["deployment_id"] = "deploy41:" + _digest(payload)
    prepare_deployment(payload)
    return payload


def prepare_deployment(deployment: dict[str, Any], *, expected_deployment_id: str | None = None,
                       expected_map_sha256: str | None = None) -> PreparedDeployment:
    """Check static identity and head membership, then cache arithmetic inputs.

    When expected IDs are omitted, the caller is explicitly establishing the
    supplied deployment as its trusted context.  Map bytes and the native map
    execution are checked separately by the caller; a hash is not authenticity.
    """
    _keys(deployment, _DEPLOYMENT_KEYS, "Deployment")
    _require(deployment["schema"] == DEPLOYMENT_SCHEMA, "Unknown deployment schema")
    _require(deployment["scope"] == SCOPE, "Unsupported assurance scope")
    body = {key: value for key, value in deployment.items() if key != "deployment_id"}
    deployment_id = "deploy41:" + _digest(body)
    _require(deployment["deployment_id"] == deployment_id, "Deployment content identity differs")
    _require(expected_deployment_id is None or deployment_id == expected_deployment_id,
             "Deployment differs from the trusted expected identity")
    _require(type(deployment["map_sha256"]) is str
             and _SHA256.fullmatch(deployment["map_sha256"]) is not None,
             "Invalid map SHA-256")
    _require(expected_map_sha256 is None or deployment["map_sha256"] == expected_map_sha256,
             "Map differs from the trusted expected identity")
    _require(type(deployment["map_bytes"]) is int and deployment["map_bytes"] > 0,
             "Map byte count must be positive")
    candidate_id = _text(deployment["candidate_id"], "Candidate ID")
    _require(deployment["normalized_error_budget"] == NORMALIZED_ERROR_BUDGET,
             "Normalized discrepancy budget differs from the frozen value")
    budget = _fraction(deployment["normalized_error_budget"], "Normalized error budget")
    head = deployment["head"]
    _keys(head, _HEAD_KEYS, "Head")
    _require(head["schema"] == "step35-stored-feature-head-v1"
             and head["kind"] == "clipped_affine" and head["assurance"] == ASSURANCE,
             "Unsupported affine head or assurance")
    dimension = head["dimension"]
    _require(type(dimension) is int and 1 <= dimension <= 4096, "Invalid head dimension")
    _text(head["feature_id"], "Feature identity")
    _require(type(head["coefficients"]) is list and len(head["coefficients"]) == dimension,
             "Coefficient dimension differs")
    coefficients = tuple(_fraction(x, "Coefficient") for x in head["coefficients"])
    intercept = _fraction(head["intercept"], "Intercept")
    kappa = _fraction(head["kappa"], "Sensitivity")
    _require(kappa >= 0, "Negative sensitivity")
    _require(head["output_interval"] == ["0", "1"], "Normalized clipping range differs")
    _require(head["class_id"] == _class_id(head), "Class identity differs")
    head_id = "head35:" + _digest(head)
    _require(deployment["head_id"] == head_id, "Head content identity differs")
    _require(sum((x*x for x in coefficients), Fraction(0)) <= kappa*kappa,
             "Exact coefficient norm exceeds the head sensitivity")
    target_min = _fraction(deployment["target_min"], "Target minimum")
    span = _fraction(deployment["span"], "Target span")
    _require(span > 0, "Target span must be positive")
    try:
        numeric = [float(x) for x in (*coefficients, intercept, target_min, span, target_min + span)]
    except OverflowError as error:
        raise CertificateError("Static values do not admit the declared binary64 path") from error
    _require(all(math.isfinite(x) for x in numeric), "Nonfinite binary64 deployment parameter")
    _require(float(span) > 0, "Target span underflows the declared binary64 path")
    denominator = math.lcm(*(x.denominator for x in (*coefficients, intercept)))
    integer_coefficients = tuple(x.numerator * (denominator // x.denominator) for x in coefficients)
    integer_intercept = intercept.numerator * (denominator // intercept.denominator)
    float_coefficients = np.asarray([float(x) for x in coefficients], dtype=np.float64)
    float_coefficients.setflags(write=False)
    return PreparedDeployment(
        deployment_id=deployment_id, candidate_id=candidate_id, feature_id=head["feature_id"],
        head_id=head_id, dimension=dimension, coefficients=coefficients, intercept=intercept,
        target_min=target_min, span=span, normalized_error_budget=budget,
        coefficient_denominator=denominator, integer_coefficients=integer_coefficients,
        integer_intercept=integer_intercept, float_coefficients=float_coefficients,
        float_intercept=float(intercept), float_target_min=float(target_min), float_span=float(span),
        deployment_bytes=canonical_bytes(deployment), map_sha256=deployment["map_sha256"],
        map_bytes=deployment["map_bytes"],
    )


def predict_float(prepared: PreparedDeployment, features_binary64: Any) -> float:
    """Actual one-query binary64 deployment formula, in physical target units."""
    vector = _features(features_binary64, prepared.dimension)
    raw = float(np.dot(prepared.float_coefficients, vector)) + prepared.float_intercept
    _require(math.isfinite(raw), "Fast affine intermediate is nonfinite")
    normalized = min(1.0, max(0.0, raw))
    physical = prepared.float_target_min + prepared.float_span * normalized
    _require(math.isfinite(physical), "Fast physical prediction is nonfinite")
    return float(physical)


def exact_prediction(prepared: PreparedDeployment, features_binary64: Any) -> Fraction:
    """Producer reference evaluation; independent verification does not call it."""
    vector = _features(features_binary64, prepared.dimension)
    raw = prepared.intercept + sum(
        (b * Fraction.from_float(float(x)) for b, x in zip(prepared.coefficients, vector)),
        Fraction(0),
    )
    return prepared.target_min + prepared.span * min(Fraction(1), max(Fraction(0), raw))


def _outward_hull(value: Fraction) -> tuple[float, float]:
    rounded = float(value)
    _require(math.isfinite(rounded), "Reference is outside finite binary64 range")
    exact_rounded = Fraction.from_float(rounded)
    if exact_rounded == value:
        return rounded, rounded
    if exact_rounded < value:
        return rounded, math.nextafter(rounded, math.inf)
    return math.nextafter(rounded, -math.inf), rounded


def produce_certificate(prepared: PreparedDeployment, features_binary64: Any,
                        reported_prediction: float, *, query_id: str) -> dict[str, Any]:
    """Construct a compact packet; no target label is an input."""
    _text(query_id, "Query ID")
    reported = _binary64(reported_prediction, "Reported prediction")
    vector = _features(features_binary64, prepared.dimension)
    reference = exact_prediction(prepared, vector)
    lower, upper = _outward_hull(reference)
    discrepancy = abs(Fraction.from_float(reported) - reference)
    _, error_upper = _outward_hull(discrepancy)
    _require(math.isfinite(lower) and math.isfinite(upper) and math.isfinite(error_upper),
             "Certificate enclosure is not finite")
    # Producing the packet does not itself assert that its error is acceptable.
    # The independent verifier enforces the frozen budget.
    return {
        "schema": CERTIFICATE_SCHEMA,
        "deployment_id": prepared.deployment_id,
        "query_id": query_id,
        "feature_sha256": feature_digest(vector),
        "reported_prediction_hex": reported.hex(),
        "reference_interval_hex": [lower.hex(), upper.hex()],
        "absolute_error_upper_hex": error_upper.hex(),
    }


def _integer_reference(prepared: PreparedDeployment, vector: np.ndarray) -> Fraction:
    """Independent integer accumulator for the fixed rational affine formula."""
    ratios = [float(x).as_integer_ratio() for x in vector]
    exponents = [den.bit_length() - 1 for _, den in ratios]
    _require(all(den == 1 << exponent for (_, den), exponent in zip(ratios, exponents)),
             "A binary64 coordinate was not dyadic")
    exponent = max(exponents)
    query_integers = [num << (exponent - power)
                      for (num, _), power in zip(ratios, exponents)]
    numerator = (prepared.integer_intercept << exponent)
    for coefficient, coordinate in zip(prepared.integer_coefficients, query_integers):
        numerator += coefficient * coordinate
    denominator = prepared.coefficient_denominator << exponent
    if numerator <= 0:
        normalized = Fraction(0)
    elif numerator >= denominator:
        normalized = Fraction(1)
    else:
        normalized = Fraction(numerator, denominator)
    return prepared.target_min + prepared.span * normalized


def _verify_tight_interval(value: Fraction, lower: float, upper: float, name: str) -> None:
    """Check exact containment and minimality without calling producer rounding."""
    lo, hi = Fraction.from_float(lower), Fraction.from_float(upper)
    _require(lo <= value <= hi, name + " does not enclose the exact value")
    if lo == hi:
        _require(lo == value and lower.hex() == upper.hex(), name + " is not an exact singleton")
        _require(value != 0 or lower.hex() == 0.0.hex(), name + " has noncanonical zero")
    else:
        _require(lo < value < hi, name + " is not the smallest binary64 enclosure")
        _require(math.nextafter(lower, math.inf).hex() == upper.hex(),
                 name + " endpoints are not adjacent binary64 values")


def verify_certificate(prepared: PreparedDeployment, features_binary64: Any,
                       certificate: dict[str, Any], *, expected_query_id: str,
                       reported_prediction: float) -> dict[str, Any]:
    """Verify bindings, exact arithmetic, tight intervals and the error budget.

    The expected query ID and actual reported physical prediction are mandatory
    external inputs.  A self-consistent packet describing an inaccurate value
    is rejected when its discrepancy exceeds the frozen normalized budget.
    """
    _keys(certificate, _CERTIFICATE_KEYS, "Certificate")
    _require(certificate["schema"] == CERTIFICATE_SCHEMA, "Unknown certificate schema")
    _require(certificate["deployment_id"] == prepared.deployment_id, "Certificate deployment differs")
    _text(expected_query_id, "Expected query ID")
    _require(certificate["query_id"] == expected_query_id, "Certificate query ID differs")
    vector = _features(features_binary64, prepared.dimension)
    _require(certificate["feature_sha256"] == feature_digest(vector), "Certificate feature bytes differ")
    reported = _binary64(reported_prediction, "Actual reported prediction")
    retained = _hex_float(certificate["reported_prediction_hex"], "Retained reported prediction")
    _require(retained.hex() == reported.hex(), "Certificate does not bind the actual reported prediction")
    interval = certificate["reference_interval_hex"]
    _require(type(interval) is list and len(interval) == 2, "Reference enclosure must have two endpoints")
    lower = _hex_float(interval[0], "Reference lower endpoint")
    upper = _hex_float(interval[1], "Reference upper endpoint")
    error_upper = _hex_float(certificate["absolute_error_upper_hex"], "Absolute-error upper bound")
    reference = _integer_reference(prepared, vector)
    _verify_tight_interval(reference, lower, upper, "Reference interval")
    error = abs(Fraction.from_float(reported) - reference)
    upper_q = Fraction.from_float(error_upper)
    _require(error <= upper_q and upper_q >= 0, "Absolute-error upper bound is false")
    _require((error == 0 and error_upper.hex() == 0.0.hex())
             or (error > 0 and (Fraction.from_float(math.nextafter(error_upper, -math.inf)) < error)),
             "Absolute-error bound is not the tight upward binary64 rounding")
    physical_budget = prepared.span * prepared.normalized_error_budget
    _require(error <= physical_budget, "Reported prediction exceeds the frozen discrepancy budget")
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "PASS",
        "deployment_id": prepared.deployment_id,
        "query_id": expected_query_id,
        "feature_sha256": certificate["feature_sha256"],
        "exact_reference_physical": str(reference),
        "exact_reported_physical": str(Fraction.from_float(reported)),
        "absolute_error_exact": str(error),
        "normalized_error_exact": str(error / prepared.span),
        "physical_error_budget": str(physical_budget),
        "normalized_error_budget": NORMALIZED_ERROR_BUDGET,
        "verified_exact_error": True,
        "accepted_prediction_accuracy": True,
        "target_label_accessed": False,
        "scope": dict(SCOPE),
    }


def encode_certificate(certificate: dict[str, Any]) -> bytes:
    """Canonical UTF-8 packet bytes; no newline, model or query bytes included."""
    return canonical_bytes(certificate)
