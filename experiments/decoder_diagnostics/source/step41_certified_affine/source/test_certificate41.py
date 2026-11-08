"""Artificial-only acceptance, boundary and adversarial checks for Step 41.

No real cohort, model fit or assessment outcome is loaded.  The independent
study auditor additionally rederives these claims from the retained payloads.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

import certificate41 as cert


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def make_head(coefficients, intercept="1/2", feature_id="artificial:step41"):
    head = {
        "schema": "step35-stored-feature-head-v1", "kind": "clipped_affine",
        "feature_id": feature_id, "class_id": "", "dimension": len(coefficients),
        "kappa": "2", "output_interval": ["0", "1"], "assurance": cert.ASSURANCE,
        "coefficients": list(map(lambda x: str(F(x)), coefficients)), "intercept": str(F(intercept)),
    }
    descriptor = {
        "schema": "step35-stored-feature-decoder-class-v1", "feature_id": feature_id,
        "metric": "Euclidean_distance_on_supplied_feature_coordinates",
        "decoder_class": "all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space",
        "kappa": "2", "output_interval": ["0", "1"], "assurance": cert.ASSURANCE,
    }
    head["class_id"] = "class35:" + canonical_hash(descriptor)
    return head


def deployment(coefficients, intercept="1/2", target_min="-10", span="30"):
    return cert.make_deployment(make_head(coefficients, intercept), target_min=target_min,
                                span=span, map_sha256="a" * 64, map_bytes=123,
                                candidate_id="artificial::affine")


def rebind(value):
    value["head_id"] = "head35:" + canonical_hash(value["head"])
    value["deployment_id"] = "deploy41:" + canonical_hash(
        {key: item for key, item in value.items() if key != "deployment_id"})
    return value


def independent_reference(payload, x):
    head = payload["head"]
    raw = F(head["intercept"])
    for b, value in zip(head["coefficients"], x):
        numerator, denominator = float(value).as_integer_ratio()
        raw += F(b) * F(numerator, denominator)
    return F(payload["target_min"]) + F(payload["span"]) * max(F(0), min(F(1), raw))


def run_checks():
    checks = []

    def passed(name, details=None):
        checks.append({"name": name, "status": "PASS", **({"details": details} if details else {})})

    def rejects(name, action, reason=None):
        try:
            action()
        except cert.CertificateError as error:
            if reason is not None:
                assert reason in str(error), (name, str(error), reason)
            passed(name, {"rejection": str(error)})
        else:
            raise AssertionError("Expected rejection: " + name)

    # Explicit clipping and rounding boundaries, including subnormal coordinates.
    fixtures = [
        (["1"], "0", [0.0], "0", "1"),
        (["1"], "0", [1.0], "0", "1"),
        (["1"], "0", [-3.0], "-20", "40"),
        (["1"], "0", [3.0], "-20", "40"),
        (["1/3", "-1/5", "1/7"], "1/2", [0.1, -0.3, 0.7], "-20", "40"),
        (["1"], "0", [math.ulp(0.0)], "0", "1"),
        (["1"], "0", [-math.ulp(0.0)], "0", "1"),
        (["0"], "1/3", [-0.0], "-1/3", "1"),
        (["0"], "1/7", [1.0], "-100/7", "100"),
        (["1", "-1"], "1/2", [float(2**50), float(2**50)], "-10", "30"),
    ]
    for i, (coefficients, intercept, x, offset, span) in enumerate(fixtures):
        payload = deployment(coefficients, intercept, offset, span)
        prepared = cert.prepare_deployment(payload, expected_deployment_id=payload["deployment_id"],
                                            expected_map_sha256="a" * 64)
        reported = cert.predict_float(prepared, x)
        packet = cert.produce_certificate(prepared, x, reported, query_id=f"boundary:{i}")
        proof = cert.verify_certificate(prepared, x, packet, expected_query_id=f"boundary:{i}",
                                         reported_prediction=reported)
        reference = independent_reference(payload, x)
        assert F(proof["exact_reference_physical"]) == reference
        assert F(proof["absolute_error_exact"]) == abs(F.from_float(reported) - reference)
        assert proof["accepted_prediction_accuracy"] is True
        passed(f"boundary_fixture_{i}")

    generator = random.Random(4101)
    dimensions = (1, 3, 12, 24, 48, 96)
    for i in range(72):
        dimension = dimensions[i % len(dimensions)]
        coefficients = [str(F(generator.randrange(-8, 9), 128)) for _ in range(dimension)]
        x = [generator.uniform(-2.0, 2.0) for _ in range(dimension)]
        payload = deployment(coefficients, "1/3", "-71/3", "157/3")
        prepared = cert.prepare_deployment(payload)
        reported = cert.predict_float(prepared, x)
        packet = cert.produce_certificate(prepared, x, reported, query_id=f"random:{i}")
        proof = cert.verify_certificate(prepared, x, packet, expected_query_id=f"random:{i}",
                                         reported_prediction=reported)
        assert F(proof["exact_reference_physical"]) == independent_reference(payload, x)
        assert cert.encode_certificate(packet) == json.dumps(
            packet, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        passed(f"random_fraction_vs_integer_{i}", {"dimension": dimension})

    payload = deployment(["1/3", "-1/5", "1/7"])
    prepared = cert.prepare_deployment(payload)
    x = [0.1, -0.3, 0.7]
    reported = cert.predict_float(prepared, x)
    packet = cert.produce_certificate(prepared, x, reported, query_id="q")

    def verify(changed, vector=x, query_id="q", y=reported):
        return cert.verify_certificate(prepared, vector, changed, expected_query_id=query_id,
                                         reported_prediction=y)

    modifications = [
        ("unknown_certificate_schema", "schema", "unrecognized", "schema"),
        ("different_deployment", "deployment_id", "deploy41:" + "0"*64, "deployment"),
        ("different_query_id", "query_id", "not:q", "query ID"),
        ("different_feature_hash", "feature_sha256", "0"*64, "feature bytes"),
        ("different_reported_value", "reported_prediction_hex", float(reported + 1).hex(), "reported prediction"),
        ("nan_reported_value", "reported_prediction_hex", "nan", "finite"),
        ("infinite_reported_value", "reported_prediction_hex", "inf", "finite"),
        ("decimal_reported_value", "reported_prediction_hex", "0.5", "canonical"),
        ("error_upper_too_small", "absolute_error_upper_hex", "0x0.0p+0", "upper bound"),
        ("error_upper_loose", "absolute_error_upper_hex", float(1).hex(), "tight upward"),
        ("error_upper_negative", "absolute_error_upper_hex", float(-1).hex(), "upper bound"),
        ("reference_reversed", "reference_interval_hex", list(reversed(packet["reference_interval_hex"])), "enclose"),
        ("reference_loose", "reference_interval_hex", [float(-1000).hex(), float(1000).hex()], "adjacent"),
        ("reference_singleton_wrong", "reference_interval_hex", [float(0).hex(), float(0).hex()], "enclose"),
        ("reference_one_endpoint", "reference_interval_hex", [float(0).hex()], "two endpoints"),
        ("reference_infinite", "reference_interval_hex", [float(0).hex(), "inf"], "finite"),
    ]
    for name, key, value, reason in modifications:
        altered = deepcopy(packet)
        altered[key] = value
        rejects(name, lambda altered=altered: verify(altered), reason)
    altered = deepcopy(packet); altered["future_response_coverage_claim"] = True
    rejects("injected_unsupported_claim", lambda: verify(altered), "fields")
    altered = deepcopy(packet); del altered["feature_sha256"]
    rejects("missing_query_binding", lambda: verify(altered), "fields")
    rejects("external_expected_query_changed", lambda: verify(packet, query_id="elsewhere"), "query ID")
    rejects("external_actual_report_changed", lambda: verify(packet, y=math.nextafter(reported, math.inf)), "reported prediction")
    changed_x = list(x); changed_x[0] = math.nextafter(changed_x[0], math.inf)
    rejects("external_query_one_ulp_changed", lambda: verify(packet, vector=changed_x), "feature bytes")
    rejects("query_dimension_wrong", lambda: verify(packet, vector=x[:2]), "dimension")
    rejects("query_nonfinite", lambda: verify(packet, vector=[math.nan, 0.0, 0.0]), "Nonfinite")
    rejects("query_float32", lambda: verify(packet, vector=np.asarray(x, dtype=np.float32)), "binary64")
    rejects("query_decimal_strings", lambda: verify(packet, vector=["0.1", "-0.3", "0.7"]), "binary64")
    rejects("actual_report_boolean", lambda: verify(packet, y=True), "binary64")

    # A coherent packet for an inaccurate value must fail the *accuracy* gate.
    bad_y = reported + float(prepared.span) / 1000
    bad_packet = cert.produce_certificate(prepared, x, bad_y, query_id="q")
    rejects("self_consistent_inaccurate_prediction", lambda: verify(bad_packet, y=bad_y), "discrepancy budget")

    # Exact budget boundary: equality passes; the next binary64 value fails.
    zero_payload = deployment(["0"], "0", "0", "1")
    zero = cert.prepare_deployment(zero_payload)
    boundary_y = float(2**-40)
    boundary_packet = cert.produce_certificate(zero, [0.0], boundary_y, query_id="budget")
    proof = cert.verify_certificate(zero, [0.0], boundary_packet, expected_query_id="budget",
                                      reported_prediction=boundary_y)
    assert proof["normalized_error_exact"] == cert.NORMALIZED_ERROR_BUDGET
    passed("exact_discrepancy_budget_equality")
    too_large = math.nextafter(boundary_y, math.inf)
    too_large_packet = cert.produce_certificate(zero, [0.0], too_large, query_id="budget")
    rejects("one_ulp_beyond_discrepancy_budget", lambda: cert.verify_certificate(
        zero, [0.0], too_large_packet, expected_query_id="budget", reported_prediction=too_large), "discrepancy budget")
    signed_packet = cert.produce_certificate(zero, [-0.0], 0.0, query_id="signed")
    rejects("signed_zero_feature_binding", lambda: cert.verify_certificate(
        zero, [0.0], signed_packet, expected_query_id="signed", reported_prediction=0.0), "feature bytes")
    negative_zero = cert.produce_certificate(zero, [0.0], -0.0, query_id="signedreport")
    cert.verify_certificate(zero, [0.0], negative_zero, expected_query_id="signedreport", reported_prediction=-0.0)
    passed("signed_zero_report_is_bound_and_exact")

    # Static attacks rebind their own content hashes, so semantic checking is tested.
    mutations = [
        ("static_budget_relaxed", lambda v: v.update(normalized_error_budget="1/1000"), "budget"),
        ("static_span_zero", lambda v: v.update(span="0"), "positive"),
        ("static_span_negative", lambda v: v.update(span="-1"), "positive"),
        ("static_map_bytes_zero", lambda v: v.update(map_bytes=0), "positive"),
        ("static_map_bytes_boolean", lambda v: v.update(map_bytes=True), "positive"),
        ("static_bad_map_hash", lambda v: v.update(map_sha256="abc"), "SHA-256"),
        ("static_training_optimality_claim", lambda v: v["scope"].update(affine_training_optimum_certified=True), "scope"),
        ("static_raw_robustness_claim", lambda v: v["scope"].update(raw_input_robustness_certified=True), "scope"),
        ("static_dimension_changed", lambda v: v["head"].update(dimension=4), "dimension"),
        ("static_coefficient_norm_failure", lambda v: v["head"].update(coefficients=["3", "0", "0"]), "norm"),
        ("static_noncanonical_rational", lambda v: v["head"].update(intercept="2/4"), "canonical"),
        ("static_changed_range", lambda v: v["head"].update(output_interval=["-1", "1"]), "range"),
        ("static_changed_class", lambda v: v["head"].update(class_id="class35:" + "0"*64), "Class"),
        ("static_unknown_head", lambda v: v["head"].update(kind="unclipped_affine"), "Unsupported"),
    ]
    for name, mutate, reason in mutations:
        altered = deepcopy(payload); mutate(altered); rebind(altered)
        rejects(name, lambda altered=altered: cert.prepare_deployment(altered), reason)
    altered = deepcopy(payload); altered["target_min"] = "-11"; rebind(altered)
    rejects("trusted_expected_deployment_binds_semantically_valid_payload",
            lambda: cert.prepare_deployment(altered, expected_deployment_id=payload["deployment_id"]), "trusted expected identity")
    rejects("trusted_expected_map_binds_payload", lambda: cert.prepare_deployment(payload, expected_map_sha256="b"*64), "trusted expected identity")
    altered = deepcopy(payload); altered["head"]["intercept"] = "1/4"
    rejects("unrebound_head_mutation", lambda: cert.prepare_deployment(altered), "content identity")
    assert len(cert.feature_payload_bytes(x)) == 29 + 8*len(x)
    passed("feature_payload_byte_accounting", {"header_bytes": 29, "coordinate_bytes": 8*len(x)})
    assert len(cert.encode_certificate(packet)) < 1024
    passed("compact_packet_artificial_fixture", {"bytes": len(cert.encode_certificate(packet)),
                                                "not_a_global_size_bound": True})
    return {
        "schema": "step41-artificial-certificate-checks-v1", "status": "PASS",
        "check_count": len(checks), "checks": checks, "seed": 4101,
        "real_cohort_values_read": False, "real_model_fits_performed": False,
        "normalized_error_budget": cert.NORMALIZED_ERROR_BUDGET,
        "scope": dict(cert.SCOPE),
        "source_sha256": hashlib.sha256(Path(cert.__file__).read_bytes()).hexdigest(),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    result = run_checks()
    if len(sys.argv) > 1:
        output = Path(sys.argv[1]); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "check_count": result["check_count"],
                      "source_sha256": result["source_sha256"], "real_cohort_values_read": False}))
