"""Data-free method and arithmetic-assay adapters for frozen Step 43.

The caller owns cohort access, development scoring, selection, serialization,
and protocol gates.  This module never opens observations or chooses a model
from assessment outcomes.  Native/legacy files are retained byte for byte.

Every affine candidate uses the Step 42 P1R1C1 recipe: binary64 centered ridge
with n*lambda, exact radial contraction if required, exact intercept recentering,
and clipping.  No exact training optimizer or ideal feature map is certified.
"""
from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "OMP_THREAD_LIMIT"):
    os.environ[_thread_name] = "1"

from copy import deepcopy
from fractions import Fraction as F
import json
import math
import time

import numpy as np

import affine41 as legacy_affine
import certificate41 as cert
import models41 as legacy_baseline
import representations35 as rep


LAMBDA_GRID = (("1e-6", "1/1000000"), ("1e-4", "1/10000"),
               ("1e-2", "1/100"), ("1", "1"), ("100", "100"))
SEEDS = (3901, 4301, 4302)
BUDGET_REJECTION = "Reported prediction exceeds the frozen discrepancy budget"
CERTIFICATE_FIELDS = {
    "schema", "deployment_id", "query_id", "feature_sha256",
    "reported_prediction_hex", "reference_interval_hex", "absolute_error_upper_hex",
}


class MethodError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise MethodError(message)


def map_names():
    names = ["raw_" + scaling for scaling in ("minmax01", "standard")]
    names += [f"nystrom_{scaling}_w{width}" for scaling in ("minmax01", "standard")
              for width in (8, 24)]
    names += [f"kahm_{scaling}_cap{cap}_p{prefix}"
              for scaling in ("minmax01", "standard")
              for cap in (4, 12, 20) for prefix in (1, 2)]
    return sorted(names)


def affine_configs():
    rows = []
    for name in map_names():
        family = name.split("_", 1)[0]
        scaling = "minmax01" if "minmax01" in name else "standard"
        for token, lam in LAMBDA_GRID:
            rows.append({"candidate_id": name + "::lambda_" + token,
                         "map": name, "family": family, "scaling": scaling,
                         "lambda_token": token, "ridge_lambda": lam,
                         "recipe": "affine", "kappa": "2"})
    _require(len(rows) == 90 and len({r['candidate_id'] for r in rows}) == 90,
             "Affine roster construction differs")
    return rows


def baseline_configs(seed):
    _require(type(seed) is int and seed in SEEDS, "Seed is outside the fixed study roster")
    rows = deepcopy(legacy_baseline.baseline_configs())
    for row in rows:
        if "random_state" in row["parameters"]:
            row["parameters"]["random_state"] = seed
    _require(len(rows) == 44, "Conventional roster construction differs")
    return rows


def fit_representations(X, seed):
    _require(type(seed) is int and seed in SEEDS, "Seed is outside the fixed study roster")
    maps, metadata = rep.fit_representations(X, deepcopy(rep.DEFAULT_CONFIG), seed)
    _require(sorted(maps) == map_names(), "Representation factory returned another roster")
    return maps, metadata


transform_one = rep.transform_one
serialize_map = rep.serialize_map
deserialize_map = rep.deserialize_map
numeric_map_bytes = rep.numeric_bytes
initialize_runtime = rep.initialize_runtime


def fit_affine(cfg, Z, exact_normalized_y, normalization, *, feature_id,
               map_sha256, map_bytes, training_binding=None):
    expected = {r["candidate_id"]: r for r in affine_configs()}
    _require(isinstance(cfg, dict) and cfg.get("candidate_id") in expected
             and cfg == expected[cfg["candidate_id"]], "Affine configuration differs from fixed roster")
    y = list(map(F, exact_normalized_y))
    _require(bool(y) and all(F(0) <= value <= F(1) for value in y),
             "Training targets must use the training-only normalized interval")
    _require(F(normalization["span"]) > 0, "Training target span must be positive")
    fitted = legacy_affine.fit_affine(
        Z, y, feature_id=feature_id, kappa="2", ridge_lambda=cfg["ridge_lambda"],
        training_binding={} if training_binding is None else training_binding)
    head = fitted["head"]
    deployment = cert.make_deployment(
        head, target_min=normalization["target_min"], span=normalization["span"],
        map_sha256=map_sha256, map_bytes=map_bytes, candidate_id=cfg["candidate_id"])
    prepared = cert.prepare_deployment(
        deployment, expected_deployment_id=deployment["deployment_id"],
        expected_map_sha256=map_sha256)
    _require(fitted["projection"]["intercept_recentered"] is True,
             "Exact intercept recentering was not retained")
    _require(sum((F(x)**2 for x in head["coefficients"]), F(0)) <= 4,
             "Exact norm bound was not retained")
    return {"fit": fitted, "head": head, "deployment": deployment, "prepared": prepared}


def fit_baseline(cfg, X, exact_normalized_y, normalization):
    _require(isinstance(cfg, dict) and isinstance(cfg.get("parameters"), dict),
             "Malformed baseline configuration")
    seed = cfg["parameters"].get("random_state", SEEDS[0])
    expected = {r["candidate_id"]: r for r in baseline_configs(seed)}
    _require(cfg.get("candidate_id") in expected and cfg == expected[cfg["candidate_id"]],
             "Baseline configuration differs from fixed roster")
    y = list(map(F, exact_normalized_y))
    _require(bool(y) and all(F(0) <= value <= F(1) for value in y),
             "Training targets must use the training-only normalized interval")
    return legacy_baseline.fit_baseline(cfg, X, y, normalization)


predict_baseline = legacy_baseline.predict_baseline
predict_one = cert.predict_float
exact_reference = cert.exact_prediction
prepare_deployment = cert.prepare_deployment
encode_certificate = cert.encode_certificate
feature_payload_bytes = cert.feature_payload_bytes


def _reported(value):
    _require(type(value) is float or isinstance(value, np.float64),
             "Reported value must be an actual binary64 value")
    result = float(value)
    _require(math.isfinite(result), "Reported value must be finite")
    return result


def binding_only_check(prepared, features, packet, *, query_id, reported):
    """Deliberately limited content-binding control, without numerical checking.

    This is not a security verifier.  In particular it does not validate either
    the reference enclosure or the discrepancy bound/budget.  A self-consistent
    packet for an inaccurate reported number can pass this control.
    """
    reason = None
    if type(packet) is not dict or set(packet) != CERTIFICATE_FIELDS:
        reason = "packet_fields"
    elif packet["schema"] != cert.CERTIFICATE_SCHEMA:
        reason = "schema"
    elif packet["deployment_id"] != prepared.deployment_id:
        reason = "deployment_id"
    elif type(query_id) is not str or not query_id or packet["query_id"] != query_id:
        reason = "query_id"
    elif packet["feature_sha256"] != cert.feature_digest(features):
        reason = "feature_bytes"
    else:
        try:
            value = _reported(reported)
            retained = float.fromhex(packet["reported_prediction_hex"])
            if not math.isfinite(retained) or retained.hex() != packet["reported_prediction_hex"]:
                reason = "reported_format"
            elif retained.hex() != value.hex():
                reason = "reported_value"
        except (MethodError, TypeError, ValueError, OverflowError):
            reason = "reported_format"
    return {"status": "PASS" if reason is None else "REJECTED",
            "accepted": reason is None, "reason": reason,
            "numerical_discrepancy_checked": False,
            "authentication_claim": False,
            "scope": "Schema/deployment/query/feature/reported-value consistency only"}


def certify(prepared, features, reported, query_id):
    """Retain an honest packet and any valid numerical-budget rejection.

    A budget rejection is a scientific outcome, not an implementation failure.
    Any other failure of a newly produced, consistently bound packet propagates.
    The independent study auditor rederives the reference and acceptance rule.
    """
    reported = _reported(reported)
    packet = cert.produce_certificate(prepared, features, reported, query_id=query_id)
    reference = cert.exact_prediction(prepared, features)
    error = abs(F.from_float(reported) - reference)
    budget = prepared.span * prepared.normalized_error_budget
    expected = error <= budget
    try:
        verification = cert.verify_certificate(
            prepared, features, packet, expected_query_id=query_id, reported_prediction=reported)
        accepted = True
    except cert.CertificateError as failure:
        if str(failure) != BUDGET_REJECTION:
            raise
        accepted = False
        verification = {"status": "REJECTED_OVER_BUDGET", "reason": str(failure)}
    _require(accepted == expected, "Checker outcome differs from exact numerical predicate")
    binding = binding_only_check(prepared, features, packet, query_id=query_id, reported=reported)
    _require(binding["accepted"], "Newly produced packet failed its content-binding control")
    return {"reported_physical_hex": reported.hex(), "exact_physical": str(reference),
            "absolute_physical_error": str(error), "physical_error_budget": str(budget),
            "accepted": accepted, "expected_acceptance": expected,
            "certificate": packet, "verification": verification, "binding_only": binding,
            "packet_json_bytes": len(cert.encode_certificate(packet)),
            "feature_payload_bytes": len(cert.feature_payload_bytes(features))}


def prediction_assay(prepared, features, reported, query_id, *, honest_record=None):
    """Compare binary64 output with physical-output float32 quantization.

    No response label is accepted.  The quantized case obtains a self-consistent
    packet for its own reported value.  It can legitimately pass when its exact
    discrepancy fits the frozen budget.  Nonfinite quantization stops the assay
    instead of silently excluding an observation.
    """
    reported = _reported(reported)
    if honest_record is None:
        honest_record = certify(prepared, features, reported, query_id)
    _require(honest_record["reported_physical_hex"] == reported.hex()
             and honest_record["certificate"]["query_id"] == query_id
             and honest_record["certificate"]["deployment_id"] == prepared.deployment_id
             and honest_record["certificate"]["feature_sha256"] == cert.feature_digest(features),
             "Reused honest receipt belongs to another prediction")
    with np.errstate(over="ignore", invalid="raise"):
        quantized = float(np.float32(reported))
    _require(math.isfinite(quantized), "Physical output is outside the finite float32 assay range")
    rounded_record = certify(prepared, features, quantized, query_id)
    return {"schema": "step43-numerical-output-assay-v1", "query_id": query_id,
            "target_label_accessed": False,
            "quantization_changed_reported_value": quantized.hex() != reported.hex(),
            "cases": {"binary64": honest_record, "physical_output_float32": rounded_record},
            "scope": "Numerical-contract enforcement beyond identity-binding consistency; no statistical-accuracy or authentication claim"}


def _different_hash(value):
    return ("0" if value[0] != "0" else "1") + value[1:]


def misbinding_assay(prepared, features, reported, query_id, *, stale_features,
                     honest_record=None):
    """Fixed first-query contract challenges; unrelated to forecasting error.

    The caller supplies the stale vector from a predeclared different query.
    Identical feature bytes are explicitly reported as an inapplicable stale-
    feature challenge, never counted as a rejection or hidden by substitution.
    """
    reported = _reported(reported)
    if honest_record is None:
        honest_record = certify(prepared, features, reported, query_id)
    original = honest_record["certificate"]
    _require(original["deployment_id"] == prepared.deployment_id
             and original["query_id"] == query_id
             and original["reported_prediction_hex"] == reported.hex()
             and original["feature_sha256"] == cert.feature_digest(features),
             "Misbinding input packet belongs to another prediction")
    wrong_query = deepcopy(original)
    wrong_query["query_id"] = query_id + "::step43_wrong_query"
    wrong_deployment = deepcopy(original)
    prefix, suffix = original["deployment_id"].split(":", 1)
    wrong_deployment["deployment_id"] = prefix + ":" + _different_hash(suffix)
    variants = [
        ("wrong_query", wrong_query, "Certificate query ID differs", "query_id"),
        ("wrong_deployment", wrong_deployment, "Certificate deployment differs", "deployment_id"),
    ]
    stale_hash = cert.feature_digest(stale_features)
    stale_differs = stale_hash != original["feature_sha256"]
    if stale_differs:
        stale_packet = cert.produce_certificate(prepared, stale_features, reported, query_id=query_id)
        variants.append(("stale_feature", stale_packet, "Certificate feature bytes differ", "feature_bytes"))
    rows = []
    for name, packet, expected_error, expected_binding_reason in variants:
        binding = binding_only_check(prepared, features, packet, query_id=query_id, reported=reported)
        _require(not binding["accepted"] and binding["reason"] == expected_binding_reason,
                 "Binding control did not reject its prescribed mismatch")
        try:
            cert.verify_certificate(prepared, features, packet,
                                    expected_query_id=query_id, reported_prediction=reported)
        except cert.CertificateError as failure:
            _require(str(failure) == expected_error, "Misbinding rejected for an unexpected reason")
            rows.append({"challenge": name, "status": "REJECTED_AS_EXPECTED",
                         "certificate": packet, "full_rejection": str(failure),
                         "binding_only": binding})
        else:
            raise MethodError("Prescribed mismatched packet was accepted")
    if not stale_differs:
        rows.append({"challenge": "stale_feature", "status": "NOT_APPLICABLE_EQUAL_FEATURE_BYTES",
                     "certificate": None, "full_rejection": None, "binding_only": None})
    return {"schema": "step43-misbinding-assay-v1", "query_id": query_id,
            "feature_sha256": original["feature_sha256"], "stale_feature_sha256": stale_hash,
            "stale_features_differ": stale_differs, "challenges": rows,
            "target_label_accessed": False, "authentication_claim": False,
            "scope": "Fixed model/query/feature content-binding contract checks"}


def _timed(fn, *, warmups, repeats):
    _require(type(warmups) is int and warmups >= 0 and type(repeats) is int and repeats > 0,
             "Invalid timing repetition counts")
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - start)
    return samples


def _verification_for_timing(prepared, z, packet, query_id, reported):
    try:
        return cert.verify_certificate(prepared, z, packet,
                                       expected_query_id=query_id, reported_prediction=reported)
    except cert.CertificateError as failure:
        if str(failure) != BUDGET_REJECTION:
            raise
        return {"status": "REJECTED_OVER_BUDGET"}


def benchmark_static_context(deployment, *, warmups=3, repeats=100):
    wire = cert.canonical_bytes(deployment)
    expected = {"expected_deployment_id": deployment["deployment_id"],
                "expected_map_sha256": deployment["map_sha256"]}
    def prepare():
        return cert.prepare_deployment(json.loads(wire), **expected)
    prepare()
    return {"nanoseconds": _timed(prepare, warmups=warmups, repeats=repeats),
            "warmups": warmups, "repeats": repeats, "static_deployment_json_bytes": len(wire),
            "scope": "Parse deployment JSON and validate/cache the trusted static context"}


def benchmark_query(prepared, mapping, X_one, stored_features, query_id, reported,
                    *, query_index, sample_ordinal, warmups=3, repeats=10):
    """Measure four prescribed primitives on already assessed selected models.

    The direct-reference control uses the same integer dot-product engine as
    the checker, with trusted inputs but no packet parsing, hashing, interval
    checks or discrepancy reporting.  It is a cost control, not an audit.
    """
    reported = _reported(reported)
    X_one = np.asarray(X_one, dtype=np.float64)
    z = np.asarray(stored_features, dtype=np.float64)
    _require(X_one.shape == (1, 24) and z.ndim == 1,
             "Timing requires one raw query and one stored feature vector")
    _require(type(query_index) is int and query_index >= 0, "Invalid benchmark query index")
    _require(type(sample_ordinal) is int and sample_ordinal >= 0,
             "Invalid selected-query sample ordinal")
    reproduced = rep.transform_one(mapping, X_one)[0]
    _require(np.array_equal(reproduced.view(np.uint64), z.view(np.uint64)),
             "Timing transform replay changed stored query features")
    _require(cert.predict_float(prepared, z).hex() == reported.hex(),
             "Timing replay changed the reported prediction")
    packet = cert.produce_certificate(prepared, z, reported, query_id=query_id)
    wire = cert.encode_certificate(packet)
    _verification_for_timing(prepared, z, packet, query_id, reported)
    _require(cert._integer_reference(prepared, z) == cert.exact_prediction(prepared, z),
             "Direct reference cost control differs from the rational reference")
    def unverified():
        return cert.predict_float(prepared, rep.transform_one(mapping, X_one)[0])
    def verified():
        zz = rep.transform_one(mapping, X_one)[0]
        value = cert.predict_float(prepared, zz)
        encoded = cert.encode_certificate(cert.produce_certificate(prepared, zz, value, query_id=query_id))
        return _verification_for_timing(prepared, zz, json.loads(encoded), query_id, value)
    functions = {
        "prediction_without_certificate": unverified,
        "certified_prediction": verified,
        "certificate_verification": lambda: _verification_for_timing(
            prepared, z, json.loads(wire), query_id, reported),
        "direct_exact_reference": lambda: cert._integer_reference(prepared, z),
    }
    pair = ["prediction_without_certificate", "certified_prediction"]
    if sample_ordinal % 2:
        pair.reverse()
    order = pair + ["certificate_verification", "direct_exact_reference"]
    samples = {name: _timed(functions[name], warmups=warmups, repeats=repeats) for name in order}
    reference = cert.exact_prediction(prepared, z)
    return {"query_id": query_id, "query_index": query_index, "sample_ordinal": sample_ordinal,
            "primitive_order": order, "warmups": warmups, "repeats": repeats,
            "nanoseconds": samples,
            "median_nanoseconds": {name: float(np.median(values)) for name, values in samples.items()},
            "packet_json_bytes": len(wire), "feature_payload_bytes": len(cert.feature_payload_bytes(z)),
            "static_deployment_json_bytes": len(prepared.deployment_bytes), "map_bytes": prepared.map_bytes,
            "absolute_physical_error": str(abs(F.from_float(reported) - reference)),
            "physical_error_budget": str(prepared.span * prepared.normalized_error_budget),
            "direct_exact_reference_scope": "Same cached integer reference as checker; no packet parsing, identity checks or discrepancy reporting",
            "timing_scope": "Local one-query execution; excludes disk/network I/O and process startup"}


def benchmark_baseline(payload, X_one, *, warmups=3, repeats=10):
    X_one = np.asarray(X_one, dtype=np.float64)
    _require(X_one.shape == (1, 24), "Baseline timing requires one raw query")
    value = legacy_baseline.predict_baseline(payload, X_one)
    _require(value.shape == (1,), "Baseline query returned another shape")
    samples = _timed(lambda: legacy_baseline.predict_baseline(payload, X_one),
                     warmups=warmups, repeats=repeats)
    return {"reported_physical_hex": float(value[0]).hex(), "nanoseconds": samples,
            "median_nanoseconds": float(np.median(samples)), "warmups": warmups, "repeats": repeats,
            "scope": "Selected conventional scaler/model/inverse-target scaling on one stored raw query"}
