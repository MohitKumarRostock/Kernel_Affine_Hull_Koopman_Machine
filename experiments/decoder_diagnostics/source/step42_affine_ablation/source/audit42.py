"""Independent arithmetic, fit-recipe, selection, and attribution audit for Step 42.

The audit does not import the producer head or scoring implementation. Numerical
ridge solves are verification replays of the declared recipe, not new candidate
searches. Exact query references use integer common-denominator accumulation.
All empirical conclusions remain diagnostic on already exposed Step 41 periods.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from math import isfinite, isqrt, lcm, sqrt, nextafter, inf
from pathlib import Path
import struct
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("household_power2010", "jena_temperature2017")
STAGES = ("f1", "f2", "f3", "final")
LAMBDAS = ("1/1000000", "1/10000", "1/100", "1", "100")
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"


class AuditError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def fraction(value, name="rational"):
    require(type(value) is str, name + " is not an explicit rational string")
    try:
        number = F(value)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise AuditError(name + " is invalid") from exc
    require(str(number) == value, name + " is not canonical")
    return number


def from_hex(value):
    require(type(value) is str, "Prediction is not a binary64 hexadecimal string")
    try:
        result = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise AuditError("Invalid binary64 prediction") from exc
    require(isfinite(result) and result.hex() == value,
            "Prediction is not finite canonical binary64")
    return result


def dyadic(value, name="binary64 rational"):
    number = fraction(value, name)
    try:
        floating = float(number)
    except OverflowError as exc:
        raise AuditError(name + " overflows binary64") from exc
    require(isfinite(floating) and F.from_float(floating) == number,
            name + " is not an exactly representable finite binary64 value")
    return number


def feature_matrix(value):
    matrix = np.asarray(value)
    require(matrix.dtype == np.dtype("float64") and matrix.ndim == 2 and
            matrix.shape[0] > 0 and matrix.shape[1] > 0 and
            bool(np.isfinite(matrix).all()), "Invalid stored binary64 feature matrix")
    return matrix


def exact_centers(z, targets):
    z = feature_matrix(z)
    ys = [F(y) for y in targets]
    require(len(ys) == len(z), "Training target count differs")
    center = [sum((F.from_float(float(x)) for x in z[:, j]), F(0)) / len(z)
              for j in range(z.shape[1])]
    return center, sum(ys, F(0)) / len(ys)


def audit_base(base, z, targets, *, feature_id, ridge_lambda, training_binding=None):
    """Replay the stated float64 ridge recipe and independently check exact centers."""
    z = feature_matrix(z)
    ys = [F(y) for y in targets]
    lam = F(ridge_lambda)
    require(lam > 0 and len(ys) == len(z), "Invalid ridge training contract")
    require(base["schema"] == "step42-ridge-base-v1" and
            base["feature_id"] == feature_id and
            base["ridge_lambda"] == str(lam) and
            base["n"] == len(z) and base["dimension"] == z.shape[1],
            "Base identity, penalty, or dimensions differ")
    if training_binding is not None:
        require(base["training_binding"] == training_binding,
                "Base training-only input binding differs")
    copied = dict(base)
    saved_id = copied.pop("base_id")
    require(saved_id == "base42:" + digest(copied), "Base content ID differs")
    beta = [dyadic(x, "source coefficient") for x in base["coefficients"]]
    b = dyadic(base["intercept"], "source intercept")
    require(len(beta) == z.shape[1], "Source coefficient dimension differs")
    require(base["penalty_multiplier"] == str(len(z) * lam),
            "Ridge penalty is not n times lambda")
    require(base["solver"] == "numpy.linalg.solve" and
            base["objective"] == "mean_squared_error_plus_lambda_coefficient_norm_squared" and
            base["intercept_penalty"] == "none" and
            base["centering"] == "binary64_unweighted_feature_and_rounded_target_means" and
            base["training_optimality_claim"] is False,
            "Base solver semantics or optimization scope differs")
    require(base["coefficient_squared_norm"] == str(sum((x*x for x in beta), F(0))),
            "Source coefficient squared norm differs")
    center, center_y = exact_centers(z, ys)
    require(base["center_features"] == [str(x) for x in center] and
            base["center_target"] == str(center_y), "Exact training centers differ")

    # Separate numerical recipe replay. No producer fitting or prediction helper
    # is used, and these coefficients do not enter candidate selection.
    yf = np.asarray([float(y) for y in ys], dtype=np.float64)
    zmean = z.mean(axis=0)
    ymean = yf.mean()
    centered = z - zmean
    centered_y = yf - ymean
    matrix = centered.T @ centered + len(z) * float(lam) * np.eye(z.shape[1])
    rhs = centered.T @ centered_y
    replay = np.linalg.solve(matrix, rhs)
    replay_b = float(ymean - zmean @ replay)
    require([str(F.from_float(float(x))) for x in replay] == base["coefficients"] and
            str(F.from_float(replay_b)) == base["intercept"],
            "Stored ridge fit is not the prescribed centered solve")
    residual = matrix @ np.asarray([float(x) for x in beta]) - rhs
    denominator = (np.linalg.norm(matrix, ord=np.inf) *
                   np.linalg.norm(replay, ord=np.inf) + np.linalg.norm(rhs, ord=np.inf))
    backward_error = float(np.linalg.norm(residual, ord=np.inf) / denominator) if denominator else 0.0
    require(isfinite(backward_error) and backward_error <= 256 * z.shape[1] * np.finfo(float).eps,
            "Numerical solve has excessive scaled residual")
    return {"n": len(z), "dimension": z.shape[1], "scaled_normal_equation_residual": backward_error,
            "base_id": saved_id, "source_intercept": str(b)}


def upper_sqrt(value, bits=80):
    require(value >= 0 and type(bits) is int and 1 <= bits <= 512,
            "Invalid exact square-root premise")
    scale = 1 << bits
    k = isqrt((value.numerator * scale * scale) // value.denominator)
    lower = F(k, scale)
    upper = lower if lower * lower == value else F(k + 1, scale)
    require(lower * lower <= value <= upper * upper and upper - lower <= F(1, scale),
            "Exact square-root enclosure failed")
    return upper


def audit_variant(base, variant, *, project, recenter, clip, kappa="2", bits=80):
    require(all(type(x) is bool for x in (project, recenter, clip)), "Nonboolean factor switch")
    kap = F(kappa)
    require(kap > 0, "Nonpositive sensitivity bound")
    require(variant["schema"] == "step42-affine-variant-v1" and
            variant["base_id"] == base["base_id"] and
            variant["feature_id"] == base["feature_id"] and
            variant["ridge_lambda"] == base["ridge_lambda"] and
            variant["switches"] == {"projection": project, "recenter": recenter, "clipping": clip} and
            variant["kappa"] == str(kap) and variant["projection_bits"] == bits,
            "Variant identity or factorial switches differ")
    initial = [fraction(x) for x in base["coefficients"]]
    initial_b = fraction(base["intercept"])
    source_squared = sum((x*x for x in initial), F(0))
    factor = kap / upper_sqrt(source_squared, bits) if project and source_squared > kap*kap else F(1)
    beta = [factor*x for x in initial]
    center = [fraction(x) for x in base["center_features"]]
    center_y = fraction(base["center_target"])
    b = center_y - sum((x*y for x, y in zip(beta, center)), F(0)) if recenter else initial_b
    squared = sum((x*x for x in beta), F(0))
    mean = b + sum((x*y for x, y in zip(beta, center)), F(0))
    head = {"schema": "step42-stored-feature-affine-head-v1",
            "kind": "clipped_affine" if clip else "affine",
            "feature_id": base["feature_id"], "dimension": base["dimension"],
            "coefficients": [str(x) for x in beta], "intercept": str(b),
            "clipping": clip, "output_interval": ["0", "1"] if clip else None,
            "assurance": ASSURANCE}
    require(variant["head"] == head and variant["head_id"] == "head42:" + digest(head),
            "Variant rational head or content identity differs")
    transformation = {"source_squared_norm": str(source_squared),
                      "coefficient_squared_norm": str(squared),
                      "projection_factor": str(factor), "projection_active": factor != 1,
                      "source_intercept": str(initial_b), "recenter_delta": str(b-initial_b),
                      "training_mean_prediction_before_clipping": str(mean),
                      "center_target": str(center_y)}
    require(variant["transformation"] == transformation, "Variant transformation receipt differs")
    expected_claims = {"norm_within_kappa": squared <= kap*kap,
                       "normalized_range": ["0", "1"] if clip else None,
                       "training_mean_affine_matches_target": mean == center_y,
                       "native_transform_error_enclosed": False, "population_risk_claim": False}
    require(variant["structural_claims"] == expected_claims, "Variant structural scope is overstated or differs")
    if project:
        require(squared <= kap*kap, "Projected coefficients exceed the declared sensitivity")
    return head


def prepare_exact(head):
    coefficients = [fraction(x) for x in head["coefficients"]]
    intercept = fraction(head["intercept"])
    denominator = lcm(intercept.denominator, *(x.denominator for x in coefficients))
    return {"denominator": denominator,
            "integer_coefficients": [x.numerator * (denominator // x.denominator) for x in coefficients],
            "integer_intercept": intercept.numerator * (denominator // intercept.denominator),
            "float_coefficients": np.asarray([float(x) for x in coefficients], dtype=np.float64),
            "float_intercept": float(intercept), "clipping": head["clipping"]}


def integer_features(row):
    ratios = [float(x).as_integer_ratio() for x in row]
    denominator = max(b for _, b in ratios)
    require(all(denominator % b == 0 for _, b in ratios), "Feature denominator is not dyadic")
    return [a*(denominator//b) for a, b in ratios], denominator


def prediction_reference(prepared, row, target_min, span, integers=None):
    """Separate integer exact evaluator and original one-query operational path."""
    a, s = F(target_min), F(span)
    require(s > 0, "Target normalization span is nonpositive")
    row = np.asarray(row, dtype=np.float64)
    require(row.ndim == 1 and len(row) == len(prepared["integer_coefficients"]) and
            bool(np.isfinite(row).all()), "Query feature shape or finiteness differs")
    values, denominator = integer_features(row) if integers is None else integers
    numerator = prepared["integer_intercept"] * denominator + sum(
        x*y for x, y in zip(prepared["integer_coefficients"], values))
    exact_raw = F(numerator, prepared["denominator"] * denominator)
    exact_normal = max(F(0), min(F(1), exact_raw)) if prepared["clipping"] else exact_raw
    exact_physical = a + s*exact_normal
    floating_raw = float(np.dot(prepared["float_coefficients"], row)) + prepared["float_intercept"]
    floating_normal = min(1.0, max(0.0, floating_raw)) if prepared["clipping"] else floating_raw
    physical = float(a) + float(s)*floating_normal
    require(isfinite(floating_raw) and isfinite(physical), "Nonfinite operational prediction")
    return {"reported_physical_hex": float(physical).hex(), "exact_physical": str(exact_physical),
            "raw_exact": exact_raw, "raw_float": floating_raw,
            "exact_clipping_active": bool(prepared["clipping"] and (exact_raw < 0 or exact_raw > 1)),
            "operational_clipping_active": bool(prepared["clipping"] and (floating_raw < 0 or floating_raw > 1)),
            "exact_clipping_side": "below" if prepared["clipping"] and exact_raw < 0 else
                                   "above" if prepared["clipping"] and exact_raw > 1 else "none",
            "operational_clipping_side": "below" if prepared["clipping"] and floating_raw < 0 else
                                         "above" if prepared["clipping"] and floating_raw > 1 else "none",
            "absolute_error": abs(F.from_float(float(physical)) - exact_physical)}


def exact_score(values, targets, *, hex_values=False):
    require(len(values) == len(targets) and len(targets) > 0, "Score roster differs or is empty")
    predictions = [F.from_float(from_hex(x)) if hex_values else F(x) for x in values]
    ys = [F(x) for x in targets]
    losses = [(a-b)**2 for a, b in zip(predictions, ys)]
    sse = sum(losses, F(0))
    mae = sum((abs(a-b) for a, b in zip(predictions, ys)), F(0))/len(ys)
    return {"n": len(ys), "sse": str(sse), "mse": str(sse/len(ys)),
            "rmse": sqrt(float(sse/len(ys))), "mae": str(mae)}


def compare_score(actual, expected, label):
    require(actual == expected, "Exact physical scoring differs: " + label)


def unique_rows(rows, field="candidate_id"):
    require(isinstance(rows, list) and all(isinstance(x, dict) and field in x for x in rows),
            "Malformed result row list")
    result = {x[field]: x for x in rows}
    require(len(result) == len(rows), "Duplicate result identities")
    return result


def choose(rows, *, legacy=False):
    require(bool(rows), "Empty selector pool")
    if legacy:
        return min(rows, key=lambda r: (F(r["pooled_physical_mse"]),
                                       r["legacy_f3_deployment_bytes"], r["map"] + "::affine"))
    return min(rows, key=lambda r: (F(r["pooled_physical_mse"]), r["candidate_id"]))


def expected_maps():
    names = [f"raw_{scale}" for scale in ("minmax01", "standard")]
    names += [f"nystrom_{scale}_w{width}" for scale, width in
              product(("minmax01", "standard"), (8, 24))]
    names += [f"kahm_{scale}_cap{cap}_p{prefix}" for scale, cap, prefix in
              product(("minmax01", "standard"), (4, 12, 20), (1, 2))]
    return sorted(names)


def expected_configurations():
    configs = []
    for name in expected_maps():
        for token, lam in zip(("1e-6", "1e-4", "1e-2", "1", "100"), LAMBDAS):
            base_key = name + "::lambda_" + token
            for project, recenter, clip in product((False, True), repeat=3):
                suffix = f"P{int(project)}R{int(recenter)}C{int(clip)}"
                configs.append({"candidate_id": base_key + "::" + suffix,
                                "base_key": base_key, "map": name,
                                "family": name.split("_")[0],
                                "scaling": "minmax01" if "_minmax01" in name else "standard",
                                "lambda_token": token, "ridge_lambda": lam,
                                "project": project, "recenter": recenter, "clip": clip})
    return configs


def expected_specs():
    specs = []
    for project, recenter, clip in product((False, True), repeat=3):
        tag = f"P{int(project)}R{int(recenter)}C{int(clip)}"
        fixed = {"project": project, "recenter": recenter, "clip": clip}
        specs.append({"label": "recipe_" + tag, "fixed": fixed, "pool_size": 90,
                      "tie_rule": "lexical_candidate_id"})
        for family, count in (("raw", 10), ("nystrom", 20), ("kahm", 60)):
            specs.append({"label": family + "_recipe_" + tag,
                          "fixed": {**fixed, "family": family}, "pool_size": count,
                          "tie_rule": "lexical_candidate_id"})
        for name in expected_maps():
            specs.append({"label": "map_" + name + "_recipe_" + tag,
                          "fixed": {**fixed, "map": name}, "pool_size": 5,
                          "tie_rule": "lexical_candidate_id"})
    specs.append({"label": "original41_replay",
                  "fixed": {"project": True, "recenter": True, "clip": True, "lambda_token": "1e-6"},
                  "pool_size": 18,
                  "tie_rule": "legacy_f3_deployment_bytes_then_lexical_candidate_id"})
    return specs


def safe_path(relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and bool(path.parts),
            "Unsafe record-relative path")
    resolved = (ROOT/path).resolve()
    require(resolved.is_relative_to(ROOT.resolve()), "Record path escapes Step42")
    return resolved


def protocol_check():
    seal = read(ROOT/"protocol_seal42.json")
    require(seal["schema"] == "step42-protocol-source-seal-v1" and
            seal["data_already_exposed"] is True and
            seal["new_grid_not_executed_before_seal"] is True,
            "Protocol provenance scope differs")
    require("protocol42.json" in seal["bound_files"], "Protocol is not sealed")
    for relative, wanted in seal["bound_files"].items():
        require(sha(safe_path(relative)) == wanted, "Frozen dependency changed: " + relative)
    protocol = read(ROOT/"protocol42.json")
    require(protocol["schema"] == "step42-controlled-head-ablation-protocol-v1" and
            protocol["step"] == 42 and protocol["status"] == "DIAGNOSTIC_ALREADY_EXPOSED_PERIODS",
            "Diagnostic status changed")
    require(protocol["tasks"] == list(TASKS) and protocol["stages"] == list(STAGES) and
            protocol["maps"] == expected_maps() and
            protocol["configurations"] == expected_configurations(), "Factorial candidate grid differs")
    require(protocol["selection"]["specifications"] == expected_specs() and
            protocol["selection"]["folds"] == ["f1", "f2", "f3"] and
            protocol["selection"]["assessment_reselection"] is False,
            "Development selection contract differs")
    require(protocol["counts"] == {"maps": 18, "lambdas": 5, "factorial_recipes": 8,
                                   "configurations_per_stage": 720, "base_solves_per_stage": 90,
                                   "stages_per_task": 4, "tasks": 2, "base_solves_total": 720,
                                   "variant_payloads_total": 5760, "selectors_per_task": 177},
            "Factorial count contract differs")
    require(protocol["fitting"]["kappa"] == "2" and protocol["fitting"]["projection_bits"] == 80 and
            protocol["fitting"]["lambda_grid"] == [
                {"token": t, "value": v} for t, v in
                zip(("1e-6", "1e-4", "1e-2", "1", "100"), LAMBDAS)],
            "Fitting constants differ")
    require(all(protocol["data"][key] is False for key in
                ("new_outcome_information", "new_domains", "new_assessment_periods", "feature_maps_refitted")) and
            protocol["data"]["no_training_on_evaluation_labels"] is True,
            "Data reuse or training scope differs")
    require(protocol["diagnostics"]["inferential_p_values"] is False and
            protocol["diagnostics"]["population_success_claim"] is False and
            protocol["prior_primary_status"] == "Step41 remains NOT_ESTABLISHED on both periods",
            "Retrospective result has been promoted to a new success claim")
    require(protocol["execution"]["producer_passes"] == 1 and
            protocol["execution"]["outcome_based_reruns"] is False,
            "Single-run diagnostic contract differs")
    return protocol, seal


def legacy_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def outward_pair(value):
    number = float(value)
    require(isfinite(number), "Reference not enclosed by finite binary64")
    represented = F.from_float(number)
    if represented == value:
        return [number.hex(), number.hex()]
    if represented < value:
        return [number.hex(), nextafter(number, inf).hex()]
    return [nextafter(number, -inf).hex(), number.hex()]


def audit_legacy_packet(variant, deployment, packet, row, *, query_id,
                        reported_hex, target_min, span, map_sha256, map_bytes,
                        candidate_id, allow_over_budget=False):
    """Check actual compatibility with the unchanged Step41 packet contract."""
    head = variant["head"]
    require(head["clipping"] is True and
            sum((F(x)**2 for x in head["coefficients"]), F(0)) <= 4,
            "Unbounded variant cannot use Step41 class certificate")
    legacy_head = {
        "schema": "step35-stored-feature-head-v1", "kind": "clipped_affine",
        "feature_id": head["feature_id"], "dimension": head["dimension"],
        "kappa": "2", "output_interval": ["0", "1"], "assurance": ASSURANCE,
        "coefficients": head["coefficients"], "intercept": head["intercept"],
    }
    descriptor = {
        "schema": "step35-stored-feature-decoder-class-v1", "feature_id": head["feature_id"],
        "metric": "Euclidean_distance_on_supplied_feature_coordinates",
        "decoder_class": "all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space",
        "kappa": "2", "output_interval": ["0", "1"], "assurance": ASSURANCE,
    }
    legacy_head["class_id"] = "class35:" + legacy_digest(descriptor)
    require(deployment["head"] == legacy_head, "Legacy conversion changed the selected fitted head")
    expected_deployment = {
        "schema": "step41-affine-deployment-v1", "head": legacy_head,
        "head_id": "head35:" + legacy_digest(legacy_head), "candidate_id": candidate_id,
        "target_min": str(F(target_min)), "span": str(F(span)),
        "map_sha256": map_sha256, "map_bytes": map_bytes,
        "normalized_error_budget": "1/1099511627776",
        "scope": {"assurance": ASSURANCE, "reference": "FIXED_STORED_RATIONAL_AFFINE_HEAD",
                  "affine_training_optimum_certified": False, "native_transform_error_enclosed": False,
                  "raw_input_robustness_certified": False, "future_response_coverage_claim": False,
                  "population_risk_claim": False, "cryptographic_authenticity_claim": False},
    }
    expected_deployment["deployment_id"] = "deploy41:" + legacy_digest(expected_deployment)
    require(deployment == expected_deployment, "Legacy deployment identity or trust scope differs")
    prediction = prediction_reference(prepare_exact(head), row, target_min, span)
    require(reported_hex == prediction["reported_physical_hex"], "Packet reported prediction differs")
    reference = F(prediction["exact_physical"])
    error = prediction["absolute_error"]
    high_error = outward_pair(error)[1]
    feature_bytes = (b"step41-feature-vector-v1\0" + struct.pack(">I", len(row)) +
                     np.asarray(row, dtype=np.float64).astype(">f8", copy=False).tobytes(order="C"))
    expected_packet = {
        "schema": "step41-affine-query-certificate-v1",
        "deployment_id": deployment["deployment_id"], "query_id": query_id,
        "feature_sha256": hashlib.sha256(feature_bytes).hexdigest(),
        "reported_prediction_hex": reported_hex, "reference_interval_hex": outward_pair(reference),
        "absolute_error_upper_hex": high_error,
    }
    require(packet == expected_packet, "Certificate packet identity or tight arithmetic fields differ")
    accepted = error <= F(span)/2**40
    require(allow_over_budget or accepted, "Certificate is self-consistent but exceeds the declared error budget")
    return {"absolute_error": str(error), "budget": str(F(span)/2**40), "accepted": accepted,
            "packet_bytes": len(json.dumps(packet, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=True, allow_nan=False).encode())}


def reference_path(reference):
    path = safe_path(reference["path"])
    require(path.is_file() and sha(path) == reference["sha256"] and
            ("bytes" not in reference or path.stat().st_size == reference["bytes"]),
            "Referenced artifact changed: " + reference["path"])
    return path


def input_stage(task, stage):
    require(task in TASKS and stage in STAGES, "Unknown task or stage")
    manifest = read(ROOT/"inputs/manifest42.json")
    spec = manifest["tasks"][task]["stages"][stage]
    require(spec["map_order"] == expected_maps(), "Stage map roster differs")
    train = read(reference_path(spec["train"]["cohort"]))
    evaluation = read(reference_path(spec["evaluation"]["cohort"]))
    normal = read(reference_path(spec["normalization"]))
    require(normal["fitted_partition"] == "train" and normal["future_data_used"] is False and
            F(normal["span"]) == max(map(F, train["raw_targets"])) - min(map(F, train["raw_targets"])) and
            F(normal["target_min"]) == min(map(F, train["raw_targets"])) and F(normal["span"]) > 0,
            "Normalization differs from the inherited training-only target range")
    matrices = {}
    for name, meta in spec["maps"].items():
        tr = feature_matrix(np.load(reference_path(meta["training_features"]), allow_pickle=False))
        ev = feature_matrix(np.load(reference_path(meta["evaluation_features"]), allow_pickle=False))
        require(tr.shape == (train["n"], meta["dimension"]) and
                ev.shape == (evaluation["n"], meta["dimension"]), "Feature/cohort dimension mismatch")
        matrices[name] = (tr, ev)
    return {"spec": spec, "train": train, "evaluation": evaluation,
            "normalization": normal, "matrices": matrices,
            "legacy_predictions": read(reference_path(spec["inherited_predictions"])),
            "legacy_scores": read(reference_path(spec["inherited_scores"])),
            "legacy_selection": read(reference_path(spec["inherited_selection"]))}


def phase_seal(directory, phase):
    path = directory/(phase + "_seal42.json")
    seal = read(path)
    require(seal["schema"] == "step42-" + phase + "-seal-v1" and
            seal["protocol_seal_sha256"] == sha(ROOT/"protocol_seal42.json"),
            "Phase seal ancestry differs")
    actual = {str(x.relative_to(ROOT)) for x in directory.rglob("*") if x.is_file() and x != path}
    require(set(seal["bound_files"]) == actual, "Phase file closure differs")
    for relative, wanted in seal["bound_files"].items():
        require(sha(safe_path(relative)) == wanted, "Phase artifact changed: " + relative)
    start = read(directory/(phase + "_started42.json"))
    finish = read(directory/(phase + "_completed42.json"))
    require(start["protocol_seal_sha256"] == sha(ROOT/"protocol_seal42.json") and
            start["study_status"] == "DIAGNOSTIC_ALREADY_EXPOSED_PERIODS" and
            start["phase"] == finish["phase"] == phase and finish["status"] == "COMPLETE",
            "Phase completion or diagnostic status differs")
    frozen_time = datetime.fromisoformat(read(ROOT/"protocol_seal42.json")["sealed_utc"])
    require(frozen_time <= datetime.fromisoformat(start["utc"]) <= datetime.fromisoformat(finish["utc"]) <=
            datetime.fromisoformat(seal["utc"]), "Phase execution chronology precedes its freeze or is reversed")
    return seal, start, finish


def gate(path, *, expected_seal=None):
    record = read(path)
    require(record["status"] == "PASS" and
            record["protocol_seal_sha256"] == sha(ROOT/"protocol_seal42.json"),
            "Required independent gate is missing or stale")
    if expected_seal is not None:
        require(record["phase_seal_sha256"] == sha(expected_seal), "Independent gate binds a stale phase")
    for relative, wanted in record.get("source_bindings", {}).items():
        require(sha(safe_path(relative)) == wanted, "Audited dependency changed: " + relative)
    return record


def record_gate(destination, label, bindings, **fields):
    record = {"schema": "step42-independent-" + label + "-audit-v1", "status": "PASS",
              "protocol_seal_sha256": sha(ROOT/"protocol_seal42.json"),
              "input_manifest_sha256": sha(ROOT/"inputs/manifest42.json"),
              "auditor_sha256": sha(Path(__file__)), "study_status": "DIAGNOSTIC_ALREADY_EXPOSED_PERIODS",
              "scientific_candidate_searches_by_auditor": 0,
              "new_generalization_success_claim": False,
              "source_bindings": {str(Path(path).relative_to(ROOT)): sha(path) for path in bindings},
              "utc": datetime.now(timezone.utc).isoformat(), **fields}
    save(destination, record)
    return record


def preflight():
    protocol, seal = protocol_check()
    from check_inputs42 import Check
    independent_inputs = Check(ROOT).run()
    retained = read(ROOT/"audit/input_audit42.json")
    require(independent_inputs["status"] == retained["status"] == "PASS" and
            independent_inputs["input_manifest_sha256"] == retained["input_manifest_sha256"] and
            independent_inputs["checker_sha256"] == retained["checker_sha256"],
            "Independent prepared-input audit has changed")
    require(not (ROOT/"output/result42.json").exists(), "Diagnostic result already exists before preflight")
    for task, stage in product(TASKS, STAGES):
        require(not (ROOT/"output"/task/stage/"stage_started42.json").exists(),
                "A diagnostic stage ran before its preflight gate")
    return record_gate(ROOT/"audit/preexecution_gate42.json", "preexecution",
                       [ROOT/"protocol42.json", ROOT/"inputs/manifest42.json", ROOT/"audit/input_audit42.json"],
                       protocol_bound_files=len(seal["bound_files"]),
                       independent_input_checks=independent_inputs["checks_count"],
                       configurations_per_stage=720, selectors_per_task=177,
                       source_outcomes_already_exposed=True)


def audit_stage(task, stage):
    protocol_check()
    gate(ROOT/"audit/preexecution_gate42.json")
    context = input_stage(task, stage)
    spec, train, evaluation, normal = (context[x] for x in ("spec", "train", "evaluation", "normalization"))
    directory = ROOT/"output"/task/stage
    seal, started, finished = phase_seal(directory, "stage")
    require(started["task"] == finished["task"] == task and
            started["stage"] == finished["stage"] == stage, "Stage phase identity differs")
    if stage == "final":
        selected_gate = gate(ROOT/"audit"/task/"selection_gate42.json",
                             expected_seal=ROOT/"output"/task/"selection/selection_seal42.json")
        require(datetime.fromisoformat(selected_gate["utc"]) <= datetime.fromisoformat(started["utc"]),
                "Final stage preceded its independently checked selection")
    names = ("bases42.json", "variants42.json", "predictions42.json", "scores42.json", "legacy_recovery42.json")
    documents = {name: read(directory/name) for name in names}
    expected_header = {"task": task, "stage": stage, "role": spec["evaluation_role"],
                       "input_manifest_sha256": sha(ROOT/"inputs/manifest42.json")}
    for name, document in documents.items():
        require(all(document[k] == v for k, v in expected_header.items()), "Stage document header differs: " + name)
    configs = expected_configurations()
    candidate_ids = [x["candidate_id"] for x in configs]
    expected_base_keys = list(dict.fromkeys(x["base_key"] for x in configs))
    bases = documents["bases42.json"]["bases"]
    variants = documents["variants42.json"]["variants"]
    predictions = unique_rows(documents["predictions42.json"]["rows"])
    scores = unique_rows(documents["scores42.json"]["rows"])
    require(set(bases) == set(expected_base_keys) and len(bases) == 90 and
            set(variants) == set(predictions) == set(scores) == set(candidate_ids),
            "Stage omitted, duplicated, or added a factorial candidate")
    require([x["candidate_id"] for x in documents["predictions42.json"]["rows"]] == candidate_ids and
            [x["candidate_id"] for x in documents["scores42.json"]["rows"]] == candidate_ids,
            "Candidate reporting order differs from the frozen grid")
    require(documents["predictions42.json"]["query_ids"] == evaluation["ids"] and
            documents["scores42.json"]["n"] == evaluation["n"], "Query or score roster differs")
    prior_rows = unique_rows(context["legacy_scores"]["candidate_rows"])
    base_audits = {}
    exact_row_cache = {name: [integer_features(row) for row in arrays[1]]
                       for name, arrays in context["matrices"].items()}
    legacy_heads = 0
    legacy_predictions = 0
    max_error = F(0)
    for cfg in configs:
        cid, name, base_key = cfg["candidate_id"], cfg["map"], cfg["base_key"]
        meta = spec["maps"][name]
        tr, ev = context["matrices"][name]
        base = bases[base_key]
        if base_key not in base_audits:
            binding = {"task": task, "stage": stage,
                       "cohort_path": spec["train"]["cohort"]["path"],
                       "cohort_sha256": spec["train"]["cohort"]["sha256"],
                       "feature_path": meta["training_features"]["path"],
                       "feature_sha256": meta["training_features"]["sha256"],
                       "map_sha256": meta["map"]["sha256"]}
            base_audits[base_key] = audit_base(base, tr, train["y"], feature_id=meta["feature_id"],
                                              ridge_lambda=cfg["ridge_lambda"], training_binding=binding)
            if cfg["lambda_token"] == "1e-6":
                prior_fit = read(reference_path(meta["inherited_fit"]))
                original = prior_fit["projection"]
                require(base["coefficients"] == original["source_coefficients"] and
                        base["intercept"] == original["source_intercept"] and
                        base["center_features"] == original["center_features"] and
                        base["center_target"] == original["center_target"],
                        "Inherited ridge source before projection/recentering changed")
        variant = variants[cid]
        head = audit_variant(base, variant, project=cfg["project"], recenter=cfg["recenter"], clip=cfg["clip"])
        prepared = prepare_exact(head)
        independently = [prediction_reference(prepared, row, normal["target_min"], normal["span"], integers)
                          for row, integers in zip(ev, exact_row_cache[name])]
        prediction = predictions[cid]
        expected_arrays = {
            "reported_physical_hex": [x["reported_physical_hex"] for x in independently],
            "raw_normalized_hex": [float(x["raw_float"]).hex() for x in independently],
            "exact_physical": [x["exact_physical"] for x in independently],
            "exact_raw_normalized": [str(x["raw_exact"]) for x in independently],
            "clipping_active": [x["operational_clipping_active"] for x in independently],
            "clipping_side": [x["operational_clipping_side"] for x in independently],
            "exact_clipping_active": [x["exact_clipping_active"] for x in independently],
            "exact_clipping_side": [x["exact_clipping_side"] for x in independently],
        }
        require(prediction == {"candidate_id": cid, **expected_arrays}, "Independent query arithmetic differs: " + cid)
        physical = exact_score(expected_arrays["reported_physical_hex"], evaluation["raw_targets"], hex_values=True)
        exact = exact_score(expected_arrays["exact_physical"], evaluation["raw_targets"])
        row_error = max(x["absolute_error"] for x in independently)
        max_error = max(max_error, row_error)
        transformation = variant["transformation"]
        expected_score = {**cfg, "physical_risk": physical, "exact_reference_risk": exact,
                          "projection_active": transformation["projection_active"],
                          "projection_would_activate": F(transformation["source_squared_norm"]) > 4,
                          "clipping_count": sum(expected_arrays["clipping_active"]),
                          "exact_clipping_count": sum(expected_arrays["exact_clipping_active"]),
                          "max_abs_numerical_error": str(row_error),
                          "coefficient_squared_norm": transformation["coefficient_squared_norm"],
                          "source_squared_norm": transformation["source_squared_norm"],
                          "recenter_delta": transformation["recenter_delta"],
                          "structural_claims": variant["structural_claims"]}
        require(scores[cid] == expected_score, "Independent exact scoring or activation differs: " + cid)
        if cfg["lambda_token"] == "1e-6" and cfg["project"] and cfg["recenter"] and cfg["clip"]:
            old_head = read(reference_path(meta["inherited_head"]))
            require(all(head[key] == old_head[key] for key in
                        ("coefficients", "intercept", "dimension", "feature_id", "output_interval")),
                    "Inherited admissible head is not recovered exactly")
            old_prediction = context["legacy_predictions"]["predictions"][name+"::affine"]
            require([x["reported_physical_hex"] for x in old_prediction] == expected_arrays["reported_physical_hex"] and
                    [x["exact_physical"] for x in old_prediction] == expected_arrays["exact_physical"],
                    "Inherited operational/reference predictions are not recovered exactly")
            old_score = prior_rows[name+"::affine"]
            old_key = "physical_test_risk" if stage == "final" else "physical_risk"
            old_exact_key = "exact_reference_test_risk" if stage == "final" else "exact_reference_risk"
            require(old_score[old_key] == physical and old_score[old_exact_key] == exact,
                    "Inherited exact empirical errors changed")
            legacy_heads += 1
            legacy_predictions += len(old_prediction)
    recovery = documents["legacy_recovery42.json"]
    require(recovery == {**expected_header, "status": "PASS", "head_count": legacy_heads,
                         "prediction_count": legacy_predictions, "exact_head_and_prediction_recovery": True},
            "Inherited recovery summary differs")
    require(legacy_heads == 18 and legacy_predictions == 18*evaluation["n"] and
            finished["base_solves"] == 90 and finished["variants"] == 720 and
            finished["evaluation_predictions"] == 720*evaluation["n"] and
            finished["legacy_heads"] == legacy_heads and finished["legacy_predictions"] == legacy_predictions,
            "Stage completion counts differ")
    bindings = [directory/name for name in names] + [directory/"stage_seal42.json"]
    return record_gate(ROOT/"audit"/task/stage/"stage_gate42.json", "stage", bindings,
                       task=task, stage=stage, phase_seal_sha256=sha(directory/"stage_seal42.json"),
                       verification_solve_replays=90, variant_heads_verified=720,
                       prediction_pairs_independently_verified=720*evaluation["n"],
                       inherited_heads_recovered=legacy_heads, inherited_predictions_recovered=legacy_predictions,
                       maximum_absolute_numerical_discrepancy=str(max_error),
                       maximum_scaled_normal_equation_residual=max(x["scaled_normal_equation_residual"] for x in base_audits.values()))


def audit_selection(task):
    protocol_check()
    directory = ROOT/"output"/task/"selection"
    phase, started, finished = phase_seal(directory, "selection")
    folds = {}
    bindings = []
    for stage in ("f1", "f2", "f3"):
        previous = gate(ROOT/"audit"/task/stage/"stage_gate42.json",
                        expected_seal=ROOT/"output"/task/stage/"stage_seal42.json")
        require(datetime.fromisoformat(previous["utc"]) <= datetime.fromisoformat(started["utc"]),
                "Selection preceded an independent development audit")
        path = ROOT/"output"/task/stage/"scores42.json"
        folds[stage] = read(path)
        bindings.append(path)
    context = input_stage(task, "f3")
    spec = context["spec"]
    selection_path = ROOT/"output"/task/"selection42.json"
    selection = read(selection_path)
    binding = read(directory/"selection_binding42.json")
    require(reference_path(binding) == selection_path, "Selection external binding differs")
    n = sum(x["n"] for x in folds.values())
    indexed = {stage: unique_rows(x["rows"]) for stage, x in folds.items()}
    expected_rows = []
    for cfg in expected_configurations():
        cid = cfg["candidate_id"]
        sse = sum((F(indexed[s][cid]["physical_risk"]["sse"]) for s in folds), F(0))
        exact_sse = sum((F(indexed[s][cid]["exact_reference_risk"]["sse"]) for s in folds), F(0))
        expected_rows.append({**cfg, "n": n, "pooled_physical_sse": str(sse),
                              "pooled_physical_mse": str(sse/n), "pooled_exact_reference_sse": str(exact_sse),
                              "pooled_exact_reference_mse": str(exact_sse/n),
                              "legacy_f3_deployment_bytes": spec["maps"][cfg["map"]]["original_deployment_payload_bytes"],
                              "folds": {stage: {"n": data["n"], "physical_risk": indexed[stage][cid]["physical_risk"],
                                                "exact_reference_risk": indexed[stage][cid]["exact_reference_risk"],
                                                "projection_active": indexed[stage][cid]["projection_active"],
                                                "clipping_count": indexed[stage][cid]["clipping_count"]}
                                        for stage, data in folds.items()}})
    require(selection["candidate_rows"] == expected_rows, "Development exact pooled score table differs")
    expected_selections = {}
    for rule in expected_specs():
        pool = [row for row in expected_rows if all(row[k] == v for k, v in rule["fixed"].items())]
        require(len(pool) == rule["pool_size"], "Independently generated selector pool is incomplete")
        chosen = choose(pool, legacy=rule["label"] == "original41_replay")
        expected_selections[rule["label"]] = {"candidate_id": chosen["candidate_id"], "pool_size": len(pool),
                                               "pooled_physical_mse": chosen["pooled_physical_mse"],
                                               "rule": rule["tie_rule"], "fixed": rule["fixed"]}
    require(selection["selections"] == expected_selections, "Development winner, exact risk, pool, or tie rule differs")
    old = context["legacy_selection"]["selections"]["affine_rolling"]
    replay = expected_selections["original41_replay"]
    require(replay["candidate_id"].split("::")[0]+"::affine" == old["candidate_id"] and
            replay["pooled_physical_mse"] == old["pooled_physical_mse"], "Original Step41 selection was not recovered")
    require(selection["schema"] == "step42-development-selection-v1" and selection["task"] == task and
            selection["study_status"] == "DIAGNOSTIC_ALREADY_EXPOSED_PERIODS" and
            selection["development_n"] == n and selection["selection_uses_development_only"] is True and
            selection["assessment_reselection"] is False and
            selection["assessment_outcomes_previously_known_from_step41"] is True and
            selection["legacy_selector_recovered"] is True and
            selection["fold_score_sha256"] == {s: sha(ROOT/"output"/task/s/"scores42.json") for s in folds},
            "Selection chronology, status, or score bindings differ")
    require(started["task"] == finished["task"] == task and finished["selectors"] == 177 and
            finished["selected_main"] == expected_selections["recipe_P1R1C1"]["candidate_id"],
            "Selection completion summary differs")
    return record_gate(ROOT/"audit"/task/"selection_gate42.json", "selection",
                       bindings+[selection_path, directory/"selection_seal42.json"],
                       task=task, phase_seal_sha256=sha(directory/"selection_seal42.json"),
                       development_n=n, pooled_candidates_verified=720, selectors_independently_verified=177,
                       legacy_selector_recovered=True, assessment_reselection=False)


def audit_bridge(task):
    protocol, _ = protocol_check()
    directory = ROOT/"output"/task/"certificate_bridge"
    seal, started, finished = phase_seal(directory, "bridge")
    final_gate = gate(ROOT/"audit"/task/"final/stage_gate42.json",
                      expected_seal=ROOT/"output"/task/"final/stage_seal42.json")
    require(datetime.fromisoformat(final_gate["utc"]) <= datetime.fromisoformat(started["utc"]),
            "Certificate construction preceded the independent final head audit")
    context = input_stage(task, "final")
    spec, normal = context["spec"], context["normalization"]
    selected = read(ROOT/"output"/task/"selection42.json")["selections"]
    labels = ["original41_replay", "recipe_P1R1C1", "recipe_P1R0C1", "raw_recipe_P1R1C1",
              "nystrom_recipe_P1R1C1", "kahm_recipe_P1R1C1"]
    require(protocol["assurance"]["certificate_bridge_selectors"] == labels, "Certificate bridge selector contract differs")
    aliases = {label: selected[label]["candidate_id"] for label in labels}
    records = read(directory/"certificates42.json")
    require(records["schema"] == "step42-inherited-certificate-bridge-v1" and records["task"] == task and
            records["aliases"] == aliases and set(records["records"]) == set(aliases.values()) and
            records["future_label_coverage"] is False and records["native_transform_error_enclosed"] is False,
            "Certificate bridge roster or scope differs")
    inherited_source = ROOT/"inputs/prior41/source/certificate41.py"
    copied_source = ROOT/"source/legacy42/certificate41.py"
    require(records["certificate_source_sha256"] == sha(copied_source) == sha(inherited_source),
            "Certificate bridge modified the inherited verifier")
    variants = read(ROOT/"output"/task/"final/variants42.json")["variants"]
    predictions = unique_rows(read(ROOT/"output"/task/"final/predictions42.json")["rows"])
    n = context["evaluation"]["n"]
    errors, packets, accepted_packets = [], 0, 0
    for cid in sorted(set(aliases.values())):
        record = records["records"][cid]
        name = cid.split("::")[0]
        meta = spec["maps"][name]
        require(record["candidate_id"] == cid and record["map"] == name and
                record["head"] == record["deployment"]["head"] and
                record["aliases"] == [label for label, value in aliases.items() if value == cid] and
                record["query_ids"] == context["evaluation"]["ids"] and
                record["physical_error_budget"] == str(F(normal["span"])/2**40) and len(record["rows"]) == n,
                "Certificate selected-head metadata differs")
        for i, (row, z, query_id) in enumerate(zip(record["rows"], context["matrices"][name][1], context["evaluation"]["ids"])):
            require(row["query_id"] == query_id and row["reported_physical_hex"] == predictions[cid]["reported_physical_hex"][i]
                    and row["exact_physical"] == predictions[cid]["exact_physical"][i], "Certificate changed a retained prediction")
            checked = audit_legacy_packet(variants[cid], record["deployment"], row["certificate"], z,
                                           query_id=query_id, reported_hex=row["reported_physical_hex"],
                                           target_min=normal["target_min"], span=normal["span"],
                                           map_sha256=meta["map"]["sha256"], map_bytes=meta["map_payload_bytes"],
                                           candidate_id=cid, allow_over_budget=True)
            error = F(checked["absolute_error"])
            verification = {"schema": "step41-affine-query-verification-v1", "status": "PASS",
                            "deployment_id": record["deployment"]["deployment_id"], "query_id": query_id,
                            "feature_sha256": row["certificate"]["feature_sha256"],
                            "exact_reference_physical": row["exact_physical"],
                            "exact_reported_physical": str(F.from_float(from_hex(row["reported_physical_hex"]))),
                            "absolute_error_exact": str(error), "normalized_error_exact": str(error/F(normal["span"])),
                            "physical_error_budget": checked["budget"], "normalized_error_budget": "1/1099511627776",
                            "verified_exact_error": True, "accepted_prediction_accuracy": True,
                            "target_label_accessed": False, "scope": record["deployment"]["scope"]}
            if not checked["accepted"]:
                verification = {"status": "REJECTED_OVER_BUDGET",
                                "reason": "Reported prediction exceeds the frozen discrepancy budget"}
            require(row["verification"] == verification and row["packet_json_bytes"] == checked["packet_bytes"],
                    "Certificate verification receipt or payload bytes differ")
            require(row["accepted"] is checked["accepted"] and row["expected_acceptance"] is checked["accepted"] and
                    row["absolute_physical_error"] == str(error), "Certificate acceptance does not match exact error budget")
            errors.append(error)
            packets += 1
            accepted_packets += int(checked["accepted"])
    require(records["all_query_certificates_verified_by_inherited_checker"] is (accepted_packets == packets),
            "Certificate aggregate acceptance status differs")
    require(started["task"] == finished["task"] == task and finished["unique_candidates"] == len(records["records"])
            and finished["packets"] == packets, "Certificate completion counts differ")
    return record_gate(ROOT/"audit"/task/"bridge_gate42.json", "certificate-bridge",
                       [directory/"certificates42.json", directory/"bridge_seal42.json",
                        ROOT/"output"/task/"final/variants42.json", ROOT/"output"/task/"final/predictions42.json"],
                       task=task, phase_seal_sha256=sha(directory/"bridge_seal42.json"),
                       unique_candidates_verified=len(records["records"]), packets_independently_verified=packets,
                       accepted_packets=accepted_packets, rejected_over_budget_packets=packets-accepted_packets,
                       maximum_absolute_error=str(max(errors)), physical_error_budget=str(F(normal["span"])/2**40),
                       inherited_certificate_source_unchanged=True)


def ordered_median(values):
    ordered = sorted(values)
    require(bool(ordered), "An effect summary has no contrasts")
    middle = len(ordered)//2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle-1]+ordered[middle])/2


def independent_effect_summary(records, reference_mse):
    reference = F(reference_mse)
    require(reference > 0 and bool(records), "Effect summary requires a positive fixed reference")
    differences = [F(x["mse_difference"]) for x in records]
    relative = [x/reference for x in differences]
    threshold = F(1, 10**10)
    return {"pairs": len(records), "exact_improved": sum(x < 0 for x in differences),
            "exact_equal": sum(x == 0 for x in differences), "exact_worsened": sum(x > 0 for x in differences),
            "beyond_display_threshold_improved": sum(x < -threshold for x in relative),
            "within_display_threshold": sum(abs(x) <= threshold for x in relative),
            "beyond_display_threshold_worsened": sum(x > threshold for x in relative),
            "mean_mse_difference": str(sum(differences, F(0))/len(differences)),
            "median_mse_difference": str(ordered_median(differences)),
            "mean_reference_percent": 100*float(sum(relative, F(0))/len(relative)),
            "median_reference_percent": 100*float(ordered_median(relative)),
            "largest_absolute_reference_percent": 100*float(max(abs(x) for x in relative)),
            "display_threshold_relative_to_reference_mse": str(threshold),
            "reference_mse": str(reference), "independent_replications": False}


def row_key(row, **replacement):
    coordinates = {key: row[key] for key in ("map", "lambda_token", "project", "recenter", "clip")}
    coordinates.update(replacement)
    return tuple(coordinates[key] for key in ("map", "lambda_token", "project", "recenter", "clip"))


def independent_contrasts(rows, reference_mse):
    lookup = {row_key(row): row for row in rows}
    require(len(lookup) == len(rows) == 720, "Contrast candidate grid is incomplete or duplicated")
    factors = {}
    for factor in ("project", "recenter", "clip"):
        pairs = []
        for off in rows:
            if off[factor]:
                continue
            on = lookup[row_key(off, **{factor: True})]
            pairs.append({"factor": factor, "map": off["map"], "lambda_token": off["lambda_token"],
                          "off_candidate_id": off["candidate_id"], "on_candidate_id": on["candidate_id"],
                          "other_switches": {k: off[k] for k in ("project", "recenter", "clip") if k != factor},
                          "mse_difference": str(F(on["physical_risk"]["mse"])-F(off["physical_risk"]["mse"])),
                          "exact_reference_mse_difference": str(F(on["exact_reference_risk"]["mse"])-F(off["exact_reference_risk"]["mse"])),
                          "projection_applied_in_pair": bool(on.get("projection_active", False) or off.get("projection_active", False)),
                          "projection_applied_fold_count": max(on.get("projection_active_fold_count", 0), off.get("projection_active_fold_count", 0))})
        require(len(pairs) == 360, "Incomplete paired factor comparisons")
        active = [row for row in pairs if row["projection_applied_in_pair"] or row["projection_applied_fold_count"] > 0]
        inactive = [row for row in pairs if not (row["projection_applied_in_pair"] or row["projection_applied_fold_count"] > 0)]
        factors[factor] = {"pairs": pairs, "summary": independent_effect_summary(pairs, reference_mse),
                           "projection_active_summary": independent_effect_summary(active, reference_mse) if active else None,
                           "projection_inactive_summary": independent_effect_summary(inactive, reference_mse) if inactive else None}
    interactions = []
    for origin in rows:
        if origin["project"] or origin["recenter"]:
            continue
        q00, q10, q01, q11 = [lookup[row_key(origin, project=p, recenter=r)]
                              for p, r in ((False, False), (True, False), (False, True), (True, True))]
        delta = F(q11["physical_risk"]["mse"])-F(q01["physical_risk"]["mse"])-F(q10["physical_risk"]["mse"])+F(q00["physical_risk"]["mse"])
        interactions.append({"map": origin["map"], "lambda_token": origin["lambda_token"], "clip": origin["clip"],
                             "candidate_ids_P0R0_P1R0_P0R1_P1R1": [x["candidate_id"] for x in (q00, q10, q01, q11)],
                             "mse_difference": str(delta), "definition": "MSE11-MSE01-MSE10+MSE00"})
    regularization = []
    for row in rows:
        if row["lambda_token"] == "1e-6":
            continue
        original = lookup[row_key(row, lambda_token="1e-6")]
        regularization.append({"map": row["map"], "lambda_token": row["lambda_token"],
                               "switches": {key: row[key] for key in ("project", "recenter", "clip")},
                               "reference_candidate_id": original["candidate_id"], "candidate_id": row["candidate_id"],
                               "mse_difference": str(F(row["physical_risk"]["mse"])-F(original["physical_risk"]["mse"]))})
    require(len(interactions) == 180 and len(regularization) == 576, "Interaction or lambda contrast grid is incomplete")
    return {"reference_mse": str(F(reference_mse)), "factor_contrasts": factors,
            "projection_recenter_interactions": {"rows": interactions, "summary": independent_effect_summary(interactions, reference_mse)},
            "regularization_contrasts": {"rows": regularization, "summary": independent_effect_summary(regularization, reference_mse)}}


def independent_clipping_ranges(contexts, predictions):
    records = []
    by_key = {row_key(cfg): cfg for cfg in expected_configurations()}
    for cfg in expected_configurations():
        if cfg["clip"]:
            continue
        off_id = cfg["candidate_id"]
        on_id = by_key[row_key(cfg, clip=True)]["candidate_id"]
        sums = {"inside": F(0), "outside": F(0)}
        counts = {"inside": 0, "outside": 0}
        for stage, context in contexts.items():
            low = F(context["normalization"]["target_min"])
            high = low + F(context["normalization"]["span"])
            raw_targets = [F(x) for x in context["evaluation"]["raw_targets"]]
            before, after = (predictions[stage][cid]["exact_physical"] for cid in (off_id, on_id))
            require(len(before) == len(after) == len(raw_targets), "Conditional clipping roster differs")
            for a, b, truth in zip(before, after, raw_targets):
                group = "inside" if low <= truth <= high else "outside"
                change = (F(b)-truth)**2-(F(a)-truth)**2
                require(group != "inside" or change <= 0, "Clipping raised exact squared loss for an in-range label")
                sums[group] += change
                counts[group] += 1
        records.append({"off_candidate_id": off_id, "on_candidate_id": on_id,
                        "map": cfg["map"], "lambda_token": cfg["lambda_token"],
                        "project": cfg["project"], "recenter": cfg["recenter"],
                        "inside_n": counts["inside"], "outside_n": counts["outside"],
                        "inside_exact_sse_difference": str(sums["inside"]),
                        "outside_exact_sse_difference": str(sums["outside"]),
                        "inside_clipping_never_increased_exact_squared_error": True})
    require(len(records) == 360, "Conditional clipping grid is incomplete")
    return records


def audit_aggregate():
    protocol, _ = protocol_check()
    result_path = ROOT/"output/result42.json"
    result = read(result_path)
    aggregate_seal = read(ROOT/"output/aggregate_seal42.json")
    require(aggregate_seal["schema"] == "step42-aggregate-seal-v1" and
            aggregate_seal["protocol_seal_sha256"] == sha(ROOT/"protocol_seal42.json") and
            set(aggregate_seal["bound_files"]) == {"output/result42.json"} |
                {f"output/{task}/contrasts42.json" for task in TASKS}, "Aggregate closure or ancestry differs")
    for name, wanted in aggregate_seal["bound_files"].items():
        require(sha(safe_path(name)) == wanted, "Aggregate artifact changed: " + name)
    prior = read(ROOT/"inputs/prior41/output/result41.json")
    require(result["schema"] == "step42-controlled-ablation-result-v1" and result["step"] == 42 and
            result["study_status"] == "DIAGNOSTIC_ALREADY_EXPOSED_PERIODS" and
            result["new_primary_generalization_claim"] is False and
            result["prior_step41_status"] == prior["primary_status"] == "NOT_ESTABLISHED" and
            result["protocol_seal_sha256"] == sha(ROOT/"protocol_seal42.json") and
            result["input_manifest_sha256"] == sha(ROOT/"inputs/manifest42.json") and
            result["counts"] == protocol["counts"] and set(result["tasks"]) == set(TASKS) and
            result["independent_global_audit_pending"] is True,
            "Global diagnostic scope, prior result, or source binding differs")
    source_bindings = [result_path, ROOT/"output/aggregate_seal42.json", ROOT/"inputs/prior41/output/result41.json"]
    science_gates = [ROOT/"audit/preexecution_gate42.json"]
    packet_count = accepted_count = 0
    for task in TASKS:
        contexts = {}
        for stage in STAGES:
            path = ROOT/"audit"/task/stage/"stage_gate42.json"
            gate(path, expected_seal=ROOT/"output"/task/stage/"stage_seal42.json")
            science_gates.append(path)
            contexts[stage] = input_stage(task, stage)
        for label, folder, phase in (("selection", "selection", "selection"), ("bridge", "certificate_bridge", "bridge")):
            path = ROOT/"audit"/task/(label+"_gate42.json")
            audited = gate(path, expected_seal=ROOT/"output"/task/folder/(phase+"_seal42.json"))
            require(datetime.fromisoformat(audited["utc"]) <= datetime.fromisoformat(result["completed_utc"]),
                    "Global result preceded its prerequisite independent gate")
            science_gates.append(path)
        selection = read(ROOT/"output"/task/"selection42.json")
        final = read(ROOT/"output"/task/"final/scores42.json")
        final_rows = unique_rows(final["rows"])
        prediction_rows = {stage: unique_rows(read(ROOT/"output"/task/stage/"predictions42.json")["rows"])
                           for stage in STAGES}
        development = []
        for row in selection["candidate_rows"]:
            folds = row["folds"]
            normal_mae = sum((F(x["physical_risk"]["mae"])*x["n"] for x in folds.values()), F(0))/row["n"]
            exact_mae = sum((F(x["exact_reference_risk"]["mae"])*x["n"] for x in folds.values()), F(0))/row["n"]
            development.append({**row,
                                "physical_risk": {"n": row["n"], "sse": row["pooled_physical_sse"],
                                                  "mse": row["pooled_physical_mse"], "rmse": sqrt(float(F(row["pooled_physical_mse"]))),
                                                  "mae": str(normal_mae)},
                                "exact_reference_risk": {"n": row["n"], "sse": row["pooled_exact_reference_sse"],
                                                         "mse": row["pooled_exact_reference_mse"], "rmse": sqrt(float(F(row["pooled_exact_reference_mse"]))),
                                                         "mae": str(exact_mae)},
                                "projection_active_fold_count": sum(x["projection_active"] for x in folds.values()),
                                "clipping_count": sum(x["clipping_count"] for x in folds.values())})
        old_test = prior["tasks"][task]
        ordinary_development = contexts["f3"]["legacy_selection"]["selections"]["ordinary_rolling"]["pooled_physical_mse"]
        ordinary_assessment = old_test["selections"]["ordinary_rolling"]["physical_test_risk"]["mse"]
        require(F(ordinary_development) > 0 and F(ordinary_assessment) > 0, "Fixed comparator has nonpositive MSE")
        contrasts_expected = {"schema": "step42-exact-descriptive-contrasts-v1", "task": task,
                              "study_status": "DIAGNOSTIC_ALREADY_EXPOSED_PERIODS",
                              "development": independent_contrasts(development, ordinary_development),
                              "assessment": independent_contrasts(final["rows"], ordinary_assessment)}
        contrasts_expected["development"]["clipping_by_target_range"] = independent_clipping_ranges(
            {stage: contexts[stage] for stage in ("f1", "f2", "f3")}, prediction_rows)
        contrasts_expected["assessment"]["clipping_by_target_range"] = independent_clipping_ranges(
            {"final": contexts["final"]}, prediction_rows)
        contrast_path = ROOT/"output"/task/"contrasts42.json"
        require(read(contrast_path) == contrasts_expected, "Independent paired contrasts or descriptive summaries differ: " + task)
        selected_expected = {}
        for rule in expected_specs():
            label = rule["label"]
            chosen = selection["selections"][label]
            row = final_rows[chosen["candidate_id"]]
            pool = [item for item in final["rows"] if all(item[key] == value for key, value in rule["fixed"].items())]
            require(len(pool) == rule["pool_size"], "Hindsight diagnostic pool differs")
            best = min(pool, key=lambda item: (F(item["physical_risk"]["mse"]), item["candidate_id"]))
            risk, reference = F(row["physical_risk"]["mse"]), F(ordinary_assessment)
            selected_expected[label] = {
                "selection": chosen, "configuration": {key: row[key] for key in expected_configurations()[0]},
                "assessment_risk": row["physical_risk"], "exact_reference_assessment_risk": row["exact_reference_risk"],
                "assessment_mse_ratio_to_original_selected_ordinary": float(risk/reference),
                "assessment_mse_reduction_percent_vs_original_selected_ordinary": 100*float(1-risk/reference),
                "assessment_within_original_five_percent_tolerance_descriptively": 20*risk <= 21*reference,
                "assessment_strictly_lower_than_original_selected_ordinary": risk < reference,
                "projection_active": row["projection_active"], "clipping_count": row["clipping_count"],
                "coefficient_squared_norm": row["coefficient_squared_norm"], "recenter_delta": row["recenter_delta"],
                "structural_claims": row["structural_claims"],
                "maximum_absolute_numerical_error": row["max_abs_numerical_error"],
                "assessment_rank_of_selected_in_pool": 1+sum(F(x["physical_risk"]["mse"]) < risk for x in pool),
                "hindsight_only": {"candidate_id": best["candidate_id"], "assessment_mse": best["physical_risk"]["mse"],
                                   "selection_regret_mse": str(risk-F(best["physical_risk"]["mse"])),
                                   "not_an_alternative_selected_predictor": True},
                "development_selected_mse": str(F(chosen["pooled_physical_mse"])),
            }
        require(selected_expected["original41_replay"]["assessment_risk"] ==
                old_test["selections"]["affine_rolling"]["physical_test_risk"], "Original Step41 primary risk changed")
        bridge_path = ROOT/"output"/task/"certificate_bridge/certificates42.json"
        bridge = read(bridge_path)
        all_bridge_rows = [row for record in bridge["records"].values() for row in record["rows"]]
        bridge_expected = {"selectors": bridge["aliases"], "unique_candidates": len(bridge["records"]),
                           "packets": len(all_bridge_rows), "accepted_packets": sum(row["accepted"] for row in all_bridge_rows),
                           "rejected_over_budget_packets": sum(not row["accepted"] for row in all_bridge_rows),
                           "all_verified": all(row["accepted"] for row in all_bridge_rows), "records_sha256": sha(bridge_path)}
        ranges = {}
        for stage, context in contexts.items():
            targets = [F(x) for x in context["evaluation"]["raw_targets"]]
            low = F(context["normalization"]["target_min"])
            high = low + F(context["normalization"]["span"])
            ranges[stage] = {"n": len(targets), "target_min": str(low), "target_max": str(high),
                             "below": sum(x < low for x in targets), "inside": sum(low <= x <= high for x in targets),
                             "above": sum(x > high for x in targets)}
        final_variants = read(ROOT/"output"/task/"final/variants42.json")["variants"]
        expected_task = {"assessment_n": final["n"], "development_n": selection["development_n"],
                         "target_range_counts": ranges, "selection_results": selected_expected,
                         "prior_comparators": {label: old_test["selections"][label] for label in
                                               ("ordinary_rolling", "ridge_rolling", "affine_rolling")},
                         "effect_summaries": {context: {factor: contrasts_expected[context]["factor_contrasts"][factor]["summary"]
                                                        for factor in ("project", "recenter", "clip")}
                                              for context in ("development", "assessment")},
                         "contrast_artifact": {"path": str(contrast_path.relative_to(ROOT)), "sha256": sha(contrast_path),
                                               "bytes": contrast_path.stat().st_size},
                         "certificate_bridge": bridge_expected,
                         "unique_final_head_ids": len({v["head_id"] for v in final_variants.values()}),
                         "all_720_assessment_configurations_retained": True}
        require(result["tasks"][task] == expected_task, "Global task summary, selected risk, oracle labeling, or range counts differ: " + task)
        source_bindings += [contrast_path, bridge_path, ROOT/"output"/task/"selection42.json", ROOT/"output"/task/"final/scores42.json"]
        packet_count += bridge_expected["packets"]
        accepted_count += bridge_expected["accepted_packets"]
    require(len(science_gates) == 13, "Independent prerequisite gate roster is incomplete")
    for path in science_gates:
        gate(path)
    return record_gate(ROOT/"audit/postrun_audit42.json", "global", source_bindings+science_gates,
                       result_sha256=sha(result_path), prior_step41_status="NOT_ESTABLISHED",
                       producer_base_solves=720, verification_solve_replays=720,
                       variant_heads_verified=5760, candidate_query_predictions_verified=251280,
                       inherited_heads_recovered=144, inherited_predictions_recovered=6282,
                       selectors_verified=354, paired_and_interaction_and_lambda_contrasts_verified=7344,
                       conditional_clipping_comparisons_verified=1440,
                       scientific_gate_count_including_global=14,
                       scientific_gates={str(path.relative_to(ROOT)): sha(path) for path in science_gates},
                       certificate_packets_verified=packet_count, accepted_certificate_packets=accepted_count,
                       rejected_over_budget_certificate_packets=packet_count-accepted_count,
                       verification_success_distinct_from_empirical_effects=True,
                       no_new_primary_generalization_claim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "stage", "selection", "bridge", "aggregate"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    if args.phase in ("stage", "selection", "bridge") and args.task is None:
        parser.error("--task is required for this phase")
    if args.phase == "stage" and args.stage is None:
        parser.error("--stage is required for a stage audit")
    try:
        if args.phase == "preflight":
            record = preflight()
        elif args.phase == "stage":
            record = audit_stage(args.task, args.stage)
        elif args.phase == "selection":
            record = audit_selection(args.task)
        elif args.phase == "bridge":
            record = audit_bridge(args.task)
        else:
            record = audit_aggregate()
    except Exception as exc:
        failure = {"status": "FAIL_STOP_RETAIN", "phase": args.phase, "task": args.task, "stage": args.stage,
                   "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                   "utc": datetime.now(timezone.utc).isoformat()}
        path = ROOT/"audit/failures"/f"{args.phase}_{args.task}_{args.stage}.json"
        if not path.exists():
            save(path, failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        raise SystemExit(2)
    print(json.dumps({"status": record["status"], "phase": args.phase, "task": args.task, "stage": args.stage,
                      "study_status": record["study_status"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

