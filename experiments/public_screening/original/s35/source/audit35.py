"""Independent Step35 audit of saved witnesses and execution provenance.

This verifier does not import the experiment's certificate producer. It checks
exact rational primal/dual witnesses, distance bounds,
deployed head membership, prediction enclosures and loss algebra directly.
Separately, it replays sealed representation and baseline payloads using their
pinned runtime to bind saved feature values to deployment artifacts; that is
reproducibility, not an independent ideal-real transform certificate. It never
fits or changes a scientific model. Preflight does not parse PM data.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
import gzip
import hashlib
import io
from itertools import combinations
import json
from math import isqrt
from pathlib import Path
import sys
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"


class AuditError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def q(value):
    require(not isinstance(value, bool), "Boolean used as a rational")
    return F.from_float(value) if isinstance(value, float) else F(value)


def canon(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, float):
        return str(F.from_float(value))
    if isinstance(value, dict):
        return {str(k): canon(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canon(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canon(value), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canon(value), indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")


def interval(record, expected=None):
    lo, hi = q(record["lower"]), q(record["upper"])
    require(lo <= hi, "Reversed interval")
    if "width" in record:
        require(q(record["width"]) == hi-lo, "Incorrect interval width")
    if expected is not None:
        require((lo, hi) == expected, "Interval differs from independent reconstruction")
    return lo, hi


def clip(value, lo, hi):
    return min(hi, max(lo, value))


def exact_sqrt(value, bits):
    value = q(value)
    require(value >= 0 and isinstance(bits, int) and 1 <= bits <= 512,
            "Invalid square-root input or precision")
    scale = 1 << bits
    floor = isqrt((value.numerator * scale * scale) // value.denominator)
    lo = F(floor, scale)
    hi = lo if lo*lo == value else F(floor+1, scale)
    require(lo*lo <= value <= hi*hi, "Independent square-root bracketing failed")
    return lo, hi, floor


def matrix(values):
    require(isinstance(values, list) and len(values) > 0, "Empty feature matrix")
    d = len(values[0])
    require(d > 0 and all(len(v) == d for v in values), "Ragged feature matrix")
    return [[q(v) for v in row] for row in values]


def squared(left, right):
    require(len(left) == len(right), "Distance dimension mismatch")
    return sum(((a-b)**2 for a, b in zip(left, right)), F(0))


def descriptor(feature_id, kap, bounds):
    return canon({"schema": "step35-stored-feature-decoder-class-v1",
        "feature_id": feature_id,
        "metric": "Euclidean_distance_on_supplied_feature_coordinates",
        "decoder_class": "all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space",
        "kappa": kap, "output_interval": bounds, "assurance": ASSURANCE})


def audit_distance_table(record):
    require(record["assurance"] == ASSURANCE, "Distance assurance changed")
    z = matrix(record["features_exact"])
    ids = record["query_ids"]
    require(len(z) <= 32 and len(ids) == len(z) and len(set(ids)) == len(ids),
            "Distance roster shape or identity failed")
    require(record["n"] == len(z) and record["dimension"] == len(z[0]),
            "Distance dimensions differ")
    require(record["feature_digest"] == digest(z), "Distance feature digest differs")
    require(record["input_digest"] == digest({"ids": ids, "features": z}),
            "Distance input digest differs")
    pairs = list(combinations(range(len(z)), 2))
    require(len(record["pairs"]) == len(pairs) == record["pair_count"],
            "Distance table is incomplete")
    result = {}
    for (i, j), row in zip(pairs, record["pairs"]):
        require(row["indices"] == [i, j] and row["query_ids"] == [ids[i], ids[j]],
                "Distance pair ordering differs")
        value = squared(z[i], z[j])
        require(q(row["squared_exact"]) == value, "Squared distance differs")
        bits = record["bits"]
        lo, hi, floor = exact_sqrt(value, bits)
        interval(row["distance"], (lo, hi))
        require(row["bits"] == bits and row["scaled_sqrt_floor"] == floor,
                "Distance integer witness differs")
        result[i, j] = (lo, hi)
    require(record.get("native_transform_error_enclosed") is False,
            "Stored-feature distances were relabeled as native enclosures")
    return z, ids, result


def audit_head(head):
    require(head["schema"] == "step35-stored-feature-head-v1" and
            head["assurance"] == ASSURANCE, "Head schema or assurance differs")
    kap = q(head["kappa"])
    lo, hi = map(q, head["output_interval"])
    require(kap >= 0 and lo < hi, "Invalid head sensitivity or range")
    cid = "class35:" + digest(descriptor(head["feature_id"], kap, [lo, hi]))
    require(head["class_id"] == cid, "Head class identity differs")
    require(isinstance(head["dimension"], int) and head["dimension"] > 0,
            "Head dimension differs")
    if head["kind"] == "clipped_affine":
        beta = list(map(q, head["coefficients"]))
        require(len(beta) == head["dimension"], "Affine coefficient width differs")
        require(sum((b*b for b in beta), F(0)) <= kap*kap,
                "Affine head exceeds the declared global Lipschitz bound")
        q(head["intercept"])
    elif head["kind"] == "symmetric_lipschitz_extension":
        anchors = matrix(head["anchor_features"])
        values = list(map(q, head["anchor_values"]))
        require(len(anchors) == len(values) == len(head["anchor_ids"]) <= 32,
                "Extension anchor count differs")
        require(len(set(head["anchor_ids"])) == len(anchors), "Repeated anchor IDs")
        require(len(anchors[0]) == head["dimension"] and all(lo <= v <= hi for v in values),
                "Extension anchor dimension or range differs")
        for i, j in combinations(range(len(anchors)), 2):
            # Squared exact inequality needs no approximate square root.
            require((values[i]-values[j])**2 <= kap*kap*squared(anchors[i], anchors[j]),
                    "Extension anchors do not satisfy the exact interpolation premise")
    else:
        raise AuditError("Unknown deployed head kind")
    return "head35:" + digest(head), kap, lo, hi


def audit_qp(record):
    require(record["schema"] == "step35-stored-feature-finite-qp-v1" and
            record["assurance"] == ASSURANCE, "QP schema or assurance differs")
    z, ids, distances = audit_distance_table(record["distance_table"])
    require(record["features_exact"] == canon(z) and record["query_ids"] == ids,
            "QP and distance-table rosters differ")
    y = list(map(q, record["targets_exact"]))
    n = len(z)
    require(len(y) == n == record["n"] and len(z[0]) == record["dimension"],
            "QP dimensions differ")
    kap = q(record["kappa"])
    lo, hi = map(q, record["output_interval"])
    require(kap >= 0 and lo < hi, "Invalid QP class")
    expected_descriptor = descriptor(record["feature_id"], kap, [lo, hi])
    require(record["class_descriptor"] == expected_descriptor and
            record["class_id"] == "class35:"+digest(expected_descriptor),
            "QP class identity differs")
    require(record["feature_digest"] == digest(z) and record["target_digest"] == digest(y),
            "QP feature or target digest differs")
    require(record["input_digest"] == digest({"ids": ids, "features": z,
            "targets": y, "class": expected_descriptor}), "QP input digest differs")
    require(record["kind"] in ("training", "class"), "Unknown QP role")
    endpoint = 0 if record["kind"] == "training" else 1
    dual = record["dual"]
    require(dual["distance_endpoint"] == ("lower" if endpoint == 0 else "upper"),
            "Incorrect dual distance direction")
    incidence = [F(0)] * n
    penalty = F(0)
    seen = set()
    grid = 1 << record["work"]["grid_bits"]
    for edge in dual["edges"]:
        i, j = edge["indices"]
        require((i, j) in distances and (i, j) not in seen, "Invalid/repeated dual edge")
        seen.add((i, j))
        lam = q(edge["lambda"])
        require((lam*grid).denominator == 1, "Multiplier left the frozen dyadic grid")
        distance = distances[i, j][endpoint]
        require(q(edge["distance_endpoint"]) == distance and
                q(edge["constraint_radius"]) == kap*distance,
                "Dual distance or radius differs")
        incidence[i] += lam
        incidence[j] -= lam
        penalty += kap*abs(lam)*distance
    minimizers = [clip(t-s/2, lo, hi) for t, s in zip(y, incidence)]
    terms = [(v-t)**2+s*v for v, t, s in zip(minimizers, y, incidence)]
    raw = sum(terms, F(0))-penalty
    lower = max(F(0), raw)
    require(list(map(q, dual["incidence"])) == incidence and
            list(map(q, dual["box_minimizers"])) == minimizers and
            list(map(q, dual["coordinate_minima"])) == terms,
            "Dual box minimization or incidence differs")
    require(q(dual["penalty"]) == penalty and q(dual["unclamped_sse"]) == raw and
            q(dual["sse"]) == lower and q(dual["mean"]) == lower/n,
            "Exact dual objective differs")
    require(dual["range_constraints_used_in_dual"] is True and
            dual["targets_clipped_in_loss"] is False, "Range-aware loss claim differs")
    primal = record["primal"]
    values = list(map(q, primal["values"]))
    require(len(values) == n and all(lo <= v <= hi for v in values),
            "Primal range or length failed")
    slacks = [kap*d[0]-abs(values[i]-values[j]) for (i, j), d in distances.items()]
    require(all(s >= 0 for s in slacks), "Conservative primal feasibility failed")
    upper = sum(((v-t)**2 for v, t in zip(values, y)), F(0))
    require(q(primal["sse"]) == upper and q(primal["mean"]) == upper/n,
            "Exact primal objective differs")
    require(q(primal["minimum_pair_slack"]) == min(slacks, default=F(0)),
            "Primal minimum slack differs")
    require(primal["pair_checks"] == len(distances), "Primal pair count differs")
    require(0 <= lower <= upper, "Primal-dual order failed")
    optimization = record["optimization"]
    gap = upper-lower
    require(q(optimization["gap_sse"]) == gap, "QP gap differs")
    interval(optimization["objective_bracket_sse"], (lower, upper))
    interval(optimization["objective_bracket_mean"], (lower/n, upper/n))
    resolved = record["fallback_reason"] is None and gap <= q(optimization["gap_target_sse"])
    require(optimization["gap_resolved"] == resolved, "Unfinished QP mislabeled as resolved")
    work = record["work"]
    require(0 <= work["accepted_updates"] <= work["coordinate_updates"] <= work["max_updates"],
            "Total coordinate budget exceeded")
    require(work["coordinate_updates"] == work["initialization_coordinate_attempts"]+
            work["cyclic_coordinate_attempts"] and work["initialization_included_in_coordinate_budget"] is True,
            "Initialization was omitted from the coordinate budget")
    checkpoints = work["checkpoints"]
    previous = None
    for checkpoint in checkpoints:
        current = (checkpoint["coordinate_updates"], q(checkpoint["dual_sse"]), q(checkpoint["primal_sse"]))
        require(current[1] <= current[2] and q(checkpoint["gap_sse"]) == current[2]-current[1],
                "Checkpoint bracket failed")
        if previous is not None:
            require(current[0] > previous[0] and current[1] >= previous[1] and current[2] <= previous[2],
                    "Checkpoint monotonicity failed")
        previous = current
    require(previous == (work["coordinate_updates"], lower, upper),
            "Final checkpoint does not match final witnesses")
    constant = clip(sum(y, F(0))/n, lo, hi)
    constant_sse = sum(((constant-t)**2 for t in y), F(0))
    require(q(record["constant_witness"]["value"]) == constant and
            q(record["constant_witness"]["sse"]) == constant_sse,
            "Initial constant witness differs")
    require(record["targets_clipped"] is False and
            record["targets_outside_predictor_range"] == sum(not lo <= t <= hi for t in y),
            "Target range accounting differs")
    for flag in ("native_transform_error_enclosed", "native_pipeline_roundoff_certified",
                 "population_risk_claim", "future_response_coverage_claim", "raw_input_lipschitz_claim"):
        require(record[flag] is False, "Unsupported assurance flag: "+flag)
    if record["kind"] == "class":
        interval(record["class_risk"], (lower/n, upper/n))
        require(record["deployable_head_returned"] is False and
                record["witness_uses_assessment_targets"] is True,
                "Diagnostic class witness was promoted to a deployed head")
    else:
        hid, _, _, _ = audit_head(record["head"])
        require(hid == record["head_id"] and record["head"]["anchor_features"] == canon(z) and
                record["head"]["anchor_values"] == canon(values) and
                record["head"]["anchor_ids"] == ids, "Training head does not match its primal")
    return {"kind": record["kind"], "n": n, "class_id": record["class_id"],
            "lower_sse": lower, "upper_sse": upper, "gap_resolved": resolved}


def audit_loss(predictions, targets, record):
    y = list(map(q, targets))
    require(len(predictions) == len(y) > 0, "Loss roster differs")
    losses = []
    for prediction, target in zip(predictions, y):
        lo, hi = interval(prediction)
        errors = (lo-target, hi-target)
        lower = F(0) if errors[0] <= 0 <= errors[1] else min(e*e for e in errors)
        upper = max(e*e for e in errors)
        losses.append((lower, upper))
    expected = tuple(sum((x[k] for x in losses), F(0))/len(y) for k in (0, 1))
    interval(record, expected)
    require(record["n"] == len(y) and record["targets_clipped"] is False,
            "Loss target accounting differs")
    require(len(record["per_row"]) == len(y), "Per-row loss length differs")
    for row, bounds in zip(record["per_row"], losses):
        interval(row, bounds)
    return expected


def audit_prediction(record):
    require(record["assurance"] == ASSURANCE, "Prediction assurance differs")
    head = record["head"]
    hid, kap, lo, hi = audit_head(head)
    require(hid == record["head_id"] and head["class_id"] == record["class_id"] and
            head["feature_id"] == record["feature_id"], "Prediction head identity differs")
    z = matrix(record["features_exact"])
    ids = record["query_ids"]
    require(record["n"] == len(z) == len(ids) == len(record["rows"]) and
            len(set(ids)) == len(ids) and record["dimension"] == len(z[0]) == head["dimension"],
            "Prediction roster shape differs")
    require(record["feature_digest"] == digest(z) and
            record["query_roster_digest"] == digest({"ids": ids, "features": z}),
            "Prediction feature digest differs")
    fallback = record["status"] == "VALID_PREDICTOR_RANGE_FALLBACK"
    if head["kind"] == "symmetric_lipschitz_extension":
        anchors = matrix(head["anchor_features"])
        values = list(map(q, head["anchor_values"]))
    else:
        beta = list(map(q, head["coefficients"]))
        intercept = q(head["intercept"])
    for i, (vector, row, prediction) in enumerate(zip(z, record["rows"], record["predictions"])):
        require(row["query_index"] == i and row["query_id"] == ids[i], "Prediction row order differs")
        if fallback:
            expected = (lo, hi)
        elif head["kind"] == "clipped_affine":
            value = intercept+sum((b*x for b, x in zip(beta, vector)), F(0))
            interval(row["raw_prediction"], (value, value))
            expected = (clip(value, lo, hi),)*2
        else:
            require(len(row["anchor_distances"]) == len(anchors), "Incomplete query-to-anchor distances")
            lower_cones, upper_cones = [], []
            for j, (anchor, value, distance) in enumerate(zip(anchors, values, row["anchor_distances"])):
                require(distance["anchor_index"] == j and distance["anchor_id"] == head["anchor_ids"][j],
                        "Query anchor ordering differs")
                sq = squared(vector, anchor)
                dlo, dhi, floor = exact_sqrt(sq, record["distance_bits"])
                require(q(distance["squared_exact"]) == sq and distance["scaled_sqrt_floor"] == floor,
                        "Query distance integer witness differs")
                interval(distance["distance"], (dlo, dhi))
                lower_cones.append((value-kap*dhi, value-kap*dlo))
                upper_cones.append((value+kap*dlo, value+kap*dhi))
            lower = tuple(max(v[k] for v in lower_cones) for k in (0, 1))
            upper = tuple(min(v[k] for v in upper_cones) for k in (0, 1))
            raw = tuple((lower[k]+upper[k])/2 for k in (0, 1))
            interval(row["lower_envelope"], lower)
            interval(row["upper_envelope"], upper)
            interval(row["raw_prediction"], raw)
            expected = tuple(clip(v, lo, hi) for v in raw)
        interval(row["prediction"], expected)
        interval(prediction, expected)
    require(len(record["predictions"]) == len(z), "Prediction count differs")
    expected_distances = 0 if fallback or head["kind"] == "clipped_affine" else len(z)*len(anchors)
    require(record["distance_evaluations"] == expected_distances, "Query distance work differs")
    losses = None
    if record["targets_exact"] is not None:
        y = list(map(q, record["targets_exact"]))
        require(record["target_digest"] == digest(y), "Prediction target digest differs")
        losses = audit_loss(record["predictions"], y, record["loss"])
        for row, loss in zip(record["rows"], record["loss"]["per_row"]):
            require(row["squared_loss"] == loss, "Row loss differs from aggregate loss")
        require(record["targets_outside_predictor_range"] == sum(not lo <= t <= hi for t in y),
                "Prediction target-range count differs")
    for flag in ("targets_clipped", "native_transform_error_enclosed", "native_pipeline_roundoff_certified",
                 "population_risk_claim", "future_response_coverage_claim", "raw_input_lipschitz_claim"):
        require(record[flag] is False, "Unsupported prediction assurance: "+flag)
    return {"n": len(z), "head_id": hid, "class_id": record["class_id"], "loss": losses}


def audit_affine_record(record):
    hid, kap, lo, hi = audit_head(record["head"])
    require(record["head_id"] == hid, "Affine wrapper head ID differs")
    projection = record["projection"]
    original = list(map(q, projection["source_coefficients"]))
    original_norm = sum((b*b for b in original), F(0))
    require(q(projection["source_squared_norm"]) == original_norm, "Source affine norm differs")
    factor = q(projection["factor"])
    require(0 <= factor <= 1 and list(map(q, record["head"]["coefficients"])) == [factor*b for b in original],
            "Affine projection factor or coefficients differ")
    actual_norm = sum(((factor*b)**2 for b in original), F(0))
    require(q(record["membership"]["coefficient_squared_norm"]) == actual_norm and
            q(record["membership"]["kappa_squared"]) == kap*kap,
            "Affine membership evidence differs")
    if projection["intercept_recentered"]:
        center = list(map(q, projection["center_features"]))
        expected = q(projection["center_target"])-sum((factor*b*x for b, x in zip(original, center)), F(0))
        require(q(record["head"]["intercept"]) == expected, "Affine recentering differs")
    else:
        require(q(record["head"]["intercept"]) == q(projection["source_intercept"]),
                "Affine intercept unexpectedly changed")
    return {"head_id": hid, "projected": factor != 1}


def audit_aggregation(record, registry):
    require(record["assurance"] == ASSURANCE, "Aggregation assurance differs")
    ids, z, y, lower_sse = [], [], [], F(0)
    for block in record["blocks"]:
        certificate = registry.get(block["certificate_digest"])
        require(certificate is not None, "Aggregation references an absent class witness")
        require(certificate["kind"] == "class" and certificate["class_id"] == record["class_id"] and
                certificate["feature_id"] == record["feature_id"], "Aggregation class differs")
        n = certificate["n"]
        require(block["n"] == n and block["query_ids"] == certificate["query_ids"] and
                q(block["lower"]) == q(certificate["dual"]["sse"])/n,
                "Aggregation block contribution differs")
        lower_sse += q(certificate["dual"]["sse"])
        ids.extend(certificate["query_ids"])
        z.extend(certificate["features_exact"])
        y.extend(certificate["targets_exact"])
    require(len(ids) == len(set(ids)) == record["n"] and record["query_ids"] == ids,
            "Aggregated blocks overlap or omit rows")
    require(record["feature_digest"] == digest(z) and record["target_digest"] == digest(y),
            "Aggregated feature or target digest differs")
    uppers = []
    for candidate in record["global_upper_candidates"]:
        if candidate["kind"] == "single_global_frozen_head":
            receipt = registry.get(candidate["prediction_receipt_digest"])
            require(receipt is not None, "Global upper references an absent prediction witness")
            require(receipt["class_id"] == record["class_id"] and receipt["query_ids"] == ids and
                    receipt["features_exact"] == z and receipt["targets_exact"] == y and
                    receipt["head_id"] == candidate["head_id"],
                    "Global upper is not one common in-class head on the full cohort")
            upper = q(receipt["loss"]["upper"])
        else:
            require(candidate["kind"] == "single_global_diagnostic_constant_witness" and
                    record["diagnostic_assessment_constant_included"] is True and
                    candidate["uses_assessment_targets"] is True and candidate["training_only_head"] is False,
                    "Unidentified aggregate upper witness")
            value = q(candidate["value"])
            upper = sum(((value-q(t))**2 for t in y), F(0))/len(y)
        require(q(candidate["upper"]) == upper, "Global upper loss differs")
        uppers.append(upper)
    require(uppers, "No common global upper witness")
    expected = (lower_sse/len(ids), min(uppers))
    interval(record["class_risk"], expected)
    require(record["block_primals_averaged_for_union_upper"] is False,
            "Block upper bounds were incorrectly averaged")
    return {"n": len(ids), "class_id": record["class_id"], "class_risk": expected}


def verify_seal(path, root):
    seal = read_json(path)
    require(isinstance(seal.get("bound_files"), dict) and seal["bound_files"], "Empty file seal")
    for relative, expected in seal["bound_files"].items():
        target = root / relative
        require(target.is_file() and sha(target) == expected, "Sealed file changed: "+relative)
    return seal


def audit_event_chain(path):
    previous, last_ns, events = None, None, []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        retained = dict(record)
        checksum = retained.pop("event_sha256")
        expected = hashlib.sha256((json.dumps(retained, sort_keys=True,
            separators=(",", ":"), allow_nan=False)+"\n").encode()).hexdigest()
        require(checksum == expected and record["previous_event_sha256"] == previous,
                "Execution event hash chain changed")
        require(last_ns is None or record["monotonic_ns"] > last_ns, "Event clock order changed")
        previous, last_ns = checksum, record["monotonic_ns"]
        events.append(record)
    return events


def independent_selfcheck():
    checks = {}
    for value, bits in [(F(0), 80), (F(9, 4), 80), (F(2), 80), (F(1, 3), 13)]:
        lo, hi, _ = exact_sqrt(value, bits)
        require(lo*lo <= value <= hi*hi and hi-lo <= F(1, 1 << bits), "Sqrt selfcheck failed")
    checks["integer_outward_sqrt_perfect_nonperfect_zero"] = True
    # A zero-distance pair with targets 0 and 1 has constant-fit optimum 1/2 SSE.
    targets, incidence = [F(0), F(1)], [F(-1), F(1)]
    minimizers = [clip(t-s/2, F(0), F(1)) for t, s in zip(targets, incidence)]
    dual = sum(((v-t)**2+s*v for v, t, s in zip(minimizers, targets, incidence)), F(0))
    require(minimizers == [F(1, 2)]*2 and dual == F(1, 2), "Collision dual selfcheck failed")
    checks["zero_distance_conflict_exact_dual"] = True
    floor = sum(((clip(t, F(0), F(1))-t)**2 for t in [F(-2), F(3)]), F(0))
    require(floor == 8, "Outside-range loss floor selfcheck failed")
    checks["outside_range_targets_retained"] = True
    # Singleton block optima are zero; their common union optimum is 1/4 MSE.
    require(F(0)+F(0) < (F(1, 2)**2+F(1, 2)**2)/2, "Union-upper counterexample failed")
    checks["block_upper_averaging_counterexample"] = True
    return checks


def preflight(root):
    seal_path = root / "protocol_seal35.json"
    seal = verify_seal(seal_path, root)
    required_sources = {"source/audit35.py", "source/data35.py", "source/finite_feature35.py",
                        "source/representations35.py", "source/run35.py", "protocol35.json"}
    require(required_sources <= set(seal["bound_files"]), "Protocol seal omits a required source")
    require(not (root / "output/development_data35.json").exists(),
            "Development numerical receipt already exists before preflight")
    require(not (root / "output/test_data35.json").exists(), "Test values were accessed before preflight")
    protocol = read_json(root / "protocol35.json")
    protocol_checks = audit_protocol(protocol)
    for relative in sorted(seal["bound_files"]):
        if relative.endswith(".py"):
            ast.parse((root / relative).read_text(), filename=relative)
    tests = []
    for path in [root / "tests/artificial_representation_preflight35.json",
                 root / "analysis/finite_feature_preflight35.json",
                 root / "tests/artificial_runner_preflight35.json"]:
        result = read_json(path)
        require(result.get("status") in ("PASS", "PASSED"), "Artificial preflight did not pass: "+path.name)
        require(str(path.relative_to(root)) in seal["bound_files"], "Artificial preflight was not sealed")
        tests.append({"path": str(path.relative_to(root)), "sha256": sha(path)})
    arithmetic = read_json(root / "analysis/finite_feature_preflight35.json")
    require(isinstance(arithmetic["source_sha256"], dict) and
            set(arithmetic["source_sha256"]) == {"finite_feature35.py", "check_finite_feature35.py"},
            "Arithmetic preflight source manifest has an unexpected schema")
    for filename, checksum in arithmetic["source_sha256"].items():
        require(checksum == sha(root / "source" / filename),
                "Arithmetic preflight used another source version: "+filename)
    artificial_counts = audit_receipt_collection(arithmetic)[0]
    representations = read_json(root / "tests/artificial_representation_preflight35.json")
    require(representations["factory_metadata"]["factory_source_sha256"] == sha(root / "source/representations35.py"),
            "Representation preflight used another producer version")
    require(representations["public_data_values_read"] == 0 and
            representations["patient_data_values_read"] == 0 and
            arithmetic["actual_public_data_rows_loaded"] == arithmetic["reserved_rows_opened"] == 0,
            "Artificial preflight touched actual scientific values")
    return {"schema": "step35-independent-preexecution-gate-v1", "status": "PASS",
        "utc": datetime.now(timezone.utc).isoformat(), "protocol_seal_sha256": sha(seal_path),
        "sealed_file_count": len(seal["bound_files"]), "artificial_preflights": tests,
        "independent_math_selfchecks": independent_selfcheck(),
        "independently_verified_artificial_receipt_counts": artificial_counts,
        "protocol_semantic_checks": protocol_checks,
        "source_review": {
            "range_aware_box_dual_and_distance_directions": "PASS",
            "conservative_primal_and_global_extension_membership": "PASS",
            "stored_binary64_feature_conditional_scope": "PASS",
            "training_only_scalers_probes_landmarks_metric_scales": "PASS",
            "causal_timestamp_based_train_decision_test_barriers": "PASS",
            "physical_native_fit_cost_includes_unused_regrouped_fits": "PASS",
            "common_global_head_required_for_union_upper": "PASS"},
        "public_pm_numeric_values_parsed_by_this_audit": False,
        "reserved_patient_arrays_accessed": 0,
        "limitations": ["A file seal and hash chain provide reproducible local provenance, not an external trusted timestamp.",
            "This preflight checks the frozen implementation and artificial cases; actual result witnesses require post-run audit."]}


RECEIPT_SCHEMAS = {
    "step35-stored-feature-finite-qp-v1": audit_qp,
    "step35-stored-feature-prediction-receipt-v1": audit_prediction,
    "step35-admissible-affine-head-v1": audit_affine_record,
}


def audit_protocol(p):
    require(p["step"] == 35 and p["seed"] == 3501, "Protocol step or seed differs")
    data = p["data"]
    require(data["uci_id"] == 381 and data["train_target_years"] == [2010, 2011, 2012] and
            data["decision_target_years"] == [2013] and data["test_target_years"] == [2014],
            "Declared data chronology differs")
    require(data["origin_hour"] == 23 and data["horizon_hours"] == data["lag_hours"] == 24 and
            data["targets_clipped"] is False and data["other_measured_variables_used"] is False,
            "Forecast target, input, or range policy differs")
    require(p["class"]["kappa"] == "2" and p["class"]["prediction_interval"] == ["0", "1"],
            "Fixed decoder class changed")
    require(p["heads"]["recipes_per_map"] == ["affine", "extension32"] and
            p["heads"]["extension32"]["anchor_count"] == 32 and
            q(p["heads"]["affine"]["ridge_lambda"]) == F(1, 1000000), "Fixed head panel changed")
    expected_representations = {"dimension": 24, "scalings": ["minmax01", "standard"],
        "nystrom_widths": [8, 24], "kahm_caps": [4, 12, 20], "kahm_prefixes": [1, 2],
        "cluster_counts": [8, 16], "max_fit": 512, "max_references": 24,
        "tau": 1e-6, "omega": 2.0, "probe_count": 32, "nystrom_eigen_floor": 1e-12,
        "temporal_selection": "equal_row_stratum_midpoints",
        "metric_normalization": "reciprocal_median_positive_pair_distance"}
    require(p["representations"] == expected_representations, "Fixed representation panel changed")
    cfg = p["certificate"]
    require((cfg["block_size"], cfg["distance_bits"], cfg["multiplier_grid_bits"], cfg["check_every"],
             cfg["class_max_coordinate_attempts_per_block"], cfg["head_max_coordinate_attempts"]) ==
            (32, 80, 40, 20, 400, 4000) and q(cfg["gap_sse"]) == F(1, 10000000),
            "Frozen certificate arithmetic or budget changed")
    require(cfg["average_block_primal_upper_permitted"] is False, "Invalid aggregate upper enabled")
    require(q(p["policy"]["epsilon_factor"]) == F(19, 20) and p["policy"]["no_test_selection"] is True,
            "Threshold or test-selection policy changed")
    scope = p["scientific_scope"]
    require(scope["assurance"] == ASSURANCE and scope["reserved_patients_total"] == 5557 and
            scope["reserved_patients_accessed"] == 0 and scope["reserved_patient_arrays_read"] is False,
            "Scientific scope or sealed cohort changed")
    for flag in ("native_transform_error_enclosed", "native_pipeline_roundoff_certified",
                 "population_risk_claim", "future_response_coverage_claim"):
        require(scope[flag] is False, "Unsupported protocol assurance: "+flag)
    require(p["execution"]["primary_run_count"] == 1 and p["execution"]["overall_wall_budget_seconds"] == 3600 and
            p["execution"]["native_family_wall_budget_seconds"] == 1800,
            "Primary run count or wall budget changed")
    require(p["baselines"]["baseline_in_threshold_cost_panel"] is False and
            p["baselines"]["hist_gradient_boosting"]["early_stopping"] is False,
            "Empirical baseline role changed")
    return {"fixed_causal_data_split": True, "same_18_map_36_head_panel": True,
            "frozen_threshold_metric_budgets": True, "finite_feature_conditional_scope": True,
            "reserved_patient_cohort_remains_sealed": True}


def iter_receipts(value):
    if isinstance(value, dict):
        schema = value.get("schema")
        if schema in RECEIPT_SCHEMAS or schema == "step35-aggregated-stored-feature-class-risk-v1":
            yield value
        else:
            for item in value.values():
                yield from iter_receipts(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_receipts(item)


def compact_receipt(receipt):
    # Prediction distance witnesses can be large. Keep only union-binding fields
    # after their independent proof checks; their full content remains on disk.
    keys = ("schema", "kind", "feature_id", "class_id", "head_id", "query_ids", "n",
            "features_exact", "targets_exact", "loss", "dual")
    result = {k: receipt[k] for k in keys if k in receipt}
    if "loss" in result:
        result["loss"] = {k: result["loss"][k] for k in ("lower", "upper", "width")}
    if "dual" in result:
        result["dual"] = {"sse": result["dual"]["sse"]}
    return result


def audit_receipt_collection(document):
    registry, aggregates, counts = {}, [], {}
    for receipt in iter_receipts(document):
        key = digest(receipt)
        if key in registry:
            continue
        schema = receipt["schema"]
        if schema == "step35-aggregated-stored-feature-class-risk-v1":
            aggregates.append(receipt)
        else:
            RECEIPT_SCHEMAS[schema](receipt)
        registry[key] = compact_receipt(receipt)
        counts[schema] = counts.get(schema, 0)+1
    for receipt in aggregates:
        audit_aggregation(receipt, registry)
    return counts, registry


def audit_cohorts(root, protocol):
    """Independent post-seal reconstruction from public integer PM observations."""
    import numpy as np
    archive = root / protocol["data"]["archive"]
    source = read_json(root / "inputs/public_source35.json")
    require(sha(archive) == source["archive_sha256"], "Public archive changed")
    observed = {}
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open("PRSA_data_2010.1.1-2014.12.31.csv") as handle:
            for row in csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig")):
                stamp = datetime(*(int(row[k]) for k in ("year", "month", "day", "hour")))
                require(stamp not in observed, "Duplicated hourly source timestamp")
                observed[stamp] = None if row["pm2.5"] == "NA" else int(row["pm2.5"])
    require(len(observed) == 43824, "Public row count differs")
    cohorts, expected = {}, {}
    for role, years in [("train", [2010, 2011, 2012]), ("decision", [2013]), ("test", [2014])]:
        ids, origins, targets, ys, xs = [], [], [], [], []
        missing_timestamp = missing_value = candidate_count = 0
        for target in sorted(observed):
            if target.year not in years or target.hour != 23:
                continue
            candidate_count += 1
            origin = target-timedelta(hours=24)
            history = [origin-timedelta(hours=23-j) for j in range(24)]
            if any(t not in observed for t in history+[target]):
                missing_timestamp += 1
                continue
            if any(observed[t] is None for t in history+[target]):
                missing_value += 1
                continue
            target_text = target.isoformat(timespec="hours")
            ids.append(role+":"+target_text)
            origins.append(origin.isoformat(timespec="hours"))
            targets.append(target_text)
            ys.append(observed[target])
            xs.append([observed[t] for t in history])
        saved = read_json(root / "output/cohorts" / (role+".json"))
        for key, value in [("ids", ids), ("origin_timestamps", origins), ("target_timestamps", targets), ("raw_targets", ys)]:
            require(saved[key] == value, "Cohort reconstruction differs: "+role+"/"+key)
        lag_path = root / saved["lag_matrix_path"]
        require(saved["lag_matrix_sha256"] == sha(lag_path), "Lag matrix file changed")
        X = np.load(lag_path, allow_pickle=False)
        require(X.dtype == np.float64 and np.array_equal(X, np.asarray(xs, dtype=np.float64)),
                "Saved causal lag matrix differs from public observations")
        roster = saved["roster"]
        require(roster["eligible"] == len(ids) and roster["candidate_daily_targets"] == candidate_count and
                roster["excluded"] == {"required_timestamp_absent": missing_timestamp, "missing_lag_or_target": missing_value},
                "Complete-case missingness accounting differs")
        cohorts[role] = saved
        expected[role] = {"X": X, "raw_targets": ys}
    low, high = min(expected["train"]["raw_targets"]), max(expected["train"]["raw_targets"])
    span = high-low
    require(span > 0, "Training target scale degenerate")
    for role, saved in cohorts.items():
        y = [F(t-low, span) for t in expected[role]["raw_targets"]]
        persistence = [F(int(t)-low, span) for t in expected[role]["X"][:, -1]]
        require(saved["y"] == list(map(str, y)) and saved["persistence"] == list(map(str, persistence)),
                "Exact target normalization or persistence differs")
        require(saved["roster"]["target_outside_prediction_range_count"] == sum(not 0 <= t <= 1 for t in y),
                "Outside-range target count differs")
    train = cohorts["train"]
    train_persistence = sum(((q(a)-q(b))**2 for a, b in zip(train["persistence"], train["y"])), F(0))/len(train["y"])
    epsilon = F(19, 20)*train_persistence
    plan = read_json(root / "output/training_plan35.json")
    indices = [((2*j+1)*len(train["ids"]))//64 for j in range(32)]
    require(q(plan["train_persistence_mse"]) == train_persistence and q(plan["epsilon"]) == epsilon and
            q(plan["epsilon_physical_mse"]) == epsilon*span*span, "Training-only threshold differs")
    require(plan["anchor_indices"] == indices and plan["anchor_ids"] == [train["ids"][i] for i in indices],
            "Training anchor roster differs")
    return cohorts, expected, epsilon, {"train_min": low, "train_max": high, "range": span}


def audit_panel(root, protocol, cohorts, expected, epsilon):
    import numpy as np
    from scipy.spatial.distance import pdist
    import representations35 as replay
    replay.initialize_runtime()
    manifest = read_json(root / "output/map_manifest35.json")
    metadata = read_json(root / "output/representations35.json")
    policy = read_json(root / "output/policy35.json")
    result = read_json(root / "output/result35.json")
    screen = read_json(root / "output/screen35.json")
    names = metadata["map_order"]
    require(len(names) == len(set(names)) == len(manifest) == 18, "18-map roster incomplete")
    require(metadata["config"] == protocol["representations"] and
            metadata["factory_source_sha256"] == sha(root / "source/representations35.py"),
            "Representation source/configuration binding differs")
    train_X = expected["train"]["X"]
    require(metadata["training_shape"] == list(train_X.shape), "Representation training shape differs")
    probes = [((2*j+1)*len(train_X))//64 for j in range(32)]
    require(metadata["probe_indices"] == probes, "Training metric probe roster differs")
    for mode, row in metadata["scalers"].items():
        low, high = train_X.min(axis=0), train_X.max(axis=0)
        constant = high == low
        offset = low if mode == "minmax01" else train_X.mean(axis=0)
        divisor = high-low if mode == "minmax01" else train_X.std(axis=0, ddof=0)
        divisor = np.where(constant, 1.0, divisor)
        require(np.array_equal(np.asarray(row["offset"]), offset) and
                np.array_equal(np.asarray(row["divisor"]), divisor) and
                row["constant_mask"] == constant.tolist() and row["training_rows"] == len(train_X),
                "Scaler was not fit on the declared training matrix")
        require(row["heldout_clipping"] is False, "Held-out input clipping enabled")
    meta_by_name = {row["name"]: row for row in metadata["maps"]}
    features, lower_by_role = {}, {"decision": {}, "test": {}}
    for name in names:
        info, map_meta = manifest[name], meta_by_name[name]
        payload = root / info["path"]
        require(sha(payload) == info["sha256"] == map_meta["payload_sha256"] and
                payload.stat().st_size == info["payload_bytes"] == map_meta["payload_bytes"] and
                info["feature_id"] == "step35-map-sha256:"+sha(payload),
                "Deployment map bytes or feature identity differs")
        restored = replay.deserialize_map(payload.read_bytes())
        require(restored.name == name and replay.numeric_bytes(restored) == info["numeric_bytes"],
                "Restored representation identity or stored numeric bytes differ")
        unscaled_probes = restored.core.transform(train_X[probes])
        distances = pdist(unscaled_probes, metric="euclidean")
        positive = distances[distances > 0]
        median = float(np.median(positive)) if len(positive) else 1.0
        metric = 1.0/median if len(positive) else 1.0
        require(float(restored.metric_scale) == metric == map_meta["metric_scale"] and
                map_meta["positive_probe_pair_count"] == len(positive),
                "Deployment metric scalar was not frozen from training probes")
        features[name] = {}
        for role in ("train", "decision", "test"):
            array = np.load(root / "output/features" / role / (name+".npy"), allow_pickle=False)
            require(array.dtype == np.float64 and np.isfinite(array).all() and
                    array.shape == (len(cohorts[role]["ids"]), map_meta["actual_width"]),
                    "Stored feature shape, dtype, or finiteness differs")
            replayed = replay.transform_one(restored, expected[role]["X"])
            require(np.array_equal(array, replayed),
                    "Sealed deployment payload does not reproduce stored features bitwise: "+name+"/"+role)
            features[name][role] = [[str(F.from_float(float(v))) for v in row] for row in array]
        if map_meta["family"] == "nystrom":
            width = map_meta["requested_width"]
            require(map_meta["landmark_indices"] == [((2*j+1)*len(train_X))//(2*width) for j in range(width)],
                    "Nyström training-only landmark roster differs")
        for role in ("decision", "test"):
            lower = read_json(root / "output/classes" / role / name / "lower.json")
            ids, z, y, sse = [], [], [], F(0)
            for index, relative in enumerate(lower["blocks"]):
                block = read_json(root / relative)
                require(block["feature_id"] == info["feature_id"] and block["kind"] == "class" and
                        block["work"]["max_updates"] == 400 and block["distance_table"]["bits"] == 80,
                        "Class source, role, or frozen budget differs")
                first, last = index*32, min((index+1)*32, len(cohorts[role]["ids"]))
                require(block["query_ids"] == cohorts[role]["ids"][first:last] and
                        block["features_exact"] == features[name][role][first:last] and
                        block["targets_exact"] == cohorts[role]["y"][first:last],
                        "Class block is not the declared consecutive cohort subset")
                ids += block["query_ids"]; z += block["features_exact"]; y += block["targets_exact"]
                sse += q(block["dual"]["sse"])
            require(ids == cohorts[role]["ids"], "Class blocks omit cohort rows")
            floor = sum((max(F(0), -q(t), q(t)-1)**2 for t in y), F(0))/len(y)
            total = max(floor, sse/len(y))
            require(q(lower["lower"]) == total and q(lower["dual_lower"]) == sse/len(y) and
                    q(lower["range_only_floor"]) == floor, "Whole-cohort lower aggregation differs")
            lower_by_role[role][name] = total
    excluded = [name for name in names if lower_by_role["decision"][name] > epsilon]
    require(screen["excluded_classes"] == policy["excluded_classes"] == excluded and
            q(screen["epsilon"]) == q(policy["epsilon"]) == epsilon,
            "Class exclusion or epsilon differs")
    require(screen["lower_by_map"] == {name: str(lower_by_role["decision"][name]) for name in names},
            "Screen risk table differs")
    rows = policy["candidate_rows"]
    all_ids = [name+"::"+recipe for name in names for recipe in ("affine", "extension32")]
    require([row["candidate_id"] for row in rows] == all_ids, "36-head finite roster incomplete")
    output_rows = {row["candidate_id"]: row for row in result["candidate_rows"]}
    require(set(output_rows) == set(all_ids), "Result panel differs from frozen panel")
    anchors = [((2*j+1)*len(cohorts["train"]["ids"]))//64 for j in range(32)]
    for row in rows:
        name, recipe = row["map"], row["recipe"]
        head = read_json(root / row["head_path"])
        fit = read_json(root / row["fit_path"])
        require(head == fit["head"] and row["head_id"] == fit["head_id"] and
                head["feature_id"] == row["feature_id"] == manifest[name]["feature_id"],
                "Deployment and fitted head bindings differ")
        map_size = (root / row["map_path"]).stat().st_size
        head_size = (root / row["head_path"]).stat().st_size
        require(row["map_payload_bytes"] == map_size and row["head_payload_bytes"] == head_size and
                row["deployment_payload_bytes"] == map_size+head_size,
                "Complete serialized deployment payload bytes differ")
        require(row["shadow_excluded_class"] == (name in excluded), "Shadow-fit role differs")
        if recipe == "extension32":
            require(head["anchor_ids"] == [cohorts["train"]["ids"][i] for i in anchors] and
                    head["anchor_features"] == [features[name]["train"][i] for i in anchors] and
                    fit["targets_exact"] == [cohorts["train"]["y"][i] for i in anchors] and
                    fit["work"]["max_updates"] == 4000,
                    "Deployed repair uses a non-training anchor/target or changed budget")
        else:
            train_z = matrix(features[name]["train"])
            center = [sum((z[j] for z in train_z), F(0))/len(train_z) for j in range(len(train_z[0]))]
            require(fit["projection"]["center_features"] == list(map(str, center)) and
                    q(fit["projection"]["center_target"]) == sum(map(q, cohorts["train"]["y"]), F(0))/len(train_z),
                    "Affine exact center did not use training rows only")
        output_row = output_rows[row["candidate_id"]]
        require(all(output_row[k] == v for k, v in row.items()), "Pre-test candidate row changed after seal")
        for role in ("decision", "test"):
            receipt = read_json(root / output_row[role+"_prediction_path"])
            require(receipt["head"] == head and receipt["query_ids"] == cohorts[role]["ids"] and
                    receipt["features_exact"] == features[name][role] and receipt["targets_exact"] == cohorts[role]["y"],
                    "Head assessment is not on the common stored feature/target cohort")
            interval(output_row[role+"_risk"], interval(receipt["loss"]))
    eligible = [r for r in rows if q(r["decision_risk"]["upper"]) <= epsilon]
    key = lambda r: (r["deployment_payload_bytes"], q(r["decision_risk"]["upper"]), r["candidate_id"])
    exhaustive = min(eligible, key=key, default=None)
    screened = min([r for r in eligible if r["map"] not in excluded], key=key, default=None)
    accuracy = min(rows, key=lambda r: (q(r["decision_risk"]["upper"]), r["deployment_payload_bytes"], r["candidate_id"]))
    expected_choices = {"screened": None if screened is None else screened["candidate_id"],
        "exhaustive_same_rule": None if exhaustive is None else exhaustive["candidate_id"],
        "ordinary_mse_minimizer": accuracy["candidate_id"]}
    require(all(policy[k]["candidate_id"] == v for k, v in expected_choices.items()) and
            expected_choices["screened"] == expected_choices["exhaustive_same_rule"],
            "Frozen selection differs from independent exhaustive policy reconstruction")
    require(all(r["map"] not in excluded for r in eligible), "Sound class screen removed an eligible head")
    frozen = read_json(root / "output/screened_policy35.json")
    require(frozen["selection"]["candidate_id"] == expected_choices["screened"] and
            frozen["candidate_ids"] == [r["candidate_id"] for r in rows if r["map"] not in excluded],
            "Screened choice was not frozen before shadows")
    avoided = [r for r in rows if r["map"] in excluded]
    avoided_seconds = sum(r["decoder_stage_seconds_including_retention"] for r in avoided)
    cert_seconds = sum(read_json(root / "output/classes/decision" / name / "lower.json")["stage_seconds_including_witness_retention"] for name in names)
    require(policy["avoided_decoder_fits_in_screened_policy"] == len(avoided) and
            policy["avoided_decoder_stage_seconds"] == avoided_seconds and
            policy["decision_class_seconds"] == cert_seconds and
            policy["net_conditional_stage_seconds_saved"] == avoided_seconds-cert_seconds and
            policy["positive_screening_utility_demonstrated"] == (len(avoided) > 0 and avoided_seconds > cert_seconds),
            "Screening utility or physical work accounting differs")
    require(result["primary_run_count"] == 1 and result["reserved_patients_accessed"] == 0 and
            result["reserved_patient_arrays_read"] is False and result["scientific_scope"] == protocol["scientific_scope"],
            "Result scope differs from frozen protocol")
    return {"map_count": len(names), "head_count": len(rows), "excluded_classes": excluded,
        "independent_policy_choices": expected_choices, "same_rule_choice_identical": True,
        "positive_screening_utility_demonstrated": policy["positive_screening_utility_demonstrated"],
        "complete_payload_bytes_verified": True, "training_only_head_and_feature_rosters_verified": True,
        "physical_work_stage_accounting_verified": True,
        "sealed_representation_payload_replay_bitwise_all_rows": True,
        "representation_replay_scope": "Pinned operational runtime; not an independent native ideal-real certificate"}


def audit_baselines(root, cohorts, expected, protocol):
    import pickle
    restored = pickle.loads((root / "output/baselines/hist_gradient_boosting.pickle").read_bytes())
    model = restored["model"]
    parameters = model.get_params()
    declared = protocol["baselines"]["hist_gradient_boosting"]
    for key in ("loss", "learning_rate", "max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf",
                "l2_regularization", "max_features", "max_bins", "categorical_features", "monotonic_cst",
                "interaction_cst", "warm_start", "early_stopping", "random_state"):
        require(parameters[key] == declared[key], "Frozen HGB parameter differs: "+key)
    checked = []
    for role in ("decision", "test"):
        for name in ("persistence", "training_mean", "hist_gradient_boosting"):
            receipt = read_json(root / "output/baselines" / (name+"_"+role+".json"))
            require(receipt["query_ids"] == cohorts[role]["ids"] and receipt["targets_exact"] == cohorts[role]["y"] and
                    receipt["class_membership_certified"] is False and receipt["population_risk_claim"] is False,
                    "Empirical baseline cohort or assurance differs")
            predictions = [{"lower": x, "upper": x, "width": "0"} for x in receipt["predictions_exact"]]
            audit_loss(predictions, receipt["targets_exact"], receipt["loss"])
            if name == "persistence":
                require(receipt["predictions_exact"] == cohorts[role]["persistence"], "Persistence forecast differs")
            elif name == "training_mean":
                mean = sum(map(q, cohorts["train"]["y"]), F(0))/len(cohorts["train"]["y"])
                require(receipt["predictions_exact"] == [str(mean)]*len(cohorts[role]["ids"]),
                        "Training-mean forecast differs")
            else:
                predictions = model.predict(restored["scaler"].transform(expected[role]["X"]))
                require(receipt["predictions_exact"] == [str(F.from_float(float(v))) for v in predictions],
                        "Sealed HGB payload does not replay its saved forecasts")
            checked.append(name+":"+role)
    return checked


def postrun(root):
    verify_seal(root / "protocol_seal35.json", root)
    model_seal = verify_seal(root / "output/model_seal35.json", root)
    require(model_seal["protocol_seal_sha256"] == sha(root / "protocol_seal35.json") and
            model_seal["all_models_and_policies_frozen"] is True and
            model_seal["test_numeric_values_parsed"] is False, "Model seal scope differs")
    protocol = read_json(root / "protocol35.json")
    audit_protocol(protocol)
    registry, origins, aggregates = {}, {}, []
    counts, outcomes = {}, []
    paths = sorted(p for p in (root / "output").rglob("*") if p.is_file() and
                   (p.name.endswith(".json") or p.name.endswith(".json.gz")))
    for path in paths:
        document = read_json(path)
        for receipt in iter_receipts(document):
            key = digest(receipt)
            if key in registry:
                continue
            schema = receipt["schema"]
            if schema == "step35-aggregated-stored-feature-class-risk-v1":
                aggregates.append(receipt)
            else:
                outcome = RECEIPT_SCHEMAS[schema](receipt)
                outcomes.append({"path": str(path.relative_to(root)), "schema": schema, **outcome})
            registry[key] = compact_receipt(receipt)
            origins[key] = str(path.relative_to(root))
            counts[schema] = counts.get(schema, 0)+1
    for receipt in aggregates:
        outcomes.append({"path": origins[digest(receipt)], "schema": receipt["schema"],
                         **audit_aggregation(receipt, registry)})
    require(counts.get("step35-stored-feature-finite-qp-v1", 0) > 0 and
            counts.get("step35-stored-feature-prediction-receipt-v1", 0) > 0,
            "No actual scientific certificate/prediction receipts were found")
    events = audit_event_chain(root / "output/access_and_execution35.jsonl")
    expected_events = ["primary_start", "development_numeric_access_start", "development_numeric_access_complete",
        "class_screen_frozen", "screened_policy_frozen", "models_and_policies_sealed",
        "test_numeric_access_start", "test_numeric_access_complete", "primary_complete"]
    require([event["event"] for event in events] == expected_events,
            "Primary or test-seal execution order differs")
    require(events[5]["model_seal_sha256"] == events[6]["model_seal_sha256"] == sha(root / "output/model_seal35.json"),
            "Test access was not bound to the frozen model seal")
    cohorts, expected, epsilon, normalization = audit_cohorts(root, protocol)
    panel = audit_panel(root, protocol, cohorts, expected, epsilon)
    baselines = audit_baselines(root, cohorts, expected, protocol)
    return {"schema": "step35-independent-postrun-audit-v1", "status": "PASS",
        "utc": datetime.now(timezone.utc).isoformat(),
        "protocol_seal_sha256": sha(root / "protocol_seal35.json"),
        "model_seal_sha256": sha(root / "output/model_seal35.json"),
        "verified_unique_receipt_counts": counts, "receipt_results": outcomes,
        "event_count": len(events), "event_hash_chain_verified": True,
        "causal_public_data_reconstruction": {role: len(value["ids"]) for role, value in cohorts.items()},
        "training_target_normalization": normalization, "exact_epsilon": str(epsilon),
        "finite_panel_and_payload_audit": panel, "empirical_baselines_verified": baselines,
        "optimizer_full_transition_replay_claim": False,
        "final_primal_dual_witnesses_independently_verified": True,
        "producer_certificate_modules_imported": False,
        "scientific_models_fitted_by_auditor": 0,
        "reserved_patient_arrays_accessed": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "postrun", "selfcheck", "witness"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--witness", type=Path)
    args = parser.parse_args()
    destination = args.root / "audit" / ("preexecution_gate35.json" if args.mode == "preflight" else "postrun_audit35.json")
    try:
        if args.mode == "selfcheck":
            result = {"status": "PASS", "checks": independent_selfcheck()}
        elif args.mode == "witness":
            require(args.witness is not None, "--witness path is required")
            result = {"status": "PASS", "results": [RECEIPT_SCHEMAS[r["schema"]](r)
                for r in iter_receipts(read_json(args.witness)) if r["schema"] in RECEIPT_SCHEMAS]}
        else:
            result = preflight(args.root) if args.mode == "preflight" else postrun(args.root)
            write_json(destination, result)
        print(json.dumps(canon(result if args.mode in ("selfcheck", "witness") else
            {"status": result["status"], "artifact": str(destination)}), sort_keys=True))
    except Exception as error:
        result = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "public_values_opened_by_preflight": False}
        if args.mode in ("preflight", "postrun"):
            write_json(destination, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
