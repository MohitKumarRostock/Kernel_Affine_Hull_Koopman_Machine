"""Independent Step39 verification of data chronology, models and certificates.

The mathematical checker imports only previously frozen independent checkers.
Operational map and baseline replay is scoped separately from the exact
stored-feature mathematical assurance. No scientific fitting is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
import gzip
import hashlib
import io
from itertools import combinations
import json
from math import isfinite, isqrt, lcm
from pathlib import Path
import re
import sys
import time
import traceback
import zipfile

import audit38 as a38
import audit_math37 as gm
import independent_math35 as im

ROOT = Path(__file__).resolve().parents[1]
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
TASKS = ("tetouan_zone1", "metro_traffic")
STAGES = ("f1", "f2", "f3", "final")
LAMBDAS = ("0", "1/4", "1")
WEIGHTS = ("0", "1/2", "1")
SELECTORS = ("joint_rolling", "joint_last_fold", "default_qp_rolling", "ridge_only_rolling",
             "extension_only_rolling", "affine_rolling", "envelope_rolling", "hgb_rolling",
             "ordinary_rolling", "kahm_joint_rolling", "raw_joint_rolling", "nystrom_joint_rolling")
STAGE_DATES = {
    "f1": {"train_start": "2017-01-01", "train_stop": "2017-06-01", "validation_start": "2017-06-01", "validation_stop": "2017-07-01"},
    "f2": {"train_start": "2017-01-01", "train_stop": "2017-07-01", "validation_start": "2017-07-01", "validation_stop": "2017-08-01"},
    "f3": {"train_start": "2017-01-01", "train_stop": "2017-08-01", "validation_start": "2017-08-01", "validation_stop": "2017-09-01"},
    "final": {"train_start": "2017-01-01", "train_stop": "2017-09-01"},
}
EXPECTED_N = {
    "tetouan_zone1": {"f1": {"train": 150, "validation": 30}, "f2": {"train": 180, "validation": 31},
                      "f3": {"train": 211, "validation": 31}, "final": {"train": 242}, "test": 91},
    "metro_traffic": {"f1": {"train": 140, "validation": 30}, "f2": {"train": 170, "validation": 29},
                      "f3": {"train": 199, "validation": 30}, "final": {"train": 229}, "test": 84},
}
REP_CONFIG = dict(a38.REP_CONFIG)
SOURCE_SPECS = {
    "tetouan_zone1": (849, "Tetuan City power consumption.csv", None, 0, 6,
                      "%m/%d/%Y %H:%M", 10, "DateTime", "Zone 1 Power Consumption"),
    "metro_traffic": (492, "Metro_Interstate_Traffic_Volume.csv.gz", "gzip", 7, 8,
                      "%Y-%m-%d %H:%M:%S", 60, "date_time", "traffic_volume"),
}


class AuditError(AssertionError):
    pass


def require(value, message):
    if not value:
        raise AuditError(message)


def q(value):
    return im.q(value)


def canon(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def canonical_bytes(value):
    return (json.dumps(canon(value), sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def content_digest(value):
    """Inherited mathematical identity digest (no canonical-file newline)."""
    return im.digest(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def checked_path(relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            "Unsafe relative artifact path")
    return ROOT / path


def bound_reference(record):
    path = checked_path(record["path"])
    require(path.is_file() and sha(path) == record["sha256"], "Artifact hash changed: " + str(path))
    return path


def box(record, expected=None):
    return im.interval(record, expected)


def clip(value):
    return max(F(0), min(F(1), value))


def exact_matrix(value):
    return a38.exact_matrix(value)


def exact_ids(ids, n):
    return a38.exact_ids(ids, n)


def maps_roster():
    return a38.maps_roster()


def array_digest(array):
    return a38.array_digest(array)


def audit_events():
    path = ROOT / "output/access_and_execution39.jsonl"
    if not path.exists():
        return []
    events = []
    previous = None
    last_ns = -1
    for line in path.read_text().splitlines():
        record = json.loads(line)
        saved = record.pop("event_sha256")
        require(saved == digest(record) and record["previous_event_sha256"] == previous,
                "Execution event hash chain changed")
        require(type(record["monotonic_ns"]) is int and record["monotonic_ns"] >= last_ns,
                "Execution monotonic event order changed")
        datetime.fromisoformat(record["utc"])
        record["event_sha256"] = saved
        events.append(record)
        previous = saved
        last_ns = record["monotonic_ns"]
    return events


def append_event(name, **fields):
    records = audit_events()
    record = {"event": name, "utc": datetime.now(timezone.utc).isoformat(),
              "monotonic_ns": time.monotonic_ns(),
              "previous_event_sha256": records[-1]["event_sha256"] if records else None,
              **fields}
    record["event_sha256"] = digest(record)
    path = ROOT / "output/access_and_execution39.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_bytes(record))
        stream.flush()
    return record


def event_once(name, **fields):
    matches = [(i, e) for i, e in enumerate(audit_events())
               if e["event"] == name and all(e.get(k) == v for k, v in fields.items())]
    require(len(matches) == 1, "Required unique event missing/repeated: " + name)
    return matches[0]


def csv_cells(record):
    """Independent byte-field scanner; never decode unrequested values."""
    body = record.rstrip(b"\r\n")
    starts = [0]
    in_quotes = False
    i = 0
    while i < len(body):
        if body[i] == 34:
            if in_quotes and i + 1 < len(body) and body[i + 1] == 34:
                i += 2
                continue
            in_quotes = not in_quotes
        elif body[i] == 44 and not in_quotes:
            starts.append(i + 1)
        i += 1
    require(not in_quotes, "Unfinished or multiline CSV quote")
    bounds = starts[1:] + [len(body) + 1]
    result = []
    for start, stop in zip(starts, bounds):
        field = body[start:stop - 1]
        if field.startswith(b'"'):
            require(field.endswith(b'"') and len(field) >= 2, "Malformed quoted CSV field")
            field = field[1:-1].replace(b'""', b'"')
        else:
            require(b'"' not in field, "Embedded quote in unquoted CSV field")
        result.append(field)
    return result


def source_bytes(task):
    require(task in TASKS, "Unknown public task")
    uid, member, compression, *_ = SOURCE_SPECS[task]
    source = read(ROOT / f"inputs/{task}/public_source39.json")
    path = ROOT / f"inputs/{task}/source_uci{uid}.zip"
    require(source["task"] == task and source["uci_id"] == uid and
            source["archive_path"] == str(path.relative_to(ROOT)) and
            source["archive_sha256"] == sha(path) and source["archive_bytes"] == path.stat().st_size and
            source["member"] == member and source["inner_compression"] == compression and
            source["measurement_values_decoded_or_parsed"] == 0 and
            source["scientific_model_fits"] == 0 and source["source_license"] == "CC BY 4.0",
            "Opaque source identity/provenance differs")
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == [member], "Archive member roster differs")
        raw_member = archive.read(member)
    raw = gzip.decompress(raw_member) if compression == "gzip" else raw_member
    return raw, source, hashlib.sha256(raw_member).hexdigest()


def iso(value):
    return value.isoformat(timespec="seconds")


def calendar_queries(task, start, stop):
    projection = read(ROOT / f"inputs/{task}/calendar_projection39.json")
    stamps = {r["timestamp"] for r in projection["rows"]}
    first = datetime.fromisoformat(start).replace(hour=23)
    last = datetime.fromisoformat(stop).replace(hour=23)
    queries = []
    while first < last:
        origin = first - timedelta(hours=24)
        history = [origin - timedelta(hours=23 - j) for j in range(24)]
        if all(iso(t) in stamps for t in [first, *history]):
            queries.append({"target_timestamp": iso(first), "origin_timestamp": iso(origin)})
        first += timedelta(days=1)
    return queries


def audit_calendar(task):
    raw, source, member_hash = source_bytes(task)
    p = read(ROOT / f"inputs/{task}/calendar_projection39.json")
    spec = SOURCE_SPECS[task]
    uid, member, compression, stamp_col, target_col, stamp_format, cadence, stamp_name, target_name = spec
    lines = raw.splitlines(keepends=True)
    headers = [v.decode("utf-8") for v in csv_cells(lines[0].removeprefix(b"\xef\xbb\xbf"))]
    require(len(headers) == 9 and headers[stamp_col] == stamp_name and headers[target_col] == target_name,
            "Official source schema differs")
    require(p["schema"] == "step39-calendar-only-byte-projection-v1" and p["task"] == task and
            p["headers"] == headers and p["parsed_data_columns"] == [stamp_col] and
            p["target_field_index"] == target_col and p["target_values_decoded_or_parsed"] == 0 and
            p["other_measurement_values_decoded_or_parsed"] == 0 and
            p["duplicate_target_agreement_examined"] is False and
            p["hourly_snapshot_rule"] == "Retain point readings at minute00 only; no hourly aggregation" and
            p["archive_sha256"] == source["archive_sha256"] and p["archive_member"] == member and
            p["archive_member_sha256"] == member_hash and p["csv_sha256"] == hashlib.sha256(raw).hexdigest() and
            p["csv_bytes"] == len(raw), "Calendar projection source or information scope differs")
    groups = {}
    offset = len(lines[0])
    last_stamp = None
    for line_number, line in enumerate(lines[1:], 2):
        cells = csv_cells(line)
        require(len(cells) == len(headers), "CSV field count differs")
        # This is the only data column decoded before the numerical access gate.
        stamp = datetime.strptime(cells[stamp_col].decode("ascii"), stamp_format)
        require(stamp.second == 0 and stamp.minute % cadence == 0 and
                (last_stamp is None or stamp >= last_stamp), "Source timestamp order/cadence differs")
        groups.setdefault(iso(stamp), []).append({"csv_line": line_number,
                                                "byte_start": offset, "byte_stop": offset + len(line)})
        last_stamp = stamp
        offset += len(line)
    rows = [{"timestamp": stamp, "records": refs} for stamp, refs in groups.items()]
    require(rows == p["rows"] and p["unique_timestamps"] == len(rows) and
            p["timestamp_rows"] == len(lines) - 1 and p["duplicate_record_excess"] == len(lines) - 1 - len(rows),
            "Calendar byte-offset map differs")
    by_month = {}
    for month in range(1, 13):
        start = f"2017-{month:02d}-01"
        stop = f"2017-{month + 1:02d}-01" if month < 12 else "2018-01-01"
        by_month[f"2017-{month:02d}"] = [r["target_timestamp"] for r in calendar_queries(task, start, stop)]
    require(p["eligible_2017_daily23_targets_by_month"] == by_month,
            "Calendar-only query eligibility differs")
    return {"status": "PASS", "task": task, "source_rows": len(lines) - 1,
            "unique_timestamps": len(rows), "duplicate_record_excess": len(lines) - 1 - len(rows),
            "monthly_queries": {k: len(v) for k, v in by_month.items()},
            "measurement_values_decoded": 0}


def authorized_values(task, timestamps):
    gate = read(ROOT / "audit/preexecution_gate39.json")
    require(gate["status"] == "PASS" and gate["protocol_seal_sha256"] == sha(ROOT / "protocol_seal39.json"),
            "Auditor numerical projection requires the frozen preexecution gate")
    timestamps = sorted(set(timestamps))
    p = read(ROOT / f"inputs/{task}/calendar_projection39.json")
    lookup = {r["timestamp"]: r["records"] for r in p["rows"]}
    require(all(t in lookup for t in timestamps), "Requested numerical timestamp absent")
    raw, _, _ = source_bytes(task)
    result = {}
    target_col = SOURCE_SPECS[task][4]
    for stamp in timestamps:
        values = []
        for ref in lookup[stamp]:
            token = csv_cells(raw[ref["byte_start"]:ref["byte_stop"]])[target_col].decode("ascii")
            require(len(token) <= 256 and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", token) is not None,
                    "Target token violates exact nonnegative decimal grammar")
            value = F(token)
            require(isfinite(float(value)), "Source target is not finite after binary64 conversion")
            values.append(value)
        require(values and len(set(values)) == 1 and values[0] >= 0,
                "Same-timestamp target disagreement or invalid measurement")
        result[stamp] = values[0]
    return result


def audit_sqrt(record, value):
    value = q(value)
    require(value >= 0 and record["bits"] == 80, "Optimizer root precision/radicand differs")
    floor = gm.exact_integer(record["floor_scaled"])
    scale = 1 << 80
    numerator = value.numerator * scale * scale
    denominator = value.denominator
    require(floor >= 0 and floor * floor * denominator <= numerator < (floor + 1) ** 2 * denominator,
            "Optimizer square-root floor is not exact")
    exact = floor * floor * denominator == numerator
    require(record["exact"] is exact, "Optimizer exact-root marker differs")
    lo, hi = F(floor, scale), F(floor + (not exact), scale)
    box(record["interval"], (lo, hi))
    require(q(record["lower_square"]) == lo * lo and q(record["upper_square"]) == hi * hi,
            "Optimizer root square receipts differ")
    return lo, hi


def audit_fit_wrapper(record, z, original_y, ids, expected=None):
    """Check the original-target regularized objective via an exact conversion."""
    expected = expected or {}
    y = list(map(q, original_y))
    n = len(y)
    exact_ids(ids, n)
    require(n > 0 and record["schema"] == "step39-regularized-training-fit-v1" and
            record["assurance"] == ASSURANCE and record["n"] == n and record["ids"] == ids,
            "Regularized wrapper schema/cohort differs")
    lam = q(record["ridge_lambda"])
    require(lam >= 0, "Negative regularization coefficient")
    center = sum(y, F(0)) / n
    scale = F(1) + lam
    effective = [(v + lam * center) / scale for v in y]
    variance = sum(((v - center) ** 2 for v in y), F(0)) / n
    constant = lam * variance / scale
    require(q(record["regularization_center"]) == center and
            record["regularization_center_rule"] == "exact_mean_of_original_training_targets" and
            record["original_target_digest"] == content_digest(y) and record["effective_target_digest"] == content_digest(effective) and
            list(map(q, record["effective_targets"])) == effective and q(record["objective_scale"]) == scale and
            q(record["objective_additive_constant"]) == constant and q(record["original_target_variance"]) == variance,
            "Regularizer center, pseudo-target or exact objective conversion differs")
    require(record["objective_formula"] == "mean((v-y)^2)+ridge_lambda*mean((v-mean(y))^2)" and
            record["effective_target_formula"] == "(y+ridge_lambda*mean(y))/(1+ridge_lambda)",
            "Declared regularized objective differs")
    for key in ("role", "feature_id", "class_id", "input_binding"):
        if key in expected:
            require(record[key] == expected[key], "Wrapper external binding differs: " + key)
    for key in ("ridge_lambda", "kappa"):
        if key in expected:
            require(q(record[key]) == q(expected[key]), "Wrapper frozen parameter differs: " + key)
    core = record["core_fit"]
    require(record["core_fit_digest"] == digest(core), "Effective-target witness digest differs")
    checked = a38.audit_fit(core, z, effective, ids, expected={
        **{k: v for k, v in expected.items() if k in ("role", "feature_id", "class_id", "input_binding", "kappa", "settings")}
    })
    for key in ("role", "feature_id", "class_id", "feature_digest", "dimension", "ids", "n", "kappa", "output_interval",
                "input_binding", "selected_qp_source", "selected_envelope_source", "head_promotion_allowed"):
        require(record[key] == core[key], "Wrapper/core identity mismatch: " + key)
    require(record["original_targets_clipped_in_loss"] is False and
            record["regularized_lower_is_not_original_mse_lower"] is True and
            record["native_transform_error_enclosed"] is False and record["future_response_coverage_claim"] is False and
            record["population_risk_claim"] is False, "Regularized assurance scope differs")
    risks = record["original_target_risks"]
    require(set(risks) == {c["source"] for c in core["primal_candidates"]}, "Original-risk candidate roster differs")
    for candidate in core["primal_candidates"]:
        values = list(map(q, candidate["values"]))
        raw_mse = sum(((v - t) ** 2 for v, t in zip(values, y)), F(0)) / n
        penalty = sum(((v - center) ** 2 for v in values), F(0)) / n
        obj = raw_mse + lam * penalty
        risk = risks[candidate["source"]]
        require(q(risk["original_target_mse"]) == raw_mse and q(risk["regularizer_mse"]) == penalty and
                q(risk["regularized_objective"]) == obj == scale * q(candidate["mean"]) + constant and
                q(risk["effective_target_mse"]) == q(candidate["mean"]),
                "Original-target/regularized candidate risk arithmetic differs")
    low, high = box(core["class_risk"])
    jlow, jhigh = scale * low + constant, scale * high + constant
    box(record["regularized_objective"], (jlow, jhigh))
    require(q(record["regularized_gap_target"]) == scale * q(core["gap_target_mean"]) and
            record["gap_resolved"] is (jhigh - jlow <= q(record["regularized_gap_target"])),
            "Regularized optimization-gap status differs")
    bound = record["optimizer_prediction_bound"]
    squared = n * (jhigh - jlow) / scale
    require(bound["schema"] == "step39-optimization-to-prediction-bound-v1" and
            q(bound["squared_upper"]) == squared == n * (high - low) and
            bound["formula"] == "sqrt(n*(J_upper-J_lower)/(1+ridge_lambda))" and
            bound["same_map_anchors_objective_kappa_range_and_extension_weight_required"] is True and
            bound["target"] == "canonical_weighted_extension_of_unique_exact_regularized_training_optimum" and
            bound["future_response_coverage_claim"] is False and bound["population_risk_claim"] is False,
            "Optimizer-to-prediction theorem binding/scope differs")
    _, delta = audit_sqrt(bound["sqrt_enclosure"], squared)
    require(q(bound["uniform_upper"]) == delta, "Uniform optimizer prediction bound differs")
    require(all(isfinite(v) and v >= 0 for v in record["timing"].values()), "Invalid wrapper runtime")
    return {"status": "PASS", "n": n, "ridge_lambda": str(lam),
            "feature_id": record["feature_id"], "class_id": record["class_id"],
            "core": checked, "regularized_objective": record["regularized_objective"],
            "optimizer_uniform_upper": str(delta), "gap_resolved": record["gap_resolved"]}


def audit_head39(head, fit=None, z=None, original_y=None, ids=None, expected=None):
    expected = expected or {}
    if head.get("schema") != "step39-weighted-extension-head-v1":
        return a38.audit_head(head, expected=expected)
    require(fit is not None and z is not None and original_y is not None and ids is not None,
            "Weighted extension requires the bound original training witness")
    require(head["kind"] == "weighted_lipschitz_extension" and head["recipe"] in ("qp_regularized", "envelope_all") and
            head["assurance"] == ASSURANCE and head["training_fit_role"] == "train" and
            fit["role"] == "train" and fit["head_promotion_allowed"] is True,
            "Weighted head role/type differs")
    require(fit["original_target_digest"] == content_digest(list(map(q, original_y))) and fit["ids"] == ids,
            "Weighted head original training-target binding differs")
    weight, lam = q(head["extension_weight"]), q(head["ridge_lambda"])
    require(0 <= weight <= 1 and lam == q(fit["ridge_lambda"]) and
            head["training_fit_sha256"] == digest(fit), "Weighted head parameters/fit hash differ")
    if head["recipe"] == "envelope_all":
        require(lam == 0 and weight == F(1, 2), "Envelope control configuration changed")
    require(head["formula"] == "clip_I((1-extension_weight)*max_i(v_i-kappa*d(z,a_i))+extension_weight*min_i(v_i+kappa*d(z,a_i)))" and
            head["clipping_order"] == "after_weighted_raw_envelope_mixture" and
            head["future_response_coverage_claim"] is False and head["native_transform_error_enclosed"] is False,
            "Weighted extension formula or assurance differs")
    # The inherited checker proves membership/interpolation from exact pair
    # inequalities and verifies that the selected training values were retained.
    # The constant convex mixture preserves the same bound for every weight.
    shadow = deepcopy(head)
    shadow.update(schema=a38.HEAD_SCHEMA, kind="symmetric_lipschitz_extension",
                  recipe="envelope_all" if head["recipe"] == "envelope_all" else "qp_all",
                  training_anchor_risk=head["training_effective_risk"],
                  formula="clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)")
    effective = list(map(q, fit["effective_targets"]))
    checked = a38.audit_head(shadow, fit["core_fit"], z, effective, ids)
    source = head["selected_training_source"]
    risks = fit["original_target_risks"][source]
    require(q(head["training_original_risk"]) == q(risks["original_target_mse"]) and
            q(head["training_regularized_objective"]) == q(risks["regularized_objective"]) and
            q(head["training_effective_risk"]) == q(risks["effective_target_mse"]),
            "Promoted original-target/regularized risk differs")
    bounded = head["recipe"] == "qp_regularized"
    require(head["optimizer_prediction_bound_valid_for_this_head"] is bounded and
            head["optimizer_prediction_bound"] == (fit["optimizer_prediction_bound"] if bounded else None),
            "Optimizer bound was attached to an unsupported candidate")
    for key in ("recipe", "feature_id", "class_id", "anchor_indices", "training_fit_path", "training_fit_sha256"):
        if key in expected:
            require(head[key] == expected[key], "Weighted head external binding differs: " + key)
    for key in ("ridge_lambda", "extension_weight"):
        if key in expected:
            require(q(head[key]) == q(expected[key]), "Weighted head configuration differs: " + key)
    hid = "head39:" + content_digest(head)
    if "head_id" in expected:
        require(expected["head_id"] == hid, "Weighted head identity differs")
    return {"status": "PASS", "head_id": hid, "class_id": head["class_id"],
            "recipe": head["recipe"], "anchor_count": len(ids),
            "membership_pair_checks": checked["pair_membership_checks"],
            "ridge_lambda": str(lam), "extension_weight": str(weight)}


def audit_prediction39(record, head, query_z, query_ids, distance=None, distance_checked=None, expected=None):
    if head.get("schema") != "step39-weighted-extension-head-v1":
        return a38.audit_prediction(record, head, query_z, query_ids, distance, distance_checked, expected)
    expected = expected or {}
    z = exact_matrix(query_z)
    n = len(z)
    exact_ids(query_ids, n)
    hid = "head39:" + content_digest(head)
    weight, kap = q(head["extension_weight"]), q(head["kappa"])
    require(record["schema"] == "step39-weighted-head-prediction-v1" and record["assurance"] == ASSURANCE and
            record["head_id"] == hid and record["feature_id"] == head["feature_id"] and
            record["class_id"] == head["class_id"] == a38.class_identity(head["feature_id"], kap) and
            record["query_ids"] == query_ids and record["query_feature_digest"] == content_digest(z) and
            record["n"] == n == len(record["rows"]) == len(record["predictions"]) and
            record["recipe"] == head["recipe"] and q(record["ridge_lambda"]) == q(head["ridge_lambda"]) and
            q(record["extension_weight"]) == weight, "Weighted prediction identity differs")
    require(record["targets_accessed"] is False and record["future_response_coverage_claim"] is False and
            record["native_transform_error_enclosed"] is False and
            record["enclosure_kind"] == "fixed_fitted_head_arithmetic" and 0 <= weight <= 1 and kap >= 0,
            "Weighted prediction assurance/parameters differ")
    require(distance is not None and distance_checked is not None and
            distance_checked["record_digest"] == digest(distance) == record["distance_digest"] and
            distance["feature_id"] == head["feature_id"] and distance_checked["queries"] == z and
            distance_checked["query_ids"] == query_ids, "Weighted prediction geometry binding differs")
    lookup = {name: i for i, name in enumerate(distance_checked["anchor_ids"])}
    require(all(name in lookup for name in head["anchor_ids"]), "Weighted head anchor absent")
    columns = [lookup[name] for name in head["anchor_ids"]]
    require(columns == record["anchor_columns"] == head["anchor_indices"] and
            exact_matrix(head["anchor_features"]) == [distance_checked["anchors"][i] for i in columns],
            "Weighted head anchor geometry/order differs")
    values = list(map(q, head["anchor_values"]))
    require(len(values) == len(columns) and all(0 <= v <= 1 for v in values), "Invalid weighted anchor values")
    widths = []
    for index, row in enumerate(record["rows"]):
        require(row["query_id"] == query_ids[index], "Weighted prediction row order differs")
        lower = tuple(max(v - kap * distance_checked[side][index][j] for v, j in zip(values, columns))
                      for side in ("upper", "lower"))
        upper = tuple(min(v + kap * distance_checked[side][index][j] for v, j in zip(values, columns))
                      for side in ("lower", "upper"))
        box(row["lower_envelope"], lower)
        box(row["upper_envelope"], upper)
        raw = tuple((1 - weight) * lower[k] + weight * upper[k] for k in (0, 1))
        fixed = tuple(clip(v) for v in raw)
        box(row["raw_prediction"], raw)
        box(row["prediction"], fixed)
        box(record["predictions"][index], fixed)
        require(F(0) <= fixed[0] <= fixed[1] <= F(1) and fixed[1] - fixed[0] <= kap / (1 << 80),
                "Weighted prediction enclosure width exceeds theorem")
        widths.append(fixed[1] - fixed[0])
    for key in ("head_id", "class_id", "feature_id"):
        if key in expected:
            require(record[key] == expected[key], "Weighted prediction external binding differs")
    return {"status": "PASS", "head_id": hid, "n": n,
            "max_width": str(max(widths)), "weighted_formula_verified": True}


def audit_optimizer_expansion(record, prediction, head, fit):
    require(head.get("schema") == "step39-weighted-extension-head-v1" and head["recipe"] == "qp_regularized" and
            head["selected_training_source"] == fit["selected_qp_source"] and
            head["training_fit_sha256"] == digest(fit) and head["optimizer_prediction_bound"] == fit["optimizer_prediction_bound"],
            "Optimizer expansion is not bound to the optimum-upper candidate")
    require(prediction["head_id"] == "head39:" + content_digest(head) and fit["role"] == "train",
            "Optimizer expansion source head/role differs")
    require(record["schema"] == "step39-exact-optimum-prediction-enclosure-v1" and
            record["source_prediction_digest"] == digest(prediction) and
            record["training_fit_sha256"] == head["training_fit_sha256"] and
            record["optimizer_prediction_bound_digest"] == digest(fit["optimizer_prediction_bound"]),
            "Optimizer expansion provenance differs")
    for key in ("head_id", "feature_id", "class_id", "query_ids", "query_feature_digest", "n"):
        require(record[key] == prediction[key], "Optimizer expansion source identity differs: " + key)
    delta = q(fit["optimizer_prediction_bound"]["uniform_upper"])
    require(q(record["optimizer_uniform_upper"]) == delta and delta >= 0 and
            record["intersected_with_output_interval"] == ["0", "1"] and
            record["enclosure_kind"] == "same_weight_extension_of_unique_exact_regularized_training_optimum" and
            record["targets_accessed"] is False and record["assurance"] == ASSURANCE and
            record["future_response_coverage_claim"] is False and record["population_risk_claim"] is False and
            record["native_transform_error_enclosed"] is False,
            "Optimizer expansion theorem scope differs")
    require(len(record["predictions"]) == prediction["n"], "Optimizer expansion row count differs")
    for original, expanded in zip(prediction["predictions"], record["predictions"]):
        low, high = box(original)
        box(expanded, (max(F(0), low - delta), min(F(1), high + delta)))
    return {"status": "PASS", "n": prediction["n"], "uniform_upper": str(delta)}


def candidate_configs():
    records = []
    for name in maps_roster():
        for lam, lt in (("0", "l0"), ("1/4", "l025"), ("1", "l1")):
            for weight, wt in (("0", "w0"), ("1/2", "w05"), ("1", "w1")):
                token = f"qp_{lt}_{wt}"
                records.append({"candidate_id": name + "::" + token, "map": name, "token": token,
                                "recipe": "qp_regularized", "ridge_lambda": lam, "extension_weight": weight})
        records.extend({"candidate_id": name + "::" + recipe, "map": name, "token": recipe,
                        "recipe": recipe, "ridge_lambda": None if recipe == "affine" else "0",
                        "extension_weight": None if recipe == "affine" else "1/2"}
                       for recipe in ("affine", "envelope_all"))
    return records


def baseline_configs():
    # Step 38 recorded a trimmed parameter dictionary.  Step 39 prospectively
    # binds the full current estimator configuration, including inactive
    # early-stopping defaults and the numeric-input categorical default.
    prior = {**read(ROOT / "inputs/prior/protocol38.json")["baselines"]["hgb_parameters"],
             "categorical_features": "from_dtype", "n_iter_no_change": 10,
             "quantile": None, "scoring": "loss", "tol": 1e-7,
             "validation_fraction": 0.1, "verbose": 0}
    records = [{"token": "persistence", "kind": "persistence"}, {"token": "training_mean", "kind": "mean"}]
    for leaves, l2 in ((7, 1), (15, 1), (31, 1), (15, 10)):
        params = {**prior, "max_leaf_nodes": leaves, "l2_regularization": float(l2), "random_state": 3901}
        records.append({"token": f"hgb_leaf{leaves}_l2_{l2}", "kind": "hgb", "parameters": params})
    for token, lam in (("ridge_l0001", "1/10000"), ("ridge_l001", "1/100"), ("ridge_l1", "1")):
        records.append({"token": token, "kind": "ridge", "ridge_lambda": lam,
                        "parameters": {"fit_intercept": True, "solver": "cholesky", "alpha_rule": "n*lambda"}})
    return [{**r, "candidate_id": "baseline::" + r["token"]} for r in records]


def verify_seal(relative, schema, **bindings):
    record = read(checked_path(relative))
    require(record["schema"] == schema, "Unknown evidence seal schema: " + relative)
    require(isinstance(record["bound_files"], dict) and record["bound_files"], "Empty evidence seal")
    for name, expected in bindings.items():
        require(record[name] == expected, "Seal ancestry differs: " + name)
    for path, wanted in record["bound_files"].items():
        require(checked_path(path).is_file() and sha(checked_path(path)) == wanted,
                "Sealed evidence changed: " + path)
    return record


def verify_protocol_seal():
    seal = verify_seal("protocol_seal39.json", "step39-protocol-source-seal-v1")
    require(seal["protocol_sha256"] == sha(ROOT / "protocol39.json") and
            seal["new_measurement_tokens_decoded"] == 0 and seal["single_primary_pass"] is True,
            "Protocol freeze did not precede numerical measurement access")
    required = {"protocol39.json", "analysis/interface39.txt", "source/audit39.py", "source/check_audit39.py",
                "source/data39.py", "source/check_data39.py", "source/prepare_data39.py", "source/protocol39.py",
                "source/common39.py", "source/run39.py", "source/execute39.py", "source/check_runner39.py",
                "source/learners39.py", "source/check_learners39.py",
                "inputs/calendar_roster39.json", "inputs/inherited_sources39.json", "inputs/prior_exposure39.json"}
    inherited = read(ROOT / "inputs/inherited_sources39.json")
    require(inherited["schema"] == "step39-inherited-source-copy-v1" and
            inherited["new_data_values_accessed"] is False and inherited["prior_scientific_artifacts_modified"] is False,
            "Inherited method provenance differs")
    prior_seal = read(ROOT / "inputs/prior/protocol_seal38.json")
    for relative, info in inherited["files"].items():
        required.add(relative)
        require(sha(checked_path(relative)) == info["sha256"], "Inherited scientific dependency changed")
        if relative in prior_seal["bound_files"]:
            require(prior_seal["bound_files"][relative] == info["sha256"], "Inherited dependency lacks its prior freeze binding")
    for task in TASKS:
        uid = SOURCE_SPECS[task][0]
        required.update({f"inputs/{task}/source_uci{uid}.zip", f"inputs/{task}/public_source39.json",
                         f"inputs/{task}/calendar_projection39.json", f"inputs/{task}/calendar_metadata39.json"})
    require(required <= set(seal["bound_files"]), "Protocol source freeze omitted a required dependency")
    require(sha(ROOT / "source/audit38.py") == "4aa7ed6ff39495de9834aaea0f9f4429f5f20e2febbab5425a0f6079ba3d161c" and
            sha(ROOT / "source/audit_math37.py") == a38.GRAPH_MATH_SHA256 and
            sha(ROOT / "source/independent_math35.py") == a38.INDEPENDENT_MATH_SHA256,
            "Inherited independent mathematics changed")
    protocol = read(ROOT / "protocol39.json")
    audit_protocol(protocol)
    return protocol, seal


def audit_protocol(p):
    require(p["schema"] == "step39-temporal-selection-protocol-v1" and p["step"] == 39 and p["seed"] == 3901 and
            p["tasks"] == list(TASKS) and p["stages"] == list(STAGES) and p["maps"] == maps_roster(),
            "Frozen task/stage/map roster differs")
    require(p["representations"] == REP_CONFIG, "Native representation dimension/scaling configuration changed")
    cl = p["class"]
    require(cl["kappa"] == "2" and cl["output_interval"] == ["0", "1"] and cl["assurance"] == ASSURANCE and
            cl["native_transform_error_enclosed"] is False and cl["raw_input_lipschitz_claim"] is False and
            cl["population_risk_claim"] is False and cl["future_response_coverage_claim"] is False,
            "Declared class or assurance scope differs")
    heads = p["heads"]
    require(heads["ridge_lambdas"] == list(LAMBDAS) and heads["extension_weights"] == list(WEIGHTS) and
            heads["configurations"] == candidate_configs() and heads["certified_heads_per_stage"] == 198 and
            heads["qp_fits_per_stage"] == 54 and heads["all_training_anchors"] is True and
            heads["distance_bits"] == heads["optimizer_sqrt_bits"] == 80 and
            heads["clipping_order"] == "AFTER_WEIGHTED_MIXTURE" and heads["affine_ridge_lambda"] == "1/1000000" and
            heads["solver_settings"] == gm.PRIMARY_SETTINGS and
            heads["finite_valid_wide_brackets"] == "RETAIN_AND_REPORT_UNRESOLVED",
            "Frozen regularization/extension/solver roster differs")
    base = p["baselines"]
    require(base["configurations"] == baseline_configs() and base["count"] == 9 and base["clipped"] is False and
            base["class_membership_certified"] is False and base["same_lag_inputs"] is True,
            "Ordinary baseline tuning/access scope differs")
    ids = [r["candidate_id"] for r in candidate_configs() + baseline_configs()]
    require(len(ids) == 207 == len(set(ids)) and p["evaluation"]["prediction_ids"] == ids,
            "Complete prediction panel differs")
    s = p["selection"]
    require(s["folds"] == ["f1", "f2", "f3"] and s["selectors"] == list(SELECTORS) and
            s["primary_selector"] == "joint_rolling" and s["reference_selector"] == "default_qp_rolling" and
            s["ordinary_selector"] == "ordinary_rolling" and s["selection_control"] == "joint_last_fold" and
            s["test_reselection_allowed"] is False and s["assessment_oracle_fits"] is False and
            s["tie_breakers"] == ["exact physical MSE upper", "f3 complete map+head payload bytes", "lexical candidate_id"],
            "Frozen selection policy differs")
    require(s["pooling"] == "Sum of physical-unit SSE divided by total held-out development query count" and
            s["optimization_selection_stability"]["criterion"] == "Selected joint rolling candidate upper strictly below every other joint candidate lower" and
            s["optimization_selection_stability"]["future_performance_claim"] is False,
            "Development pooling or refinement-stability claim differs")
    ev = p["evaluation"]
    require(ev["all_candidate_outcomes_retained"] is True and ev["primary_per_task"]["gain"] ==
            "reference_lower>0 and 20*new_upper<=19*reference_lower" and
            ev["primary_per_task"]["ordinary"] == "new_upper<=ordinary_lower" and
            ev["primary_per_task"]["metric"] == "physical MSE" and
            ev["primary_per_task"]["both_conditions_required"] is True and
            ev["overall_success"] == "Both task-specific primary criteria and all independent checks pass" and
            ev["months"] == ["2017-09", "2017-10", "2017-11"] and
            ev["population_significance_test"] is False and ev["future_response_coverage_claim"] is False,
            "Frozen confirmatory criterion differs")
    ex = p["execution"]
    require(ex["single_primary_pass"] is True and ex["scientific_reruns_allowed"] is False and ex["threads"] == 1 and
            ex["failure_policy"] == "STOP_RETAIN_FAILURE_NO_RETUNING" and ex["audit_failure_policy"] == "STOP_RETAIN_EVIDENCE",
            "Single-pass execution policy differs")
    data = p["data"]
    require(data["task_order"] == list(TASKS) and data["stages"] == STAGE_DATES and
            data["test_start"] == "2017-09-01" and data["test_stop_exclusive"] == "2017-12-01" and
            data["horizon_hours"] == data["lag_hours"] == 24 and data["daily_target_hour"] == 23 and
            data["calendar_roster_sha256"] == sha(ROOT / "inputs/calendar_roster39.json") and
            data["targets_clipped"] is False and data["no_online_refit"] is True and
            data["previously_revealed_targets_may_be_later_history"] is True and
            data["other_measurement_predictors"] is False and data["future_calendar_predictors"] is False,
            "Frozen temporal/target/history contract differs")
    for task in TASKS:
        source = read(ROOT / f"inputs/{task}/public_source39.json")
        projection = read(ROOT / f"inputs/{task}/calendar_projection39.json")
        d = data["sources"][task]
        require(d["expected_n"] == EXPECTED_N[task] and d["uci_id"] == SOURCE_SPECS[task][0] and
                d["archive_sha256"] == source["archive_sha256"] and d["archive"] == source["archive_path"] and
                d["member"] == SOURCE_SPECS[task][1] and d["target_field_index"] == SOURCE_SPECS[task][4] and
                d["csv_sha256"] == projection["csv_sha256"] and d["source_license"] == "CC BY 4.0",
                "Source numeric identity differs")
    return {"status": "PASS", "tasks": 2, "maps_per_stage": 18, "heads_per_stage": 198,
            "baselines_per_stage": 9, "selectors": 12, "assessment_class_oracle": False}


def frozen_queries(task, stage, partition):
    roster = read(ROOT / "inputs/calendar_roster39.json")
    if partition == "test":
        start, stop = "2017-09-01", "2017-12-01"
        record = roster["tasks"][task]["test"]
        n = EXPECTED_N[task]["test"]
    else:
        start, stop = STAGE_DATES[stage][partition + "_start"], STAGE_DATES[stage][partition + "_stop"]
        record = roster["tasks"][task]["stages"][stage][partition]
        n = EXPECTED_N[task][stage][partition]
    queries = [{"id": task + ":" + row["target_timestamp"], "task": task, **row}
               for row in calendar_queries(task, start, stop)]
    require(record["queries"] == queries and len(queries) == record["n"] == n and
            record["query_ids_sha256"] == digest([r["id"] for r in queries]) and
            record["calendar_start"] == start and record["calendar_stop_exclusive"] == stop,
            "Frozen calendar cohort differs")
    return queries


def lag_stamps(query):
    origin = datetime.fromisoformat(query["origin_timestamp"])
    require(datetime.fromisoformat(query["target_timestamp"]) - origin == timedelta(hours=24),
            "Forecast horizon differs")
    return [iso(origin - timedelta(hours=23 - j)) for j in range(24)]


def audit_access(record, task, stage, phase, timestamps, maximum):
    stamps = sorted(set(timestamps))
    p = read(ROOT / f"inputs/{task}/calendar_projection39.json")
    lookup = {r["timestamp"]: r["records"] for r in p["rows"]}
    duplicate = [{"timestamp": stamp, "source_records": len(lookup[stamp]), "exact_agreement": True}
                 for stamp in stamps if len(lookup[stamp]) > 1]
    require(record["task"] == task and record["stage"] == stage and record["phase"] == phase and
            record["requested_timestamps"] == stamps and record["unique_measurement_timestamps"] == len(stamps) and
            record["measurement_token_decodes"] == sum(len(lookup[s]) for s in stamps) and
            record["maximum_allowed_timestamp"] == maximum and record["maximum_decoded_timestamp"] == max(stamps) and
            max(stamps) <= maximum and record["duplicate_agreement"] == duplicate and
            record["archive_sha256"] == p["archive_sha256"] and record["csv_sha256"] == p["csv_sha256"] and
            record["other_measurement_values_decoded_or_parsed"] == 0 and record["outcome_based_row_filtering"] is False,
            "Source access projection, duplicate agreement or temporal boundary differs")
    return {"unique_measurement_timestamps": len(stamps), "duplicate_timestamps": len(duplicate)}


def audit_cohort(task, stage, partition):
    import numpy as np
    base = f"output/{task}/{stage}"
    relative = f"{base}/cohorts/{partition}.json"
    record = read(ROOT / relative)
    queries = frozen_queries(task, stage, partition)
    ids = [r["id"] for r in queries]
    lags = [lag_stamps(r) for r in queries]
    needed = sorted({s for row in lags for s in row} | {r["target_timestamp"] for r in queries})
    started_index, _ = event_once(partition + "_measurement_access_started", task=task, stage=stage)
    completed_index, complete = event_once(partition + "_measurement_access_completed", task=task, stage=stage)
    require(started_index < completed_index and complete["cohort_path"] == relative and
            complete["cohort_sha256"] == sha(ROOT / relative) and complete["n"] == len(ids) and
            complete["test_target_values_decoded"] == 0, "Development measurement access did not complete once")
    values = authorized_values(task, needed)
    raw_targets = [values[r["target_timestamp"]] for r in queries]
    raw_persistence = [values[r["origin_timestamp"]] for r in queries]
    norm_path = ROOT / f"{base}/normalization39.json"
    normal = read(norm_path)
    if partition == "train":
        training_targets, training_ids = raw_targets, ids
    else:
        training = read(ROOT / f"{base}/cohorts/train.json")
        training_targets, training_ids = list(map(q, training["raw_targets"])), training["ids"]
    low, high = min(training_targets), max(training_targets)
    require(high > low and normal["schema"] == "step39-training-target-normalization-v1" and
            normal["task"] == task and normal["stage"] == stage and normal["fitted_partition"] == "train" and
            q(normal["target_min"]) == low and q(normal["target_max"]) == high and q(normal["span"]) == high - low and
            normal["train_target_ids_sha256"] == digest(training_ids) and
            normal["raw_train_targets_sha256"] == digest(training_targets) and normal["target_values_clipped"] is False and
            normal["future_data_used"] is False, "Target normalization was not fitted on this stage's training targets")
    targets = [(v - low) / (high - low) for v in raw_targets]
    persistence = [(v - low) / (high - low) for v in raw_persistence]
    X = np.asarray([[float(values[s]) for s in row] for row in lags], dtype=np.float64)
    saved_X = np.load(ROOT / record["X_path"], allow_pickle=False)
    require(record["schema"] == "step39-causal-cohort-v1" and record["task"] == task and record["stage"] == stage and
            record["name"] == partition and record["n"] == len(ids) and record["ids"] == ids and
            record["target_timestamps"] == [r["target_timestamp"] for r in queries] and
            record["origin_timestamps"] == [r["origin_timestamp"] for r in queries] and record["lag_timestamps"] == lags and
            list(map(q, record["raw_targets"])) == raw_targets and list(map(q, record["y"])) == targets and
            list(map(q, record["raw_persistence"])) == raw_persistence and list(map(q, record["persistence"])) == persistence and
            record["normalization_path"] == str(norm_path.relative_to(ROOT)) and record["normalization_sha256"] == sha(norm_path) and
            record["X_path"] == f"{base}/cohorts/{partition}_X.npy" and record["X_sha256"] == sha(ROOT / record["X_path"]) and
            saved_X.dtype == np.float64 and np.isfinite(saved_X).all() and np.array_equal(saved_X, X) and
            record["other_measurement_predictors_used"] is False and record["target_values_clipped"] is False,
            "Cohort raw history, label normalization or identity differs")
    access = read(ROOT / f"{base}/data_access_{partition}39.json")
    stop = datetime.fromisoformat(STAGE_DATES[stage][partition + "_stop"]) - timedelta(seconds=1)
    audit_access(access, task, stage, partition, needed, iso(stop))
    require(access["schema"] == "step39-development-measurement-access-v1" and access["status"] == "PASS" and
            access["cohort_path"] == relative and access["cohort_sha256"] == sha(ROOT / relative) and
            access["normalization_sha256"] == sha(norm_path) and
            complete["access_sha256"] == sha(ROOT / f"{base}/data_access_{partition}39.json"),
            "Development access receipt changed")
    return {**record, "X": X, "y": targets, "persistence": persistence,
            "normalization": normal, "path": relative, "sha256": sha(ROOT / relative)}


def load_context(task, stage):
    import numpy as np
    import pickle
    import representations35 as replay
    manifest = read(ROOT / f"output/{task}/{stage}/fit_manifest39.json")
    replay.initialize_runtime()
    maps, features, heads, fits, baselines = {}, {}, {}, {}, {}
    for name, info in manifest["maps"].items():
        maps[name] = replay.deserialize_map(bound_reference({"path": info["map_path"], "sha256": info["map_sha256"]}).read_bytes())
        features[name] = np.load(bound_reference(info["training_features"]), allow_pickle=False)
    for row in manifest["candidate_rows"]:
        heads[row["candidate_id"]] = read(bound_reference({"path": row["head_path"], "sha256": row["head_sha256"]}))
        if row["fit_path"] not in fits:
            fits[row["fit_path"]] = read(bound_reference({"path": row["fit_path"], "sha256": row["fit_sha256"]}))
    for row in manifest["baseline_rows"]:
        path = bound_reference({"path": row["payload_path"], "sha256": row["payload_sha256"]})
        baselines[row["candidate_id"]] = pickle.loads(path.read_bytes()) if row["baseline_kind"] in ("hgb", "ridge") else read(path)
    return {"manifest": manifest, "maps": maps, "train_features": features, "heads": heads,
            "fits": fits, "baselines": baselines,
            "training": read(bound_reference(manifest["training_cohort"]))}


def audit_stage_models(protocol, task, stage, training, context=None):
    """Model-artifact check usable on already-authorized or synthetic cohorts."""
    import numpy as np
    from scipy.linalg import solve
    from scipy.spatial.distance import pdist
    import representations35 as replay
    context = load_context(task, stage) if context is None else context
    manifest = context["manifest"]
    X, y, ids = training["X"], list(map(q, training["y"])), training["ids"]
    names = maps_roster()
    configs = candidate_configs()
    rows = manifest["candidate_rows"]
    expected_ids = [r["candidate_id"] for r in configs]
    require(manifest["schema"] == "step39-fit-manifest-v1" and manifest["task"] == task and manifest["stage"] == stage and
            [r["candidate_id"] for r in rows] == expected_ids and sorted(manifest["maps"]) == names and
            manifest["training_n"] == len(y) == len(ids) == len(X) and manifest["normalization"] == training["normalization"] and
            manifest["training_cohort"] == {"path": training["path"], "sha256": training["sha256"]} and
            manifest["all_fit_data_at_or_before_held_out_origins"] is True,
            "Stage fit manifest or original training cohort differs")
    for row, cfg in zip(rows, configs):
        require(all(row[k] == v for k, v in cfg.items()), "Fitted configuration roster differs")
    if stage == "final":
        require(manifest["selection_seal_sha256"] == sha(ROOT / f"output/{task}/selection_seal39.json"),
                "Final refit was not bound to the frozen development selection")
    else:
        require(manifest["selection_seal_sha256"] is None, "Development fold used a later selection")
    metadata = read(ROOT / f"output/{task}/{stage}/representation_metadata39.json")
    probes = [((2 * j + 1) * len(X)) // 64 for j in range(32)]
    require(metadata["schema"] == "step35-frozen-representations-v1" and metadata["config"] == REP_CONFIG and
            metadata["seed"] == 3901 and metadata["training_shape"] == list(X.shape) and
            metadata["training_sha256"] == array_digest(X) and metadata["probe_indices"] == probes and
            metadata["probe_sha256"] == array_digest(X[probes]) and
            metadata["factory_source_sha256"] == sha(Path(replay.__file__)) and
            metadata["forecast_labels_used"] is False and metadata["no_heldout_input_or_label_used"] is True and
            set(metadata["map_order"]) == set(names) and len(metadata["map_order"]) == 18,
            "Stage representation fitting used another data/configuration binding")
    require(set(metadata["scalers"]) == {"minmax01", "standard"}, "Representation scaler roster differs")
    for mode, record in metadata["scalers"].items():
        low, high = X.min(axis=0), X.max(axis=0)
        constant = low == high
        offset = low if mode == "minmax01" else X.mean(axis=0)
        divisor = high - low if mode == "minmax01" else X.std(axis=0, ddof=0)
        divisor = np.where(constant, 1., divisor)
        require(np.array_equal(np.asarray(record["offset"]), offset) and
                np.array_equal(np.asarray(record["divisor"]), divisor) and record["constant_mask"] == constant.tolist() and
                record["training_rows"] == len(X) and record["heldout_clipping"] is False,
                "Input normalization was not fitted only on this stage's training rows")
    metadata_by_map = {r["name"]: r for r in metadata["maps"]}
    require(set(metadata_by_map) == set(names), "Native map metadata roster differs")
    indexed = {r["candidate_id"]: r for r in rows}
    fit_summaries, head_summaries = [], []
    for name in names:
        info = manifest["maps"][name]
        meta = metadata_by_map[name]
        model = context["maps"][name]
        z = context["train_features"][name]
        map_path = checked_path(info["map_path"])
        feature_ref = info["training_features"]
        require(map_path.stat().st_size == info["map_payload_bytes"] == meta["payload_bytes"] and
                sha(map_path) == info["map_sha256"] == meta["payload_sha256"] and model.name == name == info["name"] and
                model.family == info["family"] == meta["family"] and model.scaling == info["scaling"] == meta["scaling"] and
                replay.numeric_bytes(model) == info["map_numeric_bytes"] and z.dtype == np.float64 and np.isfinite(z).all() and
                z.shape == (len(ids), info["dimension"]) and info["dimension"] == meta["actual_width"] and
                np.array_equal(z, replay.transform_one(model, X)), "Stored map or training feature replay differs: " + name)
        require(feature_ref["feature_digest"] == content_digest(exact_matrix(z)) and
                info["feature_id"] == "feature39:" + digest({"task": task, "stage": stage, "map": name,
                    "map_sha256": info["map_sha256"], "train_feature_sha256": feature_ref["sha256"]}) and
                info["class_id"] == a38.class_identity(info["feature_id"]), "Stage-specific feature/class identity differs")
        distances = pdist(model.core.transform(X[probes]), metric="euclidean")
        positive = distances[distances > 0]
        metric = 1. / float(np.median(positive)) if len(positive) else 1.
        require(float(model.metric_scale) == metric == meta["metric_scale"] and
                meta["positive_probe_pair_count"] == len(positive), "Feature metric did not use this fold's training probes")
        if model.family == "nystrom":
            width = meta["requested_width"]
            landmarks = [((2 * j + 1) * len(X)) // (2 * width) for j in range(width)]
            require(meta["landmark_indices"] == landmarks and
                    np.array_equal(model.core.landmarks, model.core.scaler.transform(X[landmarks])),
                    "Nyström landmarks used non-training rows")
        binding = {"cohort_path": training["path"], "cohort_sha256": training["sha256"],
                   "feature_path": feature_ref["path"], "feature_sha256": feature_ref["sha256"],
                   "map_sha256": info["map_sha256"], "task": task, "stage": stage,
                   "anchor_indices": list(range(len(y)))}
        for lam, token in (("0", "l0"), ("1/4", "l025"), ("1", "l1")):
            row = indexed[name + "::qp_" + token + "_w05"]
            fit = context["fits"][row["fit_path"]]
            require(row["fit_path"] == f"output/{task}/{stage}/fits/{name}/lambda_{token}.json",
                    "Regularized fit path/configuration differs")
            checked = audit_fit_wrapper(fit, z, y, ids, expected={"role": "train", "feature_id": info["feature_id"],
                "class_id": info["class_id"], "input_binding": binding, "kappa": "2", "ridge_lambda": lam,
                "settings": protocol["heads"]["solver_settings"]})
            fit_summaries.append(checked)
        for cfg in [r for r in configs if r["map"] == name]:
            row = indexed[cfg["candidate_id"]]
            head = context["heads"][row["candidate_id"]]
            fit = context["fits"][row["fit_path"]]
            path = checked_path(row["head_path"])
            require(row["head_payload_bytes"] == path.stat().st_size and row["map_payload_bytes"] == map_path.stat().st_size and
                    row["deployment_payload_bytes"] == path.stat().st_size + map_path.stat().st_size and
                    row["feature_id"] == head["feature_id"] == info["feature_id"] and
                    row["class_id"] == head["class_id"] == info["class_id"] and row["dimension"] == info["dimension"] and
                    row["family"] == info["family"] and row["scaling"] == info["scaling"] and
                    row["training_only"] is True and row["assurance"] == ASSURANCE,
                    "Deployment head identity/source/complete payload size differs")
            expected = {"feature_id": info["feature_id"], "class_id": info["class_id"], "head_id": row["head_id"]}
            if cfg["recipe"] == "affine":
                require(fit["head"] == head and fit["head_id"] == row["head_id"] and fit["training_binding"] == binding,
                        "Affine training-source binding differs")
                im.audit_affine_record(fit)
                checked = audit_head39(head, expected=expected)
                exact = exact_matrix(z)
                center = [sum((r[j] for r in exact), F(0)) / len(y) for j in range(z.shape[1])]
                require(list(map(q, fit["projection"]["center_features"])) == center and
                        q(fit["projection"]["center_target"]) == sum(y, F(0)) / len(y),
                        "Affine recentering used another stage's features/targets")
                yf = np.asarray([float(v) for v in y])
                xc, yc = z - z.mean(axis=0), yf - yf.mean()
                beta = np.linalg.solve(xc.T @ xc + len(z) * 1e-6 * np.eye(z.shape[1]), xc.T @ yc)
                intercept = float(yf.mean() - z.mean(axis=0) @ beta)
                require(list(map(q, fit["projection"]["source_coefficients"])) == [F.from_float(float(v)) for v in beta] and
                        q(fit["projection"]["source_intercept"]) == F.from_float(intercept),
                        "Affine ridge proposal differs from the declared stage-only recipe")
            else:
                expected.update({"recipe": row["recipe"], "ridge_lambda": row["ridge_lambda"],
                                 "extension_weight": row["extension_weight"], "anchor_indices": list(range(len(y))),
                                 "training_fit_path": row["fit_path"], "training_fit_sha256": row["fit_sha256"]})
                checked = audit_head39(head, fit, z, y, ids, expected)
            require(row["anchor_count"] == checked["anchor_count"] and checked["head_id"] == row["head_id"],
                    "Retained head size/content identifier differs")
            head_summaries.append(checked)
    require([r["candidate_id"] for r in manifest["baseline_rows"]] ==
            [r["candidate_id"] for r in protocol["baselines"]["configurations"]], "Baseline configuration order differs")
    scaler = context["maps"]["raw_minmax01"].core
    SX = scaler.transform(X)
    yf = np.asarray([float(v) for v in y])
    mean = sum(y, F(0)) / len(y)
    baseline_reconstructions = 0
    for row, cfg in zip(manifest["baseline_rows"], protocol["baselines"]["configurations"]):
        require(all(row[k] == v for k, v in cfg.items()) and row["recipe"] == "baseline" and
                row["baseline_kind"] == cfg["kind"] and row["class_membership_certified"] is False and
                row["deployment_payload_bytes"] == checked_path(row["payload_path"]).stat().st_size,
                "Baseline payload/configuration differs")
        value = context["baselines"][row["candidate_id"]]
        if cfg["kind"] == "persistence":
            require(value == {"rule": "last_observed_own_target"}, "Persistence rule changed")
        elif cfg["kind"] == "mean":
            require(value == {"value": str(mean), "rule": "exact_training_mean"}, "Baseline training mean changed")
        else:
            model, restored_scaler = value["model"], value["scaler"]
            require(np.array_equal(restored_scaler.offset, scaler.offset) and
                    np.array_equal(restored_scaler.divisor, scaler.divisor) and model.n_features_in_ == 24,
                    "Baseline scaler was not this stage's training-only scaler")
            params = model.get_params()
            if cfg["kind"] == "hgb":
                require(params == cfg["parameters"] and model.n_iter_ == 200,
                        "HGB parameter/iteration budget differs")
            else:
                alpha = len(y) * float(q(cfg["ridge_lambda"]))
                require(params["alpha"] == alpha and params["fit_intercept"] is True and params["solver"] == "cholesky",
                        "Ordinary ridge parameterization differs")
                xc, yc = SX - SX.mean(axis=0), yf - yf.mean()
                matrix = xc.T @ xc
                matrix.flat[::matrix.shape[0] + 1] += alpha
                coefficient = solve(matrix, xc.T @ yc, assume_a="pos")
                intercept = float(yf.mean() - SX.mean(axis=0) @ coefficient)
                require(np.allclose(model.coef_, coefficient, rtol=2e-12, atol=2e-12) and
                        np.isclose(model.intercept_, intercept, rtol=2e-12, atol=2e-12),
                        "Ordinary ridge does not match the declared training-only normal equations")
                baseline_reconstructions += 1
    require([r["map"] for r in manifest["group_timings"]] == names and
            isfinite(manifest["representation_seconds"]) and manifest["representation_seconds"] >= 0 and
            isfinite(manifest["baseline_fit_seconds"]) and manifest["baseline_fit_seconds"] >= 0,
            "Training cost accounting has invalid groups/values")
    for group in manifest["group_timings"]:
        require([r["ridge_lambda"] for r in group["regularized_fits"]] == list(LAMBDAS) and
                all(isfinite(group[k]) and group[k] >= 0 for k in ("affine_fit_seconds", "whole_map_group_seconds")),
                "Regularized shared-fit work roster differs")
    return {"status": "PASS", "task": task, "stage": stage, "maps": 18, "heads": 198, "ordinary_baselines": 9,
            "regularized_fits": fit_summaries, "head_membership": head_summaries,
            "verified_optimization_gaps_resolved": sum(r["gap_resolved"] for r in fit_summaries),
            "affine_proposal_reconstructions": 18, "ordinary_ridge_normal_equation_checks": baseline_reconstructions,
            "new_scientific_model_fits_by_auditor": 0,
            "representation_scope": "Stored operational binary64 coordinates; native ideal-real errors not enclosed"}


def audit_prediction_bundle(protocol, task, stage, record, context, X, ids, persistence):
    import numpy as np
    import representations35 as replay
    manifest = context["manifest"]
    expected_ids = protocol["evaluation"]["prediction_ids"]
    qp_ids = [r["candidate_id"] for r in manifest["candidate_rows"] if r["recipe"] == "qp_regularized"]
    require(record["schema"] in ("step39-prediction-bundle-v1", "step39-sealed-test-predictions-v1") and
            record["task"] == task and record["stage"] == stage and record["query_ids"] == ids and
            set(record["predictions"]) == set(expected_ids) and set(record["optimizer_predictions"]) == set(qp_ids) and
            set(record["maps"]) == set(maps_roster()) and
            record["fit_seal_sha256"] == sha(ROOT / f"output/{task}/{stage}/fit_seal39.json"),
            "Complete prediction panel identity differs")
    summaries = []
    total_distances = 0
    for name in maps_roster():
        info = manifest["maps"][name]
        ref = record["maps"][name]
        z = np.load(bound_reference(ref["features"]), allow_pickle=False)
        require(z.dtype == np.float64 and np.isfinite(z).all() and
                np.array_equal(z, replay.transform_one(context["maps"][name], X)) and
                ref["features"]["feature_digest"] == content_digest(exact_matrix(z)),
                "Held-out feature replay or exact-coordinate identity differs")
        distance = read(bound_reference(ref["distances"]))
        checked = a38.audit_distances(distance, context["train_features"][name], z,
                                     context["training"]["ids"], ids, {"feature_id": info["feature_id"]})
        total_distances += checked["summary"]["distance_pairs_verified"]
        for row in [r for r in manifest["candidate_rows"] if r["map"] == name]:
            cid = row["candidate_id"]
            pred, head = record["predictions"][cid], context["heads"][cid]
            summaries.append(audit_prediction39(pred, head, z, ids, distance, checked,
                             {"head_id": row["head_id"], "feature_id": info["feature_id"], "class_id": info["class_id"]}))
            if row["recipe"] == "qp_regularized":
                audit_optimizer_expansion(record["optimizer_predictions"][cid], pred, head, context["fits"][row["fit_path"]])
    for row in manifest["baseline_rows"]:
        cid, kind = row["candidate_id"], row["baseline_kind"]
        model = context["baselines"][cid]
        if kind == "persistence":
            values = list(map(q, persistence))
        elif kind == "mean":
            values = [q(model["value"])] * len(ids)
        else:
            values = [F.from_float(float(v)) for v in model["model"].predict(model["scaler"].transform(X))]
        prediction = record["predictions"][cid]
        require(prediction["baseline"] == row["token"], "Baseline prediction method differs")
        a38.audit_baseline_prediction(prediction, ids, values)
    require(all(isfinite(v) and v >= 0 for v in record["timing"].values()), "Prediction cost receipt invalid")
    return {"status": "PASS", "queries": len(ids), "head_predictions": len(ids) * 198,
            "optimizer_prediction_enclosures": len(ids) * 162, "baseline_predictions": len(ids) * 9,
            "query_anchor_distances": total_distances, "math": summaries}


def scaled_box(record, multiplier):
    lo, hi = box(record)
    multiplier = q(multiplier)
    require(multiplier >= 0, "Negative physical-risk scale")
    return {"lower": str(lo * multiplier), "upper": str(hi * multiplier), "width": str((hi - lo) * multiplier)}


def loss_box(predictions, targets):
    require(len(predictions) == len(targets) and len(targets) > 0, "Loss cohort is empty or mismatched")
    lows, highs = [], []
    for prediction, target in zip(predictions, targets):
        low, high = box(prediction)
        target = q(target)
        lows.append((low - target) ** 2 if target < low else (high - target) ** 2 if target > high else F(0))
        highs.append(max((low - target) ** 2, (high - target) ** 2))
    lo, hi = sum(lows, F(0)) / len(targets), sum(highs, F(0)) / len(targets)
    return {"lower": str(lo), "upper": str(hi), "width": str(hi - lo)}


def pooled_box(records):
    n = sum(r["n"] for r in records)
    require(n > 0 and all(type(r["n"]) is int and r["n"] > 0 for r in records), "Invalid pooling cohort weights")
    lower = sum((r["n"] * box(r["risk"])[0] for r in records), F(0)) / n
    upper = sum((r["n"] * box(r["risk"])[1] for r in records), F(0)) / n
    return {"lower": str(lower), "upper": str(upper), "width": str(upper - lower)}


def audit_validation_values(protocol, task, stage, scores, bundle, manifest, cohort):
    require(scores["schema"] == "step39-validation-scores-v1" and scores["task"] == task and scores["stage"] == stage and
            scores["n"] == len(cohort["ids"]) and scores["normalization"] == cohort["normalization"] and
            scores["fit_seal_sha256"] == bundle["fit_seal_sha256"] and
            scores["validation_predictions_sha256"] == sha(ROOT / f"output/{task}/{stage}/validation_predictions39.json") and
            scores["cohort_sha256"] == cohort["sha256"], "Validation score ancestry differs")
    original_rows = manifest["candidate_rows"] + manifest["baseline_rows"]
    require([r["candidate_id"] for r in scores["candidate_rows"]] == protocol["evaluation"]["prediction_ids"],
            "Validation score candidate roster differs")
    scale = q(cohort["normalization"]["span"]) ** 2
    for row, original in zip(scores["candidate_rows"], original_rows):
        cid = row["candidate_id"]
        require(all(row[k] == v for k, v in original.items()), "Validation altered a fitted candidate identity")
        prediction = bundle["predictions"][cid]
        a38.audit_loss(prediction["predictions"], cohort["y"], row["validation_risk"])
        require(row["physical_validation_risk"] == scaled_box(row["validation_risk"], scale) and
                row["validation_prediction_digest"] == digest(prediction), "Validation physical scale/prediction digest differs")
        if row["recipe"] == "qp_regularized":
            raw = scaled_box(row["physical_optimizer_risk"], F(1) / scale)
            require(raw == loss_box(bundle["optimizer_predictions"][cid]["predictions"], cohort["y"]),
                    "Optimizer validation-risk enclosure differs")
        else:
            require(row["physical_optimizer_risk"] is None, "Optimizer risk attached to an unsupported head/baseline")
    return {"status": "PASS", "n": len(cohort["ids"]), "candidate_losses": len(original_rows),
            "optimizer_losses": 162, "physical_unit_pooling_ready": True}


def selector_pool(rows, label):
    require(label in SELECTORS, "Unknown independent selector")
    if label == "ordinary_rolling":
        return [r for r in rows if r["recipe"] == "baseline"]
    if label == "hgb_rolling":
        return [r for r in rows if r["baseline_kind"] == "hgb"]
    if label == "affine_rolling":
        return [r for r in rows if r["recipe"] == "affine"]
    if label == "envelope_rolling":
        return [r for r in rows if r["recipe"] == "envelope_all"]
    pool = [r for r in rows if r["recipe"] == "qp_regularized"]
    if label == "default_qp_rolling":
        pool = [r for r in pool if q(r["ridge_lambda"]) == 0 and q(r["extension_weight"]) == F(1, 2)]
    elif label == "ridge_only_rolling":
        pool = [r for r in pool if q(r["extension_weight"]) == F(1, 2)]
    elif label == "extension_only_rolling":
        pool = [r for r in pool if q(r["ridge_lambda"]) == 0]
    elif label in ("kahm_joint_rolling", "raw_joint_rolling", "nystrom_joint_rolling"):
        family = label.split("_")[0]
        pool = [r for r in pool if r["family"] == family]
    return pool


def independent_selection(rows, label):
    pool = selector_pool(rows, label)
    require(pool, "Frozen selector has no candidate")
    criterion = "last_fold_physical_risk" if label == "joint_last_fold" else "pooled_physical_risk"
    chosen = min(pool, key=lambda row: (q(row[criterion]["upper"]), row["selection_payload_bytes"], row["candidate_id"]))
    return {"candidate_id": chosen["candidate_id"], "criterion": criterion, "risk": chosen[criterion],
            "selection_payload_bytes": chosen["selection_payload_bytes"], "pool_size": len(pool),
            "selection_uses_development_only": True}


def audit_selection_values(scores, record):
    """Verify pooling, all twelve selections and solver-refinement stability."""
    require(set(scores) == {"f1", "f2", "f3"} and record["schema"] == "step39-development-selection-v1" and
            record["selection_uses_development_only"] is True and record["test_targets_accessed"] == 0,
            "Selection source/scope differs")
    stages = ("f1", "f2", "f3")
    ids = [row["candidate_id"] for row in scores["f1"]["candidate_rows"]]
    require(all([row["candidate_id"] for row in scores[s]["candidate_rows"]] == ids for s in stages) and
            [row["candidate_id"] for row in record["candidate_rows"]] == ids and len(ids) == 207,
            "Selection fold/candidate roster differs")
    index = {stage: {r["candidate_id"]: r for r in scores[stage]["candidate_rows"]} for stage in stages}
    for row in record["candidate_rows"]:
        cid = row["candidate_id"]
        last = index["f3"][cid]
        keep = ("candidate_id", "map", "token", "recipe", "ridge_lambda", "extension_weight", "family", "scaling", "baseline_kind")
        require(all(row[key] == last.get(key) for key in keep), "Selected configuration has another f3 identity")
        expected_folds = {}
        for stage in stages:
            original = index[stage][cid]
            expected_folds[stage] = {"n": scores[stage]["n"], "physical_risk": original["physical_validation_risk"],
                "physical_optimizer_risk": original["physical_optimizer_risk"], "payload_bytes": original["deployment_payload_bytes"]}
        require(row["folds"] == expected_folds, "Selection fold scores/payload bindings differ")
        pooled = pooled_box([{"n": x["n"], "risk": x["physical_risk"]} for x in expected_folds.values()])
        refined = pooled_box([{"n": x["n"], "risk": x["physical_optimizer_risk"]} for x in expected_folds.values()]) if row["recipe"] == "qp_regularized" else None
        require(row["pooled_physical_risk"] == pooled and row["pooled_optimizer_risk"] == refined and
                row["last_fold_physical_risk"] == expected_folds["f3"]["physical_risk"] and
                row["selection_payload_bytes"] == expected_folds["f3"]["payload_bytes"],
                "Physical SSE/count pooling or frozen tie-breaking size differs")
    expected_selections = {label: independent_selection(record["candidate_rows"], label) for label in SELECTORS}
    require(record["selections"] == expected_selections and record["development_n"] == sum(scores[s]["n"] for s in stages),
            "Frozen development selections differ")
    winner_id = expected_selections["joint_rolling"]["candidate_id"]
    pool = selector_pool(record["candidate_rows"], "joint_rolling")
    winner = next(row for row in pool if row["candidate_id"] == winner_id)
    others = [row for row in pool if row["candidate_id"] != winner_id]
    competitor = min(others, key=lambda row: (q(row["pooled_optimizer_risk"]["lower"]), row["candidate_id"]))
    margin = q(competitor["pooled_optimizer_risk"]["lower"]) - q(winner["pooled_optimizer_risk"]["upper"])
    expected_stability = {"schema": "step39-development-optimization-selection-stability-v1", "selected_id": winner_id,
        "selected_exact_optimum_risk": winner["pooled_optimizer_risk"],
        "competitor_minimum_lower": competitor["pooled_optimizer_risk"]["lower"],
        "nearest_competitor_id": competitor["candidate_id"], "strict_margin_lower": str(margin),
        "status": "CERTIFIED_UNIQUE_WINNER_UNDER_EXACT_OPTIMIZATION" if margin > 0 else "UNRESOLVED",
        "fixed_fold_maps_objectives_and_extension_weights": True, "future_performance_claim": False}
    require(record["optimization_selection_stability"] == expected_stability,
            "Development solver-refinement selection stability differs")
    return {"status": "PASS", "candidate_configurations": len(ids), "selection_count": 12,
            "development_n": record["development_n"], "selected_ids": {k: v["candidate_id"] for k, v in expected_selections.items()},
            "optimization_selection_stability": expected_stability}


def audit_practical_gate(new, reference, ordinary, record):
    positive = q(reference["lower"]) > 0
    improved = positive and 20 * q(new["upper"]) <= 19 * q(reference["lower"])
    competitive = q(new["upper"]) <= q(ordinary["lower"])
    expected = {"reference_positive_risk": positive, "at_least_five_percent_mse_improvement": improved,
                "no_worse_than_selected_ordinary": competitive, "numerical_criterion_pass": improved and competitive,
                "relative_gain_status": "DEFINED" if positive else "UNDEFINED_RELATIVE_GAIN",
                "population_noninferiority_claim": False}
    require(record == expected, "Frozen five-percent/baseline comparison arithmetic differs")
    return expected


def audit_phase(protocol, phase, task=None, stage=None):
    dest = ROOT / "output" / (task or "global") / (stage or phase)
    start_path, end_path = dest / (phase + "_started39.json"), dest / (phase + "_completed39.json")
    begin, end = read(start_path), read(end_path)
    i, e1 = event_once("phase_started", phase=phase, task=task, stage=stage)
    j, e2 = event_once("phase_completed", phase=phase, task=task, stage=stage)
    require(i < j and begin["schema"] == "step39-phase-start-v1" and end["schema"] == "step39-phase-completion-v1" and
            begin["phase"] == end["phase"] == phase and begin["task"] == end["task"] == task and
            begin["stage"] == end["stage"] == stage and end["status"] == "COMPLETE" and
            begin["protocol_seal_sha256"] == sha(ROOT / "protocol_seal39.json") and
            e1["marker_path"] == str(start_path.relative_to(ROOT)) and e1["marker_sha256"] == sha(start_path) and
            e2["receipt_sha256"] == sha(end_path) and isfinite(end["wall_seconds"]) and
            0 <= end["wall_seconds"] <= protocol["execution"]["phase_budget_seconds"][phase],
            "Scientific phase identity, one-pass order or fixed budget differs")
    require(not (dest / (phase + "_failure39.json")).exists(), "A stopped scientific phase was promoted")
    return i, j, end


def verify_gate(relative, event_name, **bindings):
    path = checked_path(relative)
    gate = read(path)
    require(gate["status"] == "PASS" and gate["auditor_source_sha256"] == sha(Path(__file__)),
            "Required independent gate is absent, failed, or from another verifier")
    for key, value in bindings.items():
        require(gate[key] == value, "Independent gate ancestry differs: " + key)
    fields = {key: bindings[key] for key in ("task", "stage") if key in bindings}
    index, event = event_once(event_name, **fields)
    require(event["gate_path"] == relative and event["gate_sha256"] == sha(path),
            "Independent PASS gate changed after its event")
    return gate, index


def publish_gate(relative, event_name, result):
    path = checked_path(relative)
    require(not path.exists(), "Independent primary gate already exists; no gate rewrite")
    result = {**result, "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
              "auditor_source_sha256": sha(Path(__file__))}
    write(path, result)
    append_event(event_name, gate_path=relative, gate_sha256=sha(path),
                 **{key: result[key] for key in ("task", "stage") if key in result})
    return result


def audit_artificial_receipts():
    names = ("data_preflight39.json", "artificial_learners39.json", "artificial_audit39.json", "artificial_runner39.json")
    records = {}
    for name in names:
        path = ROOT / "tests" / name
        record = read(path)
        require(record["status"] == "PASS" and record["artificial_only"] is True,
                "Required pre-freeze artificial check is not PASS: " + name)
        if "public_measurement_values_decoded_or_parsed" in record:
            require(record["public_measurement_values_decoded_or_parsed"] == 0, "Artificial check decoded public measurements")
        if "scientific_data_accessed" in record:
            require(record["scientific_data_accessed"] is False, "Artificial check used scientific data")
        for relative, wanted in record["source_sha256"].items():
            require(sha(checked_path(relative)) == wanted, "Artificial check did not bind current source: " + relative)
        if "fixture_path" in record and "fixture_sha256" in record:
            require(sha(checked_path(record["fixture_path"])) == record["fixture_sha256"], "Artificial witness fixture changed")
        records[name] = {"sha256": sha(path), "status": "PASS", "checks": record.get("check_count", len(record.get("checks", {})))}
    require(read(ROOT / "tests/artificial_audit39.json")["fixture_sha256"] == sha(ROOT / "tests/learner_fixture39.json"),
            "Independent artificial audit belongs to another producer fixture")
    return records


def preflight():
    started = time.perf_counter()
    protocol, seal = verify_protocol_seal()
    events = audit_events()
    require(len(events) == 1 and events[0]["event"] == "protocol_frozen" and
            events[0]["protocol_seal_sha256"] == sha(ROOT / "protocol_seal39.json"),
            "New scientific measurements or repeated audit preceded the protocol preflight")
    artificial = audit_artificial_receipts()
    calendars = {task: audit_calendar(task) for task in TASKS}
    roster = read(ROOT / "inputs/calendar_roster39.json")
    require(roster["schema"] == "step39-frozen-calendar-roster-v1" and
            roster["measurement_values_decoded_or_parsed"] == 0 and roster["query_horizon_hours"] == 24 and
            roster["query_lag_hours"] == 24 and roster["daily_target_hour"] == 23,
            "Metadata roster exceeded approved scope")
    for task in TASKS:
        for stage in STAGES:
            frozen_queries(task, stage, "train")
            if stage != "final":
                frozen_queries(task, stage, "validation")
        frozen_queries(task, "final", "test")
        require(roster["tasks"][task]["projection_sha256"] == sha(ROOT / f"inputs/{task}/calendar_projection39.json"),
                "Frozen roster source projection changed")
    exposure = read(ROOT / "inputs/prior_exposure39.json")
    require(exposure["matches"]["tetouan_zone1"] == exposure["matches"]["metro_traffic"] == [] and
            exposure["private_or_reserved_arrays_read"] is False and exposure["file_count"] > 0,
            "Scoped fresh-source exposure record differs")
    prior = protocol["prior_evidence"]
    require(prior["step38_result_sha256"] == sha(ROOT / "inputs/prior/result38.json") and
            prior["step38_prediction_corollary_sha256"] == sha(ROOT / "inputs/prior/optimization_prediction_corollary38.json") and
            prior["prior_steps_immutable"] is True and prior["reserved_patients_accessed"] == 0,
            "Prior evidence/scientific scope changed")
    result = {"schema": "step39-independent-preexecution-gate-v1", "protocol_sha256": sha(ROOT / "protocol39.json"),
              "protocol_seal_sha256": sha(ROOT / "protocol_seal39.json"), "bound_files": len(seal["bound_files"]),
              "protocol": audit_protocol(protocol), "calendars": calendars, "artificial_checks": artificial,
              "new_measurement_values_decoded": 0, "no_new_scientific_fits": True,
              "wall_seconds": time.perf_counter() - started}
    return publish_gate("audit/preexecution_gate39.json", "independent_preexecution_PASS", result)


def stage_transition(task, stage):
    _, initial_index = verify_gate("audit/preexecution_gate39.json", "independent_preexecution_PASS",
        protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
    if stage in ("f2", "f3"):
        previous = "f1" if stage == "f2" else "f2"
        _, initial_index = verify_gate(f"audit/{task}/{previous}/validation_gate39.json", "independent_validation_PASS",
            task=task, stage=previous, protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"),
            completion_seal_sha256=sha(ROOT / f"output/{task}/{previous}/completion_seal39.json"))
    elif stage == "final":
        _, initial_index = verify_gate(f"audit/{task}/selection_gate39.json", "independent_selection_PASS",
            task=task, protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"),
            selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal39.json"))
    return initial_index


def fit_gate(task, stage):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    require(task in TASKS and stage in STAGES, "Unknown fit audit task/stage")
    transition_index = stage_transition(task, stage)
    relative = f"output/{task}/{stage}/fit_seal39.json"
    seal = verify_seal(relative, "step39-fit-seal-v1", task=task, stage=stage,
                       protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
    manifest_path = ROOT / f"output/{task}/{stage}/fit_manifest39.json"
    require(seal["manifest_sha256"] == sha(manifest_path), "Fit seal has another training manifest")
    required = {f"output/{task}/{stage}/cohorts/train.json", f"output/{task}/{stage}/cohorts/train_X.npy",
                f"output/{task}/{stage}/normalization39.json", str(manifest_path.relative_to(ROOT)),
                f"output/{task}/{stage}/representation_metadata39.json"}
    manifest = read(manifest_path)
    for info in manifest["maps"].values():
        required.update({info["map_path"], info["training_features"]["path"]})
    for row in manifest["candidate_rows"]:
        required.update({row["fit_path"], row["head_path"]})
    required.update(row["payload_path"] for row in manifest["baseline_rows"])
    require(required <= set(seal["bound_files"]), "Fit seal omitted a retained training/model dependency")
    begin, end, phase = audit_phase(protocol, "fit", task, stage)
    access_begin, _ = event_once("train_measurement_access_started", task=task, stage=stage)
    access_end, _ = event_once("train_measurement_access_completed", task=task, stage=stage)
    seal_index, event = event_once("evidence_sealed", schema="step39-fit-seal-v1", task=task, stage=stage)
    require(transition_index < begin < access_begin < access_end < end < seal_index and
            event["path"] == relative and event["sha256"] == sha(ROOT / relative), "Fitting access/seal phase order differs")
    training = audit_cohort(task, stage, "train")
    future = frozen_queries(task, stage, "validation") if stage != "final" else frozen_queries(task, stage, "test")
    require(max(training["target_timestamps"]) <= min(r["origin_timestamp"] for r in future),
            "Training target is later than the first evaluation forecast origin")
    summary = audit_stage_models(protocol, task, stage, training)
    require(phase["training_n"] == len(training["ids"]) and phase["certified_heads"] == 198 and phase["baselines"] == 9,
            "Fit phase count summary differs")
    result = {"schema": "step39-independent-fit-gate-v1", "task": task, "stage": stage,
              "protocol_seal_sha256": sha(ROOT / "protocol_seal39.json"), "fit_seal_sha256": sha(ROOT / relative),
              "training_cohort_sha256": training["sha256"], "training_n": len(training["ids"]),
              "models": summary, "all_fit_data_at_or_before_held_out_origins": True,
              "new_scientific_model_fits_by_auditor": 0, "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/{stage}/fit_gate39.json", "independent_fit_PASS", result)


def validation_gate(task, stage):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    require(task in TASKS and stage in STAGES[:3], "Unknown development audit task/stage")
    _, fit_index = verify_gate(f"audit/{task}/{stage}/fit_gate39.json", "independent_fit_PASS", task=task, stage=stage,
        protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"),
        fit_seal_sha256=sha(ROOT / f"output/{task}/{stage}/fit_seal39.json"))
    verify_seal(f"output/{task}/{stage}/fit_seal39.json", "step39-fit-seal-v1", task=task, stage=stage,
                protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
    relative = f"output/{task}/{stage}/completion_seal39.json"
    seal = verify_seal(relative, "step39-fold-completion-seal-v1", task=task, stage=stage,
                       protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
    base = f"output/{task}/{stage}"
    require(seal["fit_seal_sha256"] == sha(ROOT / f"{base}/fit_seal39.json") and
            seal["validation_scores_sha256"] == sha(ROOT / f"{base}/validation_scores39.json") and
            {f"{base}/validation_predictions39.json", f"{base}/validation_scores39.json",
             f"{base}/cohorts/validation.json", f"{base}/cohorts/validation_X.npy"} <= set(seal["bound_files"]),
            "Validation completion seal omitted an evaluation dependency")
    begin, end, phase = audit_phase(protocol, "validate", task, stage)
    access_begin, _ = event_once("validation_measurement_access_started", task=task, stage=stage)
    access_end, _ = event_once("validation_measurement_access_completed", task=task, stage=stage)
    seal_index, event = event_once("evidence_sealed", schema="step39-fold-completion-seal-v1", task=task, stage=stage)
    require(fit_index < begin < access_begin < access_end < end < seal_index and event["path"] == relative and
            event["sha256"] == sha(ROOT / relative), "Development access/scoring seal order differs")
    cohort = audit_cohort(task, stage, "validation")
    context = load_context(task, stage)
    bundle = read(ROOT / f"{base}/validation_predictions39.json")
    require(bundle["role"] == "validation", "Development prediction role differs")
    prediction_check = audit_prediction_bundle(protocol, task, stage, bundle, context,
                                                cohort["X"], cohort["ids"], cohort["persistence"])
    scores = read(ROOT / f"{base}/validation_scores39.json")
    scoring = audit_validation_values(protocol, task, stage, scores, bundle, context["manifest"], cohort)
    require(phase["validation_n"] == len(cohort["ids"]), "Development phase count summary differs")
    result = {"schema": "step39-independent-validation-gate-v1", "task": task, "stage": stage,
              "protocol_seal_sha256": sha(ROOT / "protocol_seal39.json"),
              "fit_seal_sha256": sha(ROOT / f"{base}/fit_seal39.json"), "completion_seal_sha256": sha(ROOT / relative),
              "validation_scores_sha256": sha(ROOT / f"{base}/validation_scores39.json"),
              "predictions": prediction_check, "scoring": scoring, "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/{stage}/validation_gate39.json", "independent_validation_PASS", result)


def selection_gate(task):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    scores, indices = {}, []
    for stage in STAGES[:3]:
        _, index = verify_gate(f"audit/{task}/{stage}/validation_gate39.json", "independent_validation_PASS", task=task, stage=stage,
            protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"),
            completion_seal_sha256=sha(ROOT / f"output/{task}/{stage}/completion_seal39.json"))
        verify_seal(f"output/{task}/{stage}/completion_seal39.json", "step39-fold-completion-seal-v1", task=task, stage=stage,
                    protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
        scores[stage] = read(ROOT / f"output/{task}/{stage}/validation_scores39.json")
        indices.append(index)
    relative = f"output/{task}/selection_seal39.json"
    seal = verify_seal(relative, "step39-selection-seal-v1", task=task,
                       protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
    record = read(ROOT / f"output/{task}/selection39.json")
    require(record["task"] == task and seal["selection_sha256"] == sha(ROOT / f"output/{task}/selection39.json") and
            seal["selected_ids"] == {key: value["candidate_id"] for key, value in record["selections"].items()},
            "Development selection seal differs")
    require(record["fold_score_sha256"] == {s: sha(ROOT / f"output/{task}/{s}/validation_scores39.json") for s in STAGES[:3]},
            "Development pooling used another set of fold scores")
    begin, end, phase = audit_phase(protocol, "select", task)
    seal_index, event = event_once("evidence_sealed", schema="step39-selection-seal-v1", task=task, stage=None)
    require(max(indices) < begin < end < seal_index and event["path"] == relative and event["sha256"] == sha(ROOT / relative),
            "Selection began before all independent development checks or was not frozen")
    values = audit_selection_values(scores, record)
    require(phase["selected_ids"] == values["selected_ids"], "Selection phase completion names another choice")
    result = {"schema": "step39-independent-selection-gate-v1", "task": task,
              "protocol_seal_sha256": sha(ROOT / "protocol_seal39.json"), "selection_seal_sha256": sha(ROOT / relative),
              "selection_sha256": sha(ROOT / f"output/{task}/selection39.json"), "selection": values,
              "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/selection_gate39.json", "independent_selection_PASS", result)


def pretest(task):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    _, selected_index = verify_gate(f"audit/{task}/selection_gate39.json", "independent_selection_PASS", task=task,
        protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"),
        selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal39.json"))
    fit, fit_index = verify_gate(f"audit/{task}/final/fit_gate39.json", "independent_fit_PASS", task=task, stage="final",
        protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"),
        fit_seal_sha256=sha(ROOT / f"output/{task}/final/fit_seal39.json"))
    verify_seal(f"output/{task}/final/fit_seal39.json", "step39-fit-seal-v1", task=task, stage="final",
                protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
    selection_seal = verify_seal(f"output/{task}/selection_seal39.json", "step39-selection-seal-v1", task=task,
                protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
    manifest = read(ROOT / f"output/{task}/final/fit_manifest39.json")
    require(selected_index < fit_index and manifest["selection_seal_sha256"] == sha(ROOT / f"output/{task}/selection_seal39.json") and
            set(selection_seal["selected_ids"]) == set(SELECTORS) and
            set(selection_seal["selected_ids"].values()) <= set(protocol["evaluation"]["prediction_ids"]),
            "Final fitted concrete heads do not implement the frozen configurations")
    forbidden = [e for e in audit_events() if e.get("task") == task and e["event"].startswith("test_")]
    require(not forbidden, "Test measurement/prediction activity preceded independent pretest PASS")
    result = {"schema": "step39-independent-pretest-gate-v1", "task": task,
              "protocol_seal_sha256": sha(ROOT / "protocol_seal39.json"),
              "fit_seal_sha256": sha(ROOT / f"output/{task}/final/fit_seal39.json"),
              "selection_seal_sha256": sha(ROOT / f"output/{task}/selection_seal39.json"),
              "selected_configuration_ids": selection_seal["selected_ids"], "training_n": fit["training_n"],
              "test_n": len(frozen_queries(task, "final", "test")), "all207_predictions_required_before_each_target": True,
              "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/pretest_gate39.json", "independent_pretest_PASS", result)


def audit_test_forecasts(protocol, task, context):
    import numpy as np
    queries = frozen_queries(task, "final", "test")
    _, pretest_index = verify_gate(f"audit/{task}/pretest_gate39.json", "independent_pretest_PASS", task=task,
        protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"),
        fit_seal_sha256=sha(ROOT / f"output/{task}/final/fit_seal39.json"),
        selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal39.json"))
    cohort_path = ROOT / f"output/{task}/test/cohorts/test.json"
    cohort = read(cohort_path)
    require(cohort["schema"] == "step39-completed-test-cohort-v1" and cohort["task"] == task and
            cohort["n"] == len(queries) and cohort["ids"] == [r["id"] for r in queries] and
            cohort["target_timestamps"] == [r["target_timestamp"] for r in queries] and
            cohort["origin_timestamps"] == [r["origin_timestamp"] for r in queries] and
            cohort["sequential_prediction_then_label"] is True and len(cohort["queries"]) == len(queries),
            "Completed test cohort changed its frozen calendar roster")
    state = read(ROOT / f"output/{task}/test/access_state39.json")
    model_seal = sha(ROOT / f"output/{task}/final/fit_seal39.json")
    require(state["schema"] == "step39-chronological-test-access-state-v1" and state["task"] == task and
            state["status"] == "COMPLETE" and state["next_index"] == len(queries) and state["pending_query"] is None and
            state["previously_revealed_test_targets"] == cohort["target_timestamps"] and state["model_seal_sha256"] == model_seal,
            "Test access state is not exactly the complete frozen query prefix")
    ids = protocol["evaluation"]["prediction_ids"]
    predictions = {cid: [] for cid in ids}
    qp_ids = [r["candidate_id"] for r in context["manifest"]["candidate_rows"] if r["recipe"] == "qp_regularized"]
    optimizer = {cid: [] for cid in qp_ids}
    raw_targets, targets, timing, query_summaries = [], [], [], []
    normal = context["manifest"]["normalization"]
    low, span = q(normal["target_min"]), q(normal["span"])
    normal_hash = sha(ROOT / f"output/{task}/final/normalization39.json")
    previous_reveal = pretest_index
    overlap_queries = 0
    distance_count = 0
    for i, query in enumerate(queries):
        qid = query["id"]
        history_rel = f"output/{task}/test/history/{i:03d}.json"
        target_rel = f"output/{task}/test/targets/{i:03d}.json"
        seal_rel = f"output/{task}/test/prediction_seals/{i:03d}.json"
        prediction_rel = f"output/{task}/test/predictions/q{i:03d}.json"
        history, target, seal, prediction = [read(ROOT / r) for r in (history_rel, target_rel, seal_rel, prediction_rel)]
        hstart, hs = event_once("test_history_access_started", task=task, query_id=qid)
        hend, he = event_once("test_history_access_completed", task=task, query_id=qid)
        pwrite, pw = event_once("test_prediction_receipt_written", task=task, query_id=qid)
        pseal, ps = event_once("test_predictions_sealed", task=task, query_id=qid)
        tstart, ts = event_once("test_target_access_started", task=task, query_id=qid)
        treveal, tr = event_once("test_target_revealed", task=task, query_id=qid)
        require(previous_reveal < hstart < hend < pwrite < pseal < tstart < treveal,
                "Chronological history→complete prediction seal→target order differs")
        previous_reveal = treveal
        lags = lag_stamps(query)
        overlap = [stamp for stamp in lags if stamp in cohort["target_timestamps"][:i]]
        history_values = authorized_values(task, lags)
        history_y = [history_values[stamp] for stamp in lags]
        persistence = (history_values[query["origin_timestamp"]] - low) / span
        X = np.asarray([[float(value) for value in history_y]], dtype=np.float64)
        require(history["schema"] == "step39-causal-test-history-v1" and history["task"] == task and
                history["query_id"] == qid and history["query_index"] == i and
                history["target_timestamp"] == query["target_timestamp"] and history["origin_timestamp"] == query["origin_timestamp"] and
                history["history_timestamps"] == lags and list(map(q, history["history_values"])) == history_y and
                q(history["persistence"]) == persistence and q(history["raw_persistence"]) == history_y[-1] and
                history["model_seal_sha256"] == model_seal and history["normalization_sha256"] == normal_hash and
                history["previously_revealed_test_targets_in_history"] == overlap and
                history["current_target_value_decoded"] is False and history["other_measurement_predictors_used"] is False and
                history["X_path"] == f"output/{task}/test/history/{i:03d}_X.npy" and
                history["X_sha256"] == sha(ROOT / history["X_path"]), "Causal test-history record differs")
        saved_X = np.load(ROOT / history["X_path"], allow_pickle=False)
        require(saved_X.dtype == np.float64 and np.isfinite(saved_X).all() and np.array_equal(saved_X, X),
                "Test input differs from exact authorized raw history converted to binary64")
        audit_access(history["access"], task, "final", "test_history", lags, query["origin_timestamp"])
        require(hs["query_index"] == he["query_index"] == i and hs["model_seal_sha256"] == model_seal and
                hs["maximum_allowed_timestamp"] == query["origin_timestamp"] and he["history_receipt_path"] == history_rel and
                he["history_receipt_sha256"] == sha(ROOT / history_rel) and
                he["maximum_decoded_timestamp"] == query["origin_timestamp"] and
                he["previously_revealed_test_targets_in_history"] == overlap and he["current_target_value_decoded"] is False,
                "Test-history event binding differs")
        require(seal["schema"] == "step39-test-prediction-byte-seal-v1" and seal["task"] == task and
                seal["query_id"] == qid and seal["query_index"] == i and seal["model_seal_sha256"] == model_seal and
                seal["prediction_path"] == prediction_rel and seal["prediction_sha256"] == sha(ROOT / prediction_rel) and
                seal["history_receipt_path"] == history_rel and seal["history_receipt_sha256"] == sha(ROOT / history_rel) and
                seal["prediction_ids"] == ids and seal["prediction_count"] == 207 and seal["current_target_value_decoded"] is False,
                "Current-query complete prediction byte seal differs")
        require(pw["prediction_path"] == prediction_rel and pw["prediction_sha256"] == sha(ROOT / prediction_rel) and
                pw["prediction_count"] == 207 and ps["query_index"] == i and ps["prediction_seal_path"] == seal_rel and
                ps["prediction_seal_sha256"] == sha(ROOT / seal_rel) and ps["prediction_path"] == prediction_rel and
                ps["prediction_sha256"] == sha(ROOT / prediction_rel) and ps["prediction_count"] == 207 and
                ps["current_target_value_decoded"] is False and ts["query_index"] == i and
                ts["prediction_seal_sha256"] == sha(ROOT / seal_rel) and ts["target_timestamp"] == query["target_timestamp"],
                "Prediction-before-target event ancestry differs")
        require(prediction["role"] == f"q{i:03d}" and prediction["query_id"] == qid and
                prediction["origin_timestamp"] == query["origin_timestamp"] and prediction["target_timestamp"] == query["target_timestamp"] and
                prediction["model_seal_sha256"] == model_seal and prediction["history_receipt_path"] == history_rel and
                prediction["history_receipt_sha256"] == sha(ROOT / history_rel) and
                prediction["sealed_before_current_target_reveal"] is True and prediction["future_target_used_in_prediction"] is False,
                "Sealed forecast identity/causal scope differs")
        math = audit_prediction_bundle(protocol, task, "final", prediction, context, X, [qid], [persistence])
        distance_count += math["query_anchor_distances"]
        raw = authorized_values(task, [query["target_timestamp"]])[query["target_timestamp"]]
        normalized = (raw - low) / span
        require(target["schema"] == "step39-revealed-test-target-v1" and target["task"] == task and target["query_id"] == qid and
                target["query_index"] == i and target["target_timestamp"] == query["target_timestamp"] and
                target["origin_timestamp"] == query["origin_timestamp"] and q(target["raw_target"]) == raw and q(target["y"]) == normalized and
                target["normalization_sha256"] == normal_hash and target["prediction_seal_path"] == seal_rel and
                target["prediction_seal_sha256"] == sha(ROOT / seal_rel) and target["model_seal_sha256"] == model_seal and
                target["target_value_clipped"] is False, "Original revealed test target or normalization differs")
        audit_access(target["access"], task, "final", "test_target", [query["target_timestamp"]], query["target_timestamp"])
        require(tr["query_index"] == i and tr["target_path"] == target_rel and tr["target_sha256"] == sha(ROOT / target_rel) and
                tr["prediction_seal_sha256"] == sha(ROOT / seal_rel) and tr["next_query_index"] == i + 1 and
                tr["chronological_test_status"] == ("COMPLETE" if i + 1 == len(queries) else "READY"),
                "Target reveal did not advance exactly one frozen query")
        expected_query_record = {**query, "prediction_path": prediction_rel, "prediction_sha256": sha(ROOT / prediction_rel),
            "history_receipt_path": history_rel, "history_receipt_sha256": sha(ROOT / history_rel),
            "target_reveal": {**target, "path": target_rel, "sha256": sha(ROOT / target_rel)}}
        require(cohort["queries"][i] == expected_query_record, "Completed cohort query ancestry differs")
        targets.append(normalized)
        raw_targets.append(raw)
        timing.append(prediction["timing"])
        overlap_queries += bool(overlap)
        for cid in ids:
            predictions[cid].append(prediction["predictions"][cid]["predictions"][0])
        for cid in qp_ids:
            optimizer[cid].append(prediction["optimizer_predictions"][cid]["predictions"][0])
        query_summaries.append({"query_id": qid, "arithmetic_head_predictions": 198,
                                "optimizer_prediction_enclosures": 162, "baseline_predictions": 9,
                                "history_reuses_prior_target": bool(overlap),
                                "prediction_before_target_verified": True})
        if i % 10 == 0 or i + 1 == len(queries):
            print(json.dumps({"audit": "TEST_QUERY", "task": task, "completed": i + 1, "total": len(queries)}), flush=True)
    require(list(map(q, cohort["y"])) == targets and list(map(q, cohort["raw_targets"])) == raw_targets,
            "Completed test target vector differs from individual authorized reveals")
    all_reveals = [e["query_id"] for e in audit_events() if e["event"] == "test_target_revealed" and e.get("task") == task]
    require(all_reveals == cohort["ids"], "Test access includes a hidden, repeated or omitted target reveal")
    return {"cohort": cohort, "queries": queries, "predictions": predictions, "optimizer_predictions": optimizer,
            "targets": targets, "raw_targets": raw_targets, "query_timing": timing,
            "summary": {"status": "PASS", "test_n": len(queries), "prediction_seals": len(queries),
                        "head_predictions": len(queries) * 198, "optimizer_prediction_enclosures": len(queries) * 162,
                        "baseline_predictions": len(queries) * 9, "query_anchor_distances": distance_count,
                        "histories_reusing_earlier_revealed_targets": overlap_queries, "queries": query_summaries}}


def audit_test_result(protocol, task, context, test):
    base = ROOT / f"output/{task}/test"
    result = read(base / "result39.json")
    primary = read(base / "primary_result39.json")
    selection = read(ROOT / f"output/{task}/selection39.json")
    manifest = context["manifest"]
    require(result["schema"] == "step39-completed-task-result-v1" and result["status"] == "COMPLETE_PENDING_INDEPENDENT_AUDIT" and
            result["task"] == task and result["protocol_sha256"] == sha(ROOT / "protocol39.json") and
            result["protocol_seal_sha256"] == sha(ROOT / "protocol_seal39.json") and result["primary"] == primary and
            result["primary_result_sha256"] == sha(base / "primary_result39.json") and result["new_scientific_passes"] == 1 and
            result["training_n"] == manifest["training_n"] and result["development_n"] == selection["development_n"],
            "Task result primary/protocol ancestry differs")
    require(primary["schema"] == "step39-task-primary-result-v1" and primary["task"] == task and
            primary["test_n"] == len(test["targets"]) and primary["normalization"] == manifest["normalization"] and
            primary["query_timing"] == test["query_timing"] and isfinite(primary["forecast_seconds"]) and primary["forecast_seconds"] >= 0 and
            primary["model_seal_sha256"] == sha(ROOT / f"output/{task}/final/fit_seal39.json") and
            primary["selection_seal_sha256"] == sha(ROOT / f"output/{task}/selection_seal39.json") and
            primary["pretest_gate_sha256"] == sha(ROOT / f"audit/{task}/pretest_gate39.json") and
            primary["development_selection_stability"] == selection["optimization_selection_stability"] and
            primary["test_reselection"] is False and primary["assessment_oracle_fits"] == 0 and
            primary["future_response_coverage_claim"] is False and primary["population_risk_claim"] is False,
            "Primary test scope/model/selection binding differs")
    rows = primary["candidate_rows"]
    original = manifest["candidate_rows"] + manifest["baseline_rows"]
    require([row["candidate_id"] for row in rows] == protocol["evaluation"]["prediction_ids"],
            "Primary result omitted/reordered a retained candidate")
    scale = q(manifest["normalization"]["span"]) ** 2
    for row, source in zip(rows, original):
        cid = row["candidate_id"]
        require(all(row[key] == value for key, value in source.items()), "Primary result changes a fitted final head/baseline")
        a38.audit_loss(test["predictions"][cid], test["targets"], row["test_risk"])
        require(row["physical_test_risk"] == scaled_box(row["test_risk"], scale) and
                row["test_predictions_digest"] == digest(test["predictions"][cid]), "Test physical scale/prediction digest differs")
        if row["recipe"] == "qp_regularized":
            require(row["physical_optimizer_risk"] == scaled_box(loss_box(test["optimizer_predictions"][cid], test["targets"]), scale) and
                    row["optimizer_predictions_digest"] == digest(test["optimizer_predictions"][cid]),
                    "Exact-optimizer test-risk enclosure/digest differs")
        else:
            require(row["physical_optimizer_risk"] is None and row["optimizer_predictions_digest"] is None,
                    "Unsupported optimizer claim for control/baseline")
        require(set(row["monthly"]) == set(protocol["evaluation"]["months"]), "Descriptive month roster differs")
        for month in protocol["evaluation"]["months"]:
            indices = [i for i, query in enumerate(test["queries"]) if query["target_timestamp"].startswith(month)]
            value = row["monthly"][month]
            require(value["n"] == len(indices) and len(indices) > 0, "Descriptive month count differs")
            a38.audit_loss([test["predictions"][cid][i] for i in indices], [test["targets"][i] for i in indices], value["risk"])
            require(value["physical_risk"] == scaled_box(value["risk"], scale), "Descriptive month physical risk differs")
    lookup = {row["candidate_id"]: row for row in rows}
    selected = {label: lookup[value["candidate_id"]] for label, value in selection["selections"].items()}
    require(primary["selected"] == selected and set(selected) == set(SELECTORS),
            "Frozen development configuration was changed after test")
    gate = audit_practical_gate(selected["joint_rolling"]["physical_test_risk"],
                                selected["default_qp_rolling"]["physical_test_risk"],
                                selected["ordinary_rolling"]["physical_test_risk"], primary["primary_gate"])
    return {"status": "PASS", "candidate_losses": 207, "optimizer_loss_enclosures": 162, "monthly_losses": 621,
            "selected_configuration_ids": {k: v["candidate_id"] for k, v in selected.items()}, "primary_gate": gate,
            "development_selection_stability": primary["development_selection_stability"],
            "original_test_labels_not_clipped": True, "assessment_oracle_fits": 0}


def postrun(task):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    stage_checks = {}
    for stage in STAGES:
        fit_relative = f"output/{task}/{stage}/fit_seal39.json"
        verify_seal(fit_relative, "step39-fit-seal-v1", task=task, stage=stage,
                    protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
        fit, _ = verify_gate(f"audit/{task}/{stage}/fit_gate39.json", "independent_fit_PASS", task=task, stage=stage,
            protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"), fit_seal_sha256=sha(ROOT / fit_relative))
        stage_checks[stage] = {"fit_gate_sha256": sha(ROOT / f"audit/{task}/{stage}/fit_gate39.json"),
                               "training_n": fit["training_n"]}
        if stage != "final":
            complete_relative = f"output/{task}/{stage}/completion_seal39.json"
            verify_seal(complete_relative, "step39-fold-completion-seal-v1", task=task, stage=stage,
                        protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
            verify_gate(f"audit/{task}/{stage}/validation_gate39.json", "independent_validation_PASS", task=task, stage=stage,
                protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"), completion_seal_sha256=sha(ROOT / complete_relative))
            stage_checks[stage]["validation_gate_sha256"] = sha(ROOT / f"audit/{task}/{stage}/validation_gate39.json")
    verify_seal(f"output/{task}/selection_seal39.json", "step39-selection-seal-v1", task=task,
                protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"))
    verify_gate(f"audit/{task}/selection_gate39.json", "independent_selection_PASS", task=task,
        protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"), selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal39.json"))
    _, pretest_index = verify_gate(f"audit/{task}/pretest_gate39.json", "independent_pretest_PASS", task=task,
        protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"), fit_seal_sha256=sha(ROOT / f"output/{task}/final/fit_seal39.json"),
        selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal39.json"))
    failed = [e for e in audit_events() if e.get("task") == task and
              (e["event"].endswith("_failed") or e["event"] == "phase_stopped_error")]
    require(not failed, "A stopped scientific phase or access failure was followed by a result")
    context = load_context(task, "final")
    test = audit_test_forecasts(protocol, task, context)
    score = audit_test_result(protocol, task, context, test)
    begin, end, phase = audit_phase(protocol, "test", task)
    first_history, _ = event_once("test_history_access_started", task=task, query_id=test["queries"][0]["id"])
    last_reveal, _ = event_once("test_target_revealed", task=task, query_id=test["queries"][-1]["id"])
    result_path = ROOT / f"output/{task}/test/result39.json"
    require(pretest_index < begin < first_history < last_reveal < end and phase["test_n"] == len(test["targets"]) and
            phase["primary_gate"] == score["primary_gate"] and phase["result_sha256"] == sha(result_path),
            "Final test phase occurred outside independent gate/access order")
    result = {"schema": "step39-independent-task-postrun-audit-v1", "task": task,
              "protocol_sha256": sha(ROOT / "protocol39.json"), "protocol_seal_sha256": sha(ROOT / "protocol_seal39.json"),
              "result_sha256": sha(result_path), "primary_result_sha256": sha(ROOT / f"output/{task}/test/primary_result39.json"),
              "selection_seal_sha256": sha(ROOT / f"output/{task}/selection_seal39.json"),
              "fit_seal_sha256": sha(ROOT / f"output/{task}/final/fit_seal39.json"),
              "stage_checks": stage_checks, "test": test["summary"], "scoring": score,
              "practical_transfer_success": score["primary_gate"]["numerical_criterion_pass"],
              "single_primary_pass": True, "all_candidate_outcomes_retained": True,
              "all_scientific_sources_and_prior_gate_inputs_unchanged": True,
              "new_scientific_model_fits_by_auditor": 0, "population_risk_claim": False, "future_response_coverage_claim": False,
              "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/postrun_audit39.json", "independent_postrun_PASS", result)


def audit_execution_plan(protocol):
    record = read(ROOT / "output/primary_execution39.json")
    commands = []
    for task in TASKS:
        for stage in STAGES[:3]:
            for source, phase in (("run39.py", "fit"), ("audit39.py", "fit"),
                                  ("run39.py", "validate"), ("audit39.py", "validate")):
                commands.append([source, phase, "--task", task, "--stage", stage])
        commands.extend([["run39.py", "select", "--task", task], ["audit39.py", "selection", "--task", task],
                         ["run39.py", "fit", "--task", task, "--stage", "final"],
                         ["audit39.py", "fit", "--task", task, "--stage", "final"],
                         ["audit39.py", "pretest", "--task", task], ["run39.py", "test", "--task", task],
                         ["audit39.py", "postrun", "--task", task]])
    commands.extend([["run39.py", "aggregate"], ["audit39.py", "aggregate"]])
    require(record["schema"] == "step39-one-pass-execution-v1" and record["commands"] == commands and len(commands) == 40 and
            record["protocol_seal_sha256"] == sha(ROOT / "protocol_seal39.json") and
            record["source_sha256"] == sha(ROOT / "source/execute39.py") and
            not (ROOT / "output/execution_stopped39.json").exists(), "Single-pass execution plan was changed or stopped")
    return {"status": "PASS", "commands": len(commands), "no_scientific_reruns": True}


def aggregate():
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    result_path = ROOT / "output/result39.json"
    result = read(result_path)
    require(result["schema"] == "step39-two-domain-result-v1" and result["step"] == 39 and result["status"] == "COMPLETE" and
            result["protocol_sha256"] == sha(ROOT / "protocol39.json") and
            result["protocol_seal_sha256"] == sha(ROOT / "protocol_seal39.json") and set(result["tasks"]) == set(TASKS) and
            result["domain_count"] == 2 and result["all_task_independent_audits_pass"] is True and
            result["test_reselection"] is False and result["assessment_oracle_fits"] == 0 and
            result["population_risk_claim"] is False and result["future_response_coverage_claim"] is False and
            result["reserved_patients_accessed"] == 0, "Combined result scientific scope or task roster differs")
    tasks, pass_all, indices = {}, True, []
    for task in TASKS:
        task_result_path = ROOT / f"output/{task}/test/result39.json"
        audit_path = ROOT / f"audit/{task}/postrun_audit39.json"
        checked, index = verify_gate(str(audit_path.relative_to(ROOT)), "independent_postrun_PASS", task=task,
            protocol_seal_sha256=sha(ROOT / "protocol_seal39.json"), result_sha256=sha(task_result_path))
        primary = read(task_result_path)["primary"]
        expected = {"result_path": str(task_result_path.relative_to(ROOT)), "result_sha256": sha(task_result_path),
                    "audit_path": str(audit_path.relative_to(ROOT)), "audit_sha256": sha(audit_path),
                    "primary_gate": primary["primary_gate"], "test_n": primary["test_n"], "selected": primary["selected"],
                    "development_selection_stability": primary["development_selection_stability"]}
        require(result["tasks"][task] == expected and checked["practical_transfer_success"] is primary["primary_gate"]["numerical_criterion_pass"],
                "Combined result differs from an independently verified task")
        pass_all = pass_all and primary["primary_gate"]["numerical_criterion_pass"]
        indices.append(index)
        tasks[task] = {"status": "PASS", "result_sha256": sha(task_result_path), "audit_sha256": sha(audit_path),
                       "test_n": primary["test_n"], "practical_transfer_success": checked["practical_transfer_success"]}
    require(result["practical_transfer_success"] is pass_all and
            result["primary_status"] == ("ESTABLISHED_ON_BOTH_FIXED_ASSESSMENTS" if pass_all else "NOT_ESTABLISHED"),
            "Overall criterion is not the conjunction of both task criteria")
    begin, end, phase = audit_phase(protocol, "aggregate")
    require(max(indices) < begin < end and phase["practical_transfer_success"] is pass_all and
            phase["result_sha256"] == sha(result_path), "Aggregation preceded independent task verification")
    execution = audit_execution_plan(protocol)
    receipt = {"schema": "step39-independent-two-domain-postrun-audit-v1", "protocol_sha256": sha(ROOT / "protocol39.json"),
               "protocol_seal_sha256": sha(ROOT / "protocol_seal39.json"), "result_sha256": sha(result_path),
               "tasks": tasks, "domain_count": 2, "practical_transfer_success": pass_all,
               "primary_status": result["primary_status"], "execution": execution,
               "all_candidate_outcomes_retained": True, "test_reselection": False, "assessment_oracle_fits": 0,
               "population_risk_claim": False, "future_response_coverage_claim": False,
               "wall_seconds": time.perf_counter() - started}
    return publish_gate("audit/postrun_audit39.json", "independent_aggregate_PASS", receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("preflight", "fit", "validate", "selection", "pretest", "postrun", "aggregate"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    try:
        if args.operation == "preflight":
            result = preflight()
        elif args.operation in ("fit", "validate"):
            require(args.task is not None and args.stage is not None, "Task and stage required")
            result = {"fit": fit_gate, "validate": validation_gate}[args.operation](args.task, args.stage)
        elif args.operation in ("selection", "pretest", "postrun"):
            require(args.task is not None, "Task required")
            result = {"selection": selection_gate, "pretest": pretest, "postrun": postrun}[args.operation](args.task)
        else:
            result = aggregate()
        wanted = ("schema", "status", "task", "stage", "protocol_seal_sha256", "fit_seal_sha256", "selection_seal_sha256",
                  "result_sha256", "practical_transfer_success", "primary_status", "wall_seconds")
        print(json.dumps({k: result[k] for k in wanted if k in result}, sort_keys=True), flush=True)
    except Exception as error:
        failure = {"schema": "step39-independent-audit-failure-v1", "status": "FAIL", "operation": args.operation,
                   "task": args.task, "stage": args.stage, "type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "scientific_runs_started_by_auditor": 0,
                   "auditor_source_sha256": sha(Path(__file__))}
        dest = ROOT / "audit" / (args.task or "global") / (args.stage or args.operation) / "retained_failure39.json"
        if not dest.exists():
            write(dest, failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
