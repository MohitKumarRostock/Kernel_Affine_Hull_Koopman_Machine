"""Independent Step40 verification of data chronology, models and certificates.

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
from functools import lru_cache
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
TASKS = ("household_power2009", "jena_temperature2016")
STAGES = ("f1", "f2", "f3", "final")
LAMBDAS = ("0",)
WEIGHTS = ("1/2",)
SELECTORS = ("joint_rolling", "joint_last_fold", "qp_rolling", "affine_rolling", "ordinary_rolling", "hgb_rolling",
             "kahm_joint_rolling", "raw_joint_rolling", "nystrom_joint_rolling")
YEARS = {"household_power2009": 2009, "jena_temperature2016": 2016}
STAGE_DATES = {
    task: {
        "f1": {"train_start": f"{year}-01-01", "train_stop": f"{year}-06-01", "validation_start": f"{year}-06-01", "validation_stop": f"{year}-07-01"},
        "f2": {"train_start": f"{year}-01-01", "train_stop": f"{year}-07-01", "validation_start": f"{year}-07-01", "validation_stop": f"{year}-08-01"},
        "f3": {"train_start": f"{year}-01-01", "train_stop": f"{year}-08-01", "validation_start": f"{year}-08-01", "validation_stop": f"{year}-09-01"},
        "final": {"train_start": f"{year}-01-01", "train_stop": f"{year}-09-01"},
    } for task, year in YEARS.items()
}
# Counts are fixed from metadata/status eligibility before measurement access.
EXPECTED_N = {
    "household_power2009": {"f1": {"train": 149, "validation": 26}, "f2": {"train": 175, "validation": 31},
                             "f3": {"train": 206, "validation": 30}, "final": {"train": 236}, "test": 91},
    "jena_temperature2016": {"f1": {"train": 150, "validation": 30}, "f2": {"train": 180, "validation": 31},
                             "f3": {"train": 211, "validation": 31}, "final": {"train": 242}, "test": 86},
}
REP_CONFIG = dict(a38.REP_CONFIG)
SOURCE_SPECS = {
    "household_power2009": {
        "uci_id": 235, "year": 2009, "separator": ";", "timestamp_indices": [0, 1], "target_index": 2,
        "target_header": "Global_active_power", "timestamp_headers": ["Date", "Time"], "header_encoding": "utf-8", "header_count": 9,
        "timestamp_formats": ["%d/%m/%Y %H:%M:%S"], "cadence_minutes": 1,
        "archives": [("household", "household_power_consumption_mirror.zip", "household_power_consumption.txt")],
        "missing_markers": ["", "?"], "signed": False, "numeric_domain": "finite nonnegative decimal",
        "unit": "kilowatt", "reported_instances": 2075259,
        "snapshot_description": "Minute-averaged active power at the recorded minute00; not an hourly average",
        "timezone_convention": "Source local-naive Date and Time; no UTC offset or daylight-saving reconstruction is invented",
        "missing_status_rule": "Predeclared exact empty or question-mark target field, after stripping surrounding ASCII spaces and tabs; no valid numeric token inspected",
    },
    "jena_temperature2016": {
        "uci_id": None, "year": 2016, "separator": ",", "timestamp_indices": [0], "target_index": 2,
        "target_header": "T (degC)", "timestamp_headers": ["Date Time"], "header_encoding": "latin-1", "header_count": 22,
        "timestamp_formats": ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"], "cadence_minutes": 10,
        "archives": [("jena2016a", "mpi_roof_2016a.zip", "mpi_roof_2016a.csv"),
                     ("jena2016b", "mpi_roof_2016b.zip", "mpi_roof_2016b.csv")],
        "missing_markers": [], "signed": True,
        "numeric_domain": "finite signed decimal at or above -273.15 degree Celsius",
        "unit": "degree Celsius", "reported_instances": None,
        "snapshot_description": "Recorded 10-minute-averaged air temperature at minute00, timestamp at end of averaging period; not an hourly average",
        "timezone_convention": "Source documentation specifies MEZ (CET) and interval-end timestamps; retain recorded naive times without a daylight-saving reconstruction",
        "missing_status_rule": "No measurement missing-status inspection; exact timestamp presence only",
    },
}


def test_dates(task):
    return f"{YEARS[task]}-09-01", f"{YEARS[task]}-12-01"


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
    path = ROOT / "output/access_and_execution40.jsonl"
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
    path = ROOT / "output/access_and_execution40.jsonl"
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


def csv_cells(record, separator=","):
    """Independent byte-field scanner; never decode unrequested values."""
    sep = separator.encode("ascii") if isinstance(separator, str) else separator
    require(len(sep) == 1, "Source separator must be a single byte")
    body = record.rstrip(b"\r\n")
    starts, quoted, i = [0], False, 0
    while i < len(body):
        if body[i] == 34:
            if quoted and i + 1 < len(body) and body[i + 1] == 34:
                i += 2
                continue
            quoted = not quoted
        elif body[i] == sep[0] and not quoted:
            starts.append(i + 1)
        i += 1
    require(not quoted, "Unfinished or multiline CSV quote")
    result = []
    for left, end in zip(starts, starts[1:] + [len(body) + 1]):
        field = body[left:end - 1]
        if field.startswith(b'"'):
            require(len(field) >= 2 and field.endswith(b'"'), "Malformed quoted source field")
            interior = field[1:-1]
            require(b'"' not in interior.replace(b'""', b''), "Unescaped quote within source field")
            field = interior.replace(b'""', b'"')
        else:
            require(b'"' not in field, "Embedded quote in unquoted source field")
        result.append(field)
    return result


@lru_cache(maxsize=4)
def cached_archive_member(relative, wanted, member):
    path = checked_path(relative)
    require(path.is_file() and sha(path) == wanted, "Official archive bytes changed")
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == [member], "Original archive member roster differs")
        return archive.read(member)


def source_payloads(task):
    require(task in TASKS, "Unknown official public task")
    spec = SOURCE_SPECS[task]
    source = read(ROOT / f"inputs/{task}/public_source40.json")
    require(source["schema"] == "step40-opaque-public-source-download-v1" and source["task"] == task and
            source["uci_id"] == spec["uci_id"] and source["year"] == spec["year"] and
            source["measurement_values_decoded_or_parsed"] == source["timestamp_values_parsed"] == 0 and
            source["scientific_model_fits"] == 0 and source["source_license"] == "CC BY 4.0" and
            source["absolute_historical_never_seen_claim"] is False,
            "Opaque official source identity/access scope differs")
    metadata_hash = source.get("metadata_sha256", source.get("metadata_html_sha256"))
    require(isinstance(metadata_hash, str) and sha(checked_path(source["metadata_path"])) == metadata_hash,
            "Primary metadata evidence changed")
    for relative, wanted in source.get("supplementary_metadata_files", {}).items():
        require(sha(checked_path(relative)) == wanted, "Supplementary transport provenance changed")
    if task == "household_power2009":
        commit = "509690881d2c47c89d3d221ed2e2bb30e5afa9c0"
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/" + commit + "/household_power_consumption.zip"
        require(source["transport"] == "public_repository_mirror" and source["mirror_commit"] == commit and
                source["mirror_repository"] == "jbrownlee/Datasets" and
                source["archives"][0]["source_url"] == source["archives"][0]["resolved_url"] == url and
                "not independently established" in source["verification_limits"] and
                source["scientific_source_url"] == "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
                "Pinned mirror identity or explicit official-byte-identity limitation differs")
    else:
        require(source.get("transport", "official_provider") == "official_provider" and
                all(item["source_url"] == "https://www.bgc-jena.mpg.de/wetter/data/" + item["filename"]
                    for item in source["archives"]), "Jena source transport differs from the official archives")
    keys = ("source_id", "archive_path", "archive_sha256", "archive_bytes", "member", "source_url")
    projected = [{key: item[key] for key in keys} for item in source["archives"]]
    require(source["source_manifest_sha256"] == digest(projected) and
            len(projected) == len(spec["archives"]), "Original archive manifest digest differs")
    payloads = {}
    for item, (source_id, filename, member) in zip(source["archives"], spec["archives"]):
        relative = f"inputs/{task}/{filename}"
        require(item["source_id"] == source_id and item["archive_path"] == relative and item["member"] == member and
                checked_path(relative).stat().st_size == item["archive_bytes"], "Original archive inventory differs")
        raw = cached_archive_member(relative, item["archive_sha256"], member)
        with zipfile.ZipFile(checked_path(relative)) as archive:
            compressed_bytes = archive.getinfo(member).compress_size
        require(item["inventory"] == [{"name": member, "bytes": len(raw), "compressed_bytes": compressed_bytes}],
                "Official archive size/inventory receipt differs")
        payloads[source_id] = raw
    return payloads, source


def iso(value):
    return value.isoformat(timespec="seconds")


def source_timestamp(task, cells):
    spec = SOURCE_SPECS[task]
    token = " ".join(cells[index].decode("ascii").strip() for index in spec["timestamp_indices"])
    stamp = None
    for fmt in spec["timestamp_formats"]:
        try:
            stamp = datetime.strptime(token, fmt)
            break
        except ValueError:
            continue
    require(stamp is not None and stamp.second == 0 and stamp.minute % spec["cadence_minutes"] == 0,
            "Source calendar format or cadence differs")
    return stamp


def calendar_queries(task, start, stop, *, include_excluded=False):
    projection = read(ROOT / f"inputs/{task}/calendar_projection40.json")
    rows = {row["timestamp"]: row for row in projection["rows"]}
    require(len(rows) == len(projection["rows"]), "Repeated projected timestamp group")
    target = datetime.fromisoformat(start).replace(hour=23)
    end = datetime.fromisoformat(stop).replace(hour=23)
    queries, excluded = [], []
    while target < end:
        origin = target - timedelta(hours=24)
        required = [iso(target)] + [iso(origin - timedelta(hours=23 - j)) for j in range(24)]
        absent = [stamp for stamp in required if stamp not in rows]
        missing = [stamp for stamp in required if stamp in rows and rows[stamp]["declared_target_missing"]]
        if absent or missing:
            excluded.append({"target_timestamp": iso(target), "absent_required_timestamps": absent,
                             "declared_missing_required_timestamps": missing})
        else:
            queries.append({"target_timestamp": iso(target), "origin_timestamp": iso(origin)})
        target += timedelta(days=1)
    return (queries, excluded) if include_excluded else queries


def independent_calendar_projection(task, payloads):
    """Reconstruct byte offsets/status from calendar fields, never numeric values."""
    spec = SOURCE_SPECS[task]
    lower, upper = datetime(spec["year"], 1, 1) - timedelta(days=2), datetime(spec["year"] + 1, 1, 1)
    headers, previous, first = None, None, None
    groups, member_stats = {}, []
    native_count, unique_count, duplicates = 0, 0, 0
    years, missing_counts = Counter(), Counter()
    markers = {value.encode("ascii") for value in spec["missing_markers"]}
    require(list(payloads) == [item[0] for item in spec["archives"]], "Projected archive order differs")
    for source_id, raw in payloads.items():
        stream = io.BytesIO(raw)
        heading = stream.readline()
        current = [cell.decode(spec["header_encoding"]) for cell in
                   csv_cells(heading.removeprefix(b"\xef\xbb\xbf"), spec["separator"])]
        if headers is None:
            headers = current
        require(current == headers and len(headers) == spec["header_count"] and
                headers[spec["target_index"]] == spec["target_header"] and
                [headers[i] for i in spec["timestamp_indices"]] == spec["timestamp_headers"],
                "Original source header/target/timestamp fields differ")
        count = 0
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            cells = csv_cells(line, spec["separator"])
            require(len(cells) == len(headers), "Original source row field count differs")
            stamp = source_timestamp(task, cells)
            require(previous is None or stamp >= previous, "Source archive calendar order is noncausal")
            native_count += 1
            count += 1
            if previous == stamp:
                duplicates += 1
            else:
                unique_count += 1
                years[str(stamp.year)] += 1
            if first is None:
                first = stamp
            previous = stamp
            if stamp.minute != 0 or not lower <= stamp < upper:
                continue
            name = iso(stamp)
            missing = bool(markers) and cells[spec["target_index"]].strip(b" \t") in markers
            if markers:
                missing_counts["declared_missing" if missing else "not_declared_missing"] += 1
            group = groups.setdefault(name, {"timestamp": name, "records": [], "declared_target_missing": False})
            group["records"].append({"source_id": source_id, "source_line": count + 1,
                                     "byte_start": start, "byte_stop": stream.tell()})
            group["declared_target_missing"] = group["declared_target_missing"] or missing
        member_stats.append({"source_id": source_id, "member_bytes": len(raw),
                             "member_sha256": hashlib.sha256(raw).hexdigest(), "timestamp_rows": count})
    require(first is not None and previous is not None and groups, "Empty original calendar/source window")
    rows = list(groups.values())
    expected = int((previous - first) / timedelta(minutes=spec["cadence_minutes"])) + 1
    return {"schema": "step40-calendar-status-byte-projection-v1", "task": task,
            "headers": headers, "parsed_data_columns": spec["timestamp_indices"],
            "target_field_index": spec["target_index"], "target_values_decoded_or_parsed": 0,
            "other_measurement_values_decoded_or_parsed": 0,
            "target_missing_status_inspected": bool(markers), "missing_status_rule": spec["missing_status_rule"],
            "declared_missing_markers": spec["missing_markers"], "hourly_target_status_counts": dict(missing_counts),
            "rows": rows, "member_stats": member_stats, "timestamp_rows": native_count,
            "unique_native_timestamps": unique_count, "duplicate_record_excess": duplicates,
            "expected_native_timestamps": expected, "missing_native_timestamps": expected - unique_count,
            "first_source_timestamp": iso(first), "last_source_timestamp": iso(previous),
            "source_timestamp_counts_by_year": dict(sorted(years.items())),
            "projected_hourly_timestamps": len(rows), "projected_first_timestamp": rows[0]["timestamp"],
            "projected_last_timestamp": rows[-1]["timestamp"], "projected_window_start": iso(lower),
            "projected_window_stop_exclusive": iso(upper), "native_cadence_minutes": spec["cadence_minutes"],
            "timezone_convention": spec["timezone_convention"], "hourly_snapshot_rule": spec["snapshot_description"],
            "unit": spec["unit"], "duplicate_target_agreement_examined": False,
            "duplicate_target_policy": "Declared missing status at any duplicate excludes timestamp; otherwise authorized target values must agree exactly or STOP_RETAIN_FAILURE",
            "no_value_based_row_filtering": True, "scientific_model_fits": 0}


def audit_calendar(task):
    payloads, source = source_payloads(task)
    record = read(ROOT / f"inputs/{task}/calendar_projection40.json")
    expected = independent_calendar_projection(task, payloads)
    require({key: value for key, value in record.items() if key not in
             ("source_manifest_sha256", "projection_utc", "projection_source_sha256")} == expected and
            record["source_manifest_sha256"] == source["source_manifest_sha256"] and
            record["projection_source_sha256"] == sha(ROOT / "source/prepare_data40.py"),
            "Independent calendar/status byte projection differs")
    metadata = read(ROOT / f"inputs/{task}/calendar_metadata40.json")
    expected_metadata = {key: value for key, value in record.items() if key != "rows"}
    expected_metadata.update({"projection_sha256": sha(ROOT / f"inputs/{task}/calendar_projection40.json"),
                              "catalog_reported_instances": SOURCE_SPECS[task]["reported_instances"]})
    reported = SOURCE_SPECS[task]["reported_instances"]
    expected_metadata["catalog_count_matches_data_rows"] = None if reported is None else expected["timestamp_rows"] == reported
    expected_metadata["source_transport_verification"] = {
        "transport": source.get("transport", "official_provider"),
        "original_member_names_and_headers_verified": True,
        "archive_bytes_retained_verbatim_and_hash_bound": True,
        "measurement_values_used_for_verification": False,
        "direct_official_archive_transport_confirmed": source.get("transport") != "public_repository_mirror",
        "identity_against_independent_official_checksum_verified": False,
        "verification_limits": source.get("verification_limits", "Retrieved directly from official provider; no external independent archive checksum supplied"),
    }
    require(metadata == expected_metadata, "Calendar-only source metadata differs")
    if task == "household_power2009":
        require(expected["timestamp_rows"] == SOURCE_SPECS[task]["reported_instances"] == 2075259,
                "Mirror source row count differs from official metadata")
        require(expected["first_source_timestamp"] == "2006-12-16T17:24:00" and
                expected["last_source_timestamp"] == "2010-11-26T21:02:00",
                "Mirror source time coverage differs from the recorded dataset")
    return {"status": "PASS", "task": task, "source_rows": expected["timestamp_rows"],
            "hourly_timestamps": expected["projected_hourly_timestamps"],
            "duplicate_record_excess": expected["duplicate_record_excess"],
            "missing_status_inspected": expected["target_missing_status_inspected"],
            "measurement_values_decoded": 0}


def authorized_values(task, timestamps):
    gate = read(ROOT / "audit/preexecution_gate40.json")
    require(gate["status"] == "PASS" and gate["protocol_seal_sha256"] == sha(ROOT / "protocol_seal40.json"),
            "Auditor numerical projection requires the frozen preexecution gate")
    requested = sorted(set(timestamps))
    projection = read(ROOT / f"inputs/{task}/calendar_projection40.json")
    lookup = {row["timestamp"]: row for row in projection["rows"]}
    require(requested and all(stamp in lookup and not lookup[stamp]["declared_target_missing"] for stamp in requested),
            "Requested numeric timestamp absent or declared missing")
    payloads, _ = source_payloads(task)
    spec, result = SOURCE_SPECS[task], {}
    for stamp in requested:
        values = []
        for ref in lookup[stamp]["records"]:
            cells = csv_cells(payloads[ref["source_id"]][ref["byte_start"]:ref["byte_stop"]], spec["separator"])
            require(len(cells) == len(projection["headers"]) and iso(source_timestamp(task, cells)) == stamp,
                    "Authorized source byte range belongs to another record")
            token = cells[spec["target_index"]].strip(b" \t")
            require(0 < len(token) <= 256 and re.fullmatch(
                rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token) is not None,
                "Target token violates exact finite-decimal grammar")
            value = F(token.decode("ascii"))
            lower_limit = -F(5463, 20) if spec["signed"] else F(0)
            require(isfinite(float(value)) and value >= lower_limit, "Source numeric domain differs")
            values.append(value)
        require(values and len(set(values)) == 1, "Authorized same-timestamp targets disagree")
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


def audit_head40(head, fit=None, z=None, original_y=None, ids=None, expected=None):
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


def audit_prediction40(record, head, query_z, query_ids, distance=None, distance_checked=None, expected=None):
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
        records.append({"candidate_id": name + "::qp_l0_w05", "map": name, "token": "qp_l0_w05",
                        "recipe": "qp_regularized", "ridge_lambda": "0", "extension_weight": "1/2"})
        records.append({"candidate_id": name + "::affine", "map": name, "token": "affine",
                        "recipe": "affine", "ridge_lambda": None, "extension_weight": None})
    return records


def baseline_configs():
    # The baseline recipes are byte-bound to the completed Step39 protocol.
    records = read(ROOT / "inputs/prior/protocol39.json")["baselines"]["configurations"]
    require(len(records) == 9 and [r["candidate_id"] for r in records] == [
        "baseline::persistence", "baseline::training_mean", "baseline::hgb_leaf7_l2_1",
        "baseline::hgb_leaf15_l2_1", "baseline::hgb_leaf31_l2_1", "baseline::hgb_leaf15_l2_10",
        "baseline::ridge_l0001", "baseline::ridge_l001", "baseline::ridge_l1"],
        "Inherited nine-baseline roster differs")
    return records


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
    seal = verify_seal("protocol_seal40.json", "step40-protocol-source-seal-v1")
    require(seal["protocol_sha256"] == sha(ROOT / "protocol40.json") and
            seal["new_measurement_tokens_decoded"] == 0 and seal["single_primary_pass"] is True,
            "Protocol freeze did not precede selected-cohort numerical measurement access")
    required = {"protocol40.json", "analysis/interface40.txt", "analysis/related_work40.json",
                "source/audit40.py", "source/check_audit40.py", "source/data40.py", "source/check_data40.py",
                "source/prepare_data40.py", "source/protocol40.py", "source/common40.py", "source/run40.py",
                "source/execute40.py", "source/check_runner40.py", "source/decisions40.py", "source/check_decisions40.py",
                "source/learners39.py", "inputs/calendar_roster40.json", "inputs/inherited_sources40.json",
                "inputs/prior_exposure40.json", "tests/learner_fixture40.json", "tests/inherited_learner_fixture40.json"}
    inherited = read(ROOT / "inputs/inherited_sources40.json")
    require(inherited["schema"] == "step40-inherited-byte-identical-copies-v1" and
            inherited["prior_step_modified"] is False and isinstance(inherited["files"], list),
            "Inherited source-copy provenance differs")
    prior_seal_path = ROOT / "inputs/prior/protocol_seal39.json"
    require(sha(prior_seal_path) == "a670a96ab7e7c2fea626521a7db15df3ac79fbb700836a992c2888d4d6910d5d",
            "Retained Step39 protocol freeze differs")
    prior_seal = read(prior_seal_path)
    destinations = []
    for item in inherited["files"]:
        relative = item["destination"]
        destinations.append(relative)
        required.add(relative)
        require(sha(checked_path(relative)) == item["sha256"], "Inherited dependency copy changed")
        original = item["source"].removeprefix("step39_temporal_selection/")
        if original in prior_seal["bound_files"]:
            require(prior_seal["bound_files"][original] == item["sha256"], "Inherited dependency differs from its prior freeze")
    require(len(set(destinations)) == len(destinations), "Inherited dependency destination repeated")
    for task in TASKS:
        required.update(f"inputs/{task}/{filename}" for _, filename, _ in SOURCE_SPECS[task]["archives"])
        source = read(ROOT / f"inputs/{task}/public_source40.json")
        required.update({f"inputs/{task}/public_source40.json", source["metadata_path"],
                         f"inputs/{task}/calendar_projection40.json", f"inputs/{task}/calendar_metadata40.json"})
        required.update(source.get("supplementary_metadata_files", {}))
    required.update("tests/" + filename for filename in (
        "data_preflight40.json", "artificial_audit40.json", "artificial_decisions40.json", "runner_preflight40.json"))
    require(required <= set(seal["bound_files"]), "Protocol source freeze omitted a required dependency")
    require(sha(ROOT / "source/audit38.py") == "4aa7ed6ff39495de9834aaea0f9f4429f5f20e2febbab5425a0f6079ba3d161c" and
            sha(ROOT / "source/audit_math37.py") == a38.GRAPH_MATH_SHA256 and
            sha(ROOT / "source/independent_math35.py") == a38.INDEPENDENT_MATH_SHA256,
            "Inherited independent mathematics changed")
    protocol = read(ROOT / "protocol40.json")
    audit_protocol(protocol)
    return protocol, seal


def audit_protocol(p):
    require(p["schema"] == "step40-head-selection-protocol-v1" and p["step"] == 40 and p["seed"] == 3901 and
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
            heads["configurations"] == candidate_configs() and heads["certified_heads_per_stage"] == 36 and
            heads["qp_fits_per_stage"] == 18 and heads["all_training_anchors"] is True and
            heads["distance_bits"] == heads["optimizer_sqrt_bits"] == 80 and
            heads["clipping_order"] == "AFTER_WEIGHTED_MIXTURE" and heads["affine_ridge_lambda"] == "1/1000000" and
            heads["solver_settings"] == gm.PRIMARY_SETTINGS and heads["same_map_pairing"] is True and
            heads["affine_exact_training_optimizer_claim"] is False and heads["inherited_learner"] == "source/learners39.py" and
            heads["finite_valid_wide_brackets"] == "RETAIN_AND_REPORT_UNRESOLVED",
            "Frozen head-family/midpoint/solver roster differs")
    base = p["baselines"]
    require(base["configurations"] == baseline_configs() and base["count"] == 9 and base["clipped"] is False and
            base["class_membership_certified"] is False and base["same_lag_inputs"] is True,
            "Ordinary baseline tuning/access scope differs")
    ids = [r["candidate_id"] for r in candidate_configs() + baseline_configs()]
    require(len(ids) == 45 == len(set(ids)) and p["evaluation"]["prediction_ids"] == ids,
            "Complete prediction panel differs")
    selection = p["selection"]
    require(selection["folds"] == ["f1", "f2", "f3"] and selection["selectors"] == list(SELECTORS) and
            selection["primary_selector"] == "joint_rolling" and selection["reference_selector"] == "qp_rolling" and
            selection["ordinary_selector"] == "ordinary_rolling" and selection["affine_selector"] == "affine_rolling" and
            selection["selection_control"] == "joint_last_fold" and selection["test_reselection_allowed"] is False and
            selection["assessment_oracle_fits"] is False and selection["tie_breakers"] ==
            ["exact physical MSE upper", "f3 complete map+head payload bytes", "lexical candidate_id"] and
            selection["pooling"] == "Sum of physical-unit SSE divided by total held-out development query count",
            "Frozen nine-selector, exact pooling, or tie policy differs")
    require(selection["union_selector_logic"] ==
            "With the same criterion and tie rule, a joint affine winner equals the affine-only winner and a joint QP winner equals the QP-only winner. The five-percent QP-relative endpoint therefore tests the benefit of admitting an affine winner, not superiority over affine-only selection.",
            "Union-selector interpretation changed")
    mixed = selection["mixed_comparator_decision"]
    require(mixed["scope"] == "Fixed fitted affine coefficient payloads and exact QP anchor optima with midpoint extensions, same fold maps and fixed development labels" and
            mixed["unique_winner"] == "Chosen upper strictly below every other mixed comparator lower" and
            mixed["survivors"] == "Candidates whose lower is at most the smallest comparator upper" and
            mixed["family_separation"] == "Minimum affine risk interval compared to minimum QP risk interval" and
            mixed["affine_training_optimality_claim"] is False and mixed["computational_saving_claim"] is False and
            mixed["all_candidates_fitted"] is True and mixed["future_performance_claim"] is False,
            "Mixed decision certificate scope/criterion differs")
    ev, primary = p["evaluation"], p["evaluation"]["primary_per_task"]
    require(ev["all_candidate_outcomes_retained"] is True and
            primary["gain"] == "qp_lower>0 and20*joint_upper<=19*qp_lower" and
            primary["ordinary"] == "same final payload or joint_upper<=ordinary_lower" and
            primary["affine"] == "same final payload or joint_upper<=affine_lower" and
            primary["same_payload_fields"] == ["candidate_id", "head_id", "head_sha256", "test_predictions_digest"] and
            primary["metric"] == "physical MSE" and primary["all_three_conditions_required"] is True and
            primary["same_joint_affine_is_not_joint_superiority"] is True and
            ev["overall_success"] == "Both task-specific primary criteria and all independent checks pass" and
            ev["months_by_task"] == {task: [f"{YEARS[task]}-{month:02d}" for month in (9, 10, 11)] for task in TASKS} and
            ev["population_significance_test"] is False and ev["future_response_coverage_claim"] is False,
            "Frozen three-condition confirmatory endpoint differs")
    require(ev["monthly_role"] == "DESCRIPTIVE_NOT_INDEPENDENT_REPLICATIONS" and
            ev["same_map_comparisons"]["count_per_task"] == 18 and
            ev["same_map_comparisons"]["operational_and_exact_QP_comparators"] is True and
            ev["same_map_comparisons"]["role"] == "DESCRIPTIVE_DEPENDENT_CONFIGURATIONS_NOT_INDEPENDENT_REPLICATIONS",
            "Same-map attribution/replication scope differs")
    execution = p["execution"]
    require(execution["single_primary_pass"] is True and execution["scientific_reruns_allowed"] is False and
            execution["threads"] == 1 and execution["failure_policy"] == "STOP_RETAIN_FAILURE_NO_RETUNING" and
            execution["audit_failure_policy"] == "STOP_RETAIN_EVIDENCE" and
            execution["phase_budget_seconds"] == {"fit": 1800, "validate": 1800, "select": 120, "test": 3600, "aggregate": 120} and
            execution["native_representation_budget_seconds"] == 300,
            "One-pass fixed-budget execution policy differs")
    data = p["data"]
    require(data["task_order"] == list(TASKS) and data["stages"] == STAGE_DATES and
            data["test_intervals"] == {task: {"start": test_dates(task)[0], "stop_exclusive": test_dates(task)[1]} for task in TASKS} and
            data["horizon_hours"] == data["lag_hours"] == 24 and data["daily_target_hour"] == 23 and
            data["calendar_roster_sha256"] == sha(ROOT / "inputs/calendar_roster40.json") and
            data["targets_clipped"] is False and data["no_online_refit"] is True and
            data["previously_revealed_targets_may_be_later_history"] is True and
            data["other_measurement_predictors"] is False and data["future_calendar_predictors"] is False,
            "Frozen temporal/target/history contract differs")
    for task in TASKS:
        source = read(ROOT / f"inputs/{task}/public_source40.json")
        projection = read(ROOT / f"inputs/{task}/calendar_projection40.json")
        spec, declared = SOURCE_SPECS[task], data["sources"][task]
        manifest_keys = ("source_id", "archive_path", "archive_sha256", "archive_bytes", "member", "source_url")
        archives = [{key: item[key] for key in manifest_keys} for item in source["archives"]]
        require(declared["expected_n"] == EXPECTED_N[task] and declared["uci_id"] == spec["uci_id"] and
                declared["year"] == spec["year"] and declared["source_manifest_sha256"] == source["source_manifest_sha256"] and
                declared["archives"] == archives and declared["member_stats"] == projection["member_stats"] and
                declared["target_field_index"] == spec["target_index"] and declared["target"] == spec["target_header"] and
                declared["source_license"] == "CC BY 4.0" and declared["missing_markers"] == spec["missing_markers"] and
                declared["missing_status_rule"] == spec["missing_status_rule"] and
                declared["numeric_domain"] == spec["numeric_domain"] and declared["unit"] == spec["unit"] and
                declared["timezone_convention"] == spec["timezone_convention"] and
                declared["snapshot_description"] == spec["snapshot_description"] and
                declared["transport"] == source.get("transport", "official_provider") and
                declared["mirror_commit"] == source.get("mirror_commit") and
                declared["metadata_evidence_path"] == source["metadata_path"] and
                declared["metadata_evidence_sha256"] == source.get("metadata_sha256", source.get("metadata_html_sha256")) and
                declared["verification_limits"] == source.get("verification_limits", "Original archive bytes retrieved from official source provider and hash-sealed"),
                "Frozen original archive, numeric domain, or eligibility identity differs")
    return {"status": "PASS", "tasks": 2, "maps_per_stage": 18, "heads_per_stage": 36,
            "baselines_per_stage": 9, "selectors": 9, "assessment_class_oracle": False,
            "fixed_affine_not_exact_affine_optimizer": True, "computational_saving_claim": False}


def frozen_queries(task, stage, partition):
    roster = read(ROOT / "inputs/calendar_roster40.json")
    if partition == "test":
        start, stop = test_dates(task)
        record = roster["tasks"][task]["test"]
        n = EXPECTED_N[task]["test"]
    else:
        start = STAGE_DATES[task][stage][partition + "_start"]
        stop = STAGE_DATES[task][stage][partition + "_stop"]
        record = roster["tasks"][task]["stages"][stage][partition]
        n = EXPECTED_N[task][stage][partition]
    calendar, excluded = calendar_queries(task, start, stop, include_excluded=True)
    queries = [{"id": task + ":" + row["target_timestamp"], "task": task, **row} for row in calendar]
    require(record["queries"] == queries and len(queries) == record["n"] == n and
            record["query_ids_sha256"] == digest([row["id"] for row in queries]) and
            record["calendar_start"] == start and record["calendar_stop_exclusive"] == stop and
            record["calendar_ineligible"] == excluded,
            "Frozen calendar/status cohort or exclusion roster differs")
    return queries


def lag_stamps(query):
    origin = datetime.fromisoformat(query["origin_timestamp"])
    require(datetime.fromisoformat(query["target_timestamp"]) - origin == timedelta(hours=24),
            "Forecast horizon differs")
    return [iso(origin - timedelta(hours=23 - j)) for j in range(24)]


def audit_access(record, task, stage, phase, timestamps, maximum):
    stamps = sorted(set(timestamps))
    projection = read(ROOT / f"inputs/{task}/calendar_projection40.json")
    lookup = {row["timestamp"]: row for row in projection["rows"]}
    require(stamps and all(stamp in lookup and not lookup[stamp]["declared_target_missing"] for stamp in stamps),
            "Access receipt includes absent/status-ineligible source rows")
    duplicate = [{"timestamp": stamp, "source_records": len(lookup[stamp]["records"]), "exact_agreement": True}
                 for stamp in stamps if len(lookup[stamp]["records"]) > 1]
    references = [{"timestamp": stamp, **ref} for stamp in stamps for ref in lookup[stamp]["records"]]
    require(record["task"] == task and record["stage"] == stage and record["phase"] == phase and
            record["requested_timestamps"] == stamps and record["unique_measurement_timestamps"] == len(stamps) and
            record["measurement_token_decodes"] == len(references) and
            record["maximum_allowed_timestamp"] == maximum and record["maximum_decoded_timestamp"] == max(stamps) and
            max(stamps) <= maximum and record["duplicate_agreement"] == duplicate and
            record["source_manifest_sha256"] == projection["source_manifest_sha256"] and
            record["member_sha256"] == {item["source_id"]: item["member_sha256"] for item in projection["member_stats"]} and
            record["source_row_references"] == references and
            record["other_measurement_values_decoded_or_parsed"] == 0 and record["outcome_based_row_filtering"] is False,
            "Source access byte projection, duplicate agreement, or temporal boundary differs")
    return {"unique_measurement_timestamps": len(stamps), "duplicate_timestamps": len(duplicate),
            "original_source_row_references": len(references)}


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
    norm_path = ROOT / f"{base}/normalization40.json"
    normal = read(norm_path)
    if partition == "train":
        training_targets, training_ids = raw_targets, ids
    else:
        training = read(ROOT / f"{base}/cohorts/train.json")
        training_targets, training_ids = list(map(q, training["raw_targets"])), training["ids"]
    low, high = min(training_targets), max(training_targets)
    require(high > low and normal["schema"] == "step40-training-target-normalization-v1" and
            normal["task"] == task and normal["stage"] == stage and normal["fitted_partition"] == "train" and
            q(normal["target_min"]) == low and q(normal["target_max"]) == high and q(normal["span"]) == high - low and
            normal["train_target_ids_sha256"] == digest(training_ids) and
            normal["raw_train_targets_sha256"] == digest(training_targets) and normal["target_values_clipped"] is False and
            normal["future_data_used"] is False, "Target normalization was not fitted on this stage's training targets")
    targets = [(v - low) / (high - low) for v in raw_targets]
    persistence = [(v - low) / (high - low) for v in raw_persistence]
    X = np.asarray([[float(values[s]) for s in row] for row in lags], dtype=np.float64)
    saved_X = np.load(ROOT / record["X_path"], allow_pickle=False)
    require(record["schema"] == "step40-causal-cohort-v1" and record["task"] == task and record["stage"] == stage and
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
    access = read(ROOT / f"{base}/data_access_{partition}40.json")
    stop = datetime.fromisoformat(STAGE_DATES[task][stage][partition + "_stop"]) - timedelta(seconds=1)
    audit_access(access, task, stage, partition, needed, iso(stop))
    require(access["schema"] == "step40-development-measurement-access-v1" and access["status"] == "PASS" and
            access["cohort_path"] == relative and access["cohort_sha256"] == sha(ROOT / relative) and
            access["normalization_sha256"] == sha(norm_path) and
            complete["access_sha256"] == sha(ROOT / f"{base}/data_access_{partition}40.json"),
            "Development access receipt changed")
    return {**record, "X": X, "y": targets, "persistence": persistence,
            "normalization": normal, "path": relative, "sha256": sha(ROOT / relative)}


def load_context(task, stage):
    import numpy as np
    import pickle
    import representations35 as replay
    manifest = read(ROOT / f"output/{task}/{stage}/fit_manifest40.json")
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
    require(manifest["schema"] == "step40-fit-manifest-v1" and manifest["task"] == task and manifest["stage"] == stage and
            [r["candidate_id"] for r in rows] == expected_ids and sorted(manifest["maps"]) == names and
            manifest["training_n"] == len(y) == len(ids) == len(X) and manifest["normalization"] == training["normalization"] and
            manifest["training_cohort"] == {"path": training["path"], "sha256": training["sha256"]} and
            manifest["all_fit_data_at_or_before_held_out_origins"] is True,
            "Stage fit manifest or original training cohort differs")
    for row, cfg in zip(rows, configs):
        require(all(row[k] == v for k, v in cfg.items()), "Fitted configuration roster differs")
    if stage == "final":
        require(manifest["selection_seal_sha256"] == sha(ROOT / f"output/{task}/selection_seal40.json"),
                "Final refit was not bound to the frozen development selection")
    else:
        require(manifest["selection_seal_sha256"] is None, "Development fold used a later selection")
    metadata = read(ROOT / f"output/{task}/{stage}/representation_metadata40.json")
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
                info["feature_id"] == "feature40:" + digest({"task": task, "stage": stage, "map": name,
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
        for lam, token in (("0", "l0"),):
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
                checked = audit_head40(head, expected=expected)
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
                checked = audit_head40(head, fit, z, y, ids, expected)
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
    return {"status": "PASS", "task": task, "stage": stage, "maps": 18, "heads": 36, "ordinary_baselines": 9,
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
    require(record["schema"] in ("step40-prediction-bundle-v1", "step40-sealed-test-predictions-v1") and
            record["task"] == task and record["stage"] == stage and record["query_ids"] == ids and
            set(record["predictions"]) == set(expected_ids) and set(record["optimizer_predictions"]) == set(qp_ids) and
            set(record["maps"]) == set(maps_roster()) and
            record["fit_seal_sha256"] == sha(ROOT / f"output/{task}/{stage}/fit_seal40.json"),
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
            summaries.append(audit_prediction40(pred, head, z, ids, distance, checked,
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
    return {"status": "PASS", "queries": len(ids), "head_predictions": len(ids) * 36,
            "optimizer_prediction_enclosures": len(ids) * 18, "baseline_predictions": len(ids) * 9,
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


def decision_interval(lo, hi):
    """Canonical interval used only by the independently derived decisions."""
    require(lo <= hi, "Decision interval endpoints are reversed")
    return {"lower": str(lo), "upper": str(hi), "width": str(hi - lo)}


def decision_bounds(value):
    require(isinstance(value, dict), "Decision risk must be an interval object")
    for name in ("lower", "upper", "width"):
        if name in value:
            require(isinstance(value[name], (str, int, F)) and not isinstance(value[name], bool),
                    "Decision risks require explicit exact rational inputs")
    low, high = q(value["lower"]), q(value["upper"])
    require(0 <= low <= high, "Decision risks must be nonnegative ordered intervals")
    if "width" in value:
        require(q(value["width"]) == high - low, "Decision risk width differs")
    return low, high


def independent_comparators(rows, names, operational_key, optimizer_key):
    """Independently bind one stored affine and one exact-QP comparator/map."""
    require(isinstance(names, (list, tuple)) and names and len(names) == len(set(names)) and
            all(isinstance(name, str) and name for name in names), "Malformed declared comparison map roster")
    require(isinstance(rows, (list, tuple)) and len(rows) == 2 * len(names),
            "Mixed comparison requires exactly two certified heads per map")
    require(all(isinstance(row, dict) for row in rows), "Malformed comparison candidate")
    ids = [row.get("candidate_id") for row in rows]
    require(all(isinstance(cid, str) and cid for cid in ids) and len(set(ids)) == len(ids),
            "Mixed comparator IDs must be unique and nonempty")
    require(all(row.get("map") in names and row.get("recipe") in ("affine", "qp_regularized") for row in rows),
            "Mixed comparison contains an undeclared map or head family")
    result = []
    for name in names:
        for family, recipe in (("affine", "affine"), ("qp", "qp_regularized")):
            matches = [row for row in rows if row["map"] == name and row["recipe"] == recipe]
            require(len(matches) == 1, "Same-map comparison has missing or duplicated head family")
            row = matches[0]
            op = decision_bounds(row[operational_key])
            if family == "affine":
                require(op[0] == op[1] and row.get("ridge_lambda") is None and
                        row.get("extension_weight") is None and row.get(optimizer_key) is None,
                        "Affine comparison must concern the exact stored predictor, not an affine optimizer")
                bound, source = op, operational_key
                kind = "FIXED_AFFINE_PAYLOAD"
            else:
                require(all(isinstance(row[k], (str, int, F)) and not isinstance(row[k], bool)
                            for k in ("ridge_lambda", "extension_weight")) and
                        q(row["ridge_lambda"]) == 0 and q(row["extension_weight"]) == F(1, 2),
                        "Mixed QP comparator is not the frozen unregularized midpoint recipe")
                bound, source = decision_bounds(row[optimizer_key]), optimizer_key
                require(bound[0] <= op[0] <= op[1] <= bound[1],
                        "QP optimizer expansion does not contain its operational prediction-loss interval")
                kind = "EXACT_QP_ANCHOR_OPTIMUM_MIDPOINT_EXTENSION"
            result.append({"candidate_id": row["candidate_id"], "map": name, "head_family": family,
                           "comparator_kind": kind, "operational_risk": decision_interval(*op),
                           "mixed_comparator_risk": decision_interval(*bound), "risk_source": source})
    return result


def independent_decision_scope(*, comparison=False):
    result = {
        "comparator_scope": "FIXED_AFFINE_AND_EXACT_QP", "assurance": ASSURANCE,
        "affine_comparator": "EXACT_STORED_FITTED_COEFFICIENT_PAYLOAD",
        "qp_comparator": "MIDPOINT_EXTENSION_OF_UNIQUE_EXACT_TRAINING_ANCHOR_OPTIMUM",
        "affine_training_optimizer_certified": False, "native_transform_error_enclosed": False,
        "future_response_coverage_claim": False, "population_risk_claim": False,
        "computational_savings_claim": False, "all_candidates_fitted_at_predeclared_budget": True,
    }
    result["same_fixed_maps_training_problems_and_" + ("comparison" if comparison else "development") + "_labels"] = True
    return result


def independent_pair_rows(comparators, names):
    lookup = {(row["map"], row["head_family"]): row for row in comparators}
    pairs = []
    for name in names:
        affine, qp = lookup[(name, "affine")], lookup[(name, "qp")]
        a, b = decision_bounds(affine["mixed_comparator_risk"]), decision_bounds(qp["mixed_comparator_risk"])
        fixed_a, fixed_b = decision_bounds(affine["operational_risk"]), decision_bounds(qp["operational_risk"])
        lower, upper = b[0] - a[1], b[1] - a[0]
        status = "NO_STRICT_SEPARATION"
        if lower > 0:
            status = "AFFINE_STRICTLY_LOWER_RISK"
        elif upper < 0:
            status = "QP_STRICTLY_LOWER_RISK"
        pairs.append({"map": name, "affine_candidate_id": affine["candidate_id"],
                      "qp_candidate_id": qp["candidate_id"],
                      "difference_definition": "QP_MINUS_AFFINE_PHYSICAL_MSE",
                      "mixed_difference": decision_interval(lower, upper),
                      "operational_difference": decision_interval(fixed_b[0] - fixed_a[1], fixed_b[1] - fixed_a[0]),
                      "status": status})
    return pairs


def independent_decision_certificate(rows, selected_id, names):
    """No producer decision import: exact consequences are reconstructed here."""
    inputs = independent_comparators(rows, names, "pooled_physical_risk", "pooled_optimizer_risk")
    lookup = {row["candidate_id"]: row for row in inputs}
    require(selected_id in lookup, "Operational selection is outside the frozen mixed panel")
    bounds = {cid: decision_bounds(row["mixed_comparator_risk"]) for cid, row in lookup.items()}
    incumbent_id = min(bounds, key=lambda cid: (bounds[cid][1], cid))
    threshold = bounds[incumbent_id][1]
    surviving = [row["candidate_id"] for row in inputs if bounds[row["candidate_id"]][0] <= threshold]
    excluded = [{"candidate_id": row["candidate_id"],
                 "strict_exclusion_margin": str(bounds[row["candidate_id"]][0] - threshold)}
                for row in inputs if bounds[row["candidate_id"]][0] > threshold]
    require(surviving, "Finite nonempty interval collection lost every possible minimizer")
    unique = surviving[0] if len(surviving) == 1 else None
    competitor_id = min((cid for cid in bounds if cid != selected_id), key=lambda cid: (bounds[cid][0], cid))
    margin = bounds[competitor_id][0] - bounds[selected_id][1]
    family_bounds = {}
    for family in ("affine", "qp"):
        cells = [bounds[row["candidate_id"]] for row in inputs if row["head_family"] == family]
        family_bounds[family] = (min(cell[0] for cell in cells), min(cell[1] for cell in cells))
    lower = family_bounds["qp"][0] - family_bounds["affine"][1]
    upper = family_bounds["qp"][1] - family_bounds["affine"][0]
    status = ("AFFINE_FAMILY_STRICTLY_LOWER_RISK" if lower > 0 else
              "QP_FAMILY_STRICTLY_LOWER_RISK" if upper < 0 else "NO_STRICT_SEPARATION")
    return {
        "schema": "step40-mixed-comparator-decision-certificate-v1", **independent_decision_scope(),
        "cohort_role": "development", "source_comparator_rows_digest": digest(inputs), "comparator_rows": inputs,
        "candidate_count": len(inputs), "map_count": len(names), "operational_selected_id": selected_id,
        "operational_selection_changed": False, "test_labels_used": False,
        "candidate_exclusion": {
            "criterion": "EXCLUDE_IF_LOWER_STRICTLY_EXCEEDS_MINIMUM_UPPER",
            "incumbent_upper_candidate_id": incumbent_id, "minimum_upper": str(threshold),
            "survivor_ids": surviving, "survivor_count": len(surviving), "excluded": excluded,
            "excluded_count": len(excluded), "excluded_fraction": str(F(len(excluded), len(inputs))),
            "unique_survivor_id": unique,
            "unique_survivor_agrees_with_operational_selection": None if unique is None else unique == selected_id,
            "excluded_candidates_cannot_minimize_mixed_comparator_risk": True,
            "equality_is_retained": True, "fitting_work_avoided": False,
        },
        "selected_winner": {
            "selected_id": selected_id, "selected_mixed_comparator_risk": lookup[selected_id]["mixed_comparator_risk"],
            "nearest_competitor_id": competitor_id, "competitor_minimum_lower": str(bounds[competitor_id][0]),
            "strict_margin_lower": str(margin),
            "status": "CERTIFIED_UNIQUE_MIXED_COMPARATOR_WINNER" if margin > 0 else "UNRESOLVED",
        },
        "head_family_resolution": {
            "family_minimum_risks": {family: decision_interval(*bound) for family, bound in family_bounds.items()},
            "difference_definition": "MIN_QP_MINUS_MIN_FIXED_AFFINE_PHYSICAL_MSE",
            "difference": decision_interval(lower, upper), "status": status,
            "family_minimum_interval_rule": "[min_c lower_c, min_c upper_c] WITHIN EACH FAMILY",
        },
        "same_map_comparisons": independent_pair_rows(inputs, names),
        "attribution_scope": "PAIRED_HEAD_RECIPES_ON_THE_SAME_STORED_MAP_NOT_PURE_CAPACITY_EFFECT",
        "utility_scope": "POST_FIT_DECISION_RESOLUTION_AT_THE_PREDECLARED_NUMERICAL_BUDGET",
    }


def audit_decision_certificate(rows, selected_id, record, names=None):
    expected = independent_decision_certificate(rows, selected_id, maps_roster() if names is None else names)
    require(record == expected, "Mixed comparator decision certificate is not the exact independent reconstruction")
    return expected


def audit_paired_heads(rows, record, *, names=None, operational_key="pooled_physical_risk",
                       optimizer_key="pooled_optimizer_risk", cohort_role="development"):
    names = maps_roster() if names is None else names
    require(cohort_role in ("development", "test"), "Undeclared paired comparison cohort")
    inputs = independent_comparators(rows, names, operational_key, optimizer_key)
    expected = {"schema": "step40-same-map-head-comparisons-v1", **independent_decision_scope(comparison=True),
                "cohort_role": cohort_role, "selection_performed_by_this_function": False,
                "test_reselection_allowed": False, "map_count": len(names),
                "source_comparator_rows_digest": digest(inputs), "comparator_rows": inputs,
                "pairs": independent_pair_rows(inputs, names),
                "attribution_scope": "PAIRED_HEAD_RECIPES_ON_THE_SAME_STORED_MAP_NOT_PURE_CAPACITY_EFFECT"}
    require(record == expected, "Same-map head comparison is not the exact independent reconstruction")
    return expected


def audit_validation_values(protocol, task, stage, scores, bundle, manifest, cohort):
    require(scores["schema"] == "step40-validation-scores-v1" and scores["task"] == task and scores["stage"] == stage and
            scores["n"] == len(cohort["ids"]) and scores["normalization"] == cohort["normalization"] and
            scores["fit_seal_sha256"] == bundle["fit_seal_sha256"] and
            scores["validation_predictions_sha256"] == sha(ROOT / f"output/{task}/{stage}/validation_predictions40.json") and
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
            "optimizer_losses": 18, "physical_unit_pooling_ready": True}


def selector_pool(rows, label):
    require(label in SELECTORS, "Unknown independent selector")
    if label == "ordinary_rolling":
        return [r for r in rows if r["recipe"] == "baseline"]
    if label == "hgb_rolling":
        return [r for r in rows if r["baseline_kind"] == "hgb"]
    if label == "affine_rolling":
        return [r for r in rows if r["recipe"] == "affine"]
    if label == "qp_rolling":
        return [r for r in rows if r["recipe"] == "qp_regularized"]
    pool = [r for r in rows if r["recipe"] in ("affine", "qp_regularized")]
    if label in ("kahm_joint_rolling", "raw_joint_rolling", "nystrom_joint_rolling"):
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
    """Verify physical pooling, all nine selections and the mixed certificate."""
    require(set(scores) == {"f1", "f2", "f3"} and record["schema"] == "step40-development-selection-v1" and
            record["selection_uses_development_only"] is True and record["test_targets_accessed"] == 0,
            "Selection source/scope differs")
    stages = ("f1", "f2", "f3")
    ids = [row["candidate_id"] for row in scores["f1"]["candidate_rows"]]
    require(all([row["candidate_id"] for row in scores[s]["candidate_rows"]] == ids for s in stages) and
            [row["candidate_id"] for row in record["candidate_rows"]] == ids and len(ids) == 45,
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
    family_selector = "affine_rolling" if winner["recipe"] == "affine" else "qp_rolling"
    require(expected_selections[family_selector]["candidate_id"] == winner_id,
            "Union winner differs from its own family's identical-criterion winner")
    require(isfinite(record["decision_certificate_seconds"]) and record["decision_certificate_seconds"] >= 0,
            "Mixed decision-certificate construction time is invalid")
    checked = audit_decision_certificate(pool, winner_id, record["mixed_decision_certificate"])
    return {"status": "PASS", "candidate_configurations": len(ids), "selection_count": len(SELECTORS),
            "development_n": record["development_n"], "selected_ids": {k: v["candidate_id"] for k, v in expected_selections.items()},
            "mixed_decision_certificate": checked,
            "joint_equals_its_head_family_selector": family_selector,
            "joint_superiority_over_affine_inferred_from_identity": False}


def audit_practical_gate(new, reference, ordinary, affine, record):
    """Verify the three predeclared finite-cohort comparisons on bound rows."""
    def identical(left, right):
        fields = ("candidate_id", "head_id", "head_sha256", "test_predictions_digest")
        return all(isinstance(left.get(field), str) and bool(left[field]) and
                   left[field] == right.get(field) for field in fields)
    def compare(left, right):
        if identical(left, right):
            return True, "SAME_FINAL_CANDIDATE_PAYLOAD_EXACT_EQUALITY"
        flag = q(left["physical_test_risk"]["upper"]) <= q(right["physical_test_risk"]["lower"])
        return flag, "INTERVAL_NONINFERIORITY" if flag else "NOT_ESTABLISHED_BY_INTERVALS"
    positive = q(reference["physical_test_risk"]["lower"]) > 0
    improved = positive and 20 * q(new["physical_test_risk"]["upper"]) <= 19 * q(reference["physical_test_risk"]["lower"])
    ordinary_pass, ordinary_reason = compare(new, ordinary)
    affine_pass, affine_reason = compare(new, affine)
    expected = {"reference_positive_risk": positive, "at_least_five_percent_mse_improvement": improved,
                "no_worse_than_selected_ordinary": ordinary_pass, "no_worse_than_selected_affine": affine_pass,
                "ordinary_comparison_reason": ordinary_reason, "affine_comparison_reason": affine_reason,
                "joint_same_as_affine": identical(new, affine), "joint_superiority_over_affine_from_identity": False,
                "numerical_criterion_pass": improved and ordinary_pass and affine_pass,
                "relative_gain_status": "DEFINED" if positive else "UNDEFINED_RELATIVE_GAIN",
                "population_noninferiority_claim": False}
    require(record == expected, "Frozen five-percent/ordinary/affine comparison arithmetic or identity differs")
    return expected


def audit_phase(protocol, phase, task=None, stage=None):
    dest = ROOT / "output" / (task or "global") / (stage or phase)
    start_path, end_path = dest / (phase + "_started40.json"), dest / (phase + "_completed40.json")
    begin, end = read(start_path), read(end_path)
    i, e1 = event_once("phase_started", phase=phase, task=task, stage=stage)
    j, e2 = event_once("phase_completed", phase=phase, task=task, stage=stage)
    require(i < j and begin["schema"] == "step40-phase-start-v1" and end["schema"] == "step40-phase-completion-v1" and
            begin["phase"] == end["phase"] == phase and begin["task"] == end["task"] == task and
            begin["stage"] == end["stage"] == stage and end["status"] == "COMPLETE" and
            begin["protocol_seal_sha256"] == sha(ROOT / "protocol_seal40.json") and
            e1["marker_path"] == str(start_path.relative_to(ROOT)) and e1["marker_sha256"] == sha(start_path) and
            e2["receipt_sha256"] == sha(end_path) and isfinite(end["wall_seconds"]) and
            0 <= end["wall_seconds"] <= protocol["execution"]["phase_budget_seconds"][phase],
            "Scientific phase identity, one-pass order or fixed budget differs")
    require(not (dest / (phase + "_failure40.json")).exists(), "A stopped scientific phase was promoted")
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
    names = ("data_preflight40.json", "artificial_audit40.json", "artificial_decisions40.json", "runner_preflight40.json")
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
        if "scientific_data_values_accessed" in record:
            require(record["scientific_data_values_accessed"] == 0, "Decision fixture accessed scientific measurements")
        if "official_measurement_accesses" in record:
            require(record["official_measurement_accesses"] == 0, "Runner fixture accessed official measurements")
        for relative, wanted in record["source_sha256"].items():
            require(sha(checked_path(relative)) == wanted, "Artificial check did not bind current source: " + relative)
        if "fixture_path" in record and "fixture_sha256" in record:
            require(sha(checked_path(record["fixture_path"])) == record["fixture_sha256"], "Artificial witness fixture changed")
        records[name] = {"sha256": sha(path), "status": "PASS", "checks": record.get("check_count", len(record.get("checks", {})))}
    artificial = read(ROOT / "tests/artificial_audit40.json")
    require(artificial["fixture_sha256"] == sha(ROOT / "tests/learner_fixture40.json") and
            artificial["fixture_provenance_sha256"] == sha(ROOT / "tests/inherited_learner_fixture40.json") and
            artificial["decision_fixture_sha256"] == sha(ROOT / "tests/artificial_decisions40.json"),
            "Independent artificial audit belongs to another producer fixture")
    provenance = read(ROOT / "tests/inherited_learner_fixture40.json")
    prior_seal = read(ROOT / "inputs/prior/protocol_seal39.json")
    require(provenance["schema"] == "step40-inherited-artificial-learner-fixture-v1" and
            provenance["source"] == "step39_temporal_selection/tests/learner_fixture39.json" and
            provenance["destination"] == "tests/learner_fixture40.json" and provenance["artificial_only"] is True and
            provenance["scientific_data_accessed"] is False and provenance["prior_step_modified"] is False and
            provenance["byte_identical"] is True and provenance["new_scientific_fits"] == 0 and
            provenance["sha256"] == artificial["fixture_sha256"] == prior_seal["bound_files"]["tests/learner_fixture39.json"],
            "Inherited synthetic witness fixture is not byte-identical to its prior frozen version")
    runner = read(ROOT / "tests/runner_preflight40.json")
    require(runner["independent_audit_source_sha256"] == sha(Path(__file__)) and
            runner["baseline_recipe_source_sha256"] == sha(ROOT / "inputs/prior/protocol39.json") and
            runner["all_nine_selections_independently_checked"] is True and
            runner["all_mixed_decision_certificates_independently_checked"] is True and
            runner["all_same_map_test_contrasts_independently_checked"] is True,
            "Artificial integration did not use the current independent checker and decision definitions")
    fixture_root = checked_path(runner["fixture_root"])
    manifest_path = fixture_root / "fixture_manifest40.json"
    require(sha(manifest_path) == runner["fixture_manifest_sha256"], "Artificial integration fixture manifest changed")
    manifest = read(manifest_path)
    require(manifest["scientific_data_accessed"] is False and len(manifest["files"]) == runner["fixture_files"],
            "Artificial integration fixture scope/count differs")
    for item in manifest["files"]:
        path = fixture_root / item["path"]
        require(path.is_relative_to(fixture_root) and path.is_file() and
                path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "Artificial integration evidence changed")
    return records


def preflight():
    started = time.perf_counter()
    protocol, seal = verify_protocol_seal()
    events = audit_events()
    require(len(events) == 1 and events[0]["event"] == "protocol_frozen" and
            events[0]["protocol_seal_sha256"] == sha(ROOT / "protocol_seal40.json"),
            "New scientific measurements or repeated audit preceded the protocol preflight")
    artificial = audit_artificial_receipts()
    calendars = {task: audit_calendar(task) for task in TASKS}
    roster = read(ROOT / "inputs/calendar_roster40.json")
    require(roster["schema"] == "step40-frozen-calendar-roster-v1" and
            roster["measurement_values_decoded_or_parsed"] == 0 and roster["query_horizon_hours"] == 24 and
            roster["query_lag_hours"] == 24 and roster["daily_target_hour"] == 23,
            "Metadata roster exceeded approved scope")
    for task in TASKS:
        for stage in STAGES:
            frozen_queries(task, stage, "train")
            if stage != "final":
                frozen_queries(task, stage, "validation")
        frozen_queries(task, "final", "test")
        require(roster["tasks"][task]["projection_sha256"] == sha(ROOT / f"inputs/{task}/calendar_projection40.json"),
                "Frozen roster source projection changed")
    exposure = read(ROOT / "inputs/prior_exposure40.json")
    require(all(exposure["matches"][task] == [] for task in TASKS) and
            exposure["private_or_reserved_arrays_read"] is False and exposure["file_count"] > 0 and
            exposure["selected_cohort_measurements_decoded_before_freeze"] == 0,
            "Scoped fresh-source exposure record differs")
    prior = protocol["prior_evidence"]
    require(prior["step39_result_sha256"] == sha(ROOT / "inputs/prior/output__result39.json") and
            prior["step39_audit_sha256"] == sha(ROOT / "inputs/prior/audit__postrun_audit39.json") and
            prior["inherited_sources_sha256"] == sha(ROOT / "inputs/inherited_sources40.json") and
            prior["prior_steps_immutable"] is True and prior["reserved_patients_accessed"] == 0,
            "Prior evidence/scientific scope changed")
    result = {"schema": "step40-independent-preexecution-gate-v1", "protocol_sha256": sha(ROOT / "protocol40.json"),
              "protocol_seal_sha256": sha(ROOT / "protocol_seal40.json"), "bound_files": len(seal["bound_files"]),
              "protocol": audit_protocol(protocol), "calendars": calendars, "artificial_checks": artificial,
              "new_measurement_values_decoded": 0, "no_new_scientific_fits": True,
              "wall_seconds": time.perf_counter() - started}
    return publish_gate("audit/preexecution_gate40.json", "independent_preexecution_PASS", result)


def stage_transition(task, stage):
    _, initial_index = verify_gate("audit/preexecution_gate40.json", "independent_preexecution_PASS",
        protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
    if stage in ("f2", "f3"):
        previous = "f1" if stage == "f2" else "f2"
        _, initial_index = verify_gate(f"audit/{task}/{previous}/validation_gate40.json", "independent_validation_PASS",
            task=task, stage=previous, protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"),
            completion_seal_sha256=sha(ROOT / f"output/{task}/{previous}/completion_seal40.json"))
    elif stage == "final":
        _, initial_index = verify_gate(f"audit/{task}/selection_gate40.json", "independent_selection_PASS",
            task=task, protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"),
            selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal40.json"))
    return initial_index


def fit_gate(task, stage):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    require(task in TASKS and stage in STAGES, "Unknown fit audit task/stage")
    transition_index = stage_transition(task, stage)
    relative = f"output/{task}/{stage}/fit_seal40.json"
    seal = verify_seal(relative, "step40-fit-seal-v1", task=task, stage=stage,
                       protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
    manifest_path = ROOT / f"output/{task}/{stage}/fit_manifest40.json"
    require(seal["manifest_sha256"] == sha(manifest_path), "Fit seal has another training manifest")
    required = {f"output/{task}/{stage}/cohorts/train.json", f"output/{task}/{stage}/cohorts/train_X.npy",
                f"output/{task}/{stage}/normalization40.json", str(manifest_path.relative_to(ROOT)),
                f"output/{task}/{stage}/representation_metadata40.json"}
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
    seal_index, event = event_once("evidence_sealed", schema="step40-fit-seal-v1", task=task, stage=stage)
    require(transition_index < begin < access_begin < access_end < end < seal_index and
            event["path"] == relative and event["sha256"] == sha(ROOT / relative), "Fitting access/seal phase order differs")
    training = audit_cohort(task, stage, "train")
    future = frozen_queries(task, stage, "validation") if stage != "final" else frozen_queries(task, stage, "test")
    require(max(training["target_timestamps"]) <= min(r["origin_timestamp"] for r in future),
            "Training target is later than the first evaluation forecast origin")
    summary = audit_stage_models(protocol, task, stage, training)
    require(phase["training_n"] == len(training["ids"]) and phase["certified_heads"] == 36 and phase["baselines"] == 9,
            "Fit phase count summary differs")
    result = {"schema": "step40-independent-fit-gate-v1", "task": task, "stage": stage,
              "protocol_seal_sha256": sha(ROOT / "protocol_seal40.json"), "fit_seal_sha256": sha(ROOT / relative),
              "training_cohort_sha256": training["sha256"], "training_n": len(training["ids"]),
              "models": summary, "all_fit_data_at_or_before_held_out_origins": True,
              "new_scientific_model_fits_by_auditor": 0, "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/{stage}/fit_gate40.json", "independent_fit_PASS", result)


def validation_gate(task, stage):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    require(task in TASKS and stage in STAGES[:3], "Unknown development audit task/stage")
    _, fit_index = verify_gate(f"audit/{task}/{stage}/fit_gate40.json", "independent_fit_PASS", task=task, stage=stage,
        protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"),
        fit_seal_sha256=sha(ROOT / f"output/{task}/{stage}/fit_seal40.json"))
    verify_seal(f"output/{task}/{stage}/fit_seal40.json", "step40-fit-seal-v1", task=task, stage=stage,
                protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
    relative = f"output/{task}/{stage}/completion_seal40.json"
    seal = verify_seal(relative, "step40-fold-completion-seal-v1", task=task, stage=stage,
                       protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
    base = f"output/{task}/{stage}"
    require(seal["fit_seal_sha256"] == sha(ROOT / f"{base}/fit_seal40.json") and
            seal["validation_scores_sha256"] == sha(ROOT / f"{base}/validation_scores40.json") and
            {f"{base}/validation_predictions40.json", f"{base}/validation_scores40.json",
             f"{base}/cohorts/validation.json", f"{base}/cohorts/validation_X.npy"} <= set(seal["bound_files"]),
            "Validation completion seal omitted an evaluation dependency")
    begin, end, phase = audit_phase(protocol, "validate", task, stage)
    access_begin, _ = event_once("validation_measurement_access_started", task=task, stage=stage)
    access_end, _ = event_once("validation_measurement_access_completed", task=task, stage=stage)
    seal_index, event = event_once("evidence_sealed", schema="step40-fold-completion-seal-v1", task=task, stage=stage)
    require(fit_index < begin < access_begin < access_end < end < seal_index and event["path"] == relative and
            event["sha256"] == sha(ROOT / relative), "Development access/scoring seal order differs")
    cohort = audit_cohort(task, stage, "validation")
    context = load_context(task, stage)
    bundle = read(ROOT / f"{base}/validation_predictions40.json")
    require(bundle["role"] == "validation", "Development prediction role differs")
    prediction_check = audit_prediction_bundle(protocol, task, stage, bundle, context,
                                                cohort["X"], cohort["ids"], cohort["persistence"])
    scores = read(ROOT / f"{base}/validation_scores40.json")
    scoring = audit_validation_values(protocol, task, stage, scores, bundle, context["manifest"], cohort)
    require(phase["validation_n"] == len(cohort["ids"]), "Development phase count summary differs")
    result = {"schema": "step40-independent-validation-gate-v1", "task": task, "stage": stage,
              "protocol_seal_sha256": sha(ROOT / "protocol_seal40.json"),
              "fit_seal_sha256": sha(ROOT / f"{base}/fit_seal40.json"), "completion_seal_sha256": sha(ROOT / relative),
              "validation_scores_sha256": sha(ROOT / f"{base}/validation_scores40.json"),
              "predictions": prediction_check, "scoring": scoring, "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/{stage}/validation_gate40.json", "independent_validation_PASS", result)


def selection_gate(task):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    scores, indices = {}, []
    for stage in STAGES[:3]:
        _, index = verify_gate(f"audit/{task}/{stage}/validation_gate40.json", "independent_validation_PASS", task=task, stage=stage,
            protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"),
            completion_seal_sha256=sha(ROOT / f"output/{task}/{stage}/completion_seal40.json"))
        verify_seal(f"output/{task}/{stage}/completion_seal40.json", "step40-fold-completion-seal-v1", task=task, stage=stage,
                    protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
        scores[stage] = read(ROOT / f"output/{task}/{stage}/validation_scores40.json")
        indices.append(index)
    relative = f"output/{task}/selection_seal40.json"
    seal = verify_seal(relative, "step40-selection-seal-v1", task=task,
                       protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
    record = read(ROOT / f"output/{task}/selection40.json")
    require(record["task"] == task and seal["selection_sha256"] == sha(ROOT / f"output/{task}/selection40.json") and
            seal["selected_ids"] == {key: value["candidate_id"] for key, value in record["selections"].items()},
            "Development selection seal differs")
    require(record["fold_score_sha256"] == {s: sha(ROOT / f"output/{task}/{s}/validation_scores40.json") for s in STAGES[:3]},
            "Development pooling used another set of fold scores")
    begin, end, phase = audit_phase(protocol, "select", task)
    seal_index, event = event_once("evidence_sealed", schema="step40-selection-seal-v1", task=task, stage=None)
    require(max(indices) < begin < end < seal_index and event["path"] == relative and event["sha256"] == sha(ROOT / relative),
            "Selection began before all independent development checks or was not frozen")
    values = audit_selection_values(scores, record)
    require(phase["selected_ids"] == values["selected_ids"], "Selection phase completion names another choice")
    result = {"schema": "step40-independent-selection-gate-v1", "task": task,
              "protocol_seal_sha256": sha(ROOT / "protocol_seal40.json"), "selection_seal_sha256": sha(ROOT / relative),
              "selection_sha256": sha(ROOT / f"output/{task}/selection40.json"), "selection": values,
              "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/selection_gate40.json", "independent_selection_PASS", result)


def pretest(task):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    _, selected_index = verify_gate(f"audit/{task}/selection_gate40.json", "independent_selection_PASS", task=task,
        protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"),
        selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal40.json"))
    fit, fit_index = verify_gate(f"audit/{task}/final/fit_gate40.json", "independent_fit_PASS", task=task, stage="final",
        protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"),
        fit_seal_sha256=sha(ROOT / f"output/{task}/final/fit_seal40.json"))
    verify_seal(f"output/{task}/final/fit_seal40.json", "step40-fit-seal-v1", task=task, stage="final",
                protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
    selection_seal = verify_seal(f"output/{task}/selection_seal40.json", "step40-selection-seal-v1", task=task,
                protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
    manifest = read(ROOT / f"output/{task}/final/fit_manifest40.json")
    require(selected_index < fit_index and manifest["selection_seal_sha256"] == sha(ROOT / f"output/{task}/selection_seal40.json") and
            set(selection_seal["selected_ids"]) == set(SELECTORS) and
            set(selection_seal["selected_ids"].values()) <= set(protocol["evaluation"]["prediction_ids"]),
            "Final fitted concrete heads do not implement the frozen configurations")
    forbidden = [e for e in audit_events() if e.get("task") == task and e["event"].startswith("test_")]
    require(not forbidden, "Test measurement/prediction activity preceded independent pretest PASS")
    result = {"schema": "step40-independent-pretest-gate-v1", "task": task,
              "protocol_seal_sha256": sha(ROOT / "protocol_seal40.json"),
              "fit_seal_sha256": sha(ROOT / f"output/{task}/final/fit_seal40.json"),
              "selection_seal_sha256": sha(ROOT / f"output/{task}/selection_seal40.json"),
              "selected_configuration_ids": selection_seal["selected_ids"], "training_n": fit["training_n"],
              "test_n": len(frozen_queries(task, "final", "test")), "all45_predictions_required_before_each_target": True,
              "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/pretest_gate40.json", "independent_pretest_PASS", result)


def audit_test_forecasts(protocol, task, context):
    import numpy as np
    queries = frozen_queries(task, "final", "test")
    _, pretest_index = verify_gate(f"audit/{task}/pretest_gate40.json", "independent_pretest_PASS", task=task,
        protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"),
        fit_seal_sha256=sha(ROOT / f"output/{task}/final/fit_seal40.json"),
        selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal40.json"))
    cohort_path = ROOT / f"output/{task}/test/cohorts/test.json"
    cohort = read(cohort_path)
    require(cohort["schema"] == "step40-completed-test-cohort-v1" and cohort["task"] == task and
            cohort["n"] == len(queries) and cohort["ids"] == [r["id"] for r in queries] and
            cohort["target_timestamps"] == [r["target_timestamp"] for r in queries] and
            cohort["origin_timestamps"] == [r["origin_timestamp"] for r in queries] and
            cohort["sequential_prediction_then_label"] is True and len(cohort["queries"]) == len(queries),
            "Completed test cohort changed its frozen calendar roster")
    state = read(ROOT / f"output/{task}/test/access_state40.json")
    model_seal = sha(ROOT / f"output/{task}/final/fit_seal40.json")
    require(state["schema"] == "step40-chronological-test-access-state-v1" and state["task"] == task and
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
    normal_hash = sha(ROOT / f"output/{task}/final/normalization40.json")
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
        require(history["schema"] == "step40-causal-test-history-v1" and history["task"] == task and
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
        require(seal["schema"] == "step40-test-prediction-byte-seal-v1" and seal["task"] == task and
                seal["query_id"] == qid and seal["query_index"] == i and seal["model_seal_sha256"] == model_seal and
                seal["prediction_path"] == prediction_rel and seal["prediction_sha256"] == sha(ROOT / prediction_rel) and
                seal["history_receipt_path"] == history_rel and seal["history_receipt_sha256"] == sha(ROOT / history_rel) and
                seal["prediction_ids"] == ids and seal["prediction_count"] == 45 and seal["current_target_value_decoded"] is False,
                "Current-query complete prediction byte seal differs")
        require(pw["prediction_path"] == prediction_rel and pw["prediction_sha256"] == sha(ROOT / prediction_rel) and
                pw["prediction_count"] == 45 and ps["query_index"] == i and ps["prediction_seal_path"] == seal_rel and
                ps["prediction_seal_sha256"] == sha(ROOT / seal_rel) and ps["prediction_path"] == prediction_rel and
                ps["prediction_sha256"] == sha(ROOT / prediction_rel) and ps["prediction_count"] == 45 and
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
        require(target["schema"] == "step40-revealed-test-target-v1" and target["task"] == task and target["query_id"] == qid and
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
        query_summaries.append({"query_id": qid, "arithmetic_head_predictions": 36,
                                "optimizer_prediction_enclosures": 18, "baseline_predictions": 9,
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
                        "head_predictions": len(queries) * 36, "optimizer_prediction_enclosures": len(queries) * 18,
                        "baseline_predictions": len(queries) * 9, "query_anchor_distances": distance_count,
                        "histories_reusing_earlier_revealed_targets": overlap_queries, "queries": query_summaries}}


def audit_test_result(protocol, task, context, test):
    base = ROOT / f"output/{task}/test"
    result = read(base / "result40.json")
    primary = read(base / "primary_result40.json")
    selection = read(ROOT / f"output/{task}/selection40.json")
    manifest = context["manifest"]
    require(result["schema"] == "step40-completed-task-result-v1" and result["status"] == "COMPLETE_PENDING_INDEPENDENT_AUDIT" and
            result["task"] == task and result["protocol_sha256"] == sha(ROOT / "protocol40.json") and
            result["protocol_seal_sha256"] == sha(ROOT / "protocol_seal40.json") and result["primary"] == primary and
            result["primary_result_sha256"] == sha(base / "primary_result40.json") and result["new_scientific_passes"] == 1 and
            result["training_n"] == manifest["training_n"] and result["development_n"] == selection["development_n"],
            "Task result primary/protocol ancestry differs")
    require(primary["schema"] == "step40-task-primary-result-v1" and primary["task"] == task and
            primary["test_n"] == len(test["targets"]) and primary["normalization"] == manifest["normalization"] and
            primary["query_timing"] == test["query_timing"] and isfinite(primary["forecast_seconds"]) and primary["forecast_seconds"] >= 0 and
            primary["model_seal_sha256"] == sha(ROOT / f"output/{task}/final/fit_seal40.json") and
            primary["selection_seal_sha256"] == sha(ROOT / f"output/{task}/selection_seal40.json") and
            primary["pretest_gate_sha256"] == sha(ROOT / f"audit/{task}/pretest_gate40.json") and
            primary["development_decision_certificate"] == selection["mixed_decision_certificate"] and
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
        require(set(row["monthly"]) == set(protocol["evaluation"]["months_by_task"][task]), "Descriptive month roster differs")
        for month in protocol["evaluation"]["months_by_task"][task]:
            indices = [i for i, query in enumerate(test["queries"]) if query["target_timestamp"].startswith(month)]
            value = row["monthly"][month]
            require(value["n"] == len(indices) and len(indices) > 0, "Descriptive month count differs")
            a38.audit_loss([test["predictions"][cid][i] for i in indices], [test["targets"][i] for i in indices], value["risk"])
            require(value["physical_risk"] == scaled_box(value["risk"], scale), "Descriptive month physical risk differs")
    lookup = {row["candidate_id"]: row for row in rows}
    selected = {label: lookup[value["candidate_id"]] for label, value in selection["selections"].items()}
    require(primary["selected"] == selected and set(selected) == set(SELECTORS),
            "Frozen development configuration was changed after test")
    gate = audit_practical_gate(selected["joint_rolling"], selected["qp_rolling"],
                                selected["ordinary_rolling"], selected["affine_rolling"], primary["primary_gate"])
    pairs = audit_paired_heads([row for row in rows if row["recipe"] != "baseline"],
        primary["test_same_map_comparisons"], operational_key="physical_test_risk",
        optimizer_key="physical_optimizer_risk", cohort_role="test")
    require(isfinite(primary["test_pair_certificate_seconds"]) and primary["test_pair_certificate_seconds"] >= 0,
            "Same-map decision certificate construction time is invalid")
    return {"status": "PASS", "candidate_losses": 45, "optimizer_loss_enclosures": 18, "monthly_losses": 135,
            "selected_configuration_ids": {key: row["candidate_id"] for key, row in selected.items()},
            "primary_gate": gate, "development_decision_certificate": primary["development_decision_certificate"],
            "test_same_map_comparisons": pairs, "original_test_labels_not_clipped": True, "assessment_oracle_fits": 0}


def postrun(task):
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    stage_checks = {}
    for stage in STAGES:
        fit_relative = f"output/{task}/{stage}/fit_seal40.json"
        verify_seal(fit_relative, "step40-fit-seal-v1", task=task, stage=stage,
                    protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
        fit, _ = verify_gate(f"audit/{task}/{stage}/fit_gate40.json", "independent_fit_PASS", task=task, stage=stage,
            protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"), fit_seal_sha256=sha(ROOT / fit_relative))
        stage_checks[stage] = {"fit_gate_sha256": sha(ROOT / f"audit/{task}/{stage}/fit_gate40.json"),
                               "training_n": fit["training_n"]}
        if stage != "final":
            complete_relative = f"output/{task}/{stage}/completion_seal40.json"
            verify_seal(complete_relative, "step40-fold-completion-seal-v1", task=task, stage=stage,
                        protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
            verify_gate(f"audit/{task}/{stage}/validation_gate40.json", "independent_validation_PASS", task=task, stage=stage,
                protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"), completion_seal_sha256=sha(ROOT / complete_relative))
            stage_checks[stage]["validation_gate_sha256"] = sha(ROOT / f"audit/{task}/{stage}/validation_gate40.json")
    verify_seal(f"output/{task}/selection_seal40.json", "step40-selection-seal-v1", task=task,
                protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"))
    verify_gate(f"audit/{task}/selection_gate40.json", "independent_selection_PASS", task=task,
        protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"), selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal40.json"))
    _, pretest_index = verify_gate(f"audit/{task}/pretest_gate40.json", "independent_pretest_PASS", task=task,
        protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"), fit_seal_sha256=sha(ROOT / f"output/{task}/final/fit_seal40.json"),
        selection_seal_sha256=sha(ROOT / f"output/{task}/selection_seal40.json"))
    failed = [e for e in audit_events() if e.get("task") == task and
              (e["event"].endswith("_failed") or e["event"] == "phase_stopped_error")]
    require(not failed, "A stopped scientific phase or access failure was followed by a result")
    context = load_context(task, "final")
    test = audit_test_forecasts(protocol, task, context)
    score = audit_test_result(protocol, task, context, test)
    begin, end, phase = audit_phase(protocol, "test", task)
    first_history, _ = event_once("test_history_access_started", task=task, query_id=test["queries"][0]["id"])
    last_reveal, _ = event_once("test_target_revealed", task=task, query_id=test["queries"][-1]["id"])
    result_path = ROOT / f"output/{task}/test/result40.json"
    require(pretest_index < begin < first_history < last_reveal < end and phase["test_n"] == len(test["targets"]) and
            phase["primary_gate"] == score["primary_gate"] and phase["result_sha256"] == sha(result_path),
            "Final test phase occurred outside independent gate/access order")
    result = {"schema": "step40-independent-task-postrun-audit-v1", "task": task,
              "protocol_sha256": sha(ROOT / "protocol40.json"), "protocol_seal_sha256": sha(ROOT / "protocol_seal40.json"),
              "result_sha256": sha(result_path), "primary_result_sha256": sha(ROOT / f"output/{task}/test/primary_result40.json"),
              "selection_seal_sha256": sha(ROOT / f"output/{task}/selection_seal40.json"),
              "fit_seal_sha256": sha(ROOT / f"output/{task}/final/fit_seal40.json"),
              "stage_checks": stage_checks, "test": test["summary"], "scoring": score,
              "practical_transfer_success": score["primary_gate"]["numerical_criterion_pass"],
              "single_primary_pass": True, "all_candidate_outcomes_retained": True,
              "all_scientific_sources_and_prior_gate_inputs_unchanged": True,
              "new_scientific_model_fits_by_auditor": 0, "population_risk_claim": False, "future_response_coverage_claim": False,
              "wall_seconds": time.perf_counter() - started}
    return publish_gate(f"audit/{task}/postrun_audit40.json", "independent_postrun_PASS", result)


def audit_execution_plan(protocol):
    record = read(ROOT / "output/primary_execution40.json")
    commands = []
    for task in TASKS:
        for stage in STAGES[:3]:
            for source, phase in (("run40.py", "fit"), ("audit40.py", "fit"),
                                  ("run40.py", "validate"), ("audit40.py", "validate")):
                commands.append([source, phase, "--task", task, "--stage", stage])
        commands.extend([["run40.py", "select", "--task", task], ["audit40.py", "selection", "--task", task],
                         ["run40.py", "fit", "--task", task, "--stage", "final"],
                         ["audit40.py", "fit", "--task", task, "--stage", "final"],
                         ["audit40.py", "pretest", "--task", task], ["run40.py", "test", "--task", task],
                         ["audit40.py", "postrun", "--task", task]])
    commands.extend([["run40.py", "aggregate"], ["audit40.py", "aggregate"]])
    require(record["schema"] == "step40-one-pass-execution-v1" and record["commands"] == commands and len(commands) == 40 and
            record["protocol_seal_sha256"] == sha(ROOT / "protocol_seal40.json") and
            record["source_sha256"] == sha(ROOT / "source/execute40.py") and
            not (ROOT / "output/execution_stopped40.json").exists(), "Single-pass execution plan was changed or stopped")
    return {"status": "PASS", "commands": len(commands), "no_scientific_reruns": True}


def aggregate():
    started = time.perf_counter()
    protocol, _ = verify_protocol_seal()
    result_path = ROOT / "output/result40.json"
    result = read(result_path)
    require(result["schema"] == "step40-two-domain-result-v1" and result["step"] == 40 and result["status"] == "COMPLETE" and
            result["protocol_sha256"] == sha(ROOT / "protocol40.json") and
            result["protocol_seal_sha256"] == sha(ROOT / "protocol_seal40.json") and set(result["tasks"]) == set(TASKS) and
            result["domain_count"] == 2 and result["all_task_independent_audits_pass"] is True and
            result["test_reselection"] is False and result["assessment_oracle_fits"] == 0 and
            result["population_risk_claim"] is False and result["future_response_coverage_claim"] is False and
            result["reserved_patients_accessed"] == 0, "Combined result scientific scope or task roster differs")
    tasks, pass_all, indices = {}, True, []
    for task in TASKS:
        task_result_path = ROOT / f"output/{task}/test/result40.json"
        audit_path = ROOT / f"audit/{task}/postrun_audit40.json"
        checked, index = verify_gate(str(audit_path.relative_to(ROOT)), "independent_postrun_PASS", task=task,
            protocol_seal_sha256=sha(ROOT / "protocol_seal40.json"), result_sha256=sha(task_result_path))
        primary = read(task_result_path)["primary"]
        expected = {"result_path": str(task_result_path.relative_to(ROOT)), "result_sha256": sha(task_result_path),
                    "audit_path": str(audit_path.relative_to(ROOT)), "audit_sha256": sha(audit_path),
                    "primary_gate": primary["primary_gate"], "test_n": primary["test_n"], "selected": primary["selected"],
                    "development_decision_certificate": primary["development_decision_certificate"],
                    "test_same_map_comparisons": primary["test_same_map_comparisons"]}
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
    receipt = {"schema": "step40-independent-two-domain-postrun-audit-v1", "protocol_sha256": sha(ROOT / "protocol40.json"),
               "protocol_seal_sha256": sha(ROOT / "protocol_seal40.json"), "result_sha256": sha(result_path),
               "tasks": tasks, "domain_count": 2, "practical_transfer_success": pass_all,
               "primary_status": result["primary_status"], "execution": execution,
               "all_candidate_outcomes_retained": True, "test_reselection": False, "assessment_oracle_fits": 0,
               "population_risk_claim": False, "future_response_coverage_claim": False,
               "wall_seconds": time.perf_counter() - started}
    return publish_gate("audit/postrun_audit40.json", "independent_aggregate_PASS", receipt)


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
        failure = {"schema": "step40-independent-audit-failure-v1", "status": "FAIL", "operation": args.operation,
                   "task": args.task, "stage": args.stage, "type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "scientific_runs_started_by_auditor": 0,
                   "auditor_source_sha256": sha(Path(__file__))}
        dest = ROOT / "audit" / (args.task or "global") / (args.stage or args.operation) / "retained_failure40.json"
        if not dest.exists():
            write(dest, failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
