"""Independent Step41 audit. Exact prediction/score arithmetic is separate from producers.

The checker reconstructs authorized decimal values directly from retained archive
byte ranges and audits causal fitting and prediction seals. Native map/baseline
replay verifies stored operational binary64 values; it does not enclose ideal
real-arithmetic feature transforms. No new scientific models are fitted here.
"""
from __future__ import annotations
import argparse
import os
for _thread in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_thread]="1"
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import io
import json
from math import isfinite, isqrt, nextafter, inf
from pathlib import Path
import re
import struct
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
TASKS = ("household_power2010", "jena_temperature2017")
STAGES = ("f1", "f2", "f3", "final")
YEARS = {"household_power2010":2010, "jena_temperature2017":2017}
STAGE_DATES = {task: {
 "f1": {"train_start":f"{year}-01-01", "train_stop":f"{year}-06-01", "validation_start":f"{year}-06-01", "validation_stop":f"{year}-07-01"},
 "f2": {"train_start":f"{year}-01-01", "train_stop":f"{year}-07-01", "validation_start":f"{year}-07-01", "validation_stop":f"{year}-08-01"},
 "f3": {"train_start":f"{year}-01-01", "train_stop":f"{year}-08-01", "validation_start":f"{year}-08-01", "validation_stop":f"{year}-09-01"},
 "final": {"train_start":f"{year}-01-01", "train_stop":f"{year}-09-01"},
} for task,year in YEARS.items()}
EXPECTED_N = {
 "household_power2010":{"f1":{"train":144,"validation":30},"f2":{"train":174,"validation":31},"f3":{"train":205,"validation":24},"final":{"train":229},"test":81},
 "jena_temperature2017":{"f1":{"train":149,"validation":30},"f2":{"train":179,"validation":31},"f3":{"train":210,"validation":31},"final":{"train":241},"test":91},
}
REP_CONFIG = {"dimension":24,"scalings":["minmax01","standard"],"nystrom_widths":[8,24],"kahm_caps":[4,12,20],"kahm_prefixes":[1,2],"cluster_counts":[8,16],"max_fit":512,"max_references":24,"tau":1e-6,"omega":2.0,"probe_count":32,"nystrom_eigen_floor":1e-12,"temporal_selection":"equal_row_stratum_midpoints","metric_normalization":"reciprocal_median_positive_pair_distance"}
class AuditError(AssertionError): pass

SOURCE_SPECS = {
    "household_power2010": {
        "uci_id": 235, "year": 2010, "separator": ";", "timestamp_indices": [0, 1], "target_index": 2,
        "target_header": "Global_active_power", "timestamp_headers": ["Date", "Time"], "header_encoding": "utf-8", "header_count": 9,
        "timestamp_formats": ["%d/%m/%Y %H:%M:%S"], "cadence_minutes": 1,
        "archives": [("household", "household_power_consumption_mirror.zip", "household_power_consumption.txt")],
        "missing_markers": ["", "?"], "signed": False, "numeric_domain": "finite nonnegative decimal",
        "unit": "kilowatt", "reported_instances": 2075259,
        "snapshot_description": "Minute-averaged active power at the recorded minute00; not an hourly average",
        "timezone_convention": "Source local-naive Date and Time; no UTC offset or daylight-saving reconstruction is invented",
        "missing_status_rule": "Predeclared exact empty or question-mark target field, after stripping surrounding ASCII spaces and tabs; no valid numeric token inspected",
    },
    "jena_temperature2017": {
        "uci_id": None, "year": 2017, "separator": ",", "timestamp_indices": [0], "target_index": 2,
        "target_header": "T (degC)", "timestamp_headers": ["Date Time"], "header_encoding": "latin-1", "header_count": 22,
        "timestamp_formats": ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"], "cadence_minutes": 10,
        "archives": [("jena2017a", "mpi_roof_2017a.zip", "mpi_roof_2017a.csv"),
                     ("jena2017b", "mpi_roof_2017b.zip", "mpi_roof_2017b.csv")],
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

def require(value, message):
    if not value:
        raise AuditError(message)

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
    result=(ROOT/path).resolve()
    require(result.is_relative_to(ROOT.resolve()), "Bound path escaped project")
    return result

def bound_reference(record):
    path = checked_path(record["path"])
    require(path.is_file() and sha(path) == record["sha256"], "Artifact hash changed: " + str(path))
    return path

def clip(value):
    return max(F(0), min(F(1), value))

def audit_events():
    path = ROOT / "output/access_and_execution41.jsonl"
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
    path = ROOT / "output/access_and_execution41.jsonl"
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
    if ARTIFICIAL_FIXTURE:
        return artificial_source_payloads(task)
    require(task in TASKS, "Unknown official public task")
    spec = SOURCE_SPECS[task]
    source = read(ROOT / f"inputs/data41/{task}/public_source41.json")
    require(source["schema"] == ("step41-retained-source-forward-cohort-v1" if task == "household_power2010" else "step41-opaque-public-source-download-v1") and source["task"] == task and
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
    if task == "household_power2010":
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
        relative = f"inputs/data41/{task}/{filename}"
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
    projection = read(ROOT / f"inputs/data41/{task}/calendar_projection41.json")
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
    return {"schema": "step41-calendar-status-byte-projection-v1", "task": task,
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

def authorized_values(task, timestamps):
    gate = read(ROOT / "audit/preexecution_gate41.json")
    require(gate["status"] == "PASS" and gate["protocol_seal_sha256"] == sha(ROOT / "protocol_seal41.json"),
            "Auditor numerical projection requires the frozen preexecution gate")
    requested = sorted(set(timestamps))
    projection = read(ROOT / f"inputs/data41/{task}/calendar_projection41.json")
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

def frozen_queries(task, stage, partition):
    roster = read(ROOT / "inputs/data41/calendar_roster41.json")
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
    projection = read(ROOT / f"inputs/data41/{task}/calendar_projection41.json")
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
    norm_path = ROOT / f"{base}/normalization41.json"
    normal = read(norm_path)
    if partition == "train":
        training_targets, training_ids = raw_targets, ids
    else:
        training = read(ROOT / f"{base}/cohorts/train.json")
        training_targets, training_ids = list(map(q, training["raw_targets"])), training["ids"]
    low, high = min(training_targets), max(training_targets)
    require(high > low and normal["schema"] == "step41-training-target-normalization-v1" and
            normal["task"] == task and normal["stage"] == stage and normal["fitted_partition"] == "train" and
            q(normal["target_min"]) == low and q(normal["target_max"]) == high and q(normal["span"]) == high - low and
            normal["train_target_ids_sha256"] == digest(training_ids) and
            normal["raw_train_targets_sha256"] == digest(training_targets) and normal["target_values_clipped"] is False and
            normal["future_data_used"] is False, "Target normalization was not fitted on this stage's training targets")
    targets = [(v - low) / (high - low) for v in raw_targets]
    persistence = [(v - low) / (high - low) for v in raw_persistence]
    X = np.asarray([[float(values[s]) for s in row] for row in lags], dtype=np.float64)
    saved_X = np.load(ROOT / record["X_path"], allow_pickle=False)
    require(record["schema"] == "step41-causal-cohort-v1" and record["task"] == task and record["stage"] == stage and
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
    access = read(ROOT / f"{base}/data_access_{partition}41.json")
    stop = datetime.fromisoformat(STAGE_DATES[task][stage][partition + "_stop"]) - timedelta(seconds=1)
    audit_access(access, task, stage, partition, needed, iso(stop))
    require(access["schema"] == "step41-development-measurement-access-v1" and access["status"] == "PASS" and
            access["cohort_path"] == relative and access["cohort_sha256"] == sha(ROOT / relative) and
            access["normalization_sha256"] == sha(norm_path) and
            complete["access_sha256"] == sha(ROOT / f"{base}/data_access_{partition}41.json"),
            "Development access receipt changed")
    return {**record, "X": X, "y": targets, "persistence": persistence,
            "normalization": normal, "path": relative, "sha256": sha(ROOT / relative)}


# Exact arithmetic below is implemented independently of the producer and its
# deployment verifier. It uses Fraction throughout rather than the producer
# verifier's common-denominator integer accumulation.
def q(value):
    require(not isinstance(value, bool), 'Boolean is not an explicit numeric value')
    if isinstance(value, float):
        require(isfinite(value), 'Nonfinite binary64 value')
        return F.from_float(value)
    require(isinstance(value, (str, int, F)), 'Unsupported rational payload')
    try:
        result = F(value)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise AuditError('Invalid rational payload') from error
    return result


def math_canon(value):
    if isinstance(value, (F, float)):
        return str(q(value))
    if isinstance(value, dict):
        return {str(k): math_canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [math_canon(v) for v in value]
    return value


def content_digest(value):
    return hashlib.sha256(json.dumps(math_canon(value), sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def box(record, expected=None):
    require(isinstance(record, dict), 'Expected interval record')
    low, high = q(record['lower']), q(record['upper'])
    require(low <= high, 'Reversed interval')
    if 'width' in record:
        require(q(record['width']) == high-low, 'Interval width differs')
    if expected is not None:
        require((low, high) == tuple(expected), 'Exact interval differs')
    return low, high


def exact_matrix(value):
    import numpy as np
    value = np.asarray(value)
    require(value.ndim == 2 and min(value.shape) > 0 and
            value.dtype == np.float64 and np.isfinite(value).all(), 'Invalid stored feature matrix')
    return [[F.from_float(float(v)) for v in row] for row in value]


def array_digest(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.shape).encode('ascii'))
    h.update(value.dtype.str.encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def maps_roster():
    return sorted(['raw_'+s for s in ('minmax01','standard')]+
      ['nystrom_'+s+'_w'+str(w) for s in ('minmax01','standard') for w in (8,24)]+
      ['kahm_'+s+'_cap'+str(c)+'_p'+str(p) for s in ('minmax01','standard') for c in (4,12,20) for p in (1,2)])


def class_descriptor(feature_id, kappa='2'):
    return {'schema':'step35-stored-feature-decoder-class-v1', 'feature_id':feature_id,
      'metric':'Euclidean_distance_on_supplied_feature_coordinates',
      'decoder_class':'all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space',
      'kappa':str(q(kappa)), 'output_interval':['0','1'], 'assurance':ASSURANCE}


def class_identity(feature_id, kappa='2'):
    return 'class35:'+content_digest(class_descriptor(feature_id,kappa))


def audit_head(head):
    require(head.get('schema') == 'step35-stored-feature-head-v1' and
            head.get('kind') == 'clipped_affine' and head.get('assurance') == ASSURANCE,
            'Unknown fixed affine head schema or assurance')
    require(head.get('output_interval') == ['0','1'] and q(head['kappa']) == 2,
            'Frozen head range or sensitivity differs')
    require(isinstance(head['feature_id'],str) and head['feature_id'] and
            head['class_id'] == class_identity(head['feature_id'],head['kappa']), 'Affine class identity differs')
    require(type(head['dimension']) is int and head['dimension']>0, 'Invalid head dimension')
    beta=list(map(q,head['coefficients']))
    require(len(beta)==head['dimension'] and sum((v*v for v in beta),F(0))<=4,
            'Affine coefficient norm exceeds declared sensitivity')
    q(head['intercept'])
    return 'head35:'+content_digest(head)


def audit_affine_record(record, z=None, y=None):
    hid=audit_head(record['head'])
    require(record.get('schema')=='step35-admissible-affine-head-v1' and record['head_id']==hid,
            'Affine fitting wrapper identity differs')
    require(record['native_transform_error_enclosed'] is False and record['population_risk_claim'] is False,
            'Affine fitting record overstates scope')
    p=record['projection']; source=list(map(q,p['source_coefficients']))
    require(len(source)==record['head']['dimension'], 'Projection source dimension differs')
    squared=sum((v*v for v in source),F(0))
    require(q(p['source_squared_norm'])==squared, 'Projection source norm differs')
    if squared<=4:
        factor=F(1)
    else:
        denom=1<<80
        root=isqrt((squared.numerator<<(2*80))//squared.denominator)
        upper=F(root if F(root,denom)**2==squared else root+1,denom)
        factor=F(2)/upper
    require(q(p['factor'])==factor and p['projected'] is (factor!=1) and
            list(map(q,record['head']['coefficients']))==[factor*v for v in source],
            'Exact inherited affine projection recipe differs')
    norm=sum(((factor*v)**2 for v in source),F(0))
    require(q(record['membership']['coefficient_squared_norm'])==norm and
            q(record['membership']['kappa_squared'])==4 and norm<=4,
            'Affine norm membership receipt differs')
    require(p['intercept_recentered'] is True, 'Inherited affine recentering omitted')
    center=list(map(q,p['center_features'])); target=q(p['center_target'])
    require(len(center)==len(source) and q(record['head']['intercept']) ==
            target-sum((factor*b*x for b,x in zip(source,center)),F(0)), 'Affine recentering differs')
    if z is not None:
        import numpy as np
        z=np.asarray(z,dtype=np.float64); yr=list(map(q,y)); exact=exact_matrix(z)
        require(len(z)==len(yr), 'Affine fit row roster differs')
        expected_center=[sum((r[j] for r in exact),F(0))/len(yr) for j in range(z.shape[1])]
        require(center==expected_center and target==sum(yr,F(0))/len(yr), 'Affine center uses another cohort')
        yf=np.asarray([float(v) for v in yr]); xc=z-z.mean(axis=0); yc=yf-yf.mean()
        beta=np.linalg.solve(xc.T@xc+len(z)*1e-6*np.eye(z.shape[1]),xc.T@yc)
        intercept=float(yf.mean()-z.mean(axis=0)@beta)
        require(source==[F.from_float(float(v)) for v in beta] and q(p['source_intercept'])==F.from_float(intercept),
                'Affine numerical proposal differs from frozen training-only recipe')
    return {'status':'PASS','head_id':hid,'projected':factor!=1,'training_optimality_claim':False}


def float_hex(value):
    require(type(value) is str, 'Expected canonical binary64 hexadecimal string')
    try:
        number=float.fromhex(value)
    except (ValueError,OverflowError) as error:
        raise AuditError('Invalid binary64 hex') from error
    require(isfinite(number) and number.hex()==value, 'Nonfinite or noncanonical binary64 hex')
    return number


def tight_float_interval(value):
    value=q(value); mid=float(value)
    require(isfinite(mid), 'Finite physical reference required')
    rm=F.from_float(mid)
    low=nextafter(mid,-inf) if rm>value else mid
    high=nextafter(mid,inf) if rm<value else mid
    require(isfinite(low) and isfinite(high) and F.from_float(low)<=value<=F.from_float(high),
            'Physical reference has no finite binary64 enclosure')
    return low.hex(),high.hex()


def exact_head_value(head,z):
    audit_head(head)
    vector=[F.from_float(float(v)) for v in z]
    require(len(vector)==head['dimension'], 'Query feature dimension differs')
    raw=q(head['intercept'])+sum((q(b)*x for b,x in zip(head['coefficients'],vector)),F(0))
    return clip(raw)


def physical_loss(values, targets):
    require(len(values)==len(targets)>0, 'Loss roster mismatch')
    terms=[(q(a)-q(b))**2 for a,b in zip(values,targets)]
    sse=sum(terms,F(0)); mse=sse/len(terms)
    return {'n':len(terms),'sse':str(sse),'mse':str(mse)}


def audit_primary(affine_mse, baseline_mse):
    a,b=q(affine_mse),q(baseline_mse)
    require(a>=0 and b>=0, 'Negative assessed MSE')
    return {'baseline_positive':b>0,'within_five_percent_mse':b>0 and 20*a<=21*b,
      'strictly_lower_mse':a<b,'at_least_five_percent_lower_mse':b>0 and 20*a<=19*b,
      'relative_mse_reduction':str(1-a/b) if b>0 else None,
      'status':'PASS' if b>0 and 20*a<=21*b else 'NOT_ESTABLISHED',
      'population_noninferiority_claim':False}


def audit_pooled_selection(rows, selector, pool_ids):
    require(rows and pool_ids and len(set(pool_ids))==len(pool_ids), 'Invalid selection pool')
    lookup={r['candidate_id']:r for r in rows}
    require(len(lookup)==len(rows) and set(pool_ids)<=set(lookup), 'Selection candidate identities differ')
    pool=[lookup[cid] for cid in pool_ids]
    winner=min(pool,key=lambda r:(q(r['pooled_physical_mse']),r['selection_payload_bytes'],r['candidate_id']))
    for row in pool:
        require(type(row['selection_payload_bytes']) is int and row['selection_payload_bytes']>=0,
                'Invalid selection payload tie breaker')
    return {'selector':selector,'candidate_id':winner['candidate_id'],
       'pooled_physical_mse':winner['pooled_physical_mse'],
       'selection_payload_bytes':winner['selection_payload_bytes'],'pool_size':len(pool)}

CERTIFICATE_SCOPE = {
    'assurance':ASSURANCE, 'reference':'FIXED_STORED_RATIONAL_AFFINE_HEAD',
    'affine_training_optimum_certified':False, 'native_transform_error_enclosed':False,
    'raw_input_robustness_certified':False, 'future_response_coverage_claim':False,
    'population_risk_claim':False, 'cryptographic_authenticity_claim':False,
}


def identity_bytes(record):
    return json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def audit_deployment(record, *, expected_head=None, expected_candidate=None, expected_map_sha256=None, expected_map_bytes=None):
    keys={'schema','head','head_id','candidate_id','target_min','span','map_sha256','map_bytes',
          'normalized_error_budget','scope','deployment_id'}
    require(type(record) is dict and set(record)==keys, 'Static deployment fields differ')
    require(record['schema']=='step41-affine-deployment-v1' and record['scope']==CERTIFICATE_SCOPE,
            'Static deployment scope differs')
    body={k:v for k,v in record.items() if k!='deployment_id'}
    require(record['deployment_id']=='deploy41:'+hashlib.sha256(identity_bytes(body)).hexdigest(),
            'Static deployment content identity differs')
    hid=audit_head(record['head'])
    require(record['head_id']==hid and isinstance(record['candidate_id'],str) and record['candidate_id'],
            'Static deployment head/candidate differs')
    require(record['normalized_error_budget']=='1/1099511627776' and
            type(record['map_bytes']) is int and record['map_bytes']>0 and
            re.fullmatch('[0-9a-f]{64}',record['map_sha256']) is not None, 'Static budget/map binding differs')
    for value in [record['target_min'],record['span'],record['head']['intercept'],
                  record['head']['kappa'],*record['head']['coefficients']]:
        require(type(value) is str and str(q(value))==value, 'Noncanonical exact deployment coefficient/scale')
    require(q(record['span'])>0, 'Nonpositive physical target scale')
    if expected_head is not None:require(record['head']==expected_head,'Deployment head differs from fitted payload')
    if expected_candidate is not None:require(record['candidate_id']==expected_candidate,'Deployment candidate differs')
    if expected_map_sha256 is not None:require(record['map_sha256']==expected_map_sha256,'Deployment map hash differs')
    if expected_map_bytes is not None:require(record['map_bytes']==expected_map_bytes,'Deployment map byte size differs')
    return {'status':'PASS','deployment_id':record['deployment_id'],'head_id':hid,'dimension':record['head']['dimension']}


def feature_packet(z):
    values=[float(v) for v in z]
    require(values and all(isfinite(v) for v in values), 'Nonfinite feature packet')
    return b'step41-feature-vector-v1\0'+struct.pack('>I',len(values))+b''.join(struct.pack('>d',v) for v in values)


def audit_certificate(deployment, z, record, *, query_id, reported_prediction, check_static=True):
    if check_static:audit_deployment(deployment)
    expected_keys={'schema','deployment_id','query_id','feature_sha256','reported_prediction_hex',
                   'reference_interval_hex','absolute_error_upper_hex'}
    require(type(record) is dict and set(record)==expected_keys,'Query certificate fields differ')
    require(record['schema']=='step41-affine-query-certificate-v1' and
            record['deployment_id']==deployment['deployment_id'] and record['query_id']==query_id,
            'Certificate deployment/query identity differs')
    require(record['feature_sha256']==hashlib.sha256(feature_packet(z)).hexdigest(),
            'Certificate stored feature bytes differ')
    actual=float(reported_prediction)
    require(isfinite(actual) and record['reported_prediction_hex']==actual.hex(),
            'Certificate does not bind actual reported prediction')
    float_hex(record['reported_prediction_hex'])
    reference=q(deployment['target_min'])+q(deployment['span'])*exact_head_value(deployment['head'],z)
    expected_interval=list(tight_float_interval(reference))
    require(record['reference_interval_hex']==expected_interval, 'Certificate reference enclosure is not exact minimal hull')
    error=abs(F.from_float(actual)-reference)
    upper=tight_float_interval(error)[1]
    require(record['absolute_error_upper_hex']==upper, 'Certificate error upper is not exact upward rounding')
    require(error<=q(deployment['span'])*F(1,1<<40),'Operational prediction exceeds frozen exact discrepancy budget')
    return {'status':'PASS','exact_reference_physical':str(reference),'absolute_error_exact':str(error),
            'normalized_error_exact':str(error/q(deployment['span'])), 'accepted_prediction_accuracy':True}


def audit_risk(record, values, targets):
    import math
    expected=physical_loss(values,targets)
    mae=sum((abs(q(a)-q(b)) for a,b in zip(values,targets)),F(0))/len(targets)
    require(set(record)=={'n','sse','mse','rmse','mae'} and
            record['n']==expected['n'] and q(record['sse'])==q(expected['sse']) and
            q(record['mse'])==q(expected['mse']) and q(record['mae'])==mae and
            record['rmse']==math.sqrt(float(q(expected['mse']))), 'Exact physical score differs')
    return expected


def primary_fields(affine_mse, baseline_mse):
    result=audit_primary(affine_mse,baseline_mse);a,b=q(affine_mse),q(baseline_mse)
    return {'positive_reference_mse':b>0,'within_five_percent_mse':result['within_five_percent_mse'],
      'strictly_lower_mse':result['strictly_lower_mse'],
      'at_least_five_percent_lower_mse':result['at_least_five_percent_lower_mse'],
      'mse_ratio':float(a/b) if b>0 else None,
      'mse_reduction_percent':100*float(1-a/b) if b>0 else None,
      'scope':'FIXED_PERIOD_PRACTICAL_COMPARISON_NOT_POPULATION_NONINFERIORITY'}


def selector_ids(rows,label):
    if label=='affine_rolling':return [r['candidate_id'] for r in rows if r['recipe']=='affine']
    if label=='ordinary_rolling':return [r['candidate_id'] for r in rows if r['recipe']=='baseline']
    if label in ('raw_affine','nystrom_affine','kahm_affine'):
        family=label.split('_')[0]
        return [r['candidate_id'] for r in rows if r['recipe']=='affine' and r['family']==family]
    kinds=('ridge','hgb','svr','krr','extra_trees','random_forest')
    require(label in tuple(k+'_rolling' for k in kinds),'Unknown frozen selector')
    return [r['candidate_id'] for r in rows if r.get('baseline_kind')==label[:-8]]


def load_context(task,stage):
    import numpy as np
    import pickle
    import representations35 as replay
    manifest=read(ROOT/f'output/{task}/{stage}/fit_manifest41.json')
    replay.initialize_runtime()
    maps={};features={};heads={};fits={};deployments={};baselines={}
    for name,info in manifest['maps'].items():
        maps[name]=replay.deserialize_map(bound_reference({'path':info['map_path'],'sha256':info['map_sha256']}).read_bytes())
        features[name]=np.load(bound_reference(info['training_features']),allow_pickle=False)
    for row in manifest['candidate_rows']:
        cid=row['candidate_id']
        heads[cid]=read(bound_reference({'path':row['head_path'],'sha256':row['head_sha256']}))
        fits[cid]=read(bound_reference({'path':row['fit_path'],'sha256':row['fit_sha256']}))
        deployments[cid]=read(bound_reference({'path':row['deployment_path'],'sha256':row['deployment_sha256']}))
    for row in manifest['baseline_rows']:
        baselines[row['candidate_id']]=pickle.loads(bound_reference({'path':row['payload_path'],'sha256':row['payload_sha256']}).read_bytes())
    return {'manifest':manifest,'maps':maps,'train_features':features,'heads':heads,'fits':fits,
            'deployments':deployments,'baselines':baselines}


def audit_stage_models(protocol,task,stage,training,context=None):
    import numpy as np
    from scipy.spatial.distance import pdist
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import rbf_kernel
    import representations35 as replay
    context=load_context(task,stage) if context is None else context
    manifest=context['manifest'];X=training['X'];y=list(map(q,training['y']));ids=training['ids']
    names=maps_roster();rows=manifest['candidate_rows'];baseline_rows=manifest['baseline_rows']
    require(manifest['schema']=='step41-fit-manifest-v1' and manifest['task']==task and manifest['stage']==stage and
            manifest['training_n']==len(y)==len(X)==len(ids) and manifest['normalization']==training['normalization'] and
            manifest['training_cohort']=={'path':training['path'],'sha256':training['sha256']} and
            sorted(manifest['maps'])==names and
            [r['candidate_id'] for r in rows+baseline_rows]==protocol['evaluation']['prediction_ids'],
            'Fit manifest training/map/prediction roster differs')
    require(manifest['selection_seal_sha256']==(sha(ROOT/f'output/{task}/selection_seal41.json') if stage=='final' else None),
            'Final fit is not bound to prior development selection')
    metadata=read(ROOT/f'output/{task}/{stage}/representation_metadata41.json')
    probes=[((2*j+1)*len(X))//64 for j in range(32)]
    require(metadata['schema']=='step35-frozen-representations-v1' and metadata['config']==REP_CONFIG and
            metadata['seed']==3901 and metadata['training_shape']==list(X.shape) and
            metadata['training_sha256']==array_digest(X) and metadata['probe_indices']==probes and
            metadata['probe_sha256']==array_digest(X[probes]) and metadata['factory_source_sha256']==sha(Path(replay.__file__)) and
            metadata['forecast_labels_used'] is False and metadata['no_heldout_input_or_label_used'] is True and
            set(metadata['map_order'])==set(names) and len(metadata['map_order'])==18,
            'Representation training-only input/configuration binding differs')
    for scaling in ('minmax01','standard'):
        saved=metadata['scalers'][scaling];low,high=X.min(axis=0),X.max(axis=0);constant=low==high
        offset=low if scaling=='minmax01' else X.mean(axis=0)
        divisor=high-low if scaling=='minmax01' else X.std(axis=0,ddof=0)
        divisor=np.where(constant,1.,divisor)
        require(np.array_equal(saved['offset'],offset) and np.array_equal(saved['divisor'],divisor) and
                saved['constant_mask']==constant.tolist() and saved['training_rows']==len(X) and
                saved['heldout_clipping'] is False, 'Representation input scaler is not training-only')
    by_name={r['name']:r for r in metadata['maps']}
    for row in rows:
        name=row['map'];cid=row['candidate_id'];info=manifest['maps'][name];meta=by_name[name]
        model=context['maps'][name];z=context['train_features'][name];head=context['heads'][cid];fit=context['fits'][cid]
        require(cid==name+'::affine' and row['recipe']=='affine' and row['token']=='affine' and
                row['family']==info['family']==model.family==meta['family'] and row['scaling']==info['scaling']==model.scaling==meta['scaling'] and
                row['dimension']==info['dimension']==z.shape[1]==head['dimension']==meta['actual_width'] and
                z.dtype==np.float64 and z.shape[0]==len(X) and np.isfinite(z).all() and
                np.array_equal(z.view(np.uint64),replay.transform_one(model,X).view(np.uint64)), 'Fitted affine map/training features differ')
        map_path=checked_path(info['map_path']);dp_path=checked_path(row['deployment_path']);head_path=checked_path(row['head_path'])
        require(map_path.stat().st_size==row['map_payload_bytes']==info['map_payload_bytes']==meta['payload_bytes'] and
                dp_path.stat().st_size==row['static_deployment_bytes'] and head_path.stat().st_size==row['head_payload_bytes'] and
                row['deployment_payload_bytes']==map_path.stat().st_size+dp_path.stat().st_size and
                info['map_sha256']==meta['payload_sha256'], 'Complete deployment payload accounting differs')
        expected_fid='feature41:'+digest({'task':task,'stage':stage,'map_sha256':info['map_sha256'],
            'train_feature_sha256':info['training_features']['sha256']})
        require(info['feature_id']==row['feature_id']==head['feature_id']==expected_fid,'Feature identity differs')
        bind={'task':task,'stage':stage,'cohort_path':training['path'],'cohort_sha256':training['sha256'],
              'feature_path':info['training_features']['path'],'feature_sha256':info['training_features']['sha256'],
              'map_sha256':info['map_sha256']}
        require(fit['training_binding']==bind and fit['head']==head,'Affine fit-data/head binding differs')
        audit_affine_record(fit,z,y)
        dp=context['deployments'][cid]
        audit_deployment(dp,expected_head=head,expected_candidate=cid,expected_map_sha256=info['map_sha256'],expected_map_bytes=info['map_payload_bytes'])
        require(dp['deployment_id']==row['deployment_id'] and dp['target_min']==training['normalization']['target_min'] and
                dp['span']==training['normalization']['span'], 'Deployment normalization/identity differs')
        distances=pdist(model.core.transform(X[probes]),metric='euclidean');positive=distances[distances>0]
        metric=1./float(np.median(positive)) if len(positive) else 1.
        require(float(model.metric_scale)==metric==meta['metric_scale'] and meta['positive_probe_pair_count']==len(positive),
                'Feature metric used another probe cohort')
        if model.family=='nystrom':
            width=meta['requested_width'];indices=[((2*j+1)*len(X))//(2*width) for j in range(width)]
            require(meta['landmark_indices']==indices and np.array_equal(model.core.landmarks,model.core.scaler.transform(X[indices])),
                    'Nystrom landmarks are not fixed training-row midpoints')
    cfgs=protocol['baselines']['configurations']
    require(len(cfgs)==len(baseline_rows)==44,'Ordinary baseline panel size differs')
    yf=np.asarray([float(v) for v in y]);baseline_scaler=StandardScaler().fit(X);sx=baseline_scaler.transform(X)
    norms={'target_min':training['normalization']['target_min'],'span':training['normalization']['span']}
    linear_checks=kernel_checks=0
    for cfg,row in zip(cfgs,baseline_rows):
        cid=cfg['candidate_id'];value=context['baselines'][cid];kind=cfg['kind']
        require(all(row[k]==v for k,v in cfg.items()) and row['recipe']=='baseline' and row['family']=='ordinary' and
                row['baseline_kind']==kind and value['kind']==kind and value['normalization']==norms and
                row['deployment_payload_bytes']==checked_path(row['payload_path']).stat().st_size,
                'Baseline configuration/payload identity differs')
        if kind=='persistence':
            require(value=={'kind':kind,'normalization':norms,'rule':'last_observed_physical_own_target'},'Persistence rule differs')
        elif kind=='mean':
            expected=q(norms['target_min'])+q(norms['span'])*sum(y,F(0))/len(y)
            require(q(value['value'])==expected,'Ordinary mean was not fitted to this training cohort')
        else:
            scaler=value['scaler'];model=value['model'];params=model.get_params()
            require(value['fit_n']==len(y) and scaler.n_samples_seen_==len(X) and model.n_features_in_==24 and
                    np.array_equal(scaler.mean_,baseline_scaler.mean_) and np.array_equal(scaler.var_,baseline_scaler.var_) and
                    np.array_equal(scaler.scale_,baseline_scaler.scale_) and scaler.get_params()==baseline_scaler.get_params(),
                    'Ordinary model scaler differs from training-only StandardScaler')
            offset=float(yf.mean()) if kind=='krr' else 0.
            require(value['offset']==offset,'Ordinary target centering differs')
            expected_params=dict(cfg['parameters'])
            if kind=='ridge':
                expected_params['alpha']=len(y)*float(q(cfg['ridge_lambda']))
                require(all(params[k]==v for k,v in expected_params.items()),'Ridge parameterization differs')
                xc=sx-sx.mean(axis=0);yc=yf-yf.mean();mat=xc.T@xc+expected_params['alpha']*np.eye(24)
                from scipy.linalg import solve
                coefficient=solve(mat,xc.T@yc,assume_a='pos')
                intercept=float(yf.mean()-sx.mean(axis=0)@coefficient)
                require(np.allclose(model.coef_,coefficient,rtol=2e-11,atol=2e-12) and
                        np.isclose(model.intercept_,intercept,rtol=2e-11,atol=2e-12), 'Ridge normal equations differ')
                linear_checks+=1
            else:
                require(params==expected_params,'Frozen ordinary model parameters differ')
            if kind=='hgb':
                require(model.n_iter_==300 and float(model._baseline_prediction[0,0])==float(yf.mean()),'HGB iteration/target binding differs')
            elif kind=='svr':
                require(model.fit_status_==0 and np.array_equal(model.support_vectors_,sx[model.support_]),'SVR support rows are not this training cohort')
            elif kind=='krr':
                require(np.array_equal(model.X_fit_,sx),'KRR kernel support differs from training rows')
                kernel=rbf_kernel(sx,sx,gamma=params['gamma']);kernel.flat[::len(y)+1]+=params['alpha']
                residual=kernel@model.dual_coef_-(yf-offset)
                require(np.max(np.abs(residual))<=2e-9*(1+np.max(np.abs(yf-offset))), 'KRR trained coefficients violate declared normal equations')
                kernel_checks+=1
            elif kind in ('extra_trees','random_forest'):
                require(len(model.estimators_)==256 and all(t.n_features_in_==24 for t in model.estimators_), 'Forest count/feature budget differs')
    require([r['candidate_id'] for r in manifest['group_timings']]==protocol['evaluation']['prediction_ids'] and
            all(isfinite(r['seconds']) and r['seconds']>=0 for r in manifest['group_timings']) and
            isfinite(manifest['representation_seconds']) and manifest['representation_seconds']>=0,
            'Model work receipt differs')
    return {'status':'PASS','maps':18,'affine_heads':18,'baseline_models':44,
      'affine_ridge_proposal_reconstructions':18,'ordinary_ridge_checks':linear_checks,'kernel_ridge_equation_checks':kernel_checks,
      'new_scientific_model_fits_by_auditor':0,'stored_native_feature_replays':18}


def baseline_values(payload,X):
    import numpy as np
    kind=payload['kind']
    if kind=='persistence':result=np.asarray(X)[:,-1]
    elif kind=='mean':result=np.full(len(X),float(q(payload['value'])))
    else:
        normal=payload['normalization']
        result=float(q(normal['target_min']))+float(q(normal['span']))*(payload['model'].predict(payload['scaler'].transform(X))+payload['offset'])
    require(np.isfinite(result).all(),'Nonfinite baseline replay')
    return [float(v) for v in result]


def audit_prediction_bundle(protocol,task,stage,record,context,X,ids):
    import numpy as np
    import representations35 as replay
    require(record['schema'] in ('step41-prediction-bundle-v1','step41-sealed-test-predictions-v1') and
            record['task']==task and record['stage']==stage and record['query_ids']==ids and
            set(record['predictions'])==set(protocol['evaluation']['prediction_ids']) and
            set(record['maps'])==set(protocol['maps']) and
            record['fit_seal_sha256']==sha(ROOT/f'output/{task}/{stage}/fit_seal41.json'), 'Prediction bundle identities differ')
    count=0;errors=[]
    for row in context['manifest']['candidate_rows']:
        name=row['map'];cid=row['candidate_id'];dp=context['deployments'][cid];head=dp['head']
        z=np.load(bound_reference(record['maps'][name]['features']),allow_pickle=False)
        require(z.dtype==np.float64 and np.isfinite(z).all() and z.shape==(len(ids),head['dimension']) and
                np.array_equal(z.view(np.uint64),replay.transform_one(context['maps'][name],X).view(np.uint64)), 'Query operational feature replay differs')
        predictions=record['predictions'][cid]
        require(len(predictions)==len(ids),'Affine prediction query roster differs')
        rounded=np.asarray([float(q(v)) for v in head['coefficients']],dtype=np.float64)
        for qid,vector,saved in zip(ids,z,predictions):
            require(set(saved)=={'query_id','reported_physical_hex','exact_physical','certificate'} and saved['query_id']==qid,
                    'Affine query result binding differs')
            raw=float(np.dot(rounded,vector))+float(q(head['intercept']))
            operational=float(q(dp['target_min']))+float(q(dp['span']))*min(1.,max(0.,raw))
            require(saved['reported_physical_hex']==float(operational).hex(),'Reported affine formula replay differs')
            checked=audit_certificate(dp,vector,saved['certificate'],query_id=qid,reported_prediction=operational,check_static=False)
            require(saved['exact_physical']==checked['exact_reference_physical'],'Stored affine exact reference differs')
            errors.append(q(checked['normalized_error_exact']));count+=1
    for row in context['manifest']['baseline_rows']:
        cid=row['candidate_id'];predictions=record['predictions'][cid];values=baseline_values(context['baselines'][cid],X)
        require(len(predictions)==len(ids),'Baseline prediction query roster differs')
        for qid,value,saved in zip(ids,values,predictions):
            require(saved=={'query_id':qid,'reported_physical_hex':value.hex(),'exact_physical':None,'certificate':None},
                    'Ordinary operational prediction replay differs')
    return {'status':'PASS','queries':len(ids),'affine_prediction_certificates':count,
            'ordinary_prediction_replays':len(ids)*44,'maximum_normalized_error':str(max(errors,default=F(0)))}


ORIGINAL_ROOT=Path(__file__).resolve().parents[1]
ARTIFICIAL_FIXTURE=False
SELECTORS=('affine_rolling','ordinary_rolling','ridge_rolling','hgb_rolling','svr_rolling','krr_rolling',
           'extra_trees_rolling','random_forest_rolling','raw_affine','nystrom_affine','kahm_affine')


def artificial_scope():
    if ARTIFICIAL_FIXTURE:
        require(ROOT.resolve().is_relative_to((ORIGINAL_ROOT/'tmp').resolve()) and ROOT.resolve()!=ORIGINAL_ROOT.resolve(),
                'Artificial fixture cannot run against the scientific project root')
    return ARTIFICIAL_FIXTURE


def artificial_source_payloads(task):
    require(artificial_scope(),'Synthetic source route not authorized in scientific mode')
    source=read(ROOT/f'inputs/data41/{task}/public_source41.json');spec=SOURCE_SPECS[task]
    require(source.get('artificial_only') is True and source.get('schema')=='step41-artificial-opaque-source-v1',
            'Artificial source must identify synthetic-only content')
    require(source['task']==task and source['year']==YEARS[task] and
            sha(checked_path(source['metadata_path']))==source['metadata_sha256'], 'Artificial metadata identity differs')
    keys=('source_id','archive_path','archive_sha256','archive_bytes','member','source_url')
    projection=[{k:r[k] for k in keys} for r in source['archives']]
    require(digest(projection)==source['source_manifest_sha256'] and len(projection)==len(spec['archives']),
            'Artificial archive manifest differs')
    payloads={}
    for item,(sid,filename,member) in zip(source['archives'],spec['archives']):
        path=f'inputs/data41/{task}/{filename}'
        require(item['archive_path']==path and item['source_id']==sid and item['member']==member and
                item['source_url']==item['resolved_url']=='artificial://fixture/'+sid and
                checked_path(path).stat().st_size==item['archive_bytes'], 'Artificial archive source binding differs')
        raw=cached_archive_member(path,item['archive_sha256'],member)
        with zipfile.ZipFile(checked_path(path)) as archive:
            compressed=archive.getinfo(member).compress_size
        require(item['inventory']==[{'name':member,'bytes':len(raw),'compressed_bytes':compressed}],
                'Artificial archive inventory differs')
        payloads[sid]=raw
    return payloads,source


def audit_metadata():
    records={}
    for task in TASKS:
        payloads,source=source_payloads(task)
        expected=independent_calendar_projection(task,payloads)
        expected['source_manifest_sha256']=source['source_manifest_sha256']
        saved=read(ROOT/f'inputs/data41/{task}/calendar_projection41.json')
        for k,v in expected.items():require(saved.get(k)==v,'Independent calendar projection differs: '+task+'/'+k)
        require(set(saved)==set(expected)|{'projection_utc','projection_source_sha256'}, 'Unexpected calendar evidence field')
        datetime.fromisoformat(saved['projection_utc'])
        require(saved['projection_source_sha256']==sha(ROOT/'source/prepare_data41.py'), 'Calendar projection source code changed')
        counts={stage:{part:len(frozen_queries(task,stage,part)) for part in ('train','validation')
                      if stage!='final' or part=='train'} for stage in STAGES}
        counts['test']=len(frozen_queries(task,'final','test'))
        require(counts==EXPECTED_N[task],'Calendar eligibility counts differ')
        records[task]={'counts':counts,'source_manifest_sha256':source['source_manifest_sha256'],
                      'projection_sha256':sha(ROOT/f'inputs/data41/{task}/calendar_projection41.json'),
                      'measurement_values_decoded':0}
    return {'status':'PASS','tasks':records,'measurement_values_decoded':0,'artificial_only':artificial_scope()}


def audit_protocol(p):
    import importlib.metadata
    require(p['schema']=='step41-affine-deployment-protocol-v1' and p['step']==41 and p['seed']==3901 and
            p['tasks']==list(TASKS) and p['stages']==list(STAGES) and p['maps']==maps_roster() and
            p['representations']==REP_CONFIG, 'Frozen scientific task/map roster differs')
    require(bool(p.get('artificial_only',False)) is artificial_scope(),'Protocol artificial/scientific scope differs')
    heads=[{'candidate_id':name+'::affine','map':name,'recipe':'affine','token':'affine'} for name in maps_roster()]
    require(p['heads']['configurations']==heads and p['heads']['affine_ridge_lambda']=='1/1000000' and
            p['heads']['kappa']=='2' and p['heads']['normalized_range']==['0','1'] and
            p['heads']['training_optimality_claim'] is False, 'Frozen affine recipe differs')
    baselines=read(ROOT/'analysis/baseline_grid41.json')
    require(p['baselines']['configurations']==baselines and p['baselines']['count']==44 and
            p['baselines']['same24lags'] is True and p['baselines']['clipped'] is False and
            p['baselines']['class_membership_certified'] is False,
            'Stronger conventional baseline grid or information scope differs')
    require(Counter(r['kind'] for r in baselines)=={'persistence':1,'mean':1,'ridge':5,'hgb':8,'svr':10,'krr':9,'extra_trees':6,'random_forest':4},
            'Baseline family count differs')
    require(p['evaluation']['prediction_ids']==[r['candidate_id'] for r in heads+baselines] and
            len(set(p['evaluation']['prediction_ids']))==62, 'Complete candidate panel differs')
    sel=p['selection'];ev=p['evaluation'];cert=p['certificate'];timing=p['timing']
    require(sel['folds']==['f1','f2','f3'] and sel['selectors']==list(SELECTORS) and
            sel['primary']=='affine_rolling' and sel['reference']=='ordinary_rolling' and sel['test_reselection'] is False and
            sel['tie_breakers']==['exact pooled physical MSE','f3 map+deployment or baselinepayload bytes','lexical candidate_id'],
            'Frozen pooled selection policy differs')
    require(ev['primary_per_task']=='baselineMSE>0 and20*affineMSE<=21*baselineMSE andallvaliditygatesPASS' and
            ev['all_candidate_outcomes_retained'] is True and ev['outcome_based_reselection'] is False and
            ev['population_significance_test'] is False, 'Fixed-period five-percent primary criterion differs')
    require(cert['normalized_error_budget']=='1/1099511627776' and
            all(cert[k] is False for k in ('native_transform_error_enclosed','raw_input_robustness','future_label_coverage','population_risk')),
            'Certificate budget or mathematical assurance scope differs')
    require(timing['threads']==1 and timing['warmups_per_query_primitive']==3 and timing['repeats_per_query_primitive']==10 and
            timing['static_prepare_repeats']==100 and timing['primitives']==['transform','head_inference','certificate_construction','certificate_serialization','certificate_verification','prediction_without_certificate','certified_prediction','ordinary_prediction'] and
            timing['runtime_threshold'] is None and timing['speedup_claim'] is False, 'Deployment timing protocol differs')
    require(p['execution']['single_primary_pass'] is True and p['execution']['scientific_reruns_allowed'] is False and
            p['execution']['threads']==1 and p['execution']['failure_policy']=='STOP_RETAIN_FAILURE_NO_RETUNING',
            'Single-pass execution policy differs')
    data=p['data'];roster=read(ROOT/'inputs/data41/calendar_roster41.json')
    require(data['task_order']==list(TASKS) and data['stages']==STAGE_DATES and
            data['test_intervals']=={task:{'start':test_dates(task)[0],'stop_exclusive':test_dates(task)[1]} for task in TASKS} and
            data['horizon_hours']==data['lag_hours']==24 and data['daily_target_hour']==23 and
            data['calendar_roster_sha256']==sha(ROOT/'inputs/data41/calendar_roster41.json') and
            all(data[k] is False for k in ('targets_clipped','other_measurement_predictors','future_calendar_predictors')) and
            data['no_online_refit'] is True and data['previously_revealed_targets_may_be_later_history'] is True,
            'Frozen data chronology/information scope differs')
    for task in TASKS:
        require(data['sources'][task]['expected_n']==EXPECTED_N[task] and data['sources'][task]['year']==YEARS[task] and
                data['sources'][task]['source_manifest_sha256']==roster['tasks'][task]['source_manifest_sha256'],
                'Frozen cohort identity/count differs')
    for package,version in p['environment']['packages'].items():
        require(importlib.metadata.version(package)==version,'Scientific package version changed: '+package)
    return {'status':'PASS','tasks':2,'affine_candidates':18,'ordinary_candidates':44,'selectors':11,
            'finite_period_tolerance':'21/20','statistical_noninferiority_claim':False}


def verify_protocol_seal():
    seal=verify_seal('protocol_seal41.json','step41-protocol-source-seal-v1')
    require(seal['protocol_sha256']==sha(ROOT/'protocol41.json') and seal['new_measurement_tokens_decoded']==0 and
            seal['single_primary_pass'] is True, 'Frozen protocol must precede new measurement access')
    required={'protocol41.json','source/audit41.py','source/test_audit41.py','source/data41.py','source/prepare_data41.py',
              'source/check_data41.py','source/run41.py','source/common41.py','source/protocol41.py','source/models41.py',
              'source/affine41.py','source/certificate41.py','source/test_certificate41.py','source/representations35.py',
              'source/finite_feature35.py','analysis/interface41.txt','analysis/baseline_grid41.json','inputs/inherited41.json',
              'inputs/data41/calendar_roster41.json','inputs/data41/prior_exposure41.json'}
    inherited=read(ROOT/'inputs/inherited41.json')
    require(inherited['step40_archive_sha256']=='3da58e36ed6a318ab748604f8b58a3ac293ee96f831987874e64fc38ca3f70ee' and
            inherited['affine_function_unchanged'] is True,'Inherited prior record/function identity differs')
    import ast
    source=(ROOT/'source/affine41.py').read_text();tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_affine')
    require(hashlib.sha256((ast.get_source_segment(source,function)+'\n').encode()).hexdigest()==inherited['affine_function_source_sha256'],
            'Inherited affine fitting function changed')
    for relative,wanted in inherited['byte_identical_files'].items():
        require(sha(checked_path(relative))==wanted,'Inherited native source changed')
        required.add(relative)
    for task in TASKS:
        path=f'inputs/data41/{task}/public_source41.json';source=read(ROOT/path);required.add(path)
        required.update(r['archive_path'] for r in source['archives']);required.add(source['metadata_path'])
        required.update(source.get('supplementary_metadata_files',{}))
        required.update({f'inputs/data41/{task}/calendar_projection41.json',f'inputs/data41/{task}/calendar_metadata41.json'})
    require(required<=set(seal['bound_files']),'Scientific freeze omits required dependencies')
    protocol=read(ROOT/'protocol41.json');audit_protocol(protocol)
    return protocol,seal


def verify_gate(relative,**bindings):
    saved=read(checked_path(relative))
    require(saved['status']=='PASS' and saved['protocol_seal_sha256']==sha(ROOT/'protocol_seal41.json') and
            all(saved.get(k)==v for k,v in bindings.items()),'Required independent gate missing/mismatched: '+relative)
    return saved


def publish_gate(relative,kind,**values):
    path=checked_path(relative);require(not path.exists(),'Independent gate already exists; no rerun: '+relative)
    record={'schema':'step41-independent-'+kind+'-gate-v1','status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
            'protocol_seal_sha256':sha(ROOT/'protocol_seal41.json'),'artificial_only':artificial_scope(),**values}
    write(path,record)
    append_event('independent_'+kind+'_PASS',gate_path=relative,gate_sha256=sha(path),task=values.get('task'),stage=values.get('stage'))
    return record


def preflight():
    protocol,seal=verify_protocol_seal()
    events=audit_events();require(len(events)==1 and events[0]['event']=='protocol_frozen','Unexpected source access before preexecution audit')
    require(events[0]['protocol_seal_sha256']==sha(ROOT/'protocol_seal41.json'),'Protocol freeze event differs')
    metadata=audit_metadata()
    return publish_gate('audit/preexecution_gate41.json','preexecution',metadata=metadata,
      frozen_dependency_count=len(seal['bound_files']),measurement_values_decoded=0)


def fit_gate(task,stage):
    protocol,_=verify_protocol_seal();require(task in TASKS and stage in STAGES,'Unknown fit audit task/stage')
    verify_gate('audit/preexecution_gate41.json')
    relative=f'output/{task}/{stage}/fit_seal41.json'
    seal=verify_seal(relative,'step41-fit-seal-v1',task=task,stage=stage,protocol_seal_sha256=sha(ROOT/'protocol_seal41.json'))
    base=f'output/{task}/{stage}'
    require({base+'/fit_manifest41.json',base+'/normalization41.json',base+'/cohorts/train.json',base+'/cohorts/train_X.npy'}<=set(seal['bound_files']),
            'Fit seal omits required training/model dependencies')
    require(seal['manifest_sha256']==sha(ROOT/(base+'/fit_manifest41.json')),'Fit manifest seal changed')
    start,_=event_once('phase_started',phase='fit',task=task,stage=stage);done,_=event_once('phase_completed',phase='fit',task=task,stage=stage)
    access,_=event_once('train_measurement_access_started',task=task,stage=stage)
    require(start<access<done,'Fitting preceded/followed wrong data-access phase')
    if stage=='f1':prior,_=event_once('independent_preexecution_PASS')
    elif stage=='final':prior,_=event_once('independent_selection_PASS',task=task)
    else:prior,_=event_once('independent_validation_PASS',task=task,stage='f1' if stage=='f2' else 'f2')
    require(prior<access,'Training accessed values before prior independent gate')
    training=audit_cohort(task,stage,'train');context=load_context(task,stage)
    check=audit_stage_models(protocol,task,stage,training,context)
    expected_paths={r[k] for r in context['manifest']['candidate_rows'] for k in ('head_path','fit_path','deployment_path')}
    expected_paths.update(r['payload_path'] for r in context['manifest']['baseline_rows'])
    expected_paths.update(r['map_path'] for r in context['manifest']['maps'].values())
    expected_paths.update(r['training_features']['path'] for r in context['manifest']['maps'].values())
    require(expected_paths<=set(seal['bound_files']),'Fit seal omits deployed/trainfeature payloads')
    return publish_gate(f'audit/{task}/{stage}/fit_gate41.json','fit',task=task,stage=stage,
                        fit_seal_sha256=sha(ROOT/relative),models=check)


def validation_gate(task,stage):
    protocol,_=verify_protocol_seal();require(stage in ('f1','f2','f3'),'Final stage has no validation')
    base=f'output/{task}/{stage}';fitpath=base+'/fit_seal41.json';completion=base+'/completion_seal41.json'
    verify_gate(f'audit/{task}/{stage}/fit_gate41.json',task=task,stage=stage,fit_seal_sha256=sha(ROOT/fitpath))
    seal=verify_seal(completion,'step41-fold-completion-seal-v1',task=task,stage=stage,
        protocol_seal_sha256=sha(ROOT/'protocol_seal41.json'),fit_seal_sha256=sha(ROOT/fitpath))
    required={base+'/validation_predictions41.json',base+'/validation_scores41.json',base+'/cohorts/validation.json',base+'/cohorts/validation_X.npy'}
    require(required<=set(seal['bound_files']),'Completed fold seal omits prediction/score evidence')
    previous,_=event_once('independent_fit_PASS',task=task,stage=stage)
    start,_=event_once('validation_measurement_access_started',task=task,stage=stage)
    require(previous<start,'Validation access preceded independent fit check')
    cohort=audit_cohort(task,stage,'validation');context=load_context(task,stage)
    bundle=read(ROOT/(base+'/validation_predictions41.json'))
    predictions=audit_prediction_bundle(protocol,task,stage,bundle,context,cohort['X'],cohort['ids'])
    scores=read(ROOT/(base+'/validation_scores41.json'));original=context['manifest']['candidate_rows']+context['manifest']['baseline_rows']
    require(scores['schema']=='step41-validation-scores-v1' and scores['task']==task and scores['stage']==stage and
            scores['n']==len(cohort['ids']) and scores['cohort_sha256']==cohort['sha256'] and
            scores['validation_predictions_sha256']==sha(ROOT/(base+'/validation_predictions41.json')) and
            len(scores['candidate_rows'])==len(original),'Validation score provenance differs')
    for row,saved in zip(original,scores['candidate_rows']):
        require({k:saved[k] for k in row}==row,'Validation candidate identity differs')
        pr=bundle['predictions'][row['candidate_id']]
        audit_risk(saved['physical_risk'],[F.from_float(float_hex(v['reported_physical_hex'])) for v in pr],cohort['raw_targets'])
        if row['recipe']=='affine':audit_risk(saved['exact_reference_risk'],[q(v['exact_physical']) for v in pr],cohort['raw_targets'])
        else:require(saved['exact_reference_risk'] is None,'Baseline exact-affine reference invented')
    return publish_gate(f'audit/{task}/{stage}/validation_gate41.json','validation',task=task,stage=stage,
                        completion_seal_sha256=sha(ROOT/completion),predictions=predictions,score_rows=len(original))


def selection_gate(task):
    protocol,_=verify_protocol_seal();folds={}
    for stage in ('f1','f2','f3'):
        verify_gate(f'audit/{task}/{stage}/validation_gate41.json',task=task,stage=stage,
                    completion_seal_sha256=sha(ROOT/f'output/{task}/{stage}/completion_seal41.json'))
        folds[stage]=read(ROOT/f'output/{task}/{stage}/validation_scores41.json')
    n=sum(f['n'] for f in folds.values());path=f'output/{task}/selection41.json';saved=read(ROOT/path)
    seal=verify_seal(f'output/{task}/selection_seal41.json','step41-selection-seal-v1',task=task,
        protocol_seal_sha256=sha(ROOT/'protocol_seal41.json'),selection_sha256=sha(ROOT/path))
    require(path in seal['bound_files'] and saved['schema']=='step41-development-selection-v1' and
            saved['task']==task and saved['development_n']==n and saved['selection_uses_development_only'] is True and
            saved['test_targets_accessed']==0 and saved['fold_score_sha256']=={
              stage:sha(ROOT/f'output/{task}/{stage}/validation_scores41.json') for stage in folds},
            'Development selection ancestry/scope differs')
    require([r['candidate_id'] for r in saved['candidate_rows']]==protocol['evaluation']['prediction_ids'],
            'Pooled selection roster differs')
    for row in saved['candidate_rows']:
        cid=row['candidate_id'];source={stage:next(r for r in f['candidate_rows'] if r['candidate_id']==cid)
                                     for stage,f in folds.items()};last=source['f3']
        identity={k:last.get(k) for k in ('candidate_id','map','token','recipe','family','scaling','baseline_kind')}
        require({k:row[k] for k in identity}==identity,'Selection candidate identity differs')
        expected_folds={stage:{'n':folds[stage]['n'],'physical_risk':r['physical_risk'],
                      'exact_reference_risk':r['exact_reference_risk'],'payload_bytes':r['deployment_payload_bytes']}
                        for stage,r in source.items()}
        mse=sum((q(r['physical_risk']['sse']) for r in source.values()),F(0))/n
        exact=sum((q(r['exact_reference_risk']['sse']) for r in source.values()),F(0))/n if row['recipe']=='affine' else None
        require(row['folds']==expected_folds and q(row['pooled_physical_mse'])==mse and
                row['pooled_exact_reference_mse']==(str(exact) if exact is not None else None) and
                row['selection_payload_bytes']==last['deployment_payload_bytes'], 'Physical SSE pooling or tie payload differs')
    require(set(saved['selections'])==set(SELECTORS),'Selector roster differs')
    for label in SELECTORS:
        expected=audit_pooled_selection(saved['candidate_rows'],label,selector_ids(saved['candidate_rows'],label))
        expected.pop('selector')
        require(saved['selections'][label]==expected,'Exact pooled selection or deterministic tie-break differs: '+label)
    selected={k:r['candidate_id'] for k,r in saved['selections'].items()}
    require(seal['selected_ids']==selected,'Selection seal does not bind selected IDs')
    events=audit_events()
    require(not any(e['event']=='test_history_access_started' and e.get('task')==task for e in events),
            'Development selection occurred after test input access')
    return publish_gate(f'audit/{task}/selection_gate41.json','selection',task=task,
                        selection_seal_sha256=sha(ROOT/f'output/{task}/selection_seal41.json'),
                        selected_ids=selected,development_n=n,candidate_count=62)


def pretest(task):
    protocol,_=verify_protocol_seal()
    verify_gate(f'audit/{task}/selection_gate41.json',task=task,
                selection_seal_sha256=sha(ROOT/f'output/{task}/selection_seal41.json'))
    verify_gate(f'audit/{task}/final/fit_gate41.json',task=task,stage='final',
                fit_seal_sha256=sha(ROOT/f'output/{task}/final/fit_seal41.json'))
    selection=read(ROOT/f'output/{task}/selection41.json');manifest=read(ROOT/f'output/{task}/final/fit_manifest41.json')
    allids=[r['candidate_id'] for r in manifest['candidate_rows']+manifest['baseline_rows']]
    require(allids==protocol['evaluation']['prediction_ids'] and
            all(r['candidate_id'] in allids for r in selection['selections'].values()),'Final refit candidate identities differ')
    require(not any(e['event'].startswith('test_') and e.get('task')==task for e in audit_events()),
            'Current task test accessed before independent pretest gate')
    return publish_gate(f'audit/{task}/pretest_gate41.json','pretest',task=task,
        fit_seal_sha256=sha(ROOT/f'output/{task}/final/fit_seal41.json'),
        selection_seal_sha256=sha(ROOT/f'output/{task}/selection_seal41.json'),test_targets_accessed=0,
        prediction_count=62,selected_ids={k:r['candidate_id'] for k,r in selection['selections'].items()})


def audit_test_query(protocol,task,index,query,context,prior_target_event):
    import numpy as np
    base=f'output/{task}/test';history_path=f'{base}/history/{index:03d}.json';target_path=f'{base}/targets/{index:03d}.json'
    prediction_path=f'{base}/predictions/q{index:03d}.json';seal_path=f'{base}/prediction_seals/{index:03d}.json'
    history=read(ROOT/history_path);target=read(ROOT/target_path);prediction=read(ROOT/prediction_path);seal=read(ROOT/seal_path)
    qid=query['id'];lags=lag_stamps(query);modelsha=sha(ROOT/f'output/{task}/final/fit_seal41.json')
    names=('test_history_access_started','test_history_access_completed','test_predictions_sealed',
           'test_target_access_started','test_target_revealed')
    records=[event_once(name,task=task,query_id=qid,query_index=index) for name in names]
    order=[r[0] for r in records]
    require(prior_target_event<order[0] and order==sorted(order) and len(set(order))==5,
            'Current target was not sealed after causal history and before next query')
    hs,hc,ps,ts,tr=[r[1] for r in records]
    require(hc['history_receipt_path']==history_path and hc['history_receipt_sha256']==sha(ROOT/history_path) and
            ps['prediction_path']==prediction_path and ps['prediction_sha256']==sha(ROOT/prediction_path) and
            ps['prediction_seal_path']==seal_path and ps['prediction_seal_sha256']==sha(ROOT/seal_path) and
            ts['prediction_seal_sha256']==sha(ROOT/seal_path) and tr['target_path']==target_path and
            tr['target_sha256']==sha(ROOT/target_path) and tr['prediction_seal_sha256']==sha(ROOT/seal_path),
            'Chronological prediction/target event artifact binding differs')
    require(history['schema']=='step41-causal-test-history-v1' and history['task']==task and history['query_id']==qid and
            history['query_index']==index and history['target_timestamp']==query['target_timestamp'] and
            history['origin_timestamp']==query['origin_timestamp'] and history['history_timestamps']==lags and
            history['model_seal_sha256']==modelsha and history['current_target_value_decoded'] is False and
            history['other_measurement_predictors_used'] is False,'Test history identity/information boundary differs')
    audit_access(history['access'],task,'final','test_history',lags,query['origin_timestamp'])
    # These raw values are decoded only after the producer's unique target-reveal
    # event, already checked above. No post-run audit can reveal a future label.
    values=authorized_values(task,lags+[query['target_timestamp']]);normal=context['manifest']['normalization']
    low,span=q(normal['target_min']),q(normal['span']);rawtarget=values[query['target_timestamp']]
    rawhistory=[values[s] for s in lags]
    expectedX=np.asarray([[float(v) for v in rawhistory]],dtype=np.float64)
    X=np.load(bound_reference({'path':history['X_path'],'sha256':history['X_sha256']}),allow_pickle=False)
    require(X.dtype==np.float64 and X.shape==(1,24) and np.array_equal(X.view(np.uint64),expectedX.view(np.uint64)) and
            list(map(q,history['history_values']))==rawhistory and q(history['raw_persistence'])==rawhistory[-1] and
            q(history['persistence'])==(rawhistory[-1]-low)/span and
            history['normalization_sha256']==sha(ROOT/f'output/{task}/final/normalization41.json'), 'Causal physical input/scale differs')
    prior_queries=frozen_queries(task,'final','test')[:index]
    overlap=[stamp for stamp in lags if stamp in [r['target_timestamp'] for r in prior_queries]]
    require(history['previously_revealed_test_targets_in_history']==overlap and hc['previously_revealed_test_targets_in_history']==overlap,
            'Previously revealed target/history overlap differs')
    require(seal['schema']=='step41-test-prediction-byte-seal-v1' and seal['task']==task and seal['query_id']==qid and
            seal['query_index']==index and seal['model_seal_sha256']==modelsha and seal['prediction_path']==prediction_path and
            seal['prediction_sha256']==sha(ROOT/prediction_path) and seal['history_receipt_path']==history_path and
            seal['history_receipt_sha256']==sha(ROOT/history_path) and seal['prediction_ids']==protocol['evaluation']['prediction_ids'] and
            seal['prediction_count']==62 and seal['current_target_value_decoded'] is False,'Per-query byte seal omits/binds wrong predictions')
    require(prediction['schema']=='step41-sealed-test-predictions-v1' and prediction['query_id']==qid and
            prediction['query_index']==index and prediction['model_seal_sha256']==modelsha and
            prediction['history_receipt_sha256']==sha(ROOT/history_path) and
            prediction['selection_seal_sha256']==sha(ROOT/f'output/{task}/selection_seal41.json'), 'Test bundle seal ancestry differs')
    verified=audit_prediction_bundle(protocol,task,'final',prediction,context,X,[qid])
    require(target['schema']=='step41-revealed-test-target-v1' and target['task']==task and target['query_id']==qid and
            target['query_index']==index and target['target_timestamp']==query['target_timestamp'] and
            target['origin_timestamp']==query['origin_timestamp'] and q(target['raw_target'])==rawtarget and
            q(target['y'])==(rawtarget-low)/span and target['normalization_sha256']==history['normalization_sha256'] and
            target['prediction_seal_path']==seal_path and target['prediction_seal_sha256']==sha(ROOT/seal_path) and
            target['model_seal_sha256']==modelsha and target['target_value_clipped'] is False, 'Current revealed target identity/value differs')
    audit_access(target['access'],task,'final','test_target',[query['target_timestamp']],query['target_timestamp'])
    return {'predictions':prediction['predictions'],'raw_target':rawtarget,'last_event':order[-1],
            'prediction_path':prediction_path,'prediction_sha256':sha(ROOT/prediction_path),
            'target_path':target_path,'target_sha256':sha(ROOT/target_path),'verified':verified}


def postrun(task):
    protocol,_=verify_protocol_seal();verify_gate(f'audit/{task}/pretest_gate41.json',task=task)
    completion=f'output/{task}/test/completion_seal41.json'
    seal=verify_seal(completion,'step41-test-completion-seal-v1',task=task,
                    protocol_seal_sha256=sha(ROOT/'protocol_seal41.json'))
    result_path=f'output/{task}/primary_result41.json';result=read(ROOT/result_path)
    require(result_path in seal['bound_files'] and seal['primary_result_sha256']==sha(ROOT/result_path),
            'Test completion does not bind primary result')
    queries=frozen_queries(task,'final','test');context=load_context(task,'final')
    previous,_=event_once('independent_pretest_PASS',task=task)
    collected={cid:[] for cid in protocol['evaluation']['prediction_ids']};targets=[];refs=[];norm_errors=[]
    for i,query in enumerate(queries):
        checked=audit_test_query(protocol,task,i,query,context,previous);previous=checked['last_event']
        targets.append(checked['raw_target'])
        for cid in collected:collected[cid].extend(checked['predictions'][cid])
        refs.append({'id':query['id'],'month':query['target_timestamp'][:7],
          **{k:checked[k] for k in ('prediction_path','prediction_sha256','target_path','target_sha256')}})
        norm_errors.append(q(checked['verified']['maximum_normalized_error']))
    state=read(ROOT/f'output/{task}/test/access_state41.json')
    require(state['status']=='COMPLETE' and state['next_index']==len(queries) and state['pending_query'] is None and
            state['previously_revealed_test_targets']==[r['target_timestamp'] for r in queries], 'Test access state is not complete prefix')
    require(result['schema']=='step41-primary-result-v1' and result['task']==task and result['n']==len(queries) and
            result['query_ids']==[r['id'] for r in queries] and list(map(q,result['raw_targets']))==targets and
            result['queries']==refs and result['selection_sha256']==sha(ROOT/f'output/{task}/selection41.json') and
            result['validity_audit_pending'] is True, 'Final primary result source roster differs')
    original=context['manifest']['candidate_rows']+context['manifest']['baseline_rows']
    require(len(result['candidate_rows'])==62,'Some assessed candidates were omitted')
    byid={}
    for row,saved in zip(original,result['candidate_rows']):
        cid=row['candidate_id'];pr=collected[cid]
        require({k:saved[k] for k in row}==row and saved['test_predictions_digest']==digest(pr), 'Assessed candidate/prediction identity differs')
        audit_risk(saved['physical_test_risk'],[F.from_float(float_hex(r['reported_physical_hex'])) for r in pr],targets)
        if row['recipe']=='affine':audit_risk(saved['exact_reference_test_risk'],[q(r['exact_physical']) for r in pr],targets)
        else:require(saved['exact_reference_test_risk'] is None,'Baseline mislabeled exact-affine reference')
        byid[cid]=saved
    selection=read(ROOT/f'output/{task}/selection41.json')
    selected={label:byid[row['candidate_id']] for label,row in selection['selections'].items()}
    require(result['selections']==selected,'Test changed development-selected configurations')
    expected=primary_fields(selected['affine_rolling']['physical_test_risk']['mse'],selected['ordinary_rolling']['physical_test_risk']['mse'])
    require(result['primary_gate']==expected,'Frozen primary finite-period comparison differs')
    months=sorted({r['month'] for r in refs});require(sorted(result['monthly_results'])==months,'Descriptive month roster differs')
    for month in months:
        indices=[i for i,r in enumerate(refs) if r['month']==month]
        require(set(result['monthly_results'][month])==set(SELECTORS),'Monthly selectors changed')
        for label,row in selected.items():
            vals=[F.from_float(float_hex(collected[row['candidate_id']][i]['reported_physical_hex'])) for i in indices]
            audit_risk(result['monthly_results'][month][label],vals,[targets[i] for i in indices])
    return publish_gate(f'audit/{task}/postrun_gate41.json','postrun',task=task,
        completion_seal_sha256=sha(ROOT/completion),primary_result_sha256=sha(ROOT/result_path),
        test_n=len(queries),affine_certificates=18*len(queries),ordinary_predictions=44*len(queries),
        maximum_normalized_prediction_error=str(max(norm_errors)),primary_gate=expected,
        empirical_criterion_pass=expected['within_five_percent_mse'])


def numeric_summary_equal(got,wanted,label):
    import math
    require(type(got) is dict and set(got)==set(wanted),'Summary fields differ: '+label)
    for key,value in wanted.items():
        require(isinstance(got[key],(int,float)) and isfinite(got[key]) and
                math.isclose(float(got[key]),float(value),rel_tol=2e-14,abs_tol=1e-15),
                'Derived timing summary differs: '+label+'/'+key)


def linear_quantile(values,probability):
    values=sorted(values);position=(len(values)-1)*probability;left=int(position);weight=position-left
    return values[left] if left==len(values)-1 else values[left]+weight*(values[left+1]-values[left])


def benchmark_gate(task):
    import statistics
    import numpy as np
    protocol,_=verify_protocol_seal();post=verify_gate(f'audit/{task}/postrun_gate41.json',task=task)
    path=f'output/{task}/benchmark41.json';record=read(ROOT/path)
    result=read(ROOT/f'output/{task}/primary_result41.json');selected=result['selections']['affine_rolling']
    deployment=read(bound_reference({'path':selected['deployment_path'],'sha256':selected['deployment_sha256']}))
    config=protocol['timing'];names=config['primitives'];n=result['n']
    require(record['schema']=='step41-deployment-benchmark-v1' and record['task']==task and
            record['candidate_id']==selected['candidate_id'] and record['deployment_sha256']==selected['deployment_sha256'] and
            record['map_sha256']==deployment['map_sha256'] and record['configuration']==config and
            record['n']==n==len(record['records']) and record['postrun_gate_sha256']==sha(ROOT/f'audit/{task}/postrun_gate41.json'),
            'Deployment benchmark identity/protocol differs')
    previous,_=event_once('independent_postrun_PASS',task=task);start,_=event_once('phase_started',phase='benchmark',task=task,stage=None)
    done,_=event_once('phase_completed',phase='benchmark',task=task,stage=None)
    require(previous<start<done,'Deployment benchmark preceded completed assessment audit')
    events=audit_events()
    require(not any('measurement_access' in e['event'] or 'test_target_access' in e['event'] or 'test_history_access' in e['event']
                    for e in events[start:done+1]), 'Deployment benchmark decoded extra outcomes/history')
    static=record['static_prepare_nanoseconds']
    require(len(static)==config['static_prepare_repeats'] and all(type(v) is int and v>=0 for v in static) and
            record['static_prepare_median_ms']==statistics.median(static)/1e6,'Static parse/validation timing summary differs')
    require(record['map_bytes']==selected['map_payload_bytes']==deployment['map_bytes'] and
            record['static_deployment_bytes']==selected['static_deployment_bytes']==checked_path(selected['deployment_path']).stat().st_size and
            record['deployment_payload_bytes']==record['map_bytes']+record['static_deployment_bytes'], 'Deployed model bytes omit static evidence')
    medians={name:[] for name in names};sizes={key:[] for key in ('certificate_json_bytes','feature_bytes','certificate_plus_feature_bytes')};errors=[]
    for index,(saved,query) in enumerate(zip(record['records'],result['queries'])):
        require(saved['query_id']==query['id'] and saved['prediction_sha256']==query['prediction_sha256'] and
                set(saved['nanoseconds'])==set(saved['median_nanoseconds'])==set(names), 'Timing query/primitive roster differs')
        for name in names:
            samples=saved['nanoseconds'][name]
            require(len(samples)==config['repeats_per_query_primitive'] and all(type(v) is int and v>=0 for v in samples) and
                    saved['median_nanoseconds'][name]==float(statistics.median(samples)), 'Per-query timing median/sample count differs')
            medians[name].append(saved['median_nanoseconds'][name])
        bundle=read(bound_reference({'path':query['prediction_path'],'sha256':query['prediction_sha256']}))
        prediction=bundle['predictions'][selected['candidate_id']][0];cert=prediction['certificate']
        z=np.load(bound_reference(bundle['maps'][selected['map']]['features']),allow_pickle=False)[0]
        audit_certificate(deployment,z,cert,query_id=query['id'],reported_prediction=float_hex(prediction['reported_physical_hex']),check_static=False)
        certbytes=len(identity_bytes(cert));featurebytes=len(feature_packet(z))
        expected={'certificate_json_bytes':certbytes,'feature_bytes':featurebytes,'certificate_plus_feature_bytes':certbytes+featurebytes}
        require(all(saved[k]==v for k,v in expected.items()), 'Per-prediction certificate/required query byte accounting differs')
        for key,value in expected.items():sizes[key].append(value)
        error=abs(F.from_float(float_hex(prediction['reported_physical_hex']))-q(prediction['exact_physical']))
        require(q(saved['absolute_physical_prediction_error'])==error,'Timing record differs from assessed prediction/error')
        errors.append(error)
    require(set(record['component_summary'])==set(names),'Timing summary primitive roster differs')
    for name,values in medians.items():
        expected={'p50_ms':linear_quantile(values,.5)/1e6,'p95_ms':linear_quantile(values,.95)/1e6,
                  'min_ms':min(values)/1e6,'max_ms':max(values)/1e6}
        numeric_summary_equal(record['component_summary'][name],expected,name)
    require(set(record['packet_summary'])==set(sizes),'Packet summary roster differs')
    for key,values in sizes.items():
        expected={'min':min(values),'median':float(statistics.median(values)),'max':max(values)}
        require(record['packet_summary'][key]==expected,'Packet size summary differs')
    budget=q(deployment['span'])*F(1,1<<40)
    require(q(record['maximum_physical_error'])==max(errors) and q(record['physical_error_budget'])==budget and
            max(errors)<=budget,'Maximum deployed-prediction discrepancy/budget differs')
    return publish_gate(f'audit/{task}/benchmark_gate41.json','benchmark',task=task,benchmark_sha256=sha(ROOT/path),
         primary_result_sha256=sha(ROOT/f'output/{task}/primary_result41.json'),queries=n,primitives=len(names),
         timed_query_samples=n*len(names)*config['repeats_per_query_primitive'],static_timed_samples=len(static),
         no_new_data_access=True,timings_independently_repeated=False,
         timing_audit_scope='Source-bound method, raw measured samples, identity and summary arithmetic; no claim to externally attest hardware timing')


def aggregate():
    protocol,seal=verify_protocol_seal();record=read(ROOT/'output/result41.json')
    require(record['schema']=='step41-complete-result-v1' and record['protocol_seal_sha256']==sha(ROOT/'protocol_seal41.json') and
            record['step40_primary_unchanged']=='NOT_ESTABLISHED' and set(record['tasks'])==set(TASKS) and
            set(record['benchmarks'])==set(TASKS),'Combined result scope/identity differs')
    scientific_gates={'audit/preexecution_gate41.json':sha(ROOT/'audit/preexecution_gate41.json')};finalgates={}
    for task in TASKS:
        resultpath=f'output/{task}/primary_result41.json';benchmarkpath=f'output/{task}/benchmark41.json'
        require(record['tasks'][task]==read(ROOT/resultpath) and record['benchmarks'][task]==read(ROOT/benchmarkpath),
                'Combined task/benchmark differs from independently audited file')
        for stage in STAGES:
            rel=f'audit/{task}/{stage}/fit_gate41.json';verify_gate(rel,task=task,stage=stage,fit_seal_sha256=sha(ROOT/f'output/{task}/{stage}/fit_seal41.json'))
            scientific_gates[rel]=sha(ROOT/rel)
            verify_seal(f'output/{task}/{stage}/fit_seal41.json','step41-fit-seal-v1',task=task,stage=stage,
                        protocol_seal_sha256=sha(ROOT/'protocol_seal41.json'))
            if stage!='final':
                rel=f'audit/{task}/{stage}/validation_gate41.json';verify_gate(rel,task=task,stage=stage,completion_seal_sha256=sha(ROOT/f'output/{task}/{stage}/completion_seal41.json'))
                scientific_gates[rel]=sha(ROOT/rel)
                verify_seal(f'output/{task}/{stage}/completion_seal41.json','step41-fold-completion-seal-v1',task=task,stage=stage,
                            protocol_seal_sha256=sha(ROOT/'protocol_seal41.json'))
        for label in ('selection','pretest','postrun','benchmark'):
            rel=f'audit/{task}/{label}_gate41.json';verify_gate(rel,task=task);scientific_gates[rel]=sha(ROOT/rel)
            if label in ('postrun','benchmark'):finalgates[rel]=sha(ROOT/rel)
        require(read(ROOT/f'audit/{task}/postrun_gate41.json')['primary_result_sha256']==sha(ROOT/resultpath) and
                read(ROOT/f'audit/{task}/benchmark_gate41.json')['benchmark_sha256']==sha(ROOT/benchmarkpath),
                'Postrun/benchmark audit source binding changed')
        verify_seal(f'output/{task}/test/completion_seal41.json','step41-test-completion-seal-v1',task=task,
                    protocol_seal_sha256=sha(ROOT/'protocol_seal41.json'))
    require(record['independent_gates']==finalgates,'Combined result final gate bindings differ')
    expected='ESTABLISHED_ON_FIXED_PERIODS' if all(v['primary_gate']['within_five_percent_mse'] for v in record['tasks'].values()) else 'NOT_ESTABLISHED'
    require(record['primary_status']==expected,'Combined primary outcome differs')
    events=audit_events();bindings=0
    expected_phases={(phase,task,stage) for task in TASKS for stage in STAGES for phase in ('fit',)}
    expected_phases|={(phase,task,stage) for task in TASKS for stage in ('f1','f2','f3') for phase in ('validate',)}
    expected_phases|={(phase,task,None) for task in TASKS for phase in ('select','test','benchmark')}
    expected_phases.add(('aggregate',None,None))
    for name in ('phase_started','phase_completed'):
        actual=[(e['phase'],e.get('task'),e.get('stage')) for e in events if e['event']==name]
        require(len(actual)==len(expected_phases) and set(actual)==expected_phases,'Scientific producer phase omission/repetition')
    require(not any('failed' in e['event'].lower() or e.get('status')=='STOP_RETAIN_FAILURE' for e in events),
            'Failed execution retained; overall PASS unavailable')
    pairs=(('receipt_path','receipt_sha256'),('gate_path','gate_sha256'),('path','sha256'),
           ('cohort_path','cohort_sha256'),('access_path','access_sha256'),('history_receipt_path','history_receipt_sha256'),
           ('prediction_seal_path','prediction_seal_sha256'),('prediction_path','prediction_sha256'),('target_path','target_sha256'))
    for event in events:
        for pathkey,hashkey in pairs:
            if pathkey in event and hashkey in event:
                require(sha(checked_path(event[pathkey]))==event[hashkey], 'Event-bound artifact changed')
                bindings+=1
    result={'schema':'step41-independent-global-audit-v1','status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
            'protocol_seal_sha256':sha(ROOT/'protocol_seal41.json'),'result_sha256':sha(ROOT/'output/result41.json'),
            'primary_status':expected,'tasks':list(TASKS),'scientific_gate_count_including_global':len(scientific_gates)+1,
            'scientific_gates':scientific_gates,'producer_phase_count':len(expected_phases),'event_count_before_global_audit':len(events),
            'event_artifact_bindings':bindings,'protocol_bound_file_count':len(seal['bound_files']),
            'new_scientific_fits_by_auditor':0,'artificial_only':artificial_scope(),
            'mathematical_scope':'Exact fixed affine heads on stored binary64 features; exact operational-prediction losses; fixed-period comparisons',
            'empirical_success_distinct_from_verification':True,'timing_summary_audited_hardware_timing_not_externally_attested':True}
    path=ROOT/'audit/postrun_audit41.json';require(not path.exists(),'Global audit already exists; no rerun')
    write(path,result);append_event('independent_global_PASS',gate_path='audit/postrun_audit41.json',gate_sha256=sha(path),task=None,stage=None)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('preflight','fit','validate','selection','pretest','postrun','benchmark','aggregate'))
    parser.add_argument('--task');parser.add_argument('--stage');args=parser.parse_args()
    try:
        if args.operation=='preflight':result=preflight()
        elif args.operation in ('fit','validate'):result={'fit':fit_gate,'validate':validation_gate}[args.operation](args.task,args.stage)
        elif args.operation=='aggregate':result=aggregate()
        else:result={'selection':selection_gate,'pretest':pretest,'postrun':postrun,'benchmark':benchmark_gate}[args.operation](args.task)
        print(json.dumps({k:result[k] for k in ('schema','status','task','stage','primary_status','result_sha256') if k in result},sort_keys=True),flush=True)
    except Exception as error:
        failure={'schema':'step41-independent-audit-failure-v1','status':'FAIL','operation':args.operation,'task':args.task,
                 'stage':args.stage,'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        path=ROOT/'audit/failures'/f'{args.operation}_{args.task}_{args.stage}.json'
        if not path.exists():write(path,failure)
        print(json.dumps(failure,sort_keys=True),flush=True);raise SystemExit(2)


if __name__=='__main__':main()
