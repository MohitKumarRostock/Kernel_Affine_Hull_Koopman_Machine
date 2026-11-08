"""Opaque-source and calendar-only preparation for the frozen Step 43 design.

Valid measurement cells are never converted to text or numbers here. Only
header/timestamp fields and the declared Household missing-status markers are
interpreted. Numeric access belongs to data43.py after the final scientific gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile

from common43 import ROOT, TASKS, STAGES, canonical_bytes, digest, file_ref, read_json, sha_file, utc_now, write_json

SOURCE_SPECS = {
    "household_power2008": {
        "dataset": "Individual Household Electric Power Consumption",
        "provider": "UCI Machine Learning Repository", "uci_id": 235,
        "doi": "10.24432/C58K54", "license": "CC BY 4.0",
        "metadata_url": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
        "separator": ";", "header_encoding": "utf-8", "timestamp_indices": [0, 1],
        "timestamp_kind": "household_date_time", "timestamp_formats": ["%d/%m/%Y %H:%M:%S"],
        "target_index": 2, "target_header": "Global_active_power",
        "timestamp_headers": ["Date", "Time"], "unit": "kW", "cadence_minutes": 1,
        "numeric_domain": "finite_nonnegative_decimal", "missing_markers": ["", "?"],
        "snapshot": "Recorded minute-averaged active power at exact minute00; not an hourly average",
        "timezone": "Recorded local-naive Date and Time; no inferred UTC offset or daylight-saving reconstruction",
        "transport": "retained_pinned_public_mirror",
        "transport_limitation": "Byte identity with the unreachable official archive was not independently established in the prior study; preserve the pinned mirror identity and this limitation",
        "archives": [{
            "source_id": "household", "filename": "household_power_consumption_mirror.zip",
            "member": "household_power_consumption.txt", "nested_gzip": False,
            "source_url": "https://raw.githubusercontent.com/jbrownlee/Datasets/509690881d2c47c89d3d221ed2e2bb30e5afa9c0/household_power_consumption.zip",
            "prior_path": "step41_certified_affine/inputs/data41/household_power2010/household_power_consumption_mirror.zip",
            "expected_sha256": "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff",
        }],
        "prior_source_path": "step41_certified_affine/inputs/data41/household_power2010/public_source41.json",
        "prior_metadata_path": "step41_certified_affine/inputs/data41/household_power2010/official_metadata_retrieval40.json",
    },
    "jena_temperature2018": {
        "dataset": "MPI-BGC Beutenberg roof weather station",
        "provider": "Max Planck Institute for Biogeochemistry", "uci_id": None,
        "doi": None, "license": "CC BY 4.0",
        "metadata_url": "https://www.bgc-jena.mpg.de/wetter/weather_data.html",
        "documentation_url": "https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf",
        "separator": ",", "header_encoding": "latin-1", "timestamp_indices": [0],
        "timestamp_kind": "jena_date_time", "timestamp_formats": ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"],
        "target_index": 2, "target_header": "T (degC)", "timestamp_headers": ["Date Time"],
        "unit": "degree Celsius", "cadence_minutes": 10,
        "numeric_domain": "finite_decimal_at_least_minus273_15", "missing_markers": [],
        "snapshot": "Recorded 10-minute-averaged air temperature at exact minute00; interval-end timestamp, not an hourly average",
        "timezone": "Source documentation specifies MEZ/CET; retain recorded naive interval-end times without daylight-saving reconstruction",
        "transport": "official_provider",
        "transport_limitation": None,
        "archives": [
            {"source_id": "jena2018a", "filename": "mpi_roof_2018a.zip",
             "member": "mpi_roof_2018a.csv", "nested_gzip": False,
             "source_url": "https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2018a.zip",
             "metadata_link": "data/mpi_roof_2018a.zip"},
            {"source_id": "jena2018b", "filename": "mpi_roof_2018b.zip",
             "member": "mpi_roof_2018b.csv", "nested_gzip": False,
             "source_url": "https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2018b.zip",
             "metadata_link": "data/mpi_roof_2018b.zip"},
        ],
    },
    "metro_traffic2018": {
        "dataset": "Metro Interstate Traffic Volume",
        "provider": "UCI Machine Learning Repository", "uci_id": 492,
        "doi": "10.24432/C5X60B", "license": "CC BY 4.0",
        "metadata_url": "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        "separator": ",", "header_encoding": "utf-8", "timestamp_indices": [7],
        "timestamp_kind": "iso_local", "timestamp_formats": ["%Y-%m-%d %H:%M:%S"],
        "target_index": 8, "target_header": "traffic_volume", "timestamp_headers": ["date_time"],
        "unit": "vehicles per hour", "cadence_minutes": 60,
        "numeric_domain": "finite_nonnegative_integer", "missing_markers": [],
        "snapshot": "Recorded hourly traffic volume at exact minute00; no aggregation or weather predictors",
        "timezone": "Official source describes local CST; retain recorded local-naive timestamps without UTC or daylight-saving reinterpretation",
        "transport": "retained_official_provider_archive",
        "transport_limitation": None,
        "archives": [{
            "source_id": "metro", "filename": "source_uci492.zip",
            "member": "Metro_Interstate_Traffic_Volume.csv.gz", "nested_gzip": True,
            "source_url": "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
            "prior_path": "step39_temporal_selection/inputs/metro_traffic/source_uci492.zip",
            "expected_sha256": "b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768",
        }],
        "prior_source_path": "step39_temporal_selection/inputs/metro_traffic/public_source39.json",
        "prior_metadata_path": "step39_temporal_selection/inputs/metro_traffic/calendar_metadata39.json",
    },
}


def design_identity(*, root=ROOT):
    seal = read_json(root / "design_seal43.json")
    if (seal.get("schema") != "step43-preaccess-design-seal-v1"
            or seal.get("status") != "SEALED_BEFORE_NEW_MEASUREMENT_DECODING"
            or seal.get("design_sha256") != sha_file(root / "design43.json")
            or seal.get("measurement_values_decoded_or_parsed") != 0):
        raise ValueError("Matching sealed preaccess design required")
    design = read_json(root / "design43.json")
    if set(design["tasks"]) != set(TASKS):
        raise ValueError("Task panel differs from the fixed design")
    return sha_file(root / "design_seal43.json"), design


def metadata_guard(*, root=ROOT):
    identity, design = design_identity(root=root)
    if (root / "protocol_seal43.json").exists():
        raise ValueError("Metadata preparation is closed after scientific freeze")
    return identity, design


def byte_fields(record, separator=b","):
    """CSV field extraction with byte-only measurement handling."""
    if isinstance(separator, str):
        separator = separator.encode("ascii")
    if len(separator) != 1:
        raise ValueError("One-byte separator required")
    body = record.rstrip(b"\r\n")
    if separator == b";":
        # The retained Household source has no quoted CSV fields.
        if b'"' in body:
            raise ValueError("Unexpected Household quoted field")
        return body.split(separator)
    fields, field, quoted, closed = [], bytearray(), False, False
    i = 0
    while i < len(body):
        value = body[i]
        if quoted:
            if value == 34:
                if i + 1 < len(body) and body[i + 1] == 34:
                    field.append(34)
                    i += 2
                    continue
                quoted, closed = False, True
            else:
                field.append(value)
        elif value == separator[0]:
            fields.append(bytes(field))
            field, closed = bytearray(), False
        elif value == 34 and not field and not closed:
            quoted = True
        elif value == 34 or closed:
            raise ValueError("Malformed CSV quotation")
        else:
            field.append(value)
        i += 1
    if quoted:
        raise ValueError("Embedded newline or unterminated quotation")
    fields.append(bytes(field))
    return fields


def parse_timestamp(spec, cells):
    kind = spec["timestamp_kind"]
    if kind == "household_date_time":
        date = cells[spec["timestamp_indices"][0]].strip().split(b"/")
        time = cells[spec["timestamp_indices"][1]].strip().split(b":")
        if len(date) != 3 or len(time) != 3 or not all(x.isdigit() for x in date + time):
            raise ValueError("Unexpected Household date/time syntax")
        day, month, year = map(int, date)
        hour, minute, second = map(int, time)
        stamp = datetime(year, month, day, hour, minute, second)
    else:
        # Only explicitly declared timestamp fields are decoded.
        text = " ".join(cells[i].decode("ascii").strip() for i in spec["timestamp_indices"])
        if kind == "iso_local":
            stamp = datetime.fromisoformat(text)
            if stamp.isoformat(sep=" ", timespec="seconds") != text:
                raise ValueError("Noncanonical Metro timestamp")
        elif kind == "jena_date_time":
            for fmt in spec["timestamp_formats"]:
                try:
                    stamp = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError("Unexpected Jena timestamp")
        else:
            raise ValueError("Unknown timestamp parser")
    if stamp.tzinfo is not None or stamp.second or stamp.minute % spec["cadence_minutes"]:
        raise ValueError("Timestamp cadence or timezone differs")
    return stamp


def payload_bytes(path, member, nested_gzip):
    """Decompress bytes only; valid measurement fields remain opaque."""
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [member]:
            raise ValueError("Unexpected source archive membership")
        raw_member = archive.read(member)
    payload = gzip.decompress(raw_member) if nested_gzip else raw_member
    return raw_member, payload


def _copy_checked(source, destination, expected=None):
    if not source.is_file() or (expected is not None and sha_file(source) != expected):
        raise ValueError("Retained source identity differs: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Do not overwrite an existing preparation artifact")
    shutil.copyfile(source, destination)
    if sha_file(destination) != sha_file(source):
        raise ValueError("Opaque source copy differs")


def retain_source(task, *, root=ROOT):
    design_hash, _ = metadata_guard(root=root)
    spec = SOURCE_SPECS[task]
    directory = root / "inputs/data43" / task
    directory.mkdir(parents=True, exist_ok=True)
    started = directory / "source_started43.json"
    write_json(started, {"schema": "step43-opaque-source-start-v1", "task": task,
                        "design_seal_sha256": design_hash, "utc": utc_now(),
                        "measurement_values_decoded_or_parsed": 0})
    metadata_files = []
    if "prior_source_path" in spec:
        prior = root.parent / spec["prior_source_path"]
        copied = directory / "inherited_source_receipt.json"
        _copy_checked(prior, copied)
        metadata_files.append(file_ref(copied, root=root))
        if "prior_metadata_path" in spec:
            prior = root.parent / spec["prior_metadata_path"]
            copied = directory / "inherited_metadata.json"
            _copy_checked(prior, copied)
            metadata_files.append(file_ref(copied, root=root))
        if task == "household_power2008":
            prior_source = read_json(root.parent / spec["prior_source_path"])
            for rel, expected in prior_source.get("supplementary_metadata_files", {}).items():
                prior = root.parent / "step41_certified_affine" / rel
                copied = directory / prior.name
                _copy_checked(prior, copied, expected)
                metadata_files.append(file_ref(copied, root=root))
        metadata_page = None
    else:
        with urllib.request.urlopen(spec["metadata_url"], timeout=45) as response:
            metadata_page, metadata_resolved = response.read(), response.geturl()
        if urllib.parse.urlparse(metadata_resolved).netloc != urllib.parse.urlparse(spec["metadata_url"]).netloc:
            raise ValueError("Official metadata redirect leaves the declared provider")
        path = directory / "official_metadata.html"
        with path.open("xb") as stream:
            stream.write(metadata_page)
        metadata_files.append(file_ref(path, root=root))
    archives = []
    for declared in spec["archives"]:
        path = directory / declared["filename"]
        if "prior_path" in declared:
            _copy_checked(root.parent / declared["prior_path"], path, declared["expected_sha256"])
            resolved = declared["source_url"]
            transfer = "retained_exact_copy"
        else:
            links = [html.unescape(x.decode("ascii")) for x in re.findall(rb'href=["\']([^"\']+)["\']', metadata_page)]
            if links.count(declared["metadata_link"]) != 1:
                raise ValueError("Official metadata does not identify exactly one declared archive")
            url = urllib.parse.urljoin(spec["metadata_url"], declared["metadata_link"])
            if url != declared["source_url"]:
                raise ValueError("Official link differs from the predeclared source")
            with urllib.request.urlopen(url, timeout=45) as response:
                raw, resolved = response.read(), response.geturl()
            if urllib.parse.urlparse(resolved).netloc != urllib.parse.urlparse(url).netloc:
                raise ValueError("Official source redirect leaves the declared provider")
            with path.open("xb") as stream:
                stream.write(raw)
            transfer = "new_opaque_official_download"
        member, payload = payload_bytes(path, declared["member"], declared["nested_gzip"])
        archives.append({
            **file_ref(path, root=root), "source_id": declared["source_id"], "member": declared["member"],
            "nested_gzip": declared["nested_gzip"], "source_url": declared["source_url"], "resolved_url": resolved,
            "member_sha256": hashlib.sha256(member).hexdigest(), "member_bytes": len(member),
            "payload_sha256": hashlib.sha256(payload).hexdigest(), "payload_bytes": len(payload),
            "transfer": transfer, "prior_path": declared.get("prior_path"),
        })
    record = {"schema": "step43-opaque-source-provenance-v1", "task": task, "utc": utc_now(),
              "design_seal_sha256": design_hash, "archives": archives, "metadata_files": metadata_files,
              "source_spec": {k: v for k, v in spec.items() if k not in ("archives", "prior_source_path", "prior_metadata_path")},
              "measurement_values_decoded_or_parsed": 0, "timestamp_values_parsed": 0,
              "source_scope": "Opaque bytes and archive membership only; numerical measurements remain uninterpreted"}
    write_json(directory / "source43.json", record)
    return record


def source_payloads(task, source, *, root=ROOT):
    payloads = {}
    for item in source["archives"]:
        path = root / item["path"]
        if sha_file(path) != item["sha256"]:
            raise ValueError("Retained source archive changed")
        member, payload = payload_bytes(path, item["member"], item["nested_gzip"])
        if (hashlib.sha256(member).hexdigest() != item["member_sha256"]
                or hashlib.sha256(payload).hexdigest() != item["payload_sha256"]):
            raise ValueError("Retained source member bytes changed")
        payloads[item["source_id"]] = payload
    return payloads


def calendar_projection(task, source, task_design, payloads):
    """Pure byte/calendar transform, also used with manufactured fixtures."""
    spec = source["source_spec"]
    lower = datetime.fromisoformat(task_design["development_start"]) - timedelta(days=2)
    upper = datetime.fromisoformat(task_design["assessment_stop_exclusive"])
    markers = {x.encode("ascii") for x in spec["missing_markers"]}
    groups, headers, member_stats = {}, None, []
    previous, first = None, None
    native_rows, native_unique, duplicate_excess = 0, 0, 0
    by_year, missing_counts = Counter(), Counter()
    for item in source["archives"]:
        source_id = item["source_id"]
        raw = payloads[source_id]
        stream = io.BytesIO(raw)
        heading = stream.readline()
        current = [x.decode(spec["header_encoding"]) for x in byte_fields(heading.removeprefix(b"\xef\xbb\xbf"), spec["separator"])]
        if headers is None:
            headers = current
        if (headers != current or headers[spec["target_index"]] != spec["target_header"]
                or [headers[i] for i in spec["timestamp_indices"]] != spec["timestamp_headers"]):
            raise ValueError("Source target/header schema differs")
        row_count = 0
        for line_number, record in enumerate(stream, 2):
            stop = stream.tell()
            fields = byte_fields(record, spec["separator"])
            if len(fields) != len(headers):
                raise ValueError("Source row field count differs")
            stamp = parse_timestamp(spec, fields)
            if previous is not None and stamp < previous:
                raise ValueError("Source timestamps are not ordered")
            native_rows += 1
            row_count += 1
            if previous == stamp:
                duplicate_excess += 1
            else:
                native_unique += 1
                by_year[str(stamp.year)] += 1
            previous = stamp
            if first is None:
                first = stamp
            if stamp.minute or not lower <= stamp < upper:
                continue
            iso = stamp.isoformat(timespec="seconds")
            missing = fields[spec["target_index"]].strip(b" \t") in markers if markers else False
            if markers:
                missing_counts["declared_missing" if missing else "not_declared_missing"] += 1
            row = groups.setdefault(iso, {"timestamp": iso, "records": [], "declared_target_missing": False})
            row["records"].append({"source_id": source_id, "source_line": line_number,
                                   "byte_start": stop - len(record), "byte_stop": stop})
            row["declared_target_missing"] = row["declared_target_missing"] or missing
        member_stats.append({"source_id": source_id, "payload_sha256": hashlib.sha256(raw).hexdigest(),
                             "payload_bytes": len(raw), "timestamp_rows": row_count})
    if not groups or first is None or previous is None:
        raise ValueError("No hourly timestamps in the predeclared window")
    rows = list(groups.values())
    return {
        "schema": "step43-calendar-byte-projection-v1", "task": task, "headers": headers,
        "source_spec": spec, "rows": rows, "member_stats": member_stats,
        "timestamp_rows": native_rows, "unique_native_timestamps": native_unique,
        "duplicate_record_excess": duplicate_excess, "source_timestamp_counts_by_year": dict(sorted(by_year.items())),
        "first_source_timestamp": first.isoformat(timespec="seconds"), "last_source_timestamp": previous.isoformat(timespec="seconds"),
        "projected_hourly_timestamps": len(rows), "projected_start": lower.isoformat(timespec="seconds"),
        "projected_stop_exclusive": upper.isoformat(timespec="seconds"),
        "parsed_data_columns": spec["timestamp_indices"], "target_values_decoded_or_parsed": 0,
        "other_measurement_values_decoded_or_parsed": 0, "declared_missing_markers": spec["missing_markers"],
        "missing_status_inspected": bool(markers), "missing_status_counts": dict(missing_counts),
        "duplicate_target_agreement_examined": False,
    }


def next_month(month):
    stamp = datetime.strptime(month, "%Y-%m")
    return datetime(stamp.year + (stamp.month == 12), stamp.month % 12 + 1, 1).date().isoformat()


def required_times(target):
    origin = target - timedelta(hours=24)
    return origin, [origin - timedelta(hours=23-j) for j in range(24)]


def derive_queries(task, index, start, stop):
    target = datetime.fromisoformat(start).replace(hour=23)
    end = datetime.fromisoformat(stop).replace(hour=23)
    queries, excluded = [], []
    while target < end:
        origin, lags = required_times(target)
        stamps = [x.isoformat(timespec="seconds") for x in [target, *lags]]
        absent = [x for x in stamps if x not in index]
        missing = [x for x in stamps if x in index and index[x]["declared_target_missing"]]
        if absent or missing:
            excluded.append({"target_timestamp": stamps[0], "absent_timestamps": absent, "declared_missing_timestamps": missing})
        else:
            queries.append({"id": task + ":" + stamps[0], "task": task, "target_timestamp": stamps[0],
                            "origin_timestamp": origin.isoformat(timespec="seconds")})
        target += timedelta(days=1)
    return queries, excluded


def task_rosters(task, task_design, projection, limits):
    index = {row["timestamp"]: row for row in projection["rows"]}
    if len(index) != len(projection["rows"]):
        raise ValueError("Duplicate grouped calendar timestamps")
    stages = {}
    for stage, month in zip(STAGES[:3], task_design["validation_months"]):
        validation_start, validation_stop = month + "-01", next_month(month)
        train, train_excluded = derive_queries(task, index, task_design["development_start"], validation_start)
        validation, validation_excluded = derive_queries(task, index, validation_start, validation_stop)
        if len(train) < limits["minimum_training_rows_each_stage"] or len(validation) < limits["minimum_validation_rows_each_fold"]:
            raise ValueError("Predeclared minimum development cohort is unavailable")
        stages[stage] = {"train_start": task_design["development_start"], "train_stop_exclusive": validation_start,
                         "validation_start": validation_start, "validation_stop_exclusive": validation_stop,
                         "train": train, "validation": validation,
                         "counts": {"train": len(train), "validation": len(validation)},
                         "excluded": {"train": train_excluded, "validation": validation_excluded}}
    final, final_excluded = derive_queries(task, index, task_design["development_start"], task_design["assessment_start"])
    assessment, assessment_excluded = derive_queries(task, index, task_design["assessment_start"], task_design["assessment_stop_exclusive"])
    if len(final) < limits["minimum_training_rows_each_stage"] or len(assessment) < limits["minimum_assessment_rows_per_task"]:
        raise ValueError("Predeclared minimum final/assessment cohort is unavailable")
    stages["final"] = {"train_start": task_design["development_start"], "train_stop_exclusive": task_design["assessment_start"],
                       "train": final, "validation": [], "counts": {"train": len(final), "validation": 0},
                       "excluded": {"train": final_excluded, "validation": []}}
    return {"stages": stages, "assessment": assessment, "assessment_excluded": assessment_excluded,
            "assessment_start": task_design["assessment_start"], "assessment_stop_exclusive": task_design["assessment_stop_exclusive"],
            "counts": {"stages": {s: stages[s]["counts"] for s in STAGES}, "assessment": len(assessment)}}


def exposure_evidence(*, root=ROOT):
    """Retained-record search and exact old numeric-access timestamp evidence."""
    design_hash, design = metadata_guard(root=root)
    out = root / "inputs/exposure43"
    out.mkdir(parents=True, exist_ok=True)
    prior_roots = sorted(p.name for p in root.parent.iterdir()
                         if p.is_dir() and "library_helpers" not in p.name
                         and re.match(r"step(?:[12][0-9]|3[0-9]|4[012])_", p.name))
    command = ["rg", "--files", *prior_roots, "-g", "*.py", "-g", "*.txt", "-g", "*.tex", "-g", "*.md",
               "-g", "*protocol*.json", "-g", "PROTOCOL.json", "-g", "*source*.json", "-g", "*summary*.json",
               "-g", "*exposure*.json", "-g", "!**/tmp/**", "-g", "!**/reporting/demo42/**",
               "-g", "!**/__pycache__/**", "-g", "!**/patient*/**", "-g", "!**/reserved*/**", "-g", "!**/clinical*/**"]
    names = subprocess.check_output(command, cwd=root.parent, text=True).splitlines()
    patterns = {"household_power2008": r"household[_ -]?power[_ -]?2008",
                "jena_temperature2018": r"jena[_ -]?temperature[_ -]?2018|mpi_roof_2018[ab]|jena2018[ab]",
                "metro_traffic2018": r"metro[_ -]?(?:traffic|volume)[_ -]?2018"}
    files, matches = [], {key: [] for key in patterns}
    for name in sorted(names):
        path = root.parent / name
        if path.stat().st_size > 8*1024*1024:
            continue
        raw = path.read_bytes()
        files.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        for key, pattern in patterns.items():
            if re.search(pattern, raw.decode("utf-8", errors="replace"), re.I):
                matches[key].append(name)
    search = {"schema": "step43-scoped-prior-text-search-v1", "utc": utc_now(),
              "design_seal_sha256": design_hash, "files": files, "file_count": len(files),
              "bytes_searched": sum(x["bytes"] for x in files), "patterns": patterns, "matches": matches,
              "scope": "Retained <=8MiB text/source/protocol/provenance files in named pre-Step43 roots; no raw arrays, CSVs, archives or patient/reserved/clinical paths",
              "absolute_historical_never_seen_claim": False}
    write_json(out / "prior_text_search43.json", search)
    trace_specs = [
        ("step39_temporal_selection", 39, ["metro_traffic"]),
        ("step41_certified_affine/recovered/step40/step40_head_selection", 40, ["household_power2009", "jena_temperature2016"]),
        ("step41_certified_affine", 41, ["household_power2010", "jena_temperature2017"]),
    ]
    traces, proofs = [], []
    for directory, step, tasks in trace_specs:
        prior = root.parent / directory
        for relative in [f"protocol{step}.json", f"protocol_seal{step}.json", f"source/data{step}.py",
                         f"source/prepare_data{step}.py", f"output/access_and_execution{step}.jsonl"]:
            source_path = prior / relative
            destination = out / "proofs" / f"step{step}" / relative
            _copy_checked(source_path, destination)
            proofs.append({**file_ref(destination, root=root), "prior_path": (Path(directory)/relative).as_posix()})
        for task in tasks:
            base = prior / "output" / task
            paths = (sorted(base.glob(f"*/data_access_train{step}.json"))
                     + sorted(base.glob(f"*/data_access_validation{step}.json"))
                     + sorted((base / "test/history").glob("*.json"))
                     + sorted((base / "test/targets").glob("*.json")))
            receipts, stamps = [], set()
            for path in paths:
                value = read_json(path)
                access = value.get("access", value)
                actual = access["requested_timestamps"]
                stamps.update(actual)
                # Persist only original access metadata, which contains no numeric labels.
                access_record = {"prior_path": path.relative_to(root.parent).as_posix(), "prior_sha256": sha_file(path),
                                 "access": access}
                receipts.append(access_record)
            record = {"schema": "step43-prior-authorized-numeric-access-v1", "prior_step": step, "task": task,
                      "receipts": receipts, "receipt_count": len(receipts),
                      "unique_requested_timestamps": sorted(stamps), "unique_timestamp_count": len(stamps),
                      "minimum": min(stamps), "maximum": max(stamps),
                      "year_counts": dict(sorted(Counter(x[:4] for x in stamps).items())),
                      "numeric_values_copied": False,
                      "scope": "Exact timestamp authorization metadata extracted from retained prior receipts; original receipt hashes retained, not an external authentication claim"}
            path = out / f"prior_access_step{step}_{task}.json"
            write_json(path, record)
            traces.append({**file_ref(path, root=root), "prior_step": step, "task": task,
                           "minimum": min(stamps), "maximum": max(stamps), "unique_timestamp_count": len(stamps)})
    prior_by_domain = {
        "household_power2008": [x for x in traces if x["task"].startswith("household_")],
        "jena_temperature2018": [x for x in traces if x["task"].startswith("jena_")],
        "metro_traffic2018": [x for x in traces if x["task"].startswith("metro_")],
    }
    disjoint = {}
    for task, task_design in design["tasks"].items():
        # Include all prospective assessment histories, before missing-status exclusions.
        start = datetime.fromisoformat(task_design["assessment_start"]).replace(hour=23)
        stop = datetime.fromisoformat(task_design["assessment_stop_exclusive"]).replace(hour=23)
        proposed = set()
        target = start
        while target < stop:
            proposed.add(target.isoformat(timespec="seconds"))
            proposed.update(x.isoformat(timespec="seconds") for x in required_times(target)[1])
            target += timedelta(days=1)
        previous = set()
        for trace in prior_by_domain[task]:
            previous.update(read_json(root / trace["path"])["unique_requested_timestamps"])
        overlap = sorted(proposed & previous)
        if overlap:
            raise ValueError("Proposed assessment values overlap retained numeric exposure")
        disjoint[task] = {"assessment_and_history_timestamp_count": len(proposed),
                          "prior_numeric_timestamp_count": len(previous), "overlap_count": 0,
                          "prospective_timestamps_sha256": digest(sorted(proposed)),
                          "prior_timestamps_sha256": digest(sorted(previous)),
                          "freshness_scope": task_design["freshness_scope"]}
    record = {"schema": "step43-prior-exposure-evidence-v1", "utc": utc_now(), "status": "PASS",
              "design_seal_sha256": design_hash, "search": file_ref(out / "prior_text_search43.json", root=root),
              "traces": traces, "proof_files": proofs, "disjoint_assessment": disjoint,
              "absolute_historical_never_seen_claim": False,
              "new_domains_claim": False, "new_measurement_values_decoded": 0,
              "known_prior_sources": [
                  "Appliances Energy Prediction: Step24 (documented in Step35 README and Step39 prior exposure)",
                  "Beijing PM2.5: Step35; Bike Sharing: Steps36-37; Seoul: Step38",
                  "Tetouan zone1 and Metro2017: Step39; Household2009/Jena2016: Step40; Household2010/Jena2017: Steps41-42"],
              "metro_local_trace_caution": "Retained Step39 Metro receipts contain17 revealed assessment targets through2017-09-17 plus the2017-09-18 history; declared protocol assessment extends throughNovember. Neither authorizes2018 numeric access.",
              "mirror_limitation": SOURCE_SPECS["household_power2008"]["transport_limitation"]}
    write_json(out / "exposure43.json", record)
    return record


def prepare(*, root=ROOT):
    design_hash, design = metadata_guard(root=root)
    manifest_path = root / "inputs/data43/manifest43.json"
    if manifest_path.exists():
        raise FileExistsError("Retain the existing metadata manifest")
    exposure = exposure_evidence(root=root)
    tasks = {}
    for task in TASKS:
        source = retain_source(task, root=root)
        payloads = source_payloads(task, source, root=root)
        projection = calendar_projection(task, source, design["tasks"][task], payloads)
        directory = root / "inputs/data43" / task
        write_json(directory / "calendar43.json", projection)
        rosters = task_rosters(task, design["tasks"][task], projection, design["forecasting"])
        metadata = {k: v for k, v in projection.items() if k not in ("rows", "source_spec")}
        metadata.update({"counts": rosters["counts"], "utc": utc_now(), "design_seal_sha256": design_hash,
                         "calendar": file_ref(directory / "calendar43.json", root=root),
                         "source": file_ref(directory / "source43.json", root=root)})
        write_json(directory / "metadata43.json", metadata)
        tasks[task] = {**rosters, "spec": source["source_spec"], "archives": source["archives"],
                       "source": file_ref(directory / "source43.json", root=root),
                       "calendar": file_ref(directory / "calendar43.json", root=root),
                       "metadata": file_ref(directory / "metadata43.json", root=root),
                       "freshness": exposure["disjoint_assessment"][task]}
        print(json.dumps({"task": task, "metadata_counts": rosters["counts"], "numeric_measurements_decoded": 0}), flush=True)
    record = {"schema": "step43-data-manifest-v1", "utc": utc_now(),
              "design_seal_sha256": design_hash, "design_sha256": sha_file(root / "design43.json"),
              "tasks": tasks, "task_order": list(TASKS), "stages": list(STAGES),
              "exposure": file_ref(root / "inputs/exposure43/exposure43.json", root=root),
              "prepare_source": file_ref(root / "source/prepare43.py", root=root),
              "measurement_values_decoded_or_parsed": 0,
              "calendar_rule": "Daily23 targets, horizon24hours, oldest-first24 hourly snapshots ending at origin",
              "eligibility": "Required timestamps present; only predeclared lexical missing status can exclude; no numeric outlier exclusions",
              "numeric_access_gate": "Final protocol/source seal and independent preflight PASS; later phases require their model/selection/prediction seals"}
    write_json(manifest_path, record)
    print(json.dumps({"status": "PASS", "manifest": file_ref(manifest_path, root=root),
                      "new_measurement_values_decoded": 0}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepare(root=args.root.resolve())
