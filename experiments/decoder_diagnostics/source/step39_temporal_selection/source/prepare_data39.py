"""Prepare two public sources without interpreting any measurement values.

Only headers and timestamp fields are decoded. Data-record fields stay bytes;
the target, other zones, weather, and holiday fields are never numerically
parsed here. Calendar eligibility depends only on timestamp presence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "tetouan_zone1": {
        "dataset": "Power Consumption of Tetouan City",
        "uci_id": 849,
        "metadata_url": "https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city",
        "doi": "10.24432/C5B034",
        "citation": "Salam, A. & El Hibaoui, A. (2018). Power Consumption of Tetouan City [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5B034.",
        "member": "Tetuan City power consumption.csv",
        "inner_compression": None,
        "timestamp_index": 0,
        "target_index": 6,
        "target_header": "Zone 1 Power Consumption",
        "timestamp_format": "%m/%d/%Y %H:%M",
        "cadence_minutes": 10,
        "reported_instances": 52417,
        "timezone_convention": "dataset-recorded local-naive timestamps; no timezone conversion or invented UTC offset",
    },
    "metro_traffic": {
        "dataset": "Metro Interstate Traffic Volume",
        "uci_id": 492,
        "metadata_url": "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        "doi": "10.24432/C5X60B",
        "citation": "Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X60B.",
        "member": "Metro_Interstate_Traffic_Volume.csv.gz",
        "inner_compression": "gzip",
        "timestamp_index": 7,
        "target_index": 8,
        "target_header": "traffic_volume",
        "timestamp_format": "%Y-%m-%d %H:%M:%S",
        "cadence_minutes": 60,
        "reported_instances": 48204,
        "timezone_convention": "source describes local CST; retain recorded local-naive timestamps without UTC or daylight-saving reinterpretation",
    },
}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def archive_relative(task):
    return f"inputs/{task}/source_uci{TASKS[task]['uci_id']}.zip"


def byte_fields(record):
    """Split one CSV record as bytes, including ordinary RFC4180 quoting."""
    body = record.rstrip(b"\r\n")
    fields = []
    field = bytearray()
    quoted = False
    closed = False
    i = 0
    while i < len(body):
        value = body[i]
        if quoted:
            if value == 34:
                if i + 1 < len(body) and body[i + 1] == 34:
                    field.append(34)
                    i += 2
                    continue
                quoted = False
                closed = True
            else:
                field.append(value)
        elif value == 44:
            fields.append(bytes(field))
            field = bytearray()
            closed = False
        elif value == 34 and not field and not closed:
            quoted = True
        elif value == 34 or closed:
            raise RuntimeError("Malformed quoted CSV record")
        else:
            field.append(value)
        i += 1
    if quoted:
        raise RuntimeError("Embedded newlines or unfinished CSV quotation are not approved")
    fields.append(bytes(field))
    return fields


def download(task):
    if (ROOT / "protocol_seal39.json").exists():
        raise RuntimeError("Preparation forbidden after protocol freeze")
    spec = TASKS[task]
    archive = ROOT / archive_relative(task)
    if archive.exists():
        raise RuntimeError("Opaque archive already present; no redownload")
    started = time.perf_counter()
    with urllib.request.urlopen(spec["metadata_url"], timeout=45) as response:
        page = response.read()
        metadata_resolved_url = response.geturl()
    # The entire page remains bytes. Decode only download hrefs, not page data.
    links = re.findall(rb'href=["\']([^"\']*/static/public/' + str(spec["uci_id"]).encode() + rb'/[^"\']+)["\']', page)
    if len(set(links)) != 1:
        raise RuntimeError("Official metadata does not identify exactly one download")
    url = html.unescape(links[0].decode("ascii"))
    if url.startswith("/"):
        url = "https://archive.ics.uci.edu" + url
    if not url.startswith(f"https://archive.ics.uci.edu/static/public/{spec['uci_id']}/"):
        raise RuntimeError("Download is outside the declared official dataset path")
    with urllib.request.urlopen(url, timeout=45) as response:
        raw = response.read()
        resolved = response.geturl()
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(raw)
    receipt = {
        "schema": "step39-opaque-public-source-download-v1",
        "task": task, **spec, "source_url": url, "resolved_url": resolved,
        "metadata_resolved_url": metadata_resolved_url,
        "metadata_html_sha256": sha_bytes(page),
        "archive_path": archive_relative(task), "archive_sha256": sha_bytes(raw),
        "archive_bytes": len(raw), "download_utc": datetime.now(timezone.utc).isoformat(),
        "download_elapsed_seconds": time.perf_counter() - started,
        "source_license": "CC BY 4.0", "stage": "OPAQUE_ARCHIVE_ONLY",
        "measurement_values_decoded_or_parsed": 0, "timestamp_values_parsed": 0,
        "scientific_model_fits": 0, "absolute_historical_never_seen_claim": False,
    }
    write_json(ROOT / f"inputs/{task}/public_source39.json", receipt)
    return receipt


def source_csv(task):
    spec = TASKS[task]
    source = json.loads((ROOT / f"inputs/{task}/public_source39.json").read_text())
    archive = ROOT / archive_relative(task)
    if sha_file(archive) != source["archive_sha256"]:
        raise RuntimeError("Opaque public archive changed")
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as zipped:
        if zipped.namelist() != [spec["member"]]:
            raise RuntimeError("Public archive member roster differs: " + repr(zipped.namelist()))
        raw = zipped.read(spec["member"])
    member_sha = sha_bytes(raw)
    if spec["inner_compression"] == "gzip":
        raw = gzip.decompress(raw)
    return raw, source, member_sha


def calendar_projection(task, raw_csv):
    spec = TASKS[task]
    records = raw_csv.splitlines(keepends=True)
    if not records:
        raise RuntimeError("Empty source CSV")
    headers = [x.decode("utf-8") for x in byte_fields(records[0].removeprefix(b"\xef\xbb\xbf"))]
    if len(headers) != 9 or headers[spec["target_index"]] != spec["target_header"]:
        raise RuntimeError("Public CSV target schema differs: " + repr(headers))
    wanted_timestamp_header = "DateTime" if task == "tetouan_zone1" else "date_time"
    if headers[spec["timestamp_index"]] != wanted_timestamp_header:
        raise RuntimeError("Public CSV timestamp schema differs")
    groups = {}
    offset = len(records[0])
    previous = None
    timestamp_re = (r"\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}" if task == "tetouan_zone1"
                    else r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    for line_number, record in enumerate(records[1:], 2):
        cells = byte_fields(record)
        if len(cells) != len(headers):
            raise RuntimeError("Malformed public CSV field count")
        value = cells[spec["timestamp_index"]].decode("ascii")
        if re.fullmatch(timestamp_re, value) is None:
            raise RuntimeError("Timestamp format outside approved grammar")
        stamp = datetime.strptime(value, spec["timestamp_format"])
        if stamp.second != 0 or stamp.minute % spec["cadence_minutes"] != 0:
            raise RuntimeError("Timestamp cadence differs")
        if previous is not None and stamp < previous:
            raise RuntimeError("Unordered source timestamps")
        previous = stamp
        iso = stamp.isoformat(timespec="seconds")
        groups.setdefault(iso, []).append({"csv_line": line_number, "byte_start": offset,
                                          "byte_stop": offset + len(record)})
        offset += len(record)
    rows = [{"timestamp": stamp, "records": refs} for stamp, refs in groups.items()]
    first = datetime.fromisoformat(rows[0]["timestamp"])
    last = datetime.fromisoformat(rows[-1]["timestamp"])
    cadence = timedelta(minutes=spec["cadence_minutes"])
    expected = int((last - first) / cadence) + 1
    by_year = Counter(stamp[:4] for stamp in groups)
    hourly = {stamp for stamp in groups if datetime.fromisoformat(stamp).minute == 0}
    monthly_queries = {}
    skipped = []
    for month in range(1, 13):
        start = datetime(2017, month, 1, 23)
        stop = datetime(2018, 1, 1, 23) if month == 12 else datetime(2017, month + 1, 1, 23)
        targets = []
        while start < stop:
            origin = start - timedelta(hours=24)
            required = [origin - timedelta(hours=23 - j) for j in range(24)]
            target_text = start.isoformat(timespec="seconds")
            absent = [stamp.isoformat(timespec="seconds") for stamp in [start, *required]
                      if stamp.isoformat(timespec="seconds") not in hourly]
            if not absent:
                targets.append(target_text)
            else:
                skipped.append({"target_timestamp": target_text, "absent_required_timestamps": absent})
            start += timedelta(days=1)
        monthly_queries[f"2017-{month:02d}"] = targets
    return {
        "schema": "step39-calendar-only-byte-projection-v1", "task": task,
        "headers": headers, "parsed_data_columns": [spec["timestamp_index"]],
        "parsed_data_field_names": [wanted_timestamp_header], "target_field_index": spec["target_index"],
        "timestamp_format": spec["timestamp_format"], "timezone_convention": spec["timezone_convention"],
        "data_record_decoding": "ASCII timestamp field only; all measurement fields remain opaque bytes",
        "csv_bytes": len(raw_csv), "csv_sha256": sha_bytes(raw_csv), "rows": rows,
        "timestamp_rows": len(records) - 1, "unique_timestamps": len(rows),
        "first_timestamp": rows[0]["timestamp"], "last_timestamp": rows[-1]["timestamp"],
        "native_cadence_minutes": spec["cadence_minutes"], "expected_native_timestamps": expected,
        "missing_native_timestamps": expected - len(rows),
        "duplicate_record_excess": len(records) - 1 - len(rows),
        "duplicated_timestamps": sum(len(refs) > 1 for refs in groups.values()),
        "maximum_records_per_timestamp": max(map(len, groups.values())),
        "unique_timestamp_counts_by_year": dict(sorted(by_year.items())),
        "hourly_snapshot_rule": "Retain point readings at minute00 only; no hourly aggregation",
        "eligible_2017_daily23_targets_by_month": monthly_queries,
        "calendar_ineligible_2017_daily23_targets": skipped,
        "query_horizon_hours": 24, "query_lag_hours": 24,
        "target_values_decoded_or_parsed": 0, "other_measurement_values_decoded_or_parsed": 0,
        "duplicate_target_agreement_examined": False,
        "duplicate_target_policy": "At authorized value read, all same-timestamp target tokens must agree exactly or STOP_RETAIN_FAILURE; no outcome-based eligibility",
    }


def project(task):
    if (ROOT / "protocol_seal39.json").exists():
        raise RuntimeError("Preparation forbidden after protocol freeze")
    raw, source, member_sha = source_csv(task)
    result = calendar_projection(task, raw)
    result.update({"archive_sha256": source["archive_sha256"], "archive_member": TASKS[task]["member"],
                   "archive_member_sha256": member_sha,
                   "projection_utc": datetime.now(timezone.utc).isoformat(),
                   "projection_source_sha256": sha_file(Path(__file__))})
    projection_path = ROOT / f"inputs/{task}/calendar_projection39.json"
    write_json(projection_path, result)
    metadata = {key: value for key, value in result.items() if key not in {"rows", "eligible_2017_daily23_targets_by_month", "calendar_ineligible_2017_daily23_targets"}}
    metadata["projection_sha256"] = sha_file(projection_path)
    metadata["monthly_eligible_2017_daily23_counts"] = {k: len(v) for k, v in result["eligible_2017_daily23_targets_by_month"].items()}
    metadata["eligible_2017_daily23_count"] = sum(metadata["monthly_eligible_2017_daily23_counts"].values())
    metadata["calendar_ineligible_2017_daily23_count"] = len(result["calendar_ineligible_2017_daily23_targets"])
    metadata["catalog_reported_instances"] = TASKS[task]["reported_instances"]
    metadata["catalog_count_matches_data_rows"] = result["timestamp_rows"] == TASKS[task]["reported_instances"]
    write_json(ROOT / f"inputs/{task}/calendar_metadata39.json", metadata)
    return metadata


def prior_exposure():
    """Bounded name search in retained text artifacts; never opens array/CSV data."""
    if (ROOT / "protocol_seal39.json").exists():
        raise RuntimeError("Exposure record cannot change after freeze")
    workspace = ROOT.parent
    listing = subprocess.run(["rg", "--files", "-g", "*.py", "-g", "*.txt", "-g", "*.md", "-g", "*.tex", "-g", "*.json",
                              "-g", "!**/tmp/**", "-g", "!**/__pycache__/**"], cwd=workspace, capture_output=True, text=True, check=True)
    included = []
    for text in listing.stdout.splitlines():
        path = Path(text)
        if path.parts[0] == ROOT.name:
            continue
        # Search methods, reports, protocols, public source receipts, and audits.
        name = path.name.lower()
        if not ("source" in path.parts or "reporting" in path.parts or "audit" in path.parts or
                any(token in name for token in ("readme", "protocol", "source", "exposure", "summary", "report", "result"))):
            continue
        if any(token in str(path).lower() for token in ("private_library", "patient", "reserved", "clinical")):
            continue
        file = workspace / path
        if file.stat().st_size > 8 * 1024 * 1024:
            continue
        included.append(path)
    patterns = {
        "tetouan_zone1": rb"tet(?:ou|u)an|uci[ _]?849|10\.24432/C5B034",
        "metro_traffic": rb"metro[ _+]interstate|metro_interstate_traffic|uci[ _]?492|10\.24432/C5X60B",
        "excluded_appliances": rb"appliances[ _+]energy|energydata_complete|10\.24432/C5VC8G",
        "excluded_beijing": rb"beijing[ _+]pm2|beijing_pm25|10\.24432/C5JS49",
    }
    matches = {key: [] for key in patterns}
    bound = {}
    byte_count = 0
    for path in sorted(included):
        raw = (workspace / path).read_bytes()
        byte_count += len(raw)
        bound[str(path)] = sha_bytes(raw)
        for key, pattern in patterns.items():
            if re.search(pattern, raw, re.IGNORECASE):
                matches[key].append(str(path))
    result = {
        "schema": "step39-scoped-prior-exposure-search-v1", "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Retained method/report/protocol/public-source/audit text files at most8MiB; excludes currentStep39,tmp,patient/reserved/clinical paths and numerical array/CSV/archive formats",
        "file_count": len(bound), "bytes_searched": byte_count, "file_sha256": bound,
        "patterns": {key: pattern.decode("ascii") for key, pattern in patterns.items()}, "matches": matches,
        "freshness_claim": "No match in this explicitly scoped retained record; not an unrestricted historical never-seen claim",
        "excluded_known_prior_sources": [
            {"dataset": "UCI Appliances Energy Prediction", "prior_evaluation": "Step24, also explicitly recorded in Step35 README", "source": "step35_public_forecasting/README35.txt"},
            {"dataset": "UCI Beijing PM2.5", "prior_evaluation": "Step35", "source": "step35_public_forecasting/README35.txt"},
        ],
        "private_or_reserved_arrays_read": False,
    }
    write_json(ROOT / "inputs/prior_exposure39.json", result)
    return {key: value for key, value in result.items() if key != "file_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("download", "project", "prior-exposure"))
    parser.add_argument("task", nargs="?", choices=tuple(TASKS))
    args = parser.parse_args()
    if args.mode == "prior-exposure":
        result = prior_exposure()
    else:
        if args.task is None:
            parser.error("task required")
        result = download(args.task) if args.mode == "download" else project(args.task)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
