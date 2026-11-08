"""Retain public source archives and inspect only calendar / declared missing status.

No valid measurement token is decoded or parsed here. Household eligibility may
use the predeclared lexical missing markers; Jena eligibility is timestamp-only.
Source row references identify original archive members and byte ranges.
Household transport may use the explicitly pinned public mirror; no byte-identity
claim is made against an unreachable official archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "household_power2010": {
        "dataset": "Individual Household Electric Power Consumption", "year": 2010,
        "provider": "UCI Machine Learning Repository", "uci_id": 235,
        "metadata_url": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
        "doi": "10.24432/C58K54",
        "citation": "Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.",
        "source_license": "CC BY 4.0",
        "archives": [{"source_id": "household", "filename": "household_power_consumption_mirror.zip", "member": "household_power_consumption.txt"}],
        "transport_policy": "Pinned public repository mirror after retained official-source connection failures; scientific dataset identity remains UCI235, byte identity with official archive unverified",
        "mirror_repository": "jbrownlee/Datasets", "mirror_commit": "509690881d2c47c89d3d221ed2e2bb30e5afa9c0",
        "separator": ";", "header_encoding": "utf-8", "timestamp_indices": [0, 1], "target_index": 2,
        "target_header": "Global_active_power", "unit": "kilowatt",
        "timestamp_formats": ["%d/%m/%Y %H:%M:%S"], "cadence_minutes": 1,
        "reported_instances": 2075259, "numeric_domain": "finite nonnegative decimal",
        "snapshot_description": "Minute-averaged active power at the recorded minute00; not an hourly average",
        "timezone_convention": "Source local-naive Date and Time; no UTC offset or daylight-saving reconstruction is invented",
        "missing_status_rule": "Predeclared exact empty or question-mark target field, after stripping surrounding ASCII spaces and tabs; no valid numeric token inspected",
        "missing_markers": ["", "?"],
    },
    "jena_temperature2017": {
        "dataset": "MPI-BGC Beutenberg roof weather station", "year": 2017,
        "provider": "Max Planck Institute for Biogeochemistry", "uci_id": None, "doi": None,
        "metadata_url": "https://www.bgc-jena.mpg.de/wetter/weather_data.html",
        "documentation_url": "https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf",
        "citation": "Max Planck Institute for Biogeochemistry. Beutenberg roof weather station, 2017 half-year archives. https://www.bgc-jena.mpg.de/wetter/weather_data.html. Documentation: Olaf Kolle (4 February 2025).",
        "source_license": "CC BY 4.0",
        "transport_policy": "Direct original archives from official MPI-BGC provider",
        "archives": [
            {"source_id": "jena2017a", "filename": "mpi_roof_2017a.zip", "member": "mpi_roof_2017a.csv", "metadata_link": "data/mpi_roof_2017a.zip"},
            {"source_id": "jena2017b", "filename": "mpi_roof_2017b.zip", "member": "mpi_roof_2017b.csv", "metadata_link": "data/mpi_roof_2017b.zip"},
        ],
        "separator": ",", "header_encoding": "latin-1", "timestamp_indices": [0], "target_index": 2,
        "target_header": "T (degC)", "unit": "degree Celsius",
        "timestamp_formats": ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"], "cadence_minutes": 10,
        "reported_instances": None, "numeric_domain": "finite signed decimal at or above -273.15 degree Celsius",
        "snapshot_description": "Recorded 10-minute-averaged air temperature at minute00, timestamp at end of averaging period; not an hourly average",
        "timezone_convention": "Source documentation specifies MEZ (CET) and interval-end timestamps; retain recorded naive times without a daylight-saving reconstruction",
        "missing_status_rule": "No measurement missing-status inspection; exact timestamp presence only",
        "missing_markers": [],
    },
}


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def archive_relatives(task):
    return [f"inputs/data41/{task}/{a['filename']}" for a in TASKS[task]["archives"]]


def byte_fields(record, separator=b","):
    """Byte-only CSV splitter; does not decode measurement cells."""
    if isinstance(separator, str):
        separator = separator.encode("ascii")
    if len(separator) != 1:
        raise RuntimeError("Exactly one byte separator required")
    body = record.rstrip(b"\r\n")
    fields, field, quoted, closed = [], bytearray(), False, False
    i = 0
    while i < len(body):
        value = body[i]
        if quoted:
            if value == 34:
                if i + 1 < len(body) and body[i + 1] == 34:
                    field.append(34); i += 2; continue
                quoted, closed = False, True
            else:
                field.append(value)
        elif value == separator[0]:
            fields.append(bytes(field)); field, closed = bytearray(), False
        elif value == 34 and not field and not closed:
            quoted = True
        elif value == 34 or closed:
            raise RuntimeError("Malformed quoted source record")
        else:
            field.append(value)
        i += 1
    if quoted:
        raise RuntimeError("Embedded newline or unfinished quotation is not approved")
    fields.append(bytes(field))
    return fields


def _no_frozen_protocol():
    if (ROOT / "protocol_seal41.json").exists():
        raise RuntimeError("Metadata preparation forbidden after protocol freeze")


def source_manifest(archives):
    fields = ("source_id", "archive_path", "archive_sha256", "archive_bytes", "member", "source_url")
    return [{k: a[k] for k in fields} for a in archives]


def download(task):
    _no_frozen_protocol()
    spec = TASKS[task]
    if any((ROOT / relative).exists() for relative in archive_relatives(task)):
        raise RuntimeError("Opaque official archive already present; no redownload")
    started = time.perf_counter()
    with urllib.request.urlopen(spec["metadata_url"], timeout=45) as response:
        page, resolved_metadata = response.read(), response.geturl()
    links = re.findall(rb'href=["\']([^"\']+)["\']', page)
    archives = []
    for declared, relative in zip(spec["archives"], archive_relatives(task)):
        if spec["uci_id"] is not None:
            prefix = b"/static/public/" + str(spec["uci_id"]).encode() + b"/"
            found = sorted(set(x for x in links if prefix in x))
        else:
            found = sorted(set(x for x in links if html.unescape(x.decode("ascii", errors="strict")) == declared["metadata_link"]))
        if len(found) != 1:
            raise RuntimeError("Official metadata must identify exactly one declared archive link")
        url = urllib.parse.urljoin(spec["metadata_url"], html.unescape(found[0].decode("ascii")))
        if urllib.parse.urlparse(url).netloc != urllib.parse.urlparse(spec["metadata_url"]).netloc:
            raise RuntimeError("Archive download leaves declared official provider")
        with urllib.request.urlopen(url, timeout=60) as response:
            raw, resolved = response.read(), response.geturl()
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
            inventory = [{"name": a.filename, "bytes": a.file_size, "compressed_bytes": a.compress_size} for a in zipped.infolist()]
        archives.append({**declared, "archive_path": relative, "archive_sha256": sha_bytes(raw),
                         "archive_bytes": len(raw), "source_url": url, "resolved_url": resolved,
                         "inventory": inventory})
    metadata_path = f"inputs/data41/{task}/official_metadata.html"
    (ROOT / metadata_path).write_bytes(page)
    receipt = {"schema": "step41-opaque-public-source-download-v1", "task": task,
               **{k: v for k, v in spec.items() if k != "archives"}, "archives": archives,
               "source_manifest_sha256": sha_bytes(canonical_bytes(source_manifest(archives))),
               "metadata_path": metadata_path, "metadata_sha256": sha_bytes(page), "metadata_html_sha256": sha_bytes(page),
               "metadata_resolved_url": resolved_metadata,
               "download_utc": datetime.now(timezone.utc).isoformat(),
               "download_elapsed_seconds": time.perf_counter() - started,
               "stage": "OPAQUE_ARCHIVES_ONLY", "measurement_values_decoded_or_parsed": 0,
               "timestamp_values_parsed": 0, "scientific_model_fits": 0,
               "absolute_historical_never_seen_claim": False}
    write_json(ROOT / f"inputs/data41/{task}/public_source41.json", receipt)
    return receipt


@lru_cache(maxsize=4)
def _member_bytes(relative, archive_sha256, member):
    raw = (ROOT / relative).read_bytes()
    if sha_bytes(raw) != archive_sha256:
        raise RuntimeError("Original official archive changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        if zipped.namelist() != [member]:
            raise RuntimeError("Source archive member roster differs: " + repr(zipped.namelist()))
        return zipped.read(member)


def source_payloads(task):
    spec = TASKS[task]
    source = json.loads((ROOT / f"inputs/data41/{task}/public_source41.json").read_text())
    if source["task"] != task or source["source_manifest_sha256"] != sha_bytes(canonical_bytes(source_manifest(source["archives"]))):
        raise RuntimeError("Source manifest identity differs")
    if [a["archive_path"] for a in source["archives"]] != archive_relatives(task):
        raise RuntimeError("Original source archive roster differs")
    payloads = {}
    for declared, item in zip(spec["archives"], source["archives"]):
        if item["source_id"] != declared["source_id"] or item["member"] != declared["member"]:
            raise RuntimeError("Original source member identity differs")
        if sha_file(ROOT / item["archive_path"]) != item["archive_sha256"]:
            raise RuntimeError("Original official archive changed")
        payloads[item["source_id"]] = _member_bytes(item["archive_path"], item["archive_sha256"], item["member"])
    return payloads, source


def parse_timestamp(task, cells):
    spec = TASKS[task]
    # Only date/time fields cross the decoding boundary.
    text = " ".join(cells[i].decode("ascii").strip() for i in spec["timestamp_indices"])
    for fmt in spec["timestamp_formats"]:
        try:
            stamp = datetime.strptime(text, fmt)
            break
        except ValueError:
            pass
    else:
        raise RuntimeError("Source timestamp format differs")
    if stamp.second or stamp.minute % spec["cadence_minutes"]:
        raise RuntimeError("Source timestamp cadence differs")
    return stamp


def calendar_projection(task, payloads):
    spec = TASKS[task]
    lower = datetime(spec["year"], 1, 1) - timedelta(days=2)
    upper = datetime(spec["year"] + 1, 1, 1)
    groups, headers, member_stats = {}, None, []
    previous = None
    first = None
    native_rows, native_unique, duplicated_native = 0, 0, 0
    by_year, missing_status_counts = Counter(), Counter()
    markers = {x.encode("ascii") for x in spec["missing_markers"]}
    for source_id, raw in payloads.items():
        stream = io.BytesIO(raw)
        heading = stream.readline()
        current_headers = [x.decode(spec["header_encoding"]) for x in byte_fields(heading.removeprefix(b"\xef\xbb\xbf"), spec["separator"])]
        if headers is None:
            headers = current_headers
        if headers != current_headers or headers[spec["target_index"]] != spec["target_header"]:
            raise RuntimeError("Source target schema or cross-archive header differs")
        expected_time_headers = ["Date", "Time"] if task == "household_power2010" else ["Date Time"]
        if [headers[i] for i in spec["timestamp_indices"]] != expected_time_headers:
            raise RuntimeError("Source timestamp schema differs")
        row_count = 0
        for line_number, record in enumerate(stream, 2):
            stop = stream.tell(); start = stop - len(record)
            # Semicolon source has no quoted fields; splitting bytes is opaque.
            cells = record.rstrip(b"\r\n").split(b";") if spec["separator"] == ";" else byte_fields(record)
            if len(cells) != len(headers):
                raise RuntimeError("Source row field count differs")
            stamp = parse_timestamp(task, cells)
            if previous is not None and stamp < previous:
                raise RuntimeError("Source timestamps not in chronological archive order")
            native_rows += 1; row_count += 1
            if previous == stamp:
                duplicated_native += 1
            else:
                native_unique += 1; by_year[str(stamp.year)] += 1
            previous = stamp
            if first is None:
                first = stamp
            if stamp.minute or not lower <= stamp < upper:
                continue
            iso = stamp.isoformat(timespec="seconds")
            missing = cells[spec["target_index"]].strip(b" \t") in markers if markers else False
            if markers:
                missing_status_counts["declared_missing" if missing else "not_declared_missing"] += 1
            ref = {"source_id": source_id, "source_line": line_number, "byte_start": start, "byte_stop": stop}
            item = groups.setdefault(iso, {"timestamp": iso, "records": [], "declared_target_missing": False})
            item["records"].append(ref)
            item["declared_target_missing"] = item["declared_target_missing"] or missing
        member_stats.append({"source_id": source_id, "member_bytes": len(raw), "member_sha256": sha_bytes(raw), "timestamp_rows": row_count})
    if not groups or previous is None or first is None:
        raise RuntimeError("No eligible source calendar window")
    rows = list(groups.values())
    expected_native = int((previous - first) / timedelta(minutes=spec["cadence_minutes"])) + 1
    return {"schema": "step41-calendar-status-byte-projection-v1", "task": task,
            "headers": headers, "parsed_data_columns": spec["timestamp_indices"],
            "target_field_index": spec["target_index"], "target_values_decoded_or_parsed": 0,
            "other_measurement_values_decoded_or_parsed": 0,
            "target_missing_status_inspected": bool(markers), "missing_status_rule": spec["missing_status_rule"],
            "declared_missing_markers": spec["missing_markers"], "hourly_target_status_counts": dict(missing_status_counts),
            "rows": rows, "member_stats": member_stats, "timestamp_rows": native_rows,
            "unique_native_timestamps": native_unique, "duplicate_record_excess": duplicated_native,
            "expected_native_timestamps": expected_native, "missing_native_timestamps": expected_native - native_unique,
            "first_source_timestamp": first.isoformat(timespec="seconds"), "last_source_timestamp": previous.isoformat(timespec="seconds"),
            "source_timestamp_counts_by_year": dict(sorted(by_year.items())),
            "projected_hourly_timestamps": len(rows), "projected_first_timestamp": rows[0]["timestamp"],
            "projected_last_timestamp": rows[-1]["timestamp"],
            "projected_window_start": lower.isoformat(timespec="seconds"), "projected_window_stop_exclusive": upper.isoformat(timespec="seconds"),
            "native_cadence_minutes": spec["cadence_minutes"], "timezone_convention": spec["timezone_convention"],
            "hourly_snapshot_rule": spec["snapshot_description"], "unit": spec["unit"],
            "duplicate_target_agreement_examined": False,
            "duplicate_target_policy": "Declared missing status at any duplicate excludes timestamp; otherwise authorized target values must agree exactly or STOP_RETAIN_FAILURE",
            "no_value_based_row_filtering": True, "scientific_model_fits": 0}


def project(task):
    _no_frozen_protocol()
    payloads, source = source_payloads(task)
    result = calendar_projection(task, payloads)
    result.update({"source_manifest_sha256": source["source_manifest_sha256"],
                   "projection_utc": datetime.now(timezone.utc).isoformat(),
                   "projection_source_sha256": sha_file(Path(__file__))})
    path = ROOT / f"inputs/data41/{task}/calendar_projection41.json"
    write_json(path, result)
    metadata = {k: v for k, v in result.items() if k != "rows"}
    metadata["projection_sha256"] = sha_file(path)
    metadata["catalog_reported_instances"] = TASKS[task]["reported_instances"]
    reported = TASKS[task]["reported_instances"]
    metadata["catalog_count_matches_data_rows"] = None if reported is None else result["timestamp_rows"] == reported
    metadata["source_transport_verification"] = {
        "transport": source.get("transport", "official_provider"),
        "original_member_names_and_headers_verified": True,
        "archive_bytes_retained_verbatim_and_hash_bound": True,
        "measurement_values_used_for_verification": False,
        "direct_official_archive_transport_confirmed": source.get("transport") != "public_repository_mirror",
        "identity_against_independent_official_checksum_verified": False,
        "verification_limits": source.get("verification_limits", "Retrieved directly from official provider; no external independent archive checksum supplied"),
    }
    write_json(ROOT / f"inputs/data41/{task}/calendar_metadata41.json", metadata)
    return metadata


def inherit_household():
    """Copy the already retained original-format archive, without decoding values."""
    _no_frozen_protocol()
    task = "household_power2010"
    prior_root = ROOT / "recovered/step40/step40_head_selection"
    prior_path = prior_root / "inputs/household_power2009/public_source40.json"
    prior = json.loads(prior_path.read_text())
    spec = TASKS[task]
    dest = ROOT / f"inputs/data41/{task}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "public_source41.json").exists():
        raise RuntimeError("Source inheritance is write-once")
    archives = []
    for item, relative in zip(prior["archives"], archive_relatives(task)):
        src = prior_root / item["archive_path"]
        if sha_file(src) != item["archive_sha256"]:
            raise RuntimeError("Retained Step40 source archive changed")
        if (ROOT / relative).exists():
            raise RuntimeError("Inherited archive destination already exists")
        shutil.copyfile(src, ROOT / relative)
        archives.append({**item, "archive_path": relative})
    supplementary = {}
    for relative, expected in prior.get("supplementary_metadata_files", {}).items():
        src = prior_root / relative
        if sha_file(src) != expected:
            raise RuntimeError("Retained Step40 provenance evidence changed")
        new_relative = f"inputs/data41/{task}/inherited_" + Path(relative).name
        shutil.copyfile(src, ROOT / new_relative)
        supplementary[new_relative] = expected
    old_meta = prior_root / prior["metadata_path"]
    if sha_file(old_meta) != prior["metadata_sha256"]:
        raise RuntimeError("Retained official UCI metadata changed")
    meta_relative = f"inputs/data41/{task}/official_metadata_retrieval40.json"
    shutil.copyfile(old_meta, ROOT / meta_relative)
    prior_relative = f"inputs/data41/{task}/inherited_public_source40.json"
    shutil.copyfile(prior_path, ROOT / prior_relative)
    supplementary[prior_relative] = sha_file(prior_path)
    record = {**prior, **{k: v for k, v in spec.items() if k != "archives"},
              "schema": "step41-retained-source-forward-cohort-v1", "task": task,
              "archives": archives, "metadata_path": meta_relative,
              "supplementary_metadata_files": supplementary,
              "source_manifest_sha256": sha_bytes(canonical_bytes(source_manifest(archives))),
              "source_inheritance_utc": datetime.now(timezone.utc).isoformat(),
              "source_inheritance": "Verbatim previously retained archive bytes; current cohort changes from2009 to2010",
              "prior_cohort": "household_power2009",
              "new_cohort_measurement_values_decoded_or_parsed": 0,
              "new_cohort_fit_count": 0,
              "historical_archive_byte_access": True,
              "freshness_scope": "New calendar period within previously accessed source archive; not a new dataset or unseen raw byte claim"}
    write_json(dest / "public_source41.json", record)
    return record


def prior_exposure():
    """Inspect retained method/result traces, never current numerical source cells."""
    _no_frozen_protocol()
    workspace = ROOT.parent
    scopes = [(workspace, False), (ROOT / "recovered/step40/step40_head_selection", True)]
    bound, matches, bytes_searched = {}, {}, 0
    patterns = {
        "household_power2010_exact_task_id": rb"household[_ -]?power[_ -]?2010",
        "jena_temperature2017_exact_task_id": rb"jena[_ -]?temperature[_ -]?2017",
        "jena2017_source_archive": rb"mpi_roof_2017[ab]|jena2017[ab]",
    }
    matches = {key: [] for key in patterns}
    for scan_root, is_recovered40 in scopes:
        listing = subprocess.run(["rg", "--files", "-g", "*.py", "-g", "*.txt", "-g", "*.md", "-g", "*.tex", "-g", "*.json", "-g", "!**/tmp/**", "-g", "!**/__pycache__/**"],
                                 cwd=scan_root, capture_output=True, text=True, check=True)
        for text in sorted(listing.stdout.splitlines()):
            path = Path(text)
            if not is_recovered40 and path.parts[0] == ROOT.name:
                continue
            name = path.name.lower()
            if not ("source" in path.parts or "reporting" in path.parts or "audit" in path.parts or any(t in name for t in ("readme", "protocol", "source", "exposure", "summary", "report", "result", "metadata"))):
                continue
            if any(t in str(path).lower() for t in ("private_library", "patient", "reserved", "clinical", "library_helpers")):
                continue
            actual = scan_root / path
            if actual.stat().st_size > 8 * 1024 * 1024:
                continue
            raw = actual.read_bytes()
            identity = str(actual.relative_to(workspace))
            if identity in bound:
                continue
            bound[identity] = sha_bytes(raw)
            bytes_searched += len(raw)
            for key, pattern in patterns.items():
                if re.search(pattern, raw, re.IGNORECASE):
                    matches[key].append(identity)
    previous = ROOT / "recovered/step40/step40_head_selection"
    access_path = previous / "output/access_and_execution40.jsonl"
    accesses = []
    for line in access_path.read_text().splitlines():
        item = json.loads(line)
        if item.get("event", "").endswith("measurement_access_completed") or item.get("event") in ("test_history_access_completed", "test_target_revealed"):
            accesses.append({k: item[k] for k in ("event", "task", "stage", "query_id", "maximum_decoded_timestamp") if k in item})
    record = {"schema": "step41-scoped-forward-cohort-exposure-v1", "utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Retained method/report/protocol/public-source/audit/metadata text files at most8MiB, including recovered Step40, excluding currentStep41 except recoveredprior, tmp/private/patient/reserved/clinical and raw arrays/CSV/archives",
              "file_count": len(bound), "bytes_searched": bytes_searched, "file_sha256": bound,
              "patterns": {k: p.decode("ascii") for k, p in patterns.items()}, "matches": matches,
              "prior_step40_access_log_sha256": sha_file(access_path), "prior_step40_access_trace": accesses,
              "prior_step40_declared_tasks": ["household_power2009", "jena_temperature2016"],
              "selected_forward_tasks": list(TASKS),
              "freshness_claim": "Forward calendar periods relative to Step40; absence of exact-cohort evidence in explicitly scoped retained record, not historical never-seen or independent-domain status",
              "known_source_exposure": "Both domains were selected and evaluated in Step40; all Household archive bytes including2010 were already retained and timestamp-scanned, but the declared Step40 numeric loaders authorized only2009 cohort/history. Jena2017 archives are first retrieved forStep41.",
              "incidental_metadata_exposure": "Official station documentation contains unrelated illustrative2004 rows, already visible inStep40; no selected2017 measurements read from it",
              "private_or_reserved_arrays_read": False,
              "selected_cohort_measurements_decoded_before_freeze": 0,
              "absolute_historical_never_seen_claim": False}
    write_json(ROOT / "inputs/data41/prior_exposure41.json", record)
    return {k: v for k, v in record.items() if k not in ("file_sha256", "prior_step40_access_trace")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inherit-household", "download", "project", "prior-exposure"))
    parser.add_argument("task", nargs="?", choices=tuple(TASKS))
    args = parser.parse_args()
    if args.mode == "inherit-household":
        result = inherit_household()
    elif args.mode == "prior-exposure":
        result = prior_exposure()
    else:
        if args.task is None:
            parser.error("task required")
        if args.mode == "download" and args.task != "jena_temperature2017":
            parser.error("Household must use the retained inherited archive")
        result = download(args.task) if args.mode == "download" else project(args.task)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


