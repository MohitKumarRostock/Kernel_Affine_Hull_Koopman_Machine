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
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "household_power2009": {
        "dataset": "Individual Household Electric Power Consumption", "year": 2009,
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
    "jena_temperature2016": {
        "dataset": "MPI-BGC Beutenberg roof weather station", "year": 2016,
        "provider": "Max Planck Institute for Biogeochemistry", "uci_id": None, "doi": None,
        "metadata_url": "https://www.bgc-jena.mpg.de/wetter/weather_data.html",
        "documentation_url": "https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf",
        "citation": "Max Planck Institute for Biogeochemistry. Beutenberg roof weather station, 2016 half-year archives. https://www.bgc-jena.mpg.de/wetter/weather_data.html. Documentation: Olaf Kolle (4 February 2025).",
        "source_license": "CC BY 4.0",
        "transport_policy": "Direct original archives from official MPI-BGC provider",
        "archives": [
            {"source_id": "jena2016a", "filename": "mpi_roof_2016a.zip", "member": "mpi_roof_2016a.csv", "metadata_link": "data/mpi_roof_2016a.zip"},
            {"source_id": "jena2016b", "filename": "mpi_roof_2016b.zip", "member": "mpi_roof_2016b.csv", "metadata_link": "data/mpi_roof_2016b.zip"},
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
    return [f"inputs/{task}/{a['filename']}" for a in TASKS[task]["archives"]]


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
    if (ROOT / "protocol_seal40.json").exists():
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
    metadata_path = f"inputs/{task}/official_metadata.html"
    (ROOT / metadata_path).write_bytes(page)
    receipt = {"schema": "step40-opaque-public-source-download-v1", "task": task,
               **{k: v for k, v in spec.items() if k != "archives"}, "archives": archives,
               "source_manifest_sha256": sha_bytes(canonical_bytes(source_manifest(archives))),
               "metadata_path": metadata_path, "metadata_sha256": sha_bytes(page), "metadata_html_sha256": sha_bytes(page),
               "metadata_resolved_url": resolved_metadata,
               "download_utc": datetime.now(timezone.utc).isoformat(),
               "download_elapsed_seconds": time.perf_counter() - started,
               "stage": "OPAQUE_ARCHIVES_ONLY", "measurement_values_decoded_or_parsed": 0,
               "timestamp_values_parsed": 0, "scientific_model_fits": 0,
               "absolute_historical_never_seen_claim": False}
    write_json(ROOT / f"inputs/{task}/public_source40.json", receipt)
    return receipt


def download_household_mirror():
    """Opaque pinned transport fallback, selected before any numeric decoding."""
    _no_frozen_protocol()
    task = "household_power2009"; spec = TASKS[task]
    relative = archive_relatives(task)[0]
    if (ROOT / relative).exists():
        raise RuntimeError("Pinned mirror archive already retained; no redownload")
    metadata_path = f"inputs/{task}/official_metadata_retrieval40.json"
    metadata = json.loads((ROOT / metadata_path).read_text())
    if metadata.get("source_url") != spec["metadata_url"] or metadata.get("scientific_data_measurement_values_decoded_or_parsed") != 0:
        raise RuntimeError("Retained official UCI metadata retrieval required")
    commit_metadata_path = f"inputs/{task}/mirror_commit_metadata.json"
    commits = json.loads((ROOT / commit_metadata_path).read_text())
    if not commits or commits[0].get("sha") != spec["mirror_commit"]:
        raise RuntimeError("Pinned public mirror commit differs")
    url = f"https://raw.githubusercontent.com/{spec['mirror_repository']}/{spec['mirror_commit']}/household_power_consumption.zip"
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=60) as response:
        raw, resolved = response.read(), response.geturl()
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        inventory = [{"name": a.filename, "bytes": a.file_size, "compressed_bytes": a.compress_size} for a in zipped.infolist()]
    path = ROOT / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    archives = [{**spec["archives"][0], "archive_path": relative, "archive_sha256": sha_bytes(raw),
                 "archive_bytes": len(raw), "source_url": url, "resolved_url": resolved, "inventory": inventory}]
    receipt = {"schema": "step40-opaque-public-source-download-v1", "task": task,
               **{k: v for k, v in spec.items() if k != "archives"}, "archives": archives,
               "source_manifest_sha256": sha_bytes(canonical_bytes(source_manifest(archives))),
               "scientific_source_url": spec["metadata_url"], "transport": "public_repository_mirror",
               "mirror_repository_url": "https://github.com/" + spec["mirror_repository"],
               "mirror_commit_url": commits[0]["html_url"],
               "metadata_path": metadata_path, "metadata_sha256": sha_file(ROOT / metadata_path),
               "metadata_retrieval_surface": metadata["retrieval_surface"],
               "supplementary_metadata_files": {commit_metadata_path: sha_file(ROOT / commit_metadata_path),
                                                "inputs/source_transport_failures40.json": sha_file(ROOT / "inputs/source_transport_failures40.json")},
               "verification_limits": "Mirror archive retained verbatim and hash-sealed; structural header/count/time-coverage checks follow without value decoding. Identity with unreachable official archive is not independently established; no official archive hash was available.",
               "download_utc": datetime.now(timezone.utc).isoformat(), "download_elapsed_seconds": time.perf_counter() - started,
               "stage": "OPAQUE_ARCHIVE_ONLY", "measurement_values_decoded_or_parsed": 0,
               "timestamp_values_parsed": 0, "scientific_model_fits": 0,
               "absolute_historical_never_seen_claim": False}
    write_json(ROOT / f"inputs/{task}/public_source40.json", receipt)
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
    source = json.loads((ROOT / f"inputs/{task}/public_source40.json").read_text())
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
        expected_time_headers = ["Date", "Time"] if task == "household_power2009" else ["Date Time"]
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
    return {"schema": "step40-calendar-status-byte-projection-v1", "task": task,
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
    path = ROOT / f"inputs/{task}/calendar_projection40.json"
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
    write_json(ROOT / f"inputs/{task}/calendar_metadata40.json", metadata)
    return metadata


def prior_exposure():
    _no_frozen_protocol()
    workspace = ROOT.parent
    listing = subprocess.run(["rg", "--files", "-g", "*.py", "-g", "*.txt", "-g", "*.md", "-g", "*.tex", "-g", "*.json", "-g", "!**/tmp/**", "-g", "!**/__pycache__/**"], cwd=workspace, capture_output=True, text=True, check=True)
    included = []
    for text in listing.stdout.splitlines():
        path = Path(text); name = path.name.lower()
        if path.parts[0] == ROOT.name:
            continue
        if not ("source" in path.parts or "reporting" in path.parts or "audit" in path.parts or any(t in name for t in ("readme", "protocol", "source", "exposure", "summary", "report", "result", "metadata"))):
            continue
        if any(t in str(path).lower() for t in ("private_library", "patient", "reserved", "clinical", "library_helpers")):
            continue
        if (workspace / path).stat().st_size <= 8 * 1024 * 1024:
            included.append(path)
    patterns = {
        "household_power2009": rb"individual[ _+]?household|household_power_consumption|global_active_power|uci[ _]?235|10\.24432/C58K54",
        "jena_temperature2016": rb"jena[ _+-]?(?:climate|temperature)|MPI[_ -]?BGC|beutenberg|mpi_roof",
        "excluded_capital_bike": rb"capital[ _+-]?bikeshare|bike[ _+-]?sharing[ _+-]?dataset|uci[ _]?275|10\.24432/C5W894|Bike-Sharing-Dataset|\bhour\.csv\b",
        "excluded_seoul": rb"SeoulBikeData|seoul[ _+]?bike|10\.24432/C5F62R",
        "excluded_appliances": rb"appliances[ _+]?energy|energydata_complete|10\.24432/C5VC8G",
        "excluded_beijing": rb"beijing[ _+]?pm2|beijing_pm25|10\.24432/C5JS49",
        "excluded_tetouan": rb"tet(?:ou|u)an|10\.24432/C5B034",
        "excluded_metro": rb"metro[ _+]?interstate|metro_interstate_traffic|10\.24432/C5X60B",
    }
    matches, bound, total = {k: [] for k in patterns}, {}, 0
    for path in sorted(included):
        raw = (workspace / path).read_bytes(); total += len(raw); bound[str(path)] = sha_bytes(raw)
        for key, pattern in patterns.items():
            if re.search(pattern, raw, re.IGNORECASE):
                matches[key].append(str(path))
    record = {"schema": "step40-scoped-prior-exposure-search-v1", "utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Retained method/report/protocol/public-source/audit/metadata text files at most8MiB; excludes currentStep40,tmp,private-helper/patient/reserved/clinical paths and raw numerical array/CSV/archive formats",
              "file_count": len(bound), "bytes_searched": total, "file_sha256": bound,
              "patterns": {k: v.decode("ascii") for k, v in patterns.items()}, "matches": matches,
              "freshness_claim": "No selected-source match in the explicitly scoped retained record; not an unrestricted historical never-seen claim",
              "excluded_known_prior_sources": ["Capital Bikeshare Step36", "Seoul Step38", "Appliances Step24", "Beijing Step35", "Tetouan Step39", "Metro Step39"],
              "incidental_metadata_exposure": "Official Jena documentation includes four illustrative2004 measurement rows; mirror-search snippets include five illustrativeDecember2006 Household rows. Neither selected2016Jena nor2009Household measurements were opened. UCI metadata inspected only.",
              "private_or_reserved_arrays_read": False, "selected_cohort_measurements_decoded_before_freeze": 0}
    write_json(ROOT / "inputs/prior_exposure40.json", record)
    return {k: v for k, v in record.items() if k != "file_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("download", "download-household-mirror", "project", "prior-exposure"))
    parser.add_argument("task", nargs="?", choices=tuple(TASKS))
    args = parser.parse_args()
    if args.mode == "download-household-mirror":
        result = download_household_mirror()
    elif args.mode == "prior-exposure":
        result = prior_exposure()
    else:
        if args.task is None:
            parser.error("task required")
        result = download(args.task) if args.mode == "download" else project(args.task)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
