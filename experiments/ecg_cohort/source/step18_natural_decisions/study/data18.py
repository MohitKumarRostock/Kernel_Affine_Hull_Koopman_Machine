"""Frozen, patient-separated PTB-XL data contract for Step 18.

Metadata preparation and opaque archive acquisition are permitted before the
research freeze. Natural waveform/header decoding requires a matching source
freeze. Manufacturing tests supply their own in-memory bytes. This module never
downloads individual records and never replaces a failed record with another.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
STEP_ROOT = STUDY_DIR.parent
WORKSPACE = STEP_ROOT.parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))
from acquire18 import (DEFAULT_ARCHIVE_PATH, DEFAULT_RECEIPT_PATH,
                       EXPECTED_BYTES, acquire_archive, sha256_file)

REFERENCE_DIR = STEP_ROOT / "references" / "ptbxl_v103"
ROSTER_PATH = REFERENCE_DIR / "patient_roster18.csv"
PROTOCOL_PATH = REFERENCE_DIR / "data_protocol18.json"
SOURCE_MANIFEST_PATH = REFERENCE_DIR / "source_manifest18.json"
DEFAULT_OUTPUT_ROOT = STUDY_DIR / "run" / "data"
HISTORY_OFFSETS = np.arange(-155, 1, 5, dtype=np.int64)
HORIZON = 20
TARGET_CLIP_MV = 2.0
STAGE_COUNTS = {"support": 512, "train": 3072, "validation": 1536,
                "calibration": 4096, "terminal": 4096, "reserve": 5557}
SPLIT_SEED = 2026110618
EVALUATION_SEED = 2026110619
ROSTER_FIELDS = ["patient_id", "ecg_id", "filename_hr", "origin", "stage",
                 "stage_index", "evaluation_pool_index", "evaluation_order"]
DATA_PROTOCOL = {
    "schema": "step18-data-protocol-v1", "dataset": "PTB-XL", "version": "1.0.3",
    "record_selection": "Minimum integer ecg_id per integer patient_id; no chronological interpretation",
    "sampling_hz": 500, "expected_samples": 5000, "expected_channels": 12,
    "lead": "II", "history_offsets": HISTORY_OFFSETS.tolist(),
    "history_units": "mV", "history_clipping": "none",
    "history_filtering": "none; causal raw-sample subsampling at the listed offsets",
    "horizon_samples": HORIZON, "horizon_seconds": 0.04,
    "target": "clip(lead_II_mV[origin+20], -2, 2)/2",
    "target_bounds": [-1.0, 1.0], "target_clip_mV": TARGET_CLIP_MV,
    "origin": {"hash_input": "step18-origin-v1:" + "<canonical integer patient_id>",
               "algorithm": "SHA256 digest interpreted as unsigned big-endian integer",
               "formula": "155 + digest_integer % 4825", "inclusive_range": [155, 4979]},
    "split": {"input_order": "sorted integer patient IDs", "rng": "numpy.random.default_rng",
              "seed": SPLIT_SEED, "sequential_counts": {"support": 512, "train": 3072,
              "validation": 1536, "evaluation": 8192, "reserve": 5557},
              "evaluation_rng_seed": EVALUATION_SEED,
              "evaluation_order": "Independent permutation of the ordered 8192 evaluation IDs; first 4096 calibration, remainder terminal",
              "stage_counts": STAGE_COUNTS, "calibration_prefixes": [512, 2048, 4096]},
    "support_target_policy": "Do not index, transform, return or store a target for support patients",
    "eligibility": "Every metadata patient; no outcome-based exclusion or missing-record substitution",
    "failure_policy": "Missing, malformed, unsupported or integrity-failing required record stops the stage visibly",
    "stage_storage": "Write-once NPZ plus per-record integrity receipt; reserve extraction forbidden",
    "clinical_scope": "Technical bounded waveform forecast only; no diagnostic or clinical adequacy claim",
}


class DataIntegrityError(RuntimeError):
    pass


def _json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _canonical_integer(value):
    number = Decimal(str(value))
    if not number.is_finite() or number != number.to_integral_value():
        raise DataIntegrityError(f"Non-integral identifier: {value!r}")
    return int(number)


def _hash_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def _official_checksums():
    checksums = {}
    for line in (REFERENCE_DIR / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise DataIntegrityError("Malformed official checksum entry")
        name = parts[1].lstrip("*")
        if name in checksums:
            raise DataIntegrityError(f"Duplicate checksum name: {name}")
        checksums[name] = parts[0]
    return checksums


def prepare_roster():
    """Metadata-only, write-once construction of every split and origin."""
    for path in (ROSTER_PATH, PROTOCOL_PATH, SOURCE_MANIFEST_PATH):
        if path.exists():
            raise FileExistsError(f"Metadata design is write-once: {path}")
    checksums = _official_checksums()
    for name in ("ptbxl_database.csv", "LICENSE.txt"):
        if sha256_file(REFERENCE_DIR / name) != checksums[name]:
            raise DataIntegrityError(f"Official metadata/license hash mismatch: {name}")
    patients = {}
    record_ids = set()
    with (REFERENCE_DIR / "ptbxl_database.csv").open(newline="") as handle:
        for source in csv.DictReader(handle):
            # No clinical, quality, waveform-derived, demographic or label field is used.
            patient, ecg = _canonical_integer(source["patient_id"]), _canonical_integer(source["ecg_id"])
            name = source["filename_hr"]
            if ecg in record_ids:
                raise DataIntegrityError("Repeated ECG identifier")
            record_ids.add(ecg)
            if not re.fullmatch(r"records500/\d{5}/\d{5}_hr", name):
                raise DataIntegrityError(f"Unexpected original-rate filename: {name}")
            if any(name + extension not in checksums for extension in (".dat", ".hea")):
                raise DataIntegrityError(f"Record missing from official manifest: {name}")
            if patient not in patients or ecg < patients[patient]["ecg_id"]:
                patients[patient] = {"patient_id": patient, "ecg_id": ecg, "filename_hr": name}
    if len(patients) != 18869 or len(record_ids) != 21799:
        raise DataIntegrityError("Unexpected metadata population size")
    ids = np.asarray(sorted(patients), dtype=np.int64)
    permutation = np.random.default_rng(SPLIT_SEED).permutation(ids)
    groups = {"support": permutation[:512], "train": permutation[512:3584],
              "validation": permutation[3584:5120], "reserve": permutation[13312:]}
    pool = permutation[5120:13312]
    evaluation_order = np.random.default_rng(EVALUATION_SEED).permutation(len(pool))
    ordered_pool = pool[evaluation_order]
    groups["calibration"], groups["terminal"] = ordered_pool[:4096], ordered_pool[4096:]
    pool_indices = {int(patient): i for i, patient in enumerate(pool)}
    evaluation_indices = {int(patient): i for i, patient in enumerate(ordered_pool)}
    rows = []
    for stage in STAGE_COUNTS:
        if len(groups[stage]) != STAGE_COUNTS[stage]:
            raise DataIntegrityError("Internal split count mismatch")
        for index, value in enumerate(groups[stage]):
            patient = int(value)
            digest = hashlib.sha256(f"step18-origin-v1:{patient}".encode("ascii")).digest()
            origin = 155 + int.from_bytes(digest, "big") % 4825
            rows.append({**patients[patient], "origin": origin, "stage": stage,
                         "stage_index": index, "evaluation_pool_index": pool_indices.get(patient, -1),
                         "evaluation_order": evaluation_indices.get(patient, -1)})
    if len({row["patient_id"] for row in rows}) != 18869:
        raise DataIntegrityError("Split patient duplication or omission")
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with ROSTER_PATH.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROSTER_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _json_new(PROTOCOL_PATH, DATA_PROTOCOL)
    paths = [REFERENCE_DIR / name for name in ("ptbxl_database.csv", "LICENSE.txt", "SHA256SUMS.txt")]
    manifest = {"schema": "step18-source-manifest-v1", "prepared_unix": time.time(),
                "dataset_version": "1.0.3", "record_count": len(record_ids),
                "patient_count": len(patients), "stage_counts": STAGE_COUNTS,
                "metadata_columns_used": ["patient_id", "ecg_id", "filename_hr"],
                "metadata_only": True, "numpy_version": np.__version__,
                "files": {str(path.relative_to(WORKSPACE)): {"sha256": sha256_file(path),
                          "bytes": path.stat().st_size} for path in paths + [ROSTER_PATH, PROTOCOL_PATH]},
                "official_source": "https://physionet.org/content/ptb-xl/1.0.3/",
                "wfdb_spec_sources": ["https://wfdb.io/spec/header-files.html",
                                      "https://wfdb.io/spec/signal-files.html",
                                      "https://physionet.org/physiotools/wag/header-5.htm"]}
    _json_new(SOURCE_MANIFEST_PATH, manifest)
    return manifest


def source_files():
    """Required files for the root's workspace-relative `sources` freeze map."""
    paths = [Path(__file__).resolve(), STUDY_DIR / "acquire18.py", STUDY_DIR / "audit18_raw.py",
             STUDY_DIR / "manufactured_data18_tests.py", ROSTER_PATH,
             PROTOCOL_PATH, SOURCE_MANIFEST_PATH, DEFAULT_RECEIPT_PATH]
    paths += [REFERENCE_DIR / name for name in ("ptbxl_database.csv", "SHA256SUMS.txt", "LICENSE.txt")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Data sources are not yet ready to freeze: {path}")
    return paths


def verify_freeze(freeze_path):
    """Check the root's complete source closure before opening natural bodies."""
    freeze_path = Path(freeze_path).resolve()
    freeze = json.loads(freeze_path.read_text())
    sources = freeze.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise DataIntegrityError("Freeze must contain a nonempty sources path-to-SHA256 map")
    resolved = {}
    for name, expected in sources.items():
        path = Path(name)
        path = (WORKSPACE / path).resolve() if not path.is_absolute() else path.resolve()
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise DataIntegrityError(f"Invalid frozen SHA256 for {name}")
        if sha256_file(path) != expected:
            raise DataIntegrityError(f"Frozen source changed: {name}")
        resolved[path] = expected
    for path in source_files():
        if path.resolve() not in resolved:
            raise DataIntegrityError(f"Required data source absent from root freeze: {path}")
    if json.loads(PROTOCOL_PATH.read_text()) != DATA_PROTOCOL:
        raise DataIntegrityError("Serialized and executable data protocols disagree")
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    for name, information in manifest["files"].items():
        if sha256_file(WORKSPACE / name) != information["sha256"]:
            raise DataIntegrityError(f"Source-manifest hash mismatch: {name}")
    return {"path": str(freeze_path), "sha256": sha256_file(freeze_path)}


def _stage_name(stage):
    stage = {"val": "validation", "cal": "calibration", "test": "terminal"}.get(stage, stage)
    if stage not in STAGE_COUNTS or stage == "reserve":
        raise ValueError(f"Stage is not extractable: {stage!r}")
    return stage


def stage_roster(stage):
    stage = _stage_name(stage)
    with ROSTER_PATH.open(newline="") as handle:
        rows = [{key: (value if key in ("filename_hr", "stage") else int(value))
                 for key, value in row.items()} for row in csv.DictReader(handle) if row["stage"] == stage]
    if len(rows) != STAGE_COUNTS[stage] or [row["stage_index"] for row in rows] != list(range(len(rows))):
        raise DataIntegrityError(f"Incorrect or unordered roster for {stage}")
    return rows


_FORMAT = re.compile(r"(?P<format>\d+)(?:x(?P<spf>\d+))?(?::(?P<skew>\d+))?(?:\+(?P<offset>\d+))?\Z")
_GAIN = re.compile(r"(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
                   r"(?:\((?P<baseline>[+-]?\d+)\))?/(?P<units>[^\s]+)\Z")


def parse_header(header_bytes, expected_stem=None):
    """Parse the declared supported WFDB subset; unsupported forms fail closed.

    Supports multiplexed format16, one sample/frame, zero skew, shared byte
    offset, explicit positive gain/units, explicit or ADC-zero baseline.
    Header first values/checksums are retained only internally for validation.
    """
    try:
        text = header_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataIntegrityError("Header is not UTF-8/ASCII") from error
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise DataIntegrityError("Empty header")
    record = lines[0].split()
    if len(record) < 4 or "/" in record[0]:
        raise DataIntegrityError("Unsupported missing-field or multisegment record header")
    if expected_stem is not None and record[0] != expected_stem:
        raise DataIntegrityError("WFDB record name does not match metadata path")
    try:
        nsig, fs, nsamp = int(record[1]), float(record[2].split("/")[0]), int(record[3])
    except ValueError as error:
        raise DataIntegrityError("Malformed record dimensions") from error
    if nsig <= 0 or nsamp <= 0 or not math.isfinite(fs) or fs <= 0 or len(lines) != nsig + 1:
        raise DataIntegrityError("Invalid record dimensions or number of signal lines")
    channels = []
    for line in lines[1:]:
        fields = line.split(maxsplit=8)
        if len(fields) != 9:
            raise DataIntegrityError("All supported signal header fields must be explicit")
        file_name, format_text, gain_text = fields[:3]
        fmt, gain = _FORMAT.fullmatch(format_text), _GAIN.fullmatch(gain_text)
        if fmt is None or gain is None:
            raise DataIntegrityError("Unsupported format, gain, baseline or units field")
        if int(fmt["format"]) != 16 or int(fmt["spf"] or 1) != 1 or int(fmt["skew"] or 0) != 0:
            raise DataIntegrityError("Only format16, one sample/frame and zero skew are supported")
        try:
            adc_resolution, adc_zero, initial, checksum, blocksize = map(int, fields[3:8])
            scale = float(gain["gain"])
            baseline = int(gain["baseline"]) if gain["baseline"] is not None else adc_zero
        except ValueError as error:
            raise DataIntegrityError("Malformed ADC fields") from error
        units = gain["units"]
        mv_factor = {"mV": 1.0, "uV": 0.001, "µV": 0.001, "μV": 0.001, "V": 1000.0}.get(units)
        if not math.isfinite(scale) or scale <= 0 or mv_factor is None or blocksize != 0:
            raise DataIntegrityError("Unknown units, nonpositive gain or unsupported block size")
        if not (1 <= adc_resolution <= 16):
            raise DataIntegrityError("Unsupported ADC resolution")
        if Path(file_name).name != file_name or not file_name.endswith(".dat"):
            raise DataIntegrityError("Signal filename must be a local .dat basename")
        channels.append({"filename": file_name, "format": 16, "byte_offset": int(fmt["offset"] or 0),
                         "gain": scale, "baseline": baseline, "adc_zero": adc_zero,
                         "adc_resolution": adc_resolution, "units": units, "mv_factor": mv_factor,
                         "initial_value": initial, "checksum": checksum, "name": fields[8]})
    if len({(c["filename"], c["byte_offset"]) for c in channels}) != 1:
        raise DataIntegrityError("All channels must share one multiplexed file and offset")
    return {"record_name": record[0], "nsig": nsig, "fs": fs, "nsamp": nsamp, "channels": channels}


def decode_observation(header_bytes, signal_bytes, origin, include_target, expected_stem=None,
                       history_offsets=HISTORY_OFFSETS, horizon=HORIZON, validate_ptb=True):
    """Decode only the declared observation; no target index is read for support.

    Full digital channel sums/initial values are checked against WFDB headers,
    but no full physical-unit waveform array is produced or returned.
    """
    header = parse_header(header_bytes, expected_stem)
    nsig, nsamp = header["nsig"], header["nsamp"]
    if validate_ptb and (nsig != 12 or nsamp != 5000 or header["fs"] != 500.0):
        raise DataIntegrityError("Record is not the prescribed 12-lead, 500Hz, 5000-sample source")
    channels = header["channels"]
    if expected_stem is not None and channels[0]["filename"] != expected_stem + ".dat":
        raise DataIntegrityError("Header signal filename does not match record path")
    offset = channels[0]["byte_offset"]
    expected_bytes = offset + 2 * nsig * nsamp
    if len(signal_bytes) != expected_bytes:
        raise DataIntegrityError(f"Signal byte length mismatch: {len(signal_bytes)} != {expected_bytes}")
    digital = np.frombuffer(signal_bytes, dtype="<i2", count=nsig * nsamp, offset=offset).reshape(nsamp, nsig)
    sums = digital.sum(axis=0, dtype=np.int64)
    for j, channel in enumerate(channels):
        if int(digital[0, j]) != channel["initial_value"]:
            raise DataIntegrityError(f"WFDB initial value mismatch in channel {j}")
        if int(sums[j]) % 65536 != channel["checksum"] % 65536:
            raise DataIntegrityError(f"WFDB checksum mismatch in channel {j}")
    leads = [j for j, channel in enumerate(channels) if channel["name"].strip().upper() == "II"]
    if len(leads) != 1:
        raise DataIntegrityError("Exactly one lead named II is required")
    lead = leads[0]
    channel = channels[lead]
    indices = int(origin) + np.asarray(history_offsets, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= nsamp):
        raise DataIntegrityError("History indices outside record")
    if np.any(np.asarray(history_offsets) > 0):
        raise DataIntegrityError("History uses a future sample")
    input_digital = digital[indices, lead]
    if np.any(input_digital == -32768):
        raise DataIntegrityError("Missing WFDB sample in required history; no replacement permitted")
    X = ((input_digital.astype(np.float64) - channel["baseline"]) / channel["gain"]) * channel["mv_factor"]
    if not np.all(np.isfinite(X)):
        raise DataIntegrityError("Nonfinite physical history")
    audit = {"fs": header["fs"], "nsig": nsig, "nsamp": nsamp, "lead_index": lead,
             "lead_gain": channel["gain"], "lead_baseline": channel["baseline"],
             "lead_units": channel["units"], "lead_mV_factor": channel["mv_factor"],
             "byte_offset": offset, "format": 16, "all_initial_values_verified": True,
             "all_channel_checksums_verified": True,
             "input_values_outside_target_physical_interval": int(np.count_nonzero(np.abs(X) > TARGET_CLIP_MV)),
             "target_extracted": bool(include_target)}
    result = {"X": X, "audit": audit}
    if include_target:
        target_index = int(origin) + int(horizon)
        if target_index < 0 or target_index >= nsamp:
            raise DataIntegrityError("Target index outside record")
        target_digital = int(digital[target_index, lead])
        if target_digital == -32768:
            raise DataIntegrityError("Missing WFDB target; no replacement permitted")
        target_mv = ((target_digital - channel["baseline"]) / channel["gain"]) * channel["mv_factor"]
        if not math.isfinite(target_mv):
            raise DataIntegrityError("Nonfinite physical target")
        result["Y"] = float(np.clip(target_mv, -TARGET_CLIP_MV, TARGET_CLIP_MV) / TARGET_CLIP_MV)
        audit["target_clipped_low"] = int(target_mv < -TARGET_CLIP_MV)
        audit["target_clipped_high"] = int(target_mv > TARGET_CLIP_MV)
    return result


def _verify_archive(archive_path):
    receipt = json.loads(DEFAULT_RECEIPT_PATH.read_text())
    archive_path = Path(archive_path).resolve()
    if str(archive_path) != receipt["archive_path"]:
        raise DataIntegrityError("Archive path differs from frozen acquisition receipt")
    if archive_path.stat().st_size != EXPECTED_BYTES or receipt["bytes"] != EXPECTED_BYTES:
        raise DataIntegrityError("Archive byte count changed")
    if sha256_file(archive_path) != receipt["sha256"]:
        raise DataIntegrityError("Archive SHA256 differs from frozen transport pin")
    return receipt


def extract_stage(stage, *, freeze_path, output_root=DEFAULT_OUTPUT_ROOT,
                  archive_path=DEFAULT_ARCHIVE_PATH):
    """Extract this stage only, after root freeze and the caller's stage gates."""
    stage = _stage_name(stage)
    freeze = verify_freeze(freeze_path)  # Before ZIP opening or any header/sample access.
    archive_receipt = _verify_archive(archive_path)
    rows, checksums = stage_roster(stage), _official_checksums()
    stage_dir = Path(output_root) / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        arrays = {"X": np.empty((len(rows), 32), dtype=np.float64),
                  "patient_id": np.asarray([row["patient_id"] for row in rows], dtype=np.int64),
                  "ecg_id": np.asarray([row["ecg_id"] for row in rows], dtype=np.int64),
                  "origin": np.asarray([row["origin"] for row in rows], dtype=np.int64)}
        include_target = stage != "support"
        if include_target:
            arrays["Y"] = np.empty(len(rows), dtype=np.float64)
        records = []
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if len({information.filename for information in infos}) != len(infos):
                raise DataIntegrityError("Archive has duplicate member names")
            metadata_names = [info.filename for info in infos if info.filename == "ptbxl_database.csv"
                              or info.filename.endswith("/ptbxl_database.csv")]
            if len(metadata_names) != 1:
                raise DataIntegrityError("Cannot unambiguously determine archive member prefix")
            prefix = metadata_names[0][:-len("ptbxl_database.csv")]
            if _hash_bytes(archive.read(metadata_names[0])) != checksums["ptbxl_database.csv"]:
                raise DataIntegrityError("Archive metadata differs from official frozen metadata")
            members = {info.filename: info for info in infos}
            for index, row in enumerate(rows):
                paths = {kind: row["filename_hr"] + extension for kind, extension in (("header", ".hea"), ("signal", ".dat"))}
                bodies, identities = {}, {}
                for kind, relative in paths.items():
                    member = prefix + relative
                    if member not in members:
                        raise DataIntegrityError(f"Required archive member missing: {relative}")
                    bodies[kind] = archive.read(member)  # ZipFile also verifies the member CRC.
                    digest = _hash_bytes(bodies[kind])
                    if digest != checksums[relative]:
                        raise DataIntegrityError(f"Official member SHA256 mismatch: {relative}")
                    identities[kind] = {"relative_path": relative, "archive_member": member,
                                        "sha256": digest, "bytes": len(bodies[kind]),
                                        "zip_crc32": members[member].CRC}
                observation = decode_observation(bodies["header"], bodies["signal"], row["origin"],
                                                 include_target, Path(row["filename_hr"]).name)
                arrays["X"][index] = observation["X"]
                if include_target:
                    arrays["Y"][index] = observation["Y"]
                elif "Y" in observation:
                    raise DataIntegrityError("Support target exposure")
                records.append({"patient_id": row["patient_id"], "ecg_id": row["ecg_id"],
                                "origin": row["origin"], **identities, "decode_audit": observation["audit"]})
        artifact = stage_dir / "arrays.npz"
        with artifact.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        receipt = {"schema": "step18-stage-extraction-v1", "stage": stage,
                   "count": len(rows), "array_keys": sorted(arrays), "support_target_exposed": False,
                   "freeze": freeze, "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
                   "roster_sha256": sha256_file(ROSTER_PATH), "protocol_sha256": sha256_file(PROTOCOL_PATH),
                   "archive_sha256": archive_receipt["sha256"],
                   "archive_path": str(Path(archive_path).resolve()),
                   "artifact_sha256": sha256_file(artifact), "artifact_bytes": artifact.stat().st_size,
                   "started_unix": started, "completed_unix": time.time(),
                   "input_values_outside_target_physical_interval": sum(record["decode_audit"]["input_values_outside_target_physical_interval"] for record in records),
                   "target_clipping_counts": ({"below_minus_2mV": sum(record["decode_audit"]["target_clipped_low"] for record in records),
                                               "above_plus_2mV": sum(record["decode_audit"]["target_clipped_high"] for record in records)}
                                              if include_target else None),
                   "records": records}
        _json_new(stage_dir / "receipt.json", receipt)
        return receipt
    except Exception as error:
        _json_new(stage_dir / "FAILED.json", {"stage": stage, "error_type": type(error).__name__,
                  "error": str(error), "no_substitution_permitted": True, "failed_unix": time.time()})
        raise


def load_stage(stage, *, output_root=DEFAULT_OUTPUT_ROOT):
    """Load an already extracted stage; never triggers extraction or downloads."""
    stage = _stage_name(stage)
    stage_dir = Path(output_root) / stage
    if (stage_dir / "FAILED.json").exists():
        raise DataIntegrityError(f"Stage previously failed: {stage}")
    receipt = json.loads((stage_dir / "receipt.json").read_text())
    current_freeze = verify_freeze(receipt["freeze"]["path"])
    if current_freeze != receipt["freeze"]:
        raise DataIntegrityError("Stage root freeze changed")
    artifact = stage_dir / "arrays.npz"
    if sha256_file(artifact) != receipt["artifact_sha256"]:
        raise DataIntegrityError("Stage array artifact changed")
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    expected_keys = {"X", "patient_id", "ecg_id", "origin"} | ({"Y"} if stage != "support" else set())
    if set(arrays) != expected_keys:
        raise DataIntegrityError("Unexpected stage fields, including possible support target exposure")
    if arrays["X"].shape != (STAGE_COUNTS[stage], 32) or arrays["X"].dtype != np.float64 or not np.all(np.isfinite(arrays["X"])):
        raise DataIntegrityError("Invalid staged history array")
    rows = stage_roster(stage)
    for key in ("patient_id", "ecg_id", "origin"):
        expected = np.asarray([row[key] for row in rows], dtype=np.int64)
        if arrays[key].dtype != np.int64 or not np.array_equal(arrays[key], expected):
            raise DataIntegrityError(f"Staged {key} order differs from frozen roster")
    if stage != "support":
        Y = arrays["Y"]
        if Y.shape != (STAGE_COUNTS[stage],) or Y.dtype != np.float64 or not np.all(np.isfinite(Y)) or np.any(np.abs(Y) > 1):
            raise DataIntegrityError("Invalid bounded target array")
    arrays["receipt"] = receipt
    return arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-roster")
    commands.add_parser("acquire")
    extraction = commands.add_parser("extract")
    extraction.add_argument("stage", choices=[stage for stage in STAGE_COUNTS if stage != "reserve"])
    extraction.add_argument("--freeze", type=Path, required=True)
    extraction.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.command == "prepare-roster":
        print(json.dumps(prepare_roster(), indent=2))
    elif args.command == "acquire":
        acquire_archive()
    else:
        receipt = extract_stage(args.stage, freeze_path=args.freeze, output_root=args.output_root)
        print(json.dumps({key: value for key, value in receipt.items() if key != "records"}, indent=2))
