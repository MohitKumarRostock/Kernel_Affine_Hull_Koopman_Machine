"""Causal public-data loading with a separate, auditable test-value barrier.

The downloaded ZIP is opaque until the source/protocol seal exists. Development
loading interprets the first five timestamp columns for every row, but does not
split or interpret the remaining 2014 payload. Test loading requires the frozen
model and policy seal. No private or reserved-study data path is used here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def event(name, **fields):
    path = ROOT / "output/access_and_execution35.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if path.exists():
        lines = path.read_text().splitlines()
        if lines:
            previous = json.loads(lines[-1])["event_sha256"]
    record = {"event": name, "utc": datetime.now(timezone.utc).isoformat(),
              "monotonic_ns": time.monotonic_ns(), "previous_event_sha256": previous,
              **fields}
    record["event_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    with path.open("ab") as handle:
        handle.write(canonical_bytes(record))
        handle.flush()
    return record


def verify_protocol_seal():
    seal_path = ROOT / "protocol_seal35.json"
    if not seal_path.exists():
        raise RuntimeError("Protocol seal required before public numeric data access")
    seal = json.loads(seal_path.read_text())
    for relative, expected in seal["bound_files"].items():
        path = ROOT / relative
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError("Frozen protocol/source changed: " + relative)
    gate_path = ROOT / "audit/preexecution_gate35.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if gate.get("status") != "PASS" or gate.get("protocol_seal_sha256") != sha_file(seal_path):
        raise RuntimeError("Independent preexecution PASS required")
    return json.loads((ROOT / "protocol35.json").read_text())


def verify_model_seal():
    seal_path = ROOT / "output/model_seal35.json"
    if not seal_path.exists():
        raise RuntimeError("Model and policy seal required before test-value parsing")
    seal = json.loads(seal_path.read_text())
    if seal["protocol_seal_sha256"] != sha_file(ROOT / "protocol_seal35.json"):
        raise RuntimeError("Model seal belongs to another protocol")
    for relative, expected in seal["bound_files"].items():
        if sha_file(ROOT / relative) != expected:
            raise RuntimeError("Frozen model/policy input changed: " + relative)
    return seal


def _read_values(maximum_year):
    """Read PM values only for years <=maximum_year; skip other payload text."""
    if maximum_year not in (2013, 2014):
        raise ValueError("Only declared development and test barriers are supported")
    archive = ROOT / "inputs/beijing_pm25_uci381.zip"
    source = json.loads((ROOT / "inputs/public_source35.json").read_text())
    if sha_file(archive) != source["archive_sha256"]:
        raise RuntimeError("Public archive identity changed")
    values, all_timestamps, metadata_counts = {}, set(), {}
    with zipfile.ZipFile(archive) as zipped:
        names = [n for n in zipped.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX/")]
        if names != ["PRSA_data_2010.1.1-2014.12.31.csv"]:
            raise ValueError("Unexpected archive member roster: " + repr(names))
        with zipped.open(names[0]) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            header = stream.readline().rstrip("\r\n").split(",")
            if header[:6] != ["No", "year", "month", "day", "hour", "pm2.5"]:
                raise ValueError("Unexpected fixed timestamp/PM prefix")
            for line_number, line in enumerate(stream, start=2):
                prefix = line.rstrip("\r\n").split(",", 5)
                if len(prefix) != 6:
                    raise ValueError("Malformed metadata prefix at line " + str(line_number))
                _, year, month, day, hour, opaque_payload = prefix
                stamp = datetime(int(year), int(month), int(day), int(hour))
                if stamp.year not in range(2010, 2015) or stamp in all_timestamps:
                    raise ValueError("Unexpected or duplicated timestamp")
                all_timestamps.add(stamp)
                metadata_counts[str(stamp.year)] = metadata_counts.get(str(stamp.year), 0) + 1
                if stamp.year > maximum_year:
                    # The PM field and test missingness indicator are not inspected.
                    continue
                token = opaque_payload.split(",", 1)[0]
                if token == "NA":
                    values[stamp] = None
                else:
                    # UCI declares integer PM2.5; reject changed parsing semantics.
                    number = int(token)
                    if str(number) != token or number < 0:
                        raise ValueError("Unexpected PM value syntax or negative value")
                    values[stamp] = number
    expected = {str(y): (8784 if y == 2012 else 8760) for y in range(2010, 2015)}
    if metadata_counts != expected or len(all_timestamps) != 43824:
        raise ValueError("Source chronology differs from the declared dataset")
    return values, {"timestamp_rows_by_year": metadata_counts,
                    "numeric_years_parsed": list(range(2010, maximum_year + 1)),
                    "test_numeric_values_parsed": maximum_year == 2014,
                    "test_missingness_inspected": maximum_year == 2014,
                    "archive_sha256": source["archive_sha256"]}


def _cohort(values, target_years, name):
    origins, targets, rows, target_values = [], [], [], []
    exclusions = {"required_timestamp_absent": 0, "missing_lag_or_target": 0}
    potential = 0
    for target in sorted(values):
        if target.year not in target_years or target.hour != 23:
            continue
        potential += 1
        origin = target - timedelta(hours=24)
        lag_times = [origin - timedelta(hours=23-j) for j in range(24)]
        required = lag_times + [target]
        if any(stamp not in values for stamp in required):
            exclusions["required_timestamp_absent"] += 1
            continue
        if any(values[stamp] is None for stamp in required):
            exclusions["missing_lag_or_target"] += 1
            continue
        origins.append(origin.isoformat(timespec="hours"))
        targets.append(target.isoformat(timespec="hours"))
        rows.append([values[stamp] for stamp in lag_times])
        target_values.append(values[target])
    if not rows:
        raise ValueError("No eligible rows in " + name)
    result = {"name": name, "origin_timestamps": origins, "target_timestamps": targets,
              "ids": [name + ":" + x for x in targets],
              "X": np.asarray(rows, dtype=np.float64),
              "raw_targets": target_values,
              "roster": {"candidate_daily_targets": potential, "eligible": len(rows),
                         "excluded": exclusions, "target_years": list(target_years),
                         "first_target": targets[0], "last_target": targets[-1],
                         "target_id_sha256": hashlib.sha256(canonical_bytes(targets)).hexdigest()}}
    if potential != len(rows) + sum(exclusions.values()):
        raise AssertionError("Cohort accounting failed")
    return result


def _normalize(cohort, low, span):
    cohort["y"] = [Fraction(y-low, span) for y in cohort["raw_targets"]]
    cohort["persistence"] = [Fraction(int(x)-low, span) for x in cohort["X"][:, -1]]
    cohort["roster"]["target_outside_prediction_range_count"] = sum(y < 0 or y > 1 for y in cohort["y"])
    return cohort


def load_development():
    verify_protocol_seal()
    event("development_numeric_access_start", numeric_years_allowed=[2010, 2011, 2012, 2013],
          test_numeric_values_parsed=False, reserved_patient_arrays_read=False)
    values, receipt = _read_values(2013)
    train = _cohort(values, [2010, 2011, 2012], "train")
    decision = _cohort(values, [2013], "decision")
    low, high = min(train["raw_targets"]), max(train["raw_targets"])
    if high <= low:
        raise ValueError("Degenerate training target scale; frozen pilot cannot proceed")
    normalization = {"train_target_min": low, "train_target_max": high,
                     "train_target_range": high-low, "target_formula": "(raw-train_min)/train_range",
                     "predictor_interval": ["0", "1"], "targets_clipped": False,
                     "physical_mse_multiplier": (high-low)**2, "physical_units": "(microgram/m3)^2"}
    _normalize(train, low, high-low)
    _normalize(decision, low, high-low)
    receipt.update({"normalization": normalization, "train": train["roster"], "decision": decision["roster"],
                    "rolling_observed_history": True, "future_covariates": False,
                    "public_dataset_values_read": True, "reserved_patients_accessed": 0})
    write_json(ROOT / "output/development_data35.json", receipt)
    event("development_numeric_access_complete", receipt_sha256=sha_file(ROOT / "output/development_data35.json"),
          eligible_train=len(train["y"]), eligible_decision=len(decision["y"]), test_numeric_values_parsed=False)
    return train, decision, normalization


def load_test(normalization):
    verify_protocol_seal()
    seal = verify_model_seal()
    frozen_normalization = json.loads((ROOT / "output/development_data35.json").read_text())["normalization"]
    if normalization != frozen_normalization:
        raise RuntimeError("Test target normalization differs from the sealed development normalization")
    event("test_numeric_access_start", model_seal_sha256=sha_file(ROOT / "output/model_seal35.json"),
          all_policy_and_model_choices_frozen=True, reserved_patient_arrays_read=False)
    values, receipt = _read_values(2014)
    test = _cohort(values, [2014], "test")
    _normalize(test, normalization["train_target_min"], normalization["train_target_range"])
    receipt.update({"test": test["roster"], "normalization": normalization,
                    "model_seal_sha256": sha_file(ROOT / "output/model_seal35.json"),
                    "public_dataset_values_read": True, "reserved_patients_accessed": 0})
    write_json(ROOT / "output/test_data35.json", receipt)
    event("test_numeric_access_complete", receipt_sha256=sha_file(ROOT / "output/test_data35.json"),
          eligible_test=len(test["y"]))
    return test
