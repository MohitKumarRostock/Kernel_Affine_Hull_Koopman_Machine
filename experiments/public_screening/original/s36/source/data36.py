"""Causal Bike Sharing loading with a sealed, separately enforced test barrier.

Before the protocol gate, metadata_preflight() interprets only hour.csv's header,
dates and hours. The remaining payload is never split. Development loading leaves
all post-June-2012 payloads uninterpreted. day.csv is never opened. Count targets
share a calendar roster, but each model receives only that target's own 24 lags.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("casual", "registered", "cnt")
D = 24
ORIGIN_HOUR = 23
HORIZON_HOURS = 24
ARCHIVE_RELATIVE = "inputs/bike_sharing_uci275.zip"
MEMBER = "hour.csv"
HEADER = ("instant", "dteday", "season", "yr", "mnth", "hr", "holiday",
          "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
          "windspeed", "casual", "registered", "cnt")
PARTITIONS = {
    "train": (date(2011, 1, 1), date(2012, 1, 1)),
    "decision": (date(2012, 1, 1), date(2012, 7, 1)),
    "test": (date(2012, 7, 1), date(2013, 1, 1)),
}
DEVELOPMENT_CUTOFF = datetime(2012, 6, 30, 23)
SOURCE_FIRST = datetime(2011, 1, 1)
SOURCE_STOP = datetime(2013, 1, 1)


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
    path = ROOT / "output/access_and_execution36.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if path.exists():
        lines = path.read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            previous = last.pop("event_sha256")
            if hashlib.sha256(canonical_bytes(last)).hexdigest() != previous:
                raise RuntimeError("Existing event chain tail does not verify")
    record = {"event": name, "utc": datetime.now(timezone.utc).isoformat(),
              "monotonic_ns": time.monotonic_ns(),
              "previous_event_sha256": previous, **fields}
    record["event_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    with path.open("ab") as handle:
        handle.write(canonical_bytes(record))
        handle.flush()
    return record


def _verify_bound_files(seal, label):
    bounds = seal.get("bound_files")
    if not isinstance(bounds, dict) or not bounds:
        raise RuntimeError(label + " has no bound files")
    for relative, expected in bounds.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(label + " contains an invalid relative path")
        path = ROOT / relative_path
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(label + " changed: " + relative)


def verify_protocol_seal():
    seal_path = ROOT / "protocol_seal36.json"
    if not seal_path.exists():
        raise RuntimeError("Protocol seal required before public numeric data access")
    seal = json.loads(seal_path.read_text())
    _verify_bound_files(seal, "Frozen protocol/source")
    required = {"protocol36.json", "source/data36.py", ARCHIVE_RELATIVE,
                "inputs/public_source36.json", "inputs/metadata_preflight36.json",
                "inputs/prior_exposure36.json"}
    if not required.issubset(seal["bound_files"]):
        raise RuntimeError("Protocol seal does not bind required data-source inputs")
    gate_path = ROOT / "audit/preexecution_gate36.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if gate.get("status") != "PASS" or gate.get("protocol_seal_sha256") != sha_file(seal_path):
        raise RuntimeError("Independent preexecution PASS required")
    return json.loads((ROOT / "protocol36.json").read_text())


def verify_model_seal():
    verify_protocol_seal()
    seal_path = ROOT / "output/model_seal36.json"
    if not seal_path.exists():
        raise RuntimeError("All-target model and policy seal required before test-value parsing")
    seal = json.loads(seal_path.read_text())
    if seal.get("protocol_seal_sha256") != sha_file(ROOT / "protocol_seal36.json"):
        raise RuntimeError("Model seal belongs to another protocol")
    _verify_bound_files(seal, "Frozen model/policy input")
    if "output/development_data36.json" not in seal["bound_files"]:
        raise RuntimeError("Model seal must bind the frozen training normalizations")
    return seal


def _source_identity():
    source = json.loads((ROOT / "inputs/public_source36.json").read_text())
    archive = ROOT / ARCHIVE_RELATIVE
    if source.get("archive_path") != ARCHIVE_RELATIVE or sha_file(archive) != source["archive_sha256"]:
        raise RuntimeError("Public archive identity changed")
    return archive, source


def _timestamp_rows(zipped):
    """Yield timestamp plus opaque payload; this routine never splits count tokens."""
    if zipped.namelist().count(MEMBER) != 1:
        raise ValueError("Expected exactly one hour.csv member")
    with zipped.open(MEMBER) as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        header = tuple(stream.readline().rstrip("\r\n").split(","))
        if header != HEADER:
            raise ValueError("Unexpected hour.csv header")
        seen, previous = set(), None
        for line_number, line in enumerate(stream, start=2):
            prefix = line.rstrip("\r\n").split(",", 6)
            if len(prefix) != 7:
                raise ValueError("Malformed timestamp prefix at line " + str(line_number))
            day_text, hour_text, opaque_payload = prefix[1], prefix[5], prefix[6]
            day = date.fromisoformat(day_text)
            hour = int(hour_text)
            if day.isoformat() != day_text or str(hour) != hour_text or not 0 <= hour <= 23:
                raise ValueError("Changed timestamp syntax")
            stamp = datetime(day.year, day.month, day.day, hour)
            if not SOURCE_FIRST <= stamp < SOURCE_STOP or stamp in seen:
                raise ValueError("Unexpected or duplicated timestamp")
            if previous is not None and stamp <= previous:
                raise ValueError("Source is not strictly chronological")
            seen.add(stamp)
            previous = stamp
            yield stamp, opaque_payload


def _partition(stamp):
    for name, (start, stop) in PARTITIONS.items():
        if start <= stamp.date() < stop:
            return name
    raise ValueError("Timestamp outside frozen partitions")


def _calendar_targets(name):
    start, stop = PARTITIONS[name]
    while start < stop:
        yield datetime(start.year, start.month, start.day, ORIGIN_HOUR)
        start += timedelta(days=1)


def _required_times(target):
    origin = target - timedelta(hours=HORIZON_HOURS)
    return origin, [origin - timedelta(hours=D-1-j) for j in range(D)]


def metadata_preflight():
    """Inspect chronology only, authorized before freeze; never parse count payloads."""
    archive, source = _source_identity()
    stamps = []
    with zipfile.ZipFile(archive) as zipped:
        inventory = [{"name": x.filename, "bytes": x.file_size,
                      "compressed_bytes": x.compress_size} for x in zipped.infolist()]
        for stamp, _opaque in _timestamp_rows(zipped):
            stamps.append(stamp)
    stamp_set = set(stamps)
    all_hours = [SOURCE_FIRST + timedelta(hours=i)
                 for i in range(int((SOURCE_STOP-SOURCE_FIRST).total_seconds() // 3600))]
    absent = [s.isoformat(timespec="hours") for s in all_hours if s not in stamp_set]
    timestamp_text = [s.isoformat(timespec="hours") for s in stamps]
    calendar_rosters = {}
    for name in PARTITIONS:
        candidates = list(_calendar_targets(name))
        eligible, target_absent, lag_absent = [], 0, 0
        for target in candidates:
            _, lags = _required_times(target)
            if target not in stamp_set:
                target_absent += 1
            elif any(s not in stamp_set for s in lags):
                lag_absent += 1
            else:
                eligible.append(target.isoformat(timespec="hours"))
        calendar_rosters[name] = {
            "candidate_daily_targets": len(candidates),
            "timestamp_complete_candidates": len(eligible),
            "absent_target": target_absent, "absent_lag_with_target_present": lag_absent,
            "timestamp_complete_target_sha256": hashlib.sha256(canonical_bytes(eligible)).hexdigest(),
            "numerical_missingness_not_examined": True,
        }
    reported = source["metadata_reported_instance_count"]
    receipt = {
        "schema": "step36-timestamp-only-preflight-v1", "status": "PASS",
        "utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": source["archive_sha256"], "archive_inventory": inventory,
        "opened_member": MEMBER, "header": list(HEADER), "timestamp_rows": len(stamps),
        "timestamp_rows_by_year": {str(y): sum(s.year == y for s in stamps) for y in (2011, 2012)},
        "timestamp_rows_by_partition": {name: sum(_partition(s) == name for s in stamps) for name in PARTITIONS},
        "expected_calendar_hours": len(all_hours), "absent_calendar_hours": len(absent),
        "absent_calendar_timestamp_sha256": hashlib.sha256(canonical_bytes(absent)).hexdigest(),
        "all_timestamp_sha256": hashlib.sha256(canonical_bytes(timestamp_text)).hexdigest(),
        "first_timestamp": timestamp_text[0], "last_timestamp": timestamp_text[-1],
        "duplicates": 0, "strictly_increasing": True,
        "official_metadata_reported_rows": reported,
        "metadata_row_count_difference_archive_minus_reported": len(stamps)-reported,
        "calendar_availability": calendar_rosters,
        "count_values_numerically_parsed": 0, "test_count_values_numerically_parsed": 0,
        "count_missingness_inspected": False, "target_payload_tokens_split": False,
        "day_csv_opened": False, "scientific_model_fits": 0,
        "incidental_training_search_snippet_exposure": "See inputs/prior_exposure36.json",
    }
    write_json(ROOT / "inputs/metadata_preflight36.json", receipt)
    return receipt


def _parse_counts(opaque_payload):
    pieces = opaque_payload.rsplit(",", 3)
    if len(pieces) != 4:
        raise ValueError("Malformed count suffix")
    counts = []
    for token in pieces[-3:]:
        if token.strip().lower() in {"", "na", "nan", "+nan", "-nan", "null", "none",
                                    "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            counts.append(None)
            continue
        value = int(token)
        if str(value) != token or value < 0 or value > 2**53:
            raise ValueError("Unexpected count syntax/range; exact float64 inputs required")
        counts.append(value)
    return tuple(counts)


def _read_values(stage):
    """Enforce the appropriate seals before any actual count token is split."""
    if stage == "development":
        verify_protocol_seal()
    elif stage == "test":
        verify_model_seal()
    else:
        raise ValueError("Only development and sealed-test access stages are allowed")
    archive, source = _source_identity()
    metadata_path = ROOT / "inputs/metadata_preflight36.json"
    metadata = json.loads(metadata_path.read_text())
    values, stamps = {}, []
    numeric_rows = {name: 0 for name in PARTITIONS}
    with zipfile.ZipFile(archive) as zipped:
        for stamp, opaque_payload in _timestamp_rows(zipped):
            stamps.append(stamp.isoformat(timespec="hours"))
            if stage == "development" and stamp > DEVELOPMENT_CUTOFF:
                # Do not split, convert, or inspect the missingness of this payload.
                continue
            values[stamp] = _parse_counts(opaque_payload)
            numeric_rows[_partition(stamp)] += 1
    if hashlib.sha256(canonical_bytes(stamps)).hexdigest() != metadata["all_timestamp_sha256"]:
        raise ValueError("Timestamp roster changed after metadata preflight")
    # Validate the relation only among measurements parsed after their access gate.
    sums_checked = {name: 0 for name in PARTITIONS}
    incomplete_rows = {name: 0 for name in PARTITIONS}
    for stamp, counts in values.items():
        if any(value is None for value in counts):
            incomplete_rows[_partition(stamp)] += 1
        else:
            if counts[0] + counts[1] != counts[2]:
                raise ValueError("casual + registered != cnt in allowed parsed partition")
            sums_checked[_partition(stamp)] += 1
    receipt = {
        "stage": stage, "archive_sha256": source["archive_sha256"],
        "metadata_preflight_sha256": sha_file(metadata_path),
        "prior_exposure_sha256": sha_file(ROOT / "inputs/prior_exposure36.json"),
        "target_order": list(TARGETS), "numeric_rows_by_partition": numeric_rows,
        "count_sum_checks_by_partition": sums_checked,
        "any_missing_count_rows_by_partition": incomplete_rows,
        "test_numeric_values_parsed": stage == "test",
        "test_missingness_inspected": stage == "test", "day_csv_opened": False,
        "timestamp_rows": len(stamps), "count_fields_per_parsed_row": len(TARGETS),
        "development_numeric_end_inclusive": DEVELOPMENT_CUTOFF.isoformat(timespec="hours"),
        "post_development_payload_skipped_uninterpreted": stage == "development",
    }
    return values, receipt


def _cohorts(values, name):
    """Construct all three tasks together on a full daily calendar, never row offsets."""
    origins, targets, lag_rows = [], [], []
    exclusions = {"required_timestamp_absent": 0, "missing_lag_or_target": 0}
    absent_details = {"target_absent": 0, "lag_absent_with_target_present": 0}
    candidates = list(_calendar_targets(name))
    for target in candidates:
        origin, lags = _required_times(target)
        required = lags + [target]
        if any(stamp not in values for stamp in required):
            exclusions["required_timestamp_absent"] += 1
            absent_details["target_absent" if target not in values else "lag_absent_with_target_present"] += 1
            continue
        if any(any(value is None for value in values[stamp]) for stamp in required):
            exclusions["missing_lag_or_target"] += 1
            continue
        origins.append(origin.isoformat(timespec="hours"))
        targets.append(target.isoformat(timespec="hours"))
        lag_rows.append((lags, target))
    if not targets:
        raise ValueError("No eligible rows in " + name)
    if len(candidates) != len(targets) + sum(exclusions.values()):
        raise AssertionError("Cohort accounting failed")
    tasks = {}
    for j, target_name in enumerate(TARGETS):
        rows = [[values[stamp][j] for stamp in lags] for lags, _ in lag_rows]
        raw = [values[target][j] for _, target in lag_rows]
        tasks[target_name] = {
            "name": name, "target": target_name,
            "ids": [name + ":" + stamp for stamp in targets],
            "origin_timestamps": list(origins), "target_timestamps": list(targets),
            "X": np.asarray(rows, dtype=np.float64), "raw_targets": raw,
            "roster": {"candidate_daily_targets": len(candidates), "eligible": len(targets),
                       "excluded": dict(exclusions), "absent_timestamp_details": dict(absent_details),
                       "target_years": sorted({datetime.fromisoformat(s).year for s in targets}),
                       "target_date_start_inclusive": PARTITIONS[name][0].isoformat(),
                       "target_date_stop_exclusive": PARTITIONS[name][1].isoformat(),
                       "first_target": targets[0], "last_target": targets[-1],
                       "target_id_sha256": hashlib.sha256(canonical_bytes(targets)).hexdigest(),
                       "common_three_target_roster": True,
                       "own_target_lags_only": True, "lag_dimension": D,
                       "origin_hour": ORIGIN_HOUR, "horizon_hours": HORIZON_HOURS},
        }
    return tasks


def _normalization(train, target):
    low, high = min(train["raw_targets"]), max(train["raw_targets"])
    if high <= low:
        raise ValueError("Degenerate training target scale for " + target + "; frozen pilot must stop")
    return {"target": target, "train_target_min": low, "train_target_max": high,
            "train_target_range": high-low, "target_formula": "(raw-train_min)/train_range",
            "normalization_source": "eligible_2011_training_targets_only",
            "predictor_interval": ["0", "1"], "targets_clipped": False,
            "physical_mse_multiplier": (high-low)**2, "physical_units": "hourly_rental_count_squared"}


def _normalize(cohort, normalization):
    low, span = normalization["train_target_min"], normalization["train_target_range"]
    if span <= 0:
        raise ValueError("Nonpositive training target range")
    cohort["y"] = [Fraction(raw-low, span) for raw in cohort["raw_targets"]]
    cohort["persistence"] = [Fraction(int(raw)-low, span) for raw in cohort["X"][:, -1]]
    cohort["roster"]["target_outside_prediction_range_count"] = sum(y < 0 or y > 1 for y in cohort["y"])
    return cohort


def load_development():
    verify_protocol_seal()
    event("development_numeric_access_start", target_order=list(TARGETS),
          numeric_end_inclusive=DEVELOPMENT_CUTOFF.isoformat(timespec="hours"),
          protocol_seal_sha256=sha_file(ROOT / "protocol_seal36.json"),
          test_numeric_values_parsed=False, reserved_patient_arrays_read=False)
    values, receipt = _read_values("development")
    train_tasks, decision_tasks = _cohorts(values, "train"), _cohorts(values, "decision")
    normalizations = {}
    for target in TARGETS:
        normalizations[target] = _normalization(train_tasks[target], target)
        _normalize(train_tasks[target], normalizations[target])
        _normalize(decision_tasks[target], normalizations[target])
    receipt.update({"schema": "step36-development-data-v1", "normalizations": normalizations,
                    "train": {t: train_tasks[t]["roster"] for t in TARGETS},
                    "decision": {t: decision_tasks[t]["roster"] for t in TARGETS},
                    "rolling_observed_history": True, "future_covariates": False,
                    "features": "own_target_last_24_hourly_counts_only",
                    "calendar_or_weather_features": False, "cross_target_features": False,
                    "public_dataset_values_read": True, "reserved_patients_accessed": 0})
    path = ROOT / "output/development_data36.json"
    write_json(path, receipt)
    event("development_numeric_access_complete", receipt_sha256=sha_file(path),
          eligible_train={t: len(train_tasks[t]["y"]) for t in TARGETS},
          eligible_decision={t: len(decision_tasks[t]["y"]) for t in TARGETS},
          test_numeric_values_parsed=False)
    return train_tasks, decision_tasks, normalizations


def load_test(normalizations):
    verify_model_seal()
    frozen = json.loads((ROOT / "output/development_data36.json").read_text())["normalizations"]
    if normalizations != frozen or set(normalizations) != set(TARGETS):
        raise RuntimeError("Test normalizations differ from sealed training normalizations")
    event("test_numeric_access_start", target_order=list(TARGETS),
          model_seal_sha256=sha_file(ROOT / "output/model_seal36.json"),
          all_policy_and_model_choices_frozen=True, reserved_patient_arrays_read=False)
    values, receipt = _read_values("test")
    test_tasks = _cohorts(values, "test")
    for target in TARGETS:
        _normalize(test_tasks[target], normalizations[target])
    receipt.update({"schema": "step36-test-data-v1", "normalizations": normalizations,
                    "test": {t: test_tasks[t]["roster"] for t in TARGETS},
                    "model_seal_sha256": sha_file(ROOT / "output/model_seal36.json"),
                    "public_dataset_values_read": True, "reserved_patients_accessed": 0})
    path = ROOT / "output/test_data36.json"
    write_json(path, receipt)
    event("test_numeric_access_complete", receipt_sha256=sha_file(path),
          eligible_test={t: len(test_tasks[t]["y"]) for t in TARGETS})
    return test_tasks


def artificial_selfcheck():
    """Exercise causal roster, exact scaling and closed barriers using toy data only."""
    global ROOT
    original_root = ROOT
    checks = {}
    assert [len(list(_calendar_targets(k))) for k in PARTITIONS] == [365, 182, 184]
    checks["full_calendar_candidate_counts"] = True
    stamps = [datetime(2011, 1, 1) + timedelta(hours=i) for i in range(24*4)]
    values = {stamp: (i, i+1, 2*i+1) for i, stamp in enumerate(stamps)}
    tasks = _cohorts(values, "train")
    assert all(tasks[t]["target_timestamps"] == tasks["cnt"]["target_timestamps"] for t in TARGETS)
    assert len(tasks["cnt"]["raw_targets"]) == 3
    assert tasks["casual"]["X"][0].tolist() == list(range(24))
    assert tasks["casual"]["raw_targets"][0] == 47
    assert tasks["casual"]["roster"]["candidate_daily_targets"] == 365
    checks["causal_24_lags_horizon_and_complete_calendar"] = True
    for t in TARGETS:
        n = _normalization(tasks[t], t)
        _normalize(tasks[t], n)
        assert tasks[t]["y"] == [Fraction(0), Fraction(1, 2), Fraction(1)]
    checks["exact_training_only_normalization"] = True
    altered = dict(values)
    altered[datetime(2011, 1, 1, 3)] = (None, 4, 7)
    incomplete = _cohorts(altered, "train")
    assert all(len(incomplete[t]["raw_targets"]) == 2 for t in TARGETS)
    assert incomplete["cnt"]["roster"]["excluded"]["missing_lag_or_target"] == 1
    checks["one_missing_component_excludes_all_three_targets"] = True
    absent = dict(values)
    del absent[datetime(2011, 1, 2, 23)]
    omitted = _cohorts(absent, "train")
    assert omitted["cnt"]["roster"]["candidate_daily_targets"] == 365
    assert omitted["cnt"]["roster"]["absent_timestamp_details"]["target_absent"] > 0
    checks["absent_target_is_counted_not_silently_dropped"] = True
    toy = {"raw_targets": [-1, 3], "X": np.asarray([[4.0], [0.0]]), "roster": {}}
    _normalize(toy, {"train_target_min": 0, "train_target_range": 2})
    assert toy["y"] == [Fraction(-1, 2), Fraction(3, 2)]
    assert toy["roster"]["target_outside_prediction_range_count"] == 2
    checks["outside_interval_labels_preserved"] = True
    assert _parse_counts("calendar-and-weather-uninterpreted,1,2,3") == (1, 2, 3)
    assert _parse_counts("opaque,1,NA,3") == (1, None, 3)
    checks["artificial_count_suffix_parser"] = True
    try:
        with tempfile.TemporaryDirectory(prefix="step36_artificial_barrier_") as temporary:
            ROOT = Path(temporary)
            for method in (lambda: _read_values("development"), lambda: _read_values("test")):
                try:
                    method()
                except RuntimeError as exc:
                    assert "Protocol seal required" in str(exc)
                else:
                    raise AssertionError("Missing seal did not stop data access")
            checks["no_protocol_seal_blocks_both_value_read_paths"] = True
            (ROOT / "inputs").mkdir()
            (ROOT / "source").mkdir()
            (ROOT / "source/data36.py").write_text("artificial source marker\n")
            write_json(ROOT / "protocol36.json", {"artificial": True})
            write_json(ROOT / "inputs/prior_exposure36.json", {"artificial": True})
            artificial_lines = [",".join(HEADER),
                "1,2011-01-01,1,0,1,0,0,6,0,1,0,0,0,0,1,2,3",
                "2,2012-06-30,1,1,6,23,0,6,0,1,0,0,0,0,4,5,9",
                "3,2012-07-01,1,1,7,0,0,0,0,1,0,0,0,0,FORBIDDEN,FORBIDDEN,FORBIDDEN"]
            with zipfile.ZipFile(ROOT / ARCHIVE_RELATIVE, "w") as zipped:
                zipped.writestr(MEMBER, "\n".join(artificial_lines) + "\n")
                zipped.writestr("day.csv", "FORBIDDEN_DAILY_DATA")
            write_json(ROOT / "inputs/public_source36.json", {
                "archive_path": ARCHIVE_RELATIVE,
                "archive_sha256": sha_file(ROOT / ARCHIVE_RELATIVE),
                "metadata_reported_instance_count": 3})
            metadata_preflight()
            required = ["protocol36.json", "source/data36.py", ARCHIVE_RELATIVE,
                        "inputs/public_source36.json", "inputs/metadata_preflight36.json",
                        "inputs/prior_exposure36.json"]
            write_json(ROOT / "protocol_seal36.json", {
                "bound_files": {r: sha_file(ROOT / r) for r in required}})
            try:
                _read_values("development")
            except RuntimeError as exc:
                assert "Independent preexecution PASS required" in str(exc)
            else:
                raise AssertionError("Missing independent PASS did not stop access")
            checks["no_independent_pass_blocks_development_access"] = True
            write_json(ROOT / "audit/preexecution_gate36.json", {
                "status": "PASS", "protocol_seal_sha256": sha_file(ROOT / "protocol_seal36.json")})
            parsed, receipt = _read_values("development")
            assert len(parsed) == 2 and receipt["numeric_rows_by_partition"] == {
                "train": 1, "decision": 1, "test": 0}
            assert parsed[datetime(2012, 6, 30, 23)] == (4, 5, 9)
            checks["malformed_artificial_test_counts_skipped_uninterpreted"] = True
            try:
                _read_values("test")
            except RuntimeError as exc:
                assert "All-target model and policy seal required" in str(exc)
            else:
                raise AssertionError("Missing model seal did not stop test access")
            checks["no_model_seal_blocks_test_access"] = True
            frozen = {t: {"train_target_min": 0, "train_target_range": 1} for t in TARGETS}
            write_json(ROOT / "output/development_data36.json", {"normalizations": frozen})
            write_json(ROOT / "output/model_seal36.json", {
                "protocol_seal_sha256": sha_file(ROOT / "protocol_seal36.json"),
                "bound_files": {"output/development_data36.json": sha_file(ROOT / "output/development_data36.json")}})
            try:
                load_test({})
            except RuntimeError as exc:
                assert "Test normalizations differ" in str(exc)
            else:
                raise AssertionError("Changed normalization did not stop test access")
            checks["normalization_mismatch_blocks_before_test_payload_read"] = True
            (ROOT / "protocol36.json").write_text('{"artificial":"modified"}\n')
            try:
                _read_values("development")
            except RuntimeError as exc:
                assert "Frozen protocol/source changed: protocol36.json" in str(exc)
            else:
                raise AssertionError("Modified frozen source did not stop access")
            checks["changed_bound_file_blocks_access"] = True
    finally:
        ROOT = original_root
    return {"schema": "step36-artificial-data-preflight-v1", "status": "PASS",
            "utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
            "source_sha256": sha_file(Path(__file__)),
            "public_archive_numeric_values_read": False, "test_count_values_parsed": 0,
            "scientific_model_fits": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("metadata", "selfcheck"))
    arguments = parser.parse_args()
    if arguments.mode == "metadata":
        result = metadata_preflight()
    else:
        result = artificial_selfcheck()
        write_json(ROOT / "tests/artificial_data_preflight36.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
