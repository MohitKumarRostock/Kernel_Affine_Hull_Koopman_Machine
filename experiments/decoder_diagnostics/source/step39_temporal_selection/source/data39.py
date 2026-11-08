"""Gate-enforced two-domain rolling-development and chronological test data.

Every measurement token is decoded only by _read_authorized_values after the
matching independent gate. All previous-stage measurements may be reused as
training/history when causally available. The final test target is decoded
only after the current query's complete prediction payload has been sealed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

import prepare_data39 as prep

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "f1": {"train_start": "2017-01-01", "train_stop": "2017-06-01", "validation_start": "2017-06-01", "validation_stop": "2017-07-01"},
    "f2": {"train_start": "2017-01-01", "train_stop": "2017-07-01", "validation_start": "2017-07-01", "validation_stop": "2017-08-01"},
    "f3": {"train_start": "2017-01-01", "train_stop": "2017-08-01", "validation_start": "2017-08-01", "validation_stop": "2017-09-01"},
    "final": {"train_start": "2017-01-01", "train_stop": "2017-09-01"},
}
TEST_START = "2017-09-01"
TEST_STOP = "2017-12-01"
EXPECTED_N = {
    "tetouan_zone1": {"f1": {"train": 150, "validation": 30}, "f2": {"train": 180, "validation": 31},
                       "f3": {"train": 211, "validation": 31}, "final": {"train": 242}, "test": 91},
    "metro_traffic": {"f1": {"train": 140, "validation": 30}, "f2": {"train": 170, "validation": 29},
                       "f3": {"train": 199, "validation": 30}, "final": {"train": 229}, "test": 84},
}
DATA_REQUIRED = {"protocol39.json", "source/data39.py", "source/prepare_data39.py",
                 "inputs/calendar_roster39.json", "inputs/prior_exposure39.json"}
for _task in prep.TASKS:
    DATA_REQUIRED.update({prep.archive_relative(_task), f"inputs/{_task}/public_source39.json",
                          f"inputs/{_task}/calendar_projection39.json", f"inputs/{_task}/calendar_metadata39.json"})


def task_ids():
    return tuple(prep.TASKS)


def canon(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [canon(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    return value


def canonical_bytes(value):
    return (json.dumps(canon(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def event(name, **fields):
    path = ROOT / "output/access_and_execution39.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if path.exists():
        lines = path.read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            previous = last.pop("event_sha256")
            if digest(last) != previous:
                raise RuntimeError("Existing event tail changed")
    record = {"event": name, "utc": datetime.now(timezone.utc).isoformat(),
              "monotonic_ns": time.monotonic_ns(), "previous_event_sha256": previous, **fields}
    record["event_sha256"] = digest(record)
    with path.open("ab") as handle:
        handle.write(canonical_bytes(record))
        handle.flush()
    return record


def _task(task):
    if task not in prep.TASKS:
        raise RuntimeError("Unknown frozen task")
    return task


def _stage(stage):
    if stage not in STAGES:
        raise RuntimeError("Unknown frozen fitting stage")
    return stage


def _relative(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError("Unsafe relative evidence path")
    return path


def _verify_bound_files(seal, label):
    bounds = seal.get("bound_files")
    if not isinstance(bounds, dict) or not bounds:
        raise RuntimeError(label + " has no bound files")
    for relative, expected in bounds.items():
        path = ROOT / _relative(relative)
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(label + " changed: " + relative)


def _gate(relative, **bindings):
    path = ROOT / relative
    gate = read_json(path) if path.is_file() else {}
    if gate.get("status") != "PASS" or any(gate.get(k) != v for k, v in bindings.items()):
        raise RuntimeError("Matching independent PASS required: " + relative)
    return gate


def _iso(stamp):
    return stamp.isoformat(timespec="seconds")


def _timestamp(text):
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is not None or _iso(stamp) != text:
        raise RuntimeError("Noncanonical local-naive timestamp")
    return stamp


def _required_times(target):
    origin = target - timedelta(hours=24)
    return origin, [origin - timedelta(hours=23 - j) for j in range(24)]


def _calendar(task):
    _task(task)
    source = read_json(ROOT / f"inputs/{task}/public_source39.json")
    projection = read_json(ROOT / f"inputs/{task}/calendar_projection39.json")
    spec = prep.TASKS[task]
    if (source.get("task") != task or source.get("archive_path") != prep.archive_relative(task)
        or projection.get("task") != task or projection.get("archive_sha256") != source["archive_sha256"]
        or projection.get("archive_member") != spec["member"]
        or projection.get("parsed_data_columns") != [spec["timestamp_index"]]
        or projection.get("target_field_index") != spec["target_index"]
        or projection.get("target_values_decoded_or_parsed") != 0
        or projection.get("other_measurement_values_decoded_or_parsed") != 0):
        raise RuntimeError("Source calendar identity or approved scope differs")
    index = {row["timestamp"]: row for row in projection["rows"]}
    if len(index) != len(projection["rows"]):
        raise RuntimeError("Calendar projection contains repeated grouped timestamps")
    return source, projection, index


def _derive_queries(task, start, stop):
    _, _, rows = _calendar(task)
    target = datetime.fromisoformat(start).replace(hour=23)
    end = datetime.fromisoformat(stop).replace(hour=23)
    queries = []
    excluded = []
    while target < end:
        origin, lags = _required_times(target)
        absent = [_iso(stamp) for stamp in [target, *lags] if _iso(stamp) not in rows]
        if absent:
            excluded.append({"target_timestamp": _iso(target), "absent_required_timestamps": absent})
        else:
            queries.append({"id": task + ":" + _iso(target), "task": task,
                            "target_timestamp": _iso(target), "origin_timestamp": _iso(origin)})
        target += timedelta(days=1)
    return queries, excluded


def metadata_rosters():
    if (ROOT / "protocol_seal39.json").exists():
        raise RuntimeError("Calendar roster cannot change after protocol freeze")
    tasks = {}
    for task in task_ids():
        source, projection, _ = _calendar(task)
        stages = {}
        for stage, intervals in STAGES.items():
            parts = {}
            for part in ("train", "validation"):
                if part + "_start" not in intervals:
                    continue
                queries, excluded = _derive_queries(task, intervals[part + "_start"], intervals[part + "_stop"])
                if len(queries) != EXPECTED_N[task][stage][part]:
                    raise RuntimeError("Frozen calendar cohort size differs")
                parts[part] = {"queries": queries, "n": len(queries), "query_ids_sha256": digest([q["id"] for q in queries]),
                               "calendar_start": intervals[part + "_start"], "calendar_stop_exclusive": intervals[part + "_stop"],
                               "calendar_ineligible": excluded}
            stages[stage] = parts
        test, excluded = _derive_queries(task, TEST_START, TEST_STOP)
        if len(test) != EXPECTED_N[task]["test"]:
            raise RuntimeError("Frozen test calendar cohort size differs")
        tasks[task] = {"stages": stages, "test": {"queries": test, "n": len(test),
                      "query_ids_sha256": digest([q["id"] for q in test]), "calendar_start": TEST_START,
                      "calendar_stop_exclusive": TEST_STOP, "calendar_ineligible": excluded},
                      "archive_sha256": source["archive_sha256"], "csv_sha256": projection["csv_sha256"],
                      "projection_sha256": sha_file(ROOT / f"inputs/{task}/calendar_projection39.json")}
    record = {"schema": "step39-frozen-calendar-roster-v1", "tasks": tasks,
              "query_horizon_hours": 24, "query_lag_hours": 24, "daily_target_hour": 23,
              "measurement_values_decoded_or_parsed": 0,
              "eligibility": "Presence of target timestamp and all24 exact hourly history timestamps only; no numeric missingness/value inspection",
              "test_months": ["2017-09", "2017-10", "2017-11"],
              "unused_month": "2017-12", "other_source_years": "Only causal boundary history may be used for a2017query"}
    write_json(ROOT / "inputs/calendar_roster39.json", record)
    return record


def data_contract():
    roster = read_json(ROOT / "inputs/calendar_roster39.json")
    sources = {}
    for task in task_ids():
        source, projection, _ = _calendar(task)
        sources[task] = {"dataset": source["dataset"], "uci_id": source["uci_id"], "doi": source["doi"],
                         "archive": source["archive_path"], "archive_sha256": source["archive_sha256"],
                         "member": source["member"], "csv_sha256": projection["csv_sha256"],
                         "target": prep.TASKS[task]["target_header"], "target_field_index": prep.TASKS[task]["target_index"],
                         "timezone_convention": prep.TASKS[task]["timezone_convention"],
                         "expected_n": EXPECTED_N[task], "source_license": "CC BY 4.0"}
    return {"task_order": list(task_ids()), "sources": sources, "stages": STAGES,
            "test_start": TEST_START, "test_stop_exclusive": TEST_STOP,
            "calendar_roster_sha256": sha_file(ROOT / "inputs/calendar_roster39.json"),
            "horizon_hours": 24, "lag_hours": 24, "daily_target_hour": 23,
            "hourly_snapshot": "Exact minute00 source reading; no averaging or future aggregation",
            "eligibility": roster["eligibility"],
            "duplicate_policy": "At authorized timestamp read all source target tokens must agree exactly; disagreement STOP_RETAIN_FAILURE without row dropping",
            "numeric_policy": "Parse finite nonnegative decimal tokens as exact rationals; invalid token STOP_RETAIN_FAILURE",
            "feature_values": "Authorized exact raw rationals converted to binary64, then representation pipeline; feature-coordinate arithmetic certificate scope remains explicit",
            "target_normalization": "Exact(raw−stage_training_target_min)/(stage_training_target_max−stage_training_target_min)",
            "normalization_scope": "Eachfold training-only, finalrefit training-only; all diagnostics reverse their own stage normalization",
            "targets_clipped": False, "degenerate_range_action": "STOP_RETAIN_FAILURE",
            "no_online_refit": True, "previously_revealed_targets_may_be_later_history": True,
            "other_measurement_predictors": False, "future_calendar_predictors": False,
            "test_access": "Perquery history through origin, complete207predictionbyte seal, then current target only"}


def verify_protocol_seal():
    path = ROOT / "protocol_seal39.json"
    if not path.is_file():
        raise RuntimeError("Frozen Step39 protocol seal required before measurement decoding")
    seal = read_json(path)
    _verify_bound_files(seal, "Frozen protocol/source/input")
    if not DATA_REQUIRED <= set(seal["bound_files"]):
        raise RuntimeError("Protocol seal omits a data dependency")
    _gate("audit/preexecution_gate39.json", protocol_seal_sha256=sha_file(path))
    protocol = read_json(ROOT / "protocol39.json")
    if protocol.get("step") != 39 or protocol.get("data") != data_contract():
        raise RuntimeError("Protocol source, chronology, arithmetic, or access contract differs")
    return protocol


def calendar_queries(task, stage, partition):
    _task(task)
    _stage(stage)
    if partition not in ("train", "validation") or (stage == "final" and partition != "train"):
        raise RuntimeError("Unknown stage cohort")
    intervals = STAGES[stage]
    queries, _ = _derive_queries(task, intervals[partition + "_start"], intervals[partition + "_stop"])
    frozen = read_json(ROOT / "inputs/calendar_roster39.json")["tasks"][task]["stages"][stage][partition]
    if queries != frozen["queries"] or len(queries) != EXPECTED_N[task][stage][partition]:
        raise RuntimeError("Frozen metadata-only roster differs")
    return queries


def test_queries(task):
    _task(task)
    queries, _ = _derive_queries(task, TEST_START, TEST_STOP)
    frozen = read_json(ROOT / "inputs/calendar_roster39.json")["tasks"][task]["test"]
    if queries != frozen["queries"] or len(queries) != EXPECTED_N[task]["test"]:
        raise RuntimeError("Frozen test roster differs")
    return queries


def _verify_completion(task, stage):
    path = ROOT / f"output/{task}/{stage}/completion_seal39.json"
    if not path.is_file():
        raise RuntimeError("Previous fold's scoring must be sealed before next training phase")
    seal = read_json(path)
    if (seal.get("schema") != "step39-fold-completion-seal-v1" or seal.get("task") != task
        or seal.get("stage") != stage or seal.get("protocol_seal_sha256") != sha_file(ROOT / "protocol_seal39.json")):
        raise RuntimeError("Fold completion seal identity differs")
    _verify_bound_files(seal, "Frozen completed-fold scores")
    required = {f"output/{task}/{stage}/validation_predictions39.json", f"output/{task}/{stage}/validation_scores39.json"}
    if not required <= set(seal["bound_files"]):
        raise RuntimeError("Fold completion seal omits prediction or score records")
    _gate(f"audit/{task}/{stage}/validation_gate39.json", task=task, stage=stage,
          protocol_seal_sha256=sha_file(ROOT / "protocol_seal39.json"), completion_seal_sha256=sha_file(path))
    return seal


def verify_selection_seal(task):
    verify_protocol_seal()
    _verify_completion(task, "f3")
    path = ROOT / f"output/{task}/selection_seal39.json"
    if not path.is_file():
        raise RuntimeError("Development selection must be frozen before final refitting")
    seal = read_json(path)
    if (seal.get("schema") != "step39-selection-seal-v1" or seal.get("task") != task
        or seal.get("protocol_seal_sha256") != sha_file(ROOT / "protocol_seal39.json")):
        raise RuntimeError("Selection seal identity differs")
    _verify_bound_files(seal, "Frozen development selection")
    if f"output/{task}/selection39.json" not in seal["bound_files"]:
        raise RuntimeError("Selection seal omits selection record")
    _gate(f"audit/{task}/selection_gate39.json", task=task,
          protocol_seal_sha256=sha_file(ROOT / "protocol_seal39.json"), selection_seal_sha256=sha_file(path))
    return seal


def _verify_training_transition(task, stage):
    verify_protocol_seal()
    if stage == "f2":
        _verify_completion(task, "f1")
    elif stage == "f3":
        _verify_completion(task, "f2")
    elif stage == "final":
        verify_selection_seal(task)


def verify_fit_seal(task, stage):
    _task(task)
    _stage(stage)
    _verify_training_transition(task, stage)
    path = ROOT / f"output/{task}/{stage}/fit_seal39.json"
    if not path.is_file():
        raise RuntimeError("Completed stage fit seal required before evaluation access")
    seal = read_json(path)
    if (seal.get("schema") != "step39-fit-seal-v1" or seal.get("task") != task or seal.get("stage") != stage
        or seal.get("protocol_seal_sha256") != sha_file(ROOT / "protocol_seal39.json")):
        raise RuntimeError("Completed-fit seal identity differs")
    _verify_bound_files(seal, "Frozen stage fits")
    required = {f"output/{task}/{stage}/normalization39.json", f"output/{task}/{stage}/cohorts/train.json",
                f"output/{task}/{stage}/cohorts/train_X.npy"}
    if not required <= set(seal["bound_files"]):
        raise RuntimeError("Fit seal omits stage-specific training normalization/cohort")
    _gate(f"audit/{task}/{stage}/fit_gate39.json", task=task, stage=stage,
          protocol_seal_sha256=sha_file(ROOT / "protocol_seal39.json"), fit_seal_sha256=sha_file(path))
    return seal


def verify_model_seal(task):
    seal = verify_fit_seal(task, "final")
    _gate(f"audit/{task}/pretest_gate39.json", task=task,
          protocol_seal_sha256=sha_file(ROOT / "protocol_seal39.json"),
          fit_seal_sha256=sha_file(ROOT / f"output/{task}/final/fit_seal39.json"),
          selection_seal_sha256=sha_file(ROOT / f"output/{task}/selection_seal39.json"))
    return seal


def _state_path(task):
    return ROOT / f"output/{task}/test/access_state39.json"


def _load_test_state(task):
    if not _state_path(task).is_file():
        raise RuntimeError("No chronological test query is pending")
    state = read_json(_state_path(task))
    queries = test_queries(task)
    index = state.get("next_index")
    if (state.get("schema") != "step39-chronological-test-access-state-v1" or state.get("task") != task
        or state.get("model_seal_sha256") != sha_file(ROOT / f"output/{task}/final/fit_seal39.json")
        or state.get("status") not in ("READY", "PENDING_HISTORY", "PREDICTIONS_SEALED", "COMPLETE")
        or type(index) is not int or not 0 <= index <= len(queries)
        or state.get("previously_revealed_test_targets") != [q["target_timestamp"] for q in queries[:index]]
        or (state["status"] == "COMPLETE") != (index == len(queries))):
        raise RuntimeError("Invalid, failed, or nonprefix chronological test state")
    path = ROOT / "output/access_and_execution39.jsonl"
    if not path.is_file():
        raise RuntimeError("Chronological test state lacks access events")
    previous = None
    revealed = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        saved = item.pop("event_sha256")
        if digest(item) != saved or item["previous_event_sha256"] != previous:
            raise RuntimeError("Access-event chain changed")
        previous = saved
        if item["event"] == "test_target_revealed" and item.get("task") == task:
            revealed.append(item["query_id"])
    if revealed != [q["id"] for q in queries[:index]]:
        raise RuntimeError("Chronological state advanced without retained target-reveal events")
    return state


def _verify_query(task, query, index):
    expected = test_queries(task)
    if not 0 <= index < len(expected) or query != expected[index]:
        raise RuntimeError("Query is not the next frozen calendar query")


def _prediction_path(task, relative):
    candidate = Path(relative)
    path = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    try:
        rel = path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("Prediction file lies outside Step39") from exc
    if tuple(rel.parts[:3]) != ("output", task, "test") or not path.is_file():
        raise RuntimeError("Prediction file must be a retained task-specific test output")
    return path, str(rel)


def _parse_numeric_token(token, stamp):
    # The first source measurement decoding occurs here, after authorization.
    if len(token) > 256 or re.fullmatch(rb"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", token) is None:
        raise RuntimeError("Invalid/nonfinite/negative measurement at " + stamp + "; no value-based row dropping")
    value = F(token.decode("ascii"))
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if value < 0 or not finite:
        raise RuntimeError("Measurement is not a finite nonnegative input")
    return value


def _read_authorized_values(task, stage, phase, stamps):
    _task(task)
    _stage(stage)
    if phase == "train":
        _verify_training_transition(task, stage)
        limit = datetime.fromisoformat(STAGES[stage]["train_stop"]) - timedelta(seconds=1)
        queries = calendar_queries(task, stage, "train")
        authorized = set()
        for query in queries:
            target = _timestamp(query["target_timestamp"])
            authorized.add(target)
            authorized.update(_required_times(target)[1])
        if list(stamps) != sorted(authorized):
            raise RuntimeError("Training source read is not exactly the frozen cohort projection")
    elif phase == "validation":
        if stage == "final":
            raise RuntimeError("Final model has no validation-label access")
        verify_fit_seal(task, stage)
        limit = datetime.fromisoformat(STAGES[stage]["validation_stop"]) - timedelta(seconds=1)
        queries = calendar_queries(task, stage, "validation")
        authorized = set()
        for query in queries:
            target = _timestamp(query["target_timestamp"])
            authorized.add(target)
            authorized.update(_required_times(target)[1])
        if list(stamps) != sorted(authorized):
            raise RuntimeError("Validation source read is not exactly the frozen cohort projection")
    elif phase in ("test_history", "test_target"):
        if stage != "final":
            raise RuntimeError("Test values require the final fitting stage")
        verify_model_seal(task)
        state = _load_test_state(task)
        pending = state.get("pending_query")
        if pending is None:
            raise RuntimeError("No current test query authorizes source measurement access")
        _verify_query(task, pending, state["next_index"])
        if phase == "test_history":
            if state["status"] != "PENDING_HISTORY" or state.get("history_complete"):
                raise RuntimeError("Current test history already consumed or predictions sealed")
            limit = _timestamp(pending["origin_timestamp"])
            if list(stamps) != _required_times(_timestamp(pending["target_timestamp"]))[1]:
                raise RuntimeError("Test history is not the frozen24hour causal window")
        else:
            if state["status"] != "PREDICTIONS_SEALED" or state.get("target_decode_started"):
                raise RuntimeError("Current predictions must be sealed before exactly one target read")
            limit = _timestamp(pending["target_timestamp"])
            if list(stamps) != [limit]:
                raise RuntimeError("Only the current sealed target may be decoded")
            seal = read_json(ROOT / state["prediction_seal_path"])
            path, _ = _prediction_path(task, seal["prediction_path"])
            if (sha_file(ROOT / state["prediction_seal_path"]) != state["prediction_seal_sha256"]
                or sha_file(path) != seal["prediction_sha256"] or seal["query_id"] != pending["id"]
                or seal["model_seal_sha256"] != state["model_seal_sha256"]):
                raise RuntimeError("Current test prediction seal changed")
            state["target_decode_started"] = True
            write_json(_state_path(task), state)
    else:
        raise RuntimeError("Unknown measurement-access phase")
    if not stamps or any(stamp > limit for stamp in stamps):
        raise RuntimeError("Source read exceeds its authorized forecast boundary")
    source, projection, index = _calendar(task)
    raw, actual_source, _ = prep.source_csv(task)
    if actual_source != source or hashlib.sha256(raw).hexdigest() != projection["csv_sha256"]:
        raise RuntimeError("Opaque source bytes changed")
    values = {}
    source_record_count = 0
    duplicated = []
    for stamp in stamps:
        stamp_text = _iso(stamp)
        if stamp_text not in index:
            raise RuntimeError("Authorized timestamp absent from frozen calendar")
        tokens = []
        for ref in index[stamp_text]["records"]:
            record = raw[ref["byte_start"]:ref["byte_stop"]]
            fields = prep.byte_fields(record)
            if len(fields) != 9:
                raise RuntimeError("Source record schema differs")
            tokens.append(_parse_numeric_token(fields[prep.TASKS[task]["target_index"]], stamp_text))
        if not tokens or any(value != tokens[0] for value in tokens):
            raise RuntimeError("Duplicate target disagreement at " + stamp_text + "; STOP_RETAIN_FAILURE")
        values[stamp_text] = tokens[0]
        source_record_count += len(tokens)
        if len(tokens) > 1:
            duplicated.append({"timestamp": stamp_text, "source_records": len(tokens), "exact_agreement": True})
    receipt = {"task": task, "stage": stage, "phase": phase,
               "measurement_token_decodes": source_record_count, "unique_measurement_timestamps": len(stamps),
               "requested_timestamps": [_iso(stamp) for stamp in stamps], "maximum_allowed_timestamp": _iso(limit),
               "maximum_decoded_timestamp": _iso(max(stamps)), "archive_sha256": source["archive_sha256"],
               "csv_sha256": projection["csv_sha256"], "duplicate_agreement": duplicated,
               "other_measurement_values_decoded_or_parsed": 0, "outcome_based_row_filtering": False}
    return values, receipt


def _normalization(task, stage):
    path = ROOT / f"output/{task}/{stage}/normalization39.json"
    normal = read_json(path)
    if (normal.get("schema") != "step39-training-target-normalization-v1" or normal.get("task") != task
        or normal.get("stage") != stage or normal.get("fitted_partition") != "train"
        or F(normal["target_min"]) >= F(normal["target_max"])
        or F(normal["span"]) != F(normal["target_max"]) - F(normal["target_min"])):
        raise RuntimeError("Stage-specific training-only normalization differs")
    return normal


def _normalize(value, normalization):
    return (F(value) - F(normalization["target_min"])) / F(normalization["span"])


def _load_cohort(task, stage, partition):
    _task(task)
    _stage(stage)
    if partition == "train":
        _verify_training_transition(task, stage)
    elif partition == "validation" and stage != "final":
        verify_fit_seal(task, stage)
    else:
        raise RuntimeError("Bulk test access forbidden; use the chronological query API")
    base = f"output/{task}/{stage}"
    marker = ROOT / f"{base}/data_access_{partition}_started39.json"
    if marker.exists():
        raise RuntimeError("This data-access phase has already begun; no rerun")
    queries = calendar_queries(task, stage, partition)
    needed = set()
    for query in queries:
        target = _timestamp(query["target_timestamp"])
        needed.add(target)
        needed.update(_required_times(target)[1])
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("xb") as handle:
        handle.write(canonical_bytes({"task": task, "stage": stage, "phase": partition,
                                      "protocol_seal_sha256": sha_file(ROOT / "protocol_seal39.json")}))
    event(partition + "_measurement_access_started", task=task, stage=stage,
          protocol_seal_sha256=sha_file(ROOT / "protocol_seal39.json"),
          fit_seal_sha256=sha_file(ROOT / f"{base}/fit_seal39.json") if partition == "validation" else None,
          target_count=len(queries), unique_timestamps_requested=len(needed))
    try:
        values, access = _read_authorized_values(task, stage, partition, sorted(needed))
        targets = [values[q["target_timestamp"]] for q in queries]
        normal_path = ROOT / f"{base}/normalization39.json"
        if partition == "train":
            lo, hi = min(targets), max(targets)
            if lo == hi:
                write_json(ROOT / f"{base}/normalization_failure39.json", {
                    "status": "STOP_RETAIN_FAILURE", "task": task, "stage": stage,
                    "reason": "ZERO_TRAINING_TARGET_SPAN", "target_min": lo, "target_max": hi,
                    "span": "0", "access": access, "future_values_used": False})
                raise RuntimeError("Zero training target span: STOP_RETAIN_FAILURE")
            normal = {"schema": "step39-training-target-normalization-v1", "task": task, "stage": stage,
                      "fitted_partition": "train", "target_min": lo, "target_max": hi, "span": hi - lo,
                      "train_target_ids_sha256": digest([q["id"] for q in queries]),
                      "raw_train_targets_sha256": digest(targets), "target_values_clipped": False,
                      "future_data_used": False, "degenerate_range_action": "STOP_RETAIN_FAILURE"}
            write_json(normal_path, normal)
            normal = read_json(normal_path)
        else:
            normal = _normalization(task, stage)
        histories = []
        raw_persistence = []
        lag_timestamps = []
        for query in queries:
            origin, lags = _required_times(_timestamp(query["target_timestamp"]))
            histories.append([values[_iso(stamp)] for stamp in lags])
            raw_persistence.append(values[_iso(origin)])
            lag_timestamps.append([_iso(stamp) for stamp in lags])
        X = np.asarray([[float(value) for value in row] for row in histories], dtype=np.float64)
        if not np.isfinite(X).all():
            raise RuntimeError("Nonfinite input after raw decimal to binary64 conversion")
        X.flags.writeable = False
        X_path = f"{base}/cohorts/{partition}_X.npy"
        (ROOT / X_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(ROOT / X_path, X, allow_pickle=False)
        y = [_normalize(value, normal) for value in targets]
        persistence = [_normalize(value, normal) for value in raw_persistence]
        record = {"schema": "step39-causal-cohort-v1", "task": task, "stage": stage,
                  "name": partition, "target": prep.TASKS[task]["target_header"], "n": len(queries),
                  "ids": [q["id"] for q in queries], "target_timestamps": [q["target_timestamp"] for q in queries],
                  "origin_timestamps": [q["origin_timestamp"] for q in queries], "lag_timestamps": lag_timestamps,
                  "y": y, "raw_targets": targets, "persistence": persistence, "raw_persistence": raw_persistence,
                  "X_path": X_path, "X_sha256": sha_file(ROOT / X_path),
                  "normalization_path": str(normal_path.relative_to(ROOT)), "normalization_sha256": sha_file(normal_path),
                  "archive_sha256": access["archive_sha256"], "csv_sha256": access["csv_sha256"],
                  "other_measurement_predictors_used": False, "target_values_clipped": False}
        relative = f"{base}/cohorts/{partition}.json"
        write_json(ROOT / relative, record)
        access.update({"schema": "step39-development-measurement-access-v1", "status": "PASS",
                       "cohort_path": relative, "cohort_sha256": sha_file(ROOT / relative),
                       "normalization_sha256": record["normalization_sha256"]})
        access_relative = f"{base}/data_access_{partition}39.json"
        write_json(ROOT / access_relative, access)
        event(partition + "_measurement_access_completed", task=task, stage=stage,
              cohort_path=relative, cohort_sha256=sha_file(ROOT / relative), access_path=access_relative,
              access_sha256=sha_file(ROOT / access_relative), n=len(queries), test_target_values_decoded=0)
        return {**read_json(ROOT / relative), "X": X, "y": y, "persistence": persistence,
                "normalization": normal, "path": relative, "sha256": sha_file(ROOT / relative)}
    except Exception as exc:
        event(partition + "_measurement_access_failed", task=task, stage=stage,
              error_type=type(exc).__name__, message=str(exc))
        raise


def load_training(task, stage):
    return _load_cohort(task, stage, "train")


def load_validation(task, stage):
    return _load_cohort(task, stage, "validation")


def load_test_history(task, query):
    verify_model_seal(task)
    if _state_path(task).exists():
        state = _load_test_state(task)
    else:
        state = {"schema": "step39-chronological-test-access-state-v1", "task": task,
                 "model_seal_sha256": sha_file(ROOT / f"output/{task}/final/fit_seal39.json"),
                 "next_index": 0, "pending_query": None, "previously_revealed_test_targets": [], "status": "READY"}
    if state["status"] != "READY" or state.get("pending_query") is not None:
        raise RuntimeError("Reveal the current target before advancing; history cannot be reread")
    _verify_query(task, query, state["next_index"])
    state.update({"pending_query": dict(query), "status": "PENDING_HISTORY"})
    write_json(_state_path(task), state)
    origin, lags = _required_times(_timestamp(query["target_timestamp"]))
    event("test_history_access_started", task=task, query_id=query["id"], query_index=state["next_index"],
          target_timestamp=query["target_timestamp"], origin_timestamp=query["origin_timestamp"],
          model_seal_sha256=state["model_seal_sha256"], maximum_allowed_timestamp=_iso(origin))
    try:
        values, access = _read_authorized_values(task, "final", "test_history", lags)
        normal = _normalization(task, "final")
        X = np.asarray([[float(values[_iso(stamp)]) for stamp in lags]], dtype=np.float64)
        X.flags.writeable = False
        index = state["next_index"]
        X_path = f"output/{task}/test/history/{index:03d}_X.npy"
        (ROOT / X_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(ROOT / X_path, X, allow_pickle=False)
        overlap = [_iso(stamp) for stamp in lags if _iso(stamp) in state["previously_revealed_test_targets"]]
        normal_relative = f"output/{task}/final/normalization39.json"
        record = {"schema": "step39-causal-test-history-v1", "task": task, "query_id": query["id"],
                  "query_index": index, "target_timestamp": query["target_timestamp"],
                  "origin_timestamp": query["origin_timestamp"], "history_timestamps": [_iso(stamp) for stamp in lags],
                  "history_values": [values[_iso(stamp)] for stamp in lags],
                  "persistence": _normalize(values[_iso(origin)], normal), "raw_persistence": values[_iso(origin)],
                  "X_path": X_path, "X_sha256": sha_file(ROOT / X_path),
                  "model_seal_sha256": state["model_seal_sha256"], "normalization_sha256": sha_file(ROOT / normal_relative),
                  "previously_revealed_test_targets_in_history": overlap, "current_target_value_decoded": False,
                  "access": access, "other_measurement_predictors_used": False}
        relative = f"output/{task}/test/history/{index:03d}.json"
        write_json(ROOT / relative, record)
        state.update({"history_receipt_path": relative, "history_receipt_sha256": sha_file(ROOT / relative), "history_complete": True})
        write_json(_state_path(task), state)
        event("test_history_access_completed", task=task, query_id=query["id"], query_index=index,
              history_receipt_path=relative, history_receipt_sha256=sha_file(ROOT / relative),
              maximum_decoded_timestamp=access["maximum_decoded_timestamp"],
              previously_revealed_test_targets_in_history=overlap, current_target_value_decoded=False)
        return {**read_json(ROOT / relative), "X": X, "persistence": F(record["persistence"]),
                "normalization": normal, "history_receipt_path": relative, "history_receipt_sha256": sha_file(ROOT / relative)}
    except Exception as exc:
        state["status"] = "FAILED"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(_state_path(task), state)
        event("test_history_access_failed", task=task, query_id=query["id"], error=state["error"])
        raise


def reveal_test_target(task, query, prediction_path, prediction_sha256):
    protocol = verify_protocol_seal()
    verify_model_seal(task)
    state = _load_test_state(task)
    _verify_query(task, query, state["next_index"])
    if state["status"] != "PENDING_HISTORY" or state.get("pending_query") != query or not state.get("history_complete"):
        raise RuntimeError("Current history must be consumed once before its prediction seal")
    if sha_file(ROOT / state["history_receipt_path"]) != state["history_receipt_sha256"]:
        raise RuntimeError("Current test-history receipt changed")
    path, relative = _prediction_path(task, prediction_path)
    if sha_file(path) != prediction_sha256:
        raise RuntimeError("Provided prediction digest differs")
    prediction = read_json(path)
    ids = protocol["evaluation"]["prediction_ids"]
    if (not isinstance(ids, list) or len(ids) != len(set(ids)) or len(ids) != 207
        or not all(isinstance(i, str) and i for i in ids)
        or prediction.get("schema") != "step39-sealed-test-predictions-v1" or prediction.get("task") != task
        or prediction.get("query_id") != query["id"] or prediction.get("model_seal_sha256") != state["model_seal_sha256"]
        or prediction.get("history_receipt_sha256") != state["history_receipt_sha256"]
        or not isinstance(prediction.get("predictions"), dict) or set(prediction["predictions"]) != set(ids)):
        raise RuntimeError("Prediction record omits or changes a frozen query/model/history/forecast identity")
    index = state["next_index"]
    seal_relative = f"output/{task}/test/prediction_seals/{index:03d}.json"
    if (ROOT / seal_relative).exists():
        raise RuntimeError("This target already has a prediction seal")
    seal = {"schema": "step39-test-prediction-byte-seal-v1", "task": task, "query_id": query["id"], "query_index": index,
            "model_seal_sha256": state["model_seal_sha256"], "prediction_path": relative, "prediction_sha256": prediction_sha256,
            "history_receipt_path": state["history_receipt_path"], "history_receipt_sha256": state["history_receipt_sha256"],
            "prediction_ids": ids, "prediction_count": len(ids), "current_target_value_decoded": False}
    write_json(ROOT / seal_relative, seal)
    state.update({"status": "PREDICTIONS_SEALED", "prediction_seal_path": seal_relative,
                  "prediction_seal_sha256": sha_file(ROOT / seal_relative)})
    write_json(_state_path(task), state)
    event("test_predictions_sealed", task=task, query_id=query["id"], query_index=index,
          prediction_seal_path=seal_relative, prediction_seal_sha256=sha_file(ROOT / seal_relative),
          prediction_path=relative, prediction_sha256=prediction_sha256, prediction_count=len(ids), current_target_value_decoded=False)
    try:
        target = _timestamp(query["target_timestamp"])
        event("test_target_access_started", task=task, query_id=query["id"], query_index=index,
              prediction_seal_sha256=state["prediction_seal_sha256"], target_timestamp=query["target_timestamp"])
        values, access = _read_authorized_values(task, "final", "test_target", [target])
        normal = _normalization(task, "final")
        raw = values[_iso(target)]
        y = _normalize(raw, normal)
        record = {"schema": "step39-revealed-test-target-v1", "task": task, "query_id": query["id"], "query_index": index,
                  "target_timestamp": query["target_timestamp"], "origin_timestamp": query["origin_timestamp"],
                  "raw_target": raw, "y": y, "normalization_sha256": sha_file(ROOT / f"output/{task}/final/normalization39.json"),
                  "prediction_seal_path": seal_relative, "prediction_seal_sha256": state["prediction_seal_sha256"],
                  "model_seal_sha256": state["model_seal_sha256"], "access": access, "target_value_clipped": False}
        target_relative = f"output/{task}/test/targets/{index:03d}.json"
        write_json(ROOT / target_relative, record)
        state["previously_revealed_test_targets"].append(query["target_timestamp"])
        state["next_index"] += 1
        state["pending_query"] = None
        state["status"] = "COMPLETE" if state["next_index"] == EXPECTED_N[task]["test"] else "READY"
        for key in ("history_receipt_path", "history_receipt_sha256", "history_complete", "prediction_seal_path", "prediction_seal_sha256", "target_decode_started"):
            state.pop(key, None)
        write_json(_state_path(task), state)
        event("test_target_revealed", task=task, query_id=query["id"], query_index=index,
              target_path=target_relative, target_sha256=sha_file(ROOT / target_relative),
              prediction_seal_sha256=record["prediction_seal_sha256"], next_query_index=state["next_index"],
              chronological_test_status=state["status"])
        return {**read_json(ROOT / target_relative), "y": y, "path": target_relative, "sha256": sha_file(ROOT / target_relative)}
    except Exception as exc:
        state["status"] = "FAILED"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(_state_path(task), state)
        event("test_target_access_failed", task=task, query_id=query["id"], error=state["error"])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("metadata",))
    args = parser.parse_args()
    roster = metadata_rosters()
    print(json.dumps({"status": "PASS", "measurement_values_decoded_or_parsed": 0, "expected_n": EXPECTED_N}, sort_keys=True))
