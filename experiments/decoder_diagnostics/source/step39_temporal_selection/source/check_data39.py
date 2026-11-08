"""Synthetic-only access-control and decimal-projection checks for Step39.

These fixtures contain no public measurement values and fit no statistical
models. They exercise the real loader with compact artificial calendars,
independent-gate byte bindings, adversarial access attempts, and exact tokens.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
from fractions import Fraction as F
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

import data39 as d
import prepare_data39 as p

REAL_ROOT = d.ROOT
SOURCE_BYTES = {name: (REAL_ROOT / "source" / name).read_bytes()
                for name in ("data39.py", "prepare_data39.py", "check_data39.py")}
REAL_CONFIG = (d.ROOT, p.ROOT, copy.deepcopy(d.STAGES), d.TEST_START, d.TEST_STOP, copy.deepcopy(d.EXPECTED_N))
ORIGINAL_PARSE = d._parse_numeric_token
DECODED = []
CHECKS = []
IDS = [f"synthetic::candidate{i:03d}" for i in range(207)]
SMALL_STAGES = {
    "f1": {"train_start": "2017-01-01", "train_stop": "2017-01-03", "validation_start": "2017-01-03", "validation_stop": "2017-01-05"},
    "f2": {"train_start": "2017-01-01", "train_stop": "2017-01-05", "validation_start": "2017-01-05", "validation_stop": "2017-01-07"},
    "f3": {"train_start": "2017-01-01", "train_stop": "2017-01-07", "validation_start": "2017-01-07", "validation_stop": "2017-01-09"},
    "final": {"train_start": "2017-01-01", "train_stop": "2017-01-09"},
}
SMALL_N = {task: {"f1": {"train": 2, "validation": 2}, "f2": {"train": 4, "validation": 2},
                  "f3": {"train": 6, "validation": 2}, "final": {"train": 8}, "test": 2}
           for task in d.task_ids()}


def track(token, stamp):
    DECODED.append((stamp, bytes(token)))
    return ORIGINAL_PARSE(token, stamp)


def check(name, condition):
    CHECKS.append({"name": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def blocked(name, operation, no_decode=True):
    before = len(DECODED)
    try:
        operation()
    except (RuntimeError, FileNotFoundError, ValueError):
        check(name, not no_decode or len(DECODED) == before)
        return
    raise AssertionError("Forbidden operation succeeded: " + name)


def artificial_value(stamp, constant=False):
    if constant:
        return F(7, 10)
    day = (stamp - datetime(2016, 12, 31)).days
    return F(day * 100 + stamp.hour * 10 + 1, 10)


def artificial_csv(task, constant=False, bad_target=None, opaque=False):
    spec = p.TASKS[task]
    headers = (["DateTime", "Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows",
                "Zone 1 Power Consumption", "Zone 2 Power Consumption", "Zone 3 Power Consumption"]
               if task == "tetouan_zone1" else
               ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "weather_description", "date_time", "traffic_volume"])
    lines = [(b",".join(x.encode() for x in headers) + b"\n")]
    stamp = datetime(2016, 12, 31)
    stop = datetime(2017, 1, 11)
    while stamp < stop:
        fields = [b"OPAQUE" for _ in range(9)]
        fields[spec["timestamp_index"]] = stamp.strftime(spec["timestamp_format"]).encode()
        value = artificial_value(stamp, constant)
        token = f"{value.numerator // value.denominator}.{int((value % 1) * 10)}".encode()
        if opaque:
            token = b"NOT_A_NUMBER"
        if bad_target == "invalid" and stamp == datetime(2017, 1, 9, 23):
            token = b"NaN"
        fields[spec["target_index"]] = token
        if task == "metro_traffic":
            fields[6] = b'"synthetic, weather"'
        lines.append(b",".join(fields) + b"\n")
        if task == "metro_traffic" and (stamp.hour % 6 == 0 or stamp == datetime(2017, 1, 9, 23)):
            duplicate = list(fields)
            if bad_target == "disagree" and stamp == datetime(2017, 1, 9, 23):
                duplicate[spec["target_index"]] = b"999999"
            lines.append(b",".join(duplicate) + b"\n")
        stamp += timedelta(minutes=spec["cadence_minutes"])
    return b"".join(lines)


class Fixture:
    def __init__(self, constant=False, bad_target=None):
        self.temporary = tempfile.TemporaryDirectory(prefix="data39_synthetic_", dir=REAL_ROOT / "tmp")
        self.root = Path(self.temporary.name)
        d.ROOT = self.root
        p.ROOT = self.root
        d.STAGES = copy.deepcopy(SMALL_STAGES)
        d.TEST_START = "2017-01-09"
        d.TEST_STOP = "2017-01-11"
        d.EXPECTED_N = copy.deepcopy(SMALL_N)
        d._parse_numeric_token = track
        for name, data in SOURCE_BYTES.items():
            (self.root / "source").mkdir(exist_ok=True)
            (self.root / "source" / name).write_bytes(data)
        for task in d.task_ids():
            raw = artificial_csv(task, constant=constant, bad_target=bad_target)
            container = io.BytesIO()
            with zipfile.ZipFile(container, "w", zipfile.ZIP_DEFLATED) as zipped:
                member = gzip.compress(raw, mtime=0) if p.TASKS[task]["inner_compression"] else raw
                zipped.writestr(p.TASKS[task]["member"], member)
            archive = self.root / p.archive_relative(task)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(container.getvalue())
            source = {**p.TASKS[task], "task": task, "archive_path": p.archive_relative(task),
                      "archive_sha256": d.sha_file(archive), "artificial_only": True}
            d.write_json(self.root / f"inputs/{task}/public_source39.json", source)
            p.project(task)
        d.write_json(self.root / "inputs/prior_exposure39.json", {"artificial_only": True})
        d.metadata_rosters()
        protocol = {"step": 39, "data": d.data_contract(), "evaluation": {"prediction_ids": IDS}, "artificial_only": True}
        d.write_json(self.root / "protocol39.json", protocol)

    def close(self):
        d.ROOT, p.ROOT, d.STAGES, d.TEST_START, d.TEST_STOP, d.EXPECTED_N = REAL_CONFIG
        d._parse_numeric_token = ORIGINAL_PARSE
        self.temporary.cleanup()


def protocol_seal(gate=True):
    seal = {"artificial_only": True, "bound_files": {relative: d.sha_file(d.ROOT / relative) for relative in sorted(d.DATA_REQUIRED)}}
    d.write_json(d.ROOT / "protocol_seal39.json", seal)
    if gate:
        d.write_json(d.ROOT / "audit/preexecution_gate39.json",
                     {"status": "PASS", "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal39.json"), "artificial_only": True})


def fit_seal(task, stage, gate=True):
    base = f"output/{task}/{stage}"
    relative_files = [f"{base}/normalization39.json", f"{base}/cohorts/train.json", f"{base}/cohorts/train_X.npy"]
    seal = {"schema": "step39-fit-seal-v1", "task": task, "stage": stage,
            "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal39.json"), "artificial_only": True,
            "manifest_sha256": "artificial_no_model_fit",
            "bound_files": {relative: d.sha_file(d.ROOT / relative) for relative in relative_files}}
    path = d.ROOT / f"{base}/fit_seal39.json"
    d.write_json(path, seal)
    if gate:
        d.write_json(d.ROOT / f"audit/{task}/{stage}/fit_gate39.json",
                     {"status": "PASS", "task": task, "stage": stage, "protocol_seal_sha256": seal["protocol_seal_sha256"],
                      "fit_seal_sha256": d.sha_file(path), "artificial_only": True})


def completion_seal(task, stage, gate=True):
    base = f"output/{task}/{stage}"
    files = [f"{base}/validation_predictions39.json", f"{base}/validation_scores39.json"]
    for relative in files:
        d.write_json(d.ROOT / relative, {"artificial_only": True, "no_model_fits": True})
    path = d.ROOT / f"{base}/completion_seal39.json"
    seal = {"schema": "step39-fold-completion-seal-v1", "task": task, "stage": stage,
            "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal39.json"),
            "bound_files": {relative: d.sha_file(d.ROOT / relative) for relative in files}, "artificial_only": True}
    d.write_json(path, seal)
    if gate:
        d.write_json(d.ROOT / f"audit/{task}/{stage}/validation_gate39.json",
                     {"status": "PASS", "task": task, "stage": stage, "protocol_seal_sha256": seal["protocol_seal_sha256"],
                      "completion_seal_sha256": d.sha_file(path), "artificial_only": True})


def selection_seal(task, gate=True):
    relative = f"output/{task}/selection39.json"
    d.write_json(d.ROOT / relative, {"artificial_only": True, "selected_ids": IDS[:1]})
    path = d.ROOT / f"output/{task}/selection_seal39.json"
    seal = {"schema": "step39-selection-seal-v1", "task": task,
            "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal39.json"),
            "bound_files": {relative: d.sha_file(d.ROOT / relative)}, "selected_ids": IDS[:1], "artificial_only": True}
    d.write_json(path, seal)
    if gate:
        d.write_json(d.ROOT / f"audit/{task}/selection_gate39.json",
                     {"status": "PASS", "task": task, "protocol_seal_sha256": seal["protocol_seal_sha256"],
                      "selection_seal_sha256": d.sha_file(path), "artificial_only": True})


def pretest_gate(task):
    d.write_json(d.ROOT / f"audit/{task}/pretest_gate39.json", {
        "status": "PASS", "task": task, "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal39.json"),
        "fit_seal_sha256": d.sha_file(d.ROOT / f"output/{task}/final/fit_seal39.json"),
        "selection_seal_sha256": d.sha_file(d.ROOT / f"output/{task}/selection_seal39.json"), "artificial_only": True})


def prediction(task, query, history, change=None):
    record = {"schema": "step39-sealed-test-predictions-v1", "task": task, "query_id": query["id"],
              "model_seal_sha256": history["model_seal_sha256"], "history_receipt_sha256": history["history_receipt_sha256"],
              "predictions": {key: {"lower": "0", "upper": "0"} for key in IDS}, "artificial_only": True}
    if change is not None:
        change(record)
    relative = f"output/{task}/test/predictions/q{history['query_index']:03d}.json"
    d.write_json(d.ROOT / relative, record)
    return relative, d.sha_file(d.ROOT / relative)


def complete_development(task):
    for stage in ("f1", "f2", "f3"):
        d.load_training(task, stage)
        fit_seal(task, stage)
        d.load_validation(task, stage)
        completion_seal(task, stage)
    selection_seal(task)
    d.load_training(task, "final")
    fit_seal(task, "final")
    pretest_gate(task)


def audit_event_chain():
    previous = None
    for line in (d.ROOT / "output/access_and_execution39.jsonl").read_text().splitlines():
        event = json.loads(line)
        saved = event.pop("event_sha256")
        if d.digest(event) != saved or event["previous_event_sha256"] != previous:
            return False
        previous = saved
    return True


def main_fixture():
    fixture = Fixture()
    try:
        before = len(DECODED)
        for task in d.task_ids():
            projected = p.calendar_projection(task, artificial_csv(task, opaque=True))
            check(task + "_metadata_never_decodes_invalid_measurements", projected["target_values_decoded_or_parsed"] == 0 and len(DECODED) == before)
        task = "tetouan_zone1"
        blocked("training_requires_protocol_seal", lambda: d.load_training(task, "f1"))
        blocked("private_source_entry_requires_protocol_seal", lambda: d._read_authorized_values(task, "f1", "train", []))
        protocol_seal(gate=False)
        blocked("training_requires_independent_preexecution_gate", lambda: d.load_training(task, "f1"))
        protocol_seal()
        archive = d.ROOT / p.archive_relative(task)
        raw = archive.read_bytes()
        archive.write_bytes(raw + b"tampered")
        blocked("changed_source_blocks_before_measurement_read", lambda: d.load_training(task, "f1"))
        archive.write_bytes(raw)
        for task in d.task_ids():
            training = d.load_training(task, "f1")
            check(task + "_training_exact_decimal_targets", all(F(v) == artificial_value(d._timestamp(t)) for v, t in zip(training["raw_targets"], training["target_timestamps"])))
            check(task + "_training_raw_input_order", training["X"][0].tolist() == [float(artificial_value(d._timestamp(t))) for t in training["lag_timestamps"][0]])
            check(task + "_training_normalization_own_extrema", training["y"] == [F(0), F(1)] and F(training["normalization"]["span"]) == F(10))
            norm_path = d.ROOT / f"output/{task}/f1/normalization39.json"
            old_norm_sha = d.sha_file(norm_path)
            blocked(task + "_training_rerun_blocked", lambda: d.load_training(task, "f1"))
            blocked(task + "_nextfold_requires_previous_scores", lambda: d.load_training(task, "f2"))
            blocked(task + "_validation_requires_fit_seal", lambda: d.load_validation(task, "f1"))
            fit_seal(task, "f1", gate=False)
            blocked(task + "_validation_requires_independent_fit_gate", lambda: d.load_validation(task, "f1"))
            fit_seal(task, "f1")
            validation = d.load_validation(task, "f1")
            check(task + "_validation_original_labels_not_clipped", min(validation["y"]) > 1)
            check(task + "_validation_keeps_training_normalization", d.sha_file(norm_path) == old_norm_sha)
            blocked(task + "_validation_rerun_blocked", lambda: d.load_validation(task, "f1"))
            completion_seal(task, "f1", gate=False)
            blocked(task + "_nextfold_requires_validation_gate", lambda: d.load_training(task, "f2"))
            completion_seal(task, "f1")
            for stage in ("f2", "f3"):
                train = d.load_training(task, stage)
                check(task + "_" + stage + "_expanding_training_prefix", train["ids"][:2] == training["ids"])
                fit_seal(task, stage)
                d.load_validation(task, stage)
                completion_seal(task, stage)
            blocked(task + "_finalrefit_requires_selectionseal", lambda: d.load_training(task, "final"))
            selection_seal(task, gate=False)
            blocked(task + "_finalrefit_requires_independent_selectiongate", lambda: d.load_training(task, "final"))
            selection_seal(task)
            final = d.load_training(task, "final")
            check(task + "_finalrefit_own_normalization", F(final["normalization"]["span"]) == 70 and d.sha_file(norm_path) == old_norm_sha)
            fit_seal(task, "final")
            queries = d.test_queries(task)
            blocked(task + "_test_requires_pretestgate", lambda: d.load_test_history(task, queries[0]))
            pretest_gate(task)
            blocked(task + "_outoforder_testquery_blocked", lambda: d.load_test_history(task, queries[1]))
            blocked(task + "_target_requires_history_and_predictionseal", lambda: d.reveal_test_target(task, queries[0], "missing", "wrong"))
            for index, query in enumerate(queries):
                history = d.load_test_history(task, query)
                check(task + f"_q{index}_causal_history", max(history["history_timestamps"]) == query["origin_timestamp"] and all(t < query["target_timestamp"] for t in history["history_timestamps"]))
                if index == 0:
                    blocked(task + "_duplicate_history_blocked", lambda: d.load_test_history(task, query))
                    blocked(task + "_private_target_entry_requires_predictionseal", lambda: d._read_authorized_values(task, "final", "test_target", [d._timestamp(query["target_timestamp"])]))
                    bad = prediction(task, query, history, lambda x: x["predictions"].pop(IDS[-1]))
                    blocked(task + "_incomplete_prediction_panel_blocks_target", lambda: d.reveal_test_target(task, query, *bad))
                    wrong = prediction(task, query, history, lambda x: x.update(model_seal_sha256="wrong"))
                    blocked(task + "_wrong_model_binding_blocks_target", lambda: d.reveal_test_target(task, query, *wrong))
                relative, sha = prediction(task, query, history)
                blocked(task + f"_q{index}_wrong_prediction_digest_blocks_target", lambda: d.reveal_test_target(task, query, relative, "wrong"))
                target = d.reveal_test_target(task, query, relative, sha)
                check(task + f"_q{index}_target_exact_and_unclipped", F(target["raw_target"]) == artificial_value(d._timestamp(query["target_timestamp"])) and target["y"] > 1)
                if index == 1:
                    check(task + "_previous_target_legitimately_becomes_history", history["previously_revealed_test_targets_in_history"] == [queries[0]["target_timestamp"]])
                blocked(task + f"_q{index}_target_cannot_be_revealed_twice", lambda: d.reveal_test_target(task, query, relative, sha))
            state = d._load_test_state(task)
            check(task + "_chronological_state_complete", state["status"] == "COMPLETE" and state["next_index"] == 2)
            if task == "metro_traffic":
                receipt = d.read_json(d.ROOT / f"output/{task}/f1/data_access_train39.json")
                check("metro_duplicate_exact_agreement_retained", bool(receipt["duplicate_agreement"]) and receipt["measurement_token_decodes"] > receipt["unique_measurement_timestamps"])
        check("global_event_hash_chain_valid", audit_event_chain())
    finally:
        fixture.close()


def failure_fixtures():
    fixture = Fixture(constant=True)
    try:
        protocol_seal()
        before = len(DECODED)
        blocked("zero_training_span_stops_without_fallback", lambda: d.load_training("tetouan_zone1", "f1"), no_decode=False)
        record = d.read_json(d.ROOT / "output/tetouan_zone1/f1/normalization_failure39.json")
        check("zero_span_failure_retained_without_future_access", record["status"] == "STOP_RETAIN_FAILURE" and not (d.ROOT / "output/tetouan_zone1/f1/normalization39.json").exists() and max(stamp for stamp, token in DECODED[before:]) < "2017-01-03T00:00:00")
    finally:
        fixture.close()
    for kind, task in (("invalid", "tetouan_zone1"), ("disagree", "metro_traffic")):
        fixture = Fixture(bad_target=kind)
        try:
            protocol_seal()
            complete_development(task)
            query = d.test_queries(task)[0]
            before = len(DECODED)
            history = d.load_test_history(task, query)
            check(kind + "_unseen_test_problem_not_read_in_history", all(stamp <= query["origin_timestamp"] for stamp, token in DECODED[before:]))
            path, sha = prediction(task, query, history)
            blocked(kind + "_target_stops_after_seal", lambda: d.reveal_test_target(task, query, path, sha), no_decode=False)
            state = d.read_json(d._state_path(task))
            check(kind + "_failure_retained_without_row_dropping", state["status"] == "FAILED" and state["next_index"] == 0 and not (d.ROOT / f"output/{task}/test/targets/000.json").exists())
            blocked(kind + "_failed_study_cannot_continue", lambda: d.load_test_history(task, query))
        finally:
            fixture.close()


def main():
    try:
        main_fixture()
        failure_fixtures()
    finally:
        d.ROOT, p.ROOT, d.STAGES, d.TEST_START, d.TEST_STOP, d.EXPECTED_N = REAL_CONFIG
        d._parse_numeric_token = ORIGINAL_PARSE
    result = {"schema": "step39-data-access-artificial-preflight-v1", "status": "PASS", "checks": CHECKS,
              "artificial_only": True, "public_measurement_values_decoded_or_parsed": 0, "statistical_model_fits": 0,
              "synthetic_measurement_token_decodes": len(DECODED), "test_fixture_tasks": list(SMALL_N),
              "source_sha256": {f"source/{name}": hashlib.sha256(raw).hexdigest() for name, raw in SOURCE_BYTES.items()},
              "production_expected_n": d.EXPECTED_N, "production_data_configuration_restored": d.TEST_START == "2017-09-01"}
    d.write_json(REAL_ROOT / "tests/data_preflight39.json", result)
    print(json.dumps({"status": "PASS", "checks": len(CHECKS), "public_measurement_values_decoded_or_parsed": 0,
                      "statistical_model_fits": 0, "synthetic_measurement_token_decodes": len(DECODED)}, sort_keys=True))


if __name__ == "__main__":
    main()
