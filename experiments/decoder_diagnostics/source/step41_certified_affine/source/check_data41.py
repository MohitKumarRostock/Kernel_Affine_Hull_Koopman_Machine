"""Synthetic-only access-control and decimal-projection checks for Step41.

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

import data41 as d
import prepare_data41 as p

REAL_ROOT = d.ROOT
SOURCE_BYTES = {name: (REAL_ROOT / "source" / name).read_bytes()
                for name in ("data41.py", "prepare_data41.py", "check_data41.py")}
REAL_CONFIG = (d.ROOT, p.ROOT, d.task_stages, d.test_dates, copy.deepcopy(d.EXPECTED_N))
ORIGINAL_PARSE = d._parse_numeric_token
DECODED = []
CHECKS = []
IDS = [f"synthetic::candidate{i:03d}" for i in range(62)]
SMALL_N = {task: {"f1": {"train": 2, "validation": 2}, "f2": {"train": 4, "validation": 2},
                  "f3": {"train": 6, "validation": 2}, "final": {"train": 8}, "test": 2}
           for task in d.task_ids()}
INTEGRATION_N = {task: {"f1": {"train": 64, "validation": 8}, "f2": {"train": 72, "validation": 8},
                        "f3": {"train": 80, "validation": 8}, "final": {"train": 88}, "test": 2}
                 for task in d.task_ids()}


def integration_stages(task):
    start = datetime(p.TASKS[task]["year"], 1, 1)
    date = lambda days: (start + timedelta(days=days)).date().isoformat()
    return {stage: {"train_start": date(0), "train_stop": date(n),
                    **({"validation_start": date(n), "validation_stop": date(n + 8)} if stage != "final" else {})}
            for stage, n in (("f1", 64), ("f2", 72), ("f3", 80), ("final", 88))}


def integration_test_dates(task):
    start = datetime(p.TASKS[task]["year"], 1, 1)
    return tuple((start + timedelta(days=n)).date().isoformat() for n in (88, 90))


def integration_value(stamp, year):
    hour = int((stamp - datetime(year, 1, 1)).total_seconds() // 3600)
    return F(2000 + (hour * 17) % 271 + (hour * hour) % 19 + hour // 24, 10)


def small_stages(task):
    year = p.TASKS[task]["year"]
    return {
        "f1": {"train_start": f"{year}-01-01", "train_stop": f"{year}-01-03", "validation_start": f"{year}-01-03", "validation_stop": f"{year}-01-05"},
        "f2": {"train_start": f"{year}-01-01", "train_stop": f"{year}-01-05", "validation_start": f"{year}-01-05", "validation_stop": f"{year}-01-07"},
        "f3": {"train_start": f"{year}-01-01", "train_stop": f"{year}-01-07", "validation_start": f"{year}-01-07", "validation_stop": f"{year}-01-09"},
        "final": {"train_start": f"{year}-01-01", "train_stop": f"{year}-01-09"},
    }


def small_test_dates(task):
    year = p.TASKS[task]["year"]
    return f"{year}-01-09", f"{year}-01-11"


def track(task, token, stamp):
    DECODED.append((stamp, bytes(token)))
    return ORIGINAL_PARSE(task, token, stamp)


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
    day = stamp.day if stamp.month == 1 else 0
    return F(day * 100 + stamp.hour * 10 + 1, 10)


def artificial_csv(task, constant=False, bad_target=None, opaque=False, integration=False):
    spec = p.TASKS[task]; year = spec["year"]
    headers = (["Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
               if task == "household_power2010" else
               ["Date Time", "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m²)", "PAR (µmol/m²/s)", "max. PAR (µmol/m²/s)", "Tlog (degC)", "CO2 (ppm)"])
    separator = spec["separator"].encode()
    heading = separator.join(h.encode(spec["header_encoding"]) for h in headers) + b"\n"
    payloads = {a["source_id"]: [heading] for a in spec["archives"]}
    stamp = datetime(year - 1, 12, 31)
    stop = datetime(year, 1, 1) + timedelta(days=90) if integration else datetime(year, 1, 11)
    split = datetime(year, 1, 1) + timedelta(days=45) if integration else datetime(year, 1, 5)
    while stamp < stop:
        fields = [b"OPAQUE"] * len(headers)
        if task == "household_power2010":
            fields[0] = stamp.strftime("%d/%m/%Y").encode(); fields[1] = stamp.strftime("%H:%M:%S").encode()
            source_id = "household"
        else:
            fields[0] = stamp.strftime("%d.%m.%Y %H:%M:%S").encode()
            source_id = "jena2017a" if stamp < split else "jena2017b"
        value = integration_value(stamp, year) if integration else artificial_value(stamp, constant)
        token = f"{value.numerator // value.denominator}.{int((value % 1) * 10)}".encode()
        if opaque:
            token = b"NOT_A_NUMBER"
        if bad_target == "invalid" and stamp == datetime(year, 1, 9, 23):
            token = b"NaN"
        fields[spec["target_index"]] = token
        payloads[source_id].append(separator.join(fields) + b"\n")
        if task == "jena_temperature2017" and (stamp.hour % 6 == 0 or stamp == datetime(year, 1, 9, 23)):
            duplicate = list(fields)
            if bad_target == "disagree" and stamp == datetime(year, 1, 9, 23):
                duplicate[spec["target_index"]] = b"999999"
            payloads[source_id].append(separator.join(duplicate) + b"\n")
        stamp += timedelta(hours=1)
    return {k: b"".join(v) for k, v in payloads.items()}


class Fixture:
    def __init__(self, constant=False, bad_target=None, *, integration=False, persist=False, write_minimal_protocol=True):
        (REAL_ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = None if persist else tempfile.TemporaryDirectory(prefix="data41_synthetic_", dir=REAL_ROOT / "tmp")
        self.root = Path(tempfile.mkdtemp(prefix="data41_integration_", dir=REAL_ROOT / "tmp") if persist else self.temporary.name)
        d.ROOT = self.root; p.ROOT = self.root
        d.task_stages = integration_stages if integration else small_stages
        d.test_dates = integration_test_dates if integration else small_test_dates
        d.EXPECTED_N = copy.deepcopy(INTEGRATION_N if integration else SMALL_N)
        d._parse_numeric_token = track
        for name, raw in SOURCE_BYTES.items():
            (self.root / "source").mkdir(exist_ok=True)
            (self.root / "source" / name).write_bytes(raw)
        for task in d.task_ids():
            payloads = artificial_csv(task, constant=constant, bad_target=bad_target, integration=integration)
            archives = []
            for declared, relative in zip(p.TASKS[task]["archives"], p.archive_relatives(task)):
                container = io.BytesIO()
                with zipfile.ZipFile(container, "w", zipfile.ZIP_DEFLATED) as zipped:
                    zipped.writestr(declared["member"], payloads[declared["source_id"]])
                archive = self.root / relative; archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(container.getvalue())
                with zipfile.ZipFile(archive) as zipped:
                    inventory = [{"name": item.filename, "bytes": item.file_size,
                                  "compressed_bytes": item.compress_size} for item in zipped.infolist()]
                fixture_url = "artificial://fixture/" + declared["source_id"]
                archives.append({**declared, "archive_path": relative, "archive_sha256": d.sha_file(archive),
                                 "archive_bytes": archive.stat().st_size, "source_url": fixture_url,
                                 "resolved_url": fixture_url, "inventory": inventory})
            metadata_path = f"inputs/data41/{task}/artificial_metadata.txt"
            (self.root / metadata_path).write_text("Artificial fixture only. No public measurement data.\n")
            source = {**p.TASKS[task], "task": task, "archives": archives,
                      "schema": "step41-artificial-opaque-source-v1",
                      "source_manifest_sha256": p.sha_bytes(p.canonical_bytes(p.source_manifest(archives))),
                      "metadata_path": metadata_path, "metadata_sha256": d.sha_file(self.root / metadata_path),
                      "artificial_only": True, "measurement_values_decoded_or_parsed": 0,
                      "timestamp_values_parsed": 0, "scientific_model_fits": 0,
                      "absolute_historical_never_seen_claim": False}
            d.write_json(self.root / f"inputs/data41/{task}/public_source41.json", source)
            p.project(task)
        d.write_json(self.root / "inputs/data41/prior_exposure41.json", {"artificial_only": True})
        d.metadata_rosters()
        if write_minimal_protocol:
            protocol = {"step": 41, "data": d.data_contract(), "evaluation": {"prediction_ids": IDS}, "artificial_only": True}
            d.write_json(self.root / "protocol41.json", protocol)

    def close(self):
        d.ROOT, p.ROOT, d.task_stages, d.test_dates, d.EXPECTED_N = REAL_CONFIG
        d._parse_numeric_token = ORIGINAL_PARSE
        p._member_bytes.cache_clear()
        if self.temporary is not None:
            self.temporary.cleanup()


class IntegrationFixture(Fixture):
    """Retained artificial raw-data fixture for the actual Step41 runner/auditor.

    All ordinary source access gates and exact-rational data receipts remain in
    force. The caller redirects common41/run41/audit41 roots, creates an
    explicitly artificial protocol from the patched data contract, binds source
    and input dependencies, and invokes its normal stages. This class does not
    write any PASS gates and does not parse any public measurement values.
    Calling close restores production module configuration while retaining the
    isolated fixture files for audit/replay.
    """
    def __init__(self):
        super().__init__(integration=True, persist=True, write_minimal_protocol=False)


def protocol_seal(gate=True):
    seal = {"artificial_only": True, "bound_files": {relative: d.sha_file(d.ROOT / relative) for relative in sorted(d.required_data_files())}}
    d.write_json(d.ROOT / "protocol_seal41.json", seal)
    if gate:
        d.write_json(d.ROOT / "audit/preexecution_gate41.json",
                     {"status": "PASS", "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal41.json"), "artificial_only": True})


def fit_seal(task, stage, gate=True):
    base = f"output/{task}/{stage}"
    relative_files = [f"{base}/normalization41.json", f"{base}/cohorts/train.json", f"{base}/cohorts/train_X.npy"]
    seal = {"schema": "step41-fit-seal-v1", "task": task, "stage": stage,
            "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal41.json"), "artificial_only": True,
            "manifest_sha256": "artificial_no_model_fit",
            "bound_files": {relative: d.sha_file(d.ROOT / relative) for relative in relative_files}}
    path = d.ROOT / f"{base}/fit_seal41.json"
    d.write_json(path, seal)
    if gate:
        d.write_json(d.ROOT / f"audit/{task}/{stage}/fit_gate41.json",
                     {"status": "PASS", "task": task, "stage": stage, "protocol_seal_sha256": seal["protocol_seal_sha256"],
                      "fit_seal_sha256": d.sha_file(path), "artificial_only": True})


def completion_seal(task, stage, gate=True):
    base = f"output/{task}/{stage}"
    files = [f"{base}/validation_predictions41.json", f"{base}/validation_scores41.json"]
    for relative in files:
        d.write_json(d.ROOT / relative, {"artificial_only": True, "no_model_fits": True})
    path = d.ROOT / f"{base}/completion_seal41.json"
    seal = {"schema": "step41-fold-completion-seal-v1", "task": task, "stage": stage,
            "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal41.json"),
            "bound_files": {relative: d.sha_file(d.ROOT / relative) for relative in files}, "artificial_only": True}
    d.write_json(path, seal)
    if gate:
        d.write_json(d.ROOT / f"audit/{task}/{stage}/validation_gate41.json",
                     {"status": "PASS", "task": task, "stage": stage, "protocol_seal_sha256": seal["protocol_seal_sha256"],
                      "completion_seal_sha256": d.sha_file(path), "artificial_only": True})


def selection_seal(task, gate=True):
    relative = f"output/{task}/selection41.json"
    d.write_json(d.ROOT / relative, {"artificial_only": True, "selected_ids": IDS[:1]})
    path = d.ROOT / f"output/{task}/selection_seal41.json"
    seal = {"schema": "step41-selection-seal-v1", "task": task,
            "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal41.json"),
            "bound_files": {relative: d.sha_file(d.ROOT / relative)}, "selected_ids": IDS[:1], "artificial_only": True}
    d.write_json(path, seal)
    if gate:
        d.write_json(d.ROOT / f"audit/{task}/selection_gate41.json",
                     {"status": "PASS", "task": task, "protocol_seal_sha256": seal["protocol_seal_sha256"],
                      "selection_seal_sha256": d.sha_file(path), "artificial_only": True})


def pretest_gate(task):
    d.write_json(d.ROOT / f"audit/{task}/pretest_gate41.json", {
        "status": "PASS", "task": task, "protocol_seal_sha256": d.sha_file(d.ROOT / "protocol_seal41.json"),
        "fit_seal_sha256": d.sha_file(d.ROOT / f"output/{task}/final/fit_seal41.json"),
        "selection_seal_sha256": d.sha_file(d.ROOT / f"output/{task}/selection_seal41.json"), "artificial_only": True})


def prediction(task, query, history, change=None):
    record = {"schema": "step41-sealed-test-predictions-v1", "task": task, "query_id": query["id"],
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
    for line in (d.ROOT / "output/access_and_execution41.jsonl").read_text().splitlines():
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
        task = "household_power2010"
        blocked("training_requires_protocol_seal", lambda: d.load_training(task, "f1"))
        blocked("private_source_entry_requires_protocol_seal", lambda: d._read_authorized_values(task, "f1", "train", []))
        protocol_seal(gate=False)
        blocked("training_requires_independent_preexecution_gate", lambda: d.load_training(task, "f1"))
        protocol_seal()
        archive = d.ROOT / p.archive_relatives(task)[0]
        raw = archive.read_bytes()
        archive.write_bytes(raw + b"tampered")
        blocked("changed_source_blocks_before_measurement_read", lambda: d.load_training(task, "f1"))
        archive.write_bytes(raw)
        for task in d.task_ids():
            blocked(task + "_heldout_future_training_token_refused", lambda: d._read_authorized_values(task, "f1", "train", [datetime(p.TASKS[task]["year"], 1, 10)]))
            training = d.load_training(task, "f1")
            check(task + "_training_exact_decimal_targets", all(F(v) == artificial_value(d._timestamp(t)) for v, t in zip(training["raw_targets"], training["target_timestamps"])))
            check(task + "_training_raw_input_order", training["X"][0].tolist() == [float(artificial_value(d._timestamp(t))) for t in training["lag_timestamps"][0]])
            check(task + "_training_normalization_own_extrema", training["y"] == [F(0), F(1)] and F(training["normalization"]["span"]) == F(10))
            norm_path = d.ROOT / f"output/{task}/f1/normalization41.json"
            old_norm_sha = d.sha_file(norm_path)
            blocked(task + "_training_rerun_blocked", lambda: d.load_training(task, "f1"))
            blocked(task + "_nextfold_requires_previous_scores", lambda: d.load_training(task, "f2"))
            blocked(task + "_validation_requires_fit_seal", lambda: d.load_validation(task, "f1"))
            fit_seal(task, "f1", gate=False)
            blocked(task + "_validation_requires_independent_fit_gate", lambda: d.load_validation(task, "f1"))
            fit_seal(task, "f1")
            gatepath = d.ROOT / f"audit/{task}/f1/fit_gate41.json"
            original_gate = d.read_json(gatepath)
            d.write_json(gatepath, {**original_gate, "fit_seal_sha256": "wrong"})
            blocked(task + "_wrong_fitgate_binding_blocks_validation", lambda: d.load_validation(task, "f1"))
            d.write_json(gatepath, original_gate)
            blocked(task + "_heldout_future_validation_token_refused", lambda: d._read_authorized_values(task, "f1", "validation", [datetime(p.TASKS[task]["year"], 1, 10)]))
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
            gatepath = d.ROOT / f"audit/{task}/selection_gate41.json"
            original_gate = d.read_json(gatepath)
            d.write_json(gatepath, {**original_gate, "selection_seal_sha256": "wrong"})
            blocked(task + "_wrong_selectiongate_binding_blocks_finalrefit", lambda: d.load_training(task, "final"))
            d.write_json(gatepath, original_gate)
            final = d.load_training(task, "final")
            check(task + "_finalrefit_own_normalization", F(final["normalization"]["span"]) == 70 and d.sha_file(norm_path) == old_norm_sha)
            fit_seal(task, "final")
            queries = d.test_queries(task)
            blocked(task + "_test_requires_pretestgate", lambda: d.load_test_history(task, queries[0]))
            pretest_gate(task)
            gatepath = d.ROOT / f"audit/{task}/pretest_gate41.json"
            original_gate = d.read_json(gatepath)
            d.write_json(gatepath, {**original_gate, "fit_seal_sha256": "wrong"})
            blocked(task + "_wrong_pretestgate_binding_blocks_history", lambda: d.load_test_history(task, queries[0]))
            d.write_json(gatepath, original_gate)
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
                    wrong_query = prediction(task, query, history, lambda x: x.update(query_id=queries[1]["id"]))
                    blocked(task + "_wrong_query_prediction_blocks_target", lambda: d.reveal_test_target(task, query, *wrong_query))
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
            if task == "jena_temperature2017":
                receipt = d.read_json(d.ROOT / f"output/{task}/f1/data_access_train41.json")
                check("jena_multimember_duplicate_exact_agreement_retained", bool(receipt["duplicate_agreement"]) and receipt["measurement_token_decodes"] > receipt["unique_measurement_timestamps"])
        check("global_event_hash_chain_valid", audit_event_chain())
    finally:
        fixture.close()


def failure_fixtures():
    fixture = Fixture(constant=True)
    try:
        protocol_seal()
        before = len(DECODED)
        blocked("zero_training_span_stops_without_fallback", lambda: d.load_training("household_power2010", "f1"), no_decode=False)
        record = d.read_json(d.ROOT / "output/household_power2010/f1/normalization_failure41.json")
        check("zero_span_failure_retained_without_future_access", record["status"] == "STOP_RETAIN_FAILURE" and not (d.ROOT / "output/household_power2010/f1/normalization41.json").exists() and max(stamp for stamp, token in DECODED[before:]) < "2010-01-03T00:00:00")
    finally:
        fixture.close()
    for kind, task in (("invalid", "household_power2010"), ("disagree", "jena_temperature2017")):
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


def lexical_fixtures():
    check("signed_celsius_exact_decimal_allowed", ORIGINAL_PARSE("jena_temperature2017", b"-7.25", "artificial") == F(-29, 4))
    check("absolute_zero_celsius_boundary_allowed", ORIGINAL_PARSE("jena_temperature2017", b"-273.15", "artificial") == F(-5463, 20))
    blocked("below_absolute_zero_celsius_stops_without_row_drop", lambda: ORIGINAL_PARSE("jena_temperature2017", b"-273.15001", "artificial"), no_decode=False)
    check("nonnegative_kilowatt_exact_decimal_allowed", ORIGINAL_PARSE("household_power2010", b"0.125", "artificial") == F(1, 8))
    blocked("negative_household_measurement_is_invalid", lambda: ORIGINAL_PARSE("household_power2010", b"-1", "artificial"), no_decode=False)
    for token in (b"?", b"", b"nan", b"inf", b"1,2", b"3x", b"1e9999"):
        blocked("invalid_decimal_" + repr(token), lambda t=token: ORIGINAL_PARSE("jena_temperature2017", t, "artificial"), no_decode=False)
    raw = artificial_csv("household_power2010")["household"]
    records = raw.splitlines(keepends=True)
    for i in range(1, len(records)):
        cells = records[i].rstrip(b"\n").split(b";")
        if cells[0] == b"01/01/2010" and cells[1] in (b"10:00:00", b"11:00:00", b"12:00:00"):
            cells[2] = {b"10:00:00": b"?", b"11:00:00": b"", b"12:00:00": b"INVALID_NONMARKER"}[cells[1]]
            records[i] = b";".join(cells) + b"\n"
    altered = b"".join(records)
    projection = p.calendar_projection("household_power2010", {"household": altered})
    rows = {r["timestamp"]: r for r in projection["rows"]}
    check("declared_status_whitelist_exact", projection["hourly_target_status_counts"]["declared_missing"] == 2 and not rows["2010-01-01T12:00:00"]["declared_target_missing"])
    ref = rows["2010-01-01T12:00:00"]["records"][0]
    check("row_offsets_preserve_original_member_bytes", b"INVALID_NONMARKER" in altered[ref["byte_start"]:ref["byte_stop"]])
    original_calendar = d._calendar
    try:
        d._calendar = lambda task: ({}, projection, rows)
        queries, excluded = d._derive_queries("household_power2010", "2010-01-02", "2010-01-03")
        check("declared_missing_target_or_lag_excludes_exact_query", not queries and excluded[0]["declared_missing_required_timestamps"] == ["2010-01-01T10:00:00", "2010-01-01T11:00:00"])
    finally:
        d._calendar = original_calendar


def main():
    try:
        main_fixture()
        failure_fixtures()
        lexical_fixtures()
    finally:
        d.ROOT, p.ROOT, d.task_stages, d.test_dates, d.EXPECTED_N = REAL_CONFIG
        d._parse_numeric_token = ORIGINAL_PARSE
    result = {"schema": "step41-data-access-artificial-preflight-v1", "status": "PASS", "checks": CHECKS,
              "artificial_only": True, "scientific_data_accessed": False, "public_measurement_values_decoded_or_parsed": 0, "statistical_model_fits": 0,
              "synthetic_measurement_token_decodes": len(DECODED), "test_fixture_tasks": list(SMALL_N),
              "source_sha256": {f"source/{name}": hashlib.sha256(raw).hexdigest() for name, raw in SOURCE_BYTES.items()},
              "production_expected_n": d.EXPECTED_N, "production_data_configuration_restored": d.test_dates("household_power2010") == ("2010-09-01", "2010-12-01") and d.ROOT == REAL_ROOT}
    d.write_json(REAL_ROOT / "analysis/data_artificial_checks41.json", result)
    print(json.dumps({"status": "PASS", "checks": len(CHECKS), "public_measurement_values_decoded_or_parsed": 0,
                      "statistical_model_fits": 0, "synthetic_measurement_token_decodes": len(DECODED)}, sort_keys=True))


if __name__ == "__main__":
    main()
