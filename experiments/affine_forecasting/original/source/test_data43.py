"""Manufactured checks of Step 43 calendar parsing and numeric-access gates.

No natural source values are decoded. Synthetic gates and models exist only in
the explicitly artificial temporary fixture; they are not scientific PASS gates.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from fractions import Fraction as F
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np

import common43 as common
import prepare43 as prep
import data43 as data

ROOT = common.ROOT
CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def raises(call, types=(ValueError, FileNotFoundError), message="Expected rejection"):
    global CHECKS
    CHECKS += 1
    try:
        call()
    except types:
        return
    raise AssertionError(message)


def pure_checks():
    check(prep.byte_fields(b'"one","two,three","four""five"\r\n') == [b"one", b"two,three", b'four"five'], "Quoted byte splitter")
    check(prep.byte_fields(b"1/1/2008;00:00:00; ? \r\n", ";")[2].strip(b" \t") == b"?", "Exact missing-marker token")
    raises(lambda: prep.byte_fields(b'"unfinished,1'), message="Unterminated CSV field")
    raises(lambda: prep.byte_fields(b'"closed"x,1'), message="Trailing quoted CSV garbage")
    check(prep.parse_timestamp(prep.SOURCE_SPECS["household_power2008"], [b"1/1/2008", b"00:00:00"]) == datetime(2008, 1, 1), "Unpadded date")
    check(prep.parse_timestamp(prep.SOURCE_SPECS["jena_temperature2018"], [b"01.01.2018 00:10:00"]) == datetime(2018, 1, 1, 0, 10), "Jena timestamp")
    raises(lambda: prep.parse_timestamp(prep.SOURCE_SPECS["jena_temperature2018"], [b"01.01.2018 00:11:00"]), message="Wrong ten-minute cadence")
    check(prep.next_month("2008-02") == "2008-03-01" and prep.next_month("2018-12") == "2019-01-01", "Calendar transitions")
    origin, stamps = prep.required_times(datetime(2018, 1, 2, 23))
    check(origin == datetime(2018, 1, 1, 23) and stamps == [datetime(2018, 1, 1, h) for h in range(24)], "Exact 24-hour information set")
    for task, accepted, rejected in [
        ("household_power2008", [b"0", b"0.0001", b"+1e3"], [b"-1", b"NaN", b"inf", b"?", b"1_0"]),
        ("jena_temperature2018", [b"-273.15", b"-20.25", b"0"], [b"-273.16", b"nan", b"1,2"]),
        ("metro_traffic2018", [b"0", b"5", b"5.0"], [b"-1", b"0.5", b"infinity"]),
    ]:
        for token in accepted:
            check(isinstance(data._parse_numeric(prep.SOURCE_SPECS[task], token, "artificial"), F), "Accepted exact decimal")
        for token in rejected:
            raises(lambda token=token, task=task: data._parse_numeric(prep.SOURCE_SPECS[task], token, "artificial"), message="Rejected physical-domain token")
    # Valid numeric cells are deliberately nonsense; metadata projection still succeeds.
    spec = dict(prep.SOURCE_SPECS["household_power2008"])
    source = {"source_spec": spec, "archives": [{"source_id": "madeup"}]}
    raw = b"Date;Time;Global_active_power\n"
    for day in range(1, 4):
        for hour in range(24):
            token = b"?" if (day, hour) == (1, 5) else b"THIS_IS_NOT_A_NUMBER"
            raw += f"{day}/1/2008;{hour:02d}:00:00;".encode() + token + b"\n"
    projection = prep.calendar_projection("household_power2008", source,
        {"development_start": "2008-01-01", "assessment_stop_exclusive": "2008-01-04"}, {"madeup": raw})
    check(projection["target_values_decoded_or_parsed"] == 0 and len(projection["rows"]) == 72, "Metadata never interprets numeric cells")
    index = {x["timestamp"]: x for x in projection["rows"]}
    queries, excluded = prep.derive_queries("household_power2008", index, "2008-01-02", "2008-01-04")
    check(len(queries) == 1 and queries[0]["target_timestamp"] == "2008-01-03T23:00:00", "Only declared missing status excludes")
    check(excluded[0]["declared_missing_timestamps"] == ["2008-01-01T05:00:00"], "Missing-status exclusion is exact")
    no_markers = dict(spec)
    no_markers["missing_markers"] = []
    no_marker_projection = prep.calendar_projection("household_power2008",
        {"source_spec": no_markers, "archives": [{"source_id": "madeup"}]},
        {"development_start": "2008-01-01", "assessment_stop_exclusive": "2008-01-04"}, {"madeup": raw})
    check(not any(x["declared_target_missing"] for x in no_marker_projection["rows"]), "No undeclared marker screening")


def query(task, day):
    target = datetime(2001, 1, day, 23)
    return {"id": task + ":" + target.isoformat(timespec="seconds"), "task": task,
            "target_timestamp": target.isoformat(timespec="seconds"),
            "origin_timestamp": (target-timedelta(hours=24)).isoformat(timespec="seconds")}


def expected_value(task_index, stamp):
    return F((task_index*100 + stamp.day*7)*10 + stamp.hour, 10)


def create_fixture(root):
    for name in ("common43.py", "prepare43.py", "data43.py"):
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "source" / name, destination)
    design = {"schema": "step43-artificial-data-test-design-v1", "artificial_fixture": True,
              "tasks": {task: {} for task in common.TASKS}}
    common.write_json(root / "design43.json", design)
    common.write_json(root / "design_seal43.json", {"schema": "step43-preaccess-design-seal-v1",
                      "status": "SEALED_BEFORE_NEW_MEASUREMENT_DECODING",
                      "design_sha256": common.sha_file(root / "design43.json"),
                      "measurement_values_decoded_or_parsed": 0, "artificial_fixture": True})
    common.write_json(root / "inputs/exposure43/exposure43.json", {"status": "PASS", "artificial_fixture": True})
    tasks = {}
    for task_index, task in enumerate(common.TASKS):
        directory = root / "inputs/data43" / task
        directory.mkdir(parents=True, exist_ok=True)
        spec = dict(prep.SOURCE_SPECS["metro_traffic2018"])
        spec.update({"numeric_domain": "finite_nonnegative_decimal", "artificial_fixture": True})
        for key in ("archives", "prior_source_path", "prior_metadata_path"):
            spec.pop(key, None)
        raw = b"holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,weather_description,date_time,traffic_volume\n"
        for day in range(1, 14):
            for hour in range(24):
                value = expected_value(task_index, datetime(2001, 1, day, hour))
                numerator = value.numerator * (10//value.denominator)
                text = f"{numerator//10}.{numerator%10}"
                raw += f"opaque,opaque,opaque,opaque,opaque,opaque,opaque,2001-01-{day:02d} {hour:02d}:00:00,{text}\n".encode()
        archive = directory / "manufactured.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
            output.writestr("manufactured.csv", raw)
        arc = {**common.file_ref(archive, root=root), "source_id": "manufactured", "member": "manufactured.csv",
               "nested_gzip": False, "member_sha256": hashlib.sha256(raw).hexdigest(),
               "payload_sha256": hashlib.sha256(raw).hexdigest(), "payload_bytes": len(raw)}
        source = {"schema": "step43-opaque-source-provenance-v1", "task": task,
                  "source_spec": spec, "archives": [arc], "artificial_fixture": True}
        common.write_json(directory / "source43.json", source)
        projection = prep.calendar_projection(task, source,
            {"development_start": "2001-01-03", "assessment_stop_exclusive": "2001-01-14"}, {"manufactured": raw})
        common.write_json(directory / "calendar43.json", projection)
        stages = {}
        for stage, end, val in [("f1", 6, [6, 7]), ("f2", 8, [8, 9]), ("f3", 10, [10, 11]), ("final", 12, [])]:
            stages[stage] = {"train": [query(task, d) for d in range(3, end)],
                             "validation": [query(task, d) for d in val],
                             "train_start": "2001-01-03", "train_stop_exclusive": f"2001-01-{end:02d}",
                             "validation_start": f"2001-01-{end:02d}",
                             "validation_stop_exclusive": f"2001-01-{end+2:02d}" if val else None}
        tasks[task] = {"spec": spec, "archives": [arc], "source": common.file_ref(directory / "source43.json", root=root),
                       "calendar": common.file_ref(directory / "calendar43.json", root=root),
                       "stages": stages, "assessment": [query(task, 12), query(task, 13)]}
    manifest = {"schema": "step43-data-manifest-v1", "artificial_fixture": True, "task_order": list(common.TASKS),
                "tasks": tasks, "measurement_values_decoded_or_parsed": 0,
                "design_seal_sha256": common.sha_file(root / "design_seal43.json")}
    common.write_json(root / data.MANIFEST, manifest)
    common.write_json(root / "protocol43.json", {"schema": "step43-artificial-gate-protocol-v1", "artificial_fixture": True})


def seal_fixture(root):
    files = [p for p in root.rglob("*") if p.is_file()]
    common.write_json(root / "protocol_seal43.json",
        {"schema": "step43-protocol-source-seal-v1", "artificial_fixture": True,
         "bound_files": {p.relative_to(root).as_posix(): common.sha_file(p) for p in files}})
    return common.protocol_identity(root=root, deep=True)


def fit_seal(root, task, stage, identity, seeds=None):
    files = []
    for seed in common.SEEDS:
        path = root / f"output/{task}/{stage}/seed{seed}/manufactured_fit.json"
        common.write_json(path, {"artificial_fixture": True, "task": task, "stage": stage, "seed": seed})
        files.append(path)
    path = root / f"output/{task}/{stage}/fit_seal43.json"
    common.write_json(path, {"schema": "step43-stage-fit-seal-v1", "task": task, "stage": stage,
                             "protocol_seal_sha256": identity, "seeds": list(common.SEEDS) if seeds is None else seeds,
                             "bound_files": {p.relative_to(root).as_posix(): common.sha_file(p) for p in files}})
    return path


def prediction_record(root, task, index, identity, history, roster):
    q = query(task, 12+index)
    predictions = {key: {"candidate_id": value["candidate_id"], "reported_physical_hex": float(index+1).hex()}
                   for key, value in roster["tasks"][task]["selections"].items()}
    return {"schema": "step43-query-predictions-v1", "task": task, "query_id": q["id"], "query_index": index,
            "protocol_seal_sha256": identity, "history_receipt_sha256": history["history_receipt_sha256"],
            "predictions": predictions}


def access_checks(fixture):
    original_parse = data._parse_numeric
    counts = {"numeric_decodes": 0}
    def counted(spec, token, stamp):
        counts["numeric_decodes"] += 1
        return original_parse(spec, token, stamp)
    data._parse_numeric = counted
    try:
        create_fixture(fixture)
        task = common.TASKS[0]
        raises(lambda: data.load_training(task, "f1", root=fixture), message="Missing protocol seal")
        check(counts["numeric_decodes"] == 0, "No numeric conversion before protocol seal")
        identity = seal_fixture(fixture)
        raises(lambda: data.load_training(task, "f1", root=fixture), message="Missing preflight")
        check(counts["numeric_decodes"] == 0, "No numeric conversion before independent preflight")
        common.write_json(fixture / "audit/preflight43.json", {"status": "PASS", "protocol_seal_sha256": "wrong", "artificial_fixture": True})
        raises(lambda: data.load_training(task, "f1", root=fixture), message="Wrong gate ancestry")
        check(counts["numeric_decodes"] == 0, "Wrong gate cannot decode measurements")
        common.write_json(fixture / "audit/preflight43.json", {"status": "PASS", "protocol_seal_sha256": identity, "artificial_fixture": True}, overwrite=True)
        for task_index, task in enumerate(common.TASKS):
            for stage, end in [("f1", 6), ("f2", 8), ("f3", 10), ("final", 12)]:
                before = counts["numeric_decodes"]
                cohort = data.load_training(task, stage, root=fixture)
                check(counts["numeric_decodes"] > before, "Authorized training reads source values")
                check(cohort["X"].shape == (end-3, 24) and not cohort["X"].flags.writeable, "Read-only 24-input matrix")
                expected = [expected_value(task_index, datetime(2001, 1, day, 23)) for day in range(3, end)]
                check(cohort["raw_targets"] == expected, "Exact physical targets")
                check(cohort["normalization"]["target_min"] == expected[0] and cohort["normalization"]["target_max"] == expected[-1], "Training-target-only normalization")
                check(cohort["y"] == [F(i, len(expected)-1) for i in range(len(expected))], "Exact normalized targets")
                for row, day in enumerate(range(3, end)):
                    for hour in range(24):
                        check(cohort["X"][row,hour] == float(expected_value(task_index, datetime(2001,1,day-1,hour))), "Causal exact-to-binary64 feature conversion")
                saved = counts["numeric_decodes"]
                cached = data.load_training(task, stage, root=fixture)
                check(counts["numeric_decodes"] == saved and cached["sha256"] == cohort["sha256"], "Seed reuse reads a bound cached cohort")
                if stage == "final":
                    raises(lambda task=task: data.load_validation(task, "final", root=fixture), message="No final validation labels")
                    fit_seal(fixture, task, stage, identity)
                    continue
                before = counts["numeric_decodes"]
                raises(lambda task=task,stage=stage: data.load_validation(task, stage, root=fixture), message="Validation requires fit seal")
                check(counts["numeric_decodes"] == before, "No validation conversion before all-seed fit seal")
                path = fit_seal(fixture, task, stage, identity, seeds=[3901])
                raises(lambda task=task,stage=stage: data.load_validation(task, stage, root=fixture), message="Incomplete seed fit panel")
                check(counts["numeric_decodes"] == before, "Incomplete all-seed seal cannot decode validation")
                record = common.read_json(path)
                record["seeds"] = list(common.SEEDS)
                common.write_json(path, record, overwrite=True)
                fitted = next(iter(record["bound_files"]))
                original = (fixture/fitted).read_bytes()
                (fixture/fitted).write_bytes(original+b" ")
                raises(lambda task=task,stage=stage: data.load_validation(task, stage, root=fixture), message="Changed fitted bytes")
                check(counts["numeric_decodes"] == before, "Changed fit-bank bytes reject before numeric access")
                (fixture/fitted).write_bytes(original)
                validation = data.load_validation(task, stage, root=fixture)
                check(all(y>1 for y in validation["y"]), "Future targets retain training normalization and are not clipped")
                check(validation["normalization"] == cohort["normalization"], "Validation never refits target normalization")
                saved = counts["numeric_decodes"]
                data.load_validation(task, stage, root=fixture)
                check(counts["numeric_decodes"] == saved, "Validation read once")
        task = common.TASKS[0]
        before = counts["numeric_decodes"]
        raises(lambda: data.load_assessment_history(task, 0, root=fixture), message="No global selection seal")
        check(counts["numeric_decodes"] == before, "No assessment history before selection")
        roster = {"schema": "step43-selected-policy-roster-v1", "protocol_seal_sha256": identity,
                  "tasks": {task: {"selections": {f"{seed}::{policy}": {"candidate_id": f"manufactured-{seed}-{policy}"}
                                                 for seed in common.SEEDS for policy in common.POLICIES}}
                            for task in common.TASKS}}
        common.write_json(fixture/"output/selected_roster43.json", roster)
        model_files = [fixture/"output/selected_roster43.json"]+list((fixture/"output").glob("*/final/fit_seal43.json"))
        common.write_json(fixture/"output/selection_seal43.json", {"schema":"step43-selection-seal-v1","protocol_seal_sha256":identity,
                         "bound_files":{p.relative_to(fixture).as_posix():common.sha_file(p) for p in model_files}})
        raises(lambda: data.load_assessment_history(task, 0, root=fixture), message="No preassessment audit")
        check(counts["numeric_decodes"] == before, "No history before independent preassessment PASS")
        common.write_json(fixture/"audit/preassessment43.json", {"status":"PASS","protocol_seal_sha256":identity,
                         "selection_seal_sha256":common.sha_file(fixture/"output/selection_seal43.json"),"artificial_fixture":True})
        for task_index, task in enumerate(common.TASKS):
            before = counts["numeric_decodes"]
            raises(lambda task=task: data.load_assessment_history(task, 1, root=fixture), message="No skipped first query")
            check(counts["numeric_decodes"] == before, "Query order checked before source conversion")
            for index in range(2):
                before = counts["numeric_decodes"]
                history = data.load_assessment_history(task, index, root=fixture)
                check(counts["numeric_decodes"]-before == 24, "Exactly 24 causal raw observations")
                check(history["X"].shape == (1,24) and not history["X"].flags.writeable, "Immutable assessment row")
                check(history["raw_persistence"] == expected_value(task_index,datetime(2001,1,11+index,23)), "Own-target causal persistence")
                check(history["current_target_decoded"] is False, "Current target remains sealed")
                before = counts["numeric_decodes"]
                data.load_assessment_history(task, index, root=fixture)
                check(counts["numeric_decodes"] == before, "Repeated current history is cached")
                prediction_path = fixture/f"output/assessment/{task}/queries/{index:04d}/predictions43.json"
                if index == 0:
                    raises(lambda task=task,index=index,prediction_path=prediction_path: data.reveal_assessment_target(task,index,prediction_path,root=fixture), message="Target before predictions")
                    check(counts["numeric_decodes"] == before, "Missing predictions cannot reveal a target")
                    incomplete = prediction_record(fixture,task,index,identity,history,roster)
                    incomplete["predictions"].pop("4302::ridge")
                    common.write_json(prediction_path,incomplete)
                    raises(lambda task=task,index=index,prediction_path=prediction_path: data.reveal_assessment_target(task,index,prediction_path,root=fixture), message="Missing one selected policy/seed")
                    check(counts["numeric_decodes"] == before, "17 predictions do not authorize target decode")
                    wrong = prediction_record(fixture,task,index,identity,history,roster)
                    wrong["predictions"]["3901::raw"]["candidate_id"] = "different-model"
                    common.write_json(prediction_path,wrong,overwrite=True)
                    raises(lambda task=task,index=index,prediction_path=prediction_path: data.reveal_assessment_target(task,index,prediction_path,root=fixture), message="Wrong selected candidate")
                    check(counts["numeric_decodes"] == before, "Candidate substitution rejects before target decode")
                    wrong = prediction_record(fixture,task,index,identity,history,roster)
                    wrong["history_receipt_sha256"] = "wrong"
                    common.write_json(prediction_path,wrong,overwrite=True)
                    raises(lambda task=task,index=index,prediction_path=prediction_path: data.reveal_assessment_target(task,index,prediction_path,root=fixture), message="Wrong history identity")
                    wrong = prediction_record(fixture,task,index,identity,history,roster)
                    wrong["predictions"]["3901::raw"]["reported_physical_hex"] = "nan"
                    common.write_json(prediction_path,wrong,overwrite=True)
                    raises(lambda task=task,index=index,prediction_path=prediction_path: data.reveal_assessment_target(task,index,prediction_path,root=fixture), message="Nonfinite forecast")
                    check(counts["numeric_decodes"] == before, "Malformed forecasts never open a target")
                value = prediction_record(fixture,task,index,identity,history,roster)
                common.write_json(prediction_path,value,overwrite=prediction_path.exists())
                target = data.reveal_assessment_target(task,index,prediction_path,root=fixture)
                check(counts["numeric_decodes"]-before == 1, "Exactly one current target decode")
                check(target["raw_target"] == expected_value(task_index,datetime(2001,1,12+index,23)), "Exact authorized assessment target")
                check(target["y"] > 1 and target["target_value_clipped"] is False, "Out-of-range labels remain unaltered")
                before = counts["numeric_decodes"]
                raises(lambda task=task,index=index,prediction_path=prediction_path: data.reveal_assessment_target(task,index,prediction_path,root=fixture), message="Target cannot be decoded twice")
                check(counts["numeric_decodes"] == before, "Repeated target reveal denied")
            state = common.read_json(data._state_path(task,root=fixture))
            check(state["status"]=="COMPLETE" and state["next_index"]==2, "All chronological targets complete")
            before = counts["numeric_decodes"]
            data.load_assessment_history(task,0,root=fixture)
            check(counts["numeric_decodes"]==before, "Cost assay may reuse retained history without new decodes")
        lines=(fixture/data.DATA_LOG).read_bytes().splitlines(keepends=True)
        previous=None
        for index,line in enumerate(lines):
            value=json.loads(line)
            check(value["index"]==index and value["previous_event_sha256"]==previous,"Data-event hash ancestry")
            previous=hashlib.sha256(line).hexdigest()
        events=[json.loads(line) for line in lines]
        for task in common.TASKS:
            for index in range(2):
                kinds=[x["kind"] for x in events if x.get("task")==task and x.get("query_index")==index]
                check(kinds==["assessment_history_started","assessment_history_completed","assessment_predictions_sealed",
                              "assessment_target_started","assessment_target_revealed"],"Commit-before-target event sequence")
        return {"synthetic_numeric_token_decodes":counts["numeric_decodes"],"synthetic_tasks":3,
                "synthetic_training_cohorts":12,"synthetic_validation_cohorts":9,
                "synthetic_assessment_queries":6,"data_events_checked":len(events),
                "last_event_sha256":previous}
    finally:
        data._parse_numeric=original_parse


def main():
    started=time.perf_counter()
    receipt=ROOT/"analysis/data_artificial_checks43.json"
    if receipt.exists():
        raise FileExistsError("Retain the existing completed artificial check receipt")
    actual_manifest=ROOT/data.MANIFEST
    manifest_hash=common.sha_file(actual_manifest) if actual_manifest.exists() else None
    if (ROOT/"protocol_seal43.json").exists() or (ROOT/"output/data43").exists():
        raise ValueError("Run artificial data checks before real scientific numeric access")
    pure_checks()
    parent=ROOT/"tmp"
    parent.mkdir(parents=True,exist_ok=True)
    fixture=Path(tempfile.mkdtemp(prefix="data43_manufactured_",dir=parent))
    result=access_checks(fixture)
    check(not (ROOT/"output/data43").exists(),"No natural numeric artifacts created")
    if manifest_hash is not None:
        check(common.sha_file(actual_manifest)==manifest_hash,"Real calendar manifest unchanged")
    sources={f"source/{name}":common.sha_file(ROOT/"source"/name) for name in
             ["common43.py","prepare43.py","data43.py","test_data43.py"]}
    record={"schema":"step43-data-artificial-checks-v1","status":"PASS","utc":common.utc_now(),
            "checks":CHECKS,"wall_seconds":time.perf_counter()-started,"source_bindings":sources,
            "real_data_manifest_sha256_unchanged":manifest_hash,
            "natural_measurement_values_decoded":0,"real_model_fits":0,
            "fixture_root":fixture.relative_to(ROOT).as_posix(),"fixture_is_artificial":True,
            "scope":"Manufactured calendar/parser and exact-target/access-state checks; fake gates exist only inside the synthetic fixture and are not scientific audit gates",
            **result}
    common.write_json(receipt,record)
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
