"""Artificial-only tests of the phased count loader and chronological gate.

All measurement bytes in these tests are constructed here. The real archive is
never opened, decoded, or copied. Fake seals and gates exist only below a fresh
temporary root and cannot authorize access to the scientific source.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timedelta
from fractions import Fraction as F
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch
import zipfile

import numpy as np

import data38 as d
import prepare_data38 as prep

REAL_ROOT=Path(__file__).resolve().parents[1]
CHECKS={}
DECODED=[]
ORIGINAL_PARSE=d._parse_count_token

def observe(token,stamp):
    if d.ROOT==REAL_ROOT:raise AssertionError("Artificial test attempted scientific count access")
    DECODED.append(stamp)
    return ORIGINAL_PARSE(token,stamp)

def check(name,condition):
    if not condition:raise AssertionError(name)
    CHECKS[name]=True

def blocked(name,operation,*,no_decode=True):
    before=len(DECODED)
    try:operation()
    except (RuntimeError,FileNotFoundError):pass
    else:raise AssertionError(name+": operation unexpectedly succeeded")
    if no_decode and len(DECODED)!=before:raise AssertionError(name+": count interpreted before rejection")
    CHECKS[name]=True

def artificial_value(stamp,constant=False):
    day=(stamp-d.SOURCE_FIRST).days
    if stamp<=d.TRAIN_CUTOFF:return 7 if constant else (7*day+stamp.hour)%200
    if stamp<=d.VALIDATION_CUTOFF:return 1000+day
    return 10000+day

def make_csv(*,constant=False,bad_target=False,opaque_counts=False):
    fields=[b"Date",b"Rented Bike Count",b"Hour"]+[b"Opaque field "+str(i).encode() for i in range(3,13)]+[b"Functioning Day"]
    lines=[b",".join(fields)+b"\r\n"]
    for hour in range(8760):
        stamp=d.SOURCE_FIRST+timedelta(hours=hour)
        token=str(artificial_value(stamp,constant)).encode()
        if opaque_counts:token=b"\xffUNPARSEABLE_COUNT"
        if bad_target and stamp==datetime(2018,9,1,23):token=b"INVALID"
        # Weather/other data cannot be decoded as UTF-8 or parsed numerically.
        cells=[stamp.strftime("%d/%m/%Y").encode(),token,str(stamp.hour).encode()]+[b"\xfeNONNUMERIC"]*10+[b"No" if hour%53==0 else b"Yes"]
        lines.append(b",".join(cells)+b"\r\n")
    return b"".join(lines)

@contextmanager
def fixture(*,constant=False,bad_target=False):
    temporary_parent=REAL_ROOT/"tmp";temporary_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="artificial_data38_",dir=temporary_parent) as directory:
        root=Path(directory);(root/"inputs").mkdir();(root/"source").mkdir();(root/"output").mkdir();(root/"audit").mkdir()
        for name in ("data38.py","prepare_data38.py"):
            shutil.copyfile(REAL_ROOT/"source"/name,root/"source"/name)
        raw=make_csv(constant=constant,bad_target=bad_target)
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as zipped:zipped.writestr(d.MEMBER,raw)
        (root/d.ARCHIVE_RELATIVE).write_bytes(buffer.getvalue())
        archive_hash=d.sha_file(root/d.ARCHIVE_RELATIVE)
        d.write_json(root/"inputs/public_source38.json",{"archive_path":d.ARCHIVE_RELATIVE,"archive_sha256":archive_hash,"artificial_only":True})
        d.write_json(root/"inputs/public_metadata38.json",{"artificial_only":True})
        d.write_json(root/"inputs/prior_exposure38.json",{"artificial_only":True})
        projection=prep.calendar_projection(raw);projection.update({"archive_sha256":archive_hash,"archive_member":d.MEMBER})
        d.write_json(root/"inputs/calendar_projection38.json",projection)
        d.write_json(root/"inputs/calendar_metadata38.json",{k:v for k,v in projection.items() if k!="rows"})
        protocol=d.read_json(REAL_ROOT/"protocol38.json");protocol["data"]["archive_sha256"]=archive_hash
        d.write_json(root/"protocol38.json",protocol)
        with patch.object(d,"ROOT",root),patch.object(d,"_parse_count_token",observe):
            d.metadata_rosters()
            yield root

def protocol_seal(root,gate=True):
    record={"artificial_only":True,"bound_files":{name:d.sha_file(root/name) for name in sorted(d.DATA_REQUIRED)}}
    d.write_json(root/"protocol_seal38.json",record)
    if gate:d.write_json(root/"audit/preexecution_gate38.json",{"status":"PASS","artificial_only":True,"protocol_seal_sha256":d.sha_file(root/"protocol_seal38.json")})

def fit_seal(root,gate=True):
    d.write_json(root/"output/fake_fitted_model.json",{"constructed_train_only_model":True})
    files=["output/normalization38.json","output/cohorts/train.json","output/cohorts/train_X.npy","output/fake_fitted_model.json"]
    record={"artificial_only":True,"protocol_seal_sha256":d.sha_file(root/"protocol_seal38.json"),"bound_files":{p:d.sha_file(root/p) for p in files}}
    d.write_json(root/"output/fit_seal38.json",record)
    if gate:d.write_json(root/"audit/prevalidation_gate38.json",{"status":"PASS","artificial_only":True,"fit_seal_sha256":d.sha_file(root/"output/fit_seal38.json"),"protocol_seal_sha256":d.sha_file(root/"protocol_seal38.json")})

def model_seal(root,gate=True):
    files=list(d.read_json(root/"output/fit_seal38.json")["bound_files"])+["output/cohorts/validation.json","output/cohorts/validation_X.npy"]
    record={"artificial_only":True,"protocol_seal_sha256":d.sha_file(root/"protocol_seal38.json"),"fit_seal_sha256":d.sha_file(root/"output/fit_seal38.json"),"bound_files":{p:d.sha_file(root/p) for p in files}}
    d.write_json(root/"output/model_seal38.json",record)
    if gate:d.write_json(root/"audit/pretest_gate38.json",{"status":"PASS","artificial_only":True,"model_seal_sha256":d.sha_file(root/"output/model_seal38.json"),"fit_seal_sha256":d.sha_file(root/"output/fit_seal38.json"),"protocol_seal_sha256":d.sha_file(root/"protocol_seal38.json")})

def prediction(root,query,history,change=None):
    ids=d.read_json(root/"protocol38.json")["evaluation"]["prediction_ids"]
    record={"schema":"step38-sealed-test-predictions-v1","query_id":query["id"],
        "model_seal_sha256":d.sha_file(root/"output/model_seal38.json"),"history_receipt_sha256":history["history_receipt_sha256"],
        "predictions":{cid:{"predictions":[["1/4","1/4"]],"artificial_only":True} for cid in ids}}
    if change:change(record)
    path=f"output/artificial_predictions/{history['query_index']:03d}.json";d.write_json(root/path,record)
    return path,d.sha_file(root/path)

def audit_chain(root):
    previous=None;events=[];last_ns=-1
    for line in (root/"output/access_and_execution38.jsonl").read_text().splitlines():
        record=json.loads(line);saved=record.pop("event_sha256")
        assert d.digest(record)==saved and record["previous_event_sha256"]==previous
        assert record["monotonic_ns"]>last_ns
        previous=saved;last_ns=record["monotonic_ns"];events.append(record)
    for query in d.test_queries():
        own=[item["event"] for item in events if item.get("query_id")==query["id"]]
        assert own==["test_history_access_started","test_history_access_completed","test_predictions_sealed","test_target_access_started","test_target_revealed"]
    return {"event_count":len(events),"final_event_sha256":previous,"test_reveal_events":91}

def run_main_fixture():
    with fixture() as root:
        queries=d.test_queries();q0,q1=queries[:2]
        check("metadata_roster_242_31_91",{k:v["timestamp_complete_targets"] for k,v in d.read_json(root/"inputs/calendar_roster38.json")["partitions"].items()}==d.EXPECTED_N)
        blocked("training_requires_protocol_seal",d.load_training)
        blocked("private_count_entry_requires_protocol_seal",lambda:d._read_approved_counts([d.SOURCE_FIRST],"train"))
        protocol_seal(root,gate=False)
        blocked("training_requires_independent_preexecution_pass",d.load_training)
        d.write_json(root/"audit/preexecution_gate38.json",{"status":"PASS","protocol_seal_sha256":"wrong"})
        blocked("training_gate_is_bound_to_current_seal",d.load_training)
        protocol_seal(root)
        archive=root/d.ARCHIVE_RELATIVE;original=archive.read_bytes();archive.write_bytes(original+b"changed")
        blocked("changed_sealed_archive_blocks_before_count_decode",d.load_training);archive.write_bytes(original)
        before=len(DECODED);training=d.load_training();train_decodes=DECODED[before:]
        check("training_fixed_roster_and_raw_readonly_matrix",training["n"]==242 and training["X"].shape==(242,24) and training["X"].dtype==np.float64 and not training["X"].flags.writeable)
        check("training_chronological_limit",max(train_decodes)==d._iso(d.TRAIN_CUTOFF) and all(d._timestamp(x)<=d.TRAIN_CUTOFF for x in train_decodes))
        check("training_exact_lags_oldest_first",training["X"][0].tolist()==[artificial_value(d.SOURCE_FIRST+timedelta(hours=h)) for h in range(24)])
        normal=d.read_json(root/"output/normalization38.json");normal_hash=d.sha_file(root/"output/normalization38.json")
        check("normalization_uses_training_targets_only",normal["target_min"]==min(training["raw_targets"]) and normal["target_max"]==max(training["raw_targets"]) and training["y"]==[F(x-normal["target_min"],normal["span"]) for x in training["raw_targets"]])
        blocked("training_cannot_be_rerun",d.load_training)
        blocked("bulk_test_load_is_forbidden",lambda:d._load_cohort("test"))
        blocked("validation_requires_completed_fit_seal",d.load_validation)
        blocked("private_validation_count_entry_requires_fit_seal",lambda:d._read_approved_counts([d.VALIDATION_CUTOFF],"validation"))
        fit_seal(root,gate=False)
        blocked("validation_requires_independent_gate",d.load_validation)
        d.write_json(root/"audit/prevalidation_gate38.json",{"status":"PASS","fit_seal_sha256":"wrong","protocol_seal_sha256":d.sha_file(root/"protocol_seal38.json")})
        blocked("validation_gate_bound_to_fit_seal",d.load_validation)
        fit_seal(root)
        before=len(DECODED);validation=d.load_validation();val_decodes=DECODED[before:]
        check("validation_fixed_roster_and_raw_readonly_matrix",validation["n"]==31 and validation["X"].shape==(31,24) and not validation["X"].flags.writeable)
        check("validation_stops_before_test",all(d._timestamp(x)<=d.VALIDATION_CUTOFF for x in val_decodes))
        check("validation_preserves_training_normalization_and_unclipped_targets",normal_hash==d.sha_file(root/"output/normalization38.json") and all(y>1 for y in validation["y"]))
        blocked("validation_cannot_be_rerun",d.load_validation)
        blocked("test_requires_final_model_seal",lambda:d.load_test_history(q0))
        model_seal(root,gate=False)
        blocked("test_requires_independent_pretest_gate",lambda:d.load_test_history(q0))
        d.write_json(root/"audit/pretest_gate38.json",{"status":"PASS","model_seal_sha256":"wrong","fit_seal_sha256":d.sha_file(root/"output/fit_seal38.json"),"protocol_seal_sha256":d.sha_file(root/"protocol_seal38.json")})
        blocked("test_gate_bound_to_final_model_seal",lambda:d.load_test_history(q0))
        model_seal(root)
        blocked("target_cannot_be_revealed_before_history",lambda:d.reveal_test_target(q0,"output/missing.json","wrong"))
        blocked("test_cannot_skip_first_query",lambda:d.load_test_history(q1))
        before=len(DECODED);h0=d.load_test_history(q0);access0=DECODED[before:]
        check("first_history_exposes_only_own24_before_origin",access0==h0["history_timestamps"] and len(access0)==24 and max(access0)==q0["origin_timestamp"] and q0["target_timestamp"] not in access0)
        check("test_history_exact_readonly_shape",h0["X"].shape==(1,24) and h0["X"].dtype==np.float64 and not h0["X"].flags.writeable)
        blocked("next_history_waits_for_current_target_reveal",lambda:d.load_test_history(q1))
        blocked("same_history_cannot_be_reread",lambda:d.load_test_history(q0))
        blocked("private_history_cannot_expose_future",lambda:d._read_approved_counts([d._timestamp(q0["target_timestamp"])],"test_history"))
        blocked("private_target_read_requires_sealed_prediction",lambda:d._read_approved_counts([d._timestamp(q0["target_timestamp"])],"test_target"))
        blocked("target_requires_retained_prediction_file",lambda:d.reveal_test_target(q0,"output/missing.json","wrong"))
        path,sha=prediction(root,q0,h0)
        blocked("prediction_digest_must_match_retained_bytes",lambda:d.reveal_test_target(q0,path,"wrong"))
        path,sha=prediction(root,q0,h0,lambda r:r["predictions"].pop(next(iter(r["predictions"]))))
        blocked("all93_predictions_required",lambda:d.reveal_test_target(q0,path,sha))
        path,sha=prediction(root,q0,h0,lambda r:r.update({"history_receipt_sha256":"wrong"}))
        blocked("predictions_bind_exact_history_receipt",lambda:d.reveal_test_target(q0,path,sha))
        path,sha=prediction(root,q0,h0,lambda r:r.update({"model_seal_sha256":"wrong"}))
        blocked("predictions_bind_exact_final_model",lambda:d.reveal_test_target(q0,path,sha))
        path,sha=prediction(root,q0,h0);before=len(DECODED);target=d.reveal_test_target(q0,path,sha)
        check("sealed_prediction_reveals_only_current_target",DECODED[before:]==[q0["target_timestamp"]] and target["raw_target"]==artificial_value(d._timestamp(q0["target_timestamp"])) and target["y"]>1)
        blocked("same_target_cannot_be_revealed_again",lambda:d.reveal_test_target(q0,path,sha))
        state_path=root/"output/test_access_state38.json";saved_state=state_path.read_bytes();bad=d.read_json(state_path)
        bad["next_index"]=2;bad["previously_revealed_test_targets"]=[q["target_timestamp"] for q in queries[:2]];d.write_json(state_path,bad)
        blocked("state_cannot_advance_without_reveal_event",lambda:d.load_test_history(queries[2]));state_path.write_bytes(saved_state)
        model_path=root/"output/fake_fitted_model.json";model_bytes=model_path.read_bytes();model_path.write_bytes(model_bytes+b"changed")
        blocked("changed_frozen_model_blocks_next_history",lambda:d.load_test_history(q1));model_path.write_bytes(model_bytes)
        before=len(DECODED);h1=d.load_test_history(q1)
        check("earlier_revealed_test_target_can_be_next_history",h1["previously_revealed_test_targets_in_history"]==[q0["target_timestamp"]] and q0["target_timestamp"] in DECODED[before:] and q1["target_timestamp"] not in DECODED[before:])
        path,sha=prediction(root,q1,h1);d.reveal_test_target(q1,path,sha)
        # Every query uses a distinct retained artifact, immutable once sealed.
        for index,query in enumerate(queries[2:],2):
            history=d.load_test_history(query);path,sha=prediction(root,query,history)
            d.reveal_test_target(query,path,sha)
        state=d._load_test_state()
        check("all91_queries_finish_with_complete_state",state["status"]=="COMPLETE" and state["next_index"]==91 and len(state["previously_revealed_test_targets"])==91)
        blocked("completed_test_cannot_restart",lambda:d.load_test_history(q0))
        chain=audit_chain(root);check("all91_forecasts_sealed_before_targets_and_chain_valid",chain["test_reveal_events"]==91)
        return {"train_count_decodes":len(train_decodes),"validation_count_decodes":len(val_decodes),"test_queries":91,"chain":chain}

def run_failure_fixtures():
    with fixture(constant=True) as root:
        protocol_seal(root);before=len(DECODED)
        blocked("zero_training_span_stops_and_retains_failure",d.load_training,no_decode=False)
        failure=d.read_json(root/"output/target_normalization_failure38.json")
        check("zero_span_has_no_fallback_and_no_future_values",failure["status"]=="STOP_RETAIN_FAILURE" and failure["span"]==0 and not (root/"output/normalization38.json").exists() and all(d._timestamp(x)<=d.TRAIN_CUTOFF for x in DECODED[before:]))
    with fixture(bad_target=True) as root:
        protocol_seal(root);d.load_training();fit_seal(root);d.load_validation();model_seal(root)
        query=d.test_queries()[0];history=d.load_test_history(query);path,sha=prediction(root,query,history);before=len(DECODED)
        blocked("malformed_test_target_stops_without_row_dropping",lambda:d.reveal_test_target(query,path,sha),no_decode=False)
        state=d.read_json(root/"output/test_access_state38.json")
        check("malformed_target_failure_is_after_prediction_seal",state["status"]=="FAILED" and state["next_index"]==0 and (root/"output/test_prediction_seals/000.json").is_file() and DECODED[before:]==[query["target_timestamp"]])
        blocked("failed_test_cannot_advance",lambda:d.load_test_history(d.test_queries()[1]))

def main():
    started=time.perf_counter();projection=prep.calendar_projection(make_csv(opaque_counts=True))
    check("metadata_projection_never_decodes_count_or_weather",projection["timestamp_rows"]==8760 and projection["count_values_decoded_or_parsed"]==0 and projection["weather_values_decoded_or_parsed"]==0)
    details=run_main_fixture();run_failure_fixtures()
    check("artificial_roots_restored",d.ROOT==REAL_ROOT and d._parse_count_token is ORIGINAL_PARSE)
    result={"schema":"step38-artificial-data-gates-v1","status":"PASS","checks":CHECKS,"check_count":len(CHECKS),
        "artificial_only":True,"scientific_count_values_decoded":0,"scientific_weather_values_decoded":0,
        "reserved_patient_arrays_opened":0,"constructed_count_token_attempts":len(DECODED),"details":details,
        "source_sha256":{"source/"+name:d.sha_file(REAL_ROOT/"source"/name) for name in ("data38.py","prepare_data38.py","check_data38.py")},
        "wall_seconds":time.perf_counter()-started}
    d.write_json(REAL_ROOT/"tests/artificial_data38.json",result)
    print(json.dumps({"status":result["status"],"check_count":result["check_count"],"artificial_only":True,"scientific_count_values_decoded":0,"wall_seconds":result["wall_seconds"]},sort_keys=True))

if __name__=="__main__":main()
