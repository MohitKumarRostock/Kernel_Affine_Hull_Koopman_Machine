"""Gate-enforced Seoul forecasting data with chronological test-label reveal.

Opaque CSV bytes are projected only at explicitly allowed timestamps. Training
and validation phases have separate seals; each test query exposes only history
through its forecast origin, then requires all predictions to be sealed before
the current target count is decoded. Earlier revealed test targets may become
history for later forecasts; the access receipts make this overlap explicit.
"""
from __future__ import annotations

import argparse
from datetime import date,datetime,timedelta,timezone
from fractions import Fraction as F
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TARGET="rented_bike_count"
ARCHIVE_RELATIVE="inputs/seoul_bike_uci560.zip"
MEMBER="SeoulBikeData.csv"
SOURCE_FIRST=datetime(2017,12,1)
SOURCE_STOP=datetime(2018,12,1)
TRAIN_CUTOFF=datetime(2018,7,31,23)
VALIDATION_CUTOFF=datetime(2018,8,31,23)
PARTITIONS={"train":(date(2017,12,1),date(2018,8,1)),
            "validation":(date(2018,8,1),date(2018,9,1)),
            "test":(date(2018,9,1),date(2018,12,1))}
EXPECTED_N={"train":242,"validation":31,"test":91}
DATA_REQUIRED={"protocol38.json","source/data38.py","source/prepare_data38.py",ARCHIVE_RELATIVE,
    "inputs/public_source38.json","inputs/public_metadata38.json","inputs/prior_exposure38.json",
    "inputs/calendar_projection38.json","inputs/calendar_metadata38.json","inputs/calendar_roster38.json"}

def canon(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return [canon(x) for x in value.tolist()]
    if isinstance(value,dict):return {str(k):canon(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [canon(v) for v in value]
    return value

def canonical_bytes(value):
    return (json.dumps(canon(value),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()
def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".pending");temporary.write_bytes(canonical_bytes(value));temporary.replace(path)

def event(name,**fields):
    path=ROOT/"output/access_and_execution38.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
    previous=None
    if path.exists():
        lines=path.read_text().splitlines()
        if lines:
            last=json.loads(lines[-1]);previous=last.pop("event_sha256")
            if digest(last)!=previous:raise RuntimeError("Existing access-event tail changed")
    record={"event":name,"utc":datetime.now(timezone.utc).isoformat(),"monotonic_ns":time.monotonic_ns(),
            "previous_event_sha256":previous,**fields}
    record["event_sha256"]=digest(record)
    with path.open("ab") as handle:handle.write(canonical_bytes(record));handle.flush()
    return record

def _relative(relative):
    path=Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:raise RuntimeError("Unsafe relative input path")
    return path

def _verify_bound_files(seal,label):
    bounds=seal.get("bound_files")
    if not isinstance(bounds,dict) or not bounds:raise RuntimeError(label+" has no bound files")
    for relative,expected in bounds.items():
        path=ROOT/_relative(relative)
        if not path.is_file() or sha_file(path)!=expected:raise RuntimeError(label+" changed: "+relative)

def verify_protocol_seal():
    path=ROOT/"protocol_seal38.json"
    if not path.is_file():raise RuntimeError("Step38 protocol seal required before count decoding")
    seal=read_json(path);_verify_bound_files(seal,"Frozen protocol/source/input")
    if not DATA_REQUIRED<=set(seal["bound_files"]):raise RuntimeError("Protocol seal omits a data dependency")
    gate_path=ROOT/"audit/preexecution_gate38.json"
    gate=read_json(gate_path) if gate_path.is_file() else {}
    if gate.get("status")!="PASS" or gate.get("protocol_seal_sha256")!=sha_file(path):
        raise RuntimeError("Matching independent preexecution PASS required")
    protocol=read_json(ROOT/"protocol38.json")
    if protocol.get("step")!=38:raise RuntimeError("Wrong protocol step")
    specification=protocol.get("data",{})
    expected={"archive":ARCHIVE_RELATIVE,"member":MEMBER,"target":TARGET,"horizon_hours":24,"lag_hours":24,
        "origin_hour":23,"train_start":"2017-12-01","train_end":"2018-07-31",
        "validation_start":"2018-08-01","validation_end":"2018-08-31",
        "test_start":"2018-09-01","test_end":"2018-11-30","expected_n":EXPECTED_N,
        "partition_basis":"target_timestamp","targets_clipped":False,"no_online_refit":True,
        "previously_revealed_targets_may_be_later_history":True,"degenerate_range_action":"STOP_RETAIN_FAILURE",
        "source_timezone":"local_naive_Seoul_date_and_hour_no_UTC_reinterpretation",
        "target_normalization":"Exact(raw-training_target_min)/(training_target_max-training_target_min)",
        "operation_closures":"Keepallrecordedhoursincludingclosures;recordedrentalsnotlatentunconstraineddemand"}
    if any(specification.get(key)!=value for key,value in expected.items()):
        raise RuntimeError("Protocol changes a frozen source/chronology/normalization/closure rule")
    if specification.get("archive_sha256")!=read_json(ROOT/"inputs/public_source38.json")["archive_sha256"]:
        raise RuntimeError("Protocol source archive digest differs")
    return protocol

def verify_fit_seal():
    verify_protocol_seal();path=ROOT/"output/fit_seal38.json"
    if not path.is_file():raise RuntimeError("Completed-fit seal required before validation count decoding")
    seal=read_json(path)
    if seal.get("protocol_seal_sha256")!=sha_file(ROOT/"protocol_seal38.json"):
        raise RuntimeError("Completed-fit seal belongs to another protocol")
    _verify_bound_files(seal,"Frozen training fits")
    required={"output/normalization38.json","output/cohorts/train.json","output/cohorts/train_X.npy"}
    if not required<=set(seal["bound_files"]):raise RuntimeError("Fit seal omits training normalization/cohort")
    gate_path=ROOT/"audit/prevalidation_gate38.json";gate=read_json(gate_path) if gate_path.is_file() else {}
    if (gate.get("status")!="PASS" or gate.get("fit_seal_sha256")!=sha_file(path)
        or gate.get("protocol_seal_sha256")!=sha_file(ROOT/"protocol_seal38.json")):
        raise RuntimeError("Matching independent prevalidation PASS required")
    return seal

def verify_model_seal():
    verify_fit_seal();path=ROOT/"output/model_seal38.json"
    if not path.is_file():raise RuntimeError("Final model/selection seal required before test count decoding")
    seal=read_json(path)
    if (seal.get("protocol_seal_sha256")!=sha_file(ROOT/"protocol_seal38.json")
        or seal.get("fit_seal_sha256")!=sha_file(ROOT/"output/fit_seal38.json")):
        raise RuntimeError("Final model/selection seal has wrong ancestry")
    _verify_bound_files(seal,"Frozen final selection and models")
    required={"output/normalization38.json","output/cohorts/train.json","output/cohorts/train_X.npy",
              "output/cohorts/validation.json","output/cohorts/validation_X.npy"}
    if not required<=set(seal["bound_files"]):raise RuntimeError("Model seal omits normalization or development cohort")
    gate_path=ROOT/"audit/pretest_gate38.json";gate=read_json(gate_path) if gate_path.is_file() else {}
    if (gate.get("status")!="PASS" or gate.get("model_seal_sha256")!=sha_file(path)
        or gate.get("fit_seal_sha256")!=sha_file(ROOT/"output/fit_seal38.json")
        or gate.get("protocol_seal_sha256")!=sha_file(ROOT/"protocol_seal38.json")):
        raise RuntimeError("Matching independent pretest PASS required")
    return seal

def _iso(value):return value.isoformat(timespec="seconds")
def _timestamp(value):
    stamp=datetime.fromisoformat(value)
    if stamp.tzinfo is not None or _iso(stamp)!=value:raise RuntimeError("Noncanonical source timestamp")
    return stamp

def _calendar():
    source=read_json(ROOT/"inputs/public_source38.json")
    projection=read_json(ROOT/"inputs/calendar_projection38.json")
    if (source["archive_path"]!=ARCHIVE_RELATIVE or projection["archive_sha256"]!=source["archive_sha256"]
        or projection["archive_member"]!=MEMBER or projection["parsed_data_columns"]!=[0,2,13]
        or projection["count_values_decoded_or_parsed"]!=0 or projection["weather_values_decoded_or_parsed"]!=0):
        raise RuntimeError("Calendar metadata does not have the approved source/scope")
    rows=projection["rows"];stamps=[_timestamp(row["timestamp"]) for row in rows]
    expected=[SOURCE_FIRST+timedelta(hours=i) for i in range(int((SOURCE_STOP-SOURCE_FIRST).total_seconds()//3600))]
    if stamps!=expected or len(rows)!=8760:raise RuntimeError("Full source calendar differs")
    return source,projection,{row["timestamp"]:row for row in rows}

def _required_times(target):
    origin=target-timedelta(hours=24)
    return origin,[origin-timedelta(hours=23-j) for j in range(24)]

def _queries(name):
    _,_,rows=_calendar();start,stop=PARTITIONS[name];out=[]
    while start<stop:
        target=datetime(start.year,start.month,start.day,23);origin,lags=_required_times(target)
        if _iso(target) in rows and all(_iso(stamp) in rows for stamp in lags):
            out.append({"id":name+":"+_iso(target),"target_timestamp":_iso(target),"origin_timestamp":_iso(origin)})
        start+=timedelta(days=1)
    if len(out)!=EXPECTED_N[name]:raise RuntimeError("Frozen calendar cohort changed")
    return out

def metadata_rosters():
    """Approved calendar-only preparation, never count decoding."""
    if (ROOT/"protocol_seal38.json").exists():raise RuntimeError("Cannot rewrite frozen calendar metadata")
    _,projection,rows=_calendar();parts={}
    for name,(start,stop) in PARTITIONS.items():
        queries=_queries(name)
        parts[name]={"calendar_start":start.isoformat(),"calendar_stop_exclusive":stop.isoformat(),
            "candidate_daily_targets":(stop-start).days,"timestamp_complete_targets":len(queries),
            "queries":queries,"query_ids_sha256":digest([q["id"] for q in queries]),
            "targets_flagged_nonfunctional":sum(rows[q["target_timestamp"]]["functioning_day"] in ("No","NoFunc") for q in queries),
            "closure_filter_applied":False,"measurement_missingness_not_examined":True}
    result={"schema":"step38-frozen-calendar-roster-v1","source_first":_iso(SOURCE_FIRST),"source_stop_exclusive":_iso(SOURCE_STOP),
        "archive_sha256":projection["archive_sha256"],"projection_sha256":sha_file(ROOT/"inputs/calendar_projection38.json"),
        "origin_hour":23,"horizon_hours":24,"own_history_hours":24,"frequency":"one_forecast_per_day",
        "target":"recorded_hourly_rental_count_at_23","partitions":parts,
        "count_values_decoded_or_parsed":0,"weather_values_decoded_or_parsed":0,
        "future_operation_flag_used_as_predictor_or_filter":False,
        "later_test_histories_may_include_earlier_test_targets":True}
    write_json(ROOT/"inputs/calendar_roster38.json",result);return result

def test_queries():
    """Return calendar-only query identities; this does not expose any counts."""
    return _queries("test")

def _state_path():return ROOT/"output/test_access_state38.json"
def _load_test_state():
    if not _state_path().is_file():raise RuntimeError("No chronological test-history access is pending")
    state=read_json(_state_path())
    if (state.get("schema")!="step38-chronological-test-access-state-v1"
        or state.get("model_seal_sha256")!=sha_file(ROOT/"output/model_seal38.json")
        or state.get("status") not in ("READY","PENDING_HISTORY","PREDICTIONS_SEALED","COMPLETE")):
        raise RuntimeError("Invalid or failed chronological test state")
    queries=_queries("test");index=state.get("next_index")
    if (type(index) is not int or not 0<=index<=len(queries)
        or state.get("previously_revealed_test_targets")!=[q["target_timestamp"] for q in queries[:index]]
        or (state["status"]=="COMPLETE")!=(index==len(queries))):
        raise RuntimeError("Chronological state is not a complete revealed-target prefix")
    events_path=ROOT/"output/access_and_execution38.jsonl"
    if not events_path.is_file():raise RuntimeError("Chronological state lacks its retained access events")
    previous=None;revealed=[]
    for line in events_path.read_text().splitlines():
        item=json.loads(line);saved=item.pop("event_sha256")
        if digest(item)!=saved or item["previous_event_sha256"]!=previous:
            raise RuntimeError("Count-access event chain changed")
        previous=saved
        if item["event"]=="test_target_revealed":revealed.append(item["query_id"])
    if revealed!=[q["id"] for q in queries[:index]]:
        raise RuntimeError("Chronological state advanced without retained target-reveal events")
    return state

def _verify_query(query,index):
    expected=_queries("test")
    if not 0<=index<len(expected) or query!=expected[index]:raise RuntimeError("Test query is not the next frozen calendar query")

def _prediction_path(relative):
    candidate=Path(relative)
    path=(candidate if candidate.is_absolute() else ROOT/candidate).resolve()
    try:rel=path.relative_to(ROOT.resolve())
    except ValueError:raise RuntimeError("Prediction file lies outside Step38")
    if not rel.parts or rel.parts[0]!="output" or not path.is_file():raise RuntimeError("Prediction file must be a retained Step38 output")
    return path,str(rel)

def _parse_count_token(token,stamp):
    if re.fullmatch(rb"0|[1-9][0-9]*",token) is None:
        raise RuntimeError("Noninteger/missing count at "+stamp+"; no outcome-dependent row dropping")
    value=int(token)
    if not 0<=value<=2**53:raise RuntimeError("Count cannot be represented exactly as binary64 input")
    return value

def _read_approved_counts(stamps,stage):
    """The only numerical source-entry point; private calls also enforce gates."""
    if stage=="train":verify_protocol_seal();limit=TRAIN_CUTOFF
    elif stage=="validation":verify_fit_seal();limit=VALIDATION_CUTOFF
    elif stage in ("test_history","test_target"):
        verify_model_seal();state=_load_test_state();pending=state.get("pending_query")
        if pending is None:raise RuntimeError("No current test query authorizes count access")
        _verify_query(pending,state["next_index"])
        if stage=="test_history":
            if state["status"]!="PENDING_HISTORY" or state.get("history_complete"):
                raise RuntimeError("Test history already consumed or predictions already sealed")
            limit=_timestamp(pending["origin_timestamp"])
            expected=_required_times(_timestamp(pending["target_timestamp"]))[1]
            if list(stamps)!=expected:raise RuntimeError("Test-history read is not the frozen own24-hour window")
        else:
            if state["status"]!="PREDICTIONS_SEALED" or state.get("target_decode_started"):
                raise RuntimeError("Predictions must be sealed before exactly one target count access")
            limit=_timestamp(pending["target_timestamp"])
            if list(stamps)!=[limit]:raise RuntimeError("Only the current sealed query target may be revealed")
            seal=read_json(ROOT/state["prediction_seal_path"])
            path,_=_prediction_path(seal["prediction_path"])
            if (sha_file(ROOT/state["prediction_seal_path"] )!=state["prediction_seal_sha256"]
                or sha_file(path)!=seal["prediction_sha256"] or seal["query_id"]!=pending["id"]
                or seal["model_seal_sha256"]!=state["model_seal_sha256"]):
                raise RuntimeError("Current test prediction seal changed")
            state["target_decode_started"]=True;write_json(_state_path(),state)
    else:raise RuntimeError("Unknown count-access phase")
    if not stamps or any(stamp>limit or not SOURCE_FIRST<=stamp<SOURCE_STOP for stamp in stamps):
        raise RuntimeError("Attempted count read beyond the permitted forecast boundary")
    source,projection,index=_calendar();archive=ROOT/ARCHIVE_RELATIVE
    if sha_file(archive)!=source["archive_sha256"]:raise RuntimeError("Official archive changed")
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as zipped:
        if zipped.namelist()!=[MEMBER]:raise RuntimeError("Source archive member roster changed")
        raw=zipped.read(MEMBER)
    if hashlib.sha256(raw).hexdigest()!=projection["csv_sha256"]:raise RuntimeError("Opaque source member changed")
    values={}
    for stamp in stamps:
        stamp_text=_iso(stamp);row=index[stamp_text]
        record=raw[row["byte_start"]:row["byte_stop"]].rstrip(b"\r\n")
        cells=record.split(b",")
        if len(cells)!=14 or b'"' in record:raise RuntimeError("Source record structure differs")
        # This is the first and only point where a requested measurement token
        # is interpreted. Weather/season/holiday/operation values are not read.
        values[stamp_text]=_parse_count_token(cells[1],stamp_text)
    receipt={"stage":stage,"count_token_decodes":len(stamps),"unique_count_timestamps":len(set(stamps)),
        "requested_timestamps":[_iso(stamp) for stamp in stamps],"maximum_allowed_timestamp":_iso(limit),
        "maximum_decoded_timestamp":_iso(max(stamps)),"archive_sha256":source["archive_sha256"],
        "weather_values_decoded_or_parsed":0,"future_operation_flag_used":False}
    return values,receipt

def _normalization():
    n=read_json(ROOT/"output/normalization38.json")
    if (n.get("schema")!="step38-training-target-normalization-v1" or n.get("fitted_partition")!="train"
        or not isinstance(n["target_min"],int) or not isinstance(n["target_max"],int)
        or n["target_min"]>=n["target_max"] or n["span"]!=n["target_max"]-n["target_min"]):
        raise RuntimeError("Training-only target normalization changed")
    return n

def _normalize(value,normalization):return F(value-normalization["target_min"],normalization["span"])

def _load_cohort(name):
    if name=="train":verify_protocol_seal()
    elif name=="validation":verify_fit_seal()
    else:raise RuntimeError("Bulk test-cohort loading is forbidden; use chronological query API")
    marker=ROOT/("output/data_access_"+name+"_started38.json")
    if marker.exists():raise RuntimeError("This development phase has already begun; no rerun")
    queries=_queries(name);needed=set()
    for query in queries:
        target=_timestamp(query["target_timestamp"]);_,lags=_required_times(target);needed.add(target);needed.update(lags)
    marker.parent.mkdir(parents=True,exist_ok=True)
    with marker.open("xb") as handle:handle.write(canonical_bytes({"phase":name,"protocol_seal_sha256":sha_file(ROOT/"protocol_seal38.json")}))
    event(name+"_count_access_started",protocol_seal_sha256=sha_file(ROOT/"protocol_seal38.json"),
          fit_seal_sha256=sha_file(ROOT/"output/fit_seal38.json") if name=="validation" else None,
          target_count=len(queries),count_tokens_requested=len(needed),maximum_allowed_timestamp=_iso(TRAIN_CUTOFF if name=="train" else VALIDATION_CUTOFF))
    try:
        counts,access=_read_approved_counts(sorted(needed),name)
        raw_targets=[counts[query["target_timestamp"]] for query in queries]
        if name=="train":
            lo,hi=min(raw_targets),max(raw_targets)
            if lo==hi:
                write_json(ROOT/"output/target_normalization_failure38.json",{
                    "schema":"step38-training-target-normalization-failure-v1","status":"STOP_RETAIN_FAILURE",
                    "reason":"ZERO_TRAINING_TARGET_SPAN","target_min":lo,"target_max":hi,"span":0,
                    "raw_train_targets_sha256":digest(raw_targets),"target_ids_sha256":digest([q["id"] for q in queries]),
                    "access":access,"validation_or_test_values_used":False})
                raise RuntimeError("Zero training target span: STOP_RETAIN_FAILURE")
            normal={"schema":"step38-training-target-normalization-v1","fitted_partition":"train",
                "target_min":lo,"target_max":hi,"span":hi-lo,"degenerate_range_action":"STOP_RETAIN_FAILURE",
                "train_target_ids_sha256":digest([q["id"] for q in queries]),"raw_train_targets_sha256":digest(raw_targets),
                "target_values_clipped":False,"future_data_used":False}
            write_json(ROOT/"output/normalization38.json",normal)
        else:normal=_normalization()
        matrix=[];raw_persistence=[];lag_timestamps=[]
        for query in queries:
            target=_timestamp(query["target_timestamp"]);origin,lags=_required_times(target)
            matrix.append([counts[_iso(stamp)] for stamp in lags]);raw_persistence.append(counts[_iso(origin)])
            lag_timestamps.append([_iso(stamp) for stamp in lags])
        X=np.asarray(matrix,dtype=np.float64);X.flags.writeable=False
        relative="output/cohorts/"+name+"_X.npy";(ROOT/relative).parent.mkdir(parents=True,exist_ok=True)
        np.save(ROOT/relative,X,allow_pickle=False)
        y=[_normalize(value,normal) for value in raw_targets];persistence=[_normalize(value,normal) for value in raw_persistence]
        record={"schema":"step38-causal-cohort-v1","name":name,"target":TARGET,"n":len(queries),
            "ids":[query["id"] for query in queries],"target_timestamps":[q["target_timestamp"] for q in queries],
            "origin_timestamps":[q["origin_timestamp"] for q in queries],"lag_timestamps":lag_timestamps,
            "y":y,"raw_targets":raw_targets,"persistence":persistence,"raw_persistence":raw_persistence,
            "X_path":relative,"X_sha256":sha_file(ROOT/relative),"normalization_sha256":sha_file(ROOT/"output/normalization38.json"),
            "archive_sha256":access["archive_sha256"],"all_recorded_operation_states_retained":True,
            "weather_or_future_operation_features_used":False,"output_range_target_clipping":False}
        cohort_path="output/cohorts/"+name+".json";write_json(ROOT/cohort_path,record)
        access.update({"schema":"step38-development-count-access-v1","status":"PASS","cohort_path":cohort_path,
            "cohort_sha256":sha_file(ROOT/cohort_path),"normalization_sha256":record["normalization_sha256"]})
        access_path="output/data_access_"+name+"38.json";write_json(ROOT/access_path,access)
        event(name+"_count_access_completed",cohort_path=cohort_path,cohort_sha256=sha_file(ROOT/cohort_path),
              access_path=access_path,access_sha256=sha_file(ROOT/access_path),n=len(queries),test_target_values_decoded=0)
        return {**record,"X":X,"y":y,"persistence":persistence,"path":cohort_path,"sha256":sha_file(ROOT/cohort_path)}
    except Exception as exc:
        event(name+"_count_access_failed",error_type=type(exc).__name__,message=str(exc));raise

def load_training():return _load_cohort("train")
def load_validation():return _load_cohort("validation")

def load_test_history(query):
    verify_model_seal()
    if _state_path().exists():state=_load_test_state()
    else:
        state={"schema":"step38-chronological-test-access-state-v1","model_seal_sha256":sha_file(ROOT/"output/model_seal38.json"),
            "next_index":0,"pending_query":None,"previously_revealed_test_targets":[],"status":"READY"}
    if state["status"]!="READY" or state["pending_query"] is not None:
        raise RuntimeError("Reveal the current target before advancing; test history cannot be reread")
    _verify_query(query,state["next_index"])
    state.update({"pending_query":dict(query),"status":"PENDING_HISTORY"});write_json(_state_path(),state)
    origin,lags=_required_times(_timestamp(query["target_timestamp"]))
    event("test_history_access_started",query_id=query["id"],query_index=state["next_index"],
          target_timestamp=query["target_timestamp"],origin_timestamp=query["origin_timestamp"],
          model_seal_sha256=state["model_seal_sha256"],maximum_allowed_timestamp=_iso(origin))
    try:
        counts,access=_read_approved_counts(lags,"test_history");normal=_normalization()
        X=np.asarray([[counts[_iso(stamp)] for stamp in lags]],dtype=np.float64);X.flags.writeable=False
        index=state["next_index"];X_path=f"output/test_history/{index:03d}_X.npy";(ROOT/X_path).parent.mkdir(parents=True,exist_ok=True)
        np.save(ROOT/X_path,X,allow_pickle=False)
        overlap=[_iso(stamp) for stamp in lags if _iso(stamp) in state["previously_revealed_test_targets"]]
        record={"schema":"step38-causal-test-history-v1","query_id":query["id"],"query_index":index,
            "target_timestamp":query["target_timestamp"],"origin_timestamp":query["origin_timestamp"],
            "history_timestamps":[_iso(stamp) for stamp in lags],"history_counts":[counts[_iso(stamp)] for stamp in lags],
            "persistence":_normalize(counts[_iso(origin)],normal),"raw_persistence":counts[_iso(origin)],
            "X_path":X_path,"X_sha256":sha_file(ROOT/X_path),"model_seal_sha256":state["model_seal_sha256"],
            "normalization_sha256":sha_file(ROOT/"output/normalization38.json"),
            "previously_revealed_test_targets_in_history":overlap,"current_target_count_decoded":False,
            "access":access,"weather_or_future_operation_features_used":False}
        relative=f"output/test_history/{index:03d}.json";write_json(ROOT/relative,record)
        state.update({"history_receipt_path":relative,"history_receipt_sha256":sha_file(ROOT/relative),"history_complete":True})
        write_json(_state_path(),state)
        event("test_history_access_completed",query_id=query["id"],query_index=index,history_receipt_path=relative,
            history_receipt_sha256=sha_file(ROOT/relative),maximum_decoded_timestamp=access["maximum_decoded_timestamp"],
            previously_revealed_test_targets_in_history=overlap,current_target_count_decoded=False)
        return {**record,"X":X,"persistence":F(record["persistence"]),"history_receipt_path":relative,
                "history_receipt_sha256":sha_file(ROOT/relative)}
    except Exception as exc:
        state["status"]="FAILED";state["error"]={"type":type(exc).__name__,"message":str(exc)};write_json(_state_path(),state)
        event("test_history_access_failed",query_id=query["id"],error=state["error"]);raise

def reveal_test_target(query,prediction_path,prediction_sha256):
    protocol=verify_protocol_seal();verify_model_seal();state=_load_test_state();_verify_query(query,state["next_index"])
    if state["status"]!="PENDING_HISTORY" or state.get("pending_query")!=query or not state.get("history_complete"):
        raise RuntimeError("Current query history must be consumed once before its prediction seal")
    if sha_file(ROOT/state["history_receipt_path"] )!=state["history_receipt_sha256"]:
        raise RuntimeError("Current test-history receipt changed")
    path,relative=_prediction_path(prediction_path)
    if sha_file(path)!=prediction_sha256:raise RuntimeError("Provided prediction digest differs")
    prediction=read_json(path);ids=protocol["evaluation"]["prediction_ids"]
    if (not isinstance(ids,list) or len(ids)!=len(set(ids)) or len(ids)!=93 or not all(isinstance(i,str) and i for i in ids)
        or prediction.get("schema")!="step38-sealed-test-predictions-v1" or prediction.get("query_id")!=query["id"]
        or prediction.get("model_seal_sha256")!=state["model_seal_sha256"]
        or prediction.get("history_receipt_sha256")!=state["history_receipt_sha256"]
        or not isinstance(prediction.get("predictions"),dict) or set(prediction["predictions"])!=set(ids)):
        raise RuntimeError("Prediction record omits or changes a frozen query/model/history/forecast identity")
    index=state["next_index"];seal_path=f"output/test_prediction_seals/{index:03d}.json"
    if (ROOT/seal_path).exists():raise RuntimeError("This target already has a prediction seal")
    seal={"schema":"step38-test-prediction-byte-seal-v1","query_id":query["id"],"query_index":index,
        "model_seal_sha256":state["model_seal_sha256"],"prediction_path":relative,"prediction_sha256":prediction_sha256,
        "history_receipt_path":state["history_receipt_path"],"history_receipt_sha256":state["history_receipt_sha256"],
        "prediction_ids":ids,"prediction_count":len(ids),"current_target_count_decoded":False}
    write_json(ROOT/seal_path,seal)
    state.update({"status":"PREDICTIONS_SEALED","prediction_seal_path":seal_path,"prediction_seal_sha256":sha_file(ROOT/seal_path)})
    write_json(_state_path(),state)
    event("test_predictions_sealed",query_id=query["id"],query_index=index,prediction_seal_path=seal_path,
          prediction_seal_sha256=sha_file(ROOT/seal_path),prediction_path=relative,prediction_sha256=prediction_sha256,
          prediction_count=len(ids),current_target_count_decoded=False)
    try:
        target=_timestamp(query["target_timestamp"])
        event("test_target_access_started",query_id=query["id"],query_index=index,
              prediction_seal_sha256=state["prediction_seal_sha256"],target_timestamp=query["target_timestamp"])
        counts,access=_read_approved_counts([target],"test_target");normal=_normalization();raw=counts[_iso(target)];y=_normalize(raw,normal)
        record={"schema":"step38-revealed-test-target-v1","query_id":query["id"],"query_index":index,
            "target_timestamp":query["target_timestamp"],"origin_timestamp":query["origin_timestamp"],
            "raw_target":raw,"y":y,"normalization_sha256":sha_file(ROOT/"output/normalization38.json"),
            "prediction_seal_path":seal_path,"prediction_seal_sha256":state["prediction_seal_sha256"],
            "model_seal_sha256":state["model_seal_sha256"],"access":access,"target_value_clipped":False}
        target_path=f"output/test_targets/{index:03d}.json";write_json(ROOT/target_path,record)
        state["previously_revealed_test_targets"].append(query["target_timestamp"])
        state["next_index"]+=1;state["pending_query"]=None;state["status"]="COMPLETE" if state["next_index"]==EXPECTED_N["test"] else "READY"
        for key in ("history_receipt_path","history_receipt_sha256","history_complete","prediction_seal_path","prediction_seal_sha256"):
            state.pop(key,None)
        write_json(_state_path(),state)
        event("test_target_revealed",query_id=query["id"],query_index=index,target_path=target_path,target_sha256=sha_file(ROOT/target_path),
              prediction_seal_sha256=record["prediction_seal_sha256"],next_query_index=state["next_index"],
              chronological_test_status=state["status"])
        return {**record,"y":y,"path":target_path,"sha256":sha_file(ROOT/target_path)}
    except Exception as exc:
        state["status"]="FAILED";state["error"]={"type":type(exc).__name__,"message":str(exc)};write_json(_state_path(),state)
        event("test_target_access_failed",query_id=query["id"],error=state["error"]);raise

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("metadata",));args=parser.parse_args()
    result=metadata_rosters()
    print(json.dumps({"status":"PASS","count_values_decoded_or_parsed":0,
        "targets":{name:item["timestamp_complete_targets"] for name,item in result["partitions"].items()},
        "targets_flagged_nonfunctional":{name:item["targets_flagged_nonfunctional"] for name,item in result["partitions"].items()}}))
