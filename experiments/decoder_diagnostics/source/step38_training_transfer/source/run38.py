"""One gated train/validation/sequential-test pass for the frozen Step38 study."""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from fractions import Fraction as F
import json
from pathlib import Path
import pickle
import signal
import time
import traceback
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import data38 as d
import learners38 as m
import representations35 as rep
import finite_feature35 as ff

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"output"
def read(path):return json.loads(Path(path).read_text())
def rel(path):return str(Path(path).relative_to(ROOT))
def save(path,value):d.write_json(path,m.canonical(value));return {"path":rel(path),"sha256":d.sha_file(path)}
def save_features(role,name,z):
    path=OUT/"features"/role/(name+".npy");path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,np.asarray(z,dtype=np.float64),allow_pickle=False)
    return {"path":rel(path),"sha256":d.sha_file(path),"feature_digest":ff.digest(m.features_exact(z))}
def progress(stage,**fields):print(json.dumps({"stage":stage,**fields}),flush=True)
def timeout(signum,frame):raise TimeoutError("Frozen phase budget exceeded; retain evidence without rerun")

def start_phase(name,protocol):
    marker=OUT/(name+"_started38.json")
    if marker.exists():raise RuntimeError("A Step38 phase cannot be rerun: "+name)
    if name=="train":
        if (OUT/"primary_started38.json").exists():raise RuntimeError("Primary run already started")
        save(OUT/"primary_started38.json",{"protocol_seal_sha256":d.sha_file(ROOT/"protocol_seal38.json"),
            "started_utc":datetime.now(timezone.utc).isoformat(),"single_primary_pass":True})
    elif not (OUT/"primary_started38.json").exists():raise RuntimeError("No preceding primary training run")
    save(marker,{"phase":name,"started_utc":datetime.now(timezone.utc).isoformat(),"protocol_seal_sha256":d.sha_file(ROOT/"protocol_seal38.json")})
    d.event("phase_started",phase=name,marker_sha256=d.sha_file(marker))
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,protocol["execution"]["phase_budget_seconds"][name])

def end_phase(name,started,**fields):
    signal.setitimer(signal.ITIMER_REAL,0)
    record={"schema":"step38-phase-completion-v1","phase":name,"status":"COMPLETE","wall_seconds":time.perf_counter()-started,**fields}
    save(OUT/(name+"_completed38.json"),record);d.event("phase_completed",phase=name,receipt_sha256=d.sha_file(OUT/(name+"_completed38.json")),**fields)
    progress(name+"_COMPLETE",seconds=record["wall_seconds"],**fields)

def seal_output(name,extra):
    path=OUT/name
    if path.exists():raise RuntimeError("Cannot replace output seal")
    files=[p for p in OUT.rglob("*") if p.is_file() and p!=path and p.name!="access_and_execution38.jsonl" and not p.name.endswith(".pending")]
    record={"schema":"step38-"+("fit" if name=="fit_seal38.json" else "model-selection")+"-seal-v1",
        "sealed_utc":datetime.now(timezone.utc).isoformat(),"protocol_seal_sha256":d.sha_file(ROOT/"protocol_seal38.json"),
        "bound_files":{rel(p):d.sha_file(p) for p in sorted(files)},**extra}
    save(path,record);d.event("fit_sealed" if name=="fit_seal38.json" else "model_selection_sealed",seal_sha256=d.sha_file(path),**extra)
    return record

def train(protocol):
    started=time.perf_counter();start_phase("train",protocol)
    cohort=d.load_training();X=cohort["X"];y=cohort["y"];ids=cohort["ids"]
    if len(y)!=protocol["data"]["expected_n"]["train"]:raise RuntimeError("Unexpected training roster size")
    epsilon=F(protocol["selection"]["epsilon_factor"])*sum(((p-t)**2 for p,t in zip(cohort["persistence"],y)),F(0))/len(y)
    anchors=rep.temporal_indices(len(y),32).tolist()
    norm=read(OUT/"normalization38.json")
    plan={"schema":"step38-training-plan-v1","n":len(y),"epsilon":str(epsilon),"epsilon_physical_mse":str(epsilon*F(norm["span"])**2),
        "anchor_indices32":anchors,"anchor_ids32":[ids[i] for i in anchors],"source_role":"train_only"}
    save(OUT/"training_plan38.json",plan)
    map_start=time.perf_counter();maps,metadata=rep.fit_representations(X,protocol["representations"],protocol["seed"])
    if time.perf_counter()-map_start>protocol["execution"]["native_representation_budget_seconds"]:raise TimeoutError("Frozen representation budget exceeded")
    save(OUT/"representation_metadata38.json",metadata)
    rows=[];map_info={};group_timings=[]
    for index,name in enumerate(protocol["maps"]):
        group_start=time.perf_counter();representation=maps[name];payload=rep.serialize_map(representation)
        map_path=OUT/"maps"/(name+".pkl");map_path.parent.mkdir(parents=True,exist_ok=True);map_path.write_bytes(payload)
        z=rep.transform_one(representation,X);feature_ref=save_features("train",name,z)
        feature_id="feature38:"+m.digest({"map":name,"map_sha256":d.sha_file(map_path),"train_feature_sha256":feature_ref["sha256"]})
        info={"name":name,"family":representation.family,"scaling":representation.scaling,"dimension":int(z.shape[1]),
            "feature_id":feature_id,"class_id":ff.class_identity(feature_id,"2",("0","1")),
            "map_path":rel(map_path),"map_sha256":d.sha_file(map_path),"map_payload_bytes":len(payload),
            "map_numeric_bytes":rep.numeric_bytes(representation),"training_features":feature_ref}
        map_info[name]=info
        binding={"cohort_path":"output/cohorts/train.json","cohort_sha256":d.sha_file(OUT/"cohorts/train.json"),
            "feature_path":feature_ref["path"],"feature_sha256":feature_ref["sha256"],"map_sha256":info["map_sha256"]}
        fitdir=OUT/"fits"/name;fitdir.mkdir(parents=True,exist_ok=True)
        affine_start=time.perf_counter();affine=m.fit_affine(z,y,feature_id=feature_id,kappa="2",ridge_lambda=protocol["heads"]["ridge_lambda"],training_binding=binding)
        affine_seconds=time.perf_counter()-affine_start;affine_ref=save(fitdir/"affine.json",affine)
        old_start=time.perf_counter();legacy=ff.fit_anchor_decoder(z[anchors].tolist(),[y[i] for i in anchors],feature_id=feature_id,
            ids=[ids[i] for i in anchors],kappa="2",output_interval=("0","1"),distance_bits=80,
            max_updates=protocol["heads"]["legacy_extension_max_updates"],check_every=protocol["heads"]["legacy_check_every"],
            gap_sse=protocol["heads"]["legacy_gap_sse"],grid_bits=40)
        legacy["training_binding"]={**binding,"anchor_indices":anchors};old_seconds=time.perf_counter()-old_start
        old_ref=save(fitdir/"extension32.json",legacy)
        small=m.fit_values(z[anchors],[y[i] for i in anchors],ids=[ids[i] for i in anchors],feature_id=feature_id,
            input_binding={**binding,"anchor_indices":anchors},settings=protocol["heads"]["solver_settings"])
        small_ref=save(fitdir/"qp32.json",small)
        full=m.fit_values(z,y,ids=ids,feature_id=feature_id,input_binding={**binding,"anchor_indices":list(range(len(y)))},settings=protocol["heads"]["solver_settings"])
        full_ref=save(fitdir/"full_training.json",full)
        head_data={"affine":(affine["head"],affine_ref),"extension32":(legacy["head"],old_ref),
            "qp32":(m.make_extension(small,z[anchors],anchor_indices=anchors,recipe="qp32",fit_path=small_ref["path"],fit_sha256=small_ref["sha256"]),small_ref),
            "envelope_all":(m.make_extension(full,z,anchor_indices=list(range(len(y))),recipe="envelope_all",fit_path=full_ref["path"],fit_sha256=full_ref["sha256"]),full_ref),
            "qp_all":(m.make_extension(full,z,anchor_indices=list(range(len(y))),recipe="qp_all",fit_path=full_ref["path"],fit_sha256=full_ref["sha256"]),full_ref)}
        for recipe in protocol["heads"]["recipes"]:
            head,fit_ref=head_data[recipe];head_ref=save(OUT/"heads"/name/(recipe+".json"),head);headbytes=(ROOT/head_ref["path"]).stat().st_size
            rows.append({"candidate_id":name+"::"+recipe,"map":name,"recipe":recipe,"family":info["family"],"scaling":info["scaling"],
                "dimension":info["dimension"],"feature_id":feature_id,"class_id":info["class_id"],"head_id":m.head_identity(head),
                "head_path":head_ref["path"],"head_sha256":head_ref["sha256"],"fit_path":fit_ref["path"],"fit_sha256":fit_ref["sha256"],
                "head_payload_bytes":headbytes,"map_payload_bytes":len(payload),"deployment_payload_bytes":headbytes+len(payload),
                "anchor_count":0 if recipe=="affine" else len(head["anchor_ids"]),"training_only":True,"assurance":ff.ASSURANCE})
        group_timings.append({"map":name,"affine_fit_seconds":affine_seconds,"legacy_extension_fit_seconds":old_seconds,
            "qp32_fit_seconds":small["timing"]["total_seconds"],"shared_full_fit_seconds":full["timing"]["total_seconds"],
            "shared_full_graph_seconds":full["timing"]["exact_graph_seconds"],"shared_full_solver_seconds":full["timing"]["numerical_proposal_seconds"],
            "shared_full_exact_witness_seconds":full["timing"]["exact_witness_seconds"],"whole_map_group_seconds":time.perf_counter()-group_start})
        progress("TRAIN_MAP",completed=index+1,map=name,full_gap=float(F(full["class_risk"]["width"])),solver=full["numerical"]["status"])
    baseline_start=time.perf_counter();scaler=maps["raw_minmax01"].core
    hgb=HistGradientBoostingRegressor(**protocol["baselines"]["hgb_parameters"])
    hgb.fit(scaler.transform(X),np.asarray([float(v) for v in y]));mean=sum(y,F(0))/len(y)
    hgb_path=OUT/"baselines/hgb.pkl";hgb_path.parent.mkdir(parents=True,exist_ok=True)
    hgb_path.write_bytes(pickle.dumps({"model":hgb,"scaler":scaler},protocol=5))
    baselines={"hist_gradient_boosting":{"payload_path":rel(hgb_path),"payload_sha256":d.sha_file(hgb_path),"deployment_payload_bytes":hgb_path.stat().st_size},
        "training_mean":{"value":str(mean),"deployment_payload_bytes":len(m.canonical_bytes({"value":str(mean)}))},
        "persistence":{"deployment_payload_bytes":len(m.canonical_bytes({"rule":"last_observed_count"}))}}
    manifest={"schema":"step38-fit-manifest-v1","candidate_rows":rows,"maps":map_info,"baselines":baselines,
        "training_plan_sha256":d.sha_file(OUT/"training_plan38.json"),"training_n":len(y),"group_timings":group_timings,
        "representation_seconds":metadata["factory_total_seconds"],"baseline_fit_seconds":time.perf_counter()-baseline_start,
        "all_fitting_precedes_validation_access":True,"selection_available":False}
    save(OUT/"fit_manifest38.json",manifest)
    end_phase("train",started,head_count=len(rows),map_count=len(maps))
    seal_output("fit_seal38.json",{"fit_manifest_sha256":d.sha_file(OUT/"fit_manifest38.json"),"head_count":90,"validation_values_parsed":0,"test_values_parsed":0})

def load_models():
    manifest=read(OUT/"fit_manifest38.json");maps={};heads={};train_z={}
    for name,info in manifest["maps"].items():
        maps[name]=rep.deserialize_map((ROOT/info["map_path"]).read_bytes())
        train_z[name]=np.load(ROOT/info["training_features"]["path"],allow_pickle=False)
    for row in manifest["candidate_rows"]:heads[row["candidate_id"]]=read(ROOT/row["head_path"])
    hgb=pickle.loads((ROOT/manifest["baselines"]["hist_gradient_boosting"]["payload_path"]).read_bytes())
    training=read(OUT/"cohorts/train.json")
    return manifest,maps,heads,train_z,hgb,training

def evaluate_all(protocol,manifest,maps,heads,train_z,hgb,training,X,ids,persistence,role):
    predictions={};map_records={};features={};shared_seconds=0.;head_seconds=0.;transform_seconds=0.
    for name in protocol["maps"]:
        s=time.perf_counter();z=rep.transform_one(maps[name],X);transform_seconds+=time.perf_counter()-s
        features[name]=z;fref=save_features(role,name,z)
        s=time.perf_counter();distance=m.query_distances(train_z[name],z,anchor_ids=training["ids"],query_ids=ids,feature_id=manifest["maps"][name]["feature_id"])
        shared_seconds+=time.perf_counter()-s
        dref=save(OUT/"distances"/role/(name+".json"),distance)
        map_records[name]={"features":fref,"distances":dref,"distance_digest":m.digest(distance)}
        for recipe in protocol["heads"]["recipes"]:
            cid=name+"::"+recipe;s=time.perf_counter()
            predictions[cid]=m.evaluate_head(heads[cid],z,ids=ids,distance=distance)
            head_seconds+=time.perf_counter()-s
    values={"persistence":persistence,"training_mean":[F(manifest["baselines"]["training_mean"]["value"])]*len(ids),
        "hist_gradient_boosting":hgb["model"].predict(hgb["scaler"].transform(X)).tolist()}
    for name in protocol["baselines"]["names"]:predictions["baseline::"+name]=m.baseline_prediction(name,values[name],ids)
    if list(predictions)!=protocol["evaluation"]["prediction_ids"]:raise RuntimeError("Prediction roster/order changed")
    return predictions,map_records,features,{"shared_transform_seconds":transform_seconds,"shared_distance_seconds":shared_seconds,"all_head_interval_seconds":head_seconds}

def validate(protocol):
    started=time.perf_counter();start_phase("validate",protocol)
    cohort=d.load_validation();manifest,maps,heads,train_z,hgb,training=load_models()
    predictions,map_records,features,timing=evaluate_all(protocol,manifest,maps,heads,train_z,hgb,training,
        cohort["X"],cohort["ids"],cohort["persistence"],"validation")
    save(OUT/"validation_predictions38.json",{"schema":"step38-validation-predictions-v1","predictions":predictions,"maps":map_records,
        "fit_seal_sha256":d.sha_file(OUT/"fit_seal38.json"),"query_ids":cohort["ids"],"timing":timing})
    epsilon=F(read(OUT/"training_plan38.json")["epsilon"]);rows=[]
    for row in manifest["candidate_rows"]:
        risk=m.score(predictions[row["candidate_id"]]["predictions"],cohort["y"])
        rows.append({**row,"validation_risk":risk,"validation_eligible":F(risk["upper"])<=epsilon,
            "validation_prediction_digest":m.digest(predictions[row["candidate_id"]])})
    baseline_rows=[]
    for name in protocol["baselines"]["names"]:
        cid="baseline::"+name;risk=m.score(predictions[cid]["predictions"],cohort["y"])
        baseline_rows.append({"candidate_id":cid,"baseline":name,"validation_risk":risk,"validation_eligible":F(risk["upper"])<=epsilon,
            "deployment_payload_bytes":manifest["baselines"][name]["deployment_payload_bytes"],"validation_prediction_digest":m.digest(predictions[cid])})
    selections={recipe:m.select([row for row in rows if row["recipe"]==recipe]) for recipe in protocol["heads"]["recipes"]}
    selections["legacybest"]=m.select([row for row in rows if row["recipe"] in ("affine","extension32")])
    selections["ordinarybest"]=m.select(baseline_rows)
    result={"schema":"step38-validation-selection-v1","candidate_rows":rows,"baseline_rows":baseline_rows,"selections":selections,
        "validation_n":len(cohort["y"]),"epsilon":str(epsilon),"fit_seal_sha256":d.sha_file(OUT/"fit_seal38.json"),
        "selection_uses_validation_only":True,"test_values_parsed":0,"timing":timing}
    save(OUT/"validation_selection38.json",result)
    end_phase("validate",started,validation_n=len(cohort["y"]),selected_ids={k:v["candidate_id"] for k,v in selections.items()})
    seal_output("model_seal38.json",{"fit_seal_sha256":d.sha_file(OUT/"fit_seal38.json"),
        "selection_sha256":d.sha_file(OUT/"validation_selection38.json"),"selected_ids":{k:v["candidate_id"] for k,v in selections.items()},"test_values_parsed":0})

def test(protocol):
    started=time.perf_counter();start_phase("test",protocol)
    manifest,maps,heads,train_z,hgb,training=load_models();selection=read(OUT/"validation_selection38.json")
    queries=d.test_queries();all_predictions={cid:[] for cid in protocol["evaluation"]["prediction_ids"]};ys=[];raw_targets=[]
    features={name:[] for name in protocol["maps"]};query_records=[];timings=[]
    for index,query in enumerate(queries):
        history=d.load_test_history(query)
        persistence=history["persistence"] if isinstance(history["persistence"],list) else [history["persistence"]]
        preds,map_records,z,timing=evaluate_all(protocol,manifest,maps,heads,train_z,hgb,training,
            history["X"],[query["id"]],persistence,"test/q"+str(index).zfill(3))
        record={"schema":"step38-sealed-test-predictions-v1","query_id":query["id"],"target_timestamp":query["target_timestamp"],
            "origin_timestamp":query["origin_timestamp"],"model_seal_sha256":d.sha_file(OUT/"model_seal38.json"),
            "history_receipt_sha256":history["history_receipt_sha256"],"history_receipt_path":history["history_receipt_path"],
            "predictions":preds,"maps":map_records,"timing":timing,"sealed_before_current_target_reveal":True,
            "future_target_used_in_prediction":False}
        predpath=OUT/"test_predictions"/("q"+str(index).zfill(3)+".json");pref=save(predpath,record)
        d.event("test_prediction_receipt_written",query_id=query["id"],prediction_path=pref["path"],prediction_sha256=pref["sha256"],prediction_count=len(preds))
        revealed=d.reveal_test_target(query,pref["path"],pref["sha256"])
        target=F(revealed["y"]);ys.append(target);raw_targets.append(revealed["raw_target"])
        for cid in all_predictions:all_predictions[cid].append(preds[cid]["predictions"][0])
        for name in features:features[name].append(z[name][0])
        query_records.append({**query,"prediction_path":pref["path"],"prediction_sha256":pref["sha256"],
            "history_receipt_path":history["history_receipt_path"],"history_receipt_sha256":history["history_receipt_sha256"],
            "target_reveal":m.canonical(revealed)})
        timings.append(timing)
        if index%10==0 or index+1==len(queries):progress("TEST_QUERY",completed=index+1,total=len(queries))
    forecast_seconds=time.perf_counter()-started
    test_cohort={"schema":"step38-completed-test-cohort-v1","n":len(ys),"ids":[q["id"] for q in queries],
        "target_timestamps":[q["target_timestamp"] for q in queries],"origin_timestamps":[q["origin_timestamp"] for q in queries],
        "y":list(map(str,ys)),"raw_targets":raw_targets,"queries":query_records,"sequential_prediction_then_label_verified_by_loader":True}
    save(OUT/"cohorts/test.json",test_cohort)
    rows=[]
    for row in selection["candidate_rows"]+selection["baseline_rows"]:
        cid=row["candidate_id"];loss=m.score(all_predictions[cid],ys)
        monthly={}
        for month in ("2018-09","2018-10","2018-11"):
            indices=[i for i,q in enumerate(queries) if q["target_timestamp"].startswith(month)]
            monthly[month]=m.score([all_predictions[cid][i] for i in indices],[ys[i] for i in indices])
        rows.append({**row,"test_risk":loss,"test_monthly_risk":monthly,
            "test_eligible":F(loss["upper"])<=F(selection["epsilon"]),"test_predictions_digest":m.digest(all_predictions[cid])})
    lookup={r["candidate_id"]:r for r in rows};selected={label:lookup[s["candidate_id"]] for label,s in selection["selections"].items()}
    gate=m.practical_gate(selected["qp_all"]["test_risk"],selected["legacybest"]["test_risk"],selected["ordinarybest"]["test_risk"])
    primary={"schema":"step38-primary-forecast-result-v1","candidate_rows":rows,"selected":selected,"practical_gate":gate,
        "test_n":len(ys),"forecast_seconds":forecast_seconds,"query_timing":timings,
        "model_seal_sha256":d.sha_file(OUT/"model_seal38.json"),"selection_sha256":d.sha_file(OUT/"validation_selection38.json")}
    save(OUT/"primary_forecast_result38.json",primary)
    d.event("all_test_forecasts_and_scores_complete",receipt_sha256=d.sha_file(OUT/"primary_forecast_result38.json"),test_n=len(ys))
    secondary_start=time.perf_counter();secondary=[]
    for name in protocol["maps"]:
        z=np.asarray(features[name],dtype=np.float64);fref=save_features("test_all",name,z)
        fit=m.fit_values(z,ys,ids=test_cohort["ids"],feature_id=manifest["maps"][name]["feature_id"],
            role="assessment_diagnostic",settings=protocol["heads"]["solver_settings"],
            input_binding={"feature_path":fref["path"],"feature_sha256":fref["sha256"],"cohort_path":"output/cohorts/test.json",
                "cohort_sha256":d.sha_file(OUT/"cohorts/test.json"),"primary_forecasts_sha256":d.sha_file(OUT/"primary_forecast_result38.json")})
        fitref=save(OUT/"secondary_class"/(name+".json"),fit)
        candidates=[r for r in rows if r.get("map")==name];lower=F(fit["class_risk"]["lower"])
        upper=min(F(fit["class_risk"]["upper"]),min(F(r["test_risk"]["upper"]) for r in candidates))
        if not 0<=lower<=upper:raise ArithmeticError("Secondary class and fixed-head risks conflict")
        eps=F(selection["epsilon"]);status="ADEQUATE" if upper<=eps else "INADEQUATE" if lower>eps else "UNRESOLVED"
        excess=[{"candidate_id":r["candidate_id"],"recipe":r["recipe"],"excess":m.box(max(F(0),F(r["test_risk"]["lower"])-upper),max(F(0),F(r["test_risk"]["upper"])-lower))} for r in candidates]
        secondary.append({"map":name,"feature_id":fit["feature_id"],"class_id":fit["class_id"],"receipt":fitref,
            "class_risk":m.box(lower,upper),"epsilon":str(eps),"diagnosis":status,"head_excess":excess,"assessment_labels_used":True,"head_promotion_allowed":False})
    result={"schema":"step38-training-transfer-result-v1","step":38,"status":"COMPLETE",
        "protocol_sha256":d.sha_file(ROOT/"protocol38.json"),"protocol_seal_sha256":d.sha_file(ROOT/"protocol_seal38.json"),
        "fit_seal_sha256":d.sha_file(OUT/"fit_seal38.json"),"model_seal_sha256":d.sha_file(OUT/"model_seal38.json"),
        "pretest_gate_sha256":d.sha_file(ROOT/"audit/pretest_gate38.json"),"primary":primary,"secondary_classes":secondary,
        "secondary_class_seconds":time.perf_counter()-secondary_start,"normalization":read(OUT/"normalization38.json"),
        "training_plan":read(OUT/"training_plan38.json"),"train_n":len(training["ids"]),"validation_n":selection["validation_n"],
        "test_n":len(ys),"head_count":90,"baseline_count":3,"model_selection_changed_after_test":False,
        "reserved_patients_accessed":0,"native_transform_error_enclosed":False,"population_risk_claim":False,"future_response_coverage_claim":False}
    save(OUT/"result38.json",result)
    end_phase("test",started,result_sha256=d.sha_file(OUT/"result38.json"),test_n=len(ys))
    progress("PRIMARY_FINISHED",result_sha256=d.sha_file(OUT/"result38.json"),practical_gate=gate)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("train","validate","test"));args=parser.parse_args()
    protocol=d.verify_protocol_seal()
    try:{"train":train,"validate":validate,"test":test}[args.phase](protocol)
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        failure={"schema":"step38-retained-phase-failure-v1","phase":args.phase,"status":"STOPPED_ERROR","type":type(exc).__name__,
            "message":str(exc),"traceback":traceback.format_exc(),"no_rerun_authorized":True}
        save(OUT/(args.phase+"_failure38.json"),failure);d.event("phase_stopped_error",phase=args.phase,error_type=type(exc).__name__,message=str(exc))
        print(json.dumps(failure),flush=True);raise SystemExit(2)

if __name__=="__main__":main()
