"""Single-pass, gated chronological selection experiment for Step40."""
import os
for _thread in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_thread]="1"
import argparse
from fractions import Fraction as F
from pathlib import Path
import pickle
import signal
import time
import traceback
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import common40 as c
import data40 as d
import learners39 as m
import representations35 as rep
import finite_feature35 as ff
import decisions40 as decision

ROOT=c.ROOT;OUT=ROOT/"output"

def progress(label,**fields):
    print(c.canonical_bytes({"event":label,**fields}).decode(),end="",flush=True)

def stage_dir(task,stage):return OUT/task/stage

def timeout(signum,frame):raise TimeoutError("Frozen phase budget exceeded; no rerun")

def start(phase,protocol,task=None,stage=None):
    dest=OUT/(task or "global")/(stage or phase)
    marker=dest/(phase+"_started40.json")
    c.save(marker,{"schema":"step40-phase-start-v1","phase":phase,"task":task,"stage":stage,
                   "utc":c.utc(),"protocol_seal_sha256":c.sha(ROOT/"protocol_seal40.json")})
    d.event("phase_started",phase=phase,task=task,stage=stage,marker_path=c.relative(marker),marker_sha256=c.sha(marker))
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,protocol["execution"]["phase_budget_seconds"][phase])
    return time.perf_counter()

def end(phase,started,task=None,stage=None,**fields):
    signal.setitimer(signal.ITIMER_REAL,0)
    dest=OUT/(task or "global")/(stage or phase)
    value={"schema":"step40-phase-completion-v1","phase":phase,"task":task,"stage":stage,"status":"COMPLETE",
           "wall_seconds":time.perf_counter()-started,"utc":c.utc(),**fields}
    ref=c.save(dest/(phase+"_completed40.json"),value)
    d.event("phase_completed",phase=phase,task=task,stage=stage,receipt_sha256=ref["sha256"])
    progress(phase+"_COMPLETE",task=task,fit_stage=stage,seconds=value["wall_seconds"],**fields)

def save_binary(path,contents):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:stream.write(contents)
    return {"path":c.relative(path),"sha256":c.sha(path)}

def save_features(dest,z):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("xb") as stream:np.save(stream,np.asarray(z,dtype=np.float64),allow_pickle=False)
    return {"path":c.relative(dest),"sha256":c.sha(dest),"feature_digest":m.content_digest(m.features_exact(z))}

def seal(path,schema,files,**fields):
    record={"schema":schema,"sealed_utc":c.utc(),"protocol_seal_sha256":c.sha(ROOT/"protocol_seal40.json"),
            "bound_files":{c.relative(p):c.sha(p) for p in sorted(set(files))},**fields}
    ref=c.save(path,record);d.event("evidence_sealed",schema=schema,path=ref["path"],sha256=ref["sha256"],
                                 task=fields.get("task"),stage=fields.get("stage"))
    return record

def cohort_ref(cohort):return {"path":cohort["path"],"sha256":cohort["sha256"]}

def fit(protocol,task,stage):
    started=start("fit",protocol,task,stage);cohort=d.load_training(task,stage)
    base=stage_dir(task,stage);X=cohort["X"];y=list(map(F,cohort["y"]));ids=cohort["ids"]
    map_started=time.perf_counter();maps,metadata=rep.fit_representations(X,protocol["representations"],protocol["seed"])
    if time.perf_counter()-map_started>protocol["execution"]["native_representation_budget_seconds"]:
        raise TimeoutError("Frozen native representation budget exceeded")
    c.save(base/"representation_metadata40.json",metadata)
    info={};rows=[];timings=[]
    for index,name in enumerate(protocol["maps"]):
        group_started=time.perf_counter();representation=maps[name]
        payload=rep.serialize_map(representation);mapref=save_binary(base/"maps"/(name+".pkl"),payload)
        z=rep.transform_one(representation,X);fref=save_features(base/"features/train"/(name+".npy"),z)
        feature_id="feature40:"+c.digest({"task":task,"stage":stage,"map":name,"map_sha256":mapref["sha256"],"train_feature_sha256":fref["sha256"]})
        details={"name":name,"family":representation.family,"scaling":representation.scaling,"dimension":int(z.shape[1]),
                 "feature_id":feature_id,"class_id":ff.class_identity(feature_id,"2",("0","1")),
                 "map_path":mapref["path"],"map_sha256":mapref["sha256"],"map_payload_bytes":len(payload),
                 "map_numeric_bytes":rep.numeric_bytes(representation),"training_features":fref}
        info[name]=details
        binding={"cohort_path":cohort["path"],"cohort_sha256":cohort["sha256"],"feature_path":fref["path"],
                 "feature_sha256":fref["sha256"],"map_sha256":mapref["sha256"],"task":task,"stage":stage,
                 "anchor_indices":list(range(len(y)))}
        fits={};fitrefs={};headmap={};coretiming=[]
        for lam,token in c.LAMBDA_TOKENS:
            record=m.fit_regularized(z,y,ids=ids,feature_id=feature_id,ridge_lambda=lam,kappa="2",
                                     role="train",input_binding=binding,settings=protocol["heads"]["solver_settings"])
            fits[lam]=record;fitrefs[lam]=c.save(base/"fits"/name/("lambda_"+token+".json"),record)
            coretiming.append({"ridge_lambda":lam,**record["timing"],"core_timing":record["core_fit"]["timing"]})
        affine_started=time.perf_counter()
        affine=m.fit_affine(z,y,feature_id=feature_id,kappa="2",ridge_lambda=protocol["heads"]["affine_ridge_lambda"],training_binding=binding)
        affine_seconds=time.perf_counter()-affine_started;affine_ref=c.save(base/"fits"/name/"affine.json",affine)
        for cfg in [r for r in protocol["heads"]["configurations"] if r["map"]==name]:
            if cfg["recipe"]=="affine":head=affine["head"];fitted_ref=affine_ref
            else:
                fitted=fits[cfg["ridge_lambda"]];fitted_ref=fitrefs[cfg["ridge_lambda"]]
                head=m.make_extension(fitted,z,anchor_indices=list(range(len(y))),recipe=cfg["recipe"],
                        extension_weight=cfg["extension_weight"],fit_path=fitted_ref["path"],fit_sha256=fitted_ref["sha256"])
            href=c.save(base/"heads"/name/(cfg["token"]+".json"),head);headbytes=(ROOT/href["path"]).stat().st_size
            rows.append({**cfg,"family":details["family"],"scaling":details["scaling"],"dimension":details["dimension"],
                         "feature_id":feature_id,"class_id":details["class_id"],"head_id":m.head_identity(head),
                         "head_path":href["path"],"head_sha256":href["sha256"],"fit_path":fitted_ref["path"],"fit_sha256":fitted_ref["sha256"],
                         "head_payload_bytes":headbytes,"map_payload_bytes":len(payload),"deployment_payload_bytes":headbytes+len(payload),
                         "anchor_count":0 if cfg["recipe"]=="affine" else len(y),"training_only":True,"assurance":m.ASSURANCE})
        timings.append({"map":name,"regularized_fits":coretiming,"affine_fit_seconds":affine_seconds,
                        "whole_map_group_seconds":time.perf_counter()-group_started})
        progress("FIT_MAP",task=task,fit_stage=stage,completed=index+1,total=len(protocol["maps"]),map=name)
    baseline_started=time.perf_counter();baseline_rows=[];scaler=maps["raw_minmax01"].core
    SX=scaler.transform(X);yf=np.asarray([float(v) for v in y]);mean=sum(y,F(0))/len(y)
    for cfg in protocol["baselines"]["configurations"]:
        kind=cfg["kind"];token=cfg["token"]
        if kind in ("hgb","ridge"):
            if kind=="hgb":model=HistGradientBoostingRegressor(**cfg["parameters"])
            else:model=Ridge(alpha=len(y)*float(F(cfg["ridge_lambda"])),fit_intercept=True,solver="cholesky")
            model.fit(SX,yf);pref=save_binary(base/"baselines"/(token+".pkl"),pickle.dumps({"model":model,"scaler":scaler},protocol=5))
        else:
            value={"rule":"last_observed_own_target"} if kind=="persistence" else {"value":str(mean),"rule":"exact_training_mean"}
            pref=c.save(base/"baselines"/(token+".json"),value)
        baseline_rows.append({**cfg,"recipe":"baseline","baseline_kind":kind,"family":"ordinary",
            "payload_path":pref["path"],"payload_sha256":pref["sha256"],
            "deployment_payload_bytes":(ROOT/pref["path"]).stat().st_size,"class_membership_certified":False})
    manifest={"schema":"step40-fit-manifest-v1","task":task,"stage":stage,"training_cohort":cohort_ref(cohort),
              "normalization":cohort["normalization"],"training_n":len(y),"maps":info,"candidate_rows":rows,
              "baseline_rows":baseline_rows,"group_timings":timings,"representation_seconds":metadata["factory_total_seconds"],
              "baseline_fit_seconds":time.perf_counter()-baseline_started,"all_fit_data_at_or_before_held_out_origins":True,
              "selection_seal_sha256":c.sha(OUT/task/"selection_seal40.json") if stage=="final" else None}
    if [r["candidate_id"] for r in rows+baseline_rows]!=protocol["evaluation"]["prediction_ids"]:
        raise RuntimeError("Fitted candidate roster changed")
    c.save(base/"fit_manifest40.json",manifest)
    end("fit",started,task,stage,training_n=len(y),certified_heads=len(rows),baselines=len(baseline_rows))
    files=[p for p in base.rglob("*") if p.is_file()]
    seal(base/"fit_seal40.json","step40-fit-seal-v1",files,task=task,stage=stage,manifest_sha256=c.sha(base/"fit_manifest40.json"))

def load_models(task,stage):
    base=stage_dir(task,stage);manifest=c.read(base/"fit_manifest40.json");maps={};zs={};heads={};fits={};baselines={}
    for name,info in manifest["maps"].items():
        maps[name]=rep.deserialize_map((ROOT/info["map_path"]).read_bytes())
        zs[name]=np.load(ROOT/info["training_features"]["path"],allow_pickle=False)
    for row in manifest["candidate_rows"]:
        heads[row["candidate_id"]]=c.read(ROOT/row["head_path"])
        if row["recipe"]!="affine" and row["fit_path"] not in fits:fits[row["fit_path"]]=c.read(ROOT/row["fit_path"])
    for row in manifest["baseline_rows"]:
        path=ROOT/row["payload_path"]
        baselines[row["candidate_id"]]=pickle.loads(path.read_bytes()) if row["baseline_kind"] in ("hgb","ridge") else c.read(path)
    training=c.read(ROOT/manifest["training_cohort"]["path"])
    return manifest,maps,zs,heads,fits,baselines,training

def evaluate(protocol,task,stage,context,X,ids,persistence,role,dest):
    manifest,maps,zs,heads,fits,baselines,training=context
    predictions={};optimizer={};map_records={};seconds={"transform":0.,"distance":0.,"head_intervals":0.,"optimizer_intervals":0.,"baseline":0.}
    for name in protocol["maps"]:
        started=time.perf_counter();z=rep.transform_one(maps[name],X);seconds["transform"]+=time.perf_counter()-started
        fref=save_features(dest/"features"/role/(name+".npy"),z)
        started=time.perf_counter();distance=m.query_distances(zs[name],z,anchor_ids=training["ids"],query_ids=ids,feature_id=manifest["maps"][name]["feature_id"])
        seconds["distance"]+=time.perf_counter()-started;dref=c.save(dest/"distances"/role/(name+".json"),distance)
        map_records[name]={"features":fref,"distances":dref}
        for row in [r for r in manifest["candidate_rows"] if r["map"]==name]:
            cid=row["candidate_id"];started=time.perf_counter()
            pred=m.evaluate_head(heads[cid],z,ids=ids,distance=distance);predictions[cid]=pred
            seconds["head_intervals"]+=time.perf_counter()-started
            if row["recipe"]=="qp_regularized":
                started=time.perf_counter();optimizer[cid]=m.expand_optimizer_predictions(pred,heads[cid],fits[row["fit_path"]])
                seconds["optimizer_intervals"]+=time.perf_counter()-started
    started=time.perf_counter()
    for row in manifest["baseline_rows"]:
        cid=row["candidate_id"];kind=row["baseline_kind"];model=baselines[cid]
        if kind=="persistence":values=persistence
        elif kind=="mean":values=[F(model["value"])]*len(ids)
        else:values=model["model"].predict(model["scaler"].transform(X)).tolist()
        predictions[cid]=m.baseline_prediction(row["token"],values,ids)
    seconds["baseline"]+=time.perf_counter()-started
    if list(predictions)!=protocol["evaluation"]["prediction_ids"]:raise RuntimeError("Prediction roster changed")
    return {"schema":"step40-prediction-bundle-v1","task":task,"stage":stage,"role":role,"query_ids":ids,
            "predictions":predictions,"optimizer_predictions":optimizer,"maps":map_records,"timing":seconds,
            "fit_seal_sha256":c.sha(stage_dir(task,stage)/"fit_seal40.json")}

def validate(protocol,task,stage):
    if stage not in ("f1","f2","f3"):raise ValueError("Only predeclared development folds can be validated")
    started=start("validate",protocol,task,stage);cohort=d.load_validation(task,stage);base=stage_dir(task,stage)
    context=load_models(task,stage);manifest=context[0]
    bundle=evaluate(protocol,task,stage,context,cohort["X"],cohort["ids"],cohort["persistence"],"validation",base)
    pref=c.save(base/"validation_predictions40.json",bundle);scale=F(cohort["normalization"]["span"])**2;rows=[]
    for row in manifest["candidate_rows"]+manifest["baseline_rows"]:
        cid=row["candidate_id"];risk=m.score(bundle["predictions"][cid]["predictions"],cohort["y"])
        ideal=m.score(bundle["optimizer_predictions"][cid]["predictions"],cohort["y"]) if cid in bundle["optimizer_predictions"] else None
        rows.append({**row,"validation_risk":risk,"physical_validation_risk":c.scale_box(risk,scale),
                     "physical_optimizer_risk":c.scale_box(ideal,scale) if ideal else None,
                     "validation_prediction_digest":m.digest(bundle["predictions"][cid])})
    scores={"schema":"step40-validation-scores-v1","task":task,"stage":stage,"n":len(cohort["y"]),
            "normalization":cohort["normalization"],"fit_seal_sha256":bundle["fit_seal_sha256"],
            "validation_predictions_sha256":pref["sha256"],"cohort_sha256":cohort["sha256"],"candidate_rows":rows}
    c.save(base/"validation_scores40.json",scores);end("validate",started,task,stage,validation_n=len(cohort["y"]))
    frozen=c.read(base/"fit_seal40.json")["bound_files"]
    files=[p for p in base.rglob("*") if p.is_file() and c.relative(p) not in frozen]
    seal(base/"completion_seal40.json","step40-fold-completion-seal-v1",files,task=task,stage=stage,
         fit_seal_sha256=c.sha(base/"fit_seal40.json"),validation_scores_sha256=c.sha(base/"validation_scores40.json"))

def select(protocol,task):
    started=start("select",protocol,task)
    scores={}
    for stage in protocol["selection"]["folds"]:
        gate=ROOT/"audit"/task/stage/"validation_gate40.json"
        gate_record=c.read(gate)
        completion_path=stage_dir(task,stage)/"completion_seal40.json"
        if (gate_record["status"]!="PASS" or gate_record.get("task")!=task or gate_record.get("stage")!=stage
            or gate_record.get("protocol_seal_sha256")!=c.sha(ROOT/"protocol_seal40.json")
            or gate_record.get("completion_seal_sha256")!=c.sha(completion_path)):
            raise RuntimeError("Independent development gate does not bind the completed fold")
        completion=c.read(completion_path)
        for relative,wanted in completion["bound_files"].items():
            if c.sha(ROOT/relative)!=wanted:raise RuntimeError("Completed development artifact changed")
        scores[stage]=c.read(stage_dir(task,stage)/"validation_scores40.json")
    rows=[]
    for cid in protocol["evaluation"]["prediction_ids"]:
        folds={};last=None
        for stage,record in scores.items():
            row=next(r for r in record["candidate_rows"] if r["candidate_id"]==cid);last=row
            folds[stage]={"n":record["n"],"physical_risk":row["physical_validation_risk"],
                          "physical_optimizer_risk":row["physical_optimizer_risk"],"payload_bytes":row["deployment_payload_bytes"]}
        keep={k:last.get(k) for k in ("candidate_id","map","token","recipe","ridge_lambda","extension_weight","family","scaling","baseline_kind")}
        merged=c.pooled([{"n":r["n"],"risk":r["physical_risk"]} for r in folds.values()])
        refined=c.pooled([{"n":r["n"],"risk":r["physical_optimizer_risk"]} for r in folds.values()]) if last["recipe"]=="qp_regularized" else None
        rows.append({**keep,"folds":folds,"pooled_physical_risk":merged,"last_fold_physical_risk":folds["f3"]["physical_risk"],
                     "pooled_optimizer_risk":refined,"selection_payload_bytes":folds["f3"]["payload_bytes"]})
    selections={label:c.select(rows,label) for label in protocol["selection"]["selectors"]}
    decision_started=time.perf_counter()
    selected_id=selections["joint_rolling"]["candidate_id"]
    certified_rows=c.selector_pool(rows,"joint_rolling")
    mixed=decision.mixed_decision_certificate(certified_rows,selected_id,expected_maps=protocol["maps"])
    decision_seconds=time.perf_counter()-decision_started
    record={"schema":"step40-development-selection-v1","task":task,"candidate_rows":rows,"selections":selections,
            "mixed_decision_certificate":mixed,"decision_certificate_seconds":decision_seconds,"development_n":sum(s["n"] for s in scores.values()),
            "fold_score_sha256":{stage:c.sha(stage_dir(task,stage)/"validation_scores40.json") for stage in scores},
            "selection_uses_development_only":True,"test_targets_accessed":0}
    ref=c.save(OUT/task/"selection40.json",record);end("select",started,task,selected_ids={k:v["candidate_id"] for k,v in selections.items()})
    seal(OUT/task/"selection_seal40.json","step40-selection-seal-v1",[ROOT/ref["path"]],task=task,
         selection_sha256=ref["sha256"],selected_ids={k:v["candidate_id"] for k,v in selections.items()})

def test(protocol,task):
    started=start("test",protocol,task);base=OUT/task/"test";context=load_models(task,"final")
    manifest=context[0];selection=c.read(OUT/task/"selection40.json");queries=d.test_queries(task)
    ids=protocol["evaluation"]["prediction_ids"];all_preds={cid:[] for cid in ids}
    all_ideal={cid:[] for cid in ids if "::qp_" in cid};ys=[];raw=[];queryrecords=[];timing=[]
    for index,query in enumerate(queries):
        history=d.load_test_history(task,query)
        persistence=history["persistence"] if isinstance(history["persistence"],list) else [history["persistence"]]
        bundle=evaluate(protocol,task,"final",context,history["X"],[query["id"]],persistence,"q"+str(index).zfill(3),base)
        record={**bundle,"schema":"step40-sealed-test-predictions-v1","query_id":query["id"],
                "origin_timestamp":query["origin_timestamp"],"target_timestamp":query["target_timestamp"],
                "model_seal_sha256":c.sha(OUT/task/"final/fit_seal40.json"),
                "history_receipt_path":history["history_receipt_path"],"history_receipt_sha256":history["history_receipt_sha256"],
                "sealed_before_current_target_reveal":True,"future_target_used_in_prediction":False}
        path=base/"predictions"/("q"+str(index).zfill(3)+".json");ref=c.save(path,record)
        d.event("test_prediction_receipt_written",task=task,query_id=query["id"],prediction_path=ref["path"],prediction_sha256=ref["sha256"],prediction_count=len(ids))
        revealed=d.reveal_test_target(task,query,ref["path"],ref["sha256"])
        ys.append(F(revealed["y"]));raw.append(revealed["raw_target"])
        for cid in ids:all_preds[cid].append(bundle["predictions"][cid]["predictions"][0])
        for cid in all_ideal:all_ideal[cid].append(bundle["optimizer_predictions"][cid]["predictions"][0])
        queryrecords.append({**query,"prediction_path":ref["path"],"prediction_sha256":ref["sha256"],
                             "history_receipt_path":history["history_receipt_path"],"history_receipt_sha256":history["history_receipt_sha256"],
                             "target_reveal":m.canonical(revealed)})
        timing.append(bundle["timing"])
        if index%10==0 or index+1==len(queries):progress("TEST_QUERY",task=task,completed=index+1,total=len(queries))
    cohort={"schema":"step40-completed-test-cohort-v1","task":task,"n":len(ys),"ids":[q["id"] for q in queries],
            "target_timestamps":[q["target_timestamp"] for q in queries],"origin_timestamps":[q["origin_timestamp"] for q in queries],
            "y":list(map(str,ys)),"raw_targets":raw,"queries":queryrecords,"sequential_prediction_then_label":True}
    c.save(base/"cohorts/test.json",cohort)
    norm=manifest["normalization"];scale=F(norm["span"])**2;rows=[]
    for row in manifest["candidate_rows"]+manifest["baseline_rows"]:
        cid=row["candidate_id"];risk=m.score(all_preds[cid],ys)
        ideal=m.score(all_ideal[cid],ys) if cid in all_ideal else None;monthly={}
        for month in protocol["evaluation"]["months_by_task"][task]:
            indices=[i for i,q in enumerate(queries) if q["target_timestamp"].startswith(month)]
            monthrisk=m.score([all_preds[cid][i] for i in indices],[ys[i] for i in indices])
            monthly[month]={"n":len(indices),"risk":monthrisk,"physical_risk":c.scale_box(monthrisk,scale)}
        rows.append({**row,"test_risk":risk,"physical_test_risk":c.scale_box(risk,scale),
                     "physical_optimizer_risk":c.scale_box(ideal,scale) if ideal else None,"monthly":monthly,
                     "test_predictions_digest":m.digest(all_preds[cid]),"optimizer_predictions_digest":m.digest(all_ideal[cid]) if ideal else None})
    lookup={r["candidate_id"]:r for r in rows};selected={label:lookup[value["candidate_id"]] for label,value in selection["selections"].items()}
    gate=c.practical_gate(selected["joint_rolling"],selected["qp_rolling"],selected["ordinary_rolling"],selected["affine_rolling"])
    decision_started=time.perf_counter()
    paired=decision.paired_head_comparisons([r for r in rows if r["recipe"]!="baseline"],
        expected_maps=protocol["maps"],operational_risk_key="physical_test_risk",
        qp_optimizer_risk_key="physical_optimizer_risk",cohort_role="test")
    paired_seconds=time.perf_counter()-decision_started
    primary={"schema":"step40-task-primary-result-v1","task":task,"candidate_rows":rows,"selected":selected,"primary_gate":gate,
             "test_n":len(ys),"normalization":norm,"query_timing":timing,"forecast_seconds":time.perf_counter()-started,
             "model_seal_sha256":c.sha(OUT/task/"final/fit_seal40.json"),"selection_seal_sha256":c.sha(OUT/task/"selection_seal40.json"),
             "pretest_gate_sha256":c.sha(ROOT/"audit"/task/"pretest_gate40.json"),
             "development_decision_certificate":selection["mixed_decision_certificate"],
             "test_same_map_comparisons":paired,"test_pair_certificate_seconds":paired_seconds,
             "test_reselection":False,"assessment_oracle_fits":0,"future_response_coverage_claim":False,"population_risk_claim":False}
    c.save(base/"primary_result40.json",primary)
    result={"schema":"step40-completed-task-result-v1","status":"COMPLETE_PENDING_INDEPENDENT_AUDIT","task":task,
            "protocol_sha256":c.sha(ROOT/"protocol40.json"),"protocol_seal_sha256":c.sha(ROOT/"protocol_seal40.json"),
            "primary":primary,"training_n":manifest["training_n"],"development_n":selection["development_n"],
            "primary_result_sha256":c.sha(base/"primary_result40.json"),"new_scientific_passes":1}
    c.save(base/"result40.json",result);end("test",started,task,test_n=len(ys),primary_gate=gate,result_sha256=c.sha(base/"result40.json"))

def aggregate(protocol):
    started=start("aggregate",protocol);tasks={};pass_all=True
    for task in protocol["tasks"]:
        auditpath=ROOT/"audit"/task/"postrun_audit40.json";audit=c.read(auditpath)
        resultpath=OUT/task/"test/result40.json";result=c.read(resultpath)
        if audit["status"]!="PASS" or audit["result_sha256"]!=c.sha(resultpath):raise RuntimeError("Task result lacks matching independent PASS")
        tasks[task]={"result_path":c.relative(resultpath),"result_sha256":c.sha(resultpath),"audit_path":c.relative(auditpath),"audit_sha256":c.sha(auditpath),
                     "primary_gate":result["primary"]["primary_gate"],"test_n":result["primary"]["test_n"],
                     "selected":result["primary"]["selected"],"development_decision_certificate":result["primary"]["development_decision_certificate"],
                     "test_same_map_comparisons":result["primary"]["test_same_map_comparisons"]}
        pass_all=pass_all and result["primary"]["primary_gate"]["numerical_criterion_pass"]
    value={"schema":"step40-two-domain-result-v1","step":40,"status":"COMPLETE","tasks":tasks,
           "protocol_sha256":c.sha(ROOT/"protocol40.json"),"protocol_seal_sha256":c.sha(ROOT/"protocol_seal40.json"),
           "all_task_independent_audits_pass":True,"practical_transfer_success":pass_all,
           "primary_status":"ESTABLISHED_ON_BOTH_FIXED_ASSESSMENTS" if pass_all else "NOT_ESTABLISHED",
           "domain_count":len(tasks),"test_reselection":False,"assessment_oracle_fits":0,"population_risk_claim":False,
           "future_response_coverage_claim":False,"reserved_patients_accessed":0}
    c.save(OUT/"result40.json",value);end("aggregate",started,practical_transfer_success=pass_all,result_sha256=c.sha(OUT/"result40.json"))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("fit","validate","select","test","aggregate"))
    parser.add_argument("--task",choices=d.task_ids());parser.add_argument("--stage",choices=("f1","f2","f3","final"))
    args=parser.parse_args();protocol=d.verify_protocol_seal()
    try:
        if args.phase in ("fit","validate"):
            if args.task is None or args.stage is None:raise ValueError("Task and stage required")
            {"fit":fit,"validate":validate}[args.phase](protocol,args.task,args.stage)
        elif args.phase in ("select","test"):
            if args.task is None:raise ValueError("Task required")
            {"select":select,"test":test}[args.phase](protocol,args.task)
        else:aggregate(protocol)
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        failure={"schema":"step40-retained-phase-failure-v1","phase":args.phase,"task":args.task,"stage":args.stage,
                 "status":"STOPPED_ERROR","type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),"no_rerun":True}
        dest=OUT/(args.task or "global")/(args.stage or args.phase)/(args.phase+"_failure40.json")
        if not dest.exists():c.save(dest,failure)
        d.event("phase_stopped_error",phase=args.phase,task=args.task,stage=args.stage,error_type=type(exc).__name__,message=str(exc))
        progress("STOPPED_ERROR",**failure);raise SystemExit(2)

if __name__=="__main__":main()
