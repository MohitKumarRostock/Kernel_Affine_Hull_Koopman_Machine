#!/usr/bin/env python3
"""Independent finite design, source, random-stream and seal gate."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from independent_scores import Audit, read, sha


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def main(root):
    sys.path.insert(0,str(root))
    import common
    import study
    a=Audit();cfg=read(root/"config.json")
    expected={"master_seed":2026110104,"histories":[4,12],"horizons":[1,4,12],
              "n_train":128,"n_validation":512,"n_calibration":4096,"n_test":2048,
              "episode_length":72,"origin_stride":4,"cluster_counts":[2,4,8,16],
              "caps":[1,2,4,8,12,20],"policies":["cap2","native20","selected_cap"],
              "input_scaling":"standard","kappa":4.,"alpha":1e-4,"omega":16.,
              "max_geometry_inputs":1600,"max_train_per_cluster":24,
              "solver_tolerance":1e-7,"solver_max_iterations":20000,
              "delta":.05,"risk_delta_fraction":.8,"rmse_tolerances":[.2,.35,.5],
              "expected_repetitions":26,"development_candidates_per_repetition":70,
              "evaluated_models_per_repetition":24,"risk_tails_per_repetition":144,
              "improvement_tails_per_repetition":36,"campaign_risk_tails":3744,
              "campaign_improvement_tails":936,"campaign_total_tails":4680,
              "full_prediction_audit_repetition":0}
    for key,value in expected.items():
        a.require(cfg[key]==value,"fixed approved design",key)
    systems=[("harmonic",5),("quasiperiodic",5),("nonlinear_ar",5),("logistic",5),("vanderpol",3),("duffing",3)]
    a.require([(s["name"],s["repetitions"]) for s in cfg["systems"]]==systems,"six-law twenty-six-run roster")
    tail=cfg["delta"]/4680
    a.close([cfg["delta"]*cfg["risk_delta_fraction"]/3744,cfg["delta"]*(1-cfg["risk_delta_fraction"])/936],
            [tail,tail],"correct one-campaign split allocation")
    a.close(4680*tail,.05,"full designated-tail budget")
    jobs=common.schedule(cfg);oracle=[];split_seeds=[];geometry_seeds=[];block_seeds=[]
    for si,(system,repetitions) in enumerate(systems):
        for rep in range(repetitions):
            streams={split:int(np.random.SeedSequence([2026110104,si,rep,index,0]).generate_state(1,dtype=np.uint64)[0])
                     for index,split in enumerate(("train","validation","calibration","test"),1)}
            geometries={str(h):int(np.random.SeedSequence([2026110104,si,rep,10,h]).generate_state(1,dtype=np.uint64)[0])%(2**31)
                        for h in (4,12)}
            oracle.append({"id":f"{system}_r{rep:02d}","system":system,"system_index":si,"repetition":rep,
                           "streams":streams,"geometry_seeds":geometries,"full_prediction_audit":rep==0})
            split_seeds.extend(streams.values());geometry_seeds.extend(geometries.values())
            block_seeds.extend(v+104729*j for v in geometries.values() for j in (1,2,3,4))
    a.require(jobs==oracle,"independently derived complete randomization schedule")
    a.require(len(set(split_seeds))==104 and len(set(geometry_seeds+block_seeds))==260,
              "no collisions among intended-distinct split and geometry/block streams")
    a.require(not(set(split_seeds)&set(geometry_seeds+block_seeds)),"sampling and geometry stream IDs differ")
    # Read only declared historical configurations, never results/outcomes.
    previous=set();prior_records=[]
    prior_root=root.parent/"certified_forecasting"
    for path in sorted(prior_root.rglob("*v1.json")):
        record=read(path)
        if not isinstance(record,dict) or "seeds" not in record:
            continue
        seeds=record["seeds"]
        for value in seeds:
            value=int(value);previous.add(value)
            previous.update(value*100+offset for offset in (11,23,47,89))
            for h in record.get("histories",(4,12)):
                geometry=value+int(h);previous.add(geometry)
                previous.update(geometry+104729*j for j in (1,2,3,4))
        prior_records.append({"path":str(path.relative_to(root.parent)),"sha256":sha(path),"seeds":seeds})
    a.require(previous and not(previous&set(split_seeds+geometry_seeds+block_seeds)),
              "no reused stream IDs from archived risk configurations")
    a.require(sum(j["full_prediction_audit"] for j in jobs)==6,"six full-array audit jobs fixed before outcomes")
    origins=np.arange(11,60,4)
    a.equal(common.origins(cfg),origins,"exact common original forecast origins")
    for h in (4,12):
        history=origins[:,None]-np.arange(h-1,-1,-1)
        targets=origins[:,None]+np.array([1,4,12])
        a.require(np.all(history<=origins[:,None]) and history.min()>=0 and np.all(targets>origins[:,None])
                  and targets.max()==71,"causal histories and complete common future support",h)
    manifest=read(root/"runtime_sources.json")
    step1=root.parents[2]/"kahm_dimension_policy_step1"
    for name,record in manifest["files"].items():
        a.require(sha(root/name)==record["sha256"],"copied verified runtime identity",name)
        source=step1/record["source"]
        if source.is_file():
            a.require(sha(source)==record["sha256"],"runtime identical to retained Step1 source",name)
    import score_models
    for upper,lower,wanted in ((.01,0.,"accepted"),(.9,.6,"excluded"),(.9,0.,"unresolved"),(.01,.6,"conflicting_bounds")):
        endpoints=[{"mse_upper":upper,"mse_lower":lower} for _ in range(3)]
        a.require(score_models.tolerance_decision(endpoints,.5)==wanted,"all finite endpoint classifications are retained",wanted)
    source_map=common.sources()
    a.require(source_map["analyze.py"]==sha(root/"analyze.py") and "requirements.txt" in source_map,
              "all primary analysis and dependency declarations covered by source freeze")
    for name in ("audit/manufactured_score_checks.json","verification/manufactured_v1/verification.json"):
        r=read(root/name);a.require(r.get("passed") is True,"manufactured numerical gate passed",name)
    tamper_checks=[]
    with tempfile.TemporaryDirectory(prefix="kahm_risk_seal_audit_") as scratch:
        scratch=Path(scratch);output=scratch/"manufactured_structural_records"
        review=scratch/"manufactured_preflight.json"
        write(review,{"passed":True,"source_sha256":source_map,
                      "scope":"Structural-only fake-file gate; not authorization for production observations"})
        study.freeze(output,review)
        counter=[0]
        original_execute=study.execute_job
        def forbidden(*args,**kwargs):
            counter[0]+=1
            raise AssertionError("A forbidden transition reached a worker")
        study.execute_job=forbidden
        try:
            for stage in ("calibration","test"):
                try: study.run_stage(output,stage,1)
                except (RuntimeError,FileNotFoundError): pass
                else: raise AssertionError("Premature transition was not rejected")
                a.require(counter[0]==0,"premature held-out transition stops before any worker",stage)
                tamper_checks.append("premature_"+stage)
        finally: study.execute_job=original_execute
        for stage in ("development","calibration","test"):
            for job in jobs:
                directory=output/"jobs"/job["id"];directory.mkdir(exist_ok=True)
                for name in common.stage_job_files(stage,job):
                    (directory/name).write_bytes(("Manufactured structural file only: "+stage+"/"+job["id"]+"/"+name).encode())
            study.seal(output,stage,jobs)
            common.verify_seal(output,stage)
            a.counts["complete_structural_seals"]+=1
        path=output/"calibration_seal.json";original=path.read_bytes();record=json.loads(original)
        cases={}
        changed=copy.deepcopy(record);changed["files"]={};cases["empty_roster"]=changed
        changed=copy.deepcopy(record);changed["files"].pop(next(k for k in changed["files"] if k.endswith("calibration_predictions.npz")))
        cases["missing_prespecified_full_array"]=changed
        for key,value in (("stage","test"),("n_jobs",25),("freeze_sha256","0"*64),("preceding_seal_sha256","0"*64)):
            changed=copy.deepcopy(record);changed[key]=value;cases["rebound_"+key]=changed
        for label,changed in cases.items():
            write(path,changed)
            try: common.verify_seal(output,"calibration")
            except RuntimeError: a.counts["rejected_seal_tampers"]+=1
            else: raise AssertionError("Tampered seal accepted: "+label)
            path.write_bytes(original);tamper_checks.append(label)
        changed_path=output/next(iter(record["files"]));payload=changed_path.read_bytes()
        changed_path.write_bytes(payload+b" tampered")
        try: common.verify_seal(output,"test")
        except RuntimeError: a.counts["rejected_seal_tampers"]+=1
        else: raise AssertionError("Nested stage content mutation accepted")
        changed_path.write_bytes(payload);tamper_checks.append("nested_calibration_content_change")
        common.verify_seal(output,"test")
    result={"schema":"kahm-risk-independent-prefreeze-v1","status":"passed","passed":True,
            "completed_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":source_map,
            "audit_source_sha256":{str(p.relative_to(root)):sha(p) for p in (root/"audit").glob("*.py")},
            "supporting_receipt_sha256":sha(root/"audit/manufactured_score_checks.json"),
            "counts":dict(a.counts),"randomization":{"split_seeds":104,"geometry_and_block_seeds":260,
                "prior_unique_seed_ids":len(previous),"collisions":0,"prior_configs":prior_records},
            "seal_tamper_checks":tamper_checks,"production_data_generated":False,
            "source_review":[
                "All development candidates fit training only and best-prefix validation selects one complete bank per arm; its four retained prefixes remain nested.",
                "The cap2 risk comparator uses the original native two-dimensional construction, with no forecasting-specific d-1 reserve rule.",
                "Original RBF/affine geometry remains distinct; no same-basis claim is made for these risk comparators.",
                "All 26 developments seal before calibration generation; all calibration endpoints seal before test generation.",
                "One fixed campaign-wide 0.05 event covers 3744 risk tails and 936 improvement tails, including equal model aliases without retrospective denominator reduction.",
                "Actual coefficient arrays establish the frozen decoder norm budgets; losses average origins within independent episodes and witnesses use disjoint episode pairs.",
                "Independent finite test losses are utility/consistency evidence; unknown class-optimal risk is not replaced by test MSE in a coverage claim.",
                "Full prediction-to-loss joins cover six prespecified runs; all 26 have complete score-level certificate reconstruction.",
                "Analyzer exact-source identity is checked against freeze; CSV exports are atomic with row and cell readback.",
                "Finite-sample endpoint conflicts are retained as conflicting_bounds, never omitted by an outcome-dependent abort; all four decision states directly tested."
            ],"remaining_blockers":[]}
    write(root/"audit/prefreeze_review.json",result)
    print(json.dumps({"status":"passed","counts":dict(a.counts),"receipt":"audit/prefreeze_review.json"}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root.resolve())
