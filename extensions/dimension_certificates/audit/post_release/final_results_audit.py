#!/usr/bin/env python3
"""Complete independent released-risk audit; no model/runtime imports."""
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[name]="1"
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
import traceback
import numpy as np
from independent_scores import Audit,read,sha,verify_job
from independent_tables import job_tables,aggregate_tables,KEYS
import table_comparison


def file_map(base,mapping,audit,label):
    audit.require(bool(mapping),"nonempty sealed file map",label)
    for name,digest in mapping.items():
        path=(base/name).resolve()
        audit.require(not Path(name).is_absolute() and path.is_relative_to(base.resolve()) and path.is_file()
                      and sha(path)==digest,"exact confined sealed artifact",[label,name])


def gates(root,run,audit):
    frozen=read(run/"freeze.json")
    audit.require(set(frozen["files"])=={"config.json","PROTOCOL.md","schedule.json","manifest.json","preflight.json"},
                  "complete frozen-input record")
    file_map(run,frozen["files"],audit,"freeze")
    file_map(root,frozen["source_sha256"],audit,"scientific_source")
    cfg=read(run/"config.json");schedule=read(run/"schedule.json");manifest=read(run/"manifest.json")
    pre=read(run/"preflight.json")
    audit.require(pre["passed"] is True and pre["status"]=="passed" and pre["source_sha256"]==frozen["source_sha256"],
                  "final exact-source independent preflight")
    audit.require(cfg["master_seed"]==2026110104 and cfg["expected_repetitions"]==26
                  and cfg["campaign_total_tails"]==4680 and cfg["campaign_risk_tails"]==3744
                  and cfg["campaign_improvement_tails"]==936 and cfg["delta"]==.05,
                  "fixed complete campaign design")
    expected=[]
    for si,system in enumerate(cfg["systems"]):
        for rep in range(system["repetitions"]):
            streams={split:int(np.random.SeedSequence([2026110104,si,rep,domain,0]).generate_state(1,dtype=np.uint64)[0])
                     for domain,split in enumerate(("train","validation","calibration","test"),1)}
            geom={str(h):int(np.random.SeedSequence([2026110104,si,rep,10,h]).generate_state(1,dtype=np.uint64)[0])%(2**31)
                  for h in (4,12)}
            expected.append({"id":f"{system['name']}_r{rep:02d}","system":system["name"],"system_index":si,
                             "repetition":rep,"streams":streams,"geometry_seeds":geom,"full_prediction_audit":rep==0})
    audit.require(len(expected)==26 and schedule==expected,"independent full fresh-seed schedule reconstruction")
    stagefiles={"development":("development.json","fitted_models.pkl","coefficients.npz","validation_scores.npz"),
                "calibration":("calibration.json","calibration_scores.npz"),"test":("test.json","test_scores.npz")}
    previous=None;before=datetime.fromisoformat(manifest["frozen_utc"])
    seals={}
    for stage,base_files in stagefiles.items():
        seal=seals[stage]=read(run/(stage+"_seal.json"))
        roster=set()
        for job in schedule:
            files=list(base_files)+[stage+"_started.json",stage+"_complete.json"]
            if stage!="development" and job["full_prediction_audit"]:files.append(stage+"_predictions.npz")
            roster.update(f"jobs/{job['id']}/{name}" for name in files)
        audit.require(seal["stage"]==stage and seal["n_jobs"]==26 and set(seal["files"])==roster
                      and seal["freeze_sha256"]==sha(run/"freeze.json") and seal["preceding_seal_sha256"]==previous,
                      "complete stage and predecessor identity",stage)
        file_map(run,seal["files"],audit,stage)
        sealed=datetime.fromisoformat(seal["sealed_utc"])
        audit.require(sealed>before,"global stage chronology",stage)
        for job in schedule:
            folder=run/"jobs"/job["id"]
            start=read(folder/(stage+"_started.json"));record=read(folder/(stage+".json"))
            audit.require(start["job"]==job["id"] and record["job"]==job,
                          "exact job identity at every stage",[stage,job["id"]])
            begun=datetime.fromisoformat(start["started_utc"]);completed=datetime.fromisoformat(record["completed_utc"])
            audit.require(before<=begun<=completed<=sealed,"every job begins only after prior global seal",[stage,job["id"]])
            audit.require(not (folder/(stage+"_failure.json")).exists(),"no omitted failed scheduled run",[stage,job["id"]])
        before=sealed;previous=sha(run/(stage+"_seal.json"))
    audit.require({p.name for p in (run/"jobs").iterdir() if p.is_dir()}=={j["id"] for j in schedule},
                  "no extra or missing production jobs")
    summary=read(run/"analysis/summary.json")
    audit.require(summary["verification"]["passed"] is True and summary["verification"]["repetitions"]==26
                  and summary["freeze_sha256"]==sha(run/"freeze.json") and summary["test_seal_sha256"]==sha(run/"test_seal.json")
                  and summary["analysis_source_sha256"]==frozen["source_sha256"]["analyze.py"],
                  "complete exact-source primary analysis after final test seal")
    file_map(run/"analysis",read(run/"analysis/analysis_files.json"),audit,"analysis_exports")
    return cfg,schedule,summary


def main(root,run):
    output=root/"audit/post_release";output.mkdir(exist_ok=True)
    receipt=output/"final_results_audit.json"
    if receipt.exists():raise FileExistsError("Preserve any prior independent audit result: "+str(receipt))
    audit=Audit();result={"status":"failed","passed":False,"fixture_plumbing_only":False,
                          "started_utc":datetime.now(timezone.utc).isoformat(),
                          "audit_source_sha256":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}}
    try:
        cfg,jobs,summary=gates(root,run,audit)
        tables=defaultdict(list)
        for index,job in enumerate(jobs,1):
            folder=run/"jobs"/job["id"]
            record=verify_job(folder,cfg,fixture=False,audit=audit)
            audit.require(record["job"]==job,"scored job matches frozen schedule")
            for name,rows in job_tables(folder,cfg).items():tables[name].extend(rows)
            if index%5==0 or index==26:print(json.dumps({"independently_verified_risk_runs":index,"expected":26}),flush=True)
        aggregate_tables(tables,cfg)
        table_comparison.TABLE_KEYS=KEYS
        compared=table_comparison.table_comparison(run/"analysis",dict(tables),audit)
        audit.require(audit.counts["risk_endpoints"]==3744 and audit.counts["improvement_endpoints"]==936
                      and audit.counts["validation_candidates"]==1820 and audit.counts["full_prediction_loss_joins"]==288,
                      "complete prespecified score/full-array audit scope")
        diagnostics=summary["certificate_diagnostics"]
        checks={"risk_lower_endpoints":len(tables["endpoints.csv"]),
                "zero_lower_endpoints":sum(r["zero_lower"] for r in tables["endpoints.csv"]),
                "upper_endpoints_at_bound":sum(r["upper_reaches_loss_bound"] for r in tables["endpoints.csv"]),
                "test_mean_below_lower":sum(r["test_mean_below_lower"] for r in tables["endpoints.csv"]),
                "test_mean_above_upper":sum(r["test_mean_above_upper"] for r in tables["endpoints.csv"]),
                "positive_paired_improvement_endpoints":sum(r["certified_positive_improvement"] for r in tables["paired_contrasts.csv"]),
                "conflicting_model_tolerance_decisions":sum(r["decision"]=="conflicting_bounds" for r in tables["decisions.csv"])}
        for key,value in checks.items():audit.require(diagnostics[key]==value,"independent scientific summary count",key)
        result.update(status="passed",passed=True,counts=dict(audit.counts),tables=compared,
                      findings={"certificate_diagnostics":checks,"policy_comparison_summary":tables["policy_comparison_summary.csv"],
                                "selected_policy_summary":tables["selected_policy_summary.csv"],
                                "full_array_jobs":[j["id"] for j in jobs if j["full_prediction_audit"]]},
                      seal_sha256={n:sha(run/n) for n in ("freeze.json","development_seal.json","calibration_seal.json","test_seal.json")},
                      primary_analysis_summary_sha256=sha(run/"analysis/summary.json"),
                      confidence={"campaign_delta":.05,"designated_risk_tails":3744,"designated_improvement_tails":936,
                                  "delta_per_tail":.05/4680},
                      scope="All 26 jobs, 1820 development score candidates, 624 evaluated aliases, 4680 designated endpoints and all nine exported tables; 288 full prediction-to-loss alias joins across the six prespecified repetition-zero jobs.",
                      limitations=["The other 20 runs have complete score-level rather than full forecast reconstruction.",
                                   "Retained class-optimal risks are unknown; test interval containment is not an empirical coverage estimate.",
                                   "The campaign-wide allocation differs from the older per-repetition study; cross-study widths cannot isolate dimension effects."])
    except Exception as error:
        result.update(error=repr(error),traceback=traceback.format_exc(),counts=dict(audit.counts))
    result["completed_utc"]=datetime.now(timezone.utc).isoformat()
    receipt.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"counts":result.get("counts"),"error":result.get("error"),"receipt":str(receipt)}))
    return 0 if result["status"]=="passed" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(main(args.root.resolve(),args.run.resolve()))
