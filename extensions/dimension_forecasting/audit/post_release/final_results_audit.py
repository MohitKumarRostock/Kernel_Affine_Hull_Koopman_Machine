#!/usr/bin/env python3
"""Strict complete released-study audit, independent of fitting/scoring code."""
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
from verify_task_artifacts import Audit,read,sha,arrays,array_hash,verify_task
from verify_data_construction import audit_data
from independent_tables import score_task,outcome_tables,diagnostic_tables,derived_tables,table_comparison,TABLE_KEYS


def main(root):
    output=root/"audit/post_release";output.mkdir(exist_ok=True)
    receipt=output/"final_results_audit.json"
    if receipt.exists():raise FileExistsError("Preserve the prior independent audit attempt: "+str(receipt))
    audit=Audit()
    result={"status":"failed","fixture_plumbing_only":False,"started_utc":datetime.now(timezone.utc).isoformat(),
            "audit_sources":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}}
    try:
        analysis=read(root/"analysis/analysis_summary.json")
        audit.require(analysis["status"]=="completed" and analysis["task_count"]==192,
                      "complete frozen primary analysis must exist before final audit")
        data=audit_data(root,released=True)
        audit.require(data["status"]=="passed" and data["counts"]["D_target_joins"]==192,
                      "complete independently reconstructed released target join")
        (output/"released_data_checks.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
        cfg=read(root/"config.json");abc_manifest=read(root/"prepared/abc_manifest.json")
        terminal=read(root/"prepared/d_manifest.json")
        d_tasks={task["task_id"]:task for task in terminal["tasks"]}
        horizons=[];losses=[];screens=[];diagnostics=defaultdict(list)
        for index,task in enumerate(sorted(abc_manifest["tasks"],key=lambda t:t["task_id"]),1):
            tid=task["task_id"];d=d_tasks[tid]
            fdir=root/"results"/tid;pdir=root/"predictions"/tid
            checked=verify_task(cfg,root/task["input"],fdir,pdir,root/d["query"],audit=audit,fixture=False)
            audit.require(checked["task"]==task,"fitted task is exact prepared input record",tid)
            abc=arrays(root/task["input"]);target=arrays(root/d["target"])["D_Y"]
            prediction=checked["predictions"];fit=read(fdir/"fit_summary.json");pre=read(fdir/"pre_c_selection.json")
            query=read(pdir/"predict_summary.json")
            h,t,s=score_task(task,target,prediction,checked["scale"],fit)
            horizons.extend(h);losses.extend(t);screens.extend(s)
            table=diagnostic_tables(task,fit,pre,abc,arrays(fdir/"validation_predictions.npz"),
                                   arrays(fdir/"calibration_predictions.npz"),arrays(fdir/"basis_diagnostics.npz"),
                                   query,prediction,cfg)
            for name,rows in table.items():diagnostics[name].extend(rows)
            audit.counts["terminal_prediction_arrays_scored"]+=len(prediction)
            if index%24==0:print(json.dumps({"independently_verified_tasks":index,"expected":192}),flush=True)
        tables=outcome_tables(horizons,losses)
        tables.update(diagnostics);tables["stability_screens.csv"]=screens;derived_tables(tables)
        audit.require(set(analysis["tables"])==set(tables)==set(TABLE_KEYS),"complete twenty-seven-table roster")
        for name,rows in tables.items():
            audit.require(analysis["tables"][name]["rows"]==len(rows)
                          and analysis["tables"][name]["sha256"]==sha(root/"analysis"/name),
                          "canonical analysis table count and exact bytes",name)
        comparisons=table_comparison(root/"analysis",tables,audit)
        primary=tables["primary_contrasts.csv"]
        selected_screens=[r for r in screens if r["view"]=="context_selected"]
        established=sum(bool(r["criterion_established"]) for r in primary)
        stable=all(r["screen_pass"] for r in selected_screens)
        overall=established==6 and stable
        audit.require(analysis["primary_comparisons_established"]==established
                      and analysis["statistical_criterion_established"]==(established==6)
                      and analysis["maximum_error_screens_passed"]==stable
                      and analysis["overall_criterion_established"]==overall,
                      "reported adoption criterion follows all six tests and stability screen")
        audit.close(analysis["maximum_selected_error_ratio"],max(r["maximum_error_ratio"] for r in selected_screens),
                    "maximum absolute error ratio is independently reconstructed")
        audit.require(sorted(analysis["failed_stability_tasks"])==sorted(r["task_id"] for r in selected_screens if not r["screen_pass"]),
                      "no stability failure omitted")
        result.update(status="passed",counts=dict(audit.counts),data_counts=data["counts"],tables=comparisons,
                      primary_contrasts=primary,overall_criterion_established=overall,
                      maximum_selected_error_ratio=max(r["maximum_error_ratio"] for r in selected_screens),
                      findings={"geometry_rank_records":len(tables["local_dimensions.csv"]),
                                "zero_rank_records":sum(r["effective_rank"]==0 for r in tables["local_dimensions.csv"]),
                                "feature_fallback_row_records":sum(r["fallback_rows"] for r in tables["feature_checks.csv"]),
                                "factor_failure_records":sum(r["factor_failure_models"] for r in tables["feature_checks.csv"]),
                                "coefficient_bound_violations":sum(r["coefficient_bound_violations"] for r in tables["bound_checks.csv"]),
                                "prediction_bound_violations":sum(r["D_correction_bound_violations"] or 0 for r in tables["bound_checks.csv"])},
                      seal_sha256={n:sha(root/n) for n in ("freeze.json","FIT_LOCK.json","PREDICTION_LOCK.json","TARGET_RELEASE.json")},
                      analysis_summary_sha256=sha(root/"analysis/analysis_summary.json"),
                      scope="Every task, retained target, 162 forecast arrays per task, native basis, B/C decision, all 27 exported tables and six primary tests; no fitting or primary-analysis modules imported.",
                      limitations=["Fresh repetitions of the declared synthetic laws only.",
                                   "The paired-t family is approximate and is not a finite-sample risk certificate.",
                                   "Resource timings are replayed as recorded shared-host measurements rather than benchmarked anew."])
    except Exception as error:
        result.update(error=repr(error),traceback=traceback.format_exc(),counts=dict(audit.counts))
    result["completed_utc"]=datetime.now(timezone.utc).isoformat()
    receipt.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"counts":result.get("counts"),"error":result.get("error"),"receipt":str(receipt)}))
    return 0 if result["status"]=="passed" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    raise SystemExit(main(parser.parse_args().root.resolve()))
