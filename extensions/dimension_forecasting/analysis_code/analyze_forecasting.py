#!/usr/bin/env python3
"""Recompute the complete study from sealed arrays without loading model states."""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
import sys
sys.dont_write_bytecode=True
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from analysis_core import (LAWS, NONLINEAR, HORIZONS, METHODS, GEOMETRY, CANDIDATES, NATIVE,
    CONTEXTS, COMPARATORS, VIEWS, MODES, META, digest, array_hash, require, means,
    score_task, all_outcome_tables, export_csv, write_json)
from diagnostics import diagnose

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def confined(root,relative):
    require(isinstance(relative,str) and not Path(relative).is_absolute(),"Absolute artifact path")
    path=(root/relative).resolve()
    require(path.is_relative_to(root) and path.is_file(),"Missing/unconfined file: "+relative)
    return path


def arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def validate_config(c):
    require(c["master_seed"]==2026110103 and c["repetitions"]==24 and c["task_count"]==192,"Production repetition specification differs")
    require(tuple(c["laws"])==LAWS and tuple(c["nonlinear_laws"])==NONLINEAR,"Law roster differs")
    require(c["histories"]==[12] and tuple(c["horizons"])==HORIZONS and c["noise_levels"]==[0.,.1],"Dimension/horizon/noise roster differs")
    require(c["split_episodes"]=={"A":64,"B":32,"C":32,"D":128} and c["split_caps"]=={"A":512,"B":256,"C":256,"D":None},"Split sizes differ")
    require(c["max_episode_length"]==72 and c["origins_per_episode"]==13 and c["stride"]==4,"Episode/origin design differs")
    require(tuple(c["methods"])==METHODS and tuple(c["geometry_methods"])==GEOMETRY and tuple(c["contexts"])==CONTEXTS,"Method/context roster differs")
    require(c["candidates"]==[{"candidate_id":key,"rank_cap":value[0],"dimension_policy":value[1]} for key,value in CANDIDATES.items()],"Dimension candidate design differs")
    require(tuple(c["native_candidate_order"])==NATIVE and c["geometry_scaling"]=="standard","Native order/scaling differs")
    a=c["analysis"]
    require(a["primary_context"]=="selected" and a["primary_candidate"]=="kahm_departure" and tuple(a["primary_comparators"])==COMPARATORS,"Primary candidate family differs")
    require(a["family_alpha"]==.05 and a["primary_family_size"]==6 and a["superiority_margin"]==.03 and a["linear_noninferiority_margin"]==.05 and a["max_error_ratio"]==5.,"Primary inferential specification differs")


def verify_chain(root,c):
    phases=(("freeze.json","frozen","source"),("FIT_LOCK.json","sealed","fits"),
            ("PREDICTION_LOCK.json","sealed","predictions"),("TARGET_RELEASE.json","sealed","target_release"))
    records={}; hashes={}; checked={}; distribution=None; verified_source_paths=set()
    config_hash=digest(root/"config.json")
    for name,status,phase in phases:
        record=read(root/name)
        require(record["status"]==status and record["phase"]==phase and record["task_count"]==192 and record["config_sha256"]==config_hash,"Invalid seal: "+name)
        require(record["files"],"Empty seal file map")
        if name=="freeze.json" and (root/"source_distribution.json").exists():
            sys.path.insert(0,str(ROOT/"code"))
            from source_distribution import verify_source_distribution
            distribution=verify_source_distribution(root,record)
            verified_source_paths=set(record["files"])-set(distribution["omitted_manuscripts"])
        else:
            for relative,expected in record["files"].items():
                path=confined(root,relative)
                actual=checked.get(relative)
                if actual is None: actual=checked[relative]=digest(path)
                require(actual==expected,"Sealed file changed: "+relative)
        records[name]=record; hashes[name]=digest(root/name)
    for name in ("FIT_LOCK.json","PREDICTION_LOCK.json","TARGET_RELEASE.json"):
        require(records[name]["freeze_sha256"]==hashes["freeze.json"],"Source predecessor differs")
    for name in ("PREDICTION_LOCK.json","TARGET_RELEASE.json"):
        require(records[name]["fit_lock_sha256"]==hashes["FIT_LOCK.json"],"Fit predecessor differs")
    require(records["TARGET_RELEASE.json"]["prediction_lock_sha256"]==hashes["PREDICTION_LOCK.json"],"Prediction predecessor differs")
    moments=[datetime.fromisoformat(records[name]["created_utc"]) for name,_,_ in phases]
    require(moments==sorted(moments),"Seal chronology differs")
    require(records["freeze.json"]["production_data_generated"] is False and records["FIT_LOCK.json"]["D_generated"] is False and records["PREDICTION_LOCK.json"]["D_scoring_started"] is False and records["TARGET_RELEASE.json"]["targets_read_for_scoring_before_release"] is False,"Data role declarations differ")
    for path in Path(__file__).parent.glob("*.py"):
        relative="analysis_code/"+path.name
        expected=records["freeze.json"]["files"].get(relative)
        if distribution is not None and relative in distribution["verified_replacement_files"]:
            expected=distribution["verified_replacement_files"][relative]["distributed_sha256"]
        require(expected==digest(path),"Analyzer source identity differs: "+path.name)
    integrity={"seal_sha256":hashes,"verified_distinct_files":len(set(checked)|verified_source_paths),"verified_before_terminal_target_access":True}
    if distribution is not None:
        integrity["source_distribution"]=distribution
    return records,integrity


def manifest_roster(manifest):
    tasks=manifest["tasks"]
    expected={(law,12,noise,rep) for law in LAWS for noise in (0,1) for rep in range(24)}
    require(len(tasks)==192 and len({t["task_id"] for t in tasks})==192,"Manifest task count/uniqueness differs")
    require({(t["law"],t["history"],t["noise_index"],t["repetition"]) for t in tasks}==expected,"Manifest factorial differs")
    return {t["task_id"]:t for t in tasks}


def analyze(root=ROOT,output=None):
    root=Path(root).resolve(); c=read(root/"config.json"); validate_config(c)
    seals,integrity=verify_chain(root,c)
    abc=read(root/"prepared/abc_manifest.json"); terminal=read(root/"prepared/d_manifest.json")
    tasks=manifest_roster(abc); d_tasks=manifest_roster(terminal)
    require(set(tasks)==set(d_tasks),"ABC/D task identity differs")
    require(seals["FIT_LOCK.json"]["ABC_manifest_sha256"]==digest(root/"prepared/abc_manifest.json"),"ABC manifest hash differs")
    for name in ("PREDICTION_LOCK.json","TARGET_RELEASE.json"):
        require(seals[name]["D_manifest_sha256"]==digest(root/"prepared/d_manifest.json"),"D manifest hash differs")
    for name in ("FIT_LOCK.json","PREDICTION_LOCK.json","TARGET_RELEASE.json"):
        require(len(seals[name]["task_ids"])==192 and set(seals[name]["task_ids"])==set(tasks),"Seal task roster differs")
    horizons=[]; losses=[]; screens=[]; diagnostics=defaultdict(list); inputs=[]
    for index,task_id in enumerate(sorted(tasks),1):
        meta=tasks[task_id]; d=d_tasks[task_id]
        require(all(meta[k]==d[k] for k in META),"ABC/D task metadata differs")
        fdir=root/"results"/task_id; pdir=root/"predictions"/task_id
        fit=read(fdir/"fit_summary.json"); pre=read(fdir/"pre_c_selection.json"); query=read(pdir/"predict_summary.json")
        require(fit["task"]==pre["task"]==query["task"]==meta,"Summary task identity differs")
        require(fit["freeze_sha256"]==integrity["seal_sha256"]["freeze.json"] and query["fit_lock_sha256"]==integrity["seal_sha256"]["FIT_LOCK.json"],"Fit/query source identity differs")
        input_path=confined(root,meta["input"]); target_path=confined(root,d["target"])
        require(digest(input_path)==meta["sha256"]==fit["input_sha256"],"ABC file identity differs")
        require(digest(target_path)==d["target_sha256"] and d["target"] in seals["TARGET_RELEASE.json"]["files"],"Unreleased/different target file")
        a=arrays(input_path)
        require(not any(k.startswith("D_") for k in a),"ABC archive contains D")
        for split,rows in (("A",512),("B",256),("C",256)):
            require(a[split+"_X"].shape==(rows,12) and a[split+"_Y"].shape==(rows,3),"ABC array sizes differ")
        target=arrays(target_path)["D_Y"]; prediction=arrays(pdir/"predictions.npz")
        require(digest(pdir/"predictions.npz")==query["prediction_file_sha256"],"Prediction file identity differs")
        require(set(prediction)==set(query["prediction_array_sha256"]),"Forecast hash roster differs")
        for key,value in prediction.items(): require(array_hash(value)==query["prediction_array_sha256"][key],"Forecast array hash differs")
        tables,scale=diagnose(meta,fit,pre,arrays(fdir/"validation_predictions.npz"),arrays(fdir/"calibration_predictions.npz"),arrays(fdir/"basis_diagnostics.npz"),query,prediction,a,c)
        h,t,s=score_task(meta,target,prediction,scale,fit["selections"],fit["contexts"])
        horizons.extend(h); losses.extend(t); screens.extend(s)
        for name,rows in tables.items(): diagnostics[name].extend(rows)
        inputs.append({"task_id":task_id,"target_array_sha256":array_hash(target),"prediction_file_sha256":query["prediction_file_sha256"],"forecast_arrays":len(prediction)})
        if index%24==0: print(json.dumps({"analyzed_tasks":index,"expected":192}),flush=True)
    require(len(losses)==192*len(VIEWS)*2 and len(horizons)==3*len(losses),"Outcome table count differs")
    tables=all_outcome_tables(horizons,losses); tables.update(diagnostics); tables["stability_screens.csv"]=screens
    counts=[]
    for law in LAWS:
        for noise in (0.,.1):
            for method in GEOMETRY:
                rows=[r for r in tables["cap_selections.csv"] if r["law"]==law and r["noise_level"]==noise and r["method"]==method]
                require(len(rows)==24,"Cap-count roster differs")
                counts.extend({"law":law,"noise_level":noise,"method":method,"cap":cap,"count":sum(r["selected_cap"]==cap for r in rows)} for cap in (1,2,4,8,12,20))
    tables["cap_selection_counts.csv"]=counts
    tables["resource_summary.csv"]=means(tables["procedure_resources.csv"],("context","method"),("fit_and_selection_wall_seconds","fit_and_selection_cpu_seconds","deployment_pickle_bytes","deployment_array_bytes"))
    primary=tables["primary_contrasts.csv"]; selected_screens=[r for r in screens if r["view"]=="context_selected"]
    statistical=all(r["criterion_established"] for r in primary); stable=all(r["screen_pass"] for r in selected_screens)
    summary={"status":"completed","created_utc":datetime.now(timezone.utc).isoformat(),"task_count":192,"independent_repetitions":24,
        "primary_context":"selected","primary_contrasts":primary,"primary_comparisons_established":sum(r["criterion_established"] for r in primary),
        "statistical_criterion_established":statistical,"maximum_error_screens_passed":stable,
        "overall_criterion_established":statistical and stable,"maximum_selected_error_ratio":max(r["maximum_error_ratio"] for r in selected_screens),
        "failed_stability_tasks":[r["task_id"] for r in selected_screens if not r["screen_pass"]],
        "interpretation":"The joint criterion requires all six margin bounds and all declared stability screens. Failure does not establish equivalence or preclude further KAHM research.",
        "inference_scope":"Approximate Bonferroni one-sided paired-t family over six contrasts; independent repetitions; these are not risk certificates.",
        "generalization_scope":"Fresh repetitions of the same four numerical observation laws; no new application cohort.",
        "integrity":integrity,"input_records":inputs,"tables":{}}
    out=Path(output).resolve() if output else root/"analysis"; out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        export_csv(out/name,rows); summary["tables"][name]={"rows":len(rows),"sha256":digest(out/name)}
    write_json(out/"analysis_summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",type=Path,default=ROOT); parser.add_argument("--output",type=Path)
    args=parser.parse_args(); result=analyze(args.root,args.output)
    print(json.dumps({k:result[k] for k in ("status","task_count","primary_comparisons_established","overall_criterion_established")}))
