#!/usr/bin/env python3
"""Finite preflight checks on manufactured arrays, never production outcomes."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import copy
import csv
import importlib.util
import json
import tempfile
from pathlib import Path
import numpy as np
from analysis_core import (LAWS, METHODS, VIEWS, MODES, margin_contrast, score_task,
    all_outcome_tables, repetition_table, require, write_json, export_csv)
from diagnostics import diagnose

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text())
def arrays(path):
    with np.load(path,allow_pickle=False) as archive: return {k:archive[k] for k in archive.files}


def main():
    count=0
    original=ROOT.parent/"reconstruction_mechanism/analyze.py"
    spec=importlib.util.spec_from_file_location("original_analysis_reference",original)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rng=np.random.default_rng(761129)
    for factor,panel,comparator in ((.97,"nonlinear","common"),(1.05,"linear","ridge")):
        for candidate,control in ((rng.uniform(.1,1.,24),rng.uniform(.2,1.1,24)),(np.ones(24),np.ones(24)),(.5*np.ones(24),np.ones(24))):
            row,gaps=margin_contrast(candidate,control,factor=factor,identifier="manufactured",panel=panel,comparator_name=comparator)
            ref,rgap=module.paired_margin_contrast(candidate,control,comparator_factor=factor,contrast_id="manufactured",comparator_name=comparator,scope=panel,margin=.03 if panel=="nonlinear" else .05)
            np.testing.assert_array_equal(gaps,rgap); count+=1
            for key in ("margin_gap_mean","margin_gap_sample_sd","margin_gap_se","critical_t","simultaneous_one_sided_lower","p_one_sided","p_bonferroni","positive_margin_repetitions","criterion_established"):
                require(row[key]==ref[key],"Original margin calculation differs: "+key); count+=1
    q=ROOT/"verification/schema_fixture"; fit=read(q/"results/fit_summary.json"); pre=read(q/"results/pre_c_selection.json")
    validation=arrays(q/"results/validation_predictions.npz"); calibration=arrays(q/"results/calibration_predictions.npz")
    basis=arrays(q/"results/basis_diagnostics.npz"); query=read(q/"predictions/predict_summary.json")
    forecast=arrays(q/"predictions/predictions.npz"); abc=arrays(q/"abc.npz"); config=read(ROOT/"config.json")
    tables,scale=diagnose(fit["task"],fit,pre,validation,calibration,basis,query,forecast,abc,config); count+=1
    for kind in ("cap","gamma","policy","dimension"):
        changed=copy.deepcopy(fit)
        if kind=="cap": changed["selections"]["kahm_departure"]["selected_cap"]+=1
        elif kind=="gamma": changed["candidates"]["native20"]["calibration"]["C"]["kahm_departure"]["gamma_by_horizon"][0]=-.5
        elif kind=="policy": changed["contexts"]["selected"]["selected_policy"]="kahm_departure"
        else: changed["candidates"]["native20"]["fit_diagnostics"]["local_dimensions"][0]["effective_rank"]=-1
        try: diagnose(fit["task"],changed,pre,validation,calibration,basis,query,forecast,abc,config)
        except ValueError: count+=1
        else: raise AssertionError("Tampered artifact accepted: "+kind)
    tiled={key:np.tile(value,(int(np.ceil(1664/len(value))),1))[:1664] for key,value in forecast.items()}
    target=rng.normal(size=(1664,3))
    hrows,trows,screens=score_task(fit["task"],target,tiled,scale,fit["selections"],fit["contexts"])
    require(len(trows)==len(VIEWS)*len(MODES) and len(hrows)==3*len(trows) and len(screens)==3,"Manufactured score roster differs"); count+=1
    changed=dict(tiled); changed["context_selected__calibrated__kahm_departure"]=changed["context_selected__calibrated__kahm_departure"].copy()
    changed["context_selected__calibrated__kahm_departure"][0,0]+=1
    try: score_task(fit["task"],target,changed,scale,fit["selections"],fit["contexts"])
    except ValueError: count+=1
    else: raise AssertionError("Changed forecast view accepted")
    full_task=[]; full_horizon=[]
    for rep in range(24):
        for law in LAWS:
            for noise in (0,1):
                metadata={"task_id":f"manufactured_{rep}_{law}_{noise}","law":law,"noise_index":noise,"noise_level":.1*noise,"repetition":rep,"seed":rep,"history":12}
                factor=1.+.01*rep+.005*noise
                full_task.extend({**row,**metadata,"smse":row["smse"]*factor,"mse":row["mse"]*factor} for row in trows)
                full_horizon.extend({**row,**metadata,"smse":row["smse"]*factor,"mse":row["mse"]*factor} for row in hrows)
    outcomes=all_outcome_tables(full_horizon,full_task)
    require(len(outcomes["primary_contrasts.csv"])==6 and len(outcomes["primary_repetition_contrasts.csv"])==144,"Primary family roster differs"); count+=1
    try: repetition_table(full_task[:-1])
    except ValueError: count+=1
    else: raise AssertionError("Incomplete result roster accepted")
    with tempfile.TemporaryDirectory(prefix="kahm_analysis_export_") as temporary:
        directory=Path(temporary)
        export_csv(directory/"primary.csv",outcomes["primary_contrasts.csv"])
        with (directory/"primary.csv").open(newline="") as stream: saved=list(csv.DictReader(stream))
        require(len(saved)==6 and [float(r["simultaneous_one_sided_lower"]) for r in saved]==[r["simultaneous_one_sided_lower"] for r in outcomes["primary_contrasts.csv"]],"Primary CSV numbers changed on export"); count+=1
        export_csv(directory/"special.csv",[{"text":"comma, quote\" and\nnewline","missing":None,"value":1.2345678901234567,"nested":[1,2]}])
        with (directory/"special.csv").open(newline="") as stream: saved=list(csv.DictReader(stream))
        require(saved[0]["text"]=="comma, quote\" and\nnewline" and saved[0]["missing"]=="" and float(saved[0]["value"])==1.2345678901234567 and json.loads(saved[0]["nested"])==[1,2],"CSV quoting or precision differs"); count+=1
    receipt={"status":"passed","manufactured_only":True,"assertions":count,"tests":["Exact original six-margin formula parity, including zero variance","Complete manufactured schema and diagnostic reanalysis","Tampered B cap, C gamma, C policy, local dimension and forecast-view rejection","Complete 192-task aggregate and missing-row rejection"],"diagnostic_rows":{k:len(v) for k,v in tables.items()},"analysis_table_rows":{k:len(v) for k,v in outcomes.items()}}
    write_json(ROOT/"verification/analysis_preflight.json",receipt)
    print(json.dumps(receipt))


if __name__=="__main__": main()
