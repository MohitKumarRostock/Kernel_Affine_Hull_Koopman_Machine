#!/usr/bin/env python3
"""Parser/math plumbing only; manufactured full-factorial forecasts."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np


def main(root):
    sys.path.insert(0,str(root/"audit"));sys.path.insert(0,str(root/"analysis_code"))
    from verify_task_artifacts import Audit,arrays,read,sha
    import independent_tables as independent
    import analysis_core as primary
    from diagnostics import diagnose
    fixture=root/"verification/schema_fixture_final"
    fit=read(fixture/"results/fit_summary.json");pre=read(fixture/"results/pre_c_selection.json")
    abc=arrays(fixture/"abc.npz");prediction=arrays(fixture/"predictions/predictions.npz")
    calibration=arrays(fixture/"results/calibration_predictions.npz");validation=arrays(fixture/"results/validation_predictions.npz")
    basis=arrays(fixture/"results/basis_diagnostics.npz");query=read(fixture/"predictions/predict_summary.json")
    cfg=read(root/"config.json");audit=Audit()
    expected_diag=independent.diagnostic_tables(fit["task"],fit,pre,abc,validation,calibration,basis,query,prediction,cfg)
    actual_diag,scale=diagnose(fit["task"],fit,pre,validation,calibration,basis,query,prediction,abc,cfg)
    all_keys=dict(independent.TABLE_KEYS)
    with tempfile.TemporaryDirectory(prefix="kahm_forecast_table_plumbing_") as tmp:
        tmp=Path(tmp);diag=tmp/"diagnostics";diag.mkdir()
        for name,rows in actual_diag.items():primary.export_csv(diag/name,rows)
        independent.TABLE_KEYS={name:all_keys[name] for name in actual_diag}
        compared_diag=independent.table_comparison(diag,expected_diag,audit)
        simulated={key:np.tile(value,(88,1))[:1664] for key,value in prediction.items()}
        expected_h=[];expected_t=[];expected_s=[];actual_h=[];actual_t=[];actual_s=[]
        rng=np.random.default_rng(559098281)
        for rep in range(24):
            for law in primary.LAWS:
                for noise in (0,1):
                    task={"task_id":f"manufactured_{law}_{noise}_{rep}","law":law,"history":12,"repetition":rep,
                          "seed":559098281+rep,"noise_index":noise,"noise_level":noise*.1}
                    target=rng.normal(size=(1664,3))
                    h,t,s=independent.score_task(task,target,simulated,scale,fit)
                    expected_h+=h;expected_t+=t;expected_s+=s
                    h,t,s=primary.score_task(task,target,simulated,scale,fit["selections"],fit["contexts"])
                    actual_h+=h;actual_t+=t;actual_s+=s
        expected=independent.outcome_tables(expected_h,expected_t);expected["stability_screens.csv"]=expected_s
        actual=primary.all_outcome_tables(actual_h,actual_t);actual["stability_screens.csv"]=actual_s
        out=tmp/"outcomes";out.mkdir()
        for name,rows in actual.items():primary.export_csv(out/name,rows)
        independent.TABLE_KEYS={name:all_keys[name] for name in actual}
        compared_out=independent.table_comparison(out,expected,audit)
    independent.TABLE_KEYS=all_keys
    result={"status":"passed","completed_utc":datetime.now(timezone.utc).isoformat(),"manufactured_only":True,
            "physical_model_replay":False,"counts":dict(audit.counts),"diagnostic_tables":compared_diag,"outcome_tables":compared_out,
            "scope":"Independent full 192-task scalar-horizon/repetition/primary/decomposition reconstruction from fabricated forecasts, plus actual manufactured single-task diagnostic table replay. No production data used."}
    Path(__file__).with_name("table_oracle_fixture.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"passed","counts":dict(audit.counts)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root.resolve())
