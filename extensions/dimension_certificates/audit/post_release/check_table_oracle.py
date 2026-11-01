#!/usr/bin/env python3
"""Fabricated 26-run parser/math replay; no scheduled observations or fits."""
import argparse
from collections import defaultdict
import copy
from datetime import datetime,timezone
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile


def main(root):
    sys.path.insert(0,str(root/"audit"))
    from independent_scores import Audit,read,sha
    from independent_tables import job_tables,aggregate_tables,KEYS
    import table_comparison
    table_comparison.TABLE_KEYS=KEYS
    spec=importlib.util.spec_from_file_location("manufactured_primary",root/"analyze.py")
    primary=importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
    fixture=root/"verification/manufactured_v1"
    cfg=read(root/"config.json");cfg.update(n_train=40,n_validation=20,n_calibration=48,n_test=24,fabricated_fixture=True)
    a=Audit()
    with tempfile.TemporaryDirectory(prefix="kahm_risk_table_plumbing_") as temporary:
        run=Path(temporary)/"manufactured_run";run.mkdir();(run/"jobs").mkdir()
        def write(path,value):path.write_text(json.dumps(value,indent=2)+"\n")
        schedule=[]
        for si,system in enumerate(cfg["systems"]):
            for rep in range(system["repetitions"]):
                schedule.append({"id":f"{system['name']}_r{rep:02d}","system":system["name"],"system_index":si,
                                 "repetition":rep,"full_prediction_audit":rep==0,"fabricated_parser_only":True})
        write(run/"config.json",cfg);write(run/"schedule.json",schedule)
        for name in ("manifest.json","preflight.json"):write(run/name,{"fabricated_parser_only":True})
        (run/"PROTOCOL.md").write_text("Fabricated parser/math fixture; all forecast arrays come from the one manufactured fixture.\n")
        for job in schedule:
            folder=run/"jobs"/job["id"];folder.mkdir()
            for stage in ("development","calibration","test"):
                record=read(fixture/(stage+".json"));record["job"]=job
                if stage!="development":record["full_prediction_arrays_retained"]=job["full_prediction_audit"]
                write(folder/(stage+".json"),record)
                write(folder/(stage+"_started.json"),{"fabricated_parser_only":True})
                write(folder/(stage+"_complete.json"),{"fabricated_parser_only":True})
            for name in ("fitted_models.pkl","coefficients.npz","validation_scores.npz","calibration_scores.npz","test_scores.npz"):
                shutil.copyfile(fixture/name,folder/name)
            if job["full_prediction_audit"]:
                for stage in ("calibration","test"):shutil.copyfile(fixture/(stage+"_predictions.npz"),folder/(stage+"_predictions.npz"))
        frozen={"files":{n:sha(run/n) for n in ("config.json","PROTOCOL.md","schedule.json","manifest.json","preflight.json")},
                "source_sha256":{"analyze.py":sha(root/"analyze.py")},"fabricated_parser_only":True}
        write(run/"freeze.json",frozen);previous=None
        stagefiles={"development":("development.json","fitted_models.pkl","coefficients.npz","validation_scores.npz"),
                    "calibration":("calibration.json","calibration_scores.npz"),"test":("test.json","test_scores.npz")}
        for stage,names in stagefiles.items():
            files={}
            for job in schedule:
                selected=list(names)+[stage+"_started.json",stage+"_complete.json"]
                if stage!="development" and job["full_prediction_audit"]:selected.append(stage+"_predictions.npz")
                for name in selected:
                    path=f"jobs/{job['id']}/{name}";files[path]=sha(run/path)
            seal={"stage":stage,"n_jobs":26,"freeze_sha256":sha(run/"freeze.json"),"preceding_seal_sha256":previous,"files":files}
            write(run/(stage+"_seal.json"),seal);previous=sha(run/(stage+"_seal.json"))
        primary.analyze(run,run/"analysis")
        expected=defaultdict(list)
        for job in schedule:
            for name,rows in job_tables(run/"jobs"/job["id"],cfg).items():expected[name].extend(rows)
        aggregate_tables(expected,cfg)
        comparisons=table_comparison.table_comparison(run/"analysis",dict(expected),a)
    result={"status":"passed","manufactured_only":True,"physical_model_replay":False,"completed_utc":datetime.now(timezone.utc).isoformat(),
            "counts":dict(a.counts),"tables":comparisons,
            "scope":"Fabricated full 26-run parser/math family made from one manufactured fixture; all nine table schemas and values independently reconstructed. No production seed, law generation, fit, or outcome used."}
    Path(__file__).with_name("table_oracle_fixture.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"passed","counts":dict(a.counts)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root.resolve())
