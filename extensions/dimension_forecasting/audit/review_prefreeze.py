#!/usr/bin/env python3
"""Finite pre-production gate combining independent reconstruction and source review."""
from datetime import datetime, timezone
import argparse
import csv
import tempfile
import importlib.util
import json
import math
from pathlib import Path
from scipy.special import stdtrit
import numpy as np
from verify_task_artifacts import Audit, read, sha


def main(root):
    a = Audit()
    a.require(not (root/"freeze.json").exists() and not (root/"prepared/abc_manifest.json").exists()
              and not (root/"results").exists(), "no production source freeze, data or fitting before audit")
    receipts = ("audit/design_config_checks.json", "audit/task_artifact_fixture_checks.json",
                "audit/historical_suite_checks.json", "verification/full_comparison_adapter_tests.json",
                "verification/full_staged_chain_tests.json", "verification/implementation_readiness.json",
                "verification/analysis_preflight.json")
    for name in receipts:
        a.require(read(root/name)["status"] == "passed", "required finite gate passed", name)
    for row in read(root/"runtime/STEP1_COPY_MANIFEST.json")["files"]:
        a.require(sha(root/"runtime/step1"/row["path"]) == row["sha256"], "unchanged Step1 runtime copy", row["path"])
    for name, digest in read(root/"readiness/copied_source_manifest.json")["files"].items():
        a.require(sha(root/name) == digest, "copied source identity", name)
    fixture = root/"verification/staged_chain_fixture_final"
    stages = ("freeze.json", "FIT_LOCK.json", "PREDICTION_LOCK.json", "TARGET_RELEASE.json")
    records = {}
    for name in stages:
        record = records[name] = read(fixture/name)
        a.require(bool(record["files"]), "manufactured full-stage nonempty identity map", name)
        for path, expected in record["files"].items():
            a.require(sha(fixture/path) == expected, "manufactured stage sealed bytes", [name, path])
        if name != "freeze.json":
            a.require(record["freeze_sha256"] == sha(fixture/"freeze.json"), "stage frozen source predecessor", name)
    a.require(records["PREDICTION_LOCK.json"]["fit_lock_sha256"] == sha(fixture/"FIT_LOCK.json")
              and records["TARGET_RELEASE.json"]["fit_lock_sha256"] == sha(fixture/"FIT_LOCK.json")
              and records["TARGET_RELEASE.json"]["prediction_lock_sha256"] == sha(fixture/"PREDICTION_LOCK.json"),
              "stage predecessor chain")
    moments = [datetime.fromisoformat(records[name]["created_utc"]) for name in stages]
    a.require(moments == sorted(moments), "strict data-role chronology")
    for path in (root/"code").glob("*.py"):
        a.require(sha(path) == sha(fixture/"code"/path.name), "final worker/source bytes were exercised", path.name)
    spec = importlib.util.spec_from_file_location("manufactured_primary_only", root/"analysis_code/analysis_core.py")
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    # Deliberately use Python scalar accumulation and stdtrit rather than the
    # analyzer's vector sample moments and scipy.stats.t implementation.
    left = np.array([.17 + .001*i + .004*math.sin(i) for i in range(24)])
    right = np.array([.21 + .0007*i + .006*math.cos(i) for i in range(24)])
    for factor in (.97, 1.05):
        gaps = [factor*float(y)-float(x) for x,y in zip(left,right)]
        mean = math.fsum(gaps)/24
        sd = math.sqrt(math.fsum((g-mean)**2 for g in gaps)/23)
        critical = float(stdtrit(23,1-.05/6))
        result,_ = core.margin_contrast(left,right,factor=factor,identifier="manufactured",
                                       panel="nonlinear" if factor == .97 else "linear",comparator_name="control")
        a.close([result["margin_gap_mean"],result["margin_gap_sample_sd"],
                 result["margin_gap_se"],result["simultaneous_one_sided_lower"],result["critical_t"]],
                [mean,sd,sd/math.sqrt(24),mean-critical*sd/math.sqrt(24),critical],
                "independent six-family scalar margin formula", factor)
    a.require(read(root/"verification/analysis_preflight.json")["assertions"] == 71,
              "final analysis gate includes atomic export tests")
    with tempfile.TemporaryDirectory(prefix="kahm_forecast_export_gate_") as directory:
        directory=Path(directory)
        records=[{"number":1.2345678901234567,"text":"quoted \"cell\", next\nline","optional":None,"array":[1,2]},
                 {"number":0.,"text":"plain","optional":True,"array":[]}]
        core.export_csv(directory/"table.csv",records)
        with (directory/"table.csv").open(newline="") as stream: saved=list(csv.DictReader(stream))
        a.require(len(saved)==2 and saved[0]["text"]==records[0]["text"] and saved[0]["optional"]==""
                  and float(saved[0]["number"])==records[0]["number"] and saved[1]["optional"]=="True"
                  and json.loads(saved[0]["array"])==[1,2] and not (directory/"table.csv.tmp").exists(),
                  "independent atomic CSV exact-cell and final-install check")
        core.write_json(directory/"value.json",{"records":records})
        a.require(read(directory/"value.json")=={"records":records} and not (directory/"value.json.tmp").exists(),
                  "independent atomic JSON final-install check")
    sources = {str(p.relative_to(root)):sha(p) for directory in ("code","analysis_code","audit")
               for p in sorted((root/directory).glob("*.py"))}
    result = {"schema":"kahm-full-forecast-independent-prefreeze-v1","status":"passed",
              "completed_utc":datetime.now(timezone.utc).isoformat(),"production_data_examined":False,
              "counts":dict(a.counts),"source_sha256":sources,
              "supporting_receipts":{name:sha(root/name) for name in receipts},
              "approved_design":{"tasks":192,"independent_repetitions":24,"contexts":3,"methods_per_context":10,
                                 "B_selected_geometry_methods":5,"prediction_arrays_per_task":162,
                                 "primary_context":"selected","primary_family_size":6},
              "source_review":[
                  "Original four numerical laws, A-only noise scale, causal 13-origin windows, and paired noise/cap streams retained.",
                  "All raw grids remain complete; a candidate/factor error stops the study rather than shrinking a grid.",
                  "B cap selection and alpha tuning precede durable pre-C record; C only selects mixing and seven-procedure policy.",
                  "Global fit lock gates D generation; global prediction lock gates target release and scoring.",
                  "Stored bases independently match native min(cap,d,N_local-1), positive-eigenvalue and range rules.",
                  "All ten historical procedures exactly reproduced on manufactured C and D queries in an isolated original-runtime process.",
                  "Selected and matched-bank controls are explicitly distinguished; original six primary margins and maximum absolute error screen retained.",
                  "Primary independent unit is repetition; paired-t bounds are approximate inferential summaries, not finite-sample risk certificates.",
                  "Atomic CSV/JSON export guards independently exercised after the initial source freeze was transparently superseded before any production data generation."
              ],
              "preserved_audit_attempts":[{"file":"audit/design_config_checks_attempt1.json",
                  "reason":"Audit initially treated 25 literal historical identifier seeds as derived SeedSequence outputs; identifier handling corrected. Independent exclusion-set equality and no collision checks already passed. No scientific source changed."}],
              "remaining_blockers":[]}
    (root/"audit/prefreeze_review.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"passed","counts":dict(a.counts),"receipt":"audit/prefreeze_review.json"}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root.resolve())
