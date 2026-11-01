#!/usr/bin/env python3
"""One-task manufactured integration test of the actual staged study CLI.

The scientific observation generators are never called.  A separate fixture
contains copied executable source, a declared reduced model configuration,
hand-built manifests and toy arrays.  Failed gate attempts and all successful
stage receipts remain available.  This is not a scientific study repetition.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "code/pipeline.py"
sys.path.insert(0, str(ROOT / "code"))
# Earlier model imports insert inherited directories into sys.path.  Loading
# the exact source file avoids accidentally importing an older pipeline.py.
spec = importlib.util.spec_from_file_location("crossfit_pipeline_fixture_source", PIPELINE)
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_snapshot():
    paths = []
    for directory in ("code", "analysis_code", "audit"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".md") and "__pycache__" not in p.parts)
    paths += [ROOT / name for name in ("config.json", "PROTOCOL.md", "README.md", "requirements.txt")]
    return {str(path.relative_to(ROOT)): pipeline.digest(path) for path in sorted(paths)}


def toy_targets(x):
    """Manufactured algebraic responses, unrelated to the study's four laws."""
    return np.column_stack((.6*x[:,0] + .2*np.sin(x[:,1]),
                            .3*x[:,2]*x[:,3] + .1*x[:,0],
                            np.cos(.4*x[:,4]) + .1*x[:,5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path,
                        default=ROOT / "readiness/fixtures/pipeline_manufactured_01")
    parser.add_argument("--receipt", type=Path, default=ROOT / "readiness/pipeline_fixture.json")
    args = parser.parse_args()
    fixture = args.fixture_root.resolve()
    if fixture.exists() or args.receipt.exists():
        raise FileExistsError("Preserve fixture attempts and receipts; choose a fresh destination")
    initial_source = source_snapshot()
    fixture.mkdir(parents=True)
    for directory in ("code", "analysis_code", "audit"):
        if (ROOT / directory).exists():
            shutil.copytree(ROOT / directory, fixture / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("PROTOCOL.md", "README.md", "requirements.txt"):
        shutil.copy2(ROOT / name, fixture / name)
    (fixture / "readiness").mkdir()
    shutil.copy2(ROOT / "readiness/prior_seed_exclusions.json",
                 fixture / "readiness/prior_seed_exclusions.json")
    c = json.loads((ROOT / "config.json").read_text())
    c.update(study_id="MANUFACTURED pipeline fixture: no scientific law outcomes",
             laws=["manufactured_fixture"], nonlinear_laws=["manufactured_fixture"],
             repetitions=1, noise_levels=[0.0], task_count=1, master_seed=73198452)
    c["seed_derivation"]["law_index_order"] = ["manufactured_fixture"]
    c["seed_derivation"]["distinct_seed_count"] = 14
    c["execution"]["workers"] = 1
    c["model"].update(blocks=[2], raw_rbf_landmarks=16,
                       raw_hgb_learning_rates=[.1], raw_hgb_max_leaf_nodes=[7], raw_hgb_max_iter=8)
    c["manufactured_fixture"] = {
        "scope": "Integration only; never a scientific result or inferential unit",
        "toy_seed": 67241093, "D_toy_seed": 67241094,
        "observation_generator_invoked": False,
        "source_config_sha256": initial_source["config.json"],
        "source_files": initial_source,
    }
    pipeline.write_json(fixture / "config.json", c)
    pipeline.write_json(fixture / "FIXTURE_SCOPE.json", c["manufactured_fixture"])
    for relative, digest in initial_source.items():
        if relative != "config.json":
            if pipeline.digest(fixture / relative) != digest:
                raise AssertionError("Copied source differs: " + relative)
    checks, commands = {}, []
    log_dir = fixture / "fixture_commands"
    log_dir.mkdir()

    def cli(command, *, expected_error=None):
        args_cli = [sys.executable, str(PIPELINE), command, "--root", str(fixture), "--workers", "1"]
        start = time.perf_counter()
        result = subprocess.run(args_cli, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        record = {"command": command, "argv": args_cli, "returncode": result.returncode,
                  "started_utc": now(), "elapsed_seconds": time.perf_counter()-start,
                  "expected_error": expected_error, "output": result.stdout}
        pipeline.write_json(log_dir / f"{len(commands):02d}_{command}.json", record)
        commands.append({key:value for key,value in record.items() if key != "output"})
        if expected_error is None:
            if result.returncode:
                raise RuntimeError(f"Fixture CLI {command} failed:\n{result.stdout[-12000:]}")
        elif result.returncode == 0 or expected_error not in result.stdout:
            raise AssertionError(f"Expected {expected_error!r} rejection for {command}:\n{result.stdout[-12000:]}")
        return record

    cli("freeze")
    checks["freeze_precedes_all_manufactured_arrays"] = not (fixture / "prepared").exists()
    # A changed executable copy must be detected before stage orchestration.
    changed = fixture / "code/crossfit_model.py"
    original_bytes = changed.read_bytes()
    try:
        changed.write_bytes(original_bytes + b"\n# manufactured tamper probe\n")
        cli("fit", expected_error="Frozen file missing or changed")
    finally:
        changed.write_bytes(original_bytes)
    pipeline.verify_freeze(fixture)
    checks["changed_frozen_source_rejected"] = True

    rng = np.random.default_rng(67241093)
    arrays = {}
    for split, episodes in (("A",64),("B",32),("C",32)):
        n = episodes*8
        x = rng.normal(size=(n,12))
        y = toy_targets(x) + .05*rng.normal(size=(n,3))
        arrays[split+"_X"], arrays[split+"_Y"] = x,y
        arrays[split+"_groups"] = np.repeat(np.arange(episodes,dtype=np.int64),8)
        arrays[split+"_origins"] = np.tile(np.arange(12,44,4,dtype=np.int64),episodes)
    arrays["loss_scale"] = np.maximum(np.var(arrays["A_Y"],axis=0,ddof=0),c["model"]["target_variance_floor"])
    task = pipeline.task_roster(c)[0]
    input_path = f"prepared/abc/{task['task_id']}.npz"
    pipeline.write_npz(fixture / input_path, arrays)
    abc_task = {**task, "input":input_path,"sha256":pipeline.digest(fixture/input_path),
                "loss_scale":arrays["loss_scale"].tolist(), "A_base_episode_std":float(np.std(arrays["A_X"])),
                "array_sha256":{key:pipeline.array_hash(value) for key,value in arrays.items()},
                "manufactured_only":True}
    abc_manifest = {"schema":"crossfit-residual-abc-manufactured","created_utc":now(),
                    "config_sha256":pipeline.digest(fixture/"config.json"),
                    "freeze_sha256":pipeline.digest(fixture/"freeze.json"),
                    "tasks":[abc_task],"task_count":1,"base_files":{},"D_generated":False,
                    "scientific_generator_invoked":False}
    pipeline.write_json(fixture/"prepared/abc_manifest.json",abc_manifest)
    pipeline.verify_manifest(fixture,"abc_manifest.json")
    cli("d",expected_error="FIT_LOCK.json")
    if (fixture/"prepared/d_manifest.json").exists() or (fixture/"prepared/targets").exists():
        raise AssertionError("Missing-fit-seal probe generated terminal data")
    checks["D_generation_without_fit_seal_rejected"] = True

    cli("fit")
    cli("seal-fits")
    fit_lock = pipeline.verify_seal(fixture,"fits")
    checks["full_fit_seal_before_D"] = fit_lock["task_count"] == 1
    # Changing a fitted artifact must also block entry into D generation.
    state_path = fixture/f"results/{task['task_id']}/state.pkl.gz"
    state_bytes = state_path.read_bytes()
    try:
        state_path.write_bytes(state_bytes+b"manufactured-state-tamper")
        cli("d",expected_error="Frozen file missing or changed")
    finally:
        state_path.write_bytes(state_bytes)
    pipeline.verify_seal(fixture,"fits")
    checks["changed_sealed_state_rejected"] = True

    # Terminal toy histories and responses are constructed only now.
    rng_d = np.random.default_rng(67241094)
    dx = rng_d.normal(size=(1664,12))
    dy = toy_targets(dx)+.05*rng_d.normal(size=(1664,3))
    metadata = {"D_groups":np.repeat(np.arange(128,dtype=np.int64),13),
                "D_origins":np.tile(np.arange(12,64,4,dtype=np.int64),128)}
    query_path=f"prepared/queries/{task['task_id']}.npz"
    target_path=f"prepared/targets/{task['task_id']}.npz"
    pipeline.write_npz(fixture/query_path,{"D_X":dx,**metadata})
    pipeline.write_npz(fixture/target_path,{"D_Y":dy,**metadata})
    d_task={**task,"query":query_path,"target":target_path,
            "query_sha256":pipeline.digest(fixture/query_path),"target_sha256":pipeline.digest(fixture/target_path),
            "D_X_sha256":pipeline.array_hash(dx),"D_Y_sha256":pipeline.array_hash(dy),
            "rows":len(dx),"manufactured_only":True}
    d_manifest={"schema":"crossfit-residual-d-manufactured","created_utc":now(),
                "config_sha256":pipeline.digest(fixture/"config.json"),
                "freeze_sha256":pipeline.digest(fixture/"freeze.json"),
                "fit_lock_sha256":pipeline.digest(fixture/"FIT_LOCK.json"),
                "tasks":[d_task],"task_count":1,"base_files":{},"D_generated":True,
                "scientific_generator_invoked":False}
    pipeline.write_json(fixture/"prepared/d_manifest.json",d_manifest)
    pipeline.verify_manifest(fixture,"d_manifest.json")
    with np.load(fixture/query_path,allow_pickle=False) as z:
        if any("Y" in key for key in z.files):
            raise AssertionError("Manufactured query archive contains targets")
    cli("predict")
    cli("release",expected_error="PREDICTION_LOCK.json")
    if (fixture/"TARGET_RELEASE.json").exists():
        raise AssertionError("Target release succeeded without a prediction seal")
    checks["target_release_without_prediction_seal_rejected"] = True
    cli("seal-predictions")
    cli("release")
    cli("verify")
    release = pipeline.verify_release(fixture)

    results=fixture/"results"/task["task_id"]
    predictions_dir=fixture/"predictions"/task["task_id"]
    summary=json.loads((results/"fit_summary.json").read_text())
    receipt=json.loads((predictions_dir/"prediction_receipt.json").read_text())
    pre_c=json.loads((results/"pre_c_selection.json").read_text())
    if pre_c["C_loaded"] is not False:
        raise AssertionError("Pre-C selection receipt loaded C")
    if pre_c["crossfit_frozen_state_sha256"] != summary["crossfit"]["calibration"]["pre_C_hash"]:
        raise AssertionError("Development state differs between pre-C save and calibration")
    if summary["crossfit"]["calibration"]["pre_C_hash"] != summary["crossfit"]["calibration"]["post_C_hash"]:
        raise AssertionError("C mutated development state")
    checks["pre_C_saved_before_C_loading"] = True
    checks["C_preserves_development_state"] = True
    if receipt["array_count"] != 42 or receipt["targets_loaded"] is not False or receipt["state_unchanged"] is not True:
        raise AssertionError("Invalid target-free prediction receipt")
    if pipeline.digest(state_path) != fit_lock["files"][str(state_path.relative_to(fixture))]:
        raise AssertionError("Query changed sealed fitted bytes")
    checks["target_free_prediction_preserves_state"] = True
    with np.load(results/"calibration_predictions.npz",allow_pickle=False) as z:
        calibration_arrays={key:z[key] for key in z.files}
    with np.load(predictions_dir/"predictions.npz",allow_pickle=False) as z:
        d_arrays={key:z[key] for key in z.files}
    if set(calibration_arrays)!=pipeline.expected_keys() or set(d_arrays)!=pipeline.expected_keys():
        raise AssertionError("Fixture does not retain the complete 42-array roster")
    if any(value.shape!=(1664,3) or not np.isfinite(value).all() for value in d_arrays.values()):
        raise AssertionError("Invalid manufactured D forecasts")
    # Independent policy score/recovery oracle uses the declared C arrays only.
    scores=[]
    for identity in c["policy_order"]:
        family,method=identity.split("__",1)
        key=f"{family}__calibrated__{method}"
        score=float(np.mean(np.mean((arrays["C_Y"]-calibration_arrays[key])**2,axis=0)/arrays["loss_scale"]))
        scores.append(score)
    chosen=c["policy_order"][int(np.argmin(scores))]
    if chosen!=summary["policy"]["selected_identity"]:
        raise AssertionError("Expanded policy did not select the C minimum")
    np.testing.assert_allclose(scores,[row["score"] for row in summary["policy"]["candidates"]],rtol=0,atol=0)
    family,method=chosen.split("__",1)
    for destination,mode in ((calibration_arrays,"C"),(d_arrays,"D")):
        for view in ("calibrated","eta1"):
            srcmode="gamma1" if view=="eta1" and family=="legacy" else view
            np.testing.assert_array_equal(destination[f"policy__{view}__expanded_nonkahm"],
                                          destination[f"{family}__{srcmode}__{method}"])
    checks["C_policy_minimum_and_fixed_D_recovery"] = True
    common_bias=np.asarray(summary["crossfit"]["selections"]["shared_bias"]["value"])
    for container in (calibration_arrays,d_arrays):
        for view in ("calibrated","eta1"):
            np.testing.assert_allclose(container[f"crossfit__{view}__shared_bias"],
                                       container[f"crossfit__{view}__common"]+common_bias,rtol=0,atol=0)
    checks["shared_full_common_bias_identity"] = True
    if source_snapshot()!=initial_source:
        raise RuntimeError("Actual source changed during the integration test; preserve attempt and rerun final sources")
    for relative,digest in initial_source.items():
        if relative!="config.json" and pipeline.digest(fixture/relative)!=digest:
            raise AssertionError("Copied executable source changed during the fixture")
    result={"status":"passed","completed_utc":now(),
            "scope":"One manufactured integration task only; no scientific generator or production outcomes",
            "fixture_root":str(fixture),"task_id":task["task_id"],"methods_arrays_per_task":42,
            "rows":{"A":512,"B":256,"C":256,"D":1664},"history":12,"A_episode_count":64,
            "source_hashes":initial_source,"fixture_config_sha256":pipeline.digest(fixture/"config.json"),
            "fixture_freeze_sha256":pipeline.digest(fixture/"freeze.json"),
            "fixture_fit_seal_sha256":pipeline.digest(fixture/"FIT_LOCK.json"),
            "fixture_prediction_seal_sha256":pipeline.digest(fixture/"PREDICTION_LOCK.json"),
            "fixture_target_release_sha256":pipeline.digest(fixture/"TARGET_RELEASE.json"),
            "checks":checks,"stage_commands":commands,"release":release,
            "scientific_generator_invocations":0,"failed_gate_attempts_preserved":4,
            "source_and_copied_executables_identical_at_start_and_end":True}
    pipeline.write_json(args.receipt,result)
    print(json.dumps({"status":"passed","fixture_root":str(fixture),"checks":checks,
                      "receipt":str(args.receipt)},indent=2))


if __name__=="__main__":
    main()
