#!/usr/bin/env python3
"""Fresh, separately frozen KAHM preprocessing confirmation; two workers.

Leaves the existing matched study and all repository sources untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "kahm-study"
EXT = REPO / "extensions/certified_forecasting"
PROTOCOL = ROOT / "confirmation_protocol.json"
CONFIG_ROOT = ROOT / "confirmation_configs"
OUTPUT = ROOT / "confirmation_results"
MODES = ("standard", "minmax", "minmax_clip", "raw")
CAMPAIGNS = ("main_v1", "oscillators_v1")
SEEDS = (4701, 4702, 4703)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_hashes():
    files = list(REPO.glob("*.py"))
    files += list((EXT / "src").rglob("*.py"))
    files += list((EXT / "experiments").glob("*.py"))
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(files)}


def freeze_protocol():
    configurations = {}
    for campaign in CAMPAIGNS:
        cfg = json.loads((EXT / f"configs/{campaign}.json").read_text())
        cfg["campaign"] = "confirmation_" + campaign
        cfg["seeds"] = list(SEEDS)
        configurations[campaign] = cfg
    jobs = [[mode, campaign, system, seed]
            for campaign in CAMPAIGNS
            for system in configurations[campaign]["systems"]
            for seed in SEEDS for mode in MODES]
    protocol = {
        "study": "Fresh confirmation of the KAHM input-scaling follow-up",
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "design_status": "Frozen before any confirmation evaluation. Motivated by the matched scaling investigation; not an independently preregistered external study.",
        "primary_family": "kahm_nested",
        "primary_contrast": "minmax versus standard",
        "primary_outcome": "Per-task held-out MSE averaged over all three horizons for the validation-selected primary-family candidate; report all three fitting seeds.",
        "secondary_arms": ["minmax_clip", "raw"],
        "selection": "For each task, fitting seed, and arm, select the kahm_nested candidate solely by mean validation MSE across horizons, then fewer features, then name. Never select a scaling arm using confirmation test outcomes.",
        "freshness": "New fitting seeds4701,4702,4703 regenerate all training, validation, calibration, and test episodes using the inherited disjoint stream offsets. Every arm uses identical streams within each task and seed.",
        "modes": {
            "standard": "Exact inherited training mean/std preprocessing.",
            "minmax": "Training featurewise min/max to [-1,1]; no clipping of future extremes. Primary scaling treatment.",
            "minmax_clip": "Same fitted min/max, with transformed query clipping to [-1,1].",
            "raw": "Preserve existing mathematically bounded raw inputs without standardization.",
        },
        "intervention_scope": "Change KAHM input preprocessing only. Original targets, candidate families, hyperparameters, samples, origins, score arrays, and decoder/certificate routines are unchanged.",
        "statistical_scope": "Original risk deltas apply separately within each arm and complete replicate. They are not simultaneous across four preprocessing arms, six tasks, or all seeds. Conditional episode-loss comparisons and variation over fitting seeds must be distinguished.",
        "reporting": "Retain every completed replicate and failure; report per-task effects and per-seed signs, including regressions and inconclusive results. Do not tune preprocessing or parameters from confirmation outcomes.",
        "configurations": configurations, "jobs": jobs, "workers": 2,
        "harness_sha256": digest(__file__),
        "scaler_sha256": digest(ROOT / "scaling_features.py"),
        "source_sha256": source_hashes(),
        "matched_protocol_sha256": digest(ROOT / "protocol.json"),
    }
    if PROTOCOL.exists():
        existing = json.loads(PROTOCOL.read_text())
        comparable = dict(existing)
        comparable.pop("frozen_utc", None)
        if comparable != protocol:
            raise RuntimeError("Confirmation protocol or sources changed; do not mix evidence.")
        protocol = existing
    else:
        protocol["frozen_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(PROTOCOL, protocol)
    for campaign, cfg in configurations.items():
        path = CONFIG_ROOT / f"{campaign}.json"
        if path.exists():
            if json.loads(path.read_text()) != cfg:
                raise RuntimeError(f"Confirmation configuration changed: {path}")
        else:
            write_json(path, cfg)
    return protocol


def worker(args):
    protocol = json.loads(PROTOCOL.read_text())
    job = [args.mode, args.campaign, args.system, args.seed]
    if job not in protocol["jobs"]:
        raise RuntimeError("Job is outside the frozen confirmation schedule.")
    if (protocol["harness_sha256"] != digest(__file__) or
            protocol["scaler_sha256"] != digest(ROOT / "scaling_features.py") or
            protocol["source_sha256"] != source_hashes()):
        raise RuntimeError("Frozen confirmation sources changed.")
    cfg = json.loads((CONFIG_ROOT / f"{args.campaign}.json").read_text())
    if cfg != protocol["configurations"][args.campaign]:
        raise RuntimeError("Frozen configuration changed.")
    destination = OUTPUT / args.mode / args.campaign / f"{args.system}_{args.seed}"
    destination.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(EXT / "src"), str(EXT / "experiments"), str(ROOT)]
    import run_forecasting as runner
    import scaling_features
    import numpy as np
    import scipy
    import sklearn
    scaling_features.reset_fitted_banks()
    runner.NestedSoftFeatures = scaling_features.make_feature_class(args.mode)
    if args.campaign == "oscillators_v1":
        import run_oscillators
        settings = cfg["generator"]

        def generator(system, n_episodes, length, seed):
            return run_oscillators.oscillator_episodes(
                system, n_episodes, length, seed, dt=settings["dt"],
                observation_interval=settings["observation_interval"],
                burn_time=settings["burn_time"])

        runner.make_episodes = generator
    metadata = {
        "mode": args.mode, "configuration": cfg,
        "protocol_sha256": digest(PROTOCOL),
        "config_sha256": digest(CONFIG_ROOT / f"{args.campaign}.json"),
        "harness_sha256": digest(__file__),
        "scaler_sha256": digest(ROOT / "scaling_features.py"),
        "source_sha256": source_hashes(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "python": sys.version,
        "platform": platform.platform(),
        "thread_limits": {key: os.environ.get(key) for key in
                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    write_json(destination / "intervention_manifest.json", metadata)
    try:
        result = runner.run_one(cfg, args.system, args.seed, destination)
        write_json(destination / "scaling_diagnostics.json", [
            bank.scaling_metadata() for bank in scaling_features.FITTED_BANKS])
        print(json.dumps({"job": job, "candidates": result["n_candidates"],
                          "seconds": result["elapsed_seconds"]}), flush=True)
    except Exception as error:
        write_json(destination / "failure.json", {
            "error": repr(error), "traceback": traceback.format_exc()})
        raise


def orchestrate(args):
    protocol = freeze_protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    study_manifest = {
        "protocol_sha256": digest(PROTOCOL),
        "harness_sha256": digest(__file__),
        "scaler_sha256": digest(ROOT / "scaling_features.py"),
        "source_sha256": source_hashes(),
        "jobs": protocol["jobs"], "workers": 2,
    }
    manifest_path = OUTPUT / "study_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != study_manifest:
            raise RuntimeError("Refusing to mix confirmation manifests.")
    else:
        write_json(manifest_path, study_manifest)
    jobs = []
    for job in protocol["jobs"]:
        mode, campaign, system, seed = job
        destination = OUTPUT / mode / campaign / f"{system}_{seed}"
        if args.resume and all((destination / name).exists() for name in
                               ("results.json", "episode_scores.npz", "scaling_diagnostics.json")):
            continue
        if destination.exists():
            raise RuntimeError(f"Existing incomplete/non-resumable job: {destination}")
        jobs.append(job)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    logs = OUTPUT / "logs"
    logs.mkdir(exist_ok=True)

    def launch(job):
        mode, campaign, system, seed = job
        logfile = logs / f"{mode}__{campaign}__{system}_{seed}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--mode", mode, "--campaign", campaign,
                   "--system", system, "--seed", str(seed)]
        started = time.perf_counter()
        with logfile.open("x") as output:
            result = subprocess.run(command, cwd=EXT, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT)
        return {"job": job, "returncode": result.returncode,
                "elapsed_seconds": time.perf_counter() - started,
                "log": str(logfile)}

    completed, started = [], time.perf_counter()
    print(f"START fresh confirmation: {len(jobs)} jobs, two workers", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            item = future.result()
            completed.append(item)
            print(f"DONE {len(completed)}/{len(jobs)} {item['job']} rc={item['returncode']}", flush=True)
            write_json(OUTPUT / "progress.json", {
                "completed": completed, "scheduled": len(jobs),
                "elapsed_seconds": time.perf_counter() - started})
    failures = [row for row in completed if row["returncode"]]
    write_json(OUTPUT / "completion.json", {
        "completed": completed, "failures": failures,
        "all_scheduled_jobs": protocol["jobs"],
        "elapsed_seconds": time.perf_counter() - started})
    if failures:
        raise RuntimeError(f"{len(failures)} confirmation jobs failed; all attempts retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--campaign", choices=CAMPAIGNS)
    parser.add_argument("--system")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.worker:
        if any(value is None for value in (args.mode, args.campaign, args.system, args.seed)):
            parser.error("Worker mode requires mode, campaign, system, and seed.")
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
