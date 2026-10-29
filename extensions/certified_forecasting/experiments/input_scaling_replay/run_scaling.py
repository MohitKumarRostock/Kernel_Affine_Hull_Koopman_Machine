#!/usr/bin/env python3
"""Isolated matched preprocessing study using the unchanged manuscript runner.

This script never writes into the source repository. Each job imports the
original runner and substitutes only its KAHM feature class. All original
candidates, data streams, and score arrays are retained.
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
DEFAULT_REPO = ROOT.parent / "kahm-study"
MODES = ["standard", "minmax", "minmax_clip", "raw"]
CAMPAIGNS = ["main_v1", "oscillators_v1", "sensitivity_v1_k1",
             "sensitivity_v1_k4", "sensitivity_v1_k16"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(repo):
    ext = repo / "extensions/certified_forecasting"
    files = list(repo.glob("*.py"))
    files += list((ext / "src").rglob("*.py"))
    files += list((ext / "experiments").glob("*.py"))
    return {str(p.relative_to(repo)): digest(p) for p in sorted(files)}


def get_protocol(repo):
    ext = repo / "extensions/certified_forecasting"
    configs = {name: json.loads((ext / f"configs/{name}.json").read_text())
               for name in CAMPAIGNS + ["pendigits_v1"]}
    return {
        "study": "KAHM input scaling paired follow-up, 29 October 2026",
        "registered_before_new_results": True,
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "modes": {
            "standard": "Published training mean/std preprocessing, unchanged.",
            "minmax": "Primary contrast: featurewise training min/max maps to [-1,1]; no clipping of unseen extremes.",
            "minmax_clip": "Prespecified bounded variant: same training min/max, clip all queries to [-1,1].",
            "raw": "Prespecified control: existing mathematically bounded raw histories; no standardization.",
        },
        "scope": "Change only KAHM banks and dependent direct/hard/Frobenius/duplicate/Koopman models; all other features and targets unchanged.",
        "primary_family": "kahm_nested",
        "primary_outcome": "Mean held-out MSE over all three horizons after within-arm validation selection, summarized separately for each task.",
        "secondary_outcomes": [
            "Per-horizon RMSE and between-fitting-seed SD",
            "Other KAHM model families",
            "Coefficient-budget sensitivity at kappa 1, 4, and 16",
            "Conditional fitted-model paired episode-loss differences",
            "Existing within-arm risk endpoints and decisions, descriptive across preprocessing arms",
        ],
        "selection": "Use the manuscript's validation-only selection separately in each arm. Never select history, prefix, or scaling arm by test scores.",
        "aggregation": "Report per-task effects. An equal-task mean relative MSE change is descriptive, not a pooled physical risk. Treat PenDigits as one coordinate-pooled task.",
        "change_gate": "No manuscript or GitHub change without a reproducible, supported improvement in forecasting. Mixed task effects do not establish a universal improvement. Any proposed promotion requires a fresh confirmation campaign with preprocessing fixed before new evaluation.",
        "statistical_scope": "A post-publication-design follow-up. Replaying old streams is a matched descriptive comparison, not a fresh untouched test claim. Original risk deltas apply within each frozen arm, not jointly over all scaling arms. PenDigits writers are a fixed cohort; training seeds are not new test cohorts.",
        "training_only_scaler": "Fit on all training history rows before the original seeded geometry subsample; save parameters. Constant coordinates map to zero.",
        "preserved": ["all original seeds and streams", "train/validation/calibration/test sizes",
                      "causal histories and origins", "horizons", "target units and output clipping",
                      "kappa", "ridge", "omega", "cluster counts", "geometry sample budget",
                      "local KAHM training-point budget", "decoder solver", "non-KAHM controls"],
        "excluded_from_rerun": "Analytical two-atom known-risk and exact duplicate-invariance illustrations do not train KAHM input geometry; no scaling intervention applies.",
        "configurations": configs,
    }


def run_worker(args):
    ext = args.repo / "extensions/certified_forecasting"
    sys.path[:0] = [str(ext / "src"), str(ext / "experiments"), str(ROOT)]
    import run_forecasting as runner
    import scaling_features
    import numpy as np
    import scipy
    import sklearn

    cfg_path = ext / f"configs/{args.campaign}.json"
    cfg = json.loads(cfg_path.read_text())
    destination = args.output / args.mode / args.campaign / f"{args.system}_{args.seed}"
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True)
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
        "config_sha256": digest(cfg_path),
        "protocol_sha256": digest(ROOT / "protocol.json"),
        "harness_sha256": digest(__file__),
        "scaler_sha256": digest(ROOT / "scaling_features.py"),
        "source_sha256": source_hashes(args.repo),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "python": sys.version,
        "platform": platform.platform(),
        "thread_limits": {n: os.environ.get(n) for n in
                          ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(destination / "intervention_manifest.json", metadata)
    try:
        record = runner.run_one(cfg, args.system, args.seed, destination)
        banks = getattr(scaling_features, "FITTED_BANKS", [])
        diagnostics = [b.scaling_metadata() for b in banks
                       if hasattr(b, "scaling_metadata")]
        write_json(destination / "scaling_diagnostics.json", diagnostics)
        print(json.dumps({"mode": args.mode, "campaign": args.campaign,
                          "system": args.system, "seed": args.seed,
                          "candidates": record["n_candidates"],
                          "seconds": record["elapsed_seconds"]}), flush=True)
    except Exception as exc:
        write_json(destination / "failure.json",
                   {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


def orchestrate(args):
    ext = args.repo / "extensions/certified_forecasting"
    protocol_path = ROOT / "protocol.json"
    proposed = get_protocol(args.repo)
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != proposed:
            raise RuntimeError("Protocol changed; preserve old evidence and start a separately declared study.")
    else:
        write_json(protocol_path, proposed)
    all_jobs = []
    for campaign in args.campaigns:
        cfg = proposed["configurations"][campaign]
        for system in cfg["systems"]:
            for seed in cfg["seeds"]:
                for mode in args.modes:
                    all_jobs.append((mode, campaign, system, seed))
    jobs = []
    for job in all_jobs:
        mode, campaign, system, seed = job
        dest = args.output / mode / campaign / f"{system}_{seed}"
        if args.resume and (dest / "results.json").exists() and (dest / "episode_scores.npz").exists():
            continue
        if dest.exists():
            raise RuntimeError(f"Existing incomplete or non-resumable job: {dest}")
        jobs.append(job)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(protocol_path),
        "harness_sha256": digest(__file__),
        "scaler_sha256": digest(ROOT / "scaling_features.py"),
        "source_sha256": source_hashes(args.repo),
        "jobs": all_jobs, "new_jobs": len(jobs), "workers": args.workers,
        "cpu_count": os.cpu_count(),
    }
    manifest_path = args.output / "study_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ["protocol_sha256", "harness_sha256", "scaler_sha256", "source_sha256"]:
            if old[key] != manifest[key]:
                raise RuntimeError(f"Changed {key}; refusing to mix evidence")
    else:
        write_json(manifest_path, manifest)
    logroot = args.output / "logs"
    logroot.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONHASHSEED="0")

    def launch(job):
        mode, campaign, system, seed = job
        logfile = logroot / f"{mode}__{campaign}__{system}_{seed}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--repo", str(args.repo), "--output", str(args.output),
                   "--mode", mode, "--campaign", campaign,
                   "--system", system, "--seed", str(seed)]
        started = time.perf_counter()
        with logfile.open("x") as stream:
            result = subprocess.run(command, cwd=ext, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        return {"job": job, "returncode": result.returncode,
                "seconds": time.perf_counter() - started, "log": str(logfile)}

    completed = []
    started = time.perf_counter()
    print(f"START {len(jobs)} new jobs; {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(launch, job): job for job in jobs}
        for future in as_completed(futures):
            item = future.result()
            completed.append(item)
            print(f"DONE {len(completed)}/{len(jobs)} {item['job']} rc={item['returncode']} seconds={item['seconds']:.1f}", flush=True)
            write_json(args.output / "progress.json", {
                "completed": completed,
                "elapsed_seconds": time.perf_counter() - started,
                "scheduled": len(jobs),
            })
    failures = [item for item in completed if item["returncode"]]
    write_json(args.output / "completion.json", {
        "completed": completed, "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "all_scheduled_jobs": all_jobs,
    })
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; logs and attempts retained")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--campaigns", nargs="+", choices=CAMPAIGNS, default=CAMPAIGNS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--campaign", choices=CAMPAIGNS)
    p.add_argument("--system")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    if args.worker:
        if any(v is None for v in [args.mode, args.campaign, args.system, args.seed]):
            p.error("Workers require mode, campaign, system, and seed")
        run_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
