#!/usr/bin/env python3
"""Reproduce the paired KAHM input-scaling and confirmation schedules.

Original includes main, oscillator, budget-sensitivity and PenDigits campaigns.
Confirmation uses main, oscillators and PenDigits with fitting seeds 4701--4703.
Synthetic confirmation regenerates every episode stream; PenDigits reuses the
same writers and changes fitting randomness only. Scaling is passed explicitly
through each frozen configuration. Every output root must be new.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy
import sklearn

import run_forecasting as forecasting
from run_oscillators import check_integrator, oscillator_episodes
from run_pendigits import run as run_pendigits

MODES = ("standard", "minmax", "minmax_clip", "raw")
CAMPAIGNS = ("main_v1", "oscillators_v1", "sensitivity_v1_k1",
             "sensitivity_v1_k4", "sensitivity_v1_k16", "pendigits_v1")
CONFIRMATION_CAMPAIGNS = ("main_v1", "oscillators_v1", "pendigits_v1")
CONFIRMATION_SEEDS = [4701, 4702, 4703]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    sources = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "experiments").glob("*.py"))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    hashes.update({"parent/" + path.name: digest(path) for path in sorted(PARENT.glob("*.py"))})
    return hashes


def configurations(args):
    """Derive in-memory copies; custom --config-dir is useful for smoke runs."""
    planned = []
    for phase in args.phases:
        available = CAMPAIGNS if phase == "original" else CONFIRMATION_CAMPAIGNS
        for campaign in args.campaigns:
            if campaign not in available:
                continue
            base_path = args.config_dir / (campaign + ".json")
            original = json.loads(base_path.read_text())
            for mode in args.modes:
                cfg = json.loads(json.dumps(original))
                cfg["kahm_input_scaling"] = mode
                if phase == "confirmation":
                    cfg["campaign"] = "confirmation_" + campaign
                    cfg["seeds"] = list(CONFIRMATION_SEEDS)
                planned.append({"phase": phase, "campaign": campaign, "mode": mode,
                                "configuration": cfg, "source_config": str(base_path),
                                "source_config_sha256": digest(base_path),
                                "matches_reference_config": digest(base_path) == digest(ROOT / "configs" / base_path.name)})
    if not planned:
        raise ValueError("No campaigns remain in the requested phase schedule.")
    return planned


def freeze(args, planned):
    """Write configurations, job descriptions and source hashes before fits."""
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PARENT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    manifest = {
        "schema": "kahm-input-scaling-study-v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "base_git_commit": commit, "source_sha256": hashes,
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "workers": args.workers, "modes": args.modes, "phases": args.phases,
        "primary_contrast": "minmax versus standard; kahm_nested selected by validation only",
        "scope": "Only KAHM input preprocessing changes between paired arms; targets and non-KAHM controls are unchanged.",
        "confidence_scope": "Inherited risk allocations apply within each arm and replicate, not jointly across preprocessing arms or the entire study.",
        "confirmation_scope": "New synthetic episode streams; new PenDigits fitting seeds on the same writer cohort, not new-writer replication.",
        "all_base_configs_match_reference": all(item["matches_reference_config"] for item in planned),
    }
    if any(item["campaign"] == "pendigits_v1" for item in planned):
        manifest["prepared_pendigits"] = {"path": str(args.prepared), "sha256": digest(args.prepared),
                                         "manifest_sha256": digest(args.prepared.with_name("manifest.json"))}
    forecasting.write_json(args.output / "manifest.json", manifest)
    jobs, numerical_cache = [], {}
    for item in planned:
        phase, mode, campaign = (item[key] for key in ("phase", "mode", "campaign"))
        cfg = item["configuration"]
        config_file = args.output / "frozen_configs" / phase / mode / (campaign + ".json")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        forecasting.write_json(config_file, cfg)
        campaign_dir = args.output / ("results" if phase == "original" else "confirmation_results") / mode / campaign
        campaign_manifest = {**manifest, **item, "configuration_sha256": digest(config_file)}
        outside_manifest = args.output / "frozen_manifests" / phase / mode / (campaign + ".json")
        outside_manifest.parent.mkdir(parents=True, exist_ok=True)
        forecasting.write_json(outside_manifest, campaign_manifest)
        if campaign != "pendigits_v1":
            campaign_dir.mkdir(parents=True, exist_ok=False)
            forecasting.write_json(campaign_dir / "manifest.json", campaign_manifest)
        if campaign == "oscillators_v1":
            numerical_key = json.dumps([cfg["generator"], cfg["systems"], cfg["episode_length"]], sort_keys=True)
            if numerical_key not in numerical_cache:
                numerical_cache[numerical_key] = check_integrator(cfg)
            numerical = numerical_cache[numerical_key]
            forecasting.write_json(campaign_dir / "numerical_check.json", numerical)
            if not numerical["all_passed"]:
                raise RuntimeError("The frozen oscillator half-step consistency check failed.")
        members = [(None, None)] if campaign == "pendigits_v1" else [
            (system, seed) for system in cfg["systems"] for seed in cfg["seeds"]]
        for system, seed in members:
            member = "cohort" if system is None else f"{system}_{seed}"
            identity = f"{phase}/{mode}/{campaign}/{member}"
            destination = campaign_dir if system is None else campaign_dir / member
            jobs.append({"id": identity, "phase": phase, "mode": mode, "campaign": campaign,
                         "system": system, "seed": seed, "destination": str(destination),
                         "config_file": str(config_file), "config_sha256": digest(config_file),
                         "prepared": str(args.prepared), "study_manifest": str(args.output / "manifest.json"),
                         "log": str(args.output / "logs" / phase / mode / campaign / (member + ".log"))})
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate systems/seeds would repeat a job; each config schedule must be unique.")
    forecasting.write_json(args.output / "schedule.json", jobs)
    return jobs


def run_job(job):
    """One process-local run, with original fitting/evaluation routines."""
    started = time.perf_counter()
    destination, log = Path(job["destination"]), Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    status = {"id": job["id"], "log": str(log), "destination": str(destination)}
    with log.open("x") as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            manifest = json.loads(Path(job["study_manifest"]).read_text())
            if source_hashes() != manifest["source_sha256"]:
                raise RuntimeError("Source changed after freezing; use a new output directory.")
            config_path = Path(job["config_file"])
            if digest(config_path) != job["config_sha256"]:
                raise RuntimeError("Frozen job configuration changed.")
            cfg = json.loads(config_path.read_text())
            if job["campaign"] == "pendigits_v1":
                prepared = Path(job["prepared"])
                frozen_data = manifest["prepared_pendigits"]
                if (digest(prepared) != frozen_data["sha256"]
                        or digest(prepared.with_name("manifest.json")) != frozen_data["manifest_sha256"]):
                    raise RuntimeError("Prepared PenDigits data changed after freezing.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                summary = run_pendigits(config_path, prepared, destination)
                count = summary["n_candidates_per_coordinate_seed"] * len(cfg["coordinates"]) * len(cfg["seeds"])
            else:
                destination.mkdir(exist_ok=False)
                previous_generator = forecasting.make_episodes
                try:
                    if job["campaign"] == "oscillators_v1":
                        settings = cfg["generator"]
                        forecasting.make_episodes = partial(
                            oscillator_episodes, dt=settings["dt"],
                            observation_interval=settings["observation_interval"], burn_time=settings["burn_time"])
                    record = forecasting.run_one(cfg, job["system"], job["seed"], destination)
                finally:
                    forecasting.make_episodes = previous_generator
                count = record["n_candidates"]
            if source_hashes() != manifest["source_sha256"]:
                raise RuntimeError("Source changed during a job; results are retained as a failed run.")
            status.update(status="completed", n_candidates=count)
        except Exception as error:
            status.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            # Partial artifacts, including .tmp archives, remain available.
            destination.mkdir(parents=True, exist_ok=True)
            forecasting.write_json(destination / "failure.json", status)
            print(status["traceback"], flush=True)
    status["elapsed_seconds"] = time.perf_counter() - started
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New study root; existing paths are refused")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data_cache/pendigits/pendigits_original.npz")
    parser.add_argument("--campaigns", nargs="+", choices=CAMPAIGNS, default=list(CAMPAIGNS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--phases", nargs="+", choices=("original", "confirmation"), default=["original", "confirmation"])
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs", help="Alternate config copies, e.g. reduced smoke inputs")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in ("campaigns", "modes", "phases"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"Duplicate --{name} values would repeat a run")
    args.output, args.prepared, args.config_dir = (path.resolve() for path in (args.output, args.prepared, args.config_dir))
    if args.output.exists():
        parser.error("--output must be a new path; existing runs are never overwritten")
    planned = configurations(args)
    if any(item["campaign"] == "pendigits_v1" for item in planned):
        if not args.prepared.is_file() or not args.prepared.with_name("manifest.json").is_file():
            parser.error("PenDigits needs --prepared and its manifest; run prepare_pendigits.py first")
    try:
        jobs = freeze(args, planned)
    except Exception as error:
        if args.output.exists():
            forecasting.write_json(args.output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise
    print(f"FROZEN {len(jobs)} jobs; workers={args.workers}; output={args.output}", flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"id": job["id"], "status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
                failure_dir = args.output / "failures" / job["phase"] / job["mode"] / job["campaign"]
                failure_dir.mkdir(parents=True, exist_ok=True)
                forecasting.write_json(failure_dir / (Path(job["destination"]).name + ".json"), result)
            records.append(result)
            print(f"{result['status'].upper()} {job['id']} ({len(records)}/{len(jobs)})", flush=True)
    failed = sum(record["status"] != "completed" for record in records)
    forecasting.write_json(args.output / "completion.json", {
        "scheduled_jobs": len(jobs), "completed_jobs": len(records) - failed,
        "failed_jobs": failed, "results": sorted(records, key=lambda row: row["id"]),
    })
    if failed:
        raise SystemExit(f"{failed} jobs failed; all results and failures remain in {args.output}")


if __name__ == "__main__":
    main()
