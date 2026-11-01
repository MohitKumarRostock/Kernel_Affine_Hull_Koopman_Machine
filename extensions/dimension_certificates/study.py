#!/usr/bin/env python3
"""Frozen development, calibration, and test stages for dimension policies."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

from common import (ROOT, digest, read_json, schedule, sources, stage_job_files, utc,
                    verify_freeze, verify_seal, write_json)
import numpy as np


def freeze(output, preflight):
    import scipy
    import sklearn
    cfg = read_json(ROOT / "config.json")
    receipt = read_json(preflight)
    if receipt.get("passed") is not True or receipt.get("source_sha256") != sources():
        raise RuntimeError("A passed independent preflight for these exact sources is required")
    jobs = schedule(cfg)
    if cfg["campaign_risk_tails"] != len(jobs) * cfg["risk_tails_per_repetition"]:
        raise AssertionError("Incorrect risk allocation count")
    if cfg["campaign_improvement_tails"] != len(jobs) * cfg["improvement_tails_per_repetition"]:
        raise AssertionError("Incorrect improvement allocation count")
    if cfg["campaign_total_tails"] != cfg["campaign_risk_tails"] + cfg["campaign_improvement_tails"]:
        raise AssertionError("Incorrect campaign allocation count")
    copied = read_json(ROOT / "runtime_sources.json")["files"]
    for name, record in copied.items():
        if digest(ROOT / name) != record["sha256"]:
            raise RuntimeError(f"Verified runtime copy changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "jobs").mkdir()
    (output / "logs").mkdir()
    shutil.copyfile(ROOT / "config.json", output / "config.json")
    shutil.copyfile(ROOT / "PROTOCOL.md", output / "PROTOCOL.md")
    shutil.copyfile(preflight, output / "preflight.json")
    write_json(output / "schedule.json", jobs)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(output / "manifest.json", {
        "schema": "kahm-dimension-certificates-freeze-v1", "frozen_utc": utc(),
        "base_git_commit": commit, "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "n_jobs": len(jobs), "stage_order": ["development", "calibration", "test"],
        "stage_scope": "All scheduled jobs are sealed globally before the next stage begins",
        "total_one_sided_endpoints": cfg["campaign_total_tails"],
        "delta": cfg["delta"], "delta_per_designated_endpoint": cfg["delta"] / cfg["campaign_total_tails"],
        "full_prediction_audit_jobs": [job["id"] for job in jobs if job["full_prediction_audit"]],
        "independent_preflight_sha256": digest(output / "preflight.json"),
    })
    write_json(output / "freeze.json", {
        "source_sha256": sources(),
        "files": {name: digest(output / name) for name in
                  ("config.json", "PROTOCOL.md", "schedule.json", "manifest.json", "preflight.json")},
    })
    return {"output": str(output), "jobs": len(jobs), "freeze_sha256": digest(output / "freeze.json")}


def seal(output, stage, jobs):
    files = {}
    for job in jobs:
        folder = output / "jobs" / job["id"]
        for name in stage_job_files(stage, job):
            file = folder / name
            if not file.is_file():
                raise RuntimeError(f"Incomplete stage artifact: {file}")
            files[str(file.relative_to(output))] = digest(file)
    write_json(output / (stage + "_seal.json"), {
        "stage": stage, "sealed_utc": utc(), "n_jobs": len(jobs),
        "freeze_sha256": digest(output / "freeze.json"), "files": files,
        "preceding_seal_sha256": (None if stage == "development" else
             digest(output / (("development" if stage == "calibration" else "calibration") + "_seal.json"))),
    })


def execute_job(output, stage, job, cfg):
    folder = output / "jobs" / job["id"]
    if stage == "development":
        folder.mkdir(exist_ok=False)
    log = output / "logs" / (job["id"] + "_" + stage + ".log")
    write_json(folder / (stage + "_started.json"), {"started_utc": utc(), "job": job["id"]})
    with log.open("x") as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            if stage == "development":
                from fit_models import fit_development
                result = fit_development(job, cfg, folder)
            else:
                from score_models import evaluate
                result = evaluate(job, cfg, folder, stage)
            write_json(folder / (stage + "_complete.json"), result)
            return result
        except BaseException as error:
            write_json(folder / (stage + "_failure.json"), {
                "error": repr(error), "traceback": traceback.format_exc(), "failed_utc": utc()})
            raise


def run_stage(output, stage, workers):
    cfg, jobs = verify_freeze(output)
    if (output / (stage + "_seal.json")).exists():
        raise RuntimeError("Stage is already sealed; refusing to rerun")
    if stage != "development":
        verify_seal(output, "development")
    if stage == "test":
        verify_seal(output, "calibration")
    # Refuse partial reruns: every successful and failed attempt stays visible.
    for job in jobs:
        folder = output / "jobs" / job["id"]
        if (folder / (stage + "_started.json")).exists():
            raise RuntimeError(f"An attempt already exists for {stage}/{job['id']}")
    if workers < 1:
        raise ValueError("Workers must be positive")
    if workers == 1:
        for job in jobs:
            print(execute_job(output, stage, job, cfg), flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(execute_job, output, stage, job, cfg): job["id"] for job in jobs}
            for future in as_completed(futures):
                print(future.result(), flush=True)
    verify_freeze(output)
    seal(output, stage, jobs)
    print({"stage": stage, "sealed": True, "jobs": len(jobs)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "develop", "calibrate", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preflight", type=Path, help="Passed independent audit receipt, required for freeze")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "freeze":
        if args.preflight is None:
            parser.error("freeze requires --preflight")
        print(freeze(output, args.preflight.resolve()), flush=True)
    else:
        stage = {"develop": "development", "calibrate": "calibration", "test": "test"}[args.stage]
        run_stage(output, stage, args.workers)


if __name__ == "__main__":
    main()
