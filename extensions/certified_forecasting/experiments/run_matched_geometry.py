#!/usr/bin/env python3
"""Freeze and run the complete matched-geometry follow-up.

Use --freeze-only to review an immutable schedule, then --resume to execute it.
Resume verifies source, protocol, configuration, data and completed artifacts;
it never overwrites completed, failed or partially written attempts. Custom
--config-dir copies support separate, explicitly non-reference smoke runs.
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
import resource
import subprocess
import sys
import time
import traceback
import zipfile

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
KINDS = ("kahm", "rbf", "affine")
CONFIGS = {"main_v1": "matched_geometry_main.json",
           "oscillators_v1": "matched_geometry_oscillators.json",
           "pendigits_v1": "matched_geometry_pendigits.json"}
PROTOCOL = ROOT / "docs/MATCHED_GEOMETRY_PROTOCOL.md"
PREPARED_SHA256 = "320aceb0f7f473f8eb21efe6f20bbf87a62f729e0523ba7a60885b073d1e45c2"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "experiments").glob("*.py"))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    hashes.update({"parent/" + path.name: digest(path) for path in sorted(PARENT.glob("*.py"))})
    return hashes


def candidate_count(cfg):
    per_history = 3 * len(cfg["feature_kinds"]) * len(cfg["cluster_counts"]) + 3
    per_history += 2 if "kahm" in cfg["feature_kinds"] else 0
    return len(cfg["histories"]) * per_history + 2


def configurations(args):
    planned = []
    for campaign in args.campaigns:
        path = args.config_dir / CONFIGS[campaign]
        base = json.loads(path.read_text())
        if not base.get("matched_geometry_heads") or tuple(base["feature_kinds"]) != KINDS:
            raise ValueError("Matched configurations must include all three geometries and heads")
        for field in ("seeds", "histories", "cluster_counts", "horizons"):
            if not base[field] or len(set(base[field])) != len(base[field]):
                raise ValueError(f"{campaign}: {field} must be nonempty and unique")
        if campaign != "pendigits_v1":
            if len(set(base["systems"])) != len(base["systems"]):
                raise ValueError("Repeated systems would duplicate a scheduled job")
            if base["delta"] != 0.01 or base["risk_delta_fraction"] != 0.8:
                raise ValueError("Synthetic matched arms require delta=.01 and risk fraction=.8")
        elif base["delta"] != 0.0125 or len(base["seeds"]) < 2:
            raise ValueError("PenDigits requires delta=.0125 and at least two fitting seeds")
        for mode in args.modes:
            cfg = json.loads(json.dumps(base))
            cfg["input_scaling_by_kind"] = {kind: mode for kind in KINDS}
            planned.append({"campaign": campaign, "mode": mode, "configuration": cfg,
                            "source_config": str(path), "source_config_sha256": digest(path),
                            "matches_reference_config": digest(path) == digest(ROOT / "configs" / path.name)})
    return planned


def freeze(args, planned):
    """Write every protocol/configuration/job byte before any model is fitted."""
    args.output.mkdir(parents=True, exist_ok=False)
    protocol_copy = args.output / "protocol.md"
    protocol_copy.write_bytes(PROTOCOL.read_bytes())
    frozen_files = {"protocol.md": digest(protocol_copy)}
    sources = source_hashes()
    jobs, numerical_cache = [], {}
    for item in planned:
        campaign, mode, cfg = (item[key] for key in ("campaign", "mode", "configuration"))
        path = args.output / "frozen_configs" / mode / (campaign + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        forecasting.write_json(path, cfg)
        frozen_files[str(path.relative_to(args.output))] = digest(path)
        campaign_dir = args.output / "results" / mode / campaign
        if campaign == "oscillators_v1":
            key = json.dumps([cfg["generator"], cfg["systems"], cfg["episode_length"]], sort_keys=True)
            if key not in numerical_cache:
                numerical_cache[key] = check_integrator(cfg)
            if not numerical_cache[key]["all_passed"]:
                raise RuntimeError("The oscillator half-step consistency check failed")
            check_path = args.output / "numerical_checks" / (mode + ".json")
            check_path.parent.mkdir(parents=True, exist_ok=True)
            forecasting.write_json(check_path, numerical_cache[key])
            frozen_files[str(check_path.relative_to(args.output))] = digest(check_path)
        members = [(None, None)] if campaign == "pendigits_v1" else [
            (system, seed) for system in cfg["systems"] for seed in cfg["seeds"]]
        for system, seed in members:
            member = "cohort" if system is None else f"{system}_{seed}"
            identity = f"{mode}/{campaign}/{member}"
            jobs.append({"id": identity, "mode": mode, "campaign": campaign,
                         "system": system, "seed": seed,
                         "destination": str(campaign_dir if system is None else campaign_dir / member),
                         "config_file": str(path), "config_sha256": digest(path),
                         "prepared": str(args.prepared), "study_root": str(args.output),
                         "log": str(args.output / "logs" / mode / campaign / (member + ".log")),
                         "status_file": str(args.output / "job_status" / mode / campaign / (member + ".json")),
                         "expected_candidates": candidate_count(cfg) * (
                             len(cfg["seeds"]) * len(cfg["coordinates"]) if system is None else 1)})
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate scheduled job identities")
    protocol = {
        "schema": "kahm-matched-geometry-protocol-v1", "protocol_sha256": digest(PROTOCOL),
        "modes": list(args.modes), "campaigns": list(args.campaigns),
        "selection_order": ["mean validation MSE", "n_features", "history",
                            "mode order: standard, minmax, minmax_clip, raw", "candidate name"],
        "synthetic_joint_delta_per_system_seed": 0.05,
        "synthetic_complete_family_delta_per_arm": 0.01,
        "synthetic_cross_geometry_delta": 0.01,
        "n_primary_cross_geometry_comparisons": 6,
        "pendigits_joint_delta_conditional_on_iid_writers": 0.05,
        "pendigits_delta_per_arm_across_all_seeds_coordinates": 0.0125,
        "pendigits_scope": "Same held-out writers; fitting-seed variation, not a new cohort",
        "scheduled_jobs": len(jobs),
        "scheduled_candidate_fits": sum(job["expected_candidates"] for job in jobs),
        "timing_scope": "Head fitting excludes validation/scoring; shared geometry is charged once per bank; no deployment-runtime claim",
        "reference_design": all(item["matches_reference_config"] for item in planned)
                            and tuple(args.modes) == MODES and tuple(args.campaigns) == tuple(CONFIGS),
    }
    forecasting.write_json(args.output / "protocol.json", protocol)
    forecasting.write_json(args.output / "schedule.json", jobs)
    for name in ("protocol.json", "schedule.json"):
        frozen_files[name] = digest(args.output / name)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PARENT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    manifest = {
        "schema": "kahm-matched-geometry-study-v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "base_git_commit": commit, "source_sha256": sources,
        "protocol_sha256": digest(PROTOCOL), "machine_protocol_sha256": frozen_files["protocol.json"],
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "workers_at_freeze": args.workers, "planned": planned,
        "prepared_path": str(args.prepared),
        "native_pendigits_summary_scope": "Historical 12-family summary; matched_summary.json and raw result/CSV records include all 16 families",
    }
    if any(item["campaign"] == "pendigits_v1" for item in planned):
        prepared_hash = digest(args.prepared)
        if prepared_hash != PREPARED_SHA256:
            raise ValueError("Prepared PenDigits bytes do not match the pinned original array")
        manifest["prepared_pendigits"] = {
            "sha256": prepared_hash, "manifest_sha256": digest(args.prepared.with_name("manifest.json"))}
    forecasting.write_json(args.output / "manifest.json", manifest)
    frozen_files["manifest.json"] = digest(args.output / "manifest.json")
    forecasting.write_json(args.output / "freeze_sha256.json", frozen_files)
    return jobs


def validate_freeze(output, planned=None):
    """Revalidate frozen inputs without modifying or repairing them."""
    output = Path(output)
    frozen = json.loads((output / "freeze_sha256.json").read_text())
    for name, expected in frozen.items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen file changed: {name}")
    manifest = json.loads((output / "manifest.json").read_text())
    if source_hashes() != manifest["source_sha256"]:
        raise RuntimeError("Source changed after freezing; use a new output root")
    if digest(PROTOCOL) != manifest["protocol_sha256"]:
        raise RuntimeError("The authoritative study protocol changed after freezing")
    if planned is not None and planned != manifest["planned"]:
        raise RuntimeError("Requested configurations or schedule differ from the frozen study")
    if "prepared_pendigits" in manifest:
        prepared = Path(manifest["prepared_path"])
        data = manifest["prepared_pendigits"]
        if digest(prepared) != data["sha256"] or digest(prepared.with_name("manifest.json")) != data["manifest_sha256"]:
            raise RuntimeError("Prepared data or its manifest changed after freezing")
    return json.loads((output / "schedule.json").read_text())


def verify_score_archive(destination, record, cfg):
    path = destination / "episode_scores.npz"
    with zipfile.ZipFile(path) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise RuntimeError(f"Damaged score archive member: {damaged}")
    expected = {row["name"] + "__" + suffix for row in record["results"]
                for suffix in ("cal_loss", "pair_score", "test_loss")}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != expected:
            raise RuntimeError("Score archive members differ from the frozen candidate family")
        for row in record["results"]:
            for suffix, n in (("cal_loss", cfg["n_calibration"]),
                              ("pair_score", cfg["n_calibration"] // 2), ("test_loss", cfg["n_test"])):
                values = saved[row["name"] + "__" + suffix]
                if values.shape != (n, len(cfg["horizons"])) or not np.isfinite(values).all():
                    raise RuntimeError("Score archive has an invalid array shape or nonfinite values")
                if suffix == "test_loss":
                    np.testing.assert_array_equal(values.mean(axis=0), row["test_mse"])


def complete_pendigits_summary(destination):
    """Summarize all validation-selected model families with pooled coordinate errors."""
    record = json.loads((destination / "results.json").read_text())
    cfg = record["configuration"]
    selected = [row for row in record["results"] if row["selected_before_test"]]
    families = []
    for family in sorted({row["family"] for row in selected}):
        rows = [row for row in selected if row["family"] == family]
        per_seed = [np.sqrt(np.mean([row["cohort_writer_mean_mse"] for row in rows
                                    if row["seed"] == seed], axis=0)) for seed in cfg["seeds"]]
        families.append({"family": family,
                         "mean_coordinate_pooled_rmse_over_seeds": np.mean(per_seed, axis=0).tolist(),
                         "sd_coordinate_pooled_rmse_over_seeds": np.std(per_seed, axis=0, ddof=1).tolist(),
                         "mean_selected_features": float(np.mean([row["n_features"] for row in rows])),
                         "selected_names": [row["name"] for row in rows]})
    forecasting.write_json(destination / "matched_summary.json", {
        "families": families, "horizons": cfg["horizons"], "seeds": cfg["seeds"],
        "scope": "Native per-arm, validation-selected families on the same held-out writer cohort; final cross-arm selection is separate"})
    return record


def run_job(job):
    started = time.perf_counter()
    destination, log = Path(job["destination"]), Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    status = {"id": job["id"], "destination": str(destination), "log": str(log)}
    with log.open("x") as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            validate_freeze(job["study_root"])
            config_path = Path(job["config_file"])
            cfg = json.loads(config_path.read_text())
            if job["campaign"] == "pendigits_v1":
                destination.parent.mkdir(parents=True, exist_ok=True)
                summary = run_pendigits(config_path, Path(job["prepared"]), destination)
                count = summary["n_candidates_per_coordinate_seed"] * len(cfg["seeds"]) * len(cfg["coordinates"])
                complete_pendigits_summary(destination)
            else:
                destination.mkdir(parents=True, exist_ok=False)
                original_generator = forecasting.make_episodes
                try:
                    if job["campaign"] == "oscillators_v1":
                        settings = cfg["generator"]
                        forecasting.make_episodes = partial(oscillator_episodes, dt=settings["dt"],
                            observation_interval=settings["observation_interval"], burn_time=settings["burn_time"])
                    record = forecasting.run_one(cfg, job["system"], job["seed"], destination)
                finally:
                    forecasting.make_episodes = original_generator
                count = record["n_candidates"]
                verify_score_archive(destination, record, cfg)
            if count != job["expected_candidates"]:
                raise RuntimeError("Completed candidate count differs from the frozen schedule")
            validate_freeze(job["study_root"])
            status.update(status="completed", n_candidates=count,
                          artifact_sha256={str(path.relative_to(destination)): digest(path)
                                           for path in sorted(destination.rglob("*")) if path.is_file()})
        except Exception as error:
            status.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            destination.mkdir(parents=True, exist_ok=True)
            forecasting.write_json(destination / "failure.json", status)
            print(status["traceback"], flush=True)
    status.update(elapsed_seconds=time.perf_counter() - started,
                  worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    status_path = Path(job["status_file"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    forecasting.write_json(status_path, status)
    return status


def resume_state(jobs):
    records, pending = [], []
    for job in jobs:
        status_path, destination = Path(job["status_file"]), Path(job["destination"])
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["id"] != job["id"]:
                raise RuntimeError("A saved job status has an incompatible identity")
            if status["status"] == "completed":
                actual = {str(path.relative_to(destination)): digest(path)
                          for path in sorted(destination.rglob("*")) if path.is_file()}
                if actual != status["artifact_sha256"]:
                    raise RuntimeError(f"Completed artifacts changed: {job['id']}")
            records.append(status)
        elif destination.exists() or Path(job["log"]).exists():
            raise RuntimeError(f"Uncommitted partial attempt is preserved for review: {job['id']}; use a new output root")
        else:
            pending.append(job)
    return records, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data_cache/pendigits/pendigits_original.npz")
    parser.add_argument("--campaigns", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Revalidate immutable inputs/results and run only jobs with no prior attempt")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in ("campaigns", "modes"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate --{name} values")
    args.output, args.prepared, args.config_dir = (path.resolve() for path in (args.output, args.prepared, args.config_dir))
    if args.output.exists() != args.resume:
        parser.error("An existing output requires --resume; a new output must omit --resume")
    planned = configurations(args)
    if args.resume:
        jobs = validate_freeze(args.output, planned)
    else:
        try:
            jobs = freeze(args, planned)
        except Exception as error:
            if args.output.exists():
                forecasting.write_json(args.output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
            raise
    print(f"FROZEN {len(jobs)} jobs and {sum(job['expected_candidates'] for job in jobs)} candidates; output={args.output}", flush=True)
    if args.freeze_only:
        return
    records, pending = resume_state(jobs)
    lock = args.output / "RUNNING.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    status = future.result()
                except Exception as error:
                    status = {"id": job["id"], "status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
                    path = Path(job["status_file"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    forecasting.write_json(path, status)
                records.append(status)
                print(f"{status['status'].upper()} {job['id']} ({len(records)}/{len(jobs)})", flush=True)
        failed = sum(row["status"] != "completed" for row in records)
        forecasting.write_json(args.output / "completion.json", {
            "scheduled_jobs": len(jobs), "completed_jobs": len(records) - failed, "failed_jobs": failed,
            "workers_this_execution": args.workers, "results": sorted(records, key=lambda row: row["id"])})
    finally:
        lock.unlink()
    if failed:
        raise SystemExit(f"{failed} jobs failed; attempts and diagnostics remain in {args.output}")


if __name__ == "__main__":
    main()
