#!/usr/bin/env python3
"""Validate confirmation score archives; exactly replay corrupt artifacts only.

The frozen confirmation runner, preprocessing, protocol, seeds, and outcomes
are unchanged. Original corrupted bytes and replay provenance are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "confirmation_results"
REPAIRS = ROOT / "confirmation_repairs"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def without_timings(value):
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items()
                if key not in {"train_seconds", "elapsed_seconds"}}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


def validate_archive(path):
    try:
        if not zipfile.is_zipfile(path):
            return {"valid": False, "reason": "Not a complete ZIP archive"}
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                return {"valid": False, "reason": f"CRC failed: {bad}"}
        count = 0
        with np.load(path, allow_pickle=False) as arrays:
            if len(arrays.files) != 120:
                return {"valid": False, "reason": f"Expected120 arrays, got{len(arrays.files)}"}
            for name in arrays.files:
                data = arrays[name]
                if name.endswith("__cal_loss"):
                    shape, bound = (4096, 3), 4.0
                elif name.endswith("__pair_score"):
                    # Inherited witness is (target gap - kappa*distance)_+^2 / 2.
                    shape, bound = (2048, 3), 2.0
                elif name.endswith("__test_loss"):
                    shape, bound = (2048, 3), 4.0
                else:
                    return {"valid": False, "reason": f"Unexpected array{name}"}
                if (data.shape != shape or data.dtype != np.dtype("float64") or
                        not np.isfinite(data).all() or data.min() < 0.0 or data.max() > bound + 1e-12):
                    return {"valid": False, "reason": f"Invalid shape/type/bounds/finite values:{name}"}
                count += int(data.size)
        return {"valid": True, "crc_passed": True, "array_count": 120,
                "scalar_count": count, "dtype": "float64",
                "finite_and_protocol_shapes_and_score_bounds": True}
    except Exception as error:
        return {"valid": False, "reason": repr(error)}


def replay_worker(job):
    import run_confirmation as frozen
    mode, campaign, system, seed = job
    frozen.OUTPUT = REPAIRS
    frozen.worker(SimpleNamespace(mode=mode, campaign=campaign, system=system, seed=seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-worker", nargs=4, metavar=("MODE", "CAMPAIGN", "SYSTEM", "SEED"))
    args = parser.parse_args()
    if args.replay_worker:
        mode, campaign, system, seed = args.replay_worker
        replay_worker([mode, campaign, system, int(seed)])
        return
    completion = json.loads((OUTPUT / "completion.json").read_text())
    if completion["failures"] or len(completion["completed"]) != 72:
        raise RuntimeError("Complete all72 confirmation jobs successfully before artifact audit.")
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    validations, repairs = [], []
    for job in completion["all_scheduled_jobs"]:
        mode, campaign, system, seed = job
        original = OUTPUT / mode / campaign / f"{system}_{seed}"
        path = original / "episode_scores.npz"
        validation = validate_archive(path)
        if not validation["valid"]:
            print(f"REPAIR {job}: {validation['reason']}", flush=True)
            replay = REPAIRS / mode / campaign / f"{system}_{seed}"
            if replay.exists():
                raise RuntimeError(f"Refusing to overwrite prior repair attempt:{replay}")
            log = REPAIRS / "logs" / f"{mode}__{campaign}__{system}_{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x") as output:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--replay-worker",
                     mode, campaign, system, str(seed)], env=environment,
                    stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Exact replay failed; original retained; see{log}")
            replay_path = replay / "episode_scores.npz"
            rebuilt = validate_archive(replay_path)
            if not rebuilt["valid"]:
                raise RuntimeError(f"Replayed archive invalid:{replay_path}:{rebuilt}")
            comparisons = {}
            for filename in ("results.json", "frozen_candidates.json"):
                comparisons[filename] = (
                    without_timings(json.loads((original / filename).read_text())) ==
                    without_timings(json.loads((replay / filename).read_text())))
            if not all(comparisons.values()):
                raise RuntimeError(f"Replay numerical values differ:{comparisons}; no replacement made.")
            corrupt = original / "episode_scores.corrupt.npz"
            if corrupt.exists():
                raise RuntimeError(f"Refusing to overwrite preserved corrupt bytes:{corrupt}")
            original_sha, original_bytes = sha256(path), path.stat().st_size
            path.rename(corrupt)
            temporary = original / "episode_scores.repaired.tmp"
            shutil.copy2(replay_path, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            if sha256(path) != sha256(replay_path):
                raise RuntimeError("Copied repaired archive differs from exact replay.")
            record = {
                "reason": validation["reason"], "job": job,
                "repaired_utc": datetime.now(timezone.utc).isoformat(),
                "repair_method": "Exact replay of the same frozen confirmation job; only artifact output destination changed. No hyperparameters, seeds, preprocessing, data, or selection changed.",
                "corrupt_archive": str(corrupt), "corrupt_sha256": original_sha,
                "corrupt_bytes": original_bytes,
                "repaired_archive": str(path), "repaired_sha256": sha256(path),
                "repaired_bytes": path.stat().st_size,
                "replay_path": str(replay), "replay_archive": str(replay_path),
                "replay_log": str(log), "repair_harness_sha256": sha256(__file__),
                "original_results_equal_replay_excluding_timing": comparisons["results.json"],
                "original_frozen_candidates_equal_replay_excluding_timing": comparisons["frozen_candidates.json"],
                "original_manifest_and_json_preserved": True,
                "archive_validation": rebuilt,
            }
            write_json(original / "artifact_repair.json", record)
            repairs.append(record)
            validation = rebuilt
            print(f"REPAIRED {job}: exact JSON replay,120 valid arrays", flush=True)
        validations.append({"job": job, **validation})
    report = {
        "scheduled": 72, "validated": len(validations),
        "all_valid": all(row["valid"] for row in validations),
        "new_repairs": repairs, "validations": validations,
    }
    write_json(ROOT / "verification/confirmation_archive_audit.json", report)
    print(json.dumps({"validated": len(validations), "all_valid": report["all_valid"],
                      "new_repairs": len(repairs)}), flush=True)


if __name__ == "__main__":
    main()
