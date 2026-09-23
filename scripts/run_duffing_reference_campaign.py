#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = (
    ROOT
    / "reproduction"
    / "prospective_campaigns"
    / "duffing_reference_27"
    / "campaign_spec.json"
)
SOURCE_SCRIPT = ROOT / "experiment_02_duffing_sensitivity.py"
LOCKFILE = ROOT / "requirements-lock-arm64.txt"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    ).strip()


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("campaign_id") != "duffing_reference_27_prospective":
        raise RuntimeError("Unexpected campaign_id.")

    if int(spec.get("expected_job_count", -1)) != 27:
        raise RuntimeError("Expected job count must be 27.")

    jobs = spec.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 27:
        raise RuntimeError("Campaign must contain exactly 27 explicit jobs.")

    expected = {
        (C, omega, seed)
        for C in (15, 25, 35)
        for omega in (4.0, 8.0, 12.0)
        for seed in (0, 1, 2)
    }
    actual = {
        (int(job["C"]), float(job["omega"]), int(job["seed"]))
        for job in jobs
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Campaign grid mismatch. Missing={missing}, extra={extra}"
        )

    ids = [str(job["job_id"]) for job in jobs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate job_id values in campaign spec.")


def command_for_job(
    job: dict[str, Any],
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    fixed = {
        "n_steps": 1200,
        "dt": 0.03,
        "data_seed": 1,
        "train_fraction": 0.75,
        "subspace_dim": 4,
        "nb": 100,
        "tau": 1e-6,
        "kmeans_kind": "auto",
        "kmeans_batch_size": 4096,
        "singleton_strategy": "augment",
        "singleton_aux_mix": 0.05,
        "beta": 0.1,
        "nlms_epochs": 20,
        "n_jobs": 1,
        "batch_size": 256,
    }

    args = [
        str(SOURCE_SCRIPT),
        "--n-steps",
        str(fixed["n_steps"]),
        "--dt",
        str(fixed["dt"]),
        "--data-seed",
        str(fixed["data_seed"]),
        "--train-fraction",
        str(fixed["train_fraction"]),
        "--clusters",
        str(int(job["C"])),
        "--omegas",
        str(float(job["omega"])),
        "--seeds",
        str(int(job["seed"])),
        "--subspace-dim",
        str(fixed["subspace_dim"]),
        "--nb",
        str(fixed["nb"]),
        "--tau",
        str(fixed["tau"]),
        "--kmeans-kind",
        str(fixed["kmeans_kind"]),
        "--kmeans-batch-size",
        str(fixed["kmeans_batch_size"]),
        "--singleton-strategy",
        str(fixed["singleton_strategy"]),
        "--singleton-aux-mix",
        str(fixed["singleton_aux_mix"]),
        "--beta",
        str(fixed["beta"]),
        "--nlms-epochs",
        str(fixed["nlms_epochs"]),
        "--n-jobs",
        str(fixed["n_jobs"]),
        "--batch-size",
        str(fixed["batch_size"]),
        "--output-dir",
        str(output_dir),
        "--strict",
        "--quiet",
    ]

    actual = [sys.executable, *args]
    display = [
        "$PYTHON",
        "experiment_02_duffing_sensitivity.py",
        *args[1:-4],
        "--output-dir",
        f"jobs/{job['job_id']}/output",
        "--strict",
        "--quiet",
    ]
    return actual, display


def tree_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append(
            {
                "path": str(file.relative_to(path)),
                "bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
        )
    return rows


def tree_sha256(inventory: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in inventory:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_ledger(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "attempt_index",
        "job_id",
        "C",
        "omega",
        "seed",
        "status",
        "exit_code",
        "started_at_utc",
        "ended_at_utc",
        "duration_seconds",
        "command",
        "stdout_log",
        "stdout_sha256",
        "stderr_log",
        "stderr_sha256",
        "output_file_count",
        "output_tree_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete prospective 27-job Duffing reference campaign "
            "with failure-preserving execution accounting."
        )
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = args.spec.resolve()
    output_root = args.output_root.resolve()

    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    if not SOURCE_SCRIPT.is_file():
        raise FileNotFoundError(SOURCE_SCRIPT)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_spec(spec)

    status_before = git_status()
    if status_before:
        raise RuntimeError(
            "Campaign runner requires a clean Git worktree. "
            f"git status --porcelain returned:\n{status_before}"
        )

    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"Output root must not already contain files: {output_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    jobs_root = output_root / "jobs"
    jobs_root.mkdir()

    commit = git("rev-parse", "HEAD")
    short_commit = git("rev-parse", "--short", "HEAD")

    runner_path = Path(__file__).resolve()
    spec_sha = sha256_file(spec_path)
    runner_sha = sha256_file(runner_path)
    source_sha = sha256_file(SOURCE_SCRIPT)
    lock_sha = sha256_file(LOCKFILE) if LOCKFILE.is_file() else None

    shutil.copy2(spec_path, output_root / "campaign_spec.json")

    metadata = {
        "campaign_id": spec["campaign_id"],
        "campaign_spec_version": spec["campaign_spec_version"],
        "campaign_git_commit": commit,
        "campaign_git_short_commit": short_commit,
        "git_tree_clean_before_campaign": True,
        "started_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": Path(sys.executable).name,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "spec_sha256": spec_sha,
        "runner_sha256": runner_sha,
        "source_script_sha256": source_sha,
        "requirements_lock_sha256": lock_sha,
        "expected_job_count": 27,
    }
    (output_root / "campaign_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    attempts_jsonl = output_root / "attempts.jsonl"
    ledger_rows: list[dict[str, Any]] = []

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for attempt_index, job in enumerate(spec["jobs"], start=1):
        job_id = str(job["job_id"])
        job_root = jobs_root / job_id
        job_output = job_root / "output"
        job_root.mkdir()
        job_output.mkdir()

        stdout_path = job_root / "stdout.log"
        stderr_path = job_root / "stderr.log"

        actual_cmd, display_cmd = command_for_job(job, job_output)

        started = utc_now()
        monotonic_start = time.monotonic()

        with stdout_path.open("wb") as stdout_handle, stderr_path.open(
            "wb"
        ) as stderr_handle:
            completed = subprocess.run(
                actual_cmd,
                cwd=ROOT,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )

        duration = time.monotonic() - monotonic_start
        ended = utc_now()

        inventory = tree_inventory(job_output)
        output_hash = tree_sha256(inventory)
        status = "PASS" if completed.returncode == 0 else "FAIL"

        attempt = {
            "attempt_index": attempt_index,
            "job_id": job_id,
            "C": int(job["C"]),
            "omega": float(job["omega"]),
            "seed": int(job["seed"]),
            "status": status,
            "exit_code": int(completed.returncode),
            "started_at_utc": started,
            "ended_at_utc": ended,
            "duration_seconds": round(duration, 6),
            "command": display_cmd,
            "stdout_log": str(stdout_path.relative_to(output_root)),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_log": str(stderr_path.relative_to(output_root)),
            "stderr_sha256": sha256_file(stderr_path),
            "output_file_count": len(inventory),
            "output_tree_sha256": output_hash,
            "output_files": inventory,
        }

        with attempts_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attempt, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        ledger_rows.append(
            {
                "attempt_index": attempt_index,
                "job_id": job_id,
                "C": int(job["C"]),
                "omega": float(job["omega"]),
                "seed": int(job["seed"]),
                "status": status,
                "exit_code": int(completed.returncode),
                "started_at_utc": started,
                "ended_at_utc": ended,
                "duration_seconds": round(duration, 6),
                "command": " ".join(display_cmd),
                "stdout_log": str(stdout_path.relative_to(output_root)),
                "stdout_sha256": attempt["stdout_sha256"],
                "stderr_log": str(stderr_path.relative_to(output_root)),
                "stderr_sha256": attempt["stderr_sha256"],
                "output_file_count": len(inventory),
                "output_tree_sha256": output_hash,
            }
        )

        write_ledger(ledger_rows, output_root / "ledger.tsv")

        print(
            f"[{attempt_index:02d}/27] {status} "
            f"{job_id} exit={completed.returncode} "
            f"files={len(inventory)}"
        )

    status_after = git_status()
    pass_count = sum(row["status"] == "PASS" for row in ledger_rows)
    fail_count = len(ledger_rows) - pass_count

    summary = {
        **metadata,
        "ended_at_utc": utc_now(),
        "attempt_count": len(ledger_rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "all_27_jobs_attempted": len(ledger_rows) == 27,
        "all_jobs_passed": fail_count == 0,
        "git_tree_clean_after_campaign": status_after == "",
        "git_status_after_campaign": status_after,
        "ledger_sha256": sha256_file(output_root / "ledger.tsv"),
        "attempts_jsonl_sha256": sha256_file(attempts_jsonl),
    }

    (output_root / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== CAMPAIGN SUMMARY ===")
    print(f"attempted: {len(ledger_rows)}/27")
    print(f"passed: {pass_count}")
    print(f"failed: {fail_count}")
    print(f"git clean after campaign: {status_after == ''}")
    print(f"output root: {output_root}")

    if len(ledger_rows) != 27 or status_after != "":
        raise SystemExit(2)
    if fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
