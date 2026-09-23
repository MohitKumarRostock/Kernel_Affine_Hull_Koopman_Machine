#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_COMMIT = "eab0ad4042bff713e86188309de53a95e79df51c"
EXPECTED_GRID = {
    (C, omega, seed)
    for C, omega, seed in itertools.product(
        (15, 25, 35),
        (4.0, 8.0, 12.0),
        (0, 1, 2),
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(inventory: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in inventory:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the imported prospective 27-job Duffing reference campaign "
            "against its failure-preserving ledger and recorded checksums."
        )
    )
    parser.add_argument(
        "campaign_dir",
        type=Path,
        help="Imported campaign directory.",
    )
    parser.add_argument(
        "--write-release-files",
        action="store_true",
        help="Write verification_report.json and SHA256SUMS after verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_dir.resolve()

    required = {
        "campaign_spec.json",
        "campaign_metadata.json",
        "campaign_summary.json",
        "attempts.jsonl",
        "ledger.tsv",
    }
    missing = sorted(
        name for name in required if not (campaign_dir / name).is_file()
    )
    if missing:
        raise RuntimeError(f"Missing campaign files: {missing}")

    spec_path = campaign_dir / "campaign_spec.json"
    metadata_path = campaign_dir / "campaign_metadata.json"
    summary_path = campaign_dir / "campaign_summary.json"
    attempts_path = campaign_dir / "attempts.jsonl"
    ledger_path = campaign_dir / "ledger.tsv"

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with ledger_path.open(newline="", encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle, delimiter="\t"))

    if spec["campaign_id"] != "duffing_reference_27_prospective":
        raise RuntimeError("Unexpected campaign_id.")
    if spec["expected_job_count"] != 27:
        raise RuntimeError("Campaign spec does not expect 27 jobs.")
    if len(spec["jobs"]) != 27:
        raise RuntimeError("Campaign spec does not contain 27 explicit jobs.")

    spec_grid = {
        (int(job["C"]), float(job["omega"]), int(job["seed"]))
        for job in spec["jobs"]
    }
    if spec_grid != EXPECTED_GRID:
        raise RuntimeError("Campaign specification grid mismatch.")

    if summary["campaign_git_commit"] != EXPECTED_COMMIT:
        raise RuntimeError(
            f"Unexpected campaign commit: {summary['campaign_git_commit']}"
        )
    if metadata["campaign_git_commit"] != EXPECTED_COMMIT:
        raise RuntimeError(
            f"Unexpected metadata commit: {metadata['campaign_git_commit']}"
        )

    if summary["attempt_count"] != 27:
        raise RuntimeError("Campaign summary attempt_count is not 27.")
    if summary["pass_count"] != 27 or summary["fail_count"] != 0:
        raise RuntimeError("Campaign summary does not report 27 PASS / 0 FAIL.")
    if summary["all_27_jobs_attempted"] is not True:
        raise RuntimeError("Summary does not confirm all 27 jobs attempted.")
    if summary["all_jobs_passed"] is not True:
        raise RuntimeError("Summary does not confirm all jobs passed.")
    if summary["git_tree_clean_before_campaign"] is not True:
        raise RuntimeError("Campaign did not start from a clean Git tree.")
    if summary["git_tree_clean_after_campaign"] is not True:
        raise RuntimeError("Campaign did not end with a clean Git tree.")

    if sha256_file(spec_path) != summary["spec_sha256"]:
        raise RuntimeError("campaign_spec.json SHA-256 mismatch.")
    if sha256_file(ledger_path) != summary["ledger_sha256"]:
        raise RuntimeError("ledger.tsv SHA-256 mismatch.")
    if sha256_file(attempts_path) != summary["attempts_jsonl_sha256"]:
        raise RuntimeError("attempts.jsonl SHA-256 mismatch.")

    parent_spec = campaign_dir.parent / "campaign_spec.json"
    if parent_spec.is_file():
        if sha256_file(parent_spec) != summary["spec_sha256"]:
            raise RuntimeError(
                "Imported campaign spec does not match the frozen parent spec."
            )

    runner = ROOT / "scripts" / "run_duffing_reference_campaign.py"
    source = ROOT / "experiment_02_duffing_sensitivity.py"
    lockfile = ROOT / "requirements-lock-arm64.txt"

    if sha256_file(runner) != summary["runner_sha256"]:
        raise RuntimeError("Current campaign runner differs from executed runner.")
    if sha256_file(source) != summary["source_script_sha256"]:
        raise RuntimeError("Current Experiment 02 source differs from executed source.")
    if sha256_file(lockfile) != summary["requirements_lock_sha256"]:
        raise RuntimeError("Current arm64 lockfile differs from executed environment lock.")

    if len(attempts) != 27:
        raise RuntimeError(f"Expected 27 attempts, found {len(attempts)}.")
    if len(ledger) != 27:
        raise RuntimeError(f"Expected 27 ledger rows, found {len(ledger)}.")

    attempt_ids = [str(row["job_id"]) for row in attempts]
    ledger_ids = [str(row["job_id"]) for row in ledger]

    if len(set(attempt_ids)) != 27:
        raise RuntimeError("attempts.jsonl contains duplicate job IDs.")
    if len(set(ledger_ids)) != 27:
        raise RuntimeError("ledger.tsv contains duplicate job IDs.")
    if attempt_ids != ledger_ids:
        raise RuntimeError("Attempt and ledger job ordering differs.")

    attempt_grid = {
        (int(row["C"]), float(row["omega"]), int(row["seed"]))
        for row in attempts
    }
    ledger_grid = {
        (int(row["C"]), float(row["omega"]), int(row["seed"]))
        for row in ledger
    }

    if attempt_grid != EXPECTED_GRID:
        raise RuntimeError("Attempt grid does not equal expected 27-job grid.")
    if ledger_grid != EXPECTED_GRID:
        raise RuntimeError("Ledger grid does not equal expected 27-job grid.")

    verified_output_files = 0

    for attempt, ledger_row in zip(attempts, ledger):
        job_id = str(attempt["job_id"])

        if attempt["status"] != "PASS" or int(attempt["exit_code"]) != 0:
            raise RuntimeError(f"{job_id}: attempt not PASS/0.")
        if ledger_row["status"] != "PASS" or int(ledger_row["exit_code"]) != 0:
            raise RuntimeError(f"{job_id}: ledger not PASS/0.")

        stdout = campaign_dir / str(attempt["stdout_log"])
        stderr = campaign_dir / str(attempt["stderr_log"])

        if sha256_file(stdout) != attempt["stdout_sha256"]:
            raise RuntimeError(f"{job_id}: stdout checksum mismatch.")
        if sha256_file(stderr) != attempt["stderr_sha256"]:
            raise RuntimeError(f"{job_id}: stderr checksum mismatch.")

        if sha256_file(stdout) != ledger_row["stdout_sha256"]:
            raise RuntimeError(f"{job_id}: stdout ledger checksum mismatch.")
        if sha256_file(stderr) != ledger_row["stderr_sha256"]:
            raise RuntimeError(f"{job_id}: stderr ledger checksum mismatch.")

        inventory = list(attempt["output_files"])
        if len(inventory) != int(attempt["output_file_count"]):
            raise RuntimeError(f"{job_id}: output inventory count mismatch.")
        if len(inventory) != int(ledger_row["output_file_count"]):
            raise RuntimeError(f"{job_id}: ledger output count mismatch.")

        output_root = campaign_dir / "jobs" / job_id / "output"

        for item in inventory:
            file_path = output_root / str(item["path"])
            if not file_path.is_file():
                raise RuntimeError(f"{job_id}: missing output {item['path']}.")
            if file_path.stat().st_size != int(item["bytes"]):
                raise RuntimeError(f"{job_id}: size mismatch {item['path']}.")
            if sha256_file(file_path) != item["sha256"]:
                raise RuntimeError(f"{job_id}: checksum mismatch {item['path']}.")
            verified_output_files += 1

        recorded_tree = str(attempt["output_tree_sha256"])
        if tree_sha256(inventory) != recorded_tree:
            raise RuntimeError(f"{job_id}: recorded output-tree hash mismatch.")
        if recorded_tree != ledger_row["output_tree_sha256"]:
            raise RuntimeError(f"{job_id}: ledger output-tree hash mismatch.")

    report = {
        "verification_version": 1,
        "campaign_id": spec["campaign_id"],
        "campaign_git_commit": EXPECTED_COMMIT,
        "historical_audit_commit": spec["historical_audit_commit"],
        "expected_job_count": 27,
        "attempt_count": 27,
        "pass_count": 27,
        "fail_count": 0,
        "unique_job_count": 27,
        "verified_output_file_count": verified_output_files,
        "git_tree_clean_before_campaign": True,
        "git_tree_clean_after_campaign": True,
        "ledger_sha256": sha256_file(ledger_path),
        "attempts_jsonl_sha256": sha256_file(attempts_path),
        "spec_sha256": sha256_file(spec_path),
        "status": "PASS",
        "interpretation": (
            "Prospective reproducibility rerun. These outputs close the archived "
            "Duffing reference-grid evidence gap but are not historical originals."
        ),
    }

    if args.write_release_files:
        report_path = campaign_dir / "verification_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        targets = sorted(
            (
                path
                for path in campaign_dir.rglob("*")
                if path.is_file()
                and path.name != "SHA256SUMS"
            ),
            key=lambda path: str(path.relative_to(campaign_dir)),
        )

        checksums = campaign_dir / "SHA256SUMS"
        checksums.write_text(
            "".join(
                f"{sha256_file(path)}  {path.relative_to(campaign_dir)}\n"
                for path in targets
            ),
            encoding="utf-8",
        )

        print(f"Wrote {report_path}")
        print(f"Wrote {checksums}")

    print("PASS: prospective Duffing campaign verified")
    print("PASS: 27 unique scheduled jobs attempted")
    print("PASS: 27 jobs passed, 0 failed")
    print(f"PASS: {verified_output_files} recorded output files verified")
    print("PASS: campaign started and ended with clean Git tree")
    print(f"campaign commit: {EXPECTED_COMMIT}")


if __name__ == "__main__":
    main()
