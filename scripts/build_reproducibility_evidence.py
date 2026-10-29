#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "reproduction"

OUTPUT_JSON = REPRO / "reproducibility_evidence.json"
OUTPUT_MD = REPRO / "REPRODUCIBILITY_EVIDENCE.md"

CHECKPOINTS = {
    "environment": "8b451d0",
    "reported_result_audit": "8637818",
    "licensing_and_citation": "a809d4a",
    "raw_data_infrastructure": "1585956",
    "raw_input_plan": "5a2408c",
    "raw_capture_cleanliness_fix": "ac531ae",
    "raw_data_release": "57ffcfc",
    "historical_run_accounting": "03cd857",
    "prospective_duffing_definition": "eab0ad4",
    "prospective_duffing_archive": "c01a057",
}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path.relative_to(ROOT)}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checkpoints() -> None:
    for name, commit in CHECKPOINTS.items():
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Checkpoint {name} ({commit}) is not present in Git history."
            ) from exc


def result_matrix_summary() -> dict[str, Any]:
    path = REPRO / "reported_results.tsv"
    require_file(path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    counts = Counter(row["status"] for row in rows)

    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 result-audit rows, found {len(rows)}.")
    if counts != Counter({"PASS": 23, "REMOVE": 2}):
        raise RuntimeError(f"Unexpected result status counts: {dict(counts)}.")

    removed = [
        {
            "result_id": row["result_id"],
            "label": row.get("label", ""),
            "description": row.get("description", ""),
        }
        for row in rows
        if row["status"] == "REMOVE"
    ]

    return {
        "total_audited_results": len(rows),
        "retained_pass_results": counts["PASS"],
        "remove_results": counts["REMOVE"],
        "removed_results": removed,
        "matrix_path": str(path.relative_to(ROOT)),
    }


def raw_data_summary() -> dict[str, Any]:
    canonical = REPRO / "raw_data" / "canonical"
    manifest = load_json(canonical / "manifest.json")

    if manifest["dataset_count"] != 7:
        raise RuntimeError("Canonical raw-data manifest does not contain 7 families.")
    if manifest["all_datasets_verified"] is not True:
        raise RuntimeError("Canonical raw-data manifest is not fully verified.")
    if manifest["all_captures_started_clean"] is not True:
        raise RuntimeError("Not all canonical raw-data captures started clean.")

    expected_capture_commit = "ac531aefc66066c3fc320d94c0b362fb88f94967"
    if manifest["capture_git_commits"] != [expected_capture_commit]:
        raise RuntimeError(
            f"Unexpected raw-data capture commits: {manifest['capture_git_commits']}"
        )

    index_path = canonical / "dataset_index.tsv"
    require_file(index_path)

    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    if len(rows) != 7:
        raise RuntimeError(f"Expected 7 raw-data index rows, found {len(rows)}.")

    logical_total = sum(int(row["logical_subdatasets"]) for row in rows)
    if logical_total != 553:
        raise RuntimeError(
            f"Expected 553 logical raw-input subdatasets, found {logical_total}."
        )

    require_file(REPRO / "raw_data" / "DATASET_CARD.md")
    require_file(canonical / "SHA256SUMS")

    return {
        "family_archives": 7,
        "logical_subdatasets": logical_total,
        "all_datasets_verified": True,
        "all_captures_started_clean": True,
        "capture_git_commit": expected_capture_commit,
        "manifest_path": str((canonical / "manifest.json").relative_to(ROOT)),
        "dataset_index_path": str(index_path.relative_to(ROOT)),
        "dataset_card_path": str(
            (REPRO / "raw_data" / "DATASET_CARD.md").relative_to(ROOT)
        ),
    }


def historical_run_accounting_summary() -> dict[str, Any]:
    account = REPRO / "run_accounting"
    summary = load_json(account / "summary.json")
    completeness = load_json(account / "expected_run_completeness.json")

    if summary["pass_result_count"] != 23:
        raise RuntimeError("Historical source audit does not cover 23 PASS results.")
    if summary["unique_resolved_source_file_count"] != 320:
        raise RuntimeError(
            "Historical source audit does not report 320 resolved source files."
        )
    if summary["tabular_source_row_count"] != 11501:
        raise RuntimeError(
            "Historical source audit does not report 11,501 tabular rows."
        )
    if summary["missing_reference_count"] != 0:
        raise RuntimeError("Historical source audit still has missing references.")

    checks = completeness["checks"]
    status_counts = Counter(check["status"] for check in checks)

    if len(checks) != 8 or status_counts != Counter({"PASS": 7, "GAP": 1}):
        raise RuntimeError(
            f"Unexpected historical completeness status: {dict(status_counts)}"
        )

    gap_checks = [check for check in checks if check["status"] == "GAP"]
    if len(gap_checks) != 1:
        raise RuntimeError("Expected exactly one historical completeness gap.")

    gap = gap_checks[0]
    if gap["campaign"] != "duffing_reference_search":
        raise RuntimeError(f"Unexpected historical gap: {gap['campaign']}")

    best = gap.get("best_candidate") or {}
    if best.get("covered") != 18 or gap.get("expected_jobs") != 27:
        raise RuntimeError(
            "Historical Duffing reference gap is not recorded as 18/27."
        )

    return {
        "retained_results_covered": 23,
        "resolved_source_files": 320,
        "tabular_source_rows": 11501,
        "missing_declared_references": 0,
        "expected_run_checks": 8,
        "pass_checks": 7,
        "gap_checks": 1,
        "historical_gap": {
            "campaign": "duffing_reference_search",
            "represented_jobs": 18,
            "documented_jobs": 27,
            "missing_jobs": 9,
            "missing_dimension": "all C=35 combinations",
        },
        "summary_path": str((account / "summary.json").relative_to(ROOT)),
        "completeness_path": str(
            (account / "expected_run_completeness.json").relative_to(ROOT)
        ),
        "audit_note_path": str((account / "README.md").relative_to(ROOT)),
    }


def prospective_duffing_summary() -> dict[str, Any]:
    base = (
        REPRO
        / "prospective_campaigns"
        / "duffing_reference_27"
        / "run_eab0ad4"
    )
    summary = load_json(base / "campaign_summary.json")
    verification = load_json(base / "verification_report.json")

    expected_commit = "eab0ad4042bff713e86188309de53a95e79df51c"

    conditions = {
        "campaign_commit": summary["campaign_git_commit"] == expected_commit,
        "attempt_count": summary["attempt_count"] == 27,
        "pass_count": summary["pass_count"] == 27,
        "fail_count": summary["fail_count"] == 0,
        "all_attempted": summary["all_27_jobs_attempted"] is True,
        "all_passed": summary["all_jobs_passed"] is True,
        "clean_before": summary["git_tree_clean_before_campaign"] is True,
        "clean_after": summary["git_tree_clean_after_campaign"] is True,
        "verification_status": verification["status"] == "PASS",
        "unique_jobs": verification["unique_job_count"] == 27,
        "verified_output_files": verification["verified_output_file_count"] == 81,
        "historical_audit_link": verification["historical_audit_commit"] == "03cd857",
    }

    failed = [name for name, ok in conditions.items() if not ok]
    if failed:
        raise RuntimeError(
            "Prospective Duffing verification failed conditions: "
            + ", ".join(failed)
        )

    require_file(base / "SHA256SUMS")
    require_file(base.parent / "README.md")
    require_file(ROOT / "scripts" / "verify_duffing_reference_campaign.py")

    return {
        "campaign": "duffing_reference_27_prospective",
        "campaign_git_commit": expected_commit,
        "scheduled_jobs": 27,
        "attempted_jobs": 27,
        "passed_jobs": 27,
        "failed_jobs": 0,
        "verified_output_files": 81,
        "git_tree_clean_before_campaign": True,
        "git_tree_clean_after_campaign": True,
        "provenance_interpretation": (
            "Subsequent reproducibility run of the complete grid; the historical "
            "archive remains incomplete."
        ),
        "summary_path": str((base / "campaign_summary.json").relative_to(ROOT)),
        "verification_path": str(
            (base / "verification_report.json").relative_to(ROOT)
        ),
        "ledger_path": str((base / "ledger.tsv").relative_to(ROOT)),
    }


def repository_controls_summary() -> dict[str, Any]:
    required = [
        ROOT / "LICENSE",
        ROOT / "DATA_LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "REPRODUCIBILITY.md",
        ROOT / "requirements-lock-arm64.txt",
        ROOT / "scripts" / "bootstrap_arm64.sh",
        ROOT / "scripts" / "verify_environment.py",
    ]
    for path in required:
        require_file(path)

    return {
        "code_license": "LICENSE",
        "data_license": "DATA_LICENSE",
        "citation_metadata": "CITATION.cff",
        "reproducibility_guide": "REPRODUCIBILITY.md",
        "reference_lockfile": "requirements-lock-arm64.txt",
        "bootstrap_script": "scripts/bootstrap_arm64.sh",
        "environment_verifier": "scripts/verify_environment.py",
        "reference_platform": "Apple Silicon arm64",
        "reference_python": "3.14.7",
    }


def build_evidence() -> dict[str, Any]:
    verify_checkpoints()

    evidence = {
        "schema_version": 1,
        "purpose": (
            "Summary of retained repository reproducibility evidence. Historical "
            "archives and subsequent reproducibility runs are identified by "
            "their source commits and execution records."
        ),
        "source_checkpoints": CHECKPOINTS,
        "reported_results": result_matrix_summary(),
        "raw_scientific_inputs": raw_data_summary(),
        "historical_run_accounting": historical_run_accounting_summary(),
        "prospective_gap_closure": prospective_duffing_summary(),
        "repository_controls": repository_controls_summary(),
        "claim_boundaries": {
            "historical_archive": (
                "The historical repository has complete archived schedule coverage "
                "for 7 of 8 checked campaign structures. The documented Duffing "
                "reference grid remains historically incomplete at 18/27."
            ),
            "prospective_closure": (
                "A separately labeled prospective campaign attempted and passed "
                "all 27 Duffing reference jobs from a frozen clean commit."
            ),
            "no_cherry_picking": (
                "The prospective Duffing campaign preserves every scheduled attempt "
                "and exit status. Historical campaigns have archived schedule "
                "coverage evidence, but equivalent execution ledgers are unavailable "
                "for every historical attempt."
            ),
            "remaining_release_work": [
                "two legacy event-label items excluded from the retained result set",
                "no immutable archival release or DOI recorded",
                "completed final release review not documented",
            ],
        },
    }

    return evidence


def write_markdown(evidence: dict[str, Any]) -> None:
    results = evidence["reported_results"]
    raw = evidence["raw_scientific_inputs"]
    historical = evidence["historical_run_accounting"]
    prospective = evidence["prospective_gap_closure"]
    controls = evidence["repository_controls"]

    text = f"""# Reproducibility evidence summary

This account summarizes the retained reproducibility evidence for the parent
KAHKM experiments. Historical archives and subsequent reproducibility runs
are identified by their source commits and execution records.

## Current evidence state

| Evidence area | Status | Repository evidence |
| --- | --- | --- |
| Reported-result audit | PASS for retained results | {results['retained_pass_results']} retained `PASS` results; {results['remove_results']} results marked `REMOVE` in `{results['matrix_path']}` |
| Canonical raw scientific inputs | PASS | {raw['family_archives']} family archives; {raw['logical_subdatasets']} logical raw-input subdatasets; all captures started clean |
| Declared result-source resolution | PASS | {historical['resolved_source_files']} unique source files; {historical['tabular_source_rows']:,} tabular rows; {historical['missing_declared_references']} missing declared references |
| Historical expected-run audit | PARTIAL | {historical['pass_checks']}/{historical['expected_run_checks']} checks complete; Duffing reference archive remains {historical['historical_gap']['represented_jobs']}/{historical['historical_gap']['documented_jobs']} |
| Prospective Duffing reference closure | PASS | {prospective['attempted_jobs']}/{prospective['scheduled_jobs']} attempted; {prospective['passed_jobs']} passed; {prospective['failed_jobs']} failed; {prospective['verified_output_files']} output files verified |
| Environment capture | PASS | {controls['reference_platform']}; Python {controls['reference_python']}; frozen lockfile and bootstrap/verifier scripts |
| Licensing and citation metadata | PASS | `LICENSE`, `DATA_LICENSE`, and `CITATION.cff` present |
| Immutable archival release | Not recorded | The current citation metadata contain no archival DOI |

## Reported-result provenance

The canonical result matrix audits {results['total_audited_results']} manuscript
results. {results['retained_pass_results']} are retained as `PASS`; the two
obsolete event-label items remain marked `REMOVE` rather than being presented as
supported results.

Primary evidence:

- `{results['matrix_path']}`
- generated result-specific tables/figures under `reproduction/generated/`
- version-controlled generator scripts under `scripts/`

## Canonical raw scientific inputs

The canonical raw-data release contains {raw['family_archives']} compressed
family archives representing {raw['logical_subdatasets']} logical raw-input
subdatasets. The manifest verifies every family and records that every capture
started from a clean Git tree at:

`{raw['capture_git_commit']}`

Primary evidence:

- `{raw['manifest_path']}`
- `{raw['dataset_index_path']}`
- `{raw['dataset_card_path']}`
- `reproduction/raw_data/canonical/SHA256SUMS`

## Historical run-accounting audit

The historical audit resolves all declared raw/per-run file references for the
{historical['retained_results_covered']} retained results:

- {historical['resolved_source_files']} unique resolved source files;
- {historical['tabular_source_rows']:,} tabular source rows;
- {historical['missing_declared_references']} missing declared references.

Expected-run coverage is complete for {historical['pass_checks']} of
{historical['expected_run_checks']} checked campaign structures.

The preserved historical gap is the Duffing reference search. The documented
grid has {historical['historical_gap']['documented_jobs']} jobs, while tracked
historical run-level evidence represents {historical['historical_gap']['represented_jobs']}.
The missing jobs are the nine `C=35` combinations.

This historical gap remains visible in:

- `{historical['completeness_path']}`
- `{historical['audit_note_path']}`

## Prospective Duffing reference closure

A separately labeled prospective reproducibility campaign was executed from
frozen clean commit:

`{prospective['campaign_git_commit']}`

The campaign scheduled all {prospective['scheduled_jobs']} combinations rather
than only rerunning the nine historically absent jobs.

Results:

- scheduled: {prospective['scheduled_jobs']}
- attempted: {prospective['attempted_jobs']}
- passed: {prospective['passed_jobs']}
- failed: {prospective['failed_jobs']}
- verified scientific output files: {prospective['verified_output_files']}
- Git tree clean before campaign: yes
- Git tree clean after campaign: yes

Primary evidence:

- `{prospective['ledger_path']}`
- `{prospective['summary_path']}`
- `{prospective['verification_path']}`
- `reproduction/prospective_campaigns/duffing_reference_27/run_eab0ad4/SHA256SUMS`

These outputs establish that the complete grid can be reproduced from the
recorded source and environment. They are subsequent runs; the original
historical archive remains incomplete.

## Scope of execution accounting

Every scheduled attempt in the prospective Duffing campaign has a ledger row,
exit code, logs, and output hashes.

The older campaigns have archived evidence of scheduled configurations, but
equivalent execution ledgers are unavailable for every attempt. Coverage of a
parameter grid therefore does not establish complete retention of failed
historical executions.

## Environment, licensing, and citation

Reference environment and controls:

- `{controls['reference_lockfile']}`
- `{controls['bootstrap_script']}`
- `{controls['environment_verifier']}`
- `{controls['reproducibility_guide']}`
- `{controls['code_license']}`
- `{controls['data_license']}`
- `{controls['citation_metadata']}`

## Archival and publication scope

The result matrix excludes the two legacy event-label items marked `REMOVE`.
The retained evidence does not document a completed final release review or
an immutable archival DOI. The source commits and checksums identify the
versions assessed here. The forecasting extension provides its own manuscript,
experimental records, and scientific verification report in
`extensions/certified_forecasting/`.
"""

    OUTPUT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    evidence = build_evidence()

    OUTPUT_JSON.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(evidence)

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print("PASS: 23 retained reported results verified in audit matrix")
    print("PASS: 7 canonical raw-data families / 553 logical subdatasets")
    print("PASS: 320 historical source files / 11,501 tabular rows / 0 missing refs")
    print("PASS: historical completeness preserved as 7 PASS / 1 GAP")
    print("PASS: prospective Duffing closure = 27 attempted / 27 passed / 0 failed")
    print("PASS: claim boundary for historical failed-attempt preservation retained")


if __name__ == "__main__":
    main()
