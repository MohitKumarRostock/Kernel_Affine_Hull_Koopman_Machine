# Reproducibility evidence summary

This file summarizes the inputs, execution records and checks supporting the reported KAHKM results. It identifies incomplete records and distinguishes different experimental executions.

## Available evidence

| Evidence area | Status | Repository evidence |
| --- | --- | --- |
| Reported-result audit | PASS for retained results | 23 retained `PASS` results; 2 results marked `REMOVE` in `reproduction/reported_results.tsv` |
| Raw scientific inputs | PASS | 7 family archives; 553 logical raw-input subdatasets; all captures started clean |
| Declared result-source resolution | PASS | 320 unique source files; 11,501 tabular rows; 0 missing declared references |
| Archived-run coverage audit | PARTIAL | 7/8 checks complete; Duffing reference archive remains 18/27 |
| Separate Duffing reference experiment | PASS | 27/27 attempted; 27 passed; 0 failed; 81 output files verified |
| Environment capture | PASS | Apple Silicon arm64; Python 3.14.7; fixed lockfile and bootstrap/verifier scripts |
| Licensing and citation metadata | PASS | `LICENSE`, `DATA_LICENSE`, and `CITATION.cff` present |
| Final immutable archival release | PENDING | No archival DOI is assigned in these records |

## Reported-result provenance

The result matrix assesses 25 items. It identifies 23 supported results as `PASS`. Two event-label items are marked `REMOVE` and are excluded from the supported result set.

Primary evidence:

- `reproduction/reported_results.tsv`
- generated result-specific tables/figures under `reproduction/generated/`
- version-controlled generator scripts under `scripts/`

## Raw scientific inputs

The raw-data package contains 7 compressed
family archives representing 553 logical raw-input
subdatasets. The manifest verifies every family and records that every capture
started from a clean Git tree at:

`ac531aefc66066c3fc320d94c0b362fb88f94967`

Primary evidence:

- `reproduction/raw_data/canonical/manifest.json`
- `reproduction/raw_data/canonical/dataset_index.tsv`
- `reproduction/raw_data/DATASET_CARD.md`
- `reproduction/raw_data/canonical/SHA256SUMS`

## Coverage of archived runs

The audit resolves all declared raw/per-run file references for the
23 retained results:

- 320 unique resolved source files;
- 11,501 tabular source rows;
- 0 missing declared references.

Expected-run coverage is complete for 7 of
8 checked campaign structures.

The documented gap is the Duffing reference search. The documented
grid has 27 jobs, while tracked
archived run-level evidence represents 18.
The missing jobs are the nine `C=35` combinations.

The following files document the missing outputs:

- `reproduction/run_accounting/expected_run_completeness.json`
- `reproduction/run_accounting/README.md`

These archived records do not establish completion of all 27 scheduled jobs.

## Separate Duffing reference experiment

The complete 27-job reference experiment was executed from commit:

`eab0ad4042bff713e86188309de53a95e79df51c`

The experiment scheduled all 27 parameter combinations.

Results:

- scheduled: 27
- attempted: 27
- passed: 27
- failed: 0
- verified scientific output files: 81
- Git tree clean before campaign: yes
- Git tree clean after campaign: yes

Primary evidence:

- `reproduction/prospective_campaigns/duffing_reference_27/run_eab0ad4/ledger.tsv`
- `reproduction/prospective_campaigns/duffing_reference_27/run_eab0ad4/campaign_summary.json`
- `reproduction/prospective_campaigns/duffing_reference_27/run_eab0ad4/verification_report.json`
- `reproduction/prospective_campaigns/duffing_reference_27/run_eab0ad4/SHA256SUMS`

These records establish completion of this 27-job experiment. They are separate from the incomplete reference-search archive.

## Coverage of execution records

The 27-job Duffing experiment records every scheduled attempt, exit code,
log and output checksum.

The other archived experiments include checks of scheduled-job coverage but do not provide equivalent execution logs for every attempt. Their records therefore do not establish that every failed execution was saved.

## Environment, licensing, and citation

Reference environment and controls:

- `requirements-lock-arm64.txt`
- `scripts/bootstrap_arm64.sh`
- `scripts/verify_environment.py`
- `REPRODUCIBILITY.md`
- `LICENSE`
- `DATA_LICENSE`
- `CITATION.cff`

## Interpretation

The status entries above describe the available records and the checks performed on them. A complete execution log supports conclusions about that execution; it does not establish completeness of another experiment. Hashes establish file identity, not the correctness of a scientific claim.
