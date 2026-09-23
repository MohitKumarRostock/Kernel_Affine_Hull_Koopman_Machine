# Reproducibility evidence summary

This file consolidates the repository's current reproducibility evidence in one
reviewer-facing location. Historical evidence and prospective gap-closing runs
are deliberately kept separate.

## Current evidence state

| Evidence area | Status | Repository evidence |
| --- | --- | --- |
| Reported-result audit | PASS for retained results | 23 retained `PASS` results; 2 results marked `REMOVE` in `reproduction/reported_results.tsv` |
| Canonical raw scientific inputs | PASS | 7 family archives; 553 logical raw-input subdatasets; all captures started clean |
| Declared result-source resolution | PASS | 320 unique source files; 11,501 tabular rows; 0 missing declared references |
| Historical expected-run audit | PARTIAL | 7/8 checks complete; Duffing reference archive remains 18/27 |
| Prospective Duffing reference closure | PASS | 27/27 attempted; 27 passed; 0 failed; 81 output files verified |
| Environment capture | PASS | Apple Silicon arm64; Python 3.14.7; frozen lockfile and bootstrap/verifier scripts |
| Licensing and citation metadata | PASS | `LICENSE`, `DATA_LICENSE`, and `CITATION.cff` present |
| Final immutable archival release | PENDING | DOI/version metadata should be added only at final immutable release |

## Reported-result provenance

The canonical result matrix audits 25 manuscript
results. 23 are retained as `PASS`; the two
obsolete event-label items remain marked `REMOVE` rather than being presented as
supported results.

Primary evidence:

- `reproduction/reported_results.tsv`
- generated result-specific tables/figures under `reproduction/generated/`
- version-controlled generator scripts under `scripts/`

## Canonical raw scientific inputs

The canonical raw-data release contains 7 compressed
family archives representing 553 logical raw-input
subdatasets. The manifest verifies every family and records that every capture
started from a clean Git tree at:

`ac531aefc66066c3fc320d94c0b362fb88f94967`

Primary evidence:

- `reproduction/raw_data/canonical/manifest.json`
- `reproduction/raw_data/canonical/dataset_index.tsv`
- `reproduction/raw_data/DATASET_CARD.md`
- `reproduction/raw_data/canonical/SHA256SUMS`

## Historical run-accounting audit

The historical audit resolves all declared raw/per-run file references for the
23 retained results:

- 320 unique resolved source files;
- 11,501 tabular source rows;
- 0 missing declared references.

Expected-run coverage is complete for 7 of
8 checked campaign structures.

The preserved historical gap is the Duffing reference search. The documented
grid has 27 jobs, while tracked
historical run-level evidence represents 18.
The missing jobs are the nine `C=35` combinations.

This historical gap remains visible in:

- `reproduction/run_accounting/expected_run_completeness.json`
- `reproduction/run_accounting/README.md`

It is not rewritten or relabeled as historically complete.

## Prospective Duffing reference closure

A separately labeled prospective reproducibility campaign was executed from
frozen clean commit:

`eab0ad4042bff713e86188309de53a95e79df51c`

The campaign scheduled all 27 combinations rather
than only rerunning the nine historically absent jobs.

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

These outputs close the reproducibility evidence gap but are explicitly not
represented as original historical outputs.

## No-cherry-picking claim boundary

The prospective Duffing campaign has strong failure-preserving accounting:
every scheduled attempt has a ledger row, exit code, logs, and output hashes.

For the older historical campaigns, the repository now provides archived
schedule-completeness evidence, but it does not have equivalent execution ledgers
for every historical attempt. Therefore the repository should not claim that
every failed historical execution was preserved unless additional evidence is
added.

## Environment, licensing, and citation

Reference environment and controls:

- `requirements-lock-arm64.txt`
- `scripts/bootstrap_arm64.sh`
- `scripts/verify_environment.py`
- `REPRODUCIBILITY.md`
- `LICENSE`
- `DATA_LICENSE`
- `CITATION.cff`

## Remaining release work

Before the final immutable release:

1. remove the two `REMOVE` manuscript items and apply the already identified
   caption/protocol clarifications;
2. run final end-to-end repository verification after manuscript cleanup;
3. create the immutable archival release and then add final DOI/version metadata
   to citation/release files.

The evidence summary intentionally does not mark those remaining steps complete
before they have been performed.
