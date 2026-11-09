# Duffing reference experiment: 27 parameter combinations

This directory contains the complete Duffing reference sensitivity experiment. The separate run-accounting audit in `reproduction/run_accounting/` describes an archive that lacks the nine `C=35` combinations. The two directories describe different executions.

## Experimental configuration

Executed Git commit:

`eab0ad4042bff713e86188309de53a95e79df51c`

Run-accounting source commit:

`03cd857`

Grid:

- `C = {15, 25, 35}`
- `omega = {4, 8, 12}`
- seed `{0, 1, 2}`

Total scheduled jobs: **27**.

All 27 jobs were attempted. All 27 completed with exit code 0. The campaign
started and ended with a clean Git working tree.

## Failure-preserving evidence

The campaign runner executes every scheduled job independently and records each
attempt even if it fails.

The archived run directory contains:

- `ledger.tsv`: one row per scheduled attempt;
- `attempts.jsonl`: full per-attempt records, including output-file inventories;
- per-job `stdout.log` and `stderr.log`;
- per-job output directories;
- `campaign_metadata.json`;
- `campaign_summary.json`;
- the exact `campaign_spec.json`;
- `verification_report.json`;
- `SHA256SUMS`.

No failed attempts were omitted; this particular campaign had zero failures.

## Interpretation

The records show that all 27 parameter combinations completed in the specified environment. They provide evidence for this execution only. They do not supply the missing outputs from the separate archived reference search.

## Verification

From the repository root:

```bash
.venv-arm64/bin/python scripts/verify_duffing_reference_campaign.py \
  reproduction/prospective_campaigns/duffing_reference_27/run_eab0ad4
```

For byte-level distributed-file verification:

```bash
cd reproduction/prospective_campaigns/duffing_reference_27/run_eab0ad4
shasum -a 256 -c SHA256SUMS
```
