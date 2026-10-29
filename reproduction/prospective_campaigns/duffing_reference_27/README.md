# Prospective Duffing reference reproducibility campaign

This directory preserves a prospective reproducibility rerun of the complete
documented Duffing reference sensitivity grid.

This campaign was executed after the original experiments. The historical
run-accounting audit in `reproduction/run_accounting/` records that the tracked
historical outputs lacked the nine `C=35` jobs in the reference grid.

## Frozen campaign

Executed Git commit:

`eab0ad4042bff713e86188309de53a95e79df51c`

Historical audit checkpoint:

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

These outputs provide prospective evidence that the complete documented
27-combination Duffing reference campaign is executable under the frozen
reference environment and code revision.

They establish reproducibility of all scheduled configurations. The nine
missing `C=35` results remain absent from the original historical archive.

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
