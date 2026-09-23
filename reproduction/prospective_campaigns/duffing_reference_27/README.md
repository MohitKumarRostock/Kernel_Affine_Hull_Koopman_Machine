# Prospective Duffing reference reproducibility campaign

This directory preserves a prospective reproducibility rerun of the complete
documented Duffing reference sensitivity grid.

The campaign is **not** represented as an original historical experiment
archive. The historical run-accounting audit remains preserved separately under
`reproduction/run_accounting/` and records that the tracked historical outputs
did not contain the nine `C=35` jobs from the documented reference grid.

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

They close the reproducibility evidence gap identified by the historical audit,
but they do not retroactively convert the previously missing `C=35` historical
artifacts into original historical outputs.

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
