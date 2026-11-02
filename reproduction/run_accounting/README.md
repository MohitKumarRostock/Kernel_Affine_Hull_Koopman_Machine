# Archived run-accounting audit

This directory records an audit of the experiment outputs that existed in the
repository before the subsequent prospective reproducibility campaign.

## Declared-source coverage

The retained result matrix contains 23 `PASS` results.

The source-resolution audit found:

- 320 unique resolved source files;
- 11,501 tabular source rows;
- 0 missing declared file references;
- R23 as the sole `NO_FILE_REFERENCE` result because it is a documentary
  search-protocol result.

## Expected-run completeness

The archived-output completeness audit checks the documented scheduled grids and
selected validation structures without filtering individual rows.

Seven of eight checks are complete:

- Duffing expanded search: 192/192 scheduled jobs represented;
- Van der Pol clean search: 147/147;
- Van der Pol noise-aware search: 450/450;
- Acrobot search: 30/30;
- classic-control sensitivity: 56/56 multistep configurations and 56/56
  one-step configurations;
- selected Acrobot validation: seeds 0, 1, and 2 present;
- spectral diagnostic structure: 40 rows present.

The historical Duffing reference search is incomplete in the archived tracked
results. Its documented grid is:

- `C = {15, 25, 35}`;
- `omega = {4, 8, 12}`;
- seeds `{0, 1, 2}`;

for 27 scheduled jobs.

The tracked CSV audit found the `C=15` and `C=25` combinations in other Duffing
tuning material but no tracked result source containing any of the nine `C=35`
reference-search jobs. The reference grid itself remains documented in the
version-controlled experiment/search-protocol code.

Therefore the archived repository alone does not support a claim that the
historical Duffing reference search has complete run-level evidence.

## Interpretation

The audit reports missing historical artifacts as gaps in the original record.
The [subsequent Duffing campaign](../prospective_campaigns/duffing_reference_27/)
executes the complete 27-job grid from a fixed source commit and records every
attempt, command, exit status, and output checksum. Those results establish
reproducibility of the grid without changing the historical archive.

Archived grid coverage identifies represented configurations. It does not, by
itself, establish that every failed execution attempt was preserved; that
stronger conclusion requires the execution ledger supplied by the prospective
campaign.
