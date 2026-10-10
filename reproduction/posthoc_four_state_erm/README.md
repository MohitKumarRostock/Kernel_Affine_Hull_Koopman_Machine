# Post-hoc four-state certificate versus ERM comparison

This directory documents a *secondary analysis* of the original four-state campaign. No new experimental trajectories or random replicates were generated.

## Original evidence
- Repository historical experiment commit (archive metadata): `cde430dbcca09bf911439bdb5bc3f85350e232cd`
- Frozen publication branch starting point: `0bbb2bafa107892e32e221e882326890b14fe36e`
- Original evidence remains unchanged at `reproduction/prospective_campaigns/four_state_original_iid_final_v1_run001/`.
- Frozen experiment config SHA-256: `630b6c30f5dcbda9034b85b2a3416d92ef264953cd05133f3e53fa14e7aae182`.
- Evidence tar.gz SHA-256: `4776942edbd659f7a1c02ec554507f7c0259ca410449e2e5fd66f5353bf638d5`.

## Reanalysis
- 13 designs × 9 sample sizes × 1,000 existing datasets = 117,000 count records.
- Original certificates recomputed with maximum discrepancy 1.11e-16.
- Four methods: label-based, sample-centered, design-known fixed prototype, generic empirical-minimum risk.
- `kappa=sqrt(2)`. The constrained empirical-minimum MSE equals the within-class successor variation in this special four-state construction.
- The four-way joint comparison allocates `delta=0.05/4` **within one prespecified evaluation dataset**, *not* across 117,000 trials.
- The fixed prototypes use the known model parameter `p`, independently of evaluation data. This is an oracle-design advantage.
- All new figures and comparisons must be labeled post-hoc. Historical manuscript numerical results remain historical.

The repository now includes the complete four-state post-hoc calculation and archived outputs. Follow [`README_REPRODUCE.md`](README_REPRODUCE.md) for the verified frozen-input commands and exact expected SHA-256 values. Scripts `scripts/build_source_zip.py`, `scripts/reanalyze_historical_four_state.py` and `scripts/certificate_reanalysis.py` reproduce the computations from the unchanged original campaign. `outputs/historical_117k/` contains both 117-cell summary CSVs and full 117,000-row replicate-level CSVs stored as `.csv.gz` (decompress to verify the published uncompressed checksums). `tests/test_frozen_erm.py` checks the four-state special-case formulas and the archived summaries. The immutable results-bearing commit is `d6505683d7d899beb69b00b4831db1ad37e386f2`.

## Four-state illustrative results
See `selected_comparison_joint.csv` for the p=31/32 conflicting-successor case, 1,000 archived replicates per sample size.

## Scientific limitations
These bounds are sufficient lower risk certificates, not proofs of lower bound dominance. The centered finite-sample bound may be conservative despite a tight population expression. No Van der Pol reanalysis is claimed. Source Git identity is now anchored by this branch's parent commit; archive evidence integrity is checked by hashes.