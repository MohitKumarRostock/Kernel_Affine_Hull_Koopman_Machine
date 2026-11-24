# Chronological atmospheric CO2 forecasting

The directory contains the public-domain weekly Mauna Loa snapshot, fitted-predictor results, validation grid, assessment forecasts, diagnostics, annual loss summaries and executable analysis. [Data provenance](../docs/DATA.md) records the source and file identity; the [protocol](../docs/PROTOCOLS.md) specifies eligible origins, chronological splits, controls and selection rules.

`run_real_data.py` uses the reconstruction and encoding routines in `../ablation/run_ablation.py`. `check_results.py` independently reconstructs eligible dates and targets, recomputes loss summaries from the archived forecasts, and checks the validation selections and matched-complexity diagnostics. The array `predictions.npz` contains integer candidate IDs, predicted changes with shape (212, 614, 3), assessment origin dates, target changes, origin levels and training target variances. Candidate IDs link to `results.csv`; horizon order is 1, 4, 12 weeks. `yearly.csv` provides annual summaries linked by candidate ID. `diagnostics.csv` records local ranks, weighted-system singular values and fallback counts.

Use the root reproduction driver to verify or regenerate without overwriting the distributed records. The study uses one dependent series; reference-seed variations are descriptive, not independent data replications.
