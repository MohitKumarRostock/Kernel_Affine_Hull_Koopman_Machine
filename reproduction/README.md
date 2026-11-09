# Reproduction bundles

This directory provides two distinct routes from the research questions to executable evidence. Run commands from the repository root unless a command first changes into a bundle.

| Bundle | Scientific question | Available computation |
| --- | --- | --- |
| [Central score enrichment](score_enrichment/) | Does retaining normalization scale or local position improve the selected forecasting procedures? | Rebuild summaries and all 40 primary comparisons from 26,624 saved loss rows; 17 original evidence records are supplied |
| [Supplementary predictor capacity](capacity_followup/) | Do local-coordinate benefits persist with 64 or 256 Nyström landmarks? | Verify original files, audit saved repetition records, or regenerate the original supplementary models and observations |

The central study uses 64 repetitions under four data-generating distributions and standardized MSE. The capacity follow-up uses 24 repetitions under three data-generating distributions, MSE and 12 primary comparisons. The supporting first-stage analysis in the capacity bundle supplies its practical margins. These observations and comparison families are not pooled.

## Rebuild central statistics

```bash
python3 -m pip install -r reproduction/score_enrichment/requirements.txt
python3 reproduction/score_enrichment/check_saved_losses.py
```

The independent checker reads the lossless selected-loss CSV directly from gzip and prints JSON. Add `--output /tmp/kahm-score-analysis` to save rebuilt tables in a new directory. It checks saved statistics and selected-record consistency; it does not generate the central observations, fit models, reproduce forecasts or recover the original source package.

## Verify and audit the supplementary records

```bash
cd reproduction/capacity_followup
python3 -m reproduce verify
```

Verification needs only the Python standard library. For the statistical audit, use Python 3.12 and install the bundle requirements in your chosen environment:

```bash
python3 -m pip install -r requirements.txt
python3 -m reproduce audit --output runs/archive-audit
```

The audit rebuilds the original first-stage and capacity statistics from archived records without model fitting. From the same directory, `python3 -m reproduce full --workers 4 --run-dir runs/full` generates observations and fits the original experiments. Its [bundle guide](capacity_followup/README.md) explains runtime, output files, resumption and verification limits.

The root [reproduction guide](../REPRODUCIBILITY.md) gives document-build commands, source identifiers and the inputs required for model regeneration. The commands in these two bundles perform numerical checks and experiments.
