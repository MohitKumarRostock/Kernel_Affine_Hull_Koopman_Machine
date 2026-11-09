# Feature Augmentation for Forecasting with Reconstruction-Based Representations

This repository contains experiment code, protocols, saved per-run losses, statistical results, and verification tools for forecasting with reconstruction-based representations. The central study evaluates additional features derived from local reconstruction models. A separate predictor-capacity study supplies simulation and model-fitting code as well as saved results.

## Feature construction and scientific questions

Each local reference model reconstructs an observed time-series history. Reconstruction scores are converted to positive affinities, then normalized across reference regions to form membership features. These weights describe relative reconstruction quality; they are not calibrated probabilities of future outcomes. Normalization can remove the overall affinity scale, and reconstruction scores can discard the input's position within a region.

Kernel Affine Hull Machines (KAHMs) provide the main reconstruction method. Affine projection and kernel-weighted averaging of reference histories provide controls. The study evaluates two feature augmentations:

1. **Log normalization constants** restore the total affinity removed by normalization and make the full reconstruction-score vector recoverable in exact arithmetic.
2. **Membership-weighted local coordinates** can preserve the input under explicit positivity and rank conditions. A conditioning bound describes the sensitivity of input recovery.

Information preservation, numerical stability, and forecast accuracy require separate checks. Every evaluated procedure predicts future observations directly. The linear predictor fits ridge regression to the representation. The nonlinear predictor fits ridge regression to the representation concatenated with Gaussian-kernel Nyström features from 64 training landmarks.

## Code and evidence guide

| Question | Resource |
| --- | --- |
| How are observations, features, predictors, and comparisons defined? | [Recorded protocol](reproduction/score_enrichment/evidence/protocol.json) and [experiment guide](docs/EXPERIMENT_GUIDE.md) |
| What happens in every primary comparison? | [Complete primary table](reproduction/score_enrichment/evidence/primary_contrasts.csv) |
| How can I inspect an individual comparison? | [Explanation command and worked examples](docs/EXPERIMENT_GUIDE.md) |
| Can I recalculate the primary comparisons from per-run losses? | [Central statistical-analysis bundle](reproduction/score_enrichment/) |
| Which experiments can I regenerate? | [Predictor-capacity source and records](reproduction/capacity_followup/) |
| How do I check file identities, links, and statistical calculations? | [Reproduction and verification guide](REPRODUCIBILITY.md) |
| How do the other studies relate to the central experiment? | [Scientific interpretation](TIME_SERIES_RESEARCH.md) and [research index](docs/RESEARCH_INDEX.md) |

## Complete primary findings

Protocol `KAHM-SCORE-20261102-v1` uses 64 independent repetitions for each of four data-generating processes, giving 256 process–repetition combinations. The primary family contains 40 two-sided paired comparisons with Bonferroni adjustment. Each procedure selects its hyperparameters using the same validation observations within a task. The stored label `independent` denotes separate choices, not independent validation samples.

| Candidate / reference | Lower candidate mean loss | Higher candidate mean loss | Unresolved |
| --- | ---: | ---: | ---: |
| KAHM-local / KAHM-memberships | 6 | 0 | 2 |
| KAHM-normalizer / KAHM-memberships | 3 | 0 | 5 |
| KAHM-local / affine-local | 1 | 4 | 3 |
| KAHM-local / Kernel-local | 3 | 0 | 5 |
| KAHM-local / raw history | 3 | 3 | 2 |

Each row covers four data-generating processes and two predictors. “Local” includes memberships and membership-weighted local coordinates; “normalizer” includes memberships and log normalization constants. “Kernel-local” uses kernel-weighted averaging with local-coordinate features. Directions follow adjusted intervals, with no practical-benefit or equivalence margin. Coverage depends on the paired-Student approximation. An interval containing zero is classified as unresolved and does not establish equivalence. All four nonlinear-predictor comparisons between KAHM and kernel-weighted averaging are unresolved.

The results support local-coordinate augmentation in some settings, while comparisons between reconstruction methods favor different methods or remain inconclusive. The secondary analysis using a common dimension cap and the related studies have separate comparison rules.

## Recalculate the central comparisons

The [central evidence bundle](reproduction/score_enrichment/) contains 17 experiment records, including `selected_losses.csv.gz`. Its independent analysis script reconstructs 416 procedure summaries, 2,560 paired differences, and all 40 primary comparisons from 26,624 saved loss rows.

From the repository root, using Python 3.12:

```bash
python3 -m pip install -r reproduction/score_enrichment/requirements.txt
python3 reproduction/score_enrichment/check_saved_losses.py
```

The command prints a JSON report. Add `--output /tmp/kahm-score-analysis` to write the calculated tables to a new directory. The analysis verifies statistical calculations conditional on the saved losses. The complete central simulation and fitting source, exact task list, fitted model states, and forecast arrays are unavailable, so this bundle does not regenerate the central experiment.

## Predictor-capacity experiments

The [predictor-capacity bundle](reproduction/capacity_followup/) contains experiment source and saved records for a separate study with 24 repetitions under three data-generating processes. It compares KAHM local-coordinate features with raw histories and affine local-coordinate features at 64 and 256 Nyström landmarks. It supports file verification, statistical analysis of saved records, and full experiment regeneration. Its MSE outcomes and 12-comparison family are separate from the central study's SMSE outcomes and 40 comparisons.

The [reproduction directory](reproduction/README.md) maps the two bundles. The [reproduction guide](REPRODUCIBILITY.md) gives commands, dependencies, source identifiers, and the scope of each check.

## Availability and licenses

The [availability guide](extensions/score_enrichment_20261102/AVAILABILITY.md) identifies supplied records and unavailable central execution materials. Experiment records list sixteen numerical archives totaling 9.024 GiB; the complete set is unavailable for independent verification. Archive volume is a distribution constraint, separate from the unavailable implementation, task list, and complete numerical environment.

Code uses the [MIT License](LICENSE); research outputs retain the terms in [DATA_LICENSE](DATA_LICENSE). The source distribution includes author attribution and third-party notices.
