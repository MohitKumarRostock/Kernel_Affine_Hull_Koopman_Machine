# Predictor comparisons and affine constraint ablations

## Experiments and results

These experiments compare predictor families and hyperparameter-selection rules on specified forecasting periods.

| Experiment | Data | Result |
| --- | --- | --- |
| Lipschitz extension using all training observations | Seoul | Improves the baseline predictor but fails the conventional-reference requirement. |
| Temporal regularization and extension selection | Tetouan and Metro | Both primary comparisons fail. |
| Joint selection of affine and Lipschitz-extension predictors | Household and Jena | Household passes; Jena fails; the combined criterion fails. |
| Constrained affine forecasting | Specified Household and Jena periods | Both fail the requirement of MSE at most 1.05 times the selected conventional reference. |
| Coefficient projection, intercept recentering, and clipping ablation | The same Household and Jena periods | All 720 configurations per task are retained. |

Each ablation compares an option's off/on settings while holding the other choices fixed. Equality, improvement, and worsening counts use exact rational mean squared errors (MSEs) in the original target units. Exact rational reference-prediction losses are checked separately. Small recentering differences can fall below the original display threshold. These periods were used during model development, so the ablation does not independently validate a selection rule.

## Reproduction

Run from the repository root:

```bash
python experiments/decoder_diagnostics/replay.py --output results/decoder_diagnostics.json
```

The command checks 2,930 stored predictor-risk records, including losses in original target units and exact rational reference losses for all 1,440 affine-ablation configurations. These records are not 2,930 independent experiments. It uses the Python standard library and makes no network requests.

For interval-valued predictions, the checker minimizes and maximizes squared loss over each interval before averaging. Binary64 outputs are treated as exact rational values where specified. Comparisons use the candidate's MSE upper bound and the reference's MSE lower bound with the experiment's stated allowance. Overlapping bounds alone do not establish success. Identical selected parameters count as equality when permitted by the criterion.

## Files and verification scope

`data/` contains evaluation targets, prediction intervals or binary64 outputs, expected risks, comparison criteria, and all ablation configurations. `source/` retains the scientific code and protocols. `source_provenance.json` identifies the original archive and files.

The checker recomputes loss arithmetic and fixed comparisons. It does not repeat feature construction, model fitting, optimizer proposals, development selection, or the original numerical-output checker. Results concern the specified predictors and periods, not population-risk confidence bounds or demonstrated training-cost savings.
