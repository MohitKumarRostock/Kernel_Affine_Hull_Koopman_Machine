# Forecasting-model and hyperparameter-selection comparisons

These experiments compare predictor families and hyperparameter-selection
rules on specified forecasting periods. The results include every primary
comparison, including those that fail the required error threshold.

## Reproduce the numerical results

From the repository root:

```bash
python experiments/decoder_diagnostics/replay.py --output results/decoder_diagnostics.json
```

The replay reconstructs squared-loss bounds from retained prediction
intervals and original targets, or exact rational losses from binary64
outputs where that is the experiment's declared object. It checks 2,930
candidate-risk records. These include both losses of predictions in the original target units and losses of
exact rational reference predictions for all 1,440 affine-ablation configurations across the two evaluation
periods; they are not 2,930 independent experiments.

| Study | Assessed tasks | Result |
| --- | --- | --- |
| Lipschitz extension using all training observations | Seoul | Improves the baseline predictor but fails the conventional-reference requirement. |
| Temporal regularization and extension selection | Tetouan and Metro | Both primary criteria fail. |
| Joint selection of affine and Lipschitz-extension predictors | Household and Jena | Household passes; Jena fails, so the combined criterion fails. |
| Constrained affine forecasting | Specified Household and Jena periods | Both tasks fail the primary allowance of at most 5% higher MSE than the selected ordinary reference. |
| Projection, recentering, and clipping ablation | The same Household and Jena periods | Diagnostic comparisons, with all 720 configurations per task retained. |

The ablation pairs each switch's off/on configurations while holding the
other choices fixed. Its equality/improvement/worsening counts compare
exact rational MSEs of predictions in the original target units. Exact reference
losses are checked separately. Small recentering differences can be below
the original descriptive display threshold and do not establish a useful
predictive effect. These ablations use periods already examined during model development,
so they do not independently confirm a model-selection rule.

## Evidence and scope

`data` contains all assessment rows for the fixed selected procedures and
all ablation configurations. It includes targets, prediction intervals or
binary64 output encodings, expected risks, and original decision criteria.
`source` contains the corresponding original scientific source and protocol
files. `source_provenance.json` records their archive and file identities.

For interval predictions, the replay minimizes and maximizes squared loss
over each interval before averaging. Comparative success requires the
candidate's risk upper bound to meet the reference's risk lower bound;
overlapping bounds do not establish success. Identical selected model parameters
are recognized as equality when the criterion permits it.

This replay verifies loss arithmetic and the fixed assessment criteria.
It does not rerun model fitting, feature construction, optimizer proposals,
or development selection, and it does not retest the original numerical
output checker. The results concern the stated periods and predictors.
No population-risk, future-response-coverage, or training-cost-saving claim
follows from them. The command uses only the Python standard library and
performs no network access.
