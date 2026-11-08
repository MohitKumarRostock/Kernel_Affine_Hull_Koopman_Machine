# Decoder transfer, selection, and affine diagnostics

These experiments examine whether stronger decoder families or different
selection rules improve the fixed-period forecasting results. They support
the manuscript's limits on practical usefulness and preserve unsuccessful
primary outcomes.

## Reproduce the manuscript results

From the repository root:

```bash
python experiments/decoder_diagnostics/replay.py --output results/decoder_diagnostics.json
```

The replay reconstructs squared-loss bounds from retained prediction
intervals and original targets, or exact rational losses from binary64
outputs where that is the experiment's declared object. It checks 2,930
candidate-risk records. These include both physical-output and exact-reference
losses for all 1,440 affine-ablation configurations across the two exposed
periods; they are not 2,930 independent experiments.

| Study | Assessed tasks | Result preserved |
| --- | --- | --- |
| Training-only full-anchor transfer | Seoul | Improves the inherited head but fails the ordinary-comparator requirement. |
| Temporal regularization and extension selection | Tetouan and Metro | Both primary criteria fail. |
| Joint affine and full-anchor head selection | Household and Jena | Household passes; Jena fails, so the combined criterion fails. |
| Constrained affine forecasting | Later Household and Jena periods | Both tasks fail the primary allowance of at most 5% higher MSE than the selected ordinary reference. |
| Projection, recentering, and clipping ablation | The same exposed affine-comparison periods | Diagnostic comparisons, with all 720 configurations per task retained. |

The ablation pairs each switch's off/on configurations while holding the
other choices fixed. Its equality/improvement/worsening counts compare
exact rational MSEs of the reported physical outputs. Exact reference
losses are checked separately. Small recentering differences can be below
the original descriptive display threshold and do not establish a useful
predictive effect. These exposed-period comparisons are not independent
confirmation of a new policy.

## Evidence and scope

`data` contains all assessment rows for the frozen selected procedures and
all ablation configurations. It includes targets, prediction intervals or
binary64 output encodings, expected risks, and original decision criteria.
`source` contains the corresponding original scientific source and protocol
files. `source_provenance.json` records their archive and file identities.

For interval predictions, the replay minimizes and maximizes squared loss
over each interval before averaging. Comparative success requires the
candidate's risk upper bound to meet the reference's risk lower bound;
overlapping bounds do not establish success. Identical selected payloads
are recognized as equality when the criterion permits it.

This replay verifies loss arithmetic and the frozen assessment criteria.
It does not rerun model fitting, feature construction, optimizer proposals,
or development selection, and it does not retest the original numerical
output checker. The results concern the stated periods and predictors.
No population-risk, future-response-coverage, or training-cost-saving claim
follows from them. The command uses only the Python standard library and
performs no network access.
