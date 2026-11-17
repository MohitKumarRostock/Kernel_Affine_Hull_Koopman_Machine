# Prediction accuracy, model storage, and inference time

## Design and decision criteria

Two fixed KAHM-based model-selection procedures are compared with conventional accuracy and cost references. The six cases comprise five replications of one hidden Markov model and one chronological Appliances record, not six independent application domains. Primary forecast horizons are four steps for the synthetic process and six for Appliances. The records contain 23 distinct selected models; the two references coincide in one case.

`P_cost` selects predictors using normalized KAHM affinities; `CS_cost` uses clipped reconstruction scores. `C_accuracy` and `C_cost` are the conventional accuracy and cost references. Candidates and references are fixed before evaluation. A procedure must satisfy all three criteria:

| Quantity | Requirement |
| --- | --- |
| Mean squared error (MSE) | At most 1.1 times the MSE of `C_accuracy` |
| Numerical parameter storage | Strictly fewer bytes than `C_cost` |
| Median inference time per observation | No greater than `C_cost`, measured in batches after warm-up |

Both procedures meet the accuracy requirement in all six cases. Neither meets the storage or inference-time requirement in any case. Both therefore fail the combined criterion in all six cases.

## Reproduction

Run from the repository root:

```bash
python experiments/deployment_cost/replay.py --output results/deployment_cost.json
```

To regenerate predictions from stored raw inputs and model parameters:

```bash
python experiments/deployment_cost/replay.py --predict --output results/deployment_cost_predictions.json
```

The checker recomputes exact squared losses from binary64 targets and forecasts, numerical parameter sizes, and latency medians from the original nine-repeat timing samples. It runs offline with NumPy, SciPy, scikit-learn, and threadpoolctl.

## Files and measurement scope

`data/` contains primary-horizon evaluation rows, selected parameters, forecasts, expected risks, timing samples, and procedure identifiers. `source/` retains the protocol, model implementation, selection procedures, and evaluation code. `source_provenance.json` and per-task records identify the original archive and files.

Storage includes representation and predictor arrays and scalars, not Python-object or runtime-library overhead. It is not process-memory usage. Timing summaries use saved measurements rather than new benchmarks. Prediction regeneration evaluates fixed models without refitting or reselection; the comparison concerns the specified observations rather than a population-risk noninferiority guarantee. It does not verify exact real-arithmetic feature values.

Serialized models contain parameter dictionaries and arrays. Verify their integrity before loading them.
