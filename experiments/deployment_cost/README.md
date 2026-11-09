# Accuracy and complete deployment cost

This experiment evaluates two fixed KAHM cost-selection procedures against
conventional accuracy and cost references. The panel comprises five
replications of one hidden Markov model and one chronological Appliances
record. It does not supply six independent application domains.

## Reproduce the manuscript results

From the repository root:

```bash
python experiments/deployment_cost/replay.py --output results/deployment_cost.json
```

To also regenerate the saved predictions from raw inputs and the retained
model parameters:

```bash
python experiments/deployment_cost/replay.py --predict --output results/deployment_cost_predictions.json
```

The replay reconstructs exact squared losses from the stored binary64 targets
and forecasts, storage sizes of the complete numerical model parameters, and latency medians
from the original nine-repeat timing samples. The primary horizons are four
steps for the synthetic process and six steps for Appliances. There are 23
distinct retained primary models; the two references coincide in one
case.

The two procedures select from normalized KAHM affinities (`P_cost`) and
clipped reconstruction scores (`CS_cost`). The references are the fixed
conventional accuracy choice (`C_accuracy`) and conventional cost choice
(`C_cost`). Each successful procedure must meet all of the following criteria:

- MSE no greater than 1.1 times the accuracy reference's MSE.
- Strictly fewer bytes of numerical model parameters than the cost reference.
- Median inference time per observation after warm-up, measured in batches, no greater than the cost reference.

Each procedure passes the accuracy allowance in all six cases and passes
neither the byte-saving nor the latency criterion in any case. Thus both
fail the joint accuracy, storage and latency criterion in all six cases.
The references and candidates are fixed before evaluation.

## Evidence and scope

`data` contains the complete primary-horizon assessment rows, selected model
parameters, forecasts, exact expected risks, original timing samples, and
fixed procedure identities. `source` preserves the original protocol,
model implementation, selection procedures, and evaluation code.
`source_provenance.json` and the per-task records identify the original
archive and source files.

The numeric inventory includes the retained encoder and predictor arrays and
scalars. It excludes Python-object and runtime-library overhead; it is not
a process-memory measurement. Original timing samples are replayed, not
remeasured. The optional prediction replay evaluates the stored
models; it does not refit or reselect them, establish ideal real-arithmetic
feature values, or confer a population noninferiority guarantee.

The replay is offline and requires NumPy, SciPy, scikit-learn, and
threadpoolctl. The serialized models store
parameter dictionaries and arrays. Use the accompanying integrity checks before
loading those files.
