# Accuracy and complete deployment cost

This experiment evaluates two frozen KAHM cost-selection procedures against
conventional accuracy and cost references. The panel comprises five
replications of one hidden-Markov law and one chronological Appliances
record. It does not supply six independent application domains.

## Reproduce the manuscript results

From the repository root:

```bash
python experiments/deployment_cost/replay.py --output results/deployment_cost.json
```

To also regenerate the saved predictions from raw inputs and the retained
model payloads:

```bash
python experiments/deployment_cost/replay.py --predict --output results/deployment_cost_predictions.json
```

The replay reconstructs exact squared losses from the stored binary64 targets
and forecasts, complete retained numeric payload sizes, and latency medians
from the original nine-repeat timing samples. The primary horizons are four
steps for the controlled law and six steps for Appliances. There are 23
distinct retained primary deployments; the two references coincide in one
case.

The two procedures select from normalized KAHM affinities (`P_cost`) and
clipped reconstruction scores (`CS_cost`). The references are the frozen
conventional accuracy choice (`C_accuracy`) and conventional cost choice
(`C_cost`). Each operational success requires all of the following:

- MSE no greater than 1.1 times the accuracy reference's MSE.
- Strictly fewer retained numeric bytes than the cost reference.
- Median warm-batch time per row no greater than the cost reference.

Each procedure passes the accuracy allowance in all six cases and passes
neither the byte-saving nor the latency criterion in any case. Thus both
have zero joint deployment-utility passes. A failure is retained without
replacing the reference or selecting a different candidate after assessment.

## Evidence and scope

`data` contains the complete primary-horizon assessment rows, selected model
payloads, forecasts, exact expected risks, original timing samples, and
frozen procedure identities. `source` preserves the original protocol,
model implementation, selection procedures, and evaluation code.
`source_provenance.json` and the per-task records identify the original
archive and source files.

The numeric inventory includes the retained encoder and head arrays and
scalars. It excludes Python-object and runtime-library overhead; it is not
a process-memory measurement. Original timing samples are replayed, not
remeasured. The optional prediction replay evaluates the stored operational
models; it does not refit or reselect them, establish ideal real-arithmetic
feature values, or confer a population noninferiority guarantee.

The replay is offline and requires NumPy, SciPy, scikit-learn, and
threadpoolctl. The retained model serialization consists of original
dictionary/array payloads. Use the accompanying integrity checks before
loading those files.
