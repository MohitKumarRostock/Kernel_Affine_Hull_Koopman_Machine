# Controlled exclusion, adequacy, and calibration

This experiment compares centered block calibration and reuse of the same
disjoint-block dual values to calibrate a lower bound on the full empirical
minimum. It uses 12 complete learning repetitions in each of two bounded
synthetic regimes, with two nested representation classes per repetition.

## Reproduce the manuscript results

From the repository root:

```bash
python experiments/controlled_calibration/replay.py --output results/controlled_calibration.json
```

The command regenerates all 4,096 calibration observations for every learning
case from the frozen random streams, recomputes selected fitted predictions,
evaluates all 190,464 stored block programs, and reconstructs 1,008 one-sided
confidence endpoints and the paired decision counts. No optimizer is called.
The data files use NumPy arrays and JSON; the replay does not load pickles.

At the primary calibration size of 1,024, both methods resolve all 12
repetitions in each regime. At the secondary size of 256, recycled duals
add 11 irreducible-regime exclusions; neither method resolves the repairable
regime. The output includes all three calibration sizes and every learning
case, including unresolved outcomes.

## Evidence and numerical scope

The `records` directory contains all signed duals and primal proposals used
by this replay, their numerical diagnostics, stored feature vectors,
calibration observations, fitted coefficients, and expected endpoints.
`protocol` contains the frozen design. `source` retains the original
producer, independent auditor, and consumed model/runtime source modules.
`provenance` records original file identities and source seals. The original
record identifiers in those source files preserve their scientific identity.

This command verifies the stored mathematical calculations and decisions.
It does not refit the models or rerun native feature construction. The
controlled study uses guarded floating arithmetic and inherited feature
checks; this replay does not upgrade them to an end-to-end interval proof.
The original complete archive, `geometry_step17_reproducibility.zip`, also
contains all training candidate banks and native feature enclosures.

All inputs here are synthetic. The replay needs Python, NumPy, and SciPy
and performs no network access. The original training sources additionally
use scikit-learn and the recorded native numerical runtime.
