# Controlled exclusion, adequacy, and calibration

This experiment compares two ways to construct confidence bounds for the
best mean squared error in a predictor class. One calibrates the average of
minimum losses on disjoint groups of observations. The other reuses feasible
dual values from those groups to bound the minimum loss over the full
calibration dataset. It uses 12 complete training repetitions in each of two
bounded synthetic settings, with two nested feature sets per repetition.

## Reproduce the numerical results

From the repository root:

```bash
python experiments/controlled_calibration/replay.py --output results/controlled_calibration.json
```

The command regenerates all 4,096 calibration observations for every learning
case from the fixed random streams, recomputes selected fitted predictions,
evaluates all 190,464 stored block programs, and reconstructs 1,008 one-sided
confidence endpoints and the paired decision counts. No optimizer is called.
The data files use NumPy arrays and JSON; the replay does not load pickles.

At the primary calibration size of 1,024, both methods resolve all 12
repetitions in each regime. At the secondary size of 256, reusing the dual values
excludes both tested classes in 11 additional repetitions when the target is
independent of the observed history. Neither method resolves the setting with a
history-dependent target at that calibration size. The output includes all three
calibration sizes and every learning case, including unresolved outcomes.

## Evidence and numerical scope

The `records` directory contains all signed duals and primal proposals used
by this replay, their numerical diagnostics, stored feature vectors,
calibration observations, fitted coefficients, and expected endpoints.
`protocol` contains the fixed design. `source` retains the original
producer, independent auditor, and model and numerical-runtime source modules used by the experiment.
`provenance` records source-file identities and checksums. The original
record identifiers in those source files preserve their scientific identity.

This command verifies the stored mathematical calculations and decisions.
It does not refit the models or rerun feature construction. The
study uses floating-point calculations with numerical safeguards and recorded
feature checks. The replay does not provide interval bounds for every step of
feature generation and optimization.
The original complete archive, `geometry_step17_reproducibility.zip`, also
contains all candidate models considered during training and coordinate-wise bounds on generated features.

All inputs here are synthetic. The replay needs Python, NumPy, and SciPy
and performs no network access. The original training sources additionally
use scikit-learn and the recorded numerical runtime.
