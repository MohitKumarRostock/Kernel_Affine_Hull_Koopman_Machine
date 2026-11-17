# Predictor-class screening on Beijing and Bike Sharing data

## Design

The experiment tests whether lower bounds on minimum empirical risk exclude predictor classes or reduce model fitting under fixed mean squared error (MSE) and storage requirements. Beijing PM2.5 uses 18 representations with two admissible fitted predictors each. Bike Sharing uses the same 36-predictor configuration separately for casual, registered, and total rentals. These targets satisfy `cnt = casual + registered` and are not independent datasets.

## Reproduction

Run from the repository root with the pinned dependencies:

```bash
python reproduce.py --only public_screening --output reproduction_screening
```

The group also has a direct entry point:

```bash
python experiments/public_screening/replay.py --output reproduction_screening/public_screening.json
```

The commands use bundled archives and numerical records without network access or model fitting.

## Calculations performed

The checker reconstructs evaluation indices and lag matrices using only observations preceding each target. It checks training-only target normalization and thresholds equal to 0.95 times exact training-persistence MSE. Evaluation targets outside the predictor output interval remain in the losses.

For all 144 fitted predictors, it checks class membership, reconstructs decision and test predictions from stored features and fixed parameters, and recomputes squared-loss bounds. All 288 prediction records must match their original hashes. Affine predictions use rational arithmetic; Lipschitz-extension distances use exact squared dyadic distances and outward integer-square-root bounds. Predictor storage is recomputed; total storage uses the recorded representation sizes.

For Beijing, the checker reconstructs distance tables and verifies all 396 block optimization records, including primal feasibility and dual objectives. It recomputes all 36 decision/test minimum empirical-risk intervals, class exclusions, candidate eligibility, cost-based selections, and recorded screening costs. Aggregated upper bounds must correspond to one admissible predictor over the complete evaluation dataset.

For Bike Sharing, the checker verifies exact two-observation optima and dual values, disjoint-pair aggregation, deterministic matching, storage lower bounds available before fitting, and search transitions. The records contain 12 distinct pairwise bounds used in 60 calls and 33 search executions: one exhaustive run and five ordinary/triggered search pairs per target. It recomputes fitted-model counts, selections, timing medians, and the specified comparison criterion.

## Results

Beijing screening excludes no classes and avoids no predictor fits. It selects the same predictor as exhaustive search under the same selection rule and adds approximately 18.90 seconds of recorded bound-computation time. An inconclusive lower bound does not establish that the class meets the threshold.

For Bike Sharing, ordinary and triggered searches retain the exhaustive selection for all three targets. Each fits two predictors for casual rentals and 36 for each of registered and total rentals in every recorded repetition. The triggered procedure avoids no additional fits. It makes no pairwise-bound calls for casual rentals and six per repetition for each other target. The specified comparison criterion fails across the evaluated tasks.

## Files and numerical assumptions

`witnesses/` contains losslessly compressed numerical inputs and selected original records. `source_inventory.json` identifies original paths and hashes. `original/` contains source code, protocols, source-data archives, and evaluation records. The checker reconstructs omitted duplicate prediction and distance records and verifies their hashes.

Stored features are treated as exact dyadic numbers. The results concern fixed observations and prediction rules; feature-generation rounding error and model refitting are outside this calculation. Representation sizes and timing samples are recorded inputs. Timing summaries do not remeasure or independently authenticate the original runtimes, and the numerical checks do not establish future-observation guarantees.
