# Public-data screening

This experiment examines whether class-risk lower bounds exclude candidates or avoid predictor fits under a fixed error tolerance and a specified model-storage criterion. The results include unsuccessful screening outcomes.

The Beijing PM2.5 study evaluates 18 representations with two admissible predictors each. The Bike Sharing study uses the same 36-predictor panel separately for casual, registered, and total counts. These three targets belong to one dataset and satisfy `cnt = casual + registered`.

## Reproduce

From the repository root:

```bash
python reproduce.py --only public_screening --output reproduction_screening
```

The group can also be run directly:

```bash
python experiments/public_screening/replay.py --output reproduction_screening/public_screening.json
```

The replay requires the repository's recorded Python dependencies. It uses the bundled data archives and compact numerical records, with no network request or model fitting.

## Verification

The replay reconstructs the observation indices with histories restricted to data available before each target and lag matrices from the bundled UCI archives, checks training-only target normalization, and recalculates each tolerance as 0.95 times the exact training-persistence MSE. Assessment targets outside the predictor's output interval remain in every loss.

For all 144 predictors, it checks class membership, reconstructs decision and test predictions from the fixed predictor and stored features, and recomputes exact squared-loss enclosures. The resulting 288 complete prediction records must match the original source hashes. Affine values use rational arithmetic; extension distances use exact squared dyadic distances and outward integer-square-root bounds. Serialized predictor sizes are recomputed, and total model-storage sizes are checked against the retained representation-size inputs.

For Beijing, all 396 group-level optimization records are checked. The replay restores each complete distance table, verifies its original digest, checks primal feasibility and signed-dual objectives with the archived independent arithmetic verifier, and recomputes all 36 decision/test evaluation dataset class intervals. Aggregated upper bounds must come from one admissible predictor on the complete evaluation dataset. It then reconstructs the excluded-class set, candidate eligibility, cost-based choices, and measured screening-cost totals.

For Bike, the archived independent auditor verifies the exact two-point optima and duals, disjoint-pair aggregation, deterministic matching, bounds on model-storage size computed before fitting, and every search transition. The compact records retain 12 distinct pair certificates with bindings to 60 invocations and all 33 search executions: one exhaustive run and five alternating ordinary/triggered pairs for each target. Reported fit counts, selections, timing medians, and the utility criterion are recomputed from those records.

## Scientific outcome

The Beijing screen excludes zero classes and avoids zero predictor fits. Its selected predictor agrees with the exhaustive same-rule choice, while the recorded certificate stage adds approximately 18.90 seconds. Failure to exclude a class does not certify its adequacy.

In Bike, the ordinary and triggered searches preserve the exhaustive choice for all three targets. Both fit two predictors for casual counts and 36 predictors each for registered and total counts in every recorded repetition. The triggered procedure therefore avoids zero additional fits. It makes no pair-certificate calls for casual counts and six calls per repetition for each of the other two targets. The declared incremental-utility criterion fails across the panel.

## Records and scope

`witnesses/` contains losslessly compressed primitives and selected original records. The replay rebuilds omitted redundant prediction and distance records and checks their source hashes. `source_inventory.json` records the original paths and file digests. `original/` contains the scientific source, protocols, source-data archives and evaluation dataset records. Its numeric identifiers map those files to their source records.

These results concern the specified finite evaluation datasets and fixed prediction rules. Stored feature coordinates are treated as exact dyadic numbers. The replay does not bound roundoff introduced by feature generation or refit representations. Recorded representation-storage sizes are inputs to the cost criterion. Timing summaries aggregate saved measurements without remeasurement or independent authentication of execution times. The replay checks the supplied numerical records; it does not verify every data-acquisition and training event or establish guarantees for future observations.
