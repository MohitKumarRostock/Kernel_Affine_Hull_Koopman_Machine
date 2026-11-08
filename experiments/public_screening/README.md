# Public-data screening

This experiment examines whether class-risk lower bounds exclude candidates or avoid decoder fits under a fixed tolerance and a declared payload-cost rule. It retains the unsuccessful screening outcomes reported in the manuscript.

The Beijing PM2.5 study evaluates 18 representations with two admissible heads each. The Bike Sharing study uses the same 36-head panel separately for casual, registered, and total counts. These three targets belong to one dataset and satisfy `cnt = casual + registered`.

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

The replay reconstructs the causal observation rosters and lag matrices from the bundled UCI archives, checks training-only target normalization, and recalculates each tolerance as 0.95 times the exact training-persistence MSE. Assessment targets outside the predictor's output interval remain in every loss.

For all 144 heads, it checks class membership, reconstructs decision and test predictions from the fixed head and stored features, and recomputes exact squared-loss enclosures. The resulting 288 complete prediction receipts must match the original source hashes. Affine values use rational arithmetic; extension distances use exact squared dyadic distances and outward integer-square-root bounds. Serialized head sizes are recomputed, and total payload counts are checked against the retained representation-size inputs.

For Beijing, all 396 finite-block witnesses are checked. The replay restores each complete distance table, verifies its original digest, checks primal feasibility and signed-dual objectives with the archived independent arithmetic verifier, and recomputes all 36 decision/test cohort class intervals. Aggregated upper bounds must come from one admissible head on the complete cohort. It then reconstructs the excluded-class set, candidate eligibility, cost-based choices, and measured screening-cost totals.

For Bike, the archived independent auditor verifies the exact two-point optima and duals, disjoint-pair aggregation, deterministic matching, prefit payload bounds, and every search transition. The compact records retain 12 distinct pair certificates with bindings to 60 invocations and all 33 search executions: one exhaustive run and five alternating ordinary/triggered pairs for each target. Reported fit counts, selections, timing medians, and the utility criterion are recomputed from those records.

## Scientific outcome

The Beijing screen excludes zero classes and avoids zero decoder fits. Its selected head agrees with the exhaustive same-rule choice, while the recorded certificate stage adds approximately 18.90 seconds. Failure to exclude a class does not certify its adequacy.

In Bike, the ordinary and triggered searches preserve the exhaustive choice for all three targets. Both fit two heads for casual counts and 36 heads each for registered and total counts in every recorded repetition. The triggered procedure therefore avoids zero additional fits. It makes no pair-certificate calls for casual counts and six calls per repetition for each of the other two targets. The declared incremental-utility criterion fails across the panel.

## Records and scope

`witnesses/` contains losslessly compressed primitives and selected original records. The replay rebuilds omitted redundant prediction and distance receipts and checks their source hashes. `source_inventory.json` records the original paths and file digests. `original/` retains the scientific source, protocols, source-data archives, and cohort records; its historical numeric identifiers are provenance labels.

These results concern the specified finite cohorts and fixed prediction rules. Feature coordinates are trusted operational binary64 outputs interpreted as exact dyadic numbers; this replay does not recertify native feature-generation roundoff or refit representations. Representation payload sizes are inherited recorded inputs to the cost rule. Historical timing samples are reaggregated without remeasurement or independent attestation of execution times. The compact replay does not validate every original lifecycle event or establish population-risk or future-response coverage guarantees.
