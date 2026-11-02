# Block hierarchy on learned KAHM representations

This experiment applies blocks larger than two to actual learned KAHM soft
features. It reuses all 40 archived banks and 80 fixed selected decoders from
`learned_kahm_decision_v1`, then draws **new independent calibration episodes**.
It does not refit the representations or claim new independent training
repetitions. The two constructed sampling laws and the MSE tolerance 0.10
are inherited from the previous study.

## What is compared

- Retained classes with costs 2 and 6, both with sensitivity budget 2.
- Independent block sizes 2, 4, 8, and 16 at n=256, 1,024, and 4,096.
- Both frozen learned history partitions with stratified pairs and estimated
  masses.
- Full-sample scalar Lipschitz empirical minimization at n=256 on **exactly
  the same observations**. Full ERM is not computed at larger sample sizes.
- Upper bounds for the same fixed fitted forecasts.

Every one of the 1,760 endpoints receives the same allocation 0.05/1,760;
stratified endpoints allocate their budget internally. The one simultaneous
event covers both regimes, all fixed banks, both classes, all sample sizes,
all methods, and the reported maxima across alternatives. It is separate
from confidence events in earlier studies.

Every lower block mean, including pairs, uses KL calibration in this study.
The archived learned-decision study used empirical Bernstein lower bounds.
Changes between those studies therefore reflect both fresh observations and
the calibration construction; they cannot be attributed to block size alone.
The within-study block-size comparisons use the same KL rule and allocation.

The primary numerical artifact is a feasible signed-edge dual for each
optimization problem. Positive and negative parts give nonnegative
Lagrange multipliers. A proximal method uses the exact complete-graph
step-size bound and repairs primal candidates using Lipschitz envelopes.
Every stopping-limit outcome and actual primal-dual gap is retained.
The study also retains the development-only benchmark that exposed poor
brackets from the previous dense solver on nearly coincident KAHM features.
It does not alter any earlier results.

## Reproduce

From `extensions/certified_forecasting`:

```bash
python experiments/learned_kahm_hierarchy/run.py freeze --output results/learned_kahm_hierarchy_v1
python experiments/learned_kahm_hierarchy/run.py calibrate --output results/learned_kahm_hierarchy_v1
python experiments/learned_kahm_hierarchy/make_assets.py --results results/learned_kahm_hierarchy_v1
python verification/jmlr_representation_revision/independent_hierarchy_audit.py --results results/learned_kahm_hierarchy_v1 --output verification/jmlr_representation_revision/independent_hierarchy_audit.json
```

Existing freeze and calibration records cannot be overwritten. Use a new
output path for replay. The archived model file remains at its original
path and is pinned by its SHA-256 digest. The source snapshot includes the
feature reconstruction closure, numerical solver, prior sampling-law code,
protocols, and development benchmark records. Versioned NumPy/SciPy/Python
versions are recorded in `protocol_freeze.json`.

## Output records

`samples.jsonl.gz` records every drawn history, scalar target, feature row,
fixed prediction, learned label, seed, and array hash. No row is filtered.
`qp_certificates.jsonl.gz` stores batched signed duals, primal predictions,
raw and guarded dual values, gaps, iterations, convergence flags, and
feasibility residuals for every larger block and full-sample problem.
NumPy's upper-triangular order defines the signed edge coordinates. Pair
scores use the exact closed formula and can be reconstructed from samples.

`endpoints.csv` contains all 1,760 bounded-mean or empirical-minimum
endpoints. `stratified_details.jsonl.gz` preserves conditional means and
mass-polytope minimizers. `decisions.csv` records every method-specific
minimum-cost or joint-infeasibility decision. `summary.csv` and
`decision_summary.csv` include the complete set of means, variation over the
fixed bank collection, and outcome counts. The LaTeX table and two-panel
figure are generated directly from these rows.

The independent audit reconstructs KAHM features without calling the study
runner, recomputes feasible dual objectives in extended precision, solves
the mass minimizations independently, and checks the confidence accounting
and decisions. These numerical checks do not constitute interval-arithmetic
proofs. The conditional nature of inference, constructed mechanism, limited
full-ERM sample size, restricted cost menu, and unresolved outcomes remain
part of the interpretation.

## Disclosed v1 finalization incident

The v1 runner saved all 40-bank numerical records, 1,760 endpoints and 880
decisions, then stopped at its terminal source-hash check: a separate dense
development benchmark remained active despite stop attempts and appended
eight development-only rows to `development_benchmark.json`. The frozen
snapshot and protocol remain unchanged. Their description that the dense
benchmark had stopped early was mistaken; the original frozen log contains
20 rows across both development seeds, and the completed log contains 28.
Both versions are retained under `ancillary_log_incident` in the results.

`provenance_incident.json` records the exact hashes and correction. Every
executable, numerical configuration, archived bank and confirmation RNG design
still matches the prospective snapshot. The log is included in source
inventory checks but is never read by calibration computations. A separate
write-once `finalize_saved_results.py` reconstructed only the terminal
inventory from saved records; it did not regenerate observations, rerun
endpoints or reselect results. The independent audit checks this exception.
The summary truthfully records `source_unchanged=false` and
`executable_sources_unchanged=true`.

The lingering benchmark may have caused runtime contention. Wall times
remain execution diagnostics and support no isolated runtime comparison.
Fresh reproduction is unaffected when no concurrent benchmark updates its
log; the `freeze` stage records the completed development log as it stands.

The solver stops when the unguarded primal SSE minus the guarded dual SSE,
divided by b-1, meets its tolerance. The stored upper value additionally
includes the positive primal roundoff guard, so a stored bracket marked
converged can exceed the requested tolerance by that guard. The actual
stored gap is always retained; convergence flags never determine validity
or inclusion of a lower endpoint. Pointwise feasible-dual domination by the
exact block minimum also covers any numerical batch fallback, without
requiring approximate scores produced in one batch to be independent.
