# Bounded adaptive KAHM: focused development protocol

## Numerical amendment A1

The original complete development attempt produced four correction-head
Cholesky failures and was classified TECHNICAL_BLOCK. All 144 original runs,
387,900 scored forecasts, original source/freeze, analysis and passing
independent consistency audit are retained in `archives/initial_attempt.zip`.
The archive identity is recorded in `archives/initial_attempt_record.json`.

After those outcomes were observed, amendment A1 replaces the correction-head
normal-equation solve with a stable direct solve of the same regularized
least-squares objective from the existing bounded replay. The raw baseline,
data, representation and feature scalers, reference policy, penalties,
calibration grid, chronology, resource screen and advancement thresholds are
unchanged. The repair applies to every geometric arm. The complete original
roster is repeated, not selectively repaired or filtered.

This is an openly documented post-observation development amendment. It does
not create new confirmatory evidence. Separate diagnosis showed that the
largest recurring-regime forecast bursts also arise under fresh stable
least-squares fits; the amendment does not clip or tune away that genuine
extrapolation. See `AMENDMENT_A1.md` and its numerical diagnostic records.

## Status and question

This is a development decision study, authorized after review of the integrated
forecasting manuscript. It is not independent confirmation, a representative
survey of forecasting applications, or a replacement of earlier experiments.
The implementation descends from repository commit
`88e9c568ee983856804a2b2da41f6290f259b8cd`. Earlier source and outcomes remain intact.

The question is whether a bounded adaptive KAHM prototype supplies enough
predictive value beyond equally adaptive geometric controls and a practical
kernel comparator to justify a larger study on newly reserved sources. A
negative decision concerns this prototype and hypothesis, not all KAHM methods.

## Roster and development sources

There are twelve streams, six methods and two capacity tiers: 144 runs.

* Four seeds (20261101 through 20261104) of a stationary nonlinear scalar AR
  stream and the same four innovations under recurring regimes: eight streams.
* All three previously selected Dominick weekly series, T7830, T25851 and
  T108533 (393 observations each).
* The previously selected Sunspot daily series, the preserved final 4096
  observations of T1.

Dominick is intentionally a previously favorable example and Sunspot a
previously unfavorable example. These choices use known findings and therefore
cannot supply new-source confirmation. The natural raw arrays, upstream
checksums, identifiers and CC BY 4.0 attribution are retained. We use complete
raw arrays rather than the former subsampled A/B/C/D rows. DATA_PROVENANCE.md
and prepared_inputs/manifest.json are authoritative for exact input details.

The synthetic equation is

    y[t] = mu[s] + a[s]*(y[t-1]-mu[s]) + 0.10*(y[t-2]-mu[s])
           + b[s]*tanh(1.5*(y[t-1]-mu[s])) + 0.20*epsilon[t].

Regime A has (mu,a,b)=(-0.75,0.30,0.35), and B has
(0.75,0.45,-0.25). Innovations are independent standard normal, with PCG64
seeds paired between schedules. After 256 A-only burn-in points, retain
2048 points. The recurring schedule alternates A/B every 256 retained points;
the stationary schedule stays in A. Regime and innovation arrays are analysis
metadata and never model inputs. The histories can reveal regime information,
but no oracle identification or predictable transition assumption is given
to the models. This is a new simple generator, not a claimed inherited
Duffing/Van der Pol implementation or a test of every kind of recurrence.

Histories/horizons/seasonal lags are respectively 12/(1,4,12)/1 for synthetic,
52/(1,2,4)/52 for Dominick, and 28/(1,3,7)/1 for Sunspot. Seeds 20261097,
20261098 and 20261099 are excluded mechanics/timing streams. No reported
stream's forecasting accuracy is used to change this protocol.

## Chronology and calibration

The first floor(0.35*n) observations form the initial fitting prefix; the
next observations through floor(0.50*n)-1 are calibration. The latter half
is the reported development evaluation. Initial training uses joint-horizon
complete pairs whose most distant target is strictly inside the initial
prefix; a few otherwise available shorter-horizon pairs are conservatively
omitted from initialization. All methods receive the same initial rows.

At observation time t:

1. Score previously issued predictions whose target index equals t.
2. Release only newly matured history/target pairs to each horizon head.
3. Update the model using those labels and the currently observed history.
4. Issue and permanently store forecasts for future indices t+h.

A label is never supplied before its target time. The representation may use
currently observed inputs without their future labels. On the first evaluation
origin, select each horizon's mixing weight BEFORE processing that origin's
new observation. Candidate weights are 0, 0.25, 0.5 and 1, evaluated on saved
calibration forecast components whose targets are strictly before evaluation.
The first minimum wins, so exact ties prefer zero. Updates do not depend on
the candidate weight, making all four component combinations legitimate
calibration candidates. Selected weights are then fixed. There is no further
selection from evaluation outcomes and no refit at the calibration boundary.

The algorithm is fixed, but it is allowed to learn from matured earlier
evaluation-period observations. Saved forecasts are never overwritten after
a model refresh. Evaluator access to complete input arrays is separate from
the restricted model API. Future-perturbation tests check that distinction.

All selected streams are finite. The general rule is no imputation: forecast
only with a finite current history and score a horizon when its target is
observed. An unexpected model exception disables that run's model and uses a
phase-correct seasonal forecast thereafter; the exception, timing, and all
scorable predictions remain recorded. Such an exception blocks a positive
development recommendation, even when the exception belongs to a comparator.

## Methods and bounded state

The six method identifiers are:

* `ridge`: sliding-window regularized raw-history ridge.
* `fixed_kahm`: fixed initial KAHM geometry with continuously updated raw and
  residual ridge heads.
* `adaptive_kahm`: the same construction with bounded local geometry refreshes.
* `adaptive_affine`: the identical adaptive/reference/readout construction,
  replacing KAHM reconstruction with affine projection.
* `adaptive_rbf`: the identical construction with RBF-local center-distance
  gates and gated local coordinates.
* `kernel_rr`: sliding-window Gaussian kernel ridge with an unpenalized
  intercept and recursive add/delete Gram-inverse updates. This is not the
  published KRLS-T algorithm.

The small tier permits 64 shared geometric references and 128 replay rows
per horizon; the large tier permits 128 references and 256 replay rows.
There are four local models and rank at most two. Each reference cap is split
among local models; actual support can be smaller when few points are present.
The kernel method has up to M supports PER HORIZON, whereas the geometric
bank shares M across horizons. These are capped capacity tiers, not equal
byte allocations. Actual NumPy state and whole-process memory are reported.

All input scalers are fitted on the initial fitting rows and remain fixed.
Raw ridge and residual ridge use sum-squared-error penalties of 1, with a free
intercept. Under amendment A1 the correction is solved directly from bounded
replay against residuals of the CURRENT raw baseline as it changes; obsolete
baseline residuals are not silently reused. The original accumulated
cross-statistics implementation is preserved in the first-attempt archive.
These are bounded-window objectives, not a claim that every candidate is
optimized across all historical data.

Initial local models use the same seeded four-cluster partition and reference
selection for every geometric family. Gates plus gated local principal
coordinates give 12 features (zero-padding deficient directions). KAHM retains
the normalized signed reconstruction, covariance-projected Gaussian kernel,
reference-energy regularizer iteration and radial/angular folding score from
the manuscript. Kernel bandwidths and gate temperatures are not re-optimized
in this pilot; this is an explicitly simpler adaptive prototype, not a
streaming reproduction of the earlier six-parameter direct-training model.

Every 16 observation steps, an adaptive method considers one local model in
round-robin order. Initial assignment centers remain fixed for the common
reference policy. Half the initially chosen references remain anchors. At
most four distinct recent references are inserted/replaced in the remaining
slots, using only observed histories assigned to that local model. The affected
center, basis, covariance and factor are rebuilt. Geometric feature scaling
and every residual head are coherently rebuilt on their bounded replay rows.
All this work is charged. This is a bounded local rebuild, not a rank-one
update of a changed Gaussian metric and not a complete full-bank refit.

KAHM's relative signed-denominator tolerance is 1e-10. Unsafe prediction rows
receive zero geometric correction; unsafe fitting rows are excluded from the
geometric objective and counted. Factor failures and all exclusions are
recorded. A rank-zero local model uses its center. Geometry families share the
same reference policy, but their functional capacities and costs are not equal.

The kernel baseline uses an initial-prefix median squared-distance bandwidth,
regularizer 0.1 and bounded FIFO support. Its recursive inverse is checked
against the current Gram system every 128 updates, with recorded rebuilds if
the relative residual exceeds 1e-7. It preserves an unpenalized intercept.

Calibration chooses only the mixing weights. No hyperparameter search over
penalties, kernel bandwidths, refresh rules, reference policies or histories is
performed. A negative result therefore constrains this modest prototype.
Adaptive geometric state is maintained even if a selected correction is zero;
reported costs describe this eager prototype and are not minimum deployment
costs for a completely disabled correction. Unused observed-input buffers are
removed from ridge, fixed-KAHM and kernel models after initialization.

## Losses, resources and decision

Primary loss is mean squared error divided by the initial-prefix mean squared
seasonal difference. If that scale is nonpositive/unavailable, use initial
first-difference MSE and then 1.0. No selected task needs the fallback. Average
origins within horizon, horizons within series, and series within collection.
The two natural collections receive equal weight. Stationary and recurring
synthetic groups remain separate; they are not pooled with real data.

Report every series/seed and horizon, calibration choices, fallback activity,
geometry changes, and all pairwise comparisons of adaptive KAHM with the five
other methods. Four synthetic seeds and two selected natural collections do
not justify broad transfer intervals. Results and paired win counts are
descriptive; overlapping windows are not independent experimental replications.

Workers run serially in fresh processes with numerical threads fixed to one
and the same available CPU affinity. A fixed shuffled roster avoids a method
always occupying one temporal order. Initialization and calibration work are
reported separately. For each subsequent observation, record supervised
update time, prediction time and the complete service interval (including
pair/history assembly). The interval includes all matured horizons, refreshes,
kernel inverse audits and feature/head rebuilds. Scoring, serialization and
audit state inspection are outside service timing. Report mean, median,
p95, p99, maximum and overrun counts.

The engineering allowance is 20 ms per observation, on the recorded host.
This is a deadline-FEASIBILITY SCREEN: no hard scheduling limit is enforced.
It is not an application-derived SLA. An excluded timing-only smoke run
validates mechanics before the freeze; no reported accuracy informs the limit.
Record model-owned unique NumPy buffer bytes separately from whole-worker
peak RSS. The latter includes interpreter, libraries and evidence storage;
neither statistic alone is a precise incremental runtime-memory measurement.

The advancement heuristic is deliberately conservative and evaluated separately
for each COMPLETE tier, with no mixing accuracy at one tier and speed at another:

1. On recurring synthetic streams, adaptive KAHM must improve mean scaled MSE
   by at least 3% versus each of adaptive affine, adaptive RBF-local and kernel
   ridge, with wins on at least three of four paired seeds for each. It must
   also improve fixed KAHM and raw ridge by at least 1% there.
2. On the equal-collection natural aggregate it must improve all three main
   comparators by at least 1%. One SAME collection must improve all three by
   at least 3%, and neither collection may deteriorate by more than 5% against
   any of them.
3. Adaptive KAHM's p99 service time must meet 20 ms on EVERY stream in that
   tier. There must be no unexpected run exception or cap violation in ANY
   method; a broken comparator cannot support a positive conclusion.

PROCEED if one complete tier passes. REFINE if the synthetic mechanism gate
passes but natural/resource criteria fail. Otherwise DEFER this specific
extension. Any technical failure produces TECHNICAL_BLOCK instead. Thresholds
are development triage choices, not a hypothesis-test rejection of KAHM or
universal practical-effectiveness margins.

## Evidence and validation

Before each complete run, DEVELOPMENT_FREEZE.json records hashes of protocol,
configuration, numerical code, analysis, tests and every prepared input. A Git
checkpoint fixes the source state. The original freeze is retained inside the
first-attempt archive; the current freeze identifies amendment A1 and links
that archive. These are local checkpoints, not public preregistration.
Scientific corrections require explicit amendments and retained original
records, not silent rewriting.

Pre-run tests compare recursive heads/kernels with independent batch objectives,
check KAHM algebra and regularization, replay/bank caps, matched references,
coherent geometry refresh, target maturity, future-perturbation invariance,
calibration timing and exact baseline recovery at zero correction. The audit
independently rebuilds truth, losses, calibration selection, collection
aggregation, gate outcomes, timing summaries, coverage and capacity records
from immutable forecasts/traces without importing the forecasting model.

All predictions, timestamps, calibration candidates, failed outcomes, measured
resources and input/source hashes are preserved. Replaying computation checks
consistency; it does not turn this development experiment into independent
confirmation or provide empirical support for an outcome that was not observed.
