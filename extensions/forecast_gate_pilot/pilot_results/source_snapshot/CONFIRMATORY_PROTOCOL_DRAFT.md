# KAHM continuation: protocol for a distinctive predictive contribution

Status: research design, not a registered or executed confirmatory study.
Prepared 31 October 2026 from the supplied revised manuscript. The separate
known-law pilot in this package is exploratory development evidence only.

## 1. Claim and estimand

The candidate claim is that directly forecast-trained KAHM gates or
reconstruction parameters improve a declared final forecast at a declared
horizon relative to equally supervised local alternatives and a strong
raw-input comparison, at measured acceptable cost.

The first endpoint is the equal-collection mean of equal-series scaled MSE,
with predeclared origin and horizon weights. Fix each loss scale on the
initial training prefix. Do not pool losses with any historical study.
The statistical unit for new-source transfer is the independent collection
or other explicitly justified independent source unit, not overlapping
windows, horizons, fitting seeds, or series from the same collection.

## 2. Minimum controlled method family

Use the same frozen base f0, available histories, raw-input standardizer,
partition/reference selection, local-coordinate width, target availability,
head penalties, zero-correction option, initializations, and optimization
ceilings for KAHM, affine, locality-aware affine, and center-distance RBF
local models. Cross each geometry with fixed versus directly trained gates.

Keep raw ridge, raw RBF and persistence/seasonal naive. Before confirmation,
select at least one additional strong practical raw-input family appropriate
to the sources and resource envelope, if existing raw models are inadequate.
Choose its hyperparameters using development and selection data only.

Two distinct fairness analyses are required:

1. Equal supervision and optimization opportunity: equal parameter counts
   where feasible, identical data and horizon targets, same restart count,
   hard objective-evaluation ceilings, stopping rules and search grids.
2. Equal resources: compare accuracy under equal measured wall-time and
   memory envelopes. Equal optimizer steps do not equate computational cost.

## 3. Forecast objective and stable baseline

For geometry g, horizon h and parameter theta, fit a ridge head on support A:

    beta_g,h(theta, alpha) = argmin_beta
        ||r_A,h - F_A,g(theta) beta||^2 + alpha ||beta||^2

where r_A,h = y_A,h - f0,h(x_A), the intercept is unpenalized and all feature
standardizers are fitted on A. Fit theta against actual forecast error on a
separate development query segment B:

    J_g,h(theta, alpha) = mean_B
        (y_h - f0,h(x) - q_g,h,theta,alpha(x))^2.

A training-only positive scale can normalize this objective for optimization
without changing its minimizer. Any parameter regularization must be stated.
B trains geometry and is not confirmation or a final validation set.

Choose head penalty and correction weight on C. Hold the selected base,
bank, head, scaler, and correction weight fixed for the static final D
evaluation. No pre-D refit is allowed in the first controlled experiment.
If refitting is later studied, realign the residuals and re-evaluate it as
a separate complete procedure. Donor residuals are not bundled into the
direct-objective intervention.

## 4. Minimal gate prototype and broader method

The executable pilot freezes the bank and trains one log inverse-temperature
per partition block and horizon:

    phi_c,h(x) = softmax_in_block(exp(eta_b,h) * ell_c,g(x)),
    -log(4) <= eta_b,h <= log(4).

All four gates receive exactly three trainable parameters per horizon.
Each final forecast retains the same affinity-gated principal coordinates.
This is direct forecast-trained gating. It is not end-to-end training of
the KAHM kernel, reconstruction operator, PCA, references, or input metric.

The next richer candidate may train kernel bandwidth, regularization or
bounded low-rank/horizon-specific metrics through the final ridge solve.
Freeze hard partitions, reference membership and rank decisions initially.
Count every trainable degree of freedom and match the optimization allowance
for the controls. A common supervised metric alone does not identify the
KAHM reconstruction rule as the cause of an improvement.

For fixed references Q and A_theta = K_theta + lambda_theta I, write
a = solve(A_theta, k_theta), s = 1' a and r = Q a / s. Where s is nonzero,

    da = solve(A_theta, dk - dA a),
    dr = (Q - r 1') da / s.

This standard differentiation identity shows why signed-denominator
cancellation matters. Common log-kernel shifting avoids underflow but does
not bound 1/|s|. Prespecify the numerical safety rule and fallback, record
its rate, and do not silently turn signed KAHM weights into softmax weights.

## 5. Reserved evidence and chronology

All previously inspected collections remain development/history:
Electricity weekly, NN5 daily, Tourism monthly, FRED-MD, Traffic weekly,
Hospital monthly, Saugeen River, US births, Weather, Solar Weekly,
COVID-19 Deaths, Car Parts, Australian state electricity, Bitcoin activity,
Melbourne pedestrian counts, and KDD2018 air quality.

New simulation seeds are fresh draws on known laws. They do not establish
new-source generalization. A genuinely untouched later period supports a
temporal-transfer claim on that source, which is distinct from source transfer.

Before confirmation, seal the source roster, eligibility checks, prepared
arrays, release/vintage assumptions, training/query/selection/test boundaries,
method code, seeds, capacities, failure rules, inference plan and resource
limits. All targets for an earlier split must be available before the next
query origin; use actual label-release timestamps where relevant. Do not
choose a dataset, horizon, method or threshold after inspecting D outcomes.

## 6. Comparison, uncertainty and decisions

Freeze one primary comparator policy that selects the strongest eligible
non-KAHM procedure using C only. Charge that policy for every candidate it
needs. The primary endpoint compares directly trained KAHM with that policy
on D; never use a test-set oracle that picks the smallest D loss.

Also report prespecified paired comparisons against fixed KAHM, directly
trained affine, locality and RBF-local, the strongest eligible fixed local
control, and the raw-input policy. Declare the multiplicity family before
execution. Per-horizon and per-source tables are retained regardless of sign.

Choose the uncertainty method and sample size for the source sampling frame
and desired practical improvement. A source-cluster interval needs explicit
sampling/exchangeability assumptions; a small purposive roster warrants a
bounded descriptive claim instead. Do not claim equivalence from a
nonsignificant superiority comparison. Power planning uses source-level
development variability, not the number of overlapping forecast windows.

Record three decisions separately:

- Mechanism: does direct training improve fixed KAHM under the frozen design?
- Distinctiveness: does KAHM improve the matched non-KAHM alternatives and
  the practical raw-input policy on genuinely reserved evidence?
- Utility: is the gain large enough for the declared accuracy margin and
  latency/memory envelope, including selection and failed work?

The minimum useful gain and absolute resource ceilings depend on the intended
application and are still to be fixed before confirmation. This draft does
not invent those requirements or call a relative timing ratio a utility test.

## 7. Bounded updating as a separate claim

Begin with fixed geometry epochs. Within each epoch freeze the scaler,
metric, references, centers, bases, kernel settings, feature standardizer
and residual baseline. Maintain a fixed-capacity sliding window of fully
labeled pairs and update the readout sufficient statistics exactly.

The included generic online module uses a positive sum-loss ridge penalty
on every supplied feature, including a supplied constant feature. This
intercept convention differs from the static pilot and is explicit; the
module is a numerical building block, not an evaluated online forecaster.

An update of the geometry or baseline requires resetting or rebuilding the
head from a bounded raw-data replay buffer under the new feature version.
Never mix statistics from different geometry versions. Changing a KAHM
reference normally changes PCA/covariance and the energy-based regularizer;
an inexpensive kernel row/column update is exact only if those quantities
are deliberately frozen. That is a changed fitting rule.

A refresh has fixed limits on reference count, replay size, optimizer work
and partition work. Keep the old immutable model active while constructing
the replacement; publish it atomically. Include both versions in the memory
budget. Either spread bounded work across ticks or report the entire refresh
spike; amortized latency alone is insufficient.

At time t, train a complete forecast target from origin s only after every
required label has been released (s + h_max <= t with zero release delay).
Store each prediction before its target is available. Score the stored
prediction before allowing that label to influence later forecasts. Freeze
the entire adaptation/retirement policy before the evaluation stream opens.

## 8. Resources, failures and retirement

Measure forecast latency and update latency separately: median, p95, p99,
maximum, refresh pauses and publication delay. Repeat on fixed hardware and
thread settings. Measure end-to-end wall time, resident state, raw replay,
pending-label queues, working allocations, and process peak memory.
Serialized bytes and the historical component-sum timings are insufficient.

Retain every task, warning, failure, fallback and attempted cost. A zero
correction option does not guarantee that a fit completes or generalizes.
Use a declared fallback to the available base if the required geometry fails;
if the base is unavailable, retain the task failure rather than dropping it.

A retirement policy may use only matured losses of stored, contemporaneous
shadow forecasts. Freeze its window, threshold, tie handling and reactivation
schedule. Charge monitoring and rejected probes. No future-benefit guarantee
follows from its selection rule.

## 9. Required artifacts and stopping rule

Retain code/environment hashes, prepared input hashes, split/origin IDs,
model versions, target release times, every validation candidate, final
prediction arrays, exact loss aggregation, compute dependencies and failures.
An independent audit reconstructs losses and decisions from saved outputs.

Do not expand the method until the minimal gate experiment supplies a useful
development signal or an identified capacity limitation. If equally trained
simple gates keep matching or beating KAHM, report that result and prefer the
simpler predictor for that scope. Efficient KAHM engineering remains a
separate contribution only when it improves the actual accuracy-cost frontier.

## References

The supplied manuscript, especially mechanisms.tex, utility_question.tex,
section_target_geometry.tex, appendix_target_geometry.tex, discussion.tex,
and the generated target_geometry tables.

Bertinetto et al. (2019), Meta-learning with differentiable closed-form
solvers, ICLR. https://openreview.net/forum?id=HyxnZh0ct7

Van Vaerenbergh et al. (2010), Fixed-Budget Kernel Recursive Least-Squares,
ICASSP. https://gtas.unican.es/files/pub/vanvaerenbergh2010_fbkrls.pdf

Cawley and Talbot (2010), On Over-fitting in Model Selection and Subsequent
Selection Bias in Performance Evaluation, JMLR 11:2079-2107.
https://jmlr.org/papers/v11/cawley10a.html
