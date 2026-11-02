# Direct forecast-trained KAHM reconstruction: frozen evaluation protocol

## Status and question

This protocol defines a new experiment after the shared proxy-metric study
and a separately preserved nine-task exploratory gate-temperature pilot.
The protocol, numerical code, analysis, exact roster and prepared inputs
will be published in one Git commit before the full reserved evaluation.
This is a prospective public execution checkpoint, not external registration.
Its motivation and design use the already inspected historical evidence.

The question is whether training KAHM reconstruction bandwidths for the
final forecast and horizon improves prediction beyond gate-only training,
equally supervised local alternatives, and practical raw-input baselines.
The study establishes a specified forecasting procedure and bounds its
empirical usefulness. Comparative superiority is an outcome to test.

No outcome from this experiment is used to revise its method, roster,
penalties, stopping rules, or analysis. Any execution amendment must be
dated, preserve the previous freeze, and distinguish debugging from a new
scientific procedure. Historical findings and failed fits remain intact.

## Forecasting procedure and data roles

Four disjoint roles are used. A fits the input scaler, local banks, baseline
and correction heads. B trains the geometry against actual forecast error.
C selects the correction penalty and mixing weight, separately by horizon.
D evaluates the selected frozen procedure. D targets are not arguments to
the model-fitting API. There is no pretest refit and no donor-residual
intervention in this study. The baseline is selected by an inner A split,
refitted on all A, and then held fixed for every geometric arm.

Synthetic inner A validation excludes complete episodes. Natural inner A
validation is chronological, with training targets matured before the
first validation query. Raw histories and feature standardizers use A
only. Each correction learns the in-sample residuals of this same final
baseline. A zero correction recovers its forecast exactly.

For geometry g, horizon h, parameter theta and ridge penalty alpha, solve

    (beta, intercept) = argmin sum_A (y_h - f0_h(x)
                             - intercept - F_g,h(x;theta) beta)^2
                             + alpha ||beta||^2.

The intercept is unpenalized; features are standardized using A. The loss
convention is sum squared errors, not average squared errors. Minimize
mean B forecast squared error with mixing weight one, divided by an A-only
positive target-variance scale. Then C chooses alpha from {0.01,0.1,1} and
gamma from {0,0.25,0.5,1}. Zero is first and only strict improvement replaces
the current C incumbent. There is no guarantee of D improvement from the
available zero candidate.

## Matched local banks and interventions

The shared A bank requests 2, 4 and 8 groups, up to 24 references per group,
and local principal rank at most two. All four families retain affinity
weights and their gated principal coordinates. Frozen memberships, reference
indices, principal directions, covariance estimates and standardization are
shared. The recovered upstream fitting implementation is used and its
dependencies are included in the scientific freeze. KAHM-specific kernel
factor work is charged to the KAHM arms, not the affine/locality controls.

Each block b and horizon h has bounded parameters xi and eta in
[-log(4),log(4)], initialized to zero. Eta multiplies that block's logits
by exp(eta), followed by a within-block softmax. Xi changes the local
shape or reconstruction as follows (z is standardized history, P the local
principal basis, mu its reference mean, s and v the fixed center and normal
scales).

* KAHM: K_ij(xi)=exp(-D_ref,ij/exp(2xi)) and
  k_i(xi)=exp(-D_query,i/exp(2xi)). D is the inherited projected,
  covariance-whitened squared distance divided by twice the local rank.
  Rank-zero degenerate local models use their fixed center without a
  kernel solve. The original fitted positive kernel regularizer is held fixed.
  Reconstruction uses the signed normalized solution
  Q(K(xi)+lambda I)^(-1)k(xi)/(1'(K(xi)+lambda I)^(-1)k(xi)).
* Affine: reconstruction is mu + exp(xi) PP'(z-mu).
* Locality: the logit is -exp(xi)||z-r_affine||^2/v
  -0.1 exp(-xi)||z-mu||^2/s.
* RBF-local: the logit is minus
  [exp(xi)||P'(z-mu)||^2 + exp(-xi)||(I-PP')(z-mu)||^2]/(2s).

The KAHM and affine reconstructions feed the inherited folding score and
logit encoding. All families equal their fixed form at xi=eta=0.
The gate-only arm fixes xi=0 and trains three eta parameters per horizon;
the reconstruction/shape arm trains six parameters per horizon. Equal
parameter counts within each mode do not imply equal functional capacity.
The term reconstruction-trained describes KAHM's intervention; the other
families receive their stated matched shape intervention.

Both trainable modes receive the same hard ceiling of 256 actual objective
evaluations per horizon and alpha, the same deterministic L-BFGS-B optimizer with finite differences,
maxiter=40, maxls=20, ftol=1e-10 and gtol=1e-6, initialization and stopping
rule. Every evaluated candidate and the best
finite incumbent, including identity, are recorded. The implementation's
exact optimizer configuration is part of the freeze. A budget exit is
reported separately from convergence. Equal evaluation ceilings measure
optimization opportunity, not equal wall time or equal memory.

Common log-kernel shifting protects query evaluation from underflow. A
relative signed-denominator threshold of 1e-10 guards cancellation. Unsafe
A/B candidates are rejected; unsafe C candidates cannot replace the
incumbent. If a selected KAHM map fails on D, the entire affected horizon
uses the frozen baseline and records the fallback and its paid work. The
guard does not replace signed weights with positive softmax weights.

## Complete method roster

There are 17 procedures: fixed, gate-trained, and reconstruction/shape-trained
versions of KAHM, affine, locality and RBF-local (12); raw ridge; raw-history
plus Nystrom RBF; raw histogram gradient boosting; seasonal naive (or
persistence where the declared period is one); and one C-selected non-KAHM
policy. Every planned method, failure and fallback remains in the report.

The raw ridge uses a seven-value penalty grid 1e-4 through 1e2, selected
inside A. Raw RBF retains raw standardized history alongside up to 64
Nystrom features; its bandwidth factors {0.5,1,2} and seven penalties are
selected on B. These baselines are identified by their present complete
specification, not treated as identical to earlier study implementations.

The histogram gradient boosting reference trains on A with 100 iterations,
L2 penalty one, learning rate {0.05,0.1}, and maximum leaves {7,15}.
External validation early stopping is disabled. Its hyperparameters are
selected on B. Seasonal naive uses the correct lag for each horizon and
the declared collection period, always available within the raw context.

Raw RBF and gradient boosting also retain the frozen ridge state for a
logged horizon-wise D fallback. A D prediction failure never changes C
forecasts, selected hyperparameters or policy choice. The additional
baseline computation and serialized state are included in their costs.

The non-KAHM policy selects one complete non-KAHM procedure per task by
aggregate C MSE with a frozen strict first-minimum order. It may select any
of the nine non-KAHM local arms or the four raw/naive baselines. It never
uses D to choose the comparator. Its cost is the union of all required
candidate computation nodes, counting shared work once.

## New simulation repetitions

The inherited nonlinear AR, Van der Pol and Duffing generators are used
with seeds 84200 through 84211: 36 tasks. These are fresh repetitions of
known laws, not new-source evidence. Each task has independent A/B/C/D
episodes numbering 64/32/32/128, each 72 observations, history length 12,
and offsets 1,4,12. There are 13 forecast origins per episode. Deterministic
row caps are A512, B256 and C256; all 1,664 D rows are retained. Raw episodes,
origin indices, selected rows and generator dependencies are frozen.
Development smoke checks use separate seeds, including 932001, and do not
enter the study roster or reported loss aggregates.

The prior gate pilot used different generator code, three seeds per law,
and 160 objective evaluations. It remains exploratory and separate from
the present 12-repetition analysis.

## New natural collections and chronological protocol

The purposive roster contains four collections and ten series. All are
absent from the 16 previously inspected collections. These are retrospective
single-vintage archive experiments; they do not establish historical
real-time release availability or performance under later data revisions.

| Collection | Pinned Zenodo record | Series | History | Offsets | Seasonal period |
|---|---:|---:|---:|---|---:|
| Dominick weekly retail | 4654802 | 3 | 52 | 1,2,4 | 52 |
| Rideshare hourly, price_mean only | 5122114 | 3 | 24 | 1,3,6 | 24 |
| Wikipedia daily, missing values retained | 4656080 | 3 | 28 | 1,3,7 | 7 |
| Sunspot daily, missing values retained | 4654773 | 1 | 28 | 1,3,7 | 1 |

The exact source URLs, license, remote MD5, downloaded SHA256, selected
identifiers and preparation rules are retained in the source manifest.
Only the selected prepared inputs and provenance are versioned; large
source archives can be downloaded from the pinned records. Sunspot uses
the last 4,096 observations, without value-dependent endpoint selection.
Its single series is not split into fictitious independent sources.

A/B/C/D use chronological 50/20/15/15 percent boundaries. Every target
falls inside its own segment; history may use previously observed values
from earlier segments. Caps are A512, B128, C128 and D128. Windows with
missing history or targets are omitted, without interpolation. The estimand
therefore concerns the recorded available windows. All original indices,
segment boundaries, selected origins and availability counts are saved.

Eligibility uses the A prefix only: at least 96 finite A windows and a
positive finite training-only seasonal-difference MSE (with first-difference
fallback). A fixed hash ordering selects eligible series. No later segment's
values or forecast outcomes determine eligibility or replacement. A series
with insufficient later windows or an unsuccessful fit remains a recorded
failure rather than being replaced. See the frozen preparation source and
manifest for the hash salt and exact boundary arithmetic. The pre-freeze
design log records an A-only availability check: the originally proposed
48-hour rideshare history gave no eligible series, so the frozen history
is 24 hours, retaining the declared daily seasonal lag and eligibility
threshold. No forecast or D outcome informed this feasibility change.

A preparation audit after A-only selection found that all three selected
rideshare series have no complete B windows. They remain in the frozen
roster as anticipated feasibility failures. The full four-collection
accuracy aggregate is therefore undefined for this procedure. A separately
labelled available-case description may average the seven evaluable series
within their three collections, and then weight those collections equally;
it does not replace the planned four-collection estimand. No new series,
source, split or outcome-dependent fallback is substituted.

Each series' squared-error scale comes from its A prefix. Average equally
over origins and horizons within series, equally over selected series within
each collection, and equally over the four collections. No selected series
or collection receives weight because it has more available windows.

## Analysis and uncertainty

Synthetic task loss averages D squared error over its three horizons and
all origins. Repetitions, not overlapping windows, are the independent
units within a law. The primary family has 18 paired comparisons: for
each of the three laws, reconstruction-trained KAHM is compared with raw
RBF, gate-trained KAHM, reconstruction-trained affine, reconstruction-trained
locality, reconstruction-trained RBF-local, and the C-selected non-KAHM
policy. A positive comparator-minus-KAHM difference favors KAHM.

Two-sided t intervals use 11 degrees of freedom and Bonferroni alpha/18
at family alpha 0.05, under the paired-difference normal approximation.
Their simultaneous coverage claim concerns this family only, not every
historical contrast in the manuscript. Failure to exclude zero does not
establish equivalence. All other method, per-horizon and per-task results
are reported descriptively, including fixed KAHM and raw ridge.

Natural results are descriptive: collection-balanced means, complete
per-source/series/horizon tables, and leave-one-collection-out sensitivity.
Four purposively selected collections do not support a broad population
confidence claim. Window bootstrap intervals are not used to inflate the
effective source sample. No test-set oracle replaces the C-selected policy.

## Computation, state and failure accounting

The implementation eagerly prepares fixed gate queries, including D, and
charges that work even when C selects zero correction or a different
bandwidth. Policy cost includes the union of all candidates' measured
query work. These are conservative component sums for the implemented
procedure, not a separately timed minimal or lazy policy execution.

Record warm fitting, geometry search, C selection and D transformation/
prediction nodes on the same host with numerical thread counts fixed to
one. Charge shared nodes once per complete procedure. Preserve actual
optimization counts, convergence, failures, fallbacks, grid candidates,
serialized selected-model sizes and serialization time. Report timings as
host- and implementation-specific measurements, including absolute times.
Downloads, data preparation, process imports and evidence artifact writes
are outside model-pipeline time and are identified separately. A process
high-water RSS, if recorded, is not a per-method incremental memory figure.

This experiment measures a static accuracy-cost tradeoff. It does not
impose a deployment latency ceiling, compare models under equal measured
time budgets, measure streaming updates, or demonstrate bounded-memory
geometry adaptation. The preserved numerical online prototype establishes
no such empirical claim. Those questions require a further chronological
delayed-label study if the accuracy evidence warrants it.

Save all C/D predictions, targets, fitted-state files, candidate grids,
optimizer traces and computation dependencies. An independent audit
reconstructs input windows, scalar losses, selection decisions, fallback
counts, cost sums, method rosters and frozen hashes, and replays selected
model predictions. A failed audit is resolved or explicitly reported; it
does not justify suppressing unfavorable experimental outcomes.

## Interpretation rule

A distinctive predictive contribution requires improvement over matched
local alternatives and the selected practical comparison on genuinely
reserved evidence. Improving ridge alone, improving the training objective,
or lowering KAHM's own fixed-map loss is insufficient. A favorable result
is bounded by these laws, collections, capacities and horizons; an
unfavorable result constrains this intervention and does not prove that
every KAHM forecasting design fails. The manuscript will report the result
actually obtained and preserve the next unresolved methodological question.
