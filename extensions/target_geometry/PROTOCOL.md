# KAHM target geometry: prospective execution protocol

## Question and status

Does a residual-supervised diagonal input metric improve KAHM forecasting on
top of a competitive ridge baseline, and does it do so beyond equally trained
affine, locality and RBF-local controls? A separate comparison asks whether
honest donor residuals improve on in-sample residual targets.

This protocol is written before the new study's full execution. The source,
configuration, exact natural roster and numerical dependencies will be frozen
and published before confirmatory seeds are executed. It is an execution
freeze, not external preregistration. Development smoke tests use a separate
seed and fabricated data; their outputs are not confirmation. The original
controlled, natural, scaling and utility evidence remains unchanged. The new
method is one bounded follow-up motivated by those already inspected results.

## Complete forecasting procedures

Every target is an observed future value in its original units. All models use
the same causal history and direct horizons within a task. No test labels
inform normalization, geometry, donors, regularization or mixing. Selected
training-only fits are retained for test; there is no pretest refit.

The main base is raw ridge, selected on outer validation over
`[0.0001,0.001,0.01,0.1,1,10,100]`. Its input/target scales and coefficients
use training data only. Raw Gaussian Nyström-plus-linear regression with 64
requested landmarks and the same penalty grid is a secondary architectural
reference. Seasonal naive is included for natural data and persistence for
synthetic data. A jointly fitted raw-history-plus-KAHM **ridge** readout is a
secondary architecture control; it is not a replication of the earlier
Nyström concatenation architecture.

There are four geometric gates: KAHM reconstruction, affine residual,
locality-aware affine, and Gaussian center RBF. Each uses matched partitions,
references, truncated PCA directions and gated local coordinates. RBF-local
is a local affine expert control, not the raw Nyström reference. Banks request
2/4/8 groups, rank 2, at most 24 references per local model and 512 fitting
histories. The inherited stable Gaussian KAHM query and its signed-normalizer
convention remain unchanged. The affinity exponent is 16 and floor 1e-6.

For every gate, compare these three procedures:

1. Identity metric, in-sample residuals (`fixed_is`).
2. Identity metric, honest out-of-fold/donor residuals (`fixed_oof`).
3. Shared learned metric, honest donor residuals (`learned_oof`).

The bank is fitted using all training histories, independent of test/validation
labels. Correction heads use the identical eligible row set in all three
procedures. Each correction is training-standardized ridge, with the same
seven-penalty grid and mixing weights `[0,0.25,0.5,1]`. Zero is one exact
candidate; the 21 positive candidates are evaluated in fixed penalty and
mixing order. A strict validation improvement replaces the incumbent, so ties
retain zero or the earlier candidate. The correction does not change the fitted
raw baseline. A zero candidate is not a population nondegradation guarantee.

## Honest donor residuals

The final ridge base is selected on outer validation. Its chosen penalty never
feeds backward into the donor fits.

For synthetic data, whole independent training episodes are assigned to four
folds by episode index modulo four. Each donor uses the other three folds.
Its penalty is selected within donor data, using the first 80% of ordered donor
episode IDs for fitting and the remainder for inner validation. The selected
donor is then refitted on all its donor episodes, and predicts only excluded
episodes. All training rows have OOF targets. Episode order here is an index of
independent simulations, not a temporal ordering of one trajectory.

For natural data, the first 40% of ordered capped training origins initialize
the donor history. The remaining origins are split into three contiguous
blocks. For each query block, donor rows must satisfy
`donor_origin + horizon <= first_query_origin`; all their targets are therefore
available at the first query forecast. A donor has its own 80/20 chronological
inner split, applying the same target-availability condition at the inner
boundary. After inner selection it is fitted on all allowed donor rows. If the
inner fitting subset has fewer than eight rows, its fixed fallback penalty is
1. If the outer donor has fewer than eight rows, that block has no residual
targets and is excluded from **both** IS and OOF correction fitting. Fewer than
eight total correction rows triggers ridge fallbacks for the dependent arms.

Donor residuals are errors of earlier/separately fitted predictors, not unbiased
observations of the final base's future errors. We record donor/final prediction
disagreement on correction rows. It is descriptive; the common-law transport
identity in THEORY.md does not certify its future significance or eliminate
nonstationarity. In-sample residual fitting is a legitimate competing procedure.

## Learned diagonal metric

Training input standardization is shared with the downstream bank. The metric
is `diag(exp(theta))`, with `theta=eta-mean(eta)`. The free coordinates are
bounded by `+-log(16)/2`, so the ratio of largest to smallest metric weight is
at most 16 and the geometric mean is one.

Select at most 256 correction rows without replacement using the task seed.
On their OOF residuals, minimize a Gaussian kernel-regression proxy plus
`0.05*mean(theta**2)`. Residuals are centered by output and divided by one pooled
scale, preserving original-unit relative horizon weights. The squared bandwidth
is the median positive admissible squared input distance at the identity
metric, held fixed during optimization. Whole same-episode donors are excluded
on synthetic data. Natural metric pairs exclude equal origins and origins
within `context+horizon` steps. Other earlier/later donors **within the training
period** are allowed: this is offline supervised metric fitting, distinct from
the causal generation of baseline residuals.

L-BFGS-B uses the analytic gradient, at most 40 iterations, ftol 1e-10, gtol 1e-6
and maxls 20. A finite non-increasing objective iterate may be retained at the
iteration limit, with convergence status recorded. Nonfinite or worse results,
constant residuals or insufficient admissible pairs return the identity metric
and a logged reason. No validation/test label enters this optimization.

The identical learned metric is supplied to all four gates. Its weights apply
**after** input standardization. This is a residual-supervised kernel-regression
metric transferred to KAHM and its controls, not end-to-end KAHM optimization.
MLKR is prior art; decreasing this proxy does not establish lower forecasting
error or KAHM-specific benefit.

## Reliability and failures

For this new procedure only, requested group counts are capped by the number
of distinct partition histories. If numerical k-means still produces an empty
group, every requested count is halved down to one and the reduction is logged.
Inherited rank-zero center reconstructions remain available. Other geometry
or correction exceptions produce the fitted ridge forecast, retaining the
exception, unsuccessful work and fallback status. A metric exception produces
the declared identity metric. The secondary raw RBF has a ridge fallback.

Failed donor fits retain their records and unavailable predictions; dependent
OOF/learned arms fall back to ridge, while available baselines and in-sample
arms remain reported. An unavailable primary ridge fit or an unavailable
declared evaluation split is a task failure. All tasks remain in accounting;
full-roster losses and intervals are unavailable if required outputs are
missing. No series, seed or task is replaced after its outcome is observed.
These changes do not retrospectively repair the previous utility failures.

## Synthetic roster and estimands

Retain the inherited nonlinear AR, Van der Pol and Duffing numerical laws.
Use twelve fresh repetitions with seeds 11200 through 11211. For each law and
repetition independently generate 128 training, 64 validation and 128 test
episodes via `SeedSequence([seed,law_index,split_index,9027])`. Episodes have
length 72, history 12, direct offsets 1/4/12 and stride 4. There are 13 origins
per episode. Simulation and observation conventions are inherited unchanged.

There are 36 complete synthetic tasks and 16 methods per task. MSE averages
horizons and origins within each episode, then episodes. Each primary contrast
uses the twelve complete paired repetition-level differences. The six contrasts
listed in config.json, crossed with the three laws, form one family of 18.
Report two-sided 95% Bonferroni paired Student intervals, explicitly relying
on a normal approximation to repetition differences. Positive control-minus-
candidate loss favors the candidate. Origins and horizons do not increase n.
No joint confidence claim is made with the historical study families.

The six contrasts isolate improvement over ridge, metric change, residual-target
change, and KAHM versus each of the three matched learned geometric controls.
All 18 are reported regardless of sign. Other means, selected weights, horizons
and descriptive donor diagnostics remain secondary. Fresh simulation seeds
provide new stochastic evidence on known laws, not new-domain confirmation.

## New-collection roster

Four source collections absent from the historical 12-family roster are fixed:
Australian state electricity demand (half-hourly; context48, horizon12), Bitcoin
indicators (daily;28,7), Melbourne pedestrian counts (hourly;48,6) and KDD2018
air quality (hourly;48,6). Official Monash/Zenodo records, attribution, licenses,
archive hashes and source exclusions are pinned in sources/manifest.json.

Select three series per collection using the fixed salt and training-prefix
eligibility in data.py. Training requires at least 96 finite windows and a
positive training scale. No later outcome or availability determines selection.
The resulting exact 12-series roster is prepared/roster.json. Bitcoin selections
are activity indicators, not price. These are new collections; broad application
domains can overlap earlier electricity, traffic and weather work.

Each series has chronological60/20/20 splits, at most512 training and64
validation/test windows, evenly sampled as in the inherited window generator.
Targets lie entirely inside their segment; later test origins may use already
observed test values. Missing windows are omitted without interpolation. The
training seasonal-difference MSE (first-difference fallback) supplies the fixed
scale. Natural MSE averages origins/horizons within series, series within source,
and the four sources equally. Report all series, sources, horizon diagnostics,
fallbacks, source-balanced means and leave-one-source-out descriptive results.
No distribution-free confidence interval or representative population claim
is assigned to four purpose-selected collections. The protocol is not the
Monash leaderboard evaluation and is not compared directly to its published
scores. There is one fitting seed (491), not an additional population replicate.

## Resources, accounting and reporting

One task runs at a time with one BLAS thread. Warm wall and process CPU times
are recorded by disjoint computation nodes. Each standalone correction pays
for the needed baseline search/predictions, its residual-generation procedure,
metric learning when used, geometry construction/transforms, all correction
candidates and selected-model serialization. Shared work is charged once per
procedure; it is not charged only to the winning geometry. Times are component
sums, not independently repeated deployments. Final test geometry transforms
are conservatively computed/charged even for gamma zero; they use inputs only.
Imports, downloads, preparation, writing result artifacts and reporting are
outside model timing. Actual retained-model byte size is measured separately.
This implementation-specific cost is not a lower bound on KAHM complexity.

Save all final predictions, targets, origins, donor predictions, final-base
training predictions, correction indices, validation candidates, selected
settings, bank and metric diagnostics, warnings and failure records. The
independent analyzer reconstructs losses from arrays, checks dependency sums,
zero equality, residual alignment identity, donor exclusion and the complete
roster before generating tables and figures. Code, data hashes, environment,
protocol and publication checkpoint are retained.

Scientific conclusions follow observed comparisons. A generic metric benefit
is not credited uniquely to KAHM. A practical positive finding requires gains
over ridge and matched local controls, interpretable coverage, and a resource
requirement appropriate to an application. The present study does not choose
an application-independent accuracy/cost exchange rate, promise improvement,
or certify minimum necessary model complexity.
