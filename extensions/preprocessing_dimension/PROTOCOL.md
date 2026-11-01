# Preprocessing and validation-selected local dimensions in KAHM forecasting

## Scientific question and scope

This study tests whether training-fitted preprocessing changes the forecasting
value of kernel affine hull features after correcting and selecting their
local subspace dimensions. It extends the published complete-procedure
experiment at source commit
`3cd8e56d95ec1187deed6070541f26742ce4f1cc` of
`MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine`. The inherited numerical
generators and corrected construction are retained. No historical numerical
result is overwritten.

The principal experiment evaluates complete forecasting procedures. A
separate secondary experiment holds a fitted common forecast and a shared
bias adjustment fixed while comparing departure-only corrections. The two
architectures have different fitting allocations and answer different
questions; their errors are not pooled into one primary ranking.

The observation population consists of the declared synthetic laws. Fresh
repetitions supply new held-out episodes from those laws, not a new
natural-data application cohort. The study can support a decision about the
declared numerical forecasting protocol, but it cannot establish general
deployment suitability or risk-certificate coverage. Its source, configuration,
and analysis are frozen before production data generation; this is an
executable prospective freeze, not a public preregistration.

## Data, pairing, and independent units

The design contains 24 fresh repetitions of `linear_ar`, `nonlinear_ar`,
`vanderpol`, and `duffing`, at added-noise levels 0 and 0.1: 192 task suites.
History length is 12, episode length is 72, horizons are 1, 4, and 12, and
origin stride is 4. The A/B/C/D splits contain 64/32/32/128 independent
episodes. Their retained row counts are 512/256/256/all 1664, respectively.
The 13 forecast origins within an episode are dependent. Primary inference
therefore uses 24 repetition aggregates, not 192 tasks or individual rows as
independent observations.

The inherited observation laws, initial-condition distributions, integration
settings, burn-in rules, and observation maps are unchanged. Added noise is
`eta * s_A * Z`, where `s_A` is the population standard deviation of the
unaugmented A episode values. Histories and targets are both augmented.
Underlying episode streams, unit-noise arrays, retained row indices, and
fitting seeds are paired across preprocessing and dimension candidates; noise
levels share the same underlying draws. The new master seed is
`2026110117`. The inherited integer-component seed derivation is retained,
with intended-distinct seeds checked for collisions and against retained
prior seed manifests before production generation.

No fitted state, calibration target, or terminal prediction from a previous
study is reused. Evaluation uses exactly the prespecified task roster.
Unexpected fitting, candidate, or query failures stop execution and remain
recorded; a difficult task is not replaced or omitted.

## Native dimension policy

Each local model initializes its requested dimension at

`min(cap, ambient_dimension, local_reference_count - 1)`.

It then applies the verified native positive-variance and projected-range
rules. No extra normal-direction reservation is introduced. Requested caps
and realized local basis ranks are recorded separately.

The cap grid is `[1, 2, 4, 8, 12, 20]`. All candidates use the native policy.
At the declared ambient dimension 12, caps 12 and 20 describe the same
admissible maps and must agree numerically; ties favor the smaller cap. Thus
this experiment does not test a realized rank of 20 or truncation from an
ambient dimension greater than 20. Fixed native caps 2 and 20 are sensitivity
controls, not claims that either cap is universally optimal.

## Preprocessing and geometry pairs

Only geometry-history preprocessing changes. The raw ridge incumbent and
target units remain unchanged. The three transformations are:

1. `standard`: population mean and standard deviation fitted on the declared
   fitting histories, using the inherited constant-coordinate rule.
2. `minmax`: `2 * (x - minimum) / (maximum - minimum) - 1`, with extrema
   fitted on the declared fitting histories. Exactly constant fitting
   coordinates map to zero for every query. Later values are not clipped.
3. `tanh_z_1`: `tanh((x - mean) / scale)`, with population mean and standard
   deviation fitted on the declared fitting histories and the inherited
   standard-deviation floor. There is no additional standardization of the
   transformed histories and no selected tanh multiplier.

The complete-procedure candidates use these five pairs:

| Pair | Common-feature geometry | Departure geometry |
| --- | --- | --- |
| Standard diagonal | standard | standard |
| Min-max diagonal | minmax | minmax |
| Tanh diagonal | tanh_z_1 | tanh_z_1 |
| Tanh-common hybrid | tanh_z_1 | minmax |
| Min-max-common hybrid | minmax | tanh_z_1 |

A hybrid obtains its common block from one fitted bank and its departure
block from another. Its two banks use the same declared cap. Independently
selecting a separate cap for each block is outside this finite design.
Diagonal construction must agree with the corresponding inherited one-bank
features. Feature-level normalization and radial bounds after geometry
construction retain their inherited definitions; they are distinct from
history preprocessing.

KMeans partitions are fitted to transformed histories. A change of
preprocessing may therefore change partitions, local references, PCA bases,
gates, and projected metrics. Identical seeds and sampling rules are used;
reference identities must agree across caps within one scaler. Reference
membership need not agree between different scalers. A preprocessing effect
is an effect on this complete geometry, not an isolated causal effect of
bounding input coordinates.

## Primary A/B/C/D roles and complete procedures

All history scalers, geometry banks, feature scalers, ridge paths, and other
candidate model parameters are fitted on A. B selects model hyperparameters.
C calibrates frozen procedures and selects the non-KAHM family. D is used
only for fixed-state prediction and final scoring.

The retained complete roster is ridge, raw Nyström RBF, histogram gradient
boosting, persistence, common geometry, bounded KAHM departure, bounded RBF
departure, unbounded KAHM departure, unbounded RBF departure, and the selected
non-KAHM policy. The four departure procedures retain the inherited
concatenated common-plus-departure readouts. They are fitted to the shared
raw-ridge A residuals. The common coefficients and intercepts can differ
between these complete heads; shared features alone do not imply a shared
common forecast or bias.

For every geometry candidate, B chooses a ridge penalty separately by horizon
using the inherited grid. Each complete geometry procedure then selects one
preprocessing pair and cap jointly by the equal-horizon mean of its gamma-one
B MSE divided by the population variance of retained A targets, floored at
`1e-12`. One overall pair and cap are shared across the three horizons.
Ties follow ascending cap first, then standard, min-max, tanh, tanh-common
hybrid, and min-max-common hybrid. All candidate scores, penalty choices, and selected
states are saved before C is opened.

Common-only selection uses the three diagonal scalers; hybrid copies of an
identical common map are excluded from that selection menu even when their
diagnostic predictions are retained. KAHM, RBF, and the unbounded procedures
select independently. A matched-bank comparison is a separate secondary
view and is not substituted for the independently tuned primary comparator.

The raw Nyström RBF comparator additionally receives the three A-fitted input
scalers. Each scaler retains the inherited 64-landmark, three-bandwidth-factor,
five-ridge-penalty search, with bandwidth and penalty chosen by horizon on B.
One overall input scaler is selected by normalized mean B loss across
horizons. Landmark-sampling seeds are paired. This expansion is explicit;
the baseline's former preprocessing restriction is not retained while KAHM's
preprocessing menu is enlarged. HGB retains raw inputs, its original four
configurations, and no early stopping. Persistence is fixed. Equal menus do
not imply equal computational cost or statistical capacity; actual tuning
costs are reported.

## Calibration and retained contexts

C selects a blend multiplier from `[0, 0.25, 0.5, 1]` separately by horizon
for each retained nonridge procedure. A zero multiplier returns the shared
raw-ridge incumbent exactly. A candidate's A/B-fitted state is unchanged by
calibration. C calibration results for diagnostic candidates may be retained,
but they cannot revise a B selection.

The sole primary context uses each method's B-selected preprocessing and
cap. In that context, the non-KAHM policy selects one complete method on C
from ridge, raw RBF, HGB, persistence, common geometry, bounded RBF, and
unbounded RBF. Each geometry method's preprocessing and cap, and raw RBF's
input scaler, are already fixed on B. The policy score is mean-horizon C MSE
normalized by A target variance. Ties use the inherited method order.

Secondary contexts include fixed native caps 2 and 20 with preprocessing
selected on B; each diagonal scaler with cap selected on B; and declared
within-bank matched controls. B choices are fixed before C in every context.
Calibrated and gamma-one predictions are retained. Gamma-one forwards the
already selected method without reselecting the policy or any hyperparameter.

## Secondary experiment with an honest fixed common forecast

This experiment uses the same task episodes, but divides A by whole episode
identities. The first 32 sorted A episode IDs form P; the remaining 32 form R.
No episode contributes rows to both groups. P fits the common model, and R
supplies honest residual targets for departure regressions. This is an
independent-episode separation, not a claim of a chronological natural-series
backtest.

The P raw ridge is fitted on P and its penalty selected on B. Within each of
the three common-scaler rows, native common caps are fitted on P, common
penalties are selected on B, and one common cap is selected on B. Scores use
P target variances, not A variances, so common fitting and selection do not
depend on R targets even through a loss denominator. The common multiplier
is one; there is no separate common calibration. The chosen common predictor
is frozen before residual heads are fitted.

For each common row, the shared bias is exactly the per-horizon mean B
residual, `mean_B(Y_B - common_B)`. No additional bias multiplier is selected.
That vector and the common predictor are identical for every departure
comparison within the row. This uses B repeatedly for development, which is
permitted and declared; B is not an unbiased evaluation set.

R residual targets are `Y_R - common_R`. They are honest with respect to the
common model's fitting episodes: R outcomes did not fit or select that model.
Subtracting a fixed shared bias before response centering gives the same
ridge slopes. Departure geometry banks and history scalers are fitted on P;
post-map departure standardizers and feature means are fitted on R. Bounded
departure features are centered after radial bounding. Predictions use only
centered features times ridge slopes and add them to the frozen common-plus-
bias predictor. The regression response mean is not added as a new intercept.

Each of the five scaler cells fits the declared departure cap grid. B selects
departure penalties and a departure cap separately for KAHM and RBF, using
P target variance normalization. C selects only the incremental correction
multiplier. A further matched RBF view uses KAHM's selected departure cap but
retains its own B-selected penalty and C-selected multiplier. KAHM and RBF
therefore share a common predictor and bias in all comparisons within a row;
the matched view additionally shares the departure bank cap. Different
common rows generate their own residuals and never exchange fitted heads.

Final evaluation uses the same full-A target-variance normalization as the
principal experiment. This normalization is only an evaluation scale, not an
input to the P-only common selection. All secondary results are descriptive.

## Prespecified primary claims and decision interpretation

The four-member simultaneous primary family is:

1. Selected bounded KAHM has at least 3% lower nonlinear SMSE than the selected
   non-KAHM policy.
2. Selected bounded KAHM has at least 3% lower nonlinear SMSE than the
   independently selected bounded RBF departure procedure.
3. Tanh/tanh KAHM with B-selected cap has at least 3% lower nonlinear SMSE
   than standard/standard KAHM with B-selected cap.
4. Selected bounded KAHM has at most 5% greater SMSE than ridge on the linear
   negative control.

Use equal horizon, noise, and nonlinear-law averages within a repetition and
Bonferroni one-sided paired-t lower bounds with family alpha 0.05, four
comparisons, and 23 degrees of freedom. The exact estimands, margin gaps,
exports, and secondary contrasts are specified in
`analysis_code/ANALYSIS_PROTOCOL.md`.

The policy comparison assesses the declared complete selection procedures;
the tanh comparison assesses one preprocessing hypothesis. Passing one does
not imply passing the other. Accuracy, stability, and measured resource cost
are reported together. No general adoption claim is made from synthetic
accuracy alone, and failure to establish a margin is distinguished from
evidence that the reverse margin holds.

## Freeze, target separation, and numerical verification

Manufactured-input preflight must verify standard-arm parity, diagonal
two-bank equivalence, native cap-12/cap-20 identity, constant-coordinate
behavior, B-only selection, exact zero-multiplier recovery, and fixed shared
common/bias hashes. These checks do not inspect production terminal losses.

Source, configuration, seed exclusions, and analysis definitions are frozen
before production ABC generation. All A/B fits, selections, and C choices are
globally sealed before D episodes are generated. Prediction receives D
histories only. State hashes must remain unchanged. All prediction arrays are
sealed before terminal targets are released for scoring. Later D histories
may contain earlier values from their own episodes, but no query operation
fits or tunes across those histories.

Record native ranks, local reference identities, guards, factor errors,
feature/readout bounds, candidate choices, calibration multipliers, and
execution failures. Existing documented local numerical guards remain part
of the procedure; unexpected exceptions are not silently converted into a
successful baseline. The freeze and all retained records identify the
executed source. If a scientific change is required after production begins,
retain the failed attempt and issue a new freeze before restarting; do not
silently amend the executed protocol.

## Resources and limits

Use one numerical thread per worker and at most three workers on distinct
available CPUs. Report full study execution time and complete primary and
secondary fitting/selection stage times, which include their candidate
searches. Shared computations may be cached. Primitive node timings are
partial diagnostics and are not summed into a purported complete
per-procedure runtime or used to rank KAHM against RBF. The combined study
bundle does not isolate the deployment cost of each selected method. A
complete evidence checkpoint is not a deployment-state size or a peak-memory
measurement. Shared-host timings are descriptive, not isolated hardware speed
rankings; a method-level cost-effectiveness conclusion remains outside these
measurements.

The finite scaler and cap grids, one history length, four synthetic laws,
three horizons, modest fitting samples, and approximate paired-t comparisons
limit the scope. Independently selected banks compare entire procedures;
within-bank secondary controls support narrower mechanism comparisons. Local
rank is not the dimension of the stacked feature representation. This study
does not revise the certificate theory or infer coverage from empirical test
errors.
