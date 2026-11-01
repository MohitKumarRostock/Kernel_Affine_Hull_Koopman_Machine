# Analysis of episode-wise cross-fitted residual corrections

## Purpose and prospective inputs

The scientific analysis was fixed before original ABC generation. This
amended execution retains the same analyzer bytes, estimands, margins,
comparisons, aggregation, and output schema. A numerical solver amendment
was introduced after ABC preparation and an aborted fit, before any D
generation; see `NUMERICAL_AMENDMENT.md`. All 192 fits are restarted under
one amended source version on the same cohort. The analysis scores
already fitted, selected, calibrated, and sealed forecasts. It does not fit a
model, choose preprocessing or dimension, select a multiplier, or choose a
preferred forecasting method from terminal D outcomes.

The scientific design is in `PROTOCOL.md` and the machine-readable design is
in `config.json`. There are 24 new independent repetitions, four observation
laws, two noise levels, history width 12, and horizons 1, 4, and 12: 192 task
suites in total. The A/B/C/D episode allocation is 64/32/32/128, with
512/256/256/1664 retained rows. The many origins within an episode are not
independent observations. No earlier study's forecasts or outcomes are
pooled into this experiment.

Source/configuration/analysis freeze, fitted-state seal, prediction seal,
and target-release identities must pass before terminal target arrays are
opened. The freeze must cover the new source and all imported numerical
sources from the inherited preprocessing and dimension extensions. A
missing, duplicated, failed, nonfinite, or shape-inconsistent task is not
omitted, imputed, or replaced. Unexpected failures stop the analysis and
remain in the execution record.

## Procedures and retained forecasts

The new residual procedures share one full-A common predictor, selected on
B at unit multiplier, and one B residual-mean bias. The common predictor and
bias are frozen before C. Four-fold cross-fitting assigns whole A episodes
to folds. Each fold common predictor is fitted using only that fold's A
complement and independently selected on B from the declared three scalers
and native caps 1, 2, 4, 8, and 12. Each retained A row receives exactly one
OOF common forecast, from the predictor that excludes its whole episode.
The deployment common predictor is separately fitted and selected using
full A and B. Fold tuning cannot inherit settings selected through a
full-A fit if this would allow the held-out outcomes to influence their
OOF common forecast.

Cross-fit residual targets are A outcomes minus OOF common forecasts. The
in-sample ablation uses A outcomes minus the full-A common forecast at the
same A rows. These two procedures share the common deployment predictor,
bias, fitting sample, and feature-candidate construction. Their residual
target source differs; their own B-selected departure choices may differ
as a consequence. For both target sources, KAHM and RBF independently select
their departure scaler, cap, and ridge penalties on B. The matched RBF
uses KAHM's selected scaler and cap with its own B-selected penalty and
C-selected multiplier. C chooses only a per-horizon incremental multiplier
from 0, 0.25, 0.5, and 1 around the frozen common-plus-bias baseline.

A half-A common predictor, fitted using the first 32 declared A episode
identities and selected on B, and its B bias adjustment are retained only
as diagnostic baselines. No half-A residual-correction experiment is
included. The inherited complete-procedure suite is refitted in this
campaign, retaining its selected context only; it is distinct from the
new fixed-common correction architecture.

The terminal forecast roster contains exactly 42 arrays:

- Twenty `legacy__{calibrated,gamma1}__{method}` arrays for ridge, raw RBF,
  histogram gradient boosting, persistence, common, bounded KAHM departure,
  bounded RBF departure, unbounded KAHM departure, unbounded RBF departure,
  and the inherited non-KAHM policy.
- Twenty `crossfit__{calibrated,eta1}__{method}` arrays for common,
  shared bias, half common, half shared bias, cross-fit KAHM, cross-fit RBF,
  cross-fit matched RBF, in-sample KAHM, in-sample RBF, and in-sample
  matched RBF.
- Two `policy__{calibrated,eta1}__expanded_nonkahm` arrays.

The expanded policy selects one whole procedure on C by mean horizon MSE
divided by full-A horizon target variance. Its declared first-minimum order
is inherited ridge, raw RBF, histogram gradient boosting, persistence,
common, bounded RBF departure, unbounded RBF departure, then the new
shared-bias baseline, cross-fit RBF, and in-sample RBF. It receives only
B-selected procedures and uses their calibrated C forecasts. It excludes
KAHM and KAHM-parameter-matched RBF. Its unit-multiplier view forwards the
same selected identity using that procedure's gamma-one or eta-one
forecast; it does not reselect the policy. All other unit-multiplier views
likewise retain their saved B choices.

Discarded candidate D forecasts are not exported. Complete scalar B/C
scores, deterministic tie outcomes, selected settings, development
predictions, and cross-fit records retain selection auditability.

## Loss and aggregation

For task t, horizon h, and retained forecast m, define

`MSE(t,h,m) = mean_D((Y_D_h - prediction_D_h_m)^2)`

and

`SMSE(t,h,m) = MSE(t,h,m) / max(Var_A(Y_A_h, ddof=0), 1e-12)`.

The evaluation denominator uses all retained A targets for the task and is
identical across methods, target sources, and common fitting allocations.
It is never estimated from D. Fold-specific tuning uses its declared
training-complement normalization; final evaluation always uses full A.

Average the three horizons equally within a task. Within each repetition,
the nonlinear panel averages nonlinear autoregression, Van der Pol, and
Duffing equally, with the two noise conditions equally weighted. Its
repetition loss is therefore the mean of 18 law/noise/horizon cells. The
separate linear panel averages its two noise conditions and three horizons,
giving six cells. An all-law aggregate may be exported as descriptive only.

The reported aggregate loss is the mean of the 24 repetition losses. It is
not a mean of condition-wise percentage gains, a pooling of raw errors
across differently scaled laws, or an analysis treating folds, origins,
rows, episodes, or 192 tasks as independent replications. Raw MSE remains
available by task and horizon and does not define a pooled physical-risk
claim.

The descriptive point gain of candidate a against comparator b is

`100 * (mean_repetition_loss_b - mean_repetition_loss_a) / mean_repetition_loss_b`.

Positive values favor the candidate. If the denominator is zero, retain
the absolute difference and mark the percentage undefined; do not insert
an arbitrary denominator. A percentage display of an absolute-difference
interval rescaled by the observed comparator mean is not a risk-ratio
confidence interval.

## Four simultaneous primary comparisons

All four claims concern the calibrated cross-fit KAHM correction, with
its own B-selected scaler/cap/penalties and C-selected eta.

| ID | Panel | Comparator | Required margin |
| --- | --- | --- | --- |
| shared_bias | nonlinear | Identical full-A common-plus-shared-bias baseline | 3% superiority |
| rbf | nonlinear | Independently B-selected cross-fit RBF correction | 3% superiority |
| policy | nonlinear | Expanded C-selected non-KAHM policy | 3% superiority |
| linear | linear | Shared raw ridge | 5% noninferiority |

For each nonlinear claim, the repetition-level margin gap is

`g_r = 0.97 * L_comparator,r - L_crossfit_KAHM,r`.

For linear noninferiority, use

`g_r = 1.05 * L_ridge,r - L_crossfit_KAHM,r`.

With n = 24, family alpha = 0.05, and J = 4, the simultaneous one-sided
lower margin gap is

`mean(g) - t.ppf(1 - 0.05 / 4, 23) * sd(g, ddof=1) / sqrt(24)`.

A claim is established only when its lower endpoint is strictly positive.
Export candidate/comparator means, point gain, every repetition margin
gap, sample SD, standard error, critical t value, lower endpoint, one-sided
p value, Bonferroni-adjusted p value, and established flag. These are
approximate paired-t inferences over the declared repetition population,
not distribution-free guarantees or risk certificates.

The primary RBF comparison is against independently selected RBF, not
KAHM-parameter-matched RBF. The policy comparison concerns the strongest
procedure selected by the declared finite C policy; it is not a D oracle
or a claim about every possible non-KAHM model. A positive point gain does
not establish a 3% practical advantage unless the simultaneous bound passes.
Failure to establish superiority does not establish equivalence.

Report all four claims separately. A successful residual-mechanism claim,
linear control, numerical screen, or secondary comparison cannot substitute
for a failed policy comparison.

## Prespecified descriptive comparisons

Use two-sided, unadjusted 95% paired-t intervals for mean repetition SMSE
differences, with 23 degrees of freedom. Label these intervals descriptive.
Retain the corresponding 24 paired differences and improvement counts.
They do not enlarge or inherit protection from the primary family.

The descriptive comparisons include:

- Cross-fit KAHM against its shared-bias baseline, independently selected
  cross-fit RBF, KAHM-scaler-and-cap-matched cross-fit RBF, and the expanded
  policy, alongside the primary margin results.
- Cross-fit versus in-sample residual targets for KAHM, independently
  selected RBF, and matched RBF. These compare complete declared procedures
  after their own B/C selection; they are not a claim that the final
  selected departure settings necessarily coincide.
- Within the in-sample ablation, KAHM against the shared-bias baseline,
  independently selected RBF, and matched RBF.
- Full-A common against half-A common, and their corresponding shared-bias
  baselines. These describe the declared fitting-allocation comparison and
  must not be relabeled as an isolated cross-fitting effect.
- Shared-bias versus unadjusted common, separately for full A and half A.
- New cross-fit KAHM against the refitted inherited selected complete KAHM
  procedure and inherited non-KAHM policy. These architectures differ and
  are not treated as an isolated causal effect of residual target source.
- The expanded non-KAHM policy against the inherited non-KAHM policy,
  comparing their declared C-selected menus on the same D episodes.
- Calibrated versus unit-multiplier forecasts for each retained applicable
  procedure, always preserving its selections and policy identity.
- Law/noise/horizon heterogeneity and B/C selection frequencies.

Report these for the nonlinear and linear panels and, where exported,
the all-law descriptive panel. No favorable D condition or diagnostic arm
may replace the prespecified primary KAHM arm.

## Cross-fit and forecast identity checks

Before interpreting losses, verify and export:

1. The exact 192-task factorial and 42-array roster; finite, correctly
   shaped predictions; identical row/horizon alignment; A-only loss scales.
2. Exactly four held-out episode folds; disjoint complement/held-out sets;
   all 64 A episodes covered once; every retained A row assigned once;
   and all rows of one episode sharing a fold.
3. OOF residuals equal A targets minus their OOF common forecast; in-sample
   residuals equal A targets minus the full-A common forecast. Saved
   training-array hashes and fold identities must agree with prepared A.
4. Fold common selections replay the declared independent B argmin/tie
   rule over the three scalers and five caps. Full-A and half-A selections
   replay their respective declared tuning records.
5. The full-A common and shared bias are identical across residual target
   sources and correction methods. Shared bias equals the saved B mean
   residual. Half-A bias is checked against its own common predictor.
6. Independent KAHM/RBF B choices replay their declared candidate scores;
   each matched RBF uses KAHM's selected scaler and cap with its own
   penalty and eta. When independent and matched RBF select identical
   settings, their forecasts agree.
7. Centered correction features and slope-only training corrections have
   the declared near-zero A means. No B/C/D query mean is subtracted.
8. Every C eta/gamma and expanded-policy choice replays its saved scores;
   zero eta recovers the exact shared-bias baseline; unit views preserve
   selected settings; and common/bias diagnostic rows are mode invariant.
9. Fitted-state identity is unchanged by calibration evaluation or D
   querying, apart from explicitly saved calibration choices. No fitted
   scaler, bank, penalty, or feature statistic is recomputed on D.
10. Native dimensions, reference membership, numerical guards, and source,
    fitted-state, forecast, and target-release hashes remain auditable.

OOF-versus-in-sample residual magnitudes and common-prediction differences
on A are descriptive training diagnostics. They are not additional test
sets, inferential units, or selection criteria for the final reported arm.

## Stability, resources, and exported evidence

For each of the 192 tasks, compare the selected calibrated cross-fit KAHM
maximum absolute D error with the shared raw ridge maximum error. Retain
the declared limit of five and report every screen. Passing this screen
does not establish a predictive superiority margin. Documented zero-rank
fallbacks and numerical guards are recorded as part of the procedure;
unexpected failures are not silently converted to successful forecasts.

Export task/horizon losses, task averages, repetition aggregates, panel and
condition summaries, primary and descriptive contrasts with repetition
detail, B/C scores and selections, cross-fit audit records, numerical
screens, resources, input integrity, and a hash-indexed analysis summary.
The frozen forecast roster is compact; absent discarded-candidate D
arrays are not missing planned outcomes.

Report full study elapsed time and declared fit/cross-fit/calibration/
prediction stage resources, including the finite search. Primitive timings
and cached work are partial diagnostics and do not imply isolated
method-level deployment costs. Keep archived audit bytes and selected
deployment-state size distinct. Use one numerical thread per worker and
the frozen shared-host worker budget.

## Interpretation and limits

Lead with the four primary results, then matched-RBF and residual-target
ablations, common-strength diagnostics, calibration, heterogeneity, and
resources. State separately whether cross-fit KAHM improves its identical
common-plus-bias baseline, whether it clears the independently selected
RBF margin, and whether it clears the expanded non-KAHM policy margin.

Cross-fitting protects the declared OOF target construction; it does not
guarantee that a correction trained against smaller-fold common models
will improve a full-A deployment common model. B tuning and C calibration
are part of the evaluated finite procedure, and D tests its resulting
forecast.

Claims are limited to fresh repetitions of the four specified numerical
laws, history width 12, three horizons, finite construction and selection
grids, and approximate repetition-level inference. The experiment does
not establish natural-data deployment suitability, an optimum over
untested hyperparameters, dimensions beyond the tested ambient space,
or theoretical certificate coverage.
