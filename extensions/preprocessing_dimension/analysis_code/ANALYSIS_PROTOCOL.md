# Analysis of preprocessing and local-dimension selection

## Purpose and immutable inputs

This analysis is specified before production data generation. It scores
already fitted, selected, calibrated, and sealed forecasts. It does not fit
models, modify preprocessing, select a cap, select a calibration multiplier,
or choose a preferred method from D outcomes.

The machine-readable design is `config.json`; the scientific protocol is
`PROTOCOL.md`. The data roster has 24 independent repetitions, four laws, two
noise levels, and three horizons. The principal source lineage is
`3cd8e56d95ec1187deed6070541f26742ce4f1cc`; the new master seed is
`2026110117`. The executed source/configuration freeze, fitted-state seal,
prediction seal, and terminal-target release must be verified before scoring.

All 192 prespecified tasks are required. A missing, failed, nonfinite, or
shape-inconsistent forecast is not omitted, imputed, or replaced. Unexpected
failures stop the analysis and appear in the execution record. An unplanned
analysis change is identified separately from the frozen primary analysis.

## Loss and aggregation

For task `t`, horizon `h`, and procedure `m`, define

`MSE(t,h,m) = mean_D((Y_D_h - prediction_D_h_m)^2)`

and

`SMSE(t,h,m) = MSE(t,h,m) / max(Var_A(Y_A_h, ddof=0), 1e-12)`.

The denominator uses all retained A targets for the task. It is the same
across competing forecasts, contexts, and the secondary fixed-common study.
It is not recomputed from D. The secondary common/departure B selections use
P variance internally, as declared in the protocol; final scoring still uses
the shared A normalization.

Average the three horizons equally within each task. Within each repetition,
the nonlinear aggregate averages `nonlinear_ar`, `vanderpol`, and `duffing`
equally and averages their two noise levels equally. The linear aggregate
averages the two `linear_ar` noise levels equally. Thus the nonlinear
repetition loss is the mean of 18 law/noise/horizon cells, and the linear
repetition loss is the mean of six cells.

The reported aggregate loss is the mean of the 24 repetition losses. It is
not an unweighted mean of per-cell percentage improvements, a mean of raw
errors across differently scaled laws, or an inference treating rows,
origins, horizons, noise levels, or the 192 tasks as independent replications.

For comparator `b` and candidate `a`, the descriptive point gain is

`100 * (mean_repetition_loss_b - mean_repetition_loss_a) / mean_repetition_loss_b`.

A positive gain favors the candidate. If a denominator is zero, retain the
absolute difference and mark the percentage undefined; do not insert an
arbitrary denominator. Raw MSE by law and horizon can be reported alongside
SMSE but is not pooled into a common physical-risk claim.

## Four simultaneous primary comparisons

All primary forecasts are calibrated. Candidate definitions and contexts are
fixed by B and C before D is generated.

| ID | Panel | Candidate | Comparator | Margin |
| --- | --- | --- | --- | --- |
| policy | nonlinear | joint B-selected bounded KAHM | selected non-KAHM policy | 3% superiority |
| rbf | nonlinear | joint B-selected bounded KAHM | independently joint B-selected bounded RBF | 3% superiority |
| tanh | nonlinear | tanh/tanh bounded KAHM, B-selected cap | standard/standard bounded KAHM, B-selected cap | 3% superiority |
| linear | linear | joint B-selected bounded KAHM | shared raw ridge | 5% noninferiority |

For each nonlinear comparison, form the repetition-level margin gap

`g_r = 0.97 * L_comparator,r - L_candidate,r`.

For the linear negative control, use

`g_r = 1.05 * L_ridge,r - L_selected_KAHM,r`.

Let `n=24`, `alpha_family=0.05`, and `J=4`. The simultaneous lower margin gap
is

`mean(g) - t.ppf(1 - alpha_family / J, n - 1) * sd(g, ddof=1) / sqrt(n)`.

A claim is established only when this lower endpoint is strictly positive.
Export the comparator and candidate means, point gain, mean margin gap,
sample standard deviation, standard error, critical value, lower endpoint,
margin, independent-unit count, and established flag. These are approximate
paired-t comparisons over the declared repetition population, not
distribution-free certificates or class-optimal risk bounds.

Do not describe a positive point gain as an established 3% advantage when
the lower margin gap is nonpositive. Do not infer equivalence from failure
to reject. The four results remain individually interpretable; the policy
claim, tanh claim, KAHM-versus-RBF claim, and negative-control claim do not
imply one another. Report how many of the four declared claims are
established without substituting a successful secondary contrast for a failed
primary comparison.

## Prespecified secondary contrasts

The following are descriptive and do not enlarge the primary claim family:

- Native cap-2 and native cap-20 contexts, with preprocessing selected on B,
  versus joint preprocessing-and-cap selection.
- Standard, min-max, and tanh diagonal contexts, each with cap selected on B.
- Every prespecified hybrid and its corresponding diagonal procedures.
- KAHM and RBF on KAHM's selected bank configuration, distinct from the
  independently selected RBF primary comparator.
- Bounded versus unbounded departure heads under their declared selections.
- Gamma-one versus C-calibrated forecasts, retaining the same B choices and
  the same C-selected policy family.
- Individual law, noise-level, and horizon losses; maximum task ratios to
  ridge; improvement counts; and selection/calibration frequencies.
- All comparisons in the honest fixed-common secondary architecture.

For ordinary paired secondary differences, report the mean repetition
difference and descriptive two-sided unadjusted 95% paired-t interval with
23 degrees of freedom. Label such intervals explicitly as descriptive and
unadjusted. They do not support claims protected by the primary family
allocation.

For the min-max/tanh two-by-two preprocessing contrast, define, within each
repetition and a declared dimension context,

`I_r = (L_tanh,tanh,r - L_tanh,minmax,r)
       - (L_minmax,tanh,r - L_minmax,minmax,r)`.

This describes whether changing departure preprocessing has a different
effect under the two common preprocessing choices. Compute it separately
for fixed cap 2, fixed cap 20, and within-cell B-selected caps when those
views are retained. A contrast with independently selected caps is an
interaction between complete selection procedures, not a pure fixed-rank
mechanism effect. No favorable cell may be chosen from D and renamed the
primary hybrid.

A dimension-selection interaction can be described by the difference between
the tanh-versus-standard loss contrast under selected caps and the same
contrast under each fixed cap. Keep fixed-cap and selected-cap definitions
visible. This is secondary; it does not replace the primary tanh margin
comparison.

## Honest fixed-common secondary analysis

For every common-scaler row, verify that the common forecast and shared bias
are identical across departure methods and scales within that row. The
shared baseline is the P-fitted/B-selected common forecast plus the frozen
per-horizon mean B residual. R outcomes fit only the centered departure
heads. Any discrepancy in stored baseline or bias hashes invalidates the
matched mechanism comparison.

Within each of the five scaler cells, report common-only, shared-bias-only,
KAHM, independently selected RBF, and KAHM-cap-matched RBF losses where
available. Report incremental gains over shared bias separately from gains
over unadjusted common forecasts. For each correction, report its B-selected
cap and penalties, C-selected eta, and eta-one loss. The matched RBF uses
KAHM's departure cap but its own B-selected penalty and C-selected eta.

Check the training mean of the centered bounded R features and the mean of
the resulting slope-only R correction. Near-zero R training mean does not
imply zero mean on B, C, or D, and such a query mean must not be subtracted
after seeing evaluation outcomes.

The secondary P/R fitting allocation differs from the principal A fitting
allocation. Report its results in their own tables and interpret changes
relative to its own baselines. Do not treat a cross-architecture comparison
as evidence that one preprocessing choice alone caused the difference.

## Integrity and numerical summaries

Export and verify the following evidence before reporting conclusions:

1. Full task/horizon losses and 24 repetition aggregates for every retained
   method and context.
2. All B preprocessing/cap scores, selected penalties, deterministic tie
   outcomes, and selected identities; raw RBF scaler choices and the complete
   non-KAHM C policy menu.
3. Every C multiplier and selected policy identity, with no D-selected
   alternatives.
4. Requested and realized local ranks, reference identities, and exact
   cap-12/cap-20 checks at ambient dimension 12.
5. Geometry guard counts, unexpected numerical errors, bounded-feature and
   coefficient/correction checks, and exact zero-multiplier incumbent recovery.
6. Fit-state hashes before and after C evaluation and D prediction, shared
   common/bias identities in the secondary study, and prediction/target seals.
7. Complete task accounting, including any failed attempts or protocol
   amendments; successful outcomes cannot erase earlier failed executions.

The inherited maximum-error-ratio screen uses a threshold of five relative
to ridge. Report every screened task and the maximum ratio. Passing this
screen does not establish a primary superiority margin. Local rank-zero or
documented numerical guard behavior is reported as part of the procedure;
unexpected failures are not silently passed as normal successful forecasts.

## Resource analysis

Retain full study wall time and complete primary/secondary fitting-selection
stage times, including every fitted candidate. Retain available primitive
construction, fitting, selection, calibration, and query timings as partial
diagnostics. These primitive nodes do not cover every feature transformation
and assembly, so their sums are not complete per-procedure runtimes. The
joint study execution does not provide isolated method-level costs or a
KAHM-versus-RBF computational ranking. Cached numerical results may be reused
without describing their construction as free. Report this accounting scope
beside the numbers.

Separate evidence-checkpoint bytes, fitted deployment-state estimates when
available, and measured process memory. They are different quantities.
One numerical thread per worker and at most three workers are used on a
shared host; measured times are descriptive and do not establish isolated
hardware rankings. Do not assert a universal cost-effectiveness threshold
that was not specified in the protocol.

## Reporting and limitations

Lead the report with the four primary results, then preprocessing and
dimension selection, hybrid and fixed-common diagnostics, heterogeneity,
numerical behavior, and resource costs. State whether tanh helps under the
declared selection policy and whether the complete KAHM procedure clears the
policy margin; these are separate findings.

The finite candidate grids, fixed history length 12, synthetic observation
population, 24 repetition aggregates, and approximate inference delimit all
claims. The study does not establish benefit on natural forecasting cohorts,
rank-20 behavior above ambient dimension 20, optimality over untested
hyperparameters, general adoption, or certificate coverage. Cite exact
generated tables and source/protocol hashes rather than interpreting an
earlier study's values as results of this campaign.
