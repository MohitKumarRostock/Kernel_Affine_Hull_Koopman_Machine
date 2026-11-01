# Honest residual corrections with a full-training common predictor

## Pre-D numerical amendment

This is the amended execution of the original cross-fit study. The initial
source was publicly published at
`3b9db85c1aa8ba890dbb74cc913a9337d7360d9d` before ABC generation. Its first
fitting stage stopped on a reproducible finite-matrix SVD convergence
failure. No D episodes had been generated. This version adds a fixed,
logged alternative SVD driver only when the original ridge path raises
`LinAlgError`. It reuses exactly the original ABC bytes and all reserved D
seeds, restarts every fit, and retains every original scientific choice.
The initial attempt is preserved in `../crossfit_residual`. See
[NUMERICAL_AMENDMENT.md](NUMERICAL_AMENDMENT.md) for diagnosis, timing,
fixed validity checks, and the audit trail.

## Question, lineage, and scope

This prospective extension asks whether KAHM departure features improve a
strong common forecast when the correction targets are obtained by holding
out whole training episodes. The deployment common predictor is fitted on
all designated A training episodes. B selects model settings, C calibrates
corrections and selects the non-KAHM policy, and independent D episodes
evaluate the complete frozen procedures.

The numerical construction retains the corrected native dimension policy
from commit `3cd8e56d95ec1187deed6070541f26742ce4f1cc` and the preprocessing
implementation from commit `5cfd7a1745a52459b7b5045fe84803b9bbb2b56a` of
`MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine`. The preceding report
was published at `fe8e45340c58306686a79f6e7e4be5ce6f96271a`. Those sources
and historical results remain unchanged. This directory has its own
configuration, amended source freeze, checkpoints, analysis, and report.
The streams are the same fresh cohort reserved by the initial attempt; this
is a restart of that study, not a second independent cohort.

The population remains the four declared synthetic laws. New episodes
provide an independent evaluation of the newly specified procedures under
those laws. They do not provide evidence of natural-data adoption, establish
a universal KAHM advantage, or validate a risk certificate. The executable
initial source/configuration/analysis freeze preceded ABC generation. The
amended numerical source is frozen and publicly published after ABC and
the aborted fitting attempt, before any D generation. The inferential
claims, estimands, margins, and analysis algebra remain unchanged. This is
not represented as a separate public preregistration.

## Data and inferential units

There are 24 fresh repetitions of `linear_ar`, `nonlinear_ar`, `vanderpol`,
and `duffing`, each at added-noise levels 0 and 0.1: 192 tasks. History length
is 12; horizons are 1, 4, and 12. The inherited episode length, origin stride,
integration, burn-in, initial-condition, and observation rules are retained.
A/B/C/D contain 64/32/32/128 independently generated episodes and retain
512/256/256/1664 rows. Each episode supplies up to 13 dependent origins.

The master seed is `2026110173`. Generation, unit-noise, retained-row, and
fit streams use the inherited integer-component SeedSequence derivation.
The same draws are paired across competing methods and noise levels.
Intended-distinct streams are checked against one another and the prior
stream exclusion roster before generation. The row-index-only preflight
also verifies that all 64 A episode identities remain represented and that
every fold and the half-training bridge contain fitting rows. It does not
generate or inspect scientific outcomes.

The inherited added-noise construction is `noise_level * s_A * Z`, where
`s_A` is computed from the unaugmented A episode values; both histories and
targets are augmented. This common simulation scale is retained as part of
the controlled data-generating procedure. Accordingly, fold honesty below
means exclusion of held-out episodes from the learning algorithm at a fixed
generated dataset. It is not a claim that all generated quantities are
unconditionally independent of held-out episode values.

The independent inferential unit is a repetition, giving 24 units. Neither
the 192 tasks nor the origins, horizons, folds, or overlapping fold-training
sets are treated as independent replications. Every prespecified task is
required. Unexpected fitting, selection, or query failures stop execution;
tasks are not replaced, silently omitted, or selected for reporting.

## Common-model candidates and native geometry

The common model consists of the inherited raw ridge forecast plus a
common-geometry residual head. Its multiplier is fixed at one. It is not
the C-calibrated common forecast retained among legacy complete procedures.

The three history transformations are `standard`, `minmax`, and `tanh_z_1`.
Standardization uses fitting-history population mean and standard deviation
with the inherited constant-coordinate rule. Min-max scaling maps fitting
extrema to -1 and 1, maps constant fitting coordinates to zero, and does not
clip query values. Tanh scaling is `tanh((x - mean) / scale)` using the fitted
standardization parameters, with no additional history standardization or
selected multiplier. Feature-level standardization and radial bounding are
separate operations.

New fold, half-training, and departure searches use native caps
`[1, 2, 4, 8, 12]`. Each local model begins at
`min(cap, ambient_dimension, local_reference_count - 1)` and applies the
verified native positive-variance and projected-range rules. Requested cap
and realized rank are recorded separately. There is no extra normal-direction
reservation. Legacy full-A construction retains its cap-20 duplicate; at
ambient dimension 12 it must agree exactly with cap 12, and ties favor 12.
This design does not study a realized rank of 20.

Within a specified fitting block, the raw-ridge penalty is selected on B
separately for each horizon. Each common scaler/cap candidate then fits its
common head on that same fitting block and selects its penalty on B by
horizon. One common scaler and cap are chosen by mean-horizon B MSE divided
by the population target variance of that fitting block, floored at
`1e-12`. Candidate ties use ascending cap, then `standard`, `minmax`,
`tanh_z_1`; penalty ties use the declared ascending grid. Identical common
copies induced by legacy hybrid feature pairs do not receive additional
selection opportunities.

## Fourfold common predictions and full-A deployment refit

Assign each retained A row to the fold `original_episode_id % 4`. Entire
episodes remain together. Each fold has 16 episode identities, with the
other 48 identities supplying its training complement. Retained row counts
may differ between folds.

For each fold k, independently fit and B-select its common model using
only its complement. This exclusion applies to history scalers, partitions,
reference sampling, local bases, feature scalers, raw and common coefficients,
and target-variance normalizers. The fit seed is the same paired task seed;
held-out episode data do not enter the fitting API. Predict every retained
row in fold k once and store the prediction in its original A row position.
The assembled residual target is

`r_OOF,i = Y_A,i - g_minus_fold(i)(X_A,i)`.

The four common models may choose different scalers, caps, and penalties.
They must not reuse settings selected through a full-A fit: such settings
could indirectly depend on the outcome being held out. Sharing B across
folds is declared development reuse; B is not an unbiased evaluation set.
The fold target variance also uses only complement targets.

Separately fit and B-select the deployment common model `g_A` on all 64 A
episodes. It is the gamma-one selected common from the verified full-A
PrimarySuite, whose cap-20 candidate is redundant. Neither C nor D changes
its common scaler, cap, penalties, or coefficients. Store its full-A fitted
predictions and the in-sample residual target

`r_IS,i = Y_A,i - g_A(X_A,i)`.

The OOF common models were trained on 48 episodes whereas `g_A` was trained
on 64. Their selected settings can differ. Consequently OOF residuals are
errors of the fold procedures, not exact errors of the deployment common
model. Transferring their learned correction to `g_A` is the scientific
procedure being tested. Cross-fitting does not remove this training-size
and selection mismatch or guarantee that a useful correction transfers.

Using held-out predictions to learn an error correction and then applying it
to a predictor refitted on all training data follows the construction
described in Wolpert's original stacked-generalization paper, including
its single-predictor correction example [1]. V-fold meta-learning is also
used in the Super Learner literature [2]. These references motivate a
construction; their oracle or asymptotic results are not asserted for this
feature map, shared development set, or finite forecasting experiment. No
double-machine-learning, orthogonal-score, or causal-inference guarantee is
claimed.

## Shared bias and aligned departure features

Freeze one full-A common forecast for all new correction comparisons. Its
shared per-horizon bias is

`b = mean_B(Y_B - g_A(X_B))`.

The common-plus-bias baseline is `g_A(x) + b`. Bias uses B once according to
this formula, with no C-selected bias multiplier. Reusing B for common
selection, this bias, and departure selection is permitted development
reuse and is recorded explicitly.

All departure-history scalers and geometry banks are fitted on retained
full-A histories, with the three scalers and five native caps. Their
post-map feature standardizers are also fitted on A. Bounded features are
centered by their A mean after the radial bound. The same fitted feature
banks and feature statistics are reused for OOF and in-sample targets.
Fold-specific PCA coordinates are never concatenated as if their columns
were aligned. D uses the same fixed full-A feature map as correction fitting.

For each target mode and feature kind, fit the ridge slope using centered
training features and centered residual targets. Predict only the centered
features times the slope; omit the fitted response-mean intercept. Subtracting
the fixed vector b before response centering gives the same slopes. Thus
all forecast intercept/bias differences come from the common fixed baseline,
not a separately fitted correction intercept. A zero training-mean correction
does not imply zero mean on B, C, or D; no query-set mean is subtracted.

These full-A feature fits intentionally use all training histories. The
honesty guarantee applies to the common forecast used to form each OOF
residual, not to an assertion that every downstream transform is cross-fitted.
Correction rows and overlapping fold models are statistically dependent.

B selects a penalty by horizon and then one departure scaler/cap jointly
across horizons, minimizing the full-A-variance-normalized loss of
`g_A(B) + b + departure(B)`. KAHM and ordinary RBF select independently,
separately for OOF and in-sample targets. They have the same scaler/cap menu,
penalty grid, bounded-feature convention, and bias baseline. Ties use cap
first, then the declared scaler order. A matched RBF additionally uses its
own target mode's KAHM-selected scaler and cap, while retaining its own
B-selected penalties. It is a mechanism control and does not replace the
independently selected RBF primary comparator.

## Calibration, controls, and policy selection

C selects `eta` from `[0, 0.25, 0.5, 1]` separately for each horizon and each
of the six new corrections. The forecast is

`g_A(x) + b + eta * departure(x)`.

Eta zero must recover the shared-bias baseline exactly. C cannot revise any
common model, departure scaler/cap, ridge penalty, bias, or feature fit.
The in-sample-residual ablation has the same full-A common, bias, feature
candidates, and 512 correction rows as its OOF counterpart. Its independent
B and C choices are part of the evaluated procedure. This comparison
isolates the residual-generation scheme at the procedure level; it does not
hold the finally selected departure parameters fixed.

A descriptive training-size bridge fits and B-selects a common-only model
on the first 32 sorted original A episode identities, with that subset's
history/target statistics. Its own B mean residual supplies its bias version.
This contrasts half-training and full-training common predictors. It does
not rerun the previous P/R correction experiment, isolate all differences
between historical half-splitting and cross-fitting, or establish superiority
over that historical correction procedure.

The inherited complete-procedure roster is refitted on the fresh A/B/C
data. It includes ridge, raw Nyström RBF with its three-scaler B search, raw
histogram gradient boosting, persistence, common geometry, bounded and
unbounded KAHM, bounded and unbounded RBF, and the legacy non-KAHM policy.
Its geometry procedures retain concatenated common-plus-departure heads
and the original five preprocessing pairs and six caps. Their fitted common
coefficients/intercepts may differ between methods. They must not be
described as the new shared-common correction architecture. Legacy C
calibration retains its blend toward raw ridge; new correction eta blends
toward the common-plus-bias baseline.

The expanded non-KAHM policy selects one complete candidate family using
calibrated C forecasts and equal-horizon C MSE normalized by full-A target
variance. Its deterministic tie order is:

1. Legacy ridge.
2. Legacy raw Nyström RBF.
3. Legacy raw histogram gradient boosting.
4. Legacy persistence.
5. Legacy common geometry.
6. Legacy bounded RBF departure.
7. Legacy unbounded RBF departure.
8. New full-A common plus shared bias.
9. New independently selected OOF RBF correction.
10. New independently selected in-sample RBF correction.

Every candidate's B settings and C multiplier are fixed before the family
choice. KAHM-derived matched-RBF settings are not included in this non-KAHM
menu. This is a prespecified C-selected policy, not an oracle guaranteed to
be the strongest candidate on D. The same selected family is forwarded in
the multiplier-one diagnostic; neither D nor uncalibrated loss selects a
different family.

Exactly 42 D arrays per task are retained: ten legacy selected-context
methods in calibrated and gamma-one modes; ten new methods in calibrated
and eta-one modes; and the expanded policy in both modes. The ten new
methods are full-A common, shared bias, half-training common, its bias
version, OOF KAHM/RBF/matched RBF, and in-sample KAHM/RBF/matched RBF.
Uncalibrated diagnostics retain the already selected identities. Candidate
training/validation evidence is saved without generating a D forecast for
every rejected candidate.

## Four primary claims and descriptive analyses

For task t, horizon h, and method m, define

`SMSE(t,h,m) = mean_D((Y_D,h - prediction_D,h,m)^2)
               / max(Var_A(Y_A,h, ddof=0), 1e-12)`.

All methods share the same full-A denominator at evaluation, including the
half-training bridge. Average horizons equally, then noise levels and laws
equally within a repetition. The nonlinear repetition average uses 18
law/noise/horizon cells from `nonlinear_ar`, `vanderpol`, and `duffing`; the
linear average uses six `linear_ar` cells. The point gain is the percentage
reduction of the mean repetition loss, not an average of cell percentages.

All four primary claims use calibrated OOF KAHM forecasts:

| Panel | Comparator | Required margin |
| --- | --- | --- |
| Nonlinear | Same full-A common plus shared bias | At least 3% lower loss |
| Nonlinear | Independently selected OOF RBF correction | At least 3% lower loss |
| Nonlinear | Expanded C-selected non-KAHM policy | At least 3% lower loss |
| Linear | Raw ridge fitted on A | At most 5% higher loss |

For each nonlinear comparison the repetition margin gap is
`0.97 * L_comparator - L_OOF_KAHM`; for linear noninferiority it is
`1.05 * L_ridge - L_OOF_KAHM`. With n=24 and family alpha=0.05, use the
one-sided paired-t lower endpoint

`mean(gap) - t.ppf(1 - 0.05/4, 23) * sd(gap, ddof=1) / sqrt(24)`.

A claim is established only when this endpoint is strictly positive. The
Bonferroni family contains exactly these four claims. Report all four with
their means, point gains, gaps, uncertainty, and established flags. A failed
margin test is not evidence of equality, and a favorable secondary comparison
does not replace a failed primary claim. These are approximate repetition-level
inferences under the declared synthetic population, not distribution-free
guarantees.

Prespecified descriptive analyses cover matched RBF; OOF versus in-sample
corrections; full-A versus half-training common and bias forecasts; legacy
complete procedures and legacy versus expanded policies; eta-one versus
calibrated forecasts; law/noise/horizon heterogeneity; selected settings and
eta frequencies; fold/full common differences; and computational/numerical
diagnostics. Ordinary paired secondary intervals are unadjusted two-sided
95% t intervals over 24 repetitions and are labeled descriptive. No D-selected
scaler, cap, fold assignment, or policy is introduced.

## Execution gates, audit, and reporting

The frozen source manifest covers the configuration, fitting, generation,
staging, and analysis definitions and their inherited dependencies. A/B/C
development is completed for all 192 tasks and sealed before D generation.
Prediction workers receive only D histories; all 192 prediction artifacts
are sealed before terminal-target release and scoring. Source hashes,
canonical artifact membership, task counts, and fit/prediction state hashes
must verify at the gates. Targets may exist on the filesystem after D
generation, so this is audited procedural masking, not cryptographic or
access-control isolation.

Manufactured-data readiness checks independently verify whole-episode fold
coverage, excluded-fold fitting invariance, complement-only normalizers,
aligned feature maps, selected B settings, slope-only centering, common/bias
identities, calibration, policy selection, multiplier-zero recovery, and
target-free prediction. Production audits run only after the release gate.
Saved evidence distinguishes declared fallbacks and normalized reconstruction
guards from unexpected factorization, coordinate, kernel, or gate failures.
Unexpected failures stop execution. The documented ridge SVD recovery
policy is applied uniformly to all ridge paths and logged separately from
the original unhandled failure; see the amendment. Native zero-rank fallback use is counted
explicitly rather than labeled a positive-dimensional native fit.

The inherited maximum task-error-ratio screen compares selected OOF KAHM
with ridge at threshold five. Passing it does not establish any primary
superiority claim. Whole-stage times include all common folds, searches,
ablations, and controls. Partial primitive timings are diagnostics, not a
complete isolated KAHM-versus-RBF computational ranking. Cached construction
is not described as free. At most three single-thread numerical workers
run on the shared host; memory, evidence-checkpoint bytes, and deployment
state estimates, when available, are reported as different quantities.

The report leads with the four primary results, then the residual-generation
ablation, comparator and training-size controls, heterogeneity, numerical
behavior, and resource scope. It states the fold-to-full refit mismatch,
repeated B/C development use, dependent origins/fold fits, finite synthetic
population, and absence of a natural-data deployment claim. Any post-freeze
amendment is identified and preserved separately.

## Methodological references

1. Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2),
   241-259. [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).
   The original paper's Figures 1 and 2 explicitly construct held-out
   learning examples before querying base predictors trained on the full
   learning set; Figure 2 concerns correction of one predictor's error.
2. van der Laan, M. J., Polley, E. C., and Hubbard, A. E. (2007).
   Super Learner. *Statistical Applications in Genetics and Molecular
   Biology*, 6, Article 25.
   [doi:10.2202/1544-6115.1309](https://doi.org/10.2202/1544-6115.1309).
   [Author-hosted working paper](https://biostats.bepress.com/ucbbiostat/paper222/).
