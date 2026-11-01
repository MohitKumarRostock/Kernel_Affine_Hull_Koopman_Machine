# Incremental forecasting value of a KAHM policy component

## Scientific question and provenance

This prospective computation tests whether admitting the current centered,
bounded, in-sample KAHM residual head improves a strong forecasting selection
policy on newly generated repetitions. The hypothesis follows the results of
the preprocessing, dimension-selection, and cross-fitted residual studies.
Those observations inform the design and are not pooled with the new cohort.
The starting source is commit
`2acd4f5f8b4970ef0e066f1b91a3636d36ad932f` of
`MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine`.

The source, configuration, seed exclusions, and analysis are locally frozen
before generating any new A/B/C or D observations. This is a timestamped local
design freeze, not a public preregistration. Stage seals describe execution
order and integrity; they do not provide cryptographic access isolation.

## Policies

P0 is the unchanged ten-family expanded non-KAHM policy from the residual
study, in the following tie-breaking order:

1. Raw-history ridge.
2. Raw-history Nyström RBF.
3. Histogram gradient boosting.
4. Persistence.
5. Common geometric features.
6. Bounded joint common/RBF departures.
7. Unbounded joint common/RBF departures.
8. Full-A common predictor plus its B-fitted shared bias.
9. Cross-fitted centered RBF departure correction.
10. In-sample centered RBF departure correction.

P1 appends one independently B-selected, C-calibrated, in-sample KAHM
correction family to P0. P0plus appends one new independently B-selected,
C-calibrated raw-history Nyström RBF residual family to P0. Both additions
correct the identical full-A common forecast and shared B bias. Each policy
chooses one complete family on C by mean standardized loss across all three
horizons. Ties retain the first member of the stated order. The additions
are optional, and neither policy uses D outcomes for selection.

The extra RBF family is distinct from both the complete raw-history RBF
forecaster and the local RBF-departure correction already present in P0. It
uses A-centered, radially bounded Nyström features with 64 landmarks and an
in-sample residual slope with no additional response-mean intercept.
Landmarks, whitening, history transforms, and feature statistics use A only.
Bandwidth factors multiply the median squared landmark distance, matching
the inherited implementation.

The existing legacy and cross-fit suites are executed unchanged, including
diagnostic families not admitted into these three policies. This preserves
the P0 implementation and provides secondary comparisons without changing
the primary estimand. The full executed suite cost is recorded. It includes
diagnostic fits and is not an isolated minimal deployment cost for P0 or P1.

## Dimension, preprocessing, and selection opportunities

KAHM and local RBF residual candidates retain caps {1,2,4,8,12}, crossed with
standardization, min-max scaling, and tanh of standardized histories. The
native initial rank is min(cap,12,N_local-1), followed by the existing
positive-variance and range reductions. No additional normal direction is
reserved. At ambient dimension 12, cap 20 is identical to cap 12; the
inherited complete search retains and checks this duplicate. Requested caps
and realized local ranks are reported separately.

The extra raw-history RBF residual family searches the same three history
transforms and five bandwidth factors {0.25,0.5,1,2,4}. Both added families
therefore evaluate 15 feature configurations and the same five penalties
{0.0001,0.001,0.01,0.1,1}: 75 B loss opportunities per horizon. Penalties
are selected per horizon; one feature configuration is selected jointly
across horizons. Each addition supplies exactly one family to C selection.
This matches candidate counts, not feature dimension, runtime, or statistical
capacity. Those differences remain part of the comparison and are reported.

For every residual family C selects a multiplier in {0,0.25,0.5,1} per
horizon. Zero reproduces the identical common-plus-bias forecast exactly.
After this calibration, C selects the policy family. Reuse of C for these
development decisions is deliberate and identical across policies; D
provides the evaluation of the resulting complete procedures.

## Data and execution

The inherited generators and observation rules define four laws: linear AR,
nonlinear AR, Van der Pol, and Duffing. History width is 12; forecast horizons
are 1,4,12. Noise levels are 0 and 0.1. Twenty-four new repetition indices
produce 192 tasks. The master seed is 2026110191. All 1,272 intended-distinct
streams are checked against one another and the previous study exclusions,
including the entire original cross-fit cohort. The seed roster is fixed
before generation, with no optional stopping or outcome-based replacement.

Each task retains the original independent-episode roles: 64 A episodes for
fitting, 32 B for development selection, 32 C for calibration and policy
selection, and 128 D for terminal scoring. A/B/C row caps are 512/256/256.
The 13 origins per D episode are retained. Loss scales are full-A target
population variances, floored at 1e-12. Generator streams pair noise variants
and candidate procedures; repetition aggregates are the independent units.

Every task is fitted in a fresh worker with one numerical thread. The
unchanged checked-SVD recovery rule applies only after its declared numerical
exception. Unexpected failures halt the campaign rather than dropping a task
or changing the candidate menu. Complete fitted states, all B candidate
predictions, all C candidate and policy predictions, selected local bases,
calibration records, and resource measurements are retained. All fits are
sealed before D generation; all D forecasts are sealed before target release
and analysis. No selected settings are changed after D is scored.

## Primary analysis

For every procedure, horizon MSE is divided by the corresponding A variance.
Horizons, the two noise conditions, and laws receive equal weight within a
repetition. The nonlinear panel averages nonlinear AR, Van der Pol, and
Duffing; the linear panel uses linear AR only. The four prespecified claims
are:

1. P1 improves nonlinear loss over P0 by at least 3%.
2. P1 improves nonlinear loss over P0plus by at least 3%.
3. P1 is noninferior to P0 on the linear panel within a 5% excess-loss margin.
4. P1 is noninferior to P0plus on the linear panel within the same 5% margin.

For superiority, repetition r contributes g_r=0.97 L_reference,r-L_P1,r;
for noninferiority it contributes g_r=1.05 L_reference,r-L_P1,r. A claim is
established only if mean(g)-t_(1-0.05/4,23)*sd(g)/sqrt(24)>0. All four
bounds and failures are reported. These are approximate paired-t inferences;
the forecast origins, tasks, noise variants, and arrays are not independent
replicates. Point gains are 100*(1-mean(L_P1)/mean(L_reference)).

Secondary outputs include unshrunk forecasts using the same C-selected
family, standalone in-sample KAHM, policy selection frequencies, realized
ranks, preprocessing choices, law/noise heterogeneity, maximum-error screens,
and full resource measurements. Comparisons restricted to cases in which C
selected the added family are conditional descriptive results, not a new
superiority test. No secondary contrast replaces a failed primary claim.

## Decision and scope

Passing both nonlinear comparisons and linear safeguards would support a
narrow claim about incremental policy value under these finite generators,
menus, and budgets. Improving P0 but failing against P0plus would leave an
extra-search explanation unresolved. A null or adverse result limits the
case for routine admission of this KAHM component; it does not disprove the
previous feature-span results or all possible uses of KAHM.

A separate natural-series extension, with its own source freeze and
chronological protocol, will examine previously unused M4 series identifiers.
Its results are not pooled into these four primary claims. The original
manuscripts are not rewritten before the new evidence has been evaluated.

## Methodological references

- Cawley and Talbot (2010), On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation, JMLR 11:2079-2107.
  https://www.jmlr.org/papers/v11/cawley10a.html
- Hyndman and Athanasopoulos, Forecasting: Principles and Practice, Chapter 5.
  https://otexts.com/fpp3/toolbox.html
