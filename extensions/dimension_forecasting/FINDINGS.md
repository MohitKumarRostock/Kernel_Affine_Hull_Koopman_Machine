# Forecasting with KAHM subspace dimensions selected on validation data

## Main result

Selecting the local KAHM dimension on B improves the mean nonlinear forecast
error relative to the historical cap-2 procedure, but does not establish the
complete six-comparison performance criterion. The selected bounded KAHM
procedure attains nonlinear standardized mean squared error (SMSE)
0.028911616, compared with
0.029884836 for historical cap 2 and 0.032392322
for native cap 20. The selected procedure establishes the declared 3%
superiority margins over common geometry and the raw Nyström RBF competitor,
and the 5% noninferiority margin to ridge on the linear negative control.
It does not establish the 3% margins over the bounded RBF departure, histogram
gradient boosting, or the C-selected non-KAHM policy. The policy has lower
nonlinear mean error than KAHM. These statements concern the declared aggregate
and tuning procedures, rather than every law, noise level, or horizon.
([Primary results](analysis/primary_contrasts.csv),
[panel results](analysis/panel_summary.csv).)

## Experimental scope and estimand

The experiment comprises 24 fresh independent repetitions of four numerical
observation laws: linear AR, nonlinear AR, Van der Pol, and Duffing. Each
repetition is evaluated at added-noise levels 0 and 0.1, giving 192 tasks.
History length is 12 and the forecasting horizons are 1, 4, and 12. The
A/B/C/D splits contain 64/32/32/128 independent episodes. A/B/C retain
512/256/256 rows; D retains all 1664 rows from 13 origins per episode.
The original law parameters, observation maps, episode construction, tuning
grids, and bounded readout definitions are retained. Added noise affects both
histories and targets. Noise levels and dimension candidates share underlying
episode streams and reference-sampling seeds. The seed roster is disjoint from
the original principal experiment and the preceding dimension study.
([Configuration](config.json), [protocol](PROTOCOL.md),
[seed identities](readiness/prior_seed_exclusions.json).)

For horizon h, SMSE is D mean squared error divided by the empirical variance
of the retained A targets (divisor N, ddof=0), with denominator floor 1e-12. Horizon losses are
averaged equally; the nonlinear panel then gives equal weight to each of the
three nonlinear laws and two noise levels within a repetition. The linear
panel averages its two noise levels. The independent statistical unit is the
repetition. Rows and origins within an episode are not treated as independent
replications. The reported mean is the average of the 24 repetition aggregates.
([Analysis protocol](analysis_code/ANALYSIS_PROTOCOL.md),
[repetition losses](analysis/repetition_losses.csv).)

Every bank has requested local group counts 2, 4, and 8, with at most 24
reference rows per local model. The native policy begins at
`min(cap, d, N_local - 1)` and applies the verified native eigenvalue and
projected-range rules. The historical cap-2 comparator retains its explicit
legacy policy. Native caps 1, 2, 4, 8, 12, and 20 are fitted independently.
Each of the five geometry procedures selects one cap across horizons using
its B gamma-one SMSE after per-horizon ridge-penalty selection. Exact ties
favor the smaller cap. These choices are saved before C is loaded for
calibration. C selects multipliers from 0, 0.25, 0.5, and 1 per horizon for all
nonridge procedures. The raw RBF, HGB and persistence procedures therefore
receive the same declared calibration opportunity. Persistence is the
last-observation candidate blended with the shared ridge baseline.
([B selections](analysis/cap_selections.csv),
[B tuning choices](analysis/B_tuning_choices.csv),
[C calibration](analysis/C_gamma_choices.csv).)

The primary context uses each geometry procedure's own B-selected cap. The
two fixed contexts, historical2 and native20, are secondary. Within each
context, the non-KAHM policy selects one complete procedure on C from ridge,
raw RBF, HGB, persistence, common geometry, bounded RBF and unbounded RBF.
Neither KAHM variant is a policy candidate. C does not select dimensions:
each geometry procedure's cap has already been fixed on B.
([C policy choices](analysis/C_policy_choices.csv).)

## Complete ten-procedure comparison

The following table contains the calibrated selected context. Lower SMSE is
better. Common geometry and both RBF procedures select their own caps; they
are not constrained to use the bounded KAHM procedure's chosen bank.

| Procedure | Nonlinear SMSE | Linear SMSE |
| --- | --- | --- |
| Ridge | 0.059223 | 0.901659 |
| Raw + Nyström RBF | 0.031429 | 0.904691 |
| Histogram gradient boosting | 0.029619 | 0.904476 |
| Persistence | 0.059406 | 0.902609 |
| Common geometry | 0.031775 | 0.902291 |
| Bounded KAHM departure | 0.028912 | 0.903418 |
| Bounded RBF departure | 0.029122 | 0.902816 |
| Unbounded KAHM departure | 0.029414 | 0.903481 |
| Unbounded RBF departure | 0.029436 | 0.903145 |
| C-selected non-KAHM policy | 0.027778 | 0.906497 |

Source: [panel_summary.csv](analysis/panel_summary.csv). The policy has the
smallest nonlinear aggregate, whereas ridge has the smallest linear aggregate.
KAHM is approximately 0.195% worse than ridge on the linear panel, which is
within the declared 5% noninferiority margin; this is not a claim that KAHM
improves on the linear model.

## Six simultaneous primary comparisons

For a nonlinear comparator, each repetition's margin gap is
`0.97 × comparator SMSE − KAHM SMSE`. For the linear negative control it is
`1.05 × ridge SMSE − KAHM SMSE`. A claim is established when
`mean gap − t(1 − 0.05/6, 23) × sample_sd / sqrt(24) > 0`.
The critical t value is 2.582017198. The point gain
below is `100 × (comparator mean − KAHM mean) / comparator mean`; a negative
gain indicates higher KAHM error.

| Comparison | Margin | Point gain (%) | Simultaneous lower gap | Established |
| --- | --- | --- | --- | --- |
| Nonlinear: KAHM vs Common geometry | 3% superiority | 9.012 | 0.00152902 | Yes |
| Nonlinear: KAHM vs Bounded RBF departure | 3% superiority | 0.723 | -0.00080020 | No |
| Nonlinear: KAHM vs Raw + Nyström RBF | 3% superiority | 8.009 | 0.00115633 | Yes |
| Nonlinear: KAHM vs Histogram gradient boosting | 3% superiority | 2.389 | -0.00066232 | No |
| Nonlinear: KAHM vs C-selected non-KAHM policy | 3% superiority | -4.080 | -0.00221776 | No |
| Linear: KAHM vs ridge | 5% noninferiority | -0.195 | 0.04187953 | Yes |

Source: [primary_contrasts.csv](analysis/primary_contrasts.csv). Three of six
comparisons are established. The complete joint statistical criterion is
therefore not established. The maximum-error screens pass for every selected KAHM task;
their largest error ratio to ridge is 1.467311,
below the declared threshold 5. Passing these screens does not compensate
for failed primary margins. The paired-t bounds are approximate statistical
comparisons over the repetition population; they are not risk certificates.
([Stability screens](analysis/stability_screens.csv),
[analysis summary](analysis/analysis_summary.json).)

## Dimension selection and fixed-cap sensitivity

| KAHM procedure | Nonlinear SMSE | Gain over historical cap 2 (%) |
| --- | --- | --- |
| Historical cap 2 | 0.029885 | 0.000 |
| Native cap 1 | 0.030766 | -2.948 |
| Native cap 2 | 0.029885 | 0.000 |
| Native cap 4 | 0.029780 | 0.350 |
| Native cap 8 | 0.031139 | -4.196 |
| Native cap 12 | 0.032392 | -8.390 |
| Native cap 20 | 0.032392 | -8.390 |
| B-selected native cap | 0.028912 | 3.257 |

Source: [panel_summary.csv](analysis/panel_summary.csv). Relative to historical
cap 2, B selection reduces mean nonlinear SMSE by
3.257%, with improvement in
21 of 24 repetition means. The secondary paired
difference is -0.000973220, with descriptive
unadjusted 95% interval [-0.001229272,
-0.000717168]. Native cap 20 increases mean error by
8.390% and is worse in all 24 repetition
means. These are secondary comparisons, outside the six-member primary family.
At ambient dimension 12, native caps 12 and 20 define exactly the same maps.
They were fitted separately, and exact numerical identity was verified.
([Secondary contrasts](analysis/secondary_contrasts.csv).)

Bounded KAHM chooses the following caps on B:

| Selected cap | Nonlinear tasks (n=144) | Linear tasks (n=48) |
| --- | --- | --- |
| 1 | 15 | 4 |
| 2 | 22 | 14 |
| 4 | 68 | 18 |
| 8 | 31 | 9 |
| 12 | 8 | 3 |
| 20 | 0 | 0 |

Source: [cap_selection_counts.csv](analysis/cap_selection_counts.csv). Cap 4
is the most frequent choice, but a fixed cap 4 has nonlinear SMSE
0.029780294, above the selected procedure's
0.028911616. The frequencies do not justify
a universal fixed cap. Cap 20 is never selected because its exact tie with
cap 12 is resolved in favor of 12.

## Heterogeneity across laws, noise levels and horizons

Each entry below averages the three horizons and 24 repetitions within the
specified condition. The historical and native20 columns refer to bounded
KAHM; selected RBF is the independently selected bounded RBF procedure.

| Law | Noise η | Selected KAHM | Cap-2 KAHM | Cap-20 KAHM | Selected RBF | HGB | Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nonlinear AR | 0.0 | 0.019275 | 0.019642 | 0.019374 | 0.019297 | 0.020382 | 0.019233 |
| Nonlinear AR | 0.1 | 0.031087 | 0.031470 | 0.031340 | 0.031195 | 0.032308 | 0.031097 |
| Van der Pol | 0.0 | 0.009347 | 0.009771 | 0.018688 | 0.010642 | 0.003801 | 0.003801 |
| Van der Pol | 0.1 | 0.057353 | 0.061047 | 0.058292 | 0.056403 | 0.058629 | 0.056152 |
| Duffing | 0.0 | 0.011470 | 0.012597 | 0.015717 | 0.012890 | 0.013514 | 0.011753 |
| Duffing | 0.1 | 0.044937 | 0.044783 | 0.050943 | 0.044306 | 0.049080 | 0.044633 |

Source: [condition_summary.csv](analysis/condition_summary.csv). Dimension
selection improves KAHM over historical cap 2 in five of the six nonlinear
condition means. Duffing with added-noise level 0.1 is the exception, with a
small increase. The largest KAHM deficit to HGB and the policy occurs for
Van der Pol at noise level 0: KAHM SMSE is
0.009347494, whereas HGB
and the policy both attain
0.003801400. The C policy
selects HGB in all 24 repetitions in that condition. This heterogeneity is
visible in the complete panel and is not evidence for selecting a preferred
procedure after observing D.

The nonlinear horizon means are:

| Horizon | Selected KAHM | Cap-2 KAHM | Cap-20 KAHM | Selected RBF | HGB | Policy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.011201 | 0.011654 | 0.011207 | 0.011204 | 0.011742 | 0.011281 |
| 4 | 0.019491 | 0.021471 | 0.019108 | 0.019516 | 0.020626 | 0.019431 |
| 12 | 0.056043 | 0.056529 | 0.066862 | 0.056647 | 0.056490 | 0.052623 |

Source: [panel_horizon_summary.csv](analysis/panel_horizon_summary.csv).
Native cap 20 improves the short-horizon means relative to cap 2 but increases
the horizon-12 mean error. B-selected KAHM improves on cap 2 at all three
horizon means. The policy's aggregate advantage is particularly evident at
horizon 12. Individual condition-by-horizon values remain available in
[condition_horizon_summary.csv](analysis/condition_horizon_summary.csv);
these disaggregated results are descriptive.

## Common features and reconstruction departures

Matched controls use the bounded KAHM procedure's B-selected bank for all five
geometry heads, while preserving each head's declared B/C tuning. They differ
from the independently selected competitors in the primary comparison. The
change from historical cap 2 decomposes algebraically into a change in the
common predictor and a change in the incremental KAHM departure contribution:

`ΔKAHM = Δcommon + Δ(KAHM − common)`.

| Bank change from historical cap 2 | Complete KAHM change | Common-feature change | Change in departure increment |
| --- | --- | --- | --- |
| KAHM B-selected bank | -0.000973220 | -0.002149833 | +0.001176613 |
| Native cap-20 bank | +0.002507485 | -0.001805995 | +0.004313480 |

Source: [decomposition_summary.csv](analysis/decomposition_summary.csv).
For the selected bank, the common predictor improves by 0.002149833 SMSE,
while the benefit supplied by the KAHM departure becomes smaller by
0.001176613. Their sum is the complete improvement of 0.000973220. Thus the
dimension-selection gain is not evidence that the distinctive reconstruction
departure has become more effective. Native cap 20 also improves the common
predictor, but loses enough departure benefit to worsen the complete KAHM
forecast. The decomposition is an identity for the paired procedures; it is
not a causal attribution across independently selected banks.

In the same selected KAHM bank, bounded KAHM improves over bounded RBF by
0.682% in the
nonlinear mean. With separate B cap selection, the corresponding advantage is
0.723%.
Both descriptive paired 95% intervals exclude zero, but neither establishes
the declared 3% primary margin. The within-bank comparison supports a small
incremental distinction under this protocol; it does not support a broad
practical superiority claim.

Under the same bank, bounded KAHM improves over its unbounded-departure
variant by 2.278%,
with improvement in all 24 repetition means. With independently selected
caps, the reduction is
1.709%.
The unbounded variant retains the same bounded common block and removes only
the departure radial bound. These secondary comparisons therefore concern
that feature definition together with the declared head tuning.
([Secondary contrasts](analysis/secondary_contrasts.csv).)

The selected nonlinear KAHM multipliers are 1 in 375 of 432 task-horizon cells,
0.5 in 54, 0.25 in two, and 0 in one. In the linear panel they are 0 in 92 of
144 cells. A zero multiplier returns the shared ridge forecast exactly. The
gamma-one ablation retains the selected-versus-historical direction, with
a 3.392% mean nonlinear reduction, so calibration does not create that
dimension-selection effect. ([C calibration](analysis/C_gamma_choices.csv),
[secondary contrasts](analysis/secondary_contrasts.csv).)

## Realized dimensions, widths and numerical behavior

All 1344 candidate banks contain 14 local models. The independent audit
reconstructs all 18,816 local PCA bases from the saved references. The
requested policy cap and the realized local basis rank are distinct:

| Bank | Local models | Actual rank range | Mean rank | Full rank 12 | Rank zero | Common width |
| --- | --- | --- | --- | --- | --- | --- |
| historical2 | 2688 | 1–2 | 1.998884 | 0 | 0 | 40–42 |
| native1 | 2688 | 1–1 | 1.000000 | 0 | 0 | 28–28 |
| native2 | 2688 | 1–2 | 1.998884 | 0 | 0 | 40–42 |
| native4 | 2688 | 1–4 | 3.996652 | 0 | 0 | 64–70 |
| native8 | 2688 | 1–8 | 7.992188 | 0 | 0 | 112–126 |
| native12 | 2688 | 1–12 | 11.987723 | 2685 | 0 | 160–182 |
| native20 | 2688 | 1–12 | 11.987723 | 2685 | 0 | 160–182 |

Source: [local_dimensions.csv](analysis/local_dimensions.csv) and
[geometry_summary_by_task.csv](analysis/geometry_summary_by_task.csv).
In the native20 bank, 2685 of 2688 local models have rank 12.
The remaining three have two references and initial admissible rank 1.
All banks use the same reference sets; reference counts range from 2 to 24.
There are no additional positive-eigenvalue or projected-range rank reductions
in this realization, no native-construction failures, and no rank-zero models.
Every stacked basis matrix has rank 12, including the historical local-rank-2
and native local-rank-1 banks. A small local rank is therefore not a global
two-dimensional bottleneck.

The departure width is 168 for both KAHM and RBF at every cap. The bounded and
unbounded full readouts concatenate that block with the common block; their
width is the reported common width plus 168. Common width changes with the
realized local dimensions. All historical2/native2 and native12/native20
identity controls and all reference-sharing checks pass.

There are no logged reconstruction/coordinate/kernel/gate fallback rows,
factor failures, bounded-feature violations, coefficient-bound violations, or
bounded D-correction violations. The largest observed coefficient-norm ratio
to its bound is 0.715360; the largest bounded D-correction ratio is 0.282433.
Unbounded departure heads do not have a global feature/correction bound and
are not counted as passing such a bound. Observed numerical checks do not
replace distributional generalization evidence.
([Guard counts](documentation/observed_guard_summary.json),
[feature checks](analysis/feature_checks.csv),
[bound checks](analysis/bound_checks.csv),
[independent audit](audit/post_release/final_results_audit.json).)

## Construction cost and retained state

The following means use the inherited per-procedure dependency accounting.
The selected geometry procedures pay for all six native cap candidates and
their B selection; the policy pays the union of required construction nodes.
Querying uses the selected frozen state, rather than repeating a cap search.

| Procedure / context | Assigned fit + selection time (s) | Retained state (KiB) |
| --- | --- | --- |
| Bounded KAHM departure / historical2 | 0.0855 | 122.13 |
| Bounded KAHM departure / native20 | 0.1011 | 183.30 |
| Bounded KAHM departure / selected | 0.4899 | 136.94 |
| C-selected non-KAHM policy / selected | 1.2612 | 136.91 |
| Histogram gradient boosting / selected | 0.6810 | 370.48 |
| Raw + Nyström RBF / selected | 0.0153 | 68.05 |
| Bounded RBF departure / selected | 0.3800 | 68.74 |

Source: [resource_summary.csv](analysis/resource_summary.csv). Assigned
primitive costs differ from total process wall time, which also includes
imports, provenance checks, serialization and diagnostic work. Retained-state
sizes follow the source's pruned per-procedure convention and are not peak
RAM or the size of the complete study checkpoints. Timings were recorded on
a shared host with one numerical thread per worker and three workers; they
are descriptive and do not establish isolated hardware speed rankings. The
full fit stage took 308.219 seconds and prediction took
227.521 seconds. ([Task resources](analysis/task_resources.csv),
[archive inventory](documentation/KAHM_Forecasting_Dimension_Records_manifest.json).)

## Interpretation and limits

The results support treating local dimension as a validation-selected
construction parameter in this forecasting representation. They also retain
evidence for useful bounded reconstruction features relative to the common
and raw-RBF procedures. They do not establish the complete claim that KAHM
exceeds every declared nonlinear comparator by 3%. The non-KAHM policy has
lower aggregate error, and the incremental advantage over the matched RBF
departure is small. A construction-faithful default and a predictive
hyperparameter choice answer different questions; the native full-ambient
setting is not uniformly better here.

The observations are fresh repetitions of previously studied numerical laws,
not independent natural-data application cohorts. The design fixes history
length 12, three horizons, A-only standard scaling, one finite cap grid, one
feature family and the original raw-competitor grids. It does not evaluate
tanh scaling, longer histories, a cap-20 truncation in ambient dimension
greater than 20, risk-certificate coverage, or a deployment adoption decision.
The six primary bounds use approximate paired-t inference. Source and
analysis were frozen before generation, but this was not a public
preregistration. Condition-specific observations and secondary intervals
should be interpreted within those limits.

## Integrity and complete records

All 192 tasks completed with no omitted candidates, task failures, refits or
changes to frozen source. All fitted states were sealed before D generation,
all 31,104 prediction arrays were sealed before target release, and all eight
saved states per task were unchanged by prediction. The independent final
audit passed all 27 CSV exports and 261,450 rows, including exact generator
and target reconstruction and all 18,816 local bases.
([Audit receipt](audit/post_release/final_results_audit.json),
[analysis integrity](analysis/analysis_summary.json).)

Complete numerical records are preserved in six disjoint four-repetition
archives. Every original member's SHA-256 digest and ZIP CRC were verified;
the archive manifest records all 4041 files and the upstream phase seals.
The [README](README.md) gives restoration and reproduction instructions.
