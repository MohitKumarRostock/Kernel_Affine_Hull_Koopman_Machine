# Incremental value of a KAHM forecasting policy component

This study evaluates whether admitting the existing centered, bounded, in-sample KAHM residual correction improves an established forecasting selection policy. Its design follows the previous dimension, preprocessing and residual studies; their observations are not included in the new inferential sample. The simulations contain 24 new repetitions. A separate evaluation uses 48 M4 training-series histories and is interpreted as external benchmark-development evidence.

Neither nonlinear superiority criterion was established. Both linear noninferiority criteria were established. P1 reduced nonlinear standardized error by 0.816% relative to P0 and 0.658% relative to P0plus, below the declared 3% margins. On the external natural-series panel, P1 had 2.273% higher standardized error than P0 and 2.718% lower error than P0plus; these latter differences are descriptive. The numerical estimates and simultaneous decisions below determine the scope of these statements. Natural-series results and condition-specific summaries do not substitute for a failed synthetic criterion. [Analysis](analysis/analysis_summary.json); [independent audit](verification/policy_audit.json).

## Policies and evaluation design

P0 retains the complete ten-family expanded non-KAHM menu from the preceding residual study. P1 appends one B-selected KAHM family. P0plus appends one B-selected raw-history Nyström RBF residual family. Both added families correct the same full-training common forecast and shared B-fitted bias, use centered residual slopes, and permit a zero correction. C calibrates per-horizon correction strength and then selects a whole forecasting family across horizons. The test partition D is reserved for evaluating the resulting procedures.

The KAHM addition searches five native caps, {1, 2, 4, 8, 12}, and three input transformations: standardization, min–max and tanh of standardized histories. The extra RBF addition searches the same transformations and five bandwidth factors. Both evaluate 75 B losses per horizon, using five ridge penalties. Equal score counts do not imply equal feature width, numerical rank, runtime or statistical capacity. Native initial dimension remains min(cap, 12, N_local−1), followed by the inherited variance and range reductions. A cap of 20 is equivalent to 12 at ambient dimension 12 and is retained only in the inherited complete search. [Protocol](PROTOCOL.md); [configuration](config.json).

The synthetic design crosses four laws, two added-noise levels and 24 repetitions, giving 192 task suites. Each task predicts horizons 1, 4 and 12 from a history of width 12. Horizon mean squared errors are divided by the corresponding A-target population variances, then averaged across horizons. Law/noise cells are averaged within each repetition. The 24 repetition aggregates are the independent units; tasks, forecast arrays and horizons are not additional replications. Source, fits and predictions were sealed before their respective later stages. The local freeze documents execution order and is not an independent public preregistration.

## Synthetic policy losses

| Panel | P0 SMSE | P1 SMSE | P0plus SMSE | Independent repetitions |
| --- | --- | --- | --- | --- |
| nonlinear | 0.027268385 | 0.027045827 | 0.02722502 | 24 |
| linear | 0.91390714 | 0.91392167 | 0.91393655 | 24 |
| All (descriptive) | 0.24892807 | 0.24876479 | 0.2489029 | 24 |

SMSE denotes the standardized loss defined above. “All descriptive” pools the fixed linear and nonlinear condition means for presentation only; it has no additional primary test. The policy means describe the complete C-selection procedures, including instances where an added family was rejected. [Panel table](analysis/panel_summary.csv).

## Four prespecified primary criteria

| Criterion | Relative gain of P1 | Mean margin difference | Simultaneous upper bound | Established |
| --- | --- | --- | --- | --- |
| P1 vs P0 nonlinear | +0.816% | 0.00059549334 | 0.00071141207 | No |
| P1 vs P0plus nonlinear | +0.658% | 0.00063755714 | 0.00079781831 | No |
| P1 vs P0 linear | -0.002% | -0.04568082 | -0.044648142 | Yes |
| P1 vs P0plus linear | +0.002% | -0.045711701 | -0.044677446 | Yes |

The nonlinear criterion uses the paired difference L(P1)−0.97 L(reference); linear noninferiority uses L(P1)−1.05 L(reference). A criterion is established only if its one-sided upper Student-t bound is below zero. The four bounds use a Bonferroni family allocation of 0.05/4 and 23 degrees of freedom. This inference concerns the specified simulated laws and finite procedures, under the repetition-level t approximation. Relative gains are ratios of observed mean losses and are not ratio confidence intervals. [Primary results](analysis/primary_results.csv); [paired repetition values](analysis/primary_paired_repetitions.csv).

## Standalone corrections and condition variation

| Procedure | Nonlinear SMSE | Linear SMSE |
| --- | --- | --- |
| shared common plus bias | 0.031575479 | 0.91917995 |
| in sample KAHM | 0.027629411 | 0.92012998 |
| in sample local RBF | 0.028729268 | 0.9200081 |
| extra raw RBF residual | 0.030278075 | 0.92031855 |

| Law | Added noise | P0 | P1 | P0plus | Standalone KAHM |
| --- | --- | --- | --- | --- | --- |
| Linear AR | 0.0 | 0.91259187 | 0.91259187 | 0.91246593 | 0.91912895 |
| Linear AR | 0.1 | 0.9152224 | 0.91525148 | 0.91540717 | 0.92113101 |
| Nonlinear AR | 0.0 | 0.019450681 | 0.019462957 | 0.01945126 | 0.01953934 |
| Nonlinear AR | 0.1 | 0.031683697 | 0.031686986 | 0.031683697 | 0.031744412 |
| Van der Pol | 0.0 | 0.0040959269 | 0.0040120568 | 0.0040959269 | 0.0077574467 |
| Van der Pol | 0.1 | 0.053885607 | 0.052955168 | 0.053650789 | 0.052070142 |
| Duffing | 0.0 | 0.01102078 | 0.010873506 | 0.011008694 | 0.011180643 |
| Duffing | 0.1 | 0.04347362 | 0.043284288 | 0.043459755 | 0.04348448 |

These comparisons describe the fixed procedures and condition means. They are not additional superiority tests or a basis for selecting a favorable post hoc primary population. A useful correction to the common forecast need not be the best member of a stronger forecasting menu. [Task losses](analysis/task_losses.csv); [repetition losses](analysis/repetition_losses.csv).

## C selections, native dimensions and feature widths

**Nonlinear tasks.** Counts below are task selections; all policy choices precede D scoring.

| Selected family | P0 | P1 | P0plus |
| --- | --- | --- | --- |
| Cross-fitted local RBF correction | 4 | 2 | 3 |
| In-sample KAHM correction | 0 | 43 | 0 |
| In-sample local RBF correction | 42 | 21 | 42 |
| Extra raw-history RBF correction | 0 | 0 | 4 |
| Common geometric features | 9 | 5 | 8 |
| Histogram gradient boosting | 29 | 27 | 28 |
| Raw-history RBF | 6 | 5 | 6 |
| Bounded joint common/RBF | 16 | 14 | 16 |
| Unbounded joint common/RBF | 38 | 27 | 37 |

**Linear tasks.** Counts below are task selections; all policy choices precede D scoring.

| Selected family | P0 | P1 | P0plus |
| --- | --- | --- | --- |
| In-sample KAHM correction | 0 | 1 | 0 |
| Extra raw-history RBF correction | 0 | 0 | 2 |
| Common geometric features | 4 | 4 | 3 |
| Persistence | 16 | 16 | 16 |
| Histogram gradient boosting | 8 | 8 | 7 |
| Raw-history RBF | 13 | 12 | 13 |
| Bounded joint common/RBF | 2 | 2 | 2 |
| Unbounded joint common/RBF | 4 | 4 | 4 |
| Raw-history ridge | 1 | 1 | 1 |

**All B-selected KAHM fits:** 192 tasks. Requested-cap counts: 1: 44, 2: 71, 4: 45, 8: 24, 12: 8. Scaler counts: minmax: 93, standard: 65, tanh_z_1: 34. Realized local ranks range from 0 to 12 across 2688 selected-bank local-model occurrences.

**Tasks where C selected KAHM in P1:** 44 tasks. Requested-cap counts: 1: 16, 2: 17, 4: 9, 8: 1, 12: 1. Scaler counts: minmax: 29, standard: 9, tanh_z_1: 6. Realized local ranks range from 0 to 12 across 616 selected-bank local-model occurrences.

| Selected feature map | Minimum width | Mean width | Maximum width |
| --- | --- | --- | --- |
| KAHM feature width | 168 | 168 | 168 |
| local RBF feature width | 168 | 168 | 168 |
| extra RBF feature width | 66 | 75.84375 | 76 |

Requested caps and realized local ranks are different quantities. The width summaries describe the selected feature maps; they do not establish equal capacity. C-defined selected subsets are reported descriptively and do not change the primary claim family. [Selections](analysis/selection_frequencies.csv); [conditional summaries](analysis/conditional_descriptive.csv).

## External natural-series evidence

The separate natural evaluation contains 24 Weekly and 24 Hourly series from the M4 training archive, 32 reserved origins per series, and horizons 1, 4 and 12. It does not use official M4 Test files or produce an official M4 ranking. Frequency panels and series receive equal weights. The series are anonymous benchmark units, so no independent-source population test is reported. The natural P0 menu omits episode cross-fit RBF, because a single series supplies no independent episodes, and adds seasonal-naive and drift controls. Consequently the synthetic and natural policies are related adaptations, not identical populations or pooled experiments. [Natural protocol](../policy_augmentation_natural_20261101/PROTOCOL.md).

| Panel | P0 SMSE | P1 SMSE | P0plus SMSE | P1 gain vs P0 | P1 gain vs P0plus |
| --- | --- | --- | --- | --- | --- |
| Weekly | 0.079391977 | 0.083924635 | 0.091462265 | -5.709% | +8.241% |
| Hourly | 0.17583249 | 0.17709983 | 0.1768562 | -0.721% | -0.138% |
| Overall | 0.12761223 | 0.13051223 | 0.13415923 | -2.273% | +2.718% |

| Panel | P0 frequency MASE | P1 frequency MASE | P0plus frequency MASE |
| --- | --- | --- | --- |
| Weekly | 1.7619209 | 1.7729758 | 1.8117755 |
| Hourly | 0.99993945 | 1.0074535 | 0.99808869 |
| Overall | 1.3809302 | 1.3902147 | 1.4049321 |

Frequency MASE uses A-only lag 1 for Weekly and lag 24 for Hourly. The additional study-specific seasonal MASE uses lag 52 for Weekly and lag 24 for Hourly. These finite-horizon metrics differ from the official M4 evaluation. Denominator-floor activations: {"frequency_MASE": 0, "seasonal_MASE": 0}. [Natural metric tables](../policy_augmentation_natural_20261101/analysis/aggregates.csv).

C selected KAHM in P1 on 3 of 48 series (2 Weekly; 1 Hourly).

| C-defined subset | Comparator | Series | P1 SMSE | Reference SMSE | Relative gain | Wins / ties / losses |
| --- | --- | --- | --- | --- | --- | --- |
| KAHM selected by C | P0 | 3 | 0.24736201 | 0.20096202 | -23.089% | 1 / 0 / 2 |
| KAHM selected by C | P0plus | 3 | 0.24736201 | 0.29806857 | +17.012% | 2 / 0 / 1 |
| KAHM not selected by C | P0 | 45 | 0.12272225 | 0.12272225 | +0.000% | 0 / 45 / 0 |
| KAHM not selected by C | P0plus | 45 | 0.12272225 | 0.12323194 | +0.414% | 2 / 41 / 2 |

The natural overall, frequency and selected-subset contrasts are descriptive. Conditional errors do not estimate the causal effect of a selection rule, and a favorable benchmark comparison cannot convert a failed synthetic margin test into an established claim. [Natural contrasts](../policy_augmentation_natural_20261101/analysis/policy_contrasts.csv); [natural independent audit](../policy_augmentation_natural_20261101/audit/verification.json).

## Measured computation and numerical verification

| Measurement | Synthetic study | Natural study |
| --- | --- | --- |
| Sum of fit-worker wall seconds | 2723.648 | 382.67613 |
| Sum of fit-worker CPU seconds | 2718.917 | 381.13356 |
| Sum of prediction-worker wall seconds | 269.57619 | 10.935869 |
| Sum of extra-family fit wall seconds | 17.720132 | 4.200868 |
| Largest individual worker RSS (MiB) | 301.08594 | 238.59375 |
| Complete compressed fitted states (bytes) | 1.8968664e+09 | 1.959501e+08 |

Synthetic fit stage elapsed time: 998.61633 seconds, using the recorded execution-stage stopwatch.

Synthetic predict stage elapsed time: 257.71897 seconds, using the recorded execution-stage stopwatch.

Worker durations sum concurrent work and are not elapsed campaign times. A maximum individual-worker RSS is not total concurrent resident memory. No exact natural execution-stage stopwatch was retained. These measurements cover the complete research suite, including rejected and diagnostic candidates, and do not identify an isolated minimal P0/P1 deployment cost. The extra-family time excludes constructing the shared common predictor; matching 75 validation opportunities does not match runtime.

The synthetic audit passed 289544 assertions covering 192 tasks, 9600 saved forecast arrays and 5376 selected local bases. It independently recomputed every saved terminal loss and all four primary bounds; checked the new B and C selections, stage seals, exact policy forwarding and seed exclusions; and independently solved selected extra-RBF ridge coefficients. It does not independently reconstruct KAHM’s raw feature map, every legacy fit, or native PCA eigenspaces. Natural verification covered 48 series, 2,016 terminal arrays, 10,800 B alpha paths, 576 C multiplier choices, 144 policy choices and 576 selected ridge normal equations. Its coefficient checks reuse inherited feature maps. [Synthetic audit](verification/policy_audit.json); [natural audit](../policy_augmentation_natural_20261101/audit/verification.json).

The synthetic execution recorded one successful numerical recovery and no unrecovered failure. The natural execution recorded no numerical recovery or unrecovered failure. Successful numerical verification establishes the consistency of the preserved computation, not a universal forecasting advantage or a distribution-free risk certificate.

## Scope of the contribution

This experiment addresses the incremental value of admitting a KAHM correction to a strong finite selection menu, while retaining validation-selected native dimension and preprocessing. The observed synthetic policy benefit is small, neither declared nonlinear superiority margin is established, and the external panel does not improve on P0. Thus these results do not establish reliable general adoption of KAHM as a forecasting-menu component. They retain evidence of useful corrections to particular common forecasts and identify limits of transferring those gains into complete selection policies. Benefits against a common forecast, advantages within selected subsets, or favorable individual laws retain their narrower interpretations. The new repetitions are not pooled with the previous studies, and the natural cohort is not pooled with the synthetic repetition sample.
