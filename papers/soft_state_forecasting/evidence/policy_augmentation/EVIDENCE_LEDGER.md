# Policy-augmentation evidence ledger

## Provenance and current verification scope

The immutable scientific-source commit is `b1915813e98a59572a4bbb800adc2bd1f3fccc1b`; the exact results-release commit is `b59410407a7cd34155aa1955fe0c801f62381364`, on `research/kahm-policy-augmentation-2026-11-01` in `MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine`. Forty-one locally retrieved files match their complete Git blob identities, including the original CRLF bytes of CSV tables. See `RETRIEVAL_VERIFICATION.json`. All present analysis tables also match their recorded analysis-summary SHA-256 values.

The independent revision check recomputed all four paired primary bounds from `analysis/primary_paired_repetitions.csv` and all displayed natural policy and residual-family frequency aggregates from all 48 rows per method in `analysis/task_methods.csv`. It did not refit any model or replay the complete raw forecast archive. See `INDEPENDENT_RECOMPUTATION.json`. The original released execution's broader audits are separately preserved and must not be described as newly rerun here.

## Four primary simulation claims

The new cohort contains 24 independent repetition aggregates and 192 law/noise task suites. Horizons, origins, noise variants, task suites, and saved arrays are not additional independent replications. The nonlinear aggregate covers nonlinear AR, Van der Pol and Duffing, equally weighted with two noise levels and three horizons; the linear panel is linear AR. These new observations are not pooled with preceding studies.

The published CSV uses `d = L(P1) - factor*L(reference)` and an upper bound `mean(d)+t_(0.9875,23)*sd(d)/sqrt(24)`. A criterion passes only if that bound is strictly below zero. The protocol's opposite-sign lower-gap statement is algebraically equivalent; do not mix signs when drawing a table. The exact critical t is 2.3978750646571103.

| Criterion | P1 mean SMSE | Reference mean SMSE | Gain (%) | Mean margin difference | Simultaneous upper bound | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Nonlinear vs P0, 3% | 0.02704582691299484 | 0.02726838512983061 | 0.81617674011909 | 0.0005954933370591497 | 0.0007114120695966539 | No |
| Nonlinear vs P0plus, 3% | 0.02704582691299484 | 0.027225020380295603 | 0.6581940611895876 | 0.0006375571441081053 | 0.0007978183064349266 | No |
| Linear vs P0, 5% noninferiority | 0.9139216740018812 | 0.9139071372975609 | -0.0015906106569207878 | -0.04568082016055792 | -0.04464814200433116 | Yes |
| Linear vs P0plus, 5% noninferiority | 0.9139216740018812 | 0.9139365474760259 | 0.0016274077435540768 | -0.045711700847946256 | -0.04467744619125768 | Yes |

These are approximate paired-t inferences with a four-claim Bonferroni family at alpha 0.05. Gains are ratios of mean losses, not ratio confidence intervals. Failure of the 3% margin is not evidence of no improvement or equivalence. Ordinary unadjusted zero-margin difference intervals are favorable to P1, but are not the frozen primary criteria and need not be promoted or reported.

Primary numerical sources are `extensions/policy_augmentation_20261101/analysis/{primary_results.csv,primary_paired_repetitions.csv,panel_summary.csv}`. The retained released audit's four decisions agree with the current recomputation.

## Standalone correction and policy selection

Nonlinear standalone calibrated in-sample KAHM has SMSE 0.027629410587209153; its identical common-plus-shared-bias baseline has 0.03157547934586666; independently selected local in-sample RBF has 0.02872926765429196. The descriptive gains are 12.4972568601% and 3.8283505181%. The standalone KAHM procedure remains 1.3239708023% worse than complete P0. This is a mechanism comparison, not an alternative primary endpoint.

P1 selects KAHM on 44 of 192 tasks: 43/144 nonlinear and 1/48 linear. P0plus selects the extra RBF on 6 tasks: 4 nonlinear, 2 linear. Exact selections are in `analysis/selection_by_task.csv` and the independent released audit. The KAHM-selected subsets remain descriptive. Derived policy law/noise means are in `DERIVED_policy_law_noise.csv` and are reproducible directly from `selection_by_task.csv`.

The published `RESULTS.md` reports all-192 KAHM requested-cap counts {1:44,2:71,4:45,8:24,12:8}; scalers {standard:65,minmax:93,tanh:34}; realized local ranks range 0–12 over 2688 bank occurrences. Among the 44 KAHM policy selections, caps are {1:16,2:17,4:9,8:1,12:1}; scalers {standard:9,minmax:29,tanh:6}; realized ranks range 0–12 over 616 occurrences. These are published reported summaries; their raw fit metadata are in external numerical archives and were not independently reconstructed in this revision check. Occurrences are not independent observations.

## Natural panel

| Panel | P0 SMSE | P1 SMSE | P0plus SMSE | P1 gain vs P0 (%) | P1 gain vs P0plus (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weekly, n=24 | 0.07939197699798133 | 0.08392463500032525 | 0.09146226499271466 | -5.7092 | 8.2412 |
| Hourly, n=24 | 0.17583248929069054 | 0.17709982946804217 | 0.17685620251831743 | -0.7208 | -0.1378 |
| Equal-frequency aggregate | 0.12761223314433595 | 0.13051223223418373 | 0.13415923375551606 | -2.2725 | 2.7184 |

Natural statistics are descriptive. This panel is 24 Weekly plus 24 Hourly M4 Train-series identities, not an official M4 competition result and not independently sampled source domains. No CI, population bootstrap, p-value, or independent-series inference is warranted under the frozen protocol. New identities do not establish independence of anonymous source processes. Official Test files are excluded.

All 48 series are retained. KAHM is chosen on W160, W196 and H368; P1 equals P0 exactly on the other 45. W160 improves 13.6743121549%; W196 worsens 20.0631385459%; H368 worsens 54.7748623898%. Their P1/P0 maximum absolute-error ratios are respectively 0.7800324891, 1.0808586014, 0.9522318611. This supports a distinction between forecast loss and numerical blow-up; these diagnostics do not establish forecast accuracy. The selected subset's combined SMSE is 23.0889% higher than P0, but remains descriptive.

Standalone natural KAHM SMSE 0.15195625354893003 is 3.3250041193% lower than the identical shared-bias baseline (0.157182580836565) and 2.1507038929% lower than independently selected RBF (0.15529621529683504). P0 remains substantially better than this standalone correction.

Secondary natural seasonal MASE: P0 0.6260662464078396, P1 0.630945938990474, P0plus 0.6295275368343379. Frequency-lag MASE: P0 1.3809301905140536, P1 1.39021467220348, P0plus 1.4049320902870275. Weekly seasonal lag52 is an explicitly declared annual-lag control; frequency-lag scaling uses lag1 Weekly and lag24 Hourly. No denominator floor was activated. These selected-offset repeated-origin scores are not official M4 rankings.

Numerical sources: natural `analysis/{task_methods.csv,aggregates.csv,policy_contrasts.csv,policy_selections.csv,selected_subset.csv,report_metrics.json,analysis_summary.json}`. Full roster, lengths and selected identities are in `roster.json`; the derived 48-row combined presentation table is `DERIVED_natural_roster_policy.csv`.

## Method details that must persist

P0 is the unchanged ten-family simulation menu from the preceding residual study. P1 appends the centered in-sample KAHM family; P0plus appends an extra centered raw-history/Nystrom RBF residual family. These additions correct the same full-A common predictor and B-fitted shared bias, with no second residual response-mean intercept. B independently selects each horizon's ridge penalty and one cap/scaler (or bandwidth/scaler) jointly. C selects per-horizon multipliers in {0,0.25,0.5,1}, then one whole family across horizons. Zero recovers the identical common-plus-bias forecast. No D selection or post-C refit occurs.

KAHM caps {1,2,4,8,12} cross standardization/minmax/tanh(z); native rank initialization is min(cap,12,N_local-1), followed by existing variance/range reductions. Cap20 equals cap12 at ambient dimension12; cap does not mean realized dimension. The extra RBF uses three scalers, five bandwidth factors and64 landmarks. Both additions have15 feature configurations,75 B scores per horizon,4 C multiplier choices and one C menu slot. Counts do not equate widths, functions, capacity or compute. Selected KAHM/local-RBF feature width is168; extra RBF width ranges66–76, mean75.84375.

Synthetic tasks use four laws, two noise levels,24 repetitions; history12, horizons1/4/12. A/B/C/D episodes64/32/32/128 with A/B/C row caps512/256/256; D retains13 origins per episode. Full-A target population variances standardize horizon MSE with floor1e-12. The1272 intended-distinct streams passed comparisons against each other and5989 previous seed exclusions.

Natural P0 differs: no episode cross-fit RBF, plus seasonal-naive and drift controls, giving11 base families and12 in each augmented policy. The prior72 M4 identities are excluded. Eligibility uses only ID and length N>=475. SHA256 ordering of a declared salted Frequency|ID string picks24 per frequency. A=min(512,N-219), minimum256; B64,C64,D32 origins;12 purged origins at each boundary. D origins are the last32 with all three targets available in Train. Models stay fixed but later D histories may use earlier D observations as they become available. No future observation is used by a query.

## Source freeze and audit qualifications

Synthetic local freeze was2026-11-01T19:39:16.589238Z before A/B/C/D generation; the source was publicly committed at19:43:05Z after ABC preparation, before D generation. Fits sealed19:56:29Z; predictions20:01:53Z; targets released20:02:13Z; analysis20:02:33Z; audit20:03:51Z. Natural local freeze19:43:22Z preceded numeric parsing; fits19:46:02Z and predictions19:47:01Z were sealed; analysis19:47:19Z and audit19:47:42Z preceded public combined source publication19:54:34Z. Neither study is independently attested public preregistration or cryptographic access isolation.

The released synthetic audit passed289544 assertions,192 tasks,9600 terminal arrays and5376 selected local bases, including all primary bounds, exact forwarding, new B/C selections and selected extra-RBF normal equations. It does not independently implement all KAHM raw maps, legacy fits or native PCA eigenspaces. It recorded one successful solver recovery and no unrecovered failure. The natural audit covered48 series,2016 terminal arrays,10800 B alpha paths,576 C multiplier choices,144 policies and576 selected normal equations; it reuses inherited feature maps and does not refit every rejected candidate. It recorded no fallback or failed task.

Resource sources preserve entire research-suite cost, including rejected and diagnostic fits. Sums of worker elapsed times are not campaign wall times; maximum per-worker RSS is not total concurrent memory. Synthetic sums: fit wall2723.648s, CPU2718.917s; max RSS301.08594MiB. Natural fit wall382.6761252s, CPU381.133556658s, max RSS238.59375MiB. Do not infer equal cost from matched candidate counts.

## Defensible claim boundary

The results preserve useful geometry-dependent, bounded and calibrated correction of a retained predictor. They fail to establish the declared3% incremental policy advantage in fresh simulations, and the declared natural panel worsens relative to P0. The supported manuscript position is conditional forecasting value, with explicit limits on turning that value into a generally improved selection policy. Do not infer that KAHM has no useful forecasting applications. If a selector is redesigned from these terminal outcomes, new reserved evaluation data are necessary; these same D outcomes cannot become fresh confirmation.
