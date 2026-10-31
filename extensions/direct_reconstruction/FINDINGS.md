# Findings: forecast-trained KAHM reconstruction

**Direct reconstruction training substantially improves KAHM over its own gate-only version on Van der Pol. The experiment does not establish distinctive predictive superiority over the matched alternatives, or an accuracy–cost advantage.** The study specifies and evaluates a KAHM forecasting procedure, with every planned task accounted for. The results follow the analysis fixed before D evaluation. [Protocol](PROTOCOL.md) · [Machine-readable findings](results/summary.json)

## Execution, coverage and verification

The code, analysis, exact roster and prepared inputs were publicly frozen in [commit 39174d06dffc7e28fed73dbbf866e539560dd021](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/39174d06dffc7e28fed73dbbf866e539560dd021) before execution. This is a prospective execution checkpoint informed by earlier studies, not external registration or independent experimental replication. [Publication record](PUBLICATION.json) · [Freeze](results/freeze.json)

| Item | Result |
|---|---:|
| Planned tasks | 46 |
| Synthetic tasks | 36: twelve repetitions for each of three known laws |
| Planned natural tasks | 10 series from four collections |
| Successfully fitted task suites | 43 |
| Procedures per successful task | 17 |
| Available task–method prediction records | 731 of 782 planned |
| Anticipated unavailable tasks | 3 rideshare series, accounting for 51 unavailable records |
| Additional task failures | 0 |
| Invalid geometry-objective evaluations | 0 |
| Recorded numerical forecast fallbacks | 0 |

The three A-selected rideshare series have no complete B windows under the declared chronological and missing-window rules. This was documented before execution. The series were retained without replacement, and no forecaster was fitted for those tasks. This is a feasibility failure of the declared data protocol, distinct from numerical failure of a fitted geometry. [Prepared-source manifest](data/sources_manifest.json) · [Execution record](results/execution.json)

The frozen audit passed **543,017 checks**, including regeneration of 144 synthetic episode batches, reconstruction of histories and targets, temporal availability and A-only scales, recorded selection and optimization limits, cost dependencies, and replay of all 731 selected states on C and D—**1,462 forecast matrices**. A separately written calculation, importing neither the model nor analyzer, independently reproduced all task losses, synthetic and available-natural means, and all 18 primary intervals. Engineering validation comprises 18 model tests and 13 protocol tests, plus the preserved development smoke and input checks. These checks establish consistency and replayability; they do not imply predictive superiority or validate a distributional assumption. [Audit](results/audit_report.json) · [Validation record](VALIDATION.json) · [Independent numerical check](validation/independent_check.json)

## All seventeen procedures

Synthetic entries are arithmetic mean D MSE over twelve repetitions per law. Each repetition averages equally over its independent D episodes, their origins, and the three horizons. Natural entries are A-scaled MSE, averaged equally over available series within each available collection, then equally across the **three available collections**. The last column is a supplemental description of seven series; **the planned four-collection aggregate is undefined for every procedure**. Values below are rounded; the CSV and JSON retain full numerical precision. [Method aggregates](results/method_aggregates.csv) · [Task losses](results/task_metrics.csv) · [Horizon losses](results/horizon_metrics.csv)

| Procedure | Nonlinear AR MSE | Van der Pol MSE | Duffing MSE | Available-natural scaled MSE |
|---|---:|---:|---:|---:|
| Ridge | 0.00530404 | 0.04516608 | 0.00721197 | 0.38562529 |
| Raw RBF | 0.00495085 | 0.00648233 | 0.00439066 | 0.36300731 |
| Raw HGB | 0.00505898 | 0.00242515 | 0.00776382 | 0.35220070 |
| Seasonal naive | 0.00882621 | 0.29403780 | 0.04053792 | 0.46013446 |
| Fixed KAHM | 0.00482884 | 0.02232430 | 0.00614177 | 0.39406974 |
| Fixed affine | 0.00485237 | 0.01870234 | 0.00539038 | 0.38000618 |
| Fixed locality | 0.00488788 | 0.01513302 | 0.00520599 | 0.37626349 |
| Fixed RBF-local | 0.00479772 | 0.01758088 | 0.00588515 | 0.38619966 |
| Gate KAHM | 0.00481296 | 0.01455396 | 0.00567082 | 0.39619189 |
| Gate affine | 0.00480221 | 0.01061506 | 0.00501844 | 0.38952069 |
| Gate locality | 0.00486175 | 0.01371246 | 0.00454065 | 0.38233291 |
| Gate RBF-local | 0.00475221 | 0.01113440 | 0.00599653 | 0.36778754 |
| **Reconstruction KAHM** | **0.00480874** | **0.00852492** | **0.00535835** | **0.37193548** |
| Reconstruction affine | 0.00480340 | 0.00754486 | 0.00471091 | 0.38172205 |
| Reconstruction locality | 0.00486247 | 0.00929219 | 0.00439836 | 0.37482890 |
| Reconstruction RBF-local | 0.00480008 | 0.00779752 | 0.00502042 | 0.36383455 |
| C-selected non-KAHM policy | 0.00476659 | 0.00242515 | 0.00414346 | 0.38247198 |

“Reconstruction” identifies the KAHM bandwidth intervention and the specified matched shape intervention for each control. The raw RBF procedure retains standardized histories alongside Nyström features. HGB denotes histogram gradient boosting. The policy selects one complete non-KAHM procedure on C and pays for the union of its candidate computations; it is not a D-selected oracle. [Method definitions](PROTOCOL.md#complete-method-roster)

Relative to fixed KAHM, reconstruction training reduces mean synthetic MSE by **0.42%, 61.81%, and 12.76%** on AR, Van der Pol, and Duffing. Its reductions relative to gate-only KAHM are **0.09%, 41.43%, and 5.51%**. Only the Van der Pol gate comparison excludes zero under the primary simultaneous analysis. Fixed-KAHM comparisons are descriptive and outside that family. Gains over ridge are **9.34%, 81.13%, and 25.70%**; they do not identify the KAHM reconstruction as the unique source of improvement. [Aggregates](results/method_aggregates.csv) · [Primary contrasts](results/primary_contrasts.csv)

## The eighteen predeclared comparisons

Every row compares **reconstruction KAHM** with the named comparator. The paired difference is comparator MSE minus KAHM MSE, so positive values favor KAHM. Gain is `100 × (mean comparator MSE − mean KAHM MSE) / mean comparator MSE`. Intervals concern **absolute MSE differences**, not percentages.

The family uses twelve independent repetition means per law, a paired-difference normal approximation, 11 degrees of freedom, family alpha 0.05, and a two-sided Bonferroni adjustment for 18 comparisons. The critical t value is **3.83345205638**. This family gives **two positive, six negative, and ten unresolved intervals**. Coverage does not extend to every historical or descriptive contrast in the manuscript. An interval including zero does not establish equivalence. [Frozen analysis specification](results/protocol.json) · [Full-precision interval table](results/primary_contrasts.csv)

| Law | Comparator | Gain | Adjusted absolute-difference interval | Outcome |
|---|---|---:|---:|---|
| Nonlinear AR | Raw RBF | +2.8706% | [6.652851499e-06, 0.0002775821667] | **Positive** |
| Nonlinear AR | Gate KAHM | +0.0879% | [-3.861677504e-05, 4.707618719e-05] | Unresolved |
| Nonlinear AR | Reconstruction affine | -0.1110% | [-0.0001004036635, 8.973749425e-05] | Unresolved |
| Nonlinear AR | Reconstruction locality | +1.1051% | [-3.457138980e-05, 0.0001420393919] | Unresolved |
| Nonlinear AR | Reconstruction RBF-local | -0.1803% | [-9.496293209e-05, 7.765344890e-05] | Unresolved |
| Nonlinear AR | Non-KAHM policy | -0.8842% | [-0.0001469233923, 6.263228950e-05] | Unresolved |
| Van der Pol | Raw RBF | -31.5101% | [-0.004196945925, 0.0001117629441] | Unresolved |
| Van der Pol | Gate KAHM | +41.4254% | [0.004349467338, 0.007708602851] | **Positive** |
| Van der Pol | Reconstruction affine | -12.9898% | [-0.001805136095, -0.0001549848868] | **Negative** |
| Van der Pol | Reconstruction locality | +8.2571% | [-0.001213947555, 0.002748478606] | Unresolved |
| Van der Pol | Reconstruction RBF-local | -9.3286% | [-0.002058427415, 0.0006036236845] | Unresolved |
| Van der Pol | Non-KAHM policy | -251.5221% | [-0.007899302944, -0.004300247488] | **Negative** |
| Duffing | Raw RBF | -22.0398% | [-0.001455022492, -0.0004803575593] | **Negative** |
| Duffing | Gate KAHM | +5.5102% | [-5.752882573e-05, 0.0006824800180] | Unresolved |
| Duffing | Reconstruction affine | -13.7434% | [-0.0009921474966, -0.0003027322205] | **Negative** |
| Duffing | Reconstruction locality | -21.8260% | [-0.001603431637, -0.0003165393780] | **Negative** |
| Duffing | Reconstruction RBF-local | -6.7310% | [-0.001086296722, 0.0004104507966] | Unresolved |
| Duffing | Non-KAHM policy | -29.3207% | [-0.001588953958, -0.0008408280119] | **Negative** |

The two favorable intervals concern raw RBF on AR and KAHM's own gate-only version on Van der Pol. **No positive primary interval is against an equally shape-trained non-KAHM local control.** The six adverse comparisons include three matched-control defeats: affine on Van der Pol, and affine and locality on Duffing. The non-KAHM policy selects HGB in all twelve Van der Pol repetitions; KAHM's mean MSE there is 3.52 times the policy's. No law resolves all six comparisons in KAHM's favor. [Primary results and paired differences](results/summary.json)

## Available natural evidence and sensitivity

The planned natural roster contains Dominick, rideshare prices, Wikipedia, and Sunspot. All three rideshare tasks are unavailable. The following values describe the other seven series and **do not replace the undefined four-collection estimand**. The collections were chosen purposively, so these summaries have no population confidence interval. [Collection metrics, including unavailable entries](results/natural_collection_metrics.csv)

| Collection | Series | Ridge | Raw RBF | HGB | Reconstruction KAHM | Reconstruction RBF-local | Non-KAHM policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dominick weekly | 3 | 0.357402 | 0.348741 | 0.274227 | 0.316351 | 0.333562 | 0.356277 |
| Rideshare hourly | 3 planned | undefined | undefined | undefined | undefined | undefined | undefined |
| Wikipedia daily | 3 | 0.459317 | 0.519051 | 0.497020 | 0.459299 | 0.470223 | 0.510632 |
| Sunspot daily | 1 | 0.340157 | 0.221230 | 0.285355 | 0.340157 | 0.287719 | 0.280508 |

On the available three-collection aggregate, KAHM improves ridge by **3.55%**, fixed KAHM by **5.62%**, gate KAHM by **6.12%**, and the C-selected policy by **2.75%**. Its error is **2.46% higher than raw RBF**, **5.60% higher than HGB**, and **2.23% higher than reconstruction RBF-local**. [Available-case comparisons](results/natural_available_comparisons.csv)

The ridge improvement is concentrated in Dominick. Leaving Dominick out gives only **0.00229%** improvement over ridge on the other two collections. Relative to raw RBF, the leave-one-available-collection-out gain ranges from **−15.18% to +10.62%**, changing sign. Relative to the policy, it ranges from **−3.10% to +10.53%**, also changing sign. Relative to HGB, every such comparison remains adverse, with gains between **−17.32% and −0.57%**. These are sensitivity descriptions, not additional confirmatory tests. [Complete leave-one-collection-out table](results/natural_available_leave_one_collection_out.csv)

C selects zero KAHM correction for **12 of the 21 available natural horizons**, including all three Sunspot horizons. The Sunspot KAHM forecast therefore equals the retained ridge baseline exactly. A feasible zero correction is useful protection but cannot guarantee an improvement on D. These selection records are preserved in the task diagnostics; see the [evidence unpacking instructions](README.md).

## Optimization, cost and retained state

The completed tasks retain **3,096 geometry optimizations** and **406,832 actual objective evaluations**. There are **2,491 convergence exits**, **604 evaluation-budget exits**, and **one abnormal L-BFGS-B stop**. All retain their best finite incumbent. There are no invalid objective evaluations and no recorded numerical forecast fallbacks. A completed forecast is not synonymous with optimizer convergence. [Optimization summary](results/optimization_summary.csv) · [Fallback records](results/fallback_events.json)

The abnormal stop is `synthetic__vanderpol__84200`, `gate_locality`, horizon 12, alpha 0.01. Its exact status is `"2"`, and its message is `"ABNORMAL: "`, including a trailing space. It performs 189 evaluations and 10 iterations, records `converged=false`, `valid=true`, and no exceptions. Its retained normalized B loss is 0.06133804336506434, versus initial loss 0.06447892337787998. The stored message alone does not identify a more specific cause.

The next table gives arithmetic means over the 36 completed synthetic tasks or seven completed natural tasks. These cost averages are per task; the natural accuracy aggregate above instead weights collections equally. Charged pipeline time includes required training, selection, query, state-packing and selected-state serialization nodes, with shared work charged once per procedure. The policy pays for all required candidate dependencies. Downloads, preparation, process imports, and evidence-file writes are outside that measure. [Full cost table](results/cost_aggregates.csv) · [Cost definitions](PROTOCOL.md#computation-state-and-failure-accounting)

| Procedure | Synthetic time (s) | Available-natural time (s) | Synthetic state (KiB) | Available-natural state (KiB) |
|---|---:|---:|---:|---:|
| Ridge | 0.002035 | 0.001730 | 0.769 | 1.795 |
| Raw RBF | 0.017153 | 0.012981 | 42.265 | 57.876 |
| HGB | 0.653383 | 0.654977 | 386.674 | 383.325 |
| Fixed KAHM | 0.077426 | 0.050204 | 174.454 | 124.761 |
| Gate KAHM | 0.913916 | 0.733962 | 174.420 | 132.949 |
| Reconstruction KAHM | 10.725322 | 7.928688 | 174.488 | 104.200 |
| Reconstruction affine | 6.972800 | 5.345340 | 10.073 | 18.607 |
| Reconstruction locality | 2.721628 | 2.132935 | 10.008 | 16.329 |
| Reconstruction RBF-local | 2.765905 | 2.183987 | 10.042 | 17.743 |
| Non-KAHM policy | 15.574699 | 12.155317 | 155.879 | 17.486 |

State size is the uncompressed serialized selected model, not process resident memory, an intrinsic lower bound, or the training working set. Selecting zero correction for every horizon can omit a geometric bank. The whole run's cumulative process RSS high-water mark is 200,680 KiB; it cannot be allocated to individual procedures. Sum of measured `fit_suite` wall times is 1,123.08 seconds. These measurements are preserved in the per-task diagnostics after unpacking. No deployment latency limit, memory ceiling, or accuracy–cost exchange rate was imposed, and streaming updates were not measured. [Execution record](results/execution.json) · [Resource scope](PROTOCOL.md)

## What this study establishes

The study supplies a concrete KAHM time-series forecasting construction, matched forecast supervision, chronological and episode protocols, complete failure accounting, replayable selected states, and a reproducible within-family training benefit on Van der Pol. It also narrows the open question: a KAHM-specific contribution must exceed equally supervised local alternatives and strong practical baselines on newly reserved evidence, with a declared useful resource tradeoff. The present results do not establish that broader claim. Incremental geometry updates were not evaluated in this study.
