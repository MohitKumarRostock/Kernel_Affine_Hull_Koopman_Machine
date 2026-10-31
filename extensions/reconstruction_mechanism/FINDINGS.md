# Completed reconstruction-mechanism study

## Result and decision

**The bounded KAHM reconstruction departure improves the common gate-and-coordinate decoder, but the study does not establish the required KAHM-specific forecasting advantage. The frozen decision is `STOP_FORECASTING_PERFORMANCE_BRANCH`.**

The full experiment completed on 31 October 2026: 192 task suites, ten procedures, 1,920 saved task–procedure prediction arrays, and 9,584,640 scored forecast values. The inferential sample consists of 24 independent repetition aggregates. No task was replaced and no numerical amendment was needed.

| Comparison | KAHM point gain | Simultaneous lower bound on margin gap | Required condition |
| --- | ---: | ---: | --- |
| Common gate and coordinate basis | +12.5823% | 0.0029546109 | Passed |
| Matched bounded RBF departure | +1.1693% | -0.0007861970 | Not established |
| Raw history plus RBF | +3.9157% | -0.0000828607 | Not established |
| Histogram gradient boosting | -1.7693% | -0.0019363039 | Not established |
| C-selected non-KAHM policy | -6.0394% | -0.0029899472 | Not established |
| Ridge on the linear control | -0.1990% | 0.0416102023 | Passed |

The first five rows average three nonlinear laws and two added-noise levels equally within each repetition. The reported point gain is `1 − mean_loss(KAHM)/mean_loss(comparator)`. Positive values favor KAHM. The required contrast is `0.97*loss(comparator) − loss(KAHM)`, so the one-sided lower bound must exceed zero to establish the chosen 3% useful margin. The final row is a separate linear-control comparison, using `1.05*loss(ridge) − loss(KAHM)` and a 5% noninferiority tolerance. All six lower bounds use a Bonferroni-adjusted paired t calculation with 23 degrees of freedom; they are not confidence bounds on the percentage gain.

Only the common-basis useful-superiority condition and linear-control noninferiority condition pass. The matched RBF point gain is below 3%. The raw-RBF point gain is above 3%, but its simultaneous margin bound remains below zero; this is unresolved useful-margin evidence, not a demonstrated raw-RBF victory. KAHM has 1.77% higher aggregate loss than boosted trees and 6.04% higher loss than the C-selected policy.

## What the mechanism study established

The completed algebra gives both redundancy and distinctness results. Direct normalized KAHM target transfer is exactly a ratio of two dual kernel-ridge predictions. A symmetric two-reference bank can give the same augmented raw feature span as matched RBF. An irregular three-reference bank, under an explicit positivity condition, instead proves strict raw-span distinctness. The main empirical comparison concerns the complete reconstruction, separate block normalization, radial map and regularized decoder; raw-span identities alone do not determine fitted predictions.

The bounded KAHM departure improves the common basis by 12.58% under the declared useful-effect test. Thus the result does not say that reconstruction features are useless. It says that the added KAHM procedure did not establish enough advantage over its matched reconstruction and competitive forecasting controls to justify extending this performance branch. Every nonlinear KAHM correction was active (432/432 task/horizon cells), so universal baseline recovery does not explain the outcome.

## Stability and cost

All 192 tasks pass the fixed stability screens. The largest KAHM-to-ridge ratio of taskwise maximum absolute forecast errors is 1.5534, below the chosen engineering threshold of five. No feature-, coefficient- or correction-bound violations, local KAHM/RBF fallback rows, whole-method failures or final prediction-cell recoveries occur. The linear control activates KAHM corrections in 54/144 horizon cells and has a 0.1990% higher mean loss than ridge, within the declared noninferiority tolerance.

For mean-SSE ridge with a free intercept, the centered coefficient norm is bounded by `RMS(centered residual)/(2*sqrt(alpha))`. Separate radial maps give a uniform bound on correction magnitude. This controls the correction only: it does not bound the raw baseline, guarantee forecasting accuracy, or prove stability of online geometry changes.

On the recorded host, mean charged fit/selection time is 0.0833 s for bounded KAHM and 0.0658 s for matched bounded RBF. Prediction of all 1,664 D histories takes 29.245 ms and 25.758 ms respectively; these are batch costs, not per-forecast latency. Mean selected-state serialization is 118.81 KiB and 55.32 KiB. Shared computation is charged once to each method that needs it; policy fitting pays its full candidate union. Whole-process RSS, retained arrays, serialized bytes and per-method costs have distinct scopes. No application-specific resource ceiling was imposed.

## Reproducibility and scope

The local checkpoint `6d666b0fa3d8e0ee90a357ea7970def94e9fdda7` froze 239 source/configuration/input files before reserved forecasts. Freeze SHA-256: `22d4c492ef7a591eb1f21eca5ff458a3f48d102fdc3cd63de597e5fd7b351bda`. This was a local prospective freeze, not public preregistration. The published result reports the completed study and preserves the protocol as audit evidence.

Selected-state replay reproduces all 1,920 saved prediction arrays exactly on the recorded environment. The internal audit independently reconstructs all 516,096 A/B/C/D windows, 4,032 selected ridge coefficient vectors, 9,408 selection decisions and all six primary statistics. Its maximum independent forecast discrepancy is 1.0521e-10; the six statistical contrasts agree exactly. The audit uses its own feature evaluation, ridge forward passes and tree traversal and does not refit the forecasters. Discarded B candidate coefficients were not saved, so their recorded selection tables are checked but their losses are not independently refitted. Saved nonlinear trajectories, partitions and principal subspaces are not independently regenerated. These are computational checks, not external scientific replication.

The nonlinear laws were known during development. New independent seeds are new realizations, not new applications. Two added-noise levels are paired; oscillator observations at zero added noise retain their inherited sensor noise. A random A-estimated noise amplitude also prevents treating all augmented linear-control realizations as one unconditional fixed-variance Gaussian law. The fixed methods, sample sizes, representation width, decoder grids, noise levels and horizons limit the conclusion. The chosen margins are research-allocation screens, and the paired t analysis is approximate. A failed finite procedure is not an impossibility theorem for every geometric representation or decoder.

No additional application or adaptive experiment was undertaken after this result. This implements the declared stopping rule. The research contribution is the conditions connecting reconstruction, decoder access, stability and predictive utility, with the complete positive and negative evidence preserved.

## Evidence entry points

- [Frozen decision and six primary contrasts](analysis/decision.json)
- [All condition means](analysis/condition_metrics.csv)
- [Twenty-four repetition aggregates](analysis/replicate_metrics.csv)
- [Independent audit](results/audit.json)
- [Exact selected-state replay](results/replay.json)
- [Algebra and proofs](ALGEBRA.md)
- [Reproduction commands](README.md)
