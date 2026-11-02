# Reconstruction departures: experimental findings

## Design and primary result

The experiment compares ten forecasting procedures in 192 task suites: 24 independent repetitions of four simulation laws at two added observation-noise levels. The saved evidence contains 1,920 task–procedure prediction arrays and 9,584,640 scored forecast values. Every planned task was evaluated, and no numerical amendment was required. Statistical inference uses the 24 independent repetition aggregates.

The bounded KAHM reconstruction departure improves the common gate-and-coordinate predictor. The study does not establish the prespecified improvement over all matched and competitive forecasting controls.

| Comparator | KAHM relative error reduction | Simultaneous lower bound on the margin contrast | Criterion satisfied |
| --- | ---: | ---: | --- |
| Common gate and coordinate basis | 12.5823% | 0.0029546109 | Yes |
| Matched bounded RBF departure | 1.1693% | -0.0007861970 | No |
| Raw history plus RBF | 3.9157% | -0.0000828607 | No |
| Histogram gradient boosting | -1.7693% | -0.0019363039 | No |
| Validation-selected non-KAHM procedure | -6.0394% | -0.0029899472 | No |
| Ridge on the linear control | -0.1990% | 0.0416102023 | Yes |

Here RBF denotes radial-basis-function features. Positive relative error reductions favor KAHM. For a comparator with aggregate loss $L_c$ and KAHM aggregate loss $L_K$, the percentage is $100(1-L_K/L_c)$. Each nonlinear repetition aggregate weights three laws and two added-noise levels equally; horizons are also averaged equally after normalization by the A-partition target variance.

The first five comparisons test a 3% improvement through the repetition-level contrast $0.97L_c-L_K$. The linear comparison instead tests noninferiority within 5% of ridge through $1.05L_{\mathrm{ridge}}-L_K$. A criterion is satisfied when the corresponding one-sided lower confidence bound is positive. All six bounds use a Bonferroni-adjusted paired Student t calculation with 23 degrees of freedom. The tabulated bounds are in normalized-loss units, not percentage points.

Only the common-basis improvement and linear-control noninferiority criteria are satisfied. The 3.92% point gain over raw RBF exceeds the target margin, but its adjusted lower bound is negative. This leaves the required improvement unresolved; it does not establish equality or superiority of raw RBF. The negative gains against boosting and the selected non-KAHM procedure mean that KAHM has higher observed aggregate loss in those comparisons.

## Interpretation of the reconstruction contribution

The mathematical analysis identifies both redundancy and distinctness. Direct transfer of normalized KAHM weights to reference targets equals a ratio of two dual kernel-ridge predictions. A symmetric reference set can produce the same augmented raw feature span as matched RBF reconstruction. An irregular three-reference set satisfying the stated positivity condition produces a KAHM departure outside that span.

The fitted forecasting procedures also include coordinate standardization, separate radial transforms, ridge regularization, and validation-based selection. These operations matter: equality of raw feature spans does not imply identical predictions after regularization or nonlinear normalization. The empirical comparison therefore concerns the complete procedures defined in the protocol.

The 12.58% improvement over the common basis demonstrates useful additional forecasting functions in this setting. It does not establish that KAHM supplies those functions more effectively than the matched and practical alternatives. All 432 nonlinear task–horizon KAHM corrections are active, so the observed comparison is not explained by universal selection of the uncorrected baseline.

## Stability and computational cost

All 192 tasks satisfy the specified numerical stability checks. No violation of a feature, coefficient, or correction bound is recorded. There are no local KAHM/RBF reconstruction fallbacks, whole-method failures, or final prediction-cell recoveries. The largest taskwise ratio of maximum absolute forecast error for KAHM relative to ridge is 1.5534, below the specified threshold of five.

On the linear control, KAHM corrections are active in 54 of 144 horizon cells. Mean loss is 0.1990% higher than ridge, which satisfies the 5% noninferiority criterion.

The correction bound follows from the bounded feature construction and the mean-squared-error ridge objective with an unpenalized intercept. If $\alpha>0$ is the coefficient penalty and $e_c$ is the vector of centered training residuals, the coefficient norm is at most $\operatorname{RMS}(e_c)/(2\sqrt{\alpha})$. The [mathematical guide](MATHEMATICAL_BACKGROUND.md) defines the quantities and derives the resulting uniform correction bound. It applies to the correction, not to the baseline or the forecasting error.

| Mean resource measurement | Bounded KAHM | Matched bounded RBF |
| --- | ---: | ---: |
| Charged fitting and selection time | 0.0833 s | 0.0658 s |
| Prediction time for 1,664 held-out histories at all three horizons | 29.245 ms | 25.758 ms |
| Serialized selected model state | 118.81 KiB | 55.32 KiB |

These measurements average all 192 tasks on the recorded host. Prediction times are batch costs. Shared computation is charged once to each procedure that requires it, and the selected non-KAHM procedure is charged for the union of its candidate fits. Serialized bytes, retained arrays, and whole-process resident memory measure different aspects of resource use. No application-specific resource ceiling was imposed.

## Reproducibility

The prospective local checkpoint `6d666b0fa3d8e0ee90a357ea7970def94e9fdda7` fixed 239 source, configuration, and input files before held-out forecasts were computed. The SHA-256 digest of `freeze.json` is `22d4c492ef7a591eb1f21eca5ff458a3f48d102fdc3cd63de597e5fd7b351bda`. This records a local prospective freeze, not a public preregistration.

Selected-state replay reproduces all 1,920 saved prediction arrays exactly in the recorded environment. The independent-code audit reconstructs 516,096 A/B/C/D windows, checks 4,032 selected ridge coefficient vectors and 9,408 selection decisions, and recomputes all six primary statistics. Its largest independent forecast discrepancy is $1.0521\times10^{-10}$; the six statistical contrasts agree exactly.

The audit implements its own feature evaluation, ridge forward passes, and tree traversal. It does not refit the forecasters. Discarded B-candidate coefficients were not retained, so the audit checks their recorded selection tables without independently refitting their losses. It also does not independently regenerate the nonlinear trajectories, partitions, or principal subspaces. These checks establish computational consistency within the study and should not be interpreted as external replication.

## Scope of the conclusion

The nonlinear laws were known during development. Independent seeds provide new realizations under those laws, rather than new application domains. Noise levels are paired, and oscillator observations at zero added noise still contain the original sensor noise. The added-noise amplitude is estimated from A; this shared random amplitude also means that the augmented linear-control realizations do not form a single unconditional Gaussian law with fixed variance.

The conclusions are limited by the selected methods, reference width, sample sizes, penalty grids, noise levels, and horizons. The 3% improvement and 5% noninferiority margins are study-design criteria, not application-derived utility thresholds. The paired t calculation is an approximation under its stated distributional assumptions.

The protocol required all six accuracy conditions and the stability checks to be satisfied before further forecasting-performance experiments. Because four accuracy conditions were not established, the recorded combined decision is `STOP_FORECASTING_PERFORMANCE_BRANCH`; no subsequent application or adaptive experiment was used to replace this result. This decision concerns the evaluated procedures and does not prove that every geometric representation or decoder must fail.

## Evidence

- [Primary contrasts and combined decision](analysis/decision.json)
- [Condition-level losses](analysis/condition_metrics.csv)
- [Repetition aggregates](analysis/replicate_metrics.csv)
- [Independent-code audit](results/audit.json)
- [Selected-state replay](results/replay.json)
- [Mathematical definitions and derivations](MATHEMATICAL_BACKGROUND.md)
- [Reproduction commands](README.md)
