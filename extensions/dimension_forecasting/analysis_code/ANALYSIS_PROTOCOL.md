# Statistical analysis of dimension-selected forecasting procedures

## Estimand and comparison family

The experiment evaluates forecasting procedures under four specified numerical observation laws. A procedure includes its training rule, validation-based dimension and regularization selection, and calibration-based mixing rule. Its loss is the mean squared error on independent D episodes, divided separately at each horizon by the population variance of the retained A targets. The variance denominator is floored at 10^-12. Each task gives equal weight to horizons 1, 4 and 12. Within each repetition, the nonlinear aggregate gives equal weight to the three nonlinear laws and the two added-noise levels. The linear control is averaged separately across its two noise levels.

The primary candidate is bounded KAHM reconstruction with one native cap selected on B. The candidate is compared with independently dimension-selected common features and bounded RBF reconstruction, raw-input RBF regression, histogram gradient boosting, and the C-selected complete non-KAHM procedure. B chooses each geometric procedure's cap independently; therefore these primary comparisons assess complete forecasting procedures. Additional controls use the same bank as the selected KAHM procedure to distinguish a change in common features from the contribution of reconstruction.

For each of the five nonlinear comparisons, the paired repetition gap is 0.97 times comparator loss minus candidate loss. The linear gap is 1.05 times ridge loss minus candidate loss. For each gap, the one-sided lower bound is its mean minus t_(1-0.05/6,23) times its sample standard deviation divided by sqrt(24). All six lower bounds must be strictly positive to establish the statistical criterion. The family has approximate 95% simultaneous coverage under the paired-mean t assumptions. The independent unit is a complete repetition; overlapping histories and paired noise levels do not increase the inferential sample size.

The margins of 3% for nonlinear improvement and 5% for linear noninferiority preserve the original research criterion. They are not universal thresholds for scientific interest or deployment. These approximate confidence bounds are distinct from the bounded-loss risk certificates studied in the separate certificate workflow.

## Secondary analyses

Fixed cap 2 under the historical construction and native cap 20 are secondary contexts. Native cap candidates 1, 2, 4, 8, 12 and 20 are all retained. At ambient dimension 12, caps 12 and 20 have identical initial admissible ranks; an ascending tie rule selects 12. Historical cap 2 and native cap 2 also have identical admissible ranks at this input width. These identities are verified on the fitted arrays and predictions.

All methods, laws, noise levels and horizons are reported. Fixed-cap, matched-bank, unbounded-feature and gamma-one comparisons are secondary. Their two-sided paired-t intervals are nominal 95% intervals without adjustment for the secondary family; they cannot change the primary conclusion. The two terms in the selected-minus-historical loss decomposition are the change in common-feature loss and the change in the incremental contribution of KAHM reconstruction. The identity is evaluated using the same KAHM-selected bank for the common and reconstruction terms.

## Numerical checks and computational cost

The analysis reconstructs B cap scores, B selected hyperparameters, C gamma scores, complete C policy scores, local admissible ranks, and deterministic readout limits from retained arrays and records. The maximum-error screen compares each task's largest absolute KAHM error over all D rows and horizons with the largest ridge error; the threshold is five. It is not a ratio of mean squared errors.

Cost summaries preserve the original dependency accounting. A dimension-selected geometric procedure is charged for all six native-cap searches and its selected calibration. The non-KAHM policy is charged for the union of the work required by its seven candidates. Actual joint execution times and per-procedure dependency sums are reported separately. Timings are descriptive measurements on a shared host. Deployment storage records the selected state rather than every state retained for research.

## Data access and integrity

The analysis code is frozen before generation of production A/B/C episodes. Cap selections are written before C arrays are loaded. Every fitted state is sealed before D generation; predictions are sealed before D targets are released for scoring. The analyzer verifies the source, fit, prediction and target-release records before opening terminal target arrays. It imports no forecasting implementation and does not deserialize model states. Any unexpected missing task, changed file, nonfinite outcome or unsupported selection stops the analysis. All 192 task suites remain in every applicable result.

The study tests fresh repetitions of known laws. It does not provide a new application cohort, a public preregistration, or an adjustment for every earlier research decision.
