# Target-geometry study: audited findings

The study evaluates a shared supervised metric and alternative residual targets under a protocol fixed before execution.

The public execution checkpoint is [`0d7ee3c17515eb1d922cdd302facd2b7874dc9a9`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/0d7ee3c17515eb1d922cdd302facd2b7874dc9a9). It preceded the first confirmatory fit. This is a timestamped protocol, not an external registry preregistration.

All **48 tasks, 768 final forecasts, and 18 primary contrasts** are accounted for. There were **zero arm-level fallbacks, metric fallbacks, task failures, recorded warnings, or group reductions**. All 48 metric fits reported optimizer convergence. These execution statistics concern this study alone.

## Main findings

KAHM local-coordinate residual corrections can improve a raw ridge forecast under the evaluated protocols. The experiment does **not** establish that the KAHM gate is the source of an advantage over simpler local geometry. It also does not establish an incremental synthetic benefit from either the shared proxy metric or the switch to OOF donor residuals.

Here OOF denotes out-of-fold donor residuals: each residual target is formed using a donor baseline that excludes the corresponding held-out group or respects its chronological availability. Learned-OOF KAHM improves ridge on all three synthetic laws in the simultaneous analysis: **11.18% on nonlinear AR, 53.26% on Van der Pol, and 19.44% on Duffing**. All three metric-increment intervals and all three OOF-target-increment intervals cross zero. Among the nine equally trained local-control contrasts, none favors KAHM with an interval excluding zero; five favor the control. Raw RBF has lower mean error than learned-OOF KAHM on all three laws as a secondary comparison.

## Archive accuracy and measured computation

In the procedure identifiers, `fixed` denotes the original metric, `learned` denotes the supervised diagonal metric, `is` denotes in-sample residual targets, and `oof` denotes out-of-fold donor residual targets. The final component identifies the KAHM, affine, locality, or RBF local features. `joint_kahm_ridge` concatenates raw and KAHM features in a single ridge predictor.

The table uses equal collection weights after equal series weights. Four selected collections and twelve series do not support a population-domain confidence interval. Negative error change is favorable. Pipeline times are warm component sums on one host, including required donor, metric, bank and selection work. Ratios are not intrinsic algorithmic constants.

| Procedure | Scaled MSE | Change vs ridge | Seconds/task | Cost / ridge |
|---|---:|---:|---:|---:|
| `ridge` | 1.300458 | +0.00% | 0.001299 | 1.00 |
| `raw_rbf` | 1.948283 | +49.82% | 0.004753 | 3.66 |
| `naive` | 1.847607 | +42.07% | 0.004268 | 3.29 |
| `joint_kahm_ridge` | 1.452956 | +11.73% | 0.058315 | 44.89 |
| `fixed_is_kahm` | 1.264589 | -2.76% | 0.051012 | 39.27 |
| `fixed_is_affine` | 1.258372 | -3.24% | 0.028923 | 22.27 |
| `fixed_is_locality` | 1.263541 | -2.84% | 0.027775 | 21.38 |
| `fixed_is_rbf` | 1.257888 | -3.27% | 0.026547 | 20.44 |
| `fixed_oof_kahm` | 1.276653 | -1.83% | 0.055609 | 42.81 |
| `fixed_oof_affine` | 1.273072 | -2.11% | 0.033588 | 25.86 |
| `fixed_oof_locality` | 1.273494 | -2.07% | 0.032358 | 24.91 |
| `fixed_oof_rbf` | 1.274554 | -1.99% | 0.031161 | 23.99 |
| `learned_oof_kahm` | 1.272128 | -2.18% | 0.162198 | 124.87 |
| `learned_oof_affine` | 1.268679 | -2.44% | 0.141631 | 109.03 |
| `learned_oof_locality` | 1.266922 | -2.58% | 0.141097 | 108.62 |
| `learned_oof_rbf` | 1.284842 | -1.20% | 0.139143 | 107.12 |

Learned-OOF KAHM improves the aggregate by **2.18%**, but the fixed in-sample KAHM correction improves it by **2.76%** at substantially lower cost. The lowest registered aggregate belongs to fixed in-sample RBF-local correction (**3.27% below ridge**). Learned affine and locality corrections also have lower aggregate error than learned KAHM. Selecting the lowest method after observing this table would be a new policy, not a prospectively tested one.

Learned KAHM improves all four collection means, but only **7 of 12 individual series**; it harms four and exactly retains ridge on one. Removing one collection at a time leaves a descriptive gain between **1.49% and 3.35%**. This check concerns sensitivity to these four collections, not independent confirmations.

## All prespecified simultaneous comparisons

Differences are control MSE minus candidate MSE; positive favors the candidate. Each row uses twelve complete independent repetitions and the shared 18-comparison Bonferroni critical value. Coverage relies on the stated paired-difference normal approximation. No joint coverage is claimed with the manuscript’s older comparison families.

| Law | Control | Candidate | Mean difference | Simultaneous interval | Decision |
|---|---|---|---:|---:|---|
| nonlinear_ar | `ridge` | `learned_oof_kahm` | 0.0005656495 | [0.00045943066, 0.00067186834] | candidate_better |
| nonlinear_ar | `fixed_oof_kahm` | `learned_oof_kahm` | -1.1108174e-05 | [-0.00010507319, 8.2856841e-05] | unresolved |
| nonlinear_ar | `fixed_is_kahm` | `fixed_oof_kahm` | -2.7108039e-06 | [-5.8283681e-06, 4.0676031e-07] | unresolved |
| nonlinear_ar | `learned_oof_affine` | `learned_oof_kahm` | -3.2088143e-05 | [-0.00011066629, 4.6490005e-05] | unresolved |
| nonlinear_ar | `learned_oof_locality` | `learned_oof_kahm` | 3.8225087e-05 | [-0.00012235027, 0.00019880044] | unresolved |
| nonlinear_ar | `learned_oof_rbf` | `learned_oof_kahm` | -8.3457878e-05 | [-0.00018184744, 1.4931681e-05] | unresolved |
| vanderpol | `ridge` | `learned_oof_kahm` | 0.023510314 | [0.021236088, 0.02578454] | candidate_better |
| vanderpol | `fixed_oof_kahm` | `learned_oof_kahm` | 0.00075384348 | [-0.0009390763, 0.0024467633] | unresolved |
| vanderpol | `fixed_is_kahm` | `fixed_oof_kahm` | -3.160792e-06 | [-8.3584133e-06, 2.0368293e-06] | unresolved |
| vanderpol | `learned_oof_affine` | `learned_oof_kahm` | -0.0017508327 | [-0.0040881419, 0.00058647651] | unresolved |
| vanderpol | `learned_oof_locality` | `learned_oof_kahm` | -0.0066614622 | [-0.0087646499, -0.0045582744] | candidate_worse |
| vanderpol | `learned_oof_rbf` | `learned_oof_kahm` | -0.0051694716 | [-0.0062293211, -0.004109622] | candidate_worse |
| duffing | `ridge` | `learned_oof_kahm` | 0.0013443122 | [0.001054181, 0.0016344433] | candidate_better |
| duffing | `fixed_oof_kahm` | `learned_oof_kahm` | -0.00010302698 | [-0.00033570487, 0.0001296509] | unresolved |
| duffing | `fixed_is_kahm` | `fixed_oof_kahm` | -1.0162414e-06 | [-5.6878438e-06, 3.6553609e-06] | unresolved |
| duffing | `learned_oof_affine` | `learned_oof_kahm` | -0.00083613671 | [-0.0012540179, -0.00041825551] | candidate_worse |
| duffing | `learned_oof_locality` | `learned_oof_kahm` | -0.0010358275 | [-0.0014891144, -0.00058254061] | candidate_worse |
| duffing | `learned_oof_rbf` | `learned_oof_kahm` | -0.00024326171 | [-0.00043508705, -5.1436362e-05] | candidate_worse |

## Mechanistic interpretation and limits

The learned metric minimizes a Gaussian residual-regression proxy shared across all four gates. It does not optimize KAHM forecasting loss end to end. Synthetic learned metrics remain close to identity: maximum/minimum weight ratios range from 1.0013 to 1.1054, with median objective reduction 1.8741e-6. The archive median reduction is 2.9692e-4; one pedestrian task reaches the allowed weight ratio of 16. Proxy convergence does not imply an appropriate forecasting metric.

A conditional-risk identity explains the residual-target comparison. A correction optimized for a donor baseline need not remain optimal after that baseline changes. Its risk reduction for the final baseline equals the available final-baseline residual-mean energy minus the energy of the conditional donor-to-final discrepancy, as defined and derived in [THEORY.md](THEORY.md). This identity assumes fixed functions and one common evaluation law. Observed training-row disagreement is descriptive, and forward folds need not share that law. OOF target construction is therefore a design to test, not an improvement guarantee.

The study uses capped univariate histories, three familiar simulation laws, four selected archive collections and finite tuning grids. The four collections had not been used in the preceding studies, although some broad application domains overlap. Bitcoin series are activity indicators, not price forecasts. The archive split allows later forecast origins to use already observed values, and missing windows are omitted according to the frozen rule.

## Reproduction and evidence

- [Exact analysis summary](reports/summary.json)
- [Every task-method loss and charged cost](reports/task_metrics.csv)
- [Complete primary intervals](reports/primary_contrasts.csv)
- [Source-level results](reports/natural_sources.csv)
- [Metric and donor diagnostics](reports/metric_diagnostics.csv)
- [Validation grids](reports/validation_trials.csv)
- [28 engineering checks](reports/engineering_tests.txt)
- [Public checkpoint metadata](PUBLICATION.json)
- [Scientific freeze](freeze.json)
- [Protocol](PROTOCOL.md)

The independent auditor reconstructs input arrays, final losses, residual identities, donor exclusions, metric objectives and computation dependencies. It checks the chosen minimum against complete recorded validation grids; it does not independently refit each trial. The source and manuscript artifacts retain this qualification.
