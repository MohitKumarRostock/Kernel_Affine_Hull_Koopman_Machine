# Forecasting studies and numerical evidence

The repository organizes code and data around two research questions.

## Feature augmentation

**Feature Augmentation for Forecasting with Reconstruction-Based Representations** studies which additional features help predictors use normalized reconstruction affinities. The [experiment code and records](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) compare KAHM, affine and kernel-weighted averaging features with raw histories. The [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/REPRODUCIBILITY.md) explains the statistical analysis of saved per-run losses and the separate predictor-capacity experiment.

The central checker reconstructs 40 primary comparisons from 26,624 saved loss rows. It does not regenerate the central observations, fitted models or forecast arrays. The supplementary capacity experiment has its own source, observations and statistical comparison family.

## Lower bounds on prediction risk

**A Hierarchy of Lower Bounds on Prediction Risk for Forecasting Representations** asks whether any predictor in a specified class can meet a prediction-error requirement. The [experiment code and records](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) contain seven experiment groups. Its [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) distinguishes population-risk conclusions, class capacity on fixed evaluation datasets and numerical checks of stored forecasts.

## Supporting experiments

The [four-arm feature ablation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/archive/pre-manuscript-cleanup-2026-11-02/extensions/soft_state_ablation_20261102) uses 32 repetitions and a different feature map and protocol from the central 64-repetition feature-augmentation experiment. Its [primary comparisons](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/extensions/soft_state_ablation_20261102/analysis/primary_contrasts.csv) and [numerical audit](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/extensions/soft_state_ablation_20261102/verification/independent_audit_summary.json) provide its complete results, including unfavorable and unresolved comparisons.

The [research index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/docs/RESEARCH_INDEX.md) links the other experiments. Each study specifies its own observations, predictors, loss function and uncertainty analysis. Their results are interpreted separately.
