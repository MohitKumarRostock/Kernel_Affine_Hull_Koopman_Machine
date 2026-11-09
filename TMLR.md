# Forecasting studies and numerical evidence

The repository organizes code and data around two research questions. The [numerical library](README.md#numerical-library) supplies reusable reconstruction and forecasting methods. Each study's guide describes its experimental implementation, data and analysis.

## Feature augmentation

**Feature Augmentation for Forecasting with Reconstruction-Based Representations** studies which additional features help predictors use normalized reconstruction affinities. The [experiment code and records](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) compare KAHM, affine and kernel-weighted averaging features with raw histories. The [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/REPRODUCIBILITY.md) explains the statistical analysis of saved per-run losses and the separate predictor-capacity experiment.

The central checker reconstructs primary comparisons from saved losses. It does not regenerate the central observations, fitted models or forecast arrays. The supplementary capacity experiment has its own source, observations and statistical comparison family.

## Lower bounds on prediction risk

**A Hierarchy of Lower Bounds on Prediction Risk for Forecasting Representations** asks whether any predictor in a specified class can meet a prediction-error requirement. The [experiment code and records](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) contain seven experiment groups. Its [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) distinguishes population-risk conclusions, class capacity on fixed evaluation datasets and numerical checks of stored forecasts.

## Supporting experiments

The [supporting-study index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/docs/RESEARCH_INDEX.md) links controlled forecasting comparisons, representation-dimension and preprocessing studies, residual prediction, feature correction and common-weight feature ablation. Each study specifies its own observations, predictors, loss function and uncertainty analysis. Their results are interpreted separately.

The [reproducibility overview](REPRODUCIBILITY.md) explains which commands analyze stored records and which regenerate experimental inputs and predictions.
