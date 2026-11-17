# Forecasting studies and experimental records

The repository contains two studies and a shared [numerical library](README.md#numerical-library). Each study specifies its data, predictors, comparison criteria, and reproduction commands.

## Feature augmentation

**Feature Augmentation for Forecasting with Reconstruction-Based Representations** evaluates additional features for predictors using normalized reconstruction weights. The [code and records](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) compare KAHM, affine, and kernel-weighted averaging features with raw histories. The [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/REPRODUCIBILITY.md) covers saved-loss analysis and the separate predictor-capacity experiment.

The central checker recomputes statistical comparisons from saved losses, not the observations, fitted models, or forecast arrays. The supplementary capacity experiment has separate source, observations, and statistical comparisons.

## Prediction-risk bounds

**A Hierarchy of Risk Certificates for Representation-Based Forecasting** studies minimum mean squared error over bounded Lipschitz predictors. Its [six experiment groups](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) evaluate population-risk confidence bounds with independent episodes, empirical-risk bounds on fixed datasets, predictor fitting, and computational cost.

The [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) identifies the stored inputs and calculations for each group. Household 2010 and Jena 2017 constraint ablations remain in `decoder_diagnostics`; they are separate from the removed affine numerical-output study.

## Supporting experiments

The [supporting-study index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/docs/RESEARCH_INDEX.md) lists forecasting comparisons, representation-dimension and preprocessing studies, residual prediction, feature correction, and common-weight feature ablation. These studies have distinct observations, predictors, losses, and uncertainty analyses.

The [reproducibility overview](REPRODUCIBILITY.md) distinguishes calculations from stored records from regeneration of inputs and fitted models.
