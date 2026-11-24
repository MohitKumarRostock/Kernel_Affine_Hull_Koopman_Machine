# Research studies

The research collections below address distinct statistical and computational questions. Each collection uses its own observations, representations, predictor classes, evaluation criteria and dependency specification.

## Prediction limits and Koopman closure

For simplex-valued state representations, this study establishes lower confidence bounds on the minimum population one-step vector RMSE over spectral-norm-constrained linear predictors. Its independent-pair evaluation designs include a four-state distribution with an analytical optimum and a Van der Pol oscillator with a fixed KAHM representation. The accompanying trajectory forecasting benchmarks evaluate association prediction and numerical closure diagnostics.

- [Scientific formulation, certificate equations and results](SOFT_STATE_CERTIFICATES.md)
- [Source and data identities, reproduction procedures and verification](SOFT_STATE_REPRODUCIBILITY.md)
- [Primary experiment record](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e)
- [Four-method comparison record](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2)

Reference: M. Kumar and S. Kargaran (2026), [Prediction Limits and Koopman Closure of Geometry-Induced Soft State Abstractions](https://arxiv.org/abs/2609.32652v3).

## Forecasting with augmented features

**Feature Augmentation for Forecasting with Reconstruction-Based Representations** investigates the forecasting utility of additional features derived from reconstruction models. The representations include normalized reconstruction weights, additional normalization information and local coordinates. Comparisons include KAHM, affine and kernel-weighted averaging methods, and predictors supplied with raw observed histories.

The principal statistical analysis operates on distributed per-run prediction losses. Its scope is recomputation of comparative statistics from those losses. The underlying original generator, runner, exact task list, fitted states, forecast arrays and execution environment are not distributed for that analysis. The supplementary predictor-capacity experiment has separate executable code, observations and numerical results.

- [Scientific and computational record](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/7ce45d2949154777dfca56f2ed970eabff768063)
- [Reproduction instructions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/7ce45d2949154777dfca56f2ed970eabff768063/REPRODUCIBILITY.md)
- [Per-run loss analysis](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/7ce45d2949154777dfca56f2ed970eabff768063/reproduction/score_enrichment/README.md)
- [Predictor-capacity experiment](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/7ce45d2949154777dfca56f2ed970eabff768063/reproduction/capacity_followup/README.md)

## Risk certificates for representation-based forecasting

**A Hierarchy of Risk Certificates for Representation-Based Forecasting** addresses minimum mean squared error for constrained Lipschitz predictors of representation-derived features. Population-risk confidence statements, fixed-dataset empirical-risk bounds and attained prediction errors concern different mathematical quantities and require distinct assumptions.

The experiment groups encompass controlled calibration, PTB-XL empirical-risk analysis, Bike Sharing minimum empirical-risk analysis, deployment cost, public-data screening and predictor comparisons. The reproduction driver checks the identities of the stored inputs and source, recomputes reported bounds and comparisons, and in the deployment experiment evaluates stored models on supplied inputs. It does not regenerate every model fit or measure new runtimes.

- [Scientific and computational record](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/abd1ed2e055bad45b89b13f216ef2719d9f21274)
- [Reproduction instructions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/abd1ed2e055bad45b89b13f216ef2719d9f21274/REPRODUCIBILITY.md)
- [Numerical verification summary](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/abd1ed2e055bad45b89b13f216ef2719d9f21274/validation/summary.json)

## Supporting forecasting studies

The [supporting study index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/cb343ad65ed7c9602d1edc74c0207b77d9bbc4aa/docs/RESEARCH_INDEX.md) provides computational records for representation dimension, preprocessing, residual prediction, feature correction, forecasting comparisons and common-weight feature ablations. The studies use separate model and data configurations and must be interpreted under their respective protocols.

For the common reconstruction and forecasting interface, see [Kernel affine hull reconstruction and linear forecasting](KAHKM.md). For software and data licensing, see [LICENSE](../LICENSE) and [DATA_LICENSE](../DATA_LICENSE).
