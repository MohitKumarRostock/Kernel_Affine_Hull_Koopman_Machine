# Kernel Affine Hull Koopman Machine

This repository contains Python implementations of kernel affine hull reconstruction and forecasting, plus code and experimental records for two studies of feature representations. Reconstruction-based features assign normalized weights to local models according to their reconstruction errors on an observed time-series history.

## Forecasting studies

| Study | Subject | Code and reproduction |
| --- | --- | --- |
| **Feature Augmentation for Forecasting with Reconstruction-Based Representations** | Adding normalization factors and local coordinates to normalized reconstruction weights | [Study guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02), [reproduction instructions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/REPRODUCIBILITY.md) |
| **A Hierarchy of Risk Certificates for Representation-Based Forecasting** | Lower bounds on minimum mean squared error over bounded Lipschitz predictors | [Study guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08), [reproduction instructions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) |

The feature-augmentation study supplies per-run losses and a statistical checker. Its separate predictor-capacity experiment includes source and regeneration commands. The [supporting-study index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/docs/RESEARCH_INDEX.md) covers forecasting comparisons, representation dimension, preprocessing, residual prediction, feature correction, and common-weight feature ablation. Each study has its own data, predictors, and comparison criteria.

The prediction-risk study contains six experiment groups: controlled calibration, PTB-XL empirical-risk bounds, Bike Sharing minimum empirical-risk analysis, deployment cost, public-data screening, and predictor comparisons. Its documentation distinguishes population-risk confidence bounds from fixed-dataset empirical-risk bounds and measured forecasting performance. The [manuscript experiment index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/docs/MANUSCRIPT_ALIGNMENT.md) maps these groups to the revised paper.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the calculations each command performs and [TMLR.md](TMLR.md) for study links. The central feature-augmentation experiment supplies losses for statistical recomputation, but not its original generator, runner, exact task list, execution environment, fitted model states, or forecast arrays.

## Numerical library

| Module | Function |
| --- | --- |
| [kernel_affine_hull_koopman_machines.py](kernel_affine_hull_koopman_machines.py) | Fit representations and evaluate Koopman forecasts |
| [autoencoder.py](autoencoder.py), [autoencoder_filtering_extended.py](autoencoder_filtering_extended.py) | Fit local reconstruction models and compute reconstructions and distances |
| [parallel_autoencoders.py](parallel_autoencoders.py), [combine_multiple_autoencoders_extended.py](combine_multiple_autoencoders_extended.py) | Fit and combine multiple reconstruction models |
| [dim_reduce.py](dim_reduce.py) | Compute principal-component projections |
| [kernel_regularized_least_squares.py](kernel_regularized_least_squares.py) | Fit regularized kernel regression |
| [kxx_matrix.py](kxx_matrix.py), [kxa_matrix.py](kxa_matrix.py), [distance_matrix.py](distance_matrix.py) | Compute kernel matrices and pairwise distances |

Install the library dependencies and check the main import:

```bash
python -m pip install -r requirements.txt
python -c "import kernel_affine_hull_koopman_machines"
```

These dependencies describe the library on `main`. Use each study's own environment, protocol, and reproduction commands for its experiments.

## Source version and licenses

Cite the exact commit and study protocol used. The [source index](docs/BRANCH_ARCHIVE.json) records source identifiers; [CITATION.cff](CITATION.cff) supplies software citation metadata.

Source code uses the [MIT License](LICENSE). Unless otherwise indicated, original generated data and non-code research outputs use [CC BY 4.0](DATA_LICENSE). Third-party material retains its own license.
