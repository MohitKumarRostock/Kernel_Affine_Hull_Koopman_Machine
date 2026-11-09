# Kernel Affine Hull Koopman Machine

This repository provides numerical methods for kernel affine hull reconstruction and forecasting, together with code and evidence for two studies of forecasting representations. A reconstruction-based representation assigns normalized weights to local models according to how well they reconstruct an observed time-series history. The studies ask which additional features help prediction and how to assess the best prediction error achievable within a specified predictor class.

## Forecasting studies

| Study | Scientific question | Code and evidence |
| --- | --- | --- |
| **Feature Augmentation for Forecasting with Reconstruction-Based Representations** | Which normalization factors and local coordinates should accompany normalized reconstruction weights? | [Study guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02), [reproduction instructions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/REPRODUCIBILITY.md) |
| **A Hierarchy of Lower Bounds on Prediction Risk for Forecasting Representations** | Can any predictor in a specified class meet a mean-squared-error requirement? | [Study guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08), [reproduction instructions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) |

The feature-augmentation study supplies the central per-run loss records and a checker for its statistical comparisons. Its separate predictor-capacity experiment includes experiment source and regeneration commands. The [supporting studies](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/docs/RESEARCH_INDEX.md) cover controlled forecasting comparisons, representation dimension, preprocessing, residual prediction, feature correction and common-weight feature ablation. Each study has its own observations, predictors and comparison criteria.

The prediction-risk study provides seven experiment groups and commands that recompute checks from stored records. Its claims distinguish population risk, predictor-class capacity on fixed evaluation datasets and numerical checks of stored forecasts. The conditions on the predictor class and sampling procedure are part of each conclusion.

See [Reproducibility](REPRODUCIBILITY.md) for the scope of these commands and [the study index](TMLR.md) for additional entry points. The central feature-augmentation records support statistical recomputation; its original generator, experiment runner, exact task roster, execution environment, fitted model states and forecast arrays are not supplied.

## Numerical library

The Python modules on this branch implement kernel affine hull reconstruction and Koopman prediction:

| Module | Purpose |
| --- | --- |
| [kernel_affine_hull_koopman_machines.py](kernel_affine_hull_koopman_machines.py) | Representation fitting, associations, Koopman prediction and evaluation |
| [autoencoder.py](autoencoder.py) and [autoencoder_filtering_extended.py](autoencoder_filtering_extended.py) | Fit local reconstruction models and evaluate their reconstructions and distances |
| [parallel_autoencoders.py](parallel_autoencoders.py) and [combine_multiple_autoencoders_extended.py](combine_multiple_autoencoders_extended.py) | Fit and combine multiple reconstruction models |
| [dim_reduce.py](dim_reduce.py) | Principal-component projection |
| [kernel_regularized_least_squares.py](kernel_regularized_least_squares.py) | Regularized kernel regression |
| [kxx_matrix.py](kxx_matrix.py), [kxa_matrix.py](kxa_matrix.py) and [distance_matrix.py](distance_matrix.py) | Kernel matrices and pairwise distances |

Install the library dependencies and check the main import with:

```bash
python -m pip install -r requirements.txt
python -c "import kernel_affine_hull_koopman_machines"
```

[requirements.txt](requirements.txt) specifies the numerical library dependencies. Each linked study provides its own experiment configuration, environment information and verification commands. Use those study instructions to evaluate the corresponding scientific results.

## Source identification and licensing

The [source index](docs/BRANCH_ARCHIVE.json) records scientific source identifiers. Cite the study protocol and exact commit used, together with the software metadata in [CITATION.cff](CITATION.cff).

Source code is available under the [MIT License](LICENSE). Unless otherwise indicated, original generated data and non-code research outputs use [CC BY 4.0](DATA_LICENSE). Separately identified third-party material remains under its own license.
