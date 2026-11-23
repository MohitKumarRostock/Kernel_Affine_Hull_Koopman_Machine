# Scientific experiment index

The following experiment groups assess risk certificates for prescribed
representation–decoder classes and their computational consequences. The
corresponding research article is *A Hierarchy of Risk Certificates for
Representation-Based Prediction* by Mohit Kumar and Somayeh Kargaran. All
experimental definitions, datasets, numerical assumptions, and verification
commands are documented in this repository; the article is not needed to
recompute the stored numerical results.

The controlled calibration study concerns population MSE under iid
observations independent of development. PTB-XL and Bike Sharing instead
support deterministic statements about minimum empirical MSE on fixed
datasets. These distinctions determine the interpretation of the bounds.

## Principal experiment groups

| Directory under `experiments/` | Scientific result |
| --- | --- |
| `controlled_calibration` | Population-risk confidence bounds using independent episodes |
| `ecg_cohort` | Minimum empirical-risk lower bounds on fixed PTB-XL observations |
| `class_diagnosis` | Minimum empirical-risk and fitted-predictor excess-risk bounds for 54 Bike Sharing classes |
| `deployment_cost` | Prediction accuracy, numerical model storage, and inference time |
| `public_screening` | Beijing and Bike Sharing class exclusions and fitted-model counts |

The finite-support controlled forecasting demonstrator has separate [implementation and scientific documentation](https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/tree/d874b8ed426a635b158fdc02848d5b2efdba048b). That referenced commit is a historical evidence version; documentation revisions should use a separately identified commit.

## Supplementary predictor diagnostics

`experiments/decoder_diagnostics/` contains predictor comparisons and constraint ablations for specified forecasting periods, including Household 2010 and Jena 2017. Its README states the datasets, comparisons, numerical assumptions, and results.

## Reproduction and provenance

Cite the exact commit used for reproduction. `MANIFEST.json` identifies source code, documentation, and numerical inputs by SHA-256 hash. Experiment protocols and provenance records identify data sources, model parameters, and verification inputs.

`validation/summary.json` reports the numerical checks for its stated source manifest. Reproduction commands recompute results from stored inputs and models without repeating all training or timing experiments. [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) gives commands and numerical assumptions; [DATA.md](../DATA.md) identifies the datasets.
