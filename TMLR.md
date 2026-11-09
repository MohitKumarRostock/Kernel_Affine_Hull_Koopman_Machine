# Research materials

This branch provides code and numerical evidence for **Feature Augmentation for Forecasting with Reconstruction-Based Representations**. The study compares forecasting models that use normalized reconstruction affinities, normalization constants or membership-weighted local coordinates. Kernel Affine Hull Machines are compared with affine reconstruction, kernel-weighted averaging and raw time-series histories.

The [experiment guide](docs/EXPERIMENT_GUIDE.md) explains the comparisons. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) lists the supported commands and their limits. The [document directory](papers/score_enrichment/) contains LaTeX sources, static tables and build inputs.

The central checker reconstructs the statistical analysis from saved per-run losses. The full simulation and fitting implementation, complete task roster and numerical model/forecast archives are unavailable. The separate predictor-capacity study includes its own source and saved records. These studies use different data and comparison families.

The information-preservation results do not imply that every fitted predictor will forecast accurately. The experiment also does not test whether the representation evolves autonomously under a closed transition model.

The repository contains author attribution and third-party license notices. Journal-format headers do not indicate submission or acceptance.
