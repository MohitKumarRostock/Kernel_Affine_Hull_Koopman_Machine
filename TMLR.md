# Research resources

This repository provides code and numerical evidence for **Feature Augmentation for Forecasting with Reconstruction-Based Representations**. The central study compares forecasting models using normalized reconstruction affinities, memberships augmented with log normalization constants, and memberships augmented with weighted local coordinates. KAHM reconstructions are evaluated alongside affine projection, kernel-weighted averaging, and raw time-series histories.

The [experiment guide](docs/EXPERIMENT_GUIDE.md) explains the comparisons. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) lists commands, dependencies, source identities, and the scope of each available check. The [research index](docs/RESEARCH_INDEX.md) links related experiments.

The central statistical-analysis script reconstructs procedure summaries, paired differences, and all primary comparisons from saved per-run losses. The complete central simulation and fitting implementation, exact task list, fitted model states, and forecast arrays are unavailable. The separate predictor-capacity study provides simulation and fitting source and saved records. These studies use different data and comparison families.

Information preservation does not guarantee accurate fitted predictions. The central experiment predicts future observations directly; it does not evaluate a closed transition model for the feature vector.

Author attribution and third-party license notices accompany the source and data files.
