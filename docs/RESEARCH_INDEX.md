# Research and evidence index

## Main research topics

[Feature Augmentation for Forecasting with Reconstruction-Based Representations](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) studies information retention and forecasting with normalized reconstruction-affinity features. The [local study directory](../extensions/score_enrichment_20261102/) contains its exact primary-summary evidence and protocol identities. The linked reproduction bundle reconstructs the 40 primary comparisons from saved repetition losses.

[A Hierarchy of Lower Bounds on Prediction Risk for Forecasting Representations](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) studies lower bounds under explicit assumptions on prediction losses and reference classes. Its numerical experiments have their own inputs and statistical analysis.

## Related studies

| Scientific question | Study and evidence |
| --- | --- |
| Score-based local coordinates and predictor capacity | [Predictive geometry](../extensions/predictive_geometry/README.md): complete controlled and capacity studies, including unfavorable laws |
| Distance-based soft assignments, local position, and reconstruction differences | [Four-arm ablation](../extensions/soft_state_ablation_20261102/README.md): different gate map and a separate 32-repetition evaluation dataset |
| Validation-selected native dimension in residual forecasting | [Dimension study](../extensions/dimension_forecasting/README.md): separate retained-baseline architecture and six-criterion family |
| Input preprocessing and dimension | [Preprocessing study](../extensions/preprocessing_dimension/FINDINGS.md): complete-procedure and secondary comparisons remain distinct |
| Residual target construction | [Computational reproduction](../extensions/crossfit_residual_reproduction_20261101/README.md): original simulation units with prior outcome knowledge, not fresh independent repetitions |
| Enlarging a forecasting policy | [Policy augmentation](../extensions/policy_augmentation_20261101/RESULTS.md): neither nonlinear practical-superiority criterion established |
| Natural-series policy use | [M4 training-series evaluation](../extensions/policy_augmentation_natural_20261101/FINDINGS.md): descriptive 2.273% loss increase after KAHM augmentation; no official M4 Test data |

## Shared computational resources

The dimension study supplies the bundled geometry implementation and data-generation source used by the preprocessing, residual-prediction, policy-augmentation, and local-position studies. Each study retains its own protocol, source hashes, observations, and analysis.

The residual study verifies its scientific source against the frozen files in `extensions/crossfit_residual_v2/`. The dimension-study statistical checks use [`paired_margin_contrast`](../extensions/reconstruction_mechanism/analyze.py) as a reference implementation. These files are dependencies of the supported studies.

[Reproduction programs](../reproduction/README.md) provide statistical checks, kernel-identity diagnostics, and table generation. The [reproducibility guide](../REPRODUCIBILITY.md) states the inputs and limits of each computation.
