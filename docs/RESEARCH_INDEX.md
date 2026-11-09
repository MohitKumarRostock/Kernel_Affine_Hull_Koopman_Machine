# Experiments and source code

## Feature augmentation for forecasting

[Feature Augmentation for Forecasting with Reconstruction-Based Representations](../README.md) studies what information reconstruction-based features preserve and whether specified forecasting models benefit from additional features. Its central evaluation dataset contains 64 repetitions under each of four data-generating processes, with 40 primary comparisons.

| Resource | Purpose |
| --- | --- |
| [Central saved-loss bundle](../reproduction/score_enrichment/) | Seventeen experiment records and an independent checker rebuilding the primary statistics from saved per-run losses |
| [Supplementary capacity bundle](../reproduction/capacity_followup/) | Experiment source and records for a separate comparison with 24 repetitions under three processes at 64 and 256 Nyström landmarks |
| [Protocol and primary-summary directory](../extensions/score_enrichment_20261102/) | Summary table, protocol, summary-arithmetic checker and evidence-availability guide |
| [Experiment guide](EXPERIMENT_GUIDE.md) and [reproduction guide](../REPRODUCIBILITY.md) | Interpretation, commands, requirements and limits of each check |

The central saved-loss bundle supports statistical reconstruction. The complete central simulation and fitting implementation, exact task list, and numerical model and forecast archives are unavailable. The [availability guide](../extensions/score_enrichment_20261102/AVAILABILITY.md) identifies the supplied files and the unsupported reproduction tasks.

## Supplementary and related studies

| Question | Study and scope |
| --- | --- |
| Does a local-coordinate advantage persist with a larger finite predictor? | [Local capacity bundle](../reproduction/capacity_followup/): 24 repetitions under three data-generating processes, MSE and 12 primary comparisons; source, archived records and verify/audit/full commands are included |
| How do score-based features behave in the supporting controlled experiments? | The capacity bundle's [supporting first-stage records](../reproduction/capacity_followup/archive/stage1/) retain their 29-comparison analysis and determine the capacity study's practical margins; [scientific source snapshot](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/a67c40190458bab7b3e56b4fac76770798d074b3/extensions/predictive_geometry) identifies their source |
| What do distance-based weights, local coordinates and reconstruction residuals contribute? | [Four-arm ablation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/soft_state_ablation_20261102): a different membership map and a separate 32-repetition evaluation dataset |
| What does joint input scaling and local feature dimension selection change? | [Joint-selection study](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/00d7741f1feaf4693630f039d8d37b9db6d9f14f/extensions/soft_state_joint_selection_20261102): its own 32-repetition evaluation dataset and six primary comparisons |
| How does local feature dimension affect residual forecasting? | [Dimension study](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/dimension_forecasting): a model that adds a learned residual correction to a baseline forecast and separate six-criterion family |
| How do input preprocessing and dimension selection interact? | [Preprocessing study](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/preprocessing_dimension): primary and secondary procedures retain their separate interpretations |
| What changes under a different residual-target construction? | [Computational reproduction on the same evaluation dataset](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/crossfit_residual_reproduction_20261101): no additional independent repetition units |
| Does expanding the candidate forecasting models improve the designated criteria? | [Candidate-model augmentation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/policy_augmentation_20261101): neither nonlinear practical-superiority criterion established |
| How does that model-selection procedure perform on observed series? | [M4 training-series evaluation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/policy_augmentation_natural_20261101): descriptive unfavorable aggregate KAHM augmentation result; no official M4 Test data |

These studies address different procedures and populations. Their observations, confidence families and practical margins are not pooled into the central feature-augmentation analysis. Code from a different experiment does not supply the missing central implementation or establish that both experiments used the same numerical conventions.

## Study archive

[Commit `5123365917ae630b684f8a3bebb0294ee695ce7a`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a) contains source code, protocols and results for the linked studies. [Additional experiments](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions) cover supervised geometric features, reconstruction training and online adaptation.

The Koopman-machine experiments have a [result-to-source mapping](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/reproduction) and [environment records](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/provenance). Each study specifies its own data, dependencies and supported commands.

