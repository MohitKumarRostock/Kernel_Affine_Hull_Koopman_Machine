# Current evidence and historical studies

## Predictive enrichment

[Predictive Enrichment of Geometry-Induced Soft-State Abstractions](../papers/score_enrichment/) studies information retention and the access a specified predictor has to that information. Its central cohort contains 64 repetitions under each of four laws, with 40 primary comparisons.

| Resource | Purpose |
| --- | --- |
| [Central saved-loss bundle](../reproduction/score_enrichment/) | Seventeen original evidence records and an independent checker rebuilding the primary statistics from selected repetition losses |
| [Supplementary capacity bundle](../reproduction/capacity_followup/) | Original experiment source and records for the separate 24-repetition, three-law comparison at 64 and 256 Nyström landmarks |
| [Original publication study directory](../extensions/score_enrichment_20261102/) | Published summary table, protocol, original summary checker and historical provenance/availability documents |
| [Experiment guide](EXPERIMENT_GUIDE.md) and [reproduction guide](../REPRODUCIBILITY.md) | Interpretation, commands, requirements and limits of each check |

The central saved-loss bundle supports statistical reconstruction. The original central model-generation implementation, complete task roster and numerical model/forecast archives remain unavailable. The historical availability record in the original study directory predates this saved-loss bundle; it should be read as a record of that earlier distribution.

## Supplementary and related studies

| Question | Study and scope |
| --- | --- |
| Does a local-coordinate advantage persist with a larger finite predictor? | [Local capacity bundle](../reproduction/capacity_followup/): 24 repetitions under three laws, MSE and 12 primary comparisons; source, archived records and verify/audit/full commands are included |
| How do score-based features behave in the earlier controlled experiments? | The capacity bundle's [supporting first-stage records](../reproduction/capacity_followup/archive/stage1/) retain their 29-comparison analysis and determine the capacity study's practical margins; [original predictive-geometry snapshot](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/a67c40190458bab7b3e56b4fac76770798d074b3/extensions/predictive_geometry) identifies their source |
| What do center-distance gates, position and reconstruction departures contribute? | [Four-arm ablation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/soft_state_ablation_20261102): a different membership map and a separate 32-repetition cohort |
| What does joint input scaling and native dimension selection change? | [Joint-selection study](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/00d7741f1feaf4693630f039d8d37b9db6d9f14f/extensions/soft_state_joint_selection_20261102): its own 32-repetition cohort and six primary comparisons |
| How does native dimension affect residual forecasting? | [Dimension study](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/dimension_forecasting): a retained-baseline architecture and separate six-criterion family |
| How do input preprocessing and dimension selection interact? | [Preprocessing study](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/preprocessing_dimension): primary and secondary procedures retain their separate interpretations |
| What changes under a different residual-target construction? | [Same-cohort computational reproduction](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/crossfit_residual_reproduction_20261101): no additional independent repetition units |
| Does enlarging a forecasting policy improve the designated criteria? | [Policy augmentation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/policy_augmentation_20261101): neither nonlinear practical-superiority criterion established |
| How does that policy behave on natural series? | [M4 training-series evaluation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/policy_augmentation_natural_20261101): descriptive unfavorable aggregate KAHM augmentation result; no official M4 Test data |

These studies address different procedures and populations. Their observations, confidence families and practical margins are not pooled into the central enrichment analysis. Older source can help explain the lineage of generators and geometric computations; it does not supply the missing central runner or certify that its numerical conventions are identical.

## Immutable historical archive

The retained archive is [commit `5123365917ae630b684f8a3bebb0294ee695ce7a`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a), with branch `archive/pre-manuscript-cleanup-2026-11-02` as its entry point. The commit identifies the original files. Historical results, including unfavorable and unresolved outcomes, retain their original protocols and recorded numerical values.

Earlier manuscript editions remain at [local_position_forecasting](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/papers/local_position_forecasting), [soft_state_forecasting](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/papers/soft_state_forecasting) and [tmlr_integration](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions/tmlr_integration). Supervised geometry, reconstruction training, adaptation and other exploratory studies remain in the [archived extensions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/extensions).

The original Koopman-machine runners and outputs retain their [result mapping](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/reproduction) and [provenance records](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a/provenance). Use each historical study's declared dependencies and data requirements. The two current reproduction bundles identify their own supported commands and source identities.
