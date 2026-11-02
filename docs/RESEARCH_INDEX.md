# Research index

## Central forecasting study

The focused article is [Local Position and Reconstruction Features for Forecasting](../papers/local_position_forecasting/). Its matched experiment is [soft_state_ablation_20261102](../extensions/soft_state_ablation_20261102/). The main article reports all four primary comparisons; the supplement contains the full protocol, descriptive decompositions, related studies, and computational audit scope.

The 32-repetition central experiment tests a hierarchy of direct forecasting features. It does not combine observations or comparison families from the studies below.

## Related evidence

| Study | Evidence and scope | Record |
| --- | --- | --- |
| Validation-selected local dimension | 24 repetitions and 192 tasks; three of six criteria established. KAHM improves common features and raw-input RBF, but the selected non-KAHM policy has lower mean loss. | [Dimension findings](../extensions/dimension_forecasting/FINDINGS.md) |
| Preprocessing and dimension | 24 fresh repetitions; one of four criteria established. Primary KAHM and preprocessing superiority margins are not established. | [Preprocessing findings](../extensions/preprocessing_dimension/FINDINGS.md) |
| Residual-target computational reproduction | Regenerates 24 original simulation units with prior outcome knowledge; two of four criteria established. It contributes no additional independent repetitions. | [Residual-target results](../extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md) |
| Policy augmentation | 24 fresh repetitions and 192 tasks. Nonlinear improvements of 0.816% over the base policy and 0.658% over the added-RBF control do not establish their 3% margins. Both linear noninferiority criteria pass. | [Policy results](../extensions/policy_augmentation_20261101/RESULTS.md) |
| Natural-data policy evaluation | Descriptive panel of 48 M4 training series. KAHM augmentation has 2.273% higher aggregate loss than the base policy. Official M4 Test data are not used. | [Natural-data findings](../extensions/policy_augmentation_natural_20261101/FINDINGS.md) |

These studies address different procedures. A representation-level gain does not establish a benefit from enlarging an already selected forecasting policy. Equal search counts do not guarantee equal feature capacity or computation.

## Archival methods and manuscripts

Earlier manuscript editions are preserved under [papers/soft_state_forecasting/](../papers/soft_state_forecasting/) and [extensions/tmlr_integration/](../extensions/tmlr_integration/). Their source manifests and reporting records identify those editions; they are not the entry point for the focused article.

Other controlled, natural-data, training, adaptation, and exploratory experiments remain under [extensions/](../extensions/). Root `experiment_*.py`, `step_*.py`, `kahkm_*`, `reproduction/`, and `provenance/` belong to earlier Kernel Affine Hull Koopman Machine (KAHKM) work. KAHKM denotes a different architecture from the KAHM local reconstruction used in the central feature study. Targets, coordinate maps, and readouts differ, so results do not transfer automatically between them.

Frozen experiment files and original paths remain part of the provenance and dependency structure. The index separates reading routes without deleting unfavorable evidence or altering the studies' statistical contracts.
