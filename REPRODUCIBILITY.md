# Reproduction tasks and available materials

Each experiment has its own source identity, dependencies, observations, and statistical analysis. The [research index](docs/RESEARCH_INDEX.md) links its scientific guide. Agreement with saved evidence establishes the specified computational checks; it does not add independent experimental observations.

## Reconstruction-affinity feature augmentation

This branch includes the exact 40-row primary summary for protocol `KAHM-SCORE-20261102-v1`. With SciPy installed, run from the repository root:

```bash
python extensions/score_enrichment_20261102/verify_primary_summary.py
```

The checker verifies the CSV identity, comparison roster, sample sizes, Student critical value, standard errors, confidence endpoints, effect ratios, and decision counts from the reported means and standard deviations. It does not reconstruct repetition-level differences or rerun a model.

The [central saved-loss bundle](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02/reproduction/score_enrichment) provides a separate check using 17 original evidence records. Its independently implemented checker reads 26,624 saved selected-loss rows and reconstructs 416 procedure summaries, 2,560 paired differences, and all 40 primary comparisons. It cannot recalculate losses from predictions or verify the complete validation search because the required inputs are absent. Use the bundle's pinned dependencies and commands in its README.

The original generator, experiment runner, complete source snapshot, `cohort.json` evaluation roster and random-stream labels, fitted models, forecast arrays, and environment fingerprint are unavailable here. The source-manifest SHA256 recorded by the experiment is `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`. A source hash identifies the expected bytes; it does not supply them.

The experiment records describe sixteen numerical archives totaling 9.024 GiB. The complete set and a public download endpoint are unavailable, so the recorded total has not been independently verified from all archive bytes. Archive volume is a distribution constraint separate from the missing implementation, evaluation roster, and environment. Complete source and execution inputs would permit regeneration without distributing every saved model state.

The recorded original environment uses Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, and pytest 9.0.2. The saved-loss checker uses its own pinned Python 3.12 environment. The root requirements serve the root KAHKM experiments and are not a replacement for either environment.

## Four-representation local-position experiment

The [study guide](extensions/soft_state_ablation_20261102/README.md) describes the separate 32-repetition design. Restore and verify its three compact records with:

```bash
python extensions/soft_state_ablation_20261102/reporting/restore_compact_records.py --check-only
python extensions/soft_state_ablation_20261102/reporting/restore_compact_records.py
```

The [reporting guide](extensions/soft_state_ablation_20261102/reporting/README.md) gives statistical and saved-state checks, and the [protocol](extensions/soft_state_ablation_20261102/PROTOCOL.md) defines the source and execution requirements. Full regeneration requires the inherited dimension-study runtime and policy-study numerical solver in addition to this extension. Numerical replay requires the separately indexed data, forecasts, and fitted-state archives. An archive inventory is not a download service.

## Supporting experiments

| Study | Reproduction guide |
| --- | --- |
| Controlled representations and predictor capacity | [Predictive geometry](extensions/predictive_geometry/REPRODUCING.md) |
| Validation-selected dimension | [Dimension forecasting](extensions/dimension_forecasting/README.md) |
| Reconstruction departures | [Reconstruction mechanism](extensions/reconstruction_mechanism/README.md) |
| Forecast-trained reconstruction parameters | [Direct reconstruction](extensions/direct_reconstruction/README.md) |
| Supervised metrics and residual targets | [Target geometry](extensions/target_geometry/README.md) |
| Online updates | [Adaptive development](extensions/adaptive_development/REPRODUCING.md) |
| Policy augmentation | [Synthetic experiment](extensions/policy_augmentation_20261101/README.md) and [natural series](extensions/policy_augmentation_natural_20261101/REPRODUCTION.md) |
| Root KAHKM experiments | [Reproduction evidence](reproduction/REPRODUCIBILITY_EVIDENCE.md) |

Use new output directories for repeated computation. Preserve recorded results and original protocol identities. Reanalysis of saved losses, replay of saved fitted states, model refitting, and experiments on new independent data answer different questions.
