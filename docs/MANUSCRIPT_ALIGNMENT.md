# Alignment with the revised JMLR manuscript

This source tree supports **A Hierarchy of Risk Certificates for Representation-Based Forecasting**, by Mohit Kumar and Somayeh Kargaran, after removal of the single-stationary-trajectory contribution and the separate affine numerical-output appendix.

## Retained experiments

| Directory under `experiments/` | Manuscript evidence |
| --- | --- |
| `controlled_calibration` | Simultaneous population-risk confidence bounds using independent episodes |
| `ecg_cohort` | Empirical-risk lower bounds for the fixed PTB-XL dataset |
| `class_diagnosis` | Bike Sharing empirical-risk bounds for 54 predictor classes and 108 fitted predictors |
| `deployment_cost` | Forecast accuracy, storage, and inference time for the retained deployment comparisons |
| `public_screening` | Beijing and Bike Sharing predictor-class screening and fit-avoidance results |
| `decoder_diagnostics` | Retained predictor comparisons and Household 2010/Jena 2017 affine constraint ablations |

The source, inputs, and recorded outcomes inside these six experiment directories are unchanged. In particular, the affine constraint ablations remain: they are separate from the deleted three-period numerical-output experiment. Household, Jena, and Metro data used by retained predictor comparisons are also preserved.

## Removed material and source history

The current paper tree excludes `experiments/affine_forecasting` and `experiments/single_trajectory_certification`, their active reproduction commands, and their validation outputs. Historical commits are preserved for provenance; this cleanup does not rewrite Git history or remove material belonging to the separate feature-augmentation paper.

The consolidated source before cleanup was commit `8878ae4ebe922c0184aed1e0fbb2afbb52c304f3`. The former trajectory branch was at `02589f72a5694dfa89f1d37f89de105780b0872a`. All six retained experiment directories had identical Git tree hashes in those two revisions.

## Verification and citation

`MANIFEST.json` lists the SHA-256 hashes of the retained source, documentation, and numerical inputs. `validation/summary.json` records the numerical replay actually performed for the manifest identified in that report. A replay is a check of stored records, not a new model-training run or a new timing study.

Use the exact cleaned commit in the manuscript's repository links. The previously cited commit `4c09d24a27837d7f165b253f3ab4d3cf28b71c0e` is a historical snapshot and still includes the deleted affine experiment.

This alignment follows the uploaded manuscript source and the subsequent section and appendix deletions agreed in the manuscript revision. The final 49-page source was not available for a new line-by-line manuscript-to-code audit.
