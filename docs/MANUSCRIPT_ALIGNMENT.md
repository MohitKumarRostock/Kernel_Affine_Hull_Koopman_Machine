# Experiments retained in the revised manuscript

This source tree accompanies **A Hierarchy of Risk Certificates for Representation-Based Forecasting**, by Mohit Kumar and Somayeh Kargaran, after removal of the single-stationary-trajectory analysis and the separate affine numerical-output appendix.

| Directory under `experiments/` | Manuscript result |
| --- | --- |
| `controlled_calibration` | Population-risk confidence bounds using independent episodes |
| `ecg_cohort` | Minimum empirical-risk lower bounds on fixed PTB-XL observations |
| `class_diagnosis` | Minimum empirical-risk and fitted-predictor excess-risk bounds for 54 Bike Sharing classes |
| `deployment_cost` | Prediction accuracy, storage, and inference time |
| `public_screening` | Beijing and Bike Sharing class exclusions and fitted-model counts |
| `decoder_diagnostics` | Predictor comparisons and Household 2010/Jena 2017 constraint ablations |

Scientific code, numerical inputs, and recorded experimental outcomes are preserved. Current documentation and its checksum entries are revised separately. The retained Household/Jena ablations are distinct from the deleted numerical-output study; Household, Jena, and Metro data needed by retained comparisons remain available.

## Source history

The active tree excludes `experiments/affine_forecasting` and `experiments/single_trajectory_certification`, their reproduction commands, and their validation outputs. Historical commits and the separate feature-augmentation study are preserved.

Before cleanup, the consolidated branch pointed to `8878ae4ebe922c0184aed1e0fbb2afbb52c304f3` and the trajectory branch to `02589f72a5694dfa89f1d37f89de105780b0872a`. The six retained experiment directories had identical Git tree hashes at those revisions. The cleanup commit is `472edf8ede85d93d716e2d9db43510e82a8b4623`.

## Verification and citation

Cite the exact commit used for reproduction. The earlier manuscript link, `4c09d24a27837d7f165b253f3ab4d3cf28b71c0e`, identifies a historical snapshot that still contains the removed affine experiment.

`MANIFEST.json` lists source, documentation, and input hashes. `validation/summary.json` records the numerical checks for its stated manifest. Reproduction checks stored records; it does not repeat all training or timing experiments.

This index follows the reviewed manuscript source and the agreed deletions. The final 49-page source was not supplied for a new line-by-line manuscript-to-code comparison.
