#!/usr/bin/env python3
"""Revise current paper documentation without changing scientific inputs or code."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '472edf8ede85d93d716e2d9db43510e82a8b4623'
DOCS = {}
DOCS['README.md'] = r'''# A Hierarchy of Risk Certificates for Representation-Based Forecasting

**Mohit Kumar and Somayeh Kargaran**

## Prediction-risk bounds

This repository contains the experiments for the paper. We evaluate the minimum mean squared error (MSE) over bounded Lipschitz predictors using a fixed feature representation. Each predictor class specifies a feature map, a distance metric, an output interval, and a Lipschitz constant. A fitted predictor is one member of that class; its error need not equal the minimum error over the class.

For blocks of `b >= 2` independent and identically distributed (iid) observations, the method minimizes total squared error subject to pairwise Lipschitz and output constraints and divides the minimum by `b - 1`. The expected values are nondecreasing lower bounds on minimum population risk and converge to that risk on compact metric feature spaces. This monotonicity does not imply that larger blocks give tighter finite-sample confidence bounds.

Population risk is expected squared prediction error; empirical risk is average squared error on a specified dataset. With iid evaluation observations independent of model development, simultaneous confidence bounds can establish that every predictor in a class exceeds an MSE threshold or that an admissible fitted predictor meets it. Deterministic optimization instead bounds minimum empirical risk on fixed observations.

## Experiments

| Directory | Calculation and result |
| --- | --- |
| [controlled_calibration](experiments/controlled_calibration/) | Compares two confidence-bound procedures using independent synthetic episodes and nested feature sets. |
| [ecg_cohort](experiments/ecg_cohort/) | Excludes a specified two-coordinate Lipschitz predictor class from meeting an empirical MSE threshold on 8,192 PTB-XL patients. |
| [class_diagnosis](experiments/class_diagnosis/) | All 54 Bike Sharing predictor classes admit predictions meeting their empirical thresholds; both fitted predictors fail in 39 classes. |
| [deployment_cost](experiments/deployment_cost/) | Two KAHM-based procedures meet the accuracy allowance in all six cases but fail the combined storage and inference-time requirements. |
| [public_screening](experiments/public_screening/) | Risk bounds avoid no additional predictor fits in the Beijing and Bike Sharing experiments. |
| [decoder_diagnostics](experiments/decoder_diagnostics/) | Reports predictor comparisons and affine constraint ablations, including unsuccessful comparisons. |

The Bike Sharing feasible predictions use evaluation targets. They establish minimum empirical-risk upper bounds, not the future accuracy of independently trained predictors. The [experiment guide](docs/EXPERIMENT_GUIDE.md) includes a worked example and identifies each group's inputs and outputs.

## Reproduction

Run from this branch's repository root with Python 3.12 and the pinned dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce.py --output reproduction
```

To check one group:

```bash
python3 reproduce.py --only ecg_cohort --output reproduction_ecg
```

The driver verifies `MANIFEST.json`, runs the six groups in separate processes, writes results and logs, and rechecks file integrity. `PASS` means that the numerical checks reproduce the stored results, including unsuccessful method outcomes. These commands check stored records; they do not retrain all models or collect new timing measurements. The deployment group also regenerates predictions from stored raw inputs and model parameters.

Use an output directory outside `docs/`, `experiments/`, `scripts/`, and `validation/`, and not the repository root. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the environment, arithmetic assumptions, and verification scope; [DATA.md](DATA.md) for attribution; and [validation/summary.json](validation/summary.json) for the recorded run.

## Source version and licenses

Cite the exact commit used. `MANIFEST.json` identifies the source, documentation, and numerical inputs by SHA-256 hash. The [manuscript experiment index](docs/MANUSCRIPT_ALIGNMENT.md) maps the six groups to the revised paper. Historical protocols and source records retain their original identifiers.

Source and research-output licenses are specified in [LICENSE](LICENSE) and [DATA_LICENSE](DATA_LICENSE). Third-party datasets retain their providers' terms.
'''
DOCS['REPRODUCIBILITY.md'] = r'''# Reproduction of stored experimental results

## Environment and commands

The recorded environment uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, and mpmath 1.3.0. Install the pinned packages and run from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce.py --output reproduction
```

To select one experiment:

```bash
python3 reproduce.py --only ecg_cohort --output reproduction_ecg
python3 reproduce.py --help
```

The driver checks `MANIFEST.json`, runs each selected group in a separate process, stops on a failed check, and verifies the manifest again after completion. The output directory must not be the repository root or lie under `docs/`, `experiments/`, `scripts/`, or `validation/`.

`summary.json` records the source-manifest hash, environment, commands, results, and log locations. Most groups write `<group>.json` and `<group>.log`. `class_diagnosis` writes a subdirectory containing `summary.json`, `class_risk.csv`, and `head_excess.csv`.

## Calculations performed

| Group | Recomputed quantities | Evaluation setting |
| --- | --- | --- |
| `controlled_calibration` | 190,464 stored block optimization records, 1,008 one-sided confidence endpoints, fitted predictions, and decisions | Independent synthetic episodes; numerical safeguards have the scope stated in the group README |
| `ecg_cohort` | 2,688 exact dual objective values, 18 configurations, the MSE threshold, and a necessary lower bound on the Lipschitz constant | Fixed 8,192-patient dataset, conditional on recorded feature bounds |
| `class_diagnosis` | Pairwise distance bounds, feasible predictions, dual objective values, 54 minimum empirical-risk intervals, and 108 empirical excess-risk intervals | Fixed Bike Sharing observations and stored feature coordinates |
| `deployment_cost` | Predictions, exact squared losses, numerical parameter sizes, timing medians, and accuracy/storage/time comparisons | Five replications of one hidden Markov model and one Appliances record |
| `public_screening` | Prediction and optimization records, class exclusions, selected models, and fitted-model counts | Specified Beijing and Bike Sharing datasets |
| `decoder_diagnostics` | Prediction losses, primary comparisons, and 1,440 affine-ablation configurations | Specified forecasting periods |

The controlled group regenerates calibration observations from fixed random streams. The deployment group regenerates forecasts from stored raw inputs and model parameters, using the original comparison tolerances. Its MSE calculations treat stored binary64 targets and forecasts as exact values.

## Numerical assumptions and file integrity

Each group specifies whether stored feature coordinates are exact inputs or are enclosed by numerical intervals. Binary64 values are treated as exact dyadic rationals where required. Exact fractions, outward rounding, and strict or inclusive threshold comparisons follow the corresponding protocol. Targets outside a predictor's output interval remain in the loss calculations.

`MANIFEST.json` covers source code, documentation, and stored numerical inputs. Generated validation reports and reproduction outputs are excluded. The class-diagnosis package also checks its local manifest. Checksums identify file contents; the numerical checkers verify calculations under the stated input assumptions.

## Interpretation

`PASS` means agreement with the stored numerical result, not success of every proposed method. The recorded results include no joint deployment-criterion passes, no additional predictor fits avoided by screening, and mixed or unsuccessful predictor comparisons.

Population-risk confidence bounds use the controlled experiment's independent observations and prespecified calibration procedure. The PTB-XL and Bike Sharing optimization results concern minimum empirical risk on fixed datasets. Related rental targets and reused evaluation periods are not independent experimental replications.

The commands check stored inputs and calculations. They do not repeat every data-acquisition or feature-generation step, retrain all models, or authenticate the original execution history. Timing summaries use saved measurements rather than new benchmarks. Each experiment's source, protocol, and provenance files specify the inputs and execution steps for further regeneration. See the [experiment guide](docs/EXPERIMENT_GUIDE.md) and [data attribution](DATA.md).
'''
DOCS['docs/EXPERIMENT_GUIDE.md'] = r'''# Experimental calculations and outputs

A predictor class fixes the feature map, feature metric, output interval, and Lipschitz constant. For feature distance `d`, admissible predictions differ by at most `kappa * d`. A fitted predictor is one member of this class.

The experiments distinguish **population risk**, the expected squared error under the evaluation distribution, from **empirical risk**, the mean squared error (MSE) on specified observations. Empirical excess risk is a fitted predictor's empirical risk minus the minimum empirical risk over its class.

## Worked example: Bike Sharing

Run from the repository root:

```bash
python3 scripts/explain_bike_certificate.py
python3 scripts/explain_bike_certificate.py --json
```

This standard-library script reads a registered-rentals task with 172 evaluation observations and a 24-hour horizon. The [stored representation](../experiments/class_diagnosis/records/classes/registered/kahm_minmax01_cap4_p1.json), `kahm_minmax01_cap4_p1`, has eight KAHM coordinates. The predictor class uses output interval `[0, 1]` and Lipschitz constant 2 under Euclidean distance.

Targets are normalized using the training minimum and range. Evaluation targets are not clipped; 15 lie outside `[0, 1]`. The threshold is 0.95 times the normalized training MSE of persistence prediction, which repeats the most recent observation.

| Quantity | Normalized MSE interval, rounded for display | Decision |
| --- | --- | --- |
| Threshold | 0.03499354976 | MSE must not exceed this value. |
| Minimum empirical risk | [0.02085061427, 0.02085061427] | A feasible prediction vector meets the threshold. |
| Clipped affine ridge predictor | [0.07055668204, 0.07055668204] | Its MSE lower bound exceeds the threshold. |
| Clipped Lipschitz extension from 32 training examples | [0.05554788133, 0.05554788133] | Its MSE lower bound exceeds the threshold. |

Displayed interval endpoints coincide after rounding, not necessarily before it. The interval for minimum empirical risk has width approximately `1.615e-14`. Decisions use exact rational endpoints and threshold `504811/14425830`.

The script checks the input records against the root manifest and recomputes interval and decision arithmetic. It uses stored feasibility, dual-bound, and predictor-membership checks rather than repeating them. The feasible predictions use evaluation targets and establish an empirical-risk upper bound. The `class_diagnosis` group performs the full distance, feasibility, dual, and excess-risk checks for all 54 classes.

## Experiment groups

| Group for `--only` | Inputs and calculation |
| --- | --- |
| [controlled_calibration](../experiments/controlled_calibration/) | Fixed random streams, observations, features, fitted coefficients, and block optimization records; compare confidence bounds and decisions across calibration sizes. |
| [ecg_cohort](../experiments/ecg_cohort/) | Targets, reference predictions, feature intervals, partitions, and dual variables; compare a minimum empirical-risk lower bound with the MSE threshold. |
| [class_diagnosis](../experiments/class_diagnosis/) | Bike Sharing feature arrays, feasible predictions, and dual variables; bound minimum empirical risk and fitted-predictor excess risk. |
| [deployment_cost](../experiments/deployment_cost/) | Stored inputs, selected model parameters, forecasts, storage inventories, and timing samples; check accuracy, storage, and inference-time requirements. |
| [public_screening](../experiments/public_screening/) | Beijing and Bike Sharing prediction, optimization, and search records; recompute class exclusions, selected models, fitted-model counts, and screening cost. |
| [decoder_diagnostics](../experiments/decoder_diagnostics/) | Predictions, targets, comparison thresholds, and affine-ablation configurations; recompute each experiment's specified comparison. |

## Commands and outputs

Use Python 3.12 and run from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce.py --help
python3 reproduce.py --only ecg_cohort --output reproduction_ecg
python3 reproduce.py --output reproduction_all
```

Choose a separate output directory, not the repository root or a directory under `docs/`, `experiments/`, `scripts/`, or `validation/`. The driver checks the source manifest before and after the selected calculations.

For the ECG command, `reproduction_ecg/summary.json` records the source, environment, and status; `ecg_cohort.json` contains numerical results; and `ecg_cohort.log` contains the checker output. Other groups use their own names. The class-diagnosis subdirectory contains `class_risk.csv` for all 54 minimum empirical-risk intervals and thresholds, and `head_excess.csv` for all 108 predictor-risk and excess-risk intervals.

## Interpreting the results

For the controlled experiment, a lower confidence bound above the threshold excludes every predictor in a class. An upper confidence bound at or below the threshold establishes that the corresponding fitted predictor meets it. Otherwise the bounds leave the decision unresolved. The guarantee requires the stated sampling, class-membership, and numerical assumptions.

The ECG calculation uses one history--target pair per patient and recorded feature bounds. Its separately selected 2-, 6-, and 14-coordinate representations are not nested. Excluding the two-coordinate class does not exclude all KAHM representations. The larger classes remain unresolved.

The Bike Sharing analysis establishes empirical feasibility for all 54 classes; both fitted predictors fail in 39. The feasible predictions use evaluation targets, and the three related rental targets are not independent application replications.

The deployment procedures fail the combined accuracy, storage, and inference-time requirement in every case. Screening avoids no additional predictor fits. `PASS` preserves these reported outcomes. Saved timing measurements are not new measurements on the machine running the checker.

The commands verify stored records, not complete model-training runs. The controlled group regenerates calibration observations; the deployment group also regenerates forecasts from stored inputs and models. Each experiment README specifies its remaining inputs and checks. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for numerical assumptions and [DATA.md](../DATA.md) for dataset attribution.
'''
DOCS['docs/MANUSCRIPT_ALIGNMENT.md'] = r'''# Experiments retained in the revised manuscript

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
'''
DOCS['experiments/controlled_calibration/README.md'] = r'''# Population-risk confidence bounds with independent episodes

## Design and results

The experiment compares two lower-confidence-bound procedures for minimum population mean squared error (MSE) over a specified Lipschitz predictor class. The first averages normalized minimum-loss bounds from disjoint observation blocks. The second uses feasible dual values from those blocks to bound minimum empirical loss over the full calibration dataset before statistical calibration.

Each of two bounded synthetic settings has 12 complete training repetitions and two nested feature sets per repetition. At 1,024 calibration observations, both procedures resolve all 12 repetitions in each setting. At 256 observations, reuse of the dual values excludes both classes in 11 additional repetitions when the target is independent of the history. Neither procedure resolves the history-dependent setting at that size. Outputs include all three calibration sizes and unresolved decisions.

## Reproduction

Run from the repository root:

```bash
python experiments/controlled_calibration/replay.py --output results/controlled_calibration.json
```

The command regenerates 4,096 calibration observations for each learning case from fixed random streams, recomputes selected fitted predictions, evaluates 190,464 stored block optimization records, and reconstructs 1,008 one-sided confidence endpoints and paired decision counts. It does not call an optimizer or refit models. Inputs use NumPy arrays and JSON, not pickles.

## Files and numerical assumptions

`records/` contains dual vectors, candidate feasible predictions, numerical diagnostics, features, observations, fitted coefficients, and expected endpoints. `protocol/` contains the fixed design. `source/` retains the data-generation code, independent checker, model code, and numerical routines. `provenance/` records source identities and checksums. Original source identifiers are unchanged.

The calculations use floating-point safeguards and recorded feature checks. Reproduction does not bound every feature-generation or optimization rounding error and does not repeat feature construction. The original archive, `geometry_step17_reproducibility.zip`, also contains all training candidates and coordinate-wise feature bounds.

Inputs are synthetic. The checker requires Python, NumPy, and SciPy and makes no network requests. Original training additionally uses scikit-learn and the recorded numerical software.
'''
DOCS['experiments/ecg_cohort/README.md'] = r'''# Minimum empirical-risk lower bounds on PTB-XL

## Design and results

The experiment bounds minimum mean squared error (MSE) over specified Lipschitz predictor classes on a fixed dataset of 8,192 patients. It evaluates separately selected 2-, 6-, and 14-coordinate feature maps, two partition rules, and block sizes 32, 64, and 128. The feature maps are not nested.

The two-coordinate class has a minimum empirical-risk lower bound above the specified MSE threshold. The six- and fourteen-coordinate classes remain unresolved. A separate calculation gives a necessary lower bound on the Lipschitz constant; exceeding that constant alone does not establish that the class meets the MSE threshold.

## Reproduction

Run from the repository root:

```bash
python experiments/ecg_cohort/replay.py --output results/ecg_cohort.json
```

The checker reconstructs partitions and outward integer distance bounds from recorded feature intervals. It evaluates all 2,688 stored dual vectors using exact integer/rational arithmetic. Each block uses the maximum of its dual lower bound, lower bounds combined from smaller blocks, and zero. The command checks all 18 configurations and recomputes the MSE threshold from targets and reference predictions. It requires Python and NumPy; no solver, model fitting, waveform download, or reserved-patient access occurs.

## Files and numerical assumptions

`records/` contains targets, predictions, coordinate-wise feature intervals, dual vectors, row indices, and exact block values. Redundant distance matrices and duplicate dual records are reconstructed. `protocol/` contains the dataset and optimization designs. `source/` retains data-generation code, exact evaluators, and independent checking code; `provenance/` records source identities and checksums.

The bounds are deterministic results for these observations, conditional on the recorded feature intervals. The command does not regenerate features or establish future-patient risk guarantees. Original fitting and feature-generation records are in `step19_reproducibility.zip`.

## Data attribution

The data are [PTB-XL version 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), distributed by PhysioNet under CC BY 4.0, DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45). Wagner et al. describe the dataset in *Scientific Data* 7, 154 (2020), DOI [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).
'''
DOCS['experiments/class_diagnosis/README.md'] = r'''# Minimum empirical risk and fitted-predictor error on Bike Sharing

## Design and results

The experiment evaluates 54 Lipschitz predictor classes for casual, registered, and total rentals. Each target has 172 fixed evaluation observations. A class specifies stored feature coordinates, output interval `[0, 1]`, and Lipschitz constant 2 under Euclidean distance. Targets outside the output interval remain in the losses.

All 54 classes contain feasible prediction vectors meeting their mean squared error (MSE) thresholds. Both fitted predictors fail in 39 classes. All 108 fitted predictors have a strictly positive lower bound on empirical excess risk: their empirical MSE minus the minimum empirical MSE over their class. Target-based feasible predictions also meet all 54 thresholds without an optimizer proposal; constant predictions meet none. The maximum interval width for minimum empirical risk is approximately `2.181e-14` in normalized MSE.

## Reproduction

Use Python 3.10 or newer and NumPy. Run from the repository root:

```bash
python -m pip install -r experiments/class_diagnosis/requirements.txt
python experiments/class_diagnosis/replay.py --output /tmp/class-diagnosis-replay
```

The command checks package and input hashes, reconstructs 794,124 pairwise distance bounds, checks three feasible prediction candidates per class, evaluates dual lower bounds, and recomputes all 54 minimum empirical-risk intervals and 108 empirical excess-risk intervals. It uses an independent arithmetic checker and does not run an optimizer or fit models.

| Output | Contents |
| --- | --- |
| `summary.json` | Verification counts, per-target results, exact checks, and numerical assumptions |
| `class_risk.csv` | 54 minimum empirical-risk intervals and MSE thresholds |
| `head_excess.csv` | 108 predictor-risk intervals and recomputed empirical excess-risk intervals |

## Interpretation and numerical assumptions

Feasible predictions use evaluation targets. They establish what a class can achieve on those observations, not the accuracy of a predictor trained independently of those targets. Total rentals equal casual plus registered rentals, so the three targets are not independent application replications.

Stored binary64 feature coordinates are treated as exact inputs. The checker verifies distance bounds, feasibility, dual objectives, and empirical-risk arithmetic under this assumption. It does not bound feature-construction error or identify the cause of a fitted predictor's excess risk.

Predictor membership and achieved-risk checks come from the public-screening analysis. This package verifies their file hashes and recomputes excess risk; it does not regenerate those fitted predictions. Checksums establish file consistency, not independent authentication of execution history.

## Files and provenance

| Path | Contents |
| --- | --- |
| `inputs/decision/` | Three evaluation datasets, 54 feature arrays, representation identities, thresholds, and predictor metadata |
| `inputs/prior/` | Public-screening protocols, model hashes, numerical checks, and data-source metadata |
| `records/classes/` | 54 complete-graph primal and dual optimization records |
| `records/result.json` | Original aggregate result, including 108 predictor rows |
| `records/original_numerical_audit.json` | Original independent numerical checks |
| `protocol/` | Protocol, checksums, and software requirements |
| `source/` | Data-generation, input-preparation, and independent checking code |
| `MANIFEST.json` | Package file sizes and SHA-256 hashes |

The data are [Bike Sharing, UCI dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing), by Hadi Fanaee-T (2013), DOI `10.24432/C5W894`. `replay.py` is the supported offline entry point. Regenerating features and fitted predictors from raw data requires the public-screening source and data.
'''
DOCS['experiments/deployment_cost/README.md'] = r'''# Prediction accuracy, model storage, and inference time

## Design and decision criteria

Two fixed KAHM-based model-selection procedures are compared with conventional accuracy and cost references. The six cases comprise five replications of one hidden Markov model and one chronological Appliances record, not six independent application domains. Primary forecast horizons are four steps for the synthetic process and six for Appliances. The records contain 23 distinct selected models; the two references coincide in one case.

`P_cost` selects predictors using normalized KAHM affinities; `CS_cost` uses clipped reconstruction scores. `C_accuracy` and `C_cost` are the conventional accuracy and cost references. Candidates and references are fixed before evaluation. A procedure must satisfy all three criteria:

| Quantity | Requirement |
| --- | --- |
| Mean squared error (MSE) | At most 1.1 times the MSE of `C_accuracy` |
| Numerical parameter storage | Strictly fewer bytes than `C_cost` |
| Median inference time per observation | No greater than `C_cost`, measured in batches after warm-up |

Both procedures meet the accuracy requirement in all six cases. Neither meets the storage or inference-time requirement in any case. Both therefore fail the combined criterion in all six cases.

## Reproduction

Run from the repository root:

```bash
python experiments/deployment_cost/replay.py --output results/deployment_cost.json
```

To regenerate predictions from stored raw inputs and model parameters:

```bash
python experiments/deployment_cost/replay.py --predict --output results/deployment_cost_predictions.json
```

The checker recomputes exact squared losses from binary64 targets and forecasts, numerical parameter sizes, and latency medians from the original nine-repeat timing samples. It runs offline with NumPy, SciPy, scikit-learn, and threadpoolctl.

## Files and measurement scope

`data/` contains primary-horizon evaluation rows, selected parameters, forecasts, expected risks, timing samples, and procedure identifiers. `source/` retains the protocol, model implementation, selection procedures, and evaluation code. `source_provenance.json` and per-task records identify the original archive and files.

Storage includes representation and predictor arrays and scalars, not Python-object or runtime-library overhead. It is not process-memory usage. Timing summaries use saved measurements rather than new benchmarks. Prediction regeneration evaluates fixed models without refitting or reselection; the comparison concerns the specified observations rather than a population-risk noninferiority guarantee. It does not verify exact real-arithmetic feature values.

Serialized models contain parameter dictionaries and arrays. Verify their integrity before loading them.
'''
DOCS['experiments/public_screening/README.md'] = r'''# Predictor-class screening on Beijing and Bike Sharing data

## Design

The experiment tests whether lower bounds on minimum empirical risk exclude predictor classes or reduce model fitting under fixed mean squared error (MSE) and storage requirements. Beijing PM2.5 uses 18 representations with two admissible fitted predictors each. Bike Sharing uses the same 36-predictor configuration separately for casual, registered, and total rentals. These targets satisfy `cnt = casual + registered` and are not independent datasets.

## Reproduction

Run from the repository root with the pinned dependencies:

```bash
python reproduce.py --only public_screening --output reproduction_screening
```

The group also has a direct entry point:

```bash
python experiments/public_screening/replay.py --output reproduction_screening/public_screening.json
```

The commands use bundled archives and numerical records without network access or model fitting.

## Calculations performed

The checker reconstructs evaluation indices and lag matrices using only observations preceding each target. It checks training-only target normalization and thresholds equal to 0.95 times exact training-persistence MSE. Evaluation targets outside the predictor output interval remain in the losses.

For all 144 fitted predictors, it checks class membership, reconstructs decision and test predictions from stored features and fixed parameters, and recomputes squared-loss bounds. All 288 prediction records must match their original hashes. Affine predictions use rational arithmetic; Lipschitz-extension distances use exact squared dyadic distances and outward integer-square-root bounds. Predictor storage is recomputed; total storage uses the recorded representation sizes.

For Beijing, the checker reconstructs distance tables and verifies all 396 block optimization records, including primal feasibility and dual objectives. It recomputes all 36 decision/test minimum empirical-risk intervals, class exclusions, candidate eligibility, cost-based selections, and recorded screening costs. Aggregated upper bounds must correspond to one admissible predictor over the complete evaluation dataset.

For Bike Sharing, the checker verifies exact two-observation optima and dual values, disjoint-pair aggregation, deterministic matching, storage lower bounds available before fitting, and search transitions. The records contain 12 distinct pairwise bounds used in 60 calls and 33 search executions: one exhaustive run and five ordinary/triggered search pairs per target. It recomputes fitted-model counts, selections, timing medians, and the specified additional-cost criterion.

## Results

Beijing screening excludes no classes and avoids no predictor fits. It selects the same predictor as exhaustive search under the same selection rule and adds approximately 18.90 seconds of recorded bound-computation time. An inconclusive lower bound does not establish that the class meets the threshold.

For Bike Sharing, ordinary and triggered searches retain the exhaustive selection for all three targets. Each fits two predictors for casual rentals and 36 for each of registered and total rentals in every recorded repetition. The triggered procedure avoids no additional fits. It makes no pairwise-bound calls for casual rentals and six per repetition for each other target. The specified additional-cost criterion fails across the evaluated tasks.

## Files and numerical assumptions

`witnesses/` contains losslessly compressed numerical inputs and selected original records. `source_inventory.json` identifies original paths and hashes. `original/` contains source code, protocols, source-data archives, and evaluation records. The checker reconstructs omitted duplicate prediction and distance records and verifies their hashes.

Stored features are treated as exact dyadic numbers. The results concern fixed observations and prediction rules; feature-generation rounding error and model refitting are outside this calculation. Representation sizes and timing samples are recorded inputs. Timing summaries do not remeasure or independently authenticate the original runtimes, and the numerical checks do not establish future-observation guarantees.
'''
DOCS['experiments/decoder_diagnostics/README.md'] = r'''# Predictor comparisons and affine constraint ablations

## Experiments and results

These experiments compare predictor families and hyperparameter-selection rules on specified forecasting periods.

| Experiment | Data | Result |
| --- | --- | --- |
| Lipschitz extension using all training observations | Seoul | Improves the baseline predictor but fails the conventional-reference requirement. |
| Temporal regularization and extension selection | Tetouan and Metro | Both primary comparisons fail. |
| Joint selection of affine and Lipschitz-extension predictors | Household and Jena | Household passes; Jena fails; the combined criterion fails. |
| Constrained affine forecasting | Specified Household and Jena periods | Both fail the requirement of MSE at most 1.05 times the selected conventional reference. |
| Coefficient projection, intercept recentering, and clipping ablation | The same Household and Jena periods | All 720 configurations per task are retained. |

Each ablation compares an option's off/on settings while holding the other choices fixed. Equality, improvement, and worsening counts use exact rational mean squared errors (MSEs) in the original target units. Exact rational reference-prediction losses are checked separately. Small recentering differences can fall below the original display threshold. These periods were used during model development, so the ablation does not independently validate a selection rule.

## Reproduction

Run from the repository root:

```bash
python experiments/decoder_diagnostics/replay.py --output results/decoder_diagnostics.json
```

The command checks 2,930 stored predictor-risk records, including losses in original target units and exact rational reference losses for all 1,440 affine-ablation configurations. These records are not 2,930 independent experiments. It uses the Python standard library and makes no network requests.

For interval-valued predictions, the checker minimizes and maximizes squared loss over each interval before averaging. Binary64 outputs are treated as exact rational values where specified. Comparisons use the candidate's MSE upper bound and the reference's MSE lower bound with the experiment's stated allowance. Overlapping bounds alone do not establish success. Identical selected parameters count as equality when permitted by the criterion.

## Files and verification scope

`data/` contains evaluation targets, prediction intervals or binary64 outputs, expected risks, comparison criteria, and all ablation configurations. `source/` retains the scientific code and protocols. `source_provenance.json` identifies the original archive and files.

The checker recomputes loss arithmetic and fixed comparisons. It does not repeat feature construction, model fitting, optimizer proposals, development selection, or the original numerical-output checker. Results concern the specified predictors and periods, not population-risk confidence bounds or demonstrated training-cost savings.
'''

EXPECTED = {
 'README.md':'5140a78c2e59f88aed9855789a182e7de02863c5',
 'REPRODUCIBILITY.md':'4062d191e5eda1ec89abe3f711edb897aa062086',
 'docs/EXPERIMENT_GUIDE.md':'d127bb76e32668ee1a474b2b235cf2cf8ed06954',
 'docs/MANUSCRIPT_ALIGNMENT.md':'052215e7387170d89dda564b5fe7f73fd776455f',
 'experiments/controlled_calibration/README.md':'86ff19aa974228b5575e25205db78a6294630a63',
 'experiments/ecg_cohort/README.md':'4457f1dd76f463c52b226fd647629a84389f392d',
 'experiments/class_diagnosis/README.md':'e07c9da5cca10b1be992527c49cbe1eba64f6981',
 'experiments/deployment_cost/README.md':'52bde229534313f1909cdbc0e87c29590869d3fb',
 'experiments/public_screening/README.md':'16fa54d51dec142f937ecc8eace60aa58cf705d3',
 'experiments/decoder_diagnostics/README.md':'0218adc5c9a1af462e32dc23eb5ef02ce0f1e119',
 'DATA.md':'97071131470961a448eaafac1237e982cd79b17e',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def blob_digest(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

original_manifest = json.loads((ROOT/'MANIFEST.json').read_text())
original_manifest_sha = digest(ROOT/'MANIFEST.json')
for row in original_manifest['files']:
    p=ROOT/row['path']
    if not p.is_file() or p.stat().st_size != row['bytes'] or digest(p) != row['sha256']:
        raise RuntimeError('Original input integrity failed: '+row['path'])
for name, sha in EXPECTED.items():
    if blob_digest((ROOT/name).read_bytes()) != sha:
        raise RuntimeError('Unexpected documentation version: '+name)

# Preserve all dataset links, identifiers, attribution, and license terms.
data=(ROOT/'DATA.md').read_text()
data=data.replace('The experiments retain the original study-specific partitions, normalizations, target definitions, selected model identities, and scientific source records. The table below identifies the datasets and their providers.', 'Each experiment retains its original data split, normalization, target definition, selected models, and source records. The table identifies the datasets used by the six retained experiment groups.')
data=data.replace('| ECG evaluation dataset |','| PTB-XL empirical-risk bounds |').replace('| Public screening and class diagnosis |','| Screening and minimum empirical-risk analysis |').replace('| Predictor diagnostics |','| Predictor comparisons and ablations |')
data=data.replace('The ECG package contains the specified 8,192-patient evaluation dataset. Reserved patients are not included. The experiment documentation distinguishes population-risk conclusions, results on a fixed dataset and numerical guarantees conditional on stored inputs.', 'The ECG package contains the fixed 8,192-patient evaluation dataset and excludes reserved patients. Each experiment README specifies its observations, numerical input assumptions, and verification procedure.')
DOCS['DATA.md']=data

stats={}
for name, content in DOCS.items():
    p=ROOT/name
    previous=p.read_text()
    content=content.strip()+'\n'
    stats[name]={'words_before':len(previous.split()),'words_after':len(content.split()),'bytes_before':len(p.read_bytes()),'bytes_after':len(content.encode())}
    p.write_text(content,encoding='utf-8')

# Update only the documentation entry in the package-local manifest.
local=ROOT/'experiments/class_diagnosis/MANIFEST.json'
m=json.loads(local.read_text())
r=ROOT/'experiments/class_diagnosis/README.md'
m['files']['README.md']['bytes']=r.stat().st_size
m['files']['README.md']['sha256']=digest(r)
local.write_text(json.dumps(m,indent=2)+'\n')

allowed=set(DOCS)|{'experiments/class_diagnosis/MANIFEST.json'}
unchanged=0
for row in original_manifest['files']:
    p=ROOT/row['path']
    if row['path'] not in allowed:
        if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
            raise RuntimeError('Scientific file changed: '+row['path'])
        unchanged+=1
    row['bytes']=p.stat().st_size
    row['sha256']=digest(p)
(ROOT/'MANIFEST.json').write_text(json.dumps(original_manifest,indent=2)+'\n')

# Local link checks apply to current guides, not preserved historical protocols.
links=0
for name in DOCS:
    p=ROOT/name
    for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^\s)]+)\)',p.read_text()):
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):
            continue
        dest=(p.parent/target.split('#',1)[0]).resolve()
        if not dest.is_relative_to(ROOT) or not dest.exists():
            raise RuntimeError('Broken local documentation link: '+name+' -> '+target)
        links+=1

# No added scientific files and no deleted experiment paths.
for directory in ['affine_forecasting','single_trajectory_certification']:
    if (ROOT/'experiments'/directory).exists():
        raise RuntimeError('Removed experiment unexpectedly present: '+directory)
for name in DOCS:
    text=(ROOT/name).read_text()
    if name!='docs/MANUSCRIPT_ALIGNMENT.md' and ('experiments/affine_forecasting' in text or 'experiments/single_trajectory_certification' in text):
        raise RuntimeError('Obsolete experiment reference in active guide: '+name)

report={
 'status':'PASS',
 'scope':'Documentation, checksum updates, local links, and preservation of scientific code and numerical inputs. Numerical reproduction is recorded separately in summary.json.',
 'base_commit':BASE,
 'original_manifest_sha256':original_manifest_sha,
 'source_manifest_sha256':digest(ROOT/'MANIFEST.json'),
 'source_manifest_files':len(original_manifest['files']),
 'documentation_files':stats,
 'words_before':sum(v['words_before'] for v in stats.values()),
 'words_after':sum(v['words_after'] for v in stats.values()),
 'unchanged_manifest_files':unchanged,
 'scientific_code_and_numerical_inputs_unchanged':True,
 'local_documentation_links_checked':links,
 'root_and_class_diagnosis_manifests_updated':True,
 'historical_protocols_and_provenance_unchanged':True,
}
(ROOT/'validation/documentation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
