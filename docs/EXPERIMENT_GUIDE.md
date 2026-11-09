# Experiments on lower bounds for prediction risk

A forecast error describes one fitted predictor. A class-risk bound asks whether any predictor using a specified representation, distance, Lipschitz bound, and output range could meet an error requirement. The experiments separate this question from two others: whether a selection rule improves deployment costs, and whether a stored prediction is evaluated accurately.

The Lipschitz bound limits how predictions may change: if two feature vectors are distance `d` apart, their predictions may differ by at most `kappa * d`. The declared class contains all predictors satisfying this rule and the output-range constraint. An admissible fitted predictor is one member whose coefficients or other parameters have been chosen from training data. Failure of that member need not imply failure of the whole class.

This guide maps those questions to the retained inputs and executable checks. Run all commands from the repository root. The [reproduction guide](../REPRODUCIBILITY.md) records the environment and numerical assumptions; the [data guide](../DATA.md) identifies the original datasets and provenance.

## Start with a fixed-dataset worked example

```bash
python3 scripts/explain_bike_certificate.py
python3 scripts/explain_bike_certificate.py --json
```

This explanation uses only the Python standard library. It reads one registered-rentals forecasting case with 172 assessed observations and a 24-hour horizon. The chosen Kernel Affine Hull Machine (KAHM) representation, `kahm_minmax01_cap4_p1`, has eight stored coordinates. The class has output range `[0, 1]` and Lipschitz constant 2 with respect to Euclidean distance. Its [saved class record](../experiments/class_diagnosis/records/classes/registered/kahm_minmax01_cap4_p1.json) and associated protocol and evaluation dataset records are checked against the root manifest.

Here, **class risk** is the best achievable average squared error within the declared class on these observations. A fitted predictor's **excess risk** is its average squared error minus that best class risk.

Targets are normalized by subtracting the training minimum and dividing by the training range. Assessment targets remain unclipped: 15 lie outside `[0, 1]`, although predictions are constrained to that interval. The tolerance is fixed at 0.95 times the normalized training MSE of the persistence forecast, which repeats the latest observed value.

| Quantity | Normalized MSE, rounded for display | Conclusion from the exact numbers |
| --- | --- | --- |
| Tolerance | 0.03499354976 | The requirement is risk at most this value. |
| Best class risk | [0.02085061427, 0.02085061427] | The class upper bound meets the requirement. |
| Clipped affine ridge predictor | [0.07055668204, 0.07055668204] | Its risk lower bound exceeds the tolerance. |
| Clipped Lipschitz extension from 32 training examples | [0.05554788133, 0.05554788133] | Its risk lower bound exceeds the tolerance. |

Both endpoints of each interval round to the same displayed value; they are not asserted to be equal. The class-risk bracket has width about `1.615e-14`. All decisions use the exact rational endpoints and the exact tolerance `504811/14425830`. The class therefore admits sufficiently accurate predictions on these observations even though both fitted predictors fail their requirement and have positive excess risk.

The script evaluates the stored interval and decision arithmetic with exact rational numbers. It uses the saved checks of feasibility, dual bounds and fitted-predictor class membership; it does not repeat those checks or train a model. The feasible class prediction uses assessment targets and establishes capacity on this fixed dataset, not future-prediction performance. The `class_diagnosis` replay below performs the fuller numerical checks across all 54 classes.

## Choose the scientific question

| Question | Group for `--only` | Retained inputs and what to inspect |
| --- | --- | --- |
| When do calibrated bounds support feature augmentation or identify a class that cannot meet the error requirement? | `controlled_calibration` | [Calibration records](../experiments/controlled_calibration/records/) contain fixed random streams, observations, features, coefficients, candidate predictions, and optimization lower-bound records. Inspect the lower and upper endpoints and decisions at each calibration size. |
| Can one class be excluded on an already specified patient evaluation dataset? | `ecg_cohort` | [Evaluation data](../experiments/ecg_cohort/records/) contain targets, reference predictions, feature enclosures, partitions, and optimization lower-bound records. Inspect the class lower bound relative to the specified tolerance. |
| Can a class meet its dataset requirement even when its fitted predictors fail? | `class_diagnosis` | [Decision inputs](../experiments/class_diagnosis/inputs/decision/) and [class-bound records](../experiments/class_diagnosis/records/classes/) support risk brackets and fitted-predictor excess bounds on the Bike observations. |
| Does a selected procedure meet accuracy, storage, and latency requirements together? | `deployment_cost` | [Deployment data](../experiments/deployment_cost/data/) contain selected models, forecasts, model-storage inventories, and original timing samples. Inspect each component of the joint criterion. |
| Do class bounds exclude candidates or avoid predictor fits? | `public_screening` | [Screening records](../experiments/public_screening/witnesses/) and [original inputs](../experiments/public_screening/original/) retain the Beijing and Bike decision, prediction, and search records. Inspect exclusions, additional fits avoided, and certificate cost. |
| Do alternative prediction families or selection rules improve the recorded comparisons? | `decoder_diagnostics` | [Assessment data](../experiments/decoder_diagnostics/data/) retain selected predictions, targets, criteria, and all affine-ablation configurations. Inspect each study's own primary requirement. |
| Are the selected stored affine forecasts evaluated within their numerical error budgets? | `affine_forecasting` | [Numerical records](../experiments/affine_forecasting/witnesses/) contain development predictions, fixed models, supplied features, query identities, output checks, assessment targets, and timing samples. Numerical error and forecast error are separate quantities. |

Each group has a README with its full protocol, expected outcomes, and original-source paths. These groups assess different observations and claims; their rows are not interchangeable independent replications of one experiment.

## Certification from one stationary trajectory

The [single-trajectory study](../experiments/single_trajectory_certification/) assesses simultaneous class-risk and fitted-risk bounds under a known dependent data-generating law. Its empirical loss uses all retained forecasting windows, while the confidence calculation uses the number of separated raw blocks. A dependence allowance covers the boundary after development and the boundaries between those blocks. The [methods](../experiments/single_trajectory_certification/METHODS.md) define the process, feature maps, sensitivity constraints, confidence allocations, numerical calculations, and population risks.

The study has 12 settings and 200 repetitions per setting. All 2,400 joint risk sandwiches cover the known population quantities. This observation does not mean that all predictors are accepted: several settings have no accepted predictor, and minimum feasible coordinate cost is seldom certified. The diagnostic calculations separate failure of the fitted predictor, weakness of the selected graph constraints, and statistical uncertainty.

Its standard-library driver is separate from the seven groups in the table above:

```bash
python3 experiments/single_trajectory_certification/reproduce.py --mode check --output /absolute/path/trajectory_check
python3 experiments/single_trajectory_certification/reproduce.py --mode run --output /absolute/path/trajectory_run --workers 6
```

The first command verifies the retained evidence, runs the original 97 tests, reconstructs saved full certificates, and reproduces the summaries and both post-hoc diagnostics. The second repeats the complete fixed experiment. The [reproduction guide](../experiments/single_trajectory_certification/REPRODUCIBILITY.md) defines their outputs and optional figure generation. Choose a separate output directory outside the study's source and evidence directory.

These population decisions concern stationary average squared error under supplied process assumptions. They do not establish those assumptions from an arbitrary finite record or provide conditional accuracy for each next observation. The implementation evaluates its declared rational affine-score features exactly; applying the same theorem to a Gaussian KAHM feature pipeline requires numerical guarantees for that pipeline.

## Recompute one group's evidence

Use Python 3.12 and install the recorded dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce.py --help
python3 reproduce.py --only ecg_cohort --output reproduction_ecg
```

Replace `ecg_cohort` with a group from the table. The driver verifies the current source/input manifest, invokes that group's numerical checker, and checks the manifest again. Use a separate output directory for each run. It must not be the repository root or lie under `docs/`, `experiments/`, `scripts/`, or `validation/`.

For the ECG command, inspect:

| Output | Meaning |
| --- | --- |
| `reproduction_ecg/summary.json` | Overall completion status, source-manifest identity, environment, command, and per-group result location |
| `reproduction_ecg/ecg_cohort.json` | Recomputed evaluation dataset results, threshold comparisons, and numerical checks |
| `reproduction_ecg/ecg_cohort.log` | Checker output and any failure information |

Other groups use the same layout with their group name. `class_diagnosis` instead writes a subdirectory containing `summary.json`, `class_risk.csv`, and `head_excess.csv`. The first CSV contains all 54 class-risk brackets and tolerances; the second contains all 108 fitted-predictor risk and excess intervals. Its exact arithmetic may take several minutes.

Run the complete retained evidence checks with:

```bash
python3 reproduce.py --output reproduction_all
```

The controlled group regenerates calibration observations from the fixed random streams and checks the stored optimization records without calling an optimizer. The deployment group also regenerates forecasts from stored raw inputs and model parameters. The other group READMEs identify the exact calculations performed. These commands evaluate the supplied records. Model training and runtime benchmarking require separate execution of the corresponding source.

## Interpret the result at the right level

**Population-risk decisions.** In the controlled experiment, independent episodes support the stated calibration. Exclusion uses a lower bound above the tolerance. Acceptance requires an upper bound at or below that tolerance for a fitted predictor belonging to the same declared class. Otherwise the decision remains unresolved. These conclusions depend on the sampling, calibration, class-membership, and numerical assumptions.

**Fixed-dataset decisions.** The ECG calculation concerns 8,192 distinct patients, with one specified history–target pair per patient. Its exact arithmetic is conditional on the recorded bounds on feature values. Excluding the two-coordinate class does not exclude every KAHM geometry; the three separately selected geometries are not a nested sequence of feature sets. The unresolved larger classes are not thereby certified adequate.

For Bike, feasible predictions establish that all 54 examined classes can meet their requirements on the assessed observations, while both fitted predictors fail in 39 classes. Those feasible predictions use assessment targets. They demonstrate capacity on that dataset, without establishing future performance of a predictor trained independently of those targets. The related rental targets are not independent application replications.

**Numerical-output checks.** In affine forecasting, an accepted output has sufficiently small discrepancy from a fixed rational affine reference evaluated on the trusted stored features. This does not bound the prediction error against a future observation or verify every step of feature generation. The record also contains the separate evaluation dataset accuracy comparisons, adverse monthly results, and checking costs.

**Accuracy and computational cost.** A successful reproduction preserves unsuccessful scientific results. The deployment procedures have zero cases meeting the joint accuracy, storage and latency requirement; the screening experiments retain zero additional fits avoided under their respective comparisons. The timing summaries use the original measurements. Running the check on another computer does not turn them into measurements of that computer.

## Continue to model regeneration

`PASS` means that a checker agrees with the retained numerical result under its stated premises. Checksum agreement identifies the supplied files; it does not independently authenticate execution or establish an independent scientific replication.

The experiment directories retain the original source and protocols for further investigation. Full feature generation, model fitting, and raw-data extraction require each study's declared data, environment, and execution sequence. Use its README and provenance records to identify those requirements before adapting the original source. The supported driver above starts from the retained numerical inputs and checks calculations conditional on the supplied inputs.

