# Experimental calculations and outputs

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
