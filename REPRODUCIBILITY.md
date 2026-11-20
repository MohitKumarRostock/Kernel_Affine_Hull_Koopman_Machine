# Reproduction of stored experimental results

## Environment and commands

The recorded environment uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, and mpmath 1.3.0. Install the pinned packages and run from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce.py --output reproduction
```

To reproduce the five experiment groups reported in the manuscript:

```bash
python3 reproduce.py \
  --only controlled_calibration ecg_cohort class_diagnosis \
         deployment_cost public_screening \
  --output reproduction_manuscript
```

The command without `--only` also includes the supplementary `decoder_diagnostics` group. To select one experiment:

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

The controlled group regenerates calibration observations from fixed random streams. The deployment group regenerates forecasts from stored raw inputs and model parameters, using the specified comparison tolerances. Its MSE calculations treat stored binary64 targets and forecasts as exact values.

## Numerical assumptions and file integrity

Each group specifies whether stored feature coordinates are exact inputs or are enclosed by numerical intervals. Binary64 values are treated as exact dyadic rationals where required. Exact fractions, outward rounding, and strict or inclusive threshold comparisons follow the corresponding protocol. Targets outside a predictor's output interval remain in the loss calculations.

`MANIFEST.json` covers source code, documentation, and stored numerical inputs. Generated validation reports and reproduction outputs are excluded. Experiment packages also verify their local manifests where supplied. Checksums identify file contents; the numerical checkers verify calculations under the stated input assumptions.

## Interpretation

`PASS` means agreement with the stored numerical result, not success of every proposed method. The recorded results include no joint deployment-criterion passes, no additional predictor fits avoided by screening, and mixed or unsuccessful predictor comparisons.

Population-risk confidence bounds use the controlled experiment's independent observations and prespecified calibration procedure. The PTB-XL and Bike Sharing optimization results concern minimum empirical risk on fixed datasets. Related rental targets and reused evaluation periods are not independent experimental replications.

The commands check stored inputs and calculations. They do not repeat every data-acquisition or feature-generation step, retrain all models, or authenticate the original execution history. Timing summaries use saved measurements rather than new benchmarks. Each experiment's source, protocol, and provenance files specify the inputs and execution steps for further regeneration. See the [experiment guide](docs/EXPERIMENT_GUIDE.md) and [data attribution](DATA.md).
