# Numerical reproduction

The [experiment guide](docs/EXPERIMENT_GUIDE.md) connects each scientific question to its input records, executable checks, outputs, and interpretation.

## Environment and execution

The supplied replay was checked with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and threadpoolctl 3.6.0. `requirements.txt` records the Python dependencies. The default replay reads the supplied files locally.

```bash
python reproduce.py --output reproduction
```

The driver checks `MANIFEST.json`, invokes each experiment in a separate process, and stops on a failed check. It reads the supplied input records and writes results to the selected output directory. `reproduction/summary.json` identifies the checked source manifest, environment, commands, completion status, and per-group results.

`MANIFEST.json` covers the scientific sources, replay code, documentation, and fixed numerical inputs. Generated reports in `validation/` and reproduction output directories are excluded. The driver checks the source manifest again after all requested experiments finish.

The deployment-cost group also reconstructs forecasts from the stored raw inputs and model parameters. Its comparison against stored predictions uses the original implementation-fidelity tolerances; the associated MSEs are recomputed exactly from the stored binary64 forecasts and targets. The timing summaries aggregate the saved measurements; they do not measure execution time on the computer running the replay.

## What the checks establish

| Check | Recomputed quantities | Scope |
| --- | --- | --- |
| Controlled calibration | All 190,464 block programs, 1,008 one-sided endpoints, fitted forecasts, and paired decisions | The selected independent-episode study; guarded arithmetic retains its stated limits |
| ECG evaluation dataset | All 2,688 exact dual objectives and 18 configurations, the specified reference threshold, and sensitivity requirement | The fixed 8,192-patient evaluation dataset, conditional on the recorded bounds on feature values |
| Class diagnosis | Complete-graph distance enclosures, feasible vectors, signed dual values, 54 class decisions, and 108 predictor excess bounds | The specified Bike evaluation datasets and exact stored coordinates |
| Deployment cost | Primary forecasts, exact losses, model-storage inventories, timing medians, and both procedures' utility criteria | Five replications of one hidden Markov model and one Appliances record |
| Public screening | Stored certificate and policy calculations, including unsuccessful screening and fit-avoidance outcomes | The declared Beijing and Bike panels |
| Predictor diagnostics | Selected assessment losses, fixed primary criteria, and 1,440 affine-ablation configurations | The specified evaluation datasets; ablation remains diagnostic |
| Affine forecasting | Exact evaluation dataset loss comparisons, numerical outputs and challenge outcomes, and recorded cost aggregates | Three specified periods in known domains; trusted stored features |

These checks start from the supplied observations, features and model records. They do not verify every data-acquisition or feature-generation step, train every model, or independently authenticate execution logs. The scientific source and protocols specify the data, software environments and order of steps required for model training. The commands above reproduce the supported calculations from saved records.

## Single-trajectory certification

The [single-trajectory experiment](experiments/single_trajectory_certification/) uses a separate driver and an exact-rational scientific source archive. Its verified environment uses Python 3.12.14 and the standard library for calculations; optional figure generation also needs Matplotlib. The root driver above continues to run its seven listed studies.

From the repository root:

```bash
python3 experiments/single_trajectory_certification/reproduce.py --mode check --output /absolute/path/trajectory_check
```

The check verifies archive members and supplied evidence, executes 97 tests, checks the complete table of 2,400 outcomes, reconstructs the 12 saved full certificate and population-oracle records, and reproduces the aggregate results and two post-hoc diagnostics. The outcome table is stored losslessly as `runs.jsonl.gz`; the manifest records both its stored identity and its decompressed logical identity. A complete new experiment is available through `--mode run`, with its fixed protocol and deterministic random streams.

The [methods](experiments/single_trajectory_certification/METHODS.md) specify the stationary binary law, the representation and decoder classes, the raw-block layout, exact numerical bounds, and the population oracle. The [reproduction guide](experiments/single_trajectory_certification/REPRODUCIBILITY.md) gives commands and output definitions. The scientific risk guarantee is unconditional over development and calibration and requires a supplied dependence envelope. The confidence limits for the repeated simulation study are separate statements. Cells sharing a random-stream prefix are not pooled as independent repetitions.

## Data and arithmetic

Binary64 coordinates are interpreted as exact dyadic rationals where the numerical analysis specifies this. Exact fractions, outward bounds, and strict versus inclusive decision inequalities retain their original meaning. Targets outside a fitted predictor's output range remain in the losses. Confidence allocations are study-specific; related tasks and reused observations are not independent replications.

The compact records retain the inputs needed for the displayed claims. Each experiment provides provenance tying its records and original source modules to the preserved research archive. Checksums detect changes to these files; they are not evidence of independent attestation or a substitute for numerical verification.

## Interpreting unsuccessful outcomes

A reproduction `PASS` means agreement with the recorded scientific result. It does not mean that every proposed method succeeds. The results include zero joint deployment-utility passes, unsuccessful screening, mixed or failed predictor-transfer criteria, and monthly forecasting failures. The aggregate affine-forecasting criterion compares mean squared errors on a fixed evaluation dataset. It is not a statistical noninferiority test.

