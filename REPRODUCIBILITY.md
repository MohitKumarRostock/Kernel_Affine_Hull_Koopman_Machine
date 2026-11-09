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

## Data and arithmetic

Binary64 coordinates are interpreted as exact dyadic rationals where the numerical analysis specifies this. Exact fractions, outward bounds, and strict versus inclusive decision inequalities retain their original meaning. Targets outside a fitted predictor's output range remain in the losses. Confidence allocations are study-specific; related tasks and reused observations are not independent replications.

The compact records retain the inputs needed for the displayed claims. Each experiment provides provenance tying its records and original source modules to the preserved research archive. Checksums detect changes to these files; they are not evidence of independent attestation or a substitute for numerical verification.

## Interpreting unsuccessful outcomes

A reproduction `PASS` means agreement with the recorded scientific result. It does not mean that every proposed method succeeds. The results include zero joint deployment-utility passes, unsuccessful screening, mixed or failed predictor-transfer criteria, and monthly forecasting failures. The aggregate affine-forecasting criterion compares mean squared errors on a fixed evaluation dataset. It is not a statistical noninferiority test.
