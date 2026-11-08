# Numerical reproduction

## Environment and execution

The supplied replay was checked with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and threadpoolctl 3.6.0. `requirements.txt` records the Python dependencies. The default replay requires no data download or access to a previous workspace.

```bash
python reproduce.py --output reproduction
```

The driver checks `MANIFEST.json`, invokes each experiment in a separate process, and stops on a failed check. It keeps the frozen input records separate from newly generated results. `reproduction/summary.json` identifies the checked source manifest, environment, commands, completion status, and per-group results.

`MANIFEST.json` covers the scientific sources, replay code, documentation, and frozen numerical inputs. The rendered manuscript and the reports in `validation/` are generated outputs. The manuscript checksum and build diagnostics are recorded in `validation/manuscript_build.json`; rebuilding the PDF does not invalidate the numerical-input manifest. The driver checks the source manifest again after all requested experiments finish.

The deployment-cost group also reconstructs forecasts from the retained raw-input model payloads. Its comparison against stored predictions uses the original implementation-fidelity tolerances; the associated MSEs are recomputed exactly from the stored binary64 forecasts and targets. The original timing samples are aggregated again without treating a new host's timing as the historical measurement.

## What the checks establish

| Check | Recomputed quantities | Scope |
| --- | --- | --- |
| Controlled calibration | All 190,464 block programs, 1,008 one-sided endpoints, fitted forecasts, and paired decisions | The selected independent-episode study; guarded arithmetic retains its stated limits |
| ECG cohort | All 2,688 exact dual objectives and 18 configurations, the unchanged reference threshold, and sensitivity requirement | The fixed 8,192-patient cohort, conditional on inherited feature enclosures |
| Class diagnosis | Complete-graph distance enclosures, feasible vectors, signed dual values, 54 class decisions, and 108 head excess bounds | The previously assessed Bike cohorts and exact stored coordinates |
| Deployment cost | Primary forecasts, exact losses, numeric payload inventories, timing medians, and both procedures' utility criteria | Five replications of one HMM law and one Appliances record |
| Public screening | Stored certificate and policy calculations, including unsuccessful screening and fit-avoidance outcomes | The declared Beijing and Bike panels |
| Decoder diagnostics | Selected assessment losses, fixed primary criteria, and 1,440 affine-ablation configurations | Retained assessment cohorts; ablation remains diagnostic |
| Affine forecasting | Exact cohort loss comparisons, numerical outputs and challenge outcomes, and recorded cost aggregates | Three specified periods in known domains; trusted stored features |

These checks do not reproduce every original data-acquisition event, train every model again, establish independent external attestation of event logs, or recertify the complete native feature pipeline. The original scientific source and protocols are included for inspection and further reproduction. Re-executing the original training procedures requires their declared data, environments, and lifecycle gates; the compact replay is the validated entry point in this release.

## Data and arithmetic

Binary64 coordinates are interpreted as exact dyadic rationals where the numerical contract specifies this. Exact fractions, outward bounds, and strict versus inclusive decision inequalities retain their original meaning. Targets outside a fitted predictor's output range remain in the losses. Confidence allocations are study-specific; related tasks and reused observations are not independent replications.

The compact records retain the inputs needed for the displayed claims. Each experiment provides provenance tying its records and original source modules to the preserved research archive. Checksums detect changes to these files; they are not evidence of independent attestation or a substitute for numerical verification.

## Manuscript build

```bash
python manuscript/build_manuscript.py
```

The script runs the LaTeX and bibliography passes and checks for unresolved citations/references and overflow. On installations where TeX trees are present but not configured, `--texmf-root PATH` may be supplied repeatedly; a normally configured TeX installation needs no such option. The PDF and source tree in the repository correspond to the same text.

## Interpreting unsuccessful outcomes

A reproduction `PASS` means agreement with the recorded scientific result. It does not mean that every proposed method succeeds. The retained results include zero joint deployment-utility passes, unsuccessful screening, mixed or failed decoder-transfer criteria, and monthly forecasting failures. The final aggregate forecasting criterion is a finite-cohort inequality rather than a statistical noninferiority test.
