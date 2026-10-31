# Natural forecasting: accuracy, cost, and selection of geometric features

This study evaluates whether KAHM features improve a raw-history forecasting model, how much computation the complete procedure requires, and whether a diagnostic fitted on development sources predicts useful cases on reserved sources.

The design contains eight original source families, 26 selected series, and two paired history–horizon settings per series. Two capacity settings and two fitting seeds give 208 task runs: 128 development runs and 80 confirmation runs. The independent source groups remain four development families and four confirmation families. Settings and provenance are specified in [PROTOCOL.md](PROTOCOL.md), [protocol.json](protocol.json), and [data_sources.json](data_sources.json).

## Confirmation results

The following values are source-balanced means from the saved confirmation results. Mean squared error is normalized by a training-prefix seasonal-difference scale, with the protocol's first-difference fallback.

| Procedure | Scaled test MSE | Warm pipeline wall time per task |
| --- | ---: | ---: |
| Raw-history RBF | 0.8075209 | 0.168791 s |
| Raw history plus KAHM, with an RBF decoder | 0.8419228 | 0.387412 s |
| Raw-history ridge | 0.7798112 | 0.188043 s |

RBF denotes a finite radial-basis-function decoder. Adding KAHM produces 4.26% higher scaled mean squared error and 2.30 times the measured warm pipeline cost of raw-history RBF. The cost includes the specified preparation, search, selection, refitting, prediction, serialization, cleanup, and bookkeeping. It is a measurement of this implementation and host, not a hardware-independent complexity ratio.

In the predeclared diagnostic comparison, adding geometric diagnostic features to source metadata and a low-cost predictive probe changes none of the 80 confirmation decisions. The complete portfolio and source-level results are in `runs/natural_v1/confirmation_analysis.json` and `runs/natural_v1/reports/summary.json` after evidence restoration.

These findings are descriptive evidence for four reserved source families. Related series, capacities, fitting seeds, and overlapping forecast targets do not create additional independent families. The study does not establish a transferable rule for deciding when KAHM will improve forecasting.

## Source and execution records

The earlier source checkpoint is `a67c40190458bab7b3e56b4fac76770798d074b3`. [CHECKPOINT.json](CHECKPOINT.json) records the SHA-256 hashes of its 217 tracked files. The natural-study code, source allocation, task identities, and initial analysis specification were published before development outcomes in [commit 2600e45220b0562b5f67ef0c769fcae6c11e120e](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2600e45220b0562b5f67ef0c769fcae6c11e120e). The diagnostic coefficients and confirmation rules were published after development and before confirmation in [commit 4ab972ba08a4d5678347c1ea124ed6509c806d5a](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/4ab972ba08a4d5678347c1ea124ed6509c806d5a).

These are public execution records. They are not registration with a dedicated preregistration service. Scientific source hashes identify the code that generated the evidence; later reporting and packaging tools have separate identities.

## Create an isolated historical source tree

The natural-study audit verifies all 217 historical files, including their original documentation. The repository's current manuscript and reader guides have a separate document identity. To run that whole-checkpoint audit, create an isolated tree containing the exact historical files and the natural-study implementation:

```bash
python scripts/materialize_historical_checkpoint.py --output /tmp/kahm-natural-reference
```

Run this command from a complete repository checkout. The helper checks every historical hash, preserves the current source tree, and refuses an existing destination. It reads local Git objects without downloading history. A shallow or partial clone may require an explicit fetch of the recorded commit and its objects before materialization:

```bash
git fetch --refetch --no-filter --depth 1 origin a67c40190458bab7b3e56b4fac76770798d074b3
```

The selected natural-study prefix must also be populated in a sparse checkout. `MATERIALIZATION_MANIFEST.json` distinguishes exact historical files from copied current study files. The materialized tree is a reproduction source view, not a new scientific freeze.

## Restore and analyze the saved evidence

Use Python 3.12 with the pinned natural-study requirements. In the isolated tree:

```bash
cd /tmp/kahm-natural-reference
python -m pip install -r extensions/natural_forecasting/requirements.txt
python extensions/natural_forecasting/package.py check-source
python extensions/natural_forecasting/package.py restore-reference \
  extensions/natural_forecasting/reference/natural_v1.zip.parts.json
python extensions/natural_forecasting/package.py audit .
python extensions/natural_forecasting/package.py replay . --output /tmp/kahm-natural-analysis
```

The public evidence is stored in ordered binary parts. Restoration verifies each part, the reconstructed archive, and its file inventory before creating `extensions/natural_forecasting/runs/natural_v1`. It refuses to replace an existing run. An expanded standalone reproduction archive already contains that run and does not need restoration.

The audit checks frozen source, input, and output hashes; recomputes errors from the saved prediction arrays; reconstructs target values and forecast origins; checks the complete method and task rosters; and evaluates the frozen confirmation policies. It also refits the small diagnostic regressions to verify their coefficients. It does not refit forecasting models or geometry banks.

The replay command writes recomputed development and confirmation analyses to a separate empty directory. Newly measured diagnostic-regression and scalar-decision timings can differ; original model timing records remain the evidence for the reported resource comparison.

## Repeat the fitting experiment

Use a new run directory in the isolated historical source tree:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
NATURAL_RUN=extensions/natural_forecasting/runs/computational-reproduction
python -m extensions.natural_forecasting.data fetch --role all
python -m extensions.natural_forecasting.run prepare --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.run initial-freeze --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.run run --role development --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.run freeze-confirmation --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.run run --role confirmation --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.run analyze --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.report \
  --run-root "$NATURAL_RUN" \
  --reports-dir "$NATURAL_RUN/reports" \
  --manuscript-dir "$NATURAL_RUN/manuscript-generated"
```

The fetch command verifies included source archives and downloads only missing pinned public snapshots. Preparation selects the prescribed roster through training-prefix eligibility and a salted hash of source and series identity. Targets must fall entirely within the relevant chronological partition. Missing windows do not trigger replacement series, and the loader does not interpolate missing values.

After development, `freeze-confirmation` audits the saved predictions, evaluates the diagnostic by leaving out source families, and fits the four predefined diagnostic variants on all development families. Confirmation then evaluates those fixed choices. Repetition of these published inputs and settings is computational reproduction; it does not add independent source evidence.

## Accuracy and resource interpretation

The principal pair compares raw-history RBF with raw-history-plus-KAHM RBF. Affine and locality controls distinguish the effects of gates and access to raw histories. All finite RBF/Nyström predictors use an ordinary Euclidean ridge penalty. Capacity settings bound examples and model size, not realized wall-clock time. The prescribed tuning grids do not constitute an exhaustive comparison with contemporary forecasting systems.

Task loss averages horizons and recorded origins before normalization. Source-level summaries then weight families equally through the declared hierarchy of series, settings, capacities, and seeds. Four reserved families support bounded descriptive conclusions and sensitivity analysis, not broad-population significance claims.

Resource records distinguish wall time, process CPU time, serialized model size, and process resident memory. Selection procedures are charged for their required search and diagnostic computations even if they ultimately reject KAHM. The cost of acquiring development labels for a diagnostic, fitting the diagnostic, and applying it to a new task are separate quantities. No hypothetical deployment count is used to amortize the initial cost.

Availability affects interpretation. The longer Hospital and Traffic settings have only 21 and 29 initial training windows, respectively. The longer setting for NN5 series T88 has seven test origins. History length and forecast horizon change together, so their separate effects are not identified. FRED-MD is a fixed-vintage archive whose upstream processing is not completely audited at the series level; training-only preprocessing claims apply to this study's handling of that snapshot.

## Files and manuscript

| Location | Contents |
| --- | --- |
| `data.py`, `data_sources.json`, `datasets/` | Source snapshots, attribution, series selection, and chronological windows |
| `models.py`, `baselines.py` | Geometric forecasting procedures and controls |
| `run.py`, `protocol.json` | Fitting, selection, measurements, and execution rules |
| `analysis.py` | Source-balanced analysis and diagnostic policies |
| `registration/` | Public checkpoint and validation records |
| `reference/` | Hash-verified parts containing the reference run |
| `runs/natural_v1/` | Restored prepared inputs, predictions, results, and analyses |
| `package.py` | Historical checkpoint verification, evidence restoration, and replay |

The [integrated manuscript](../tmlr_integration/README.md) contains this study's methods, results, and limitations. The local `manuscript/` directory retains the earlier standalone study document.
