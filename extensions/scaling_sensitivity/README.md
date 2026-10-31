# Sensitivity to geometry input scaling

This extension compares training z-score geometry scaling with training min-max
scaling to `[-1, 1]` in the integrated manuscript. It is a
paired **post-hoc sensitivity analysis** on already inspected natural sources
and synthetic draws. The original studies and their recorded results remain
unchanged. No scaler or benefit diagnostic is selected using these new test
outcomes.

The protocol and execution code were published at commit
`c5b8817a9157de685fd1c70a22d83542a7bdf766` before any sensitivity data fits.
`protocol_freeze.json` checks 609 execution-source and original-evidence files;
`EXECUTION_CHECKPOINT.json` records that publication boundary. The checkpoint preceded these fits, but the observations had already been evaluated. The analysis therefore provides sensitivity evidence rather than independent source confirmation.

## Fixed design

| Component | Data and methods | Recorded comparisons |
|---|---|---|
| Natural forecasting | All 208 original configurations; raw plus affine, locality, and KAHM under both scalers; fixed raw RBF and raw ridge references | 1,664 standalone method runs |
| Controlled forecasting | All 24 complete draws for each of three laws; both 64/256 landmark budgets; raw, affine local, locality local, KAHM local, and KAHM score features under both scalers | 72 paired jobs and 1,440 head fits |

Min-max statistics use only the fitting histories, before the geometry
subsample. The natural validation-search and final pretest fits learn separate
scalers. Queries outside the fitting range remain outside `[-1, 1]`: there is
no input clipping. Exactly constant training coordinates map to zero in the
geometry; the raw skip retains its original standardization. Feature and target
scales, original target clipping rules, data splits, seeds, decoder capacities,
validation grids, and query arithmetic are unchanged.

Changing a scaler upstream of an unchanged z-score step would largely cancel.
The adapters therefore replace the scaler inside the geometric model. This
changes the complete pipeline and does not isolate a particular partition,
principal direction, kernel regularizer, or gate as the causal mechanism.

The natural primary comparison uses the original four formerly reserved source
families and their equal-family hierarchy. The four original development
families remain separate. Controlled dispersion uses complete data/fit draws.
There are no new confirmatory p-values, confidence guarantees, or independent
source counts inferred from overlapping windows, horizons, or random seeds.

## Restore and audit the recorded evidence

Use Python 3.12 and the dependencies in `extensions/natural_forecasting/requirements.txt`. The integrated audit checks the original 217-file natural-study checkpoint, including historical documentation. Create an isolated source view from a complete checkout:

```bash
python scripts/materialize_historical_checkpoint.py \
  --include extensions/natural_forecasting \
  --include extensions/scaling_sensitivity \
  --include extensions/tmlr_integration \
  --output /tmp/kahm-scaling-reference
cd /tmp/kahm-scaling-reference
python extensions/natural_forecasting/package.py restore-reference \
  extensions/natural_forecasting/reference/natural_v1.zip.parts.json
python -m extensions.scaling_sensitivity.reference restore
python extensions/tmlr_integration/package.py check-source
python extensions/tmlr_integration/package.py audit .
```

The helper verifies every historical hash without changing the current checkout or any scientific freeze. A shallow or partial clone requires the historical objects and selected study prefixes to be populated first; see the [natural-study guide](../natural_forecasting/README.md#create-an-isolated-historical-source-tree). An expanded standalone historical review supplement can use the integrated audit directly.

The natural restoration is needed only when its reference run is absent. Sensitivity restoration verifies every part, the combined archive, and the file inventory, accepting identical files and refusing conflicting replacements.

`analysis.py` independently reconstructs natural losses, input/target scaler
statistics, source hierarchies, archived parity, cost-node sums, and all
controlled episode-loss summaries. `report.py` generates the manuscript tables
and number macros only from the audited summary. To regenerate into a new
directory:

```bash
python -m extensions.scaling_sensitivity.analysis --output /new/path/analysis
python -m extensions.scaling_sensitivity.report \
  --summary /new/path/analysis/summary.json
```

The published analysis is in [analysis/summary.json](analysis/summary.json),
with exact per-task and paired-difference CSVs alongside generated TeX tables.
`analysis/AUDIT.json` distinguishes compact replay from the pre-export full
controlled-prediction audit.

## Recreate the fixed sensitivity fits

Use new output directories and run the two phases serially with one benchmark
worker and one BLAS thread. These commands recreate the same observations and
do not add independent evidence:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m extensions.scaling_sensitivity.natural --output /new/path/natural
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m extensions.scaling_sensitivity.controlled \
  --workers 1 --output /new/path/controlled
```

The controlled runner additionally accepts `--historical-results PATH` for
the original full `results_capacity` directory, if independently regenerated
or retained. That option checks full original prediction parity as well as the
compact-record parity available in every run. The recorded sensitivity execution
used the full original arrays. All standard natural methods and both fixed
references are also checked against their original forecasts.

Full controlled sensitivity predictions and fitted banks are local run outputs.
The compact published artifact retains unchanged job records, episode losses,
selected configurations, data/fit seeds, scaler and range diagnostics, artifact
hashes, and the independent full-prediction audit. Replaying compact losses is
explicitly distinguished from recreating and checking omitted full predictions.
All natural sensitivity predictions are included.

## Cost interpretation

Natural method times are newly measured search, selection cleanup, selected
refit, forecast, and serialization costs. Current data loading, launch-to-result
wall time, and extra sensitivity audits are separate; original source
preparation is reused. Launch-to-result includes child execution and
communication, not isolated startup overhead. The recorded import timer begins
after some adapter dependencies have loaded and does not measure full cold
startup. Comparable time ratios use the raw references measured in this same execution. Ratios against earlier archived timings would mix different executions and measurement conditions. Controlled banks are shared, so their timing records are
execution diagnostics, not standalone deployment comparisons.

The frozen benefit diagnostic is not reevaluated for min-max geometry. The
original positional-information arguments and branch/Airfoil experiments are
also outside this bounded intervention. Any newly chosen practical method
would need fresh reserved evidence before a new transfer claim.
