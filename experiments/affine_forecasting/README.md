# Model selection and numerical verification for affine forecasting

This experiment selects clipped affine forecasting models on development data
for three tasks, measures their prediction error and verifies their numerical
outputs. It also measures the computational cost of numerical verification.
The directory contains the source code and saved inputs needed to repeat the
reported checks.

## Run the numerical replay

From the repository root, with Python and NumPy installed:

```bash
python experiments/affine_forecasting/replay.py --output results/affine_forecasting.json
```

The output location must be outside this experiment directory. The replay
reads JSON/gzip records and exact binary64 coordinates; it does not load a
pickle, download data, fit a model, or overwrite retained observations.

The replay performs these computations:

- Reconstructs 1,206 pooled development losses from all 134 candidate forecast
  sequences for each of three tasks and three seeds, then repeats all 54
  policy selections with the fixed lexical tie rule.
- Checks all 15 distinct stored affine models, including exact coefficient
  norm, predictor-class identity and stored-model identity checks.
- Computes 1,350 exact clipped affine references with rational arithmetic and
  verifies the corresponding output records with a separate integer
  arithmetic verifier. Every one of the 270 assessment queries is covered by
  five distinct selected affine models.
- Applies the same exact error calculation to all 1,350 float32 conversions of predictions
  in the original target units. It also replays the 45 checks for incorrect query, model or
  feature associations against the expected identifiers.
- Recomputes every retained assessment and monthly policy loss from forecasts in the original
  target units and unclipped targets, including all seeds and adverse months.
- Reaggregates the original ten timing repetitions at each of 32 selected
  queries per task. These are summaries of recorded measurements, not a new
  runtime benchmark.

## Results and their scope

The primary procedure selects an affine model of the raw input features with seed
3901. Its comparison pool is the conventional family of 44 configurations
(`ordinary`) and its five-configuration ridge
subset. The 90-candidate affine representation pool (`all_map`) is a different
secondary policy. Selection uses only the three development folds.

The primary criterion on the evaluation dataset is
`20 * SSE_raw <= 21 * SSE_reference` for both references on each task. Each
task has 90 assessment targets. All six pooled inequalities pass:

| Task | Raw affine RMSE | Conventional pool RMSE | Ridge RMSE | MSE change versus pool | MSE change versus ridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Household power, 2008 (kW) | 1.09691759 | 1.08483969 | 1.07093076 | +2.2391% | +4.9120% |
| Jena temperature, 2018 (°C) | 2.60618700 | 2.54365436 | 2.54365436 | +4.9772% | +4.9772% |
| Metro traffic, 2018 (vehicles/hour) | 519.26784944 | 531.28576995 | 523.95665851 | −4.4729% | −1.7818% |

This comparison uses a fixed set of evaluation observations. It is not a
population noninferiority test. The seed repetitions reuse the same targets.
The five-percent condition fails in eight task/month/reference comparisons:
household September and October versus ridge and November versus the conventional pool; Jena
September and October versus both references; Metro September versus ridge.
Every month is included in the input records and generated report. The
[forecasting-model comparisons](../decoder_diagnostics/) also report failed
5% MSE criteria on Household and Jena and ablations on those same periods.
Those ablations use data already examined during model development and do
not provide an independent confirmation.

All 1,350 unmodified binary64 forecast records are accepted. Quantization changes every
reported value; 1,349 float32 variants exceed the fixed discrepancy budget
and are rejected, while one remains inside the budget and is accepted. All
45 prescribed incorrect query, model or feature associations are rejected. Acceptance certifies the discrepancy
between the supplied prediction in original target units and the fixed stored rational affine
reference, conditional on the supplied exact binary64 feature coordinates.
The normalized budget is `2^-40`; the budget in original target units is the training target
span multiplied by that value. Target labels are not inputs to numerical output verification.

This guarantee is conditional on the stored feature values. It does not bound
errors introduced during feature generation, establish a training optimum or
guarantee accuracy on future targets. File hashes establish content identity;
they do not authenticate the source.

The retained timing observations are unfavorable to a speedup claim. For the
primary raw policy, the median full verified prediction costs 13.22–13.51
times the median plain prediction. A check using a cached verification record costs 3.44–3.49
times the corresponding direct exact-reference recomputation. The report
includes all selected policies, not only the primary timing row.

## Inputs and data-generation source

`witnesses/assessment.json.gz` stores exact model parameters, binary64
feature hex strings, unmodified and quantized forecast records, predictions
in original target units, query IDs, targets and incorrect-association checks. Feature values are trusted inputs to these checks. The original per-query `.npy` hashes are retained alongside
their exact coordinates. `development.json.gz` stores candidate configurations,
fold forecast values, targets, and numerical reference outcomes.
`timing.json.gz` retains the actual nanosecond repetitions. `source_inventory.json`
records paths, hashes, and sizes of the original evidence files used to build
the compact inputs. `reference_result.json.gz` contains the recorded aggregate results;
the replay computes those results from the other inputs.

`original/source/` retains the original representation factory, affine
and baseline fitting adapters, data access and generation code, producer,
verifier, cost assay, and audit code. `original/inputs/native/` retains the exact
KAHM source files used by that factory. `original/output/data43/` contains
all training and validation datasets, their raw lag arrays, exact
target tokens, and training-only normalization records. These are sufficient
for offline method development from the supplied datasets. The original design
and protocol describe data extraction, checks for overlap with data used in
model development, the hyperparameter grid, random seeds and execution order.

The replay evaluates the supplied numerical records. Repeating raw-data
extraction requires the archives identified below and the documented extraction
procedure. Model training requires those inputs, the specified environment and
the training source. The paths recorded with the data identify the source
files and execution directories.

## Dataset provenance

| Task | Primary dataset identity | Retained source identity and qualification |
| --- | --- | --- |
| Household power | UCI Individual Household Electric Power Consumption, DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54), CC BY 4.0 | Pinned public mirror archive at commit `509690881d2c47c89d3d221ed2e2bb30e5afa9c0`, SHA256 `9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff`. Byte identity with the unavailable official archive was not independently established. |
| Jena temperature | [MPI-BGC Beutenberg roof weather station](https://www.bgc-jena.mpg.de/wetter/weather_data.html), CC BY 4.0 in retained source metadata | Official `mpi_roof_2018a.zip` and `mpi_roof_2018b.zip`; archive and member hashes are in `original/inputs/data43/jena_temperature2018/source43.json`. |
| Metro traffic | UCI Metro Interstate Traffic Volume, DOI [10.24432/C5X60B](https://doi.org/10.24432/C5X60B), CC BY 4.0 | Retained official UCI archive, SHA256 `b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768`. |

The original data adapters use only each task's own lagged target. Household
power and Jena use the recorded minute-00 snapshots; these are observations at the start of each hour,
not hourly averages. Source timestamps retain the original recorded conventions.
Exact URLs, archive/member hashes, measurement semantics, missing-data rules,
and transport qualifications are preserved in the source records.

Code and original study outputs follow the repository licenses. Third-party
dataset terms and attribution remain those of their respective providers.
