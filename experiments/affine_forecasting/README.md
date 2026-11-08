# Selected affine forecasting policies

This directory contains the focused S43 experiment: development selection on
three forecasting tasks, exact numerical verification of selected clipped
affine predictions, and a retained implementation-cost assay. The release
preserves the original computation source and the numerical premises needed
for an offline replay. It does not require the historical reporting PDFs.

## Run the numerical replay

From the repository root, with Python and NumPy installed:

```bash
python experiments/affine_forecasting/replay.py --output results/affine_forecasting.json
```

The output location must be outside this experiment directory. The replay
reads JSON/gzip witnesses and exact binary64 coordinates; it does not load a
pickle, download data, fit a model, or overwrite retained observations.

The replay performs these computations:

- Reconstructs 1,206 pooled development losses from all 134 candidate forecast
  sequences for each of three tasks and three seeds, then repeats all 54
  policy selections with the frozen lexical tie rule.
- Checks all 15 distinct stored affine deployments, including exact coefficient
  norm, class identity, and deployment identity checks.
- Computes 1,350 exact clipped affine references with rational arithmetic and
  verifies the corresponding packets with the original separate integer
  arithmetic verifier. Every one of the 270 assessment queries is covered by
  five distinct selected affine deployments.
- Applies the same exact error calculation to all 1,350 physical-output
  float32 conversions. It also replays the 45 wrong-query, wrong-deployment,
  and stale-feature challenges against the external expected context.
- Recomputes every retained assessment and monthly policy loss from physical
  forecasts and unclipped targets, including all seeds and adverse months.
- Reaggregates the original ten timing repetitions at each of 32 selected
  queries per task. These are summaries of recorded measurements, not a new
  runtime benchmark.

## Results and their scope

The frozen primary policy is the raw-feature affine family selected with seed
3901. Its comparison pool is the conventional family of 44 configurations
(`ordinary`, called Pool in the manuscript) and its five-configuration ridge
subset. The 90-candidate affine representation pool (`all_map`) is a different
secondary policy. Selection uses only the three development folds.

The primary finite-assessment criterion is
`20 * SSE_raw <= 21 * SSE_reference` for both references on each task. Each
task has 90 assessment targets. All six pooled inequalities pass:

| Task | Raw affine RMSE | Conventional pool RMSE | Ridge RMSE | MSE change versus pool | MSE change versus ridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Household power, 2008 (kW) | 1.09691759 | 1.08483969 | 1.07093076 | +2.2391% | +4.9120% |
| Jena temperature, 2018 (°C) | 2.60618700 | 2.54365436 | 2.54365436 | +4.9772% | +4.9772% |
| Metro traffic, 2018 (vehicles/hour) | 519.26784944 | 531.28576995 | 523.95665851 | −4.4729% | −1.7818% |

This is a comparison on a fixed historical assessment roster. It is not a
population noninferiority test. The seed repetitions reuse the same targets.
The five-percent condition fails in eight task/month/reference comparisons:
household September and October versus ridge and November versus Pool; Jena
September and October versus both references; Metro September versus ridge.
Those months remain in the witness inputs and generated report. The earlier
S41 primary status remains **NOT_ESTABLISHED**, and the S42 periods remain
**DIAGNOSTIC_ALREADY_EXPOSED_PERIODS**.

All 1,350 honest binary64 packets are accepted. Quantization changes every
reported value; 1,349 float32 variants exceed the frozen discrepancy budget
and are rejected, while one remains inside the budget and is accepted. All
45 prescribed misbindings are rejected. Acceptance certifies the discrepancy
between the supplied physical prediction and the fixed stored rational affine
reference, conditional on the supplied exact binary64 feature coordinates.
The normalized budget is `2^-40`; the physical budget is the training target
span multiplied by that value. Labels are not inputs to packet verification.

The contract does not enclose errors of an ideal real-arithmetic KAHM/native
transform, certify a training optimum, provide future-label coverage, or
authenticate a source. Hashes bind retained content; they are not signatures.

The retained timing observations are unfavorable to a speedup claim. For the
primary raw policy, the median full verified prediction costs 13.22–13.51
times the median plain prediction. A cached packet check costs 3.44–3.49
times the corresponding direct exact-reference recomputation. The report
includes all selected policies, not only the primary timing row.

## Inputs and original generation source

`witnesses/assessment.json.gz` stores exact deployment parameters, binary64
feature hex strings, honest and quantized packet contents, physical forecast
values, query IDs, targets, and misbinding challenges. Features are explicit
trusted premises. The original per-query `.npy` hashes are retained alongside
their exact coordinates. `development.json.gz` stores candidate configurations,
fold forecast values, targets, and numerical reference outcomes.
`timing.json.gz` retains the actual nanosecond repetitions. `source_inventory.json`
records paths, hashes, and sizes of the original evidence files used to build
the compact inputs. `reference_result.json.gz` is the historical aggregate;
the replay computes its scientific outcomes from the other premises.

`original/source/` retains the original representation factory, native affine
and baseline fitting adapters, data access and generation code, producer,
verifier, cost assay, and audit code. `original/inputs/native/` retains the exact
native KAHM sources expected by that factory. `original/output/data43/` contains
all training and development validation cohorts, their raw lag arrays, exact
target tokens, and training-only normalization records. These are sufficient
for offline method development from the curated cohorts. The original design
and frozen protocol describe the complete data extraction, exposure checks,
fit grid, seeds, and chronology.

The consolidated entrypoint is a numerical replay, not a rerun of the original
data-acquisition and sealing workflow. Historical paths inside the original
receipts identify their original run locations. Reproduction of the raw-source
extraction would additionally require the archives identified below and the
documented original workflow. Full model retraining was not run during this
consolidation.

## Dataset provenance

| Task | Primary dataset identity | Retained source identity and qualification |
| --- | --- | --- |
| Household power | UCI Individual Household Electric Power Consumption, DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54), CC BY 4.0 | Pinned public mirror archive at commit `509690881d2c47c89d3d221ed2e2bb30e5afa9c0`, SHA256 `9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff`. Byte identity with the unavailable official archive was not independently established. |
| Jena temperature | [MPI-BGC Beutenberg roof weather station](https://www.bgc-jena.mpg.de/wetter/weather_data.html), CC BY 4.0 in retained source metadata | Official `mpi_roof_2018a.zip` and `mpi_roof_2018b.zip`; archive and member hashes are in `original/inputs/data43/jena_temperature2018/source43.json`. |
| Metro traffic | UCI Metro Interstate Traffic Volume, DOI [10.24432/C5X60B](https://doi.org/10.24432/C5X60B), CC BY 4.0 | Retained official UCI archive, SHA256 `b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768`. |

The original data adapters use only each task's own lagged target. Household
power and Jena use the recorded minute-00 snapshots; these must not be described
as hourly averages. Source timestamps retain the original recorded conventions.
Exact URLs, archive/member hashes, measurement semantics, missing-data rules,
and transport qualifications are preserved in the source receipts.

Code and original study outputs follow the repository licenses. Third-party
dataset terms and attribution remain those of their respective providers.
