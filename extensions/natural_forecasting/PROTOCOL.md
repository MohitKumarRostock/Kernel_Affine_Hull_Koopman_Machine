# Natural forecasting extension: prospective protocol

Protocol date: 2026-10-30. This document describes the bounded study specified in `protocol.json`, `data_sources.json`, and the initial analysis specification. It was written before comparative development outcomes. The timestamped `initial_freeze.json` and `confirmation_freeze.json` bind the implemented protocol, task roster, code, dependencies, and diagnostic coefficients. These are internal reproducibility records, not an externally timestamped preregistration.

## 1. Questions and scope

The study asks three related questions on observed natural time series:

1. Does KAHM geometry improve forecasting over a matched raw-input head, after retaining the raw history explicitly?
2. How do complete fitted pipelines compare in prediction error, measured wall time, CPU time, model size, and process memory under two declared capacity caps?
3. Can a diagnostic available before a task's future test outcomes improve the loss–cost tradeoff beyond task metadata and a cheap validation probe?

This is a compact, deliberately selected CPU study. It does not establish a universally useful representation, a population-wide meta-learning result, or readiness for a particular journal. All original synthetic-study files and archives remain unchanged. The extension imports the pinned historical geometry/head implementation through an adapter and records the historical tracked-file checkpoint.

## 2. Sources, allocation, and independent units

Source choices were fixed from public provenance, domain diversity, series length, and manageable download size. No comparative forecasting outcomes were used to choose sources. All eight inputs are versioned snapshots from the [Monash Time Series Forecasting Archive](https://forecastingdata.org/). The exact filenames, ZIP/member SHA-256 hashes, upstream MD5 hashes, original-source references, creator attribution, licenses, and API metadata digests are in `data_sources.json`.

| Role | Source ID | Original source family/domain | Snapshot | Available series | Fixed selected series |
|---|---|---|---|---:|---|
| Development | `electricity_weekly` | UCI electricity consumption | [Zenodo 4656141, v3](https://zenodo.org/records/4656141) | 321 | T72, T286, T139, T136 |
| Development | `nn5_daily` | NN5 ATM cash withdrawals | [Zenodo 4656110, v3](https://zenodo.org/records/4656110) | 111 | T61, T104, T9, T88 |
| Development | `tourism_monthly` | Tourism forecasting competition | [Zenodo 4656096, v3](https://zenodo.org/records/4656096) | 366 | T219, T335, T283, T265 |
| Development | `fred_md_monthly` | FRED-MD macroeconomic indicators | [Zenodo 4654833, v2](https://zenodo.org/records/4654833) | 107 | T2, T98, T34, T37 |
| Confirmation | `traffic_weekly` | Caltrans Bay Area road occupancy | [Zenodo 4656135, v3](https://zenodo.org/records/4656135) | 862 | T801, T569, T458, T770 |
| Confirmation | `hospital_monthly` | Hospital-related patient counts in `expsmooth` | [Zenodo 4656014, v2](https://zenodo.org/records/4656014) | 767 | T239, T360, T73, T578 |
| Confirmation | `saugeen_daily` | Saugeen River flow | [Zenodo 4656058, v2](https://zenodo.org/records/4656058) | 1 | T1 |
| Confirmation | `us_births_daily` | US births in `mosaicData` | [Zenodo 4656049, v2](https://zenodo.org/records/4656049) | 1 | T1 |

The eight official records specify [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the distributed archive snapshots. Original-source citations remain part of the provenance; the archive license is not presented as an additional license for every upstream collection. The ZIPs total 1,172,711 bytes.

The original source family is the independent descriptive grouping. Channels, individual series, aggregation resolutions, horizons, fitting seeds, and capacity settings from the same source are nested observations. For example, another temporal aggregate of the same UCI electricity data would remain in the electricity family. It would not become an independent confirmation source. Development and confirmation families are disjoint, but the design does not claim that every possible broad application domain is absent from development.

Each source contributes at most four series. Eligible IDs are ordered by SHA-256 of the fixed salt `natural-forecasting-roster-v1`, a NUL separator, source ID, a NUL separator, and series ID. Eligibility uses series length and the training prefix only: both forecast regimes must contain at least 16 complete finite training windows, and the training-prefix error scale must be positive and finite. The first four eligible series are retained, or all eligible series if fewer exist. No later loss or validation/test availability criterion changes that roster. In NN5, 109 of the 111 series meet training eligibility.

There are 26 selected series and two paired history/horizon settings per series: **52 base tasks**. Two capacity settings and seeds 17 and 43 produce **208 declared task runs: 128 development and 80 confirmation**. These counts do not enlarge the number of independent source families beyond four in each phase.

## 3. Chronological task construction

For a series of length `n`, indices are zero-based and split ends are exclusive:

```text
training:   [0, floor(3*n/5))
validation: [floor(3*n/5), floor(4*n/5))
test:       [floor(4*n/5), n)
```

An origin `t` denotes the last observed index. Its context is `values[t-L+1:t+1]` and its target is `values[t+1:t+1+H]`. The entire H-step target interval must be inside its assigned split. A context may contain earlier observed values from a preceding split. Stateful baselines use the first-future convention internally; the runner explicitly passes `t + 1` to them.

| Frequency | Short regime `(L,H)` | Long regime `(L,H)` | Fixed scale/baseline period |
|---|---:|---:|---:|
| Monthly | (12, 3) | (24, 6) | 12 |
| Weekly | (13, 4) | (26, 8) | 52 |
| Daily | (28, 7) | (56, 14) | 7 |

The first engineering proposal used monthly (24,3)/(48,6) and weekly (26,4)/(52,8). Before any model outcomes, the length check showed that Hospital's 84 observations yield no training window at (48,6), while Traffic's 104 observations yield only three at (52,8). The adopted settings yield 21 Hospital and 29 Traffic potential training windows in the longer regime. Daily settings were unchanged. The context/horizon settings are paired rather than fully crossed, so their separate effects cannot be identified from this design.

The loader retains the exact numerical values in the supplied snapshots. It performs no learned scaling or interpolation. The original NN5 variant with missing values is used. Windows containing any missing context or target are excluded by a predeclared availability rule. At most 64 available validation origins and 64 test origins are selected at evenly spaced chronological indices. Training caps are 512 or 1,024 windows, selected by the same deterministic rule. Availability is inspected to construct observed-error support, but never to replace a selected series.

Candidate input, feature, and target normalizers are fitted using that candidate's training data. Normalizers are fitted again using the pretest refit data after hyperparameters are selected. No candidate uses validation targets while fitting a search configuration or test targets while fitting its final model.

For final deployment, the runner reconstructs all eligible windows whose complete targets precede the test boundary, then applies the capacity cap. This legitimately includes windows that crossed the old training/validation boundary and were unavailable during model selection. The selected configuration is refitted once on that historical set. During rolling evaluation, already observed validation/test values may enter later contexts or state updates; model parameters are not retuned on test outcomes.

All models for a task use the same recorded evaluation origins. Predictions and origin losses are saved, and reported MSEs are independently recomputed from those arrays. A selected series lacking usable later windows would remain an explicit unsupported task rather than being replaced. The metadata-only roster audit found all 52 base tasks supported. NN5 T88 at (56,14) has only **seven available test origins out of 146 candidates**; this sparse observed support must remain visible in reporting.

## 4. Metric and source-level weighting

The task loss is the mean squared error across H targets within each origin and then across the selected origins. The denominator is the mean squared seasonal difference from the entire uncapped initial training prefix, using finite pairs only. At least two finite pairs and a positive finite mean square are required. If unavailable or zero, the denominator falls back to positive first-difference MSE under the same pair-count rule. A series with no such training scale is ineligible before selection.

This denominator remains fixed during model comparison and pretest refitting. It is not estimated from validation or test data. It supports cross-series descriptive scaling, while retaining the original-unit MSE and scale for audit.

Aggregation assigns equal weight to original source families, then equal weight at each nested level: dataset, series, horizon, history length, capacity setting, and seed. Overlapping windows are not independent observations for inferential purposes. Four confirmation families support descriptive family outcomes, between-family variation, and leave-one-family-out sensitivity; they do not justify window/seed-level p-values or broad-population confidence intervals. No such significance claims are produced by the analysis module.

## 5. Forecasting methods and declared search

The twelve methods are:

| Group | Method IDs | Role |
|---|---|---|
| Matched raw heads | `raw_ridge`, `raw_rbf` | Raw finite history with a linear or finite RBF-feature head |
| Local geometry | `affine_rbf`, `locality_rbf`, `kahm_rbf` | Matched local coordinates with the specified gate |
| Raw-history controls | `raw_affine_rbf`, `raw_locality_rbf`, `raw_kahm_rbf` | Explicit standardized raw history concatenated with each local code |
| Practical controls | `seasonal_naive`, `holt`, `hist_gradient`, `mlp` | Seasonal repetition, Holt trend smoothing, direct gradient boosting, and a direct MLP |

The primary matched comparison is `raw_rbf` versus `raw_kahm_rbf`. Raw-history controls are required because the bank has at most 2+4+8 = 14 local models of rank at most two. Its stacked projection basis therefore has rank at most 28, below the daily long-history dimension 56. Rank deficiency alone does not prove full-code information loss because gates may contain information, but it prevents using the full-column-rank sufficient recovery argument. Numerical ranks and conditioning are diagnostics, not exact mathematical proofs. Explicit raw-history retention avoids relying on that recovery condition.

Geometry uses the declared cluster counts (2,4,8), requested rank two, at most 24 references per local model, a 512-row fitting cap, training-standardized inputs, `omega=16`, and `tau=1e-6`. Small clusters may have lower effective rank; the realized ranks/fallback reasons are recorded. Affine and locality controls construct their own partitions and bases without fitting a KAHM kernel. No bank is silently shared between alternative deployed methods.

KAHM uses the adapter's stable log-kernel query arithmetic. Common row rescaling preserves the normalized kernel weights in exact arithmetic away from a zero signed denominator. The historical exact-zero convention remains, and diagnostics record zero/near-cancelling normalizers and kernel underflow indicators. This is a declared numerical implementation choice in the new extension; it does not alter the historical experiment archive.

Ridge penalties are selected from 0.01, 1, and 100 using chronological validation MSE. The RBF bandwidth multiplier is fixed at one. Penalties within a single geometric pipeline reuse its fitted representation and head sufficient statistics; preprocessing is still fitted on training data only. A final selected pipeline is fitted independently on pretest data. The finite RBF head concatenates linear and Nyström features and uses Euclidean ridge penalization; it is not claimed to implement the canonical RKHS penalty.

| Capacity setting | Maximum training windows | RBF landmarks | Boosting leaves / iterations | MLP hidden width / maximum iterations |
|---|---:|---:|---:|---:|
| Small | 512 | 32 | 15 / 100 | 32 / 200 |
| Medium | 1,024 | 64 | 31 / 150 | 64 / 300 |

Boosting and MLP regularization are selected from 0.01 and 1. Holt selects damped versus undamped trend. Seasonal naive has no tuned hyperparameters. The search spaces are explicit and modest; they do not establish best achievable performance for every method. Capacity caps are not equal realized compute budgets. The analysis therefore reports actual loss and resource cost jointly.

Seasonal naive may access its declared lag from the full observed past even when that lag exceeds the tabular context; it falls back to the last finite observation for unavailable lags, and period one before a complete seasonal lag exists. Holt estimates its parameters on a capped historical segment, forward-fills missing values within that segment only, and advances state using observations strictly before each forecast. For missing online observations it performs a state prediction without a measurement update. The practical controls have explicitly described information and capacity rules rather than an assertion of identical function classes.

## 6. Benefit diagnostic and its controls

The diagnostic is prospective with respect to a new task's test outcomes, not a zero-data forecast. It may inspect that task's training prefix and chronological validation probe. Its cross-task coefficients are learned on development families only.

The supervised diagnostic target is the log ratio of scaled test MSE for `raw_rbf` to scaled test MSE for `raw_kahm_rbf`, with the declared `1e-12` loss floor. Development test losses may supply these labels; confirmation losses may not train, standardize, tune, or select the diagnostic.

Four fixed weighted ridge regressions are retained:

- `metadata`: log training-window count, log horizon/history ratio, and training-prefix lag-one autocorrelation.
- `metadata_geometry`: metadata plus KAHM gate entropy and log affine projection residual.
- `metadata_probe`: metadata plus the log validation-loss ratio of a cheap matched probe.
- `metadata_probe_geometry`: metadata, probe ratio, and geometry features.

The probe uses at most 128 training windows, 16 landmarks, a 128-row bank-fitting cap, and penalty one. Geometry features come from its training inputs. Feature standardization, intercept, and coefficients are fitted using source-balanced development data. The ridge penalty is fixed at one. Development evaluation leaves one complete source family out at a time, with standardization refitted inside each fold. After development, all four models are fitted on all four development families and frozen before confirmation execution.

The primary incremental comparison is `metadata_probe_geometry` versus `metadata_probe`. A useful geometry diagnostic must add value beyond both task metadata and an affordable validation probe. Secondary controls include always raw, always KAHM, validation selection between the matched pair, validation selection within raw/practical or KAHM portfolios, selection over the full portfolio, and direct cheap-probe selection. Validation portfolio selectors pay for every method searched but only refit/forecast the selected method in counterfactual deployment accounting.

The learned policy chooses KAHM when predicted log benefit is strictly positive and chooses raw on ties. The predeclared 5% relative MSE reduction is a secondary practical-benefit/harm margin, not a fitted decision threshold. All four diagnostic variants and all declared selector controls are reported; no diagnostic is chosen retrospectively from confirmation performance. The test-set oracle is only a hindsight regret reference, never an available selector.

## 7. Cost, hardware, and failures

Measured benchmarking uses one active child process and one BLAS thread at a time. Method order is deterministically shuffled per task. Seeds 17 and 43 affect fitting; no method receives parallel workers. The per-method timeout is 120 seconds and the per-task timeout is 1,800 seconds. No paid compute or GPU execution is used.

Wall time and process CPU time are separate measurements. The scientific-pipeline cost records data/checksum preparation, array loading, candidate fitting and validation, selection bookkeeping, selected pretest refitting, batch forecasting, and model serialization/metadata. Source preparation is allocated equally across that source's base tasks and charged identically to every alternative. Failed attempts remain in measured search costs. Shared required nodes are counted once when a selector uses several components.

Diagnostic/probe computation is charged even when a policy rejects KAHM. Geometry-only diagnostic accounting conservatively charges its recorded fitted probe component; the implementation does not claim an unmeasured optimized feature extractor. Offline diagnostic fitting is reported separately and is not amortized over an invented number of future deployments. Scalar policy decision time is also measured.

The initial learning investment additionally includes development-label acquisition: the matched raw/KAHM searches, refits, and forecasts needed to create the learning targets, plus each diagnostic variant's required feature/probe nodes. This cost is reported separately from fitting the small regression and from the actual elapsed cost of the full twelve-method research benchmark. Standalone costs summed across tasks are conservative when sharing data across seeds or capacity settings could reduce work; no such unmeasured reuse or deployment-count amortization is assumed.

Child wall/CPU duration and launch-to-result wall duration are recorded separately from the sum of scientific-pipeline nodes. Process launch/IPC and shared library initialization are runner overheads rather than arbitrary per-selector allocations. Time comparisons must identify which of these scopes they use. CPU time is not treated as interchangeable with wall time, GPU time, energy, or monetary cost.

Serialized fitted-model bytes are recorded. Memory uses process high-water RSS, including the interpreter, inherited arrays, and earlier process history. Its maximum is not an incremental allocation, an additive memory cost, or an exact reconstruction of a hypothetical selector's live memory. These qualifications must accompany memory comparisons.

Failures, timeouts, nonfinite predictions, numerical fallbacks, and optimization warnings are retained. A failed declared method does not trigger method substitution or dropping a source. The analysis refuses incomplete primary method/task rosters. Any corrective implementation change after outcomes must preserve the failed artifacts and explicitly state the amendment and rerun scope; it must not be described as the unchanged original freeze.

## 8. Execution gates and reproducibility

The staged commands, run from the repository root in the pinned environment, are:

```bash
python -m extensions.natural_forecasting.data fetch --role all
python -m extensions.natural_forecasting.run prepare
python -m extensions.natural_forecasting.run initial-freeze
python -m extensions.natural_forecasting.run run --role development
python -m extensions.natural_forecasting.run freeze-confirmation
python -m extensions.natural_forecasting.run run --role confirmation
python -m extensions.natural_forecasting.run analyze
```

`prepare` constructs the complete fixed roster and hashed arrays without evaluating forecasting models. The initial freeze precedes all comparative execution. Prepared protocol/source-manifest hashes must match the current configuration. Execution verifies scientific code, roster, Python/dependency versions, and prepared-array hashes. Before the confirmation freeze, development prediction artifacts and reported losses are independently checked; the diagnostic is then fitted from the complete declared development roster.

Confirmation execution requires an intact final freeze and rejects an arbitrary task subset. Resumption skips only complete results with matching prediction-artifact hashes. Final validation rechecks predictions, the frozen protocol, and unchanged historical tracked files. The exact dependency pins are in the extension's `requirements.txt`; environment and threadpool details are stored in freeze records.

Synthetic tests cover target containment, legitimate pretest refits, first-future baseline indices, validation-only selection, immutable artifacts, and rejection of changed code, dependencies, diagnostic coefficients, and partial confirmation requests. They are engineering validation and supply no natural-data efficacy evidence.

## 9. Interpretation limits to retain in reporting

Four reserved source families are a small confirmation collection. The design is selected rather than a random sample of forecasting domains. Individual source panels contain related series, and the single-series sources provide no independent within-source replication. Multiple origins, horizons, seeds, and capacities do not solve that limitation.

The study guarantees train-only processing in this extension relative to the supplied snapshots. It cannot reconstruct every upstream transformation or original data-release schedule. In particular, the FRED-MD record describes differenced/log-transformed series but does not specify a complete, auditable per-series transformation pipeline in its metadata. The archived values are used unchanged; the study does not certify that every upstream preprocessing decision was training-only. The fixed-vintage FRED evaluation is not a real-time macroeconomic backtest and does not account for later revisions or indicator release delays. Any forecasting claim must retain this scope.

Frequency-based seasonal periods are conventions, not estimates of each series' optimal cycle. Saugeen may exhibit longer seasonal behavior than the fixed daily period of seven captures. The finite contexts, paired horizons, capped training samples, limited tuning grids, and observed missingness define the evaluated problem. They do not establish dominance for optimally tuned large models, other horizons, other source vintages, or deployment settings.

Positive results would support a bounded claim about accuracy and resource tradeoffs under this declared design. Negative or mixed results are equally reportable. A broad claim about an ex ante predictor requires it to outperform the metadata and cheap-probe controls on the frozen families after accounting for its acquisition and fitting cost; correlation with retrospective gains alone is insufficient.
