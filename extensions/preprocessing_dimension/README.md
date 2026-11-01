# Preprocessing and subspace dimension in KAHM forecasting

This extension studies standard, min–max and tanh history preprocessing with
validation-selected native local dimensions. Two hybrid feature constructions
combine tanh common geometry with min–max reconstruction departures and the
reverse pairing. It builds on the published corrected dimension study at
commit `3cd8e56d95ec1187deed6070541f26742ce4f1cc`.

The primary comparison uses complete concatenated feature heads. The separate
secondary study fixes a common forecast and shared bias, then fits centered
departure slopes to honest residuals from disjoint episodes. These are distinct
architectures and their performance rankings must not be pooled.

The design uses 24 fresh independent repetitions of four numerical observation
laws at two noise levels. There are 192 task suites, but the independent unit
for inference is a repetition. This is a controlled simulation study; it does
not establish performance on natural-data application cohorts.

## Design and source

- [PROTOCOL.md](PROTOCOL.md) specifies all fitting, selection, calibration and
  held-out evaluation roles.
- [config.json](config.json) records the finite grids, seeds and sample sizes.
- [analysis_code/ANALYSIS_PROTOCOL.md](analysis_code/ANALYSIS_PROTOCOL.md)
  specifies the four simultaneous primary comparisons and secondary estimands.
- [code/preprocessing_model.py](code/preprocessing_model.py) supplies the new
  preprocessing adapter while importing the verified immutable numerical
  runtime in `../dimension_forecasting`.
- [code/fixed_common_secondary.py](code/fixed_common_secondary.py) supplies the
  honest residual mechanism experiment.
- [audit/verify_primary.py](audit/verify_primary.py) checks manufactured inputs
  against the published adapter, independent scaler/hybrid oracles and phase
  isolation.

The preceding extension's `code`, `runtime`, `vendor_original` and their
manifests are required. Preserve their directory structure. Install the pinned
packages in [requirements.txt](requirements.txt). Use one numerical thread per
worker; the runner uses at most three workers on distinct allowed CPU cores.

## Reproduce the frozen execution

From this extension, use a clean copy with the source freeze retained and with
no `prepared`, `results`, `predictions`, `execution`, `FIT_LOCK.json`,
`PREDICTION_LOCK.json` or `TARGET_RELEASE.json`. The following commands replay
the declared seeds. They do not constitute an independent new research sample.

```bash
python code/pipeline.py abc
python code/pipeline.py fit --workers 3
python code/pipeline.py seal-fits
python code/pipeline.py d
python code/pipeline.py predict --workers 3
python code/pipeline.py seal-predictions
python code/pipeline.py release
python analysis_code/analyze.py --root . --output verification/reanalysis
```

Production source, configuration and analysis definitions were frozen before
development data generation. This is a local prospective freeze, not a public
preregistration. A/B-only choices are saved before C arrays are loaded. All
fits are sealed before any D episodes are generated. Query workers receive
only D histories; they do not open target files. Analysis requires the complete
prediction seal and target-release record. These are auditable procedural
separations, not cryptographic access isolation.

Unexpected failures stop the stage and are retained. Existing outputs are
never overwritten. Pure numerical denominator guards inherited from the
published reconstruction are logged; unexpected factor, coordinate, gate and
kernel failures raise errors.

## Numerical records

Each task preserves development inputs, selected states, all validation and
calibration predictions, local bases/references, 502 final forecast arrays,
query diagnostics and execution receipts. Large numerical records are supplied
as separate reproducibility archives, with an archive manifest. Sources,
protocols, aggregate tables and reports can be retained in Git without those
large arrays.

Full measured construction/selection and prediction stage times include all
searched candidates. Component timing nodes are diagnostic and do not cover
all Python, feature-assembly and serialization overhead. They must not be
reported as complete isolated deployment runtimes or hardware speed rankings.
