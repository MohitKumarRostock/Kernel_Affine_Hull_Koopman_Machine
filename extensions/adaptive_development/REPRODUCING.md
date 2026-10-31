# Online adaptation with bounded reference and replay storage

This study evaluates six online forecasting procedures on eight paired synthetic streams and four previously inspected natural series, at two capacity tiers. It asks whether updating KAHM geometry improves prediction relative to fixed geometry, matched adaptive local models, and sliding-window kernel ridge regression.

Both capacity tiers fail the prescribed accuracy criteria. A numerical repair removes implementation exceptions but does not prevent severe extrapolation on the recurring-regime generator. The [findings](FINDINGS.md) report all procedures, resource measurements, and both executions.

## Experimental design

The small tier permits 64 shared geometric references and 128 supervised replay rows per horizon; the large tier permits 128 references and 256 replay rows. These are storage quotas, not bounds on feature or forecast magnitude. The geometric procedures share reference selection and refresh opportunities. The kernel comparator maintains a separate support set for each horizon, so equal row quotas do not imply equal memory use.

The first 35% of each stream initializes the model, the next 15% supplies chronological calibration, and the final 50% supplies evaluation. At each observation, the procedure scores previously issued forecasts, releases newly available targets, updates its state, and issues new forecasts. A correction multiplier is selected separately for each horizon from 0, 0.25, 0.5, and 1 using calibration targets available at the evaluation boundary. This multiplier remains fixed during evaluation, although the model can continue learning from newly observed targets.

The synthetic streams comprise four stationary nonlinear autoregressive series and four paired recurring-regime series. Natural inputs comprise three Dominick weekly profit series and one Sunspot daily series, all selected from previously inspected manuscript inputs. Their use provides development evidence rather than independent confirmation on new data sources. Equations, chronology, settings, and data attribution are specified in [PROTOCOL.md](PROTOCOL.md), [config.json](config.json), and [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Numerical amendment and provenance

The first complete execution encountered four unexpected model exceptions and generated 8,659 framework fallback forecasts. Its source and evidence remain in `archives/initial_attempt.zip`. Amendment A1 replaced the correction-head solver with a numerically stable solution of the same ridge objective, applied to every geometric procedure. The full roster was then evaluated again on the same inputs and with the same calibration grid and decision criteria.

The amended execution comprises 144 method–tier–stream runs and 387,900 forecasts, with no unexpected model exceptions, framework fallback forecasts, or observed storage-cap violations. Both executions belong to one development study; they are not independent replications. The exact source identities are recorded in `DEVELOPMENT_FREEZE.json` and the amendment records. The original `README.md` is included in that immutable checkpoint; this guide is the current reader entry point.

## Recompute the saved analysis

From the repository root, use Python 3.12 and the pinned dependencies:

```bash
python -m pip install -r extensions/adaptive_development/requirements.txt
python -m extensions.reconstruction_mechanism.evidence unpack
python -m extensions.adaptive_development.analyze --output /tmp/kahm-adaptive-analysis
python -m extensions.adaptive_development.audit \
  --analysis /tmp/kahm-adaptive-analysis \
  --output /tmp/kahm-adaptive-analysis/audit.json
```

The repository's lossless transport restores the adaptive and reconstruction-departure evidence. An expanded standalone adaptive archive already contains its evidence and does not require that transport command. The analysis and audit reconstruct losses, calibration choices, coverage, chronology, resource summaries, and decision criteria without refitting forecasting models.

## Repeat the forecasting procedure

Use a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest extensions.adaptive_development.tests_models \
  extensions.adaptive_development.tests_protocol -v
python -m extensions.adaptive_development.run --all --output /tmp/kahm-adaptive-replay
python -m extensions.adaptive_development.analyze \
  --results /tmp/kahm-adaptive-replay --output /tmp/kahm-adaptive-reanalysis
python -m extensions.adaptive_development.audit \
  --results /tmp/kahm-adaptive-replay \
  --analysis /tmp/kahm-adaptive-reanalysis \
  --output /tmp/kahm-adaptive-reanalysis/audit.json
```

Timing varies with hardware and scheduling. Numerical values can also depend on the numerical libraries and CPU-dispatch path. Repeated execution provides computational reproduction of the fixed procedure, not additional independent scientific evidence.

The optional command `python -m extensions.adaptive_development.data --prepare` recreates input archives. It requires the four original natural NPZ files and source manifest under `extensions/direct_reconstruction/data`. Ordinary forecasting replay requires only the prepared adaptive inputs.

## Interpretation of resources

The raw baseline and geometric residual predictors fit current sliding-window objectives. Adaptive KAHM, affine, and RBF-local procedures use the same reference contents and update opportunities. The kernel comparator is Gaussian sliding-window kernel ridge with recursive add/delete inverse updates and an unpenalized intercept; it is not a reproduction of a named KRLS-T algorithm.

Service time includes feature construction, newly available target updates, geometry refreshes, predictor rebuilding, kernel-inverse checks, and prediction. Model-owned NumPy buffers and whole-process resident memory are reported separately. The 20 ms observation allowance is a monitored criterion, not an enforced real-time guarantee. Geometry remains active even when calibration selects zero correction, so the measured cost is that of the specified implementation.

## Attribution

Software follows the repository MIT license. Natural observations retain the attribution and licenses stated in `DATA_PROVENANCE.md` and the prepared-input manifest. AI tools assisted implementation, numerical checks, analysis, and reporting. The verification records describe internal computational checks and do not claim external replication.
