# Adaptive KAHM development inputs

These twelve streams support a focused **development** decision. No forecast
results were used to alter the generator or select additional natural series.
The natural collections are deliberately chosen from already inspected evidence;
this is not a reserved-source confirmation study.

## Natural observations

All raw observations are copied exactly from the four preserved NPZ files at
parent commit `88e9c568ee983856804a2b2da41f6290f259b8cd`.
The parent NPZ checksums are checked against its `sources_manifest.json` before
preparation. Each new task retains the parent file, archive, and raw-value hashes.

| Collection | Preserved series | Observations each | History | Horizons | Seasonal lag |
| --- | --- | ---: | ---: | --- | ---: |
| Dominick weekly | T7830, T25851, T108533 | 393 | 52 | 1, 2, 4 | 52 |
| Sunspot daily | T1, parent final-4096 crop | 4096 | 28 | 1, 3, 7 | 1 |

Dominick previously showed a KAHM improvement over ridge. The prior Sunspot
KAHM forecast reproduced ridge and was less accurate than raw RBF. These facts
motivate a favorable and an unfavorable development collection; they do not
determine any new model's result. Every one of the three previously selected
Dominick series and the one preserved Sunspot series is included.

The source records are [Dominick v2](https://doi.org/10.5281/zenodo.4654802) and
[Sunspot v4](https://doi.org/10.5281/zenodo.4654773). Their pinned archive records
declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Archive creators
are Rakshitha Godahewa, Christoph Bergmeir, Geoff Webb, Rob Hyndman, and Pablo
Montero-Manso; each record's exact order and affiliation metadata remain in the
prepared manifest. Original sources are the University of Chicago Kilts Center
Dominick dataset and SILSO sunspot numbers. The repository software license does
not replace these data licenses.

The new procedure uses complete chronological `raw_values`, not the earlier
capped A/B/C/D window rows. In particular, no Sunspot origin subsampling is
carried into the online replay. The values are not rescaled, imputed, or
interpolated during preparation.

## Synthetic observations

Four seeds, `20261101` through `20261104`, each generate a stationary and a
recurring stream with the same innovations. Each stream has 2048 retained
observations after 256 burn-in observations, history 12, horizons 1, 4, and 12,
and seasonal lag 1. The new generator is deliberately simple development code;
it is not a recovered Duffing or Van der Pol experiment.

For regime s, the recursion is

    y_t = mu_s + a_s * (y_(t-1) - mu_s)
          + 0.10 * (y_(t-2) - mu_s)
          + b_s * tanh(1.5 * (y_(t-1) - mu_s))
          + 0.20 * epsilon_t,

where epsilon_t are independent standard normal draws from NumPy's PCG64.

| Regime | mu | a | b |
| --- | ---: | ---: | ---: |
| A | -0.75 | 0.30 | 0.35 |
| B | 0.75 | 0.45 | -0.25 |

Burn-in uses A, starting with two values equal to -0.75. The stationary stream
uses A throughout. The recurring stream alternates A/B every 256 retained
observations, beginning in A. Both regimes occur before evaluation. The
maximum sum of absolute derivatives with respect to the two lags is 0.925 in
A and 0.55 in B, a deterministic contraction check. This does not establish
a KAHM advantage or make every noisy future value predictable.

The level and conditional dynamics change observably across regimes, but finite
histories do not perfectly identify the regime under noise. No regime labels,
transition warnings, or innovations may enter a forecaster. These arrays are
stored only for independent input checks and analysis. Seed `20261099` is
reserved for excluded mechanics/timing smoke checks and does not enter the
twelve-stream study. Original generated data follow the repository's CC BY 4.0
`DATA_LICENSE`; the source code follows its MIT `LICENSE`.

## Chronology, loss scale, and availability

The initial fit uses observations `[0, floor(0.35*n))`. Calibration forecast
origins begin at `floor(0.35*n)`. Online evaluation origins begin at
`floor(0.50*n)`. Calibration selection may use only targets strictly before
the evaluation boundary. An initial training row is allowed only when all
its requested targets have matured before the initial-fit boundary. The
manifest records exact first/last origin indices and per-horizon counts.

For a forecast made at t for t+h, score the permanently saved forecast when
y_(t+h) arrives, and only then permit its supervised pair in an update. Updates
must not rewrite previously issued forecasts. Pending targets for different
horizons mature at different times.

Each task's squared-error scale is the mean squared seasonal difference using
only its initial fitting observations. If that value is nonpositive, nonfinite,
or unavailable, use first differences over the same prefix; if those also
fail, use one. The actual method and pair count are recorded. These are squared
scales: normalized MSE divides raw MSE by the recorded number, without taking
a square root. No new task invokes the fallback rules.

All selected real and generated observations are finite. The general declared
rule is nevertheless explicit: no interpolation; issue only from finite
histories and score individual horizons only when their targets are observed.
Missing targets, missing histories, and model failures are different outcomes.

## Reproduction

From the repository root:

```bash
python -m extensions.adaptive_development.data --prepare
```

`prepared_inputs/manifest.json` inventories all twelve NPZ files and their
byte-level SHA-256 values. ZIP member order, timestamps, and numeric byte order
are fixed. `load_tasks()` verifies each file and raw-value checksum before
returning it. Regenerating synthetic floating-point arrays also depends on the
recorded NumPy/runtime environment; exact hashes identify the supplied inputs.
