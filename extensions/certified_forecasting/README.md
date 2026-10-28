# Certified geometric soft-state forecasting

Research extension of **Prediction Limits and Koopman Closure of
Geometry-Induced Soft State Abstractions**,
[arXiv:2609.32652v3](https://arxiv.org/abs/2609.32652v3).

This directory contains an executable forecasting method, finite-sample
prediction-risk certificates, complete retained experiment summaries, and
reproduction/audit tools. The manuscript is a research draft for author review;
this repository does not represent an accepted JMLR paper.

## Method

1. Build causal observation histories and fit the original KAHM reconstruction
   models on training data. Evaluate soft associations with stable log-space
   normalization.
2. Retain dictionaries of 2, 4, 8, and 16 soft states. Their concatenated prefix
   dimensions are 2, 6, 14, and 30. Earlier coordinate functions remain fixed.
3. Fit direct multi-horizon raw-output decoders with an unpenalized intercept,
   known-range output clipping, and a constraint on the **sum of block
   Frobenius norms**. An explicit convex objective gap records solver accuracy.
4. Freeze all candidate models. Use independent complete calibration episodes
   for an upper confidence bound on each fitted predictor's population MSE.
5. Pair episodes disjointly at matching forecast origins. The squared positive
   target-distance discrepancy supplies a lower bound for every decoder in a
   specified Lipschitz class. Select the least-dimensional candidate whose
   simultaneous upper bounds meet all requested tolerances, or return
   `unresolved`.

The lower, upper, and improvement statements have different meanings. A zero
lower bound does not prove a representation adequate. A large upper bound does
not prove a model inaccurate. A positive class obstruction does not prove that
a proposed refinement will help; paired episode-loss comparisons check actual
improvement separately.

The block norm matches the maximum Euclidean distance across feature blocks.
This preserves the prediction class when a block is appended and leaves the
class unchanged under exact feature duplication. Ordinary Euclidean/Frobenius
stacking does not have the latter property. A squared ridge penalty itself is
not invariant to duplicating coefficient blocks.

## What the guarantee assumes

- Calibration episodes are independent draws from the stated deployment
  episode law and independent of model construction. Windows inside an episode
  can overlap and be arbitrarily dependent.
- The prediction target has a known population range. Training minima/maxima
  and clipping predictions alone do not establish that range.
- Histories, horizons, origin distribution, representation, candidates, and
  sensitivity budgets are frozen before calibration, or a separately justified
  fresh-data adaptive protocol is used.
- The certificate concerns **expected squared error**, averaged over a new
  episode and the stated origins. It is not a conditional prediction interval
  for every future observation.

A chronological split of one dependent series is not sufficient. The real
PenDigits analysis explicitly avoids treating strokes or windows as independent
writers; its small writer count yields weak assumption-based bounds.

## Install and run

Keep this extension inside the complete parent repository because the primary
features import the original KAHM implementation and its helper modules.

```bash
cd extensions/certified_forecasting
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-reference.txt
python -m pip install -e .
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=src
python -m unittest discover -s tests -v
```

The reference environment is Python 3.12 on Linux. The requirements file pins
the main numerical libraries used for the retained results. Other platforms
may differ in floating-point details.

Run the main and analytically known-risk studies into **new directories**:

```bash
python experiments/run_forecasting.py \
  --config configs/main_v1.json --output results/reproduced_main
python experiments/run_known_risk.py --output results/reproduced_known_risk
python experiments/prepare_pendigits.py
python experiments/run_pendigits.py --help
python experiments/run_sensitivity.py --help
python experiments/run_oscillators.py --help
```

The [reproduction guide](docs/REPRODUCIBILITY.md) gives exact follow-up,
audit, figure-generation, and user-data commands. Runners refuse to overwrite
retained runs. Failures are recorded instead of silently removed.

## Use your own independently sampled episodes

The practical example fits and exports models and a calibration decision:

```bash
python examples/fit_and_certify.py --help
python examples/fit_and_certify.py \
  --train train.npz --calibration calibration.npz \
  --history 12 --horizons 1 4 12 \
  --target-min -1 --target-max 1 --rmse-tolerance 0.35 \
  --output models/my_forecaster
```

Inputs are NPZ arrays named `episodes`, with shape `(episode, time, 1)`, and
separate training/calibration files. Supply the known target range and requested
RMSE tolerance explicitly. The program saves every fitted stage, the frozen
candidate descriptions, full calibration scores, and the confidence report.
An unresolved result is preserved; prediction then requires an explicit model
choice rather than being labeled certified.

## Actual retained evidence

| Study | Retained scope | Main finding |
|---|---|---|
| Analytically known risk | 28,800 two-atom trials | One joint certificate violation; the direct pair bound can be substantially tighter than the centered comparator near collapsed input geometry. This finite simulation is not a proof of coverage. |
| Main dynamics | 20 complete runs, 800 candidates | Useful all-horizon KAHM certificates on some tasks, unresolved decisions on others; several simpler baselines are more accurate. |
| Norm-budget follow-up | 36 runs, 756 candidates, fresh seeds | Increasing the budget from 1 to 4 matters; moving from 4 to 16 gives little additional KAHM benefit in these settings. |
| Van der Pol and Duffing | 6 runs, 240 candidates | Certificates refer to a bounded sensor observation; different comparison methods outperform KAHM on these examples. |
| Original PenDigits | 240 candidates, 13 eligible held-out writers | AR and quadratic regression outperform KAHM; writer-level population bounds are too wide for useful tight tolerances. |

The main analysis retains 119 positive KAHM class-risk lower endpoints out of
480 candidate–horizon statements and 72 positive certified adjacent-refinement
comparisons out of 360. Confidence is joint **within the declared replicate**,
not automatically over every benchmark and repetition in the paper.

An independent audit reconstructed all 4,800 main one-sided endpoints and all
360 main paired comparisons with zero numerical discrepancy. A complete
harmonic replay matched every non-timing JSON result and every value in 120
score arrays exactly. Additional audit reports cover the sensitivity and
oscillator studies. These numerical checks do not replace the theorem's
assumptions or independent scientific review.

## Files and provenance

- `src/certified_forecasting/`: features, convex decoders, bounded episode
  generators, and statistical certificates.
- `experiments/`: frozen studies and the official-data downloader/parser.
- `configs/`: all retained configurations and random seeds.
- `results/`: every retained candidate's summary and certificate sufficient
  statistics, exact-risk trial archive, real-data writer losses, source hashes,
  failure records, and independent audits.
- `scripts/`: audit and publication-asset generators.
- `tests/`: mathematical invariants, causal indexing, budget/metric matching,
  optimization checks, and defensive input validation.
- `examples/`: an end-to-end fit, certify, save, and predict interface.
- `docs/`: theory, literature, data provenance, and reproducibility notes.

Large generated `episode_scores.npz` files and downloaded source data are not
checked into Git. They are reproducibly regenerated by the supplied runners;
the originally generated scores are also retained in the accompanying complete
evidence archive. The full editable manuscript is delivered separately for
author review. Exact-risk trial rows are retained as `trials.csv.gz`.

Original code inherits the parent repository's MIT license. Original generated
research results inherit its CC BY 4.0 terms. PenDigits retains its cited UCI
CC BY 4.0 license. The literature audit separates inherited statistical tools
from the proposed extension and makes no first-in-literature or acceptance claim.
