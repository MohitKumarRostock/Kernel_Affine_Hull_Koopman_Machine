# Forecasting with Kernel Affine Hull Machines

**Representation Refinement and Risk Certification**

Research extension of **Prediction Limits and Koopman Closure of
Geometry-Induced Soft State Abstractions**,
[arXiv:2609.32652v3](https://arxiv.org/abs/2609.32652v3).

This directory contains an executable forecasting method, finite-sample
prediction-risk certificates, retained experiment summaries, and reproducibility
utilities. The [manuscript](paper/) develops the mathematical results, specifies
their assumptions, and reports the empirical comparisons.

KAHM is the central implemented representation. Its local reconstruction
models supply soft geometric associations of a causal history, and retained
dictionaries let the forecasting representation grow while preserving earlier
predictors. The new forecasting procedure learns the future observed output
directly and assesses the resulting predictor and representation class using
independent episodes. The original KAHM reconstruction mechanism is inherited;
the generic risk and refinement results also apply to other feature maps.

The analysis separates the contribution of the geometric representation, the
choice of forecasting decoder, and the information supplied by finite-sample
certificates. The [study overview](docs/MANUSCRIPT_REVISION.md) relates the
comparisons to these three questions.

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
This preserves every earlier predictor when a block is appended and leaves the
class unchanged under exact feature duplication. Ordinary Euclidean/Frobenius
stacking does not have the latter property. A squared ridge penalty itself is
not invariant to duplicating coefficient blocks.

## Assumptions and statistical interpretation

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

A chronological split of one dependent series does not establish these
independence conditions. The PenDigits analysis averages within writers and
uses writers as the possible independent units. Its small writer count yields
wide bounds under an unverified iid-writer sampling assumption.

Independence also concerns study design. Reusing calibration observations after
they informed a follow-up does not regain a confidence guarantee by fixing the
new configuration before rerunning it. The original-seed scaling follow-up and
the repeated PenDigits comparisons report descriptive endpoints. Fresh
synthetic confirmation and matched-geometry experiments use new calibration
episodes; their confidence allocations are specified separately.

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

## Optional KAHM input scaling

The input-scaling study evaluates a task-dependent modeling choice. Min–max preprocessing
improves the soft KAHM PenDigits forecaster, including new fitting seeds on the
same held-out writers, but some synthetic settings and hard KAHM assignments
regress. The original default remains training mean/std standardization.
The [retained comparison](results/input_scaling_study/) and
[manuscript section](paper/sections/input_scaling.tex) report the full comparison,
including unfavorable results and the fixed-cohort limitation.

`NestedSoftFeatures(input_scaling=...)` accepts:

| Mode | Input transformation |
|---|---|
| `standard` | Original training mean/std standardization. |
| `minmax` | Training coordinate extrema map training inputs to `[-1,1]`; new extremes can fall outside that range. |
| `minmax_clip` | The same fitted map, with transformed queries clipped to `[-1,1]`. |
| `raw` | Keep the existing raw histories, which are already bounded in these experiments. |

Omitting the new argument preserves the historical `standardize` flag. All
min/max parameters use training histories only, before geometry subsampling;
training-constant coordinates map to zero. Targets and error units are
unchanged. Experiment configs can opt in using `"kahm_input_scaling": "minmax"`;
this affects only KAHM and its dependent variants. The standalone example
accepts `--kahm-input-scaling minmax`.

Reproduce both the original-seed comparison and confirmation schedules:

```bash
python experiments/run_input_scaling.py \
  --output results/reproduced_input_scaling --workers 4 \
  --prepared data_cache/pendigits/pendigits_original.npz
```

The runner freezes new config copies and source manifests, retains all
candidates and failures, and uses separate processes with explicit per-job
scaling settings. See the reproduction guide for subsets and score audits.
Historical manifests remain pinned to their original source bytes; they are
not rewritten to match the new optional API.

## Matched geometry and forecasting heads

The [study protocol](docs/MATCHED_GEOMETRY_PROTOCOL.md) separates geometric
representation effects from the choice of Frobenius-constrained or rollout decoder.
KAHM, RBF, and local affine-hull geometries each receive all three heads at
2, 6, 14, and 30 retained features, both original histories, and the same four
preprocessing choices. Validation selects preprocessing, history, and size
separately within each geometry/head. Fixed-dimension comparisons additionally
hold representation size fixed. All seven tasks and five new fitting seeds are
retained, including unfavorable results.

Reproduce the full frozen design:

```bash
python experiments/run_matched_geometry.py \
  --output results/reproduced_matched_geometry --workers 6 \
  --prepared data_cache/pendigits/pendigits_original.npz --freeze-only
python experiments/run_matched_geometry.py \
  --output results/reproduced_matched_geometry --workers 6 \
  --prepared data_cache/pendigits/pendigits_original.npz --resume
python experiments/analyze_matched_geometry.py \
  --study results/reproduced_matched_geometry \
  --output results/reproduced_matched_geometry_analysis
```

The new controls are opt-in through `"matched_geometry_heads": true` and
`"input_scaling_by_kind": {"kahm": "minmax", "rbf": "minmax", "affine": "minmax"}`
in configuration copies. Earlier configurations retain their original
candidate families and numerical behavior. The reference follow-up comprises
13,440 candidate fits. Its scalar records, all writer losses, complete score
index, and analysis are retained in `results/matched_geometry_study/`.

For each synthetic task/seed, the four complete arm families share 0.04 error
probability, and six development-selected, within-head paired comparisons
share 0.01. This yields a joint confidence event of probability at least 0.95
within that replicate; the event is
not joint over every task or seed. PenDigits comparative effects are
descriptive on the original held-out writer cohort. Equal dimensions and
search options do not imply identical model capacity or computational cost.

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

## Experimental evidence

| Study | Retained scope | Main finding |
|---|---|---|
| Analytically known risk | 28,800 two-atom trials | One joint certificate violation; the direct pair bound can be substantially tighter than the centered comparator near collapsed input geometry. This finite simulation is not a proof of coverage. |
| Main dynamics | 20 complete runs, 800 candidates | Useful all-horizon KAHM certificates on some tasks, unresolved decisions on others; several simpler baselines are more accurate. |
| Norm-budget follow-up | 36 runs, 756 candidates, fresh seeds | Increasing the budget from 1 to 4 matters; moving from 4 to 16 gives little additional KAHM benefit in these settings. |
| Van der Pol and Duffing | 6 runs, 240 candidates | Certificates refer to a bounded sensor observation; different comparison methods outperform KAHM on these examples. |
| Original PenDigits | 240 candidates, 13 eligible held-out writers | AR and quadratic regression outperform KAHM; writer-level population bounds are too wide for useful tight tolerances. |
| Matched geometry/head follow-up | 124 jobs, 13,440 candidates, five fresh fitting seeds | KAHM improves over the named geometry in 28 of 42 task/head comparisons and regresses in 14; nonlinear AR and PenDigits show selective gains, while the new affine oscillator controls are stronger. These are descriptive counts. |

The main analysis retains 119 positive KAHM class-risk lower endpoints out of
480 candidate–horizon statements and 72 positive certified adjacent-refinement
comparisons out of 360. Confidence is joint **within the declared replicate**,
not automatically over every benchmark and repetition in the paper.

The archived independent audit reconstructed all 4,800 main one-sided endpoints and all
360 main paired comparisons with zero numerical discrepancy. A complete
harmonic replay matched every non-timing JSON result and every value in 120
score arrays exactly. Additional audit reports cover the sensitivity and
oscillator studies. These numerical checks assess implementation and retained
records; the statistical interpretation still depends on the stated sampling
and independence assumptions. The [scientific verification report](docs/SCIENTIFIC_VERIFICATION.md)
records the subsequent proof, implementation, and numerical consistency checks.

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
- `docs/`: mathematical details, related literature, data provenance, and reproduction procedures.

Large generated `episode_scores.npz` files and downloaded source data are not
checked into Git. They are reproducibly regenerated by the supplied runners;
the scaling follow-up also supplies a separate primary-evidence package with
the validation-selected soft KAHM scores. The complete scaling archive index
records the hashes of all full score files. The editable manuscript is in
`paper/`, together with a compiled PDF and build instructions.
Exact-risk trial rows are retained as `trials.csv.gz`.

Original code inherits the parent repository's MIT license. Original generated
research results inherit its CC BY 4.0 terms. PenDigits retains its cited UCI
CC BY 4.0 license. The [related-literature account](docs/LITERATURE_AUDIT.md)
identifies the inherited statistical tools and the scope of the extension.
