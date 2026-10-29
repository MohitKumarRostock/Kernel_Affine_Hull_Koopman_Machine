# Prediction-limit certificates and representation refinement

**Kernel Affine Hull Machine forecasting and general representation analysis**

Research extension of **Prediction Limits and Koopman Closure of
Geometry-Induced Soft State Abstractions**,
[arXiv:2609.32652v3](https://arxiv.org/abs/2609.32652v3).

This directory contains an executable forecasting method, finite-sample
prediction-risk certificates, complete experiment summaries, and reproducibility
utilities. The [manuscript](paper/) develops the mathematical results, specifies
their assumptions, and reports the empirical comparisons.
The software and supporting evidence are maintained as a reusable research
resource, independently of the manuscript's publication venue.

KAHM is the central implemented representation. Its local reconstruction
models supply soft geometric associations of a causal history, and retained
dictionaries let the forecasting representation grow while preserving earlier
predictors. The forecasting procedure learns the future observed output
directly and assesses the resulting predictor and representation class using
independent episodes. The original KAHM reconstruction mechanism is inherited;
the generic risk and refinement results also apply to other feature maps.

The analysis separates the contribution of the geometric representation, the
choice of forecasting decoder, and the information supplied by finite-sample
certificates. The [scientific study guide](docs/STUDY_GUIDE.md) explains the
prediction task, the meaning of each certificate, and the evidence addressing
these three questions. The [implementation guide](IMPLEMENTATION_NOTES.md)
defines the feature and decoder notation used by the software.

## Method

1. Build causal observation histories and fit the original KAHM reconstruction
   models on training data. Evaluate soft associations with stable log-space
   normalization.
2. Retain dictionaries of 2, 4, 8, and 16 soft states. Their concatenated prefix
   dimensions are 2, 6, 14, and 30. Earlier coordinate functions remain fixed.
3. Fit a decoder that predicts the observed output at each requested forecast
   horizon. The decoder has an unpenalized intercept and a constraint on the
   **sum of block Frobenius norms** of its coefficient matrices. Predictions
   are clipped to the known target range. A computed upper bound on the
   remaining training-objective error records solver accuracy.
4. Freeze all candidate models. Use independent complete calibration episodes
   for an upper confidence bound on each fitted predictor's population MSE.
5. Form disjoint pairs of episodes and compare their targets at the same
   forecast origins. A Lipschitz constraint limits how far apart their
   predictions can be for a given feature distance. Target separation beyond
   that limit gives a lower bound on prediction risk for the entire decoder
   class. Stratified pairing makes these comparisons within groups defined
   from histories before calibration, and accounts for uncertainty in group
   probabilities. Combining class lower bounds with fitted-model upper bounds
   can certify the minimum feasible representation cost among predefined
   classes. If every class is excluded, the tolerance is infeasible within that
   family. Otherwise the available bounds may leave the decision unresolved.
   The standalone example implements common-origin calibration; the controlled
   decision study also implements stratified calibration and cost decisions.

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
- Histories, horizons, origin distribution, representations, candidates,
  sensitivity budgets, partitions, and concentration methods are fixed before
  calibration. An adaptive procedure instead requires a separately justified
  protocol with fresh evaluation data.
- The certificate concerns **expected squared error**, averaged over a new
  episode and the stated origins. It is not a conditional prediction interval
  for every future observation.

A chronological split of one dependent series does not establish these
independence conditions. The PenDigits analysis averages within writers and
uses writers as the possible independent units. Its small writer count yields
wide bounds under an unverified iid-writer sampling assumption. The separate
speech study supplies 978 calibration recording groups and informative upper
bounds under an explicitly assumed iid group-record law. Session identifiers
and a disjoint split do not establish population sampling or unique identity.

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

## Choosing KAHM input scaling

Input preprocessing is part of the fitted representation. The input-scaling
study evaluates a task-dependent modeling choice. Min–max preprocessing
improves the soft KAHM PenDigits forecaster, including new fitting seeds on the
same held-out writers, but some synthetic settings and hard KAHM assignments
regress. The default is training mean/standard-deviation standardization.
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

When `input_scaling` is omitted, the `standardize` flag determines the map. All
min/max parameters use training histories only, before geometry subsampling;
training-constant coordinates map to zero, including when a held-out input
varies along that coordinate. Targets and error units are unchanged. Choose
the scaling mode on development data and freeze it before independent
calibration. Comparing several preprocessing modes on calibration requires
a simultaneous confidence allocation over those modes. Experiment configs can opt in using `"kahm_input_scaling": "minmax"`;
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
The source hashes in each manifest identify the implementation used for that
execution; reproducing an archived run requires its recorded source version.

The [speech scaling comparison](experiments/speech_scaling/) extends all four
preprocessing modes to the speech-energy task with 2,688 paired candidate fits.
Min–max KAHM reduces mean MSE by 1.41% relative to standardized KAHM, while raw
KAHM inputs perform better still. The same comparison slightly improves RBF
features and slightly worsens affine-hull features. These are descriptive
comparisons on the earlier speech cohort; they do not receive new population
confidence guarantees. The study retains every validation candidate, frozen
scaler, selected-model group loss, and an exact source snapshot.

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

The matched candidate families are enabled by `"matched_geometry_heads": true` and
`"input_scaling_by_kind": {"kahm": "minmax", "rbf": "minmax", "affine": "minmax"}`
in configuration copies. Earlier configurations retain their original
candidate families and numerical behavior. The matched study comprises
13,440 candidate fits. Its scalar records, all writer losses, complete score
index, and analysis are retained in `results/matched_geometry_study/`.

For each synthetic task/seed, the four complete arm families share 0.04 error
probability, and six development-selected, within-head paired comparisons
share 0.01. This yields a joint confidence event of probability at least 0.95
within that replicate; the event is
not joint over every task or seed. PenDigits comparative effects are
descriptive on the original held-out writer cohort. Equal dimensions and
search options do not imply identical model capacity or computational cost.

## Certificate comparisons, cost decisions, and speech forecasting

Three studies evaluate lower-bound accuracy, representation-cost decisions,
and speech-energy forecasting. Their protocols specify the designs before
calibration observations are drawn or inspected. These archived specifications
are not external preregistrations.

- [Witness comparison](experiments/witness_comparison/): range-aware centered,
  disjoint-pair, certified empirical-minimum, and estimated-mass stratified
  bounds on the same scalar Lipschitz class. Analytic and primal/dual checks
  distinguish population slack from finite-sample penalties.
- [Decision value](experiments/decision_value/): positive-tolerance exclusions,
  minimum-cost certificates, and an irreducible control, including the
  small-sample disadvantage of stratification.
- [Speech forecasting](experiments/speech_forecasting/): a documented group
  split, a causal bounded observable, stronger conventional comparators, and
  a frozen upper-calibration protocol. The initial scoring-order deviation and
  identical strict-order replay are retained explicitly.

The concentration comparators use established bounded-loss inequalities.
The paired Lipschitz regression argument has a close antecedent in Akash et al.
(2021). The manuscript analyzes the difference between the pairwise bound and
the minimum achievable risk, conditional pairing with estimated group masses,
and decisions obtained by combining upper and lower certificates. It also
identifies when conditional bounds are sharp and their optimal decoders can
be combined into one globally admissible decoder. The manuscript's
[related-work section](paper/sections/related_work.tex) gives the literature
context.

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

| Study | Experimental scope | Main finding |
|---|---|---|
| Two-atom calibration | 28,800 trials | One observed joint certificate violation. The centered comparator in this experiment does not use the known zero input variance in the collapsed case; the eight-law comparison includes a range-aware comparator. |
| General lower-bound comparison | Eight laws, 9,600 trials, 76,800 endpoints | Empirical-minimum calibration is often tighter. Stratification recovers some population obstructions but can worsen others; all adverse outcomes are retained. |
| Positive-tolerance decision study | 200 independent regime/repetition instances, 600 fits | Stratified bounds certify minimum cost or joint infeasibility among three declared classes at large sample sizes. Upper-only selection returns the same accepted predictor. |
| Speech energy | 7,178 clips, 1,630 recording identifiers, 978 calibration groups, three fits | Six learned families certify the all-horizon 18 dB resolution under the stated sampling assumption; tighter tolerances remain unresolved. Conventional regressors are more accurate than KAHM. |
| Main dynamics | 20 complete runs, 800 candidates | Useful all-horizon KAHM certificates on some tasks, unresolved decisions on others; several simpler baselines are more accurate. |
| Norm-budget sensitivity | 36 runs, 756 candidates, fresh seeds | Increasing the budget from 1 to 4 matters; moving from 4 to 16 gives little additional KAHM benefit in these settings. |
| Van der Pol and Duffing | 6 runs, 240 candidates | Certificates refer to a bounded sensor observation; different comparison methods outperform KAHM on these examples. |
| Original PenDigits | 240 candidates, 13 eligible held-out writers | AR and quadratic regression outperform KAHM; writer-level population bounds are too wide for useful tight tolerances. |
| Matched geometry/decoder comparison | 124 jobs, 13,440 candidates, five fresh fitting seeds | KAHM improves over the named geometry in 28 of 42 task/head comparisons and regresses in 14; nonlinear AR and PenDigits show selective gains, while affine-hull controls are stronger on the oscillator tasks. These are descriptive counts. |

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
and independence assumptions. The [verification archive](docs/ARCHIVE_INDEX.md) identifies the numerical
checks, their source versions, and their scope.

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
- `docs/`: the scientific study guide, implementation context, data provenance,
  reproduction procedures, and an index of version-specific verification records.

Large generated `episode_scores.npz` files and downloaded source data are not
checked into Git. They are reproducibly regenerated by the supplied runners;
the scaling follow-up also supplies a separate primary-evidence package with
the validation-selected soft KAHM scores. The complete scaling archive index
records the hashes of all full score files. The editable manuscript is in
`paper/`, together with a compiled PDF and build instructions.
Exact-risk trial rows are retained as `trials.csv.gz`.

Original code inherits the parent repository's MIT license. Original generated
research results inherit its CC BY 4.0 terms. PenDigits and Speech Commands
retain their separately cited CC BY 4.0 licenses. The manuscript's
[related-work section](paper/sections/related_work.tex) distinguishes the
inherited statistical tools from the representation analysis developed here.
