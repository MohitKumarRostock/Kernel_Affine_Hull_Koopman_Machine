# Reproducing the certified forecasting experiments

The commands below are run from **extensions/certified_forecasting** inside
the complete parent repository. KAHM feature construction imports the
reconstruction implementation from the repository root. The
[scientific study guide](STUDY_GUIDE.md) explains the questions, estimands, and
independent sampling units. This guide specifies the computational procedures
and the evidence available for reproducing each result.

Mean squared error (MSE) is the expected squared prediction error under the
study's episode and forecast-origin law. Root-mean-square error (RMSE) is its
square root. An upper confidence endpoint evaluates a fitted predictor; a lower
endpoint applies to every decoder in the stated representation class. The
confidence allocation is stated for each study and should not be extended to
other studies or repeated fits without an additional multiplicity argument.

## Reference environment

Retained runs used Linux x86-64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, Matplotlib 3.10.8, pandas 2.2.3, and joblib 1.5.3.
The campaign manifests record exact platform and Python build strings.
Bitwise equivalence across operating systems or BLAS implementations is not
assumed.

Create an isolated environment:

    python3.12 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-reference.txt
    python -m pip install -e . --no-deps

Alternatively, installing the extension without the no-dependencies option
installs compatible dependencies whose versions may differ from the reference.
Set the thread policy before importing numerical libraries:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export PYTHONPATH=src
    export MPLCONFIGDIR="$PWD/.cache/matplotlib"
    mkdir -p "$MPLCONFIGDIR"

Synthetic experiments need no downloaded datasets. PenDigits preparation
requires network access on the first run and either the gzip executable or
the optional Python package unlzw3.

## Forecasting and two-atom experiments

| Experiment | Configuration | Retained results | Evaluation |
|---|---|---|---|
| Exact-risk two-atom experiment | configs/known_risk_v1.json | results/known_risk_v1/ | Analytically known risks, 28,800 trials |
| Independent synthetic episodes | configs/main_v1.json | results/main_v1/ | Four laws, five learning repetitions, 40 candidates, three horizons |
| Nonlinear oscillator follow-up | configs/oscillators_v1.json | results/oscillators_v1/ | Independently initialized Van der Pol and Duffing episodes, three repetitions each |
| Original PenDigits trajectories | configs/pendigits_v1.json | results/pendigits_v1b/ | Two coordinates and three seeds; writer-disjoint development and evaluation |
| Coefficient-budget sensitivity | configs/sensitivity_v1_k1.json, sensitivity_v1_k4.json, sensitivity_v1_k16.json | results/sensitivity_v1_k1/, sensitivity_v1_k4/, sensitivity_v1_k16/ | Four systems, three seeds, three budgets; common random numbers across budgets |

Reduced development runs check implementation and are excluded from the
reported results. The first PenDigits execution, `results/pendigits_v1/`,
stopped at an eligibility check before test prediction. Its records and the
correction to the eligible-writer count are documented in
`results/pendigits_v1b/RUN_NOTES.md`. The eligibility rule and forecast
protocol were unchanged, and the correction used no test loss.

The optional `results/reproduction_check/` directory contains the duplicate
run used for the archived deterministic-replay check. The audit records the
comparison and its array counts; duplicate arrays are excluded from Git.

The repository contains scalar results, candidate descriptions, writer losses,
and compressed two-atom trial records. These support independent checks of
selection, table aggregation, and confidence arithmetic from saved moments.
A full episode-score audit additionally requires the corresponding NPZ
archives, which are excluded from Git. Regenerating these archives requires
the specified source snapshot and numerical environment. The archived audit
reports describe checks performed when the full arrays were available; they
do not imply that a scalar-only checkout contains those arrays.

## Tests

    python -m unittest discover -s tests -v

The tests check the original KAHM execution path, stable normalization,
decoder feasibility and optimality diagnostics, affine Koopman composition,
causal windows, bounded episodes, independent sampling units, duplicate-block
invariance, and rejection of invalid certificate inputs. Scaling checks cover
training-only extrema, extrapolation versus clipping, constant coordinates,
exact default behavior, and saved models. Archive checks exercise complete
writes, interrupted writes, and refusal to overwrite. They do not replace the
mathematical proofs.

## KAHM input-scaling comparison

The input-scaling study compares four specified preprocessing arms while
retaining the original training mean/standard-deviation configuration as its
reference:

| Mode | Geometry input | Behavior beyond training extrema |
|---|---|---|
| `standard` | Original training mean/std standardization | Affine extrapolation |
| `minmax` | Featurewise training map `2*(x-min)/(max-min)-1` | Can leave `[-1,1]` |
| `minmax_clip` | The same fitted training min/max map | Clip to `[-1,1]` |
| `raw` | Existing raw histories, already bounded in these tasks | No extra input transform |

Min/max parameters are fitted on all training histories before the original
seeded geometry subsampling. A training-constant coordinate maps to zero.
Neither validation nor calibration/test observations refit the scaler.
Forecast targets, output clipping, loss units, and non-KAHM features retain
their original definitions. The numerical coefficient budget continues to
refer to the resulting soft-feature decoder.

The primary comparison is the validation-selected `kahm_nested` forecaster
with `minmax` versus `standard`. Histories and stages are selected separately
within each arm using the original validation criterion; the scaling arm is
not selected using test errors. Strictly bounded `minmax_clip` and already
bounded `raw` are separate diagnostic arms. All outcomes are included, including deteriorations for the PenDigits
hard-assignment variant. The PenDigits soft-feature result is specific to the
reported task and held-out cohort.

After preparing the pinned PenDigits data, reproduce both schedules into a
new study root:

    python experiments/run_input_scaling.py \
      --output results/reproduced_input_scaling --workers 4 \
      --prepared data_cache/pendigits/pendigits_original.npz

The original-seed phase evaluates the main, oscillator, three coefficient-budget,
and PenDigits configurations in all four arms. The confirmation phase runs main,
oscillator, and PenDigits configurations with fitting seeds 4701, 4702, and
4703. Synthetic confirmation regenerates all four episode streams with the
inherited offsets. PenDigits confirmation changes training-stroke subsampling
and geometry randomness while reusing the original training, validation and
held-out writers: its repeated fits assess fitting variability on a fixed cohort. The
analytic two-atom and exact duplication illustrations fit no KAHM input
geometry and are not part of the scaling intervention.

The following commands reproduce specified subsets:

    python experiments/run_input_scaling.py \
      --output results/reproduced_scaling_main \
      --phases original --campaigns main_v1 \
      --modes standard minmax --workers 2

    python experiments/run_input_scaling.py \
      --output results/reproduced_scaling_pendigits_confirmation \
      --phases confirmation --campaigns pendigits_v1 --workers 1 \
      --prepared data_cache/pendigits/pendigits_original.npz

`--campaigns` accepts the original config stems listed by `--help`.
Norm-budget sensitivity is in the original phase only; phase/campaign
selections are intersected with the fixed schedules. `--modes` selects a
predeclared subset of the four arms. `--config-dir` can point to separate
config copies for small interface checks; reduced runs serve as implementation checks. The manifest records whether
their base configurations match the reference bytes.

The output layout is:

- `results/<mode>/<campaign>/`: original-seed runs, with one subdirectory per
  synthetic system/seed. PenDigits retains its complete coordinate/seed family
  directly in its campaign directory.
- `confirmation_results/<mode>/<campaign>/`: confirmation runs in the same
  result schemas.
- `frozen_configs/<phase>/<mode>/<campaign>.json`: exact copied configuration,
  including the explicit `kahm_input_scaling` setting.
- `frozen_manifests/`, `manifest.json` and `schedule.json`: source hashes,
  versions, configuration hashes, cohort scope and the complete schedule
  written before fitting.
- `logs/` and `completion.json`: per-job logs and complete success/failure
  accounting. Failed runs retain partial artifacts and exception records.

Every output root must be a new path. The runner uses separate processes and
passes the scaling mode through each job's configuration. The original oscillator generator is temporarily
installed within its process and restored after that job. All oscillator arms
also retain the matched half-step consistency check.

Audit a new full synthetic arm using its frozen config:

    python scripts/audit_results.py \
      --campaign results/reproduced_input_scaling/results/minmax/main_v1 \
      --config results/reproduced_input_scaling/frozen_configs/original/minmax/main_v1.json \
      --output results/reproduced_input_scaling/minmax_main_audit.json

Risk allocations retain their original within-arm, within-replicate scope.
They do not provide a simultaneous event over all preprocessing modes or all
tasks. PenDigits writer-population bounds retain their explicit, unverified
iid-writer assumption. The full retained follow-up is in
`results/input_scaling_study/` and the manuscript discussion is in
`paper/sections/input_scaling.tex`.

### Optional scaling outside the study runner

Existing experiment runners honor `"kahm_input_scaling": "minmax"` in a new
config copy, applying it only to KAHM and dependent KAHM variants. Explicit
configs save the fitted scaler parameters in candidate descriptions.
Configurations without this key keep the historical result schema.

The direct Python API is `NestedSoftFeatures(input_scaling="minmax")`;
`input_scaling=None` preserves the historical `standardize` flag. The reusable
example accepts `--kahm-input-scaling minmax` or `minmax_clip` and saves the
frozen scaler together with the fitted model.

### Historical provenance and atomic score archives

The original retained campaign manifests refer to source commit
`a8bcf6c8016499e1a4b64f7dd4196af43c60bd78`. Audit those historical source hashes
in a separate checkout of that commit. The source version in a manifest is part of an experiment's identity.
Implementations with different optional preprocessing or archive-writing code
have different hashes even when their default numerical behavior agrees.
Every execution therefore records its own source and configuration hashes;
archived manifests continue to identify the sources used for those runs.

Each forecasting NPZ archive is written through a temporary file,
flushes and fsyncs it, then atomically renames it. A completed `results.json`
is published after its requested archive. An interrupted write leaves its
temporary file for diagnosis rather than exposing a truncated archive under
the completed filename. Existing archives and run directories are not
overwritten. These storage checks do not alter any fitted prediction or
statistical formula.

## Matched geometry and decoder comparison

The [matched-comparison protocol](MATCHED_GEOMETRY_PROTOCOL.md) was specified
before fitting. It compares KAHM, radial basis function (RBF), and local
affine-hull features under three decoders: direct prediction with a sum of
block Frobenius norms, direct prediction with an overall Frobenius norm, and
iterated affine feature-transition prediction. The design includes seven
tasks, four preprocessing options, and four retained feature dimensions. Seeds 5701--5705 regenerate all synthetic streams and give new
PenDigits training subsamples on the same held-out writers. Each study has its own result directory and source manifest.

The reference configurations are `configs/matched_geometry_main.json`,
`configs/matched_geometry_oscillators.json`, and
`configs/matched_geometry_pendigits.json`. Each scalar fit has 84 candidates:
72 matched geometry/head/history/prefix candidates, four KAHM hard/duplicate
controls, six AR/polynomial candidates, persistence, and a constant. The four
arms contain 120 synthetic jobs plus four PenDigits cohort jobs (ten scalar
fits each), giving 13,440 fitted candidates.

Freeze and execute into a new output root:

    python experiments/run_matched_geometry.py \
      --output results/reproduced_matched_geometry --workers 6 \
      --prepared data_cache/pendigits/pendigits_original.npz --freeze-only
    python experiments/run_matched_geometry.py \
      --output results/reproduced_matched_geometry --workers 6 \
      --prepared data_cache/pendigits/pendigits_original.npz --resume
    python experiments/analyze_matched_geometry.py \
      --study results/reproduced_matched_geometry \
      --output results/reproduced_matched_geometry_analysis

Omitting `--freeze-only` performs the same freeze immediately before execution.
Resume validates the immutable sources, protocol, configuration bytes, pinned
data, and completed artifacts. It skips verified completed jobs and refuses
to overwrite failed or partial attempts. Reduced configuration copies and
separate development seeds support implementation checks and are excluded
from the reference results.

The run root retains `manifest.json`, `protocol.md`, `protocol.json`,
`freeze_sha256.json`, `schedule.json`, all `frozen_configs/`, numerical
integrator checks, per-job logs/status, and final completion accounting.
Raw outputs are in `results/<mode>/<campaign>/`; synthetic system/seed
subdirectories contain complete scores and scalar results. PenDigits stores
all writer losses and an additional 16-family matched summary; its inherited
12-family summary is retained with its historical scope explicitly labeled.

The analysis refuses missing or duplicate tasks, seeds, modes, candidates,
feature sizes, and coordinates. It checks identical input transforms across
geometries, ordinary-control invariance, score archives, scalar aggregates,
and the recorded certificate endpoints. All model choices use validation
MSE, with the tie order fixed in the protocol. Its outputs include the full
candidate table; selected models; all-horizon and per-horizon comparisons;
fixed-dimension and head-selected secondary comparisons; paired seed effects;
writer differences; and the six primary calibration comparisons per
synthetic task/seed. The selected-score index identifies the full-precision arrays used for these
comparisons; selection depends only on validation data.

The synthetic confidence allocation differs from the earlier scaling study:
each arm receives delta=0.01 for its full risk/refinement family, and the six
validation-selected cross-geometry comparisons receive delta=0.01/6 each.
The union is 0.05 for one task/seed. Fixed-dimension and head-selected
comparisons remain descriptive. PenDigits uses delta=0.0125 per arm across
all seeds, coordinates, candidates, and horizons, giving 0.05 across arms
under its original conditional iid-writer assumption. It has no separate
calibration writer cohort for the comparative effects.

`scripts/package_matched_geometry.py` archives every scalar record and writer
loss, exact frozen source, and every full-score hash. Its separate evidence
ZIP retains the union of primary and head-selected full-precision synthetic
scores. Archive members are re-read and verified against their original
hashes; selected arrays are checked exactly against the full source archives.
Full score arrays are reproducible with the runner and are excluded from Git.

For exact fitting-source reproduction, the archived snapshot preserves both
parent helper modules and the required extension ancestry. With the reference
Python environment activated, run from the extension directory:

```bash
matched_snapshot_dir="$(mktemp -d)"
tar -xzf results/matched_geometry_study/source_snapshot.tar.gz \
  -C "$matched_snapshot_dir"
cd "$matched_snapshot_dir/extensions/certified_forecasting"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src
python experiments/prepare_pendigits.py --output-dir data_cache/pendigits
python experiments/run_matched_geometry.py \
  --output results/reproduced_matched_geometry --workers 6 \
  --prepared data_cache/pendigits/pendigits_original.npz
python experiments/analyze_matched_geometry.py \
  --study results/reproduced_matched_geometry \
  --output results/reproduced_matched_geometry_analysis
```

The snapshot contains the recorded fitting and analysis sources, not Git
history. Its documented source/configuration hashes identify the computation.
A resumed execution must use those same sources; an edited source tree requires
a separate output directory and its own manifest.

An additional standard-library implementation reconstructs the primary
selections and all 42 taskwise effects without importing the runner or analyzer:

    python scripts/verify_matched_summary.py \
      --study results/reproduced_matched_geometry \
      --analysis results/reproduced_matched_geometry_analysis \
      --output results/reproduced_matched_geometry/independent_primary_check.json

It also works on the extracted scalar/writer record archive, as explained in
`results/matched_geometry_study/README.md`; no model fits or synthetic NPZ
files are needed for that independent summary check.

## Independent-episode campaign

The runner preserves completed experiments by requiring a new destination:

    python experiments/run_forecasting.py \
      --config configs/main_v1.json \
      --output results/reproduced_main_v1

To repeat one member of the frozen schedule:

    python experiments/run_forecasting.py \
      --config configs/main_v1.json \
      --output results/reproduced_single_run \
      --system harmonic --seed 1701

Selecting a system or seed runs a subset without changing the frozen
configuration. A full-campaign audit expects all scheduled runs. Independent
jobs can be distributed by system and seed; avoid simultaneous first writes
to the same manifest or run directory.

Each completed run retains:

- **frozen_candidates.json:** development-only descriptions, validation losses,
  norm diagnostics, and optimization gaps, written before calibration data
  are generated.
- **results.json:** every candidate certificate, independent test summary,
  paired refinement statement, and certified selection, including unresolved
  choices.
- **episode_scores.npz:** full-precision calibration losses, disjoint-pair
  witnesses, and test losses for every candidate and horizon.

The campaign manifest includes the configuration hash, package versions,
runner hash, extension-source hashes, and parent Python source hashes,
including the actual KAHM implementation. A source or configuration change
requires a new campaign directory. Training, validation, calibration, and test
seed domains are recorded separately.

All candidates use the same observed targets and forecast-origin set. Overlapping windows
are dependent. The confidence sample size is 4,096 independent calibration
episodes or 2,048 disjoint episode pairs, never the number of windows.

The 95% event covers one complete frozen candidate family and all reported
paired refinements within a replicate. It does not simultaneously cover all
20 experiment repetitions. Finite test-set estimates are descriptive accuracy
measurements; agreement with an upper bound is not a population-coverage test.

## Independent checks from the scalar records

The standard-library verifier reconstructs all original empirical table rows,
validation-selected model choices, PenDigits writer aggregation and coordinate
pooling, the two-atom minimum-risk formula, and confidence arithmetic from the
saved sufficient statistics:

    python scripts/verify_retained_statistics.py \
      --output verification/retained_statistics.json

The output path must be new and outside `results/`, preserving the archived
experimental records. `--root` optionally selects another extension checkout;
`--paper` selects a separate manuscript directory. The JSON report includes
input hashes, the verifier hash, arithmetic discrepancies, table checks, and
explicit verification limits. This check requires no fitting or synthetic NPZ
archives. It verifies the reported arithmetic conditional on the saved moments;
recomputing those moments requires the episode-score audit below.

## Episode-score audit and deterministic replay

The full-score audit requires complete episode-score archives and source files
matching the campaign manifest. The main-campaign source checks refer to commit
`a8bcf6c8016499e1a4b64f7dd4196af43c60bd78`. They compare file bytes, including
comments and docstrings, with the recorded SHA-256 values. A documentation-only
edit inside a Python source file therefore changes its identity even when the
executable calculations are unchanged. Do not replace a recorded hash with
the hash of a different source version or disable the check.

To reproduce and verify the main campaign with its recorded source version,
activate the reference Python environment described above, then run the
following from the **repository root**. The fetch obtains the exact commit
when it is absent from a shallow clone; the separate worktree preserves the
current checkout.

```bash
git fetch origin a8bcf6c8016499e1a4b64f7dd4196af43c60bd78
forecast_replay_parent="$(mktemp -d)"
git worktree add --detach "$forecast_replay_parent/reference" \
  a8bcf6c8016499e1a4b64f7dd4196af43c60bd78
cd "$forecast_replay_parent/reference/extensions/certified_forecasting"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src
python experiments/run_forecasting.py \
  --config configs/main_v1.json --output results/reproduced_main_v1
python scripts/audit_results.py \
  --campaign results/reproduced_main_v1 --config configs/main_v1.json \
  --output results/reproduced_main_v1_check.json
```

This procedure generates and verifies a replay under the archived source.
Auditing the original result directory additionally requires its complete
`episode_scores.npz` files. Restoring a source checkout does not supply arrays
excluded from Git. For other campaigns, use the source commit or snapshot
identified by that campaign's own manifest; the main-campaign commit is not a
universal source version for all studies.

The commands below assume that the original full-score archives and matching
source files have been restored. They are not full-score checks of a
scalar-only checkout.

Audit the original main campaign:

    python scripts/audit_results.py \
      --campaign results/main_v1 \
      --config configs/main_v1.json \
      --output results/audits/recomputed_main_v1_audit.json

To include an independently rerun member:

    python scripts/audit_results.py \
      --campaign results/main_v1 \
      --config configs/main_v1.json \
      --replay results/reproduced_single_run \
      --output results/audits/recomputed_main_v1_replay_audit.json

For a newly generated complete campaign, change the campaign argument to
results/reproduced_main_v1. The audit independently implements the confidence
formulas, rather than calling the library certificate functions. It checks
the schedule, frozen sources/configuration, sample units, confidence budget,
recorded norms, all risk endpoints and refinement statements, test arithmetic,
and selection rules.

The retained results/audits/main_v1_audit.json reports:

- all 20 scheduled runs completed, with no missing or failed main run;
- 2,400 candidate-horizon certificates, comprising 4,800 one-sided risk
  endpoints, and 360 paired statements recomputed;
- zero difference in recomputed confidence formulas;
- an exact harmonic_1701 rerun: all non-timing JSON values and all 120 score
  arrays, containing 983,040 scalars, agree exactly.

Timing fields are excluded from replay comparisons. Norm checks use retained
diagnostics; complete coefficient fits are independently repeated only for
the replayed run. The audit verifies recorded numerical calculations and provenance; theorem
validity and scientific novelty require separate assessment.

## Nonlinear oscillator follow-up

    python experiments/run_oscillators.py \
      --config configs/oscillators_v1.json \
      --output results/reproduced_oscillators_v1

The oscillator driver supplies its generator to the shared forecasting
procedure. Its manifest hashes both scripts and the imported
implementations. Van der Pol and forced Duffing states are initialized
independently, integrated with RK4 at step 0.025, burned in for ten time units,
and observed every 0.1 time unit. The bounded sensor is
0.95 tanh(position) plus independent uniform noise in [-0.02,0.02].
The prediction certificates concern this sensor, not the unobserved physical
position or velocity.

The six-run schedule retains all 240 candidates and their full episode
scores. A matched-initialization and matched-noise half-step check is run
before forecasting. Its maximum sensor differences are 1.69e-6 for Van der
Pol and 3.49e-6 for Duffing, below the fixed 1e-3 threshold. This is a numerical
consistency check, not a rigorous global ODE integration-error certificate.

Audit this follow-up with the same independent score auditor:

    python scripts/audit_results.py \
      --campaign results/oscillators_v1 \
      --config configs/oscillators_v1.json \
      --output results/audits/recomputed_oscillators_v1_audit.json

The retained oscillator audit verifies all six runs, 720 candidate-horizon
certificates (1,440 one-sided endpoints), and 108 paired refinement statements,
with zero discrepancy in the confidence formulas.

## Exact-risk experiment

    python experiments/run_known_risk.py \
      --config configs/known_risk_v1.json \
      --output results/reproduced_known_risk_v1

Every trial is retained in `trials.csv`, compressed as `trials.csv.gz` in the
repository. Minimum achievable and fitted population risks
are analytic for the specified two-atom law, allowing actual certificate
failure counts without a finite test-set proxy. Confidence is joint over the
three reported directions within each trial, not all Monte Carlo trials.
Observed coverage on this finite grid is not a proof of general coverage.

## Original PenDigits

Prepare the pinned UCI archive:

    python experiments/prepare_pendigits.py --output-dir data_cache/pendigits

For a separately downloaded source ZIP:

    python experiments/prepare_pendigits.py \
      --archive /absolute/path/to/pendigits-source.zip \
      --output-dir data_cache/pendigits

The loader verifies archive SHA-256:

    1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64

Only the named original trajectory members are parsed. Pen-down strokes remain
separate, pen-up gaps are not treated as transitions, and no per-stroke
resampling or normalization uses future coordinates. The declared acquisition
range [0,500] is mapped by x/250 - 1. Dataset provenance, 100 ms sampling
interval, eligibility summaries, and the inferred writer-field interpretation
are recorded in data_cache/pendigits/manifest.json.

    python experiments/run_pendigits.py \
      --config configs/pendigits_v1.json \
      --prepared data_cache/pendigits/pendigits_original.npz \
      --output results/reproduced_pendigits_v1

The original development file supplies 24 training and six validation writer
groups. All six coordinate/seed families are frozen before scoring the original
test file. The 32-point first-stroke eligibility rule leaves 13 of the 14
documented held-out writers; writer 5 has no eligible first stroke. Scores
are averaged within strokes and then writers. Every candidate and its
writer-level losses are retained.

The dataset does not establish iid writer sampling. Cohort errors are
descriptive. Additional population bounds explicitly require that assumption
and are uninformative at the manuscript's small error tolerances. Windows or
strokes from one writer cannot be counted as independent writers. Variation
over training seeds is not fresh-test-cohort uncertainty.

Source: E. Alpaydin and F. Alimoglu (1996), Pen-Based Recognition of Handwritten
Digits, UCI Machine Learning Repository,
[DOI 10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K), CC BY 4.0.

## Applying the procedure to new episode data

The standalone example accepts numeric NPZ files with an array named episodes,
shaped (number of episodes, time points, 1). Each calibration row must be an
independent episode, independent of all model development and representative
of new deployment episodes. Distinct file hashes do not establish that
assumption or exclude partially overlapping datasets.

Provide known population target bounds explicitly:

    python examples/fit_and_certify.py \
      --train /absolute/path/to/train.npz \
      --calibration /absolute/path/to/calibration.npz \
      --history 8 --horizons 1 4 12 --origin-stride 4 \
      --target-min -1 --target-max 1 --rmse-tolerance 0.35 \
      --output results/my_forecaster

Replace the illustrative bounds and tolerance with ones justified for the
application. Sample minima and maxima are not population bounds. The norm
budget defaults to 4 and can be changed with the kappa argument before
calibration; it has units determined by the target and feature coordinates.

To run the independent bounded synthetic demonstration:

    python examples/fit_and_certify.py \
      --demo --target-min -1 --target-max 1 --rmse-tolerance 0.35 \
      --output results/example_forecaster

The example fits four nested KAHM stages before loading calibration
observations. Its joint two-sided confidence allocation includes all four
stages and all requested horizons. It chooses the smallest stage that passes
the requested RMSE tolerance at every horizon. If none passes, status is
**unresolved**; all four fitted models, certificates, and calibration scores
are still saved. The output directory includes:

- certificate.json: risk bounds, assumptions, and selection outcome;
- frozen_candidates.json: model descriptions fixed before calibration;
- models.pkl: the feature bank and every fitted decoder;
- calibration_scores.npz: independent-episode losses and paired witnesses;
- provenance.json: software versions and source/model hashes.

For prediction, store a numeric histories array of shape (queries, history
length) in a separate NPZ file, ordered oldest-to-newest:

    python examples/fit_and_certify.py \
      --predict-model results/my_forecaster/models.pkl \
      --histories /absolute/path/to/query_histories.npz \
      --output results/new_forecasts.npz

If calibration is unresolved, prediction requires an explicit stage choice,
for example adding `--stage 4`. The chosen forecast then has no certificate
at the requested tolerance. Even a certified stage has a population mean-error
guarantee, not a per-query prediction interval.

The Python functions fit_export, load_model, and predict_history are also
available from the example module. Load only a model pickle you created or
otherwise trust: loading an untrusted pickle can execute arbitrary code.

## Manuscript

The independent scalar-record checks can also be run without fitting models:

```bash
python scripts/verify_retained_statistics.py --output verification/retained_statistics.json
python scripts/verify_scaling_statistics.py --output verification/scaling_statistics.json
```

Both commands require new output paths and preserve the archived results.
The first reconstructs the original empirical tables, exact-risk benchmark,
writer aggregation, and confidence arithmetic from saved sufficient statistics.
The second reads the scaling record archives directly and reconstructs model
selection and mean-effect summaries. Neither reconstructs missing full synthetic
score archives. The [verification archive](ARCHIVE_INDEX.md) identifies the source versions
and scope of the available reconstruction checks.

With latexmk, pdfLaTeX, BibTeX, and the packages loaded by paper/main.tex:

    cd paper
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The compiled PDF is distributed as paper/main.pdf relative to the extension directory.
The table and figure generators derive manuscript assets from the recorded
results. Their inputs identify the numerical evidence supporting each
reported comparison.

## Practical limits

Stored training-time fields include shared geometry work for several
candidates, and some simple baselines have zero timing metadata. They do not
support a fair per-model speed ranking. Complete-run time is recorded but is
hardware dependent.

The independent-episode theorem does not automatically certify an arbitrary
single dependent series or distribution shift. Numerical-library changes may
slightly alter the reused KAHM solver's fitted values. Recompute certificates
for the actual fitted predictor and deployment sampling law.

## General lower-certificate comparison

The eight-law comparison uses fresh iid calibration draws and the same full
scalar Lipschitz class for every primary method. Its protocol distinguishes
population slack, finite-sample correction, positive-tolerance exclusion, and
certificate-only computation. Run the analytic/numerical checks and a fresh
study with:

```bash
python -m unittest discover -s experiments/witness_comparison -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/witness_comparison/run.py \
  --output results/reproduced_witness_comparison
python experiments/witness_comparison/analyze.py \
  --results results/reproduced_witness_comparison --paper paper
```

The analysis exports every trial and secondary method. Its checks reconstruct
76,800 endpoints and verify guarded primal/dual optimization brackets. The
source versions, protocol hash, and reporting provenance are documented
in `results/witness_comparison/SOURCE_AND_AUDIT.md`. Numerical brackets are
checked in floating point, not formal interval arithmetic. Each method has a
separate 0.05 failure budget; the comparison is not one joint 95% statement
across methods or simulated trials.

## Representation-cost decisions at a positive error tolerance

The controlled decision study compares three representation classes with
costs 1, 3, and 5, measured by feature-coordinate count, under a common
sensitivity budget. It compares upper-only acceptance with
ordinary or stratified lower calibration under one allocated event covering
all five sample sizes, both partitions, and all prefixes within a repetition.

```bash
python experiments/decision_value/audit_existing.py \
  --output results/reproduced_decision_existing_audit
python experiments/decision_value/run.py \
  --output results/reproduced_decision_value
python experiments/decision_value/verify.py \
  --results results/reproduced_decision_value \
  --output verification/reproduced_decision_value.json
python experiments/decision_value/make_assets.py
```

The independent verifier reconstructs streams and endpoints, enumerates the
finite population and mass-polytope vertices, and checks every fit. The asset
generator reads the retained `results/decision_value/fresh` archive; its source
states that fixed input explicitly. These constructed representations test the
general theorem and do not constitute an additional KAHM accuracy benchmark.
The release includes their compact simulated-state and coefficient archives.

## Speech energy and strict-order verification replay

The [speech study instructions](../experiments/speech_forecasting/README.md)
provide the official data URL, checksum, complete execution commands, and
source attribution. The audio archive is external; derived energy, split
identifiers, development records, selected predictors, and all group losses
are retained. Both the original execution and strict-order replay are one
study, with unchanged models and outcomes.

To verify the retained results independently after downloading the archive:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/speech_forecasting/verify_study.py \
  --archive data_cache/speech/mini_speech_commands.zip \
  --results results/speech_forecasting_v1 \
  --replay-results results/speech_forecasting_v1_replay \
  --output verification/reproduced_speech_audit.json
python experiments/speech_forecasting/analyze.py \
  --results results/speech_forecasting_v1_replay \
  --output results/reproduced_speech_analysis --paper paper
```

The first execution froze every development selection before held-out scoring
but interleaved calibration and test evaluation by model. The replay follows
the specified order: all development, all calibration, then all testing. It
reproduces every selected predictor value, group loss, and non-timing result.
The audit records the deviation and verifies equivalence rather than treating
the replay as independent evidence. All 72 upper bounds share one 0.05 budget
under the stated iid group-record assumption. Repeated fits on the same
identifiers do not create additional independent calibration groups.

The focused speech tests check frame causality, known output bounds, history
indexing, equal group weighting, and KL inversion. They are included in the
standard `python -m unittest discover -s tests -v` suite. The experiment
runners write to their specified output directories and preserve the archived
experimental records.
