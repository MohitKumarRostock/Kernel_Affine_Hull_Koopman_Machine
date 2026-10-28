# Reproducing the certified forecasting extension

Run commands from **extensions/certified_forecasting** inside the complete
parent repository. KAHM features import the original implementation from that
repository; copying only the extension directory is insufficient.

## Reference environment

Retained runs used Linux x86-64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, Matplotlib 3.10.8, and joblib 1.5.3. Exact platform and
Python build strings are in the campaign manifests. These are observed
versions, not a promise of bitwise equivalence across operating systems or
BLAS implementations.

Create an isolated environment:

    python3.12 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0 matplotlib==3.10.8 joblib==1.5.3
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

## Retained experiments

| Experiment | Configuration | Retained results | Evaluation |
|---|---|---|---|
| Exact-risk two-atom experiment | configs/known_risk_v1.json | results/known_risk_v1/ | Analytically known risks, 28,800 trials |
| Independent synthetic episodes | configs/main_v1.json | results/main_v1/ | Four laws, five learning repetitions, 40 candidates, three horizons |
| Nonlinear oscillator follow-up | configs/oscillators_v1.json | results/oscillators_v1/ | Independently initialized Van der Pol and Duffing episodes, three repetitions each |
| Original PenDigits trajectories | configs/pendigits_v1.json | results/pendigits_v1b/ | Two coordinates and three seeds; writer-disjoint development and evaluation |

Development smoke outputs are not manuscript evidence. The first PenDigits
attempt, results/pendigits_v1/, failed an eligibility check before test
prediction and is preserved as run-accounting evidence. The successful
correction is explained in results/pendigits_v1b/RUN_NOTES.md. It does not
change the forecast protocol or tune a parameter using test error.

The optional results/reproduction_check/ directory contains a duplicate run
used to verify deterministic reproduction. Its large duplicate score arrays
are not needed in the publication package; the retained audit records the
comparison.

## Tests

    python -m unittest discover -s tests -v

The tests check the original KAHM execution path, stable normalization,
decoder feasibility and optimality diagnostics, affine Koopman composition,
causal windows, bounded episodes, independent sampling units, duplicate-block
invariance, and rejection of invalid certificate inputs. They do not replace
the mathematical proofs.

## Independent-episode campaign

The runner refuses to overwrite a completed run. Use a new destination:

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

All candidates share physical targets and origin support. Overlapping windows
are dependent. The confidence sample size is 4,096 independent calibration
episodes or 2,048 disjoint episode pairs, never the number of windows.

The 95% event covers one complete frozen candidate family and all reported
paired refinements within a replicate. It does not simultaneously cover all
20 experiment repetitions. Finite test-set estimates are descriptive accuracy
measurements; agreement with an upper bound is not a population-coverage test.

## Audit results and compare a full replay

Audit the retained main campaign:

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
the replayed run. The audit does not establish novelty or prove theorems.

## Nonlinear oscillator follow-up

    python experiments/run_oscillators.py \
      --config configs/oscillators_v1.json \
      --output results/reproduced_oscillators_v1

The harness explicitly supplies its oscillator generator to the unchanged
main forecasting runner. Its manifest hashes both scripts and the imported
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

Every trial is retained in trials.csv. Oracle and fitted population risks
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

## Fit and certify your own episode data

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

If calibration was unresolved, prediction requires an explicit stage choice,
for example adding the stage argument with value 4. This makes an uncertified
forecast at the requested tolerance; it does not silently promote that stage
to certified status. Even a certified stage has a population mean-error
guarantee, not a per-query prediction interval.

The Python functions fit_export, load_model, and predict_history are also
available from the example module. Load only a model pickle you created or
otherwise trust: loading an untrusted pickle can execute arbitrary code.

## Manuscript

With latexmk, pdfLaTeX, BibTeX, and the packages loaded by paper/main.tex:

    cd paper
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The generated PDF is paper/main.pdf relative to the extension directory.
Tables and claims must remain traceable to the retained results. Hand-editing
a table does not replace regenerating its source calculation.

## Practical limits

Stored training-time fields include shared geometry work for several
candidates, and some simple baselines have zero timing metadata. They do not
support a fair per-model speed ranking. Complete-run time is recorded but is
hardware dependent.

The independent-episode theorem does not automatically certify an arbitrary
single dependent series or distribution shift. Numerical-library changes may
slightly alter the reused KAHM solver's fitted values. Recompute certificates
for the actual fitted predictor and deployment sampling law.
