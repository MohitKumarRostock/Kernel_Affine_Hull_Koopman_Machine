# Cross-fitted residual corrections for KAHM forecasting: numerical amendment

The initial execution stopped on a finite-matrix ridge SVD convergence
failure before any test data existed. This directory contains the amended
complete procedure: the original solver runs first, with a logged `gesvd`
fallback only on `LinAlgError`. All scientific choices and all 192 task
seeds are retained, the exact ABC bytes are reused, and every fit is
restarted. The unchanged original attempt remains in `../crossfit_residual`.
See [NUMERICAL_AMENDMENT.md](NUMERICAL_AMENDMENT.md).

This extension tests whether centered KAHM residual corrections remain useful
when the common predictor is fitted on the full designated training block.
Four episode folds create out-of-fold residual targets. Each fold repeats the
complete common-model fitting and validation procedure without its held-out
episodes. A separately fitted full-training common predictor is used at
deployment. An in-sample residual ablation holds that final common predictor,
shared bias, correction sample, and feature representation fixed.

The numerical benchmark has 24 fresh independent repetitions, four observation
laws, two noise levels, history length 12, and horizons 1/4/12. The independent
unit of inference is a repetition. This experiment concerns the declared
numerical laws; it is not a natural-series adoption study.

## Design and source

- [PROTOCOL.md](PROTOCOL.md) defines cross-fitting, split roles, finite tuning
  grids, comparator policies, and failure handling.
- [code/numerical_solver.py](code/numerical_solver.py) defines the uniform,
  checked ridge SVD recovery rule.
- [config.json](config.json) records exact sample sizes, seeds, and choices.
- [analysis_code/ANALYSIS_PROTOCOL.md](analysis_code/ANALYSIS_PROTOCOL.md)
  defines four simultaneous primary comparisons and descriptive diagnostics.
- [code/crossfit_model.py](code/crossfit_model.py) implements the new common
  fits and centered corrections.
- [code/pipeline.py](code/pipeline.py) seals fitting and target-free forecasts
  before evaluation.
- [audit/verify_crossfit.py](audit/verify_crossfit.py) tests target exclusion,
  selection, shared-state identities, and calibration isolation on manufactured
  inputs before production.

The extension imports the verified source in `../dimension_forecasting` and
`../preprocessing_dimension`. Preserve both directories and their relative
locations. The preceding completed research release is commit
`fe8e45340c58306686a79f6e7e4be5ce6f96271a` of
[Kernel_Affine_Hull_Koopman_Machine](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine).
The new source freeze records all imported numerical files as well as the
new model, runner, protocol, and analysis definitions.

The native subspace rule remains `min(cap, d, N_local - 1)` followed by the
verified native variance and numerical range reductions. There is no extra
normal-direction reservation. New fold and correction searches use caps
1, 2, 4, 8, and 12. The inherited complete comparator also fits cap 20, which is
exactly equivalent to cap 12 at the fixed ambient dimension 12.

## Reproduce the experiment

Install [requirements.txt](requirements.txt), retain the source freeze, and
start with no generated `prepared`, `results`, `predictions`, `execution`, or
stage-lock files. From this extension:

```bash
python code/pipeline.py reuse-abc
python code/pipeline.py fit --workers 3
python code/pipeline.py seal-fits
python code/pipeline.py d
python code/pipeline.py predict --workers 3
python code/pipeline.py seal-predictions
python code/pipeline.py release
python analysis_code/analyze.py --root . --output verification/reanalysis
```

The initial source and analysis were frozen before development-data
generation. The numerical amendment is frozen and published after the
initial failed fit, before any D generation. Restore the original attempt
provenance and ABC records alongside this directory before `reuse-abc`; it
verifies and copies those exact bytes without regenerating or dropping tasks. The declared seeds are checked against all earlier study streams.
Every A/B fit and selection is completed before C arrays are loaded. All fits
are sealed before D episodes are generated, and all forecasts are sealed
before scoring. These are auditable procedural separations, not cryptographic
access isolation. Reusing the declared seeds reproduces this experiment and
does not constitute an independent new research sample.

The runner uses fresh worker processes, no more than three workers, and one
numerical thread per worker. Outputs are never silently overwritten. Any
unexpected failure stops the stage and is retained. Inherited numerical
denominator and zero-rank guards remain explicitly recorded. Every ridge
SVD recovery is recorded in `numerical_solver_events.json` and the fitted
state seal. The initial abort and its cost are reported separately.

## Numerical records and interpretation

Every task retains development and test arrays, row-aligned out-of-fold
predictions, fold membership and selected fold states, full-training common
predictions, complete B selection evidence, calibrated C predictions, final
fitted states, local-basis records, and all 42 final forecast arrays. Discarded
candidate D predictions are not part of the new scientific question and are
not retained. Full numerical records are supplied separately from Git.

Cross-fitting excludes each A episode from the common predictor forming its
own residual target. The final correction representation and slopes still
learn from all A. Fold common predictors use 48 episodes, whereas the final
common uses 64; transfer of these residuals to the larger common predictor is
the empirical question, not an assumed unbiasedness guarantee. A half-training
common-only diagnostic assesses the effect of training allocation without
claiming a comparison with the preceding half-split correction procedure.

The complete non-KAHM policy includes the previous strong forecasting
procedures and independently selected cross-fitted and in-sample RBF
corrections. Method-level timing claims must be supported by actual measured
scope; total search timings do not establish isolated deployment efficiency.
