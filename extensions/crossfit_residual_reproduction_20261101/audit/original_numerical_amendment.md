# Numerical amendment: checked ridge SVD recovery before D generation

## Status and timing

This document defines the amended complete procedure for the same
cross-fitted residual study. The initial source freeze was created at
2026-11-01 14:25:39 UTC and publicly published in commit
`3b9db85c1aa8ba890dbb74cc913a9337d7360d9d`, verified at 14:33:06 UTC,
before original ABC generation. The first production fitting stage began
at 14:33:24 UTC and stopped at 14:34:50 UTC after an unhandled ridge SVD
convergence error. No D episodes, prediction seal, or terminal scores
existed when this amendment was chosen.

The amended source is frozen and publicly published after ABC preparation
and the original fitting attempt, before any D generation. It is not
described as having preceded original ABC generation. Some original tasks
had already completed calibration; the solver diagnosis and amendment
choice use finite-matrix decomposition and ridge optimality checks, not
forecast-score comparisons. The four scientific claims, margins, paired
aggregation, analysis algebra, and forecasting roster are unchanged.

The original extension, source freeze, execution records, failed task,
completed artifacts, and diagnostic evidence remain unchanged in
`../crossfit_residual`. A preservation manifest binds 144 initial-attempt
files. The stage coordinator recorded 12 completed futures before the
error; three other already-running tasks finished while the executor
shut down. Thus 15 completed fitted-task directories and one failed-task
directory are retained. None contributes a fitted state to the amended
analysis.

## Diagnosed case

| Item | Observation |
|---|---|
| Task | `duffing__h12__noise0__rep01` |
| Fit seed | `2073710093` |
| Fitting role | Fold 0 common candidate, standard scaler, native cap 12 |
| Centered design | 383 rows by 182 columns, float64, all finite |
| Maximum absolute entry | 7.355786516792574 |
| Numerical rank under the original tolerance | 155 |
| NumPy default SVD | Failed in all three exact replays |
| SciPy `gesdd` SVD | Failed in all three exact replays |
| SciPy `gesvd` SVD | Succeeded in all three exact replays |

The centered design's raw C-order SHA-256 is
`dff2558eab1b0c2369f9554c47e086f7189d3973b8d2b53ee236396acbd739da`.
Direct LAPACK checks returned `INFO=1` for `dgesdd` when singular vectors
were requested and `INFO=0` for singular values only; `dgesvd` returned
`INFO=0` in both modes. This localizes the failure to the divide-and-conquer
singular-vector calculation on this finite matrix. The internal cause is
not otherwise established; observed rank deficiency is not itself a data
error or an exclusion criterion.

The alternative decomposition had relative reconstruction error
`5.25e-15` and left/right orthogonality Frobenius errors `3.59e-14` and
`3.46e-14`. Across all five original penalties, the largest relative ridge
normal-equation residual was `2.12e-14`. Coefficients agreed with an
independent positive-definite solve within `3.10e-12` relative error.
No B/C/D predictive loss was compared to choose the solver.

These observations are retained in the original
`readiness/svd_failure_duffing_rep01` directory, including the exact arrays,
capture receipt, repeated driver replay, and low-level LAPACK statuses.

## Fixed numerical rule

The scoped adapter replaces only the inherited runtime's `_ridge_path`
entry point. It does not globally replace NumPy SVD or change native PCA,
kernel, geometry, coordinate, or reconstruction routines. The adapter is
installed before every production fit, so the same rule applies to raw
ridge, legacy controls, full-A common models, fold and half-A common models,
and both KAHM and RBF correction paths.

1. Call the original centered ridge implementation first. A successful
   result is returned unchanged.
2. Catch only `numpy.linalg.LinAlgError` from that ridge path. Other
   exceptions retain their original failure behavior.
3. Validate the same feature and target arrays and repeat the original
   centering. Require finite inputs and centered arrays. Do not perturb
   data, delete features, change rank truncation, change penalties, or
   choose a solver from predictive performance.
4. Attempt exactly one alternative decomposition with
   `scipy.linalg.svd(centered, full_matrices=False, lapack_driver="gesvd",
   check_finite=True, overwrite_a=False)`.
5. Keep the original coefficient algebra, rank tolerance, response mean,
   response scale, head fields, and per-penalty output order. For each
   original alpha, use the same regularization `n * alpha`.
6. Require finite factors and coefficients, relative reconstruction error
   at most `1e-10`, each orthogonality Frobenius error at most `1e-8`, and
   each relative ridge normal-equation residual at most `1e-8`. The precise
   normalizations and zero-norm guards are fixed in
   `code/numerical_solver.py`.
7. Record every attempted recovery with context, matrix/response hashes,
   penalties, original exception, solver provenance, numerical checks, and
   recovered or failed status. If the fallback or a validity gate fails,
   retain the event and stop execution. No third solver or silent retry is
   permitted by this amendment.

The objective is still the same centered ridge least-squares objective.
Alternative valid floating-point decompositions need not produce
bit-identical coefficients, and close validation ties could consequently
differ. That is why every task is refitted under one amended procedure and
the amended procedure itself is evaluated.

## Cohort and phase gates

Retain the same master seed `2026110173`, 24 repetitions, 192 tasks, and
1272 intended-distinct streams. All original A/B/C NPZ and base-episode
files are copied byte-for-byte and verified against the original manifest.
The same reserved D streams remain unused until every amended fit is
complete and sealed. No failing task, law, noise condition, or repetition
is replaced or omitted. New ABC generation is forbidden in the amended
production runner; its `reuse-abc` command verifies the preserved original
attempt, task identities, stream roster, and copied bytes.

All 192 fitted states are newly computed under the amended source. No
successful state from the aborted attempt is reused. B/C selection rules,
the full-A common predictor, fold exclusion, shared bias, corrections,
expanded non-KAHM policy, multiplier grids, and 42 forecast arrays are
unchanged. Fits are sealed before D generation; target-free predictions
are sealed before terminal target release and frozen analysis.

The original seed exclusions remain appropriate: this is the same study
cohort under a disclosed numerical amendment, not a fresh independent
cohort. Reusing these streams is explicitly recorded and is not counted as
another research repetition.

## Verification and reporting

Independent readiness verification checks the saved failing matrix, exact
pass-through behavior on successful original solves, exception scope,
nonfinite rejection, failed-alternative behavior, recovery logging, and
numerical identities. The actual staged CLI is also checked on a separate
manufactured fixture. These are implementation checks, not additional
scientific repetitions or solver-selection data.

The final report distinguishes the original unhandled failure and aborted
run cost from fallback activations in the amended run. It reports the
number and context of recovered SVD paths and any remaining unhandled
failure. It never describes the entire study history as having zero
numerical failures. All primary conclusions apply to the amended complete
procedures on the declared synthetic cohort.

The immutable analyzer is byte-identical to the initial frozen analyzer.
Only protocol timing, numerical recovery provenance, execution safeguards,
and audit/reporting support change; no statistical comparison is added,
removed, or redefined in response to the failure.

## Numerical documentation

- [NumPy SVD documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
  documents the `gesdd` implementation and convergence exception.
- [SciPy SVD documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)
  documents `gesdd` and `gesvd` as alternative SVD drivers and their finite
  input and convergence behavior. Neither driver is guaranteed to converge
  on every matrix.

The numerical environment used for the saved diagnosis was NumPy 2.3.5,
SciPy 1.17.0, and one numerical thread. Exact package versions and inherited
source hashes are retained in each source freeze.
