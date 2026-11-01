# Reproduced numerical failure before terminal data generation

The unchanged production task `duffing__h12__noise0__rep01`, fit seed
`2073710093`, fails reproducibly at fold 0's standard-scaled, cap-12 common
ridge fit. No C targets or terminal data were read by this diagnostic.
All recorded frozen-source, prepared-input, and original failure-file hashes
were identical before and after the reproduction.

## Exact failing object

The centered float64 ridge design has shape **383 × 182**, contains only
finite values, and has maximum absolute entry 7.355786516792574. Its raw
C-order array SHA-256 is
`dff2558eab1b0c2369f9554c47e086f7189d3973b8d2b53ee236396acbd739da`.
The design and centered response are saved without modification in
`failure_00/centered.npy` and `failure_00/response.npy`. Original uncentered
features/targets and their means are saved beside them. Stack-derived role,
scaler, cap, fold, training-array hashes, and all five original penalties are
in `failure_00/metadata.json`.

The successful decomposition gives numerical rank 155 under the runtime's
existing tolerance `max(shape) * eps * largest_singular_value`, largest
singular value 39.873550389093744, and smallest singular value
3.3737011152559355e-17. Rank deficiency is observed; this is not evidence of
invalid data, and it does not make positive-penalty ridge regression undefined.

## Driver isolation

All replays use one numerical thread and the exact saved matrix.

| Driver | Repeated full decompositions | Direct LAPACK status with vectors | Status with singular values only |
|---|---:|---:|---:|
| NumPy default SVD (`gesdd`) | 3 failures / 3 attempts | — | — |
| SciPy `gesdd` | 3 failures / 3 attempts | INFO = 1 | INFO = 0 |
| SciPy `gesvd` | 3 successes / 3 attempts | INFO = 0 | INFO = 0 |

The observed cause is failure of the divide-and-conquer singular-vector
computation on a valid finite input. The underlying LAPACK secular-equation
or implementation-level mechanism has not been proven. Both installed
libraries report OpenBLAS 0.3.30. The successful `gesvd` singular values are
bit-identical across its three high-level replays, and each input copy is
unchanged after the call.

For `gesvd`, relative Frobenius reconstruction error is
**5.2471265851823384e-15**, and the left/right orthogonality Frobenius errors
are **3.592349878439816e-14 / 3.460862287926678e-14**. All factors are finite.
These results are recorded in `driver_replay.json`; direct LAPACK statuses
are in `low_level_lapack.json`.

## Ridge equivalence

The original ridge algebra is retained:

`W = V diag(s / (s² + n alpha)) Uᵀ response`.

For every original alpha in `[0.0001, 0.001, 0.01, 0.1, 1]`, the successful
decomposition gives finite coefficients and satisfies
`(XᵀX + n alpha I) W = Xᵀ response`. The worst relative normal-equation
residual is **2.1111358792028017e-14**. Against an independent positive
definite solve, the worst relative coefficient difference is
**3.0999892326925284e-12** and the worst relative fitted-training-correction
difference is **1.1429798133713167e-13**. No B, C, or D forecast loss was
compared to choose a solver.

## Minimal proposed numerical amendment

Keep the existing NumPy SVD as the first attempt. If that ridge-design SVD
raises `numpy.linalg.LinAlgError`, retry exactly the same finite matrix with
`scipy.linalg.svd(centered, full_matrices=False, lapack_driver="gesvd")`.
Retain the existing centering, rank tolerance, positive penalties, coefficient
formula, intercept treatment, selection rules, and forecast methods.

The fallback should be confined to the ridge solver, with explicit event
records. It should not globally replace NumPy SVD, perturb the matrix, drop
features, truncate singular values, change the alpha grid, or select a solver
using validation performance. A separately identified amendment can adapt the
shared ridge runtime while keeping the inherited source files unchanged.
This diagnostic has **not** installed that amendment or resumed fitting.

## Reproduction and integrity evidence

`../diagnostic.py` implements the capture/replay. During capture it calls the
original NumPy SVD unchanged, saves evidence only after the original exception,
then re-raises. `capture_receipt.json` records 205 SVD calls, one captured
failure, package/thread details, files read, and matching before/after hashes.
The original production failure remains at its original task location.

This is development-stage numerical debugging only. Cohort continuation,
amendment freeze, and any later terminal-data generation are separate decisions.
