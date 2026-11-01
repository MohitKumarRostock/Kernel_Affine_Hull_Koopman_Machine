"""Scalar Lipschitz block witnesses with feasible quadratic-program duals.

For b independent observations, the exact score is the minimum residual sum
of squares divided by b-1.  At b=2 this is the pair witness.  The returned
lower score uses a nonnegative Lagrange multiplier, never a primal minimum.
Floating-point guards and recorded primal/dual gaps are numerical checks;
they are not interval-arithmetic proofs.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize, nnls


def block_score_bound(block_size: int, target_diameter: float) -> float:
    """Sharp range bound for scalar targets in an interval of this length."""
    if isinstance(block_size, bool) or not np.isfinite(block_size) or int(block_size) != block_size or block_size < 2:
        raise ValueError("block_size must be an integer of at least two")
    if not np.isfinite(target_diameter) or target_diameter < 0:
        raise ValueError("target_diameter must be finite and nonnegative")
    b = int(block_size)
    try:
        value = float(target_diameter) ** 2 * (b*b//4) / (b*(b-1))
    except OverflowError as exc:
        raise ValueError("target range is too large for finite arithmetic") from exc
    if not np.isfinite(value):
        raise ValueError("target range is too large for finite arithmetic")
    return value


def lipschitz_block_bracket(features, targets, kappa: float, block_slices=None):
    """Bracket min SSE/(b-1) for scalar targets on a finite feature sample.

    Euclidean feature distance is the default.  A partition in block_slices
    instead uses the maximum Euclidean block distance, matching a sum-of-block
    coefficient-norm budget. Exact repeated feature rows are aggregated.
    Every nonnegative dual vector gives a lower bound regardless of optimizer
    convergence. Consumers should retain the reported gap and feasibility.
    """
    x, y = np.asarray(features, float), np.asarray(targets, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] < 1 or y.ndim != 1 or len(y) != len(x) or len(y) < 2:
        raise ValueError("Need at least two feature rows and one scalar target per row")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Features and targets must be finite")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if block_slices is None:
        block_slices = [slice(0, x.shape[1])]
    covered = []
    for s in block_slices:
        if not isinstance(s, slice) or s.step not in (None, 1):
            raise ValueError("Blocks must be contiguous slices")
        start = 0 if s.start is None else s.start
        stop = x.shape[1] if s.stop is None else s.stop
        if not 0 <= start < stop <= x.shape[1]:
            raise ValueError("Invalid feature block")
        covered.extend(range(start, stop))
    if covered != list(range(x.shape[1])):
        raise ValueError("Blocks must partition feature coordinates exactly once")

    b = len(y)
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    if kappa == 0:
        unique, inverse = unique[:1], np.zeros(b, dtype=int)
    m = len(unique)
    weights = np.bincount(inverse, minlength=m).astype(float)
    means = np.bincount(inverse, weights=y, minlength=m)/weights
    within = float(np.sum((y-means[inverse])**2))
    differences = unique[:, None, :] - unique[None, :, :]
    distances = np.max(np.stack([
        np.linalg.norm(differences[..., s], axis=-1) for s in block_slices
    ]), axis=0)
    if not np.all(np.isfinite(distances)) or not np.isfinite(within):
        raise ValueError("Overflow in feature distances or target residuals")
    i, j = np.triu_indices(m, 1)
    matrix = np.zeros((len(i), m))
    matrix[np.arange(len(i)), i] = 1.
    matrix[np.arange(len(j)), j] = -1.
    matrix = np.vstack((matrix, -matrix))
    bound = np.tile(kappa*distances[i, j], 2)
    if not np.all(np.isfinite(bound)):
        raise ValueError("Overflow in Lipschitz constraints")

    def objective(v):
        return float(np.sum(weights*(v-means)**2))

    def gradient(v):
        return 2*weights*(v-means)

    def contract(v):
        center = float(weights@v/weights.sum())
        centered = v-center
        diffs = matrix@centered
        violated = diffs > bound
        scale = min(1., float(np.min(bound[violated]/diffs[violated]))) if violated.any() else 1.
        return center + centered*scale*(1-32*np.finfo(float).eps)

    iterations = 0
    fallback = False
    if m == 1 or np.all(matrix@means <= bound):
        fitted = means.copy()
        multipliers = np.zeros(len(bound))
    else:
        initial = np.full(m, float(weights@means/weights.sum()))
        result = minimize(objective, initial, jac=gradient, method="SLSQP",
                          constraints={"type":"ineq", "fun":lambda v:bound-matrix@v,
                                       "jac":lambda v:-matrix},
                          options={"ftol":1e-12, "maxiter":1000})
        fitted = contract(result.x)
        multipliers = np.zeros(len(bound))
        active = np.flatnonzero(bound-matrix@result.x <= 1e-7)
        if len(active):
            try:
                multipliers[active] = nnls(matrix[active].T, -gradient(result.x), maxiter=10000)[0]
            except RuntimeError:
                pass  # Zero is dual feasible; the fallback can improve it.
        iterations = int(result.nit)

    def values(v, lam):
        at = matrix.T@lam
        linear = float(lam@(matrix@means-bound))
        quadratic = float(np.sum(at*at/(4*weights)))
        raw = within+linear-quadratic
        guard = 1e-11*(1+abs(within)+abs(linear)+abs(quadratic))
        lower = max(0., raw-guard)
        primal = within+objective(v)
        upper = primal+1e-11*(1+abs(primal))
        violation = float(max(0., np.max(matrix@v-bound))) if len(bound) else 0.
        return lower, upper, violation, raw, guard

    low, high, violation, raw, guard = values(fitted, multipliers)
    if len(bound) and (high-low > 1e-7*(b-1) or violation > 1e-10):
        fallback = True
        def negative_dual(lam):
            at = matrix.T@lam
            v = means-at/(2*weights)
            value = within+lam@(matrix@means-bound)-np.sum(at*at/(4*weights))
            return -float(value), bound-matrix@v
        result = minimize(negative_dual, multipliers, jac=True, method="L-BFGS-B",
                          bounds=[(0., None)]*len(bound),
                          options={"ftol":1e-15,"gtol":1e-10,"maxiter":5000,"maxls":100})
        multipliers = np.maximum(result.x, 0.)
        candidate = contract(means-matrix.T@multipliers/(2*weights))
        if objective(candidate) < objective(fitted):
            fitted = candidate
        iterations += int(result.nit)
        low, high, violation, raw, guard = values(fitted, multipliers)
    if not all(math.isfinite(v) for v in (low, high, violation, raw, guard)):
        raise ArithmeticError("Nonfinite quadratic-program bracket")
    return {
        "block_size":b, "lower":low/(b-1), "upper":high/(b-1),
        "gap":(high-low)/(b-1), "feasibility_residual":violation,
        "sse_lower":low, "sse_upper":high, "dual_raw":raw, "dual_guard":guard,
        "support":unique.tolist(), "counts":weights.astype(int).tolist(),
        "target_means":means.tolist(), "within_sse":within,
        "primal":fitted.tolist(), "dual_multipliers":multipliers.tolist(),
        "constraint_bound":bound.tolist(), "iterations":iterations, "fallback":fallback,
    }
