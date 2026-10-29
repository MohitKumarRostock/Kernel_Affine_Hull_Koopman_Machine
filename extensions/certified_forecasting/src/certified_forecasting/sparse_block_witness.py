"""Matrix-free scalar Lipschitz block QPs with directly checkable duals.

For each unordered edge (i,j), let a_ij be a signed multiplier and let
div(a) = sum_j a_ij - sum_j a_ji. The dual SSE value is

    a dot (y_i-y_j) - ||div(a)||^2/4 - kappa*d_ij dot abs(a).

Every finite signed a gives feasible nonnegative multipliers by taking its
positive and negative parts, irrespective of optimizer convergence. FISTA
uses step 2/b: the complete-graph incidence Laplacian has largest eigenvalue
b, so the differentiable negative dual has Lipschitz gradient constant b/2.
Primal candidates use McShane envelopes instead of a contraction that can be
unstable when distinct feature rows are nearly coincident.

Roundoff guards and residual checks are diagnostics, not interval proofs.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _geometry(features, block_slices):
    x = np.asarray(features, dtype=float)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] < 1:
        raise ValueError('features must have shape (blocks, observations, coordinates)')
    if not np.all(np.isfinite(x)):
        raise ValueError('Nonfinite features')
    blocks = [slice(0, x.shape[2])] if block_slices is None else block_slices
    covered = []
    for s in blocks:
        if not isinstance(s, slice) or s.step not in (None, 1):
            raise ValueError('Blocks must be contiguous slices')
        start = 0 if s.start is None else s.start
        stop = x.shape[2] if s.stop is None else s.stop
        if not 0 <= start < stop <= x.shape[2]:
            raise ValueError('Invalid feature block')
        covered.extend(range(start, stop))
    if covered != list(range(x.shape[2])):
        raise ValueError('Feature blocks must partition coordinates')
    differences = x[:, :, None, :] - x[:, None, :, :]
    distance = np.maximum.reduce([np.linalg.norm(differences[..., s], axis=-1) for s in blocks])
    if not np.all(np.isfinite(distance)):
        raise ValueError('Distance overflow')
    return x, distance


def lipschitz_block_brackets(features, targets, kappa, block_slices=None,
                             max_iter=4000, tolerance=1e-8, check_every=20):
    """Return batched feasible-dual lower values for min SSE/(b-1).

    Distinct observations are retained even when their feature rows coincide.
    ``signed_dual`` uses NumPy's upper-triangular pair ordering. The returned
    dual and primal arrays suffice for verification without rerunning FISTA.
    A stopping-limit outcome is retained with its actual bracket and gap.
    """
    x, distances = _geometry(features, block_slices)
    y = np.asarray(targets, dtype=float)
    if y.shape != x.shape[:2] or not np.all(np.isfinite(y)):
        raise ValueError('Targets must be finite and match feature blocks')
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError('kappa must be finite and nonnegative')
    if max_iter < 1 or check_every < 1 or tolerance <= 0:
        raise ValueError('Invalid optimization controls')
    count, b = y.shape
    i, j = np.triu_indices(b, 1)
    edges = len(i)
    incidence = csr_matrix((np.concatenate((np.ones(edges), -np.ones(edges))),
                            (np.concatenate((i, j)), np.tile(np.arange(edges), 2))),
                           shape=(b, edges))
    bound = kappa*distances
    edge_bound = bound[:, i, j]
    dy = y[:, i]-y[:, j]
    best_dual = np.zeros((count, edges))
    best_lower = np.zeros(count)
    best_primal = np.broadcast_to(y.mean(axis=1)[:, None], y.shape).copy()
    best_upper = np.sum((y-best_primal)**2, axis=1)
    iterations = np.zeros(count, dtype=int)
    converged = np.zeros(count, dtype=bool)
    active = np.arange(count)
    alpha = np.zeros((count, edges))
    extrapolated = alpha.copy()
    momentum = 1.
    step = 2./b

    for iteration in range(1, int(max_iter)+1):
        div = (incidence @ extrapolated.T).T
        trial = extrapolated + step*(dy[active]-.5*(div[:, i]-div[:, j]))
        updated = np.sign(trial)*np.maximum(np.abs(trial)-step*edge_bound[active], 0.)
        next_momentum = (1.+np.sqrt(1.+4.*momentum**2))/2.
        next_extrapolated = updated + ((momentum-1.)/next_momentum)*(updated-alpha)
        alpha, extrapolated, momentum = updated, next_extrapolated, next_momentum
        if iteration % check_every and iteration != max_iter:
            continue
        div = (incidence @ alpha.T).T
        linear = np.sum(alpha*dy[active], axis=1)
        penalty = np.sum(np.abs(alpha)*edge_bound[active], axis=1)
        quadratic = .25*np.sum(div*div, axis=1)
        raw = linear-penalty-quadratic
        guard = 1e-11*(1.+np.abs(linear)+np.abs(penalty)+np.abs(quadratic))
        lower = np.maximum(0., raw-guard)
        better = lower > best_lower[active]
        best_lower[active[better]] = lower[better]
        best_dual[active[better]] = alpha[better]

        candidate = y[active]-.5*div
        lower_envelope = np.min(candidate[:, None, :]+bound[active], axis=2)
        upper_envelope = np.max(candidate[:, None, :]-bound[active], axis=2)
        primal = .5*(lower_envelope+upper_envelope)
        primal += (y[active]-primal).mean(axis=1)[:, None]
        primal = np.clip(primal, y[active].min(axis=1)[:, None], y[active].max(axis=1)[:, None])
        upper = np.sum((y[active]-primal)**2, axis=1)
        better = upper < best_upper[active]
        best_upper[active[better]] = upper[better]
        best_primal[active[better]] = primal[better]
        gaps = (best_upper[active]-best_lower[active])/(b-1)
        finished = gaps <= tolerance
        iterations[active] = iteration
        converged[active[finished]] = True
        active = active[~finished]
        alpha, extrapolated = alpha[~finished], extrapolated[~finished]
        if len(active) == 0:
            break

    div = (incidence @ best_dual.T).T
    linear = np.sum(best_dual*dy, axis=1)
    penalty = np.sum(np.abs(best_dual)*edge_bound, axis=1)
    quadratic = .25*np.sum(div*div, axis=1)
    raw = linear-penalty-quadratic
    guard = 1e-11*(1.+np.abs(linear)+np.abs(penalty)+np.abs(quadratic))
    lower = np.maximum(0., raw-guard)
    upper = best_upper+1e-11*(1.+np.abs(best_upper))
    violation = np.maximum(0., np.max(np.abs(best_primal[:, i]-best_primal[:, j])-edge_bound, axis=1))
    if not all(np.all(np.isfinite(z)) for z in [lower, upper, raw, guard, violation, best_dual, best_primal]):
        raise ArithmeticError('Nonfinite block bracket')
    return {'block_size': b, 'lower': lower/(b-1), 'upper': upper/(b-1),
            'gap': (upper-lower)/(b-1), 'sse_lower': lower, 'sse_upper': upper,
            'dual_raw': raw, 'dual_guard': guard, 'signed_dual': best_dual,
            'primal': best_primal, 'iterations': iterations, 'converged': converged,
            'feasibility_residual': violation}


def lipschitz_block_bracket(features, targets, kappa, block_slices=None, **options):
    """Single-block convenience wrapper preserving auditable arrays."""
    result = lipschitz_block_brackets(np.asarray(features)[None, ...],
                                     np.asarray(targets)[None, ...], kappa,
                                     block_slices=block_slices, **options)
    return {name: (value[0] if isinstance(value, np.ndarray) else value)
            for name, value in result.items()}
