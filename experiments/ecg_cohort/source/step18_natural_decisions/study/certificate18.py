"""Finite-pool class-risk lower certificates from reused signed-edge duals.

The frozen scalar Lipschitz class is evaluated on a uniform sample WITHOUT
replacement from one fixed patient pool.  The target is risk over the entire
pool, including sampled members; it is not risk on the random complement or
on unseen patients.  Hoeffding's convex-order MGF comparison and a fixed
near-minimizer prove the same empirical-infimum binary-KL lower endpoint as
Step 17.  See manuscript/step18_finite_pool_theory.tex.

At each n, consecutive and feature-greedy disjoint partitions are solved
separately.  Both relax the SAME empirical infimum, so their pointwise maximum
uses one statistical event.  This module supplies no centered-block mean
calibration and no sample-based fitted-risk upper endpoints.

The Step 17 solver is copied byte-identically as block_solver18.py.  It solves
on nominal features; saved signed duals are then re-evaluated using upper
distances from supplied per-row, per-feature-block norm radii.  This relaxation
preserves the lower-bound direction for canonical enclosed features.  Primal
arrays certify only the relaxed finite program.  Arithmetic guards and
long-double replay are not formal interval-arithmetic proofs.
"""
from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
import time

import numpy as np

import block_solver18 as solver


PARTITIONS = ('consecutive', 'greedy_nearest')


def source_files():
    """Local source dependencies to include in the caller's source freeze."""
    return [Path(__file__).resolve(), Path(solver.__file__).resolve()]


def _scalar(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(name + ' must be a real scalar')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + ' must be finite')
    if (nonnegative and result < 0) or (positive and result <= 0):
        raise ValueError(name + ' is outside its allowed range')
    return result


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name + ' must be an integer at least ' + str(minimum))
    return int(value)


def _array(value, name, ndim):
    if np.iscomplexobj(value):
        raise ValueError(name + ' must be real')
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(name + ' must have the stated shape and be nonempty and finite')
    return np.ascontiguousarray(result)


def _alpha(value):
    result = _scalar(value, 'alpha', positive=True)
    if result >= 1:
        raise ValueError('alpha must lie strictly between zero and one')
    return result


def _sizes(values, name, minimum):
    result = tuple(_integer(v, name, minimum) for v in values)
    if not result or tuple(sorted(set(result))) != result:
        raise ValueError(name + ' must be nonempty, unique and strictly increasing')
    return result


def binary_kl(q, p):
    """Step 17 binary KL with its continuous endpoint conventions."""
    q, p = _scalar(q, 'q'), _scalar(p, 'p')
    if not 0 <= q <= 1 or not 0 <= p <= 1:
        raise ValueError('Bernoulli parameters must lie in [0,1]')
    if q == p:
        return 0.0
    if p in (0.0, 1.0):
        return math.inf
    if q == 0:
        return -math.log1p(-p)
    if q == 1:
        return -math.log(p)
    return max(0.0, q * (math.log(q) - math.log(p))
               + (1.0-q) * (math.log1p(-q) - math.log1p(-p)))


def kl_lower(q, radius):
    """Step 17 conservative numerical lower root of kl(q||p) <= radius."""
    q = _scalar(q, 'q')
    radius = _scalar(radius, 'radius', nonnegative=True)
    if not 0 <= q <= 1:
        raise ValueError('q must lie in [0,1]')
    if q == 0 or radius == 0:
        return q
    low, high = 0.0, q
    for _ in range(90):
        mid = low + (high-low)/2.0
        if mid == low or mid == high:
            break
        if binary_kl(q, mid) > radius:
            low = mid
        else:
            high = mid
    return float(np.nextafter(low, 0.0))


def finite_pool_lower(mean, n, bound, alpha, population_size):
    """One frozen-class event, conditional on a fixed finite pool and SRSWOR.

    `mean` must already be a pointwise lower bound on the sample empirical
    minimum; this function cannot check that semantic condition.  At a census
    the empirical lower is deterministic and needs no concentration penalty.
    """
    mean = _scalar(mean, 'mean', nonnegative=True)
    n = _integer(n, 'n')
    population_size = _integer(population_size, 'population_size')
    bound = _scalar(bound, 'bound', positive=True)
    alpha = _alpha(alpha)
    if n > population_size:
        raise ValueError('sample size exceeds finite population size')
    if mean > bound:
        raise ValueError('mean exceeds the declared loss range')
    radius = 0.0 if n == population_size else math.log(1.0/alpha)/n
    lower = mean if n == population_size else bound*kl_lower(mean/bound, radius)
    return {'mean':mean, 'n':n, 'bound':bound, 'alpha':alpha,
            'population_size':population_size, 'kl_radius':radius, 'lower':lower,
            'sampling':'simple random sample without replacement',
            'risk_scope':'uniform risk over the entire fixed finite pool',
            'confidence_event_required':n < population_size,
            'calibration':'exact census lower' if n == population_size else
                          'empirical-infimum binary KL via convex-order MGF'}


def _feature_slices(specification, dimension):
    result, covered = [], []
    for block in specification:
        if isinstance(block, slice):
            if block.step not in (None, 1):
                raise ValueError('feature blocks must be contiguous')
            start = 0 if block.start is None else block.start
            stop = dimension if block.stop is None else block.stop
        else:
            if len(block) != 2:
                raise ValueError('feature block needs start and stop')
            start, stop = block
        start, stop = _integer(start, 'block start', 0), _integer(stop, 'block stop')
        if not start < stop <= dimension:
            raise ValueError('invalid feature block')
        result.append(slice(start, stop))
        covered.extend(range(start, stop))
    if covered != list(range(dimension)):
        raise ValueError('feature blocks must partition all coordinates in order')
    return result


def _feature_radius(value, n, number_blocks):
    """Radius is an L2 norm bound for each feature block, not per coordinate."""
    if value is None:
        return np.zeros((n, number_blocks), dtype=np.float64)
    if np.isscalar(value):
        scalar = _scalar(value, 'feature_radius', nonnegative=True)
        return np.full((n, number_blocks), scalar, dtype=np.float64)
    radii = _array(value, 'feature_radius', 2)
    if radii.shape != (n, number_blocks) or np.any(radii < 0):
        raise ValueError('feature_radius must have shape (observations, feature blocks) and be nonnegative')
    return radii


def make_partition(features, block_size, method):
    """Build a deterministic complete partition within exactly these rows.

    Greedy distance is ordinary Euclidean distance on concatenated nominal
    features.  It only chooses a valid relaxation; the QP class metric remains
    the maximum Euclidean distance over the declared feature blocks.
    """
    x = _array(features, 'features', 2)
    b = _integer(block_size, 'block_size', 2)
    n = len(x)
    if n % b:
        raise ValueError('complete partitions require n divisible by block_size')
    if method not in PARTITIONS:
        raise ValueError('unknown partition method')
    if method == 'consecutive':
        return np.arange(n, dtype=np.int64).reshape(-1, b)
    unused = np.ones(n, dtype=bool)
    rows = []
    while np.any(unused):
        candidates = np.flatnonzero(unused)
        anchor = int(candidates[0])
        rest = candidates[1:]
        difference = x[rest]-x[anchor]
        squared_distance = np.einsum('ij,ij->i', difference, difference)
        if not np.isfinite(squared_distance).all():
            raise ArithmeticError('greedy partition distance overflow')
        order = np.lexsort((rest, squared_distance))
        group = np.concatenate(([anchor], rest[order[:b-1]])).astype(np.int64)
        if len(group) != b:
            raise ArithmeticError('incomplete greedy block')
        unused[group] = False
        rows.append(group)
    indices = np.stack(rows)
    if not np.array_equal(np.sort(indices.ravel()), np.arange(n)):
        raise ArithmeticError('greedy partition is not a disjoint complete cover')
    return indices


def upper_distances(features, feature_block_radius, block_slices):
    """Conservative-distance relaxation for enclosed canonical features.

    Returned arrays have shape (QP blocks, b, b).  Off diagonal,
    d_upper=max_s(||x_i^s-x_j^s||2+eps_is+eps_js).  Long-double accumulation
    followed by an outward float64 step reduces numerical under-rounding;
    this is explicitly not a verified interval norm implementation.
    """
    x = np.asarray(features, dtype=np.float64)
    eps = np.asarray(feature_block_radius, dtype=np.float64)
    if x.ndim != 3 or eps.shape != (x.shape[0], x.shape[1], len(block_slices)):
        raise ValueError('batched feature radii have incompatible shape')
    if not np.isfinite(x).all() or not np.isfinite(eps).all() or np.any(eps < 0):
        raise ValueError('nonfinite features or invalid radii')
    difference = x[:, :, None, :]-x[:, None, :, :]
    nominal = np.maximum.reduce([np.linalg.norm(difference[..., s], axis=-1)
                                  for s in block_slices])
    extended = x.astype(np.longdouble)
    candidates = []
    for k, s in enumerate(block_slices):
        dx = extended[:, :, None, s]-extended[:, None, :, s]
        norm = np.sqrt(np.sum(dx*dx, axis=-1, dtype=np.longdouble))
        radius = eps[..., k].astype(np.longdouble)
        value = norm+radius[:, :, None]+radius[:, None, :]
        candidates.append(np.nextafter(np.asarray(value, dtype=np.float64), np.inf))
    upper = np.maximum(nominal, np.maximum.reduce(candidates))
    diagonal = np.arange(x.shape[1])
    upper[:, diagonal, diagonal] = 0.0
    nominal[:, diagonal, diagonal] = 0.0
    if not np.isfinite(upper).all() or np.any(upper < nominal):
        raise ArithmeticError('invalid conservative feature distances')
    return nominal, upper


def correct_dual_bracket(bracket, features, targets, feature_block_radius,
                         block_slices, kappa, tolerance):
    """Re-evaluate the existing signed dual on a relaxed upper-distance QP.

    Any finite signed dual remains valid irrespective of the nominal solver's
    convergence.  The original nominal primal is feasible for the relaxed
    distances; it is not asserted to interpolate the canonical feature class.
    """
    result = dict(bracket)
    y = np.asarray(targets, dtype=np.float64)
    count, b = y.shape
    ii, jj = np.triu_indices(b, 1)
    alpha = np.asarray(bracket['signed_dual'], dtype=np.float64)
    if alpha.shape != (count, len(ii)):
        raise ValueError('signed-dual dimensions do not match the supplied blocks')
    nominal, upper = upper_distances(features, feature_block_radius, block_slices)
    divergence = np.zeros((count, b), dtype=np.float64)
    for vertex in range(b):
        divergence[:, vertex] = (np.sum(alpha[:, ii == vertex], axis=1)
                                 -np.sum(alpha[:, jj == vertex], axis=1))
    linear = np.sum(alpha*(y[:, ii]-y[:, jj]), axis=1)
    penalty = kappa*np.sum(np.abs(alpha)*upper[:, ii, jj], axis=1)
    quadratic = .25*np.sum(divergence*divergence, axis=1)
    raw = linear-penalty-quadratic
    guard = 1e-11*(1.0+np.abs(linear)+np.abs(penalty)+np.abs(quadratic))
    nominal_lower = np.asarray(bracket['sse_lower']).copy()
    # The cap only lowers the result; it also makes uncertainty correction
    # monotone despite a different floating reduction order during replay.
    lower = np.minimum(nominal_lower, np.maximum(0.0, raw-guard))
    sse_upper = np.asarray(bracket['sse_upper']).copy()
    primal = np.asarray(bracket['primal'])
    violation = np.maximum(0.0, np.max(np.abs(primal[:, ii]-primal[:, jj])
                                      -kappa*upper[:, ii, jj], axis=1))
    penalty_inflation = kappa*np.sum(np.abs(alpha)*(upper[:, ii, jj]-nominal[:, ii, jj]), axis=1)
    arrays = (raw, guard, lower, sse_upper, violation, penalty_inflation, upper)
    if not all(np.isfinite(a).all() for a in arrays):
        raise ArithmeticError('nonfinite uncertainty-corrected dual bracket')
    if np.any(lower > sse_upper+1e-8) or np.max(violation) > 1e-8:
        raise ArithmeticError('uncertainty-corrected bracket failed its feasibility/SSE checks')
    result.update(nominal_sse_lower=nominal_lower,
                  nominal_sse_upper=np.asarray(bracket['sse_upper']).copy(),
                  nominal_dual_raw=np.asarray(bracket['dual_raw']).copy(),
                  nominal_dual_guard=np.asarray(bracket['dual_guard']).copy(),
                  nominal_converged=np.asarray(bracket['converged']).copy(),
                  nominal_gap=np.asarray(bracket['gap']).copy(),
                  nominal_distance=nominal, distance_upper=upper,
                  feature_block_radius=np.asarray(feature_block_radius).copy(),
                  dual_linear=linear, dual_penalty=penalty, dual_quadratic=quadratic,
                  dual_raw=raw, dual_guard=guard, sse_lower=lower,
                  sse_upper=sse_upper, lower=lower/(b-1), upper=sse_upper/(b-1),
                  gap=(sse_upper-lower)/(b-1),
                  converged=(sse_upper-lower)/(b-1) <= tolerance,
                  feasibility_residual=violation,
                  dual_penalty_inflation=penalty_inflation,
                  sse_lower_uncertainty_adjustment=nominal_lower-lower,
                  primal_scope='feasible for relaxed upper-distance finite QP only',
                  gap_normalization='SSE/(b-1), solver diagnostic only; no block-mean inference')
    return result


def _guarded_sum_lower(values):
    """Downward floating reduction of already guarded nonnegative values."""
    total = float(np.sum(np.asarray(values), dtype=np.longdouble))
    return 0.0 if total == 0 else float(np.nextafter(total, 0.0))


def evaluate_calibration(features, targets, predictions=None, *, sample_sizes,
                         block_size=32, block_slices, kappa,
                         target_bounds=(-1.0, 1.0), alpha=.05/9,
                         population_size=8192, feature_radius=None,
                         solver_config):
    """Evaluate all prespecified n values for ONE frozen scalar Lipschitz class.

    `features` and `targets` are in SRSWOR calibration order, not sorted by
    outcome.  Each n prefix is partitioned and solved separately.  The caller
    must provide the sampling provenance and freeze the class before sampling.

    `feature_radius[i,s]` bounds the L2 error of feature block s at row i.
    A nonnegative scalar can specify the same bound for every row/block.
    Coordinatewise radii must be converted to block norm radii by the caller.
    Optional `predictions` are nominal frozen-head outputs used only for
    relaxed-program feasibility and empirical-SSE sanity checks; canonical
    full-pool fitted-risk bounds are the caller's separate census calculation.

    Returns JSON-compatible scalar records plus auditable NumPy arrays in
    block_evidence[str(n)][partition_name].  Every evidence entry includes
    the exact calibration-row indices, saved signed duals, primal values,
    corrected and nominal SSEs, conservative distances, radii, and gaps.
    """
    x = _array(features, 'features', 2)
    y = _array(targets, 'targets', 1)
    if len(x) != len(y):
        raise ValueError('features and targets must share observations')
    pred = None if predictions is None else _array(predictions, 'predictions', 1)
    if pred is not None and len(pred) != len(y):
        raise ValueError('predictions must share observations')
    ns = _sizes(sample_sizes, 'sample_sizes', 2)
    b = _integer(block_size, 'block_size', 2)
    M = _integer(population_size, 'population_size')
    if max(ns) != len(y) or min(ns) < b or max(ns) > M:
        raise ValueError('sample sizes must end at supplied n and fit block/population sizes')
    if any(n % b for n in ns):
        raise ValueError('each sample size must be divisible by block_size')
    slices = _feature_slices(block_slices, x.shape[1])
    radius = _feature_radius(feature_radius, len(y), len(slices))
    kappa = _scalar(kappa, 'kappa', nonnegative=True)
    alpha = _alpha(alpha)
    if len(target_bounds) != 2:
        raise ValueError('target_bounds needs two known deterministic endpoints')
    low, high = [_scalar(v, 'target bound') for v in target_bounds]
    if not low < high:
        raise ValueError('target interval must have positive diameter')
    if np.any(y < low) or np.any(y > high):
        raise ValueError('targets lie outside the declared interval')
    if pred is not None and (np.any(pred < low) or np.any(pred > high)):
        raise ValueError('predictions lie outside the declared interval')
    bound = (high-low)**2
    if not math.isfinite(bound):
        raise ValueError('target loss range overflow')
    if set(solver_config) != {'max_iter', 'tolerance', 'check_every'}:
        raise ValueError('solver_config must specify all three frozen controls')
    config = dict(max_iter=_integer(solver_config['max_iter'], 'max_iter'),
                  tolerance=_scalar(solver_config['tolerance'], 'tolerance', positive=True),
                  check_every=_integer(solver_config['check_every'], 'check_every'))
    records, evidence = [], {}
    program_count = 0
    total_solve = total_partition = total_correction = 0.0
    for n in ns:
        local, summary = {}, []
        for name in PARTITIONS:
            start = time.perf_counter()
            indices = make_partition(x[:n], b, name)
            partition_seconds = time.perf_counter()-start
            xb, yb, rb = x[indices], y[indices], radius[indices]
            start = time.perf_counter()
            raw = solver.lipschitz_block_brackets(xb, yb, kappa,
                                                  block_slices=slices, **config)
            solve_seconds = time.perf_counter()-start
            start = time.perf_counter()
            bracket = correct_dual_bracket(raw, xb, yb, rb, slices, kappa, config['tolerance'])
            correction_seconds = time.perf_counter()-start
            bracket['indices'] = indices
            bracket['constant_sse'] = np.sum((yb-yb.mean(axis=1)[:, None])**2, axis=1)
            sharp_sse_bound = bound*(b*b//4)/b
            if np.any(bracket['sse_lower'] < 0) or np.any(bracket['sse_lower'] > sharp_sse_bound):
                raise ArithmeticError('dual SSE exceeds the universal bounded-target range')
            if np.any(bracket['sse_lower'] > bracket['constant_sse']+1e-8):
                raise ArithmeticError('dual exceeds a feasible constant empirical SSE')
            if pred is not None:
                pb = pred[indices]
                ii, jj = np.triu_indices(b, 1)
                pv = np.maximum(0.0, np.max(np.abs(pb[:, ii]-pb[:, jj])
                                           -kappa*bracket['distance_upper'][:, ii, jj], axis=1))
                fitted_sse = np.sum((yb-pb)**2, axis=1)
                bracket['nominal_fitted_sse'] = fitted_sse
                bracket['nominal_fitted_relaxed_feasibility_residual'] = pv
                if np.max(pv) > 1e-8:
                    raise ArithmeticError('nominal fitted predictions violate the declared class budget')
                if np.any(bracket['sse_lower'] > fitted_sse+1e-8):
                    raise ArithmeticError('dual exceeds a feasible fitted empirical SSE')
            sse_sum = _guarded_sum_lower(bracket['sse_lower'])
            mean = 0.0 if sse_sum == 0 else float(np.nextafter(sse_sum/n, 0.0))
            local[name] = bracket
            summary.append({'partition':name, 'n_blocks':len(indices),
                            'covered':int(indices.size), 'uncovered':0,
                            'sse_sum':sse_sum, 'empirical_minimum_lower':mean,
                            'max_solver_gap':float(np.max(bracket['gap'])),
                            'converged_blocks':int(np.sum(bracket['converged'])),
                            'iteration_limit_blocks':int(np.sum(~raw['converged'])),
                            'max_feasibility_residual':float(np.max(bracket['feasibility_residual'])),
                            'total_dual_penalty_inflation':float(np.sum(bracket['dual_penalty_inflation'])),
                            'total_sse_uncertainty_adjustment':float(np.sum(bracket['sse_lower_uncertainty_adjustment'])),
                            'partition_seconds':partition_seconds,
                            'solve_seconds':solve_seconds,
                            'correction_seconds':correction_seconds})
            program_count += len(indices)
            total_partition += partition_seconds
            total_solve += solve_seconds
            total_correction += correction_seconds
        winner = max(range(len(summary)), key=lambda k: summary[k]['empirical_minimum_lower'])
        endpoint = finite_pool_lower(summary[winner]['empirical_minimum_lower'], n, bound, alpha, M)
        empirical_mse = None if pred is None else float(np.mean((y[:n]-pred[:n])**2))
        if empirical_mse is not None and endpoint['mean'] > empirical_mse+1e-8:
            raise ArithmeticError('partition maximum exceeds nominal fitted empirical risk')
        records.append({'n':n, 'lower':endpoint['lower'], 'recycled':endpoint,
                        'selected_partition':summary[winner]['partition'],
                        'partitions':summary, 'empirical_nominal_fitted_mse':empirical_mse,
                        'one_event_for_both_partitions':True,
                        'confidence_events':{'class_lower':int(n < M)},
                        'full_coverage':True, 'block_programs':2*(n//b),
                        'statistical_denominator':n,
                        'no_centered_block_statistical_calibration':True})
        evidence[str(n)] = local
    return {'records':records, 'block_evidence':evidence,
            'sample_sizes':list(ns), 'block_size':b, 'partitions':list(PARTITIONS),
            'block_slices':[(s.start,s.stop) for s in slices], 'kappa':kappa,
            'target_bounds':[low,high], 'bound':bound, 'alpha':alpha,
            'population_size':M, 'solver_config':config,
            'total_block_programs':program_count,
            'total_partition_seconds':total_partition,
            'total_block_solve_seconds':total_solve,
            'total_dual_correction_seconds':total_correction,
            'maximum_feature_block_radius':float(np.max(radius)),
            'sampling':'simple random sample without replacement; provenance checked by caller',
            'statistical_unit':'one prespecified observation per patient in a fixed finite pool',
            'risk_scope':'uniform risk over entire fixed pool, including calibration patients',
            'numeric_scope':'guarded float64 duals/KL and long-double distance replay; not formal interval proof'}
