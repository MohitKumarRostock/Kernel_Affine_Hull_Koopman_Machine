"""Matched-computation lower calibration for one frozen Lipschitz class.

The disjoint-block solver is an unchanged copy of the manuscript's existing
signed-edge solver.  The new comparison recycles the SAME dual SSE values as
a lower bound on the full-sample empirical infimum.  It therefore uses n in
the existing empirical-minimum KL theorem, with loss range D**2, rather than
K=floor(n/b) in block-mean calibration, with the sharp block-score range.

Taking a maximum over valid numerical lower values for this ONE empirical
infimum costs no additional confidence event.  It does not authorize choosing
the representation, decoder class, target law, or norm budget on calibration.
Numerical dual guards and replay checks are not interval-arithmetic proofs.
"""
from __future__ import annotations

import math
import time
from numbers import Integral

import numpy as np

from block_solver17 import lipschitz_block_brackets


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
    """Binary KL divergence with its continuous endpoint conventions."""
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
    """Conservative numerical lower root of kl(q||p) <= radius."""
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


def kl_upper(q, radius):
    """Conservative numerical upper root of kl(q||p) <= radius."""
    q = _scalar(q, 'q')
    radius = _scalar(radius, 'radius', nonnegative=True)
    if not 0 <= q <= 1:
        raise ValueError('q must lie in [0,1]')
    if q == 1 or radius == 0:
        return q
    low, high = q, 1.0
    for _ in range(90):
        mid = low + (high-low)/2.0
        if mid == low or mid == high:
            break
        if binary_kl(q, mid) > radius:
            high = mid
        else:
            low = mid
    return float(np.nextafter(high, 1.0))


def kl_mean_lower(mean, n, bound, alpha):
    """One lower event for a bounded iid mean or a justified empirical infimum."""
    mean = _scalar(mean, 'mean', nonnegative=True)
    n = _integer(n, 'n')
    bound = _scalar(bound, 'bound', positive=True)
    alpha = _alpha(alpha)
    if mean > bound:
        raise ValueError('mean exceeds its declared population range')
    radius = math.log(1.0/alpha)/n
    return {'mean':mean, 'n':n, 'bound':bound, 'alpha':alpha,
            'kl_radius':radius, 'lower':bound*kl_lower(mean/bound, radius)}


def kl_mean_upper(values, bound, alpha):
    """One upper event for fixed fitted losses, one scalar per iid episode."""
    x = _array(values, 'losses', 1)
    bound = _scalar(bound, 'bound', positive=True)
    alpha = _alpha(alpha)
    if np.any(x < 0) or np.any(x > bound):
        raise ValueError('losses exceed their declared population range')
    mean = float(np.mean(x))
    radius = math.log(1.0/alpha)/len(x)
    return {'mean':mean, 'n':len(x), 'bound':bound, 'alpha':alpha,
            'kl_radius':radius, 'upper':bound*kl_upper(mean/bound, radius)}


def block_score_bound(block_size, target_diameter):
    b = _integer(block_size, 'block_size', 2)
    diameter = _scalar(target_diameter, 'target_diameter', positive=True)
    value = diameter*diameter*(b*b//4)/(b*(b-1))
    if not math.isfinite(value):
        raise ValueError('target range overflow')
    return value


def _sse_lower(values, block_size, target_diameter):
    b = _integer(block_size, 'block_size', 2)
    values = _array(values, 'block SSE lower values', 1)
    bound = block_score_bound(b, target_diameter)
    if np.any(values < 0) or np.any(values > (b-1)*bound):
        raise ValueError('SSE lower values lie outside the universal block range')
    return values, b, bound


def block_mean_lower(sse_lower, block_size, target_diameter, alpha):
    """KL calibration of exact block scores from pointwise dual lower values.

    Blocks must be disjoint iid blocks of the given size. Numerical outputs
    may share computation: monotonicity compares their mean to the iid exact
    block-score mean. The pointwise dual inequality is required independently
    of these statistical calculations.
    """
    values, b, bound = _sse_lower(sse_lower, block_size, target_diameter)
    mean = float(np.mean(values))/(b-1)
    record = kl_mean_lower(mean, len(values), bound, alpha)
    record.update(block_size=b, n_blocks=len(values), normalization='SSE/(b-1)',
                  sse_sum=float(np.sum(values)))
    return record


def recycled_partition_lower(sse_lower, block_size, n, target_diameter, alpha):
    """Reuse disjoint-block duals as a full-sample empirical-minimum lower value.

    The supplied blocks cover distinct observations among n. Their number may
    be smaller than floor(n/b); all uncovered observations contribute zero.
    The statistical denominator remains n. In evaluate_calibration the blocks
    are the consecutive complete blocks and this disjointness is constructed.
    """
    values, b, _ = _sse_lower(sse_lower, block_size, target_diameter)
    n = _integer(n, 'n')
    if b*len(values) > n:
        raise ValueError('declared disjoint blocks cover more than n observations')
    bound = float(target_diameter)**2
    sse_sum = float(np.sum(values))
    record = kl_mean_lower(sse_sum/n, n, bound, alpha)
    record.update(block_size=b, n_blocks=len(values), covered=b*len(values),
                  uncovered=n-b*len(values), sse_sum=sse_sum,
                  normalization='sum block dual SSE/n',
                  endpoint_role='lower bound on the same full empirical infimum')
    return record


def recycle_maximum(partition_records):
    """Maximize valid lower values for ONE a_n, with one confidence event.

    All records must describe the same n, loss range, and alpha. A caller must
    also ensure the same representation, metric, target law and class. The
    statistical theorem does not provide that semantic check.
    """
    records = list(partition_records)
    if not records:
        raise ValueError('at least one partition record is required')
    first = records[0]
    if any(any(r[key] != first[key] for key in ('n','bound','alpha')) for r in records):
        raise ValueError('partition records do not refer to the same empirical-infimum event')
    winner = max(range(len(records)), key=lambda i: records[i]['mean'])
    best = records[winner]
    result = kl_mean_lower(best['mean'], first['n'], first['bound'], first['alpha'])
    result.update(selected_block_size=best['block_size'], selected_partition_index=winner,
                  one_event_for_all_partitions=True,
                  per_partition_lower=[r['lower'] for r in records])
    return result


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


def evaluate_calibration(features, targets, predictions, *, sample_sizes,
                         block_sizes, block_slices, kappa, target_bounds,
                         alpha, solver_config):
    """Evaluate one frozen fitted class on fresh scalar-target observations.

    Returns ``records`` (one JSON-compatible dict per n) and ``block_evidence``
    (one solver result per b, containing all dual/primal arrays). Solve once
    on the complete supplied sample and reuse consecutive blocks for each
    prefix. The same solved SSE values serve BOTH lower methods. Calibration
    never modifies predictions or chooses a representation or fitted head.
    """
    x = _array(features, 'features', 2)
    y = _array(targets, 'targets', 1)
    pred = _array(predictions, 'predictions', 1)
    if len(x) != len(y) or len(pred) != len(y):
        raise ValueError('features, targets and predictions must share observations')
    ns = _sizes(sample_sizes, 'sample_sizes', 2)
    bs = _sizes(block_sizes, 'block_sizes', 2)
    if max(ns) != len(y) or min(ns) < max(bs):
        raise ValueError('sample roster must end at supplied n and include every block size')
    slices = _feature_slices(block_slices, x.shape[1])
    kappa = _scalar(kappa, 'kappa', nonnegative=True)
    alpha = _alpha(alpha)
    if len(target_bounds) != 2:
        raise ValueError('target_bounds needs two known deterministic endpoints')
    low, high = [_scalar(v, 'target bound') for v in target_bounds]
    if not low < high:
        raise ValueError('target interval must have positive diameter')
    if np.any(y < low) or np.any(y > high) or np.any(pred < low) or np.any(pred > high):
        raise ValueError('targets and frozen predictions must lie in the declared interval')
    D = high-low
    bound = D*D
    if not math.isfinite(bound):
        raise ValueError('target loss range overflow')
    allowed = {'max_iter','tolerance','check_every'}
    if set(solver_config) != allowed:
        raise ValueError('solver_config must specify all three frozen controls')
    config = dict(max_iter=_integer(solver_config['max_iter'], 'max_iter'),
                  tolerance=_scalar(solver_config['tolerance'], 'tolerance', positive=True),
                  check_every=_integer(solver_config['check_every'], 'check_every'))
    evidence, solve_seconds = {}, {}
    for b in bs:
        K = len(y)//b
        start = time.perf_counter()
        bracket = lipschitz_block_brackets(x[:K*b].reshape(K,b,-1),
                    y[:K*b].reshape(K,b), kappa, block_slices=slices, **config)
        solve_seconds[str(b)] = time.perf_counter()-start
        _sse_lower(bracket['sse_lower'], b, D)
        evidence[str(b)] = bracket

    losses = (y-pred)**2
    records = []
    for n in ns:
        post_seconds = {'upper':0.0, 'block_KL':0.0, 'recycled_KL':0.0}
        start = time.perf_counter()
        upper = kl_mean_upper(losses[:n], bound, alpha)
        post_seconds['upper'] += time.perf_counter()-start
        block, recycled = [], []
        for b in bs:
            values = evidence[str(b)]['sse_lower'][:n//b]
            start = time.perf_counter()
            block.append(block_mean_lower(values, b, D, alpha))
            post_seconds['block_KL'] += time.perf_counter()-start
            start = time.perf_counter()
            recycled.append(recycled_partition_lower(values, b, n, D, alpha))
            post_seconds['recycled_KL'] += time.perf_counter()-start
        block_winner = max(range(len(bs)), key=lambda i: block[i]['lower'])
        start = time.perf_counter()
        recycled_max = recycle_maximum(recycled)
        post_seconds['recycled_KL'] += time.perf_counter()-start
        hierarchy_lower = block[block_winner]['lower']
        combined_lower = max(hierarchy_lower, recycled_max['lower'])
        records.append({'n':n, 'upper':upper, 'block':block,
                        'hierarchy_lower':hierarchy_lower,
                        'hierarchy_selected_block_size':bs[block_winner],
                        'recycled_per_block':recycled,
                        'recycled':recycled_max,
                        'combined_lower':combined_lower,
                        'combined_selected_method':('hierarchy' if hierarchy_lower >= recycled_max['lower']
                                                    else 'recycled'),
                        'combined_has_no_additional_event':True,
                        'postprocessing_seconds':post_seconds,
                        'confidence_events':{'fitted_upper':1,
                            'block_lower':len(bs), 'recycled_lower':1},
                        'same_solved_dual_values':True})
    return {'records':records, 'block_evidence':evidence,
            'block_solve_seconds':solve_seconds,
            'solver_config':config,
            'block_slices':[(s.start,s.stop) for s in slices],
            'sample_sizes':list(ns), 'block_sizes':list(bs),
            'kappa':kappa, 'target_bounds':[low,high], 'alpha':alpha,
            'statistical_unit':'independent complete scalar-target episode',
            'numeric_scope':'guarded float64 dual values and numerical replay, not interval verification'}
