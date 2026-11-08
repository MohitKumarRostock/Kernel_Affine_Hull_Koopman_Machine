"""Exact dyadic signed-dual verification conditional on inherited feature boxes.

No producer, optimizer, raw-data loader or model-selection module is imported.
Binary64 values are interpreted as their exact dyadic real values. Integer box
distances are verified before use; every objective operation uses Python ints.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
import time

import numpy as np

GRID_POWER = 28
Q = 1 << GRID_POWER
MAX_FEATURE_BLOCK = 8
MAX_SQUARED_DISTANCE = MAX_FEATURE_BLOCK * Q * Q  # 2**59, below signed int64.


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def f64_array(value, ndim, name):
    value = np.asarray(value)
    need(value.dtype == np.dtype("float64") and value.ndim == ndim and value.size > 0,
         name + " must be a nonempty binary64 array with the declared rank")
    need(np.all(np.isfinite(value)), name + " contains nonfinite entries")
    return np.ascontiguousarray(value)


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def feature_slices(blocks, dimension):
    result, covered = [], []
    for item in blocks:
        if isinstance(item, slice):
            need(item.step in (None, 1), "Noncontiguous feature block")
            a, b = 0 if item.start is None else item.start, dimension if item.stop is None else item.stop
        else:
            need(len(item) == 2, "Feature block must have two endpoints")
            a, b = item
        need(isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))
             and not isinstance(a, (bool, np.bool_)) and not isinstance(b, (bool, np.bool_)),
             "Feature block endpoints must be integers")
        a, b = int(a), int(b)
        need(0 <= a < b <= dimension and b-a <= MAX_FEATURE_BLOCK, "Unsupported feature block width")
        result.append((a, b)); covered.extend(range(a, b))
    need(covered == list(range(dimension)), "Feature blocks do not partition coordinates")
    return tuple(result)


def quantize_boxes(lower, upper):
    """Outward Q-grid boxes; check EVERY endpoint with exact integer ratios.

    The fast ldexp/floor/ceil path is exact in this domain. The ratio checks
    make that fact directly auditable without relying on floating rounding.
    """
    started = time.perf_counter()
    lower, upper = f64_array(lower, 2, "box lower"), f64_array(upper, 2, "box upper")
    need(lower.shape == upper.shape and np.all(lower <= upper), "Reversed or mismatched boxes")
    need(np.all(lower >= 0) and np.all(upper <= 1), "Feature boxes must be contained in [0,1]")
    lo = np.floor(np.ldexp(lower, GRID_POWER)).astype(np.int64)
    hi = np.ceil(np.ldexp(upper, GRID_POWER)).astype(np.int64)
    for original, integer in zip(lower.ravel(), lo.ravel(), strict=True):
        numerator, denominator = float(original).as_integer_ratio()
        need(int(integer) == (numerator * Q) // denominator, "Lower quantization differs from exact floor")
    for original, integer in zip(upper.ravel(), hi.ravel(), strict=True):
        numerator, denominator = float(original).as_integer_ratio()
        need(int(integer) == -((-numerator * Q) // denominator), "Upper quantization differs from exact ceiling")
    need(np.all(0 <= lo) and np.all(lo <= hi) and np.all(hi <= Q), "Quantized box range failed")
    metrics = {"grid_power": GRID_POWER, "denominator": Q, "endpoint_count": 2 * lower.size,
               "all_endpoint_integer_ratio_checks_passed": True,
               "lower_endpoint_sha256": array_hash(lower), "upper_endpoint_sha256": array_hash(upper),
               "integer_lower_sha256": array_hash(lo), "integer_upper_sha256": array_hash(hi),
               "lower_changed_count": int(np.count_nonzero(np.ldexp(lo.astype(float), -GRID_POWER) != lower)),
               "upper_changed_count": int(np.count_nonzero(np.ldexp(hi.astype(float), -GRID_POWER) != upper)),
               "max_integer_box_width": int(np.max(hi-lo)),
               "seconds": time.perf_counter()-started}
    return lo, hi, metrics


def ceil_sqrt_integer(values):
    """Fast proposal plus exact int64 repair/invariants, with proved range safety."""
    values = np.asarray(values)
    need(values.dtype == np.dtype("int64") and np.all(values >= 0), "Squared distances must be nonnegative int64")
    need(np.all(values <= MAX_SQUARED_DISTANCE), "Squared distance exceeds proved integer range")
    roots = np.sqrt(values.astype(np.float64)).astype(np.int64)
    need(np.all(roots >= 0) and np.all(roots < 1 << 30), "Square-root proposal outside safe range")
    corrections = 0
    for _ in range(4):
        too_small = roots*roots < values
        too_large = (roots > 0) & ((roots-1)*(roots-1) >= values)
        if not np.any(too_small | too_large):
            break
        corrections += int(np.count_nonzero(too_small | too_large))
        roots[too_small] += 1
        roots[too_large] -= 1
    need(np.all(roots*roots >= values), "Integer ceiling sqrt upper invariant failed")
    need(np.all((roots == 0) == (values == 0)), "Integer ceiling sqrt zero invariant failed")
    need(np.all((roots == 0) | ((roots-1)*(roots-1) < values)), "Integer ceiling sqrt lower invariant failed")
    return roots, {"count": int(values.size), "integer_repair_count": corrections,
                   "all_integer_invariants_verified": True,
                   "max_squared_numerator": int(np.max(values)) if values.size else 0,
                   "max_distance_numerator": int(np.max(roots)) if roots.size else 0}


def edge_upper_numerators(integer_lower, integer_upper, indices, blocks):
    """Dnum/Q for every row-major i<j edge of each supplied disjoint group."""
    started = time.perf_counter()
    lo, hi, indices = map(np.asarray, (integer_lower, integer_upper, indices))
    need(lo.dtype == hi.dtype == np.dtype("int64") and lo.ndim == 2 and lo.shape == hi.shape,
         "Integer boxes have wrong shape or dtype")
    need(np.all(0 <= lo) and np.all(lo <= hi) and np.all(hi <= Q), "Integer box bounds invalid")
    slices = feature_slices(blocks, lo.shape[1])
    need(indices.dtype == np.dtype("int64") and indices.ndim == 2 and 2 <= indices.shape[1] <= 128,
         "Group indices must be int64 with supported group width")
    need(np.all(indices >= 0) and np.all(indices < len(lo)), "Group index outside feature population")
    need(np.unique(indices).size == indices.size, "Groups repeat a patient row")
    b = indices.shape[1]
    ii, jj = np.triu_indices(b, 1)
    L, U = lo[indices], hi[indices]
    squared = np.zeros((len(indices), len(ii)), dtype=np.int64)
    for a, z in slices:
        # Signed differences BEFORE squaring; uint subtraction would wrap.
        span = np.maximum(U[:, ii, a:z]-L[:, jj, a:z], U[:, jj, a:z]-L[:, ii, a:z])
        need(np.all(span >= 0) and np.all(span <= Q), "Coordinate separation violates proved range")
        block_squared = np.sum(span*span, axis=-1, dtype=np.int64)
        need(np.all(block_squared >= 0) and np.all(block_squared <= (z-a)*Q*Q), "Integer distance overflow")
        squared = np.maximum(squared, block_squared)
    distance, metrics = ceil_sqrt_integer(squared)
    metrics.update(blocks=len(indices), block_size=b, edges_per_block=len(ii),
                   seconds=time.perf_counter()-started)
    return distance, metrics


def _common_dyadic(values):
    ratios = [float(value).as_integer_ratio() for value in values]
    powers = [denominator.bit_length()-1 for _, denominator in ratios]
    power = max(powers, default=0)
    integers = [numerator << (power-p) for (numerator, denominator), p in zip(ratios, powers, strict=True)]
    return integers, power


def _fraction_power(integer, power):
    return Fraction(integer, 1 << power)


def exact_dual(alpha, targets, distance_numerators, kappa=1.0):
    """Evaluate the complete signed-edge dual exactly with Python integers.

    Return the raw Fraction (possibly negative) and exact component metrics.
    Zero is a separate admissible witness; callers must explicitly select it.
    """
    started = time.perf_counter()
    alpha, targets = f64_array(alpha, 1, "signed dual"), f64_array(targets, 1, "targets")
    d = np.asarray(distance_numerators)
    b = len(targets); ii, jj = np.triu_indices(b, 1)
    need(alpha.shape == d.shape == (len(ii),) and d.dtype == np.dtype("int64"), "Edge layout or integer distance dtype differs")
    need(np.all(d >= 0) and np.all(d < 1 << 30), "Distance numerator outside proved range")
    need(np.all(np.abs(targets) <= 1), "Targets exceed frozen bounded range")
    need(isinstance(kappa, (float, np.floating)) and math.isfinite(float(kappa)) and float(kappa) >= 0,
         "Kappa must be a finite nonnegative binary64 value")
    kn, kd = float(kappa).as_integer_ratio(); kp = kd.bit_length()-1
    active = np.flatnonzero(alpha != 0)
    A, ap = _common_dyadic(alpha[active])
    Y, yp = _common_dyadic(targets)
    divergence = [0] * b
    linear = penalty = l1 = 0
    for edge, multiplier in zip(active.tolist(), A, strict=True):
        i, j = int(ii[edge]), int(jj[edge])
        divergence[i] += multiplier; divergence[j] -= multiplier
        linear += multiplier * (Y[i]-Y[j])
        absolute = abs(multiplier)
        penalty += absolute * int(d[edge]); l1 += absolute
    quadratic = sum(value*value for value in divergence)
    exponents = (ap+yp, 2*ap+2, ap+GRID_POWER+kp)
    common = max(exponents)
    numerator = ((linear << (common-exponents[0]))
                 -(quadratic << (common-exponents[1]))
                 -((kn*penalty) << (common-exponents[2])))
    value = _fraction_power(numerator, common)
    components = {"linear": _fraction_power(linear, exponents[0]),
                  "quadratic": _fraction_power(quadratic, exponents[1]),
                  "distance_penalty": _fraction_power(kn*penalty, exponents[2]),
                  "alpha_l1": _fraction_power(l1, ap),
                  "extra_quantization_sse_loss_bound": _fraction_power(7*kn*l1, ap+GRID_POWER+kp),
                  "active_edges": int(len(active)), "edges": int(len(alpha)),
                  "alpha_common_denominator_power": ap, "target_common_denominator_power": yp,
                  "objective_common_denominator_power": common,
                  "max_divergence_integer_bits": max((abs(value).bit_length() for value in divergence), default=0),
                  "signed_dual_sha256": array_hash(alpha),
                  "seconds": time.perf_counter()-started}
    need(value == components["linear"]-components["quadratic"]-components["distance_penalty"],
         "Internal exact objective composition differs")
    return value, components


def exact_mse(targets, predictions):
    targets, predictions = f64_array(targets, 1, "MSE targets"), f64_array(predictions, 1, "MSE predictions")
    need(targets.shape == predictions.shape and np.all(np.abs(targets) <= 1) and np.all(np.abs(predictions) <= 1),
         "MSE arrays differ or exceed frozen bounds")
    integers, power = _common_dyadic(np.concatenate((targets, predictions)))
    n = len(targets)
    square_sum = sum((integers[i]-integers[n+i])**2 for i in range(n))
    return Fraction(square_sum, n * (1 << (2*power)))


def rational_record(value):
    value = Fraction(value)
    try:
        nominal = float(value)
    except OverflowError:
        nominal = math.inf if value > 0 else -math.inf
    if math.isfinite(nominal):
        represented = Fraction.from_float(nominal)
        lower = nominal if represented <= value else math.nextafter(nominal, -math.inf)
        upper = nominal if represented >= value else math.nextafter(nominal, math.inf)
        need(Fraction.from_float(lower) <= value <= Fraction.from_float(upper), "Outward rational display failed")
    else:
        lower = upper = None
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "float_lower": lower, "float_upper": upper,
            "float_nearest": nominal if math.isfinite(nominal) else None}


def from_record(record):
    value = Fraction(int(record["numerator"]), int(record["denominator"]))
    need(str(value.numerator) == record["numerator"] and str(value.denominator) == record["denominator"],
         "Rational record is not reduced/canonical")
    expected = rational_record(value)
    need(record == expected, "Rational display or exact numerator/denominator was corrupted")
    return value


def embed_children(left, right):
    """Zero-extend two same-size child witnesses into concatenated parent rows."""
    left, right = f64_array(left, 1, "left child dual"), f64_array(right, 1, "right child dual")
    need(left.shape == right.shape, "Child witness sizes differ")
    discriminant = 1 + 8 * len(left)
    root = math.isqrt(discriminant)
    need(root*root == discriminant and (1+root) % 2 == 0, "Invalid triangular child edge count")
    b = (1+root)//2
    need(b in (2, 4, 8, 16, 32, 64), "Unsupported child size")
    ii, jj = np.triu_indices(2*b, 1)
    result = np.zeros(len(ii), dtype=np.float64)
    first, second = jj < b, ii >= b
    need(np.count_nonzero(first) == len(left) and np.count_nonzero(second) == len(right), "Embedding edge census")
    result[first], result[second] = left, right
    need(np.array_equal(result[first], left) and np.array_equal(result[second], right), "Embedding changed child bits")
    return result


def exact_best(new_alpha, new_value, *, left_alpha=None, right_alpha=None,
               left_value=None, right_value=None):
    """Choose among zero, new, exact children; deterministic ties: new, children, zero."""
    new_alpha = f64_array(new_alpha, 1, "new witness")
    new_value = Fraction(new_value)
    candidates = [(new_value, 2, "new", new_alpha)]
    if left_alpha is not None:
        need(right_alpha is not None and left_value is not None and right_value is not None,
             "Incomplete exact child witness")
        left_value, right_value = Fraction(left_value), Fraction(right_value)
        need(left_value >= 0 and right_value >= 0, "Exact children must include the zero option")
        embedded = embed_children(left_alpha, right_alpha)
        need(embedded.shape == new_alpha.shape, "Parent and embedded witness shapes differ")
        candidates.append((left_value+right_value, 1, "exact_children", embedded))
    else:
        need(right_alpha is None and left_value is None and right_value is None, "Unexpected partial child information")
    candidates.append((Fraction(0), 0, "zero", np.zeros_like(new_alpha)))
    value, _, source, chosen = max(candidates, key=lambda item: (item[0], item[1]))
    # A zero objective always uses the zero witness; this guarantees zeroing of
    # negative children and makes the exact chosen array uniquely recoverable.
    if value == 0:
        source, chosen = "zero", np.zeros_like(new_alpha)
    return chosen.copy(), value, source
