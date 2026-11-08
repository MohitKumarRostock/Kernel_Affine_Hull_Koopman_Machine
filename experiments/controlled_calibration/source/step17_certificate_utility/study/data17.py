"""Independent bounded episodes for the frozen Step 17 mechanism study.

The published smooth-product/hidden-product laws are retained, but every pool
and native geometry seed is new.  Importing this module draws no data.
Only X and the specifically revealed training/validation Y enter learning.
Driver/noise arrays are saved exclusively as generator-audit evidence.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np

REGIMES = ("repairable", "irreducible")
TARGET_BOUNDS = (-.85, .85)
NOISE_HALF_WIDTH = .05
AMPLITUDE = .8
CENTER = .9
JITTER_HALF_WIDTH = .1


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(name + " is below its minimum")
    return value


def _history(rng, n):
    signs = rng.choice(np.asarray([-1., 1.]), size=(n, 2))
    return signs * (CENTER + rng.uniform(-JITTER_HALF_WIDTH, JITTER_HALF_WIDTH,
                                        size=(n, 2)))


def draw_support(n, seed):
    """One independent input per row, with no target/noise computation."""
    n, seed = _integer(n, "support size", 1), _integer(seed, "support seed")
    x = _history(np.random.default_rng(seed), n)
    if x.shape != (n, 2) or not np.isfinite(x).all() or np.max(abs(x)) > 1.:
        raise ArithmeticError("Support draw violates its fixed law")
    return {"X": np.ascontiguousarray(x)}


def draw_episodes(n, seed, regime):
    """Draw n independent one-origin episodes in the documented RNG order.

    1. X signs; X jitters.
    2. In irreducible only, independent hidden-driver signs and jitters.
    3. Independent uniform target noise.
    A sample-size experiment must slice this one maximal draw rather than
    redraw smaller n, since row-major RNG layouts depend on n.
    """
    n, seed = _integer(n, "episode count", 1), _integer(seed, "episode seed")
    if regime not in REGIMES:
        raise ValueError("Unknown Step 17 regime")
    rng = np.random.default_rng(seed)
    x = _history(rng, n)
    driver = x.copy() if regime == "repairable" else _history(rng, n)
    noise = rng.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, size=n)
    y = AMPLITUDE * np.prod(driver, axis=1) + noise
    result = {"X": x, "Y": y, "driver": driver, "noise": noise}
    if (not all(np.isfinite(a).all() for a in result.values())
            or np.min(y) < TARGET_BOUNDS[0] or np.max(y) > TARGET_BOUNDS[1]):
        raise ArithmeticError("Episode draw violates its declared bounded task")
    return {key: np.ascontiguousarray(value) for key, value in result.items()}


def bayes_risk(regime):
    """Evaluation-only population Bayes risk; no selected-model claim."""
    if regime not in REGIMES:
        raise ValueError("Unknown Step 17 regime")
    noise_variance = NOISE_HALF_WIDTH ** 2 / 3.
    if regime == "repairable":
        return noise_variance
    return (AMPLITUDE ** 2 * (CENTER ** 2 + JITTER_HALF_WIDTH ** 2 / 3.) ** 2
            + noise_variance)


def conditional_moments(x, regime):
    """Evaluator-only conditional mean/variance at supplied terminal inputs.

    For irreducible, the zero constant attains the Bayes risk in every
    declared decoder class.  For repairable, the Bayes floor is not asserted
    to equal the risk minimum of a learned KAHM class.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("Inputs must be a finite n-by-two matrix")
    if regime == "repairable":
        return AMPLITUDE * np.prod(x, axis=1), np.full(len(x), bayes_risk(regime))
    if regime == "irreducible":
        return np.zeros(len(x)), np.full(len(x), bayes_risk(regime))
    raise ValueError("Unknown Step 17 regime")
