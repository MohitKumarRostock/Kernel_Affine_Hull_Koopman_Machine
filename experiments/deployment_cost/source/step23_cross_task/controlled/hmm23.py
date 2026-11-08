"""Fixed finite-history HMM benchmark and same-information exact oracle.

Step 23 specifies this benchmark; it does not generate its benchmark data.
The separate check_controlled23.py uses exhaustive short manufactured
histories and scripted integer draws only. No model is fitted here.

The mathematical HMM has rational transition/emission probabilities. Its
oracle consumes only the supplied observed history, starting from the
stationary prior immediately before the first observation. Known model
parameters are oracle knowledge; hidden states and prehistory are not.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from numbers import Integral
from typing import Protocol, Sequence

import numpy as np


BASE_SEED = 2026110623
SEED_NAMESPACE = 2301
REPLICATIONS = 5
HISTORY_LENGTH = 32
HORIZONS = (1, 4, 16)
PRIMARY_HORIZON = 4
ROLE_COUNTS = {"support": 512, "train": 3072, "validation": 1536, "test": 8192}
ROLE_CODES = {"support": 0, "train": 1, "validation": 2, "test": 3}
STATE_NAMES = ("negative_large", "negative_small", "positive_large", "positive_small")
TRANSITION_DENOMINATOR = 512
TRANSITION_NUMERATORS = (
    (465, 31, 15, 1),
    (31, 465, 1, 15),
    (15, 1, 465, 31),
    (1, 15, 31, 465),
)
EMISSION_DENOMINATOR = 32
EMISSION_NUMERATORS = (
    (24, 6, 1, 1),
    (5, 18, 8, 1),
    (1, 1, 6, 24),
    (1, 8, 18, 5),
)
SYMBOL_NUMERATORS = (-2, -1, 1, 2)
SYMBOL_DENOMINATOR = 2
SYMBOL_VALUES = (-1.0, -0.5, 0.5, 1.0)


class IntegerSource(Protocol):
    def integers(self, low: int, high: int) -> int: ...


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _symbols(history: Sequence[int]) -> tuple[int, ...]:
    result = tuple(history)
    if not result:
        raise ValueError("A nonempty observed history is required")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= int(v) < 4
           for v in result):
        raise ValueError("History symbols must be integer indices in {0,1,2,3}")
    return tuple(int(v) for v in result)


def symbols_from_values(history: Sequence[float]) -> tuple[int, ...]:
    """Decode the exact unscaled observed values, without approximate matching."""
    lookup = {value: i for i, value in enumerate(SYMBOL_VALUES)}
    result = []
    for value in history:
        if isinstance(value, (bool, np.bool_)) or float(value) not in lookup:
            raise ValueError("Oracle inputs must be the raw four-symbol observations")
        result.append(lookup[float(value)])
    return _symbols(result)


def _matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Small arbitrary-precision integer matrix product; no fixed-width overflow."""
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(len(b)))
                       for j in range(len(b[0]))) for i in range(len(a)))


@lru_cache(maxsize=None)
def transition_numerator_power(horizon: int) -> tuple[tuple[int, ...], ...]:
    horizon = _positive_integer(horizon, "horizon")
    result = tuple(tuple(int(i == j) for j in range(4)) for i in range(4))
    base = TRANSITION_NUMERATORS
    remaining = horizon
    while remaining:
        if remaining & 1:
            result = _matmul(result, base)
        remaining >>= 1
        if remaining:
            base = _matmul(base, base)
    return result


@lru_cache(maxsize=None)
def future_symbol_numerators(horizon: int) -> tuple[tuple[int, ...], ...]:
    return _matmul(transition_numerator_power(horizon), EMISSION_NUMERATORS)


def forward_integer_weights(history: Sequence[int]) -> tuple[int, ...]:
    """Exact unnormalized posterior weights for the supplied finite history.

    The uniform prior's common 1/4 factor and every common transition/emission
    denominator cancel at normalization. The first observation is emitted
    before the first transition. No earlier observation is conditioned on.
    """
    observations = _symbols(history)
    weights = tuple(EMISSION_NUMERATORS[i][observations[0]] for i in range(4))
    for observation in observations[1:]:
        weights = tuple(sum(weights[i] * TRANSITION_NUMERATORS[i][j] for i in range(4))
                        * EMISSION_NUMERATORS[j][observation] for j in range(4))
    return weights


@dataclass(frozen=True)
class OracleResult:
    history_length: int
    horizon: int
    belief: tuple[Fraction, ...]
    predictive_symbol_probabilities: tuple[Fraction, ...]
    mean: Fraction
    second_moment: Fraction
    variance: Fraction


def oracle_from_symbols(history: Sequence[int], horizon: int) -> OracleResult:
    """Same-history Bayes predictor for squared loss, using exact rationals."""
    observations = _symbols(history)
    horizon = _positive_integer(horizon, "horizon")
    weights = forward_integer_weights(observations)
    normalizer = sum(weights)
    future = future_symbol_numerators(horizon)
    denominator = normalizer * TRANSITION_DENOMINATOR ** horizon * EMISSION_DENOMINATOR
    numerators = tuple(sum(weights[i] * future[i][j] for i in range(4)) for j in range(4))
    predictive = tuple(Fraction(n, denominator) for n in numerators)
    mean = Fraction(sum(n * y for n, y in zip(numerators, SYMBOL_NUMERATORS)),
                    denominator * SYMBOL_DENOMINATOR)
    second = Fraction(sum(n * y * y for n, y in zip(numerators, SYMBOL_NUMERATORS)),
                      denominator * SYMBOL_DENOMINATOR ** 2)
    return OracleResult(len(observations), horizon,
                        tuple(Fraction(w, normalizer) for w in weights),
                        predictive, mean, second, second - mean * mean)


def binary64_bracket(value: Fraction) -> dict[str, float]:
    """Nearest binary64 representative and proven outward adjacent endpoints."""
    nearest = float(value)
    if not math.isfinite(nearest):
        raise ValueError("The exact value must be finite in binary64 range")
    nearest_exact = Fraction.from_float(nearest)
    lower = nearest if nearest_exact <= value else math.nextafter(nearest, -math.inf)
    upper = nearest if nearest_exact >= value else math.nextafter(nearest, math.inf)
    return {"nominal": nearest, "lower": lower, "upper": upper}


def exact_conditional_regret(prediction: float, oracle: OracleResult) -> Fraction:
    """Conditional excess risk of one clipped binary64 forecast over the oracle."""
    prediction = float(prediction)
    if not math.isfinite(prediction) or not -1.0 <= prediction <= 1.0:
        raise ValueError("Forecasts must be finite and clipped to [-1,1]")
    return (Fraction.from_float(prediction) - oracle.mean) ** 2


def _categorical(numerators: Sequence[int], denominator: int, rng: IntegerSource) -> int:
    draw = int(rng.integers(0, denominator))
    if not 0 <= draw < denominator:
        raise ValueError("Integer source produced a draw outside its requested range")
    cumulative = 0
    for index, numerator in enumerate(numerators):
        cumulative += numerator
        if draw < cumulative:
            return index
    raise ValueError("Categorical probabilities do not sum to their denominator")


def simulate_one(rng: IntegerSource, *, history_length: int = HISTORY_LENGTH,
                 horizons: Sequence[int] = HORIZONS) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Draw one trajectory and return observed symbols only.

    This function is specified for later execution. The Step23 manufactured
    checks call it only with scripted endpoint integers, never benchmark RNGs.
    An empty horizon tuple draws only the support history and no future target.
    """
    history_length = _positive_integer(history_length, "history_length")
    horizons = tuple(_positive_integer(h, "horizon") for h in horizons)
    if len(set(horizons)) != len(horizons):
        raise ValueError("Horizons must be distinct")
    length = history_length + (max(horizons) if horizons else 0)
    state = int(rng.integers(0, 4))
    if not 0 <= state < 4:
        raise ValueError("Initial-state draw outside {0,1,2,3}")
    observations = []
    for time in range(length):
        if time:
            state = _categorical(TRANSITION_NUMERATORS[state], TRANSITION_DENOMINATOR, rng)
        observations.append(_categorical(EMISSION_NUMERATORS[state], EMISSION_DENOMINATOR, rng))
    history = tuple(observations[:history_length])
    targets = tuple(observations[history_length - 1 + h] for h in horizons)
    return history, targets


def seed_entropy(replication: int, role: str) -> tuple[int, ...]:
    if isinstance(replication, bool) or not isinstance(replication, Integral) or not 0 <= replication < REPLICATIONS:
        raise ValueError("Replication index must be in {0,1,2,3,4}")
    if role not in ROLE_CODES:
        raise ValueError("Unknown benchmark split role")
    # Varying identifiers precede the fixed root seed; no seed arithmetic.
    return (SEED_NAMESPACE, int(replication), ROLE_CODES[role], BASE_SEED)


def generate_role(replication: int, role: str) -> dict[str, object]:
    """Later execution entry point; never called by Step23 preflight.

    Returns one example per independently initiated trajectory, with no latent
    states, posterior beliefs, oracle means, or prehistory in learned inputs.
    The fixed role count cannot be overridden. This routine writes no files.
    """
    entropy = seed_entropy(replication, role)
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))
    n = ROLE_COUNTS[role]
    horizons = () if role == "support" else HORIZONS
    x_symbols = np.empty((n, HISTORY_LENGTH), dtype=np.uint8)
    y_symbols = None if role == "support" else np.empty((n, len(horizons)), dtype=np.uint8)
    for row in range(n):
        history, targets = simulate_one(rng, horizons=horizons)
        x_symbols[row] = history
        if y_symbols is not None:
            y_symbols[row] = targets
    values = np.asarray(SYMBOL_VALUES, dtype=np.float64)
    return {"X": values[x_symbols],
            "Y": None if y_symbols is None else values[y_symbols],
            "metadata": {"schema": "step23-hmm-role-v1", "replication": int(replication),
                         "role": role, "rows": n, "history_length": HISTORY_LENGTH,
                         "horizons": list(horizons), "seed_entropy": list(entropy),
                         "bit_generator": "PCG64DXSM", "numpy_version": np.__version__,
                         "one_example_per_independent_trajectory": True,
                         "hidden_states_or_oracles_in_inputs": False,
                         "support_future_targets_generated": False}}


def normalization_fixture() -> tuple[dict[str, object], ...]:
    """Four exact rational score states for an algebraic normalization control.

    These are engineered score states, not scores learned by a KAHM. Keeping
    exact rational c avoids interpreting near-collisions after binary64 export
    as exact information loss. Omega is exactly 1 in this separate fixture.
    """
    tau = Fraction.from_float(1e-6)
    rows = []
    for scale in (Fraction(1, 4), Fraction(1, 2)):
        for direction in (Fraction(1, 4), Fraction(3, 4)):
            weights = (scale * direction, scale * (1 - direction))
            scores = tuple(1 + tau - u for u in weights)
            probabilities = tuple(u / sum(weights) for u in weights)
            rows.append({"probability": Fraction(1, 4), "scale": scale,
                         "direction": direction, "weights": weights, "scores": scores,
                         "normalized_state": probabilities,
                         "signal_target": 8 * scale - 3,
                         "nuisance_target": 4 * direction - 2})
    return tuple(rows)


if __name__ == "__main__":
    raise SystemExit("This module defines the future benchmark. Run check_controlled23.py for manufactured checks only.")
