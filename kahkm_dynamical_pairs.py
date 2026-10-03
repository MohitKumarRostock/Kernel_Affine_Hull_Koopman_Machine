"""Independent current/successor pairs for dynamical certificate studies.

This module implements the evaluation law frozen in

    reproduction/certificates/DYNAMICAL_PROTOCOL.md

for the Duffing and Van der Pol benchmarks.

Each retained observation is generated from its own independently seeded
trajectory. A separate independently seeded RNG chooses the current-state
time index T. Consequently, transitions taken from a common trajectory are
never counted as separate i.i.d. certificate observations.

The matched-stochastic dynamics intentionally reproduce the historical
benchmark conventions:

Duffing
-------
    z0 = [0.7, 0.0] + 0.05 N(0, I)
    dt = supplied by caller (benchmark value 0.03)
    RK4
    process noise after each step: 1e-4 N(0, I)

Van der Pol
-----------
    z0 = [2.0, 0.0] + 0.05 N(0, I)
    dt = supplied by caller (benchmark value 0.02)
    mu = supplied by caller (benchmark value 1.0)
    RK4
    process noise after each step: 1e-5 N(0, I)

The deterministic mode preserves the same initial-condition distribution but
sets per-step process noise identically to zero.

This module does not fit a representation and does not compute a certificate.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


SystemName = Literal["duffing", "vanderpol"]
EvaluationMode = Literal["matched_stochastic", "deterministic"]
FloatArray = NDArray[np.float64]

PAIR_LAW_ID = "independent_uniform_time_pair_v1"

VALID_SYSTEMS: tuple[str, ...] = (
    "duffing",
    "vanderpol",
)

VALID_MODES: tuple[str, ...] = (
    "matched_stochastic",
    "deterministic",
)


@dataclass(frozen=True)
class DynamicalPair:
    """One independently generated current/successor evaluation pair."""

    pair_law_id: str
    system: str
    evaluation_mode: str
    trajectory_seed: int
    time_seed: int
    time_index: int
    horizon_steps: int
    dt: float
    vanderpol_mu: float
    current_state: tuple[float, float]
    successor_state: tuple[float, float]


def _validated_seed(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    seed = int(value)

    if seed < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )

    return seed


def _validated_positive_int(
    name: str,
    value: int,
) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    result = int(value)

    if result <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return result


def _validated_positive_float(
    name: str,
    value: float,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            f"{name} must be a real scalar."
        )

    result = float(value)

    if (
        not np.isfinite(result)
        or result <= 0.0
    ):
        raise ValueError(
            f"{name} must be finite and positive."
        )

    return result


def _validated_system(
    system: str,
) -> SystemName:
    if not isinstance(system, str):
        raise TypeError(
            "system must be a string."
        )

    if system not in VALID_SYSTEMS:
        raise ValueError(
            f"Unsupported system: {system!r}."
        )

    return system  # type: ignore[return-value]


def _validated_mode(
    evaluation_mode: str,
) -> EvaluationMode:
    if not isinstance(
        evaluation_mode,
        str,
    ):
        raise TypeError(
            "evaluation_mode must be a string."
        )

    if evaluation_mode not in VALID_MODES:
        raise ValueError(
            "Unsupported evaluation_mode: "
            f"{evaluation_mode!r}."
        )

    return evaluation_mode  # type: ignore[return-value]


def _duffing_vector_field(
    z: FloatArray,
) -> FloatArray:
    delta = 0.25
    alpha = -1.0
    beta = 1.0

    q = float(z[0])
    p = float(z[1])

    return np.array(
        [
            p,
            -delta * p
            - alpha * q
            - beta * q**3,
        ],
        dtype=np.float64,
    )


def _vanderpol_vector_field(
    z: FloatArray,
    *,
    mu: float,
) -> FloatArray:
    q = float(z[0])
    p = float(z[1])

    return np.array(
        [
            p,
            float(mu)
            * (1.0 - q * q)
            * p
            - q,
        ],
        dtype=np.float64,
    )


def _rk4_step(
    *,
    system: SystemName,
    z: FloatArray,
    dt: float,
    vanderpol_mu: float,
) -> FloatArray:
    if system == "duffing":

        def field(
            state: FloatArray,
        ) -> FloatArray:
            return _duffing_vector_field(
                state
            )

    elif system == "vanderpol":

        def field(
            state: FloatArray,
        ) -> FloatArray:
            return _vanderpol_vector_field(
                state,
                mu=vanderpol_mu,
            )

    else:
        raise RuntimeError(
            f"Internal unsupported system: {system!r}."
        )

    k1 = field(z)
    k2 = field(
        z + 0.5 * dt * k1
    )
    k3 = field(
        z + 0.5 * dt * k2
    )
    k4 = field(
        z + dt * k3
    )

    return np.asarray(
        z
        + (dt / 6.0)
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        ),
        dtype=np.float64,
    )


def _initial_state(
    *,
    system: SystemName,
    rng: np.random.Generator,
) -> FloatArray:
    if system == "duffing":
        center = np.array(
            [0.7, 0.0],
            dtype=np.float64,
        )
    elif system == "vanderpol":
        center = np.array(
            [2.0, 0.0],
            dtype=np.float64,
        )
    else:
        raise RuntimeError(
            f"Internal unsupported system: {system!r}."
        )

    return np.asarray(
        center
        + 0.05
        * rng.normal(size=2),
        dtype=np.float64,
    )


def _process_noise_std(
    *,
    system: SystemName,
    evaluation_mode: EvaluationMode,
) -> float:
    if evaluation_mode == "deterministic":
        return 0.0

    if system == "duffing":
        return 1e-4

    if system == "vanderpol":
        return 1e-5

    raise RuntimeError(
        f"Internal unsupported system: {system!r}."
    )


def _advance_one_step(
    *,
    system: SystemName,
    z: FloatArray,
    dt: float,
    vanderpol_mu: float,
    evaluation_mode: EvaluationMode,
    rng: np.random.Generator,
) -> FloatArray:
    successor = _rk4_step(
        system=system,
        z=z,
        dt=dt,
        vanderpol_mu=vanderpol_mu,
    )

    sigma = _process_noise_std(
        system=system,
        evaluation_mode=evaluation_mode,
    )

    if sigma != 0.0:
        successor = np.asarray(
            successor
            + sigma
            * rng.normal(size=2),
            dtype=np.float64,
        )

    return successor


def sample_independent_pair(
    *,
    system: str,
    evaluation_mode: str,
    horizon_steps: int,
    dt: float,
    trajectory_seed: int,
    time_seed: int,
    vanderpol_mu: float = 1.0,
) -> DynamicalPair:
    """Generate one independent current/successor pair.

    T is uniform on {0, ..., horizon_steps - 1}. The trajectory RNG and the
    time-index RNG are separate. For matched-stochastic mode, the trajectory
    prefix for a fixed trajectory_seed therefore agrees with the historical
    benchmark generator using that same integer seed.

    The returned current state is z_T and the successor is z_{T+1}.
    """
    system_value = _validated_system(
        system
    )

    mode_value = _validated_mode(
        evaluation_mode
    )

    horizon_value = _validated_positive_int(
        "horizon_steps",
        horizon_steps,
    )

    dt_value = _validated_positive_float(
        "dt",
        dt,
    )

    trajectory_seed_value = _validated_seed(
        "trajectory_seed",
        trajectory_seed,
    )

    time_seed_value = _validated_seed(
        "time_seed",
        time_seed,
    )

    mu_value = _validated_positive_float(
        "vanderpol_mu",
        vanderpol_mu,
    )

    time_rng = np.random.default_rng(
        time_seed_value
    )

    time_index = int(
        time_rng.integers(
            0,
            horizon_value,
        )
    )

    trajectory_rng = np.random.default_rng(
        trajectory_seed_value
    )

    z = _initial_state(
        system=system_value,
        rng=trajectory_rng,
    )

    current: FloatArray | None = None

    for step in range(
        time_index + 1
    ):
        if step == time_index:
            current = z.copy()

        z = _advance_one_step(
            system=system_value,
            z=z,
            dt=dt_value,
            vanderpol_mu=mu_value,
            evaluation_mode=mode_value,
            rng=trajectory_rng,
        )

    if current is None:
        raise RuntimeError(
            "Internal error: current state was not retained."
        )

    if (
        current.shape != (2,)
        or z.shape != (2,)
        or not np.all(
            np.isfinite(current)
        )
        or not np.all(
            np.isfinite(z)
        )
    ):
        raise RuntimeError(
            "Dynamical pair generation produced invalid state data."
        )

    return DynamicalPair(
        pair_law_id=PAIR_LAW_ID,
        system=system_value,
        evaluation_mode=mode_value,
        trajectory_seed=trajectory_seed_value,
        time_seed=time_seed_value,
        time_index=time_index,
        horizon_steps=horizon_value,
        dt=dt_value,
        vanderpol_mu=mu_value,
        current_state=(
            float(current[0]),
            float(current[1]),
        ),
        successor_state=(
            float(z[0]),
            float(z[1]),
        ),
    )
