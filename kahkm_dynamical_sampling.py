"""Deterministic independent-pair sampling for the dynamical certificate pilot.

This module implements the seed scheme frozen in

    reproduction/certificates/configs/dynamical_pilot_v1.json

Each retained pair has two independently derived RNG streams:

- trajectory_seed
- time_seed

Both are derived from the full immutable seed-material tuple

    (
        campaign_key,
        system_key,
        mode_key,
        sample_size,
        replicate_index,
        pair_index,
        stream_key,
        root_seed,
    )

using NumPy SeedSequence. The resulting 64-bit integer is then supplied to
``sample_independent_pair``.

No global RNG state is read or modified. No retry, replacement, or adaptive
seed selection is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from kahkm_dynamical_pairs import (
    PAIR_LAW_ID,
    sample_independent_pair,
)


DYNAMICAL_SAMPLING_ID = "dynamical_certificate_sampling_v1"

BIT_GENERATOR_ID = "PCG64"
SEED_SEQUENCE_ID = "numpy.random.SeedSequence"

ROOT_SEED = 20261003
CAMPAIGN_KEY = 3

SYSTEM_KEYS = {
    "duffing": 1,
    "vanderpol": 2,
}

MODE_KEYS = {
    "matched_stochastic": 1,
    "deterministic": 2,
}

STREAM_KEYS = {
    "trajectory_seed": 1,
    "time_seed": 2,
}

TRAINING_LITERAL_SEEDS = (
    0,
    1,
    2,
)

SystemName = Literal[
    "duffing",
    "vanderpol",
]

EvaluationMode = Literal[
    "matched_stochastic",
    "deterministic",
]

StreamName = Literal[
    "trajectory_seed",
    "time_seed",
]


@dataclass(frozen=True)
class SeedRecord:
    """Complete deterministic seed provenance for one RNG stream."""

    sampling_id: str
    bit_generator: str
    seed_sequence: str
    campaign_key: int
    system: str
    system_key: int
    evaluation_mode: str
    mode_key: int
    sample_size: int
    replicate_index: int
    pair_index: int
    stream_name: str
    stream_key: int
    root_seed: int
    seed_material: tuple[int, ...]
    generated_uint32_words: tuple[int, int]
    generated_seed: int


@dataclass(frozen=True)
class SampledDynamicalDataset:
    """One fixed-size independently generated dynamical evaluation dataset."""

    sampling_id: str
    pair_law_id: str
    system: str
    evaluation_mode: str
    sample_size: int
    replicate_index: int
    dt: float
    horizon_steps: int
    vanderpol_mu: float
    current_states: NDArray[np.float64]
    successor_states: NDArray[np.float64]
    time_indices: tuple[int, ...]
    trajectory_seed_records: tuple[SeedRecord, ...]
    time_seed_records: tuple[SeedRecord, ...]


def _plain_nonnegative_int(
    name: str,
    value: Integral,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            Integral,
        )
    ):
        raise TypeError(
            f"{name} must be an integer, not boolean."
        )

    result = int(
        value
    )

    if result < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )

    return result


def _positive_int(
    name: str,
    value: Integral,
) -> int:
    result = _plain_nonnegative_int(
        name,
        value,
    )

    if result == 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return result


def _positive_float(
    name: str,
    value: float,
) -> float:
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be a real scalar."
        )

    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(
            f"{name} must be a real scalar."
        ) from exc

    if (
        not np.isfinite(
            result
        )
        or result <= 0.0
    ):
        raise ValueError(
            f"{name} must be finite and positive."
        )

    return result


def _validated_system(
    system: str,
) -> SystemName:
    if not isinstance(
        system,
        str,
    ):
        raise TypeError(
            "system must be a string."
        )

    if system not in SYSTEM_KEYS:
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

    if evaluation_mode not in MODE_KEYS:
        raise ValueError(
            f"Unsupported evaluation_mode: {evaluation_mode!r}."
        )

    return evaluation_mode  # type: ignore[return-value]


def _validated_stream(
    stream_name: str,
) -> StreamName:
    if not isinstance(
        stream_name,
        str,
    ):
        raise TypeError(
            "stream_name must be a string."
        )

    if stream_name not in STREAM_KEYS:
        raise ValueError(
            f"Unsupported stream_name: {stream_name!r}."
        )

    return stream_name  # type: ignore[return-value]


def seed_record(
    *,
    system: str,
    evaluation_mode: str,
    sample_size: Integral,
    replicate_index: Integral,
    pair_index: Integral,
    stream_name: str,
) -> SeedRecord:
    """Derive one deterministic 64-bit stream seed from frozen seed material."""
    system_value = _validated_system(
        system
    )

    mode_value = _validated_mode(
        evaluation_mode
    )

    stream_value = _validated_stream(
        stream_name
    )

    sample_size_value = _positive_int(
        "sample_size",
        sample_size,
    )

    replicate_value = _plain_nonnegative_int(
        "replicate_index",
        replicate_index,
    )

    pair_value = _plain_nonnegative_int(
        "pair_index",
        pair_index,
    )

    if pair_value >= sample_size_value:
        raise ValueError(
            "pair_index must lie in [0, sample_size-1]."
        )

    material = (
        CAMPAIGN_KEY,
        SYSTEM_KEYS[
            system_value
        ],
        MODE_KEYS[
            mode_value
        ],
        sample_size_value,
        replicate_value,
        pair_value,
        STREAM_KEYS[
            stream_value
        ],
        ROOT_SEED,
    )

    sequence = np.random.SeedSequence(
        entropy=material
    )

    words = sequence.generate_state(
        2,
        dtype=np.uint32,
    )

    low = int(
        words[0]
    )

    high = int(
        words[1]
    )

    generated_seed = (
        low
        | (
            high
            << 32
        )
    )

    # The full seed material is already disjoint from the literal training
    # seeds, but also reject the practically impossible accidental integer
    # collision explicitly rather than silently remapping it.
    if generated_seed in TRAINING_LITERAL_SEEDS:
        raise RuntimeError(
            "Derived evaluation seed collided with a literal representation "
            "training seed; frozen seed material must be revised before "
            "execution rather than remapped adaptively."
        )

    return SeedRecord(
        sampling_id=
            DYNAMICAL_SAMPLING_ID,
        bit_generator=
            BIT_GENERATOR_ID,
        seed_sequence=
            SEED_SEQUENCE_ID,
        campaign_key=
            CAMPAIGN_KEY,
        system=
            system_value,
        system_key=
            SYSTEM_KEYS[
                system_value
            ],
        evaluation_mode=
            mode_value,
        mode_key=
            MODE_KEYS[
                mode_value
            ],
        sample_size=
            sample_size_value,
        replicate_index=
            replicate_value,
        pair_index=
            pair_value,
        stream_name=
            stream_value,
        stream_key=
            STREAM_KEYS[
                stream_value
            ],
        root_seed=
            ROOT_SEED,
        seed_material=
            tuple(
                int(value)
                for value
                in material
            ),
        generated_uint32_words=(
            low,
            high,
        ),
        generated_seed=
            generated_seed,
    )


def sample_dynamical_dataset(
    *,
    system: str,
    evaluation_mode: str,
    sample_size: Integral,
    replicate_index: Integral,
    horizon_steps: Integral,
    dt: float,
    vanderpol_mu: float = 1.0,
) -> SampledDynamicalDataset:
    """Generate one independent dataset with no retry or replacement seeds."""
    system_value = _validated_system(
        system
    )

    mode_value = _validated_mode(
        evaluation_mode
    )

    sample_size_value = _positive_int(
        "sample_size",
        sample_size,
    )

    replicate_value = _plain_nonnegative_int(
        "replicate_index",
        replicate_index,
    )

    horizon_value = _positive_int(
        "horizon_steps",
        horizon_steps,
    )

    dt_value = _positive_float(
        "dt",
        dt,
    )

    mu_value = _positive_float(
        "vanderpol_mu",
        vanderpol_mu,
    )

    current = np.empty(
        (
            2,
            sample_size_value,
        ),
        dtype=np.float64,
    )

    successor = np.empty_like(
        current
    )

    time_indices: list[int] = []
    trajectory_records: list[
        SeedRecord
    ] = []
    time_records: list[
        SeedRecord
    ] = []

    for pair_index in range(
        sample_size_value
    ):
        trajectory_record = seed_record(
            system=
                system_value,
            evaluation_mode=
                mode_value,
            sample_size=
                sample_size_value,
            replicate_index=
                replicate_value,
            pair_index=
                pair_index,
            stream_name=
                "trajectory_seed",
        )

        time_record = seed_record(
            system=
                system_value,
            evaluation_mode=
                mode_value,
            sample_size=
                sample_size_value,
            replicate_index=
                replicate_value,
            pair_index=
                pair_index,
            stream_name=
                "time_seed",
        )

        pair = sample_independent_pair(
            system=
                system_value,
            evaluation_mode=
                mode_value,
            horizon_steps=
                horizon_value,
            dt=
                dt_value,
            trajectory_seed=
                trajectory_record.generated_seed,
            time_seed=
                time_record.generated_seed,
            vanderpol_mu=
                mu_value,
        )

        if pair.pair_law_id != PAIR_LAW_ID:
            raise RuntimeError(
                "Independent-pair implementation identifier mismatch."
            )

        if (
            pair.trajectory_seed
            != trajectory_record.generated_seed
            or pair.time_seed
            != time_record.generated_seed
        ):
            raise RuntimeError(
                "Independent-pair sampler did not preserve derived seeds."
            )

        current[
            :,
            pair_index,
        ] = np.asarray(
            pair.current_state,
            dtype=np.float64,
        )

        successor[
            :,
            pair_index,
        ] = np.asarray(
            pair.successor_state,
            dtype=np.float64,
        )

        time_indices.append(
            int(
                pair.time_index
            )
        )

        trajectory_records.append(
            trajectory_record
        )

        time_records.append(
            time_record
        )

    if (
        not np.all(
            np.isfinite(
                current
            )
        )
        or not np.all(
            np.isfinite(
                successor
            )
        )
    ):
        raise RuntimeError(
            "Generated dynamical dataset contains nonfinite state values."
        )

    return SampledDynamicalDataset(
        sampling_id=
            DYNAMICAL_SAMPLING_ID,
        pair_law_id=
            PAIR_LAW_ID,
        system=
            system_value,
        evaluation_mode=
            mode_value,
        sample_size=
            sample_size_value,
        replicate_index=
            replicate_value,
        dt=
            dt_value,
        horizon_steps=
            horizon_value,
        vanderpol_mu=
            mu_value,
        current_states=
            current,
        successor_states=
            successor,
        time_indices=
            tuple(
                time_indices
            ),
        trajectory_seed_records=
            tuple(
                trajectory_records
            ),
        time_seed_records=
            tuple(
                time_records
            ),
    )
