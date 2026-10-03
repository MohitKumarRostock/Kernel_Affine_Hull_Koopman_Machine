"""Fresh confirmatory dynamical-pair sampling.

The representation is already frozen before this module is used.

This module preserves the validated dynamical pair law and RNG construction
while using the prospectively frozen confirmatory campaign key ``4``.

Confirmatory design:
- system: Van der Pol only;
- modes: matched_stochastic and deterministic;
- M = 16384 independent pairs per replicate;
- 32 replicates per mode;
- campaign key = 4;
- trajectory stream key = 1;
- time-index stream key = 2;
- root seed = 20261003.

Every sampled pair receives its own trajectory and time-index seed derived
from an eight-integer SeedSequence entropy tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

import numpy as np

from kahkm_dynamical_pairs import (
    sample_independent_pair,
)


DYNAMICAL_SAMPLING_ID: Final[str] = (
    "dynamical_certificate_sampling_v1"
)

CONFIRMATORY_CAMPAIGN_ID: Final[str] = (
    "dynamical_original_iid_confirmatory_v1"
)

CONFIRMATORY_SYSTEM: Final[str] = (
    "vanderpol"
)

CONFIRMATORY_ROOT_SEED: Final[int] = (
    20261003
)

CONFIRMATORY_CAMPAIGN_KEY: Final[int] = (
    4
)

PILOT_CAMPAIGN_KEY: Final[int] = (
    3
)

CONFIRMATORY_SYSTEM_KEY: Final[int] = (
    2
)

MODE_KEYS: Final[dict[str, int]] = {
    "matched_stochastic":
        1,

    "deterministic":
        2,
}

STREAM_KEYS: Final[dict[str, int]] = {
    "trajectory_seed":
        1,

    "time_seed":
        2,
}

CONFIRMATORY_SAMPLE_SIZE: Final[int] = (
    16384
)

CONFIRMATORY_REPLICATES_PER_MODE: Final[int] = (
    32
)

CONFIRMATORY_HORIZON_STEPS: Final[int] = (
    1200
)

CONFIRMATORY_DT: Final[float] = (
    0.02
)

CONFIRMATORY_VANDERPOL_MU: Final[float] = (
    1.0
)

SEED_SEQUENCE_ID: Final[str] = (
    "numpy.random.SeedSequence"
)

BIT_GENERATOR_ID: Final[str] = (
    "PCG64"
)

CONFIRMATORY_NUM_MODES: Final[int] = (
    len(
        MODE_KEYS
    )
)

CONFIRMATORY_ATTEMPT_COUNT: Final[int] = (
    CONFIRMATORY_NUM_MODES
    * CONFIRMATORY_REPLICATES_PER_MODE
)

CONFIRMATORY_PAIR_COUNT: Final[int] = (
    CONFIRMATORY_ATTEMPT_COUNT
    * CONFIRMATORY_SAMPLE_SIZE
)

CONFIRMATORY_SEED_RECORD_COUNT: Final[int] = (
    CONFIRMATORY_PAIR_COUNT
    * len(
        STREAM_KEYS
    )
)


@dataclass(frozen=True)
class SeedRecord:
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

    seed_material: tuple[
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
    ]

    generated_uint32_words: tuple[
        int,
        int,
    ]

    generated_seed: int


@dataclass(frozen=True)
class ConfirmatorySampledDataset:
    sampling_id: str
    campaign_id: str

    system: str
    evaluation_mode: str

    sample_size: int
    replicate_index: int

    horizon_steps: int
    dt: float
    vanderpol_mu: float

    root_seed: int
    campaign_key: int
    system_key: int
    mode_key: int

    current_states: np.ndarray
    successor_states: np.ndarray
    time_indices: np.ndarray

    trajectory_seed_records: tuple[
        SeedRecord,
        ...,
    ]

    time_seed_records: tuple[
        SeedRecord,
        ...,
    ]


def _require_integer(
    name: str,
    value: Integral,
) -> int:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Integral,
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    return int(
        value
    )


def _validate_mode(
    evaluation_mode: str,
) -> int:
    if (
        not isinstance(
            evaluation_mode,
            str,
        )
        or evaluation_mode
        not in MODE_KEYS
    ):
        raise ValueError(
            "evaluation_mode must be one of "
            f"{tuple(MODE_KEYS)}."
        )

    return MODE_KEYS[
        evaluation_mode
    ]


def _validate_replicate(
    replicate_index: Integral,
) -> int:
    value = _require_integer(
        "replicate_index",
        replicate_index,
    )

    if not (
        0
        <= value
        < CONFIRMATORY_REPLICATES_PER_MODE
    ):
        raise ValueError(
            "replicate_index is outside the frozen confirmatory "
            f"range [0, {CONFIRMATORY_REPLICATES_PER_MODE})."
        )

    return value


def _validate_pair_index(
    pair_index: Integral,
    *,
    sample_size: int,
) -> int:
    value = _require_integer(
        "pair_index",
        pair_index,
    )

    if not (
        0
        <= value
        < sample_size
    ):
        raise ValueError(
            "pair_index is outside the requested sample."
        )

    return value


def confirmatory_seed_material(
    *,
    evaluation_mode: str,
    sample_size: Integral,
    replicate_index: Integral,
    pair_index: Integral,
    stream_name: str,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """Return the exact frozen eight-integer confirmatory seed material."""

    mode_key = _validate_mode(
        evaluation_mode
    )

    sample_size_int = _require_integer(
        "sample_size",
        sample_size,
    )

    if (
        sample_size_int
        != CONFIRMATORY_SAMPLE_SIZE
    ):
        raise ValueError(
            "sample_size must equal the prospectively frozen "
            f"value {CONFIRMATORY_SAMPLE_SIZE}."
        )

    replicate = _validate_replicate(
        replicate_index
    )

    pair = _validate_pair_index(
        pair_index,
        sample_size=
            sample_size_int,
    )

    if (
        not isinstance(
            stream_name,
            str,
        )
        or stream_name
        not in STREAM_KEYS
    ):
        raise ValueError(
            "stream_name must be one of "
            f"{tuple(STREAM_KEYS)}."
        )

    stream_key = STREAM_KEYS[
        stream_name
    ]

    return (
        CONFIRMATORY_CAMPAIGN_KEY,
        CONFIRMATORY_SYSTEM_KEY,
        mode_key,
        sample_size_int,
        replicate,
        pair,
        stream_key,
        CONFIRMATORY_ROOT_SEED,
    )


def confirmatory_seed_record(
    *,
    evaluation_mode: str,
    sample_size: Integral,
    replicate_index: Integral,
    pair_index: Integral,
    stream_name: str,
) -> SeedRecord:
    """Derive one trajectory or time-index seed record."""

    material = (
        confirmatory_seed_material(
            evaluation_mode=
                evaluation_mode,

            sample_size=
                sample_size,

            replicate_index=
                replicate_index,

            pair_index=
                pair_index,

            stream_name=
                stream_name,
        )
    )

    seed_sequence = (
        np.random.SeedSequence(
            entropy=
                material
        )
    )

    words_array = (
        seed_sequence.generate_state(
            2,
            dtype=np.uint32,
        )
    )

    words = (
        int(
            words_array[
                0
            ]
        ),
        int(
            words_array[
                1
            ]
        ),
    )

    generated_seed = (
        words[
            0
        ]
        | (
            words[
                1
            ]
            << 32
        )
    )

    mode_key = MODE_KEYS[
        evaluation_mode
    ]

    stream_key = STREAM_KEYS[
        stream_name
    ]

    return SeedRecord(
        sampling_id=
            DYNAMICAL_SAMPLING_ID,

        bit_generator=
            BIT_GENERATOR_ID,

        seed_sequence=
            SEED_SEQUENCE_ID,

        campaign_key=
            CONFIRMATORY_CAMPAIGN_KEY,

        system=
            CONFIRMATORY_SYSTEM,

        system_key=
            CONFIRMATORY_SYSTEM_KEY,

        evaluation_mode=
            evaluation_mode,

        mode_key=
            mode_key,

        sample_size=
            int(
                sample_size
            ),

        replicate_index=
            int(
                replicate_index
            ),

        pair_index=
            int(
                pair_index
            ),

        stream_name=
            stream_name,

        stream_key=
            stream_key,

        root_seed=
            CONFIRMATORY_ROOT_SEED,

        seed_material=
            material,

        generated_uint32_words=
            words,

        generated_seed=
            generated_seed,
    )


def pilot_seed_material_for_disjointness_check(
    *,
    evaluation_mode: str,
    sample_size: Integral,
    replicate_index: Integral,
    pair_index: Integral,
    stream_name: str,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """Construct the corresponding pilot-key tuple for disjointness checks."""

    confirmatory = (
        confirmatory_seed_material(
            evaluation_mode=
                evaluation_mode,

            sample_size=
                sample_size,

            replicate_index=
                replicate_index,

            pair_index=
                pair_index,

            stream_name=
                stream_name,
        )
    )

    return (
        PILOT_CAMPAIGN_KEY,
        *confirmatory[
            1:
        ],
    )


def confirmatory_schedule_audit() -> dict[str, object]:
    """Return deterministic cardinality/disjointness facts for the design."""

    if (
        CONFIRMATORY_CAMPAIGN_KEY
        == PILOT_CAMPAIGN_KEY
    ):
        raise RuntimeError(
            "Confirmatory and pilot campaign keys must differ."
        )

    if set(
        MODE_KEYS.values()
    ) != {
        1,
        2,
    }:
        raise RuntimeError(
            "Unexpected confirmatory mode-key mapping."
        )

    if set(
        STREAM_KEYS.values()
    ) != {
        1,
        2,
    }:
        raise RuntimeError(
            "Unexpected confirmatory stream-key mapping."
        )

    if (
        CONFIRMATORY_ATTEMPT_COUNT
        != 64
    ):
        raise RuntimeError(
            "Confirmatory attempt count must be 64."
        )

    if (
        CONFIRMATORY_PAIR_COUNT
        != 1_048_576
    ):
        raise RuntimeError(
            "Confirmatory pair count must be 1,048,576."
        )

    if (
        CONFIRMATORY_SEED_RECORD_COUNT
        != 2_097_152
    ):
        raise RuntimeError(
            "Confirmatory seed-record count must be 2,097,152."
        )

    # The complete entropy tuple is injective over the Cartesian product
    # because every varying coordinate appears explicitly in the tuple.
    # Pilot disjointness follows from the different first coordinate.
    return {
        "campaign_id":
            CONFIRMATORY_CAMPAIGN_ID,

        "campaign_key":
            CONFIRMATORY_CAMPAIGN_KEY,

        "pilot_campaign_key":
            PILOT_CAMPAIGN_KEY,

        "system":
            CONFIRMATORY_SYSTEM,

        "system_key":
            CONFIRMATORY_SYSTEM_KEY,

        "evaluation_modes":
            tuple(
                MODE_KEYS
            ),

        "mode_keys":
            dict(
                MODE_KEYS
            ),

        "stream_keys":
            dict(
                STREAM_KEYS
            ),

        "sample_size":
            CONFIRMATORY_SAMPLE_SIZE,

        "replicates_per_mode":
            CONFIRMATORY_REPLICATES_PER_MODE,

        "attempt_count":
            CONFIRMATORY_ATTEMPT_COUNT,

        "pair_count":
            CONFIRMATORY_PAIR_COUNT,

        "seed_record_count":
            CONFIRMATORY_SEED_RECORD_COUNT,

        "root_seed":
            CONFIRMATORY_ROOT_SEED,

        "seed_material_length":
            8,

        "seed_material_order": (
            "campaign_key",
            "system_key",
            "mode_key",
            "sample_size",
            "replicate_index",
            "pair_index",
            "stream_key",
            "root_seed",
        ),

        "seed_material_injective":
            True,

        "pilot_seed_material_disjoint":
            True,
    }


def sample_confirmatory_dataset(
    *,
    evaluation_mode: str,
    replicate_index: Integral,
) -> ConfirmatorySampledDataset:
    """Generate one prospectively frozen confirmatory dataset."""

    mode_key = _validate_mode(
        evaluation_mode
    )

    replicate = _validate_replicate(
        replicate_index
    )

    sample_size = (
        CONFIRMATORY_SAMPLE_SIZE
    )

    current_states = np.empty(
        (
            2,
            sample_size,
        ),
        dtype=np.float64,
    )

    successor_states = np.empty(
        (
            2,
            sample_size,
        ),
        dtype=np.float64,
    )

    time_indices = np.empty(
        sample_size,
        dtype=np.int64,
    )

    trajectory_records: list[
        SeedRecord
    ] = []

    time_records: list[
        SeedRecord
    ] = []

    for pair_index in range(
        sample_size
    ):
        trajectory_record = (
            confirmatory_seed_record(
                evaluation_mode=
                    evaluation_mode,

                sample_size=
                    sample_size,

                replicate_index=
                    replicate,

                pair_index=
                    pair_index,

                stream_name=
                    "trajectory_seed",
            )
        )

        time_record = (
            confirmatory_seed_record(
                evaluation_mode=
                    evaluation_mode,

                sample_size=
                    sample_size,

                replicate_index=
                    replicate,

                pair_index=
                    pair_index,

                stream_name=
                    "time_seed",
            )
        )

        if (
            trajectory_record.generated_seed
            == time_record.generated_seed
        ):
            raise RuntimeError(
                "Trajectory and time-index generated seeds collided "
                f"at pair {pair_index}."
            )

        pair = sample_independent_pair(
            system=
                CONFIRMATORY_SYSTEM,

            evaluation_mode=
                evaluation_mode,

            horizon_steps=
                CONFIRMATORY_HORIZON_STEPS,

            dt=
                CONFIRMATORY_DT,

            trajectory_seed=
                trajectory_record.generated_seed,

            time_seed=
                time_record.generated_seed,

            vanderpol_mu=
                CONFIRMATORY_VANDERPOL_MU,
        )

        if pair.system != CONFIRMATORY_SYSTEM:
            raise RuntimeError(
                "Pair sampler returned the wrong system."
            )

        if pair.evaluation_mode != evaluation_mode:
            raise RuntimeError(
                "Pair sampler returned the wrong evaluation mode."
            )

        if (
            pair.trajectory_seed
            != trajectory_record.generated_seed
        ):
            raise RuntimeError(
                "Pair sampler trajectory-seed mismatch."
            )

        if (
            pair.time_seed
            != time_record.generated_seed
        ):
            raise RuntimeError(
                "Pair sampler time-seed mismatch."
            )

        if (
            pair.horizon_steps
            != CONFIRMATORY_HORIZON_STEPS
        ):
            raise RuntimeError(
                "Pair sampler horizon mismatch."
            )

        if (
            pair.dt
            != CONFIRMATORY_DT
        ):
            raise RuntimeError(
                "Pair sampler dt mismatch."
            )

        current = np.asarray(
            pair.current_state,
            dtype=np.float64,
        ).reshape(
            -1
        )

        successor = np.asarray(
            pair.successor_state,
            dtype=np.float64,
        ).reshape(
            -1
        )

        if (
            current.shape
            != (
                2,
            )
            or successor.shape
            != (
                2,
            )
        ):
            raise RuntimeError(
                "Pair sampler returned unexpected state dimension."
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
                "Pair sampler returned non-finite state coordinates."
            )

        time_index = int(
            pair.time_index
        )

        if not (
            0
            <= time_index
            < CONFIRMATORY_HORIZON_STEPS
        ):
            raise RuntimeError(
                "Pair sampler returned an invalid time index."
            )

        current_states[
            :,
            pair_index,
        ] = current

        successor_states[
            :,
            pair_index,
        ] = successor

        time_indices[
            pair_index
        ] = time_index

        trajectory_records.append(
            trajectory_record
        )

        time_records.append(
            time_record
        )

    if (
        len(
            trajectory_records
        )
        != sample_size
        or len(
            time_records
        )
        != sample_size
    ):
        raise RuntimeError(
            "Seed-record count mismatch."
        )

    return ConfirmatorySampledDataset(
        sampling_id=
            DYNAMICAL_SAMPLING_ID,

        campaign_id=
            CONFIRMATORY_CAMPAIGN_ID,

        system=
            CONFIRMATORY_SYSTEM,

        evaluation_mode=
            evaluation_mode,

        sample_size=
            sample_size,

        replicate_index=
            replicate,

        horizon_steps=
            CONFIRMATORY_HORIZON_STEPS,

        dt=
            CONFIRMATORY_DT,

        vanderpol_mu=
            CONFIRMATORY_VANDERPOL_MU,

        root_seed=
            CONFIRMATORY_ROOT_SEED,

        campaign_key=
            CONFIRMATORY_CAMPAIGN_KEY,

        system_key=
            CONFIRMATORY_SYSTEM_KEY,

        mode_key=
            mode_key,

        current_states=
            current_states,

        successor_states=
            successor_states,

        time_indices=
            time_indices,

        trajectory_seed_records=
            tuple(
                trajectory_records
            ),

        time_seed_records=
            tuple(
                time_records
            ),
    )
