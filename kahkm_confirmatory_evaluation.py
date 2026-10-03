"""Confirmatory dynamical evaluation/evidence compatibility layer.

The confirmatory sampler intentionally has its own frozen campaign-specific
dataset type.  The already validated dynamical evidence subsystem accepts
the earlier generic ``SampledDynamicalDataset`` type.

This module performs a strict, lossless conversion between those two
representations and delegates numerical evaluation and evidence persistence
to the already validated implementations.

It does not fit or alter a representation and does not itself generate
dynamical pairs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kahkm_confirmatory_sampling import (
    BIT_GENERATOR_ID,
    CONFIRMATORY_CAMPAIGN_ID,
    CONFIRMATORY_CAMPAIGN_KEY,
    CONFIRMATORY_DT,
    CONFIRMATORY_HORIZON_STEPS,
    CONFIRMATORY_REPLICATES_PER_MODE,
    CONFIRMATORY_ROOT_SEED,
    CONFIRMATORY_SAMPLE_SIZE,
    CONFIRMATORY_SYSTEM,
    CONFIRMATORY_SYSTEM_KEY,
    CONFIRMATORY_VANDERPOL_MU,
    DYNAMICAL_SAMPLING_ID,
    MODE_KEYS,
    SEED_SEQUENCE_ID,
    STREAM_KEYS,
    ConfirmatorySampledDataset,
    SeedRecord as ConfirmatorySeedRecord,
)
from kahkm_dynamical_evaluation import (
    DynamicalEvaluationResult,
    evaluate_frozen_dynamical_pairs,
)
from kahkm_dynamical_evidence import (
    write_evaluation_evidence,
    write_sample_evidence,
)
from kahkm_dynamical_pairs import (
    PAIR_LAW_ID,
)
from kahkm_dynamical_sampling import (
    SampledDynamicalDataset,
    SeedRecord as RetainedSeedRecord,
)


CONFIRMATORY_EVALUATION_ADAPTER_ID = (
    "confirmatory_dynamical_evaluation_adapter_v1"
)


def _expected_seed_material(
    *,
    evaluation_mode: str,
    sample_size: int,
    replicate_index: int,
    pair_index: int,
    stream_name: str,
) -> tuple[int, ...]:
    return (
        CONFIRMATORY_CAMPAIGN_KEY,
        CONFIRMATORY_SYSTEM_KEY,
        MODE_KEYS[
            evaluation_mode
        ],
        sample_size,
        replicate_index,
        pair_index,
        STREAM_KEYS[
            stream_name
        ],
        CONFIRMATORY_ROOT_SEED,
    )


def _generated_seed_from_words(
    words: tuple[int, int],
) -> int:
    if (
        not isinstance(
            words,
            tuple,
        )
        or len(
            words
        ) != 2
    ):
        raise ValueError(
            "generated_uint32_words must contain exactly two words."
        )

    first = int(
        words[
            0
        ]
    )

    second = int(
        words[
            1
        ]
    )

    for value in (
        first,
        second,
    ):
        if not (
            0
            <= value
            <= np.iinfo(
                np.uint32
            ).max
        ):
            raise ValueError(
                "generated_uint32_words contains a non-uint32 value."
            )

    return (
        first
        | (
            second
            << 32
        )
    )


def _validate_seed_record(
    *,
    record: ConfirmatorySeedRecord,
    dataset: ConfirmatorySampledDataset,
    pair_index: int,
    stream_name: str,
) -> None:
    if not isinstance(
        record,
        ConfirmatorySeedRecord,
    ):
        raise TypeError(
            "Confirmatory seed provenance must use ConfirmatorySeedRecord."
        )

    expected_material = (
        _expected_seed_material(
            evaluation_mode=
                dataset.evaluation_mode,

            sample_size=
                dataset.sample_size,

            replicate_index=
                dataset.replicate_index,

            pair_index=
                pair_index,

            stream_name=
                stream_name,
        )
    )

    expected_values = {
        "sampling_id":
            DYNAMICAL_SAMPLING_ID,

        "bit_generator":
            BIT_GENERATOR_ID,

        "seed_sequence":
            SEED_SEQUENCE_ID,

        "campaign_key":
            CONFIRMATORY_CAMPAIGN_KEY,

        "system":
            CONFIRMATORY_SYSTEM,

        "system_key":
            CONFIRMATORY_SYSTEM_KEY,

        "evaluation_mode":
            dataset.evaluation_mode,

        "mode_key":
            MODE_KEYS[
                dataset.evaluation_mode
            ],

        "sample_size":
            dataset.sample_size,

        "replicate_index":
            dataset.replicate_index,

        "pair_index":
            pair_index,

        "stream_name":
            stream_name,

        "stream_key":
            STREAM_KEYS[
                stream_name
            ],

        "root_seed":
            CONFIRMATORY_ROOT_SEED,

        "seed_material":
            expected_material,
    }

    for field, expected in expected_values.items():
        actual = getattr(
            record,
            field,
        )

        if actual != expected:
            raise ValueError(
                "Confirmatory seed record mismatch for "
                f"{field}: {actual!r} != {expected!r}."
            )

    words = tuple(
        int(
            value
        )
        for value
        in record.generated_uint32_words
    )

    if (
        _generated_seed_from_words(
            words
        )
        != int(
            record.generated_seed
        )
    ):
        raise ValueError(
            "Confirmatory generated seed does not match retained uint32 "
            "words."
        )

    direct_words_array = (
        np.random.SeedSequence(
            entropy=
                expected_material
        ).generate_state(
            2,
            dtype=np.uint32,
        )
    )

    direct_words = (
        int(
            direct_words_array[
                0
            ]
        ),
        int(
            direct_words_array[
                1
            ]
        ),
    )

    if words != direct_words:
        raise ValueError(
            "Confirmatory seed record differs from direct SeedSequence "
            "reconstruction."
        )


def validate_confirmatory_dataset(
    dataset: ConfirmatorySampledDataset,
) -> None:
    """Validate one sampled dataset against the frozen confirmatory design."""

    if not isinstance(
        dataset,
        ConfirmatorySampledDataset,
    ):
        raise TypeError(
            "dataset must be ConfirmatorySampledDataset."
        )

    expected_scalars = {
        "sampling_id":
            DYNAMICAL_SAMPLING_ID,

        "campaign_id":
            CONFIRMATORY_CAMPAIGN_ID,

        "system":
            CONFIRMATORY_SYSTEM,

        "sample_size":
            CONFIRMATORY_SAMPLE_SIZE,

        "horizon_steps":
            CONFIRMATORY_HORIZON_STEPS,

        "dt":
            CONFIRMATORY_DT,

        "vanderpol_mu":
            CONFIRMATORY_VANDERPOL_MU,

        "root_seed":
            CONFIRMATORY_ROOT_SEED,

        "campaign_key":
            CONFIRMATORY_CAMPAIGN_KEY,

        "system_key":
            CONFIRMATORY_SYSTEM_KEY,
    }

    for field, expected in expected_scalars.items():
        actual = getattr(
            dataset,
            field,
        )

        if actual != expected:
            raise ValueError(
                "Confirmatory dataset mismatch for "
                f"{field}: {actual!r} != {expected!r}."
            )

    if dataset.evaluation_mode not in MODE_KEYS:
        raise ValueError(
            "Unexpected confirmatory evaluation mode."
        )

    if (
        dataset.mode_key
        != MODE_KEYS[
            dataset.evaluation_mode
        ]
    ):
        raise ValueError(
            "Confirmatory mode key does not match evaluation mode."
        )

    if not (
        0
        <= int(
            dataset.replicate_index
        )
        < CONFIRMATORY_REPLICATES_PER_MODE
    ):
        raise ValueError(
            "Confirmatory replicate index is outside the frozen design."
        )

    M = int(
        dataset.sample_size
    )

    current = np.asarray(
        dataset.current_states,
        dtype=np.float64,
    )

    successor = np.asarray(
        dataset.successor_states,
        dtype=np.float64,
    )

    time_indices = np.asarray(
        dataset.time_indices,
        dtype=np.int64,
    )

    if current.shape != (
        2,
        M,
    ):
        raise ValueError(
            "Confirmatory current_states shape mismatch."
        )

    if successor.shape != (
        2,
        M,
    ):
        raise ValueError(
            "Confirmatory successor_states shape mismatch."
        )

    if time_indices.shape != (
        M,
    ):
        raise ValueError(
            "Confirmatory time_indices shape mismatch."
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
        raise ValueError(
            "Confirmatory state arrays contain non-finite values."
        )

    if (
        np.any(
            time_indices
            < 0
        )
        or np.any(
            time_indices
            >= CONFIRMATORY_HORIZON_STEPS
        )
    ):
        raise ValueError(
            "Confirmatory time indices lie outside the frozen horizon."
        )

    if (
        len(
            dataset.trajectory_seed_records
        )
        != M
        or len(
            dataset.time_seed_records
        )
        != M
    ):
        raise ValueError(
            "Confirmatory seed-record count does not equal sample size."
        )

    for pair_index in range(
        M
    ):
        trajectory = (
            dataset.trajectory_seed_records[
                pair_index
            ]
        )

        time_record = (
            dataset.time_seed_records[
                pair_index
            ]
        )

        _validate_seed_record(
            record=
                trajectory,

            dataset=
                dataset,

            pair_index=
                pair_index,

            stream_name=
                "trajectory_seed",
        )

        _validate_seed_record(
            record=
                time_record,

            dataset=
                dataset,

            pair_index=
                pair_index,

            stream_name=
                "time_seed",
        )

        if (
            trajectory.generated_seed
            == time_record.generated_seed
        ):
            raise ValueError(
                "Trajectory/time generated seeds collide within a pair."
            )


def _to_retained_seed_record(
    record: ConfirmatorySeedRecord,
) -> RetainedSeedRecord:
    return RetainedSeedRecord(
        sampling_id=
            record.sampling_id,

        bit_generator=
            record.bit_generator,

        seed_sequence=
            record.seed_sequence,

        campaign_key=
            int(
                record.campaign_key
            ),

        system=
            record.system,

        system_key=
            int(
                record.system_key
            ),

        evaluation_mode=
            record.evaluation_mode,

        mode_key=
            int(
                record.mode_key
            ),

        sample_size=
            int(
                record.sample_size
            ),

        replicate_index=
            int(
                record.replicate_index
            ),

        pair_index=
            int(
                record.pair_index
            ),

        stream_name=
            record.stream_name,

        stream_key=
            int(
                record.stream_key
            ),

        root_seed=
            int(
                record.root_seed
            ),

        seed_material=
            tuple(
                int(
                    value
                )
                for value
                in record.seed_material
            ),

        generated_uint32_words=
            tuple(
                int(
                    value
                )
                for value
                in record.generated_uint32_words
            ),

        generated_seed=
            int(
                record.generated_seed
            ),
    )


def to_retained_dynamical_dataset(
    dataset: ConfirmatorySampledDataset,
) -> SampledDynamicalDataset:
    """Losslessly adapt confirmatory sampling to validated evidence APIs."""

    validate_confirmatory_dataset(
        dataset
    )

    current = np.array(
        dataset.current_states,
        dtype=np.float64,
        copy=True,
        order="C",
    )

    successor = np.array(
        dataset.successor_states,
        dtype=np.float64,
        copy=True,
        order="C",
    )

    current.setflags(
        write=False
    )

    successor.setflags(
        write=False
    )

    return SampledDynamicalDataset(
        sampling_id=
            DYNAMICAL_SAMPLING_ID,

        pair_law_id=
            PAIR_LAW_ID,

        system=
            dataset.system,

        evaluation_mode=
            dataset.evaluation_mode,

        sample_size=
            int(
                dataset.sample_size
            ),

        replicate_index=
            int(
                dataset.replicate_index
            ),

        dt=
            float(
                dataset.dt
            ),

        horizon_steps=
            int(
                dataset.horizon_steps
            ),

        vanderpol_mu=
            float(
                dataset.vanderpol_mu
            ),

        current_states=
            current,

        successor_states=
            successor,

        time_indices=
            tuple(
                int(
                    value
                )
                for value
                in np.asarray(
                    dataset.time_indices,
                    dtype=np.int64,
                )
            ),

        trajectory_seed_records=
            tuple(
                _to_retained_seed_record(
                    record
                )
                for record
                in dataset.trajectory_seed_records
            ),

        time_seed_records=
            tuple(
                _to_retained_seed_record(
                    record
                )
                for record
                in dataset.time_seed_records
            ),
    )


def evaluate_confirmatory_dataset(
    *,
    dataset: ConfirmatorySampledDataset,
    abstraction_model: dict[str, Any],
    B: np.ndarray,
    omega: float,
    tau: float,
    kappas: Iterable[float],
    delta: float,
    n_jobs: int,
    batch_size: int,
) -> DynamicalEvaluationResult:
    """Evaluate a validated confirmatory dataset with the frozen model."""

    validate_confirmatory_dataset(
        dataset
    )

    return evaluate_frozen_dynamical_pairs(
        abstraction_model=
            abstraction_model,

        B=
            B,

        omega=
            omega,

        tau=
            tau,

        current_states=
            dataset.current_states,

        successor_states=
            dataset.successor_states,

        kappas=
            kappas,

        delta=
            delta,

        n_jobs=
            n_jobs,

        batch_size=
            batch_size,
    )


def write_confirmatory_sample_evidence(
    *,
    attempt_dir: str | os.PathLike[str],
    dataset: ConfirmatorySampledDataset,
) -> dict[str, Any]:
    """Persist sample evidence through the validated generic writer."""

    retained = (
        to_retained_dynamical_dataset(
            dataset
        )
    )

    return write_sample_evidence(
        attempt_dir=
            Path(
                attempt_dir
            ),

        dataset=
            retained,
    )


def write_confirmatory_evaluation_evidence(
    *,
    attempt_dir: str | os.PathLike[str],
    dataset: ConfirmatorySampledDataset,
    evaluation: DynamicalEvaluationResult,
) -> dict[str, Any]:
    """Persist evaluation evidence through the validated generic writer."""

    retained = (
        to_retained_dynamical_dataset(
            dataset
        )
    )

    return write_evaluation_evidence(
        attempt_dir=
            Path(
                attempt_dir
            ),

        dataset=
            retained,

        evaluation=
            evaluation,
    )
