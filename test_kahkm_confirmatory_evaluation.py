from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import kahkm_confirmatory_evaluation as adapter
from kahkm_confirmatory_sampling import (
    ConfirmatorySampledDataset,
    SeedRecord,
)
from kahkm_dynamical_sampling import (
    SampledDynamicalDataset,
)


def seed_record(
    *,
    mode: str,
    sample_size: int,
    replicate: int,
    pair_index: int,
    stream_name: str,
) -> SeedRecord:
    stream_key = adapter.STREAM_KEYS[
        stream_name
    ]

    material = (
        adapter.CONFIRMATORY_CAMPAIGN_KEY,
        adapter.CONFIRMATORY_SYSTEM_KEY,
        adapter.MODE_KEYS[
            mode
        ],
        sample_size,
        replicate,
        pair_index,
        stream_key,
        adapter.CONFIRMATORY_ROOT_SEED,
    )

    words_array = np.random.SeedSequence(
        entropy=
            material
    ).generate_state(
        2,
        dtype=np.uint32,
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

    return SeedRecord(
        sampling_id=
            adapter.DYNAMICAL_SAMPLING_ID,

        bit_generator=
            adapter.BIT_GENERATOR_ID,

        seed_sequence=
            adapter.SEED_SEQUENCE_ID,

        campaign_key=
            adapter.CONFIRMATORY_CAMPAIGN_KEY,

        system=
            adapter.CONFIRMATORY_SYSTEM,

        system_key=
            adapter.CONFIRMATORY_SYSTEM_KEY,

        evaluation_mode=
            mode,

        mode_key=
            adapter.MODE_KEYS[
                mode
            ],

        sample_size=
            sample_size,

        replicate_index=
            replicate,

        pair_index=
            pair_index,

        stream_name=
            stream_name,

        stream_key=
            stream_key,

        root_seed=
            adapter.CONFIRMATORY_ROOT_SEED,

        seed_material=
            material,

        generated_uint32_words=
            words,

        generated_seed=
            generated_seed,
    )


def small_dataset(
    *,
    mode: str = "matched_stochastic",
    replicate: int = 2,
    sample_size: int = 3,
) -> ConfirmatorySampledDataset:
    trajectory = tuple(
        seed_record(
            mode=
                mode,

            sample_size=
                sample_size,

            replicate=
                replicate,

            pair_index=
                pair_index,

            stream_name=
                "trajectory_seed",
        )
        for pair_index in range(
            sample_size
        )
    )

    times = tuple(
        seed_record(
            mode=
                mode,

            sample_size=
                sample_size,

            replicate=
                replicate,

            pair_index=
                pair_index,

            stream_name=
                "time_seed",
        )
        for pair_index in range(
            sample_size
        )
    )

    return ConfirmatorySampledDataset(
        sampling_id=
            adapter.DYNAMICAL_SAMPLING_ID,

        campaign_id=
            adapter.CONFIRMATORY_CAMPAIGN_ID,

        system=
            adapter.CONFIRMATORY_SYSTEM,

        evaluation_mode=
            mode,

        sample_size=
            sample_size,

        replicate_index=
            replicate,

        horizon_steps=
            adapter.CONFIRMATORY_HORIZON_STEPS,

        dt=
            adapter.CONFIRMATORY_DT,

        vanderpol_mu=
            adapter.CONFIRMATORY_VANDERPOL_MU,

        root_seed=
            adapter.CONFIRMATORY_ROOT_SEED,

        campaign_key=
            adapter.CONFIRMATORY_CAMPAIGN_KEY,

        system_key=
            adapter.CONFIRMATORY_SYSTEM_KEY,

        mode_key=
            adapter.MODE_KEYS[
                mode
            ],

        current_states=
            np.arange(
                2
                * sample_size,
                dtype=np.float64,
            ).reshape(
                2,
                sample_size,
            ),

        successor_states=
            (
                np.arange(
                    2
                    * sample_size,
                    dtype=np.float64,
                ).reshape(
                    2,
                    sample_size,
                )
                + 0.5
            ),

        time_indices=
            np.arange(
                sample_size,
                dtype=np.int64,
            ),

        trajectory_seed_records=
            trajectory,

        time_seed_records=
            times,
    )


class ConfirmatoryEvaluationAdapterTests(
    unittest.TestCase
):
    def test_adapter_id_and_pair_law_are_frozen(
        self,
    ):
        self.assertEqual(
            adapter.CONFIRMATORY_EVALUATION_ADAPTER_ID,
            "confirmatory_dynamical_evaluation_adapter_v1",
        )

        self.assertEqual(
            adapter.PAIR_LAW_ID,
            "independent_uniform_time_pair_v1",
        )


    def test_lossless_conversion_to_validated_evidence_dataset(
        self,
    ):
        dataset = small_dataset()

        with mock.patch.object(
            adapter,
            "CONFIRMATORY_SAMPLE_SIZE",
            dataset.sample_size,
        ):
            retained = (
                adapter.to_retained_dynamical_dataset(
                    dataset
                )
            )

        self.assertIsInstance(
            retained,
            SampledDynamicalDataset,
        )

        self.assertEqual(
            retained.sampling_id,
            dataset.sampling_id,
        )

        self.assertEqual(
            retained.pair_law_id,
            adapter.PAIR_LAW_ID,
        )

        self.assertEqual(
            retained.system,
            dataset.system,
        )

        self.assertEqual(
            retained.evaluation_mode,
            dataset.evaluation_mode,
        )

        self.assertEqual(
            retained.sample_size,
            dataset.sample_size,
        )

        self.assertEqual(
            retained.replicate_index,
            dataset.replicate_index,
        )

        np.testing.assert_array_equal(
            retained.current_states,
            dataset.current_states,
        )

        np.testing.assert_array_equal(
            retained.successor_states,
            dataset.successor_states,
        )

        self.assertEqual(
            retained.time_indices,
            tuple(
                int(value)
                for value
                in dataset.time_indices
            ),
        )

        self.assertFalse(
            retained.current_states.flags.writeable
        )

        self.assertFalse(
            retained.successor_states.flags.writeable
        )

        self.assertEqual(
            len(
                retained.trajectory_seed_records
            ),
            dataset.sample_size,
        )

        self.assertEqual(
            len(
                retained.time_seed_records
            ),
            dataset.sample_size,
        )

        for original, converted in zip(
            dataset.trajectory_seed_records,
            retained.trajectory_seed_records,
            strict=True,
        ):
            self.assertEqual(
                converted.seed_material,
                original.seed_material,
            )

            self.assertEqual(
                converted.generated_seed,
                original.generated_seed,
            )


    def test_validation_rejects_pilot_campaign_key(
        self,
    ):
        dataset = small_dataset()

        bad = replace(
            dataset,
            campaign_key=
                3,
        )

        with mock.patch.object(
            adapter,
            "CONFIRMATORY_SAMPLE_SIZE",
            dataset.sample_size,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "campaign_key",
            ):
                adapter.validate_confirmatory_dataset(
                    bad
                )


    def test_validation_rejects_changed_seed_material(
        self,
    ):
        dataset = small_dataset()

        records = list(
            dataset.trajectory_seed_records
        )

        records[
            0
        ] = replace(
            records[
                0
            ],
            seed_material=(
                3,
                *records[
                    0
                ].seed_material[
                    1:
                ],
            ),
        )

        bad = replace(
            dataset,
            trajectory_seed_records=
                tuple(
                    records
                ),
        )

        with mock.patch.object(
            adapter,
            "CONFIRMATORY_SAMPLE_SIZE",
            dataset.sample_size,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "seed_material",
            ):
                adapter.validate_confirmatory_dataset(
                    bad
                )


    def test_evaluation_delegates_without_resampling(
        self,
    ):
        dataset = small_dataset()

        sentinel = object()

        with (
            mock.patch.object(
                adapter,
                "CONFIRMATORY_SAMPLE_SIZE",
                dataset.sample_size,
            ),
            mock.patch.object(
                adapter,
                "evaluate_frozen_dynamical_pairs",
                return_value=
                    sentinel,
            ) as evaluator,
        ):
            result = (
                adapter.evaluate_confirmatory_dataset(
                    dataset=
                        dataset,

                    abstraction_model={
                        "model":
                            "frozen"
                    },

                    B=
                        np.eye(
                            2,
                            dtype=np.float64,
                        ),

                    omega=
                        128.0,

                    tau=
                        1e-6,

                    kappas=(
                        1.0,
                        1.1,
                    ),

                    delta=
                        0.05,

                    n_jobs=
                        -1,

                    batch_size=
                        256,
                )
            )

        self.assertIs(
            result,
            sentinel,
        )

        evaluator.assert_called_once()

        kwargs = (
            evaluator.call_args.kwargs
        )

        self.assertIs(
            kwargs[
                "current_states"
            ],
            dataset.current_states,
        )

        self.assertIs(
            kwargs[
                "successor_states"
            ],
            dataset.successor_states,
        )

        self.assertEqual(
            tuple(
                kwargs[
                    "kappas"
                ]
            ),
            (
                1.0,
                1.1,
            ),
        )

        self.assertEqual(
            kwargs[
                "delta"
            ],
            0.05,
        )


    def test_sample_evidence_wrapper_passes_validated_generic_type(
        self,
    ):
        dataset = small_dataset()

        with tempfile.TemporaryDirectory() as tmp:
            attempt = (
                Path(
                    tmp
                )
                / "attempt"
            )

            with (
                mock.patch.object(
                    adapter,
                    "CONFIRMATORY_SAMPLE_SIZE",
                    dataset.sample_size,
                ),
                mock.patch.object(
                    adapter,
                    "write_sample_evidence",
                    return_value={
                        "ok":
                            True
                    },
                ) as writer,
            ):
                result = (
                    adapter.write_confirmatory_sample_evidence(
                        attempt_dir=
                            attempt,

                        dataset=
                            dataset,
                    )
                )

        self.assertEqual(
            result,
            {
                "ok":
                    True
            },
        )

        writer.assert_called_once()

        supplied = (
            writer.call_args.kwargs[
                "dataset"
            ]
        )

        self.assertIsInstance(
            supplied,
            SampledDynamicalDataset,
        )

        self.assertEqual(
            supplied.pair_law_id,
            adapter.PAIR_LAW_ID,
        )


    def test_evaluation_evidence_wrapper_passes_validated_generic_type(
        self,
    ):
        dataset = small_dataset()

        evaluation = object()

        with tempfile.TemporaryDirectory() as tmp:
            attempt = (
                Path(
                    tmp
                )
                / "attempt"
            )

            with (
                mock.patch.object(
                    adapter,
                    "CONFIRMATORY_SAMPLE_SIZE",
                    dataset.sample_size,
                ),
                mock.patch.object(
                    adapter,
                    "write_evaluation_evidence",
                    return_value={
                        "ok":
                            True
                    },
                ) as writer,
            ):
                result = (
                    adapter.write_confirmatory_evaluation_evidence(
                        attempt_dir=
                            attempt,

                        dataset=
                            dataset,

                        evaluation=
                            evaluation,
                    )
                )

        self.assertEqual(
            result,
            {
                "ok":
                    True
            },
        )

        writer.assert_called_once()

        supplied = (
            writer.call_args.kwargs[
                "dataset"
            ]
        )

        self.assertIsInstance(
            supplied,
            SampledDynamicalDataset,
        )

        self.assertIs(
            writer.call_args.kwargs[
                "evaluation"
            ],
            evaluation,
        )


if __name__ == "__main__":
    unittest.main()
