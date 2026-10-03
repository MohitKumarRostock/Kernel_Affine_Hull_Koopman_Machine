from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import kahkm_confirmatory_sampling as sampling


class ConfirmatorySamplingTests(
    unittest.TestCase
):
    def test_frozen_schedule_constants(
        self,
    ):
        self.assertEqual(
            sampling.DYNAMICAL_SAMPLING_ID,
            "dynamical_certificate_sampling_v1",
        )

        self.assertEqual(
            sampling.CONFIRMATORY_CAMPAIGN_ID,
            "dynamical_original_iid_confirmatory_v1",
        )

        self.assertEqual(
            sampling.CONFIRMATORY_SYSTEM,
            "vanderpol",
        )

        self.assertEqual(
            sampling.CONFIRMATORY_ROOT_SEED,
            20261003,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_CAMPAIGN_KEY,
            4,
        )

        self.assertEqual(
            sampling.PILOT_CAMPAIGN_KEY,
            3,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_SYSTEM_KEY,
            2,
        )

        self.assertEqual(
            sampling.MODE_KEYS,
            {
                "matched_stochastic":
                    1,

                "deterministic":
                    2,
            },
        )

        self.assertEqual(
            sampling.STREAM_KEYS,
            {
                "trajectory_seed":
                    1,

                "time_seed":
                    2,
            },
        )

        self.assertEqual(
            sampling.CONFIRMATORY_SAMPLE_SIZE,
            16384,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_REPLICATES_PER_MODE,
            32,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_HORIZON_STEPS,
            1200,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_DT,
            0.02,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_VANDERPOL_MU,
            1.0,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_ATTEMPT_COUNT,
            64,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_PAIR_COUNT,
            1_048_576,
        )

        self.assertEqual(
            sampling.CONFIRMATORY_SEED_RECORD_COUNT,
            2_097_152,
        )


    def test_seed_material_order_is_exact(
        self,
    ):
        material = (
            sampling.confirmatory_seed_material(
                evaluation_mode=
                    "matched_stochastic",

                sample_size=
                    16384,

                replicate_index=
                    7,

                pair_index=
                    1234,

                stream_name=
                    "time_seed",
            )
        )

        self.assertEqual(
            material,
            (
                4,
                2,
                1,
                16384,
                7,
                1234,
                2,
                20261003,
            ),
        )


    def test_seed_record_matches_direct_seedsequence_construction(
        self,
    ):
        record = (
            sampling.confirmatory_seed_record(
                evaluation_mode=
                    "deterministic",

                sample_size=
                    16384,

                replicate_index=
                    31,

                pair_index=
                    16383,

                stream_name=
                    "trajectory_seed",
            )
        )

        expected_material = (
            4,
            2,
            2,
            16384,
            31,
            16383,
            1,
            20261003,
        )

        direct = np.random.SeedSequence(
            entropy=
                expected_material
        ).generate_state(
            2,
            dtype=np.uint32,
        )

        expected_words = (
            int(
                direct[
                    0
                ]
            ),
            int(
                direct[
                    1
                ]
            ),
        )

        expected_seed = (
            expected_words[
                0
            ]
            | (
                expected_words[
                    1
                ]
                << 32
            )
        )

        self.assertEqual(
            record.seed_material,
            expected_material,
        )

        self.assertEqual(
            record.generated_uint32_words,
            expected_words,
        )

        self.assertEqual(
            record.generated_seed,
            expected_seed,
        )

        self.assertEqual(
            record.seed_sequence,
            "numpy.random.SeedSequence",
        )

        self.assertEqual(
            record.bit_generator,
            "PCG64",
        )


    def test_confirmatory_material_is_disjoint_from_corresponding_pilot_material(
        self,
    ):
        confirmatory = (
            sampling.confirmatory_seed_material(
                evaluation_mode=
                    "matched_stochastic",

                sample_size=
                    16384,

                replicate_index=
                    0,

                pair_index=
                    0,

                stream_name=
                    "trajectory_seed",
            )
        )

        pilot = (
            sampling.pilot_seed_material_for_disjointness_check(
                evaluation_mode=
                    "matched_stochastic",

                sample_size=
                    16384,

                replicate_index=
                    0,

                pair_index=
                    0,

                stream_name=
                    "trajectory_seed",
            )
        )

        self.assertEqual(
            confirmatory[
                0
            ],
            4,
        )

        self.assertEqual(
            pilot[
                0
            ],
            3,
        )

        self.assertEqual(
            confirmatory[
                1:
            ],
            pilot[
                1:
            ],
        )

        self.assertNotEqual(
            confirmatory,
            pilot,
        )


    def test_schedule_audit_has_frozen_cardinalities(
        self,
    ):
        audit = (
            sampling.confirmatory_schedule_audit()
        )

        self.assertEqual(
            audit[
                "attempt_count"
            ],
            64,
        )

        self.assertEqual(
            audit[
                "pair_count"
            ],
            1_048_576,
        )

        self.assertEqual(
            audit[
                "seed_record_count"
            ],
            2_097_152,
        )

        self.assertEqual(
            audit[
                "seed_material_order"
            ],
            (
                "campaign_key",
                "system_key",
                "mode_key",
                "sample_size",
                "replicate_index",
                "pair_index",
                "stream_key",
                "root_seed",
            ),
        )

        self.assertTrue(
            audit[
                "seed_material_injective"
            ]
        )

        self.assertTrue(
            audit[
                "pilot_seed_material_disjoint"
            ]
        )


    def test_invalid_protocol_coordinates_are_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            sampling.confirmatory_seed_material(
                evaluation_mode=
                    "invalid",

                sample_size=
                    16384,

                replicate_index=
                    0,

                pair_index=
                    0,

                stream_name=
                    "trajectory_seed",
            )

        with self.assertRaises(
            ValueError
        ):
            sampling.confirmatory_seed_material(
                evaluation_mode=
                    "deterministic",

                sample_size=
                    2048,

                replicate_index=
                    0,

                pair_index=
                    0,

                stream_name=
                    "trajectory_seed",
            )

        with self.assertRaises(
            ValueError
        ):
            sampling.confirmatory_seed_material(
                evaluation_mode=
                    "deterministic",

                sample_size=
                    16384,

                replicate_index=
                    32,

                pair_index=
                    0,

                stream_name=
                    "trajectory_seed",
            )

        with self.assertRaises(
            ValueError
        ):
            sampling.confirmatory_seed_material(
                evaluation_mode=
                    "deterministic",

                sample_size=
                    16384,

                replicate_index=
                    0,

                pair_index=
                    16384,

                stream_name=
                    "trajectory_seed",
            )

        with self.assertRaises(
            ValueError
        ):
            sampling.confirmatory_seed_material(
                evaluation_mode=
                    "deterministic",

                sample_size=
                    16384,

                replicate_index=
                    0,

                pair_index=
                    0,

                stream_name=
                    "invalid",
            )


    def test_dataset_sampler_wires_independent_seed_streams_into_pair_law(
        self,
    ):
        fake_sample_size = 3

        calls = []

        def fake_pair_sampler(
            *,
            system,
            evaluation_mode,
            horizon_steps,
            dt,
            trajectory_seed,
            time_seed,
            vanderpol_mu,
        ):
            pair_index = len(
                calls
            )

            calls.append(
                {
                    "system":
                        system,

                    "evaluation_mode":
                        evaluation_mode,

                    "horizon_steps":
                        horizon_steps,

                    "dt":
                        dt,

                    "trajectory_seed":
                        trajectory_seed,

                    "time_seed":
                        time_seed,

                    "vanderpol_mu":
                        vanderpol_mu,
                }
            )

            return SimpleNamespace(
                system=
                    system,

                evaluation_mode=
                    evaluation_mode,

                trajectory_seed=
                    trajectory_seed,

                time_seed=
                    time_seed,

                time_index=
                    pair_index,

                horizon_steps=
                    horizon_steps,

                dt=
                    dt,

                current_state=
                    np.array(
                        [
                            float(
                                pair_index
                            ),
                            float(
                                pair_index
                                + 1
                            ),
                        ],
                        dtype=np.float64,
                    ),

                successor_state=
                    np.array(
                        [
                            float(
                                pair_index
                                + 2
                            ),
                            float(
                                pair_index
                                + 3
                            ),
                        ],
                        dtype=np.float64,
                    ),
            )

        with (
            mock.patch.object(
                sampling,
                "CONFIRMATORY_SAMPLE_SIZE",
                fake_sample_size,
            ),
            mock.patch.object(
                sampling,
                "sample_independent_pair",
                side_effect=
                    fake_pair_sampler,
            ),
        ):
            dataset = (
                sampling.sample_confirmatory_dataset(
                    evaluation_mode=
                        "matched_stochastic",

                    replicate_index=
                        2,
                )
            )

        self.assertEqual(
            dataset.sample_size,
            fake_sample_size,
        )

        self.assertEqual(
            dataset.evaluation_mode,
            "matched_stochastic",
        )

        self.assertEqual(
            dataset.replicate_index,
            2,
        )

        self.assertEqual(
            dataset.current_states.shape,
            (
                2,
                fake_sample_size,
            ),
        )

        self.assertEqual(
            dataset.successor_states.shape,
            (
                2,
                fake_sample_size,
            ),
        )

        self.assertEqual(
            dataset.time_indices.tolist(),
            [
                0,
                1,
                2,
            ],
        )

        self.assertEqual(
            len(
                dataset.trajectory_seed_records
            ),
            fake_sample_size,
        )

        self.assertEqual(
            len(
                dataset.time_seed_records
            ),
            fake_sample_size,
        )

        self.assertEqual(
            len(
                calls
            ),
            fake_sample_size,
        )

        for pair_index, call in enumerate(
            calls
        ):
            trajectory_record = (
                dataset.trajectory_seed_records[
                    pair_index
                ]
            )

            time_record = (
                dataset.time_seed_records[
                    pair_index
                ]
            )

            self.assertEqual(
                call[
                    "trajectory_seed"
                ],
                trajectory_record.generated_seed,
            )

            self.assertEqual(
                call[
                    "time_seed"
                ],
                time_record.generated_seed,
            )

            self.assertNotEqual(
                call[
                    "trajectory_seed"
                ],
                call[
                    "time_seed"
                ],
            )

            self.assertEqual(
                trajectory_record.seed_material[
                    0
                ],
                4,
            )

            self.assertEqual(
                time_record.seed_material[
                    0
                ],
                4,
            )

            self.assertEqual(
                trajectory_record.seed_material[
                    5
                ],
                pair_index,
            )

            self.assertEqual(
                time_record.seed_material[
                    5
                ],
                pair_index,
            )

            self.assertEqual(
                trajectory_record.seed_material[
                    6
                ],
                1,
            )

            self.assertEqual(
                time_record.seed_material[
                    6
                ],
                2,
            )


if __name__ == "__main__":
    unittest.main()
