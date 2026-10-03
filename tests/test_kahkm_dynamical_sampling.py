"""Tests for deterministic dynamical certificate-pilot sampling.

The tests validate the frozen seed namespace, NumPy seed derivation,
reproducibility, independence from global RNG state, and no-retry semantics.
No KAHM representation is fitted and no certificate is evaluated.
"""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import kahkm_dynamical_sampling as module
from kahkm_dynamical_pairs import (
    PAIR_LAW_ID,
)
from kahkm_dynamical_sampling import (
    BIT_GENERATOR_ID,
    CAMPAIGN_KEY,
    DYNAMICAL_SAMPLING_ID,
    MODE_KEYS,
    ROOT_SEED,
    SEED_SEQUENCE_ID,
    STREAM_KEYS,
    SYSTEM_KEYS,
    TRAINING_LITERAL_SEEDS,
    SampledDynamicalDataset,
    SeedRecord,
    sample_dynamical_dataset,
    seed_record,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


class DynamicalSamplingTests(unittest.TestCase):
    # 1
    def test_frozen_sampling_constants_match_configuration(self):
        config = json.loads(
            (
                PROJECT_ROOT
                / "reproduction"
                / "certificates"
                / "configs"
                / "dynamical_pilot_v1.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        scheme = config[
            "seed_scheme"
        ]

        self.assertEqual(
            DYNAMICAL_SAMPLING_ID,
            "dynamical_certificate_sampling_v1",
        )

        self.assertEqual(
            BIT_GENERATOR_ID,
            scheme[
                "bit_generator"
            ],
        )

        self.assertEqual(
            SEED_SEQUENCE_ID,
            scheme[
                "seed_sequence"
            ],
        )

        self.assertEqual(
            ROOT_SEED,
            scheme[
                "root_seed"
            ],
        )

        self.assertEqual(
            CAMPAIGN_KEY,
            scheme[
                "campaign_key"
            ],
        )

        self.assertEqual(
            SYSTEM_KEYS,
            scheme[
                "system_keys"
            ],
        )

        self.assertEqual(
            MODE_KEYS,
            scheme[
                "mode_keys"
            ],
        )

        self.assertEqual(
            STREAM_KEYS,
            scheme[
                "stream_keys"
            ],
        )

        self.assertEqual(
            list(
                TRAINING_LITERAL_SEEDS
            ),
            scheme[
                "training_seed_namespace"
            ],
        )

    # 2
    def test_seed_record_matches_direct_numpy_recipe(self):
        record = seed_record(
            system="duffing",
            evaluation_mode="matched_stochastic",
            sample_size=128,
            replicate_index=3,
            pair_index=17,
            stream_name="trajectory_seed",
        )

        expected_material = (
            3,
            1,
            1,
            128,
            3,
            17,
            1,
            20261003,
        )

        sequence = np.random.SeedSequence(
            entropy=expected_material
        )

        words = sequence.generate_state(
            2,
            dtype=np.uint32,
        )

        expected_seed = (
            int(
                words[0]
            )
            | (
                int(
                    words[1]
                )
                << 32
            )
        )

        self.assertEqual(
            record.seed_material,
            expected_material,
        )

        self.assertEqual(
            record.generated_uint32_words,
            (
                int(
                    words[0]
                ),
                int(
                    words[1]
                ),
            ),
        )

        self.assertEqual(
            record.generated_seed,
            expected_seed,
        )

    # 3
    def test_seed_record_metadata_is_complete(self):
        record = seed_record(
            system="vanderpol",
            evaluation_mode="deterministic",
            sample_size=512,
            replicate_index=7,
            pair_index=511,
            stream_name="time_seed",
        )

        self.assertEqual(
            record.sampling_id,
            DYNAMICAL_SAMPLING_ID,
        )

        self.assertEqual(
            record.bit_generator,
            "PCG64",
        )

        self.assertEqual(
            record.seed_sequence,
            "numpy.random.SeedSequence",
        )

        self.assertEqual(
            record.campaign_key,
            3,
        )

        self.assertEqual(
            record.system_key,
            2,
        )

        self.assertEqual(
            record.mode_key,
            2,
        )

        self.assertEqual(
            record.stream_key,
            2,
        )

        self.assertEqual(
            record.root_seed,
            20261003,
        )

    # 4
    def test_each_frozen_seed_component_changes_material(self):
        base = seed_record(
            system="duffing",
            evaluation_mode="matched_stochastic",
            sample_size=128,
            replicate_index=0,
            pair_index=0,
            stream_name="trajectory_seed",
        )

        variants = (
            seed_record(
                system="vanderpol",
                evaluation_mode="matched_stochastic",
                sample_size=128,
                replicate_index=0,
                pair_index=0,
                stream_name="trajectory_seed",
            ),
            seed_record(
                system="duffing",
                evaluation_mode="deterministic",
                sample_size=128,
                replicate_index=0,
                pair_index=0,
                stream_name="trajectory_seed",
            ),
            seed_record(
                system="duffing",
                evaluation_mode="matched_stochastic",
                sample_size=512,
                replicate_index=0,
                pair_index=0,
                stream_name="trajectory_seed",
            ),
            seed_record(
                system="duffing",
                evaluation_mode="matched_stochastic",
                sample_size=128,
                replicate_index=1,
                pair_index=0,
                stream_name="trajectory_seed",
            ),
            seed_record(
                system="duffing",
                evaluation_mode="matched_stochastic",
                sample_size=128,
                replicate_index=0,
                pair_index=1,
                stream_name="trajectory_seed",
            ),
            seed_record(
                system="duffing",
                evaluation_mode="matched_stochastic",
                sample_size=128,
                replicate_index=0,
                pair_index=0,
                stream_name="time_seed",
            ),
        )

        for variant in variants:
            with self.subTest(
                variant=
                    variant.seed_material
            ):
                self.assertNotEqual(
                    variant.seed_material,
                    base.seed_material,
                )

                self.assertNotEqual(
                    variant.generated_seed,
                    base.generated_seed,
                )

    # 5
    def test_seed_derivation_is_repeatable_and_order_independent(self):
        kwargs = {
            "system":
                "vanderpol",
            "evaluation_mode":
                "deterministic",
            "sample_size":
                2048,
            "replicate_index":
                4,
            "pair_index":
                1234,
            "stream_name":
                "time_seed",
        }

        first = seed_record(
            **kwargs
        )

        # Unrelated derivations must not affect a later result.
        for index in range(50):
            seed_record(
                system="duffing",
                evaluation_mode="matched_stochastic",
                sample_size=128,
                replicate_index=index,
                pair_index=0,
                stream_name="trajectory_seed",
            )

        second = seed_record(
            **kwargs
        )

        self.assertEqual(
            first,
            second,
        )

    # 6
    def test_full_pilot_seed_material_and_generated_seeds_are_unique(self):
        sample_sizes = (
            128,
            512,
            2048,
        )

        materials = set()
        generated = set()

        stream_count = 0

        for system in (
            "duffing",
            "vanderpol",
        ):
            for mode in (
                "matched_stochastic",
                "deterministic",
            ):
                for sample_size in sample_sizes:
                    for replicate_index in range(
                        8
                    ):
                        for pair_index in range(
                            sample_size
                        ):
                            for stream_name in (
                                "trajectory_seed",
                                "time_seed",
                            ):
                                record = seed_record(
                                    system=
                                        system,
                                    evaluation_mode=
                                        mode,
                                    sample_size=
                                        sample_size,
                                    replicate_index=
                                        replicate_index,
                                    pair_index=
                                        pair_index,
                                    stream_name=
                                        stream_name,
                                )

                                self.assertNotIn(
                                    record.seed_material,
                                    materials,
                                )

                                self.assertNotIn(
                                    record.generated_seed,
                                    generated,
                                )

                                self.assertNotIn(
                                    record.generated_seed,
                                    TRAINING_LITERAL_SEEDS,
                                )

                                materials.add(
                                    record.seed_material
                                )

                                generated.add(
                                    record.generated_seed
                                )

                                stream_count += 1

        self.assertEqual(
            stream_count,
            172032,
        )

        self.assertEqual(
            len(materials),
            172032,
        )

        self.assertEqual(
            len(generated),
            172032,
        )

    # 7
    def test_generated_seed_collision_with_literal_training_seed_aborts(self):
        fake_sequence = mock.Mock()

        fake_sequence.generate_state.return_value = np.array(
            [
                1,
                0,
            ],
            dtype=np.uint32,
        )

        with mock.patch.object(
            module.np.random,
            "SeedSequence",
            return_value=
                fake_sequence,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "collided",
            ):
                seed_record(
                    system="duffing",
                    evaluation_mode="deterministic",
                    sample_size=128,
                    replicate_index=0,
                    pair_index=0,
                    stream_name="trajectory_seed",
                )

    # 8
    def test_invalid_seed_record_identifiers_are_rejected(self):
        cases = (
            (
                {
                    "system":
                        "lorenz",
                    "evaluation_mode":
                        "deterministic",
                    "sample_size":
                        128,
                    "replicate_index":
                        0,
                    "pair_index":
                        0,
                    "stream_name":
                        "trajectory_seed",
                },
                ValueError,
            ),
            (
                {
                    "system":
                        "duffing",
                    "evaluation_mode":
                        "wrong",
                    "sample_size":
                        128,
                    "replicate_index":
                        0,
                    "pair_index":
                        0,
                    "stream_name":
                        "trajectory_seed",
                },
                ValueError,
            ),
            (
                {
                    "system":
                        "duffing",
                    "evaluation_mode":
                        "deterministic",
                    "sample_size":
                        128,
                    "replicate_index":
                        0,
                    "pair_index":
                        0,
                    "stream_name":
                        "wrong",
                },
                ValueError,
            ),
        )

        for kwargs, expected in cases:
            with self.subTest(
                kwargs=kwargs
            ):
                with self.assertRaises(
                    expected
                ):
                    seed_record(
                        **kwargs
                    )

    # 9
    def test_invalid_indices_and_sample_sizes_are_rejected(self):
        base = {
            "system":
                "duffing",
            "evaluation_mode":
                "deterministic",
            "sample_size":
                128,
            "replicate_index":
                0,
            "pair_index":
                0,
            "stream_name":
                "trajectory_seed",
        }

        cases = (
            (
                "sample_size",
                0,
                ValueError,
            ),
            (
                "sample_size",
                True,
                TypeError,
            ),
            (
                "replicate_index",
                -1,
                ValueError,
            ),
            (
                "replicate_index",
                1.5,
                TypeError,
            ),
            (
                "pair_index",
                -1,
                ValueError,
            ),
            (
                "pair_index",
                128,
                ValueError,
            ),
        )

        for key, value, expected in cases:
            kwargs = dict(
                base
            )
            kwargs[key] = value

            with self.subTest(
                key=key,
                value=value,
            ):
                with self.assertRaises(
                    expected
                ):
                    seed_record(
                        **kwargs
                    )

    # 10
    def test_small_real_dataset_is_repeatable(self):
        kwargs = {
            "system":
                "duffing",
            "evaluation_mode":
                "deterministic",
            "sample_size":
                5,
            "replicate_index":
                2,
            "horizon_steps":
                20,
            "dt":
                0.03,
            "vanderpol_mu":
                1.0,
        }

        first = sample_dynamical_dataset(
            **kwargs
        )

        second = sample_dynamical_dataset(
            **kwargs
        )

        np.testing.assert_array_equal(
            first.current_states,
            second.current_states,
        )

        np.testing.assert_array_equal(
            first.successor_states,
            second.successor_states,
        )

        self.assertEqual(
            first.time_indices,
            second.time_indices,
        )

        self.assertEqual(
            first.trajectory_seed_records,
            second.trajectory_seed_records,
        )

        self.assertEqual(
            first.time_seed_records,
            second.time_seed_records,
        )

    # 11
    def test_small_real_matched_stochastic_vanderpol_dataset(self):
        result = sample_dynamical_dataset(
            system="vanderpol",
            evaluation_mode="matched_stochastic",
            sample_size=4,
            replicate_index=1,
            horizon_steps=12,
            dt=0.02,
            vanderpol_mu=1.0,
        )

        self.assertEqual(
            result.current_states.shape,
            (
                2,
                4,
            ),
        )

        self.assertEqual(
            result.successor_states.shape,
            (
                2,
                4,
            ),
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    result.current_states
                )
            )
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    result.successor_states
                )
            )
        )

        self.assertEqual(
            len(
                result.time_indices
            ),
            4,
        )

        self.assertTrue(
            all(
                0 <= value < 12
                for value
                in result.time_indices
            )
        )

    # 12
    def test_dataset_metadata_matches_request(self):
        result = sample_dynamical_dataset(
            system="duffing",
            evaluation_mode="deterministic",
            sample_size=3,
            replicate_index=7,
            horizon_steps=9,
            dt=0.03,
            vanderpol_mu=1.0,
        )

        self.assertEqual(
            result.sampling_id,
            DYNAMICAL_SAMPLING_ID,
        )

        self.assertEqual(
            result.pair_law_id,
            PAIR_LAW_ID,
        )

        self.assertEqual(
            result.system,
            "duffing",
        )

        self.assertEqual(
            result.evaluation_mode,
            "deterministic",
        )

        self.assertEqual(
            result.sample_size,
            3,
        )

        self.assertEqual(
            result.replicate_index,
            7,
        )

        self.assertEqual(
            result.dt,
            0.03,
        )

        self.assertEqual(
            result.horizon_steps,
            9,
        )

    # 13
    def test_dataset_passes_exact_derived_seeds_to_pair_sampler(self):
        calls = []

        def fake_pair(**kwargs):
            calls.append(
                dict(
                    kwargs
                )
            )

            return SimpleNamespace(
                pair_law_id=
                    PAIR_LAW_ID,
                trajectory_seed=
                    kwargs[
                        "trajectory_seed"
                    ],
                time_seed=
                    kwargs[
                        "time_seed"
                    ],
                time_index=
                    len(calls) - 1,
                current_state=(
                    float(
                        len(calls)
                    ),
                    0.0,
                ),
                successor_state=(
                    float(
                        len(calls)
                    ),
                    1.0,
                ),
            )

        with mock.patch.object(
            module,
            "sample_independent_pair",
            side_effect=fake_pair,
        ):
            result = sample_dynamical_dataset(
                system="duffing",
                evaluation_mode="deterministic",
                sample_size=3,
                replicate_index=2,
                horizon_steps=10,
                dt=0.03,
            )

        self.assertEqual(
            len(calls),
            3,
        )

        for pair_index, call in enumerate(
            calls
        ):
            self.assertEqual(
                call[
                    "trajectory_seed"
                ],
                result.trajectory_seed_records[
                    pair_index
                ].generated_seed,
            )

            self.assertEqual(
                call[
                    "time_seed"
                ],
                result.time_seed_records[
                    pair_index
                ].generated_seed,
            )

            self.assertEqual(
                call[
                    "system"
                ],
                "duffing",
            )

            self.assertEqual(
                call[
                    "evaluation_mode"
                ],
                "deterministic",
            )

    # 14
    def test_pair_sampler_failure_propagates_without_retry(self):
        with mock.patch.object(
            module,
            "sample_independent_pair",
            side_effect=RuntimeError(
                "synthetic pair failure"
            ),
        ) as pair_mock:
            with self.assertRaisesRegex(
                RuntimeError,
                "synthetic pair failure",
            ):
                sample_dynamical_dataset(
                    system="duffing",
                    evaluation_mode="deterministic",
                    sample_size=5,
                    replicate_index=0,
                    horizon_steps=10,
                    dt=0.03,
                )

        self.assertEqual(
            pair_mock.call_count,
            1,
        )

    # 15
    def test_pair_law_identifier_mismatch_is_rejected(self):
        fake = SimpleNamespace(
            pair_law_id=
                "wrong",
            trajectory_seed=1,
            time_seed=2,
            time_index=0,
            current_state=(
                0.0,
                0.0,
            ),
            successor_state=(
                0.0,
                0.0,
            ),
        )

        with mock.patch.object(
            module,
            "sample_independent_pair",
            return_value=fake,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "identifier mismatch",
            ):
                sample_dynamical_dataset(
                    system="duffing",
                    evaluation_mode="deterministic",
                    sample_size=1,
                    replicate_index=0,
                    horizon_steps=10,
                    dt=0.03,
                )

    # 16
    def test_pair_sampler_seed_mutation_is_rejected(self):
        def fake_pair(**kwargs):
            return SimpleNamespace(
                pair_law_id=
                    PAIR_LAW_ID,
                trajectory_seed=
                    kwargs[
                        "trajectory_seed"
                    ]
                    + 1,
                time_seed=
                    kwargs[
                        "time_seed"
                    ],
                time_index=0,
                current_state=(
                    0.0,
                    0.0,
                ),
                successor_state=(
                    0.0,
                    0.0,
                ),
            )

        with mock.patch.object(
            module,
            "sample_independent_pair",
            side_effect=fake_pair,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "preserve derived seeds",
            ):
                sample_dynamical_dataset(
                    system="duffing",
                    evaluation_mode="deterministic",
                    sample_size=1,
                    replicate_index=0,
                    horizon_steps=10,
                    dt=0.03,
                )

    # 17
    def test_legacy_global_rng_is_not_advanced(self):
        np.random.seed(
            12345
        )

        expected = float(
            np.random.random()
        )

        np.random.seed(
            12345
        )

        sample_dynamical_dataset(
            system="duffing",
            evaluation_mode="deterministic",
            sample_size=3,
            replicate_index=0,
            horizon_steps=8,
            dt=0.03,
        )

        actual = float(
            np.random.random()
        )

        self.assertEqual(
            actual,
            expected,
        )

    # 18
    def test_legacy_global_seed_does_not_change_dataset(self):
        kwargs = {
            "system":
                "duffing",
            "evaluation_mode":
                "matched_stochastic",
            "sample_size":
                3,
            "replicate_index":
                0,
            "horizon_steps":
                8,
            "dt":
                0.03,
        }

        np.random.seed(
            1
        )

        first = sample_dynamical_dataset(
            **kwargs
        )

        np.random.seed(
            999999
        )

        second = sample_dynamical_dataset(
            **kwargs
        )

        np.testing.assert_array_equal(
            first.current_states,
            second.current_states,
        )

        np.testing.assert_array_equal(
            first.successor_states,
            second.successor_states,
        )

    # 19
    def test_invalid_dataset_parameters_are_rejected_before_pair_sampling(self):
        cases = (
            (
                {
                    "system":
                        "wrong",
                    "evaluation_mode":
                        "deterministic",
                    "sample_size":
                        1,
                    "replicate_index":
                        0,
                    "horizon_steps":
                        10,
                    "dt":
                        0.03,
                },
                ValueError,
            ),
            (
                {
                    "system":
                        "duffing",
                    "evaluation_mode":
                        "deterministic",
                    "sample_size":
                        0,
                    "replicate_index":
                        0,
                    "horizon_steps":
                        10,
                    "dt":
                        0.03,
                },
                ValueError,
            ),
            (
                {
                    "system":
                        "duffing",
                    "evaluation_mode":
                        "deterministic",
                    "sample_size":
                        1,
                    "replicate_index":
                        0,
                    "horizon_steps":
                        0,
                    "dt":
                        0.03,
                },
                ValueError,
            ),
            (
                {
                    "system":
                        "duffing",
                    "evaluation_mode":
                        "deterministic",
                    "sample_size":
                        1,
                    "replicate_index":
                        0,
                    "horizon_steps":
                        10,
                    "dt":
                        0.0,
                },
                ValueError,
            ),
        )

        for kwargs, expected in cases:
            with (
                self.subTest(
                    kwargs=kwargs
                ),
                mock.patch.object(
                    module,
                    "sample_independent_pair",
                ) as pair_mock,
            ):
                with self.assertRaises(
                    expected
                ):
                    sample_dynamical_dataset(
                        **kwargs
                    )

                pair_mock.assert_not_called()

    # 20
    def test_records_and_dataset_dataclasses_are_frozen(self):
        record = seed_record(
            system="duffing",
            evaluation_mode="deterministic",
            sample_size=1,
            replicate_index=0,
            pair_index=0,
            stream_name="trajectory_seed",
        )

        self.assertIsInstance(
            record,
            SeedRecord,
        )

        with self.assertRaises(
            FrozenInstanceError
        ):
            record.generated_seed = 7

        dataset = sample_dynamical_dataset(
            system="duffing",
            evaluation_mode="deterministic",
            sample_size=1,
            replicate_index=0,
            horizon_steps=2,
            dt=0.03,
        )

        self.assertIsInstance(
            dataset,
            SampledDynamicalDataset,
        )

        with self.assertRaises(
            FrozenInstanceError
        ):
            dataset.sample_size = 2


if __name__ == "__main__":
    unittest.main()
