"""Tests for the independent dynamical-pilot numerical verifier.

The tests use synthetic retained arrays and the frozen configuration only.
They do not read or modify the completed dynamical pilot evidence directory,
do not generate dynamical trajectories, and do not fit KAHM representations.
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from kahkm_certificate_statistics import statistics_from_pairs
from kahkm_certificates import certificate_from_statistics
from scripts import verify_dynamical_pilot_numerics as verifier


class DynamicalPilotVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(
            self.temporary.name
        )

        self.config = json.loads(
            (
                verifier.ROOT
                / verifier.CONFIG_PATH
            ).read_text(
                encoding="utf-8"
            )
        )

    def tearDown(self):
        self.temporary.cleanup()

    def job(self):
        return {
            "attempt_id":
                "s1_q2_m3_r004",
            "attempt_number":
                1,
            "system":
                "duffing",
            "system_key":
                1,
            "evaluation_mode":
                "deterministic",
            "mode_key":
                2,
            "sample_size":
                3,
            "replicate_index":
                4,
        }

    def write_json(
        self,
        path,
        payload,
    ):
        path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def make_sample_fixture(
        self,
        *,
        name="attempt",
        metadata_overrides=None,
        corrupt_time_index=False,
    ):
        attempt = (
            self.base
            / name
        )

        attempt.mkdir()

        job = self.job()

        M = job[
            "sample_size"
        ]

        current = np.array(
            [
                [
                    -0.9,
                    0.8,
                    -0.2,
                ],
                [
                    0.0,
                    0.0,
                    0.1,
                ],
            ],
            dtype=np.float64,
        )

        successor = np.array(
            [
                [
                    -0.8,
                    0.7,
                    0.1,
                ],
                [
                    0.1,
                    -0.1,
                    0.2,
                ],
            ],
            dtype=np.float64,
        )

        arrays = {
            "current_states":
                current,
            "successor_states":
                successor,
        }

        horizon = int(
            self.config[
                "pair_sampling"
            ][
                "horizon_steps"
            ]
        )

        time_indices = np.empty(
            M,
            dtype=np.int64,
        )

        for stream_name in (
            "trajectory_seed",
            "time_seed",
        ):
            materials = np.empty(
                (
                    M,
                    8,
                ),
                dtype=np.uint64,
            )

            words = np.empty(
                (
                    M,
                    2,
                ),
                dtype=np.uint32,
            )

            seeds = np.empty(
                M,
                dtype=np.uint64,
            )

            for pair_index in range(
                M
            ):
                material = (
                    verifier.expected_seed_material(
                        config=
                            self.config,
                        job=
                            job,
                        pair_index=
                            pair_index,
                        stream_name=
                            stream_name,
                    )
                )

                generated_words, seed = (
                    verifier.generated_seed_from_material(
                        material
                    )
                )

                materials[
                    pair_index
                ] = np.asarray(
                    material,
                    dtype=np.uint64,
                )

                words[
                    pair_index
                ] = np.asarray(
                    generated_words,
                    dtype=np.uint32,
                )

                seeds[
                    pair_index
                ] = np.uint64(
                    seed
                )

                if (
                    stream_name
                    == "time_seed"
                ):
                    time_indices[
                        pair_index
                    ] = int(
                        np.random.default_rng(
                            seed
                        ).integers(
                            horizon
                        )
                    )

            arrays[
                f"{stream_name}_material"
            ] = materials

            arrays[
                f"{stream_name}_uint32_words"
            ] = words

            arrays[
                f"{stream_name}_generated_seed"
            ] = seeds

        if corrupt_time_index:
            time_indices[
                0
            ] = (
                int(
                    time_indices[
                        0
                    ]
                )
                + 1
            ) % horizon

        arrays[
            "time_indices"
        ] = time_indices

        arrays_path = (
            attempt
            / "sample_arrays.npz"
        )

        np.savez(
            arrays_path,
            **arrays,
        )

        system_config = (
            self.config[
                "frozen_representation"
            ][
                "systems"
            ][
                "duffing"
            ]
        )

        metadata = {
            "sample_evidence_id":
                verifier.EXPECTED_SAMPLE_EVIDENCE_ID,
            "sampling_id":
                verifier.EXPECTED_SAMPLING_ID,
            "pair_law_id":
                verifier.EXPECTED_PAIR_LAW_ID,
            "system":
                "duffing",
            "evaluation_mode":
                "deterministic",
            "sample_size":
                3,
            "replicate_index":
                4,
            "state_dimension":
                2,
            "dt":
                float(
                    system_config[
                        "dt"
                    ]
                ),
            "horizon_steps":
                horizon,
            "vanderpol_mu":
                float(
                    system_config[
                        "vanderpol_mu"
                    ]
                ),
            "array_file":
                "sample_arrays.npz",
            "array_sha256":
                verifier.sha256_file(
                    arrays_path
                ),
            "array_members":
                sorted(
                    verifier.SAMPLE_ARRAY_MEMBERS
                ),
            "seed_record_fields_retained": [
                "generated_seed",
                "generated_uint32_words",
                "seed_material",
            ],
            "evaluation_performed":
                False,
        }

        if metadata_overrides:
            metadata.update(
                metadata_overrides
            )

        self.write_json(
            attempt
            / "sample.json",
            metadata,
        )

        return (
            attempt,
            job,
            metadata,
            arrays,
        )

    def make_evaluation_fixture(
        self,
        *,
        name="evaluation_attempt",
        metadata_overrides=None,
    ):
        (
            attempt,
            job,
            _sample_metadata,
            sample_arrays,
        ) = self.make_sample_fixture(
            name=name,
        )

        phi_columns = np.array(
            [
                [
                    0.8,
                    0.1,
                    0.6,
                ],
                [
                    0.2,
                    0.9,
                    0.4,
                ],
            ],
            dtype=np.float64,
        )

        successor_columns = np.array(
            [
                [
                    0.7,
                    0.2,
                    0.3,
                ],
                [
                    0.3,
                    0.8,
                    0.7,
                ],
            ],
            dtype=np.float64,
        )

        phi_rows = (
            phi_columns.T
        )

        successor_rows = (
            successor_columns.T
        )

        centers = np.array(
            [
                [
                    -1.0,
                    1.0,
                ],
                [
                    0.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        )

        labels, distances = (
            verifier.nearest_center_labels(
                centers,
                sample_arrays[
                    "current_states"
                ],
            )
        )

        statistics = (
            verifier.independent_statistics(
                phi_rows,
                successor_rows,
                labels,
            )
        )

        B = np.array(
            [
                [
                    1.2,
                    0.0,
                ],
                [
                    0.0,
                    0.8,
                ],
            ],
            dtype=np.float64,
        )

        spectral_norm = float(
            np.linalg.norm(
                B,
                ord=2,
            )
        )

        prediction_columns = (
            B.T
            @ phi_columns
        )

        residual_columns = (
            prediction_columns
            - successor_columns
        )

        mse = float(
            np.mean(
                np.sum(
                    residual_columns
                    * residual_columns,
                    axis=0,
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
        )

        rmse = float(
            np.sqrt(
                mse
            )
        )

        arrays = {
            "phi_rows":
                phi_rows,
            "successor_rows":
                successor_rows,
            "labels":
                labels,
            "selected_center_squared_distances":
                distances,
            "class_counts":
                statistics[
                    "class_counts"
                ],
            "class_successor_sse":
                statistics[
                    "class_successor_sse"
                ],
            "class_successor_means":
                statistics[
                    "class_successor_means"
                ],
        }

        arrays_path = (
            attempt
            / "evaluation_arrays.npz"
        )

        np.savez(
            arrays_path,
            **arrays,
        )

        delta = float(
            self.config[
                "certificate"
            ][
                "delta"
            ]
        )

        budgets = []

        for kappa in verifier.norm_budgets(
            self.config,
            spectral_norm,
        ):
            certificate = (
                verifier.independent_certificate(
                    f_hat=
                        statistics[
                            "f_hat"
                        ],
                    s_hat=
                        statistics[
                            "s_hat"
                        ],
                    n_pairs=3,
                    kappa=
                        kappa,
                    delta=
                        delta,
                )
            )

            feasibility_atol = (
                64.0
                * np.finfo(
                    np.float64
                ).eps
                * max(
                    1.0,
                    spectral_norm,
                    kappa,
                )
            )

            budgets.append(
                {
                    "kappa":
                        kappa,
                    "frozen_predictor_within_budget":
                        bool(
                            spectral_norm
                            <= kappa
                            + feasibility_atol
                        ),
                    "frozen_predictor_rmse_minus_bound":
                        float(
                            rmse
                            - certificate[
                                "L_kappa_delta"
                            ]
                        ),
                    "certificate":
                        certificate,
                }
            )

        current = np.asarray(
            sample_arrays[
                "current_states"
            ],
            dtype=np.float64,
        )

        successor = np.asarray(
            sample_arrays[
                "successor_states"
            ],
            dtype=np.float64,
        )

        metadata = {
            "evaluation_evidence_id":
                verifier.EXPECTED_EVALUATION_EVIDENCE_ID,
            "evaluation_id":
                verifier.EXPECTED_EVALUATION_ID,
            "reference_class_map_id":
                verifier.EXPECTED_REFERENCE_MAP_ID,
            "system":
                "duffing",
            "evaluation_mode":
                "deterministic",
            "sample_size":
                3,
            "replicate_index":
                4,
            "n_pairs":
                3,
            "n_classes":
                2,
            "state_dimension":
                2,
            "delta":
                delta,
            "frozen_predictor_spectral_norm":
                spectral_norm,
            "frozen_predictor_mse":
                mse,
            "frozen_predictor_rmse":
                rmse,
            "statistics": {
                "f_hat":
                    statistics[
                        "f_hat"
                    ],
                "s_hat":
                    statistics[
                        "s_hat"
                    ],
                "current_max_row_sum_error":
                    statistics[
                        "current_max_row_sum_error"
                    ],
                "successor_max_row_sum_error":
                    statistics[
                        "successor_max_row_sum_error"
                    ],
                "simplex_sum_atol":
                    1e-12,
            },
            "budget_certificates":
                budgets,
            "coordinate_ranges": {
                "phi_min":
                    float(
                        np.min(
                            phi_columns
                        )
                    ),
                "phi_max":
                    float(
                        np.max(
                            phi_columns
                        )
                    ),
                "successor_min":
                    float(
                        np.min(
                            successor_columns
                        )
                    ),
                "successor_max":
                    float(
                        np.max(
                            successor_columns
                        )
                    ),
            },
            "state_ranges": {
                "current_min":
                    list(
                        np.min(
                            current,
                            axis=1,
                        )
                    ),
                "current_max":
                    list(
                        np.max(
                            current,
                            axis=1,
                        )
                    ),
                "successor_min":
                    list(
                        np.min(
                            successor,
                            axis=1,
                        )
                    ),
                "successor_max":
                    list(
                        np.max(
                            successor,
                            axis=1,
                        )
                    ),
            },
            "array_file":
                "evaluation_arrays.npz",
            "array_sha256":
                verifier.sha256_file(
                    arrays_path
                ),
            "array_members":
                sorted(
                    verifier.EVALUATION_ARRAY_MEMBERS
                ),
            "sample_evidence_present_before_evaluation_write":
                True,
        }

        if metadata_overrides:
            metadata.update(
                metadata_overrides
            )

        self.write_json(
            attempt
            / "evaluation.json",
            metadata,
        )

        model = SimpleNamespace(
            abstraction_model={
                "cluster_centers":
                    centers,
            },
            B=
                B,
            omega=
                2.0,
            tau=
                1e-6,
        )

        return (
            attempt,
            job,
            sample_arrays,
            metadata,
            model,
            phi_columns,
            successor_columns,
        )

    # 1
    def test_frozen_verifier_constants_pin_execution_and_config(self):
        self.assertEqual(
            verifier.EXPECTED_EXECUTION_COMMIT,
            "4cda98e027cdc6b820e37f13869155e6f35956a8",
        )

        raw = (
            verifier.ROOT
            / verifier.CONFIG_PATH
        ).read_bytes()

        self.assertEqual(
            hashlib.sha256(
                raw
            ).hexdigest(),
            verifier.CONFIG_SHA256,
        )

    # 2
    def test_load_config_from_evidence_accepts_exact_committed_bytes(self):
        evidence = (
            self.base
            / "config_evidence"
        )

        evidence.mkdir()

        raw = (
            verifier.ROOT
            / verifier.CONFIG_PATH
        ).read_bytes()

        (
            evidence
            / "campaign_spec.json"
        ).write_bytes(
            raw
        )

        loaded = (
            verifier.load_config_from_evidence(
                evidence
            )
        )

        self.assertEqual(
            loaded,
            self.config,
        )

    # 3
    def test_reconstructed_schedule_is_exact_frozen_design(self):
        jobs = (
            verifier.reconstruct_schedule(
                self.config
            )
        )

        self.assertEqual(
            len(jobs),
            96,
        )

        self.assertEqual(
            len(
                {
                    job[
                        "attempt_id"
                    ]
                    for job
                    in jobs
                }
            ),
            96,
        )

        self.assertEqual(
            sum(
                job[
                    "sample_size"
                ]
                for job
                in jobs
            ),
            86016,
        )

        cells = {
            (
                job[
                    "system"
                ],
                job[
                    "evaluation_mode"
                ],
                job[
                    "sample_size"
                ],
            )
            for job
            in jobs
        }

        self.assertEqual(
            len(cells),
            12,
        )

    # 4
    def test_expected_seed_material_is_the_frozen_eight_component_namespace(self):
        material = (
            verifier.expected_seed_material(
                config=
                    self.config,
                job=
                    self.job(),
                pair_index=
                    2,
                stream_name=
                    "time_seed",
            )
        )

        self.assertEqual(
            material,
            (
                3,
                1,
                2,
                3,
                4,
                2,
                2,
                20261003,
            ),
        )

    # 5
    def test_generated_seed_matches_direct_seedsequence_recipe(self):
        material = (
            3,
            2,
            1,
            2048,
            7,
            123,
            1,
            20261003,
        )

        words, seed = (
            verifier.generated_seed_from_material(
                material
            )
        )

        sequence = np.random.SeedSequence(
            entropy=
                material
        )

        direct = sequence.generate_state(
            2,
            dtype=np.uint32,
        )

        expected = (
            int(
                direct[
                    0
                ]
            )
            | (
                int(
                    direct[
                        1
                    ]
                )
                << 32
            )
        )

        self.assertEqual(
            words,
            (
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
            ),
        )

        self.assertEqual(
            seed,
            expected,
        )

    # 6
    def test_verifier_reconstructs_unique_full_frozen_seed_namespace(self):
        materials = set()
        seeds = set()
        count = 0

        for job in verifier.reconstruct_schedule(
            self.config
        ):
            for pair_index in range(
                job[
                    "sample_size"
                ]
            ):
                for stream_name in (
                    "trajectory_seed",
                    "time_seed",
                ):
                    material = (
                        verifier.expected_seed_material(
                            config=
                                self.config,
                            job=
                                job,
                            pair_index=
                                pair_index,
                            stream_name=
                                stream_name,
                        )
                    )

                    _, seed = (
                        verifier.generated_seed_from_material(
                            material
                        )
                    )

                    self.assertNotIn(
                        material,
                        materials,
                    )

                    self.assertNotIn(
                        seed,
                        seeds,
                    )

                    self.assertNotIn(
                        seed,
                        (
                            0,
                            1,
                            2,
                        ),
                    )

                    materials.add(
                        material
                    )

                    seeds.add(
                        seed
                    )

                    count += 1

        self.assertEqual(
            count,
            172032,
        )

        self.assertEqual(
            len(materials),
            172032,
        )

        self.assertEqual(
            len(seeds),
            172032,
        )

    # 7
    def test_norm_budgets_match_frozen_recipe(self):
        actual = (
            verifier.norm_budgets(
                self.config,
                1.25,
            )
        )

        self.assertEqual(
            actual,
            tuple(
                sorted(
                    {
                        1.0,
                        float(
                            np.sqrt(
                                2.0
                            )
                        ),
                        1.25,
                        2.5,
                    }
                )
            ),
        )

    # 8
    def test_independent_certificate_matches_validated_implementation(self):
        cases = (
            (
                0.0,
                0.0,
                128,
                1.0,
                0.05,
            ),
            (
                0.04,
                0.20,
                512,
                1.25,
                0.05,
            ),
            (
                0.13,
                0.47,
                2048,
                2.5,
                0.05,
            ),
        )

        for (
            f_hat,
            s_hat,
            M,
            kappa,
            delta,
        ) in cases:
            with self.subTest(
                case=(
                    f_hat,
                    s_hat,
                    M,
                    kappa,
                    delta,
                )
            ):
                independent = (
                    verifier.independent_certificate(
                        f_hat=
                            f_hat,
                        s_hat=
                            s_hat,
                        n_pairs=
                            M,
                        kappa=
                            kappa,
                        delta=
                            delta,
                    )
                )

                production = (
                    certificate_from_statistics(
                        f_hat=
                            f_hat,
                        s_hat=
                            s_hat,
                        n_pairs=
                            M,
                        kappa=
                            kappa,
                        delta=
                            delta,
                    )
                )

                for key in (
                    "r_delta",
                    "F_delta",
                    "V_delta",
                    "L_kappa_delta",
                ):
                    self.assertEqual(
                        independent[
                            key
                        ],
                        getattr(
                            production,
                            key,
                        ),
                    )

    # 9
    def test_independent_statistics_match_validated_implementation_with_empty_class(self):
        phi = np.array(
            [
                [
                    0.8,
                    0.2,
                    0.0,
                ],
                [
                    0.1,
                    0.9,
                    0.0,
                ],
                [
                    0.6,
                    0.4,
                    0.0,
                ],
            ],
            dtype=np.float64,
        )

        successors = np.array(
            [
                [
                    0.7,
                    0.3,
                    0.0,
                ],
                [
                    0.2,
                    0.8,
                    0.0,
                ],
                [
                    0.3,
                    0.7,
                    0.0,
                ],
            ],
            dtype=np.float64,
        )

        labels = np.array(
            [
                0,
                1,
                0,
            ],
            dtype=np.int64,
        )

        independent = (
            verifier.independent_statistics(
                phi,
                successors,
                labels,
            )
        )

        production = (
            statistics_from_pairs(
                phi=
                    phi,
                successors=
                    successors,
                labels=
                    labels,
            )
        )

        self.assertEqual(
            independent[
                "f_hat"
            ],
            production.f_hat,
        )

        self.assertEqual(
            independent[
                "s_hat"
            ],
            production.s_hat,
        )

        np.testing.assert_array_equal(
            independent[
                "class_counts"
            ],
            np.asarray(
                production.class_counts
            ),
        )

        np.testing.assert_array_equal(
            independent[
                "class_successor_means"
            ],
            np.asarray(
                production.class_successor_means
            ),
        )

        np.testing.assert_array_equal(
            independent[
                "class_successor_sse"
            ],
            np.asarray(
                production.class_successor_sse
            ),
        )

    # 10
    def test_nearest_center_labels_use_euclidean_distance_and_smallest_index_tie(self):
        centers = np.array(
            [
                [
                    -1.0,
                    1.0,
                ],
                [
                    0.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        )

        states = np.array(
            [
                [
                    -1.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        )

        labels, distances = (
            verifier.nearest_center_labels(
                centers,
                states,
            )
        )

        np.testing.assert_array_equal(
            labels,
            np.array(
                [
                    0,
                    1,
                    0,
                ],
                dtype=np.int64,
            ),
        )

        np.testing.assert_array_equal(
            distances,
            np.array(
                [
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
        )

    # 11
    def test_safe_relative_path_rejects_traversal_absolute_and_backslash(self):
        self.assertEqual(
            verifier.safe_relative_path(
                "attempt_data/a/sample.json"
            ),
            Path(
                "attempt_data/a/sample.json"
            ),
        )

        for value in (
            "../escape",
            "/absolute/path",
            "a\\b",
            "",
        ):
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    ValueError
                ):
                    verifier.safe_relative_path(
                        value
                    )

    # 12
    def test_checksum_inventory_verifies_exact_files_and_detects_tampering(self):
        folder = (
            self.base
            / "inventory"
        )

        folder.mkdir()

        (
            folder
            / "a.txt"
        ).write_bytes(
            b"a"
        )

        (
            folder
            / "b.txt"
        ).write_bytes(
            b"b"
        )

        manifest = (
            folder
            / "SHA256SUMS"
        )

        manifest.write_text(
            (
                f"{verifier.sha256_file(folder / 'a.txt')}  a.txt\n"
                f"{verifier.sha256_file(folder / 'b.txt')}  b.txt\n"
            ),
            encoding="utf-8",
        )

        with mock.patch.object(
            verifier,
            "EXPECTED_EVIDENCE_INVENTORY_COUNT",
            2,
        ):
            result = verifier.check_inventory(
                folder
            )

            self.assertEqual(
                result[
                    "verified_file_count"
                ],
                2,
            )

            (
                folder
                / "a.txt"
            ).write_bytes(
                b"tampered"
            )

            with self.assertRaisesRegex(
                ValueError,
                "SHA256 mismatch",
            ):
                verifier.check_inventory(
                    folder
                )

    # 13
    def test_jsonl_reader_requires_newline_terminated_objects(self):
        path = (
            self.base
            / "records.jsonl"
        )

        path.write_text(
            '{"x":1}\n',
            encoding="utf-8",
        )

        self.assertEqual(
            verifier.jsonl_records(
                path
            ),
            [
                {
                    "x":
                        1,
                }
            ],
        )

        path.write_text(
            '{"x":1}',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "newline",
        ):
            verifier.jsonl_records(
                path
            )

    # 14
    def test_npz_loader_rejects_unexpected_member_inventory(self):
        path = (
            self.base
            / "arrays.npz"
        )

        np.savez(
            path,
            a=
                np.array(
                    [
                        1.0,
                    ]
                ),
            b=
                np.array(
                    [
                        2.0,
                    ]
                ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "inventory",
        ):
            verifier.load_npz(
                path,
                {
                    "a",
                },
            )

    # 15
    def test_attempt_event_log_accepts_one_complete_started_completed_pair(self):
        evidence = (
            self.base
            / "events"
        )

        evidence.mkdir()

        (
            evidence
            / "attempts.jsonl"
        ).write_text(
            (
                '{"attempt_id":"a","event":"started","utc":"x"}\n'
                '{"attempt_id":"a","event":"completed",'
                '"sampling_seconds":1.0,"evaluation_seconds":2.0,"utc":"y"}\n'
            ),
            encoding="utf-8",
        )

        schedule = [
            {
                "attempt_id":
                    "a",
            }
        ]

        with mock.patch.object(
            verifier,
            "EXPECTED_ATTEMPTS",
            1,
        ):
            verifier.verify_attempt_event_log(
                evidence,
                schedule,
            )

    # 16
    def test_attempt_event_log_rejects_failed_or_out_of_order_terminal_event(self):
        evidence = (
            self.base
            / "bad_events"
        )

        evidence.mkdir()

        (
            evidence
            / "attempts.jsonl"
        ).write_text(
            (
                '{"attempt_id":"a","event":"started","utc":"x"}\n'
                '{"attempt_id":"a","event":"failed","stage":"evaluation","utc":"y"}\n'
            ),
            encoding="utf-8",
        )

        with mock.patch.object(
            verifier,
            "EXPECTED_ATTEMPTS",
            1,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "completed",
            ):
                verifier.verify_attempt_event_log(
                    evidence,
                    [
                        {
                            "attempt_id":
                                "a",
                        }
                    ],
                )

    # 17
    def test_sample_verification_reconstructs_all_seed_and_time_draws(self):
        (
            attempt,
            job,
            metadata,
            _arrays,
        ) = self.make_sample_fixture()

        materials = set()
        seeds = set()

        (
            verified_metadata,
            verified_arrays,
            time_draws,
        ) = verifier.verify_sample(
            attempt_dir=
                attempt,
            job=
                job,
            config=
                self.config,
            global_materials=
                materials,
            global_generated_seeds=
                seeds,
        )

        self.assertEqual(
            verified_metadata,
            metadata,
        )

        self.assertEqual(
            time_draws,
            3,
        )

        self.assertEqual(
            len(materials),
            6,
        )

        self.assertEqual(
            len(seeds),
            6,
        )

        self.assertEqual(
            verified_arrays[
                "current_states"
            ].shape,
            (
                2,
                3,
            ),
        )

    # 18
    def test_sample_verification_rejects_tampered_time_index_draw(self):
        (
            attempt,
            job,
            _metadata,
            _arrays,
        ) = self.make_sample_fixture(
            name=
                "bad_time",
            corrupt_time_index=
                True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "time-index draw mismatch",
        ):
            verifier.verify_sample(
                attempt_dir=
                    attempt,
                job=
                    job,
                config=
                    self.config,
                global_materials=
                    set(),
                global_generated_seeds=
                    set(),
            )

    # 19
    def test_sample_verification_enforces_physical_design_metadata(self):
        cases = (
            (
                {
                    "dt":
                        0.123,
                },
                "dt",
            ),
            (
                {
                    "horizon_steps":
                        17,
                },
                "horizon",
            ),
            (
                {
                    "state_dimension":
                        9,
                },
                "state",
            ),
            (
                {
                    "vanderpol_mu":
                        7.0,
                },
                "vanderpol",
            ),
        )

        for index, (
            override,
            pattern,
        ) in enumerate(
            cases
        ):
            with self.subTest(
                override=override
            ):
                (
                    attempt,
                    job,
                    _metadata,
                    _arrays,
                ) = self.make_sample_fixture(
                    name=
                        f"metadata_{index}",
                    metadata_overrides=
                        override,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    pattern,
                ):
                    verifier.verify_sample(
                        attempt_dir=
                            attempt,
                        job=
                            job,
                        config=
                            self.config,
                        global_materials=
                            set(),
                        global_generated_seeds=
                            set(),
                    )

    # 20
    def test_evaluation_verification_independently_recomputes_numerics(self):
        (
            attempt,
            job,
            sample_arrays,
            metadata,
            model,
            phi_columns,
            successor_columns,
        ) = self.make_evaluation_fixture()

        with mock.patch.object(
            verifier,
            "kahm_associations",
            side_effect=[
                phi_columns,
                successor_columns,
            ],
        ) as association_mock:
            (
                verified_metadata,
                verified_arrays,
            ) = verifier.verify_evaluation(
                attempt_dir=
                    attempt,
                job=
                    job,
                sample_arrays=
                    sample_arrays,
                config=
                    self.config,
                model=
                    model,
            )

        self.assertEqual(
            association_mock.call_count,
            2,
        )

        self.assertEqual(
            verified_metadata,
            metadata,
        )

        np.testing.assert_array_equal(
            verified_arrays[
                "phi_rows"
            ],
            phi_columns.T,
        )

    # 21
    def test_direct_script_launch_can_import_repository_modules(self):
        script = (
            verifier.ROOT
            / "scripts"
            / "verify_dynamical_pilot_numerics.py"
        )

        environment = dict(
            os.environ
        )

        # Make the regression independent of any user PYTHONPATH that could
        # accidentally mask the direct-script import problem.
        environment.pop(
            "PYTHONPATH",
            None,
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(
                    script
                ),
                "--help",
            ],
            cwd=
                verifier.ROOT,
            env=
                environment,
            stdin=
                subprocess.DEVNULL,
            stdout=
                subprocess.PIPE,
            stderr=
                subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                "Direct verifier launch failed.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            ),
        )

        self.assertIn(
            "--evidence",
            completed.stdout,
        )

        self.assertNotIn(
            "ModuleNotFoundError",
            completed.stderr,
        )

    # 22
    def test_evaluation_verification_enforces_evidence_order_and_dimensions(self):
        cases = (
            (
                {
                    "sample_evidence_present_before_evaluation_write":
                        False,
                },
                "sample",
            ),
            (
                {
                    "n_pairs":
                        999,
                },
                "pair",
            ),
            (
                {
                    "n_classes":
                        999,
                },
                "class",
            ),
            (
                {
                    "state_dimension":
                        999,
                },
                "state",
            ),
            (
                {
                    "delta":
                        0.25,
                },
                "delta",
            ),
        )

        for index, (
            override,
            pattern,
        ) in enumerate(
            cases
        ):
            with self.subTest(
                override=override
            ):
                (
                    attempt,
                    job,
                    sample_arrays,
                    _metadata,
                    model,
                    phi_columns,
                    successor_columns,
                ) = self.make_evaluation_fixture(
                    name=
                        f"evaluation_metadata_{index}",
                    metadata_overrides=
                        override,
                )

                with mock.patch.object(
                    verifier,
                    "kahm_associations",
                    side_effect=[
                        phi_columns,
                        successor_columns,
                    ],
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        pattern,
                    ):
                        verifier.verify_evaluation(
                            attempt_dir=
                                attempt,
                            job=
                                job,
                            sample_arrays=
                                sample_arrays,
                            config=
                                self.config,
                            model=
                                model,
                        )


if __name__ == "__main__":
    unittest.main()
