"""Tests for lossless dynamical certificate-pilot evidence serialization.

No KAHM is fitted and no random dynamical pair is generated.
"""

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

import kahkm_dynamical_evidence as module
from kahkm_certificate_statistics import statistics_from_pairs
from kahkm_certificates import certificate_from_statistics
from kahkm_dynamical_evaluation import (
    DYNAMICAL_EVALUATION_ID,
    BudgetCertificate,
    DynamicalEvaluationResult,
)
from kahkm_dynamical_evidence import (
    EVALUATION_ARRAY_FILENAME,
    EVALUATION_EVIDENCE_ID,
    EVALUATION_METADATA_FILENAME,
    SAMPLE_ARRAY_FILENAME,
    SAMPLE_EVIDENCE_ID,
    SAMPLE_METADATA_FILENAME,
    load_evaluation_evidence,
    load_sample_evidence,
    write_evaluation_evidence,
    write_sample_evidence,
)
from kahkm_dynamical_pairs import PAIR_LAW_ID
from kahkm_dynamical_sampling import (
    DYNAMICAL_SAMPLING_ID,
    SampledDynamicalDataset,
    SeedRecord,
)
from kahkm_reference_classes import REFERENCE_CLASS_MAP_ID


class DynamicalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(
            self.temporary.name
        )

    def tearDown(self):
        self.temporary.cleanup()

    def seed_record(
        self,
        *,
        pair_index,
        stream_name,
        generated_seed,
    ):
        stream_key = (
            1
            if stream_name
            == "trajectory_seed"
            else 2
        )

        material = (
            3,
            1,
            2,
            3,
            4,
            pair_index,
            stream_key,
            20261003,
        )

        return SeedRecord(
            sampling_id=
                DYNAMICAL_SAMPLING_ID,
            bit_generator=
                "PCG64",
            seed_sequence=
                "numpy.random.SeedSequence",
            campaign_key=3,
            system=
                "duffing",
            system_key=1,
            evaluation_mode=
                "deterministic",
            mode_key=2,
            sample_size=3,
            replicate_index=4,
            pair_index=
                pair_index,
            stream_name=
                stream_name,
            stream_key=
                stream_key,
            root_seed=20261003,
            seed_material=
                material,
            generated_uint32_words=(
                generated_seed
                & 0xFFFFFFFF,
                (
                    generated_seed
                    >> 32
                )
                & 0xFFFFFFFF,
            ),
            generated_seed=
                generated_seed,
        )

    def dataset(self):
        current = np.array(
            [
                [-0.9, 0.8, -0.2],
                [0.0, 0.0, 0.1],
            ],
            dtype=np.float64,
        )

        successor = np.array(
            [
                [-0.8, 0.7, 0.1],
                [0.1, -0.1, 0.2],
            ],
            dtype=np.float64,
        )

        trajectory = tuple(
            self.seed_record(
                pair_index=index,
                stream_name=
                    "trajectory_seed",
                generated_seed=
                    1000 + index,
            )
            for index
            in range(3)
        )

        times = tuple(
            self.seed_record(
                pair_index=index,
                stream_name=
                    "time_seed",
                generated_seed=
                    2000 + index,
            )
            for index
            in range(3)
        )

        return SampledDynamicalDataset(
            sampling_id=
                DYNAMICAL_SAMPLING_ID,
            pair_law_id=
                PAIR_LAW_ID,
            system=
                "duffing",
            evaluation_mode=
                "deterministic",
            sample_size=3,
            replicate_index=4,
            dt=0.03,
            horizon_steps=1200,
            vanderpol_mu=1.0,
            current_states=
                current,
            successor_states=
                successor,
            time_indices=(
                0,
                17,
                1199,
            ),
            trajectory_seed_records=
                trajectory,
            time_seed_records=
                times,
        )

    def evaluation(self):
        phi = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.6, 0.4],
            ],
            dtype=np.float64,
        )

        successors = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.3, 0.7],
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

        statistics = statistics_from_pairs(
            phi=phi,
            successors=
                successors,
            labels=labels,
        )

        budgets = []

        for kappa in (
            1.0,
            2.0,
        ):
            certificate = (
                certificate_from_statistics(
                    f_hat=
                        statistics.f_hat,
                    s_hat=
                        statistics.s_hat,
                    n_pairs=3,
                    kappa=kappa,
                    delta=0.05,
                )
            )

            budgets.append(
                BudgetCertificate(
                    kappa=kappa,
                    frozen_predictor_within_budget=
                        (
                            kappa
                            >= 1.25
                        ),
                    certificate=
                        certificate,
                    frozen_predictor_rmse_minus_bound=
                        0.4
                        - certificate.L_kappa_delta,
                )
            )

        phi.setflags(
            write=False
        )
        successors.setflags(
            write=False
        )

        return DynamicalEvaluationResult(
            evaluation_id=
                DYNAMICAL_EVALUATION_ID,
            reference_class_map_id=
                REFERENCE_CLASS_MAP_ID,
            n_pairs=3,
            n_classes=2,
            state_dimension=2,
            delta=0.05,
            frozen_predictor_spectral_norm=
                1.25,
            frozen_predictor_mse=
                0.16,
            frozen_predictor_rmse=
                0.4,
            statistics=
                statistics,
            budget_certificates=
                tuple(
                    budgets
                ),
            phi_rows=
                phi,
            successor_rows=
                successors,
            labels=(
                0,
                1,
                0,
            ),
            selected_center_squared_distances=(
                0.01,
                0.04,
                0.65,
            ),
            phi_coordinate_min=
                0.1,
            phi_coordinate_max=
                0.9,
            successor_coordinate_min=
                0.2,
            successor_coordinate_max=
                0.8,
            current_state_min=(
                -0.9,
                0.0,
            ),
            current_state_max=(
                0.8,
                0.1,
            ),
            successor_state_min=(
                -0.8,
                -0.1,
            ),
            successor_state_max=(
                0.7,
                0.2,
            ),
        )

    # 1
    def test_sample_evidence_roundtrip_preserves_every_array(self):
        dataset = self.dataset()

        attempt = (
            self.root
            / "attempt"
        )

        metadata = write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
        )

        loaded = load_sample_evidence(
            attempt
        )

        self.assertEqual(
            metadata[
                "sample_evidence_id"
            ],
            SAMPLE_EVIDENCE_ID,
        )

        self.assertIs(
            metadata[
                "evaluation_performed"
            ],
            False,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "current_states"
            ],
            dataset.current_states,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "successor_states"
            ],
            dataset.successor_states,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "time_indices"
            ],
            np.asarray(
                dataset.time_indices,
                dtype=np.int64,
            ),
        )

        self.assertEqual(
            loaded.metadata,
            metadata,
        )

    # 2
    def test_sample_seed_material_roundtrip_is_exact(self):
        dataset = self.dataset()

        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
        )

        loaded = load_sample_evidence(
            attempt
        )

        expected_trajectory_seeds = np.asarray(
            [
                record.generated_seed
                for record
                in dataset.trajectory_seed_records
            ],
            dtype=np.uint64,
        )

        expected_time_seeds = np.asarray(
            [
                record.generated_seed
                for record
                in dataset.time_seed_records
            ],
            dtype=np.uint64,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "trajectory_seed_generated_seed"
            ],
            expected_trajectory_seeds,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "time_seed_generated_seed"
            ],
            expected_time_seeds,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "trajectory_seed_material"
            ],
            np.asarray(
                [
                    record.seed_material
                    for record
                    in dataset.trajectory_seed_records
                ],
                dtype=np.uint64,
            ),
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "time_seed_material"
            ],
            np.asarray(
                [
                    record.seed_material
                    for record
                    in dataset.time_seed_records
                ],
                dtype=np.uint64,
            ),
        )

    # 3
    def test_sample_evidence_bytes_are_deterministic(self):
        dataset = self.dataset()

        first = (
            self.root
            / "first"
        )

        second = (
            self.root
            / "second"
        )

        write_sample_evidence(
            attempt_dir=
                first,
            dataset=
                dataset,
        )

        write_sample_evidence(
            attempt_dir=
                second,
            dataset=
                dataset,
        )

        self.assertEqual(
            (
                first
                / SAMPLE_ARRAY_FILENAME
            ).read_bytes(),
            (
                second
                / SAMPLE_ARRAY_FILENAME
            ).read_bytes(),
        )

        self.assertEqual(
            (
                first
                / SAMPLE_METADATA_FILENAME
            ).read_bytes(),
            (
                second
                / SAMPLE_METADATA_FILENAME
            ).read_bytes(),
        )

    # 4
    def test_evaluation_evidence_roundtrip_preserves_coordinates_and_statistics(self):
        dataset = self.dataset()
        evaluation = self.evaluation()

        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
        )

        metadata = (
            write_evaluation_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    dataset,
                evaluation=
                    evaluation,
            )
        )

        loaded = (
            load_evaluation_evidence(
                attempt
            )
        )

        self.assertEqual(
            metadata[
                "evaluation_evidence_id"
            ],
            EVALUATION_EVIDENCE_ID,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "phi_rows"
            ],
            evaluation.phi_rows,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "successor_rows"
            ],
            evaluation.successor_rows,
        )

        np.testing.assert_array_equal(
            loaded.arrays[
                "labels"
            ],
            np.asarray(
                evaluation.labels,
                dtype=np.int64,
            ),
        )

        self.assertEqual(
            loaded.metadata[
                "statistics"
            ][
                "f_hat"
            ],
            evaluation.statistics.f_hat,
        )

        self.assertEqual(
            loaded.metadata[
                "statistics"
            ][
                "s_hat"
            ],
            evaluation.statistics.s_hat,
        )

    # 5
    def test_evaluation_evidence_retains_all_budget_certificate_fields(self):
        dataset = self.dataset()
        evaluation = self.evaluation()

        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
        )

        metadata = (
            write_evaluation_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    dataset,
                evaluation=
                    evaluation,
            )
        )

        records = metadata[
            "budget_certificates"
        ]

        self.assertEqual(
            len(records),
            2,
        )

        for source, retained in zip(
            evaluation.budget_certificates,
            records,
            strict=True,
        ):
            certificate = (
                source.certificate
            )

            self.assertEqual(
                retained[
                    "kappa"
                ],
                source.kappa,
            )

            self.assertEqual(
                retained[
                    "frozen_predictor_within_budget"
                ],
                source.frozen_predictor_within_budget,
            )

            self.assertEqual(
                retained[
                    "certificate"
                ][
                    "L_kappa_delta"
                ],
                certificate.L_kappa_delta,
            )

            self.assertEqual(
                retained[
                    "certificate"
                ][
                    "F_delta"
                ],
                certificate.F_delta,
            )

            self.assertEqual(
                retained[
                    "certificate"
                ][
                    "V_delta"
                ],
                certificate.V_delta,
            )

    # 6
    def test_evaluation_evidence_bytes_are_deterministic(self):
        dataset = self.dataset()
        evaluation = self.evaluation()

        first = (
            self.root
            / "first"
        )

        second = (
            self.root
            / "second"
        )

        for attempt in (
            first,
            second,
        ):
            write_sample_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    dataset,
            )

            write_evaluation_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    dataset,
                evaluation=
                    evaluation,
            )

        self.assertEqual(
            (
                first
                / EVALUATION_ARRAY_FILENAME
            ).read_bytes(),
            (
                second
                / EVALUATION_ARRAY_FILENAME
            ).read_bytes(),
        )

        self.assertEqual(
            (
                first
                / EVALUATION_METADATA_FILENAME
            ).read_bytes(),
            (
                second
                / EVALUATION_METADATA_FILENAME
            ).read_bytes(),
        )

    # 7
    def test_sample_directory_must_be_new(self):
        attempt = (
            self.root
            / "attempt"
        )
        attempt.mkdir()

        with self.assertRaises(
            FileExistsError
        ):
            write_sample_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    self.dataset(),
            )

    # 8
    def test_bad_sample_state_shape_is_rejected_before_directory_creation(self):
        dataset = self.dataset()

        bad = SampledDynamicalDataset(
            **{
                **dataset.__dict__,
                "current_states":
                    np.zeros(
                        (
                            2,
                            2,
                        ),
                        dtype=np.float64,
                    ),
            }
        )

        attempt = (
            self.root
            / "attempt"
        )

        with self.assertRaisesRegex(
            ValueError,
            "shape",
        ):
            write_sample_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    bad,
            )

        self.assertFalse(
            attempt.exists()
        )

    # 9
    def test_nonfinite_sample_state_is_rejected(self):
        dataset = self.dataset()

        current = np.array(
            dataset.current_states,
            copy=True,
        )

        current[
            0,
            0,
        ] = np.nan

        bad = SampledDynamicalDataset(
            **{
                **dataset.__dict__,
                "current_states":
                    current,
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "nonfinite",
        ):
            write_sample_evidence(
                attempt_dir=
                    self.root
                    / "attempt",
                dataset=
                    bad,
            )

    # 10
    def test_invalid_time_indices_are_rejected(self):
        dataset = self.dataset()

        bad = SampledDynamicalDataset(
            **{
                **dataset.__dict__,
                "time_indices":
                    (
                        0,
                        17,
                        1200,
                    ),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "outside",
        ):
            write_sample_evidence(
                attempt_dir=
                    self.root
                    / "attempt",
                dataset=
                    bad,
            )

    # 11
    def test_seed_record_stream_or_metadata_mismatch_is_rejected(self):
        dataset = self.dataset()

        wrong_record = SeedRecord(
            **{
                **dataset.trajectory_seed_records[
                    0
                ].__dict__,
                "stream_name":
                    "time_seed",
            }
        )

        bad = SampledDynamicalDataset(
            **{
                **dataset.__dict__,
                "trajectory_seed_records":
                    (
                        wrong_record,
                        *dataset.trajectory_seed_records[
                            1:
                        ],
                    ),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "wrong stream",
        ):
            write_sample_evidence(
                attempt_dir=
                    self.root
                    / "attempt",
                dataset=
                    bad,
            )

    # 12
    def test_sample_write_failure_preserves_successfully_written_array_bytes(self):
        attempt = (
            self.root
            / "attempt"
        )

        with mock.patch.object(
            module,
            "_write_json_exclusive",
            side_effect=OSError(
                "synthetic metadata failure"
            ),
        ):
            with self.assertRaisesRegex(
                OSError,
                "synthetic metadata failure",
            ):
                write_sample_evidence(
                    attempt_dir=
                        attempt,
                    dataset=
                        self.dataset(),
                )

        self.assertTrue(
            attempt.is_dir()
        )

        self.assertTrue(
            (
                attempt
                / SAMPLE_ARRAY_FILENAME
            ).is_file()
        )

        self.assertFalse(
            (
                attempt
                / SAMPLE_METADATA_FILENAME
            ).exists()
        )

    # 13
    def test_evaluation_requires_persisted_sample_files(self):
        attempt = (
            self.root
            / "attempt"
        )
        attempt.mkdir()

        with self.assertRaisesRegex(
            RuntimeError,
            "Sample evidence",
        ):
            write_evaluation_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    self.dataset(),
                evaluation=
                    self.evaluation(),
            )

    # 14
    def test_evaluation_evidence_may_not_be_overwritten(self):
        dataset = self.dataset()
        evaluation = self.evaluation()

        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
        )

        write_evaluation_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
            evaluation=
                evaluation,
        )

        with self.assertRaises(
            FileExistsError
        ):
            write_evaluation_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    dataset,
                evaluation=
                    evaluation,
            )

    # 15
    def test_evaluation_pair_count_mismatch_is_rejected(self):
        dataset = self.dataset()
        evaluation = self.evaluation()

        bad = DynamicalEvaluationResult(
            **{
                **evaluation.__dict__,
                "n_pairs":
                    2,
            }
        )

        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
        )

        with self.assertRaisesRegex(
            ValueError,
            "pair count",
        ):
            write_evaluation_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    dataset,
                evaluation=
                    bad,
            )

    # 16
    def test_evaluation_association_shape_mismatch_is_rejected(self):
        dataset = self.dataset()
        evaluation = self.evaluation()

        bad = DynamicalEvaluationResult(
            **{
                **evaluation.__dict__,
                "phi_rows":
                    np.ones(
                        (
                            2,
                            2,
                        ),
                        dtype=np.float64,
                    ),
            }
        )

        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"\(M, C\)",
        ):
            write_evaluation_evidence(
                attempt_dir=
                    attempt,
                dataset=
                    dataset,
                evaluation=
                    bad,
            )

    # 17
    def test_sample_array_corruption_is_detected(self):
        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                self.dataset(),
        )

        path = (
            attempt
            / SAMPLE_ARRAY_FILENAME
        )

        path.write_bytes(
            path.read_bytes()
            + b"tamper"
        )

        with self.assertRaisesRegex(
            ValueError,
            "SHA256",
        ):
            load_sample_evidence(
                attempt
            )

    # 18
    def test_evaluation_array_corruption_is_detected(self):
        dataset = self.dataset()
        evaluation = self.evaluation()

        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
        )

        write_evaluation_evidence(
            attempt_dir=
                attempt,
            dataset=
                dataset,
            evaluation=
                evaluation,
        )

        path = (
            attempt
            / EVALUATION_ARRAY_FILENAME
        )

        path.write_bytes(
            path.read_bytes()
            + b"tamper"
        )

        with self.assertRaisesRegex(
            ValueError,
            "SHA256",
        ):
            load_evaluation_evidence(
                attempt
            )

    # 19
    def test_metadata_array_inventory_tampering_is_detected(self):
        attempt = (
            self.root
            / "attempt"
        )

        write_sample_evidence(
            attempt_dir=
                attempt,
            dataset=
                self.dataset(),
        )

        metadata_path = (
            attempt
            / SAMPLE_METADATA_FILENAME
        )

        payload = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        payload[
            "array_members"
        ] = [
            "current_states"
        ]

        metadata_path.write_text(
            json.dumps(
                payload
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "inventory",
        ):
            load_sample_evidence(
                attempt,
                verify_hash=False,
            )

    # 20
    def test_deterministic_npz_rejects_object_dtype(self):
        path = (
            self.root
            / "bad.npz"
        )

        with self.assertRaisesRegex(
            TypeError,
            "Object dtype",
        ):
            module._write_deterministic_npz(
                path,
                {
                    "bad":
                        np.array(
                            [
                                object(),
                            ],
                            dtype=object,
                        )
                },
            )


if __name__ == "__main__":
    unittest.main()
