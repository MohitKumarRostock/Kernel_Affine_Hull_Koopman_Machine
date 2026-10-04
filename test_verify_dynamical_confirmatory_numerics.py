from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kahkm_certificate_statistics import statistics_from_pairs
from kahkm_certificates import certificate_from_statistics
import scripts.verify_dynamical_confirmatory_numerics as verifier


ROOT = Path(verifier.__file__).resolve().parents[1]


class ConfirmatoryNumericsVerifierTests(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual(
            verifier.EXPECTED_EXECUTION_COMMIT,
            "18dc9f3b965e2cc634764e05785ff60d5fbd6422",
        )
        self.assertEqual(verifier.EXPECTED_ATTEMPTS, 64)
        self.assertEqual(verifier.EXPECTED_RETAINED_PAIRS, 1_048_576)
        self.assertEqual(verifier.EXPECTED_SEED_STREAMS, 2_097_152)
        self.assertEqual(verifier.EXPECTED_EVIDENCE_INVENTORY_COUNT, 344)
        self.assertEqual(
            verifier.EXPECTED_EVIDENCE_SHA256SUMS_SHA256,
            "df9b1b7abd4a4c38b5bf8d0a5f43d7f6cb299ed23c070960b506f52641243221",
        )

    def test_repository_config_and_protocol_hashes(self):
        self.assertEqual(
            hashlib.sha256(
                (ROOT / verifier.CONFIG_PATH).read_bytes()
            ).hexdigest(),
            verifier.CONFIG_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / verifier.PROTOCOL_PATH).read_bytes()
            ).hexdigest(),
            verifier.PROTOCOL_SHA256,
        )

    def test_reconstructed_schedule_exact_design(self):
        config = verifier.strict_json_load(
            ROOT / verifier.CONFIG_PATH
        )
        jobs = verifier.reconstruct_schedule(config)

        self.assertEqual(len(jobs), 64)
        self.assertEqual(
            sum(job["sample_size"] for job in jobs),
            1_048_576,
        )
        self.assertEqual(jobs[0]["attempt_id"], "s2_q1_m16384_r000")
        self.assertEqual(jobs[31]["attempt_id"], "s2_q1_m16384_r031")
        self.assertEqual(jobs[32]["attempt_id"], "s2_q2_m16384_r000")
        self.assertEqual(jobs[-1]["attempt_id"], "s2_q2_m16384_r031")

    def test_expected_seed_material_exact_namespace(self):
        config = verifier.strict_json_load(
            ROOT / verifier.CONFIG_PATH
        )
        job = verifier.reconstruct_schedule(config)[0]

        self.assertEqual(
            verifier.expected_seed_material(
                config=config,
                job=job,
                pair_index=1234,
                stream_name="trajectory_seed",
            ),
            (4, 2, 1, 16384, 0, 1234, 1, 20261003),
        )
        self.assertEqual(
            verifier.expected_seed_material(
                config=config,
                job=job,
                pair_index=1234,
                stream_name="time_seed",
            ),
            (4, 2, 1, 16384, 0, 1234, 2, 20261003),
        )

    def test_generated_seed_and_words_match_seedsequence(self):
        material = (
            4,
            2,
            2,
            16384,
            31,
            16383,
            2,
            20261003,
        )
        words = np.random.SeedSequence(
            entropy=material
        ).generate_state(
            2,
            dtype=np.uint32,
        )
        expected_words = (
            int(words[0]),
            int(words[1]),
        )
        expected_seed = (
            expected_words[0]
            | (expected_words[1] << 32)
        )

        self.assertEqual(
            verifier._direct_seed_words(material),
            expected_words,
        )
        helper_words, helper_seed = (
            verifier.generated_seed_from_material(
                material
            )
        )

        self.assertEqual(
            tuple(
                int(value)
                for value in helper_words
            ),
            expected_words,
        )

        self.assertEqual(
            int(
                helper_seed
            ),
            expected_seed,
        )

    def test_independent_certificate_matches_validated_formula(self):
        expected = certificate_from_statistics(
            f_hat=0.022219286717808893,
            s_hat=0.11894444380043478,
            n_pairs=16384,
            kappa=1.0203452494452896,
            delta=0.05,
        )
        actual = verifier.independent_certificate(
            f_hat=0.022219286717808893,
            s_hat=0.11894444380043478,
            n_pairs=16384,
            kappa=1.0203452494452896,
            delta=0.05,
        )

        for key in (
            "f_hat",
            "s_hat",
            "r_delta",
            "F_delta",
            "V_delta",
            "L_kappa_delta",
        ):
            self.assertAlmostEqual(
                actual[key],
                getattr(expected, key),
                places=14,
            )

    def test_independent_statistics_match_validated_implementation(self):
        phi = np.array(
            [
                [0.7, 0.3, 0.0],
                [0.2, 0.8, 0.0],
                [0.4, 0.1, 0.5],
                [0.6, 0.2, 0.2],
            ],
            dtype=np.float64,
        )
        successors = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.3, 0.2, 0.5],
                [0.5, 0.2, 0.3],
            ],
            dtype=np.float64,
        )
        labels = np.array(
            [0, 1, 2, 0],
            dtype=np.int64,
        )

        expected = statistics_from_pairs(
            phi=phi,
            successors=successors,
            labels=labels,
        )
        actual = verifier.independent_statistics(
            phi,
            successors,
            labels,
        )

        self.assertAlmostEqual(
            actual["f_hat"],
            expected.f_hat,
            places=14,
        )
        self.assertAlmostEqual(
            actual["s_hat"],
            expected.s_hat,
            places=14,
        )
        np.testing.assert_array_equal(
            actual["class_counts"],
            expected.class_counts,
        )
        np.testing.assert_allclose(
            actual["class_successor_sse"],
            expected.class_successor_sse,
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["class_successor_means"],
            expected.class_successor_means,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_nearest_center_labels_smallest_index_tie(self):
        centers = np.array(
            [
                [0.0, 2.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        current = np.array(
            [
                [1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        labels, distances = verifier.nearest_center_labels(
            centers,
            current,
        )

        np.testing.assert_array_equal(
            labels,
            np.array([0, 1, 0]),
        )
        np.testing.assert_allclose(
            distances,
            np.array([1.0, 0.0, 0.0]),
        )

    def test_inventory_helper_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "a.txt").write_text(
                "a",
                encoding="utf-8",
            )
            digest = hashlib.sha256(b"a").hexdigest()
            (folder / "SHA256SUMS").write_text(
                f"{digest}  a.txt\n",
                encoding="utf-8",
            )

            report = verifier.check_inventory(
                folder,
                expected_count=1,
                expected_fingerprint=None,
            )
            self.assertEqual(
                report["verified_file_count"],
                1,
            )

            (folder / "a.txt").write_text(
                "tampered",
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                verifier.check_inventory(
                    folder,
                    expected_count=1,
                    expected_fingerprint=None,
                )


if __name__ == "__main__":
    unittest.main()
