from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import scripts.verify_confirmatory_dynamical_representation as verifier


ROOT = Path(
    verifier.__file__
).resolve().parents[1]


class ConfirmatoryRepresentationVerifierTests(
    unittest.TestCase
):
    def test_config_and_protocol_hash_constants_are_exact(
        self,
    ):
        self.assertEqual(
            hashlib.sha256(
                (
                    ROOT
                    / verifier.CONFIRMATORY_CONFIG_RELATIVE
                ).read_bytes()
            ).hexdigest(),
            verifier.CONFIRMATORY_CONFIG_SHA256,
        )

        self.assertEqual(
            hashlib.sha256(
                (
                    ROOT
                    / verifier.CONFIRMATORY_PROTOCOL_RELATIVE
                ).read_bytes()
            ).hexdigest(),
            verifier.CONFIRMATORY_PROTOCOL_SHA256,
        )


    def test_build_summary_contract_accepts_expected_values(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(
                tmp
            )

            summary = {
                "runner_id":
                    verifier.EXPECTED_RUNNER_ID,

                "campaign_id":
                    verifier.EXPECTED_CAMPAIGN_ID,

                "campaign_role":
                    verifier.EXPECTED_CAMPAIGN_ROLE,

                "status":
                    "completed",

                "execution_source_commit":
                    verifier.EXPECTED_EXECUTION_COMMIT,

                "configuration_sha256":
                    verifier.CONFIRMATORY_CONFIG_SHA256,

                "protocol_sha256":
                    verifier.CONFIRMATORY_PROTOCOL_SHA256,

                "representation_derivation_id":
                    verifier.DERIVATION_ID,

                "num_systems_planned":
                    1,

                "num_systems_built":
                    1,

                "systems_built": [
                    "vanderpol"
                ],

                "state_space_kmeans_refits":
                    0,

                "autoencoder_refits":
                    0,

                "nlms_operator_refits":
                    1,

                "random_evaluation_pairs_generated":
                    0,

                "certificate_evaluation_pairs_generated":
                    0,

                "source_unchanged":
                    True,

                "exit_code":
                    0,

                "build_records": [
                    {
                        "system":
                            "vanderpol",

                        "source_archive_sha256":
                            verifier.EXPECTED_SOURCE_ARCHIVE_SHA256,

                        "source_manifest_sha256":
                            verifier.EXPECTED_SOURCE_MANIFEST_SHA256,
                    }
                ],
            }

            (
                evidence
                / "build_summary.json"
            ).write_text(
                json.dumps(
                    summary
                ),
                encoding="utf-8",
            )

            result = (
                verifier.verify_build_summary(
                    evidence
                )
            )

            self.assertEqual(
                result[
                    "execution_source_commit"
                ],
                verifier.EXPECTED_EXECUTION_COMMIT,
            )

            self.assertTrue(
                result[
                    "source_unchanged"
                ]
            )


    def test_build_summary_rejects_wrong_execution_commit(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(
                tmp
            )

            summary = {
                "runner_id":
                    verifier.EXPECTED_RUNNER_ID,

                "campaign_id":
                    verifier.EXPECTED_CAMPAIGN_ID,

                "campaign_role":
                    verifier.EXPECTED_CAMPAIGN_ROLE,

                "status":
                    "completed",

                "execution_source_commit":
                    "0"
                    * 40,

                "configuration_sha256":
                    verifier.CONFIRMATORY_CONFIG_SHA256,

                "protocol_sha256":
                    verifier.CONFIRMATORY_PROTOCOL_SHA256,

                "representation_derivation_id":
                    verifier.DERIVATION_ID,

                "num_systems_planned":
                    1,

                "num_systems_built":
                    1,

                "systems_built": [
                    "vanderpol"
                ],

                "state_space_kmeans_refits":
                    0,

                "autoencoder_refits":
                    0,

                "nlms_operator_refits":
                    1,

                "random_evaluation_pairs_generated":
                    0,

                "certificate_evaluation_pairs_generated":
                    0,

                "source_unchanged":
                    True,

                "exit_code":
                    0,

                "build_records": [
                    {
                        "system":
                            "vanderpol",

                        "source_archive_sha256":
                            verifier.EXPECTED_SOURCE_ARCHIVE_SHA256,

                        "source_manifest_sha256":
                            verifier.EXPECTED_SOURCE_MANIFEST_SHA256,
                    }
                ],
            }

            (
                evidence
                / "build_summary.json"
            ).write_text(
                json.dumps(
                    summary
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "execution_source_commit",
            ):
                verifier.verify_build_summary(
                    evidence
                )


    def test_checksum_inventory_detects_unlisted_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(
                tmp
            )

            (
                evidence
                / "a.txt"
            ).write_text(
                "a",
                encoding="utf-8",
            )

            (
                evidence
                / "b.txt"
            ).write_text(
                "b",
                encoding="utf-8",
            )

            a_sha = hashlib.sha256(
                b"a"
            ).hexdigest()

            (
                evidence
                / "SHA256SUMS"
            ).write_text(
                f"{a_sha}  a.txt\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "inventory mismatch",
            ):
                verifier.verify_checksum_inventory(
                    evidence
                )


    def test_verify_evidence_removes_scratch_and_reports_no_evaluation_pairs(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(
                tmp
            )

            evidence = (
                base
                / "evidence"
            )

            evidence.mkdir()

            scratch = (
                base
                / "scratch"
            )

            with (
                mock.patch.object(
                    verifier,
                    "verify_checksum_inventory",
                    return_value={
                        "verified_file_count":
                            55,

                        "sha256sums_fingerprint":
                            "a"
                            * 64,
                    },
                ),
                mock.patch.object(
                    verifier,
                    "verify_configuration_and_protocol",
                    return_value={
                        "configuration_sha256":
                            verifier.CONFIRMATORY_CONFIG_SHA256,

                        "protocol_sha256":
                            verifier.CONFIRMATORY_PROTOCOL_SHA256,

                        "repository_evidence_bytes_identical":
                            True,
                    },
                ),
                mock.patch.object(
                    verifier,
                    "verify_build_summary",
                    return_value={
                        "execution_source_commit":
                            verifier.EXPECTED_EXECUTION_COMMIT,

                        "source_unchanged":
                            True,

                        "certificate_evaluation_pairs_generated":
                            0,
                    },
                ),
                mock.patch.object(
                    verifier,
                    "verify_representation_numerics",
                    return_value={
                        "system":
                            "vanderpol",

                        "nlms_operator_recomputed_independently":
                            True,
                    },
                ),
                mock.patch.object(
                    verifier,
                    "verify_no_leftover_work",
                ),
            ):
                report = (
                    verifier.verify_evidence(
                        evidence_dir=
                            evidence,

                        scratch_dir=
                            scratch,
                    )
                )

            self.assertEqual(
                report[
                    "status"
                ],
                "passed",
            )

            self.assertEqual(
                report[
                    "certificate_evaluation_pairs_generated"
                ],
                0,
            )

            self.assertFalse(
                report[
                    "retained_representation_modified"
                ]
            )

            self.assertFalse(
                scratch.exists()
            )


    def test_representation_numerics_requires_exact_operator_match(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(
                tmp
            )

            evidence = (
                base
                / "evidence"
            )

            artifact = (
                evidence
                / "representations"
                / "vanderpol"
            )

            artifact.mkdir(
                parents=True
            )

            scratch = (
                base
                / "scratch"
            )

            scratch.mkdir()

            source_artifact = (
                scratch
                / "source"
            )

            source_artifact.mkdir()

            entries = [
                f"000/ae_cluster_{index + 1:05d}.joblib"
                for index in range(
                    25
                )
            ]

            shard_hashes = {
                "autoencoders/"
                + entry:
                    "f"
                    * 64
                for entry in entries
            }

            centers = np.zeros(
                (
                    2,
                    25,
                ),
                dtype=np.float64,
            )

            common_model_meta = {
                "n_clusters":
                    25,

                "cluster_strategy":
                    "kmeans_state",

                "soft_alpha":
                    None,

                "soft_topk":
                    10,

                "scaling_enabled":
                    False,

                "l2_normalization_enabled":
                    False,

                "kahkm_cluster_on":
                    "state",

                "kahkm_normalization":
                    "paper_eq_7",

                "kahkm_tau":
                    1e-6,
            }

            retained_manifest = {
                "classifier_entries":
                    entries,

                "files_sha256":
                    dict(
                        shard_hashes
                    ),

                "model_metadata": {
                    **common_model_meta,
                    "kahkm_omega":
                        128.0,
                },

                "training_metadata": {
                    "derivation_id":
                        verifier.DERIVATION_ID,

                    "configuration_sha256":
                        verifier.CONFIRMATORY_CONFIG_SHA256,

                    "protocol_sha256":
                        verifier.CONFIRMATORY_PROTOCOL_SHA256,

                    "source_representation_archive_sha256":
                        verifier.EXPECTED_SOURCE_ARCHIVE_SHA256,

                    "source_representation_manifest_sha256":
                        verifier.EXPECTED_SOURCE_MANIFEST_SHA256,

                    "source_representation_build_execution_commit":
                        verifier.EXPECTED_SOURCE_BUILD_COMMIT,

                    "source_representation_verifier_source_commit":
                        verifier.EXPECTED_SOURCE_VERIFIER_COMMIT,

                    "source_representation_verification_status":
                        "passed",

                    "source_omega":
                        4.0,

                    "source_tau":
                        1e-6,

                    "omega":
                        128.0,

                    "tau":
                        1e-6,

                    "train_seeds": [
                        0,
                        1,
                        2,
                    ],

                    "n_steps_train":
                        1200,

                    "n_train_snapshots":
                        3600,

                    "n_clusters":
                        25,

                    "beta":
                        0.1,

                    "nlms_epochs":
                        20,

                    "nlms_shuffle":
                        False,

                    "nlms_initial_matrix":
                        "identity",

                    "predictor_orientation":
                        "B.T @ Phi",

                    "project_stochastic":
                        False,

                    "reuse_state_space_kmeans_centers":
                        True,

                    "reuse_kahm_autoencoders":
                        True,

                    "refit_kmeans":
                        False,

                    "refit_autoencoders":
                        False,

                    "nlms_spectral_norm":
                        1.0,
                },
            }

            source_manifest = {
                "classifier_entries":
                    entries,

                "files_sha256":
                    dict(
                        shard_hashes
                    ),

                "model_metadata": {
                    **common_model_meta,
                    "kahkm_omega":
                        4.0,
                },
            }

            source_manifest_path = (
                source_artifact
                / "manifest.json"
            )

            source_manifest_path.write_text(
                "{}",
                encoding="utf-8",
            )

            retained = SimpleNamespace(
                format_id=
                    verifier.FORMAT_ID,

                system=
                    "vanderpol",

                omega=
                    128.0,

                tau=
                    1e-6,

                B=
                    2.0
                    * np.eye(
                        25,
                        dtype=np.float64,
                    ),

                nlms_history=
                    tuple(
                        0.0
                        for _ in range(
                            20
                        )
                    ),

                train_closure_error=
                    0.0,

                association_r2=
                    1.0,

                abstraction_model={
                    "cluster_centers":
                        centers.copy(),

                    "cluster_centers_init":
                        centers.copy(),
                },

                manifest=
                    retained_manifest,
            )

            source = SimpleNamespace(
                abstraction_model={
                    "cluster_centers":
                        centers.copy(),

                    "cluster_centers_init":
                        centers.copy(),
                },

                manifest=
                    source_manifest,
            )

            restored = SimpleNamespace(
                archive_sha256=
                    verifier.EXPECTED_SOURCE_ARCHIVE_SHA256,

                configuration_sha256=
                    verifier.EXPECTED_SOURCE_CONFIG_SHA256,

                vanderpol_artifact_dir=
                    str(
                        source_artifact
                    ),
            )

            phi = np.full(
                (
                    25,
                    3600,
                ),
                1.0 / 25.0,
                dtype=np.float64,
            )

            with (
                mock.patch.object(
                    verifier,
                    "load_frozen_representation",
                    side_effect=[
                        retained,
                        source,
                    ],
                ),
                mock.patch.object(
                    verifier,
                    "restore_repository_frozen_representations",
                    return_value=
                        restored,
                ),
                mock.patch.object(
                    verifier,
                    "sha256_file",
                    side_effect=lambda path:
                        (
                            verifier.EXPECTED_SOURCE_MANIFEST_SHA256
                            if Path(
                                path
                            ).name
                            == "manifest.json"
                            else hashlib.sha256(
                                Path(
                                    path
                                ).read_bytes()
                            ).hexdigest()
                        ),
                ),
                mock.patch.object(
                    verifier,
                    "make_concatenated_training_data",
                    return_value=(
                        np.zeros(
                            (
                                2,
                                3600,
                            ),
                            dtype=np.float64,
                        ),
                        np.zeros(
                            (
                                2,
                                3600,
                            ),
                            dtype=np.float64,
                        ),
                    ),
                ),
                mock.patch.object(
                    verifier,
                    "kahm_associations",
                    return_value=
                        phi,
                ),
                mock.patch.object(
                    verifier,
                    "nlms_koopman_closure",
                    return_value=(
                        np.eye(
                            25,
                            dtype=np.float64,
                        ),
                        tuple(
                            0.0
                            for _ in range(
                                20
                            )
                        ),
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "recomputed B differs",
                ):
                    verifier.verify_representation_numerics(
                        evidence,
                        scratch,
                    )


if __name__ == "__main__":
    unittest.main()
