"""Tests for repository-retained frozen dynamical representation restoration.

No representation is fitted and no certificate-evaluation pair is generated.
"""

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np

import kahkm_dynamical_frozen_source as module
from kahkm_dynamical_frozen_source import (
    ARCHIVE_FORMAT_ID,
    EXPECTED_ARCHIVE_MEMBER_FILE_COUNT,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_BUILD_EXECUTION_COMMIT,
    EXPECTED_CONFIGURATION_SHA256,
    EXPECTED_EVIDENCE_SHA256SUMS_FINGERPRINT,
    EXPECTED_VERIFICATION_CHECKSUMS_SHA256,
    EXPECTED_VERIFICATION_REPORT_SHA256,
    EXPECTED_VERIFIER_SOURCE_COMMIT,
    RestoredFrozenRepresentations,
    _require_safe_member,
    _verify_archive_metadata,
    restore_repository_frozen_representations,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


class FrozenSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_root = Path(
            self.temporary.name
        )

    def tearDown(self):
        self.temporary.cleanup()

    def valid_metadata(self):
        return {
            "archive_format_id":
                ARCHIVE_FORMAT_ID,
            "archive_file":
                module.ARCHIVE_FILENAME,
            "archive_sha256":
                EXPECTED_ARCHIVE_SHA256,
            "archive_member_file_count":
                EXPECTED_ARCHIVE_MEMBER_FILE_COUNT,
            "configuration_sha256":
                EXPECTED_CONFIGURATION_SHA256,
            "build_execution_commit":
                EXPECTED_BUILD_EXECUTION_COMMIT,
            "verifier_source_commit":
                EXPECTED_VERIFIER_SOURCE_COMMIT,
            "evidence_sha256sums_fingerprint":
                EXPECTED_EVIDENCE_SHA256SUMS_FINGERPRINT,
            "verification_report_sha256":
                EXPECTED_VERIFICATION_REPORT_SHA256,
            "verification_checksums_sha256":
                EXPECTED_VERIFICATION_CHECKSUMS_SHA256,
            "verification_status":
                "passed",
            "representations_verified":
                2,
            "random_samples_regenerated":
                0,
            "representations_refit":
                0,
            "certificate_evaluation_pairs_generated":
                0,
            "evidence_modified_by_verifier":
                False,
        }

    # 1
    def test_frozen_archive_constants_are_exact(self):
        self.assertEqual(
            EXPECTED_ARCHIVE_SHA256,
            "04a5ad3d4dae3a8fafce2576512fdce06d6c738907ced7d3ad9fde600d8c0e2a",
        )

        self.assertEqual(
            EXPECTED_ARCHIVE_MEMBER_FILE_COUNT,
            70,
        )

        self.assertEqual(
            EXPECTED_CONFIGURATION_SHA256,
            "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1",
        )

        self.assertEqual(
            EXPECTED_BUILD_EXECUTION_COMMIT,
            "1add19b94372257d16118f60c69e1e8afa668259",
        )

        self.assertEqual(
            EXPECTED_VERIFIER_SOURCE_COMMIT,
            "65db148f865c94f988676381c6b5f12222356b31",
        )

    # 2
    def test_archive_metadata_accepts_frozen_values(self):
        metadata = self.valid_metadata()

        result = _verify_archive_metadata(
            metadata
        )

        self.assertIs(
            result,
            metadata,
        )

    # 3
    def test_archive_metadata_rejects_non_mapping(self):
        for value in (
            None,
            [],
            "metadata",
            3,
        ):
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    TypeError
                ):
                    _verify_archive_metadata(
                        value
                    )

    # 4
    def test_archive_metadata_rejects_each_frozen_mismatch(self):
        base = self.valid_metadata()

        for key in tuple(
            base
        ):
            changed = dict(
                base
            )

            value = changed[
                key
            ]

            if isinstance(
                value,
                bool,
            ):
                changed[
                    key
                ] = not value
            elif isinstance(
                value,
                int,
            ):
                changed[
                    key
                ] = value + 1
            else:
                changed[
                    key
                ] = "changed"

            with self.subTest(
                key=key
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    key,
                ):
                    _verify_archive_metadata(
                        changed
                    )

    # 5
    def test_safe_regular_tar_member_is_accepted(self):
        member = tarfile.TarInfo(
            "evidence/file.bin"
        )
        member.type = tarfile.REGTYPE
        member.size = 3

        _require_safe_member(
            member
        )

    # 6
    def test_absolute_tar_member_is_rejected(self):
        member = tarfile.TarInfo(
            "/absolute/file.bin"
        )
        member.type = tarfile.REGTYPE

        with self.assertRaisesRegex(
            ValueError,
            "Unsafe",
        ):
            _require_safe_member(
                member
            )

    # 7
    def test_parent_traversal_tar_member_is_rejected(self):
        member = tarfile.TarInfo(
            "evidence/../escape.bin"
        )
        member.type = tarfile.REGTYPE

        with self.assertRaisesRegex(
            ValueError,
            "Unsafe",
        ):
            _require_safe_member(
                member
            )

    # 8
    def test_nonregular_tar_members_are_rejected(self):
        for tar_type in (
            tarfile.DIRTYPE,
            tarfile.SYMTYPE,
            tarfile.LNKTYPE,
        ):
            member = tarfile.TarInfo(
                "evidence/item"
            )
            member.type = tar_type

            with self.subTest(
                tar_type=tar_type
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "regular files",
                ):
                    _require_safe_member(
                        member
                    )

    # 9
    def test_restoration_inside_repository_is_rejected(self):
        destination = (
            PROJECT_ROOT
            / "forbidden_restore"
        )

        with self.assertRaisesRegex(
            ValueError,
            "outside",
        ):
            restore_repository_frozen_representations(
                repository_root=
                    PROJECT_ROOT,
                restoration_dir=
                    destination,
            )

    # 10
    def test_existing_external_restoration_directory_is_rejected(self):
        destination = (
            self.temp_root
            / "restore"
        )
        destination.mkdir()

        with self.assertRaises(
            FileExistsError
        ):
            restore_repository_frozen_representations(
                repository_root=
                    PROJECT_ROOT,
                restoration_dir=
                    destination,
            )

    # 11
    def test_missing_committed_archive_is_rejected(self):
        fake_repository = (
            self.temp_root
            / "repo"
        )

        archive_dir = (
            fake_repository
            / module.ARCHIVE_RELATIVE_DIR
        )
        archive_dir.mkdir(
            parents=True
        )

        (
            archive_dir
            / "ARCHIVE.json"
        ).write_text(
            json.dumps(
                self.valid_metadata()
            ),
            encoding="utf-8",
        )

        with self.assertRaises(
            FileNotFoundError
        ):
            restore_repository_frozen_representations(
                repository_root=
                    fake_repository,
                restoration_dir=
                    self.temp_root
                    / "restore",
            )

    # 12
    def test_missing_archive_metadata_is_rejected(self):
        fake_repository = (
            self.temp_root
            / "repo"
        )

        archive_dir = (
            fake_repository
            / module.ARCHIVE_RELATIVE_DIR
        )
        archive_dir.mkdir(
            parents=True
        )

        (
            archive_dir
            / module.ARCHIVE_FILENAME
        ).write_bytes(
            b"not-a-real-archive"
        )

        with self.assertRaises(
            FileNotFoundError
        ):
            restore_repository_frozen_representations(
                repository_root=
                    fake_repository,
                restoration_dir=
                    self.temp_root
                    / "restore",
            )

    # 13
    def test_wrong_archive_hash_is_rejected_before_extraction(self):
        fake_repository = (
            self.temp_root
            / "repo"
        )

        archive_dir = (
            fake_repository
            / module.ARCHIVE_RELATIVE_DIR
        )
        archive_dir.mkdir(
            parents=True
        )

        (
            archive_dir
            / "ARCHIVE.json"
        ).write_text(
            json.dumps(
                self.valid_metadata()
            ),
            encoding="utf-8",
        )

        (
            archive_dir
            / module.ARCHIVE_FILENAME
        ).write_bytes(
            b"wrong archive bytes"
        )

        destination = (
            self.temp_root
            / "restore"
        )

        with self.assertRaisesRegex(
            ValueError,
            "SHA256 mismatch",
        ):
            restore_repository_frozen_representations(
                repository_root=
                    fake_repository,
                restoration_dir=
                    destination,
            )

        self.assertFalse(
            destination.exists()
        )

    # 14
    def test_duplicate_tar_paths_are_rejected_and_partial_restore_removed(self):
        fake_repository = (
            self.temp_root
            / "repo"
        )
        archive_dir = (
            fake_repository
            / module.ARCHIVE_RELATIVE_DIR
        )
        archive_dir.mkdir(
            parents=True
        )

        archive = (
            archive_dir
            / module.ARCHIVE_FILENAME
        )

        payload = b"x"

        with tarfile.open(
            archive,
            mode="w:gz",
        ) as tar:
            for _ in range(2):
                info = tarfile.TarInfo(
                    "evidence/duplicate.bin"
                )
                info.type = tarfile.REGTYPE
                info.size = len(
                    payload
                )

                tar.addfile(
                    info,
                    io.BytesIO(
                        payload
                    ),
                )

        metadata = self.valid_metadata()

        with (
            mock.patch.object(
                module,
                "EXPECTED_ARCHIVE_SHA256",
                module._sha256_file(
                    archive
                ),
            ),
            mock.patch.object(
                module,
                "EXPECTED_ARCHIVE_MEMBER_FILE_COUNT",
                2,
            ),
        ):
            metadata[
                "archive_sha256"
            ] = module.EXPECTED_ARCHIVE_SHA256
            metadata[
                "archive_member_file_count"
            ] = 2

            (
                archive_dir
                / "ARCHIVE.json"
            ).write_text(
                json.dumps(
                    metadata
                ),
                encoding="utf-8",
            )

            destination = (
                self.temp_root
                / "restore"
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicate",
            ):
                restore_repository_frozen_representations(
                    repository_root=
                        fake_repository,
                    restoration_dir=
                        destination,
                )

            self.assertFalse(
                destination.exists()
            )

    # 15
    def test_nonfile_archive_member_is_rejected_and_cleanup_occurs(self):
        fake_repository = (
            self.temp_root
            / "repo"
        )
        archive_dir = (
            fake_repository
            / module.ARCHIVE_RELATIVE_DIR
        )
        archive_dir.mkdir(
            parents=True
        )

        archive = (
            archive_dir
            / module.ARCHIVE_FILENAME
        )

        with tarfile.open(
            archive,
            mode="w:gz",
        ) as tar:
            info = tarfile.TarInfo(
                "evidence"
            )
            info.type = tarfile.DIRTYPE

            tar.addfile(
                info
            )

        actual_hash = (
            module._sha256_file(
                archive
            )
        )

        metadata = self.valid_metadata()
        metadata[
            "archive_sha256"
        ] = actual_hash
        metadata[
            "archive_member_file_count"
        ] = 1

        (
            archive_dir
            / "ARCHIVE.json"
        ).write_text(
            json.dumps(
                metadata
            ),
            encoding="utf-8",
        )

        destination = (
            self.temp_root
            / "restore"
        )

        with (
            mock.patch.object(
                module,
                "EXPECTED_ARCHIVE_SHA256",
                actual_hash,
            ),
            mock.patch.object(
                module,
                "EXPECTED_ARCHIVE_MEMBER_FILE_COUNT",
                1,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "regular files",
            ):
                restore_repository_frozen_representations(
                    repository_root=
                        fake_repository,
                    restoration_dir=
                        destination,
                )

        self.assertFalse(
            destination.exists()
        )

    # 16
    def test_restored_representation_requires_expected_system_metadata(self):
        loaded = SimpleLoaded(
            system="duffing",
            n_clusters=9,
            omega=2.0,
            tau=1e-6,
        )

        with mock.patch.object(
            module,
            "load_frozen_representation",
            return_value=loaded,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "n_clusters",
            ):
                module._verify_restored_representation(
                    self.temp_root,
                    "duffing",
                )

    # 17
    def test_archived_verification_report_requires_expected_verifier_commit(self):
        restored = (
            self.temp_root
            / "restored"
        )

        verification = (
            restored
            / "verification"
        )
        verification.mkdir(
            parents=True
        )

        report = {
            "status":
                "passed",
            "num_representations":
                2,
            "execution_source_commit":
                EXPECTED_BUILD_EXECUTION_COMMIT,
            "configuration_sha256":
                EXPECTED_CONFIGURATION_SHA256,
            "sha256sums_fingerprint":
                EXPECTED_EVIDENCE_SHA256SUMS_FINGERPRINT,
            "random_samples_regenerated":
                0,
            "representations_refit":
                0,
            "certificate_evaluation_pairs_generated":
                0,
            "evidence_modified":
                False,
            "source_unchanged":
                True,
            "repository_evidence_bytes_identical":
                True,
            "verifier_source": {
                "verifier_source_commit":
                    "wrong",
            },
        }

        report_path = (
            verification
            / "verification_report.json"
        )

        report_path.write_text(
            json.dumps(
                report
            ),
            encoding="utf-8",
        )

        checksums_path = (
            verification
            / "verification_checksums.json"
        )

        checksums_path.write_text(
            json.dumps(
                {
                    "verification_report.json":
                        module._sha256_file(
                            report_path
                        )
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(
                module,
                "EXPECTED_VERIFICATION_REPORT_SHA256",
                module._sha256_file(
                    report_path
                ),
            ),
            mock.patch.object(
                module,
                "EXPECTED_VERIFICATION_CHECKSUMS_SHA256",
                module._sha256_file(
                    checksums_path
                ),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "verifier commit",
            ):
                module._verify_archived_verification_report(
                    restored
                )

    # 18
    def test_real_committed_archive_restores_and_validates(self):
        destination = (
            self.temp_root
            / "real_restore"
        )

        result = (
            restore_repository_frozen_representations(
                repository_root=
                    PROJECT_ROOT,
                restoration_dir=
                    destination,
            )
        )

        self.assertIsInstance(
            result,
            RestoredFrozenRepresentations,
        )

        self.assertEqual(
            result.archive_sha256,
            EXPECTED_ARCHIVE_SHA256,
        )

        self.assertEqual(
            result.configuration_sha256,
            EXPECTED_CONFIGURATION_SHA256,
        )

        self.assertEqual(
            result.build_execution_commit,
            EXPECTED_BUILD_EXECUTION_COMMIT,
        )

        self.assertEqual(
            result.verifier_source_commit,
            EXPECTED_VERIFIER_SOURCE_COMMIT,
        )

        self.assertEqual(
            result.archived_verification_status,
            "passed",
        )

        self.assertEqual(
            result.archive_member_file_count,
            70,
        )

        self.assertTrue(
            Path(
                result.duffing_artifact_dir
            ).is_dir()
        )

        self.assertTrue(
            Path(
                result.vanderpol_artifact_dir
            ).is_dir()
        )

        self.assertEqual(
            len(
                [
                    path
                    for path
                    in destination.rglob("*")
                    if path.is_file()
                ]
            ),
            70,
        )

    # 19
    def test_real_archive_source_bytes_are_not_modified_by_restore(self):
        archive = (
            PROJECT_ROOT
            / module.ARCHIVE_RELATIVE_DIR
            / module.ARCHIVE_FILENAME
        )

        before = (
            module._sha256_file(
                archive
            )
        )

        destination = (
            self.temp_root
            / "restore"
        )

        restore_repository_frozen_representations(
            repository_root=
                PROJECT_ROOT,
            restoration_dir=
                destination,
        )

        after = (
            module._sha256_file(
                archive
            )
        )

        self.assertEqual(
            before,
            after,
        )

        self.assertEqual(
            after,
            EXPECTED_ARCHIVE_SHA256,
        )

    # 20
    def test_result_dataclass_is_immutable(self):
        result = RestoredFrozenRepresentations(
            restoration_dir="/tmp/a",
            archive_sha256=
                EXPECTED_ARCHIVE_SHA256,
            configuration_sha256=
                EXPECTED_CONFIGURATION_SHA256,
            build_execution_commit=
                EXPECTED_BUILD_EXECUTION_COMMIT,
            verifier_source_commit=
                EXPECTED_VERIFIER_SOURCE_COMMIT,
            duffing_artifact_dir=
                "/tmp/a/duffing",
            vanderpol_artifact_dir=
                "/tmp/a/vanderpol",
            archived_verification_status=
                "passed",
            archive_member_file_count=70,
        )

        with self.assertRaises(
            Exception
        ):
            result.archive_sha256 = "changed"


class SimpleLoaded:
    def __init__(
        self,
        *,
        system,
        n_clusters,
        omega,
        tau,
    ):
        self.format_id = FORMAT_ID
        self.system = system
        self.omega = omega
        self.tau = tau
        self.abstraction_model = {
            "n_clusters":
                n_clusters,
        }
        self.manifest = {
            "training_metadata": {
                "configuration_sha256":
                    EXPECTED_CONFIGURATION_SHA256,
            }
        }


if __name__ == "__main__":
    unittest.main()
