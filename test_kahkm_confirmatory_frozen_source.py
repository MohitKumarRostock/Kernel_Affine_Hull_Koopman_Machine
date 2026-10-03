from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from kahkm_confirmatory_frozen_source import (
    ARCHIVE_MANIFEST_RELATIVE_PATH,
    ARCHIVE_RELATIVE_PATH,
    CAMPAIGN_ID,
    CONFIRMATORY_FROZEN_SOURCE_ID,
    EXPECTED_ARCHIVE_MANIFEST_SHA256,
    EXPECTED_ARCHIVE_MEMBER_COUNT,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_BUILD_EXECUTION_COMMIT,
    EXPECTED_CONFIGURATION_SHA256,
    EXPECTED_PACKAGE_CHECKSUMS_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_VERIFIER_SOURCE_COMMIT,
    OMEGA,
    PACKAGE_CHECKSUMS_RELATIVE_PATH,
    SYSTEM,
    TAU,
    _safe_archive_name,
    restore_repository_confirmatory_representation,
)
from kahkm_frozen_representation import (
    load_frozen_representation,
)


ROOT = Path(
    __file__
).resolve().parent


def sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


class ConfirmatoryFrozenSourceTests(
    unittest.TestCase
):
    def test_repository_package_fingerprints_are_exact(
        self,
    ):
        self.assertEqual(
            sha256_file(
                ROOT
                / ARCHIVE_RELATIVE_PATH
            ),
            EXPECTED_ARCHIVE_SHA256,
        )

        self.assertEqual(
            sha256_file(
                ROOT
                / ARCHIVE_MANIFEST_RELATIVE_PATH
            ),
            EXPECTED_ARCHIVE_MANIFEST_SHA256,
        )

        self.assertEqual(
            sha256_file(
                ROOT
                / PACKAGE_CHECKSUMS_RELATIVE_PATH
            ),
            EXPECTED_PACKAGE_CHECKSUMS_SHA256,
        )


    def test_safe_archive_name_accepts_canonical_member(
        self,
    ):
        result = _safe_archive_name(
            "evidence/representations/vanderpol/manifest.json"
        )

        self.assertEqual(
            result.as_posix(),
            "evidence/representations/vanderpol/manifest.json",
        )


    def test_safe_archive_name_rejects_traversal(
        self,
    ):
        bad_names = (
            "../escape",
            "evidence/../escape",
            "/absolute/path",
            "./relative",
            "evidence\\windows",
            "",
        )

        for name in bad_names:
            with self.subTest(
                name=name
            ):
                with self.assertRaises(
                    ValueError
                ):
                    _safe_archive_name(
                        name
                    )


    def test_real_repository_archive_restores_verified_representation(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            destination = (
                Path(
                    tmp
                )
                / "confirmatory"
            )

            restored = (
                restore_repository_confirmatory_representation(
                    repository_root=
                        ROOT,

                    restoration_dir=
                        destination,
                )
            )

            self.assertEqual(
                restored.source_id,
                CONFIRMATORY_FROZEN_SOURCE_ID,
            )

            self.assertEqual(
                restored.campaign_id,
                CAMPAIGN_ID,
            )

            self.assertEqual(
                restored.system,
                SYSTEM,
            )

            self.assertEqual(
                restored.omega,
                OMEGA,
            )

            self.assertEqual(
                restored.tau,
                TAU,
            )

            self.assertEqual(
                restored.archive_sha256,
                EXPECTED_ARCHIVE_SHA256,
            )

            self.assertEqual(
                restored.archive_manifest_sha256,
                EXPECTED_ARCHIVE_MANIFEST_SHA256,
            )

            self.assertEqual(
                restored.package_checksums_sha256,
                EXPECTED_PACKAGE_CHECKSUMS_SHA256,
            )

            self.assertEqual(
                restored.configuration_sha256,
                EXPECTED_CONFIGURATION_SHA256,
            )

            self.assertEqual(
                restored.protocol_sha256,
                EXPECTED_PROTOCOL_SHA256,
            )

            self.assertEqual(
                restored.build_execution_source_commit,
                EXPECTED_BUILD_EXECUTION_COMMIT,
            )

            self.assertEqual(
                restored.verifier_source_commit,
                EXPECTED_VERIFIER_SOURCE_COMMIT,
            )

            self.assertEqual(
                restored.certificate_evaluation_pairs_generated,
                0,
            )

            self.assertTrue(
                restored.vanderpol_artifact_dir.is_dir()
            )

            loaded = (
                load_frozen_representation(
                    restored.vanderpol_artifact_dir,
                    verify_hashes=True,
                )
            )

            self.assertEqual(
                loaded.system,
                SYSTEM,
            )

            self.assertEqual(
                loaded.omega,
                OMEGA,
            )

            self.assertEqual(
                loaded.tau,
                TAU,
            )

            self.assertEqual(
                len(
                    loaded.manifest[
                        "classifier_entries"
                    ]
                ),
                25,
            )

            self.assertEqual(
                EXPECTED_ARCHIVE_MEMBER_COUNT,
                58,
            )


    def test_restore_refuses_existing_destination(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            destination = (
                Path(
                    tmp
                )
                / "already-there"
            )

            destination.mkdir()

            with self.assertRaises(
                FileExistsError
            ):
                restore_repository_confirmatory_representation(
                    repository_root=
                        ROOT,

                    restoration_dir=
                        destination,
                )


    def test_restore_rejects_missing_repository_package(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(
                tmp
            )

            fake_repo = (
                base
                / "fake-repository"
            )

            fake_repo.mkdir()

            destination = (
                base
                / "restored"
            )

            with self.assertRaises(
                FileNotFoundError
            ):
                restore_repository_confirmatory_representation(
                    repository_root=
                        fake_repo,

                    restoration_dir=
                        destination,
                )

            self.assertFalse(
                destination.exists()
            )


if __name__ == "__main__":
    unittest.main()
