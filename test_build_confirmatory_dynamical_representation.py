from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_confirmatory_dynamical_representation as runner
from kahkm_confirmatory_representation import (
    CONFIRMATORY_CONFIG_SHA256,
    CONFIRMATORY_PROTOCOL_SHA256,
    ConfirmatoryRepresentationResult,
    DERIVATION_ID,
    EXPECTED_SOURCE_ARCHIVE_SHA256,
)


ROOT = Path(
    runner.__file__
).resolve().parent


class ConfirmatoryRepresentationBuildRunnerTests(
    unittest.TestCase
):
    def test_committed_config_and_protocol_hashes_are_exact(
        self,
    ):
        config = (
            ROOT
            / runner.CONFIRMATORY_CONFIG_RELATIVE
        )

        protocol = (
            ROOT
            / runner.CONFIRMATORY_PROTOCOL_RELATIVE
        )

        self.assertEqual(
            hashlib.sha256(
                config.read_bytes()
            ).hexdigest(),
            CONFIRMATORY_CONFIG_SHA256,
        )

        self.assertEqual(
            hashlib.sha256(
                protocol.read_bytes()
            ).hexdigest(),
            CONFIRMATORY_PROTOCOL_SHA256,
        )


    def test_load_configuration_accepts_only_frozen_design(
        self,
    ):
        raw, config = (
            runner.load_configuration()
        )

        self.assertEqual(
            hashlib.sha256(
                raw
            ).hexdigest(),
            CONFIRMATORY_CONFIG_SHA256,
        )

        self.assertEqual(
            config[
                "campaign_id"
            ],
            "dynamical_original_iid_confirmatory_v1",
        )

        self.assertEqual(
            config[
                "campaign_role"
            ],
            "confirmatory",
        )

        self.assertEqual(
            config[
                "confirmatory_design"
            ][
                "systems"
            ],
            [
                "vanderpol"
            ],
        )


    def test_load_protocol_is_exact(
        self,
    ):
        raw = (
            runner.load_protocol()
        )

        self.assertEqual(
            hashlib.sha256(
                raw
            ).hexdigest(),
            CONFIRMATORY_PROTOCOL_SHA256,
        )


    def test_required_source_contract_contains_critical_files(
        self,
    ):
        required = set(
            runner.REQUIRED_SOURCE_FILES
        )

        for relative in (
            "build_confirmatory_dynamical_representation.py",
            "kahkm_confirmatory_representation.py",
            "test_build_confirmatory_dynamical_representation.py",
            "test_kahkm_confirmatory_representation.py",
            runner.CONFIRMATORY_CONFIG_RELATIVE,
            runner.CONFIRMATORY_PROTOCOL_RELATIVE,
            "kahkm_dynamical_frozen_source.py",
            "kahkm_frozen_representation.py",
            "requirements-lock-arm64.txt",
            "scripts/verify_environment.py",
        ):
            self.assertIn(
                relative,
                required,
            )


    def test_existing_output_is_rejected_before_source_snapshot(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            output = (
                Path(
                    temporary
                )
                / "existing"
            )

            output.mkdir()

            with mock.patch.object(
                runner,
                "source_snapshot",
            ) as source:
                with self.assertRaises(
                    FileExistsError
                ):
                    runner.run_build(
                        output
                    )

            source.assert_not_called()


    def test_output_inside_repository_is_rejected(
        self,
    ):
        output = (
            ROOT
            / "forbidden-build-output"
        )

        with self.assertRaisesRegex(
            ValueError,
            "outside this working tree",
        ):
            runner.run_build(
                output
            )


    def test_successful_build_writes_complete_evidence_contract(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(
                temporary
            )

            output = (
                temp
                / "evidence"
            )

            source = {
                "execution_source_commit":
                    "a"
                    * 40,

                "git_status_porcelain":
                    "",

                "source_file_sha256": {
                    "example.py":
                        "b"
                        * 64,
                },
            }

            result = (
                ConfirmatoryRepresentationResult(
                    system=
                        "vanderpol",

                    artifact_dir=
                        str(
                            output
                            / "representations"
                            / "vanderpol"
                        ),

                    derivation_id=
                        DERIVATION_ID,

                    configuration_sha256=
                        CONFIRMATORY_CONFIG_SHA256,

                    protocol_sha256=
                        CONFIRMATORY_PROTOCOL_SHA256,

                    source_archive_sha256=
                        EXPECTED_SOURCE_ARCHIVE_SHA256,

                    source_manifest_sha256=
                        "c"
                        * 64,

                    n_train_snapshots=
                        3600,

                    n_clusters=
                        25,

                    omega=
                        128.0,

                    tau=
                        1e-6,

                    nlms_spectral_norm=
                        1.0203452494452896,

                    train_closure_error=
                        0.0832664988123408,

                    association_r2=
                        0.9122466631070364,
                )
            )

            def fake_derive(
                *,
                repository_root,
                output_dir,
                work_dir,
            ):
                self.assertEqual(
                    Path(
                        repository_root
                    ).resolve(),
                    ROOT,
                )

                destination = Path(
                    output_dir
                )

                destination.mkdir(
                    parents=True,
                    exist_ok=False,
                )

                (
                    destination
                    / "manifest.json"
                ).write_text(
                    "{}\n",
                    encoding="utf-8",
                )

                self.assertFalse(
                    Path(
                        work_dir
                    ).exists()
                )

                return replace(
                    result,
                    artifact_dir=
                        str(
                            destination.resolve()
                        ),
                )

            with (
                mock.patch.object(
                    runner,
                    "source_snapshot",
                    side_effect=[
                        source,
                        source,
                        source,
                    ],
                ),
                mock.patch.object(
                    runner,
                    "run_preflight",
                ) as preflight,
                mock.patch.object(
                    runner,
                    "derive_confirmatory_representation",
                    side_effect=fake_derive,
                ) as derive,
            ):
                code = runner.run_build(
                    output
                )

            self.assertEqual(
                code,
                0,
            )

            preflight.assert_called_once_with(
                output.resolve()
            )

            derive.assert_called_once()

            required = {
                "configuration.json",
                "protocol.md",
                "source_before.json",
                "source_after.json",
                "build_metadata.json",
                "build_summary.json",
                "SHA256SUMS",
                "representations/vanderpol/manifest.json",
            }

            actual = {
                path.relative_to(
                    output
                ).as_posix()
                for path in output.rglob(
                    "*"
                )
                if path.is_file()
            }

            self.assertTrue(
                required.issubset(
                    actual
                )
            )

            self.assertEqual(
                (
                    output
                    / "configuration.json"
                ).read_bytes(),
                (
                    ROOT
                    / runner.CONFIRMATORY_CONFIG_RELATIVE
                ).read_bytes(),
            )

            self.assertEqual(
                (
                    output
                    / "protocol.md"
                ).read_bytes(),
                (
                    ROOT
                    / runner.CONFIRMATORY_PROTOCOL_RELATIVE
                ).read_bytes(),
            )

            summary = json.loads(
                (
                    output
                    / "build_summary.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                summary[
                    "runner_id"
                ],
                runner.RUNNER_ID,
            )

            self.assertEqual(
                summary[
                    "status"
                ],
                "completed",
            )

            self.assertEqual(
                summary[
                    "execution_source_commit"
                ],
                "a"
                * 40,
            )

            self.assertEqual(
                summary[
                    "systems_built"
                ],
                [
                    "vanderpol"
                ],
            )

            self.assertEqual(
                summary[
                    "num_systems_built"
                ],
                1,
            )

            self.assertEqual(
                summary[
                    "state_space_kmeans_refits"
                ],
                0,
            )

            self.assertEqual(
                summary[
                    "autoencoder_refits"
                ],
                0,
            )

            self.assertEqual(
                summary[
                    "nlms_operator_refits"
                ],
                1,
            )

            self.assertEqual(
                summary[
                    "random_evaluation_pairs_generated"
                ],
                0,
            )

            self.assertEqual(
                summary[
                    "certificate_evaluation_pairs_generated"
                ],
                0,
            )

            self.assertTrue(
                summary[
                    "source_unchanged"
                ]
            )

            checksum_lines = (
                output
                / "SHA256SUMS"
            ).read_text(
                encoding="utf-8"
            ).splitlines()

            checksum_paths = {
                line.partition(
                    "  "
                )[2]
                for line in checksum_lines
                if line
            }

            self.assertEqual(
                checksum_paths,
                actual
                - {
                    "SHA256SUMS"
                },
            )


    def test_derivation_failure_is_retained_without_retry(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            output = (
                Path(
                    temporary
                )
                / "failure-evidence"
            )

            source = {
                "execution_source_commit":
                    "d"
                    * 40,

                "git_status_porcelain":
                    "",

                "source_file_sha256": {},
            }

            with (
                mock.patch.object(
                    runner,
                    "source_snapshot",
                    side_effect=[
                        source,
                        source,
                        source,
                    ],
                ),
                mock.patch.object(
                    runner,
                    "run_preflight",
                ),
                mock.patch.object(
                    runner,
                    "derive_confirmatory_representation",
                    side_effect=RuntimeError(
                        "synthetic derivation failure"
                    ),
                ) as derive,
            ):
                code = runner.run_build(
                    output
                )

            self.assertEqual(
                code,
                1,
            )

            derive.assert_called_once()

            self.assertTrue(
                (
                    output
                    / "failure.json"
                ).is_file()
            )

            self.assertTrue(
                (
                    output
                    / "build_summary.json"
                ).is_file()
            )

            self.assertTrue(
                (
                    output
                    / "SHA256SUMS"
                ).is_file()
            )

            summary = json.loads(
                (
                    output
                    / "build_summary.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                summary[
                    "status"
                ],
                "aborted",
            )

            self.assertEqual(
                summary[
                    "num_systems_built"
                ],
                0,
            )

            self.assertEqual(
                summary[
                    "nlms_operator_refits"
                ],
                0,
            )

            self.assertEqual(
                summary[
                    "certificate_evaluation_pairs_generated"
                ],
                0,
            )


    def test_source_change_after_preflight_aborts_before_derivation(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            output = (
                Path(
                    temporary
                )
                / "source-change"
            )

            before = {
                "execution_source_commit":
                    "e"
                    * 40,

                "git_status_porcelain":
                    "",

                "source_file_sha256": {
                    "x":
                        "1"
                        * 64,
                },
            }

            changed = {
                "execution_source_commit":
                    "e"
                    * 40,

                "git_status_porcelain":
                    "",

                "source_file_sha256": {
                    "x":
                        "2"
                        * 64,
                },
            }

            with (
                mock.patch.object(
                    runner,
                    "source_snapshot",
                    side_effect=[
                        before,
                        changed,
                        changed,
                    ],
                ),
                mock.patch.object(
                    runner,
                    "run_preflight",
                ),
                mock.patch.object(
                    runner,
                    "derive_confirmatory_representation",
                ) as derive,
            ):
                code = runner.run_build(
                    output
                )

            self.assertEqual(
                code,
                1,
            )

            derive.assert_not_called()

            failure = json.loads(
                (
                    output
                    / "failure.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                failure[
                    "aborted_stage"
                ],
                "preflight",
            )


if __name__ == "__main__":
    unittest.main()
