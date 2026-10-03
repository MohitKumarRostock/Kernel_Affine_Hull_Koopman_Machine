"""Tests for the provenance-locked frozen-representation build runner.

All expensive representation fitting and association evaluation are mocked.
The tests exercise provenance, accounting, output guards, failure retention,
and finalization logic only.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import build_dynamical_certificate_representations as runner
from kahkm_dynamical_representation import (
    DYNAMICAL_PILOT_CONFIG_SHA256,
    FrozenBuildResult,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
)


class RepresentationBuildRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_root = Path(
            self.temporary.name
        )

    def tearDown(self):
        self.temporary.cleanup()

    def external_path(
        self,
        name="build-output",
    ):
        return (
            self.temp_root
            / name
        )

    def source_record(
        self,
        *,
        commit="abc123",
    ):
        return {
            "execution_source_commit":
                commit,
            "git_status_porcelain":
                "",
            "source_file_sha256": {
                "dummy.py":
                    "0" * 64,
            },
        }

    def config(self):
        return {
            "pilot_design": {
                "systems": [
                    "duffing",
                    "vanderpol",
                ]
            }
        }

    def build_result(
        self,
        system,
    ):
        if system == "duffing":
            clusters = 10
            omega = 2.0
            norm = 1.25
        else:
            clusters = 25
            omega = 4.0
            norm = 1.75

        return FrozenBuildResult(
            system=system,
            artifact_dir=str(
                self.external_path(
                    f"artifact-{system}"
                ).resolve()
            ),
            training_snapshot_count=3600,
            n_clusters=clusters,
            omega=omega,
            tau=1e-6,
            nlms_spectral_norm=norm,
            train_closure_error=0.125,
            association_r2=0.875,
            configuration_sha256=
                DYNAMICAL_PILOT_CONFIG_SHA256,
        )

    def probe(
        self,
        system,
    ):
        result = self.build_result(
            system
        )

        return {
            "format_id":
                FORMAT_ID,
            "system":
                system,
            "n_clusters":
                result.n_clusters,
            "state_dimension":
                2,
            "probe_shape": [
                result.n_clusters,
                result.n_clusters,
            ],
            "probe_coordinate_min":
                0.0,
            "probe_coordinate_max":
                1.0,
            "probe_max_simplex_error":
                0.0,
            "nlms_spectral_norm":
                result.nlms_spectral_norm,
            "train_closure_error":
                result.train_closure_error,
            "association_r2":
                result.association_r2,
        }

    # 1
    def test_runner_constants_are_frozen(self):
        self.assertEqual(
            runner.RUNNER_ID,
            "dynamical_certificate_representation_build_v1",
        )

        self.assertEqual(
            DYNAMICAL_PILOT_CONFIG_SHA256,
            "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1",
        )

        self.assertIn(
            "tests/test_build_dynamical_certificate_representations.py",
            runner.REQUIRED_SOURCE_FILES,
        )

        self.assertIn(
            "reproduction/certificates/DYNAMICAL_PROTOCOL.md",
            runner.REQUIRED_SOURCE_FILES,
        )

    # 2
    def test_validate_output_rejects_repository_path(self):
        candidate = (
            runner.ROOT
            / "forbidden-output"
        )

        with self.assertRaisesRegex(
            ValueError,
            "outside",
        ):
            runner.validate_output_location(
                candidate
            )

    # 3
    def test_validate_output_rejects_existing_external_path(self):
        candidate = self.external_path()
        candidate.mkdir()

        with self.assertRaises(
            FileExistsError
        ):
            runner.validate_output_location(
                candidate
            )

    # 4
    def test_validate_output_accepts_new_external_path(self):
        candidate = self.external_path()

        runner.validate_output_location(
            candidate
        )

        self.assertFalse(
            candidate.exists()
        )

    # 5
    def test_write_json_is_strict_and_exclusive(self):
        path = (
            self.temp_root
            / "record.json"
        )

        runner.write_json(
            path,
            {
                "value": 1.25,
            },
        )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            payload,
            {
                "value": 1.25,
            },
        )

        with self.assertRaises(
            FileExistsError
        ):
            runner.write_json(
                path,
                {
                    "value": 2.0,
                },
            )

        with self.assertRaises(
            ValueError
        ):
            runner.strict_json_text(
                {
                    "bad":
                        float("nan"),
                }
            )

    # 6
    def test_source_snapshot_rejects_wrong_repository_root(self):
        def fake_git(*args):
            if args == (
                "rev-parse",
                "--show-toplevel",
            ):
                return (
                    str(
                        self.temp_root
                    )
                    + "\n"
                ).encode()

            raise AssertionError(
                args
            )

        with mock.patch.object(
            runner,
            "git_bytes",
            side_effect=fake_git,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "repository root",
            ):
                runner.source_snapshot()

    # 7
    def test_source_snapshot_rejects_dirty_tree(self):
        def fake_git(*args):
            if args == (
                "rev-parse",
                "--show-toplevel",
            ):
                return (
                    str(
                        runner.ROOT
                    )
                    + "\n"
                ).encode()

            if args == (
                "rev-parse",
                "HEAD",
            ):
                return b"abc123\n"

            if args == (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ):
                return b"?? dirty.txt\n"

            raise AssertionError(
                args
            )

        with mock.patch.object(
            runner,
            "git_bytes",
            side_effect=fake_git,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "clean source",
            ):
                runner.source_snapshot()

    # 8
    def test_source_snapshot_detects_changed_required_bytes(self):
        real = (
            runner.ROOT
            / runner.REQUIRED_SOURCE_FILES[0]
        ).read_bytes()

        def fake_git(*args):
            if args == (
                "rev-parse",
                "--show-toplevel",
            ):
                return (
                    str(
                        runner.ROOT
                    )
                    + "\n"
                ).encode()

            if args == (
                "rev-parse",
                "HEAD",
            ):
                return b"abc123\n"

            if args == (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ):
                return b""

            if (
                len(args) == 2
                and args[0] == "show"
            ):
                return (
                    b"different"
                    if args[1].endswith(
                        ":"
                        + runner.REQUIRED_SOURCE_FILES[0]
                    )
                    else real
                )

            raise AssertionError(
                args
            )

        with (
            mock.patch.object(
                runner,
                "REQUIRED_SOURCE_FILES",
                (
                    runner.REQUIRED_SOURCE_FILES[0],
                ),
            ),
            mock.patch.object(
                runner,
                "git_bytes",
                side_effect=fake_git,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "differ from HEAD",
            ):
                runner.source_snapshot()

    # 9
    def test_assert_source_unchanged_detects_change(self):
        first = self.source_record(
            commit="one"
        )

        second = self.source_record(
            commit="two"
        )

        with mock.patch.object(
            runner,
            "source_snapshot",
            return_value=second,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "changed",
            ):
                runner.assert_source_unchanged(
                    first
                )

    # 10
    def test_run_command_check_retains_command_and_result(self):
        logs = (
            self.temp_root
            / "logs"
        )
        logs.mkdir()

        completed = SimpleNamespace(
            returncode=0
        )

        with mock.patch.object(
            runner.subprocess,
            "run",
            return_value=completed,
        ) as run_mock:
            runner.run_command_check(
                logs=logs,
                name="synthetic",
                command=(
                    "python",
                    "-V",
                ),
            )

        command = json.loads(
            (
                logs
                / "synthetic.command.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        result = json.loads(
            (
                logs
                / "synthetic.result.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            command[
                "command"
            ],
            [
                "python",
                "-V",
            ],
        )

        self.assertEqual(
            result[
                "returncode"
            ],
            0,
        )

        run_mock.assert_called_once()

    # 11
    def test_run_command_check_failure_is_not_hidden(self):
        logs = (
            self.temp_root
            / "logs"
        )
        logs.mkdir()

        completed = SimpleNamespace(
            returncode=7
        )

        with mock.patch.object(
            runner.subprocess,
            "run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Preflight",
            ):
                runner.run_command_check(
                    logs=logs,
                    name="synthetic",
                    command=(
                        "false",
                    ),
                )

        result = json.loads(
            (
                logs
                / "synthetic.result.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            result[
                "returncode"
            ],
            7,
        )

    # 12
    def test_write_checksums_inventories_files_and_excludes_itself(self):
        output = self.external_path()
        output.mkdir()

        (
            output
            / "a.txt"
        ).write_text(
            "alpha",
            encoding="utf-8",
        )

        nested = (
            output
            / "nested"
        )
        nested.mkdir()

        (
            nested
            / "b.bin"
        ).write_bytes(
            b"beta"
        )

        runner.write_checksums(
            output
        )

        lines = (
            output
            / "SHA256SUMS"
        ).read_text(
            encoding="utf-8"
        ).splitlines()

        names = {
            line.split(
                "  ",
                1,
            )[1]
            for line in lines
        }

        self.assertEqual(
            names,
            {
                "a.txt",
                "nested/b.bin",
            },
        )

        self.assertNotIn(
            "SHA256SUMS",
            names,
        )

    # 13
    def test_artifact_probe_accepts_valid_loaded_model(self):
        clusters = 3

        loaded = SimpleNamespace(
            abstraction_model={
                "cluster_centers":
                    np.array(
                        [
                            [-1.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                        ],
                        dtype=np.float64,
                    )
            },
            omega=2.0,
            tau=1e-6,
            B=np.diag(
                [
                    0.5,
                    1.25,
                    0.75,
                ]
            ),
            train_closure_error=0.1,
            association_r2=0.9,
            format_id=FORMAT_ID,
            system="duffing",
            manifest={
                "training_metadata": {
                    "nlms_spectral_norm":
                        1.25,
                }
            },
        )

        phi = np.eye(
            clusters,
            dtype=np.float64,
        )

        with (
            mock.patch.object(
                runner,
                "load_frozen_representation",
                return_value=loaded,
            ),
            mock.patch.object(
                runner,
                "kahm_associations",
                return_value=phi,
            ),
        ):
            result = runner.artifact_probe(
                self.external_path(
                    "artifact"
                )
            )

        self.assertEqual(
            result[
                "n_clusters"
            ],
            3,
        )

        self.assertEqual(
            result[
                "nlms_spectral_norm"
            ],
            1.25,
        )

        self.assertEqual(
            result[
                "probe_max_simplex_error"
            ],
            0.0,
        )

    # 14
    def test_artifact_probe_rejects_simplex_violation(self):
        loaded = SimpleNamespace(
            abstraction_model={
                "cluster_centers":
                    np.array(
                        [
                            [0.0, 1.0],
                            [0.0, 1.0],
                        ]
                    )
            },
            omega=2.0,
            tau=1e-6,
            B=np.eye(2),
            train_closure_error=0.1,
            association_r2=0.9,
            format_id=FORMAT_ID,
            system="duffing",
            manifest={
                "training_metadata": {
                    "nlms_spectral_norm":
                        1.0,
                }
            },
        )

        bad_phi = np.array(
            [
                [0.8, 0.5],
                [0.8, 0.5],
            ],
            dtype=np.float64,
        )

        with (
            mock.patch.object(
                runner,
                "load_frozen_representation",
                return_value=loaded,
            ),
            mock.patch.object(
                runner,
                "kahm_associations",
                return_value=bad_phi,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simplex",
            ):
                runner.artifact_probe(
                    self.external_path(
                        "artifact"
                    )
                )

    # 15
    def test_artifact_probe_rejects_recorded_norm_mismatch(self):
        loaded = SimpleNamespace(
            abstraction_model={
                "cluster_centers":
                    np.array(
                        [
                            [0.0, 1.0],
                            [0.0, 1.0],
                        ]
                    )
            },
            omega=2.0,
            tau=1e-6,
            B=np.diag(
                [
                    1.0,
                    2.0,
                ]
            ),
            train_closure_error=0.1,
            association_r2=0.9,
            format_id=FORMAT_ID,
            system="duffing",
            manifest={
                "training_metadata": {
                    "nlms_spectral_norm":
                        1.0,
                }
            },
        )

        with (
            mock.patch.object(
                runner,
                "load_frozen_representation",
                return_value=loaded,
            ),
            mock.patch.object(
                runner,
                "kahm_associations",
                return_value=np.eye(2),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "spectral norm",
            ):
                runner.artifact_probe(
                    self.external_path(
                        "artifact"
                    )
                )

    # 16
    def test_execute_builds_both_frozen_systems_and_finalizes(self):
        output = self.external_path()
        source = self.source_record()

        build_calls = []

        def fake_build(
            *,
            repository_root,
            system,
            output_dir,
            work_dir,
        ):
            self.assertEqual(
                Path(
                    repository_root
                ).resolve(),
                runner.ROOT,
            )

            Path(
                output_dir
            ).mkdir(
                parents=True,
                exist_ok=False,
            )

            build_calls.append(
                system
            )

            return replace(
                self.build_result(
                    system
                ),
                artifact_dir=str(
                    Path(
                        output_dir
                    ).resolve()
                ),
            )

        def fake_probe(
            artifact_dir,
        ):
            return self.probe(
                Path(
                    artifact_dir
                ).name
            )

        config_path = (
            runner.ROOT
            / runner.DYNAMICAL_PILOT_CONFIG_RELATIVE
        )

        real_config_bytes = (
            config_path.read_bytes()
        )

        with (
            mock.patch.object(
                runner,
                "source_snapshot",
                return_value=source,
            ),
            mock.patch.object(
                runner,
                "load_frozen_dynamical_pilot_config",
                return_value=self.config(),
            ),
            mock.patch.object(
                runner,
                "file_sha256",
                wraps=runner.file_sha256,
            ) as hash_mock,
            mock.patch.object(
                runner,
                "run_preflight",
            ) as preflight,
            mock.patch.object(
                runner,
                "fit_and_freeze_dynamical_representation",
                side_effect=fake_build,
            ),
            mock.patch.object(
                runner,
                "artifact_probe",
                side_effect=fake_probe,
            ),
        ):
            # Preserve the real frozen configuration hash while allowing all
            # other file hashing to use the implementation.
            real_hash = hashlib.sha256(
                real_config_bytes
            ).hexdigest()

            original_side_effect = (
                hash_mock._mock_wraps
            )

            def hash_side_effect(path):
                path = Path(path)

                if (
                    path.resolve()
                    == config_path.resolve()
                ):
                    return real_hash

                return original_side_effect(
                    path
                )

            hash_mock.side_effect = (
                hash_side_effect
            )

            summary = runner.execute(
                output_dir=output
            )

        self.assertEqual(
            build_calls,
            [
                "duffing",
                "vanderpol",
            ],
        )

        preflight.assert_called_once_with(
            output.resolve()
        )

        self.assertEqual(
            summary[
                "status"
            ],
            "completed",
        )

        self.assertEqual(
            summary[
                "num_systems_built"
            ],
            2,
        )

        self.assertEqual(
            summary[
                "certificate_evaluation_pairs_generated"
            ],
            0,
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

        self.assertTrue(
            (
                output
                / "configuration.json"
            ).is_file()
        )

        self.assertEqual(
            (
                output
                / "configuration.json"
            ).read_bytes(),
            real_config_bytes,
        )

    # 17
    def test_execute_rejects_unexpected_system_design(self):
        output = self.external_path()
        source = self.source_record()

        bad_config = {
            "pilot_design": {
                "systems": [
                    "duffing",
                ]
            }
        }

        with (
            mock.patch.object(
                runner,
                "source_snapshot",
                return_value=source,
            ),
            mock.patch.object(
                runner,
                "load_frozen_dynamical_pilot_config",
                return_value=bad_config,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "system order",
            ):
                runner.execute(
                    output_dir=output
                )

        self.assertFalse(
            output.exists()
        )

    # 18
    def test_execute_detects_source_change_after_build(self):
        output = self.external_path()

        first = self.source_record(
            commit="one"
        )

        changed = self.source_record(
            commit="two"
        )

        config_path = (
            runner.ROOT
            / runner.DYNAMICAL_PILOT_CONFIG_RELATIVE
        )

        calls = 0

        def snapshots():
            nonlocal calls
            calls += 1

            if calls <= 2:
                return first

            return changed

        def fake_build(
            *,
            repository_root,
            system,
            output_dir,
            work_dir,
        ):
            Path(
                output_dir
            ).mkdir(
                parents=True,
                exist_ok=False,
            )

            return replace(
                self.build_result(
                    system
                ),
                artifact_dir=str(
                    Path(
                        output_dir
                    ).resolve()
                ),
            )

        with (
            mock.patch.object(
                runner,
                "source_snapshot",
                side_effect=snapshots,
            ),
            mock.patch.object(
                runner,
                "load_frozen_dynamical_pilot_config",
                return_value=self.config(),
            ),
            mock.patch.object(
                runner,
                "run_preflight",
            ),
            mock.patch.object(
                runner,
                "fit_and_freeze_dynamical_representation",
                side_effect=fake_build,
            ),
            mock.patch.object(
                runner,
                "artifact_probe",
                side_effect=lambda path:
                    self.probe(
                        Path(
                            path
                        ).name
                    ),
            ),
            mock.patch.object(
                runner,
                "file_sha256",
                return_value=
                    DYNAMICAL_PILOT_CONFIG_SHA256,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "changed",
            ):
                runner.execute(
                    output_dir=output
                )

        self.assertTrue(
            output.exists()
        )

    # 19
    def test_main_failure_retains_failure_and_checksums(self):
        output = self.external_path()
        output.mkdir()

        namespace = SimpleNamespace(
            output_dir=output
        )

        with (
            mock.patch.object(
                runner,
                "parse_args",
                return_value=namespace,
            ),
            mock.patch.object(
                runner,
                "execute",
                side_effect=RuntimeError(
                    "synthetic failure"
                ),
            ),
        ):
            code = runner.main()

        self.assertEqual(
            code,
            1,
        )

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
                "status"
            ],
            "failed",
        )

        self.assertEqual(
            failure[
                "exception_type"
            ],
            "RuntimeError",
        )

        self.assertIn(
            "synthetic failure",
            failure[
                "exception_message"
            ],
        )

        self.assertTrue(
            (
                output
                / "SHA256SUMS"
            ).is_file()
        )

    # 20
    def test_main_success_returns_zero_without_extra_execution(self):
        output = self.external_path()

        namespace = SimpleNamespace(
            output_dir=output
        )

        summary = {
            "runner_id":
                runner.RUNNER_ID,
            "status":
                "completed",
        }

        with (
            mock.patch.object(
                runner,
                "parse_args",
                return_value=namespace,
            ),
            mock.patch.object(
                runner,
                "execute",
                return_value=summary,
            ) as execute_mock,
        ):
            code = runner.main()

        self.assertEqual(
            code,
            0,
        )

        execute_mock.assert_called_once_with(
            output_dir=output.resolve()
        )


if __name__ == "__main__":
    unittest.main()
