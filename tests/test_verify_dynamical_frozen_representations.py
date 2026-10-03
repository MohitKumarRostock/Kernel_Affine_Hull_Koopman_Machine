"""Tests for the frozen dynamical representation evidence verifier.

The tests use synthetic evidence or mocked portable representations. They do
not refit a KAHM, regenerate random samples, or generate certificate pairs.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

import scripts.verify_dynamical_frozen_representations as verifier
from kahkm_dynamical_representation import (
    DYNAMICAL_PILOT_CONFIG_RELATIVE,
    DYNAMICAL_PILOT_CONFIG_SHA256,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


def sha256_bytes(value):
    return hashlib.sha256(
        value
    ).hexdigest()


class FrozenRepresentationVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_root = Path(
            self.temporary.name
        )
        self.evidence_counter = 0

    def tearDown(self):
        self.temporary.cleanup()

    def make_inventory_evidence(self):
        self.evidence_counter += 1

        evidence = (
            self.temp_root
            / f"evidence_{self.evidence_counter:03d}"
        )

        evidence.mkdir()

        (
            evidence
            / "alpha.txt"
        ).write_text(
            "alpha",
            encoding="utf-8",
        )

        nested = (
            evidence
            / "nested"
        )
        nested.mkdir()

        (
            nested
            / "beta.bin"
        ).write_bytes(
            b"beta"
        )

        files = sorted(
            path
            for path
            in evidence.rglob("*")
            if path.is_file()
        )

        with (
            evidence
            / "SHA256SUMS"
        ).open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for path in files:
                handle.write(
                    f"{verifier.sha256_file(path)}  "
                    f"{path.relative_to(evidence).as_posix()}\n"
                )

        return evidence

    def make_loaded(
        self,
        system,
        *,
        norm=None,
        recorded_norm=None,
        n_clusters=None,
    ):
        design = (
            verifier.EXPECTED_SYSTEMS[
                system
            ]
        )

        clusters = (
            design[
                "n_clusters"
            ]
            if n_clusters is None
            else n_clusters
        )

        if norm is None:
            norm = (
                1.25
                if system == "duffing"
                else 1.75
            )

        if recorded_norm is None:
            recorded_norm = norm

        centers = np.linspace(
            -1.0,
            1.0,
            2 * clusters,
            dtype=np.float64,
        ).reshape(
            2,
            clusters,
        )

        B = np.zeros(
            (
                clusters,
                clusters,
            ),
            dtype=np.float64,
        )

        B[0, 0] = float(
            norm
        )

        manifest = {
            "model_metadata": {
                "n_clusters":
                    clusters,
                "kahkm_omega":
                    design[
                        "omega"
                    ],
                "kahkm_tau":
                    design[
                        "tau"
                    ],
            },
            "training_metadata": {
                "configuration_sha256":
                    DYNAMICAL_PILOT_CONFIG_SHA256,
                "n_train_snapshots":
                    design[
                        "n_train_snapshots"
                    ],
                "nlms_spectral_norm":
                    float(
                        recorded_norm
                    ),
            },
            "classifier_entries": [
                f"000/ae_cluster_{index + 1:05d}.joblib"
                for index
                in range(
                    clusters
                )
            ],
            "files_sha256": {
                "model_arrays.npz":
                    "0" * 64,
                "operator.npy":
                    "1" * 64,
                **{
                    (
                        f"autoencoders/000/"
                        f"ae_cluster_{index + 1:05d}.joblib"
                    ):
                        "2" * 64
                    for index
                    in range(
                        clusters
                    )
                },
            },
        }

        return SimpleNamespace(
            format_id=FORMAT_ID,
            system=system,
            abstraction_model={
                "cluster_centers":
                    centers,
                "cluster_centers_init":
                    centers.copy(),
            },
            B=B,
            omega=design[
                "omega"
            ],
            tau=design[
                "tau"
            ],
            train_closure_error=0.125,
            association_r2=0.875,
            manifest=manifest,
        )

    # 1
    def test_verifier_constants_are_frozen(self):
        self.assertEqual(
            verifier.VERIFIER_ID,
            "dynamical_frozen_representation_verifier_v1",
        )

        self.assertEqual(
            verifier.EXPECTED_EXECUTION_COMMIT,
            "1add19b94372257d16118f60c69e1e8afa668259",
        )

        self.assertEqual(
            DYNAMICAL_PILOT_CONFIG_SHA256,
            "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1",
        )

        self.assertIn(
            "tests/test_verify_dynamical_frozen_representations.py",
            verifier.REQUIRED_VERIFIER_SOURCE_FILES,
        )

    # 2
    def test_checksum_inventory_accepts_complete_valid_inventory(self):
        evidence = self.make_inventory_evidence()

        result = (
            verifier.verify_checksum_inventory(
                evidence
            )
        )

        self.assertEqual(
            result[
                "verified_file_count"
            ],
            2,
        )

        self.assertEqual(
            result[
                "sha256sums_fingerprint"
            ],
            verifier.sha256_file(
                evidence
                / "SHA256SUMS"
            ),
        )

    # 3
    def test_checksum_inventory_detects_changed_bytes(self):
        evidence = self.make_inventory_evidence()

        (
            evidence
            / "alpha.txt"
        ).write_text(
            "changed",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "SHA256 mismatch",
        ):
            verifier.verify_checksum_inventory(
                evidence
            )

    # 4
    def test_checksum_inventory_detects_unlisted_and_missing_files(self):
        evidence = self.make_inventory_evidence()

        (
            evidence
            / "extra.txt"
        ).write_text(
            "extra",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "inventory mismatch",
        ):
            verifier.verify_checksum_inventory(
                evidence
            )

        evidence = self.make_inventory_evidence()

        (
            evidence
            / "alpha.txt"
        ).unlink()

        with self.assertRaisesRegex(
            ValueError,
            "inventory mismatch",
        ):
            verifier.verify_checksum_inventory(
                evidence
            )

    # 5
    def test_checksum_inventory_rejects_duplicate_and_unsafe_entries(self):
        evidence = self.make_inventory_evidence()

        checksum = (
            evidence
            / "SHA256SUMS"
        )

        first = checksum.read_text(
            encoding="utf-8"
        ).splitlines()[0]

        checksum.write_text(
            first
            + "\n"
            + first
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate",
        ):
            verifier.verify_checksum_inventory(
                evidence
            )

        digest = "0" * 64

        checksum.write_text(
            f"{digest}  ../outside.txt\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsafe",
        ):
            verifier.verify_checksum_inventory(
                evidence
            )

    # 6
    def test_configuration_verification_accepts_exact_bytes(self):
        evidence = (
            self.temp_root
            / "evidence"
        )
        evidence.mkdir()

        repository_config = (
            PROJECT_ROOT
            / DYNAMICAL_PILOT_CONFIG_RELATIVE
        )

        (
            evidence
            / "configuration.json"
        ).write_bytes(
            repository_config.read_bytes()
        )

        result = (
            verifier.verify_configuration(
                evidence
            )
        )

        self.assertEqual(
            result[
                "configuration_sha256"
            ],
            DYNAMICAL_PILOT_CONFIG_SHA256,
        )

        self.assertIs(
            result[
                "repository_evidence_bytes_identical"
            ],
            True,
        )

    # 7
    def test_configuration_verification_rejects_changed_retained_bytes(self):
        evidence = (
            self.temp_root
            / "evidence"
        )
        evidence.mkdir()

        (
            evidence
            / "configuration.json"
        ).write_text(
            "{}\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "fingerprint",
        ):
            verifier.verify_configuration(
                evidence
            )

    # 8
    def test_build_summary_accepts_exact_provenance(self):
        evidence = (
            self.temp_root
            / "evidence"
        )
        evidence.mkdir()

        payload = {
            "runner_id":
                verifier.EXPECTED_RUNNER_ID,
            "status":
                "completed",
            "execution_source_commit":
                verifier.EXPECTED_EXECUTION_COMMIT,
            "configuration_sha256":
                DYNAMICAL_PILOT_CONFIG_SHA256,
            "num_systems_built":
                2,
            "source_unchanged":
                True,
            "certificate_evaluation_pairs_generated":
                0,
            "systems_built": [
                "duffing",
                "vanderpol",
            ],
            "build_records": [
                {
                    "system":
                        "duffing",
                },
                {
                    "system":
                        "vanderpol",
                },
            ],
        }

        (
            evidence
            / "build_summary.json"
        ).write_text(
            json.dumps(
                payload
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

        self.assertEqual(
            result[
                "certificate_evaluation_pairs_generated"
            ],
            0,
        )

    # 9
    def test_build_summary_rejects_wrong_execution_commit(self):
        evidence = (
            self.temp_root
            / "evidence"
        )
        evidence.mkdir()

        payload = {
            "runner_id":
                verifier.EXPECTED_RUNNER_ID,
            "status":
                "completed",
            "execution_source_commit":
                "wrong",
            "configuration_sha256":
                DYNAMICAL_PILOT_CONFIG_SHA256,
            "num_systems_built":
                2,
            "source_unchanged":
                True,
            "certificate_evaluation_pairs_generated":
                0,
            "systems_built": [
                "duffing",
                "vanderpol",
            ],
            "build_records": [
                {},
                {},
            ],
        }

        (
            evidence
            / "build_summary.json"
        ).write_text(
            json.dumps(
                payload
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

    # 10
    def test_representation_verification_accepts_duffing(self):
        artifact = (
            self.temp_root
            / "evidence"
            / "representations"
            / "duffing"
        )
        artifact.mkdir(
            parents=True
        )

        loaded = self.make_loaded(
            "duffing"
        )

        manifest_files = set(
            loaded.manifest[
                "files_sha256"
            ]
        )

        for relative in manifest_files:
            path = (
                artifact
                / relative
            )
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            path.write_bytes(
                b"x"
            )

        (
            artifact
            / "manifest.json"
        ).write_text(
            "{}",
            encoding="utf-8",
        )

        phi = np.full(
            (
                10,
                10,
            ),
            0.1,
            dtype=np.float64,
        )

        with (
            mock.patch.object(
                verifier,
                "load_frozen_representation",
                return_value=loaded,
            ),
            mock.patch.object(
                verifier,
                "kahm_associations",
                return_value=phi,
            ),
        ):
            result = (
                verifier.verify_representation(
                    self.temp_root
                    / "evidence",
                    "duffing",
                )
            )

        self.assertEqual(
            result[
                "n_clusters"
            ],
            10,
        )

        self.assertEqual(
            result[
                "shard_count"
            ],
            10,
        )

        self.assertAlmostEqual(
            result[
                "nlms_spectral_norm"
            ],
            1.25,
        )

    # 11
    def test_representation_verification_rejects_wrong_metadata(self):
        artifact = (
            self.temp_root
            / "evidence"
            / "representations"
            / "duffing"
        )
        artifact.mkdir(
            parents=True
        )

        loaded = self.make_loaded(
            "duffing"
        )

        loaded.manifest[
            "model_metadata"
        ][
            "kahkm_omega"
        ] = 12.0

        with mock.patch.object(
            verifier,
            "load_frozen_representation",
            return_value=loaded,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "kahkm_omega",
            ):
                verifier.verify_representation(
                    self.temp_root
                    / "evidence",
                    "duffing",
                )

    # 12
    def test_representation_verification_rejects_spectral_norm_mismatch(self):
        artifact = (
            self.temp_root
            / "evidence"
            / "representations"
            / "duffing"
        )
        artifact.mkdir(
            parents=True
        )

        loaded = self.make_loaded(
            "duffing",
            norm=1.25,
            recorded_norm=1.0,
        )

        with mock.patch.object(
            verifier,
            "load_frozen_representation",
            return_value=loaded,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "spectral norm",
            ):
                verifier.verify_representation(
                    self.temp_root
                    / "evidence",
                    "duffing",
                )

    # 13
    def test_representation_verification_rejects_simplex_violation(self):
        artifact = (
            self.temp_root
            / "evidence"
            / "representations"
            / "duffing"
        )
        artifact.mkdir(
            parents=True
        )

        loaded = self.make_loaded(
            "duffing"
        )

        phi = np.full(
            (
                10,
                10,
            ),
            0.2,
            dtype=np.float64,
        )

        with (
            mock.patch.object(
                verifier,
                "load_frozen_representation",
                return_value=loaded,
            ),
            mock.patch.object(
                verifier,
                "kahm_associations",
                return_value=phi,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "simplex",
            ):
                verifier.verify_representation(
                    self.temp_root
                    / "evidence",
                    "duffing",
                )

    # 14
    def test_representation_verification_rejects_internal_inventory_mismatch(self):
        artifact = (
            self.temp_root
            / "evidence"
            / "representations"
            / "duffing"
        )
        artifact.mkdir(
            parents=True
        )

        loaded = self.make_loaded(
            "duffing"
        )

        (
            artifact
            / "unexpected.bin"
        ).write_bytes(
            b"x"
        )

        with (
            mock.patch.object(
                verifier,
                "load_frozen_representation",
                return_value=loaded,
            ),
            mock.patch.object(
                verifier,
                "kahm_associations",
                return_value=np.full(
                    (
                        10,
                        10,
                    ),
                    0.1,
                ),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "internal artifact inventory",
            ):
                verifier.verify_representation(
                    self.temp_root
                    / "evidence",
                    "duffing",
                )

    # 15
    def test_leftover_work_directory_is_rejected(self):
        evidence = (
            self.temp_root
            / "run001"
        )
        evidence.mkdir()

        leftover = (
            evidence.parent
            / (
                f".{evidence.name}."
                "duffing.representation-work"
            )
        )
        leftover.mkdir()

        with self.assertRaisesRegex(
            ValueError,
            "Leftover",
        ):
            verifier.verify_no_leftover_work(
                evidence
            )

    # 16
    def test_verifier_source_snapshot_rejects_dirty_tree(self):
        def fake_git(*args):
            if args == (
                "rev-parse",
                "--show-toplevel",
            ):
                return (
                    str(
                        verifier.ROOT
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
                return b" M verifier.py\n"

            raise AssertionError(
                args
            )

        with mock.patch.object(
            verifier,
            "git_bytes",
            side_effect=fake_git,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "clean source",
            ):
                verifier.verifier_source_snapshot()

    # 17
    def test_verify_evidence_composes_all_checks_without_random_generation(self):
        evidence = (
            self.temp_root
            / "evidence"
        )
        evidence.mkdir()

        with (
            mock.patch.object(
                verifier,
                "verify_checksum_inventory",
                return_value={
                    "verified_file_count":
                        67,
                    "sha256sums_fingerprint":
                        "a" * 64,
                },
            ) as inventory,
            mock.patch.object(
                verifier,
                "verify_configuration",
                return_value={
                    "configuration_sha256":
                        DYNAMICAL_PILOT_CONFIG_SHA256,
                    "repository_evidence_bytes_identical":
                        True,
                },
            ) as configuration,
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
            ) as summary,
            mock.patch.object(
                verifier,
                "verify_representation",
                side_effect=[
                    {
                        "system":
                            "duffing",
                    },
                    {
                        "system":
                            "vanderpol",
                    },
                ],
            ) as representations,
            mock.patch.object(
                verifier,
                "verify_no_leftover_work",
            ) as leftovers,
        ):
            report = (
                verifier.verify_evidence(
                    evidence_dir=evidence
                )
            )

        inventory.assert_called_once()
        configuration.assert_called_once()
        summary.assert_called_once()

        self.assertEqual(
            representations.call_count,
            2,
        )

        leftovers.assert_called_once()

        self.assertEqual(
            report[
                "random_samples_regenerated"
            ],
            0,
        )

        self.assertEqual(
            report[
                "representations_refit"
            ],
            0,
        )

        self.assertEqual(
            report[
                "certificate_evaluation_pairs_generated"
            ],
            0,
        )

        self.assertIs(
            report[
                "evidence_modified"
            ],
            False,
        )

    # 18
    def test_main_rejects_existing_output_without_verifying(self):
        output = (
            self.temp_root
            / "verification"
        )
        output.mkdir()

        namespace = SimpleNamespace(
            evidence_dir=
                self.temp_root
                / "evidence",
            output_dir=output,
        )

        with (
            mock.patch.object(
                verifier,
                "parse_args",
                return_value=namespace,
            ),
            mock.patch.object(
                verifier,
                "verify_evidence",
            ) as verify_mock,
        ):
            code = verifier.main()

        self.assertEqual(
            code,
            1,
        )

        verify_mock.assert_not_called()

    # 19
    def test_main_failure_does_not_create_success_output(self):
        output = (
            self.temp_root
            / "verification"
        )

        namespace = SimpleNamespace(
            evidence_dir=
                self.temp_root
                / "evidence",
            output_dir=output,
        )

        with (
            mock.patch.object(
                verifier,
                "parse_args",
                return_value=namespace,
            ),
            mock.patch.object(
                verifier,
                "verifier_source_snapshot",
                return_value={
                    "verifier_source_commit":
                        "abc",
                },
            ),
            mock.patch.object(
                verifier,
                "verify_evidence",
                side_effect=ValueError(
                    "synthetic corruption"
                ),
            ),
        ):
            code = verifier.main()

        self.assertEqual(
            code,
            1,
        )

        self.assertFalse(
            output.exists()
        )

    def test_direct_script_entrypoint_can_import_repository_modules(self):
        completed = subprocess.run(
            [
                verifier.sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "verify_dynamical_frozen_representations.py"
                ),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr,
        )

        self.assertIn(
            "--evidence-dir",
            completed.stdout,
        )

        self.assertNotIn(
            "ModuleNotFoundError",
            completed.stderr,
        )

    # 21
    def test_main_success_writes_report_and_checksum(self):
        output = (
            self.temp_root
            / "verification"
        )

        evidence = (
            self.temp_root
            / "evidence"
        )

        namespace = SimpleNamespace(
            evidence_dir=evidence,
            output_dir=output,
        )

        source = {
            "verifier_source_commit":
                "abc123",
            "git_status_porcelain":
                "",
            "source_file_sha256": {
                "x":
                    "0" * 64,
            },
        }

        report = {
            "verifier_id":
                verifier.VERIFIER_ID,
            "status":
                "passed",
            "evidence_dir":
                str(
                    evidence.resolve()
                ),
        }

        with (
            mock.patch.object(
                verifier,
                "parse_args",
                return_value=namespace,
            ),
            mock.patch.object(
                verifier,
                "verifier_source_snapshot",
                side_effect=[
                    source,
                    source,
                ],
            ),
            mock.patch.object(
                verifier,
                "verify_evidence",
                return_value=dict(
                    report
                ),
            ),
        ):
            code = verifier.main()

        self.assertEqual(
            code,
            0,
        )

        report_path = (
            output
            / "verification_report.json"
        )

        checksum_path = (
            output
            / "verification_checksums.json"
        )

        self.assertTrue(
            report_path.is_file()
        )

        self.assertTrue(
            checksum_path.is_file()
        )

        written = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            written[
                "verifier_source"
            ],
            source,
        )

        checksums = json.loads(
            checksum_path.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            checksums[
                "verification_report.json"
            ],
            verifier.sha256_file(
                report_path
            ),
        )


if __name__ == "__main__":
    unittest.main()
