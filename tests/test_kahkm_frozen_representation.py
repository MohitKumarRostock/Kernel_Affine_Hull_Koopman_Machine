"""Tests for portable frozen KAHKM representation artifacts.

All fixtures are synthetic. These tests do not fit a KAHM or generate
certificate-evaluation data.
"""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from kahkm_frozen_representation import (
    FORMAT_ID,
    load_frozen_representation,
    save_frozen_representation,
)


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class FrozenRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_counter = 0

    def tearDown(self):
        self.temporary.cleanup()

    def make_source(self, n_clusters=3):
        self.source_counter += 1

        source = (
            self.root
            / f"source_autoencoders_{self.source_counter:03d}"
        )

        source.mkdir()

        entries = []

        for index in range(n_clusters):
            relative = Path(
                "000"
            ) / (
                f"ae_cluster_{index + 1:05d}.joblib"
            )

            path = source / relative
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_bytes(
                b"KAHM-SYNTHETIC-SHARD\0"
                + index.to_bytes(
                    4,
                    "little",
                )
            )

            entries.append(
                relative.as_posix()
            )

        return source, entries

    def make_model(self):
        source, entries = self.make_source()

        centers = np.array(
            [
                [-1.0, 0.0, 2.0],
                [0.5, 1.5, -0.25],
            ],
            dtype=np.float64,
        )

        model = {
            "classifier": list(entries),
            "classifier_dir": str(source.resolve()),
            "model_id": "machine-specific-id-not-exported",
            "cluster_centers_init":
                centers.copy(),
            "cluster_centers":
                centers.copy(),
            "n_clusters": 3,
            "cluster_strategy":
                "kmeans_state",
            "soft_alpha": None,
            "soft_topk": 10,
            "scaling_enabled": False,
            "l2_normalization_enabled":
                False,
            "kahkm_cluster_on": "state",
            "kahkm_normalization":
                "paper_eq_7",
            "kahkm_omega": 4.0,
            "kahkm_tau": 1e-6,
        }

        return model

    def save(
        self,
        *,
        model=None,
        output_name="artifact",
        B=None,
        training_metadata=None,
    ):
        if model is None:
            model = self.make_model()

        if B is None:
            B = np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.05, 0.15, 0.8],
                ],
                dtype=np.float64,
            )

        if training_metadata is None:
            training_metadata = {
                "train_seeds": [0, 1, 2],
                "n_steps_train": 1200,
                "configuration_id":
                    "synthetic_fixture_v1",
            }

        destination = (
            self.root
            / output_name
        )

        save_frozen_representation(
            output_dir=destination,
            system="duffing",
            abstraction_model=model,
            B=B,
            train_closure_error=0.125,
            association_r2=0.875,
            nlms_history=(
                0.5,
                0.25,
                0.125,
            ),
            training_metadata=
                training_metadata,
        )

        return destination

    # 1
    def test_roundtrip_arrays_operator_and_fit_metadata(self):
        model = self.make_model()
        artifact = self.save(
            model=model
        )

        loaded = load_frozen_representation(
            artifact
        )

        self.assertEqual(
            loaded.format_id,
            FORMAT_ID,
        )
        self.assertEqual(
            loaded.system,
            "duffing",
        )
        self.assertEqual(
            loaded.omega,
            4.0,
        )
        self.assertEqual(
            loaded.tau,
            1e-6,
        )
        self.assertEqual(
            loaded.train_closure_error,
            0.125,
        )
        self.assertEqual(
            loaded.association_r2,
            0.875,
        )
        self.assertEqual(
            loaded.nlms_history,
            (
                0.5,
                0.25,
                0.125,
            ),
        )

        np.testing.assert_array_equal(
            loaded.abstraction_model[
                "cluster_centers"
            ],
            model[
                "cluster_centers"
            ],
        )

        np.testing.assert_array_equal(
            loaded.abstraction_model[
                "cluster_centers_init"
            ],
            model[
                "cluster_centers_init"
            ],
        )

        np.testing.assert_array_equal(
            loaded.B,
            np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.05, 0.15, 0.8],
                ],
                dtype=np.float64,
            ),
        )

        self.assertEqual(
            loaded.manifest[
                "training_metadata"
            ][
                "train_seeds"
            ],
            [0, 1, 2],
        )

    # 2
    def test_shards_are_copied_byte_for_byte_and_remain_relative(self):
        model = self.make_model()

        originals = {
            entry:
                (
                    Path(
                        model[
                            "classifier_dir"
                        ]
                    )
                    / entry
                ).read_bytes()
            for entry
            in model["classifier"]
        }

        artifact = self.save(
            model=model
        )

        loaded = load_frozen_representation(
            artifact
        )

        self.assertEqual(
            loaded.abstraction_model[
                "classifier"
            ],
            model["classifier"],
        )

        for entry, expected in originals.items():
            copied = (
                artifact
                / "autoencoders"
                / entry
            )

            self.assertEqual(
                copied.read_bytes(),
                expected,
            )

    # 3
    def test_artifact_is_relocatable(self):
        artifact = self.save()
        relocated = (
            self.root
            / "relocated"
            / "frozen_model"
        )

        relocated.parent.mkdir()

        shutil.copytree(
            artifact,
            relocated,
        )

        shutil.rmtree(
            artifact
        )

        loaded = load_frozen_representation(
            relocated
        )

        expected_dir = str(
            (
                relocated
                / "autoencoders"
            ).resolve()
        )

        self.assertEqual(
            loaded.abstraction_model[
                "classifier_dir"
            ],
            expected_dir,
        )

        manifest_text = (
            relocated
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "source_autoencoders",
            manifest_text,
        )

    # 4
    def test_manifest_inventory_hashes_every_nonmanifest_file(self):
        artifact = self.save()

        manifest = json.loads(
            (
                artifact
                / "manifest.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        expected_files = {
            path.relative_to(
                artifact
            ).as_posix()
            for path in artifact.rglob("*")
            if path.is_file()
            and path.name != "manifest.json"
        }

        self.assertEqual(
            set(
                manifest[
                    "files_sha256"
                ]
            ),
            expected_files,
        )

        for relative, expected_hash in (
            manifest[
                "files_sha256"
            ].items()
        ):
            self.assertEqual(
                sha256_file(
                    artifact
                    / relative
                ),
                expected_hash,
            )

    # 5
    def test_existing_output_is_rejected_without_modification(self):
        destination = (
            self.root
            / "artifact"
        )
        destination.mkdir()

        sentinel = (
            destination
            / "sentinel.txt"
        )
        sentinel.write_text(
            "preserve me",
            encoding="utf-8",
        )

        with self.assertRaises(
            FileExistsError
        ):
            save_frozen_representation(
                output_dir=destination,
                system="duffing",
                abstraction_model=
                    self.make_model(),
                B=np.eye(3),
                train_closure_error=0.1,
                association_r2=0.9,
                nlms_history=(0.1,),
                training_metadata={},
            )

        self.assertEqual(
            sentinel.read_text(
                encoding="utf-8"
            ),
            "preserve me",
        )

    # 6
    def test_in_memory_classifier_entries_are_rejected(self):
        model = self.make_model()

        model["classifier"] = [
            object(),
            object(),
            object(),
        ]

        with self.assertRaises(
            TypeError
        ):
            self.save(
                model=model
            )

    # 7
    def test_missing_classifier_dir_is_rejected(self):
        model = self.make_model()
        model["classifier_dir"] = None

        with self.assertRaises(
            TypeError
        ):
            self.save(
                model=model
            )

    # 8
    def test_missing_source_shard_is_rejected(self):
        model = self.make_model()

        missing = (
            Path(
                model[
                    "classifier_dir"
                ]
            )
            / model[
                "classifier"
            ][1]
        )
        missing.unlink()

        with self.assertRaises(
            FileNotFoundError
        ):
            self.save(
                model=model
            )

    # 9
    def test_leaf_source_symlink_is_rejected(self):
        model = self.make_model()

        first = (
            Path(
                model[
                    "classifier_dir"
                ]
            )
            / model[
                "classifier"
            ][0]
        )

        second = (
            Path(
                model[
                    "classifier_dir"
                ]
            )
            / model[
                "classifier"
            ][1]
        )

        first.unlink()

        try:
            first.symlink_to(
                second
            )
        except OSError as exc:
            self.skipTest(
                f"symlink unavailable: {exc}"
            )

        with self.assertRaisesRegex(
            ValueError,
            "symlink",
        ):
            self.save(
                model=model
            )

    # 10
    def test_relative_path_traversal_is_rejected(self):
        model = self.make_model()

        outside = (
            self.root
            / "outside.joblib"
        )
        outside.write_bytes(
            b"outside"
        )

        model["classifier"][0] = (
            "../outside.joblib"
        )

        with self.assertRaises(
            ValueError
        ):
            self.save(
                model=model
            )

    # 11
    def test_absolute_shard_outside_classifier_dir_is_rejected(self):
        model = self.make_model()

        outside = (
            self.root
            / "outside.joblib"
        )
        outside.write_bytes(
            b"outside"
        )

        model["classifier"][0] = str(
            outside.resolve()
        )

        with self.assertRaises(
            ValueError
        ):
            self.save(
                model=model
            )

    # 12
    def test_duplicate_classifier_shard_is_rejected(self):
        model = self.make_model()

        model["classifier"][1] = (
            model["classifier"][0]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate",
        ):
            self.save(
                model=model
            )

    # 13
    def test_classifier_count_must_equal_n_clusters(self):
        model = self.make_model()

        model["classifier"] = (
            model["classifier"][:2]
        )

        with self.assertRaisesRegex(
            ValueError,
            "shard count",
        ):
            self.save(
                model=model
            )

    # 14
    def test_cluster_center_shapes_and_counts_are_validated(self):
        model = self.make_model()

        model[
            "cluster_centers_init"
        ] = np.ones(
            (2, 2)
        )

        with self.assertRaisesRegex(
            ValueError,
            "same shape",
        ):
            self.save(
                model=model,
                output_name="bad_shape",
            )

        model = self.make_model()
        model["n_clusters"] = 4

        with self.assertRaisesRegex(
            ValueError,
            "cluster-center columns",
        ):
            self.save(
                model=model,
                output_name="bad_count",
            )

    # 15
    def test_operator_shape_and_nonfinite_values_are_validated(self):
        with self.assertRaisesRegex(
            ValueError,
            "B shape",
        ):
            self.save(
                B=np.eye(2),
                output_name="bad_operator_shape",
            )

        bad = np.eye(3)
        bad[1, 1] = np.nan

        with self.assertRaisesRegex(
            ValueError,
            "finite",
        ):
            self.save(
                B=bad,
                output_name="bad_operator_nan",
            )

    # 16
    def test_changed_artifact_file_is_detected_by_hash(self):
        artifact = self.save()

        shard = next(
            (
                artifact
                / "autoencoders"
            ).rglob(
                "*.joblib"
            )
        )

        shard.write_bytes(
            shard.read_bytes()
            + b"TAMPER"
        )

        with self.assertRaisesRegex(
            ValueError,
            "SHA256 mismatch",
        ):
            load_frozen_representation(
                artifact
            )

    # 17
    def test_missing_and_unlisted_artifact_files_are_rejected(self):
        artifact = self.save()

        shard = next(
            (
                artifact
                / "autoencoders"
            ).rglob(
                "*.joblib"
            )
        )

        original = shard.read_bytes()
        shard.unlink()

        with self.assertRaises(
            FileNotFoundError
        ):
            load_frozen_representation(
                artifact
            )

        shard.write_bytes(
            original
        )

        extra = (
            artifact
            / "unexpected.bin"
        )
        extra.write_bytes(
            b"not inventoried"
        )

        with self.assertRaisesRegex(
            ValueError,
            "inventory mismatch",
        ):
            load_frozen_representation(
                artifact
            )

    # 18
    def test_artifact_symlink_file_is_rejected_even_without_hash_verification(self):
        artifact = self.save()

        shard = next(
            (
                artifact
                / "autoencoders"
            ).rglob(
                "*.joblib"
            )
        )

        external = (
            self.root
            / "external-shard.bin"
        )
        external.write_bytes(
            shard.read_bytes()
        )

        shard.unlink()

        try:
            shard.symlink_to(
                external
            )
        except OSError as exc:
            self.skipTest(
                f"symlink unavailable: {exc}"
            )

        with self.assertRaisesRegex(
            ValueError,
            "symlink",
        ):
            load_frozen_representation(
                artifact,
                verify_hashes=False,
            )

    # 19
    def test_export_failure_removes_partial_destination(self):
        model = self.make_model()

        destination = (
            self.root
            / "artifact"
        )

        with mock.patch(
            "kahkm_frozen_representation.shutil.copyfile",
            side_effect=OSError(
                "synthetic copy failure"
            ),
        ):
            with self.assertRaisesRegex(
                OSError,
                "synthetic copy failure",
            ):
                save_frozen_representation(
                    output_dir=destination,
                    system="duffing",
                    abstraction_model=model,
                    B=np.eye(3),
                    train_closure_error=0.1,
                    association_r2=0.9,
                    nlms_history=(0.1,),
                    training_metadata={},
                )

        self.assertFalse(
            destination.exists()
        )

    # 20
    def test_hash_verification_can_be_disabled_only_for_content_check(self):
        artifact = self.save()

        shard = next(
            (
                artifact
                / "autoencoders"
            ).rglob(
                "*.joblib"
            )
        )

        shard.write_bytes(
            b"changed-but-still-a-regular-file"
        )

        loaded = load_frozen_representation(
            artifact,
            verify_hashes=False,
        )

        self.assertEqual(
            loaded.format_id,
            FORMAT_ID,
        )

        with self.assertRaisesRegex(
            ValueError,
            "SHA256 mismatch",
        ):
            load_frozen_representation(
                artifact,
                verify_hashes=True,
            )


if __name__ == "__main__":
    unittest.main()
