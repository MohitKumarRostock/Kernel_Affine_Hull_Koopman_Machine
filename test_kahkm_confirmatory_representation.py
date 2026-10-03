from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import kahkm_confirmatory_representation as module


def valid_config() -> dict:
    return {
        "campaign_id":
            "dynamical_original_iid_confirmatory_v1",

        "campaign_role":
            "confirmatory",

        "protocol":
            module.CONFIRMATORY_PROTOCOL_RELATIVE,

        "protocol_sha256":
            module.CONFIRMATORY_PROTOCOL_SHA256,

        "confirmatory_design": {
            "systems": [
                "vanderpol"
            ],
            "sample_sizes": [
                16384
            ],
            "replicates_per_cell":
                32,
            "planned_attempts":
                64,
            "planned_retained_pairs":
                1048576,
        },

        "source_frozen_abstraction": {
            "representation_archive_sha256":
                module.EXPECTED_SOURCE_ARCHIVE_SHA256,

            "source_configuration_sha256":
                module.EXPECTED_SOURCE_CONFIG_SHA256,

            "representation_build_execution_commit":
                module.EXPECTED_SOURCE_BUILD_COMMIT,

            "representation_verifier_source_commit":
                module.EXPECTED_SOURCE_VERIFIER_COMMIT,

            "reuse_state_space_kmeans_centers":
                True,

            "reuse_kahm_autoencoders":
                True,

            "refit_kmeans":
                False,

            "refit_autoencoders":
                False,
        },

        "confirmatory_representation": {
            "implementation_id":
                module.DERIVATION_ID,

            "system":
                "vanderpol",

            "omega":
                128.0,

            "tau":
                1e-6,

            "operator_refit_uses_training_data_only":
                True,

            "train_seeds": [
                0,
                1,
                2,
            ],

            "n_steps_train":
                1200,

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

            "batch_size":
                256,

            "n_jobs":
                -1,
        },
    }


class ConfirmatoryRepresentationTests(
    unittest.TestCase
):
    def test_validate_design_accepts_frozen_design(
        self,
    ):
        design = module._validate_design(
            valid_config()
        )

        self.assertEqual(
            design[
                "systems"
            ],
            [
                "vanderpol"
            ],
        )

        self.assertEqual(
            design[
                "sample_sizes"
            ],
            [
                16384
            ],
        )

        self.assertEqual(
            design[
                "replicates_per_cell"
            ],
            32,
        )


    def test_validate_design_rejects_duffing(
        self,
    ):
        config = valid_config()

        config[
            "confirmatory_design"
        ][
            "systems"
        ] = [
            "duffing",
            "vanderpol",
        ]

        with self.assertRaisesRegex(
            ValueError,
            "only Van der Pol",
        ):
            module._validate_design(
                config
            )


    def test_representation_settings_are_exact(
        self,
    ):
        settings = (
            module._representation_settings(
                valid_config()
            )
        )

        self.assertEqual(
            settings[
                "omega"
            ],
            128.0,
        )

        self.assertEqual(
            settings[
                "tau"
            ],
            1e-6,
        )

        self.assertEqual(
            settings[
                "train_seeds"
            ],
            [
                0,
                1,
                2,
            ],
        )

        self.assertEqual(
            settings[
                "n_steps_train"
            ],
            1200,
        )

        self.assertEqual(
            settings[
                "beta"
            ],
            0.1,
        )

        self.assertEqual(
            settings[
                "nlms_epochs"
            ],
            20,
        )


    def test_representation_settings_reject_design_drift(
        self,
    ):
        cases = (
            (
                "omega",
                64.0,
            ),
            (
                "operator_refit_uses_training_data_only",
                False,
            ),
            (
                "nlms_shuffle",
                True,
            ),
            (
                "nlms_initial_matrix",
                "zeros",
            ),
            (
                "project_stochastic",
                True,
            ),
        )

        for field, bad_value in cases:
            with self.subTest(
                field=field,
                bad_value=bad_value,
            ):
                config = valid_config()

                config[
                    "confirmatory_representation"
                ][
                    field
                ] = bad_value

                with self.assertRaises(
                    ValueError
                ):
                    module._representation_settings(
                        config
                    )


    def test_metric_helpers_match_direct_formulas(
        self,
    ):
        phi = np.array(
            [
                [
                    1.0,
                    0.5,
                    0.0,
                ],
                [
                    0.0,
                    0.5,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        target = np.array(
            [
                [
                    0.8,
                    0.4,
                    0.1,
                ],
                [
                    0.2,
                    0.6,
                    0.9,
                ],
            ],
            dtype=np.float64,
        )

        prediction = (
            0.9
            * phi
        )

        expected_error = (
            float(
                np.sum(
                    (
                        target
                        - prediction
                    ) ** 2
                )
            )
            / float(
                np.sum(
                    target
                    * target
                )
            )
        )

        centered = (
            target
            - target.mean(
                axis=1,
                keepdims=True,
            )
        )

        expected_r2 = (
            1.0
            - float(
                np.sum(
                    (
                        target
                        - prediction
                    ) ** 2
                )
            )
            / float(
                np.sum(
                    centered
                    * centered
                )
            )
        )

        self.assertAlmostEqual(
            module._relative_closure_error(
                prediction,
                target,
            ),
            expected_error,
            places=15,
        )

        self.assertAlmostEqual(
            module._association_r2(
                prediction,
                target,
            ),
            expected_r2,
            places=15,
        )


    def test_derivation_uses_training_only_source_abstraction(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(
                tmp
            )

            config = valid_config()

            source_artifact = (
                tmp_path
                / "source"
            )

            source_artifact.mkdir(
                parents=True
            )

            (
                source_artifact
                / "manifest.json"
            ).write_text(
                "{}",
                encoding="utf-8",
            )

            autoencoder_dir = (
                source_artifact
                / "autoencoders"
                / "000"
            )

            autoencoder_dir.mkdir(
                parents=True
            )

            classifier_entries = []

            for index in range(
                25
            ):
                name = (
                    f"ae_cluster_{index + 1:05d}.joblib"
                )

                path = (
                    autoencoder_dir
                    / name
                )

                path.write_bytes(
                    b"test"
                )

                classifier_entries.append(
                    f"000/{name}"
                )

            abstraction = {
                "classifier":
                    classifier_entries,

                "classifier_dir":
                    str(
                        source_artifact
                        / "autoencoders"
                    ),

                "model_id":
                    None,

                "cluster_centers_init":
                    np.zeros(
                        (
                            2,
                            25,
                        ),
                        dtype=np.float64,
                    ),

                "cluster_centers":
                    np.zeros(
                        (
                            2,
                            25,
                        ),
                        dtype=np.float64,
                    ),

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

                "kahkm_omega":
                    4.0,

                "kahkm_tau":
                    1e-6,
            }

            source_manifest = {
                "training_metadata": {
                    "train_seeds": [
                        0,
                        1,
                        2,
                    ],
                    "n_steps_train":
                        1200,
                    "beta":
                        0.1,
                    "nlms_epochs":
                        20,
                    "dt":
                        0.02,
                    "vanderpol_mu":
                        1.0,
                },
            }

            source_loaded = (
                SimpleNamespace(
                    system=
                        "vanderpol",

                    format_id=
                        module.FORMAT_ID,

                    omega=
                        4.0,

                    tau=
                        1e-6,

                    abstraction_model=
                        abstraction,

                    manifest=
                        source_manifest,
                )
            )

            restored = (
                SimpleNamespace(
                    archive_sha256=
                        module.EXPECTED_SOURCE_ARCHIVE_SHA256,

                    configuration_sha256=
                        module.EXPECTED_SOURCE_CONFIG_SHA256,

                    build_execution_commit=
                        module.EXPECTED_SOURCE_BUILD_COMMIT,

                    verifier_source_commit=
                        module.EXPECTED_SOURCE_VERIFIER_COMMIT,

                    archived_verification_status=
                        "passed",

                    vanderpol_artifact_dir=
                        str(
                            source_artifact
                        ),
                )
            )

            training_calls = []
            association_calls = []
            saved = {}
            load_count = {
                "value":
                    0,
            }

            def fake_load(
                artifact_dir,
                *,
                verify_hashes=True,
            ):
                self.assertTrue(
                    verify_hashes
                )

                load_count[
                    "value"
                ] += 1

                if load_count[
                    "value"
                ] == 1:
                    return source_loaded

                artifact = Path(
                    artifact_dir
                )

                manifest = json.loads(
                    (
                        artifact
                        / "manifest.json"
                    ).read_text(
                        encoding="utf-8"
                    )
                )

                operator = np.load(
                    artifact
                    / "operator.npy",
                    allow_pickle=False,
                )

                return SimpleNamespace(
                    system=
                        "vanderpol",

                    format_id=
                        module.FORMAT_ID,

                    omega=
                        128.0,

                    tau=
                        1e-6,

                    B=
                        operator,

                    manifest=
                        manifest,
                )

            def fake_training(
                system,
                *,
                train_seeds,
                n_steps_train,
                dt,
                vanderpol_mu,
            ):
                training_calls.append(
                    (
                        system,
                        train_seeds,
                        n_steps_train,
                        dt,
                        vanderpol_mu,
                    )
                )

                x0 = np.zeros(
                    (
                        2,
                        3600,
                    ),
                    dtype=np.float64,
                )

                x1 = np.ones(
                    (
                        2,
                        3600,
                    ),
                    dtype=np.float64,
                )

                return (
                    x0,
                    x1,
                )

            def fake_associations(
                model,
                X,
                *,
                omega,
                tau,
                n_jobs,
                batch_size,
                show_progress,
            ):
                association_calls.append(
                    (
                        float(
                            omega
                        ),
                        float(
                            model[
                                "kahkm_omega"
                            ]
                        ),
                        X.shape,
                    )
                )

                n_samples = (
                    X.shape[1]
                )

                out = np.full(
                    (
                        25,
                        n_samples,
                    ),
                    1.0 / 25.0,
                    dtype=np.float64,
                )

                alternating = (
                    (
                        np.arange(
                            n_samples
                        )
                        % 2
                    )
                    * 2.0
                    - 1.0
                )

                amplitude = (
                    0.005
                    if float(
                        np.mean(
                            X
                        )
                    ) < 0.5
                    else 0.010
                )

                out[
                    0,
                    :
                ] += (
                    amplitude
                    * alternating
                )

                out[
                    1,
                    :
                ] -= (
                    amplitude
                    * alternating
                )

                return out

            def fake_nlms(
                Phi,
                Chi,
                *,
                beta,
                epochs,
                shuffle,
                random_state,
                B0,
            ):
                self.assertEqual(
                    Phi.shape,
                    (
                        25,
                        3600,
                    ),
                )

                self.assertEqual(
                    Chi.shape,
                    (
                        25,
                        3600,
                    ),
                )

                self.assertEqual(
                    beta,
                    0.1,
                )

                self.assertEqual(
                    epochs,
                    20,
                )

                self.assertFalse(
                    shuffle
                )

                self.assertEqual(
                    random_state,
                    0,
                )

                self.assertIsNone(
                    B0
                )

                return (
                    np.eye(
                        25,
                        dtype=np.float64,
                    ),
                    tuple(
                        float(
                            index + 1
                        )
                        for index in range(
                            20
                        )
                    ),
                )

            def fake_save(
                *,
                output_dir,
                system,
                abstraction_model,
                B,
                train_closure_error,
                association_r2,
                nlms_history,
                training_metadata,
            ):
                destination = Path(
                    output_dir
                )

                destination.mkdir(
                    parents=True,
                    exist_ok=False,
                )

                np.save(
                    destination
                    / "operator.npy",
                    B,
                    allow_pickle=False,
                )

                manifest = {
                    "training_metadata":
                        dict(
                            training_metadata
                        ),
                }

                (
                    destination
                    / "manifest.json"
                ).write_text(
                    json.dumps(
                        manifest
                    ),
                    encoding="utf-8",
                )

                saved.update(
                    {
                        "system":
                            system,

                        "omega":
                            abstraction_model[
                                "kahkm_omega"
                            ],

                        "training_metadata":
                            dict(
                                training_metadata
                            ),

                        "history":
                            tuple(
                                nlms_history
                            ),

                        "train_closure_error":
                            train_closure_error,

                        "association_r2":
                            association_r2,
                    }
                )

                return destination

            output = (
                tmp_path
                / "derived"
            )

            work = (
                tmp_path
                / "work"
            )

            with (
                mock.patch.object(
                    module,
                    "_load_confirmatory_config",
                    return_value=config,
                ),
                mock.patch.object(
                    module,
                    "restore_repository_frozen_representations",
                    return_value=restored,
                ),
                mock.patch.object(
                    module,
                    "load_frozen_representation",
                    side_effect=fake_load,
                ),
                mock.patch.object(
                    module,
                    "make_concatenated_training_data",
                    side_effect=fake_training,
                ),
                mock.patch.object(
                    module,
                    "kahm_associations",
                    side_effect=fake_associations,
                ),
                mock.patch.object(
                    module,
                    "nlms_koopman_closure",
                    side_effect=fake_nlms,
                ),
                mock.patch.object(
                    module,
                    "save_frozen_representation",
                    side_effect=fake_save,
                ),
            ):
                result = (
                    module.derive_confirmatory_representation(
                        repository_root=
                            tmp_path,

                        output_dir=
                            output,

                        work_dir=
                            work,
                    )
                )

            self.assertEqual(
                training_calls,
                [
                    (
                        "vanderpol",
                        (
                            0,
                            1,
                            2,
                        ),
                        1200,
                        0.02,
                        1.0,
                    )
                ],
            )

            self.assertEqual(
                len(
                    association_calls
                ),
                2,
            )

            self.assertTrue(
                all(
                    call[
                        0
                    ] == 128.0
                    and call[
                        1
                    ] == 128.0
                    for call in association_calls
                )
            )

            self.assertEqual(
                saved[
                    "system"
                ],
                "vanderpol",
            )

            self.assertEqual(
                saved[
                    "omega"
                ],
                128.0,
            )

            metadata = saved[
                "training_metadata"
            ]

            self.assertEqual(
                metadata[
                    "derivation_id"
                ],
                module.DERIVATION_ID,
            )

            self.assertEqual(
                metadata[
                    "source_omega"
                ],
                4.0,
            )

            self.assertEqual(
                metadata[
                    "omega"
                ],
                128.0,
            )

            self.assertFalse(
                metadata[
                    "refit_kmeans"
                ]
            )

            self.assertFalse(
                metadata[
                    "refit_autoencoders"
                ]
            )

            self.assertEqual(
                metadata[
                    "predictor_orientation"
                ],
                "B.T @ Phi",
            )

            self.assertEqual(
                result.system,
                "vanderpol",
            )

            self.assertEqual(
                result.omega,
                128.0,
            )

            self.assertEqual(
                result.n_train_snapshots,
                3600,
            )

            self.assertEqual(
                result.n_clusters,
                25,
            )

            self.assertAlmostEqual(
                result.nlms_spectral_norm,
                1.0,
                places=15,
            )

            self.assertFalse(
                work.exists()
            )


    def test_derivation_removes_partial_output_on_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(
                tmp
            )

            config = valid_config()

            source_artifact = (
                tmp_path
                / "source"
            )

            source_artifact.mkdir()

            (
                source_artifact
                / "manifest.json"
            ).write_text(
                "{}",
                encoding="utf-8",
            )

            source_loaded = (
                SimpleNamespace(
                    system=
                        "vanderpol",

                    format_id=
                        module.FORMAT_ID,

                    omega=
                        4.0,

                    tau=
                        1e-6,

                    abstraction_model={
                        "n_clusters":
                            25,
                    },

                    manifest={
                        "training_metadata": {
                            "train_seeds": [
                                0,
                                1,
                                2,
                            ],
                            "n_steps_train":
                                1200,
                            "beta":
                                0.1,
                            "nlms_epochs":
                                20,
                            "dt":
                                0.02,
                            "vanderpol_mu":
                                1.0,
                        },
                    },
                )
            )

            restored = (
                SimpleNamespace(
                    archive_sha256=
                        module.EXPECTED_SOURCE_ARCHIVE_SHA256,

                    configuration_sha256=
                        module.EXPECTED_SOURCE_CONFIG_SHA256,

                    build_execution_commit=
                        module.EXPECTED_SOURCE_BUILD_COMMIT,

                    verifier_source_commit=
                        module.EXPECTED_SOURCE_VERIFIER_COMMIT,

                    archived_verification_status=
                        "passed",

                    vanderpol_artifact_dir=
                        str(
                            source_artifact
                        ),
                )
            )

            output = (
                tmp_path
                / "derived"
            )

            work = (
                tmp_path
                / "work"
            )

            def fail_associations(
                *_args,
                **_kwargs,
            ):
                output.mkdir(
                    exist_ok=True
                )

                raise RuntimeError(
                    "synthetic failure"
                )

            with (
                mock.patch.object(
                    module,
                    "_load_confirmatory_config",
                    return_value=config,
                ),
                mock.patch.object(
                    module,
                    "restore_repository_frozen_representations",
                    return_value=restored,
                ),
                mock.patch.object(
                    module,
                    "load_frozen_representation",
                    return_value=source_loaded,
                ),
                mock.patch.object(
                    module,
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
                    module,
                    "kahm_associations",
                    side_effect=fail_associations,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic failure",
                ):
                    module.derive_confirmatory_representation(
                        repository_root=
                            tmp_path,

                        output_dir=
                            output,

                        work_dir=
                            work,
                    )

            self.assertFalse(
                output.exists()
            )

            self.assertFalse(
                work.exists()
            )


if __name__ == "__main__":
    unittest.main()
