"""Tests for configuration-driven frozen dynamical representations.

These tests mock the expensive KAHM fitting operation. No real representation
is trained and no certificate-evaluation data are generated.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import kahkm_dynamical_representation as module
from kahkm_dynamical_representation import (
    DYNAMICAL_PILOT_CONFIG_RELATIVE,
    DYNAMICAL_PILOT_CONFIG_SHA256,
    _system_training_settings,
    fit_and_freeze_dynamical_representation,
    load_frozen_dynamical_pilot_config,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


class DynamicalRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_root = Path(
            self.temporary.name
        )

    def tearDown(self):
        self.temporary.cleanup()

    def frozen_config(self):
        return load_frozen_dynamical_pilot_config(
            repository_root=PROJECT_ROOT
        )

    def fake_training_data(
        self,
        n_steps=1200,
    ):
        n = 3 * int(n_steps)

        x0 = np.linspace(
            -1.0,
            1.0,
            2 * n,
            dtype=np.float64,
        ).reshape(2, n)

        x1 = x0 + 0.001

        return x0, x1

    def fake_fit(
        self,
        *,
        n_clusters,
        omega,
        tau,
        B=None,
        effective_clusters=None,
    ):
        if B is None:
            B = np.eye(
                n_clusters,
                dtype=np.float64,
            )

        if effective_clusters is None:
            effective_clusters = n_clusters

        abstraction_model = {
            "n_clusters":
                int(
                    effective_clusters
                ),
            "kahkm_omega":
                float(omega),
            "kahkm_tau":
                float(tau),
        }

        return SimpleNamespace(
            abstraction_model=
                abstraction_model,
            B=np.asarray(
                B,
                dtype=np.float64,
            ),
            train_closure_error=0.125,
            association_r2=0.875,
            nlms_history=(
                0.5,
                0.25,
                0.125,
            ),
        )

    def run_mock_build(
        self,
        system,
        *,
        B=None,
        effective_clusters=None,
    ):
        config = self.frozen_config()
        settings = _system_training_settings(
            config,
            system,
        )

        x0, x1 = self.fake_training_data(
            settings[
                "n_steps_train"
            ]
        )

        fit = self.fake_fit(
            n_clusters=
                settings[
                    "n_clusters"
                ],
            omega=settings[
                "omega"
            ],
            tau=settings[
                "tau"
            ],
            B=B,
            effective_clusters=
                effective_clusters,
        )

        output = (
            self.temp_root
            / f"{system}_artifact"
        )

        work = (
            self.temp_root
            / f"{system}_work"
        )

        with (
            mock.patch.object(
                module,
                "make_concatenated_training_data",
                return_value=(
                    x0,
                    x1,
                ),
            ) as make_data,
            mock.patch.object(
                module,
                "fit_kahkm",
                return_value=fit,
            ) as fit_mock,
            mock.patch.object(
                module,
                "save_frozen_representation",
            ) as save_mock,
        ):
            # Mimic the persistence function creating the retained artifact.
            save_mock.side_effect = (
                lambda **kwargs:
                    Path(
                        kwargs[
                            "output_dir"
                        ]
                    ).mkdir(
                        parents=True,
                        exist_ok=False,
                    )
                    or Path(
                        kwargs[
                            "output_dir"
                        ]
                    )
            )

            result = (
                fit_and_freeze_dynamical_representation(
                    repository_root=
                        PROJECT_ROOT,
                    system=system,
                    output_dir=output,
                    work_dir=work,
                )
            )

        return (
            result,
            output,
            work,
            make_data,
            fit_mock,
            save_mock,
            settings,
        )

    # 1
    def test_committed_configuration_hash_is_exact(self):
        path = (
            PROJECT_ROOT
            / DYNAMICAL_PILOT_CONFIG_RELATIVE
        )

        actual = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        self.assertEqual(
            actual,
            DYNAMICAL_PILOT_CONFIG_SHA256,
        )

        self.assertEqual(
            actual,
            "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1",
        )

    # 2
    def test_configuration_loader_returns_expected_campaign(self):
        config = self.frozen_config()

        self.assertEqual(
            config[
                "campaign_id"
            ],
            "dynamical_original_iid_pilot_v1",
        )

        self.assertEqual(
            config[
                "campaign_role"
            ],
            "pilot",
        )

    # 3
    def test_duffing_frozen_settings_do_not_equal_experiment15_defaults(self):
        settings = _system_training_settings(
            self.frozen_config(),
            "duffing",
        )

        self.assertEqual(
            settings[
                "n_clusters"
            ],
            10,
        )

        self.assertEqual(
            settings[
                "omega"
            ],
            2.0,
        )

        # Guard specifically against accidentally using the historical
        # Experiment 15 defaults C=15, omega=12.
        self.assertNotEqual(
            (
                settings[
                    "n_clusters"
                ],
                settings[
                    "omega"
                ],
            ),
            (
                15,
                12.0,
            ),
        )

    # 4
    def test_vanderpol_frozen_settings_are_retained_tuned_values(self):
        settings = _system_training_settings(
            self.frozen_config(),
            "vanderpol",
        )

        self.assertEqual(
            settings[
                "n_clusters"
            ],
            25,
        )

        self.assertEqual(
            settings[
                "omega"
            ],
            4.0,
        )

        # Guard against Experiment 15's C=20 default.
        self.assertNotEqual(
            settings[
                "n_clusters"
            ],
            20,
        )

    # 5
    def test_common_frozen_training_settings(self):
        for system in (
            "duffing",
            "vanderpol",
        ):
            with self.subTest(
                system=system
            ):
                settings = _system_training_settings(
                    self.frozen_config(),
                    system,
                )

                self.assertEqual(
                    settings[
                        "train_seeds"
                    ],
                    (
                        0,
                        1,
                        2,
                    ),
                )

                self.assertEqual(
                    settings[
                        "n_steps_train"
                    ],
                    1200,
                )

                self.assertEqual(
                    settings[
                        "subspace_dim"
                    ],
                    4,
                )

                self.assertEqual(
                    settings[
                        "Nb"
                    ],
                    100,
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

                self.assertEqual(
                    settings[
                        "kmeans_kind"
                    ],
                    "full",
                )

                self.assertIsNone(
                    settings[
                        "max_train_per_cluster"
                    ]
                )

                self.assertEqual(
                    settings[
                        "random_state"
                    ],
                    0,
                )

                self.assertEqual(
                    settings[
                        "batch_size"
                    ],
                    256,
                )

                self.assertEqual(
                    settings[
                        "n_jobs"
                    ],
                    -1,
                )

    # 6
    def test_system_specific_dynamics_settings(self):
        duffing = _system_training_settings(
            self.frozen_config(),
            "duffing",
        )

        vdp = _system_training_settings(
            self.frozen_config(),
            "vanderpol",
        )

        self.assertEqual(
            duffing[
                "dt"
            ],
            0.03,
        )

        self.assertEqual(
            vdp[
                "dt"
            ],
            0.02,
        )

        self.assertEqual(
            duffing[
                "vanderpol_mu"
            ],
            1.0,
        )

        self.assertEqual(
            vdp[
                "vanderpol_mu"
            ],
            1.0,
        )

    # 7
    def test_duffing_build_passes_exact_frozen_training_arguments(self):
        (
            result,
            output,
            work,
            make_data,
            fit_mock,
            _save_mock,
            settings,
        ) = self.run_mock_build(
            "duffing"
        )

        make_data.assert_called_once_with(
            "duffing",
            train_seeds=(
                0,
                1,
                2,
            ),
            n_steps_train=1200,
            dt=0.03,
            vanderpol_mu=1.0,
        )

        _, kwargs = (
            fit_mock.call_args
        )

        self.assertEqual(
            kwargs[
                "n_clusters"
            ],
            10,
        )
        self.assertEqual(
            kwargs[
                "omega"
            ],
            2.0,
        )
        self.assertEqual(
            kwargs[
                "tau"
            ],
            1e-6,
        )
        self.assertEqual(
            kwargs[
                "subspace_dim"
            ],
            4,
        )
        self.assertEqual(
            kwargs[
                "Nb"
            ],
            100,
        )
        self.assertEqual(
            kwargs[
                "beta"
            ],
            0.1,
        )
        self.assertEqual(
            kwargs[
                "nlms_epochs"
            ],
            20,
        )
        self.assertEqual(
            kwargs[
                "random_state"
            ],
            0,
        )
        self.assertEqual(
            kwargs[
                "kmeans_kind"
            ],
            "full",
        )
        self.assertIsNone(
            kwargs[
                "max_train_per_cluster"
            ]
        )
        self.assertIs(
            kwargs[
                "save_ae_to_disk"
            ],
            True,
        )
        self.assertIs(
            kwargs[
                "project_stochastic"
            ],
            False,
        )
        self.assertIs(
            kwargs[
                "preload_classifier_after_fit"
            ],
            False,
        )
        self.assertEqual(
            kwargs[
                "batch_size"
            ],
            256,
        )
        self.assertEqual(
            kwargs[
                "n_jobs"
            ],
            -1,
        )
        self.assertIs(
            kwargs[
                "verbose"
            ],
            False,
        )

        self.assertEqual(
            result.n_clusters,
            settings[
                "n_clusters"
            ],
        )
        self.assertTrue(
            output.is_dir()
        )
        self.assertFalse(
            work.exists()
        )

    # 8
    def test_vanderpol_build_passes_exact_frozen_training_arguments(self):
        (
            result,
            _output,
            _work,
            make_data,
            fit_mock,
            _save_mock,
            _settings,
        ) = self.run_mock_build(
            "vanderpol"
        )

        make_data.assert_called_once_with(
            "vanderpol",
            train_seeds=(
                0,
                1,
                2,
            ),
            n_steps_train=1200,
            dt=0.02,
            vanderpol_mu=1.0,
        )

        _, kwargs = fit_mock.call_args

        self.assertEqual(
            kwargs[
                "n_clusters"
            ],
            25,
        )

        self.assertEqual(
            kwargs[
                "omega"
            ],
            4.0,
        )

        self.assertEqual(
            result.n_clusters,
            25,
        )

    # 9
    def test_spectral_norm_is_computed_from_fitted_raw_B(self):
        B = np.diag(
            [
                0.5,
                2.0,
                1.25,
                0.75,
                0.1,
                0.2,
                0.3,
                0.4,
                0.6,
                0.9,
            ]
        )

        (
            result,
            _output,
            _work,
            _make_data,
            _fit_mock,
            save_mock,
            _settings,
        ) = self.run_mock_build(
            "duffing",
            B=B,
        )

        self.assertEqual(
            result.nlms_spectral_norm,
            2.0,
        )

        _, kwargs = save_mock.call_args

        self.assertEqual(
            kwargs[
                "training_metadata"
            ][
                "nlms_spectral_norm"
            ],
            2.0,
        )

        np.testing.assert_array_equal(
            kwargs[
                "B"
            ],
            B,
        )

    # 10
    def test_training_metadata_records_frozen_provenance(self):
        (
            _result,
            _output,
            _work,
            _make_data,
            _fit_mock,
            save_mock,
            _settings,
        ) = self.run_mock_build(
            "duffing"
        )

        metadata = save_mock.call_args.kwargs[
            "training_metadata"
        ]

        self.assertEqual(
            metadata[
                "configuration_relative_path"
            ],
            DYNAMICAL_PILOT_CONFIG_RELATIVE,
        )

        self.assertEqual(
            metadata[
                "configuration_sha256"
            ],
            DYNAMICAL_PILOT_CONFIG_SHA256,
        )

        self.assertEqual(
            metadata[
                "system"
            ],
            "duffing",
        )

        self.assertEqual(
            metadata[
                "train_seeds"
            ],
            [
                0,
                1,
                2,
            ],
        )

        self.assertEqual(
            metadata[
                "n_train_snapshots"
            ],
            3600,
        )

        self.assertEqual(
            metadata[
                "n_clusters"
            ],
            10,
        )

        self.assertEqual(
            metadata[
                "omega"
            ],
            2.0,
        )

        self.assertIs(
            metadata[
                "project_stochastic"
            ],
            False,
        )

    # 11
    def test_build_result_records_fit_diagnostics(self):
        (
            result,
            output,
            _work,
            _make_data,
            _fit_mock,
            _save_mock,
            _settings,
        ) = self.run_mock_build(
            "duffing"
        )

        self.assertEqual(
            result.system,
            "duffing",
        )

        self.assertEqual(
            result.artifact_dir,
            str(
                output.resolve()
            ),
        )

        self.assertEqual(
            result.training_snapshot_count,
            3600,
        )

        self.assertEqual(
            result.train_closure_error,
            0.125,
        )

        self.assertEqual(
            result.association_r2,
            0.875,
        )

        self.assertEqual(
            result.configuration_sha256,
            DYNAMICAL_PILOT_CONFIG_SHA256,
        )

    # 12
    def test_existing_output_is_rejected_before_training(self):
        output = (
            self.temp_root
            / "artifact"
        )
        output.mkdir()

        work = (
            self.temp_root
            / "work"
        )

        with (
            mock.patch.object(
                module,
                "make_concatenated_training_data",
            ) as make_data,
            mock.patch.object(
                module,
                "fit_kahkm",
            ) as fit_mock,
        ):
            with self.assertRaises(
                FileExistsError
            ):
                fit_and_freeze_dynamical_representation(
                    repository_root=
                        PROJECT_ROOT,
                    system="duffing",
                    output_dir=output,
                    work_dir=work,
                )

        make_data.assert_not_called()
        fit_mock.assert_not_called()

    # 13
    def test_existing_work_directory_is_rejected_before_training(self):
        output = (
            self.temp_root
            / "artifact"
        )

        work = (
            self.temp_root
            / "work"
        )
        work.mkdir()

        with (
            mock.patch.object(
                module,
                "make_concatenated_training_data",
            ) as make_data,
            mock.patch.object(
                module,
                "fit_kahkm",
            ) as fit_mock,
        ):
            with self.assertRaises(
                FileExistsError
            ):
                fit_and_freeze_dynamical_representation(
                    repository_root=
                        PROJECT_ROOT,
                    system="duffing",
                    output_dir=output,
                    work_dir=work,
                )

        make_data.assert_not_called()
        fit_mock.assert_not_called()

    # 14
    def test_output_and_work_must_differ(self):
        path = (
            self.temp_root
            / "same"
        )

        with self.assertRaisesRegex(
            ValueError,
            "must differ",
        ):
            fit_and_freeze_dynamical_representation(
                repository_root=
                    PROJECT_ROOT,
                system="duffing",
                output_dir=path,
                work_dir=path,
            )

    # 15
    def test_bad_training_data_shape_is_rejected_before_fit(self):
        bad_x0 = np.zeros(
            (2, 3599),
            dtype=np.float64,
        )
        bad_x1 = bad_x0.copy()

        with (
            mock.patch.object(
                module,
                "make_concatenated_training_data",
                return_value=(
                    bad_x0,
                    bad_x1,
                ),
            ),
            mock.patch.object(
                module,
                "fit_kahkm",
            ) as fit_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "training-data shape",
            ):
                fit_and_freeze_dynamical_representation(
                    repository_root=
                        PROJECT_ROOT,
                    system="duffing",
                    output_dir=
                        self.temp_root
                        / "artifact",
                    work_dir=
                        self.temp_root
                        / "work",
                )

        fit_mock.assert_not_called()

    # 16
    def test_nonfinite_training_data_is_rejected_before_fit(self):
        x0, x1 = self.fake_training_data()

        x0[0, 0] = np.nan

        with (
            mock.patch.object(
                module,
                "make_concatenated_training_data",
                return_value=(
                    x0,
                    x1,
                ),
            ),
            mock.patch.object(
                module,
                "fit_kahkm",
            ) as fit_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "nonfinite",
            ):
                fit_and_freeze_dynamical_representation(
                    repository_root=
                        PROJECT_ROOT,
                    system="duffing",
                    output_dir=
                        self.temp_root
                        / "artifact",
                    work_dir=
                        self.temp_root
                        / "work",
                )

        fit_mock.assert_not_called()

    # 17
    def test_wrong_operator_shape_cleans_work_and_output(self):
        config = self.frozen_config()
        settings = _system_training_settings(
            config,
            "duffing",
        )

        x0, x1 = self.fake_training_data()

        fit = self.fake_fit(
            n_clusters=10,
            omega=2.0,
            tau=1e-6,
            B=np.eye(
                9,
                dtype=np.float64,
            ),
        )

        output = (
            self.temp_root
            / "artifact"
        )
        work = (
            self.temp_root
            / "work"
        )

        with (
            mock.patch.object(
                module,
                "make_concatenated_training_data",
                return_value=(
                    x0,
                    x1,
                ),
            ),
            mock.patch.object(
                module,
                "fit_kahkm",
                return_value=fit,
            ),
            mock.patch.object(
                module,
                "save_frozen_representation",
            ) as save_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "NLMS matrix shape",
            ):
                fit_and_freeze_dynamical_representation(
                    repository_root=
                        PROJECT_ROOT,
                    system="duffing",
                    output_dir=output,
                    work_dir=work,
                )

        save_mock.assert_not_called()
        self.assertFalse(
            output.exists()
        )
        self.assertFalse(
            work.exists()
        )
        self.assertEqual(
            settings[
                "n_clusters"
            ],
            10,
        )

    # 18
    def test_effective_cluster_count_change_is_rejected(self):
        config = self.frozen_config()
        settings = _system_training_settings(
            config,
            "duffing",
        )

        x0, x1 = self.fake_training_data()

        fit = self.fake_fit(
            n_clusters=10,
            omega=2.0,
            tau=1e-6,
            effective_clusters=9,
        )

        output = (
            self.temp_root
            / "artifact"
        )
        work = (
            self.temp_root
            / "work"
        )

        with (
            mock.patch.object(
                module,
                "make_concatenated_training_data",
                return_value=(
                    x0,
                    x1,
                ),
            ),
            mock.patch.object(
                module,
                "fit_kahkm",
                return_value=fit,
            ),
            mock.patch.object(
                module,
                "save_frozen_representation",
            ) as save_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Effective fitted cluster count",
            ):
                fit_and_freeze_dynamical_representation(
                    repository_root=
                        PROJECT_ROOT,
                    system="duffing",
                    output_dir=output,
                    work_dir=work,
                )

        save_mock.assert_not_called()
        self.assertFalse(
            work.exists()
        )
        self.assertEqual(
            settings[
                "n_clusters"
            ],
            10,
        )

    # 19
    def test_export_failure_cleans_output_and_work(self):
        x0, x1 = self.fake_training_data()

        fit = self.fake_fit(
            n_clusters=10,
            omega=2.0,
            tau=1e-6,
        )

        output = (
            self.temp_root
            / "artifact"
        )
        work = (
            self.temp_root
            / "work"
        )

        def fail_export(
            **kwargs,
        ):
            destination = Path(
                kwargs[
                    "output_dir"
                ]
            )

            destination.mkdir(
                parents=True,
                exist_ok=False,
            )

            (
                destination
                / "partial.txt"
            ).write_text(
                "partial",
                encoding="utf-8",
            )

            raise OSError(
                "synthetic export failure"
            )

        with (
            mock.patch.object(
                module,
                "make_concatenated_training_data",
                return_value=(
                    x0,
                    x1,
                ),
            ),
            mock.patch.object(
                module,
                "fit_kahkm",
                return_value=fit,
            ),
            mock.patch.object(
                module,
                "save_frozen_representation",
                side_effect=fail_export,
            ),
        ):
            with self.assertRaisesRegex(
                OSError,
                "synthetic export failure",
            ):
                fit_and_freeze_dynamical_representation(
                    repository_root=
                        PROJECT_ROOT,
                    system="duffing",
                    output_dir=output,
                    work_dir=work,
                )

        self.assertFalse(
            output.exists()
        )

        self.assertFalse(
            work.exists()
        )

    # 20
    def test_malformed_configurations_are_rejected(self):
        base = self.frozen_config()

        cases = []

        changed = deepcopy(base)
        changed[
            "frozen_representation"
        ][
            "save_ae_to_disk"
        ] = False
        cases.append(
            changed
        )

        changed = deepcopy(base)
        changed[
            "frozen_representation"
        ][
            "project_stochastic"
        ] = True
        cases.append(
            changed
        )

        changed = deepcopy(base)
        changed[
            "frozen_representation"
        ][
            "train_seeds"
        ] = [
            0,
            0,
            2,
        ]
        cases.append(
            changed
        )

        changed = deepcopy(base)
        changed[
            "frozen_representation"
        ][
            "systems"
        ][
            "duffing"
        ][
            "n_clusters"
        ] = 0
        cases.append(
            changed
        )

        for index, config in enumerate(
            cases
        ):
            with self.subTest(
                case=index
            ):
                with self.assertRaises(
                    (
                        TypeError,
                        ValueError,
                    )
                ):
                    _system_training_settings(
                        config,
                        "duffing",
                    )


if __name__ == "__main__":
    unittest.main()
