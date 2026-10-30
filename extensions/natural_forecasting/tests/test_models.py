"""Small synthetic checks; no natural-data performance is evaluated here."""
from __future__ import annotations

import copy
import json
import pickle
import unittest
from unittest import mock

import numpy as np

from extensions.natural_forecasting.models import (
    ForecastRegressor, LocalFeatureBank, _stable_kahm_reconstruction,
)
from extensions.predictive_geometry.study import geometry as historical_geometry
from extensions.predictive_geometry.study import heads as historical_heads


class StandaloneModelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(841793)
        self.x = self.rng.normal(size=(96, 5)) * np.arange(1, 6) + np.arange(5)
        self.query = self.rng.normal(size=(17, 5)) * np.arange(1, 6) + np.arange(5)
        self.parameters = dict(seed=13, cluster_counts=(2, 4), rank=2,
                               max_refs=12, max_fit=64, omega=16.0, tau=1e-6)

    def test_historical_kahm_and_independently_fitted_affine_features_match(self):
        historical = historical_geometry.GeometryBank(**self.parameters).fit(self.x)
        expected = historical.transform(self.query)
        for gate, name in (("kahm", "kahm:local"), ("affine", "affine:local")):
            with self.subTest(gate=gate):
                independent = LocalFeatureBank(gate, query_mode="historical", **self.parameters).fit(self.x)
                actual = independent.transform(self.query)
                np.testing.assert_allclose(actual, expected[name], rtol=1e-11, atol=1e-12)
                old_models = [m for block in historical.blocks_ for m in block["models"]]
                for new, old in zip(independent.models_, old_models):
                    np.testing.assert_array_equal(new.reference_indices, old.reference_indices)
                    np.testing.assert_array_equal(new.references, old.references)
                    np.testing.assert_allclose(new.basis, old.basis, rtol=0, atol=0)

    def test_affine_and_locality_paths_never_construct_or_query_kahm(self):
        original = historical_geometry._upstream
        forbidden = {"autoencoder", "kxx_matrix", "kxa_matrix", "kernel_regularized_least_squares"}

        def guarded(name):
            if name in forbidden:
                raise AssertionError(f"Non-KAHM pipeline attempted to load {name}")
            return original(name)

        for gate in ("affine", "locality"):
            with self.subTest(gate=gate), mock.patch.object(historical_geometry, "_upstream", guarded):
                bank = LocalFeatureBank(gate, **self.parameters).fit(self.x)
                features, diagnostics = bank.transform(self.query, diagnostics=True)
                self.assertTrue(np.isfinite(features).all())
                self.assertEqual(bank.metadata()["fitted_kahm_kernels"], 0)
                self.assertEqual(diagnostics["kernel_query_diagnostics"]["weight_rows"], 0)
                self.assertTrue(all(model.ae is None for model in bank.models_))

    def test_locality_gates_and_coordinate_augmentation_match_declared_control(self):
        historical = historical_geometry.GeometryBank(**self.parameters).fit(self.x)
        expected_phi = historical.transform(self.query)["native_affine:phi"]
        z = historical._scale_inputs(self.query)
        models = [m for block in historical.blocks_ for m in block["models"]]
        expected_coordinates = [expected_phi[:, c:c + 1] * ((z - m.center) @ m.basis)
                                for c, m in enumerate(models)]
        expected = np.concatenate([expected_phi] + expected_coordinates, axis=1)
        bank = LocalFeatureBank("locality", **self.parameters).fit(self.x)
        np.testing.assert_allclose(bank.transform(self.query), expected, rtol=1e-10, atol=1e-12)
        start = 0
        for count in bank.cluster_counts:
            np.testing.assert_allclose(expected_phi[:, start:start + count].sum(axis=1), 1.0)
            start += count

    def test_stable_queries_match_nonunderflow_arithmetic_and_fix_row_underflow(self):
        references = self.rng.normal(size=(4, 24))
        ae = historical_geometry._upstream("autoencoder").autoencoder(references, 2)
        query = self.rng.normal(scale=0.5, size=(1003, 4))
        old, old_stats = historical_geometry._kahm_reconstruction(query, ae, True)
        stable, stable_stats = _stable_kahm_reconstruction(query, ae, True)
        self.assertEqual(old_stats["zero_normalizers"], 0)
        self.assertEqual(stable_stats["zero_normalizers"], 0)
        np.testing.assert_allclose(stable, old, rtol=1e-9, atol=1e-11)
        far = np.broadcast_to(references.mean(axis=1) + 1000 * ae["PC"][:, 0], (7, 4)).copy()
        _, old_far = historical_geometry._kahm_reconstruction(far, ae, True)
        stable_far, new_far = _stable_kahm_reconstruction(far, ae, True)
        self.assertEqual(old_far["zero_normalizers"], 7)
        self.assertEqual(new_far["original_all_zero_kernel_rows"], 7)
        self.assertEqual(new_far["zero_normalizers"], 0)
        self.assertTrue(np.isfinite(stable_far).all())

    def test_fixed_heads_match_historical_predictors_with_target_scaling(self):
        y = np.column_stack((100 + 0.03 * self.x[:, 0] - 0.2 * self.x[:, 1],
                             -50 + 20 * np.sin(self.x[:, 2])))
        validation_x = self.query[:7]
        validation_y = np.zeros((7, 2))  # One fixed candidate; cannot select using these outcomes.
        for head in ("ridge", "rbf"):
            with self.subTest(head=head):
                model = ForecastRegressor("raw", head, alpha=0.01, n_landmarks=12,
                                           bandwidth_multiplier=1.0, head_seed=39).fit(self.x, y)
                train_features = model.transform_features(self.x)
                val_features = model.transform_features(validation_x)
                query_features = model.transform_features(self.query)
                expected, _, _ = historical_heads.fit_predict(
                    train_features, y, val_features, validation_y, query_features,
                    head=head, alphas=(0.01,), n_landmarks=12,
                    bandwidth_multipliers=(1.0,), seed=39)
                np.testing.assert_allclose(model.predict(self.query), expected, rtol=1e-9, atol=1e-10)

    def test_training_only_scales_are_frozen_and_predictions_keep_target_units(self):
        y = np.column_stack((1000 + 100 * self.x[:, 0], np.full(len(self.x), -7.0)))
        model = ForecastRegressor("raw", "rbf", alpha=0.01, n_landmarks=12).fit(self.x, y)
        np.testing.assert_allclose(model.input_mean_, self.x.mean(axis=0))
        np.testing.assert_allclose(model.target_mean_, y.mean(axis=0))
        self.assertEqual(model.target_scale_[1], 1.0)
        arrays = [model.input_mean_, model.input_scale_, model.target_mean_, model.target_scale_,
                  model.basis_.mean, model.basis_.scale, model.basis_mean_, model.basis_scale_]
        originals = [array.copy() for array in arrays]
        prediction = model.predict(self.query + 500)
        self.assertEqual(prediction.shape, (17, 2))
        np.testing.assert_allclose(prediction[:, 1], -7.0, atol=1e-12)
        for original, after in zip(originals, arrays):
            np.testing.assert_array_equal(original, after)
        single = ForecastRegressor().fit(self.x, y[:, 0])
        self.assertEqual(single.predict(self.query).shape, (17,))
        with self.assertRaisesRegex(ValueError, "row counts"):
            ForecastRegressor().fit(self.x, y[:-1])

    def test_high_dimensional_raw_skip_retains_every_raw_coordinate(self):
        x = self.rng.normal(size=(128, 40))
        y = np.column_stack((x[:, 35], x[:, 39]))
        settings = dict(max_fit=96, max_refs=12, cluster_counts=(2, 4, 8), rank=2)
        model = ForecastRegressor("affine_local", "ridge", raw_skip=True,
                                   geometry_parameters=settings, seed=7).fit(x, y)
        features = model.transform_features(x)
        np.testing.assert_allclose(features[:, :40] * model.input_scale_ + model.input_mean_, x)
        metadata = model.metadata()
        self.assertLessEqual(metadata["geometry"]["stack"]["numerical_rank"], 28)
        self.assertFalse(metadata["geometry"]["stack"]["full_numerical_column_rank"])
        self.assertTrue(metadata["raw_inputs_explicitly_retained"])
        realized_coordinates = metadata["geometry"]["projected_coordinates"]
        self.assertLessEqual(realized_coordinates, 28)
        self.assertEqual(features.shape[1], 40 + 14 + realized_coordinates)
        self.assertEqual(metadata["feature_names"][:40], [f"raw_{j}" for j in range(40)])

    def test_metadata_records_actual_head_shapes_and_is_strict_json(self):
        model = ForecastRegressor("kahm_local", "rbf", n_landmarks=11,
                                   geometry_parameters={"cluster_counts": (2,), "max_refs": 10,
                                                        "max_fit": 64}).fit(self.x, self.x[:, :2])
        before = copy.deepcopy(model.metadata())
        serialized = json.dumps(before, allow_nan=False)
        self.assertTrue(serialized)
        self.assertEqual(before["actual_landmarks"], 11)
        self.assertEqual(before["basis_width"], before["feature_width"] + before["retained_nystrom_rank"])
        self.assertEqual(before["coefficient_shape"], [before["basis_width"], 2])
        self.assertEqual(before["decoder_coefficient_count"], 2 * (before["basis_width"] + 1))
        model.predict(self.query)
        self.assertEqual(before, model.metadata())
        with self.assertRaises(ValueError):
            ForecastRegressor("raw", raw_skip=True)
        with self.assertRaises(RuntimeError):
            ForecastRegressor().predict(self.query)

    def test_alpha_path_matches_independent_fits_and_builds_each_method_once(self):
        # Unequal target scales make selection in original target units observable.
        y = np.column_stack((200 + 30 * np.sin(self.x[:, 0]) + self.x[:, 1],
                             -4 + 0.02 * self.x[:, 2]))
        validation_y = np.column_stack((200 + 30 * np.sin(self.query[:, 0]) + self.query[:, 1],
                                        -4 + 0.02 * self.query[:, 2]))
        alphas = (0.01, 1.0, 100.0)
        cases = [("raw", "ridge", False), ("raw", "rbf", False)]
        cases += [(representation, "rbf", skip)
                  for representation in ("affine_local", "locality_local", "kahm_local")
                  for skip in (False, True)]
        for representation, head, raw_skip in cases:
            with self.subTest(representation=representation, head=head, raw_skip=raw_skip):
                kwargs = dict(representation=representation, head=head, raw_skip=raw_skip,
                              n_landmarks=12, seed=13, head_seed=39,
                              geometry_parameters={k: v for k, v in self.parameters.items() if k != "seed"})
                independent = [ForecastRegressor(alpha=alpha, **kwargs).fit(self.x, y) for alpha in alphas]
                losses = [float(np.mean((model.predict(self.query) - validation_y) ** 2))
                          for model in independent]
                selected = ForecastRegressor(**kwargs)
                original_fit, original_transform = LocalFeatureBank.fit, LocalFeatureBank.transform
                with mock.patch.object(LocalFeatureBank, "fit", autospec=True,
                                       side_effect=original_fit) as bank_fit, \
                     mock.patch.object(LocalFeatureBank, "transform", autospec=True,
                                       side_effect=original_transform) as bank_transform, \
                     mock.patch.object(historical_heads, "_bases", wraps=historical_heads._bases) as bases, \
                     mock.patch.object(selected, "_fit_system", wraps=selected._fit_system) as system:
                    trials = selected.fit_select(self.x, y, self.query, validation_y, alphas)
                    self.assertEqual(bank_fit.call_count, 0 if representation == "raw" else 1)
                    self.assertEqual(bank_transform.call_count, 0 if representation == "raw" else 2)
                    self.assertEqual(bases.call_count, 1)
                    self.assertEqual(system.call_count, 1)
                self.assertEqual([trial["alpha"] for trial in trials], list(alphas))
                np.testing.assert_allclose([trial["validation_mse"] for trial in trials], losses,
                                           rtol=1e-13, atol=1e-13)
                expected = independent[int(np.argmin(losses))]
                self.assertEqual(selected.alpha, expected.alpha)
                np.testing.assert_allclose(selected.predict(self.query), expected.predict(self.query),
                                           rtol=1e-13, atol=1e-13)
                # The fitted object's state has no extra training designs or search caches.
                self.assertEqual(set(selected.__dict__), set(expected.__dict__))
                selected_metadata, expected_metadata = selected.metadata(), expected.metadata()
                self.assertIn("validation selection", selected_metadata.pop("tuning_scope"))
                expected_metadata.pop("tuning_scope")
                self.assertEqual(selected_metadata, expected_metadata)
                restored = pickle.loads(pickle.dumps(selected))
                np.testing.assert_array_equal(restored.predict(self.query), selected.predict(self.query))
                json.dumps(trials, allow_nan=False)

    def test_alpha_path_ties_single_target_and_invalid_candidates(self):
        y = np.full(len(self.x), 17.0)
        validation_y = np.full(len(self.query), 17.0)
        model = ForecastRegressor("raw", "ridge")
        trials = model.fit_select(self.x, y, self.query, validation_y, (100.0, 0.01, 1.0))
        self.assertEqual(model.alpha, 100.0)
        self.assertEqual([trial["validation_mse"] for trial in trials], [0.0, 0.0, 0.0])
        self.assertEqual(model.predict(self.query).shape, (len(self.query),))
        for alphas in ((), (0.0,), (-1.0,), (float("nan"),), (float("inf"),)):
            with self.subTest(alphas=alphas), self.assertRaisesRegex(ValueError, "alphas"):
                model.fit_select(self.x, y, self.query, validation_y, alphas)
        with self.assertRaisesRegex(ValueError, "Validation target dimension"):
            model.fit_select(self.x, y, self.query, np.column_stack((validation_y, validation_y)), (1.0,))
        model.fit(self.x, y)
        self.assertIn("fixed configuration", model.metadata()["tuning_scope"])


if __name__ == "__main__":
    unittest.main()
