"""Small algebra/provenance tests, not forecasting-performance experiments."""
from __future__ import annotations

import copy
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from extensions.forecasting_utility.models import AdditiveForecastRegressor, RawForecastRegressor
from extensions.natural_forecasting.models import ForecastRegressor, LocalFeatureBank
from extensions.predictive_geometry.study import geometry as historical_geometry


class AdditiveModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(530186)
        self.x = rng.normal(size=(48, 4)) * [1.0, 2.0, 3.0, 4.0] + [0.0, 3.0, -2.0, 8.0]
        self.query = rng.normal(size=(11, 4)) * [1.0, 2.0, 3.0, 4.0] + [0.0, 3.0, -2.0, 8.0]
        self.y = np.column_stack((30 + 2 * self.x[:, 0] + np.sin(self.x[:, 1]),
                                  -200 + 0.01 * self.x[:, 2] ** 2))
        self.y_query = np.column_stack((30 + 2 * self.query[:, 0] + np.sin(self.query[:, 1]),
                                        -200 + 0.01 * self.query[:, 2] ** 2))
        self.geometry = {"cluster_counts": (2, 4), "rank": 2,
                         "max_refs": 10, "max_fit": 40, "omega": 16.0, "tau": 1e-6}
        self.raw = RawForecastRegressor(alpha=0.01, n_landmarks=8, head_seed=19).fit(self.x, self.y)

    def additive(self, gate="kahm", **settings):
        return AdditiveForecastRegressor(gate, seed=7, geometry_parameters=self.geometry, **settings)

    def test_raw_wrapper_preserves_inherited_predictions(self):
        inherited = ForecastRegressor("raw", "rbf", alpha=0.01, n_landmarks=8,
                                       head_seed=19).fit(self.x, self.y)
        np.testing.assert_array_equal(self.raw.predict(self.query), inherited.predict(self.query))
        self.assertEqual(self.raw.metadata()["representation"], "raw")

    def test_gamma_zero_is_exact_baseline_and_does_not_evaluate_correction(self):
        model = self.additive(gamma=1).fit(self.x, self.y, raw_model=self.raw)
        baseline = self.raw.predict(self.query)
        with mock.patch.object(model.correction_model_, "predict", side_effect=AssertionError("evaluated correction")):
            np.testing.assert_array_equal(model.predict(self.query, gamma=0), baseline)
        with mock.patch.object(self.raw, "predict", side_effect=AssertionError("recomputed cached raw prediction")), \
                mock.patch.object(model.correction_model_, "predict", side_effect=AssertionError("evaluated correction")):
            np.testing.assert_array_equal(model.predict(self.query, gamma=0, raw_prediction=baseline), baseline)

    def test_zero_final_fit_omits_geometry_and_reports_zero_correction_cost_shape(self):
        model = self.additive(gamma=0)
        with mock.patch.object(LocalFeatureBank, "fit", side_effect=AssertionError("fit unnecessary geometry")), \
                mock.patch.object(self.raw, "predict", side_effect=AssertionError("unnecessary training prediction")):
            model.fit(self.x, self.y, raw_model=self.raw)
        np.testing.assert_array_equal(model.predict(self.query), self.raw.predict(self.query))
        self.assertEqual(model.correction_basis_width_, 0)
        self.assertEqual(model.stored_reference_rows_, 0)
        with self.assertRaisesRegex(RuntimeError, "nonzero gamma"):
            model.predict(self.query, gamma=1)

    def test_correction_does_not_mutate_frozen_raw_state(self):
        before = pickle.dumps(self.raw, protocol=5)
        prediction = self.raw.predict(self.query)
        model = self.additive(alpha=1, gamma=0.5).fit(self.x, self.y, raw_model=self.raw)
        self.assertIs(model.raw_model_, self.raw)
        self.assertEqual(before, pickle.dumps(self.raw, protocol=5))
        np.testing.assert_array_equal(prediction, self.raw.predict(self.query))
        expected = prediction + 0.5 * model.predict_correction(self.query)
        np.testing.assert_array_equal(model.predict(self.query), expected)

    def test_training_only_scalers_and_constant_target(self):
        y = np.column_stack((self.y[:, 0], np.full(len(self.x), -7.0)))
        raw = RawForecastRegressor(alpha=0.01, n_landmarks=8).fit(self.x, y)
        model = self.additive().fit(self.x, y, raw_model=raw)
        bank = model.correction_model_.bank_
        np.testing.assert_allclose(bank.mean_, self.x.mean(axis=0))
        np.testing.assert_allclose(bank.scale_, self.x.std(axis=0))
        state = pickle.dumps(model, protocol=5)
        prediction = model.predict(self.query + 100)
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(prediction[:, 1], -7, atol=1e-12)
        self.assertEqual(state, pickle.dumps(model, protocol=5))

    def test_single_target_and_multioutput_conventions(self):
        for target in (self.y[:, 0], self.y[:, :1], self.y):
            with self.subTest(shape=target.shape):
                raw = RawForecastRegressor(head="ridge", alpha=0.01).fit(self.x, target)
                model = self.additive("affine").fit(self.x, target, raw_model=raw)
                expected = (len(self.query),) if target.ndim == 1 else (len(self.query), target.shape[1])
                self.assertEqual(model.predict(self.query).shape, expected)
                self.assertTrue(np.isfinite(model.predict(self.query)).all())

    def test_train_provenance_rejects_other_inputs_targets_and_refitted_raw(self):
        with self.assertRaisesRegex(ValueError, "different training"):
            self.additive().fit(self.x + 1, self.y, raw_model=self.raw)
        with self.assertRaisesRegex(ValueError, "different training"):
            self.additive().fit(self.x, self.y + 1, raw_model=self.raw)
        model = self.additive().fit(self.x, self.y, raw_model=self.raw)
        self.raw.fit(self.x, self.y + 1)
        with self.assertRaisesRegex(RuntimeError, "shared raw model changed"):
            model.predict(self.query, gamma=0)

    def test_alpha_gamma_path_matches_separate_fixed_fits_and_reuses_geometry(self):
        alphas, gammas = (0.01, 1.0), (0.0, 0.25, 1.0)
        baseline = self.raw.predict(self.query)
        expected = [float(np.mean((baseline - self.y_query) ** 2))]
        predictions = [baseline]
        for alpha in alphas:
            fixed = self.additive("locality", alpha=alpha).fit(self.x, self.y, raw_model=self.raw)
            for gamma in gammas:
                if gamma == 0:
                    continue
                prediction = fixed.predict(self.query, gamma=gamma)
                expected.append(float(np.mean((prediction - self.y_query) ** 2)))
                predictions.append(prediction)
        selected = self.additive("locality")
        fit = LocalFeatureBank.fit
        transform = LocalFeatureBank.transform
        with mock.patch.object(LocalFeatureBank, "fit", autospec=True, side_effect=fit) as fits, \
                mock.patch.object(LocalFeatureBank, "transform", autospec=True, side_effect=transform) as transforms:
            trials = selected.fit_select(self.x, self.y, self.query, self.y_query,
                                         alphas, gammas, raw_model=self.raw)
        self.assertEqual(fits.call_count, 1)
        self.assertEqual(transforms.call_count, 2)
        self.assertEqual(len(trials), 1 + len(alphas) * (len(gammas) - 1))
        self.assertEqual(sum(trial["gamma"] == 0 for trial in trials), 1)
        self.assertFalse(trials[0]["alpha_applies"])
        np.testing.assert_allclose([trial["validation_mse"] for trial in trials], expected,
                                   rtol=1e-12, atol=1e-12)
        index = int(np.argmin(expected))
        self.assertEqual(selected.alpha, trials[index]["alpha"])
        self.assertEqual(selected.gamma, trials[index]["gamma"])
        np.testing.assert_allclose(selected.predict(self.query), predictions[index], rtol=1e-12, atol=1e-12)

    def test_global_exact_tie_prefers_smaller_gamma_across_alpha_order(self):
        # Artificial ridge outputs isolate the selection rule: alpha .01 with
        # gamma 1 and alpha 1 with gamma .25 both add exactly one to the raw
        # prediction. The latter must win despite being enumerated later.
        correction = SimpleNamespace(
            _fit_system=lambda x, y: (None, None, None),
            _transform_features=lambda x: np.ones((len(x), 1)),
            basis_=SimpleNamespace(transform=lambda features: features),
            basis_mean_=np.zeros(1), basis_scale_=np.ones(1),
            standardized_target_mean_=np.zeros(2), target_scale_=np.ones(2),
            target_mean_=np.zeros(2), basis_width_=1,
            bank_=SimpleNamespace(models_=[]),
            _solve_alpha=lambda eigenvalues, eigenvectors, rhs, alpha: np.full((1, 2), 1.0 if alpha == 0.01 else 4.0),
        )
        baseline = self.raw.predict(self.query)
        selected = self.additive()
        with mock.patch.object(selected, "_new_correction", return_value=correction):
            trials = selected.fit_select(
                self.x, self.y, self.query, baseline + 1.0,
                alphas=(0.01, 1.0), gammas=(0.0, 0.25, 1.0), raw_model=self.raw,
                raw_validation_prediction=baseline,
            )
        self.assertEqual(selected.alpha, 1.0)
        self.assertEqual(selected.gamma, 0.25)
        tied = [trial for trial in trials if trial["validation_mse"] == 0]
        self.assertEqual(len(tied), 2)

    def test_validation_targets_do_not_change_features_or_training_scaling(self):
        first, second = self.additive(), self.additive()
        first.fit_select(self.x, self.y, self.query, self.y_query, raw_model=self.raw)
        second.fit_select(self.x, self.y, self.query, self.y_query + 1000, raw_model=self.raw)
        a, b = first.correction_model_, second.correction_model_
        self.assertEqual(a.bank_.metadata(), b.bank_.metadata())
        for attr in ("input_mean_", "input_scale_", "target_mean_", "target_scale_", "basis_mean_", "basis_scale_"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))

    def test_matched_gate_controls_share_references_and_bases(self):
        models = [self.additive(gate).fit(self.x, self.y, raw_model=self.raw) for gate in ("kahm", "affine", "locality")]
        banks = [model.correction_model_.bank_ for model in models]
        for control in banks[1:]:
            for reference, alternate in zip(banks[0].models_, control.models_):
                np.testing.assert_array_equal(reference.reference_indices, alternate.reference_indices)
                np.testing.assert_array_equal(reference.references, alternate.references)
                np.testing.assert_array_equal(reference.basis, alternate.basis)
                self.assertIsNone(alternate.ae)

    def test_affine_and_locality_do_not_fit_kahm_kernels(self):
        original = historical_geometry._upstream
        forbidden = {"autoencoder", "kxx_matrix", "kxa_matrix", "kernel_regularized_least_squares"}

        def guarded(name):
            if name in forbidden:
                raise AssertionError(f"Control used KAHM machinery: {name}")
            return original(name)

        with mock.patch.object(historical_geometry, "_upstream", guarded):
            for gate in ("affine", "locality"):
                model = self.additive(gate).fit(self.x, self.y, raw_model=self.raw)
                self.assertEqual(model.fitted_kahm_kernels_, 0)
                self.assertTrue(np.isfinite(model.predict(self.query)).all())

    def test_ridge_penalty_changes_only_correction_coefficients(self):
        first = self.additive(alpha=0.01).fit(self.x, self.y, raw_model=self.raw)
        second = self.additive(alpha=100.0).fit(self.x, self.y, raw_model=self.raw)
        self.assertEqual(first.correction_model_.bank_.metadata(), second.correction_model_.bank_.metadata())
        self.assertLess(np.linalg.norm(second.correction_model_.coefficients_),
                        np.linalg.norm(first.correction_model_.coefficients_))
        np.testing.assert_array_equal(first.raw_model_.coefficients_, second.raw_model_.coefficients_)

    def test_deterministic_predictions_serialization_and_strict_json_metadata(self):
        first = self.additive(alpha=1, gamma=0.5).fit(self.x, self.y, raw_model=self.raw)
        second = self.additive(alpha=1, gamma=0.5).fit(self.x, self.y, raw_model=self.raw)
        np.testing.assert_array_equal(first.predict(self.query), second.predict(self.query))
        restored = pickle.loads(pickle.dumps(first, protocol=5))
        np.testing.assert_array_equal(first.predict(self.query), restored.predict(self.query))
        before = copy.deepcopy(first.metadata())
        json.dumps(before, allow_nan=False)
        self.assertFalse(before["correction_has_nystrom_basis"])
        self.assertGreater(before["stored_reference_rows"], 0)
        self.assertEqual(before["coefficient_count"], first.coefficient_count_)
        first.predict(self.query)
        self.assertEqual(before, first.metadata())

    def test_invalid_gamma_and_shape_and_absent_zero_candidate(self):
        for value in (-0.01, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                self.additive(gamma=value)
        model = self.additive().fit(self.x, self.y, raw_model=self.raw)
        with self.assertRaisesRegex(ValueError, "shape"):
            model.predict(self.query, raw_prediction=np.zeros(len(self.query)))
        with self.assertRaisesRegex(ValueError, "gamma-zero"):
            self.additive().fit_select(self.x, self.y, self.query, self.y_query,
                                       gammas=(0.25, 1.0), raw_model=self.raw)
        with self.assertRaisesRegex(ValueError, "different row counts"):
            self.additive().fit(self.x, self.y[:-1], raw_model=self.raw)


if __name__ == "__main__":
    unittest.main()
