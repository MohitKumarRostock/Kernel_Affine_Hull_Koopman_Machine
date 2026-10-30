"""Scientific contract tests on synthetic arrays; no benchmark outcomes are used."""
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from extensions.natural_forecasting import models as original
from extensions.scaling_sensitivity import natural


class NaturalSensitivityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(98271)
        self.x = rng.normal(size=(48, 4)) * np.array([1., 3., .5, 2.]) + np.array([2., -1., 3., 8.])
        self.y = np.column_stack((.2 * self.x[:, 0] + self.x[:, 1] ** 2,
                                  np.sin(self.x[:, 2]) + .1 * self.x[:, 3]))
        self.query = rng.normal(size=(13, 4)) * 4 + 5
        self.target = np.column_stack((.2 * self.query[:, 0] + self.query[:, 1] ** 2,
                                       np.sin(self.query[:, 2]) + .1 * self.query[:, 3]))
        self.parameters = dict(head="rbf", raw_skip=True, seed=17, head_seed=1017,
            n_landmarks=8, geometry_parameters=dict(cluster_counts=[2], rank=2,
                max_refs=12, max_fit=32, omega=16., tau=1e-6, scaling="standard"))
        self.threads = threadpool_limits(limits=1)
        self.threads.__enter__()

    def tearDown(self):
        self.threads.__exit__(None, None, None)

    def test_standard_adapter_matches_original_all_gates(self):
        for representation in ("affine_local", "locality_local", "kahm_local"):
            with self.subTest(representation=representation):
                expected = original.ForecastRegressor(representation, **self.parameters)
                actual = natural.ForecastRegressor(representation, geometry_scaling="standard", **self.parameters)
                left = expected.fit_select(self.x, self.y, self.query, self.target, [.01, 1., 100.])
                right = actual.fit_select(self.x, self.y, self.query, self.target, [.01, 1., 100.])
                self.assertEqual(left, right)
                np.testing.assert_array_equal(actual.predict(self.query), expected.predict(self.query))
                np.testing.assert_array_equal(actual.transform_features(self.query), expected.transform_features(self.query))

    def test_minmax_training_range_and_unclipped_heldout(self):
        bank = natural.MinMaxLocalFeatureBank("affine", seed=17, cluster_counts=[2], rank=2,
                                              max_refs=12, max_fit=32).fit(self.x)
        z = bank.scale_inputs(self.x)
        np.testing.assert_allclose(z.min(axis=0), -1., atol=1e-14)
        np.testing.assert_allclose(z.max(axis=0), 1., atol=1e-14)
        outside = self.x.max(axis=0)[None, :] + np.ptp(self.x, axis=0)[None, :]
        np.testing.assert_allclose(bank.scale_inputs(outside), 3., atol=1e-14)
        minimum, maximum = bank.minimum_.copy(), bank.maximum_.copy()
        bank.transform(np.vstack([self.query, outside]))
        np.testing.assert_array_equal(bank.minimum_, minimum)
        np.testing.assert_array_equal(bank.maximum_, maximum)

    def test_constant_geometry_column_zero_raw_skip_preserved(self):
        x = self.x.copy()
        x[:, 3] = 7.
        model = natural.ForecastRegressor("affine_local", geometry_scaling="minmax", **self.parameters).fit(x, self.y)
        query = self.query.copy()
        query[:, 3] = 12.
        np.testing.assert_array_equal(model.bank_.scale_inputs(query)[:, 3], 0.)
        features = model.transform_features(query)
        mean, scale = original._standardizer(x, 1e-12)
        np.testing.assert_array_equal(features[:, :4], (query - mean) / scale)
        np.testing.assert_array_equal(features[:, 3], 5.)

    def test_minmax_raw_skip_target_scale_and_alpha_selection_contract(self):
        model = natural.ForecastRegressor("kahm_local", geometry_scaling="minmax", **self.parameters)
        trials = model.fit_select(self.x, self.y, self.query, self.target, [.01, 1., 100.])
        mean, scale = original._standardizer(self.x, 1e-12)
        target_mean, target_scale = original._standardizer(self.y, 1e-12)
        np.testing.assert_array_equal(model.transform_features(self.query)[:, :4], (self.query - mean) / scale)
        np.testing.assert_array_equal(model.target_mean_, target_mean)
        np.testing.assert_array_equal(model.target_scale_, target_scale)
        for trial in trials:
            separately_fitted = natural.ForecastRegressor("kahm_local", geometry_scaling="minmax",
                alpha=trial["alpha"], **self.parameters).fit(self.x, self.y)
            loss = np.mean((separately_fitted.predict(self.query) - self.target) ** 2)
            self.assertAlmostEqual(loss, trial["validation_mse"], places=10)
        self.assertEqual(model.alpha, min(trials, key=lambda row: row["validation_mse"])["alpha"])

    def test_raw_minmax_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Raw references"):
            natural.ForecastRegressor("raw", "rbf", geometry_scaling="minmax")


if __name__ == "__main__":
    unittest.main()
