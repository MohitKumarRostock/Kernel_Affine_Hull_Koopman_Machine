"""Focused invariance, derivative, matching, and degeneracy checks."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from extensions.natural_forecasting.models import LocalFeatureBank
from extensions.target_geometry.models import (
    LearnedDiagonalMetric, RidgeHead, RobustLocalFeatureBank, _metric_objective,
)


class TestPredictiveMetric(unittest.TestCase):
    def test_gradient_includes_softmax_and_centered_parameterization(self):
        rng = np.random.default_rng(611)
        x, y = rng.normal(size=(13, 4)), rng.normal(size=(13, 3))
        squared = (x[:, None] - x[None, :]) ** 2
        groups = np.arange(13) // 3
        allowed = groups[:, None] != groups[None, :]
        allowed[4] = False  # Query-subset derivative when no donor is admissible.
        eta = rng.normal(size=4) / 3
        args = (squared, y, allowed, 4.0, 0.07)
        value, gradient = _metric_objective(eta, *args)
        numerical = []
        for k in range(4):
            step = np.eye(4)[k] * 1e-6
            numerical.append((_metric_objective(eta + step, *args)[0]
                              - _metric_objective(eta - step, *args)[0]) / 2e-6)
        self.assertTrue(np.isfinite(value))
        assert_allclose(gradient, numerical, rtol=2e-5, atol=1e-8)
        self.assertAlmostEqual(float(gradient.sum()), 0.0, places=12)

    def test_training_fit_bounded_reproducible_and_query_invariant(self):
        rng = np.random.default_rng(612)
        x = rng.normal(size=(90, 5))
        residual = (x[:, 0] ** 2 + 0.1 * x[:, 1])[:, None]
        groups = np.arange(90) // 3
        one = LearnedDiagonalMetric(max_rows=70, max_iter=15, seed=77).fit(x, residual, groups=groups)
        two = LearnedDiagonalMetric(max_rows=70, max_iter=15, seed=77).fit(x, residual, groups=groups)
        assert_array_equal(one.fit_indices_, two.fit_indices_)
        assert_allclose(one.weights_, two.weights_, atol=1e-13)
        self.assertLessEqual(one.final_objective_, one.initial_objective_ + 1e-12)
        self.assertLessEqual(one.weights_.max() / one.weights_.min(), 16 + 1e-10)
        self.assertAlmostEqual(float(np.log(one.weights_).sum()), 0.0, places=10)
        saved_weights = one.weights_.copy()
        saved_metadata = one.metadata()
        q = one.transform(x[:3])
        all_queries = one.transform(np.vstack((x[:3], np.full((20, 5), 1e6))))
        assert_allclose(q, all_queries[:3], atol=0)
        assert_array_equal(one.weights_, saved_weights)
        self.assertEqual(one.metadata(), saved_metadata)

    def test_group_and_position_exclusion_counts(self):
        x = np.column_stack((np.arange(8), np.arange(8) ** 2))
        groups = np.repeat(np.arange(4), 2)
        positions = np.arange(8) * 2
        metric = LearnedDiagonalMetric(max_iter=2).fit(
            x, np.arange(8), groups=groups, positions=positions, exclusion_radius=3)
        expected = ((groups[:, None] != groups[None, :])
                    & (np.abs(positions[:, None] - positions[None, :]) > 3)).sum(axis=1)
        assert_array_equal(metric.donor_counts_, expected)
        self.assertEqual(metric.fit_group_count_, 4)

    def test_unidentifiable_metric_returns_logged_identity(self):
        x = np.arange(24).reshape(8, 3)
        y = np.arange(8)
        grouped = LearnedDiagonalMetric().fit(x, y, groups=np.ones(8))
        constant = LearnedDiagonalMetric().fit(x, np.ones(8))
        repeated = LearnedDiagonalMetric().fit(np.ones((8, 3)), y)
        for metric in (grouped, constant, repeated):
            assert_array_equal(metric.weights_, np.ones(3))
            self.assertIsNotNone(metric.fallback_reason_)

    def test_metric_and_bank_use_same_training_input_units(self):
        rng = np.random.default_rng(613)
        full = rng.normal(size=(35, 4))
        correction = full[15:]
        metric = LearnedDiagonalMetric(max_iter=3).fit(
            correction, correction[:, 0] ** 2, scaling_data=full)
        bank = RobustLocalFeatureBank("affine", metric_weights=metric.weights_,
                                      cluster_counts=(2,), max_refs=8).fit(full)
        assert_allclose(metric.transform(full), bank.scale_inputs(full))


class TestRobustBanks(unittest.TestCase):
    def test_identity_reproduces_historical_features(self):
        rng = np.random.default_rng(614)
        x, query = rng.normal(size=(65, 4)), rng.normal(size=(19, 4))
        parameters = dict(seed=14, cluster_counts=(2, 4), rank=2, max_refs=12, max_fit=64)
        for gate in ("kahm", "affine", "locality"):
            old = LocalFeatureBank(gate, **parameters).fit(x)
            new = RobustLocalFeatureBank(gate, **parameters).fit(x)
            assert_allclose(new.transform(query), old.transform(query), rtol=1e-12, atol=1e-12)
            self.assertEqual(new.group_reductions_, [])

    def test_metric_is_not_cancelled_and_partitions_match(self):
        rng = np.random.default_rng(615)
        x = rng.normal(size=(80, 4))
        weights = np.array([4, 1, 1, .25])
        banks = [RobustLocalFeatureBank(g, metric_weights=weights, seed=42,
                                       cluster_counts=(2, 4), max_refs=12).fit(x)
                 for g in ("kahm", "affine", "locality", "rbf")]
        standardized = (x - x.mean(axis=0)) / x.std(axis=0)
        for bank in banks:
            assert_allclose(bank.scale_inputs(x), standardized * np.sqrt(weights))
            self.assertFalse(np.allclose(bank.scale_inputs(x), standardized))
            self.assertTrue(np.isfinite(bank.transform(x)).all())
            for before, after in zip(banks[0].models_, bank.models_):
                assert_array_equal(before.reference_indices, after.reference_indices)

    def test_seven_distinct_histories_with_eight_requested_groups(self):
        rng = np.random.default_rng(616)
        unique = rng.normal(size=(7, 6))
        x = unique[np.arange(23) % 7]
        for gate in ("kahm", "affine", "locality", "rbf"):
            bank = RobustLocalFeatureBank(gate, seed=3, cluster_counts=(2, 4, 8)).fit(x)
            self.assertEqual(bank.cluster_counts, (2, 4, 7))
            self.assertTrue(bank.group_reductions_)
            self.assertTrue(np.isfinite(bank.transform(x)).all())

    def test_constant_training_histories_have_rank_zero_fallbacks(self):
        x = np.ones((12, 5))
        for gate in ("kahm", "affine", "locality", "rbf"):
            bank = RobustLocalFeatureBank(gate, cluster_counts=(2, 4, 8)).fit(x)
            self.assertEqual(bank.cluster_counts, (1, 1, 1))
            self.assertTrue(all(model.basis.shape[1] == 0 for model in bank.models_))
            assert_allclose(bank.transform(x), np.ones((12, 3)))


class TestRidgeHead(unittest.TestCase):
    def test_path_matches_direct_penalized_normal_equations(self):
        rng = np.random.default_rng(617)
        x, y = rng.normal(size=(60, 7)), rng.normal(size=(60, 3))
        x[:, 6] = 2  # Constant columns must remain harmless.
        for alpha, head in RidgeHead.fit_path(x, y, (.01, .1, 1.)).items():
            z = (x - head.input_mean_) / head.input_scale_
            target = (y - head.target_mean_) / head.target_scale_
            beta = np.linalg.solve(z.T @ z / len(z) + alpha * np.eye(z.shape[1]),
                                   z.T @ target / len(z))
            expected = z @ beta * head.target_scale_ + head.target_mean_
            assert_allclose(head.predict(x), expected, atol=1e-12)
            independent = RidgeHead(alpha).fit(x, y)
            assert_allclose(head.predict(x), independent.predict(x), atol=1e-12)

    def test_constant_and_zero_residuals_preserve_intercept(self):
        x = np.arange(60).reshape(20, 3)
        for constant in (0., 3.5):
            head = RidgeHead(.1).fit(x, np.full(20, constant))
            assert_allclose(head.predict(x * 10), constant)
            assert_allclose(head.coefficients_, 0)


if __name__ == "__main__":
    unittest.main()
