"""Numerical and data-protocol checks; run with Python's unittest discovery."""
import unittest

import numpy as np

from certified_forecasting.data import SYSTEM_BOUNDS, histories_targets, make_episodes
from certified_forecasting.features import NestedSoftFeatures, stable_associations
from certified_forecasting.models import (
    BlockBoundedRidge, BoundedRidge, KoopmanRidge, Persistence, project_block_ball,
)


class TestProtocol(unittest.TestCase):
    def test_causal_windows_and_common_support(self):
        episodes = np.arange(30, dtype=float)[None, :, None]
        x, y = histories_targets(episodes, 3, [1, 4], origin_stride=2, first_origin=5)
        np.testing.assert_array_equal(x[0, 0], [3, 4, 5])
        np.testing.assert_array_equal(y[0, 0], [6, 9])
        x_short, y_short = histories_targets(episodes, 1, [1, 4],
                                            origin_stride=2, first_origin=5)
        np.testing.assert_array_equal(y, y_short)
        np.testing.assert_array_equal(x[:, :, -1:], x_short)
        # Alter observations strictly after the first origin: its input cannot change.
        altered = episodes.copy()
        altered[:, 6:] = -99
        altered_x, _ = histories_targets(altered, 3, [1, 4],
                                         origin_stride=2, first_origin=5)
        np.testing.assert_array_equal(x[:, 0], altered_x[:, 0])

    def test_generator_bounds_and_seed_reproducibility(self):
        for system, (lo, hi) in SYSTEM_BOUNDS.items():
            first = make_episodes(system, 12, 80, 9)
            second = make_episodes(system, 12, 80, 9)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (12, 80, 1))
            self.assertTrue(np.all((first >= lo) & (first <= hi)))
            self.assertFalse(np.array_equal(first[0], first[1]))


class TestDecoder(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(812)
        self.x = rng.normal(size=(100, 8))
        self.y = self.x[:, :3] * 3 + rng.normal(size=(100, 3)) * 0.1

    def test_constraint_and_stationarity(self):
        model = BoundedRidge(alpha=1e-5, kappa=0.7, bounds=None).fit(self.x, self.y)
        self.assertLessEqual(model.frobenius_norm_, 0.7 + 1e-10)
        self.assertAlmostEqual(model.frobenius_norm_, 0.7, places=9)
        xc = self.x - self.x.mean(axis=0)
        yc = self.y - self.y.mean(axis=0)
        residual = xc.T @ (xc @ model.coef_ - yc) / len(xc) + model.ridge_ * model.coef_
        np.testing.assert_allclose(residual, 0, atol=1e-9)

    def test_clipping_and_lipschitz(self):
        model = BoundedRidge(kappa=1.2, bounds=(-1, 1)).fit(self.x, self.y)
        pred = model.predict(self.x)
        self.assertTrue(np.all(np.abs(pred) <= 1))
        displacement = self.x[::2] - self.x[1::2]
        predicted = np.abs(pred[::2] - pred[1::2])
        upper = np.linalg.norm(displacement, axis=1)[:, None] * model.lipschitz_
        self.assertTrue(np.all(predicted <= upper + 1e-12))

    def test_constant_budget_zero_and_zero_padding(self):
        zero = BoundedRidge(kappa=0, bounds=None).fit(self.x, self.y)
        np.testing.assert_allclose(zero.predict(self.x), np.tile(self.y.mean(axis=0), (100, 1)))
        small = BoundedRidge(alpha=1e-5, kappa=1.0, bounds=None).fit(self.x[:, :3], self.y)
        padded = np.vstack([small.coef_, np.zeros((5, 3))])
        np.testing.assert_allclose(self.x @ padded + small.intercept_,
                                   small.predict(self.x[:, :3]))
        self.assertAlmostEqual(np.linalg.norm(padded), small.frobenius_norm_)

    def test_persistence_uses_latest_available_value(self):
        model = Persistence().fit(self.x, self.y)
        np.testing.assert_array_equal(model.predict(self.x), np.repeat(self.x[:, -1:], 3, axis=1))


class TestGeometry(unittest.TestCase):
    def test_stable_normalization_retains_relative_affinities(self):
        # Legacy denominator <= 1e-12 would incorrectly make this row uniform.
        scores = np.array([[0.22, 0.30] + [0.90] * 23])
        feature = stable_associations(scores, omega=128, tau=1e-6)
        self.assertGreater(feature[0, 0], 0.999)
        self.assertAlmostEqual(feature.sum(), 1.0, places=12)
        np.testing.assert_allclose(stable_associations(np.ones((2, 3))), 1/3)

    def test_prefix_exactly_preserved_and_each_block_simplex(self):
        rng = np.random.default_rng(523)
        x = rng.normal(size=(100, 3))
        for kind in ("affine", "rbf"):
            feature = NestedSoftFeatures((2, 4), kind=kind, seed=4).fit(x)
            full = feature.transform(x)
            prefix = feature.transform(x, n_blocks=1)
            np.testing.assert_array_equal(full[:, :2], prefix)
            for block in feature.block_slices_:
                np.testing.assert_allclose(full[:, block].sum(axis=1), 1, atol=1e-12)
                self.assertTrue(np.all(full[:, block] >= 0))

    def test_original_kahm_smoke(self):
        # Exercises the original repository's actual KAHM model, not a surrogate.
        rng = np.random.default_rng(92)
        x = rng.normal(size=(70, 3))
        feature = NestedSoftFeatures((2,), kind="kahm", seed=2,
                                     max_train_per_cluster=12).fit(x)
        phi = feature.transform(x[:11])
        self.assertEqual(phi.shape, (11, 2))
        self.assertTrue(np.isfinite(phi).all())
        np.testing.assert_allclose(phi.sum(axis=1), 1, atol=1e-12)


class TestKoopman(unittest.TestCase):
    def test_effective_forecast_matches_explicit_affine_rollout(self):
        rng = np.random.default_rng(426)
        f = rng.normal(size=(100, 4))
        a = np.diag([0.7, 0.8, 0.5, 0.6])
        a[0, 1] = 0.1
        shift = np.array([0.1, -0.2, 0.05, 0.3])
        f_next = f @ a + shift
        y = f @ np.array([0.2, -0.3, 0.4, 0.5]) + 0.15
        model = KoopmanRidge(alpha=1e-5, bounds=(-1, 1)).fit(f, f_next, y)
        horizons = [1, 4, 2, 4]
        output, state = {}, f.copy()
        for step in range(1, max(horizons) + 1):
            state = state @ model.transition_ + model.shift_
            output[step] = np.clip(state @ model.decoder_ + model.decoder_intercept_, -1, 1)
        expected = np.column_stack([output[h] for h in horizons])
        np.testing.assert_allclose(model.predict(f, horizons), expected, atol=1e-12)
        w, b = model.effective_coefficients(horizons)
        np.testing.assert_allclose(np.clip(f @ w + b, -1, 1), expected, atol=1e-12)
        np.testing.assert_allclose(model.lipschitz_for(horizons), np.linalg.norm(w, axis=0))
        pred = model.predict(f, horizons)
        distance = np.linalg.norm(f[::2] - f[1::2], axis=1)
        self.assertTrue(np.all(np.abs(pred[::2] - pred[1::2])
                               <= distance[:, None] * model.lipschitz_for(horizons) + 1e-12))


class TestBlockBudget(unittest.TestCase):
    def test_projection_known_solution_and_feasibility(self):
        w = np.array([[3.0], [4.0], [0.0], [2.0]])
        blocks = (slice(0, 2), slice(2, 4))
        projected = project_block_ball(w, blocks, 3.0)
        # Norm vector (5,2) projects to (3,0), common threshold 2.
        np.testing.assert_allclose(projected[:, 0], [1.8, 2.4, 0, 0], atol=1e-12)
        self.assertAlmostEqual(sum(np.linalg.norm(projected[s]) for s in blocks), 3.0)
        np.testing.assert_array_equal(project_block_ball(w, blocks, 8), w)
        np.testing.assert_array_equal(project_block_ball(w, blocks, 0), np.zeros_like(w))

    def test_single_block_agrees_with_exact_ridge_and_gap(self):
        rng = np.random.default_rng(744)
        x = rng.normal(size=(200, 5))
        y = x[:, :2] * 2 + rng.normal(size=(200, 2)) * 0.1
        exact = BoundedRidge(alpha=1e-4, kappa=0.7, bounds=None).fit(x, y)
        iterative = BlockBoundedRidge(alpha=1e-4, kappa=0.7, bounds=None,
                                     block_slices=[slice(0, 5)], tol=1e-9).fit(x, y)
        self.assertTrue(iterative.converged_)
        self.assertLessEqual(iterative.norm_sum_, 0.7 + 1e-12)
        np.testing.assert_allclose(iterative.coef_, exact.coef_, atol=3e-6)
        def objective(m):
            return np.mean(np.sum((m.predict(x) - y)**2, axis=1)) + 1e-4 * np.sum(m.coef_**2)
        self.assertLessEqual(objective(iterative) - objective(exact),
                             iterative.optimization_gap_ + 1e-10)

    def test_duplicate_class_merging_and_block_metric(self):
        rng = np.random.default_rng(994)
        x = rng.normal(size=(120, 3))
        duplicated = np.concatenate([x, x], axis=1)
        y = (x[:, :1] - 0.3 * x[:, 1:2])
        blocks = [slice(0, 3), slice(3, 6)]
        model = BlockBoundedRidge(kappa=0.8, bounds=(-1, 1), block_slices=blocks,
                                 tol=1e-8).fit(duplicated, y)
        self.assertTrue(model.converged_)
        merged = model.coef_[:3] + model.coef_[3:]
        self.assertLessEqual(np.linalg.norm(merged), model.norm_sum_ + 1e-12)
        self.assertLessEqual(model.norm_sum_, 0.8 + 1e-12)
        np.testing.assert_allclose(np.clip(x @ merged + model.intercept_, -1, 1),
                                   model.predict(duplicated), atol=1e-12)
        pred = model.predict(duplicated)
        d = np.linalg.norm(x[::2] - x[1::2], axis=1)
        self.assertTrue(np.all(np.abs(pred[::2, 0] - pred[1::2, 0])
                               <= d * model.lipschitz_[0] + 1e-12))


if __name__ == "__main__":
    unittest.main()
