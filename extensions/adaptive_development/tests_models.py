"""Numerical and causal checks for the bounded development models.

These checks validate the declared objectives and updates, not efficacy.
Run: python -m unittest extensions.adaptive_development.tests_models -v
"""
import copy
import unittest

import numpy as np
from scipy.linalg import lstsq

from .models import (METHODS, OnlineForecaster, _ReplayHead, _GeometryBank,
                     _SlidingKernel, _penalty, _with_intercept, DEFAULT_CONFIG)


class SlidingHeadTests(unittest.TestCase):
    @staticmethod
    def independent_centered_batch(head):
        """SVD on an unequilibrated augmented design, never normal equations."""
        valid = head.valid[:head.count]
        x, y = head.x[:head.count][valid], head.y[:head.count][valid]
        f = head.f[:head.count, 1:][valid]
        residual = y - x @ head.raw_coef
        center, target_mean = f.mean(axis=0), float(residual.mean())
        augmented = np.vstack((f - center, np.sqrt(head.geo_penalty[1:, 1:])))
        target = np.r_[residual - target_mean, np.zeros(head.feature_width)]
        slopes = lstsq(augmented, target, cond=1e-16,
                       lapack_driver="gelsd")[0]
        prediction = target_mean + (f - center) @ slopes
        objective = float(np.sum((residual - prediction)**2)
                          + slopes @ head.geo_penalty[1:, 1:] @ slopes)
        return slopes, center, target_mean, objective

    def test_current_residual_target_matches_independent_batch_objective(self):
        rng = np.random.default_rng(831)
        head = _ReplayHead(24, 5, 7)
        for i in range(130):
            z, f = rng.normal(size=5), rng.normal(size=7)
            y = float(z[0] - f[1] + rng.normal(scale=.3))
            head.push(z, y, f, valid=i % 11 != 0, origin=i)
            head.solve()
            x, target = head.x[:head.count], head.y[:head.count]
            raw = np.linalg.solve(x.T @ x + _penalty(6, 1), x.T @ target)
            valid = head.valid[:head.count]
            features = head.f[:head.count][valid]
            residual = target[valid] - x[valid] @ raw
            if len(features):
                geo = np.linalg.solve(features.T @ features + _penalty(8, 1),
                                      features.T @ residual)
                np.testing.assert_allclose(head.geo_coef, geo, atol=3e-12, rtol=3e-12)
            np.testing.assert_allclose(head.raw_coef, raw, atol=2e-12, rtol=2e-12)

    def test_feature_rebuild_matches_batch(self):
        rng = np.random.default_rng(73)
        head = _ReplayHead(16, 4, 3)
        for i in range(33):
            head.push(rng.normal(size=4), rng.normal(), rng.normal(size=3), origin=i)
        changed = rng.normal(size=(head.count, 3))
        valid = np.arange(head.count) % 5 != 0
        head.rebuild_features(changed, valid)
        x, y = head.x[:head.count], head.y[:head.count]
        f = _with_intercept(changed[valid])
        raw = np.linalg.solve(x.T @ x + _penalty(5, 1), x.T @ y)
        geo = np.linalg.solve(f.T @ f + _penalty(4, 1), f.T @ (y[valid] - x[valid] @ raw))
        np.testing.assert_allclose(head.raw_coef, raw, atol=1e-12)
        np.testing.assert_allclose(head.geo_coef, geo, atol=1e-12)

    def test_huge_transient_features_can_leave_sliding_window(self):
        # Original accumulated F'F subtraction retained catastrophic errors
        # after feature rows of this magnitude had left the bounded replay.
        rng = np.random.default_rng(667)
        head = _ReplayHead(24, 3, 5)
        baseline_only = _ReplayHead(24, 3)
        for i in range(112):
            z, f = rng.normal(size=3), rng.normal(size=5)
            if 16 <= i < 40:
                f[:3] *= 1e11
            y = float(.3*z[0] + .2*np.tanh(f[0]) + rng.normal(scale=.2))
            head.push(z, y, f, valid=i % 13 != 12, origin=i)
            baseline_only.push(z, y, origin=i)
            head.solve()
            baseline_only.solve()
            np.testing.assert_array_equal(head.raw_coef, baseline_only.raw_coef)
            slopes, center, mean, optimum = self.independent_centered_batch(head)
            use = head.valid[:head.count]
            frows = head.f[:head.count, 1:][use]
            residual = head.y[:head.count][use] - head.x[:head.count][use] @ head.raw_coef
            fitted = head.geo_target_mean + (frows - head.geo_center) @ head.geo_coef[1:]
            objective = np.sum((residual - fitted)**2) + np.sum(head.geo_coef[1:]**2)
            self.assertTrue(np.isfinite(head.geo_coef).all())
            self.assertLessEqual(objective, optimum*(1 + 2e-5) + 1e-9)
            if i >= 64:
                # After the huge rows expire, accuracy returns immediately;
                # neither a geometry rebuild nor an error fallback is needed.
                np.testing.assert_allclose(head.geo_coef[1:], slopes, rtol=1e-10, atol=1e-11)
                query = rng.normal(size=5)
                expected = mean + (query - center) @ slopes
                self.assertAlmostEqual(head.components(z, query)[1], expected, places=10)

    def test_equilibration_keeps_free_intercept_and_original_ridge_penalty(self):
        rng = np.random.default_rng(744)
        head = _ReplayHead(80, 3, 5)
        for i in range(80):
            z = rng.normal(size=3)
            f = rng.normal(size=5)*np.array([1e9, 1e4, 1.0, 1e-4, 1e-9])
            y = float(7 + .4*f[2] + rng.normal(scale=.1))
            head.push(z, y, f, origin=i)
        head.solve()
        slopes, center, mean, optimum = self.independent_centered_batch(head)
        query = np.array([1e9, -3e4, .8, 2e-4, -1e-9])
        actual = head.components(np.zeros(3), query)[1]
        expected = mean + (query - center) @ slopes
        self.assertAlmostEqual(actual, expected, places=7)
        residual = head.y - head.x @ head.raw_coef
        fitted = head.geo_target_mean + (head.f[:, 1:] - head.geo_center) @ head.geo_coef[1:]
        self.assertAlmostEqual(float(np.mean(residual - fitted)), 0.0, places=12)
        objective = np.sum((residual - fitted)**2) + np.sum(head.geo_coef[1:]**2)
        self.assertAlmostEqual(float(objective), optimum, places=9)

    def test_no_valid_rows_clear_centered_correction_state(self):
        head = _ReplayHead(8, 2, 3)
        for i in range(8):
            head.push(np.array([i, -i]), float(i*i), np.array([i, i*i, 2*i]))
        head.solve()
        for i in range(8):
            head.push(np.array([i, -i]), float(i), np.zeros(3), valid=False)
            head.solve()
        self.assertEqual(head.valid_count, 0)
        self.assertEqual(head.components(np.ones(2), np.ones(3))[1], 0.0)
        np.testing.assert_array_equal(head.geo_center, np.zeros(3))
        np.testing.assert_array_equal(head.geo_coef, np.zeros(4))


class KernelTests(unittest.TestCase):
    def test_recursive_inverse_and_free_intercept_match_batch(self):
        rng = np.random.default_rng(91)
        x = rng.normal(size=(15, 4))
        y = np.sin(x[:, 0]) + 3.0
        model = _SlidingKernel(20, 3.2, .1, audit_every=16).fit(x, y)
        for i in range(230):
            row = rng.normal(size=4)
            target = float(np.sin(row[0]) + .2 * row[1] + 3.0)
            model.push(row, target)
            if i % 7 == 0:
                gram = model.kernel(model.x, model.x) + .1 * np.eye(len(model.x))
                inv = np.linalg.inv(gram)
                intercept = float(inv.sum(axis=1) @ model.y / inv.sum())
                coef = inv @ (model.y - intercept)
                query = rng.normal(size=4)
                predicted = intercept + model.kernel(query, model.x)[0] @ coef
                self.assertLessEqual(len(model.x), 20)
                np.testing.assert_allclose(model.inverse, inv, atol=1e-9, rtol=1e-9)
                self.assertAlmostEqual(model.predict(query), predicted, places=10)


class GeometryTests(unittest.TestCase):
    def test_same_target_free_references_for_geometric_families(self):
        rng = np.random.default_rng(42)
        z = rng.normal(size=(70, 8))
        banks = [_GeometryBank(g, 32, 6, DEFAULT_CONFIG).fit(z)
                 for g in ["kahm", "affine", "rbf"]]
        for i in range(25):
            observed = rng.normal(size=(40, 8)) + .03 * i
            for bank in banks:
                bank.refresh(observed)
            for locals_ in zip(*(bank.models for bank in banks)):
                np.testing.assert_array_equal(locals_[0].q, locals_[1].q)
                np.testing.assert_array_equal(locals_[0].q, locals_[2].q)
            for bank in banks:
                self.assertLessEqual(sum(len(m.q) for m in bank.models), 32)
                f, safe, _ = bank.transform(observed)
                self.assertEqual(f.shape, (40, 12))
                self.assertTrue(np.isfinite(f).all())
                self.assertTrue(safe.all())

    def test_kahm_formula_and_original_regularizer(self):
        # Independent dense calculation reproduces the preserved formula.
        rng = np.random.default_rng(198)
        z = rng.normal(size=(80, 6))
        bank = _GeometryBank("kahm", 32, 71, DEFAULT_CONFIG).fit(z)
        query = rng.normal(size=(15, 6))
        f, valid, _ = bank.transform(query)
        logits, coords = [], []
        for model in bank.models:
            q, p = model.q, model.p
            projected = q @ p
            precision = np.linalg.inv(np.atleast_2d(np.cov(projected, rowvar=False)))
            gram_distance = np.empty((len(q), len(q)))
            query_distance = np.empty((len(query), len(q)))
            for i in range(len(q)):
                for j in range(len(q)):
                    delta = projected[i] - projected[j]
                    gram_distance[i, j] = delta @ precision @ delta
            projected_query = query @ p
            for i in range(len(query)):
                for j in range(len(q)):
                    delta = projected_query[i] - projected[j]
                    query_distance[i, j] = delta @ precision @ delta
            kernel = np.exp(-gram_distance / (2 * p.shape[1]))
            nu, change, iteration = float(np.mean(q * q)), 1.0, 0
            lam = 2.5 * nu
            while change > 1e-3 and iteration < 100:
                used = lam
                inverse = np.linalg.inv(kernel + used * np.eye(len(q)))
                residual = q - kernel @ inverse @ q
                lam = 2 * nu + float(np.mean(residual * residual))
                change = abs(lam - used)
                iteration += 1
            weights = np.exp(-query_distance / (2 * p.shape[1])) @ inverse
            reconstruction = weights @ q / weights.sum(axis=1, keepdims=True)
            radial = 1 - np.exp(-np.linalg.norm(query - reconstruction, axis=1))
            denom = np.linalg.norm(query, axis=1) * np.linalg.norm(reconstruction, axis=1)
            cos = np.divide(np.sum(query * reconstruction, axis=1), denom,
                            out=np.zeros(len(query)), where=denom != 0)
            angle = np.arccos(np.clip(cos, -1, 1)) / np.pi
            score = np.sqrt((radial**2 + angle**2) / 2)
            logits.append(16 * np.log(1 - np.clip(score, 0, 1) + 1e-6))
            coords.append((query - model.center) @ p)
            self.assertAlmostEqual(model.regularizer, used, places=11)
        logs = np.column_stack(logits)
        probs = np.exp(logs - logs.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        expected = np.column_stack([probs] + [probs[:, j:j+1] * c for j, c in enumerate(coords)])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(f, expected, atol=3e-11, rtol=3e-11)


class ForecasterTests(unittest.TestCase):
    @staticmethod
    def inputs():
        rng = np.random.default_rng(551)
        x = rng.normal(size=(60, 6))
        y = np.column_stack((.4 * x[:, -1] + .2 * x[:, 0]**2,
                             .2 * x[:, -1] - .3 * x[:, 2]))
        return rng, x, y

    def test_all_methods_are_finite_bounded_and_causal(self):
        for method in METHODS:
            rng, x, y = self.inputs()
            model = OnlineForecaster(method, 16, 24, [1, 3], seed=5,
                                     config={"refresh_every": 4}).fit(x, y)
            original_anchors = ([m.q[:m.anchor_count].copy() for m in model.bank.models]
                                if model.geometric else [])
            for t in range(80, 125):
                current = rng.normal(size=6)
                matured = [{"horizon_index": h, "x": rng.normal(size=6), "y": rng.normal(),
                            "origin": t - offset, "target": t}
                           for h, offset in enumerate([1, 3])]
                model.update(matured, t, current)
                base, correction, diag = model.predict_components(current)
                self.assertTrue(np.isfinite(np.r_[base, correction]).all())
                # Mixing gamma=0 reproduces baseline exactly.
                np.testing.assert_array_equal(base + 0.0 * correction, base)
                for head in model.heads:
                    self.assertLessEqual(head.count, 24)
                    raw = np.linalg.solve(head.x[:head.count].T @ head.x[:head.count]
                                          + head.raw_penalty,
                                          head.x[:head.count].T @ head.y[:head.count])
                    np.testing.assert_allclose(head.raw_coef, raw, atol=1e-10, rtol=1e-10)
                if model.geometric:
                    for local, initial in zip(model.bank.models, original_anchors):
                        np.testing.assert_array_equal(local.q[:local.anchor_count], initial)
                summary = model.state_summary()
                self.assertGreater(summary["owned_numpy_bytes"], 0)
                self.assertLessEqual(summary.get("total_geometry_references", 0), 16)
                if method.startswith("adaptive_"):
                    self.assertGreater(summary["observed_input_rows"], 0)
                    self.assertIsNotNone(model.observed)
                else:
                    self.assertEqual(summary["observed_input_rows"], 0)
                    self.assertIsNone(model.observed)
            before = copy.deepcopy(model.state_summary())
            bad = [{"horizon_index": 0, "x": x[0], "y": 1.0,
                    "origin": 130, "target": 131}]
            with self.assertRaisesRegex(ValueError, "causally mature"):
                model.update(bad, 130, x[0])
            self.assertEqual(model.state_summary(), before)

    def test_prediction_does_not_change_fitted_parameters(self):
        _, x, y = self.inputs()
        model = OnlineForecaster("adaptive_kahm", 16, 24, [1, 3], seed=5).fit(x, y)
        before = [(h.raw_coef.copy(), h.geo_coef.copy(), h.xx.copy(),
                   h.f.copy(), h.geo_center.copy(), h.geo_target_mean) for h in model.heads]
        first = model.predict_components(x[0])
        second = model.predict_components(x[0])
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        for head, snapshot in zip(model.heads, before):
            for actual, expected in zip([head.raw_coef, head.geo_coef, head.xx,
                                         head.f, head.geo_center, head.geo_target_mean], snapshot):
                np.testing.assert_array_equal(actual, expected)

    def test_refresh_heads_match_current_feature_map_and_batch_residuals(self):
        rng, x, y = self.inputs()
        model = OnlineForecaster("adaptive_kahm", 16, 24, [1, 3], seed=5,
                                 config={"refresh_every": 1}).fit(x, y)
        for t in range(70, 85):
            current = rng.normal(size=6)
            model.update([{"horizon_index": h, "x": rng.normal(size=6), "y": rng.normal(),
                           "origin": t - offset, "target": t}
                          for h, offset in enumerate([1, 3])], t, current)
            for head in model.heads:
                z = head.x[:head.count, 1:]
                f, valid, _ = model._features(z)
                np.testing.assert_allclose(head.f[:head.count][valid], _with_intercept(f[valid]), atol=1e-12)
                rows = head.x[:head.count]
                target = head.y[:head.count]
                raw = np.linalg.solve(rows.T @ rows + head.raw_penalty, rows.T @ target)
                design = _with_intercept(f[valid])
                geo = np.linalg.solve(design.T @ design + head.geo_penalty,
                                      design.T @ (target[valid] - rows[valid] @ raw))
                np.testing.assert_allclose(head.geo_coef, geo, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
