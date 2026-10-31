"""Numerical semantics checks only; not a forecast-accuracy/latency benchmark.

Run: python test_online.py
Writes check_online.json beside this file and exits nonzero on test failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import unittest

import numpy as np

from online import (BoundedWindowRidge, GeometryVersionError, KAHMDenominatorError,
                    NumericalUpdateError, cache_kahm)


METRICS: dict[str, object] = {}


def batch_fit(x, y, alpha):
    return np.linalg.solve(alpha * np.eye(x.shape[1]) + x.T @ x, x.T @ y)


class TestBoundedOnline(unittest.TestCase):
    def test_multioutput_sliding_window_matches_batch(self):
        rng = np.random.default_rng(61847)
        x = rng.normal(size=(300, 7))
        y = rng.normal(size=(300, 3))
        readout = BoundedWindowRidge(19, 7, 3, alpha=0.7, geometry_version="bank-v1", audit_capacity=8)
        max_error = 0.0
        max_prediction_error = 0.0
        for i in range(len(x)):
            report = readout.update(x[i], y[i], geometry_version="bank-v1")
            start = max(0, i + 1 - 19)
            expected = batch_fit(x[start:i+1], y[start:i+1], 0.7)
            max_error = max(max_error, float(np.max(np.abs(readout.coefficients - expected))))
            probe = rng.normal(size=(4, 7))
            max_prediction_error = max(max_prediction_error, float(np.max(np.abs(
                readout.predict(probe, geometry_version="bank-v1") - probe @ expected))))
            np.testing.assert_allclose(readout.coefficients, expected, rtol=1e-9, atol=1e-10)
            self.assertEqual(report.active_rows, min(i + 1, 19))
            self.assertEqual(report.evicted, i >= 19)
        self.assertEqual(readout.total_evictions, 281)
        self.assertEqual(len(readout.recent_audit_events), 8)
        self.assertEqual(readout.state_summary()["owned_numerical_array_bytes"],
                         8 * (2 * 7 * 7 + 2 * 7 * 3 + 19 * (7 + 3)))
        METRICS["sliding_window"] = {
            "updates": len(x), "window": 19, "features": 7, "outputs": 3,
            "max_coefficient_absolute_error": max_error,
            "max_prediction_absolute_error": max_prediction_error,
            "numerical_fallbacks": readout.numerical_fallbacks,
        }

    def test_single_row_window_and_penalized_intercept(self):
        rng = np.random.default_rng(419)
        readout = BoundedWindowRidge(1, 4, 2, alpha=0.3, geometry_version="v1")
        for _ in range(50):
            x = np.r_[1.0, rng.normal(size=3)]
            y = rng.normal(size=2)
            readout.update(x, y, geometry_version="v1")
            expected = batch_fit(x[None, :], y[None, :], 0.3)
            np.testing.assert_allclose(readout.coefficients, expected, rtol=1e-9, atol=1e-10)
        intercept_only = BoundedWindowRidge(5, 1, 1, alpha=2.0, geometry_version="v1")
        intercept_only.update([1], [6], geometry_version="v1")
        np.testing.assert_allclose(intercept_only.coefficients, [[2.0]])

    def test_near_singular_downdate_uses_logged_replay(self):
        readout = BoundedWindowRidge(1, 1, 1, alpha=1.0, geometry_version="v1")
        first = readout.update([1e6], [2e6], geometry_version="v1")
        second = readout.update([2.0], [3.0], geometry_version="v1")
        self.assertTrue(first.rebuilt)
        self.assertTrue(second.rebuilt)
        self.assertEqual(second.reason, "unsafe_rank_one_denominator")
        np.testing.assert_allclose(readout.coefficients, [[6.0/5.0]], rtol=1e-12, atol=1e-12)
        self.assertEqual(readout.numerical_fallbacks, 2)
        METRICS["fallback_reports"] = [first.to_dict(), second.to_dict()]

    def test_version_guards_reset_and_replay(self):
        rng = np.random.default_rng(201)
        x = rng.normal(size=(6, 4))
        y = rng.normal(size=(6, 2))
        readout = BoundedWindowRidge(6, 4, 2, alpha=0.8, geometry_version="old")
        readout.update(x[0], y[0], geometry_version="old")
        before = readout.coefficients
        with self.assertRaises(GeometryVersionError):
            readout.update(x[1], y[1], geometry_version="new")
        with self.assertRaises(GeometryVersionError):
            readout.predict(x[1], geometry_version="new")
        np.testing.assert_array_equal(before, readout.coefficients)
        self.assertEqual(readout.active_rows, 1)
        # These stand in for explicitly recomputed features of a new map.
        new_features = x * np.array([1.0, 2.0, 0.5, 3.0])
        readout.replay_geometry("new", new_features, y)
        np.testing.assert_allclose(readout.coefficients, batch_fit(new_features, y, 0.8),
                                   rtol=1e-12, atol=1e-12)
        with self.assertRaises(GeometryVersionError):
            readout.predict(x[0], geometry_version="old")
        readout.reset_geometry("third")
        self.assertEqual(readout.active_rows, 0)
        np.testing.assert_array_equal(readout.predict(x[0], geometry_version="third"), np.zeros(2))
        self.assertEqual(readout.geometry_replays, 1)
        self.assertEqual(readout.geometry_resets, 1)

    def test_invalid_inputs_do_not_mutate_state(self):
        readout = BoundedWindowRidge(2, 2, 1, alpha=1.0, geometry_version="v1")
        readout.update([1, 2], [3], geometry_version="v1")
        before = readout.coefficients
        summary = readout.state_summary()
        with self.assertRaises(ValueError):
            readout.update([np.nan, 1], [1], geometry_version="v1")
        with self.assertRaises(NumericalUpdateError):
            readout.update([1e308, 1e308], [1e308], geometry_version="v1")
        with self.assertRaises(ValueError):
            readout.replay_geometry("new", np.ones((3, 2)), np.ones((3, 1)))
        with self.assertRaises(ValueError):
            readout.replay_geometry("new", [[np.nan, 2]], [[1]])
        np.testing.assert_array_equal(before, readout.coefficients)
        self.assertEqual(readout.state_summary(), summary)

    def test_bounded_scheduled_refactors_and_zero_features(self):
        readout = BoundedWindowRidge(7, 3, 2, alpha=0.2, geometry_version="v1", refactor_every=5)
        for i in range(20):
            report = readout.update(np.zeros(3), [i, -i], geometry_version="v1")
            if (i + 1) % 5 == 0:
                self.assertEqual(report.reason, "scheduled_rebuild")
        np.testing.assert_array_equal(readout.coefficients, np.zeros((3, 2)))
        self.assertEqual(readout.scheduled_rebuilds, 4)
        self.assertEqual(readout.numerical_fallbacks, 0)
        self.assertEqual(readout.active_rows, 7)


class TestKAHMCache(unittest.TestCase):
    def test_fixed_query_equivalence_and_common_scaling(self):
        rng = np.random.default_rng(12819)
        q = rng.normal(size=(5, 9))
        distances = np.sum((q[:, :, None] - q[:, None, :])**2, axis=0)
        kernel = np.exp(-distances / 4.0)
        b = np.linalg.solve(kernel + 0.8 * np.eye(9), np.eye(9))
        queries = rng.normal(size=(30, 5))
        query_kernel = np.exp(-np.sum((queries[:, :, None] - q[None, :, :])**2, axis=1) / 4.0)
        cache = cache_kahm(q, b)
        weights = query_kernel @ b.T
        direct = weights @ q.T / np.sum(weights, axis=1, keepdims=True)
        cached = cache.reconstruct(query_kernel)
        np.testing.assert_allclose(cached, direct, rtol=1e-12, atol=1e-12)
        scaled = cache.reconstruct(query_kernel * np.exp(rng.uniform(-80, 80, size=(30, 1))))
        np.testing.assert_allclose(scaled, cached, rtol=1e-12, atol=1e-12)
        self.assertEqual(cache.reconstruct(query_kernel[0]).shape, (5,))
        self.assertFalse(cache.reconstruction_matrix.flags.writeable)
        METRICS["kahm_cache"] = {
            "queries": 30, "references": 9, "dimension": 5,
            "max_absolute_equivalence_error": float(np.max(np.abs(cached - direct))),
            "max_common_scaling_error": float(np.max(np.abs(cached - scaled))),
        }

    def test_nonsymmetric_B_and_singular_denominator(self):
        q = np.array([[1., 2.], [3., 4.]])
        b = np.array([[2., 1.], [0., 3.]])
        k = np.array([0.5, 0.2])
        direct = q @ b @ k / (np.ones(2) @ b @ k)
        np.testing.assert_allclose(cache_kahm(q, b).reconstruct(k), direct)
        bad = cache_kahm(q, np.diag([1.0, -1.0]))
        with self.assertRaises(KAHMDenominatorError):
            bad.reconstruct([1.0, 1.0])
        with self.assertRaises(KAHMDenominatorError):
            bad.reconstruct([0.0, 0.0])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "scope": "Numerical fixed-feature/readout and fixed-bank-cache checks only",
        "not_established": ["streaming forecast accuracy", "KAHM predictive advantage",
                            "geometry update correctness", "latency or memory benchmark"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "metrics": METRICS,
        "complexity": {
            "ordinary_update": "O(p^2 + p*H)",
            "bounded_rebuild": "O(W*p^2 + W*p*H + p^3 + p^2*H)",
            "owned_numerical_state": "O(W*(p+H) + p^2 + p*H)",
            "cache_construction": "O(d*m^2)",
            "cache_query_excluding_kernel": "O(d*m)",
        },
    }
    target = Path(__file__).with_name("check_online.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
