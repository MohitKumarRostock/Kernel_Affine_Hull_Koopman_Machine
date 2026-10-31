"""Computational checks for the reconstructed exploratory implementation."""
import json
import unittest

import numpy as np
from scipy.special import softmax

from geometry import (GATES, SharedBank, _baseline_ridge, DEFAULT_CONFIG,
                      _finite_loss, _ridge_fit, _ridge_predict, _train_temperature,
                      feature_matrix, fit_suite)


def run_nontrivial_objective_check():
    """Reconstruct the optimized B forecast loss on a separate development seed."""
    rng = np.random.default_rng(923751)
    xa = rng.normal(size=(128, 5))
    xb = rng.normal(size=(72, 5))
    def response(x):
        return (np.sin(2.1 * x[:, 0]) + 0.45 * x[:, 1] * x[:, 2]
                + 0.2 * x[:, 3] + rng.normal(scale=0.03, size=len(x)))
    ya, yb = response(xa), response(xb)
    config = {**DEFAULT_CONFIG, "max_references": 12,
              "max_objective_evaluations": 64}
    base = _ridge_fit(xa, ya, 0.1)
    base_a, base_b = _ridge_predict(base, xa), _ridge_predict(base, xb)
    residual_a, residual_b = ya - base_a, yb - base_b
    bank = SharedBank(923752, config).fit(xa)
    caches = {"A": bank.transform(xa, "A"), "B": bank.transform(xb, "B")}
    scale = max(float(np.var(ya)), 1e-12)
    rows = []
    for gate in GATES:
        eta, optimizer = _train_temperature(caches, gate, residual_a, residual_b,
                                            0.1, scale, config)
        fa = feature_matrix(caches["A"], gate, eta)
        fb = feature_matrix(caches["B"], gate, eta)
        head = _ridge_fit(fa, residual_a, 0.1)
        correction = _ridge_predict(head, fb)
        final_forecast = base_b + correction
        actual_b_mse = _finite_loss(yb, final_forecast)
        residual_b_mse = _finite_loss(residual_b, correction)
        recorded_b_mse = optimizer["best_normalized_b_mse"] * scale
        np.testing.assert_allclose(actual_b_mse, recorded_b_mse, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_b_mse, residual_b_mse, rtol=1e-12, atol=1e-12)
        if not np.all(np.isfinite(final_forecast)):
            raise AssertionError("Nonfinite nontrivial B forecast")
        if optimizer["nfev"] > config["max_objective_evaluations"]:
            raise AssertionError("Objective evaluation cap exceeded")
        if optimizer["best_normalized_b_mse"] > optimizer["initial_normalized_b_mse"]:
            raise AssertionError("Best objective is worse than identity")
        rows.append({"gate": gate, "eta": eta.tolist(), "nfev": optimizer["nfev"],
                     "status": optimizer["status"],
                     "initial_b_mse": optimizer["initial_normalized_b_mse"] * scale,
                     "recorded_best_b_mse": recorded_b_mse,
                     "reconstructed_final_forecast_b_mse": actual_b_mse,
                     "residual_form_b_mse": residual_b_mse})
    if not any(np.linalg.norm(row["eta"]) > 1e-6 for row in rows):
        raise AssertionError("Nontrivial check did not exercise temperature changes")
    return {"seed": 923751, "bank_seed": 923752,
            "scope": "Separate development arrays; no declared simulation-law data or D targets",
            "a_rows": 128, "b_rows": 72, "objective_budget": 64,
            "checks_passed": True, "gates": rows}


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(177)
        self.x = self.rng.normal(size=(64, 5))
        self.config = {**DEFAULT_CONFIG, "max_references": 8,
                       "alpha_grid": [0.1], "baseline_alpha_grid": [0.1, 1.0],
                       "max_objective_evaluations": 12,
                       "raw_rbf_landmarks": 8, "raw_rbf_bandwidth_factors": [1.0]}

    def test_identity_temperature_and_common_features(self):
        bank = SharedBank(43, self.config).fit(self.x)
        cache = bank.transform(self.x[:9], "check")
        for gate in GATES:
            f = feature_matrix(cache, gate, np.zeros(3))
            for start, end in cache["block_slices"]:
                expected = softmax(cache["logits"][gate][:, start:end], axis=1)
                np.testing.assert_allclose(f[:, start:end], expected, atol=1e-13)
                np.testing.assert_allclose(f[:, start:end].sum(axis=1), 1.0, atol=1e-13)
            self.assertEqual(f.shape[1], bank.diagnostics["feature_width"])
            self.assertTrue(np.all(np.isfinite(f)))

    def test_stable_kahm_matches_unshifted_query(self):
        bank = SharedBank(49, self.config).fit(self.x)
        z = (self.x[:7] - bank.mean) / bank.scale
        for model in bank.models:
            if model["rank"] == 0:
                continue
            query = ((z - model["center"]) @ model["p"]) @ model["whitening"]
            logs = -np.sum((query[:, None, :] - model["uq"][None, :, :]) ** 2, axis=2) / (2 * model["rank"])
            v1 = np.exp(logs) @ model["inverse"]
            v2 = np.exp(logs - logs.max(axis=1, keepdims=True)) @ model["inverse"]
            r1 = (v1 @ model["q"]) / v1.sum(axis=1, keepdims=True)
            r2 = (v2 @ model["q"]) / v2.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(r1, r2, rtol=1e-10, atol=1e-10)

    def test_selection_budget_and_zero_baseline_recovery(self):
        groups = np.repeat(np.arange(8), 8)
        xa = self.x
        xb, xc, xd = (self.rng.normal(size=(20, 5)) for _ in range(3))
        def response(x):
            return np.column_stack((x[:, 0] + 0.3 * x[:, 1] ** 2,
                                    x[:, 1] - 0.2 * np.sin(x[:, 2])))
        ya, yb = response(xa), response(xb)
        baseline, _ = _baseline_ridge(xa, ya, groups, xb, xc, xd, self.config)
        # Every positive correction must lose to an exactly correct C baseline.
        result = fit_suite(xa, ya, groups, xb, yb, xc, baseline["C"], xd, 933, self.config)
        diagnostics = result["diagnostics"]
        json.dumps(diagnostics, allow_nan=False)
        self.assertEqual(len(result["predictions_d"]), 10)
        self.assertEqual(diagnostics["failures"], [])
        for gate in GATES:
            for prefix in ("fixed_", "direct_"):
                name = prefix + gate
                np.testing.assert_array_equal(result["predictions_c"][name], baseline["C"])
                np.testing.assert_array_equal(result["predictions_d"][name], baseline["D"])
                for horizon in diagnostics["methods"][name]["horizons"]:
                    self.assertEqual(horizon["selected"]["gamma"], 0.0)
                    self.assertEqual(horizon["candidates"][0]["candidate_id"], "zero")
                    self.assertEqual(len(horizon["candidates"]), 4)
                    for opt in horizon["optimizations"]:
                        self.assertLessEqual(opt["nfev"], self.config["max_objective_evaluations"])
                        self.assertLessEqual(opt["best_normalized_b_mse"], opt["initial_normalized_b_mse"])
                        self.assertTrue(np.all(np.abs(opt["eta"]) <= np.log(4.0) + 1e-12))
        self.assertEqual(diagnostics["methods"]["raw_ridge"]["inner_training_groups"], list(range(6)))
        self.assertEqual(diagnostics["methods"]["raw_ridge"]["inner_validation_groups"], [6, 7])

    def test_degenerate_bank_remains_finite(self):
        constant = np.ones((20, 5))
        bank = SharedBank(1, self.config).fit(constant)
        cache = bank.transform(constant[:3], "constant")
        self.assertEqual(bank.diagnostics["effective_blocks"], [1, 1, 1])
        for gate in GATES:
            self.assertTrue(np.isfinite(feature_matrix(cache, gate, np.zeros(3))).all())

    def test_nontrivial_direct_objective_is_final_b_forecast_mse(self):
        result = run_nontrivial_objective_check()
        self.assertTrue(result["checks_passed"])
        self.assertTrue(any(row["recorded_best_b_mse"] < row["initial_b_mse"] - 1e-8
                            for row in result["gates"]))


if __name__ == "__main__":
    unittest.main()
