"""Numerical and information-boundary checks using fabricated data only."""
from __future__ import annotations

import copy
import inspect
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from extensions.direct_reconstruction.model import SharedBank
from extensions.reconstruction_mechanism import models


SMALL = {"blocks": [2, 3], "rank": 2, "max_references": 8,
         "bank_row_cap": 64, "alpha_grid": [.001, .1, 1.],
         "gamma_grid": [0., .25, .5, 1.], "raw_rbf_landmarks": 8,
         "raw_rbf_bandwidth_factors": [.5, 1.], "raw_hgb_learning_rates": [.05],
         "raw_hgb_max_leaf_nodes": [7], "raw_hgb_max_iter": 5}


def development_arrays():
    rng = np.random.default_rng(732001)
    x = rng.normal(size=(128, 6))
    y = np.column_stack((.5 * x[:, -1] + .8 * x[:, 0] ** 2,
                         .4 * x[:, -2] + np.sin(x[:, 1])))
    return x[:64], y[:64], x[64:88], y[64:88], x[88:112], y[88:112], x[112:]


class RidgeTests(unittest.TestCase):
    def test_mean_row_objective_matches_augmented_least_squares(self):
        rng = np.random.default_rng(732002)
        x = rng.normal(size=(31, 7)) + 3.
        x[:, 6] = x[:, 0] + x[:, 1]  # explicit singular design
        y = rng.normal(size=(31, 3)) + np.array([5., -2., 1.])
        alpha = .17
        head = models._ridge_path(x, y, [alpha])[0]
        design = np.vstack((x - x.mean(axis=0), np.sqrt(len(x) * alpha) * np.eye(x.shape[1])))
        response = np.vstack((y - y.mean(axis=0), np.zeros((x.shape[1], y.shape[1]))))
        expected = np.linalg.lstsq(design, response, rcond=None)[0]
        np.testing.assert_allclose(head["coef"], expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(models._ridge_predict(head, x).mean(axis=0), y.mean(axis=0), atol=1e-13)
        self.assertEqual(head["feature_rank"], 6)

    def test_free_intercept_and_no_implicit_feature_scaling(self):
        rng = np.random.default_rng(732003)
        x = rng.normal(size=(40, 3)) * np.array([1., 4., .1])
        y = rng.normal(size=(40, 2))
        first = models._ridge_path(x, y, [.3])[0]
        second = models._ridge_path(x + 9., y + np.array([8., -2.]), [.3])[0]
        np.testing.assert_allclose(first["coef"], second["coef"], atol=1e-13)
        np.testing.assert_allclose(models._ridge_predict(second, x + 9.) - models._ridge_predict(first, x),
                                   np.tile([8., -2.], (len(x), 1)), atol=2e-13)
        gradient = (x - x.mean(axis=0)).T @ (models._ridge_predict(first, x) - y) / len(x) + .3 * first["coef"]
        np.testing.assert_allclose(gradient, 0., atol=1e-13)

    def test_B_selects_alpha_per_horizon(self):
        rng = np.random.default_rng(732004)
        xa, xb = rng.normal(size=(40, 4)), rng.normal(size=(21, 4))
        ya = rng.normal(size=(40, 2))
        candidates = models._ridge_path(xa, ya, [.001, 1.])
        yb = np.column_stack((models._ridge_predict(candidates[0], xb)[:, 0],
                              models._ridge_predict(candidates[1], xb)[:, 1]))
        selected, record = models._select_ridge(xa, ya, xb, yb, np.zeros_like(yb), [.001, 1.], {}, "test")
        np.testing.assert_array_equal(selected["alpha_by_horizon"], [.001, 1.])
        self.assertEqual([r["candidate_index"] for r in record["selected"]], [0, 1])

    def test_C_gamma_is_horizon_specific_and_zero_is_exact(self):
        rng = np.random.default_rng(732005)
        baseline = rng.normal(size=(17, 3))
        candidate = baseline + rng.normal(size=(17, 3))
        truth = models._blend(baseline, candidate, [0., .5, 1.])
        gamma, prediction, record = models._select_gamma(truth, baseline, candidate, [0., .25, .5, 1.])
        np.testing.assert_array_equal(gamma, [0., .5, 1.])
        np.testing.assert_array_equal(prediction[:, 0], baseline[:, 0])
        np.testing.assert_array_equal(prediction, truth)
        self.assertEqual(record["gamma_by_horizon"], [0., .5, 1.])
        invalid = np.full_like(baseline, np.nan)
        np.testing.assert_array_equal(models._blend(baseline, invalid, [0., 0., 0.]), baseline)

    def test_sharp_centered_ridge_bound_in_actual_decoder_coordinates(self):
        rng = np.random.default_rng(732006)
        a = models.radial_bound(rng.normal(size=(47, 8)))
        y = rng.normal(size=(47, 3)) + [2., -.5, 1.]
        head = models._ridge_path(a, y, [.007])[0]
        gamma = np.array([0., .25, 1.])
        bound = models._readout_bound(head, np.sqrt(a.shape[1]), gamma)
        self.assertEqual(bound["coefficient_bound_violations"], 0)
        query = models.radial_bound(rng.normal(size=(93, 8)) * 100.)
        correction = models._ridge_predict(head, query) * gamma
        self.assertTrue(np.all(np.max(np.abs(correction), axis=0) <= np.array(bound["correction_bound_by_horizon"]) + 1e-12))


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = development_arrays()[0]
        cls.bank = models._InstrumentedBank(seed=732001, config={**models.DEFAULT_CONFIG, **SMALL}).fit(cls.x)
        cls.state = cls.bank.state("kahm")
        cls.raw, cls.diag = models._raw_feature_maps(cls.state, cls.x[:17])

    def test_common_map_agrees_with_fixed_inherited_map_and_gates(self):
        old = SharedBank(seed=732001, config=SMALL).fit(self.x)
        expected = old.features(old.prepare(self.x[:17]), "rbf_local")
        np.testing.assert_allclose(self.raw["common"], expected, rtol=3e-13, atol=3e-13)
        for left, right in zip(self.bank.models, old.models):
            np.testing.assert_array_equal(left["q"], right["q"])
            np.testing.assert_array_equal(left["p"], right["p"])
            if left["rank"]:
                np.testing.assert_array_equal(left["inverse"], right["inverse"])
        self.assertLess(self.diag["gate_block_sum_max_abs_error"], 1e-14)

    def test_positive_and_signed_reconstructions_have_declared_equations(self):
        kernel = np.array([[.2, .8, .4], [1., .1, .5]])
        q = np.array([[1., 3.], [2., -1.], [-2., .5]])
        rbf, record = models.normalized_reconstruction(kernel, q)
        np.testing.assert_allclose(rbf, kernel @ q / kernel.sum(axis=1, keepdims=True), atol=1e-15)
        self.assertGreaterEqual(record["minimum_normalized_weight"], 0)
        self.assertLess(record["max_weight_sum_error"], 1e-14)
        inverse = np.array([[1., -.2, 0.], [-.2, 2., -.1], [0., -.1, 1.]])
        kahm, record = models.normalized_reconstruction(kernel, q, inverse=inverse)
        weight = kernel @ inverse
        np.testing.assert_allclose(kahm, weight @ q / weight.sum(axis=1, keepdims=True), atol=1e-15)
        self.assertEqual(record["fallback_count"], 0)

    def test_signed_cancellation_falls_back_only_affected_row(self):
        kernel = np.array([[1., 1.], [1., .2]])
        q = np.array([[2., 0.], [0., 3.]])
        inverse = np.diag([1., -1. + 1e-12])
        affine = np.array([[.5, .7], [-1., 2.]])
        result, record = models.normalized_reconstruction(kernel, q, inverse=inverse, fallback=affine)
        np.testing.assert_array_equal(result[0], affine[0])
        self.assertFalse(np.array_equal(result[1], affine[1]))
        self.assertEqual(record["fallback_rows"], [0])
        self.assertEqual(record["fallback_count"], 1)

    def test_departures_use_the_same_projected_query_kernel(self):
        prepared = self.bank.prepare(self.x[:17])
        gates = self.raw["common"][:, :len(self.bank.models)]
        for j, (local, item) in enumerate(zip(self.bank.models, prepared["local"])):
            if not local["rank"]:
                continue
            logs = -item["query_distance"]
            kernel = np.exp(logs - logs.max(axis=1, keepdims=True))
            for kind, inverse in (("kahm", local["inverse"]), ("rbf", None)):
                reconstruction, _ = models.normalized_reconstruction(kernel, local["q"], inverse=inverse, fallback=item["affine"])
                expected = gates[:, j:j + 1] * (reconstruction - item["affine"])
                actual = self.raw[kind][:, j * self.x.shape[1]:(j + 1) * self.x.shape[1]]
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    def test_log_rescaling_avoids_kernel_underflow(self):
        raw, diag = models._raw_feature_maps(self.state, np.full((3, self.x.shape[1]), 1e5))
        self.assertTrue(all(np.isfinite(value).all() for value in raw.values()))
        self.assertEqual(diag["rbf_fallback_count"], 0)
        self.assertEqual(diag["kernel_fallback_rows"], [])

    def test_radial_bound_matches_formula_and_handles_extreme_finite_values(self):
        rng = np.random.default_rng(732007)
        x = rng.normal(size=(31, 8))
        expected = x / np.sqrt(1 + (x * x).sum(axis=1, keepdims=True) / x.shape[1])
        np.testing.assert_allclose(models.radial_bound(x), expected, rtol=3e-15, atol=3e-15)
        extreme = np.vstack((np.full(8, 1e308), np.full(8, -1e308), np.zeros(8)))
        actual = models.radial_bound(extreme)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all(np.linalg.norm(actual, axis=1) <= np.sqrt(8) * (1 + 1e-14)))

    def test_bound_and_unbounded_arms_share_common_and_pretransform_scales(self):
        raw, _ = models._raw_feature_maps(self.state, self.x)
        scalers = {key: models._scaler_fit(value, 1e-6) for key, value in raw.items()}
        transform = models.ReconstructionFeatures(self.state, scalers)
        features, diagnostics = transform.transform(self.x[:17])
        width = features["common"].shape[1]
        for method in models.GEOMETRY_METHODS:
            np.testing.assert_array_equal(features[method][:, :width], features["common"])
        for kind in ("kahm", "rbf"):
            unbounded = features[f"{kind}_unbounded"][:, width:]
            np.testing.assert_array_equal(features[f"{kind}_departure"][:, width:], models.radial_bound(unbounded))
        self.assertTrue(all(item["violations"] == 0 for item in diagnostics["feature_bounds"].values()))

    def test_constant_rank_zero_bank_is_defined(self):
        bank = models._InstrumentedBank(seed=732008, config={**models.DEFAULT_CONFIG, **SMALL}).fit(np.ones((64, 6)))
        raw, diagnostics = models._raw_feature_maps(bank.state("kahm"), np.ones((5, 6)))
        self.assertTrue(all(item["rank"] == 0 for item in bank.models))
        np.testing.assert_array_equal(raw["kahm"], 0.)
        np.testing.assert_array_equal(raw["rbf"], 0.)
        self.assertEqual(diagnostics["kahm_fallback_count"], 0)


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays = development_arrays()
        cls.suite = models.fit_suite(*cls.arrays[:6], seed=732001, config=SMALL)
        cls.result = cls.suite.predict(cls.arrays[6])

    def test_API_never_accepts_reserved_target_or_features_during_fit(self):
        names = list(inspect.signature(models.fit_suite).parameters)
        self.assertEqual(names, ["xa", "ya", "xb", "yb", "xc", "yc", "seed", "config"])
        self.assertEqual(list(inspect.signature(models.ForecastSuite.predict).parameters), ["self", "x"])

    def test_complete_roster_finite_predictions_and_policy_identity(self):
        self.assertEqual(tuple(self.result["predictions"]), models.METHODS)
        for prediction in self.result["predictions"].values():
            self.assertEqual(prediction.shape, (16, 2))
            self.assertTrue(np.isfinite(prediction).all())
        selected = self.suite.diagnostics["policy"]["selected_method"]
        self.assertIn(selected, models.NONKAHM_METHODS)
        np.testing.assert_array_equal(self.result["predictions"]["nonkahm_policy"], self.result["predictions"][selected])
        self.assertEqual(self.suite.diagnostics["method_failures"], {})
        self.assertTrue(all(n == 0 for n in self.result["diagnostics"]["recovered_prediction_cells"].values()))

    def test_calibration_logs_and_bound_diagnostics_are_complete(self):
        self.assertEqual(set(self.suite.diagnostics["B"]), set(models.METHODS[:-1]))
        for method in models.METHODS[1:-1]:
            record = self.suite.diagnostics["C"][method]
            self.assertEqual(len(record["candidates"]), 4)
            self.assertEqual(len(record["gamma_by_horizon"]), 2)
        for method, bound in self.result["diagnostics"]["bounds"].items():
            self.assertEqual(bound["coefficient_bound_violations"], 0)
            if not method.endswith("unbounded"):
                self.assertEqual(bound["violations"], 0)
        self.assertEqual(set(self.result["diagnostics"]["feature_bounds"]), {"common", "kahm_departure", "rbf_departure"})

    def test_pickle_replay_and_no_prediction_state_mutation(self):
        before = pickle.dumps(self.suite, protocol=5)
        replayed = pickle.loads(before)
        actual = replayed.predict(self.arrays[6])
        for method in models.METHODS:
            np.testing.assert_array_equal(actual["predictions"][method], self.result["predictions"][method])
        self.suite.predict(self.arrays[6])
        self.assertEqual(before, pickle.dumps(self.suite, protocol=5))
        self.assertLess(len(before), 500000)

    def test_common_scaler_is_fitted_only_to_A(self):
        raw, _ = models._raw_feature_maps(self.suite.geometry.bank, self.arrays[0])
        for key in ("common", "kahm", "rbf"):
            expected = models._scaler_fit(raw[key], 1e-6)
            np.testing.assert_array_equal(self.suite.geometry.scalers[key]["mean"], expected["mean"])
            np.testing.assert_array_equal(self.suite.geometry.scalers[key]["scale"], expected["scale"])

    def test_query_failure_retained_with_baseline_recovery(self):
        suite = copy.deepcopy(self.suite)
        suite.raw_rbf[0]["head"]["coef"][:] = np.nan
        result = suite.predict(self.arrays[6])
        self.assertEqual(result["diagnostics"]["recovered_prediction_cells"]["raw_rbf"], 16)
        np.testing.assert_array_equal(result["predictions"]["raw_rbf"][:, 0], result["predictions"]["ridge"][:, 0])
        self.assertTrue(any(item["method"] == "raw_rbf" for item in result["diagnostics"]["failures"]))

    def test_whole_method_failure_is_logged_and_gamma_zero(self):
        with patch.object(models, "_fit_raw_rbf", side_effect=RuntimeError("fabricated solver failure")):
            suite = models.fit_suite(*self.arrays[:6], seed=732001, config=SMALL)
        self.assertIn("raw_rbf", suite.diagnostics["method_failures"])
        np.testing.assert_array_equal(suite.gammas["raw_rbf"], 0.)
        prediction = suite.predict(self.arrays[6])["predictions"]
        np.testing.assert_array_equal(prediction["raw_rbf"], prediction["ridge"])

    def test_resource_attribution_excludes_K_inverse_from_RBF_and_common(self):
        resources = self.suite.resources["per_method"]
        self.assertNotIn("bank_kahm_factor", resources["rbf_departure"]["fit_and_selection"]["nodes"])
        self.assertNotIn("bank_kahm_factor", resources["common"]["fit_and_selection"]["nodes"])
        self.assertIn("bank_kahm_factor", resources["kahm_departure"]["fit_and_selection"]["nodes"])
        policy = set(resources["nonkahm_policy"]["fit_and_selection"]["nodes"])
        for method in models.NONKAHM_METHODS:
            self.assertTrue(set(resources[method]["fit_and_selection"]["nodes"]) <= policy)
        for method in models.METHODS:
            self.assertGreater(resources[method]["state"]["pickle_bytes"], 0)
            self.assertGreaterEqual(resources[method]["state"]["array_buffer_bytes"], 0)
            self.assertFalse(any("rank_diagnostic" in name for name in resources[method]["fit_and_selection"]["nodes"]))


if __name__ == "__main__":
    unittest.main()
