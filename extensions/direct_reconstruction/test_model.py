"""Numerical and scientific-contract tests, using development data only."""
from __future__ import annotations

import inspect
import math
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from extensions.direct_reconstruction import model
from extensions.target_geometry.models import RobustLocalFeatureBank


SMALL = {"blocks": [2, 3], "rank": 2, "max_references": 8,
         "bank_row_cap": 64, "alpha_grid": [0.1], "gamma_grid": [0, 0.5, 1],
         "baseline_alpha_grid": [0.1, 1.0], "max_objective_evaluations": 8,
         "optimizer_maxiter": 2, "raw_rbf_landmarks": 8,
         "raw_rbf_bandwidth_factors": [1.0], "raw_hgb_learning_rates": [0.05],
         "raw_hgb_max_leaf_nodes": [7], "raw_hgb_max_iter": 5}


def development_arrays():
    rng = np.random.default_rng(932001)
    x = rng.normal(size=(136, 6))
    y = np.column_stack((x[:, -1] + 0.8 * x[:, 0] ** 2,
                         0.4 * x[:, -2] + np.sin(x[:, 1])))
    return x[:64], y[:64], np.repeat(np.arange(8), 8), x[64:88], y[64:88], x[88:112], y[88:112], x[112:]


class BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = development_arrays()[0]
        cls.bank = model.SharedBank(seed=932001, config=SMALL).fit(cls.x)
        cls.prepared = cls.bank.prepare(cls.x[:17])

    def test_zero_features_agree_with_recovered_upstream(self):
        for gate in model.GATES:
            source_gate = "rbf" if gate == "rbf_local" else gate
            source = RobustLocalFeatureBank(source_gate, seed=932001,
                cluster_counts=(2, 3), rank=2, max_refs=8, max_fit=64).fit(self.x)
            expected = source.transform(self.x[:17])
            actual = self.bank.features(self.prepared, gate)
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
            for left, right in zip(self.bank.models, source.models_):
                np.testing.assert_array_equal(left["reference_indices"], right.reference_indices)
                np.testing.assert_allclose(left["p"], right.basis, atol=0, rtol=0)
                if gate == "kahm" and left["rank"]:
                    np.testing.assert_allclose(left["inverse"], right.ae["B"], atol=2e-13, rtol=2e-13)

    def test_original_lambda_is_inverse_regularizer_not_next_update(self):
        for local in self.bank.models:
            if local["rank"]:
                kernel = np.exp(-local["reference_distance"])
                expected = np.linalg.inv(kernel + local["lambda_original"] * np.eye(len(kernel)))
                np.testing.assert_allclose(local["inverse"], expected, atol=1e-13, rtol=1e-13)

    def test_bandwidth_changes_both_reference_and_query_kernel(self):
        xi = np.array([0.3, -0.2])
        actual = self.bank._compute_logits(self.prepared, "kahm", xi)
        for j, (local, values) in enumerate(zip(self.bank.models, self.prepared["local"])):
            if not local["rank"]:
                continue
            value = xi[local["block"]]
            kernel_ref = np.exp(-local["reference_distance"] / np.exp(2 * value))
            inverse = np.linalg.inv(kernel_ref + local["lambda_original"] * np.eye(len(kernel_ref)))
            logs = -values["query_distance"] / np.exp(2 * value)
            kernel_query = np.exp(logs - logs.max(axis=1, keepdims=True))
            reconstruction, _ = model._guarded_reconstruction(kernel_query, inverse, local["q"], 1e-10)
            expected = model._folding_logits(self.prepared["z"], reconstruction)
            np.testing.assert_allclose(actual[:, j], expected, atol=3e-12, rtol=3e-12)
            self.assertFalse(np.allclose(inverse, local["inverse"]))

    def test_gate_only_is_nested_and_blocks_sum_to_one(self):
        eta = np.array([0.2, -0.25])
        for gate in model.GATES:
            features = self.bank.features(self.prepared, gate, np.zeros(2), eta)
            fixed_logits = self.bank._compute_logits(self.prepared, gate, np.zeros(2))
            for b, (start, end) in enumerate(self.bank.block_slices):
                scores = np.exp(eta[b]) * fixed_logits[:, start:end]
                expected = np.exp(scores - model.logsumexp(scores, axis=1, keepdims=True))
                np.testing.assert_allclose(features[:, start:end], expected, atol=1e-14, rtol=1e-14)
                np.testing.assert_allclose(features[:, start:end].sum(axis=1), 1, atol=1e-14)

    def test_parameterized_control_shapes_have_declared_equations(self):
        xi = np.array([0.31, -0.18])
        z = self.prepared["z"]
        actual = {gate: self.bank._compute_logits(self.prepared, gate, xi)
                  for gate in ("affine", "locality", "rbf_local")}
        for j, (local, values) in enumerate(zip(self.bank.models, self.prepared["local"])):
            v = xi[local["block"]]
            affine = local["center"] + np.exp(v) * ((z - local["center"]) @ local["p"]) @ local["p"].T
            np.testing.assert_allclose(actual["affine"][:, j], model._folding_logits(z, affine), atol=2e-13)
            expected_locality = (-np.exp(v) * values["normal_d2"] / local["projection_scale"]
                                 - 0.1 * np.exp(-v) * values["center_d2"] / local["radius"])
            np.testing.assert_allclose(actual["locality"][:, j], expected_locality, atol=1e-13)
            expected_rbf = -0.5 * (np.exp(v) * values["tangent_d2"] + np.exp(-v) * values["normal_d2"]) / local["radius"]
            np.testing.assert_allclose(actual["rbf_local"][:, j], expected_rbf, atol=1e-13)
            np.testing.assert_allclose(values["tangent_d2"] + values["normal_d2"], values["center_d2"], atol=3e-13)

    def test_common_signed_denominator_guard_rejects_cancellation(self):
        kernel = np.array([[1.0, 1.0]])
        inverse = np.array([[1.0, 0], [0, -1.0 + 1e-12]])
        refs = np.array([[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaisesRegex(FloatingPointError, "signed denominator guard"):
            model._guarded_reconstruction(kernel, inverse, refs, 1e-10)

    def test_fixed_K_D_failure_is_isolated_from_controls_and_other_bandwidths(self):
        bank = model.SharedBank(seed=932001, config=SMALL).fit(self.x)
        prepared = bank.prepare(self.x[:17], "D")
        original = bank._compute_logits

        def fail_fixed_only(query, gate, xi):
            if gate == "kahm" and not np.any(xi):
                raise FloatingPointError("injected fixed K D guard failure")
            return original(query, gate, xi)

        with patch.object(bank, "_compute_logits", side_effect=fail_fixed_only):
            for gate in model.GATES:
                bank.precompute_fixed(prepared, gate, "D")
            with self.assertRaisesRegex(FloatingPointError, "injected fixed K"):
                bank.features(prepared, "kahm")
            for gate in model.GATES[1:]:
                self.assertTrue(np.isfinite(bank.features(prepared, gate)).all())
            self.assertTrue(np.isfinite(bank.features(prepared, "kahm", [0.2, 0.2])).all())

    def test_rank_zero_constant_bank_remains_finite(self):
        bank = model.SharedBank(seed=932001, config=SMALL).fit(np.ones((64, 6)))
        prepared = bank.prepare(np.ones((3, 6)))
        self.assertTrue(all(local["rank"] == 0 for local in bank.models))
        for gate in model.GATES:
            self.assertTrue(np.isfinite(bank.features(prepared, gate)).all())

    def test_control_serialization_excludes_kahm_factors_and_replays(self):
        for gate in model.GATES:
            restored = model.SharedBank.from_state(pickle.loads(pickle.dumps(self.bank.state(gate))))
            features = restored.features(restored.prepare(self.x[:17]), gate, [0.12, -0.1], [0.1, -0.1])
            expected = self.bank.features(self.prepared, gate, [0.12, -0.1], [0.1, -0.1])
            np.testing.assert_allclose(features, expected, atol=1e-13, rtol=1e-13)
            if gate != "kahm":
                self.assertTrue(all("inverse" not in local and "q" not in local for local in restored.models))


class ProcedureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays = development_arrays()
        cls.result = model.fit_suite(*cls.arrays, seed=932001, config=SMALL,
                                    horizons=[1, 3], naive_period=3)

    def test_no_test_targets_in_model_api(self):
        names = {name.lower() for name in inspect.signature(model.fit_suite).parameters}
        self.assertNotIn("yd", names)
        self.assertNotIn("y_d", names)
        self.assertNotIn("targets_d", names)

    def test_complete_finite_roster_and_selected_model_replay(self):
        result = self.result
        self.assertEqual(tuple(result["predictions_d"]), model.METHODS)
        self.assertEqual(len(model.METHODS), 17)
        for name in model.METHODS:
            self.assertTrue(np.isfinite(result["predictions_d"][name]).all())
            expected = result["predictions_d"][name]
            actual = model.predict_serialized(result["serialized_models"][name], self.arrays[-1])
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
            c_actual = model.predict_serialized(result["serialized_models"][name], self.arrays[5])
            np.testing.assert_allclose(c_actual, result["predictions_c"][name], atol=2e-12, rtol=2e-12)

    def test_optimizer_actual_budget_bounds_and_identity_incumbent(self):
        for name, record in self.result["diagnostics"]["methods"].items():
            for horizon in record.get("horizons", []):
                for optimization in horizon["optimizations"]:
                    evaluations = optimization["evaluations"]
                    self.assertEqual(optimization["nfev"], len(evaluations))
                    self.assertLessEqual(len(evaluations), SMALL["max_objective_evaluations"])
                    self.assertLessEqual(optimization["best_normalized_b_mse"], optimization["initial_normalized_b_mse"])
                    for evaluation in evaluations:
                        self.assertLessEqual(max(np.abs(evaluation["xi"] + evaluation["eta"])), math.log(4) + 1e-12)
                        if name.startswith("gate_"):
                            self.assertTrue(all(value == 0 for value in evaluation["xi"]))

    def test_C_selection_is_first_strict_minimum(self):
        for record in self.result["diagnostics"]["methods"].values():
            for horizon in record.get("horizons", []):
                valid = [candidate for candidate in horizon["candidates"] if candidate["c_mse"] is not None]
                expected = min(valid, key=lambda item: item["c_mse"])
                self.assertEqual(expected["candidate_id"], horizon["selected"]["candidate_id"])
        policy = self.result["diagnostics"]["methods"]["nonkahm_policy"]
        self.assertEqual(policy["selected_method"], min(policy["candidates"], key=lambda item: item["c_mse"])["method"])
        self.assertFalse("kahm" in policy["selected_method"])

    def test_cost_dependency_union_serialization_and_no_kernel_charge_to_controls(self):
        diagnostics = self.result["diagnostics"]
        nodes, records = diagnostics["cost_nodes"], diagnostics["methods"]
        union = set().union(*(set(records[name]["cost_dependencies"]) for name in model.NONKAHM_METHODS))
        self.assertEqual(set(records["nonkahm_policy"]["cost_dependencies"]), union | {"nonkahm_policy_selection"})
        for name, record in records.items():
            expected = sum(nodes[key]["wall_seconds"] for key in set(record["cost_dependencies"]) if key in nodes)
            self.assertAlmostEqual(record["charged_fit_wall_seconds"], expected)
            self.assertEqual(record["serialized_bytes"], len(self.result["serialized_models"][name]))
            if name.endswith(("affine", "locality", "rbf_local")):
                self.assertNotIn("kahm_factors", record["cost_dependencies"])

    def test_naive_forecast_uses_declared_horizon_and_available_history(self):
        x = np.arange(12).reshape(2, 6)
        state = {"kind": "seasonal_naive", "period": 3, "horizons": [1, 3, 4, 7]}
        np.testing.assert_array_equal(model.predict_state(state, x), x[:, [3, 5, 3, 3]])

    def test_purged_inner_masks_are_used_and_all_A_refits_final_baseline(self):
        xa, ya = self.arrays[:2]
        fit = np.arange(len(xa)) < 40
        validation = np.arange(len(xa)) >= 52
        prediction, record, state = model._baseline_ridge(
            xa, ya, self.arrays[2], {"A": xa}, {**model.DEFAULT_CONFIG, **SMALL}, fit, validation)
        self.assertEqual(record["inner_purged_rows"], list(range(40, 52)))
        selected = record["selected_alpha"]
        expected = model._ridge_fit(xa, ya, selected)
        np.testing.assert_array_equal(state["head"]["mean"], xa.mean(axis=0))
        np.testing.assert_allclose(prediction["A"], model._ridge_predict(expected, xa), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            model._baseline_ridge(xa, ya, self.arrays[2], {"A": xa}, model.DEFAULT_CONFIG, fit, fit)

    def test_D_query_failure_replays_entire_horizon_baseline(self):
        # Force a deterministic numerical guard failure only in the selected
        # deployment map. The pure replay API must not emit a partly corrected
        # horizon or invent a replacement comparator.
        xa, ya = self.arrays[:2]
        bank = model.SharedBank(seed=932001, config=SMALL).fit(xa)
        head = model._ridge_fit(bank.features(bank.prepare(xa), "kahm"), ya[:, 0], 0.1)
        baseline = model._ridge_fit(xa, ya, 0.1)
        state = {"kind": "residual_geometry", "gate": "kahm", "baseline": baseline,
                 "bank": bank.state("kahm"), "selected": [
                     {"head": head, "xi": np.zeros(2), "eta": np.zeros(2), "gamma": 1}, None]}
        state["bank"]["config"]["relative_denominator_tolerance"] = 2.0
        prediction = model.predict_state(state, self.arrays[-1])
        np.testing.assert_array_equal(prediction, model._ridge_predict(baseline, self.arrays[-1]))

    def test_raw_D_only_failure_preserves_C_selection_and_other_horizons(self):
        original = model._raw_prediction_horizon

        def fail_first_D_horizon(state, x, horizon, features=None):
            if horizon == 0 and np.array_equal(x, self.arrays[-1]):
                raise FloatingPointError("injected raw D-only failure")
            return original(state, x, horizon, features)

        with patch.object(model, "_raw_prediction_horizon", side_effect=fail_first_D_horizon):
            failed = model.fit_suite(*self.arrays, seed=932001, config=SMALL,
                                      horizons=[1, 3], naive_period=3)
            for name in model.METHODS:
                np.testing.assert_array_equal(failed["predictions_c"][name], self.result["predictions_c"][name])
            for name in ("raw_rbf", "raw_hgb"):
                record = failed["diagnostics"]["methods"][name]
                self.assertEqual(record["selected"], self.result["diagnostics"]["methods"][name]["selected"])
                self.assertEqual(record["d_fallback_horizons"], [0])
                self.assertEqual(len(record["fallbacks"]), 1)
                np.testing.assert_array_equal(failed["predictions_d"][name][:, 0], failed["predictions_d"]["ridge"][:, 0])
                np.testing.assert_array_equal(failed["predictions_d"][name][:, 1], self.result["predictions_d"][name][:, 1])
                replay = model.predict_serialized(failed["serialized_models"][name], self.arrays[-1])
                np.testing.assert_array_equal(replay, failed["predictions_d"][name])
            self.assertEqual(failed["diagnostics"]["methods"]["nonkahm_policy"]["selected_method"],
                             self.result["diagnostics"]["methods"]["nonkahm_policy"]["selected_method"])


if __name__ == "__main__":
    unittest.main()
