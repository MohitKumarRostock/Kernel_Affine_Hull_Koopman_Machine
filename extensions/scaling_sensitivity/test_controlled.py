"""Implementation checks on toy inputs, never additional research outcomes."""
from __future__ import annotations

import unittest
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from .controlled import GeometryBank, exceedance, historical_parity, locality_coordinates


class ControlledScalingTests(unittest.TestCase):
    def test_minmax_is_train_only_unclipped_and_raw_is_identical(self):
        rng = np.random.default_rng(913)
        x = rng.normal(size=(48, 4))
        x[:, 3] = 3.0
        query = np.vstack([x[0], x.max(axis=0) + 5.0])
        options = dict(seed=29, cluster_counts=(2,), rank=1, max_refs=8, max_fit=40)
        with threadpool_limits(limits=1):
            standard = GeometryBank(scaling="standard", **options).fit(x)
            minmax = GeometryBank(scaling="minmax", **options).fit(x)
            raw_standard = standard.transform(query)["raw"]
            raw_minmax = minmax.transform(query)["raw"]
        np.testing.assert_array_equal(raw_standard, raw_minmax)
        np.testing.assert_array_equal(minmax.input_min_, x.min(axis=0))
        np.testing.assert_array_equal(minmax.input_max_, x.max(axis=0))
        z = minmax._scale_inputs(query)
        self.assertTrue(np.all(z[-1, :3] > 1.0))
        np.testing.assert_array_equal(z[:, 3], 0.0)
        observed = exceedance(minmax, query)
        self.assertEqual(observed["constant_training_coordinates"], 1)
        self.assertEqual(observed["constant_coordinate_changed_count"], 1)
        self.assertEqual(observed["clipped_coordinates"], 0)
        self.assertEqual(observed["outside_coordinate_count"], 3)

    def test_locality_control_matches_independent_gate_and_coordinate_formula(self):
        rng = np.random.default_rng(211)
        x = rng.normal(size=(48, 4))
        query = rng.normal(size=(7, 4))
        with threadpool_limits(limits=1):
            bank = GeometryBank(seed=13, cluster_counts=(2, 3), rank=2,
                                max_refs=8, max_fit=40, scaling="minmax").fit(x)
            generated = bank.transform(query)
        actual = locality_coordinates(bank, query, generated)
        z = bank._scale_inputs(query)
        affinities, gated = [], []
        for block in bank.blocks_:
            scores, coords = [], []
            for model in block["models"]:
                centered = z - model.center
                coordinate = centered @ model.basis
                reconstruction = model.center + coordinate @ model.basis.T
                scores.append(np.sum((z - reconstruction)**2, axis=1) / model.residual_scale
                              + 0.1 * np.sum(centered**2, axis=1) / model.scale)
                coords.append(coordinate)
            score = np.column_stack(scores)
            phi = np.exp(-score - logsumexp(-score, axis=1, keepdims=True))
            affinities.append(phi)
            gated.extend(phi[:, i:i+1] * coordinate for i, coordinate in enumerate(coords))
        expected = np.concatenate([np.concatenate(affinities, axis=1)] + gated, axis=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    def test_historical_parity_rejects_changed_predictions_or_selection(self):
        record = {
            "test_mse": 0.25, "test_mse_by_target": [0.25],
            "selected": [{"alpha": 0.1, "bandwidth_multiplier": 1.0,
                          "basis_dimension": 7, "bandwidth_sq": 2.0,
                          "validation_mse_by_target": [0.3]}],
            "prediction_key": "p", "unit_loss_key": "l",
        }
        prediction, losses = np.ones((3, 1)), np.full((3, 1), 0.25)
        original = {"p": prediction.copy(), "l": losses.copy()}
        result = historical_parity(record, prediction, losses, record, original)
        self.assertTrue(result["full_prediction_checked"])
        self.assertEqual(result["prediction_max_absolute_error"], 0.0)
        with self.assertRaises(AssertionError):
            historical_parity(record, prediction + 0.01, losses, record, original)
        changed = dict(record, selected=[dict(record["selected"][0], alpha=1.0)])
        with self.assertRaises(AssertionError):
            historical_parity(changed, prediction, losses, record, original)


if __name__ == "__main__":
    unittest.main()
