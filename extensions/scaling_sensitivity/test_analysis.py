"""Meaningful invariants for the independent descriptive sensitivity audit."""
from __future__ import annotations

import copy
import hashlib
import unittest

import numpy as np

from .analysis import (_natural_scaler_audit, grouped_square_errors,
                       hierarchical_weights, paired_summary)


class AnalysisTests(unittest.TestCase):
    def test_unbalanced_source_hierarchy_is_not_a_flat_task_mean(self):
        rows = [
            {"family": "a", "series": "one", "seed": 1},
            {"family": "a", "series": "one", "seed": 2},
            {"family": "a", "series": "two", "seed": 1},
            {"family": "b", "series": "one", "seed": 1},
        ]
        weights = hierarchical_weights(rows, ("family", "series", "seed"))
        np.testing.assert_array_equal(weights, [0.125, 0.125, 0.25, 0.5])
        self.assertNotEqual(float(weights @ [0, 0, 0, 4]), float(np.mean([0, 0, 0, 4])))

    def test_episode_aggregation_respects_unequal_origin_counts(self):
        target = np.zeros((3, 2))
        prediction = np.array([[1, 3], [3, 1], [5, 5]], dtype=float)
        losses = grouped_square_errors(prediction, target, np.array([2, 2, 7]))
        np.testing.assert_array_equal(losses, [[5, 5], [25, 25]])
        self.assertEqual(float(losses.mean()), 15.0)
        self.assertNotEqual(float(losses.mean()), float(((prediction - target) ** 2).mean()))

    def test_relative_change_is_ratio_of_aggregate_means(self):
        summary = paired_summary([2.0, 9.0], [1.0, 10.0])
        self.assertEqual(summary["relative_change_percent"], 0.0)
        self.assertEqual(summary["mean_difference"], 0.0)
        self.assertAlmostEqual(summary["difference_se"], 1.0)
        self.assertEqual(summary["candidate_lower_count"], 1)
        self.assertFalse(any("interval" in key or "p_value" in key for key in summary))

    def test_scaler_audit_catches_query_fitted_statistics_and_clipping(self):
        x = np.array([[1.0, 3.0], [2.0, 3.0], [4.0, 3.0]])
        query = np.array([[9.0, 4.0], [0.0, 3.0]])
        y = np.array([[0.0], [2.0], [4.0]])
        arrays = {"X_train": x, "X_refit": x, "Y_train": y, "Y_refit": y,
                  "X_validation": query, "X_test": query}
        minimum, maximum = x.min(axis=0), x.max(axis=0)
        span, constant = maximum - minimum, maximum == minimum
        raw_scale = x.std(axis=0)
        raw_scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
        header = f"{x.dtype.str}:{x.shape}:".encode()
        metadata = {
            "training_rows": len(x),
            "training_input_sha256": hashlib.sha256(header + x.tobytes()).hexdigest(),
            "mode": "minmax", "clip": False,
            "training_minimum": minimum.tolist(), "training_maximum": maximum.tolist(),
            "constant_columns": constant.tolist(),
            "raw_input_mean": x.mean(axis=0).tolist(), "raw_input_scale": raw_scale.tolist(),
            "effective_center": (minimum + span / 2).tolist(),
            "effective_scale": np.where(constant, 1.0, span / 2).tolist(),
            "target_mean": y.mean(axis=0).tolist(), "target_scale": y.std(axis=0).tolist(),
        }
        outside = (query < minimum) | (query > maximum)
        z = 2 * (query - minimum) / np.where(constant, 1.0, span) - 1
        z[:, constant] = 0
        diagnostic = {
            "query_rows": len(query), "query_coordinates": query.size,
            "outside_training_range_coordinates": int(outside.sum()),
            "outside_training_range_rows": int(outside.any(axis=1).sum()),
            "outside_training_range_coordinate_fraction": float(outside.mean()),
            "outside_training_range_row_fraction": float(outside.any(axis=1).mean()),
            "maximum_abs_minmax_coordinate": float(np.abs(z).max()),
        }
        candidate = {"candidate_id": "raw_kahm_rbf__minmax", "geometry_scaling": "minmax",
                     "search_scaler_metadata": metadata, "refit_scaler_metadata": metadata,
                     "validation_out_of_range": diagnostic, "test_out_of_range": diagnostic}
        self.assertEqual(_natural_scaler_audit(candidate, arrays), 2)
        changed = copy.deepcopy(candidate)
        changed["search_scaler_metadata"]["training_maximum"][0] = 9.0
        with self.assertRaisesRegex(ValueError, "maximum"):
            _natural_scaler_audit(changed, arrays)
        changed = copy.deepcopy(candidate)
        changed["refit_scaler_metadata"]["clip"] = True
        with self.assertRaisesRegex(ValueError, "clipping"):
            _natural_scaler_audit(changed, arrays)


if __name__ == "__main__":
    unittest.main()
