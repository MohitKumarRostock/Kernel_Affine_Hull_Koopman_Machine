"""Checks of certificate misuse rejection and duplicate-state invariance."""
import json
import unittest

import numpy as np

from certified_forecasting.certificates import (
    certify_forecast, mean_interval, paired_improvement, pair_witness_scores,
)
from certified_forecasting.models import BlockBoundedRidge


class ProtocolTests(unittest.TestCase):
    def test_overlapping_windows_cannot_be_silently_flattened(self):
        with self.assertRaises(ValueError):
            mean_interval(np.zeros((10, 20)), 1.0, 0.05)
        with self.assertRaises(ValueError):
            paired_improvement(np.zeros((10, 20)), np.zeros((10, 20)), 1.0, 0.05)

    def test_incomplete_or_overlapping_block_partition_is_rejected(self):
        features = np.zeros((8, 2, 4))
        targets = np.zeros((8, 2, 1))
        for blocks in ([slice(0, 2)], [slice(0, 3), slice(2, 4)],
                       [slice(0, 0), slice(0, 4)]):
            with self.assertRaises(ValueError):
                pair_witness_scores(features, targets, 1.0, block_slices=blocks)

    def test_duplicate_blocks_leave_pair_witness_unchanged(self):
        rng = np.random.default_rng(614)
        features = rng.dirichlet(np.ones(3), size=(20, 4))
        targets = rng.uniform(-1, 1, size=(20, 4, 2))
        single = pair_witness_scores(features, targets, 0.3)
        duplicate = np.concatenate([features, features, features], axis=-1)
        repeated = pair_witness_scores(
            duplicate, targets, 0.3,
            block_slices=[slice(0, 3), slice(3, 6), slice(6, 9)],
        )
        np.testing.assert_array_equal(single, repeated)

    def test_target_diameter_must_be_positive(self):
        features = np.zeros((8, 2, 2))
        targets = np.zeros((8, 2, 1))
        for diameter in (0.0, -2.0, float("nan")):
            with self.assertRaises(ValueError):
                certify_forecast(features, targets, targets, 1.0, diameter, 0.01, 0.01)

    def test_zero_ridge_duplicate_optima_agree_within_optimization_gaps(self):
        # With alpha=0 the feasible prediction class and objective both agree.
        rng = np.random.default_rng(210)
        f = rng.dirichlet(np.ones(4), size=150)
        y = rng.uniform(-1, 1, size=(150, 2)) + 2.0 * f[:, :2]
        single = BlockBoundedRidge(alpha=0, kappa=0.7, bounds=None,
                                  block_slices=[slice(0, 4)], tol=1e-9).fit(f, y)
        duplicated = np.concatenate([f, f, f], axis=1)
        multi = BlockBoundedRidge(alpha=0, kappa=0.7, bounds=None,
                                 block_slices=[slice(0, 4), slice(4, 8), slice(8, 12)],
                                 tol=1e-9).fit(duplicated, y)
        risk_single = np.mean(np.sum((single.predict(f) - y)**2, axis=1))
        risk_multi = np.mean(np.sum((multi.predict(duplicated) - y)**2, axis=1))
        self.assertTrue(single.converged_ and multi.converged_)
        self.assertLessEqual(abs(risk_single - risk_multi),
                             single.optimization_gap_ + multi.optimization_gap_ + 1e-12)
        # Strict JSON output must not hide a NaN or infinity in diagnostics.
        json.dumps({"single_gap": single.optimization_gap_,
                    "duplicate_gap": multi.optimization_gap_,
                    "duplicate_norm": multi.norm_sum_,
                    "risk_single": risk_single, "risk_duplicate": risk_multi},
                   allow_nan=False)


if __name__ == "__main__":
    unittest.main()
