"""Checks on independent artificial draws; not study confirmation samples."""
import json
import pickle
import unittest

import numpy as np

from datasets import make_branch_split, branch_targets, make_forecast_split
from geometry import GeometryBank
from mechanism_diagnostics import geometry_diagnostics


class DiagnosticTests(unittest.TestCase):
    def test_branch_reflection_and_mismatch_identities(self):
        train = make_branch_split(160, 773011, curvature=1)
        test = make_branch_split(70, 773019, curvature=1)
        bank = GeometryBank(seed=773023, cluster_counts=(2,), max_refs=16).fit(train["X"])
        bank.transform(test["X"][:9], diagnostics=True)
        before = pickle.dumps(bank.state_dict(), protocol=5)
        prior_diagnostics = pickle.dumps(bank.last_diagnostics_, protocol=5)
        result = geometry_diagnostics(bank, train, test, branch=True, max_nn_train=120, max_nn_test=50)
        json.dumps(result, allow_nan=False)
        self.assertEqual(before, pickle.dumps(bank.state_dict(), protocol=5))
        self.assertEqual(prior_diagnostics, pickle.dumps(bank.last_diagnostics_, protocol=5))
        self.assertTrue(result["frozen_model_state_unchanged"])
        for stage in result["score_recovery_from_phi_plus_logz"].values():
            for value in stage.values():
                self.assertTrue(value["within_5e_12"])
        reflection = result["branch_reflection"]
        self.assertAlmostEqual(reflection["empirical_squared_target_contrast_per_unit_lambda"], float(np.mean(12 * test["T"]**2)))
        expected_rms = np.sqrt(2 * np.mean(test["T"]**2) / np.var(train["T"]))
        self.assertAlmostEqual(reflection["metrics"]["raw"]["training_standardized_coordinate_rms"], expected_rms)
        self.assertEqual(result["nearest_neighbors"]["raw"]["selected_neighbor_raw_distance_rank_percentile"]["maximum"], 0.0)
        query = np.asarray(result["nearest_neighbor_design"]["query_held_out_row_indices"])
        lam = .37
        _, train_mean = branch_targets(train, lam)
        _, test_mean = branch_targets(test, lam)
        for value in result["nearest_neighbors"].values():
            selected = np.asarray(value["selected_training_row_indices"])
            moments = value["branch_noiseless_target_mismatch"]
            predicted = ((1-lam) * moments["A_branch_component"] + lam * moments["B_longitudinal_component"]
                         + np.sqrt(lam*(1-lam)) * moments["C_cross_component"])
            empirical = float(np.mean((test_mean[query] - train_mean[selected])**2))
            self.assertAlmostEqual(predicted, empirical)

    def test_group_balanced_forecast_target_mismatch(self):
        train = make_forecast_split("nonlinear_ar", 8, 773101)
        test = make_forecast_split("nonlinear_ar", 4, 773103)
        bank = GeometryBank(seed=773107, cluster_counts=(2,), max_refs=16).fit(train["X"])
        result = geometry_diagnostics(bank, train, test, max_nn_train=80, max_nn_test=19)
        json.dumps(result, allow_nan=False)
        self.assertNotIn("branch_reflection", result)
        query = np.asarray(result["nearest_neighbor_design"]["query_held_out_row_indices"])
        for value in result["nearest_neighbors"].values():
            selected = np.asarray(value["selected_training_row_indices"])
            loss = (test["Y"][query] - train["Y"][selected])**2
            groups = test["groups"][query]
            expected = np.stack([loss[groups == g].mean(axis=0) for g in np.unique(groups)]).mean(axis=0)
            np.testing.assert_allclose(value["observed_target_mismatch"]["group_balanced_mse_by_target"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
