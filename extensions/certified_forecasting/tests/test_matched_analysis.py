"""Checks against selection leakage, unmatched grids and incorrect aggregation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/analyze_matched_geometry.py"
spec = importlib.util.spec_from_file_location("matched_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def candidate(name, mode="standard", features=2, history=4, validation=(0.2, 0.4)):
    return {"name": name, "mode": mode, "n_features": features,
            "history": history, "validation_mse": list(validation)}


class MatchedAnalysisTests(unittest.TestCase):
    def test_selection_ignores_test_and_calibration_and_obeys_all_ties(self):
        a = candidate("z", mode="standard")
        b = candidate("a", mode="minmax")
        a.update(test_mse=[4., 4.], calibration_mse=[4., 4.])
        b.update(test_mse=[0., 0.], calibration_mse=[0., 0.])
        self.assertIs(analysis.select_candidate([b, a]), a)
        a["test_mse"], b["test_mse"] = b["test_mse"], a["test_mse"]
        self.assertIs(analysis.select_candidate([b, a]), a)
        b["validation_mse"] = [0.1, 0.1]
        self.assertIs(analysis.select_candidate([a, b]), b)
        b["validation_mse"] = a["validation_mse"]
        a["n_features"] = 6
        self.assertIs(analysis.select_candidate([a, b]), b)
        a["n_features"], a["history"] = 2, 8
        self.assertIs(analysis.select_candidate([a, b]), b)
        b["history"], a["mode"] = 8, "minmax"
        self.assertIs(analysis.select_candidate([a, b]), b)

    def test_relative_effect_is_ratio_of_means_and_keeps_regressions(self):
        result = analysis.paired_seed_summary([1., 3.], [2., 2.])
        self.assertEqual(result["relative_mse_change_percent"], 0.)
        self.assertEqual(result["n_seeds_improved"], 1)
        self.assertEqual(result["n_seeds_degraded"], 1)
        self.assertEqual(result["seed_effects"], [-1., 1.])
        result = analysis.paired_seed_summary([1., 4.], [1., 2.])
        self.assertAlmostEqual(result["relative_mse_change_percent"], 100. * (5. / 3. - 1.))
        self.assertEqual(result["effect_sign"], "degraded")

    def test_pen_pooling_averages_coordinate_mse_before_square_root(self):
        selections = []
        for coordinate, test in ((0, [0., 1.]), (1, [1., 0.])):
            c = candidate("m") | {"task": "pendigits", "seed": 5701,
                                  "coordinate": coordinate, "horizons": [1, 2], "test_mse": test}
            selections.append({"analysis": "primary", "head": "nested", "dimension": "selected",
                               "geometry": "kahm", "candidate": c})
        pooled, _ = analysis.pool_selections(selections)
        self.assertEqual(len(pooled), 3)
        self.assertTrue(all(row["test_mse"] == 0.5 for row in pooled))
        self.assertTrue(all(row["test_rmse"] == np.sqrt(0.5) for row in pooled))
        with self.assertRaises(ValueError):
            analysis.pool_selections(selections[:1])

    def test_grid_rejects_missing_head_cell_and_unequal_preprocessing(self):
        config = {"cluster_counts": [2, 4], "histories": [4]}
        cs = []
        for family in analysis.FAMILIES:
            for dimension in (2, 6):
                cs.append(candidate(f"{family}_{dimension}", features=dimension) |
                          {"family": family, "input_scaling": {"mode": "standard", "mean": [0.]}})
        for i in range(7):
            cs.append(candidate(f"control{i}") | {"family": f"control{i}"})
        self.assertEqual(analysis.validate_candidate_grid(cs, config, "standard"), [2, 6])
        with self.assertRaises(ValueError):
            analysis.validate_candidate_grid(cs[1:], config, "standard")
        changed = copy.deepcopy(cs)
        changed[0]["input_scaling"]["mean"] = [1.]
        with self.assertRaises(ValueError):
            analysis.validate_candidate_grid(changed, config, "standard")

    def test_paired_calibration_is_horizon_average_at_reserved_one_sided_delta(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "scores.npz"
            arrays = {"kahm__cal_loss": np.tile([0.1, 0.3], (4096, 1)),
                      "rbf__cal_loss": np.tile([1.0, 1.2], (4096, 1)),
                      "affine__cal_loss": np.tile([0.05, 0.15], (4096, 1))}
            np.savez(archive, **arrays)
            pooled, members = [], {}
            for geometry, mse in (("kahm", 0.2), ("rbf", 1.1), ("affine", 0.1)):
                c = candidate(geometry) | {"task": "harmonic", "seed": 5701, "coordinate": -1,
                                           "uid": ("harmonic", 5701, -1, "standard", geometry)}
                key = ("harmonic", 5701, "primary", "nested", "selected", geometry)
                members[key] = [c]
                pooled.append(dict(zip(("task", "seed", "analysis", "head", "dimension", "geometry"), key)) |
                              {"horizon": "all", "test_mse": mse, "selected_models": []})
            _, summaries, certificates, _ = analysis.compare(
                pooled, members, {("harmonic", 5701, "standard"): archive}, {})
            self.assertEqual(len(certificates), 2)
            by_alternative = {r["alternative"]: r for r in certificates}
            self.assertTrue(by_alternative["rbf"]["certified_kahm_improvement"])
            self.assertFalse(by_alternative["affine"]["certified_kahm_improvement"])
            self.assertEqual(by_alternative["rbf"]["n_independent"], 4096)
            self.assertEqual(by_alternative["rbf"]["bound"], 8.)
            self.assertAlmostEqual(by_alternative["rbf"]["delta_per_tail"], 0.01 / 6)
            self.assertAlmostEqual(by_alternative["rbf"]["empirical_improvement"], 0.9)
            self.assertEqual(len(summaries), 2)


if __name__ == "__main__":
    unittest.main()
