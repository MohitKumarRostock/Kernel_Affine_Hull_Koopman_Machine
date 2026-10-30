"""Engineering checks on toy arrays only; no declared study outcome is used."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import data, experiment
from .analysis import audit_task, paired_interval
from extensions.natural_forecasting.data import SeriesRecord


class ProtocolTests(unittest.TestCase):
    def test_training_only_eligibility_and_selection(self):
        cfg = data.config()["natural"]
        spec = dict(cfg["sources"][3])
        base = np.arange(60, dtype=float)
        records = [SeriesRecord(spec["source_id"], f"toy{i}", base + i, "", "monthly") for i in range(6)]
        selected, _ = data.select_records(iter(records), spec, cfg)
        changed = [SeriesRecord(r.source_id, r.series_id, np.r_[r.values[:36], np.full(24, np.nan)], "", r.frequency) for r in records]
        selected_changed, _ = data.select_records(iter(changed), spec, cfg)
        self.assertEqual([key for key, _ in selected], [key for key, _ in selected_changed])

    def test_target_boundaries_and_prefix_refit(self):
        cfg = data.config()["natural"]
        spec = cfg["sources"][3]
        record = SeriesRecord(spec["source_id"], "toy", np.arange(80, dtype=float), "", "monthly")
        arrays, metadata = data.natural_arrays(record, spec, cfg)
        for split, start, end in [("train", 0, 48), ("validation", 48, 64), ("test", 64, 80), ("refit", 0, 64)]:
            origins = arrays[f"origins_{split}"]
            self.assertTrue(np.all(origins + 1 >= start))
            self.assertTrue(np.all(origins + 1 + spec["horizon"] <= end))
            np.testing.assert_array_equal(arrays[f"Y_{split}"][:, 0], origins + 1)
        self.assertEqual(metadata["train_end"], 48)

    def test_synthetic_nested_training_and_common_evaluation(self):
        settings = {"laws": ["harmonic"], "training_episodes": [2, 4], "validation_episodes": 2,
                    "test_episodes": 3, "history": 4, "horizons": [1, 2], "stride": 8}
        small, _ = data.synthetic_splits("harmonic", 17, 2, settings)
        large, _ = data.synthetic_splits("harmonic", 17, 4, settings)
        np.testing.assert_array_equal(small["X_train"], large["X_train"][large["groups_train"] < 2])
        for key in ("X_validation", "Y_validation", "X_test", "Y_test"):
            np.testing.assert_array_equal(small[key], large[key])

    def test_policy_threshold_and_zero_loss(self):
        self.assertEqual(experiment.utility_choice(1.0, .95, .05), "add_kahm")
        self.assertEqual(experiment.utility_choice(1.0, .951, .05), "raw_rbf")
        self.assertEqual(experiment.utility_choice(0.0, 0.0, .05), "raw_rbf")
        losses = {"raw_rbf": 1.0, "add_kahm": .96, "add_affine": .94, "add_locality": .94}
        self.assertEqual(experiment.geometry_choice(losses, .05), "add_affine")
        self.assertEqual(experiment.geometry_choice({name: 1.0 for name in losses}, .05), "raw_rbf")

    def test_seasonal_naive_is_causal_and_uses_full_history(self):
        values = np.arange(30, dtype=float)
        values[11] = np.nan
        arrays = {"values": values, "origins_test": np.array([13, 17]),
                  "X_test": np.array([[12., 13.], [16., 17.]]), "Y_train": np.zeros((1, 2))}
        task = {"phase": "natural", "seasonal_period": 7}
        prediction = experiment.naive_prediction(arrays, "test", task)
        np.testing.assert_array_equal(prediction, [[7., 8.], [17., 12.]])
        altered = dict(arrays, values=np.r_[values[:18], np.full(12, -999.)])
        np.testing.assert_array_equal(experiment.naive_prediction(altered, "test", task), prediction)
        task["seasonal_period"] = 52
        np.testing.assert_array_equal(experiment.naive_prediction(arrays, "test", task), [[13., 13.], [17., 17.]])

    def test_cost_union_charges_rejected_additive_search(self):
        ids = {"data_preparation", "utility_kahm_5pct:decision"}
        ids |= experiment.search_dependencies("raw_rbf") | experiment.search_dependencies("add_kahm")
        ids |= experiment.final_dependencies("raw_rbf", "natural")
        self.assertIn("add_kahm:search", ids)
        self.assertNotIn("add_kahm:refit", ids)
        nodes = [{"id": name, "wall_seconds": 1.0} for name in ids]
        self.assertEqual(experiment.sum_nodes(ids, nodes), len(ids))

    def test_paired_interval_is_simultaneous_and_keeps_zero(self):
        differences = np.linspace(-.1, .1, 12)
        result = paired_interval(differences)
        self.assertEqual(result["n"], 12)
        self.assertEqual(result["family_size"], 24)
        self.assertLess(result["lower"], 0)
        self.assertGreater(result["upper"], 0)

    def test_end_to_end_toy_runner_preserves_all_methods(self):
        rng = np.random.default_rng(421)
        cfg = data.config()
        cfg["geometry"] = {**cfg["geometry"], "cluster_counts": [2], "max_refs": 8, "max_fit": 30}
        cfg["hist_gradient"] = {**cfg["hist_gradient"], "max_iter": 3}
        spec = {"context": 4, "horizon": 2, "seasonal_period": 7}
        settings = {**cfg["natural"], "max_train_windows": 30, "max_eval_windows": 12}
        series = SeriesRecord("engineering", "white_noise", rng.normal(size=80), "", "daily")
        arrays, metadata = data.natural_arrays(series, spec, settings)
        task = {"task_id": "engineering-white-noise", "phase": "natural", "seed": 421,
                "landmarks": 4, **spec, **metadata}
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            destination = Path(temporary) / "toy"
            result = experiment.run_task(task, arrays, metadata, cfg, destination)
            if result["complete"]:
                rows, trials, episodes, horizons = audit_task(result, destination, cfg)
                self.assertEqual(len(rows), len(cfg["methods"]) + len(experiment.POLICIES))
                self.assertEqual(len(horizons), len(rows) * 2)
                self.assertEqual(episodes, [])
        self.assertTrue(result["complete"], result.get("errors"))
        self.assertEqual(len(result["methods"]), len(cfg["methods"]))
        self.assertEqual(set(result["policies"]), set(experiment.POLICIES))


if __name__ == "__main__":
    unittest.main()
