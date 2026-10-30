"""Tests of source separation, chronological information sets and selector costs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from extensions.natural_forecasting import analysis as a


def task(family, *, phase="development", seed=17, series="s0"):
    key = f"{phase}-{family}-{series}-{seed}"
    nodes = [
        {"id": "metadata", "seconds": 1.0, "process_cpu_seconds": 0.5, "peak_rss_bytes": 100},
        {"id": "geometry", "seconds": 7.0, "process_cpu_seconds": 3.5, "peak_rss_bytes": 300},
        {"id": "probe", "seconds": 4.0, "process_cpu_seconds": 2.0, "peak_rss_bytes": 200},
    ]
    candidates = []
    family_index = int(family[-1])
    for index, name in enumerate(a.DEFAULT_CANDIDATES):
        search, final = name + "-search", name + "-final"
        search_seconds = 3.0 if name == "raw_rbf" else 9.0 if name == "raw_kahm_rbf" else 2.0
        final_seconds = 5.0 if name == "raw_rbf" else 10.0 if name == "raw_kahm_rbf" else 1.0
        nodes += [{"id": search, "seconds": search_seconds, "process_cpu_seconds": search_seconds/2, "peak_rss_bytes": 400},
                  {"id": final, "seconds": final_seconds, "process_cpu_seconds": final_seconds/2, "peak_rss_bytes": 500}]
        loss = 1.0 + 0.2*index
        if name == "raw_rbf":
            loss = 2.0 + 0.1*family_index
        elif name == "raw_kahm_rbf":
            loss = (2.0 + 0.1*family_index) * (0.75 + 0.15*family_index)
        candidates.append({
            "candidate_id": name, "validation_loss": 1.0 if name == "raw_rbf" else 2.0 + index,
            "test_loss": loss,
            "selection_cost_node_ids": [search], "final_cost_node_ids": [final],
            "cost_node_ids": [search, final],
            "tuning_fit_max_target_index": 59, "selection_max_target_index": 79,
            "refit_max_target_index": 79, "test_min_target_index": 80, "test_max_target_index": 99,
        })
    return {
        "task_id": key, "phase": phase, "source_family": family, "dataset_id": family + "-data",
        "series_id": series, "horizon": 3, "history_length": 12, "seed": seed,
        "budget_id": "small", "loss_scale": 2.0,
        "split": {"train_end_exclusive": 60, "validation_start": 60, "validation_end_exclusive": 80,
                  "test_start": 80, "test_end_exclusive": 100},
        "features": {"n_train_windows": 40 + family_index, "lag1_autocorrelation": 0.1*family_index,
                     "kahm_gate_entropy": 0.2 + 0.1*family_index, "affine_projection_residual": 0.05 + 0.05*family_index},
        "feature_provenance": {"metadata_max_observation_index": 59, "geometry_max_observation_index": 59},
        "feature_cost_node_ids": {"metadata": ["metadata"], "geometry": ["geometry"]},
        "cost_nodes": nodes, "candidates": candidates,
        "probe": {"raw_validation_loss": 1.0, "kahm_validation_loss": 1.5,
                  "cost_node_ids": ["probe"], "tuning_fit_max_target_index": 59, "selection_max_target_index": 79},
    }


def development():
    return [task(f"dev{i}", seed=seed) for i in range(4) for seed in (17, 43)]


def confirmation():
    return [task(f"confirm{i}", phase="confirmation") for i in range(4)]


def config(dev=None, confirm=None):
    dev, confirm = dev or development(), confirm or confirmation()
    return {"reserved_source_families": sorted({row["source_family"] for row in confirm}),
            "expected_development_task_ids": [row["task_id"] for row in dev],
            "expected_confirmation_task_ids": [row["task_id"] for row in confirm]}


class AnalysisTests(unittest.TestCase):
    def test_hierarchical_weights_do_not_count_extra_seeds_as_source_replications(self):
        rows = [task("dev0", seed=i) for i in range(8)] + [task("dev1")]
        weights = a.hierarchical_weights(rows)
        self.assertAlmostEqual(float(weights[:8].sum()), 0.5)
        self.assertAlmostEqual(float(weights[8]), 0.5)
        # Adding a second series splits its source's mass, not the other source's.
        rows.append(task("dev0", series="s1"))
        weights = a.hierarchical_weights(rows)
        self.assertAlmostEqual(float(weights[:8].sum()), 0.25)
        self.assertAlmostEqual(float(weights[-1]), 0.25)
        self.assertAlmostEqual(float(weights[8]), 0.5)

    def test_leave_source_out_fits_standardization_only_on_other_families(self):
        rows = development()
        for row in rows:
            if row["source_family"] == "dev3":
                row["features"]["n_train_windows"] = 1_000_000
        report = a.analyze_development(rows, config(rows))
        fold = next(row for row in report["folds"] if row["held_out_source_family"] == "dev3")
        self.assertNotIn("dev3", fold["training_source_families"])
        model = fold["models"]["metadata"]
        training = [row for row in rows if row["source_family"] != "dev3"]
        expected = sum(a.hierarchical_weights(training) * np.log1p([row["features"]["n_train_windows"] for row in training]))
        self.assertAlmostEqual(model["mean"][0], expected)
        self.assertTrue(all("dev3" not in key for key in model["training_task_ids"]))
        self.assertEqual(report["policies"]["metadata"]["source_families"], 4)

    def test_confirmation_requires_explicit_gate_and_complete_reserved_roster(self):
        dev, held = development(), confirmation()
        frozen = a.freeze_diagnostic(dev, config=config(dev, held))
        with self.assertRaises(ValueError):
            a.evaluate_frozen(held, frozen)
        with self.assertRaises(ValueError):
            a.analyze_development(held)
        with self.assertRaises(ValueError):
            a.fit_diagnostic(held, a.METADATA_FEATURES)
        with self.assertRaises(ValueError):
            a.evaluate_frozen(held[:-1], frozen, allow_confirmation=True)
        report = a.evaluate_frozen(held, frozen, allow_confirmation=True)
        self.assertEqual(report["policies"]["metadata_probe_geometry"]["source_families"], 4)
        changed = deepcopy(held)
        changed[0]["source_family"] = "dev0"
        with self.assertRaises(ValueError):
            a.evaluate_frozen(changed, frozen, allow_confirmation=True)

    def test_future_losses_do_not_change_any_frozen_prediction_or_decision(self):
        dev, held = development(), confirmation()
        frozen = a.freeze_diagnostic(dev, config=config(dev, held))
        first = a.evaluate_frozen(held, frozen, allow_confirmation=True)
        changed = deepcopy(held)
        for row in changed:
            for candidate in row["candidates"]:
                candidate["test_loss"] = 1000.0/(1 + len(candidate["candidate_id"]))
        second = a.evaluate_frozen(changed, frozen, allow_confirmation=True)
        for left, right in zip(first["task_policy_rows"], second["task_policy_rows"]):
            self.assertEqual(left["policy"], right["policy"])
            self.assertEqual(left["chosen_candidate"], right["chosen_candidate"])
            self.assertEqual(left["predicted_pair_log_benefit"], right["predicted_pair_log_benefit"])
            self.assertEqual(left["required_cost_node_ids"], right["required_cost_node_ids"])

    def test_validation_selector_charges_searches_but_only_selected_refit(self):
        row = task("dev0")
        result = {item["policy"]: item for item in a.policy_rows(row, {})}
        selected = result["validation_pair"]
        self.assertEqual(selected["chosen_candidate"], "raw_rbf")
        self.assertEqual(selected["measured_node_wall_seconds"], 3.0 + 9.0 + 5.0)
        self.assertNotIn("raw_kahm_rbf-final", selected["required_cost_node_ids"])
        self.assertEqual(result["always_raw"]["measured_node_wall_seconds"], 8.0)
        self.assertEqual(result["cheap_probe"]["measured_node_wall_seconds"], 8.0 + 4.0)
        self.assertEqual(result["cheap_probe"]["maximum_measured_node_peak_rss_bytes"], 500.0)

    def test_rejected_kahm_still_pays_for_geometry_and_probe(self):
        row = task("dev0")
        features = a.MODEL_FEATURES["metadata_probe_geometry"]
        model = {"features": list(features), "mean": [0.0]*len(features),
                 "scale": [1.0]*len(features), "coefficients": [0.0]*len(features), "intercept": -1.0}
        result = next(item for item in a.policy_rows(row, {"metadata_probe_geometry": model})
                      if item["policy"] == "metadata_probe_geometry")
        self.assertEqual(result["chosen_candidate"], "raw_rbf")
        self.assertEqual(result["measured_node_wall_seconds"], 8.0 + 1.0 + 7.0 + 4.0)
        # Reusing a genuinely identical measured node cannot charge it twice.
        row["feature_cost_node_ids"]["metadata"] = ["raw_rbf-search"]
        result = next(item for item in a.policy_rows(row, {"metadata_probe_geometry": model})
                      if item["policy"] == "metadata_probe_geometry")
        self.assertEqual(result["measured_node_wall_seconds"], 8.0 + 7.0 + 4.0)

    def test_one_off_fit_and_per_task_inference_costs_remain_distinct(self):
        dev, held = development(), confirmation()
        frozen = a.freeze_diagnostic(dev, config=config(dev, held))
        self.assertEqual(set(frozen["offline_fit_by_model"]), set(a.MODEL_FEATURES))
        for value in frozen["offline_fit_by_model"].values():
            self.assertGreaterEqual(value["wall_seconds"], 0)
            self.assertGreaterEqual(value["cpu_seconds"], 0)
        report = a.evaluate_frozen(held, frozen, allow_confirmation=True)
        self.assertEqual(report["offline_diagnostic_fit_by_model"], frozen["offline_fit_by_model"])
        for row in report["task_policy_rows"]:
            if row["policy"] in a.MODEL_FEATURES:
                self.assertGreaterEqual(row["decision_wall_seconds"], 0)
                self.assertAlmostEqual(row["total_policy_wall_seconds"],
                    row["measured_node_wall_seconds"] + row["decision_wall_seconds"])
                self.assertAlmostEqual(row["total_policy_cpu_seconds"],
                    row["measured_node_cpu_seconds"] + row["decision_cpu_seconds"])
        for candidate_id in a.DEFAULT_CANDIDATES:
            self.assertIn("standalone_" + candidate_id, report["policies"])

    def test_metadata_diagnostic_does_not_access_geometry_or_probe(self):
        row = task("dev0")
        del row["probe"]
        del row["features"]["kahm_gate_entropy"]
        del row["features"]["affine_projection_residual"]
        model = {"features": list(a.METADATA_FEATURES), "mean": [0.0]*3,
                 "scale": [1.0]*3, "coefficients": [0.0]*3, "intercept": -1.0}
        self.assertEqual(a.predict_diagnostic(row, model), -1.0)

    def test_chronological_and_cost_integrity_violations_are_rejected(self):
        mutations = (
            lambda row: row["candidates"][0].update(refit_max_target_index=80),
            lambda row: row["candidates"][0].update(tuning_fit_max_target_index=60),
            lambda row: row["candidates"][0].update(selection_max_target_index=80),
            lambda row: row["feature_provenance"].update(geometry_max_observation_index=60),
            lambda row: row["probe"].update(selection_max_target_index=80),
            lambda row: row["candidates"][0].update(final_cost_node_ids=["not-a-recorded-node"]),
            lambda row: row["cost_nodes"][0].update(seconds=-1),
            lambda row: row["feature_cost_node_ids"].update(geometry=[]),
            lambda row: row["probe"].update(cost_node_ids=[]),
            lambda row: row["cost_nodes"][0].update(cpu_seconds=row["cost_nodes"][0].pop("process_cpu_seconds")),
        )
        for mutate in mutations:
            row = task("dev0")
            mutate(row)
            with self.assertRaises(ValueError):
                a.validate_tasks([row])

    def test_frozen_model_cannot_be_modified_or_overwritten(self):
        rows = development()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostic_freeze.json"
            frozen = a.freeze_diagnostic(rows, path, config(rows))
            self.assertEqual(a.verify_frozen(path)["freeze_sha256"], frozen["freeze_sha256"])
            with self.assertRaises(FileExistsError):
                a.freeze_diagnostic(rows, path, config(rows))
            frozen["models"]["metadata"]["intercept"] += 1
            with self.assertRaises(ValueError):
                a.verify_frozen(frozen)

    def test_independent_prediction_audit_checks_horizon_and_origin_means(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        predictions = np.array([[0.0, 3.0], [1.0, 4.0]])
        result = a.audit_prediction_arrays(y, predictions, 1.5, reported_origin_losses=np.array([1.0, 2.0]))
        self.assertEqual(result["origins"], 2)
        self.assertEqual(result["horizons"], 2)
        with self.assertRaises(ValueError):
            a.audit_prediction_arrays(y, predictions, 1.0)
        with self.assertRaises(ValueError):
            a.audit_prediction_arrays(y, predictions, 1.5, reported_origin_losses=np.array([1.5, 1.5]))


if __name__ == "__main__":
    unittest.main()
