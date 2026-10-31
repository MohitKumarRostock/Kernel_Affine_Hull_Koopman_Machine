"""Protocol checks on fabricated data only; no study outcomes are consulted.

The tests target consequential risks: held-out-label leakage, unequal correction
rows, metric fitting on evaluation data, incomplete failure accounting, and
incorrect component-cost attribution. They do not claim statistical coverage.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from extensions.target_geometry import experiment


def toy_config():
    cfg = copy.deepcopy(experiment.config())
    cfg["base_alphas"] = [0.01, 1.0]
    cfg["correction_alphas"] = [0.01, 1.0]
    cfg["gammas"] = [0.0, 0.5, 1.0]
    cfg["geometry"] = {"cluster_counts": [2], "rank": 1, "max_refs": 8,
                       "max_fit": 64, "omega": 16.0, "tau": 1e-6}
    cfg["metric"] = {"max_rows": 24, "max_iter": 2,
                     "regularization": 0.05, "max_weight_ratio": 4.0}
    cfg["rbf_landmarks"] = 4
    cfg["synthetic"]["outer_folds"] = 3
    cfg["natural"].update(initial_fraction=0.4, forward_folds=3)
    return cfg


def toy_arrays(seed=891):
    rng = np.random.default_rng(seed)
    arrays = {}
    for split, groups in (("train", 9), ("validation", 2), ("test", 2)):
        x = rng.normal(size=(groups * 6, 4))
        y = np.column_stack((1.2 * x[:, 0] + 0.4 * x[:, 1] ** 2,
                             -0.7 * x[:, 2] + 0.3 * x[:, 0] * x[:, 3]))
        y += rng.normal(scale=0.03, size=y.shape)
        arrays[f"X_{split}"], arrays[f"Y_{split}"] = x, y
        arrays[f"groups_{split}"] = np.repeat(np.arange(groups), 6)
        arrays[f"origins_{split}"] = np.tile(np.arange(4, 10), groups)
    return arrays


def toy_task(phase="synthetic"):
    return {"task_id": f"fabricated__{phase}", "phase": phase,
            "law": "fabricated", "source_id": "fabricated", "seed": 891,
            "loss_scale": 1.0, "context": 4, "horizon": 2}


class TestDonorIsolation(unittest.TestCase):
    def test_synthetic_donors_and_inner_tuning_hold_out_whole_episodes(self):
        arrays, cfg, task = toy_arrays(), toy_config(), toy_task()
        with threadpool_limits(limits=1):
            indices, prediction, records = experiment.oof_predictions(arrays, task, cfg)
        assert_array_equal(indices, np.arange(len(arrays["X_train"])))
        self.assertTrue(np.isfinite(prediction).all())
        for record in records:
            self.assertEqual(record["status"], "ok")
            g = arrays["groups_train"]
            query = set(g[record["query_indices"]])
            donor = set(g[record["donor_indices"]])
            fit = set(g[record["inner_fit_indices"]])
            valid = set(g[record["inner_validation_indices"]])
            self.assertFalse(query & donor)
            self.assertFalse(fit & valid)
            self.assertTrue(fit | valid <= donor)

        # Corrupt one entire held-out fold's labels. Its own donor prediction
        # and selected donor alpha must remain exactly unchanged.
        changed = {key: value.copy() for key, value in arrays.items()}
        first_query = np.asarray(records[0]["query_indices"])
        changed["Y_train"][first_query] += 10000.0
        with threadpool_limits(limits=1):
            after_indices, after_prediction, after_records = experiment.oof_predictions(changed, task, cfg)
        assert_array_equal(indices, after_indices)
        assert_allclose(prediction[first_query], after_prediction[first_query], atol=0, rtol=0)
        self.assertEqual(records[0]["alpha"], after_records[0]["alpha"])
        self.assertEqual(records[0]["trials"], after_records[0]["trials"])

    def test_natural_outer_and_inner_donors_observe_complete_targets(self):
        arrays, cfg, task = toy_arrays(), toy_config(), toy_task("natural")
        n = len(arrays["X_train"])
        arrays["origins_train"] = np.arange(20, 20 + n)
        task["horizon"] = 4
        with threadpool_limits(limits=1):
            indices, prediction, records = experiment.oof_predictions(arrays, task, cfg)
        self.assertGreater(len(indices), 8)
        for record in records:
            self.assertEqual(record["status"], "ok")
            origins = arrays["origins_train"]
            queries = np.asarray(record["query_indices"])
            donors = np.asarray(record["donor_indices"])
            self.assertLessEqual(int(np.max(origins[donors] + task["horizon"])),
                                 int(np.min(origins[queries])))
            fits = np.asarray(record["inner_fit_indices"])
            valid = np.asarray(record["inner_validation_indices"])
            self.assertLessEqual(int(np.max(origins[fits] + task["horizon"])),
                                 int(np.min(origins[valid])))
            self.assertTrue(set(fits) <= set(donors))
            self.assertTrue(set(valid) <= set(donors))

        # Changing all labels at and after the first query block cannot affect
        # that block's donor or its internal tuning.
        changed = {key: value.copy() for key, value in arrays.items()}
        first_query = np.asarray(records[0]["query_indices"])
        changed["Y_train"][first_query[0]:] -= 10000.0
        with threadpool_limits(limits=1):
            after_indices, after_prediction, after_records = experiment.oof_predictions(changed, task, cfg)
        assert_array_equal(indices, after_indices)
        selected = np.isin(indices, first_query)
        assert_allclose(prediction[selected], after_prediction[selected], atol=0, rtol=0)
        self.assertEqual(records[0]["alpha"], after_records[0]["alpha"])
        self.assertEqual(records[0]["trials"], after_records[0]["trials"])


class TestCompleteProcedure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="kahm-protocol-toy-")
        cls.arrays, cls.cfg, cls.task = toy_arrays(), toy_config(), toy_task()
        cls.metric_calls, cls.head_calls = [], []
        original_metric_fit = experiment.LearnedDiagonalMetric.fit
        original_choose_head = experiment.choose_head

        def metric_spy(self, x, residuals, **kwargs):
            cls.metric_calls.append({"x": x.copy(), "residuals": residuals.copy(),
                                     "scaling_data": kwargs["scaling_data"].copy(),
                                     "groups": kwargs["groups"].copy()})
            return original_metric_fit(self, x, residuals, **kwargs)

        def head_spy(features, residuals, *args, **kwargs):
            cls.head_calls.append({"rows": len(features), "residuals": residuals.copy()})
            return original_choose_head(features, residuals, *args, **kwargs)

        first = Path(cls.directory.name) / "original"
        with threadpool_limits(limits=1), \
                patch.object(experiment.LearnedDiagonalMetric, "fit", metric_spy), \
                patch.object(experiment, "choose_head", head_spy):
            cls.report = experiment.run_task(cls.task, cls.arrays, cls.cfg, first)
        with np.load(first / "predictions.npz", allow_pickle=False) as saved:
            cls.saved = {key: saved[key].copy() for key in saved.files}

        changed = {key: value.copy() for key, value in cls.arrays.items()}
        changed["Y_test"] = changed["Y_test"] * -1000.0 + 4321.0
        second = Path(cls.directory.name) / "changed_test_labels"
        with threadpool_limits(limits=1):
            cls.changed_report = experiment.run_task(cls.task, changed, cls.cfg, second)
        with np.load(second / "predictions.npz", allow_pickle=False) as saved:
            cls.changed_saved = {key: saved[key].copy() for key in saved.files}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_all_arms_use_identical_correction_indices_and_intended_targets(self):
        indices = self.saved["correction_indices"]
        expected_is = self.arrays["Y_train"][indices] - self.saved["final_base_train_predictions"]
        expected_oof = self.arrays["Y_train"][indices] - self.saved["donor_predictions"]
        self.assertEqual(len(self.head_calls), 3 * len(self.cfg["gates"]))
        for index, call in enumerate(self.head_calls):
            self.assertEqual(call["rows"], len(indices))
            expected = expected_is if index % 3 == 0 else expected_oof
            assert_allclose(call["residuals"], expected, atol=0, rtol=0)

    def test_metric_receives_only_training_inputs_and_oof_targets(self):
        self.assertEqual(len(self.metric_calls), 1)
        call = self.metric_calls[0]
        indices = self.saved["correction_indices"]
        assert_array_equal(call["x"], self.arrays["X_train"][indices])
        assert_array_equal(call["scaling_data"], self.arrays["X_train"])
        assert_array_equal(call["groups"], self.arrays["groups_train"][indices])
        assert_allclose(call["residuals"], self.arrays["Y_train"][indices] - self.saved["donor_predictions"],
                        atol=0, rtol=0)

    def test_test_labels_cannot_change_fits_choices_or_predictions(self):
        self.assertEqual(self.report["metric"], self.changed_report["metric"])
        self.assertEqual(self.report["oof_folds"], self.changed_report["oof_folds"])
        for name in experiment.methods(self.cfg):
            assert_allclose(self.saved[f"prediction__{name}"],
                            self.changed_saved[f"prediction__{name}"], atol=0, rtol=0)
            self.assertEqual(self.report["methods"][name]["selected"],
                             self.changed_report["methods"][name]["selected"])

    def test_roster_and_component_costs_are_complete_and_nonduplicated(self):
        self.assertEqual(set(self.report["methods"]), set(experiment.methods(self.cfg)))
        nodes = self.report["nodes"]
        for name, record in self.report["methods"].items():
            deps = record["dependencies"]
            self.assertEqual(len(deps), len(set(deps)))
            self.assertTrue(set(deps) <= set(nodes))
            self.assertAlmostEqual(record["pipeline_wall_seconds"],
                                   sum(nodes[key]["wall_seconds"] for key in deps), places=12)
            self.assertAlmostEqual(record["pipeline_cpu_seconds"],
                                   sum(nodes[key]["cpu_seconds"] for key in deps), places=12)
            if name.startswith("learned_oof"):
                self.assertIn("metric:fit", deps)
                self.assertIn("oof:fit", deps)
            elif name.startswith("fixed_oof"):
                self.assertIn("oof:fit", deps)
                self.assertNotIn("metric:fit", deps)
            elif name.startswith("fixed_is"):
                self.assertNotIn("metric:fit", deps)
                self.assertNotIn("oof:fit", deps)

    def test_saved_corrections_satisfy_observed_loss_identity(self):
        base = self.saved["prediction__ridge"]
        target = self.saved["target"]
        for name, record in self.report["methods"].items():
            if "residual_alignment" not in record:
                continue
            delta = self.saved[f"prediction__{name}"] - base
            gain = float(np.mean((target - base) ** 2) - np.mean((target - base - delta) ** 2))
            self.assertAlmostEqual(gain, 2 * record["residual_alignment"] - record["correction_energy"],
                                   places=11)


class TestFailureAndDeclaredFamily(unittest.TestCase):
    def test_joint_head_failure_is_isolated_and_failed_search_is_charged(self):
        arrays, cfg, task = toy_arrays(), toy_config(), toy_task()
        with tempfile.TemporaryDirectory(prefix="kahm-joint-failure-") as temporary:
            destination = Path(temporary) / "run"
            with threadpool_limits(limits=1), \
                    patch.object(experiment, "choose_joint",
                                 side_effect=np.linalg.LinAlgError("deliberate toy joint-head failure")):
                result = experiment.run_task(task, arrays, cfg, destination)
            self.assertEqual(set(result["methods"]), set(experiment.methods(cfg)))
            joint = result["methods"]["joint_kahm_ridge"]
            self.assertEqual(joint["status"], "fallback")
            self.assertIn("joint_kahm_ridge:search", joint["dependencies"])
            self.assertIn("ridge:search", joint["dependencies"])
            with np.load(destination / "predictions.npz", allow_pickle=False) as saved:
                assert_array_equal(saved["prediction__joint_kahm_ridge"], saved["prediction__ridge"])
            for name in ("ridge", "raw_rbf", "naive"):
                self.assertEqual(result["methods"][name]["status"], "ok")

    def test_internal_identity_metric_fallback_is_visible_on_learned_arms(self):
        arrays, cfg, task = toy_arrays(), toy_config(), toy_task()
        original_fit = experiment.LearnedDiagonalMetric.fit

        def force_uninformative_target(self, x, residuals, **kwargs):
            return original_fit(self, x, np.zeros_like(residuals), **kwargs)

        with tempfile.TemporaryDirectory(prefix="kahm-metric-fallback-") as temporary:
            with threadpool_limits(limits=1), \
                    patch.object(experiment.LearnedDiagonalMetric, "fit", force_uninformative_target):
                result = experiment.run_task(task, arrays, cfg, Path(temporary) / "run")
        self.assertIsNotNone(result["metric"]["fallback_reason"])
        for name, record in result["methods"].items():
            if name.startswith("learned_oof"):
                self.assertTrue(record["fallback_metric_identity"])

    def test_geometry_failure_keeps_all_forecasts_and_charges_failed_fit(self):
        arrays, cfg, task = toy_arrays(), toy_config(), toy_task()
        with tempfile.TemporaryDirectory(prefix="kahm-bank-failure-") as temporary:
            destination = Path(temporary) / "run"
            with threadpool_limits(limits=1), \
                    patch.object(experiment.RobustLocalFeatureBank, "fit",
                                 side_effect=ValueError("deliberate toy bank failure")):
                result = experiment.run_task(task, arrays, cfg, destination)
            with np.load(destination / "predictions.npz", allow_pickle=False) as saved:
                base = saved["prediction__ridge"]
                for name, record in result["methods"].items():
                    if name.startswith(("fixed_", "learned_", "joint_")):
                        self.assertEqual(record["status"], "fallback")
                        assert_array_equal(saved[f"prediction__{name}"], base)
                        self.assertTrue(any(key.startswith("bank:") and key.endswith(":fit")
                                            for key in record["dependencies"]))
            self.assertEqual(set(result["methods"]), set(experiment.methods(cfg)))

    def test_oof_failure_preserves_baselines_and_declared_dependent_fallbacks(self):
        arrays, cfg, task = toy_arrays(), toy_config(), toy_task()
        with tempfile.TemporaryDirectory(prefix="kahm-donor-failure-") as temporary:
            destination = Path(temporary) / "run"
            with threadpool_limits(limits=1), \
                    patch.object(experiment, "oof_predictions",
                                 side_effect=ValueError("deliberate toy donor failure")):
                result = experiment.run_task(task, arrays, cfg, destination)
            with np.load(destination / "predictions.npz", allow_pickle=False) as saved:
                base = saved["prediction__ridge"]
                for name, record in result["methods"].items():
                    if "_oof_" in name:
                        self.assertEqual(record["status"], "fallback")
                        assert_array_equal(saved[f"prediction__{name}"], base)
                        self.assertIn("oof:fit", record["dependencies"])
            self.assertEqual(set(result["methods"]), set(experiment.methods(cfg)))

    def test_insufficient_correction_rows_do_not_discard_available_baselines(self):
        arrays, cfg, task = toy_arrays(), toy_config(), toy_task()
        for key in ("X_train", "Y_train", "groups_train", "origins_train"):
            arrays[key] = arrays[key][:6]
        with tempfile.TemporaryDirectory(prefix="kahm-insufficient-correction-") as temporary:
            destination = Path(temporary) / "run"
            with threadpool_limits(limits=1):
                result = experiment.run_task(task, arrays, cfg, destination)
            self.assertEqual(set(result["methods"]), set(experiment.methods(cfg)))
            for name in ("ridge", "raw_rbf", "naive"):
                self.assertEqual(result["methods"][name]["status"], "ok")
            with np.load(destination / "predictions.npz", allow_pickle=False) as saved:
                for name in experiment.methods(cfg):
                    if name.startswith(("fixed_", "learned_")):
                        self.assertEqual(result["methods"][name]["status"], "fallback")
                        assert_array_equal(saved[f"prediction__{name}"], saved["prediction__ridge"])

    def test_declared_primary_family_matches_laws_and_contrasts(self):
        cfg = experiment.config()
        contrasts = cfg["primary_contrasts"]
        self.assertEqual(len(contrasts), len(set(map(tuple, contrasts))))
        self.assertEqual(cfg["confidence"]["primary_family_size"],
                         len(cfg["synthetic"]["laws"]) * len(contrasts))
        for pair in contrasts:
            self.assertEqual(len(pair), 2)
            self.assertTrue(set(pair) <= set(experiment.methods(cfg)))


if __name__ == "__main__":
    unittest.main()
