"""Analysis verification with fabricated predictions only; no forecast fitting.

The complete temporary 192-task fixture is deliberately manufactured evidence.
It tests accounting and decision logic, and is never a research result.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extensions.reconstruction_mechanism import analyze as analysis


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="kahm_fabricated_analysis_")
        cls.study = Path(cls.temporary.name) / "extensions" / "reconstruction_mechanism"
        cls.study.mkdir(parents=True)
        source = Path(analysis.__file__).parent
        for name in ("config.json", "PROTOCOL.md", "analyze.py"):
            shutil.copyfile(source / name, cls.study / name)
        cls.results = cls.study / "results"
        cls.results.mkdir()
        prepared = cls.study / "prepared_inputs"
        prepared.mkdir()
        cls.train = np.tile(np.array([[-1., -2., -3.], [1., 2., 3.]]), (256, 1))
        cls.target = np.zeros((1664, 3))
        cls.baseline = np.tile([1., 2., 3.], (1664, 1))
        cls.tasks = []
        for law, noise, seed in itertools.product(analysis.LAWS, analysis.NOISE_LEVELS, analysis.SEEDS):
            identifier = f"fabricated_{law}_{noise}_{seed}"
            path = prepared / f"{identifier}.npz"
            np.savez_compressed(path, A_Y=cls.train, D_Y=cls.target,
                                D_groups=np.repeat(np.arange(128), 13), offsets=[1, 4, 12])
            task = {"task_id": identifier, "law": law, "noise_level": noise,
                    "seed": seed, "repetition": analysis.SEEDS.index(seed),
                    "input": path.relative_to(cls.study).as_posix(),
                    "sha256": analysis.sha256(path), "loss_scale": [1., 4., 9.]}
            cls.tasks.append(task)
        analysis.write_json(prepared / "manifest.json", {
            "development_only": False, "tasks": cls.tasks,
            "fixture_warning": "Fabricated predictions for analyzer verification only"})
        frozen = [cls.study / name for name in ("config.json", "PROTOCOL.md", "analyze.py")]
        frozen += list(prepared.iterdir())
        analysis.write_json(cls.study / "freeze.json", {"files": [
            {"path": path.relative_to(Path(cls.temporary.name)).as_posix(),
             "sha256": analysis.sha256(path)} for path in frozen]})
        cls.freeze_hash = analysis.sha256(cls.study / "freeze.json")
        cls.write_predictions()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def write_predictions(cls, candidate_factor=.9, extreme_task=None):
        for task in cls.tasks:
            predictions = {method: cls.baseline.copy() for method in analysis.METHODS}
            predictions["kahm_departure"] *= candidate_factor
            if task["task_id"] == extreme_task:
                predictions["kahm_departure"][0] = 10 * cls.baseline[0]
            directory = cls.results / task["task_id"]
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory / "predictions.npz", Y=cls.target, **predictions)
            gammas = {method: [1., 1., 1.] for method in analysis.METHODS
                      if method not in ("ridge", "nonkahm_policy")}
            bounds = {}
            for method in ("common", "kahm_departure", "rbf_departure"):
                radius = math.sqrt(42 if method == "common" else 210)
                bound = [5 * radius] * 3
                bounds[method] = {
                    "radius": radius, "training_feature_mean_norm": 0.,
                    "response_mean_by_horizon": [0.] * 3,
                    "centered_response_rms_by_horizon": [10.] * 3,
                    "alpha_by_horizon": [1.] * 3,
                    "coefficient_norm_by_horizon": [.1] * 3,
                    "coefficient_norm_bound_by_horizon": [5.] * 3,
                    "readout_bound_by_horizon": bound,
                    "correction_bound_by_horizon": bound,
                    "max_abs_correction_by_horizon": np.max(
                        np.abs(predictions[method] - cls.baseline), axis=0).tolist(),
                    "coefficient_bound_violations": 0, "violations": 0, "global_bound": True}
            summary = {
                "status": "completed", "task": task, "input_sha256": task["sha256"],
                "freeze_sha256": cls.freeze_hash, "development_only": False,
                "fit_diagnostics": {
                    "gamma_by_horizon": gammas, "method_failures": {},
                    "policy": {"selected_method": "ridge", "selection_split": "C",
                               "A_target_variance": [1., 4., 9.],
                               "candidates": [{"method": method, "normalized_mse": float(i)}
                                              for i, method in enumerate(analysis.NONKAHM_METHODS)]}},
                "prediction_diagnostics": {
                    "recovered_prediction_cells": {method: 0 for method in analysis.METHODS},
                    "feature_bounds": {
                        method: {"finite": True, "width": width, "rows": 1664,
                                 "radius": math.sqrt(width), "max_norm": 1., "violations": 0}
                        for method, width in (("common", 42), ("kahm_departure", 168),
                                              ("rbf_departure", 168))},
                    "bounds": bounds, "kahm_fallback_count": 0},
                "resources": {"fit": {"wall_seconds": .001},
                              "prediction": {"wall_seconds": .001},
                              "fixture_warning": "Fabricated, not measured"},
            }
            analysis.write_json(directory / "summary.json", summary)

    def output(self):
        return self.study / "checks" / self.id().split(".")[-1]

    def test_complete_roster_advances_and_normalizes_each_horizon(self):
        result = analysis.analyze(self.study, self.results, self.output())
        self.assertEqual(result["decision"], "ADVANCE_TO_APPLICATION")
        self.assertEqual(result["completed_valid_tasks"], 192)
        self.assertEqual(result["independent_repetitions"], 24)
        self.assertEqual(result["scored_target_cells_per_method"], 958464)
        self.assertEqual(len(result["primary_contrasts"]), 6)
        for contrast in result["primary_contrasts"]:
            self.assertAlmostEqual(contrast["candidate_mean_nmse"], .81, places=12)
            self.assertAlmostEqual(contrast["comparator_mean_nmse"], 1., places=12)
            self.assertEqual(contrast["df"], 23)
        self.assertAlmostEqual(result["primary_contrasts"][0]["margin_gap_mean"], .16, places=12)

    def test_hand_computed_paired_t_bound(self):
        index = np.arange(24)
        row, gaps = analysis.paired_margin_contrast(
            .85 + .005 * index, 1 + .01 * index, comparator_factor=.97,
            contrast_id="fabricated", comparator_name="common", scope="nonlinear", margin=.03)
        np.testing.assert_allclose(gaps, .12 + .0047 * index, rtol=1e-13, atol=1e-15)
        # Sample variance of the integers 0,...,23 is 24*25/12 = 50.
        mean = .12 + .0047 * 11.5
        se = .0047 * math.sqrt(50 / 24)
        self.assertAlmostEqual(row["margin_gap_mean"], mean, places=14)
        self.assertAlmostEqual(row["margin_gap_se"], se, places=14)
        self.assertAlmostEqual(row["critical_t"], 2.582017198304117, places=12)
        self.assertAlmostEqual(row["simultaneous_one_sided_lower"],
                               mean - 2.582017198304117 * se, places=14)

    def test_nonpositive_margin_is_not_established(self):
        row, _ = analysis.paired_margin_contrast(
            np.full(24, .97), np.ones(24), comparator_factor=.97,
            contrast_id="fabricated", comparator_name="common", scope="nonlinear", margin=.03)
        self.assertEqual(row["simultaneous_one_sided_lower"], 0.)
        self.assertFalse(row["criterion_established"])

    def test_statistical_failure_stops_without_rescuing_secondary_results(self):
        try:
            self.write_predictions(candidate_factor=1.)
            result = analysis.analyze(self.study, self.results, self.output())
            self.assertEqual(result["decision"], "STOP_FORECASTING_PERFORMANCE_BRANCH")
            self.assertEqual(len(result["failed_contrasts"]), 5)
            self.assertTrue(result["stability_gate_pass"])
            self.assertTrue(result["primary_contrasts"][-1]["criterion_established"])
        finally:
            self.write_predictions()

    def test_catastrophic_error_screen_can_stop_statistical_success(self):
        try:
            self.write_predictions(extreme_task=self.tasks[0]["task_id"])
            result = analysis.analyze(self.study, self.results, self.output())
            self.assertEqual(result["decision"], "STOP_FORECASTING_PERFORMANCE_BRANCH")
            self.assertTrue(result["statistical_gate_pass"])
            self.assertEqual(result["failed_stability_tasks"], [self.tasks[0]["task_id"]])
        finally:
            self.write_predictions()

    def test_missing_task_blocks_and_removes_stale_aggregates(self):
        output = self.output()
        self.assertEqual(analysis.analyze(self.study, self.results, output)["decision"], "ADVANCE_TO_APPLICATION")
        path = self.results / self.tasks[0]["task_id"] / "summary.json"
        original = path.read_bytes()
        try:
            path.unlink()
            result = analysis.analyze(self.study, self.results, output)
            self.assertEqual(result["decision"], "TECHNICAL_BLOCK")
            self.assertEqual(result["completed_valid_tasks"], 191)
            self.assertFalse((output / "primary_contrasts.csv").exists())
            self.assertFalse((output / "replicate_metrics.csv").exists())
        finally:
            path.write_bytes(original)

    def test_variance_mismatch_is_rejected(self):
        task = copy.deepcopy(self.tasks[0])
        task["loss_scale"] = [2., 4., 9.]
        with self.assertRaisesRegex(analysis.InvalidEvidence, "variance"):
            analysis.load_task(self.study, self.results, task)

    def test_missing_or_duplicated_manifest_task_is_rejected(self):
        with self.assertRaisesRegex(analysis.InvalidEvidence, "Incomplete roster"):
            analysis.validate_roster({"tasks": self.tasks[:-1]})
        with self.assertRaisesRegex(analysis.InvalidEvidence, "repeated"):
            analysis.validate_roster({"tasks": self.tasks + self.tasks[:1]})

    def test_changed_frozen_source_is_rejected(self):
        path = self.study / "PROTOCOL.md"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\nChanged fabricated fixture\n")
            result = analysis.analyze(self.study, self.results, self.output())
            self.assertEqual(result["decision"], "TECHNICAL_BLOCK")
            self.assertIn("Frozen file changed", result["task_errors"][0]["error"])
        finally:
            path.write_bytes(original)

    def test_whole_method_failure_and_prediction_recovery_are_blocked(self):
        path = self.results / self.tasks[0]["task_id"] / "summary.json"
        original = path.read_bytes()
        try:
            summary = json.loads(original)
            summary["fit_diagnostics"]["method_failures"] = {"raw_hgb": "Fabricated failure"}
            analysis.write_json(path, summary)
            with self.assertRaisesRegex(analysis.InvalidEvidence, "Wholly failed"):
                analysis.load_task(self.study, self.results, self.tasks[0])
            summary["fit_diagnostics"]["method_failures"] = {}
            summary["prediction_diagnostics"]["recovered_prediction_cells"]["kahm_departure"] = 1
            analysis.write_json(path, summary)
            with self.assertRaisesRegex(analysis.InvalidEvidence, "prediction-cell recovery"):
                analysis.load_task(self.study, self.results, self.tasks[0])
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
