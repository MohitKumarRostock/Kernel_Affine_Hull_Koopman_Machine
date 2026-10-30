"""Synthetic-only integration checks for chronology and execution freezes."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from extensions.natural_forecasting import baselines, data, run


class HistorySpy:
    def fit_history(self, values, cutoff, max_history=None):
        self.cutoff = cutoff
        self.history_cap = max_history
        return self

    def predict_history(self, values, origins, horizon):
        self.origins = np.asarray(origins)
        return np.zeros((len(origins), horizon))


class ConstantRegressor:
    def __init__(self, constant):
        self.constant = float(constant)

    def fit(self, X, Y):
        self.fitted_rows = len(X)
        self.horizon = Y.shape[1]
        return self

    def predict(self, X):
        return np.full((len(X), self.horizon), self.constant)

    def metadata(self):
        return {"synthetic_constant": self.constant, "fitted_rows": self.fitted_rows}


def synthetic_task(cap=32):
    source = data.SeriesRecord("synthetic", "T1", np.arange(120, dtype=float),
        "2000-01-01T00:00:00", "daily")
    spec = {"family_id": "synthetic_family", "role": "development", "seasonal_period": 4,
        "zenodo": {"record_id": 0, "doi": "synthetic", "version": "test"},
        "archive": {"sha256": "synthetic", "member_sha256": "synthetic"},
        "license": {"id": "test"}}
    task = data.make_task(source, spec, {"context": 8, "horizon": 3},
        max_train_windows=cap, max_eval_windows=8)
    row = task.to_dict()
    row.update(capacity_regime={"max_train_windows": cap}, seed=17)
    arrays = {"values": task.values}
    for split in ("train", "validation", "test"):
        for key in ("X", "Y", "origin"):
            arrays[f"{key}_{split}"] = getattr(task, f"{key}_{split}")
    refit = data.make_windows(task.values, task.context, task.horizon,
        0, task.validation_end, cap)
    arrays.update(X_refit=refit.X, Y_refit=refit.Y, origin_refit=refit.origins)
    return task, row, arrays


class RunnerChronologyTests(unittest.TestCase):
    def test_history_adapters_use_fit_cutoff_and_first_future_origin(self):
        _, task, arrays = synthetic_task()
        spy = HistorySpy()
        run.fit_pipeline(spy, "seasonal_naive", arrays, task, "train")
        self.assertEqual(spy.cutoff, 72)
        self.assertEqual(spy.history_cap, 40)
        run.predict_pipeline(spy, "seasonal_naive", arrays, task, "validation")
        np.testing.assert_array_equal(spy.origins, arrays["origin_validation"] + 1)
        self.assertEqual(spy.origins.min(), 72)
        run.fit_pipeline(spy, "holt", arrays, task, "refit")
        self.assertEqual(spy.cutoff, 96)
        run.predict_pipeline(spy, "holt", arrays, task, "test")
        self.assertEqual(spy.origins.min(), 96)

    def test_actual_seasonal_baseline_excludes_current_and_future_targets(self):
        _, task, arrays = synthetic_task()
        # The first test target is96; the most recent complete season is92..95.
        arrays["origin_test"] = np.array([95])
        baseline = baselines.SeasonalNaive(4)
        run.fit_pipeline(baseline, "seasonal_naive", arrays, task, "refit")
        prediction = run.predict_pipeline(baseline, "seasonal_naive", arrays, task, "test")
        np.testing.assert_array_equal(prediction, [[92, 93, 94]])
        altered = dict(arrays)
        altered["values"] = arrays["values"].copy()
        altered["values"][96:] = -1e15
        np.testing.assert_array_equal(prediction,
            run.predict_pipeline(baseline, "seasonal_naive", altered, task, "test"))

    def test_validation_winner_is_not_changed_by_favorable_test_outcome(self):
        _, task, arrays = synthetic_task()
        arrays["Y_validation"] = np.zeros_like(arrays["Y_validation"])
        arrays["Y_test"] = np.ones_like(arrays["Y_test"])
        fitted = []

        def construct(method, configuration, task, protocol):
            model = ConstantRegressor(configuration["regularization"])
            fitted.append(model)
            return model

        with patch.object(run, "construct", side_effect=construct):
            result = run.run_method(task, arrays, "mlp", {"practical_search": {"regularization": [0, 1]}})
        candidate = result["candidate"]
        self.assertEqual(candidate["selected_parameters"], {"regularization": 0})
        self.assertEqual(candidate["validation_loss"], 0.0)
        self.assertEqual(candidate["test_loss"], 1.0)
        self.assertEqual(len(fitted), 3)  # Two train-only candidates, one selected refit.
        self.assertEqual(fitted[-1].fitted_rows, len(arrays["X_refit"]))
        self.assertLess(candidate["tuning_fit_max_target_index"], task["train_end"])
        self.assertLess(candidate["selection_max_target_index"], task["validation_end"])
        self.assertLess(candidate["refit_max_target_index"], task["test_start"])
        self.assertGreaterEqual(candidate["test_min_target_index"], task["test_start"])
        for node in candidate["cost_nodes"]:
            self.assertGreaterEqual(node["seconds"], 0)
            self.assertGreaterEqual(node["process_cpu_seconds"], 0)
            self.assertGreater(node["peak_rss_bytes"], 0)

    def test_prepare_constructs_pretest_refit_without_test_targets(self):
        protocol = {
            "capacity_regimes": {"small": {"max_train_windows": 32}, "medium": {"max_train_windows": 64}},
            "data": {"max_eval_origins": 8, "max_series_per_source": 1}, "seeds": [17, 43],
        }
        manifest = {"sources": [{"source_id": "synthetic", "role": "development"}]}

        def tasks(**kwargs):
            return [synthetic_task(kwargs["max_train_windows"])[0]]

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run, "load_protocol", return_value=protocol), \
                patch.object(run.data, "load_manifest", return_value=manifest), \
                patch.object(run.data, "load_tasks", side_effect=tasks), \
                contextlib.redirect_stdout(io.StringIO()):
            roster = run.prepare(directory)
            self.assertEqual(len(roster["tasks"]), 4)
            self.assertEqual(len({t["task_id"] for t in roster["tasks"]}), 4)
            # Seeds share prepared data, while the two capacity caps have their
            # own arrays. No comparative model is called by prepare.
            self.assertEqual(len({t["cache_path"] for t in roster["tasks"]}), 2)
            for task in roster["tasks"]:
                arrays = run.load_arrays(directory, task)
                self.assertTrue(np.all(arrays["origin_train"] + 3 < 72))
                self.assertTrue(np.all(arrays["origin_refit"] + 3 < 96))
                self.assertTrue(np.all(arrays["origin_test"] + 1 >= 96))
                # Old train/validation-straddling windows become legitimate
                # historical observations at pretest refit, not discarded forever.
                self.assertTrue(any(69 <= origin <= 71 for origin in arrays["origin_refit"]))
                np.testing.assert_array_equal(arrays["Y_refit"][:, -1], arrays["origin_refit"] + 3)
            Path(directory, "development").mkdir()
            with self.assertRaisesRegex(RuntimeError, "after model execution"):
                run.prepare(directory)


class FreezeBoundaryTests(unittest.TestCase):
    def test_confirmation_cannot_start_without_final_freeze(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run, "run_task") as execute:
            with self.assertRaises(FileNotFoundError):
                run.run_phase("confirmation", directory)
            execute.assert_not_called()
            self.assertFalse(Path(directory, "confirmation").exists())

    def test_confirmation_does_not_allow_convenient_task_subset(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run, "check_freeze", return_value={}), \
                patch.object(run, "load_protocol", return_value={}), \
                patch.object(run, "run_task") as execute:
            Path(directory, "task_roster.json").write_text(json.dumps({"tasks": [
                {"task_id": "heldout1", "role": "confirmation"},
                {"task_id": "heldout2", "role": "confirmation"},
            ]}))
            with self.assertRaisesRegex(ValueError, "cannot select a subset"):
                run.run_phase("confirmation", directory, limit=1)
            execute.assert_not_called()
            self.assertFalse(Path(directory, "confirmation").exists())

    def test_freeze_rejects_code_roster_dependency_and_diagnostic_changes(self):
        environment = {"python": "test-python", "packages": {"numpy": "test-version"}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run, "code_hashes", return_value={"scientific.py": "original"}) as code, \
                patch.object(run, "environment", return_value=environment) as runtime:
            root = Path(directory)
            roster = root / "task_roster.json"
            diagnostic = root / "frozen_diagnostic.json"
            development = root / "development" / "task.json"
            development.parent.mkdir()
            roster.write_text('{"tasks": []}\n')
            diagnostic.write_text('{"coefficients": [1]}\n')
            development.write_text('{"loss": 1}\n')
            freeze = {"code_sha256": code.return_value, "environment": environment,
                "roster_sha256": run.digest_file(roster),
                "diagnostic_sha256": run.digest_file(diagnostic),
                "development_result_sha256": {development.name: run.digest_file(development)}}
            run.write_json(root / "confirmation_freeze.json", freeze)
            self.assertEqual(run.check_freeze(root, "confirmation_freeze.json"), freeze)
            code.return_value = {"scientific.py": "changed"}
            with self.assertRaisesRegex(RuntimeError, "code/protocol changed"):
                run.check_freeze(root, "confirmation_freeze.json")
            code.return_value = freeze["code_sha256"]
            roster.write_text('{"tasks": ["added"]}\n')
            with self.assertRaisesRegex(RuntimeError, "roster changed"):
                run.check_freeze(root, "confirmation_freeze.json")
            roster.write_text('{"tasks": []}\n')
            runtime.return_value = {"python": "test-python", "packages": {"numpy": "changed-version"}}
            with self.assertRaisesRegex(RuntimeError, "versions changed"):
                run.check_freeze(root, "confirmation_freeze.json")
            runtime.return_value = environment
            development.write_text('{"loss": 2}\n')
            with self.assertRaisesRegex(RuntimeError, "Development outcomes changed"):
                run.check_freeze(root, "confirmation_freeze.json")
            development.write_text('{"loss": 1}\n')
            diagnostic.write_text('{"coefficients": [2]}\n')
            with self.assertRaisesRegex(RuntimeError, "diagnostic changed"):
                run.check_freeze(root, "confirmation_freeze.json")

    def test_initial_freeze_rejects_configuration_changes_after_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            roster = {"tasks": [], "protocol_sha256": "stale-protocol",
                "source_manifest_sha256": run.digest_file(run.ROOT / "data_sources.json")}
            run.write_json(Path(directory) / "task_roster.json", roster)
            with self.assertRaisesRegex(RuntimeError, "Protocol changed after preparing"):
                run.initial_freeze(directory)
            roster["protocol_sha256"] = run.digest_file(run.ROOT / "protocol.json")
            roster["source_manifest_sha256"] = "stale-source-manifest"
            run.write_json(Path(directory) / "task_roster.json", roster)
            with self.assertRaisesRegex(RuntimeError, "Source manifest changed after preparing"):
                run.initial_freeze(directory)

    def test_prediction_audit_checks_saved_losses_and_frozen_target_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = root / "development"
            development.mkdir()
            target = np.array([[1., 2.], [3., 4.]])
            origins = np.array([9, 11])
            original = root / "original.npz"
            np.savez_compressed(original, Y_test=target, origin_test=origins)
            task = {"task_id": "synthetic", "cache_path": original.name,
                "cache_sha256": run.digest_file(original)}
            run.write_json(root / "task_roster.json", {"tasks": [task]})
            prediction = development / "prediction.npz"

            def save_prediction(y):
                np.savez_compressed(prediction, target=y, origin=origins,
                    synthetic_method=np.zeros_like(y),
                    synthetic_method__origin_losses=np.mean(y ** 2, axis=1))

            save_prediction(target)
            row = {"task_id": "synthetic", "predictions_file": prediction.name,
                "predictions_sha256": run.digest_file(prediction),
                "candidates": [{"candidate_id": "synthetic_method", "test_loss": 7.5}]}
            result = development / "result.json"
            run.write_json(result, row)
            audit = run.audit_results(root, roles=("development",))
            self.assertEqual(audit["predictions_recomputed"], 1)
            self.assertTrue(audit["source_targets_and_origins_equal"])
            row["candidates"][0]["test_loss"] = 0.0
            run.write_json(result, row)
            with self.assertRaises(ValueError):
                run.audit_results(root, roles=("development",))
            row["candidates"][0]["test_loss"] = 7.5
            save_prediction(target + 1)
            # A self-consistent new artifact hash cannot legitimize changed
            # evaluation targets: they must still match the prepared windows.
            row["predictions_sha256"] = run.digest_file(prediction)
            run.write_json(result, row)
            with self.assertRaisesRegex(ValueError, "targets differ from frozen"):
                run.audit_results(root, roles=("development",))

    def test_cache_digest_prevents_silent_prepared_data_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "arrays.npz")
            np.savez_compressed(path, values=np.arange(10))
            task = {"cache_path": path.name, "cache_sha256": run.digest_file(path)}
            np.testing.assert_array_equal(run.load_arrays(directory, task)["values"], np.arange(10))
            np.savez_compressed(path, values=np.arange(10) + 1)
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                run.load_arrays(directory, task)

    def test_analysis_roster_preserves_family_and_complete_nested_task_ids(self):
        roster = {"tasks": [
            {"task_id": "dev__short__seed17", "role": "development", "family_id": "dev_family"},
            {"task_id": "dev__long__seed43", "role": "development", "family_id": "dev_family"},
            {"task_id": "heldout__short__seed17", "role": "confirmation", "family_id": "reserved_family"},
            {"task_id": "heldout__long__seed43", "role": "confirmation", "family_id": "reserved_family"},
        ]}
        configuration = run.analysis_configuration(roster)
        self.assertEqual(configuration["reserved_source_families"], ["reserved_family"])
        self.assertEqual(set(configuration["expected_confirmation_task_ids"]),
            {"heldout__short__seed17", "heldout__long__seed43"})
        self.assertEqual(len(configuration["expected_development_task_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
