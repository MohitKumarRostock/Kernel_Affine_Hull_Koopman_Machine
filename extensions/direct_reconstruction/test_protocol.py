"""Tests of consequential chronology, inference, and evidence-freeze invariants.

These tests do not evaluate declared study seeds or natural model outcomes.
Synthetic numerical-law smoke checks use only the excluded development seed.
Run from the repository root with ``python extensions/direct_reconstruction/test_protocol.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run
import analyze
import prepare_sources


class ProtocolTests(unittest.TestCase):
    def test_history_and_targets_never_cross_episode_boundaries(self):
        episodes = np.vstack((np.arange(30), 100 + np.arange(30))).astype(float)
        windows = run.make_windows(episodes, history=4, offsets=(1, 3, 6), stride=5)
        self.assertEqual(windows["X"].shape, (10, 4))
        np.testing.assert_array_equal(windows["X"][0], [0, 1, 2, 3])
        np.testing.assert_array_equal(windows["Y"][0], [4, 6, 9])
        np.testing.assert_array_equal(windows["X"][5], [100, 101, 102, 103])
        np.testing.assert_array_equal(windows["Y"][5], [104, 106, 109])
        self.assertTrue(np.all(windows["groups"][:5] == 0))
        self.assertTrue(np.all(windows["groups"][5:] == 1))

    def test_seed_allocation_has_no_declared_split_collision(self):
        seeds = [run.seed_from([seed, law_index, split, run.GENERATOR_SALT])
                 for seed in run.STUDY_SEEDS for law_index in range(3) for split in range(4)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertNotIn(run.DEVELOPMENT_SEED, run.STUDY_SEEDS)

    def test_unequal_episode_sizes_do_not_create_pseudoreplication(self):
        target = np.zeros((4, 3))
        prediction = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [3, 3, 3]])
        result = analyze.score(target, prediction, np.array([0, 0, 0, 1]), [1, 4, 12])
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(result["mse"], 5.0)
        self.assertNotEqual(result["mse"], float(np.mean(prediction ** 2)))

    def test_primary_interval_sign_and_replication_unit(self):
        control = np.linspace(1, 2, 12)
        candidate = control - np.linspace(0.1, 0.2, 12)
        result = analyze.paired_interval(control, candidate)
        self.assertEqual(result["n_repetitions"], 12)
        self.assertEqual(result["degrees_of_freedom"], 11)
        self.assertAlmostEqual(result["mean_difference"], 0.15)
        self.assertGreater(result["interval_low"], 0)
        reverse = analyze.paired_interval(candidate, control)
        self.assertAlmostEqual(reverse["interval_high"], -result["interval_low"])
        unadjusted = analyze.paired_interval(control, candidate, family_size=1)
        self.assertGreater(result["critical_t"], unadjusted["critical_t"])

    def test_incomplete_pairs_cannot_silently_drop_a_repetition(self):
        candidate = np.ones(12)
        candidate[5] = np.nan
        with self.assertRaises(ValueError):
            analyze.paired_interval(np.ones(12), candidate)

    def test_equal_collection_weight_is_not_equal_series_weight(self):
        rows = [{"source_key": "three_series", "scaled_mse": 1.0} for _ in range(3)]
        rows += [{"source_key": "one_series", "scaled_mse": 9.0}]
        mean, by_collection = analyze.equal_collection_mean(rows)
        self.assertEqual(mean, 5.0)
        self.assertEqual(by_collection, {"one_series": 9.0, "three_series": 1.0})
        self.assertNotEqual(mean, np.mean([row["scaled_mse"] for row in rows]))

    def test_available_windows_use_only_requested_targets(self):
        raw = np.arange(40, dtype=float)
        raw[21] = np.nan
        offsets = np.array([1, 3, 7])
        available, _ = prepare_sources.eligible_origins(raw, 4, offsets, 20, 30)
        self.assertIn(19, available)  # targets20,22,26; unused21 may be missing.
        self.assertNotIn(20, available)  # requested first target21 is missing.
        for origin in available:
            self.assertGreaterEqual(origin + min(offsets), 20)
            self.assertLess(origin + max(offsets), 30)

    def test_cap_and_chronological_purge_preserve_availability(self):
        available = np.arange(11, 211)
        positions = prepare_sources.cap_positions(len(available), 96)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], len(available) - 1)
        self.assertTrue(np.all(np.diff(positions) > 0))
        origins = available[positions]
        validation = np.arange(len(origins)) >= 4 * len(origins) // 5
        training = (~validation) & (origins + 12 <= origins[validation][0])
        self.assertLessEqual(np.max(origins[training] + 12), np.min(origins[validation]))
        self.assertTrue(np.any((~training) & (~validation)))

    def test_seasonal_loss_scale_cannot_use_later_values(self):
        prefix = np.arange(20, dtype=float)
        scale, method, count = prepare_sources.training_loss_scale(prefix, 4)
        self.assertEqual(scale, 16)
        self.assertEqual(method, "seasonal_mse")
        self.assertEqual(count, 16)
        prefix[5] = np.nan
        scale, method, count = prepare_sources.training_loss_scale(prefix, 4)
        self.assertEqual(scale, 16)
        self.assertEqual(count, 14)

    def test_identifier_selection_does_not_depend_on_observations(self):
        digest = prepare_sources.identifier_hash("collection", "series")
        self.assertEqual(digest, prepare_sources.identifier_hash("collection", "series"))
        self.assertNotEqual(digest, prepare_sources.identifier_hash("collection", "other"))
        self.assertNotEqual(digest, prepare_sources.identifier_hash("other", "series"))

    def test_canonical_array_hash_detects_dtype_shape_and_value(self):
        original = np.arange(6, dtype=np.float64).reshape(2, 3)
        expected = run.array_hash(original)
        self.assertEqual(expected, run.array_hash(np.asfortranarray(original)))
        self.assertNotEqual(expected, run.array_hash(original.astype(np.float32)))
        self.assertNotEqual(expected, run.array_hash(original.reshape(3, 2)))
        changed = original.copy()
        changed[0, 0] = 1
        self.assertNotEqual(expected, run.array_hash(changed))

    def test_freeze_rejects_changed_inputs_and_changed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, output = root / "repo", root / "run"
            source = repository / "numeric.py"
            source.parent.mkdir()
            source.write_text("version=1\n")
            output.mkdir()
            snapshot = output / "source_snapshot/numeric.py"
            snapshot.parent.mkdir()
            snapshot.write_bytes(source.read_bytes())
            run.write_json(output / "protocol.json", {"roles": "A/B/C/D"})
            (output / "inputs.npz").write_bytes(b"frozen data")
            run.write_json(output / "freeze.json", {
                "schema_version": run.SCHEMA_VERSION,
                "protocol_sha256": run.sha256_file(output / "protocol.json"),
                "source_snapshot": "source_snapshot",
                "scientific_files": {"numeric.py": run.sha256_file(source)},
                "prepared_inputs": {"inputs.npz": run.sha256_file(output / "inputs.npz")},
            })
            with patch.object(run, "REPOSITORY", repository):
                run.verify_freeze(output)
                source.write_text("version=2\n")
                with self.assertRaisesRegex(ValueError, "Scientific source changed"):
                    run.verify_freeze(output)
                source.write_text("version=1\n")
                (output / "inputs.npz").write_bytes(b"changed data")
                with self.assertRaisesRegex(ValueError, "Prepared inputs changed"):
                    run.verify_freeze(output)

    def test_failed_collection_keeps_full_roster_aggregate_undefined(self):
        methods = ["ridge", "reconstruction_kahm", *run.PRIMARY_CONTROLS]
        tasks = [{"task_id": f"{law}_{seed}", "suite": "synthetic", "law": law, "seed": seed}
                 for law in run.LAWS for seed in run.STUDY_SEEDS]
        for source, count in (("a", 3), ("b_unavailable", 3), ("c", 3), ("d", 1)):
            tasks.extend({"task_id": f"{source}_{index}", "suite": "natural", "source_key": source}
                         for index in range(count))
        protocol = {"schema_version": run.SCHEMA_VERSION, "methods": methods, "tasks": tasks,
                    "laws": list(run.LAWS), "seeds": list(run.STUDY_SEEDS),
                    "primary_candidate": run.PRIMARY_CANDIDATE,
                    "primary_controls": list(run.PRIMARY_CONTROLS),
                    "primary_family_size": 18, "family_alpha": 0.05}
        metrics = []
        for task in tasks:
            if task.get("source_key") == "b_unavailable":
                continue
            for method in methods:
                metrics.append({"task_id": task["task_id"], "suite": task["suite"],
                                "source_key": task.get("source_key", ""), "method": method,
                                "mse": 1.0, "scaled_mse": 1.0,
                                "charged_wall_seconds": 1.0, "charged_cpu_seconds": 1.0,
                                "selected_serialized_bytes": 1})
        summary = analyze.summarize(protocol, metrics, [], [], [])
        complete_rows = [row for row in summary["method_aggregates"]
                         if row["group"] == "equal_collection"]
        self.assertTrue(all(row["mean_loss"] is None and not row["complete"] for row in complete_rows))
        available_rows = [row for row in summary["method_aggregates"]
                          if row["group"] == "available_collection"]
        self.assertTrue(all(row["mean_loss"] == 1.0 and row["task_count"] == 7 for row in available_rows))
        failed_rows = [row for row in summary["natural_collection_metrics"]
                       if row["source_key"] == "b_unavailable"]
        self.assertTrue(all(row["mean_scaled_mse"] is None and row["available_series_count"] == 0
                            and row["series_count"] == 3 for row in failed_rows))
        self.assertFalse(summary["complete_roster"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
