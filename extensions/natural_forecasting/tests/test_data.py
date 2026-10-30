"""Chronology, availability and train-only selection invariants (no model scores)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import (
    SeriesRecord, capped_origins, load_manifest,
    make_task, make_windows, parse_tsf, select_series,
    split_cutoffs, training_mse_scale, valid_origins, verify_archive,
)


def example_spec() -> dict:
    return {
        "source_id": "example", "family_id": "family", "role": "development",
        "seasonal_period": 4,
        "zenodo": {"record_id": 1, "doi": "example-doi", "version": "1"},
        "archive": {"sha256": "archive-digest", "member_sha256": "member-digest"},
        "license": {"id": "cc-by-4.0"},
    }


def record(identifier: str, values: np.ndarray) -> SeriesRecord:
    return SeriesRecord("example", identifier, values, "2000-01-01T00:00:00", "daily")


class TemporalWindowTests(unittest.TestCase):
    def test_complete_target_intervals_and_hand_computed_boundaries(self):
        values = np.arange(40, dtype=float)
        self.assertEqual(split_cutoffs(40), (24, 32, 40))
        training = make_windows(values, 5, 4, 0, 24)
        validation = make_windows(values, 5, 4, 24, 32)
        testing = make_windows(values, 5, 4, 32, 40)
        np.testing.assert_array_equal(training.origins, np.arange(4, 20))
        np.testing.assert_array_equal(validation.origins, np.arange(23, 28))
        np.testing.assert_array_equal(testing.origins, np.arange(31, 36))
        np.testing.assert_array_equal(training.X[-1], [15, 16, 17, 18, 19])
        np.testing.assert_array_equal(training.Y[-1], [20, 21, 22, 23])
        np.testing.assert_array_equal(validation.X[0], [19, 20, 21, 22, 23])
        np.testing.assert_array_equal(validation.Y[0], [24, 25, 26, 27])
        # Origins 20..22 would straddle the training/validation split and must
        # occur in neither split, despite all observations being available.
        self.assertFalse(set(range(20, 23)) & set(training.origins))
        self.assertFalse(set(range(20, 23)) & set(validation.origins))
        for windows, start, end in ((training, 0, 24), (validation, 24, 32), (testing, 32, 40)):
            self.assertTrue(np.all(windows.target_start >= start))
            self.assertTrue(np.all(windows.target_end <= end))
            self.assertTrue(np.all(windows.X[:, -1] < windows.Y[:, 0]))

    def test_missingness_excludes_any_context_or_target_without_interpolation(self):
        values = np.arange(12, dtype=float)
        values[5] = np.nan
        windows = make_windows(values, 3, 2, 0, 12)
        # A five-point window includes position5 for origins3..7.
        np.testing.assert_array_equal(windows.origins, [2, 8, 9])
        self.assertEqual(windows.candidate_count, 8)
        self.assertEqual(windows.available_count, 3)
        np.testing.assert_array_equal(windows.X[0], [0, 1, 2])
        np.testing.assert_array_equal(windows.Y[0], [3, 4])
        self.assertTrue(np.isnan(values[5]))

    def test_origin_cap_spans_period_and_uses_indices_only(self):
        origins = np.arange(17, 117)
        np.testing.assert_array_equal(capped_origins(origins, 4), [17, 50, 83, 116])
        np.testing.assert_array_equal(capped_origins(origins, 1), [116])
        with self.assertRaisesRegex(ValueError, "positive"):
            capped_origins(np.array([], dtype=int), 0)
        values = np.arange(200, dtype=float)
        transformed = 100 * values + 7
        a = make_windows(values, 12, 6, 120, 160, 7)
        b = make_windows(transformed, 12, 6, 120, 160, 7)
        np.testing.assert_array_equal(a.origins, b.origins)
        np.testing.assert_array_equal(b.X, 100 * a.X + 7)
        np.testing.assert_array_equal(b.Y, 100 * a.Y + 7)

    def test_metadata_length_feasibility_before_model_outcomes(self):
        hospital = np.arange(84, dtype=float)
        traffic = np.arange(104, dtype=float)
        self.assertEqual(len(valid_origins(hospital, 48, 6, 0, 50)[0]), 0)
        self.assertEqual(len(valid_origins(traffic, 52, 8, 0, 62)[0]), 3)
        self.assertEqual(len(valid_origins(hospital, 24, 6, 0, 50)[0]), 21)
        self.assertEqual(len(valid_origins(traffic, 26, 8, 0, 62)[0]), 29)


class TrainOnlyTests(unittest.TestCase):
    def test_difference_mse_has_closed_form_and_declared_fallback(self):
        values = 2 * np.arange(50, dtype=float) + 11
        scale, method, count = training_mse_scale(values, 30, 3)
        self.assertEqual(scale, 36.0)
        self.assertEqual(method, "seasonal_naive_mse")
        self.assertEqual(count, 27)
        periodic = np.tile([0.0, 1.0], 20)
        scale, method, count = training_mse_scale(periodic, 20, 2)
        self.assertEqual(scale, 1.0)
        self.assertEqual(method, "first_difference_mse_fallback")
        self.assertEqual(count, 19)
        short = np.arange(10, dtype=float)
        self.assertEqual(training_mse_scale(short, 10, 52), (1.0, "first_difference_mse_fallback", 9))
        with self.assertRaisesRegex(ValueError, "constant"):
            training_mse_scale(np.ones(20), 12, 4)

    def test_adversarial_evaluation_changes_cannot_change_training_or_roster(self):
        records = [record(f"T{i}", np.arange(120, dtype=float) * (i + 1) + i) for i in range(7)]
        constant_train = np.r_[np.ones(72), np.arange(48, dtype=float)]
        records.append(record("constant_train", constant_train))
        regimes = [{"context": 4, "horizon": 2}, {"context": 8, "horizon": 3}]
        selected, details = select_series(records, regimes, 4, maximum=4)
        changed = []
        for source_record in records:
            values = source_record.values.copy()
            values[72:] = np.nan  # Alter both magnitudes and future availability.
            changed.append(record(source_record.series_id, values))
        selected_changed, details_changed = select_series(reversed(changed), regimes, 4, maximum=4)
        self.assertEqual([r.series_id for r in selected], [r.series_id for r in selected_changed])
        self.assertEqual(details, details_changed)
        self.assertIn("constant_train", details["excluded_series"])
        first = make_task(selected[0], example_spec(), regimes[0])
        altered = make_task(selected_changed[0], example_spec(), regimes[0])
        np.testing.assert_array_equal(first.X_train, altered.X_train)
        np.testing.assert_array_equal(first.Y_train, altered.Y_train)
        self.assertEqual(first.loss_scale, altered.loss_scale)
        self.assertTrue(first.supported)
        self.assertFalse(altered.supported)
        self.assertEqual(altered.task_id, first.task_id)
        self.assertIn("validation", altered.unsupported_reason)
        self.assertIn("test", altered.unsupported_reason)

    def test_roster_eligibility_requires_both_regimes_and_is_budget_independent(self):
        regimes = [{"context": 4, "horizon": 2}, {"context": 24, "horizon": 6}]
        # n60 -> training36 -> only7 long-regime windows.  n100 ->training60 ->31.
        records = [record("short", np.arange(60, dtype=float)), record("long", np.arange(100, dtype=float))]
        selected, detail = select_series(records, regimes, 4, maximum=4)
        self.assertEqual([r.series_id for r in selected], ["long"])
        self.assertIn("short", detail["excluded_series"])
        a = make_task(selected[0], example_spec(), regimes[0], max_train_windows=16)
        b = make_task(selected[0], example_spec(), regimes[0], max_train_windows=512)
        self.assertEqual(a.task_id, b.task_id)
        self.assertEqual(a.loss_scale, b.loss_scale)
        self.assertEqual(a.selection_hash, b.selection_hash)
        self.assertEqual(len(a.X_train), 16)
        self.assertGreater(len(b.X_train), 16)
        metadata = a.to_dict()
        self.assertEqual(metadata["validation_start"], 60)
        self.assertEqual(metadata["test_start"], 80)
        self.assertLessEqual(max(metadata["split_details"]["train"]["target_end"]), 60)
        json.dumps(metadata, allow_nan=False)


class SourceIntegrityTests(unittest.TestCase):
    def test_tsf_preserves_raw_values_dates_and_missing_markers(self):
        text = """# Source attribution retained in archive.
@relation example
@attribute series_name string
@attribute start_timestamp date
@frequency monthly
@missing true
@data
T1:2001-02-01 00-00-00:10,-2,?,3.25
T2:1999-01-01 00-00-00:1,2,3
"""
        records = parse_tsf(text, "example")
        self.assertEqual(records[0].start_timestamp, "2001-02-01T00:00:00")
        self.assertEqual(records[0].frequency, "monthly")
        np.testing.assert_array_equal(records[0].values, [10, -2, np.nan, 3.25])
        self.assertFalse(records[0].values.flags.writeable)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_tsf(text + "T1:2001-02-01 00-00-00:1,2\n", "example")

    def test_archive_mutation_is_rejected(self):
        payload = b"fixed public snapshot"
        spec = {"archive": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.zip"
            archive.write_bytes(payload)
            verify_archive(archive, spec)
            archive.write_bytes(b"changed public data!")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_archive(archive, spec)

    def test_manifest_families_disjoint_and_versioned_attribution_complete(self):
        manifest = load_manifest()
        development = {s["family_id"] for s in manifest["sources"] if s["role"] == "development"}
        confirmation = {s["family_id"] for s in manifest["sources"] if s["role"] == "confirmation"}
        self.assertEqual(len(development), 4)
        self.assertEqual(len(confirmation), 4)
        self.assertFalse(development & confirmation)
        for source in manifest["sources"]:
            self.assertEqual(source["license"]["id"], "cc-by-4.0")
            self.assertTrue(source["primary_references"])
            self.assertTrue(source["creators"])
            self.assertEqual(len(source["archive"]["sha256"]), 64)
            self.assertEqual(len(source["archive"]["member_sha256"]), 64)
            self.assertTrue(source["zenodo"]["version"])
        corrupted = copy.deepcopy(manifest)
        corrupted["sources"][-1]["family_id"] = corrupted["sources"][0]["family_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crosses development/confirmation"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
