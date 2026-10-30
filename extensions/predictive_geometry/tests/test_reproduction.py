"""Checks for scientific provenance, portability and independent loss auditing."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reproduce.__main__ import check_destination, start_run
from reproduce.io import ROOT, materialize, read_json, verify_archive, write_json
from reproduce.worker import compare_records, prediction_audit, require_numerical_reproduction


class ReproductionTests(unittest.TestCase):
    def test_compressed_archive_preserves_every_historical_freeze_dependency(self):
        verified = verify_archive()
        self.assertGreater(verified["source_files_verified"], 40)
        with tempfile.TemporaryDirectory() as temporary:
            work = materialize(Path(temporary) / "work")
            import hashlib
            for name in ("protocol_freeze.json", "followup_freeze.json"):
                for relative, expected in read_json(work / name)["sha256"].items():
                    self.assertEqual(hashlib.sha256((work / relative).read_bytes()).hexdigest(), expected, relative)

    def test_prediction_audit_uses_independent_groups_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            y = np.array([[1.], [2.], [5.]])
            pred = np.array([[2.], [4.], [8.]])
            groups = np.array([7, 7, 9])
            # Correct unit means are (1+4)/2 and 9, not the row mean.
            loss = np.array([[2.5], [9.]])
            row = {"unit_loss_file": "values.npz", "unit_loss_key": "loss", "prediction_key": "pred", "test_mse": 5.75}
            np.savez(root / "values.npz", Y=y, pred=pred, groups=groups, loss=loss)
            self.assertEqual(prediction_audit([row], root)["prediction_and_loss_records_checked"], 1)
            permutation = np.array([0, 2, 1])
            np.savez(root / "values.npz", Y=y[permutation], pred=pred[permutation], groups=groups[permutation], loss=loss)
            self.assertEqual(prediction_audit([row], root, expected_records=1, expected_files=1)["prediction_and_loss_records_checked"], 1)
            np.savez(root / "values.npz", Y=y, pred=pred, groups=groups, loss=loss + .1)
            with self.assertRaises(ValueError):
                prediction_audit([row], root)

    def test_missing_required_prediction_metadata_cannot_pass_an_empty_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                prediction_audit([{"test_mse": .1}], temporary, expected_records=1)

    def test_comparison_rejects_missing_or_duplicate_replications(self):
        rows = [{"replicate": i, "test_mse": .1 + .01*i} for i in range(3)]
        result = compare_records(rows, rows, ("replicate",))
        self.assertEqual(result["maximum_absolute_test_mse_difference"], 0.)
        with self.assertRaises(ValueError):
            compare_records(rows[:-1], rows, ("replicate",))
        with self.assertRaises(ValueError):
            compare_records(rows + rows[:1], rows, ("replicate",))

    def test_run_directories_cannot_overwrite_evidence(self):
        for path in (ROOT, ROOT / "study", ROOT / "study/temporary", ROOT / "archive/new"):
            with self.assertRaises(ValueError):
                check_destination(path)
        self.assertEqual(check_destination(ROOT / "runs/temporary"), ROOT / "runs/temporary")

    def test_full_reproduction_cannot_accept_arbitrary_finite_mse_drift(self):
        same = compare_records([{"id": 1, "test_mse": .1}], [{"id": 1, "test_mse": .1}], ("id",))
        changed = compare_records([{"id": 1, "test_mse": .2}], [{"id": 1, "test_mse": .1}], ("id",))
        require_numerical_reproduction({"stage1": same, "capacity": same, "numerical_sensitivity": same})
        with self.assertRaises(ValueError):
            require_numerical_reproduction({"stage1": same, "capacity": changed, "numerical_sensitivity": same})

    def test_resume_rejects_stale_or_differently_scoped_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            args = Namespace(run_dir=run, resume=False)
            _, manifest = start_run(args, "smoke")
            self.assertEqual(manifest["mode"], "smoke")
            args.resume = True
            with self.assertRaises(ValueError):
                start_run(args, "full")
            manifest["packaging_sha256"] = {}
            write_json(run / "execution_manifest.json", manifest)
            with self.assertRaises(ValueError):
                start_run(args, "smoke")


if __name__ == "__main__":
    unittest.main()
