"""Focused failures for the new saved-statistics checker; no fitting or simulation."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_saved_losses", ROOT / "check_saved_losses.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class SavedLossFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence, _ = checker.load_verified_bundle()
        cls.rows = checker.csv_rows(cls.evidence["selected_losses.csv"], checker.SELECTED_FIELDS, "losses")
        cls.indexed = checker.validate_selected(cls.rows)

    def test_changed_compressed_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            shutil.copytree(ROOT, bundle)
            path = bundle / "evidence/selected_losses.csv.gz"
            data = bytearray(path.read_bytes())
            data[-16] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(checker.CheckError, "File identity mismatch"):
                checker.load_verified_bundle(bundle)

    def test_duplicate_record_cannot_replace_a_missing_repetition(self):
        rows = list(self.rows)
        rows[-1] = rows[0]
        with self.assertRaisesRegex(checker.CheckError, "Duplicate selected-loss row"):
            checker.validate_selected(rows)

    def test_shared_cap_must_follow_the_recorded_selection_rule(self):
        # Change both C and D together, so C/D identity alone cannot catch it.
        rows = list(self.rows)
        for i, row in enumerate(rows):
            if (row["law"], row["repetition"], row["view"], row["procedure"]) == (
                    "linear_ar", "0", "shared", "K__norm__linear"):
                row = dict(row)
                row["cap"] = str(next(cap for cap in checker.CAPS if str(cap) != row["anchor_cap"]))
                rows[i] = row
        with self.assertRaisesRegex(checker.CheckError, "Shared cap differs"):
            checker.validate_selected(rows)

    def test_wrong_interval_classification_is_rejected(self):
        _, rebuilt, _ = checker.rebuild_primary(self.indexed)
        recorded = checker.csv_rows(self.evidence["primary_contrasts.csv"], checker.PRIMARY_FIELDS, "contrasts")
        recorded[0]["classification"] = "candidate_higher_mean_loss"
        text = {"id", "law", "decoder", "candidate", "reference", "view", "question", "classification"}
        counts = {"n", "candidate_lower_repetitions", "candidate_higher_repetitions"}
        with self.assertRaisesRegex(checker.CheckError, "Label/decision mismatch.*classification"):
            checker.compare_table(rebuilt, recorded, ("id",), set(checker.PRIMARY_FIELDS) - text - counts,
                                  counts, "contrasts")


if __name__ == "__main__":
    unittest.main()
