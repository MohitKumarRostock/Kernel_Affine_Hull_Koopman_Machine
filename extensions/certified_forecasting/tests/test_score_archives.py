"""A failed score write must never masquerade as a completed NPZ archive."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
from run_forecasting import write_npz_atomic


class ScoreArchiveTests(unittest.TestCase):
    def test_complete_archive_roundtrip_and_no_overwrite(self):
        arrays = {"losses": np.arange(24, dtype=float).reshape(8, 3)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_scores.npz"
            write_npz_atomic(path, arrays)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["losses"], arrays["losses"])
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_npz_atomic(path, {"losses": np.zeros((8, 3))})
            self.assertEqual(path.read_bytes(), before)

    def test_partial_write_remains_temporary_and_preserves_failure(self):
        def interrupted(stream, **arrays):
            stream.write(b"incomplete ZIP data")
            raise OSError("simulated interrupted write")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_scores.npz"
            with patch("run_forecasting.np.savez_compressed", side_effect=interrupted):
                with self.assertRaisesRegex(OSError, "simulated interrupted"):
                    write_npz_atomic(path, {"losses": np.zeros((8, 3))})
            self.assertFalse(path.exists())
            self.assertEqual(path.with_name(path.name + ".tmp").read_bytes(), b"incomplete ZIP data")


if __name__ == "__main__":
    unittest.main()
