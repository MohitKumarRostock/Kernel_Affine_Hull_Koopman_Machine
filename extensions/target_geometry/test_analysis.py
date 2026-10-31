"""Consequential auditor integration checks on fabricated data, never study outcomes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from . import analysis, experiment
from .test_protocol import toy_arrays, toy_config, toy_task


class TestPredictionAudit(unittest.TestCase):
    def run_audit(self, phase="synthetic", donor_failure=False):
        cfg, task = toy_config(), toy_task(phase)
        cfg["synthetic"]["horizons"] = [1, 2]
        if phase == "natural":
            task["seasonal_period"] = 7
            rng = np.random.default_rng(188)
            values = np.sin(np.arange(140) / 5) + rng.normal(scale=0.2, size=140)
            arrays = {"values": values}
            for split, origins in (("train", np.arange(15, 69)), ("validation", np.arange(80, 92)),
                                   ("test", np.arange(95, 107))):
                arrays["origins_" + split] = origins
                arrays["X_" + split] = np.asarray([values[t - 3:t + 1] for t in origins])
                arrays["Y_" + split] = np.asarray([values[t + 1:t + 3] for t in origins])
        else:
            arrays = toy_arrays()
        with tempfile.TemporaryDirectory(prefix="kahm-audit-fabricated-") as temp:
            directory = Path(temp) / "task"
            with threadpool_limits(limits=1):
                if donor_failure:
                    with patch.object(experiment, "oof_predictions", side_effect=ValueError("fabricated donor failure")):
                        experiment.run_task(task, arrays, cfg, directory)
                else:
                    experiment.run_task(task, arrays, cfg, directory)
            with patch.object(analysis, "reconstruct_inputs", return_value=arrays):
                rows, horizons, identities, trials, diagnostic = analysis.audit_task(directory, task, cfg)
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(identities), 12)
            self.assertEqual(len(horizons), 32)
            if donor_failure:
                self.assertEqual(sum(row["status"] == "fallback" for row in rows), 8)
                self.assertIsNone(diagnostic["donor_final_disagreement_mse"])
            # Changing even one archived prediction byte is detected before loss interpretation.
            archive = directory / "predictions.npz"
            with archive.open("ab") as stream:
                stream.write(b"altered")
            with patch.object(analysis, "reconstruct_inputs", return_value=arrays):
                with self.assertRaisesRegex(ValueError, "archive hash"):
                    analysis.audit_task(directory, task, cfg)

    def test_synthetic_full_schema_and_metric_audit(self):
        self.run_audit()

    def test_natural_positions_and_label_availability_audit(self):
        self.run_audit("natural")

    def test_donor_failure_is_retained_and_charged(self):
        self.run_audit(donor_failure=True)

    def test_bonferroni_interval_uses_repetition_sd(self):
        cfg = toy_config()
        values = np.arange(12, dtype=float) / 1000
        report = analysis.paired_interval(values, cfg)
        self.assertEqual(report["n"], 12)
        self.assertEqual(report["family_size"], 18)
        self.assertGreater(report["critical_t"], 3)
        self.assertAlmostEqual(report["upper"] - report["mean_difference"],
                               report["critical_t"] * np.std(values, ddof=1) / np.sqrt(12))


if __name__ == "__main__":
    unittest.main()
