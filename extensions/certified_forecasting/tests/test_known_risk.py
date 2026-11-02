"""Disjoint-pair sampling and sample-size checks for the exact-risk benchmark."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from certified_forecasting.certificates import mean_interval


RUNNER_PATH = Path(__file__).resolve().parents[1] / "experiments" / "run_known_risk.py"
SPEC = importlib.util.spec_from_file_location("known_risk_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class KnownRiskPairingTests(unittest.TestCase):
    def test_three_states_cannot_supply_two_independent_pairs(self):
        # Previously lengths 2 and 1 broadcast, reusing the middle observation.
        for n in range(4):
            with self.subTest(n=n), self.assertRaises(ValueError):
                RUNNER.two_atom_pair_scores(np.arange(n) % 2, 1.0)

    def test_five_states_discard_only_the_final_state(self):
        scores = RUNNER.two_atom_pair_scores([0, 1, 1, 1, 0], 1.5)
        np.testing.assert_array_equal(scores, [1.125, 0.0])
        altered = RUNNER.two_atom_pair_scores([0, 1, 1, 1, 1], 1.5)
        np.testing.assert_array_equal(scores, altered)
        endpoint = mean_interval(scores, 2.0, 0.05)
        self.assertEqual(endpoint["n_independent"], 2)
        logterm = np.log(2.0 / 0.05)
        expected_radius = np.sqrt(2.0 * np.var(scores, ddof=1) * logterm / 2)
        expected_radius += 7.0 * 2.0 * logterm / 3.0
        self.assertAlmostEqual(endpoint["radius"], expected_radius)

    def test_even_sample_scores_match_retained_calculation_exactly(self):
        rng = np.random.default_rng(154)
        for n in (4, 64, 256, 1024, 4096):
            states = rng.binomial(1, 0.2, n)
            for deficit in (0.0, 0.3, 2.0):
                with self.subTest(n=n, deficit=deficit):
                    historical = (states[::2] != states[1::2]).astype(float) * deficit**2 / 2.0
                    np.testing.assert_array_equal(
                        RUNNER.two_atom_pair_scores(states, deficit), historical)

    def test_runner_uses_two_pair_scores_and_five_loss_scores(self):
        config = {
            "probabilities": [0.2], "softness": [0.6], "kappa": [1.0],
            "calibration_sizes": [5], "n_training": 8, "replicates": 1,
            "seed": 23, "delta": 0.10,
        }
        observed = []

        def record_interval(values, bound, delta):
            result = mean_interval(values, bound, delta)
            observed.append((bound, result["n_independent"]))
            return result

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            with mock.patch.object(RUNNER, "mean_interval", side_effect=record_interval):
                with contextlib.redirect_stdout(io.StringIO()):
                    RUNNER.run(config_path, Path(directory) / "result")
            summary = json.loads((Path(directory) / "result" / "summary.json").read_text())
        self.assertEqual(observed, [(2.0, 2), (4.0, 5)])
        self.assertIn("at least 90%", summary["confidence_scope"])
        self.assertNotIn("95%", summary["confidence_scope"])

    def test_invalid_runner_size_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({"calibration_sizes": [3]}))
            output = Path(directory) / "result"
            with self.assertRaisesRegex(ValueError, "at least four"):
                RUNNER.run(config_path, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
