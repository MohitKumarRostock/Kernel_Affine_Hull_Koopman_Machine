"""Checks for causal construction and group-level speech-risk evaluation."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import brentq

SOURCE = Path(__file__).resolve().parents[1] / "experiments/speech_forecasting/run_study.py"
spec = importlib.util.spec_from_file_location("speech_forecasting_run", SOURCE)
speech = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = speech
spec.loader.exec_module(speech)


class SpeechForecastingTests(unittest.TestCase):
    def test_energy_range_floor_and_causality(self):
        silence = np.zeros(16000, dtype=np.int16)
        assert np.array_equal(speech.energy_observable(silence), np.zeros(100))
        full = np.full(16000, -32768, dtype=np.int16)
        assert np.array_equal(speech.energy_observable(full), np.ones(100))
        rng = np.random.default_rng(71)
        original = rng.integers(-32768, 32768, 16000, dtype=np.int16)
        changed = original.copy()
        changed[8000:] = 0
        assert np.array_equal(speech.energy_observable(original)[:50], speech.energy_observable(changed)[:50])


    def test_history_origins_and_horizons_are_causal(self):
        energy = np.arange(200).reshape(2, 100)
        for m in (4, 12):
            x, y = speech.histories(energy, m)
            assert x.shape == (40, m)
            for clip in range(2):
                for j, origin in enumerate(speech.ORIGINS):
                    np.testing.assert_array_equal(x[clip*20+j], energy[clip, origin-m+1:origin+1])
                    np.testing.assert_array_equal(y[clip*20+j], energy[clip, origin+speech.HORIZONS])


    def test_speakers_receive_equal_weight_despite_unequal_clip_counts(self):
        prediction = np.zeros((60, 3))
        prediction[40:] = 1
        target = np.zeros_like(prediction)
        groups, losses = speech.group_losses(prediction, target, np.array(["a", "a", "b"]))
        np.testing.assert_array_equal(groups, ["a", "b"])
        np.testing.assert_array_equal(losses, [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(losses.mean(axis=0), [.5, .5, .5])


    def test_kl_upper_matches_independent_root_and_boundary_formula(self):
        n, alpha = 978, .05/72
        boundary = 1 - alpha**(1/n)
        assert abs(speech.kl_upper(0., n, alpha) - boundary) < 2e-15
        assert speech.kl_upper(1., n, alpha) == 1.
        for mean in (.001, .1, .5, .95):
            def residual(probability):
                return mean*np.log(mean/probability)+(1-mean)*np.log((1-mean)/(1-probability))-np.log(1/alpha)/n
            reference = brentq(residual, mean+1e-14, 1-1e-14, xtol=5e-15)
            assert abs(speech.kl_upper(mean, n, alpha)-reference) < 1e-13


if __name__ == "__main__":
    unittest.main()
