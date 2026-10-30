"""Synthetic information-set and library-equivalence checks for baselines."""
from __future__ import annotations

import json
import unittest
import warnings

import numpy as np
from statsmodels.tsa.holtwinters import Holt

from extensions.natural_forecasting.baselines import (
    HoltBaseline, SeasonalNaive, TabularBaseline,
)


def _fit_holt(values, *, damped, initial=None, smoothing=None):
    kwargs = {"initialization_method": "estimated"}
    if initial is not None:
        kwargs = {"initialization_method": "known", "initial_level": initial[0],
                  "initial_trend": initial[1]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Holt(np.asarray(values, float), damped_trend=damped, **kwargs)
        return model.fit(optimized=True) if smoothing is None else model.fit(
            smoothing_level=smoothing[0], smoothing_trend=smoothing[1],
            damping_trend=smoothing[2] if damped else None, optimized=False)


class HoltBaselineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(18273)
        time = np.arange(110)
        self.values = 50 + .17 * time + 1.5 * np.sin(time / 4.5) + rng.normal(0, .4, len(time))
        self.cutoff = 52

    def test_training_cutoff_forecast_matches_complete_statsmodels_fit(self):
        for damped in (False, True):
            with self.subTest(damped=damped):
                model = HoltBaseline(damped).fit_history(self.values, self.cutoff)
                past = self.values[:self.cutoff]
                mean, scale = past.mean(), past.std()
                reference = _fit_holt((past - mean) / scale, damped=damped)
                expected = mean + scale * np.asarray(reference.forecast(13))
                np.testing.assert_allclose(model.predict_history(self.values, self.cutoff, 13)[0],
                                           expected, rtol=1e-10, atol=1e-10)
                self.assertEqual(model.metadata()["fit_cutoff"], self.cutoff)
                json.dumps(model.metadata(), allow_nan=False)

    def test_single_multiple_unsorted_and_duplicate_origins_match_fixed_library_replay(self):
        # Interior parameters exercise both trend and level updates even if an
        # optimizer would otherwise choose a boundary smoothing value.
        for damped in (False, True):
            with self.subTest(damped=damped):
                model = HoltBaseline(damped).fit_history(self.values, self.cutoff)
                standardized = (self.values - model.mean_) / model.scale_
                initial = (-.4, .12)
                smoothing = (.37, .21, .91 if damped else 1.0)
                reference_training = _fit_holt(standardized[:self.cutoff], damped=damped,
                                               initial=initial, smoothing=smoothing)
                model.alpha_, model.beta_, model.phi_ = smoothing
                model.level_ = float(reference_training.level[-1])
                model.trend_ = float(reference_training.trend[-1])
                origins = np.asarray([82, self.cutoff, 63, 82, 71])
                expected = []
                for origin in origins:
                    replay = _fit_holt(standardized[:origin], damped=damped,
                                       initial=initial, smoothing=smoothing)
                    expected.append(model.mean_ + model.scale_ * np.asarray(replay.forecast(9)))
                actual = model.predict_history(self.values, origins, 9)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(model.predict_history(self.values, 63, 9)[0], expected[2],
                                           rtol=1e-10, atol=1e-10)

    def test_future_observations_never_change_an_earlier_forecast_or_fit(self):
        model = HoltBaseline().fit_history(self.values, self.cutoff)
        origin = self.cutoff + 7
        expected = model.predict_history(self.values, [origin, 85], 8)
        changed = self.values.copy()
        changed[origin:] = -1e6
        actual = model.predict_history(changed, [85, origin], 8)
        np.testing.assert_array_equal(actual[1], expected[0])
        # Replaying a later origin must not mutate the stored training state.
        np.testing.assert_array_equal(model.predict_history(self.values, origin, 8)[0], expected[0])
        changed_training_future = self.values.copy()
        changed_training_future[self.cutoff:] = 1e6
        refit = HoltBaseline().fit_history(changed_training_future, self.cutoff)
        np.testing.assert_array_equal(
            refit.predict_history(changed_training_future, self.cutoff, 8),
            model.predict_history(self.values, self.cutoff, 8))

    def test_missing_training_is_causal_and_has_an_explicit_audit_trail(self):
        values = self.values.copy()
        values[:6] = np.nan
        values[9] = np.nan
        values[15] = np.nan
        model = HoltBaseline().fit_history(values, cutoff=22, max_history=18)
        metadata = model.metadata()
        self.assertEqual(metadata["fit_start_index"], 6)
        self.assertEqual(metadata["discarded_leading_missing"], 2)
        self.assertEqual(metadata["forward_filled_training_values"], 2)
        self.assertEqual(metadata["finite_training_observations"], 14)
        self.assertEqual(metadata["training_observations"], 16)
        cleaned = values[6:22].copy()
        cleaned[3], cleaned[9] = values[8], values[14]
        np.testing.assert_allclose([model.mean_, model.scale_], [cleaned.mean(), cleaned.std()])
        reference = _fit_holt((cleaned - cleaned.mean()) / cleaned.std(), damped=True)
        expected = cleaned.mean() + cleaned.std() * reference.forecast(5)
        np.testing.assert_allclose(model.predict_history(values, 22, 5)[0], expected,
                                   rtol=1e-10, atol=1e-10)

    def test_missing_online_values_propagate_the_state_without_a_measurement(self):
        model = HoltBaseline().fit_history(self.values, self.cutoff)
        values = self.values.copy()
        values[self.cutoff:self.cutoff + 3] = np.nan
        uninterrupted_forecast = model.predict_history(values, self.cutoff, 10)[0]
        after_missing = model.predict_history(values, self.cutoff + 3, 7)[0]
        np.testing.assert_allclose(after_missing, uninterrupted_forecast[3:], rtol=1e-11, atol=1e-11)
        self.assertIn("without measurement", model.metadata()["missing_online"])

    def test_invalid_history_indices_fail_explicitly(self):
        with self.assertRaises(RuntimeError):
            HoltBaseline().predict_history(self.values, 10, 3)
        for cutoff in (-1, 0, 2.5, len(self.values) + 1):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                HoltBaseline().fit_history(self.values, cutoff)
        for maximum in (0, -5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                HoltBaseline().fit_history(self.values, self.cutoff, max_history=maximum)
        model = HoltBaseline().fit_history(self.values, self.cutoff)
        for origin in (self.cutoff - 1, len(self.values) + 1, 52.5):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                model.predict_history(self.values, origin, 3)
        self.assertEqual(model.predict_history(self.values, [], 3).shape, (0, 3))


class SeasonalNaiveTests(unittest.TestCase):
    def test_last_observed_season_repeats_for_long_horizons(self):
        values = np.arange(20, dtype=float)
        model = SeasonalNaive(4).fit_history(values, cutoff=8)
        expected = [[6, 7, 8, 9, 6, 7, 8, 9, 6], [3, 4, 5, 6, 3, 4, 5, 6, 3]]
        np.testing.assert_array_equal(model.predict_history(values, [10, 7], 9), expected)
        np.testing.assert_array_equal(model.predict_history(values, 10, 9)[0], expected[0])
        changed = values.copy()
        changed[10:] = -999
        np.testing.assert_array_equal(model.predict_history(changed, 10, 9)[0], expected[0])

    def test_last_observed_data_origin_requires_plus_one_adapter(self):
        # data.py origin=9 means observed indices <=9, so the first future is10.
        values = np.arange(20, dtype=float)
        data_origin = 9
        prediction = SeasonalNaive(4).predict_history(values, data_origin + 1, 4)
        np.testing.assert_array_equal(prediction[0], values[6:10])
        self.assertNotEqual(prediction[0, -1], values[data_origin - 1])

    def test_missing_lag_and_short_history_fallbacks_are_explicit(self):
        values = np.arange(15, dtype=float)
        values[5], values[7] = np.nan, np.nan
        model = SeasonalNaive(4).fit_history(values, cutoff=8, max_history=3)
        np.testing.assert_array_equal(model.predict_history(values, 8, 6)[0], [4, 6, 6, 6, 4, 6])
        short = np.asarray([np.nan, 4.0, np.nan, 8.0])
        np.testing.assert_array_equal(SeasonalNaive(7).predict_history(short, 3, 5)[0], np.full(5, 4.0))
        metadata = model.metadata()
        self.assertIn("does not truncate", metadata["history"])
        self.assertEqual(metadata["requested_max_history"], 3)
        json.dumps(metadata, allow_nan=False)
        with self.assertRaises(ValueError):
            SeasonalNaive(4).predict_history([np.nan, np.nan, 3.0], 2, 3)

    def test_invalid_indices_and_periods_are_rejected(self):
        for period in (0, -1, 2.5):
            with self.subTest(period=period), self.assertRaises(ValueError):
                SeasonalNaive(period)
        model = SeasonalNaive(2)
        for origin in (-1, 0, 4, 1.5):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                model.predict_history([1, 2, 3], origin, 2)
        with self.assertRaises(ValueError):
            model.predict_history([1, 2, 3], 3, 0)
        self.assertEqual(model.predict_history([1, 2, 3], [], 4).shape, (0, 4))


class TabularBaselineTests(unittest.TestCase):
    def test_scaling_is_training_only_and_constant_target_is_preserved(self):
        rng = np.random.default_rng(633)
        x = 30 + 4 * rng.normal(size=(48, 5))
        y = np.column_stack((10 * x[:, 0] - x[:, 1], np.full(len(x), -8.0)))
        for kind in ("hist_gradient", "mlp"):
            with self.subTest(kind=kind):
                model = TabularBaseline(kind, width=4, iterations=8).fit(x, y)
                mean_x, scale_x = model.x_scaler_.mean_.copy(), model.x_scaler_.scale_.copy()
                mean_y, scale_y = model.y_scaler_.mean_.copy(), model.y_scaler_.scale_.copy()
                prediction = model.predict(x[:7] + 100)
                self.assertEqual(prediction.shape, (7, 2))
                self.assertTrue(np.isfinite(prediction).all())
                np.testing.assert_allclose(mean_x, x.mean(axis=0))
                np.testing.assert_allclose(mean_y, y.mean(axis=0))
                np.testing.assert_array_equal(model.x_scaler_.mean_, mean_x)
                np.testing.assert_array_equal(model.x_scaler_.scale_, scale_x)
                np.testing.assert_array_equal(model.y_scaler_.mean_, mean_y)
                np.testing.assert_array_equal(model.y_scaler_.scale_, scale_y)
                if kind == "hist_gradient":
                    np.testing.assert_allclose(prediction[:, 1], -8.0)
                json.dumps(model.metadata(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
