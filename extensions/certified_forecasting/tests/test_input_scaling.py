"""Frozen input-map isolation, real KAHM equivalence, and saved-model checks."""
import json
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch

import numpy as np

from certified_forecasting.data import make_episodes
from certified_forecasting.features import NestedSoftFeatures


class RecordingBlock:
    """Observe fitter inputs without coupling edge-case checks to a KAHM fit."""
    def __init__(self, n_clusters, *args):
        self.n_features_out_ = n_clusters

    def fit(self, X):
        self.training_inputs_ = X.copy()
        return self

    def transform(self, X):
        self.last_inputs_ = X.copy()
        return np.full((len(X), self.n_features_out_), 1.0 / self.n_features_out_)


class InputScalingTests(unittest.TestCase):
    def test_training_extrema_precede_geometry_subsampling_and_remain_frozen(self):
        X = np.random.default_rng(301).normal(size=(40, 3))
        X[:, 2] = 7.0
        chosen = np.sort(np.random.default_rng(9).choice(40, 8, replace=False))
        omitted = np.setdiff1d(np.arange(40), chosen)
        X[omitted[0], 0], X[omitted[1], 1] = -99.0, 111.0
        with patch("certified_forecasting.features._KAHMBlock", RecordingBlock):
            bank = NestedSoftFeatures((2,), seed=9, max_fit=8, input_scaling="minmax").fit(X)
        expected = 2.0 * ((X[:, :2] - X[:, :2].min(axis=0)) / np.ptp(X[:, :2], axis=0)) - 1.0
        expected = np.column_stack([expected, np.zeros(len(X))])
        np.testing.assert_array_equal(bank.fit_indices_, chosen)
        np.testing.assert_array_equal(bank.blocks_[0].training_inputs_, expected[chosen])
        frozen = bank.input_scaling_metadata()
        bank.transform(np.array([[1e6, -1e6, -999.0]]))
        self.assertEqual(bank.input_scaling_metadata(), frozen)
        self.assertEqual(bank.blocks_[0].last_inputs_[0, 2], 0.0)
        self.assertGreater(bank.blocks_[0].last_inputs_[0, 0], 1.0)
        self.assertLess(bank.blocks_[0].last_inputs_[0, 1], -1.0)
        json.dumps(frozen, allow_nan=False)

    def test_clipping_is_explicit_and_does_not_update_training_parameters(self):
        X = np.array([[-2., 3., 7.], [0., 5., 7.], [2., 7., 7.], [1., 4., 7.]])
        query = np.array([[-4., 9., 999.], [0., 5., 7.]])
        with patch("certified_forecasting.features._KAHMBlock", RecordingBlock):
            raw = NestedSoftFeatures((2,), input_scaling="minmax").fit(X)
            clipped = NestedSoftFeatures((2,), input_scaling="minmax_clip").fit(X)
        before = clipped.input_scaling_metadata()
        raw.transform(query)
        clipped.transform(query)
        np.testing.assert_array_equal(raw.blocks_[0].last_inputs_, [[-2., 2., 0.], [0., 0., 0.]])
        np.testing.assert_array_equal(clipped.blocks_[0].last_inputs_, [[-1., 1., 0.], [0., 0., 0.]])
        self.assertEqual(clipped.input_scaling_metadata(), before)

    def test_default_standardization_matches_manual_real_kahm(self):
        X = np.random.default_rng(302).normal(size=(55, 3)) * [0.5, 2., 0.2] + [2., -1., 4.]
        query = X[:9] + [0.2, 0.3, -0.1]
        kwargs = dict(cluster_counts=(2, 4), seed=12, max_fit=45, max_train_per_cluster=10)
        fitted = NestedSoftFeatures(**kwargs).fit(X)
        center, scale = X.mean(axis=0), X.std(axis=0)
        manual = NestedSoftFeatures(**kwargs, standardize=False).fit((X-center)/scale)
        explicit = NestedSoftFeatures(**kwargs, standardize=False, input_scaling="standard").fit(X)
        expected = manual.transform((query-center)/scale)
        np.testing.assert_array_equal(fitted.transform(query), expected)
        np.testing.assert_array_equal(explicit.transform(query), expected)
        np.testing.assert_array_equal(fitted.fit_indices_, manual.fit_indices_)

    def test_real_kahm_modes_match_manual_preprocessing_and_pickle(self):
        X = np.random.default_rng(303).uniform(-0.7, 0.6, size=(55, 3))
        query = np.vstack([X[:8], [-1.5, 1.2, 0.1]])
        kwargs = dict(cluster_counts=(2, 4), seed=13, max_fit=45, max_train_per_cluster=10)
        lo, span = X.min(axis=0), np.ptp(X, axis=0)
        for mode in ("raw", "minmax", "minmax_clip"):
            with self.subTest(mode=mode):
                train_scaled = X if mode == "raw" else 2.0 * ((X-lo)/span) - 1.0
                query_scaled = query if mode == "raw" else 2.0 * ((query-lo)/span) - 1.0
                if mode == "minmax_clip":
                    query_scaled = np.clip(query_scaled, -1., 1.)
                manual = NestedSoftFeatures(**kwargs, standardize=False).fit(train_scaled)
                fitted = NestedSoftFeatures(**kwargs, input_scaling=mode).fit(X)
                for stage in (1, 2):
                    np.testing.assert_array_equal(fitted.transform(query, stage), manual.transform(query_scaled, stage))
                restored = pickle.loads(pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL))
                np.testing.assert_array_equal(restored.transform(query), fitted.transform(query))
                self.assertEqual(restored.input_scaling_metadata(), fitted.input_scaling_metadata())

    def test_legacy_saved_banks_remain_predictable(self):
        X = np.random.default_rng(304).normal(size=(40, 3))
        for standardize in (True, False):
            bank = NestedSoftFeatures((2,), seed=3, standardize=standardize,
                                      max_train_per_cluster=10).fit(X)
            expected = bank.transform(X[:7])
            del bank.input_scaling
            del bank.input_scaling_
            restored = pickle.loads(pickle.dumps(bank))
            np.testing.assert_array_equal(restored.transform(X[:7]), expected)

    def test_invalid_input_scaling_is_rejected(self):
        with self.assertRaises(ValueError):
            NestedSoftFeatures((2,), input_scaling="invalid").fit(np.ones((10, 2)))
        with self.assertRaises(ValueError):
            NestedSoftFeatures(input_scaling="minmax").input_scaling_metadata()

    def test_nonfinite_and_zero_dimension_inputs_are_rejected(self):
        for mode in ("standard", "raw", "minmax", "minmax_clip"):
            for X in (np.empty((5, 0)), np.full((5, 2), np.nan),
                      np.full((5, 2), np.inf), np.full((5, 2), -np.inf)):
                with self.subTest(mode=mode, shape=X.shape), self.assertRaises(ValueError):
                    NestedSoftFeatures((2,), input_scaling=mode).fit(X)
            with patch("certified_forecasting.features._KAHMBlock", RecordingBlock):
                bank = NestedSoftFeatures((2,), input_scaling=mode).fit(
                    np.arange(12, dtype=float).reshape(6, 2))
            frozen = bank.input_scaling_metadata()
            for value in (np.nan, np.inf, -np.inf):
                with self.assertRaises(ValueError):
                    bank.transform([[value, 0.]])
                self.assertEqual(bank.input_scaling_metadata(), frozen)

    def test_finite_inputs_that_overflow_preprocessing_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "parameters must be finite"):
            NestedSoftFeatures((2,), input_scaling="standard").fit(
                np.full((5, 2), 1e308))
        X = np.array([[-1e-10, 0.], [1e-10, 1.], [-1e-10, 0.], [1e-10, 1.]])
        with patch("certified_forecasting.features._KAHMBlock", RecordingBlock):
            bank = NestedSoftFeatures((2,), input_scaling="standard").fit(X)
        with self.assertRaisesRegex(ValueError, "Scaled inputs must be finite"):
            bank.transform([[1e308, 0.]])

    def test_config_intervention_only_affects_kahm_and_opt_in_metadata(self):
        experiments = Path(__file__).resolve().parents[1] / "experiments"
        if str(experiments) not in sys.path:
            sys.path.insert(0, str(experiments))
        from run_forecasting import build_candidates, candidate_description
        cfg = {
            "histories": [4], "horizons": [1, 2], "origin_stride": 2,
            "feature_kinds": ["kahm", "rbf", "affine"], "cluster_counts": [2, 4],
            "omega": 16., "max_geometry_inputs": 60, "max_train_per_cluster": 10,
            "alpha": 1e-4, "kappa": 4.,
        }
        train = make_episodes("harmonic", 16, 24, 310)
        validation = make_episodes("harmonic", 20, 24, 311)
        inherited, _ = build_candidates(train, validation, cfg, 20)
        explicit, _ = build_candidates(train, validation, {**cfg, "kahm_input_scaling": "standard"}, 20)
        minmax, _ = build_candidates(train, validation, {**cfg, "kahm_input_scaling": "minmax"}, 20)
        for baseline, standard, scaled in zip(inherited, explicit, minmax):
            self.assertEqual(baseline.name, standard.name)
            self.assertEqual(baseline.name, scaled.name)
            np.testing.assert_array_equal(baseline.validation_mse, standard.validation_mse)
            self.assertNotIn("input_scaling", candidate_description(baseline))
            if baseline.family.startswith("kahm"):
                self.assertEqual(candidate_description(standard)["input_scaling"]["mode"], "standard")
                self.assertEqual(candidate_description(scaled)["input_scaling"]["mode"], "minmax")
            else:
                np.testing.assert_array_equal(baseline.validation_mse, scaled.validation_mse)
                self.assertNotIn("input_scaling", candidate_description(standard))
                self.assertNotIn("input_scaling", candidate_description(scaled))


if __name__ == "__main__":
    unittest.main()
