"""Focused checks for preprocessing isolation, leakage, and real KAHM replay."""
import json
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from certified_forecasting.features import NestedSoftFeatures
import scaling_features as sf


class RecordingBlock:
    """Observe exactly what the inherited fitter supplies to KAHM geometry."""
    def __init__(self, n_clusters, *args):
        self.n_features_out_ = n_clusters

    def fit(self, X):
        self.training_inputs_ = X.copy()
        return self

    def transform(self, X):
        scores = X[:, :1] * np.arange(self.n_features_out_)[None, :]
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        return weights / weights.sum(axis=1, keepdims=True)


class ScalingTests(unittest.TestCase):
    def setUp(self):
        sf.reset_fitted_banks()

    def test_standard_is_exact_original_class(self):
        self.assertIs(sf.make_feature_class("standard"), NestedSoftFeatures)
        with self.assertRaises(ValueError):
            sf.make_feature_class("undefined")

    def test_non_kahm_baselines_are_exactly_unchanged(self):
        rng = np.random.default_rng(100)
        X = rng.normal(size=(60, 3)) * [1.0, 2.0, 0.1] + [7.0, 0.0, 2.0]
        query = rng.normal(size=(13, 3))
        for kind in ("rbf", "affine"):
            kwargs = dict(cluster_counts=(2, 4), kind=kind, seed=29, max_fit=30)
            original = NestedSoftFeatures(**kwargs).fit(X)
            for mode in ("raw", "minmax", "minmax_clip"):
                alternative = sf.make_feature_class(mode)(**kwargs).fit(X)
                np.testing.assert_array_equal(alternative.mean_, original.mean_)
                np.testing.assert_array_equal(alternative.scale_, original.scale_)
                np.testing.assert_array_equal(alternative.fit_indices_, original.fit_indices_)
                np.testing.assert_array_equal(alternative.transform(query), original.transform(query))
                self.assertEqual(alternative.scaling_metadata()["effective_mode"], "standard")

    def test_extrema_use_all_training_rows_before_subsampling(self):
        rng = np.random.default_rng(192)
        X = rng.normal(size=(50, 3))
        X[:, 2] = 7.0
        chosen = np.sort(np.random.default_rng(5).choice(50, 8, replace=False))
        omitted = np.setdiff1d(np.arange(50), chosen)
        X[omitted[0], 0] = -99.0
        X[omitted[1], 1] = 111.0
        with patch("certified_forecasting.features._KAHMBlock", RecordingBlock):
            bank = sf.MinMaxSoftFeatures((2,), seed=5, max_fit=8).fit(X)
        np.testing.assert_array_equal(bank.fit_indices_, chosen)
        self.assertEqual(bank.raw_train_min_[0], -99.0)
        self.assertEqual(bank.raw_train_max_[1], 111.0)
        Z = bank.preprocess_inputs(X)
        np.testing.assert_array_equal(Z[:, 2], 0.0)
        np.testing.assert_array_equal(Z[:, :2].min(axis=0), -1.0)
        np.testing.assert_array_equal(Z[:, :2].max(axis=0), 1.0)
        np.testing.assert_array_equal(bank.blocks_[0].training_inputs_, Z[chosen])
        np.testing.assert_array_equal(bank.mean_, 0.0)
        np.testing.assert_array_equal(bank.scale_, 1.0)
        self.assertIs(sf.FITTED_BANKS[-1], bank)

    def test_pure_affine_extrapolation_and_clipped_diagnostic_are_distinct(self):
        X = np.array([[-2., 3., 7.], [0., 5., 7.], [2., 7., 7.], [1., 4., 7.]])
        queries = np.array([[-4., 9., 999.], [0., 5., 7.]])
        with patch("certified_forecasting.features._KAHMBlock", RecordingBlock):
            plain = sf.MinMaxSoftFeatures((2,)).fit(X)
            clipped = sf.ClippedMinMaxSoftFeatures((2,)).fit(X)
        np.testing.assert_array_equal(plain.preprocess_inputs(queries), [[-2., 2., 0.], [0., 0., 0.]])
        np.testing.assert_array_equal(clipped.preprocess_inputs(queries), [[-1., 1., 0.], [0., 0., 0.]])
        with sf.scaling_phase("test"):
            clipped.transform(queries)
        metadata = clipped.scaling_metadata()
        rec = metadata["diagnostics"]["test"]
        self.assertEqual(rec["clipped_coordinates"], 2)
        self.assertEqual(rec["outside_training_extrema_coordinates"], 3)
        self.assertEqual(rec["evaluated_rows_including_repeats"], 2)
        json.dumps(metadata, allow_nan=False)

    def test_held_out_values_do_not_refit_or_change_other_queries(self):
        X = np.array([[-2., 3.], [0., 5.], [2., 7.], [1., 4.]])
        with patch("certified_forecasting.features._KAHMBlock", RecordingBlock):
            bank = sf.MinMaxSoftFeatures((2,)).fit(X)
        before_min = bank.raw_train_min_.copy()
        before_max = bank.raw_train_max_.copy()
        first = bank.transform([[1., 6.]])
        combined = bank.transform([[1., 6.], [-1000., 1000.]])
        np.testing.assert_array_equal(first[0], combined[0])
        np.testing.assert_array_equal(bank.raw_train_min_, before_min)
        np.testing.assert_array_equal(bank.raw_train_max_, before_max)

    def test_raw_mode_matches_original_unstandardized_real_kahm(self):
        X = np.random.default_rng(201).uniform(-1., 1., (45, 3))
        kwargs = dict(cluster_counts=(2,), seed=3, max_fit=40, max_train_per_cluster=10)
        inherited = NestedSoftFeatures(**kwargs, standardize=False).fit(X)
        raw = sf.RawSoftFeatures(**kwargs).fit(X)
        np.testing.assert_array_equal(raw.transform(X[:12]), inherited.transform(X[:12]))
        np.testing.assert_array_equal(raw.fit_indices_, inherited.fit_indices_)

    def test_scaled_real_kahm_matches_manual_preprocessing_and_pickle(self):
        X = np.random.default_rng(204).uniform(-0.8, 0.6, (45, 3))
        kwargs = dict(cluster_counts=(2,), seed=3, max_fit=40, max_train_per_cluster=10)
        scaled = sf.MinMaxSoftFeatures(**kwargs).fit(X)
        manual_X = 2.0 * ((X - X.min(axis=0)) / np.ptp(X, axis=0)) - 1.0
        inherited = NestedSoftFeatures(**kwargs, standardize=False).fit(manual_X)
        expected = inherited.transform(manual_X[:12])
        actual = scaled.transform(X[:12])
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual.sum(axis=1), 1., atol=1e-12)
        restored = pickle.loads(pickle.dumps(scaled))
        np.testing.assert_array_equal(restored.transform(X[:12]), actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
