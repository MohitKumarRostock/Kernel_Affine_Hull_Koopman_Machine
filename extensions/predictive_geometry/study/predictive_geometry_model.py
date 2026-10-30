"""Reusable implementation of the eight-candidate validation selector in stage2.

This wrapper is a post-study implementation of the already evaluated recipe.
It does not choose a geometry from test outcomes. Its predictions are checked
against the independently executed frozen runner by replay_capacity_selector.py.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np

from geometry import GeometryBank
from heads import fit_predict, ALPHAS, BANDWIDTH_MULTIPLIERS


class PredictiveGeometryRegressor:
    """Select raw, affine-local, KAHM-local, or KAHM-score regression on validation.

    Call fit with disjoint training and validation observations. For grouped
    time series, provide validation_groups identifying independent episodes.
    Several forecast horizons share one selected pipeline. Bounds, when known
    from the data law, clip predictions; use None for unbounded applications.
    No predictions are guaranteed to improve over a baseline on a new task.
    """

    def __init__(self, seed=0, head_seed=1, bounds=None, geometry_parameters=None,
                 landmark_counts=(64, 256),
                 features=("raw", "affine:local", "kahm:local", "kahm:scores"),
                 alphas=ALPHAS, bandwidth_multipliers=BANDWIDTH_MULTIPLIERS):
        self.seed, self.head_seed, self.bounds = int(seed), int(head_seed), bounds
        self.geometry_parameters = dict(geometry_parameters or {
            "cluster_counts": (2, 4, 8), "rank": 2, "max_refs": 24,
            "max_fit": 1600, "scaling": "standard", "omega": 16., "tau": 1e-6})
        self.landmark_counts = tuple(landmark_counts)
        self.features = tuple(features)
        self.alphas = tuple(alphas)
        self.bandwidth_multipliers = tuple(bandwidth_multipliers)

    def fit(self, X_train, Y_train, X_validation, Y_validation, validation_groups=None):
        train_y, val_y = np.asarray(Y_train, float), np.asarray(Y_validation, float)
        if train_y.ndim == 1:
            train_y = train_y[:, None]
        if val_y.ndim == 1:
            val_y = val_y[:, None]
        self.bank_ = GeometryBank(seed=self.seed, **self.geometry_parameters).fit(X_train)
        train = self.bank_.transform(X_train)
        validation = self.bank_.transform(X_validation)
        self.candidates_ = []
        best = float("inf")
        for landmarks in self.landmark_counts:
            for feature in self.features:
                _, selected, model = fit_predict(
                    train[feature], train_y, validation[feature], val_y, validation[feature],
                    head="rbf", seed=self.head_seed, bounds=self.bounds,
                    validation_groups=validation_groups, n_landmarks=landmarks,
                    alphas=self.alphas, bandwidth_multipliers=self.bandwidth_multipliers)
                risk = float(np.mean(selected[0]["validation_mse_by_target"]))
                summary = {"feature": feature, "landmarks": landmarks,
                           "validation_mse": risk, "parameters": selected}
                self.candidates_.append(summary)
                if risk < best:
                    best = risk
                    self.selected_ = summary
                    self.head_ = model
                    self.selected_feature_ = feature
        return self

    def predict(self, X):
        if not hasattr(self, "head_"):
            raise RuntimeError("Fit the model using disjoint training and validation data first")
        if self.selected_feature_ == "raw":
            features = (np.asarray(X, float) - self.bank_.raw_mean_) / self.bank_.raw_scale_
        else:
            features = self.bank_.transform(X)[self.selected_feature_]
        return self.head_.predict(features)

    def save(self, path):
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle, protocol=5)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
