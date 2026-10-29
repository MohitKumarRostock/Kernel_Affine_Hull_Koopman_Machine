"""Geometry-induced soft states with a nested, retained feature bank.

``kind='kahm'`` calls the original repository's KAHM training and folding
implementation. Association weights are normalized in log space.
The ``affine`` and ``rbf`` options define separate geometric baselines.
All fitted transformations use development inputs only.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


def _matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("Expected a nonempty finite matrix with samples in rows.")
    return X


def stable_associations(folding_scores, omega=16.0, tau=1e-6):
    """Normalize rows of folding scores without a small-denominator fallback.

    Input and output shape is ``(samples, states)``.  The formula is exactly
    proportional to ``(1 - clip(T, 0, 1) + tau)**omega`` in real arithmetic.
    """
    T = _matrix(folding_scores)
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(tau) or tau <= 0:
        raise ValueError("omega and tau must be positive and finite.")
    logs = float(omega) * np.log(1.0 - np.clip(T, 0.0, 1.0) + float(tau))
    return np.exp(logs - logsumexp(logs, axis=1, keepdims=True))


def _kahm_module():
    """Import the pinned parent-repository implementation, without copying it."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "kernel_affine_hull_koopman_machines.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return importlib.import_module("kernel_affine_hull_koopman_machines")
    raise ImportError(
        "KAHM features require this extension inside its parent research "
        "repository; the affine and rbf feature maps are self-contained."
    )


@dataclass
class _KAHMBlock:
    n_clusters: int
    seed: int
    omega: float
    tau: float
    max_train_per_cluster: int
    subspace_dim: int

    def fit(self, X):
        original = _kahm_module()
        self.model_ = original.train_kahm_state_regime_abstraction(
            X.T,
            n_clusters=self.n_clusters,
            subspace_dim=min(self.subspace_dim, X.shape[1]),
            Nb=self.max_train_per_cluster,
            random_state=self.seed,
            verbose=False,
            kmeans_kind="full",
            max_train_per_cluster=self.max_train_per_cluster,
            model_dtype="float64",
            save_ae_to_disk=False,
        )
        self.n_features_out_ = int(self.model_["n_clusters"])
        return self

    def transform(self, X):
        original = _kahm_module()
        scores = original.kahm_folding_distance_matrix(
            self.model_, X.T, n_jobs=1, batch_size=1000, show_progress=False
        ).T
        return stable_associations(scores, self.omega, self.tau)


@dataclass
class _GeometricBlock:
    n_clusters: int
    seed: int
    kind: str
    sharpness: float
    subspace_dim: int
    locality: float = 0.1

    def fit(self, X):
        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed,
                    n_init=5, max_iter=150).fit(X)
        self.centers_ = km.cluster_centers_
        self.n_features_out_ = self.n_clusters
        self.bases_, self.scales_, self.residual_scales_ = [], [], []
        for c in range(self.n_clusters):
            local = X[km.labels_ == c] - self.centers_[c]
            scale = max(float(np.mean(np.sum(local * local, axis=1))), 1e-6)
            if len(local) >= 2:
                _, _, vt = np.linalg.svd(local, full_matrices=False)
                # Retain at least one normal direction whenever possible.
                rank = min(self.subspace_dim, max(X.shape[1] - 1, 0), len(vt))
                basis = vt[:rank].T
            else:
                basis = np.empty((X.shape[1], 0))
            residual = local - (local @ basis) @ basis.T
            residual_scale = max(
                float(np.mean(np.sum(residual * residual, axis=1))),
                0.05 * scale, 1e-6,
            )
            self.bases_.append(basis)
            self.scales_.append(scale)
            self.residual_scales_.append(residual_scale)
        self.scales_ = np.asarray(self.scales_)
        self.residual_scales_ = np.asarray(self.residual_scales_)
        # One global bandwidth avoids an implicit state-dependent density prior.
        self.bandwidth_ = max(float(np.median(self.scales_)), 1e-6)
        return self

    def transform(self, X):
        scores = np.empty((len(X), self.n_clusters))
        for c in range(self.n_clusters):
            centered = X - self.centers_[c]
            d2 = np.sum(centered * centered, axis=1)
            if self.kind == "rbf":
                scores[:, c] = 0.5 * d2 / self.bandwidth_
            else:
                basis = self.bases_[c]
                residual = centered - (centered @ basis) @ basis.T
                scores[:, c] = (
                    np.sum(residual * residual, axis=1)
                    / self.residual_scales_[c]
                    + self.locality * d2 / self.scales_[c]
                )
        logs = -self.sharpness * scores
        return np.exp(logs - logsumexp(logs, axis=1, keepdims=True))


class NestedSoftFeatures:
    """Retained simplex blocks whose prefixes are exactly nested.

    Concatenation is a product of simplices, not a single simplex.  Keeping all
    existing blocks gives an isometric zero-padding embedding of decoder
    coefficient matrices.  A fixed Frobenius constraint therefore defines
    nested predictor classes; replacing a coarse map by a new map does not.

    ``max_fit`` limits geometry-training inputs using a seeded sample without
    replacement.  It does not limit the inputs used to fit the decoder.

    ``input_scaling=None`` preserves the historical ``standardize`` flag.
    Explicit modes are ``standard``, ``raw``, ``minmax``, and ``minmax_clip``.
    Both min/max modes fit coordinate extrema on all training histories before
    geometry subsampling and map training-constant coordinates to zero. Only
    ``minmax_clip`` clips new inputs to [-1, 1]; plain ``minmax`` extrapolates
    outside the training extrema. Target/output scaling is never changed.
    """

    def __init__(self, cluster_counts=(2, 4, 8, 16), kind="kahm", seed=0,
                 omega=16.0, tau=1e-6, max_fit=2500,
                 max_train_per_cluster=48, subspace_dim=2,
                 sharpness=1.0, standardize=True, input_scaling=None):
        self.cluster_counts = tuple(int(c) for c in cluster_counts)
        self.kind = kind
        self.seed = int(seed)
        self.omega, self.tau = float(omega), float(tau)
        self.max_fit = None if max_fit is None else int(max_fit)
        self.max_train_per_cluster = int(max_train_per_cluster)
        self.subspace_dim = int(subspace_dim)
        self.sharpness = float(sharpness)
        self.standardize = bool(standardize)
        self.input_scaling = input_scaling

    def fit(self, X):
        X = _matrix(X)
        if self.kind not in {"kahm", "affine", "rbf"}:
            raise ValueError("kind must be 'kahm', 'affine', or 'rbf'.")
        if not self.cluster_counts or min(self.cluster_counts) < 1:
            raise ValueError("cluster_counts must contain positive integers.")
        if self.max_fit is not None and self.max_fit < max(self.cluster_counts):
            raise ValueError("max_fit must be at least the largest cluster count.")
        if self.max_train_per_cluster < 2 or self.subspace_dim < 1:
            raise ValueError("Need at least two prototypes and one subspace dimension.")
        if not np.isfinite(self.sharpness) or self.sharpness <= 0:
            raise ValueError("sharpness must be positive and finite.")
        if len(X) < max(self.cluster_counts):
            raise ValueError("There are fewer observations than requested states.")
        requested = getattr(self, "input_scaling", None)
        self.input_scaling_ = (
            "standard" if self.standardize else "raw"
        ) if requested is None else requested
        if self.input_scaling_ not in ("standard", "raw", "minmax", "minmax_clip"):
            raise ValueError("input_scaling must be standard, raw, minmax, minmax_clip, or None.")
        self.n_features_in_ = X.shape[1]
        standard = self.input_scaling_ == "standard"
        with np.errstate(over="ignore", invalid="ignore"):
            self.mean_ = X.mean(axis=0) if standard else np.zeros(X.shape[1])
            self.scale_ = X.std(axis=0) if standard else np.ones(X.shape[1])
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.scale_).all():
            raise ValueError("Training centering and scaling parameters must be finite.")
        self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)
        if self.input_scaling_ in ("minmax", "minmax_clip"):
            self.input_min_, self.input_max_ = X.min(axis=0), X.max(axis=0)
            self.input_range_ = self.input_max_ - self.input_min_
            if not np.isfinite(self.input_range_).all():
                raise ValueError("Training coordinate ranges must be finite.")
            self.constant_features_ = self.input_range_ == 0.0
        Z = self._scale_inputs(X, clip=False)
        rng = np.random.default_rng(self.seed)
        if self.max_fit is not None and len(Z) > self.max_fit:
            self.fit_indices_ = np.sort(rng.choice(len(Z), self.max_fit, replace=False))
        else:
            self.fit_indices_ = np.arange(len(Z))
        train = Z[self.fit_indices_]
        self.blocks_, self.block_slices_ = [], []
        start = 0
        for j, count in enumerate(self.cluster_counts):
            seed = self.seed + 104729 * (j + 1)
            if self.kind == "kahm":
                block = _KAHMBlock(count, seed, self.omega, self.tau,
                                   self.max_train_per_cluster, self.subspace_dim)
            else:
                block = _GeometricBlock(count, seed, self.kind,
                                         self.sharpness, self.subspace_dim)
            block.fit(train)
            end = start + block.n_features_out_
            self.blocks_.append(block)
            self.block_slices_.append(slice(start, end))
            start = end
        self.n_features_out_ = start
        return self

    def _scale_inputs(self, X, *, clip=True):
        # Older saved banks lack input_scaling_; their mean_/scale_ remain valid.
        mode = getattr(self, "input_scaling_", None)
        if mode in ("minmax", "minmax_clip"):
            span = np.where(self.constant_features_, 1.0, self.input_range_)
            X = 2.0 * ((X - self.input_min_) / span) - 1.0
            X[:, self.constant_features_] = 0.0
            if not np.isfinite(X).all():
                raise ValueError("Scaled inputs must be finite.")
            if clip and mode == "minmax_clip":
                X = np.clip(X, -1.0, 1.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled = (X - self.mean_) / self.scale_
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled inputs must be finite.")
        return scaled

    def input_scaling_metadata(self):
        """JSON-safe frozen input-map parameters; no held-out statistics."""
        if not hasattr(self, "blocks_"):
            raise ValueError("Fit the feature map before requesting scaling metadata.")
        mode = getattr(self, "input_scaling_", "standard" if self.standardize else "raw")
        description = {
            "mode": mode, "clip": mode == "minmax_clip",
            "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
        }
        if mode in ("minmax", "minmax_clip"):
            description.update({
                "training_min": self.input_min_.tolist(),
                "training_max": self.input_max_.tolist(),
                "training_range": self.input_range_.tolist(),
                "constant_features": self.constant_features_.tolist(),
                "constant_feature_policy": "map to zero",
                "formula": "2*(x-training_min)/training_range-1",
            })
        return description

    def transform(self, X, n_blocks=None):
        X = _matrix(X)
        if not hasattr(self, "blocks_"):
            raise ValueError("Fit the feature map before transforming inputs.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input dimension differs from the fitted feature map.")
        n = len(self.blocks_) if n_blocks is None else int(n_blocks)
        if n < 1 or n > len(self.blocks_):
            raise ValueError("n_blocks must select a nonempty fitted prefix.")
        Z = self._scale_inputs(X)
        return np.concatenate([b.transform(Z) for b in self.blocks_[:n]], axis=1)

    def fit_transform(self, X, n_blocks=None):
        return self.fit(X).transform(X, n_blocks=n_blocks)

