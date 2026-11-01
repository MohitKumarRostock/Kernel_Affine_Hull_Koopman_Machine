"""Matched KAHM/affine reconstruction banks for the diagnostic study.

Training calls the pinned upstream ``autoencoder`` without changing its float32
reference conversion, PCA, covariance kernel, or local regularizer.  Queries use
upstream ``kxa_matrix`` and the exact normalization/reconstruction algebra from
``autoencoder_filtering_extended``; this permits weight diagnostics without a
second kernel evaluation.  Tests compare both branches of its chunked query
implementation against the unchanged upstream filtering function.

This is a controlled geometry experiment, not an exact reproduction of the
original end-to-end learner: all geometries share each fitted partition, local
reference subset, effective PCA basis, and folding score.  There is one local
model per cluster.  ``native_*`` means the original *score formula* applied to
these shared references, not a separately fitted native baseline.

All feature statistics are fit using X passed to fit().  transform() neither
fits nor selects anything.  ``raw`` and ``*:phi_raw`` use training-standardized
raw histories even when a different geometry input scaling is requested.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


UPSTREAM_COMMIT = "d6d74d1953a4310b73065a556df816e5ab5e3ebc"
_UPSTREAM = Path(__file__).resolve().parent / "vendor" / "upstream"


def _upstream(name: str):
    """Use the pinned implementation and reject an ambiguous module import."""
    if not (_UPSTREAM / f"{name}.py").is_file():
        raise ImportError(f"Missing pinned upstream file: {name}.py")
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != (_UPSTREAM / f"{name}.py").resolve():
        raise ImportError(f"Already imported {name} from a different location")
    return module


def _matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("Expected a nonempty finite matrix with observations in rows")
    return X


def _array_hash(array):
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}:{array.shape}:".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def folding_components(Z, reconstruction):
    """Return upstream folding score, transformed radial error, angle, and norm."""
    query = np.asarray(Z, dtype=float).T
    estimate = np.asarray(reconstruction, dtype=float).T
    radius = np.linalg.norm(estimate - query, axis=0)
    radial = 1.0 - np.exp(-radius)
    dot = np.sum(estimate * query, axis=0)
    denom = np.linalg.norm(estimate, axis=0) * np.linalg.norm(query, axis=0)
    cosine = np.zeros_like(dot)
    mask = denom != 0
    cosine[mask] = dot[mask] / denom[mask]
    angle = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
    score = np.sqrt(radial**2 + angle**2) / np.sqrt(2.0)
    return score, radial, angle, radius


def associations(scores, omega=16.0, tau=1e-6):
    """Return (phi, log normalization coordinate, log phi) in stable arithmetic."""
    scores = _matrix(scores)
    if omega <= 0 or tau <= 0 or not np.isfinite([omega, tau]).all():
        raise ValueError("omega and tau must be finite and positive")
    logs = float(omega) * np.log(1.0 - np.clip(scores, 0.0, 1.0) + float(tau))
    logz = logsumexp(logs, axis=1, keepdims=True)
    logphi = logs - logz
    return np.exp(logphi), logz[:, 0], logphi


def _kahm_reconstruction(Z, ae, collect_diagnostics=False):
    """Upstream query algebra, including its exact-zero-normalizer convention."""
    kxa_matrix = _upstream("kxa_matrix").kxa_matrix
    y = np.asarray(Z, dtype=float).T
    x = ae["PC"].T @ y
    n = x.shape[1]
    estimate = np.zeros((y.shape[0], n), dtype=float)
    stats = {
        "weight_rows": 0, "weight_entries": 0, "zero_normalizers": 0,
        "negative_weight_entries": 0, "rows_with_negative_weights": 0,
        "normalizer_min": float("inf"), "normalizer_abs_min": float("inf"),
        "normalizer_max": float("-inf"), "weight_l1_sum": 0.0,
        "weight_l1_max": 0.0,
    }
    for start in range(0, n, 1000):
        stop = min(n, start + 1000)
        kernels = kxa_matrix(x[:, start:stop], ae["y_data_n_subspace"],
                             ae["sqrt_weights_matrix"], ae["kerneltype"])
        weights = kernels @ ae["B"]
        normalizers = weights.sum(axis=1, keepdims=True)
        safe = normalizers.copy()
        safe[safe == 0] = 1.0
        weights = weights / safe
        estimate[:, start:stop] = ae["y_data"] @ weights.T
        if collect_diagnostics:
            l1 = np.sum(np.abs(weights), axis=1)
            stats["weight_rows"] += len(weights)
            stats["weight_entries"] += weights.size
            stats["zero_normalizers"] += int(np.count_nonzero(normalizers == 0))
            stats["negative_weight_entries"] += int(np.count_nonzero(weights < 0))
            stats["rows_with_negative_weights"] += int(np.count_nonzero(np.any(weights < 0, axis=1)))
            stats["normalizer_min"] = min(stats["normalizer_min"], float(normalizers.min()))
            stats["normalizer_abs_min"] = min(stats["normalizer_abs_min"], float(np.abs(normalizers).min()))
            stats["normalizer_max"] = max(stats["normalizer_max"], float(normalizers.max()))
            stats["weight_l1_sum"] += float(l1.sum())
            stats["weight_l1_max"] = max(stats["weight_l1_max"], float(l1.max()))
    return estimate.T, stats if collect_diagnostics else None


@dataclass
class LocalModel:
    reference_indices: np.ndarray
    references: np.ndarray
    center: np.ndarray
    basis: np.ndarray
    ae: dict[str, Any] | None
    requested_rank: int
    scale: float
    residual_scale: float
    fallback_reason: str | None = None


class GeometryBank:
    """One frozen scaler and bank of exactly shared local reference geometries.

    ``oracle_labels`` may be a mapping from cluster count to full-training label
    vector, or one vector when there is only one block.  It is a training-only
    partition control; transform does not consume labels.  Its use must be
    reported separately from a learned partition experiment.

    ``local`` concatenates all affinity coordinates and affinity-gated local PCA
    coordinates, phi_c * ((z-center_c) @ basis_c).  It adds local position to the
    same soft gates.  Stored dimensions and simplex affine dimensions are both
    exposed.  Optional reconstruction vectors are diagnostic, not dimension-
    matched alternatives.
    """

    def __init__(self, seed=0, cluster_counts=(2, 4, 8), rank=2, max_refs=24,
                 scaling="standard", oracle_labels=None, omega=16.0, tau=1e-6,
                 max_fit=1600):
        self.seed = int(seed)
        self.cluster_counts = tuple(int(c) for c in cluster_counts)
        self.rank = int(rank)
        self.max_refs = int(max_refs)
        self.scaling = str(scaling)
        self.oracle_labels = oracle_labels
        self.omega, self.tau = float(omega), float(tau)
        self.max_fit = None if max_fit is None else int(max_fit)
        if self.seed < 0 or not self.cluster_counts or min(self.cluster_counts) < 1:
            raise ValueError("Need a nonnegative seed and positive cluster counts")
        if self.rank < 1 or self.max_refs < 2:
            raise ValueError("rank >= 1 and max_refs >= 2 are required")
        if self.max_fit is not None and self.max_fit < max(self.cluster_counts):
            raise ValueError("max_fit must cover the largest cluster count")
        if self.scaling not in {"standard", "raw", "minmax", "minmax_clip"}:
            raise ValueError("Unsupported input scaling")
        if not np.isfinite([self.omega, self.tau]).all() or min(self.omega, self.tau) <= 0:
            raise ValueError("omega and tau must be finite and positive")

    def _scale_inputs(self, X):
        if self.scaling == "standard":
            Z = (X - self.raw_mean_) / self.raw_scale_
        elif self.scaling == "raw":
            Z = X.copy()
        else:
            span = np.where(self.constant_features_, 1.0, self.input_range_)
            Z = 2.0 * ((X - self.input_min_) / span) - 1.0
            Z[:, self.constant_features_] = 0.0
            if self.scaling == "minmax_clip":
                Z = np.clip(Z, -1.0, 1.0)
        if not np.isfinite(Z).all():
            raise ValueError("Scaled inputs are not finite")
        return Z

    def _labels(self, X, count, seed):
        if self.oracle_labels is None:
            km = KMeans(n_clusters=count, random_state=seed, n_init=5, max_iter=150).fit(X)
            if len(np.unique(km.labels_)) != count:
                raise ValueError("Fewer nonempty clusters than requested; do not silently drop states")
            return km.labels_, km.cluster_centers_, "kmeans"
        source = self.oracle_labels
        if isinstance(source, dict):
            if count not in source:
                raise ValueError(f"No oracle labels supplied for {count} clusters")
            source = source[count]
        elif len(self.cluster_counts) != 1:
            raise ValueError("Multiple oracle blocks require labels keyed by cluster count")
        source = np.asarray(source)
        if source.ndim != 1 or len(source) != self.n_training_rows_:
            raise ValueError("Oracle labels must cover all training rows")
        _, full = np.unique(source, return_inverse=True)
        labels = full[self.fit_indices_]
        if set(np.unique(labels)) != set(range(count)):
            raise ValueError("Oracle partition does not have the requested nonempty groups")
        centers = np.stack([X[labels == c].mean(axis=0) for c in range(count)])
        return labels, centers, "oracle"

    def fit(self, X):
        started = time.perf_counter()
        X = _matrix(X)
        if X.shape[1] < 2:
            raise ValueError("The matched normal-direction design requires at least two input coordinates")
        if len(X) < max(self.cluster_counts):
            raise ValueError("Fewer training rows than requested clusters")
        self.n_training_rows_, self.n_features_in_ = X.shape
        self.training_input_sha256_ = _array_hash(X)
        self.raw_mean_ = X.mean(axis=0)
        self.raw_scale_ = X.std(axis=0)
        self.raw_scale_ = np.where(self.raw_scale_ > 1e-12, self.raw_scale_, 1.0)
        self.input_min_, self.input_max_ = X.min(axis=0), X.max(axis=0)
        self.input_range_ = self.input_max_ - self.input_min_
        self.constant_features_ = self.input_range_ == 0
        # Common aliases ease metadata inspection and compatibility with heads.
        self.mean_, self.scale_ = self.raw_mean_.copy(), self.raw_scale_.copy()
        Z = self._scale_inputs(X)
        rng = np.random.default_rng(self.seed)
        if self.max_fit is not None and len(Z) > self.max_fit:
            self.fit_indices_ = np.sort(rng.choice(len(Z), self.max_fit, replace=False))
        else:
            self.fit_indices_ = np.arange(len(Z))
        train = Z[self.fit_indices_]
        train_ae = _upstream("autoencoder").autoencoder
        self.blocks_, self.block_slices_, self.reference_indices_ = [], [], []
        offset = 0
        ae_seconds = 0.0
        partition_seconds = 0.0
        for b, count in enumerate(self.cluster_counts):
            seed = (self.seed + 104729 * (b + 1)) % (2**32)
            tick = time.perf_counter()
            labels, partition_centers, partition_kind = self._labels(train, count, seed)
            partition_seconds += time.perf_counter() - tick
            local_rng = np.random.default_rng(seed + 1)
            models = []
            for c in range(count):
                ids = self.fit_indices_[np.flatnonzero(labels == c)]
                if len(ids) > self.max_refs:
                    ids = np.sort(local_rng.choice(ids, self.max_refs, replace=False))
                # Both reconstruction methods use exactly the upstream float32 references.
                refs = np.asarray(Z[ids], dtype=np.float32)
                requested_rank = min(self.rank, X.shape[1] - 1, len(refs) - 1)
                ae, fallback_reason = None, None
                tick = time.perf_counter()
                if requested_rank > 0:
                    try:
                        ae = train_ae(refs.T, requested_rank)
                    except ValueError as error:
                        if str(error) != "No eigenvalues > eps":
                            raise
                        fallback_reason = "no positive empirical-variance direction"
                else:
                    fallback_reason = "fewer than two reference rows"
                ae_seconds += time.perf_counter() - tick
                basis = np.empty((X.shape[1], 0)) if ae is None else ae["PC"].copy()
                center = refs.astype(np.float64).mean(axis=0)
                local = refs.astype(np.float64) - center
                residual = local - (local @ basis) @ basis.T
                scale = max(float(np.mean(np.sum(local**2, axis=1))), 1e-6)
                residual_scale = max(float(np.mean(np.sum(residual**2, axis=1))), 0.05 * scale, 1e-6)
                models.append(LocalModel(ids.copy(), refs.copy(), center, basis, ae,
                                         requested_rank, scale, residual_scale, fallback_reason))
            self.blocks_.append({"count": count, "seed": seed,
                                 "labels": labels.copy(), "partition_centers": partition_centers.copy(),
                                 "partition_kind": partition_kind, "models": models,
                                 "rbf_bandwidth": float(np.median([m.scale for m in models]))})
            self.reference_indices_.append([m.reference_indices.copy() for m in models])
            self.block_slices_.append(slice(offset, offset + count))
            offset += count
        self.n_features_out_ = offset
        self.simplex_affine_dimension_ = offset - len(self.blocks_)
        self.fit_diagnostics_ = {
            "seconds": time.perf_counter() - started,
            "partition_seconds": partition_seconds,
            "upstream_ae_seconds": ae_seconds,
            "fallback_clusters": sum(m.ae is None for b in self.blocks_ for m in b["models"]),
            "total_local_models": offset,
            "total_reference_rows_with_repetition": sum(len(m.references) for b in self.blocks_ for m in b["models"]),
            "stored_affinity_coordinates": offset,
            "simplex_affine_dimension_upper_bound": self.simplex_affine_dimension_,
            "primary_geometry_array_bytes": {
                "kahm": sum(sum(value.nbytes for value in m.ae.values() if isinstance(value, np.ndarray))
                            if m.ae is not None else m.center.nbytes
                            for b in self.blocks_ for m in b["models"]),
                "affine": sum(m.center.nbytes + m.basis.nbytes for b in self.blocks_ for m in b["models"]),
                "centroid": sum(m.center.nbytes for b in self.blocks_ for m in b["models"]),
            },
            "geometry_bytes_scope": "Numerical arrays required by the primary reconstruction map, excluding shared scaler, training/partition records, Python overhead, and decoder. Local-coordinate variants need additional bases.",
        }
        return self

    def transform(self, X, diagnostics=False, include_reconstruction=False):
        started = time.perf_counter()
        X = _matrix(X)
        if not hasattr(self, "blocks_"):
            raise ValueError("Fit GeometryBank before transforming")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Query dimension differs from fitted histories")
        Z = self._scale_inputs(X)
        raw = (X - self.raw_mean_) / self.raw_scale_
        geometries = ("kahm", "affine", "centroid")
        names = ("phi", "log_phi", "logz", "scores", "radial", "angle", "distance", "gated", "reconstruction")
        pieces = {g: {name: [] for name in names} for g in geometries}
        native = {"native_rbf:phi": [], "native_affine:phi": []}
        diag = {g: {"best_reconstruction_mse_by_block": [], "gated_reconstruction_mse_by_block": [],
                    "mean_score_by_block": [], "mean_entropy_by_block": [], "reconstruction_seconds": 0.0}
                for g in geometries}
        weight_stats = []
        for block in self.blocks_:
            count = block["count"]
            scores = {g: np.empty((len(Z), count)) for g in geometries}
            radials = {g: np.empty((len(Z), count)) for g in geometries}
            angles = {g: np.empty((len(Z), count)) for g in geometries}
            radii = {g: np.empty((len(Z), count)) for g in geometries}
            reconstruction_vectors = {g: [] for g in geometries}
            coordinates = []
            rbf_scores, affine_scores = np.empty((len(Z), count)), np.empty((len(Z), count))
            for c, model in enumerate(block["models"]):
                centered = Z - model.center
                coordinates.append(centered @ model.basis)
                tick = time.perf_counter()
                if model.ae is None:
                    kahm_recon, wstats = np.broadcast_to(model.center, Z.shape), None
                else:
                    kahm_recon, wstats = _kahm_reconstruction(Z, model.ae, diagnostics)
                diag["kahm"]["reconstruction_seconds"] += time.perf_counter() - tick
                if wstats is not None:
                    weight_stats.append(wstats)
                tick = time.perf_counter()
                affine_recon = model.center + coordinates[-1] @ model.basis.T
                diag["affine"]["reconstruction_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                centroid_recon = np.broadcast_to(model.center, Z.shape)
                diag["centroid"]["reconstruction_seconds"] += time.perf_counter() - tick
                for g, recon in (("kahm", kahm_recon), ("affine", affine_recon), ("centroid", centroid_recon)):
                    score, radial, angle, radius = folding_components(Z, recon)
                    scores[g][:, c], radials[g][:, c] = score, radial
                    angles[g][:, c], radii[g][:, c] = angle, radius
                    if include_reconstruction:
                        reconstruction_vectors[g].append(np.asarray(recon))
                d2 = np.sum(centered**2, axis=1)
                rbf_scores[:, c] = 0.5 * d2 / block["rbf_bandwidth"]
                affine_scores[:, c] = radii["affine"][:, c]**2 / model.residual_scale + 0.1 * d2 / model.scale
            for g in geometries:
                phi, logz, logphi = associations(scores[g], self.omega, self.tau)
                for name, value in (("phi", phi), ("log_phi", logphi), ("logz", logz[:, None]),
                                    ("scores", scores[g]), ("radial", radials[g]), ("angle", angles[g]),
                                    ("distance", radii[g])):
                    pieces[g][name].append(value)
                pieces[g]["gated"].extend(phi[:, c:c+1] * coordinates[c] for c in range(count))
                if include_reconstruction:
                    pieces[g]["reconstruction"].extend(reconstruction_vectors[g])
                if diagnostics:
                    squared = radii[g]**2 / self.n_features_in_
                    diag[g]["best_reconstruction_mse_by_block"].append(float(np.min(squared, axis=1).mean()))
                    diag[g]["gated_reconstruction_mse_by_block"].append(float(np.sum(phi * squared, axis=1).mean()))
                    diag[g]["mean_score_by_block"].append(float(scores[g].mean()))
                    diag[g]["mean_entropy_by_block"].append(float(-np.sum(phi * logphi, axis=1).mean()))
            for key, score in (("native_rbf:phi", rbf_scores), ("native_affine:phi", affine_scores)):
                native[key].append(np.exp(-score - logsumexp(-score, axis=1, keepdims=True)))
        output = {"raw": raw}
        for g in geometries:
            for name in ("phi", "log_phi", "logz", "scores", "radial", "angle", "distance"):
                output[f"{g}:{name}"] = np.concatenate(pieces[g][name], axis=1)
            phi = output[f"{g}:phi"]
            output[f"{g}:phi_logz"] = np.concatenate([phi, output[f"{g}:logz"]], axis=1)
            output[f"{g}:radial_angle"] = np.concatenate([output[f"{g}:radial"], output[f"{g}:angle"]], axis=1)
            output[f"{g}:phi_raw"] = np.concatenate([phi, raw], axis=1)
            output[f"{g}:local"] = np.concatenate([phi] + pieces[g]["gated"], axis=1)
            if include_reconstruction:
                output[f"{g}:reconstruction"] = np.concatenate(pieces[g]["reconstruction"], axis=1)
        output.update({key: np.concatenate(value, axis=1) for key, value in native.items()})
        if any(not np.isfinite(value).all() for value in output.values()):
            raise FloatingPointError("Nonfinite geometry features; retain this failure rather than substituting features")
        self.last_diagnostics_ = {
            "n_queries": len(X), "seconds": time.perf_counter() - started,
            "geometries": diag, "feature_dimensions": {key: value.shape[1] for key, value in output.items()},
            "weight_diagnostics": self._aggregate_weights(weight_stats) if diagnostics else None,
            "timing_scope": "shared-bank execution diagnostics; not standalone end-to-end timing comparisons",
        }
        return output

    @staticmethod
    def _aggregate_weights(stats):
        if not stats:
            return {"weight_rows": 0, "note": "No nondegenerate KAHM models"}
        sums = ("weight_rows", "weight_entries", "zero_normalizers", "negative_weight_entries",
                "rows_with_negative_weights", "weight_l1_sum")
        result = {name: sum(s[name] for s in stats) for name in sums}
        for name in ("normalizer_min", "normalizer_abs_min"):
            result[name] = min(s[name] for s in stats)
        for name in ("normalizer_max", "weight_l1_max"):
            result[name] = max(s[name] for s in stats)
        result["weight_l1_mean"] = result["weight_l1_sum"] / result["weight_rows"]
        result["negative_weight_entry_fraction"] = result["negative_weight_entries"] / result["weight_entries"]
        result["negative_weight_row_fraction"] = result["rows_with_negative_weights"] / result["weight_rows"]
        return result

    def metadata(self):
        if not hasattr(self, "blocks_"):
            raise ValueError("Fit GeometryBank first")
        return {
            "upstream_commit": UPSTREAM_COMMIT,
            "config": {"seed": self.seed, "cluster_counts": list(self.cluster_counts), "rank": self.rank,
                       "max_refs": self.max_refs, "scaling": self.scaling, "omega": self.omega,
                       "tau": self.tau, "max_fit": self.max_fit},
            "n_training_rows": self.n_training_rows_, "input_dimension": self.n_features_in_,
            "training_input_sha256": self.training_input_sha256_,
            "fit_indices": self.fit_indices_.tolist(),
            "scaler": {"raw_mean": self.raw_mean_.tolist(), "raw_scale": self.raw_scale_.tolist(),
                       "minimum": self.input_min_.tolist(), "maximum": self.input_max_.tolist()},
            "blocks": [{"count": b["count"], "seed": b["seed"], "partition_kind": b["partition_kind"],
                        "partition_labels": b["labels"].tolist(),
                        "partition_labels_sha256": _array_hash(b["labels"]),
                        "models": [{"reference_indices": m.reference_indices.tolist(),
                                    "reference_indices_sha256": _array_hash(m.reference_indices),
                                    "references_sha256": _array_hash(m.references),
                                    "basis_sha256": _array_hash(m.basis),
                                    "requested_rank": m.requested_rank, "effective_rank": m.basis.shape[1],
                                    "fallback_reason": m.fallback_reason}
                                   for m in b["models"]]} for b in self.blocks_],
            "fit_diagnostics": self.fit_diagnostics_.copy(),
            "caveats": ["One local model per cluster; all geometries share float32 local references and effective PCA basis.",
                        "Native controls use original score forms on shared references, not separately fit end-to-end baselines.",
                        "Stored affinity coordinate count differs from simplex affine dimension.",
                        "Timings are execution diagnostics with shared computation."],
        }

    def state_dict(self):
        """Full numerical state, including exact training reference IDs and AEs."""
        if not hasattr(self, "blocks_"):
            raise ValueError("Fit GeometryBank first")
        return {key: value for key, value in self.__dict__.items()
                if key not in {"last_diagnostics_", "oracle_labels"}}

    def save(self, path):
        """Persist a trusted local model state plus human-readable frozen metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump({"format": "matched_geometry_bank_v1", "state": self.state_dict()}, stream, protocol=5)
        with path.with_suffix(path.suffix + ".json").open("w") as stream:
            json.dump(self.metadata(), stream, indent=2)
        return path

    @classmethod
    def load(cls, path):
        """Read only a model state produced locally by this study (pickle)."""
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("format") != "matched_geometry_bank_v1":
            raise ValueError("Unknown geometry model format")
        bank = cls()
        bank.__dict__.update(payload["state"])
        bank.oracle_labels = None
        return bank
