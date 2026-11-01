"""Standalone, fixed-configuration models for the natural forecasting study.

Every fit is a complete pipeline. ``fit_select`` shares training-only geometry
and head construction across an explicit validation-selected alpha grid within
one method; no banks or caches are shared between methods. Affine and locality-only pipelines
construct the same reference partitions and principal bases as the historical
KAHM study, but never train or query a KAHM kernel. The original study files are
imported read-only and remain unchanged.

The default KAHM query uses analytic Gaussian log kernels with common row
rescaling. This preserves normalized weights in exact arithmetic away from a
zero signed denominator. ``query_mode='historical'`` retains the original
floating-point query arithmetic for equivalence checks. Both modes retain the
historical exact-zero signed-denominator convention; diagnostics expose it.

Input, feature, and target scales use training observations only. Predictions
are returned in the original target units and are not clipped. ``raw_skip``
explicitly concatenates all standardized raw inputs with the local code; no
claim of injectivity is inferred from the local feature count or numerical
basis rank alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from extensions.predictive_geometry.study import geometry as historical_geometry
from extensions.predictive_geometry.study import heads as historical_heads


REPRESENTATIONS = ("raw", "affine_local", "locality_local", "kahm_local")
HEADS = ("ridge", "rbf")
DEFAULT_GEOMETRY_PARAMETERS = {
    "cluster_counts": (2, 4, 8), "rank": 2, "max_refs": 24,
    "max_fit": 512, "omega": 16.0, "tau": 1e-6,
}


def _matrix(value, *, name="input"):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix with observations in rows")
    return value


def _standardizer(value, threshold):
    mean, scale = value.mean(axis=0), value.std(axis=0)
    return mean, np.where(scale > threshold, scale, 1.0)


def _basis_without_kernel(references, requested_rank):
    """Reproduce the pinned autoencoder's PCA/range rule, without its kernel."""
    if requested_rank <= 0:
        return np.empty((references.shape[1], 0)), "fewer than two reference rows"
    dim_reduce = historical_geometry._upstream("dim_reduce").dim_reduce
    rank = requested_rank
    try:
        projected, basis = dim_reduce(references.T, rank)
        while np.min(np.ptp(projected, axis=1)) < 1e-3 and rank > 1:
            rank -= 1
            projected, basis = dim_reduce(references.T, rank)
    except ValueError as error:
        if str(error) != "No eigenvalues > eps":
            raise
        return np.empty((references.shape[1], 0)), "no positive empirical-variance direction"
    return basis, None


def _stable_kahm_reconstruction(z, ae, collect_diagnostics=False):
    """Same fitted AE and distance expression; exponentiate centered log rows."""
    distance_matrix = historical_geometry._upstream("distance_matrix").distance_matrix
    z = _matrix(z)
    projected = ae["PC"].T @ z.T
    precision_sqrt = np.asarray(ae["sqrt_weights_matrix"], dtype=float)
    references = precision_sqrt @ np.asarray(ae["y_data_n_subspace"], dtype=float)
    result = np.empty_like(z)
    stats = {
        "weight_rows": 0, "weight_entries": 0, "zero_normalizers": 0,
        "near_cancelling_normalizers": 0, "original_all_zero_kernel_rows": 0,
        "negative_weight_entries": 0, "maximum_weight_l1": 0.0,
    }
    for start in range(0, len(z), 1000):
        stop = min(start + 1000, len(z))
        distances = distance_matrix((precision_sqrt @ projected[:, start:stop]).T,
                                    references.T)
        logs = -(0.5 / projected.shape[0]) * distances
        if not np.isfinite(logs).all():
            raise FloatingPointError("Nonfinite analytic Gaussian log kernel")
        maxima = logs.max(axis=1, keepdims=True)
        kernel = np.exp(logs - maxima)
        weights = kernel @ ae["B"]
        normalizers = weights.sum(axis=1, keepdims=True)
        safe = np.where(normalizers == 0, 1.0, normalizers)
        normalized = weights / safe
        result[start:stop] = (ae["y_data"] @ normalized.T).T
        if collect_diagnostics:
            weight_l1 = np.abs(weights).sum(axis=1)
            stats["weight_rows"] += len(weights)
            stats["weight_entries"] += int(weights.size)
            stats["zero_normalizers"] += int(np.count_nonzero(normalizers == 0))
            stats["near_cancelling_normalizers"] += int(np.count_nonzero(
                (weight_l1 > 0) & (np.abs(normalizers[:, 0]) <= 1e-12 * weight_l1)))
            with np.errstate(under="ignore"):
                stats["original_all_zero_kernel_rows"] += int(np.count_nonzero(np.exp(maxima) == 0))
            stats["negative_weight_entries"] += int(np.count_nonzero(normalized < 0))
            stats["maximum_weight_l1"] = max(stats["maximum_weight_l1"],
                                              float(np.abs(normalized).sum(axis=1).max()))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite KAHM reconstruction; no replacement method was used")
    return result, stats if collect_diagnostics else None


@dataclass
class _LocalModel:
    reference_indices: np.ndarray
    references: np.ndarray
    center: np.ndarray
    basis: np.ndarray
    scale: float
    residual_scale: float
    requested_rank: int
    fallback_reason: str | None
    ae: dict[str, Any] | None = None


class LocalFeatureBank:
    """One independently fitted geometry and its affinity-gated coordinates.

    ``affine`` uses historical folding scores of affine reconstruction.
    ``locality`` uses the historical native-affine locality score, with the same
    local coordinates. ``kahm`` uses the pinned KAHM reconstruction and folding
    score. Reference selection/PCA is identical for a common seed and config.
    """

    def __init__(self, gate="kahm", *, seed=0, cluster_counts=(2, 4, 8), rank=2,
                 max_refs=24, max_fit=512, omega=16.0, tau=1e-6,
                 query_mode="stable"):
        self.gate, self.seed = gate, int(seed)
        self.cluster_counts = tuple(int(c) for c in cluster_counts)
        self.rank, self.max_refs = int(rank), int(max_refs)
        self.max_fit = None if max_fit is None else int(max_fit)
        self.omega, self.tau = float(omega), float(tau)
        self.query_mode = query_mode
        if gate not in {"affine", "locality", "kahm"}:
            raise ValueError("gate must be affine, locality, or kahm")
        if query_mode not in {"stable", "historical"}:
            raise ValueError("query_mode must be stable or historical")
        if self.seed < 0 or not self.cluster_counts or min(self.cluster_counts) < 1:
            raise ValueError("Need a nonnegative seed and positive cluster counts")
        if self.rank < 1 or self.max_refs < 2:
            raise ValueError("rank must be positive and max_refs at least two")
        if self.max_fit is not None and self.max_fit < max(self.cluster_counts):
            raise ValueError("max_fit must cover the largest cluster count")
        if min(self.omega, self.tau) <= 0 or not np.isfinite([self.omega, self.tau]).all():
            raise ValueError("omega and tau must be finite and positive")

    def fit(self, x):
        x = _matrix(x)
        if x.shape[1] < 2:
            raise ValueError("The matched normal-direction geometry requires at least two input coordinates")
        if len(x) < max(self.cluster_counts):
            raise ValueError("Fewer training rows than requested clusters")
        self.n_training_rows_, self.n_features_in_ = x.shape
        self.mean_, self.scale_ = _standardizer(x, 1e-12)
        z = self.scale_inputs(x)
        rng = np.random.default_rng(self.seed)
        self.fit_indices_ = (np.sort(rng.choice(len(z), self.max_fit, replace=False))
                             if self.max_fit is not None and len(z) > self.max_fit
                             else np.arange(len(z)))
        partition_data = z[self.fit_indices_]
        self.blocks_, self.models_ = [], []
        self.training_input_sha256_ = historical_geometry._array_hash(x)
        # The affine path deliberately never imports or calls the autoencoder.
        fit_ae = historical_geometry._upstream("autoencoder").autoencoder if self.gate == "kahm" else None
        for block_index, count in enumerate(self.cluster_counts):
            block_seed = (self.seed + 104729 * (block_index + 1)) % (2**32)
            partition = KMeans(n_clusters=count, random_state=block_seed,
                               n_init=5, max_iter=150).fit(partition_data)
            if len(np.unique(partition.labels_)) != count:
                raise ValueError("Fewer nonempty clusters than requested; no states were dropped")
            local_rng = np.random.default_rng(block_seed + 1)
            block_models = []
            for cluster in range(count):
                ids = self.fit_indices_[np.flatnonzero(partition.labels_ == cluster)]
                if len(ids) > self.max_refs:
                    ids = np.sort(local_rng.choice(ids, self.max_refs, replace=False))
                refs = np.asarray(z[ids], dtype=np.float32)
                requested = min(self.rank, x.shape[1] - 1, len(refs) - 1)
                ae, fallback = None, None
                if self.gate == "kahm" and requested > 0:
                    try:
                        ae = fit_ae(refs.T, requested)
                    except ValueError as error:
                        if str(error) != "No eigenvalues > eps":
                            raise
                        fallback = "no positive empirical-variance direction"
                    basis = np.empty((x.shape[1], 0)) if ae is None else ae["PC"]
                else:
                    basis, fallback = _basis_without_kernel(refs, requested)
                center = refs.astype(np.float64).mean(axis=0)
                centered = refs.astype(np.float64) - center
                residual = centered - (centered @ basis) @ basis.T
                scale = max(float(np.mean(np.sum(centered ** 2, axis=1))), 1e-6)
                residual_scale = max(float(np.mean(np.sum(residual ** 2, axis=1))),
                                     0.05 * scale, 1e-6)
                model = _LocalModel(ids.copy(), refs, center, basis, scale,
                                    residual_scale, requested, fallback, ae)
                block_models.append(model)
                self.models_.append(model)
            self.blocks_.append({"count": count, "seed": block_seed,
                                 "models": block_models,
                                 "labels": partition.labels_.copy()})
        self.n_gates_ = len(self.models_)
        self.n_projected_coordinates_ = sum(m.basis.shape[1] for m in self.models_)
        self.n_features_out_ = self.n_gates_ + self.n_projected_coordinates_
        stack = np.concatenate([m.basis.T for m in self.models_], axis=0)
        singular = np.linalg.svd(stack, compute_uv=False) if len(stack) else np.zeros(0)
        singular = np.pad(singular, (0, max(0, self.n_features_in_ - len(singular))))
        largest, smallest = float(singular[0]), float(singular[-1])
        tolerance = largest * max(stack.shape) * np.finfo(float).eps
        numerical_rank = int(np.count_nonzero(singular > tolerance))
        self.stack_diagnostics_ = {
            "rows": int(stack.shape[0]), "columns": self.n_features_in_,
            "numerical_rank": numerical_rank, "singular_values": singular.tolist(),
            "rank_tolerance": tolerance, "sigma_min": smallest, "sigma_max": largest,
            "full_numerical_column_rank": numerical_rank == self.n_features_in_,
            "condition_number": largest / smallest if numerical_rank == self.n_features_in_ else None,
            "scope": "Numerical rank is a diagnostic, not an exact rank proof or a converse about full-code information loss.",
        }
        return self

    def scale_inputs(self, x):
        if not hasattr(self, "mean_"):
            raise RuntimeError("Fit the bank before transforming")
        x = _matrix(x)
        if x.shape[1] != len(self.mean_):
            raise ValueError("Query dimension differs from fitted inputs")
        return (x - self.mean_) / self.scale_

    def transform(self, x, *, diagnostics=False):
        if not hasattr(self, "models_"):
            raise RuntimeError("Fit the bank before transforming")
        z = self.scale_inputs(x)
        affinities, gated_coordinates = [], []
        entropy_means, effective_means, center_means, residual_means, reconstruction_means = [], [], [], [], []
        query_stats = []
        for block in self.blocks_:
            count = block["count"]
            scores = np.empty((len(z), count))
            centers_d2 = np.empty_like(scores)
            residuals_d2 = np.empty_like(scores)
            reconstruction_d2 = np.empty_like(scores)
            coordinates = []
            for index, model in enumerate(block["models"]):
                centered = z - model.center
                coordinate = centered @ model.basis
                coordinates.append(coordinate)
                affine = (model.center + coordinate @ model.basis.T
                          if self.gate != "kahm" or diagnostics or model.ae is None else None)
                if diagnostics or self.gate == "locality":
                    centers_d2[:, index] = np.sum(centered ** 2, axis=1)
                    residuals_d2[:, index] = np.sum((z - affine) ** 2, axis=1)
                if self.gate == "kahm" and model.ae is not None:
                    reconstruction_function = (_stable_kahm_reconstruction if self.query_mode == "stable"
                                               else historical_geometry._kahm_reconstruction)
                    reconstruction, weights = reconstruction_function(z, model.ae, diagnostics)
                    if weights is not None:
                        query_stats.append(weights)
                else:
                    reconstruction = affine
                if self.gate == "locality":
                    scores[:, index] = (residuals_d2[:, index] / model.residual_scale
                                         + 0.1 * centers_d2[:, index] / model.scale)
                else:
                    scores[:, index] = historical_geometry.folding_components(z, reconstruction)[0]
                if diagnostics:
                    reconstruction_d2[:, index] = np.sum((z - reconstruction) ** 2, axis=1)
            if self.gate == "locality":
                log_phi = -scores - logsumexp(-scores, axis=1, keepdims=True)
                phi = np.exp(log_phi)
            else:
                phi, _, log_phi = historical_geometry.associations(scores, self.omega, self.tau)
            affinities.append(phi)
            gated_coordinates.extend(phi[:, c:c + 1] * coordinates[c] for c in range(count))
            if diagnostics:
                entropy = -np.sum(phi * log_phi, axis=1)
                entropy_means.append(float(entropy.mean() / np.log(count)) if count > 1 else 0.0)
                effective_means.append(float(np.exp(entropy).mean()))
                center_means.append(float(np.mean(np.sum(phi * centers_d2, axis=1)) / self.n_features_in_))
                residual_means.append(float(np.mean(np.sum(phi * residuals_d2, axis=1)) / self.n_features_in_))
                reconstruction_means.append(float(np.mean(np.sum(phi * reconstruction_d2, axis=1)) / self.n_features_in_))
        phi = np.concatenate(affinities, axis=1)
        features = np.concatenate([phi] + gated_coordinates, axis=1)
        if not np.isfinite(features).all():
            raise FloatingPointError("Nonfinite local features; no replacement representation was used")
        if not diagnostics:
            return features
        condition = self.stack_diagnostics_["condition_number"]
        summary = {
            "stack_rank_fraction": self.stack_diagnostics_["numerical_rank"] / self.n_features_in_,
            "stack_sigma_min": self.stack_diagnostics_["sigma_min"],
            "stack_log10_condition": float(np.log10(condition)) if condition is not None else None,
            "gate_entropy_fraction": float(np.mean(entropy_means)),
            "effective_gate_count": float(np.mean(effective_means)),
            "gate_minimum": float(phi.min()),
            "gate_zero_fraction": float(np.mean(phi == 0)),
            "gated_center_distance_mse": float(np.mean(center_means)),
            "gated_projection_residual_mse": float(np.mean(residual_means)),
            "gated_reconstruction_mse": float(np.mean(reconstruction_means)),
        }
        weights = {
            "weight_rows": sum(s["weight_rows"] for s in query_stats),
            "zero_normalizers": sum(s["zero_normalizers"] for s in query_stats),
            "near_cancelling_normalizers": (sum(s["near_cancelling_normalizers"] for s in query_stats)
                                             if self.gate == "kahm" and self.query_mode == "stable" else None),
            "original_all_zero_kernel_rows": (sum(s["original_all_zero_kernel_rows"] for s in query_stats)
                                              if self.gate == "kahm" and self.query_mode == "stable" else None),
        }
        return features, {"diagnostic_features": summary, "kernel_query_diagnostics": weights,
                          "scope": "Descriptive diagnostics for the supplied queries, not population or prediction guarantees."}

    def metadata(self):
        if not hasattr(self, "models_"):
            raise RuntimeError("Fit the bank first")
        return {
            "gate": self.gate, "upstream_commit": historical_geometry.UPSTREAM_COMMIT,
            "config": {"seed": self.seed, "cluster_counts": list(self.cluster_counts),
                       "rank": self.rank, "max_refs": self.max_refs, "max_fit": self.max_fit,
                       "omega": self.omega, "tau": self.tau},
            "query_mode": self.query_mode if self.gate == "kahm" else None,
            "n_training_rows": self.n_training_rows_, "input_dimension": self.n_features_in_,
            "fit_indices": self.fit_indices_.tolist(), "training_input_sha256": self.training_input_sha256_,
            "input_mean": self.mean_.tolist(), "input_scale": self.scale_.tolist(),
            "stored_gates": self.n_gates_, "simplex_affine_dimension_upper_bound": self.n_gates_ - len(self.blocks_),
            "projected_coordinates": self.n_projected_coordinates_, "local_feature_width": self.n_features_out_,
            "fitted_kahm_kernels": sum(m.ae is not None for m in self.models_),
            "stack": copy.deepcopy(self.stack_diagnostics_),
            "local_models": [{"reference_indices": m.reference_indices.tolist(),
                               "reference_shape": list(m.references.shape),
                               "reference_sha256": historical_geometry._array_hash(m.references),
                               "basis_shape": list(m.basis.shape),
                               "basis_sha256": historical_geometry._array_hash(m.basis),
                               "kernel_matrix_shape": list(m.ae["B"].shape) if m.ae is not None else None,
                               "requested_rank": m.requested_rank, "effective_rank": m.basis.shape[1],
                               "fallback_reason": m.fallback_reason}
                              for m in self.models_],
        }


class ForecastRegressor:
    """A complete regression pipeline, with explicit optional alpha selection.

    ``fit(X, Y)`` accepts one or multiple targets. ``predict`` returns the same
    target dimensionality convention (a vector for one-dimensional training Y,
    otherwise a matrix). ``fit_select`` accepts a declared alpha grid and a
    separate validation set, and returns every trial's validation MSE while
    retaining the first minimum. It fits preprocessing on training data only.
    For an RBF head, a finite Nyström basis is concatenated with linear input
    features and standardized, exactly as in the historical study; its penalty
    is an ordinary Euclidean ridge penalty, not the canonical RKHS norm.
    """

    def __init__(self, representation="raw", head="ridge", *, raw_skip=False,
                 alpha=1.0, n_landmarks=64, bandwidth_multiplier=1.0, seed=0,
                 head_seed=1, geometry_parameters=None, query_mode="stable"):
        if representation not in REPRESENTATIONS or head not in HEADS:
            raise ValueError(f"Use representation in {REPRESENTATIONS} and head in {HEADS}")
        if representation == "raw" and raw_skip:
            raise ValueError("raw_skip is only defined for a local representation")
        self.representation, self.head, self.raw_skip = representation, head, bool(raw_skip)
        self.alpha, self.bandwidth_multiplier = float(alpha), float(bandwidth_multiplier)
        self.n_landmarks, self.seed, self.head_seed = int(n_landmarks), int(seed), int(head_seed)
        self.query_mode = query_mode
        self.geometry_parameters = dict(DEFAULT_GEOMETRY_PARAMETERS)
        self.geometry_parameters.update(dict(geometry_parameters or {}))
        # Geometry scaling is fixed to training-standardized inputs for this study.
        if self.geometry_parameters.pop("scaling", "standard") != "standard":
            raise ValueError("This study fixes geometry scaling to training-standardized inputs")
        if min(self.alpha, self.bandwidth_multiplier) <= 0 or not np.isfinite([self.alpha, self.bandwidth_multiplier]).all():
            raise ValueError("alpha and bandwidth_multiplier must be finite and positive")
        if self.n_landmarks < 1 or min(self.seed, self.head_seed) < 0:
            raise ValueError("Need a positive landmark count and nonnegative seeds")
        if query_mode not in {"stable", "historical"}:
            raise ValueError("query_mode must be stable or historical")

    def _fit_system(self, x, y):
        """Fit training-only preprocessing and return a local ridge eigensystem.

        Training features, design matrices, and eigensystems are deliberately
        returned as local values rather than retained in the serialized model.
        """
        x = _matrix(x)
        y = np.asarray(y, dtype=float)
        self.single_target_ = y.ndim == 1
        if self.single_target_:
            y = y[:, None]
        y = _matrix(y, name="target")
        if len(x) != len(y):
            raise ValueError("Training inputs and targets have different row counts")
        self.n_training_rows_, self.n_features_in_ = x.shape
        self.n_targets_ = y.shape[1]
        self.target_mean_, self.target_scale_ = _standardizer(y, 1e-12)
        standardized_target = (y - self.target_mean_) / self.target_scale_
        self.bank_ = None
        if self.representation == "raw":
            self.input_mean_, self.input_scale_ = _standardizer(x, 1e-12)
            features = (x - self.input_mean_) / self.input_scale_
            self.feature_names_ = [f"raw_{j}" for j in range(x.shape[1])]
            self.geometry_diagnostics_ = {"diagnostic_features": {}, "kernel_query_diagnostics": None}
        else:
            gate = {"affine_local": "affine", "locality_local": "locality", "kahm_local": "kahm"}[self.representation]
            self.bank_ = LocalFeatureBank(gate, seed=self.seed, query_mode=self.query_mode,
                                          **self.geometry_parameters).fit(x)
            self.input_mean_, self.input_scale_ = self.bank_.mean_.copy(), self.bank_.scale_.copy()
            features, self.geometry_diagnostics_ = self.bank_.transform(x, diagnostics=True)
            self.feature_names_ = ([f"gate_{j}" for j in range(self.bank_.n_gates_)]
                                   + [f"local_{j}_{k}" for j, model in enumerate(self.bank_.models_)
                                      for k in range(model.basis.shape[1])])
            if self.raw_skip:
                features = np.column_stack(((x - self.input_mean_) / self.input_scale_, features))
                self.feature_names_ = [f"raw_{j}" for j in range(x.shape[1])] + self.feature_names_
        self.feature_width_ = features.shape[1]
        self.basis_ = historical_heads._bases(features, self.head, self.head_seed,
                                             self.n_landmarks, (self.bandwidth_multiplier,))[0]
        basis_values = self.basis_.transform(features)
        self.basis_mean_, self.basis_scale_ = historical_heads.fit_standardizer(basis_values)
        design = (basis_values - self.basis_mean_) / self.basis_scale_
        self.standardized_target_mean_ = standardized_target.mean(axis=0)
        centered_target = standardized_target - self.standardized_target_mean_
        gram = (design.T @ design) / len(design)
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        rhs = eigenvectors.T @ ((design.T @ centered_target) / len(design))
        self.basis_width_ = design.shape[1]
        self.retained_nystrom_rank_ = (0 if self.basis_.inverse_sqrt is None
                                      else self.basis_.inverse_sqrt.shape[1])
        self.diagnostic_features_ = {
            "training_rows": float(len(x)), "input_dimension": float(x.shape[1]),
            "feature_dimension": float(self.feature_width_),
            "retained_basis_dimension": float(self.basis_width_),
            **self.geometry_diagnostics_["diagnostic_features"],
        }
        return eigenvalues, eigenvectors, rhs

    @staticmethod
    def _solve_alpha(eigenvalues, eigenvectors, rhs, alpha):
        coefficients = eigenvectors @ (rhs / (eigenvalues + alpha)[:, None])
        if not np.isfinite(coefficients).all():
            raise FloatingPointError("Nonfinite fitted ridge coefficients")
        return coefficients

    def fit(self, x, y):
        """Fit one complete fixed configuration using training observations."""
        eigenvalues, eigenvectors, rhs = self._fit_system(x, y)
        self.coefficients_ = self._solve_alpha(eigenvalues, eigenvectors, rhs, self.alpha)
        self.tuning_scope_ = "One complete fixed configuration; no validation or test data were used by fit."
        return self

    def fit_select(self, x, y, x_validation, y_validation, alphas):
        """Select alpha by original-unit validation MSE within one method.

        Geometry, training and validation features, the finite head basis, and
        the Gram eigensystem are each constructed once. Every candidate uses
        identical training-only scalers and basis state. Ties retain the first
        supplied alpha. The returned trial list contains all candidates in the
        supplied order, and this object holds the selected fitted predictor.
        No training/validation design, eigensystem, or trial cache is retained.
        """
        alphas = tuple(float(alpha) for alpha in alphas)
        if not alphas or not np.isfinite(alphas).all() or min(alphas) <= 0:
            raise ValueError("alphas must be a nonempty sequence of finite positive values")
        x_validation = _matrix(x_validation, name="validation input")
        y_validation = np.asarray(y_validation, dtype=float)
        if y_validation.ndim == 1:
            y_validation = y_validation[:, None]
        y_validation = _matrix(y_validation, name="validation target")
        if len(x_validation) != len(y_validation):
            raise ValueError("Validation inputs and targets have different row counts")
        eigenvalues, eigenvectors, rhs = self._fit_system(x, y)
        if y_validation.shape[1] != self.n_targets_:
            raise ValueError("Validation target dimension differs from fitted targets")
        validation_features = self._transform_features(x_validation)
        validation_basis = self.basis_.transform(validation_features)
        validation_design = (validation_basis - self.basis_mean_) / self.basis_scale_
        trials, best_loss, best_alpha, best_coefficients = [], float("inf"), None, None
        for alpha in alphas:
            coefficients = self._solve_alpha(eigenvalues, eigenvectors, rhs, alpha)
            prediction = ((validation_design @ coefficients + self.standardized_target_mean_)
                          * self.target_scale_ + self.target_mean_)
            mse = float(np.mean((prediction - y_validation) ** 2))
            if not np.isfinite(mse):
                raise FloatingPointError("Nonfinite validation MSE; no alpha candidate was silently dropped")
            trials.append({"alpha": alpha, "validation_mse": mse})
            if mse < best_loss:
                best_loss, best_alpha, best_coefficients = mse, alpha, coefficients
        self.alpha, self.coefficients_ = best_alpha, best_coefficients
        self.tuning_scope_ = ("Explicit validation selection of alpha within one method; geometry, input/target/feature "
                              "scaling, basis, and Gram eigensystem use training observations only. "
                              "No test observations were used and no cache is shared between methods.")
        return trials

    def transform_features(self, x):
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Fit the model before prediction")
        return self._transform_features(x)

    def _transform_features(self, x):
        x = _matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Query dimension differs from fitted inputs")
        if self.representation == "raw":
            return (x - self.input_mean_) / self.input_scale_
        local = self.bank_.transform(x)
        if self.raw_skip:
            raw = (x - self.input_mean_) / self.input_scale_
            return np.column_stack((raw, local))
        return local

    def predict(self, x):
        features = self.transform_features(x)
        basis_values = self.basis_.transform(features)
        standardized_prediction = (((basis_values - self.basis_mean_) / self.basis_scale_)
                                   @ self.coefficients_ + self.standardized_target_mean_)
        prediction = standardized_prediction * self.target_scale_ + self.target_mean_
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite predictions")
        return prediction[:, 0] if self.single_target_ else prediction

    def metadata(self):
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Fit the model before requesting metadata")
        return {
            "representation": self.representation, "head": self.head, "raw_skip": self.raw_skip,
            "seed": self.seed, "head_seed": self.head_seed,
            "alpha": self.alpha, "bandwidth_multiplier": self.bandwidth_multiplier if self.head == "rbf" else None,
            "requested_landmarks": self.n_landmarks if self.head == "rbf" else 0,
            "actual_landmarks": 0 if self.basis_.landmarks is None else len(self.basis_.landmarks),
            "retained_nystrom_rank": self.retained_nystrom_rank_,
            "landmark_shape": None if self.basis_.landmarks is None else list(self.basis_.landmarks.shape),
            "nystrom_transform_shape": None if self.basis_.inverse_sqrt is None else list(self.basis_.inverse_sqrt.shape),
            "input_dimension": self.n_features_in_, "target_dimension": self.n_targets_,
            "n_training_rows": self.n_training_rows_, "feature_width": self.feature_width_,
            "basis_width": self.basis_width_, "coefficient_shape": list(self.coefficients_.shape),
            "decoder_coefficient_count": int(self.coefficients_.size + self.n_targets_),
            "decoder_count_scope": "Ridge coefficients plus target intercepts; does not count unsupervised basis or geometry state as fitted supervised coefficients.",
            "feature_names": self.feature_names_.copy(),
            "input_mean": self.input_mean_.tolist(), "input_scale": self.input_scale_.tolist(),
            "target_mean": self.target_mean_.tolist(), "target_scale": self.target_scale_.tolist(),
            "raw_inputs_explicitly_retained": self.representation == "raw" or self.raw_skip,
            "diagnostic_features": copy.deepcopy(self.diagnostic_features_),
            "training_geometry_diagnostics": copy.deepcopy(self.geometry_diagnostics_),
            "geometry": None if self.bank_ is None else self.bank_.metadata(),
            "tuning_scope": self.tuning_scope_,
        }
