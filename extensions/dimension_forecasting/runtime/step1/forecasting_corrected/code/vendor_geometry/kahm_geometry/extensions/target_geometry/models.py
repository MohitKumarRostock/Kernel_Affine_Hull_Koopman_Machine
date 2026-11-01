"""Training-only predictive metrics and robust matched local representations.

The learned diagonal metric minimizes a leave-group-out Gaussian kernel
regression proxy for supplied residual targets. It is shared by the downstream
KAHM, affine, locality and RBF-local controls; it is *not* end-to-end KAHM
optimization. Its lineage is supervised metric learning for kernel regression.

All preprocessing is fitted from explicitly supplied training observations.
The caller owns chronological/episode residual construction, validation,
zero-correction selection, final prediction, and complete cost accounting.
Archived implementations are imported read-only.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from extensions.natural_forecasting.models import LocalFeatureBank, _standardizer


def _matrix(value: Any, name: str = "input") -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


def _target(value: Any, rows: int) -> tuple[np.ndarray, bool]:
    value = np.asarray(value, dtype=np.float64)
    single = value.ndim == 1
    value = _matrix(value[:, None] if single else value, "target")
    if len(value) != rows:
        raise ValueError("Input and target row counts differ")
    return value, single


def _hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value, dtype=np.float64)
    h = hashlib.sha256(str(value.shape).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def _metric_objective(eta: np.ndarray, squared_differences: np.ndarray,
                      residuals: np.ndarray, admissible: np.ndarray,
                      bandwidth_squared: float, regularization: float):
    """Return proxy loss and its analytic gradient in bounded free parameters.

    theta = eta - mean(eta) removes global metric scale. The bandwidth is
    computed once at the identity metric and held fixed during optimization.
    Queries with no admissible donors are excluded from this training proxy.
    """
    theta = eta - np.mean(eta)
    weights = np.exp(theta)
    logs = -0.5 * np.einsum("ijk,k->ij", squared_differences, weights) / bandwidth_squared
    logs = np.where(admissible, logs, -np.inf)
    query_ids = np.flatnonzero(np.any(admissible, axis=1))
    selected_logs = logs[query_ids]
    probabilities = np.exp(selected_logs - logsumexp(selected_logs, axis=1, keepdims=True))
    predicted = probabilities @ residuals
    error = predicted - residuals[query_ids]
    loss = float(np.mean(error * error) + regularization * np.mean(theta * theta))
    # d loss / d logit_ij, with row-softmax centering included.
    affinity_gradient = (2.0 / error.size) * probabilities * (
        error @ residuals.T - np.sum(error * predicted, axis=1, keepdims=True)
    )
    gradient_theta = (-0.5 / bandwidth_squared) * weights * np.einsum(
        "ij,ijk->k", affinity_gradient, squared_differences[query_ids]
    )
    gradient_theta += 2.0 * regularization * theta / len(theta)
    gradient_eta = gradient_theta - np.mean(gradient_theta)
    return loss, gradient_eta


class LearnedDiagonalMetric:
    """Bounded diagonal metric fitted by training residual prediction.

    The supplied residuals are centered and divided by ONE pooled standard
    deviation. Thus their relative horizon weights remain those of original-
    unit average squared error; this is not separate horizon standardization.
    Input scales may use a larger training-only ``scaling_data`` matrix so a
    metric fitted on correction rows agrees with banks fitted on all training
    inputs. Neither validation nor test data belongs in this argument.

    ``groups`` excludes donors in the query's group, e.g. an entire independent
    episode. ``positions`` and ``exclusion_radius`` can additionally exclude
    nearby natural-series origins. Donors later within the training period are
    allowed: this is offline supervised metric fitting, not a prequential
    forecast. Residual construction must itself obey the desired causality.

    Free parameters eta lie in +/- log(max_weight_ratio)/2. Centering eta gives
    sum(theta)=0 and guarantees max(exp(theta))/min(exp(theta)) <= the declared
    ratio. Individual centered theta bounds are NOT the eta box bounds.
    """

    def __init__(self, *, max_rows=256, max_iter=40, regularization=0.05,
                 max_weight_ratio=16.0, seed=0):
        self.max_rows, self.max_iter = int(max_rows), int(max_iter)
        self.regularization = float(regularization)
        self.max_weight_ratio, self.seed = float(max_weight_ratio), int(seed)
        if self.max_rows < 2 or self.max_iter < 1 or self.seed < 0:
            raise ValueError("Need max_rows >= 2, max_iter >= 1 and nonnegative seed")
        if not np.isfinite([self.regularization, self.max_weight_ratio]).all():
            raise ValueError("Metric hyperparameters must be finite")
        if self.regularization < 0 or self.max_weight_ratio < 1:
            raise ValueError("Need nonnegative regularization and weight ratio >= 1")

    def fit(self, x, residuals, *, groups=None, positions=None,
            exclusion_radius=0, scaling_data=None):
        x = _matrix(x)
        target, _ = _target(residuals, len(x))
        scale_x = x if scaling_data is None else _matrix(scaling_data, "scaling data")
        if scale_x.shape[1] != x.shape[1]:
            raise ValueError("Scaling data and metric input dimensions differ")
        radius = float(exclusion_radius)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("exclusion_radius must be finite and nonnegative")
        groups = np.arange(len(x)) if groups is None else np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(x):
            raise ValueError("groups must have one label per metric observation")
        if positions is not None:
            positions = np.asarray(positions, dtype=np.float64)
            if positions.shape != (len(x),) or not np.isfinite(positions).all():
                raise ValueError("positions must be finite and aligned with metric rows")
        elif radius:
            raise ValueError("positions are required for nonzero exclusion_radius")

        self.n_features_in_, self.n_training_rows_ = x.shape[1], len(x)
        self.mean_, self.scale_ = _standardizer(scale_x, 1e-12)
        self.input_sha256_, self.residual_sha256_ = _hash(x), _hash(target)
        self.scaling_input_sha256_ = _hash(scale_x)
        rng = np.random.default_rng(self.seed)
        self.fit_indices_ = (np.sort(rng.choice(len(x), self.max_rows, replace=False))
                             if len(x) > self.max_rows else np.arange(len(x)))
        ids = self.fit_indices_
        z = (x[ids] - self.mean_) / self.scale_
        centered = target - np.mean(target, axis=0)
        self.residual_pooled_scale_ = float(np.sqrt(np.mean(centered * centered)))
        self.residual_mean_ = np.mean(target, axis=0)
        admissible = groups[ids, None] != groups[None, ids]
        np.fill_diagonal(admissible, False)
        if positions is not None:
            admissible &= np.abs(positions[ids, None] - positions[None, ids]) > radius
        donor_counts = admissible.sum(axis=1)
        self.weights_ = np.ones(x.shape[1])
        self.theta_ = np.zeros(x.shape[1])
        self.bandwidth_squared_ = None
        self.initial_objective_ = self.final_objective_ = None
        self.optimization_success_ = False
        self.optimization_message_ = "not attempted"
        self.optimization_iterations_ = self.optimization_evaluations_ = 0
        self.fallback_reason_ = None
        self.fit_group_count_ = int(len(np.unique(groups[ids])))
        self.valid_query_count_ = int(np.count_nonzero(donor_counts))
        self.donor_counts_ = donor_counts
        self.exclusion_radius_ = radius
        self.uses_position_exclusion_ = positions is not None
        if self.fit_group_count_ < 2 or self.valid_query_count_ < 2:
            self.fallback_reason_ = "fewer than two donor groups or admissible queries"
            return self
        if self.residual_pooled_scale_ <= 1e-12:
            self.fallback_reason_ = "constant residual target; no centered predictive signal"
            return self

        delta = z[:, None, :] - z[None, :, :]
        squared = delta * delta
        euclidean_squared = np.sum(squared, axis=2)
        positive_distances = euclidean_squared[admissible & (euclidean_squared > 1e-12)]
        if not len(positive_distances):
            self.fallback_reason_ = "no positive admissible input distance"
            return self
        self.bandwidth_squared_ = float(np.median(positive_distances))
        normalized_target = centered[ids] / self.residual_pooled_scale_
        args = (squared, normalized_target, admissible,
                self.bandwidth_squared_, self.regularization)
        start = np.zeros(x.shape[1])
        self.initial_objective_ = float(_metric_objective(start, *args)[0])
        self.final_objective_ = self.initial_objective_
        half_box = 0.5 * np.log(self.max_weight_ratio)
        if half_box == 0:
            self.optimization_message_ = "fixed identity by max_weight_ratio=1"
            self.optimization_success_ = True
            return self
        try:
            result = minimize(_metric_objective, start, args=args, method="L-BFGS-B",
                              jac=True, bounds=[(-half_box, half_box)] * len(start),
                              options={"maxiter": self.max_iter, "ftol": 1e-10,
                                       "gtol": 1e-6, "maxls": 20})
            self.optimization_success_ = bool(result.success)
            self.optimization_message_ = str(result.message)
            self.optimization_iterations_ = int(result.nit)
            self.optimization_evaluations_ = int(result.nfev)
            final_value, final_gradient = _metric_objective(np.asarray(result.x), *args)
            theta = np.asarray(result.x) - np.mean(result.x)
            if (not np.isfinite(final_value) or not np.isfinite(final_gradient).all()
                    or not np.isfinite(theta).all()):
                self.fallback_reason_ = "nonfinite metric optimization result"
            elif final_value > self.initial_objective_ + 1e-12:
                self.fallback_reason_ = "optimizer did not reduce its training objective"
            else:
                # A finite improved iterate is retained on an iteration limit;
                # convergence status is reported rather than claimed silently.
                self.theta_, self.weights_ = theta, np.exp(theta)
                self.final_objective_ = float(final_value)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            self.fallback_reason_ = f"metric optimization error: {type(error).__name__}: {error}"
        return self

    def transform(self, x):
        if not hasattr(self, "weights_"):
            raise RuntimeError("Fit the metric first")
        x = _matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Metric query dimension differs from fitted input")
        return ((x - self.mean_) / self.scale_) * np.sqrt(self.weights_)

    def metadata(self):
        if not hasattr(self, "weights_"):
            raise RuntimeError("Fit the metric first")
        return {
            "method": "training_leave_group_out_gaussian_residual_metric",
            "config": {"max_rows": self.max_rows, "max_iter": self.max_iter,
                       "regularization": self.regularization,
                       "max_weight_ratio": self.max_weight_ratio, "seed": self.seed},
            "n_training_rows": self.n_training_rows_,
            "input_dimension": self.n_features_in_,
            "fit_indices": self.fit_indices_.tolist(),
            "input_sha256": self.input_sha256_, "residual_sha256": self.residual_sha256_,
            "scaling_input_sha256": self.scaling_input_sha256_,
            "input_mean": self.mean_.tolist(), "input_scale": self.scale_.tolist(),
            "residual_mean": self.residual_mean_.tolist(),
            "residual_pooled_scale": self.residual_pooled_scale_,
            "weights": self.weights_.tolist(), "theta": self.theta_.tolist(),
            "realized_weight_ratio": float(self.weights_.max() / self.weights_.min()),
            "bandwidth_squared": self.bandwidth_squared_,
            "bandwidth_scope": "identity-metric median positive admissible squared distance, fixed during optimization",
            "residual_scale_scope": "one pooled training residual standard deviation, preserving relative horizon MSE weights",
            "groups": self.fit_group_count_, "valid_queries": self.valid_query_count_,
            "donor_counts": self.donor_counts_.tolist(),
            "position_exclusion": self.uses_position_exclusion_,
            "exclusion_radius": self.exclusion_radius_,
            "initial_objective": self.initial_objective_,
            "final_objective": self.final_objective_,
            "optimizer_success": self.optimization_success_,
            "optimizer_message": self.optimization_message_,
            "iterations": self.optimization_iterations_,
            "evaluations": self.optimization_evaluations_,
            "fallback_reason": self.fallback_reason_,
            "scope": "Training-only supervised kernel proxy shared across geometries; not end-to-end KAHM optimization or test performance evidence.",
        }


class RobustLocalFeatureBank(LocalFeatureBank):
    """Inherited local codes with an explicit metric and robust group counts.

    Metric weights apply AFTER the training standardizer, so standardization
    does not cancel learned anisotropy. Identical seeds and weights give the
    geometric controls identical partitions and reference-selection rules.

    The additional ``rbf`` control uses exp(-||z-center||^2/(2 local_scale))
    normalized within each block and the same gated PCA coordinates. It is a
    variable-bandwidth RBF-local affine representation, not raw RBF regression.
    """

    def __init__(self, gate="kahm", *, metric_weights=None, **parameters):
        if gate not in {"kahm", "affine", "locality", "rbf"}:
            raise ValueError("gate must be kahm, affine, locality or rbf")
        self.requested_gate = gate
        self.metric_weights = None if metric_weights is None else np.asarray(metric_weights, dtype=float).copy()
        super().__init__(gate="affine" if gate == "rbf" else gate, **parameters)
        self.requested_cluster_counts = tuple(self.cluster_counts)

    def scale_inputs(self, x):
        standardized = super().scale_inputs(x)
        return standardized * np.sqrt(self.metric_weights_)

    def fit(self, x):
        x = _matrix(x)
        weights = np.ones(x.shape[1]) if self.metric_weights is None else self.metric_weights.copy()
        if weights.shape != (x.shape[1],) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("metric_weights must be finite positive values, one per input column")
        self.metric_weights_ = weights
        rng = np.random.default_rng(self.seed)
        ids = (np.sort(rng.choice(len(x), self.max_fit, replace=False))
               if self.max_fit is not None and len(x) > self.max_fit else np.arange(len(x)))
        self.partition_unique_rows_ = int(len(np.unique(x[ids], axis=0)))
        self.cluster_counts = tuple(min(c, self.partition_unique_rows_) for c in self.requested_cluster_counts)
        self.group_reductions_ = []
        if self.cluster_counts != self.requested_cluster_counts:
            self.group_reductions_.append({"reason": "available distinct partition rows",
                                           "from": list(self.requested_cluster_counts),
                                           "to": list(self.cluster_counts)})
        while True:
            try:
                super().fit(x)
                break
            except ValueError as error:
                if "Fewer nonempty clusters than requested" not in str(error) or max(self.cluster_counts) == 1:
                    raise
                previous = self.cluster_counts
                self.cluster_counts = tuple(max(1, c // 2) for c in previous)
                self.group_reductions_.append({"reason": "numerically empty kmeans group",
                                               "from": list(previous), "to": list(self.cluster_counts)})
        return self

    def transform(self, x, *, diagnostics=False):
        if self.requested_gate != "rbf":
            return super().transform(x, diagnostics=diagnostics)
        if not hasattr(self, "models_"):
            raise RuntimeError("Fit the bank first")
        z = self.scale_inputs(x)
        affinities, gated_coordinates = [], []
        for block in self.blocks_:
            logs, coordinates = [], []
            for model in block["models"]:
                centered = z - model.center
                logs.append(-0.5 * np.sum(centered * centered, axis=1) / model.scale)
                coordinates.append(centered @ model.basis)
            logs = np.column_stack(logs)
            phi = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
            affinities.append(phi)
            gated_coordinates.extend(phi[:, j:j + 1] * coordinate
                                     for j, coordinate in enumerate(coordinates))
        result = np.column_stack(affinities + gated_coordinates)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite RBF-local features")
        if diagnostics:
            return result, {"gate": "rbf", "kernel_query_diagnostics": None,
                            "scope": "Variable-bandwidth center RBF gates and matched local coordinates."}
        return result

    def metadata(self):
        result = super().metadata()
        result.update(gate=self.requested_gate,
                      requested_cluster_counts=list(self.requested_cluster_counts),
                      realized_cluster_counts=list(self.cluster_counts),
                      partition_unique_rows=self.partition_unique_rows_,
                      group_reductions=copy.deepcopy(self.group_reductions_),
                      metric_weights=self.metric_weights_.tolist(),
                      metric_application="after training input standardization")
        return result


class RidgeHead:
    """Training-standardized ridge readout with unpenalized intercept.

    Objective: mean squared standardized-target error plus
    alpha * ||coefficients||_F^2. All alpha candidates in ``fit_path`` reuse
    the same training-only eigensystem and preprocessing. Validation selection
    is intentionally external, so the caller can retain complete trial logs.
    """

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")

    @classmethod
    def fit_path(cls, features, targets, alphas):
        x = _matrix(features, "features")
        y, single = _target(targets, len(x))
        alphas = tuple(float(a) for a in alphas)
        if not alphas or not np.isfinite(alphas).all() or min(alphas) <= 0:
            raise ValueError("alphas must be nonempty finite positive values")
        mean_x, scale_x = _standardizer(x, 1e-12)
        mean_y, scale_y = _standardizer(y, 1e-12)
        design = (x - mean_x) / scale_x
        target = (y - mean_y) / scale_y
        gram = (design.T @ design) / len(design)
        eig, vectors = np.linalg.eigh((gram + gram.T) / 2)
        eig = np.maximum(eig, 0.0)
        rhs = vectors.T @ (design.T @ target / len(design))
        input_hash, target_hash = _hash(x), _hash(y)
        models = {}
        for alpha in alphas:
            model = cls(alpha)
            model.input_mean_, model.input_scale_ = mean_x, scale_x
            model.target_mean_, model.target_scale_ = mean_y, scale_y
            model.coefficients_ = vectors @ (rhs / (eig + alpha)[:, None])
            model.n_features_in_, model.n_targets_, model.n_training_rows_ = x.shape[1], y.shape[1], len(x)
            model.single_target_ = single
            model.input_sha256_, model.target_sha256_ = input_hash, target_hash
            models[alpha] = model
        return models

    def fit(self, features, targets):
        fitted = self.fit_path(features, targets, (self.alpha,))[self.alpha]
        self.__dict__.update(fitted.__dict__)
        return self

    def predict(self, features):
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Fit the ridge head first")
        x = _matrix(features, "features")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from fitted head")
        prediction = (((x - self.input_mean_) / self.input_scale_) @ self.coefficients_)
        prediction = prediction * self.target_scale_ + self.target_mean_
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite ridge predictions")
        return prediction[:, 0] if self.single_target_ else prediction

    def metadata(self):
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Fit the ridge head first")
        return {"alpha": self.alpha, "training_rows": self.n_training_rows_,
                "feature_dimension": self.n_features_in_, "target_dimension": self.n_targets_,
                "coefficient_count": int(self.coefficients_.size + self.n_targets_),
                "input_sha256": self.input_sha256_, "target_sha256": self.target_sha256_,
                "input_mean": self.input_mean_.tolist(), "input_scale": self.input_scale_.tolist(),
                "target_mean": self.target_mean_.tolist(), "target_scale": self.target_scale_.tolist()}
