"""Forecast-directed local reconstruction, with matched controls and held-out D.

The input standardizer, partitions, reference rows, PCA bases and local scale
floors are inherited from the pinned ``RobustLocalFeatureBank`` affine path.
KAHM factors reproduce its original regularization iteration.  The new model
changes kernel bandwidth (in both reference and query kernels), not the fitted
regularizer.  The original coordinates remain the readout coordinates.

For each horizon and ridge penalty, B minimizes the final forecast error of
the fixed all-A baseline plus an all-A residual head.  C selects the penalty
and correction multiplier.  D labels are deliberately absent from this API.
All new KAHM arms use the same signed-denominator safety rule.  Failed C
candidates remain in the log; failed D predictions return the fixed baseline
for the entire affected horizon.  The archived exploratory pilot is unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import pickle
import time
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingRegressor

from extensions.target_geometry.models import RobustLocalFeatureBank
from extensions.predictive_geometry.study import geometry as historical_geometry


GATES = ("kahm", "affine", "locality", "rbf_local")
ARMS = ("fixed", "gate", "reconstruction")
METHODS = ("ridge", "raw_rbf", "raw_hgb", "seasonal_naive") + tuple(
    f"{arm}_{gate}" for arm in ARMS for gate in GATES) + ("nonkahm_policy",)
NONKAHM_METHODS = ("ridge", "raw_rbf", "raw_hgb", "seasonal_naive") + tuple(
    f"{arm}_{gate}" for arm in ARMS for gate in GATES if gate != "kahm")
DEFAULT_CONFIG = {
    "alpha_grid": [0.01, 0.1, 1.0],
    "gamma_grid": [0.0, 0.25, 0.5, 1.0],
    "baseline_alpha_grid": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "blocks": [2, 4, 8], "rank": 2, "max_references": 24,
    "bank_row_cap": 512, "parameter_bound": 4.0,
    "relative_denominator_tolerance": 1e-10,
    "max_objective_evaluations": 256, "optimizer_maxiter": 40,
    "optimizer_ftol": 1e-10, "optimizer_gtol": 1e-6,
    "optimizer_maxls": 20,
    "raw_rbf_landmarks": 64, "raw_rbf_bandwidth_factors": [0.5, 1.0, 2.0],
    "raw_hgb_learning_rates": [0.05, 0.1], "raw_hgb_max_leaf_nodes": [7, 15],
    "raw_hgb_max_iter": 100, "raw_hgb_l2_regularization": 1.0,
}
MODEL_CONFIG = DEFAULT_CONFIG


@contextmanager
def timed(nodes, name):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        record = nodes.setdefault(name, {"wall_seconds": 0.0, "cpu_seconds": 0.0,
                                         "calls": 0})
        record["wall_seconds"] += time.perf_counter() - wall
        record["cpu_seconds"] += time.process_time() - cpu
        record["calls"] += 1


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


def _scale_fit(x):
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    return mean, np.where(scale > 1e-12, scale, 1.0)


def _solve_spd(a, b):
    return cho_solve(cho_factor(a, lower=True, check_finite=False), b,
                     check_finite=False)


def _ridge_fit(x, y, alpha):
    """Minimize SUM squared errors + alpha ||beta||^2, with free intercept.

    Input columns are standardized on this fit's A rows. Targets remain in
    their original units. This explicitly retains the exploratory pilot's
    penalty convention, not the upstream mean-SSE RidgeHead convention.
    """
    mean, scale = _scale_fit(x)
    xs, ymean = (x - mean) / scale, np.mean(y, axis=0)
    coefficient = _solve_spd(xs.T @ xs + float(alpha) * np.eye(xs.shape[1]),
                             xs.T @ (y - ymean))
    if not np.isfinite(coefficient).all():
        raise FloatingPointError("Nonfinite ridge coefficients")
    return {"mean": mean, "scale": scale, "ymean": ymean,
            "coef": coefficient, "alpha": float(alpha)}


def _ridge_predict(model, x):
    prediction = model["ymean"] + ((x - model["mean"]) / model["scale"]) @ model["coef"]
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite ridge prediction")
    return prediction


def _finite_loss(y, prediction):
    with np.errstate(over="ignore", invalid="ignore"):
        loss = float(np.mean(np.square(y - prediction)))
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite squared loss")
    return loss


def _folding_logits(z, reconstruction):
    score = historical_geometry.folding_components(z, reconstruction)[0]
    return 16.0 * np.log(1.0 - np.clip(score, 0.0, 1.0) + 1e-6)


def _guarded_reconstruction(kernel, inverse, references, tolerance):
    weights = kernel @ inverse
    denominator = weights.sum(axis=1)
    l1 = np.abs(weights).sum(axis=1)
    relative = np.divide(np.abs(denominator), l1, out=np.zeros_like(l1), where=l1 > 0)
    minimum = float(np.min(relative))
    if not np.isfinite(weights).all() or not np.isfinite(relative).all():
        raise FloatingPointError("Nonfinite KAHM weights or relative denominator")
    unsafe = relative <= tolerance
    if np.any(unsafe):
        raise FloatingPointError(
            f"KAHM signed denominator guard: {int(np.sum(unsafe))} rows; "
            f"minimum relative denominator {minimum:.17g} <= {tolerance:.17g}")
    reconstruction = (weights @ references) / denominator[:, None]
    if not np.isfinite(reconstruction).all():
        raise FloatingPointError("Nonfinite KAHM reconstruction")
    return reconstruction, {"minimum_relative_denominator": minimum,
                             "rows": len(kernel),
                             "negative_weight_entries": int(np.count_nonzero(weights < 0))}


class SharedBank:
    """A-only, source-compatible references and coordinates; variable gates.

    Parameter vectors are [xi_1,...,xi_B,eta_1,...,eta_B]. The regularizer,
    covariance metric, reference positions and PCA coordinates never change.
    Working caches are excluded from the serialized deployment state.
    """

    def __init__(self, seed=0, config=None):
        self.seed = int(seed)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.nodes = {}
        self.diagnostics = {"warnings": [], "local_models": [],
                            "fixed_logit_failures": []}
        self._factor_cache = {}

    def fit(self, x):
        with timed(self.nodes, "bank_structure"):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                source = RobustLocalFeatureBank(
                    "affine", seed=self.seed, cluster_counts=tuple(self.config["blocks"]),
                    rank=int(self.config["rank"]), max_refs=int(self.config["max_references"]),
                    max_fit=int(self.config["bank_row_cap"])).fit(x)
            self.diagnostics["warnings"] = [str(item.message) for item in seen]
            self.mean, self.scale = source.mean_.copy(), source.scale_.copy()
            self.models, self.block_slices = [], []
            for b, block in enumerate(source.blocks_):
                start = len(self.models)
                for model in block["models"]:
                    self.models.append({
                        "q": model.references.astype(np.float64),
                        "center": model.center.copy(), "p": model.basis.copy(),
                        "rank": model.basis.shape[1], "block": b,
                        "radius": model.scale, "projection_scale": model.residual_scale,
                        "reference_indices": model.reference_indices.copy(),
                        "fallback_reason": model.fallback_reason,
                    })
                self.block_slices.append((start, len(self.models)))
            self.diagnostics.update({
                "lineage": "Pinned RobustLocalFeatureBank affine partitions/references/PCA",
                "source_training_input_sha256": source.training_input_sha256_,
                "requested_blocks": list(self.config["blocks"]),
                "effective_blocks": list(source.cluster_counts),
                "group_reductions": source.group_reductions_,
                "feature_width": source.n_features_out_,
                "reference_dtype": "source float32 storage, float64 arithmetic",
                "reconstruction_regularizer_frozen": True,
                "local_coordinates_frozen": True,
                "relative_denominator_tolerance": self.config["relative_denominator_tolerance"],
                "stack_diagnostics": source.stack_diagnostics_,
            })
        with timed(self.nodes, "kahm_factors"):
            for index, model in enumerate(self.models):
                record = {"model": index, "block": model["block"], "rank": model["rank"],
                          "reference_count": len(model["q"]),
                          "reference_indices": model["reference_indices"].tolist(),
                          "fallback_reason": model["fallback_reason"]}
                if model["rank"]:
                    try:
                        self._fit_kahm_factor(model, record)
                    except Exception as exc:
                        model["factor_error"] = repr(exc)
                        record["factor_error"] = repr(exc)
                self.diagnostics["local_models"].append(record)
        return self

    @staticmethod
    def _fit_kahm_factor(model, record):
        q, p = model["q"], model["p"]
        projected = (p.T @ q.T)
        covariance = np.atleast_2d(np.cov(projected.T, rowvar=False, bias=False))
        precision = np.linalg.inv(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(precision)
        if not np.isfinite(eigenvalues).all() or np.min(eigenvalues) <= 0:
            raise FloatingPointError("Nonpositive projected precision")
        whitening = (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
        kernel = historical_geometry._upstream("kxx_matrix").kxx_matrix(
            projected, precision, "Gaussian")
        # Preserve the source iteration and the lambda actually used for B;
        # its last proposed update is not the lambda of the returned inverse.
        nu = float(np.sum(q.T ** 2) / q.size)
        lam, change, iteration = 2.5 * nu, 1.0, 0
        next_lam = lam
        while change > 1e-3 and iteration < 100:
            used_lam = lam
            inverse = np.linalg.inv(used_lam * np.eye(len(q)) + kernel)
            residual = q.T - q.T @ (kernel @ inverse).T
            next_lam = 2.0 * nu + float(np.sum(residual ** 2) / q.size)
            change = abs(next_lam - lam)
            lam = next_lam
            iteration += 1
        if iteration == 0 or not np.isfinite(inverse).all() or used_lam <= 0:
            raise FloatingPointError("Invalid original KAHM regularization factor")
        model.update({"whitening": whitening, "uq": (whitening @ projected).T,
                      "inverse": inverse, "lambda_original": used_lam,
                      "reference_distance": -np.log(kernel)})
        record.update({"regularizer": used_lam, "next_regularizer": next_lam,
                       "iterations": iteration, "converged": change <= 1e-3})

    def prepare(self, x, split_name=None):
        x = _matrix(x, "query")
        if x.shape[1] != len(self.mean):
            raise ValueError("Query dimension differs from fitted bank")
        common_name = f"transform_common_{split_name}" if split_name else "prediction_common"
        with timed(self.nodes, common_name):
            z = (x - self.mean) / self.scale
            local = []
            for model in self.models:
                centered = z - model["center"]
                coordinate = centered @ model["p"]
                tangent = coordinate @ model["p"].T
                normal = centered - tangent
                local.append({"coordinate": coordinate, "tangent": tangent,
                              "affine": model["center"] + tangent,
                              "center_d2": np.sum(centered ** 2, axis=1),
                              "tangent_d2": np.sum(coordinate ** 2, axis=1),
                              "normal_d2": np.sum(normal ** 2, axis=1)})
        kernel_name = f"query_distances_kahm_{split_name}" if split_name else "prediction_kernel_distances"
        with timed(self.nodes, kernel_name):
            distance_matrix = historical_geometry._upstream("distance_matrix").distance_matrix
            for model, values in zip(self.models, local):
                if model["rank"] and "whitening" in model and not model.get("factor_error"):
                    projected = (model["whitening"] @ (model["p"].T @ z.T)).T
                    values["query_distance"] = distance_matrix(projected, model["uq"]) / (2.0 * model["rank"])
        return {"z": z, "local": local, "fixed_logits": {}, "last_logits": {},
                "query_diagnostics": {}, "split": split_name}

    def precompute_fixed(self, prepared, gate, split_name):
        with timed(self.nodes, f"fixed_logits_{gate}_{split_name}"):
            try:
                prepared["fixed_logits"][gate] = self._compute_logits(
                    prepared, gate, np.zeros(len(self.block_slices)))
            except Exception as exc:
                prepared["fixed_logits"][gate] = repr(exc)
                self.diagnostics["fixed_logit_failures"].append(
                    {"gate": gate, "split": split_name, "exception": repr(exc)})

    def _inverse(self, index, xi):
        model = self.models[index]
        if model.get("factor_error"):
            raise FloatingPointError(model["factor_error"])
        if xi == 0.0:
            return model["inverse"]
        previous = self._factor_cache.get(index)
        if previous is not None and previous[0] == xi:
            return previous[1]
        kernel = np.exp(-model["reference_distance"] * math.exp(-2.0 * xi))
        inverse = _solve_spd(kernel + model["lambda_original"] * np.eye(len(kernel)),
                             np.eye(len(kernel)))
        if not np.isfinite(inverse).all():
            raise FloatingPointError("Nonfinite bandwidth-specific KAHM inverse")
        self._factor_cache[index] = (xi, inverse)
        return inverse

    def _compute_logits(self, prepared, gate, xi):
        z, columns = prepared["z"], []
        query_record = {"minimum_relative_denominator": 1.0, "rows": 0,
                        "negative_weight_entries": 0}
        for index, (model, values) in enumerate(zip(self.models, prepared["local"])):
            shape = float(xi[model["block"]])
            if gate == "kahm":
                if model["rank"] == 0:
                    reconstruction = np.broadcast_to(model["center"], z.shape)
                else:
                    inverse = self._inverse(index, shape)
                    logs = -values["query_distance"] * math.exp(-2.0 * shape)
                    if not np.isfinite(logs).all():
                        raise FloatingPointError("Nonfinite analytic Gaussian log kernel")
                    kernel = np.exp(logs - logs.max(axis=1, keepdims=True))
                    reconstruction, stats = _guarded_reconstruction(
                        kernel, inverse, model["q"], self.config["relative_denominator_tolerance"])
                    query_record["minimum_relative_denominator"] = min(
                        query_record["minimum_relative_denominator"], stats["minimum_relative_denominator"])
                    query_record["rows"] += stats["rows"]
                    query_record["negative_weight_entries"] += stats["negative_weight_entries"]
                column = _folding_logits(z, reconstruction)
            elif gate == "affine":
                reconstruction = model["center"] + math.exp(shape) * values["tangent"]
                column = _folding_logits(z, reconstruction)
            elif gate == "locality":
                column = (-math.exp(shape) * values["normal_d2"] / model["projection_scale"]
                          - 0.1 * math.exp(-shape) * values["center_d2"] / model["radius"])
            elif gate == "rbf_local":
                # At zero use the source expression exactly; tangent + normal
                # equals center distance in exact arithmetic for these PCA bases.
                distance = (values["center_d2"] if shape == 0.0 else
                            math.exp(shape) * values["tangent_d2"]
                            + math.exp(-shape) * values["normal_d2"])
                column = -0.5 * distance / model["radius"]
            else:
                raise ValueError(f"Unknown gate {gate}")
            columns.append(column)
        logits = np.column_stack(columns)
        if not np.isfinite(logits).all():
            raise FloatingPointError("Nonfinite local logits")
        if gate == "kahm":
            prepared["query_diagnostics"][tuple(float(v) for v in xi)] = query_record
        return logits

    def features(self, prepared, gate, xi=None, eta=None):
        blocks = len(self.block_slices)
        xi = np.zeros(blocks) if xi is None else np.asarray(xi, dtype=float)
        eta = np.zeros(blocks) if eta is None else np.asarray(eta, dtype=float)
        if xi.shape != (blocks,) or eta.shape != (blocks,) or not np.isfinite(np.r_[xi, eta]).all():
            raise ValueError("Need one finite xi and eta per partition block")
        key = tuple(float(v) for v in xi)
        if not np.any(xi) and gate in prepared["fixed_logits"]:
            logits = prepared["fixed_logits"][gate]
            if isinstance(logits, str):
                raise FloatingPointError(logits)
        elif prepared["last_logits"].get(gate, (None,))[0] == key:
            logits = prepared["last_logits"][gate][1]
        else:
            logits = self._compute_logits(prepared, gate, xi)
            prepared["last_logits"][gate] = (key, logits)
        phi = np.empty_like(logits)
        for b, (start, end) in enumerate(self.block_slices):
            scaled = math.exp(float(eta[b])) * logits[:, start:end]
            phi[:, start:end] = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
        features = np.column_stack([phi] + [phi[:, j:j + 1] * values["coordinate"]
                                             for j, values in enumerate(prepared["local"])])
        if not np.isfinite(features).all():
            raise FloatingPointError("Nonfinite gated local features")
        return features

    def state(self, gate):
        models = []
        excluded = {"reference_indices", "fallback_reason"}
        if gate != "kahm":
            excluded |= {"q", "whitening", "uq", "inverse", "lambda_original", "reference_distance", "factor_error"}
        for model in self.models:
            models.append({key: value for key, value in model.items() if key not in excluded})
        return {"mean": self.mean, "scale": self.scale, "models": models,
                "block_slices": self.block_slices, "config": {
                    "relative_denominator_tolerance": self.config["relative_denominator_tolerance"]}}

    @classmethod
    def from_state(cls, state):
        bank = cls(config=state["config"])
        bank.mean, bank.scale = state["mean"], state["scale"]
        bank.models, bank.block_slices = state["models"], state["block_slices"]
        return bank


class _EvaluationBudget(Exception):
    pass


def _train_parameters(bank, prepared, gate, arm, residual_a, target_b,
                      baseline_b, alpha, target_scale, config):
    blocks = len(bank.block_slices)
    dimension = blocks if arm == "gate" else 2 * blocks
    initial = np.zeros(dimension)
    budget = int(config["max_objective_evaluations"])
    evaluations, exceptions = [], []
    best = {"loss": float("inf"), "theta": initial.copy(), "b_mse": None}

    def unpack(theta):
        return ((np.zeros(blocks), theta) if arm == "gate" else (theta[:blocks], theta[blocks:]))

    def objective(theta):
        if len(evaluations) >= budget:
            raise _EvaluationBudget()
        xi, eta = unpack(np.asarray(theta))
        error, mse = None, None
        try:
            fa = bank.features(prepared["A"], gate, xi, eta)
            fb = bank.features(prepared["B"], gate, xi, eta)
            head = _ridge_fit(fa, residual_a, alpha)
            mse = _finite_loss(target_b, baseline_b + _ridge_predict(head, fb))
            value = mse / target_scale
        except Exception as exc:
            error, value = repr(exc), 1e100
            exceptions.append(error)
        evaluations.append({"xi": xi.tolist(), "eta": eta.tolist(),
                            "normalized_b_mse": float(value), "b_mse": mse,
                            "exception": error})
        if value < best["loss"]:
            best.update(loss=value, theta=np.asarray(theta).copy(), b_mse=mse)
        return value

    initial_loss = objective(initial)
    status, nit, message, converged = "single_evaluation", 0, "", False
    if budget > 1:
        bound = math.log(float(config["parameter_bound"]))
        try:
            result = minimize(objective, initial, method="L-BFGS-B",
                              bounds=[(-bound, bound)] * dimension,
                              options={"maxiter": int(config["optimizer_maxiter"]),
                                       "maxfun": budget, "ftol": float(config["optimizer_ftol"]),
                                       "gtol": float(config["optimizer_gtol"]),
                                       "maxls": int(config["optimizer_maxls"])})
            status, nit, message = str(result.status), int(result.nit), str(result.message)
            converged = bool(result.success)
        except _EvaluationBudget:
            status, message = "evaluation_budget", "Exact actual-objective evaluation cap reached"
        except Exception as exc:
            status, message = "optimizer_exception", repr(exc)
            exceptions.append(message)
    valid = best["b_mse"] is not None
    xi, eta = unpack(best["theta"])
    record = {"alpha": float(alpha), "arm": arm, "parameter_count": dimension,
              "optimizer_method": "L-BFGS-B with numerical forward differences",
              "parameter_order": "eta by block" if arm == "gate" else "xi by block, then eta by block",
              "initial_normalized_b_mse": float(initial_loss),
              "best_normalized_b_mse": float(best["loss"]), "best_b_mse": best["b_mse"],
              "xi": xi.tolist(), "eta": eta.tolist(), "nfev": len(evaluations),
              "budget": budget, "nit": nit, "status": status, "message": message,
              "converged": converged, "valid": valid, "exceptions": exceptions,
              "evaluations": evaluations}
    return (xi, eta) if valid else None, record


def _baseline_ridge(xa, ya, groups, queries, config, train_mask, validation_mask):
    if (train_mask is None) != (validation_mask is None):
        raise ValueError("Supply both inner-A masks or neither")
    if train_mask is None:
        unique = np.unique(groups)
        if len(unique) < 2:
            raise ValueError("At least two A episode groups or explicit chronological inner masks required")
        count = max(1, min(len(unique) - 1, int(math.floor(0.8 * len(unique)))))
        fit, validation = np.isin(groups, unique[:count]), np.isin(groups, unique[count:])
        split_record = {"inner_split": "whole episodes, first 80 percent of ordered group IDs",
                        "inner_training_groups": unique[:count].tolist(),
                        "inner_validation_groups": unique[count:].tolist()}
    else:
        fit, validation = np.asarray(train_mask), np.asarray(validation_mask)
        if fit.dtype != bool or validation.dtype != bool:
            raise ValueError("Inner-A masks must be boolean")
        if fit.shape != (len(xa),) or validation.shape != fit.shape or np.any(fit & validation):
            raise ValueError("Inner-A masks must have the right shape and be disjoint")
        split_record = {"inner_split": "caller-supplied chronological target-maturity-purged masks"}
    if not np.any(fit) or not np.any(validation):
        raise ValueError("Inner-A training and validation sets must be nonempty")
    candidates, best = [], None
    for alpha in config["baseline_alpha_grid"]:
        model = _ridge_fit(xa[fit], ya[fit], alpha)
        loss = _finite_loss(ya[validation], _ridge_predict(model, xa[validation]))
        candidates.append({"alpha": float(alpha), "inner_a_mse": loss})
        if best is None or loss < best[0]:
            best = loss, float(alpha)
    model = _ridge_fit(xa, ya, best[1])
    prediction = {name: _ridge_predict(model, x) for name, x in queries.items()}
    return prediction, {"selected_alpha": best[1], "candidates": candidates,
                        "selection_split": "inner A", "final_fit_split": "all A",
                        "baseline_held_fixed": True,
                        "inner_training_rows": np.flatnonzero(fit).tolist(),
                        "inner_validation_rows": np.flatnonzero(validation).tolist(),
                        "inner_purged_rows": np.flatnonzero(~(fit | validation)).tolist(),
                        **split_record}, {"kind": "ridge", "head": model}


def _raw_rbf(xa, ya, xb, yb, xc, seed, config):
    rng = np.random.default_rng(seed)
    mean, scale = _scale_fit(xa)
    za, zb = (xa - mean) / scale, (xb - mean) / scale
    ids = np.sort(rng.choice(len(xa), min(int(config["raw_rbf_landmarks"]), len(xa)), replace=False))
    landmarks = za[ids]
    pair = cdist(landmarks, landmarks, metric="sqeuclidean")
    positive = pair[pair > 0]
    base_bandwidth = float(np.median(positive)) if len(positive) else 1.0
    candidates, best, state = [], None, None
    for factor in config["raw_rbf_bandwidth_factors"]:
        bandwidth = base_bandwidth * float(factor)
        eigenvalues, eigenvectors = np.linalg.eigh(np.exp(-pair / (2.0 * bandwidth)))
        keep = eigenvalues > max(1e-12, 1e-10 * float(np.max(eigenvalues)))
        if not np.any(keep):
            raise FloatingPointError("No retained Nystrom directions")
        transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
        fa = np.column_stack((za, np.exp(-cdist(za, landmarks, metric="sqeuclidean") / (2.0 * bandwidth)) @ transform))
        fb = np.column_stack((zb, np.exp(-cdist(zb, landmarks, metric="sqeuclidean") / (2.0 * bandwidth)) @ transform))
        for alpha in config["baseline_alpha_grid"]:
            head = _ridge_fit(fa, ya, alpha)
            loss = _finite_loss(yb, _ridge_predict(head, fb))
            record = {"bandwidth_factor": float(factor), "bandwidth_squared": bandwidth,
                      "alpha": float(alpha), "b_mse": loss, "rank": int(np.sum(keep))}
            candidates.append(record)
            if best is None or loss < best[0]:
                best = loss, record
                state = {"kind": "raw_rbf", "mean": mean, "scale": scale,
                         "landmarks": landmarks, "transform": transform,
                         "bandwidth_squared": bandwidth, "head": head}
    prediction_c = predict_state(state, xc)
    return prediction_c, {"selected": best[1], "candidates": candidates,
                         "landmark_indices": ids.tolist(), "selection_split": "B",
                         "final_fit_split": "A", "raw_plus_nystrom": True}, state


def _raw_hgb(xa, ya, xb, yb, xc, seed, config):
    candidates, best, state = [], None, None
    for rate in config["raw_hgb_learning_rates"]:
        for leaves in config["raw_hgb_max_leaf_nodes"]:
            models = [HistGradientBoostingRegressor(
                learning_rate=float(rate), max_leaf_nodes=int(leaves),
                max_iter=int(config["raw_hgb_max_iter"]),
                l2_regularization=float(config["raw_hgb_l2_regularization"]),
                early_stopping=False, random_state=int(seed)).fit(xa, ya[:, h])
                for h in range(ya.shape[1])]
            loss = _finite_loss(yb, np.column_stack([model.predict(xb) for model in models]))
            record = {"learning_rate": float(rate), "max_leaf_nodes": int(leaves),
                      "max_iter": int(config["raw_hgb_max_iter"]),
                      "l2_regularization": float(config["raw_hgb_l2_regularization"]),
                      "early_stopping": False, "b_mse": loss,
                      "estimator_parameters": models[0].get_params(deep=False)}
            candidates.append(record)
            if best is None or loss < best[0]:
                best, state = (loss, record), {"kind": "raw_hgb", "models": models}
    return predict_state(state, xc), {
        "selected": best[1], "candidates": candidates, "selection_split": "B",
        "selection_unit": "one configuration for the complete multihorizon forecast",
        "fit_split": "A", "target_specific_trees": True}, state


def _raw_rbf_features(state, x):
    z = (x - state["mean"]) / state["scale"]
    kernel = np.exp(-cdist(z, state["landmarks"], metric="sqeuclidean") / (2.0 * state["bandwidth_squared"]))
    result = np.column_stack((z, kernel @ state["transform"]))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite raw RBF query features")
    return result


def _raw_prediction_horizon(state, x, horizon, features=None):
    if state["kind"] == "raw_rbf":
        head = state["head"]
        scalar_head = {**head, "ymean": head["ymean"][horizon], "coef": head["coef"][:, horizon]}
        prediction = _ridge_predict(scalar_head, features)
    elif state["kind"] == "raw_hgb":
        prediction = state["models"][horizon].predict(x)
    else:
        raise ValueError("Raw numerical fallback requires a raw RBF or HGB state")
    if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite or invalid raw prediction")
    return prediction


def _safe_raw_prediction(state, x, baseline, baseline_prediction=None):
    """Preserve C selection; query failures fall back independently by horizon."""
    prediction = (_ridge_predict(baseline, x) if baseline_prediction is None
                  else baseline_prediction).copy()
    failures, features = [], None
    if state["kind"] == "raw_rbf":
        try:
            features = _raw_rbf_features(state, x)
        except Exception as exc:
            failures = [{"horizon": h, "stage": "D_prediction", "exception": repr(exc)}
                        for h in range(prediction.shape[1])]
            return prediction, failures
    for h in range(prediction.shape[1]):
        try:
            prediction[:, h] = _raw_prediction_horizon(state, x, h, features)
        except Exception as exc:
            failures.append({"horizon": h, "stage": "D_prediction", "exception": repr(exc)})
    return prediction, failures


def predict_state(state, x):
    """Replay a selected model; any local failure falls back for a whole horizon."""
    x = _matrix(x, "prediction inputs")
    kind = state["kind"]
    if kind == "ridge":
        return _ridge_predict(state["head"], x)
    if kind == "raw_rbf":
        return _ridge_predict(state["head"], _raw_rbf_features(state, x))
    if kind == "raw_hgb":
        return np.column_stack([model.predict(x) for model in state["models"]])
    if kind == "safe_raw":
        return _safe_raw_prediction(state["model"], x, state["baseline"])[0]
    if kind == "seasonal_naive":
        period = state["period"]
        indices = x.shape[1] - period + ((np.asarray(state["horizons"]) - 1) % period)
        return x[:, indices].copy()
    if kind == "residual_geometry":
        result = _ridge_predict(state["baseline"], x).copy()
        if state["bank"] is None:
            return result
        bank = SharedBank.from_state(state["bank"])
        prepared = bank.prepare(x)
        for h, selected in enumerate(state["selected"]):
            if selected is not None:
                try:
                    features = bank.features(prepared, state["gate"], selected["xi"], selected["eta"])
                    value = result[:, h] + selected["gamma"] * _ridge_predict(selected["head"], features)
                    if not np.isfinite(value).all():
                        raise FloatingPointError("Nonfinite local forecast")
                    result[:, h] = value
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    pass
        return result
    raise ValueError(f"Unknown serialized model kind {kind}")


def predict_serialized(blob, x):
    """Load trusted study-generated pickle bytes and replay forecasts."""
    return predict_state(pickle.loads(blob), x)


def fit_suite(Xa, Ya, groups_a, Xb, Yb, Xc, Yc, Xd, seed, config=None, *,
              A_inner_train_mask=None, A_inner_validation_mask=None,
              horizons=None, naive_period=1):
    """Fit 17 prespecified procedures and return forecasts, logs, and states.

    All target arrays are (rows, horizons); X lags run oldest to newest.
    Natural-series callers must provide target-maturity-purged inner-A masks.
    Synthetic callers use ordered whole-episode groups. Seasonality is a
    declared input, not chosen from C or D. No D labels are accepted here.
    """
    config = copy.deepcopy({**DEFAULT_CONFIG, **(config or {})})
    xa, ya, xb, yb, xc, yc, xd = [_matrix(value, name) for value, name in
        zip((Xa, Ya, Xb, Yb, Xc, Yc, Xd), ("Xa", "Ya", "Xb", "Yb", "Xc", "Yc", "Xd"))]
    if len(xa) != len(ya) or len(xb) != len(yb) or len(xc) != len(yc):
        raise ValueError("Input and target rows must agree in A, B and C")
    if len({xa.shape[1], xb.shape[1], xc.shape[1], xd.shape[1]}) != 1 or len({ya.shape[1], yb.shape[1], yc.shape[1]}) != 1:
        raise ValueError("Input and horizon dimensions must agree across splits")
    groups = np.asarray(groups_a)
    if groups.shape != (len(xa),):
        raise ValueError("groups_a must identify every A row")
    horizons = list(range(1, ya.shape[1] + 1)) if horizons is None else list(horizons)
    if len(horizons) != ya.shape[1] or any(int(h) != h or h < 1 for h in horizons):
        raise ValueError("horizons must contain one positive integer per target column")
    if int(naive_period) != naive_period or not 1 <= naive_period <= xa.shape[1]:
        raise ValueError("naive_period must be a positive integer no larger than the lag window")
    if int(config["max_objective_evaluations"]) < 1 or float(config["parameter_bound"]) <= 1:
        raise ValueError("Positive objective budget and parameter_bound > 1 required")
    if not config["gamma_grid"] or float(config["gamma_grid"][0]) != 0.0:
        raise ValueError("Gamma grid must start with the exact zero incumbent")
    if any(float(a) <= 0 for a in config["alpha_grid"] + config["baseline_alpha_grid"]):
        raise ValueError("Ridge penalties must be positive")
    diagnostics = {"config": config, "seed": int(seed), "methods": {}, "cost_nodes": {},
                   "scope": "Forecast-directed reconstruction with frozen coordinates and fixed final baseline",
                   "ridge_penalty_convention": "sum squared errors + alpha * squared coefficient norm; unpenalized intercept; A feature scaling; targets unscaled",
                   "geometry_objective": "B MSE of final all-A baseline plus all-A residual head, gamma=1; divided by all-A target variance",
                   "target_horizons": [int(h) for h in horizons], "target_horizon_count": ya.shape[1],
                   "d_failure_granularity": "whole horizon on the supplied D batch",
                   "failures": []}
    nodes = diagnostics["cost_nodes"]
    with timed(nodes, "baseline_ridge"):
        base, base_record, base_state = _baseline_ridge(
            xa, ya, groups, {"A": xa, "B": xb, "C": xc, "D": xd}, config,
            A_inner_train_mask, A_inner_validation_mask)
    pc, pd, states = {"ridge": base["C"].copy()}, {"ridge": base["D"].copy()}, {"ridge": base_state}
    base_record["cost_dependencies"] = ["baseline_ridge"]
    diagnostics["methods"]["ridge"] = base_record
    for name, fit_function, offset in (("raw_rbf", _raw_rbf, 7919), ("raw_hgb", _raw_hgb, 104729)):
        with timed(nodes, name):
            try:
                prediction_c, record, raw_state = fit_function(xa, ya, xb, yb, xc, seed + offset, config)
            except Exception as exc:
                prediction_c, prediction_d = base["C"].copy(), base["D"].copy()
                record, state = {"fallback": "ridge", "exception": repr(exc)}, base_state
                diagnostics["failures"].append({"method": name, "exception": repr(exc)})
            else:
                prediction_d, failures = _safe_raw_prediction(raw_state, xd, base_state["head"], base["D"])
                record["fallbacks"] = failures
                record["d_fallback_horizons"] = [failure["horizon"] for failure in failures]
                record["baseline_for_numerical_fallback"] = True
                diagnostics["failures"].extend({"method": name, **failure} for failure in failures)
                state = {"kind": "safe_raw", "model": raw_state, "baseline": base_state["head"]}
        pc[name], pd[name], states[name] = prediction_c, prediction_d, state
        record["cost_dependencies"] = [name, "baseline_ridge"]
        diagnostics["methods"][name] = record
    with timed(nodes, "seasonal_naive"):
        states["seasonal_naive"] = {"kind": "seasonal_naive", "period": int(naive_period),
                                    "horizons": [int(h) for h in horizons]}
        pc["seasonal_naive"] = predict_state(states["seasonal_naive"], xc)
        pd["seasonal_naive"] = predict_state(states["seasonal_naive"], xd)
    diagnostics["methods"]["seasonal_naive"] = {
        "period": int(naive_period), "prespecified": True, "cost_dependencies": ["seasonal_naive"]}

    bank, bank_failure, prepared = SharedBank(seed=seed + 1543, config=config), None, {}
    try:
        bank.fit(xa)
        for split, x in (("A", xa), ("B", xb), ("C", xc), ("D", xd)):
            prepared[split] = bank.prepare(x, split)
            for gate in GATES:
                bank.precompute_fixed(prepared[split], gate, split)
    except Exception as exc:
        bank_failure = repr(exc)
        diagnostics["failures"].append({"method": "shared_bank", "exception": bank_failure})
    nodes.update(bank.nodes)
    diagnostics["bank"] = bank.diagnostics
    residual_a = ya - base["A"]
    target_scale = np.maximum(np.var(ya, axis=0), 1e-12)
    blocks = len(config["blocks"])
    for arm in ARMS:
        for gate in GATES:
            name = f"{arm}_{gate}"
            pc[name], pd[name] = base["C"].copy(), base["D"].copy()
            selected_states = []
            record = {"horizons": [], "fallbacks": [], "gate": gate, "arm": arm,
                      "fit_split": "A", "geometry_fit_split": "B" if arm != "fixed" else None,
                      "selection_split": "C", "parameter_count_per_horizon":
                          0 if arm == "fixed" else blocks if arm == "gate" else 2 * blocks,
                      "parameters_per_alpha_candidate": True,
                      "capacity_note": "Matched parameter counts and budgets do not imply identical functional capacity"}
            with timed(nodes, f"heads_{name}"):
                for h in range(ya.shape[1]):
                    selected = {"candidate_id": "zero", "alpha": None, "gamma": 0.0,
                                "xi": np.zeros(blocks).tolist(), "eta": np.zeros(blocks).tolist(),
                                "c_mse": _finite_loss(yc[:, h], base["C"][:, h]), "fallback": None}
                    candidates, optimizations = [dict(selected)], []
                    best_c, best_d = base["C"][:, h].copy(), base["D"][:, h].copy()
                    selected_state = None
                    if bank_failure:
                        selected["fallback"] = "shared_bank_failure"
                        record["fallbacks"].append({"horizon": h, "exception": bank_failure})
                    for alpha in config["alpha_grid"]:
                        xi = eta = None
                        try:
                            if bank_failure:
                                raise FloatingPointError(bank_failure)
                            if arm == "fixed":
                                xi, eta = np.zeros(blocks), np.zeros(blocks)
                            else:
                                parameters, optimization = _train_parameters(
                                    bank, prepared, gate, arm, residual_a[:, h], yb[:, h],
                                    base["B"][:, h], alpha, target_scale[h], config)
                                optimizations.append(optimization)
                                if parameters is None:
                                    raise FloatingPointError("No finite B objective evaluated")
                                xi, eta = parameters
                            fa = bank.features(prepared["A"], gate, xi, eta)
                            fc = bank.features(prepared["C"], gate, xi, eta)
                            head = _ridge_fit(fa, residual_a[:, h], alpha)
                            correction = _ridge_predict(head, fc)
                            for gamma in config["gamma_grid"]:
                                if float(gamma) == 0.0:
                                    continue
                                prediction = base["C"][:, h] + float(gamma) * correction
                                loss = _finite_loss(yc[:, h], prediction)
                                candidate = {"candidate_id": f"alpha={float(alpha):g},gamma={float(gamma):g}",
                                             "alpha": float(alpha), "gamma": float(gamma),
                                             "xi": xi.tolist(), "eta": eta.tolist(), "c_mse": loss,
                                             "fallback": None}
                                candidates.append(candidate)
                                if loss < selected["c_mse"]:
                                    selected, best_c = dict(candidate), prediction.copy()
                                    selected_state = {"head": head, "xi": xi.copy(), "eta": eta.copy(),
                                                      "gamma": float(gamma)}
                        except Exception as exc:
                            failure = {"horizon": h, "alpha": float(alpha), "stage": "A_B_C_candidate",
                                       "exception": repr(exc)}
                            record["fallbacks"].append(failure)
                            diagnostics["failures"].append({"method": name, **failure})
                            for gamma in config["gamma_grid"]:
                                if float(gamma) != 0.0:
                                    candidates.append({"candidate_id": f"alpha={float(alpha):g},gamma={float(gamma):g}",
                                                       "alpha": float(alpha), "gamma": float(gamma),
                                                       "xi": None if xi is None else xi.tolist(),
                                                       "eta": None if eta is None else eta.tolist(),
                                                       "c_mse": None, "fallback": repr(exc)})
                    deployment_fallback, query_diagnostics = None, None
                    if selected_state is not None:
                        try:
                            fd = bank.features(prepared["D"], gate, selected_state["xi"], selected_state["eta"])
                            best_d = base["D"][:, h] + selected_state["gamma"] * _ridge_predict(selected_state["head"], fd)
                            if not np.isfinite(best_d).all():
                                raise FloatingPointError("Nonfinite D forecast")
                            if gate == "kahm":
                                key = tuple(float(v) for v in selected_state["xi"])
                                query_diagnostics = prepared["D"]["query_diagnostics"].get(key)
                        except Exception as exc:
                            best_d = base["D"][:, h].copy()
                            deployment_fallback = repr(exc)
                            failure = {"horizon": h, "stage": "D_prediction", "exception": repr(exc)}
                            record["fallbacks"].append(failure)
                            diagnostics["failures"].append({"method": name, **failure})
                    pc[name][:, h], pd[name][:, h] = best_c, best_d
                    selected_states.append(selected_state)
                    record["horizons"].append({"horizon_index": h, "horizon": int(horizons[h]),
                                               "selected": selected, "candidates": candidates,
                                               "optimizations": optimizations,
                                               "deployment_fallback": deployment_fallback,
                                               "d_query_diagnostics": query_diagnostics})
            with timed(nodes, f"serialize_{name}"):
                states[name] = {"kind": "residual_geometry", "gate": gate,
                                "baseline": base_state["head"], "selected": selected_states,
                                "bank": bank.state(gate) if any(s is not None for s in selected_states) else None}
            dependencies = ["baseline_ridge", "bank_structure", f"heads_{name}"]
            if gate == "kahm":
                dependencies.append("kahm_factors")
            for split in (("A", "C", "D") if arm == "fixed" else ("A", "B", "C", "D")):
                dependencies += [f"transform_common_{split}", f"fixed_logits_{gate}_{split}"]
                if gate == "kahm":
                    dependencies.append(f"query_distances_kahm_{split}")
            record["cost_dependencies"] = dependencies
            diagnostics["methods"][name] = record

    # The non-KAHM policy selects one complete procedure; never an oracle
    # assembled using D outcomes or separate test-selected horizon winners.
    with timed(nodes, "nonkahm_policy_selection"):
        selected_name, selected_loss, policy_candidates = None, float("inf"), []
        for name in NONKAHM_METHODS:
            loss = _finite_loss(yc, pc[name])
            policy_candidates.append({"method": name, "c_mse": loss})
            if loss < selected_loss:
                selected_name, selected_loss = name, loss
        pc["nonkahm_policy"], pd["nonkahm_policy"] = pc[selected_name].copy(), pd[selected_name].copy()
        states["nonkahm_policy"] = states[selected_name]
    policy_dependencies = sorted(set(
        dependency for name in NONKAHM_METHODS
        for dependency in diagnostics["methods"][name]["cost_dependencies"]))
    diagnostics["methods"]["nonkahm_policy"] = {
        "selected_method": selected_name, "candidates": policy_candidates,
        "selection_split": "C", "selection_unit": "whole multihorizon procedure",
        "tie_rule": "first strict minimum in declared order, ridge first",
        "cost_dependencies": policy_dependencies + ["nonkahm_policy_selection"],
        "search_cost_includes_all_candidate_methods": True}

    serialized = {}
    for name in METHODS:
        with timed(nodes, f"serialize_{name}"):
            serialized[name] = pickle.dumps(states[name], protocol=pickle.HIGHEST_PROTOCOL)
        record = diagnostics["methods"][name]
        dependencies = record["cost_dependencies"]
        record["charged_fit_wall_seconds"] = sum(nodes[k]["wall_seconds"] for k in set(dependencies) if k in nodes)
        record["charged_fit_cpu_seconds"] = sum(nodes[k]["cpu_seconds"] for k in set(dependencies) if k in nodes)
        record["serialization_cost_node"] = f"serialize_{name}"
        record["serialized_bytes"] = len(serialized[name])
        record["charged_wall_seconds"] = record["charged_fit_wall_seconds"] + nodes[f"serialize_{name}"]["wall_seconds"]
        record["charged_cpu_seconds"] = record["charged_fit_cpu_seconds"] + nodes[f"serialize_{name}"]["cpu_seconds"]
        record["selected_c_mse"] = _finite_loss(yc, pc[name])
    diagnostics["cost_scope"] = (
        "Measured fitting/selection and query computation, with shared nodes charged once per method; "
        "policy cost is the union of every searched non-KAHM procedure's dependencies. "
        "Selected-state packing and pickle serialization are a separate measured node; "
        "raw-model state construction is included in its fit. Artifact file writes are excluded. "
        "Serialized bytes measure model state, not peak resident memory or streaming update cost.")
    return {"predictions_c": {name: pc[name] for name in METHODS},
            "predictions_d": {name: pd[name] for name in METHODS},
            "diagnostics": diagnostics, "serialized_models": serialized}
