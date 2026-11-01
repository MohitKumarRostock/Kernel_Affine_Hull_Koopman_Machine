"""Frozen-input reconstruction mechanisms with independently selected readouts.

Only A fits state, B selects a hyperparameter for each horizon, and C selects
each horizon's shrinkage multiplier and one complete non-KAHM procedure.  The
fit API has no reserved-outcome argument.  ``predict`` never fits or selects.

All heads minimize n**-1 ||Y - 1 b' - F W||_F**2 + alpha ||W||_F**2,
with a free intercept.  Thus the penalty is a mean *row* squared-error
penalty, independently for each horizon, not a sum-error or mean-cell-error
penalty.  Geometry columns are not restandardized after their radial maps.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import math
import pickle
import time
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingRegressor

from extensions.direct_reconstruction.model import SharedBank
from extensions.predictive_geometry.study import geometry as historical_geometry


METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common",
           "kahm_departure", "rbf_departure", "kahm_unbounded",
           "rbf_unbounded", "nonkahm_policy")
NONKAHM_METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common",
                  "rbf_departure", "rbf_unbounded")
GEOMETRY_METHODS = ("common", "kahm_departure", "rbf_departure",
                    "kahm_unbounded", "rbf_unbounded")
DEFAULT_CONFIG = {
    "alpha_grid": [1e-4, .001, .01, .1, 1.0],
    "gamma_grid": [0.0, .25, .5, 1.0],
    "blocks": [2, 4, 8], "rank": 2, "max_references": 24,
    "bank_row_cap": 512, "relative_denominator_tolerance": 1e-10,
    "feature_scale_floor": 1e-6, "target_variance_floor": 1e-12,
    "raw_rbf_landmarks": 64, "raw_rbf_bandwidth_factors": [.5, 1., 2.],
    "raw_hgb_learning_rates": [.05, .1], "raw_hgb_max_leaf_nodes": [7, 15],
    "raw_hgb_max_iter": 100, "raw_hgb_l2_regularization": 1.,
}
MODEL_CONFIG = DEFAULT_CONFIG


@contextmanager
def timed(nodes, name):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        record = nodes.setdefault(name, {"wall_seconds": 0., "cpu_seconds": 0., "calls": 0})
        record["wall_seconds"] += time.perf_counter() - wall
        record["cpu_seconds"] += time.process_time() - cpu
        record["calls"] += 1


def _matrix(value, name="matrix"):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


def _scaler_fit(x, floor):
    mean, raw = np.mean(x, axis=0), np.std(x, axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite A-only feature statistics")
    return {"mean": mean, "scale": np.maximum(raw, floor),
            "floor": float(floor), "floored_columns": int(np.count_nonzero(raw < floor))}


def _standardize(state, x):
    with np.errstate(over="ignore", invalid="ignore"):
        z = (x - state["mean"]) / state["scale"]
    if not np.isfinite(z).all():
        raise FloatingPointError("Nonfinite standardized features")
    return z


def radial_bound(v, radius=None):
    """T_B(v) with overflow-safe arithmetic for every finite input row."""
    v = _matrix(v, "radial-map input")
    radius = math.sqrt(v.shape[1]) if radius is None else float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("The radial radius must be positive and finite")
    scale = np.maximum(np.max(np.abs(v), axis=1, keepdims=True), radius)
    scaled = v / scale
    denominator = np.sqrt((radius / scale) ** 2 + np.sum(scaled ** 2, axis=1, keepdims=True))
    return radius * scaled / denominator


def _row_norm(x):
    scale = np.max(np.abs(x), axis=1)
    unit = np.divide(x, scale[:, None], out=np.zeros_like(x), where=scale[:, None] > 0)
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * np.sqrt(np.sum(unit * unit, axis=1))


def _feature_summary(x, radius=None):
    norm = _row_norm(x)
    finite = bool(np.isfinite(x).all() and np.isfinite(norm).all())
    result = {"rows": int(len(x)), "width": int(x.shape[1]), "finite": finite,
              "minimum": float(np.min(x)), "maximum": float(np.max(x)),
              "max_abs": float(np.max(np.abs(x))),
              "max_norm": float(np.max(norm)) if np.isfinite(norm).all() else None}
    if radius is not None:
        result.update(radius=float(radius),
                      violations=int(np.count_nonzero(norm > float(radius) * (1 + 1e-12))))
    return result


def _mse(y, prediction):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.mean((y - prediction) ** 2, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite squared prediction loss")
    return result


def _ridge_path(x, y, alphas):
    """One centered SVD for all penalties; no implicit input scaling."""
    x, y = _matrix(x, "ridge features"), _matrix(y, "ridge targets")
    if len(x) != len(y):
        raise ValueError("Ridge feature and target rows differ")
    mean, ymean = x.mean(axis=0), y.mean(axis=0)
    centered, response = x - mean, y - ymean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    projected = u.T @ response
    rank_tolerance = max(centered.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.)
    rank = int(np.count_nonzero(singular > rank_tolerance))
    response_rms = np.sqrt(np.mean(response * response, axis=0))
    output = []
    for alpha in alphas:
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge penalties must be positive and finite")
        coefficient = vt.T @ ((singular / (singular ** 2 + len(x) * alpha))[:, None] * projected)
        if not np.isfinite(coefficient).all():
            raise FloatingPointError("Nonfinite centered-SVD ridge coefficient")
        output.append({"mean": mean, "ymean": ymean, "coef": coefficient,
                       "alpha_by_horizon": np.full(y.shape[1], alpha),
                       "response_rms": response_rms, "feature_rank": rank,
                       "feature_width": x.shape[1], "training_rows": len(x)})
    return output


def _ridge_predict(head, x):
    prediction = head["ymean"] + (x - head["mean"]) @ head["coef"]
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite centered-SVD ridge prediction")
    return prediction


def _combine_heads(heads, index_by_horizon):
    result = {key: value for key, value in heads[0].items() if key not in ("coef", "alpha_by_horizon")}
    result["coef"] = np.column_stack([heads[int(i)]["coef"][:, h] for h, i in enumerate(index_by_horizon)])
    result["alpha_by_horizon"] = np.array([heads[int(i)]["alpha_by_horizon"][h]
                                           for h, i in enumerate(index_by_horizon)])
    return result


def _scalar_head(head, h):
    # Copies retain only the selected horizon, not a view pinning all unused
    # candidate coefficient columns in the live state.
    return {**head, "ymean": head["ymean"][h:h + 1].copy(), "coef": head["coef"][:, h:h + 1].copy(),
            "alpha_by_horizon": head["alpha_by_horizon"][h:h + 1].copy(),
            "response_rms": head["response_rms"][h:h + 1].copy()}


def _select_ridge(xa, ya, xb, yb, baseline_b, alphas, nodes, prefix):
    with timed(nodes, f"{prefix}_fit_A"):
        heads = _ridge_path(xa, ya, alphas)
    with timed(nodes, f"{prefix}_select_B"):
        losses = np.array([_mse(yb, baseline_b + _ridge_predict(head, xb)) for head in heads])
        indices = np.argmin(losses, axis=0)
        selected = _combine_heads(heads, indices)
    record = {"candidates": [{"alpha": float(alpha), "mse_by_horizon": loss.tolist(),
                               "status": "ok"} for alpha, loss in zip(alphas, losses)],
              "selected": [{"alpha": float(alphas[int(i)]), "mse": float(losses[i, h]),
                             "candidate_index": int(i)} for h, i in enumerate(indices)],
              "fit_split": "A", "selection_split": "B", "selection": "per horizon",
              "feature_rank": int(selected["feature_rank"]),
              "feature_width": int(selected["feature_width"])}
    return selected, record


def _blend(baseline, candidate, gamma):
    """The gamma=0 incumbent is recovered exactly, including its bit pattern."""
    gamma = np.asarray(gamma, dtype=float)
    out = baseline.copy()
    for h in range(baseline.shape[1]):
        if gamma[h] != 0:
            out[:, h] = baseline[:, h] + gamma[h] * (candidate[:, h] - baseline[:, h])
    return out


def _select_gamma(yc, baseline, candidate, gammas, *, failed=False):
    losses = np.array([_mse(yc, _blend(baseline, candidate, np.full(yc.shape[1], gamma)))
                       for gamma in gammas])
    indices = np.zeros(yc.shape[1], dtype=int) if failed else np.argmin(losses, axis=0)
    gamma = np.asarray(gammas, dtype=float)[indices]
    prediction = _blend(baseline, candidate, gamma)
    return gamma, prediction, {
        "candidates": [{"gamma": float(g), "mse_by_horizon": loss.tolist(),
                        "status": "baseline_fallback" if failed and g != 0 else "ok"}
                       for g, loss in zip(gammas, losses)],
        "gamma_by_horizon": gamma.tolist(), "selected_indices": indices.tolist(),
        "selected_mse_by_horizon": _mse(yc, prediction).tolist(),
        "selection_split": "C", "selection": "per horizon",
        "all_positive_candidates_failed": bool(failed)}


class _InstrumentedBank(SharedBank):
    """Preserve the inherited factors; separately charge projected metrics.

    The inherited factor routine recomputes its metric.  That small redundant
    computation is charged to KAHM, so an RBF-only deployment is not charged
    for inverse matrices it does not need.
    """
    def _fit_kahm_factor(self, model, record):
        with timed(self.nodes, "bank_projected_metric"):
            projected = model["p"].T @ model["q"].T
            covariance = np.atleast_2d(np.cov(projected.T, rowvar=False, bias=False))
            precision = np.linalg.inv(covariance)
            values, vectors = np.linalg.eigh(precision)
            if not np.isfinite(values).all() or np.min(values) <= 0:
                raise FloatingPointError("Nonpositive projected precision")
            whitening = (vectors * np.sqrt(values)[None, :]) @ vectors.T
            model.update(whitening=whitening, uq=(whitening @ projected).T)
        with timed(self.nodes, "bank_kahm_factor"):
            SharedBank._fit_kahm_factor(model, record)

    def fit(self, x):
        super().fit(x)
        self.nodes.pop("kahm_factors", None)  # parent aggregate would double-count
        return self


def normalized_reconstruction(kernel, references, *, inverse=None, tolerance=1e-10,
                              fallback=None):
    """Normalized RBF or signed KAHM reconstruction with row-level guards.

    Invalid rows receive the supplied affine reconstruction.  Their gated
    departures therefore equal zero, and their exact indices remain logged.
    """
    kernel, references = _matrix(kernel, "rescaled kernel"), _matrix(references, "references")
    weights = kernel if inverse is None else kernel @ inverse
    denominator = weights.sum(axis=1)
    l1 = np.sum(np.abs(weights), axis=1)
    relative = np.divide(np.abs(denominator), l1, out=np.zeros_like(l1), where=l1 > 0)
    safe = (np.isfinite(weights).all(axis=1) & np.isfinite(relative) & (relative > tolerance))
    if fallback is None:
        fallback = np.zeros((len(kernel), references.shape[1]))
    result = np.asarray(fallback, dtype=float).copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        proposal = np.divide(weights @ references, denominator[:, None],
                             out=np.zeros_like(result), where=denominator[:, None] != 0)
    safe &= np.isfinite(proposal).all(axis=1)
    result[safe] = proposal[safe]
    finite_relative = relative[np.isfinite(relative)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        probability = np.divide(weights, denominator[:, None], out=np.zeros_like(weights),
                                where=denominator[:, None] != 0)
    return result, {"rows": len(kernel), "fallback_rows": np.flatnonzero(~safe).tolist(),
                    "fallback_count": int(np.count_nonzero(~safe)),
                    "minimum_relative_denominator": float(np.min(finite_relative)) if len(finite_relative) else None,
                    "negative_weight_entries": int(np.count_nonzero(weights < 0)),
                    "minimum_normalized_weight": float(np.min(probability[safe])) if np.any(safe) else None,
                    "max_weight_sum_error": float(np.max(np.abs(probability[safe].sum(axis=1) - 1))) if np.any(safe) else None}


def _raw_feature_maps(bank, x, nodes=None, tag="query"):
    nodes = {} if nodes is None else nodes
    local, diagnostics = [], {"local_models": [], "gate_fallback_rows": [],
                              "coordinate_fallback_rows": [], "kernel_fallback_rows": []}
    with timed(nodes, f"common_raw_{tag}"):
        z = (x - bank["mean"]) / bank["scale"]
        logits = np.empty((len(x), len(bank["models"])))
        for j, model in enumerate(bank["models"]):
            with np.errstate(over="ignore", invalid="ignore"):
                centered = z - model["center"]
                coordinate = centered @ model["p"]
                invalid = ~np.isfinite(coordinate).all(axis=1)
                coordinate[invalid] = 0.
                affine = model["center"] + coordinate @ model["p"].T
                logits[:, j] = -.5 * np.sum(centered ** 2, axis=1) / model["radius"]
            if np.any(invalid):
                diagnostics["coordinate_fallback_rows"].append({"model": j, "rows": np.flatnonzero(invalid).tolist()})
            local.append({"coordinate": coordinate, "affine": affine})
        logits[~np.isfinite(logits)] = -np.inf
        gates = np.empty_like(logits)
        for b, (start, stop) in enumerate(bank["block_slices"]):
            scores = logits[:, start:stop].copy()
            failed = ~np.isfinite(scores).any(axis=1)
            scores[failed] = 0.
            gates[:, start:stop] = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
            if np.any(failed):
                diagnostics["gate_fallback_rows"].append({"block": b, "rows": np.flatnonzero(failed).tolist()})
        common = np.column_stack([gates] + [gates[:, j:j + 1] * item["coordinate"]
                                             for j, item in enumerate(local)])
    kernels = []
    with timed(nodes, f"projected_kernel_{tag}"):
        distance_matrix = historical_geometry._upstream("distance_matrix").distance_matrix
        for j, model in enumerate(bank["models"]):
            if not model["rank"] or "whitening" not in model:
                kernels.append(None)
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                projected = (model["whitening"] @ (model["p"].T @ z.T)).T
                logs = -distance_matrix(projected, model["uq"]) / (2. * model["rank"])
                invalid = ~np.isfinite(logs).all(axis=1)
                logs[invalid] = 0.
                kernel = np.exp(logs - np.max(logs, axis=1, keepdims=True))
            kernels.append((kernel, invalid))
            if np.any(invalid):
                diagnostics["kernel_fallback_rows"].append({"model": j, "rows": np.flatnonzero(invalid).tolist()})
    departures = {}
    for kind in ("kahm", "rbf"):
        pieces, affected = [], set()
        with timed(nodes, f"{kind}_raw_{tag}"):
            for j, (model, item, pair) in enumerate(zip(bank["models"], local, kernels)):
                if not model["rank"]:
                    departure = np.zeros_like(item["affine"])
                    record = {"model": j, "kind": kind, "fallback_count": 0,
                              "fallback_rows": [], "rank_zero": True}
                elif pair is None or (kind == "kahm" and "inverse" not in model):
                    departure = np.zeros_like(item["affine"])
                    record = {"model": j, "kind": kind, "fallback_count": len(x),
                              "fallback_rows": list(range(len(x))),
                              "factor_failure": model.get("factor_error", "missing projected metric")}
                else:
                    kernel, invalid = pair
                    estimate, record = normalized_reconstruction(
                        kernel, model["q"], inverse=model["inverse"] if kind == "kahm" else None,
                        tolerance=bank["config"]["relative_denominator_tolerance"], fallback=item["affine"])
                    estimate[invalid] = item["affine"][invalid]
                    rows = sorted(set(record["fallback_rows"]) | set(np.flatnonzero(invalid).tolist()))
                    record.update(model=j, kind=kind, fallback_rows=rows, fallback_count=len(rows))
                    with np.errstate(over="ignore", invalid="ignore"):
                        departure = gates[:, j:j + 1] * (estimate - item["affine"])
                    bad = ~np.isfinite(departure).all(axis=1)
                    if np.any(bad):
                        departure[bad] = 0.
                        rows = sorted(set(rows) | set(np.flatnonzero(bad).tolist()))
                        record.update(fallback_rows=rows, fallback_count=len(rows))
                pieces.append(departure)
                affected.update(record["fallback_rows"])
                diagnostics["local_models"].append(record)
            departures[kind] = np.column_stack(pieces)
        diagnostics[f"{kind}_fallback_rows"] = sorted(affected)
        diagnostics[f"{kind}_fallback_count"] = len(affected)
    diagnostics["gate_block_sum_max_abs_error"] = float(max(
        np.max(np.abs(gates[:, a:b].sum(axis=1) - 1.)) for a, b in bank["block_slices"]))
    diagnostics["raw_features"] = {"common": _feature_summary(common),
                                    **{kind: _feature_summary(value) for kind, value in departures.items()}}
    return {"common": common, **departures}, diagnostics


@dataclass
class ReconstructionFeatures:
    bank: dict
    scalers: dict

    def transform_raw(self, raw, *, nodes=None, tag="query"):
        nodes = {} if nodes is None else nodes
        scaled, bounded = {}, {}
        for block in ("common", "kahm", "rbf"):
            with timed(nodes, f"{block}_scale_{tag}"):
                scaled[block] = _standardize(self.scalers[block], raw[block])
            with timed(nodes, f"{block}_bound_{tag}"):
                bounded[block] = radial_bound(scaled[block])
        features = {"common": bounded["common"]}
        for kind in ("kahm", "rbf"):
            with timed(nodes, f"{kind}_departure_assemble_{tag}"):
                features[f"{kind}_departure"] = np.column_stack((bounded["common"], bounded[kind]))
            with timed(nodes, f"{kind}_unbounded_assemble_{tag}"):
                features[f"{kind}_unbounded"] = np.column_stack((bounded["common"], scaled[kind]))
        summary = {
            "feature_bounds": {"common": _feature_summary(bounded["common"], math.sqrt(bounded["common"].shape[1])),
                               **{f"{k}_departure": _feature_summary(bounded[k], math.sqrt(bounded[k].shape[1]))
                                  for k in ("kahm", "rbf")}},
            "standardized_features": {key: _feature_summary(value) for key, value in scaled.items()},
            "method_features": {key: _feature_summary(value,
                                math.sqrt(value.shape[1]) if not key.endswith("unbounded") else None)
                                for key, value in features.items()}}
        return features, summary

    def transform(self, x, *, nodes=None, tag="query"):
        raw, diagnostics = _raw_feature_maps(self.bank, x, nodes, tag)
        features, transformed = self.transform_raw(raw, nodes=nodes, tag=tag)
        diagnostics.update(transformed)
        return features, diagnostics


def _readout_bound(head, radius, gamma):
    coefficient_bound = head["response_rms"] / (2 * np.sqrt(head["alpha_by_horizon"]))
    actual_norm = np.linalg.norm(head["coef"], axis=0)
    mean_norm = float(np.linalg.norm(head["mean"]))
    bound = (np.abs(head["ymean"]) + (float(radius) + mean_norm) * coefficient_bound
             if radius is not None else None)
    return {"radius": radius, "training_feature_mean_norm": mean_norm,
            "response_mean_by_horizon": head["ymean"].tolist(),
            "centered_response_rms_by_horizon": head["response_rms"].tolist(),
            "alpha_by_horizon": head["alpha_by_horizon"].tolist(),
            "coefficient_norm_by_horizon": actual_norm.tolist(),
            "coefficient_norm_bound_by_horizon": coefficient_bound.tolist(),
            "coefficient_bound_violations": int(np.count_nonzero(actual_norm > coefficient_bound * (1 + 1e-10) + 1e-12)),
            "readout_bound_by_horizon": None if bound is None else bound.tolist(),
            "correction_bound_by_horizon": None if bound is None else (np.asarray(gamma) * bound).tolist(),
            "global_bound": radius is not None}


def _raw_rbf_features(state, z):
    kernel = np.exp(-cdist(z, state["landmarks"], "sqeuclidean") / (2 * state["bandwidth_squared"]))
    return _standardize(state["scaler"], np.column_stack((z, kernel @ state["transform"])))


def _fit_raw_rbf(za, ya, zb, yb, zc, seed, config, nodes):
    with timed(nodes, "raw_rbf_geometry_A"):
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(len(za), min(int(config["raw_rbf_landmarks"]), len(za)), replace=False))
        landmarks = za[ids].copy()
        pair = cdist(landmarks, landmarks, "sqeuclidean")
        positive = pair[pair > 0]
        median = float(np.median(positive)) if len(positive) else 1.
    best_loss, states = np.full(ya.shape[1], np.inf), [None] * ya.shape[1]
    candidates, failures, c_features = [], [], {}
    for factor in config["raw_rbf_bandwidth_factors"]:
        try:
            with timed(nodes, "raw_rbf_geometry_A"):
                bandwidth = median * float(factor)
                eigenvalues, eigenvectors = np.linalg.eigh(np.exp(-pair / (2 * bandwidth)))
                keep = eigenvalues > max(1e-12, 1e-10 * float(np.max(eigenvalues)))
                if not np.any(keep):
                    raise FloatingPointError("No retained Nystrom directions")
                transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
                raw_a = np.column_stack((za, np.exp(-cdist(za, landmarks, "sqeuclidean") / (2 * bandwidth)) @ transform))
                state = {"landmarks": landmarks, "transform": transform, "bandwidth_squared": bandwidth,
                         "scaler": _scaler_fit(raw_a, config["feature_scale_floor"]),
                         "bandwidth_factor": float(factor)}
            with timed(nodes, "raw_rbf_transform_A_B_C"):
                fa = _standardize(state["scaler"], raw_a)
                fb, fc = _raw_rbf_features(state, zb), _raw_rbf_features(state, zc)
            with timed(nodes, "raw_rbf_fit_A"):
                heads = _ridge_path(fa, ya, config["alpha_grid"])
            with timed(nodes, "raw_rbf_select_B"):
                for head, alpha in zip(heads, config["alpha_grid"]):
                    loss = _mse(yb, _ridge_predict(head, fb))
                    index = len(candidates)
                    record = {"bandwidth_factor": float(factor), "bandwidth_squared": bandwidth,
                              "alpha": float(alpha), "nystrom_rank": int(np.sum(keep)),
                              "mse_by_horizon": loss.tolist(), "status": "ok"}
                    candidates.append(record)
                    for h in range(ya.shape[1]):
                        if loss[h] < best_loss[h]:
                            best_loss[h] = loss[h]
                            states[h] = {"features": state, "head": _scalar_head(head, h),
                                         "candidate_index": index}
                c_features[float(factor)] = fc
        except Exception as exc:
            record = {"bandwidth_factor": float(factor), "status": "failed", "exception": repr(exc)}
            failures.append(record)
            for alpha in config["alpha_grid"]:
                candidates.append({**record, "alpha": float(alpha), "mse_by_horizon": None})
    if any(state is None for state in states):
        raise FloatingPointError(f"No usable raw RBF candidate: {failures!r}")
    with timed(nodes, "raw_rbf_predict_C"):
        prediction = np.column_stack([_ridge_predict(state["head"], c_features[state["features"]["bandwidth_factor"]])[:, 0]
                                      for state in states])
    selected = [{**candidates[state["candidate_index"]], "candidate_index": state["candidate_index"],
                 "mse": float(best_loss[h])} for h, state in enumerate(states)]
    return states, prediction, {"candidates": candidates, "selected": selected,
                                "landmark_indices": ids.tolist(), "raw_plus_nystrom": True,
                                "fit_split": "A", "selection_split": "B", "selection": "per horizon",
                                "failures": failures}


def _fit_raw_hgb(xa, ya, xb, yb, xc, seed, config, nodes):
    best_loss, selected = np.full(ya.shape[1], np.inf), [None] * ya.shape[1]
    selected_record, candidates, failures = [None] * ya.shape[1], [], []
    for rate in config["raw_hgb_learning_rates"]:
        for leaves in config["raw_hgb_max_leaf_nodes"]:
            record = {"learning_rate": float(rate), "max_leaf_nodes": int(leaves),
                      "max_iter": int(config["raw_hgb_max_iter"]), "early_stopping": False,
                      "l2_regularization": float(config["raw_hgb_l2_regularization"])}
            losses = []
            for h in range(ya.shape[1]):
                try:
                    with timed(nodes, "raw_hgb_fit_A"):
                        model = HistGradientBoostingRegressor(
                            learning_rate=float(rate), max_leaf_nodes=int(leaves),
                            max_iter=int(config["raw_hgb_max_iter"]), early_stopping=False,
                            l2_regularization=float(config["raw_hgb_l2_regularization"]),
                            random_state=int(seed)).fit(xa, ya[:, h])
                    with timed(nodes, "raw_hgb_select_B"):
                        loss = float(_mse(yb[:, h:h + 1], model.predict(xb)[:, None])[0])
                        if loss < best_loss[h]:
                            best_loss[h], selected[h] = loss, model
                            selected_record[h] = {**record, "mse": loss, "candidate_index": len(candidates)}
                    losses.append(loss)
                except Exception as exc:
                    failures.append({**record, "horizon_index": h, "exception": repr(exc)})
                    losses.append(None)
            candidates.append({**record, "mse_by_horizon": losses,
                               "status": "ok" if all(x is not None for x in losses) else "partial_failure"})
    if any(model is None for model in selected):
        raise FloatingPointError(f"No usable HGB candidate: {failures!r}")
    with timed(nodes, "raw_hgb_predict_C"):
        prediction = np.column_stack([model.predict(xc) for model in selected])
    return selected, prediction, {"candidates": candidates, "selected": selected_record,
                                 "fit_split": "A", "selection_split": "B", "selection": "per horizon",
                                 "failures": failures}


def _array_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(v, seen) for v in value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _array_bytes(vars(value), seen)
    return 0


def _measure_state(value):
    return {"pickle_bytes": len(pickle.dumps(value, protocol=5)), "array_buffer_bytes": _array_bytes(value)}


def _node_cost(nodes, dependencies):
    return {"wall_seconds": float(sum(nodes[name]["wall_seconds"] for name in dependencies if name in nodes)),
            "cpu_seconds": float(sum(nodes[name]["cpu_seconds"] for name in dependencies if name in nodes)),
            "nodes": sorted(set(dependencies) & set(nodes))}


def _dependencies(nodes, method, selected_policy=None, *, prediction=False):
    if method == "nonkahm_policy":
        if prediction:
            return sorted(set(_dependencies(nodes, selected_policy, prediction=True)) | {"policy_forward"})
        return sorted({node for member in NONKAHM_METHODS for node in _dependencies(nodes, member)} | {"policy_select_C"})
    result = {name for name in nodes if name.startswith("baseline_") or name == "input_scaler_A"}
    if method == "ridge":
        return sorted(result)
    result |= {name for name in nodes if name.startswith(f"{method}_")}
    if method in GEOMETRY_METHODS:
        result |= {name for name in nodes if name == "bank_structure" or name.startswith("common_raw_")
                   or name.startswith("common_scale_") or name.startswith("common_bound_")
                   or name == "common_scaler_A"}
        if method != "common":
            kind = method.split("_")[0]
            result |= {name for name in nodes if name == "bank_projected_metric" or name.startswith("projected_kernel_")
                       or name.startswith(f"{kind}_raw_") or name.startswith(f"{kind}_scale_")
                       or name == f"{kind}_scaler_A"}
            if not method.endswith("unbounded"):
                result |= {name for name in nodes if name.startswith(f"{kind}_bound_")}
            if kind == "kahm":
                result.add("bank_kahm_factor")
    return sorted(name for name in result if "rank_diagnostic" not in name)


@dataclass
class ForecastSuite:
    input_scaler: dict
    baseline: dict
    geometry: ReconstructionFeatures | None
    heads: dict
    raw_rbf: list | None
    raw_hgb: list | None
    gammas: dict
    selected_policy: str
    diagnostics: dict
    resources: dict
    config: dict

    def _method_state(self, method):
        if method == "nonkahm_policy":
            return self._method_state(self.selected_policy)
        state = {"input_scaler": self.input_scaler, "baseline": self.baseline}
        if method != "ridge":
            state["gamma"] = self.gammas[method]
        if method in GEOMETRY_METHODS and self.geometry is not None:
            kind = method.split("_")[0]
            exclude = {"reference_distance", "factor_error"}
            if kind == "common":
                exclude |= {"q", "inverse", "whitening", "uq", "lambda_original"}
            elif kind == "rbf":
                exclude |= {"inverse", "lambda_original"}
            bank = {**self.geometry.bank, "models": [{k: v for k, v in m.items() if k not in exclude}
                                                     for m in self.geometry.bank["models"]]}
            state.update(bank=bank, head=self.heads.get(method),
                         scalers={key: self.geometry.scalers[key] for key in ("common", kind)})
        elif method == "raw_rbf":
            state["model"] = self.raw_rbf
        elif method == "raw_hgb":
            state["model"] = self.raw_hgb
        return state

    def predict(self, x):
        """Predict without target input, state mutation, fitting or selection."""
        x = _matrix(x, "prediction inputs")
        if x.shape[1] != len(self.input_scaler["mean"]):
            raise ValueError("Prediction input width differs from A")
        nodes, diagnostics = {}, {"failures": [], "feature_bounds": {}, "bounds": {},
                                  "recovered_prediction_cells": {method: 0 for method in METHODS}}
        with timed(nodes, "baseline_transform_query"):
            z = _standardize(self.input_scaler, x)
        with timed(nodes, "baseline_predict_query"):
            baseline = _ridge_predict(self.baseline, z)
        prediction = {"ridge": baseline}
        features = {}
        if self.geometry is not None:
            try:
                features, geom = self.geometry.transform(x, nodes=nodes)
                diagnostics.update(geom)
            except Exception as exc:
                diagnostics["failures"].append({"method": "all_geometry", "exception": repr(exc)})
        candidates = {}
        for method in GEOMETRY_METHODS:
            with timed(nodes, f"{method}_predict_query"):
                try:
                    if method not in self.heads or method not in features:
                        raise FloatingPointError("Unavailable fitted geometry head or query features")
                    candidate = baseline + _ridge_predict(self.heads[method], features[method])
                except Exception as exc:
                    candidate = baseline.copy()
                    diagnostics["recovered_prediction_cells"][method] += int(baseline.size)
                    diagnostics["failures"].append({"method": method, "exception": repr(exc)})
                candidates[method] = candidate
        with timed(nodes, "persistence_predict_query"):
            candidates["persistence"] = np.repeat(x[:, -1:], baseline.shape[1], axis=1)
        with timed(nodes, "raw_rbf_transform_query"):
            rbf_cache = {}
            if self.raw_rbf is not None:
                for state in self.raw_rbf:
                    key = state["features"]["bandwidth_factor"]
                    if key not in rbf_cache:
                        try:
                            rbf_cache[key] = _raw_rbf_features(state["features"], z)
                        except Exception as exc:
                            rbf_cache[key] = exc
        for method, states in (("raw_rbf", self.raw_rbf), ("raw_hgb", self.raw_hgb)):
            candidate = baseline.copy()
            with timed(nodes, f"{method}_predict_query"):
                if states is None:
                    diagnostics["recovered_prediction_cells"][method] += int(baseline.size)
                    diagnostics["failures"].append({"method": method, "exception": "No fitted candidate"})
                else:
                    for h, state in enumerate(states):
                        try:
                            if method == "raw_hgb":
                                value = state.predict(x)
                            else:
                                f = rbf_cache[state["features"]["bandwidth_factor"]]
                                if isinstance(f, Exception):
                                    raise f
                                value = _ridge_predict(state["head"], f)[:, 0]
                            bad = ~np.isfinite(value)
                            candidate[~bad, h] = value[~bad]
                            if np.any(bad):
                                diagnostics["recovered_prediction_cells"][method] += int(bad.sum())
                                diagnostics["failures"].append({"method": method, "horizon_index": h,
                                                               "nonfinite_rows": np.flatnonzero(bad).tolist()})
                        except Exception as exc:
                            diagnostics["recovered_prediction_cells"][method] += int(len(x))
                            diagnostics["failures"].append({"method": method, "horizon_index": h, "exception": repr(exc)})
            candidates[method] = candidate
        for method in METHODS[1:-1]:
            with timed(nodes, f"{method}_blend_query"):
                prediction[method] = _blend(baseline, candidates[method], self.gammas[method])
                if not np.isfinite(prediction[method]).all():
                    bad = ~np.isfinite(prediction[method])
                    prediction[method][bad] = baseline[bad]
                    diagnostics["recovered_prediction_cells"][method] += int(bad.sum())
                    diagnostics["failures"].append({"method": method, "stage": "blend", "fallback_cells": int(bad.sum())})
            if method in self.heads:
                bound = copy.deepcopy(self.diagnostics["bounds"][method])
                absolute = np.max(np.abs(prediction[method] - baseline), axis=0)
                bound["max_abs_correction_by_horizon"] = absolute.tolist()
                if bound["correction_bound_by_horizon"] is not None:
                    limit = np.asarray(bound["correction_bound_by_horizon"])
                    bound["violations"] = int(np.count_nonzero(absolute > limit * (1 + 1e-10) + 1e-12))
                else:
                    bound["violations"] = None
                diagnostics["bounds"][method] = bound
        with timed(nodes, "policy_forward"):
            prediction["nonkahm_policy"] = prediction[self.selected_policy].copy()
            diagnostics["recovered_prediction_cells"]["nonkahm_policy"] = diagnostics["recovered_prediction_cells"][self.selected_policy]
        prediction = {method: prediction[method] for method in METHODS}
        diagnostics.update(selected_policy=self.selected_policy,
                           gamma_by_horizon={key: value.tolist() for key, value in self.gammas.items()},
                           prediction_max_abs={method: float(np.max(np.abs(value))) for method, value in prediction.items()})
        per_method = {}
        for method in METHODS:
            per_method[method] = {**self.resources["per_method"][method],
                                  "predict": _node_cost(nodes, _dependencies(nodes, method, self.selected_policy, prediction=True))}
        return {"predictions": prediction, "diagnostics": diagnostics,
                "resources": {"predict_nodes": nodes, "per_method": per_method,
                              "charging": "Each procedure pays every required shared node once; policy fitting pays candidate union."}}


def fit_suite(xa, ya, xb, yb, xc, yc, *, seed, config=None):
    """Fit and freeze all procedures from A/B/C; no reserved targets accepted."""
    config = copy.deepcopy({**DEFAULT_CONFIG, **(config or {})})
    xa, ya, xb, yb, xc, yc = [_matrix(value, name) for value, name in
                             zip((xa, ya, xb, yb, xc, yc), ("xa", "ya", "xb", "yb", "xc", "yc"))]
    if any(len(x) != len(y) for x, y in ((xa, ya), (xb, yb), (xc, yc))):
        raise ValueError("A/B/C feature and target rows must agree")
    if len({xa.shape[1], xb.shape[1], xc.shape[1]}) != 1 or len({ya.shape[1], yb.shape[1], yc.shape[1]}) != 1:
        raise ValueError("A/B/C feature and target widths must agree")
    if not config["gamma_grid"] or config["gamma_grid"][0] != 0 or any(g < 0 or g > 1 for g in config["gamma_grid"]):
        raise ValueError("Gamma grid must begin with zero and remain in [0,1]")
    if not config["alpha_grid"] or any(not np.isfinite(a) or a <= 0 for a in config["alpha_grid"]):
        raise ValueError("The alpha grid must contain positive finite penalties")
    if config["feature_scale_floor"] <= 0 or config["target_variance_floor"] <= 0:
        raise ValueError("Feature and variance floors must be positive")
    nodes = {}
    diagnostics = {"schema_version": 1, "seed": int(seed), "config": config,
                   "A_B_C_rows": [len(xa), len(xb), len(xc)], "input_width": xa.shape[1],
                   "horizons": ya.shape[1], "B": {}, "C": {}, "failures": [], "bounds": {},
                   "methods": list(METHODS), "nonkahm_order": list(NONKAHM_METHODS),
                   "objective": "mean row SSE + alpha * squared coefficient norm; free intercept",
                   "feature_scaling": "A only; common radial block is identical in all geometry arms; no readout restandardization"}
    with timed(nodes, "input_scaler_A"):
        scaler = _scaler_fit(xa, config["feature_scale_floor"])
    with timed(nodes, "baseline_transform_A_B_C"):
        za, zb, zc = [_standardize(scaler, x) for x in (xa, xb, xc)]
    baseline, diagnostics["B"]["ridge"] = _select_ridge(
        za, ya, zb, yb, np.zeros_like(yb), config["alpha_grid"], nodes, "baseline")
    with timed(nodes, "baseline_predict_A_B_C"):
        ba, bb, bc = [_ridge_predict(baseline, z) for z in (za, zb, zc)]
        residual = ya - ba
    candidates_c = {"ridge": bc}
    with timed(nodes, "persistence_predict_C"):
        candidates_c["persistence"] = np.repeat(xc[:, -1:], ya.shape[1], axis=1)
    failed_methods, heads, geometry = set(), {}, None
    try:
        bank = _InstrumentedBank(seed=seed, config=config).fit(xa)
        nodes.update(bank.nodes)
        diagnostics["bank"] = bank.diagnostics
        diagnostics["bank"].update(
            history_standardizer="Inherited A mean/std; std<=1e-12 replaced by1; not the subsequent feature scale floor",
            local_radius="Inherited mean squared centered reference radius, floored at1e-6",
            gate="Blockwise softmax(-0.5*squared standardized center distance/local radius)")
        state = bank.state("kahm")
        # Neither reference-distance training matrices nor source row indices
        # are required to replay this fixed map.
        for model in state["models"]:
            model.pop("reference_distance", None)
        raw, feature_diagnostics = {}, {}
        for label, x in (("A", xa), ("B", xb), ("C", xc)):
            raw[label], feature_diagnostics[label] = _raw_feature_maps(state, x, nodes, label)
        scalers = {}
        for key in ("common", "kahm", "rbf"):
            with timed(nodes, f"{key}_scaler_A"):
                scalers[key] = _scaler_fit(raw["A"][key], config["feature_scale_floor"])
        geometry = ReconstructionFeatures(state, scalers)
        features = {}
        for label in ("A", "B", "C"):
            features[label], transformed = geometry.transform_raw(raw[label], nodes=nodes, tag=label)
            feature_diagnostics[label].update(transformed)
        diagnostics["features_A_B_C"] = feature_diagnostics
        diagnostics["scalers"] = {key: {"mean": value["mean"].tolist(), "scale": value["scale"].tolist(),
                                         "floor": value["floor"], "floored_columns": value["floored_columns"]}
                                    for key, value in scalers.items()}
        diagnostics["feature_ranks_A"] = {}
        for key in ("common", "kahm", "rbf"):
            with timed(nodes, f"{key}_rank_diagnostic_A"):
                scaled = _standardize(scalers[key], raw["A"][key])
                capped = radial_bound(scaled)
                diagnostics["feature_ranks_A"][key] = {
                    "raw_centered_rank": int(np.linalg.matrix_rank(raw["A"][key] - raw["A"][key].mean(axis=0))),
                    "standardized_centered_rank": int(np.linalg.matrix_rank(scaled - scaled.mean(axis=0))),
                    "bounded_centered_rank": int(np.linalg.matrix_rank(capped - capped.mean(axis=0))),
                    "width": int(scaled.shape[1])}
        for method in GEOMETRY_METHODS:
            try:
                heads[method], diagnostics["B"][method] = _select_ridge(
                    features["A"][method], residual, features["B"][method], yb, bb,
                    config["alpha_grid"], nodes, method)
                with timed(nodes, f"{method}_predict_C"):
                    candidates_c[method] = bc + _ridge_predict(heads[method], features["C"][method])
            except Exception as exc:
                failed_methods.add(method)
                diagnostics["failures"].append({"method": method, "stage": "fit_A_select_B", "exception": repr(exc)})
                diagnostics["B"][method] = {"status": "failed", "exception": repr(exc)}
                candidates_c[method] = bc.copy()
    except Exception as exc:
        geometry = None
        diagnostics["failures"].append({"method": "all_geometry", "stage": "bank_or_features", "exception": repr(exc)})
        for method in GEOMETRY_METHODS:
            failed_methods.add(method)
            diagnostics["B"][method] = {"status": "failed", "exception": repr(exc)}
            candidates_c[method] = bc.copy()
    raw_rbf, raw_hgb = None, None
    for method, fitter, arrays in (("raw_rbf", _fit_raw_rbf, (za, ya, zb, yb, zc)),
                                    ("raw_hgb", _fit_raw_hgb, (xa, ya, xb, yb, xc))):
        try:
            fitted, candidates_c[method], diagnostics["B"][method] = fitter(*arrays, seed, config, nodes)
            if method == "raw_rbf":
                raw_rbf = fitted
            else:
                raw_hgb = fitted
        except Exception as exc:
            failed_methods.add(method)
            diagnostics["failures"].append({"method": method, "stage": "fit_A_select_B", "exception": repr(exc)})
            diagnostics["B"][method] = {"status": "failed", "exception": repr(exc)}
            candidates_c[method] = bc.copy()
    diagnostics["B"]["persistence"] = {"status": "fixed", "selection_split": None}
    gammas, calibrated = {}, {"ridge": bc}
    diagnostics["C"]["ridge"] = {"gamma_by_horizon": [0.] * ya.shape[1],
                                   "selected_mse_by_horizon": _mse(yc, bc).tolist(), "status": "fixed_baseline"}
    for method in METHODS[1:-1]:
        with timed(nodes, f"{method}_calibrate_C"):
            gammas[method], calibrated[method], diagnostics["C"][method] = _select_gamma(
                yc, bc, candidates_c[method], config["gamma_grid"], failed=method in failed_methods)
        if method in heads:
            width = heads[method]["feature_width"]
            diagnostics["bounds"][method] = _readout_bound(
                heads[method], None if method.endswith("unbounded") else math.sqrt(width), gammas[method])
    with timed(nodes, "policy_select_C"):
        variance = np.maximum(np.var(ya, axis=0, ddof=0), config["target_variance_floor"])
        scores = [float(np.mean(_mse(yc, calibrated[method]) / variance)) for method in NONKAHM_METHODS]
        selected_policy = NONKAHM_METHODS[int(np.argmin(scores))]
    diagnostics["policy"] = {"selected_method": selected_policy, "selection_split": "C",
                              "score": "mean horizon C MSE / A target variance (ddof=0, floored)",
                              "A_target_variance": variance.tolist(),
                              "candidates": [{"method": method, "normalized_mse": score}
                                             for method, score in zip(NONKAHM_METHODS, scores)]}
    diagnostics["gamma_by_horizon"] = {key: value.tolist() for key, value in gammas.items()}
    diagnostics["method_failures"] = {method: diagnostics["B"][method].get("exception", "No usable declared candidate")
                                       for method in sorted(failed_methods)}
    resources = {"fit_nodes": nodes, "per_method": {},
                 "diagnostic_nodes": [name for name in nodes if "rank_diagnostic" in name],
                 "charging": "Shared work charged once to each procedure needing it; non-KAHM policy pays candidate union; diagnostic-only ranks excluded."}
    suite = ForecastSuite(scaler, baseline, geometry, heads, raw_rbf, raw_hgb,
                          gammas, selected_policy, diagnostics, resources, config)
    for method in METHODS:
        resources["per_method"][method] = {
            "fit_and_selection": _node_cost(nodes, _dependencies(nodes, method)),
            "state": _measure_state(suite._method_state(method))}
    resources["all_retained_model_array_bytes"] = _array_bytes(
        {"input_scaler": scaler, "baseline": baseline, "geometry": geometry,
         "heads": heads, "raw_rbf": raw_rbf, "raw_hgb": raw_hgb})
    return suite
