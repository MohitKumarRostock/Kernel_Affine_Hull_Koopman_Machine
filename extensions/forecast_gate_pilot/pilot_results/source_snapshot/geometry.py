"""Exploratory forecast-trained KAHM gates, reconstructed from manuscript formulas.

This is a new implementation, not an upstream replication.  The reference bank
and reconstructions are fixed.  Direct fitting changes one log inverse
temperature per partition and forecast horizon using B forecast loss, with a
ridge head fitted on A at every objective evaluation.  C selects alpha and
mixing; D is only predicted.  No D labels are accepted by this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


DEFAULT_CONFIG = {
    "alpha_grid": [0.01, 0.1, 1.0],
    "gamma_grid": [0.0, 0.25, 0.5, 1.0],
    "baseline_alpha_grid": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "blocks": [2, 4, 8],
    "rank": 2,
    "max_references": 24,
    "bank_row_cap": 512,
    "temperature_bound": 4.0,
    "max_objective_evaluations": 160,
    "optimizer_maxiter": 40,
    "optimizer_ftol": 1e-10,
    "optimizer_gtol": 1e-6,
    "raw_rbf_landmarks": 64,
    "raw_rbf_bandwidth_factors": [0.5, 1.0, 2.0],
}
GATES = ("kahm", "affine", "locality", "rbf_local")


@contextmanager
def timed(nodes, name):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        nodes[name] = {
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
        }


def _scale_fit(x):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return mean, scale


def _solve_spd(a, b):
    return cho_solve(cho_factor(a, lower=True, check_finite=False), b,
                     check_finite=False)


def _ridge_fit(x, y, alpha):
    mean, scale = _scale_fit(x)
    xs = (x - mean) / scale
    ymean = np.mean(y, axis=0)
    coef = _solve_spd(xs.T @ xs + float(alpha) * np.eye(xs.shape[1]),
                      xs.T @ (y - ymean))
    if not np.all(np.isfinite(coef)):
        raise FloatingPointError("Nonfinite ridge coefficients")
    return {"mean": mean, "scale": scale, "ymean": ymean, "coef": coef}


def _ridge_predict(model, x):
    return model["ymean"] + ((x - model["mean"]) / model["scale"]) @ model["coef"]


def _finite_loss(y, pred):
    loss = float(np.mean(np.square(y - pred)))
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite squared loss")
    return loss


def _folding_logits(z, recon):
    radial = -np.expm1(-np.linalg.norm(z - recon, axis=1))
    denominator = np.linalg.norm(z, axis=1) * np.linalg.norm(recon, axis=1)
    cosine = np.divide(np.sum(z * recon, axis=1), denominator,
                       out=np.zeros(len(z)), where=denominator != 0)
    angular = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
    score = np.sqrt((radial ** 2 + angular ** 2) / 2.0)
    return 16.0 * np.log(1.0 - np.clip(score, 0.0, 1.0) + 1e-6)


class SharedBank:
    """Common A-only standardizer, partitions, references and local PCA."""

    def __init__(self, seed=0, config=None):
        self.seed = int(seed)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.nodes = {}
        self.diagnostics = {"warnings": [], "local_models": []}

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or not np.all(np.isfinite(x)):
            raise ValueError("Bank requires at least two finite input rows")
        rng = np.random.default_rng(self.seed)
        self.models, self.block_slices = [], []
        with timed(self.nodes, "bank_structure"):
            self.mean, self.scale = _scale_fit(x)
            z = (x - self.mean) / self.scale
            inds = np.arange(len(z))
            if len(inds) > self.config["bank_row_cap"]:
                inds = np.sort(rng.choice(inds, self.config["bank_row_cap"], replace=False))
            sampled = z[inds]
            distinct = len(np.unique(sampled, axis=0))
            for block_index, requested in enumerate(self.config["blocks"]):
                groups = max(1, min(int(requested), distinct))
                while True:
                    with warnings.catch_warnings(record=True) as seen:
                        warnings.simplefilter("always")
                        if groups == 1:
                            labels = np.zeros(len(sampled), dtype=int)
                        else:
                            labels = KMeans(n_clusters=groups, n_init=5, max_iter=150,
                                            random_state=int(rng.integers(0, 2**31 - 1))).fit_predict(sampled)
                        self.diagnostics["warnings"].extend(str(w.message) for w in seen)
                    if len(np.unique(labels)) == groups:
                        break
                    reduced = max(1, groups // 2)
                    self.diagnostics["warnings"].append(f"Empty groups: reduced {groups} to {reduced}")
                    groups = reduced
                start = len(self.models)
                for group in range(groups):
                    eligible = np.flatnonzero(labels == group)
                    if len(eligible) > self.config["max_references"]:
                        selected = np.sort(rng.choice(eligible, self.config["max_references"], replace=False))
                    else:
                        selected = eligible
                    # Preserve the manuscript's reference float32 storage conversion.
                    q = sampled[selected].astype(np.float32).astype(np.float64)
                    center = q.mean(axis=0)
                    centered = q - center
                    rank = min(int(self.config["rank"]), x.shape[1] - 1, len(q) - 1)
                    if rank > 0:
                        cov = centered.T @ centered / (len(q) - 1)
                        values, vectors = np.linalg.eigh(cov)
                        keep = np.flatnonzero(values > np.finfo(float).eps)[::-1][:rank]
                        p = vectors[:, keep]
                        rank = p.shape[1]
                        while rank > 1 and np.min(np.ptp(centered @ p, axis=0)) < 1e-3:
                            p = p[:, :-1]
                            rank -= 1
                    else:
                        p = np.empty((x.shape[1], 0))
                    radius = max(float(np.mean(np.sum(centered ** 2, axis=1))), 1e-6)
                    affine = center + (centered @ p) @ p.T
                    projection_scale = max(float(np.mean(np.sum((q - affine) ** 2, axis=1))),
                                           0.05 * radius, 1e-6)
                    self.models.append({"q": q, "center": center, "p": p,
                                        "rank": rank, "block": block_index,
                                        "radius": radius, "projection_scale": projection_scale,
                                        "reference_indices": inds[selected]})
                self.block_slices.append((start, len(self.models)))
        with timed(self.nodes, "kahm_factors"):
            for c, model in enumerate(self.models):
                q, p, r = model["q"], model["p"], model["rank"]
                record = {"model": c, "block": model["block"], "rank": r,
                          "reference_count": len(q),
                          "reference_indices": model["reference_indices"].tolist()}
                if r:
                    u = (q - model["center"]) @ p
                    covariance = u.T @ u / (len(q) - 1)
                    ev, evec = np.linalg.eigh(covariance)
                    if np.min(ev) <= 0:
                        raise FloatingPointError("Nonpositive projected covariance")
                    whitening = (evec / np.sqrt(ev)) @ evec.T
                    uq = u @ whitening
                    kernel = np.exp(-cdist(uq, uq, metric="sqeuclidean") / (2.0 * r))
                    nu = float(np.mean(q ** 2))
                    lam = 2.5 * nu
                    converged = False
                    for iteration in range(100):
                        inverse = _solve_spd(kernel + lam * np.eye(len(q)), np.eye(len(q)))
                        updated = 2.0 * nu + float(np.mean((q - kernel @ inverse @ q) ** 2))
                        if abs(updated - lam) <= 1e-3:
                            converged = True
                            break
                        if iteration < 99:
                            lam = updated
                    model.update({"whitening": whitening, "uq": uq, "inverse": inverse})
                    record.update({"regularizer": lam, "next_regularizer": updated,
                                   "iterations": iteration + 1, "converged": converged})
                self.diagnostics["local_models"].append(record)
        self.diagnostics.update({"requested_blocks": list(self.config["blocks"]),
                                 "effective_blocks": [end - start for start, end in self.block_slices],
                                 "feature_width": len(self.models) + sum(m["rank"] for m in self.models),
                                 "reference_dtype": "float32 stored, float64 arithmetic",
                                 "fixed_reconstruction": True})
        return self

    def transform(self, x, split_name):
        x = np.asarray(x, dtype=np.float64)
        with timed(self.nodes, f"transform_common_{split_name}"):
            z = (x - self.mean) / self.scale
            coordinates = [(z - m["center"]) @ m["p"] for m in self.models]
            affine = [m["center"] + u @ m["p"].T for m, u in zip(self.models, coordinates)]
        logits = {}
        query_diagnostics = {"zero_denominators": 0, "minimum_relative_denominator": 1.0}
        for gate in GATES:
            try:
                with timed(self.nodes, f"transform_{gate}_{split_name}"):
                    columns = []
                    for model, u, affine_recon in zip(self.models, coordinates, affine):
                        if gate == "kahm":
                            if model["rank"] == 0:
                                recon = np.broadcast_to(model["center"], z.shape)
                            else:
                                logs = -cdist(u @ model["whitening"], model["uq"],
                                              metric="sqeuclidean") / (2.0 * model["rank"])
                                kernel = np.exp(logs - logs.max(axis=1, keepdims=True))
                                weights = kernel @ model["inverse"]
                                denominator = weights.sum(axis=1)
                                relative = np.abs(denominator) / np.maximum(np.sum(np.abs(weights), axis=1), 1e-300)
                                query_diagnostics["minimum_relative_denominator"] = min(
                                    query_diagnostics["minimum_relative_denominator"], float(relative.min()))
                                query_diagnostics["zero_denominators"] += int(np.sum(denominator == 0))
                                denominator = np.where(denominator == 0, 1.0, denominator)
                                recon = (weights @ model["q"]) / denominator[:, None]
                            column = _folding_logits(z, recon)
                        elif gate == "affine":
                            column = _folding_logits(z, affine_recon)
                        elif gate == "locality":
                            column = (-np.sum((z - affine_recon) ** 2, axis=1) / model["projection_scale"]
                                      - 0.1 * np.sum((z - model["center"]) ** 2, axis=1) / model["radius"])
                        else:
                            column = -np.sum((z - model["center"]) ** 2, axis=1) / (2.0 * model["radius"])
                        columns.append(column)
                    matrix = np.column_stack(columns)
                    if not np.all(np.isfinite(matrix)):
                        raise FloatingPointError(f"Nonfinite {gate} logits")
                    logits[gate] = matrix
            except Exception as exc:
                logits[gate] = None
                self.diagnostics.setdefault("transform_failures", []).append(
                    {"split": split_name, "gate": gate, "exception": repr(exc)})
        self.diagnostics.setdefault("query", {})[split_name] = query_diagnostics
        return {"logits": logits, "coordinates": coordinates,
                "block_slices": self.block_slices, "n_rows": len(x)}


def feature_matrix(cache, gate, eta):
    raw = cache["logits"][gate]
    if raw is None:
        raise FloatingPointError(f"Unavailable {gate} logits")
    eta = np.asarray(eta, dtype=float)
    affinities = np.empty_like(raw)
    for b, (start, end) in enumerate(cache["block_slices"]):
        logits = math.exp(float(eta[b])) * raw[:, start:end]
        affinities[:, start:end] = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return np.column_stack([affinities] + [affinities[:, c:c + 1] * u
                                          for c, u in enumerate(cache["coordinates"])])


class _EvaluationBudget(Exception):
    pass


def _train_temperature(caches, gate, residual_a, residual_b, alpha, scale, config):
    dimension = len(caches["A"]["block_slices"])
    eta0 = np.zeros(dimension)
    budget = int(config["max_objective_evaluations"])
    evaluations, best, exceptions = [], {"loss": float("inf"), "eta": eta0.copy()}, []

    def objective(eta):
        if len(evaluations) >= budget:
            raise _EvaluationBudget()
        error = None
        try:
            fa = feature_matrix(caches["A"], gate, eta)
            fb = feature_matrix(caches["B"], gate, eta)
            head = _ridge_fit(fa, residual_a, alpha)
            mse = _finite_loss(residual_b, _ridge_predict(head, fb))
            value = mse / scale
        except Exception as exc:
            error = repr(exc)
            exceptions.append(error)
            value, mse = 1e100, None
        evaluations.append({"eta": np.asarray(eta).tolist(), "normalized_b_mse": float(value),
                            "b_mse": mse, "exception": error})
        if value < best["loss"]:
            best.update(loss=value, eta=np.asarray(eta).copy())
        return value

    initial = objective(eta0)
    status, nit, message, converged = "not_started", None, "", False
    if budget > 1:
        try:
            bound = math.log(float(config["temperature_bound"]))
            result = minimize(objective, eta0, method="L-BFGS-B",
                              bounds=[(-bound, bound)] * dimension,
                              options={"maxiter": int(config["optimizer_maxiter"]),
                                       "maxfun": budget,
                                       "ftol": float(config["optimizer_ftol"]),
                                       "gtol": float(config["optimizer_gtol"]), "maxls": 20})
            converged = bool(result.success)
            status, nit, message = str(result.status), int(result.nit), str(result.message)
        except _EvaluationBudget:
            status, message = "evaluation_budget", "Exact objective evaluation budget reached"
        except Exception as exc:
            status, message = "optimizer_exception", repr(exc)
            exceptions.append(repr(exc))
    # Retain the best evaluated point, including identity. No C/D loss enters this decision.
    if not np.isfinite(best["loss"]) or best["loss"] >= 1e99:
        raise FloatingPointError("No finite temperature objective")
    return best["eta"], {"alpha": float(alpha), "initial_normalized_b_mse": float(initial),
                         "best_normalized_b_mse": float(best["loss"]),
                         "eta": best["eta"].tolist(), "nfev": len(evaluations), "nit": nit,
                         "status": status, "message": message, "converged": converged,
                         "exceptions": exceptions, "evaluations": evaluations}


def _baseline_ridge(xa, ya, groups, xb, xc, xd, config):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two A episode groups required for baseline selection")
    count = max(1, min(len(unique) - 1, int(math.floor(0.8 * len(unique)))))
    fit = np.isin(groups, unique[:count])
    validation = ~fit
    candidates, best = [], None
    for alpha in config["baseline_alpha_grid"]:
        model = _ridge_fit(xa[fit], ya[fit], alpha)
        loss = _finite_loss(ya[validation], _ridge_predict(model, xa[validation]))
        candidates.append({"alpha": float(alpha), "inner_a_mse": loss})
        if best is None or loss < best[0]:
            best = (loss, float(alpha))
    model = _ridge_fit(xa, ya, best[1])
    predictions = {name: _ridge_predict(model, x) for name, x in
                   (("A", xa), ("B", xb), ("C", xc), ("D", xd))}
    return predictions, {"selected_alpha": best[1], "candidates": candidates,
                         "inner_training_groups": unique[:count].tolist(),
                         "inner_validation_groups": unique[count:].tolist(),
                         "selection_split": "inner A", "final_fit_split": "all A",
                         "baseline_held_fixed": True}


def _raw_rbf(xa, ya, xb, yb, xc, xd, seed, config):
    rng = np.random.default_rng(seed)
    mean, scale = _scale_fit(xa)
    zz = {name: (x - mean) / scale for name, x in (("A", xa), ("B", xb), ("C", xc), ("D", xd))}
    m = min(int(config["raw_rbf_landmarks"]), len(xa))
    ids = np.sort(rng.choice(len(xa), m, replace=False))
    landmarks = zz["A"][ids]
    pair = cdist(landmarks, landmarks, metric="sqeuclidean")
    positive = pair[pair > 0]
    base_bandwidth = float(np.median(positive)) if len(positive) else 1.0
    candidates, best, best_predictions, retained_rank = [], None, None, None
    for factor in config["raw_rbf_bandwidth_factors"]:
        bandwidth = base_bandwidth * float(factor)
        kernel = np.exp(-pair / (2.0 * bandwidth))
        values, vectors = np.linalg.eigh(kernel)
        keep = values > max(1e-12, 1e-10 * float(np.max(values)))
        if not np.any(keep):
            raise FloatingPointError("No retained Nyström directions")
        transform = vectors[:, keep] / np.sqrt(values[keep])
        features = {}
        for split, z in zz.items():
            k = np.exp(-cdist(z, landmarks, metric="sqeuclidean") / (2.0 * bandwidth))
            features[split] = np.column_stack((z, k @ transform))
        for alpha in config["baseline_alpha_grid"]:
            model = _ridge_fit(features["A"], ya, alpha)
            bpred = _ridge_predict(model, features["B"])
            loss = _finite_loss(yb, bpred)
            record = {"bandwidth_factor": float(factor), "bandwidth_squared": bandwidth,
                      "alpha": float(alpha), "b_mse": loss, "rank": int(np.sum(keep))}
            candidates.append(record)
            if best is None or loss < best[0]:
                best = (loss, dict(record))
                retained_rank = int(np.sum(keep))
                best_predictions = {split: _ridge_predict(model, features[split]) for split in ("C", "D")}
    return best_predictions, {"selected": best[1], "candidates": candidates,
                              "landmark_indices": ids.tolist(), "retained_rank": retained_rank,
                              "selection_split": "B", "final_fit_split": "A", "raw_plus_nystrom": True}


def fit_suite(Xa, Ya, groups_a, Xb, Yb, Xc, Yc, Xd, seed, config=None):
    """Return C/D forecasts and auditable settings for all declared procedures.

    Targets have shape (rows, horizons).  The caller owns chronological/episode
    split construction and D target evaluation.  Gamma=0 returns a copy of the
    same raw ridge forecast, including when a geometric fit fails.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    xa, ya, xb, yb, xc, yc, xd = [np.asarray(a, dtype=np.float64) for a in
                                  (Xa, Ya, Xb, Yb, Xc, Yc, Xd)]
    groups = np.asarray(groups_a)
    if ya.ndim != 2 or yb.ndim != 2 or yc.ndim != 2:
        raise ValueError("Targets must have shape (rows, horizons)")
    if not all(np.all(np.isfinite(a)) for a in (xa, ya, xb, yb, xc, yc, xd)):
        raise ValueError("All supplied arrays must be finite")
    diagnostics = {"config": config, "seed": int(seed), "methods": {}, "cost_nodes": {},
                   "scope": "Exploratory reconstruction; frozen bank; direct forecast-trained gate temperatures",
                   "ridge_penalty_convention": "sum squared errors + alpha * squared coefficient norm",
                   "target_horizon_count": ya.shape[1], "failures": []}
    nodes = diagnostics["cost_nodes"]
    with timed(nodes, "baseline_ridge"):
        base, baseline_diagnostic = _baseline_ridge(xa, ya, groups, xb, xc, xd, config)
    pc, pd = {"raw_ridge": base["C"].copy()}, {"raw_ridge": base["D"].copy()}
    diagnostics["methods"]["raw_ridge"] = baseline_diagnostic
    with timed(nodes, "raw_rbf"):
        try:
            rbf, record = _raw_rbf(xa, ya, xb, yb, xc, xd, seed + 7919, config)
            pc["raw_rbf"], pd["raw_rbf"] = rbf["C"], rbf["D"]
            diagnostics["methods"]["raw_rbf"] = record
        except Exception as exc:
            pc["raw_rbf"], pd["raw_rbf"] = base["C"].copy(), base["D"].copy()
            diagnostics["methods"]["raw_rbf"] = {"fallback": "raw_ridge", "exception": repr(exc)}
            diagnostics["failures"].append({"method": "raw_rbf", "exception": repr(exc)})
    bank = SharedBank(seed=seed + 1543, config=config)
    bank_failure = None
    try:
        bank.fit(xa)
        caches = {split: bank.transform(x, split) for split, x in
                  (("A", xa), ("B", xb), ("C", xc), ("D", xd))}
    except Exception as exc:
        bank_failure = repr(exc)
        diagnostics["failures"].append({"method": "shared_bank", "exception": bank_failure})
    nodes.update(bank.nodes)
    diagnostics["bank"] = bank.diagnostics
    residual_a, residual_b = ya - base["A"], yb - base["B"]
    target_scale = np.maximum(np.var(ya, axis=0), 1e-12)
    dimension = len(config["blocks"])
    for gate in GATES:
        for direct in (False, True):
            name = ("direct_" if direct else "fixed_") + gate
            pc[name], pd[name] = base["C"].copy(), base["D"].copy()
            method_record = {"horizons": [], "fallbacks": [], "gate": gate,
                             "direct_training": direct, "fit_split": "A", "geometry_fit_split": "B" if direct else None,
                             "selection_split": "C", "temperature_parameter_count_per_horizon": dimension if direct else 0}
            with timed(nodes, f"heads_{name}"):
                for h in range(ya.shape[1]):
                    initial_loss = _finite_loss(yc[:, h], base["C"][:, h])
                    selected = {"candidate_id": "zero", "alpha": None, "gamma": 0.0,
                                "eta": np.zeros(dimension).tolist(), "c_mse": initial_loss,
                                "fallback": None}
                    candidates = [dict(selected)]
                    optimizations = []
                    best_c, best_d = base["C"][:, h].copy(), base["D"][:, h].copy()
                    best_head, best_eta = None, None
                    if bank_failure:
                        selected["fallback"] = "shared_bank_failure"
                        method_record["fallbacks"].append({"horizon": h, "exception": bank_failure})
                    else:
                        for alpha in config["alpha_grid"]:
                            try:
                                if direct:
                                    eta, opt = _train_temperature(caches, gate, residual_a[:, h], residual_b[:, h],
                                                                  alpha, target_scale[h], config)
                                    optimizations.append(opt)
                                else:
                                    eta = np.zeros(dimension)
                                fa = feature_matrix(caches["A"], gate, eta)
                                fc = feature_matrix(caches["C"], gate, eta)
                                head = _ridge_fit(fa, residual_a[:, h], alpha)
                                qc = _ridge_predict(head, fc)
                                for gamma in config["gamma_grid"]:
                                    if float(gamma) == 0.0:
                                        continue
                                    pred_c = base["C"][:, h] + float(gamma) * qc
                                    loss = _finite_loss(yc[:, h], pred_c)
                                    record = {"candidate_id": f"alpha={float(alpha):g},gamma={float(gamma):g}",
                                              "alpha": float(alpha), "gamma": float(gamma), "eta": eta.tolist(),
                                              "c_mse": loss, "fallback": None}
                                    candidates.append(record)
                                    if loss < selected["c_mse"]:
                                        selected = dict(record)
                                        best_c = pred_c.copy()
                                        best_head, best_eta = head, eta.copy()
                            except Exception as exc:
                                failure = {"horizon": h, "alpha": float(alpha), "exception": repr(exc)}
                                method_record["fallbacks"].append(failure)
                                diagnostics["failures"].append({"method": name, **failure})
                                for gamma in config["gamma_grid"]:
                                    if float(gamma) != 0.0:
                                        candidates.append({"candidate_id": f"alpha={float(alpha):g},gamma={float(gamma):g}",
                                                           "alpha": float(alpha), "gamma": float(gamma), "eta": None,
                                                           "c_mse": None, "fallback": repr(exc)})
                    deployment_fallback = None
                    if best_head is not None:
                        try:
                            fd = feature_matrix(caches["D"], gate, best_eta)
                            qd = _ridge_predict(best_head, fd)
                            if not np.all(np.isfinite(qd)):
                                raise FloatingPointError("Nonfinite D correction")
                            best_d = base["D"][:, h] + selected["gamma"] * qd
                        except Exception as exc:
                            deployment_fallback = repr(exc)
                            failure = {"horizon": h, "stage": "D_prediction", "exception": deployment_fallback}
                            method_record["fallbacks"].append(failure)
                            diagnostics["failures"].append({"method": name, **failure})
                    pc[name][:, h], pd[name][:, h] = best_c, best_d
                    method_record["horizons"].append({"horizon_index": h, "selected": selected,
                                                       "candidates": candidates, "optimizations": optimizations,
                                                       "deployment_fallback": deployment_fallback})
            diagnostics["methods"][name] = method_record
    for name, record in diagnostics["methods"].items():
        if name == "raw_ridge":
            dependencies = ["baseline_ridge"]
        elif name == "raw_rbf":
            dependencies = ["raw_rbf"]
            if record.get("fallback"):
                dependencies.append("baseline_ridge")
        else:
            gate = record["gate"]
            dependencies = ["baseline_ridge", "bank_structure", f"heads_{name}"]
            if gate == "kahm":
                dependencies.append("kahm_factors")
            # B gates are needed only by directly trained procedures. Baseline
            # timing conservatively includes all split predictions in every arm.
            for split in (["A", "B", "C", "D"] if record["direct_training"] else ["A", "C", "D"]):
                dependencies += [f"transform_common_{split}", f"transform_{gate}_{split}"]
        record["cost_dependencies"] = dependencies
        record["charged_wall_seconds"] = sum(nodes.get(k, {}).get("wall_seconds", 0.0) for k in dependencies)
        record["charged_cpu_seconds"] = sum(nodes.get(k, {}).get("cpu_seconds", 0.0) for k in dependencies)
        record["selected_c_mse"] = _finite_loss(yc, pc[name])
    return {"predictions_c": pc, "predictions_d": pd, "diagnostics": diagnostics}
