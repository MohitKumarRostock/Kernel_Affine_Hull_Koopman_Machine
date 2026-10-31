"""Bounded causal development forecasters; no final-stream selection lives here.

The KAHM reconstruction is Q (K + lambda I)^-1 k / (1' (K + lambda I)^-1 k),
with the covariance-projected Gaussian kernel, reference-energy regularizer
iteration, signed-denominator guard and radial/angular folding score of the
preserved 88e9c56 manuscript implementation. This module changes the bounded
reference policy and the temporal fitting procedure, not that reconstruction
into an ordinary positive-weight kernel smoother.

Geometry updates are bounded LOCAL REBUILDS. They are not advertised as
rank-one updates of a changing covariance Gaussian Gram matrix. The separate
kernel comparator is exact sliding-window Gaussian kernel ridge with a free
intercept and recursive Gram inverse updates; it is not published KRLS-T.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp
from sklearn.cluster import KMeans


METHODS = ("ridge", "fixed_kahm", "adaptive_kahm", "adaptive_affine",
           "adaptive_rbf", "kernel_rr")
DEFAULT_CONFIG = {
    "baseline_alpha": 1.0,
    "correction_alpha": 1.0,
    "kernel_alpha": 0.1,
    "local_models": 4,
    "rank": 2,
    "refresh_every": 16,
    "max_replacements": 4,
    "relative_denominator_tolerance": 1e-10,
    "kernel_inverse_audit_every": 128,
    "kernel_inverse_audit_tolerance": 1e-7,
}


def _matrix(x, name="input"):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return x


def _spd(a, b):
    return cho_solve(cho_factor((a + a.T) * 0.5, lower=True, check_finite=False),
                     b, check_finite=False)


def _with_intercept(z):
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return np.r_[1.0, z]
    return np.column_stack((np.ones(len(z)), z))


def _penalty(width, alpha):
    p = np.eye(width) * float(alpha)
    p[0, 0] = 0.0
    return p


def _scale(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, np.where(std > 1e-12, std, 1.0)


def owned_array_bytes(value):
    """Count unique owned NumPy buffers reachable from a model, not Python RSS."""
    seen_objects, seen_buffers = set(), set()

    def walk(obj):
        if id(obj) in seen_objects:
            return 0
        seen_objects.add(id(obj))
        if isinstance(obj, np.ndarray):
            root = obj
            while isinstance(root.base, np.ndarray):
                root = root.base
            key = (int(root.__array_interface__["data"][0]), int(root.nbytes))
            if key in seen_buffers:
                return 0
            seen_buffers.add(key)
            return int(root.nbytes)
        if isinstance(obj, dict):
            return sum(walk(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(walk(v) for v in obj)
        if hasattr(obj, "__dict__"):
            return walk(vars(obj))
        return 0

    return walk(value)


class _InputBuffer:
    def __init__(self, capacity, dimension):
        self.values = np.zeros((capacity, dimension), dtype=float)
        self.capacity, self.count, self.next = int(capacity), 0, 0

    def push(self, z):
        self.values[self.next] = z
        self.next = (self.next + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def rows(self):
        if self.count < self.capacity:
            return self.values[:self.count].copy()
        return np.concatenate((self.values[self.next:], self.values[:self.next]))


class _ReplayHead:
    """Exact sliding-window raw ridge and current-baseline residual ridge.

    Raw ridge retains its shared sliding sufficient-statistic implementation.
    The correction objective is solved directly from the bounded replay, with
    residuals recomputed against the CURRENT raw coefficients. Centered,
    column-equilibrated augmented QR avoids subtracting enormous feature
    outer products when transient rows leave the window. Equilibration also
    transforms the penalty rows, so it does not change the ridge objective.
    """
    def __init__(self, capacity, dimension, feature_width=0,
                 baseline_alpha=1.0, correction_alpha=1.0):
        self.capacity, self.dimension = int(capacity), int(dimension)
        self.feature_width = int(feature_width)
        self.count, self.next, self.valid_count = 0, 0, 0
        self.x = np.zeros((capacity, dimension + 1))
        self.y = np.zeros(capacity)
        self.valid = np.zeros(capacity, dtype=bool)
        self.origin = np.full(capacity, -1, dtype=np.int64)
        self.xx = np.zeros((dimension + 1, dimension + 1))
        self.xy = np.zeros(dimension + 1)
        self.raw_penalty = _penalty(dimension + 1, baseline_alpha)
        self.raw_coef = np.zeros(dimension + 1)
        if feature_width:
            g = feature_width + 1
            self.f = np.zeros((capacity, g))
            self.geo_penalty = _penalty(g, correction_alpha)
            self.geo_coef = np.zeros(g)
            self.geo_center = np.zeros(feature_width)
            self.geo_target_mean = 0.0

    def push(self, z, y, f=None, valid=True, origin=-1):
        i = self.next
        if self.count == self.capacity:
            oldx = self.x[i]
            self.xx -= np.outer(oldx, oldx)
            self.xy -= oldx * self.y[i]
            if self.feature_width and self.valid[i]:
                self.valid_count -= 1
        else:
            self.count += 1
        xv = _with_intercept(z)
        self.x[i], self.y[i], self.origin[i] = xv, float(y), int(origin)
        self.xx += np.outer(xv, xv)
        self.xy += xv * y
        self.valid[i] = bool(valid)
        if self.feature_width:
            fv = _with_intercept(f) if valid else np.zeros(self.feature_width + 1)
            self.f[i] = fv
            if valid:
                self.valid_count += 1
        self.next = (self.next + 1) % self.capacity

    def solve(self):
        if not self.count:
            return
        self.raw_coef = _spd(self.xx + self.raw_penalty, self.xy)
        if self.feature_width:
            if self.valid_count:
                use = self.valid[:self.count]
                f = self.f[:self.count, 1:][use]
                residual = self.y[:self.count][use] - self.x[:self.count][use] @ self.raw_coef
                self.geo_center = f.mean(axis=0)
                self.geo_target_mean = float(residual.mean())
                centered = f - self.geo_center
                root_penalty = np.sqrt(np.diag(self.geo_penalty)[1:])
                # This is a change of variables inside the solver only. The
                # existing feature normalizer and alpha remain untouched.
                scale = np.maximum(np.max(np.abs(centered), axis=0), root_penalty)
                scale = np.where(scale > 0.0, scale, 1.0)
                design = np.vstack((centered / scale,
                                    np.diag(root_penalty / scale)))
                target = np.r_[residual - self.geo_target_mean,
                               np.zeros(self.feature_width)]
                q, r, pivot = qr(design, mode="economic", pivoting=True,
                                 check_finite=False)
                coefficients = np.empty(self.feature_width)
                coefficients[pivot] = solve_triangular(
                    r, q.T @ target, check_finite=False)
                slopes = coefficients / scale
                self.geo_coef = np.r_[self.geo_target_mean - self.geo_center @ slopes,
                                      slopes]
            else:
                self.geo_coef.fill(0.0)
                self.geo_center.fill(0.0)
                self.geo_target_mean = 0.0
        if not np.isfinite(self.raw_coef).all():
            raise FloatingPointError("Nonfinite sliding raw ridge coefficients")
        if self.feature_width and not np.isfinite(self.geo_coef).all():
            raise FloatingPointError("Nonfinite sliding residual ridge coefficients")

    def physical_rows(self):
        return np.arange(self.count)

    def rebuild_features(self, features, valid):
        n = self.count
        if features.shape != (n, self.feature_width) or valid.shape != (n,):
            raise ValueError("Replacement features do not match replay rows")
        self.f.fill(0.0)
        self.f[:n] = _with_intercept(features)
        self.f[:n][~valid] = 0.0
        self.valid[:n] = valid
        self.valid_count = int(valid.sum())
        x, y = self.x[:n], self.y[:n]
        # Reconstruct raw sufficient statistics too, bounding accumulated error.
        self.xx, self.xy = x.T @ x, x.T @ y
        self.solve()

    def components(self, z, f=None, valid=True):
        baseline = float(_with_intercept(z) @ self.raw_coef)
        correction = (float(self.geo_target_mean
                            + (np.asarray(f) - self.geo_center) @ self.geo_coef[1:])
                      if self.feature_width and valid else 0.0)
        return baseline, correction


@dataclass
class _LocalGeometry:
    q: np.ndarray
    anchor_count: int
    capacity: int
    next_slot: int
    center: np.ndarray | None = None
    p: np.ndarray | None = None
    radius: float = 1.0
    inverse: np.ndarray | None = None
    whitening: np.ndarray | None = None
    uq: np.ndarray | None = None
    regularizer: float | None = None
    factor_failure: str | None = None


class _GeometryBank:
    def __init__(self, gate, budget, seed, config):
        self.gate, self.budget, self.seed, self.config = gate, int(budget), int(seed), config
        self.local_count = int(config["local_models"])
        self.rank = int(config["rank"])
        self.refresh_index = 0
        self.refresh_count = 0
        self.reference_changes = 0
        self.factor_failures = 0

    def fit(self, z):
        count = min(self.local_count, len(np.unique(z, axis=0)))
        self.local_count = count
        km = KMeans(n_clusters=count, random_state=self.seed,
                    n_init=5, max_iter=150).fit(z)
        self.assignment_centers = km.cluster_centers_.copy()
        rng = np.random.default_rng(self.seed + 1)
        self.models = []
        capacities = [self.budget // count + int(i < self.budget % count)
                      for i in range(count)]
        for c in range(count):
            ids = np.flatnonzero(km.labels_ == c)
            if len(ids) > capacities[c]:
                ids = np.sort(rng.choice(ids, capacities[c], replace=False))
            q = np.asarray(z[ids], dtype=np.float32).astype(np.float64)
            anchors = max(1, (len(q) + 1) // 2)
            local = _LocalGeometry(q, anchors, capacities[c], anchors)
            self._rebuild(local)
            self.models.append(local)
        self.feature_width = count * (1 + self.rank)
        return self

    def _rebuild(self, local):
        q = local.q
        local.center = q.mean(axis=0)
        centered = q - local.center
        if len(q) < 2:
            p = np.empty((q.shape[1], 0))
        else:
            covariance = centered.T @ centered / (len(q) - 1)
            ev, vectors = np.linalg.eigh(covariance)
            order = np.argsort(-ev)
            ev, vectors = ev[order], vectors[:, order]
            valid = int(np.count_nonzero(ev > np.finfo(float).eps))
            rank = min(self.rank, q.shape[1] - 1, len(q) - 1, valid)
            p = vectors[:, :rank]
            while rank > 1 and np.min(np.ptp(q @ p, axis=0)) < 1e-3:
                rank -= 1
                p = vectors[:, :rank]
        local.p = p
        local.radius = max(float(np.mean(np.sum(centered * centered, axis=1))), 1e-6)
        local.factor_failure = None
        local.inverse = local.whitening = local.uq = None
        local.regularizer = None
        if self.gate != "kahm" or p.shape[1] == 0:
            return
        try:
            projected = q @ p
            covariance = np.atleast_2d(np.cov(projected, rowvar=False, bias=False))
            precision = np.linalg.inv(covariance)
            eig, vectors = np.linalg.eigh(precision)
            if not np.isfinite(eig).all() or np.min(eig) <= 0:
                raise FloatingPointError("Nonpositive projected precision")
            whitening = (vectors * np.sqrt(eig)[None, :]) @ vectors.T
            uq = projected @ whitening
            distance = cdist(uq, uq, metric="sqeuclidean")
            kernel = np.exp(-distance / (2.0 * p.shape[1]))
            # Exactly the preserved reference-energy regularizer iteration,
            # with float32 reference storage and float64 arithmetic.
            nu = float(np.mean(q * q))
            lam, change, iteration = 2.5 * nu, 1.0, 0
            while change > 1e-3 and iteration < 100:
                used_lam = lam
                inverse = _spd(kernel + used_lam * np.eye(len(q)), np.eye(len(q)))
                residual = q - kernel @ inverse @ q
                lam = 2.0 * nu + float(np.mean(residual * residual))
                change = abs(lam - used_lam)
                iteration += 1
            if iteration == 0 or used_lam <= 0 or not np.isfinite(inverse).all():
                raise FloatingPointError("Invalid KAHM regularizer")
            local.inverse, local.whitening, local.uq = inverse, whitening, uq
            local.regularizer = float(used_lam)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            local.factor_failure = f"{type(exc).__name__}: {exc}"
            self.factor_failures += 1

    def transform(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        n = len(z)
        logits = np.zeros((n, self.local_count))
        coordinates = []
        safe = np.ones(n, dtype=bool)
        min_relative = 1.0
        negative_weights = 0
        tolerance = self.config["relative_denominator_tolerance"]
        for j, local in enumerate(self.models):
            centered = z - local.center
            coordinate = centered @ local.p
            padded = np.zeros((n, self.rank))
            padded[:, :coordinate.shape[1]] = coordinate
            coordinates.append(padded)
            if self.gate == "rbf":
                logits[:, j] = -0.5 * np.sum(centered * centered, axis=1) / local.radius
                continue
            if self.gate == "affine" or local.p.shape[1] == 0:
                reconstruction = local.center + coordinate @ local.p.T
            elif local.factor_failure:
                safe[:] = False
                reconstruction = np.broadcast_to(local.center, z.shape)
            else:
                query = (z @ local.p) @ local.whitening
                logs = -cdist(query, local.uq, metric="sqeuclidean") / (2.0 * local.p.shape[1])
                kernel = np.exp(logs - np.max(logs, axis=1, keepdims=True))
                weight = kernel @ local.inverse
                denominator = weight.sum(axis=1)
                l1 = np.abs(weight).sum(axis=1)
                relative = np.divide(np.abs(denominator), l1,
                                     out=np.zeros_like(l1), where=l1 > 0)
                rowsafe = (relative > tolerance) & np.isfinite(relative)
                rowsafe &= np.isfinite(weight).all(axis=1)
                safe &= rowsafe
                min_relative = min(min_relative, float(np.min(relative)))
                negative_weights += int(np.count_nonzero(weight < 0))
                den = np.where(rowsafe, denominator, 1.0)
                reconstruction = weight @ local.q / den[:, None]
                reconstruction[~rowsafe] = local.center
            radial = 1.0 - np.exp(-np.linalg.norm(z - reconstruction, axis=1))
            denom = np.linalg.norm(z, axis=1) * np.linalg.norm(reconstruction, axis=1)
            cosine = np.divide(np.sum(z * reconstruction, axis=1), denom,
                               out=np.zeros(n), where=denom != 0)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
            score = np.sqrt((radial * radial + angle * angle) / 2.0)
            logits[:, j] = 16.0 * np.log(1.0 - np.clip(score, 0, 1) + 1e-6)
        phi = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        feature = np.column_stack([phi] + [phi[:, j:j + 1] * coord
                                            for j, coord in enumerate(coordinates)])
        safe &= np.isfinite(feature).all(axis=1)
        feature[~safe] = 0.0
        return feature, safe, {"minimum_relative_denominator": min_relative,
                               "negative_weight_entries": negative_weights,
                               "unsafe_rows": int(np.count_nonzero(~safe))}

    def refresh(self, observed):
        """Target-free common policy, at most four changes in one local bank."""
        j = self.refresh_index % self.local_count
        self.refresh_index += 1
        local = self.models[j]
        assigned = np.argmin(cdist(observed, self.assignment_centers, "sqeuclidean"), axis=1)
        candidates = observed[assigned == j][::-1]
        selected = []
        for row in candidates:
            row = np.asarray(row, dtype=np.float32).astype(float)
            if np.any(np.all(local.q == row, axis=1)):
                continue
            if any(np.array_equal(row, previous) for previous in selected):
                continue
            selected.append(row)
            if len(selected) == int(self.config["max_replacements"]):
                break
        changes = 0
        for row in reversed(selected):
            if len(local.q) < local.capacity:
                local.q = np.vstack((local.q, row))
            elif local.anchor_count < local.capacity:
                if local.next_slot >= local.capacity:
                    local.next_slot = local.anchor_count
                local.q[local.next_slot] = row
                local.next_slot += 1
            else:
                continue
            changes += 1
        if changes:
            self._rebuild(local)
            self.refresh_count += 1
            self.reference_changes += changes
        return {"local_model": j, "reference_changes": changes,
                "geometry_refreshed": bool(changes),
                "factor_failed": bool(local.factor_failure)}


class _SlidingKernel:
    """Exact windowed Gaussian KRR with an unpenalized constant mean."""
    def __init__(self, capacity, bandwidth2, alpha=0.1, audit_every=128, tolerance=1e-7):
        self.capacity, self.bandwidth2, self.alpha = int(capacity), float(bandwidth2), float(alpha)
        self.audit_every, self.tolerance = int(audit_every), float(tolerance)
        self.updates, self.rebuilds, self.audit_checks = 0, 0, 0

    def kernel(self, a, b):
        return np.exp(-cdist(np.atleast_2d(a), np.atleast_2d(b), "sqeuclidean")
                      / (2.0 * self.bandwidth2))

    def fit(self, x, y):
        self.x = np.asarray(x[-self.capacity:], dtype=float).copy()
        self.y = np.asarray(y[-self.capacity:], dtype=float).copy()
        self.inverse = _spd(self.kernel(self.x, self.x) + self.alpha * np.eye(len(self.x)),
                            np.eye(len(self.x)))
        self._solve()
        return self

    def _solve(self):
        ones = self.inverse.sum(axis=1)
        denom = float(ones.sum())
        if not math.isfinite(denom) or denom <= 0:
            raise FloatingPointError("Invalid kernel intercept denominator")
        self.intercept = float(ones @ self.y / denom)
        self.coef = self.inverse @ (self.y - self.intercept)
        if not np.isfinite(self.coef).all():
            raise FloatingPointError("Nonfinite kernel coefficients")

    def _rebuild(self):
        self.inverse = _spd(self.kernel(self.x, self.x) + self.alpha * np.eye(len(self.x)),
                            np.eye(len(self.x)))
        self.rebuilds += 1

    def push(self, x, y):
        current = self.inverse
        oldx, oldy = self.x, self.y
        if len(oldx) == self.capacity:
            current = current[1:, 1:] - np.outer(current[1:, 0], current[0, 1:]) / current[0, 0]
            oldx, oldy = oldx[1:], oldy[1:]
        k = self.kernel(oldx, x)[:, 0]
        u = current @ k
        schur = 1.0 + self.alpha - float(k @ u)
        self.x, self.y = np.vstack((oldx, x)), np.r_[oldy, y]
        self.updates += 1
        if not math.isfinite(schur) or schur <= 1e-12:
            self._rebuild()
        else:
            inv = np.empty((len(self.x), len(self.x)))
            inv[:-1, :-1] = current + np.outer(u, u) / schur
            inv[:-1, -1] = inv[-1, :-1] = -u / schur
            inv[-1, -1] = 1.0 / schur
            self.inverse = (inv + inv.T) * 0.5
        if self.audit_every and self.updates % self.audit_every == 0:
            matrix = self.kernel(self.x, self.x) + self.alpha * np.eye(len(self.x))
            error = float(np.max(np.abs(matrix @ self.inverse - np.eye(len(self.x)))))
            self.audit_checks += 1
            if not math.isfinite(error) or error > self.tolerance:
                self._rebuild()
        self._solve()

    def predict(self, z):
        return float(self.intercept + self.kernel(z, self.x)[0] @ self.coef)


class OnlineForecaster:
    """Six bounded algorithms with explicit, causal matured-target updates."""
    def __init__(self, method, reference_budget, replay_size, horizons, seed=0, config=None):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}")
        self.method = method
        self.reference_budget, self.replay_size = int(reference_budget), int(replay_size)
        self.horizons = tuple(int(h) for h in horizons)
        self.seed = int(seed)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        if self.reference_budget < 8 or self.replay_size < 8 or not self.horizons or min(self.horizons) < 1:
            raise ValueError("Positive horizons and reference/replay capacities >= 8 required")
        self.version = self.update_calls = self.supervised_updates = 0
        self.geometry_version = 0
        self.last_now = None
        self.counters = {"prediction_fallbacks": 0, "training_feature_fallback_rows": 0,
                         "reencoded_feature_fallback_rows": 0,
                         "geometry_refresh_attempts": 0, "geometry_refreshes": 0,
                         "reference_changes": 0, "maximum_replay_rows": 0}

    @property
    def geometric(self):
        return self.method in {"fixed_kahm", "adaptive_kahm", "adaptive_affine", "adaptive_rbf"}

    def fit(self, X, Y):
        X = _matrix(X)
        Y = _matrix(Y, "targets")
        if len(X) != len(Y) or Y.shape[1] != len(self.horizons):
            raise ValueError("Fit targets must match inputs and horizons")
        self.dimension = X.shape[1]
        self.mean, self.scale = _scale(X)
        z = (X - self.mean) / self.scale
        self.observed = _InputBuffer(self.replay_size, self.dimension)
        for row in z[-self.replay_size:]:
            self.observed.push(row)
        if self.geometric:
            gate = ("kahm" if "kahm" in self.method else
                    "affine" if self.method == "adaptive_affine" else "rbf")
            self.bank = _GeometryBank(gate, self.reference_budget, self.seed, self.config).fit(z)
            self._fit_feature_scale(self.observed.rows())
            width = self.bank.feature_width
            feature, valid, _ = self._features(z[-self.replay_size:])
        else:
            width = 0
            feature = np.empty((min(len(z), self.replay_size), 0))
            valid = np.ones(len(feature), dtype=bool)
        self.heads = []
        recent_z, recent_y = z[-self.replay_size:], Y[-self.replay_size:]
        for h in range(len(self.horizons)):
            head = _ReplayHead(self.replay_size, self.dimension, width,
                               self.config["baseline_alpha"], self.config["correction_alpha"])
            for i, (row, target) in enumerate(zip(recent_z, recent_y[:, h])):
                head.push(row, target, feature[i], bool(valid[i]), origin=-1)
            head.solve()
            self.heads.append(head)
        if self.method == "kernel_rr":
            probe = z[np.linspace(0, len(z) - 1, min(128, len(z)), dtype=int)]
            distances = pdist(probe, "sqeuclidean")
            positive = distances[distances > 1e-12]
            self.kernel_bandwidth2 = float(np.median(positive)) if len(positive) else 1.0
            self.kernels = [_SlidingKernel(self.reference_budget, self.kernel_bandwidth2,
                                            self.config["kernel_alpha"],
                                            self.config["kernel_inverse_audit_every"],
                                            self.config["kernel_inverse_audit_tolerance"]).fit(z, Y[:, h])
                            for h in range(len(self.horizons))]
        # Only adaptive geometry subsequently uses unlabeled input replay.
        # Fixed geometry has already fitted its feature normalizer above.
        if not self.method.startswith("adaptive_"):
            self.observed = None
        self.counters["maximum_replay_rows"] = max(head.count for head in self.heads)
        return self

    def _fit_feature_scale(self, z):
        raw, valid, _ = self.bank.transform(z)
        if np.any(valid):
            self.feature_mean, self.feature_scale = _scale(raw[valid])
        else:
            self.feature_mean = np.zeros(self.bank.feature_width)
            self.feature_scale = np.ones(self.bank.feature_width)

    def _features(self, z):
        raw, valid, diagnostics = self.bank.transform(z)
        f = (raw - self.feature_mean) / self.feature_scale
        f[~valid] = 0.0
        return f, valid, diagnostics

    def update(self, matured, now, observed_x=None):
        now = int(now)
        if self.last_now is not None and now <= self.last_now:
            raise ValueError("Observation times must increase strictly")
        prepared = []
        for record in matured:
            h = int(record["horizon_index"])
            if not 0 <= h < len(self.horizons):
                raise ValueError("Unknown horizon index")
            origin, target = int(record["origin"]), int(record["target"])
            if target > now or target != origin + self.horizons[h]:
                raise ValueError("Supervised target is not causally mature at this time")
            x = np.asarray(record["x"], dtype=float)
            y = float(record["y"])
            if x.shape != (self.dimension,) or not np.isfinite(x).all() or not math.isfinite(y):
                raise ValueError("Nonfinite or incompatible matured pair")
            prepared.append((h, (x - self.mean) / self.scale, y, origin))
        if observed_x is not None:
            observed_x = np.asarray(observed_x, dtype=float)
            if observed_x.shape != (self.dimension,) or not np.isfinite(observed_x).all():
                raise ValueError("Observed input must be a finite current history")
        self.last_now = now
        self.update_calls += 1
        self.version += 1
        diagnostics = {"matured_pairs": len(prepared), "geometry_refreshed": False,
                       "reference_changes": 0, "version": self.version,
                       "geometry_version": self.geometry_version,
                       "training_feature_fallback_rows": 0}
        changed_heads = set()
        for h, z, y, origin in prepared:
            if self.geometric:
                f, valid, _ = self._features(z[None, :])
                fv, good = f[0], bool(valid[0])
                if not good:
                    self.counters["training_feature_fallback_rows"] += 1
                    diagnostics["training_feature_fallback_rows"] += 1
            else:
                fv, good = None, True
            self.heads[h].push(z, y, fv, good, origin=origin)
            if self.method == "kernel_rr":
                self.kernels[h].push(z, y)
            changed_heads.add(h)
        for h in changed_heads:
            self.heads[h].solve()
        self.supervised_updates += len(prepared)
        if observed_x is not None and self.observed is not None:
            self.observed.push((observed_x - self.mean) / self.scale)
        if self.method.startswith("adaptive_") and self.update_calls % self.config["refresh_every"] == 0:
            self.counters["geometry_refresh_attempts"] += 1
            refresh = self.bank.refresh(self.observed.rows())
            diagnostics.update(refresh)
            if refresh["geometry_refreshed"]:
                self.geometry_version += 1
                self.counters["geometry_refreshes"] += 1
                self.counters["reference_changes"] += refresh["reference_changes"]
                self._fit_feature_scale(self.observed.rows())
                for head in self.heads:
                    f, valid, _ = self._features(head.x[:head.count, 1:])
                    self.counters["reencoded_feature_fallback_rows"] += int(np.count_nonzero(~valid))
                    head.rebuild_features(f, valid)
                diagnostics["geometry_version"] = self.geometry_version
        self.counters["maximum_replay_rows"] = max(
            self.counters["maximum_replay_rows"], max(head.count for head in self.heads))
        return diagnostics

    def predict_components(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,) or not np.isfinite(x).all():
            raise ValueError("Prediction input must be one finite history")
        z = (x - self.mean) / self.scale
        baseline, correction = np.zeros(len(self.horizons)), np.zeros(len(self.horizons))
        diagnostics = {"fallback": False, "fallback_horizons": [], "version": self.version,
                       "geometry_version": self.geometry_version}
        if self.geometric:
            f, valid, stats = self._features(z[None, :])
            feature, good = f[0], bool(valid[0])
            diagnostics.update(stats)
        else:
            feature, good = None, True
        for h, head in enumerate(self.heads):
            baseline[h], correction[h] = head.components(z, feature, good)
            if self.method == "kernel_rr":
                correction[h] = self.kernels[h].predict(z) - baseline[h]
            if not good or not math.isfinite(correction[h]):
                correction[h] = 0.0
                diagnostics["fallback_horizons"].append(h)
        if not np.isfinite(baseline).all():
            raise FloatingPointError("Raw baseline returned a nonfinite forecast")
        diagnostics["fallback"] = bool(diagnostics["fallback_horizons"])
        if diagnostics["fallback"]:
            self.counters["prediction_fallbacks"] += 1
        return baseline, correction, diagnostics

    def state_summary(self):
        result = {"method": self.method, "reference_budget": self.reference_budget,
                  "replay_size": self.replay_size, "horizons": list(self.horizons),
                  "version": self.version, "geometry_version": self.geometry_version,
                  "update_calls": self.update_calls, "supervised_updates": self.supervised_updates,
                  "replay_rows_per_horizon": [h.count for h in self.heads],
                  "valid_correction_rows_per_horizon": ([h.valid_count for h in self.heads]
                                                         if self.geometric else None),
                  "observed_input_rows": self.observed.count if self.observed is not None else 0,
                  "owned_numpy_bytes": owned_array_bytes(self),
                  "memory_scope": "Unique reachable NumPy buffers; excludes Python objects, allocator overhead, runtime libraries and runner queues/logs.",
                  "counters": dict(self.counters), "config": dict(self.config)}
        if self.geometric:
            result.update(reference_counts=[len(m.q) for m in self.bank.models],
                          reference_sha256=[hashlib.sha256(m.q.astype("<f8").tobytes()).hexdigest() for m in self.bank.models],
                          anchor_sha256=[hashlib.sha256(m.q[:m.anchor_count].astype("<f8").tobytes()).hexdigest() for m in self.bank.models],
                          total_geometry_references=sum(len(m.q) for m in self.bank.models),
                          anchor_counts=[m.anchor_count for m in self.bank.models],
                          geometry_feature_width=self.bank.feature_width,
                          local_ranks=[m.p.shape[1] for m in self.bank.models],
                          regularizers=[m.regularizer for m in self.bank.models],
                          factor_failures=self.bank.factor_failures,
                          correction_solver="Current bounded replay; centered column-equilibrated augmented QR; unchanged free-intercept ridge objective.")
        if self.method == "kernel_rr":
            result.update(kernel_supports_per_horizon=[len(k.x) for k in self.kernels],
                          kernel_inverse_rebuilds=sum(k.rebuilds for k in self.kernels),
                          kernel_inverse_audit_checks=sum(k.audit_checks for k in self.kernels),
                          kernel_bandwidth_squared=self.kernel_bandwidth2,
                          kernel_algorithm="Exact sliding-window Gaussian kernel ridge; recursive add/delete Gram inverse; free intercept; not KRLS-T.")
        return result
