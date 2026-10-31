"""Bounded numerical building blocks for a FUTURE KAHM streaming study.

This module does not update geometry, run a chronological forecasting benchmark,
or establish prediction-accuracy or latency improvements. It implements a
fixed-feature sliding-window ridge readout and a fixed-bank KAHM query cache.

Readout objective (sum loss, fixed positive alpha)::

    min_W sum_{i in active window} ||y_i - x_i @ W||^2 + alpha ||W||_F^2

Every supplied feature is penalized. If callers append a constant column, that
intercept is ALSO penalized. There is no implicit intercept, centering, scaling,
forgetting factor, baseline, donor residual generation, or target availability
policy. An application must supply only already available labels.

An ordinary update applies Sherman--Morrison addition/deletion to the inverse
normal matrix and recursive coefficient updates, costing O(p^2 + p*H).
Safety checks have the same order. A bounded replay rebuild costs
O(W*p^2 + W*p*H + p^3 + p^2*H), often written O(W*p^2 + p^3) for small H.
Owned numerical state is O(W*(p+H) + p^2 + p*H), independent of stream length.
Recent audit events are capped; persist returned update reports externally for
a complete record. A geometry reset/replay is an explicit bounded rebuild,
not an incremental geometry update.

The version identifier must represent the complete feature-coordinate and
target convention: scaler, metric, references, gates, feature normalization,
and baseline version when supplied targets are residuals. Numeric arrays alone
cannot prove that the caller has actually recomputed features for a new map.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class GeometryVersionError(ValueError):
    """A caller attempted to mix features or predictions from different maps."""


class NumericalUpdateError(RuntimeError):
    """Even bounded replay could not produce a finite positive-definite fit."""


class KAHMDenominatorError(FloatingPointError):
    """A cached reconstruction has a zero or relatively cancelled denominator."""


@dataclass(frozen=True)
class UpdateReport:
    update_number: int
    geometry_version: str
    active_rows: int
    evicted: bool
    rebuilt: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _version(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("geometry_version must be a nonempty string")
    return value


class BoundedWindowRidge:
    """Exact-arithmetic ridge recurrences with finite-precision safeguards.

    Unsafe rank-one steps or residual checks trigger a complete solve of the
    capped current window. If that solve itself fails, the operation raises
    without mutating the previously usable state. No numerical algorithm can
    guarantee accuracy for arbitrary ill-conditioned or overflowing inputs.
    """

    def __init__(
        self,
        window_size: int,
        n_features: int,
        n_outputs: int,
        *,
        alpha: float,
        geometry_version: str,
        denominator_tolerance: float = 1e-10,
        residual_tolerance: float = 1e-8,
        refactor_every: int | None = None,
        audit_capacity: int = 64,
    ) -> None:
        self.window_size = _positive_int(window_size, "window_size")
        self.n_features = _positive_int(n_features, "n_features")
        self.n_outputs = _positive_int(n_outputs, "n_outputs")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive")
        for name, value in (("denominator_tolerance", denominator_tolerance),
                            ("residual_tolerance", residual_tolerance)):
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if refactor_every is not None:
            refactor_every = _positive_int(refactor_every, "refactor_every")
        self.alpha = float(alpha)
        self.geometry_version = _version(geometry_version)
        self.denominator_tolerance = float(denominator_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.refactor_every = refactor_every
        self._rows: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._audit: deque[dict[str, Any]] = deque(maxlen=_positive_int(audit_capacity, "audit_capacity"))
        self.total_updates = 0
        self.total_evictions = 0
        self.total_rebuilds = 0
        self.numerical_fallbacks = 0
        self.scheduled_rebuilds = 0
        self.geometry_resets = 0
        self.geometry_replays = 0
        self._empty_state()

    def _empty_state(self) -> None:
        self._gram = self.alpha * np.eye(self.n_features)
        self._inverse = np.eye(self.n_features) / self.alpha
        self._cross = np.zeros((self.n_features, self.n_outputs))
        self._coef = np.zeros_like(self._cross)

    @property
    def active_rows(self) -> int:
        return len(self._rows)

    @property
    def coefficients(self) -> np.ndarray:
        return self._coef.copy()

    @property
    def recent_audit_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._audit]

    def _check_version(self, version: str) -> None:
        if version != self.geometry_version:
            raise GeometryVersionError(
                f"Features are tagged {version!r}, but active state is "
                f"{self.geometry_version!r}. Reset or replay recomputed features "
                "before using a new geometry version."
            )

    def _row(self, value: Any, width: int, name: str) -> np.ndarray:
        row = np.asarray(value, dtype=np.float64)
        if row.shape != (width,):
            raise ValueError(f"{name} must have shape ({width},), got {row.shape}")
        if not np.all(np.isfinite(row)):
            raise ValueError(f"{name} must be finite")
        return row.copy()

    def _rank_one(
        self,
        inverse: np.ndarray,
        coefficients: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        sign: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
        """Add (sign=+1) or delete (sign=-1) one labeled feature row."""
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                u = inverse @ x
                leverage = float(x @ u)
                denominator = 1.0 + sign * leverage
                floor = self.denominator_tolerance * (1.0 + abs(leverage))
                if not np.isfinite(denominator) or denominator <= floor:
                    return None, None, "unsafe_rank_one_denominator"
                if leverage < -self.denominator_tolerance:
                    return None, None, "negative_inverse_quadratic_form"
                # A huge positive update nearly subtracts the whole inverse
                # in its affected direction; direct replay is safer.
                if sign == 1 and 1.0 / denominator <= self.denominator_tolerance:
                    return None, None, "large_leverage_cancellation"
                new_inverse = inverse - sign * np.outer(u, u) / denominator
                error = y - x @ coefficients
                new_coefficients = coefficients + sign * np.outer(u, error) / denominator
                new_inverse = 0.5 * (new_inverse + new_inverse.T)
                if (not np.all(np.isfinite(new_inverse))
                    or not np.all(np.isfinite(new_coefficients))
                    or np.any(np.diag(new_inverse) <= 0)):
                    return None, None, "nonfinite_or_nonpositive_rank_one_state"
        except FloatingPointError:
            return None, None, "rank_one_arithmetic_error"
        return new_inverse, new_coefficients, None

    def _rebuild(
        self, rows: list[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not rows:
            return (self.alpha * np.eye(self.n_features),
                    np.zeros((self.n_features, self.n_outputs)),
                    np.eye(self.n_features) / self.alpha,
                    np.zeros((self.n_features, self.n_outputs)))
        x = np.stack([row[0] for row in rows])
        y = np.stack([row[1] for row in rows])
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                gram = self.alpha * np.eye(self.n_features) + x.T @ x
                cross = x.T @ y
                factor = np.linalg.cholesky(gram)
                inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, np.eye(self.n_features)))
                coefficients = np.linalg.solve(factor.T, np.linalg.solve(factor, cross))
                inverse = 0.5 * (inverse + inverse.T)
                if not all(np.all(np.isfinite(a)) for a in (gram, cross, inverse, coefficients)):
                    raise FloatingPointError("nonfinite rebuilt state")
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise NumericalUpdateError(
                "Bounded replay could not form a finite positive-definite ridge fit; "
                "the previous state remains unchanged."
            ) from exc
        return gram, cross, inverse, coefficients

    def update(self, x: Any, y: Any, *, geometry_version: str) -> UpdateReport:
        """Consume ONE fully observed target row; label chronology is caller-owned."""
        self._check_version(geometry_version)
        feature = self._row(x, self.n_features, "x")
        target = self._row(y, self.n_outputs, "y")
        old = self._rows[0] if len(self._rows) == self.window_size else None
        next_update = self.total_updates + 1
        reason = None
        inverse, coefficients = self._inverse, self._coef
        try:
            with np.errstate(over="raise", invalid="raise"):
                gram = self._gram + np.outer(feature, feature)
                cross = self._cross + np.outer(feature, target)
                if old is not None:
                    gram = gram - np.outer(old[0], old[0])
                    cross = cross - np.outer(old[0], old[1])
        except FloatingPointError as exc:
            raise NumericalUpdateError("Moment arithmetic overflow; previous state is unchanged") from exc

        if old is not None:
            inverse, coefficients, reason = self._rank_one(inverse, coefficients, old[0], old[1], -1)
        if reason is None:
            inverse, coefficients, reason = self._rank_one(inverse, coefficients, feature, target, 1)

        if reason is None:
            # Probe one inverse column and one target column per update. This
            # costs O(p^2), unlike forming the full inverse residual G @ P.
            with np.errstate(over="ignore", invalid="ignore"):
                j = (next_update - 1) % self.n_features
                unit = np.zeros(self.n_features)
                unit[j] = 1.0
                inverse_error = np.linalg.norm(gram @ inverse[:, j] - unit, ord=np.inf)
                h = (next_update - 1) % self.n_outputs
                coefficient_error = np.linalg.norm(gram @ coefficients[:, h] - cross[:, h], ord=np.inf)
                coefficient_error /= 1.0 + np.linalg.norm(cross[:, h], ord=np.inf)
            if (not np.isfinite(inverse_error) or not np.isfinite(coefficient_error)
                or inverse_error > self.residual_tolerance
                or coefficient_error > self.residual_tolerance):
                reason = "sampled_normal_equation_residual"

        scheduled = self.refactor_every is not None and next_update % self.refactor_every == 0
        if reason is None and scheduled:
            reason = "scheduled_rebuild"
        if reason is not None:
            replay_rows = list(self._rows)
            if old is not None:
                replay_rows = replay_rows[1:]
            replay_rows.append((feature, target))
            gram, cross, inverse, coefficients = self._rebuild(replay_rows)

        # Commit only after all arithmetic has succeeded.
        if old is not None:
            self._rows.popleft()
            self.total_evictions += 1
        self._rows.append((feature, target))
        self._gram, self._cross, self._inverse, self._coef = gram, cross, inverse, coefficients
        self.total_updates = next_update
        if reason is not None:
            self.total_rebuilds += 1
            if reason == "scheduled_rebuild":
                self.scheduled_rebuilds += 1
            else:
                self.numerical_fallbacks += 1
        report = UpdateReport(next_update, self.geometry_version, len(self._rows),
                              old is not None, reason is not None, reason)
        self._audit.append({"event": "update", **report.to_dict()})
        return report

    def predict(self, x: Any, *, geometry_version: str) -> np.ndarray:
        self._check_version(geometry_version)
        features = np.asarray(x, dtype=np.float64)
        if features.ndim not in (1, 2) or features.shape[-1] != self.n_features:
            raise ValueError("x must be one feature row or a matrix with n_features columns")
        if not np.all(np.isfinite(features)):
            raise ValueError("x must be finite")
        with np.errstate(over="raise", invalid="raise"):
            return features @ self._coef

    def reset_geometry(self, new_version: str) -> None:
        """Discard all old-map rows; no previous features are silently relabeled."""
        new_version = _version(new_version)
        self._rows.clear()
        self._empty_state()
        self.geometry_version = new_version
        self.geometry_resets += 1
        self._audit.append({"event": "geometry_reset", "geometry_version": new_version,
                            "active_rows": 0, "update_number": self.total_updates})

    def replay_geometry(self, new_version: str, features: Any, targets: Any) -> None:
        """Atomically replace state using <= W recomputed features of ONE map.

        Callers must regenerate every supplied feature and, when relevant,
        realign every residual target. The feature width remains fixed; create
        another readout if the new map has a different width.
        """
        new_version = _version(new_version)
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError("features must have shape (n_rows, n_features)")
        if y.shape != (len(x), self.n_outputs):
            raise ValueError("targets must have shape (n_rows, n_outputs)")
        if len(x) > self.window_size:
            raise ValueError("Replay input exceeds the declared window cap")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Replay features and targets must be finite")
        rows = [(a.copy(), b.copy()) for a, b in zip(x, y)]
        gram, cross, inverse, coefficients = self._rebuild(rows)
        self._rows = deque(rows)
        self._gram, self._cross, self._inverse, self._coef = gram, cross, inverse, coefficients
        self.geometry_version = new_version
        self.total_rebuilds += 1
        self.geometry_replays += 1
        self._audit.append({"event": "geometry_replay", "geometry_version": new_version,
                            "active_rows": len(rows), "update_number": self.total_updates})

    def state_summary(self) -> dict[str, Any]:
        """Small metadata only; numerical byte count excludes Python overhead."""
        numerical_bytes = sum(a.nbytes for a in (self._gram, self._inverse, self._cross, self._coef))
        numerical_bytes += sum(x.nbytes + y.nbytes for x, y in self._rows)
        return {
            "geometry_version": self.geometry_version,
            "window_size": self.window_size,
            "active_rows": len(self._rows),
            "n_features": self.n_features,
            "n_outputs": self.n_outputs,
            "alpha_sum_loss": self.alpha,
            "all_features_penalized": True,
            "total_updates": self.total_updates,
            "total_evictions": self.total_evictions,
            "total_rebuilds": self.total_rebuilds,
            "numerical_fallbacks": self.numerical_fallbacks,
            "scheduled_rebuilds": self.scheduled_rebuilds,
            "geometry_resets": self.geometry_resets,
            "geometry_replays": self.geometry_replays,
            "owned_numerical_array_bytes": numerical_bytes,
            "recent_audit_events": len(self._audit),
            "audit_capacity": self._audit.maxlen,
        }


@dataclass(frozen=True)
class KAHMCache:
    """Fixed coefficients A=Q B and b=B.T 1 (b=B1 when B is symmetric)."""

    reconstruction_matrix: np.ndarray
    denominator_vector: np.ndarray

    def reconstruct(self, kernels: Any, *, denominator_tolerance: float = 1e-12) -> np.ndarray:
        """Evaluate Q B k / (1.T B k), with explicit cancellation rejection.

        The cache does not compute the kernels. Callers retain the fixed bank's
        kernel and its stable log-kernel normalization. A failure here raises;
        it does not silently replace a denominator or claim an original-method
        fallback. Common positive row scaling is invariant in exact arithmetic.
        """
        if not np.isfinite(denominator_tolerance) or not 0 <= denominator_tolerance < 1:
            raise ValueError("denominator_tolerance must lie in [0, 1)")
        k = np.asarray(kernels, dtype=np.float64)
        if k.ndim not in (1, 2) or k.shape[-1] != self.denominator_vector.shape[0]:
            raise ValueError("Kernel row width must equal the number of cached references")
        if not np.all(np.isfinite(k)):
            raise ValueError("Kernel values must be finite")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            denominator = k @ self.denominator_vector
            scale = np.sum(np.abs(k * self.denominator_vector), axis=-1)
            if np.any(np.abs(denominator) <= denominator_tolerance * scale):
                raise KAHMDenominatorError("Zero or relatively cancelled signed KAHM denominator")
            numerator = k @ self.reconstruction_matrix.T
            return numerator / np.expand_dims(denominator, axis=-1)


def cache_kahm(Q: Any, B: Any) -> KAHMCache:
    """Cache a fixed local reconstruction; Q has shape (d,m), B (m,m).

    Construction costs O(d*m^2), queries O(d*m), and cached state O(d*m+m).
    This excludes kernel evaluation and any geometry refresh. Using B.T for
    the denominator also preserves the stated formula for nonsymmetric input.
    """
    q = np.asarray(Q, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] < 1:
        raise ValueError("Q must be a nonempty (dimension, references) matrix")
    if b.shape != (q.shape[1], q.shape[1]):
        raise ValueError("B must be square with one row per reference")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(b)):
        raise ValueError("Q and B must be finite")
    with np.errstate(over="raise", invalid="raise"):
        numerator = q @ b
        denominator = b.T @ np.ones(q.shape[1])
    numerator.setflags(write=False)
    denominator.setflags(write=False)
    return KAHMCache(numerator, denominator)
