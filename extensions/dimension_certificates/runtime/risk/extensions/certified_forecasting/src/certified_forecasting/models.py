"""Convex direct multi-horizon decoders and transparent forecasting baselines."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class BoundedRidge:
    """Ridge regression with a joint Frobenius norm constraint and clipping.

    Solve mean squared vector loss plus ``alpha * ||W||_F**2`` under
    ``||W||_F <= kappa`` and an unpenalized intercept.  A scalar Lagrange
    multiplier is found by bisection.  The clipped predictor is Lipschitz in
    its supplied features, with per-target constant ``lipschitz_[h]``.

    Bounds must come from the task definition, never from calibration data.
    Multiple targets may represent forecast horizons or output coordinates.
    """

    def __init__(self, alpha=1e-5, kappa=None, bounds=(-1.0, 1.0)):
        self.alpha = float(alpha)
        self.kappa = None if kappa is None else float(kappa)
        self.bounds = bounds

    def fit(self, X, Y):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or len(X) < 1:
            raise ValueError("X and Y must be nonempty matrices with equal rows.")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("Training arrays must be finite.")
        if self.alpha < 0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite and nonnegative.")
        if self.kappa is not None and (self.kappa < 0 or not np.isfinite(self.kappa)):
            raise ValueError("kappa must be finite and nonnegative, or None.")
        self.x_mean_, self.y_mean_ = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - self.x_mean_, Y - self.y_mean_
        gram = (Xc.T @ Xc) / len(X)
        rhs = (Xc.T @ Yc) / len(X)
        values, vectors = np.linalg.eigh(gram)
        values = np.maximum(values, 0.0)
        coordinates = vectors.T @ rhs

        def solve(penalty):
            denominator = values + penalty
            cutoff = max(float(values.max(initial=0.0)), 1.0) * 1e-13
            inverse = np.divide(1.0, denominator, out=np.zeros_like(denominator),
                                where=denominator > cutoff)
            return vectors @ (coordinates * inverse[:, None])

        penalty = self.alpha
        weights = solve(penalty)
        if self.kappa == 0:
            weights = np.zeros_like(weights)
            penalty = float("inf")
        elif self.kappa is not None and np.linalg.norm(weights) > self.kappa:
            low, high = penalty, max(penalty, 1e-8)
            while np.linalg.norm(solve(high)) > self.kappa:
                high *= 2.0
            for _ in range(70):
                middle = (low + high) / 2.0
                if np.linalg.norm(solve(middle)) > self.kappa:
                    low = middle
                else:
                    high = middle
            penalty, weights = high, solve(high)
        self.coef_ = weights
        self.intercept_ = self.y_mean_ - self.x_mean_ @ weights
        self.ridge_ = penalty
        self.lipschitz_ = np.linalg.norm(weights, axis=0)
        self.frobenius_norm_ = float(np.linalg.norm(weights))
        self.operator_norm_ = float(np.linalg.norm(weights, ord=2))
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = Y.shape[1]
        if self.bounds is not None:
            lower, upper = (np.asarray(b, dtype=float) for b in self.bounds)
            lower = np.broadcast_to(lower, (self.n_outputs_,)).copy()
            upper = np.broadcast_to(upper, (self.n_outputs_,)).copy()
            if not (np.isfinite(lower).all() and np.isfinite(upper).all()
                    and np.all(lower < upper)):
                raise ValueError("Bounds must be finite and strictly ordered.")
            self.lower_, self.upper_ = lower, upper
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("Fit the decoder before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Invalid prediction matrix.")
        prediction = X @ self.coef_ + self.intercept_
        return (prediction if self.bounds is None
                else np.clip(prediction, self.lower_, self.upper_))


def _validated_blocks(block_slices, n_features):
    if block_slices is None:
        return (slice(0, n_features),)
    blocks = tuple(block_slices)
    if not blocks:
        raise ValueError("At least one nonempty feature block is required.")
    cover = np.zeros(n_features, dtype=int)
    validated = []
    for block in blocks:
        if not isinstance(block, slice):
            raise ValueError("Feature blocks must be slices.")
        start, stop, step = block.indices(n_features)
        if step != 1 or stop <= start:
            raise ValueError("Blocks must be nonempty contiguous slices.")
        cover[start:stop] += 1
        validated.append(slice(start, stop))
    if not np.all(cover == 1):
        raise ValueError("Blocks must partition the feature coordinates exactly.")
    return tuple(validated)


def project_block_ball(weights, block_slices, kappa):
    """Euclidean projection onto ``sum_s ||W_s||_F <= kappa``.

    The vector of block norms is projected onto a nonnegative l1 ball, then
    each nonzero block is radially rescaled. Blocks must partition the rows.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or not np.isfinite(weights).all():
        raise ValueError("Weights must be a finite matrix.")
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative.")
    blocks = _validated_blocks(block_slices, weights.shape[0])
    norms = np.array([np.linalg.norm(weights[s]) for s in blocks])
    if norms.sum() <= kappa:
        return weights.copy()
    if kappa == 0:
        return np.zeros_like(weights)
    ordered = np.sort(norms)[::-1]
    cumulative = np.cumsum(ordered) - kappa
    admissible = ordered > cumulative / np.arange(1, len(norms) + 1)
    last = int(np.flatnonzero(admissible)[-1])
    threshold = cumulative[last] / (last + 1)
    projected = np.zeros_like(weights)
    for block, norm in zip(blocks, norms):
        if norm > threshold:
            projected[block] = weights[block] * (1.0 - threshold / norm)
    return projected


class BlockBoundedRidge:
    """Ridge decoder with a sum-of-block-norms budget and an optimization gap.

    Minimize mean squared vector error plus ``alpha * ||W||_F**2``, with an
    unpenalized intercept and ``sum_s ||W_s||_F <= kappa``. The corresponding
    feature metric is ``max_s ||F_s - F'_s||_2``. Old blocks are retained;
    appending zero rows preserves feasibility. Duplicate blocks do not enlarge
    the constrained *predictor class*, since their coefficients can be merged.

    A squared-Frobenius ridge penalty is not invariant to duplicate blocks;
    predictor-class invariance does not imply estimator invariance.
    The returned Frank-Wolfe gap upper-bounds regularized objective suboptimality
    at the final feasible iterate. Certification of prediction risk remains
    valid even if a requested numerical optimization tolerance is not reached.
    """

    def __init__(self, alpha=1e-4, kappa=4.0, bounds=(-1.0, 1.0),
                 block_slices=None, tol=1e-7, max_iter=20000):
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.bounds = bounds
        self.block_slices = block_slices
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def fit(self, X, Y):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or not len(X):
            raise ValueError("X and Y must be nonempty matrices with equal rows.")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("Training arrays must be finite.")
        if (not np.isfinite(self.alpha) or self.alpha < 0
                or not np.isfinite(self.kappa) or self.kappa < 0
                or not np.isfinite(self.tol) or self.tol <= 0 or self.max_iter < 1):
            raise ValueError("Invalid regularization, norm budget, or optimizer settings.")
        self.block_slices_ = _validated_blocks(self.block_slices, X.shape[1])
        self.x_mean_, self.y_mean_ = X.mean(axis=0), Y.mean(axis=0)
        xc, yc = X - self.x_mean_, Y - self.y_mean_
        gram = (xc.T @ xc) / len(X)
        gram += self.alpha * np.eye(X.shape[1])
        rhs = (xc.T @ yc) / len(X)
        constant = float(np.mean(np.sum(yc * yc, axis=1)))
        smoothness = max(2.0 * float(np.linalg.eigvalsh(gram)[-1]), 1e-15)

        def objective(w):
            return float(np.sum(w * (gram @ w)) - 2.0 * np.sum(w * rhs) + constant)

        def gradient(w):
            return 2.0 * (gram @ w - rhs)

        def gap(w):
            g = gradient(w)
            support = self.kappa * max(np.linalg.norm(g[s]) for s in self.block_slices_)
            return max(0.0, float(np.sum(g * w) + support))

        weights = np.zeros((X.shape[1], Y.shape[1]))
        extrapolated = weights.copy()
        momentum = 1.0
        current = objective(weights)
        current_gap = gap(weights)
        iterations = 0
        if self.kappa > 0:
            for iteration in range(1, self.max_iter + 1):
                candidate = project_block_ball(
                    extrapolated - gradient(extrapolated) / smoothness,
                    self.block_slices_, self.kappa,
                )
                candidate_objective = objective(candidate)
                if candidate_objective > current + 1e-14 * max(1.0, abs(current)):
                    # Monotone restart: an ordinary projected-gradient step.
                    momentum = 1.0
                    candidate = project_block_ball(
                        weights - gradient(weights) / smoothness,
                        self.block_slices_, self.kappa,
                    )
                    candidate_objective = objective(candidate)
                next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
                extrapolated = candidate + ((momentum - 1.0) / next_momentum) * (candidate - weights)
                weights, current, momentum = candidate, candidate_objective, next_momentum
                iterations = iteration
                if iteration == 1 or iteration % 10 == 0 or iteration == self.max_iter:
                    current_gap = gap(weights)
                    if current_gap <= self.tol * max(1.0, abs(current)):
                        break
        current_gap = gap(weights)
        self.coef_ = weights
        self.intercept_ = self.y_mean_ - self.x_mean_ @ weights
        self.block_norms_ = np.array([np.linalg.norm(weights[s]) for s in self.block_slices_])
        self.norm_sum_ = float(self.block_norms_.sum())
        self.frobenius_norm_ = float(np.linalg.norm(weights))
        self.lipschitz_ = sum(np.linalg.norm(weights[s], axis=0) for s in self.block_slices_)
        self.optimization_gap_ = current_gap
        self.objective_ = current
        self.n_iter_ = iterations
        self.converged_ = current_gap <= self.tol * max(1.0, abs(current))
        self.n_features_in_, self.n_outputs_ = X.shape[1], Y.shape[1]
        if self.bounds is not None:
            lower, upper = (np.asarray(b, dtype=float) for b in self.bounds)
            lower = np.broadcast_to(lower, (self.n_outputs_,)).copy()
            upper = np.broadcast_to(upper, (self.n_outputs_,)).copy()
            if not (np.isfinite(lower).all() and np.isfinite(upper).all()
                    and np.all(lower < upper)):
                raise ValueError("Bounds must be finite and strictly ordered.")
            self.lower_, self.upper_ = lower, upper
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("Fit the decoder before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Invalid prediction matrix.")
        prediction = X @ self.coef_ + self.intercept_
        return (prediction if self.bounds is None
                else np.clip(prediction, self.lower_, self.upper_))


class Persistence:
    """Repeat the most recent scalar observation at every horizon."""

    def fit(self, X, Y):
        Y = np.asarray(Y)
        self.n_outputs_ = Y.shape[1] if Y.ndim == 2 else 1
        return self

    def predict(self, X):
        # histories_targets uses oldest-to-newest ordering.
        X = np.asarray(X, dtype=float)
        return np.repeat(X[:, -1:], self.n_outputs_, axis=1)


class PolynomialMap:
    """Direct polynomial regression features, not iterated EDMD dynamics."""

    def __init__(self, degree=2):
        self.transformer = PolynomialFeatures(degree=int(degree), include_bias=False)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.maximum(X.std(axis=0), 1e-8)
        self.transformer.fit((X - self.mean_) / self.scale_)
        return self

    def transform(self, X):
        return self.transformer.transform((np.asarray(X) - self.mean_) / self.scale_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KoopmanRidge:
    """Iterated affine feature regression with a scalar observation decoder.

    Fit ``F_next = F @ A + b`` and ``current_y = F @ d + c`` using ridge
    regression. Forecast by composing the learned affine transition exactly,
    then decode and clip the scalar observation to the declared target bounds.
    No simplex projection or intermediate output clipping is performed. This
    empirical feature-transition fit does not assert exact Koopman closure.

    ``predict(F, horizons)`` has shape ``(samples, len(horizons))``.
    ``lipschitz_for(horizons)`` returns the Euclidean coefficient norm of each
    composed horizon decoder, valid also after scalar output clipping.
    """

    def __init__(self, alpha=1e-3, bounds=(-1.0, 1.0)):
        self.alpha = float(alpha)
        self.bounds = bounds

    def fit(self, F, F_next, current_y):
        F = np.asarray(F, dtype=float)
        F_next = np.asarray(F_next, dtype=float)
        y = np.asarray(current_y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if F.ndim != 2 or F_next.shape != F.shape:
            raise ValueError("Current and next features must be matching matrices.")
        if y.shape != (len(F), 1):
            raise ValueError("current_y must have one scalar output per feature row.")
        transition = BoundedRidge(alpha=self.alpha, bounds=None).fit(F, F_next)
        decoder = BoundedRidge(alpha=self.alpha, bounds=None).fit(F, y)
        self.transition_ = transition.coef_
        self.shift_ = transition.intercept_
        self.decoder_ = decoder.coef_[:, 0]
        self.decoder_intercept_ = float(decoder.intercept_[0])
        self.n_features_in_ = F.shape[1]
        self._composition_cache = {}
        if self.bounds is not None:
            lower, upper = (float(v) for v in self.bounds)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError("Bounds must be finite and strictly ordered.")
            self.lower_, self.upper_ = lower, upper
        return self

    def effective_coefficients(self, horizons):
        """Return ``(W,b)`` such that the unclipped forecast is ``F @ W + b``."""
        if not hasattr(self, "transition_"):
            raise ValueError("Fit the feature transition before forecasting.")
        raw = np.asarray(horizons)
        if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Horizons must be a nonempty vector of positive integers.")
        hs = tuple(int(h) for h in raw)
        if min(hs) < 1:
            raise ValueError("Horizons must be positive integers.")
        if hs not in self._composition_cache:
            coeff, offset = self.decoder_.copy(), self.decoder_intercept_
            required, compositions = set(hs), {}
            for step in range(1, max(hs) + 1):
                # Compose the previous scalar forecast with one affine state step.
                offset = float(self.shift_ @ coeff + offset)
                coeff = self.transition_ @ coeff
                if step in required:
                    compositions[step] = (coeff.copy(), offset)
            W = np.column_stack([compositions[h][0] for h in hs])
            b = np.array([compositions[h][1] for h in hs])
            if not np.isfinite(W).all() or not np.isfinite(b).all():
                raise FloatingPointError("The learned feature rollout overflowed.")
            self._composition_cache[hs] = (W, b)
        W, b = self._composition_cache[hs]
        return W.copy(), b.copy()

    def lipschitz_for(self, horizons):
        W, _ = self.effective_coefficients(horizons)
        return np.linalg.norm(W, axis=0)

    def predict(self, F, horizons):
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or not np.isfinite(F).all():
            raise ValueError("Prediction features must be a finite matrix.")
        W, b = self.effective_coefficients(horizons)
        if F.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature dimension differs from training.")
        prediction = F @ W + b
        return (prediction if self.bounds is None
                else np.clip(prediction, self.lower_, self.upper_))

