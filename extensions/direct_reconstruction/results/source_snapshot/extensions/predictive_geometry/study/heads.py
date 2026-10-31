"""Validation-selected diagnostic heads; every fitted quantity uses training data.

The objective is mean squared error plus alpha * squared coefficient norm.
The intercept is unpenalized.  These are diagnostic heads, not the original
manuscript's fixed Frobenius/block-norm constrained decoder classes.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist, pdist


ALPHAS = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
BANDWIDTH_MULTIPLIERS = (0.25, 1.0, 4.0)


def group_mean(values, groups=None):
    values = np.asarray(values, dtype=float)
    if groups is None:
        return values.mean(axis=0)
    _, inv = np.unique(groups, return_inverse=True)
    totals = np.zeros((inv.max() + 1,) + values.shape[1:])
    np.add.at(totals, inv, values)
    counts = np.bincount(inv)
    return (totals / counts.reshape((-1,) + (1,) * (values.ndim - 1))).mean(axis=0)


def grouped_losses(prediction, target, groups):
    values = (np.asarray(prediction) - np.asarray(target)) ** 2
    ids, inv = np.unique(groups, return_inverse=True)
    totals = np.zeros((len(ids), values.shape[1]))
    np.add.at(totals, inv, values)
    return totals / np.bincount(inv)[:, None]


def fit_standardizer(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def quadratic(X):
    """Linear terms and all unique degree-two products; no constant column."""
    ii, jj = np.triu_indices(X.shape[1])
    return np.column_stack((X, X[:, ii] * X[:, jj]))


@dataclass
class Basis:
    kind: str
    mean: np.ndarray
    scale: np.ndarray
    landmarks: np.ndarray | None = None
    inverse_sqrt: np.ndarray | None = None
    bandwidth_sq: float | None = None
    multiplier: float | None = None

    def transform(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean) / self.scale
        if self.kind == "ridge":
            return Z
        kernel = np.exp(-cdist(Z, self.landmarks, "sqeuclidean") / (2 * self.bandwidth_sq))
        return np.column_stack((Z, kernel @ self.inverse_sqrt))


@dataclass
class FittedHead:
    bases: list
    basis_indices: list
    means: list
    scales: list
    coefficients: list
    intercepts: list
    bounds: tuple | None
    separate_targets: bool

    def predict(self, X):
        cache = {j: self.bases[j].transform(X) for j in set(self.basis_indices)}
        predictions = []
        for j, mean, scale, coef, intercept in zip(
                self.basis_indices, self.means, self.scales,
                self.coefficients, self.intercepts):
            predictions.append(((cache[j] - mean) / scale) @ coef + intercept)
        result = np.column_stack(predictions) if self.separate_targets else predictions[0]
        if self.bounds is not None:
            result = np.clip(result, *self.bounds)
        return result


def _bases(X, head, seed, n_landmarks, bandwidth_multipliers):
    mean, scale = fit_standardizer(X)
    if head == "ridge":
        return [Basis(head, mean, scale)]
    if head != "rbf":
        raise ValueError(head)
    Z = (X - mean) / scale
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(X), size=min(n_landmarks, len(X)), replace=False)
    landmarks = Z[ids].copy()
    distances = pdist(landmarks, "sqeuclidean")
    positive = distances[distances > 1e-14]
    median_sq = float(np.median(positive)) if len(positive) else 1.0
    output = []
    for multiplier in bandwidth_multipliers:
        bandwidth_sq = max(median_sq * multiplier ** 2, 1e-12)
        Kmm = np.exp(-cdist(landmarks, landmarks, "sqeuclidean") / (2 * bandwidth_sq))
        eig, vec = np.linalg.eigh((Kmm + Kmm.T) / 2)
        keep = eig > max(float(eig.max()) * 1e-10, 1e-12)
        inverse_sqrt = vec[:, keep] / np.sqrt(eig[keep])[None, :]
        output.append(Basis(head, mean, scale, landmarks, inverse_sqrt,
                            bandwidth_sq, float(multiplier)))
    return output


def fit_predict(X, Y, X_validation, Y_validation, X_test, *, head="ridge",
                seed=0, bounds=None, separate_targets=False,
                validation_groups=None, alphas=ALPHAS, n_landmarks=64,
                bandwidth_multipliers=BANDWIDTH_MULTIPLIERS):
    """Tune on validation only, retain training fit, then predict untouched test.

    Forecast horizons share one selected alpha/bandwidth.  Distinct branch
    target laws select independently, even when solved together for efficiency.
    The RBF head is a 64-landmark Nystrom approximation with a linear component;
    it is a finite diagnostic model, not an optimal-decoder oracle.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    bases = _bases(X, head, seed, n_landmarks, bandwidth_multipliers)
    target_mean = Y.mean(axis=0)
    centered_y = Y - target_mean
    candidates = []
    for basis_index, basis in enumerate(bases):
        F = basis.transform(X)
        Fv = basis.transform(X_validation)
        mean, scale = fit_standardizer(F)
        Z, Zv = (F - mean) / scale, (Fv - mean) / scale
        gram = (Z.T @ Z) / len(Z)
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        rhs = eigenvectors.T @ ((Z.T @ centered_y) / len(Z))
        for alpha in alphas:
            coef = eigenvectors @ (rhs / (eigenvalues + alpha)[:, None])
            pred = Zv @ coef + target_mean
            if bounds is not None:
                pred = np.clip(pred, *bounds)
            losses = group_mean((pred - Y_validation) ** 2, validation_groups)
            if not np.isfinite(losses).all():
                raise FloatingPointError("Non-finite validation loss")
            candidates.append({"basis": basis_index, "mean": mean, "scale": scale,
                               "coef": coef, "losses": losses, "alpha": float(alpha)})
    losses = np.asarray([item["losses"] for item in candidates])
    choices = np.argmin(losses, axis=0) if separate_targets else [int(np.argmin(losses.mean(axis=1)))]
    selected, model = [], FittedHead(bases, [], [], [], [], [], bounds, separate_targets)
    for column, choice in enumerate(choices):
        item = candidates[int(choice)]
        j = item["basis"]
        selection = slice(column, column + 1) if separate_targets else slice(None)
        model.basis_indices.append(j)
        model.means.append(item["mean"])
        model.scales.append(item["scale"])
        model.coefficients.append(item["coef"][:, selection].copy())
        model.intercepts.append(target_mean[selection].copy())
        selected.append({"alpha": item["alpha"],
                         "bandwidth_multiplier": bases[j].multiplier,
                         "bandwidth_sq": bases[j].bandwidth_sq,
                         "basis_dimension": len(item["mean"]),
                         "validation_mse_by_target": item["losses"][selection].tolist()})
    prediction = model.predict(X_test)
    return prediction, selected, model
