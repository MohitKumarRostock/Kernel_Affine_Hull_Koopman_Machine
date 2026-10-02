"""Empirical statistics for the original i.i.d.-pair certificate.

Reference: arXiv:2609.32652v1, Proposition 2, pp. 6-7.
Rows represent evaluation pairs. Both coordinate arrays have shape (M, C).
Labels are the fixed CURRENT-state reference labels, encoded as 0,...,C-1.
They are not inferred from argmax coordinates or successor states.

Computation uses float64, with a fixed row-sum tolerance of 1e-12.
Entries outside [0, 1] are rejected. No clipping, renormalization, row
removal, label remapping, or variance-bias correction is performed.
Independence and separation from construction/selection data are caller
responsibilities; these properties cannot be checked from the arrays.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

STATISTICS_ID = "original_iid_pair_statistics_v1"
SIMPLEX_SUM_ATOL = 1e-12


@dataclass(frozen=True)
class PairStatistics:
    """Immutable summary; class-indexed tuples retain empty classes."""

    n_pairs: int
    n_classes: int
    f_hat: float
    s_hat: float
    class_counts: tuple[int, ...]
    class_successor_means: tuple[tuple[float, ...], ...]
    class_successor_sse: tuple[float, ...]
    current_max_row_sum_error: float
    successor_max_row_sum_error: float
    simplex_sum_atol: float


def _simplex_rows(
    name: str, value: ArrayLike
) -> tuple[NDArray[np.float64], float]:
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{name} must not be a masked array.")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric coordinates.")
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(f"{name} must have nonempty shape (M, C).")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates.")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} entries must lie in [0, 1].")
    error = float(np.max(np.abs(
        np.sum(array, axis=1, dtype=np.float64) - 1.0
    )))
    if error > SIMPLEX_SUM_ATOL:
        raise ValueError(
            f"{name} rows must sum to 1 within {SIMPLEX_SUM_ATOL}; "
            f"maximum error was {error}."
        )
    return array, error


def statistics_from_pairs(
    *, phi: ArrayLike, successors: ArrayLike, labels: ArrayLike
) -> PairStatistics:
    """Compute f_hat and s_hat, each normalized by the pair count M.

    Successors are already encoded vectors Z_i = Phi(X_i^+), not physical
    states. Each observed class mean is computed from these same pairs.
    An empty class is assigned the uniform simplex vector as a placeholder;
    its count and sum of squared residuals are zero. A singleton class also
    contributes zero residual variance.

    No independence claim or confidence bound is produced by this function.
    Retain input pairs or generating sufficient statistics separately: this
    summary alone does not reconstruct the evaluation dataset.
    """
    current, current_error = _simplex_rows("phi", phi)
    successor, successor_error = _simplex_rows("successors", successors)
    if current.shape != successor.shape:
        raise ValueError("phi and successors must have the same (M, C) shape.")
    M, C = current.shape

    if np.ma.isMaskedArray(labels):
        raise TypeError("labels must not be a masked array.")
    classes = np.asarray(labels)
    if classes.ndim != 1 or classes.shape[0] != M:
        raise ValueError("labels must have shape (M,).")
    if classes.dtype.kind not in "iu":
        raise TypeError("labels must have integer dtype, not boolean or float.")
    if np.any(classes < 0) or np.any(classes >= C):
        raise ValueError("labels must be zero-based integers in [0, C-1].")
    classes = classes.astype(np.intp, copy=False)

    counts = np.bincount(classes, minlength=C)
    sums = np.zeros((C, C), dtype=np.float64)
    np.add.at(sums, classes, successor)
    means = np.full((C, C), 1.0 / C, dtype=np.float64)
    occupied = counts > 0
    means[occupied] = sums[occupied] / counts[occupied, None]

    # Center first: avoid subtracting two nearly equal sums of squares.
    residuals = successor - means[classes]
    row_sse = np.sum(residuals * residuals, axis=1, dtype=np.float64)
    class_sse = np.bincount(classes, weights=row_sse, minlength=C)
    assignment_losses = (1.0 - current[np.arange(M), classes]) ** 2

    return PairStatistics(
        n_pairs=int(M),
        n_classes=int(C),
        f_hat=float(np.mean(assignment_losses, dtype=np.float64)),
        s_hat=float(np.sum(class_sse, dtype=np.float64) / M),
        class_counts=tuple(int(n) for n in counts),
        class_successor_means=tuple(
            tuple(float(x) for x in row) for row in means
        ),
        class_successor_sse=tuple(float(x) for x in class_sse),
        current_max_row_sum_error=current_error,
        successor_max_row_sum_error=successor_error,
        simplex_sum_atol=SIMPLEX_SUM_ATOL,
    )
