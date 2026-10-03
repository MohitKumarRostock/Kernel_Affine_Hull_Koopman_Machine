"""Fixed out-of-sample reference-class map for fitted KAHM abstractions.

The KAHKM abstraction is trained from state-space K-means reference classes,
but the stored model retains the final cluster centers rather than the original
scikit-learn estimator. For an independent evaluation state x, this module
defines c(x) by the Euclidean-nearest stored reference-class center.

This is the standard Voronoi extension associated with K-means centers and is
kept distinct from argmax(Phi(x)): the hard reference label and the largest
soft coordinate play different roles in the manuscript.

The map uses only the already fitted abstraction model. It does not inspect
successor states, certificate values, prediction errors, or evaluation-sample
statistics. Therefore it can be frozen before independent certificate
evaluation.

Input convention follows the KAHKM implementation:
    X.shape == (D, N)
    model["cluster_centers"].shape == (D, C)

Returned labels are zero-based integers in {0,...,C-1}. Ties are resolved by
numpy.argmin, hence by the smallest class index. This deterministic tie rule is
part of the class-map definition and must remain fixed for reproduction.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


REFERENCE_CLASS_MAP_ID = "nearest_stored_kmeans_center_v1"


@dataclass(frozen=True)
class ReferenceClassResult:
    """Labels plus diagnostic squared distance to the selected center."""

    class_map_id: str
    labels: tuple[int, ...]
    selected_squared_distances: tuple[float, ...]
    n_states: int
    n_classes: int
    state_dimension: int


def _finite_matrix(name: str, value: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(value)

    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")

    if raw.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")

    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(f"{name} must have nonzero dimensions.")

    array = np.asarray(raw, dtype=np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def reference_class_labels(
    *,
    abstraction_model: dict[str, Any],
    X: ArrayLike,
) -> ReferenceClassResult:
    """Assign each state to its Euclidean-nearest stored reference-class center.

    No normalization or scaling is applied: this uses the same raw state
    coordinate convention under which the reference classes were constructed.

    The implementation computes squared distances directly and never takes a
    square root. This preserves the nearest-center ordering while avoiding an
    unnecessary numerical operation.
    """
    if not isinstance(abstraction_model, dict):
        raise TypeError("abstraction_model must be a dictionary.")

    if "cluster_centers" not in abstraction_model:
        raise ValueError(
            "abstraction_model does not contain 'cluster_centers'."
        )

    centers = _finite_matrix(
        "cluster_centers",
        abstraction_model["cluster_centers"],
    )
    states = _finite_matrix("X", X)

    if states.shape[0] != centers.shape[0]:
        raise ValueError(
            "State dimension does not match stored cluster-center dimension: "
            f"{states.shape[0]} != {centers.shape[0]}."
        )

    # centers: D x C, states: D x N
    #
    # Compute distances from coordinate differences rather than through
    # ||c||^2 + ||x||^2 - 2 c^T x. The Gram-form identity is algebraically
    # equivalent but can lose precision through catastrophic cancellation
    # when centers and states share a large common translation.
    #
    # Looping over classes keeps the largest temporary at D x N rather than
    # allocating a D x C x N tensor.
    squared = np.empty(
        (centers.shape[1], states.shape[1]),
        dtype=np.float64,
    )

    for class_index in range(centers.shape[1]):
        difference = (
            states
            - centers[:, class_index:class_index + 1]
        )

        squared[class_index] = np.sum(
            difference * difference,
            axis=0,
            dtype=np.float64,
        )

    if not np.all(np.isfinite(squared)):
        raise RuntimeError(
            "Squared-distance computation produced a nonfinite value."
        )

    if np.any(squared < 0.0):
        raise RuntimeError(
            "Squared-distance computation produced a negative value."
        )

    labels = np.argmin(
        squared,
        axis=0,
    ).astype(np.int64, copy=False)

    chosen = squared[
        labels,
        np.arange(states.shape[1]),
    ]

    return ReferenceClassResult(
        class_map_id=REFERENCE_CLASS_MAP_ID,
        labels=tuple(int(value) for value in labels),
        selected_squared_distances=tuple(float(value) for value in chosen),
        n_states=int(states.shape[1]),
        n_classes=int(centers.shape[1]),
        state_dimension=int(states.shape[0]),
    )
