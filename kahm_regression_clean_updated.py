"""
KAHM-based multivariate regression via output clustering.



The model keeps the original KAHM workflow:
1. cluster target/output samples Y,
2. train one KAHM/OTFL autoencoder classifier per output cluster,
3. predict by hard nearest-cluster assignment or soft distance-to-probability mixing.

Expected data layout
--------------------
X: (D_in, N)  input features as columns
Y: (D_out, N) regression targets as columns

Requires
--------
pip install numpy scikit-learn joblib

The local OTFL/KAHM modules must also be importable:
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py
"""

from __future__ import annotations

import gc
import inspect
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast, overload
from uuid import uuid4

import numpy as np
from joblib import Parallel, delayed, dump, load
from numpy.typing import DTypeLike
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

try:
    from parallel_autoencoders import parallel_autoencoders  # type: ignore[import]
    from combine_multiple_autoencoders_extended import (  # type: ignore[import]
        combine_multiple_autoencoders_extended,
    )
except ImportError as exc:  # pragma: no cover - depends on local project files
    raise ImportError(
        "Could not import the local KAHM/OTFL helpers. Ensure that "
        "parallel_autoencoders.py and combine_multiple_autoencoders_extended.py "
        "are available on PYTHONPATH or in the same directory as this script."
    ) from exc

Pathish = str | os.PathLike[str] | Path
KahmModel = dict[str, Any]
AutoencoderRef = Any


@dataclass(frozen=True)
class SoftTuningResult:
    """Result returned by tune_soft_params."""

    best_alpha: float
    best_topk: int | None
    best_mse: float


@dataclass(frozen=True)
class NLMSCenterTuningResult:
    """Result returned by tune_cluster_centers_nlms."""

    mu: float
    epsilon: float
    epochs: int
    batch_size: int
    final_mse: float
    mse_history: tuple[float, ...]


# -----------------------------------------------------------------------------
# Array, dtype, and classifier helpers
# -----------------------------------------------------------------------------


def _as_float_ndarray(x: Any, *, min_dtype: DTypeLike = np.float32) -> np.ndarray:
    """Convert input to a floating ndarray while preserving existing precision."""
    arr = np.asarray(x)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64, copy=False)
    dtype = np.result_type(arr.dtype, min_dtype)
    return arr.astype(dtype, copy=False)


def _resolve_work_dtype(model_dtype: str, *arrays: np.ndarray) -> np.dtype[Any]:
    md = str(model_dtype).lower().strip()
    if md in {"auto", "none", ""}:
        return np.dtype(np.result_type(*(arr.dtype for arr in arrays)))
    if md in {"float32", "f32"}:
        return np.dtype(np.float32)
    if md in {"float64", "f64"}:
        return np.dtype(np.float64)
    raise ValueError("model_dtype must be one of {'auto', 'float32', 'float64'}.")


def _validate_positive_finite_float(name: str, value: float) -> float:
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value_f


def _validate_positive_int(name: str, value: int) -> int:
    value_i = int(value)
    if value_i <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value_i


def _validate_optional_topk(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(name, int(value))


def _validate_alpha_topk(alpha: float | None, topk: int | None) -> None:
    if alpha is not None:
        _validate_positive_finite_float("alpha", float(alpha))
    _validate_optional_topk("topk", topk)


def _ae_as_list(ae: AutoencoderRef) -> list[Any]:
    """Normalize an autoencoder object/reference to a flat component list."""
    if isinstance(ae, (list, tuple)):
        if len(ae) == 1 and isinstance(ae[0], (list, tuple)):
            return list(ae[0])
        return list(ae)
    return [ae]


def _call_combine_multiple_autoencoders_extended(
    X: np.ndarray,
    ae_list: Sequence[Any],
    distance_type: str,
    *,
    n_jobs: int | None = None,
) -> Any:
    """Call combine_multiple_autoencoders_extended across API variants.

    Older local helper versions did not accept ``n_jobs``. Inspecting the
    callable avoids swallowing unrelated TypeError exceptions raised inside the
    helper implementation.
    """
    fn = cast(Any, combine_multiple_autoencoders_extended)
    if n_jobs is None:
        return fn(X, ae_list, distance_type)

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        # Some callables do not expose signatures. Fall back once only for the
        # known legacy API error; otherwise preserve the original exception.
        try:
            return fn(X, ae_list, distance_type, n_jobs=n_jobs)
        except TypeError as exc:
            msg = str(exc)
            unsupported_n_jobs = "n_jobs" in msg and (
                "unexpected keyword" in msg
                or "got an unexpected" in msg
                or "invalid keyword" in msg
                or "positional arguments" in msg
            )
            if unsupported_n_jobs:
                return fn(X, ae_list, distance_type)
            raise

    accepts_n_jobs = "n_jobs" in params or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    if accepts_n_jobs:
        return fn(X, ae_list, distance_type, n_jobs=n_jobs)
    return fn(X, ae_list, distance_type)


def _is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, os.PathLike, Path))


def _resolve_ae_path(ae_ref: AutoencoderRef, classifier_dir: Any) -> AutoencoderRef:
    if not _is_pathlike(ae_ref):
        return ae_ref
    path = Path(cast(Pathish, ae_ref))
    if not path.is_absolute() and classifier_dir is not None:
        path = Path(cast(Pathish, classifier_dir)) / path
    return str(path)


def _extract_pc(obj: Any) -> np.ndarray | None:
    if isinstance(obj, dict):
        pc = obj.get("PC")
        return pc if isinstance(pc, np.ndarray) else None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            pc = _extract_pc(item)
            if pc is not None:
                return pc
    return None


def _guard_pc_shape(ae_obj: AutoencoderRef, *, d_in: int, ae_ref: Any, cluster_idx: int | None) -> None:
    """Fail early if a loaded AE shard clearly belongs to a different input dimension."""
    pc = _extract_pc(ae_obj)
    if pc is None or pc.ndim != 2:
        return
    if pc.shape[0] == d_in:
        return
    if pc.shape[1] == d_in:
        raise ValueError(
            f"Incompatible AE projection matrix PC shape {pc.shape} for input dimension D_in={d_in} "
            f"(cluster {cluster_idx}). The shape suggests a wrong/collided AE cache. AE ref={ae_ref!r}."
        )
    raise ValueError(
        f"Incompatible AE projection matrix PC shape {pc.shape} for input dimension D_in={d_in} "
        f"(cluster {cluster_idx}). AE ref={ae_ref!r}."
    )


def _downcast_obj(obj: Any, dtype: np.dtype[Any]) -> Any:
    """Recursively downcast floating arrays in model objects."""
    if isinstance(obj, np.ndarray) and obj.dtype.kind in "fc":
        return obj.astype(dtype, copy=False)
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _downcast_obj(value, dtype)
        return obj
    if isinstance(obj, list):
        return [_downcast_obj(value, dtype) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_downcast_obj(value, dtype) for value in obj)
    return obj


def _labels_to_groups(labels_zero: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Return sample-index arrays for labels 0..n_groups-1 using one stable sort."""
    labels = np.asarray(labels_zero, dtype=np.int64).reshape(-1)
    groups: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(int(n_groups))]
    if labels.size == 0:
        return groups

    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    split_points = np.flatnonzero(np.diff(sorted_labels)) + 1
    for group in np.split(order, split_points):
        if group.size:
            groups[int(labels[int(group[0])])] = group.astype(np.int64, copy=False)
    return groups


def _compute_cluster_centers(Y: np.ndarray, labels_zero: np.ndarray, n_clusters: int, dtype: np.dtype[Any]) -> np.ndarray:
    """Compute target centroids without a Python loop over clusters."""
    labels = np.asarray(labels_zero, dtype=np.int64).reshape(-1)
    if labels.size != Y.shape[1]:
        raise ValueError(f"labels length {labels.size} does not match Y sample count {Y.shape[1]}.")

    counts = np.bincount(labels, minlength=int(n_clusters)).astype(np.float64, copy=False)
    if np.any(counts <= 0):
        empty = np.where(counts <= 0)[0][:10].tolist()
        raise ValueError(f"Cannot compute centers for empty clusters. First empty labels: {empty}")

    centers_t = np.zeros((int(n_clusters), int(Y.shape[0])), dtype=dtype)
    np.add.at(centers_t, labels, Y.T.astype(dtype, copy=False))
    centers_t = centers_t / counts[:, None].astype(dtype, copy=False)
    return centers_t.T.astype(dtype, copy=False)


def _standardize_rows_for_clustering(A: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return row-wise standardized data for clustering only."""
    A_arr = _as_float_ndarray(A)
    mu = A_arr.mean(axis=1, keepdims=True)
    sd = A_arr.std(axis=1, keepdims=True)
    return ((A_arr - mu) / np.maximum(sd, eps)).astype(A_arr.dtype, copy=False)


def _make_kmeans_estimator(
    n_clusters: int,
    *,
    random_state: int | None,
    kmeans_kind: Literal["auto", "full", "minibatch"],
    kmeans_batch_size: int,
) -> KMeans | MiniBatchKMeans:
    """Create the same KMeans variant used by the main training routine."""
    n_clusters_i = _validate_positive_int("n_clusters", int(n_clusters))
    batch_size_i = _validate_positive_int("kmeans_batch_size", int(kmeans_batch_size))
    use_minibatch = kmeans_kind == "minibatch" or (kmeans_kind == "auto" and n_clusters_i >= 2000)
    if use_minibatch:
        return MiniBatchKMeans(
            n_clusters=n_clusters_i,
            random_state=random_state,
            batch_size=batch_size_i,
            n_init="auto",
            reassignment_ratio=0.01,
        )
    return KMeans(n_clusters=n_clusters_i, random_state=random_state, n_init="auto")


def _fit_kmeans_labels(
    data_t: np.ndarray,
    n_clusters: int,
    *,
    random_state: int | None,
    kmeans_kind: Literal["auto", "full", "minibatch"],
    kmeans_batch_size: int,
) -> tuple[KMeans | MiniBatchKMeans, np.ndarray]:
    """Fit a KMeans variant on row-major samples and return zero-based labels."""
    estimator = _make_kmeans_estimator(
        n_clusters,
        random_state=random_state,
        kmeans_kind=kmeans_kind,
        kmeans_batch_size=kmeans_batch_size,
    )
    estimator.fit(np.ascontiguousarray(data_t))
    return estimator, np.asarray(estimator.labels_, dtype=np.int64)


def _hierarchical_y_then_x_labels(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_y_clusters: int,
    max_x_splits_per_y_cluster: int,
    min_child_size: int,
    random_state: int | None,
    kmeans_kind: Literal["auto", "full", "minibatch"],
    kmeans_batch_size: int,
    standardize_x: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Cluster Y first, then split each Y-cluster into X-space child clusters.

    The returned labels are zero-based and contiguous. Child-cluster centroids are
    still computed from Y later in the normal training pipeline; this function
    only decides which samples belong to each final classifier/prototype group.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays shaped (D, N).")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must contain the same number of samples.")

    n_samples = int(X.shape[1])
    n_y_clusters_i = _validate_positive_int("n_y_clusters", int(n_y_clusters))
    max_splits_i = _validate_positive_int("max_x_splits_per_y_cluster", int(max_x_splits_per_y_cluster))
    min_child_i = _validate_positive_int("min_child_size", int(min_child_size))
    if n_y_clusters_i > n_samples:
        raise ValueError(f"n_y_clusters={n_y_clusters_i} cannot exceed number of samples N={n_samples}.")

    if verbose:
        print(f"Running parent KMeans on Y with {n_y_clusters_i} cluster(s)...")
    _, parent_labels = _fit_kmeans_labels(
        Y.T,
        n_y_clusters_i,
        random_state=random_state,
        kmeans_kind=kmeans_kind,
        kmeans_batch_size=kmeans_batch_size,
    )

    labels = np.full(n_samples, -1, dtype=np.int64)
    next_label = 0

    for parent_idx in range(n_y_clusters_i):
        sample_idx = np.where(parent_labels == int(parent_idx))[0]
        parent_size = int(sample_idx.size)
        if parent_size == 0:
            continue

        max_feasible_splits = parent_size // min_child_i
        requested_splits = min(max_splits_i, max_feasible_splits)
        if requested_splits <= 1:
            labels[sample_idx] = next_label
            next_label += 1
            continue

        X_parent = X[:, sample_idx]
        if standardize_x:
            X_parent = _standardize_rows_for_clustering(X_parent)

        accepted_child_labels: np.ndarray | None = None
        accepted_child_count = 1

        # KMeans can produce imbalanced child clusters. Back off until each child
        # has enough samples for downstream autoencoder training.
        for child_count in range(int(requested_splits), 1, -1):
            try:
                _, child_labels = _fit_kmeans_labels(
                    X_parent.T,
                    child_count,
                    random_state=random_state,
                    kmeans_kind=kmeans_kind,
                    kmeans_batch_size=kmeans_batch_size,
                )
            except ValueError:
                continue

            child_counts = np.bincount(child_labels, minlength=int(child_count))
            if np.all(child_counts >= min_child_i):
                accepted_child_labels = child_labels
                accepted_child_count = int(child_count)
                break

        if accepted_child_labels is None:
            labels[sample_idx] = next_label
            next_label += 1
            continue

        for child_idx in range(accepted_child_count):
            child_sample_idx = sample_idx[accepted_child_labels == int(child_idx)]
            if child_sample_idx.size == 0:
                continue
            labels[child_sample_idx] = next_label
            next_label += 1

    if np.any(labels < 0):
        raise RuntimeError("Internal error: not all samples received hierarchical labels.")
    return labels.astype(np.int64, copy=False)


def _resolve_classifier_entries(model: KahmModel) -> list[AutoencoderRef]:
    cached = model.get("_classifier_cache")
    if isinstance(cached, (list, tuple)) and len(cached) > 0:
        return list(cached)

    clf = model.get("classifier")
    if isinstance(clf, (str, os.PathLike, Path)):
        path = Path(cast(Pathish, clf))
        if path.is_dir():
            files = sorted(path.rglob("*.joblib"))
            if not files:
                raise TypeError(f"Classifier directory contains no *.joblib files: {path}")
            return [str(file) for file in files]
        raise TypeError("A string classifier must be a directory path or a list of AE shard paths.")
    if isinstance(clf, (list, tuple)) and len(clf) > 0:
        return list(clf)
    raise TypeError("model['classifier'] must be a non-empty list/tuple, or a directory containing AE joblib files.")


def _load_ae_maybe(model: KahmModel, ae_ref: AutoencoderRef, *, cluster_idx: int, d_in: int) -> tuple[AutoencoderRef, bool]:
    cache = model.get("_classifier_cache")
    if isinstance(cache, (list, tuple)) and 0 <= cluster_idx < len(cache):
        ae_obj = cache[cluster_idx]
        _guard_pc_shape(ae_obj, d_in=d_in, ae_ref="(preloaded cache)", cluster_idx=cluster_idx + 1)
        return ae_obj, False

    resolved = _resolve_ae_path(ae_ref, model.get("classifier_dir"))
    if _is_pathlike(resolved):
        ae_obj = load(cast(str, resolved))
        _guard_pc_shape(ae_obj, d_in=d_in, ae_ref=resolved, cluster_idx=cluster_idx + 1)
        return ae_obj, True

    _guard_pc_shape(resolved, d_in=d_in, ae_ref="(in memory)", cluster_idx=cluster_idx + 1)
    return resolved, False


def _validate_classifier_assets_available(model: KahmModel, *, context: str) -> None:
    """Validate that disk-backed AE shard references are resolvable."""
    clf = model.get("classifier")
    clf_dir = model.get("classifier_dir")

    if isinstance(clf, (str, os.PathLike, Path)):
        path = Path(cast(Pathish, clf))
        if not path.exists():
            raise FileNotFoundError(f"Classifier path for {context} does not exist: {path}")
        if path.is_dir() and not any(path.rglob("*.joblib")):
            raise FileNotFoundError(f"Classifier directory for {context} contains no *.joblib shards: {path}")
        return

    if not isinstance(clf, (list, tuple)):
        return

    pathlike_entries = [entry for entry in clf if _is_pathlike(entry)]
    if not pathlike_entries:
        return

    if clf_dir is None:
        unresolved = [str(entry) for entry in pathlike_entries if not Path(cast(Pathish, entry)).is_absolute()]
        if unresolved:
            raise FileNotFoundError(
                f"Model has relative classifier shard paths but no classifier_dir for {context}. "
                f"First unresolved shard: {unresolved[0]!r}."
            )

    missing: list[str] = []
    for entry in pathlike_entries:
        path = Path(cast(Pathish, entry))
        if not path.is_absolute() and clf_dir is not None:
            path = Path(cast(Pathish, clf_dir)) / path
        if not path.exists():
            missing.append(str(path))
            if len(missing) >= 5:
                break

    if missing:
        raise FileNotFoundError(
            f"Missing classifier shard file(s) for {context}. "
            f"First missing paths: {missing}. Keep the saved model together with its AE shard directory."
        )


def _maybe_tqdm_total(total: int, desc: str, unit: str, enabled: bool) -> Any:
    if not enabled:
        return None
    try:
        from tqdm import tqdm  # type: ignore[import]

        return tqdm(total=int(total), desc=desc, unit=unit, leave=False)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_kahm_regressor(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters: int,
    subspace_dim: int = 20,
    Nb: int = 100,
    random_state: int | None = 0,
    verbose: bool = True,
    *,
    kmeans_kind: Literal["auto", "full", "minibatch"] = "auto",
    kmeans_batch_size: int = 4096,
    max_train_per_cluster: int | None = None,
    model_dtype: Literal["auto", "float32", "float64"] = "auto",
    save_ae_to_disk: bool = True,
    ae_dir: Pathish | None = None,
    ae_cache_root: Pathish = "kahm_ae_cache",
    overwrite_ae_dir: bool = False,
    model_id: str | None = None,
    ae_compress: int = 3,
    singleton_strategy: Literal["augment", "merge"] = "augment",
    singleton_aux_mix: float = 0.05,
    cluster_strategy: Literal["kmeans_y", "y_then_x"] = "y_then_x",
    n_y_clusters: int | None = None,
    max_x_splits_per_y_cluster: int = 3,
    min_child_size: int = 2,
    standardize_x_for_cluster_split: bool = True,
) -> KahmModel:
    """Train a KAHM regressor without input scaling or L2 normalization.

    Parameters are intentionally restricted to operations that do not rescale input
    columns or normalize target centroids. Existing callers that previously passed
    `input_scale` or `cluster_center_normalization` should remove those arguments.
    """
    X_arr = _as_float_ndarray(X)
    Y_arr = _as_float_ndarray(Y)
    if X_arr.ndim != 2 or Y_arr.ndim != 2:
        raise ValueError("X and Y must be 2D arrays shaped (D, N).")
    if X_arr.shape[1] != Y_arr.shape[1]:
        raise ValueError(f"X and Y must have the same number of samples. Got {X_arr.shape=} and {Y_arr.shape=}.")

    n_clusters_i = _validate_positive_int("n_clusters", int(n_clusters))
    subspace_dim_i = _validate_positive_int("subspace_dim", int(subspace_dim))
    nb_i = _validate_positive_int("Nb", int(Nb))
    kmeans_batch_size_i = _validate_positive_int("kmeans_batch_size", int(kmeans_batch_size))
    if max_train_per_cluster is not None and int(max_train_per_cluster) < 2:
        raise ValueError("max_train_per_cluster must be None or an integer >= 2.")
    if kmeans_kind not in {"auto", "full", "minibatch"}:
        raise ValueError("kmeans_kind must be one of {'auto', 'full', 'minibatch'}.")
    if singleton_strategy not in {"augment", "merge"}:
        raise ValueError("singleton_strategy must be one of {'augment', 'merge'}.")
    if cluster_strategy not in {"kmeans_y", "y_then_x"}:
        raise ValueError("cluster_strategy must be one of {'kmeans_y', 'y_then_x'}.")
    max_x_splits_i = _validate_positive_int("max_x_splits_per_y_cluster", int(max_x_splits_per_y_cluster))
    min_child_size_i = _validate_positive_int("min_child_size", int(min_child_size))
    n_y_clusters_i: int | None = None
    if n_y_clusters is not None:
        n_y_clusters_i = _validate_positive_int("n_y_clusters", int(n_y_clusters))

    d_in, n_samples = X_arr.shape
    d_out = int(Y_arr.shape[0])
    if n_clusters_i > n_samples:
        raise ValueError(f"n_clusters={n_clusters_i} cannot exceed number of samples N={int(n_samples)}.")

    work_dtype = _resolve_work_dtype(model_dtype, X_arr, Y_arr)
    X_arr = X_arr.astype(work_dtype, copy=False)
    Y_arr = Y_arr.astype(work_dtype, copy=False)

    if verbose:
        print(f"Training KAHM regressor on {int(n_samples)} samples.")
        print(f"Input dim: {int(d_in)}, output dim: {d_out}")
        print(f"Requested clusters: {n_clusters_i}")
        print(f"Cluster strategy: {cluster_strategy}")
        print("Input scaling: disabled")
        print("L2 normalization: disabled")

    # 1) Cluster samples. The rest of the training path consumes only labels_zero,
    # so hierarchical clustering can plug in without changing inference.
    skip_kmeans = cluster_strategy == "kmeans_y" and n_clusters_i == int(n_samples)
    kmeans: KMeans | MiniBatchKMeans | None
    if skip_kmeans:
        if verbose:
            print("Skipping KMeans because n_clusters equals the number of samples.")
        kmeans = None
        labels_zero = np.arange(int(n_samples), dtype=np.int64)
    elif cluster_strategy == "y_then_x":
        print(f"Using hierarchical y_then_x clustering strategy with n_y_clusters={n_y_clusters_i} and max_x_splits_per_y_cluster={max_x_splits_i}...")
        if n_y_clusters_i is None:
            # Interpret n_clusters as an approximate requested final cluster count.
            n_y_clusters_i = max(1, int(np.ceil(n_clusters_i / max(1, max_x_splits_i))))
        if n_y_clusters_i > int(n_samples):
            raise ValueError(f"n_y_clusters={n_y_clusters_i} cannot exceed number of samples N={int(n_samples)}.")
        if verbose:
            print(
                "Running hierarchical y_then_x clustering: "
                f"parent Y clusters={n_y_clusters_i}, "
                f"max X splits per parent={max_x_splits_i}, "
                f"min child size={min_child_size_i}"
            )
        labels_zero = _hierarchical_y_then_x_labels(
            X_arr,
            Y_arr,
            n_y_clusters=n_y_clusters_i,
            max_x_splits_per_y_cluster=max_x_splits_i,
            min_child_size=min_child_size_i,
            random_state=random_state,
            kmeans_kind=kmeans_kind,
            kmeans_batch_size=kmeans_batch_size_i,
            standardize_x=bool(standardize_x_for_cluster_split),
            verbose=verbose,
        )
        kmeans = None
        n_clusters_i = int(np.max(labels_zero)) + 1
        if verbose:
            print(f"Hierarchical clustering produced {n_clusters_i} final child cluster(s).")
    else:
        if verbose:
            use_minibatch = kmeans_kind == "minibatch" or (kmeans_kind == "auto" and n_clusters_i >= 2000)
            if use_minibatch:
                print(f"Running MiniBatchKMeans on Y with batch_size={kmeans_batch_size_i}...")
            else:
                print("Running full KMeans on Y...")
        kmeans, labels_zero = _fit_kmeans_labels(
            Y_arr.T,
            n_clusters_i,
            random_state=random_state,
            kmeans_kind=kmeans_kind,
            kmeans_batch_size=kmeans_batch_size_i,
        )

    # 1a) Handle singleton clusters.
    counts = np.bincount(labels_zero, minlength=n_clusters_i)
    singletons = np.where(counts == 1)[0]
    if singletons.size and verbose:
        print(f"Handling {int(singletons.size)} singleton cluster(s) using strategy={singleton_strategy!r}...")

    if singletons.size:
        if singleton_strategy == "merge":
            if kmeans is None and skip_kmeans:
                raise ValueError("singleton_strategy='merge' cannot be used when each sample is its own cluster.")
            if kmeans is not None:
                centers = np.asarray(kmeans.cluster_centers_, dtype=work_dtype)
            else:
                centers = np.asarray(_compute_cluster_centers(Y_arr, labels_zero, n_clusters_i, work_dtype).T, dtype=work_dtype)
            centers_sq = np.einsum("ij,ij->i", centers, centers)
            for cluster in singletons:
                sample_indices = np.where(labels_zero == int(cluster))[0]
                if sample_indices.size != 1:
                    continue
                sample_idx = int(sample_indices[0])
                y_sample = Y_arr[:, sample_idx]
                d2 = centers_sq + float(np.dot(y_sample, y_sample)) - 2.0 * centers.dot(y_sample)
                candidates = np.where(counts >= 2)[0]
                candidates = candidates[candidates != int(cluster)]
                if candidates.size == 0:
                    continue
                target = int(candidates[np.argmin(d2[candidates])])
                labels_zero[sample_idx] = target
                counts[target] += 1
                counts[int(cluster)] -= 1
        else:
            groups = _labels_to_groups(labels_zero, n_clusters_i)
            mix = float(singleton_aux_mix)
            mix = 0.0 if not np.isfinite(mix) else max(0.0, min(1.0, mix))
            n_nn = int(X_arr.shape[1])
            if n_nn < 2:
                raise RuntimeError("Cannot augment singleton clusters with fewer than two training samples.")

            X_nn = np.asarray(X_arr.T, dtype=np.float32 if X_arr.dtype == np.float64 else X_arr.dtype)
            k_nn = 5 if n_nn >= 5 else n_nn
            nn = NearestNeighbors(n_neighbors=k_nn, metric="euclidean", algorithm="auto")
            nn.fit(X_nn)
            nn_idx = np.asarray(cast(Any, nn.kneighbors(X_nn, return_distance=False)), dtype=np.int64)
            row_ids = np.arange(n_nn, dtype=np.int64)
            nearest = nn_idx[:, 1].copy()
            self_mask = nearest == row_ids
            if np.any(self_mask).item():
                for kth in range(2, int(nn_idx.shape[1])):
                    candidate_col = nn_idx[:, kth]
                    replace = np.logical_and(self_mask, candidate_col != row_ids)
                    nearest[replace] = candidate_col[replace]
                    self_mask = nearest == row_ids
                    if not np.any(self_mask).item():
                        break
            if np.any(self_mask).item():
                for bad_idx in np.where(self_mask)[0]:
                    nearest[int(bad_idx)] = int((int(bad_idx) + 1) % n_nn)

            X_new_cols: list[np.ndarray] = []
            Y_new_cols: list[np.ndarray] = []
            new_labels: list[int] = []
            for cluster in singletons:
                members = groups[int(cluster)]
                if members.size != 1:
                    continue
                sample_idx = int(members[0])
                neighbor_idx = int(nearest[sample_idx])
                if neighbor_idx == sample_idx:
                    neighbor_idx = int((sample_idx + 1) % int(X_arr.shape[1]))

                # Raw convex interpolation only: no norm preservation or L2 normalization.
                x_aux = (1.0 - mix) * X_arr[:, sample_idx] + mix * X_arr[:, neighbor_idx]
                X_new_cols.append(x_aux.reshape(-1, 1).astype(work_dtype, copy=False))
                Y_new_cols.append(Y_arr[:, sample_idx].reshape(-1, 1).astype(work_dtype, copy=False))
                new_labels.append(int(cluster))

            if X_new_cols:
                X_arr = np.concatenate([X_arr, *X_new_cols], axis=1)
                Y_arr = np.concatenate([Y_arr, *Y_new_cols], axis=1)
                labels_zero = np.concatenate([labels_zero, np.asarray(new_labels, dtype=np.int64)])
                counts = np.bincount(labels_zero, minlength=n_clusters_i)

            remaining_singletons = np.where(counts == 1)[0]
            if remaining_singletons.size:
                raise RuntimeError(
                    f"Augmentation failed to eliminate {int(remaining_singletons.size)} singleton cluster(s). "
                    "Try fewer clusters or singleton_strategy='merge'."
                )

    # 1b) Drop empty clusters and remap labels to 0..K_eff-1.
    used_clusters = np.unique(labels_zero)
    n_clusters_eff = int(used_clusters.size)
    map_arr = np.full(n_clusters_i, -1, dtype=np.int32)
    map_arr[used_clusters.astype(np.int64)] = np.arange(n_clusters_eff, dtype=np.int32)
    labels_mapped = map_arr[labels_zero.astype(np.int64)]
    if np.any(labels_mapped < 0):
        raise RuntimeError("Internal error while remapping cluster labels.")
    labels_mapped = labels_mapped.astype(np.int64, copy=False)

    if verbose:
        print(f"Effective clusters after preprocessing: {n_clusters_eff}")

    # 1c) Compute centers. For the n_clusters == original N case, original targets define centers.
    if skip_kmeans and n_clusters_eff == int(n_samples):
        cluster_centers = Y_arr[:, : int(n_samples)].copy()
    else:
        cluster_centers = _compute_cluster_centers(Y_arr, labels_mapped, n_clusters_eff, work_dtype)

    final_counts = np.bincount(labels_mapped, minlength=n_clusters_eff)
    if verbose:
        print(f"Minimum cluster size after preprocessing: {int(final_counts.min())}")

    # 2) Optional per-cluster subsampling for classifier training.
    labels_for_clf = labels_mapped
    X_clf = X_arr
    if max_train_per_cluster is not None:
        max_per = int(max_train_per_cluster)
        if max_per <= 0:
            raise ValueError("max_train_per_cluster must be a positive integer or None.")
        rng = np.random.default_rng(random_state)
        keep_parts: list[np.ndarray] = []
        for indices in _labels_to_groups(labels_mapped, n_clusters_eff):
            if indices.size <= max_per:
                keep_parts.append(indices)
            else:
                keep_parts.append(rng.choice(indices, size=max_per, replace=False).astype(np.int64, copy=False))
        keep_idx = np.concatenate(keep_parts).astype(np.int64, copy=False)
        keep_idx.sort()
        X_clf = X_arr[:, keep_idx]
        labels_for_clf = labels_mapped[keep_idx]
        if verbose:
            print(f"Autoencoder training samples after subsampling: {int(X_clf.shape[1])} (was {int(X_arr.shape[1])})")

    # Free target matrix before AE training.
    del Y_arr
    gc.collect()

    X_clf = X_clf.astype(np.float32, copy=False)
    groups_for_clf = _labels_to_groups(labels_for_clf, n_clusters_eff)
    if any(indices.size < 2 for indices in groups_for_clf):
        bad = [idx + 1 for idx, indices in enumerate(groups_for_clf) if indices.size < 2][:10]
        raise ValueError(f"At least one cluster has fewer than 2 samples after preprocessing/subsampling: {bad}")

    if verbose:
        print("Training one autoencoder set per output cluster...")

    ae_refs: list[AutoencoderRef] = []
    classifier_dir: str | None = None
    model_id_for_model: str | None = None

    if save_ae_to_disk:
        run_id = str(model_id) if model_id is not None else uuid4().hex[:10]
        model_id_for_model = run_id
        if ae_dir is None:
            root = Path(ae_cache_root)
            root.mkdir(parents=True, exist_ok=True)
            ae_dir_resolved = (root / f"kahm_{run_id}").resolve()
        else:
            ae_dir_resolved = Path(ae_dir).resolve()

        if ae_dir_resolved.exists() and any(ae_dir_resolved.rglob("*.joblib")) and not overwrite_ae_dir:
            raise FileExistsError(
                f"AE directory already contains joblib files: {ae_dir_resolved}. "
                "Use overwrite_ae_dir=True or choose a different ae_dir."
            )
        ae_dir_resolved.mkdir(parents=True, exist_ok=True)
        classifier_dir = str(ae_dir_resolved)
        if verbose:
            print(f"Saving AE shards under: {ae_dir_resolved}")

        for cluster_idx, indices in enumerate(groups_for_clf):
            X_cluster = X_clf[:, indices]
            if verbose:
                print(f" Cluster {cluster_idx + 1}/{n_clusters_eff}: {int(X_cluster.shape[1])} samples")
            ae_list = parallel_autoencoders(
                X_cluster,
                subspace_dim=subspace_dim_i,
                Nb=nb_i,
                n_jobs=1,
                verbose=False,
            )
            ae_list = _downcast_obj(ae_list, work_dtype)
            shard_dir = ae_dir_resolved / f"{cluster_idx // 1000:03d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            ae_path = shard_dir / f"ae_cluster_{cluster_idx + 1:05d}.joblib"
            dump(ae_list, ae_path, compress=int(ae_compress))
            ae_refs.append(os.path.relpath(str(ae_path), str(ae_dir_resolved)))
            del X_cluster, ae_list
    else:
        for cluster_idx, indices in enumerate(groups_for_clf):
            X_cluster = X_clf[:, indices]
            if verbose:
                print(f" Cluster {cluster_idx + 1}/{n_clusters_eff}: {int(X_cluster.shape[1])} samples")
            ae_list = parallel_autoencoders(
                X_cluster,
                subspace_dim=subspace_dim_i,
                Nb=nb_i,
                n_jobs=1,
                verbose=False,
            )
            ae_refs.append(_downcast_obj(ae_list, work_dtype))
            del X_cluster, ae_list

    if verbose:
        print("Training finished.")

    return {
        "classifier": ae_refs,
        "classifier_dir": classifier_dir,
        "model_id": model_id_for_model,
        "cluster_centers_init": cluster_centers.copy(),
        "cluster_centers": cluster_centers,
        "n_clusters": int(n_clusters_eff),
        "cluster_strategy": cluster_strategy,
        "n_y_clusters": int(n_y_clusters_i) if n_y_clusters_i is not None else None,
        "max_x_splits_per_y_cluster": int(max_x_splits_i),
        "min_child_size": int(min_child_size_i),
        "standardize_x_for_cluster_split": bool(standardize_x_for_cluster_split),
        "soft_alpha": None,
        "soft_topk": 10,
        "scaling_enabled": False,
        "l2_normalization_enabled": False,
    }


# -----------------------------------------------------------------------------
# Soft probability mapping
# -----------------------------------------------------------------------------


def _topk_truncate_inplace(scores: np.ndarray, k: int, *, chunk_cols: int = 64) -> None:
    """Keep only top-k entries per column in-place."""
    if scores.ndim != 2:
        raise ValueError("scores must be 2D shaped (C, N).")
    c_count, n_cols = scores.shape
    k_i = int(k)
    if k_i <= 0 or k_i >= c_count or n_cols == 0:
        return

    max_index_bytes = 32 * 1024 * 1024
    max_chunk = max(1, int(max_index_bytes // (8 * max(1, c_count))))
    col_chunk = max(1, min(int(chunk_cols), max_chunk))

    for start in range(0, n_cols, col_chunk):
        end = min(n_cols, start + col_chunk)
        sub = scores[:, start:end]
        idx = np.argpartition(sub, -k_i, axis=0)[-k_i:, :]
        vals = np.take_along_axis(sub, idx, axis=0).copy()
        sub.fill(scores.dtype.type(0.0))
        np.put_along_axis(sub, idx, vals, axis=0)


def distances_to_probabilities_one_minus_sharp(
    distance_matrix: np.ndarray,
    *,
    alpha: float = 10.0,
    topk: int | None = 10,
    eps: float = 1e-12,
    inplace: bool = False,
) -> np.ndarray:
    """Map distances D to probabilities P with S=(1-D)^alpha and optional top-k truncation."""
    distances = _as_float_ndarray(distance_matrix)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must be 2D shaped (C, N).")
    _validate_positive_finite_float("alpha", float(alpha))
    _validate_optional_topk("topk", topk)

    c_count, _ = distances.shape
    dtype = distances.dtype
    one = dtype.type(1.0)
    zero = dtype.type(0.0)
    eps_t = dtype.type(eps)

    if inplace:
        scores = distances if distances.flags.writeable else distances.copy()
        np.subtract(one, scores, out=scores)
    else:
        scores = np.subtract(one, distances)
    np.clip(scores, zero, one, out=scores)

    alpha_f = float(alpha)
    if alpha_f != 1.0:
        np.power(scores, dtype.type(alpha_f), out=scores)

    if topk is not None:
        k_i = int(topk)
        if 0 < k_i < c_count:
            _topk_truncate_inplace(scores, k_i)

    denom = scores.sum(axis=0, dtype=dtype)
    zero_cols = denom <= eps_t
    if np.any(zero_cols):
        denom = denom.copy()
        denom[zero_cols] = one
    np.divide(scores, denom, out=scores)
    if np.any(zero_cols):
        scores[:, zero_cols] = dtype.type(1.0 / c_count)
    return scores


def _ensure_distance_matrix_shape(D: np.ndarray, c_eff: int, n_samples: int, labels: np.ndarray | None = None) -> np.ndarray:
    """Ensure distances are shaped (C_eff, N)."""
    if D.ndim != 2:
        raise ValueError(f"distance_matrix must be 2D; got shape {D.shape}.")
    if D.shape == (c_eff, n_samples) and c_eff != n_samples:
        return D
    if D.shape == (n_samples, c_eff) and c_eff != n_samples:
        return D.T
    if D.shape != (c_eff, n_samples) and D.shape != (n_samples, c_eff):
        raise ValueError(f"distance_matrix shape mismatch. Expected {(c_eff, n_samples)} or {(n_samples, c_eff)}, got {D.shape}.")
    if c_eff != n_samples or labels is None:
        return D if D.shape == (c_eff, n_samples) else D.T

    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    if lab.size != n_samples:
        raise ValueError(f"labels length mismatch: labels={lab.size}, N={n_samples}")
    sample_size = min(512, n_samples)
    idx = np.linspace(0, n_samples - 1, sample_size, dtype=np.int64) if sample_size < n_samples else np.arange(n_samples)
    pred_cn = np.argmin(D[:, idx], axis=0).astype(np.int64)
    pred_nc = np.argmin(D[idx, :], axis=1).astype(np.int64)
    acc_cn = float(np.mean(pred_cn == lab[idx]))
    acc_nc = float(np.mean(pred_nc == lab[idx]))
    return D if acc_cn >= acc_nc else D.T


def _get_soft_params_from_model(model: KahmModel, alpha: float | None, topk: int | None) -> tuple[float, int | None]:
    alpha_resolved = float(alpha if alpha is not None else (model.get("soft_alpha") or 10.0))
    _validate_positive_finite_float("alpha", alpha_resolved)

    if topk is not None:
        return alpha_resolved, _validate_optional_topk("topk", topk)

    if "soft_topk" in model:
        model_topk = cast(int | None, model.get("soft_topk"))
        return alpha_resolved, _validate_optional_topk("soft_topk", model_topk)
    return alpha_resolved, 10


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def preload_kahm_classifier(model: KahmModel, *, n_jobs: int = 1, prefer: Literal["threads", "processes"] = "threads") -> None:
    """Load all disk-backed autoencoder shards into model['_classifier_cache']."""
    ae_entries = _resolve_classifier_entries(model)
    if not ae_entries:
        raise TypeError("No classifier entries found to preload.")
    if not _is_pathlike(ae_entries[0]):
        model["_classifier_cache"] = list(ae_entries)
        return

    paths = [cast(str, _resolve_ae_path(entry, model.get("classifier_dir"))) for entry in ae_entries]
    loaded = Parallel(n_jobs=int(n_jobs), prefer=prefer)(delayed(load)(path) for path in paths)
    model["_classifier_cache"] = list(loaded)


def _update_topk_inplace(
    best_d: np.ndarray,
    best_i: np.ndarray,
    worst_pos: np.ndarray,
    worst_val: np.ndarray,
    distances: np.ndarray,
    cluster_idx: int,
) -> None:
    mask = distances < worst_val
    if not np.any(mask):
        return
    cols = np.nonzero(mask)[0]
    pos = worst_pos[cols]
    best_d[pos, cols] = distances[cols]
    best_i[pos, cols] = int(cluster_idx)
    sub = best_d[:, cols]
    new_worst_pos = np.argmax(sub, axis=0)
    worst_pos[cols] = new_worst_pos
    worst_val[cols] = sub[new_worst_pos, np.arange(cols.size)]


def _soft_predict_from_topk(
    idx: np.ndarray,
    dist: np.ndarray,
    *,
    centers: np.ndarray,
    alpha: float,
    out_dtype: np.dtype[Any],
) -> np.ndarray:
    k_count, batch_cols = dist.shape
    weights = 1.0 - dist
    np.clip(weights, 0.0, 1.0, out=weights)
    if alpha != 1.0:
        np.power(weights, float(alpha), out=weights)

    denom = weights.sum(axis=0)
    zero_cols = denom <= 1e-12
    denom = np.where(zero_cols, 1.0, denom)
    weights /= denom

    y_hat = np.zeros((centers.shape[0], batch_cols), dtype=out_dtype)
    idx_safe = idx.copy()
    idx_safe[idx_safe < 0] = 0
    for row in range(k_count):
        y_hat += centers[:, idx_safe[row, :]] * weights[row, :][None, :]
    if np.any(zero_cols):
        y_hat[:, zero_cols] = centers.mean(axis=1, keepdims=True).astype(out_dtype, copy=False)
    return y_hat


@overload
def kahm_regress(
    model: KahmModel,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: Literal["hard", "soft"] = "hard",
    return_probabilities: Literal[False] = False,
    alpha: float | None = None,
    topk: int | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> np.ndarray: ...


@overload
def kahm_regress(
    model: KahmModel,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: Literal["hard", "soft"] = "hard",
    return_probabilities: Literal[True],
    alpha: float | None = None,
    topk: int | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]: ...


def kahm_regress(
    model: KahmModel,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: Literal["hard", "soft"] = "hard",
    return_probabilities: bool = False,
    alpha: float | None = None,
    topk: int | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Predict outputs for new inputs using a trained KAHM regressor."""
    X_eval = _as_float_ndarray(X_new)
    if X_eval.ndim != 2:
        raise ValueError("X_new must be 2D shaped (D_in, N_new).")

    # No input scaling. The model is evaluated on raw X_new values.
    ae_entries = _resolve_classifier_entries(model)
    cluster_centers = _as_float_ndarray(model["cluster_centers"])
    if cluster_centers.ndim != 2:
        raise ValueError("model['cluster_centers'] must be 2D.")

    c_eff = int(cluster_centers.shape[1])
    n_new = int(X_eval.shape[1])
    if len(ae_entries) != c_eff:
        raise ValueError(f"Mismatch: got {len(ae_entries)} autoencoders but cluster_centers has {c_eff} clusters.")

    out_dtype = np.dtype(np.result_type(cluster_centers.dtype, np.float32))
    distance_type = "folding"
    d_in = int(X_eval.shape[0])

    if mode == "hard":
        best_dist = np.full((n_new,), np.inf, dtype=np.float64)
        best_idx = np.zeros((n_new,), dtype=np.int64)
        bs = int(batch_size) if batch_size is not None and int(batch_size) > 0 else None
        pbar = _maybe_tqdm_total(c_eff * n_new, "KAHM hard: distance eval", "sample", show_progress)
        try:
            for c_idx, ae_ref in enumerate(ae_entries):
                ae_obj, from_disk = _load_ae_maybe(model, ae_ref, cluster_idx=c_idx, d_in=d_in)
                try:
                    if bs is None:
                        d = _call_combine_multiple_autoencoders_extended(X_eval, _ae_as_list(ae_obj), distance_type, n_jobs=n_jobs)
                        d_vec = np.asarray(d, dtype=np.float64).reshape(-1)
                        if d_vec.size != n_new:
                            raise ValueError(f"Distance vector for cluster {c_idx + 1} has size {d_vec.size}; expected {n_new}.")
                        mask = d_vec < best_dist
                        best_dist[mask] = d_vec[mask]
                        best_idx[mask] = c_idx
                        if pbar is not None:
                            pbar.update(n_new)
                    else:
                        for start in range(0, n_new, bs):
                            end = min(start + bs, n_new)
                            d = _call_combine_multiple_autoencoders_extended(
                                X_eval[:, start:end], _ae_as_list(ae_obj), distance_type, n_jobs=n_jobs
                            )
                            d_vec = np.asarray(d, dtype=np.float64).reshape(-1)
                            if d_vec.size != end - start:
                                raise ValueError(f"Distance vector for cluster {c_idx + 1} has size {d_vec.size}; expected {end - start}.")
                            cols = np.arange(start, end)
                            mask = d_vec < best_dist[cols]
                            best_dist[cols[mask]] = d_vec[mask]
                            best_idx[cols[mask]] = c_idx
                            if pbar is not None:
                                pbar.update(end - start)
                finally:
                    if from_disk:
                        del ae_obj
                        gc.collect()
        finally:
            if pbar is not None:
                pbar.close()

        y_pred = cluster_centers[:, best_idx]
        if return_probabilities:
            probs = np.zeros((c_eff, n_new), dtype=out_dtype)
            probs[best_idx, np.arange(n_new)] = 1.0
            return y_pred, probs
        return y_pred

    if mode != "soft":
        raise ValueError("mode must be either 'hard' or 'soft'.")

    alpha_resolved, topk_resolved = _get_soft_params_from_model(model, alpha, topk)

    # Fast top-k path: avoids materializing full C x N probabilities when probabilities are not requested.
    if not return_probabilities and topk_resolved is not None:
        k_req = int(topk_resolved)
        if 0 < k_req < c_eff:
            bs = int(batch_size) if batch_size is not None and int(batch_size) > 0 else n_new
            bs = max(1, bs)
            slices = [(start, min(start + bs, n_new)) for start in range(0, n_new, bs)]
            k_eff = min(k_req, c_eff)
            best_d_list = [np.full((k_eff, end - start), np.inf, dtype=np.float64) for start, end in slices]
            best_i_list = [np.full((k_eff, end - start), -1, dtype=np.int64) for start, end in slices]
            worst_pos_list = [np.zeros((end - start,), dtype=np.int64) for start, end in slices]
            worst_val_list = [np.full((end - start,), np.inf, dtype=np.float64) for start, end in slices]

            pbar = _maybe_tqdm_total(c_eff * n_new, "KAHM soft: distance eval", "sample", show_progress)
            try:
                for c_idx, ae_ref in enumerate(ae_entries):
                    ae_obj, from_disk = _load_ae_maybe(model, ae_ref, cluster_idx=c_idx, d_in=d_in)
                    try:
                        for b_idx, (start, end) in enumerate(slices):
                            d = _call_combine_multiple_autoencoders_extended(
                                X_eval[:, start:end], _ae_as_list(ae_obj), distance_type, n_jobs=n_jobs
                            )
                            d_vec = np.asarray(d, dtype=np.float64).reshape(-1)
                            if d_vec.size != end - start:
                                raise ValueError(f"Distance vector for cluster {c_idx + 1} has size {d_vec.size}; expected {end - start}.")
                            _update_topk_inplace(
                                best_d_list[b_idx],
                                best_i_list[b_idx],
                                worst_pos_list[b_idx],
                                worst_val_list[b_idx],
                                d_vec,
                                c_idx,
                            )
                            if pbar is not None:
                                pbar.update(end - start)
                    finally:
                        if from_disk:
                            del ae_obj
                            gc.collect()
            finally:
                if pbar is not None:
                    pbar.close()

            y_pred = np.empty((cluster_centers.shape[0], n_new), dtype=out_dtype)
            pbar2 = _maybe_tqdm_total(n_new, "KAHM soft: assemble", "sample", show_progress)
            try:
                for b_idx, (start, end) in enumerate(slices):
                    order = np.argsort(best_d_list[b_idx], axis=0)
                    dist_sorted = np.take_along_axis(best_d_list[b_idx], order, axis=0)
                    idx_sorted = np.take_along_axis(best_i_list[b_idx], order, axis=0)
                    y_pred[:, start:end] = _soft_predict_from_topk(
                        idx_sorted,
                        dist_sorted,
                        centers=cluster_centers,
                        alpha=float(alpha_resolved),
                        out_dtype=out_dtype,
                    )
                    if pbar2 is not None:
                        pbar2.update(end - start)
            finally:
                if pbar2 is not None:
                    pbar2.close()
            return y_pred

    # Dense batched path without probability return: C x N distances stored on disk if batch_size is provided.
    if batch_size is not None and int(batch_size) > 0 and not return_probabilities:
        bs = int(batch_size)
        tmp = tempfile.NamedTemporaryFile(prefix="kahm_D_", suffix=".dat", delete=False)
        tmp_path = tmp.name
        tmp.close()
        D_mm: np.memmap | None = None
        pbar = _maybe_tqdm_total(c_eff * n_new, "KAHM soft(dense): distance eval", "sample", show_progress)
        try:
            D_mm = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=(c_eff, n_new))
            for c_idx, ae_ref in enumerate(ae_entries):
                ae_obj, from_disk = _load_ae_maybe(model, ae_ref, cluster_idx=c_idx, d_in=d_in)
                try:
                    for start in range(0, n_new, bs):
                        end = min(start + bs, n_new)
                        d = _call_combine_multiple_autoencoders_extended(
                            X_eval[:, start:end], _ae_as_list(ae_obj), distance_type, n_jobs=n_jobs
                        )
                        d_vec = np.asarray(d, dtype=np.float32).reshape(-1)
                        if d_vec.size != end - start:
                            raise ValueError(f"Distance vector for cluster {c_idx + 1} has size {d_vec.size}; expected {end - start}.")
                        D_mm[c_idx, start:end] = d_vec
                        if pbar is not None:
                            pbar.update(end - start)
                finally:
                    if from_disk:
                        del ae_obj
                        gc.collect()
            D_mm.flush()

            y_pred = np.empty((cluster_centers.shape[0], n_new), dtype=out_dtype)
            pbar2 = _maybe_tqdm_total(n_new, "KAHM soft(dense): predict", "sample", show_progress)
            try:
                for start in range(0, n_new, bs):
                    end = min(start + bs, n_new)
                    D_batch = _ensure_distance_matrix_shape(np.asarray(D_mm[:, start:end]), c_eff, end - start)
                    P_batch = distances_to_probabilities_one_minus_sharp(
                        D_batch,
                        alpha=float(alpha_resolved),
                        topk=topk_resolved,
                        inplace=False,
                    )
                    y_pred[:, start:end] = cluster_centers @ P_batch
                    if pbar2 is not None:
                        pbar2.update(end - start)
            finally:
                if pbar2 is not None:
                    pbar2.close()
            return y_pred
        finally:
            if pbar is not None:
                pbar.close()
            if D_mm is not None:
                del D_mm
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Dense path, required when probabilities are requested.
    D = np.empty((c_eff, n_new), dtype=np.float64)
    pbar = _maybe_tqdm_total(c_eff * n_new, "KAHM soft(dense): distance eval", "sample", show_progress)
    try:
        for c_idx, ae_ref in enumerate(ae_entries):
            ae_obj, from_disk = _load_ae_maybe(model, ae_ref, cluster_idx=c_idx, d_in=d_in)
            try:
                d = _call_combine_multiple_autoencoders_extended(X_eval, _ae_as_list(ae_obj), distance_type, n_jobs=n_jobs)
                d_vec = np.asarray(d, dtype=np.float64).reshape(-1)
                if d_vec.size != n_new:
                    raise ValueError(f"Distance vector for cluster {c_idx + 1} has size {d_vec.size}; expected {n_new}.")
                D[c_idx, :] = d_vec
                if pbar is not None:
                    pbar.update(n_new)
            finally:
                if from_disk:
                    del ae_obj
                    gc.collect()
    finally:
        if pbar is not None:
            pbar.close()

    D = _ensure_distance_matrix_shape(D, c_eff, n_new)
    P = distances_to_probabilities_one_minus_sharp(D, alpha=float(alpha_resolved), topk=topk_resolved, inplace=True)
    y_pred = cluster_centers @ P
    if return_probabilities:
        return y_pred, P
    return y_pred


# -----------------------------------------------------------------------------
# Tuning utilities
# -----------------------------------------------------------------------------


def tune_soft_params(
    model: KahmModel,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    alphas: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0),
    topks: Sequence[int | None] | None = None,
    n_jobs: int = -1,
    nlms_mu: float = 0.1,
    nlms_epsilon: float | None = 1.0,
    nlms_batch_size: int = 1024,
    nlms_shuffle: bool = True,
    nlms_random_state: int | None = 0,
    nlms_anchor_lambda: float = 0.0,
    verbose: bool = True,
) -> SoftTuningResult:
    """Tune soft alpha with topk fixed to n_clusters and one-epoch NLMS prototypes.

    ``topks`` is accepted only for backward compatibility. The effective soft
    top-k is always fixed to the model's current effective cluster count
    (``model['n_clusters']`` / ``cluster_centers.shape[1]``), as requested.

    For each candidate alpha, the function:
    1. builds soft probabilities from the validation distance matrix with
       ``topk == n_clusters``;
    2. performs one epoch of mini-batch NLMS updates on a candidate copy of the
       cluster prototypes/centers;
    3. computes validation MSE *after* that NLMS epoch.

    The model is updated with the best alpha, the fixed top-k, and the best
    one-epoch NLMS-tuned cluster centers.
    """
    Xv = _as_float_ndarray(X_val)
    Yv = _as_float_ndarray(Y_val)
    if Xv.ndim != 2 or Yv.ndim != 2:
        raise ValueError("X_val and Y_val must be 2D arrays shaped (D, N).")
    if Xv.shape[1] != Yv.shape[1]:
        raise ValueError("X_val and Y_val must have the same number of samples.")

    ae_entries = _resolve_classifier_entries(model)
    cluster_centers = _as_float_ndarray(model["cluster_centers"])
    if cluster_centers.ndim != 2:
        raise ValueError("model['cluster_centers'] must be 2D.")
    if cluster_centers.shape[0] != Yv.shape[0]:
        raise ValueError(
            f"Output dimension mismatch: centers have {cluster_centers.shape[0]} rows but Y_val has {Yv.shape[0]} rows."
        )

    c_eff = int(cluster_centers.shape[1])
    n_val = int(Xv.shape[1])
    fixed_topk = c_eff
    if len(ae_entries) != c_eff:
        raise ValueError(f"Mismatch: got {len(ae_entries)} autoencoders but cluster_centers has {c_eff} clusters.")

    alphas_t = tuple(_validate_positive_finite_float("alpha", float(a)) for a in alphas)
    if not alphas_t:
        raise ValueError("alphas must contain at least one value.")
    # ``topks`` intentionally ignored, but validate any supplied values so that
    # invalid legacy calls still fail early instead of being silently accepted.
    if topks is not None:
        tuple(_validate_optional_topk("topk", t) for t in topks)

    mu_f = _validate_positive_finite_float("nlms_mu", float(nlms_mu))
    eps_f = 1.0 if nlms_epsilon is None else _validate_positive_finite_float("nlms_epsilon", float(nlms_epsilon))
    nlms_bs_i = _validate_positive_int("nlms_batch_size", int(nlms_batch_size))
    if not np.isfinite(float(nlms_anchor_lambda)) or float(nlms_anchor_lambda) < 0.0:
        raise ValueError("nlms_anchor_lambda must be a finite non-negative number.")

    original_dtype = cluster_centers.dtype
    initial_centers = np.asarray(cluster_centers, dtype=np.float64).copy()
    if model.get("cluster_centers_init") is None:
        model["cluster_centers_init"] = initial_centers.astype(original_dtype, copy=True)
    anchor = None
    if float(nlms_anchor_lambda) > 0.0:
        anchor = _as_float_ndarray(model["cluster_centers_init"]).astype(np.float64, copy=False)
        if anchor.shape != initial_centers.shape:
            raise ValueError(
                f"cluster_centers_init shape {anchor.shape} does not match cluster_centers shape {initial_centers.shape}."
            )

    D_bytes = c_eff * n_val * 8
    use_memmap = D_bytes > int(model.get("tune_memmap_threshold_bytes", 512 * 1024 * 1024))
    D_mm: np.memmap | None = None
    tmp_path: str | None = None

    try:
        if use_memmap:
            tmp = tempfile.NamedTemporaryFile(prefix="kahm_Dval_", suffix=".dat", delete=False)
            tmp_path = tmp.name
            tmp.close()
            D_store: np.ndarray = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=(c_eff, n_val))
            D_mm = cast(np.memmap, D_store)
            if verbose:
                print(f"Using on-disk validation distance matrix: {tmp_path}")
        else:
            D_store = np.empty((c_eff, n_val), dtype=np.float64)

        for c_idx, ae_ref in enumerate(ae_entries):
            ae_obj, from_disk = _load_ae_maybe(model, ae_ref, cluster_idx=c_idx, d_in=int(Xv.shape[0]))
            try:
                d = _call_combine_multiple_autoencoders_extended(Xv, _ae_as_list(ae_obj), "folding", n_jobs=n_jobs)
                d_vec = np.asarray(d).reshape(-1)
                if d_vec.size != n_val:
                    raise ValueError(f"Distance vector for cluster {c_idx + 1} has size {d_vec.size}; expected {n_val}.")
                D_store[c_idx, :] = np.asarray(d_vec, dtype=D_store.dtype)
            finally:
                if from_disk:
                    del ae_obj
                    gc.collect()

        if isinstance(D_store, np.memmap):
            D_store.flush()
        D_val = _ensure_distance_matrix_shape(_as_float_ndarray(D_store), c_eff, n_val)

        if verbose:
            print("Tuning soft alpha with fixed topk and one-epoch NLMS prototype tuning...")
            print(f"Grid: alphas={list(alphas_t)}, fixed topk={fixed_topk}")
            print(f"NLMS: mu={mu_f:g}, epsilon={eps_f:g}, epochs=1, batch_size={nlms_bs_i}")

        best_mse = float("inf")
        best_alpha = float(alphas_t[0])
        best_topk: int | None = fixed_topk
        best_centers = initial_centers.copy()

        work_max_bytes = int(model.get("tune_eval_work_max_bytes", 256 * 1024 * 1024))
        bs_cfg = int(model.get("tune_eval_batch_cols", 4096))
        bs_mem = max(1, int(work_max_bytes // (max(1, c_eff) * 8)))
        eval_bs = max(1, min(n_val, bs_cfg, bs_mem))
        nlms_bs = max(1, min(n_val, nlms_bs_i, bs_mem))
        work_width = max(eval_bs, nlms_bs)

        work = np.empty((c_eff, work_width), dtype=np.float64)
        denom = np.empty((work_width,), dtype=np.float64)

        def _fill_weights(cols: slice | np.ndarray, width: int, alpha_f: float) -> np.ndarray:
            weights = work[:, :width]
            np.subtract(1.0, D_val[:, cols], out=weights)
            np.clip(weights, 0.0, 1.0, out=weights)
            if alpha_f != 1.0:
                np.power(weights, alpha_f, out=weights)
            # fixed_topk == c_eff, so this is intentionally a no-op in the
            # current setting. The branch is retained for safety if c_eff ever
            # changes before this local helper is called.
            if 0 < fixed_topk < c_eff:
                _topk_truncate_inplace(weights, fixed_topk)
            denom_b = denom[:width]
            denom_b[:] = weights.sum(axis=0, dtype=np.float64)
            zero_cols = denom_b <= 1e-12
            if np.any(zero_cols):
                denom_b[zero_cols] = 1.0
            np.divide(weights, denom_b, out=weights)
            if np.any(zero_cols):
                weights[:, zero_cols] = 1.0 / c_eff
            return weights

        nlms_order = np.arange(n_val, dtype=np.int64)
        if nlms_shuffle:
            rng = np.random.default_rng(nlms_random_state)
            rng.shuffle(nlms_order)

        for alpha_f in alphas_t:
            centers_work = initial_centers.copy()
            order = nlms_order

            # One NLMS epoch over validation samples using probabilities induced
            # by the candidate alpha and fixed topk=n_clusters.
            for start in range(0, n_val, nlms_bs):
                end = min(start + nlms_bs, n_val)
                sel = order[start:end]
                width = int(sel.size)
                probs = _fill_weights(sel, width, float(alpha_f)).copy()
                y_batch = Yv[:, sel]
                y_hat = centers_work @ probs
                error = (y_batch - y_hat).astype(np.float64, copy=False)
                p_norm2 = np.sum(probs * probs, axis=0)
                step = mu_f / (eps_f + mu_f * p_norm2)
                centers_work += (error * step[None, :]) @ probs.T
                if anchor is not None:
                    centers_work -= (mu_f * float(nlms_anchor_lambda)) * (centers_work - anchor)

            # Validation MSE after the single NLMS epoch.
            sse = 0.0
            count = 0
            for start in range(0, n_val, eval_bs):
                end = min(start + eval_bs, n_val)
                width = end - start
                probs = _fill_weights(slice(start, end), width, float(alpha_f))
                y_hat = centers_work @ probs
                diff = y_hat - Yv[:, start:end]
                sse += float(np.sum(diff * diff))
                count += int(diff.size)

            mse = sse / max(1, count)
            if verbose:
                print(f"  alpha={alpha_f:g}, fixed topk={fixed_topk}: post-NLMS MSE={mse:.6g}")
            if mse < best_mse:
                best_mse = float(mse)
                best_alpha = float(alpha_f)
                best_centers = centers_work.copy()

        model["soft_alpha"] = best_alpha
        model["soft_topk"] = best_topk
        model["cluster_centers"] = best_centers.astype(original_dtype, copy=False)
        if verbose:
            print(f"Best soft params: alpha={best_alpha:g}, topk={best_topk}, post-NLMS val MSE={best_mse:.6g}")
        return SoftTuningResult(best_alpha=best_alpha, best_topk=best_topk, best_mse=best_mse)
    finally:
        if D_mm is not None:
            del D_mm
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def tune_cluster_centers_nlms(
    model: KahmModel,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    mu: float = 0.1,
    epsilon: float | None = None,
    epochs: int = 1,
    batch_size: int = 1024,
    shuffle: bool = True,
    random_state: int | None = 0,
    alpha: float | None = None,
    topk: int | None = None,
    anchor_lambda: float = 0.0,
    n_jobs: int = -1,
    preload_classifier: bool = False,
    verbose: bool = True,
) -> NLMSCenterTuningResult:
    """Refine cluster centers with mini-batch Normalized LMS."""
    X_train = _as_float_ndarray(X)
    Y_train = _as_float_ndarray(Y)
    if X_train.ndim != 2 or Y_train.ndim != 2:
        raise ValueError("X and Y must be 2D arrays shaped (D, N).")
    if X_train.shape[1] != Y_train.shape[1]:
        raise ValueError("X and Y must have the same number of samples.")
    if int(epochs) <= 0:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be >= 1.")

    mu_f = _validate_positive_finite_float("mu", float(mu))
    eps_f = 1.0 if epsilon is None else _validate_positive_finite_float("epsilon", float(epsilon))
    _validate_alpha_topk(alpha, topk)
    if not np.isfinite(float(anchor_lambda)) or float(anchor_lambda) < 0.0:
        raise ValueError("anchor_lambda must be a finite non-negative number.")

    centers = _as_float_ndarray(model.get("cluster_centers"))
    if centers.ndim != 2:
        raise ValueError("model['cluster_centers'] must be 2D.")
    if centers.shape[0] != Y_train.shape[0]:
        raise ValueError(
            f"Output dimension mismatch: centers have {centers.shape[0]} rows but Y has {Y_train.shape[0]} rows."
        )

    if model.get("cluster_centers_init") is None:
        model["cluster_centers_init"] = np.asarray(centers, dtype=np.float64).copy()

    anchor = None
    if float(anchor_lambda) > 0.0:
        anchor = _as_float_ndarray(model["cluster_centers_init"]).astype(np.float64, copy=False)

    original_centers_obj = model.get("cluster_centers")
    original_dtype = centers.dtype
    centers_work = np.asarray(centers, dtype=np.float64).copy()
    model["cluster_centers"] = centers_work

    try:
        if preload_classifier:
            preload_jobs = max(1, int(abs(n_jobs) if n_jobs != 0 else 1))
            try:
                preload_kahm_classifier(model, n_jobs=preload_jobs)
            except Exception:
                if verbose:
                    print("Classifier preload failed; continuing with on-demand loading.")

        n_samples = int(X_train.shape[1])
        rng = np.random.default_rng(random_state)
        history: list[float] = []

        for epoch in range(int(epochs)):
            order = np.arange(n_samples)
            if shuffle:
                rng.shuffle(order)
            sse = 0.0
            count = 0

            for start in range(0, n_samples, int(batch_size)):
                end = min(start + int(batch_size), n_samples)
                sel = order[start:end]
                X_batch = X_train[:, sel]
                Y_batch = Y_train[:, sel]

                y_hat, probs = kahm_regress(
                    model,
                    X_batch,
                    n_jobs=int(n_jobs),
                    mode="soft",
                    return_probabilities=True,
                    alpha=alpha,
                    topk=topk,
                    batch_size=None,
                    show_progress=False,
                )
                if probs.shape[1] != end - start:
                    raise RuntimeError(f"Unexpected probability matrix shape: {probs.shape}")
                if y_hat.shape != Y_batch.shape:
                    raise RuntimeError(f"Unexpected prediction shape: {y_hat.shape}; expected {Y_batch.shape}")

                error = (Y_batch - y_hat).astype(np.float64, copy=False)
                probs64 = np.asarray(probs, dtype=np.float64)
                p_norm2 = np.sum(probs64 * probs64, axis=0)
                step = mu_f / (eps_f + mu_f * p_norm2)
                centers_work += (error * step[None, :]) @ probs64.T

                if anchor is not None:
                    centers_work -= (mu_f * float(anchor_lambda)) * (centers_work - anchor)

                sse += float(np.sum(error * error))
                count += int(error.size)
                if verbose:
                    print(f"[NLMS] epoch {epoch + 1}/{int(epochs)} | samples {end}/{n_samples}")

            mse = sse / max(1, count)
            history.append(float(mse))
            if verbose:
                print(f"[NLMS] epoch {epoch + 1}/{int(epochs)} | MSE={mse:.6g} | mu={mu_f:g} | eps={eps_f:g}")

    except Exception:
        model["cluster_centers"] = original_centers_obj
        raise

    model["cluster_centers"] = centers_work.astype(original_dtype, copy=False) if centers_work.dtype != original_dtype else centers_work
    return NLMSCenterTuningResult(
        mu=mu_f,
        epsilon=eps_f,
        epochs=int(epochs),
        batch_size=int(batch_size),
        final_mse=float(history[-1] if history else np.nan),
        mse_history=tuple(history),
    )


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def save_kahm_regressor(model: KahmModel, path: str) -> None:
    """Save a KAHM regressor.

    Runtime-only keys starting with ``_`` are omitted. Disk-backed autoencoder
    shards are not embedded in the saved model; keep the saved ``.joblib`` model
    file together with its ``classifier_dir`` shard directory.
    """
    _validate_classifier_assets_available(model, context="save")
    model_to_save = {key: value for key, value in model.items() if not str(key).startswith("_")}
    clf_dir = model_to_save.get("classifier_dir")
    disk_backed = isinstance(clf_dir, (str, os.PathLike, Path))
    if disk_backed:
        try:
            model_dir = Path(path).resolve().parent
            clf_dir_path = Path(cast(Pathish, clf_dir))
            if clf_dir_path.is_absolute():
                model_to_save["classifier_dir"] = os.path.relpath(str(clf_dir_path), str(model_dir))
        except Exception:
            pass
    dump(model_to_save, path)
    print(f"KAHM regressor saved to {path}")
    if disk_backed:
        print("Note: AE shards are stored separately; keep classifier_dir with the saved model file.")


def load_kahm_regressor(path: str, *, base_dir: str | None = None) -> KahmModel:
    """Load a saved KAHM regressor and resolve relative classifier_dir paths.

    For disk-backed models, the classifier shard directory must still be present
    relative to the model file, or supplied explicitly via ``base_dir``.
    """
    model = cast(KahmModel, load(path))
    if base_dir is not None:
        model["classifier_dir"] = str(base_dir)
    else:
        clf_dir = model.get("classifier_dir")
        if isinstance(clf_dir, (str, os.PathLike, Path)):
            clf_path = Path(cast(Pathish, clf_dir))
            if not clf_path.is_absolute():
                model["classifier_dir"] = str((Path(path).resolve().parent / clf_path).resolve())
    _validate_classifier_assets_available(model, context="load")
    print(f"KAHM regressor loaded from {path}")
    return model


# -----------------------------------------------------------------------------
# Minimal smoke-test example
# -----------------------------------------------------------------------------




if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d_in, d_out, n = 5, 3, 5000

    X_demo = np.tanh(rng.normal(size=(d_in, n))).astype(np.float32)
    Y_demo = np.vstack(
        [
            2.0 * X_demo[0, :] + 0.5 * X_demo[1, :] ** 2 + 0.05 * rng.normal(size=n),
            -X_demo[2, :] + np.sin(X_demo[3, :]) + 0.05 * rng.normal(size=n),
            1.5 * X_demo[4, :] + 0.05 * rng.normal(size=n),
        ]
    ).astype(np.float32)

    # Keep the demo small but separate train/validation/test splits so that
    # soft-parameter tuning has an honest validation set.
    n_train = int(0.85 * n)
    train_slice = slice(0, n_train)
    test_slice = slice(n_train, n)

    X_train_demo, Y_train_demo = X_demo[:, train_slice], Y_demo[:, train_slice]
    X_test_demo, Y_test_demo = X_demo[:, test_slice], Y_demo[:, test_slice]

    model_demo = train_kahm_regressor(
        X_train_demo,
        Y_train_demo,
        n_clusters=int(0.1 * n_train),
        subspace_dim=20,
        Nb=100,
        random_state=0,
        save_ae_to_disk=False,
        verbose=True,
        cluster_strategy="y_then_x",
        max_x_splits_per_y_cluster=10
    )

    # Baseline: default soft parameters from kahm_regress().
    pred_hard = kahm_regress(
        model_demo,
        X_test_demo,
        mode="hard",
        batch_size=1024,
        show_progress=True,
    )
    
    def _r2_overall(yhat: np.ndarray, ytrue: np.ndarray) -> float:
        residual_ss = float(np.sum((yhat - ytrue) ** 2))
        total_ss = float(np.sum((ytrue - ytrue.mean(axis=1, keepdims=True)) ** 2))
        return 1.0 - residual_ss / total_ss

    mse_hard = float(np.mean((pred_hard - Y_test_demo) ** 2))
    r2_hard = _r2_overall(pred_hard, Y_test_demo);

    # 1) Tune soft alpha on validation data. topk is fixed internally to
    # n_clusters, and the reported MSE is computed after one NLMS epoch on the
    # cluster prototypes.
    soft_result = tune_soft_params(
        model_demo,
        X_train_demo,
        Y_train_demo,
        alphas=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0),
        n_jobs=-1,
        verbose=True,
    )
    print(
        "Best soft + 1-epoch NLMS tuning: "
        f"alpha={soft_result.best_alpha:g}, "
        f"topk={soft_result.best_topk}, "
        f"post-NLMS val MSE={soft_result.best_mse:.6g}"
    )

    pred_tuned_soft = kahm_regress(
        model_demo,
        X_test_demo,
        mode="soft",
        batch_size=1024,
        show_progress=True,
    )
    mse_tuned_soft = float(np.mean((pred_tuned_soft - Y_test_demo) ** 2))
    r2_tuned_soft = _r2_overall(pred_tuned_soft, Y_test_demo);

    # 2) Refine cluster prototypes/centers with mini-batch NLMS.
    # Passing alpha=None/topk=None means the function uses the tuned values stored
    # in model_demo['soft_alpha'] and model_demo['soft_topk'].

    nlms_result = tune_cluster_centers_nlms(
        model_demo,
        X_train_demo,
        Y_train_demo, 
        mu=0.1,
        epsilon=1.0,
        epochs=20,
        batch_size=1024,
        shuffle=True,
        random_state=0,
        alpha=None,
        topk=None,
        anchor_lambda=0.0,
        n_jobs=-1,
        preload_classifier=True,
        verbose=True,
    )
    print(
        "NLMS prototype tuning: "
        f"final train MSE={nlms_result.final_mse:.6g}"
    )

    pred_after_nlms = kahm_regress(
        model_demo,
        X_test_demo,
        mode="soft",
        batch_size=1024,
        show_progress=True,
    )
    mse_after_nlms = float(np.mean((pred_after_nlms - Y_test_demo) ** 2))
    r2_after_nlms = _r2_overall(pred_after_nlms, Y_test_demo);

    print(f"Default hard MSE: {mse_hard:.6g}")
    print(f"Default hard R2: {r2_hard:.6g}")
    print(f"Tuned soft + 1-epoch NLMS MSE (alpha={soft_result.best_alpha}, topk={soft_result.best_topk}): {mse_tuned_soft:.6g}")
    print(f"Tuned soft + 1-epoch NLMS R2: {r2_tuned_soft:.6g}")
    print(f"Tuned soft + NLMS MSE: {mse_after_nlms:.6g}")
    print(f"Tuned soft + NLMS R2: {r2_after_nlms:.6g}")

