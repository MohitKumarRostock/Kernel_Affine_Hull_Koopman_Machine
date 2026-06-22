"""
Kernel Affine Hull Koopman Machines using the original KAHM/OTFL folding pipeline.

This standalone version keeps the KAHKM state-regime abstraction code local.
It keeps the KAHKM-specific state-regime abstraction, association normalization,
folding-distance evaluation, and NLMS Koopman closure in this file.

It still depends on the original local KAHM/OTFL helper modules that perform the
autoencoder training and folding-score evaluation:

- ``parallel_autoencoders.py``
- ``combine_multiple_autoencoders_extended.py``

Important design choices
------------------------
1. Regimes are learned directly in current-state space ``X0`` by KMeans or
   MiniBatchKMeans over state columns.
2. One original KAHM/OTFL autoencoder set is trained per state-regime cluster.
3. Only the manuscript normalization is implemented:

       psi_c(x) = (1 - T_c(x) + tau)^omega
                  / sum_l (1 - T_l(x) + tau)^omega.

4. Hierarchical ``y_then_x`` clustering and the regression script's alternative
   soft mapper are intentionally not used, because this file implements KAHKM
   regime abstraction rather than the earlier output-regression workflow.
5. Omega is treated as an abstraction sharpness parameter and should be selected
   by external validation/sensitivity analysis.

Data layout
-----------
X0: (D, N) current states as columns
X1: (D, N) next states as columns

Minimal dependencies
--------------------
pip install numpy scikit-learn joblib

Run the toy example
-------------------
python kernel_affine_hull_koopman_machines.py
"""

from __future__ import annotations

import gc
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast
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
        "are available on PYTHONPATH or in the same directory as this script. "
        "The KAHKM state-regime abstraction code is implemented locally in this file."
    ) from exc

Pathish = str | os.PathLike[str] | Path
KahmModel = dict[str, Any]
AutoencoderRef = Any


# -----------------------------------------------------------------------------
# Local KAHM/OTFL training helpers
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


def _validate_positive_int(name: str, value: int) -> int:
    value_i = int(value)
    if value_i <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value_i


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

    Older local helper versions did not accept ``n_jobs``. Signature inspection
    avoids swallowing unrelated ``TypeError`` exceptions raised inside the helper.
    """
    fn = cast(Any, combine_multiple_autoencoders_extended)
    if n_jobs is None:
        return fn(X, ae_list, distance_type)

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
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


def _compute_cluster_centers(X: np.ndarray, labels_zero: np.ndarray, n_clusters: int, dtype: np.dtype[Any]) -> np.ndarray:
    """Compute state centroids shaped (D, C)."""
    labels = np.asarray(labels_zero, dtype=np.int64).reshape(-1)
    if labels.size != X.shape[1]:
        raise ValueError(f"labels length {labels.size} does not match X sample count {X.shape[1]}.")

    counts = np.bincount(labels, minlength=int(n_clusters)).astype(np.float64, copy=False)
    if np.any(counts <= 0):
        empty = np.where(counts <= 0)[0][:10].tolist()
        raise ValueError(f"Cannot compute centers for empty clusters. First empty labels: {empty}")

    centers_t = np.zeros((int(n_clusters), int(X.shape[0])), dtype=dtype)
    np.add.at(centers_t, labels, X.T.astype(dtype, copy=False))
    centers_t = centers_t / counts[:, None].astype(dtype, copy=False)
    return centers_t.T.astype(dtype, copy=False)


def _make_kmeans_estimator(
    n_clusters: int,
    *,
    random_state: int | None,
    kmeans_kind: Literal["auto", "full", "minibatch"],
    kmeans_batch_size: int,
) -> KMeans | MiniBatchKMeans:
    """Create the KMeans variant used for state-regime clustering."""
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


def _maybe_tqdm_total(total: int, desc: str, unit: str, enabled: bool) -> Any:
    if not enabled:
        return None
    try:
        from tqdm import tqdm  # type: ignore[import]

        return tqdm(total=int(total), desc=desc, unit=unit, leave=False)
    except Exception:
        return None


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


def train_kahm_state_regime_abstraction(
    X: np.ndarray,
    *,
    n_clusters: int,
    subspace_dim: int = 20,
    Nb: int = 100,
    random_state: int | None = 0,
    verbose: bool = True,
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
) -> KahmModel:
    """Train KAHM/OTFL state-regime structures without importing the regression script.

    This is the local KAHKM-only state-regime training path: samples are
    clustered directly in state space, then one KAHM/OTFL autoencoder set is
    trained per resulting regime cluster.
    """
    X_arr = _as_float_ndarray(X)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array shaped (D, N).")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X must contain only finite values.")

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

    d_in, n_samples = X_arr.shape
    if n_clusters_i > n_samples:
        raise ValueError(f"n_clusters={n_clusters_i} cannot exceed number of samples N={int(n_samples)}.")

    work_dtype = _resolve_work_dtype(model_dtype, X_arr)
    X_arr = X_arr.astype(work_dtype, copy=False)

    if verbose:
        print(f"Training KAHM state-regime abstraction on {int(n_samples)} samples.")
        print(f"State dim: {int(d_in)}")
        print(f"Requested regimes: {n_clusters_i}")
        print("Regime clustering: state-space KMeans/MiniBatchKMeans")
        print("Input scaling: disabled")
        print("L2 normalization: disabled")

    skip_kmeans = n_clusters_i == int(n_samples)
    kmeans: KMeans | MiniBatchKMeans | None
    if skip_kmeans:
        if verbose:
            print("Skipping KMeans because n_clusters equals the number of samples.")
        kmeans = None
        labels_zero = np.arange(int(n_samples), dtype=np.int64)
    else:
        if verbose:
            use_minibatch = kmeans_kind == "minibatch" or (kmeans_kind == "auto" and n_clusters_i >= 2000)
            print(
                f"Running {'MiniBatchKMeans' if use_minibatch else 'full KMeans'} on current states X0"
                + (f" with batch_size={kmeans_batch_size_i}" if use_minibatch else "")
                + "..."
            )
        kmeans, labels_zero = _fit_kmeans_labels(
            X_arr.T,
            n_clusters_i,
            random_state=random_state,
            kmeans_kind=kmeans_kind,
            kmeans_batch_size=kmeans_batch_size_i,
        )

    counts = np.bincount(labels_zero, minlength=n_clusters_i)
    singletons = np.where(counts == 1)[0]
    if singletons.size and verbose:
        print(f"Handling {int(singletons.size)} singleton regime cluster(s) using strategy={singleton_strategy!r}...")

    if singletons.size:
        if singleton_strategy == "merge":
            if kmeans is None and skip_kmeans:
                raise ValueError("singleton_strategy='merge' cannot be used when each sample is its own cluster.")
            if kmeans is not None:
                centers = np.asarray(kmeans.cluster_centers_, dtype=work_dtype)
            else:
                centers = np.asarray(_compute_cluster_centers(X_arr, labels_zero, n_clusters_i, work_dtype).T, dtype=work_dtype)
            centers_sq = np.einsum("ij,ij->i", centers, centers)
            for cluster in singletons:
                sample_indices = np.where(labels_zero == int(cluster))[0]
                if sample_indices.size != 1:
                    continue
                sample_idx = int(sample_indices[0])
                x_sample = X_arr[:, sample_idx]
                d2 = centers_sq + float(np.dot(x_sample, x_sample)) - 2.0 * centers.dot(x_sample)
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
                new_labels.append(int(cluster))

            if X_new_cols:
                X_arr = np.concatenate([X_arr, *X_new_cols], axis=1)
                labels_zero = np.concatenate([labels_zero, np.asarray(new_labels, dtype=np.int64)])
                counts = np.bincount(labels_zero, minlength=n_clusters_i)

            remaining_singletons = np.where(counts == 1)[0]
            if remaining_singletons.size:
                raise RuntimeError(
                    f"Augmentation failed to eliminate {int(remaining_singletons.size)} singleton cluster(s). "
                    "Try fewer clusters or singleton_strategy='merge'."
                )

    used_clusters = np.unique(labels_zero)
    n_clusters_eff = int(used_clusters.size)
    map_arr = np.full(n_clusters_i, -1, dtype=np.int32)
    map_arr[used_clusters.astype(np.int64)] = np.arange(n_clusters_eff, dtype=np.int32)
    labels_mapped = map_arr[labels_zero.astype(np.int64)]
    if np.any(labels_mapped < 0):
        raise RuntimeError("Internal error while remapping cluster labels.")
    labels_mapped = labels_mapped.astype(np.int64, copy=False)

    if verbose:
        print(f"Effective regimes after preprocessing: {n_clusters_eff}")

    if skip_kmeans and n_clusters_eff == int(n_samples):
        cluster_centers = X_arr[:, : int(n_samples)].copy()
    else:
        cluster_centers = _compute_cluster_centers(X_arr, labels_mapped, n_clusters_eff, work_dtype)

    final_counts = np.bincount(labels_mapped, minlength=n_clusters_eff)
    if verbose:
        print(f"Minimum regime cluster size after preprocessing: {int(final_counts.min())}")

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

    X_clf = X_clf.astype(np.float32, copy=False)
    groups_for_clf = _labels_to_groups(labels_for_clf, n_clusters_eff)
    if any(indices.size < 2 for indices in groups_for_clf):
        bad = [idx + 1 for idx, indices in enumerate(groups_for_clf) if indices.size < 2][:10]
        raise ValueError(f"At least one regime has fewer than 2 samples after preprocessing/subsampling: {bad}")

    if verbose:
        print("Training one autoencoder set per state-regime cluster...")

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
                print(f" Regime {cluster_idx + 1}/{n_clusters_eff}: {int(X_cluster.shape[1])} samples")
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
                print(f" Regime {cluster_idx + 1}/{n_clusters_eff}: {int(X_cluster.shape[1])} samples")
            ae_list = parallel_autoencoders(
                X_cluster,
                subspace_dim=subspace_dim_i,
                Nb=nb_i,
                n_jobs=1,
                verbose=False,
            )
            ae_refs.append(_downcast_obj(ae_list, work_dtype))
            del X_cluster, ae_list

    gc.collect()
    if verbose:
        print("State-regime abstraction training finished.")

    return {
        "classifier": ae_refs,
        "classifier_dir": classifier_dir,
        "model_id": model_id_for_model,
        "cluster_centers_init": cluster_centers.copy(),
        "cluster_centers": cluster_centers,
        "n_clusters": int(n_clusters_eff),
        "cluster_strategy": "kmeans_state",
        "soft_alpha": None,
        "soft_topk": 10,
        "scaling_enabled": False,
        "l2_normalization_enabled": False,
    }


@dataclass(frozen=True)
class KAHKMFitResult:
    """Training result for a Kernel Affine Hull Koopman Machine."""

    abstraction_model: KahmModel
    B: np.ndarray
    B_stochastic: np.ndarray | None
    train_closure_error: float
    association_r2: float
    nlms_history: tuple[float, ...]
    omega: float
    tau: float


@dataclass(frozen=True)
class KAHKMEvaluationResult:
    """Held-out closure diagnostics."""

    closure_error: float
    association_r2: float
    simplex_violation_raw: float
    simplex_violation_stochastic: float | None


# -----------------------------------------------------------------------------
# Validation and diagnostics
# -----------------------------------------------------------------------------


def _validate_snapshot_pair(X0: Any, X1: Any) -> tuple[np.ndarray, np.ndarray]:
    X0_arr = _as_float_ndarray(X0)
    X1_arr = _as_float_ndarray(X1)
    if X0_arr.ndim != 2 or X1_arr.ndim != 2:
        raise ValueError("X0 and X1 must be 2D arrays shaped (D, N).")
    if X0_arr.shape != X1_arr.shape:
        raise ValueError(f"X0 and X1 must have the same shape. Got {X0_arr.shape=} and {X1_arr.shape=}.")
    if not np.all(np.isfinite(X0_arr)) or not np.all(np.isfinite(X1_arr)):
        raise ValueError("X0 and X1 must contain only finite values.")
    return X0_arr, X1_arr


def _validate_positive(name: str, value: float) -> float:
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value_f


def _relative_closure_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """Relative squared closure residual: ||target - pred||_F^2 / ||target||_F^2."""
    num = float(np.sum((target - pred) ** 2))
    den = float(np.sum(target * target))
    return num / max(den, float(eps))


def _association_r2(pred: np.ndarray, target: np.ndarray) -> float:
    residual_ss = float(np.sum((pred - target) ** 2))
    total_ss = float(np.sum((target - target.mean(axis=1, keepdims=True)) ** 2))
    if total_ss <= 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def simplex_violation(P: np.ndarray) -> float:
    """Return a scalar violation of nonnegativity and unit column sums."""
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2:
        raise ValueError("P must be 2D shaped (C, N).")
    neg = np.maximum(-P_arr, 0.0)
    sum_err = np.abs(P_arr.sum(axis=0) - 1.0)
    return float(max(np.max(neg) if neg.size else 0.0, np.max(sum_err) if sum_err.size else 0.0))


# -----------------------------------------------------------------------------
# Original KAHM/OTFL folding-distance evaluation path
# -----------------------------------------------------------------------------


def kahm_folding_distance_matrix(
    model: KahmModel,
    X: np.ndarray,
    *,
    n_jobs: int = -1,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Evaluate the dense C x N KAHM folding-score/distance matrix T.

    The actual folding evaluation is delegated to the original local KAHM helper
    through ``combine_multiple_autoencoders_extended(..., distance_type="folding")``.
    This function returns the per-regime scores T_c(x) used by the paper's
    association formula.
    """
    X_eval = _as_float_ndarray(X)
    if X_eval.ndim != 2:
        raise ValueError("X must be 2D shaped (D, N).")

    ae_entries = _resolve_classifier_entries(model)
    cluster_centers = _as_float_ndarray(model["cluster_centers"])
    c_eff = int(cluster_centers.shape[1])
    n_new = int(X_eval.shape[1])
    d_in = int(X_eval.shape[0])
    if len(ae_entries) != c_eff:
        raise ValueError(f"Mismatch: got {len(ae_entries)} autoencoders but model has {c_eff} clusters.")

    T = np.empty((c_eff, n_new), dtype=np.float64)
    pbar = _maybe_tqdm_total(c_eff * n_new, "KAHKM: folding eval", "sample", show_progress)
    try:
        for c_idx, ae_ref in enumerate(ae_entries):
            ae_obj, from_disk = _load_ae_maybe(model, ae_ref, cluster_idx=c_idx, d_in=d_in)
            try:
                if batch_size is None or int(batch_size) <= 0:
                    d = _call_combine_multiple_autoencoders_extended(
                        X_eval,
                        _ae_as_list(ae_obj),
                        "folding",
                        n_jobs=n_jobs,
                    )
                    d_vec = np.asarray(d, dtype=np.float64).reshape(-1)
                    if d_vec.size != n_new:
                        raise ValueError(
                            f"Distance vector for cluster {c_idx + 1} has size {d_vec.size}; expected {n_new}."
                        )
                    T[c_idx, :] = d_vec
                    if pbar is not None:
                        pbar.update(n_new)
                else:
                    bs = int(batch_size)
                    for start in range(0, n_new, bs):
                        end = min(start + bs, n_new)
                        d = _call_combine_multiple_autoencoders_extended(
                            X_eval[:, start:end],
                            _ae_as_list(ae_obj),
                            "folding",
                            n_jobs=n_jobs,
                        )
                        d_vec = np.asarray(d, dtype=np.float64).reshape(-1)
                        if d_vec.size != end - start:
                            raise ValueError(
                                f"Distance vector for cluster {c_idx + 1} has size {d_vec.size}; "
                                f"expected {end - start}."
                            )
                        T[c_idx, start:end] = d_vec
                        if pbar is not None:
                            pbar.update(end - start)
            finally:
                if from_disk:
                    del ae_obj
                    gc.collect()
    finally:
        if pbar is not None:
            pbar.close()

    return T


def kahm_associations(
    model: KahmModel,
    X: np.ndarray,
    *,
    omega: float,
    tau: float,
    n_jobs: int = -1,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Map states to paper-normalized KAHM associations Psi(X).

    Given folding scores T_c(x), this implements

        s_c(x)   = 1 - T_c(x),
        psi_c(x) = (s_c(x) + tau)^omega / sum_l (s_l(x) + tau)^omega.

    The returned array has shape (C, N), with one simplex-valued association
    vector per column.
    """
    omega_f = _validate_positive("omega", omega)
    tau_f = _validate_positive("tau", tau)

    T = kahm_folding_distance_matrix(
        model,
        X,
        n_jobs=n_jobs,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    # The paper assumes T_c(x) in [0, 1]. The original helper should return a
    # folding score in this range; clipping makes the implementation robust to
    # small numerical overshoots without changing the intended formula.
    np.clip(T, 0.0, 1.0, out=T)
    scores = 1.0 - T
    scores += tau_f
    if omega_f != 1.0:
        np.power(scores, omega_f, out=scores)

    denom = scores.sum(axis=0, keepdims=True)
    zero_cols = denom.reshape(-1) <= 1e-12
    if np.any(zero_cols):
        denom[:, zero_cols] = 1.0
    Psi = scores / denom
    if np.any(zero_cols):
        Psi[:, zero_cols] = 1.0 / Psi.shape[0]
    return Psi.astype(np.float64, copy=False)


# -----------------------------------------------------------------------------
# Koopman closure by NLMS
# -----------------------------------------------------------------------------


def nlms_koopman_closure(
    Phi: np.ndarray,
    Chi: np.ndarray,
    *,
    beta: float = 0.5,
    epochs: int = 1,
    shuffle: bool = False,
    random_state: int | None = 0,
    B0: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Learn B from Chi ≈ B.T @ Phi using the manuscript KAHKM NLMS recursion.

    Phi and Chi must be shaped (C, N), where columns are association vectors.
    For sample i, with phi = Phi[:, i] and chi = Chi[:, i], this implements
    Definition 6.2 / Eq. (29) from the manuscript exactly:

        B <- B + beta * phi (chi - B.T phi)^T / (1 + beta * ||phi||_2^2).

    The manuscript assumes 0 < beta < 1. The default ``epochs=1`` corresponds
    to the single pass i = 1, ..., M in the manuscript. Larger values simply
    replay the same recursion over the data stream multiple times.
    """
    Phi_arr = np.asarray(Phi, dtype=np.float64)
    Chi_arr = np.asarray(Chi, dtype=np.float64)
    if Phi_arr.ndim != 2 or Chi_arr.ndim != 2:
        raise ValueError("Phi and Chi must be 2D shaped (C, N).")
    if Phi_arr.shape != Chi_arr.shape:
        raise ValueError(f"Phi and Chi must have the same shape. Got {Phi_arr.shape=} and {Chi_arr.shape=}.")
    if not np.all(np.isfinite(Phi_arr)) or not np.all(np.isfinite(Chi_arr)):
        raise ValueError("Phi and Chi must contain only finite values.")

    beta_f = float(beta)
    if not np.isfinite(beta_f) or not (0.0 < beta_f < 1.0):
        raise ValueError("The manuscript NLMS recursion assumes beta satisfies 0 < beta < 1.")

    epochs_i = int(epochs)
    if epochs_i <= 0:
        raise ValueError("epochs must be >= 1.")

    c_count, n_samples = Phi_arr.shape
    if B0 is None:
        B = np.eye(c_count, dtype=np.float64)
    else:
        B = np.asarray(B0, dtype=np.float64).copy()
        if B.shape != (c_count, c_count):
            raise ValueError(f"B0 must have shape {(c_count, c_count)}; got {B.shape}.")
        if not np.all(np.isfinite(B)):
            raise ValueError("B0 must contain only finite values.")

    rng = np.random.default_rng(random_state)
    base_order = np.arange(n_samples, dtype=np.int64)
    history: list[float] = []

    for _ in range(epochs_i):
        order = base_order.copy()
        if shuffle:
            rng.shuffle(order)
        for idx in order:
            phi = Phi_arr[:, int(idx)]
            chi = Chi_arr[:, int(idx)]
            err = chi - B.T @ phi
            denom = 1.0 + beta_f * float(phi @ phi)
            B += (beta_f / denom) * np.outer(phi, err)
        pred = B.T @ Phi_arr
        history.append(_relative_closure_error(pred, Chi_arr))

    return B, tuple(history)


# -----------------------------------------------------------------------------
# Optional simplex projection for regime-transition interpretation
# -----------------------------------------------------------------------------


def project_vector_to_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Euclidean projection of a vector onto {x >= 0, sum x = z}."""
    v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if z <= 0.0:
        raise ValueError("z must be positive.")
    n = v_arr.size
    u = np.sort(v_arr)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(v_arr, z / n)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(v_arr - theta, 0.0)


def project_rows_to_simplex(B: np.ndarray) -> np.ndarray:
    """Project each row of B to the probability simplex."""
    B_arr = np.asarray(B, dtype=np.float64)
    if B_arr.ndim != 2:
        raise ValueError("B must be 2D.")
    return np.vstack([project_vector_to_simplex(row) for row in B_arr])


# -----------------------------------------------------------------------------
# Main KAHKM fit / predict / evaluate functions
# -----------------------------------------------------------------------------


def fit_kahkm(
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    n_clusters: int,
    subspace_dim: int = 20,
    Nb: int = 100,
    omega: float = 10.0,
    tau: float = 1e-6,
    beta: float = 0.5,
    nlms_epochs: int = 1,
    random_state: int | None = 0,
    kmeans_kind: Literal["auto", "full", "minibatch"] = "auto",
    kmeans_batch_size: int = 4096,
    singleton_strategy: Literal["augment", "merge"] = "augment",
    singleton_aux_mix: float = 0.05,
    max_train_per_cluster: int | None = None,
    save_ae_to_disk: bool = False,
    ae_dir: str | None = None,
    overwrite_ae_dir: bool = False,
    n_jobs: int = -1,
    batch_size: int | None = None,
    project_stochastic: bool = True,
    preload_classifier_after_fit: bool = False,
    verbose: bool = True,
) -> KAHKMFitResult:
    """Fit a Kernel Affine Hull Koopman Machine using original KAHM folding.

    The KAHM/OTFL abstraction model is trained locally by clustering current
    states and training one original KAHM/OTFL autoencoder set per state-regime
    cluster. This file does not import the earlier regression helper.

    After the abstraction is fixed, the Koopman closure is learned from
    Phi = Psi(X0), Chi = Psi(X1).
    """
    X0_arr, X1_arr = _validate_snapshot_pair(X0, X1)
    _validate_positive("omega", omega)
    _validate_positive("tau", tau)

    if verbose:
        print("Training KAHM/OTFL state-regime abstraction with the original pipeline...")
        print("  regime clustering: KMeans/MiniBatchKMeans on current states X0")
        print("  association normalization: paper Eq. (7)")

    abstraction_model = train_kahm_state_regime_abstraction(
        X0_arr,
        n_clusters=int(n_clusters),
        subspace_dim=int(subspace_dim),
        Nb=int(Nb),
        random_state=random_state,
        verbose=verbose,
        kmeans_kind=kmeans_kind,
        kmeans_batch_size=kmeans_batch_size,
        max_train_per_cluster=max_train_per_cluster,
        save_ae_to_disk=save_ae_to_disk,
        ae_dir=ae_dir,
        overwrite_ae_dir=overwrite_ae_dir,
        singleton_strategy=singleton_strategy,
        singleton_aux_mix=singleton_aux_mix,
    )
    abstraction_model["kahkm_cluster_on"] = "state"
    abstraction_model["kahkm_normalization"] = "paper_eq_7"
    abstraction_model["kahkm_omega"] = float(omega)
    abstraction_model["kahkm_tau"] = float(tau)

    if preload_classifier_after_fit:
        preload_kahm_classifier(abstraction_model, n_jobs=max(1, int(n_jobs)))

    if verbose:
        print("Computing KAHM associations for current and next states...")
    Phi = kahm_associations(
        abstraction_model,
        X0_arr,
        omega=omega,
        tau=tau,
        n_jobs=n_jobs,
        batch_size=batch_size,
        show_progress=verbose,
    )
    Chi = kahm_associations(
        abstraction_model,
        X1_arr,
        omega=omega,
        tau=tau,
        n_jobs=n_jobs,
        batch_size=batch_size,
        show_progress=verbose,
    )

    if verbose:
        print("Learning Koopman closure matrix by NLMS...")
    B, history = nlms_koopman_closure(
        Phi,
        Chi,
        beta=beta,
        epochs=nlms_epochs,
        shuffle=False,
        random_state=random_state,
    )
    pred = B.T @ Phi
    err = _relative_closure_error(pred, Chi)
    r2 = _association_r2(pred, Chi)
    B_stochastic = project_rows_to_simplex(B) if project_stochastic else None

    return KAHKMFitResult(
        abstraction_model=abstraction_model,
        B=B,
        B_stochastic=B_stochastic,
        train_closure_error=err,
        association_r2=r2,
        nlms_history=history,
        omega=float(omega),
        tau=float(tau),
    )


def predict_kahkm_association(
    fit: KAHKMFitResult,
    X: np.ndarray,
    *,
    use_stochastic: bool = False,
    n_jobs: int = -1,
    batch_size: int | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Predict next-step association vectors for new current states."""
    Phi = kahm_associations(
        fit.abstraction_model,
        X,
        omega=fit.omega,
        tau=fit.tau,
        n_jobs=n_jobs,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    if use_stochastic:
        if fit.B_stochastic is None:
            raise ValueError("fit.B_stochastic is None because project_stochastic=False during fitting.")
        return fit.B_stochastic.T @ Phi
    return fit.B.T @ Phi


def evaluate_kahkm(
    fit: KAHKMFitResult,
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    n_jobs: int = -1,
    batch_size: int | None = None,
    show_progress: bool = False,
) -> KAHKMEvaluationResult:
    """Evaluate held-out KAHKM closure diagnostics."""
    X0_arr, X1_arr = _validate_snapshot_pair(X0, X1)
    Phi = kahm_associations(
        fit.abstraction_model,
        X0_arr,
        omega=fit.omega,
        tau=fit.tau,
        n_jobs=n_jobs,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    Chi = kahm_associations(
        fit.abstraction_model,
        X1_arr,
        omega=fit.omega,
        tau=fit.tau,
        n_jobs=n_jobs,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    pred_raw = fit.B.T @ Phi
    raw_violation = simplex_violation(pred_raw)

    stochastic_violation = None
    if fit.B_stochastic is not None:
        pred_stoch = fit.B_stochastic.T @ Phi
        stochastic_violation = simplex_violation(pred_stoch)

    return KAHKMEvaluationResult(
        closure_error=_relative_closure_error(pred_raw, Chi),
        association_r2=_association_r2(pred_raw, Chi),
        simplex_violation_raw=raw_violation,
        simplex_violation_stochastic=stochastic_violation,
    )


def kahm_koopman_kernel(Psi_A: np.ndarray, Psi_B: np.ndarray) -> np.ndarray:
    """Finite-rank KAHM Koopman kernel matrix K_ij = Psi_A[:, i]^T Psi_B[:, j]."""
    A = np.asarray(Psi_A, dtype=np.float64)
    B = np.asarray(Psi_B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Psi_A and Psi_B must be 2D shaped (C, N).")
    if A.shape[0] != B.shape[0]:
        raise ValueError("Psi_A and Psi_B must have the same number of rows/regimes.")
    return A.T @ B


# -----------------------------------------------------------------------------
# Toy nonlinear dynamical-system example
# -----------------------------------------------------------------------------


def make_duffing_snapshots(n_steps: int = 2500, dt: float = 0.03, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate snapshot pairs from an unforced damped Duffing oscillator.

    State z = [q, p]. The continuous-time dynamics are
        q' = p,
        p' = -delta p - alpha q - beta q^3,
    integrated by RK4.
    """
    rng = np.random.default_rng(seed)
    delta, alpha, beta = 0.25, -1.0, 1.0

    def f(z: np.ndarray) -> np.ndarray:
        q, p = float(z[0]), float(z[1])
        return np.array([p, -delta * p - alpha * q - beta * q**3], dtype=np.float64)

    def rk4_step(z: np.ndarray) -> np.ndarray:
        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    z = np.array([0.7, 0.0], dtype=np.float64) + 0.05 * rng.normal(size=2)
    states = np.empty((2, n_steps + 1), dtype=np.float64)
    states[:, 0] = z
    for t in range(n_steps):
        z = rk4_step(z) + 1e-4 * rng.normal(size=2)
        states[:, t + 1] = z
    return states[:, :-1], states[:, 1:]


def _demo() -> None:
    X0, X1 = make_duffing_snapshots(n_steps=1200, dt=0.03, seed=1)
    split = int(0.75 * X0.shape[1])
    X0_train, X1_train = X0[:, :split], X1[:, :split]
    X0_test, X1_test = X0[:, split:], X1[:, split:]

    fit = fit_kahkm(
        X0_train,
        X1_train,
        n_clusters=25,
        subspace_dim=4,
        Nb=100,
        omega=8.0,
        tau=1e-6,
        beta=0.1,
        nlms_epochs=20,
        random_state=0,
        kmeans_kind="auto",
        save_ae_to_disk=False,
        n_jobs=-1,
        batch_size=256,
        project_stochastic=True,
        verbose=True,
    )

    test = evaluate_kahkm(
        fit,
        X0_test,
        X1_test,
        n_jobs=-1,
        batch_size=256,
        show_progress=True,
    )

    print("\n=== KAHKM demo diagnostics ===")
    print(f"Effective regimes: {fit.B.shape[0]}")
    print(f"Train closure error: {fit.train_closure_error:.6g}")
    print(f"Train association R^2: {fit.association_r2:.6g}")
    print(f"NLMS history: {[round(v, 6) for v in fit.nlms_history]}")
    print(f"Test closure error: {test.closure_error:.6g}")
    print(f"Test association R^2: {test.association_r2:.6g}")
    print(f"Raw predicted-simplex violation: {test.simplex_violation_raw:.6g}")
    if test.simplex_violation_stochastic is not None:
        print(f"Projected stochastic violation: {test.simplex_violation_stochastic:.6g}")


if __name__ == "__main__":
    _demo()
