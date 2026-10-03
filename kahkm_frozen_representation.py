"""Portable frozen KAHKM representation artifacts.

A fitted KAHKM abstraction may store one joblib autoencoder shard per regime.
The live model records an absolute ``classifier_dir`` together with classifier
entries that are relative to that directory. Absolute training-machine paths
must not become part of archived certificate evidence.

This module therefore serializes only the stable fitted material:

- autoencoder shard files, copied losslessly into ``autoencoders/``;
- cluster_centers and cluster_centers_init in ``model_arrays.npz``;
- the learned raw NLMS matrix B in ``operator.npy``;
- scalar/list metadata in ``manifest.json``.

On load, ``classifier_dir`` is reconstructed from the artifact's current
location, making the artifact relocatable.

The entire fitted result object is deliberately NOT pickled.

The format is intended for frozen certificate-evaluation representations and
is not a general-purpose persistence API for arbitrary KAHM models.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FORMAT_ID = "kahkm_frozen_representation_v1"

_REQUIRED_MODEL_METADATA = (
    "n_clusters",
    "cluster_strategy",
    "soft_alpha",
    "soft_topk",
    "scaling_enabled",
    "l2_normalization_enabled",
    "kahkm_cluster_on",
    "kahkm_normalization",
    "kahkm_omega",
    "kahkm_tau",
)

_ARRAY_KEYS = (
    "cluster_centers_init",
    "cluster_centers",
)


@dataclass(frozen=True)
class LoadedFrozenRepresentation:
    """Portable frozen representation restored for evaluation."""

    format_id: str
    artifact_dir: str
    system: str
    abstraction_model: dict[str, Any]
    B: NDArray[np.float64]
    omega: float
    tau: float
    train_closure_error: float
    association_r2: float
    nlms_history: tuple[float, ...]
    manifest: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _require_finite_array(
    name: str,
    value: Any,
    *,
    ndim: int,
) -> NDArray[np.float64]:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}D."
        )

    if any(size <= 0 for size in array.shape):
        raise ValueError(
            f"{name} must have nonzero dimensions."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return np.asarray(
        array,
        dtype=np.float64,
    )


def _require_finite_float(
    name: str,
    value: Any,
) -> float:
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a real scalar."
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a real scalar."
        ) from exc

    if not np.isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _strict_json_dump(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_source_shards(
    abstraction_model: Mapping[str, Any],
) -> tuple[Path, tuple[Path, ...], tuple[str, ...]]:
    classifier = abstraction_model.get(
        "classifier"
    )

    if not isinstance(
        classifier,
        (list, tuple),
    ) or len(classifier) == 0:
        raise TypeError(
            "Frozen-representation export requires "
            "a non-empty disk-backed classifier list."
        )

    classifier_dir_raw = abstraction_model.get(
        "classifier_dir"
    )

    if not isinstance(
        classifier_dir_raw,
        (str, os.PathLike, Path),
    ):
        raise TypeError(
            "Frozen-representation export requires "
            "a disk-backed classifier_dir."
        )

    classifier_dir = Path(
        classifier_dir_raw
    ).expanduser().resolve()

    if not classifier_dir.is_dir():
        raise FileNotFoundError(
            f"classifier_dir does not exist: "
            f"{classifier_dir}"
        )

    source_paths: list[Path] = []
    relative_names: list[str] = []
    seen: set[str] = set()

    for index, entry in enumerate(classifier):
        if not isinstance(
            entry,
            (str, os.PathLike, Path),
        ):
            raise TypeError(
                "Frozen-representation export requires "
                "every classifier entry to be a path."
            )

        entry_path = Path(entry)

        if entry_path.is_absolute():
            candidate = entry_path
        else:
            candidate = (
                classifier_dir
                / entry_path
            )

        # Check the path as supplied before resolving it. Path.resolve()
        # follows a leaf symlink, after which is_symlink() would no longer
        # reveal that the configured classifier entry was itself a symlink.
        if candidate.is_symlink():
            raise ValueError(
                "Classifier shards must not be symlinks."
            )

        source = candidate.resolve()

        if entry_path.is_absolute():
            try:
                relative = source.relative_to(
                    classifier_dir
                )
            except ValueError as exc:
                raise ValueError(
                    "Absolute classifier shard lies outside "
                    "classifier_dir."
                ) from exc
        else:
            relative = entry_path

        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative == Path(".")
        ):
            raise ValueError(
                f"Unsafe classifier shard path at "
                f"index {index}: {entry!r}."
            )

        try:
            source.relative_to(
                classifier_dir
            )
        except ValueError as exc:
            raise ValueError(
                "Classifier shard resolves outside "
                "classifier_dir."
            ) from exc

        if not source.is_file():
            raise FileNotFoundError(
                f"Classifier shard not found: "
                f"{source}"
            )

        relative_posix = relative.as_posix()

        if relative_posix in seen:
            raise ValueError(
                "Duplicate classifier shard path: "
                f"{relative_posix}"
            )

        seen.add(relative_posix)
        source_paths.append(source)
        relative_names.append(
            relative_posix
        )

    return (
        classifier_dir,
        tuple(source_paths),
        tuple(relative_names),
    )


def _validate_model_metadata(
    abstraction_model: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        key
        for key in _REQUIRED_MODEL_METADATA
        if key not in abstraction_model
    ]

    if missing:
        raise ValueError(
            "Abstraction model is missing required "
            "metadata: "
            + ", ".join(missing)
        )

    n_clusters = abstraction_model[
        "n_clusters"
    ]

    if (
        isinstance(n_clusters, bool)
        or not isinstance(
            n_clusters,
            (int, np.integer),
        )
        or int(n_clusters) <= 0
    ):
        raise ValueError(
            "n_clusters must be a positive integer."
        )

    metadata = {
        "n_clusters": int(n_clusters),
        "cluster_strategy": str(
            abstraction_model[
                "cluster_strategy"
            ]
        ),
        "soft_alpha":
            None
            if abstraction_model[
                "soft_alpha"
            ] is None
            else _require_finite_float(
                "soft_alpha",
                abstraction_model[
                    "soft_alpha"
                ],
            ),
        "soft_topk": int(
            abstraction_model[
                "soft_topk"
            ]
        ),
        "scaling_enabled": bool(
            abstraction_model[
                "scaling_enabled"
            ]
        ),
        "l2_normalization_enabled": bool(
            abstraction_model[
                "l2_normalization_enabled"
            ]
        ),
        "kahkm_cluster_on": str(
            abstraction_model[
                "kahkm_cluster_on"
            ]
        ),
        "kahkm_normalization": str(
            abstraction_model[
                "kahkm_normalization"
            ]
        ),
        "kahkm_omega": _require_finite_float(
            "kahkm_omega",
            abstraction_model[
                "kahkm_omega"
            ],
        ),
        "kahkm_tau": _require_finite_float(
            "kahkm_tau",
            abstraction_model[
                "kahkm_tau"
            ],
        ),
    }

    if metadata["soft_topk"] <= 0:
        raise ValueError(
            "soft_topk must be positive."
        )

    if metadata["kahkm_omega"] <= 0.0:
        raise ValueError(
            "kahkm_omega must be positive."
        )

    if metadata["kahkm_tau"] <= 0.0:
        raise ValueError(
            "kahkm_tau must be positive."
        )

    return metadata


def save_frozen_representation(
    *,
    output_dir: str | os.PathLike[str],
    system: str,
    abstraction_model: Mapping[str, Any],
    B: Any,
    train_closure_error: Any,
    association_r2: Any,
    nlms_history: Sequence[Any],
    training_metadata: Mapping[str, Any],
) -> Path:
    """Write one portable frozen KAHKM representation artifact.

    ``output_dir`` must not already exist. The function never overwrites an
    existing artifact.
    """
    if not isinstance(system, str) or not system:
        raise ValueError(
            "system must be a non-empty string."
        )

    destination = Path(
        output_dir
    ).expanduser().resolve()

    if destination.exists():
        raise FileExistsError(
            f"Output already exists: {destination}"
        )

    if not isinstance(
        training_metadata,
        Mapping,
    ):
        raise TypeError(
            "training_metadata must be a mapping."
        )

    model_metadata = _validate_model_metadata(
        abstraction_model
    )

    arrays: dict[str, NDArray[np.float64]] = {}

    for key in _ARRAY_KEYS:
        if key not in abstraction_model:
            raise ValueError(
                f"Abstraction model is missing {key!r}."
            )

        arrays[key] = _require_finite_array(
            key,
            abstraction_model[key],
            ndim=2,
        )

    if (
        arrays["cluster_centers"].shape
        != arrays["cluster_centers_init"].shape
    ):
        raise ValueError(
            "cluster_centers and cluster_centers_init "
            "must have the same shape."
        )

    if (
        arrays["cluster_centers"].shape[1]
        != int(model_metadata["n_clusters"])
    ):
        raise ValueError(
            "n_clusters does not match cluster-center columns."
        )

    operator = _require_finite_array(
        "B",
        B,
        ndim=2,
    )

    expected_operator_shape = (
        int(model_metadata["n_clusters"]),
        int(model_metadata["n_clusters"]),
    )

    if operator.shape != expected_operator_shape:
        raise ValueError(
            "B shape does not match n_clusters: "
            f"{operator.shape} != "
            f"{expected_operator_shape}."
        )

    train_error = _require_finite_float(
        "train_closure_error",
        train_closure_error,
    )

    r2 = _require_finite_float(
        "association_r2",
        association_r2,
    )

    history = tuple(
        _require_finite_float(
            f"nlms_history[{index}]",
            value,
        )
        for index, value
        in enumerate(nlms_history)
    )

    (
        _source_classifier_dir,
        source_shards,
        relative_shards,
    ) = _resolve_source_shards(
        abstraction_model
    )

    if (
        len(relative_shards)
        != int(model_metadata["n_clusters"])
    ):
        raise ValueError(
            "Classifier shard count does not match n_clusters."
        )

    destination.mkdir(
        parents=True,
        exist_ok=False,
    )

    shard_root = (
        destination
        / "autoencoders"
    )
    shard_root.mkdir()

    try:
        for source, relative_name in zip(
            source_shards,
            relative_shards,
            strict=True,
        ):
            target = (
                shard_root
                / Path(relative_name)
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copyfile(
                source,
                target,
            )

        arrays_path = (
            destination
            / "model_arrays.npz"
        )

        np.savez(
            arrays_path,
            cluster_centers_init=
                arrays[
                    "cluster_centers_init"
                ],
            cluster_centers=
                arrays[
                    "cluster_centers"
                ],
        )

        operator_path = (
            destination
            / "operator.npy"
        )

        np.save(
            operator_path,
            operator,
            allow_pickle=False,
        )

        files: dict[str, str] = {}

        for path in sorted(
            destination.rglob("*")
        ):
            if (
                path.is_file()
                and path.name
                != "manifest.json"
            ):
                files[
                    path.relative_to(
                        destination
                    ).as_posix()
                ] = _sha256_file(path)

        manifest = {
            "format_id": FORMAT_ID,
            "system": system,
            "model_metadata":
                model_metadata,
            "classifier_entries":
                list(relative_shards),
            "fit": {
                "train_closure_error":
                    train_error,
                "association_r2":
                    r2,
                "nlms_history":
                    list(history),
            },
            "training_metadata":
                dict(training_metadata),
            "files_sha256":
                files,
        }

        _strict_json_dump(
            destination
            / "manifest.json",
            manifest,
        )

    except BaseException:
        # Never leave a directory that looks complete after a failed export.
        shutil.rmtree(
            destination,
            ignore_errors=True,
        )
        raise

    return destination


def load_frozen_representation(
    artifact_dir: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
) -> LoadedFrozenRepresentation:
    """Load and validate a portable frozen representation."""
    artifact = Path(
        artifact_dir
    ).expanduser().resolve()

    if not artifact.is_dir():
        raise FileNotFoundError(
            f"Artifact directory not found: {artifact}"
        )

    manifest_path = (
        artifact
        / "manifest.json"
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if not isinstance(
        manifest,
        dict,
    ):
        raise TypeError(
            "manifest.json must contain a JSON object."
        )

    if manifest.get(
        "format_id"
    ) != FORMAT_ID:
        raise ValueError(
            "Unsupported frozen-representation format."
        )

    system = manifest.get(
        "system"
    )

    if not isinstance(system, str) or not system:
        raise ValueError(
            "Manifest system must be a non-empty string."
        )

    files_sha256 = manifest.get(
        "files_sha256"
    )

    if not isinstance(
        files_sha256,
        dict,
    ) or not files_sha256:
        raise ValueError(
            "Manifest files_sha256 must be a non-empty object."
        )

    expected_paths = set()

    for relative_name, expected_hash in files_sha256.items():
        if (
            not isinstance(
                relative_name,
                str,
            )
            or not isinstance(
                expected_hash,
                str,
            )
        ):
            raise TypeError(
                "Invalid files_sha256 entry."
            )

        relative = Path(
            relative_name
        )

        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative == Path(".")
        ):
            raise ValueError(
                f"Unsafe manifest path: {relative_name!r}."
            )

        path = (
            artifact
            / relative
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Artifact file missing: {relative_name}"
            )

        if path.is_symlink():
            raise ValueError(
                f"Artifact file must not be a symlink: "
                f"{relative_name}"
            )

        expected_paths.add(
            relative.as_posix()
        )

        if verify_hashes:
            actual_hash = _sha256_file(
                path
            )

            if actual_hash != expected_hash:
                raise ValueError(
                    "SHA256 mismatch for "
                    f"{relative_name}: "
                    f"{actual_hash} != "
                    f"{expected_hash}"
                )

    actual_paths = {
        path.relative_to(
            artifact
        ).as_posix()
        for path in artifact.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
    }

    if actual_paths != expected_paths:
        missing_from_manifest = sorted(
            actual_paths
            - expected_paths
        )
        missing_from_disk = sorted(
            expected_paths
            - actual_paths
        )

        raise ValueError(
            "Frozen artifact inventory mismatch; "
            f"unlisted={missing_from_manifest}, "
            f"missing={missing_from_disk}."
        )

    arrays_path = (
        artifact
        / "model_arrays.npz"
    )

    with np.load(
        arrays_path,
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != {
            "cluster_centers_init",
            "cluster_centers",
        }:
            raise ValueError(
                "Unexpected model_arrays.npz members."
            )

        cluster_centers_init = (
            _require_finite_array(
                "cluster_centers_init",
                archive[
                    "cluster_centers_init"
                ],
                ndim=2,
            ).copy()
        )

        cluster_centers = (
            _require_finite_array(
                "cluster_centers",
                archive[
                    "cluster_centers"
                ],
                ndim=2,
            ).copy()
        )

    operator = _require_finite_array(
        "B",
        np.load(
            artifact
            / "operator.npy",
            allow_pickle=False,
        ),
        ndim=2,
    ).copy()

    model_metadata_raw = manifest.get(
        "model_metadata"
    )

    if not isinstance(
        model_metadata_raw,
        dict,
    ):
        raise TypeError(
            "model_metadata must be a JSON object."
        )

    model_metadata = _validate_model_metadata(
        model_metadata_raw
    )

    n_clusters = int(
        model_metadata[
            "n_clusters"
        ]
    )

    if (
        cluster_centers.shape
        != cluster_centers_init.shape
    ):
        raise ValueError(
            "Frozen cluster-center arrays have mismatched shapes."
        )

    if (
        cluster_centers.shape[1]
        != n_clusters
    ):
        raise ValueError(
            "Frozen cluster-center count does not match n_clusters."
        )

    if operator.shape != (
        n_clusters,
        n_clusters,
    ):
        raise ValueError(
            "Frozen operator shape does not match n_clusters."
        )

    classifier_entries_raw = manifest.get(
        "classifier_entries"
    )

    if not isinstance(
        classifier_entries_raw,
        list,
    ) or len(
        classifier_entries_raw
    ) != n_clusters:
        raise ValueError(
            "classifier_entries must contain exactly n_clusters paths."
        )

    classifier_entries: list[str] = []

    for entry in classifier_entries_raw:
        if not isinstance(entry, str):
            raise TypeError(
                "Classifier entries must be strings."
            )

        relative = Path(entry)

        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative == Path(".")
        ):
            raise ValueError(
                f"Unsafe classifier entry: {entry!r}."
            )

        shard = (
            artifact
            / "autoencoders"
            / relative
        )

        if not shard.is_file():
            raise FileNotFoundError(
                f"Classifier shard missing: {entry}"
            )

        classifier_entries.append(
            relative.as_posix()
        )

    abstraction_model: dict[str, Any] = {
        "classifier":
            classifier_entries,
        "classifier_dir":
            str(
                (
                    artifact
                    / "autoencoders"
                ).resolve()
            ),
        "model_id": None,
        "cluster_centers_init":
            cluster_centers_init,
        "cluster_centers":
            cluster_centers,
        **model_metadata,
    }

    fit_raw = manifest.get(
        "fit"
    )

    if not isinstance(
        fit_raw,
        dict,
    ):
        raise TypeError(
            "fit must be a JSON object."
        )

    train_error = _require_finite_float(
        "train_closure_error",
        fit_raw.get(
            "train_closure_error"
        ),
    )

    association_r2 = _require_finite_float(
        "association_r2",
        fit_raw.get(
            "association_r2"
        ),
    )

    history_raw = fit_raw.get(
        "nlms_history"
    )

    if not isinstance(
        history_raw,
        list,
    ):
        raise TypeError(
            "nlms_history must be a JSON list."
        )

    history = tuple(
        _require_finite_float(
            f"nlms_history[{index}]",
            value,
        )
        for index, value
        in enumerate(history_raw)
    )

    return LoadedFrozenRepresentation(
        format_id=FORMAT_ID,
        artifact_dir=str(
            artifact
        ),
        system=system,
        abstraction_model=
            abstraction_model,
        B=operator,
        omega=float(
            model_metadata[
                "kahkm_omega"
            ]
        ),
        tau=float(
            model_metadata[
                "kahkm_tau"
            ]
        ),
        train_closure_error=
            train_error,
        association_r2=
            association_r2,
        nlms_history=history,
        manifest=manifest,
    )
