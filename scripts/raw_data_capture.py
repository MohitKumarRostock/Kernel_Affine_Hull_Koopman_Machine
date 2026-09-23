#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

FORMAT_VERSION = 1
DATASET_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def git_state(root: Path) -> dict[str, Any]:
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    return {
        "commit": commit,
        "tree_dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def canonical_array_content_sha256(
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()

    for name in sorted(arrays):
        if not name:
            raise ValueError("Array names must be non-empty.")

        array = np.asarray(arrays[name])

        if array.dtype.hasobject:
            raise TypeError(
                f"Object arrays are not supported for raw-data capture: {name}"
            )

        contiguous = np.ascontiguousarray(array)
        shape_json = json.dumps(
            list(contiguous.shape),
            separators=(",", ":"),
        ).encode("utf-8")

        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(shape_json)
        digest.update(b"\0")
        digest.update(memoryview(contiguous).cast("B"))
        digest.update(b"\0")

    return digest.hexdigest()


def array_schema(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for name in sorted(arrays):
        array = np.asarray(arrays[name])

        result[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "nbytes": int(array.nbytes),
        }

    return result


def capture_environment(root: Path) -> dict[str, Any]:
    lockfile = root / "requirements-lock-arm64.txt"

    return {
        "python": sys.version.split()[0],
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "requirements_lock_arm64_sha256": (
            sha256_file(lockfile) if lockfile.is_file() else None
        ),
    }


def save_raw_dataset(
    *,
    root: Path,
    output_dir: Path,
    dataset_id: str,
    role: str,
    source_script: str,
    generator_seed: int | str | None,
    arrays: Mapping[str, np.ndarray],
    generation_parameters: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    root = root.resolve()
    output_dir = output_dir.resolve()

    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError(
            "dataset_id may contain only letters, digits, '.', '_' and '-'."
        )

    if not arrays:
        raise ValueError("At least one array is required.")

    normalized: dict[str, np.ndarray] = {}

    for name, value in arrays.items():
        array = np.asarray(value)

        if array.dtype.hasobject:
            raise TypeError(
                f"Object arrays are not supported for raw-data capture: {name}"
            )

        normalized[str(name)] = np.ascontiguousarray(array)

    pre_capture_state = git_state(root)

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{dataset_id}.npz"
    metadata_path = output_dir / f"{dataset_id}.json"

    tmp_npz = output_dir / f".{dataset_id}.npz.tmp"

    with tmp_npz.open("wb") as handle:
        np.savez_compressed(handle, **normalized)

    tmp_npz.replace(npz_path)

    content_hash = canonical_array_content_sha256(normalized)
    archive_hash = sha256_file(npz_path)

    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "dataset_id": dataset_id,
        "role": role,
        "source_script": source_script,
        "generator_seed": generator_seed,
        "generation_parameters": dict(generation_parameters),
        "arrays": array_schema(normalized),
        "content_sha256": content_hash,
        "archive_sha256": archive_hash,
        "archive_bytes": npz_path.stat().st_size,
        "git": pre_capture_state,
        "environment": capture_environment(root),
    }

    if extra_metadata:
        metadata["extra_metadata"] = dict(extra_metadata)

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return npz_path, metadata_path


def verify_raw_dataset(
    npz_path: Path,
    metadata_path: Path,
) -> None:
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if sha256_file(npz_path) != metadata["archive_sha256"]:
        raise RuntimeError(
            f"Archive SHA-256 mismatch: {npz_path}"
        )

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
        }

    actual_content_hash = canonical_array_content_sha256(arrays)

    if actual_content_hash != metadata["content_sha256"]:
        raise RuntimeError(
            f"Array-content SHA-256 mismatch: {npz_path}"
        )

    expected_schema = metadata["arrays"]
    actual_schema = array_schema(arrays)

    if actual_schema != expected_schema:
        raise RuntimeError(
            f"Array schema mismatch: {npz_path}"
        )


__all__ = [
    "FORMAT_VERSION",
    "array_schema",
    "canonical_array_content_sha256",
    "capture_environment",
    "git_state",
    "save_raw_dataset",
    "sha256_file",
    "verify_raw_dataset",
]
