"""Execution-freeze verification shared by the two sensitivity runners.

The freeze is a new execution checkpoint for a post-hoc analysis. It does not
reclassify previously inspected datasets as fresh confirmation evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_freeze(root: Path | None = None) -> dict:
    root = ROOT if root is None else Path(root).resolve()
    extension = root / "extensions/scaling_sensitivity"
    freeze = json.loads((extension / "protocol_freeze.json").read_text())
    if freeze.get("format") != "scaling_sensitivity_execution_freeze_v1":
        raise ValueError("Unexpected sensitivity execution freeze")
    if digest(extension / "protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("Sensitivity protocol changed after the execution freeze")
    for name, expected in freeze["sha256"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("Unsafe frozen dependency path")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or escaped frozen dependency: {name}")
        if digest(path) != expected:
            raise ValueError(f"Frozen sensitivity dependency changed: {name}")
    return freeze
