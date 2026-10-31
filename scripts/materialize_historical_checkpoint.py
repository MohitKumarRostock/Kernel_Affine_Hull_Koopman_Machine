#!/usr/bin/env python3
"""Create an isolated source tree for a hash-bound historical study.

The historical files are reconstructed from their recorded Git commit. Current
study files are copied from explicitly selected tracked prefixes. The source
checkout, scientific checkpoint, and existing destinations are never modified.
This utility performs no network requests and does not fit forecasting models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile


EDITORIAL_PATHS = frozenset({
    "README.md",
    "REPRODUCIBILITY.md",
    "extensions/predictive_geometry/PAPER_READINESS.md",
    "extensions/predictive_geometry/README.md",
    "extensions/predictive_geometry/REPRODUCING.md",
    "extensions/predictive_geometry/TMLR_SUBMISSION.md",
})


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or "\\" in value or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError(f"Unsafe repository-relative path: {value!r}")
    return path.as_posix()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(source: Path, *arguments: str) -> bytes:
    environment = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(
        ["git", "-C", str(source), *arguments], capture_output=True,
        env=environment, check=False,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git could not read the required local object: {detail}")
    return result.stdout


def regular_source(source: Path, relative: str) -> Path:
    path = source / safe_relative(relative)
    if path.is_symlink() or not path.resolve().is_relative_to(source) or not path.is_file():
        raise FileNotFoundError(
            f"Required tracked file is absent or unsafe: {relative}. "
            "Populate the selected prefix in a complete checkout before materialization."
        )
    return path


def materialize(source: Path, checkpoint_relative: str, output: Path, prefixes: list[str], *, verify_current_source: bool = False) -> dict:
    source = source.resolve()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Destination already exists; choose a new path: {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("The destination must be outside the source checkout.")
    checkpoint_relative = safe_relative(checkpoint_relative)
    checkpoint_path = regular_source(source, checkpoint_relative)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    commit = checkpoint["checkpoint_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The checkpoint must contain a full hexadecimal Git commit identifier.")
    expected = checkpoint["tracked_file_sha256"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("The checkpoint must contain a nonempty tracked_file_sha256 mapping.")
    for name, value in expected.items():
        safe_relative(name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Invalid SHA-256 entry: {name}")
    current_files_verified = 0
    if verify_current_source:
        for name, expected_hash in expected.items():
            if name in EDITORIAL_PATHS:
                continue
            path = regular_source(source, name)
            if digest(path) != expected_hash:
                raise ValueError(f"Current scientific or archival source differs from the historical checkpoint: {name}")
            current_files_verified += 1
    prefixes = [safe_relative(p.rstrip("/")) for p in prefixes]
    tracked = git(source, "ls-files", "-z", "--", *prefixes).decode("utf-8").split("\0")
    selected = sorted({safe_relative(p) for p in tracked if p})
    if not selected:
        raise ValueError("No tracked files match the selected study prefixes.")
    if checkpoint_relative not in selected:
        selected.append(checkpoint_relative)
    # Validate all working-tree sources before creating the destination.
    sources = {name: regular_source(source, name) for name in selected if name not in expected}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kahm-checkpoint-", dir=output.parent) as temporary:
        stage = Path(temporary) / "tree"
        stage.mkdir()
        copied = {}
        for name, path in sorted(sources.items()):
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied[name] = digest(destination)
        for name, expected_hash in sorted(expected.items()):
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            current = source / name
            if current.is_file() and not current.is_symlink() and current.resolve().is_relative_to(source) and digest(current) == expected_hash:
                shutil.copy2(current, destination)
            else:
                try:
                    payload = git(source, "show", f"{commit}:{name}")
                except RuntimeError as error:
                    raise RuntimeError(
                        f"Historical content is unavailable locally for {name}. "
                        f"The required commit is {commit}. In a shallow or partial clone, "
                        "explicitly fetch the historical commit and its blobs before retrying "
                        "(for example, git fetch --refetch --no-filter --depth 1 origin " + commit + "). "
                        "This helper does not download Git history or objects."
                    ) from error
                destination.write_bytes(payload)
            actual = digest(destination)
            if actual != expected_hash:
                raise ValueError(f"Historical hash mismatch for {name}: expected {expected_hash}, found {actual}")
        record = {
            "schema_version": 1,
            "historical_commit": commit,
            "checkpoint_path": checkpoint_relative,
            "checkpoint_sha256": digest(checkpoint_path),
            "historical_files_verified": len(expected),
            "current_source_verification": {
                "required": verify_current_source,
                "files_verified": current_files_verified,
                "editorial_exceptions": sorted(EDITORIAL_PATHS & expected.keys()) if verify_current_source else [],
            },
            "historical_file_sha256": dict(sorted(expected.items())),
            "copied_current_study_file_sha256": copied,
            "scope": "Isolated historical source reconstruction plus explicitly selected current study files; no new execution or preregistration.",
        }
        (stage / "MATERIALIZATION_MANIFEST.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Destination appeared during preparation: {output}")
        stage.rename(output)
    return {
        "output": str(output), "historical_commit": commit,
        "historical_files_verified": len(expected),
        "current_study_files_copied": len(copied),
        "current_source_files_verified": current_files_verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1], help="Complete source checkout; never modified.")
    parser.add_argument("--checkpoint", default="extensions/natural_forecasting/CHECKPOINT.json", help="Repository-relative historical checkpoint JSON.")
    parser.add_argument("--include", action="append", dest="prefixes", help="Tracked current study prefix to copy; repeat for multiple prefixes. Defaults to extensions/natural_forecasting.")
    parser.add_argument("--verify-current-source", action="store_true", help="Require every current checkpoint file except the six named reader documents to retain its historical hash before materialization.")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the source checkout; existing paths are rejected.")
    arguments = parser.parse_args()
    try:
        result = materialize(arguments.source, arguments.checkpoint, arguments.output,
                             arguments.prefixes or ["extensions/natural_forecasting"],
                             verify_current_source=arguments.verify_current_source)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"Materialization failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
