#!/usr/bin/env python3
"""Materialize committed scientific inputs in a new isolated directory.

All tracked files come from the current Git HEAD. Three fixed Markdown inputs
use their exact identities recorded by the scientific freezes. Code, data,
protocols, and freeze files retain their current committed bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys


SOURCE_COMMIT = "5123365917ae630b684f8a3bebb0294ee695ce7a"
# These SHA-256 values are recorded in target_geometry/freeze.json and
# adaptive_development/DEVELOPMENT_FREEZE.json at SOURCE_COMMIT.
DOCUMENTS = {
    "extensions/target_geometry/THEORY.md": {
        "git_blob": "040819f2f14a7cd72e4f46b0a5af8c096f9ead68",
        "sha256": "91de995e9777fc975658742077daa672afd97c1a298bbdef5eafc2c6e75b4a49",
    },
    "extensions/adaptive_development/README.md": {
        "git_blob": "d0e01136a81c3dd7a0b01d3e7e4266dd1be82a13",
        "sha256": "9513971b74a0809a0cf4a1ddca5bf949bde1cbc262f752770a985b03cf1b482f",
    },
    "extensions/adaptive_development/REVIEW_NOTES.md": {
        "git_blob": "b9c5bc4c4ad402da7b1204f93ad0b639aa1f66c2",
        "sha256": "260aba5b4baa526b93801dace459c5d0f9ddc5bbded4eaf5beeaf6791de756f6",
    },
}


class ContextError(RuntimeError):
    """The requested scientific context cannot be verified."""


def git(repository: Path, *arguments: str, check: bool = True):
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments], capture_output=True,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    if check and result.returncode:
        raise ContextError(result.stderr.decode(errors="replace").strip())
    return result


def blob_identity(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def file_identity(path: Path) -> str:
    digest = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def committed_files(repository: Path, commit: str):
    files = []
    for record in git(repository, "ls-tree", "-r", "-z", "--full-tree", commit).stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, identity = metadata.decode("ascii").split()
        name = os.fsdecode(raw_path)
        parts = PurePosixPath(name).parts
        if (not parts or PurePosixPath(name).is_absolute()
                or any(part in (".", "..") or part.lower() == ".git" for part in parts)):
            raise ContextError(f"Unsafe tracked path: {name}")
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ContextError(f"Unsupported tracked object at {name}: {mode} {kind}")
        files.append((name, mode, identity))
    return files


def copy_committed_files(repository: Path, output: Path, files):
    # One Git process streams all binary/text blobs without export attributes.
    with subprocess.Popen(
        ["git", "-C", str(repository), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    ) as process:
        try:
            for name, mode, identity in files:
                process.stdin.write(identity.encode("ascii") + b"\n")
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3 or header[:2] != [identity.encode("ascii"), b"blob"]:
                    raise ContextError(f"Cannot read committed blob: {name}")
                remaining = int(header[2])
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    while remaining:
                        block = process.stdout.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ContextError(f"Incomplete committed blob: {name}")
                        stream.write(block)
                        remaining -= len(block)
                if process.stdout.read(1) != b"\n":
                    raise ContextError(f"Invalid Git blob framing: {name}")
                target.chmod(0o755 if mode == "100755" else 0o644)
            process.stdin.close()
            if process.wait():
                raise ContextError("Git could not finish reading committed files")
        except BaseException:
            process.kill()
            process.wait()
            raise


def materialize(repository: Path, requested_output: Path) -> dict:
    repository = Path(os.fsdecode(git(repository, "rev-parse", "--show-toplevel").stdout.strip())).resolve()
    requested_output = requested_output.expanduser()
    if os.path.lexists(requested_output):
        raise ContextError("Output already exists; choose a new directory")
    output = requested_output.resolve()
    if output == repository or output.is_relative_to(repository):
        raise ContextError("Output must be outside the repository")
    if git(repository, "rev-parse", "--show-object-format").stdout.strip() != b"sha1":
        raise ContextError("The recorded source identities require a SHA-1 Git repository")
    current_commit = git(repository, "rev-parse", "--verify", "HEAD^{commit}").stdout.decode().strip()
    source = git(repository, "cat-file", "-t", SOURCE_COMMIT, check=False)
    if source.returncode or source.stdout.strip() != b"commit":
        command = shlex.join(["git", "-C", str(repository), "fetch", "origin", SOURCE_COMMIT])
        raise ContextError(f"Required source commit is not available locally. Run: {command}")
    files = committed_files(repository, current_commit)
    head_paths = {name for name, _, _ in files}
    documents, identities = {}, []
    for name, expected in DOCUMENTS.items():
        if name not in head_paths:
            raise ContextError(f"Current HEAD lacks the required documentation path: {name}")
        raw = git(repository, "cat-file", "blob", f"{SOURCE_COMMIT}:{name}").stdout
        actual_blob, actual_sha256 = blob_identity(raw), hashlib.sha256(raw).hexdigest()
        if actual_blob != expected["git_blob"] or actual_sha256 != expected["sha256"]:
            raise ContextError(f"Source documentation identity mismatch: {name}")
        documents[name] = raw
        identities.append({"path": name, "git_blob": actual_blob,
                           "sha256": actual_sha256, "bytes": len(raw)})

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()  # Exclusive creation; never replace an existing directory.
    except FileExistsError as error:
        raise ContextError("Output already exists; choose a new directory") from error
    try:
        copy_committed_files(repository, output, files)
        for name, raw in documents.items():
            (output / name).write_bytes(raw)
        for name, mode, identity in files:
            expected = DOCUMENTS[name]["git_blob"] if name in DOCUMENTS else identity
            target = output / name
            if file_identity(target) != expected:
                raise ContextError(f"Copied file identity mismatch: {name}")
            if target.stat().st_mode & 0o777 != (0o755 if mode == "100755" else 0o644):
                raise ContextError(f"Copied file mode mismatch: {name}")
    except BaseException:
        shutil.rmtree(output)
        raise
    return {"status": "PASS", "current_commit": current_commit,
            "source_commit": SOURCE_COMMIT, "output": str(output),
            "verified_files": len(files), "current_head_files": len(files) - len(DOCUMENTS),
            "documentation_inputs": identities}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New isolated directory outside the repository")
    arguments = parser.parse_args(argv)
    try:
        result = materialize(Path(__file__).resolve().parents[1], arguments.output)
    except (ContextError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
