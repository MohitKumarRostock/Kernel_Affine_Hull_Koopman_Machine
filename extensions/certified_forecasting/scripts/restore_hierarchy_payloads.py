#!/usr/bin/env python3
"""Restore exact frozen hierarchy records from their checked archive parts.

No gzip stream is decompressed or recompressed. All part and joined-file
hashes are checked before any destination file is installed. Installation is
atomic per file and never overwrites an existing file with different bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def safe_name(value: object) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or "/" in value or "\\" in value or Path(value).is_absolute()):
        raise ValueError(f"Invalid archive filename: {value!r}")
    return value


def check_record(record: dict) -> None:
    safe_name(record["filename"])
    if type(record.get("size_bytes")) is not int or record["size_bytes"] < 1:
        raise ValueError("Every archive record must have a positive byte size")
    for key, length in (("sha256", 64), ("git_blob_sha1", 40)):
        digest = record.get(key)
        if (not isinstance(digest, str) or len(digest) != length
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"Invalid {key} for {record['filename']}")


def hashes(size: int):
    git = hashlib.sha1()
    git.update(f"blob {size}\0".encode("ascii"))
    return hashlib.sha256(), git


def check_hashes(record: dict, size: int, sha, git) -> None:
    if (size != record["size_bytes"] or sha.hexdigest() != record["sha256"]
            or git.hexdigest() != record["git_blob_sha1"]):
        raise ValueError(f"Size or hash mismatch: {record['filename']}")


def verify_existing(path: Path, record: dict) -> bool:
    if path.is_symlink():
        raise ValueError(f"Refusing a symbolic-link destination: {path}")
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"Destination is not a regular file: {path}")
    sha, git = hashes(record["size_bytes"])
    size = 0
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(data)
            sha.update(data)
            git.update(data)
    try:
        check_hashes(record, size, sha, git)
    except ValueError as error:
        raise ValueError(f"Existing destination differs; left unchanged: {path}") from error
    return True


def restore(archive_dir: Path, output_dir: Path) -> dict:
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported archive manifest schema")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Archive manifest contains no files")
    original_names, part_names = set(), set()
    for record in records:
        check_record(record)
        if record["filename"] in original_names:
            raise ValueError("Duplicate original filename in archive manifest")
        original_names.add(record["filename"])
        parts = record.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("Archive file contains no parts")
        for part in parts:
            check_record(part)
            if part["filename"] in part_names:
                raise ValueError("Duplicate part filename in archive manifest")
            part_names.add(part["filename"])
        if sum(part["size_bytes"] for part in parts) != record["size_bytes"]:
            raise ValueError(f"Part sizes do not sum to original size: {record['filename']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        verify_existing(output_dir / record["filename"], record)

    staged = []
    installed, unchanged = [], []
    try:
        # Stage and validate the complete batch before installing any file.
        for record in records:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{record['filename']}.restore-", suffix=".tmp", dir=output_dir)
            temporary = Path(temporary)
            staged.append((temporary, record))
            whole_sha, whole_git = hashes(record["size_bytes"])
            whole_size = 0
            with os.fdopen(descriptor, "wb") as destination:
                for part in record["parts"]:
                    part_sha, part_git = hashes(part["size_bytes"])
                    part_size = 0
                    with (archive_dir / part["filename"]).open("rb") as source:
                        for data in iter(lambda: source.read(1024 * 1024), b""):
                            destination.write(data)
                            part_size += len(data)
                            whole_size += len(data)
                            part_sha.update(data)
                            part_git.update(data)
                            whole_sha.update(data)
                            whole_git.update(data)
                    check_hashes(part, part_size, part_sha, part_git)
                check_hashes(record, whole_size, whole_sha, whole_git)
                destination.flush()
                os.fsync(destination.fileno())

        for temporary, record in staged:
            destination = output_dir / record["filename"]
            if verify_existing(destination, record):
                unchanged.append(str(destination))
                continue
            try:
                # Same-directory hard linking atomically installs the complete
                # file and refuses to replace a concurrently created target.
                os.link(temporary, destination)
            except FileExistsError:
                verify_existing(destination, record)
                unchanged.append(str(destination))
            else:
                installed.append(str(destination))
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return {"verified_files": len(records), "installed": installed,
            "existing_matching_files_left_unchanged": unchanged}


def main() -> int:
    default = (Path(__file__).resolve().parents[1] / "results"
               / "learned_kahm_hierarchy_v1" / "archive_parts")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=default,
                        help="Directory containing manifest.json and its byte parts")
    parser.add_argument("--output-dir", type=Path,
                        help="Destination directory (default: parent of archive directory)")
    args = parser.parse_args()
    archive_dir = args.archive_dir.resolve()
    output_dir = (args.output_dir or archive_dir.parent).resolve()
    try:
        result = restore(archive_dir, output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Restoration failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
