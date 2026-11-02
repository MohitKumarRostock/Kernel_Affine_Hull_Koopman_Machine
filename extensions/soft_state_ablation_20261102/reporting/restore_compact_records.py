#!/usr/bin/env python3
"""Restore three losslessly compressed, already released study records.

This reporting utility performs byte/hash checks and gzip decompression only.
It never imports scientific code, fits models, scores predictions or changes
the release manifest. Existing identical files are accepted; conflicting files
are rejected before any restoration. The three destination paths are fixed.

Run from the repository root::

    python extensions/soft_state_ablation_20261102/reporting/restore_compact_records.py

Use --check-only to verify the compressed payloads without creating originals.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "reporting/compact_release_manifest.json"
SCHEMA = "soft-state-compact-release-v1"
FILES = (
    "analysis/B_candidate_scores.csv",
    "analysis/basis_dimensions.csv",
    "verification/independent_audit.json",
)
BLOCK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def confined(root, relative):
    path = Path(relative)
    resolved = (root / path).resolve()
    require(not path.is_absolute() and ".." not in path.parts and
            path.as_posix() not in ("", ".") and resolved.is_relative_to(root),
            "Record path leaves the study: " + str(relative))
    probe = root
    for part in path.parts:
        probe = probe / part
        require(not probe.is_symlink(), "Symlink is not an unambiguous record: " + str(relative))
    return resolved


def identity(path):
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "Record is not a regular file: " + str(path))
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            size += len(block)
            digest.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns),
            "Record changed during verification: " + str(path))
    return {"bytes": size, "sha256": digest.hexdigest()}


def check_identity(path, expected, description):
    actual = identity(path)
    require(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"],
            description + " differs in size or SHA256: " + str(path))
    return actual


def read_manifest(root):
    path = confined(root, MANIFEST)

    def reject_constant(value):
        raise ValueError("Non-finite manifest constant: " + value)

    manifest = json.loads(path.read_bytes(), parse_constant=reject_constant)
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete" and
            manifest.get("reporting_only") is True, "Unexpected compact release manifest")
    records = manifest["records"]
    require(isinstance(records, list) and len(records) == len(FILES) and
            {record["path"] for record in records} == set(FILES),
            "Only the three declared study records may be restored")
    index = {record["path"]: record for record in records}
    for name, record in index.items():
        require(record["gzip"]["path"] == name + ".gz", "Unexpected compressed record path")
        for value in (record, record["gzip"]):
            require(type(value["bytes"]) is int and value["bytes"] > 0 and
                    isinstance(value["sha256"], str) and
                    re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
                    "Invalid size/hash metadata for " + name)
    return [index[name] for name in FILES]


def restore(root, check_only=False):
    root = root.resolve()
    require(root.is_dir(), "Study directory is absent")
    records = read_manifest(root)
    work = []
    # Check every existing destination and compressed source first, so a known
    # conflict cannot produce a partially restored release.
    for record in records:
        target = confined(root, record["path"])
        compressed = confined(root, record["gzip"]["path"])
        present = target.exists()
        if present:
            check_identity(target, record, "Conflicting existing original")
        require(compressed.is_file(), "Compressed record is absent: " + str(compressed))
        check_identity(compressed, record["gzip"], "Compressed record")
        work.append({"record": record, "target": target, "compressed": compressed,
                     "present": present, "temporary": None})

    try:
        # Verify the decompressed bytes even for an existing identical original.
        # Stage all new originals before installing any destination.
        for item in work:
            record, target = item["record"], item["target"]
            destination = None
            if not check_only and not item["present"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix="." + target.name + ".restore-",
                                            suffix=".tmp", dir=target.parent)
                item["temporary"] = Path(name)
                destination = os.fdopen(fd, "wb")
            digest, size = hashlib.sha256(), 0
            try:
                with gzip.open(item["compressed"], "rb") as stream:
                    for block in iter(lambda: stream.read(BLOCK), b""):
                        size += len(block)
                        require(size <= record["bytes"], "Decompressed record exceeds its declared size")
                        digest.update(block)
                        if destination is not None:
                            destination.write(block)
                require(size == record["bytes"] and digest.hexdigest() == record["sha256"],
                        "Decompressed record differs in size or SHA256: " + record["path"])
                if destination is not None:
                    destination.flush()
                    os.fsync(destination.fileno())
            finally:
                if destination is not None:
                    destination.close()

        completed = []
        for item in work:
            record, target, temporary = item["record"], item["target"], item["temporary"]
            status = "verified_no_write" if check_only else "already_identical"
            if temporary is not None:
                os.chmod(temporary, 0o644)
                try:
                    # A same-directory hard link installs without replacing an
                    # existing path, including one created during verification.
                    os.link(temporary, target)
                    status = "restored"
                except FileExistsError:
                    check_identity(target, record, "Conflicting original appeared during restoration")
                    status = "already_identical"
                check_identity(target, record, "Restored original")
            completed.append({"path": record["path"], "status": status,
                              "bytes": record["bytes"], "sha256": record["sha256"]})
        return {"status": "complete", "reporting_only": True, "check_only": check_only,
                "manifest": MANIFEST, "records": completed}
    finally:
        for item in work:
            temporary = item["temporary"]
            if temporary is not None and temporary.exists():
                temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(restore(args.study_root, args.check_only), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
