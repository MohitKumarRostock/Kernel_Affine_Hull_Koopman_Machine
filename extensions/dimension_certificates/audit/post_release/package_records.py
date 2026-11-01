#!/usr/bin/env python3
"""Archive complete production records separately from generated analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from common import verify_freeze, verify_seal


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/20261101")
    args = parser.parse_args()
    run, archive = args.run.resolve(), args.archive.resolve()
    verify_freeze(run)
    verify_seal(run, "test")
    paths = [path for path in sorted(run.rglob("*"))
             if path.is_file() and path.relative_to(run).parts[0] != "analysis"]
    inventory = {str(path.relative_to(REPOSITORY)): {
        "bytes": path.stat().st_size, "sha256": digest(path)} for path in paths}
    record_root = str(run.relative_to(REPOSITORY))
    manifest = {
        "schema": "kahm-risk-dimension-numerical-records-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_freeze_sha256": digest(run / "freeze.json"),
        "test_seal_sha256": digest(run / "test_seal.json"),
        "record_root": record_root, "restore_at": "repository root",
        "generated_analysis_location": record_root + "/analysis (distributed in the source checkout)",
        "production_files": len(paths), "uncompressed_bytes": sum(row["bytes"] for row in inventory.values()),
        "files": inventory,
    }
    manifest_name = record_root + "/numerical_records_manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".partial.zip")
    if archive.exists() or temporary.exists():
        raise FileExistsError("A numerical record archive or attempt already exists")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in paths:
            zipped.write(path, str(path.relative_to(REPOSITORY)))
        zipped.writestr(manifest_name, manifest_bytes)
    with zipfile.ZipFile(temporary) as zipped:
        if set(zipped.namelist()) != set(inventory) | {manifest_name}:
            raise AssertionError("Archive roster differs from the complete record inventory")
        for name, record in inventory.items():
            h, size = hashlib.sha256(), 0
            with zipped.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
                    size += len(block)
            if size != record["bytes"] or h.hexdigest() != record["sha256"]:
                raise AssertionError(f"Archived data differ: {name}")
        if zipped.read(manifest_name) != manifest_bytes:
            raise AssertionError("Archived manifest changed")
        if zipped.testzip() is not None:
            raise AssertionError("ZIP CRC verification failed")
    temporary.rename(archive)
    receipt = {
        "schema": "kahm-risk-dimension-record-archive-verification-v1",
        "archive_name": archive.name, "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest(archive), "production_files": len(paths),
        "archive_members": len(paths) + 1, "record_uncompressed_bytes": manifest["uncompressed_bytes"],
        "manifest_member": manifest_name, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "crc_verified": True, "every_record_sha256_verified": True,
        "source_freeze_sha256": manifest["source_freeze_sha256"],
        "test_seal_sha256": manifest["test_seal_sha256"],
        "packaging_source_sha256": digest(Path(__file__)),
    }
    receipt_bytes = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    with archive.with_suffix(".verification.json").open("x") as stream:
        stream.write(receipt_bytes)
    with (ROOT / "NUMERICAL_RECORDS.json").open("x") as stream:
        stream.write(receipt_bytes)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
