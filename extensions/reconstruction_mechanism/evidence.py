#!/usr/bin/env python3
"""Deterministic, lossless transport of completed mechanism/adaptive evidence.

Original files are never deleted.  The public tree may omit exactly the member
paths in PACKED_EVIDENCE.json and retain that manifest and its <=4MiB parts.
Unpacking rejects conflicting bytes and verifies the original scientific
freezes; this transport is not a replacement freeze or a new research result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from .package_results import describe, json_bytes, read_json, safe_file, safe_name


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = Path("extensions/reconstruction_mechanism")
ADAPTIVE = Path("extensions/adaptive_development")
MANIFEST = MECHANISM / "PACKED_EVIDENCE.json"
PARTS = MECHANISM / "evidence_parts"
INVENTORY = "EVIDENCE_INVENTORY.json"
PART_BYTES = 4 * 1024 * 1024
SCHEMA = "kahm-completed-evidence-transport-v1"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _add_tree(root, directory, files):
    path = root / directory
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Missing or linked evidence directory: {directory}")
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"Linked evidence member: {child}")
        if child.is_file():
            name = child.relative_to(root).as_posix()
            if any(part.startswith(".") or part == "__pycache__" for part in child.relative_to(path).parts):
                raise ValueError(f"Unexpected hidden/cache evidence member: {name}")
            files[name] = safe_file(root, name)


def frozen_integrity(root, *, require_adaptive=True):
    """Verify restored paths against both unchanged original scientific freezes."""
    mechanism = read_json(safe_file(root, MECHANISM / "freeze.json"))
    checked = 0
    for record in mechanism["files"]:
        path = safe_file(root, record["path"])
        if describe(path)["sha256"] != record["sha256"]:
            raise ValueError(f"Mechanism frozen file differs: {record['path']}")
        checked += 1
    adaptive_path = root / ADAPTIVE / "DEVELOPMENT_FREEZE.json"
    if require_adaptive or adaptive_path.exists():
        adaptive = read_json(safe_file(root, ADAPTIVE / "DEVELOPMENT_FREEZE.json"))
        for name, expected in adaptive["files"].items():
            path = safe_file(root, ADAPTIVE / safe_name(name))
            if describe(path) != expected:
                raise ValueError(f"Adaptive frozen file differs: {name}")
            checked += 1
    return {"status": "passed", "frozen_files": checked,
            "mechanism_freeze": describe(root / MECHANISM / "freeze.json"),
            "adaptive_freeze": describe(adaptive_path) if adaptive_path.exists() else None}


def payload(root):
    files = {}
    manifest = read_json(safe_file(root, MECHANISM / "prepared_inputs/manifest.json"))
    execution = read_json(safe_file(root, MECHANISM / "results/execution.json"))
    tasks = manifest["tasks"]
    if manifest.get("development_only") is not False or len(tasks) != 192 or not execution.get("completed_utc"):
        raise ValueError("Pack only the completed 192-task execution, never a partial running study")
    declared = {task["task_id"] for task in tasks}
    if len(declared) != 192:
        raise ValueError("Repeated mechanism task identity")
    for task in tasks:
        name = (MECHANISM / safe_name(task["input"])).as_posix()
        if not Path(name).is_relative_to(MECHANISM / "prepared_inputs/tasks"):
            raise ValueError("Unexpected prepared-input transport path")
        files[name] = safe_file(root, name)
        directory = MECHANISM / "results" / safe_name(task["task_id"])
        summary = read_json(safe_file(root, directory / "summary.json"))
        if summary.get("status") != "completed":
            raise ValueError(f"Final mechanism task is incomplete: {task['task_id']}")
        _add_tree(root, directory, files)
    development = read_json(safe_file(root, MECHANISM / "development_inputs/manifest.json"))
    if development.get("development_only") is not True:
        raise ValueError("Development transport must retain its development designation")
    for task in development["tasks"]:
        name = (MECHANISM / safe_name(task["input"])).as_posix()
        if not Path(name).is_relative_to(MECHANISM / "development_inputs/tasks"):
            raise ValueError("Unexpected development-input transport path")
        files[name] = safe_file(root, name)
        _add_tree(root, MECHANISM / "development_results" / safe_name(task["task_id"]), files)
    adaptive_execution = read_json(safe_file(root, ADAPTIVE / "results/execution.json"))
    if not adaptive_execution.get("finished_at") or len(adaptive_execution["roster"]) != 144:
        raise ValueError("The complete retained adaptive execution is required")
    for data_file in sorted((root / ADAPTIVE / "prepared_inputs").glob("*.npz")):
        name = data_file.relative_to(root).as_posix()
        files[name] = safe_file(root, name)
    for task_id, method, capacity in adaptive_execution["roster"]:
        directory = ADAPTIVE / "results" / safe_name(f"{task_id}__{capacity}__{method}")
        _add_tree(root, directory, files)
    original = ADAPTIVE / "archives/initial_attempt.zip"
    files[original.as_posix()] = safe_file(root, original)
    # Detailed derivative tables exceed a connector blob's practical size;
    # compact decisions/contrasts/aggregates remain immediately readable.
    for name in ("analysis/resources.csv", "analysis/safety_diagnostics.csv"):
        relative = MECHANISM / name
        files[relative.as_posix()] = safe_file(root, relative)
    return files


def _zip_inventory(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or INVENTORY not in names or len(names) > 30000:
            raise ValueError("Incomplete, duplicate or excessive evidence ZIP entries")
        if sum(entry.file_size for entry in entries) > 4 * 1024 ** 3:
            raise ValueError("Evidence archive exceeds the uncompressed-byte ceiling")
        for entry in entries:
            safe_name(entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                raise ValueError("Directory, symlink or encrypted evidence ZIP member")
        if archive.getinfo(INVENTORY).file_size > 32 * 1024 * 1024:
            raise ValueError("Excessive evidence inventory size")
        inventory = json.loads(archive.read(INVENTORY))
        if inventory.get("schema_version") != SCHEMA or set(names) != set(inventory["files"]) | {INVENTORY}:
            raise ValueError("Evidence ZIP differs from its inventory")
        for name, expected in inventory["files"].items():
            digest, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if {"bytes": size, "sha256": digest.hexdigest()} != expected:
                raise ValueError(f"Evidence ZIP member differs: {name}")
    return inventory


def _write_identical_or_new(path, content, *, replace_transport=False):
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"Existing differing transport file is preserved: {path}")
        if describe(path) == describe(content):
            return
        if not replace_transport:
            raise FileExistsError(f"Existing differing transport file is preserved: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".transport-stage-", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def pack(root=ROOT, *, replace_transport=False):
    root = Path(root).resolve()
    frozen = frozen_integrity(root)
    files = payload(root)
    inventory = {"schema_version": SCHEMA, "files": {name: describe(path) for name, path in sorted(files.items())}}
    with tempfile.TemporaryDirectory(prefix="kahm_evidence_pack_") as temporary:
        archive_path = Path(temporary) / "completed_evidence.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted({**files, INVENTORY: json_bytes(inventory)}.items()):
                entry = zipfile.ZipInfo(name, date_time=ZIP_DATE)
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                with archive.open(entry, "w", force_zip64=True) as target:
                    if isinstance(value, bytes):
                        target.write(value)
                    else:
                        with value.open("rb") as source:
                            shutil.copyfileobj(source, target, length=1024 * 1024)
        if _zip_inventory(archive_path) != inventory:
            raise ValueError("Staged ZIP inventory changed")
        part_records = []
        with archive_path.open("rb") as stream:
            number = 1
            for content in iter(lambda: stream.read(PART_BYTES), b""):
                name = (PARTS / f"completed_evidence.zip.part{number:04d}").as_posix()
                _write_identical_or_new(root / name, content, replace_transport=replace_transport)
                part_records.append({"path": name, **describe(content)})
                number += 1
        manifest = {"schema_version": SCHEMA, "scope": "Lossless publication transport; no Git-history or unpublished-roadmap bundle",
                    "archive": {"name": "completed_evidence.zip", **describe(archive_path)},
                    "part_bytes": PART_BYTES, "parts": part_records,
                    "files": inventory["files"], "scientific_freezes": frozen,
                    "loose_files_policy": "The public tree may omit only these inventoried member paths. Code, freezes, manifests, execution/audit/replay summaries, compact analysis aggregates and findings remain loose; the two large derivative detail tables are losslessly restored.",
                    "restoration": "python -m extensions.reconstruction_mechanism.evidence unpack"}
        _write_identical_or_new(root / MANIFEST, json_bytes(manifest), replace_transport=replace_transport)
    return {"status": "packed", "members": len(files), "parts": len(part_records),
            "archive": manifest["archive"], "original_files_deleted": 0,
            "public_files": [MANIFEST.as_posix()] + [entry["path"] for entry in part_records],
            "excluded_loose_paths_source": MANIFEST.as_posix() + ":files"}


def _join_and_verify(root, temporary):
    manifest = read_json(safe_file(root, MANIFEST))
    if manifest.get("schema_version") != SCHEMA or manifest.get("part_bytes") != PART_BYTES:
        raise ValueError("Unexpected evidence transport format")
    parts = manifest["parts"]
    if not parts or len({part["path"] for part in parts}) != len(parts):
        raise ValueError("Empty or duplicate evidence part roster")
    archive_path = Path(temporary) / "completed_evidence.zip"
    with archive_path.open("wb") as output:
        for part in parts:
            path = safe_file(root, part["path"])
            if not 0 < part["bytes"] <= PART_BYTES or describe(path) != {key: part[key] for key in ("bytes", "sha256")}:
                raise ValueError(f"Evidence part differs: {part['path']}")
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
    if describe(archive_path) != {key: manifest["archive"][key] for key in ("bytes", "sha256")}:
        raise ValueError("Joined evidence archive differs from its manifest")
    inventory = _zip_inventory(archive_path)
    if inventory["files"] != manifest["files"]:
        raise ValueError("Internal and public evidence inventories disagree")
    return manifest, archive_path


def verify(root=ROOT):
    root = Path(root).resolve()
    with tempfile.TemporaryDirectory(prefix="kahm_evidence_verify_") as temporary:
        manifest, _ = _join_and_verify(root, temporary)
    return {"status": "passed", "members": len(manifest["files"]), "parts": len(manifest["parts"]),
            "manifest": describe(root / MANIFEST), "archive": manifest["archive"]}


def unpack(root=ROOT):
    root = Path(root).resolve()
    if not (root / MANIFEST).exists():
        # Expanded downloadable reproduction bundles intentionally omit parts.
        return {"status": "already_expanded", "restored": 0,
                "frozen_integrity": frozen_integrity(root, require_adaptive=False)}
    with tempfile.TemporaryDirectory(prefix="kahm_evidence_unpack_") as temporary:
        manifest, archive_path = _join_and_verify(root, temporary)
        pending, identical = [], 0
        for name, expected in manifest["files"].items():
            safe_name(name)
            destination = root / name
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Escaped restoration path: {name}")
            for component in (destination, *destination.parents):
                if component == root:
                    break
                if component.is_symlink():
                    raise ValueError(f"Linked restoration destination: {name}")
            if destination.exists():
                if not destination.is_file() or describe(destination) != expected:
                    raise FileExistsError(f"Existing differing evidence is preserved: {name}")
                identical += 1
            else:
                pending.append((name, expected))
        # Validate every source/member and every existing destination before any write.
        with zipfile.ZipFile(archive_path) as archive:
            for name, expected in pending:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if describe(destination) != expected:
                    raise ValueError(f"Restored evidence bytes differ: {name}")
    checked = frozen_integrity(root)
    return {"status": "passed", "restored": len(pending), "existing_identical": identical,
            "members": len(manifest["files"]), "frozen_integrity": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "unpack", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--replace-transport", action="store_true",
                        help="For pack only: replace generated transport parts/manifest, never original evidence")
    args = parser.parse_args()
    if args.replace_transport and args.command != "pack":
        parser.error("--replace-transport is only valid with pack")
    result = (pack(args.root, replace_transport=args.replace_transport) if args.command == "pack"
              else {"unpack": unpack, "verify": verify}[args.command](args.root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
