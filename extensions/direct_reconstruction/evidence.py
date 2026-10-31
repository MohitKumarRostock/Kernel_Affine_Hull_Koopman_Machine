#!/usr/bin/env python3
"""Lossless transport of completed per-task evidence; no scientific computation.

``pack`` preserves original task directories and writes deterministic archives.
``unpack`` verifies every archive/member before restoring exact file paths and
bytes. Existing identical files are accepted; differing files are never
overwritten by restoration. This utility does not fit, score, or revise models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "PACKED_MANIFEST.json"
ARCHIVE_DIRECTORY = "packed_tasks"
SCHEMA = "kahm-lossless-task-evidence-v1"
METADATA_FILES = ("freeze.json", "protocol.json", "execution.json")
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
MAX_FILES = 30000
MAX_BYTES = 2 * 1024 ** 3


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def describe_bytes(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def describe_file(path):
    digest = hashlib.sha256()
    length = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            length += len(chunk)
            digest.update(chunk)
    return {"bytes": length, "sha256": digest.hexdigest()}


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError("Evidence paths must be strings")
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\0" in name
            or any(part in {"", ".", ".."} or ":" in part for part in name.split("/"))):
        raise ValueError(f"Unsafe evidence path: {name!r}")
    return name


def checked_root(run_dir):
    path = Path(run_dir).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Evidence paths cannot follow symbolic links")
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Missing run directory: {path}")
    return path


def checked_path(root, name, *, require_file=False):
    path = root / safe_name(name)
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError(f"Symbolic link in evidence path: {name}")
        if ancestor != path and ancestor.exists() and not ancestor.is_dir():
            raise ValueError(f"Evidence parent is not a directory: {name}")
    if path.exists() and not (path.is_file() or path.is_dir()):
        raise ValueError(f"Special filesystem object in evidence: {name}")
    if require_file and not path.is_file():
        raise ValueError(f"Missing regular evidence file: {name}")
    return path


def metadata(root):
    contents = {name: checked_path(root, name, require_file=True).read_bytes() for name in METADATA_FILES}
    protocol = json.loads(contents["protocol.json"])
    execution = json.loads(contents["execution.json"])
    if execution.get("status") not in {"complete", "complete_with_failures"}:
        raise ValueError("Evidence packing/restoration requires a finished execution record")
    tasks = protocol.get("tasks", [])
    identifiers = [task.get("task_id") for task in tasks]
    if (not identifiers or any(not isinstance(name, str)
                              or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name)
                              for name in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError("Protocol must declare unique safe task identifiers")
    return protocol, execution, {name: describe_bytes(content) for name, content in contents.items()}


def original_tasks(root, protocol):
    task_root = checked_path(root, "tasks")
    if not task_root.is_dir():
        raise ValueError("Original tasks directory is absent; unpack first if packed evidence exists")
    expected = {task["task_id"] for task in protocol["tasks"]}
    entries = list(task_root.iterdir())
    if ({path.name for path in entries} != expected
            or any(not path.is_dir() or path.is_symlink() for path in entries)):
        raise ValueError("Original task directories differ from the complete protocol roster")
    result = []
    for task in protocol["tasks"]:
        identifier = task["task_id"]
        base = task_root / identifier
        task_path = checked_path(root, f"tasks/{identifier}/task.json", require_file=True)
        if json.loads(task_path.read_bytes()) != task:
            raise ValueError(f"Task metadata differs from the protocol: {identifier}")
        complete, failed = (base / "complete.json").is_file(), (base / "failure.json").is_file()
        if complete == failed:
            raise ValueError(f"Task needs exactly one completed or failed outcome: {identifier}")
        files = []
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(root).as_posix()
            checked_path(root, relative)
            if path.is_file():
                files.append(relative)
        if not files:
            raise ValueError(f"Task contains no evidence files: {identifier}")
        result.append((identifier, "complete" if complete else "failed", files))
    return result


def pack(run_dir, *, overwrite=False):
    root = checked_root(run_dir)
    protocol, execution, input_hashes = metadata(root)
    tasks = original_tasks(root, protocol)
    packed_dir = checked_path(root, ARCHIVE_DIRECTORY)
    if packed_dir.exists() and not packed_dir.is_dir():
        raise ValueError("Packed evidence location is not a directory")
    packed_dir.mkdir(exist_ok=True)
    expected_archives = {identifier + ".zip" for identifier, _, _ in tasks}
    if {path.name for path in packed_dir.glob("*.zip")} - expected_archives:
        raise ValueError("Packed directory contains archives outside the protocol roster")
    manifest = {
        "schema_version": SCHEMA,
        "scope": "Lossless transport only; original scientific evidence files and metadata are unchanged.",
        "metadata_files": input_hashes,
        "task_count": len(tasks), "archives": [],
    }
    with tempfile.TemporaryDirectory(prefix=".pack_stage_", dir=packed_dir) as temporary:
        temporary = Path(temporary)
        products = []
        for identifier, outcome, names in tasks:
            filename = identifier + ".zip"
            archive_path = temporary / filename
            files = {}
            with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
                for name in names:
                    content = checked_path(root, name, require_file=True).read_bytes()
                    files[name] = describe_bytes(content)
                    info = zipfile.ZipInfo(safe_name(name), date_time=ZIP_DATE)
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    info.compress_type = (zipfile.ZIP_STORED if name.endswith((".npz", ".zip", ".png", ".pdf"))
                                          else zipfile.ZIP_DEFLATED)
                    archive.writestr(info, content, compresslevel=9)
            manifest["archives"].append({
                "task_id": identifier, "outcome": outcome,
                "path": ARCHIVE_DIRECTORY + "/" + filename,
                **describe_file(archive_path), "files": files,
            })
            products.append((archive_path, packed_dir / filename))
        manifest["unpacked_file_count"] = sum(len(item["files"]) for item in manifest["archives"])
        manifest["unpacked_bytes"] = sum(value["bytes"] for item in manifest["archives"] for value in item["files"].values())
        manifest["packed_bytes"] = sum(item["bytes"] for item in manifest["archives"])
        if manifest["unpacked_file_count"] > MAX_FILES or manifest["unpacked_bytes"] > MAX_BYTES:
            raise ValueError("Evidence exceeds the declared transport limits")
        staged_manifest = temporary / MANIFEST_NAME
        staged_manifest.write_bytes(json_bytes(manifest))
        products.append((staged_manifest, root / MANIFEST_NAME))
        # Check every existing destination before publishing any new artifact.
        for source, destination in products:
            relative = destination.relative_to(root).as_posix()
            checked_path(root, relative)
            if destination.exists() and (not destination.is_file()
                                         or (describe_file(destination) != describe_file(source) and not overwrite)):
                raise FileExistsError(f"Existing differing packed output is preserved: {relative}")
        for source, destination in products:
            if destination.exists() and describe_file(destination) == describe_file(source):
                continue
            os.replace(source, destination)
    # Packing never removes or changes the original task directories.
    return {"status": "packed", "task_count": len(tasks),
            "manifest": str(root / MANIFEST_NAME), **describe_file(root / MANIFEST_NAME),
            "unpacked_file_count": manifest["unpacked_file_count"],
            "unpacked_bytes": manifest["unpacked_bytes"], "packed_bytes": manifest["packed_bytes"],
            "originals_modified": False}


def validate_description(value):
    if (not isinstance(value, dict) or not isinstance(value.get("bytes"), int)
            or isinstance(value["bytes"], bool) or value["bytes"] < 0
            or not isinstance(value.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
        raise ValueError("Invalid file size or SHA-256 in packed manifest")


def verified_plan(run_dir, *, check_destinations=False):
    root = checked_root(run_dir)
    protocol, execution, metadata_hashes = metadata(root)
    manifest = json.loads(checked_path(root, MANIFEST_NAME, require_file=True).read_bytes())
    if (manifest.get("schema_version") != SCHEMA
            or manifest.get("metadata_files") != metadata_hashes):
        raise ValueError("Packed manifest schema or linked run metadata differs")
    records = manifest.get("archives", [])
    identifiers = [task["task_id"] for task in protocol["tasks"]]
    if (len(records) != len(identifiers) or manifest.get("task_count") != len(identifiers)
            or [record.get("task_id") for record in records] != identifiers):
        raise ValueError("Packed task roster differs from the protocol")
    all_files = set()
    unpacked_bytes = 0
    packed_bytes = 0
    for record in records:
        identifier = record["task_id"]
        if (record.get("path") != f"{ARCHIVE_DIRECTORY}/{identifier}.zip"
                or record.get("outcome") not in {"complete", "failed"}
                or not isinstance(record.get("files"), dict) or not record["files"]):
            raise ValueError("Invalid packed task path, outcome, or member inventory")
        validate_description(record)
        packed_bytes += record["bytes"]
        prefix = "tasks/" + identifier + "/"
        for name, expected in record["files"].items():
            safe_name(name)
            validate_description(expected)
            if not name.startswith(prefix) or name in all_files:
                raise ValueError("Member escapes its task or appears in multiple archives")
            all_files.add(name)
            unpacked_bytes += expected["bytes"]
        if len(all_files) > MAX_FILES or unpacked_bytes > MAX_BYTES:
            raise ValueError("Packed manifest exceeds the declared transport limits")
    if (manifest.get("unpacked_file_count") != len(all_files)
            or manifest.get("unpacked_bytes") != unpacked_bytes
            or manifest.get("packed_bytes") != packed_bytes):
        raise ValueError("Packed manifest totals differ from its inventory")
    task_lookup = {task["task_id"]: task for task in protocol["tasks"]}
    for record in records:
        archive_path = checked_path(root, record["path"], require_file=True)
        if describe_file(archive_path) != {key: record[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"Packed archive checksum differs: {record['path']}")
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or set(names) != set(record["files"]):
                raise ValueError("Packed member list is duplicated, incomplete, or contains extra entries")
            for entry in entries:
                safe_name(entry.filename)
                if (entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16)
                        or entry.flag_bits & 1
                        or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                        or entry.file_size != record["files"][entry.filename]["bytes"]):
                    raise ValueError("Unsafe packed entry or inconsistent member size")
                content = archive.read(entry.filename)
                if describe_bytes(content) != record["files"][entry.filename]:
                    raise ValueError(f"Packed member checksum differs: {entry.filename}")
                if check_destinations:
                    destination = checked_path(root, entry.filename)
                    if destination.exists() and (not destination.is_file()
                                                 or describe_file(destination) != record["files"][entry.filename]):
                        raise FileExistsError(f"Existing differing evidence is preserved: {entry.filename}")
            base = f"tasks/{record['task_id']}/"
            complete, failed = base + "complete.json", base + "failure.json"
            marker, other = (complete, failed) if record["outcome"] == "complete" else (failed, complete)
            if marker not in names or other in names or base + "task.json" not in names:
                raise ValueError("Packed task outcome markers are inconsistent")
            if json.loads(archive.read(base + "task.json")) != task_lookup[record["task_id"]]:
                raise ValueError("Packed task metadata differs from the protocol")
    return root, manifest


def verify(run_dir):
    root, manifest = verified_plan(run_dir)
    return {"status": "verified", "task_count": manifest["task_count"],
            "files": manifest["unpacked_file_count"], "unpacked_bytes": manifest["unpacked_bytes"],
            "packed_bytes": manifest["packed_bytes"],
            "manifest_sha256": describe_file(root / MANIFEST_NAME)["sha256"], "files_written": 0}


def unpack(run_dir):
    # Every archive, every member, and every existing destination is checked
    # before any missing evidence file is created.
    root, manifest = verified_plan(run_dir, check_destinations=True)
    restored = 0
    identical = 0
    for record in manifest["archives"]:
        archive_path = checked_path(root, record["path"], require_file=True)
        if describe_file(archive_path) != {key: record[key] for key in ("bytes", "sha256")}:
            raise ValueError("Packed archive changed during restoration")
        with zipfile.ZipFile(archive_path) as archive:
            for name, expected in record["files"].items():
                destination = checked_path(root, name)
                if destination.exists():
                    if not destination.is_file() or describe_file(destination) != expected:
                        raise FileExistsError(f"Existing differing evidence is preserved: {name}")
                    identical += 1
                    continue
                content = archive.read(name)
                if describe_bytes(content) != expected:
                    raise ValueError(f"Packed member changed during restoration: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                checked_path(root, name)
                descriptor, temporary_name = tempfile.mkstemp(prefix=".restore_", dir=destination.parent)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(content)
                    try:
                        # Creating a hard link is atomic and cannot replace an
                        # existing name. The temporary and target share a filesystem.
                        os.link(temporary, destination)
                        restored += 1
                    except FileExistsError:
                        if not destination.is_file() or describe_file(destination) != expected:
                            raise FileExistsError(f"Concurrent differing evidence is preserved: {name}")
                        identical += 1
                finally:
                    temporary.unlink(missing_ok=True)
    return {"status": "unpacked", "task_count": manifest["task_count"],
            "files_restored": restored, "existing_identical_files": identical,
            "manifest_sha256": describe_file(root / MANIFEST_NAME)["sha256"],
            "differing_files_overwritten": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack", help="archive a finished run without altering its original tasks")
    packing.add_argument("--run-dir", type=Path, default=ROOT / "results")
    packing.add_argument("--overwrite", action="store_true", help="explicitly replace differing packed derivatives only")
    for name in ("unpack", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.command == "pack":
        result = pack(args.run_dir, overwrite=args.overwrite)
    elif args.command == "unpack":
        result = unpack(args.run_dir)
    else:
        result = verify(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
