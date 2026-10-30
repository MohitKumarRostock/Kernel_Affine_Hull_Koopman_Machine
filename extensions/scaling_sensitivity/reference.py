"""Deterministic, hash-checked storage for the compact sensitivity results.

Git stores the reference ZIP as consecutive parts of at most 4 MiB.  The ZIP
contains unchanged files below results/natural and results/controlled plus one
complete per-file manifest.  It never contains full controlled predictions or
pickle model states.  Restoring validates every part, the assembled ZIP, the
complete inventory, and each payload before writing; existing different files
or unlisted results are preserved and cause a failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
PART_BYTES = 4 * 1024**2
MAX_BYTES = 2 * 1024**3
MAX_FILES = 10000
ARCHIVE_NAME = "scaling_v1.zip"
PARTS_NAME = ARCHIVE_NAME + ".parts.json"
MANIFEST_NAME = "SCALING_REFERENCE_MANIFEST.json"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
CHUNK = 1024**2


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def info(value):
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    path = Path(value)
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def read_json(value):
    return json.loads(value, object_pairs_hook=_unique_object)


def safe_name(name, *, payload=False, single=False):
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError("Unsafe reference path")
    path = PurePosixPath(name)
    if (path.is_absolute() or ":" in path.parts[0]
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe reference path: {name}")
    if single and len(path.parts) != 1:
        raise ValueError("Reference transport filenames must be simple basenames")
    if payload and (len(path.parts) < 3 or path.parts[0] != "results"
                    or path.parts[1] not in {"natural", "controlled"}):
        raise ValueError("Reference payload must be within a declared results subtree")
    return name


def regular_file(root, name):
    root = Path(root)
    safe_name(name)
    path = root
    if root.is_symlink():
        raise ValueError("Reference root is a symlink")
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink in reference path: {name}")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Missing or nonregular reference file: {name}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Reference path escapes root")
    return path


def _check_info(value):
    if (not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
            or type(value["bytes"]) is not int or not 0 <= value["bytes"] <= MAX_BYTES
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError("Invalid reference byte/hash record")


def payload_files(root):
    root = Path(root)
    results = root / "results"
    if results.is_symlink() or not results.is_dir():
        raise ValueError("Missing or symlinked results directory")
    for branch in ("natural", "controlled"):
        path = results / branch
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Missing or symlinked {branch} results")
    files = {}
    for path in sorted(results.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Symlink in result archive: {relative}")
        if path.is_dir():
            continue
        safe_name(relative, payload=True)
        source = regular_file(root, relative)
        if source.suffix not in {".json", ".npz", ".csv", ".txt", ".md"}:
            raise ValueError(f"Unexpected compact-result file type: {relative}")
        if relative.startswith("results/controlled/") and source.suffix == ".npz" and source.name != "unit_losses.npz":
            raise ValueError("Controlled public archive retains episode losses, not full prediction NPZs")
        files[relative] = source
    if not files or len(files) > MAX_FILES or sum(path.stat().st_size for path in files.values()) > MAX_BYTES:
        raise ValueError("Reference payload exceeds fixed size/count limits")
    return files


def _audit_gate(root):
    """Require a complete independent audit whose checked files are unchanged."""
    path = regular_file(root, "analysis/AUDIT.json")
    audit = read_json(path.read_bytes())
    if (audit.get("status") != "passed" or audit["natural"].get("tasks") != 208
            or audit["natural"].get("predictions_checked") != 1664
            or audit["controlled"].get("jobs") != 72
            or audit["controlled"].get("per_episode_loss_arrays_checked") != 1440):
        raise ValueError("Reference build requires the complete independent sensitivity audit")
    full = audit["controlled"].get("full_prediction_arrays_checked") == 1440
    attestation = audit.get("recorded_full_controlled_prediction_audit", {})
    if not full and attestation.get("full_prediction_arrays_verified_in_recorded_audit") != 1440:
        raise ValueError("Reference build requires full controlled prediction audit evidence")
    if digest(regular_file(root, "analysis.py")) != audit["analysis_code_sha256"]:
        raise ValueError("Analysis source differs from the complete audit")
    checked = 0
    for branch in ("natural", "controlled"):
        for relative, expected in audit[branch]["source_files_sha256"].items():
            safe_name(relative, payload=True)
            if digest(regular_file(root, relative)) != expected:
                raise ValueError("Result changed after the independent audit: " + relative)
            checked += 1
    if checked != 560:  # 208 natural JSON/NPZ pairs + 72 controlled JSON/NPZ pairs.
        raise ValueError("Independent audit does not cover the expected 560 result files")
    return {"audit_sha256": digest(path), "analysis_code_sha256": audit["analysis_code_sha256"],
            "checked_result_files": checked, "scope": "Complete post-hoc sensitivity; no new confirmation claim"}


def write_zip(path, files, provenance):
    """Write stable bytes, independent of source mtimes and filesystem modes."""
    entries = {safe_name(name, payload=True): info(value) for name, value in files.items()}
    manifest = {"format": "scaling_sensitivity_reference_v1", "files": entries,
                "provenance": provenance, "manifest_self_exclusion": MANIFEST_NAME}
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name in sorted([*files, MANIFEST_NAME]):
            data = json_bytes(manifest) if name == MANIFEST_NAME else Path(files[name]).read_bytes()
            member = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            member.compress_type = zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED
            archive.writestr(member, data, compresslevel=9)
    return manifest


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= MAX_FILES + 1:
            raise ValueError("Invalid reference ZIP entry count")
        inventory = {}
        total = 0
        for member in members:
            name = safe_name(member.filename, payload=member.filename != MANIFEST_NAME)
            if name in inventory:
                raise ValueError("Duplicate reference ZIP entry: " + name)
            mode = member.external_attr >> 16
            if member.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError("Reference ZIP must contain only regular files")
            if member.flag_bits & 1 or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Unsupported reference ZIP encryption/compression")
            total += member.file_size
            if total > MAX_BYTES:
                raise ValueError("Uncompressed reference ZIP exceeds fixed limit")
            inventory[name] = member
        if MANIFEST_NAME not in inventory or inventory[MANIFEST_NAME].file_size > 16 * 1024**2:
            raise ValueError("Missing or oversized reference ZIP manifest")
        manifest = read_json(archive.read(MANIFEST_NAME))
        if manifest.get("format") != "scaling_sensitivity_reference_v1" or manifest.get("manifest_self_exclusion") != MANIFEST_NAME:
            raise ValueError("Unsupported reference ZIP manifest")
        expected = manifest.get("files")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Reference manifest lacks a complete file inventory")
        for name, entry in expected.items():
            safe_name(name, payload=True)
            _check_info(entry)
        if set(inventory) != set(expected) | {MANIFEST_NAME}:
            raise ValueError("Reference ZIP contains missing or unlisted files")
        for name, expected_info in expected.items():
            if inventory[name].file_size != expected_info["bytes"] or info(archive.read(name)) != expected_info:
                raise ValueError("Reference payload differs from manifest: " + name)
    return {"status": "passed", "payload_files": len(expected), "archive": info(path), "manifest": manifest}


def _read_parts(manifest_path):
    manifest_path = Path(manifest_path)
    path = regular_file(manifest_path.parent, manifest_path.name)
    if path.name != PARTS_NAME or path.stat().st_size > 1024**2:
        raise ValueError("Unexpected reference-parts manifest filename or size")
    value = read_json(path.read_bytes())
    if value.get("format") != "scaling_sensitivity_reference_parts_v1" or value.get("part_size_bytes") != PART_BYTES:
        raise ValueError("Unsupported reference-parts manifest")
    archive = value.get("archive", {})
    if archive.get("filename") != ARCHIVE_NAME:
        raise ValueError("Unexpected reference ZIP filename")
    _check_info({key: archive[key] for key in ("bytes", "sha256")})
    parts = value.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_BYTES // PART_BYTES:
        raise ValueError("Invalid reference part count")
    names = set()
    for index, part in enumerate(parts, start=1):
        name = safe_name(part["filename"], single=True)
        if name != f"{ARCHIVE_NAME}.part{index:03d}" or name in names:
            raise ValueError("Duplicate or out-of-order reference part")
        names.add(name)
        _check_info({key: part[key] for key in ("bytes", "sha256")})
        size = part["bytes"]
        if not 0 < size <= PART_BYTES or (index < len(parts) and size != PART_BYTES):
            raise ValueError("Reference part sizes do not form fixed-size consecutive chunks")
    if sum(part["bytes"] for part in parts) != archive["bytes"]:
        raise ValueError("Reference part sizes do not sum to archive size")
    observed = {entry.name for entry in manifest_path.parent.glob(ARCHIVE_NAME + ".part[0-9]*")}
    if observed != names:
        raise ValueError("Reference directory contains missing or unlisted archive parts")
    return value


@contextmanager
def assembled_reference(manifest_path):
    manifest_path = Path(manifest_path)
    parts = _read_parts(manifest_path)
    with tempfile.TemporaryDirectory(prefix="scaling-reference-verify-") as temporary:
        assembled = Path(temporary) / ARCHIVE_NAME
        with assembled.open("wb") as destination:
            for entry in parts["parts"]:
                source = regular_file(manifest_path.parent, entry["filename"])
                expected = {key: entry[key] for key in ("bytes", "sha256")}
                if info(source) != expected:
                    raise ValueError("Reference part differs from manifest: " + entry["filename"])
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, destination, CHUNK)
        if info(assembled) != {key: parts["archive"][key] for key in ("bytes", "sha256")}:
            raise ValueError("Assembled reference ZIP differs from whole-archive hash")
        yield assembled, parts


def verify(manifest_path):
    with assembled_reference(manifest_path) as (archive, parts):
        verified = verify_zip(archive)
    return {"status": "passed", "parts": len(parts["parts"]), "payload_files": verified["payload_files"],
            "archive_sha256": parts["archive"]["sha256"], "archive_bytes": parts["archive"]["bytes"]}


def _copy_without_replacement(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.is_symlink():
        raise ValueError("Refusing a symlink destination")
    if destination.exists():
        if not destination.is_file() or info(destination) != info(source):
            raise FileExistsError("Refusing to replace different reference bytes: " + str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as stream, destination.open("xb") as target:
            shutil.copyfileobj(stream, target, CHUNK)


def build(root=ROOT, output=None):
    root = Path(root).resolve()
    output = root / "reference" if output is None else Path(output).absolute()
    if output.is_symlink() or output.resolve().is_relative_to(root / "results"):
        raise ValueError("Reference output must be outside the results tree without a symlink")
    provenance = _audit_gate(root)
    files = payload_files(root)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scaling-reference-build-") as temporary:
        staged = Path(temporary)
        archive = staged / ARCHIVE_NAME
        write_zip(archive, files, provenance)
        verify_zip(archive)
        specification = {"format": "scaling_sensitivity_reference_parts_v1", "part_size_bytes": PART_BYTES,
                         "archive": {"filename": ARCHIVE_NAME, **info(archive)}, "parts": []}
        with archive.open("rb") as stream:
            for index in range(1, MAX_BYTES // PART_BYTES + 1):
                data = stream.read(PART_BYTES)
                if not data:
                    break
                name = f"{ARCHIVE_NAME}.part{index:03d}"
                (staged / name).write_bytes(data)
                specification["parts"].append({"filename": name, **info(data)})
        (staged / PARTS_NAME).write_bytes(json_bytes(specification))
        verify(staged / PARTS_NAME)
        # Preflight all destinations before installing any new artifact.
        names = {PARTS_NAME} | {entry["filename"] for entry in specification["parts"]}
        old_parts = {path.name for path in output.glob(ARCHIVE_NAME + ".part[0-9]*")}
        if old_parts - names:
            raise ValueError("Unlisted existing archive parts are preserved; choose a new output directory")
        for name in names:
            existing = output / name
            if existing.is_symlink() or (existing.exists() and (not existing.is_file() or info(existing) != info(staged / name))):
                raise FileExistsError("Refusing to replace different reference bytes: " + name)
        # Detect changes to the source evidence during compression.
        _audit_gate(root)
        if set(payload_files(root)) != set(files):
            raise ValueError("Result inventory changed during reference build")
        for name, expected in verify_zip(archive)["manifest"]["files"].items():
            if info(regular_file(root, name)) != expected:
                raise ValueError("Result changed during reference build: " + name)
        for entry in specification["parts"]:
            _copy_without_replacement(staged / entry["filename"], output / entry["filename"])
        _copy_without_replacement(staged / PARTS_NAME, output / PARTS_NAME)
    return verify(output / PARTS_NAME)


def restore(manifest_path, destination=ROOT):
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise ValueError("Restore destination is a symlink")
    with assembled_reference(manifest_path) as (archive_path, parts):
        verified = verify_zip(archive_path)
        expected = verified["manifest"]["files"]
        existing_root = destination / "results"
        if existing_root.is_symlink():
            raise ValueError("Existing results root is a symlink")
        if existing_root.exists():
            if not existing_root.is_dir():
                raise ValueError("Existing results root is not a directory")
            for path in existing_root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Existing results contain a symlink")
                if path.is_dir():
                    continue
                relative = path.relative_to(destination).as_posix()
                if relative not in expected:
                    raise ValueError("Existing results contain an unlisted file: " + relative)
                if info(regular_file(destination, relative)) != expected[relative]:
                    raise FileExistsError("Existing result differs; no files overwritten: " + relative)
        for name in expected:
            target = destination / name
            for ancestor in (target, *target.parents):
                if ancestor == destination.parent:
                    break
                if ancestor.is_symlink():
                    raise ValueError("Symlink in restore destination path")
                if ancestor != target and ancestor.exists() and not ancestor.is_dir():
                    raise ValueError("Nondirectory in restore destination path")
        restored = 0
        with zipfile.ZipFile(archive_path) as archive:
            for name in sorted(expected):
                target = destination / name
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output, CHUNK)
                if info(target) != expected[name]:
                    raise ValueError("Restored file hash differs: " + name)
                restored += 1
    return {"status": "restored", "new_files": restored, "verified_payload_files": len(expected),
            "archive_sha256": parts["archive"]["sha256"], "scientific_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--output", type=Path, default=ROOT / "reference")
    for name in ("verify", "restore"):
        command = commands.add_parser(name)
        command.add_argument("manifest", nargs="?", type=Path, default=ROOT / "reference" / PARTS_NAME)
        if name == "restore":
            command.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "build":
        result = build(output=args.output)
    elif args.command == "verify":
        result = verify(args.manifest)
    else:
        result = restore(args.manifest, args.destination)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
