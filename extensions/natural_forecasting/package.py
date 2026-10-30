#!/usr/bin/env python3
"""Post-study release packaging, offline integrity checks, and analysis replay.

This module is outside the frozen forecasting implementation. Verification uses
the standard library. Audit/replay import the pinned analysis dependencies but
never fit a forecasting model. Replay writes only to an explicitly separate
directory. Building requires completed frozen results and a final manuscript.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterator, Mapping
import zipfile


EXTENSION = Path(__file__).resolve().parent
REPOSITORY = EXTENSION.parents[1]
RELATIVE_EXTENSION = Path("extensions/natural_forecasting")
MANIFEST = "PACKAGE_MANIFEST.json"
SOURCE_NAME = "KAHM_Natural_Forecasting_Manuscript_Source.zip"
REPRO_NAME = "KAHM_Natural_Forecasting_Reproducibility.zip"
HANDOFF_NAME = "KAHM_Natural_Forecasting_Handoff.zip"
PDF_NAME = "Natural_Forecasting_Manuscript.pdf"
RELEASE_NAME = "RELEASE_MANIFEST.json"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
CHUNK = 1024 * 1024
REFERENCE_PART_BYTES = 8 * 1024**2
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_ENTRIES = 30000
Payload = Path | bytes
REMEASURED_FIELDS = frozenset({
    "decision_wall_seconds", "decision_cpu_seconds", "total_policy_wall_seconds",
    "total_policy_cpu_seconds", "additional_wall_seconds", "offline_fit_wall_seconds",
    "offline_fit_cpu_seconds",
})


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_info(value: Payload) -> dict[str, Any]:
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    return {"bytes": value.stat().st_size, "sha256": file_digest(value)}


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or ":" in path.parts[0]
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return name


def source_file(root: Path, relative: str | Path) -> Path:
    name = safe_name(Path(relative).as_posix())
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Symlink or path escaping source root: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"Required file is absent: {name}")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checkpoint(repository: Path) -> dict[str, Any]:
    checkpoint = read_json(source_file(repository, RELATIVE_EXTENSION / "CHECKPOINT.json"))
    files = checkpoint["tracked_file_sha256"]
    if len(files) != 217:
        raise ValueError("Expected the preserved 217-file historical checkpoint")
    for name, expected in files.items():
        if file_digest(source_file(repository, name)) != expected:
            raise ValueError(f"Historical checkpoint file changed: {name}")
    return {"historical_files_verified": len(files), "checkpoint_commit": checkpoint["checkpoint_commit"]}


def _verify_scientific_hashes(repository: Path, freeze: Mapping[str, Any]) -> None:
    extension = repository / RELATIVE_EXTENSION
    required = {"run.py", "data.py", "models.py", "baselines.py", "analysis.py", "protocol.json", "data_sources.json", "requirements.txt"}
    if not required.issubset(freeze["code_sha256"]):
        raise ValueError("Scientific freeze omits a required source dependency")
    for name, expected in freeze["code_sha256"].items():
        # Frozen historical imports legitimately use ../predictive_geometry.
        # They must still resolve inside this repository, without symlinks.
        path = extension / name
        if path.is_symlink() or not path.resolve().is_relative_to(repository.resolve()) or not path.is_file():
            raise ValueError(f"Unsafe or missing frozen dependency: {name}")
        if file_digest(path) != expected:
            raise ValueError(f"Frozen scientific dependency changed: {name}")


def check_source(repository: Path) -> dict[str, Any]:
    result = verify_checkpoint(repository)
    extension = repository / RELATIVE_EXTENSION
    registration = extension / "registration" / "initial_freeze.json"
    if registration.is_file():
        freeze = read_json(registration)
        _verify_scientific_hashes(repository, freeze)
        result["registered_scientific_dependencies_verified"] = len(freeze["code_sha256"])
    result["status"] = "passed"
    return result


def verify_run(repository: Path, run_directory: Path, *, require_validation: bool = True) -> dict[str, Any]:
    """Byte/provenance checks only; no forecast or regression calculations."""
    result = verify_checkpoint(repository)
    extension = repository / RELATIVE_EXTENSION
    roster_path = source_file(run_directory, "task_roster.json")
    roster = read_json(roster_path)
    if roster["protocol_sha256"] != file_digest(extension / "protocol.json"):
        raise ValueError("Prepared roster protocol digest differs from source")
    if roster["source_manifest_sha256"] != file_digest(extension / "data_sources.json"):
        raise ValueError("Prepared roster source-manifest digest differs from source")
    freezes = {}
    for name in ("initial_freeze.json", "confirmation_freeze.json"):
        freeze = read_json(source_file(run_directory, name))
        _verify_scientific_hashes(repository, freeze)
        if freeze["roster_sha256"] != file_digest(roster_path):
            raise ValueError(f"Roster changed after {name}")
        freezes[name] = freeze
    final_freeze = freezes["confirmation_freeze.json"]
    if file_digest(source_file(run_directory, "frozen_diagnostic.json")) != final_freeze["diagnostic_sha256"]:
        raise ValueError("Frozen diagnostic bytes changed")
    for name, expected in final_freeze["development_result_sha256"].items():
        if file_digest(source_file(run_directory, Path("development") / name)) != expected:
            raise ValueError(f"Frozen development result changed: {name}")
    ids = [task["task_id"] for task in roster["tasks"]]
    if len(ids) != len(set(ids)) or len(ids) != 208:
        raise ValueError("The final release must contain the complete 208-task roster")
    expected_by_phase = {phase: {task["task_id"] for task in roster["tasks"] if task["role"] == phase}
                         for phase in ("development", "confirmation")}
    if {phase: len(ids) for phase, ids in expected_by_phase.items()} != {"development": 128, "confirmation": 80}:
        raise ValueError("Unexpected development/confirmation task allocation")
    checked_caches: set[str] = set()
    for task in roster["tasks"]:
        if task["cache_path"] not in checked_caches:
            cache = source_file(run_directory, task["cache_path"])
            if file_digest(cache) != task["cache_sha256"]:
                raise ValueError(f"Prepared-array checksum mismatch: {task['task_id']}")
            checked_caches.add(task["cache_path"])
    methods = set(read_json(extension / "protocol.json")["methods"])
    records = predictions = 0
    for phase, expected in expected_by_phase.items():
        observed: set[str] = set()
        for path in sorted((run_directory / phase).glob("*.json")):
            row = read_json(path)
            identifier = row["task_id"]
            if identifier in observed or identifier not in expected or row.get("phase") != phase:
                raise ValueError(f"Duplicate/unexpected phase record: {path.name}")
            observed.add(identifier)
            if row.get("status") != "complete" or row.get("failures"):
                raise ValueError(f"Incomplete result cannot be released: {identifier}")
            candidates = row["candidates"]
            if len(candidates) != len(methods) or {c["candidate_id"] for c in candidates} != methods:
                raise ValueError(f"Incomplete candidate portfolio: {identifier}")
            if any(candidate.get("status") != "ok" for candidate in candidates):
                raise ValueError(f"A failed candidate remains in result: {identifier}")
            artifact = source_file(run_directory / phase, row["predictions_file"])
            if file_digest(artifact) != row["predictions_sha256"]:
                raise ValueError(f"Prediction artifact changed: {identifier}")
            records += 1
            predictions += len(candidates)
        if observed != expected:
            raise ValueError(f"Incomplete {phase} roster: {len(observed)}/{len(expected)}")
    for name in ("development_analysis.json", "confirmation_analysis.json"):
        source_file(run_directory, name)
    if require_validation:
        validation = read_json(source_file(run_directory, "VALIDATION.json"))
        if (validation.get("status") != "passed" or validation.get("predictions_recomputed") != predictions
                or validation.get("historical_tracked_files_unchanged") != 217
                or validation.get("frozen_protocol_intact") is not True):
            raise ValueError("Final computational validation is not passed")
    result.update(status="passed", task_records=records, candidate_predictions=predictions,
                  prepared_array_files=len(checked_caches),
                  confirmation_freeze_sha256=file_digest(run_directory / "confirmation_freeze.json"),
                  information_scope="byte/provenance audit; no model fitting or forecast evaluation")
    return result


def run_payload(run_directory: Path) -> dict[str, Payload]:
    files: dict[str, Payload] = {}
    for path in sorted(run_directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_directory)
        if any(part in {"__pycache__", "tmp", "build"} for part in relative.parts):
            continue
        if path.name.endswith((".tmp", ".part", ".pyc", ".pyo")) or relative.as_posix() == MANIFEST:
            raise ValueError(f"Unfinished or reserved run artifact is present: {relative}")
        files[relative.as_posix()] = source_file(run_directory, relative)
    return files


def reference_parts_path(archive_path: Path) -> Path:
    return archive_path.with_name(archive_path.name + ".parts.json")


def _read_reference_parts(manifest_path: Path) -> dict[str, Any]:
    """Validate an ordered, bounded inventory before opening any part path."""
    manifest_path = source_file(manifest_path.parent, manifest_path.name)
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("Reference-parts manifest is too large")
    specification = read_json(manifest_path)
    if (not isinstance(specification, dict) or specification.get("schema_version") != 1
            or specification.get("kind") != "reference-run-parts"
            or specification.get("part_size_bytes") != REFERENCE_PART_BYTES):
        raise ValueError("Unsupported reference-parts manifest")
    archive = specification.get("archive", {})
    if not isinstance(archive, dict):
        raise ValueError("Invalid reference archive metadata")
    archive_name = archive.get("filename", "")
    if not isinstance(archive_name, str):
        raise ValueError("Unsafe reference archive filename")
    safe_name(archive_name)
    if ("/" in archive_name or not archive_name.endswith(".zip")
            or manifest_path.name != archive_name + ".parts.json"):
        raise ValueError("Unsafe or inconsistent reference archive filename")
    parts = specification.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_UNCOMPRESSED_BYTES // REFERENCE_PART_BYTES:
        raise ValueError("Invalid reference part count")
    total = 0
    for index, entry in enumerate(parts, start=1):
        if not isinstance(entry, dict):
            raise ValueError("Invalid reference part entry")
        name = entry.get("filename", "")
        if not isinstance(name, str):
            raise ValueError("Unsafe reference part filename")
        safe_name(name)
        if name != f"{archive_name}.part-{index:03d}":
            raise ValueError("Unsafe, duplicate, or out-of-order reference part filename")
        size = entry.get("bytes")
        if type(size) is not int or not 1 <= size <= REFERENCE_PART_BYTES:
            raise ValueError("Invalid reference part length")
        if index < len(parts) and size != REFERENCE_PART_BYTES:
            raise ValueError("Only the last reference part may be shorter than 8 MiB")
        total += size
    if type(archive.get("bytes")) is not int or not 0 < archive["bytes"] <= MAX_UNCOMPRESSED_BYTES or total != archive["bytes"]:
        raise ValueError("Reference part lengths differ from the archive length")
    for entry in [archive, *parts]:
        digest = entry.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid reference SHA-256 digest")
    return specification


def _write_reference_parts(archive_path: Path, directory: Path) -> tuple[Path, dict[str, Any]]:
    """Split the exact ZIP bytes; parts are transport units, not new archives."""
    archive_name = archive_path.name
    specification = {
        "schema_version": 1, "kind": "reference-run-parts", "part_size_bytes": REFERENCE_PART_BYTES,
        "archive": {"filename": archive_name, **payload_info(archive_path)}, "parts": [],
    }
    with archive_path.open("rb") as source:
        index = 1
        for block in iter(lambda: source.read(REFERENCE_PART_BYTES), b""):
            name = f"{archive_name}.part-{index:03d}"
            with (directory / name).open("xb") as destination:
                destination.write(block)
            specification["parts"].append({"filename": name, **payload_info(block)})
            index += 1
    manifest_path = directory / (archive_name + ".parts.json")
    manifest_path.write_bytes(json_bytes(specification))
    _read_reference_parts(manifest_path)
    return manifest_path, specification


@contextmanager
def reference_archive_view(path: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Use an existing ZIP or temporarily assemble verified multipart bytes."""
    if path.is_file() and not path.name.endswith(".parts.json"):
        yield path, {}
        return
    manifest_path = path if path.name.endswith(".parts.json") else reference_parts_path(path)
    specification = _read_reference_parts(manifest_path)
    with tempfile.TemporaryDirectory(prefix="natural-reference-assemble-") as temporary:
        assembled = Path(temporary) / specification["archive"]["filename"]
        complete_digest = hashlib.sha256()
        complete_size = 0
        with assembled.open("xb") as destination:
            for entry in specification["parts"]:
                part = source_file(manifest_path.parent, entry["filename"])
                if part.stat().st_size != entry["bytes"]:
                    raise ValueError(f"Reference part payload changed: {part.name}")
                part_digest = hashlib.sha256()
                part_size = 0
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        part_size += len(block)
                        if part_size > entry["bytes"]:
                            raise ValueError(f"Reference part payload changed while reading: {part.name}")
                        part_digest.update(block)
                        complete_digest.update(block)
                        destination.write(block)
                if part_size != entry["bytes"] or part_digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Reference part payload changed: {part.name}")
                complete_size += part_size
        if (complete_size != specification["archive"]["bytes"]
                or complete_digest.hexdigest() != specification["archive"]["sha256"]):
            raise ValueError("Assembled reference archive differs from its manifest")
        yield assembled, {"multipart_parts_verified": len(specification["parts"]),
                          "multipart_manifest_sha256": file_digest(manifest_path)}


def verify_reference_parts(path: Path) -> dict[str, Any]:
    with reference_archive_view(path) as (archive_path, part_verification):
        result = verify_zip(archive_path)
        if result["kind"] != "reference-run" or result["root"] != "natural_v1":
            raise ValueError("Multipart storage must contain a natural_v1 reference-run archive")
        result.update(part_verification)
        return result


def build_reference(repository: Path, run_directory: Path, output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Create a run ZIP plus bounded Git-storable parts, without source duplication."""
    verification = verify_run(repository, run_directory)
    if output.resolve().is_relative_to(run_directory.resolve()):
        raise ValueError("Reference archive output cannot be inside the run it archives")
    if output.suffix != ".zip":
        raise ValueError("Reference archive output must have a .zip suffix")
    parts_manifest = reference_parts_path(output)
    if (output.exists() or parts_manifest.exists()) and not overwrite:
        raise FileExistsError("Reference archive already exists; use explicit --overwrite to replace it")
    old_parts = {}
    if parts_manifest.exists():
        old_parts = {entry["filename"]: entry for entry in _read_reference_parts(parts_manifest)["parts"]}
        for name, entry in old_parts.items():
            path = output.parent / name
            if path.exists() or path.is_symlink():
                if payload_info(source_file(output.parent, name)) != {key: entry[key] for key in ("bytes", "sha256")}:
                    raise ValueError(f"Existing reference part changed; preserve it for inspection: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = run_payload(run_directory)
    if any(Path(name).name in {"main.pdf", PDF_NAME} for name in files):
        raise ValueError("The manuscript PDF belongs in the release, not the Git-backed reference run")
    with tempfile.TemporaryDirectory(prefix="reference-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        write_zip(staged, "natural_v1", files, kind="reference-run", provenance=verification)
        staged_manifest, specification = _write_reference_parts(staged, Path(temporary))
        verify_reference_parts(staged_manifest)
        for entry in specification["parts"]:
            destination = output.parent / entry["filename"]
            if (destination.exists() or destination.is_symlink()) and entry["filename"] not in old_parts:
                raise FileExistsError(f"Refusing to replace a part not owned by the previous manifest: {destination.name}")
        verify_run(repository, run_directory)
        for entry in specification["parts"]:
            (Path(temporary) / entry["filename"]).replace(output.parent / entry["filename"])
        staged_manifest.replace(parts_manifest)
        staged.replace(output)
        new_names = {entry["filename"] for entry in specification["parts"]}
        for name in set(old_parts) - new_names:
            path = output.parent / name
            if path.exists():
                expected = {key: old_parts[name][key] for key in ("bytes", "sha256")}
                if payload_info(source_file(output.parent, name)) != expected:
                    raise ValueError(f"Stale owned part changed during build; it was not removed: {name}")
                path.unlink()
    return {"status": "complete", "reference_archive": output.name, **payload_info(output),
            "multipart_manifest": {"filename": parts_manifest.name, **payload_info(parts_manifest)},
            "git_parts": specification["parts"],
            "payload_files": len(files), "forecasting_fits": 0,
            "scope": "Completed run only. Git stores parts plus manifest; the full ZIP stays local. Full release excludes reference storage and includes the expanded run instead."}


def restore_reference(archive_path: Path, repository: Path) -> dict[str, Any]:
    with reference_archive_view(archive_path) as (path, part_verification):
        result = _restore_reference_zip(path, repository)
        result.update(part_verification)
        return result


def _restore_reference_zip(archive_path: Path, repository: Path) -> dict[str, Any]:
    verification = verify_zip(archive_path)
    if verification["kind"] != "reference-run" or verification["root"] != "natural_v1":
        raise ValueError("Expected a natural_v1 reference-run archive")
    target = repository / RELATIVE_EXTENSION / "runs/natural_v1"
    if target.exists():
        raise FileExistsError("The reference run destination already exists; restoration never overwrites evidence")
    with zipfile.ZipFile(archive_path) as archive:
        for name in ("initial_freeze.json", "confirmation_freeze.json"):
            freeze = json.loads(archive.read(f"natural_v1/{name}"))
            _verify_scientific_hashes(repository, freeze)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="reference-restore-", dir=target.parent) as temporary:
            staged_run = Path(temporary) / "natural_v1"
            staged_run.mkdir()
            for info in archive.infolist():
                relative = info.filename.split("/", 1)[1]
                if relative == MANIFEST:
                    continue  # Package metadata is not part of the original run.
                destination = staged_run / safe_name(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, CHUNK)
            checked = verify_run(repository, staged_run)
            if target.exists():
                raise FileExistsError("Reference destination appeared while restoring; no files overwritten")
            staged_run.replace(target)
    return {"status": "restored", "run_directory": (RELATIVE_EXTENSION / "runs/natural_v1").as_posix(),
            "reference_zip_sha256": verification["zip_sha256"], "audit": checked,
            "forecasting_fits": 0, "scientific_source_modified": False}


def manuscript_payload(extension: Path) -> dict[str, Payload]:
    root = extension / "manuscript"
    files: dict[str, Payload] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {"build", "tmp", "__pycache__"} or part.startswith(".") for part in relative.parts):
            continue
        if path.name in {"main.pdf", "JMLR_sample_reference.tex"}:
            continue
        allowed = path.suffix in {".tex", ".bib", ".sty", ".bst", ".bbl", ".md", ".txt", ".json"}
        allowed |= "generated" in relative.parts and path.suffix in {".pdf", ".png", ".svg", ".eps", ".csv", ".npz"}
        if allowed:
            files[relative.as_posix()] = source_file(root, relative)
    for required in ("main.tex", "references.bib", "jmlr2e.sty", "TEMPLATE_PROVENANCE.json"):
        if required not in files:
            raise FileNotFoundError(f"Missing manuscript source: {required}")
    files["README_ARCHIVE.md"] = SOURCE_README.encode("utf-8")
    return files


def reproducibility_payload(repository: Path, run_directory: Path, extra_logs: list[Path]) -> dict[str, Payload]:
    extension = repository / RELATIVE_EXTENSION
    checkpoint = read_json(extension / "CHECKPOINT.json")
    files: dict[str, Payload] = {name: source_file(repository, name) for name in checkpoint["tracked_file_sha256"]}
    excluded_directories = {"runs", "build", "tmp", "dist", "__pycache__", ".git", ".pytest_cache", ".venv"}
    excluded_suffixes = {".pyc", ".pyo", ".aux", ".out", ".log", ".blg", ".fls", ".fdb_latexmk", ".toc"}
    for path in sorted(extension.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(extension)
        if any(part in excluded_directories for part in relative.parts) or path.name in {".DS_Store", "JMLR_sample_reference.tex"}:
            continue
        if relative.parts[0] == "reference":
            continue  # Expanded run data replace every reference ZIP/part/intermediate.
        if relative.parts[0] == "archive" and path.suffix in {".zip", ".gz", ".tar"}:
            continue  # Never recursively include a prior release.
        if path.suffix in excluded_suffixes or path.name.endswith((".synctex.gz", ".tmp", ".part")):
            continue
        files[(RELATIVE_EXTENSION / relative).as_posix()] = source_file(extension, relative)
    workflow = ".github/workflows/natural-forecasting.yml"
    files[workflow] = source_file(repository, workflow)
    # Only this declared completed run is included, never other ignored runs.
    archive_run = RELATIVE_EXTENSION / "runs" / "natural_v1"
    for name, path in run_payload(run_directory).items():
        files[(archive_run / name).as_posix()] = path
    for path in extra_logs:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Extra log must be a regular file: {path.name}")
        name = (archive_run / "external_logs" / path.name).as_posix()
        if name in files:
            raise ValueError(f"Duplicate extra-log destination: {path.name}")
        files[name] = path
    manifest = read_json(extension / "data_sources.json")
    for source in manifest["sources"]:
        raw = source_file(extension, Path("datasets/raw") / source["archive"]["filename"])
        if payload_info(raw) != {"bytes": source["archive"]["bytes"], "sha256": source["archive"]["sha256"]}:
            raise ValueError(f"Original data archive changed: {source['source_id']}")
        metadata = source_file(extension, source["zenodo"]["metadata_file"])
        if file_digest(metadata) != source["zenodo"]["metadata_sha256"]:
            raise ValueError(f"Official source metadata changed: {source['source_id']}")
    files["README_ARCHIVE.md"] = REPRO_README.encode("utf-8")
    return files


def write_zip(path: Path, root_name: str, files: Mapping[str, Payload], *, kind: str,
              provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    safe_name(root_name)
    if MANIFEST in files:
        raise ValueError("Reserved package manifest name")
    inventory = {safe_name(name): payload_info(value) for name, value in sorted(files.items())}
    manifest = {"schema_version": 1, "kind": kind, "root": root_name,
                "files": inventory, "provenance": dict(provenance or {}),
                "manifest_self_exclusion": "This manifest is the only unhashed internal entry; the outer release hashes the entire ZIP, including this manifest.",
                "integrity_scope": "Checksums establish consistency, not external authenticity or scientific validity."}
    payloads = dict(files)
    payloads[MANIFEST] = json_bytes(manifest)
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name, value in sorted(payloads.items()):
            info = zipfile.ZipInfo(f"{root_name}/{safe_name(name)}", date_time=ZIP_DATE)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_STORED if Path(name).suffix in {".zip", ".npz", ".pdf", ".png", ".gz"} else zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as destination:
                if isinstance(value, bytes):
                    destination.write(value)
                else:
                    with value.open("rb") as source:
                        shutil.copyfileobj(source, destination, CHUNK)
    verify_zip(path)
    return manifest


def _zip_inventory(archive: zipfile.ZipFile) -> tuple[str, dict[str, zipfile.ZipInfo]]:
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES or sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Archive exceeds the declared verification resource bound")
    names: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = safe_name(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        if name in names or info.is_dir() or mode not in {0, stat.S_IFREG} or info.flag_bits & 1:
            raise ValueError(f"Duplicate, nonregular, directory, or encrypted entry: {name}")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError(f"Unsupported ZIP compression: {name}")
        names[name] = info
    manifests = [name for name in names if name.endswith("/" + MANIFEST) and name.count("/") == 1]
    if len(manifests) != 1:
        raise ValueError("Expected exactly one top-level package manifest")
    return manifests[0].split("/", 1)[0], names


def verify_zip(path: Path) -> dict[str, Any]:
    """Stream checksums without extracting or executing archive contents."""
    with zipfile.ZipFile(path) as archive:
        root, names = _zip_inventory(archive)
        manifest_name = f"{root}/{MANIFEST}"
        if names[manifest_name].file_size > 20 * 1024**2:
            raise ValueError("Unexpectedly large package manifest")
        manifest = json.loads(archive.read(manifest_name))
        if manifest.get("schema_version") != 1 or manifest["root"] != root:
            raise ValueError("Invalid package-manifest schema/root")
        expected = {f"{root}/{safe_name(name)}" for name in manifest["files"]} | {manifest_name}
        if set(names) != expected:
            raise ValueError("Archive entries differ from the complete manifest inventory")
        for name, information in manifest["files"].items():
            member = f"{root}/{name}"
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    digest.update(block)
            if names[member].file_size != information["bytes"] or digest.hexdigest() != information["sha256"]:
                raise ValueError(f"Archive payload changed: {name}")
    return {"status": "passed", "kind": manifest["kind"], "payload_files_verified": len(manifest["files"]),
            "zip_sha256": file_digest(path), "root": root, "manifest": manifest}


def verify_directory(directory: Path) -> dict[str, Any]:
    manifest = read_json(source_file(directory, MANIFEST))
    if manifest.get("schema_version") != 1:
        raise ValueError("Invalid package-manifest schema")
    expected = set(manifest["files"]) | {MANIFEST}
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Extracted archive contains a symlink")
    actual = {path.relative_to(directory).as_posix() for path in paths if path.is_file()}
    if expected != actual:
        raise ValueError(f"Extracted inventory differs; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    for name, information in manifest["files"].items():
        if payload_info(source_file(directory, name)) != information:
            raise ValueError(f"Extracted payload changed: {name}")
    return {"status": "passed", "kind": manifest["kind"], "payload_files_verified": len(manifest["files"]),
            "root": manifest["root"], "manifest": manifest}


def compare_content(saved: Any, recomputed: Any, *, ignored_fields: frozenset[str] = frozenset(),
                    path: str = "report") -> None:
    """Compare reproducible report content, with explicit timing exceptions."""
    if isinstance(saved, dict) and isinstance(recomputed, dict):
        left = {key: value for key, value in saved.items() if key not in ignored_fields}
        right = {key: value for key, value in recomputed.items() if key not in ignored_fields}
        if set(left) != set(right):
            raise ValueError(f"Archived report keys differ at {path}")
        for key in left:
            compare_content(left[key], right[key], ignored_fields=ignored_fields, path=f"{path}.{key}")
    elif isinstance(saved, list) and isinstance(recomputed, list):
        if len(saved) != len(recomputed):
            raise ValueError(f"Archived report list length differs at {path}")
        for index, (left, right) in enumerate(zip(saved, recomputed)):
            compare_content(left, right, ignored_fields=ignored_fields, path=f"{path}[{index}]")
    elif isinstance(saved, (int, float)) and not isinstance(saved, bool) and isinstance(recomputed, (int, float)) and not isinstance(recomputed, bool):
        if not math.isclose(float(saved), float(recomputed), rel_tol=1e-9, abs_tol=1e-11):
            raise ValueError(f"Archived report numeric value differs at {path}: {saved!r} vs {recomputed!r}")
    elif type(saved) is not type(recomputed) or saved != recomputed:
        raise ValueError(f"Archived report value differs at {path}")


def audit_analysis_report(saved: Mapping[str, Any], recomputed: Mapping[str, Any],
                          analysis: Any, configuration: Mapping[str, Any]) -> None:
    compare_content(saved, recomputed, ignored_fields=REMEASURED_FIELDS)
    rows = saved["task_policy_rows"]
    for row in rows:
        for clock in ("wall", "cpu"):
            decision = row[f"decision_{clock}_seconds"]
            measured = row[f"measured_node_{clock}_seconds"]
            total = row[f"total_policy_{clock}_seconds"]
            if isinstance(decision, bool) or not isinstance(decision, (int, float)) or not math.isfinite(decision) or decision < 0:
                raise ValueError("Archived decision timing is invalid")
            expected = None if measured is None else measured + decision
            compare_content(total, expected, path=f"{row['task_id']}.{row['policy']}.{clock}_total")
    # Original measured decision times cannot be recreated. Their contribution
    # to every archived aggregate can still be independently checked exactly.
    compare_content(saved["policies"], analysis.summarize_policies(rows), path="archived_timing_and_metric_summaries")
    compare_content(saved["calibration"], analysis.calibration(rows, configuration), path="archived_calibration")
    compare_content(saved["paired_policy_comparisons"], analysis.paired_policy_comparisons(rows), path="archived_paired_comparisons")


@contextmanager
def repository_view(path: Path) -> Iterator[Path]:
    """Extract only after complete validation; source archives remain unchanged."""
    if path.is_file():
        verification = verify_zip(path)
        if verification["kind"] != "reproducibility":
            raise ValueError("Audit/replay requires the reproducibility archive, not manuscript/handoff ZIP")
        with tempfile.TemporaryDirectory(prefix="natural-forecasting-audit-") as temporary:
            temporary_root = Path(temporary)
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    target = temporary_root / safe_name(info.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, CHUNK)
            yield temporary_root / verification["root"]
    else:
        if (path / MANIFEST).is_file():
            verification = verify_directory(path)
            if verification["kind"] != "reproducibility":
                raise ValueError("Audit/replay requires the reproducibility source tree")
        if not (path / RELATIVE_EXTENSION / "CHECKPOINT.json").is_file():
            raise ValueError("Expected the repository root or extracted reproducibility root")
        yield path.resolve()


def audit_repository(repository: Path, *, replay_output: Path | None = None) -> dict[str, Any]:
    """Read-only scientific checks, optionally writing analysis replay elsewhere."""
    run_directory = repository / RELATIVE_EXTENSION / "runs/natural_v1"
    verification = verify_run(repository, run_directory)
    if replay_output is not None:
        replay_output = replay_output.resolve()
        if replay_output.is_relative_to(repository.resolve()):
            raise ValueError("Replay output must be outside the preserved source/archive tree")
        replay_output.mkdir(parents=True, exist_ok=True)
        if any(replay_output.iterdir()):
            raise ValueError("Replay output must be empty; archived evidence is never overwritten")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    sys.dont_write_bytecode = True
    # A fresh CLI process is required so imports cannot refer to another checkout.
    if any(name.startswith("extensions.natural_forecasting.run") or name.startswith("extensions.natural_forecasting.analysis") for name in sys.modules):
        raise RuntimeError("Run audit/replay in a fresh Python CLI process")
    # `python -m extensions.natural_forecasting.package` has already imported
    # namespace parents from the caller's checkout. They must not redirect an
    # audit of a different extracted source tree back to that caller's files.
    for name in list(sys.modules):
        if name == "extensions" or name.startswith("extensions."):
            del sys.modules[name]
    sys.path.insert(0, str(repository))
    runner = importlib.import_module("extensions.natural_forecasting.run")
    analysis = importlib.import_module("extensions.natural_forecasting.analysis")
    numeric = runner.audit_results(run_directory)
    roster = read_json(run_directory / "task_roster.json")
    configuration = runner.analysis_configuration(roster)
    development = runner.load_results(run_directory, "development")
    confirmation = runner.load_results(run_directory, "confirmation")
    analysis.validate_tasks(development, configuration, role="development")
    analysis.validate_tasks(confirmation, configuration, role="confirmation", allow_confirmation=True)
    frozen = read_json(run_directory / "frozen_diagnostic.json")
    confirmation_report = analysis.evaluate_frozen(confirmation, frozen, allow_confirmation=True)
    development_report = analysis.analyze_development(development, config=configuration)
    for phase, report in (("development", development_report), ("confirmation", confirmation_report)):
        audit_analysis_report(read_json(run_directory / f"{phase}_analysis.json"), report, analysis, configuration)
    for name, features in analysis.MODEL_FEATURES.items():
        fitted = analysis.fit_diagnostic(development, features, configuration)
        compare_content(frozen["models"][name], fitted, ignored_fields=REMEASURED_FIELDS, path=f"frozen_models.{name}")
    if frozen["development_input_sha256"] != analysis._digest(sorted(development, key=lambda task: task["task_id"])):
        raise ValueError("Diagnostic development-input digest differs from saved records")
    verification.update(numeric_audit=numeric, analysis_validation="passed", forecasting_fits=0,
                        source_modified=False,
                        information_scope="Frozen bytes, saved forecast errors, diagnostic coefficients, and archived scientific reports; zero forecasting-model or geometry-bank fits.",
                        archived_scientific_reports_recomputed=True,
                        frozen_diagnostic_coefficients_recomputed=True,
                        timing_comparison_scope="New decision/offline-fit timings excluded from cross-replay equality; archived total/summarized timing arithmetic is checked using its own original row timings.",
                        runtime_note="Offline audit checks frozen bytes and numeric evidence; it does not require byte-identical Python build strings or recreate timing measurements.")
    if replay_output is not None:
        verification["replay_scope"] = "Saved forecast errors and frozen confirmation policies; development replay refits only small diagnostic ridge regressions. New diagnostic/decision timings need not equal archived timings."
        for name, value in (("development_analysis_replay.json", development_report),
                            ("confirmation_analysis_replay.json", confirmation_report),
                            ("REPLAY_VALIDATION.json", verification)):
            (replay_output / name).write_bytes(json_bytes(value))
    return verification


def build(repository: Path, run_directory: Path, output: Path, *, handoff: bool = False,
          extra_logs: list[Path] | None = None, overwrite: bool = False,
          manuscript_pdf: Path | None = None) -> dict[str, Any]:
    verification = verify_run(repository, run_directory)
    extension = repository / RELATIVE_EXTENSION
    pdf = manuscript_pdf if manuscript_pdf is not None else source_file(extension, "manuscript/main.pdf")
    if pdf.is_symlink() or not pdf.is_file():
        raise ValueError("The final manuscript must be a regular PDF file")
    pdf = pdf.resolve()
    with pdf.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("The final manuscript is not a PDF")
    sources = manuscript_payload(extension)
    input_mtimes = [value.stat().st_mtime for name, value in sources.items()
                    if isinstance(value, Path) and Path(name).suffix in {".tex", ".bib", ".sty", ".pdf", ".png"}]
    if input_mtimes and max(input_mtimes) > pdf.stat().st_mtime + 1:
        raise ValueError("Manuscript inputs are newer than main.pdf; rebuild/review before packaging")
    reproduction = reproducibility_payload(repository, run_directory, extra_logs or [])
    reproduction[(RELATIVE_EXTENSION / "manuscript/main.pdf").as_posix()] = pdf
    output.mkdir(parents=True, exist_ok=True)
    names = [SOURCE_NAME, REPRO_NAME, PDF_NAME, RELEASE_NAME, "README_HANDOFF.md"]
    if handoff:
        names.append(HANDOFF_NAME)
    conflicts = [name for name in names if (output / name).exists()
                 and not (name == PDF_NAME and (output / name).resolve() == pdf)]
    if not overwrite and conflicts:
        raise FileExistsError("A named release output already exists; use a new directory or explicit --overwrite")
    with tempfile.TemporaryDirectory(prefix="natural-release-", dir=output) as temporary:
        staging = Path(temporary)
        write_zip(staging / SOURCE_NAME, "KAHM_Natural_Forecasting_Manuscript_Source", sources,
                  kind="manuscript-source", provenance=verification)
        write_zip(staging / REPRO_NAME, "KAHM_Natural_Forecasting_Reproducibility", reproduction,
                  kind="reproducibility", provenance=verification)
        shutil.copyfile(pdf, staging / PDF_NAME)
        (staging / "README_HANDOFF.md").write_text(HANDOFF_README, encoding="utf-8")
        release = {"schema_version": 1, "status": "complete",
                   "payloads": {name: payload_info(staging / name) for name in (SOURCE_NAME, REPRO_NAME, PDF_NAME, "README_HANDOFF.md")},
                   "historical_and_frozen_run_audit": verification,
                   "scope": "Post-study packaging; no forecasting code or saved scientific evidence rewritten."}
        (staging / RELEASE_NAME).write_bytes(json_bytes(release))
        if handoff:
            members = {name: staging / name for name in names if name != HANDOFF_NAME}
            write_zip(staging / HANDOFF_NAME, "KAHM_Natural_Forecasting_Handoff", members, kind="handoff")
        # Recheck scientific provenance before publishing staged artifacts.
        verify_run(repository, run_directory)
        for name in names:
            (staging / name).replace(output / name)
    return {"status": "complete", "outputs": {name: payload_info(output / name) for name in names},
            "historical_files_included": 217, "forecasting_fits": 0}


SOURCE_README = """# Manuscript source archive

Install TeX Live, latexmk, and the lmodern fonts. From this extracted manuscript
directory, build `main.tex` with the bundled official JMLR style and bibliography:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Generated figures and tables contain the archived study outputs. Building the
paper does not fit models. TEMPLATE_PROVENANCE.json identifies the style source.
PACKAGE_MANIFEST.json hashes every payload file; the manifest's self-exclusion
is explicit, and RELEASE_MANIFEST.json outside the ZIP hashes the complete ZIP.
Verify the archive before rebuilding, since a build creates auxiliary files.
"""

REPRO_README = """# Natural forecasting reproducibility archive

This extracted directory is a complete source-checkout view. It preserves all
217 historical checkpoint files, the new extension, official data snapshots,
the frozen task arrays, both phases of predictions/results, and analysis records.
No repository checkout or dataset download is needed after extraction.

From this directory, first verify bytes with the Python standard library:

    python extensions/natural_forecasting/package.py verify .

Install the extension's pinned dependencies in an environment outside this
preserved directory, then audit and replay without forecasting-model fits:

    python extensions/natural_forecasting/package.py audit .
    python extensions/natural_forecasting/package.py replay . --output ../analysis-replay

Replay output must be outside this source tree and initially empty. It recomputes
statistics from saved predictions and refits only the tiny development diagnostic
regressions. It never refits forecasting models or geometry banks. Timing values
for replayed scalar decisions/regressions can differ; archived measurements remain
unchanged. The archive does not execute code during checksum verification.

Read extensions/natural_forecasting/README.md and PROTOCOL.md for full independent
computational reproduction, cost scopes, and scientific limitations. The historical
root README and older manuscript are preserved evidence, not the entry point for
this new experiment. Reproduce expensive fits only in a separate fresh run path.
"""

HANDOFF_README = """# Natural forecasting study handoff

Natural_Forecasting_Manuscript.pdf is the paper. The Manuscript_Source ZIP
contains editable LaTeX, bibliography, official style, and generated assets.
The Reproducibility ZIP contains the full historical source closure, new code,
eight public-source snapshots, frozen data windows, saved predictions, and reports.

RELEASE_MANIFEST.json hashes these payloads. Each ZIP also carries an internal
manifest of every payload entry, with a declared self-exclusion for that manifest.
The optional Handoff ZIP binds the complete set in its own payload manifest.

Start with README_ARCHIVE.md inside the reproducibility archive. Checksum
verification needs only Python's standard library. Prediction auditing and analysis
replay require the pinned scientific environment and do not repeat expensive fits.
The study's limited source collection and measured cost scopes remain qualifications
on its findings; packaging is not additional scientific evidence.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("check-source", help="Verify the historical checkpoint and public scientific registration")
    source.add_argument("--repository", type=Path, default=REPOSITORY)
    reference = subparsers.add_parser("build-reference", help="Build the completed-run ZIP plus 8 MiB Git-storage parts")
    reference.add_argument("--repository", type=Path, default=REPOSITORY)
    reference.add_argument("--run-dir", type=Path)
    reference.add_argument("--output", type=Path)
    reference.add_argument("--overwrite", action="store_true")
    restore = subparsers.add_parser("restore-reference", help="Verify and restore the reference run without overwriting an existing run")
    restore.add_argument("path", type=Path, help="Reference ZIP or .zip.parts.json; absent ZIP is reconstructed from sibling parts")
    restore.add_argument("--repository", type=Path, default=REPOSITORY)
    builder = subparsers.add_parser("build", help="Package only a final complete study; no model fitting")
    builder.add_argument("--repository", type=Path, default=REPOSITORY)
    builder.add_argument("--run-dir", type=Path)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--manuscript-pdf", type=Path, help="Final reviewed PDF if not manuscript/main.pdf")
    builder.add_argument("--handoff", action="store_true")
    builder.add_argument("--extra-log", type=Path, action="append", default=[])
    builder.add_argument("--overwrite", action="store_true")
    for name in ("verify", "audit", "replay"):
        command = subparsers.add_parser(name)
        command.add_argument("path", type=Path, help="ZIP, extracted reproducibility root, or source repository (audit/replay)")
        if name == "replay":
            command.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "check-source":
        result = check_source(arguments.repository.resolve())
    elif arguments.command == "build-reference":
        repository = arguments.repository.resolve()
        run_directory = (arguments.run_dir or repository / RELATIVE_EXTENSION / "runs/natural_v1").resolve()
        output = (arguments.output or repository / RELATIVE_EXTENSION / "reference/natural_v1.zip").resolve()
        result = build_reference(repository, run_directory, output, overwrite=arguments.overwrite)
    elif arguments.command == "restore-reference":
        result = restore_reference(arguments.path.resolve(), arguments.repository.resolve())
    elif arguments.command == "build":
        repository = arguments.repository.resolve()
        run_directory = (arguments.run_dir or repository / RELATIVE_EXTENSION / "runs/natural_v1").resolve()
        result = build(repository, run_directory, arguments.output.resolve(), handoff=arguments.handoff,
                       extra_logs=arguments.extra_log, overwrite=arguments.overwrite,
                       manuscript_pdf=arguments.manuscript_pdf)
    elif arguments.command == "verify":
        if arguments.path.is_dir():
            result = verify_directory(arguments.path)
        elif arguments.path.name.endswith(".parts.json") or not arguments.path.is_file():
            result = verify_reference_parts(arguments.path)
        else:
            result = verify_zip(arguments.path)
        result.pop("manifest", None)
    else:
        with repository_view(arguments.path.resolve()) as repository:
            result = audit_repository(repository, replay_output=arguments.output if arguments.command == "replay" else None)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
