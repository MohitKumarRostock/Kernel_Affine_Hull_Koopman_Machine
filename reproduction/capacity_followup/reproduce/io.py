"""Hash-checked, byte-preserving handling of the compact historical archive."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bytes_from(path):
    path = Path(path)
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def read_json(path):
    return json.loads(bytes_from(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_archive(root=ROOT):
    root = Path(root)
    snapshot = read_json(root / "study/SNAPSHOT.json")
    for logical_name, entry in snapshot["files"].items():
        data = bytes_from(root / entry["path"])
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Historical source changed: {logical_name}")
    archive = read_json(root / "archive/MANIFEST.json")
    for name, entry in archive["files"].items():
        path = root / "archive" / name
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Historical archive bytes changed: {name}")
        if hashlib.sha256(bytes_from(path)).hexdigest() != entry["uncompressed_sha256"]:
            raise ValueError(f"Historical decompressed bytes changed: {name}")
    return {"source_files_verified": len(snapshot["files"]),
            "archive_files_verified": len(archive["files"]),
            "snapshot_sha256": digest(root / "study/SNAPSHOT.json"),
            "archive_manifest_sha256": digest(root / "archive/MANIFEST.json")}


def materialize(destination, root=ROOT):
    """Copy exact source bytes into a NEW execution directory.

    The capacity freeze references old stage1 records because they determined
    its prespecified margins. Those records remain historical evidence here;
    newly computed phase4 outcomes have different filenames.
    """
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing workspace: {destination}")
    verify_archive(root)
    destination.mkdir(parents=True)
    snapshot = read_json(root / "study/SNAPSHOT.json")
    for logical_name, entry in snapshot["files"].items():
        target = destination / logical_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes_from(root / entry["path"]))
    (destination / "records.json").write_bytes(bytes_from(root / "archive/stage1/records.json.gz"))
    return destination


def package_sources(root=ROOT):
    """Version orchestration separately from the unchanged scientific snapshot."""
    root = Path(root)
    files = list((root / "reproduce").glob("*.py"))
    files += list((root / "scripts").glob("*.py"))
    files += [root / name for name in ("requirements.txt", "pyproject.toml", "study/SNAPSHOT.json", "archive/MANIFEST.json")]
    return {str(p.relative_to(root)): digest(p) for p in sorted(files) if p.is_file()}


def environment():
    packages = {}
    names = [line.split("==", 1)[0] for line in (ROOT / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "blas_threads": 1}


def copy_file(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
