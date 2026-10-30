"""Store and restore the complete extension outcomes with byte-level checks.

This transport does not refit or change a scientific result. Predictions are
retained in full. Existing different files are never replaced on restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from .data import digest, write_json

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference"
RUN = ROOT / "results/utility_v1"
PART_BYTES = 4 * 1024**2
MANIFEST = "REFERENCE_FILES.json"
PARTS = "utility_v1.zip.parts.json"


def description(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or ":" in path.parts[0]):
        raise ValueError("Unsafe archive path")
    return name


def verify_contents(archive):
    names = archive.namelist()
    if len(names) != len(set(names)) or MANIFEST not in names:
        raise ValueError("Duplicate entries or missing file manifest")
    manifest = json.loads(archive.read(MANIFEST))
    if set(names) != set(manifest["files"]) | {MANIFEST}:
        raise ValueError("Archive inventory differs from the complete manifest")
    for name, expected in manifest["files"].items():
        safe_name(name)
        if not name.startswith("results/utility_v1/"):
            raise ValueError("Payload outside the declared reference run")
        if description(archive.read(name)) != expected:
            raise ValueError("Reference file checksum mismatch: " + name)
    return manifest


def build():
    if REFERENCE.exists():
        raise FileExistsError("Reference already exists; preserve the original artifact")
    from .analysis import summarize
    summary = summarize(RUN)
    if not summary["complete"] or summary["observed_tasks"] != 168 or summary["expected_tasks"] != 168:
        raise ValueError("A complete independently audited 168-task study is required")
    files = {}
    for path in sorted(RUN.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in reference outcomes")
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    inventory = {"schema_version": 1, "scope": "All final predictions, targets, trials, costs and analyses for the full forecasting-utility extension",
                 "files": {name: description(path.read_bytes()) for name, path in files.items()},
                 "analysis_summary_sha256": digest(RUN / "analysis/summary.json"),
                 "freeze_sha256": digest(ROOT / "freeze.json")}
    with tempfile.TemporaryDirectory(prefix="kahm_reference_") as temporary:
        packed = Path(temporary) / "utility_v1.zip"
        with zipfile.ZipFile(packed, "w", allowZip64=True) as archive:
            for name in sorted([*files, MANIFEST]):
                content = ((json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
                           if name == MANIFEST else files[name].read_bytes())
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED
                archive.writestr(info, content, compresslevel=9)
        with zipfile.ZipFile(packed) as archive:
            verify_contents(archive)
        REFERENCE.mkdir()
        parts = []
        with packed.open("rb") as stream:
            while content := stream.read(PART_BYTES):
                name = f"utility_v1.zip.part{len(parts) + 1:03d}"
                (REFERENCE / name).write_bytes(content)
                parts.append({"filename": name, **description(content)})
        write_json(REFERENCE / PARTS, {
            "schema_version": 1, "archive": {"filename": packed.name, "bytes": packed.stat().st_size, "sha256": digest(packed)},
            "parts": parts, "payload_files": len(files), "freeze_sha256": inventory["freeze_sha256"],
        })
    print(json.dumps({"reference": str(REFERENCE), "parts": len(parts), "files": len(files)}))


def restore():
    transport = json.loads((REFERENCE / PARTS).read_text())
    with tempfile.TemporaryDirectory(prefix="kahm_restore_") as temporary:
        packed = Path(temporary) / "utility_v1.zip"
        with packed.open("wb") as stream:
            for part in transport["parts"]:
                name = safe_name(part["filename"])
                if len(PurePosixPath(name).parts) != 1:
                    raise ValueError("Part filename must be a basename")
                content = (REFERENCE / name).read_bytes()
                if description(content) != {k: part[k] for k in ("bytes", "sha256")}:
                    raise ValueError("Reference part checksum mismatch")
                stream.write(content)
        if packed.stat().st_size != transport["archive"]["bytes"] or digest(packed) != transport["archive"]["sha256"]:
            raise ValueError("Assembled reference checksum mismatch")
        with zipfile.ZipFile(packed) as archive:
            inventory = verify_contents(archive)
            for name, expected in inventory["files"].items():
                target = ROOT / name
                for parent in (target, *target.parents):
                    if parent == ROOT.parent:
                        break
                    if parent.is_symlink():
                        raise ValueError("Symlink in restore destination")
                if target.exists() and (not target.is_file() or description(target.read_bytes()) != expected):
                    raise FileExistsError("Existing different outcome is preserved: " + str(target))
            if RUN.exists():
                extra = {p.relative_to(ROOT).as_posix() for p in RUN.rglob("*") if p.is_file()} - set(inventory["files"])
                if extra:
                    raise FileExistsError("Existing unlisted outcomes are preserved")
            for name in inventory["files"]:
                target = ROOT / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
    print(json.dumps({"restored": str(RUN), "files": len(inventory["files"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "restore"))
    args = parser.parse_args()
    {"build": build, "restore": restore}[args.command]()


if __name__ == "__main__":
    main()
