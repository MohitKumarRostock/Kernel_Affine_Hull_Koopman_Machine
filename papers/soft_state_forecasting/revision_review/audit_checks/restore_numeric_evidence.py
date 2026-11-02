#!/usr/bin/env python3
"""Verify the pinned transport and restore principal numeric evidence only.

No fitting code, model pickle, archive script, or scientific source is executed.
Each part and the concatenated ZIP are checked against the public SHA256
inventory. Only NPZ inputs, saved forecast NPZs, JSON summaries, and the two
principal descriptive detail tables are extracted to the pinned evidence tree.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "remote_evidence"
STUDY = "extensions/reconstruction_mechanism/"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def selected(name):
    if name.startswith(STUDY + "prepared_inputs/tasks/") and name.endswith(".npz"):
        return True
    if name.startswith(STUDY + "results/") and name.endswith(("/predictions.npz", "/summary.json")):
        return True
    return name in {STUDY + "analysis/resources.csv", STUDY + "analysis/safety_diagnostics.csv"}


def main():
    global EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE,
                        help="Root containing the downloaded pinned repository paths")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_checks/numeric_evidence_restoration.json",
                        help="Restoration audit JSON, outside the immutable source paths")
    args = parser.parse_args()
    EVIDENCE = args.evidence_root.resolve()
    manifest_path = EVIDENCE / STUDY / "PACKED_EVIDENCE.json"
    manifest = json.loads(manifest_path.read_text())
    archive = EVIDENCE / "completed_evidence.zip"
    part_records = []
    with archive.open("wb") as target:
        for row in manifest["parts"]:
            path = EVIDENCE / row["path"]
            data = path.read_bytes()
            checksum = hashlib.sha256(data).hexdigest()
            assert len(data) == row["bytes"] and checksum == row["sha256"], row["path"]
            target.write(data)
            part_records.append({"path": row["path"], "bytes": len(data), "sha256": checksum})
    checksum = digest(archive)
    assert archive.stat().st_size == manifest["archive"]["bytes"]
    assert checksum == manifest["archive"]["sha256"]
    restored = []
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        assert len(set(names)) == len(names)
        assert set(names) == set(manifest["files"]) | {"EVIDENCE_INVENTORY.json"}, "Transport membership differs"
        embedded = json.loads(source.read("EVIDENCE_INVENTORY.json"))
        assert embedded["files"] == manifest["files"], "Embedded inventory differs"
        for name in names:
            if name == "EVIDENCE_INVENTORY.json":
                continue
            info = source.getinfo(name)
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not info.flag_bits & 1
            mode = stat.S_IFMT(info.external_attr >> 16)
            assert mode in (0, stat.S_IFREG), name
            assert info.file_size == manifest["files"][name]["bytes"]
            if not selected(name):
                continue
            data = source.read(info)
            checksum_member = hashlib.sha256(data).hexdigest()
            assert checksum_member == manifest["files"][name]["sha256"], name
            destination = EVIDENCE / name
            assert destination.resolve().is_relative_to(EVIDENCE.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert destination.read_bytes() == data, name
            else:
                destination.write_bytes(data)
            restored.append({"path": name, "bytes": len(data), "sha256": checksum_member})
    result = {
        "status": "passed", "pinned_commit": "7dda356138675934aab5e327ec667bde0a5eda79",
        "transport_parts_checked": len(part_records), "transport_members_inventoried": len(names),
        "archive_bytes": archive.stat().st_size, "archive_sha256": checksum,
        "restored_file_count": len(restored), "restored_bytes": sum(x["bytes"] for x in restored),
        "scope": "Byte-level transport verification and selective restoration; no fitted states extracted or executed.",
        "restored": sorted(restored, key=lambda x: x["path"]),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "restored"}, indent=2))


if __name__ == "__main__":
    main()
