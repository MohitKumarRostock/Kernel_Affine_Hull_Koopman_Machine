"""Produce the fully accounted reference archive, retaining declared failures.

This post-execution transport follows REPORTING_AMENDMENT.md. It does not
modify the frozen reference producer or its original success-only precondition.
The unchanged reference.restore command reads the resulting checked format.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

from .data import digest, write_json
from .reference import MANIFEST, PART_BYTES, PARTS, REFERENCE, ROOT, RUN, description, verify_contents
from .reporting import verify_report


def build():
    if REFERENCE.exists():
        raise FileExistsError("The reference archive already exists; preserve its immutable bytes")
    report = verify_report(RUN)
    required = {
        "verification_passed": True, "all_tasks_attempted": True,
        "all_methods_succeeded": False, "expected_tasks": 168,
        "observed_tasks": 168, "completed_tasks": 166,
        "attempted_methods": 1344, "successful_methods": 1336,
        "failed_method_count": 8, "attempted_policies": 336,
        "successful_policies": 332, "failed_policy_count": 4,
        "archive_common_support_tasks": 22, "archive_common_support_series": 11,
        "archive_full_roster_kahm_mean": None,
    }
    for name, expected in required.items():
        if report.get(name) != expected:
            raise ValueError(f"Reference failure-accounting gate differs: {name}")
    original = json.loads((RUN / "analysis/summary.json").read_text())
    if original["complete"] is not False or len(original["failed_tasks"]) != 2:
        raise ValueError("Original incomplete-study analysis has not been preserved")
    files = {}
    for path in sorted(RUN.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in research outcomes")
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    inventory = {
        "schema_version": 1,
        "scope": "All 168 attempted tasks, 1336 successful standalone forecasts, eight method failures and four policy failures; original analysis and distinct post-hoc common-support reporting both retained",
        "files": {name: description(path.read_bytes()) for name, path in files.items()},
        "freeze_sha256": digest(ROOT / "freeze.json"),
        "reporting_amendment_sha256": digest(ROOT / "REPORTING_AMENDMENT.md"),
        "report_summary_sha256": digest(RUN / "report/summary.json"),
        "failure_accounting": required,
    }
    with tempfile.TemporaryDirectory(prefix="kahm_accounted_reference_") as temporary:
        packed = Path(temporary) / "utility_v1.zip"
        with zipfile.ZipFile(packed, "w", allowZip64=True) as archive:
            for name in sorted([*files, MANIFEST]):
                payload = ((json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
                           if name == MANIFEST else files[name].read_bytes())
                member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                member.external_attr = 0o100644 << 16
                member.compress_type = zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED
                archive.writestr(member, payload, compresslevel=9)
        with zipfile.ZipFile(packed) as archive:
            verify_contents(archive)
        REFERENCE.mkdir()
        parts = []
        with packed.open("rb") as stream:
            while payload := stream.read(PART_BYTES):
                name = f"utility_v1.zip.part{len(parts) + 1:03d}"
                (REFERENCE / name).write_bytes(payload)
                parts.append({"filename": name, **description(payload)})
        write_json(REFERENCE / PARTS, {
            "schema_version": 1,
            "archive": {"filename": packed.name, "bytes": packed.stat().st_size, "sha256": digest(packed)},
            "parts": parts, "payload_files": len(files),
            "freeze_sha256": inventory["freeze_sha256"],
            "failure_accounting": required,
        })
    print(json.dumps({"reference": str(REFERENCE), "parts": len(parts), "files": len(files), "all_methods_succeeded": False}))


if __name__ == "__main__":
    build()
