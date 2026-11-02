#!/usr/bin/env python3
"""Package the completed, audited manuscript after current document review.

This is a reporting utility, outside the frozen numerical experiment.
It never fits or scores a model. Review receipts must match the current PDFs.
"""
from pathlib import Path
import datetime
import hashlib
import json
import zipfile
import fitz

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / "build_evidence/ablation_completed_20261102"
OUT = ROOT.parent / "soft_state_forecasting_source.zip"
SKIP_SUFFIXES = {".aux", ".blg", ".fdb_latexmk", ".fls", ".out", ".pyc"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(path):
    relative = path.relative_to(ROOT)
    return (path.is_file() and "__pycache__" not in relative.parts
            and not any(part.startswith(".soft_state") for part in relative.parts)
            and path.suffix not in SKIP_SUFFIXES
            and str(relative) not in ("main.log", "supplement.log"))


def main():
    status = json.loads((ROOT / "REVISION_STATUS.json").read_text())
    if status.get("status") != "complete_audited_revision":
        raise ValueError("Completed revision status is required")
    pages = {}
    for stem in ("main", "supplement"):
        pdf = ROOT / (stem + ".pdf")
        receipt = json.loads((RECEIPTS / (stem + "_review.json")).read_text())
        bound_hash = receipt.get("pdf_sha256")
        if bound_hash is None and isinstance(receipt.get("pdf"), dict):
            bound_hash = receipt["pdf"].get("sha256")
        if receipt.get("status") != "passed" or receipt.get("issues") or bound_hash != sha(pdf):
            raise ValueError("Visual review does not approve current " + stem + ".pdf")
        with fitz.open(pdf) as document:
            pages[stem] = len(document)
    verified = json.loads((RECEIPTS / "document_verification.json").read_text())
    isolated = json.loads((RECEIPTS / "isolated_build_verification.json").read_text())
    if verified.get("status") != "passed" or isolated.get("status") != "passed":
        raise ValueError("Document and clean build verification are required")
    for stem in pages:
        if verified["pdfs"][stem]["sha256"] != sha(ROOT / (stem + ".pdf")):
            raise ValueError("Document verification refers to a different PDF")

    record = ROOT / "SOURCE_FILE_MANIFEST.json"
    files = sorted(path for path in ROOT.rglob("*") if eligible(path) and path != record)
    manifest = {
        "title": "Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines",
        "revision_date": "2026-11-02",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Completed four-arm representation study, audited results, revised main paper and full supplementary studies",
        "anonymous_review_upload": False,
        "baseline_archive_sha256": "d49f0580a832f73258592016bfd6a87b04e5609f6021604371d34b1d9ea906e0",
        "entry_points": ["main.tex", "supplement.tex"],
        "pages": pages,
        "scientific_source_commit": status["scientific_source_commit"],
        "completed_results_commit": status["completed_results_commit"],
        "repository_branch": status["repository_branch"],
        "new_production_model_fits": True,
        "new_evaluation_observations": True,
        "tasks": 256,
        "independent_repetitions": 32,
        "primary_criteria_established": 3,
        "primary_family_size": 4,
        "independent_production_audit_passed": True,
        "generated_files_preserved": 106,
        "evidence_files_preserved": 48,
        "inventory_excludes": ["SOURCE_FILE_MANIFEST.json", "temporary TeX and Python outputs"],
        "files": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size,
                    "sha256": sha(path)} for path in files},
    }
    record.write_text(json.dumps(manifest, indent=2) + "\n")
    files.append(record)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, "KAHM_Reconstruction_Prediction/" + str(path.relative_to(ROOT)))
    with zipfile.ZipFile(OUT) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity failure")
        for relative, metadata in manifest["files"].items():
            data = archive.read("KAHM_Reconstruction_Prediction/" + relative)
            if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
                raise ValueError("Archive content differs: " + relative)
    print(json.dumps({"path": str(OUT), "members": len(files), "bytes": OUT.stat().st_size,
                      "sha256": sha(OUT), "pages": pages}))


if __name__ == "__main__":
    main()
