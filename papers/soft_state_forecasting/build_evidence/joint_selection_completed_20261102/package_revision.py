#!/usr/bin/env python3
"""Package the completed joint-selection manuscript and retained provenance.

Reporting only: no fitting, scoring, compilation or external writes. Current
PDF-bound visual, document and isolated-build receipts are required. The
completed study and display manifest supply the outcome counts; none is
hardcoded as a favorable result. The ZIP retains the original source root
KAHM_Reconstruction_Prediction. Existing source manifests/ZIPs are replaced
atomically only after the new archive passes its byte checks.

Required current status and receipt fields are documented in SCHEMA below.
`document_verification.new_results_publication` is a plain full commit-SHA
string, equal to `REVISION_STATUS.completed_results_commit`, not an object.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

import fitz


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = ROOT.parent.parent / "extensions/soft_state_joint_selection_20261102"
RECEIPT_RELATIVE = "build_evidence/joint_selection_completed_20261102"
ZIP_ROOT = "KAHM_Reconstruction_Prediction"
BASELINE_COMMIT = "6c39a003e61ace058851ec6badca3e8dd13ef183"
BASELINE_MANIFEST = "revision_review/prior_metadata/pre_joint_selection_SOURCE_FILE_MANIFEST.json"
BASELINE_MANIFEST_SHA256 = "fa1bf694fdfd82c18acca44ed4a6200c8ed97c48deb7a58c4ba3dff9948899db"
BASELINE_COUNTS = {"generated": 121, "evidence": 76}
SKIP_SUFFIXES = {".aux", ".blg", ".fdb_latexmk", ".fls", ".out", ".pyc", ".pyo",
                 ".toc", ".lof", ".lot", ".nav", ".snm", ".vrb", ".synctex"}
SKIP_PARTS = {"__pycache__", "__MACOSX", ".git", ".pytest_cache", ".mypy_cache",
              ".ruff_cache", ".ipynb_checkpoints", "node_modules"}
SOURCE_SUFFIXES = {".tex", ".bib", ".sty", ".bst"}
GRAPHIC_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".svg", ".eps"}
SCHEMA = {
    "REVISION_STATUS.json": {
        "status": "complete_audited_joint_selection_revision",
        "scientific_source_commit": "full lowercase 40-character SHA, matching the study PUBLICATION.json",
        "completed_results_commit": "full lowercase 40-character SHA string of the completed result publication",
        "repository_branch": "matching study PUBLICATION.json branch",
        "source_published": True, "production_data_generated": True, "production_audit_completed": True,
        "tasks": "integer matching completed analysis/audit", "independent_repetitions": "integer matching analysis/audit",
        "primary_family_size": 6, "criteria_established": "integer matching analysis/audit",
        "audit_checks": "integer matching passed independent audit",
        "freeze_sha256": "current study freeze digest", "analysis_summary_sha256": "current complete analysis digest",
        "independent_audit_sha256": "current full passed audit digest",
        "joint_asset_manifest_sha256": "current generated/joint_selection/asset_manifest.json digest",
        "verification_directory": RECEIPT_RELATIVE,
    },
    "main_review.json and supplement_review.json": {
        "status": "passed", "issues": [], "pdf_sha256": "matching current PDF",
        "pages": "matching current PDF page count", "coverage": "nonempty, truthful review coverage description",
    },
    "document_verification.json": {
        "status": "passed", "scientific_source_commit": "same SHA string as current status",
        "new_results_publication": "same SHA STRING as current status completed_results_commit; not an object",
        "joint_asset_manifest_sha256": "same current asset digest as status",
        "pdfs": {"main and supplement": {"sha256": "current PDF", "pages": "current count", "bytes": "current size",
            "unresolved_or_overfull": [], "stale_status_markers": []}},
        "preservation": {"generated": {"files": 121, "differences": []}, "evidence": {"files": 76, "differences": []}},
    },
    "isolated_build_verification.json": {
        "status": "passed", "source_build_inputs": "exact map from build_input_hashes(paper_root)",
        "source_input_zip_sha256": "digest of actual extracted source-input ZIP",
        "network_required": False, "model_fitting_required": False,
        "pdfs": {"main and supplement": {"sha256": "current PDF", "pages": "current count", "bytes": "current size",
            "no_unresolved_or_overfull_warnings": True}},
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def hash_string(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def confined(root, relative):
    rel = Path(relative)
    path = (root / rel).resolve()
    require(not rel.is_absolute() and ".." not in rel.parts and path.is_relative_to(root),
            "Relative file leaves its declared root")
    return path


def eligible(path, root=ROOT):
    path, root = Path(path), Path(root)
    relative = path.relative_to(root)
    if path.is_symlink():
        raise ValueError("Symlinks are not packaged: " + str(relative))
    return (path.is_file() and not any(part in SKIP_PARTS for part in relative.parts)
            and not any(part.startswith((".soft_state", ".joint-selection", ".kahm-package")) for part in relative.parts)
            and path.name != ".DS_Store" and path.suffix not in SKIP_SUFFIXES
            and not path.name.endswith(".synctex.gz")
            and str(relative) not in ("main.log", "supplement.log"))


def build_input_hashes(root=ROOT):
    """Hash current TeX/Bib/style inputs and generated graphics, without receipts.

    Call this after the final clean build has been installed. Historical review,
    build-evidence and compact-evidence trees are not compiler input roots.
    Compiled main/supplement PDFs and transient files are excluded. Root BBLs
    remain in the delivered ZIP but are regenerated by the isolated BibTeX run.
    """
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in ("build_evidence", "revision_review", "evidence") for part in relative.parts):
            continue
        if eligible(path, root) and (path.suffix in SOURCE_SUFFIXES or
                                     (relative.parts[0] == "generated" and path.suffix in GRAPHIC_SUFFIXES)):
            result[str(relative)] = sha(path)
    require("main.tex" in result and "supplement.tex" in result, "Both document entry points are required")
    return result


def check_preservation(root):
    baseline = root / BASELINE_MANIFEST
    require(sha(baseline) == BASELINE_MANIFEST_SHA256, "Archived baseline source manifest changed")
    source = read(baseline)["files"]
    preserved = {}
    for name, expected_count in BASELINE_COUNTS.items():
        selected = {key: value for key, value in source.items() if key.startswith(name + "/")}
        require(len(selected) == expected_count, "Wrong baseline " + name + " roster")
        for relative, metadata in selected.items():
            path = confined(root, relative)
            require(path.is_file() and path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"],
                    "Historical payload changed or missing: " + relative)
            require(eligible(path, root), "Historical payload would be omitted: " + relative)
        preserved[name] = {"files": len(selected), "differences": []}
    return preserved


def verify_scientific_status(root, study, status):
    require(status.get("status") == "complete_audited_joint_selection_revision", "Completed joint-selection revision required")
    for key in ("source_published", "production_data_generated", "production_audit_completed"):
        require(status.get(key) is True, "Incomplete revision status: " + key)
    require(status.get("verification_directory") == RECEIPT_RELATIVE, "Wrong current review directory")
    for key in ("scientific_source_commit", "completed_results_commit"):
        require(hash_string(status.get(key), 40), "A full actual commit SHA string is required: " + key)
    publication = read(study / "PUBLICATION.json")
    summary_path = study / "analysis/analysis_summary.json"
    audit_path = study / "verification/independent_audit.json"
    summary, audit = read(summary_path), read(audit_path)
    asset_path = root / "generated/joint_selection/asset_manifest.json"
    asset = read(asset_path)
    require(publication.get("status") == "published" and publication.get("source_files_verified") is True and
            status["scientific_source_commit"] == publication["source_commit"] and
            status["repository_branch"] == publication["branch"], "Published source identity differs from current status")
    require(summary.get("status") == "complete" and audit.get("status") == "passed" and
            asset.get("status") == "complete", "Completed analysis, audit and display generation required")
    require(summary["schema"] == "soft-state-joint-selection-analysis-v1" and
            audit["schema"] == "independent-soft-state-joint-audit-v1" and
            asset["schema"] == "kahm-joint-selection-manuscript-assets-v1", "Wrong completed study artifact schema")
    for key, expected in (("tasks", summary["tasks"]), ("independent_repetitions", summary["independent_repetitions"]),
                          ("criteria_established", summary["criteria_established"])):
        require(type(status.get(key)) is int and status[key] == expected == audit[key], "Outcome count differs: " + key)
    require(status["tasks"] == 256 and status["independent_repetitions"] == 32 and
            status["primary_family_size"] == summary["primary_family_size"] == 6,
            "Wrong fixed cohort or six-comparison family")
    require(len(summary["primary_contrasts"]) == len(audit["primary_contrasts"]) == len(asset["primary_recomputed"]) == 6,
            "Incomplete primary comparisons")
    require(type(status.get("audit_checks")) is int and status["audit_checks"] == audit["checks"], "Audit check count differs")
    hashes = {"freeze_sha256": sha(study / "freeze.json"), "analysis_summary_sha256": sha(summary_path),
              "independent_audit_sha256": sha(audit_path), "joint_asset_manifest_sha256": sha(asset_path)}
    for key, actual in hashes.items():
        require(status.get(key) == actual, "Current status hash differs: " + key)
    require(publication["freeze_sha256"] == summary["freeze_sha256"] == audit["freeze_sha256"] == hashes["freeze_sha256"],
            "Source freeze identity differs")
    require(audit["analysis_summary_sha256"] == hashes["analysis_summary_sha256"], "Audit does not bind the current analysis")
    require(asset["source_commit"] == publication["source_commit"] and
            asset["inputs"]["analysis/analysis_summary.json"]["sha256"] == hashes["analysis_summary_sha256"] and
            asset["inputs"]["verification/independent_audit.json"]["sha256"] == hashes["independent_audit_sha256"],
            "Displays do not bind the current completed evidence")
    for name, metadata in asset["outputs"].items():
        path = confined(asset_path.parent, name)
        require(path.is_file() and path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"],
                "Generated joint-selection asset changed: " + name)
    return publication, summary, audit, hashes


def verify_documents(root, status, preservation):
    receipts = root / RECEIPT_RELATIVE
    verified = read(receipts / "document_verification.json")
    isolated = read(receipts / "isolated_build_verification.json")
    require(verified.get("status") == isolated.get("status") == "passed", "Current document and isolated-build verification required")
    require(verified.get("scientific_source_commit") == status["scientific_source_commit"], "Document review source identity differs")
    result_commit = verified.get("new_results_publication")
    require(hash_string(result_commit, 40) and result_commit == status["completed_results_commit"],
            "Document result publication must be the same full SHA string as current status")
    require(verified.get("joint_asset_manifest_sha256") == status["joint_asset_manifest_sha256"],
            "Document verification refers to different result displays")
    require(verified.get("preservation") == preservation, "Document preservation receipt differs from checked baseline")
    require(isolated.get("network_required") is False and isolated.get("model_fitting_required") is False,
            "Clean source build must require neither network nor model fitting")
    require(hash_string(isolated.get("source_input_zip_sha256")), "Actual isolated source-input archive digest required")
    inputs = build_input_hashes(root)
    require(isolated.get("source_build_inputs") == inputs, "Compiler inputs changed after the isolated build")
    pages, pdfs, receipt_hashes = {}, {}, {}
    for filename in ("document_verification.json", "isolated_build_verification.json"):
        receipt_hashes[filename] = sha(receipts / filename)
    for stem in ("main", "supplement"):
        pdf = root / (stem + ".pdf")
        with fitz.open(pdf) as document:
            pages[stem] = len(document)
        metadata = {"sha256": sha(pdf), "pages": pages[stem], "bytes": pdf.stat().st_size}
        review_path = receipts / (stem + "_review.json")
        review = read(review_path)
        require(review.get("status") == "passed" and review.get("issues") == [] and
                review.get("pdf_sha256") == metadata["sha256"] and review.get("pages") == pages[stem] and
                bool(review.get("coverage")), "Visual review does not approve the current " + stem + ".pdf")
        for name, receipt in (("document", verified), ("isolated build", isolated)):
            other = receipt["pdfs"][stem]
            for key, value in metadata.items():
                require(other.get(key) == value, name + " verification binds a different " + stem + ".pdf")
        require(verified["pdfs"][stem].get("unresolved_or_overfull") == [] and
                verified["pdfs"][stem].get("stale_status_markers") == [], "Unresolved current document checks")
        require(isolated["pdfs"][stem].get("no_unresolved_or_overfull_warnings") is True,
                "Unresolved isolated-build warnings")
        receipt_hashes[stem + "_review.json"] = sha(review_path)
        pdfs[stem] = metadata
    return pages, pdfs, receipt_hashes, inputs


def package(root, study, output):
    root, study, output = Path(root).resolve(), Path(study).resolve(), Path(output).resolve()
    require(not output.is_relative_to(root), "ZIP output must be outside its source tree")
    require(output.suffix.lower() == ".zip", "Output must be a ZIP file")
    status_path = root / "REVISION_STATUS.json"
    status = read(status_path)
    publication, summary, audit, evidence_hashes = verify_scientific_status(root, study, status)
    preservation = check_preservation(root)
    pages, pdfs, receipt_hashes, compiler_inputs = verify_documents(root, status, preservation)
    record = root / "SOURCE_FILE_MANIFEST.json"
    original_record_hash = sha(record) if record.is_file() else None
    files = sorted(path for path in root.rglob("*") if eligible(path, root) and path != record)
    inventory = {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in files}
    now = datetime.now(timezone.utc)
    manifest = {
        "title": "Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines",
        "revision_date": status.get("revision_date", now.date().isoformat()), "created_utc": now.isoformat(),
        "status": status["status"],
        "scope": "Completed joint input-scaling and native-dimension follow-up, audited same-cohort comparisons, revised main paper and supplementary studies; prior evidence retained separately",
        "anonymous_review_upload": False, "submission_or_acceptance_claimed": False,
        "zip_root": ZIP_ROOT, "entry_points": ["main.tex", "supplement.tex"], "pages": pages, "pdfs": pdfs,
        "scientific_source_commit": status["scientific_source_commit"], "source_tree_sha": publication["source_tree_sha"],
        "completed_results_commit": status["completed_results_commit"], "repository_branch": status["repository_branch"],
        "new_production_model_fits": status["production_data_generated"], "new_evaluation_observations": status["production_data_generated"],
        "tasks": summary["tasks"], "independent_repetitions": summary["independent_repetitions"],
        "forecast_arrays": summary["forecast_arrays"], "primary_family_size": summary["primary_family_size"],
        "primary_criteria_established": summary["criteria_established"], "independent_production_audit_passed": audit["status"] == "passed",
        "audit_checks": audit["checks"], "audit_replay_counts": {key: audit[key] for key in
            ("B_penalty_path_arrays_replayed", "C_forecast_arrays_replayed", "D_forecast_arrays_replayed")},
        "evidence_hashes": evidence_hashes, "current_receipt_hashes": receipt_hashes,
        "verification_directory": RECEIPT_RELATIVE, "compiler_input_count": len(compiler_inputs),
        "baseline_preservation": {"commit": BASELINE_COMMIT, "manifest": BASELINE_MANIFEST,
            "manifest_sha256": BASELINE_MANIFEST_SHA256, **preservation},
        "generated_files_preserved": preservation["generated"]["files"],
        "evidence_files_preserved": preservation["evidence"]["files"],
        "inventory_excludes": ["SOURCE_FILE_MANIFEST.json (self reference)", "transient TeX/Python outputs and caches",
                               "root main.log and supplement.log; dated build/provenance logs remain included"],
        "files": inventory,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = None
    temporary_manifest = None
    try:
        handle = tempfile.NamedTemporaryFile(prefix=".kahm-package-", suffix=".zip", dir=output.parent, delete=False)
        temporary_archive = Path(handle.name)
        handle.close()
        with zipfile.ZipFile(temporary_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.write(path, ZIP_ROOT + "/" + str(path.relative_to(root)))
            archive.writestr(ZIP_ROOT + "/SOURCE_FILE_MANIFEST.json", payload)
        with zipfile.ZipFile(temporary_archive) as archive:
            require(archive.testzip() is None, "Archive CRC integrity failure")
            expected_names = {ZIP_ROOT + "/" + relative for relative in inventory} | {ZIP_ROOT + "/SOURCE_FILE_MANIFEST.json"}
            require(len(archive.namelist()) == len(expected_names) and set(archive.namelist()) == expected_names,
                    "Archive members differ from complete source manifest")
            for relative, metadata in inventory.items():
                data = archive.read(ZIP_ROOT + "/" + relative)
                require(len(data) == metadata["bytes"] and hashlib.sha256(data).hexdigest() == metadata["sha256"],
                        "Archive payload differs: " + relative)
                path = root / relative
                require(path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"],
                        "Source changed during packaging: " + relative)
            require(archive.read(ZIP_ROOT + "/SOURCE_FILE_MANIFEST.json") == payload, "Packaged source manifest differs")
        require((sha(record) if record.is_file() else None) == original_record_hash, "Source manifest changed concurrently")
        handle = tempfile.NamedTemporaryFile(prefix=".kahm-package-", suffix=".json", dir=root, delete=False)
        temporary_manifest = Path(handle.name)
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, record)
        temporary_manifest = None
        os.replace(temporary_archive, output)
        temporary_archive = None
    finally:
        for path in (temporary_archive, temporary_manifest):
            if path is not None and path.exists():
                path.unlink()
    return {"status": "complete", "path": str(output), "members": len(inventory) + 1,
            "bytes": output.stat().st_size, "sha256": sha(output), "pages": pages,
            "tasks": summary["tasks"], "independent_repetitions": summary["independent_repetitions"],
            "primary_family_size": summary["primary_family_size"], "criteria_established": summary["criteria_established"],
            "baseline_generated_files_preserved": preservation["generated"]["files"],
            "baseline_evidence_files_preserved": preservation["evidence"]["files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=ROOT)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "soft_state_forecasting_source.zip")
    parser.add_argument("--print-schema", action="store_true", help="Print receipt/status fields without packaging")
    args = parser.parse_args()
    if args.print_schema:
        print(json.dumps(SCHEMA, indent=2, sort_keys=True))
        return
    print(json.dumps(package(args.paper_root, args.study_root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
