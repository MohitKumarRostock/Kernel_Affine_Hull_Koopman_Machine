#!/usr/bin/env python3
"""Package the coordinated author source and verify an isolated LaTeX rebuild.

Run from a complete checkout after compiling and reviewing main/supplement.
This is a document build, not an experimental run or a numerical audit.
The prior author archive is required so its historical provenance is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
INTEGRATION = HERE.parents[1]
MANUSCRIPT = INTEGRATION / "manuscript"
PREFIX = "KAHM_Reconstruction_Prediction/"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def info(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def pdf_text(path):
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-archive", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    snapshot = json.loads((HERE / "input_snapshot.json").read_text())
    expected = "cd11be1c71936cf7e3001e67040db42c6fe0497fcd639eed65a011f4a6a05298"
    if sha(args.source_archive) != expected:
        raise ValueError("The required prior author source archive has changed")
    required_receipts = ["revision_verification.json", "principal_display_verification.json",
                         "main_visual_review.json", "supplement_visual_review.json"]
    for name in required_receipts:
        value = json.loads((HERE / name).read_text())
        if value.get("status") != "passed":
            raise ValueError(f"Document verification not passed: {name}")
    for entry in ["main", "supplement"]:
        visual = json.loads((HERE / (entry + "_visual_review.json")).read_text())
        pdf = MANUSCRIPT / (entry + ".pdf")
        pages = len(PdfReader(pdf).pages)
        if visual.get("pdf_sha256") != sha(pdf):
            raise ValueError(f"Visual review does not identify current {entry} PDF")
        if sorted(visual.get("inspected_pages", [])) != list(range(1, pages + 1)):
            raise ValueError(f"Visual review does not cover every {entry} page")
    delivered_names = {"main": "KAHM_Reconstruction_Prediction_Manuscript.pdf",
                       "supplement": "KAHM_Supplementary_Studies.pdf"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kahm_coordinated_source_") as temporary:
        stage = Path(temporary) / PREFIX.rstrip("/")
        stage.mkdir()
        files = set()
        with zipfile.ZipFile(args.source_archive) as old:
            for member in old.infolist():
                if member.is_dir():
                    continue
                if not member.filename.startswith(PREFIX):
                    raise ValueError(f"Unexpected archive prefix: {member.filename}")
                relative = Path(member.filename[len(PREFIX):])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive member")
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(old.read(member))
                files.add(relative.as_posix())
        prior = stage / "revision_review/prior_metadata/dimension_SOURCE_FILE_MANIFEST.json"
        shutil.copyfile(stage / "SOURCE_FILE_MANIFEST.json", prior)
        files.add(prior.relative_to(stage).as_posix())
        # Overlay all previously supplied current sources, preserving ZIP-only history.
        for relative in sorted(files - {"SOURCE_FILE_MANIFEST.json"}):
            current = MANUSCRIPT / relative
            if current.is_file():
                shutil.copyfile(current, stage / relative)
        additions = ["build_followup_assets.py", "core_followup_experiments.tex",
                     "supporting/section_preprocessing_dimension.tex",
                     "supporting/section_crossfit_residual.tex"]
        optional = MANUSCRIPT / "supporting/appendix_followup_protocols.tex"
        if optional.is_file():
            additions.append(optional.relative_to(MANUSCRIPT).as_posix())
        for namespace in ["preprocessing_dimension", "crossfit_residual"]:
            additions.extend(p.relative_to(MANUSCRIPT).as_posix()
                             for p in sorted((MANUSCRIPT / "generated" / namespace).rglob("*"))
                             if p.is_file())
        for relative in additions:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MANUSCRIPT / relative, target)
            files.add(relative)
        for current in sorted(HERE.iterdir()):
            if not current.is_file() or current.name in {"package_build_verification.json", "publication_receipt.json"}:
                continue
            relative = "build_evidence/coordinated_20261101/" + current.name
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(current, target)
            files.add(relative)

        revision = json.loads((HERE / "revision_verification.json").read_text())
        for relative, digest in revision["checks"]["latex_input_paths"]["source_sha256"].items():
            if relative not in files or sha(stage / relative) != digest:
                raise ValueError(f"Packaged source differs from revision verification: {relative}")
        displays = json.loads((HERE / "principal_display_verification.json").read_text())
        for name, record in displays["verified_displays"].items():
            relative = "generated/dimension_forecasting/" + name
            if relative not in files or sha(stage / relative) != record["sha256"]:
                raise ValueError(f"Packaged principal display differs from verification: {relative}")
        for namespace in ["preprocessing_dimension", "crossfit_residual"]:
            base = stage / "generated" / namespace
            if sha(base / "asset_manifest.json") != revision["checks"]["assets_" + namespace]["manifest_sha256"]:
                raise ValueError(f"Packaged follow-up manifest differs from verification: {namespace}")
            assets = json.loads((base / "asset_manifest.json").read_text())
            if sha(stage / "build_followup_assets.py") != assets["builder_sha256"]:
                raise ValueError("Follow-up builder does not match asset provenance")
            for name, digest in assets["outputs"].items():
                if sha(base / name) != digest:
                    raise ValueError(f"Packaged follow-up output differs from manifest: {namespace}/{name}")

        builds = {}
        environment = dict(os.environ)
        cleared_search_variables = ["TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXMFOUTPUT"]
        for variable in cleared_search_variables:
            environment.pop(variable, None)
        system_roots = [Path(p) for p in ["/etc/texmf", "/usr/share/texlive",
                        "/usr/share/texmf", "/usr/local/share/texmf", "/var/lib/texmf"]]
        for entry in ["main", "supplement"]:
            original_pdf = MANUSCRIPT / (entry + ".pdf")
            expected_text = pdf_text(original_pdf)
            run = subprocess.run(["latexmk", "-g", "-pdf", "-interaction=nonstopmode",
                                  "-halt-on-error", entry + ".tex"], cwd=stage,
                                 env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if run.returncode:
                raise RuntimeError(f"Isolated {entry} build failed:\n{run.stdout[-12000:]}")
            log = (stage / (entry + ".log")).read_text()
            errors = [line for line in log.splitlines()
                      if re.search(r"Overfull|undefined|multiply defined|LaTeX Error|Missing character", line)]
            if errors:
                raise ValueError(f"Isolated {entry} has unresolved layout/reference warnings: {errors}")
            rebuilt = stage / (entry + ".pdf")
            if pdf_text(rebuilt) != expected_text:
                raise ValueError(f"Isolated {entry} PDF text differs from the reviewed PDF")
            closure = set()
            for line in (stage / (entry + ".fls")).read_text().splitlines():
                if not line.startswith("INPUT "):
                    continue
                p = Path(line[6:])
                p = p if p.is_absolute() else stage / p
                try:
                    relative = p.resolve().relative_to(stage).as_posix()
                except ValueError:
                    if not any(p.resolve().is_relative_to(root) for root in system_roots):
                        raise ValueError(f"External non-system dependency in isolated build: {p}")
                    continue
                if p.suffix in {".aux", ".out", ".bbl", ".toc"}:
                    continue
                if relative not in files:
                    raise ValueError(f"Input omitted from source package: {relative}")
                closure.add(relative)
            bibliography_inputs = re.findall(r"^(?:The style file|Database file #\d+): (.+)$",
                                             (stage / (entry + ".blg")).read_text(), re.MULTILINE)
            if not bibliography_inputs:
                raise ValueError("No bibliography input records found")
            for relative in bibliography_inputs:
                if relative not in files or not (stage / relative).is_file():
                    raise ValueError(f"Bibliography dependency omitted from package: {relative}")
                closure.add(relative)
            pages = len(PdfReader(original_pdf).pages)
            if len(PdfReader(rebuilt).pages) != pages:
                raise ValueError("Rebuild page count differs")
            builds[entry] = {"status": "passed", "pages": pages,
                             "reviewed_pdf": info(original_pdf),
                             "isolated_rebuild_text_and_page_count_match": True,
                             "overfull_or_unresolved_reference_warnings": errors,
                             "underfull_vbox_messages": len(re.findall(r"Underfull \\vbox", log)),
                             "local_input_closure": {r: sha(stage / r) for r in sorted(closure)},
                             "bibliography_inputs": bibliography_inputs,
                             "final_log_sha256": hashlib.sha256(log.encode()).hexdigest()}
            # Deliver exactly the visually reviewed build, rather than a timestamp variant.
            shutil.copyfile(original_pdf, rebuilt)
            shutil.copyfile(original_pdf, args.output_dir / delivered_names[entry])

        verification = {"schema": "kahm-coordinated-document-build-v1", "status": "passed",
                        "date": "2026-11-01", "timezone": "Europe/Vienna",
                        "scope": "Standalone author-source compilation and document input closure; no scientific refitting or new statistical inference",
                        "prior_author_archive_sha256": expected,
                        "repository_parent_commit": snapshot["base_commit"],
                        "cleared_inherited_search_variables": cleared_search_variables,
                        "allowed_external_tex_system_roots": [str(p) for p in system_roots],
                        "builds": builds,
                        "verification_receipts": {n: info(HERE / n) for n in required_receipts},
                        "packaging_script_sha256": sha(Path(__file__))}
        write_json(HERE / "package_build_verification.json", verification)
        relative = "build_evidence/coordinated_20261101/package_build_verification.json"
        shutil.copyfile(HERE / "package_build_verification.json", stage / relative)
        files.add(relative)
        # The new inventory excludes itself and includes the preserved older inventory.
        files.discard("SOURCE_FILE_MANIFEST.json")
        manifest = {"revision_date": "2026-11-01", "timezone": "Europe/Vienna",
                    "title": "Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans, Bounded Corrections, and Matched Evaluation",
                    "scope": "Coordinated standalone author source, reviewed PDFs, follow-up display assets, and preserved historical provenance",
                    "entry_points": ["main.tex", "supplement.tex"],
                    "pages": {entry: builds[entry]["pages"] for entry in builds},
                    "anonymous_review_upload": False,
                    "baseline_archive_sha256": expected,
                    "repository_parent_commit": snapshot["base_commit"],
                    "publication_branch": "research/kahm-crossfit-residual-2026-11-01",
                    "inventory_excludes": ["SOURCE_FILE_MANIFEST.json"],
                    "files": [{"path": r, **info(stage / r)} for r in sorted(files)]}
        write_json(stage / "SOURCE_FILE_MANIFEST.json", manifest)
        shutil.copyfile(stage / "SOURCE_FILE_MANIFEST.json", MANUSCRIPT / "SOURCE_FILE_MANIFEST.json")
        files.add("SOURCE_FILE_MANIFEST.json")
        archive = args.output_dir / "KAHM_Reconstruction_Prediction_Manuscript_Source.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as result:
            for relative in sorted(files):
                zinfo = zipfile.ZipInfo(PREFIX + relative, date_time=(2026, 11, 1, 0, 0, 0))
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                zinfo.external_attr = 0o100644 << 16
                result.writestr(zinfo, (stage / relative).read_bytes())
        with zipfile.ZipFile(archive) as result:
            if result.testzip() is not None:
                raise ValueError("Archive CRC verification failed")
            for record in manifest["files"]:
                data = result.read(PREFIX + record["path"])
                if hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError("Archive inventory verification failed")
        artifacts = []
        for name in [*delivered_names.values(), archive.name]:
            path = args.output_dir / name
            shutil.copyfile(path, INTEGRATION / "review" / name)
            artifacts.append({"path": name, **info(path)})
        release = {"schema": "kahm-coordinated-manuscript-release-v1", "date": "2026-11-01",
                   "timezone": "Europe/Vienna", "repository_parent_commit": snapshot["base_commit"],
                   "publication_branch": "research/kahm-crossfit-residual-2026-11-01",
                   "scope": "Coordinated manuscript and repository presentation of the original principal study and two distinct follow-ups",
                   "scientific_inputs_modified": False, "new_scientific_experiments": 0,
                   "primary_families": {"dimension": {"size": 6, "established": 3},
                                        "preprocessing": {"size": 4, "established": 1},
                                        "residual_targets": {"size": 4, "established": 2,
                                                             "additional_independent_repetitions": 0}},
                   "pages": manifest["pages"], "source_archive_members": len(files),
                   "artifacts": artifacts,
                   "current_verification": "../revision_evidence/coordinated_20261101/",
                   "previous_dimension_release": "DIMENSION_MANUSCRIPT_RELEASE.json"}
        write_json(INTEGRATION / "review/COORDINATED_MANUSCRIPT_RELEASE.json", release)
        print(json.dumps({"status": "passed", "pages": release["pages"],
                          "archive_members": len(files), "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
