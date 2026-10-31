#!/usr/bin/env python3
"""Package the revised manuscript and target-geometry evidence, without refits.

This release reuses path safety, byte descriptions, JSON encoding, manuscript
collection rules and fixed ZIP metadata from the historical utility packager.
Historical packagers and scientific freezes are never modified. The evidence
ZIP concerns only target geometry; the manuscript source also includes earlier
generated tables and figures needed to compile the complete integrated paper.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

from . import package_utility as shared


ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("extensions/target_geometry")
RUN = STUDY / "results/v1"
REPORTS = STUDY / "reports"
PAPER = Path("extensions/tmlr_integration/manuscript")
PACKAGER = Path("extensions/tmlr_integration/package_target.py")
HELPER = Path("extensions/tmlr_integration/package_utility.py")
PDF_NAME = "KAHM_Prediction_Revised.pdf"
SOURCE_NAME = "KAHM_Prediction_Revised_Manuscript_Source.zip"
REPRO_NAME = "KAHM_Target_Geometry_Reproduction.zip"
INVENTORY = shared.INVENTORY
REPOSITORY_URL = shared.REPOSITORY_URL
CSV_FILES = ("task_metrics.csv", "horizon_metrics.csv", "residual_identity.csv", "validation_trials.csv",
             "synthetic_means.csv", "primary_contrasts.csv", "natural_sources.csv", "natural_aggregate.csv",
             "selection_frequency.csv", "metric_diagnostics.csv", "failures.csv")
json_bytes, description = shared.json_bytes, shared.description
safe_name, read_source = shared.safe_name, shared.read_source


def checked_freeze(root):
    content = read_source(root, STUDY / "freeze.json")
    freeze = json.loads(content)
    for name, expected in freeze["files"].items():
        if description(read_source(root, name))["sha256"] != expected:
            raise ValueError(f"Scientific freeze changed: {name}")
    publication = json.loads(read_source(root, STUDY / "PUBLICATION.json"))
    freeze_sha = description(content)["sha256"]
    commit = publication.get("commit", "")
    if (publication.get("freeze_sha256") != freeze_sha or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or publication.get("url") != f"{REPOSITORY_URL}/commit/{commit}"):
        raise ValueError("Missing matching public execution checkpoint")
    return freeze, freeze_sha, publication


def check_summary(summary):
    if (summary.get("complete") is not True or summary.get("expected_tasks") != 48
            or summary.get("completed_tasks") != 48 or summary.get("expected_predictions") != 768
            or summary.get("audited_predictions") != 768 or len(summary.get("methods", [])) != 16
            or len(summary.get("primary_contrasts", [])) != 18 or summary.get("failures")):
        raise ValueError("This release requires the audited full 48-task, 768-forecast target-geometry roster")
    if any(row.get("n") != 12 or not row.get("complete") for row in summary["primary_contrasts"]):
        raise ValueError("A prespecified synthetic contrast lost a repetition")


def audit_results(root, run_dir, reports_dir):
    """Replay the frozen independent analysis into scratch, without forecasting fits."""
    root, run_dir, reports_dir = Path(root), Path(run_dir), Path(reports_dir)
    freeze, freeze_sha, publication = checked_freeze(root)
    original_bytes = read_source(reports_dir, "summary.json")
    original = json.loads(original_bytes)
    check_summary(original)
    execution_bytes = read_source(run_dir, "execution.json")
    execution = json.loads(execution_bytes)
    if dt.datetime.fromisoformat(publication["recorded_at_utc"]) > dt.datetime.fromisoformat(execution["created_at_utc"]):
        raise ValueError("Publication record follows the confirmatory execution start")
    if (execution["freeze_sha256"] != freeze_sha or original["source_hashes"]["freeze"] != freeze_sha
            or original["source_hashes"]["execution"] != description(execution_bytes)["sha256"]):
        raise ValueError("Published snapshot, executed study and audited report disagree")
    with tempfile.TemporaryDirectory(prefix="kahm_target_reaudit_") as temporary:
        result = subprocess.run(
            [sys.executable, "-m", "extensions.target_geometry.analysis", "--run-dir", str(run_dir.resolve()),
             "--output", str(Path(temporary) / "reports")], cwd=root, check=False,
            capture_output=True, text=True,
        )
        if result.returncode:
            raise ValueError(f"Independent saved-prediction audit failed:\n{result.stdout}\n{result.stderr}")
        replay_root = Path(temporary) / "reports"
        replay = json.loads(read_source(replay_root, "summary.json"))
        check_summary(replay)
        # The reporting timestamp is the only permitted replay difference.
        original_comparable = {key: value for key, value in original.items() if key != "generated_at_utc"}
        replay_comparable = {key: value for key, value in replay.items() if key != "generated_at_utc"}
        if original_comparable != replay_comparable:
            raise ValueError("Independent analysis replay differs from the supplied release report")
        for name in CSV_FILES:
            if read_source(reports_dir, name) != read_source(replay_root, name):
                raise ValueError(f"Detailed report differs from the independent replay: {name}")
    return {"status": "passed", "forecasting_fits": 0, "complete_tasks": 48,
            "audited_predictions": 768, "methods_per_task": 16, "primary_contrasts": 18,
            "repetitions_per_contrast": 12, "fallback_predictions": original["fallback_predictions"],
            "missing_or_failed_tasks": 0, "summary_sha256": description(original_bytes)["sha256"],
            "execution_sha256": description(execution_bytes)["sha256"],
            "scope": "Frozen prediction/data/cost audit and exact report replay; no independent refitting of validation candidate models."}


def source_readme():
    return f"""# Complete editable revised manuscript

This is the public, unsubmitted KAHM prediction manuscript, including the new
target-geometry study and the preserved earlier studies. It is not a journal
submission, acceptance artifact, or anonymous-review export.

All editable LaTeX/BibTeX files, template/style licenses, and generated tables and
figures required to compile the complete paper are included. Compilation
byproducts and the final PDF are excluded from this source ZIP.

With a TeX distribution containing latexmk and the standard LaTeX packages:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The companion {REPRO_NAME} contains the new target-geometry evidence and its
frozen numerical dependencies. Historical generated tables and figures are
included here for compilation; the historical complete experimental evidence
remains at {REPOSITORY_URL} and is not replaced by that companion ZIP.

ARTIFACT_MANIFEST.json records each included file's byte count and SHA-256.
LICENSE, DATA_LICENSE, and retained third-party/template notices apply to their
respective content. No new DOI, external registration, submission, or author
approval is implied by this packaging operation.
"""


def reproduction_readme():
    return f"""# Target-geometry extension: complete saved-evidence reproduction bundle

**Scope: the new target-geometry extension only.** The bundle contains its
scientific freeze and public checkpoint record; every frozen transitive source
dependency; original source ZIPs with pinned official metadata and attribution;
12 prepared natural series; all 48 result records and saved prediction archives;
the complete independent audit reports; and the downstream rendering code.
There are 36 synthetic tasks (three established laws, 12 fresh repetitions each)
and 12 archive tasks (three series from each of four new collections), with
16 declared forecasting procedures per task and 768 final forecasts in total.
Any model-level baseline fallbacks remain explicit in the saved records and
retain their incurred computation. Packaging never refits or replaces them.

Earlier controlled, natural, scaling, and forecasting-utility evidence remains
in {REPOSITORY_URL}. This bundle is not a replacement export for those studies.
The separate complete manuscript-source ZIP includes their generated tables and
figures so that the entire revised paper compiles without regenerating them.

## Audit saved evidence without forecasting fits

From the extracted root, use Python 3.12:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r extensions/target_geometry/requirements.txt
python -m extensions.target_geometry.freeze verify
python -m extensions.target_geometry.data verify
python -m extensions.target_geometry.analysis \\
  --run-dir extensions/target_geometry/results/v1 \\
  --output /tmp/kahm-target-independent-audit
python -m extensions.tmlr_integration.package_target audit \\
  --run-dir extensions/target_geometry/results/v1 \\
  --reports-dir extensions/target_geometry/reports
```

The audit reconstructs pinned inputs and losses from saved predictions; checks
the exact roster, hashes, donor episode/temporal exclusions, metric proxy,
residual identities, zero-correction equality, validation-grid minima and
complete charged costs; and recomputes paired intervals and source means.
It does not independently refit each validation candidate. The packaging audit
repeats this frozen analysis into scratch and requires exact agreement with
the supplied summary (apart from its generation timestamp) and every report CSV.
Original result archives and supplied reports are preserved.

To verify an intact ZIP's byte inventory without model fitting:

```bash
python -m extensions.tmlr_integration.package_target verify \\
  /path/to/{REPRO_NAME}
```

## Optional full refit into a new directory

The primary procedure performs no pretest refit on validation observations.
The following command reruns the entire prespecified study from its training
data; it is distinct from auditing already saved evidence:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python -m extensions.target_geometry.experiment \\
  --phase all --output /tmp/kahm-target-independent-run
python -m extensions.target_geometry.analysis \\
  --run-dir /tmp/kahm-target-independent-run \\
  --output /tmp/kahm-target-independent-refit-audit
```

Use a new output path. The runner refuses to overwrite an existing run without
explicit resume. Refit results and hardware timings can differ; the original
frozen evidence remains untouched. Component costs include required baseline,
donor, metric, geometry and validation searches plus prediction/serialization;
they exclude imports, downloads, data preparation, artifact I/O and reporting.

## Rebuild publication assets and run engineering checks

```bash
python -m extensions.target_geometry.build_assets \\
  --reports extensions/target_geometry/reports \\
  --output /tmp/kahm-target-regenerated-assets
python -m unittest extensions.target_geometry.test_models \\
  extensions.target_geometry.test_protocol \\
  extensions.target_geometry.test_analysis -v
```

Tests use fabricated data. Assets are computed from the audited reports and do
not select new methods or refit forecasts. Synthetic inference uses only the
18 declared paired Student contrasts with Bonferroni allocation and a normality
approximation. Archive results are descriptive equal-collection means across
four selected collections; 12 series or repeated origins do not create extra
independent domains. The learned metric is a shared kernel-regression proxy,
not end-to-end KAHM learning or evidence of general predictive superiority.

## Provenance and rights

All four official Zenodo source records declare CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/); their complete
creator attribution, source versions, DOIs, original-source links, compressed
archive hashes and TSF-member hashes are in
extensions/target_geometry/sources/manifest.json and the four metadata records.
Prepared windows and selected-series provenance are in prepared/roster.json.
The Bitcoin tasks predict three activity indicators, not prices. The four
collections are new to these studies but can share broad application domains
with older collections. The custom capped-window protocol does not reproduce
the Monash leaderboard protocol.

LICENSE covers the repository's code; DATA_LICENSE covers original research
outputs. Preserved upstream licenses and THIRD_PARTY_NOTICES.md apply to their
respective dependencies. ARTIFACT_MANIFEST.json inventories all bytes and hashes.
PUBLICATION.json documents a public source checkpoint before execution; it is
not an external registry preregistration. No invented DOI, journal submission,
acceptance, or author sign-off is implied.
"""


def manuscript_payload(root):
    files = shared.manuscript_payload(root)
    required = ("section_target_geometry.tex", "appendix_target_geometry.tex",
                "generated/target_geometry/synthetic_main.tex", "generated/target_geometry/archive_main.tex",
                "generated/target_geometry/numbers.tex", "generated/target_geometry/assets.json")
    for name in required:
        if name not in files:
            raise ValueError(f"Missing revised manuscript input: {name}")
    files["README_RELEASE.md"] = source_readme().encode()
    return files


def reproduction_payload(root, run_dir, reports_dir, freeze):
    files = {name: read_source(root, name) for name in freeze["files"]}
    for path in sorted((root / STUDY).iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt", ".cff"}:
            name = path.relative_to(root).as_posix()
            files[name] = read_source(root, name)
    # Original compressed public sources are small enough to preserve offline.
    source_manifest = json.loads(read_source(root, STUDY / "sources/manifest.json"))
    for source in source_manifest["sources"]:
        archive = source["archive"]
        name = (STUDY / "sources" / archive["filename"]).as_posix()
        content = read_source(root, name)
        if description(content) != {"bytes": archive["bytes"], "sha256": archive["sha256"]}:
            raise ValueError(f"Original source archive changed: {name}")
        files[name] = content
    # The historical source inventories substantiate collection-level exclusions.
    for item in source_manifest["historical_exclusions"]:
        content = read_source(root, item["path"])
        if description(content)["sha256"] != item["sha256"]:
            raise ValueError("Historical collection-exclusion inventory changed")
        files[item["path"]] = content
    for origin, destination in ((run_dir, RUN), (reports_dir, REPORTS)):
        for path in sorted(origin.rglob("*")):
            if path.is_symlink():
                raise ValueError("Symbolic link in saved evidence")
            relative = path.relative_to(origin)
            if (not path.is_file() or any(part.startswith(".") or part in shared.SKIP_PARTS
                                          for part in relative.parts)):
                continue
            files[(destination / relative).as_posix()] = read_source(origin, relative)
    for name in ("LICENSE", "DATA_LICENSE", "CITATION.cff",
                 "extensions/predictive_geometry/LICENSE",
                 "extensions/predictive_geometry/THIRD_PARTY_NOTICES.md",
                 "extensions/predictive_geometry/study/upstream_manifest.json",
                 "extensions/predictive_geometry/study/vendor/upstream/LICENSE",
                 PACKAGER.as_posix(), HELPER.as_posix()):
        files[name] = read_source(root, name)
    for name in list(files):
        for parent in Path(name).parents:
            initializer = parent / "__init__.py"
            if (root / initializer).is_file():
                files[initializer.as_posix()] = read_source(root, initializer)
    files["README.md"] = reproduction_readme().encode()
    return files


def write_archive(path, files, *, kind, freeze_sha, publication, audit):
    if INVENTORY in files:
        raise ValueError("Reserved inventory name in payload")
    inventory = {"schema_version": 1, "kind": kind,
                 "scope": "Public unsubmitted revised manuscript; target-geometry evidence only in the reproduction ZIP.",
                 "scientific_freeze_sha256": freeze_sha, "public_execution_commit": publication["commit"],
                 "evidence_audit": audit,
                 "files": {name: description(content) for name, content in sorted(files.items())}}
    payload = {**files, INVENTORY: json_bytes(inventory)}
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name, content in sorted(payload.items()):
            safe_name(name)
            info = zipfile.ZipInfo(name, date_time=shared.ZIP_DATE)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED if name.endswith((".npz", ".pdf", ".png", ".zip")) else zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compresslevel=9)
    return verify_archive(path)


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) > shared.MAX_ENTRIES or len(names) != len(set(names)) or INVENTORY not in names:
            raise ValueError("Invalid archive count, duplicate names, or missing manifest")
        if sum(entry.file_size for entry in entries) > shared.MAX_BYTES:
            raise ValueError("Archive exceeds declared uncompressed size limit")
        for entry in entries:
            safe_name(entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                raise ValueError("Directories, links and encrypted entries are not release inputs")
        inventory = json.loads(archive.read(INVENTORY))
        if set(names) != set(inventory["files"]) | {INVENTORY}:
            raise ValueError("Archive inventory is incomplete or has extra files")
        for name, expected in inventory["files"].items():
            if description(archive.read(name)) != expected:
                raise ValueError(f"Archive content changed: {name}")
        if inventory["kind"] == "target_geometry_reproduction":
            freeze_bytes = archive.read((STUDY / "freeze.json").as_posix())
            if description(freeze_bytes)["sha256"] != inventory["scientific_freeze_sha256"]:
                raise ValueError("Packaged scientific freeze differs from manifest")
            for name, expected in json.loads(freeze_bytes)["files"].items():
                if name not in inventory["files"] or inventory["files"][name]["sha256"] != expected:
                    raise ValueError(f"Frozen transitive input omitted or changed: {name}")
            publication = json.loads(archive.read((STUDY / "PUBLICATION.json").as_posix()))
            if (publication["commit"] != inventory["public_execution_commit"]
                    or publication["freeze_sha256"] != inventory["scientific_freeze_sha256"]):
                raise ValueError("Packaged public checkpoint does not match the freeze")
            summary_bytes = archive.read((REPORTS / "summary.json").as_posix())
            summary = json.loads(summary_bytes)
            check_summary(summary)
            if description(summary_bytes)["sha256"] != inventory["evidence_audit"]["summary_sha256"]:
                raise ValueError("Packaged report differs from independently audited report")
            execution = json.loads(archive.read((RUN / "execution.json").as_posix()))
            expected_ids = [task["task_id"] for task in execution["tasks"]]
            records = [name for name in names if name.startswith(RUN.as_posix() + "/") and Path(name).name == "result.json"]
            archives = [name for name in names if name.startswith(RUN.as_posix() + "/") and Path(name).name == "predictions.npz"]
            if (len(records) != 48 or len(archives) != 48
                    or {Path(name).parent.name for name in records} != set(expected_ids)
                    or {Path(name).parent.name for name in archives} != set(expected_ids)):
                raise ValueError("Reproduction ZIP omitted or added a declared result archive")
            forecasts, fallbacks = 0, 0
            for task in execution["tasks"]:
                base = RUN / safe_name(task["task_id"])
                record = json.loads(archive.read((base / "result.json").as_posix()))
                if record["task"] != task or set(record["methods"]) != set(execution["methods"]):
                    raise ValueError("Packaged task/method roster changed")
                prediction_bytes = archive.read((base / "predictions.npz").as_posix())
                if description(prediction_bytes)["sha256"] != record["predictions_sha256"]:
                    raise ValueError("Packaged predictions differ from their result record")
                with zipfile.ZipFile(io.BytesIO(prediction_bytes)) as predictions:
                    predicted = [name.removeprefix("prediction__").removesuffix(".npy")
                                 for name in predictions.namelist() if name.startswith("prediction__")]
                if len(predicted) != 16 or set(predicted) != set(execution["methods"]):
                    raise ValueError("A saved archive lacks one of the 16 declared forecasts")
                forecasts += len(predicted)
                fallbacks += sum(method["status"] == "fallback" for method in record["methods"].values())
            if forecasts != 768 or fallbacks != summary["fallback_predictions"]:
                raise ValueError("Forecast or fallback accounting changed in the ZIP")
        elif inventory["kind"] != "manuscript_source":
            raise ValueError("Unknown artifact kind")
    return {"status": "passed", "kind": inventory["kind"], "files": len(inventory["files"]),
            **description(Path(path).read_bytes())}


def build(pdf, output, run_dir, reports_dir):
    root, output = ROOT.resolve(), Path(output).resolve()
    run_dir, reports_dir = Path(run_dir).resolve(), Path(reports_dir).resolve()
    freeze, freeze_sha, publication = checked_freeze(root)
    pdf = Path(pdf).resolve()
    pdf_bytes = read_source(pdf.parent, pdf.name)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Supplied manuscript is not a compiled PDF")
    audit = audit_results(root, run_dir, reports_dir)
    source, reproduction = manuscript_payload(root), reproduction_payload(root, run_dir, reports_dir, freeze)
    with tempfile.TemporaryDirectory(prefix="kahm_target_release_") as temporary:
        temporary = Path(temporary)
        (temporary / PDF_NAME).write_bytes(pdf_bytes)
        write_archive(temporary / SOURCE_NAME, source, kind="manuscript_source", freeze_sha=freeze_sha,
                      publication=publication, audit=audit)
        write_archive(temporary / REPRO_NAME, reproduction, kind="target_geometry_reproduction", freeze_sha=freeze_sha,
                      publication=publication, audit=audit)
        manifest = {"schema_version": 1,
                    "scope": "Public, unsubmitted revised manuscript; complete new target-geometry evidence; historical full evidence remains in the repository.",
                    "source_repository": REPOSITORY_URL, "public_execution_commit": publication["commit"],
                    "scientific_freeze_sha256": freeze_sha, "evidence_audit": audit,
                    "artifacts": {name: description((temporary / name).read_bytes())
                                  for name in (PDF_NAME, SOURCE_NAME, REPRO_NAME)}}
        (temporary / "RELEASE_MANIFEST.json").write_bytes(json_bytes(manifest))
        output.mkdir(parents=True, exist_ok=True)
        products = sorted(temporary.iterdir())
        for product in products:
            destination = output / product.name
            if ((destination.exists() or destination.is_symlink())
                    and (destination.is_symlink() or not destination.is_file() or destination.read_bytes() != product.read_bytes())):
                raise FileExistsError(f"Existing different release output is preserved: {destination}")
        for product in products:
            if not (output / product.name).exists():
                shutil.copyfile(product, output / product.name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--pdf", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--run-dir", type=Path, default=ROOT / RUN)
    build_parser.add_argument("--reports-dir", type=Path, default=ROOT / REPORTS)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("archive", type=Path)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--run-dir", type=Path, default=ROOT / RUN)
    audit_parser.add_argument("--reports-dir", type=Path, default=ROOT / REPORTS)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.pdf, args.output, args.run_dir, args.reports_dir)
    elif args.command == "verify":
        result = verify_archive(args.archive)
    else:
        result = audit_results(ROOT, args.run_dir, args.reports_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
