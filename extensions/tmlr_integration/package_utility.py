#!/usr/bin/env python3
"""Package the public forecasting-utility research release, without model fits.

The reproduction ZIP contains the new extension and its exact frozen numerical
dependencies, prepared inputs, and expanded results for all attempted tasks. Historical study
archives remain in the public repository; this is not their replacement export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("extensions/forecasting_utility")
PAPER = Path("extensions/tmlr_integration/manuscript")
PACKAGER = Path("extensions/tmlr_integration/package_utility.py")
RUN = STUDY / "results/utility_v1"
PDF_NAME = "KAHM_Forecasting_Utility_Manuscript.pdf"
SOURCE_NAME = "KAHM_Forecasting_Utility_Manuscript_Source.zip"
REPRO_NAME = "KAHM_Forecasting_Utility_Reproduction.zip"
INVENTORY = "ARTIFACT_MANIFEST.json"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
MAX_ENTRIES = 30_000
MAX_BYTES = 2 * 1024**3
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", "cache", "reference"}
SOURCE_SUFFIXES = {".tex", ".bib", ".sty", ".bst", ".txt", ".md", ".json", ".csv"}
FIGURE_SUFFIXES = {".pdf", ".png", ".svg", ".eps"}
REPOSITORY_URL = "https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine"


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def description(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or ":" in path.parts[0]):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return name


def read_source(root, name):
    root = Path(root).resolve()
    name = safe_name(Path(name).as_posix())
    path = root / name
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Missing or escaped release input: {name}")
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError(f"Symbolic links are not release inputs: {name}")
    return path.read_bytes()


def checked_freeze(root):
    content = read_source(root, STUDY / "freeze.json")
    freeze = json.loads(content)
    for name, expected in freeze["files"].items():
        if description(read_source(root, name))["sha256"] != expected:
            raise ValueError(f"Scientific freeze changed: {name}")
    for name, expected in ((STUDY / "PROTOCOL.md", freeze["protocol_sha256"]),
                           (STUDY / "prepared/roster.json", freeze["roster_sha256"])):
        if description(read_source(root, name))["sha256"] != expected:
            raise ValueError(f"Protocol or roster freeze changed: {name}")
    return freeze, description(content)["sha256"]


def audit_results(root, run_dir):
    """Recompute saved predictions and selections; never fit a forecasting model."""
    checked_freeze(root)
    original = subprocess.run(
        [sys.executable, "-m", "extensions.forecasting_utility.analysis", "--run-dir", str(run_dir)],
        cwd=root, check=False, capture_output=True, text=True,
    )
    # Exit 1 is the original audit's honest scientific-completeness result.
    # Require its successful JSON audit output as well: a traceback/error is
    # never accepted merely because it shares the same process exit code.
    original_output = json.loads(original.stdout)
    if (original.returncode != 1 or original_output.get("complete") is not False
            or original_output.get("expected_tasks") != 168 or original_output.get("observed_tasks") != 168):
        raise ValueError("Unexpected original analysis result for the preserved failed reference run")
    subprocess.run(
        [sys.executable, "-m", "extensions.forecasting_utility.reporting", "--run-dir", str(run_dir)],
        cwd=root, check=True, capture_output=True, text=True,
    )
    summary = json.loads(read_source(run_dir, "report/summary.json"))
    if (summary.get("verification_passed") is not True or summary.get("all_tasks_attempted") is not True
            or summary.get("all_methods_succeeded") is not False):
        raise ValueError("Release requires verified complete accounting with the original failures retained")
    cfg = json.loads(read_source(root, STUDY / "config.json"))
    execution = json.loads(read_source(run_dir, "execution.json"))
    predictions, failures, policies, policy_failures = 0, 0, 0, 0
    failed_tasks = []
    if len(execution["expected_tasks"]) != 168:
        raise ValueError("Reference run does not contain all 168 declared attempts")
    for task in execution["expected_tasks"]:
        relative = Path("tasks") / task["task_id"]
        record = json.loads(read_source(run_dir, relative / "result.json"))
        if [m["method"] for m in record["methods"]] != cfg["methods"]:
            raise ValueError("Incomplete attempted method roster in release")
        if not record["complete"]:
            failed_tasks.append(task["task_id"])
        for method in record["methods"]:
            if method["status"] == "ok":
                read_source(run_dir, relative / safe_name(method["prediction_file"]))
                predictions += 1
            elif method["status"] == "failed" and method.get("error"):
                failures += 1
            else:
                raise ValueError("A method attempt lacks a successful forecast or explicit failure")
        if set(record["policies"]) != set(cfg["policies"]):
            raise ValueError("Incomplete attempted policy roster")
        for policy in record["policies"].values():
            policies += 1
            if policy["status"] == "failed" and policy.get("error"):
                policy_failures += 1
            elif policy["status"] != "ok":
                raise ValueError("A policy attempt lacks an outcome or explicit failure")
    if (predictions, failures, policies, policy_failures, len(failed_tasks)) != (1336, 8, 336, 4, 2):
        raise ValueError("Reference accounting must preserve 1,336 forecasts, eight model failures, and four policy failures on two tasks")
    return {"status": "verification_passed", "all_tasks_attempted": True, "all_methods_succeeded": False,
            "task_count": 168, "method_attempts": predictions + failures,
            "final_method_forecasts": predictions, "method_failures": failures,
            "attempted_policies": policies, "failed_policies": policy_failures, "failed_tasks": failed_tasks,
            "forecasting_fits": 0, "original_analysis_exit_code": original.returncode,
            "analysis_summary_sha256": description(read_source(run_dir, "analysis/summary.json"))["sha256"],
            "failure_aware_summary_sha256": description(read_source(run_dir, "report/summary.json"))["sha256"]}


def source_readme():
    return """# Editable manuscript source

This is the public, unsubmitted forecasting-utility research manuscript. It is
not a journal submission or acceptance artifact. All editable LaTeX/BibTeX,
required style files and notices, and generated table/figure inputs are present.
Compilation byproducts and the final compiled manuscript are excluded here.

With a TeX distribution containing latexmk and the standard LaTeX packages:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The companion KAHM_Forecasting_Utility_Reproduction.zip audits the new extension.
The paper also reports preserved historical studies; their original complete
evidence remains in the public repository. This source ZIP contains the already
generated historical tables and figures needed to compile the complete paper.
ARTIFACT_MANIFEST.json records every included file's size and SHA-256 checksum.
Repository code, output, and third-party template notices remain applicable.
"""


def reproduction_readme():
    return f"""# Forecasting-utility extension: complete audit and reproduction bundle

**Scope: the new forecasting-utility extension only.** This bundle contains its
exact scientific freeze, publication record, code and imported numerical
dependencies, pinned source metadata, selected prepared arrays, and all expanded
results for all 168 attempted tasks: 1,336 saved method forecasts and eight
preserved model-fit failures (plus four dependent policy failures). The failures
affect the two capacities of one Car Parts series; they are not refitted,
replaced, or silently excluded. It does not replace the original
controlled, eight-source, or post-hoc scaling archives. Those remain available at
{REPOSITORY_URL}.

Solar Weekly is aggregated simulated NREL data. Archive results are descriptive
over four selected collections. See the protocol for the scope of the controlled
paired intervals and the limits of validation-selected corrections.

## Audit without fitting forecasting models

From this extracted directory, use Python 3.12:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r extensions/forecasting_utility/requirements.txt
python -m extensions.forecasting_utility.reporting \\
  --run-dir extensions/forecasting_utility/results/utility_v1
python -m extensions.forecasting_utility.independent_audit
```

The full saved result tree is already expanded. Do not run `reference restore`
here: the repository's split transport archives are deliberately omitted to avoid
duplicating the same results. The included reference.py remains unchanged because
it belongs to the original execution freeze.

The report calls the frozen analysis to check inputs, exact task/method/policy rosters, saved
target/origin/prediction hashes, candidate selections, losses and timing unions.
It writes summary.json and detailed CSV tables beneath the run's report folder.
Its successful exit means that every attempt is accounted for, not that every
model fit succeeded. The unchanged original analysis command exits 1 because two
tasks have model failures; its original incomplete summary remains in analysis/.
The independent checker reads the raw saved arrays and JSON records directly,
without importing that analysis or reporting code, and recomputes prediction
losses, paired intervals, source weights, and exact gamma-zero equalities.
Archive comparisons on 22 common-support tasks are labeled diagnostic. Separate
full-roster baseline and coverage tables retain the complete 24-task accounting.
[REPORTING_AMENDMENT.md](extensions/forecasting_utility/REPORTING_AMENDMENT.md)
explains this reporting-only response to the observed failure.

## Refit the prespecified study into a fresh output directory

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python -m extensions.forecasting_utility.experiment \\
  --phase all --output-dir /tmp/kahm-utility-independent-run
python -m extensions.forecasting_utility.analysis \\
  --run-dir /tmp/kahm-utility-independent-run
```

Refitting is optional and incurs the full study cost. Original results are never
overwritten. Hardware-specific timings can differ. Model-only costs exclude data
preparation and artifact I/O; policy costs charge the union of measured required
components, including rejected correction searches.

## Engineering tests and example

```bash
python -m pip install pytest==9.2.0
python -m pytest extensions/forecasting_utility/test_models.py \\
  extensions/forecasting_utility/test_protocol.py -q
python examples/forecasting_utility.py
```

Tests and the example use toy data, separate from the research outcomes.

ARTIFACT_MANIFEST.json inventories every file with a byte count and SHA-256.
LICENSE covers code; DATA_LICENSE covers original research outputs. Dataset
records, THIRD_PARTY_NOTICES.md, and retained upstream/template licenses preserve
third-party attribution. This is a public research distribution, not an anonymous
review export and not a new external registration. No author approval, journal
submission, acceptance, or invented DOI is implied.
"""


def manuscript_payload(root):
    files = {}
    manuscript = root / PAPER
    for path in sorted(manuscript.rglob("*")):
        relative = path.relative_to(manuscript)
        if any(part in SKIP_PARTS or part.startswith(".") for part in relative.parts):
            continue
        # Keep historical review-build guidance and QA records in the repository;
        # this public release supplies its own accurate compilation instructions.
        if not path.is_file() or path.name in {"main.pdf", "BUILD_CHECK.json", "README_BUILD.md"}:
            continue
        if path.suffix in SOURCE_SUFFIXES or ("generated" in relative.parts and path.suffix in FIGURE_SUFFIXES):
            files[relative.as_posix()] = read_source(manuscript, relative)
    for name in ("main.tex", "references.bib", "tmlr.sty", "tmlr.bst", "fancyhdr.sty", "LICENSE_TMLR.txt", "generated/utility/numbers.tex"):
        if name not in files:
            raise ValueError(f"Missing complete manuscript input: {name}")
    for name in ("LICENSE", "DATA_LICENSE"):
        files[name] = read_source(root, name)
    files["README_RELEASE.md"] = source_readme().encode()
    return files


def reproduction_payload(root, run_dir, freeze):
    files = {name: read_source(root, name) for name in freeze["files"]}
    # Preserve any package initializers that the complete repository supplies;
    # absent initializers remain valid Python namespace packages.
    for name in list(files):
        for parent in Path(name).parents:
            initializer = parent / "__init__.py"
            if (root / initializer).is_file():
                files[initializer.as_posix()] = read_source(root, initializer)
    for path in sorted((root / STUDY).iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt", ".cff"}:
            name = path.relative_to(root)
            files[name.as_posix()] = read_source(root, name)
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symbolic link in saved outcomes")
        if path.is_file():
            relative = path.relative_to(run_dir)
            if any(part.startswith(".") or part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            files[(RUN / relative).as_posix()] = read_source(run_dir, relative)
    for name in ("LICENSE", "DATA_LICENSE", "CITATION.cff", "examples/forecasting_utility.py",
                 "extensions/forecasting_utility/REPORTING_AMENDMENT.md",
                 "extensions/forecasting_utility/CITATION.cff",
                 "extensions/predictive_geometry/LICENSE",
                 "extensions/predictive_geometry/THIRD_PARTY_NOTICES.md",
                 "extensions/predictive_geometry/study/upstream_manifest.json",
                 "extensions/predictive_geometry/study/vendor/upstream/LICENSE"):
        files[name] = read_source(root, name)
    files[PACKAGER.as_posix()] = read_source(root, PACKAGER)
    files["README.md"] = reproduction_readme().encode()
    return files


def write_archive(path, files, *, kind, freeze_sha, publication, audit):
    if INVENTORY in files:
        raise ValueError("Reserved inventory name in payload")
    inventory = {"schema_version": 1, "kind": kind, "scope": "Public research release; new extension evidence only, with complete editable integrated manuscript in its separate source ZIP.",
                 "scientific_freeze_sha256": freeze_sha,
                 "public_execution_commit": publication["initial_execution_commit"],
                 "evidence_audit": audit,
                 "files": {name: description(content) for name, content in sorted(files.items())}}
    payload = {**files, INVENTORY: json_bytes(inventory)}
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name, content in sorted(payload.items()):
            safe_name(name)
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED if name.endswith((".npz", ".pdf", ".png")) else zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compresslevel=9)
    verify_archive(path)


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) > MAX_ENTRIES or len(names) != len(set(names)) or INVENTORY not in names:
            raise ValueError("Invalid archive entry count, duplicate names, or missing manifest")
        if sum(entry.file_size for entry in entries) > MAX_BYTES:
            raise ValueError("Archive exceeds the declared uncompressed size limit")
        for entry in entries:
            safe_name(entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                raise ValueError("Directories, links, and encrypted entries are not release payloads")
        inventory = json.loads(archive.read(INVENTORY))
        if set(names) != set(inventory["files"]) | {INVENTORY}:
            raise ValueError("Archive inventory is incomplete or has extra files")
        for name, expected in inventory["files"].items():
            if description(archive.read(name)) != expected:
                raise ValueError(f"Archive content changed: {name}")
        if inventory["kind"] == "forecasting_utility_reproduction":
            freeze_name = (STUDY / "freeze.json").as_posix()
            freeze_bytes = archive.read(freeze_name)
            if description(freeze_bytes)["sha256"] != inventory["scientific_freeze_sha256"]:
                raise ValueError("Packaged scientific freeze differs from manifest")
            freeze = json.loads(freeze_bytes)
            for name, expected in freeze["files"].items():
                if name not in inventory["files"] or inventory["files"][name]["sha256"] != expected:
                    raise ValueError(f"Frozen dependency omitted or changed in ZIP: {name}")
            if any(name.startswith((STUDY / "reference").as_posix() + "/") for name in names):
                raise ValueError("Duplicate split transport archives are not part of this expanded reproduction bundle")
            summary = json.loads(archive.read((RUN / "report/summary.json").as_posix()))
            if (summary.get("verification_passed") is not True or summary.get("all_tasks_attempted") is not True
                    or summary.get("all_methods_succeeded") is not False):
                raise ValueError("Packaged report does not preserve complete accounting and explicit scientific failures")
            prediction_names = [name for name in names if name.startswith((RUN / "tasks").as_posix() + "/") and Path(name).name.startswith("prediction_") and name.endswith(".npz")]
            if len(prediction_names) != 1336:
                raise ValueError("Reproduction ZIP must contain all 1,336 successful final method forecasts")
            results = [name for name in names if name.startswith((RUN / "tasks").as_posix() + "/") and Path(name).name == "result.json"]
            records = [json.loads(archive.read(name)) for name in results]
            if (len(records) != 168 or sum(m["status"] == "failed" for r in records for m in r["methods"]) != 8
                    or sum(p["status"] == "failed" for r in records for p in r["policies"].values()) != 4):
                raise ValueError("Reproduction ZIP lost a task or preserved failure record")
        elif inventory["kind"] != "manuscript_source":
            raise ValueError("Unknown release artifact kind")
    return {"status": "passed", "kind": inventory["kind"], "files": len(inventory["files"]),
            "bytes": Path(path).stat().st_size, "sha256": description(Path(path).read_bytes())["sha256"]}


def build(pdf, output, run_dir):
    root, output, run_dir = ROOT.resolve(), Path(output).resolve(), Path(run_dir).resolve()
    freeze, freeze_sha = checked_freeze(root)
    publication = json.loads(read_source(root, STUDY / "PUBLICATION.json"))
    if publication["freeze_sha256"] != freeze_sha or not publication.get("initial_execution_commit"):
        raise ValueError("Missing matching public execution checkpoint")
    pdf = Path(pdf).resolve()
    pdf_bytes = read_source(pdf.parent, pdf.name)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The supplied manuscript is not a compiled PDF")
    audit = audit_results(root, run_dir)
    source = manuscript_payload(root)
    reproduction = reproduction_payload(root, run_dir, freeze)
    with tempfile.TemporaryDirectory(prefix="kahm_utility_release_") as temporary:
        temporary = Path(temporary)
        (temporary / PDF_NAME).write_bytes(pdf_bytes)
        write_archive(temporary / SOURCE_NAME, source, kind="manuscript_source", freeze_sha=freeze_sha, publication=publication, audit=audit)
        write_archive(temporary / REPRO_NAME, reproduction, kind="forecasting_utility_reproduction", freeze_sha=freeze_sha, publication=publication, audit=audit)
        manifest = {"schema_version": 1, "scope": "Public, unsubmitted forecasting-utility manuscript and new-extension reproduction bundle; historical full archives remain in the source repository.",
                    "source_repository": REPOSITORY_URL, "public_execution_commit": publication["initial_execution_commit"],
                    "scientific_freeze_sha256": freeze_sha, "evidence_audit": audit,
                    "artifacts": {name: description((temporary / name).read_bytes()) for name in (PDF_NAME, SOURCE_NAME, REPRO_NAME)}}
        (temporary / "RELEASE_MANIFEST.json").write_bytes(json_bytes(manifest))
        output.mkdir(parents=True, exist_ok=True)
        products = sorted(temporary.iterdir())
        for product in products:
            destination = output / product.name
            if (destination.exists() or destination.is_symlink()) and (destination.is_symlink() or not destination.is_file() or destination.read_bytes() != product.read_bytes()):
                raise FileExistsError(f"Existing different release output is preserved: {destination}")
        for product in products:
            destination = output / product.name
            if not destination.exists():
                shutil.copyfile(product, destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--pdf", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--run-dir", type=Path, default=ROOT / RUN)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("archive", type=Path)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--run-dir", type=Path, default=ROOT / RUN)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.pdf, args.output, args.run_dir)
    elif args.command == "verify":
        result = verify_archive(args.archive)
    else:
        result = audit_results(ROOT, args.run_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
