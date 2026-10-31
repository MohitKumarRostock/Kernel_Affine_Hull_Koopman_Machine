#!/usr/bin/env python3
"""Package completed reconstruction-mechanism findings without forecast refits.

This post-execution utility is deliberately outside the scientific freeze.
It never imports the forecasting implementation or unpickles fitted states.
The only numerical subprocess is the new study's frozen independent analyzer;
historical studies, their auditors and their CPU-sensitive replays are untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("extensions/reconstruction_mechanism")
PAPER = Path("extensions/tmlr_integration/manuscript")
ASSETS = Path("generated/reconstruction_mechanism")
ASSET_BUILDER = Path("extensions/tmlr_integration/build_mechanism_assets.py")
PDF_NAME = "KAHM_Reconstruction_Prediction_Manuscript.pdf"
SOURCE_NAME = "KAHM_Reconstruction_Prediction_Manuscript_Source.zip"
REPRO_NAME = "KAHM_Reconstruction_Mechanism_Reproduction.zip"
RELEASE_NAME = "RELEASE_MANIFEST.json"
INVENTORY = "ARTIFACT_MANIFEST.json"
PRODUCTS = (PDF_NAME, SOURCE_NAME, REPRO_NAME, RELEASE_NAME)
REPOSITORY_URL = "https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine"
ROADMAP_COMMIT = "a2be8b5684e2a8da6f58177bc0b8625d081c2056"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
MAX_ENTRIES = 30000
MAX_BYTES = 4 * 1024 ** 3
SKIP_PARTS = {"__pycache__", ".pytest_cache", "build", "dist", "release", "releases",
              "development_inputs", "development_results", "review", "cache", "evidence_parts"}
SOURCE_SUFFIXES = {".tex", ".bib", ".sty", ".bst", ".txt", ".md", ".json", ".csv"}
FIGURE_SUFFIXES = {".pdf", ".png", ".svg", ".eps"}
DECISIONS = {"ADVANCE_TO_APPLICATION", "STOP_FORECASTING_PERFORMANCE_BRANCH"}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant {value!r} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError(f"Unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\0" in name
            or any(part in ("", ".", "..") for part in name.split("/")) or ":" in path.parts[0]):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return name


def safe_file(root, name):
    root = Path(root).resolve()
    name = safe_name(Path(name).as_posix())
    path = root / name
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Missing or escaped input: {name}")
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError(f"Linked input is not a release file: {name}")
    return path


def describe(value):
    digest, count = hashlib.sha256(), 0
    if isinstance(value, bytes):
        digest.update(value)
        count = len(value)
    else:
        with Path(value).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(chunk)
                digest.update(chunk)
    return {"bytes": count, "sha256": digest.hexdigest()}


def source_checkpoint(root):
    """Read local provenance if present; extracted bundles need no Git metadata."""
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=False)
        if top.returncode or Path(top.stdout.strip()).resolve() != root:
            return {"status": "not_available", "commit": None}
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        excluded = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", ROADMAP_COMMIT, head],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if excluded.returncode == 0:
            raise ValueError("The unpublished roadmap-only commit must not be in this publication lineage")
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
        return {"status": "available", "commit": head,
                "tracked_worktree_changes": bool(dirty.strip()),
                "excluded_roadmap_commit": ROADMAP_COMMIT,
                "roadmap_commit_in_ancestry": False if excluded.returncode == 1 else "object_not_available",
                "scope": "Local source checkpoint; archive inventories identify every actual packaged byte."}
    except FileNotFoundError:
        return {"status": "not_available", "commit": None}


def _same_value(left, right, *, rtol, atol, label="value"):
    """Compare reports at the already declared replay precision, never decisions."""
    if isinstance(left, dict) and isinstance(right, dict):
        keys_left, keys_right = set(left) - {"analysis_utc"}, set(right) - {"analysis_utc"}
        if keys_left != keys_right:
            raise ValueError(f"Report keys differ: {label}")
        for key in keys_left:
            _same_value(left[key], right[key], rtol=rtol, atol=atol, label=f"{label}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"Report lengths differ: {label}")
        for index, (a, b) in enumerate(zip(left, right)):
            _same_value(a, b, rtol=rtol, atol=atol, label=f"{label}[{index}]")
    elif isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            raise ValueError(f"Report Boolean differs: {label}")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
            raise ValueError(f"Report number differs: {label}: {left!r} != {right!r}")
    elif left != right:
        raise ValueError(f"Report value differs: {label}: {left!r} != {right!r}")


def _csv_values(path):
    def number(value):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else value
        except ValueError:
            return value
    with Path(path).open(newline="") as stream:
        return [{key: number(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def audit_results(root=ROOT, *, recompute=True):
    root = Path(root).resolve()
    study = root / STUDY
    freeze_path = safe_file(root, STUDY / "freeze.json")
    freeze = read_json(freeze_path)
    if freeze.get("schema_version") != "reconstruction-mechanism-v1" or freeze.get("public_preregistration") is not False:
        raise ValueError("Expected the explicitly local reconstruction-mechanism freeze")
    frozen = {}
    for item in freeze["files"]:
        name = safe_name(item["path"])
        if name in frozen:
            raise ValueError(f"Repeated frozen path: {name}")
        path = safe_file(root, name)
        if describe(path)["sha256"] != item["sha256"]:
            raise ValueError(f"Frozen source/input changed: {name}")
        frozen[name] = item["sha256"]
    config = read_json(study / "config.json")
    manifest = read_json(study / "prepared_inputs/manifest.json")
    tasks, methods = manifest["tasks"], config["methods"]
    if (len(tasks) != 192 or len({task["task_id"] for task in tasks}) != 192
            or manifest.get("development_only") is not False or len(methods) != 10
            or len(set(methods)) != 10):
        raise ValueError("Incomplete or unexpected reserved task/method roster")
    freeze_sha = describe(freeze_path)["sha256"]
    execution = read_json(study / "results/execution.json")
    if (execution.get("freeze_sha256") != freeze_sha or not execution.get("completed_utc")
            or not re.fullmatch(r"[0-9a-f]{40}", execution.get("local_freeze_commit", ""))
            or execution.get("public_preregistration") is not False):
        raise ValueError("Completed execution identity does not match the local freeze")
    evidence = {}
    for task in tasks:
        input_path = safe_file(study, task["input"])
        if describe(input_path)["sha256"] != task["sha256"]:
            raise ValueError(f"Prepared input changed: {task['task_id']}")
        directory = study / "results" / safe_name(task["task_id"])
        summary = read_json(safe_file(directory, "summary.json"))
        if (summary.get("status") != "completed" or summary.get("development_only") is not False
                or summary.get("freeze_sha256") != freeze_sha or summary.get("input_sha256") != task["sha256"]
                or summary.get("task", {}).get("task_id") != task["task_id"]
                or set(summary.get("prediction_array_sha256", {})) != set(methods)
                or set(summary.get("files", {})) != {"predictions.npz", "fitted_suite.pkl.gz"}):
            raise ValueError(f"Task evidence is incomplete: {task['task_id']}")
        for name, digest in summary["files"].items():
            path = safe_file(directory, name)
            if describe(path)["sha256"] != digest:
                raise ValueError(f"Task artifact changed: {task['task_id']}/{name}")
        evidence[task["task_id"]] = describe(directory / "summary.json")
    independent = read_json(study / "results/audit.json")
    expected_audit = {"schema_version": "reconstruction-mechanism-independent-audit-v1",
                      "status": "passed", "complete_roster": True, "tasks_verified": 192,
                      "method_task_predictions_verified": 1920, "primary_contrasts_verified": 6,
                      "scored_cells_per_method": 958464}
    for key, value in expected_audit.items():
        if independent.get(key) != value:
            raise ValueError(f"Independent audit is incomplete or failed: {key}")
    if independent.get("audit_source_sha256") != describe(safe_file(study, "audit.py"))["sha256"]:
        raise ValueError("The independent audit source differs from its recorded code identity")
    replay = read_json(study / "results/replay.json")
    expected_pairs = {(task["task_id"], method) for task in tasks for method in methods}
    records = replay.get("records", [])
    if (replay.get("status") != "passed" or len(records) != 1920
            or {(row["task_id"], row["method"]) for row in records} != expected_pairs
            or replay.get("rtol") != config["execution"]["replay_rtol"]
            or replay.get("atol") != config["execution"]["replay_atol"]):
        raise ValueError("A complete selected-state replay at the frozen tolerances is required")
    decision_path = study / "analysis/decision.json"
    decision = read_json(decision_path)
    if (decision.get("decision") not in DECISIONS or decision.get("technical_complete") is not True
            or decision.get("completed_valid_tasks") != 192 or decision.get("task_errors")
            or decision.get("scored_target_cells_per_method") != 958464):
        raise ValueError("A completed, technically valid scientific decision is required; STOP is publishable")
    agreement = independent.get("published_analysis_agreement", {})
    if agreement.get("contrasts_compared") != 6 or agreement.get("sha256") != describe(decision_path)["sha256"]:
        raise ValueError("Independent audit is not bound to this exact completed six-contrast analysis")
    analysis_manifest = read_json(study / "analysis/analysis_manifest.json")
    report_files = {}
    for item in analysis_manifest["output_files"]:
        path = safe_file(study, item["path"])
        if describe(path)["sha256"] != item["sha256"]:
            raise ValueError(f"Saved analysis report changed: {item['path']}")
        report_files[path.name] = path
    if "decision.json" not in report_files or "primary_contrasts.csv" not in report_files:
        raise ValueError("The saved independent-analysis inventory is incomplete")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="kahm_mechanism_reanalysis_") as temporary:
            output = Path(temporary)
            command = [sys.executable, str(study / "analyze.py"), "--study-dir", str(study), "--output", str(output)]
            result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if result.returncode:
                raise ValueError(f"New-study independent reanalysis failed:\n{result.stdout}\n{result.stderr}")
            rtol, atol = config["execution"]["replay_rtol"], config["execution"]["replay_atol"]
            for name, original in report_files.items():
                regenerated = output / name
                if not regenerated.is_file():
                    raise ValueError(f"Independent analysis did not regenerate {name}")
                if original.suffix == ".csv":
                    _same_value(_csv_values(original), _csv_values(regenerated), rtol=rtol, atol=atol, label=name)
                elif original.suffix == ".json":
                    _same_value(read_json(original), read_json(regenerated), rtol=rtol, atol=atol, label=name)
                elif original.read_bytes() != regenerated.read_bytes():
                    raise ValueError(f"Independent report bytes differ: {name}")
    return {
        "status": "passed", "forecast_refits": 0, "model_imports_by_packager": 0,
        "historical_studies_replayed": 0, "new_independent_analysis_recomputed": bool(recompute),
        "tasks": 192, "methods": 10, "method_task_predictions": 1920,
        "scored_cells_per_method": 958464, "decision": decision["decision"],
        "public_preregistration": False, "local_freeze_commit": execution["local_freeze_commit"],
        "freeze": describe(freeze_path), "independent_audit": describe(study / "results/audit.json"),
        "selected_state_replay": describe(study / "results/replay.json"),
        "analysis_decision": describe(decision_path), "source_checkpoint": source_checkpoint(root),
        "reports": {name: describe(path) for name, path in sorted(report_files.items())},
        "task_summary_inventory": evidence,
        "scope": "Completed findings under a local prospective freeze; no journal submission or public preregistration claim.",
    }


def _tree_files(root, directory):
    origin = safe_file(root, directory / "README.md").parent if (root / directory / "README.md").is_file() else root / directory
    if not origin.is_dir() or origin.is_symlink():
        raise ValueError(f"Missing or linked payload directory: {directory}")
    files = {}
    for path in sorted(origin.rglob("*")):
        relative = path.relative_to(origin)
        if any(part.startswith(".") or part in SKIP_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Linked payload: {path}")
        if (path.is_file() and path.name != "PACKED_EVIDENCE.json"
                and path.suffix not in {".pyc", ".aux", ".log", ".bbl", ".blg", ".out", ".fls", ".fdb_latexmk"}):
            name = (directory / relative).as_posix()
            files[name] = safe_file(root, name)
    return files


def source_readme(audit):
    return f"""# Completed KAHM reconstruction-to-prediction research manuscript

This archive is the complete editable source of the current unsubmitted
research manuscript. It includes LaTeX, BibTeX, preserved local styles and
notices, and all generated historical and new-study tables and figures.
It reports completed findings; the protocol is supporting reproducibility.
The frozen mechanism-study decision is `{audit['decision']}`.

From this archive's root, with TeX Live, latexmk, pdfLaTeX, BibTeX and Latin Modern:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Forecasting models are not fitted to compile the manuscript. The source uses
the supplied TMLR preprint style and accurately identifies an unsubmitted
research draft. The template's original files and licenses remain intact.
The companion `{REPRO_NAME}` preserves the full new-study evidence and its
exact inherited numerical source. Historical evidence remains separately
available in {REPOSITORY_URL}; its publication displays are included here.

`{INVENTORY}` records every included byte size and SHA-256. Its evidence
metadata binds this source package to the exact frozen study, completed
decision, independent audit, and selected-state replay. Root code/data licenses
and local third-party/template notices retain their respective scopes.
"""


def reproduction_readme(audit):
    return f"""# Completed reconstruction-to-prediction mechanism experiment

Start with `extensions/reconstruction_mechanism/FINDINGS.md` and the companion
`{PDF_NAME}`. The declared decision is `{audit['decision']}`. All 192 task suites,
ten procedures and 1,920 method/task predictions are retained, with 958,464
scored target cells per method. Inference uses 24 independent repetition
aggregates, not the individual overlapping forecasting windows.

This archive contains the final study code, exact inherited numerical files,
prepared inputs, frozen protocol, all predictions and selected model states,
selection/guard/bound/cost records, complete analysis, and independent audit.
Repository-relative paths are retained. The scientific freeze was local and
prospective; it was not publicly preregistered. Post-execution reporting and
packaging are identified separately from the unchanged frozen experiment.

## Verify and reproduce saved findings

This downloadable reproduction ZIP already contains expanded new-study
evidence. In a public GitHub source checkout, first restore the lossless parts
using `python -m extensions.reconstruction_mechanism.evidence unpack`; that
checkout also restores the separately retained adaptive-development evidence.
The reproduction ZIP deliberately excludes transport parts and the separate
adaptive evidence, avoiding archive recursion and duplicate storage.

Use Python 3.12 and install the pinned study requirements in your own environment:

```bash
python -m pip install -r extensions/reconstruction_mechanism/requirements.txt
python -m extensions.reconstruction_mechanism.package_results audit
python -m extensions.reconstruction_mechanism.analyze --output /tmp/kahm-mechanism-reanalysis
python -m extensions.reconstruction_mechanism.audit --output /tmp/kahm-mechanism-independent-audit.json
python extensions/tmlr_integration/build_mechanism_assets.py --output /tmp/kahm-mechanism-displays
```

The packager verifies byte inventories and runs only the new study's independent
saved-prediction analysis. The independent audit separately reconstructs stored
state inference and checks the fitting equations and selectors. It does not
refit the forecasting candidates or execute a historical study's replay.
Selected-state replay remains available using:

```bash
python -m extensions.reconstruction_mechanism.run replay
```

That explicit replay loads the supplied trusted Python states. Archive byte
verification alone never unpickles them. Replay writes a new replay record;
use a disposable extracted copy when preserving every supplied byte matters.

## Optional fresh-process refits

To reproduce fitting, choose an unused output directory outside the supplied
results and run each task from `prepared_inputs/manifest.json` in its own
process. For example, replace TASK_ID and the output path below with a declared
task identifier and a new destination:

```bash
python -m extensions.reconstruction_mechanism.run worker --task-id TASK_ID --output /tmp/new-kahm-fit/TASK_ID
```

The worker checks the frozen source/inputs and never replaces an existing
result. A fresh process pins one CPU and one numerical thread. Complete-run
execution provenance is retained in `results/execution.json`; a source checkout
with the recorded local Git freeze commit additionally supports the `run` CLI.
Use the declared environment when assessing arithmetic reproducibility.

## Scope and attribution

The archive covers the completed new mechanism study and exact dependencies.
It does not replace the separately preserved earlier experimental archives at
{REPOSITORY_URL}. The editable integrated manuscript is supplied as
`{SOURCE_NAME}`. No journal submission, acceptance, DOI or external replication
is implied by publication. Original software follows LICENSE, generated data
and non-code outputs follow DATA_LICENSE, and bundled upstream/template
notices retain their own terms. `{INVENTORY}` inventories all included files.
"""


def manuscript_payload(root, audit):
    files = {}
    for name, path in _tree_files(root, PAPER).items():
        relative = Path(name).relative_to(PAPER)
        if relative.name in {"main.pdf", "BUILD_CHECK.json", "README_BUILD.md", "README_RELEASE.md"}:
            continue
        if path.suffix in SOURCE_SUFFIXES or ("generated" in relative.parts and path.suffix in FIGURE_SUFFIXES):
            files[relative.as_posix()] = path
    for name in ("main.tex", "references.bib", "tmlr.sty", "tmlr.bst", "fancyhdr.sty", "LICENSE_TMLR.txt"):
        if name not in files:
            raise ValueError(f"Missing required manuscript source: {name}")
    if not any(name.startswith(ASSETS.as_posix() + "/") and name.endswith(".tex") for name in files):
        raise ValueError("The manuscript must contain the completed new-study generated assets")
    provenance = read_json(safe_file(root, PAPER / ASSETS / "ASSET_PROVENANCE.json"))
    if provenance.get("builder") != describe(safe_file(root, ASSET_BUILDER)):
        raise ValueError("New-study display builder differs from its recorded source")
    if (provenance.get("checks", {}).get("decision") != audit["decision"]
            or provenance.get("checks", {}).get("complete_tasks") != 192):
        raise ValueError("New-study publication displays do not describe the completed audited decision")
    for name, expected in provenance["inputs"].items():
        if describe(safe_file(root, STUDY / name)) != expected:
            raise ValueError(f"Publication-display input differs from its provenance: {name}")
    for name, expected in provenance["outputs"].items():
        if describe(safe_file(root, PAPER / ASSETS / name)) != expected:
            raise ValueError(f"Publication display differs from its provenance: {name}")
    for name in ("LICENSE", "DATA_LICENSE"):
        files[name] = safe_file(root, name)
    files["README_RELEASE.md"] = source_readme(audit).encode()
    files["STUDY_EVIDENCE.json"] = json_bytes({key: value for key, value in audit.items()
                                               if key != "task_summary_inventory"})
    return files


def reproduction_payload(root, audit):
    freeze = read_json(root / STUDY / "freeze.json")
    files = {item["path"]: safe_file(root, item["path"]) for item in freeze["files"]}
    files.update(_tree_files(root, STUDY))
    required = ("LICENSE", "DATA_LICENSE", "CITATION.cff",
                "extensions/predictive_geometry/LICENSE",
                "extensions/predictive_geometry/THIRD_PARTY_NOTICES.md",
                "extensions/predictive_geometry/study/upstream_manifest.json",
                "extensions/predictive_geometry/study/vendor/upstream/LICENSE")
    for name in required:
        files[name] = safe_file(root, name)
    files[ASSET_BUILDER.as_posix()] = safe_file(root, ASSET_BUILDER)
    files.update(_tree_files(root, PAPER / ASSETS))
    for name in list(files):
        for parent in Path(name).parents:
            initializer = parent / "__init__.py"
            if (root / initializer).is_file():
                files[initializer.as_posix()] = safe_file(root, initializer)
    for name in ("FINDINGS.md", "audit.py", "package_results.py", "results/audit.json", "results/replay.json"):
        if (STUDY / name).as_posix() not in files:
            raise ValueError(f"Missing completed-study publication input: {name}")
    files["README.md"] = reproduction_readme(audit).encode()
    return files


def _entries(archive):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(entries) > MAX_ENTRIES or len(names) != len(set(names)):
        raise ValueError("Excessive or duplicate archive entries")
    if sum(entry.file_size for entry in entries) > MAX_BYTES:
        raise ValueError("Archive exceeds its declared uncompressed-byte limit")
    for entry in entries:
        safe_name(entry.filename)
        mode = entry.external_attr >> 16
        if entry.is_dir() or stat.S_ISLNK(mode) or entry.flag_bits & 1:
            raise ValueError("Directory, linked or encrypted archive entry")
    return names


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = _entries(archive)
        if INVENTORY not in names:
            raise ValueError("Missing artifact inventory")
        if archive.getinfo(INVENTORY).file_size > 32 * 1024 * 1024:
            raise ValueError("Excessive artifact inventory size")
        inventory = json.loads(archive.read(INVENTORY))
        if set(names) != set(inventory["files"]) | {INVENTORY}:
            raise ValueError("Archive roster differs from its complete inventory")
        for name, expected in inventory["files"].items():
            digest, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if {"bytes": size, "sha256": digest.hexdigest()} != expected:
                raise ValueError(f"Archive member differs from inventory: {name}")
        if inventory["kind"] == "reconstruction_mechanism_reproduction":
            freeze_name = (STUDY / "freeze.json").as_posix()
            freeze_bytes = archive.read(freeze_name)
            if describe(freeze_bytes) != inventory["evidence"]["freeze"]:
                raise ValueError("Archived freeze identity differs from release evidence")
            for item in json.loads(freeze_bytes)["files"]:
                if inventory["files"].get(item["path"], {}).get("sha256") != item["sha256"]:
                    raise ValueError(f"Exact frozen dependency absent or modified: {item['path']}")
            for filename, field in (("results/audit.json", "independent_audit"),
                                    ("results/replay.json", "selected_state_replay"),
                                    ("analysis/decision.json", "analysis_decision")):
                if describe(archive.read((STUDY / filename).as_posix())) != inventory["evidence"][field]:
                    raise ValueError(f"Archived evidence identity differs: {filename}")
        elif inventory["kind"] == "manuscript_source":
            if not {"main.tex", "references.bib", "README_RELEASE.md", "STUDY_EVIDENCE.json"} <= set(names):
                raise ValueError("Incomplete manuscript archive")
        else:
            raise ValueError("Unknown artifact kind")
    return {"status": "passed", "kind": inventory["kind"], "files": len(inventory["files"]),
            "artifact": describe(path)}


def write_archive(path, files, *, kind, audit):
    if INVENTORY in files:
        raise ValueError("Reserved inventory path in payload")
    inventory = {"schema_version": 1, "kind": kind, "evidence": audit,
                 "source_repository": REPOSITORY_URL,
                 "files": {safe_name(name): describe(value) for name, value in sorted(files.items())}}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted({**files, INVENTORY: json_bytes(inventory)}.items()):
            entry = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(entry, "w", force_zip64=True) as target:
                if isinstance(value, bytes):
                    target.write(value)
                else:
                    with Path(value).open("rb") as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
    return verify_archive(path)


def verify_release(directory):
    directory = Path(directory).resolve()
    release = read_json(safe_file(directory, RELEASE_NAME))
    if set(release["artifacts"]) != set(PRODUCTS) - {RELEASE_NAME}:
        raise ValueError("Unexpected final release artifact roster")
    for name, expected in release["artifacts"].items():
        if describe(safe_file(directory, name)) != expected:
            raise ValueError(f"Release artifact differs from manifest: {name}")
    verified = [verify_archive(directory / name) for name in (SOURCE_NAME, REPRO_NAME)]
    if not (directory / PDF_NAME).read_bytes().startswith(b"%PDF-"):
        raise ValueError("Manuscript payload is not a PDF")
    return {"status": "passed", "decision": release["evidence"]["decision"],
            "artifacts": release["artifacts"], "archives": verified}


def build(pdf, output, *, root=ROOT, overwrite=False):
    root, pdf, output = Path(root).resolve(), Path(pdf).resolve(), Path(output).resolve()
    if output in {root, root / STUDY, root / PAPER} or output.is_relative_to(root / STUDY) or output.is_relative_to(root / PAPER):
        raise ValueError("Use a dedicated release destination outside scientific/manuscript source directories")
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("Supply an already compiled manuscript PDF")
    audit = audit_results(root)
    manuscript = manuscript_payload(root, audit)
    reproduction = reproduction_payload(root, audit)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kahm_mechanism_package_", dir=output.parent) as temporary:
        staging = Path(temporary)
        shutil.copyfile(pdf, staging / PDF_NAME)
        write_archive(staging / SOURCE_NAME, manuscript, kind="manuscript_source", audit=audit)
        write_archive(staging / REPRO_NAME, reproduction, kind="reconstruction_mechanism_reproduction", audit=audit)
        manifest = {"schema_version": 1, "source_repository": REPOSITORY_URL, "evidence": audit,
                    "scope": "Completed research findings and reproducible artifacts; unsubmitted manuscript.",
                    "artifacts": {name: describe(staging / name) for name in PRODUCTS if name != RELEASE_NAME}}
        (staging / RELEASE_NAME).write_bytes(json_bytes(manifest))
        verify_release(staging)
        for name in PRODUCTS:
            destination = output / name
            if destination.exists() and describe(destination) != describe(staging / name) and not overwrite:
                raise FileExistsError(f"Different existing release preserved: {destination}; choose a new output or explicit --overwrite")
        for name in PRODUCTS:
            destination = output / name
            if not destination.exists() or describe(destination) != describe(staging / name):
                shutil.copyfile(staging / name, destination)
    return verify_release(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="verify complete new evidence and reanalyze predictions; no forecast refits")
    audit.add_argument("--root", type=Path, default=ROOT)
    create = commands.add_parser("build", help="package a compiled final-results manuscript and completed study")
    create.add_argument("--root", type=Path, default=ROOT)
    create.add_argument("--pdf", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--overwrite", action="store_true", help="replace only the four named products, preserving scientific evidence")
    verify = commands.add_parser("verify", help="verify ZIP or release byte inventories without importing models")
    verify.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_results(args.root)
    elif args.command == "build":
        result = build(args.pdf, args.output, root=args.root, overwrite=args.overwrite)
    else:
        result = verify_release(args.path) if args.path.is_dir() else verify_archive(args.path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
