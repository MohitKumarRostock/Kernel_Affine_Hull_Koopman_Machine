#!/usr/bin/env python3
"""Package the forecast-trained manuscript and preserved evidence without refits.

The new run is re-audited on a temporary copy with its frozen numerical source.
Its supplied reports must match that replay exactly. Historical studies remain
separate: the manuscript includes their existing publication assets, while the
reproduction ZIP includes the new extension, its exact inherited dependencies,
and the preserved exploratory gate pilot at their repository-relative paths.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
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
STUDY = Path("extensions/direct_reconstruction")
RUN = STUDY / "results"
PILOT = Path("extensions/forecast_gate_pilot")
PAPER = Path("extensions/tmlr_integration/manuscript")
PACKAGER = Path("extensions/tmlr_integration/package_direct.py")
HELPER = Path("extensions/tmlr_integration/package_utility.py")
PDF_NAME = "KAHM_Forecast_Trained_Manuscript.pdf"
SOURCE_NAME = "KAHM_Forecast_Trained_Manuscript_Source.zip"
REPRO_NAME = "KAHM_Direct_Reconstruction_Reproduction.zip"
RELEASE_NAME = "RELEASE_MANIFEST.json"
INVENTORY = shared.INVENTORY
REPOSITORY_URL = shared.REPOSITORY_URL
GENERATED_FILES = (
    "method_table.tex", "primary_table.tex", "natural_table.tex",
    "cost_table.tex", "result_macros.tex",
)
REPORT_FILES = (
    "summary.json", "audit_report.json", "task_metrics.csv", "horizon_metrics.csv",
    "primary_contrasts.csv", "method_aggregates.csv", "natural_collection_metrics.csv",
    "natural_available_comparisons.csv", "natural_available_leave_one_collection_out.csv",
    "cost_aggregates.csv", "optimization_summary.csv", "fallback_events.json",
) + tuple("generated/" + name for name in GENERATED_FILES)
SCOPE = (
    "Public, unsubmitted integrated manuscript; new direct-reconstruction evidence "
    "and separately preserved exploratory gate pilot in the reproduction ZIP. "
    "Earlier complete experimental archives remain in the source repository."
)
json_bytes, description = shared.json_bytes, shared.description
safe_name, read_source = shared.safe_name, shared.read_source


def tree_payload(origin, destination=Path(), *, exclude=()):
    """Read regular files without following links or copying caches/build output."""
    origin, destination = Path(origin), Path(destination)
    if not origin.is_dir() or origin.is_symlink():
        raise ValueError(f"Missing or linked payload directory: {origin}")
    files = {}
    excluded = tuple(Path(name) for name in exclude)
    for path in sorted(origin.rglob("*")):
        relative = path.relative_to(origin)
        if any(relative.is_relative_to(prefix) for prefix in excluded):
            continue
        if any(part.startswith(".") or part in shared.SKIP_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic link in payload: {path}")
        if path.is_file():
            files[(destination / relative).as_posix()] = read_source(origin, relative)
    return files


def restore_packed_evidence(root, run_dir):
    """Materialize a packed checkout without replacing any differing evidence."""
    manifest = Path(run_dir) / "PACKED_MANIFEST.json"
    if manifest.exists() or manifest.is_symlink():
        # This optional transport module performs only byte/manifest operations;
        # no scientific module or fitting API is imported here.
        from extensions.direct_reconstruction import evidence
        evidence.unpack(run_dir)
    elif not (Path(run_dir) / "tasks").is_dir():
        raise ValueError("Neither original task evidence nor its packed manifest is available")


def publication_record(root, run_dir):
    candidates = [Path(root) / STUDY / "PUBLICATION.json", Path(run_dir) / "PUBLICATION.json"]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise ValueError("A PUBLICATION.json record for the public source checkpoint is required")
    contents = read_source(existing[0].parent, existing[0].name)
    value = json.loads(contents)
    if any(json.loads(read_source(path.parent, path.name)) != value for path in existing[1:]):
        raise ValueError("Conflicting publication records")
    return contents, value


def checked_freeze(root, run_dir):
    """Check immutable snapshots, inputs, execution linkage, and public metadata."""
    root, run_dir = Path(root), Path(run_dir)
    freeze_bytes = read_source(run_dir, "freeze.json")
    freeze = json.loads(freeze_bytes)
    if freeze.get("schema_version") != "direct-reconstruction-v1":
        raise ValueError("Unexpected direct-reconstruction freeze schema")
    protocol_bytes = read_source(run_dir, "protocol.json")
    if description(protocol_bytes)["sha256"] != freeze["protocol_sha256"]:
        raise ValueError("Protocol differs from the scientific freeze")
    for name, expected in freeze["scientific_files"].items():
        safe_name(name)
        snapshot = Path(freeze["source_snapshot"]) / name
        if (description(read_source(run_dir, snapshot))["sha256"] != expected
                or description(read_source(root, name))["sha256"] != expected):
            raise ValueError(f"Scientific source or preserved snapshot changed: {name}")
    for name, expected in freeze["prepared_inputs"].items():
        if description(read_source(run_dir, name))["sha256"] != expected:
            raise ValueError(f"Prepared input differs from the freeze: {name}")
    execution_bytes = read_source(run_dir, "execution.json")
    execution = json.loads(execution_bytes)
    freeze_sha = description(freeze_bytes)["sha256"]
    publication_bytes, publication = publication_record(root, run_dir)
    commit = publication.get("commit", "")
    if (not re.fullmatch(r"[0-9a-f]{40}", commit)
            or publication.get("url") != f"{REPOSITORY_URL}/commit/{commit}"
            or publication.get("freeze_sha256") != freeze_sha
            or publication.get("protocol_sha256") != freeze["protocol_sha256"]
            or execution.get("public_freeze_commit") != commit
            or execution.get("freeze_sha256") != freeze_sha
            or execution.get("protocol_sha256") != freeze["protocol_sha256"]):
        raise ValueError("Public checkpoint, frozen inputs, and execution metadata disagree")
    recorded = dt.datetime.fromisoformat(publication["recorded_at_utc"])
    started = dt.datetime.fromisoformat(execution["started_utc"])
    if recorded.tzinfo is None or started.tzinfo is None or recorded > started:
        raise ValueError("Public checkpoint record must precede execution and use explicit UTC offsets")
    if execution.get("status") not in {"complete", "complete_with_failures"}:
        raise ValueError("The declared execution roster has not finished")
    return freeze, freeze_sha, publication_bytes, publication, execution


def local_commit_check(root, run_dir, freeze, commit):
    """Verify local Git contents when present; this is not remote timing proof."""
    root, run_dir = Path(root), Path(run_dir)
    try:
        found = subprocess.run(["git", "-C", str(root), "cat-file", "-e", commit + "^{commit}"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        found = None
    if found is None or found.returncode:
        return {"status": "not_available", "scope": "No matching local Git object; hashes and the recorded checkpoint are still checked."}
    expected = {
        (RUN / "freeze.json").as_posix(): description(read_source(run_dir, "freeze.json"))["sha256"],
        (RUN / "protocol.json").as_posix(): freeze["protocol_sha256"],
        **freeze["scientific_files"],
    }
    expected.update({(RUN / freeze["source_snapshot"] / name).as_posix(): digest
                     for name, digest in freeze["scientific_files"].items()})
    expected.update({(RUN / name).as_posix(): digest for name, digest in freeze["prepared_inputs"].items()})
    for name, digest in expected.items():
        result = subprocess.run(["git", "-C", str(root), "show", commit + ":" + safe_name(name)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode or description(result.stdout)["sha256"] != digest:
            raise ValueError(f"Recorded local public-commit tree does not contain the frozen bytes: {name}")
    return {"status": "passed", "checked_files": len(expected),
            "scope": "Matching local Git tree contents; no independent certification of remote publication time or unrecorded evaluations."}


def check_analysis(summary, report, protocol):
    """Require complete accounting, without relabelling unavailable tasks as fits."""
    methods, tasks = protocol["methods"], protocol["tasks"]
    if (len(tasks) != 46 or len({task["task_id"] for task in tasks}) != 46
            or len(methods) != 17 or len(set(methods)) != 17
            or report.get("passed") is not True or report.get("failures")
            or report.get("selected_models_replayed") is not True
            or summary.get("audit_passed") is not True
            or summary.get("expected_tasks") != len(tasks)
            or summary.get("expected_method_outputs") != len(tasks) * len(methods)
            or report.get("expected_tasks") != len(tasks)
            or len(summary.get("primary_contrasts", [])) != 18):
        raise ValueError("Release requires the passed full-roster audit with selected-model replay")
    unavailable = summary.get("documented_unavailable_tasks", [])
    unavailable_ids = {task["task_id"] for task in unavailable}
    if (len(unavailable_ids) != len(unavailable)
            or not unavailable_ids <= {task["task_id"] for task in tasks}
            or unavailable != report.get("documented_unavailable_tasks")
            or summary.get("available_method_outputs") != (len(tasks) - len(unavailable)) * len(methods)
            or report.get("available_method_outputs") != summary["available_method_outputs"]
            or summary.get("complete_roster") != (not unavailable)
            or report.get("complete_roster") != summary["complete_roster"]):
        raise ValueError("Successful forecasts and unavailable tasks do not account for the full roster")


def audit_results(root, run_dir, work_dir=None):
    """Restore packed checkouts if needed, then audit a copy without replacing evidence."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    work_dir = Path(work_dir or root / "review").resolve()
    restore_packed_evidence(root, run_dir)
    freeze, freeze_sha, publication_bytes, publication, execution = checked_freeze(root, run_dir)
    protocol = json.loads(read_source(run_dir, "protocol.json"))
    original_summary = json.loads(read_source(run_dir, "summary.json"))
    original_report = json.loads(read_source(run_dir, "audit_report.json"))
    check_analysis(original_summary, original_report, protocol)
    originals = {name: read_source(run_dir, name) for name in REPORT_FILES if (run_dir / name).is_file()}
    for name in ("summary.json", "audit_report.json", "primary_contrasts.csv", "method_aggregates.csv",
                 "fallback_events.json", *("generated/" + name for name in GENERATED_FILES)):
        if name not in originals:
            raise ValueError(f"Missing independent-analysis report: {name}")
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="direct_reaudit_", dir=work_dir) as temporary:
        temporary = Path(temporary)
        for name in freeze["scientific_files"]:
            target = temporary / safe_name(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read_source(run_dir, Path(freeze["source_snapshot"]) / name))
        copied_run = temporary / RUN
        for name, content in tree_payload(run_dir, exclude=("packed_tasks", "PACKED_MANIFEST.json")).items():
            target = copied_run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        # Remove only known analysis products in the disposable copy so stale
        # output cannot masquerade as a successful independent regeneration.
        for name in REPORT_FILES:
            (copied_run / name).unlink(missing_ok=True)
        result = subprocess.run(
            [sys.executable, str(temporary / STUDY / "analyze.py"), str(copied_run), "--verify-live-source"],
            cwd=temporary, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if result.returncode:
            raise ValueError(f"Frozen saved-evidence audit failed:\n{result.stdout}\n{result.stderr}")
        replay_summary = json.loads(read_source(copied_run, "summary.json"))
        replay_report = json.loads(read_source(copied_run, "audit_report.json"))
        check_analysis(replay_summary, replay_report, protocol)
        replayed = {name: read_source(copied_run, name) for name in REPORT_FILES if (copied_run / name).is_file()}
        if originals.keys() != replayed.keys():
            raise ValueError("Supplied and independently regenerated report rosters differ")
        for name, content in originals.items():
            if content != replayed[name]:
                raise ValueError(f"Supplied report differs from frozen independent replay: {name}")
    return {
        "status": "passed", "forecasting_fits": 0,
        "expected_tasks": len(protocol["tasks"]), "methods_per_task": len(protocol["methods"]),
        "available_method_outputs": original_summary["available_method_outputs"],
        "complete_roster": original_summary["complete_roster"],
        "documented_unavailable_tasks": original_summary["documented_unavailable_tasks"],
        "primary_contrasts": len(original_summary["primary_contrasts"]),
        "complete_primary_contrasts": sum(row.get("complete") is True for row in original_summary["primary_contrasts"]),
        "selected_models_replayed": True,
        "freeze_sha256": freeze_sha,
        "protocol_sha256": freeze["protocol_sha256"],
        "execution_sha256": description(read_source(run_dir, "execution.json"))["sha256"],
        "publication_sha256": description(publication_bytes)["sha256"],
        "report_files": {name: description(content) for name, content in originals.items()},
        "local_commit_check": local_commit_check(root, run_dir, freeze, publication["commit"]),
        "scope": "Exact frozen report replay and selected-state prediction replay; no candidate refits, external replication, or pooling with historical evidence.",
    }


def source_readme():
    return f"""# Complete editable forecast-trained KAHM manuscript

This is a public, unsubmitted research manuscript. All editable LaTeX and
BibTeX, local styles and notices, and existing generated tables and figures
needed for the complete integrated paper are included. Compilation byproducts
and the final PDF are excluded from this source archive.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The companion {REPRO_NAME} contains the new direct-reconstruction experiment,
its exact inherited numerical dependencies, and the separately preserved
exploratory gate pilot. The complete evidence for earlier studies remains at
{REPOSITORY_URL}; their existing generated publication assets are included
here to compile the paper. Historical results are not silently recomputed or
reclassified by packaging. ARTIFACT_MANIFEST.json inventories every included
file's bytes and SHA-256. The preserved software, data, and template licenses
apply to their respective content. No submission, acceptance, new DOI, external
registration, or author sign-off is implied by creating these files.
"""


def reproduction_readme(audit):
    return f"""# Direct reconstruction: saved-evidence reproduction bundle

This archive contains the new direct-reconstruction extension, its public
source checkpoint, frozen source snapshots and exact inherited dependencies,
46 prepared tasks, every saved outcome or documented unavailable task, the
independent audit and report assets, selected model states, and the preserved
exploratory gate pilot. Paths match their locations in the source repository.
The new run accounts for {audit['available_method_outputs']} available method
outputs and {len(audit['documented_unavailable_tasks'])} unavailable tasks, with
17 procedures on each successfully fitted task. A fit is not invented for an
unavailable task. The full-roster completeness flag is {audit['complete_roster']}.

The synthetic study has 36 tasks: twelve new repetitions for each of three
known laws, with an eighteen-contrast primary family. The natural roster has
ten series from four chosen collections. Three rideshare tasks lack usable B
windows under the frozen missingness rule; their failure remains explicit.
The planned complete four-collection aggregate is undefined if a required
collection is unavailable. Separately labeled available-collection summaries
are descriptive and do not create population-level evidence from overlapping
origins. The exploratory gate pilot is not pooled with the new study.

## Audit saved evidence without refitting forecasting models

From the extracted root, install the recorded requirements, then run:

```bash
python -m pip install -r extensions/direct_reconstruction/requirements.txt
python -m extensions.tmlr_integration.package_direct audit \\
  --run-dir extensions/direct_reconstruction/results
```

The packaging audit runs the frozen analyzer on a temporary copy under
`review/`, verifies all frozen inputs, reconstructs windows, losses, selection
and cost accounting, replays selected model predictions, and requires exact
agreement with the supplied analysis products. Original evidence is retained.
It never calls a fitting API. It does not independently refit every optimizer
candidate, establish global optimality, or certify unrecorded evaluations.
The local Git membership check is available when the recorded commit object
exists; standalone extracted archives still verify bytes and recorded hashes.

Repository checkouts transport task evidence as lossless per-task archives.
The packaging audit verifies and restores those files automatically when a
packed manifest is present. Before running the original analyzer directly on
a packed checkout, restore its evidence with:

```bash
python -m extensions.direct_reconstruction.evidence unpack \\
  --run-dir extensions/direct_reconstruction/results
```

This reproduction ZIP already contains the restored original task files and
omits duplicate transport archives, so it needs no unpacking step. Restoration
checks archive hashes, exact member inventories and member hashes, and never
overwrites an existing differing file. Packing and restoration do not fit,
score, or modify any scientific result.

To run the original analyzer directly, which regenerates reports in place:

```bash
python extensions/direct_reconstruction/analyze.py \\
  extensions/direct_reconstruction/results --verify-live-source
```

To verify archive entries and every inventoried byte without importing models:

```bash
python -m extensions.tmlr_integration.package_direct verify /path/to/{REPRO_NAME}
```

Selected states are Python serialization artifacts from this preserved run.
Byte verification does not execute them; replay is an explicit audit operation.
The historical gate pilot retains its own source freeze, inputs, predictions,
audits, exploratory designation, and reports under
`extensions/forecast_gate_pilot/`. Its numerical online prototype does not
demonstrate useful incremental geometry, update latency, or bounded online
memory. A new delayed-label experiment is needed for those claims.

## Scope and rights

The complete integrated manuscript is supplied separately as
{SOURCE_NAME}. It includes previously generated historical tables and figures
for compilation. This reproduction archive does not replace the complete
controlled, natural, scaling, utility, or proxy-metric archives at
{REPOSITORY_URL}.

Original public natural-source ZIPs are not bundled. Pinned download URLs,
published checksums, local archive and TSF-member SHA-256 values, exact selected
series, and preparation code are included. The ten selected natural input
files and the 46 frozen prepared task files are present for offline replay.
The four pinned Zenodo records declare CC-BY-4.0; attribution and changes are
documented in `extensions/direct_reconstruction/data/README.md` and its source
manifest. Root LICENSE and DATA_LICENSE, and retained upstream/template notices,
apply to their respective content. ARTIFACT_MANIFEST.json inventories all files.
No submission, acceptance, new DOI, external registration, or author sign-off
is implied by this release.
"""


def manuscript_payload(root, run_dir):
    files = shared.manuscript_payload(Path(root))
    for name in ("section_direct_reconstruction.tex", "appendix_direct_reconstruction.tex"):
        if name not in files:
            raise ValueError(f"Missing direct-reconstruction manuscript source: {name}")
    for name in GENERATED_FILES:
        destination = "generated/direct_reconstruction/" + name
        if files.get(destination) != read_source(run_dir, "generated/" + name):
            raise ValueError(f"Manuscript asset differs from the audited new report: {destination}")
    files["README_RELEASE.md"] = source_readme().encode("utf-8")
    return files


def reproduction_payload(root, run_dir, freeze, publication_bytes, audit):
    root, run_dir = Path(root), Path(run_dir)
    files = {name: read_source(run_dir, Path(freeze["source_snapshot"]) / name)
             for name in freeze["scientific_files"]}
    for name, content in tree_payload(root / STUDY, STUDY, exclude=("results",)).items():
        if name in files and files[name] != content:
            raise ValueError(f"Live extension source differs from preserved bytes: {name}")
        files[name] = content
    files.update(tree_payload(run_dir, RUN, exclude=("packed_tasks", "PACKED_MANIFEST.json")))
    files.update(tree_payload(root / PILOT, PILOT))
    files[(STUDY / "PUBLICATION.json").as_posix()] = publication_bytes
    for name in (
        "LICENSE", "DATA_LICENSE", "CITATION.cff", PACKAGER.as_posix(), HELPER.as_posix(),
        "extensions/predictive_geometry/LICENSE",
        "extensions/predictive_geometry/THIRD_PARTY_NOTICES.md",
        "extensions/predictive_geometry/study/upstream_manifest.json",
        "extensions/predictive_geometry/study/vendor/upstream/LICENSE",
    ):
        files[name] = read_source(root, name)
    for name in list(files):
        for parent in Path(name).parents:
            initializer = parent / "__init__.py"
            if (root / initializer).is_file():
                files[initializer.as_posix()] = read_source(root, initializer)
    files["README.md"] = reproduction_readme(audit).encode("utf-8")
    return files


def zip_entries(archive):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(entries) > shared.MAX_ENTRIES or len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP entries or excessive entry count")
    if sum(entry.file_size for entry in entries) > shared.MAX_BYTES:
        raise ValueError("Archive exceeds the uncompressed byte limit")
    for entry in entries:
        safe_name(entry.filename)
        if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
            raise ValueError("Directories, symbolic links, and encrypted entries are not release payloads")
    return names


def verify_pilot(archive):
    base = PILOT / "pilot_results"
    freeze = json.loads(archive.read((base / "freeze.json").as_posix()))
    protocol_bytes = archive.read((base / "protocol.json").as_posix())
    protocol = json.loads(protocol_bytes)
    if (description(protocol_bytes)["sha256"] != freeze["protocol_sha256"]
            or protocol.get("not_confirmatory") is not True):
        raise ValueError("Preserved pilot protocol or exploratory designation changed")
    for name, digest in freeze["source_sha256"].items():
        contents = archive.read((base / "source_snapshot" / safe_name(name)).as_posix())
        if description(contents)["sha256"] != digest:
            raise ValueError(f"Preserved pilot snapshot changed: {name}")
    for task in protocol["tasks"]:
        task_dir = base / "tasks" / safe_name(task["task_id"])
        record = json.loads(archive.read((task_dir / "task.json").as_posix()))
        for name, key in (("inputs.npz", "inputs_file_sha256"),
                          ("predictions.npz", "prediction_file_sha256"),
                          ("diagnostics.json", "diagnostics_file_sha256")):
            if description(archive.read((task_dir / name).as_posix()))["sha256"] != record[key]:
                raise ValueError(f"Preserved pilot task bytes changed: {task['task_id']}/{name}")


def verify_reproduction(archive, inventory, names):
    def read(name):
        return archive.read(safe_name(Path(name).as_posix()))

    freeze_bytes = read(RUN / "freeze.json")
    freeze = json.loads(freeze_bytes)
    if description(freeze_bytes)["sha256"] != inventory["scientific_freeze_sha256"]:
        raise ValueError("Packaged freeze differs from the artifact manifest")
    protocol_bytes = read(RUN / "protocol.json")
    protocol = json.loads(protocol_bytes)
    if description(protocol_bytes)["sha256"] != freeze["protocol_sha256"]:
        raise ValueError("Packaged protocol differs from freeze")
    for name, digest in freeze["scientific_files"].items():
        for path in (Path(name), RUN / freeze["source_snapshot"] / name):
            if description(read(path))["sha256"] != digest:
                raise ValueError(f"Frozen dependency missing or changed: {path}")
    for name, digest in freeze["prepared_inputs"].items():
        if description(read(RUN / name))["sha256"] != digest:
            raise ValueError(f"Frozen input missing or changed: {name}")
    audit = inventory["evidence_audit"]
    for name, expected in audit["report_files"].items():
        if description(read(RUN / name)) != expected:
            raise ValueError(f"Packaged report differs from independently audited report: {name}")
    publication_bytes = read(STUDY / "PUBLICATION.json")
    publication = json.loads(publication_bytes)
    execution_bytes = read(RUN / "execution.json")
    execution = json.loads(execution_bytes)
    if (description(publication_bytes)["sha256"] != audit["publication_sha256"]
            or description(execution_bytes)["sha256"] != audit["execution_sha256"]
            or publication["commit"] != inventory["public_execution_commit"]
            or execution["public_freeze_commit"] != publication["commit"]
            or execution["freeze_sha256"] != inventory["scientific_freeze_sha256"]
            or publication["freeze_sha256"] != inventory["scientific_freeze_sha256"]):
        raise ValueError("Packaged publication, execution, and freeze linkage changed")
    summary = json.loads(read(RUN / "summary.json"))
    report = json.loads(read(RUN / "audit_report.json"))
    check_analysis(summary, report, protocol)
    task_prefix = (RUN / "tasks").as_posix() + "/"
    task_ids = {name[len(task_prefix):].split("/")[0] for name in names if name.startswith(task_prefix)}
    if task_ids != {task["task_id"] for task in protocol["tasks"]}:
        raise ValueError("Packaged task directories differ from the declared roster")
    unavailable_ids = {task["task_id"] for task in summary["documented_unavailable_tasks"]}
    completed_count = 0
    for task in protocol["tasks"]:
        base = RUN / "tasks" / safe_name(task["task_id"])
        record = json.loads(read(base / "task.json"))
        if record != task:
            raise ValueError(f"Packaged task metadata changed: {task['task_id']}")
        complete_name = (base / "complete.json").as_posix()
        failed_name = (base / "failure.json").as_posix()
        if task["task_id"] in unavailable_ids:
            if failed_name not in names or complete_name in names or (base / "predictions.npz").as_posix() in names:
                raise ValueError("Unavailable task does not preserve an explicit failure without invented forecasts")
            continue
        if complete_name not in names or failed_name in names:
            raise ValueError("Available task is missing its unique completed outcome")
        completed = json.loads(read(complete_name))
        for name, key in (("task.json", "task_sha256"), ("predictions.npz", "predictions_sha256"),
                          ("diagnostics.json", "diagnostics_sha256"),
                          ("selected_models.zip", "selected_models_sha256")):
            if description(read(base / name))["sha256"] != completed[key]:
                raise ValueError(f"Completed task artifact changed: {task['task_id']}/{name}")
        expected_predictions = {f"{split}__{method}.npy" for split in ("C", "D") for method in protocol["methods"]}
        expected_predictions |= {"C_targets.npy", "D_targets.npy"}
        with zipfile.ZipFile(io.BytesIO(read(base / "predictions.npz"))) as predictions:
            if set(zip_entries(predictions)) != expected_predictions:
                raise ValueError("Saved prediction archive has a changed method or target roster")
        diagnostics = json.loads(read(base / "diagnostics.json"))
        models = diagnostics["serialized_models"]
        if set(models) != set(protocol["methods"]):
            raise ValueError("Selected model roster differs from the declared method roster")
        with zipfile.ZipFile(io.BytesIO(read(base / "selected_models.zip"))) as states:
            if set(zip_entries(states)) != {value["member"] for value in models.values()}:
                raise ValueError("Selected model archive omits or adds a model")
            for value in models.values():
                if (value["archive"] != "selected_models.zip"
                        or description(states.read(safe_name(value["member"])))
                        != {"bytes": value["bytes"], "sha256": value["sha256"]}):
                    raise ValueError("Selected model serialization changed")
        completed_count += 1
    if completed_count * len(protocol["methods"]) != summary["available_method_outputs"]:
        raise ValueError("Packaged forecast accounting differs from the audited summary")
    verify_pilot(archive)


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = zip_entries(archive)
        if INVENTORY not in names:
            raise ValueError("Missing archive byte inventory")
        inventory = json.loads(archive.read(INVENTORY))
        if set(names) != set(inventory["files"]) | {INVENTORY}:
            raise ValueError("Archive inventory is incomplete or has extra files")
        for name, expected in inventory["files"].items():
            if description(archive.read(safe_name(name))) != expected:
                raise ValueError(f"Archive byte inventory mismatch: {name}")
        if inventory["kind"] == "direct_reconstruction_reproduction":
            verify_reproduction(archive, inventory, set(names))
        elif inventory["kind"] == "manuscript_source":
            required = {"main.tex", "section_direct_reconstruction.tex", "appendix_direct_reconstruction.tex"}
            if not required <= set(names):
                raise ValueError("Complete revised manuscript source is missing")
            for name in GENERATED_FILES:
                expected = inventory["evidence_audit"]["report_files"]["generated/" + name]
                if description(archive.read("generated/direct_reconstruction/" + name)) != expected:
                    raise ValueError("Manuscript generated asset differs from the audited report")
        else:
            raise ValueError("Unknown artifact kind")
    return {"status": "passed", "kind": inventory["kind"], "files": len(inventory["files"]),
            **description(Path(path).read_bytes())}


def write_archive(path, files, *, kind, freeze_sha, publication, audit):
    if INVENTORY in files:
        raise ValueError("Reserved artifact inventory name in payload")
    inventory = {
        "schema_version": 1, "kind": kind, "scope": SCOPE,
        "scientific_freeze_sha256": freeze_sha, "public_execution_commit": publication["commit"],
        "evidence_audit": audit,
        "files": {name: description(content) for name, content in sorted(files.items())},
    }
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name, content in sorted({**files, INVENTORY: json_bytes(inventory)}.items()):
            safe_name(name)
            info = zipfile.ZipInfo(name, date_time=shared.ZIP_DATE)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = (zipfile.ZIP_STORED if name.endswith((".npz", ".pdf", ".png", ".zip"))
                                  else zipfile.ZIP_DEFLATED)
            archive.writestr(info, content, compresslevel=9)
    return verify_archive(path)


def verify_release(directory):
    directory = Path(directory)
    manifest = json.loads(read_source(directory, RELEASE_NAME))
    if set(manifest["artifacts"]) != {PDF_NAME, SOURCE_NAME, REPRO_NAME}:
        raise ValueError("Release has an incomplete or unexpected artifact roster")
    verified = {}
    for name, expected in manifest["artifacts"].items():
        content = read_source(directory, name)
        if description(content) != expected:
            raise ValueError(f"Release artifact checksum changed: {name}")
        if name.endswith(".zip"):
            verified[name] = verify_archive(directory / name)
        elif not content.startswith(b"%PDF-"):
            raise ValueError("Released manuscript is not a compiled PDF")
    return {"status": "passed", "release_manifest": description(read_source(directory, RELEASE_NAME)),
            "archives": verified, "artifacts": manifest["artifacts"]}


def checked_output(output, root, run_dir, *, overwrite):
    output = Path(output).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("Release output cannot follow symbolic links")
    output = output.resolve()
    for protected in (Path(root) / STUDY, Path(root) / PILOT, Path(root) / PAPER, Path(run_dir)):
        if output.is_relative_to(protected.resolve()):
            raise ValueError("Release output must be separate from the preserved source and evidence trees")
    if output == Path(root).resolve():
        raise ValueError("Choose a dedicated review/release output directory")
    if output.exists() and (not output.is_dir() or not overwrite):
        raise FileExistsError("Existing output is preserved; use a new directory or explicit --overwrite")
    for name in (PDF_NAME, SOURCE_NAME, REPRO_NAME, RELEASE_NAME):
        destination = output / name
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError(f"Output artifact path is not a regular file: {destination}")
    return output


def build(pdf, output, run_dir=None, *, overwrite=False, work_dir=None):
    root = ROOT.resolve()
    run_dir = Path(run_dir or root / RUN).resolve()
    output = checked_output(output, root, run_dir, overwrite=overwrite)
    work_dir = Path(work_dir or root / "review").resolve()
    pdf = Path(pdf).absolute()
    pdf_bytes = read_source(pdf.parent, pdf.name)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Supplied manuscript is not a compiled PDF")
    freeze, freeze_sha, publication_bytes, publication, _ = checked_freeze(root, run_dir)
    audit = audit_results(root, run_dir, work_dir)
    source = manuscript_payload(root, run_dir)
    reproduction = reproduction_payload(root, run_dir, freeze, publication_bytes, audit)
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="direct_release_", dir=work_dir) as temporary:
        temporary = Path(temporary)
        (temporary / PDF_NAME).write_bytes(pdf_bytes)
        write_archive(temporary / SOURCE_NAME, source, kind="manuscript_source",
                      freeze_sha=freeze_sha, publication=publication, audit=audit)
        write_archive(temporary / REPRO_NAME, reproduction, kind="direct_reconstruction_reproduction",
                      freeze_sha=freeze_sha, publication=publication, audit=audit)
        manifest = {
            "schema_version": 1, "scope": SCOPE, "source_repository": REPOSITORY_URL,
            "public_execution_commit": publication["commit"], "scientific_freeze_sha256": freeze_sha,
            "evidence_audit": audit,
            "artifacts": {name: description((temporary / name).read_bytes())
                          for name in (PDF_NAME, SOURCE_NAME, REPRO_NAME)},
        }
        (temporary / RELEASE_NAME).write_bytes(json_bytes(manifest))
        verify_release(temporary)
        # Recheck after the potentially lengthy audit, and publish the manifest
        # last. An existing output directory always requires explicit overwrite.
        checked_output(output, root, run_dir, overwrite=overwrite)
        output.mkdir(parents=True, exist_ok=True)
        for name in (PDF_NAME, SOURCE_NAME, REPRO_NAME, RELEASE_NAME):
            descriptor, staged_name = tempfile.mkstemp(prefix=".direct_release_", dir=output)
            os.close(descriptor)
            staged = Path(staged_name)
            try:
                shutil.copyfile(temporary / name, staged)
                os.replace(staged, output / name)
            finally:
                staged.unlink(missing_ok=True)
    verify_release(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    release = commands.add_parser("build", help="audit and package an already compiled manuscript")
    release.add_argument("--pdf", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--run-dir", type=Path, default=ROOT / RUN)
    release.add_argument("--work-dir", type=Path, default=ROOT / "review")
    release.add_argument("--overwrite", action="store_true", help="replace only the four named products in an existing output directory")
    audit = commands.add_parser("audit", help="replay the frozen saved-evidence audit on a temporary copy")
    audit.add_argument("--run-dir", type=Path, default=ROOT / RUN)
    audit.add_argument("--work-dir", type=Path, default=ROOT / "review")
    verify = commands.add_parser("verify", help="verify a ZIP or complete release directory without model imports")
    verify.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.pdf, args.output, args.run_dir, overwrite=args.overwrite, work_dir=args.work_dir)
    elif args.command == "audit":
        result = audit_results(ROOT, args.run_dir, args.work_dir)
    else:
        result = verify_release(args.path) if args.path.is_dir() else verify_archive(args.path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
