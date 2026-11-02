#!/usr/bin/env python3
"""Package completed numerical records without copying scientific source code.

This post-study reporting utility is outside the original scientific freeze.
It must not be run until the frozen analysis and independent audit are complete.
It does not import scientific code, fit models, select procedures, score
forecasts, or reinterpret numerical outcomes. It reads metadata and hashes the
retained files byte for byte, writes three ZIPs, then reads every ZIP member
back and verifies its SHA256. All outputs are installed together in a NEW
directory outside the repository; existing paths are never overwritten.

Example from the repository root::

    python extensions/soft_state_ablation_20261102/reporting/package_numerical_records.py \
        --output-dir /path/outside/repository/soft_state_numerical_records

The default size limit is strictly less than 500,000,000 bytes per ZIP (decimal
500 MB, not 500 MiB). A size failure does not silently drop or resample records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
DEFAULT_STUDY = SCRIPT.parents[1]
REPOSITORY = SCRIPT.parents[3]
PREFIX = "extensions/soft_state_ablation_20261102/"
DATA_ARCHIVE = "Data_and_Forecasts.zip"
LOW_ARCHIVE = "Fitted_Models_00_15.zip"
HIGH_ARCHIVE = "Fitted_Models_16_31.zip"
ARCHIVES = (DATA_ARCHIVE, LOW_ARCHIVE, HIGH_ARCHIVE)
LIMIT = 500_000_000
BLOCK = 1024 * 1024
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
SOURCE_SUFFIXES = {".py", ".pyc", ".pyo", ".so", ".pyd", ".ipynb"}
FIT_FILES = {"state.json", "state.npz", "pre_c_selection.json", "validation_predictions.npz",
             "diagnostic_predictions.npz", "C_diagnostics.json", "numerical_solver_events.json", "fit_summary.json"}
PREDICTION_FILES = {"predictions.npz", "query_diagnostics.json", "prediction_receipt.json"}
ROOT_METADATA = ("config.json", "PROTOCOL.md", "freeze.json", "PUBLICATION.json",
                 "FIT_LOCK.json", "PREDICTION_LOCK.json", "TARGET_RELEASE.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            result.update(block)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    def reject_constant(value):
        raise ValueError("Non-finite JSON constant: " + value)
    raw = Path(path).read_bytes()
    return json.loads(raw, parse_constant=reject_constant), hashlib.sha256(raw).hexdigest()


def safe_relative(root, relative):
    path = Path(relative)
    result = (root / path).resolve()
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() not in ("", ".")
            and result.is_relative_to(root), "Artifact path leaves the study: " + str(relative))
    # Resolve confinement does not authorize an in-root symlink: archives retain
    # regular files only, with an unambiguous original path.
    probe = root
    for part in path.parts:
        probe = probe / part
        require(not probe.is_symlink(), "Symlink cannot be archived: " + str(relative))
    return result


def walk_files(root, relative):
    directory = safe_relative(root, relative)
    require(directory.is_dir(), "Required record directory is absent: " + relative)
    found = []
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Symlink in record tree: " + str(path))
        if path.is_dir():
            continue
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "Nonregular artifact: " + str(path))
        require(path.name != "failure.json", "A failed task cannot be packaged as completed")
        require(path.suffix.lower() not in SOURCE_SUFFIXES,
                "Scientific code unexpectedly appeared in a numerical record directory: " + str(path))
        found.append(path.relative_to(root).as_posix())
    require(bool(found), "Empty record directory: " + relative)
    return found


def time_of(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    require(value.tzinfo is not None, "A stage timestamp lacks timezone")
    return value


def validate_completed_study(root):
    """Validate the completed audit before opening any numerical payload."""
    summary, summary_hash = read_json(root / "analysis/analysis_summary.json")
    audit, audit_hash = read_json(root / "verification/independent_audit.json")
    require(summary.get("schema") == "soft-state-ablation-analysis-v1" and summary.get("status") == "complete",
            "A complete frozen analysis is required before packaging")
    require(audit.get("schema") == "independent-soft-state-audit-v1" and audit.get("status") == "passed",
            "A passed complete independent audit is required before packaging")
    require(audit["analysis_summary_sha256"] == summary_hash,
            "The independent audit is not bound to this exact analysis summary")
    for value in (summary, audit):
        require(value["tasks"] == 256 and value["independent_repetitions"] == 32,
                "Partial study output cannot be packaged as the complete cohort")
    require(summary["forecast_arrays"] == 2048 and summary["primary_family_size"] == 4 and
            audit["D_forecast_arrays_replayed"] == 2048 and audit["C_forecast_arrays_replayed"] == 2048 and
            audit["B_penalty_path_arrays_replayed"] == 25600, "Incomplete declared analysis/audit coverage")
    require(summary["criteria_established"] == audit["criteria_established"], "Analysis/audit decisions disagree")

    documents, read_hashes = {"analysis/analysis_summary.json": summary,
                              "verification/independent_audit.json": audit}, {
                              "analysis/analysis_summary.json": summary_hash,
                              "verification/independent_audit.json": audit_hash}
    json_names = [name for name in ROOT_METADATA if name.endswith(".json")]
    json_names += ["prepared/abc_manifest.json", "prepared/d_manifest.json",
                   "execution/fit/stage_receipt.json", "execution/predict/stage_receipt.json"]
    for name in json_names:
        documents[name], read_hashes[name] = read_json(safe_relative(root, name))
    config, freeze = documents["config.json"], documents["freeze.json"]
    require(config["master_seed"] == 2026110203 and config["repetitions"] == 32 and config["task_count"] == 256 and
            not config.get("manufactured_fixture", False), "The declared production cohort is required")
    require(tuple(config["laws"]) == LAWS and config["noise_levels"] == [0.0, 0.1], "Production condition roster changed")
    require(freeze["status"] == "frozen" and freeze["data_generated"] is False and freeze["D_generated"] is False and
            not freeze.get("manufactured_fixture", False) and freeze["config_sha256"] == read_hashes["config.json"],
            "The source freeze does not bind the production config")
    for record in (summary, audit):
        require(record["config_sha256"] == read_hashes["config.json"] and
                record["freeze_sha256"] == read_hashes["freeze.json"], "Analysis/audit source identity mismatch")
    require(summary["analyzer_sha256"] == freeze["files"]["analysis_code/analyze_soft_state.py"],
            "The reported analyzer differs from the frozen source")
    for name in ("verify_soft_state.py", "independent_oracle.py"):
        require(audit["source_hashes"][name] == freeze["files"]["audit/" + name], "Independent audit source mismatch")

    publication = documents["PUBLICATION.json"]
    require(publication["schema"] == "kahm-soft-state-publication-v1" and publication["status"] == "published" and
            publication["source_files_verified"] is True and publication["freeze_sha256"] == read_hashes["freeze.json"] and
            publication["repository"] == config["source"]["repository"] and bool(publication["branch"]),
            "Source publication receipt does not bind this study")
    for key in ("source_commit", "source_tree_sha"):
        require(re.fullmatch(r"[0-9a-f]{40}", publication[key]) is not None, "Invalid publication Git identity")
    abc, terminal = documents["prepared/abc_manifest.json"], documents["prepared/d_manifest.json"]
    expected = {(rep, law, noise) for rep in range(32) for law in LAWS for noise in (0, 1)}
    manifests = []
    for manifest in (abc, terminal):
        require(len(manifest["tasks"]) == 256, "Incomplete prepared task roster")
        roster = {row["task_id"]: row for row in manifest["tasks"]}
        require(len(roster) == 256 and {(row["repetition"], row["law"], row["noise_index"])
                                      for row in roster.values()} == expected, "Duplicate or missing factorial task")
        for tid, row in roster.items():
            require(isinstance(tid, str) and tid not in (".", "..") and
                    re.fullmatch(r"[A-Za-z0-9_.-]+", tid) is not None, "Unsafe canonical task name")
            require(type(row["repetition"]) is int and 0 <= row["repetition"] <= 31,
                    "Task repetition is not a canonical integer")
        manifests.append(roster)
        require(manifest["publication_sha256"] == read_hashes["PUBLICATION.json"], "Prepared publication link mismatch")
    by_task, terminal_tasks = manifests
    require(set(by_task) == set(terminal_tasks), "ABC/D canonical task names differ")
    for tid in by_task:
        for key in ("repetition", "law", "noise_index", "noise_level"):
            require(by_task[tid][key] == terminal_tasks[tid][key], "ABC/D canonical task metadata differ")
    fits, predictions, release = (documents[name] for name in ("FIT_LOCK.json", "PREDICTION_LOCK.json", "TARGET_RELEASE.json"))
    for record in (abc, terminal, fits, predictions):
        require(record["config_sha256"] == read_hashes["config.json"] and record["freeze_sha256"] == read_hashes["freeze.json"],
                "Stage source links mismatch")
    for lock, phase, manifest_name, execution in (
            (fits, "fits", "prepared/abc_manifest.json", "fit"),
            (predictions, "predictions", "prepared/d_manifest.json", "predict")):
        require(lock["status"] == "sealed" and lock["phase"] == phase and lock["task_count"] == 256 and
                len(lock["task_ids"]) == 256 and set(lock["task_ids"]) == set(by_task) and
                lock["manifest_sha256"] == read_hashes[manifest_name], "Incomplete or mismatched global seal")
        stage_name = "execution/" + execution + "/stage_receipt.json"
        stage = documents[stage_name]
        require(stage["phase"] == execution and stage["status"] == "completed" and stage["total_expected"] == 256 and
                len(stage["completed_tasks"]) == 256 and set(stage["completed_tasks"]) == set(by_task),
                "Incomplete execution stage receipt")
        require(lock["files"][stage_name] == read_hashes[stage_name], "Execution receipt differs from sealed bytes")
    require(terminal["fit_lock_sha256"] == read_hashes["FIT_LOCK.json"] and
            terminal["ABC_manifest_sha256"] == read_hashes["prepared/abc_manifest.json"], "D preparation seal link mismatch")
    require(release["status"] == "released" and release["task_count"] == 256, "Targets have not been released for this complete cohort")
    for field, name in (("fit_lock_sha256", "FIT_LOCK.json"), ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                         ("d_manifest_sha256", "prepared/d_manifest.json")):
        require(release[field] == read_hashes[name], "Target release link mismatch")
    for field, name in (("fit_lock_sha256", "FIT_LOCK.json"), ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                         ("target_release_sha256", "TARGET_RELEASE.json")):
        require(summary[field] == read_hashes[name], "Analysis release link mismatch")
    chronology = [freeze, abc, fits, terminal, predictions, release, summary, audit]
    require(all(time_of(a["created_utc"]) <= time_of(b["created_utc"])
                for a, b in zip(chronology, chronology[1:])), "Recorded stage chronology is inconsistent")
    require(time_of(freeze["created_utc"]) <= time_of(publication["published_utc"]) < time_of(abc["created_utc"]),
            "Recorded source publication did not precede production generation")
    return documents, read_hashes, by_task


def plan_coverage(root, documents, by_task):
    """Partition complete file trees using canonical manifest IDs only."""
    assigned = {archive: set() for archive in ARCHIVES}
    categories = {}

    def add_tree(relative, archive):
        files = walk_files(root, relative)
        assigned[archive].update(files)
        categories[relative] = {"files": files, "archives": [archive]}

    for relative in ("prepared", "predictions", "execution/predict", "analysis", "verification", "readiness"):
        add_tree(relative, DATA_ARCHIVE)
    for name in ROOT_METADATA:
        require(safe_relative(root, name).is_file(), "Missing root evidence record: " + name)
        assigned[DATA_ARCHIVE].add(name)
    # Include frozen non-code contracts/preflight receipts, while the scientific
    # implementation remains in its pinned repository commit and hash map.
    for name in documents["freeze.json"]["files"]:
        if Path(name).suffix.lower() not in SOURCE_SUFFIXES:
            require(Path(name).suffix.lower() in {".json", ".md", ".txt"}, "Unexpected frozen non-code asset: " + name)
            assigned[DATA_ARCHIVE].add(name)
    for name in ("README.md", "requirements.txt"):
        if (root / name).is_file():
            assigned[DATA_ARCHIVE].add(name)

    results = set(walk_files(root, "results"))
    prediction_files = set(categories["predictions"]["files"])
    require({Path(name).parts[1] for name in results} == set(by_task), "Results directory does not cover exactly the manifest tasks")
    require({Path(name).parts[1] for name in prediction_files} == set(by_task), "Prediction directory does not cover exactly the manifest tasks")
    expected_fit_files = {name for name in documents["FIT_LOCK.json"]["files"] if name.startswith("results/")}
    expected_predictions = {name for name in documents["PREDICTION_LOCK.json"]["files"] if name.startswith("predictions/")}
    require(results == expected_fit_files, "Result files differ from the sealed full-state roster")
    require(prediction_files == expected_predictions, "Prediction files differ from the sealed roster")
    for tid, row in by_task.items():
        task_files = {name for name in results if Path(name).parts[1] == tid}
        require({"results/" + tid + "/" + name for name in FIT_FILES}.issubset(task_files),
                "A complete state/B-path/C-diagnostics record is missing for " + tid)
        require({"predictions/" + tid + "/" + name for name in PREDICTION_FILES}.issubset(prediction_files),
                "A complete D forecast record is missing for " + tid)
        archive = LOW_ARCHIVE if row["repetition"] <= 15 else HIGH_ARCHIVE
        assigned[archive].update(task_files)

    fit_logs = set(walk_files(root, "execution/fit"))
    expected_logs = {"execution/fit/" + tid + ".log" for tid in by_task}
    stage_receipt = "execution/fit/stage_receipt.json"
    require(fit_logs == expected_logs | {stage_receipt}, "Fit execution files differ from the canonical log roster")
    predict_logs = set(categories["execution/predict"]["files"])
    require(predict_logs == {"execution/predict/" + tid + ".log" for tid in by_task}
            | {"execution/predict/stage_receipt.json"}, "Prediction execution files differ from the canonical log roster")
    for tid, row in by_task.items():
        archive = LOW_ARCHIVE if row["repetition"] <= 15 else HIGH_ARCHIVE
        assigned[archive].add("execution/fit/" + tid + ".log")
    # Both halves retain the same common receipt/seal; these are deliberately
    # duplicated metadata, never duplicated fitted state or scientific code.
    for archive in (LOW_ARCHIVE, HIGH_ARCHIVE):
        assigned[archive].update((stage_receipt, "FIT_LOCK.json"))
    categories["results"] = {"files": sorted(results), "archives": [LOW_ARCHIVE, HIGH_ARCHIVE]}
    categories["execution/fit"] = {"files": sorted(fit_logs), "archives": [LOW_ARCHIVE, HIGH_ARCHIVE]}

    # Preserve any phase-driver logs kept directly in execution/, if present.
    # No task assignment is guessed for an unexpected nested execution tree.
    for path in sorted((root / "execution").iterdir()):
        if path.name in {"fit", "predict"}:
            continue
        require(path.is_file() and not path.is_symlink() and path.suffix.lower() in {".log", ".json", ".txt"},
                "Unexpected execution artifact requires explicit review: " + str(path))
        assigned[DATA_ARCHIVE].add(path.relative_to(root).as_posix())

    union = set().union(*assigned.values())
    for relative, group in categories.items():
        require(set(group["files"]).issubset(union), "Incomplete archive coverage for " + relative)
    require(all(Path(name).suffix.lower() not in SOURCE_SUFFIXES for name in union), "Source code must not be copied into numerical archives")
    return assigned, categories


def expected_hashes(root, documents, read_hashes):
    """Merge byte identities from source freeze, stage seals and analysis."""
    expected, provenance = {}, {}

    def add(name, value, origin):
        safe_relative(root, name)
        require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA256 declaration: " + name)
        require(name not in expected or expected[name] == value, "Conflicting sealed digest declarations: " + name)
        expected[name] = value
        provenance.setdefault(name, []).append(origin)

    for name, value in read_hashes.items():
        add(name, value, "metadata snapshot read by packager")
    freeze = documents["freeze.json"]
    for name, value in freeze["files"].items():
        add(name, value, "freeze.json/files")
    for manifest_name in ("prepared/abc_manifest.json", "prepared/d_manifest.json"):
        manifest = documents[manifest_name]
        for name, value in manifest["base_files"].items():
            add(name, value, manifest_name + "/base_files")
        add("prepared/seed_manifest.json", manifest["seed_manifest_sha256"], manifest_name + "/seed_manifest_sha256")
        for row in manifest["tasks"]:
            if manifest_name.endswith("abc_manifest.json"):
                add(row["input"], row["sha256"], manifest_name + "/task input")
            else:
                add(row["query"], row["query_sha256"], manifest_name + "/task query")
                add(row["target"], row["target_sha256"], manifest_name + "/task target")
    for lock in ("FIT_LOCK.json", "PREDICTION_LOCK.json"):
        for name, value in documents[lock]["files"].items():
            add(name, value, lock + "/files")
    for name, value in documents["analysis/analysis_summary.json"]["tables"].items():
        add("analysis/" + name, value["sha256"], "analysis/analysis_summary.json/tables")
    return expected, provenance


def snapshot_files(root, assigned, expected, provenance):
    snapshots = {}
    all_names = set().union(*assigned.values())
    for name in sorted(all_names):
        duplicate_of = None
        if name.endswith(".tmp"):
            # A completed atomic-write copy may remain after its canonical
            # record has been installed. Preserve it only when BOTH exact byte
            # identities are already sealed and the canonical record is also
            # covered. An unsealed/inconsistent temporary still aborts.
            duplicate_of = name[:-4]
            require(name in expected and duplicate_of in expected and duplicate_of in all_names and
                    expected[name] == expected[duplicate_of],
                    "Unsealed or inconsistent temporary artifact: " + name)
        path = safe_relative(root, name)
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "Missing/nonregular archive source: " + name)
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), "Artifact changed while hashing: " + name)
        require(name not in expected or sha == expected[name], "Sealed byte identity mismatch: " + name)
        snapshots[name] = {"bytes": after.st_size, "sha256": sha,
                           "digest_provenance": provenance.get(name, ["retained unsealed ancillary record; packaging snapshot only"]),
                           "archives": [archive for archive in ARCHIVES if name in assigned[archive]]}
        if duplicate_of is not None:
            snapshots[name]["sealed_identical_copy_of"] = duplicate_of
    # Ensure every declared generated artifact is covered. Scientific source is
    # verified separately and intentionally omitted from these numerical ZIPs.
    for name in expected:
        if name.startswith(("prepared/", "results/", "predictions/", "analysis/", "execution/")):
            require(name in snapshots, "Declared numerical evidence was omitted: " + name)
    return snapshots


def verify_source_maps(root, freeze):
    groups = {"study": (root, freeze["files"]),
              "dimension_forecasting": (root.parent / "dimension_forecasting", freeze["inherited_files"]),
              "policy_augmentation_20261101": (root.parent / "policy_augmentation_20261101", freeze["numerical_solver_files"])}
    result = {}
    for label, (source_root, files) in groups.items():
        source_root = source_root.resolve()
        for name, expected in files.items():
            path = safe_relative(source_root, name)
            require(path.is_file() and digest(path) == expected, "Frozen source differs at packaging: " + label + "/" + name)
        result[label] = {"file_count": len(files), "sha256_by_relative_path": files}
    return result


def zip_info(name, *, compressed):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build_zip(root, destination, names, snapshots, payloads, limit):
    members = {}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        for name in sorted(names):
            path = safe_relative(root, name)
            member = PREFIX + name
            require(member not in members, "Duplicate archive member")
            # NPZ is already a compressed ZIP, so store it without a redundant
            # compression pass. JSON/CSV/logs remain losslessly compressed.
            info = zip_info(member, compressed=path.suffix.lower() != ".npz")
            hashed, count = hashlib.sha256(), 0
            before = path.stat()
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(BLOCK), b""):
                    target.write(block)
                    hashed.update(block)
                    count += len(block)
            after = path.stat()
            require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_ino, after.st_size, after.st_mtime_ns), "Artifact changed during archive copy: " + name)
            require(count == snapshots[name]["bytes"] and hashed.hexdigest() == snapshots[name]["sha256"],
                    "Archive copy does not match the verified snapshot: " + name)
            members[member] = {"study_relative_path": name, "bytes": count, "sha256": hashed.hexdigest(),
                               "kind": "retained study record"}
            require(destination.stat().st_size < limit, "Archive exceeds the size limit: " + destination.name)
        for name, payload in sorted(payloads.items()):
            member = PREFIX + "_archive_metadata/" + name
            require(member not in members, "Duplicate generated archive metadata")
            archive.writestr(zip_info(member, compressed=True), payload)
            members[member] = {"study_relative_path": None, "bytes": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest(), "kind": "reporting-only package metadata"}
    require(destination.stat().st_size < limit,
            f"{destination.name} is {destination.stat().st_size:,} bytes; must be strictly below {limit:,}. No records were dropped.")

    # Reopen the completed ZIP and verify roster, lengths, CRC during reading,
    # and SHA256 for every member. Never extract or execute archived content.
    with zipfile.ZipFile(destination, "r") as archive:
        infos = archive.infolist()
        require(len(infos) == len(members) and {info.filename for info in infos} == set(members),
                "ZIP member coverage differs after closing: " + destination.name)
        for info in infos:
            require(not info.is_dir() and info.file_size == members[info.filename]["bytes"], "ZIP member length mismatch")
            hashed, count = hashlib.sha256(), 0
            with archive.open(info, "r") as stream:
                for block in iter(lambda: stream.read(BLOCK), b""):
                    hashed.update(block)
                    count += len(block)
            require(count == info.file_size and hashed.hexdigest() == members[info.filename]["sha256"],
                    "ZIP member SHA256 mismatch: " + info.filename)
            members[info.filename]["compressed_bytes"] = info.compress_size
    return {"bytes": destination.stat().st_size, "sha256": digest(destination),
            "member_count": len(members), "uncompressed_bytes": sum(row["bytes"] for row in members.values()),
            "member_verification": "Every member reopened and SHA256 verified; ZIP CRC checked on read",
            "members": members}


def package(root, output, limit):
    require(not output.exists(), "Output directory already exists; refusing overwrite: " + str(output))
    for repository in {REPOSITORY.resolve(), root.parent.parent.resolve()}:
        require(not output.is_relative_to(repository) and not repository.is_relative_to(output),
                "Numerical archives must be outside the repository")
    require(0 < limit <= LIMIT, "Maximum archive size must be positive and at most 500,000,000 bytes")
    documents, read_hashes, by_task = validate_completed_study(root)
    assigned, categories = plan_coverage(root, documents, by_task)
    expected, provenance = expected_hashes(root, documents, read_hashes)
    snapshots = snapshot_files(root, assigned, expected, provenance)
    source_maps = verify_source_maps(root, documents["freeze.json"])
    publication = documents["PUBLICATION.json"]
    identity = {"schema": "soft-state-numerical-source-identity-v1",
        "repository": publication["repository"], "branch": publication["branch"],
        "source_commit": publication["source_commit"], "source_tree_sha": publication["source_tree_sha"],
        "source_parent_commit": documents["freeze.json"]["source_parent_commit"],
        "publication_sha256": read_hashes["PUBLICATION.json"], "freeze_sha256": read_hashes["freeze.json"],
        "config_sha256": read_hashes["config.json"],
        "analysis_summary_sha256": read_hashes["analysis/analysis_summary.json"],
        "independent_audit_sha256": read_hashes["verification/independent_audit.json"],
        "packager_sha256": digest(SCRIPT), "reporting_only_packager_outside_original_freeze": True}
    roster = [{"task_id": tid, "repetition": row["repetition"], "law": row["law"],
               "noise_index": row["noise_index"], "noise_level": row["noise_level"],
               "fitted_model_archive": LOW_ARCHIVE if row["repetition"] <= 15 else HIGH_ARCHIVE}
              for tid, row in sorted(by_task.items(), key=lambda item: (item[1]["repetition"], item[0]))]
    halves = {LOW_ARCHIVE: [row for row in roster if row["repetition"] <= 15],
              HIGH_ARCHIVE: [row for row in roster if row["repetition"] >= 16]}
    require(len(halves[LOW_ARCHIVE]) == len(halves[HIGH_ARCHIVE]) == 128, "Model halves do not each cover 128 canonical tasks")
    metadata = {}
    for archive in ARCHIVES:
        records = roster if archive == DATA_ARCHIVE else halves[archive]
        metadata[archive] = {archive.removesuffix(".zip") + ".json": json_bytes({
            **identity, "archive": archive, "canonical_task_roster": records,
            "source_file_count": len(assigned[archive]),
            "common_fit_metadata": ["FIT_LOCK.json", "execution/fit/stage_receipt.json"] if archive != DATA_ARCHIVE else [],
            "scope": "Lossless retained records; no fitting, selection, scoring, filtering, or numerical conversion."})}
    metadata[DATA_ARCHIVE]["source_hashes.json"] = json_bytes({**identity, "verified_frozen_source_maps": source_maps,
        "code_omitted_intentionally": "Retrieve exact scientific implementation from the pinned repository commit; these ZIPs contain numerical records and non-code evidence only."})
    metadata[DATA_ARCHIVE]["README.txt"] = (
        "SOFT-STATE ABLATION: COMPLETE NUMERICAL RECORDS\n\n"
        "Use numerical_archive_index.json beside the three ZIPs to verify archive and member SHA256 hashes.\n"
        "Extract all three ZIPs into one chosen data directory. Paths begin with extensions/soft_state_ablation_20261102/.\n"
        "The two fitted-model archives repeat identical FIT_LOCK.json and fit-stage receipt metadata; all fitted task state is partitioned without duplication.\n"
        "Data_and_Forecasts contains prepared physical episodes/windows/targets, D forecasts, prediction logs, complete analysis, independent audit and readiness records.\n"
        "The fitted-model halves contain complete portable states, every B prediction path, C diagnostics, selection/numerical records, and per-task fit logs.\n"
        "The source_hashes.json record pins the scientific code and inherited dependencies. Code is not duplicated here.\n"
        "The packager validates recorded stage/source links and checks retained bytes. It does not rerun the independent audit or establish scientific correctness.\n"
        "Log files not covered by scientific seals are preserved as packaging snapshots and identified as ancillary records in the index.\n"
        "A retained .tmp record is included only if it and its canonical counterpart are already sealed with the same exact SHA256; the index identifies such redundant copies.\n"
        "Recorded order and source publication are procedural evidence, not cryptographic target isolation.\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".soft_state_numerical_archive-", dir=output.parent))
    try:
        archive_info = {}
        for archive in ARCHIVES:
            archive_info[archive] = build_zip(root, staging / archive, assigned[archive], snapshots, metadata[archive], limit)
            print(json.dumps({"archive_verified": archive, "bytes": archive_info[archive]["bytes"],
                              "members": archive_info[archive]["member_count"]}), flush=True)
        archived = {row["study_relative_path"] for info in archive_info.values() for row in info["members"].values()
                    if row["study_relative_path"] is not None}
        require(archived == set(snapshots), "Final archive union does not cover the entire verified source snapshot")
        index = {"schema": "soft-state-numerical-archive-index-v1", "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(), "reporting_only": True,
            "identity": identity, "maximum_archive_bytes_exclusive": limit,
            "canonical_task_roster": roster, "tasks": 256, "independent_repetitions": 32,
            "archives": archive_info, "source_files": snapshots,
            "coverage": {"all_requested_trees_complete": True,
                "unique_source_file_count": len(snapshots),
                "unique_source_bytes": sum(row["bytes"] for row in snapshots.values()),
                "categories": categories,
                "intentional_duplicate_metadata": {name: row["archives"] for name, row in snapshots.items()
                                                   if len(row["archives"]) > 1},
                "sealed_identical_temporary_copies": {name: row["sealed_identical_copy_of"]
                    for name, row in snapshots.items() if "sealed_identical_copy_of" in row},
                "model_tasks_00_15": len(halves[LOW_ARCHIVE]), "model_tasks_16_31": len(halves[HIGH_ARCHIVE]),
                "full_state_and_B_C_diagnostics_preserved": True,
                "scientific_code_omitted_with_exact_source_hashes": True},
            "checks": {"passed_independent_audit_bound_to_exact_analysis": True,
                "source_publication_and_recorded_stage_links": True,
                "sealed_payload_hashes": True, "frozen_source_and_dependency_hashes": True,
                "every_archive_member_reopened_and_sha256_verified": True,
                "all_archives_strictly_below_size_limit": True},
            "limits": ["This reporting utility is outside the original prospective scientific freeze.",
                "Metadata/status/hash checks do not rerun the independent computational audit.",
                "Retained ancillary logs not covered by scientific seals are packaging snapshots, explicitly labeled.",
                "Only declared study trees and in-study execution logs are included; external shell/session logs are not inferred.",
                "Source chronology is recorded procedural evidence, not cryptographic target isolation.",
                "Scientific source code is retrieved from the pinned Git commit and is not duplicated in these numerical ZIPs."]}
        payload = json_bytes(index)
        require(len(payload) < limit, "Archive index exceeds the maximum single-file size")
        index_path = staging / "numerical_archive_index.json"
        with index_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        require(index_path.read_bytes() == payload, "Archive index round trip differs")
        require(not output.exists(), "Output directory appeared during packaging; refusing overwrite")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"status": "complete", "output_dir": str(output),
            "index_sha256": digest(output / "numerical_archive_index.json"),
            "archives": {name: {key: info[key] for key in ("bytes", "sha256", "member_count")}
                         for name, info in archive_info.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="NEW directory outside the repository; receives three verified ZIPs and their index")
    parser.add_argument("--max-archive-bytes", type=int, default=LIMIT,
                        help="Exclusive byte limit, no greater than 500000000 (default)")
    args = parser.parse_args()
    print(json.dumps(package(args.study_root.resolve(), args.output_dir.resolve(), args.max_archive_bytes), indent=2))


if __name__ == "__main__":
    main()
