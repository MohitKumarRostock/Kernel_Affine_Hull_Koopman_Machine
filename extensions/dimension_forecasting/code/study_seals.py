#!/usr/bin/env python3
"""Create immutable source, fit, prediction and target-release records.

Hashing target-file bytes for identity is permitted before release; this module
never loads target arrays or computes forecasting losses. The analysis program
requires the final target-release record before opening terminal outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from study_contract import expected_prediction_keys

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {"historical2", "native1", "native2", "native4", "native8", "native12", "native20"}
STATE_IDS = CANDIDATES | {"shared"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def resolved(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Invalid confined file: " + str(relative))
    return path


def write(path, value):
    require(not path.exists(), "Refusing to replace seal: " + str(path))
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(temporary.read_bytes() == data, "Seal write readback failed")
    temporary.replace(path)


def verify_files(root, record):
    require(isinstance(record.get("files"), dict) and record["files"], "Empty file-identity map")
    for relative, expected in record["files"].items():
        require(sha(resolved(root, relative)) == expected, "Identity mismatch: " + relative)


def add_file(root, files, path, expected=None):
    path = path.resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing confined artifact: " + str(path))
    relative = str(path.relative_to(root))
    actual = sha(path)
    require(expected is None or expected == actual, "Artifact changed: " + relative)
    require(relative not in files or files[relative] == actual, "Conflicting artifact identity")
    files[relative] = actual


def task_roster(root, manifest):
    config = read(root / "config.json")
    tasks = manifest["tasks"]
    expected_count = (len(config["laws"]) * len(config["histories"])
                      * len(config["noise_levels"]) * config["repetitions"])
    require(expected_count == config["task_count"] == len(tasks), "Incomplete task count")
    require(len({t["task_id"] for t in tasks}) == expected_count, "Duplicate task identifier")
    actual = {(t["law"], int(t["history"]), int(t["noise_index"]), int(t["repetition"])) for t in tasks}
    expected = {(law, history, noise, repetition) for law in config["laws"]
                for history in config["histories"] for noise in range(len(config["noise_levels"]))
                for repetition in range(config["repetitions"])}
    require(actual == expected, "Incomplete/extra task roster")
    for task in tasks:
        require(task["noise_level"] == config["noise_levels"][task["noise_index"]], "Noise identity differs")
    return tasks


def source_freeze(root):
    require(not (root / "prepared/abc_manifest.json").exists(), "Production preparation predates freeze")
    require(not (root / "prepared/d_manifest.json").exists(), "D preparation predates freeze")
    require(not (root / "results").exists(), "Production fitting predates freeze")
    config = read(root / "config.json")
    review_path = root / "audit/prefreeze_review.json"
    review = read(review_path)
    require(review.get("status") == "passed", "Independent preflight has not passed")
    files = {}
    for relative in ("config.json", "PROTOCOL.md", "requirements.txt"):
        add_file(root, files, root / relative)
    for directory in ("code", "analysis_code", "runtime", "vendor_original", "readiness", "audit"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                add_file(root, files, path)
    for path in sorted((root / "verification").glob("*.json")):
        add_file(root, files, path)
    record = {"schema": "kahm-dimension-source-freeze-v1", "status": "frozen",
              "phase": "source", "created_utc": now(), "task_count": config["task_count"],
              "config_sha256": sha(root / "config.json"), "files": files,
              "production_data_generated": False, "public_preregistration": False,
              "independent_preflight_sha256": sha(review_path),
              "scope": "Protocol, analysis, preparation, numerical runtime and successful manufactured gates"}
    write(root / "freeze.json", record)
    return record


def verify_freeze(root):
    record = read(root / "freeze.json")
    require(record["status"] == "frozen", "Invalid source freeze")
    require(record["config_sha256"] == sha(root / "config.json"), "Frozen configuration changed")
    if (root / "source_distribution.json").exists():
        from source_distribution import verify_source_distribution
        verification = verify_source_distribution(root, record)
        record = dict(record)
        record["distribution_verification"] = verification
    else:
        verify_files(root, record)
    return record


def fits_lock(root):
    verify_freeze(root)
    require(not (root / "prepared/d_manifest.json").exists(), "D generated before fitted-state lock")
    config = read(root / "config.json")
    manifest_path = root / "prepared/abc_manifest.json"
    manifest = read(manifest_path)
    require(manifest["freeze_sha256"] == sha(root / "freeze.json")
            and manifest["config_sha256"] == sha(root / "config.json"), "ABC predecessor identity differs")
    tasks = task_roster(root, manifest)
    files = {}
    add_file(root, files, manifest_path)
    add_file(root, files, root / "prepared/seed_manifest.json", manifest["seed_manifest_sha256"])
    for relative, digest in manifest["base_files"].items():
        add_file(root, files, root / relative, digest)
    require(not list((root / "results").rglob("fit_failure.json")), "Unexpected fitting failure exists")
    require({p.name for p in (root / "results").iterdir() if p.is_dir()} == {t["task_id"] for t in tasks},
            "Result directory roster differs")
    for task in tasks:
        add_file(root, files, root / task["input"], task["sha256"])
        directory = root / "results" / task["task_id"]
        summary = read(directory / "fit_summary.json")
        require(summary["status"] == "completed" and summary["task"] == task, "Incomplete or mismatched fit")
        require(summary["freeze_sha256"] == sha(root / "freeze.json"), "Fit source freeze differs")
        require(set(summary["candidates"]) == CANDIDATES, "Incomplete candidate roster")
        require(set(summary["contexts"]) == {"historical2","native20","selected"}, "Incomplete context roster")
        require(set(summary["selections"]) == set(config['geometry_methods']), "Incomplete five-method cap selection")
        require(summary['shared']['fit_diagnostics']['raw_grid_failures'] == [], "Unexpected raw grid failure")
        require('states/shared.pkl.gz' in summary['files'], "Shared competitor state missing")
        require(summary["historical_native2_identity"]["status"] == "passed", "Cap2 identity control failed")
        if task["history"] == 12:
            require(summary["native12_native20_identity"]["status"] == "passed", "Width12 cap12/cap20 identity failed")
        pre = read(directory / "pre_c_selection.json")
        require(pre["C_arrays_loaded"] is False and pre["D_generated_or_loaded"] is False,
                "Invalid pre-C cap selection stage")
        require(pre["selections"] == summary["selections"], "Cap choices changed after pre-C seal")
        require(pre["completed_utc"] <= summary["completed_utc"], "Pre-C chronology is inconsistent")
        for relative, digest in summary["files"].items():
            add_file(root, files, directory / relative, digest)
        add_file(root, files, directory / "fit_summary.json")
    record = {"schema": "kahm-dimension-fit-lock-v1", "status": "sealed", "phase": "fits",
              "created_utc": now(), "task_count": len(tasks), "task_ids": [t["task_id"] for t in tasks],
              "freeze_sha256": sha(root / "freeze.json"), "config_sha256": sha(root / "config.json"),
              "ABC_manifest_sha256": sha(manifest_path), "files": files,
              "D_generated": False, "unexpected_fit_failures": 0,
              "candidate_count_per_task": len(CANDIDATES)}
    write(root / "FIT_LOCK.json", record)
    return record


def predictions_lock(root):
    verify_freeze(root)
    fit_lock = read(root / "FIT_LOCK.json")
    require(fit_lock["status"] == "sealed" and fit_lock["phase"] == "fits", "Invalid fitted-state lock")
    require(fit_lock["freeze_sha256"] == sha(root / "freeze.json")
            and fit_lock["config_sha256"] == sha(root / "config.json"), "Fit-lock predecessor identity differs")
    verify_files(root, fit_lock)
    manifest_path = root / "prepared/d_manifest.json"
    config=read(root/'config.json')
    manifest = read(manifest_path)
    require(manifest["freeze_sha256"] == sha(root / "freeze.json")
            and manifest["config_sha256"] == sha(root / "config.json")
            and manifest["fit_lock_sha256"] == sha(root / "FIT_LOCK.json"), "D predecessor identity differs")
    tasks = task_roster(root, manifest)
    require(set(fit_lock["task_ids"]) == {t["task_id"] for t in tasks}, "D roster differs from fit roster")
    files = {}
    add_file(root, files, manifest_path)
    require({p.name for p in (root / "predictions").iterdir() if p.is_dir()} == {t["task_id"] for t in tasks},
            "Prediction directory roster differs")
    for task in tasks:
        directory = root / "predictions" / task["task_id"]
        summary = read(directory / "predict_summary.json")
        require(summary["status"] == "completed", "Incomplete prediction task")
        require(summary["task"]["task_id"] == task["task_id"], "Prediction task mismatch")
        require(summary["fit_lock_sha256"] == sha(root / "FIT_LOCK.json"), "Prediction fit lock differs")
        require(summary["D_target_file_opened"] is False, "Prediction opened D targets")
        require(set(summary['prediction_array_sha256'])==expected_prediction_keys(config),
                'Incomplete prediction-array roster')
        require(summary['prediction_array_count']==len(expected_prediction_keys(config)),
                'Prediction-array count differs')
        require(summary['D_rows']==task['rows'] and summary['horizon_count']==len(config['horizons']),
                'Prediction shape differs from frozen D design')
        require(set(summary["state_identity"]) == STATE_IDS, "Incomplete state-identity roster")
        for value in summary["state_identity"].values():
            require(value["unchanged"] and value["before_sha256"] == value["after_sha256"], "Query mutated a fitted state")
        add_file(root, files, root / task["query"], task["query_sha256"])
        require(summary["query_file_sha256"] == task["query_sha256"], "Query input differs")
        require(summary["fit_summary_sha256"] == sha(root / "results" / task["task_id"] / "fit_summary.json"),
                "Prediction used different fitted state")
        add_file(root, files, directory / "predictions.npz", summary["prediction_file_sha256"])
        add_file(root, files, directory / "predict_summary.json")
    record = {"schema": "kahm-dimension-prediction-lock-v1", "status": "sealed", "phase": "predictions",
              "created_utc": now(), "task_count": len(tasks), "task_ids": [t["task_id"] for t in tasks],
              "freeze_sha256": sha(root / "freeze.json"), "config_sha256": sha(root / "config.json"),
              "fit_lock_sha256": sha(root / "FIT_LOCK.json"), "D_manifest_sha256": sha(manifest_path),
              "files": files, "D_scoring_started": False,
              "candidate_count_per_task": len(CANDIDATES)}
    write(root / "PREDICTION_LOCK.json", record)
    return record


def target_release(root):
    verify_freeze(root)
    prediction_lock = read(root / "PREDICTION_LOCK.json")
    require(prediction_lock["status"] == "sealed" and prediction_lock["phase"] == "predictions",
            "Invalid prediction lock")
    require(prediction_lock["freeze_sha256"] == sha(root / "freeze.json")
            and prediction_lock["config_sha256"] == sha(root / "config.json")
            and prediction_lock["fit_lock_sha256"] == sha(root / "FIT_LOCK.json"),
            "Prediction-lock predecessor identity differs")
    fit_lock = read(root / "FIT_LOCK.json")
    require(fit_lock["status"] == "sealed" and fit_lock["phase"] == "fits"
            and fit_lock["freeze_sha256"] == sha(root / "freeze.json")
            and fit_lock["config_sha256"] == sha(root / "config.json"), "Released fit-lock identity differs")
    verify_files(root, fit_lock)
    verify_files(root, prediction_lock)
    manifest_path = root / "prepared/d_manifest.json"
    manifest = read(manifest_path)
    require(prediction_lock["D_manifest_sha256"] == sha(manifest_path)
            and manifest["fit_lock_sha256"] == sha(root / "FIT_LOCK.json"), "Released D manifest identity differs")
    tasks = task_roster(root, manifest)
    require(set(prediction_lock["task_ids"]) == {t["task_id"] for t in tasks}, "Target roster differs")
    files = {}
    add_file(root, files, manifest_path)
    for relative, digest in manifest["base_files"].items():
        add_file(root, files, root / relative, digest)
    for task in tasks:
        add_file(root, files, root / task["target"], task["target_sha256"])
    record = {"schema": "kahm-dimension-target-release-v1", "status": "sealed", "phase": "target_release",
              "created_utc": now(), "task_count": len(tasks), "task_ids": [t["task_id"] for t in tasks],
              "freeze_sha256": sha(root / "freeze.json"), "config_sha256": sha(root / "config.json"),
              "fit_lock_sha256": sha(root / "FIT_LOCK.json"),
              "prediction_lock_sha256": sha(root / "PREDICTION_LOCK.json"),
              "D_manifest_sha256": sha(manifest_path), "files": files,
              "targets_read_for_scoring_before_release": False}
    write(root / "TARGET_RELEASE.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fits", "predictions", "release", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    functions = {"freeze": source_freeze, "fits": fits_lock, "predictions": predictions_lock,
                 "release": target_release, "verify": verify_freeze}
    result = functions[args.phase](root)
    summary = {"phase": args.phase, "status": result["status"],
               "task_count": result.get("task_count"), "file_count": len(result["files"])}
    if "distribution_verification" in result:
        summary["status"] = "verified_available_source"
        summary["file_count_scope"] = "historical_freeze_entries"
        summary["distribution_verification"] = result["distribution_verification"]
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
