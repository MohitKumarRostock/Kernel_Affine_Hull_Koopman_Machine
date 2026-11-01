#!/usr/bin/env python3
"""Freeze locally, execute in isolated processes, and replay the fixed study.

The local freeze is an audit checkpoint, not public preregistration.  Results,
including failed tasks, are retained.  No completed result is overwritten.
"""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import numpy as np
from extensions.reconstruction_mechanism.data import array_hash, sha256_file, utcnow, write_json


def environment():
    return {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "PYTHONHASHSEED")},
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def pin_one_cpu():
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {allowed[0]})


def frozen_paths(study_dir):
    inherited = json.loads((REPOSITORY / "extensions/direct_reconstruction/results/freeze.json").read_text())
    paths = {REPOSITORY / p for p in inherited["scientific_files"]}
    for pattern in ("*.py", "*.md", "*.json", "requirements.txt"):
        paths.update(study_dir.glob(pattern))
    paths.add(study_dir / "prepared_inputs/manifest.json")
    paths.update((study_dir / "prepared_inputs/tasks").glob("*.npz"))
    return sorted(paths)


def freeze(study_dir):
    study_dir = Path(study_dir).resolve()
    target = study_dir / "freeze.json"
    if target.exists():
        raise FileExistsError("An existing freeze is immutable; do not replace it")
    required = ["config.json", "PROTOCOL.md", "ALGEBRA.md", "algebra.py", "algebra_results.json",
                "models.py", "test_models.py", "test_inputs.py", "data.py", "run.py", "analyze.py",
                "prepared_inputs/manifest.json"]
    for name in required:
        if not (study_dir / name).is_file():
            raise FileNotFoundError(study_dir / name)
    algebra = json.loads((study_dir / "algebra_results.json").read_text())
    if algebra.get("status") != "passed" or algebra.get("algebra_decision") != "GO_TO_ONE_MATCHED_FORECAST_STUDY":
        raise ValueError("The algebraic precheck must justify a distinct candidate before reserved evaluation")
    manifest = json.loads((study_dir / "prepared_inputs/manifest.json").read_text())
    if manifest["development_only"] or manifest["task_count"] != 192:
        raise ValueError("Freeze requires the complete reserved roster, not development inputs")
    if (study_dir / "results").exists():
        raise FileExistsError("Reserved results already exist; cannot claim a prospective freeze")
    record = {
        "schema_version": "reconstruction-mechanism-v1", "created_utc": utcnow(),
        "status": "Local prospective source/configuration/input freeze; no reserved model outcomes evaluated",
        "public_preregistration": False,
        "publication_contract": "Publish final completed research, not a roadmap-only draft",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "task_count": manifest["task_count"],
        "files": [{"path": str(p.relative_to(REPOSITORY)), "sha256": sha256_file(p)} for p in frozen_paths(study_dir)],
        "environment": environment(),
    }
    write_json(target, record)
    print(json.dumps({"freeze": str(target), "files": len(record["files"]),
                      "sha256": sha256_file(target)}), flush=True)
    return record


def verify_freeze(study_dir):
    study_dir = Path(study_dir).resolve()
    frozen = json.loads((study_dir / "freeze.json").read_text())
    if frozen["schema_version"] != "reconstruction-mechanism-v1":
        raise ValueError("Unknown freeze schema")
    for item in frozen["files"]:
        path = REPOSITORY / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen file changed: {item['path']}")
    return frozen


def find_task(study_dir, task_id, *, development=False):
    folder = "development_inputs" if development else "prepared_inputs"
    manifest = json.loads((study_dir / folder / "manifest.json").read_text())
    tasks = [t for t in manifest["tasks"] if t["task_id"] == task_id]
    if len(tasks) != 1 or bool(manifest["development_only"]) != bool(development):
        raise ValueError("Task identity/development scope mismatch")
    return tasks[0]


def worker(study_dir, task_id, output, *, development=False):
    study_dir, output = Path(study_dir).resolve(), Path(output).resolve()
    pin_one_cpu()
    if not development:
        verify_freeze(study_dir)
    task = find_task(study_dir, task_id, development=development)
    config = json.loads((study_dir / "config.json").read_text())
    input_path = study_dir / task["input"]
    if sha256_file(input_path) != task["sha256"]:
        raise ValueError("Task input changed")
    if output.exists():
        raise FileExistsError(f"Never overwrite an existing result: {output}")
    output.mkdir(parents=True)
    started, wall, cpu = utcnow(), time.perf_counter(), time.process_time()
    summary = {"schema_version": config["schema_version"], "task": task,
               "development_only": bool(development), "started_utc": started,
               "environment": environment(), "input_sha256": task["sha256"],
               "freeze_sha256": None if development else sha256_file(study_dir / "freeze.json")}
    try:
        from extensions.reconstruction_mechanism.models import METHODS, fit_suite
        if list(METHODS) != config["methods"]:
            raise ValueError("Model roster differs from frozen configuration")
        with np.load(input_path, allow_pickle=False) as arrays:
            fit_wall, fit_cpu = time.perf_counter(), time.process_time()
            suite = fit_suite(*(arrays[k] for k in ("A_X", "A_Y", "B_X", "B_Y", "C_X", "C_Y")),
                              seed=task["fit_seed"], config=config["model"])
            fit_elapsed = {"wall_seconds": time.perf_counter() - fit_wall,
                           "cpu_seconds": time.process_time() - fit_cpu}
            # Serialize the selected state BEFORE reading reserved query arrays.
            state = gzip.compress(pickle.dumps(suite, protocol=5), compresslevel=6, mtime=0)
            (output / "fitted_suite.pkl.gz").write_bytes(state)
            prediction_wall, prediction_cpu = time.perf_counter(), time.process_time()
            result = suite.predict(arrays["D_X"])
            predict_elapsed = {"wall_seconds": time.perf_counter() - prediction_wall,
                               "cpu_seconds": time.process_time() - prediction_cpu}
            # Targets first enter this wrapper after every forecast is fixed.
            target = arrays["D_Y"]
            predictions = result["predictions"]
            if list(predictions) != config["methods"] or any(p.shape != target.shape for p in predictions.values()):
                raise ValueError("Incomplete prediction roster or shape mismatch")
            np.savez_compressed(output / "predictions.npz", Y=target, **predictions)
            summary.update(status="completed", target_sha256=array_hash(target),
                           prediction_array_sha256={m: array_hash(p) for m, p in predictions.items()},
                           fit_diagnostics=suite.diagnostics,
                           prediction_diagnostics=result["diagnostics"],
                           resources={"fit": suite.resources, "prediction": result["resources"],
                                      "suite_fit_including_diagnostics": fit_elapsed,
                                      "suite_predict_including_diagnostics": predict_elapsed,
                                      "compressed_full_suite_bytes": len(state),
                                      "process_cumulative_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                      "rss_scope": "Whole fresh task process including imports, all ten arms and shared diagnostics; not per-method RAM"},
                           files={name: sha256_file(output / name) for name in
                                  ("predictions.npz", "fitted_suite.pkl.gz")})
    except Exception as exc:
        summary.update(status="failed", exception=repr(exc), traceback=traceback.format_exc())
        (output / "exception.txt").write_text(summary["traceback"])
    summary.update(completed_utc=utcnow(),
                   process_elapsed={"wall_seconds": time.perf_counter() - wall,
                                    "cpu_seconds": time.process_time() - cpu})
    write_json(output / "summary.json", summary)
    print(json.dumps({"task_id": task_id, "status": summary["status"],
                      "wall_seconds": summary["process_elapsed"]["wall_seconds"]}), flush=True)
    return 0 if summary["status"] == "completed" else 1


def execute(study_dir, freeze_commit, *, resume=False):
    study_dir = Path(study_dir).resolve()
    verify_freeze(study_dir)
    if len(freeze_commit) != 40 or any(c not in "0123456789abcdef" for c in freeze_commit):
        raise ValueError("Supply the full local commit containing the prospective freeze")
    relative = str((study_dir / "freeze.json").relative_to(REPOSITORY))
    committed = subprocess.check_output(["git", "show", f"{freeze_commit}:{relative}"], cwd=REPOSITORY)
    if committed != (study_dir / "freeze.json").read_bytes():
        raise ValueError("The supplied commit does not contain this exact freeze")
    results = study_dir / "results"
    record_path = results / "execution.json"
    if record_path.exists():
        if not resume:
            raise FileExistsError("Use --resume only for the identical interrupted execution")
        record = json.loads(record_path.read_text())
        if record["local_freeze_commit"] != freeze_commit or record["freeze_sha256"] != sha256_file(study_dir / "freeze.json"):
            raise ValueError("Resume identity mismatch")
    else:
        record = {"started_utc": utcnow(), "local_freeze_commit": freeze_commit,
                  "freeze_sha256": sha256_file(study_dir / "freeze.json"),
                  "public_preregistration": False, "environment": environment(),
                  "task_execution": "Fresh subprocess per task; serial; each worker pins one CPU and one numerical thread",
                  "events": []}
        write_json(record_path, record)
    manifest = json.loads((study_dir / "prepared_inputs/manifest.json").read_text())
    env = dict(os.environ, PYTHONHASHSEED="0")
    for index, task in enumerate(manifest["tasks"]):
        output = results / task["task_id"]
        if output.exists():
            if not resume or not (output / "summary.json").exists():
                raise FileExistsError(f"Existing/incomplete task preserved; explicit documented repair required: {output}")
            summary = json.loads((output / "summary.json").read_text())
            if summary["input_sha256"] != task["sha256"] or summary["freeze_sha256"] != record["freeze_sha256"]:
                raise ValueError("Existing result identity mismatch")
            if summary["status"] != "completed":
                print(json.dumps({"preserved_failed_task": task["task_id"]}), flush=True)
            continue
        proc = subprocess.run([sys.executable, "-m", "extensions.reconstruction_mechanism.run", "worker",
                               "--study-dir", str(study_dir), "--task-id", task["task_id"],
                               "--output", str(output)], cwd=REPOSITORY, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if output.exists():
            (output / "stdout.txt").write_text(proc.stdout)
            (output / "stderr.txt").write_text(proc.stderr)
        record["events"].append({"task_id": task["task_id"], "returncode": proc.returncode,
                                  "completed_utc": utcnow()})
        write_json(record_path, record)
        print(json.dumps({"task": index + 1, "of": len(manifest["tasks"]),
                          "task_id": task["task_id"], "returncode": proc.returncode}), flush=True)
    record["completed_utc"] = utcnow()
    write_json(record_path, record)
    verify_freeze(study_dir)


def replay(study_dir, *, task_id=None):
    study_dir = Path(study_dir).resolve()
    pin_one_cpu()
    verify_freeze(study_dir)
    config = json.loads((study_dir / "config.json").read_text())
    manifest = json.loads((study_dir / "prepared_inputs/manifest.json").read_text())
    records = []
    for task in manifest["tasks"]:
        if task_id and task["task_id"] != task_id:
            continue
        output = study_dir / "results" / task["task_id"]
        summary = json.loads((output / "summary.json").read_text())
        for name, expected in summary["files"].items():
            if sha256_file(output / name) != expected:
                raise ValueError("Saved result/state checksum differs")
        # Load only trusted states generated by this frozen local study.
        suite = pickle.loads(gzip.decompress((output / "fitted_suite.pkl.gz").read_bytes()))
        with np.load(study_dir / task["input"], allow_pickle=False) as inputs:
            result = suite.predict(inputs["D_X"])["predictions"]
        with np.load(output / "predictions.npz", allow_pickle=False) as saved:
            for method in config["methods"]:
                observed = result[method]
                np.testing.assert_allclose(observed, saved[method],
                                           rtol=config["execution"]["replay_rtol"],
                                           atol=config["execution"]["replay_atol"])
                records.append({"task_id": task["task_id"], "method": method,
                                "max_abs_difference": float(np.max(np.abs(observed - saved[method]))),
                                "exact_equal": bool(np.array_equal(observed, saved[method]))})
    if not records:
        raise ValueError("No replay tasks selected")
    filename = "replay.json" if task_id is None else f"replay_{task_id}.json"
    record = {"completed_utc": utcnow(), "environment": environment(),
              "status": "passed", "records": records,
              "rtol": config["execution"]["replay_rtol"], "atol": config["execution"]["replay_atol"],
              "scope": "Selected-state inference replay; no refitting or reselection"}
    write_json(study_dir / "results" / filename, record)
    print(json.dumps({"replay": "passed", "method_task_records": len(records),
                      "max_abs_difference": max(r["max_abs_difference"] for r in records)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "verify", "worker", "run", "replay"):
        child = sub.add_parser(command)
        child.add_argument("--study-dir", type=Path, default=ROOT)
        if command == "worker":
            child.add_argument("--task-id", required=True)
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--development", action="store_true")
        elif command == "run":
            child.add_argument("--freeze-commit", required=True)
            child.add_argument("--resume", action="store_true")
        elif command == "replay":
            child.add_argument("--task-id")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.study_dir)
    elif args.command == "verify":
        record = verify_freeze(args.study_dir)
        print(json.dumps({"verified_files": len(record["files"])}))
    elif args.command == "worker":
        return worker(args.study_dir, args.task_id, args.output, development=args.development)
    elif args.command == "run":
        execute(args.study_dir, args.freeze_commit, resume=args.resume)
    elif args.command == "replay":
        replay(args.study_dir, task_id=args.task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
