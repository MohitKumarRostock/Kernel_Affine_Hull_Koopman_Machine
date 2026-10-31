#!/usr/bin/env python3
"""Prepare, freeze, and execute the forecast-trained reconstruction experiment.

Preparation creates the complete inputs and roster without fitting a predictor.
Execution verifies the published freeze before exposing reserved D outcomes to
the analysis.  The fit API receives D histories, never D targets.  The separate
``smoke`` command uses an explicitly excluded development seed and no natural
task.  See PROTOCOL.md for the estimands, selection rules, and claim boundaries.
"""
from __future__ import annotations

import os

# These assignments must precede all numerical imports.
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import resource
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent.parent
GENERATOR_PATH = (
    REPOSITORY / "extensions/predictive_geometry/study/datasets.py"
)
SPLITS = ("A", "B", "C", "D")
LAWS = ("nonlinear_ar", "vanderpol", "duffing")
STUDY_SEEDS = tuple(range(84200, 84212))
DEVELOPMENT_SEED = 932001
HISTORY = 12
OFFSETS = (1, 4, 12)
EPISODE_LENGTH = 72
ORIGIN_STRIDE = 4
SPLIT_EPISODES = {"A": 64, "B": 32, "C": 32, "D": 128}
SPLIT_CAPS = {"A": 512, "B": 256, "C": 256, "D": None}
NATURAL_CAPS = {"A": 512, "B": 128, "C": 128, "D": 128}
GENERATOR_SALT = 842601
CAP_SALT = 842602
FIT_SALT = 842603
PRIMARY_CONTROLS = (
    "raw_rbf", "gate_kahm", "reconstruction_affine",
    "reconstruction_locality", "reconstruction_rbf_local", "nonkahm_policy",
)
PRIMARY_CANDIDATE = "reconstruction_kahm"
SCHEMA_VERSION = "direct-reconstruction-v1"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    """Canonical array hash independent of a ZIP container's timestamps."""
    value = np.ascontiguousarray(value)
    prefix = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True).encode("utf-8")
    return hashlib.sha256(prefix + b"\0" + value.tobytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model():
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    return load_module(ROOT / "model.py", "kahm_direct_reconstruction_model")


def generator():
    return load_module(GENERATOR_PATH, "kahm_preserved_episode_generator")


def seed_from(parts):
    return int(np.random.SeedSequence([int(x) for x in parts]).generate_state(1)[0])


def make_windows(episodes, history=HISTORY, offsets=OFFSETS, stride=ORIGIN_STRIDE):
    """Independently index episode-major causal histories and future targets."""
    episodes = np.asarray(episodes, dtype=np.float64)
    if episodes.ndim == 3 and episodes.shape[-1] == 1:
        episodes = episodes[:, :, 0]
    if episodes.ndim != 2 or not np.isfinite(episodes).all():
        raise ValueError("Expected finite scalar episodes")
    offsets = np.asarray(offsets, dtype=np.int64)
    origins = np.arange(history - 1, episodes.shape[1] - int(offsets.max()),
                        stride, dtype=np.int64)
    history_index = origins[:, None] - np.arange(history - 1, -1, -1)[None, :]
    target_index = origins[:, None] + offsets[None, :]
    n = len(episodes)
    count = n * len(origins)
    return {
        "X": np.ascontiguousarray(episodes[:, history_index].reshape(count, history)),
        "Y": np.ascontiguousarray(episodes[:, target_index].reshape(count, len(offsets))),
        "groups": np.repeat(np.arange(n, dtype=np.int64), len(origins)),
        "origins": np.tile(origins, n),
        "row_indices": np.arange(count, dtype=np.int64),
    }


def make_synthetic_inputs(law, seed, *, smoke=False):
    upstream = generator()
    law_index = LAWS.index(law)
    counts = ({"A": 24, "B": 12, "C": 12, "D": 12}
              if smoke else SPLIT_EPISODES)
    caps = ({"A": 192, "B": 96, "C": 96, "D": None}
            if smoke else SPLIT_CAPS)
    arrays = {"offsets": np.asarray(OFFSETS, dtype=np.int64)}
    records = {}
    for split_index, split in enumerate(SPLITS):
        components = [seed, law_index, split_index, GENERATOR_SALT]
        generation_seed = seed_from(components)
        episodes = upstream.generate_episodes(
            law, counts[split], EPISODE_LENGTH, generation_seed
        )[:, :, 0]
        windows = make_windows(episodes)
        full_count = len(windows["X"])
        cap_components = [seed, law_index, split_index, CAP_SALT]
        cap_seed = seed_from(cap_components)
        cap = caps[split]
        if cap is not None and full_count > cap:
            chosen = np.sort(np.random.default_rng(cap_seed).choice(
                full_count, size=cap, replace=False
            ))
            windows = {key: value[chosen] for key, value in windows.items()}
        arrays.update({f"{split}_{key}": value for key, value in windows.items()})
        arrays[f"{split}_episodes"] = episodes
        records[split] = {
            "generator_seed_sequence": components, "generator_seed": generation_seed,
            "cap_seed_sequence": cap_components, "cap_seed": cap_seed,
            "episode_count": counts[split], "uncapped_rows": full_count,
            "retained_rows": len(windows["X"]),
            "array_sha256": {key: array_hash(value) for key, value in windows.items()},
            "episodes_sha256": array_hash(episodes),
        }
    task_id = f"synthetic__{law}__{seed}"
    record = {
        "task_id": task_id, "suite": "synthetic", "law": law, "seed": seed,
        "fit_seed": seed_from([seed, law_index, FIT_SALT]),
        "L": HISTORY, "p": 1, "Hmax": max(OFFSETS), "offsets": list(OFFSETS),
        "loss_scale": 1.0, "split_records": records,
        "generator_source": str(GENERATOR_PATH.relative_to(REPOSITORY)),
        "generator_sha256": sha256_file(GENERATOR_PATH),
        "independence_unit": "repetition (with independent episodes within split)",
        "development_only": bool(smoke),
    }
    return arrays, record


def validate_arrays(arrays, task):
    """Structural/pre-outcome validation only; this does not score predictions."""
    offsets = np.asarray(task["offsets"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) != 3 or np.any(offsets <= 0):
        raise ValueError(f"{task['task_id']}: exactly three positive offsets required")
    if int(task["p"]) > int(task["L"]):
        raise ValueError("Seasonal-naive period must be available in each history")
    if not np.array_equal(arrays["offsets"], offsets):
        raise ValueError("Task offsets and prepared input disagree")
    for split in SPLITS:
        x, y = arrays[f"{split}_X"], arrays[f"{split}_Y"]
        if (x.ndim != 2 or y.shape != (len(x), len(offsets))
                or x.shape[1] != task["L"]
                or (len(x) == 0 and (task["suite"] != "natural" or split == "A"))):
            raise ValueError(f"Invalid {split} matrix shape for {task['task_id']}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite retained input for {task['task_id']}")
        for name in ("groups", "origins", "row_indices"):
            if arrays[f"{split}_{name}"].shape != (len(x),):
                raise ValueError(f"Invalid {split}_{name}")
    if not np.isfinite(task["loss_scale"]) or task["loss_scale"] <= 0:
        raise ValueError("A positive, finite loss denominator is required")
    if task["suite"] == "natural":
        train = np.asarray(arrays["A_inner_train_mask"], dtype=bool)
        validation = np.asarray(arrays["A_inner_validation_mask"], dtype=bool)
        if train.shape != (len(arrays["A_X"]),) or validation.shape != train.shape:
            raise ValueError("Natural inner-A masks do not match retained rows")
        if not train.any() or not validation.any() or (train & validation).any():
            raise ValueError("Natural inner-A masks must be nonempty and disjoint")
        origins = arrays["A_origins"]
        if np.max(origins[train] + max(offsets)) > np.min(origins[validation]):
            raise ValueError("Inner-A fitting labels extend past validation availability")


def scientific_paths():
    paths = [ROOT / name for name in (
        "run.py", "model.py", "analyze.py", "prepare_sources.py",
        "test_model.py", "test_protocol.py", "PROTOCOL.md", "PRE_FREEZE_DESIGN_LOG.md", "requirements.txt",
    )]
    paths.extend([
        GENERATOR_PATH,
        REPOSITORY / "extensions/target_geometry/models.py",
        REPOSITORY / "extensions/natural_forecasting/models.py",
        GENERATOR_PATH.parent / "geometry.py", GENERATOR_PATH.parent / "heads.py",
    ])
    # The inherited geometry loads small pinned modules dynamically. Freeze its
    # entire vendored Python tree, including the source of the numerical laws.
    paths.extend(sorted((GENERATOR_PATH.parent / "vendor/upstream").rglob("*.py")))
    for ancestor in (REPOSITORY / "extensions", REPOSITORY / "extensions/target_geometry",
                     REPOSITORY / "extensions/natural_forecasting",
                     REPOSITORY / "extensions/predictive_geometry", GENERATOR_PATH.parent):
        if (ancestor / "__init__.py").is_file():
            paths.append(ancestor / "__init__.py")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Freeze requires all scientific files: {missing}")
    return paths


def environment():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "packages": packages,
        "threads": {name: os.environ[name] for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
        )},
    }


def prepare(output, *, natural_manifest=None):
    output = Path(output).resolve()
    if (output / "freeze.json").exists():
        raise FileExistsError("A freeze already exists; choose a new directory")
    if (output / "tasks").exists():
        raise FileExistsError("Refusing to prepare over evaluated task artifacts")
    output.mkdir(parents=True, exist_ok=True)
    input_directory = output / "prepared_inputs"
    input_directory.mkdir(exist_ok=True)
    model = load_model()
    manifest_path = Path(natural_manifest or ROOT / "data/sources_manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["tasks"]) != 10:
        raise ValueError("The frozen natural roster must contain exactly ten tasks")
    if len({task["source_key"] for task in manifest["tasks"]}) != 4:
        raise ValueError("The frozen natural roster must contain four collections")
    tasks = []
    for law in LAWS:
        for seed in STUDY_SEEDS:
            arrays, record = make_synthetic_inputs(law, seed)
            validate_arrays(arrays, record)
            path = input_directory / f"{record['task_id']}.npz"
            np.savez_compressed(path, **arrays)
            record["prepared_path"] = str(path.relative_to(output))
            record["prepared_sha256"] = sha256_file(path)
            record["array_sha256"] = {key: array_hash(value) for key, value in arrays.items()}
            tasks.append(record)
    for index, original in enumerate(manifest["tasks"]):
        record = dict(original)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", record["task_id"]):
            raise ValueError("Unsafe task identifier")
        record.update({
            "suite": "natural", "fit_seed": seed_from([842000, index, FIT_SALT]),
            "development_only": False,
        })
        source = ROOT / record["npz_path"]
        with np.load(source, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        validate_arrays(arrays, record)
        for split in SPLITS:
            if len(arrays[f"{split}_X"]) > NATURAL_CAPS[split]:
                raise ValueError("Natural cap differs from the declared protocol")
        target = input_directory / f"{record['task_id']}.npz"
        shutil.copyfile(source, target)
        record["prepared_path"] = str(target.relative_to(output))
        record["prepared_sha256"] = sha256_file(target)
        record["array_sha256"] = {key: array_hash(value) for key, value in arrays.items()}
        tasks.append(record)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task identifier")
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective code/configuration/input freeze; D not evaluated",
        "prepared_utc": utcnow(), "methods": list(model.METHODS),
        "model_config": model.DEFAULT_CONFIG, "laws": list(LAWS),
        "seeds": list(STUDY_SEEDS), "excluded_development_seed": DEVELOPMENT_SEED,
        "split_episodes": SPLIT_EPISODES, "split_caps": SPLIT_CAPS,
        "natural_caps": NATURAL_CAPS, "history": HISTORY,
        "offsets": list(OFFSETS), "origin_stride": ORIGIN_STRIDE,
        "generator_seed_salt": GENERATOR_SALT, "cap_seed_salt": CAP_SALT,
        "fit_seed_salt": FIT_SALT,
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_controls": list(PRIMARY_CONTROLS),
        "primary_family_size": 18, "family_alpha": 0.05,
        "primary_interval": "paired repetition-mean t interval, df=11, Bonferroni18",
        "primary_sign": "control loss minus reconstruction_kahm loss; positive favors KAHM",
        "natural_aggregation": "raw MSE / A-prefix loss_scale; equal series within equal collection",
        "natural_inference": "descriptive only; no population or window-based confidence interval",
        "tasks": tasks,
    }
    write_json(output / "protocol.json", protocol)
    snapshot = output / "source_snapshot"
    hashes = {}
    for source in scientific_paths() + [manifest_path]:
        relative = source.relative_to(REPOSITORY)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = sha256_file(source)
    freeze = {
        "schema_version": SCHEMA_VERSION, "created_utc": utcnow(),
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "scientific_files": hashes,
        "prepared_inputs": {task["prepared_path"]: task["prepared_sha256"] for task in tasks},
        "source_snapshot": "source_snapshot", "task_count": len(tasks),
        "method_count": len(model.METHODS), "environment": environment(),
        "evaluation_status_at_freeze": "no declared synthetic or natural D model outcomes evaluated",
        "public_commit": "supplied at execution; commit must contain this freeze",
    }
    write_json(output / "freeze.json", freeze)
    print(json.dumps({"prepared": str(output), "tasks": len(tasks),
                      "freeze_sha256": sha256_file(output / "freeze.json")}), flush=True)
    return freeze


def verify_freeze(output, *, live_source=True):
    output = Path(output).resolve()
    freeze = json.loads((output / "freeze.json").read_text())
    if freeze.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown freeze schema")
    if sha256_file(output / "protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("Protocol changed after freeze")
    for relative, expected in freeze["scientific_files"].items():
        snapshot = output / freeze["source_snapshot"] / relative
        if sha256_file(snapshot) != expected:
            raise ValueError(f"Frozen snapshot changed: {relative}")
        if live_source and sha256_file(REPOSITORY / relative) != expected:
            raise ValueError(f"Scientific source changed after freeze: {relative}")
    for relative, expected in freeze["prepared_inputs"].items():
        if sha256_file(output / relative) != expected:
            raise ValueError(f"Prepared inputs changed after freeze: {relative}")
    return freeze


def _execute_task(task, arrays, destination, model, config):
    validate_arrays(arrays, task)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite task artifacts: {destination}")
    destination.mkdir(parents=True)
    write_json(destination / "task.json", task)
    if any(len(arrays[f"{split}_X"]) == 0 for split in SPLITS):
        raise ValueError("Frozen A-eligible task has an empty later split; retained as an unavailable outcome")
    kwargs = {"horizons": task["offsets"], "naive_period": task["p"]}
    if task["suite"] == "natural":
        kwargs.update({key: arrays[key] for key in (
            "A_inner_train_mask", "A_inner_validation_mask"
        )})
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    result = model.fit_suite(
        arrays["A_X"], arrays["A_Y"], arrays["A_groups"],
        arrays["B_X"], arrays["B_Y"], arrays["C_X"], arrays["C_Y"],
        arrays["D_X"], seed=task["fit_seed"], config=config, **kwargs,
    )
    timing = {
        "suite_fit_wall_seconds": time.perf_counter() - wall_started,
        "suite_fit_cpu_seconds": time.process_time() - cpu_started,
        "process_cumulative_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "cumulative process high-water mark, including all preceding tasks and all arms",
        "fit_scope": "fit_suite; excludes input generation/imports/artifact writing/replay checks",
    }
    predictions = {"C_targets": arrays["C_Y"], "D_targets": arrays["D_Y"]}
    for split, result_key in (("C", "predictions_c"), ("D", "predictions_d")):
        if set(result[result_key]) != set(model.METHODS):
            raise ValueError("The model returned an incomplete or changed method roster")
        for method, prediction in result[result_key].items():
            prediction = np.asarray(prediction, dtype=np.float64)
            if prediction.shape != arrays[f"{split}_Y"].shape or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid {split} prediction: {method}")
            predictions[f"{split}__{method}"] = prediction
    models_metadata = {}
    if set(result["serialized_models"]) != set(model.METHODS):
        raise ValueError("Every selected procedure must be serializable")
    archive_path = destination / "selected_models.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for method, contents in result["serialized_models"].items():
            if not isinstance(contents, bytes):
                raise TypeError("Serialized model contents must be bytes")
            member = f"{method}.pkl"
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, contents, compresslevel=9)
            models_metadata[method] = {
                "archive": archive_path.name, "member": member,
                "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest(),
            }
    np.savez_compressed(destination / "predictions.npz", **predictions)
    diagnostics = result["diagnostics"]
    diagnostics["runner_timing"] = timing
    diagnostics["serialized_models"] = models_metadata
    write_json(destination / "diagnostics.json", diagnostics)
    write_json(destination / "complete.json", {
        "completed_utc": utcnow(), "task_id": task["task_id"],
        "predictions_sha256": sha256_file(destination / "predictions.npz"),
        "diagnostics_sha256": sha256_file(destination / "diagnostics.json"),
        "task_sha256": sha256_file(destination / "task.json"),
        "selected_models_sha256": sha256_file(archive_path),
    })
    return timing


def execute(output, public_freeze_commit, *, resume=False):
    output = Path(output).resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", public_freeze_commit):
        raise ValueError("A full public Git commit SHA containing the freeze is required")
    freeze = verify_freeze(output)
    protocol = json.loads((output / "protocol.json").read_text())
    model = load_model()
    if list(model.METHODS) != protocol["methods"] or jsonable(model.DEFAULT_CONFIG) != protocol["model_config"]:
        raise ValueError("Model roster/configuration differs from prospective freeze")
    execution_path = output / "execution.json"
    if execution_path.exists():
        if not resume:
            raise FileExistsError("Execution already started; use --resume for missing tasks")
        execution = json.loads(execution_path.read_text())
        if execution["public_freeze_commit"].lower() != public_freeze_commit.lower():
            raise ValueError("A resumed run cannot change its public freeze commit")
    else:
        execution = {
            "started_utc": utcnow(), "public_freeze_commit": public_freeze_commit.lower(),
            "freeze_sha256": sha256_file(output / "freeze.json"),
            "protocol_sha256": freeze["protocol_sha256"],
            "environment": environment(), "status": "running", "failures": [],
        }
        write_json(execution_path, execution)
    for index, task in enumerate(protocol["tasks"]):
        destination = output / "tasks" / task["task_id"]
        complete_path = destination / "complete.json"
        if complete_path.exists() and resume:
            complete = json.loads(complete_path.read_text())
            for filename, key in (("predictions.npz", "predictions_sha256"),
                                  ("diagnostics.json", "diagnostics_sha256"),
                                  ("task.json", "task_sha256"),
                                  ("selected_models.zip", "selected_models_sha256")):
                if sha256_file(destination / filename) != complete[key]:
                    raise ValueError(f"Completed task changed: {task['task_id']}/{filename}")
            continue
        if destination.exists():
            if resume and (destination / "failure.json").exists():
                # A documented failed task is part of the roster. It is never
                # silently retried or replaced after other D results are seen.
                continue
            raise FileExistsError(
                f"Incomplete task {task['task_id']} is preserved; move it to an explicit failure archive before resuming"
            )
        with np.load(output / task["prepared_path"], allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        print(json.dumps({"task": task["task_id"], "index": index + 1,
                          "total": len(protocol["tasks"]), "status": "started"}), flush=True)
        try:
            timing = _execute_task(task, arrays, destination, model, protocol["model_config"])
        except Exception as error:
            failure = {"task_id": task["task_id"], "utc": utcnow(),
                       "exception": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()}
            execution["status"] = "running_with_failures"
            execution["failures"].append(failure)
            write_json(execution_path, execution)
            write_json(destination / "failure.json", failure)
            print(json.dumps({"task": task["task_id"], "status": "failed",
                              "exception": type(error).__name__}), flush=True)
            continue
        print(json.dumps({"task": task["task_id"], "status": "complete",
                          "wall_seconds": timing["suite_fit_wall_seconds"]}), flush=True)
    execution.update({"status": "complete_with_failures" if execution["failures"] else "complete",
                      "completed_utc": utcnow(),
                      "completed_tasks": sum((output / "tasks" / task["task_id"] / "complete.json").exists()
                                             for task in protocol["tasks"])})
    write_json(execution_path, execution)


def smoke(output, law="nonlinear_ar"):
    """Development-only mechanical run; never part of the confirmatory roster."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh development-smoke destination")
    output.mkdir(parents=True)
    arrays, task = make_synthetic_inputs(law, DEVELOPMENT_SEED, smoke=True)
    np.savez_compressed(output / "inputs.npz", **arrays)
    model = load_model()
    _execute_task(task, arrays, output / "task", model, model.DEFAULT_CONFIG)
    write_json(output / "development_status.json", {
        "status": "excluded development smoke only", "seed": DEVELOPMENT_SEED,
        "law": law, "method_count": len(model.METHODS), "completed_utc": utcnow(),
    })
    print(json.dumps({"smoke_complete": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preparation = subparsers.add_parser("prepare", help="create inputs and freeze without fitting")
    preparation.add_argument("--output", type=Path, default=ROOT / "results")
    preparation.add_argument("--natural-manifest", type=Path)
    execution = subparsers.add_parser("run", help="verify public freeze and execute all frozen tasks")
    execution.add_argument("--output", type=Path, default=ROOT / "results")
    execution.add_argument("--public-freeze-commit", required=True)
    execution.add_argument("--resume", action="store_true")
    development = subparsers.add_parser("smoke", help="excluded seed932001 mechanical test only")
    development.add_argument("--output", type=Path, required=True)
    development.add_argument("--law", choices=LAWS, default="nonlinear_ar")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.output, natural_manifest=arguments.natural_manifest)
    elif arguments.command == "run":
        execute(arguments.output, arguments.public_freeze_commit, resume=arguments.resume)
    else:
        smoke(arguments.output, arguments.law)


if __name__ == "__main__":
    main()
