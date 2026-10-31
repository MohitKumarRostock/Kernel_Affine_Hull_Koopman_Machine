#!/usr/bin/env python3
"""Exploratory, known-law KAHM forecast-objective pilot.

This is a new implementation of the numerical laws described in the supplied
manuscript, not an execution of its archived source or a confirmatory study.
Before fitting, the runner snapshots scientific source and writes a fixed
protocol, complete task roster, configuration, and hashes. Test outcomes never
alter that protocol. See README and PROTOCOL for scope and limitations.
"""
from __future__ import annotations

# Set these before importing any numerical package (including geometry).
import os
for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
import ast
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LAWS = ("nonlinear_ar", "van_der_pol", "duffing")
STUDY_SEEDS = (73101, 73102, 73103)
QUICK_SEEDS = (931071,)
SPLIT_NAMES = ("A", "B", "C", "D")
HISTORY = 12
OFFSETS = np.array([1, 4, 12], dtype=np.int64)
ORIGINS = np.arange(11, 60, 4, dtype=np.int64)
EPISODE_LENGTH = 72
GENERATOR_SALT = 613147
CAP_SALT = 613148
FIT_SALT = 613149
METHODS = (
    "raw_ridge", "raw_rbf", "fixed_kahm", "direct_kahm",
    "fixed_affine", "direct_affine", "fixed_locality", "direct_locality",
    "fixed_rbf_local", "direct_rbf_local", "persistence",
)
MODEL_CONFIG = {
    "alpha_grid": [0.01, 0.1, 1.0],
    "gamma_grid": [0.0, 0.25, 0.5, 1.0],
    "max_objective_evaluations": 160,
    "temperature_bound": 4.0,
    "blocks": [2, 4, 8],
    "max_references": 24,
    "rank": 2,
    "raw_rbf_landmarks": 64,
}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    """Preserve JSON-safe diagnostics without silently emitting NaN tokens."""
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


def write_json(path, obj):
    path = Path(path)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    """Hash dtype, shape, and canonical C-order bytes, not ZIP metadata."""
    value = np.ascontiguousarray(value)
    prefix = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True).encode("utf-8")
    return hashlib.sha256(prefix + b"\0" + value.tobytes()).hexdigest()


def generate_ar(n_episodes, rng):
    """Discard the first 80 stored states, including the two initial states."""
    episodes = np.empty((n_episodes, EPISODE_LENGTH), dtype=np.float64)
    for episode in range(n_episodes):
        stored = np.empty(80 + EPISODE_LENGTH, dtype=np.float64)
        stored[:2] = rng.uniform(-0.8, 0.8, size=2)
        innovations = rng.uniform(-0.1, 0.1, size=stored.size - 2)
        for index in range(2, stored.size):
            current, previous = stored[index - 1], stored[index - 2]
            stored[index] = np.tanh(
                1.25 * current - 0.65 * previous
                + 0.25 * np.sin(3.0 * current) + innovations[index - 2]
            )
        episodes[episode] = stored[80:]
    return episodes


def ode_rhs(state, time_value, phase, law):
    x, velocity = state
    if law == "van_der_pol":
        acceleration = (1.0 - x * x) * velocity - x
    elif law == "duffing":
        acceleration = (-0.2 * velocity + x - x ** 3
                        + 0.3 * np.cos(1.2 * time_value + phase))
    else:
        raise ValueError(f"Unknown ODE law: {law}")
    return np.array([velocity, acceleration], dtype=np.float64)


def rk4_step(state, time_value, step, phase, law):
    k1 = ode_rhs(state, time_value, phase, law)
    k2 = ode_rhs(state + 0.5 * step * k1, time_value + 0.5 * step,
                 phase, law)
    k3 = ode_rhs(state + 0.5 * step * k2, time_value + 0.5 * step,
                 phase, law)
    k4 = ode_rhs(state + step * k3, time_value + step, phase, law)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_ode(n_episodes, rng, law):
    episodes = np.empty((n_episodes, EPISODE_LENGTH), dtype=np.float64)
    step = 0.025
    first_observation_step = 400  # t=10.0, following the 10-unit burn-in.
    observation_stride = 4       # 0.1 / 0.025.
    final_step = first_observation_step + observation_stride * (EPISODE_LENGTH - 1)
    for episode in range(n_episodes):
        state = rng.uniform(-2.0, 2.0, size=2).astype(np.float64)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        # Retain the phase draw for Van der Pol and draw noise before integration.
        noise = rng.uniform(-0.02, 0.02, size=EPISODE_LENGTH)
        for integration_step in range(final_step + 1):
            if (integration_step >= first_observation_step
                    and (integration_step - first_observation_step) % observation_stride == 0):
                obs = (integration_step - first_observation_step) // observation_stride
                episodes[episode, obs] = 0.95 * np.tanh(state[0]) + noise[obs]
            if integration_step < final_step:
                state = rk4_step(state, integration_step * step, step, phase, law)
    return episodes


def make_windows(episodes):
    x = np.concatenate([episodes[:, t - HISTORY + 1:t + 1]
                        for t in ORIGINS], axis=0)
    y = np.concatenate([episodes[:, t + OFFSETS] for t in ORIGINS], axis=0)
    # Reorder origin-major construction into stable episode-major rows.
    n = len(episodes)
    permutation = np.arange(n * len(ORIGINS)).reshape(len(ORIGINS), n).T.ravel()
    return {
        "X": np.ascontiguousarray(x[permutation], dtype=np.float64),
        "Y": np.ascontiguousarray(y[permutation], dtype=np.float64),
        "groups": np.repeat(np.arange(n, dtype=np.int64), len(ORIGINS)),
        "origins": np.tile(ORIGINS, n),
        "row_indices": np.arange(n * len(ORIGINS), dtype=np.int64),
    }


def make_task_inputs(law, seed, split_episodes, split_caps):
    law_index = LAWS.index(law)
    arrays = {}
    split_records = {}
    for split_index, split_name in enumerate(SPLIT_NAMES):
        sequence = [int(seed), law_index, split_index, GENERATOR_SALT]
        rng = np.random.default_rng(np.random.SeedSequence(sequence))
        count = int(split_episodes[split_name])
        episodes = (generate_ar(count, rng) if law == "nonlinear_ar"
                    else generate_ode(count, rng, law))
        windows = make_windows(episodes)
        full_count = len(windows["X"])
        cap = split_caps.get(split_name)
        cap_sequence = [int(seed), law_index, split_index, CAP_SALT]
        if cap is not None and full_count > cap:
            cap_rng = np.random.default_rng(np.random.SeedSequence(cap_sequence))
            chosen = np.sort(cap_rng.choice(full_count, size=int(cap), replace=False))
            windows = {key: value[chosen] for key, value in windows.items()}
        for key, value in windows.items():
            arrays[f"{split_name}_{key}"] = value
        # Preserve uncapped observations, allowing complete generator/window audits.
        arrays[f"{split_name}_episodes"] = episodes
        split_records[split_name] = {
            "generator_seed_sequence": sequence,
            "cap_seed_sequence": cap_sequence,
            "episodes": count,
            "uncapped_rows": full_count,
            "retained_rows": len(windows["X"]),
            "array_sha256": {key: array_hash(value) for key, value in windows.items()},
            "episodes_sha256": array_hash(episodes),
        }
    return arrays, split_records


def load_geometry(path):
    spec = importlib.util.spec_from_file_location("kahm_frozen_geometry", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_protocol(output, mode):
    """Snapshot the full numerical implementation before evaluating any D target."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    names = sorted({path.name for pattern in ("*.py", "*.md")
                    for path in source.glob(pattern) if path.is_file()})
    for required in ("geometry.py", "run_pilot.py", "audit_pilot.py"):
        if required not in names:
            raise FileNotFoundError(source / required)
    for name in ("attachment_integrity.json", "check_geometry.json", "check_online.json", "requirements.txt"):
        if (source / name).is_file():
            names.append(name)
    hashes = {}
    for name in names:
        original = source / name
        if not original.is_file():
            raise FileNotFoundError(original)
        shutil.copy2(original, snapshot / name)
        hashes[name] = sha256_file(snapshot / name)
    quick = mode == "quick"
    episodes = dict(zip(SPLIT_NAMES, (12, 8, 8, 12) if quick else (64, 32, 32, 64)))
    caps = dict(zip(SPLIT_NAMES, (96, 64, 64, None) if quick else (512, 256, 256, None)))
    seeds = QUICK_SEEDS if quick else STUDY_SEEDS
    geometry_ast = ast.parse((snapshot / "geometry.py").read_text(encoding="utf-8"))
    declared_defaults = next(ast.literal_eval(node.value) for node in geometry_ast.body
                             if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "DEFAULT_CONFIG"
                                     for target in node.targets))
    config = {**declared_defaults, **MODEL_CONFIG}
    if quick:
        config.update({"max_objective_evaluations": 24, "blocks": [2, 4],
                       "max_references": 12, "raw_rbf_landmarks": 16})
    protocol = {
        "schema_version": 1,
        "created_at_utc": utcnow(),
        "study_status": "implementation_check" if quick else "exploratory_known_law_pilot",
        "not_confirmatory": True,
        "mode": mode,
        "source_sha256": hashes,
        "methods": list(METHODS),
        "laws": list(LAWS),
        "seeds": list(seeds),
        "tasks": [{"task_id": f"{law}_{seed}", "law": law, "seed": seed}
                  for law in LAWS for seed in seeds],
        "split_episodes": episodes,
        "split_row_caps": caps,
        "split_roles": {
            "A": "bank/reference construction, baseline/correction fitting",
            "B": "direct forecast-geometry development objective; see geometry diagnostics",
            "C": "selection of declared final settings, including zero correction",
            "D": "untuned exploratory evaluation, never selects settings",
        },
        "model_config": config,
        "generator": {
            "source": "Supplied manuscript appendix_controlled.tex: Exact forecasting laws and target construction",
            "new_implementation": True,
            "dtype": "float64",
            "generator_salt": GENERATOR_SALT,
            "cap_salt": CAP_SALT,
            "fit_salt": FIT_SALT,
            "episode_length": EPISODE_LENGTH,
            "history": HISTORY,
            "offsets": OFFSETS.tolist(),
            "origins_zero_based": ORIGINS.tolist(),
            "ar_initial": "two independent Uniform[-.8,.8]",
            "ar_law": "u[t+1]=tanh(1.25*u[t]-.65*u[t-1]+.25*sin(3*u[t])+eta[t+1])",
            "ar_innovation": "iid Uniform[-.1,.1]",
            "ar_burn_stored_states": 80,
            "van_der_pol": "dx=v; dv=(1-x*x)*v-x",
            "duffing": "dx=v; dv=-.2*v+x-x**3+.3*cos(1.2*t+theta)",
            "ode_initial": "x,v iid Uniform[-2,2]; theta Uniform[0,2*pi] retained for both laws",
            "ode_rk4_step": 0.025,
            "ode_burn_time": 10.0,
            "ode_first_observation_time": 10.0,
            "ode_observation_interval": 0.1,
            "ode_sensor": ".95*tanh(x)+iid Uniform[-.02,.02]",
            "ode_noise_drawn_before_integration": True,
        },
        "analysis": {
            "primary_metric": "equal-episode, equal-origin, equal-horizon MSE on D",
            "horizon_metrics": "MSE separately at offsets 1,4,12",
            "aggregation": "arithmetic mean of task MSE over seeds within each law",
            "uncertainty": "none: three exploratory repetitions do not support a confirmatory claim",
            "failures": "retain task/method failures; missing methods invalidate complete-roster aggregates",
            "timing": "single numerical-library thread; reported complete charged component sums and observed task wall time",
            "selection_audit_scope": "replay recorded C candidate losses and selected predictions; no independent refit of every candidate",
        },
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__,
                    "scipy": importlib.metadata.version("scipy"),
                    "scikit_learn": importlib.metadata.version("scikit-learn"),
                    "thread_environment": {key: os.environ[key] for key in (
                        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}},
    }
    write_json(output / "protocol.json", protocol)
    # Separate freeze record includes immutable protocol bytes and is present
    # before importing the snapshotted forecasting implementation.
    write_json(output / "freeze.json", {
        "created_at_utc": utcnow(),
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "source_sha256": hashes,
        "observed_D_outcomes_before_freeze": False,
    })
    return protocol


def evaluate_prediction(target, prediction, groups):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != target {target.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite prediction")
    errors = (target - prediction) ** 2
    episode_errors = [errors[groups == group].mean(axis=0)
                      for group in np.unique(groups)]
    horizon_mse = np.mean(episode_errors, axis=0)
    return {
        "mse": float(np.mean(horizon_mse)),
        "horizon_mse": {str(int(h)): float(v) for h, v in zip(OFFSETS, horizon_mse)},
        "episode_mse": [float(np.mean(values)) for values in episode_errors],
    }


def run_task(task, protocol, geometry, output):
    task_dir = Path(output) / "tasks" / task["task_id"]
    task_dir.mkdir(parents=True)
    preparation_start = time.perf_counter()
    arrays, split_records = make_task_inputs(
        task["law"], task["seed"], protocol["split_episodes"],
        protocol["split_row_caps"],
    )
    np.savez_compressed(task_dir / "inputs.npz", **arrays)
    fit_seed_sequence = [task["seed"], LAWS.index(task["law"]), FIT_SALT]
    fit_seed = int(np.random.SeedSequence(fit_seed_sequence).generate_state(1)[0])
    record = {
        **task,
        "started_at_utc": utcnow(),
        "fit_seed_sequence": fit_seed_sequence,
        "fit_seed": fit_seed,
        "split_records": split_records,
        "preparation_seconds": time.perf_counter() - preparation_start,
        "inputs_file_sha256": sha256_file(task_dir / "inputs.npz"),
        "status": "running",
    }
    write_json(task_dir / "task.json", record)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    try:
        fitted = geometry.fit_suite(
            arrays["A_X"], arrays["A_Y"], arrays["A_groups"],
            arrays["B_X"], arrays["B_Y"], arrays["C_X"], arrays["C_Y"],
            arrays["D_X"], seed=fit_seed, config=protocol["model_config"],
        )
        record["suite_wall_seconds"] = time.perf_counter() - start_wall
        record["suite_cpu_seconds"] = time.process_time() - start_cpu
        diagnostics = fitted.get("diagnostics", {})
        predictions_d = dict(fitted["predictions_d"])
        predictions_c = dict(fitted["predictions_c"])
        record["status"] = "completed"
    except Exception as exc:
        record["suite_wall_seconds"] = time.perf_counter() - start_wall
        record["suite_cpu_seconds"] = time.process_time() - start_cpu
        record["status"] = "failed"
        record["exception"] = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()}
        diagnostics, predictions_d, predictions_c = {}, {}, {}
    persistence_start = time.perf_counter()
    predictions_d["persistence"] = np.repeat(arrays["D_X"][:, -1:], len(OFFSETS), axis=1)
    predictions_c["persistence"] = np.repeat(arrays["C_X"][:, -1:], len(OFFSETS), axis=1)
    record["persistence_seconds"] = time.perf_counter() - persistence_start
    saved = {f"D__{key}": np.asarray(value, dtype=np.float64)
             for key, value in predictions_d.items()}
    saved.update({f"C__{key}": np.asarray(value, dtype=np.float64)
                  for key, value in predictions_c.items()})
    np.savez_compressed(task_dir / "predictions.npz", **saved)
    write_json(task_dir / "diagnostics.json", diagnostics)
    record["prediction_file_sha256"] = sha256_file(task_dir / "predictions.npz")
    record["diagnostics_file_sha256"] = sha256_file(task_dir / "diagnostics.json")
    record["prediction_array_sha256"] = {key: array_hash(value) for key, value in saved.items()}
    record["missing_methods"] = [method for method in METHODS
                                 if method not in predictions_d or method not in predictions_c]
    record["unexpected_methods"] = sorted(set(predictions_d) - set(METHODS))
    if record["missing_methods"]:
        record["status"] = "incomplete"
    record["metrics"] = {}
    record["invalid_predictions"] = {}
    for method, prediction in predictions_d.items():
        try:
            metrics = evaluate_prediction(arrays["D_Y"], prediction, arrays["D_groups"])
            metrics["C_mse"] = float(np.mean((arrays["C_Y"] - predictions_c[method]) ** 2))
            record["metrics"][method] = metrics
        except Exception as exc:
            record["invalid_predictions"][method] = str(exc)
            record["status"] = "incomplete"
    record["finished_at_utc"] = utcnow()
    write_json(task_dir / "task.json", record)
    return record


def summarize(records, protocol, output):
    rows = []
    for record in records:
        for method in METHODS:
            metric = record["metrics"].get(method)
            row = {"task_id": record["task_id"], "law": record["law"],
                   "seed": record["seed"], "method": method,
                   "status": "available" if metric is not None else "missing",
                   "mse": None if metric is None else metric["mse"],
                   "C_mse": None if metric is None else metric["C_mse"]}
            for horizon in OFFSETS:
                row[f"mse_h{horizon}"] = None if metric is None else metric["horizon_mse"][str(horizon)]
            rows.append(row)
    with (Path(output) / "task_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for law in LAWS:
        for method in METHODS:
            selection = [r for r in rows if r["law"] == law and r["method"] == method]
            complete = len(selection) == len(protocol["seeds"]) and all(r["mse"] is not None for r in selection)
            aggregate.append({
                "law": law, "method": method, "complete": complete,
                "tasks_available": sum(r["mse"] is not None for r in selection),
                "tasks_expected": len(protocol["seeds"]),
                "mean_mse": float(np.mean([r["mse"] for r in selection])) if complete else None,
                "horizon_mse": {str(int(h)): float(np.mean([r[f"mse_h{h}"] for r in selection]))
                                if complete else None for h in OFFSETS},
            })
    summary = {
        "study_status": protocol["study_status"], "not_confirmatory": True,
        "task_count": len(records),
        "complete_task_count": sum(r["status"] == "completed" for r in records),
        "expected_task_count": len(protocol["tasks"]),
        "expected_forecasts": len(protocol["tasks"]) * len(METHODS),
        "available_forecasts": sum(len(r["metrics"]) for r in records),
        "law_method_aggregates": aggregate,
        "interpretation": "Descriptive three-repetition exploratory known-law pilot; no confidence intervals, no new-source confirmation.",
    }
    write_json(Path(output) / "summary.json", summary)
    return summary


def run(output, mode):
    output = Path(output).resolve()
    protocol = create_protocol(output, mode)
    geometry = load_geometry(output / "source_snapshot" / "geometry.py")
    records = []
    for index, task in enumerate(protocol["tasks"], start=1):
        print(json.dumps({"event": "task_started", "index": index,
                          "total": len(protocol["tasks"]), **task}), flush=True)
        record = run_task(task, protocol, geometry, output)
        records.append(record)
        print(json.dumps({"event": "task_finished", "task_id": task["task_id"],
                          "status": record["status"],
                          "suite_wall_seconds": record["suite_wall_seconds"]}), flush=True)
    summary = summarize(records, protocol, output)
    print(json.dumps({"event": "run_finished", "output": str(output),
                      "complete_tasks": summary["complete_task_count"],
                      "expected_tasks": summary["expected_task_count"]}), flush=True)
    return 0 if summary["complete_task_count"] == summary["expected_task_count"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--quick", action="store_true", help="implementation checks on separate seeds, not study evidence")
    action.add_argument("--run", action="store_true", help="run the frozen exploratory 3-law x 3-seed batch")
    action.add_argument("--audit", action="store_true", help="audit an existing output directory")
    parser.add_argument("--output", type=Path, required=True, help="new output directory, or existing directory for --audit")
    args = parser.parse_args()
    if args.audit:
        spec = importlib.util.spec_from_file_location("kahm_audit", Path(__file__).with_name("audit_pilot.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.audit(args.output)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["passed"] else 1
    return run(args.output, "quick" if args.quick else "run")


if __name__ == "__main__":
    raise SystemExit(main())
