"""Execute the frozen forecasting-utility study; preserve every task and failure."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import pickle
import platform
import time
import traceback
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_info, threadpool_limits

from . import data
from .models import AdditiveForecastRegressor, RawForecastRegressor
from extensions.natural_forecasting.models import ForecastRegressor

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
FREEZE = ROOT / "freeze.json"
POLICIES = ("utility_kahm_5pct", "utility_geometry_5pct")
GEOMETRY_CANDIDATES = ("add_kahm", "add_affine", "add_locality")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify_freeze():
    if not FREEZE.is_file():
        raise RuntimeError("Execution forbidden before the source/protocol/roster freeze")
    frozen = json.loads(FREEZE.read_text())
    checks = {**frozen["files"],
              "extensions/forecasting_utility/PROTOCOL.md": frozen["protocol_sha256"],
              "extensions/forecasting_utility/prepared/roster.json": frozen["roster_sha256"]}
    for name, expected in checks.items():
        if data.digest(REPOSITORY / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    return frozen


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "logical_cpus": os.cpu_count(), "workers": 1, "blas_threads": 1,
            "threadpools": [{k: item.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
                            for item in threadpool_info()],
            "timing_scope": "Charged sums of warm-library computation nodes. Model pipeline includes required search, selected archive pretest refit, prediction and pure pickle serialization; data-inclusive pipeline additionally includes load/generation. Imports, download, prediction artifact I/O and shared research reporting excluded from model pipeline. Selector costs are component unions, not separately timed policy executions."}


def finite_mse(prediction, target):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction shape/finiteness failure; no replacement forecast")
    loss = float(np.mean(np.square(prediction - target)))
    if not np.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    return loss


def utility_choice(raw_loss, additive_loss, threshold):
    if min(raw_loss, additive_loss) < 0 or not np.isfinite([raw_loss, additive_loss]).all():
        raise ValueError("Policy losses must be finite and nonnegative")
    return "add_kahm" if raw_loss > 0 and additive_loss <= (1.0 - threshold) * raw_loss else "raw_rbf"


def geometry_choice(losses, threshold):
    """Fixed geometry order breaks correction ties; raw requires 5% gain."""
    values = [losses[name] for name in ("raw_rbf", *GEOMETRY_CANDIDATES)]
    if min(values) < 0 or not np.isfinite(values).all():
        raise ValueError("Policy losses must be finite and nonnegative")
    best = min(GEOMETRY_CANDIDATES, key=lambda name: losses[name])
    return best if losses["raw_rbf"] > 0 and losses[best] <= (1.0 - threshold) * losses["raw_rbf"] else "raw_rbf"


def search_dependencies(method):
    if method.startswith("add_"):
        return {"raw_rbf:search", "raw_rbf:train_cache", "raw_rbf:validation_cache", f"{method}:search"}
    return {f"{method}:search"}


def final_dependencies(method, phase, gamma=None):
    result = {f"{method}:predict", f"{method}:serialize"}
    if phase == "natural":
        result.add(f"{method}:refit")
    if method.startswith("add_"):
        result.add("raw_rbf:predict")
        if phase == "natural":
            result.add("raw_rbf:refit")
            if gamma and gamma > 0:
                result.add("raw_rbf:refit_train_cache")
    return result


def sum_nodes(node_ids, nodes, metric="wall_seconds"):
    lookup = {node["id"]: node for node in nodes}
    missing = set(node_ids) - set(lookup)
    if missing:
        raise ValueError(f"Missing charged computation nodes: {sorted(missing)}")
    return sum(lookup[key][metric] for key in set(node_ids))


def construct(method, task, cfg, *, alpha=None, gamma=1.0):
    alpha = cfg["alphas"][0] if alpha is None else alpha
    seed = task["seed"]
    common = {"alpha": alpha, "n_landmarks": task["landmarks"],
              "bandwidth_multiplier": cfg["bandwidth_multiplier"], "seed": seed, "head_seed": seed + 1000}
    if method in ("raw_ridge", "raw_rbf"):
        return RawForecastRegressor(head="ridge" if method == "raw_ridge" else "rbf", **common)
    if method == "concat_kahm":
        return ForecastRegressor(representation="kahm_local", head="rbf", raw_skip=True,
                                 geometry_parameters=cfg["geometry"], query_mode=cfg["query_mode"], **common)
    if method.startswith("add_"):
        return AdditiveForecastRegressor(gate=method[4:], alpha=alpha, gamma=gamma,
                                         seed=seed, geometry_parameters=cfg["geometry"])
    if method == "hist_gradient":
        return MultiOutputRegressor(HistGradientBoostingRegressor(random_state=seed, **cfg["hist_gradient"]), n_jobs=1)
    if method == "naive":
        return None
    raise ValueError(method)


def naive_prediction(arrays, split, task, model=None):
    width = arrays["Y_train"].shape[1]
    if task["phase"] == "natural":
        # Same causal seasonal formula as the inherited SeasonalNaive, kept
        # local to avoid importing its unused statsmodels/Holt dependency.
        values = arrays["values"]
        forecasts = []
        for last_observed in arrays[f"origins_{split}"]:
            first_future = int(last_observed) + 1
            finite = np.flatnonzero(np.isfinite(values[:first_future]))
            if not len(finite):
                raise ValueError("No finite historical observation for naive forecast")
            last = float(values[finite[-1]])
            period = task["seasonal_period"] if first_future >= task["seasonal_period"] else 1
            indices = first_future - period + (np.arange(width) % period)
            if not np.all(indices < first_future):
                raise AssertionError("Seasonal naive attempted future access")
            forecast = np.asarray(values[indices], dtype=float).copy()
            forecast[~np.isfinite(forecast)] = last
            forecasts.append(forecast)
        return np.asarray(forecasts).reshape(-1, width)
    return np.repeat(arrays[f"X_{split}"][:, -1:], width, axis=1)


def run_task(task, arrays, metadata, cfg, destination, *, preparation_seconds=0.0):
    destination.mkdir(parents=True, exist_ok=False)
    nodes = [{"id": "data_preparation", "wall_seconds": preparation_seconds, "cpu_seconds": None, "status": "ok"}]
    records, search_models, final_models, predictions = {}, {}, {}, {}
    raw_train, raw_validation, raw_refit_train = None, None, None
    errors, all_warnings = [], []
    target_width = arrays["Y_train"].shape[1]

    def measured(identifier, function):
        started, cpu = time.perf_counter(), time.process_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = function()
            except Exception as error:
                nodes.append({"id": identifier, "wall_seconds": time.perf_counter() - started,
                              "cpu_seconds": time.process_time() - cpu, "status": "failed"})
                errors.append({"node": identifier, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
                raise
            finally:
                all_warnings.extend({"node": identifier, "category": item.category.__name__, "message": str(item.message)} for item in caught)
        nodes.append({"id": identifier, "wall_seconds": time.perf_counter() - started,
                      "cpu_seconds": time.process_time() - cpu, "status": "ok"})
        return value

    # Every validation search precedes every test prediction. Raw selection is
    # fixed before residual targets are formed; test targets never select a fit.
    for method in cfg["methods"]:
        record = {"method": method, "status": "pending", "trials": []}
        records[method] = record
        try:
            model = construct(method, task, cfg)
            if method.startswith("add_"):
                if "raw_rbf" not in search_models:
                    raise RuntimeError("Required raw backbone failed; additive method remains a failure")
                if raw_train is None:
                    raw_train = measured("raw_rbf:train_cache", lambda: search_models["raw_rbf"].predict(arrays["X_train"]))
                if raw_validation is None:
                    raw_validation = measured("raw_rbf:validation_cache", lambda: search_models["raw_rbf"].predict(arrays["X_validation"]))
                trials = measured(f"{method}:search", lambda: model.fit_select(
                    arrays["X_train"], arrays["Y_train"], arrays["X_validation"], arrays["Y_validation"],
                    cfg["alphas"], cfg["gammas"], raw_model=search_models["raw_rbf"],
                    raw_train_prediction=raw_train, raw_validation_prediction=raw_validation))
                record.update(alpha=float(model.alpha), gamma=float(model.gamma))
            elif method in ("raw_ridge", "raw_rbf", "concat_kahm"):
                trials = measured(f"{method}:search", lambda: model.fit_select(
                    arrays["X_train"], arrays["Y_train"], arrays["X_validation"], arrays["Y_validation"], cfg["alphas"]))
                record["alpha"] = float(model.alpha)
            elif method == "hist_gradient":
                def tree_search():
                    model.fit(arrays["X_train"], arrays["Y_train"])
                    return [{"fixed_configuration": cfg["hist_gradient"], "validation_mse": finite_mse(model.predict(arrays["X_validation"]), arrays["Y_validation"])}]
                trials = measured(f"{method}:search", tree_search)
            else:
                def naive_search():
                    return [{"fixed_configuration": "seasonal naive with causal fallback" if task["phase"] == "natural" else "persistence",
                             "validation_mse": finite_mse(naive_prediction(arrays, "validation", task, model), arrays["Y_validation"])}]
                trials = measured(f"{method}:search", naive_search)
            record.update(status="selected", trials=trials, validation_mse=min(row["validation_mse"] for row in trials))
            if hasattr(model, "metadata"):
                record["search_model_metadata"] = model.metadata()
            search_models[method] = model
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")

    policies = {}
    if all(records[name]["status"] == "selected" for name in ("raw_rbf", "add_kahm")):
        selected = measured("utility_kahm_5pct:decision", lambda: utility_choice(
            records["raw_rbf"]["validation_mse"], records["add_kahm"]["validation_mse"],
            cfg["utility_relative_validation_improvement"]))
        policies["utility_kahm_5pct"] = {"status": "selected", "selected_method": selected,
                                             "selection_methods": ["raw_rbf", "add_kahm"]}
    else:
        policies["utility_kahm_5pct"] = {"status": "failed", "error": "Required validation candidate failed"}
    geometry_methods = ["raw_rbf", *GEOMETRY_CANDIDATES]
    if all(records[name]["status"] == "selected" for name in geometry_methods):
        selected = measured("utility_geometry_5pct:decision", lambda: geometry_choice(
            {name: records[name]["validation_mse"] for name in geometry_methods}, cfg["utility_relative_validation_improvement"]))
        policies["utility_geometry_5pct"] = {"status": "selected", "selected_method": selected,
                                               "selection_methods": geometry_methods}
    else:
        policies["utility_geometry_5pct"] = {"status": "failed", "error": "At least one required geometry candidate failed"}

    # Refit only archive-source tasks. Synthetic fitted models keep exactly N
    # training episodes; their selected in-memory search fit is retained.
    for method in cfg["methods"]:
        record = records[method]
        if record["status"] != "selected":
            continue
        try:
            if task["phase"] == "synthetic":
                final = search_models[method]
                record["refit_policy"] = "none; selected training-only fit retained"
            else:
                final = construct(method, task, cfg, alpha=record.get("alpha"), gamma=record.get("gamma", 1.0))
                if method.startswith("add_"):
                    if "raw_rbf" not in final_models:
                        raise RuntimeError("Final raw backbone failed")
                    if record["gamma"] > 0 and raw_refit_train is None:
                        raw_refit_train = measured("raw_rbf:refit_train_cache", lambda: final_models["raw_rbf"].predict(arrays["X_refit"]))
                    measured(f"{method}:refit", lambda: final.fit(arrays["X_refit"], arrays["Y_refit"],
                        raw_model=final_models["raw_rbf"], raw_train_prediction=raw_refit_train))
                elif method == "naive":
                    measured(f"{method}:refit", lambda: None)
                else:
                    measured(f"{method}:refit", lambda: final.fit(arrays["X_refit"], arrays["Y_refit"]))
                record["refit_policy"] = "all eligible pretest targets, capped512; preprocessing refitted"
            final_models[method] = final
            if method.startswith("add_"):
                if "raw_rbf" not in predictions:
                    raise RuntimeError("Final raw test prediction failed")
                prediction = measured(f"{method}:predict", lambda: final.predict(arrays["X_test"], raw_prediction=predictions["raw_rbf"]))
            elif method == "naive":
                prediction = measured(f"{method}:predict", lambda: naive_prediction(arrays, "test", task, final))
            else:
                prediction = measured(f"{method}:predict", lambda: final.predict(arrays["X_test"]))
            loss = finite_mse(prediction, arrays["Y_test"])
            predictions[method] = prediction
            record.update(test_mse=loss, scaled_mse=loss / float(task.get("loss_scale", 1.0)))
            if method.startswith("add_"):
                record["gamma_zero_matches_raw_exactly"] = bool(np.array_equal(prediction, predictions["raw_rbf"])) if record["gamma"] == 0 else None
            if hasattr(final, "metadata"):
                record["final_model_metadata"] = final.metadata()
            def serialize():
                packed = pickle.dumps(final if final is not None else {"method": "naive", "target_width": target_width, "seasonal_period": task.get("seasonal_period", 1)}, protocol=pickle.HIGHEST_PROTOCOL)
                return len(packed)
            record["serialized_model_bytes"] = measured(f"{method}:serialize", serialize)
            measured(f"{method}:artifact_write", lambda: np.savez_compressed(destination / f"prediction_{method}.npz", prediction=prediction))
            record["prediction_file"] = f"prediction_{method}.npz"
            record["prediction_sha256"] = data.array_hash(prediction)
            record["status"] = "ok"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")

    for method, record in records.items():
        if record["status"] != "ok":
            continue
        ids = {"data_preparation"} | search_dependencies(method) | final_dependencies(method, task["phase"], record.get("gamma"))
        record["charged_nodes"] = sorted(ids)
        record["pipeline_seconds"] = sum_nodes(ids, nodes)
        record["model_pipeline_seconds"] = sum_nodes(ids - {"data_preparation"}, nodes)
        record["model_cpu_seconds"] = sum_nodes(ids - {"data_preparation"}, nodes, "cpu_seconds")
    for name, policy in policies.items():
        if policy["status"] != "selected":
            continue
        selected = policy["selected_method"]
        if records[selected]["status"] != "ok":
            policy.update(status="failed", error="Selected final fit or forecast failed")
            continue
        ids = {"data_preparation", f"{name}:decision"}
        for method in policy["selection_methods"]:
            ids |= search_dependencies(method)
        ids |= final_dependencies(selected, task["phase"], records[selected].get("gamma"))
        policy.update(status="ok", charged_nodes=sorted(ids), pipeline_seconds=sum_nodes(ids, nodes),
                      model_pipeline_seconds=sum_nodes(ids - {"data_preparation"}, nodes),
                      model_cpu_seconds=sum_nodes(ids - {"data_preparation"}, nodes, "cpu_seconds"),
                      test_mse=records[selected]["test_mse"], scaled_mse=records[selected]["scaled_mse"],
                      serialized_model_bytes=records[selected]["serialized_model_bytes"])
    targets = {key: value for key, value in arrays.items() if key.startswith(("Y_", "groups_", "origins_"))}
    measured("targets:artifact_write", lambda: np.savez_compressed(destination / "targets.npz", **targets))
    payload = {"schema_version": 1, "task": task, "data_metadata": metadata, "nodes": nodes,
               "methods": list(records.values()), "policies": policies, "warnings": all_warnings,
               "errors": errors, "complete": all(record["status"] == "ok" for record in records.values()) and all(p["status"] == "ok" for p in policies.values()),
               "completed_at_utc": now()}
    data.write_json(destination / "result.json", payload)
    return payload


def task_roster(cfg):
    tasks = []
    for law in cfg["synthetic"]["laws"]:
        for seed in cfg["synthetic"]["repetition_seeds"]:
            for n in cfg["synthetic"]["training_episodes"]:
                for m in cfg["landmarks"]:
                    tasks.append({"task_id": f"synthetic__{law}__seed{seed}__N{n}__M{m}",
                                  "phase": "synthetic", "law": law, "seed": seed, "training_episodes": n,
                                  "landmarks": m, "loss_scale": 1.0})
    roster = json.loads(data.ROSTER_PATH.read_text())
    for base in roster["tasks"]:
        for m in cfg["landmarks"]:
            tasks.append({**base, "task_id": f"natural__{base['base_task_id']}__M{m}",
                          "phase": "natural", "seed": cfg["natural"]["seed"], "landmarks": m})
    return tasks


def run(phase, output, *, resume=False):
    verify_freeze()
    cfg = data.config()
    freeze_sha = data.digest(FREEZE)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "execution.json"
    expected = task_roster(cfg)
    if snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text())
        if previous["freeze_sha256"] != freeze_sha:
            raise RuntimeError("Output belongs to another execution freeze")
    else:
        if any(output.iterdir()):
            raise FileExistsError("New run directory must be empty")
        data.write_json(snapshot_path, {"started_at_utc": now(), "freeze_sha256": freeze_sha,
                                       "freeze": json.loads(FREEZE.read_text()), "expected_tasks": expected})
    selected = [task for task in expected if phase == "all" or task["phase"] == phase]
    completed, failed = 0, 0
    with threadpool_limits(limits=1):
        if not (output / "environment.json").exists():
            data.write_json(output / "environment.json", environment())
        for task in selected:
            destination = output / "tasks" / task["task_id"]
            if destination.exists():
                if not resume or not (destination / "result.json").exists():
                    raise FileExistsError("Task directory already exists; only explicit resume of completed records is permitted")
                record = json.loads((destination / "result.json").read_text())
                completed += 1
                failed += not record["complete"]
                continue
            verify_freeze()
            started = time.perf_counter()
            try:
                if task["phase"] == "synthetic":
                    arrays, metadata = data.synthetic_splits(task["law"], task["seed"], task["training_episodes"])
                else:
                    if not task["supported"]:
                        raise RuntimeError("Selected source series has no usable later split; no replacement allowed")
                    arrays, metadata = data.load_natural(task), task
                preparation_seconds = time.perf_counter() - started
                result = run_task(task, arrays, metadata, cfg, destination, preparation_seconds=preparation_seconds)
            except Exception as error:
                destination.mkdir(parents=True, exist_ok=True)
                result = {"task": task, "complete": False, "fatal_error": f"{type(error).__name__}: {error}",
                          "traceback": traceback.format_exc(), "methods": [], "policies": {}, "completed_at_utc": now()}
                data.write_json(destination / "result.json", result)
            completed += 1
            failed += not result["complete"]
            data.write_json(output / f"checkpoint_{phase}.json", {"completed": completed, "expected": len(selected), "failed": failed, "last_task": task["task_id"]})
            print(f"{completed}/{len(selected)} {task['task_id']}: {'complete' if result['complete'] else 'FAILED'}", flush=True)
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("synthetic", "natural", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "utility_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.phase, args.output_dir, resume=args.resume))


if __name__ == "__main__":
    main()
