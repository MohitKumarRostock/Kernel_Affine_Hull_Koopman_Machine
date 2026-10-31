"""Frozen strong-baseline, residual-target and supervised-metric experiment.

No historical source or outcome is overwritten. Outer validation selects the
final base and heads. Every OOF donor is tuned using its own training data only.
All reported predictions come from training-only fits (no pretest refit).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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
from threadpoolctl import threadpool_limits, threadpool_info

from . import data
from .models import LearnedDiagonalMetric, RobustLocalFeatureBank, RidgeHead
from extensions.forecasting_utility.models import RawForecastRegressor
from extensions.predictive_geometry.study.datasets import make_forecast_split

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def config():
    return json.loads((ROOT / "config.json").read_text())


def methods(cfg):
    return ["ridge", "raw_rbf", "naive", "joint_kahm_ridge"] + [
        f"{stage}_{gate}" for stage in ("fixed_is", "fixed_oof", "learned_oof") for gate in cfg["gates"]]


def mse(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid prediction shape or nonfinite values")
    result = float(np.mean((pred - target) ** 2))
    if not np.isfinite(result):
        raise FloatingPointError("Nonfinite mean squared loss")
    return result


def verify_freeze():
    frozen = json.loads((ROOT / "freeze.json").read_text())
    for name, expected in frozen["files"].items():
        if data.digest(REPO / name) != expected:
            raise ValueError(f"Frozen scientific input changed: {name}")
    publication = json.loads((ROOT / "PUBLICATION.json").read_text())
    if publication["freeze_sha256"] != data.digest(ROOT / "freeze.json"):
        raise ValueError("Public execution checkpoint does not match this freeze")
    return frozen


def roster(cfg):
    tasks = []
    for law in cfg["synthetic"]["laws"]:
        for seed in cfg["synthetic"]["repetition_seeds"]:
            tasks.append({"task_id": f"synthetic__{law}__{seed}", "phase": "synthetic",
                          "law": law, "source_id": law, "seed": seed, "loss_scale": 1.0,
                          "context": cfg["synthetic"]["history"],
                          "horizon": max(cfg["synthetic"]["horizons"])})
    natural = json.loads(data.ROSTER_PATH.read_text())
    for item in natural["tasks"]:
        tasks.append({**item, "phase": "natural", "task_id": "natural__" + item["base_task_id"],
                      "seed": cfg["natural"]["seed"]})
    return tasks


def arrays_for(task, cfg):
    if task["phase"] == "natural":
        return data.load_natural(task)
    settings = cfg["synthetic"]
    law_index = settings["laws"].index(task["law"])
    result = {}
    for split_index, (name, count) in enumerate((("train", settings["training_episodes"]),
                                                ("validation", settings["validation_episodes"]),
                                                ("test", settings["test_episodes"]))):
        seed = int(np.random.SeedSequence([task["seed"], law_index, split_index,
                                          settings["seed_namespace"]]).generate_state(1)[0])
        part = make_forecast_split(task["law"], count, seed, history=settings["history"],
                                   horizons=tuple(settings["horizons"]), stride=settings["stride"])
        for key in ("X", "Y", "groups"):
            result[f"{key}_{name}"] = part[key]
        result[f"origins_{name}"] = np.tile(part["origins"], count)
    return result


def tune_raw(x, y, xv, yv, cfg, seed, head="ridge"):
    model = RawForecastRegressor(head=head, n_landmarks=cfg["rbf_landmarks"],
                                 seed=seed, head_seed=seed + 19)
    trials = model.fit_select(x, y, xv, yv, alphas=cfg["base_alphas"])
    return model, trials


def donor_split(indices, arrays, task):
    """Inner donor tuning never sees the corresponding outer OOF target."""
    if task["phase"] == "synthetic":
        groups = arrays["groups_train"]
        donor_groups = np.unique(groups[indices])
        cut = max(1, min(len(donor_groups) - 1, int(.8 * len(donor_groups))))
        fit = indices[np.isin(groups[indices], donor_groups[:cut])]
        valid = indices[np.isin(groups[indices], donor_groups[cut:])]
    else:
        cut = max(1, min(len(indices) - 1, int(.8 * len(indices))))
        valid = indices[cut:]
        origins = arrays["origins_train"]
        fit = indices[:cut]
        fit = fit[origins[fit] + task["horizon"] <= origins[valid[0]]]
    return fit, valid


def oof_plan(arrays, task, cfg):
    """Eligible rows depend only on indices and label availability, not fitted errors."""
    n = len(arrays["X_train"])
    if task["phase"] == "synthetic":
        groups = arrays["groups_train"]
        folds = [np.flatnonzero(groups % cfg["synthetic"]["outer_folds"] == k)
                 for k in range(cfg["synthetic"]["outer_folds"])]
    else:
        begin = max(1, int(np.ceil(cfg["natural"]["initial_fraction"] * n)))
        folds = list(np.array_split(np.arange(begin, n), cfg["natural"]["forward_folds"]))
    plan = []
    for number, query in enumerate(folds):
        if not len(query):
            continue
        if task["phase"] == "synthetic":
            donors = np.flatnonzero(~np.isin(arrays["groups_train"], arrays["groups_train"][query]))
        else:
            origins = arrays["origins_train"]
            donors = np.flatnonzero(origins + task["horizon"] <= origins[query[0]])
        plan.append({"fold": number, "query_indices": query.tolist(), "donor_indices": donors.tolist(),
                     "eligible": len(donors) >= 8})
    return plan


def oof_predictions(arrays, task, cfg):
    x, y = arrays["X_train"], arrays["Y_train"]
    output = np.full_like(y, np.nan)
    records = []
    plan = oof_plan(arrays, task, cfg)
    for record in plan:
        number = record["fold"]
        query, donors = np.asarray(record["query_indices"], int), np.asarray(record["donor_indices"], int)
        if not record["eligible"]:
            records.append({**record, "status": "insufficient_donors"})
            continue
        inner_fit, inner_validation = donor_split(donors, arrays, task)
        try:
            if len(inner_fit) >= 8 and len(inner_validation) >= 1:
                selected, trials = tune_raw(x[inner_fit], y[inner_fit], x[inner_validation],
                                           y[inner_validation], cfg, task["seed"] + number)
                alpha = selected.alpha
                tuning = "donor_only_inner_selection"
            else:
                alpha, trials, tuning = 1.0, [], "fixed_alpha_insufficient_inner_rows"
            donor = RawForecastRegressor(head="ridge", alpha=alpha, seed=task["seed"] + number)
            donor.fit(x[donors], y[donors])
            output[query] = donor.predict(x[query])
            records.append({**record, "status": "ok", "alpha": alpha, "tuning": tuning,
                            "inner_fit_indices": inner_fit.tolist(),
                            "inner_validation_indices": inner_validation.tolist(), "trials": trials})
        except Exception as error:
            records.append({**record, "status": "failed", "error": repr(error),
                            "inner_fit_indices": inner_fit.tolist(), "inner_validation_indices": inner_validation.tolist()})
    parts = [np.asarray(r["query_indices"], int) for r in plan if r["eligible"]]
    retained = np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=int)
    return retained, output[retained], records


def naive(arrays, task, split):
    width = arrays[f"Y_{split}"].shape[1]
    if task["phase"] == "synthetic":
        return np.repeat(arrays[f"X_{split}"][:, -1:], width, axis=1)
    values = arrays["values"]
    result = []
    for origin in arrays[f"origins_{split}"]:
        first = int(origin) + 1
        finite = np.flatnonzero(np.isfinite(values[:first]))
        last = values[finite[-1]]
        period = task["seasonal_period"] if first >= task["seasonal_period"] else 1
        indexes = first - period + np.arange(width) % period
        candidate = values[indexes]
        result.append(np.where(np.isfinite(candidate), candidate, last))
    return np.asarray(result)


def choose_head(features, residuals, validation_features, baseline_validation, validation_target, cfg):
    # The exact zero correction is first; strict improvement is required to replace it.
    best = {"alpha": None, "gamma": 0.0, "validation_mse": mse(baseline_validation, validation_target)}
    trials = [dict(best)]
    best_head = None
    heads = RidgeHead.fit_path(features, residuals, cfg["correction_alphas"])
    for alpha in cfg["correction_alphas"]:
        head = heads[alpha]
        correction = head.predict(validation_features)
        for gamma in cfg["gammas"]:
            if gamma == 0:
                continue
            loss = mse(baseline_validation + gamma * correction, validation_target)
            trial = {"alpha": alpha, "gamma": gamma, "validation_mse": loss}
            trials.append(trial)
            if loss < best["validation_mse"]:
                best, best_head = trial, head
    return best_head, best, trials


def choose_joint(features, target, validation_features, validation_target, cfg):
    path = RidgeHead.fit_path(features, target, cfg["correction_alphas"])
    best_loss, selected, chosen_alpha, trials = np.inf, None, None, []
    for alpha in cfg["correction_alphas"]:
        head = path[alpha]
        loss = mse(head.predict(validation_features), validation_target)
        trials.append({"alpha": alpha, "validation_mse": loss})
        if loss < best_loss:
            best_loss, selected, chosen_alpha = loss, head, alpha
    return selected, chosen_alpha, trials


def run_task(task, arrays, cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    x, y = arrays["X_train"], arrays["Y_train"]
    xv, yv = arrays["X_validation"], arrays["Y_validation"]
    xt, yt = arrays["X_test"], arrays["Y_test"]
    if min(len(x), len(xv), len(xt)) == 0:
        raise ValueError("Unavailable declared split; record failure without replacing task")
    nodes, records, predictions, models = {}, {}, {}, {}
    caught = []

    def measured(name, action):
        start, cpu = time.perf_counter(), time.process_time()
        try:
            return action()
        finally:
            nodes[name] = {"wall_seconds": time.perf_counter() - start,
                           "cpu_seconds": time.process_time() - cpu}

    def register(name, pred, selected, dependencies, model=None, extra=None):
        predictions[name] = np.asarray(pred)
        loss = mse(pred, yt)
        serialized = measured(name + ":serialize", lambda: pickle.dumps(model, protocol=5)) if model is not None else b""
        dependencies = list(dict.fromkeys([*dependencies, *([name + ":serialize"] if model is not None else [])]))
        records[name] = {"test_mse": loss, "scaled_test_mse": loss / task["loss_scale"],
                         "horizon_mse": np.mean((pred - yt) ** 2, axis=0).tolist(),
                         "selected": selected, "dependencies": dependencies,
                         "pipeline_wall_seconds": sum(nodes[node]["wall_seconds"] for node in dependencies),
                         "pipeline_cpu_seconds": sum(nodes[node]["cpu_seconds"] for node in dependencies),
                         "serialized_model_bytes": len(serialized), "status": "ok", **(extra or {})}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw, raw_trials = measured("ridge:search", lambda: tune_raw(x, y, xv, yv, cfg, task["seed"]))
        raw_train = measured("ridge:train_predict", lambda: raw.predict(x))
        raw_val = measured("ridge:validation_predict", lambda: raw.predict(xv))
        raw_test = measured("ridge:predict", lambda: raw.predict(xt))
        register("ridge", raw_test, {"alpha": raw.alpha, "trials": raw_trials},
                 ["ridge:search", "ridge:predict"], raw)
        try:
            rbf, trials = measured("raw_rbf:search", lambda: tune_raw(x, y, xv, yv, cfg, task["seed"], "rbf"))
            pred = measured("raw_rbf:predict", lambda: rbf.predict(xt))
            register("raw_rbf", pred, {"alpha": rbf.alpha, "trials": trials}, ["raw_rbf:search", "raw_rbf:predict"], rbf)
        except Exception as error:
            register("raw_rbf", raw_test, {"fallback": "ridge", "error": repr(error)},
                     ["ridge:search", "ridge:predict", *[n for n in nodes if n.startswith("raw_rbf:")]], raw,
                     {"status": "fallback"})
        pred = measured("naive:predict", lambda: naive(arrays, task, "test"))
        register("naive", pred, {}, ["naive:predict"])

        plan = oof_plan(arrays, task, cfg)
        parts = [np.asarray(r["query_indices"], int) for r in plan if r["eligible"]]
        indices = np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=int)
        try:
            actual_indices, donor_predictions, fold_records = measured("oof:fit", lambda: oof_predictions(arrays, task, cfg))
        except Exception as error:
            actual_indices = indices.copy()
            donor_predictions = np.full_like(y[indices], np.nan)
            fold_records = [{**item, "status": "failed" if item["eligible"] else "insufficient_donors",
                             "error": repr(error)} for item in plan]
        if not np.array_equal(actual_indices, indices):
            raise AssertionError("OOF fitted outcomes changed deterministic correction row eligibility")
        oof_failure = ("Fewer than eight eligible correction targets" if len(indices) < 8 else
                       None if np.isfinite(donor_predictions).all() else
                       "At least one eligible donor fit failed; dependent arms use ridge")
        xc = x[indices]
        residuals = {"is": y[indices] - raw_train[indices], "oof": None if oof_failure else y[indices] - donor_predictions}
        metric_groups = (arrays["groups_train"][indices] if task["phase"] == "synthetic"
                         else arrays["origins_train"][indices])
        metric_radius = 0 if task["phase"] == "synthetic" else task["context"] + task["horizon"]
        metric_positions = None if task["phase"] == "synthetic" else arrays["origins_train"][indices]
        metric = LearnedDiagonalMetric(seed=task["seed"], **cfg["metric"])
        metric_error = None
        try:
            if oof_failure:
                raise ValueError(oof_failure)
            measured("metric:fit", lambda: metric.fit(xc, residuals["oof"], groups=metric_groups,
                                                      positions=metric_positions,
                                                      exclusion_radius=metric_radius, scaling_data=x))
            weights, metric_metadata = metric.weights_, metric.metadata()
        except Exception as error:
            weights, metric_error = np.ones(x.shape[1]), repr(error)
            metric_metadata = {"fallback": "identity", "error": metric_error}
            nodes.setdefault("metric:fit", {"wall_seconds": 0.0, "cpu_seconds": 0.0})
        metric_identity_fallback = metric_error is not None or metric_metadata.get("fallback_reason") is not None
        bank_records = {}
        for gate in cfg["gates"]:
            for geometry in ("fixed", "learned"):
                key = f"bank:{geometry}:{gate}"
                bank = RobustLocalFeatureBank(gate=gate, metric_weights=(None if geometry == "fixed" else weights),
                                               seed=task["seed"], **cfg["geometry"])
                try:
                    measured(key + ":fit", lambda: bank.fit(x))
                    features = measured(key + ":train", lambda: bank.transform(xc))
                    fval = measured(key + ":validation", lambda: bank.transform(xv))
                    ftest = measured(key + ":test", lambda: bank.transform(xt))
                    bank_records[f"{geometry}_{gate}"] = bank.metadata()
                    failure = None
                except Exception as error:
                    failure = repr(error)
                    bank_records[f"{geometry}_{gate}"] = {"failure": failure}
                stages = ("is", "oof") if geometry == "fixed" else ("oof",)
                for mode in stages:
                    name = f"{geometry}_{mode}_{gate}"
                    dependencies = ["ridge:search", "ridge:train_predict", "ridge:validation_predict", "ridge:predict"]
                    if mode == "oof":
                        dependencies.append("oof:fit")
                    if geometry == "learned":
                        dependencies.append("metric:fit")
                    dependencies += [n for n in nodes if n.startswith(key + ":")]
                    if failure or len(indices) < 8 or (mode == "oof" and oof_failure):
                        register(name, raw_test, {"fallback": "ridge", "error": failure or oof_failure}, dependencies, raw,
                                 {"status": "fallback"})
                        continue
                    try:
                        head, chosen, trials = measured(name + ":search", lambda: choose_head(
                            features, residuals[mode], fval, raw_val, yv, cfg))
                        if head is None:
                            pred = raw_test.copy()
                            model_state = {"raw": raw, "gamma": 0.0}
                        else:
                            correction = measured(name + ":predict", lambda: head.predict(ftest))
                            pred = raw_test + chosen["gamma"] * correction
                            dependencies.append(name + ":predict")
                            model_state = {"raw": raw, "bank": bank, "head": head, "gamma": chosen["gamma"]}
                        delta = pred - raw_test
                        remaining = yt - raw_test
                        register(name, pred, {**chosen, "trials": trials}, [*dependencies, name + ":search"], model_state,
                                 {"residual_alignment": float(np.mean(remaining * delta)),
                                  "correction_energy": float(np.mean(delta ** 2)),
                                  "fallback_metric_identity": metric_identity_fallback and geometry == "learned"})
                    except Exception as error:
                        register(name, raw_test, {"fallback": "ridge", "error": repr(error)},
                                 [*dependencies, *[n for n in nodes if n.startswith(name + ":")]], raw,
                                 {"status": "fallback"})
                if gate == "kahm" and geometry == "fixed":
                    name = "joint_kahm_ridge"
                    dependencies = [n for n in nodes if n.startswith(key + ":")]
                    if failure:
                        register(name, raw_test, {"fallback": "ridge", "error": failure},
                                 ["ridge:search", "ridge:predict", *dependencies], raw, {"status": "fallback"})
                    else:
                        try:
                            ftrain_all = measured(name + ":train_features", lambda: bank.transform(x))
                            joint_train, joint_val, joint_test = (np.column_stack((x, ftrain_all)),
                                                                 np.column_stack((xv, fval)), np.column_stack((xt, ftest)))
                            selected_head, chosen_alpha, trials = measured(name + ":search", lambda: choose_joint(
                                joint_train, y, joint_val, yv, cfg))
                            pred = measured(name + ":predict", lambda: selected_head.predict(joint_test))
                            register(name, pred, {"alpha": chosen_alpha, "trials": trials},
                                     [*dependencies, name + ":train_features", name + ":search", name + ":predict"],
                                     {"bank": bank, "head": selected_head})
                        except Exception as error:
                            register(name, raw_test, {"fallback": "ridge", "error": repr(error)},
                                     ["ridge:search", "ridge:predict", *dependencies,
                                      *[n for n in nodes if n.startswith(name + ":")]], raw,
                                     {"status": "fallback"})
    warning_records = [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]
    if set(records) != set(methods(cfg)):
        raise AssertionError("Method roster incomplete")
    stored = {"target": yt, "validation_target": yv, "origins_test": arrays["origins_test"],
              "correction_indices": indices, "donor_predictions": donor_predictions,
              "final_base_train_predictions": raw_train[indices], "correction_targets": y[indices],
              **{f"prediction__{name}": value for name, value in predictions.items()}}
    if "groups_test" in arrays:
        stored["groups_test"] = arrays["groups_test"]
    np.savez_compressed(directory / "predictions.npz", **stored)
    disagreement = None if oof_failure else float(np.mean((donor_predictions - raw_train[indices]) ** 2))
    report = {"task": task, "created_at_utc": now(), "methods": records, "nodes": nodes,
              "warnings": warning_records, "metric": metric_metadata, "banks": bank_records,
              "oof_folds": fold_records, "oof_failure": oof_failure, "correction_rows": len(indices),
              "donor_final_disagreement_mse": disagreement,
              "in_sample_residual_mse": float(np.mean(residuals["is"] ** 2)) if len(indices) else None,
              "oof_residual_mse": None if oof_failure else float(np.mean(residuals["oof"] ** 2)),
              "arrays_sha256": {key: data.array_hash(value) for key, value in arrays.items()},
              "predictions_sha256": data.digest(directory / "predictions.npz")}
    data.write_json(directory / "result.json", report)
    return report


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "workers": 1, "blas_threads": 1, "threadpools": threadpool_info(),
            "timing_scope": "Warm component sums, including required baseline/OOF/metric/geometry/head selection, test prediction and retained-model serialization. Shared components charged once per standalone procedure. Imports, source download, preparation, artifact writes and reporting excluded. Geometry transforms of test inputs are conservatively charged even when gamma=0; no test outcomes enter selection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v1")
    parser.add_argument("--phase", choices=("all", "synthetic", "natural"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg, frozen = config(), verify_freeze()
    tasks = [t for t in roster(cfg) if args.phase == "all" or t["phase"] == args.phase]
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new run directory or explicit --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    execution = {"created_at_utc": now(), "config_sha256": data.digest(ROOT / "config.json"),
                 "freeze_sha256": data.digest(ROOT / "freeze.json"), "tasks": tasks, "methods": methods(cfg)}
    with threadpool_limits(limits=1):
        execution["environment"] = environment()
        manifest = args.output / "execution.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            for key in ("config_sha256", "freeze_sha256", "tasks", "methods"):
                if previous[key] != execution[key]:
                    raise ValueError(f"Cannot resume a different {key}")
        else:
            data.write_json(manifest, execution)
        for number, task in enumerate(tasks):
            directory = args.output / task["task_id"]
            if directory.exists():
                if args.resume and ((directory / "result.json").exists() or (directory / "failure.json").exists()):
                    continue
                raise FileExistsError(f"Incomplete task output retained for inspection: {directory}")
            print(f"{number+1}/{len(tasks)} {task['task_id']}", flush=True)
            try:
                arrays = arrays_for(task, cfg)
                result = run_task(task, arrays, cfg, directory)
                print(f"completed: {len(result['methods'])} forecasts, {sum(v['status'] == 'fallback' for v in result['methods'].values())} fallbacks", flush=True)
            except Exception as error:
                directory.mkdir(parents=True, exist_ok=True)
                data.write_json(directory / "failure.json", {"task": task, "error": repr(error),
                                                            "traceback": traceback.format_exc(), "created_at_utc": now()})
                print(f"FAILED: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
