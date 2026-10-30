"""Recompute outcomes from saved predictions and summarize the frozen study."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from . import data
from .experiment import (
    FREEZE, POLICIES, final_dependencies, geometry_choice, search_dependencies,
    sum_nodes, task_roster, utility_choice, verify_freeze,
)

ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(differences, family_size=24):
    values = np.asarray(differences, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need complete finite paired repetitions")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half_width = float(student_t.ppf(1 - 0.05 / (2 * family_size), len(values) - 1) * sd / math.sqrt(len(values)))
    return {"n": len(values), "mean_control_minus_additive": mean,
            "paired_sd": sd, "lower": mean - half_width, "upper": mean + half_width,
            "family_size": family_size, "coverage": "95% Bonferroni paired Student family; normal differences or approximation required"}


def audit_trials(method, cfg):
    """Check the entire declared grid and its deterministic selection rule."""
    trials = method.get("trials", [])
    if not trials:
        if method["status"] == "ok":
            raise ValueError("Successful method has no saved validation trials")
        return
    losses = np.asarray([trial["validation_mse"] for trial in trials], dtype=float)
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("Invalid saved validation loss")
    name = method["method"]
    if name.startswith("add_"):
        expected = [(cfg["alphas"][0], 0.0, False)] + [
            (alpha, gamma, True) for alpha in cfg["alphas"]
            for gamma in cfg["gammas"] if gamma > 0
        ]
        actual = [(trial["alpha"], trial["gamma"], trial["alpha_applies"]) for trial in trials]
        if actual != expected:
            raise ValueError("Correction trials differ from the frozen grid or repeat gamma zero")
        selected = min(trials, key=lambda trial: (
            trial["validation_mse"], trial["gamma"], cfg["alphas"].index(trial["alpha"])))
        if method["alpha"] != selected["alpha"] or method["gamma"] != selected["gamma"]:
            raise ValueError("Correction selection violates the loss/gamma/alpha-order rule")
    elif name in ("raw_ridge", "raw_rbf", "concat_kahm"):
        if [trial["alpha"] for trial in trials] != cfg["alphas"]:
            raise ValueError("Raw/concatenated trials differ from the frozen alpha grid")
        selected = min(trials, key=lambda trial: trial["validation_mse"])
        if method["alpha"] != selected["alpha"]:
            raise ValueError("Raw/concatenated alpha selection violates first-minimum rule")
    elif len(trials) != 1:
        raise ValueError("A fixed baseline must have one validation trial")
    if method["validation_mse"] != float(losses.min()):
        raise ValueError("Selected validation loss differs from the saved trial minimum")


def audit_cost(entry, expected_nodes, nodes):
    if (len(entry["charged_nodes"]) != len(expected_nodes)
            or set(entry["charged_nodes"]) != expected_nodes):
        raise ValueError("Charged computation dependencies differ from the frozen procedure")
    lookup = {node["id"]: node for node in nodes}
    if len(lookup) != len(nodes):
        raise ValueError("Repeated computation-node identifiers make costs ambiguous")
    if any(lookup.get(name, {}).get("status") != "ok" for name in expected_nodes):
        raise ValueError("A successful procedure depends on missing or failed computation")
    model_nodes = expected_nodes - {"data_preparation"}
    expected = {
        "pipeline_seconds": sum_nodes(expected_nodes, nodes),
        "model_pipeline_seconds": sum_nodes(model_nodes, nodes),
        "model_cpu_seconds": sum_nodes(model_nodes, nodes, "cpu_seconds"),
    }
    for name, value in expected.items():
        if not np.isfinite(value) or value < 0 or not np.isclose(entry[name], value, atol=1e-12, rtol=1e-12):
            raise ValueError(f"Cost accounting mismatch: {name}")


def audit_task(record, directory, cfg):
    task = record["task"]
    if [method["method"] for method in record["methods"]] != cfg["methods"]:
        raise ValueError("Task method roster/order differs from the frozen configuration")
    if set(record["policies"]) != set(POLICIES):
        raise ValueError("Task policy roster differs from the frozen configuration")
    all_ok = (all(method["status"] == "ok" for method in record["methods"])
              and all(policy["status"] == "ok" for policy in record["policies"].values()))
    if bool(record["complete"]) != all_ok:
        raise ValueError("Task completion flag disagrees with recorded method/policy status")
    hashes = record["data_metadata"]["arrays_sha256"]
    if task["phase"] == "natural" and hashes != task["arrays_sha256"]:
        raise ValueError("Natural target metadata differs from the frozen prepared-data roster")
    saved_names = {name for name in hashes if name.startswith(("Y_", "groups_", "origins_"))}
    with np.load(directory / "targets.npz", allow_pickle=False) as saved:
        if set(saved.files) != saved_names:
            raise ValueError("Saved target/group/origin array roster differs from data metadata")
        for name in saved.files:
            if data.array_hash(saved[name]) != hashes[name]:
                raise ValueError(f"Saved target/group/origin checksum mismatch: {name}")
        target = saved["Y_test"]
        episode_groups = saved["groups_test"] if "groups_test" in saved.files else None
    if target.ndim != 2 or not len(target) or not np.isfinite(target).all():
        raise ValueError("Invalid saved test targets")
    horizons = cfg["synthetic"]["horizons"] if task["phase"] == "synthetic" else list(range(1, target.shape[1] + 1))
    if len(horizons) != target.shape[1]:
        raise ValueError("Target width differs from declared horizons")
    if episode_groups is not None and episode_groups.shape != (len(target),):
        raise ValueError("Episode identifiers do not match saved target rows")
    scale = float(task.get("loss_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Task loss scale must be finite and positive")
    predictions, rows, trials, episode_rows, horizon_rows = {}, [], [], [], []
    for method in record["methods"]:
        audit_trials(method, cfg)
        for index, trial in enumerate(method.get("trials", [])):
            trials.append({"task_id": task["task_id"], "method": method["method"], "trial": index,
                           **{key: value for key, value in trial.items() if isinstance(value, (str, float, int, bool))}})
        if method["status"] != "ok":
            continue
        with np.load(directory / method["prediction_file"], allow_pickle=False) as saved:
            prediction = saved["prediction"]
        if data.array_hash(prediction) != method["prediction_sha256"]:
            raise ValueError("Prediction checksum mismatch")
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid prediction array")
        squared_error = np.square(prediction - target)
        loss = float(np.mean(squared_error))
        if not np.isclose(loss, method["test_mse"], rtol=1e-13, atol=1e-15):
            raise ValueError("Saved loss differs from prediction-derived loss")
        expected_nodes = {"data_preparation"} | search_dependencies(method["method"]) | final_dependencies(method["method"], task["phase"], method.get("gamma"))
        audit_cost(method, expected_nodes, record["nodes"])
        predictions[method["method"]] = prediction
        model_metadata = method.get("final_model_metadata", {})
        head_metadata = model_metadata.get("raw", model_metadata)
        row = {"task_id": task["task_id"], "phase": task["phase"], "source_id": task.get("source_id", ""),
                     "series_id": task.get("series_id", ""), "law": task.get("law", ""),
                     "training_episodes": task.get("training_episodes", 0), "landmarks": task["landmarks"], "seed": task["seed"],
                     "method": method["method"], "validation_mse": method["validation_mse"], "test_mse": loss,
                     "scaled_mse": loss / scale, "pipeline_seconds": method["pipeline_seconds"],
                     "model_pipeline_seconds": method["model_pipeline_seconds"],
                     "model_cpu_seconds": method["model_cpu_seconds"],
                     "serialized_model_bytes": method["serialized_model_bytes"], "alpha": method.get("alpha"), "gamma": method.get("gamma"),
                     "head_requested_landmarks": head_metadata.get("requested_landmarks"),
                     "head_actual_landmarks": head_metadata.get("actual_landmarks"),
                     "head_retained_nystrom_rank": head_metadata.get("retained_nystrom_rank"),
                     "head_basis_width": head_metadata.get("basis_width"),
                     "stored_correction_basis_width": model_metadata.get("correction_basis_width")}
        if method["method"].startswith("add_"):
            raw = predictions["raw_rbf"]
            delta = prediction - raw
            raw_loss = float(np.mean(np.square(target - raw)))
            alignment = float(2 * np.mean((target - raw) * delta))
            energy = float(np.mean(delta ** 2))
            improvement = raw_loss - loss
            tolerance = 64 * np.finfo(float).eps * max(raw_loss, loss, abs(alignment), energy, np.finfo(float).tiny)
            if not np.isclose(improvement, alignment - energy, rtol=1e-10, atol=tolerance):
                raise ValueError("Residual alignment/energy loss identity failed")
            row.update(raw_mse=raw_loss, correction_alignment=alignment, correction_energy=energy,
                       raw_minus_additive_mse=improvement,
                       identity_residual=improvement - (alignment - energy),
                       scaled_alignment=alignment / scale, scaled_correction_energy=energy / scale,
                       scaled_raw_minus_additive_mse=improvement / scale)
            if method["search_model_metadata"]["raw"]["alpha"] != next(
                    item["alpha"] for item in record["methods"] if item["method"] == "raw_rbf"):
                raise ValueError("Correction search used a different raw alpha than the selected backbone")
        rows.append(row)
        for column, horizon in enumerate(horizons):
            horizon_loss = float(squared_error[:, column].mean())
            horizon_rows.append({"task_id": task["task_id"], "method": method["method"],
                                 "horizon": horizon, "mse": horizon_loss, "scaled_mse": horizon_loss / scale})
        if episode_groups is not None:
            losses = np.mean(squared_error, axis=1)
            for group in np.unique(episode_groups):
                episode_rows.append({"task_id": task["task_id"], "method": method["method"], "episode": int(group),
                                     "mse": float(losses[episode_groups == group].mean())})
    for method in record["methods"]:
        if method["status"] == "ok" and method["method"].startswith("add_") and method["gamma"] == 0:
            if not np.array_equal(predictions[method["method"]], predictions["raw_rbf"]):
                raise ValueError("Gamma-zero prediction differs from the actual raw baseline")
    record_by_name = {row["method"]: row for row in record["methods"]}
    for name, policy in record["policies"].items():
        if policy["status"] != "ok":
            continue
        selected = policy["selected_method"]
        if name == "utility_kahm_5pct":
            expected = utility_choice(record_by_name["raw_rbf"]["validation_mse"], record_by_name["add_kahm"]["validation_mse"], cfg["utility_relative_validation_improvement"])
            selection_methods = ["raw_rbf", "add_kahm"]
        elif name == "utility_geometry_5pct":
            selection_methods = ["raw_rbf", "add_kahm", "add_affine", "add_locality"]
            expected = geometry_choice({method: record_by_name[method]["validation_mse"]
                                        for method in selection_methods}, cfg["utility_relative_validation_improvement"])
        else:
            raise ValueError("Undeclared policy")
        if selected != expected:
            raise ValueError("Selection rule differs from frozen validation-only rule")
        if policy["selection_methods"] != selection_methods:
            raise ValueError("Selector candidate set differs from the frozen policy")
        expected_nodes = {"data_preparation", f"{name}:decision"}
        for method in selection_methods:
            expected_nodes |= search_dependencies(method)
        expected_nodes |= final_dependencies(selected, task["phase"], record_by_name[selected].get("gamma"))
        audit_cost(policy, expected_nodes, record["nodes"])
        selected_row = next(row for row in rows if row["method"] == selected)
        if (policy["test_mse"] != selected_row["test_mse"]
                or policy["serialized_model_bytes"] != selected_row["serialized_model_bytes"]):
            raise ValueError("Selector outcome differs from the selected fitted method")
        rows.append({**selected_row, "method": name, "selected_method": selected,
                     "pipeline_seconds": policy["pipeline_seconds"],
                     "model_pipeline_seconds": policy["model_pipeline_seconds"],
                     "model_cpu_seconds": policy["model_cpu_seconds"], "alpha": None, "gamma": None})
        episode_rows.extend([{**row, "method": name} for row in episode_rows if row["method"] == selected])
        horizon_rows.extend([{**row, "method": name} for row in horizon_rows if row["method"] == selected])
    return rows, trials, episode_rows, horizon_rows


def summarize(run_dir):
    run_dir = Path(run_dir)
    frozen = verify_freeze()
    cfg = data.config()
    execution = json.loads((run_dir / "execution.json").read_text())
    if execution["freeze_sha256"] != data.digest(FREEZE) or execution["freeze"] != frozen:
        raise ValueError("Execution snapshot differs from the verified source freeze")
    expected = task_roster(cfg)
    if execution["expected_tasks"] != expected:
        raise ValueError("Execution task roster differs from the frozen design and prepared roster")
    expected_ids = {task["task_id"] for task in expected}
    if len(expected_ids) != len(expected):
        raise ValueError("Duplicate tasks in the frozen roster")
    found_ids = {path.name for path in (run_dir / "tasks").iterdir() if path.is_dir()} if (run_dir / "tasks").exists() else set()
    if found_ids - expected_ids:
        raise ValueError("Unexpected task directories are present in the run")
    results, failures, missing = [], [], []
    rows, trials, episode_rows, horizon_rows, nodes = [], [], [], [], []
    for task in execution["expected_tasks"]:
        directory = run_dir / "tasks" / task["task_id"]
        path = directory / "result.json"
        if not path.exists():
            missing.append(task["task_id"])
            continue
        record = json.loads(path.read_text())
        if record["task"] != task:
            raise ValueError("Recorded task fields differ from the frozen task roster")
        results.append(record)
        if not record["complete"]:
            failures.append({"task_id": task["task_id"], "fatal_error": record.get("fatal_error"),
                             "failed_methods": [m for m in record["methods"] if m["status"] != "ok"]})
        if not record["methods"]:
            if record["complete"]:
                raise ValueError("A complete task cannot omit its methods")
            continue
        new_rows, new_trials, new_episodes, new_horizons = audit_task(record, directory, cfg)
        rows.extend(new_rows)
        trials.extend(new_trials)
        episode_rows.extend(new_episodes)
        horizon_rows.extend(new_horizons)
        nodes.extend({"task_id": task["task_id"], **node} for node in record["nodes"])
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(output / "task_methods.csv", rows)
    write_csv(output / "validation_trials.csv", trials)
    write_csv(output / "episode_losses.csv", episode_rows)
    write_csv(output / "horizon_losses.csv", horizon_rows)
    decomposition_columns = ["task_id", "phase", "source_id", "series_id", "law", "training_episodes", "landmarks", "seed", "method", "gamma", "raw_mse", "test_mse", "correction_alignment", "correction_energy", "raw_minus_additive_mse", "identity_residual", "scaled_alignment", "scaled_correction_energy", "scaled_raw_minus_additive_mse"]
    write_csv(output / "correction_decomposition.csv", [
        {key: row[key] for key in decomposition_columns} for row in rows
        if row["method"].startswith("add_")
    ])
    write_csv(output / "timing_nodes.csv", nodes)
    summary = {"expected_tasks": len(execution["expected_tasks"]), "observed_tasks": len(results),
               "complete": not failures and not missing, "failed_tasks": failures, "missing_tasks": missing,
               "execution_freeze_sha256": execution["freeze_sha256"],
               "prediction_audit": "Verified frozen task/method/policy rosters and saved target/group/origin/prediction hashes; final losses, all saved-grid selections, zero-gamma equality, selector choices and cost unions checked. Validation trial losses are recorded, not independently re-evaluated from candidate predictions.",
               "archive_source_scope": "Four newly reserved Monash source collections chosen for feasibility, including NREL aggregated simulated Solar Weekly; not four independent random observational domains.",
               "cost_scope": "Union of measured required computation nodes, shared nodes counted once; selector costs are charged component sums, not separately timed full-policy executions. model_pipeline_seconds excludes data loading/simulation and output-artifact I/O; pipeline_seconds also includes data preparation. Model serialization remains included and is a separate node.",
               "decomposition_scope": "Descriptive held-out squared-error identity: raw loss minus additive loss = twice residual/correction alignment minus correction energy; not a population risk estimate or selection rule."}
    if failures or missing:
        data.write_json(output / "summary.json", summary)
        return summary
    df = pd.DataFrame(rows)
    numeric = ["test_mse", "scaled_mse", "pipeline_seconds", "model_pipeline_seconds", "model_cpu_seconds", "serialized_model_bytes"]
    synthetic = df[df.phase == "synthetic"]
    cell_keys = ["law", "training_episodes", "landmarks", "method"]
    syn_summary = synthetic.groupby(cell_keys, sort=False)[numeric].mean().reset_index()
    syn_summary.to_csv(output / "synthetic_methods.csv", index=False)
    contrasts = []
    for (law, count, landmarks), group in synthetic.groupby(["law", "training_episodes", "landmarks"], sort=False):
        table = group.pivot(index="seed", columns="method", values="test_mse")
        if len(table) != len(cfg["synthetic"]["repetition_seeds"]):
            raise ValueError("Incomplete paired repetitions")
        for control in ("raw_rbf", "concat_kahm"):
            difference = table[control] - table["add_kahm"]
            contrasts.append({"law": law, "training_episodes": int(count), "landmarks": int(landmarks),
                              "control": control, "additive_mean_mse": float(table["add_kahm"].mean()),
                              "control_mean_mse": float(table[control].mean()), **paired_interval(difference, family_size=24)})
    write_csv(output / "synthetic_primary_contrasts.csv", contrasts)
    # Equal capacity weights within each series, equal series weights within a
    # source, then equal source weights. No time window is an inference unit.
    archive = df[df.phase == "natural"]
    series = archive.groupby(["source_id", "series_id", "method"], sort=False)[numeric].mean().reset_index()
    sources = series.groupby(["source_id", "method"], sort=False)[numeric].mean().reset_index()
    natural = sources.groupby("method", sort=False)[numeric].mean().reset_index()
    sources.to_csv(output / "archive_sources.csv", index=False)
    series.to_csv(output / "archive_series.csv", index=False)
    natural.to_csv(output / "archive_methods.csv", index=False)
    source_pivot = sources.pivot(index="source_id", columns="method", values="scaled_mse")
    leave_one_out = []
    for omitted in source_pivot.index:
        rest = source_pivot.drop(index=omitted).mean()
        leave_one_out.append({"omitted_source": omitted, "raw_rbf": rest["raw_rbf"], "add_kahm": rest["add_kahm"],
                              "relative_change_pct": 100 * (rest["add_kahm"] / rest["raw_rbf"] - 1)})
    write_csv(output / "archive_leave_one_source_out.csv", leave_one_out)
    gamma = df[df.method.str.startswith("add_")].groupby(["phase", "method", "gamma"], dropna=False).size().reset_index(name="tasks")
    gamma.to_csv(output / "gamma_choices.csv", index=False)
    policy = df[df.method.isin(POLICIES)].groupby(["phase", "method", "selected_method"]).size().reset_index(name="tasks")
    policy.to_csv(output / "policy_choices.csv", index=False)
    summary.update(synthetic_methods=syn_summary.to_dict(orient="records"), synthetic_primary_contrasts=contrasts,
                   archive_methods=natural.to_dict(orient="records"), archive_sources=sources.to_dict(orient="records"),
                   archive_leave_one_source_out=leave_one_out, gamma_choices=gamma.to_dict(orient="records"),
                   policy_choices=policy.to_dict(orient="records"),
                   warnings_count=sum(len(record["warnings"]) for record in results),
                   synthetic_inference="24 paired-Student contrasts, 12 fresh complete-fit repetitions each; Bonferroni simultaneous95% under paired-difference normality or approximation. Capacity/sample-size cells share data within repetitions; no iid claim across cells.",
                   archive_inference="Descriptive equal-source summaries only; overlapping windows, repeated capacities, and selected series are not independent sources.")
    data.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "utility_v1")
    args = parser.parse_args()
    result = summarize(args.run_dir)
    print(json.dumps({key: result[key] for key in ("expected_tasks", "observed_tasks", "complete", "prediction_audit")}, indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
