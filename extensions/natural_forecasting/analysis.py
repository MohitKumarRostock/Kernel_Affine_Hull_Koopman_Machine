"""Predeclared, source-family-aware analysis of natural forecasting experiments.

This module never fits a forecasting model. Candidate configurations are selected
on chronological validation data by the runner; a later evaluator adds their
future-test losses. Development test losses may train a small diagnostic across
tasks. Confirmation losses cannot train, tune, standardize or select diagnostics.

The independent descriptive grouping is the original source family, not a
window, horizon, series or fitting seed. Four development/confirmation families
cannot justify a broad-population meta-learning claim; no window/seed-level
p-values or confidence intervals are produced here.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


VERSION = "natural_forecasting_analysis_v1"
GROUP_KEYS = ("source_family", "dataset_id", "series_id", "horizon",
              "history_length", "budget_id", "seed")
METADATA_FEATURES = ("log_n_train_windows", "log_horizon_ratio", "lag1_autocorrelation")
GEOMETRY_FEATURES = ("kahm_gate_entropy", "log_affine_projection_residual")
MODEL_FEATURES = {
    "metadata": METADATA_FEATURES,
    "metadata_geometry": METADATA_FEATURES + GEOMETRY_FEATURES,
    "metadata_probe": METADATA_FEATURES + ("probe_log_ratio",),
    "metadata_probe_geometry": METADATA_FEATURES + ("probe_log_ratio",) + GEOMETRY_FEATURES,
}
DEFAULT_CANDIDATES = (
    "raw_ridge", "raw_rbf", "affine_rbf", "locality_rbf", "kahm_rbf",
    "raw_affine_rbf", "raw_locality_rbf", "raw_kahm_rbf", "seasonal_naive",
    "holt", "hist_gradient", "mlp",
)
DEFAULT_CONFIG = {
    "raw_candidate": "raw_rbf", "kahm_candidate": "raw_kahm_rbf",
    "raw_portfolio": ["raw_ridge", "raw_rbf", "seasonal_naive", "holt", "hist_gradient", "mlp"],
    "kahm_portfolio": ["kahm_rbf", "raw_kahm_rbf"],
    "full_portfolio": list(DEFAULT_CANDIDATES),
    "ridge_penalty": 1.0, "loss_floor": 1e-12,
    "practical_relative_margin": 0.05,
    "minimum_development_families": 4,
    "reserved_source_families": [],
    "expected_development_task_ids": [], "expected_confirmation_task_ids": [],
}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"Invalid {name}: {value!r}")
    return result


def _integer(value, name):
    result = _number(value, name)
    if result != int(result):
        raise ValueError(f"{name} must be an integer")
    return int(result)


def _config(config=None):
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown analysis configuration keys: {sorted(unknown)}")
    result = json.loads(json.dumps(DEFAULT_CONFIG))
    result.update(supplied)
    _number(result["ridge_penalty"], "ridge_penalty", positive=True)
    _number(result["loss_floor"], "loss_floor", positive=True)
    margin = _number(result["practical_relative_margin"], "practical_relative_margin", positive=True)
    if margin >= 1:
        raise ValueError("The practical relative margin must be below one")
    if _integer(result["minimum_development_families"], "minimum_development_families") < 3:
        raise ValueError("At least three development families are required")
    for key in ("raw_portfolio", "kahm_portfolio", "full_portfolio", "reserved_source_families",
                "expected_development_task_ids", "expected_confirmation_task_ids"):
        values = result[key]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"{key} must be a list with no duplicate values")
    if result["raw_candidate"] == result["kahm_candidate"]:
        raise ValueError("The matched raw and KAHM candidates must be distinct")
    if not set(result["raw_portfolio"] + result["kahm_portfolio"] +
               [result["raw_candidate"], result["kahm_candidate"]]).issubset(result["full_portfolio"]):
        raise ValueError("All compared candidates must be included in full_portfolio")
    return result


def prespecified_spec(config=None):
    """Outcome-free specification: record its hash before development results."""
    return {
        "format": VERSION, "config": _config(config),
        "models": {name: list(features) for name, features in MODEL_FEATURES.items()},
        "primary_diagnostic": "metadata_probe_geometry",
        "primary_diagnostic_control": "metadata_probe",
        "diagnostic_target": "log((raw_test_mse/loss_scale + loss_floor) / (kahm_test_mse/loss_scale + loss_floor))",
        "decision": "Choose KAHM only when predicted log benefit is strictly positive; raw on ties.",
        "secondary_benefit": "At least the fixed practical relative MSE reduction, with the declared loss floor near zero.",
        "feature_standardization": "Weighted development mean and population standard deviation, learned within each source-family fold; constant scale replaced by one.",
        "ridge_objective": "sum_i weight_i*(target_i-intercept-standardized_features_i@coef)^2 + ridge_penalty*||coef||^2; intercept unpenalized",
        "weighting_hierarchy": list(GROUP_KEYS),
        "weights": "Equal source-family weight, then equal weight at every nested level; no duplicated task keys.",
        "costs": "Sum unique declared required cost nodes; validation selectors pay all searched candidates but refit/forecast only the chosen one; learned/probe selectors pay their diagnostic/probe nodes even when KAHM is rejected.",
        "inference": "Descriptive family-level outcomes, between-family dispersion and leave-one-family-out sensitivity only; no population coverage or significance claim.",
        "selection": "All four predeclared diagnostic models are frozen and reported. No best diagnostic is selected using confirmation outcomes.",
    }


def write_prespecified_spec(path, config=None):
    spec = prespecified_spec(config)
    value = {"spec": spec, "spec_sha256": _digest(spec), "analysis_code_sha256": code_hash(),
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Recorded internal analysis specification before development outcomes; not external preregistration."}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def _nodes(task):
    values = task["cost_nodes"]
    if not isinstance(values, list):
        raise ValueError("cost_nodes must be a list")
    result = {}
    for value in values:
        key = value["id"]
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("Cost-node IDs must be distinct nonempty strings")
        _number(value["seconds"], f"cost node {key} seconds", nonnegative=True)
        if "cpu_seconds" in value and "process_cpu_seconds" not in value:
            raise ValueError("Cost-node CPU uses process_cpu_seconds; cpu_seconds would be silently uncounted")
        for field in ("process_cpu_seconds", "peak_rss_bytes"):
            if value.get(field) is not None:
                _number(value[field], f"cost node {key} {field}", nonnegative=True)
        result[key] = value
    return result


def _node_ids(values, nodes, name):
    if not isinstance(values, list) or len(values) != len(set(values)):
        raise ValueError(f"{name} must list distinct cost-node IDs")
    if not set(values).issubset(nodes):
        raise ValueError(f"Unknown cost nodes in {name}: {sorted(set(values)-set(nodes))}")
    return set(values)


def _candidates(task):
    result = {}
    for row in task["candidates"]:
        name = row["candidate_id"]
        if name in result:
            raise ValueError(f"Duplicate candidate in {task['task_id']}: {name}")
        result[name] = row
    return result


def validate_tasks(tasks, config=None, *, role="development", allow_confirmation=False,
                   check_roster=True):
    """Validate chronology, completeness, losses and the declared cost graph.

    The confirmation gate is checked before accessing candidate test losses.
    Indices are zero-based; split ends are exclusive, recorded maxima inclusive.
    """
    config = _config(config)
    tasks = list(tasks)
    if not tasks:
        raise ValueError("No tasks were supplied")
    if role not in {"development", "confirmation"}:
        raise ValueError("Unknown evaluation role")
    if role == "confirmation" and not allow_confirmation:
        raise ValueError("Confirmation outcome access requires explicit allow_confirmation=True after diagnostic freezing")
    if any(task.get("phase") != role for task in tasks):
        raise ValueError("Tasks from another phase cannot enter this analysis")
    ids, keys = set(), set()
    for task in tasks:
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError("task_id must be distinct and nonempty")
        ids.add(task_id)
        key = tuple(task[field] for field in GROUP_KEYS)
        if key in keys:
            raise ValueError("Duplicated source/series/horizon/history/budget/seed task")
        keys.add(key)
        for field in ("source_family", "dataset_id", "series_id", "budget_id"):
            if not str(task[field]):
                raise ValueError(f"Empty task field: {field}")
        for field in ("horizon", "history_length"):
            if _integer(task[field], field) < 1:
                raise ValueError(f"{field} must be positive")
        scale = _number(task["loss_scale"], "loss_scale", positive=True)
        split = task["split"]
        train_end, val_start, val_end, test_start, test_end = (
            _integer(split[field], field) for field in
            ("train_end_exclusive", "validation_start", "validation_end_exclusive", "test_start", "test_end_exclusive"))
        if not (0 < train_end <= val_start < val_end <= test_start < test_end):
            raise ValueError(f"Nonchronological split for {task_id}")
        provenance = task["feature_provenance"]
        for field in ("metadata_max_observation_index", "geometry_max_observation_index"):
            if not (0 <= _integer(provenance[field], field) < train_end):
                raise ValueError(f"Training-only diagnostic feature uses later observations: {task_id}/{field}")
        features = task["features"]
        if _integer(features["n_train_windows"], "n_train_windows") < 1:
            raise ValueError("n_train_windows must be a positive integer")
        acf = _number(features["lag1_autocorrelation"], "lag1_autocorrelation")
        entropy = _number(features["kahm_gate_entropy"], "kahm_gate_entropy")
        if not -1.00000001 <= acf <= 1.00000001 or not -1e-10 <= entropy <= 1.00000001:
            raise ValueError("Autocorrelation or normalized gate entropy is outside its declared range")
        _number(features["affine_projection_residual"], "affine_projection_residual", nonnegative=True)
        nodes = _nodes(task)
        for field in ("metadata", "geometry"):
            if not _node_ids(task["feature_cost_node_ids"][field], nodes, f"{field} features"):
                raise ValueError("Diagnostic feature extraction requires explicit cost nodes")
        probe = task["probe"]
        for field in ("raw_validation_loss", "kahm_validation_loss"):
            _number(probe[field], field, nonnegative=True)
        if not _node_ids(probe["cost_node_ids"], nodes, "probe"):
            raise ValueError("Probe execution requires explicit cost nodes")
        if not 0 <= _integer(probe["tuning_fit_max_target_index"], "probe tuning maximum") < train_end:
            raise ValueError("Probe fitting crosses its training boundary")
        if not val_start <= _integer(probe["selection_max_target_index"], "probe selection maximum") < val_end:
            raise ValueError("Probe selection uses outcomes outside validation")
        candidates = _candidates(task)
        if set(candidates) != set(config["full_portfolio"]):
            raise ValueError(f"Incomplete or changed candidate roster for {task_id}")
        for name, row in candidates.items():
            _number(row["validation_loss"], f"{name} validation loss", nonnegative=True)
            _number(row["test_loss"], f"{name} test loss", nonnegative=True)
            selection = _node_ids(row["selection_cost_node_ids"], nodes, name + " selection")
            final = _node_ids(row["final_cost_node_ids"], nodes, name + " final")
            if not selection or not final:
                raise ValueError("Candidate search and final deployment costs must be explicit")
            if "cost_node_ids" in row and _node_ids(row["cost_node_ids"], nodes, name) != selection | final:
                raise ValueError("Candidate total nodes differ from selection/final union")
            if not 0 <= _integer(row["tuning_fit_max_target_index"], name + " tuning maximum") < train_end:
                raise ValueError("Hyperparameter fitting uses observations beyond training")
            if not val_start <= _integer(row["selection_max_target_index"], name + " selection maximum") < val_end:
                raise ValueError("Candidate model selection uses outcomes outside validation")
            if not 0 <= _integer(row["refit_max_target_index"], name + " refit maximum") < test_start:
                raise ValueError("Final model fitting uses future test outcomes")
            if not test_start <= _integer(row["test_min_target_index"], name + " test minimum") <= _integer(row["test_max_target_index"], name + " test maximum") < test_end:
                raise ValueError("Reported evaluation targets cross the test boundary")
            if not math.isfinite(float(row["test_loss"]) / scale):
                raise ValueError("Scaled test loss is nonfinite")
    if check_roster:
        expected = set(config[f"expected_{role}_task_ids"])
        if expected and expected != ids:
            raise ValueError(f"{role} task roster differs: missing={sorted(expected-ids)}, unexpected={sorted(ids-expected)}")
    families = {task["source_family"] for task in tasks}
    reserved = set(config["reserved_source_families"])
    if role == "development" and families & reserved:
        raise ValueError("Reserved source family appeared in development")
    if role == "confirmation" and families != reserved:
        raise ValueError("Confirmation source families differ from the frozen reserved set")
    return tasks


def hierarchical_weights(rows, keys=GROUP_KEYS):
    """Each original source contributes equally regardless of its row count."""
    if not rows:
        raise ValueError("Cannot weight an empty collection")
    weights = np.zeros(len(rows), dtype=float)

    def assign(indices, depth, mass):
        if depth == len(keys):
            weights[indices] = mass / len(indices)
            return
        groups = defaultdict(list)
        for index in indices:
            groups[rows[index][keys[depth]]].append(index)
        for group in groups.values():
            assign(group, depth + 1, mass / len(groups))

    assign(list(range(len(rows))), 0, 1.0)
    return weights


def feature_values(task, config=None, names=None):
    config = _config(config)
    requested = set(names if names is not None else METADATA_FEATURES + GEOMETRY_FEATURES + ("probe_log_ratio",))
    values, result = task["features"], {}
    if "log_n_train_windows" in requested:
        result["log_n_train_windows"] = math.log1p(float(values["n_train_windows"]))
    if "log_horizon_ratio" in requested:
        result["log_horizon_ratio"] = math.log1p(float(task["horizon"]) / float(task["history_length"]))
    if "lag1_autocorrelation" in requested:
        result["lag1_autocorrelation"] = float(values["lag1_autocorrelation"])
    if "kahm_gate_entropy" in requested:
        result["kahm_gate_entropy"] = float(values["kahm_gate_entropy"])
    if "log_affine_projection_residual" in requested:
        result["log_affine_projection_residual"] = math.log1p(float(values["affine_projection_residual"]))
    if "probe_log_ratio" in requested:
        probe, eps, scale = task["probe"], config["loss_floor"], float(task["loss_scale"])
        result["probe_log_ratio"] = math.log((float(probe["raw_validation_loss"]) / scale + eps) /
                                             (float(probe["kahm_validation_loss"]) / scale + eps))
    if set(result) != requested:
        raise ValueError("An undeclared diagnostic feature was requested")
    return result


def _target(task, config):
    candidates = _candidates(task)
    raw = float(candidates[config["raw_candidate"]]["test_loss"]) / task["loss_scale"]
    kahm = float(candidates[config["kahm_candidate"]]["test_loss"]) / task["loss_scale"]
    return math.log((raw + config["loss_floor"]) / (kahm + config["loss_floor"]))


def fit_diagnostic(tasks, feature_names, config=None):
    """Low-complexity weighted ridge; call only on allowed development folds."""
    started, cpu_started = time.perf_counter(), time.process_time()
    config = _config(config)
    if not tasks or any(task["phase"] != "development" for task in tasks):
        raise ValueError("Diagnostic fitting accepts development tasks only")
    feature_rows = [feature_values(task, config, feature_names) for task in tasks]
    x = np.asarray([[row[name] for name in feature_names] for row in feature_rows], dtype=float)
    y = np.asarray([_target(task, config) for task in tasks], dtype=float)
    w = hierarchical_weights(tasks)
    mean = w @ x
    scale = np.sqrt(w @ ((x - mean) ** 2))
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    intercept = float(w @ y)
    coef = np.linalg.solve(z.T @ (w[:, None] * z) + config["ridge_penalty"] * np.eye(z.shape[1]),
                           z.T @ (w * (y - intercept)))
    return {"features": list(feature_names), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": coef.tolist(), "intercept": intercept,
            "ridge_penalty": config["ridge_penalty"],
            "training_source_families": sorted({task["source_family"] for task in tasks}),
            "training_task_ids": sorted(task["task_id"] for task in tasks),
            "offline_fit_wall_seconds": time.perf_counter() - started,
            "offline_fit_cpu_seconds": time.process_time() - cpu_started,
            "fit_cost_scope": "One diagnostic variant: feature assembly, source weighting, development-only standardization and ridge fitting. Forecasting/probe data acquisition costs belong to the recorded task nodes; no amortization assumed."}


def predict_diagnostic(task, model, config=None):
    values = feature_values(task, config, model["features"])
    x = np.asarray([values[name] for name in model["features"]], dtype=float)
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    result = float(model["intercept"] + z @ np.asarray(model["coefficients"]))
    if not math.isfinite(result):
        raise ValueError("Diagnostic prediction is nonfinite")
    return result


def cost_summary(task, required_ids):
    nodes = _nodes(task)
    required = set(required_ids)
    if not required.issubset(nodes):
        raise ValueError("A policy references unknown cost nodes")
    selected = [nodes[key] for key in sorted(required)]
    cpu_known = all(row.get("process_cpu_seconds") is not None for row in selected)
    rss_known = all(row.get("peak_rss_bytes") is not None for row in selected)
    return {
        "required_cost_node_ids": sorted(required),
        "measured_node_wall_seconds": sum(float(row["seconds"]) for row in selected),
        "measured_node_cpu_seconds": sum(float(row["process_cpu_seconds"]) for row in selected) if cpu_known else None,
        "maximum_measured_node_peak_rss_bytes": max((float(row["peak_rss_bytes"]) for row in selected), default=0.0) if rss_known else None,
        "accounting": "Counterfactual scientific-pipeline cost from measured dependencies; unique nodes are charged once. RSS is the maximum recorded process high-water mark at required nodes, including interpreter/input and earlier-process work, not incremental memory or exact counterfactual live memory.",
    }


def _validation_choice(candidates, ordered_ids):
    # Input order, frozen in the specification, determines ties.
    return min(ordered_ids, key=lambda name: float(candidates[name]["validation_loss"]))


def policy_rows(task, models, config=None):
    config = _config(config)
    candidates = _candidates(task)
    raw_id, kahm_id = config["raw_candidate"], config["kahm_candidate"]
    scale = float(task["loss_scale"])
    raw_loss = float(candidates[raw_id]["test_loss"]) / scale
    kahm_loss = float(candidates[kahm_id]["test_loss"]) / scale
    full_oracle = min(float(row["test_loss"]) / scale for row in candidates.values())
    threshold = config["practical_relative_margin"] * max(raw_loss, config["loss_floor"])
    rows = []

    def emit(name, chosen, required, prediction=None, decision_wall=0.0, decision_cpu=0.0):
        selected = candidates[chosen]
        loss = float(selected["test_loss"]) / scale
        cost = cost_summary(task, required)
        cpu = cost["measured_node_cpu_seconds"]
        rows.append({
            **{key: task[key] for key in ("task_id", "phase") + GROUP_KEYS},
            "policy": name, "chosen_candidate": chosen,
            "test_mse": float(selected["test_loss"]), "scaled_test_mse": loss,
            "serialized_model_bytes": selected.get("serialized_model_bytes"),
            "raw_scaled_test_mse": raw_loss, "kahm_scaled_test_mse": kahm_loss,
            "benefit_vs_always_raw": raw_loss - loss,
            "harm_vs_always_raw": float(loss > raw_loss),
            "practical_harm_vs_always_raw": float(loss - raw_loss > threshold),
            "practical_benefit_vs_always_raw": float(raw_loss - loss > threshold),
            "oracle_excess_scaled_loss": loss - full_oracle,
            "oracle_scope": "Best listed candidate on this task's test data; hindsight diagnostic, not an available selector.",
            "observed_pair_log_benefit": _target(task, config),
            "predicted_pair_log_benefit": prediction,
            "selected_kahm_pair_member": float(chosen == kahm_id),
            **cost,
            "decision_wall_seconds": decision_wall,
            "decision_cpu_seconds": decision_cpu,
            "total_policy_wall_seconds": cost["measured_node_wall_seconds"] + decision_wall,
            "total_policy_cpu_seconds": cpu + decision_cpu if cpu is not None else None,
        })

    def complete(name):
        row = candidates[name]
        return set(row["selection_cost_node_ids"]) | set(row["final_cost_node_ids"])

    emit("always_raw", raw_id, complete(raw_id))
    emit("always_kahm", kahm_id, complete(kahm_id))
    for candidate_id in config["full_portfolio"]:
        emit("standalone_" + candidate_id, candidate_id, complete(candidate_id))
    for name, roster in (("validation_pair", [raw_id, kahm_id]),
                         ("validation_raw_portfolio", config["raw_portfolio"]),
                         ("validation_kahm_family", config["kahm_portfolio"]),
                         ("validation_full_portfolio", config["full_portfolio"])):
        start, cpu = time.perf_counter(), time.process_time()
        chosen = _validation_choice(candidates, roster)
        elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
        required = set().union(*(set(candidates[key]["selection_cost_node_ids"]) for key in roster))
        required |= set(candidates[chosen]["final_cost_node_ids"])
        emit(name, chosen, required, decision_wall=elapsed, decision_cpu=cpu_elapsed)
    start, cpu = time.perf_counter(), time.process_time()
    chosen = kahm_id if task["probe"]["kahm_validation_loss"] < task["probe"]["raw_validation_loss"] else raw_id
    elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
    emit("cheap_probe", chosen, complete(chosen) | set(task["probe"]["cost_node_ids"]),
         decision_wall=elapsed, decision_cpu=cpu_elapsed)
    for name, model in models.items():
        start, cpu = time.perf_counter(), time.process_time()
        prediction = predict_diagnostic(task, model, config)
        chosen = kahm_id if prediction > 0 else raw_id
        elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
        required = complete(chosen) | set(task["feature_cost_node_ids"]["metadata"])
        if any(feature in GEOMETRY_FEATURES for feature in model["features"]):
            required |= set(task["feature_cost_node_ids"]["geometry"])
        if "probe_log_ratio" in model["features"]:
            required |= set(task["probe"]["cost_node_ids"])
        emit(name, chosen, required, prediction, elapsed, cpu_elapsed)
    return rows


SUMMARY_FIELDS = (
    "scaled_test_mse", "benefit_vs_always_raw", "harm_vs_always_raw",
    "practical_harm_vs_always_raw", "practical_benefit_vs_always_raw",
    "oracle_excess_scaled_loss", "total_policy_wall_seconds", "total_policy_cpu_seconds",
    "maximum_measured_node_peak_rss_bytes", "selected_kahm_pair_member",
    "decision_wall_seconds", "decision_cpu_seconds", "serialized_model_bytes",
)


def _weighted_metrics(rows, fields=SUMMARY_FIELDS):
    weights = hierarchical_weights(rows)
    return {field: float(weights @ np.asarray([row[field] for row in rows]))
            if all(row.get(field) is not None for row in rows) else None for field in fields}


def summarize_policies(rows):
    policies = defaultdict(list)
    for row in rows:
        policies[row["policy"]].append(row)
    report = {}
    for name, group in sorted(policies.items()):
        families = sorted({row["source_family"] for row in group})
        by_family = {family: _weighted_metrics([row for row in group if row["source_family"] == family]) for family in families}
        family_losses = np.asarray([by_family[family]["scaled_test_mse"] for family in families])
        report[name] = {
            "source_families": len(families), "tasks": len(group),
            "equal_source_family_metrics": _weighted_metrics(group),
            "by_source_family": by_family,
            "by_budget": {budget: _weighted_metrics([row for row in group if row["budget_id"] == budget])
                          for budget in sorted({row["budget_id"] for row in group})},
            "descriptive_between_family_loss_sd": float(np.std(family_losses, ddof=1)) if len(families) > 1 else None,
            "family_loss_range": [float(family_losses.min()), float(family_losses.max())],
            "leave_one_family_out_mean_loss_range": [float(min(np.delete(family_losses, index).mean() for index in range(len(families)))),
                                                      float(max(np.delete(family_losses, index).mean() for index in range(len(families))))] if len(families) > 1 else None,
            "uncertainty_scope": "Descriptive variation over the observed original source families, not a confidence interval for unseen datasets or windows.",
        }
    return report


def calibration(rows, config=None):
    config = _config(config)
    result = {}
    for name in MODEL_FEATURES:
        selected = [row for row in rows if row["policy"] == name]
        if not selected:
            continue
        w = hierarchical_weights(selected)
        predicted = np.asarray([row["predicted_pair_log_benefit"] for row in selected])
        observed = np.asarray([row["observed_pair_log_benefit"] for row in selected])
        threshold = -math.log1p(-config["practical_relative_margin"])
        result[name] = {
            "weighted_log_benefit_rmse": float(np.sqrt(w @ ((predicted - observed) ** 2))),
            "weighted_log_benefit_bias": float(w @ (predicted - observed)),
            "weighted_sign_accuracy": float(w @ ((predicted > 0) == (observed > 0))),
            "weighted_practical_benefit_classification_accuracy": float(w @ ((predicted > threshold) == (observed > threshold))),
            "by_source_family": {},
            "scope": "Predeclared descriptive estimated-benefit checks; no recalibration is fitted on held-out outcomes.",
        }
        for family in sorted({row["source_family"] for row in selected}):
            group = [row for row in selected if row["source_family"] == family]
            weights = hierarchical_weights(group)
            result[name]["by_source_family"][family] = {
                "mean_predicted_log_benefit": float(weights @ np.asarray([row["predicted_pair_log_benefit"] for row in group])),
                "mean_observed_log_benefit": float(weights @ np.asarray([row["observed_pair_log_benefit"] for row in group])),
            }
    return result


def paired_policy_comparisons(rows):
    by_task = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["policy"]] = row
    comparisons = (
        ("metadata_probe_geometry", "metadata_probe"),
        ("metadata_geometry", "metadata"),
        ("metadata_probe_geometry", "always_raw"),
        ("metadata_probe_geometry", "always_kahm"),
        ("metadata_probe_geometry", "validation_pair"),
        ("metadata_probe_geometry", "cheap_probe"),
        ("metadata_probe_geometry", "validation_raw_portfolio"),
        ("metadata_probe_geometry", "validation_kahm_family"),
        ("metadata_probe_geometry", "validation_full_portfolio"),
    )
    output = {}
    for candidate, control in comparisons:
        paired = []
        for _, task in sorted(by_task.items()):
            if candidate not in task or control not in task:
                continue
            left, right = task[candidate], task[control]
            paired.append({**{key: left[key] for key in GROUP_KEYS},
                           "loss_advantage": right["scaled_test_mse"] - left["scaled_test_mse"],
                           "additional_wall_seconds": left["total_policy_wall_seconds"] - right["total_policy_wall_seconds"]})
        if not paired:
            continue
        fields = ("loss_advantage", "additional_wall_seconds")
        families = sorted({row["source_family"] for row in paired})
        by_family = {family: _weighted_metrics([row for row in paired if row["source_family"] == family], fields)
                     for family in families}
        values = [by_family[family]["loss_advantage"] for family in families]
        output[candidate + "_vs_" + control] = {
            "equal_source_family_metrics": _weighted_metrics(paired, fields),
            "by_source_family": by_family,
            "families_favoring_candidate": sum(value > 0 for value in values),
            "families_favoring_control": sum(value < 0 for value in values),
            "source_families": len(families),
            "interpretation": "Positive loss advantage favors candidate. Source-family counts and means are descriptive, not significance tests.",
        }
    return output


def _report(rows, scope, config, extra=None):
    return {
        "format": VERSION, "scope": scope, "spec_sha256": _digest(prespecified_spec(config)),
        "policies": summarize_policies(rows), "calibration": calibration(rows, config),
        "paired_policy_comparisons": paired_policy_comparisons(rows), "task_policy_rows": rows,
        "independence": "Original source families are the highest-level grouping. Horizons, contexts, budgets, seeds, series and overlapping windows are not independent replicates.",
        "generalization_limit": "The small deliberately chosen family collection supports bounded descriptive evidence, not a validated universal meta-learner or broad-population confidence claim.",
        "cost_scope": "Required recorded scientific-pipeline nodes plus measured scalar decision overhead. Offline diagnostic training costs are reported separately, not silently amortized over an assumed deployment count. Experimental process launch/IPC and shared library initialization are separate runner measurements rather than arbitrary per-selector allocations. Node RSS is a process high-water maximum including earlier process history, not incremental memory or an exact counterfactual live-memory measurement.",
        **(extra or {}),
    }


def analyze_development(tasks, config=None):
    config = _config(config)
    tasks = validate_tasks(tasks, config)
    families = sorted({task["source_family"] for task in tasks})
    if len(families) < config["minimum_development_families"]:
        raise ValueError("Too few development source families for the declared diagnostic analysis")
    rows, folds = [], []
    for held_out in families:
        training = [task for task in tasks if task["source_family"] != held_out]
        evaluation = [task for task in tasks if task["source_family"] == held_out]
        models = {name: fit_diagnostic(training, features, config) for name, features in MODEL_FEATURES.items()}
        folds.append({"held_out_source_family": held_out,
                      "training_source_families": sorted({task["source_family"] for task in training}),
                      "models": models})
        for task in evaluation:
            rows.extend(policy_rows(task, models, config))
    return _report(rows, "Leave-one-original-source-family-out development evaluation; all models/specifications fixed in advance.",
                   config, {"folds": folds})


def freeze_diagnostic(tasks, path=None, config=None):
    """Fit all declared rules on development and write a new immutable freeze."""
    config = _config(config)
    tasks = validate_tasks(tasks, config)
    families = sorted({task["source_family"] for task in tasks})
    if len(families) < config["minimum_development_families"]:
        raise ValueError("Too few development source families to freeze the diagnostic")
    if not config["reserved_source_families"] or not config["expected_confirmation_task_ids"]:
        raise ValueError("Reserve original source families and the complete confirmation task roster before freezing")
    start, cpu = time.perf_counter(), time.process_time()
    models = {name: fit_diagnostic(tasks, features, config) for name, features in MODEL_FEATURES.items()}
    fit_wall, fit_cpu = time.perf_counter() - start, time.process_time() - cpu
    fit_by_model = {name: {"wall_seconds": model["offline_fit_wall_seconds"],
                           "cpu_seconds": model["offline_fit_cpu_seconds"]}
                    for name, model in models.items()}
    frozen = {
        "format": "natural_forecasting_diagnostic_freeze_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_code_sha256": code_hash(), "spec": prespecified_spec(config),
        "spec_sha256": _digest(prespecified_spec(config)),
        "development_source_families": families,
        "development_task_ids": sorted(task["task_id"] for task in tasks),
        "development_input_sha256": _digest(sorted(tasks, key=lambda task: task["task_id"])),
        "development_label_sha256": _digest(sorted((task["task_id"], _target(task, config)) for task in tasks)),
        "models": models,
        "offline_fit_wall_seconds": fit_wall,
        "offline_fit_cpu_seconds": fit_cpu,
        "offline_fit_by_model": fit_by_model,
        "offline_fit_cost_scope": "One-off fitting of all four predeclared variants from the saved development records; no assumed deployment-count amortization. Per-task feature, probe, forecasting and scalar inference costs are reported in policy outcomes. Freeze serialization and hashing are excluded from model fitting time.",
        "scope": "Internal freeze after development and before reserved-family confirmation; no external timestamp attestation. All four fixed models are retained.",
    }
    frozen["freeze_sha256"] = _digest(frozen)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(frozen, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    return frozen


def verify_frozen(frozen):
    if isinstance(frozen, (str, Path)):
        frozen = json.loads(Path(frozen).read_text())
    value = dict(frozen)
    claimed = value.pop("freeze_sha256")
    if _digest(value) != claimed:
        raise ValueError("Diagnostic freeze bytes/fields changed")
    if frozen["analysis_code_sha256"] != code_hash():
        raise ValueError("Analysis implementation changed after the diagnostic freeze")
    if _digest(frozen["spec"]) != frozen["spec_sha256"]:
        raise ValueError("Frozen analysis specification changed")
    if frozen["spec"] != prespecified_spec(frozen["spec"]["config"]):
        raise ValueError("Frozen model/feature specification does not match the implementation")
    if set(frozen["models"]) != set(MODEL_FEATURES):
        raise ValueError("A predeclared diagnostic model is missing or extra")
    for name, features in MODEL_FEATURES.items():
        model = frozen["models"][name]
        if model["features"] != list(features):
            raise ValueError("Frozen diagnostic features changed")
        if model["training_source_families"] != frozen["development_source_families"]:
            raise ValueError("Frozen model training-source inventory differs")
        if model["training_task_ids"] != frozen["development_task_ids"]:
            raise ValueError("Frozen model training-task inventory differs")
    return frozen


def evaluate_frozen(tasks, frozen, *, allow_confirmation=False):
    # The explicit gate precedes even the source-roster/loss validation.
    if not allow_confirmation:
        raise ValueError("Confirmation evaluation requires explicit allow_confirmation=True")
    frozen = verify_frozen(frozen)
    config = frozen["spec"]["config"]
    tasks = validate_tasks(tasks, config, role="confirmation", allow_confirmation=True)
    if {task["source_family"] for task in tasks} & set(frozen["development_source_families"]):
        raise ValueError("A development source family appears in confirmation")
    if {task["task_id"] for task in tasks} & set(frozen["development_task_ids"]):
        raise ValueError("A development task appears in confirmation")
    rows = [row for task in tasks for row in policy_rows(task, frozen["models"], config)]
    return _report(rows, "Frozen diagnostic evaluation on the complete reserved original-source-family roster; no refitting or recalibration.",
                   config, {"freeze_sha256": frozen["freeze_sha256"],
                            "offline_diagnostic_fit_wall_seconds": frozen["offline_fit_wall_seconds"],
                            "offline_diagnostic_fit_cpu_seconds": frozen["offline_fit_cpu_seconds"],
                            "offline_diagnostic_fit_by_model": frozen["offline_fit_by_model"],
                            "offline_diagnostic_fit_cost_scope": frozen["offline_fit_cost_scope"]})


def audit_prediction_arrays(targets, predictions, reported_mse, *, reported_origin_losses=None):
    """Independently check mean horizon/origin squared losses after reveal.

    The caller controls the held-out-data access gate. Inputs may be vectors or
    matrices (origins by horizons); origins/horizons are not inference units.
    """
    y, prediction = np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    if y.shape != prediction.shape or y.ndim not in (1, 2) or y.size == 0:
        raise ValueError("Prediction/target shapes differ or are empty")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Predictions and test targets must be finite")
    origin_losses = np.square(y - prediction)
    if origin_losses.ndim == 2:
        origin_losses = origin_losses.mean(axis=1)
    mse = float(origin_losses.mean())
    if not math.isclose(mse, float(reported_mse), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Reported test MSE differs from independent prediction residuals")
    if reported_origin_losses is not None:
        saved = np.asarray(reported_origin_losses, dtype=float)
        if saved.shape != origin_losses.shape or not np.allclose(saved, origin_losses, rtol=1e-10, atol=1e-12):
            raise ValueError("Saved per-origin squared losses differ from predictions")
    return {"origins": len(origin_losses), "horizons": y.shape[1] if y.ndim == 2 else 1,
            "independently_recomputed_mse": mse}
