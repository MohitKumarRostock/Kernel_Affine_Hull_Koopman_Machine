"""Independent, descriptive audit of the bounded input-scaling sensitivity.

This module never fits a model and does not import either experimental runner.
Natural losses are reconstructed from saved predictions. Controlled predictions
are audited when the full local archive is supplied; the compact public archive
supports a separate replay from per-episode losses. The latter is deliberately
not described as a fresh prediction audit. Existing source roles and synthetic
draws are reused: no p-values, confidence intervals, or new confirmation claims
are produced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import math

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
NATURAL_ORIGINAL = REPOSITORY / "extensions/natural_forecasting/runs/natural_v1"
CONTROLLED_RECORDS = REPOSITORY / "extensions/predictive_geometry/archive/capacity/capacity_records.json"
GROUP_KEYS = ("source_family", "dataset_id", "series_id", "horizon", "history_length", "budget_id", "seed")
SCALERS = ("standard", "minmax")
NATURAL_BASE_METHODS = ("raw_affine_rbf", "raw_locality_rbf", "raw_kahm_rbf")
NATURAL_METHODS = ("raw_ridge", "raw_rbf") + tuple(
    f"{method}__{scaling}" for method in NATURAL_BASE_METHODS for scaling in SCALERS
)
CONTROLLED_FEATURES = ("raw", "affine:local", "locality:local", "kahm:local", "kahm:scores")
CONTROLLED_SYSTEMS = ("nonlinear_ar", "vanderpol", "duffing")
NATURAL_CONTRASTS = (
    ("raw_kahm_rbf__minmax", "raw_kahm_rbf__standard"),
    ("raw_affine_rbf__minmax", "raw_affine_rbf__standard"),
    ("raw_locality_rbf__minmax", "raw_locality_rbf__standard"),
    ("raw_kahm_rbf__minmax", "raw_rbf"),
    ("raw_kahm_rbf__minmax", "raw_ridge"),
    ("raw_kahm_rbf__minmax", "raw_affine_rbf__minmax"),
    ("raw_kahm_rbf__minmax", "raw_locality_rbf__minmax"),
    ("raw_kahm_rbf__standard", "raw_rbf"),
    ("raw_kahm_rbf__standard", "raw_ridge"),
    ("raw_kahm_rbf__standard", "raw_affine_rbf__standard"),
    ("raw_kahm_rbf__standard", "raw_locality_rbf__standard"),
)
CONTROLLED_CONTRASTS = tuple(
    (("minmax", feature), ("standard", feature))
    for feature in CONTROLLED_FEATURES if feature != "raw"
) + (
    (("minmax", "kahm:local"), ("standard", "raw")),
    (("minmax", "kahm:local"), ("minmax", "affine:local")),
    (("minmax", "kahm:local"), ("minmax", "locality:local")),
    (("minmax", "affine:local"), ("standard", "raw")),
    (("minmax", "locality:local"), ("standard", "raw")),
    (("minmax", "kahm:scores"), ("standard", "raw")),
    (("standard", "kahm:local"), ("standard", "raw")),
    (("standard", "kahm:local"), ("standard", "affine:local")),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(condition, description):
    if not condition:
        raise ValueError(description)


def close(actual, expected, description, *, rtol=1e-10, atol=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, description + ": shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), description + ": nonfinite value")
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), description + ": numerical mismatch")
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def finite_positive(value, description):
    value = float(value)
    require(math.isfinite(value) and value > 0, description)
    return value


def hierarchical_weights(rows, keys=GROUP_KEYS):
    """Independent implementation of equal mass at every nested task level."""
    require(bool(rows), "Cannot aggregate an empty set")
    output = np.ones(len(rows), dtype=float)
    for index, row in enumerate(rows):
        prefix = []
        for depth, key in enumerate(keys):
            children = {
                candidate[key] for candidate in rows
                if all(candidate[keys[j]] == prefix[j] for j in range(depth))
            }
            output[index] /= len(children)
            prefix.append(row[key])
        duplicate_count = sum(all(candidate[key] == row[key] for key in keys) for candidate in rows)
        output[index] /= duplicate_count
    close(output.sum(), 1.0, "Hierarchical weights sum")
    return output


def grouped_square_errors(prediction, target, groups):
    """Average origins within an episode, retaining every target horizon."""
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    groups = np.asarray(groups)
    require(prediction.shape == target.shape and target.ndim == 2, "Controlled prediction/target shape")
    require(groups.shape == (len(target),), "Controlled group shape")
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), "Nonfinite controlled prediction")
    squared = (prediction - target) ** 2
    return np.stack([squared[groups == group].mean(axis=0) for group in np.unique(groups)])


def paired_summary(candidate, reference):
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    require(candidate.ndim == 1 and candidate.shape == reference.shape and len(candidate) >= 2,
            "Paired summaries need at least two matched complete units")
    require(np.isfinite(candidate).all() and np.isfinite(reference).all(), "Nonfinite paired outcome")
    difference = candidate - reference
    reference_mean = finite_positive(reference.mean(), "Positive comparison mean required")
    sd = float(difference.std(ddof=1))
    return {
        "units": int(len(candidate)), "candidate_mean": float(candidate.mean()),
        "reference_mean": reference_mean, "mean_difference": float(difference.mean()),
        "relative_change_percent": 100.0 * (float(candidate.mean()) / reference_mean - 1.0),
        "difference_sd": sd, "difference_se": sd / math.sqrt(len(candidate)),
        "difference_min": float(difference.min()), "difference_max": float(difference.max()),
        "candidate_lower_count": int(np.count_nonzero(difference < 0)),
        "candidate_equal_count": int(np.count_nonzero(difference == 0)),
        "descriptive_five_percent_reduction": float(candidate.mean()) <= 0.95 * reference_mean,
        "inference": "Descriptive paired dispersion on reused units; no interval or significance claim.",
    }


def _natural_scaler_audit(candidate, arrays):
    """Validate fitted scaler statistics and range diagnostics independently."""
    scaling = candidate.get("geometry_scaling")
    if candidate["candidate_id"] in ("raw_ridge", "raw_rbf"):
        scaling = "standard"
    checked = 0
    for stage, key in (("search", "X_train"), ("refit", "X_refit")):
        metadata = candidate[stage + "_scaler_metadata"]
        x = np.asarray(arrays[key], float)
        query = np.asarray(arrays["X_validation" if stage == "search" else "X_test"], float)
        y = np.asarray(arrays["Y_train" if stage == "search" else "Y_refit"], float)
        minimum, maximum = x.min(axis=0), x.max(axis=0)
        span, constant = maximum - minimum, maximum == minimum
        expected_mean = x.mean(axis=0)
        expected_std = x.std(axis=0)
        expected_std = np.where(expected_std > 1e-12, expected_std, 1.0)
        require(metadata["training_rows"] == len(x), stage + " fitting row count")
        header = f"{x.dtype.str}:{x.shape}:".encode()
        require(metadata["training_input_sha256"] == hashlib.sha256(header + np.ascontiguousarray(x).tobytes()).hexdigest(),
                stage + " training input hash")
        require(metadata["mode"] == scaling and metadata["clip"] is False,
                stage + " changed scaling mode or clipping")
        close(metadata["training_minimum"], minimum, stage + " minimum")
        close(metadata["training_maximum"], maximum, stage + " maximum")
        require(np.array_equal(np.asarray(metadata["constant_columns"], bool), constant), stage + " constant columns")
        # The raw skip always uses z scores, independent of geometry scaling.
        close(metadata["raw_input_mean"], expected_mean, stage + " raw mean")
        close(metadata["raw_input_scale"], expected_std, stage + " raw scale")
        if scaling == "standard":
            center, scale = expected_mean, expected_std
        else:
            center = minimum + 0.5 * span
            scale = np.where(constant, 1.0, 0.5 * span)
        close(metadata["effective_center"], center, stage + " geometry center")
        close(metadata["effective_scale"], scale, stage + " geometry scale")
        target_std = y.std(axis=0)
        close(metadata["target_mean"], y.mean(axis=0), stage + " target mean")
        close(metadata["target_scale"], np.where(target_std > 1e-12, target_std, 1.0), stage + " target scale")
        diagnostic = candidate[("validation" if stage == "search" else "test") + "_out_of_range"]
        outside = (query < minimum) | (query > maximum)
        scaled_query = 2.0 * (query - minimum) / np.where(constant, 1.0, span) - 1.0
        scaled_query[:, constant] = 0.0
        require(diagnostic["query_rows"] == len(query) and diagnostic["query_coordinates"] == query.size,
                stage + " query dimensions")
        require(diagnostic["outside_training_range_coordinates"] == int(outside.sum()), stage + " outside coordinate count")
        require(diagnostic["outside_training_range_rows"] == int(outside.any(axis=1).sum()), stage + " outside row count")
        close(diagnostic["outside_training_range_coordinate_fraction"], outside.mean(), stage + " outside coordinate fraction")
        close(diagnostic["outside_training_range_row_fraction"], outside.any(axis=1).mean(), stage + " outside row fraction")
        close(diagnostic["maximum_abs_minmax_coordinate"], np.abs(scaled_query).max(), stage + " minmax extrapolation")
        checked += 1
    return checked


def audit_natural(results_root):
    results_root = Path(results_root)
    original_roster = read_json(NATURAL_ORIGINAL / "task_roster.json")["tasks"]
    roster = {task["task_id"]: task for task in original_roster}
    require(len(roster) == 208, "Original natural task roster changed")
    rows, seen = [], set()
    counters = {"tasks": 0, "predictions_checked": 0, "origin_loss_arrays_checked": 0,
                "historical_predictions_checked": 0, "scaler_statistic_sets_checked": 0,
                "maximum_historical_prediction_absolute_difference": 0.0,
                "maximum_loss_reconstruction_absolute_difference": 0.0,
                "maximum_loss_reconstruction_relative_difference": 0.0}
    source_files = {}
    for phase in ("development", "confirmation"):
        paths = sorted((results_root / phase).glob("*.json"))
        for path in paths:
            result = read_json(path)
            if "task_id" not in result:
                continue
            task_id = result["task_id"]
            require(task_id in roster and task_id not in seen, "Unknown/duplicate natural task: " + task_id)
            seen.add(task_id)
            task = roster[task_id]
            require(result.get("status") == "complete" and not result.get("failures"), "Incomplete natural task")
            require(result["phase"] == phase == task["role"], "Changed original natural role")
            metadata_map = {"source_family": "family_id", "dataset_id": "source_id",
                            "series_id": "series_id", "horizon": "horizon", "history_length": "context",
                            "budget_id": "budget_id", "seed": "seed"}
            for key, original_key in metadata_map.items():
                require(result[key] == task[original_key], "Changed natural task field: " + key)
            scale = finite_positive(result["loss_scale"], "Invalid natural loss scale")
            close(scale, task["loss_scale"], "Natural loss scale changed")
            cache_path = NATURAL_ORIGINAL / task["cache_path"]
            require(digest(cache_path) == task["cache_sha256"], "Changed prepared natural arrays")
            require(result["source_task_cache_sha256"] == task["cache_sha256"], "Changed recorded prepared-array hash")
            prediction_path = path.with_suffix(".npz")
            if result.get("predictions_sha256") is not None:
                require(digest(prediction_path) == result["predictions_sha256"], "Natural prediction file hash")
            original_name = hashlib.sha256(task_id.encode()).hexdigest()[:24]
            original_path = NATURAL_ORIGINAL / phase / (original_name + ".json")
            original_result = read_json(original_path)
            original_candidates = {row["candidate_id"]: row for row in original_result["candidates"]}
            candidates = {row["candidate_id"]: row for row in result["candidates"]}
            require(len(candidates) == len(result["candidates"]) and set(candidates) == set(NATURAL_METHODS),
                    "Changed or incomplete natural candidate roster")
            source_files[str(path.relative_to(ROOT))] = digest(path)
            source_files[str(prediction_path.relative_to(ROOT))] = digest(prediction_path)
            with np.load(cache_path, allow_pickle=False) as arrays, \
                    np.load(prediction_path, allow_pickle=False) as predictions, \
                    np.load(original_path.with_suffix(".npz"), allow_pickle=False) as historical:
                target = predictions["target"]
                close(target, arrays["Y_test"], "Natural saved targets", rtol=0, atol=0)
                close(predictions["origin"], arrays["origin_test"], "Natural test origins", rtol=0, atol=0)
                close(predictions["loss_scale"], historical["loss_scale"], "Natural NPZ loss scale", rtol=0, atol=0)
                for name in NATURAL_METHODS:
                    candidate = candidates[name]
                    require(candidate["status"] == "ok", "Incomplete natural method")
                    prediction = predictions[name]
                    require(prediction.shape == target.shape, "Natural prediction shape")
                    require(np.isfinite(prediction).all(), "Nonfinite natural prediction")
                    origin_losses = ((prediction - target) ** 2).mean(axis=1)
                    close(predictions[name + "__origin_losses"], origin_losses, "Natural per-origin loss")
                    mse = float(origin_losses.mean())
                    error = close(candidate["test_loss"], mse, "Natural MSE")
                    counters["maximum_loss_reconstruction_absolute_difference"] = max(
                        counters["maximum_loss_reconstruction_absolute_difference"], error)
                    counters["maximum_loss_reconstruction_relative_difference"] = max(
                        counters["maximum_loss_reconstruction_relative_difference"], error / max(abs(mse), 1e-300))
                    counters["predictions_checked"] += 1
                    counters["origin_loss_arrays_checked"] += 1
                    base = name.split("__")[0]
                    scaling = name.split("__")[1] if "__" in name else "invariant"
                    require(candidate["base_method"] == base, "Changed method identity")
                    require(candidate["geometry_scaling"] == ("standard" if scaling == "invariant" else scaling),
                            "Changed natural scaling arm")
                    trials = candidate["trials"]
                    require([trial["alpha"] for trial in trials] == [0.01, 1.0, 100.0], "Changed alpha grid")
                    winner = min(trials, key=lambda trial: trial["validation_mse"])
                    require(candidate["selected_parameters"] == {"alpha": winner["alpha"]},
                            "Natural alpha selection differs from first validation minimum")
                    close(candidate["validation_loss"], winner["validation_mse"], "Natural winning validation MSE")
                    if scaling in ("standard", "invariant"):
                        difference = close(prediction, historical[base], "Historical standard prediction parity",
                                           rtol=1e-9, atol=1e-10)
                        close(candidate["test_loss"], original_candidates[base]["test_loss"],
                              "Historical standard loss parity", rtol=1e-9, atol=1e-10)
                        require(candidate["selected_parameters"] == original_candidates[base]["selected_parameters"],
                                "Historical standard alpha selection changed")
                        counters["historical_predictions_checked"] += 1
                        counters["maximum_historical_prediction_absolute_difference"] = max(
                            counters["maximum_historical_prediction_absolute_difference"], difference)
                    counters["scaler_statistic_sets_checked"] += _natural_scaler_audit(candidate, arrays)
                    cost = float(candidate["cost_seconds"])
                    require(math.isfinite(cost) and cost >= 0, "Invalid natural cost")
                    nodes = candidate["cost_nodes"]
                    require(len(nodes) == 5 and len({node["id"] for node in nodes}) == 5,
                            "Natural method cost graph changed")
                    close(cost, sum(node["seconds"] for node in nodes), "Natural wall-cost sum")
                    cpu = float(candidate["cost_process_cpu_seconds"])
                    close(cpu, sum(node["process_cpu_seconds"] for node in nodes), "Natural CPU-cost sum")
                    stage_costs = {}
                    for stage in ("search", "selection_overhead", "refit", "forecast", "serialization"):
                        stage_nodes = [node for node in nodes if node["id"] == name + ":" + stage]
                        require(len(stage_nodes) == 1, "Missing or duplicate method cost stage")
                        stage_costs[stage + "_seconds"] = stage_nodes[0]["seconds"]
                    row = {key: result[key] for key in GROUP_KEYS}
                    row.update(phase=phase, task_id=task_id, candidate_id=name, base_method=base,
                               geometry_scaling=scaling, test_mse=mse, loss_scale=scale,
                               scaled_mse=mse / scale, cost_seconds=cost,
                               process_cpu_seconds=cpu, **stage_costs,
                               audit_diagnostic_seconds=candidate["audit_diagnostic_seconds"],
                               launch_to_result_seconds=candidate["launch_to_result_seconds"],
                               data_load_seconds=result["data_load_seconds"])
                    for stage in ("validation", "test"):
                        diagnostic = candidate[stage + "_out_of_range"]
                        row[stage + "_outside_coordinate_fraction"] = diagnostic["outside_training_range_coordinate_fraction"]
                        row[stage + "_outside_row_fraction"] = diagnostic["outside_training_range_row_fraction"]
                        row[stage + "_maximum_abs_minmax_coordinate"] = diagnostic["maximum_abs_minmax_coordinate"]
                    row["refit_constant_coordinates"] = sum(candidate["refit_scaler_metadata"]["constant_columns"])
                    row["search_constant_coordinates"] = sum(candidate["search_scaler_metadata"]["constant_columns"])
                    rows.append(row)
            counters["tasks"] += 1
    require(seen == set(roster), "Natural study is missing declared tasks")
    require(len(rows) == 1664, "Natural result count differs from the bounded protocol")
    return rows, {**counters, "source_files_sha256": source_files,
                  "scope": "All saved natural predictions, per-origin losses, prepared targets, and original standard outputs checked."}


def natural_summary(rows):
    methods, source_rows, contrasts, phase_summaries = [], [], [], {}
    metric_fields = ("scaled_mse", "cost_seconds", "process_cpu_seconds", "search_seconds",
                     "selection_overhead_seconds", "refit_seconds", "forecast_seconds", "serialization_seconds",
                     "audit_diagnostic_seconds", "launch_to_result_seconds", "data_load_seconds",
                     "test_outside_coordinate_fraction", "test_outside_row_fraction",
                     "validation_outside_coordinate_fraction", "validation_outside_row_fraction")
    for phase in ("development", "confirmation"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        task_ids = sorted({row["task_id"] for row in phase_rows})
        families = sorted({row["source_family"] for row in phase_rows})
        require(len(task_ids) == (128 if phase == "development" else 80) and len(families) == 4,
                "Natural phase roster size changed")
        lookup = {(row["task_id"], row["candidate_id"]): row for row in phase_rows}
        by_method = {}
        by_family = defaultdict(dict)
        timing_diagnostics = {}
        for name in NATURAL_METHODS:
            values = [lookup[(task, name)] for task in task_ids]
            weights = hierarchical_weights(values)
            metrics = {field: float(weights @ np.asarray([row[field] for row in values]))
                       for field in metric_fields}
            family_losses = []
            for family in families:
                subset = [row for row in values if row["source_family"] == family]
                local_weights = hierarchical_weights(subset)
                local_metrics = {field: float(local_weights @ np.asarray([row[field] for row in subset]))
                                 for field in metric_fields}
                local_metrics["maximum_test_abs_minmax_coordinate"] = float(max(
                    row["test_maximum_abs_minmax_coordinate"] for row in subset))
                local_metrics["maximum_validation_abs_minmax_coordinate"] = float(max(
                    row["validation_maximum_abs_minmax_coordinate"] for row in subset))
                by_family[family][name] = local_metrics
                family_losses.append(local_metrics["scaled_mse"])
                source_rows.append({"phase": phase, "source_family": family, "candidate_id": name,
                                    "task_count": len(subset), **local_metrics})
            metrics["between_family_loss_sd"] = float(np.std(family_losses, ddof=1))
            metrics["source_count"] = len(families)
            metrics["task_count"] = len(values)
            metrics["by_budget"] = {}
            for budget in ("small", "medium"):
                budget_values = [row for row in values if row["budget_id"] == budget]
                budget_weights = hierarchical_weights(budget_values)
                metrics["by_budget"][budget] = {
                    field: float(budget_weights @ np.asarray([row[field] for row in budget_values]))
                    for field in metric_fields
                }
            metrics["test_outside_coordinate_fraction"] = float(weights @ np.asarray([
                row["test_outside_coordinate_fraction"] for row in values]))
            metrics["test_outside_row_fraction"] = float(weights @ np.asarray([
                row["test_outside_row_fraction"] for row in values]))
            metrics["maximum_test_abs_minmax_coordinate"] = float(max(
                row["test_maximum_abs_minmax_coordinate"] for row in values))
            metrics["maximum_validation_abs_minmax_coordinate"] = float(max(
                row["validation_maximum_abs_minmax_coordinate"] for row in values))
            metrics["tasks_with_constant_refit_coordinates"] = sum(
                row["refit_constant_coordinates"] > 0 for row in values)
            metrics["tasks_with_constant_search_coordinates"] = sum(
                row["search_constant_coordinates"] > 0 for row in values)
            by_method[name] = metrics
            methods.append({"phase": phase, "candidate_id": name,
                            **{key: value for key, value in metrics.items() if key != "by_budget"}})
            by_task_weights = {row["task_id"]: float(weight) for row, weight in zip(values, weights)}
            highest = sorted(values, key=lambda row: (-row["cost_seconds"], row["task_id"]))[:3]
            timing_diagnostics[name] = {
                "unweighted_task_median_seconds": float(np.median([row["cost_seconds"] for row in values])),
                "unweighted_task_minimum_seconds": float(min(row["cost_seconds"] for row in values)),
                "unweighted_task_maximum_seconds": float(max(row["cost_seconds"] for row in values)),
                "largest_three_recorded_tasks": [
                    {**{key: row[key] for key in ("task_id", "source_family", "budget_id", "seed", "cost_seconds",
                                                  "process_cpu_seconds", "search_seconds", "selection_overhead_seconds",
                                                  "refit_seconds", "forecast_seconds", "serialization_seconds",
                                                  "audit_diagnostic_seconds", "launch_to_result_seconds")},
                     "hierarchical_weight": by_task_weights[row["task_id"]],
                     "contribution_to_aggregate_seconds": row["cost_seconds"] * by_task_weights[row["task_id"]]}
                    for row in highest
                ],
                "scope": "Single execution diagnostics. Every recorded timing is retained; no trimming or replacement runs.",
            }
        local_contrasts = []
        for candidate, reference in NATURAL_CONTRASTS:
            c = np.array([by_family[family][candidate]["scaled_mse"] for family in families])
            r = np.array([by_family[family][reference]["scaled_mse"] for family in families])
            result = paired_summary(c, r)
            # The four-source range/LOFO description is preferable to treating
            # reused tasks, windows, or fitting seeds as independent domains.
            result.pop("difference_se")
            result["candidate_cost_seconds"] = by_method[candidate]["cost_seconds"]
            result["reference_cost_seconds"] = by_method[reference]["cost_seconds"]
            result["cost_ratio"] = result["candidate_cost_seconds"] / finite_positive(
                result["reference_cost_seconds"], "Positive reference timing required")
            family_details = []
            leave_one_out = []
            for index, family in enumerate(families):
                family_details.append({"source_family": family, "candidate_loss": float(c[index]),
                                       "reference_loss": float(r[index]), "difference": float(c[index] - r[index]),
                                       "relative_change_percent": float(100 * (c[index] / r[index] - 1))})
                keep = np.arange(len(families)) != index
                leave_one_out.append({"omitted_source_family": family,
                                      "candidate_mean": float(c[keep].mean()),
                                      "reference_mean": float(r[keep].mean()),
                                      "mean_difference": float((c[keep] - r[keep]).mean()),
                                      "relative_change_percent": float(100 * (c[keep].mean() / r[keep].mean() - 1))})
            entry = {"phase": phase, "candidate": candidate, "reference": reference, **result,
                     "by_source": family_details, "leave_one_source_out": leave_one_out}
            local_contrasts.append(entry)
            contrasts.append({key: value for key, value in entry.items() if key not in ("by_source", "leave_one_source_out")})
        phase_summaries[phase] = {"task_count": len(task_ids), "source_count": len(families),
                                  "sources": families, "methods": by_method,
                                  "by_source": dict(by_family), "contrasts": local_contrasts,
                                  "primary_scaling_contrast": local_contrasts[0],
                                  "timing_diagnostics": timing_diagnostics,
                                  "timing_scope": "Contemporaneous warm method totals exclude reused data preparation, current loading, process startup and sensitivity audit work. Launch-to-result and audit elapsed times are separately recorded. The inherited import timer is not a full cold-start measurement.",
                                  "role_interpretation": "Original study role; every new result is post-hoc sensitivity."}
    return {"phases": phase_summaries, "metric": "Ratio/difference of hierarchical aggregate scaled MSEs; not mean task-level percent changes.",
            "inference": "Four original source families per phase. Descriptive family dispersion and leave-one-family-out results only."}, methods, source_rows, contrasts


def _controlled_historical():
    records = read_json(CONTROLLED_RECORDS)
    if isinstance(records, dict):
        records = records["records"]
    return {(row["replicate"], row["setting_id"], row["feature"], row["landmarks"]): row
            for row in records if row["feature"] != "validation_selector"}


def audit_controlled(results_root, full_predictions_root=None):
    results_root = Path(results_root)
    full_root = Path(full_predictions_root) if full_predictions_root is not None else None
    paths = sorted(results_root.glob("**/result.json"))
    require(len(paths) == 72, "Controlled study needs all 72 complete jobs")
    historical = _controlled_historical()
    seen, rows = set(), []
    counters = {"jobs": 0, "per_episode_loss_arrays_checked": 0,
                "full_prediction_arrays_checked": 0, "historical_standard_mses_checked": 0,
                "cross_arm_raw_arrays_checked": 0,
                "runner_attested_full_historical_prediction_parity": 0,
                "maximum_loss_reconstruction_absolute_difference": 0.0,
                "maximum_historical_mse_absolute_difference": 0.0}
    files = {}
    for path in paths:
        result = read_json(path)
        require(result.get("status") == "complete", "Incomplete controlled job")
        system, rep = result["system"], result["replicate"]
        require(system in CONTROLLED_SYSTEMS and rep in range(24) and (system, rep) not in seen,
                "Unknown/duplicated controlled job")
        seen.add((system, rep))
        records = result["records"]
        expected = {(scaling, feature, width) for scaling in SCALERS
                    for feature in CONTROLLED_FEATURES for width in (64, 256)}
        keys = [(record["scaling"], record["feature"], record["landmarks"]) for record in records]
        require(len(keys) == 20 and set(keys) == expected, "Changed controlled record roster")
        loss_path = path.parent / "unit_losses.npz"
        require(digest(loss_path) == result["artifacts_sha256"]["unit_losses.npz"], "Controlled loss file hash")
        files[str(path.relative_to(ROOT))] = digest(path)
        files[str(loss_path.relative_to(ROOT))] = digest(loss_path)
        prediction_path = path.parent / "predictions.npz"
        if full_root is not None:
            candidates = [full_root / result["job_id"] / "predictions.npz",
                          full_root / "jobs" / result["job_id"] / "predictions.npz"]
            prediction_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            require(prediction_path.is_file(), "Missing full controlled predictions for requested independent audit")
        if not prediction_path.is_file():
            predictions = None
        else:
            require(digest(prediction_path) == result["artifacts_sha256"]["predictions.npz"], "Controlled prediction file hash")
            predictions = np.load(prediction_path, allow_pickle=False)
            test_metadata = next(row for row in result["split_metadata"] if row["split"] == "test")
            require(array_digest(predictions["Y"]) == test_metadata["Y_sha256"], "Controlled saved target hash")
            require(array_digest(predictions["groups"]) == test_metadata["group_sha256"], "Controlled saved group hash")
        with np.load(loss_path, allow_pickle=False) as losses:
            keyed = {}
            for record in records:
                loss_array = losses[record["unit_loss_key"]]
                require(loss_array.shape == (512, 3) and np.isfinite(loss_array).all() and (loss_array >= 0).all(),
                        "Invalid per-episode controlled losses")
                error = close(record["test_mse"], loss_array.mean(), "Controlled MSE reconstruction")
                close(record["test_mse_by_target"], loss_array.mean(axis=0), "Controlled per-target MSE")
                counters["maximum_loss_reconstruction_absolute_difference"] = max(
                    counters["maximum_loss_reconstruction_absolute_difference"], error)
                counters["per_episode_loss_arrays_checked"] += 1
                if predictions is not None:
                    reconstructed = grouped_square_errors(predictions[record["prediction_key"]],
                                                           predictions["Y"], predictions["groups"])
                    close(loss_array, reconstructed, "Controlled saved prediction to per-episode loss")
                    counters["full_prediction_arrays_checked"] += 1
                if record["scaling"] == "standard" and record["feature"] != "locality:local":
                    key = (rep, result["setting_id"], record["feature"], record["landmarks"])
                    previous = historical[key]
                    error = close(record["test_mse"], previous["test_mse"], "Historical controlled standard MSE",
                                  rtol=1e-9, atol=1e-11)
                    require(len(record["selected"]) == len(previous["selected"]), "Changed standard controlled head count")
                    for current_selection, previous_selection in zip(record["selected"], previous["selected"]):
                        for field in ("alpha", "bandwidth_multiplier", "basis_dimension"):
                            require(current_selection[field] == previous_selection[field], "Changed selected standard controlled head")
                        for field in ("bandwidth_sq", "validation_mse_by_target"):
                            close(current_selection[field], previous_selection[field], "Historical selected head metadata",
                                  rtol=1e-10, atol=1e-12)
                    if record["historical_parity"]["full_prediction_checked"]:
                        counters["runner_attested_full_historical_prediction_parity"] += 1
                    counters["historical_standard_mses_checked"] += 1
                    counters["maximum_historical_mse_absolute_difference"] = max(
                        counters["maximum_historical_mse_absolute_difference"], error)
                key = (record["scaling"], record["feature"], record["landmarks"])
                keyed[key] = loss_array
                rows.append({"system": system, "setting_id": result["setting_id"], "replicate": rep,
                             "scaling": record["scaling"], "feature": record["feature"],
                             "landmarks": record["landmarks"], "test_mse": float(loss_array.mean()),
                             "validation_mse": record["validation_mse"],
                             "fit_predict_seconds": record["fit_predict_seconds"]})
            for width in (64, 256):
                close(keyed[("standard", "raw", width)], keyed[("minmax", "raw", width)],
                      "Invariant raw control", rtol=0, atol=0)
                counters["cross_arm_raw_arrays_checked"] += 1
        if predictions is not None:
            predictions.close()
        counters["jobs"] += 1
    require(len(rows) == 1440, "Controlled method result count differs from the protocol")
    require(counters["full_prediction_arrays_checked"] in (0, 1440), "Partial full prediction audit is not acceptable")
    return rows, {**counters, "source_files_sha256": files,
                  "mode": "full_prediction_audit" if counters["full_prediction_arrays_checked"] else "compact_per_episode_loss_replay",
                  "scope": ("Independent replay of every saved prediction and episode loss."
                            if counters["full_prediction_arrays_checked"] else
                            "Compact public per-episode loss replay; full predictions were not read in this invocation.")}


def controlled_summary(rows):
    methods, contrasts, pair_rows, cells = [], [], [], []
    for system in CONTROLLED_SYSTEMS:
        for landmarks in (64, 256):
            group = [row for row in rows if row["system"] == system and row["landmarks"] == landmarks]
            lookup = {(row["scaling"], row["feature"], row["replicate"]): row for row in group}
            require(len(lookup) == len(group) == 240, "Controlled cell is incomplete")
            cell_methods = []
            values = {}
            for scaling in SCALERS:
                for feature in CONTROLLED_FEATURES:
                    losses = np.array([lookup[(scaling, feature, rep)]["test_mse"] for rep in range(24)])
                    values[(scaling, feature)] = losses
                    entry = {"system": system, "landmarks": landmarks, "scaling": scaling, "feature": feature,
                             "repetitions": 24, "mean_mse": float(losses.mean()), "mse_sd": float(losses.std(ddof=1)),
                             "mse_se": float(losses.std(ddof=1) / math.sqrt(24))}
                    methods.append(entry)
                    cell_methods.append(entry)
            cell_contrasts = []
            for candidate, reference in CONTROLLED_CONTRASTS:
                statistics = paired_summary(values[candidate], values[reference])
                entry = {"system": system, "landmarks": landmarks,
                         "candidate_scaling": candidate[0], "candidate_feature": candidate[1],
                         "reference_scaling": reference[0], "reference_feature": reference[1], **statistics}
                contrasts.append(entry)
                cell_contrasts.append(entry)
                for rep in range(24):
                    pair_rows.append({"system": system, "landmarks": landmarks, "replicate": rep,
                                      "candidate_scaling": candidate[0], "candidate_feature": candidate[1],
                                      "reference_scaling": reference[0], "reference_feature": reference[1],
                                      "candidate_mse": float(values[candidate][rep]),
                                      "reference_mse": float(values[reference][rep]),
                                      "difference": float(values[candidate][rep] - values[reference][rep])})
            cells.append({"system": system, "landmarks": landmarks, "methods": cell_methods,
                          "contrasts": cell_contrasts})
    return {"cells": cells, "unit": "A complete paired train/validation/test draw, not a test origin or horizon.",
            "inference": "24 previously inspected independent draws per law; paired SD/SE are descriptive, not new confidence guarantees.",
            "timing_scope": "Shared-bank execution diagnostics; not standalone pipeline cost comparisons."}, methods, contrasts, pair_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(rows), "Cannot write an empty analysis table")
    fields = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_full_audit_attestation(controlled_root):
    """Check that compact files are byte-identical to independently audited files.

    This verifies an earlier full-prediction audit record, not the omitted
    predictions themselves. Reproduction with --full-controlled-root rechecks
    the predictions directly after regenerating them from the frozen runner.
    """
    controlled_root = Path(controlled_root)
    path = controlled_root / "FULL_PREDICTION_AUDIT.json"
    audit = read_json(path)
    expected = {"status": "passed", "jobs": 72, "per_episode_loss_arrays_checked": 1440,
                "full_prediction_arrays_checked": 1440, "historical_standard_mses_checked": 576,
                "cross_arm_raw_arrays_checked": 144, "mode": "full_prediction_audit"}
    for key, value in expected.items():
        require(audit[key] == value, "Invalid full controlled prediction audit: " + key)
    require(audit["analysis_code_sha256"] == digest(__file__), "Full prediction audit used different analysis code")
    checked = 0
    for relative, expected_hash in audit["source_files_sha256"].items():
        parts = Path(relative).parts
        require("jobs" in parts, "Unexpected full-audit path")
        suffix = Path(*parts[parts.index("jobs"):])
        require(suffix.name in ("result.json", "unit_losses.npz"), "Unexpected full-audit artifact")
        require(digest(controlled_root / suffix) == expected_hash, "Compact artifact differs from full prediction audit")
        checked += 1
    require(checked == 144, "Incomplete full-audit compact artifact mapping")
    return {"audit_sha256": digest(path), "analysis_code_sha256": audit["analysis_code_sha256"],
            "full_prediction_arrays_verified_in_recorded_audit": 1440,
            "byte_identical_compact_artifacts_verified_now": checked,
            "scope": "Recorded full-prediction audit checked against every published loss/result file; this invocation does not reread omitted predictions."}


def generate(results_root=None, *, full_controlled_root=None, output=None):
    results_root = Path(results_root or ROOT / "results")
    output = Path(output or ROOT / "analysis")
    natural_rows, natural_audit = audit_natural(results_root / "natural")
    controlled_rows, controlled_audit = audit_controlled(results_root / "controlled", full_controlled_root)
    natural, natural_methods, natural_sources, natural_contrasts = natural_summary(natural_rows)
    controlled, controlled_methods, controlled_contrasts, controlled_pairs = controlled_summary(controlled_rows)
    summary = {"format": "kahm_scaling_sensitivity_analysis_v1", "scope": "Post-hoc paired sensitivity, not a new confirmation study.",
               "scalers": ["training standardization", "training min-max [-1,1], unclipped"],
               "natural": natural, "controlled": controlled,
               "analysis_code_sha256": digest(__file__)}
    audit = {"status": "passed", "natural": natural_audit, "controlled": controlled_audit,
             "analysis_code_sha256": digest(__file__), "summary_source": "Saved predictions and per-episode losses; no fitted model invoked."}
    if full_controlled_root is None:
        audit["recorded_full_controlled_prediction_audit"] = verify_full_audit_attestation(results_root / "controlled")
    write_json(output / "summary.json", summary)
    write_json(output / "AUDIT.json", audit)
    write_csv(output / "natural_task_methods.csv", natural_rows)
    write_csv(output / "natural_methods.csv", natural_methods)
    write_csv(output / "natural_sources.csv", natural_sources)
    write_csv(output / "natural_contrasts.csv", natural_contrasts)
    capacity_rows = []
    for phase, phase_summary in natural["phases"].items():
        for method, metrics in phase_summary["methods"].items():
            for budget, budget_metrics in metrics["by_budget"].items():
                capacity_rows.append({"phase": phase, "candidate_id": method,
                                      "budget_id": budget, **budget_metrics})
    write_csv(output / "natural_capacity.csv", capacity_rows)
    write_csv(output / "controlled_methods.csv", controlled_methods)
    write_csv(output / "controlled_contrasts.csv", controlled_contrasts)
    write_csv(output / "controlled_pairs.csv", controlled_pairs)
    return {"status": "passed", "natural_tasks": natural_audit["tasks"],
            "natural_predictions_checked": natural_audit["predictions_checked"],
            "controlled_jobs": controlled_audit["jobs"],
            "controlled_full_predictions_checked": controlled_audit["full_prediction_arrays_checked"],
            "controlled_mode": controlled_audit["mode"], "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--full-controlled-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    print(json.dumps(generate(args.results_root, full_controlled_root=args.full_controlled_root,
                              output=args.output), indent=2))


if __name__ == "__main__":
    main()
