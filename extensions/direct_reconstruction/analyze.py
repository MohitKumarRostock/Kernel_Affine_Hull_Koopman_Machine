#!/usr/bin/env python3
"""Independently audit saved evidence and report the frozen primary analyses.

No model, hyperparameter, source, or row is selected here.  Saved C/D forecasts
are scored directly; selected serialized predictors are replayed; recorded
development decisions and compute dependencies are checked.  Candidate heads
are not all independently refitted.  The eighteen paired intervals concern
the newly frozen synthetic family only.  Natural-source summaries are purely
descriptive because four purposively selected collections are not a random
sample of forecasting problems.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import run as runner

import numpy as np
from scipy.stats import t as student_t


LABELS = {
    "ridge": "Ridge", "raw_rbf": "Raw RBF", "raw_hgb": "Boosted trees",
    "seasonal_naive": "Seasonal naive", "nonkahm_policy": "Selected non-KAHM",
    "fixed_kahm": "Fixed KAHM", "gate_kahm": "Gate KAHM",
    "reconstruction_kahm": "Reconstruction KAHM",
    "fixed_affine": "Fixed affine", "gate_affine": "Gate affine",
    "reconstruction_affine": "Reconstruction affine",
    "fixed_locality": "Fixed locality", "gate_locality": "Gate locality",
    "reconstruction_locality": "Reconstruction locality",
    "fixed_rbf_local": "Fixed RBF-local", "gate_rbf_local": "Gate RBF-local",
    "reconstruction_rbf_local": "Reconstruction RBF-local",
}


def finite(value):
    return isinstance(value, (int, float, np.number)) and np.isfinite(value)


def close(a, b, rtol=1e-9, atol=1e-12):
    return finite(a) and finite(b) and bool(np.isclose(a, b, rtol=rtol, atol=atol))


def score(target, prediction, groups, offsets):
    """Equal group, then origin, then horizon loss (one group for natural data)."""
    target, prediction, groups = map(np.asarray, (target, prediction, groups))
    if target.shape != prediction.shape or target.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction matrix")
    error = np.square(target - prediction)
    unique = np.unique(groups)
    if len(groups) != len(error) or len(unique) == 0:
        raise ValueError("Invalid scoring groups")
    per_group_horizon = np.stack([error[groups == group].mean(axis=0) for group in unique])
    horizon = per_group_horizon.mean(axis=0)
    return {
        "mse": float(horizon.mean()), "n_groups": len(unique), "n_origins": len(error),
        "horizon_mse": {str(int(offset)): float(value) for offset, value in zip(offsets, horizon)},
        "group_mse": per_group_horizon.mean(axis=1).tolist(),
    }


def paired_interval(control, candidate, *, family_size=18, alpha=0.05):
    """Two-sided Bonferroni t interval; positive differences favor candidate."""
    control, candidate = np.asarray(control, float), np.asarray(candidate, float)
    if control.ndim != 1 or control.shape != candidate.shape or len(control) < 2:
        raise ValueError("Paired vectors of at least two repetitions required")
    if not np.isfinite(control).all() or not np.isfinite(candidate).all():
        raise ValueError("Incomplete paired evidence cannot yield a primary interval")
    if family_size < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid simultaneous-interval family")
    difference = control - candidate
    n = len(difference)
    mean, sd = float(difference.mean()), float(difference.std(ddof=1))
    se = sd / np.sqrt(n)
    critical = float(student_t.ppf(1 - alpha / (2 * family_size), n - 1))
    low, high = mean - critical * se, mean + critical * se
    return {
        "n_repetitions": n, "degrees_of_freedom": n - 1,
        "family_size": family_size, "family_alpha": alpha,
        "mean_control_mse": float(control.mean()), "mean_candidate_mse": float(candidate.mean()),
        "mean_difference": mean, "sd_difference": sd, "standard_error": float(se),
        "critical_t": critical, "interval_low": float(low), "interval_high": float(high),
        "relative_improvement_percent": (100 * mean / float(control.mean())
                                         if float(control.mean()) > 0 else None),
        "positive_repetitions": int(np.sum(difference > 0)),
        "negative_repetitions": int(np.sum(difference < 0)),
        "tied_repetitions": int(np.sum(difference == 0)),
        "conclusion": ("positive" if low > 0 else "negative" if high < 0 else "unresolved"),
        "differences": difference.tolist(),
    }


def equal_collection_mean(rows, value_key="scaled_mse"):
    """Equal series inside each source, then equal source; no precision weights."""
    grouped = defaultdict(list)
    for row in rows:
        if not finite(row[value_key]):
            raise ValueError("A complete roster is required for natural aggregation")
        grouped[row["source_key"]].append(float(row[value_key]))
    if not grouped:
        raise ValueError("No natural collection records")
    by_collection = {key: float(np.mean(values)) for key, values in sorted(grouped.items())}
    return float(np.mean(list(by_collection.values()))), by_collection


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError("An empty CSV needs an explicit column schema")
    fieldnames = fieldnames or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.failures = []

    def check(self, condition, message, category="general"):
        self.counts[category] += 1
        if not condition:
            self.failures.append(message)

    def array(self, observed, expected, message, *, exact=True, category="input"):
        observed, expected = np.asarray(observed), np.asarray(expected)
        same = observed.shape == expected.shape
        if same:
            same = (np.array_equal(observed, expected, equal_nan=True) if exact
                    else np.allclose(observed, expected, rtol=1e-10, atol=1e-10, equal_nan=False))
        self.check(same, message, category)


def audit_inputs(audit, task, arrays):
    task_id = task["task_id"]
    try:
        runner.validate_arrays(arrays, task)
    except Exception as error:
        audit.check(False, f"{task_id}: structure {error}", "input")
    for key, expected in task["array_sha256"].items():
        audit.check(key in arrays and runner.array_hash(arrays[key]) == expected,
                    f"{task_id}/{key}: canonical array hash", "hash")
    history, offsets = int(task["L"]), np.asarray(task["offsets"], dtype=int)
    upstream = runner.generator() if task["suite"] == "synthetic" else None
    for split in runner.SPLITS:
        origins, groups = arrays[f"{split}_origins"], arrays[f"{split}_groups"]
        x, y = arrays[f"{split}_X"], arrays[f"{split}_Y"]
        if task["suite"] == "synthetic":
            episodes = arrays[f"{split}_episodes"]
            expected_x = np.stack([episodes[g, t - history + 1:t + 1]
                                   for g, t in zip(groups, origins)])
            expected_y = np.stack([episodes[g, t + offsets] for g, t in zip(groups, origins)])
            all_origins = np.arange(history - 1, episodes.shape[1] - max(offsets),
                                    runner.ORIGIN_STRIDE)
            rows = arrays[f"{split}_row_indices"]
            audit.array(groups, rows // len(all_origins), f"{task_id}/{split}: episode labels")
            audit.array(origins, all_origins[rows % len(all_origins)],
                        f"{task_id}/{split}: original row order")
            metadata = task["split_records"][split]
            regenerated = upstream.generate_episodes(
                task["law"], metadata["episode_count"], runner.EPISODE_LENGTH,
                metadata["generator_seed"],
            )[:, :, 0]
            audit.array(episodes, regenerated, f"{task_id}/{split}: inherited generator replay", category="generator")
            count, cap = len(episodes) * len(all_origins), runner.SPLIT_CAPS[split]
            chosen = np.arange(count)
            if cap is not None and cap < count:
                chosen = np.sort(np.random.default_rng(metadata["cap_seed"]).choice(
                    count, size=cap, replace=False))
            audit.array(rows, chosen, f"{task_id}/{split}: deterministic cap")
            audit.check(len(episodes) == runner.SPLIT_EPISODES[split],
                        f"{task_id}/{split}: episode count", "input")
        else:
            raw = arrays["raw_values"]
            expected_x = raw[origins[:, None] - np.arange(history - 1, -1, -1)[None, :]]
            expected_y = raw[origins[:, None] + offsets[None, :]]
            lo, hi = task["segment_bounds"][split]
            audit.check(bool(np.all(origins + 1 >= lo) and np.all(origins + max(offsets) < hi)),
                        f"{task_id}/{split}: target segment availability", "chronology")
            audit.check(np.all(np.diff(origins) > 0),
                        f"{task_id}/{split}: strictly chronological retained origins", "chronology")
            audit.array(groups, np.zeros(len(origins), dtype=groups.dtype),
                        f"{task_id}/{split}: natural groups")
            audit.array(arrays[f"{split}_global_origins"], origins + task["raw_start_index"],
                        f"{task_id}/{split}: global origins")
            audit.check(len(origins) <= runner.NATURAL_CAPS[split],
                        f"{task_id}/{split}: declared cap", "input")
            available = []
            for t in range(max(history - 1, lo - int(offsets.min())), hi - int(offsets.max())):
                if (np.isfinite(raw[t - history + 1:t + 1]).all()
                        and np.isfinite(raw[t + offsets]).all()):
                    available.append(t)
            available = np.asarray(available, dtype=np.int64)
            audit.array(arrays[f"{split}_available_origins"], available,
                        f"{task_id}/{split}: finite available window roster")
            cap = runner.NATURAL_CAPS[split]
            positions = (np.arange(len(available), dtype=np.int64) if len(available) <= cap
                         else (np.arange(cap, dtype=np.int64) * (len(available) - 1)) // (cap - 1))
            audit.array(arrays[f"{split}_row_indices"], positions,
                        f"{task_id}/{split}: deterministic cap positions")
            audit.array(origins, available[positions], f"{task_id}/{split}: capped origin roster")
        audit.array(x, expected_x, f"{task_id}/{split}: causal history reconstruction")
        audit.array(y, expected_y, f"{task_id}/{split}: target reconstruction")
    if task["suite"] == "natural":
        for before, after in zip(runner.SPLITS[:-1], runner.SPLITS[1:]):
            if len(arrays[f"{before}_origins"]) == 0 or len(arrays[f"{after}_origins"]) == 0:
                continue
            last_label = int(np.max(arrays[f"{before}_origins"]) + max(offsets))
            first_origin = int(np.min(arrays[f"{after}_origins"]))
            audit.check(last_label <= first_origin,
                        f"{task_id}: {before} targets unavailable at first {after} origin", "chronology")
        prefix = arrays["raw_values"][:task["segment_bounds"]["A"][1]]
        period = int(task["p"])
        difference = prefix[period:] - prefix[:-period]
        squared = np.square(difference[np.isfinite(difference)])
        scale = float(squared.mean()) if len(squared) else float("nan")
        if not np.isfinite(scale) or scale <= 0:
            difference = np.diff(prefix)
            squared = np.square(difference[np.isfinite(difference)])
            scale = float(squared.mean()) if len(squared) else float("nan")
        audit.check(close(scale, task["loss_scale"]), f"{task_id}: A-only loss denominator", "chronology")
        origins = arrays["A_origins"]
        train, validation = arrays["A_inner_train_mask"], arrays["A_inner_validation_mask"]
        audit.check(np.max(origins[train] + max(offsets)) <= np.min(origins[validation]),
                    f"{task_id}: purged inner-A labels", "chronology")
        expected_validation = np.arange(len(origins)) >= 4 * len(origins) // 5
        expected_train = (~expected_validation) & (origins + max(offsets) <= origins[expected_validation][0])
        audit.array(validation, expected_validation, f"{task_id}: inner-A validation rule", category="chronology")
        audit.array(train, expected_train, f"{task_id}: inner-A training rule", category="chronology")


def audit_diagnostics(audit, task, arrays, predictions, diagnostics, config, methods):
    task_id = task["task_id"]
    method_records = diagnostics.get("methods", {})
    nodes = diagnostics.get("cost_nodes", {})
    audit.check(diagnostics.get("config") == config and diagnostics.get("seed") == task["fit_seed"],
                f"{task_id}: executed configuration/seed differ from freeze", "selection")
    audit.check(set(method_records) == set(methods), f"{task_id}: diagnostics method roster", "selection")
    optimization_records, fallback_records = [], []
    for method in methods:
        record = method_records.get(method, {})
        prediction_c = predictions[f"C__{method}"]
        expected_c = float(np.mean(np.square(arrays["C_Y"] - prediction_c)))
        if "selected_c_mse" in record:
            audit.check(close(expected_c, record["selected_c_mse"]),
                        f"{task_id}/{method}: selected aggregate C MSE", "selection")
        dependencies = record.get("cost_dependencies", [])
        audit.check(len(dependencies) == len(set(dependencies)),
                    f"{task_id}/{method}: repeated compute dependency", "cost")
        audit.check(bool(dependencies) and all(name in nodes for name in dependencies),
                    f"{task_id}/{method}: missing compute dependency", "cost")
        for measure in ("wall_seconds", "cpu_seconds"):
            if dependencies and all(name in nodes and measure in nodes[name] for name in dependencies):
                expected = sum(nodes[name][measure] for name in dependencies)
                audit.check(close(expected, record.get(f"charged_fit_{measure}")),
                            f"{task_id}/{method}: charged fit {measure}", "cost")
                serialization = record.get("serialization_cost_node")
                audit.check(serialization in nodes, f"{task_id}/{method}: serialization compute node", "cost")
                if serialization in nodes:
                    audit.check(close(expected + nodes[serialization][measure], record.get(f"charged_{measure}")),
                                f"{task_id}/{method}: total charged {measure}", "cost")
        for fallback in record.get("fallbacks", []):
            fallback_records.append({"task_id": task_id, "method": method, "event": fallback})
            if fallback.get("stage") == "D_prediction" and isinstance(fallback.get("horizon"), int):
                h = fallback["horizon"]
                audit.array(predictions[f"D__{method}"][:, h], predictions["D__ridge"][:, h],
                            f"{task_id}/{method}/{h}: logged D fallback equals frozen baseline", category="fallback")
        if record.get("fallback"):
            fallback_records.append({"task_id": task_id, "method": method,
                                     "event": {"fallback": record["fallback"], "exception": record.get("exception")}})
        if method == "ridge":
            candidates = record.get("candidates", [])
            audit.check([row.get("alpha") for row in candidates] == config["baseline_alpha_grid"],
                        f"{task_id}: complete baseline penalty grid", "selection")
            finite_candidates = [candidate for candidate in candidates if finite(candidate.get("inner_a_mse"))]
            audit.check(bool(finite_candidates), f"{task_id}: missing inner-A ridge scores", "selection")
            if finite_candidates:
                first = min(finite_candidates, key=lambda row: row["inner_a_mse"])
                audit.check(record.get("selected_alpha") == first["alpha"],
                            f"{task_id}: ridge not first inner-A minimum", "selection")
            training_rows, validation_rows = record.get("inner_training_rows", []), record.get("inner_validation_rows", [])
            audit.check(bool(training_rows) and bool(validation_rows)
                        and not (set(training_rows) & set(validation_rows)),
                        f"{task_id}: inner-A row separation", "chronology")
            if task["suite"] == "natural":
                audit.array(training_rows, np.flatnonzero(arrays["A_inner_train_mask"]),
                            f"{task_id}: actual purged inner-A fitting rows", category="chronology")
                audit.array(validation_rows, np.flatnonzero(arrays["A_inner_validation_mask"]),
                            f"{task_id}: actual chronological inner-A validation rows", category="chronology")
            elif training_rows and validation_rows:
                audit.check(not (set(arrays["A_groups"][training_rows]) & set(arrays["A_groups"][validation_rows])),
                            f"{task_id}: actual inner-A episode separation", "chronology")
        if method in ("raw_rbf", "raw_hgb") and not record.get("fallback"):
            candidates = record.get("candidates", [])
            expected_count = (len(config["raw_rbf_bandwidth_factors"]) * len(config["baseline_alpha_grid"])
                              if method == "raw_rbf" else
                              len(config["raw_hgb_learning_rates"]) * len(config["raw_hgb_max_leaf_nodes"]))
            audit.check(len(candidates) == expected_count, f"{task_id}/{method}: complete B grid", "selection")
            if candidates:
                first = min(candidates, key=lambda row: row["b_mse"])
                audit.check(record.get("selected") == first,
                            f"{task_id}/{method}: not first B minimum", "selection")
        if method == "seasonal_naive":
            lag_indices = int(task["L"]) - int(task["p"]) + ((np.asarray(task["offsets"]) - 1) % int(task["p"]))
            for split in ("C", "D"):
                audit.array(predictions[f"{split}__{method}"], arrays[f"{split}_X"][:, lag_indices],
                            f"{task_id}/{split}: horizon-specific seasonal naive", category="selection")
        if method.startswith(("fixed_", "gate_", "reconstruction_")):
            horizons = record.get("horizons", [])
            audit.check(len(horizons) == len(task["offsets"]),
                        f"{task_id}/{method}: incomplete horizon records", "selection")
            for index, horizon in enumerate(horizons):
                candidates, selected = horizon.get("candidates", []), horizon.get("selected", {})
                audit.check(bool(candidates), f"{task_id}/{method}/{index}: missing C grid", "selection")
                expected_ids = ["zero"] + [f"alpha={float(alpha):g},gamma={float(gamma):g}"
                                           for alpha in config["alpha_grid"] for gamma in config["gamma_grid"]
                                           if float(gamma) != 0]
                audit.check([row.get("candidate_id") for row in candidates] == expected_ids,
                            f"{task_id}/{method}/{index}: complete ordered C grid", "selection")
                eligible = [candidate for candidate in candidates if finite(candidate.get("c_mse"))]
                if eligible:
                    first = min(eligible, key=lambda row: row["c_mse"])
                    audit.check(selected.get("candidate_id") == first.get("candidate_id"),
                                f"{task_id}/{method}/{index}: not first C minimum", "selection")
                    audit.check(close(selected.get("c_mse"), float(np.mean(np.square(
                        arrays["C_Y"][:, index] - prediction_c[:, index])))),
                        f"{task_id}/{method}/{index}: selected C loss", "selection")
                else:
                    audit.check(False, f"{task_id}/{method}/{index}: no finite C grid entry", "selection")
                if candidates:
                    audit.check(candidates[0].get("gamma") == 0,
                                f"{task_id}/{method}/{index}: zero correction must be first", "selection")
                if selected.get("gamma") == 0:
                    for split in ("C", "D"):
                        audit.array(predictions[f"{split}__{method}"][:, index],
                                    predictions[f"{split}__ridge"][:, index],
                                    f"{task_id}/{method}/{index}: exact zero-correction {split}",
                                    category="zero_correction")
                for optimization in horizon.get("optimizations", []):
                    evaluations = optimization.get("evaluations", [])
                    audit.check(optimization.get("nfev") == len(evaluations),
                                f"{task_id}/{method}/{index}: evaluation count", "optimization")
                    audit.check(len(evaluations) <= config["max_objective_evaluations"],
                                f"{task_id}/{method}/{index}: evaluation budget", "optimization")
                    bound = np.log(config["parameter_bound"])
                    blocks = len(config["blocks"])
                    expected_dimension = blocks if method.startswith("gate_") else 2 * blocks
                    audit.check(optimization.get("parameter_count") == expected_dimension,
                                f"{task_id}/{method}/{index}: parameter count", "optimization")
                    for evaluation in evaluations + [optimization]:
                        xi, eta = np.asarray(evaluation.get("xi"), float), np.asarray(evaluation.get("eta"), float)
                        audit.check(xi.shape == eta.shape == (blocks,)
                                    and np.isfinite(xi).all() and np.isfinite(eta).all()
                                    and np.all(np.abs(xi) <= bound + 1e-12)
                                    and np.all(np.abs(eta) <= bound + 1e-12),
                                    f"{task_id}/{method}/{index}: bounded finite parameters", "optimization")
                        if method.startswith("gate_"):
                            audit.check(np.array_equal(xi, np.zeros(blocks)),
                                        f"{task_id}/{method}/{index}: gate arm changed reconstruction", "optimization")
                    initial = optimization.get("initial_normalized_b_mse")
                    retained = optimization.get("best_normalized_b_mse")
                    if finite(initial) and finite(retained):
                        audit.check(retained <= initial + 1e-12,
                                    f"{task_id}/{method}/{index}: identity incumbent lost", "optimization")
                    optimization_records.append({
                        "task_id": task_id, "method": method, "horizon": task["offsets"][index],
                        "alpha": optimization.get("alpha"), "nfev": optimization.get("nfev"),
                        "status": optimization.get("status"), "converged": optimization.get("converged"),
                        "initial_normalized_b_mse": initial, "best_normalized_b_mse": retained,
                        "invalid_evaluation_count": sum(evaluation.get("exception") is not None for evaluation in evaluations),
                    })
    policy = method_records.get("nonkahm_policy", {})
    candidates = policy.get("candidates", [])
    if candidates:
        candidate_methods = [row["method"] for row in candidates]
        losses = {method: float(np.mean(np.square(arrays["C_Y"] - predictions[f"C__{method}"])))
                  for method in candidate_methods}
        first = min(candidate_methods, key=lambda method: losses[method])
        selected = policy.get("selected_method")
        audit.check(selected == first, f"{task_id}: policy not first whole-procedure C minimum", "selection")
        audit.check(candidate_methods[0] == "ridge" and all("kahm" not in name for name in candidate_methods),
                    f"{task_id}: policy candidate admissibility", "selection")
        if selected in methods:
            for split in ("C", "D"):
                audit.array(predictions[f"{split}__nonkahm_policy"], predictions[f"{split}__{selected}"],
                            f"{task_id}: selected policy {split}", category="selection")
        expected_union = set().union(*(set(method_records[name].get("cost_dependencies", []))
                                      for name in candidate_methods))
        actual_dependencies = set(policy.get("cost_dependencies", []))
        audit.check(expected_union <= actual_dependencies,
                    f"{task_id}: policy undercharges candidate compute", "cost")
    else:
        audit.check(False, f"{task_id}: missing policy C roster", "selection")
    return optimization_records, fallback_records


def summarize(protocol, metrics, horizons, optimizations, fallbacks, unavailable=()):
    methods, tasks = protocol["methods"], protocol["tasks"]
    lookup = {(row["task_id"], row["method"]): row for row in metrics}
    aggregates, primary, collection_rows, natural_comparisons, leave_one_out = [], [], [], [], []
    for law in protocol["laws"]:
        roster = [task for task in tasks if task["suite"] == "synthetic" and task["law"] == law]
        for method in methods:
            rows = [lookup.get((task["task_id"], method)) for task in roster]
            complete = len(rows) == len(protocol["seeds"]) and all(row is not None for row in rows)
            aggregates.append({
                "suite": "synthetic", "group": law, "method": method,
                "complete": complete, "task_count": sum(row is not None for row in rows),
                "mean_loss": float(np.mean([row["mse"] for row in rows])) if complete else None,
                "loss_units": "raw MSE; equal repetition means",
            })
        for control in protocol["primary_controls"]:
            candidate_rows = [lookup.get((task["task_id"], protocol["primary_candidate"])) for task in roster]
            control_rows = [lookup.get((task["task_id"], control)) for task in roster]
            complete = (len(roster) == 12 and all(row is not None for row in candidate_rows + control_rows))
            row = {"law": law, "control": control, "candidate": protocol["primary_candidate"],
                   "complete": complete, "seeds": [task["seed"] for task in roster]}
            if complete:
                row.update(paired_interval([record["mse"] for record in control_rows],
                                           [record["mse"] for record in candidate_rows],
                                           family_size=protocol["primary_family_size"],
                                           alpha=protocol["family_alpha"]))
            primary.append(row)
    natural_tasks = [task for task in tasks if task["suite"] == "natural"]
    natural_source_means = {}
    natural_source_rosters = {source: [task for task in natural_tasks if task["source_key"] == source]
                              for source in sorted({task["source_key"] for task in natural_tasks})}
    for method in methods:
        rows = [lookup.get((task["task_id"], method)) for task in natural_tasks]
        complete = len(rows) == 10 and all(row is not None for row in rows)
        mean, _ = (equal_collection_mean(rows) if complete else (None, {}))
        aggregates.append({"suite": "natural", "group": "equal_collection", "method": method,
                           "complete": complete, "task_count": sum(row is not None for row in rows),
                           "mean_loss": mean, "loss_units": "A-seasonal scaled MSE; equal collection then series"})
        available_rows = [row for row in rows if row is not None]
        available_mean, by_collection = (equal_collection_mean(available_rows) if available_rows else (None, {}))
        natural_source_means[method] = by_collection
        aggregates.append({"suite": "natural", "group": "available_collection", "method": method,
                           "complete": complete, "task_count": len(available_rows),
                           "mean_loss": available_mean,
                           "loss_units": "supplemental available-case A-scaled MSE; equal available collection then available series"})
        for source, source_roster in natural_source_rosters.items():
            source_rows = [lookup.get((task["task_id"], method)) for task in source_roster]
            source_complete = all(row is not None for row in source_rows)
            available_count = sum(row is not None for row in source_rows)
            collection_rows.append({"source_key": source, "method": method,
                                    "mean_scaled_mse": by_collection.get(source) if source_complete else None,
                                    "complete": source_complete, "series_count": len(source_roster),
                                    "available_series_count": available_count})
    candidate = protocol["primary_candidate"]
    for control in methods:
        if control == candidate:
            continue
        controls, candidates = natural_source_means[control], natural_source_means[candidate]
        if controls and controls.keys() == candidates.keys():
            control_mean, candidate_mean = np.mean(list(controls.values())), np.mean(list(candidates.values()))
            paired_tasks = [task for task in natural_tasks if (task["task_id"], control) in lookup
                            and (task["task_id"], candidate) in lookup]
            series_differences = [lookup[(task["task_id"], control)]["scaled_mse"]
                                  - lookup[(task["task_id"], candidate)]["scaled_mse"] for task in paired_tasks]
            natural_comparisons.append({
                "control": control, "candidate": candidate,
                "mean_control_scaled_mse": float(control_mean),
                "mean_candidate_scaled_mse": float(candidate_mean),
                "mean_difference": float(control_mean - candidate_mean),
                "relative_improvement_percent": float(100 * (control_mean - candidate_mean) / control_mean),
                "source_wins": sum(candidates[source] < controls[source] for source in controls),
                "source_count": len(controls), "series_wins": sum(value > 0 for value in series_differences),
                "series_count": len(series_differences),
                "inference": "supplemental available-case descriptive only; registered four-collection aggregate undefined if incomplete",
            })
            for excluded in sorted(controls):
                retained = [source for source in controls if source != excluded]
                a, b = np.mean([controls[source] for source in retained]), np.mean([candidates[source] for source in retained])
                leave_one_out.append({"control": control, "candidate": candidate,
                                      "excluded_collection": excluded, "mean_control_scaled_mse": float(a),
                                      "mean_candidate_scaled_mse": float(b), "mean_difference": float(a - b),
                                      "relative_improvement_percent": float(100 * (a - b) / a),
                                      "inference": "leave one available collection out; supplemental descriptive only"})
    costs = []
    for suite in ("synthetic", "natural"):
        for method in methods:
            rows = [row for row in metrics if row["suite"] == suite and row["method"] == method]
            if rows:
                cost = {"suite": suite, "method": method, "tasks": len(rows)}
                for quantity in ("charged_wall_seconds", "charged_cpu_seconds", "selected_serialized_bytes"):
                    values = [row[quantity] for row in rows if finite(row[quantity])]
                    cost[f"mean_{quantity}"] = float(np.mean(values)) if len(values) == len(rows) else None
                    cost[f"median_{quantity}"] = float(np.median(values)) if len(values) == len(rows) else None
                costs.append(cost)
    primary_outcomes = Counter(row.get("conclusion", "incomplete") for row in primary)
    supported_laws = [law for law in protocol["laws"] if all(
        row.get("conclusion") == "positive" for row in primary if row["law"] == law)]
    return {
        "schema_version": protocol["schema_version"],
        "expected_tasks": len(tasks), "expected_method_outputs": len(tasks) * len(methods),
        "available_method_outputs": len(metrics), "complete_roster": len(metrics) == len(tasks) * len(methods),
        "documented_unavailable_tasks": list(unavailable),
        "method_aggregates": aggregates, "primary_contrasts": primary,
        "primary_outcome_counts": dict(primary_outcomes), "all_six_positive_laws": supported_laws,
        "natural_collection_metrics": collection_rows,
        "natural_available_comparisons": natural_comparisons,
        "natural_available_leave_one_collection_out": leave_one_out, "cost_aggregates": costs,
        "optimization_count": len(optimizations),
        "objective_evaluations": sum(row["nfev"] or 0 for row in optimizations),
        "optimization_statuses": dict(Counter(str(row["status"]) for row in optimizations)),
        "optimization_converged_count": sum(row["converged"] is True for row in optimizations),
        "invalid_objective_evaluations": sum(row.get("invalid_evaluation_count", 0) for row in optimizations),
        "recorded_fallback_events": len(fallbacks),
        "interpretation": {
            "synthetic": "New eighteen-contrast family only; approximate paired-t inference over twelve independent repetitions per law.",
            "natural": "Ten-series/four-collection frozen roster. Unavailable tasks remain recorded; complete-roster aggregate is undefined if any fail. Separately labeled available-case descriptions have no population interpretation.",
            "cost": "Charged dependency sums are procedure costs; serialized bytes are selected-model storage, not resident memory or update latency.",
            "scope": "No pooling with the three-seed exploratory gate pilot or any historical experiment; no whole-paper coverage claim.",
        },
    }


def write_displays(output, summary, methods, laws):
    generated = output / "generated"
    generated.mkdir(exist_ok=True)
    aggregates = {(row["group"], row["method"]): row["mean_loss"] for row in summary["method_aggregates"]}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Procedure & Nonlinear AR & Van der Pol & Duffing & Natural available-case scaled MSE \\", r"\midrule"]
    for method in methods:
        values = [aggregates.get((law, method)) for law in laws] + [aggregates.get(("available_collection", method))]
        cells = [f"{value:.5g}" if finite(value) else "undefined" for value in values]
        label = LABELS.get(method, method.replace("_", r"\_"))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "method_table.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{llrrl}", r"\toprule",
             r"Law & Control & Gain (\%) & Difference interval & Outcome \\", r"\midrule"]
    for row in summary["primary_contrasts"]:
        if row["complete"]:
            law = {"nonlinear_ar": "AR", "vanderpol": "VDP", "duffing": "Duffing"}[row["law"]]
            label = LABELS[row["control"]]
            interval = f"[{row['interval_low']:.3g}, {row['interval_high']:.3g}]"
            lines.append(f"{law} & {label} & {row['relative_improvement_percent']:.2f} & {interval} & {row['conclusion']}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "primary_table.tex").write_text("\n".join(lines) + "\n")
    sources = sorted({row["source_key"] for row in summary["natural_collection_metrics"]})
    collection_lookup = {(row["source_key"], row["method"]): row
                         for row in summary["natural_collection_metrics"]}
    source_labels = {"dominick_weekly": "Dominick", "rideshare_price_hourly": "Rideshare",
                     "wikipedia_daily": "Wikipedia", "sunspot_daily": "Sunspot"}
    lines = [r"\begin{tabular}{l" + "r" * len(sources) + "}", r"\toprule",
             "Procedure & " + " & ".join(source_labels.get(source, source.replace("_", r"\_"))
                                            for source in sources) + r" \\", r"\midrule"]
    for method in methods:
        values = [collection_lookup.get((source, method), {}).get("mean_scaled_mse") for source in sources]
        lines.append(LABELS[method] + " & " + " & ".join(f"{value:.5g}" if finite(value) else "undefined"
                                                         for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "natural_table.tex").write_text("\n".join(lines) + "\n")
    cost_lookup = {(row["suite"], row["method"]): row for row in summary["cost_aggregates"]}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Procedure & Synthetic time (s) & Available natural time (s) & Synthetic state (KiB) & Available natural state (KiB) \\",
             r"\midrule"]
    for method in methods:
        synthetic = cost_lookup.get(("synthetic", method), {})
        natural = cost_lookup.get(("natural", method), {})
        values = [synthetic.get("mean_charged_wall_seconds"), natural.get("mean_charged_wall_seconds"),
                  synthetic.get("mean_selected_serialized_bytes"), natural.get("mean_selected_serialized_bytes")]
        cells = [f"{value / (1024 if index >= 2 else 1):.4g}" if finite(value) else "undefined"
                 for index, value in enumerate(values)]
        lines.append(LABELS[method] + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "cost_table.tex").write_text("\n".join(lines) + "\n")
    candidate_collections = [row for row in summary["natural_collection_metrics"]
                             if row["method"] == runner.PRIMARY_CANDIDATE]
    macros = {
        "DirectPlannedTasks": summary["expected_tasks"],
        "DirectAvailableTasks": summary["available_method_outputs"] // len(methods),
        "DirectAvailableOutputs": summary["available_method_outputs"],
        "DirectPlannedOutputs": summary["expected_method_outputs"],
        "DirectUnavailableTasks": len(summary["documented_unavailable_tasks"]),
        "DirectNaturalAvailableCollections": sum(row["available_series_count"] > 0 for row in candidate_collections),
        "DirectNaturalAvailableSeries": sum(row["available_series_count"] for row in candidate_collections),
        "DirectPrimaryPositive": summary["primary_outcome_counts"].get("positive", 0),
        "DirectPrimaryNegative": summary["primary_outcome_counts"].get("negative", 0),
        "DirectPrimaryUnresolved": summary["primary_outcome_counts"].get("unresolved", 0),
        "DirectOptimizations": summary["optimization_count"],
        "DirectObjectiveEvaluations": summary["objective_evaluations"],
        "DirectFallbackEvents": summary["recorded_fallback_events"],
    }
    (generated / "result_macros.tex").write_text(
        "% Generated from the preserved direct-reconstruction summary; no manual numerical edits.\n"
        + "\n".join("\\newcommand{\\" + name + "}{" + str(value) + "}" for name, value in macros.items()) + "\n")


def analyze(output, *, replay=True, live_source=False):
    output = Path(output).resolve()
    audit = Audit()
    try:
        runner.verify_freeze(output, live_source=live_source or replay)
        audit.check(True, "Frozen evidence hashes verified", "hash")
    except Exception as error:
        audit.check(False, f"Freeze verification: {error}", "hash")
    protocol = json.loads((output / "protocol.json").read_text())
    execution_path = output / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
    audit.check(execution.get("status") in ("complete", "complete_with_failures"),
                "Execution did not finish the frozen roster", "roster")
    methods = protocol["methods"]
    audit.check(len(methods) == 17 and len(set(methods)) == 17, "Expected seventeen unique methods", "roster")
    audit.check(len(protocol["tasks"]) == 46, "Expected forty-six tasks", "roster")
    audit.check(protocol["seeds"] == list(runner.STUDY_SEEDS), "Changed primary seed roster", "roster")
    metrics, horizons, optimizations, fallbacks, unavailable = [], [], [], [], []
    model = runner.load_model() if replay else None
    for task in protocol["tasks"]:
        task_id = task["task_id"]
        destination = output / "tasks" / task_id
        if not (destination / "complete.json").exists():
            failure_path = destination / "failure.json"
            if failure_path.exists():
                failure = json.loads(failure_path.read_text())
                with np.load(output / task["prepared_path"], allow_pickle=False) as data:
                    arrays = {key: data[key] for key in data.files}
                audit_inputs(audit, task, arrays)
                empty = [split for split in runner.SPLITS if len(arrays[f"{split}_X"]) == 0]
                documented = (task["suite"] == "natural" and bool(empty)
                              and failure.get("exception") == "ValueError"
                              and failure.get("message") == "Frozen A-eligible task has an empty later split; retained as an unavailable outcome")
                audit.check(documented, f"{task_id}: unexpected task failure", "roster")
                audit.check(json.loads((destination / "task.json").read_text()) == task,
                            f"{task_id}: unavailable task metadata differs from freeze", "hash")
                audit.check(not (destination / "predictions.npz").exists(),
                            f"{task_id}: unavailable task must not invent predictions", "roster")
                unavailable.append({"task_id": task_id, "source_key": task.get("source_key"),
                                    "empty_splits": empty, "reason": failure.get("message"),
                                    "anticipated_before_freeze": True})
            else:
                audit.check(False, f"Missing complete task {task_id}", "roster")
            continue
        completed = json.loads((destination / "complete.json").read_text())
        for filename, key in (("predictions.npz", "predictions_sha256"),
                              ("diagnostics.json", "diagnostics_sha256"), ("task.json", "task_sha256"),
                              ("selected_models.zip", "selected_models_sha256")):
            audit.check(runner.sha256_file(destination / filename) == completed[key],
                        f"{task_id}/{filename}: completed artifact hash", "hash")
        audit.check(json.loads((destination / "task.json").read_text()) == task,
                    f"{task_id}: saved task metadata differs from freeze", "hash")
        with np.load(output / task["prepared_path"], allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        audit_inputs(audit, task, arrays)
        with np.load(destination / "predictions.npz", allow_pickle=False) as stored:
            predictions = {key: stored[key] for key in stored.files}
        diagnostics = json.loads((destination / "diagnostics.json").read_text())
        audit.array(predictions["C_targets"], arrays["C_Y"], f"{task_id}: saved C targets", category="loss")
        audit.array(predictions["D_targets"], arrays["D_Y"], f"{task_id}: saved D targets", category="loss")
        expected_keys = {f"{split}__{method}" for split in ("C", "D") for method in methods}
        audit.check(set(predictions) == expected_keys | {"C_targets", "D_targets"},
                    f"{task_id}: exact forecast roster", "roster")
        if not expected_keys <= predictions.keys():
            continue
        task_optimizations, task_fallbacks = audit_diagnostics(
            audit, task, arrays, predictions, diagnostics, protocol["model_config"], methods
        )
        optimizations.extend(task_optimizations)
        fallbacks.extend(task_fallbacks)
        for method in methods:
            try:
                loss = score(arrays["D_Y"], predictions[f"D__{method}"], arrays["D_groups"], task["offsets"])
            except Exception as error:
                audit.check(False, f"{task_id}/{method}: loss {error}", "loss")
                continue
            method_record = diagnostics["methods"].get(method, {})
            serialized = diagnostics["serialized_models"].get(method, {})
            path = destination / serialized.get("archive", "missing")
            audit.check(path.is_file(), f"{task_id}/{method}: selected model missing", "replay")
            if path.is_file():
                with zipfile.ZipFile(path) as archive:
                    expected_members = {f"{name}.pkl" for name in methods}
                    audit.check(set(archive.namelist()) == expected_members,
                                f"{task_id}/{method}: selected archive roster", "replay")
                    contents = archive.read(serialized["member"])
                audit.check(len(contents) == serialized.get("bytes")
                            and hashlib.sha256(contents).hexdigest() == serialized.get("sha256"),
                            f"{task_id}/{method}: selected model size/hash", "replay")
                if replay:
                    try:
                        for split in ("C", "D"):
                            reproduced = model.predict_serialized(contents, arrays[f"{split}_X"])
                            audit.array(reproduced, predictions[f"{split}__{method}"],
                                        f"{task_id}/{method}: serialized {split} forecast replay", exact=False, category="replay")
                    except Exception as error:
                        audit.check(False, f"{task_id}/{method}: selected replay {error}", "replay")
            row = {
                "task_id": task_id, "suite": task["suite"], "law": task.get("law", ""),
                "source_key": task.get("source_key", ""), "series_id": task.get("series_id", ""),
                "seed": task.get("seed", ""), "method": method,
                "mse": loss["mse"], "scaled_mse": loss["mse"] / task["loss_scale"],
                "c_mse": float(np.mean(np.square(arrays["C_Y"] - predictions[f"C__{method}"]))),
                "n_groups": loss["n_groups"], "n_origins": loss["n_origins"],
                "charged_wall_seconds": method_record.get("charged_wall_seconds"),
                "charged_cpu_seconds": method_record.get("charged_cpu_seconds"),
                "selected_serialized_bytes": serialized.get("bytes"),
            }
            metrics.append(row)
            for offset, value in loss["horizon_mse"].items():
                horizons.append({"task_id": task_id, "suite": task["suite"],
                                 "law": task.get("law", ""), "source_key": task.get("source_key", ""),
                                 "method": method, "horizon": int(offset), "mse": value,
                                 "scaled_mse": value / task["loss_scale"]})
    summary = summarize(protocol, metrics, horizons, optimizations, fallbacks, unavailable)
    audit.check(len(metrics) + len(unavailable) * len(methods) == len(protocol["tasks"]) * len(methods),
                "Unaccounted forecast outcome in declared roster", "roster")
    audit.check(len(summary["primary_contrasts"]) == 18, "Primary family must contain exactly eighteen contrasts", "roster")
    summary["audit_passed"] = not audit.failures
    runner.write_json(output / "summary.json", summary)
    if metrics:
        write_csv(output / "task_metrics.csv", metrics)
    if horizons:
        write_csv(output / "horizon_metrics.csv", horizons)
    write_csv(output / "primary_contrasts.csv", summary["primary_contrasts"], [
        "law", "control", "candidate", "complete", "n_repetitions", "degrees_of_freedom",
        "family_size", "family_alpha", "mean_control_mse", "mean_candidate_mse", "mean_difference",
        "sd_difference", "standard_error", "critical_t", "interval_low", "interval_high",
        "relative_improvement_percent", "positive_repetitions", "negative_repetitions", "tied_repetitions", "conclusion",
    ])
    write_csv(output / "method_aggregates.csv", summary["method_aggregates"])
    if summary["natural_collection_metrics"]:
        write_csv(output / "natural_collection_metrics.csv", summary["natural_collection_metrics"])
    if summary["natural_available_comparisons"]:
        write_csv(output / "natural_available_comparisons.csv", summary["natural_available_comparisons"])
        write_csv(output / "natural_available_leave_one_collection_out.csv", summary["natural_available_leave_one_collection_out"])
    if summary["cost_aggregates"]:
        write_csv(output / "cost_aggregates.csv", summary["cost_aggregates"])
    if optimizations:
        write_csv(output / "optimization_summary.csv", optimizations)
    runner.write_json(output / "fallback_events.json", fallbacks)
    write_displays(output, summary, methods, protocol["laws"])
    report = {
        "passed": not audit.failures, "checks": dict(audit.counts), "failures": audit.failures,
        "expected_tasks": len(protocol["tasks"]), "available_method_outputs": len(metrics),
        "complete_roster": summary["complete_roster"], "documented_unavailable_tasks": unavailable,
        "selected_models_replayed": bool(replay),
        "selected_model_count": len(metrics),
        "replayed_forecast_matrices": 2 * len(metrics) if replay else 0,
        "public_freeze_commit": execution.get("public_freeze_commit"),
        "scope": [
            "Checks frozen hashes, saved raw histories/targets and chronology, losses, recorded selection, budget accounting, and selected-state replay.",
            "Does not independently refit every candidate or certify that a public commit predates an unrecorded private evaluation.",
            "Statistical intervals require independent repetitions and a paired-t approximation; they cover this new family only.",
            "Four natural collections support descriptive transfer evidence only; no window-level inferential sample size is used.",
        ],
    }
    runner.write_json(output / "audit_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-replay", action="store_true", help="explicitly weaken audit; not the publication audit")
    parser.add_argument("--verify-live-source", action="store_true")
    arguments = parser.parse_args()
    report = analyze(arguments.output, replay=not arguments.skip_replay,
                     live_source=arguments.verify_live_source)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
