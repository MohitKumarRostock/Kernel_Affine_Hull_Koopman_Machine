"""Independent prediction, split, selection and cost audit for target geometry.

The primary inference unit is one complete fresh synthetic repetition, not an
origin, horizon, episode or parameter setting. Archive results are descriptive
and equally weighted first by series within collection, then by collection.
Missing task outputs make full-roster means undefined; successful fallbacks
remain real predictions and retain their charged computation.

This auditor does not fit forecasting models or import the experiment runner.
It reconstructs inputs from the pinned preparation/generator, verifies their
hashes, and recomputes losses and diagnostic identities from saved forecasts.
Validation selection is checked against the complete saved trial grid; the
individual validation trial predictions are not independently refitted here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t

from . import data
from extensions.predictive_geometry.study.datasets import make_forecast_split

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def expected_methods(cfg):
    return ["ridge", "raw_rbf", "naive", "joint_kahm_ridge"] + [
        f"{stage}_{gate}" for stage in ("fixed_is", "fixed_oof", "learned_oof")
        for gate in cfg["gates"]]


def expected_roster(cfg):
    result = []
    for law in cfg["synthetic"]["laws"]:
        for seed in cfg["synthetic"]["repetition_seeds"]:
            result.append({"task_id": f"synthetic__{law}__{seed}", "phase": "synthetic",
                           "law": law, "source_id": law, "seed": seed, "loss_scale": 1.0,
                           "context": cfg["synthetic"]["history"],
                           "horizon": max(cfg["synthetic"]["horizons"])})
    for task in json.loads(data.ROSTER_PATH.read_text())["tasks"]:
        result.append({**task, "phase": "natural", "task_id": "natural__" + task["base_task_id"],
                       "seed": cfg["natural"]["seed"]})
    return result


def reconstruct_inputs(task, cfg):
    if task["phase"] == "natural":
        return data.load_natural(task)
    settings = cfg["synthetic"]
    law_index = settings["laws"].index(task["law"])
    result = {}
    counts = (settings["training_episodes"], settings["validation_episodes"], settings["test_episodes"])
    for split_index, (split, count) in enumerate(zip(("train", "validation", "test"), counts)):
        seed = int(np.random.SeedSequence([task["seed"], law_index, split_index,
                                          settings["seed_namespace"]]).generate_state(1)[0])
        generated = make_forecast_split(task["law"], count, seed, history=settings["history"],
                                        horizons=tuple(settings["horizons"]), stride=settings["stride"])
        result.update({f"{key}_{split}": generated[key] for key in ("X", "Y", "groups")})
        result[f"origins_{split}"] = np.tile(generated["origins"], count)
    return result


def _close(actual, expected, message, *, rtol=2e-11, atol=1e-13):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False):
        raise ValueError(message)


def _model_hash(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    hasher = hashlib.sha256(str(value.shape).encode("ascii"))
    hasher.update(value.tobytes())
    return hasher.hexdigest()


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    columns = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def audit_trials(name, entry, cfg):
    selected = entry["selected"]
    if entry["status"] == "fallback" or name == "naive":
        return
    trials = selected.get("trials", [])
    if not trials:
        raise ValueError(f"Missing validation trials: {name}")
    losses = np.asarray([trial["validation_mse"] for trial in trials], dtype=float)
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("Nonfinite/negative validation loss")
    if name.startswith(("fixed_is_", "fixed_oof_", "learned_oof_")):
        expected = [(None, 0.0)] + [(alpha, gamma) for alpha in cfg["correction_alphas"]
                                  for gamma in cfg["gammas"] if gamma != 0]
        actual = [(trial["alpha"], trial["gamma"]) for trial in trials]
        if actual != expected:
            raise ValueError("Correction validation grid/order differs from frozen specification")
        winner = trials[int(np.argmin(losses))]
        if selected["alpha"] != winner["alpha"] or selected["gamma"] != winner["gamma"]:
            raise ValueError("Correction selection differs from first-minimum validation rule")
        _close(selected["validation_mse"], winner["validation_mse"], "Selected validation loss mismatch")
    else:
        expected = cfg["correction_alphas"] if name == "joint_kahm_ridge" else cfg["base_alphas"]
        if [trial["alpha"] for trial in trials] != expected:
            raise ValueError("Raw/joint alpha grid differs from frozen specification")
        if selected["alpha"] != trials[int(np.argmin(losses))]["alpha"]:
            raise ValueError("Raw/joint alpha selection differs from first-minimum rule")


def audit_cost(name, entry, nodes):
    dependencies = entry["dependencies"]
    if len(dependencies) != len(set(dependencies)) or any(key not in nodes for key in dependencies):
        raise ValueError("Duplicate or absent charged computation node")
    for key in dependencies:
        for quantity in ("wall_seconds", "cpu_seconds"):
            if not np.isfinite(nodes[key][quantity]) or nodes[key][quantity] < 0:
                raise ValueError("Invalid measured computation cost")
    _close(entry["pipeline_wall_seconds"], sum(nodes[key]["wall_seconds"] for key in dependencies),
           "Wall-time dependency sum mismatch")
    _close(entry["pipeline_cpu_seconds"], sum(nodes[key]["cpu_seconds"] for key in dependencies),
           "CPU-time dependency sum mismatch")
    if entry["status"] == "fallback":
        required = {"ridge:search", "ridge:predict", name + ":serialize"}
        # Every attempted node for the failed method/bank remains charged.
        if name.startswith(("fixed_is_", "fixed_oof_", "learned_oof_")):
            geometry, mode, gate = name.rsplit("_", 2)
            required |= {"ridge:train_predict", "ridge:validation_predict"}
            if mode == "oof":
                required.add("oof:fit")
            if geometry == "learned":
                required.add("metric:fit")
            key = f"bank:{geometry}:{gate}:"
            required |= {n for n in nodes if n.startswith(key) or n.startswith(name + ":")}
        elif name == "raw_rbf":
            required |= {n for n in nodes if n.startswith("raw_rbf:")}
        elif name == "joint_kahm_ridge":
            required |= {n for n in nodes if n.startswith("bank:fixed:kahm:") or n.startswith(name + ":")}
        else:
            raise ValueError("Unexpected baseline fallback")
        if set(dependencies) != required:
            raise ValueError("Fallback silently omits or adds computation")
        return
    if name == "ridge":
        required = {"ridge:search", "ridge:predict", "ridge:serialize"}
    elif name == "raw_rbf":
        required = {"raw_rbf:search", "raw_rbf:predict", "raw_rbf:serialize"}
    elif name == "naive":
        required = {"naive:predict"}
    elif name == "joint_kahm_ridge":
        required = {f"bank:fixed:kahm:{part}" for part in ("fit", "train", "validation", "test")}
        required |= {name + ":" + part for part in ("train_features", "search", "predict", "serialize")}
    else:
        geometry, mode, gate = name.rsplit("_", 2)
        required = {f"ridge:{part}" for part in ("search", "train_predict", "validation_predict", "predict")}
        required |= {f"bank:{geometry}:{gate}:{part}" for part in ("fit", "train", "validation", "test")}
        required |= {name + ":search", name + ":serialize"}
        if mode == "oof":
            required.add("oof:fit")
        if geometry == "learned":
            required.add("metric:fit")
        if entry["selected"]["gamma"] != 0:
            required.add(name + ":predict")
    if set(dependencies) != required:
        raise ValueError(f"Charged dependencies differ from complete procedure: {name}; "
                         f"missing={required-set(dependencies)}, extra={set(dependencies)-required}")


def audit_oof(report, arrays, saved, cfg):
    task = report["task"]
    n = len(arrays["X_train"])
    all_eligible, all_queries, failed_queries = [], [], []
    fold_ids = [fold["fold"] for fold in report["oof_folds"]]
    fold_count = (cfg["synthetic"]["outer_folds"] if task["phase"] == "synthetic"
                  else cfg["natural"]["forward_folds"])
    if fold_ids != list(range(fold_count)):
        raise ValueError("OOF fold roster/order differs from the frozen design")
    for fold in report["oof_folds"]:
        query = np.asarray(fold["query_indices"], dtype=int)
        donors = np.asarray(fold["donor_indices"], dtype=int)
        if len(np.unique(query)) != len(query) or len(np.unique(donors)) != len(donors):
            raise ValueError("OOF fold repeats rows")
        if (np.any(query < 0) or np.any(query >= n) or np.any(donors < 0) or np.any(donors >= n)
                or np.intersect1d(query, donors).size):
            raise ValueError("OOF query/donor index invalidity or overlap")
        all_queries.extend(query.tolist())
        if task["phase"] == "synthetic":
            groups = arrays["groups_train"]
            expected_query = np.flatnonzero(groups % cfg["synthetic"]["outer_folds"] == fold["fold"])
            expected_donors = np.flatnonzero(~np.isin(groups, groups[query]))
        else:
            origins = arrays["origins_train"]
            first = max(1, int(np.ceil(cfg["natural"]["initial_fraction"] * n)))
            expected_query = np.array_split(np.arange(first, n), cfg["natural"]["forward_folds"])[fold["fold"]]
            expected_donors = np.flatnonzero(origins + task["horizon"] <= origins[query[0]])
        np.testing.assert_array_equal(query, expected_query, err_msg="OOF query roster changed")
        np.testing.assert_array_equal(donors, expected_donors, err_msg="OOF donor roster changed")
        if fold["eligible"] != (len(donors) >= 8):
            raise ValueError("OOF eligibility was changed by fitted outcomes")
        if not fold["eligible"]:
            if fold["status"] != "insufficient_donors":
                raise ValueError("Unexplained omitted OOF fold")
            continue
        all_eligible.extend(query.tolist())
        if fold["status"] == "failed":
            if not fold.get("error"):
                raise ValueError("Donor failure lacks an explicit error record")
            failed_queries.extend(query.tolist())
            continue
        if fold["status"] != "ok":
            raise ValueError("Unknown OOF donor status")
        inner_fit = np.asarray(fold["inner_fit_indices"], dtype=int)
        inner_valid = np.asarray(fold["inner_validation_indices"], dtype=int)
        if (not np.isin(inner_fit, donors).all() or not np.isin(inner_valid, donors).all()
                or np.intersect1d(inner_fit, inner_valid).size):
            raise ValueError("Inner tuning uses observations outside its donor set")
        if task["phase"] == "synthetic":
            if np.intersect1d(groups[inner_fit], groups[inner_valid]).size:
                raise ValueError("Inner synthetic tuning crosses episode boundary")
        elif len(inner_fit) and len(inner_valid):
            if np.max(origins[inner_fit] + task["horizon"]) > np.min(origins[inner_valid]):
                raise ValueError("An inner donor training label was unavailable at its validation origin")
        if fold["tuning"] == "donor_only_inner_selection":
            trials = fold["trials"]
            if [entry["alpha"] for entry in trials] != cfg["base_alphas"]:
                raise ValueError("OOF donor alpha grid differs from protocol")
            winner = min(trials, key=lambda entry: entry["validation_mse"])
            if winner["alpha"] != fold["alpha"]:
                raise ValueError("OOF donor selection differs from inner validation rule")
        elif fold["alpha"] != 1.0 or fold["trials"]:
            raise ValueError("Invalid fixed-alpha donor fallback")
    if len(all_queries) != len(set(all_queries)):
        raise ValueError("An OOF query appears in multiple folds")
    indices = np.asarray(saved["correction_indices"], dtype=int)
    np.testing.assert_array_equal(indices, np.sort(all_eligible), err_msg="Saved correction rows differ from outcome-independent support")
    if len(indices) != report["correction_rows"]:
        raise ValueError("Saved correction-row count differs from report")
    np.testing.assert_array_equal(saved["correction_targets"], arrays["Y_train"][indices])
    donor = saved["donor_predictions"]
    deployed = saved["final_base_train_predictions"]
    if donor.shape != deployed.shape or donor.shape != saved["correction_targets"].shape:
        raise ValueError("OOF and final-model residual targets are unaligned")
    if not np.isfinite(deployed).all():
        raise ValueError("Nonfinite final-model training prediction")
    residual_is = saved["correction_targets"] - deployed
    if len(indices):
        _close(np.mean(residual_is ** 2), report["in_sample_residual_mse"], "In-sample residual MSE mismatch")
    elif report["in_sample_residual_mse"] is not None:
        raise ValueError("An empty correction set has a fabricated residual diagnostic")
    if report.get("oof_failure"):
        if len(indices) >= 8 and (np.isfinite(donor).all() or not failed_queries):
            raise ValueError("OOF failure lacks failed donors and nonfinite prediction records")
        failed_mask = np.isin(indices, failed_queries)
        if not np.isnan(donor[failed_mask]).all() or not np.isfinite(donor[~failed_mask]).all():
            raise ValueError("OOF failure rows differ from preserved donor failure records")
        if report["donor_final_disagreement_mse"] is not None or report["oof_residual_mse"] is not None:
            raise ValueError("Failed OOF diagnostics were replaced by fabricated finite losses")
        for name, entry in report["methods"].items():
            if name.startswith(("fixed_oof_", "learned_oof_")) and entry["status"] != "fallback":
                raise ValueError("An OOF-dependent arm silently used failed donor targets")
        return indices, None, None
    if failed_queries or not np.isfinite(donor).all():
        raise ValueError("Unreported failed donor predictions")
    disagreement = float(np.mean((donor - deployed) ** 2))
    residual_oof = saved["correction_targets"] - donor
    _close(disagreement, report["donor_final_disagreement_mse"], "Residual transport diagnostic mismatch")
    _close(np.mean(residual_oof ** 2), report["oof_residual_mse"], "OOF residual MSE mismatch")
    return indices, residual_oof, disagreement


def audit_metric(report, arrays, indices, residuals, cfg):
    metric = report["metric"]
    if metric.get("fallback") == "identity":
        if not metric.get("error"):
            raise ValueError("Metric exception fallback lacks its error record")
        for gate in cfg["gates"]:
            bank = report["banks"].get("learned_" + gate, {})
            if "failure" not in bank:
                np.testing.assert_array_equal(bank["metric_weights"], np.ones(arrays["X_train"].shape[1]))
        return
    if residuals is None:
        raise ValueError("A learned metric was fitted despite failed OOF residuals")
    x = arrays["X_train"][indices]
    if (metric["input_sha256"] != _model_hash(x)
            or metric["residual_sha256"] != _model_hash(residuals)
            or metric["scaling_input_sha256"] != _model_hash(arrays["X_train"])):
        raise ValueError("Metric fitting data/hash differs from training-only residual data")
    ids = np.asarray(metric["fit_indices"], dtype=int)
    expected_ids = (np.sort(np.random.default_rng(report["task"]["seed"]).choice(
        len(x), cfg["metric"]["max_rows"], replace=False)) if len(x) > cfg["metric"]["max_rows"]
        else np.arange(len(x)))
    np.testing.assert_array_equal(ids, expected_ids, err_msg="Metric row subsample differs from frozen seed")
    mean = np.mean(arrays["X_train"], axis=0)
    scale = np.std(arrays["X_train"], axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    _close(metric["input_mean"], mean, "Metric input mean uses different data")
    _close(metric["input_scale"], scale, "Metric input scale uses different data")
    _close(metric["residual_mean"], residuals.mean(axis=0), "Metric target centering differs from training residuals")
    weights = np.asarray(metric["weights"], dtype=float)
    theta = np.asarray(metric["theta"], dtype=float)
    _close(weights, np.exp(theta), "Metric positive weights differ from centered log weights")
    _close(theta.sum(), 0.0, "Metric changes global log scale", atol=1e-10)
    if weights.max() / weights.min() > cfg["metric"]["max_weight_ratio"] * (1 + 1e-10):
        raise ValueError("Metric weight ratio exceeds its declared constraint")
    task = report["task"]
    if task["phase"] == "synthetic":
        groups = arrays["groups_train"][indices][ids]
        admissible = groups[:, None] != groups[None, :]
        if metric["position_exclusion"] or metric["exclusion_radius"] != 0:
            raise ValueError("Synthetic metric incorrectly imposes natural-series time distances")
    else:
        positions = arrays["origins_train"][indices][ids]
        admissible = np.abs(positions[:, None] - positions[None, :]) > task["context"] + task["horizon"]
        if not metric["position_exclusion"] or metric["exclusion_radius"] != task["context"] + task["horizon"]:
            raise ValueError("Natural metric omits its context-plus-horizon exclusion")
    np.fill_diagonal(admissible, False)
    np.testing.assert_array_equal(metric["donor_counts"], admissible.sum(axis=1),
                                  err_msg="Metric donor exclusion counts differ from declared rules")
    if metric["valid_queries"] != int(admissible.any(axis=1).sum()):
        raise ValueError("Metric usable-query count differs from its donor exclusions")
    if metric.get("initial_objective") is None:
        if not metric.get("fallback_reason"):
            raise ValueError("Missing metric objective without an explicit fallback")
        return
    z = (x[ids] - mean) / scale
    squared = np.square(z[:, None, :] - z[None, :, :])
    distances = squared.sum(axis=2)
    positive = distances[admissible & (distances > 1e-12)]
    bandwidth = float(np.median(positive))
    _close(metric["bandwidth_squared"], bandwidth, "Metric bandwidth changed from identity rule")
    centered = residuals - residuals.mean(axis=0)
    pooled = float(np.sqrt(np.mean(centered ** 2)))
    _close(metric["residual_pooled_scale"], pooled, "Metric changes relative target-horizon weights")
    targets = centered[ids] / pooled
    usable = np.flatnonzero(admissible.any(axis=1))

    def objective(w, logw):
        logs = -np.einsum("ijk,k->ij", squared, w) / (2 * bandwidth)
        logs = np.where(admissible, logs, -np.inf)[usable]
        p = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
        return float(np.mean((p @ targets - targets[usable]) ** 2)
                     + cfg["metric"]["regularization"] * np.mean(logw ** 2))

    _close(metric["initial_objective"], objective(np.ones(len(theta)), np.zeros(len(theta))),
           "Identity-metric objective cannot be independently reconstructed")
    _close(metric["final_objective"], objective(weights, theta),
           "Learned-metric objective cannot be independently reconstructed")
    if metric["final_objective"] > metric["initial_objective"] + 1e-11:
        raise ValueError("Retained metric increases its regularized training objective")
    for gate in cfg["gates"]:
        bank = report["banks"].get("learned_" + gate, {})
        if "failure" not in bank:
            _close(bank["metric_weights"], weights, "A geometric control receives a different learned metric")


def audit_task(directory, expected_task, cfg):
    report = json.loads((directory / "result.json").read_text())
    if report["task"] != expected_task:
        raise ValueError("Saved task metadata differs from the prespecified roster")
    arrays = reconstruct_inputs(expected_task, cfg)
    expected_hashes = {key: data.array_hash(value) for key, value in arrays.items()}
    if report["arrays_sha256"] != expected_hashes:
        raise ValueError("Data regeneration does not reproduce the saved input hashes")
    archive = directory / "predictions.npz"
    if data.digest(archive) != report["predictions_sha256"]:
        raise ValueError("Saved prediction archive hash changed")
    with np.load(archive, allow_pickle=False) as handle:
        saved = {key: handle[key] for key in handle.files}
    np.testing.assert_array_equal(saved["target"], arrays["Y_test"])
    np.testing.assert_array_equal(saved["validation_target"], arrays["Y_validation"])
    np.testing.assert_array_equal(saved["origins_test"], arrays["origins_test"])
    names = expected_methods(cfg)
    if set(report["methods"]) != set(names):
        raise ValueError("Saved method roster differs from all declared procedures")
    predicted_keys = {key.removeprefix("prediction__") for key in saved if key.startswith("prediction__")}
    if predicted_keys != set(names):
        raise ValueError("Prediction arrays are missing, extra or duplicated")
    indices, residuals, disagreement = audit_oof(report, arrays, saved, cfg)
    audit_metric(report, arrays, indices, residuals, cfg)
    target = saved["target"]
    ridge_prediction = saved["prediction__ridge"]
    ridge_loss = float(np.mean((ridge_prediction - target) ** 2))
    rows, horizons, decompositions, trial_rows = [], [], [], []
    target_horizons = (cfg["synthetic"]["horizons"] if expected_task["phase"] == "synthetic"
                       else list(range(1, target.shape[1] + 1)))
    for name in names:
        entry, pred = report["methods"][name], saved["prediction__" + name]
        if pred.shape != target.shape or not np.isfinite(pred).all():
            raise ValueError("Invalid forecast shape or nonfinite prediction")
        if entry["status"] not in {"ok", "fallback"}:
            raise ValueError("Unknown forecast completion state")
        errors = np.square(pred - target)
        loss = float(errors.mean())
        scale = expected_task["loss_scale"]
        _close(entry["test_mse"], loss, "Saved test loss differs from forecasts")
        _close(entry["scaled_test_mse"], loss / scale, "Saved scaled test loss differs from forecasts")
        _close(entry["horizon_mse"], errors.mean(axis=0), "Horizon loss differs from forecasts")
        audit_trials(name, entry, cfg)
        audit_cost(name, entry, report["nodes"])
        if expected_task["phase"] == "synthetic":
            np.testing.assert_array_equal(saved["groups_test"], arrays["groups_test"])
            episode_losses = [float(errors[arrays["groups_test"] == group].mean())
                              for group in np.unique(arrays["groups_test"])]
            _close(np.mean(episode_losses), loss, "Synthetic origin average differs from episode-balanced loss")
        if entry["status"] == "fallback":
            np.testing.assert_array_equal(pred, ridge_prediction, err_msg="Fallback differs from baseline")
        selected = entry["selected"]
        if selected.get("gamma") == 0:
            np.testing.assert_array_equal(pred, ridge_prediction, err_msg="Zero correction changes baseline")
        if name.startswith(("fixed_is_", "fixed_oof_", "learned_oof_")):
            delta, remaining = pred - ridge_prediction, target - ridge_prediction
            alignment, energy = float(np.mean(delta * remaining)), float(np.mean(delta ** 2))
            gain = ridge_loss - loss
            _close(gain, 2 * alignment - energy, "Squared-loss residual identity failed",
                   atol=2e-12 * max(1.0, ridge_loss, loss))
            if entry["status"] == "ok":
                _close(entry["residual_alignment"], alignment, "Reported residual alignment differs")
                _close(entry["correction_energy"], energy, "Reported correction energy differs")
            decompositions.append({"task_id": expected_task["task_id"], "phase": expected_task["phase"],
                                   "source_id": expected_task["source_id"], "method": name,
                                   "alignment": alignment, "energy": energy, "ridge_minus_method_mse": gain,
                                   "identity_error": gain - (2 * alignment - energy)})
        row = {"task_id": expected_task["task_id"], "phase": expected_task["phase"],
               "source_id": expected_task["source_id"], "series_id": expected_task.get("series_id", ""),
               "seed": expected_task["seed"], "method": name, "status": entry["status"],
               "test_mse": loss, "loss_scale": scale, "scaled_test_mse": loss / scale,
               "ridge_minus_method_mse": ridge_loss - loss,
               "ratio_to_ridge": loss / ridge_loss if ridge_loss > 0 else None,
               "alpha": selected.get("alpha"), "gamma": selected.get("gamma"),
               "pipeline_wall_seconds": entry["pipeline_wall_seconds"],
               "pipeline_cpu_seconds": entry["pipeline_cpu_seconds"],
               "serialized_model_bytes": entry["serialized_model_bytes"],
               "standalone_prediction_wall_seconds": sum(report["nodes"][node]["wall_seconds"]
                   for node in entry["dependencies"] if node.endswith(":predict") or node.endswith(":test")),
               "correction_rows": report["correction_rows"]}
        rows.append(row)
        for horizon, horizon_loss in zip(target_horizons, errors.mean(axis=0)):
            horizons.append({"task_id": expected_task["task_id"], "phase": expected_task["phase"],
                             "source_id": expected_task["source_id"], "method": name,
                             "horizon": horizon, "test_mse": float(horizon_loss),
                             "scaled_test_mse": float(horizon_loss / scale)})
        for index, trial in enumerate(selected.get("trials", [])):
            trial_rows.append({"task_id": expected_task["task_id"], "method": name, "trial_index": index, **trial})
    metric = report["metric"]
    initial, final = metric.get("initial_objective"), metric.get("final_objective")
    diagnostic = {"task_id": expected_task["task_id"], "phase": expected_task["phase"],
                  "source_id": expected_task["source_id"], "correction_rows": report["correction_rows"],
                  "metric_initial_objective": initial, "metric_final_objective": final,
                  "metric_training_objective_reduction": None if initial is None or final is None else initial - final,
                  "metric_optimizer_success": metric.get("optimizer_success", False),
                  "metric_iterations": metric.get("iterations"),
                  "metric_weight_ratio": metric.get("realized_weight_ratio", 1.0),
                  "metric_fallback": metric.get("fallback_reason") or metric.get("fallback"),
                  "donor_final_disagreement_mse": disagreement,
                  "in_sample_residual_mse": report["in_sample_residual_mse"],
                  "oof_residual_mse": report["oof_residual_mse"],
                  "oof_failure": report.get("oof_failure"),
                  "warning_count": len(report["warnings"]),
                  "group_reduction_bank_count": sum(bool(bank.get("group_reductions")) for bank in report["banks"].values())}
    return rows, horizons, decompositions, trial_rows, diagnostic


def paired_interval(values, cfg):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 or not np.isfinite(values).all():
        raise ValueError("Paired intervals require finite independent complete repetitions")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    family = cfg["confidence"]["primary_family_size"]
    alpha = cfg["confidence"]["family_alpha"]
    critical = float(student_t.ppf(1 - alpha / (2 * family), n - 1))
    width = critical * sd / math.sqrt(n)
    return {"n": n, "mean_difference": mean, "paired_sd": sd,
            "lower": mean - width, "upper": mean + width, "critical_t": critical,
            "family_size": family, "family_alpha": alpha}


def summarize(rows, diagnostics, cfg, tasks, failures):
    frame = pd.DataFrame(rows)
    names = expected_methods(cfg)
    synthetic, contrasts, source_rows, natural, selection = [], [], [], [], []
    repetitions = cfg["synthetic"]["repetition_seeds"]
    for law in cfg["synthetic"]["laws"]:
        for name in names:
            subset = frame[(frame.phase == "synthetic") & (frame.source_id == law) & (frame.method == name)]
            complete = len(subset) == len(repetitions)
            synthetic.append({"law": law, "method": name, "completed": len(subset), "expected": len(repetitions),
                              "mean_mse": float(subset.test_mse.mean()) if complete else None,
                              "sd_mse": float(subset.test_mse.std(ddof=1)) if complete else None,
                              "mean_wall_seconds": float(subset.pipeline_wall_seconds.mean()) if complete else None,
                              "fallbacks": int((subset.status == "fallback").sum())})
        for control, candidate in cfg["primary_contrasts"]:
            subset = frame[(frame.phase == "synthetic") & (frame.source_id == law)]
            pivot = subset.pivot(index="seed", columns="method", values="test_mse").reindex(repetitions)
            complete = control in pivot and candidate in pivot and pivot[[control, candidate]].notna().all().all()
            row = {"law": law, "control": control, "candidate": candidate,
                   "sign": "positive favors candidate", "complete": bool(complete)}
            if complete:
                row.update(paired_interval((pivot[control] - pivot[candidate]).to_numpy(), cfg))
                row["control_mean_mse"], row["candidate_mean_mse"] = float(pivot[control].mean()), float(pivot[candidate].mean())
                row["relative_improvement_of_means_pct"] = 100 * row["mean_difference"] / row["control_mean_mse"] if row["control_mean_mse"] > 0 else None
                row["decision"] = "candidate_better" if row["lower"] > 0 else "candidate_worse" if row["upper"] < 0 else "unresolved"
            else:
                row.update({"n": None, "mean_difference": None, "paired_sd": None, "lower": None, "upper": None,
                            "family_size": cfg["confidence"]["primary_family_size"], "decision": "incomplete_roster"})
            contrasts.append(row)
    natural_tasks = [task for task in tasks if task["phase"] == "natural"]
    source_ids = list(dict.fromkeys(task["source_id"] for task in natural_tasks))
    for source in source_ids:
        expected = sum(task["source_id"] == source for task in natural_tasks)
        for name in names:
            subset = frame[(frame.phase == "natural") & (frame.source_id == source) & (frame.method == name)]
            complete = len(subset) == expected
            source_rows.append({"source_id": source, "method": name, "completed": len(subset), "expected": expected,
                                "mean_raw_mse": float(subset.test_mse.mean()) if complete else None,
                                "scaled_mse": float(subset.scaled_test_mse.mean()) if complete else None,
                                "mean_wall_seconds": float(subset.pipeline_wall_seconds.mean()) if complete else None,
                                "mean_cpu_seconds": float(subset.pipeline_cpu_seconds.mean()) if complete else None,
                                "mean_prediction_wall_seconds": float(subset.standalone_prediction_wall_seconds.mean()) if complete else None,
                                "mean_model_bytes": float(subset.serialized_model_bytes.mean()) if complete else None,
                                "fallbacks": int((subset.status == "fallback").sum())})
    for name in names:
        subset = [row for row in source_rows if row["method"] == name]
        complete = all(row["scaled_mse"] is not None for row in subset)
        natural.append({"method": name, "completed": sum(row["completed"] for row in subset),
                        "expected": len(natural_tasks), "source_count": len(source_ids),
                        "scaled_mse": float(np.mean([row["scaled_mse"] for row in subset])) if complete else None,
                        "mean_wall_seconds": float(np.mean([row["mean_wall_seconds"] for row in subset])) if complete else None,
                        "mean_cpu_seconds": float(np.mean([row["mean_cpu_seconds"] for row in subset])) if complete else None,
                        "mean_prediction_wall_seconds": float(np.mean([row["mean_prediction_wall_seconds"] for row in subset])) if complete else None,
                        "mean_model_bytes": float(np.mean([row["mean_model_bytes"] for row in subset])) if complete else None,
                        "fallbacks": sum(row["fallbacks"] for row in subset)})
    baseline = next(row for row in natural if row["method"] == "ridge")
    for row in natural:
        row["relative_change_vs_ridge_pct"] = (100 * (row["scaled_mse"] / baseline["scaled_mse"] - 1)
            if row["scaled_mse"] is not None and baseline["scaled_mse"] is not None and baseline["scaled_mse"] > 0 else None)
        row["wall_ratio_vs_ridge"] = (row["mean_wall_seconds"] / baseline["mean_wall_seconds"]
            if row["mean_wall_seconds"] is not None and baseline["mean_wall_seconds"] is not None and baseline["mean_wall_seconds"] > 0 else None)
    for phase in ("synthetic", "natural"):
        for name in names:
            subset = frame[(frame.phase == phase) & (frame.method == name)]
            if name.startswith(("fixed_is_", "fixed_oof_", "learned_oof_")):
                for gamma in cfg["gammas"]:
                    selection.append({"phase": phase, "method": name, "gamma": gamma,
                                      "count": int((subset.gamma == gamma).sum()), "completed": len(subset),
                                      "fallbacks": int((subset.status == "fallback").sum())})
    return {"synthetic": synthetic, "primary_contrasts": contrasts, "natural_sources": source_rows,
            "natural": natural, "selection": selection,
            "metric_diagnostics": diagnostics, "failures": failures}


def analyze(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    cfg = json.loads((ROOT / "config.json").read_text())
    frozen = json.loads((ROOT / "freeze.json").read_text())
    for path, expected in frozen["files"].items():
        if data.digest(REPO / path) != expected:
            raise ValueError(f"Scientific source changed after freeze: {path}")
    execution = json.loads((run_dir / "execution.json").read_text())
    tasks = expected_roster(cfg)
    if execution["tasks"] != tasks or execution["methods"] != expected_methods(cfg):
        raise ValueError("Execution roster differs from the complete prespecified study")
    if execution["config_sha256"] != data.digest(ROOT / "config.json") or execution["freeze_sha256"] != data.digest(ROOT / "freeze.json"):
        raise ValueError("Execution belongs to a different scientific freeze")
    if len(cfg["primary_contrasts"]) * len(cfg["synthetic"]["laws"]) != cfg["confidence"]["primary_family_size"]:
        raise ValueError("Primary confidence-family size disagrees with declared contrasts")
    expected_ids = {task["task_id"] for task in tasks}
    actual_ids = {path.name for path in run_dir.iterdir() if path.is_dir()}
    if actual_ids - expected_ids:
        raise ValueError("Unexpected task directories cannot be silently omitted")
    rows, horizons, decompositions, trials, diagnostics, failures = [], [], [], [], [], []
    for task in tasks:
        directory = run_dir / task["task_id"]
        if (directory / "result.json").exists() and (directory / "failure.json").exists():
            raise ValueError("Task contains both success and failure records")
        if (directory / "failure.json").exists():
            failure = json.loads((directory / "failure.json").read_text())
            if failure["task"] != task:
                raise ValueError("Failure record belongs to a different task")
            failures.append({"task_id": task["task_id"], "phase": task["phase"],
                             "source_id": task["source_id"], "error": failure["error"], "status": "task_failure"})
            continue
        if not (directory / "result.json").exists():
            failures.append({"task_id": task["task_id"], "phase": task["phase"],
                             "source_id": task["source_id"], "error": "Missing completed task record", "status": "missing"})
            continue
        print(f"Audit predictions and data: {task['task_id']}", flush=True)
        task_rows, task_horizons, task_decompositions, task_trials, diagnostic = audit_task(directory, task, cfg)
        rows.extend(task_rows)
        horizons.extend(task_horizons)
        decompositions.extend(task_decompositions)
        trials.extend(task_trials)
        diagnostics.append(diagnostic)
    if not rows:
        raise ValueError("No auditable predictions available")
    sections = summarize(rows, diagnostics, cfg, tasks, failures)
    complete = not failures and len(rows) == len(tasks) * len(expected_methods(cfg))
    summary = {"schema_version": 1, "study_id": cfg["study_id"],
               "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "complete": complete, "expected_tasks": len(tasks), "completed_tasks": len(diagnostics),
               "expected_predictions": len(tasks) * len(expected_methods(cfg)), "audited_predictions": len(rows),
               "fallback_predictions": sum(row["status"] == "fallback" for row in rows),
               "methods": expected_methods(cfg), "config": cfg,
               "source_hashes": {"execution": data.digest(run_dir / "execution.json"),
                                 "freeze": data.digest(ROOT / "freeze.json"),
                                 "config": data.digest(ROOT / "config.json"),
                                 "roster": data.digest(data.ROSTER_PATH)},
               "tasks": tasks,
               "scope": {"synthetic": "18 Bonferroni paired Student contrasts over 12 complete independent repetitions per law; normal-difference approximation. No family claim spanning older studies.",
                         "natural": "Descriptive equal-collection mean of equal-series means on four selected new collections and twelve series. No domain-population confidence interval; no claims of independent windows or broad superiority.",
                         "failures": "Prespecified baseline fallbacks count as predictions with incurred cost. A missing task makes affected full-roster means and complete-repetition intervals undefined; no matched-success replacement.",
                         "metric": "Training-only kernel-regression proxy shared by all four gates; optimizing it is not end-to-end KAHM learning or test improvement evidence.",
                         "timing": execution["environment"]["timing_scope"],
                         "audit": "Inputs/hashes, saved forecasts/losses, residual identity, OOF exclusions, metric objective and complete computation dependencies independently reconstructed without forecasting refits. Recorded validation grid and chosen minimum checked, but individual trial fits not independently refitted."},
               "environment": execution["environment"], **sections}
    output.mkdir(parents=True, exist_ok=True)
    data.write_json(output / "summary.json", summary)
    for filename, records in (("task_metrics.csv", rows), ("horizon_metrics.csv", horizons),
                              ("residual_identity.csv", decompositions), ("validation_trials.csv", trials),
                              ("synthetic_means.csv", sections["synthetic"]),
                              ("primary_contrasts.csv", sections["primary_contrasts"]),
                              ("natural_sources.csv", sections["natural_sources"]),
                              ("natural_aggregate.csv", sections["natural"]),
                              ("selection_frequency.csv", sections["selection"]),
                              ("metric_diagnostics.csv", diagnostics), ("failures.csv", failures)):
        write_csv(output / filename, records)
    print(json.dumps({key: summary[key] for key in ("complete", "completed_tasks", "audited_predictions", "fallback_predictions")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    summary = analyze(args.run_dir, args.output)
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
