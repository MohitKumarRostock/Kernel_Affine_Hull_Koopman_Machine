#!/usr/bin/env python3
"""Score sealed cross-fit residual forecasts under the prospective protocol.

This program performs no fitting or forecast selection from D outcomes.
All inferential units are the 24 independent repetition aggregates.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
PANELS = {"nonlinear": LAWS[1:], "linear": LAWS[:1], "all_descriptive": LAWS}
HORIZONS = (1, 4, 12)
CAPS = (1, 2, 4, 8, 12)
SCALERS = ("standard", "minmax", "tanh_z_1")
REPETITIONS, TASKS, ARRAY_COUNT = 24, 192, 42
GAMMAS = (0.0, 0.25, 0.5, 1.0)
ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0)
LEGACY_METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common",
                  "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy")
GEOMETRY_METHODS = ("common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded")
LEGACY_POLICY = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded")
LEGACY_PAIRS = ("standard", "minmax", "tanh", "hybrid_tanh_minmax", "hybrid_minmax_tanh")
LEGACY_CAPS = (1, 2, 4, 8, 12, 20)
NEW_METHODS = ("common", "shared_bias", "half_common", "half_shared_bias", "crossfit_kahm", "crossfit_rbf",
               "crossfit_matched_rbf", "insample_kahm", "insample_rbf", "insample_matched_rbf")
CORRECTIONS = NEW_METHODS[4:]
POLICY_ORDER = tuple("legacy__" + x for x in LEGACY_POLICY) + (
    "crossfit__shared_bias", "crossfit__crossfit_rbf", "crossfit__insample_rbf")
META = ("task_id", "law", "history", "repetition", "seed", "noise_index", "noise_level")
IDENTITY = ("family", "mode", "method", "forecast_key")
FAMILY_ALPHA, FAMILY_SIZE = 0.05, 4
SUPERIORITY_MARGIN, NONINFERIORITY_MARGIN, STABILITY_RATIO = 0.03, 0.05, 5.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def confined(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Artifact path must be relative")
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing or unconfined artifact: " + relative)
    return path


def finite(value, shape, label):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind in "fiu" and np.isfinite(value).all(),
            "Invalid array: " + label)
    return value


def close(left, right, label, *, exact=False):
    valid = np.array_equal(left, right) if exact else np.allclose(left, right, rtol=2e-12, atol=2e-13)
    require(valid, "Identity check failed: " + label)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    with Path(path).open("x") as destination:
        json.dump(json_safe(value), destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def export_csv(path, rows):
    require(bool(rows), "Cannot export an empty table: " + str(path))
    fields = list(rows[0])
    require(all(set(row) == set(fields) for row in rows), "Heterogeneous columns: " + str(path))
    with Path(path).open("x", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def forecast_specs():
    result = {}
    for family, modes, methods in (("legacy", ("calibrated", "gamma1"), LEGACY_METHODS),
                                   ("crossfit", ("calibrated", "eta1"), NEW_METHODS),
                                   ("policy", ("calibrated", "eta1"), ("expanded_nonkahm",))):
        for mode in modes:
            for method in methods:
                key = f"{family}__{mode}__{method}"
                result[key] = dict(family=family, mode=mode, method=method, forecast_key=key)
    require(len(result) == ARRAY_COUNT, "Internal prospective array roster differs")
    return result


SPECS = forecast_specs()


def validate_config(config):
    require(config["schema_version"] == "kahm-crossfit-residual-v1", "Unexpected study schema")
    require(config["master_seed"] == 2026110173, "Unexpected master seed")
    require(config["repetitions"] == REPETITIONS and config["task_count"] == TASKS, "Task/repetition count differs")
    require(tuple(config["laws"]) == LAWS and tuple(config["horizons"]) == HORIZONS, "Laws/horizons differ")
    require(config["histories"] == [12] and config["noise_levels"] == [0.0, 0.1], "History/noise differs")
    require(config["split_episodes"] == {"A": 64, "B": 32, "C": 32, "D": 128}, "Episode allocation differs")
    require(config["split_caps"] == {"A": 512, "B": 256, "C": 256, "D": None}, "Row allocation differs")
    require(config["origins_per_episode"] == 13 and config["crossfit"]["folds"] == 4, "Origin/fold count differs")
    require(config["crossfit"]["fold_assignment"] == "original A episode ID modulo 4", "Fold assignment differs")
    require(tuple(config["native_cap_order"]) == CAPS and tuple(config["scaler_order"]) == SCALERS,
            "New candidate grid/tie order differs")
    require(tuple(config["methods"]) == LEGACY_METHODS and tuple(config["crossfit_methods"]) == NEW_METHODS,
            "Forecast method roster differs")
    require(tuple(config["policy_order"]) == POLICY_ORDER, "Expanded policy order differs")
    require(tuple(config["model"]["gamma_grid"]) == GAMMAS and tuple(config["model"]["alpha_grid"]) == ALPHAS,
            "Multiplier/penalty grid differs")
    require(config["model"]["target_variance_floor"] == 1e-12, "Target-variance floor differs")
    require(config["execution"]["output_array_count_per_task"] == ARRAY_COUNT, "Forecast count differs")
    analysis = config["analysis"]
    require(analysis["independent_unit"] == "repetition" and analysis["primary_mode"] == "calibrated",
            "Inferential unit/mode differs")
    require(analysis["family_alpha"] == FAMILY_ALPHA and analysis["primary_family_size"] == FAMILY_SIZE,
            "Primary family allocation differs")
    require(analysis["superiority_margin"] == SUPERIORITY_MARGIN
            and analysis["linear_noninferiority_margin"] == NONINFERIORITY_MARGIN
            and analysis["max_error_ratio"] == STABILITY_RATIO, "Primary/stability margins differ")


def manifest_roster(manifest):
    tasks = manifest["tasks"]
    expected = {(law, 12, noise, rep) for law in LAWS for noise in (0, 1) for rep in range(REPETITIONS)}
    require(len(tasks) == TASKS and len({t["task_id"] for t in tasks}) == TASKS, "Incomplete/duplicated task roster")
    require({(t["law"], t["history"], t["noise_index"], t["repetition"]) for t in tasks} == expected,
            "Manifest factorial differs")
    return {task["task_id"]: task for task in tasks}


def blend(base, forecast, gamma):
    gamma = finite(gamma, (3,), "calibration multiplier")
    require(np.isin(gamma, GAMMAS).all(), "Undeclared calibration multiplier")
    result = base.copy()
    for h, value in enumerate(gamma):
        if value != 0.0:
            result[:, h] = base[:, h] + value * (forecast[:, h] - base[:, h])
    return result


def policy_key(identifier, *, calibrated=True):
    family, method = identifier.split("__", 1)
    require(identifier in POLICY_ORDER, "Ineligible expanded policy member")
    mode = "calibrated" if calibrated else ("gamma1" if family == "legacy" else "eta1")
    return f"{family}__{mode}__{method}"


def score_task(meta, predictions, targets, scale):
    require(set(predictions) == set(SPECS), "The complete 42-array forecast roster differs")
    targets = finite(targets, (1664, 3), "D outcomes")
    scale = finite(scale, (3,), "A target variance")
    require((scale >= 1e-12).all(), "Invalid target-variance floor")
    horizon_rows, task_rows = [], []
    ridge = finite(predictions["legacy__calibrated__ridge"], targets.shape, "ridge")
    ridge_max = float(np.max(np.abs(targets - ridge)))
    screen = None
    for key, identity in SPECS.items():
        pred = finite(predictions[key], targets.shape, key)
        residual = targets - pred
        mse = np.mean(residual * residual, axis=0)
        require(np.isfinite(mse).all(), "Nonfinite squared loss")
        record = {**{name: meta[name] for name in META}, **identity}
        for h, horizon in enumerate(HORIZONS):
            horizon_rows.append({**record, "horizon": horizon, "D_rows": len(targets),
                                 "A_target_variance": float(scale[h]), "mse": float(mse[h]),
                                 "smse": float(mse[h] / scale[h])})
        task_rows.append({**record, "mse": float(mse.mean()), "smse": float(np.mean(mse / scale))})
        if key == "crossfit__calibrated__crossfit_kahm":
            maximum = float(np.max(np.abs(residual)))
            ratio = maximum / ridge_max if ridge_max else (0.0 if maximum == 0 else math.inf)
            require(math.isfinite(ratio), "Undefined selected maximum-error screen")
            screen = {**record, "maximum_absolute_error": maximum, "ridge_maximum_absolute_error": ridge_max,
                      "maximum_error_ratio": ratio, "screen_threshold": STABILITY_RATIO,
                      "screen_pass": bool(ratio <= STABILITY_RATIO)}
    require(screen is not None, "Missing selected-procedure screen")
    return horizon_rows, task_rows, screen


def grouped(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def average_rows(rows, fields, values=("mse", "smse"), *, spread=False):
    output = []
    for key, members in grouped(rows, fields).items():
        record = {**dict(zip(fields, key)), "n_records": len(members)}
        if spread:
            require(len(members) == REPETITIONS, "Descriptive mean interval must use 24 independent repetitions")
            require(len({m["repetition"] for m in members}) == REPETITIONS, "Duplicated descriptive repetition")
        for name in values:
            values_array = np.asarray([m[name] for m in members], dtype=float)
            record[name] = float(values_array.mean())
            if spread:
                sd = float(values_array.std(ddof=1)); se = sd / np.sqrt(REPETITIONS)
                critical = float(student_t.ppf(0.975, REPETITIONS - 1))
                record.update({name + "_sd": sd, name + "_se": float(se),
                               name + "_lower95_descriptive": record[name] - critical * se,
                               name + "_upper95_descriptive": record[name] + critical * se})
        output.append(record)
    return output


def repetition_table(task_rows):
    output = []
    for panel, laws in PANELS.items():
        eligible = [row for row in task_rows if row["law"] in laws]
        fields = ("repetition", *IDENTITY)
        for key, members in grouped(eligible, fields).items():
            require(len(members) == 2 * len(laws), "Incomplete repetition condition set")
            require(len({(m["law"], m["noise_index"]) for m in members}) == len(members),
                    "Duplicated repetition conditions")
            output.append({"panel": panel, **dict(zip(fields, key)), "n_conditions": len(members),
                           "mse": float(np.mean([m["mse"] for m in members])),
                           "smse": float(np.mean([m["smse"] for m in members]))})
    require(len(output) == REPETITIONS * ARRAY_COUNT * len(PANELS), "Missing repetition/forecast views")
    return output


def vector_lookup(repetition_rows):
    lookup = {(r["panel"], r["forecast_key"], r["repetition"]): r["smse"] for r in repetition_rows}
    require(len(lookup) == len(repetition_rows), "Duplicated repetition-forecast identity")
    def vector(panel, forecast_key):
        values = np.asarray([lookup[panel, forecast_key, rep] for rep in range(REPETITIONS)], dtype=float)
        require(values.shape == (REPETITIONS,) and np.isfinite(values).all() and np.all(values >= 0),
                "Invalid paired loss vector")
        return values
    return vector


def primary_tables(vector):
    candidate_key = "crossfit__calibrated__crossfit_kahm"
    cases = (
        ("crossfit_KAHM_vs_shared_bias", "nonlinear", "crossfit__calibrated__shared_bias", 0.97, "3% superiority"),
        ("crossfit_KAHM_vs_independent_RBF", "nonlinear", "crossfit__calibrated__crossfit_rbf", 0.97, "3% superiority"),
        ("crossfit_KAHM_vs_expanded_nonKAHM", "nonlinear", "policy__calibrated__expanded_nonkahm", 0.97, "3% superiority"),
        ("linear_crossfit_KAHM_vs_ridge", "linear", "legacy__calibrated__ridge", 1.05, "5% noninferiority"),
    )
    summaries, details = [], []
    critical = float(student_t.ppf(1 - FAMILY_ALPHA / FAMILY_SIZE, REPETITIONS - 1))
    for identifier, panel, comparator_key, factor, margin in cases:
        candidate, comparator = vector(panel, candidate_key), vector(panel, comparator_key)
        gap = factor * comparator - candidate
        mean, sd = float(gap.mean()), float(gap.std(ddof=1)); se = sd / np.sqrt(REPETITIONS)
        lower = float(mean - critical * se)
        p = float(student_t.sf(mean / se, REPETITIONS - 1)) if se else (0.0 if mean > 0 else (1.0 if mean < 0 else 0.5))
        denominator = float(comparator.mean())
        summaries.append({"contrast_id": identifier, "panel": panel, "candidate_key": candidate_key,
                          "comparator_key": comparator_key, "mode": "calibrated", "margin": margin,
                          "comparator_factor": factor, "candidate_mean_smse": float(candidate.mean()),
                          "comparator_mean_smse": denominator,
                          "point_relative_gain_percent": float(100 * (1 - candidate.mean() / denominator)) if denominator else None,
                          "n_repetitions": REPETITIONS, "df": REPETITIONS - 1,
                          "family_alpha": FAMILY_ALPHA, "family_size": FAMILY_SIZE,
                          "margin_gap_mean": mean, "margin_gap_sample_sd": sd, "margin_gap_se": float(se),
                          "critical_t": critical, "simultaneous_one_sided_lower": lower,
                          "p_one_sided": p, "p_bonferroni": min(1.0, FAMILY_SIZE * p),
                          "positive_margin_repetitions": int(np.count_nonzero(gap > 0)),
                          "criterion_established": bool(lower > 0)})
        details.extend({"contrast_id": identifier, "panel": panel, "repetition": rep,
                        "candidate_smse": float(candidate[rep]), "comparator_smse": float(comparator[rep]),
                        "comparator_factor": factor, "margin_gap": float(gap[rep])} for rep in range(REPETITIONS))
    return summaries, details


def descriptive_contrast(identifier, panel, left_key, right_key, vector, *, scope):
    left, right = vector(panel, left_key), vector(panel, right_key)
    difference = left - right
    mean = float(difference.mean()); se = float(difference.std(ddof=1) / np.sqrt(REPETITIONS))
    critical = float(student_t.ppf(0.975, REPETITIONS - 1)); denominator = float(right.mean())
    identity = dict(contrast_id=identifier, panel=panel, scope=scope, left_key=left_key, right_key=right_key)
    summary = {**identity, "inference": "descriptive_unadjusted_95_percent_paired_t", "n_repetitions": REPETITIONS,
               "left_mean_smse": float(left.mean()), "right_mean_smse": denominator,
               "relative_gain_percent": float(100 * (1 - left.mean() / denominator)) if denominator else None,
               "mean_difference": mean, "standard_error": se, "lower95": mean - critical * se,
               "upper95": mean + critical * se, "better_repetitions": int(np.count_nonzero(difference < 0)),
               "ties": int(np.count_nonzero(difference == 0)), "worse_repetitions": int(np.count_nonzero(difference > 0))}
    details = [{**identity, "repetition": rep, "left_smse": float(left[rep]), "right_smse": float(right[rep]),
                "difference": float(difference[rep])} for rep in range(REPETITIONS)]
    return summary, details


def descriptive_tables(vector):
    cases = []
    for mode in ("calibrated", "eta1"):
        new = lambda method: f"crossfit__{mode}__{method}"
        old = lambda method: f"legacy__{'calibrated' if mode == 'calibrated' else 'gamma1'}__{method}"
        for comparator in ("shared_bias", "crossfit_rbf", "crossfit_matched_rbf"):
            cases.append((f"crossfit_KAHM_minus_{comparator}__{mode}", new("crossfit_kahm"), new(comparator), "fixed_common_mechanism"))
        cases.append(("crossfit_KAHM_minus_expanded_policy__" + mode, new("crossfit_kahm"),
                      f"policy__{mode}__expanded_nonkahm", "complete_procedure"))
        cases.append(("expanded_minus_legacy_policy__" + mode, f"policy__{mode}__expanded_nonkahm",
                      old("nonkahm_policy"), "declared_policy_menu"))
        for kind in ("kahm", "rbf", "matched_rbf"):
            cases.append((f"crossfit_minus_insample_{kind}__{mode}", new("crossfit_" + kind),
                          new("insample_" + kind), "residual_target_source"))
        for comparator in ("shared_bias", "insample_rbf", "insample_matched_rbf"):
            cases.append((f"insample_KAHM_minus_{comparator}__{mode}", new("insample_kahm"),
                          new(comparator), "insample_ablation"))
        for method in ("kahm_departure", "nonkahm_policy"):
            cases.append((f"crossfit_KAHM_minus_legacy_{method}__{mode}", new("crossfit_kahm"),
                          old(method), "different_complete_architectures"))
    for left, right, name in (("common", "half_common", "full_minus_half_common"),
                              ("shared_bias", "half_shared_bias", "full_minus_half_shared_bias"),
                              ("shared_bias", "common", "full_shared_bias_minus_common"),
                              ("half_shared_bias", "half_common", "half_shared_bias_minus_common")):
        cases.append((name, "crossfit__calibrated__" + left, "crossfit__calibrated__" + right, "common_allocation_or_bias"))
    for family, methods, unit in (("legacy", LEGACY_METHODS, "gamma1"),
                                  ("crossfit", CORRECTIONS, "eta1"),
                                  ("policy", ("expanded_nonkahm",), "eta1")):
        for method in methods:
            cases.append((f"calibrated_minus_unit__{family}__{method}", f"{family}__calibrated__{method}",
                          f"{family}__{unit}__{method}", "calibration_same_selection"))
    summaries, details = [], []
    require(len({c[0] for c in cases}) == len(cases), "Duplicated descriptive contrast ID")
    for panel in PANELS:
        for identifier, left, right, scope in cases:
            record, paired = descriptive_contrast(identifier, panel, left, right, vector, scope=scope)
            summaries.append(record); details.extend(paired)
    return summaries, details


def _mse(prediction, target):
    prediction = finite(prediction, target.shape, "development forecast")
    result = np.mean((target - prediction) ** 2, axis=0)
    require(np.isfinite(result).all(), "Nonfinite development loss")
    return result


def _alpha_replay(record, label):
    candidates = record["candidates"]
    require([r["alpha"] for r in candidates] == list(ALPHAS), "Undeclared alpha grid: " + label)
    losses = finite([r["mse_by_horizon"] for r in candidates], (len(ALPHAS), 3), label + " alpha scores")
    indices = np.argmin(losses, axis=0)
    expected = np.asarray(ALPHAS)[indices]
    close([r["alpha"] for r in record["selected"]], expected, label + " alpha argmin", exact=True)
    require([r["candidate_index"] for r in record["selected"]] == indices.tolist(), label + " alpha indices differ")
    close([r["mse"] for r in record["selected"]], losses[indices, np.arange(3)], label + " selected alpha losses")
    return expected


def development_audit(meta, fit, pre_c, arrays, training, validation):
    """Replay development choices and cross-fit membership without fitting a model."""
    ax, ay, bx, by, groups = [arrays[k] for k in ("A_X", "A_Y", "B_X", "B_Y", "A_groups")]
    finite(ax, (512, 12), "A histories"); finite(ay, (512, 3), "A outcomes")
    finite(bx, (256, 12), "B histories"); finite(by, (256, 3), "B outcomes")
    require(groups.shape == (512,) and groups.dtype.kind in "iu", "A episode IDs are not integer row labels")
    close(np.unique(groups), np.arange(64), "All original A episode identities", exact=True)
    scale = np.maximum(np.var(ay, axis=0, ddof=0), 1e-12)
    close(arrays["loss_scale"], scale, "Full-A evaluation scale", exact=True)
    identity = {key: meta[key] for key in META}
    scores, choices, fold_rows, bias_rows, training_diagnostics = [], [], [], [], []
    checks = 0
    legacy, cross = fit["legacy"], fit["crossfit"]
    selection, diagnostic = cross["selections"], cross["diagnostics"]
    require(pre_c["C_loaded"] is False, "Pre-C record did not precede C access")
    require(pre_c["legacy"] == {key: legacy[key] for key in ("selections", "b_scores", "raw_rbf_selected_scaler")},
            "Legacy B choices changed after C")
    require(pre_c["crossfit"] == selection, "Cross-fit B choices changed after C")
    require(pre_c["crossfit_frozen_state_sha256"] == cross["calibration"]["pre_C_hash"]
            == cross["calibration"]["post_C_hash"], "C changed cross-fit fitting state")
    require(legacy["calibration_records"]["pre_C_hash"] == legacy["calibration_records"]["post_C_hash"],
            "C changed legacy fitting state")
    checks += 5
    actual_hashes = {key: array_hash(arrays[key]) for key in ("A_X", "A_Y", "B_X", "B_Y")}
    require(actual_hashes == legacy["diagnostics"]["input_hashes"] == diagnostic["input_hashes"],
            "Fitted input hashes differ")
    require(diagnostic["groups_A_sha256"] == array_hash(groups), "Saved A groups differ")
    require(diagnostic["A_rows"] == 512 and diagnostic["B_rows"] == 256 and diagnostic["A_episodes"] == 64,
            "Cross-fit fitting allocation differs")
    require(diagnostic["number_of_folds"] == 4 and diagnostic["OOF_assignment_counts"] == {"1": 512},
            "OOF coverage record differs")
    require(diagnostic["full_A_feature_state_reused_from_reference_primary"] is True
            and diagnostic["fold_outputs_used_as_features"] is False
            and diagnostic["bias_target_modes_share_exactly_one_state"] is True,
            "Declared common/feature sharing differs")
    checks += 5

    def score_row(scope, method, candidate_id, scaler, cap, mse, normalizer, selected, alpha):
        scores.append({**identity, "scope": scope, "method": method, "candidate_id": candidate_id,
                       "scaler": scaler, "cap": cap, "B_normalized_mse": float(np.mean(mse / normalizer)),
                       "B_mse_h1": float(mse[0]), "B_mse_h4": float(mse[1]), "B_mse_h12": float(mse[2]),
                       "normalizer_h1": float(normalizer[0]), "normalizer_h4": float(normalizer[1]),
                       "normalizer_h12": float(normalizer[2]), "selected": bool(selected),
                       "alpha_by_horizon": json.dumps(np.asarray(alpha).tolist())})

    def choice_row(scope, method, candidate_id, scaler, cap, alpha, *, selection_rule="B first minimum"):
        choices.append({**identity, "scope": scope, "method": method, "candidate_id": candidate_id,
                        "scaler": scaler, "cap": cap, "alpha_by_horizon": json.dumps(np.asarray(alpha).tolist()),
                        "selection_rule": selection_rule, "selection_split": "B", "frozen_before_C": True})

    legacy_order = [f"{pair}__cap{cap}" for cap in LEGACY_CAPS for pair in LEGACY_PAIRS]
    require(set(legacy["b_scores"]) == set(legacy_order), "Legacy B candidate roster differs")
    selected_legacy = legacy["selections"]["selected"]
    require(set(selected_legacy) == set(GEOMETRY_METHODS), "Legacy selected geometry roster differs")
    for key in legacy_order:
        pair, cap_text = key.rsplit("__cap", 1)
        for method in GEOMETRY_METHODS:
            prediction = validation[f"legacy__{key}__{method}"]
            mse = _mse(prediction, by)
            close(legacy["b_scores"][key][method], np.mean(mse / scale), "Legacy B loss " + key + "/" + method)
            alpha = _alpha_replay(legacy["candidates"][key]["B"][method], "legacy/" + key + "/" + method)
            score_row("legacy", method, key, pair, int(cap_text), mse, scale, selected_legacy[method] == key, alpha)
            checks += 2
    for method, chosen in selected_legacy.items():
        eligible = legacy_order if method != "common" else [k for k in legacy_order if k.rsplit("__cap", 1)[0] in LEGACY_PAIRS[:3]]
        require(chosen == min(eligible, key=lambda k: legacy["b_scores"][k][method]), "Legacy B argmin/tie differs")
        pair, cap_text = chosen.rsplit("__cap", 1)
        alpha = [r["alpha"] for r in legacy["candidates"][chosen]["B"][method]["selected"]]
        choice_row("legacy", method, chosen, pair, int(cap_text), alpha)
        checks += 1
    raw_scores = legacy["diagnostics"]["raw_rbf_scores"]
    require(set(raw_scores) == set(SCALERS) and np.isfinite(list(raw_scores.values())).all(), "Raw RBF scaler score roster differs")
    require(legacy["raw_rbf_selected_scaler"] == min(SCALERS, key=raw_scores.__getitem__), "Raw RBF scaler argmin/tie differs")
    choice_row("legacy", "raw_rbf", legacy["raw_rbf_selected_scaler"], legacy["raw_rbf_selected_scaler"], "", [],
               selection_rule="B first minimum over declared raw-input scalers")
    checks += 1

    expected_training = {"A_oof_common", "A_oof_residual", "A_full_common", "A_insample_residual", "A_fold_id",
                         "A_groups", "A_oof_assignment_count", "B_common", "B_shared_bias", "B_half_common", "B_half_shared_bias"}
    require(set(training) == expected_training, "Cross-fit training array roster differs")
    for key in ("A_oof_common", "A_oof_residual", "A_full_common", "A_insample_residual"):
        finite(training[key], ay.shape, key)
    for key in ("B_common", "B_shared_bias", "B_half_common", "B_half_shared_bias"):
        finite(training[key], by.shape, key)
    close(training["A_groups"], groups, "Training episode IDs", exact=True)
    close(training["A_fold_id"], groups % 4, "Original-episode modulo-four folds", exact=True)
    close(training["A_oof_assignment_count"], np.ones(len(ay), dtype=np.int64), "OOF exact-once assignments", exact=True)
    close(training["A_oof_residual"], ay - training["A_oof_common"], "OOF residual construction", exact=True)
    close(training["A_insample_residual"], ay - training["A_full_common"], "In-sample residual construction", exact=True)
    for mode, key in (("crossfit", "A_oof_residual"), ("insample", "A_insample_residual")):
        require(diagnostic["residual_target_hashes"][mode] == array_hash(training[key]), "Residual target hash differs")
    checks += 7
    close(selection["A_target_variance"], scale, "Shared departure target normalization", exact=True)
    full = selection["full_common"]
    require(full["selected_candidate"] == selected_legacy["common"], "Full-A common differs from selected legacy gamma-one head")
    require(full["selected_cap"] in CAPS and full["common_gamma"] == 1.0, "Full-A common cap/multiplier differs")
    close(full["training_target_variance"], scale, "Full-A common normalization", exact=True)
    close(full["training_episode_ids"], np.arange(64), "Full-A common episode allocation", exact=True)
    close(full["training_row_indices"], np.arange(512), "Full-A common row allocation", exact=True)
    close(training["B_common"], validation[f"legacy__{full['selected_candidate']}__common"], "Full-A B common identity", exact=True)
    require(full["B_prediction_sha256"] == array_hash(training["B_common"])
            and full["state_sha256"] == diagnostic["full_common_state_sha256"], "Full-A common identity hashes differ")
    choice_row("full_common", "common", full["selected_candidate"], full["selected_scaler"], full["selected_cap"], full["selected_alpha_by_horizon"])
    checks += 7
    bias = np.mean(by - training["B_common"], axis=0)
    half_bias = np.mean(by - training["B_half_common"], axis=0)
    close(selection["shared_bias"]["value"], bias, "Full-A B shared bias", exact=True)
    close(selection["half_common"]["shared_bias"], half_bias, "Half-A B shared bias", exact=True)
    close(training["B_shared_bias"], training["B_common"] + bias, "Full-A B bias baseline", exact=True)
    close(training["B_half_shared_bias"], training["B_half_common"] + half_bias, "Half-A B bias baseline", exact=True)
    require(selection["shared_bias"]["fit_split"] == "B" and selection["shared_bias"]["gamma"] == 1.0
            and selection["half_common"]["bias_fit_split"] == "B", "Bias fitting role differs")
    require(selection["shared_bias"]["state_sha256"] == diagnostic["shared_baseline_sha256"], "Shared baseline state differs")
    for allocation, values in (("full_A", bias), ("half_A", half_bias)):
        bias_rows.extend({**identity, "common_allocation": allocation, "horizon": horizon,
                          "shared_bias": float(values[h]), "fit_split": "B"} for h, horizon in enumerate(HORIZONS))
    checks += 6

    common_order = [f"{scaler}__cap{cap}" for cap in CAPS for scaler in SCALERS]
    folds = selection["folds"]
    require(len(folds) == 4 and [r["fold_id"] for r in folds] == list(range(4)), "Fold record roster differs")
    common_records = [(f"fold{r['fold_id']}", r, np.flatnonzero(groups % 4 != r["fold_id"])) for r in folds]
    common_records.append(("half", selection["half_common"], np.flatnonzero(groups < 32)))
    for scope, record, train_rows in common_records:
        close(record["training_row_indices"], train_rows, scope + " training rows", exact=True)
        close(record["training_episode_ids"], np.unique(groups[train_rows]), scope + " training episodes", exact=True)
        normalizer = np.maximum(np.var(ay[train_rows], axis=0, ddof=0), 1e-12)
        close(record["training_target_variance"], normalizer, scope + " complement-only normalizer", exact=True)
        require(record["candidate_order"] == common_order and [r["candidate_id"] for r in record["B_scores"]] == common_order,
                scope + " common candidate roster/order differs")
        require(record["selection_split"] == "B" and record["common_gamma"] == 1.0, scope + " common selection role differs")
        expected_inputs = {key: array_hash(value) for key, value in
                           (("A_X", ax[train_rows]), ("A_Y", ay[train_rows]), ("B_X", bx), ("B_Y", by))}
        require(record["input_hashes"] == expected_inputs, scope + " training input hashes differ")
        raw_record = diagnostic["common_fits"][scope]["fit"]["raw_ridge"]
        raw_alpha = _alpha_replay(raw_record, scope + "/raw_ridge")
        close(record["raw_alpha_by_horizon"], raw_alpha, scope + " raw ridge alpha", exact=True)
        close(_mse(validation[f"crossfit__{scope}__raw_ridge"], by),
              [r["mse"] for r in raw_record["selected"]], scope + " raw ridge selected B loss")
        checks += 3
        candidates = record["B_scores"]
        for candidate in candidates:
            key = candidate["candidate_id"]
            prediction = validation[f"crossfit__{scope}__{key}__common"]
            mse = _mse(prediction, by)
            require(candidate["B_prediction_sha256"] == array_hash(prediction), scope + " B common prediction hash differs")
            close(candidate["mse_by_horizon"], mse, scope + " B common MSE")
            close(candidate["score"], np.mean(mse / normalizer), scope + " B common normalized score")
            alpha = _alpha_replay(candidate["head_selection"], scope + "/" + key)
            close(candidate["selected_alpha_by_horizon"], alpha, scope + " candidate penalty identity", exact=True)
            score_row(scope, "common", key, candidate["scaler"], candidate["cap"], mse, normalizer,
                      key == record["selected_candidate"], alpha)
            checks += 5
        chosen_index = int(np.argmin([r["score"] for r in candidates])); chosen = candidates[chosen_index]
        require(record["selected_index"] == chosen_index and record["selected_candidate"] == chosen["candidate_id"],
                scope + " common B argmin/tie differs")
        require(record["selected_scaler"] == chosen["scaler"] and record["selected_cap"] == chosen["cap"],
                scope + " chosen geometry differs")
        close(record["selected_alpha_by_horizon"], chosen["selected_alpha_by_horizon"], scope + " selected penalty", exact=True)
        choice_row(scope, "common", chosen["candidate_id"], chosen["scaler"], chosen["cap"], chosen["selected_alpha_by_horizon"])
        checks += 9
        if scope == "half":
            close(training["B_half_common"], validation[f"crossfit__half__{chosen['candidate_id']}__common"], "Half-A B common identity", exact=True)
            require(record["B_prediction_sha256"] == array_hash(training["B_half_common"]), "Half-A prediction hash differs")
            checks += 2
        else:
            fold = record["fold_id"]; heldout = np.flatnonzero(groups % 4 == fold)
            close(record["heldout_row_indices"], heldout, scope + " held-out rows", exact=True)
            close(record["heldout_episode_ids"], np.unique(groups[heldout]), scope + " held-out episodes", exact=True)
            require(len(record["training_episode_ids"]) == 48 and len(record["heldout_episode_ids"]) == 16
                    and not set(record["training_episode_ids"]).intersection(record["heldout_episode_ids"]),
                    scope + " whole-episode exclusion differs")
            require(record["heldout_targets_accepted_by_fit"] is False, scope + " common fit accepted held-out targets")
            for field, expected in (("training_group_sha256", groups[train_rows]), ("heldout_group_sha256", groups[heldout]),
                                    ("training_rows_sha256", train_rows), ("heldout_rows_sha256", heldout),
                                    ("heldout_history_sha256", ax[heldout]), ("OOF_prediction_sha256", training["A_oof_common"][heldout])):
                require(record[field] == array_hash(expected), scope + " fold array identity differs: " + field)
            require(record["state_before_OOF"] == record["state_after_OOF"] == record["selected_state_sha256"],
                    scope + " OOF querying changed selected common state")
            fold_rows.append({**identity, "fold_id": fold, "training_episodes": 48, "heldout_episodes": 16,
                              "training_rows": len(train_rows), "heldout_rows": len(heldout),
                              "training_episode_ids": json.dumps(record["training_episode_ids"]),
                              "heldout_episode_ids": json.dumps(record["heldout_episode_ids"]),
                              "selected_candidate": chosen["candidate_id"], "selected_scaler": chosen["scaler"],
                              "selected_cap": chosen["cap"], "normalizer_h1": float(normalizer[0]),
                              "normalizer_h4": float(normalizer[1]), "normalizer_h12": float(normalizer[2]),
                              "OOF_prediction_sha256": record["OOF_prediction_sha256"],
                              "selected_state_sha256": record["selected_state_sha256"],
                              "whole_episode_exclusion_verified": True, "training_only_normalizer_verified": True})
            checks += 11

    require(set(selection["target_modes"]) == {"crossfit", "insample"}, "Residual target mode roster differs")
    for mode, methods in selection["target_modes"].items():
        require(set(methods) == {"kahm", "rbf", "matched_rbf"}, "Correction method roster differs")
        residual = training["A_oof_residual" if mode == "crossfit" else "A_insample_residual"]
        for kind in ("kahm", "rbf"):
            record = methods[kind]; candidates = record["candidates"]
            require(record["candidate_order"] == common_order and [r["candidate_id"] for r in candidates] == common_order,
                    "Departure candidate roster/order differs")
            for candidate in candidates:
                key = candidate["state_key"]
                require(key == f"{mode}__{candidate['candidate_id']}__{kind}", "Departure candidate state identity differs")
                prediction = validation["crossfit__" + key]
                paths = finite(validation["crossfit__" + key + "__alpha_paths"], (len(ALPHAS), 256, 3), key + " alpha paths")
                loss_paths = np.mean((by[None, :, :] - paths) ** 2, axis=1)
                indices = np.argmin(loss_paths, axis=0)
                close(candidate["B_mse_by_alpha_horizon"], loss_paths, key + " alpha loss replay")
                close(candidate["selected_alpha_indices"], indices, key + " alpha indices", exact=True)
                close(candidate["selected_alpha_by_horizon"], np.asarray(ALPHAS)[indices], key + " selected alpha", exact=True)
                reconstructed = np.column_stack([paths[indices[h], :, h] for h in range(3)])
                close(prediction, reconstructed, key + " selected alpha forecast")
                mse = _mse(prediction, by)
                close(candidate["mse_by_horizon"], mse, key + " B MSE")
                close(candidate["score"], np.mean(mse / scale), key + " B normalized score")
                require(candidate["B_prediction_sha256"] == array_hash(prediction)
                        and candidate["B_alpha_prediction_sha256"] == array_hash(paths)
                        and candidate["residual_target_sha256"] == array_hash(residual)
                        and candidate["shared_baseline_sha256"] == diagnostic["shared_baseline_sha256"],
                        key + " correction input/output identities differ")
                require(candidate["A_mean_centered_feature_max_abs"] < 1e-10, key + " A features not centered")
                mean_correction = finite(candidate["A_mean_centered_correction"], (3,), key + " training correction mean")
                require(np.max(np.abs(mean_correction)) <= 1e-10 * max(1.0, float(np.max(np.abs(residual)))),
                        key + " A correction not centered")
                score_row(mode, kind, candidate["candidate_id"], candidate["scaler"], candidate["cap"], mse, scale,
                          candidate["candidate_id"] == record["selected_candidate"], candidate["selected_alpha_by_horizon"])
                checks += 9
            index = int(np.argmin([r["score"] for r in candidates])); chosen = candidates[index]
            require(record["selected_index"] == index and record["selected_candidate"] == chosen["candidate_id"],
                    mode + "/" + kind + " B argmin/tie differs")
            require(record["selected_scaler"] == chosen["scaler"] and record["selected_cap"] == chosen["cap"], "Selected departure geometry differs")
            close(record["selected_alpha_by_horizon"], chosen["selected_alpha_by_horizon"], "Selected departure penalty", exact=True)
            choice_row(mode, kind, chosen["candidate_id"], chosen["scaler"], chosen["cap"], chosen["selected_alpha_by_horizon"])
            checks += 3
        matched, kahm = methods["matched_rbf"], methods["kahm"]
        match_record = next(r for r in methods["rbf"]["candidates"] if r["candidate_id"] == kahm["selected_candidate"])
        require(matched["selected_candidate"] == kahm["selected_candidate"]
                and matched["selected_scaler"] == kahm["selected_scaler"]
                and matched["selected_cap"] == kahm["selected_cap"], "Matched RBF is not at KAHM scaler/cap")
        close(matched["selected_alpha_by_horizon"], match_record["selected_alpha_by_horizon"], "Matched RBF own penalty", exact=True)
        choice_row(mode, "matched_rbf", matched["selected_candidate"], matched["selected_scaler"], matched["selected_cap"],
                   matched["selected_alpha_by_horizon"], selection_rule="KAHM scaler/cap; RBF own B penalty")
        checks += 2

    for h, horizon in enumerate(HORIZONS):
        delta = training["A_oof_common"][:, h] - training["A_full_common"][:, h]
        training_diagnostics.append({**identity, "horizon": horizon,
            "OOF_common_training_mse": float(np.mean(training["A_oof_residual"][:, h] ** 2)),
            "insample_common_training_mse": float(np.mean(training["A_insample_residual"][:, h] ** 2)),
            "OOF_common_training_smse": float(np.mean(training["A_oof_residual"][:, h] ** 2) / scale[h]),
            "insample_common_training_smse": float(np.mean(training["A_insample_residual"][:, h] ** 2) / scale[h]),
            "OOF_minus_full_common_mean": float(delta.mean()), "OOF_minus_full_common_rms": float(np.sqrt(np.mean(delta * delta))),
            "scope": "Descriptive A-training diagnostic; not a test loss or inferential unit"})
    return {"checks": checks, "B_candidate_scores": scores, "B_selections": choices,
            "crossfit_audit": fold_rows, "shared_biases": bias_rows, "training_diagnostics": training_diagnostics}


def prediction_identities(predictions, fit):
    """Verify selected forecast identities with the already sealed B/C choices."""
    require(set(predictions) == set(SPECS), "Forecast identity roster differs")
    shape = np.asarray(predictions["legacy__calibrated__ridge"]).shape
    require(len(shape) == 2 and shape[1] == 3, "Forecast horizon shape differs")
    for key, value in predictions.items():
        finite(value, shape, key)
    legacy, cross, policy = fit["legacy"], fit["crossfit"], fit["policy"]
    checks = 0
    ridge = predictions["legacy__calibrated__ridge"]
    close(ridge, predictions["legacy__gamma1__ridge"], "Mode-invariant raw ridge", exact=True)
    checks += 1
    for method in GEOMETRY_METHODS:
        selected = legacy["selections"]["selected"][method]
        gamma = legacy["calibration"][selected][method]
        close(predictions[f"legacy__calibrated__{method}"],
              blend(ridge, predictions[f"legacy__gamma1__{method}"], gamma), "Legacy selected gamma replay " + method)
        checks += 1
    for method in ("raw_rbf", "raw_hgb", "persistence"):
        gamma = legacy["calibration_records"]["shared_records"][method]["gamma_by_horizon"]
        close(predictions[f"legacy__calibrated__{method}"],
              blend(ridge, predictions[f"legacy__gamma1__{method}"], gamma), "Legacy shared gamma replay " + method)
        checks += 1
    legacy_choice = legacy["context_policies"]["selected"]
    require(legacy_choice in LEGACY_POLICY
            and legacy_choice == legacy["calibration_records"]["policies"]["selected"]["selected_method"],
            "Legacy policy selected an ineligible/different procedure")
    for mode in ("calibrated", "gamma1"):
        close(predictions[f"legacy__{mode}__nonkahm_policy"], predictions[f"legacy__{mode}__{legacy_choice}"],
              "Legacy same-identity policy forwarding", exact=True)
        checks += 1
    common = predictions["crossfit__calibrated__common"]
    shared = predictions["crossfit__calibrated__shared_bias"]
    bias = finite(cross["selections"]["shared_bias"]["value"], (3,), "Frozen full-A shared bias")
    half_bias = finite(cross["selections"]["half_common"]["shared_bias"], (3,), "Frozen half-A shared bias")
    close(common, predictions["legacy__gamma1__common"], "Full-A common equals selected unit-multiplier legacy common", exact=True)
    close(shared, common + bias, "Shared full-A common-plus-bias identity", exact=True)
    close(predictions["crossfit__calibrated__half_shared_bias"],
          predictions["crossfit__calibrated__half_common"] + half_bias, "Shared half-A common-plus-bias identity", exact=True)
    checks += 3
    for method in NEW_METHODS[:4]:
        close(predictions[f"crossfit__calibrated__{method}"], predictions[f"crossfit__eta1__{method}"],
              "Common/bias mode invariance " + method, exact=True)
        checks += 1
    for method in CORRECTIONS:
        eta = cross["calibration"]["choices"][method]["selected_eta"]
        close(predictions[f"crossfit__calibrated__{method}"],
              blend(shared, predictions[f"crossfit__eta1__{method}"], eta), "Cross-fit C eta replay " + method)
        for h, value in enumerate(eta):
            if value == 0.0:
                close(predictions[f"crossfit__calibrated__{method}"][:, h], shared[:, h], "Exact eta-zero baseline", exact=True)
                checks += 1
        checks += 1
    for target_mode in ("crossfit", "insample"):
        record = cross["selections"]["target_modes"][target_mode]
        if record["rbf"]["selected_candidate"] == record["matched_rbf"]["selected_candidate"]:
            for mode in ("calibrated", "eta1"):
                close(predictions[f"crossfit__{mode}__{target_mode}_rbf"],
                      predictions[f"crossfit__{mode}__{target_mode}_matched_rbf"], "Same-geometry RBF identity", exact=True)
                checks += 1
    require([r["identity"] for r in policy["candidates"]] == list(POLICY_ORDER), "Expanded policy candidate roster/order differs")
    scores = finite([r["score"] for r in policy["candidates"]], (len(POLICY_ORDER),), "Expanded policy C scores")
    chosen = POLICY_ORDER[int(np.argmin(scores))]
    require(policy["selected_identity"] == chosen and policy["selection_split"] == "C", "Expanded policy argmin/tie differs")
    for mode in ("calibrated", "eta1"):
        close(predictions[f"policy__{mode}__expanded_nonkahm"], predictions[policy_key(chosen, calibrated=mode == "calibrated")],
              "Expanded same-identity policy forwarding", exact=True)
        checks += 1
    return checks + 2


def calibration_audit(meta, fit, calibration_predictions, arrays):
    """Recalculate selected-procedure C score paths; no terminal data are used."""
    cx = finite(arrays["C_X"], (256, 12), "C histories")
    cy = finite(arrays["C_Y"], (256, 3), "C outcomes")
    scale = finite(arrays["loss_scale"], (3,), "A target scale")
    legacy, cross, policy = fit["legacy"], fit["crossfit"], fit["policy"]
    checks = prediction_identities(calibration_predictions, fit)
    rows = []
    identity = {key: meta[key] for key in META}
    for record in (legacy["calibration_records"], cross["calibration"]):
        require(record["C_X_sha256"] == array_hash(cx) and record["C_Y_sha256"] == array_hash(cy), "C data hash differs")
        checks += 1

    def replay(scope, method, record, base, raw, calibrated, expected_multiplier):
        candidates = record["candidates"]
        require([r["gamma"] for r in candidates] == list(GAMMAS), "C multiplier grid/order differs")
        losses = np.array([_mse(blend(base, raw, np.full(3, value)), cy) for value in GAMMAS])
        close([r["mse_by_horizon"] for r in candidates], losses, scope + "/" + method + " C score path")
        require(record["all_positive_candidates_failed"] is False, "Unexpected all-positive calibration failure")
        indices = np.argmin(losses, axis=0); multiplier = np.asarray(GAMMAS)[indices]
        close(record["selected_indices"], indices, "C selected indices", exact=True)
        close(record["gamma_by_horizon"], multiplier, "C first-minimum multiplier", exact=True)
        close(expected_multiplier, multiplier, "Saved C multiplier identity", exact=True)
        close(record["selected_mse_by_horizon"], _mse(calibrated, cy), "C calibrated loss replay")
        for g, value in enumerate(GAMMAS):
            for h, horizon in enumerate(HORIZONS):
                rows.append({**identity, "scope": scope, "method": method, "horizon": horizon,
                             "candidate": float(value), "score": float(losses[g, h]),
                             "score_type": "horizon_MSE", "selected": bool(indices[h] == g), "selection_split": "C"})
        return 7

    ridge = calibration_predictions["legacy__calibrated__ridge"]
    for method in GEOMETRY_METHODS:
        candidate = legacy["selections"]["selected"][method]
        record = legacy["calibration_records"]["candidate_records"][candidate][method]
        checks += replay("legacy", method, record, ridge,
                         calibration_predictions[f"legacy__gamma1__{method}"], calibration_predictions[f"legacy__calibrated__{method}"],
                         legacy["calibration"][candidate][method])
    for method in ("raw_rbf", "raw_hgb", "persistence"):
        record = legacy["calibration_records"]["shared_records"][method]
        checks += replay("legacy", method, record, ridge,
                         calibration_predictions[f"legacy__gamma1__{method}"], calibration_predictions[f"legacy__calibrated__{method}"],
                         record["gamma_by_horizon"])
    shared = calibration_predictions["crossfit__calibrated__shared_bias"]
    for method in CORRECTIONS:
        record = cross["calibration"]["choices"][method]
        checks += replay("crossfit", method, record, shared,
                         calibration_predictions[f"crossfit__eta1__{method}"], calibration_predictions[f"crossfit__calibrated__{method}"],
                         record["selected_eta"])
    for method in NEW_METHODS:
        require(cross["calibration"]["calibrated_prediction_sha256"][method]
                == array_hash(calibration_predictions[f"crossfit__calibrated__{method}"])
                and cross["calibration"]["uncalibrated_prediction_sha256"][method]
                == array_hash(calibration_predictions[f"crossfit__eta1__{method}"]), "Saved C prediction hash differs")
        checks += 1
    legacy_policy = legacy["calibration_records"]["policies"]["selected"]
    require(legacy_policy["candidate_order"] == list(LEGACY_POLICY), "Legacy C policy order differs")
    legacy_scores = {method: float(np.mean(_mse(calibration_predictions[f"legacy__calibrated__{method}"], cy) / scale))
                     for method in LEGACY_POLICY}
    close([legacy_policy["scores"][method] for method in LEGACY_POLICY], list(legacy_scores.values()), "Legacy policy C score replay")
    require(legacy_policy["selected_method"] == min(LEGACY_POLICY, key=legacy_scores.__getitem__), "Legacy C policy argmin differs")
    for method in LEGACY_POLICY:
        rows.append({**identity, "scope": "legacy_policy", "method": "nonkahm_policy", "horizon": "",
                     "candidate": method, "score": legacy_scores[method], "score_type": "mean_horizon_A_normalized_MSE",
                     "selected": method == legacy_policy["selected_method"], "selection_split": "C"})
    close(policy["loss_scale"], scale, "Expanded policy A normalization", exact=True)
    require(policy["C_Y_sha256"] == array_hash(cy), "Expanded policy C target identity differs")
    for row in policy["candidates"]:
        pred = calibration_predictions[policy_key(row["identity"])]
        mse = _mse(pred, cy)
        close(row["mse_by_horizon"], mse, "Expanded policy C MSE replay")
        close(row["score"], np.mean(mse / scale), "Expanded policy C normalized score replay")
        require(row["C_prediction_sha256"] == array_hash(pred), "Expanded policy C forecast hash differs")
        rows.append({**identity, "scope": "expanded_policy", "method": "expanded_nonkahm", "horizon": "",
                     "candidate": row["identity"], "score": row["score"], "score_type": "mean_horizon_A_normalized_MSE",
                     "selected": row["identity"] == policy["selected_identity"], "selection_split": "C"})
        checks += 3
    return checks + 5, rows


def query_audit(meta, fit, diagnostics):
    legacy, cross = diagnostics["legacy"], diagnostics["crossfit"]
    require(legacy["state_before"] == legacy["state_after"], "D query changed legacy fitting state")
    # The inherited whole-state hash serializes a Python object graph. Pickle
    # reload can normalize aliases without changing numerical state, so compare
    # it within each process. The release gate verifies the exact sealed state
    # file; common/bias component hashes below also bridge C and D.
    require(cross["frozen_state_before"] == cross["frozen_state_after"],
            "D query changed cross-fit fitting state")
    for field in ("common_state_sha256", "bias_sha256", "shared_baseline_sha256"):
        require(cross[field] == fit["crossfit"]["calibration"][field], "D common/bias identity changed: " + field)
    require(set(cross["bounds"]) == set(CORRECTIONS), "D centered-bound roster differs")
    for method, record in cross["bounds"].items():
        require(record["query_bound_violations"] == 0 and record["coefficient_bound_violations"] == 0,
                "Centered departure bound violation")
        observed = finite(record["observed_max_abs_correction_by_horizon"], (3,), "Observed D correction bound")
        limit = finite(record["correction_bound_by_horizon"], (3,), "Declared D correction bound")
        require(np.all(observed <= limit * (1 + 1e-10) + 1e-12), "Observed D correction exceeds centered bound")
        close(cross["etas"][method], fit["crossfit"]["calibration"]["choices"][method]["selected_eta"], "D eta identity", exact=True)
    require(all(r["coefficient_bound_violations"] == 0 for r in legacy["bounds"]), "Legacy coefficient bound violation")
    return {**{key: meta[key] for key in META}, "legacy_state_unchanged": True, "crossfit_state_unchanged": True,
            "common_and_bias_state_unchanged": True, "centered_bound_checks": len(CORRECTIONS),
            "centered_bound_violations": 0, "legacy_bound_records": len(legacy["bounds"]),
            "common_state_sha256": cross["common_state_sha256"], "shared_baseline_sha256": cross["shared_baseline_sha256"],
            "C_in_process_state_sha256": fit["crossfit"]["calibration"]["pre_C_hash"],
            "D_loaded_in_process_state_sha256": cross["frozen_state_before"],
            "state_integrity_scope": "Exact sealed serialized file; within-process whole-state invariance; cross-stage common/bias hashes"}


def analyze(root=ROOT, output=None):
    root = Path(root).resolve()
    config = read_json(root / "config.json")
    validate_config(config)
    if str(ROOT / "code") not in sys.path:
        sys.path.insert(0, str(ROOT / "code"))
    contract_name = "_crossfit_pipeline_contract"
    contract_spec = importlib.util.spec_from_file_location(contract_name, ROOT / "code/pipeline.py")
    require(contract_spec is not None and contract_spec.loader is not None, "Cannot load the frozen cross-fit pipeline contract")
    contract = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_name] = contract
    contract_spec.loader.exec_module(contract)
    # No terminal target archive is opened before this complete seal gate.
    release = contract.verify_release(root)
    freeze = read_json(root / "freeze.json")
    require(freeze.get("files", {}).get("analysis_code/analyze.py") == digest(Path(__file__)),
            "Analyzer is not the frozen scientific source")
    abc = read_json(root / "prepared/abc_manifest.json")
    terminal = read_json(root / "prepared/d_manifest.json")
    tasks, terminal_tasks = manifest_roster(abc), manifest_roster(terminal)
    require(set(tasks) == set(terminal_tasks), "ABC and D task rosters differ")
    out = Path(output).resolve() if output else root / "analysis"
    out.mkdir(parents=True, exist_ok=False)
    tables = {name: [] for name in ("task_horizon", "task_losses", "stability_screens", "B_candidate_scores", "B_selections",
              "crossfit_audit", "shared_biases", "training_diagnostics", "C_scores_choices", "query_integrity",
              "task_resources", "input_integrity")}
    development_checks = calibration_checks = forecast_checks = 0
    for index, task_id in enumerate(sorted(tasks), 1):
        meta, target_meta = tasks[task_id], terminal_tasks[task_id]
        require(all(meta[name] == target_meta[name] for name in META), "ABC/D task metadata differs")
        input_path = confined(root, meta["input"])
        target_path = confined(root, target_meta["target"])
        folder = root / "results" / task_id
        prediction_path = root / "predictions" / task_id / "predictions.npz"
        fit = read_json(folder / "fit_summary.json")
        require(fit["status"] == "completed" and fit["task"] == meta, "Incomplete or mismatched fit record")
        require(digest(input_path) == meta["sha256"] and digest(target_path) == target_meta["target_sha256"],
                "Prepared input/target digest differs")
        with np.load(input_path, allow_pickle=False) as z:
            require(not any(key.startswith("D_") for key in z.files), "Development archive contains terminal arrays")
            arrays = {key: z[key] for key in ("A_X", "A_Y", "A_groups", "B_X", "B_Y", "C_X", "C_Y", "loss_scale")}
        with np.load(folder / "training_arrays.npz", allow_pickle=False) as z:
            training = {key: z[key] for key in z.files}
        with np.load(folder / "validation_predictions.npz", allow_pickle=False) as z:
            validation = {key: z[key] for key in z.files}
        pre_c = read_json(folder / "pre_c_selection.json")
        audited = development_audit(meta, fit, pre_c, arrays, training, validation)
        development_checks += audited.pop("checks")
        for key, rows in audited.items():
            tables[key].extend(rows)
        with np.load(folder / "calibration_predictions.npz", allow_pickle=False) as z:
            calibration = {key: z[key] for key in z.files}
        count, c_rows = calibration_audit(meta, fit, calibration, arrays)
        calibration_checks += count; tables["C_scores_choices"].extend(c_rows)
        with np.load(prediction_path, allow_pickle=False) as z:
            predictions = {key: z[key] for key in z.files}
        count = prediction_identities(predictions, fit); forecast_checks += count
        query = read_json(root / "predictions" / task_id / "query_diagnostics.json")
        tables["query_integrity"].append(query_audit(meta, fit, query))
        # Targets are read only after seals and selected-forecast identities pass.
        with np.load(target_path, allow_pickle=False) as z:
            targets = z["D_Y"]
        horizon_rows, task_rows, screen = score_task(meta, predictions, targets, arrays["loss_scale"])
        tables["task_horizon"].extend(horizon_rows); tables["task_losses"].extend(task_rows)
        tables["stability_screens"].append(screen)
        prediction_receipt = read_json(root / "predictions" / task_id / "prediction_receipt.json")
        tables["task_resources"].append({**{key: meta[key] for key in META},
            "fit_resources_json": json.dumps(fit["resources"], sort_keys=True),
            "prediction_wall_seconds": prediction_receipt["wall_seconds"],
            "prediction_cpu_seconds": prediction_receipt["cpu_seconds"]})
        tables["input_integrity"].append({"task_id": task_id, "ABC_file_sha256": digest(input_path),
            "target_file_sha256": digest(target_path), "target_array_sha256": array_hash(targets),
            "prediction_file_sha256": digest(prediction_path), "training_file_sha256": digest(folder / "training_arrays.npz"),
            "calibration_file_sha256": digest(folder / "calibration_predictions.npz"),
            "sealed_state_file_sha256": prediction_receipt["state_sha256"],
            "forecast_arrays": len(predictions), "terminal_prediction_identity_checks": count})
        if index % 24 == 0:
            print(json.dumps({"analyzed_tasks": index, "expected_tasks": TASKS}), flush=True)
    require(len(tables["task_horizon"]) == TASKS * ARRAY_COUNT * 3 and len(tables["task_losses"]) == TASKS * ARRAY_COUNT,
            "Scored task/horizon counts differ")
    require(len(tables["crossfit_audit"]) == TASKS * 4 and len(tables["stability_screens"]) == TASKS,
            "Cross-fit or stability audit task counts differ")
    reps = repetition_table(tables["task_losses"]); vector = vector_lookup(reps)
    primary, primary_detail = primary_tables(vector)
    descriptive, descriptive_detail = descriptive_tables(vector)
    tables.update(repetition_losses=reps,
                  panel_summary=average_rows(reps, ("panel", *IDENTITY), spread=True),
                  condition_summary=average_rows(tables["task_losses"], ("law", "noise_level", *IDENTITY), spread=True),
                  condition_horizon_summary=average_rows(tables["task_horizon"], ("law", "noise_level", "horizon", *IDENTITY), spread=True),
                  primary_contrasts=primary, primary_contrast_repetitions=primary_detail,
                  descriptive_contrasts=descriptive, descriptive_contrast_repetitions=descriptive_detail)
    fields = ("law", "noise_level", "scope", "method", "scaler", "cap")
    tables["B_selection_counts"] = [{**dict(zip(fields, key)), "count": len(rows)}
                                    for key, rows in grouped(tables["B_selections"], fields).items()]
    c_choice_fields = ("law", "noise_level", "scope", "method", "horizon", "candidate")
    tables["C_choice_counts"] = [{**dict(zip(c_choice_fields, key)), "count": len(rows)}
                                for key, rows in grouped([r for r in tables["C_scores_choices"] if r["selected"]], c_choice_fields).items()]
    exported = {}
    for name, rows in tables.items():
        path = out / (name + ".csv"); export_csv(path, rows)
        exported[path.name] = {"rows": len(rows), "sha256": digest(path)}
    established = sum(r["criterion_established"] for r in primary)
    screens = tables["stability_screens"]
    summary = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_schema": config["schema_version"], "task_count": TASKS, "independent_repetitions": REPETITIONS,
        "forecast_arrays_per_task": ARRAY_COUNT, "primary_family_size": FAMILY_SIZE,
        "primary_comparisons_established": established, "all_four_primary_criteria_established": established == FAMILY_SIZE,
        "maximum_error_screens_passed": all(r["screen_pass"] for r in screens),
        "maximum_selected_error_ratio": max(r["maximum_error_ratio"] for r in screens),
        "failed_stability_tasks": [r["task_id"] for r in screens if not r["screen_pass"]],
        "primary_contrasts": primary,
        "primary_interpretation": "Four distinct calibrated cross-fit KAHM claims: 3% gains over shared bias, independently selected cross-fit RBF, and the expanded C-selected policy; linear 5% noninferiority to ridge. Matched RBF is secondary.",
        "inference_scope": "Approximate Bonferroni one-sided paired-t bounds at family alpha .05 over 24 independent repetition aggregates, df23; not distribution-free certificates.",
        "secondary_scope": "All secondary intervals are descriptive unadjusted 95% paired-t intervals. Cross-fit versus in-sample comparisons retain common/bias and the feature candidate construction; their own B/C selections may differ.",
        "crossfit_scope": "OOF common models exclude whole episodes and independently tune using complement-only fit/normalization and B. Their residuals are transferred to the separately fitted full-A common predictor; cross-fitting does not guarantee beneficial transfer.",
        "scoring_scope": "Only the 42 frozen logical forecast arrays are scored. Unit-multiplier and policy views preserve selected identities; discarded candidate D predictions are not planned outcomes.",
        "generalization_scope": "Fresh repetitions of four specified numerical laws at history width12 and three horizons; no natural-data deployment or theoretical certificate claim.",
        "integrity": {"release_receipt": json_safe(release), "verified_before_terminal_target_access": True,
                      "development_identity_checks": development_checks, "calibration_identity_checks": calibration_checks,
                      "terminal_prediction_identity_checks": forecast_checks,
                      "config_sha256": digest(root / "config.json"), "freeze_sha256": digest(root / "freeze.json"),
                      "ABC_manifest_sha256": digest(root / "prepared/abc_manifest.json"),
                      "D_manifest_sha256": digest(root / "prepared/d_manifest.json")},
        "tables": exported,
    }
    write_json(out / "analysis_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.output)
    print(json.dumps({key: result[key] for key in ("status", "task_count", "primary_comparisons_established", "maximum_error_screens_passed")}))
