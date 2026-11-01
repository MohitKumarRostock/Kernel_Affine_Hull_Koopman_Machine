#!/usr/bin/env python3
"""Analyze the frozen preprocessing--dimension study from sealed forecasts.

No terminal outcome is used to fit, calibrate, or select a forecasting
procedure. The inferential units are the 24 independent repetition aggregates.
The fixed-common residual experiment has a distinct sample allocation and is
reported separately from the primary complete-procedure experiment.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
NONLINEAR = LAWS[1:]
PANELS = {"nonlinear": NONLINEAR, "linear": LAWS[:1], "all_descriptive": LAWS}
HORIZONS = (1, 4, 12)
CAPS = (1, 2, 4, 8, 12, 20)
REPETITIONS = 24
MODES = ("calibrated", "gamma1")
METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common",
           "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy")
GEOMETRY = ("common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded")
RAW = ("ridge", "raw_rbf", "raw_hgb", "persistence")
PAIRS = ("standard", "minmax", "tanh", "hybrid_tanh_minmax", "hybrid_minmax_tanh")
CONTEXTS = ("selected", "cap2", "cap20", *PAIRS)
SECONDARY_PAIRS = ("ss", "mm", "tt", "tm", "mt")
COMMON_SCALER = {"ss": "standard", "mm": "minmax", "tt": "tanh_z_1", "tm": "tanh_z_1", "mt": "minmax"}
META = ("task_id", "law", "history", "repetition", "seed", "noise_index", "noise_level")
IDENTITY = ("family", "context", "pair", "cap", "view", "mode", "method")
FAMILY_ALPHA = 0.05
FAMILY_SIZE = 4
SUPERIORITY_MARGIN = 0.03
NONINFERIORITY_MARGIN = 0.05
STABILITY_RATIO = 5.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in "fiu" and np.isfinite(result).all(),
            "Invalid array: " + label)
    return result


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
    require(bool(rows), "Cannot export an empty result table: " + str(path))
    fields = list(rows[0])
    require(all(set(row) == set(fields) for row in rows), "Heterogeneous CSV columns: " + str(path))
    with Path(path).open("x", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def forecast_specs():
    output = {}
    for context in CONTEXTS:
        for mode in MODES:
            for method in METHODS:
                key = f"primary__{context}__{mode}__{method}"
                output[key] = dict(family="primary", context=context, pair=context if context in PAIRS else "",
                                   cap="", view="primary__" + context, mode=mode, method=method)
    for pair in PAIRS:
        for cap in CAPS:
            for mode in MODES:
                for method in GEOMETRY:
                    key = f"candidate__{pair}__cap{cap}__{mode}__{method}"
                    output[key] = dict(family="candidate", context="fixed_pair_and_cap", pair=pair, cap=cap,
                                       view=f"candidate__{pair}__cap{cap}", mode=mode, method=method)
    for mode in MODES:
        for scaler in ("standard", "minmax", "tanh_z_1"):
            for method in ("common", "shared_bias"):
                key = f"secondary__{mode}__{method}__{scaler}"
                output[key] = dict(family="secondary", context=scaler, pair="", cap="",
                                   view="secondary__" + scaler, mode=mode, method=method)
        for pair in SECONDARY_PAIRS:
            for method in ("kahm", "rbf", "matched_rbf"):
                key = f"secondary__{mode}__{pair}__{method}"
                output[key] = dict(family="secondary", context=pair, pair=pair, cap="",
                                   view="secondary__" + pair, mode=mode, method=method)
    require(len(output) == 502, "Internal forecast roster count differs")
    return output


SPECS = forecast_specs()


def validate_config(config):
    require(config["master_seed"] == 2026110117, "Unexpected master seed")
    require(config["repetitions"] == REPETITIONS and config["task_count"] == 192, "Production repetition/task count differs")
    require(tuple(config["laws"]) == LAWS and tuple(config["horizons"]) == HORIZONS, "Observation laws/horizons differ")
    require(config["histories"] == [12] and config["noise_levels"] == [0.0, 0.1], "History/noise design differs")
    require(config["split_episodes"] == {"A": 64, "B": 32, "C": 32, "D": 128}, "Episode allocation differs")
    require(config["split_caps"] == {"A": 512, "B": 256, "C": 256, "D": None}, "Retained-row allocation differs")
    require(config["origins_per_episode"] == 13, "Origin count differs")
    analysis = config["analysis"]
    require(analysis["family_alpha"] == FAMILY_ALPHA and analysis["primary_family_size"] == FAMILY_SIZE,
            "Primary family allocation differs")
    require(analysis["superiority_margin"] == SUPERIORITY_MARGIN
            and analysis["linear_noninferiority_margin"] == NONINFERIORITY_MARGIN,
            "Primary practical margins differ")


def manifest_roster(manifest):
    tasks = manifest["tasks"]
    expected = {(law, 12, noise, rep) for law in LAWS for noise in (0, 1) for rep in range(REPETITIONS)}
    require(len(tasks) == 192 and len({t["task_id"] for t in tasks}) == 192, "Incomplete/duplicated manifest task roster")
    require({(t["law"], t["history"], t["noise_index"], t["repetition"]) for t in tasks} == expected,
            "Manifest factorial differs")
    return {task["task_id"]: task for task in tasks}


def blend(base, forecast, gamma):
    gamma = finite(gamma, (3,), "calibration multiplier")
    require(np.isin(gamma, (0.0, 0.25, 0.5, 1.0)).all(), "Undeclared calibration multiplier")
    result = base.copy()
    for h, value in enumerate(gamma):
        if value != 0.0:
            result[:, h] = base[:, h] + value * (forecast[:, h] - base[:, h])
    return result


def prediction_identities(predictions, fit):
    """Check saved prediction views and C blending without any new selection."""
    checks = 0
    baseline = predictions["primary__selected__calibrated__ridge"]
    for mode in MODES:
        for context in CONTEXTS:
            for method in RAW:
                close(predictions[f"primary__{context}__{mode}__{method}"],
                      predictions[f"primary__selected__{mode}__{method}"],
                      f"shared raw procedure {context}/{mode}/{method}", exact=True)
                checks += 1
            close(predictions[f"primary__{context}__{mode}__ridge"], baseline, "raw ridge mode invariant", exact=True)
            checks += 1
        for pair in PAIRS:
            for method in GEOMETRY:
                close(predictions[f"candidate__{pair}__cap12__{mode}__{method}"],
                      predictions[f"candidate__{pair}__cap20__{mode}__{method}"],
                      f"native cap12/cap20 {pair}/{mode}/{method}", exact=True)
                checks += 1
            for cap in CAPS:
                for hybrid, diagonal in (("hybrid_tanh_minmax", "tanh"), ("hybrid_minmax_tanh", "minmax")):
                    if pair == hybrid:
                        close(predictions[f"candidate__{hybrid}__cap{cap}__{mode}__common"],
                              predictions[f"candidate__{diagonal}__cap{cap}__{mode}__common"],
                              "hybrid common feature head identity", exact=True)
                        checks += 1
        for scaler in ("standard", "minmax", "tanh_z_1"):
            for method in ("common", "shared_bias"):
                close(predictions[f"secondary__{mode}__{method}__{scaler}"],
                      predictions[f"secondary__calibrated__{method}__{scaler}"],
                      "fixed secondary common/bias mode invariant", exact=True)
                checks += 1
    selections = fit["selections"]
    require(set(selections) == set(CONTEXTS), "Frozen context selection roster differs")
    policies = fit["context_policies"]
    calibration = fit["calibration"]
    ordered_candidates = [f"{pair}__cap{cap}" for cap in CAPS for pair in PAIRS]
    require(set(fit["b_scores"]) == set(ordered_candidates), "B candidate score roster differs")
    for context in CONTEXTS:
        require(set(selections[context]) == set(GEOMETRY), "Incomplete frozen geometry selections")
        for method, candidate_id in selections[context].items():
            require(candidate_id in {f"{p}__cap{c}" for p in PAIRS for c in CAPS}, "Undeclared selected candidate")
            eligible = ordered_candidates
            if context in ("cap2", "cap20"):
                eligible = [key for key in eligible if key.endswith("__" + context)]
            elif context in PAIRS:
                eligible = [key for key in eligible if key.startswith(context + "__cap")]
            if method == "common" and context in ("selected", "cap2", "cap20"):
                eligible = [key for key in eligible if key.rsplit("__cap", 1)[0] in ("standard", "minmax", "tanh")]
            require(all(np.isfinite(fit["b_scores"][key][method]) for key in eligible), "Nonfinite B score")
            require(candidate_id == min(eligible, key=lambda key: fit["b_scores"][key][method]), "B argmin/tie replay differs")
            checks += 1
            for mode in MODES:
                close(predictions[f"primary__{context}__{mode}__{method}"],
                      predictions[f"candidate__{candidate_id}__{mode}__{method}"],
                      f"B-frozen context view {context}/{method}", exact=True)
                checks += 1
        policy_method = policies[context]
        if isinstance(policy_method, dict):
            policy_method = policy_method["selected_method"]
        require(policy_method in ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded"),
                "Non-KAHM policy selected an ineligible procedure")
        for mode in MODES:
            close(predictions[f"primary__{context}__{mode}__nonkahm_policy"],
                  predictions[f"primary__{context}__{mode}__{policy_method}"],
                  "C-frozen policy forwarding", exact=True)
            checks += 1
    for pair in PAIRS:
        for cap in CAPS:
            candidate_id = f"{pair}__cap{cap}"
            for method in GEOMETRY:
                gamma = calibration[candidate_id][method]
                close(predictions[f"candidate__{candidate_id}__calibrated__{method}"],
                      blend(baseline, predictions[f"candidate__{candidate_id}__gamma1__{method}"], gamma),
                      "C multiplier replay " + candidate_id + "/" + method)
                checks += 1
    return checks


def secondary_identities(predictions, secondary):
    diagnostics, selection = secondary["diagnostics"], secondary["selections"]
    calibration = secondary["calibration"]
    checks = 0
    pids, rids = diagnostics["P_episode_ids"], diagnostics["R_episode_ids"]
    require(len(pids) == len(rids) == 32 and not set(pids).intersection(rids), "Secondary P/R episode allocation differs")
    require(diagnostics["P_target_access_only_for_common"] is True, "Secondary common model has undeclared target access")
    require(calibration["frozen_state_before"] == calibration["frozen_state_after"], "C changed secondary development state")
    for arm in ("standard", "minmax", "tanh_z_1"):
        row = diagnostics["common_rows"][arm]
        require(row["R_common_target_access"] is False and row["bias_fit_split"] == "B", "Secondary residual or bias roles differ")
        common = predictions[f"secondary__calibrated__common__{arm}"]
        shared = predictions[f"secondary__calibrated__shared_bias__{arm}"]
        close(shared, common + finite(row["bias"], (3,), "secondary shared bias"), "Secondary shared-bias replay", exact=True)
        candidates = selection["common"][arm]["common_candidates"]
        require([record["cap"] for record in candidates] == list(CAPS), "Secondary common cap roster differs")
        chosen = min(candidates, key=lambda record: record["score"])["cap"]
        require(selection["common"][arm]["common_cap"] == chosen, "Secondary common B argmin differs")
        checks += 2
    for pair in SECONDARY_PAIRS:
        record = selection["departure_pairs"][pair]
        arm = COMMON_SCALER[pair]
        require(record["common_scaler"] == arm, "Secondary common row differs")
        require(record["shared_baseline_sha256"] == diagnostics["common_rows"][arm]["common_frozen_state_sha256"],
                "Secondary departures have a different common/bias state")
        require(record["matched_rbf_departure_cap"] == record["selected_departure_caps"]["kahm"], "Matched RBF cap differs from KAHM")
        for kind in ("kahm", "rbf"):
            candidates = record["candidates"][kind]
            require([candidate["cap"] for candidate in candidates] == list(CAPS), "Secondary departure cap roster differs")
            require(record["selected_departure_caps"][kind] == min(candidates, key=lambda candidate: candidate["score"])["cap"],
                    "Secondary departure B argmin differs")
            require(all(candidate["mean_centered_R_feature_max_abs"] < 1e-10 for candidate in candidates),
                    "Secondary R features are not centered")
            checks += 2
        shared = predictions[f"secondary__calibrated__shared_bias__{arm}"]
        for method in ("kahm", "rbf", "matched_rbf"):
            key = pair + "__" + method
            eta = calibration["choices"][key]["selected_eta"]
            close(predictions[f"secondary__calibrated__{key}"],
                  blend(shared, predictions[f"secondary__gamma1__{key}"], eta), "Secondary C eta replay")
            checks += 1
        if record["selected_departure_caps"]["kahm"] == record["selected_departure_caps"]["rbf"]:
            for mode in MODES:
                close(predictions[f"secondary__{mode}__{pair}__rbf"], predictions[f"secondary__{mode}__{pair}__matched_rbf"],
                      "Secondary same-cap RBF identity", exact=True)
                checks += 1
    return checks


def score_task(meta, predictions, targets, scale):
    require(set(predictions) == set(SPECS), "The complete 502-array forecast roster differs")
    targets = finite(targets, (1664, 3), "terminal outcomes")
    scale = finite(scale, (3,), "A target variance")
    require((scale >= 1e-12).all(), "Invalid target-variance floor")
    horizon_rows, task_rows, stability = [], [], []
    baseline = predictions["primary__selected__calibrated__ridge"]
    ridge_max_error = float(np.max(np.abs(targets - baseline)))
    for key, identity in SPECS.items():
        prediction = finite(predictions[key], targets.shape, key)
        residual = targets - prediction
        mse = np.mean(residual * residual, axis=0)
        require(np.isfinite(mse).all(), "Nonfinite forecast loss")
        record = {**{name: meta[name] for name in META}, **identity, "prediction_key": key}
        for h, horizon in enumerate(HORIZONS):
            horizon_rows.append({**record, "horizon": horizon, "D_rows": len(targets),
                                 "A_target_variance": float(scale[h]), "mse": float(mse[h]),
                                 "smse": float(mse[h] / scale[h])})
        task_rows.append({**record, "mse": float(mse.mean()), "smse": float(np.mean(mse / scale))})
        if identity["family"] == "primary" and identity["mode"] == "calibrated" and identity["method"] == "kahm_departure":
            maximum = float(np.max(np.abs(residual)))
            ratio = maximum / ridge_max_error if ridge_max_error else (0.0 if maximum == 0.0 else math.inf)
            require(math.isfinite(ratio), "Undefined maximum-error screen")
            stability.append({**record, "maximum_absolute_error": maximum,
                              "ridge_maximum_absolute_error": ridge_max_error, "maximum_error_ratio": ratio,
                              "screen_threshold": STABILITY_RATIO, "screen_pass": bool(ratio <= STABILITY_RATIO)})
    return horizon_rows, task_rows, stability


def grouped(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def average_rows(rows, fields, values=("mse", "smse"), *, spread=False):
    result = []
    for key, members in grouped(rows, fields).items():
        record = {**dict(zip(fields, key)), "n_records": len(members)}
        for name in values:
            data = np.asarray([member[name] for member in members], dtype=float)
            record[name] = float(np.mean(data))
            if spread:
                sd = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
                se = sd / np.sqrt(len(data))
                critical = float(student_t.ppf(0.975, len(data) - 1)) if len(data) > 1 else 0.0
                record[name + "_sd"] = sd
                record[name + "_se"] = float(se)
                record[name + "_lower95_descriptive"] = float(np.mean(data) - critical * se)
                record[name + "_upper95_descriptive"] = float(np.mean(data) + critical * se)
        result.append(record)
    return result


def repetition_table(task_rows, *, view_count=502):
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
    require(len(output) == REPETITIONS * view_count * len(PANELS), "Repetition table has missing forecast views")
    return output


def matched_primary_rows(task_rows, selected_candidate):
    lookup = {row["prediction_key"]: row for row in task_rows}
    result = []
    for mode in MODES:
        for method in GEOMETRY:
            original = lookup[f"candidate__{selected_candidate}__{mode}__{method}"]
            result.append({**original, "family": "matched_primary", "context": "KAHM_selected_pair_cap",
                           "pair": "", "cap": "", "view": "matched_primary__selected",
                           "matched_source_candidate_id": selected_candidate})
    return result


def matched_primary_contrasts(vector):
    result, details = [], []
    view = "matched_primary__selected"
    for panel in PANELS:
        for mode in MODES:
            for comparator in ("common", "rbf_departure", "kahm_unbounded"):
                record, detail = descriptive_contrast("matched_primary_KAHM_minus_" + comparator, panel, mode,
                    vector(panel, view, "kahm_departure", mode), vector(panel, view, comparator, mode),
                    left_view=view, left_method="kahm_departure", right_view=view, right_method=comparator)
                result.append(record)
                details.extend(detail)
    return result, details


def vector_lookup(repetition_rows):
    lookup = {(r["panel"], r["view"], r["mode"], r["method"], r["repetition"]): r["smse"] for r in repetition_rows}
    require(len(lookup) == len(repetition_rows), "Duplicated repetition-forecast identity")

    def vector(panel, view, method, mode="calibrated"):
        values = np.asarray([lookup[panel, view, mode, method, rep] for rep in range(REPETITIONS)], dtype=float)
        require(values.shape == (24,) and np.isfinite(values).all() and np.all(values >= 0), "Invalid paired loss vector")
        return values

    return vector


def primary_tables(vector):
    cases = (
        ("selected_KAHM_vs_nonKAHM_policy", "nonlinear", "selected", "kahm_departure", "selected", "nonkahm_policy", 0.97, "3% superiority"),
        ("selected_KAHM_vs_selected_RBF", "nonlinear", "selected", "kahm_departure", "selected", "rbf_departure", 0.97, "3% superiority"),
        ("tanh_KAHM_vs_standard_KAHM", "nonlinear", "tanh", "kahm_departure", "standard", "kahm_departure", 0.97, "3% superiority"),
        ("linear_selected_KAHM_vs_ridge", "linear", "selected", "kahm_departure", "selected", "ridge", 1.05, "5% noninferiority"),
    )
    summaries, details = [], []
    critical = float(student_t.ppf(1 - FAMILY_ALPHA / FAMILY_SIZE, REPETITIONS - 1))
    for identifier, panel, candidate_context, candidate_method, comparator_context, comparator_method, factor, margin in cases:
        candidate = vector(panel, "primary__" + candidate_context, candidate_method)
        comparator = vector(panel, "primary__" + comparator_context, comparator_method)
        gaps = factor * comparator - candidate
        mean = float(gaps.mean())
        sd = float(gaps.std(ddof=1))
        se = sd / np.sqrt(REPETITIONS)
        lower = float(mean - critical * se)
        p = float(student_t.sf(mean / se, REPETITIONS - 1)) if se else (0.0 if mean > 0 else (1.0 if mean < 0 else 0.5))
        comparison = float(comparator.mean())
        summaries.append({"contrast_id": identifier, "panel": panel, "candidate_context": candidate_context,
                          "candidate": candidate_method, "comparator_context": comparator_context,
                          "comparator": comparator_method, "mode": "calibrated", "margin": margin,
                          "comparator_factor": factor, "candidate_mean_smse": float(candidate.mean()),
                          "comparator_mean_smse": comparison,
                          "point_relative_gain_percent": float(100 * (1 - candidate.mean() / comparison)) if comparison else None,
                          "n_repetitions": REPETITIONS, "df": REPETITIONS - 1,
                          "family_alpha": FAMILY_ALPHA, "family_size": FAMILY_SIZE,
                          "margin_gap_mean": mean, "margin_gap_sample_sd": sd, "margin_gap_se": float(se),
                          "critical_t": critical, "simultaneous_one_sided_lower": lower,
                          "p_one_sided": p, "p_bonferroni": min(1.0, FAMILY_SIZE * p),
                          "positive_margin_repetitions": int(np.count_nonzero(gaps > 0)),
                          "criterion_established": bool(lower > 0)})
        details.extend({"contrast_id": identifier, "panel": panel, "repetition": rep,
                        "candidate_smse": float(candidate[rep]), "comparator_smse": float(comparator[rep]),
                        "comparator_factor": factor, "margin_gap": float(gaps[rep])} for rep in range(REPETITIONS))
    return summaries, details


def descriptive_contrast(identifier, panel, mode, left, right, *, left_view, left_method, right_view, right_method):
    difference = np.asarray(left) - np.asarray(right)
    require(difference.shape == (24,) and np.isfinite(difference).all(), "Invalid secondary contrast")
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(REPETITIONS))
    critical = float(student_t.ppf(0.975, REPETITIONS - 1))
    denominator = float(np.mean(right))
    identity = dict(contrast_id=identifier, panel=panel, mode=mode, left_view=left_view,
                    left_method=left_method, right_view=right_view, right_method=right_method)
    summary = {**identity, "inference": "descriptive_unadjusted_95_percent_paired_t",
               "n_repetitions": REPETITIONS, "left_mean_smse": float(np.mean(left)),
               "right_mean_smse": denominator,
               "relative_gain_percent": float(100 * (1 - np.mean(left) / denominator)) if denominator else None,
               "mean_difference": mean, "standard_error": se, "lower95": mean - critical * se,
               "upper95": mean + critical * se, "better_repetitions": int(np.count_nonzero(difference < 0)),
               "ties": int(np.count_nonzero(difference == 0)), "worse_repetitions": int(np.count_nonzero(difference > 0))}
    details = [{**identity, "repetition": rep, "left_smse": float(left[rep]), "right_smse": float(right[rep]),
                "difference": float(difference[rep])} for rep in range(REPETITIONS)]
    return summary, details


def primary_secondary_tables(vector):
    pairs = [
        ("tanh_minus_standard_KAHM", "tanh", "kahm_departure", "standard", "kahm_departure"),
        ("tanh_minus_minmax_KAHM", "tanh", "kahm_departure", "minmax", "kahm_departure"),
        ("selected_minus_cap2_KAHM", "selected", "kahm_departure", "cap2", "kahm_departure"),
        ("selected_minus_cap20_KAHM", "selected", "kahm_departure", "cap20", "kahm_departure"),
        ("selected_KAHM_minus_RBF", "selected", "kahm_departure", "selected", "rbf_departure"),
        ("selected_KAHM_minus_common", "selected", "kahm_departure", "selected", "common"),
        ("selected_KAHM_minus_nonKAHM", "selected", "kahm_departure", "selected", "nonkahm_policy"),
        ("selected_bounded_minus_unbounded_KAHM", "selected", "kahm_departure", "selected", "kahm_unbounded"),
        ("tanh_common_minmax_departure_minus_tanh_diagonal", "hybrid_tanh_minmax", "kahm_departure", "tanh", "kahm_departure"),
        ("minmax_common_tanh_departure_minus_minmax_diagonal", "hybrid_minmax_tanh", "kahm_departure", "minmax", "kahm_departure"),
    ]
    output, details = [], []
    for panel in PANELS:
        for mode in MODES:
            for identifier, lc, lm, rc, rm in pairs:
                left_view, right_view = "primary__" + lc, "primary__" + rc
                record, detail = descriptive_contrast(identifier, panel, mode,
                    vector(panel, left_view, lm, mode), vector(panel, right_view, rm, mode),
                    left_view=left_view, left_method=lm, right_view=right_view, right_method=rm)
                output.append(record)
                details.extend(detail)
    return output, details


def interaction_record(identifier, panel, mode, terms, vectors, definition, *, architecture):
    values = np.zeros(REPETITIONS)
    for coefficient, term in zip(terms, vectors):
        values += coefficient * term
    require(np.isfinite(values).all(), "Nonfinite interaction")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(REPETITIONS))
    critical = float(student_t.ppf(0.975, REPETITIONS - 1))
    identity = dict(contrast_id=identifier, architecture=architecture, panel=panel, mode=mode, definition=definition)
    summary = {**identity, "inference": "descriptive_unadjusted_95_percent_paired_t", "n_repetitions": REPETITIONS,
               "mean_interaction_smse": mean, "standard_error": se, "lower95": mean - critical * se,
               "upper95": mean + critical * se, "negative_repetitions": int(np.count_nonzero(values < 0)),
               "positive_repetitions": int(np.count_nonzero(values > 0))}
    detail = [{**identity, "repetition": rep, "interaction_smse": float(values[rep])} for rep in range(REPETITIONS)]
    return summary, detail


def interaction_tables(vector):
    output, details = [], []
    for panel in PANELS:
        for mode in MODES:
            for cap in (None, *CAPS):
                def get(pair):
                    view = "primary__" + pair if cap is None else f"candidate__{pair}__cap{cap}"
                    return vector(panel, view, "kahm_departure", mode)
                values = [get(pair) for pair in ("tanh", "hybrid_tanh_minmax", "hybrid_minmax_tanh", "minmax")]
                label = "independently_selected_caps" if cap is None else f"fixed_cap{cap}"
                record, detail = interaction_record("primary_scaling_interaction_" + label, panel, mode,
                    (1, -1, -1, 1), values, "L(tanh,tanh)-L(tanh,minmax)-L(minmax,tanh)+L(minmax,minmax)",
                    architecture="complete_concatenated_heads")
                output.append(record)
                details.extend(detail)
            for cap in (2, 20):
                values = [vector(panel, "primary__tanh", "kahm_departure", mode),
                          vector(panel, "primary__standard", "kahm_departure", mode),
                          vector(panel, f"candidate__tanh__cap{cap}", "kahm_departure", mode),
                          vector(panel, f"candidate__standard__cap{cap}", "kahm_departure", mode)]
                record, detail = interaction_record(f"tanh_dimension_selection_interaction_vs_cap{cap}", panel, mode,
                    (1, -1, -1, 1), values, f"(L(tanh,selected_cap)-L(standard,selected_cap))-(L(tanh,cap{cap})-L(standard,cap{cap}))",
                    architecture="complete_concatenated_heads")
                output.append(record)
                details.extend(detail)
    return output, details


def fixed_common_tables(vector):
    output, details, interactions, interaction_details = [], [], [], []
    for panel in PANELS:
        for mode in MODES:
            for pair in SECONDARY_PAIRS:
                view = "secondary__" + pair
                baseline_view = "secondary__" + COMMON_SCALER[pair]
                pairs = [("kahm", baseline_view, "shared_bias", "KAHM_minus_shared_bias"),
                         ("rbf", baseline_view, "shared_bias", "RBF_minus_shared_bias"),
                         ("kahm", view, "rbf", "KAHM_minus_independently_selected_RBF"),
                         ("kahm", view, "matched_rbf", "KAHM_minus_same_cap_RBF")]
                for left_method, right_view, right_method, label in pairs:
                    record, detail = descriptive_contrast("fixed_common_" + pair + "_" + label, panel, mode,
                        vector(panel, view, left_method, mode), vector(panel, right_view, right_method, mode),
                        left_view=view, left_method=left_method, right_view=right_view, right_method=right_method)
                    output.append(record)
                    details.extend(detail)
            for scaler in ("standard", "minmax", "tanh_z_1"):
                view = "secondary__" + scaler
                record, detail = descriptive_contrast("fixed_common_" + scaler + "_bias_minus_common", panel, mode,
                    vector(panel, view, "shared_bias", mode), vector(panel, view, "common", mode),
                    left_view=view, left_method="shared_bias", right_view=view, right_method="common")
                output.append(record)
                details.extend(detail)
            for left, right in (("tm", "tt"), ("mt", "mm")):
                for method in ("kahm", "rbf", "matched_rbf"):
                    left_view, right_view = "secondary__" + left, "secondary__" + right
                    record, detail = descriptive_contrast(f"fixed_common_departure_change_{left}_minus_{right}_{method}", panel, mode,
                        vector(panel, left_view, method, mode), vector(panel, right_view, method, mode),
                        left_view=left_view, left_method=method, right_view=right_view, right_method=method)
                    output.append(record)
                    details.extend(detail)
            for method in ("kahm", "rbf", "matched_rbf"):
                values = [vector(panel, "secondary__" + pair, method, mode) for pair in ("tt", "tm", "mt", "mm")]
                record, detail = interaction_record("fixed_common_scaling_interaction_" + method, panel, mode,
                    (1, -1, -1, 1), values, "L(tt)-L(tm)-L(mt)+L(mm); fixed common predictor within each row; B-selected departure caps",
                    architecture="honest_residual_fixed_common_centered_slopes")
                interactions.append(record)
                interaction_details.extend(detail)
    return output, details, interactions, interaction_details


def selection_tables(meta, fit):
    records = []
    for context in CONTEXTS:
        for method in GEOMETRY:
            candidate_id = fit["selections"][context][method]
            pair, cap = candidate_id.rsplit("__cap", 1)
            records.append({**{name: meta[name] for name in META}, "context": context, "method": method,
                            "candidate_id": candidate_id, "preprocessing_pair": pair, "selected_cap": int(cap),
                            "selection_split": "B", "selected_before_C": True})
    policies = []
    for context in CONTEXTS:
        selected = fit["context_policies"][context]
        if isinstance(selected, dict):
            selected = selected["selected_method"]
        policies.append({**{name: meta[name] for name in META}, "context": context,
                         "selected_method": selected, "selection_split": "C", "B_candidate_choices_frozen": True})
    gammas = []
    for candidate_id, methods in fit["calibration"].items():
        for method in GEOMETRY:
            gamma = finite(methods[method], (3,), "saved C gamma")
            for h, value in zip(HORIZONS, gamma):
                gammas.append({**{name: meta[name] for name in META}, "candidate_id": candidate_id,
                               "method": method, "horizon": h, "gamma": float(value), "selection_split": "C"})
    scores = []
    for candidate_id, methods in fit["b_scores"].items():
        pair, cap = candidate_id.rsplit("__cap", 1)
        for method in GEOMETRY:
            scores.append({**{name: meta[name] for name in META}, "candidate_id": candidate_id,
                           "preprocessing_pair": pair, "cap": int(cap), "method": method,
                           "B_normalized_mse": float(methods[method]), "selection_split": "B"})
    return records, policies, gammas, scores


def secondary_selection_tables(meta, secondary):
    selection, calibration, diagnostics = secondary["selections"], secondary["calibration"], secondary["diagnostics"]
    identity = {name: meta[name] for name in META}
    cap_rows, eta_rows, bias_rows = [], [], []
    for pair in SECONDARY_PAIRS:
        record = selection["departure_pairs"][pair]
        for method in ("kahm", "rbf", "matched_rbf"):
            cap = record["matched_rbf_departure_cap"] if method == "matched_rbf" else record["selected_departure_caps"][method]
            cap_rows.append({**identity, "pair": pair, "method": method,
                             "common_scaler": record["common_scaler"], "departure_scaler": record["departure_scaler"],
                             "common_cap": record["common_cap"], "departure_cap": cap,
                             "selection_split": "B", "shared_baseline_sha256": record["shared_baseline_sha256"]})
            eta = calibration["choices"][pair + "__" + method]["selected_eta"]
            for horizon, value in zip(HORIZONS, eta):
                eta_rows.append({**identity, "pair": pair, "method": method, "horizon": horizon,
                                 "eta": float(value), "selection_split": "C"})
    for arm in ("standard", "minmax", "tanh_z_1"):
        common = diagnostics["common_rows"][arm]
        for horizon, value in zip(HORIZONS, common["bias"]):
            bias_rows.append({**identity, "common_scaler": arm, "common_cap": common["common_cap"],
                              "horizon": horizon, "shared_bias": float(value), "bias_fit_split": "B",
                              "frozen_common_state_sha256": common["common_frozen_state_sha256"]})
    return cap_rows, eta_rows, bias_rows


def selection_counts(rows):
    fields = ("law", "noise_level", "context", "method", "preprocessing_pair", "selected_cap")
    return [{**dict(zip(fields, key)), "count": len(values)} for key, values in grouped(rows, fields).items()]


def analyze(root=ROOT, output=None):
    root = Path(root).resolve()
    config = read_json(root / "config.json")
    validate_config(config)
    sys.path.insert(0, str(root / "code"))
    from pipeline import verify_release
    # This is the mandatory gate: no terminal array is opened before it passes.
    release_receipt = verify_release(root)
    freeze = read_json(root / "freeze.json")
    require(freeze.get("files", {}).get("analysis_code/analyze.py") == digest(Path(__file__)), "Analyzer is not part of the frozen source")
    abc = read_json(root / "prepared/abc_manifest.json")
    terminal = read_json(root / "prepared/d_manifest.json")
    tasks, d_tasks = manifest_roster(abc), manifest_roster(terminal)
    require(set(tasks) == set(d_tasks), "ABC and D task IDs differ")
    out = Path(output).resolve() if output else root / "analysis"
    out.mkdir(parents=True, exist_ok=False)
    horizon_path = out / "task_horizon.csv"
    task_rows, screens, selections, policies, gammas, source_rows, resources = [], [], [], [], [], [], []
    matched_rows, b_scores, secondary_caps, secondary_etas, secondary_biases = [], [], [], [], []
    all_identity_checks = 0
    horizon_count = 0
    with horizon_path.open("x", newline="") as stream:
        writer = None
        for index, task_id in enumerate(sorted(tasks), 1):
            meta, terminal_meta = tasks[task_id], d_tasks[task_id]
            require(all(meta[name] == terminal_meta[name] for name in META), "ABC/D task metadata differs")
            input_path = confined(root, meta["input"])
            target_path = confined(root, terminal_meta["target"])
            prediction_path = root / "predictions" / task_id / "predictions.npz"
            fit_path = root / "results" / task_id / "fit_summary.json"
            fit = read_json(fit_path)
            require(fit["status"] == "completed" and fit["task"]["task_id"] == task_id, "Incomplete/mismatched fit record")
            require(digest(input_path) == meta["sha256"], "ABC input hash differs")
            require(digest(target_path) == terminal_meta["target_sha256"], "Terminal target hash differs")
            with np.load(input_path, allow_pickle=False) as fitting:
                require(not any(name.startswith("D_") for name in fitting.files), "Fitting archive contains terminal data")
                ay = finite(fitting["A_Y"], (512, 3), "A targets")
                scale = np.maximum(np.var(ay, axis=0, ddof=0), 1e-12)
                close(fitting["loss_scale"], scale, "A-only loss normalization", exact=True)
            with np.load(target_path, allow_pickle=False) as target:
                outcomes = target["D_Y"]
            with np.load(prediction_path, allow_pickle=False) as forecast:
                predictions = {key: forecast[key] for key in forecast.files}
            checks = prediction_identities(predictions, fit["primary"])
            checks += secondary_identities(predictions, fit["secondary"])
            all_identity_checks += checks
            h_rows, t_rows, stability = score_task(meta, predictions, outcomes, scale)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(h_rows[0]))
                writer.writeheader()
            writer.writerows(h_rows)
            horizon_count += len(h_rows)
            task_rows.extend(t_rows)
            matched_rows.extend(matched_primary_rows(t_rows, fit["primary"]["selections"]["selected"]["kahm_departure"]))
            screens.extend(stability)
            selected, policy, gamma, scores = selection_tables(meta, fit["primary"])
            selections.extend(selected)
            policies.extend(policy)
            gammas.extend(gamma)
            b_scores.extend(scores)
            caps, etas, biases = secondary_selection_tables(meta, fit["secondary"])
            secondary_caps.extend(caps)
            secondary_etas.extend(etas)
            secondary_biases.extend(biases)
            source_rows.append({"task_id": task_id, "ABC_file_sha256": digest(input_path),
                                "target_file_sha256": digest(target_path), "target_array_sha256": array_hash(outcomes),
                                "prediction_file_sha256": digest(prediction_path), "forecast_arrays": len(predictions),
                                "prediction_identity_checks": checks})
            if "resources" in fit:
                # Preserve exact worker accounting without implying a speed comparison.
                resources.append({"task_id": task_id, "law": meta["law"], "repetition": meta["repetition"],
                                  "noise_level": meta["noise_level"], "resources_json": json.dumps(fit["resources"], sort_keys=True)})
            if index % 24 == 0:
                print(json.dumps({"analyzed_tasks": index, "expected_tasks": 192}), flush=True)
    require(len(task_rows) == 192 * 502 and horizon_count == 192 * 502 * 3, "Outcome record counts differ")
    reps = repetition_table(task_rows)
    matched_reps = repetition_table(matched_rows, view_count=10)
    vector = vector_lookup(reps + matched_reps)
    primary, primary_detail = primary_tables(vector)
    descriptive, descriptive_detail = primary_secondary_tables(vector)
    interactions, interaction_detail = interaction_tables(vector)
    fixed, fixed_detail, fixed_interactions, fixed_interaction_detail = fixed_common_tables(vector)
    matched_contrasts, matched_detail = matched_primary_contrasts(vector)
    panels = average_rows(reps, ("panel", *IDENTITY), spread=True)
    conditions = average_rows(task_rows, ("law", "noise_level", *IDENTITY), spread=True)
    tables = {
        "task_losses.csv": task_rows,
        "repetition_losses.csv": reps,
        "panel_summary.csv": panels,
        "context_summary.csv": [r for r in panels if r["family"] == "primary"],
        "condition_summary.csv": conditions,
        "primary_contrasts.csv": primary,
        "primary_contrast_repetitions.csv": primary_detail,
        "secondary_contrasts.csv": descriptive,
        "secondary_contrast_repetitions.csv": descriptive_detail,
        "preprocessing_interactions.csv": interactions,
        "preprocessing_interaction_repetitions.csv": interaction_detail,
        "fixed_common_panel_summary.csv": [r for r in panels if r["family"] == "secondary"],
        "fixed_common_contrasts.csv": fixed,
        "fixed_common_contrast_repetitions.csv": fixed_detail,
        "fixed_common_interactions.csv": fixed_interactions,
        "fixed_common_interaction_repetitions.csv": fixed_interaction_detail,
        "matched_primary_task_losses.csv": matched_rows,
        "matched_primary_repetition_losses.csv": matched_reps,
        "matched_primary_panel_summary.csv": average_rows(matched_reps, ("panel", *IDENTITY), spread=True),
        "matched_primary_contrasts.csv": matched_contrasts,
        "matched_primary_contrast_repetitions.csv": matched_detail,
        "stability_screens.csv": screens,
        "B_selections.csv": selections,
        "B_candidate_scores.csv": b_scores,
        "B_selection_counts.csv": selection_counts(selections),
        "C_policy_choices.csv": policies,
        "C_gamma_choices.csv": gammas,
        "fixed_common_B_selections.csv": secondary_caps,
        "fixed_common_C_etas.csv": secondary_etas,
        "fixed_common_shared_biases.csv": secondary_biases,
        "input_integrity.csv": source_rows,
    }
    if resources:
        require(len(resources) == 192, "Only some tasks have resource records")
        tables["task_resources.csv"] = resources
    table_records = {"task_horizon.csv": {"rows": horizon_count, "sha256": digest(horizon_path)}}
    for name, rows in tables.items():
        export_csv(out / name, rows)
        table_records[name] = {"rows": len(rows), "sha256": digest(out / name)}
    selected_screens = [row for row in screens if row["context"] == "selected"]
    established = sum(row["criterion_established"] for row in primary)
    summary = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(), "task_count": 192,
        "independent_repetitions": REPETITIONS, "forecast_arrays_per_task": 502,
        "primary_comparisons_established": established, "primary_family_size": FAMILY_SIZE,
        "all_four_primary_criteria_established": established == FAMILY_SIZE,
        "maximum_error_screens_passed": all(row["screen_pass"] for row in selected_screens),
        "maximum_selected_error_ratio": max(row["maximum_error_ratio"] for row in selected_screens),
        "failed_stability_tasks": [row["task_id"] for row in selected_screens if not row["screen_pass"]],
        "primary_contrasts": primary,
        "primary_interpretation": "Each of the four frozen claims is assessed separately with a simultaneous one-sided paired-t bound. The tanh-specific claim and the selected-procedure adoption comparison are distinct. Failure is not equivalence.",
        "inference_scope": "Approximate Bonferroni one-sided paired-t family at alpha 0.05 over 24 independent repetition aggregates; not distribution-free risk certificates.",
        "secondary_scope": "All secondary intervals are descriptive unadjusted 95% paired-t intervals. Fixed-common results have a separate P/R sample allocation and are not pooled into the primary ranking.",
        "generalization_scope": "Fresh repetitions of four previously studied numerical observation laws; no natural-data application cohort or general deployment conclusion.",
        "scoring_scope": "Only frozen forecast arrays are evaluated. Neither gamma-one arms nor candidate diagnostics are reselected using D outcomes.",
        "integrity": {"release_receipt": json_safe(release_receipt), "verified_before_terminal_target_access": True,
                      "prediction_identity_checks": all_identity_checks, "config_sha256": digest(root / "config.json"),
                      "freeze_sha256": digest(root / "freeze.json"), "ABC_manifest_sha256": digest(root / "prepared/abc_manifest.json"),
                      "D_manifest_sha256": digest(root / "prepared/d_manifest.json")},
        "tables": table_records,
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
