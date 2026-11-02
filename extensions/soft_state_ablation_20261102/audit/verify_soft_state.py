#!/usr/bin/env python3
"""Independently verify the complete frozen soft-state ablation.

Replays every retained B penalty path with a Cholesky affine-ridge solution,
every selected C/D forecast, source/seed/stage hashes, and the full statistical
family. No production model, fitting function, analyzer, or law generator is
imported. Saved base trajectories are used to rederive every history/target.
"""
from __future__ import annotations
import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from independent_oracle import (ARMS, VIEWS, BLOCKS, CAPS, REPLAY_RTOL, REPLAY_ATOL,
    Auditor, array_hash, file_hash, read_portable, validate_bank, raw_blocks,
    transformed_features, check_diagnostics, fit_ridge_path)

ROOT = Path(__file__).resolve().parents[1]
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
KEYS = tuple(f"{view}__{arm}" for view in VIEWS for arm in ARMS)
HORIZONS = (1, 4, 12)
PRIMARY = (("P_vs_G_nonlinear", "nonlinear", "P", "G", .97),
           ("K_vs_P_nonlinear", "nonlinear", "K", "P", .97),
           ("K_vs_R_nonlinear", "nonlinear", "K", "R", .97),
           ("K_vs_P_linear", "linear", "K", "P", 1.05))


def read(path):
    return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def verified_files(root, mapping, audit):
    root = Path(root).resolve()
    audit.check(isinstance(mapping, dict) and bool(mapping), "Nonempty hash manifest")
    for relative, digest in mapping.items():
        rel = Path(relative)
        path = (root / rel).resolve()
        audit.check(not rel.is_absolute() and ".." not in rel.parts and path.is_relative_to(root),
                    "Confined artifact path")
        audit.check(path.is_file() and file_hash(path) == digest, "Sealed bytes: " + relative)


def time_of(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def mse(prediction, target):
    residual = prediction - target
    return np.einsum("ij,ij->j", residual, residual) / len(target)


def verify_seeds(config, root, audit):
    actual = read(root / "prepared/seed_manifest.json")["records"]
    expected = []
    master, purpose = config["master_seed"], config["seed_purposes"]
    for rep in range(config["repetitions"]):
        expected.append({"role": "identifier", "repetition": rep,
                         "components": [master, rep, purpose["identifier"]]})
        for li, law in enumerate(config["laws"]):
            expected.append({"role": "fit", "repetition": rep, "law": law,
                             "components": [master, rep, li, purpose["fit"]]})
            for split, si in config["split_indices"].items():
                for name in ("generation", "unit_noise", "rows"):
                    expected.append({"role": name, "repetition": rep, "law": law, "split": split,
                                     "components": [master, rep, li, si, purpose[name]]})
    for row in expected:
        row["seed"] = int(np.random.SeedSequence(row["components"]).generate_state(1)[0])
    audit.check(actual == expected, "Independent complete seed derivation")
    seeds = {row["seed"] for row in actual}
    audit.check(len(seeds) == len(actual) == 1696, "Distinct 32-repetition seed roster")
    prior = read(root / "readiness/prior_seed_exclusions.json")["records"]
    audit.check(not seeds.intersection(int(row["seed"]) for row in prior), "No prior-cohort seed collision")
    return expected


def verify_windows(config, initial, terminal, root, audit):
    """Reconstruct all windows from archived physical episodes, not law reruns."""
    prepared = npz(root / initial["input"])
    query, target = npz(root / terminal["query"]), npz(root / terminal["target"])
    audit.check(not any(key.endswith("_Y") for key in query), "D query archive contains no target array")
    audit.check("D_X" not in target, "D target archive does not duplicate query array")
    for key, digest in initial["array_sha256"].items():
        audit.check(key in prepared and array_hash(prepared[key]) == digest, "ABC array identity")
    audit.check(array_hash(query["D_X"]) == terminal["D_X_sha256"], "D history array identity")
    audit.check(array_hash(target["D_Y"]) == terminal["D_Y_sha256"], "D target array identity")
    bases = {**npz(root / initial["base_file"]), **npz(root / terminal["base_file"])}
    amplitude = float(np.std(bases["A_base_episodes"], ddof=0))
    audit.near(amplitude, initial["A_base_episode_std"], "A-only observation noise amplitude", exact=True)
    audit.near(amplitude, terminal["A_base_episode_std"], "D reuses A noise amplitude", exact=True)
    width, stride = config["histories"][0], config["stride"]
    offsets = np.asarray(config["horizons"])
    for split in ("A", "B", "C", "D"):
        meta = (terminal if split == "D" else initial)["splits"][split]
        base, unit = bases[split + "_base_episodes"], bases[split + "_unit_noise"]
        audit.check(base.shape == (config["split_episodes"][split], config["max_episode_length"]),
                    "Base episode dimensions")
        audit.check(array_hash(base) == meta["base_sha256"] and array_hash(unit) == meta["unit_noise_sha256"],
                    "Episode/noise array hashes")
        # Reconstruct Gaussian noise independently from its own reserved stream.
        expected_unit = np.random.default_rng(meta["streams"]["unit_noise"]["seed"]).standard_normal(base.shape)
        audit.near(unit, expected_unit, "Unit-noise reserved stream", exact=True)
        noisy = base + float(initial["noise_level"]) * amplitude * unit
        origins = list(range(width - 1, base.shape[1] - int(offsets.max()), stride))
        audit.check(len(origins) == config["origins_per_episode"], "Origin count within episode")
        samples = [(episode, origin) for episode in range(len(base)) for origin in origins]
        count, cap = len(samples), config["split_caps"][split]
        chosen = (np.arange(count, dtype=np.int64) if cap is None or cap >= count else
                  np.sort(np.random.default_rng(meta["streams"]["rows"]["seed"]).choice(count, cap, replace=False)))
        audit.check(array_hash(chosen) == meta["retained_indices_sha256"], "Independent retained-row selection")
        pairs = [samples[int(index)] for index in chosen]
        expected_x = np.asarray([noisy[episode, origin - width + 1:origin + 1] for episode, origin in pairs])
        expected_y = np.asarray([noisy[episode, origin + offsets] for episode, origin in pairs])
        observed = {**query, **target} if split == "D" else prepared
        audit.near(observed[split + "_X"], expected_x, "Independent " + split + " histories", exact=True)
        audit.near(observed[split + "_Y"], expected_y, "Independent " + split + " targets", exact=True)
        audit.near(observed[split + "_groups"], [p[0] for p in pairs], "Independent episode identities", exact=True)
        audit.near(observed[split + "_origins"], [p[1] for p in pairs], "Independent origin identities", exact=True)
        audit.near(observed[split + "_row_indices"], chosen, "Independent retained row identities", exact=True)
    return prepared, query["D_X"], target["D_Y"]


def audit_task(root, task_id, audit, *, data=None):
    """Replay one already generated task; usable on explicitly artificial fixtures."""
    root = Path(root)
    config = read(root / "config.json")
    directory = root / "results" / task_id
    summary = read(directory / "fit_summary.json")
    task = summary["task"]
    if data is None:
        initial = next(row for row in read(root / "prepared/abc_manifest.json")["tasks"] if row["task_id"] == task_id)
        terminal = next(row for row in read(root / "prepared/d_manifest.json")["tasks"] if row["task_id"] == task_id)
        data = verify_windows(config, initial, terminal, root, audit)
    abc, dx, dy = data
    state, fingerprint = read_portable(directory, audit)
    prec = read(directory / "pre_c_selection.json")
    cdiag = read(directory / "C_diagnostics.json")
    ddiag = read(root / "predictions" / task_id / "query_diagnostics.json")
    receipt = read(root / "predictions" / task_id / "prediction_receipt.json")
    c_arrays, d_arrays = npz(directory / "diagnostic_predictions.npz"), npz(root / "predictions" / task_id / "predictions.npz")
    validation = npz(directory / "validation_predictions.npz")
    audit.check(set(c_arrays) == set(d_arrays) == set(KEYS), "Complete eight-array C/D roster")
    audit.check(state["cap_order"] == config["native_cap_order"], "Complete state cap order")
    audit.check(state["schema"] == "kahm-soft-state-model-v1" and state["arms"] == list(ARMS), "Model schema")
    audit.check(state["retained_baseline"] is False and state["autonomous_dynamics_fitted"] is False,
                "Representation-only direct readouts")
    audit.check(state["C_role"] == "diagnostic only" and state["geometry_scaling"] == "standard", "State scope")
    audit.check(prec["C_loaded"] is False and prec["selection_complete"] is True, "Pre-C selection receipt")
    audit.check(cdiag["selection_performed"] is False and cdiag["state_unchanged"] is True and
                receipt["targets_loaded"] is False and receipt["state_unchanged"] is True, "No C/D refitting")
    for record in (summary, prec, cdiag, receipt):
        audit.check(record["state_fingerprint"] == fingerprint, "Fitted state invariant across stages")
    audit.check(summary["state_fingerprint_before_C"] == fingerprint, "A/B state fingerprint")
    audit.check(prec["selections"] == summary["selections"] == state["selections"], "Selections unchanged after C")
    audit.check(set(state["input_hashes"]) == {"A_X", "A_Y", "B_X", "B_Y"}, "Fitting target/input split roster")
    for key, expected in state["input_hashes"].items():
        audit.check(array_hash(abc[key]) == expected, "Fitted A/B array identity")
    scale = np.maximum(np.var(abc["A_Y"], axis=0, ddof=0), 1e-12)
    audit.near(state["loss_scale"], scale, "A-only loss normalization", exact=True)
    alpha_grid = config["model"]["alpha_grid"]
    cap_order = config["native_cap_order"]
    expected_b = {f"cap{cap}__{arm}{suffix}" for cap in cap_order for arm in ARMS for suffix in ("", "__alpha_paths")}
    audit.check(set(validation) == expected_b | set(KEYS), "Complete B penalty-path roster")
    selected_ids = {choice["candidate_id"] for choices in state["selections"].values() for choice in choices.values()}
    audit.check(set(cdiag["predictions"]["candidates"]) == set(ddiag["candidates"]) == selected_ids,
                "Selected query geometry roster")
    audit.check(cdiag["predictions"]["targets_accepted"] is False and ddiag["targets_accepted"] is False,
                "Query interface excludes targets")
    scores, dimensions, widths, fallbacks = {}, [], [], []
    first_bank, first_gates = None, None
    for cap in cap_order:
        cid = f"cap{cap}"
        local_state, record = state["candidates"][cid], summary["candidates"][cid]
        audit.check(record["C_used"] is False and record["D_used"] is False, "Candidate fitted only on A/B")
        audit.check(record["seed"] == task["fit_seed"], "Shared cap fitting seed")
        bank = local_state["bank"]
        audit.check(record["bank"]["requested_blocks"] == config["model"]["blocks"], "Declared partition counts")
        dims = validate_bank(bank, record, abc["A_X"], cap, audit)
        dimensions.extend(dims)
        if first_bank is None:
            first_bank = bank
        else:
            for key in ("mean", "scale"):
                audit.near(bank[key], first_bank[key], "Exact common input scaling across caps", exact=True)
            for a, b in zip(bank["models"], first_bank["models"]):
                for key in ("q", "center", "radius"):
                    audit.near(a[key], b[key], "Exact references/centers/radii across caps", exact=True)
        features, raws = {}, {}
        for stage in ("A", "B"):
            raws[stage], raw_diagnostic = raw_blocks(bank, abc[stage + "_X"], audit)
            features[stage] = transformed_features(local_state, raws[stage], audit)
            check_diagnostics(record["features_A_B"][stage], raw_diagnostic, features[stage], audit)
            for row in raw_diagnostic["local_models"]:
                fallbacks.append({"cap": cap, "stage": stage, "model_index": row["model"],
                                  "kind": row["kind"], "rank_zero": row["rank_zero"],
                                  "fallback_count": len(row["fallback_rows"])})
        if first_gates is None:
            first_gates = raws["A"]["gates"]
        else:
            audit.near(raws["A"]["gates"], first_gates, "Exact raw memberships across caps", exact=True)
        for block in BLOCKS:
            raw = raws["A"][block]
            scaler = local_state["scalers"][block]
            audit.near(scaler["mean"], raw.mean(axis=0), "A-only block means")
            audit.near(scaler["scale"], np.maximum(raw.std(axis=0), config["model"]["feature_scale_floor"]),
                       "A-only block scales")
            audit.check(record["block_widths"][block] == raw.shape[1], "Raw block width")
        coefficients = {}
        scores[cid] = {}
        for arm in ARMS:
            candidate_record = record["arms"][arm]
            path, pb, fm, ym = fit_ridge_path(features["A"][arm], abc["A_Y"], features["B"][arm], alpha_grid)
            saved = validation[f"{cid}__{arm}__alpha_paths"]
            audit.near(saved, pb, "Independent all-alpha B forecasts")
            losses = np.stack([mse(prediction, abc["B_Y"]) for prediction in saved])
            indices = np.argmin(losses, axis=0)
            audit.near(candidate_record["alpha_indices_by_horizon"], indices, "First B penalty minimum", exact=True)
            audit.near(candidate_record["alpha_by_horizon"], np.asarray(alpha_grid)[indices], "B-selected penalties", exact=True)
            audit.check(len(candidate_record["alpha_candidates"]) == len(alpha_grid), "All B penalties retained")
            for ai, row in enumerate(candidate_record["alpha_candidates"]):
                audit.check(row["alpha"] == alpha_grid[ai] and row["alpha_index"] == ai and row["status"] == "valid",
                            "Penalty order and validity")
                audit.near(row["B_mse_by_horizon"], losses[ai], "Recomputed B penalty losses", rtol=3e-12, atol=3e-13)
                audit.near(row["B_smse_by_horizon"], losses[ai] / scale, "Recomputed B standardized losses", rtol=3e-12, atol=3e-13)
            chosen = np.column_stack([saved[int(index), :, h] for h, index in enumerate(indices)])
            audit.near(validation[f"{cid}__{arm}"], chosen, "Selected B penalty forwarding", rtol=1e-12, atol=1e-12)
            audit.check(array_hash(validation[f"{cid}__{arm}"]) == candidate_record["B_prediction_sha256"], "Selected B array hash")
            score = float(np.mean(mse(validation[f"{cid}__{arm}"], abc["B_Y"]) / scale))
            audit.near(candidate_record["B_mean_smse"], score, "Candidate cap score", rtol=3e-12, atol=3e-13)
            scores[cid][arm] = score
            head = local_state["heads"][arm]
            coefficient = np.column_stack([path[int(index), :, h] for h, index in enumerate(indices)])
            coefficients[arm] = (coefficient, fm, ym)
            audit.near(head["coef"], coefficient, "Independent selected ridge coefficients")
            audit.near(head["mean"], fm, "Selected head feature centering")
            audit.near(head["ymean"], ym, "Unpenalized response intercept", exact=True)
            audit.near(head["alpha_by_horizon"], np.asarray(alpha_grid)[indices], "Stored selected penalties", exact=True)
            audit.check(head["feature_width"] == record["feature_widths"][arm] == features["A"][arm].shape[1],
                        "Feature dimension and readout identity")
            audit.check(head["training_rows"] == len(abc["A_Y"]), "All A rows used for readout")
            widths.append({"cap": cap, "arm": arm, "feature_width": head["feature_width"],
                           "centered_feature_rank": head["feature_rank"],
                           "independently_recomputed_local_rank_sum": sum(d["effective_rank"] for d in dims)})
        if cid in selected_ids:
            for stage, x, arrays, diagnostic in (("C", abc["C_X"], c_arrays, cdiag["predictions"]["candidates"][cid]),
                                                 ("D", dx, d_arrays, ddiag["candidates"][cid])):
                raw, raw_diagnostic = raw_blocks(bank, x, audit)
                feature = transformed_features(local_state, raw, audit)
                check_diagnostics(diagnostic, raw_diagnostic, feature, audit)
                for row in raw_diagnostic["local_models"]:
                    fallbacks.append({"cap": cap, "stage": stage, "model_index": row["model"],
                                      "kind": row["kind"], "rank_zero": row["rank_zero"],
                                      "fallback_count": len(row["fallback_rows"])})
                for view in VIEWS:
                    for arm in ARMS:
                        if state["selections"][view][arm]["candidate_id"] == cid:
                            coefficient, fm, ym = coefficients[arm]
                            prediction = ym + (feature[arm] - fm) @ coefficient
                            audit.near(arrays[f"{view}__{arm}"], prediction, "Independent selected " + stage + " forecasts")
    for view in VIEWS:
        for arm in ARMS:
            selector = "P" if view == "shared" else arm
            ci = int(np.argmin([scores[f"cap{cap}"][selector] for cap in cap_order]))
            expected_cap, choice = cap_order[ci], state["selections"][view][arm]
            audit.check(choice["cap"] == expected_cap and choice["candidate_id"] == f"cap{expected_cap}", "B-only cap selection")
            audit.check(choice["cap_selection_arm"] == selector and choice["cap_selection_split"] == "B", "Cap selector scope")
            audit.near(validation[f"{view}__{arm}"], validation[f"cap{expected_cap}__{arm}"], "Exact selected B view", exact=True)
    audit.near(c_arrays["independent__P"], c_arrays["shared__P"], "P view identity C", exact=True)
    audit.near(d_arrays["independent__P"], d_arrays["shared__P"], "P view identity D", exact=True)
    horizon_losses, task_losses = [], []
    for key in KEYS:
        c_mse = mse(c_arrays[key], abc["C_Y"])
        audit.near(cdiag["losses"][key]["mse_by_horizon"], c_mse, "C diagnostic losses", rtol=3e-12, atol=3e-13)
        audit.near(cdiag["losses"][key]["smse_by_horizon"], c_mse / scale, "C diagnostic standardized losses", rtol=3e-12, atol=3e-13)
        d_mse = mse(d_arrays[key], dy)
        d_smse = d_mse / scale
        task_losses.append({"task_id": task_id, "repetition": task["repetition"], "law": task["law"],
            "noise_index": task["noise_index"], "forecast_key": key,
            "mse": float(math.fsum(d_mse) / len(d_mse)), "smse": float(math.fsum(d_smse) / len(d_smse))})
        for h, mv, sv in zip(config["horizons"], d_mse, d_smse):
            horizon_losses.append({"task_id": task_id, "forecast_key": key, "horizon": h,
                                   "mse": float(mv), "smse": float(sv)})
    events = read(directory / "numerical_solver_events.json")
    audit.check(events["fallback_count"] == len(events["events"]) and events["failed_count"] == 0,
                "No silently failed numerical fit")
    audit.check(all(row["status"] == "recovered" for row in events["events"]), "SVD fallback recovery status")
    return {"task_id": task_id, "dimensions": dimensions, "widths": widths, "fallbacks": fallbacks,
            "selections": state["selections"],
            "numerical_solver_fallbacks": events["fallback_count"],
            "task_losses": task_losses, "horizon_losses": horizon_losses}


def statistics_check(records, analysis, audit):
    """Separate scalar aggregation; never call or import the scientific analyzer."""
    rows = [row for record in records for row in record["task_losses"]]
    index = {(r["repetition"], r["law"], r["noise_index"], r["forecast_key"]): r["smse"] for r in rows}
    audit.check(len(index) == len(rows) == 32 * 4 * 2 * 8, "All independent task loss units")
    vectors = {}
    for panel, laws in (("linear", LAWS[:1]), ("nonlinear", LAWS[1:]), ("all_descriptive", LAWS)):
        for key in KEYS:
            vectors[panel, key] = [math.fsum(index[rep, law, noise, key] for law in laws for noise in (0, 1)) / (2 * len(laws))
                                   for rep in range(32)]
    primary = []
    critical = float(t.isf(.05 / 4, 31))
    for cid, panel, candidate, comparator, factor in PRIMARY:
        left, right = vectors[panel, "shared__" + candidate], vectors[panel, "shared__" + comparator]
        delta = [a - factor * b for a, b in zip(left, right)]
        mean = math.fsum(delta) / 32
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in delta) / 31)
        se = sd / math.sqrt(32)
        upper = mean + critical * se
        pvalue = float(t.cdf(mean / se, 31)) if se else (0. if mean < 0 else 1. if mean > 0 else .5)
        primary.append({"contrast_id": cid, "panel": panel, "candidate": candidate, "comparator": comparator,
            "comparator_factor": factor, "candidate_mean_smse": math.fsum(left) / 32,
            "comparator_mean_smse": math.fsum(right) / 32,
            "point_relative_gain_percent": 100 * (1 - math.fsum(left) / math.fsum(right)),
            "margin_difference_mean": mean,
            "margin_difference_sd": sd, "margin_difference_se": se, "critical_t": critical,
            "simultaneous_one_sided_upper": upper, "criterion_established": upper < 0,
            "p_one_sided": pvalue, "p_bonferroni": min(1., 4 * pvalue)})
    analysis = Path(analysis)
    report = read(analysis / "analysis_summary.json")
    verified_files(analysis, {name: row["sha256"] for name, row in report["tables"].items()}, audit)
    audit.check(report["independent_repetitions"] == 32 and report["tasks"] == 256 and
                report["primary_family_size"] == 4, "Analysis population/family units")
    for filename, expected, keys in (
            ("task_losses.csv", rows, ("task_id", "forecast_key")),
            ("horizon_losses.csv", [r for record in records for r in record["horizon_losses"]], ("task_id", "forecast_key", "horizon"))):
        with (analysis / filename).open(newline="") as stream:
            actual = list(csv.DictReader(stream))
        lookup = {tuple(str(row[k]) for k in keys): row for row in actual}
        audit.check(len(actual) == len(expected) == len(lookup), "Complete " + filename)
        for row in expected:
            other = lookup[tuple(str(row[k]) for k in keys)]
            for value in ("mse", "smse"):
                audit.near(float(other[value]), row[value], "Independent exported " + value, rtol=3e-11, atol=3e-13)
    with (analysis / "repetition_losses.csv").open(newline="") as stream:
        repeated = list(csv.DictReader(stream))
    audit.check(len(repeated) == 3 * 8 * 32, "Complete repetition table")
    for row in repeated:
        audit.near(float(row["smse"]), vectors[row["panel"], row["forecast_key"]][int(row["repetition"])],
                   "Independent repetition aggregate", rtol=3e-11, atol=3e-13)
    with (analysis / "primary_contrasts.csv").open(newline="") as stream:
        published = list(csv.DictReader(stream))
    audit.check([row["contrast_id"] for row in published] == [row["contrast_id"] for row in primary], "Complete primary contrast roster")
    for actual, expected in zip(published, primary):
        for key, value in expected.items():
            if isinstance(value, bool):
                audit.check(actual[key] == str(value), "Primary strict decision")
            elif isinstance(value, (int, float)):
                audit.near(float(actual[key]), value, "Independent primary " + key, rtol=3e-10, atol=3e-13)
            else:
                audit.check(actual[key] == value, "Primary label identity")
        audit.check(int(actual["n_repetitions"]) == 32 and int(actual["df"]) == 31 and
                    int(actual["family_size"]) == 4 and float(actual["family_alpha"]) == .05 and actual["view"] == "shared",
                    "Primary sampling/family allocation")
    audit.check(report["criteria_established"] == sum(row["criterion_established"] for row in primary), "Summary primary decisions")
    audit.check(len(report["primary_contrasts"]) == 4, "JSON complete primary family")
    for published_row, expected in zip(report["primary_contrasts"], primary):
        for key, value in expected.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                audit.near(published_row[key], value, "Independent JSON primary " + key, rtol=3e-10, atol=3e-13)
            else:
                audit.check(published_row[key] == value, "JSON primary label/decision identity")
    with (analysis / "secondary_descriptive.csv").open(newline="") as stream:
        secondary = list(csv.DictReader(stream))
    expected_pairs = {(panel, f"{view}__{a}", f"{view}__{b}")
                      for panel in ("linear", "nonlinear") for view in VIEWS
                      for a, b in (("P", "G"), ("K", "P"), ("K", "R"), ("R", "P"))}
    expected_pairs |= {(panel, f"independent__{arm}", f"shared__{arm}")
                       for panel in ("linear", "nonlinear") for arm in ARMS}
    audit.check(len(secondary) == 24 and {(r["panel"], r["candidate"], r["comparator"]) for r in secondary} == expected_pairs,
                "Complete prespecified secondary comparison roster")
    critical95 = float(t.isf(.025, 31))
    for row in secondary:
        left, right = vectors[row["panel"], row["candidate"]], vectors[row["panel"], row["comparator"]]
        difference = [a - b for a, b in zip(left, right)]
        mean = math.fsum(difference) / 32
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in difference) / 31)
        se = sd / math.sqrt(32)
        for key, value in (("difference_mean", mean), ("difference_sd", sd), ("difference_se", se),
                           ("descriptive_95_lower", mean - critical95 * se),
                           ("descriptive_95_upper", mean + critical95 * se)):
            audit.near(float(row[key]), value, "Independent secondary " + key, rtol=3e-10, atol=3e-13)
        audit.check("unadjusted" in row["scope"] and int(row["n_repetitions"]) == 32,
                    "Secondary intervals remain descriptive repetition intervals")
    for filename, record_name, row_keys, value_keys in (
            ("basis_dimensions.csv", "dimensions", ("task_id", "cap", "model_index"),
             ("reference_count", "initial_admissible_rank", "effective_rank", "range_reduction_steps")),
            ("feature_widths.csv", "widths", ("task_id", "cap", "arm"),
             ("feature_width", "centered_feature_rank"))):
        expected_rows = [{"task_id": item["task_id"], **row} for item in records for row in item[record_name]]
        with (analysis / filename).open(newline="") as stream:
            actual_rows = list(csv.DictReader(stream))
        lookup = {tuple(row[k] for k in row_keys): row for row in actual_rows}
        audit.check(len(actual_rows) == len(expected_rows) == len(lookup), "Complete dimension table " + filename)
        for expected_row in expected_rows:
            actual = lookup[tuple(str(expected_row[k]) for k in row_keys)]
            for key in value_keys:
                audit.check(int(actual[key]) == expected_row[key], "Verified dimension export " + key)
    with (analysis / "selection_by_task.csv").open(newline="") as stream:
        selected = list(csv.DictReader(stream))
    lookup = {(row["task_id"], row["view"], row["arm"]): row for row in selected}
    audit.check(len(selected) == len(lookup) == 256 * 8, "Complete selection table")
    for item in records:
        for view in VIEWS:
            for arm in ARMS:
                actual = lookup[item["task_id"], view, arm]
                expected = item["selections"][view][arm]
                for key in ("cap", "feature_width", "centered_feature_rank"):
                    audit.check(int(actual[key]) == expected[key], "Verified selected dimensions")
                for h, alpha in zip(HORIZONS, expected["alpha_by_horizon"]):
                    audit.check(float(actual[f"alpha_h{h}"]) == alpha, "Verified selected penalty export")
    return primary


def verify_study(root, analysis, output):
    root, analysis, output = Path(root).resolve(), Path(analysis).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    audit = Auditor()
    config, freeze = read(root / "config.json"), read(root / "freeze.json")
    audit.check(config["master_seed"] == 2026110203 and config["repetitions"] == 32 and config["task_count"] == 256,
                "Frozen independent production cohort")
    audit.check(not config.get("manufactured_fixture", False), "Artificial fixture is not scientific evidence")
    audit.check(tuple(config["laws"]) == LAWS and tuple(config["native_cap_order"]) == CAPS and
                tuple(config["horizons"]) == HORIZONS, "Fixed law/cap/horizon roster")
    audit.check(config["model"]["blocks"] == [2, 4, 8], "Production partition dimensions")
    audit.check(config["analysis"]["primary_claims"] == [
        {"candidate": a, "population": panel, "reference": b, "reference_factor": factor}
        for _, panel, a, b, factor in PRIMARY], "Frozen primary claim definitions")
    audit.check(config["analysis"]["primary_context"] == "shared" and
                config["analysis"]["primary_family_size"] == 4 and config["analysis"]["family_alpha"] == .05 and
                config["analysis"]["independent_unit"] == "repetition", "Primary allocation and unit")
    audit.check(freeze["status"] == "frozen" and freeze["data_generated"] is False and freeze["D_generated"] is False,
                "Prespecified source freeze")
    audit.check(freeze["config_sha256"] == file_hash(root / "config.json"), "Frozen config identity")
    verified_files(root, freeze["files"], audit)
    for name in ("verify_soft_state.py", "independent_oracle.py"):
        audit.check(freeze["files"].get("audit/" + name) == file_hash(Path(__file__).with_name(name)),
                    "Executing independent audit source frozen")
    verified_files(root.parent / "dimension_forecasting", freeze["inherited_files"], audit)
    verified_files(root.parent / "policy_augmentation_20261101", freeze["numerical_solver_files"], audit)
    seeds = verify_seeds(config, root, audit)
    abc, terminal = read(root / "prepared/abc_manifest.json"), read(root / "prepared/d_manifest.json")
    fits, predictions, release = read(root / "FIT_LOCK.json"), read(root / "PREDICTION_LOCK.json"), read(root / "TARGET_RELEASE.json")
    order = [freeze, abc, fits, terminal, predictions, release]
    audit.check(all(time_of(a["created_utc"]) <= time_of(b["created_utc"]) for a, b in zip(order, order[1:])),
                "Freeze/ABC/fit-seal/D/prediction-seal/release chronology")
    publication = read(root / "PUBLICATION.json")
    audit.check(publication["status"] == "published" and publication["source_files_verified"] is True and
                publication["freeze_sha256"] == file_hash(root / "freeze.json"), "Recorded publication binds source freeze")
    audit.check(publication["repository"] == config["source"]["repository"], "Published repository identity")
    for key in ("source_commit", "source_tree_sha"):
        audit.check(len(publication[key]) == 40 and all(c in "0123456789abcdef" for c in publication[key]),
                    "Publication Git identity")
    audit.check(time_of(freeze["created_utc"]) <= time_of(publication["published_utc"]) < time_of(abc["created_utc"]),
                "Source publication precedes all scientific data")
    for manifest in (abc, terminal):
        audit.check(manifest["publication_sha256"] == file_hash(root / "PUBLICATION.json"), "Prepared publication hash link")
    audit.check(terminal["fit_lock_sha256"] == file_hash(root / "FIT_LOCK.json") and
                terminal["ABC_manifest_sha256"] == file_hash(root / "prepared/abc_manifest.json"), "D generation seal links")
    for field, filename in (("fit_lock_sha256", "FIT_LOCK.json"), ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                            ("d_manifest_sha256", "prepared/d_manifest.json")):
        audit.check(release[field] == file_hash(root / filename), "Release artifact chain")
    audit.check(release["status"] == "released", "Scoring released after prediction seal")
    aindex, dindex = {r["task_id"]: r for r in abc["tasks"]}, {r["task_id"]: r for r in terminal["tasks"]}
    audit.check(len(abc["tasks"]) == len(terminal["tasks"]) == len(aindex) == len(dindex) == 256 and
                set(aindex) == set(dindex), "Complete prepared task identities")
    factorial = {(r["repetition"], r["law"], r["noise_index"]) for r in terminal["tasks"]}
    audit.check(factorial == {(r, law, noise) for r in range(32) for law in LAWS for noise in (0, 1)},
                "Exact evaluation population")
    for document in (abc, terminal, fits, predictions):
        audit.check(document["config_sha256"] == file_hash(root / "config.json") and
                    document["freeze_sha256"] == file_hash(root / "freeze.json"), "Stage config/source hash links")
    for document in (abc, terminal):
        verified_files(root, document["base_files"], audit)
        audit.check(document["seed_manifest_sha256"] == file_hash(root / "prepared/seed_manifest.json"), "Prepared seed roster link")
    for document, phase in ((fits, "fits"), (predictions, "predictions")):
        audit.check(document["status"] == "sealed" and document["phase"] == phase and document["task_count"] == 256,
                    "Complete stage seal")
        audit.check(len(document["task_ids"]) == 256 and set(document["task_ids"]) == set(aindex), "Sealed task population")
        manifest = "abc_manifest.json" if phase == "fits" else "d_manifest.json"
        audit.check(document["manifest_sha256"] == file_hash(root / "prepared" / manifest), "Seal binds exact prepared manifest")
        verified_files(root, document["files"], audit)
    audit.check(not list((root / "results").rglob("failure.json")) and
                not list((root / "predictions").rglob("failure.json")), "No failed or silently omitted task")
    fit_seeds = {(s["repetition"], s["law"]): s["seed"] for s in seeds if s["role"] == "fit"}
    identifiers = {s["repetition"]: s["seed"] for s in seeds if s["role"] == "identifier"}
    records = []
    with threadpool_limits(limits=1):
        for number, tid in enumerate(sorted(aindex), 1):
            initial, final = aindex[tid], dindex[tid]
            for key in ("law", "law_index", "history", "seed", "repetition", "noise_index", "noise_level", "fit_seed"):
                audit.check(initial[key] == final[key], "Task metadata fixed across stages")
            audit.check(initial["fit_seed"] == fit_seeds[initial["repetition"], initial["law"]] and
                        initial["seed"] == identifiers[initial["repetition"]], "Task reserved fit/repetition streams")
            audit.check(initial["noise_level"] == config["noise_levels"][initial["noise_index"]] and
                        initial["law_index"] == config["laws"].index(initial["law"]) and initial["history"] == 12,
                        "Declared task condition labels")
            for split in ("A", "B", "C", "D"):
                expected_streams = {s["role"]: s for s in seeds if s.get("repetition") == initial["repetition"]
                                    and s.get("law") == initial["law"] and s.get("split") == split}
                actual_streams = (final if split == "D" else initial)["splits"][split]["streams"]
                audit.check(actual_streams == expected_streams, "Prepared split uses its declared reserved streams")
            verified_files(root, {initial["input"]: initial["sha256"], final["query"]: final["query_sha256"],
                                  final["target"]: final["target_sha256"]}, audit)
            data = verify_windows(config, initial, final, root, audit)
            records.append(audit_task(root, tid, audit, data=data))
            if number % 8 == 0 or number == 256:
                print(json.dumps({"audit_tasks_completed": number, "total": 256, "checks": audit.checks}), flush=True)
    primary = statistics_check(records, analysis, audit)
    report = {"schema": "independent-soft-state-audit-v1", "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "tasks": 256, "independent_repetitions": 32,
        "D_forecast_arrays_replayed": 2048, "C_forecast_arrays_replayed": 2048,
        "B_penalty_path_arrays_replayed": 256 * 5 * 4 * 5,
        "checks": audit.checks, "primary_contrasts": primary,
        "criteria_established": sum(r["criterion_established"] for r in primary),
        "replay_rtol": REPLAY_RTOL, "replay_atol": REPLAY_ATOL,
        "maximum_absolute_numerical_differences": dict(audit.max_absolute_difference),
        "source_hashes": {"verify_soft_state.py": file_hash(Path(__file__)),
                          "independent_oracle.py": file_hash(Path(__file__).with_name("independent_oracle.py"))},
        "config_sha256": file_hash(root / "config.json"), "freeze_sha256": file_hash(root / "freeze.json"),
        "analysis_summary_sha256": file_hash(analysis / "analysis_summary.json"),
        "numerical_solver_recoveries": sum(r["numerical_solver_fallbacks"] for r in records),
        "tasks_detail": [{k: v for k, v in r.items() if k not in ("task_losses", "horizon_losses")} for r in records],
        "scope": ["All task histories/targets replayed from retained physical episodes and reserved noise/row streams.",
                  "All B penalty paths independently refitted by affine Cholesky normal equations.",
                  "Every selected C/D forecast independently replayed from portable A-fitted geometry and coefficients.",
                  "All primary aggregate losses and bounds independently recomputed; units are complete repetitions.",
                  "Local reference eligibility, native spectral/range dimensions and kernel identities checked.",
                  "Partition k-means fitting and physical law integration are not independently reimplemented.",
                  "Stage hashes and source inspection establish recorded workflow order, not cryptographic target isolation.",
                  "Replay agreement is numerical within frozen tolerances; sealed byte and prefix hashes are exact."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-task", help="Replay one explicitly artificial integration task only")
    args = parser.parse_args()
    if args.fixture_task:
        if not read(args.root / "config.json").get("manufactured_fixture"):
            raise ValueError("A partial audit is allowed only for explicitly artificial fixture data")
        audit = Auditor()
        with threadpool_limits(limits=1):
            record = audit_task(args.root, args.fixture_task, audit)
        print(json.dumps({"status": "passed", "production_data_used": False,
            "task_id": record["task_id"], "checks": audit.checks,
            "maximum_absolute_numerical_differences": dict(audit.max_absolute_difference)}, indent=2))
    else:
        report = verify_study(args.root, args.analysis or args.root / "analysis",
                              args.output or args.root / "verification/independent_audit.json")
        print(json.dumps({"status": report["status"], "tasks": report["tasks"], "checks": report["checks"]}))


if __name__ == "__main__":
    main()
