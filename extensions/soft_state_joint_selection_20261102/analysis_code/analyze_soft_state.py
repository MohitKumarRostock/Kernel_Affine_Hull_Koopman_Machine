#!/usr/bin/env python3
"""Analyze the frozen joint input-scaling/native-dimension follow-up.

The independent unit is a complete simulation repetition. No model selection
is performed here. One six-contrast family contains four shared-geometry
representation questions and two same-cohort joint-versus-standard K questions.
Other comparisons and all law/noise/horizon decompositions are descriptive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
ARMS = ("G", "P", "K", "R")
VIEWS = ("shared", "independent", "standard_shared", "standard_independent")
KEYS = tuple(f"{view}__{arm}" for view in VIEWS for arm in ARMS)
CAPS = (1, 2, 4, 8, 12)
SCALINGS = ("standard", "minmax", "tanh_z_1")
CANDIDATES = tuple(f"{scaling}_cap{cap}" for scaling in SCALINGS for cap in CAPS)
HORIZONS = (1, 4, 12)
N = 32
FAMILY_ALPHA = 0.05
CLAIMS = (
    ("P_vs_G_nonlinear", "nonlinear", "shared", "P", "shared", "G", 0.97, "3% superiority"),
    ("K_vs_P_nonlinear", "nonlinear", "shared", "K", "shared", "P", 0.97, "3% superiority"),
    ("K_vs_R_nonlinear", "nonlinear", "shared", "K", "shared", "R", 0.97, "3% superiority"),
    ("K_vs_P_linear", "linear", "shared", "K", "shared", "P", 1.05, "5% noninferiority"),
    ("joint_K_vs_standard_K_nonlinear", "nonlinear", "independent", "K",
     "standard_independent", "K", 0.97, "3% superiority"),
    ("joint_K_vs_standard_K_linear", "linear", "independent", "K",
     "standard_independent", "K", 1.05, "5% noninferiority"),
)
PANELS = {"nonlinear": LAWS[1:], "linear": LAWS[:1], "all_descriptive": LAWS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def confined(root, relative):
    relative = Path(relative)
    result = (root / relative).resolve()
    require(not relative.is_absolute() and ".." not in relative.parts and
            result.is_relative_to(root), "Artifact path leaves study root")
    return result


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    require(bool(rows), "Empty output table: " + str(path))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config):
    require(config["master_seed"] == 2026110204 and not config.get("manufactured_fixture", False),
            "Scientific analysis requires the declared fresh production cohort")
    require(config["repetitions"] == N and config["task_count"] == 256,
            "The fixed production budget is 32 repetitions and 256 tasks")
    require(tuple(config["laws"]) == LAWS, "Law roster changed")
    require(tuple(config["horizons"]) == HORIZONS and config["histories"] == [12],
            "History/horizon roster changed")
    require(config["noise_levels"] == [0.0, 0.1], "Noise roster changed")
    require(tuple(config["native_cap_order"]) == CAPS, "Native cap menu changed")
    require(config["split_episodes"] == {"A": 64, "B": 32, "C": 32, "D": 128},
            "Episode allocation changed")
    analysis = config["analysis"]
    require(analysis["family_alpha"] == FAMILY_ALPHA and
            analysis["primary_family_size"] == len(CLAIMS), "Primary family changed")
    require(analysis["superiority_margin"] == .03 and
            analysis["linear_noninferiority_margin"] == .05, "Primary margins changed")
    require(analysis["independent_unit"] == "repetition", "Wrong inferential unit")
    require(analysis["primary_context"] == "mixed_prespecified", "Primary comparison contexts changed")
    require(analysis["primary_claims"] == [{"candidate_context": av, "candidate": a,
                "population": panel, "reference_context": bv, "reference": b,
                "reference_factor": factor}
                for _, panel, av, a, bv, b, factor, _ in CLAIMS], "Primary contrast roster changed")
    require(config["arms"] == list(ARMS) and config["contexts"] == list(VIEWS), "Forecast arm/view roster changed")
    require(tuple(config["input_scaling_order"]) == SCALINGS, "Input-scaling candidate order changed")
    require(all(config["decoder"][key] is False for key in
                ("raw_history_baseline", "residual_target", "calibration")), "Declared direct decoder changed")


def check_stage_chain(root):
    """Require complete outcome sealing before reading any terminal targets."""
    freeze = read_json(root / "freeze.json")
    fits = read_json(root / "FIT_LOCK.json")
    predictions = read_json(root / "PREDICTION_LOCK.json")
    release = read_json(root / "TARGET_RELEASE.json")
    require(freeze["status"] == "frozen", "Source not frozen")
    require(fits["status"] == predictions["status"] == "sealed", "Incomplete stage seal")
    require(release["status"] == "released", "Terminal targets are not released")
    config_hash, freeze_hash = sha256(root / "config.json"), sha256(root / "freeze.json")
    require(freeze["config_sha256"] == config_hash, "Frozen config changed")
    for relative, expected in freeze["files"].items():
        path = confined(root, relative)
        require(path.is_file() and sha256(path) == expected, "Frozen study file changed: " + relative)
    own_key = "analysis_code/" + Path(__file__).name
    require(freeze["files"].get(own_key) == sha256(Path(__file__)), "Executing analyzer differs from frozen source")
    for label, lock in (("fit", fits), ("prediction", predictions)):
        require(lock["task_count"] == 256, "Incomplete " + label + " task roster")
        require(lock["config_sha256"] == config_hash and lock["freeze_sha256"] == freeze_hash,
                "Changed " + label + " config/source link")
    require(fits["phase"] == "fits" and predictions["phase"] == "predictions", "Stage seal role changed")
    for lock, filename in ((fits, "abc_manifest.json"), (predictions, "d_manifest.json")):
        manifest_path = root / "prepared" / filename
        require(lock["manifest_sha256"] == sha256(manifest_path), "Sealed manifest changed: " + filename)
        manifest = read_json(manifest_path)
        require(manifest["config_sha256"] == config_hash and manifest["freeze_sha256"] == freeze_hash,
                "Prepared manifest source/config link changed")
    terminal = read_json(root / "prepared/d_manifest.json")
    require(terminal["fit_lock_sha256"] == sha256(root / "FIT_LOCK.json") and
            terminal["ABC_manifest_sha256"] == sha256(root / "prepared/abc_manifest.json"),
            "D generation no longer bound to frozen fits and ABC")
    publication = read_json(root / "PUBLICATION.json")
    require(publication["status"] == "published" and publication["source_files_verified"] is True and
            publication["freeze_sha256"] == freeze_hash, "Source publication receipt does not bind this freeze")
    publication_time = datetime.fromisoformat(publication["published_utc"].replace("Z", "+00:00"))
    require(datetime.fromisoformat(freeze["created_utc"].replace("Z", "+00:00")) <= publication_time,
            "Publication predates the source freeze")
    for name in ("abc_manifest.json", "d_manifest.json"):
        manifest = read_json(root / "prepared" / name)
        require(manifest["publication_sha256"] == sha256(root / "PUBLICATION.json") and
                publication_time < datetime.fromisoformat(manifest["created_utc"].replace("Z", "+00:00")),
                "Prepared data no longer bound to prior source publication")
    for field, filename in (("fit_lock_sha256", "FIT_LOCK.json"),
                            ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                            ("d_manifest_sha256", "prepared/d_manifest.json")):
        require(release[field] == sha256(root / filename), "Release link changed: " + filename)
    return fits, predictions


def repeated_loss_tables(task_rows):
    """Construct equal-cell repetition aggregates from already standardized losses."""
    index = {}
    for row in task_rows:
        key = (row["repetition"], row["law"], row["noise_index"], row["forecast_key"])
        require(key not in index, "Duplicate task loss")
        require(np.isfinite(row["smse"]) and row["smse"] >= 0, "Invalid task loss")
        require(np.isfinite(row["mse"]) and row["mse"] >= 0, "Invalid raw task loss")
        index[key] = row
    expected = {(rep, law, noise, key) for rep in range(N) for law in LAWS
                for noise in (0, 1) for key in KEYS}
    require(set(index) == expected, "Incomplete factorial task-loss roster")
    repetitions, summaries, conditions = [], [], []
    for panel, laws in PANELS.items():
        for key in KEYS:
            for rep in range(N):
                members = [index[rep, law, noise, key] for law in laws for noise in (0, 1)]
                repetitions.append({"panel": panel, "repetition": rep, "forecast_key": key,
                    "condition_count": len(members),
                    "mse": float(np.mean([r["mse"] for r in members])),
                    "smse": float(np.mean([r["smse"] for r in members]))})
    vectors = {(panel, key): np.array([r["smse"] for r in repetitions
                if r["panel"] == panel and r["forecast_key"] == key])
                for panel in PANELS for key in KEYS}
    for (panel, key), values in vectors.items():
        require(values.shape == (N,), "Wrong repetition count")
        summaries.append({"panel": panel, "forecast_key": key,
            "mean_smse": float(values.mean()), "repetition_sd": float(values.std(ddof=1)),
            "n_repetitions": N})
    for law in LAWS:
        for noise in (0, 1):
            for key in KEYS:
                values = np.array([index[rep, law, noise, key]["smse"] for rep in range(N)])
                conditions.append({"law": law, "noise_index": noise, "forecast_key": key,
                    "mean_smse": float(values.mean()), "repetition_sd": float(values.std(ddof=1)),
                    "n_repetitions": N, "inference": "descriptive condition; no primary claim"})
    primary, paired, distributions = [], [], []
    critical = float(student.ppf(1 - FAMILY_ALPHA / len(CLAIMS), N - 1))
    for cid, panel, candidate_view, candidate, comparator_view, comparator, factor, margin in CLAIMS:
        candidate_key, comparator_key = candidate_view + "__" + candidate, comparator_view + "__" + comparator
        left, right = vectors[panel, candidate_key], vectors[panel, comparator_key]
        delta = left - factor * right
        mean, sd = float(delta.mean()), float(delta.std(ddof=1))
        se = sd / np.sqrt(N)
        upper = mean + critical * se
        p = float(student.cdf(mean / se, N - 1)) if se else (0. if mean < 0 else 1. if mean > 0 else .5)
        primary.append({"contrast_id": cid,
            "view": candidate_view if candidate_view == comparator_view else candidate_view + "_vs_" + comparator_view,
            "panel": panel, "candidate_context": candidate_view, "comparator_context": comparator_view,
            "candidate_key": candidate_key, "comparator_key": comparator_key,
            "candidate": candidate, "comparator": comparator, "margin": margin,
            "comparator_factor": factor, "candidate_mean_smse": float(left.mean()),
            "comparator_mean_smse": float(right.mean()),
            "point_relative_gain_percent": relative_gain(left, right),
            "margin_difference_mean": mean, "margin_difference_sd": sd,
            "margin_difference_se": float(se), "critical_t": critical,
            "simultaneous_one_sided_upper": float(upper),
            "criterion_established": bool(upper < 0), "p_one_sided": p,
            "p_bonferroni": min(1., len(CLAIMS) * p), "family_alpha": FAMILY_ALPHA,
            "family_size": len(CLAIMS), "n_repetitions": N, "df": N - 1,
            "strict_D_win_repetitions": int(np.count_nonzero(left < right))})
        for rep in range(N):
            paired.append({"contrast_id": cid, "repetition": rep,
                "candidate_key": candidate_key, "comparator_key": comparator_key,
                "candidate_smse": float(left[rep]), "comparator_smse": float(right[rep]),
                "margin_difference": float(delta[rep])})
        raw_delta = left - right
        distributions.append({"contrast_id": cid, "panel": panel,
            "candidate_key": candidate_key, "comparator_key": comparator_key,
            "n_repetitions": N, "difference_mean": float(raw_delta.mean()),
            "difference_sd": float(raw_delta.std(ddof=1)),
            "minimum": float(raw_delta.min()), "q25": float(np.quantile(raw_delta, .25)),
            "median": float(np.median(raw_delta)), "q75": float(np.quantile(raw_delta, .75)),
            "maximum": float(raw_delta.max()), "wins": int(np.count_nonzero(raw_delta < 0)),
            "ties": int(np.count_nonzero(raw_delta == 0)), "losses": int(np.count_nonzero(raw_delta > 0)),
            "scope": "descriptive distribution of unadjusted paired loss differences; primary decisions use margin bounds"})
    secondary = []
    critical95 = float(student.ppf(.975, N - 1))
    for panel in ("nonlinear", "linear"):
        comparisons = [(view + "__" + a, view + "__" + b)
                       for view in VIEWS for a, b in (("P", "G"), ("K", "P"), ("K", "R"), ("R", "P"))]
        comparisons += [(independent + "__" + arm, shared + "__" + arm)
                        for independent, shared in (("independent", "shared"),
                                                    ("standard_independent", "standard_shared"))
                        for arm in ARMS]
        comparisons += [(view + "__" + arm, "standard_" + view + "__" + arm)
                        for view in ("shared", "independent") for arm in ARMS]
        for candidate, comparator in comparisons:
            left, right = vectors[panel, candidate], vectors[panel, comparator]
            delta = left - right
            mean, sd = float(delta.mean()), float(delta.std(ddof=1))
            se = sd / np.sqrt(N)
            secondary.append({"panel": panel, "candidate": candidate, "comparator": comparator,
                "candidate_mean_smse": float(left.mean()), "comparator_mean_smse": float(right.mean()),
                "difference_mean": mean, "difference_sd": sd, "difference_se": float(se),
                "descriptive_95_lower": float(mean - critical95 * se),
                "descriptive_95_upper": float(mean + critical95 * se),
                "point_relative_gain_percent": relative_gain(left, right),
                "strict_D_win_repetitions": int(np.count_nonzero(left < right)),
                "n_repetitions": N,
                "scope": "unadjusted descriptive paired interval; no primary claim or selection"})
    optimization_pairs, optimization_summary = [], []
    for panel in ("nonlinear", "linear"):
        for view in ("shared", "independent"):
            for arm in ARMS:
                left = vectors[panel, view + "__" + arm]
                right = vectors[panel, "standard_" + view + "__" + arm]
                delta = left - right
                for rep in range(N):
                    optimization_pairs.append({"panel": panel, "view": view, "arm": arm,
                        "repetition": rep, "joint_smse": float(left[rep]),
                        "standard_smse": float(right[rep]), "difference": float(delta[rep]),
                        "scope": "paired fresh-cohort comparison; no independent task or origin units"})
                sd = float(delta.std(ddof=1))
                se = sd / np.sqrt(N)
                optimization_summary.append({"panel": panel, "view": view, "arm": arm,
                    "joint_mean_smse": float(left.mean()), "standard_mean_smse": float(right.mean()),
                    "point_relative_gain_percent": relative_gain(left, right),
                    "difference_mean": float(delta.mean()), "difference_sd": sd,
                    "descriptive_95_lower": float(delta.mean() - critical95 * se),
                    "descriptive_95_upper": float(delta.mean() + critical95 * se),
                    "strict_D_win_repetitions": int(np.count_nonzero(delta < 0)),
                    "tie_repetitions": int(np.count_nonzero(delta == 0)),
                    "strict_D_loss_repetitions": int(np.count_nonzero(delta > 0)),
                    "n_repetitions": N,
                    "scope": "unadjusted descriptive interval; only declared independent K contrasts have primary bounds"})
    return {"repetition_losses.csv": repetitions, "panel_summary.csv": summaries,
            "condition_summary.csv": conditions, "primary_contrasts.csv": primary,
            "primary_paired_repetitions.csv": paired, "secondary_descriptive.csv": secondary,
            "primary_difference_distribution.csv": distributions,
            "optimization_repetition_pairs.csv": optimization_pairs,
            "optimization_summary.csv": optimization_summary}


def relative_gain(left, right):
    """Ratio of aggregate means; undefined when the reference mean is zero."""
    reference = float(np.mean(right))
    return float(100 * (1 - np.mean(left) / reference)) if reference > 0 else None


def numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for name, child in value.items():
            yield from numeric_leaves(child, prefix + ("/" if prefix else "") + str(name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(np.isfinite(value), "Nonfinite resource value")
        yield prefix, float(value)


def analyze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Analysis destination already exists")
    config = read_json(root / "config.json")
    validate_config(config)
    fit_lock, prediction_lock = check_stage_chain(root)
    abc_manifest, d_manifest = [read_json(root / "prepared" / name)
                               for name in ("abc_manifest.json", "d_manifest.json")]
    abc = {row["task_id"]: row for row in abc_manifest["tasks"]}
    terminal = {row["task_id"]: row for row in d_manifest["tasks"]}
    require(len(abc_manifest["tasks"]) == len(d_manifest["tasks"]) == len(abc) == len(terminal) == 256,
            "Incomplete or duplicated prepared task roster")
    require(set(abc) == set(terminal), "Prepared task rosters differ")
    factorial = {(row["repetition"], row["law"], row["noise_index"]) for row in terminal.values()}
    require(factorial == {(rep, law, noise) for rep in range(N) for law in LAWS for noise in (0, 1)},
            "Prepared factorial roster differs")
    task_rows, horizon_rows, selections, resources, optimization_tasks = [], [], [], [], []
    basis_rows, width_rows, b_scores, c_losses = [], [], [], []
    for tid in sorted(terminal):
        initial, final = abc[tid], terminal[tid]
        for field in ("repetition", "law", "noise_index", "noise_level"):
            require(initial[field] == final[field], "Task metadata changed")
        basic = {"task_id": tid, **{k: final[k] for k in ("repetition", "law", "noise_index", "noise_level")}}
        apath, ypath = confined(root, initial["input"]), confined(root, final["target"])
        require(sha256(apath) == initial["sha256"], "A/B/C arrays changed")
        require(sha256(ypath) == final["target_sha256"], "D targets changed")
        with np.load(apath, allow_pickle=False) as data:
            ay = data["A_Y"]
            scale = np.maximum(np.var(ay, axis=0, ddof=0), 1e-12)
            require(scale.shape == (3,) and np.isfinite(scale).all(), "Invalid A normalization")
            if "loss_scale" in data.files:
                require(np.array_equal(scale, data["loss_scale"]), "Stored A normalization differs")
        with np.load(ypath, allow_pickle=False) as data:
            y = data["D_Y"]
        require(y.shape == (1664, 3) and np.isfinite(y).all(), "Wrong D target shape or values")
        relative = f"predictions/{tid}/predictions.npz"
        ppath = root / relative
        require(prediction_lock["files"].get(relative) == sha256(ppath), "Unsealed D predictions")
        with np.load(ppath, allow_pickle=False) as data:
            require(set(data.files) == set(KEYS), "Wrong D prediction roster")
            for key in KEYS:
                pred = data[key]
                require(pred.shape == y.shape and np.isfinite(pred).all(), "Invalid D forecast")
                mse = np.mean((y - pred) ** 2, axis=0)
                smse = mse / scale
                task_rows.append({**basic, "forecast_key": key,
                    "mse": float(mse.mean()), "smse": float(smse.mean())})
                for h, m, s, variance in zip(HORIZONS, mse, smse, scale):
                    horizon_rows.append({**basic, "forecast_key": key, "horizon": h,
                        "mse": float(m), "smse": float(s), "A_target_variance": float(variance),
                        "D_rows": len(y)})
        srelative = f"results/{tid}/fit_summary.json"
        require(fit_lock["files"].get(srelative) == sha256(root / srelative), "Unsealed fit summary")
        summary = read_json(root / srelative)
        require(set(summary["candidates"]) == set(CANDIDATES), "Incomplete scaler/cap candidate roster")
        for cid, candidate in summary["candidates"].items():
            cap, scaling = candidate["cap"], candidate["input_scaling"]
            require(cid == f"{scaling}_cap{cap}" and cap in CAPS and scaling in SCALINGS,
                    "Unexpected candidate identity")
            geometry_key = {"candidate_id": cid, "input_scaling": scaling, "cap": cap}
            for dimension in candidate["local_dimensions"]:
                basis_rows.append({**basic, **geometry_key, "model_index": dimension["model_index"],
                    "block_index": dimension["block_index"], "requested_cap": dimension["requested_cap"],
                    "initial_admissible_rank": dimension["initial_admissible_rank"],
                    "effective_rank": dimension["effective_rank"],
                    "reference_count": dimension["local_reference_count"],
                    "rank_zero": dimension["effective_rank"] == 0,
                    "range_reduction_steps": dimension["range_reduction_steps"], "status": dimension["status"]})
            for arm in ARMS:
                head = candidate["arms"][arm]
                width_rows.append({**basic, **geometry_key, "arm": arm,
                    "gate_width": candidate["block_widths"]["gates"],
                    "position_width": candidate["block_widths"]["position"],
                    "departure_width": candidate["block_widths"]["kahm"] if arm == "K" else
                                       candidate["block_widths"]["rbf"] if arm == "R" else 0,
                    "feature_width": head["feature_width"], "centered_feature_rank": head["centered_feature_rank"],
                    "coefficient_count": head["coefficient_count"], "B_mean_smse": head["B_mean_smse"]})
                for penalty in head["alpha_candidates"]:
                    for hi, h in enumerate(HORIZONS):
                        b_scores.append({**basic, **geometry_key, "arm": arm, "alpha": penalty["alpha"],
                            "horizon": h, "B_mse": penalty["B_mse_by_horizon"][hi],
                            "B_smse": penalty["B_smse_by_horizon"][hi], "status": penalty["status"]})
        for view in VIEWS:
            require(set(summary["selections"][view]) == set(ARMS), "Incomplete representation selections")
            eligible = tuple(cid for cid in CANDIDATES
                             if not view.startswith("standard_") or cid.startswith("standard_cap"))
            selector_shared = view in ("shared", "standard_shared")
            for arm in ARMS:
                choice = summary["selections"][view][arm]
                require(choice["cap"] in CAPS, "Undeclared selected cap")
                require(choice["input_scaling"] in SCALINGS and
                        choice["candidate_id"] == f'{choice["input_scaling"]}_cap{choice["cap"]}',
                        "Selected scaler/cap identity differs")
                selector = "P" if selector_shared else arm
                scores = [summary["candidates"][cid]["arms"][selector]["B_mean_smse"] for cid in eligible]
                require(np.isfinite(scores).all() and min(scores) >= 0, "Invalid B candidate scores")
                best_id = eligible[int(np.argmin(scores))]
                require(choice["candidate_id"] == best_id, "Selected geometry differs from B first minimum")
                chosen_record = summary["candidates"][best_id]["arms"][arm]
                for field in ("B_mean_smse", "B_smse_by_horizon", "alpha_by_horizon"):
                    require(choice[field] == chosen_record[field], "Selected record differs from its cached candidate")
                require(len(choice["alpha_by_horizon"]) == 3, "Wrong selected penalty dimension")
                local = summary["candidates"][choice["candidate_id"]]["local_dimensions"]
                ranks = [row["effective_rank"] for row in local]
                selections.append({**basic, "view": view, "arm": arm,
                    "candidate_id": choice["candidate_id"], "input_scaling": choice["input_scaling"],
                    "cap": choice["cap"],
                    "feature_width": choice["feature_width"],
                    "centered_feature_rank": choice["centered_feature_rank"],
                    "local_rank_sum": sum(ranks), "minimum_local_rank": min(ranks),
                    "maximum_local_rank": max(ranks), "rank_zero_models": ranks.count(0),
                    "alpha_h1": choice["alpha_by_horizon"][0],
                    "alpha_h4": choice["alpha_by_horizon"][1],
                    "alpha_h12": choice["alpha_by_horizon"][2],
                    "B_smse_h1": choice["B_smse_by_horizon"][0],
                    "B_smse_h4": choice["B_smse_by_horizon"][1],
                    "B_smse_h12": choice["B_smse_by_horizon"][2],
                    "B_mean_smse": choice["B_mean_smse"]})
            if selector_shared:
                require(len({summary["selections"][view][arm]["candidate_id"] for arm in ARMS}) == 1,
                        "Shared-geometry view uses different banks")
        task_scores = {row["forecast_key"]: row["smse"] for row in task_rows if row["task_id"] == tid}
        for view in ("shared", "independent"):
            for arm in ARMS:
                joint = summary["selections"][view][arm]
                standard = summary["selections"]["standard_" + view][arm]
                if view == "independent" or arm == "P":
                    require(joint["B_mean_smse"] <= standard["B_mean_smse"],
                            "Joint search violates its validation superset guarantee")
                joint_d = task_scores[view + "__" + arm]
                standard_d = task_scores["standard_" + view + "__" + arm]
                optimization_tasks.append({**basic, "view": view, "arm": arm,
                    "joint_candidate_id": joint["candidate_id"],
                    "standard_candidate_id": standard["candidate_id"],
                    "selected_geometry_changed": joint["candidate_id"] != standard["candidate_id"],
                    "joint_B_mean_smse": joint["B_mean_smse"],
                    "standard_B_mean_smse": standard["B_mean_smse"],
                    "B_difference": joint["B_mean_smse"] - standard["B_mean_smse"],
                    "joint_D_smse": joint_d, "standard_D_smse": standard_d,
                    "D_difference": joint_d - standard_d,
                    "scope": "paired task diagnostics; inference uses complete repetition means"})
        crelative = f"results/{tid}/C_diagnostics.json"
        require(fit_lock["files"].get(crelative) == sha256(root / crelative), "Unsealed C diagnostics")
        cdiagnostic = read_json(root / crelative)
        require(cdiagnostic["selection_performed"] is False, "C selected an outcome")
        for key in KEYS:
            for hi, horizon in enumerate(HORIZONS):
                c_losses.append({**basic, "forecast_key": key, "horizon": horizon,
                    "mse": cdiagnostic["losses"][key]["mse_by_horizon"][hi],
                    "smse": cdiagnostic["losses"][key]["smse_by_horizon"][hi],
                    "role": "C diagnostic only; no selection or calibration"})
        for measure, value in numeric_leaves(summary["resources"]):
            resources.append({**basic, "resource": "fit/" + measure, "value": value})
        receipt_path = root / "predictions" / tid / "prediction_receipt.json"
        require(prediction_lock["files"].get(str(receipt_path.relative_to(root))) == sha256(receipt_path),
                "Unsealed prediction resource receipt")
        receipt = read_json(receipt_path)
        for measure in ("wall_seconds", "cpu_seconds"):
            if measure in receipt:
                resources.append({**basic, "resource": "prediction/" + measure, "value": float(receipt[measure])})
    tables = {"task_losses.csv": task_rows, "horizon_losses.csv": horizon_rows,
              **repeated_loss_tables(task_rows), "selection_by_task.csv": selections,
              "resource_by_task.csv": resources, "basis_dimensions.csv": basis_rows,
              "feature_widths.csv": width_rows, "B_candidate_scores.csv": b_scores,
              "C_diagnostic_losses.csv": c_losses, "optimization_task_pairs.csv": optimization_tasks}
    horizon_summary = []
    for panel, laws in PANELS.items():
        for key in KEYS:
            for horizon in HORIZONS:
                values = [r["smse"] for r in horizon_rows if r["law"] in laws and
                          r["forecast_key"] == key and r["horizon"] == horizon]
                require(len(values) == N * 2 * len(laws), "Incomplete descriptive horizon aggregation")
                horizon_summary.append({"panel": panel, "forecast_key": key, "horizon": horizon,
                    "mean_smse": float(np.mean(values)), "n_repetitions": N,
                    "scope": "descriptive horizon; no independent horizon units or primary claim"})
    tables["horizon_summary.csv"] = horizon_summary
    frequencies, scaling_frequencies, joint_frequencies = [], [], []
    for panel, laws in PANELS.items():
        for view in VIEWS:
            for arm in ARMS:
                rows = [r for r in selections if r["law"] in laws and r["view"] == view and r["arm"] == arm]
                counts = Counter(r["cap"] for r in rows)
                scaler_counts = Counter(r["input_scaling"] for r in rows)
                joint_counts = Counter(r["candidate_id"] for r in rows)
                for cap in CAPS:
                    frequencies.append({"panel": panel, "view": view, "arm": arm, "cap": cap,
                        "task_count": counts[cap], "total_tasks": len(rows), "fraction": counts[cap] / len(rows)})
                for scaling in SCALINGS:
                    scaling_frequencies.append({"panel": panel, "view": view, "arm": arm,
                        "input_scaling": scaling, "task_count": scaler_counts[scaling],
                        "total_tasks": len(rows), "fraction": scaler_counts[scaling] / len(rows)})
                    for cap in CAPS:
                        cid = f"{scaling}_cap{cap}"
                        joint_frequencies.append({"panel": panel, "view": view, "arm": arm,
                            "candidate_id": cid, "input_scaling": scaling, "cap": cap,
                            "task_count": joint_counts[cid], "total_tasks": len(rows),
                            "fraction": joint_counts[cid] / len(rows)})
    tables["cap_selection_frequencies.csv"] = frequencies
    tables["scaling_selection_frequencies.csv"] = scaling_frequencies
    tables["joint_selection_frequencies.csv"] = joint_frequencies
    grouped = defaultdict(list)
    for row in resources:
        grouped[row["resource"]].append(row["value"])
    resource_summary = []
    for measure, values in sorted(grouped.items()):
        require(len(values) == 256, "Incomplete task resource measure: " + measure)
        resource_summary.append({"resource": measure, "n_tasks": len(values),
            "sum": float(sum(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "minimum": float(min(values)), "maximum": float(max(values)),
            "scope": "measured study computation; reused banks do not equal isolated arm costs"})
    tables["resource_summary.csv"] = resource_summary
    output.mkdir(parents=True)
    for name, rows in tables.items():
        write_csv(output / name, rows)
    primary = tables["primary_contrasts.csv"]
    summary = {"schema": "soft-state-joint-selection-analysis-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": 256, "forecast_arrays": 256 * len(KEYS), "independent_repetitions": N,
        "primary_family_size": len(CLAIMS), "criteria_established": sum(r["criterion_established"] for r in primary),
        "primary_contrasts": primary, "panel_summary": tables["panel_summary.csv"],
        "config_sha256": sha256(root / "config.json"), "freeze_sha256": sha256(root / "freeze.json"),
        "fit_lock_sha256": sha256(root / "FIT_LOCK.json"),
        "prediction_lock_sha256": sha256(root / "PREDICTION_LOCK.json"),
        "target_release_sha256": sha256(root / "TARGET_RELEASE.json"),
        "analyzer_sha256": sha256(Path(__file__)),
        "tables": {name: {"rows": len(rows), "sha256": sha256(output / name)} for name, rows in tables.items()},
        "limitations": [
            "Four inherited simulated laws with fresh independent repetitions, not new application domains.",
            "The target is an observed future response, not a soft coordinate or a Koopman rollout.",
            "The 3% threshold is an inherited research convention, not a deployment-derived utility threshold.",
            "Paired Student bounds assume independent repetition aggregates and are approximate outside normality.",
            "Shared geometry is selected by P; the independently selected K-versus-standard K pair is a separate primary question.",
            "Other independent-geometry comparisons, including K versus R, remain descriptive.",
            "A larger candidate set guarantees no greater selected B loss for its selector, but not lower D loss.",
            "The finite scaler/cap/penalty grid does not optimize all geometry parameters or establish a global optimum.",
            "Nested feature blocks do not imply monotone validation or test loss under fitted regularization.",
            "Failure of a positive margin is not an equivalence finding or proof of zero benefit.",
            "Secondary intervals are unadjusted descriptive intervals and cannot replace primary claims."]}
    write_json(output / "analysis_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.output or args.root / "analysis")
    print(json.dumps({"status": result["status"], "tasks": result["tasks"],
        "criteria_established": result["criteria_established"], "primary_family_size": len(CLAIMS)}))


if __name__ == "__main__":
    main()
