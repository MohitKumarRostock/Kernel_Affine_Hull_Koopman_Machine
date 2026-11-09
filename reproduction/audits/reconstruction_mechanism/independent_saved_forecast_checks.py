#!/usr/bin/env python3
"""Independent arithmetic audit of pinned principal-study numeric evidence.

This script does not import the study's fitting, analysis, or verifier modules.
It checks original numeric NPZs (pickle disabled), their published hashes,
window construction and paired-noise records, saved-forecast errors, and the
eight reported statistical tables. Selected B/C candidate scores are checked
for internal consistency, not recomputed by refitting candidates. Resource
records and bound-diagnostic arithmetic are checked, not remeasured.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "remote_evidence/extensions/reconstruction_mechanism"
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
NONLINEAR = LAWS[1:]
NOISES = (0.0, 0.1)
SEEDS = tuple(range(946100, 946124))
HORIZONS = (1, 4, 12)
METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "kahm_departure",
           "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy")
POLICY = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded")
COUNTS = Counter()
MAXDIFF = defaultdict(float)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(a):
    a = np.ascontiguousarray(a)
    descriptor = json.dumps({"dtype": str(a.dtype), "shape": list(a.shape)}, sort_keys=True).encode()
    return hashlib.sha256(descriptor + b"\0" + a.tobytes()).hexdigest()


def avg(values):
    values = list(values)
    return math.fsum(float(x) for x in values) / len(values)


def sample_sd(values):
    values = list(values)
    center = avg(values)
    return math.sqrt(math.fsum((float(x) - center) ** 2 for x in values) / (len(values) - 1))


def close(actual, expected, label):
    actual, expected = float(actual), float(expected)
    assert math.isfinite(actual) and math.isfinite(expected), label
    MAXDIFF[label.split("/")[0]] = max(MAXDIFF[label.split("/")[0]], abs(actual - expected))
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (label, actual, expected)
    COUNTS[label.split("/")[0] + "_numeric_checks"] += 1


def table(name):
    with (STUDY / "analysis" / (name + ".csv")).open(newline="") as f:
        rows = list(csv.DictReader(f))
    COUNTS[name + "_rows"] = len(rows)
    return rows


def compare(row, values, name):
    for key, value in values.items():
        if isinstance(value, str):
            assert row[key] == value, (name, key, row[key], value)
            COUNTS[name + "_categorical_checks"] += 1
        elif isinstance(value, (bool, np.bool_)):
            assert row[key] == str(bool(value)), (name, key)
            COUNTS[name + "_categorical_checks"] += 1
        else:
            close(row[key], value, name + "/" + key)


def main():
    global STUDY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=STUDY,
                        help="Restored extensions/reconstruction_mechanism directory")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_checks/independent_saved_forecast_checks.json",
                        help="Output JSON for the independent numeric audit")
    args = parser.parse_args()
    STUDY = args.study_dir.resolve()
    manifest = json.loads((STUDY / "prepared_inputs/manifest.json").read_text())
    config = json.loads((STUDY / "config.json").read_text())
    assert manifest["config_sha256"] == digest(STUDY / "config.json")
    assert len(manifest["tasks"]) == 192
    expected_roster = set(itertools.product(LAWS, NOISES, SEEDS))
    assert {(x["law"], x["noise_level"], x["seed"]) for x in manifest["tasks"]} == expected_roster
    metrics, by_id, paired, scales, summaries = {}, {}, {}, {}, {}
    policy_counts = {scope: Counter() for scope in ("nonlinear", "linear")}
    gamma_counts = {m: Counter() for m in METHODS if m != "ridge"}
    active_counts = {m: Counter() for m in METHODS if m != "ridge"}
    measured = {m: defaultdict(list) for m in METHODS}
    generation_seeds = set()
    for task in manifest["tasks"]:
        name = task["task_id"]
        key = (task["law"], task["noise_level"], task["seed"])
        by_id[name] = key
        input_path = STUDY / task["input"]
        prediction_path = STUDY / "results" / name / "predictions.npz"
        summary = json.loads((prediction_path.parent / "summary.json").read_text())
        summaries[name] = summary
        assert digest(input_path) == task["sha256"] == summary["input_sha256"]
        assert digest(prediction_path) == summary["files"]["predictions.npz"]
        assert summary["freeze_sha256"] == digest(STUDY / "freeze.json")
        COUNTS["source_input_and_prediction_files_checked"] += 2
        with np.load(input_path, allow_pickle=False) as source:
            a = {k: source[k] for k in source.files}
        assert set(a) == set(task["array_sha256"])
        for field, value in a.items():
            assert value.dtype.kind in "fiu" and np.isfinite(value).all(), (name, field)
            assert array_digest(value) == task["array_sha256"][field], (name, field)
            COUNTS["input_array_hashes_checked"] += 1
        sa = float(np.std(a["A_base_episodes"], ddof=0))
        close(a["A_base_episode_std"], sa, "window_construction/A_scale")
        close(a["noise_std"], task["noise_level"] * sa, "window_construction/noise_scale")
        for split, episodes, nrows in (("A", 64, 512), ("B", 32, 256), ("C", 32, 256), ("D", 128, 1664)):
            raw = a[split + "_episodes"]
            assert raw.shape == (episodes, 72)
            assert np.array_equal(raw, a[split + "_base_episodes"] + task["noise_level"] * sa * a[split + "_unit_noise"])
            groups, origins, indices = (a[split + suffix] for suffix in ("_groups", "_origins", "_row_indices"))
            assert groups.shape == origins.shape == indices.shape == (nrows,)
            assert len(np.unique(indices)) == nrows and np.all(np.diff(indices) > 0)
            assert np.array_equal(groups, indices // 13)
            assert np.array_equal(origins, 11 + 4 * (indices % 13))
            assert np.array_equal(a[split + "_X"], raw[groups[:, None], origins[:, None] - np.arange(11, -1, -1)])
            assert np.array_equal(a[split + "_Y"], raw[groups[:, None], origins[:, None] + np.array(HORIZONS)])
            if task["noise_level"] == 0.0:
                generation_seeds.add(task["splits"][split]["generation_seed"])
            COUNTS["episode_partition_and_window_checks"] += 1
        pair_key = (task["law"], task["seed"])
        pair_record = {k: task["array_sha256"][k] for k in a if k.endswith(("_base_episodes", "_unit_noise", "_row_indices"))}
        if pair_key in paired:
            assert paired[pair_key] == pair_record
            COUNTS["paired_noise_task_pairs_checked"] += 1
        else:
            paired[pair_key] = pair_record
        target_means = [avg(a["A_Y"][:, h]) for h in range(3)]
        variance = np.array([avg((float(y) - target_means[h]) ** 2 for y in a["A_Y"][:, h]) for h in range(3)])
        scale = np.maximum(variance, 1e-12)
        scales[name] = (variance, scale)
        for h in range(3):
            close(a["loss_scale"][h], scale[h], "normalization/variance")
        with np.load(prediction_path, allow_pickle=False) as source:
            assert set(source.files) == {"Y", *METHODS}
            assert np.array_equal(source["Y"], a["D_Y"])
            predictions = {m: source[m] for m in METHODS}
        selected = summary["fit_diagnostics"]["policy"]["selected_method"]
        assert selected in POLICY
        assert np.array_equal(predictions["nonkahm_policy"], predictions[selected])
        COUNTS["selected_policy_forecast_identities_checked"] += 1
        scope = "linear" if task["law"] == "linear_ar" else "nonlinear"
        policy_counts[scope][selected] += 1
        for method, prediction in predictions.items():
            assert prediction.shape == (1664, 3) and np.isfinite(prediction).all()
            assert array_digest(prediction) == summary["prediction_array_sha256"][method]
            error = prediction - a["D_Y"]
            mse = np.array([avg(float(e) ** 2 for e in error[:, h]) for h in range(3)])
            metrics[key + (method,)] = {"h_nmse": mse / scale, "h_mse": mse,
                "h_max": np.max(np.abs(error), axis=0), "nmse": avg(mse / scale),
                "raw_mse": avg(mse), "max_abs_error": float(np.max(np.abs(error)))}
            COUNTS["forecast_arrays_checked"] += 1
            COUNTS["forecast_values_checked"] += prediction.size
        fd = summary["fit_diagnostics"]
        for method, record in fd["B"].items():
            if "selected" not in record:
                continue
            assert record["fit_split"] == "A" and record["selection_split"] == "B"
            for h, choice in enumerate(record["selected"]):
                values = [float(x["mse_by_horizon"][h]) for x in record["candidates"]]
                assert choice["candidate_index"] == int(np.argmin(values))
                close(choice["mse"], min(values), "recorded_selection/B_mse")
                COUNTS["B_selected_horizon_records_checked"] += 1
        for method, record in fd["C"].items():
            if "candidates" not in record:
                continue
            assert record["selection_split"] == "C"
            for h in range(3):
                index = int(np.argmin([x["mse_by_horizon"][h] for x in record["candidates"]]))
                assert record["selected_indices"][h] == index
                assert record["gamma_by_horizon"][h] == record["candidates"][index]["gamma"]
                close(record["selected_mse_by_horizon"][h], record["candidates"][index]["mse_by_horizon"][h], "recorded_selection/C_mse")
                COUNTS["C_selected_horizon_records_checked"] += 1
        policy_scores = fd["policy"]["candidates"]
        assert tuple(x["method"] for x in policy_scores) == POLICY
        for item in policy_scores:
            score = avg(np.array(fd["C"][item["method"]]["selected_mse_by_horizon"]) / scale)
            close(item["normalized_mse"], score, "recorded_selection/policy_score")
        assert selected == min(policy_scores, key=lambda x: x["normalized_mse"])["method"]
        for method in gamma_counts:
            gammas = fd["gamma_by_horizon"][selected] if method == "nonkahm_policy" and selected != "ridge" else (fd["gamma_by_horizon"][method] if method != "nonkahm_policy" else [0., 0., 0.])
            for gamma in gammas:
                gamma_counts[method][str(gamma)] += 1
                active_counts[method][scope] += int(gamma > 0)
        for method, bound in fd["bounds"].items():
            if not bound.get("global_bound", False):
                continue
            for h in range(3):
                coefficient = bound["centered_response_rms_by_horizon"][h] / (2 * math.sqrt(bound["alpha_by_horizon"][h]))
                close(bound["coefficient_norm_bound_by_horizon"][h], coefficient, "recorded_bounds/coefficient")
                unscaled = abs(bound["response_mean_by_horizon"][h]) + (bound["radius"] + bound["training_feature_mean_norm"]) * coefficient
                close(bound["readout_bound_by_horizon"][h], unscaled, "recorded_bounds/readout")
                close(bound["correction_bound_by_horizon"][h], fd["gamma_by_horizon"][method][h] * unscaled, "recorded_bounds/correction")
                assert bound["coefficient_norm_by_horizon"][h] <= coefficient + 1e-12
        resource = summary["resources"]
        for method, record in resource["prediction"]["per_method"].items():
            for stage, nodes in (("fit_and_selection", resource["fit"]["fit_nodes"]), ("predict", resource["prediction"]["predict_nodes"])):
                names = record[stage]["nodes"]
                assert len(names) == len(set(names))
                for measure in ("wall_seconds", "cpu_seconds"):
                    close(record[stage][measure], math.fsum(nodes[n][measure] for n in names), "recorded_resources/" + measure)
            measured[method]["fit_seconds"].append(record["fit_and_selection"]["wall_seconds"])
            measured[method]["prediction_milliseconds"].append(1000 * record["predict"]["wall_seconds"])
            measured[method]["state_kib"].append(record["state"]["pickle_bytes"] / 1024)
    assert len(generation_seeds) == 384
    for row in table("task_metrics"):
        key = by_id[row["task_id"]] + (row["method"],)
        compare(row, {k: metrics[key][k] for k in ("nmse", "raw_mse", "max_abs_error")}, "task_metrics")
        compare(row, {"n_origins": 1664, "n_horizons": 3, "n_target_cells": 4992}, "task_metrics")
    for row in table("horizon_metrics"):
        metric = metrics[by_id[row["task_id"]] + (row["method"],)]
        h = HORIZONS.index(int(row["horizon"]))
        variance, scale = scales[row["task_id"]]
        compare(row, {"nmse": metric["h_nmse"][h], "raw_mse": metric["h_mse"][h], "max_abs_error": metric["h_max"][h],
                      "A_target_variance": variance[h], "loss_scale": scale[h], "variance_floor_used": variance[h] < 1e-12, "n_origins": 1664}, "horizon_metrics")
    for name, horizon_specific in (("condition_metrics", False), ("condition_horizon_metrics", True)):
        for row in table(name):
            records = [metrics[(row["law"], float(row["noise_level"]), seed, row["method"])] for seed in SEEDS]
            h = HORIZONS.index(int(row["horizon"])) if horizon_specific else None
            values = [r["nmse"] if h is None else r["h_nmse"][h] for r in records]
            raw = [r["raw_mse"] if h is None else r["h_mse"][h] for r in records]
            compare(row, {"mean_nmse": avg(values), "sample_sd_nmse": sample_sd(values), "mean_raw_mse": avg(raw), "n_repetitions": 24, "inference": "descriptive"}, name)
    reps = {}
    scopes = {"nonlinear": NONLINEAR, "linear": ("linear_ar",), "all_descriptive": LAWS}
    for scope, laws in scopes.items():
        for method in METHODS:
            reps[(scope, method)] = {seed: avg(metrics[(law, noise, seed, method)]["nmse"] for law in laws for noise in NOISES) for seed in SEEDS}
    for row in table("replicate_metrics"):
        compare(row, {"nmse": reps[(row["scope"], row["method"])][int(row["seed"])], "n_conditions": 2 * len(scopes[row["scope"]]), "equal_horizon_weight": True, "equal_condition_weight": True}, "replicate_metrics")
    for row in table("overview_metrics"):
        values = list(reps[(row["scope"], row["method"])].values())
        compare(row, {"mean_nmse": avg(values), "sample_sd_nmse": sample_sd(values), "n_repetitions": 24}, "overview_metrics")
    primary, contrasts = [], {}
    critical = float(t.isf(0.05 / 6, 23))
    for scope, method in [("nonlinear", m) for m in ("common", "rbf_departure", "raw_rbf", "raw_hgb", "nonkahm_policy")] + [("linear", "ridge")]:
        factor = 0.97 if scope == "nonlinear" else 1.05
        name = "nonlinear_vs_" + method if scope == "nonlinear" else "linear_noninferiority_vs_ridge"
        candidate, comparator = reps[(scope, "kahm_departure")], reps[(scope, method)]
        gaps = {seed: factor * comparator[seed] - candidate[seed] for seed in SEEDS}
        mean, sd = avg(gaps.values()), sample_sd(gaps.values())
        se = sd / math.sqrt(24)
        lower = mean - critical * se
        p = float(t.sf(mean / se, 23))
        result = {"contrast_id": name, "candidate_mean_nmse": avg(candidate.values()), "comparator_mean_nmse": avg(comparator.values()),
            "point_relative_gain": 1 - avg(candidate.values()) / avg(comparator.values()), "margin_gap_mean": mean,
            "margin_gap_sample_sd": sd, "margin_gap_se": se, "critical_t": critical, "simultaneous_one_sided_lower": lower,
            "p_one_sided": p, "p_bonferroni": min(1., 6 * p), "positive_margin_repetitions": sum(x > 0 for x in gaps.values()),
            "criterion_established": lower > 0, "n_repetitions": 24, "df": 23, "family_alpha": 0.05, "family_size": 6,
            "comparator_factor": factor, "margin": 0.03 if scope == "nonlinear" else 0.05}
        primary.append(result)
        contrasts[name] = (candidate, comparator, gaps, factor)
    for row in table("primary_contrasts"):
        result = next(x for x in primary if x["contrast_id"] == row["contrast_id"])
        compare(row, result, "primary_contrasts")
    for row in table("replicate_contrasts"):
        candidate, comparator, gaps, factor = contrasts[row["contrast_id"]]
        seed = int(row["seed"])
        compare(row, {"candidate_nmse": candidate[seed], "comparator_nmse": comparator[seed], "comparator_factor": factor, "margin_gap": gaps[seed]}, "replicate_contrasts")
    stability = []
    for row in table("stability"):
        key = by_id[row["task_id"]]
        c, b = (metrics[key + (m,)]["max_abs_error"] for m in ("kahm_departure", "ridge"))
        ratio = c / b
        compare(row, {"candidate_max_abs_error": c, "ridge_max_abs_error": b, "max_error_ratio": ratio, "max_error_pass": ratio <= 5}, "stability")
        stability.append(ratio)
    result = {"status": "passed", "commit": "7dda356138675934aab5e327ec667bde0a5eda79",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "scope": "Separately written numeric audit. No study fitting, analysis, or verifier module imported; no model pickle loaded; no model fitted. Candidate-selection and resources/bounds are recorded-score or recorded-diagnostic consistency checks, not fitting or timing repetition.",
        "counts": dict(sorted(COUNTS.items())), "maximum_absolute_numeric_differences": dict(sorted(MAXDIFF.items())),
        "primary_contrasts": primary, "established_contrasts": [x["contrast_id"] for x in primary if x["criterion_established"]],
        "maximum_error_ratio": max(stability), "policy_selected_task_counts": policy_counts,
        "gamma_counts": gamma_counts, "active_horizon_counts": active_counts,
        "resource_means_from_recorded_measurements": {m: {k: avg(v) for k, v in values.items()} for m, values in measured.items()},
        "distinct_base_generation_seeds": len(generation_seeds),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "counts", "established_contrasts", "maximum_error_ratio")}, indent=2))


if __name__ == "__main__":
    main()
