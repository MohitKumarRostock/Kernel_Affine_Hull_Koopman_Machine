#!/usr/bin/env python3
"""Analyze sealed fresh-cohort optional-feature policies without selecting on D.

The four primary claims use paired repetition-level standardized losses and
one-sided Bonferroni-adjusted Student t bounds. Conditional selection summaries
are descriptive, including when conditioning creates very small subsets.
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
from scipy.stats import t

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
PANELS = {"nonlinear": LAWS[1:], "linear": LAWS[:1], "all_descriptive": LAWS}
P0 = ("legacy__ridge", "legacy__raw_rbf", "legacy__raw_hgb", "legacy__persistence",
      "legacy__common", "legacy__rbf_departure", "legacy__rbf_unbounded",
      "crossfit__shared_bias", "crossfit__crossfit_rbf", "crossfit__insample_rbf")
ORDERS = {"P0": P0, "P1": P0 + ("crossfit__insample_kahm",),
          "P0plus": P0 + ("extra__extra_rbf",)}
POLICIES = tuple(ORDERS)
N = 24
FAMILY_ALPHA = 0.05
FAMILY_SIZE = 4
CLAIMS = (("P1_vs_P0_nonlinear", "nonlinear", "P0", .97, "3% superiority"),
          ("P1_vs_P0plus_nonlinear", "nonlinear", "P0plus", .97, "3% superiority"),
          ("P1_vs_P0_linear", "linear", "P0", 1.05, "5% noninferiority"),
          ("P1_vs_P0plus_linear", "linear", "P0plus", 1.05, "5% noninferiority"))


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def csv_out(path, rows):
    require(bool(rows), "Refusing empty table: " + str(path))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_key(identity, mode="calibrated"):
    family, method = identity.split("__", 1)
    actual = "gamma1" if family == "legacy" and mode == "eta1" else mode
    return f"{family}__{actual}__{method}"


def expected_keys():
    legacy = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "kahm_departure",
              "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy")
    residual = ("common", "shared_bias", "half_common", "half_shared_bias", "crossfit_kahm",
                "crossfit_rbf", "crossfit_matched_rbf", "insample_kahm", "insample_rbf", "insample_matched_rbf")
    keys = {f"legacy__{mode}__{method}" for mode in ("calibrated", "gamma1") for method in legacy}
    keys |= {f"crossfit__{mode}__{method}" for mode in ("calibrated", "eta1") for method in residual}
    keys |= {f"extra__{mode}__extra_rbf" for mode in ("calibrated", "eta1")}
    keys |= {f"policy__{mode}__{name}" for mode in ("calibrated", "eta1")
             for name in (*POLICIES, "expanded_nonkahm")}
    require(len(keys) == 50, "Internal forecast roster")
    return keys


def validate_config(c):
    require(c["master_seed"] == 2026110191, "Not the fresh declared cohort")
    require(c["repetitions"] == N and c["task_count"] == 192, "Incomplete cohort")
    require(tuple(c["laws"]) == LAWS and c["horizons"] == [1, 4, 12], "Law/horizon mismatch")
    require(c["noise_levels"] == [0.0, 0.1] and c["histories"] == [12], "Noise/history mismatch")
    require(c["analysis"]["family_alpha"] == FAMILY_ALPHA and
            c["analysis"]["primary_family_size"] == FAMILY_SIZE, "Primary family mismatch")
    require(c["analysis"]["superiority_margin"] == .03 and
            c["analysis"]["linear_noninferiority_margin"] == .05, "Margins mismatch")
    require(c["analysis"]["independent_unit"] == "repetition", "Incorrect inferential unit")


def verify_release(root):
    release = load(root / "TARGET_RELEASE.json")
    require(release["status"] == "released", "D targets are not released")
    for field, filename in (("fit_lock_sha256", "FIT_LOCK.json"),
                            ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                            ("d_manifest_sha256", "prepared/d_manifest.json")):
        require(release[field] == sha(root / filename), "Release link mismatch: " + filename)
    lock = load(root / "PREDICTION_LOCK.json")
    require(lock["task_count"] == 192 and lock["status"] == "sealed", "Incomplete prediction seal")
    require(lock["config_sha256"] == sha(root / "config.json") and
            lock["freeze_sha256"] == sha(root / "freeze.json"), "Prediction seal changed")
    return lock


def flatten_numeric(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for name, child in value.items():
            output.update(flatten_numeric(child, prefix + ("/" if prefix else "") + str(name)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output


def analyze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    c = load(root / "config.json")
    validate_config(c)
    prediction_lock = verify_release(root)
    a = load(root / "prepared/abc_manifest.json")
    d = load(root / "prepared/d_manifest.json")
    abc = {r["task_id"]: r for r in a["tasks"]}
    ds = {r["task_id"]: r for r in d["tasks"]}
    require(len(abc) == len(ds) == 192 and set(abc) == set(ds), "Task manifest mismatch")
    factorial = {(r["law"], r["noise_index"], r["repetition"]) for r in ds.values()}
    require(factorial == {(law, noise, rep) for law in LAWS for noise in (0, 1) for rep in range(N)},
            "Missing or duplicated factorial cells")
    require(not output.exists(), "Analysis output already exists")
    output.mkdir(parents=True)
    task_rows, horizon_rows, selection_rows, resource_rows, width_rows = [], [], [], [], []
    keys = sorted(expected_keys())
    for task_id in sorted(ds):
        meta, initial = ds[task_id], abc[task_id]
        pred_path = root / "predictions" / task_id / "predictions.npz"
        require(sha(pred_path) == prediction_lock["files"][str(pred_path.relative_to(root))], "Changed D forecasts")
        require(sha(root / meta["target"]) == meta["target_sha256"], "Changed D targets")
        require(sha(root / initial["input"]) == initial["sha256"], "Changed A/B/C arrays")
        with np.load(root / meta["target"], allow_pickle=False) as z:
            target = z["D_Y"]
        with np.load(root / initial["input"], allow_pickle=False) as z:
            scale = np.maximum(np.var(z["A_Y"], axis=0, ddof=0), 1e-12)
            require(np.array_equal(scale, z["loss_scale"]), "Incorrect training normalization")
        require(target.shape == (1664, 3) and np.isfinite(target).all(), "Invalid D targets")
        basic = {"task_id": task_id, "repetition": meta["repetition"], "law": meta["law"],
                 "noise_index": meta["noise_index"], "noise_level": meta["noise_level"]}
        values = {}
        with np.load(pred_path, allow_pickle=False) as z:
            require(set(z.files) == set(keys), "Incomplete 50-array D roster")
            for key in keys:
                prediction = z[key]
                require(prediction.shape == target.shape and np.isfinite(prediction).all(), "Invalid prediction")
                mse = np.mean(np.square(target - prediction), axis=0)
                smse = mse / scale
                values[key] = float(smse.mean())
                task_rows.append({**basic, "forecast_key": key, "mse": float(mse.mean()), "smse": values[key]})
                for hi, horizon in enumerate(c["horizons"]):
                    horizon_rows.append({**basic, "forecast_key": key, "horizon": horizon,
                                         "mse": float(mse[hi]), "smse": float(smse[hi]),
                                         "A_target_variance": float(scale[hi]), "D_rows": len(target)})
        summary = load(root / "results" / task_id / "fit_summary.json")
        require(set(summary["policies"]) == set(POLICIES), "Policy summaries missing")
        for name in POLICIES:
            record = summary["policies"][name]
            identity = record["selected_identity"]
            require(identity in ORDERS[name], "Ineligible policy choice")
            choice = next(x for x in record["candidates"] if x["identity"] == identity)
            score = values[f"policy__calibrated__{name}"]
            comparator = values["policy__calibrated__P0"]
            selection_rows.append({**basic, "policy": name, "selected_identity": identity,
                                   "C_selected_smse": float(choice["score"]), "D_smse": score,
                                   "P0_D_smse": comparator, "D_difference_from_P0": score - comparator,
                                   "D_win_vs_P0": bool(score < comparator),
                                   "D_equal_to_P0": bool(score == comparator),
                                   "conditioning": "C selection only; D subgroup summaries are descriptive"})
        for measure, value in flatten_numeric(summary["resources"]).items():
            resource_rows.append({**basic, "resource": measure, "value": value})
        prediction_receipt = load(root / "predictions" / task_id / "prediction_receipt.json")
        for measure in ("wall_seconds", "cpu_seconds"):
            resource_rows.append({**basic, "resource": "prediction/" + measure,
                                  "value": float(prediction_receipt[measure])})
        extra = summary.get("extra", {})
        for measure, value in flatten_numeric(extra.get("diagnostics", {})).items():
            if any(word in measure.lower() for word in ("width", "feature", "landmark", "candidate", "score_count")):
                width_rows.append({**basic, "measure": measure, "value": value})
    # Each horizon is averaged within task; law/noise cells within repetition.
    repetition_rows = []
    for panel, laws in PANELS.items():
        groups = defaultdict(list)
        for row in task_rows:
            if row["law"] in laws:
                groups[(row["repetition"], row["forecast_key"])].append(row)
        for (rep, key), members in sorted(groups.items()):
            require(len(members) == 2 * len(laws), "Incomplete repetition aggregate")
            repetition_rows.append({"panel": panel, "repetition": rep, "forecast_key": key,
                                    "condition_count": len(members),
                                    "mse": float(np.mean([m["mse"] for m in members])),
                                    "smse": float(np.mean([m["smse"] for m in members]))})
    index = {(r["panel"], r["forecast_key"], r["repetition"]): r["smse"] for r in repetition_rows}
    def vector(panel, key):
        return np.array([index[panel, key, rep] for rep in range(N)])
    panels = []
    for panel in PANELS:
        for key in keys:
            values = vector(panel, key)
            panels.append({"panel": panel, "forecast_key": key, "mean_smse": float(values.mean()),
                           "repetition_sd": float(values.std(ddof=1)), "n_repetitions": N})
    primary, paired = [], []
    critical = float(t.ppf(1 - FAMILY_ALPHA / FAMILY_SIZE, N - 1))
    for identifier, panel, other, factor, margin in CLAIMS:
        candidate = vector(panel, "policy__calibrated__P1")
        comparator = vector(panel, f"policy__calibrated__{other}")
        difference = candidate - factor * comparator
        mean = float(difference.mean())
        sd = float(difference.std(ddof=1)); se = sd / np.sqrt(N)
        upper = mean + critical * se
        pvalue = float(t.cdf(mean / se, N - 1)) if se else (0. if mean < 0 else 1. if mean > 0 else .5)
        primary.append({"contrast_id": identifier, "panel": panel, "candidate": "P1", "comparator": other,
                        "margin": margin, "comparator_factor": factor,
                        "candidate_mean_smse": float(candidate.mean()), "comparator_mean_smse": float(comparator.mean()),
                        "point_relative_gain_percent": float(100 * (1 - candidate.mean() / comparator.mean())),
                        "margin_difference_mean": mean, "margin_difference_sd": sd,
                        "margin_difference_se": float(se), "critical_t": critical,
                        "simultaneous_one_sided_upper": float(upper), "criterion_established": bool(upper < 0),
                        "p_one_sided": pvalue, "p_bonferroni": min(1., FAMILY_SIZE * pvalue),
                        "family_alpha": FAMILY_ALPHA, "family_size": FAMILY_SIZE, "n_repetitions": N, "df": N-1,
                        "strict_D_win_repetitions": int(np.count_nonzero(candidate < comparator))})
        for rep in range(N):
            paired.append({"contrast_id": identifier, "repetition": rep, "candidate_smse": float(candidate[rep]),
                           "comparator_smse": float(comparator[rep]), "margin_difference": float(difference[rep])})
    frequencies, conditional = [], []
    for panel, laws in PANELS.items():
        for policy, order in ORDERS.items():
            rows = [r for r in selection_rows if r["policy"] == policy and r["law"] in laws]
            counts = Counter(r["selected_identity"] for r in rows)
            for identity in order:
                frequencies.append({"panel": panel, "policy": policy, "selected_identity": identity,
                                    "task_count": counts[identity], "total_tasks": len(rows),
                                    "fraction": counts[identity] / len(rows)})
                members = [r for r in rows if r["selected_identity"] == identity]
                if members:
                    conditional.append({"panel": panel, "policy": policy, "selected_identity": identity,
                                        "task_count": len(members),
                                        "represented_repetitions": len({r["repetition"] for r in members}),
                                        "mean_D_smse": float(np.mean([r["D_smse"] for r in members])),
                                        "mean_P0_D_smse": float(np.mean([r["P0_D_smse"] for r in members])),
                                        "strict_D_win_count": sum(r["D_win_vs_P0"] for r in members),
                                        "equal_D_count": sum(r["D_equal_to_P0"] for r in members),
                                        "inference": "descriptive selection-conditioned subset; no confidence claim"})
    resource_summary = []
    groups = defaultdict(list)
    for row in resource_rows:
        groups[row["resource"]].append(row["value"])
    for name, values in sorted(groups.items()):
        require(len(values) == 192, "Resource missing for some task: " + name)
        resource_summary.append({"resource": name, "n_tasks": len(values), "sum": float(sum(values)),
                                 "mean": float(np.mean(values)), "median": float(np.median(values)),
                                 "minimum": float(min(values)), "maximum": float(max(values)),
                                 "scope": "Measured task-component cost; shared search is not an isolated marginal policy cost"})
    tables = {"task_losses.csv": task_rows, "horizon_losses.csv": horizon_rows,
              "repetition_losses.csv": repetition_rows, "panel_summary.csv": panels,
              "primary_results.csv": primary, "primary_paired_repetitions.csv": paired,
              "selection_by_task.csv": selection_rows, "selection_frequencies.csv": frequencies,
              "conditional_descriptive.csv": conditional, "resource_by_task.csv": resource_rows,
              "resource_summary.csv": resource_summary}
    if width_rows:
        tables["extra_widths_by_task.csv"] = width_rows
    for name, rows in tables.items():
        csv_out(output / name, rows)
    result = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "study_root": str(root), "config_sha256": sha(root / "config.json"),
              "freeze_sha256": sha(root / "freeze.json"), "prediction_lock_sha256": sha(root / "PREDICTION_LOCK.json"),
              "target_release_sha256": sha(root / "TARGET_RELEASE.json"), "analyzer_sha256": sha(__file__),
              "tasks": 192, "arrays_per_task": len(keys), "forecast_arrays": 192 * len(keys),
              "independent_repetitions": N, "primary_family_size": FAMILY_SIZE,
              "primary_results": primary, "criteria_established": sum(r["criterion_established"] for r in primary),
              "panel_summary": [r for r in panels if r["forecast_key"] in
                                 [f"policy__calibrated__{name}" for name in POLICIES]],
              "limitations": ["Synthetic laws and fixed finite tuning menus; no natural-data claim.",
                              "Student-t bounds use 24 independent simulation repetition aggregates and their approximation assumptions.",
                              "Equal marginal B score count does not establish equal wall time or equal model complexity.",
                              "Nested candidate menus guarantee nonincrease on C only; they do not guarantee D improvement.",
                              "Selection-conditioned results are descriptive and cannot replace failed primary criteria.",
                              "Resource tables retain full measured experiment costs; reuse does not identify every isolated policy cost."],
              "tables": {name: {"rows": len(rows), "sha256": sha(output / name)} for name, rows in tables.items()}}
    write_json(output / "analysis_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "analysis"
    result = analyze(args.root, output)
    print(json.dumps({"status": result["status"], "tasks": result["tasks"],
                      "criteria_established": result["criteria_established"], "output": str(output)}))


if __name__ == "__main__":
    main()
