"""Build publication displays from the COMPLETE reconstruction-mechanism study.

Reporting only: no model imports, model fitting, tuning or new forecasts. The
builder refuses incomplete or technically blocked evidence and checks its
displayed contrasts against the saved independent-repetition aggregates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kahm_reconstruction_mechanism_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "extensions/reconstruction_mechanism"
OUTPUT = Path(__file__).resolve().parent / "manuscript/generated/reconstruction_mechanism"
METHODS = (
    ("ridge", "Raw ridge"), ("raw_rbf", "Raw RBF"),
    ("raw_hgb", "Boosted trees"), ("persistence", "Persistence blend"),
    ("common", "Common basis"), ("kahm_departure", "KAHM departure (bounded)"),
    ("rbf_departure", "RBF departure (bounded)"),
    ("kahm_unbounded", "KAHM departure (unbounded)"),
    ("rbf_unbounded", "RBF departure (unbounded)"),
    ("nonkahm_policy", "C-selected non-KAHM policy"),
)
LABELS = dict(METHODS)
PRIMARY = ("common", "rbf_departure", "raw_rbf", "raw_hgb", "nonkahm_policy")
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
NONLINEAR = ("nonlinear_ar", "vanderpol", "duffing")
LAW_LABELS = {"linear_ar": "Linear AR", "nonlinear_ar": "Nonlinear AR",
              "vanderpol": "Van der Pol", "duffing": "Duffing"}
NOISE = (0.0, 0.1)
SEEDS = tuple(range(946100, 946124))
HORIZONS = (1, 4, 12)


def description(path: Path) -> dict:
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def read_bool(value) -> bool:
    if value in (True, "True", "true"):
        return True
    if value in (False, "False", "false"):
        return False
    raise ValueError(f"Invalid Boolean display field: {value!r}")


def close(actual, expected, label, *, rtol=1e-11, atol=1e-12):
    if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
        raise ValueError(f"Display check failed for {label}: {actual} != {expected}")


def number(value, decimals=5) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite display value")
    return f"{value:.{decimals}f}"


def tex_integer(value) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def compact(value) -> str:
    value = float(value)
    if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e4):
        mantissa, exponent = f"{value:.3e}".split("e")
        return rf"\ensuremath{{{mantissa}\times10^{{{int(exponent)}}}}}"
    return number(value)


def tabular(columns, header, rows):
    return "\n".join([
        "% Generated from the complete saved reconstruction-mechanism analysis.",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
        *(" & ".join(map(str, row)) + r" \\" for row in rows),
        r"\bottomrule", r"\end{tabular}", "",
    ])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(study: Path, output: Path) -> dict:
    study, output = study.resolve(), output.resolve()
    inputs = {}

    def record(relative):
        path = study / relative
        inputs[relative] = description(path)
        return path

    def load_json(relative):
        return json.loads(record(relative).read_text())

    def load_csv(relative):
        with record(relative).open(newline="") as stream:
            return list(csv.DictReader(stream))

    decision = load_json("analysis/decision.json")
    if (not decision["technical_complete"] or decision["completed_valid_tasks"] != 192
            or decision["independent_repetitions"] != 24 or decision["task_errors"]):
        raise ValueError("Complete, technically valid evidence for all 192 tasks is required")
    if decision["decision"] not in ("STOP_FORECASTING_PERFORMANCE_BRANCH", "ADVANCE_TO_APPLICATION"):
        raise ValueError("A completed fixed decision is required")
    execution = load_json("results/execution.json")
    if "completed_utc" not in execution or len(execution["events"]) != 192:
        raise ValueError("Execution record must confirm all 192 reserved tasks finished")
    if any(event["returncode"] != 0 for event in execution["events"]):
        raise ValueError("A failed reserved task cannot be hidden by a display builder")
    if execution["public_preregistration"] is not False:
        raise ValueError("This study used a local freeze, not public preregistration")
    config = load_json("config.json")
    freeze = load_json("freeze.json")
    record("PROTOCOL.md")
    record("ALGEBRA.md")
    if execution["freeze_sha256"] != inputs["freeze.json"]["sha256"]:
        raise ValueError("Execution identity differs from the recorded local freeze")
    if (config["seeds"] != list(SEEDS) or config["laws"] != list(LAWS)
            or config["methods"] != [name for name, _ in METHODS]):
        raise ValueError("Scientific roster differs from the declared reporting scope")
    validation = load_json("analysis/task_validation.json")
    if validation["valid_tasks"] != 192 or validation["errors"]:
        raise ValueError("The saved independent analysis did not validate the full roster")
    analysis_manifest = load_json("analysis/analysis_manifest.json")
    expected_hashes = {row["path"]: row["sha256"] for row in analysis_manifest["output_files"]}
    for relative in ("analysis/decision.json", "analysis/task_validation.json"):
        if expected_hashes.get(relative) != inputs[relative]["sha256"]:
            raise ValueError(f"Analysis output has changed: {relative}")

    datasets = {}
    expected_rows = {
        "task_metrics": 1920, "horizon_metrics": 5760,
        "condition_metrics": 80, "condition_horizon_metrics": 240,
        "replicate_metrics": 720, "primary_contrasts": 6,
        "replicate_contrasts": 144, "overview_metrics": 30,
        "activation": 5760, "feature_bounds": 576, "readout_bounds": 1728,
        "stability": 192,
    }
    for name in (*expected_rows, "resources", "safety_diagnostics"):
        relative = f"analysis/{name}.csv"
        datasets[name] = load_csv(relative)
        if expected_hashes.get(relative) != inputs[relative]["sha256"]:
            raise ValueError(f"Analysis output has changed: {relative}")
        if name in expected_rows and len(datasets[name]) != expected_rows[name]:
            raise ValueError(f"Unexpected complete-table length: {name}")

    task = {(row["task_id"], row["method"]): row for row in datasets["task_metrics"]}
    condition = {(row["law"], float(row["noise_level"]), row["method"]): row
                 for row in datasets["condition_metrics"]}
    replicate = {(row["scope"], int(row["seed"]), row["method"]): row
                 for row in datasets["replicate_metrics"]}
    overview = {(row["scope"], row["method"]): row for row in datasets["overview_metrics"]}
    if len(task) != 1920 or len(condition) != 80 or len(replicate) != 720 or len(overview) != 30:
        raise ValueError("Repeated rows in complete display inputs")
    task_ids = sorted({identifier for identifier, _ in task})
    for identifier in task_ids:
        if {method for task_id, method in task if task_id == identifier} != set(LABELS):
            raise ValueError("Incomplete task-method roster")
    for (law, noise, method), row in condition.items():
        chosen = [item for item in task.values() if item["law"] == law
                  and float(item["noise_level"]) == noise and item["method"] == method]
        if {int(item["seed"]) for item in chosen} != set(SEEDS) or len(chosen) != 24:
            raise ValueError("Condition is missing independent repetitions")
        close(row["mean_nmse"], mean(float(item["nmse"]) for item in chosen), (law, noise, method))
    for (scope, seed, method), row in replicate.items():
        laws = NONLINEAR if scope == "nonlinear" else ("linear_ar",) if scope == "linear" else LAWS
        chosen = [item for item in task.values() if item["law"] in laws
                  and int(item["seed"]) == seed and item["method"] == method]
        if len(chosen) != len(laws) * 2:
            raise ValueError("Independent repetition aggregate is incomplete")
        close(row["nmse"], mean(float(item["nmse"]) for item in chosen), (scope, seed, method))
    for (scope, method), row in overview.items():
        close(row["mean_nmse"], mean(float(replicate[scope, seed, method]["nmse"]) for seed in SEEDS),
              (scope, method))

    primary = datasets["primary_contrasts"]
    expected_ids = [f"nonlinear_vs_{method}" for method in PRIMARY] + ["linear_noninferiority_vs_ridge"]
    if [row["contrast_id"] for row in primary] != expected_ids:
        raise ValueError("The complete six-contrast family/order is required")
    critical = float(student_t.ppf(1 - .05 / 6, 23))
    for row in primary:
        scope, method = row["scope"], row["comparator"]
        factor = .97 if scope == "nonlinear" else 1.05
        candidate = [float(replicate[scope, seed, "kahm_departure"]["nmse"]) for seed in SEEDS]
        comparator = [float(replicate[scope, seed, method]["nmse"]) for seed in SEEDS]
        gaps = [factor * c - k for c, k in zip(comparator, candidate)]
        lower = mean(gaps) - critical * stdev(gaps) / math.sqrt(24)
        close(row["candidate_mean_nmse"], mean(candidate), row["contrast_id"])
        close(row["comparator_mean_nmse"], mean(comparator), row["contrast_id"])
        close(row["margin_gap_mean"], mean(gaps), row["contrast_id"])
        close(row["simultaneous_one_sided_lower"], lower, row["contrast_id"])
        close(row["point_relative_gain"], 1 - mean(candidate) / mean(comparator), row["contrast_id"])
        if read_bool(row["criterion_established"]) != (lower > 0):
            raise ValueError("Displayed statistical criterion disagrees with recomputation")
    statistical = all(read_bool(row["criterion_established"]) for row in primary)
    stable = all(read_bool(row[field]) for row in datasets["stability"]
                 for field in ("max_error_pass", "feature_bounds_pass", "readout_bounds_pass"))
    expected_decision = "ADVANCE_TO_APPLICATION" if statistical and stable else "STOP_FORECASTING_PERFORMANCE_BRANCH"
    if decision["decision"] != expected_decision:
        raise ValueError("Reported fixed decision does not follow its declared conditions")

    output.mkdir(parents=True, exist_ok=True)
    contents = {}
    primary_rows = []
    for row in primary:
        name = LABELS[row["comparator"]]
        if row["scope"] == "linear":
            name = "Linear control: ridge"
        primary_rows.append([
            name, number(row["candidate_mean_nmse"]), number(row["comparator_mean_nmse"]),
            number(100 * float(row["point_relative_gain"]), 2), compact(row["margin_gap_mean"]),
            compact(row["simultaneous_one_sided_lower"]),
            "Yes" if read_bool(row["criterion_established"]) else "No",
        ])
    contents["primary_table.tex"] = tabular(
        "@{}lrrrrrc@{}", ["Comparator", r"$L_K$", r"$L_m$", "Gain (\\%)",
                           r"$\bar d$", "Lower bound", "Pass"], primary_rows)

    conditions = [(law, noise) for law in (*NONLINEAR, "linear_ar") for noise in NOISE]
    contents["law_noise_table.tex"] = tabular(
        "@{}lrrrrrrrr@{}",
        ["Procedure", r"\multicolumn{2}{c}{Nonlinear AR}", r"\multicolumn{2}{c}{Van der Pol}",
         r"\multicolumn{2}{c}{Duffing}", r"\multicolumn{2}{c}{Linear AR}"],
        [[r"Added noise $\eta$", *["0" if noise == 0 else "0.1" for _, noise in conditions]],
         *[[label, *[number(condition[law, noise, method]["mean_nmse"], 4)
                      for law, noise in conditions]] for method, label in METHODS]],
    )

    resource_lookup = {(row["task_id"], row["resource"]): row["value"] for row in datasets["resources"]}
    resource_fields = {
        "fit_wall_seconds": "fit.per_method.{method}.fit_and_selection.wall_seconds",
        "fit_cpu_seconds": "fit.per_method.{method}.fit_and_selection.cpu_seconds",
        "predict_wall_seconds": "prediction.per_method.{method}.predict.wall_seconds",
        "predict_cpu_seconds": "prediction.per_method.{method}.predict.cpu_seconds",
        "pickle_bytes": "fit.per_method.{method}.state.pickle_bytes",
        "array_buffer_bytes": "fit.per_method.{method}.state.array_buffer_bytes",
    }
    resources = []
    for method, label in METHODS:
        row = {"method": method, "tasks": 192}
        for field, key in resource_fields.items():
            values = [float(resource_lookup[identifier, key.format(method=method)]) for identifier in task_ids]
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError("Invalid measured resource value")
            row[f"mean_{field}"] = mean(values)
            row[f"median_{field}"] = float(np.median(values))
            row[f"max_{field}"] = max(values)
        resources.append(row)
    resource = {row["method"]: row for row in resources}
    contents["resource_table.tex"] = tabular(
        "@{}lrrrrr@{}", ["Procedure", r"\shortstack{Fit/select\\wall (s)}",
                          r"\shortstack{Fit/select\\CPU (s)}", r"\shortstack{D predict\\wall (ms)}",
                          r"\shortstack{State\\(KiB)}", r"\shortstack{Arrays\\(KiB)}"],
        [[label, number(resource[method]["mean_fit_wall_seconds"], 4),
          number(resource[method]["mean_fit_cpu_seconds"], 4),
          number(1000 * resource[method]["mean_predict_wall_seconds"], 3),
          number(resource[method]["mean_pickle_bytes"] / 1024, 2),
          number(resource[method]["mean_array_buffer_bytes"] / 1024, 2)] for method, label in METHODS],
    )

    bound_summary = []
    for method in ("common", "kahm_departure", "rbf_departure"):
        frows = [row for row in datasets["feature_bounds"] if row["block"] == method]
        rrows = [row for row in datasets["readout_bounds"] if row["method"] == method]
        if len(frows) != 192 or len(rrows) != 576:
            raise ValueError("Incomplete bounded-procedure diagnostics")
        def ratio(numerator, denominator):
            if denominator == 0:
                if numerator != 0:
                    raise ValueError("Positive value with a zero theoretical bound")
                return 0.
            return numerator / denominator
        row = {
            "method": method, "feature_checks": len(frows), "horizon_checks": len(rrows),
            "max_feature_norm_ratio": max(float(r["max_norm"]) / float(r["radius"]) for r in frows),
            "max_coefficient_bound_ratio": max(ratio(float(r["coefficient_norm"]),
                                                     float(r["coefficient_norm_bound"])) for r in rrows),
            "max_correction_bound_ratio": max(ratio(float(r["max_abs_correction"]),
                                                    float(r["correction_bound"])) for r in rrows),
            "max_abs_correction": max(float(r["max_abs_correction"]) for r in rrows),
            "feature_violations": sum(not read_bool(r["bound_pass"]) for r in frows),
            "coefficient_violations": sum(not read_bool(r["coefficient_bound_pass"]) for r in rrows),
            "correction_violations": sum(not read_bool(r["correction_bound_pass"]) for r in rrows),
        }
        bound_summary.append(row)
    contents["bounds_table.tex"] = tabular(
        "@{}lrrrr@{}", ["Bounded procedure", r"\shortstack{Feature norm\\/ radius}",
                         r"\shortstack{Coefficient norm\\/ bound}",
                         r"\shortstack{Correction\\/ bound}", "Violations"],
        [[LABELS[row["method"]], number(row["max_feature_norm_ratio"], 5),
          number(row["max_coefficient_bound_ratio"], 5), number(row["max_correction_bound_ratio"], 5),
          str(row["feature_violations"] + row["coefficient_violations"] + row["correction_violations"])]
         for row in bound_summary],
    )

    activation_summary = []
    for method, label in METHODS:
        for scope, laws in (("nonlinear", NONLINEAR), ("linear", ("linear_ar",))):
            rows = [r for r in datasets["activation"] if r["method"] == method and r["law"] in laws]
            values = [float(r["gamma"]) if r["gamma"] != "" else 0.
                      for r in rows if r["gamma"] != "" or method == "nonkahm_policy"]
            if any(r["gamma"] == "" and method == "nonkahm_policy"
                   and r["selected_policy_method"] != "ridge" for r in rows):
                raise ValueError("A policy gamma is missing for a selected non-ridge method")
            counts = {str(gamma): sum(value == gamma for value in values)
                      for gamma in (0., .25, .5, 1.)}
            activation_summary.append({"method": method, "scope": scope, "horizon_cells": len(rows),
                                       "applicable_cells": len(values),
                                       "active_cells": sum(value > 0 for value in values),
                                       **{f"gamma_{k}": v for k, v in counts.items()}})
    activation = {(row["scope"], row["method"]): row for row in activation_summary}
    contents["activation_table.tex"] = tabular(
        "@{}lrrrrrr@{}", ["Procedure", r"\shortstack{Nonlinear\\active / 432}",
                          r"\shortstack{Linear\\active / 144}",
                          r"$\gamma=0$", r"$\gamma=1/4$", r"$\gamma=1/2$", r"$\gamma=1$"],
        [[label, "---" if method == "ridge" else str(activation["nonlinear", method]["active_cells"]),
          "---" if method == "ridge" else str(activation["linear", method]["active_cells"]),
          *["---" if method == "ridge" else str(sum(activation[scope, method][f"gamma_{gamma}"]
                                                     for scope in ("nonlinear", "linear")))
            for gamma in ("0.0", "0.25", "0.5", "1.0")]] for method, label in METHODS],
    )
    policy_rows = [row for row in datasets["task_metrics"] if row["method"] == "nonkahm_policy"]
    policy_summary = []
    for method, label in METHODS:
        if method in ("kahm_departure", "kahm_unbounded", "nonkahm_policy"):
            continue
        counts = [sum(r["selected_policy_method"] == method and r["law"] in laws for r in policy_rows)
                  for laws in (NONLINEAR, ("linear_ar",))]
        policy_summary.append({"method": method, "nonlinear_tasks": counts[0], "linear_tasks": counts[1]})
    contents["policy_table.tex"] = tabular(
        "@{}lrr@{}", ["C-selected complete procedure", "Nonlinear / 144", "Linear / 48"],
        [[LABELS[row["method"]], row["nonlinear_tasks"], row["linear_tasks"]] for row in policy_summary],
    )

    fallback = {}
    for kind in ("kahm", "rbf"):
        for split in ("A", "B", "C", "D"):
            stage = "prediction" if split == "D" else "fit"
            key = f"{kind}_fallback_count" if split == "D" else f"features_A_B_C.{split}.{kind}_fallback_count"
            selected = [row for row in datasets["safety_diagnostics"]
                        if row["stage"] == stage and row["diagnostic"] == key]
            if len(selected) != 192:
                raise ValueError(f"Missing full-roster local fallback accounting: {kind}/{split}")
            fallback[kind, split] = sum(int(row["value"]) for row in selected)
    feature_violations = sum(row["feature_violations"] for row in bound_summary)
    coefficient_violations = sum(row["coefficient_violations"] for row in bound_summary)
    correction_violations = sum(row["correction_violations"] for row in bound_summary)
    ratio_max = max(float(row["max_error_ratio"]) for row in datasets["stability"])
    worst = max(datasets["stability"], key=lambda row: float(row["max_error_ratio"]))
    useful_passes = sum(read_bool(row["criterion_established"]) for row in primary[:5])
    linear_pass = read_bool(primary[-1]["criterion_established"])
    numbers = {
        "TaskCount": "192", "RepeatCount": "24", "MethodCount": "10", "RecordCount": tex_integer(1920),
        "TargetCells": tex_integer(958464), "UsefulPassCount": str(useful_passes),
        "ContrastPassCount": str(useful_passes + int(linear_pass)),
        "KNonlinearNMSE": number(overview["nonlinear", "kahm_departure"]["mean_nmse"]),
        "KLinearNMSE": number(overview["linear", "kahm_departure"]["mean_nmse"]),
        "MaxErrorRatio": number(ratio_max, 3), "FeatureViolations": str(feature_violations),
        "CoefficientViolations": str(coefficient_violations), "CorrectionViolations": str(correction_violations),
        "StabilityFailures": str(len(decision["failed_stability_tasks"])),
        "KActiveNonlinear": str(activation["nonlinear", "kahm_departure"]["active_cells"]),
        "KActiveLinear": str(activation["linear", "kahm_departure"]["active_cells"]),
        "KFallbackD": tex_integer(fallback["kahm", "D"]),
        "RFallbackD": tex_integer(fallback["rbf", "D"]),
        "KFallbackABC": tex_integer(sum(fallback["kahm", split] for split in ("A", "B", "C"))),
        "PeakProcessMiB": number(max(float(resource_lookup[identifier, "process_cumulative_peak_rss_kib"])
                                     for identifier in task_ids) / 1024, 1),
        "FreezeShort": execution["local_freeze_commit"][:12],
        "KFitSeconds": number(resource["kahm_departure"]["mean_fit_wall_seconds"], 4),
        "RFitSeconds": number(resource["rbf_departure"]["mean_fit_wall_seconds"], 4),
        "RawRBFFitSeconds": number(resource["raw_rbf"]["mean_fit_wall_seconds"], 4),
        "KPredictionMilliseconds": number(1000 * resource["kahm_departure"]["mean_predict_wall_seconds"], 3),
        "RPredictionMilliseconds": number(1000 * resource["rbf_departure"]["mean_predict_wall_seconds"], 3),
        "KStateKiB": number(resource["kahm_departure"]["mean_pickle_bytes"] / 1024, 2),
    }
    for row in primary[:5]:
        suffix = {"common": "Common", "rbf_departure": "RBFDeparture", "raw_rbf": "RawRBF",
                  "raw_hgb": "Boosting", "nonkahm_policy": "Policy"}[row["comparator"]]
        numbers[f"Kvs{suffix}Gain"] = number(100 * float(row["point_relative_gain"]), 2)
        numbers[f"Kvs{suffix}Lower"] = compact(row["simultaneous_one_sided_lower"])
    numbers["LinearLower"] = compact(primary[-1]["simultaneous_one_sided_lower"])
    numbers["LinearNIPassText"] = "passes" if linear_pass else "does not pass"
    numbers["DecisionText"] = ("the overall improvement criterion is satisfied" if expected_decision == "ADVANCE_TO_APPLICATION"
                               else "the overall improvement criterion is not satisfied")
    contents["numbers.tex"] = "% Complete saved-result macros; generated without fitting models.\n" + "\n".join(
        rf"\providecommand{{\RM{key}}}{{{value}}}" for key, value in numbers.items()) + "\n"

    summary = [
        r"The bounded KAHM departure attains nonlinear aggregate normalized mean squared error $\RMKNonlinearNMSE{}$.",
        r"Its point relative gains over the common basis, matched bounded RBF departure, raw RBF, boosted trees, and the C-selected non-KAHM procedure are",
        r"$\RMKvsCommonGain{}\%$, $\RMKvsRBFDepartureGain{}\%$, $\RMKvsRawRBFGain{}\%$, $\RMKvsBoostingGain{}\%$, and $\RMKvsPolicyGain{}\%$, respectively.",
        r"Positive gains denote smaller mean KAHM loss. These are point estimates; the simultaneous margin bounds are reported in Table~\ref{tab:mechanism-primary}.",
        rf"Of the five nonlinear improvement criteria, {useful_passes} is satisfied; the linear-control noninferiority criterion {numbers['LinearNIPassText']}.",
    ]
    summary.append(r"All feature-, coefficient-, correction-, and maximum-error checks are satisfied."
                   if stable else
                   r"At least one stability check fails on \RMStabilityFailures{} task suites.")
    if expected_decision == "STOP_FORECASTING_PERFORMANCE_BRANCH":
        summary.append(r"The results therefore do not satisfy the prespecified overall improvement criterion.")
    else:
        summary.append(r"The results satisfy the prespecified overall improvement criterion under the tested laws and observation procedures.")
    contents["result_summary.tex"] = "\n".join(summary) + "\n"

    # Descriptive condition summaries. There are no post-selection confidence bars.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9), sharey=True)
    plot_data = []
    for method, title in (("rbf_departure", "Matched RBF departure"),
                          ("nonkahm_policy", "C-selected non-KAHM policy")):
        values = []
        for law, noise in conditions:
            k = float(condition[law, noise, "kahm_departure"]["mean_nmse"])
            c = float(condition[law, noise, method]["mean_nmse"])
            values.append(100 * (1 - k / c))
            plot_data.append({"law": law, "noise_level": noise, "comparator": method,
                              "candidate_nmse": k, "comparator_nmse": c,
                              "relative_gain_percent": values[-1], "inference": "descriptive"})
        axis = axes[0 if method == "rbf_departure" else 1]
        positions = np.arange(len(conditions))[::-1]
        colors = ["#21618c" if value >= 0 else "#b35422" for value in values]
        axis.axvline(0, color="#5e6870", linewidth=1)
        axis.hlines(positions, 0, values, color=colors, linewidth=2.1)
        axis.scatter(values, positions, c=colors, s=36, zorder=3)
        span = max(max(values) - min(values), 1.)
        axis.set_xlim(min(min(values), 0.) - .25 * span, max(max(values), 0.) + .25 * span)
        for position, value in zip(positions, values):
            axis.annotate(f"{value:+.2f}", (value, position),
                          xytext=(6 if value >= 0 else -6, 0), textcoords="offset points",
                          ha="left" if value >= 0 else "right", va="center", fontsize=9)
        axis.axhline(1.5, color="#ccd2d6", linewidth=.8, linestyle="--")
        axis.set_title(title, fontsize=10.5, pad=10)
        axis.grid(axis="x", color="#e6e9ec", linewidth=.7)
        axis.set_axisbelow(True)
        axis.set_xlabel("KAHM relative loss reduction (%)", fontsize=9.5)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=9, length=0)
    axes[0].set_yticks(np.arange(len(conditions))[::-1],
                      [f"{LAW_LABELS[law]}   η={noise:g}" for law, noise in conditions])
    fig.subplots_adjust(left=.25, right=.98, bottom=.14, top=.9, wspace=.17)
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(output / "law_noise_contrasts.pdf", bbox_inches="tight",
                    metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(output / "law_noise_contrasts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for name, content in contents.items():
        (output / name).write_text(content, encoding="utf-8")
    csv_outputs = {"resource_summary.csv": resources, "bound_summary.csv": bound_summary,
                   "activation_summary.csv": activation_summary, "policy_summary.csv": policy_summary,
                   "law_noise_plot_data.csv": plot_data}
    for name, rows in csv_outputs.items():
        write_csv(output / name, rows)
    findings = {
        "decision": expected_decision, "tasks": 192, "independent_repetitions": 24,
        "useful_superiority_conditions_passed": useful_passes, "linear_noninferiority_passed": linear_pass,
        "stability_passed": stable, "max_error_ratio": ratio_max, "max_error_task": worst,
        "feature_violations": feature_violations, "coefficient_violations": coefficient_violations,
        "correction_violations": correction_violations,
        "local_fallback_rows": {f"{kind}_{split}": value for (kind, split), value in fallback.items()},
        "primary_contrasts": primary, "resource_summary": resources, "activation_summary": activation_summary,
        "point_estimates": datasets["overview_metrics"], "public_preregistration": False,
        "local_freeze_commit": execution["local_freeze_commit"],
    }
    (output / "REPORT_FINDINGS.json").write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    output_names = [*contents, *csv_outputs, "law_noise_contrasts.pdf", "law_noise_contrasts.png", "REPORT_FINDINGS.json"]
    provenance = {
        "schema_version": 1, "scope": "Completed fixed study; reporting only, no models fitted or forecasts generated",
        "builder": description(Path(__file__)), "inputs": inputs,
        "outputs": {name: description(output / name) for name in sorted(output_names)},
        "checks": {"complete_tasks": 192, "task_method_records": 1920, "independent_repetitions": 24,
                   "primary_contrasts": 6, "decision": expected_decision,
                   "task_to_condition_and_repetition_aggregation": "passed",
                   "independent_recomputation_of_six_t_lower_bounds": "passed",
                   "frozen_decision_consistency": "passed", "saved_analysis_output_hashes": "passed",
                   "new_model_fits": 0, "new_forecasts": 0, "public_preregistration": False},
    }
    (output / "ASSET_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {"status": "passed", "output": str(output), **provenance["checks"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.study, args.output), indent=2))
