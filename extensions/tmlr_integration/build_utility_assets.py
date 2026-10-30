"""Reconstruct manuscript assets for the separate forecasting-utility extension.

Reporting only: this script never fits, tunes, or selects a forecasting model.
It requires an audited, fully attempted run with explicit failure accounting,
checks aggregation against task records, and derives residual alignment directly
from saved test predictions. Common-support archive displays are post-hoc
diagnostics and never replace undefined full-roster geometric losses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kahm-utility-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY = ROOT / "extensions/forecasting_utility"
METHODS = ["naive", "raw_ridge", "raw_rbf", "concat_kahm", "add_kahm",
           "add_affine", "add_locality", "hist_gradient",
           "utility_kahm_5pct", "utility_geometry_5pct"]
ORDER = ["raw_ridge", "raw_rbf", "concat_kahm", "add_kahm", "add_affine",
         "add_locality", "naive", "hist_gradient",
         "utility_kahm_5pct", "utility_geometry_5pct"]
LABELS = {"naive": "Persistence / seasonal naive", "raw_ridge": "Raw ridge",
          "raw_rbf": "Raw RBF", "concat_kahm": "Concatenated KAHM",
          "add_kahm": "Additive KAHM", "add_affine": "Additive affine",
          "add_locality": "Additive locality", "hist_gradient": "Histogram gradient boosting",
          "utility_kahm_5pct": "KAHM 5\\% rule", "utility_geometry_5pct": "All-geometry 5\\% rule"}
LAWS = {"nonlinear_ar": "Nonlinear AR", "vanderpol": "Van der Pol", "duffing": "Duffing"}
SOURCES = {"weather_daily": "Weather", "solar_weekly": "Solar Weekly",
           "covid_deaths_daily": "COVID-19 Deaths", "car_parts_monthly": "Car Parts"}
COLORS = {"raw_ridge": "#718096", "raw_rbf": "#333333", "concat_kahm": "#9b6ca7",
          "add_kahm": "#126782", "add_affine": "#d17a22", "add_locality": "#4c956c",
          "naive": "#a3a3a3", "hist_gradient": "#706993",
          "utility_kahm_5pct": "#126782", "utility_geometry_5pct": "#41644a"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, digits=4):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite value cannot enter a manuscript table")
    return f"{value:.{digits}f}"


def compact(value):
    return f"{float(value):.5g}"


def table(caption, label, spec, heading, rows, *, size=r"\small"):
    return "\n".join([
        "% Generated from the audited forecasting-utility extension; no model fitting.",
        r"\begin{table}[!htbp]", r"\centering" + size + r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(heading) + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def source_means(frame, values):
    """Equal capacities within series, series within source, then sources."""
    series = frame.groupby(["source_id", "series_id", "method"], sort=False)[values].mean().reset_index()
    sources = series.groupby(["source_id", "method"], sort=False)[values].mean().reset_index()
    overall = sources.groupby("method", sort=False)[values].mean().reset_index()
    return sources, overall


def check_table(actual, expected, keys, values):
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError(f"Aggregation roster differs for {keys}")
    np.testing.assert_allclose(a[values].to_numpy(float), b[values].to_numpy(float), rtol=1e-11, atol=1e-13)


def build(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    sources_used = {}

    def register(path):
        path = Path(path)
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        sources_used[name] = digest(path)
        return path

    def read_json(path):
        return json.loads(register(path).read_text())

    def read_csv(name):
        return pd.read_csv(register(run_dir / "report" / name))

    summary = read_json(run_dir / "report/summary.json")
    cfg = read_json(STUDY / "config.json")
    roster = read_json(STUDY / "prepared/roster.json")
    read_json(STUDY / "freeze.json")
    register(STUDY / "PROTOCOL.md")
    register(STUDY / "REPORTING_AMENDMENT.md")
    register(STUDY / "reporting.py")
    register(STUDY / "sources.json")
    execution = read_json(run_dir / "execution.json")
    if not summary.get("verification_passed") or not summary.get("all_tasks_attempted"):
        raise ValueError("Require an audited, fully attempted run with explicit failure accounting")
    register(run_dir / "report/failures.csv")
    register(run_dir / "report/coverage_by_method.csv")
    register(run_dir / "report/archive_support.csv")
    expected_synthetic = (len(cfg["synthetic"]["laws"]) * len(cfg["synthetic"]["training_episodes"])
                          * len(cfg["landmarks"]) * len(cfg["synthetic"]["repetition_seeds"]))
    expected_archive = len(roster["tasks"]) * len(cfg["landmarks"])
    expected_tasks = expected_synthetic + expected_archive
    if summary["expected_tasks"] != expected_tasks or summary["observed_tasks"] != expected_tasks:
        raise ValueError("Observed task count differs from the configured factorial/roster")
    if summary["execution_freeze_sha256"] != digest(STUDY / "freeze.json"):
        raise ValueError("Analysis belongs to a different execution freeze")
    frame = read_csv("task_methods.csv")
    if len(frame) != summary["successful_methods"] + summary["successful_policies"] or frame.duplicated(["task_id", "method"]).any():
        raise ValueError("Successful-output count or unique task-method accounting differs")
    if set(frame.method) != set(METHODS):
        raise ValueError("Unexpected model or policy roster")
    metrics = ["test_mse", "scaled_mse", "pipeline_seconds", "model_pipeline_seconds",
               "model_cpu_seconds", "serialized_model_bytes"]
    if not np.isfinite(frame[metrics].to_numpy(float)).all():
        raise ValueError("Incomplete numeric cost or loss evidence")
    synthetic = frame[frame.phase == "synthetic"]
    archive_all = frame[frame.phase == "natural"]
    archive = read_csv("archive_common_support_task_methods.csv")
    support_ids = set(archive.task_id)
    if len(support_ids) != summary["archive_common_support_tasks"]:
        raise ValueError("Archive common-support count differs from the reporting amendment")
    if not all(set(group.method) == set(METHODS) for _, group in synthetic.groupby("task_id")):
        raise ValueError("A synthetic task is missing a declared comparator")
    if not all(set(group.method) == set(METHODS) for _, group in archive.groupby("task_id")):
        raise ValueError("A common-support task is missing a declared comparator")
    check_table(archive, archive_all[archive_all.task_id.isin(support_ids)], ["task_id", "method"], metrics)
    cell_keys = ["law", "training_episodes", "landmarks", "method"]
    syn = synthetic.groupby(cell_keys, sort=False)[metrics].mean().reset_index()
    archive_sources, arc = source_means(archive, metrics)
    complete_archive_methods = [method for method in METHODS
                                if len(archive_all[archive_all.method == method]) == expected_archive]
    _, arc_full = source_means(archive_all[archive_all.method.isin(complete_archive_methods)], metrics)
    check_table(arc_full, read_csv("archive_full_roster_baselines.csv"), ["method"], metrics)
    check_table(syn, read_csv("synthetic_methods.csv"), cell_keys, metrics)
    check_table(archive_sources, read_csv("archive_sources.csv"), ["source_id", "method"], metrics)
    check_table(arc, read_csv("archive_methods.csv"), ["method"], metrics)
    contrasts = read_csv("synthetic_primary_contrasts.csv")
    if len(contrasts) != 24 or not (contrasts.n == 12).all() or not (contrasts.family_size == 24).all():
        raise ValueError("Primary interval family differs from the declared design")
    choices = read_csv("policy_choices.csv")
    gamma = read_csv("gamma_choices.csv")
    leave_one = read_csv("archive_leave_one_source_out.csv")
    output.mkdir(parents=True, exist_ok=True)
    arc = arc.set_index("method")
    arc_full = arc_full.set_index("method")
    syn = syn.set_index(cell_keys)
    archive_sources = archive_sources.set_index(["source_id", "method"])

    # Reconstruct the standard residual identity from full saved test forecasts.
    decomposition = []
    failure_records = []
    identity_error = 0.0
    for task in execution["expected_tasks"]:
        directory = run_dir / "tasks" / task["task_id"]
        record = read_json(directory / "result.json")
        with np.load(register(directory / "targets.npz"), allow_pickle=False) as data:
            target = data["Y_test"]
        records = {row["method"]: row for row in record["methods"]}
        failed_methods = [name for name, item in records.items() if item["status"] != "ok"]
        failed_policies = [name for name, item in record["policies"].items() if item["status"] != "ok"]
        if failed_methods or failed_policies:
            prepared = next(item for item in roster["tasks"] if item["base_task_id"] == task["base_task_id"])
            with np.load(register(STUDY / "prepared" / prepared["array_file"]), allow_pickle=False) as saved:
                histories = saved["X_train"]
                distinct_histories = len(np.unique(histories, axis=0))
            failure_records.append({"task_id": task["task_id"], "source": SOURCES[task["source_id"]],
                                    "series": task["series_id"], "landmarks": task["landmarks"],
                                    "training_rows": len(histories), "distinct_histories": distinct_histories,
                                    "failed_methods": failed_methods, "failed_policies": failed_policies})

        def prediction(method):
            with np.load(register(directory / records[method]["prediction_file"]), allow_pickle=False) as saved:
                return saved["prediction"]

        raw = prediction("raw_rbf")
        fitted = {"raw_rbf": raw}
        for method in ("add_kahm", "add_affine", "add_locality"):
            if records[method]["status"] == "ok":
                fitted[method] = prediction(method)
        for method in METHODS[-2:]:
            if record["policies"][method]["status"] == "ok":
                fitted[method] = fitted[record["policies"][method]["selected_method"]]
        scale = float(task.get("loss_scale", 1.0))
        residual = target - raw
        for method, pred in fitted.items():
            if method == "raw_rbf":
                continue
            correction = pred - raw
            alignment = 2 * float(np.mean(residual * correction)) / scale
            energy = float(np.mean(correction ** 2)) / scale
            gain = float(np.mean(residual ** 2) - np.mean((target - pred) ** 2)) / scale
            error = abs(gain - (alignment - energy))
            identity_error = max(identity_error, error)
            if not np.isclose(gain, alignment - energy, rtol=1e-10, atol=1e-12):
                raise ValueError("Saved predictions violate the squared-error identity")
            decomposition.append({"task_id": task["task_id"], "phase": task["phase"],
                                  "law": task.get("law", ""), "source_id": task.get("source_id", ""),
                                  "series_id": task.get("series_id", ""), "landmarks": task["landmarks"],
                                  "training_episodes": task.get("training_episodes", 0), "method": method,
                                  "twice_residual_alignment": alignment, "correction_energy": energy,
                                  "gain": gain, "identity_absolute_error": error})
    decomposition = pd.DataFrame(decomposition)
    audited_decomposition = read_csv("correction_decomposition.csv")[[
        "task_id", "method", "scaled_alignment", "scaled_correction_energy", "scaled_raw_minus_additive_mse"]].rename(columns={
        "scaled_alignment": "twice_residual_alignment",
        "scaled_correction_energy": "correction_energy",
        "scaled_raw_minus_additive_mse": "gain"})
    check_table(decomposition[decomposition.method.str.startswith("add_")],
                audited_decomposition, ["task_id", "method"],
                ["twice_residual_alignment", "correction_energy", "gain"])
    decomposition.to_csv(output / "residual_decomposition_tasks.csv", index=False)
    terms = ["twice_residual_alignment", "correction_energy", "gain"]
    residual_sources, residual_overall = source_means(
        decomposition[(decomposition.phase == "natural") & decomposition.task_id.isin(support_ids)], terms)
    residual_sources.to_csv(output / "residual_decomposition_sources.csv", index=False)
    residual_overall.to_csv(output / "residual_decomposition_overall.csv", index=False)

    macro_methods = {"Raw": "raw_rbf", "Ridge": "raw_ridge", "Concat": "concat_kahm",
                     "KAHM": "add_kahm", "Affine": "add_affine", "Locality": "add_locality",
                     "Naive": "naive", "HGB": "hist_gradient", "Policy": "utility_kahm_5pct",
                     "GeometryPolicy": "utility_geometry_5pct"}
    macros = {"UtilityTaskCount": str(expected_tasks), "UtilitySyntheticTasks": str(expected_synthetic),
              "UtilityArchiveTasks": str(expected_archive), "UtilityWarnings": str(summary["warnings_count"]),
              "UtilityArchiveCompleteTasks": str(summary["archive_common_support_tasks"]),
              "UtilitySuccessfulForecasts": str(summary["successful_methods"]),
              "UtilityFailedFits": str(summary["failed_method_count"]),
              "UtilityFailedPolicies": str(summary["failed_policy_count"])}
    for prefix, method in macro_methods.items():
        row = arc.loc[method]
        macros.update({f"Utility{prefix}Loss": number(row.scaled_mse),
                       f"Utility{prefix}Cost": number(row.model_pipeline_seconds),
                       f"Utility{prefix}CostRatio": number(row.model_pipeline_seconds / arc.loc["raw_rbf"].model_pipeline_seconds, 2),
                       f"Utility{prefix}GainPercent": number(100 * (1 - row.scaled_mse / arc.loc["raw_rbf"].scaled_mse), 2)})
        if method in arc_full.index:
            macros[f"Utility{prefix}FullRosterLoss"] = number(arc_full.loc[method].scaled_mse)
    for control, prefix in (("raw_rbf", "Raw"), ("concat_kahm", "Concat")):
        group = contrasts[contrasts.control == control]
        macros[f"Utility{prefix}PositiveContrasts"] = str(int((group.lower > 0).sum()))
        macros[f"Utility{prefix}NegativeContrasts"] = str(int((group.upper < 0).sum()))
        macros[f"Utility{prefix}UnresolvedContrasts"] = str(int(((group.lower <= 0) & (group.upper >= 0)).sum()))
    for policy, prefix in (("utility_kahm_5pct", "KAHM"), ("utility_geometry_5pct", "Geometry")):
        selected = choices[(choices.phase == "natural") & (choices.method == policy)]
        macros[f"Utility{prefix}Adoptions"] = str(int(selected[selected.selected_method != "raw_rbf"].tasks.sum()))
    for phase, prefix in (("natural", "Archive"), ("synthetic", "Synthetic")):
        selected = gamma[(gamma.phase == phase) & (gamma.method == "add_kahm")]
        macros[f"Utility{prefix}KAHMNonzero"] = str(int(selected[selected.gamma > 0].tasks.sum()))
    (output / "numbers.tex").write_text("% Generated from the fully attempted, failure-aware audited utility run.\n% Archive method means without FullRoster in their name are POST-HOC common-support diagnostics.\n" + "".join(
        "\\newcommand{\\" + key + "}{" + value + "}\n" for key, value in macros.items()))

    # Main synthetic table retains both adverse and favorable cells.
    rows, control_rows = [], []
    cells = [(law, n, m) for law in LAWS for n in (32, 128) for m in (16, 64)]
    for law, n, m in cells:
        fields = [LAWS[law], str(n), str(m)]
        rows.append(" & ".join(fields + [number(1000 * syn.loc[(law, n, m, method)].test_mse)
                    for method in ("raw_rbf", "concat_kahm", "add_kahm", "add_affine", "add_locality")]) + r" \\")
        control_rows.append(" & ".join(fields + [number(1000 * syn.loc[(law, n, m, method)].test_mse)
                    for method in ("raw_ridge", "naive", "hist_gradient", "utility_kahm_5pct", "utility_geometry_5pct")]) + r" \\")
    (output / "synthetic_main.tex").write_text(table(
        "New synthetic factorial: mean test MSE multiplied by $10^3$, over 12 fresh complete repetitions per cell. "
        "$N$ counts training episodes; $M$ counts raw RBF landmarks. All correction heads are ridge heads. "
        "These are cell means; simultaneous primary intervals appear in Figure~\\ref{fig:utility-contrasts}.",
        "tab:utility-synthetic", "lrrrrrrr", ["Law", "$N$", "$M$", "Raw RBF", "Concat.", "+KAHM", "+Affine", "+Locality"], rows))
    (output / "synthetic_controls.tex").write_text(table(
        "Remaining controls and declared policies in the same new synthetic factorial. Mean MSE is multiplied by $10^3$. "
        "The two policies use the fixed 5\\% validation margin; their test losses are descriptive.",
        "tab:utility-synthetic-controls", "lrrrrrrr", ["Law", "$N$", "$M$", "Raw ridge", "Persistence", "Boosting", "KAHM rule", "Geometry rule"], control_rows))

    primary_rows = []
    for law, n, m in cells:
        for control in ("raw_rbf", "concat_kahm"):
            row = contrasts[(contrasts.law == law) & (contrasts.training_episodes == n)
                            & (contrasts.landmarks == m) & (contrasts.control == control)].iloc[0]
            primary_rows.append(" & ".join([LAWS[law], str(n), str(m), "Raw RBF" if control == "raw_rbf" else "Concat.",
                                number(row.mean_control_minus_additive * 1000),
                                "[" + number(row.lower * 1000) + ", " + number(row.upper * 1000) + "]"]) + r" \\")
    (output / "synthetic_intervals.tex").write_text(table(
        "Every primary new synthetic contrast. Differences and interval endpoints are multiplied by $10^3$; positive differences favor additive KAHM. "
        "The 24 intervals form their own 95\\% Bonferroni paired Student family over 12 complete repetitions. "
        "The normality approximation is not a distribution-free guarantee or joint coverage claim with earlier studies.",
        "tab:utility-intervals", "lrrlrr", ["Law", "$N$", "$M$", "Control", "$\\Delta$", "Simultaneous interval"], primary_rows))

    archive_rows, cost_rows, source_rows = [], [], []
    for method in ORDER:
        row = arc.loc[method]
        coverage = len(archive_all[archive_all.method == method])
        full_loss = number(arc_full.loc[method].scaled_mse) if method in arc_full.index else r"\textemdash"
        archive_rows.append(" & ".join([LABELS[method], f"{coverage}/{expected_archive}", full_loss,
                                         number(row.scaled_mse), number(row.model_pipeline_seconds)]) + r" \\")
        cost_rows.append(" & ".join([LABELS[method], number(row.model_pipeline_seconds), number(row.model_cpu_seconds),
                                     number(row.pipeline_seconds - row.model_pipeline_seconds), number(row.pipeline_seconds),
                                     number(row.serialized_model_bytes / 1024, 1)]) + r" \\")
        source_rows.append(" & ".join([LABELS[method]] + [number(archive_sources.loc[(source, method)].scaled_mse)
                                                         for source in SOURCES]) + r" \\")
    (output / "archive_main.tex").write_text(table(
        "New archive-source coverage and losses. Full-roster scaled MSE uses all 24 configurations and is undefined (dash) for procedures with failed fits. "
        "Matched loss and wall seconds use only the post-hoc 22-configuration common-support diagnostic (11 series, including two Car Parts series), with equal source weights. "
        "Pipeline costs include all required searches, validation, selected refit, prediction, and serialization. "
        "Policy costs are unions of measured required components, including rejected corrections; they are not separately timed full policy executions. "
        "Conditioning on success does not complete the prespecified full-roster evaluation.",
        "tab:utility-archive", "lrrrr", ["Procedure", "Outputs", "Full-roster MSE", "Matched MSE", "Matched seconds"], archive_rows))
    (output / "archive_sources.tex").write_text(table(
        "Post-hoc matched-success scaled MSE by new archive source, averaging capacities within series and then series. "
        "The 22 configurations contain three Weather, three Solar, three COVID, and two Car Parts series; the failed Car Parts series is excluded only from this diagnostic. "
        "Solar Weekly has simulation provenance. All procedures and both frozen policies are shown.",
        "tab:utility-archive-sources", "lrrrr", ["Procedure", "Weather", "Solar", "COVID", "Car Parts"], source_rows))
    (output / "archive_costs.tex").write_text(table(
        "Post-hoc common-support archive resource components, in source-balanced mean seconds per successful task. "
        "Model wall time is the main cost measure; preparation-plus-model adds common data preparation. "
        "Downloads, imports, artifact I/O, and research analysis remain outside these model measurements. "
        "The node-based policy totals charge shared work once. Model KiB describe actual stored state, not minimal storage. "
        "These diagnostic averages do not remove failed-attempt records from the artifact.",
        "tab:utility-cost-detail", "lrrrrr", ["Procedure", "Model wall", "Model CPU", "Prep.", "Prep. + model", "Model KiB"], cost_rows))

    window_rows = []
    for task in roster["tasks"]:
        counts = [str(task["split_details"][split]["n_windows"]) for split in ("train", "validation", "refit", "test")]
        window_rows.append(" & ".join([SOURCES[task["source_id"]], task["series_id"], str(task["n_values"]), *counts,
                                       "Seasonal" if task["scale_method"] == "seasonal_naive_mse" else "First diff."]) + r" \\")
    (output / "archive_windows.tex").write_text(table(
        "Exact available-window counts for all twelve originally selected series, including Car Parts T1324 whose geometric fits fail, after the fixed caps. "
        "Inputs and targets are shared by the two capacities and every procedure. "
        "Initial training observations set the scale; refit windows stop before the reserved test boundary. "
        "These windows are related observations, not independent source replications.",
        "tab:utility-windows", "llrrrrrl", ["Source", "Series", "Length", "Train", "Valid.", "Refit", "Test", "Scale"], window_rows))
    failure_rows = [" & ".join([row["source"], row["series"], str(row["landmarks"]), str(row["training_rows"]),
                                 str(row["distinct_histories"]), str(len(row["failed_methods"])),
                                 str(len(row["failed_policies"]))]) + r" \\" for row in failure_records]
    (output / "failures.tex").write_text(table(
        "Every incomplete configuration in the frozen execution. The shared bank requests eight nonempty groups but these training inputs have only seven distinct histories. "
        "Concatenated KAHM and additive KAHM, affine, and locality each fail; both dependent policies have no decision. "
        "No source substitution, changed cluster count, or silent fallback is applied. The four baseline forecasts are retained.",
        "tab:utility-failures", "llrrrrr", ["Source", "Series", "$M$", "Train rows", "Distinct", "Failed fits", "Failed policies"], failure_rows))

    def integer_range(values):
        values = np.asarray(values, float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Effective decoder metadata is missing")
        lo, hi = int(values.min()), int(values.max())
        return str(lo) if lo == hi else f"{lo}--{hi}"

    decoder_rows = []
    for source in SOURCES:
        for requested in cfg["landmarks"]:
            subset = archive[(archive.source_id == source) & (archive.landmarks == requested)]
            raw_head = subset[subset.method == "raw_rbf"]
            concat_head = subset[subset.method == "concat_kahm"]
            decoder_rows.append(" & ".join([SOURCES[source], str(requested),
                integer_range(raw_head.head_actual_landmarks), integer_range(raw_head.head_retained_nystrom_rank),
                integer_range(raw_head.head_basis_width), integer_range(concat_head.head_retained_nystrom_rank)]) + r" \\")
    (output / "decoder_sizes.tex").write_text(table(
        "Actual archive decoder sizes on post-hoc common support after the selected pretest refit. Entries are ranges across three retained series per source, or two for Car Parts. "
        r"Requested landmarks are capped by available fitting histories; the retained Nystr\"om rank can be smaller still. "
        "Raw basis width includes the appended history coordinates. Additive methods preserve this same raw head and add a separate ridge correction.",
        "tab:utility-decoder-sizes", "lrrrrr", ["Source", "Requested $M$", "Raw landmarks", "Raw rank", "Raw width", "Concat. rank"], decoder_rows))

    gamma_rows = []
    for phase in ("synthetic", "natural"):
        for method in ("add_kahm", "add_affine", "add_locality"):
            group = gamma[(gamma.phase == phase) & (gamma.method == method)].set_index("gamma")
            gamma_rows.append(" & ".join(["Synthetic" if phase == "synthetic" else "Archive", LABELS[method]]
                                         + [str(int(group.loc[g].tasks)) if g in group.index else "0" for g in (0, .25, .5, 1)]) + r" \\")
    (output / "gamma_choices.tex").write_text(table(
        "Selected correction weights before the additional 5\\% policy gate. Synthetic counts concern 144 configurations and archive counts the 22 successful geometric configurations. "
        "Each archive geometry additionally has two failed fits with no selected weight. "
        "They characterize the frozen selector; frequencies across related configurations are not independent Bernoulli observations.",
        "tab:utility-gamma", "llrrrr", ["Study", "Correction", "$0$", "$0.25$", "$0.5$", "$1$"], gamma_rows))
    policy_rows = []
    for phase in ("synthetic", "natural"):
        for method in METHODS[-2:]:
            group = choices[(choices.phase == phase) & (choices.method == method)].set_index("selected_method")
            policy_rows.append(" & ".join(["Synthetic" if phase == "synthetic" else "Archive", LABELS[method]]
                                          + [str(int(group.loc[g].tasks)) if g in group.index else "0"
                                             for g in ("raw_rbf", "add_kahm", "add_affine", "add_locality")]) + r" \\")
    (output / "policy_choices.tex").write_text(table(
        "Choices made by both declared 5\\% validation policies. Archive decisions cover the 22 completed evaluations; each policy also has two failures and no decision on those configurations. "
        "Costs include examining rejected corrections; a raw decision is not a free decision.",
        "tab:utility-policy-choices", "llrrrr", ["Study", "Rule", "Raw", "+KAHM", "+Affine", "+Locality"], policy_rows))

    residual_rows = []
    for source in SOURCES:
        for method in ("add_kahm", "add_affine", "add_locality"):
            row = residual_sources[(residual_sources.source_id == source) & (residual_sources.method == method)].iloc[0]
            residual_rows.append(" & ".join([SOURCES[source], LABELS[method]] + [compact(row[key]) for key in terms]) + r" \\")
    residual_rows.append(r"\midrule")
    for method in ("add_kahm", "add_affine", "add_locality"):
        row = residual_overall[residual_overall.method == method].iloc[0]
        residual_rows.append(" & ".join(["Equal-source mean", LABELS[method]] + [compact(row[key]) for key in terms]) + r" \\")
    (output / "residual_alignment.tex").write_text(table(
        "Descriptive test residual decomposition on the post-hoc common-support archive subset. For $d=\\widehat f_{\\rm add}-\\widehat f_{\\rm raw}$, "
        "alignment is $2\\,\\mathrm{mean}\\langle Y-\\widehat f_{\\rm raw},d\\rangle$ and energy is $\\mathrm{mean}\\|d\\|^2$, "
        "averaged over target coordinates and divided by the fixed training scale before hierarchical aggregation. "
        "Raw-minus-additive MSE equals alignment minus energy before rounding. These are identities of the saved predictions, not fresh inference or a population certificate.",
        "tab:utility-residual", "llrrr", ["Source", "Correction", "Alignment", "Energy", "Gain"], residual_rows))
    leave_rows = [" & ".join([SOURCES[row.omitted_source], number(row.raw_rbf), number(row.add_kahm),
                               f"{row.relative_change_pct:+.2f}\\% "]) + r" \\" for row in leave_one.itertuples()]
    (output / "leave_one_source_out.tex").write_text(table(
        "Descriptive leave-one-source-out sensitivity of the post-hoc common-support additive KAHM comparison. Relative change is additive/raw minus one. "
        "These re-aggregations reuse the same four sources and do not create additional evidence or select a preferred aggregate.",
        "tab:utility-leave-one", "lrrr", ["Omitted source", "Raw RBF", "Additive KAHM", "Relative change"], leave_rows))

    # A compact forest plot makes every primary contrast visible.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.5), sharey=True)
    for ax, law in zip(axes, LAWS):
        for j, (n, m) in enumerate([(32, 16), (32, 64), (128, 16), (128, 64)]):
            for control, offset, color, marker in [("raw_rbf", -.11, "#126782", "o"), ("concat_kahm", .11, "#9b6ca7", "s")]:
                row = contrasts[(contrasts.law == law) & (contrasts.training_episodes == n)
                                & (contrasts.landmarks == m) & (contrasts.control == control)].iloc[0]
                mean, lower, upper = 1000 * np.array([row.mean_control_minus_additive, row.lower, row.upper])
                ax.errorbar(mean, j + offset, xerr=np.array([[mean - lower], [upper - mean]]),
                            fmt=marker, color=color, markersize=4.5, capsize=2.5,
                            label=LABELS[control] if j == 0 else None)
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.set_title(LAWS[law], fontsize=10, weight="bold")
        ax.grid(axis="x", alpha=.2)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_xlabel("Control − additive KAHM MSE\n(× 10⁻³)")
    axes[0].set_yticks(range(4), ["N=32, M=16", "N=32, M=64", "N=128, M=16", "N=128, M=64"])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.56, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / "synthetic_contrasts.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(output / "synthetic_contrasts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(ORDER))
    ratios = arc.loc[ORDER].scaled_mse.to_numpy() / arc.loc["raw_rbf"].scaled_mse
    costs = arc.loc[ORDER].model_pipeline_seconds.to_numpy() / arc.loc["raw_rbf"].model_pipeline_seconds
    axes[0].barh(y, ratios, color=[COLORS[m] for m in ORDER], height=.65)
    axes[1].scatter(costs, y, c=[COLORS[m] for m in ORDER], s=34)
    for ax in axes:
        ax.axvline(1, color="#555555", linewidth=.8, linestyle="--")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(y, [LABELS[m].replace("\\%", "%") for m in ORDER])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Scaled MSE / raw RBF\n(lower is better)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Model-pipeline wall time / raw RBF\n(log scale; lower is cheaper)")
    fig.suptitle("Post-hoc common-support diagnostic: 22 tasks, 11 series", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "archive_accuracy_cost.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(output / "archive_accuracy_cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {"schema_version": 1, "study": cfg["study_id"],
                "scope": "Post-execution reporting only. Synthetic results retain all144 tasks. Archive common-support summaries are explicitly post-hoc diagnostics; full-roster geometric losses remain undefined. No model fitting, tuning, or test-based selection.",
                "failure_accounting": {key: summary[key] for key in ("completed_tasks", "successful_methods", "failed_method_count",
                                      "successful_policies", "failed_policy_count", "archive_common_support_tasks",
                                      "archive_common_support_series", "archive_common_support_sources")},
                "execution_freeze_sha256": summary["execution_freeze_sha256"],
                "input_sha256": sources_used, "macros": macros,
                "residual_identity_max_absolute_error": identity_error,
                "generated_sha256": {p.name: digest(p) for p in sorted(output.iterdir())
                                     if p.is_file() and p.name != "ASSET_PROVENANCE.json"}}
    (output / "ASSET_PROVENANCE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"status": "passed", "forecasting_fits": 0, "task_count": expected_tasks,
            "input_files": len(sources_used), "generated_files": len(manifest["generated_sha256"]),
            "residual_identity_max_absolute_error": identity_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=STUDY / "results/utility_v1")
    parser.add_argument("--output-dir", type=Path, default=HERE / "manuscript/generated/utility")
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir, args.output_dir), indent=2))
