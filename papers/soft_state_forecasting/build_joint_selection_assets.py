#!/usr/bin/env python3
"""Build manuscript displays from the completed joint-selection follow-up.

This reporting-only program verifies released analysis-table hashes and row
counts, requires the matching passed independent audit, and recomputes all six
primary paired bounds using scalar arithmetic. It never reads model states,
episodes, target arrays or forecast arrays and performs no model fitting.
Output is written atomically to a new directory; existing assets are not
overwritten. TeX table fragments contain no captions or labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import tempfile

from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
DEFAULT_STUDY = HERE.parent.parent / "extensions/soft_state_joint_selection_20261102"
ARMS = ("G", "P", "K", "R")
VIEWS = ("shared", "independent", "standard_shared", "standard_independent")
KEYS = tuple(f"{view}__{arm}" for view in VIEWS for arm in ARMS)
SCALINGS = ("standard", "minmax", "tanh_z_1")
CAPS = (1, 2, 4, 8, 12)
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
LAW_LABELS = {"linear_ar": "Linear AR", "nonlinear_ar": "Nonlinear AR",
              "vanderpol": "Van der Pol", "duffing": "Duffing"}
PANELS = {"nonlinear": LAWS[1:], "linear": LAWS[:1], "all_descriptive": LAWS}
N = 32
TASKS = 256
ALPHA = .05
CLAIMS = (
    ("P_vs_G_nonlinear", "nonlinear", "shared__P", "shared__G", .97, "PG"),
    ("K_vs_P_nonlinear", "nonlinear", "shared__K", "shared__P", .97, "KP"),
    ("K_vs_R_nonlinear", "nonlinear", "shared__K", "shared__R", .97, "KR"),
    ("K_vs_P_linear", "linear", "shared__K", "shared__P", 1.05, "LinearKP"),
    ("joint_K_vs_standard_K_nonlinear", "nonlinear", "independent__K",
     "standard_independent__K", .97, "JointKNonlinear"),
    ("joint_K_vs_standard_K_linear", "linear", "independent__K",
     "standard_independent__K", 1.05, "JointKLinear"),
)
EXPECTED_ROWS = {
    "task_losses.csv": 4096, "horizon_losses.csv": 12288,
    "repetition_losses.csv": 1536, "panel_summary.csv": 48,
    "condition_summary.csv": 128, "primary_contrasts.csv": 6,
    "primary_paired_repetitions.csv": 192, "secondary_descriptive.csv": 64,
    "primary_difference_distribution.csv": 6,
    "optimization_repetition_pairs.csv": 512, "optimization_summary.csv": 16,
    "selection_by_task.csv": 4096, "resource_by_task.csv": None,
    "basis_dimensions.csv": None, "feature_widths.csv": 15360,
    "B_candidate_scores.csv": 230400, "C_diagnostic_losses.csv": 12288,
    "optimization_task_pairs.csv": 2048, "horizon_summary.csv": 144,
    "cap_selection_frequencies.csv": 240, "scaling_selection_frequencies.csv": 144,
    "joint_selection_frequencies.csv": 720, "resource_summary.csv": None,
}
NEEDED_TABLES = {"task_losses.csv", "repetition_losses.csv", "panel_summary.csv",
                 "primary_contrasts.csv", "primary_paired_repetitions.csv",
                 "selection_by_task.csv", "scaling_selection_frequencies.csv",
                 "resource_summary.csv"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    result = float(value)
    require(math.isfinite(result), f"Nonfinite reporting value: {value!r}")
    return result


def near(actual, expected, label):
    if expected is None:
        require(actual is None or actual == "", "Expected undefined ratio: " + label)
        return
    require(math.isclose(finite(actual), finite(expected), rel_tol=3e-10, abs_tol=3e-13),
            f"Display verification failed at {label}: {actual!r} != {expected!r}")


def boolean(value):
    require(isinstance(value, bool) or value in ("True", "False"), "Invalid Boolean")
    return value if isinstance(value, bool) else value == "True"


def mean(values):
    values = [finite(x) for x in values]
    require(bool(values), "Empty reporting mean")
    return math.fsum(values) / len(values)


def sd(values):
    values = [finite(x) for x in values]
    mu = mean(values)
    require(len(values) > 1, "Sample SD requires multiple observations")
    return math.sqrt(math.fsum((x - mu) ** 2 for x in values) / (len(values) - 1))


def gain(candidate, reference):
    a, b = finite(candidate), finite(reference)
    require(a >= 0 and b >= 0, "Negative mean loss")
    return 100 * (1 - a / b) if b else None


def indexed(rows, keys, label):
    result = {tuple(str(row[key]) for key in keys): row for row in rows}
    require(len(result) == len(rows), "Duplicate " + label)
    return result


class Inputs:
    def __init__(self):
        self.records = {}

    def raw(self, path, name):
        value = Path(path).read_bytes()
        record = {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        require(name not in self.records or self.records[name] == record, "Input changed during reporting")
        self.records[name] = record
        return value

    def json(self, path, name):
        def reject(value):
            raise ValueError("Nonfinite JSON constant: " + value)
        return json.loads(self.raw(path, name), parse_constant=reject)

    def digest(self, name):
        return self.records[name]["sha256"]


def load_evidence(study, inputs):
    config = inputs.json(study / "config.json", "study/config.json")
    require(config["master_seed"] == 2026110204 and config["repetitions"] == N and
            config["task_count"] == TASKS and not config.get("manufactured_fixture"),
            "Completed fresh production cohort required")
    require(tuple(config["input_scaling_order"]) == SCALINGS and
            tuple(config["native_cap_order"]) == CAPS and tuple(config["contexts"]) == VIEWS,
            "Joint-selection grid or views changed")
    require(config["analysis"]["primary_family_size"] == 6 and
            config["analysis"]["family_alpha"] == ALPHA, "Primary family changed")
    freeze = inputs.json(study / "freeze.json", "study/freeze.json")
    publication = inputs.json(study / "PUBLICATION.json", "study/PUBLICATION.json")
    summary = inputs.json(study / "analysis/analysis_summary.json", "analysis/analysis_summary.json")
    audit = inputs.json(study / "verification/independent_audit.json", "verification/independent_audit.json")
    require(freeze["status"] == "frozen" and publication["status"] == "published" and
            publication["source_files_verified"] is True, "Frozen and published source required")
    require(summary["schema"] == "soft-state-joint-selection-analysis-v1" and summary["status"] == "complete",
            "Completed joint-selection analysis required")
    require(audit["schema"] == "independent-soft-state-joint-audit-v1" and audit["status"] == "passed",
            "Passed independent joint-selection audit required")
    for document in (summary, audit):
        require(document["tasks"] == TASKS and document["independent_repetitions"] == N,
                "Incomplete audited evaluation population")
        require(document["config_sha256"] == inputs.digest("study/config.json") and
                document["freeze_sha256"] == inputs.digest("study/freeze.json"),
                "Analysis/audit source identity differs")
    require(freeze["config_sha256"] == inputs.digest("study/config.json") and
            publication["freeze_sha256"] == inputs.digest("study/freeze.json"), "Freeze/publication link differs")
    require(audit["analysis_summary_sha256"] == inputs.digest("analysis/analysis_summary.json"),
            "Audit does not verify this exact analysis")
    require(summary["forecast_arrays"] == 4096 and summary["primary_family_size"] == 6 and
            audit["D_forecast_arrays_replayed"] == audit["C_forecast_arrays_replayed"] == 4096 and
            audit["B_penalty_path_arrays_replayed"] == 76800, "Incomplete forecast/audit coverage")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", publication["source_commit"])), "Invalid published commit")
    for relative in ("analysis_code/analyze_soft_state.py", "audit/verify_soft_state.py", "audit/independent_oracle.py"):
        inputs.raw(study / relative, "study/" + relative)
        actual = inputs.digest("study/" + relative)
        require(freeze["files"][relative] == actual, "Reporting source no longer matches freeze: " + relative)
        declared = summary["analyzer_sha256"] if relative.startswith("analysis_code/") else audit["source_hashes"][Path(relative).name]
        require(actual == declared, "Executed analyzer/audit source differs")
    for filename, field, expected_status in (("FIT_LOCK.json", "fit_lock_sha256", "sealed"),
                                            ("PREDICTION_LOCK.json", "prediction_lock_sha256", "sealed"),
                                            ("TARGET_RELEASE.json", "target_release_sha256", "released")):
        document = inputs.json(study / filename, "study/" + filename)
        require(document["status"] == expected_status and summary[field] == inputs.digest("study/" + filename),
                "Completed analysis stage link differs: " + filename)
    require(set(summary["tables"]) == set(EXPECTED_ROWS), "Released table roster differs")
    tables = {}
    for name, nominal_rows in EXPECTED_ROWS.items():
        raw = inputs.raw(study / "analysis" / name, "analysis/" + name)
        expected = summary["tables"][name]
        require(inputs.digest("analysis/" + name) == expected["sha256"], "Changed released table: " + name)
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
        if name in NEEDED_TABLES:
            rows = list(reader)
            count = len(rows)
            tables[name] = rows
        else:
            count = sum(1 for _ in reader)
        require(count == expected["rows"] and (nominal_rows is None or count == nominal_rows),
                "Incomplete released table: " + name)
        inputs.records["analysis/" + name]["rows"] = count
    return config, publication, summary, audit, tables


def recompute(tables, summary, audit):
    tasks = indexed(tables["task_losses.csv"], ("repetition", "law", "noise_index", "forecast_key"), "task losses")
    expected = {(str(rep), law, str(noise), key) for rep in range(N) for law in LAWS for noise in (0, 1) for key in KEYS}
    require(set(tasks) == expected, "Complete factorial task-loss roster required")
    repeated = indexed(tables["repetition_losses.csv"], ("panel", "repetition", "forecast_key"), "repetition losses")
    panels = indexed(tables["panel_summary.csv"], ("panel", "forecast_key"), "panel summaries")
    vectors = {}
    for panel, laws in PANELS.items():
        for key in KEYS:
            values = [mean(tasks[str(rep), law, str(noise), key]["smse"] for law in laws for noise in (0, 1)) for rep in range(N)]
            for rep, value in enumerate(values):
                near(repeated[panel, str(rep), key]["smse"], value, "equal-cell repetition loss")
            near(panels[panel, key]["mean_smse"], mean(values), "panel mean")
            near(panels[panel, key]["repetition_sd"], sd(values), "panel repetition SD")
            vectors[panel, key] = values
    paired = indexed(tables["primary_paired_repetitions.csv"], ("contrast_id", "repetition"), "primary paired rows")
    require(set(paired) == {(cid, str(rep)) for cid, *_ in CLAIMS for rep in range(N)}, "Incomplete six-contrast paired roster")
    published_sets = [indexed(rows, ("contrast_id",), label) for rows, label in (
        (tables["primary_contrasts.csv"], "primary CSV"), (summary["primary_contrasts"], "primary JSON"),
        (audit["primary_contrasts"], "audit primary rows"))]
    require(all(set(rows) == {(claim[0],) for claim in CLAIMS} for rows in published_sets), "Primary contrast roster differs")
    critical = float(student_t.isf(ALPHA / len(CLAIMS), N - 1))
    primary = []
    for cid, panel, candidate, reference, factor, suffix in CLAIMS:
        left, right, differences = [], [], []
        for rep in range(N):
            row = paired[cid, str(rep)]
            require(row["candidate_key"] == candidate and row["comparator_key"] == reference,
                    "Cross-context pair identity differs")
            aa, bb = finite(row["candidate_smse"]), finite(row["comparator_smse"])
            near(aa, vectors[panel, candidate][rep], "paired candidate loss")
            near(bb, vectors[panel, reference][rep], "paired reference loss")
            difference = aa - factor * bb
            near(row["margin_difference"], difference, "paired margin adjustment")
            left.append(aa)
            right.append(bb)
            differences.append(difference)
        delta_mean, delta_sd = mean(differences), sd(differences)
        se = delta_sd / math.sqrt(N)
        upper = delta_mean + critical * se
        result = {"contrast_id": cid, "panel": panel, "candidate_key": candidate, "comparator_key": reference,
            "comparator_factor": factor, "candidate_mean_smse": mean(left), "comparator_mean_smse": mean(right),
            "point_relative_gain_percent": gain(mean(left), mean(right)), "margin_difference_mean": delta_mean,
            "margin_difference_sd": delta_sd, "margin_difference_se": se, "critical_t": critical,
            "simultaneous_one_sided_upper": upper, "criterion_established": upper < 0,
            "strict_D_win_repetitions": sum(a < b for a, b in zip(left, right)), "macro_suffix": suffix}
        for published in published_sets:
            row = published[cid,]
            for name in ("comparator_factor", "candidate_mean_smse", "comparator_mean_smse", "point_relative_gain_percent",
                         "margin_difference_mean", "margin_difference_sd", "margin_difference_se", "critical_t",
                         "simultaneous_one_sided_upper"):
                near(row[name], result[name], cid + "/" + name)
            require(boolean(row["criterion_established"]) == result["criterion_established"], "Primary decision differs")
            require(row["panel"] == panel and row["candidate_key"] == candidate and row["comparator_key"] == reference,
                    "Primary labels differ")
        primary.append(result)
    require(summary["criteria_established"] == audit["criteria_established"] == sum(r["criterion_established"] for r in primary),
            "Primary-family decision count differs")
    laws = []
    for law in LAWS:
        by_key = {key: [mean(tasks[str(rep), law, str(noise), key]["smse"] for noise in (0, 1)) for rep in range(N)]
                  for key in ("standard_independent__K", "independent__K", "independent__R")}
        standard, joint, rbf = [mean(by_key[key]) for key in ("standard_independent__K", "independent__K", "independent__R")]
        laws.append({"law": law, "standard_K_smse": standard, "joint_K_smse": joint, "joint_R_smse": rbf,
                     "joint_K_vs_standard_K_gain_percent": gain(joint, standard),
                     "joint_K_vs_joint_R_gain_percent": gain(joint, rbf),
                     "n_repetitions": N, "scope": "descriptive equal-noise per-law means; no per-law primary inference"})
    selection_rows = []
    selected = tables["selection_by_task.csv"]
    frequencies = indexed(tables["scaling_selection_frequencies.csv"], ("panel", "view", "arm", "input_scaling"), "scaler frequencies")
    for panel in ("nonlinear", "linear"):
        for arm in ("P", "K", "R"):
            rows = [r for r in selected if r["view"] == "independent" and r["arm"] == arm and r["law"] in PANELS[panel]]
            require(len(rows) == N * 2 * len(PANELS[panel]), "Incomplete independent selection summary")
            counts = Counter(r["input_scaling"] for r in rows)
            require(set(counts) <= set(SCALINGS), "Undeclared selected scaler")
            for scaling in SCALINGS:
                published = frequencies[panel, "independent", arm, scaling]
                require(int(published["task_count"]) == counts[scaling] and int(published["total_tasks"]) == len(rows),
                        "Selection frequency differs from task records")
                near(published["fraction"], counts[scaling] / len(rows), "selected-scaler fraction")
            caps = [int(r["cap"]) for r in rows]
            require(set(caps) <= set(CAPS), "Undeclared selected cap")
            selection_rows.append({"panel": panel, "arm": arm, "tasks": len(rows),
                "counts": {scaling: counts[scaling] for scaling in SCALINGS}, "median_cap": statistics.median(caps)})
    return primary, panels, laws, selection_rows


def number(value, *, decimals=None):
    if value is None:
        return "n/a"
    value = finite(value)
    if decimals is not None:
        return f"{value:.{decimals}f}"
    text = f"{value:.6g}"
    if "e" in text:
        coefficient, exponent = text.split("e")
        return r"\ensuremath{" + coefficient + r"\times10^{" + str(int(exponent)) + "}}"
    return text


def table(columns, headers, rows):
    lines = [r"\begin{tabular}{@{}" + columns + "@{}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def make_figure(path, law_rows):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/kahm-joint-report-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    for ax, row in zip(axes.flat, law_rows):
        values = [row["standard_K_smse"], row["joint_K_smse"]]
        ax.bar([0, 1], values, width=.58, color=["#89939f", "#087f8c"], zorder=3)
        ax.set_xticks([0, 1], ["Standard only", "Joint selection"])
        ax.set_ylabel("Mean standardized MSE")
        ax.set_title(LAW_LABELS[row["law"]], loc="left", fontweight="bold")
        ax.set_ylim(0, max(values) * 1.31 if max(values) else 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        ax.grid(axis="y", color="#dce0e4", linewidth=.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for position, value in enumerate(values):
            ax.annotate(f"{value:.5g}", (position, value), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    fig.suptitle("Independently selected K: same-cohort comparison", fontsize=11, fontweight="bold", y=.995)
    fig.text(.5, .015, "32 repetitions; equal noise-level weighting; separate y scales; lower is better.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .96), h_pad=1.5, w_pad=1.7)
    fig.savefig(path / "independent_K_joint_vs_standard.pdf", metadata={"Title": "Same-cohort joint versus standard-only KAHM", "Creator": Path(__file__).name})
    fig.savefig(path / "independent_K_joint_vs_standard.png", dpi=220)
    plt.close(fig)


def build(staging, primary, panels, laws, selections, summary, audit, resources):
    macros = {}

    def macro(name, value, *, decimals=None):
        display = value if isinstance(value, str) else number(value, decimals=decimals)
        macros[name] = {"value": value, "display": display}

    for name, value in (("JSRepetitions", N), ("JSTasks", TASKS), ("JSCandidates", 15),
                        ("JSScalings", 3), ("JSCaps", 5), ("JSFamilySize", 6),
                        ("JSCriteriaEstablished", summary["criteria_established"]),
                        ("JSAuditChecks", audit["checks"]), ("JSSolverRecoveries", audit["numerical_solver_recoveries"])):
        macro(name, value)
    macros["JSAuditChecks"]["display"] = f"{audit['checks']:,}".replace(",", "{,}")
    macro("JSCriticalT", primary[0]["critical_t"], decimals=6)
    primary_rows = []
    for row in primary:
        suffix = row["macro_suffix"]
        relative_gain = row["point_relative_gain_percent"]
        gain_decimals = 4 if abs(relative_gain) < 1 else 2
        macro("JSGain" + suffix, relative_gain, decimals=gain_decimals)
        macro("JSRelativeLossChange" + suffix, -relative_gain, decimals=gain_decimals)
        macro("JSGap" + suffix, row["margin_difference_mean"])
        macro("JSUpper" + suffix, row["simultaneous_one_sided_upper"])
        macro("JSEstablished" + suffix, "Yes" if row["criterion_established"] else "No")
        if suffix in ("JointKNonlinear", "JointKLinear"):
            comparison = r"$K_{\mathrm{joint}}/K_{\mathrm{standard}}$"
        else:
            comparison = "$" + row["candidate_key"].split("__")[1] + "/" + row["comparator_key"].split("__")[1] + "$"
        primary_rows.append([row["panel"].title(), comparison,
                             r"3\% superiority" if row["comparator_factor"] == .97 else r"5\% noninferiority",
                             number(relative_gain, decimals=gain_decimals),
                             number(row["simultaneous_one_sided_upper"]), "Yes" if row["criterion_established"] else "No"])
    (staging / "primary_table.tex").write_text(table("lllrrc", ["Panel", "Contrast", "Criterion", r"Gain (\%)", "$U$", "Met"], primary_rows))
    aggregate_rows = []
    view_macros = {"shared": "Shared", "independent": "Independent", "standard_shared": "StandardShared", "standard_independent": "StandardIndependent"}
    for view in VIEWS:
        for arm in ARMS:
            for panel in ("nonlinear", "linear"):
                macro("JSLoss" + view_macros[view] + panel.title() + arm, finite(panels[panel, f"{view}__{arm}"]["mean_smse"]))
            nl, li = panels["nonlinear", f"{view}__{arm}"], panels["linear", f"{view}__{arm}"]
            aggregate_rows.append(["Standard only" if view.startswith("standard_") else "Joint",
                "P-shared" if view.endswith("shared") and not view.endswith("independent") else "Own",
                arm, number(nl["mean_smse"]), number(li["mean_smse"])])
    (staging / "aggregate_table.tex").write_text(table("lllrr", ["Search", "Geometry", "Arm", "Nonlinear SMSE", "Linear SMSE"], aggregate_rows))
    selection_rows = []
    for row in selections:
        cells = []
        for scaling in SCALINGS:
            count = row["counts"][scaling]
            cells.append(f"{count} ({100 * count / row['tasks']:.1f}" + r"\%)")
        selection_rows.append([row["panel"].title(), row["arm"], *cells, number(row["median_cap"])])
    (staging / "selection_table.tex").write_text(table("llrrrr", ["Panel", "Arm", "Standard", "Min--max", "Tanh", "Median cap"], selection_rows))
    descriptive_rows = []
    for row in laws:
        scaling_gain = row["joint_K_vs_standard_K_gain_percent"]
        reconstruction_gain = row["joint_K_vs_joint_R_gain_percent"]
        descriptive_rows.append([LAW_LABELS[row["law"]], number(row["standard_K_smse"]), number(row["joint_K_smse"]),
                                number(row["joint_R_smse"]), number(scaling_gain, decimals=4 if abs(scaling_gain) < 1 else 2),
                                number(reconstruction_gain, decimals=4 if abs(reconstruction_gain) < 1 else 2)])
    (staging / "descriptive_table.tex").write_text(table("lrrrrr", ["Law", r"$K_{\mathrm{standard}}$", r"$K_{\mathrm{joint}}$", r"$R_{\mathrm{joint}}$",
        r"Joint/standard (\%)", r"K/R joint (\%)"], descriptive_rows))
    resource_index = indexed(resources, ("resource",), "resource summary")
    for name, key, statistic, divisor in (
        ("JSFitCpuMinutes", "fit/fit_all_scalers_caps_four_arms/cpu_seconds", "sum", 60),
        ("JSFitWallSumMinutes", "fit/fit_all_scalers_caps_four_arms/wall_seconds", "sum", 60),
        ("JSFitMedianSeconds", "fit/fit_all_scalers_caps_four_arms/wall_seconds", "median", 1),
        ("JSQueryWallSumMinutes", "prediction/wall_seconds", "sum", 60)):
        require((key,) in resource_index, "Required measured resource is absent: " + key)
        macro(name, finite(resource_index[key,][statistic]) / divisor, decimals=2)
    (staging / "numbers.tex").write_text("% Generated from completed, independently audited joint-selection evidence.\n" +
        "\n".join("\\newcommand{\\" + name + "}{" + item["display"] + "}" for name, item in macros.items()) + "\n")
    make_figure(staging, laws)
    metrics = {"reporting_only": True, "primary": primary, "per_law_descriptive": laws,
               "independent_scaler_selections": selections, "macros": macros,
               "figure": {"file": "independent_K_joint_vs_standard.pdf", "values": laws,
                          "aggregation": "Equal noise-level means within each repetition, then equal mean over 32 repetitions",
                          "separate_y_scales": True, "uncertainty_intervals": "none; descriptive mean comparison"},
               "audit_scope": audit["scope"], "limitations": summary["limitations"]}
    (staging / "report_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (staging / "README.md").write_text(
        "# Joint-selection manuscript displays\n\n"
        "Generated only after complete analysis and a passed matching independent audit. "
        "Table fragments have no captions or labels. `numbers.tex` defines the JS macro namespace; "
        "`report_metrics.json` records every macro value and the figure data. "
        "For each primary suffix, `JSRelativeLossChange` is the negative of `JSGain`, "
        "so a positive value denotes a percentage loss increase. Percentage gains and loss changes "
        "use four decimals below 1% in magnitude and two decimals otherwise.\n\n"
        "- `primary_table.tex`: all six simultaneous primary criteria. Gain is 100(1-candidate/reference); "
        "U is the one-sided margin-adjusted paired-t upper bound, with six-comparison Bonferroni adjustment.\n"
        "- `aggregate_table.tex`: all 16 forecast means on both panels. P-shared geometry uses P's "
        "validation choice; Own uses each arm's validation choice.\n"
        "- `selection_table.tex`: independent P/K/R scaler counts and percentages (192 nonlinear "
        "tasks or 64 linear tasks) and median selected cap, not median effective local rank.\n"
        "- `descriptive_table.tex`: equal-noise per-law independent-selection comparisons; no per-law primary inference.\n"
        "- `independent_K_joint_vs_standard.pdf` and `.png`: exact same-cohort means, with separate y scales.\n\n"
        "The figure, per-law table and selection frequencies are descriptive. The two declared "
        "joint-versus-standard K primary contrasts remain in the six-comparison family. "
        "The asset builder verifies CSV/reporting arithmetic; raw-array replay belongs to the bound independent audit.\n")
    return macros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, default=HERE / "generated/joint_selection")
    args = parser.parse_args()
    study, output = args.study_root.resolve(), args.output.resolve()
    require(not output.exists(), "Output exists; use a new reporting-only directory")
    require(not output.is_relative_to(study) and not study.is_relative_to(output),
            "Reporting output must not overlap scientific evidence")
    inputs = Inputs()
    inputs.raw(Path(__file__), "reporting/build_joint_selection_assets.py")
    config, publication, summary, audit, tables = load_evidence(study, inputs)
    primary, panels, laws, selections = recompute(tables, summary, audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joint-selection-report-", dir=output.parent))
    try:
        macros = build(staging, primary, panels, laws, selections, summary, audit, tables["resource_summary.csv"])
        outputs = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in sorted(staging.iterdir()) if p.is_file()}
        manifest = {"schema": "kahm-joint-selection-manuscript-assets-v1", "status": "complete", "reporting_only": True,
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": publication["source_commit"],
            "source_tree_sha": publication["source_tree_sha"], "inputs": inputs.records, "outputs": outputs,
            "primary_recomputed": primary, "macro_names": sorted(macros),
            "checks": {"all_analysis_table_hashes_and_row_counts": True, "matching_passed_independent_audit": True,
                "task_to_repetition_and_panel_aggregation": True, "six_primary_bounds_from_paired_repetitions": True,
                "primary_matches_csv_json_and_independent_audit": True, "per_law_and_selection_arithmetic": True},
            "scope": "Reporting arithmetic and metadata verification; no model fits or raw scientific array replay.",
            "environment": {"python": platform.python_version()}}
        (staging / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        require(not output.exists(), "Output appeared during build; refusing overwrite")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"status": "complete", "output": str(output), "primary_criteria_established": summary["criteria_established"],
                      "payload_files": len(outputs), "macros": len(macros)}))


if __name__ == "__main__":
    main()
