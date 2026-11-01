#!/usr/bin/env python3
"""Render publication figures from the completed, frozen analysis tables.

This reporting module does not read terminal prediction arrays, select models,
or alter the scientific analysis. Percentage error bars are linear display
rescalings of the declared paired SMSE-difference intervals, not ratio CIs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kahm_preprocessing_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
CAPS = (1, 2, 4, 8, 12, 20)
REPETITIONS = 24
COLORS = {
    "standard": "#6B7280", "minmax": "#D97706", "tanh": "#1D65A6",
    "kahm": "#1D65A6", "rbf": "#D97706", "matched_rbf": "#737373",
    "policy": "#26816A", "neutral": "#64748B", "negative": "#A34E4E",
}
SCALER_LABELS = {"standard": "Standardization", "minmax": "Min–max", "tanh": "Tanh(z)"}
METHODS = (
    "ridge", "raw_rbf", "raw_hgb", "persistence", "common", "kahm_departure",
    "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy",
)
METHOD_LABELS = {
    "ridge": "Ridge", "raw_rbf": "Raw inputs + Nyström RBF", "raw_hgb": "Histogram gradient boosting",
    "persistence": "Persistence", "common": "Common geometry", "kahm_departure": "Bounded KAHM departure",
    "rbf_departure": "Bounded RBF departure", "kahm_unbounded": "Unbounded KAHM departure",
    "rbf_unbounded": "Unbounded RBF departure", "nonkahm_policy": "Selected non-KAHM policy",
}
SECONDARY_PAIRS = ("ss", "mm", "tt", "tm", "mt")
PAIR_LABELS = {
    "ss": "Standard / standard", "mm": "Min–max / min–max", "tt": "Tanh / tanh",
    "tm": "Tanh / min–max", "mt": "Min–max / tanh",
}
COMMON_SCALERS = {"ss": "standard", "mm": "minmax", "tt": "tanh_z_1", "tm": "tanh_z_1", "mt": "minmax"}
PRIMARY_ORDER = (
    "selected_KAHM_vs_nonKAHM_policy", "selected_KAHM_vs_selected_RBF",
    "tanh_KAHM_vs_standard_KAHM", "linear_selected_KAHM_vs_ridge",
)
PRIMARY_LABELS = {
    "selected_KAHM_vs_nonKAHM_policy": "Selected KAHM vs non-KAHM policy\n3% superiority",
    "selected_KAHM_vs_selected_RBF": "Selected KAHM vs selected RBF\n3% superiority",
    "tanh_KAHM_vs_standard_KAHM": "Tanh vs standard KAHM\n3% superiority",
    "linear_selected_KAHM_vs_ridge": "Selected KAHM vs ridge (linear)\n5% noninferiority",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_table(path, columns):
    table = pd.read_csv(path)
    require(set(columns).issubset(table.columns), "Missing exported columns in " + path.name)
    return table


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 12, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "axes.edgecolor": "#9CA3AF", "text.color": "#17202A",
        "axes.labelcolor": "#17202A", "xtick.color": "#374151", "ytick.color": "#374151",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    })


def one_row(table, **conditions):
    selected = table
    for column, value in conditions.items():
        selected = selected.loc[selected[column] == value]
    require(len(selected) == 1, "Expected exactly one aggregate row for " + repr(conditions))
    return selected.iloc[0]


def mean_interval(row):
    require(int(row["n_records"]) == REPETITIONS, "Aggregate interval must use 24 repetitions")
    mean = float(row["smse"])
    lower = float(row["smse_lower95_descriptive"])
    upper = float(row["smse_upper95_descriptive"])
    require(np.isfinite([mean, lower, upper]).all() and lower <= mean <= upper, "Invalid descriptive interval")
    return mean, lower, upper


def vector(table, view, method, *, panel="nonlinear", mode="calibrated"):
    selected = table.loc[(table["panel"] == panel) & (table["view"] == view)
                         & (table["method"] == method) & (table["mode"] == mode)].sort_values("repetition")
    require(len(selected) == REPETITIONS and selected["repetition"].tolist() == list(range(REPETITIONS)),
            "Incomplete repetition vector: " + view + "/" + method)
    values = selected["smse"].to_numpy(dtype=float)
    require(np.isfinite(values).all() and np.all(values >= 0), "Invalid repetition loss")
    return values


def footer(fig, text):
    fig.text(0.015, 0.025, text, va="bottom", ha="left", fontsize=7.3, color="#4B5563", linespacing=1.45)


def save_figure(fig, output, stem, caption, values):
    png, pdf = output / (stem + ".png"), output / (stem + ".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.14,
                metadata={"Title": caption.split(".")[0], "Subject": caption,
                          "Creator": "KAHM preprocessing–dimension study: frozen-table reporting"})
    plt.close(fig)
    return {"stem": stem, "caption": caption, "files": [
        {"filename": png.name, "sha256": digest(png), "bytes": png.stat().st_size},
        {"filename": pdf.name, "sha256": digest(pdf), "bytes": pdf.stat().st_size},
    ], "plotted_values": values}


def figure_dimension(panel, output):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.23)
    xs = np.arange(len(CAPS))
    selected_x = len(CAPS) + 0.65
    records = []
    ax.axvspan(len(CAPS) - 0.25, selected_x + 0.43, facecolor="#F3F4F6", zorder=0)
    for scaler in ("standard", "minmax", "tanh"):
        means, lowers, uppers = [], [], []
        for cap in CAPS:
            row = one_row(panel, panel="nonlinear", view=f"candidate__{scaler}__cap{cap}",
                          mode="calibrated", method="kahm_departure")
            mean, lower, upper = mean_interval(row)
            means.append(mean); lowers.append(lower); uppers.append(upper)
            records.append({"scaler": scaler, "cap": cap, "selection": "fixed_cap",
                            "mean_smse": mean, "lower95": lower, "upper95": upper})
        ax.plot(xs, means, "o-", color=COLORS[scaler], markersize=4, linewidth=1.6,
                label=SCALER_LABELS[scaler], zorder=3)
        ax.fill_between(xs, lowers, uppers, color=COLORS[scaler], alpha=0.08, linewidth=0)
        selected = one_row(panel, panel="nonlinear", view="primary__" + scaler,
                           mode="calibrated", method="kahm_departure")
        mean, lower, upper = mean_interval(selected)
        offset = {"standard": -0.12, "minmax": 0, "tanh": 0.12}[scaler]
        ax.errorbar(selected_x + offset, mean, yerr=[[mean - lower], [upper - mean]], fmt="D",
                    color=COLORS[scaler], markersize=5, capsize=2.5, elinewidth=1.0, zorder=4)
        records.append({"scaler": scaler, "cap": "B-selected", "selection": "independently_selected_cap",
                        "mean_smse": mean, "lower95": lower, "upper95": upper})
    ax.set_xticks([*xs, selected_x], [*[str(cap) for cap in CAPS], "B-selected\ncap"])
    ax.set_xlim(-0.3, selected_x + 0.45)
    ax.set_ylabel("Nonlinear aggregate SMSE  (lower is better)")
    ax.set_xlabel("Requested native dimension cap")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3, frameon=False, borderaxespad=0)
    fig.suptitle("Preprocessing and local dimension", x=0.12, y=0.98, ha="left", fontsize=13)
    footer(fig, "Means and descriptive 95% intervals over 24 independent repetitions; C-calibrated forecasts.\n"
                "Caps 12 and 20 have identical admissible dimensions here (history width 12). The separate right column allows task-specific cap choices.")
    return save_figure(fig, output, "figure_01_preprocessing_dimension",
        "Preprocessing and native local-dimension cap in the primary complete-head procedure. Curves show the three diagonal scaling arms. "
        "Shaded bands and selected-cap error bars are descriptive unadjusted 95% intervals of mean SMSE over 24 independent repetition aggregates. "
        "The B-selected marker aggregates task-specific choices and is therefore not positioned at a single numerical cap. "
        "Caps 12 and 20 are numerically equivalent at the fixed ambient history dimension 12. All forecasts are C-calibrated.", records)


def figure_procedures(panel, output):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    fig.subplots_adjust(left=0.35, right=0.97, top=0.88, bottom=0.21)
    records = []
    for position, method in enumerate(METHODS):
        y = len(METHODS) - 1 - position
        row = one_row(panel, panel="nonlinear", family="primary", context="selected",
                      mode="calibrated", method=method)
        mean, lower, upper = mean_interval(row)
        color = (COLORS["kahm"] if method == "kahm_departure" else COLORS["policy"] if method == "nonkahm_policy"
                 else COLORS["rbf"] if method == "rbf_departure" else COLORS["neutral"])
        if method in ("kahm_departure", "nonkahm_policy"):
            ax.axhspan(y - 0.36, y + 0.36, facecolor=color, alpha=0.045, zorder=0)
        ax.errorbar(mean, y, xerr=[[mean - lower], [upper - mean]], fmt="o", color=color,
                    markersize=5.2 if method in ("kahm_departure", "nonkahm_policy") else 4.2,
                    capsize=2.6, elinewidth=1.2, zorder=3)
        records.append({"method": method, "mean_smse": mean, "lower95": lower, "upper95": upper})
    ax.set_yticks(list(reversed(range(len(METHODS)))), [METHOD_LABELS[method] for method in METHODS])
    ax.set_ylim(-0.7, len(METHODS) - 0.3)
    ax.set_xlabel("Nonlinear aggregate SMSE  (lower is better)")
    ax.grid(axis="x", zorder=0)
    fig.suptitle("Complete forecasting procedures", x=0.04, y=0.98, ha="left", fontsize=13)
    footer(fig, "Each method uses its declared B selection and C calibration; the non-KAHM policy selects its method on C.\n"
                "Intervals describe each mean over 24 repetitions. Formal claims use paired, multiplicity-adjusted comparison tests.")
    return save_figure(fig, output, "figure_02_complete_procedures",
        "Primary selected-context comparison of ten complete forecasting procedures on the nonlinear panel. "
        "Dots and error bars are mean SMSE and descriptive unadjusted 95% intervals across 24 independent repetition aggregates. "
        "These mean intervals are not paired tests of procedure differences. The paired primary comparisons use the separate declared four-member family.", records)


def figure_fixed_common(repetitions, output):
    fig, ax = plt.subplots(figsize=(8.1, 5.1))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.23)
    methods = ("kahm", "rbf", "matched_rbf")
    labels = {"kahm": "KAHM, own cap", "rbf": "RBF, own cap", "matched_rbf": "RBF, KAHM cap"}
    markers = {"kahm": "o", "rbf": "s", "matched_rbf": "^"}
    offsets = {"kahm": 0.21, "rbf": 0.0, "matched_rbf": -0.21}
    critical = float(student_t.ppf(0.975, REPETITIONS - 1))
    records = []
    for position, pair in enumerate(SECONDARY_PAIRS):
        y = len(SECONDARY_PAIRS) - 1 - position
        baseline = vector(repetitions, "secondary__" + COMMON_SCALERS[pair], "shared_bias")
        denominator = float(baseline.mean())
        require(denominator > 0, "Cannot display a percentage gain over a zero-loss baseline")
        for method in methods:
            loss = vector(repetitions, "secondary__" + pair, method)
            improvement = baseline - loss
            mean = float(improvement.mean())
            se = float(improvement.std(ddof=1) / np.sqrt(REPETITIONS))
            lower, upper = mean - critical * se, mean + critical * se
            factor = 100.0 / denominator
            value, low, high = factor * mean, factor * lower, factor * upper
            ax.errorbar(value, y + offsets[method], xerr=[[value - low], [high - value]],
                        fmt=markers[method], color=COLORS[method], markersize=4.7, capsize=2.2,
                        elinewidth=1.05, label=labels[method] if position == 0 else None, zorder=3)
            records.append({"pair": pair, "method": method, "mean_baseline_smse": denominator,
                            "mean_method_smse": float(loss.mean()), "mean_improvement_smse": mean,
                            "improvement_lower95_smse": lower, "improvement_upper95_smse": upper,
                            "display_gain_percent": value, "display_lower95_percent": low, "display_upper95_percent": high})
    ax.axvline(0, color="#4B5563", linewidth=0.85, linestyle="--", zorder=1)
    ax.set_yticks(list(reversed(range(len(SECONDARY_PAIRS)))), [PAIR_LABELS[pair] for pair in SECONDARY_PAIRS])
    ax.set_ylim(-0.65, len(SECONDARY_PAIRS) - 0.35)
    ax.set_xlabel("Gain beyond shared bias (% of mean baseline SMSE)  → better")
    ax.grid(axis="x", zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, frameon=False, borderaxespad=0,
              columnspacing=1.25, handletextpad=0.5)
    fig.suptitle("Incremental correction with the common forecast fixed", x=0.04, y=0.98, ha="left", fontsize=12.7)
    fig.text(0.04, 0.91, "Rows: common / departure preprocessing", fontsize=8.4, color="#4B5563")
    footer(fig, "Separate P/R residual-fitting architecture; methods in each row share the same common forecast and bias.\n"
                "Paired, unadjusted 95% SMSE-difference intervals are rescaled by the observed mean baseline loss; these are not risk-ratio confidence intervals.")
    return save_figure(fig, output, "figure_03_fixed_common_increment",
        "Secondary fixed-common experiment: incremental forecasting gain beyond the identical common-plus-shared-bias baseline within each row. "
        "KAHM and RBF use independently B-selected departure caps; matched RBF uses KAHM's selected cap with its own penalty and C multiplier. "
        "The five rows use the common/departure preprocessing pairs shown. Intervals are descriptive unadjusted 95% paired-t intervals for SMSE improvement, "
        "linearly rescaled by the observed mean baseline SMSE. They are not confidence intervals for an unknown risk ratio. "
        "The P/R sample allocation differs from the primary complete-head experiment; the rankings must not be pooled.", records)


def bool_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    require(str(value).lower() in ("true", "false"), "Invalid established flag")
    return str(value).lower() == "true"


def figure_primary_bounds(primary, output):
    require(set(primary["contrast_id"]) == set(PRIMARY_ORDER) and len(primary) == 4, "Primary contrast roster differs")
    records = []
    for identifier in PRIMARY_ORDER:
        row = one_row(primary, contrast_id=identifier)
        require(int(row["n_repetitions"]) == 24 and int(row["family_size"]) == 4
                and float(row["family_alpha"]) == 0.05, "Primary family differs")
        denominator = float(row["comparator_mean_smse"])
        require(denominator > 0, "Primary comparator has zero mean loss")
        mean, lower = float(row["margin_gap_mean"]), float(row["simultaneous_one_sided_lower"])
        established = bool_value(row["criterion_established"])
        require(established == (lower > 0), "Primary established flag disagrees with lower bound")
        require(np.isfinite([mean, lower]).all() and lower <= mean + 1e-14, "Invalid primary bound")
        records.append({"contrast_id": identifier, "mean_margin_gap_smse": mean,
                        "simultaneous_lower_smse": lower, "comparator_mean_smse": denominator,
                        "display_mean_margin_gap_percent": 100.0 * mean / denominator,
                        "display_lower_percent": 100.0 * lower / denominator,
                        "criterion_established": established})
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    fig.subplots_adjust(left=0.38, right=0.97, top=0.82, bottom=0.26)
    positions = list(reversed(range(4)))
    for y, record in zip(positions, records):
        color = COLORS["policy"] if record["criterion_established"] else COLORS["negative"]
        mean, lower = record["display_mean_margin_gap_percent"], record["display_lower_percent"]
        ax.plot([lower, mean], [y, y], color=color, linewidth=1.8, zorder=2)
        ax.plot(lower, y, marker="|", color=color, markersize=11, markeredgewidth=1.6, linestyle="none", zorder=3)
        ax.plot(mean, y, marker="D", color=color, markersize=5.3, linestyle="none", zorder=3)
        ax.annotate(f"{lower:+.2f}", (lower, y), xytext=(0, -15), textcoords="offset points",
                    ha="center", va="top", fontsize=7.4, color=color)
    ax.axvline(0, color="#374151", linewidth=0.9, linestyle="--", zorder=1)
    extrema = [0.0, *[r["display_mean_margin_gap_percent"] for r in records], *[r["display_lower_percent"] for r in records]]
    width = max(max(extrema) - min(extrema), 1.0)
    ax.set_xlim(min(extrema) - 0.11 * width, max(extrema) + 0.13 * width)
    ax.set_ylim(-0.65, 3.45)
    ax.set_yticks(positions, [PRIMARY_LABELS[identifier] for identifier in PRIMARY_ORDER])
    ax.set_xlabel("Margin gap (% of mean comparator SMSE)  → favors the declared claim")
    ax.grid(axis="x", zorder=0)
    ax.legend(handles=[
        Line2D([0], [0], color="#475569", marker="|", linestyle="none", markersize=10, label="Simultaneous lower bound"),
        Line2D([0], [0], color="#475569", marker="D", linestyle="none", markersize=5, label="Mean margin gap"),
    ], loc="lower left", bbox_to_anchor=(-0.02, 1.06), ncol=2, frameon=False, borderaxespad=0)
    fig.suptitle("Four prespecified performance claims", x=0.04, y=0.98, ha="left", fontsize=13)
    footer(fig, "A claim is established only when its simultaneous lower bound is positive. The declared practical margin is already incorporated.\n"
                "One-sided Bonferroni paired-t bounds: family α = 0.05, four comparisons, 24 independent repetitions. Labels give the displayed lower bound.\n"
                "Dividing the absolute margin gap by the observed mean comparator loss is a display rescaling; the inferential decision is unchanged.")
    return save_figure(fig, output, "figure_04_primary_margin_bounds",
        "Four prespecified claims with their mean margin gaps and simultaneous one-sided lower bounds. "
        "For nonlinear comparisons the gap is 0.97 times comparator loss minus candidate loss; for linear noninferiority it is 1.05 times ridge loss minus KAHM loss. "
        "The paired-t bounds use 24 independent repetition aggregates and Bonferroni family allocation 0.05/4. "
        "The plotted values divide absolute SMSE margin gaps by the observed mean comparator SMSE and multiply by 100 for legibility. "
        "This display rescaling does not change the sign or the claim decision and is not a risk-ratio confidence interval. No upper confidence bound is shown.", records)


def build(analysis, output):
    analysis, output = Path(analysis).resolve(), Path(output).resolve()
    summary_path = analysis / "analysis_summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary["status"] == "completed" and summary["task_count"] == 192
            and summary["independent_repetitions"] == 24, "Analysis must be complete before plotting")
    inputs = {
        "panel_summary.csv": ["panel", "family", "context", "view", "mode", "method", "n_records", "smse",
                              "smse_lower95_descriptive", "smse_upper95_descriptive"],
        "repetition_losses.csv": ["panel", "view", "mode", "method", "repetition", "smse"],
        "primary_contrasts.csv": ["contrast_id", "n_repetitions", "family_size", "family_alpha", "comparator_mean_smse",
                                  "margin_gap_mean", "simultaneous_one_sided_lower", "criterion_established"],
    }
    frames, hashes = {}, {}
    for name, columns in inputs.items():
        path = analysis / name
        hashes[name] = digest(path)
        require(summary["tables"][name]["sha256"] == hashes[name], "Exported analysis table changed: " + name)
        frames[name] = read_table(path, columns)
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    figures = [
        figure_dimension(frames["panel_summary.csv"], output),
        figure_procedures(frames["panel_summary.csv"], output),
        figure_fixed_common(frames["repetition_losses.csv"], output),
        figure_primary_bounds(frames["primary_contrasts.csv"], output),
    ]
    manifest = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_summary_sha256": digest(summary_path), "source_table_sha256": hashes,
        "reporting_script_sha256": digest(Path(__file__)),
        "matplotlib_version": matplotlib.__version__, "png_resolution_dpi": 300,
        "scientific_scope": "Display of frozen analysis results; no model or arm selection from test outcomes.",
        "figures": figures,
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    captions = ["# Figure captions", ""]
    for index, figure in enumerate(figures, 1):
        captions.extend([f"## Figure {index}", "", figure["caption"], ""])
    (output / "FIGURE_CAPTIONS.md").write_text("\n".join(captions))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "analysis")
    parser.add_argument("--output", type=Path, default=ROOT / "reporting" / "figures")
    args = parser.parse_args()
    result = build(args.analysis, args.output)
    print(json.dumps({"status": result["status"], "figures": len(result["figures"]), "output": str(args.output.resolve())}))
