"""Build publication assets from the independent audited report; never fit models.

Synthetic intervals have the frozen 18-contrast Bonferroni family. Archive
points, collection contrasts and leave-one-collection-out means are descriptive
only; the code does not construct pseudo-replicated confidence intervals from
windows or series. All prespecified methods and all four collections are shown.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kahm-target-geometry-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullLocator
import numpy as np

from . import data
from .analysis import write_csv

ROOT = Path(__file__).resolve().parent
LAW_LABELS = {"nonlinear_ar": "Nonlinear AR", "vanderpol": "Van der Pol", "duffing": "Duffing"}
SOURCE_LABELS = {"australian_electricity_halfhourly": "Australian electricity",
                 "bitcoin_daily": "Bitcoin indicators", "pedestrian_hourly": "Pedestrian counts",
                 "kdd_air_quality_hourly": "KDD air quality"}
BASE_LABELS = {"ridge": "Raw ridge", "raw_rbf": "Raw RBF", "naive": "Persistence / seasonal naive",
               "joint_kahm_ridge": "Joint raw + KAHM"}
GATE_LABELS = {"kahm": "KAHM", "affine": "Affine", "locality": "Locality", "rbf": "RBF local"}
STAGE_LABELS = {"fixed_is": "Fixed, IS", "fixed_oof": "Fixed, OOF", "learned_oof": "Learned, OOF"}
COLORS = {"kahm": "#0072B2", "affine": "#D55E00", "locality": "#009E73", "rbf": "#CC79A7"}
MARKERS = {"fixed_is": "o", "fixed_oof": "s", "learned_oof": "^"}


def label(method):
    if method in BASE_LABELS:
        return BASE_LABELS[method]
    stage, gate = method.rsplit("_", 1)
    return f"{STAGE_LABELS[stage]} {GATE_LABELS[gate]}"


def tex(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def number(value, digits=4, scale=1.0):
    if value is None or not np.isfinite(value):
        return r"\textemdash"
    value *= scale
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 10 ** (-digits)):
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return f"{value:.{digits}f}"


def table(path, columns, rows, caption, table_label, aligns=None, size="small"):
    aligns = aligns or "l" + "r" * (len(columns) - 1)
    lines = ["% Generated only from audited target-geometry predictions; no fitting.",
             r"\begin{table}[!htbp]", rf"\centering\{size}\setlength{{\tabcolsep}}{{4pt}}",
             rf"\caption{{{caption}}}", rf"\label{{{table_label}}}",
             rf"\begin{{tabular}}{{{aligns}}}", r"\toprule", " & ".join(columns) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def source_balanced_sensitivity(summary):
    sources = list(dict.fromkeys(row["source_id"] for row in summary["natural_sources"]))
    records = []
    for omitted in sources:
        for method in summary["methods"]:
            selected = [row for row in summary["natural_sources"]
                        if row["source_id"] != omitted and row["method"] == method]
            complete = all(row["scaled_mse"] is not None for row in selected)
            records.append({"omitted_source": omitted, "method": method,
                            "source_count": len(selected),
                            "scaled_mse": float(np.mean([row["scaled_mse"] for row in selected])) if complete else None})
    for row in records:
        ridge = next(item["scaled_mse"] for item in records
                     if item["omitted_source"] == row["omitted_source"] and item["method"] == "ridge")
        row["ratio_to_ridge"] = (row["scaled_mse"] / ridge
                                 if row["scaled_mse"] is not None and ridge is not None and ridge > 0 else None)
    return records


def build_tables(summary, output):
    methods, laws = summary["methods"], summary["config"]["synthetic"]["laws"]
    synthetic = {(row["method"], row["law"]): row for row in summary["synthetic"]}
    table(output / "synthetic_main.tex", ["Procedure"] + [LAW_LABELS[law] for law in laws],
          [[tex(label(method))] + [number(synthetic[method, law]["mean_mse"], 3, 1000) for law in laws]
           for method in methods],
          r"Fresh synthetic repetitions: mean test MSE $\times10^3$ over all 12 independent repetitions per law, with equal episode, origin and horizon weights. Every declared procedure is shown. A dash denotes an incomplete full roster; baseline fallbacks retain their actual forecasts. Differences are assessed by the 18 prespecified paired intervals in Table~\ref{tab:target-synthetic-intervals}, not by ranking these means.",
          "tab:target-synthetic-main")
    rows = []
    for contrast in summary["primary_contrasts"]:
        lo, hi = contrast.get("lower"), contrast.get("upper")
        interval = f"[{number(lo, 3, 1000)}, {number(hi, 3, 1000)}]" if lo is not None else r"\textemdash"
        rows.append([LAW_LABELS[contrast["law"]], tex(label(contrast["control"])),
                     tex(label(contrast["candidate"])), number(contrast["mean_difference"], 3, 1000), interval])
    table(output / "synthetic_intervals.tex", ["Law", "Control", "Candidate", r"$\Delta\times10^3$", "Family interval"], rows,
          r"The complete family of 18 prespecified synthetic comparisons. $\Delta$ is control minus candidate test MSE; positive values favor the candidate. Intervals use the paired repetition differences, Student's $t_{11}$ quantile, and a two-sided Bonferroni allocation of $0.05/18$. Family coverage is approximate because normality of repetition differences is not guaranteed. No claim joins this family to earlier experiments.",
          "tab:target-synthetic-intervals", aligns="lllrr", size="scriptsize")
    natural = {row["method"]: row for row in summary["natural"]}
    table(output / "archive_main.tex",
          ["Procedure", "Outputs", "Scaled MSE", r"Change (\%)", "Seconds", "Fallbacks"],
          [[tex(label(method)), f"{natural[method]['completed']}/{natural[method]['expected']}",
            number(natural[method]["scaled_mse"]), number(natural[method]["relative_change_vs_ridge_pct"], 2),
            number(natural[method]["mean_wall_seconds"], 3), natural[method]["fallbacks"]] for method in methods],
          r"The new-collection archive experiment: equal-collection means of equal-series scaled MSE and charged standalone wall seconds, covering four collections and three series per collection. Scaling uses training-only seasonal-difference MSE (first-difference fallback). Change is relative to raw ridge; negative is favorable. Fallbacks predict raw ridge and retain attempted computation. Missing tasks make the affected full-roster means undefined. These are descriptive results, with no population confidence intervals or independence claims for adjacent origins.",
          "tab:target-archive-main")
    sources = list(dict.fromkeys(row["source_id"] for row in summary["natural_sources"]))
    source_rows = {(row["method"], row["source_id"]): row for row in summary["natural_sources"]}
    short_sources = {"australian_electricity_halfhourly": "Electricity", "bitcoin_daily": "Bitcoin",
                     "pedestrian_hourly": "Pedestrian", "kdd_air_quality_hourly": "Air quality"}
    table(output / "archive_sources.tex", ["Procedure"] + [short_sources[source] for source in sources],
          [[tex(label(method))] + [number(source_rows[method, source]["scaled_mse"]) for source in sources]
           for method in methods],
          "Scaled MSE by collection, averaging its three declared series. Collections receive equal weights in the aggregate; neither series nor repeated origins increase the collection count. No confidence intervals are attached to these four descriptive collection means.",
          "tab:target-archive-sources")
    rows = []
    for task in summary["tasks"]:
        if task["phase"] != "natural":
            continue
        arrays = data.load_natural(task)
        rows.append([SOURCE_LABELS[task["source_id"]], tex(task["series_id"]), task["context"], task["horizon"],
                     "/".join(str(len(arrays["X_" + split])) for split in ("train", "validation", "test")),
                     number(task["loss_scale"], 3)])
    table(output / "archive_windows.tex", ["Collection", "Series", "$L$", "$H$", "Train/val/test", "Loss scale"], rows,
          r"Exact natural-series roster, selected before forecasts by a salted hash among training-prefix-eligible candidates. $L$ is history length and $H$ the complete forecast path length. Chronological boundaries are 60/20/20; windows with missing inputs or targets are omitted without imputation. At most 512/64/64 windows are retained by the frozen deterministic sampling rule. The loss scale is computed before validation. Bitcoin series are network indicators, not prices.",
          "tab:target-archive-windows", aligns="llrrrr", size="scriptsize")
    rows = []
    for phase, label_phase in (("synthetic", "Synthetic"), ("natural", "Archive")):
        diagnostics = [row for row in summary["metric_diagnostics"] if row["phase"] == phase]
        valid = [row for row in diagnostics if row["metric_training_objective_reduction"] is not None]
        transport = [row["donor_final_disagreement_mse"] / row["oof_residual_mse"]
                     for row in diagnostics if row["donor_final_disagreement_mse"] is not None and row["oof_residual_mse"] > 0]
        rows.append([label_phase, len(diagnostics), sum(bool(row["metric_optimizer_success"]) for row in diagnostics),
                     sum(bool(row["metric_fallback"]) for row in diagnostics),
                     number(float(np.median([row["metric_training_objective_reduction"] for row in valid])) if valid else None),
                     number(float(np.median(transport)) if transport else None)])
    table(output / "diagnostics.tex", ["Phase", "Tasks", "Converged", "Metric fallback", r"Median $J_0-J_*$", r"Median $D/R_{\rm OOF}$"], rows,
          r"Training diagnostics, not independent evidence of test benefit. $J$ is the regularized Gaussian residual-regression proxy shared by all geometric arms. A finite improved iterate may be retained at an iteration limit, separately from optimizer convergence. $D$ is mean squared disagreement between donor and final ridge predictions on the same correction rows; $R_{\rm OOF}$ is their OOF residual MSE. Medians summarize observed tasks and are descriptive. Exact objectives, warnings, weight ratios, missing diagnostics and residual identities are retained in the report CSV files.",
          "tab:target-diagnostics", size="scriptsize")
    sensitivity = source_balanced_sensitivity(summary)
    indexed = {(row["method"], row["omitted_source"]): row for row in sensitivity}
    table(output / "leave_one_source_out.tex", ["Procedure"] + [short_sources[source] for source in sources],
          [[tex(label(method))] + [number(indexed[method, source]["ratio_to_ridge"], 3) for source in sources]
           for method in methods],
          r"Descriptive sensitivity to collection composition: each column omits the named collection and recomputes the mean scaled loss over the remaining three collections, divided by the corresponding raw-ridge loss. Values below one favor the procedure. Every omission is shown; none is selected as the primary result. These ratios are not independent replications, confidence intervals, or additional confirmatory tests.",
          "tab:target-leave-one-source-out")
    write_csv(output / "leave_one_source_out.csv", sensitivity)
    rows = []
    for method in methods:
        if method in BASE_LABELS:
            continue
        row = [tex(label(method))]
        for phase in ("synthetic", "natural"):
            choices = [entry for entry in summary["selection"] if entry["method"] == method and entry["phase"] == phase]
            row.append(" / ".join(str(entry["count"]) for entry in choices))
            row.append(str(choices[0]["fallbacks"]) if choices else r"\textemdash")
        rows.append(row)
    table(output / "selection_frequency.tex", ["Procedure", "Synthetic gamma counts", "Fallback", "Archive gamma counts", "Fallback"], rows,
          r"Validation-selected correction strengths, with counts ordered as $\gamma=0,0.25,0.5,1$. A zero correction is the exact deployed ridge predictor. A failed arm is recorded separately as a baseline fallback, not as an observed zero-strength validation selection. Selection counts summarize tasks and do not supply an additional inference sample.",
          "tab:target-selection", size="scriptsize")


def save_figure(fig, output, stem):
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=230, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def ratio_ticks(axis, values):
    values = [value for value in values if value is not None and np.isfinite(value) and value > 0]
    if not values:
        return
    low, high = min(values), max(values)
    if low == high:
        low, high = low * 0.9, high * 1.1
    if high / low < 4:
        ticks = MaxNLocator(nbins=4).tick_values(low, high)
        axis.set_major_locator(FixedLocator(ticks[ticks > 0]))
    else:
        axis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=6))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_locator(NullLocator())


def build_figures(summary, output):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    contrasts = summary["primary_contrasts"]
    labels = ["Ridge → learned KAHM", "Fixed → learned KAHM", "IS → OOF KAHM",
              "Affine → KAHM", "Locality → KAHM", "RBF local → KAHM"]
    # These three panels are placed at 6.5-inch manuscript width. Twelve-point
    # source tick labels remain about 7.2 pt after that reduction.
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.3), sharey=True, constrained_layout=True)
    for ax, law in zip(axes, summary["config"]["synthetic"]["laws"]):
        selected = [row for row in contrasts if row["law"] == law]
        for index, row in enumerate(selected):
            if row.get("mean_difference") is None:
                continue
            mean, lower, upper = (row[key] * 1000 for key in ("mean_difference", "lower", "upper"))
            color = "#0072B2" if lower > 0 else "#D55E00" if upper < 0 else "#666666"
            ax.errorbar(mean, index, xerr=[[mean - lower], [upper - mean]], fmt="o", color=color,
                        capsize=3, markersize=5, elinewidth=1.3)
        ax.axvline(0, color="#777777", lw=0.8, ls="--")
        ax.set_title(LAW_LABELS[law], fontsize=14)
        ax.set_xlabel("Control − candidate MSE (×10³)", fontsize=12)
        ax.set_yticks(range(len(labels)), labels, fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
    axes[0].invert_yaxis()
    fig.suptitle("Paired synthetic effects: 18 Bonferroni-adjusted Student intervals\nPositive favors the candidate; 12 independent repetitions per law", fontsize=13)
    save_figure(fig, output, "synthetic_contrasts")

    natural = {row["method"]: row for row in summary["natural"]}
    fig, ax = plt.subplots(figsize=(7.1, 4.8), constrained_layout=True)
    for method in summary["methods"]:
        row = natural[method]
        if row["wall_ratio_vs_ridge"] is None or row["relative_change_vs_ridge_pct"] is None:
            continue
        x, y = row["wall_ratio_vs_ridge"], 1 + row["relative_change_vs_ridge_pct"] / 100
        if method in BASE_LABELS:
            marker = {"ridge": "*", "raw_rbf": "D", "naive": "P", "joint_kahm_ridge": "X"}[method]
            ax.scatter(x, y, marker=marker, color="#222222", s=70 if method == "ridge" else 40,
                       label=BASE_LABELS[method], zorder=4)
        else:
            stage, gate = method.rsplit("_", 1)
            ax.scatter(x, y, marker=MARKERS[stage], color=COLORS[gate], s=42,
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(1, color="#777777", lw=0.8, ls="--")
    ax.axvline(1, color="#777777", lw=0.8, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="Charged standalone wall time / ridge (log scale)",
           ylabel="Collection-balanced scaled MSE / ridge (log scale)",
           title="Archive accuracy and computation: descriptive four-collection means")
    ratio_ticks(ax.xaxis, [row["wall_ratio_vs_ridge"] for row in natural.values()])
    ratio_ticks(ax.yaxis, [1 + row["relative_change_vs_ridge_pct"] / 100 for row in natural.values()
                          if row["relative_change_vs_ridge_pct"] is not None])
    ax.grid(color="#ededed", lw=0.5)
    baseline_handles, baseline_labels = ax.get_legend_handles_labels()
    handles = baseline_handles + [Line2D([], [], marker="o", ls="", color=color, label=GATE_LABELS[gate])
                                 for gate, color in COLORS.items()]
    handles += [Line2D([], [], marker=marker, ls="", color="#777777", label=STAGE_LABELS[stage])
                for stage, marker in MARKERS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=3,
              frameon=False, columnspacing=1.2, handletextpad=0.5)
    save_figure(fig, output, "archive_accuracy_cost")

    sources = list(dict.fromkeys(row["source_id"] for row in summary["natural_sources"]))
    indexed = {(row["source_id"], row["method"]): row for row in summary["natural_sources"]}
    methods = ["joint_kahm_ridge", "fixed_is_kahm", "fixed_oof_kahm", "learned_oof_kahm",
               "learned_oof_affine", "learned_oof_locality", "learned_oof_rbf"]
    all_ratios = [indexed[source, method]["scaled_mse"] / indexed[source, "ridge"]["scaled_mse"]
                  for source in sources for method in methods
                  if indexed[source, method]["scaled_mse"] is not None
                  and indexed[source, "ridge"]["scaled_mse"] is not None
                  and indexed[source, "ridge"]["scaled_mse"] > 0]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, constrained_layout=True)
    for ax, source in zip(axes.flat, sources):
        ridge = indexed[source, "ridge"]["scaled_mse"]
        for index, method in enumerate(methods):
            loss = indexed[source, method]["scaled_mse"]
            if loss is None or ridge is None or ridge <= 0:
                continue
            color = "#222222" if method == "joint_kahm_ridge" else COLORS[method.rsplit("_", 1)[1]]
            ax.plot(loss / ridge, index, "o", color=color, ms=4)
        ax.axvline(1, color="#777777", lw=0.8, ls="--")
        ax.set_yticks(range(len(methods)), [label(method) for method in methods])
        ax.invert_yaxis()
        ax.set_title(SOURCE_LABELS[source])
        ax.set_xscale("log")
        ratio_ticks(ax.xaxis, all_ratios)
        ax.set_xlabel("Scaled MSE / ridge (log scale)")
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
    fig.suptitle("Per-collection descriptive effects; no population uncertainty intervals", fontsize=10)
    save_figure(fig, output, "archive_source_effects")


def build_numbers(summary, output):
    natural = {row["method"]: row for row in summary["natural"]}
    learned = natural["learned_oof_kahm"]
    values = {"TGTaskCount": summary["expected_tasks"], "TGCompletedTaskCount": summary["completed_tasks"],
              "TGForecastCount": summary["audited_predictions"], "TGMethodCount": len(summary["methods"]),
              "TGFallbackCount": summary["fallback_predictions"],
              "TGPrimaryPositive": sum(row["decision"] == "candidate_better" for row in summary["primary_contrasts"]),
              "TGPrimaryNegative": sum(row["decision"] == "candidate_worse" for row in summary["primary_contrasts"]),
              "TGPrimaryUnresolved": sum(row["decision"] == "unresolved" for row in summary["primary_contrasts"]),
              "TGNaturalLearnedMSE": number(learned["scaled_mse"]),
              "TGNaturalRidgeMSE": number(natural["ridge"]["scaled_mse"]),
              "TGNaturalChangePercent": number(learned["relative_change_vs_ridge_pct"], 2),
              "TGNaturalCostRatio": number(learned["wall_ratio_vs_ridge"], 1)}
    lines = ["% Generated target-geometry numbers; change is signed, negative favorable."]
    lines += [rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()]
    (output / "numbers.tex").write_text("\n".join(lines) + "\n")


def build(reports, output):
    reports, output = Path(reports), Path(output)
    summary = json.loads((reports / "summary.json").read_text())
    if summary["source_hashes"]["roster"] != data.digest(data.ROSTER_PATH):
        raise ValueError("Current prepared roster differs from the audited report")
    output.mkdir(parents=True, exist_ok=True)
    build_tables(summary, output)
    build_figures(summary, output)
    build_numbers(summary, output)
    metadata = {"summary_sha256": data.digest(reports / "summary.json"), "complete": summary["complete"],
                "scope": summary["scope"],
                "files": {path.name: data.digest(path) for path in sorted(output.iterdir())
                          if path.is_file() and path.name != "assets.json"}}
    data.write_json(output / "assets.json", metadata)
    print(json.dumps({"output": str(output), "files": len(metadata["files"]), "complete": summary["complete"]}, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "generated")
    args = parser.parse_args()
    build(args.reports, args.output)


if __name__ == "__main__":
    main()
