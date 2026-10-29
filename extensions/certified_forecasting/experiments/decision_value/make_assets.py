#!/usr/bin/env python3
"""Generate the decision-study table and figure from retained records."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/decision_value/fresh"
PAPER = ROOT / "paper"


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    summary = load_csv(RESULTS / "decision_summary.csv")
    by_key = {(r["regime"], int(r["n_calibration"]), r["lower_method"]): r for r in summary}
    sizes = sorted({int(r["n_calibration"]) for r in summary})
    table = [r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{rccccc}", r"\toprule",
        r" & \multicolumn{3}{c}{Repairable regime} & \multicolumn{2}{c}{Irreducible regime} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"$n$ & Upper & \multicolumn{2}{c}{Minimum cost $5$} & \multicolumn{2}{c}{All classes excluded} \\",
        r" & acceptance & Unconditional & Stratified & Unconditional & Stratified \\",
        r"\midrule"]
    for n in sizes:
        repair_u = by_key[("repairable", n, "unconditional")]
        repair_s = by_key[("repairable", n, "stratified")]
        noise_u = by_key[("irreducible", n, "unconditional")]
        noise_s = by_key[("irreducible", n, "stratified")]
        values = [repair_u["upper_acceptances"], repair_u["minimum_cost_certificates"],
                  repair_s["minimum_cost_certificates"], noise_u["infeasibility_certificates"],
                  noise_s["infeasibility_certificates"]]
        table.append(f"{n:,}" + " & " + " & ".join(values) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Decision counts out of 100 independent repetitions in each constructed regime, at the prespecified RMSE tolerance $0.30$. Upper acceptance certifies a fitted predictor; equality of the cost endpoints certifies the least feasible cost among the three predefined classes. The unconditional and stratified methods share the same upper endpoint and fitted model. Every irreducible-regime upper decision is unresolved. One failure budget $0.05$ covers all five calibration sizes and all reported endpoints within each regime and repetition.}",
        r"\label{tab:decision-outcomes}", r"\end{table}", ""]
    (PAPER / "tables/decision_outcomes.tex").write_text("\n".join(table))
    rows = load_csv(RESULTS / "endpoints.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.labelsize": 9.5, "axes.titlesize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.25), sharey=True)
    series = [("lower_unconditional", "#a65628", "o", "Unconditional lower"),
              ("lower_stratified", "#20639b", "s", "Stratified lower")]
    handles = []
    for ax, regime, prefix, title in zip(axes, ("repairable", "irreducible"), (2, 3),
            ("(a) Repairable: cost 3", "(b) Irreducible: cost 5")):
        for field, color, marker, label in series:
            values = [np.array([float(row[field]) for row in rows if row["regime"] == regime
                and int(row["prefix"]) == prefix and int(row["n_calibration"]) == n]) for n in sizes]
            median = [np.median(v) for v in values]
            line, = ax.plot(sizes, median, color=color, marker=marker, markersize=3.8,
                           linewidth=1.5, label=label)
            ax.fill_between(sizes, [v.min() for v in values], [v.max() for v in values],
                            color=color, alpha=.16, linewidth=0)
            if regime == "repairable":
                handles.append(line)
        if regime == "repairable":
            values = [np.array([float(row["upper"]) for row in rows if row["regime"] == regime
                and int(row["prefix"]) == 3 and int(row["n_calibration"]) == n]) for n in sizes]
            line, = ax.plot(sizes, [np.median(v) for v in values], color="#27844b", marker="^",
                           markersize=3.8, linewidth=1.5, label="Repaired cost-5 upper")
            ax.fill_between(sizes, [v.min() for v in values], [v.max() for v in values],
                            color="#27844b", alpha=.16, linewidth=0)
            handles.append(line)
        threshold = ax.axhline(.09, color="#555555", linestyle="--", linewidth=1.1,
                              label="MSE tolerance 0.09")
        ax.set(xscale="log", ylim=(-.005, .255), title=title, xlabel="Calibration episodes")
        ax.set_xlim(sizes[0]*.65, sizes[-1]*1.6)
        ax.set_xticks(sizes, ["256", "1,024", "4,096", "16,384", "65,536"])
        ax.tick_params(axis="x", rotation=35)
        ax.set_yticks([0, .05, .10, .15, .20, .25])
        ax.grid(axis="y", alpha=.2, linewidth=.5)
    axes[0].set_ylabel("MSE certificate endpoint")
    handles.append(threshold)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.53, 0), columnspacing=1.2, handlelength=1.7)
    fig.subplots_adjust(left=.105, right=.99, bottom=.38, top=.88, wspace=.13)
    for extension in ("pdf", "png"):
        target = PAPER / ("figures/decision_endpoints." + extension)
        temporary = target.with_name("decision_endpoints.tmp." + extension)
        options = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {"dpi": 200}
        fig.savefig(temporary, **options)
        temporary.replace(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
