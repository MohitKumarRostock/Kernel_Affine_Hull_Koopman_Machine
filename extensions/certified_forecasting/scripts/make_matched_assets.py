#!/usr/bin/env python3
"""Create paper tables and the complete dimension figure from audited results.

    python scripts/make_matched_assets.py --analysis results/matched_geometry_study/analysis

Development smoke outcomes are deliberately refused as manuscript evidence.
The figure uses all seven tasks, all three heads and every predefined retained
dimension. Negative relative MSE means KAHM improves on the indicated geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("harmonic", "quasiperiodic", "nonlinear_ar", "logistic", "vanderpol", "duffing", "pendigits")
LABELS = {"harmonic": "Harmonic", "quasiperiodic": "Quasiperiodic", "nonlinear_ar": "Nonlinear AR",
          "logistic": "Logistic", "vanderpol": "Van der Pol", "duffing": "Duffing", "pendigits": "PenDigits"}
HEADS = ("nested", "frobenius", "koopman")
HEAD_LABELS = ("Block direct", "Frobenius direct", "Rollout")


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def primary_table(comparisons, n_seeds):
    lookup = {(r["task"], r["head"], r["alternative"]): r for r in comparisons
              if r["analysis"] == "primary" and r["horizon"] == "all"}
    if len(lookup) != len(TASKS) * len(HEADS) * 2:
        raise ValueError("Incomplete primary comparison table")
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"& \multicolumn{2}{c}{Block direct} & \multicolumn{2}{c}{Frobenius direct} & \multicolumn{2}{c}{Rollout} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Task & RBF & Affine & RBF & Affine & RBF & Affine \\", r"\midrule"]
    for task in TASKS:
        cells = []
        for head in HEADS:
            for alternative in ("rbf", "affine"):
                row = lookup[(task, head, alternative)]
                change = float(row["relative_mse_change_percent"])
                wins = int(row["n_seeds_improved"])
                cells.append(f"${change:+.1f}$ ({wins})")
        out.append(LABELS[task] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Matched geometry follow-up: percentage change in test MSE of KAHM relative to the indicated alternative, within each head. Negative values favor KAHM. Parentheses give the number of " + str(n_seeds) +
            r" fitting seeds with lower KAHM MSE. Preprocessing, history and retained dimension are selected separately for each geometry and head by validation MSE only. Percentages use ratios of MSE averaged over horizons and fitting seeds within each task; PenDigits first pools coordinate losses and equally weights eligible writers. Synthetic calibration comparisons use new episodes and a separate multiplicity allocation. The PenDigits comparisons reuse the same cohort. All test effects in this table are descriptive.}",
            r"\label{tab:matched-geometry-primary}", r"\end{table}", ""]
    return "\n".join(out)


def mse_table(model_rows):
    lookup = {(r["task"], r["head"], r["geometry"]): r for r in model_rows
              if r["analysis"] == "primary" and r["horizon"] == "all"}
    if len(lookup) != len(TASKS) * len(HEADS) * 3:
        raise ValueError("Incomplete absolute MSE table")
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
           r"& \multicolumn{3}{c}{Block direct} & \multicolumn{3}{c}{Frobenius direct} & \multicolumn{3}{c}{Rollout} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
           r"Task & KAHM & RBF & Affine & KAHM & RBF & Affine & KAHM & RBF & Affine \\", r"\midrule"]
    for task in TASKS:
        cells = [f"{float(lookup[(task, head, geometry)]['mse_mean']):.5f}"
                 for head in HEADS for geometry in ("kahm", "rbf", "affine")]
        out.append(LABELS[task] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Absolute test MSE for all nine matched geometry--head families in the follow-up. Each cell averages the predefined horizons and five fitting seeds after validation-only selection of preprocessing, history and retained dimension within that family. PenDigits additionally pools both selected coordinate losses with equal eligible-writer weights on the same evaluation cohort. Corresponding seed variation, per-horizon losses and exact selected candidates are retained in the repository.}",
            r"\label{tab:matched-geometry-mse}", r"\end{table}", ""]
    return "\n".join(out)


def dimension_figure(comparisons, destination):
    lookup = {(r["task"], r["head"], int(r["dimension"]), r["alternative"]): r for r in comparisons
              if r["analysis"] == "fixed_dimension" and r["horizon"] == "all"}
    dimensions = (2, 6, 14, 30)
    if len(lookup) != len(TASKS) * len(HEADS) * len(dimensions) * 2:
        raise ValueError("Incomplete fixed-dimension figure")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    # At the JMLR six-inch text width this leaves about one inch for a caption
    # within the 8.5-inch text height, while the smallest type remains >7 pt.
    fig, axes = plt.subplots(len(TASKS), len(HEADS), figsize=(6.6, 8.2), sharex=True, sharey="row")
    for ri, task in enumerate(TASKS):
        for ci, head in enumerate(HEADS):
            ax = axes[ri, ci]
            ax.axhline(0, color="0.45", linewidth=0.7, linestyle="--", zorder=1)
            for alternative, color, marker, label in (
                    ("rbf", "#286EB0", "o", "KAHM relative to RBF"),
                    ("affine", "#C47A13", "s", "KAHM relative to affine")):
                values = [float(lookup[(task, head, dim, alternative)]["relative_mse_change_percent"])
                          for dim in dimensions]
                ax.plot(range(len(dimensions)), values, color=color, marker=marker,
                        linewidth=1.25, markersize=3.2, label=label, zorder=2)
            ax.grid(axis="y", color="0.91", linewidth=0.55)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(range(len(dimensions)), [str(d) for d in dimensions])
            ax.tick_params(length=2.5, width=0.65)
            if ri == 0:
                ax.set_title(HEAD_LABELS[ci], pad=6)
            if ci == 0:
                ax.set_ylabel(LABELS[task] + "\nMSE change (%)", labelpad=7)
            if ri == len(TASKS) - 1:
                ax.set_xlabel("Retained features")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.0),
               ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.15, right=0.987, top=0.947, bottom=0.055, hspace=0.36, wspace=0.12)
    fig.savefig(destination, metadata={"Title": "Matched KAHM geometry: complete fixed-dimension effects",
                                       "Subject": "Validation-selected history and preprocessing at each fixed retained dimension; descriptive test effects"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--paper", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    summary = json.loads((args.analysis / "summary.json").read_text())
    audit = json.loads((args.analysis / "audit.json").read_text())
    if summary["allow_smoke"] or audit["status"] != "passed":
        raise ValueError("Manuscript assets require a complete audited production study")
    if audit["seeds"] != [5701, 5702, 5703, 5704, 5705]:
        raise ValueError("Unexpected fitting-seed schedule")
    comparisons, model_rows = rows(args.analysis / "taskwise_comparisons.csv"), rows(args.analysis / "taskwise_models.csv")
    tables, figures = args.paper / "tables", args.paper / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (tables / "matched_geometry_primary.tex").write_text(primary_table(comparisons, len(audit["seeds"])))
    (tables / "matched_geometry_mse.tex").write_text(mse_table(model_rows))
    values = {"MatchedCandidates": summary["n_candidates"],
              "MatchedComparisons": summary["n_primary_taskwise_comparisons"],
              "MatchedImproved": summary["n_primary_mean_improved"],
              "MatchedDegraded": summary["n_primary_mean_degraded"],
              "MatchedCalComparisons": summary["n_calibration_comparisons"],
              "MatchedCalCertified": summary["n_positive_calibration_lower_bounds"]}
    (tables / "matched_geometry_numbers.tex").write_text(
        "% Generated from the complete audited matched-geometry follow-up.\n" +
        "\n".join("\\newcommand{\\" + key + "}{" + f"{value:,}".replace(",", "{,}") + "}" for key, value in values.items()) + "\n")
    dimension_figure(comparisons, figures / "matched_geometry_dimensions.pdf")
    print(json.dumps({"tables": ["matched_geometry_primary.tex", "matched_geometry_mse.tex", "matched_geometry_numbers.tex"],
                      "figure": "matched_geometry_dimensions.pdf", "values": values}, indent=2))


if __name__ == "__main__":
    main()
