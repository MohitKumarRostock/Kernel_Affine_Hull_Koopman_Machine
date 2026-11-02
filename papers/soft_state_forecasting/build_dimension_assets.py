#!/usr/bin/env python3
"""Build manuscript displays from the sealed dimension_forecasting analysis.

Run with --study-root PATH_TO_REPO/extensions/dimension_forecasting.  In a
repository checkout that extension is detected automatically.  Compilation
uses the committed generated files and needs neither study states nor raw
forecast archives.  This builder performs no model fitting or selection.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kahm_manuscript_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ["ridge", "raw_rbf", "raw_hgb", "persistence", "common",
           "kahm_departure", "rbf_departure", "kahm_unbounded",
           "rbf_unbounded", "nonkahm_policy"]
GEOMETRY = ["common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded"]
LABEL = dict(zip(METHODS, ["Raw ridge", "Raw RBF", "Boosted trees", "Persistence",
    "Common features", "Bounded KAHM departure", "Bounded RBF departure",
    "Unbounded KAHM departure", "Unbounded RBF departure", "Selected non-KAHM"]))
LAWS = ["nonlinear_ar", "vanderpol", "duffing", "linear_ar"]
CAPS = [1, 2, 4, 8, 12, 20]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "generated/dimension_forecasting")
    args = parser.parse_args()
    study = args.study_root
    if study is None:
        study = next((p / "extensions/dimension_forecasting" for p in HERE.parents
                      if (p / "extensions/dimension_forecasting/analysis/analysis_summary.json").is_file()), None)
    if study is None:
        parser.error("Pass --study-root for the sealed extensions/dimension_forecasting study.")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "dimension_manuscript_assets_v1", "inputs": {},
                "outputs": {}, "numeric_cells": [], "source_freeze_sha256": sha(study / "freeze.json")}
    summary = json.loads((study / "analysis/analysis_summary.json").read_text())
    assert summary["status"] == "completed" and summary["task_count"] == 192
    assert summary["primary_comparisons_established"] == 3
    assert summary["integrity"]["seal_sha256"]["freeze.json"] == manifest["source_freeze_sha256"]
    manifest["inputs"]["analysis/analysis_summary.json"] = sha(study / "analysis/analysis_summary.json")

    def rows(name):
        path = study / "analysis" / (name + ".csv")
        digest = sha(path)
        assert summary["tables"][name + ".csv"]["sha256"] == digest
        manifest["inputs"]["analysis/" + name + ".csv"] = digest
        with path.open(newline="") as stream:
            result = list(csv.DictReader(stream))
        assert len(result) == summary["tables"][name + ".csv"]["rows"]
        return result

    def one(records, **query):
        result = [r for r in records if all(r[k] == str(v) for k, v in query.items())]
        assert len(result) == 1, (query, len(result))
        return result[0]

    def num(value, precision=6, key=""):
        text = f"{float(value):.{precision}f}"
        manifest["numeric_cells"].append({"key": key, "value": float(value), "display": text})
        return text

    def write(name, content):
        path = out / name
        path.write_text(content + "\n")
        manifest["outputs"][name] = sha(path)

    def table(name, columns, header, lines):
        write(name, "\n".join([r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule",
            header + r" \\", r"\midrule", *[line + r" \\" for line in lines], r"\bottomrule", r"\end{tabular}"]))

    panel = rows("panel_summary")
    primary = rows("primary_contrasts")
    secondary = rows("secondary_contrasts")
    condition = rows("condition_summary")
    horizon = rows("panel_horizon_summary")
    geometry = rows("geometry_summary_by_task")
    local = rows("local_dimensions")
    resources = rows("resource_summary")
    caps = rows("cap_selections")
    gammas = rows("C_gamma_choices")
    policies = rows("C_policy_choices")
    bounds = rows("bound_checks")
    features = rows("feature_checks")
    decomp = rows("decomposition_summary")
    task_resources = rows("task_resources")

    def loss(method, view="context_selected", panel_name="nonlinear", mode="calibrated"):
        return float(one(panel, panel=panel_name, view=view, method=method, mode=mode)["smse"])

    lines = [" & ".join([LABEL[m], num(loss(m), 6, "selected nonlinear " + m),
              num(loss(m, panel_name="linear"), 6, "selected linear " + m)]) for m in METHODS]
    table("aggregate_table.tex", "lrr", "Procedure & Nonlinear SMSE & Linear SMSE", lines)
    lines = []
    for r in primary:
        label = LABEL[r["comparator"]] + (" (linear NI)" if r["panel"] == "linear" else "")
        lines.append(" & ".join([label, num(r["candidate_mean_smse"], 5), num(r["comparator_mean_smse"], 5),
            num(r["point_relative_gain_percent"], 2), num(r["margin_gap_mean"], 6),
            num(r["simultaneous_one_sided_lower"], 6), "Yes" if r["criterion_established"] == "True" else "No"]))
    table("primary_table.tex", "lrrrrrc", r"Comparator & $L_K$ & $L_m$ & Gain (\%) & $\bar d$ & Lower bound & Established", lines)

    lines = []
    for m in METHODS:
        vals = [one(condition, law=law, noise_level=eta, view="context_selected", method=m, mode="calibrated")["smse"]
                for law in LAWS for eta in [0.0, 0.1]]
        lines.append(" & ".join([LABEL[m], *[num(x, 5) for x in vals]]))
    table("law_noise_table.tex", "lrrrrrrrr", r"Procedure & \multicolumn{2}{c}{Nonlinear AR} & \multicolumn{2}{c}{Van der Pol} & \multicolumn{2}{c}{Duffing} & \multicolumn{2}{c}{Linear AR} \\"
          + "\n" + r" & $\eta=0$ & $0.1$ & $\eta=0$ & $0.1$ & $\eta=0$ & $0.1$ & $\eta=0$ & $0.1$", lines)

    variants = [("Fixed cap 2", "context_historical2"), *[("Native cap " + str(c), "native" + str(c)) for c in CAPS],
                ("B-selected cap", "context_selected")]
    lines = []
    for label, view in variants:
        k = loss("kahm_departure", view)
        gain = 100 * (1 - k / loss("kahm_departure", "context_historical2"))
        lines.append(" & ".join([label, num(k, 6), num(loss("common", view), 6), num(loss("rbf_departure", view), 6), num(gain, 2)]))
    table("dimension_table.tex", "lrrrr", r"Dimension rule & KAHM & Common & RBF & KAHM gain (\%)", lines)

    lines = []
    for view, label in [("matched", "KAHM-selected bank"), ("native20", "Native cap 20")]:
        r = one(decomp, panel="nonlinear", view=view, mode="calibrated")
        lines.append(" & ".join([label, *[num(r[k], 6) for k in ["complete_change", "common_change", "departure_increment_change", "new_departure_increment"]]]))
    table("decomposition_table.tex", "lrrrr", r"Bank rule & Total change & Common change & Increment change & $L_K-L_0$", lines)

    lines = []
    for cid, label in [("historical2", "Fixed cap 2"), *[("native" + str(c), "Native cap " + str(c)) for c in CAPS]]:
        b = [r for r in geometry if r["candidate_id"] == cid]
        q = [r for r in local if r["candidate_id"] == cid]
        ranks = [int(r["effective_rank"]) for r in q]
        widths = [int(r["common_width"]) for r in b]
        lines.append(" & ".join([label, f"{min(ranks)}--{max(ranks)}", num(np.mean(ranks), 3),
                     str(sum(r == 12 for r in ranks)), f"{min(widths)}--{max(widths)}", "168"]))
    table("geometry_table.tex", "lrrrrr", r"Bank rule & Rank range & Mean rank & Rank 12 & Common width & Departure width", lines)

    lines = []
    for m in GEOMETRY:
        for pname, is_linear in [("Nonlinear", False), ("Linear", True)]:
            counts = collections.Counter(int(r["selected_cap"]) for r in caps if r["method"] == m and (r["law"] == "linear_ar") == is_linear)
            lines.append(" & ".join([LABEL[m], pname, *[str(counts[c]) for c in CAPS]]))
    table("cap_table.tex", "llrrrrrr", r"Procedure & Panel & 1 & 2 & 4 & 8 & 12 & 20", lines)

    lines = []
    for m in METHODS:
        r = one(resources, context="selected", method=m)
        lines.append(" & ".join([LABEL[m], num(r["fit_and_selection_wall_seconds"], 3),
                 num(r["fit_and_selection_cpu_seconds"], 3), num(float(r["deployment_pickle_bytes"]) / 1024, 1),
                 num(float(r["deployment_array_bytes"]) / 1024, 1)]))
    for context, label in [("historical2", "KAHM, fixed cap 2"), ("native20", "KAHM, native cap 20")]:
        r = one(resources, context=context, method="kahm_departure")
        lines.append(" & ".join([label, num(r["fit_and_selection_wall_seconds"], 3), num(r["fit_and_selection_cpu_seconds"], 3),
            num(float(r["deployment_pickle_bytes"]) / 1024, 1), num(float(r["deployment_array_bytes"]) / 1024, 1)]))
    table("resource_table.tex", "lrrrr", r"Procedure & Wall (s) & CPU (s) & State (KiB) & Arrays (KiB)", lines)

    lines = []
    for m in GEOMETRY[:3]:
        fs = [r for r in features if r["method"] == m]
        bs = [r for r in bounds if r["method"] == m]
        fmax = max(float(r["maximum_norm"]) / float(r["radius"]) for r in fs)
        cmax = max(float(r["maximum_coefficient_bound_ratio"]) for r in bs)
        dmax = max(float(r["maximum_D_correction_bound_ratio"]) for r in bs)
        violations = sum(int(r["bound_violations"]) for r in fs) + sum(int(r["coefficient_bound_violations"]) + int(r["D_correction_bound_violations"]) for r in bs)
        lines.append(" & ".join([LABEL[m], num(fmax, 4), num(cmax, 4), num(dmax, 4), str(violations)]))
    table("bounds_table.tex", "lrrrr", "Procedure & Feature ratio & Coefficient ratio & Correction ratio & Violations", lines)

    selection = {(r["task_id"], r["method"]): r["selected_candidate"] for r in caps}
    policy = {r["task_id"]: r["selected_method"] for r in policies if r["context"] == "selected"}
    g = {(r["task_id"], r["candidate_id"], r["method"], int(r["horizon"])): float(r["gamma"]) for r in gammas}
    task_law = {r["task_id"]: r["law"] for r in task_resources}
    activation = {}
    lines = []
    for m in METHODS[1:]:
        counts = collections.Counter()
        active = {False: 0, True: 0}
        for task, law in task_law.items():
            method = policy[task] if m == "nonkahm_policy" else m
            candidate = selection[task, method] if method in GEOMETRY else "shared"
            for h in [1, 4, 12]:
                gamma = 0 if method == "ridge" else g[task, candidate, method, h]
                counts[gamma] += 1
                active[law == "linear_ar"] += int(gamma != 0)
        activation[m] = {"counts": dict(counts), "nonlinear": active[False], "linear": active[True]}
        lines.append(" & ".join([LABEL[m], str(active[False]), str(active[True]), *[str(counts[x]) for x in [0, .25, .5, 1]]]))
    table("activation_table.tex", "lrrrrrr", r"Procedure & NL active & Linear active & $\gamma=0$ & $1/4$ & $1/2$ & $1$", lines)

    lines = []
    policy_methods = [m for m in METHODS if m not in ["kahm_departure", "kahm_unbounded", "nonkahm_policy"]]
    for law in LAWS:
        for eta in [0.0, .1]:
            cnt = collections.Counter(r["selected_method"] for r in policies if r["context"] == "selected" and r["law"] == law and float(r["noise_level"]) == eta)
            name = {"nonlinear_ar": "Nonlinear AR", "vanderpol": "Van der Pol", "duffing": "Duffing", "linear_ar": "Linear AR"}[law]
            lines.append(" & ".join([name, str(eta), *[str(cnt[m]) for m in policy_methods]]))
    table("policy_table.tex", "lrrrrrrrr", r"Law & $\eta$ & Ridge & Raw RBF & Boosting & Persist. & Common & RBF & RBF unb.", lines)

    lines = []
    for label, view, m in [("KAHM, fixed cap 2", "context_historical2", "kahm_departure"),
                           ("KAHM, native cap 20", "context_native20", "kahm_departure"),
                           *[(LABEL[m], "context_selected", m) for m in METHODS]]:
        values = [one(horizon, panel="nonlinear", view=view, method=m, mode="calibrated", horizon=h)["smse"] for h in [1, 4, 12]]
        lines.append(" & ".join([label, *[num(x, 6) for x in values]]))
    table("horizon_table.tex", "lrrr", r"Procedure & Horizon 1 & Horizon 4 & Horizon 12", lines)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    cond_keys = [(law, eta) for law in LAWS for eta in [0.0, .1]]
    labels = [f"{dict(zip(LAWS,['Nonlinear AR','Van der Pol','Duffing','Linear AR']))[law]}, $\\eta={eta:g}$" for law, eta in cond_keys]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.85), sharey=True)
    y = np.arange(8)
    for ax, comp, title, color in zip(axes, ["rbf_departure", "nonkahm_policy"],
          ["Independently selected RBF departure", "C-selected non-KAHM procedure"], ["#245d84", "#a3482b"]):
        values = []
        for law, eta in cond_keys:
            kval = float(one(condition, law=law, noise_level=eta, view="context_selected", method="kahm_departure", mode="calibrated")["smse"])
            cval = float(one(condition, law=law, noise_level=eta, view="context_selected", method=comp, mode="calibrated")["smse"])
            values.append(100 * (1 - kval/cval))
        ax.axvline(0, color=".5", linewidth=.75)
        ax.axhline(5.5, color=".7", linewidth=.7, linestyle="--")
        ax.scatter(values, y, color=color, s=24, zorder=3)
        lo, hi = min(values), max(values)
        span = max(hi-lo, 1)
        ax.set_xlim(lo - .12*span, hi + .32*span)
        for yi, v in zip(y, values):
            ax.annotate(f"{v:.2f}%", (v, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_title(title, fontsize=9, pad=10)
        ax.set_xlabel("KAHM relative gain (%)")
        ax.grid(axis="x", color=".9", linewidth=.5)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.tight_layout(pad=.8, w_pad=1.6)
    fig.savefig(out / "law_noise_contrasts.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    manifest["outputs"]["law_noise_contrasts.pdf"] = sha(out / "law_noise_contrasts.pdf")

    primary_by = {r["comparator"]: r for r in primary}
    selected_secondary = one(secondary, contrast_id="selected_minus_historical_KAHM", panel="nonlinear", mode="calibrated")
    macros = {
        "DFGainCommon": num(primary_by["common"]["point_relative_gain_percent"], 2),
        "DFGainRBF": num(primary_by["rbf_departure"]["point_relative_gain_percent"], 2),
        "DFGainRawRBF": num(primary_by["raw_rbf"]["point_relative_gain_percent"], 2),
        "DFGainHGB": num(primary_by["raw_hgb"]["point_relative_gain_percent"], 2),
        "DFWorsePolicy": num(-float(primary_by["nonkahm_policy"]["point_relative_gain_percent"]), 2),
        "DFDimensionGain": num(selected_secondary["relative_gain_percent"], 2),
        "DFMaxErrorRatio": num(summary["maximum_selected_error_ratio"], 4),
        "DFKActiveNonlinear": str(activation["kahm_departure"]["nonlinear"]),
        "DFKActiveLinear": str(activation["kahm_departure"]["linear"]),
        "DFKFitSeconds": num(one(resources, context="selected", method="kahm_departure")["fit_and_selection_wall_seconds"], 3),
        "DFRFitSeconds": num(one(resources, context="selected", method="rbf_departure")["fit_and_selection_wall_seconds"], 3),
        "DFPeakProcessMiB": num(max(float(r["peak_rss_kib"]) for r in task_resources)/1024, 1),
        "DFTargetCells": "958,464"
    }
    write("numbers.tex", "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items()))
    manifest["activation"] = activation
    manifest["scope"] = "Displayed tables summarize the sealed selected primary context or explicitly labeled secondary dimension/matched views. No numerical experiments are run."
    (out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "generated_files": len(manifest["outputs"]),
                      "verified_input_tables": len(manifest["inputs"]), "numeric_cells": len(manifest["numeric_cells"])}))


if __name__ == "__main__":
    main()
