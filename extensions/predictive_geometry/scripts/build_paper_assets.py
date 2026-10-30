#!/usr/bin/env python3
"""Build manuscript assets from archived outcomes, independently checking claims.

Usage (from the extension root):
    python scripts/build_paper_assets.py --input archive --output manuscript/generated

No experiment is fitted or selected here. The two complete declared contrast
families are recomputed from replicate-level outcomes and checked against the
archived analysis. Files named *.json.gz are accepted in place of *.json.
The generated PDF figures contain vector graphics; PNG counterparts are 300 dpi.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import t as student_t


SYSTEMS = ("forecast_nonlinear_ar", "forecast_van_der_pol", "forecast_duffing")
SYSTEM_LABELS = dict(zip(SYSTEMS, ("Nonlinear AR", "Van der Pol", "Duffing")))
BRANCH = tuple(f"branch_c{c}_n1024_l{l}" for c in (0, 1) for l in ("0", "0p5", "1"))
PRIMARY_SETTINGS = BRANCH + SYSTEMS
FEATURE_LABELS = {
    "raw": "Raw", "affine:phi": r"Affine $\phi$", "kahm:phi": r"KAHM $\phi$",
    "native_affine:phi": "Affine locality", "affine:local": "Affine local",
    "kahm:local": "KAHM local", "kahm:scores": "KAHM scores",
    "kahm:phi_logz": r"KAHM $(\phi,z)$",
}
REPLICATES = tuple(range(24))


class Archive:
    def __init__(self, root):
        self.root = Path(root)
        self.sources = []
        self.margin_provenance = {}

    def read(self, relative, optional=False):
        path = self.root / relative
        if not path.exists() and path.suffix == ".json":
            path = path.with_suffix(".json.gz")
        if not path.exists():
            if optional:
                return None
            raise FileNotFoundError(f"Required archived input: {path}")
        content = path.read_bytes()
        decoded = gzip.decompress(content) if path.suffix == ".gz" else content
        self.sources.append({
            "path": str(path.relative_to(self.root)), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "decompressed_sha256": hashlib.sha256(decoded).hexdigest(),
        })
        if relative.endswith(".csv"):
            return list(csv.DictReader(decoded.decode().splitlines()))
        return json.loads(decoded)


def series(records, setting, feature, head="ridge", phase="confirmation", landmarks=None):
    rows = [r for r in records if r["setting_id"] == setting and r["feature"] == feature
            and r["head"] == head and r["phase"] == phase
            and (landmarks is None or int(r["landmarks"]) == landmarks)]
    rows.sort(key=lambda r: int(r["replicate"]))
    expected = tuple(range(2)) if phase == "pilot" else REPLICATES
    if tuple(int(r["replicate"]) for r in rows) != expected:
        raise ValueError(f"Missing/duplicate replicates: {(phase, setting, feature, head, landmarks)}")
    values = np.asarray([r["test_mse"] for r in rows], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Nonfinite or negative test loss")
    return values


def mean(records, *args, **kwargs):
    return float(series(records, *args, **kwargs).mean())


def close(actual, expected, name):
    if not math.isclose(float(actual), float(expected), rel_tol=2e-10, abs_tol=2e-13):
        raise ValueError(f"Numerical claim mismatch: {name}: {actual!r} != {expected!r}")


def paired_stats(control, candidate, family, margin):
    difference = np.asarray(control) - np.asarray(candidate)
    critical = float(student_t.ppf(1 - .05 / (2 * family), 23))
    sd = float(np.std(difference, ddof=1))
    se = sd / math.sqrt(24)
    center = float(difference.mean())
    lower, upper = center - critical * se, center + critical * se
    classification = ("practical_benefit" if lower > margin else
                      "positive_practical_size_unresolved" if lower > 0 else
                      "harm" if upper < 0 else "direction_unresolved")
    return {
        "control_mean_test_mse": float(np.mean(control)),
        "candidate_mean_test_mse": float(np.mean(candidate)),
        "n_replicates": 24, "mean_improvement": center,
        "sd_improvement": sd, "se_improvement": se,
        "simultaneous_ci_lower": lower, "simultaneous_ci_upper": upper,
        "critical_t": critical, "practical_margin": float(margin),
        "family_size": family, "classification": classification,
        "within_prespecified_practical_band": lower >= -margin and upper <= margin,
        "positive_replicates": int(np.sum(difference > 0)),
        "paired_differences": difference.tolist(),
    }


def verify_contrasts(records, archived, margins, capacity=False):
    expected_ids = (
        {f"{s}__m{m}__{c}_minus_kahm_local" for s in SYSTEMS for m in (64, 256)
         for c in ("raw", "affine_local")} if capacity else
        {f"{s}__{c}" for s in PRIMARY_SETTINGS for c in ("G", "N", "L")} | {"I_c0", "I_c1"}
    )
    ids = [r["contrast_id"] for r in archived]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("Archived primary family is incomplete or contains unexpected contrasts")
    output = {}
    for row in archived:
        contrast_id = row["contrast_id"]
        setting = row["setting_id"]
        if capacity:
            width = int(row["landmarks"])
            control = series(records, setting, row["control_feature"], "rbf", "capacity_confirmation", width)
            candidate = series(records, setting, "kahm:local", "rbf", "capacity_confirmation", width)
            margin = margins["cells"][setting]["margin"]
        elif contrast_id.startswith("I_c"):
            c = int(contrast_id[-1])
            s0, s1 = f"branch_c{c}_n1024_l0", f"branch_c{c}_n1024_l1"
            control = series(records, s0, "affine:phi") - series(records, s0, "kahm:phi")
            candidate = series(records, s1, "affine:phi") - series(records, s1, "kahm:phi")
            margin = margins["interactions"][contrast_id]["margin"]
        else:
            control = series(records, setting, row["control_feature"])
            candidate = series(records, setting, row["candidate_feature"])
            margin = margins["cells"][setting]["margin"]
        computed = paired_stats(control, candidate, 12 if capacity else 29, margin)
        for name, value in computed.items():
            if name in row and isinstance(value, (float, int)) and not isinstance(value, bool):
                close(value, row[name], f"{contrast_id}/{name}")
        if row["classification"] != computed["classification"]:
            raise ValueError(f"Classification mismatch: {contrast_id}")
        output[contrast_id] = {**row, **computed}
    return output


def num(value, digits=6):
    value = float(value)
    if value != 0 and abs(value) < .5 * 10**(-digits):
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"\ensuremath{{{mantissa}\times10^{{{int(exponent)}}}}}"
    return f"{value:.{digits}f}"


def sci(value, digits=2):
    mantissa, exponent = f"{float(value):.{digits}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def interval(row, scale=1, digits=6):
    center, lo, hi = (float(row[k]) * scale for k in
                      ("mean_improvement", "simultaneous_ci_lower", "simultaneous_ci_upper"))
    return rf"${num(center,digits)}\;[{num(lo,digits)},\,{num(hi,digits)}]$"


def tabular(path, columns, header, rows):
    lines = ["% Generated by scripts/build_paper_assets.py; do not edit.",
             rf"\begin{{tabular}}{{{columns}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(r"\midrule" if row is None else " & ".join(map(str, row)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def setting_label(setting, latex=True):
    if setting in SYSTEM_LABELS:
        return SYSTEM_LABELS[setting]
    c = setting.split("_c")[1][0]
    lam = setting.rsplit("_l", 1)[1].replace("p", ".")
    return rf"$c={c},\lambda={lam}$" if latex else f"c={c}, lambda={lam}"


def primary_appendix(path, contrasts, capacity=False):
    caption = ("Complete fresh capacity family (12 contrasts). All numerical loss quantities are multiplied by $10^3$."
               if capacity else "Complete first-stage family (29 contrasts), in original MSE units.")
    caption += (" Intervals use the two-sided 95\\% Bonferroni paired-replicate Student procedure with 23 degrees of freedom."
                " Under paired normality, each t interval is exact and Bonferroni family coverage is at least 95\\%;"
                " otherwise coverage is approximate. The two stages have separate families.")
    label = "tab:all-capacity" if capacity else "tab:all-stageone"
    header = ("Setting / budget", "Contrast", r"$\overline D$", "Simultaneous interval", r"$\delta$", "Assessment")
    lines = ["% Generated; this file owns its longtable environment.", r"\begingroup\small\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{llrrrl}", rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule", r"\endfirsthead", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule", r"\endhead", r"\bottomrule\endfoot"]
    for key, row in contrasts.items():
        scale, digits = (1000, 4) if capacity else (1, 6)
        if capacity:
            name = f"{SYSTEM_LABELS[row['setting_id']]} / {row['landmarks']}"
            contrast = "Raw--K" if row["control_feature"] == "raw" else "Affine--K"
        elif key.startswith("I_c"):
            name, contrast = rf"$c={key[-1]}$ interaction", rf"$I_{key[-1]}$"
        else:
            name, contrast = setting_label(row["setting_id"]), f"${row['contrast_code']}$"
        assessment = {"practical_benefit": "practical +", "positive_practical_size_unresolved": "positive",
                      "harm": "negative", "direction_unresolved": "unresolved"}[row["classification"]]
        if key.startswith("I_c") and row["classification"] == "harm":
            assessment = "opposite sign"
        if row["within_prespecified_practical_band"]:
            assessment += r" $\dagger$"
        ci = rf"$[{num(scale*row['simultaneous_ci_lower'],digits)},\,{num(scale*row['simultaneous_ci_upper'],digits)}]$"
        lines.append(" & ".join((name, contrast, num(scale*row["mean_improvement"], digits), ci,
                                  num(scale*row["practical_margin"], digits), assessment)) + r" \\")
    lines += [r"\end{longtable}", r"\noindent $\dagger$: the entire interval lies within the fixed practical band $[-\delta,\delta]$."
              " An unresolved sign alone does not establish equivalence.", r"\endgroup", ""]
    path.write_text("\n".join(lines))


def build_tables(output, s1, cap, sensitivity, a1, a2, numerical_audit, mechanisms):
    out = output / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in BRANCH:
        c, l = s.split("_c")[1][0], s.rsplit("_l", 1)[1].replace("p", ".")
        rows.append([c, l, num(mean(s1, s, "affine:phi")), num(mean(s1, s, "kahm:phi")),
                     num(mean(s1, s, "native_affine:phi")), interval(a1[s + "__G"])])
    tabular(out / "branch_primary.tex", "rrrrrl", ["$c$", r"$\lambda$", r"Affine $\phi$", r"KAHM $\phi$",
            "Affine locality", "$G$ [interval]"], rows)
    rows = [[setting_label(s), num(mean(s1, s, "kahm:phi")), num(mean(s1, s, "kahm:local")),
             num(mean(s1, s, "affine:local")), num(mean(s1, s, "raw")), interval(a1[s + "__L"])]
            for s in PRIMARY_SETTINGS]
    tabular(out / "ridge_local.tex", "lrrrrl", ["Setting", r"KAHM $\phi$", "KAHM local", "Affine local", "Raw", "$L$ [interval]"], rows)
    rows = []
    for s in SYSTEMS:
        for width in (64, 256):
            for control in ("raw", "affine_local"):
                row = a2[f"{s}__m{width}__{control}_minus_kahm_local"]
                rows.append([SYSTEM_LABELS[s], width, "Raw" if control == "raw" else "Affine local",
                             num(row["control_mean_test_mse"]*1000, 4),
                             num(row["candidate_mean_test_mse"]*1000, 4), interval(row, 1000, 4)])
        if s != SYSTEMS[-1]:
            rows.append(None)
    tabular(out / "capacity_primary.tex", "lrlrrl", ["System", "$M$", "Control", "Control MSE", "KAHM local MSE", r"$\Delta$ [interval]"], rows)
    numerical_rows, numerical_claims = [], []
    for feature in ("kahm:local", "kahm:scores"):
        for width in (64, 256):
            rows = sorted([r for r in sensitivity if r["feature"] == feature and int(r["landmarks"]) == width],
                          key=lambda r: int(r["replicate"]))
            if tuple(int(r["replicate"]) for r in rows) != REPLICATES:
                raise ValueError("Incomplete numerical sensitivity cell")
            original = series(cap, SYSTEMS[1], feature, "rbf", "capacity_confirmation", width)
            stable = np.array([r["test_mse"] for r in rows])
            raw = series(cap, SYSTEMS[1], "raw", "rbf", "capacity_confirmation", width)
            affine = series(cap, SYSTEMS[1], "affine:local", "rbf", "capacity_confirmation", width)
            for i, row in enumerate(rows):
                close(original[i], row["original_test_mse"], "sensitivity original outcome")
                close(raw[i], row["raw_baseline_mse"], "sensitivity raw outcome")
                close(affine[i], row["affine_local_baseline_mse"], "sensitivity affine outcome")
            entry = {"feature": feature, "landmarks": width, "original_mean_mse": float(original.mean()),
                     "stabilized_mean_mse": float(stable.mean()),
                     "max_absolute_replicate_mse_change": float(np.max(np.abs(original-stable))),
                     "mean_original_minus_stabilized_mse": float(np.mean(original-stable)),
                     "wins_vs_raw": int(np.sum(stable < raw)), "wins_vs_affine_local": int(np.sum(stable < affine))}
            if numerical_audit:
                ref = next(x for x in numerical_audit["methods"] if x["feature"] == feature and x["landmarks"] == width)
                for name, value in entry.items():
                    if isinstance(value, (int, float)):
                        close(value, ref[name], "numerical independent audit/" + name)
            numerical_claims.append(entry)
            numerical_rows.append(["Local" if feature.endswith("local") else "Scores", width,
                                   f"{original.mean():.9f}", f"{stable.mean():.9f}",
                                   sci(entry["max_absolute_replicate_mse_change"]),
                                   f"{entry['wins_vs_raw']}/24; {entry['wins_vs_affine_local']}/24"])
    tabular(out / "numerical_sensitivity.tex", "lrrrrl", ["Encoding", "$M$", "Original MSE", "Stable MSE", r"Max. $|\Delta_r|$", "Wins raw; affine"], numerical_rows)
    airfoil = []
    for feature in ("raw", "affine:phi", "kahm:phi", "affine:local", "kahm:local", "kahm:scores"):
        airfoil.append([FEATURE_LABELS[feature]] + [num(mean(s1, "airfoil_grouped", feature, h, "natural"), 4)
                                                   for h in ("ridge", "rbf")])
    tabular(out / "airfoil.tex", "lrr", ["Representation", "Ridge MSE", "RBF$_{64}$ MSE"], airfoil)
    dimension_rows, capacity_rows, selector_rows = [], [], []
    for setting in SYSTEMS:
        for width in (64, 256):
            for feature in ("raw", "affine:local", "kahm:local", "kahm:scores"):
                chosen = [r for r in cap if r["setting_id"] == setting and r["feature"] == feature
                          and r["head"] == "rbf" and int(r["landmarks"]) == width]
                dims = [int(r["selected"][0]["basis_dimension"]) for r in chosen]
                dimension_rows.append([SYSTEM_LABELS[setting], width, FEATURE_LABELS[feature],
                                       num(np.mean(dims), 2), f"{min(dims)}--{max(dims)}"])
            capacity_rows.append([SYSTEM_LABELS[setting], width] +
                                 [num(mean(cap, setting, f, "rbf", "capacity_confirmation", width)*1000, 4)
                                  for f in ("raw", "affine:local", "kahm:local", "kahm:scores")])
        chosen = [r for r in cap if r["setting_id"] == setting and r["head"] == "selected"]
        counts = Counter((r["selected"]["pipeline_feature"], int(r["selected"]["landmarks"])) for r in chosen)
        selection_text = "; ".join(f"{FEATURE_LABELS[f]}, {w}: {n}/24" for (f,w),n in sorted(counts.items()))
        selector_rows.append([SYSTEM_LABELS[setting], num(np.mean([r["test_mse"] for r in chosen])*1000, 4), selection_text])
    tabular(out / "capacity_dimensions.tex", "lrlrr", ["System", "$M$", "Representation", "Mean dimension", "Range"], dimension_rows)
    tabular(out / "capacity_secondary.tex", "lrrrrr", ["System", "$M$", "Raw", "Affine local", "KAHM local", "KAHM scores"], capacity_rows)
    tabular(out / "capacity_selector.tex", "lrl", ["System", "Selector MSE", "Validation selections"], selector_rows)
    primary_appendix(out / "all_primary_twenty_nine.tex", a1)
    primary_appendix(out / "all_capacity_twelve.tex", a2, capacity=True)
    mechanism_claims = build_mechanism_tables(out, mechanisms) if mechanisms else None
    return numerical_claims, mechanism_claims


def build_mechanism_tables(out, records):
    if isinstance(records, dict):
        records = records["records"]
    bank_settings = ("branch_c0_n1024", "branch_c1_n1024") + SYSTEMS
    summaries, recon_rows, reflection_rows = [], [], []
    for setting in bank_settings:
        rows = [r for r in records if r["phase"] == "confirmation" and r["bank_setting_id"] == setting]
        rows.sort(key=lambda r: int(r["replicate"]))
        if tuple(int(r["replicate"]) for r in rows) != REPLICATES:
            raise ValueError(f"Missing mechanism bank repetitions: {setting}")
        entry = {"bank_setting_id": setting, "n_replicates": 24}
        geometry = {}
        for g in ("kahm", "affine"):
            geometry[g] = {key: np.array([np.mean(r["test_geometry_diagnostics"]["geometries"][g][key]) for r in rows])
                           for key in ("best_reconstruction_mse_by_block", "gated_reconstruction_mse_by_block")}
        entry["reconstruction"] = {g: {key: float(v.mean()) for key, v in values.items()} for g, values in geometry.items()}
        entry["kahm_lower_reconstruction_count"] = {key: int(np.sum(geometry["kahm"][key] < geometry["affine"][key]))
                                                    for key in geometry["kahm"]}
        label = SYSTEM_LABELS.get(setting, rf"Branch $c={setting.split('_c')[1][0]}$" if "_c" in setting else setting)
        recon_rows.append([label] + [sci(geometry[g][key].mean(), 3)
                                    for key in ("best_reconstruction_mse_by_block", "gated_reconstruction_mse_by_block")
                                    for g in ("kahm", "affine")])
        if setting.startswith("branch"):
            reflection = {}
            for feature in ("kahm:phi", "affine:phi", "kahm:local", "affine:local"):
                values = [r["mechanism_diagnostics"]["branch_reflection"]["metrics"][feature] for r in rows]
                reflection[feature] = {key: float(np.mean([v[key] for v in values]))
                                       for key in ("stored_coordinate_rms", "training_standardized_coordinate_rms")}
                reflection_rows.append([setting.split("_c")[1][0], FEATURE_LABELS[feature]] +
                                       [sci(reflection[feature][key], 3) for key in reflection[feature]])
            entry["reflection"] = reflection
        summaries.append(entry)
    tabular(out / "reconstruction.tex", "lrrrr", ["Setting", "KAHM best", "Affine best", "KAHM weighted", "Affine weighted"], recon_rows)
    tabular(out / "reflection.tex", "rlrr", ["$c$", "Encoding", "Stored-coordinate RMS", "Standardized RMS"], reflection_rows)
    return summaries


def build_figures(output, s1, cap, a2):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kahm-paper-mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    out = output / "figures"
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    colors = {"raw": "#333333", "affine:phi": "#0072B2", "kahm:phi": "#D55E00",
              "native_affine:phi": "#009E73", "affine:local": "#0072B2", "kahm:local": "#CC79A7",
              "kahm:scores": "#E69F00"}
    labels = {"raw": "Raw ridge", "affine:phi": "Matched affine affinities", "kahm:phi": "KAHM affinities",
              "native_affine:phi": "Affine locality score", "kahm:local": "KAHM local coordinates"}
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.95), sharey=True)
    features = ("affine:phi", "kahm:phi", "native_affine:phi", "kahm:local", "raw")
    for c, ax in enumerate(axes):
        for f, marker in zip(features, ("o", "s", "^", "D", "x")):
            ys = [mean(s1, f"branch_c{c}_n1024_l{l}", f) for l in ("0", "0p5", "1")]
            ax.plot((0, .5, 1), ys, marker=marker, markersize=4.4, linewidth=1.3, color=colors[f], label=labels[f])
        ax.axhline(.1**2/3, color="#999999", linestyle=":", linewidth=1, zorder=0)
        ax.set(yscale="log", xticks=(0, .5, 1), ylim=(.0026, 1.5), xlabel=r"Within-branch target weight $\lambda$",
               title=f"{'Straight' if c == 0 else 'Curved'} branches ($c={c}$)")
        ax.grid(axis="y", color="#dddddd", linewidth=.5, which="major")
    axes[0].set_ylabel("Mean held-out MSE (log scale)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, -.14), ncol=2, frameon=False)
    fig.tight_layout(w_pad=2)
    save(fig, "branch_alignment")

    fig, axes = plt.subplots(1, 3, figsize=(6.65, 2.65), sharey=False)
    for s, ax in zip(SYSTEMS, axes):
        for f, marker in zip(("raw", "affine:local", "kahm:local", "kahm:scores"), ("x", "o", "s", "^")):
            ys = [mean(cap, s, f, "rbf", "capacity_confirmation", width)*1000 for width in (64, 256)]
            ax.plot((0, 1), ys, marker=marker, markersize=4.5, linewidth=1.3, color=colors[f],
                    label=FEATURE_LABELS[f].replace("$", ""))
        ax.set(xticks=(0, 1), xticklabels=("64", "256"), xlim=(-.12, 1.12), title=SYSTEM_LABELS[s], xlabel="Landmark budget")
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
    axes[0].set_ylabel(r"Mean held-out MSE $\times 10^3$")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, -.09), ncol=4, frameon=False)
    fig.tight_layout(w_pad=1.7)
    save(fig, "forecast_capacity")

    fig, axes = plt.subplots(1, 3, figsize=(6.65, 3.05), sharex=False)
    for s, ax in zip(SYSTEMS, axes):
        keys = [(64, "raw"), (64, "affine_local"), (256, "raw"), (256, "affine_local")]
        margin = float(a2[f"{s}__m64__raw_minus_kahm_local"]["practical_margin"])*1000
        ax.axvspan(-margin, margin, color="#eeeeee", zorder=0)
        ax.axvline(0, color="#777777", linewidth=.8)
        for y, (width, control) in enumerate(keys):
            row = a2[f"{s}__m{width}__{control}_minus_kahm_local"]
            center, lo, hi = [float(row[k])*1000 for k in ("mean_improvement", "simultaneous_ci_lower", "simultaneous_ci_upper")]
            color = colors["raw" if control == "raw" else "affine:local"]
            ax.errorbar(center, y, xerr=[[center-lo], [hi-center]], fmt="o", color=color,
                        markersize=4, linewidth=1.2, capsize=2.8)
        ax.set(yticks=range(4), yticklabels=("Raw, 64", "Affine, 64", "Raw, 256", "Affine, 256"),
               ylim=(3.5, -.5), title=SYSTEM_LABELS[s])
        ax.grid(axis="x", color="#dddddd", linewidth=.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
    fig.supxlabel(r"Mean test-MSE improvement (control $-$ KAHM local), $\times10^3$", fontsize=9, y=.015)
    fig.tight_layout(w_pad=1.2, rect=(0, .07, 1, 1))
    save(fig, "capacity_contrasts")


def build_claims(output, archive, s1, cap, a1, a2, numerical, mechanism, numerical_audit):
    macros = {
        "ConfirmationReplicates": "24", "StageOneFamilySize": "29", "CapacityFamilySize": "12",
        "StageOneCriticalValue": num(student_t.ppf(1-.05/58, 23), 6),
        "CapacityCriticalValue": num(student_t.ppf(1-.05/24, 23), 6),
        "FlatWithinAffineMSE": num(mean(s1, BRANCH[2], "affine:phi")),
        "FlatWithinKAHMMSE": num(mean(s1, BRANCH[2], "kahm:phi")),
        "FlatWithinNativeAffineMSE": num(mean(s1, BRANCH[2], "native_affine:phi")),
        "FlatWithinRawMSE": num(mean(s1, BRANCH[2], "raw")),
        "FlatWithinGeometryGain": num(a1[BRANCH[2]+"__G"]["mean_improvement"]),
        "FlatAlignmentInteraction": num(a1["I_c0"]["mean_improvement"]),
        "CurvedAlignmentInteraction": num(a1["I_c1"]["mean_improvement"]),
        "VDPPhiRidgeMSE": num(mean(s1, SYSTEMS[1], "kahm:phi")),
        "VDPKAHMLocalRidgeMSE": num(mean(s1, SYSTEMS[1], "kahm:local")),
        "VDPAffineLocalRidgeMSE": num(mean(s1, SYSTEMS[1], "affine:local")),
        "AirfoilRawRidgeMSE": num(mean(s1, "airfoil_grouped", "raw", "ridge", "natural"), 4),
        "AirfoilRawRBFMSE": num(mean(s1, "airfoil_grouped", "raw", "rbf", "natural"), 4),
        "AirfoilKAHMLocalRBFMSE": num(mean(s1, "airfoil_grouped", "kahm:local", "rbf", "natural"), 4),
        "LocalPracticalBenefitCount": str(sum(a1[s+"__L"]["classification"] == "practical_benefit" for s in PRIMARY_SETTINGS)),
        "CapacityPracticalBenefitCount": str(sum(r["classification"] == "practical_benefit" for r in a2.values())),
    }
    for width, word in ((64, "Narrow"), (256, "Wide")):
        raw = mean(cap, SYSTEMS[1], "raw", "rbf", "capacity_confirmation", width)
        affine = mean(cap, SYSTEMS[1], "affine:local", "rbf", "capacity_confirmation", width)
        kahm = mean(cap, SYSTEMS[1], "kahm:local", "rbf", "capacity_confirmation", width)
        macros.update({f"VDPRaw{word}MSE": num(raw, 9), f"VDPAffine{word}MSE": num(affine, 9),
                       f"VDPKAHM{word}MSE": num(kahm, 9),
                       f"VDPRaw{word}RelativeGain": num(100*(raw-kahm)/raw, 1),
                       f"VDPAffine{word}RelativeGain": num(100*(affine-kahm)/affine, 1)})
    if numerical_audit:
        counts = numerical_audit["independently_recomputed_test_normalizers"]
        macros["NumericalOriginalZeroCount"] = f"{counts['original_exact_zero_normalizers']:,}"
        macros["NumericalQueryClusterCount"] = f"{counts['query_cluster_pairs']:,}"
    (output / "claim_numbers.tex").write_text("% Generated from archived records; descriptive ratios have no ratio intervals.\n" +
        "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()) + "\n")
    selectors = {}
    for setting in SYSTEMS:
        rows = [r for r in cap if r["setting_id"] == setting and r["head"] == "selected"]
        selectors[setting] = {"n": len(rows), "mean_test_mse": float(np.mean([r["test_mse"] for r in rows])),
                              "selections": [r["selected"] for r in rows]}
    dimensions = []
    for setting in SYSTEMS:
        for width in (64, 256):
            for feature in ("raw", "affine:local", "kahm:local", "kahm:scores"):
                rows = [r for r in cap if r["setting_id"] == setting and r["feature"] == feature
                        and r["head"] == "rbf" and int(r["landmarks"]) == width]
                values = [int(r["selected"][0]["basis_dimension"]) for r in rows]
                dimensions.append({"setting_id": setting, "feature": feature, "landmarks": width,
                                   "mean": float(np.mean(values)), "minimum": min(values), "maximum": max(values)})
    payload = {
        "scope": "Source-derived manuscript numbers; no fitting or outcome-based selection in this script.",
        "sources": archive.sources, "interval_validation": {"stage1": len(a1), "capacity": len(a2),
            "replicates": 24, "degrees_of_freedom": 23, "families_separate": True,
            "parametric_or_approximate_not_distribution_free": True},
        "fixed_margin_provenance": archive.margin_provenance,
        "latex_macros": macros, "stage1_contrasts": list(a1.values()), "capacity_contrasts": list(a2.values()),
        "numerical_sensitivity": numerical, "mechanism_diagnostics": mechanism,
        "capacity_secondary_selector": selectors,
        "capacity_retained_dimensions": dimensions,
        "claim_limits": [
            "The primary G contrast changes reconstruction under a shared score construction; it does not compare all affine methods.",
            "Local features add affinity-gated raw-history PCA coordinates; they are not full reconstruction vectors.",
            "The primary N and L contrasts establish augmentation effects relative to KAHM phi, not KAHM versus affine augmentation effects.",
            "Fresh capacity data support KAHM local gains on this Van der Pol law at the two tested finite budgets.",
            "No primary width interaction is tested; changing the landmark budget also changes the realized kernel basis.",
            "Approximate 95% simultaneous coverage is separate for 29 first-stage and 12 fresh contrasts; no joint 95% claim.",
            "Natural Airfoil repeats share one fixed split; fit seeds are not independent natural-data cohorts.",
            "Numerical sensitivity refits on reused phase4 data and carries no new confirmatory confidence claim.",
        ]}
    (output / "claim_sheet.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    notes = {
        "branch_primary.tex": "All six primary branch settings, 1024 training points, 24 complete confirmation repetitions. Original MSE units. G is affine-phi minus KAHM-phi ridge; its intervals belong to the complete first-stage 29 family. Affine locality is a secondary scoring-rule control. Bayes MSE=1/300.",
        "ridge_local.tex": "All nine primary settings. Original MSE units. Kphi, Klocal, affine local and raw use ridge heads. L is Kphi minus Klocal, with first-stage 29-family intervals. Cross-geometry local comparisons are descriptive at stage1.",
        "capacity_primary.tex": "All loss quantities, differences, and interval endpoints are MULTIPLIED BY 1000. All 12 fresh-stage primary paired contrasts, each control minus KAHM local, over 24 complete fresh train/validation/test repetitions. Separate Bonferroni family12, df23, approximate outside paired normality.",
        "capacity_secondary.tex": "All mean MSEs multiplied by 1000; KAHM scores are secondary. No additional intervals are asserted.",
        "capacity_dimensions.tex": "Selected RBF-plus-linear basis dimension after training kernel eigentruncation. Includes linear features; it is not the nominal landmark count and does not by itself equal statistical degrees of freedom.",
        "capacity_selector.tex": "Mean MSE multiplied by 1000. Prespecified secondary selector among eight complete feature/budget pipelines, validation only. No additional primary confidence claim.",
        "numerical_sensitivity.tex": "Original MSE units. Original and stabilized mean test MSE; maximum absolute repetition-level MSE change; count beating corresponding raw and affine-local baseline. Same phase4 data, 24 banks, two encodings/two widths, descriptive reuse. Not a third fresh confirmation.",
        "airfoil.tex": "Fixed grouped test cohort, mean group-weighted squared error in dB^2 over 24 fitting seeds; static regression. Seeds are not independent natural cohorts. No population interval.",
        "branch_alignment.pdf": "Means of 24 complete first-stage repetitions; log MSE axis. Dotted line is Bayes noise MSE=1/300. Banks/noise fixed within repetition across lambda. Lines connect tested mixtures only. No error bars or additional interval claim.",
        "forecast_capacity.pdf": "Mean fresh-stage test MSE multiplied by 1000. Four feature representations and two finite landmark budgets. Means only, separate vertical scales by law. Scores are a secondary control. Budgets are not equal effective capacities.",
        "capacity_contrasts.pdf": "All 12 fresh-stage contrasts: control minus KAHM local. Loss differences multiplied by 1000. Separate x-axis scales by law. Gray bands show fixed +/- practical margin; points and bars are 24-repetition means and two-sided Bonferroni family12 Student intervals (df23). Coverage is approximate outside paired normality. No joint95 event with family29.",
        "reconstruction.tex": "Per standardized input-coordinate squared reconstruction errors, averaged over queries, blocks, then repetitions. Best means minimum across models within each block; weighted means affinity-weighted individual errors. It is not error of the averaged reconstruction vector. Descriptive diagnostic.",
        "reflection.tex": "Mean per-repetition per-coordinate RMS reflection difference, raw reflection before frozen pipeline. Stored and training-standardized coordinate metrics. Diagnostic scale cutoff1e-12 differs from fitted-head cutoff1e-8. Not a population certificate."
    }
    (output / "caption_notes.json").write_text(json.dumps(notes, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("archive"))
    parser.add_argument("--output", type=Path, default=Path("manuscript/generated"))
    args = parser.parse_args()
    archive = Archive(args.input)
    s1 = archive.read("stage1/records.json")
    cap = archive.read("capacity/capacity_records.json")
    sensitivity = archive.read("numerical_sensitivity/numerical_sensitivity_records.json")
    m1, m2 = archive.read("stage1/margins.json"), archive.read("capacity/capacity_margins.json")
    # Fresh executions estimate new outcomes but do not redefine historical
    # practical margins. Optional byte-preserving snapshots supply only that
    # fixed design provenance; all contrasts and figures still use s1/cap.
    historical_pilots = archive.read("stage1/margin_provenance_pilots.json", optional=True)
    if historical_pilots is not None:
        source = archive.sources[-1]
        expected = {r["sha256"] for r in m1["input_sources"]}
        if source["decompressed_sha256"] not in expected:
            raise ValueError("Pilot margin provenance does not match the frozen source hash")
        archive.margin_provenance["stage1"] = {**source, "use": "fixed pilot margins only"}
    else:
        historical_pilots = s1
        archive.margin_provenance["stage1"] = {"use": "pilot rows in the ordinary stage1 archive"}
    historical_stage1 = archive.read("stage1/margin_provenance_records.json", optional=True)
    if historical_stage1 is not None:
        source = archive.sources[-1]
        expected = {r["sha256"] for r in m2["first_stage_sources"]}
        if source["decompressed_sha256"] not in expected:
            raise ValueError("Capacity margin provenance does not match the frozen source hash")
        archive.margin_provenance["capacity"] = {**source, "use": "fixed capacity margins only"}
    else:
        historical_stage1 = s1
        archive.margin_provenance["capacity"] = {"use": "confirmation rows in the ordinary stage1 archive"}
    a1 = verify_contrasts(s1, archive.read("stage1/primary_contrasts.csv"), m1)
    a2 = verify_contrasts(cap, archive.read("capacity/capacity_primary_contrasts.csv"), m2, capacity=True)
    for s in PRIMARY_SETTINGS:
        close(.05 * series(historical_pilots, s, "raw", phase="pilot").mean(), m1["cells"][s]["margin"], "stage1 fixed margin")
    for s in SYSTEMS:
        close(.05 * mean(historical_stage1, s, "raw", "rbf"), m2["cells"][s]["margin"], "capacity fixed margin")
    numerical_audit = archive.read("numerical_sensitivity/numerical_sensitivity_audit.json", optional=True)
    mechanisms = archive.read("stage1/mechanism_records.json", optional=True)
    args.output.mkdir(parents=True, exist_ok=True)
    numerical, mechanism = build_tables(args.output, s1, cap, sensitivity, a1, a2, numerical_audit, mechanisms)
    build_figures(args.output, s1, cap, a2)
    build_claims(args.output, archive, s1, cap, a1, a2, numerical, mechanism, numerical_audit)
    print(json.dumps({"status": "ok", "verified_primary_contrasts": [len(a1), len(a2)],
                      "mechanism_banks_available": mechanisms is not None, "output": str(args.output)}))


if __name__ == "__main__":
    main()
