#!/usr/bin/env python3
"""Generate manuscript tables from the complete audited dimension-study data."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


SYSTEMS = ("harmonic", "quasiperiodic", "nonlinear_ar", "logistic", "vanderpol", "duffing")
NAMES = {"harmonic": "Harmonic", "quasiperiodic": "Quasiperiodic",
         "nonlinear_ar": "Nonlinear AR", "logistic": "Logistic",
         "vanderpol": "Van der Pol", "duffing": "Duffing"}
ARMS = ("cap2", "native20", "selected_cap", "rbf", "affine", "ridge_ar",
        "polynomial", "persistence", "constant")
ARM_NAMES = {"cap2": "KAHM cap 2", "native20": "KAHM native 20",
             "selected_cap": "KAHM selected cap", "rbf": "RBF", "affine": "Affine hull",
             "ridge_ar": "Ridge AR", "polynomial": "Quadratic direct",
             "persistence": "Persistence", "constant": "Constant"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, lines):
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    paper = args.paper.resolve()
    data, tables = paper / "dimension_study", paper / "tables"
    provenance = json.loads((data / "provenance.json").read_text())
    for name, metadata in provenance["files"].items():
        if digest(data / name) != metadata["sha256"]:
            raise RuntimeError("Dimension-study input differs from its archived hash: " + name)
    metrics = json.loads((data / "report_metrics.json").read_text())
    audit = json.loads((data / "final_results_audit.json").read_text())
    summary = json.loads((data / "summary.json").read_text())
    selected = rows(data / "selected_forecasts.csv")
    policy = rows(data / "selected_policy_summary.csv")
    resolutions = rows(data / "certificate_resolution.csv")
    comparisons = rows(data / "paired_contrasts.csv")
    rank_rows = rows(data / "selected_bank_local_ranks.csv")
    lookup = {(row["system"], row["arm"]): row for row in policy}
    primary = {row["system"]: row for row in metrics["forecasting_by_law"]}
    if not audit.get("passed", audit.get("status") == "passed"):
        raise RuntimeError("The complete numerical audit did not pass")
    if len(selected) != 234 or len(policy) != 54:
        raise RuntimeError("Incomplete dimension-study family")
    generated = []

    numbers = {
        "DimRepetitions": 26, "DimDevelopmentFits": 1820, "DimEvaluatedModels": 624,
        "DimRiskTails": 3744, "DimImprovementTails": 936, "DimTotalTails": 4680,
        "DimBetterRuns": metrics["selected_cap_vs_cap2"]["favorable_repetitions"],
        "DimEqualRuns": metrics["selected_cap_vs_cap2"]["equal_repetitions"],
        "DimWorseRuns": metrics["selected_cap_vs_cap2"]["unfavorable_repetitions"],
        "DimCapOneCount": metrics["selected_cap_counts"]["1"],
        "DimCapTwoCount": metrics["selected_cap_counts"]["2"],
        "DimCapFourCount": metrics["selected_cap_counts"]["4"],
        "DimSelectedForecasts": metrics["selected_KAHM_forecast_count"],
        "DimSelectedLowerZero": metrics["certificate_decisions"]["zero_selected_KAHM_horizon_lower_endpoints"],
        "DimPositivePrefix": metrics["certificate_decisions"]["positive_retained_prefix_improvement_endpoints"],
        "DimPrefixTails": metrics["certificate_decisions"]["retained_prefix_comparison_endpoints"],
        "DimPositivePolicy": metrics["certificate_decisions"]["positive_policy_improvement_endpoints"],
        "DimPolicyTails": metrics["certificate_decisions"]["policy_comparison_endpoints"],
        "DimAllLowerZero": summary["certificate_diagnostics"]["zero_lower_endpoints"],
        "DimAllLowerCount": summary["certificate_diagnostics"]["risk_lower_endpoints"],
        "DimAcceptLow": metrics["certificate_decisions"]["accepted_per_policy"]["0.2"],
        "DimAcceptMedium": metrics["certificate_decisions"]["accepted_per_policy"]["0.35"],
        "DimAcceptHigh": metrics["certificate_decisions"]["accepted_per_policy"]["0.5"],
        "DimNativeDuffingIncrease": f"{primary['duffing']['native20_percent_change_from_cap2']:.2f}",
        "DimAuditTableRows": audit["counts"]["export_rows_verified"],
        "DimAuditPredictionJoins": audit["counts"]["full_prediction_loss_joins"],
    }
    name = "dimension_policy_numbers.tex"
    write(tables / name, ["% Generated from dimension_study/; do not hand-edit values."] +
          [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in numbers.items()])
    generated.append(name)

    main_rows = [
        r"\begin{table}[t]", r"\centering\small",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Law & Cap 2 & Native 20 & Selected cap & Change (\%) \\",
        r"\midrule"]
    for system in SYSTEMS:
        result = primary[system]
        values = [float(lookup[(system, arm)]["mean_test_mse"])
                  for arm in ("cap2", "native20", "selected_cap")]
        main_rows.append(NAMES[system] + " & " + " & ".join(f"{v:.6f}" for v in values)
                         + f" & {result['selected_cap_percent_change_from_cap2']:+.2f}" + r" \\")
    main_rows += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Local-dimension policies for the block-direct KAHM decoder.",
        r"Entries are test MSE averaged over horizons within each law and then",
        r"over five fitting repetitions for the first four laws and three for each",
        r"oscillator. The last column compares selected cap with cap two.",
        r"Validation selects the history and prefix within each policy; the",
        r"selected-cap arm also selects the cap. These test-error contrasts are",
        r"descriptive. Supplement Section~S9 reports per-horizon errors,",
        r"variation across fits, comparators, and the simultaneous risk statements.}",
        r"\label{tab:dimension-policy-summary}", r"\end{table}"]
    name = "dimension_policy_summary.tex"
    write(tables / name, main_rows)
    generated.append(name)

    full = [
        r"\begingroup\small", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}llrrrr@{}}",
        r"\caption{Dimension-study test MSE for all nine validation-selected",
        r"forecasting families. Horizon-specific entries average fitting repetitions.",
        r"The final column gives the mean over horizons, with its sample standard",
        r"deviation across repetitions in parentheses. The four nontrivial geometric",
        r"and conventional comparators retain their declared definitions and budgets;",
        r"the cross-family contrasts are descriptive.}",
        r"\label{tab:dimension-full-results}\\",
        r"\toprule Law & Family & $h=1$ & $h=4$ & $h=12$ & Mean (SD) \\",
        r"\midrule\endfirsthead",
        r"\toprule Law & Family & $h=1$ & $h=4$ & $h=12$ & Mean (SD) \\",
        r"\midrule\endhead", r"\bottomrule\endfoot"]
    for system in SYSTEMS:
        for index, arm in enumerate(ARMS):
            record = lookup[(system, arm)]
            label = NAMES[system] if index == 0 else ""
            means = [float(record[f"mean_test_mse_h{h}"]) for h in (1, 4, 12)]
            mean, sd = float(record["mean_test_mse"]), float(record["sd_test_mse_over_repetitions"])
            full.append(label + " & " + ARM_NAMES[arm] + " & "
                        + " & ".join(f"{value:.5f}" for value in means)
                        + f" & {mean:.5f} ({sd:.5f})" + (r" \\*" if index < len(ARMS)-1 else r" \\"))
        if system != SYSTEMS[-1]:
            full.append(r"\addlinespace")
    full += [r"\end{longtable}", r"\endgroup"]
    name = "dimension_policy_full_results.tex"
    write(tables / name, full)
    generated.append(name)

    choices = [r"\begin{table}[htbp]", r"\centering\small",
               r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
               r"Law & Cap 1 & Cap 2 & Cap 4 & $m=4$ & $m=12$ \\",
               r"\midrule"]
    for system in SYSTEMS:
        chosen = [row for row in selected if row["system"] == system and row["arm"] == "selected_cap"]
        counts = collections.Counter(row["cap"] for row in chosen)
        histories = collections.Counter(row["history"] for row in chosen)
        values = [counts[str(cap)] for cap in (1, 2, 4)] + [histories["4"], histories["12"]]
        choices.append(NAMES[system] + " & " + " & ".join(map(str, values)) + r" \\")
    choices += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Validation-selected requested caps and history lengths.",
        r"Caps 8, 12, and 20 were available but were not selected.",
        r"All selected forecasts used the 30-coordinate prefix. Requested caps",
        r"remain distinct from actual local ranks after numerical reductions.",
        r"Caps at least as large as the history dimension give equivalent native",
        r"requests for fixed references; ties favor the smaller cap.}",
        r"\label{tab:dimension-policy-choices}", r"\end{table}"]
    name = "dimension_policy_choices.tex"
    write(tables / name, choices)
    generated.append(name)

    ranks = [r"\begin{table}[htbp]", r"\centering\small",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Policy & Rank 1 & Rank 2 & Rank 3 & Rank 4 & Rank 12 \\",
             r"\midrule"]
    for arm in ("cap2", "native20", "selected_cap"):
        values = collections.Counter(row["effective_rank"] for row in rank_rows if row["arm"] == arm)
        ranks.append(ARM_NAMES[arm] + " & " + " & ".join(str(values[str(rank)]) for rank in (1, 2, 3, 4, 12)) + r" \\")
    ranks += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Actual local PCA ranks in the selected KAHM banks.",
        r"Each policy uses 780 local encoders across 26 banks. The rank-three",
        r"encoders arise from the native projected-range reduction. Histories",
        r"have ambient dimension four or twelve, so native cap twenty does not",
        r"produce rank twenty in this study. Counts include the full retained",
        r"banks; each selected forecast also uses all four blocks.}",
        r"\label{tab:dimension-policy-ranks}", r"\end{table}"]
    name = "dimension_policy_ranks.tex"
    write(tables / name, ranks)
    generated.append(name)

    resolution = [
        r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Law & Policy & $\varepsilon=0.20$ & $\varepsilon=0.35$ & $\varepsilon=0.50$ & $L>0$ \\",
        r"\midrule"]
    for system in SYSTEMS:
        for index, arm in enumerate(("cap2", "native20", "selected_cap")):
            entries = [row for row in resolutions if row["system"] == system and row["arm"] == arm
                       and row["scope"] == "all_retained_prefixes"]
            by_tolerance = {row["rmse_tolerance"]: row for row in entries}
            values = ["/".join(by_tolerance[t][field] for field in ("accepted", "excluded", "unresolved"))
                      for t in ("0.2", "0.35", "0.5")]
            positive = 12 * primary[system]["repetitions"] - int(entries[0]["zero_lower_endpoints"])
            resolution.append((NAMES[system] if index == 0 else "") + " & " + ARM_NAMES[arm]
                              + " & " + " & ".join(values) + f" & {positive}" + r" \\")
        if system != SYSTEMS[-1]:
            resolution.append(r"\addlinespace")
    resolution += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{All retained-prefix KAHM classes in the dimension study.",
        r"Each tolerance entry gives accepted/excluded/unresolved counts for the",
        r"four prefixes in each repetition, using the all-horizon criterion.",
        r"The final column counts positive horizon-specific lower endpoints",
        r"over all prefixes. The study-wide event covers all designated",
        r"endpoints jointly. No conflicting tolerance decision occurred.}",
        r"\label{tab:dimension-policy-resolution}", r"\end{table}"]
    name = "dimension_policy_resolution.tex"
    write(tables / name, resolution)
    generated.append(name)

    improvements = [r"\begin{table}[t]", r"\centering\small",
                    r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
                    r"Law & Cap 2 & Native 20 & Selected cap & Per-policy total \\",
                    r"\midrule"]
    for system in SYSTEMS:
        positive = [sum(row["system"] == system and row["type"] == "retained_prefix"
                        and row["contrast"] == arm and row["certified_positive_improvement"] == "True"
                        for row in comparisons) for arm in ("cap2", "native20", "selected_cap")]
        denominator = 9 * primary[system]["repetitions"]
        improvements.append(NAMES[system] + " & " + " & ".join(map(str, positive))
                            + f" & {denominator}" + r" \\")
    improvements += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Positive simultaneous lower endpoints for adjacent-prefix",
        r"fitted-loss improvements. Each policy has three adjacent prefix pairs",
        r"and three horizons per repetition. These are fitted-forecast comparisons,",
        r"not differences between unknown class optima. The separate 234",
        r"policy-selected forecast comparisons had no positive lower endpoint.}",
        r"\label{tab:dimension-policy-improvements}", r"\end{table}"]
    name = "dimension_policy_improvements.tex"
    write(tables / name, improvements)
    generated.append(name)
    manifest = {"generator_sha256": digest(Path(__file__)),
                "provenance_sha256": digest(data / "provenance.json"),
                "source_freeze_sha256": provenance["source_freeze_sha256"],
                "generated": {name: digest(tables / name) for name in generated}}
    (tables / "dimension_policy_assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(generated)} dimension-study TeX assets from verified source tables.")


if __name__ == "__main__":
    main()
