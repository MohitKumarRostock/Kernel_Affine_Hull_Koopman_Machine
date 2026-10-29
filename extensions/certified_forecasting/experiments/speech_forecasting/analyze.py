"""Summarize frozen speaker-level forecasts and regenerate the manuscript table.

The analysis performs no fitting or selection. Paired comparisons are
explicitly descriptive and use the same held-out identifiers in each fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ("kahm", "rbf", "affine", "ar", "quadratic", "kernel", "persistence", "constant")
LABELS = {"kahm": "KAHM blocks", "rbf": "RBF blocks", "affine": "Affine-hull blocks",
          "ar": "Ridge AR", "quadratic": "Quadratic", "kernel": "Gaussian kernel",
          "persistence": "Persistence", "constant": "Constant"}


def analyze(results, destination, paper):
    destination.mkdir(parents=True, exist_ok=True)
    saved = json.loads((results / "results.json").read_text())
    records = saved["results"]
    scores = pd.read_csv(results / "speaker_losses.csv.gz", dtype={"speaker": str})
    if len(records) != 24 or len(scores) != 88056:
        raise ValueError("The complete prescribed evaluation is required.")
    rows, timings = [], []
    for record in records:
        for split in ("calibration", "test"):
            block = scores[(scores.seed == record["seed"]) & (scores.family == record["family"])
                           & (scores.split == split)]
            for j, h in enumerate((1, 4, 12)):
                observed = block.loc[block.horizon == h, "mse"].mean()
                if not np.isclose(observed, record[split + "_mse"][j], atol=2e-16, rtol=0):
                    raise ValueError("Frozen losses disagree with the result record.")
        for j, h in enumerate((1, 4, 12)):
            rows.append({"seed": record["seed"], "family": record["family"], "horizon": h,
                         "test_rmse_db": 60 * np.sqrt(record["test_mse"][j]),
                         "upper_rmse_db": 60 * np.sqrt(record["mse_upper"][j]),
                         "test_mse": record["test_mse"][j],
                         "calibration_mse": record["calibration_mse"][j]})
        development = json.loads((results / f"development_{record['seed']}.json").read_text())
        bank = next((b for b in development["banks"] if b["bank_id"] == record["bank_id"]), {})
        timings.append({"seed": record["seed"], "family": record["family"],
                        "selected_head_fit_seconds": record["head_fit_seconds"],
                        "shared_bank_fit_seconds": bank.get("geometry_fit_seconds", np.nan),
                        "shared_bank_development_transform_seconds": bank.get("development_transform_seconds", np.nan),
                        "test_prediction_seconds": record["test_prediction_seconds"],
                        "test_prediction_rows": record["test_prediction_rows"]})
    long = pd.DataFrame(rows)
    summary = long.groupby(["family", "horizon"]).agg(
        mean_test_rmse_db=("test_rmse_db", "mean"),
        sd_test_rmse_db=("test_rmse_db", "std"),
        maximum_upper_rmse_db=("upper_rmse_db", "max"))
    long.to_csv(destination / "selected_results.csv", index=False)
    summary.to_csv(destination / "summary.csv")
    pd.DataFrame(timings).to_csv(destination / "timings.csv", index=False)
    paired = []
    test = scores[scores.split == "test"]
    for seed in sorted(test.seed.unique()):
        for horizon in (1, 4, 12):
            wide = test[(test.seed == seed) & (test.horizon == horizon)].pivot(
                index="speaker", columns="family", values="mse")
            if wide.shape != (245, 8) or wide.isna().any().any():
                raise ValueError("Paired speaker records are incomplete.")
            for family in FAMILIES:
                difference = wide[family] - wide.kahm
                for speaker, value in difference.items():
                    paired.append({"seed": seed, "horizon": horizon, "speaker": speaker,
                                   "alternative": family, "alternative_minus_kahm_mse": value})
    paired = pd.DataFrame(paired)
    paired.to_csv(destination / "paired_test_differences.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0})
    paired.groupby(["seed", "horizon", "alternative"]).agg(
        mean_difference=("alternative_minus_kahm_mse", "mean"),
        median_difference=("alternative_minus_kahm_mse", "median"),
        kahm_lower_loss_groups=("alternative_minus_kahm_mse", lambda x: int((x > 0).sum()))
    ).to_csv(destination / "paired_test_summary.csv")
    passes = {family: {str(db): sum(max(r["mse_upper"]) <= (db/60)**2 for r in records
                                   if r["family"] == family) for db in (6, 12, 18)} for family in FAMILIES}
    audit = {"status": "passed", "selected_models": len(records), "speaker_loss_records": len(scores),
             "all_selected_converged": all(r["converged"] for r in records),
             "all_horizon_pass_counts_out_of_three": passes,
             "scope": saved["confidence_scope"],
             "fitting_variation": "Three fits share the same group split; SD is not cohort uncertainty.",
             "paired_comparisons": "Descriptive; no separately allocated population-superiority test."}
    (destination / "summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    table = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Test RMSE in dB: mean (SD)} & \multicolumn{3}{c}{Maximum upper RMSE in dB} & All $\leq18$ dB\\",
             r"Family & 10 ms & 40 ms & 120 ms & 10 ms & 40 ms & 120 ms & Fits / 3\\", r"\midrule"]
    for family in FAMILIES:
        cells = [f"{summary.loc[(family,h),'mean_test_rmse_db']:.2f} ({summary.loc[(family,h),'sd_test_rmse_db']:.2f})" for h in (1,4,12)]
        cells += [f"{summary.loc[(family,h),'maximum_upper_rmse_db']:.2f}" for h in (1,4,12)]
        cells += [str(passes[family]["18"]) + "/3"]
        table.append(LABELS[family] + " & " + " & ".join(cells) + r"\\")
    table += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Speech-energy forecasting on 245 held-out identifiers. Test means and sample standard deviations use three fits on the same group split. Upper columns take the maximum over those fits, with simultaneous coverage of all 72 selected family--fit--horizon statements under the stated iid group-record assumption. All six learned families pass the 18 dB requirement at every horizon in each fit; none passes the all-horizon 6 or 12 dB requirement. Errors concern the floored energy observable.}",
              r"\label{tab:speech}", r"\end{table}", ""]
    (paper / "tables" / "speech_forecasting.tex").write_text("\n".join(table))
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    analyze(args.results, args.output, args.paper)


if __name__ == "__main__":
    main()
