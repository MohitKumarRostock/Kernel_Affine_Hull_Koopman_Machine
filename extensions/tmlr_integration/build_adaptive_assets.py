"""Build manuscript displays from the completed adaptive development study.

Reporting only: no model fitting, tuning, new forecasts, or protocol changes.
The source remains the amended saved evidence, with the first attempt retained.
This lightweight builder checks task-to-collection aggregation and hashes its
inputs and outputs. It does not replace the study's saved-prediction audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "extensions/adaptive_development"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "manuscript/generated/adaptive_development"
METHODS = (
    ("ridge", "Raw ridge"),
    ("fixed_kahm", "Fixed KAHM"),
    ("adaptive_kahm", "Adaptive KAHM"),
    ("adaptive_affine", "Adaptive affine"),
    ("adaptive_rbf", "Adaptive RBF-local"),
    ("kernel_rr", "Sliding-window KRR"),
)
COLLECTIONS = ("synthetic_stationary", "synthetic_recurring", "dominick_weekly", "sunspot_daily")
BUDGETS = ("small", "large")


def description(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build(study: Path, output: Path) -> dict:
    inputs = {}

    def record(relative):
        path = study / relative
        inputs[relative] = description(path)
        return path

    def read_json(relative):
        return json.loads(record(relative).read_text())

    def read_csv(relative):
        with record(relative).open(newline="") as stream:
            return list(csv.DictReader(stream))

    tasks = read_csv("analysis/task_metrics.csv")
    collections = read_csv("analysis/collection_metrics.csv")
    decision = read_json("analysis/decision.json")
    audit = read_json("analysis/audit.json")
    original = read_json("archives/initial_attempt_record.json")
    peaks = read_json("analysis/post_run_diagnostics.json")
    numerical = read_json("AMENDMENT_DIAGNOSTICS.json")
    for name in ("PROTOCOL.md", "AMENDMENT_A1.md", "DEVELOPMENT_FREEZE.json", "config.json"):
        record(name)

    if len(tasks) != 144 or len({(r["task_id"], r["method"], r["budget"]) for r in tasks}) != 144:
        raise ValueError("Expected the complete 144-run amended roster")
    if sum(int(r["scored_forecasts"]) for r in tasks) != 387900:
        raise ValueError("Forecast count differs from the preserved amended study")
    if decision["completed_runs"] != 144 or decision["scored_forecasts"] != 387900:
        raise ValueError("Decision accounting differs from task records")
    if audit["status"] != "passed" or audit["run_count"] != 144 or audit["scored_forecasts"] != 387900:
        raise ValueError("A matching successful saved-evidence audit is required")
    if decision["decision"] != "DEFER" or not decision["technical_validity"] or decision["cap_violations"]:
        raise ValueError("This manuscript section describes the completed, valid DEFER study")
    for field in ("framework_error_count", "framework_fallback_forecasts", "prediction_fallback_calls",
                  "training_feature_fallback_rows", "reencoded_feature_fallback_rows"):
        if any(int(r[field]) for r in tasks):
            raise ValueError(f"Unexpected nonzero amended-run count: {field}")
    if (original["runs"], original["scored_forecasts"], original["framework_errors"],
        original["framework_fallback_forecasts"]) != (144, 387900, 4, 8659):
        raise ValueError("Original-attempt accounting differs from the disclosed amendment")

    lookup = {(r["budget"], r["collection"], r["method"]): r for r in collections}
    expected_keys = {(b, c, m) for b in BUDGETS
                     for c in (*COLLECTIONS, "natural_equal_collection") for m, _ in METHODS}
    if len(collections) != 60 or set(lookup) != expected_keys:
        raise ValueError("Incomplete or duplicated collection table")
    aggregated = {}
    for budget in BUDGETS:
        for method, _ in METHODS:
            for collection in COLLECTIONS:
                selected = [r for r in tasks if (r["budget"], r["method"], r["collection"])
                            == (budget, method, collection)]
                expected_count = 3 if collection == "dominick_weekly" else 1 if collection == "sunspot_daily" else 4
                if len(selected) != expected_count:
                    raise ValueError("Task count differs from the fixed source roster")
                key = (budget, collection, method)
                value = mean(float(r["scaled_mse"]) for r in selected)
                if not math.isclose(value, float(lookup[key]["mean_scaled_mse"]), rel_tol=1e-12, abs_tol=1e-13):
                    raise ValueError(f"Collection aggregation differs for {key}")
                aggregated[key] = value
            key = (budget, "natural_equal_collection", method)
            value = mean(aggregated[(budget, c, method)] for c in ("dominick_weekly", "sunspot_daily"))
            if not math.isclose(value, float(lookup[key]["mean_scaled_mse"]), rel_tol=1e-12, abs_tol=1e-13):
                raise ValueError(f"Equal-collection aggregation differs for {key}")
            aggregated[key] = value

    def f(value):
        if not math.isfinite(value):
            raise ValueError("Nonfinite display value")
        if value >= 10000:
            mantissa, exponent = f"{value:.3e}".split("e")
            return rf"${mantissa}\times 10^{{{int(exponent)}}}$"
        return f"{value:.4f}"

    table = [
        "% Generated from saved amended development evidence; no model fitting.",
        r"\begin{table}[!htbp]", r"\centering\small\setlength{\tabcolsep}{5pt}",
        r"\caption{Budgeted adaptive development study: scaled MSE, lower is better. "
        r"Origins are averaged within horizon, horizons within series, and series within collection. "
        r"The natural mean gives Dominick and Sunspot equal weight. The two capacity tiers "
        r"are quotas, not equal-byte allocations. All non-ridge arms, including KRR, "
        r"use calibration-selected mixing with the common ridge baseline. "
        r"No losses are pooled with earlier studies.}",
        r"\label{tab:adaptive-development-loss}",
        r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"Method & Stationary & Recurring & Dominick & Sunspot & Natural mean \\",
    ]
    for budget, caption in (("small", "Small: 64 shared references / 128 replay rows per horizon"),
                            ("large", "Large: 128 shared references / 256 replay rows per horizon")):
        table.extend([r"\midrule", rf"\multicolumn{{6}}{{l}}{{\textit{{{caption}}}}} \\"])
        for method, label in METHODS:
            values = [f(aggregated[(budget, c, method)]) for c in (*COLLECTIONS, "natural_equal_collection")]
            table.append(" & ".join([label, *values]) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    resources = [
        "% Generated from saved task-level resource measurements.",
        r"\begin{table}[!htbp]", r"\centering\small\setlength{\tabcolsep}{5pt}",
        r"\caption{Measured service and state by method/tier. P99 is first computed "
        r"per stream, then maximized across the 12 streams. Other columns also report "
        r"maxima; they may occur on different streams. Owned NumPy buffers exclude "
        r"Python objects and temporary work arrays; worker RSS includes interpreter, "
        r"libraries and evidence storage. Neither is a precise incremental deployment footprint.}",
        r"\label{tab:adaptive-development-resources}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Method & Tier & P99 (ms) & Max (ms) & NumPy (KiB) & RSS (MiB) \\", r"\midrule",
    ]
    maxima = {}
    for method, label in METHODS:
        for budget in BUDGETS:
            selected = [r for r in tasks if r["method"] == method and r["budget"] == budget]
            vals = (
                max(float(r["p99_service_ms"]) for r in selected),
                max(float(r["max_service_ms"]) for r in selected),
                max(int(r["max_observed_owned_numpy_bytes"]) for r in selected) / 1024,
                max(float(r["process_peak_rss_kib"]) for r in selected) / 1024,
            )
            maxima[(budget, method)] = vals
            resources.append(" & ".join([label, budget.capitalize(), *(f"{v:.2f}" for v in vals)]) + r" \\")
    resources.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    metrics = {"RunCount": "144", "ForecastCount": "387{,}900", "AuditChecks": f"{audit['checks']:,}".replace(",", "{,}")}
    for budget in BUDGETS:
        tier = next(t for t in decision["capacity_tiers"] if t["budget"] == budget)
        if (tier["decision"], tier["mechanism_gate"], tier["natural_gate"], tier["p99_allowance_gate"]) != ("DEFER", False, False, True):
            raise ValueError("Tier decision no longer matches the manuscript")
        if not math.isclose(tier["worst_kahm_stream_p99_ms"], maxima[(budget, "adaptive_kahm")][0], rel_tol=1e-12):
            raise ValueError("Decision and task timing maxima disagree")
        kahm = aggregated[(budget, "natural_equal_collection", "adaptive_kahm")]
        kernel = aggregated[(budget, "natural_equal_collection", "kernel_rr")]
        metrics[budget.title() + "KRRExcess"] = f"{100 * (kahm / kernel - 1):.2f}"
        metrics[budget.title() + "PninetyNine"] = f"{maxima[(budget, 'adaptive_kahm')][0]:.2f}"
        natural = [r for r in tasks if r["budget"] == budget and r["method"] == "adaptive_kahm" and r["kind"] == "natural"]
        if (sum(int(r["active_horizons"]) for r in natural), sum(int(r["horizon_count"]) for r in natural)) != (2, 12):
            raise ValueError("Natural correction-activation count differs")
    kahm_tasks = [r for r in tasks if r["method"] == "adaptive_kahm"]
    metrics["Overruns"] = str(sum(int(r["allowance_overruns"]) for r in kahm_tasks))
    metrics["ServiceSteps"] = f"{sum(int(r['evaluation_observations']) for r in kahm_tasks):,}".replace(",", "{,}")
    peak = next(r for r in peaks["peaks"] if r["method"] == "adaptive_kahm")
    diagnostic = next(r for r in numerical["cases"] if r["method"] == "adaptive_kahm")
    if (peak["task_id"], peak["origin"], peak["horizon"]) != (diagnostic["task_id"], diagnostic["origin"], diagnostic["horizon"]):
        raise ValueError("Numerical diagnosis and amended peak concern different states")
    if not math.isclose(peak["correction"], diagnostic["fresh_centered_svd_correction"], rel_tol=1e-10):
        raise ValueError("Fresh-solve diagnostic no longer agrees with the amended correction")
    metrics["PeakPrediction"] = f"{peak['prediction']:,.2f}".replace(",", "{,}")
    metrics["PeakTruth"] = f"{peak['truth']:.4f}"
    metrics["QueryFeatureMagnitude"] = f"{diagnostic['query_feature_max_abs'] / 1e6:.2f}"
    metrics["ReplayFeatureMagnitude"] = f"{diagnostic['replay_feature_max_abs']:.0f}"
    metrics["RelativeDenominator"] = f"{diagnostic['feature_diagnostics']['minimum_relative_denominator']:.3f}"
    numbers = "% Generated numerical macros from the amended adaptive development evidence.\n"
    numbers += "\n".join(rf"\providecommand{{\AD{k}}}{{{v}}}" for k, v in metrics.items()) + "\n"

    output.mkdir(parents=True, exist_ok=True)
    contents = {"loss_table.tex": "\n".join(table), "resource_table.tex": "\n".join(resources), "numbers.tex": numbers}
    for name, content in contents.items():
        (output / name).write_text(content)
    manifest = {
        "schema_version": 1,
        "scope": "Manuscript displays for completed amended development evidence; no model fitting or new evaluation.",
        "builder": description(Path(__file__)),
        "inputs": inputs,
        "outputs": {name: description(output / name) for name in sorted(contents)},
        "checks": {
            "task_records": len(tasks), "collection_rows": len(collections), "scored_forecasts": 387900,
            "task_to_collection_aggregation": "passed", "natural_equal_collection_aggregation": "passed",
            "amendment_accounting": "passed", "resource_decision_consistency": "passed",
            "preserved_diagnostic_consistency": "passed", "new_model_fits": 0,
        },
    }
    (output / "ASSET_PROVENANCE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"status": "passed", "output": str(output), **manifest["checks"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.study, args.output), indent=2))
