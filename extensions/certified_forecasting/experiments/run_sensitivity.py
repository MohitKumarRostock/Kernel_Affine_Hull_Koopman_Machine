#!/usr/bin/env python3
"""Follow-up coefficient-budget study using the unchanged forecasting runner.

The three fixed budgets use new seeds, with common random numbers across budgets.
Each budget receives delta=.05/3. The resulting joint event covers the three
families and refinement gates within one system/seed replicate, not all twelve
system/seed replicates simultaneously. No budget is selected using test risk.

Examples, from the extension root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    python experiments/run_sensitivity.py --workers 3
  python experiments/run_sensitivity.py --summarize-only
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from run_forecasting import write_json


BUDGETS = [1, 4, 16]
SEEDS = [2701, 2702, 2703]
SYSTEMS = ["harmonic", "quasiperiodic", "nonlinear_ar", "logistic"]
HORIZONS = [1, 4, 12]
FAMILIES = ["kahm_nested", "ridge_ar", "polynomial"]
SYSTEM_LABELS = {
    "harmonic": "Harmonic", "quasiperiodic": "Quasiperiodic",
    "nonlinear_ar": "Nonlinear AR", "logistic": "Logistic",
}


def validate_configs(root):
    base = json.loads((root / "configs/main_v1.json").read_text())
    changed = {"campaign", "seeds", "histories", "kappa", "delta"}
    for budget in BUDGETS:
        path = root / f"configs/sensitivity_v1_k{budget}.json"
        cfg = json.loads(path.read_text())
        if cfg["seeds"] != SEEDS or cfg["histories"] != [12]:
            raise ValueError(f"Unexpected seed/history protocol in {path}")
        if cfg["systems"] != SYSTEMS or cfg["horizons"] != HORIZONS:
            raise ValueError(f"Unexpected system/horizon protocol in {path}")
        if cfg["kappa"] != budget or not np.isclose(cfg["delta"], .05/3, rtol=0, atol=1e-15):
            raise ValueError(f"Unexpected norm or confidence budget in {path}")
        for name, value in base.items():
            if name not in changed and cfg[name] != value:
                raise ValueError(f"Sensitivity changes unexpected setting {name}")


def run_budget(root, budget, logdir):
    config = Path(f"configs/sensitivity_v1_k{budget}.json")
    output = Path(f"results/sensitivity_v1_k{budget}")
    logfile = logdir / f"k{budget}.log"
    started = time.perf_counter()
    with logfile.open("x") as stream:
        completed = subprocess.run(
            [sys.executable, "experiments/run_forecasting.py", "--config", str(config), "--output", str(output)],
            cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    print(f"FINISHED kappa={budget} returncode={completed.returncode} seconds={time.perf_counter()-started:.1f}", flush=True)
    if completed.returncode:
        raise RuntimeError(f"Budget {budget} failed; retained output and log at {logfile}")
    return {"budget": budget, "seconds": time.perf_counter()-started, "returncode": completed.returncode}


def collect(root):
    detail = []
    streams = {}
    confidence_deltas = []
    expected_files = []
    for budget in BUDGETS:
        for system in SYSTEMS:
            for seed in SEEDS:
                path = root / f"results/sensitivity_v1_k{budget}/{system}_{seed}/results.json"
                if not path.exists():
                    raise RuntimeError(f"Missing required result {path}")
                expected_files.append(path)
                run = json.loads(path.read_text())
                if run["system"] != system or run["seed"] != seed:
                    raise ValueError(f"Unexpected run identity in {path}")
                if run["n_candidates"] != 21:
                    raise ValueError(f"Expected 21 candidates after fixing history, got {run['n_candidates']}")
                if not np.isclose(run["delta"], .05/3, rtol=0, atol=1e-15):
                    raise ValueError(f"Unexpected confidence allocation in {path}")
                key = (system, seed)
                if key in streams and streams[key] != run["streams"]:
                    raise ValueError("Common random number streams differ across budgets")
                streams[key] = run["streams"]
                confidence_deltas.append(run["delta"])
                kahm = [r for r in run["results"] if r["family"] == "kahm_nested"]
                any_kahm_pass = any(all(c["rmse_upper"] <= .35 for c in r["certificates"]) for r in kahm)
                for family in FAMILIES:
                    choices = [r for r in run["results"] if r["family"] == family]
                    chosen = min(choices, key=lambda r: (float(np.mean(r["validation_mse"])), r["n_features"], r["name"]))
                    detail.append({
                        "system": system, "seed": seed, "kappa": budget,
                        "family": family, "chosen_on_validation": chosen["name"],
                        "n_features": chosen["n_features"],
                        "validation_mse": chosen["validation_mse"],
                        "test_mse": chosen["test_mse"],
                        "test_rmse": np.sqrt(chosen["test_mse"]).tolist(),
                        "rmse_upper": [c["rmse_upper"] for c in chosen["certificates"]],
                        "rmse_lower": [c["rmse_lower"] for c in chosen["certificates"]],
                        "passes_all_horizons_at_0.35": all(c["rmse_upper"] <= .35 for c in chosen["certificates"]),
                        "any_kahm_candidate_passes_all_horizons_at_0.35": any_kahm_pass,
                    })
    grouped = []
    for system in SYSTEMS:
        for budget in BUDGETS:
            for family in FAMILIES:
                rows = [r for r in detail if r["system"]==system and r["kappa"]==budget and r["family"]==family]
                if len(rows) != len(SEEDS):
                    raise ValueError("Incomplete replicate group")
                rmse = np.asarray([r["test_rmse"] for r in rows])
                grouped.append({
                    "system": system, "kappa": budget, "family": family,
                    "n_replicates": len(rows),
                    "mean_test_rmse": rmse.mean(axis=0).tolist(),
                    "sd_test_rmse": rmse.std(axis=0, ddof=1).tolist(),
                    "mean_selected_features": float(np.mean([r["n_features"] for r in rows])),
                    "all_horizons_certified_count_at_0.35": sum(r["passes_all_horizons_at_0.35"] for r in rows),
                    "any_kahm_certified_count_at_0.35": sum(r["any_kahm_candidate_passes_all_horizons_at_0.35"] for r in rows),
                })
    result = {
        "study": "Follow-up sensitivity after examining the main experiment; new independent seeds",
        "budgets": BUDGETS, "seeds": SEEDS, "systems": SYSTEMS,
        "horizons": HORIZONS, "history": 12, "n_runs": len(expected_files),
        "confidence_scope": "Joint 0.95 over all three budget-specific frozen families and paired gates within each system/seed replicate; not simultaneous over all twelve replicates",
        "family_selection": "Lowest mean validation MSE across horizons; no test-based budget selection",
        "common_random_numbers_across_budgets_verified": True,
        "all_results_retained": True,
        "details": detail, "grouped": grouped,
        "result_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in expected_files},
    }
    write_json(root / "results/sensitivity_summary.json", result)
    csv_rows = []
    for row in detail:
        for i, horizon in enumerate(HORIZONS):
            csv_rows.append({
                key: value for key, value in row.items()
                if key not in {"validation_mse", "test_mse", "test_rmse", "rmse_upper", "rmse_lower"}
            } | {"horizon": horizon} | {
                key: row[key][i] for key in ["validation_mse", "test_mse", "test_rmse", "rmse_upper", "rmse_lower"]
            })
    with (root / "results/sensitivity_selected_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    make_table(root, result)
    return result


def make_table(root, summary):
    lines = [
        "% Automatically generated by experiments/run_sensitivity.py.",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"System & $\kappa$ & Features & $h=1$ & $h=4$ & $h=12$ & Passes \\",
        r"\midrule",
    ]
    for system in SYSTEMS:
        for budget in BUDGETS:
            row = next(r for r in summary["grouped"] if r["system"]==system and r["kappa"]==budget and r["family"]=="kahm_nested")
            values = [f"{value:.4f}" for value in row["mean_test_rmse"]]
            lines.append(
                f"{SYSTEM_LABELS[system]} & {budget} & {row['mean_selected_features']:.1f} & "
                + " & ".join(values)
                + f" & {row['all_horizons_certified_count_at_0.35']}/3"
                + r" \\")
        if system != SYSTEMS[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (root / "paper/tables/sensitivity_results.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, choices=[1,2,3])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    validate_configs(root)
    if not args.summarize_only:
        logdir = root / "results/sensitivity_logs"
        if logdir.exists() and any(logdir.iterdir()):
            raise RuntimeError("Refusing to overwrite sensitivity logs; use --summarize-only for completed results")
        logdir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_budget, root, budget, logdir): budget for budget in BUDGETS}
            outcomes, errors = [], []
            for future in as_completed(futures):
                try:
                    outcomes.append(future.result())
                except Exception as error:
                    errors.append({"budget": futures[future], "error": repr(error)})
            write_json(logdir / "orchestrator.json", {"outcomes": outcomes, "errors": errors})
            if errors:
                raise RuntimeError(f"{len(errors)} budgets failed; all output and failure records retained")
    summary = collect(root)
    print(json.dumps({"n_runs":summary["n_runs"],"grouped":summary["grouped"]}, indent=2))


if __name__ == "__main__":
    main()
