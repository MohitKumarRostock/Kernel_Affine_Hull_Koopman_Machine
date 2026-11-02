#!/usr/bin/env python3
"""Audit positive-tolerance decisions using every retained original JSON record.

No sample-size curve is inferred from full-sample moments. No model is refitted.
The confidence scope remains that of each original complete repetition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = ("main_v1", "oscillators_v1", "sensitivity_v1_k1",
             "sensitivity_v1_k4", "sensitivity_v1_k16")
TOLERANCES = (0.20, 0.35, 0.50)
HORIZONS = (1, 4, 12)


def decision(lower, upper, tolerance):
    accepted = all(u <= tolerance ** 2 for u in upper)
    excluded = any(l > tolerance ** 2 for l in lower)
    if accepted and excluded:
        raise ValueError("Inconsistent retained lower/upper decisions")
    return "accepted" if accepted else "excluded" if excluded else "unresolved"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    detailed, selected, files, reconstruction_errors = [], [], [], []
    for campaign in CAMPAIGNS:
        for path in sorted((ROOT / "results" / campaign).glob("*/results.json")):
            raw = path.read_bytes()
            report = json.loads(raw)
            files.append({"path": str(path.relative_to(ROOT)),
                          "sha256": hashlib.sha256(raw).hexdigest()})
            candidates = report["results"]
            kahm = [r for r in candidates if r["family"] == "kahm_nested"]
            # Match the retained runner's stable ordering for the full family.
            picked_all = report["selected_by_validation"]
            picked_kahm = min(kahm, key=lambda r: (sum(r["validation_mse"]) / 3,
                                                   r["n_features"], r["name"]))["name"]
            for row in candidates:
                lower = [c["mse_lower"] for c in row["certificates"]]
                upper = [c["mse_upper"] for c in row["certificates"]]
                base = {"campaign": campaign, "system": report["system"],
                        "seed": report["seed"], "candidate": row["name"],
                        "family": row["family"], "n_features": row["n_features"],
                        "validation_selected_all": row["name"] == picked_all,
                        "validation_selected_kahm": row["name"] == picked_kahm}
                for h, lo, hi in zip(HORIZONS, lower, upper):
                    for tolerance in TOLERANCES:
                        detailed.append(dict(base, horizon=h, tolerance=tolerance,
                            lower=lo, upper=hi,
                            status=decision([lo], [hi], tolerance)))
                for tolerance in TOLERANCES:
                    detailed.append(dict(base, horizon="all", tolerance=tolerance,
                        lower=max(lower), upper=max(upper),
                        status=decision(lower, upper, tolerance)))
                if row["name"] in (picked_all, picked_kahm):
                    for scope, name in (("all", picked_all), ("kahm", picked_kahm)):
                        if row["name"] == name:
                            selected.append(dict(base, selection_scope=scope,
                                zero_lower_horizons=sum(x == 0 for x in lower),
                                n_horizons=3, all_lower_zero=all(x == 0 for x in lower),
                                maximum_lower=max(lower), maximum_upper=max(upper)))
            # Recheck the original acceptance-status summaries independently.
            for summary in report.get("certified_selections", []):
                pool = candidates if summary["family"] == "all" else kahm
                eligible = [r for r in pool if all(
                    c["mse_upper"] <= summary["tolerance"] ** 2
                    for c in r["certificates"])]
                if bool(eligible) != (summary["selected"] is not None):
                    reconstruction_errors.append({"file": str(path), "summary": summary})
    grouped = defaultdict(Counter)
    for row in detailed:
        scopes = ["all_candidates"]
        if row["family"] == "kahm_nested":
            scopes.append("kahm_candidates")
        if row["validation_selected_all"]:
            scopes.append("validation_selected_all")
        if row["validation_selected_kahm"]:
            scopes.append("validation_selected_kahm")
        for scope in scopes:
            for system in (row["system"], "ALL"):
                grouped[(row["campaign"], system, scope, row["horizon"],
                         row["tolerance"])][row["status"]] += 1
    aggregate = []
    for (campaign, system, scope, horizon, tolerance), count in sorted(
            grouped.items(), key=lambda z: tuple(str(v) for v in z[0])):
        aggregate.append({"campaign": campaign, "system": system, "scope": scope,
            "horizon": horizon, "tolerance": tolerance,
            "accepted": count["accepted"], "excluded": count["excluded"],
            "unresolved": count["unresolved"], "total": sum(count.values())})
    selected_summary = []
    for campaign in CAMPAIGNS:
        for scope in ("all", "kahm"):
            subset = [r for r in selected if r["campaign"] == campaign
                      and r["selection_scope"] == scope]
            selected_summary.append({"campaign": campaign, "scope": scope,
                "n_selected_models": len(subset),
                "zero_lower_horizons": sum(r["zero_lower_horizons"] for r in subset),
                "n_horizon_statements": 3 * len(subset),
                "models_with_all_lower_zero": sum(r["all_lower_zero"] for r in subset)})
    write_csv(args.output / "existing_candidate_decisions.csv", detailed)
    write_csv(args.output / "existing_decision_counts.csv", aggregate)
    write_csv(args.output / "existing_selected_lower_endpoints.csv", selected)
    result = {"status": "passed" if not reconstruction_errors else "failed",
        "n_runs": len(files), "n_candidates": len(detailed) // (4 * len(TOLERANCES)),
        "tolerances": TOLERANCES, "horizons": HORIZONS,
        "selected_lower_summary": selected_summary,
        "acceptance_reconstruction_errors": reconstruction_errors,
        "sample_size_curves": "not computed: episode score arrays unavailable in the retained Git records",
        "confidence_scope": "original per-repetition events; aggregate counts are descriptive",
        "inputs": files}
    (args.output / "existing_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "n_runs", "n_candidates",
                                           "selected_lower_summary")}, indent=2))
    if reconstruction_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
