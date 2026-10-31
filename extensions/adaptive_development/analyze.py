"""Prespecified descriptive analysis and conservative development decision.

Four synthetic repetitions and two deliberately selected natural collections
do not support a broad efficacy claim. No window-level significance tests are
used. The thresholds below are taken from the pre-run config, not fitted to
the observed results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def gain(kahm, comparator):
    return 1.0-kahm/comparator if comparator>0 else None


def analyze(results, output):
    cfg=json.loads((HERE/"config.json").read_text())
    manifest=json.loads((HERE/"prepared_inputs"/"manifest.json").read_text())
    summaries=[json.loads(p.read_text()) for p in sorted(results.glob("*/summary.json"))]
    if len(summaries)!=144:
        raise RuntimeError(f"Expected all 144 runs, found {len(summaries)}; incomplete runs remain failures")
    output.mkdir(parents=True,exist_ok=True)
    task_rows=[]
    horizon_rows=[]
    cap_violations=[]
    for s in summaries:
        identity={k:s[k] for k in ("task_id","kind","collection","method","budget")}
        samples=s["state_samples"]
        final=s["final_state"]
        counters=final.get("counters",{})
        for sample in samples:
            state=sample["summary"]
            violations=[]
            if any(n>s["budget_parameters"]["replay_size"] for n in state["replay_rows_per_horizon"]):
                violations.append("replay")
            if state.get("observed_input_rows",0)>s["budget_parameters"]["replay_size"]:
                violations.append("observed_inputs")
            if state.get("total_geometry_references",0)>s["budget_parameters"]["reference_budget"]:
                violations.append("geometry_references")
            if any(n>s["budget_parameters"]["reference_budget"] for n in state.get("kernel_supports_per_horizon",[])):
                violations.append("kernel_support")
            if violations:
                cap_violations.append({**identity,"time":sample["time"],"violations":violations})
        task_rows.append({**identity,"scaled_mse":s["mean_scaled_mse"],
                          "scored_forecasts":sum(x["count"] for x in s["per_horizon"]),
                          "active_horizons":sum(g>0 for g in s["gamma"]),
                          "horizon_count":len(s["gamma"]),
                          "gamma":"|".join(map(str,s["gamma"])),
                          "initialization_seconds":s["initialization_seconds"],
                          "calibration_service_seconds":s["calibration_service_seconds"],
                          "mean_service_ms":1000*s["evaluation_service"]["mean"],
                          "median_service_ms":1000*s["evaluation_service"]["median"],
                          "p95_service_ms":1000*s["evaluation_service"]["p95"],
                          "p99_service_ms":1000*s["evaluation_service"]["p99"],
                          "max_service_ms":1000*s["evaluation_service"]["maximum"],
                          "mean_update_ms":1000*s["evaluation_update"]["mean"],
                          "mean_prediction_ms":1000*s["evaluation_prediction"]["mean"],
                          "allowance_overruns":s["allowance_overruns"],
                          "evaluation_observations":s["evaluation_observations"],
                          "max_observed_owned_numpy_bytes":max(x["summary"]["owned_numpy_bytes"] for x in samples),
                          "process_peak_rss_kib":s["process_peak_rss_kib"],
                          "framework_error_count":len(s["framework_errors"]),
                          "framework_fallback_forecasts":s["framework_fallback_forecasts"],
                          "prediction_fallback_calls":counters.get("prediction_fallbacks",0),
                          "training_feature_fallback_rows":counters.get("training_feature_fallback_rows",0),
                          "reencoded_feature_fallback_rows":counters.get("reencoded_feature_fallback_rows",0),
                          "geometry_refreshes":counters.get("geometry_refreshes",0),
                          "reference_changes":counters.get("reference_changes",0),
                          "kernel_inverse_rebuilds":final.get("kernel_inverse_rebuilds",0),
                          "kernel_inverse_audit_checks":final.get("kernel_inverse_audit_checks",0)})
        for h in s["per_horizon"]:
            horizon_rows.append({**identity,**h})
    write_csv(output/"task_metrics.csv",task_rows)
    write_csv(output/"horizon_metrics.csv",horizon_rows)
    collection_rows=[]
    for budget in cfg["budgets"]:
        for collection in sorted({r["collection"] for r in task_rows}):
            for method in cfg["methods"]:
                rows=[r for r in task_rows if (r["budget"],r["collection"],r["method"])==
                      (budget,collection,method)]
                collection_rows.append({"budget":budget,"collection":collection,"method":method,
                    "series_or_seed_count":len(rows),"mean_scaled_mse":float(np.mean([r["scaled_mse"] for r in rows])),
                    "mean_service_ms":float(np.mean([r["mean_service_ms"] for r in rows])),
                    "worst_stream_p99_ms":max(r["p99_service_ms"] for r in rows),
                    "max_owned_numpy_kib":max(r["max_observed_owned_numpy_bytes"] for r in rows)/1024,
                    "active_horizons":sum(r["active_horizons"] for r in rows),
                    "horizon_count":sum(r["horizon_count"] for r in rows),
                    "framework_errors":sum(r["framework_error_count"] for r in rows)})
        for method in cfg["methods"]:
            rows=[r for r in collection_rows if r["budget"]==budget and r["method"]==method
                  and r["collection"] in ("dominick_weekly","sunspot_daily")]
            collection_rows.append({"budget":budget,"collection":"natural_equal_collection","method":method,
                "series_or_seed_count":4,"mean_scaled_mse":float(np.mean([r["mean_scaled_mse"] for r in rows])),
                "mean_service_ms":float(np.mean([r["mean_service_ms"] for r in rows])),
                "worst_stream_p99_ms":max(r["worst_stream_p99_ms"] for r in rows),
                "max_owned_numpy_kib":max(r["max_owned_numpy_kib"] for r in rows),
                "active_horizons":sum(r["active_horizons"] for r in rows),
                "horizon_count":sum(r["horizon_count"] for r in rows),
                "framework_errors":sum(r["framework_errors"] for r in rows)})
    write_csv(output/"collection_metrics.csv",collection_rows)
    lookup={(r["budget"],r["collection"],r["method"]):r for r in collection_rows}
    task_lookup={(r["budget"],r["task_id"],r["method"]):r for r in task_rows}
    comparisons=[]
    for budget in cfg["budgets"]:
        for collection in sorted({r["collection"] for r in collection_rows}):
            k=lookup[(budget,collection,"adaptive_kahm")]["mean_scaled_mse"]
            for other in [m for m in cfg["methods"] if m!="adaptive_kahm"]:
                c=lookup[(budget,collection,other)]["mean_scaled_mse"]
                rows=[r for r in task_rows if r["budget"]==budget and r["collection"]==collection
                      and r["method"]=="adaptive_kahm"]
                wins=sum(r["scaled_mse"]<task_lookup[(budget,r["task_id"],other)]["scaled_mse"] for r in rows)
                comparisons.append({"budget":budget,"collection":collection,"comparator":other,
                                    "kahm_loss":k,"comparator_loss":c,"kahm_gain_percent":100*gain(k,c),
                                    "paired_wins":wins if rows else None,"paired_units":len(rows) if rows else None})
    write_csv(output/"comparisons.csv",comparisons)
    cl={(r["budget"],r["collection"],r["comparator"]):r for r in comparisons}
    gates=[]
    g=cfg["advancement_gate"]
    technical_validity=(not cap_violations and all(r["framework_error_count"]==0 for r in task_rows))
    for budget in cfg["budgets"]:
        recurring=[cl[(budget,"synthetic_recurring",m)] for m in g["comparators"]]
        mechanism=(all(r["kahm_gain_percent"]>=100*g["recurring_min_gain"] and
                       r["paired_wins"]>=g["recurring_min_seed_wins"] for r in recurring)
                   and all(cl[(budget,"synthetic_recurring",m)]["kahm_gain_percent"]>=
                           100*g["recurring_min_gain_vs_fixed_and_ridge"] for m in ("fixed_kahm","ridge")))
        natural_aggregate=all(cl[(budget,"natural_equal_collection",m)]["kahm_gain_percent"]>=
                              100*g["natural_aggregate_min_gain"] for m in g["comparators"])
        sources=("dominick_weekly","sunspot_daily")
        one_source=any(all(cl[(budget,source,m)]["kahm_gain_percent"]>=100*g["natural_one_source_min_gain"]
                           for m in g["comparators"]) for source in sources)
        no_large_harm=all(cl[(budget,source,m)]["kahm_gain_percent"]>=-100*g["natural_max_other_source_harm"]
                          for m in g["comparators"] for source in sources)
        krows=[r for r in task_rows if r["budget"]==budget and r["method"]=="adaptive_kahm"]
        resource_ok=all(r["p99_service_ms"]<=1000*g["maximum_p99_service_seconds"] for r in krows)
        no_failures=all(r["framework_error_count"]==0 for r in krows)
        natural=natural_aggregate and one_source and no_large_harm
        proceed=mechanism and natural and resource_ok and no_failures and technical_validity
        decision=("TECHNICAL_BLOCK" if not technical_validity else
                  "PROCEED" if proceed else "REFINE" if mechanism else "DEFER")
        gates.append({"budget":budget,"decision":decision,"mechanism_gate":mechanism,
                      "natural_gate":natural,"natural_aggregate_gate":natural_aggregate,
                      "same_source_beats_all_comparators":one_source,
                      "no_source_harm_above_five_percent":no_large_harm,
                      "p99_allowance_gate":resource_ok,"zero_framework_failures":no_failures,
                      "worst_kahm_stream_p99_ms":max(r["p99_service_ms"] for r in krows)})
    decision=("TECHNICAL_BLOCK" if not technical_validity else
              "PROCEED" if any(x["decision"]=="PROCEED" for x in gates) else
              "REFINE" if any(x["decision"]=="REFINE" for x in gates) else "DEFER")
    result={"study_type":"development","decision":decision,"capacity_tiers":gates,
            "expected_runs":144,"completed_runs":len(task_rows),
            "scored_forecasts":sum(r["scored_forecasts"] for r in task_rows),
            "framework_errors":sum(r["framework_error_count"] for r in task_rows),
            "framework_fallback_forecasts":sum(r["framework_fallback_forecasts"] for r in task_rows),
            "technical_validity":technical_validity,"cap_violations":cap_violations,
            "interpretation":"Gate is conservative development triage, not a hypothesis-test rejection of KAHM. No independent natural-source confirmation is claimed.",
            "scope":"One generator, four paired seeds, two previously inspected collections, one fixed implementation and finite calibration gamma grid; no bandwidth/penalty search.",
            "memory_scope":"Capacity tiers match stated quotas, not byte footprints; kernel has M supports per horizon, geometry shares M across horizons. Unique NumPy state and whole-process peak RSS are separate measurements.",
            "timing_scope":"Single-thread serial fresh worker processes; p99 assessed per stream. 20 ms allowance monitored, not enforced. Warm adaptive state maintained even at zero selected correction.",
            "config_sha256":hashlib.sha256((HERE/"config.json").read_bytes()).hexdigest()}
    write_json(output/"decision.json",result)
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,default=HERE/"results")
    p.add_argument("--output",type=Path,default=HERE/"analysis")
    a=p.parse_args()
    analyze(a.results,a.output)
