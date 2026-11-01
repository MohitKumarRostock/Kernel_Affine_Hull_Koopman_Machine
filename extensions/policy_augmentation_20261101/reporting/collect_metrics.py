#!/usr/bin/env python3
"""Collect verified reporting metrics without changing any scientific decision.

This reporting-only program reads frozen analyses and retained fit summaries.
It neither fits models nor chooses a procedure from terminal outcomes. Both
the synthetic and natural independent audits must have passed before use.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("P0", "P1", "P0plus")
SYNTHETIC_METHODS = {
    **{name: "policy__calibrated__" + name for name in POLICIES},
    "in_sample_KAHM": "crossfit__calibrated__insample_kahm",
    "shared_common_plus_bias": "crossfit__calibrated__shared_bias",
    "in_sample_local_RBF": "crossfit__calibrated__insample_rbf",
    "extra_raw_RBF_residual": "extra__calibrated__extra_rbf",
    "raw_ridge": "legacy__calibrated__ridge",
    "independent_raw_RBF": "legacy__calibrated__raw_rbf",
}
NATURAL_METHODS = {label: key.replace("crossfit__", "residual__")
                   for label, key in SYNTHETIC_METHODS.items()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite reporting value")
        return result
    return value


class Evidence:
    def __init__(self):
        self.files = {}

    def bytes(self, path):
        path = Path(path).resolve()
        payload = path.read_bytes()
        item = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        previous = self.files.setdefault(str(path), item)
        require(previous == item, "Reporting input changed while being read: " + str(path))
        return payload

    def json(self, path):
        return json.loads(self.bytes(path))

    def csv(self, path):
        return [{key: scalar(value) for key, value in row.items()}
                for row in csv.DictReader(io.StringIO(self.bytes(path).decode()))]


def stats(values):
    values = [float(v) for v in values]
    require(bool(values) and all(math.isfinite(v) for v in values), "Invalid reporting statistics")
    return {"count": len(values), "minimum": min(values), "maximum": max(values),
            "mean": mean(values), "median": median(values)}


def counts(values):
    return {str(key): value for key, value in sorted(Counter(values).items(), key=lambda pair: str(pair[0]))}


def point_comparisons(panel_rows, panel_field, method_field, metric_field, method_map):
    """Point estimates are descriptive arithmetic, never an added test."""
    index = {(r[panel_field], r[method_field]): r[metric_field] for r in panel_rows}
    results = []
    for panel in sorted({r[panel_field] for r in panel_rows}):
        for left, right in (("P1", "P0"), ("P1", "P0plus"),
                            ("in_sample_KAHM", "shared_common_plus_bias"),
                            ("in_sample_KAHM", "in_sample_local_RBF"),
                            ("in_sample_KAHM", "P0")):
            a, b = index[panel, method_map[left]], index[panel, method_map[right]]
            results.append({"panel": panel, "candidate": left, "comparator": right,
                            "candidate_loss": a, "comparator_loss": b, "difference": a-b,
                            "gain_percent": 100 * (1-a/b) if b else None,
                            "inference": "descriptive point arithmetic; consult frozen primary family separately"})
    return results


def summary_selection(root, evidence, manifest, natural=False):
    """Read selection metadata, actual ranks and widths; do not load forecasts."""
    selections, dimensions, widths, resource_rows, solver_rows = [], [], [], [], []
    for task in manifest:
        task_id = task["task_id"]
        fit = evidence.json(root / "results" / task_id / "fit_summary.json")
        require(fit["status"] == "completed", "Incomplete fit in reporting")
        owner = fit["residual"] if natural else fit["crossfit"]
        selection = owner["selections"] if natural else owner["selections"]["target_modes"]["insample"]
        common = owner["selections"]["common" if natural else "full_common"]
        meta = {"task_id": task_id, **({"frequency": task["frequency"], "series_id": task["series_id"]}
                if natural else {"law": task["law"], "noise_level": task["noise_level"], "repetition": task["repetition"]})}
        p1_choice = fit["policies"]["P1"]["selected_identity"]
        kahm_admitted = p1_choice == ("residual__insample_kahm" if natural else "crossfit__insample_kahm")
        row = {**meta, "P1_selected_KAHM": kahm_admitted,
               "policy_choices": {name: fit["policies"][name]["selected_identity"] for name in POLICIES}}
        for label, record in (("common", common), ("in_sample_KAHM", selection["kahm"]),
                              ("in_sample_local_RBF", selection["rbf"])):
            row[label] = {name: record[name] for name in ("selected_cap", "selected_scaler", "selected_candidate")}
            key = f"{record['selected_scaler']}__cap{record['selected_cap']}"
            bank = fit["legacy"]["banks"][key]
            for local in bank["local_dimensions"]:
                dimensions.append({**meta, "procedure": label, "P1_selected_KAHM": kahm_admitted,
                                   "requested_cap": record["selected_cap"], "local_rank": local["effective_rank"],
                                   "initial_admissible_rank": local["initial_admissible_rank"],
                                   "local_reference_count": local["local_reference_count"],
                                   "rank_zero_fallback": local["fallback_applied"],
                                   "rank_reduction_reasons": local["rank_reduction_reasons"]})
        k_selected = selection["kahm"]["candidates"][selection["kahm"]["selected_index"]]
        r_selected = selection["rbf"]["candidates"][selection["rbf"]["selected_index"]]
        e_selection = fit["extra"]["selections"]
        e_selected = e_selection["candidates"][e_selection["selected_index"]]
        row["extra_raw_RBF_residual"] = {
            "selected_scaler": e_selection["selected_scaler"],
            "selected_bandwidth_factor": e_selection["selected_bandwidth_factor"],
            "selected_candidate": e_selection["selected_candidate"]}
        selections.append(row)
        widths.append({**meta, "P1_selected_KAHM": kahm_admitted,
                       "KAHM_feature_width": k_selected["feature_width"],
                       "local_RBF_feature_width": r_selected["feature_width"],
                       "extra_RBF_feature_width": e_selected["feature_width"],
                       "extra_RBF_centered_feature_rank": e_selected["centered_feature_rank"],
                       "extra_RBF_nystrom_rank": e_selected["nystrom_rank"],
                       "extra_RBF_landmark_count": e_selected["landmark_count"]})
        prediction = evidence.json(root / "predictions" / task_id / "prediction_receipt.json")
        resources = fit["resources"]
        resource_rows.append({**meta, "fit_wall_seconds": resources["total_wall_seconds"],
                              "fit_cpu_seconds": resources["total_cpu_seconds"],
                              "peak_worker_rss_MiB": resources["peak_rss_KiB"]/1024,
                              "state_gzip_bytes": resources["state_gzip_bytes"],
                              "prediction_wall_seconds": prediction["wall_seconds"],
                              "prediction_cpu_seconds": prediction.get("cpu_seconds"),
                              "extra_fit_wall_seconds": fit["extra"]["diagnostics"]["fit_wall_seconds"],
                              "extra_fit_cpu_seconds": fit["extra"]["diagnostics"]["fit_cpu_seconds"],
                              "fit_completed_utc": fit["utc" if natural else "completed_utc"],
                              "prediction_completed_utc": prediction["utc"]})
        if natural:
            events = fit["numerical_solver"]["events"]
            solver_rows.append({**meta, "fallback_count": len(events),
                                "recovered_count": sum(e["status"] == "recovered" for e in events),
                                "unrecovered_count": sum(e["status"] != "recovered" for e in events)})
        else:
            solver = evidence.json(root / "results" / task_id / "numerical_solver_events.json")
            solver_rows.append({**meta, "fallback_count": solver["fallback_count"],
                                "recovered_count": solver["recovered_count"],
                                "unrecovered_count": solver["failed_count"]})
    cap_summary, rank_summary, width_summary = [], [], []
    for subset in ("all_tasks", "P1_KAHM_selected"):
        selected_rows = [r for r in selections if subset == "all_tasks" or r["P1_selected_KAHM"]]
        for label in ("common", "in_sample_KAHM", "in_sample_local_RBF"):
            cap_summary.append({"subset": subset, "procedure": label, "task_count": len(selected_rows),
                                "cap_counts": counts(r[label]["selected_cap"] for r in selected_rows),
                                "scaler_counts": counts(r[label]["selected_scaler"] for r in selected_rows),
                                "cap_scaler_counts": counts(r[label]["selected_candidate"] for r in selected_rows)})
            rows = [r for r in dimensions if r["procedure"] == label and
                    (subset == "all_tasks" or r["P1_selected_KAHM"])]
            rank_summary.append({"subset": subset, "procedure": label, "local_model_occurrences": len(rows),
                                 "effective_rank": stats([r["local_rank"] for r in rows]) if rows else None,
                                 "rank_counts": counts(r["local_rank"] for r in rows),
                                 "initial_admissible_rank_counts": counts(r["initial_admissible_rank"] for r in rows),
                                 "rank_zero_fallback_occurrences": sum(r["rank_zero_fallback"] for r in rows),
                                 "reduction_reason_counts": counts(reason for r in rows for reason in r["rank_reduction_reasons"]),
                                 "scope": "Occurrences in selected banks; a shared bank can occur in more than one procedure role"})
        chosen_widths = [r for r in widths if subset == "all_tasks" or r["P1_selected_KAHM"]]
        for measure in ("KAHM_feature_width", "local_RBF_feature_width", "extra_RBF_feature_width",
                        "extra_RBF_centered_feature_rank", "extra_RBF_nystrom_rank", "extra_RBF_landmark_count"):
            width_summary.append({"subset": subset, "measure": measure,
                                  "values": stats([r[measure] for r in chosen_widths]) if chosen_widths else None})
    resource_summary = {}
    for key in ("fit_wall_seconds", "fit_cpu_seconds", "prediction_wall_seconds", "prediction_cpu_seconds",
                "extra_fit_wall_seconds", "extra_fit_cpu_seconds", "peak_worker_rss_MiB", "state_gzip_bytes"):
        values = [r[key] for r in resource_rows if r[key] is not None]
        resource_summary[key] = stats(values) if values else None
        if values and key not in ("peak_worker_rss_MiB",):
            resource_summary[key]["sum"] = sum(values)
    stage = {}
    for phase in ("fit", "predict"):
        path = root/"execution"/phase/"stage_receipt.json"
        if path.is_file():
            stage[phase] = {"measurement": "execution-stage stopwatch", "receipt": evidence.json(path)}
        else:
            stage[phase] = {"measurement": "unavailable; report worker-duration sums separately",
                            "stage_wall_seconds": None,
                            "limitation": "No exact stage stopwatch was retained; launch and finalization overhead are not identified"}
    return {"task_selection_details": selections, "cap_scaler_counts": cap_summary,
            "selected_local_rank_summary": rank_summary, "feature_width_summary": width_summary,
            "feature_width_by_task": widths, "resources_by_task": resource_rows,
            "resource_summary": resource_summary, "execution_stage_times": stage,
            "numerical_recovery": {"fallback_count": sum(r["fallback_count"] for r in solver_rows),
                                   "recovered_count": sum(r["recovered_count"] for r in solver_rows),
                                   "unrecovered_count": sum(r["unrecovered_count"] for r in solver_rows)},
            "cost_scope": "Complete measured research suite, including rejected and diagnostic candidates; not isolated policy deployment latency"}


def collect(synthetic_root, natural_root, output):
    synthetic_root, natural_root, output = map(lambda p: Path(p).resolve(), (synthetic_root,natural_root,output))
    require(not output.exists(), "Refusing to overwrite an earlier metrics record")
    ev = Evidence()
    # Check both gates before reading any substantive result for this report.
    sa = ev.json(synthetic_root/"verification/policy_audit.json")
    na = ev.json(natural_root/"audit/verification.json")
    require(sa["status"] == na["status"] == "passed", "Both independent audits must pass before reporting")
    require(sa["analysis_summary_sha256"] == digest(synthetic_root/"analysis/analysis_summary.json"),
            "Synthetic audit does not authenticate the current analysis")
    require(na["analysis_summary_sha256"] == digest(natural_root/"analysis/analysis_summary.json"),
            "Natural audit does not authenticate the current analysis")
    ss, ns = ev.json(synthetic_root/"analysis/analysis_summary.json"), ev.json(natural_root/"analysis/analysis_summary.json")
    require(ss["tasks"] == 192 and ss["independent_repetitions"] == 24 and ns["series_count"] == 48,
            "Unexpected evidence cohort sizes")
    for name, record in ss["tables"].items():
        require(digest(synthetic_root/"analysis"/name) == record["sha256"], "Changed synthetic table")
    for name, value in ns["files"].items():
        require(digest(natural_root/"analysis"/name) == value, "Changed natural table")
    synthetic_config, natural_config = ev.json(synthetic_root/"config.json"), ev.json(natural_root/"config.json")
    synthetic_panels = ev.csv(synthetic_root/"analysis/panel_summary.csv")
    synthetic_tasks = ev.csv(synthetic_root/"analysis/task_losses.csv")
    panel_means = [{"procedure": label, **row} for row in synthetic_panels for label,key in SYNTHETIC_METHODS.items()
                   if row["forecast_key"] == key]
    cells = defaultdict(list)
    for row in synthetic_tasks:
        if row["forecast_key"] in SYNTHETIC_METHODS.values():
            cells[row["law"],row["noise_level"],row["forecast_key"]].append(row["smse"])
    labels = {key:label for label,key in SYNTHETIC_METHODS.items()}
    condition_means = []
    for (law,noise,key), values in sorted(cells.items()):
        require(len(values) == 24, "Incomplete condition summary")
        condition_means.append({"law":law,"noise_level":noise,"procedure":labels[key],"forecast_key":key,
                                "mean_smse":mean(values),"repetitions":24,"inference":"descriptive condition summary"})
    synthetic_manifest = ev.json(synthetic_root/"prepared/abc_manifest.json")["tasks"]
    synthetic_fit = summary_selection(synthetic_root,ev,synthetic_manifest)
    natural_aggregates = ev.csv(natural_root/"analysis/aggregates.csv")
    natural_task_rows = ev.csv(natural_root/"analysis/task_methods.csv")
    natural_manifest = ev.json(natural_root/"prepared/abc_manifest.json")["tasks"]
    natural_fit = summary_selection(natural_root,ev,natural_manifest,natural=True)
    natural_means = [{"procedure":label, **row} for row in natural_aggregates for label,key in NATURAL_METHODS.items()
                    if row["method_key"] == key]
    per_task = {(r["task_id"],r["method_key"]):r for r in natural_task_rows}
    natural_roster=[]
    for row in natural_fit["task_selection_details"]:
        natural_roster.append({"task_id":row["task_id"],"series_id":row["series_id"],"frequency":row["frequency"],
                               "P1_selected_KAHM":row["P1_selected_KAHM"],"policy_choices":row["policy_choices"],
                               "policy_losses":{name:{metric:per_task[row["task_id"],NATURAL_METHODS[name]][metric]
                                                         for metric in ("SMSE","seasonal_MASE","frequency_MASE")}
                                                for name in POLICIES}})
    report={"schema":"kahm-verified-policy-report-metrics-v1","status":"verified_report_metrics",
            "created_utc":datetime.now(timezone.utc).isoformat(),"collector_sha256":digest(__file__),
            "scientific_decisions_changed":False,"new_model_fits":0,
            "synthetic":{"design":{"tasks":192,"independent_repetitions":24,"laws":synthetic_config["laws"],
                                    "noise_levels":synthetic_config["noise_levels"],"horizons":synthetic_config["horizons"],
                                    "master_seed":synthetic_config["master_seed"],"primary_family_size":4},
                         "verification":sa,"primary_results":ss["primary_results"],
                         "criteria_established":ss["criteria_established"],"panel_means":panel_means,
                         "descriptive_point_comparisons":point_comparisons(synthetic_panels,"panel","forecast_key","mean_smse",SYNTHETIC_METHODS),
                         "condition_means":condition_means,
                         "selection_frequencies":ev.csv(synthetic_root/"analysis/selection_frequencies.csv"),
                         "conditional_descriptive":ev.csv(synthetic_root/"analysis/conditional_descriptive.csv"),
                         **synthetic_fit},
            "natural":{"design":{"series_count":48,"frequency_panels":ns["frequencies"],"origins_per_series":32,
                                  "horizons":ns["horizons"],"evidence_status":ns["evidence_status"],
                                  "inference":ns["statistical_inference"],"official_M4_Test_files_downloaded":False,
                                  "policy_roster_differs_from_simulation":True,
                                  "policy_difference":"Natural P0 omits episode cross-fit RBF and adds seasonal naive and drift controls"},
                       "verification":na,"analysis_summary":ns,"frequency_panel_means":natural_means,
                       "SMSE_descriptive_point_comparisons":point_comparisons(natural_aggregates,"panel","method_key","SMSE",NATURAL_METHODS),
                       "policy_contrasts":ns["policy_contrasts"],"selection_counts":ns["selection_counts"],
                       "conditional_subsets":ns["conditional_subsets"],"roster_and_policy_choices":natural_roster,
                       "MASE_denominator_floors":ns["MASE_denominator_floors"],
                       **natural_fit},
            "interpretation_limits":["Only the four frozen synthetic criteria receive confirmatory decision labels.",
                                     "Synthetic and natural losses use different designs and are not pooled.",
                                     "Natural benchmark-development results are descriptive, not official M4 test rankings or independent-source inference.",
                                     "Conditional selected-subset and per-condition differences are descriptive; selection frequencies are not causal usefulness tests.",
                                     "Candidate-score matching does not equalize feature width, runtime, effective rank or statistical capacity.",
                                     "Nested C selection does not guarantee terminal nondegradation.",
                                     "Worker-duration sums, peak individual-worker RSS and execution-stage elapsed time are different measurements."]}
    report["input_provenance"] = sorted(ev.files.values(),key=lambda x:x["path"])
    # Stable compact aliases for the report builder. Full detail remains above.
    syn, nat = report["synthetic"], report["natural"]
    syn["policy_panel_means"] = [r for r in syn["panel_means"] if r["procedure"] in POLICIES]
    syn["standalone_panel_means"] = [r for r in syn["panel_means"] if r["procedure"] not in POLICIES]
    syn["policy_selections"] = {"frequencies": syn["selection_frequencies"],
                                 "conditional_descriptive": syn["conditional_descriptive"],
                                 "task_details": syn["task_selection_details"]}
    syn["dimension_selection"] = {key: syn[key] for key in
                                    ("cap_scaler_counts", "selected_local_rank_summary", "feature_width_summary")}
    syn["costs"] = {"summary": syn["resource_summary"], "execution_stage_times": syn["execution_stage_times"],
                     "scope": syn["cost_scope"]}
    syn["audit"] = sa
    nat["contrasts"] = nat["policy_contrasts"]
    nat["policy_means_by_panel"] = [r for r in nat["frequency_panel_means"] if r["procedure"] in POLICIES]
    nat["conditional"] = nat["conditional_subsets"]
    nat["costs"] = {"summary": nat["resource_summary"], "execution_stage_times": nat["execution_stage_times"],
                     "scope": nat["cost_scope"]}
    nat["audit"] = na
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root",type=Path,default=ROOT)
    parser.add_argument("--natural-root",type=Path,default=ROOT.parent/"policy_augmentation_natural_20261101")
    parser.add_argument("--output",type=Path,default=ROOT.parents[2]/"deliverables/KAHM_Policy_Augmentation_Metrics.json")
    args=parser.parse_args()
    value=collect(args.synthetic_root,args.natural_root,args.output)
    print(json.dumps({"status":value["status"],"output":str(args.output.resolve()),
                      "input_files":len(value["input_provenance"]),"synthetic_primary_passes":value["synthetic"]["criteria_established"],
                      "natural_inference":"descriptive only"}))


if __name__=="__main__":
    main()
