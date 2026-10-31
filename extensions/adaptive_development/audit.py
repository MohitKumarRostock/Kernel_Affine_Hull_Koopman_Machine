"""Independent evidence audit without importing the model, runner, or analyzer.

Reconstruct targets, calibration, losses, resource summaries and the development
decision from immutable forecast/arrival records. This verifies consistency of
the supplied execution; it does not independently fit models or establish
scientific replication. Shared-baseline checks use fixed 1e-8 relative/absolute
tolerances to allow different floating-point sufficient-statistic rebuilds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METHODS = ("ridge", "fixed_kahm", "adaptive_kahm", "adaptive_affine",
           "adaptive_rbf", "kernel_rr")
ADAPTIVE = ("adaptive_kahm", "adaptive_affine", "adaptive_rbf")
NATURAL = ("dominick_weekly", "sunspot_daily")


class AuditFailure(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checks:
    def __init__(self):
        self.count = 0

    def require(self, condition, label, count=1):
        self.count += int(count)
        if not bool(condition):
            raise AuditFailure(label)

    def equal(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape and np.array_equal(a, b), label,
                     max(a.size, b.size, 1))

    def close(self, actual, expected, label, rtol=1e-10, atol=1e-10):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == b.shape and
                     np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True),
                     label, max(a.size, b.size, 1))


def independent_scale(values, fit_end, period):
    initial = values[:fit_end]
    for lag in (period, 1):
        if len(initial) > lag:
            differences = initial[lag:] - initial[:-lag]
            differences = differences[np.isfinite(differences)]
            if len(differences):
                scale = float(np.mean(differences ** 2))
                if np.isfinite(scale) and scale > 0:
                    return scale
    return 1.0


def prepared_inputs(check):
    manifest = read_json(HERE / "prepared_inputs/manifest.json")
    tasks = {}
    for entry in manifest["tasks"]:
        path = HERE / "prepared_inputs" / entry["filename"]
        check.require(digest(path) == entry["sha256"], f"Input hash: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"]))
            values = archive["values"].astype(float)
        name = entry["task_id"]
        check.require(name not in tasks, f"Duplicate input: {name}")
        check.equal([meta[k] for k in ("task_id", "kind", "collection")],
                    [entry[k] for k in ("task_id", "kind", "collection")],
                    f"Input metadata: {name}")
        raw_hash = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
        check.require(raw_hash == meta["metadata"]["raw_values_sha256"],
                      f"Raw-array hash: {name}")
        check.equal([meta["fit_end"], meta["eval_start"]],
                    [35 * len(values) // 100, len(values) // 2], f"Boundaries: {name}")
        scale = independent_scale(values, meta["fit_end"], meta["seasonal_lag"])
        check.close(meta["scale"], scale, f"Initial-only scale: {name}")
        if meta["kind"] == "natural":
            original = REPO / meta["metadata"]["source_npz"]
            if original.exists():
                check.require(digest(original) == meta["metadata"]["source_npz_sha256"],
                              f"Inherited source hash: {name}")
                with np.load(original, allow_pickle=False) as inherited:
                    check.close(values, inherited["raw_values"], f"Inherited values: {name}",
                                rtol=0, atol=0)
        tasks[name] = {**meta, "values": values, "scale": scale}
    check.require(len(tasks) == 12, "Expected twelve development streams")
    check.require(manifest["task_count"] == len(tasks), "Input manifest count")
    return manifest, tasks


def timing_summary(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)),
            "maximum": float(np.max(x))}


def audit_run(directory, task, cfg, check):
    s = read_json(directory / "summary.json")
    tr = read_json(directory / "trace.json")
    with np.load(directory / "forecasts.npz", allow_pickle=False) as archive:
        a = {name: archive[name].copy() for name in archive.files}
    key = (s["budget"], s["task_id"], s["method"])
    label = "/".join(key)
    check.require(digest(directory / "forecasts.npz") == s["forecasts_sha256"],
                  f"Forecast hash: {label}")
    check.require(digest(directory / "trace.json") == s["trace_sha256"],
                  f"Trace hash: {label}")
    check.require(not s["timing_only"], f"Timing-only run entered roster: {label}")
    budget = cfg["budgets"][s["budget"]]
    check.require(s["budget_parameters"] == budget, f"Budget metadata: {label}")
    v = task["values"]
    n, length = len(v), task["history"]
    fit, start = task["fit_end"], task["eval_start"]
    hs = np.asarray(task["horizons"], dtype=int)
    for name, expected in (("history", length), ("fit_end", fit),
                           ("evaluation_start", start), ("raw_length", n)):
        check.require(s[name] == expected, f"Run {name}: {label}")
    check.equal(s["horizons"], hs, f"Horizons: {label}")
    check.close(s["scale"], task["scale"], f"Run scale: {label}")
    initial = [o for o in range(length - 1, fit - max(hs))
               if np.isfinite(v[o-length+1:o+1]).all() and np.isfinite(v[o+hs]).all()]
    check.require(len(initial) == s["initial_training_rows"], f"Warm-up rows: {label}")
    check.require(initial[-1] + max(hs) == s["initial_latest_target"] < fit,
                  f"Warm-up maturity: {label}")
    finite_origins = np.asarray([t for t in range(fit, n)
                                if np.isfinite(v[t-length+1:t+1]).all()], dtype=int)
    expected_origins = np.repeat(finite_origins, len(hs))
    expected_horizons = np.tile(hs, len(finite_origins))
    for name in a:
        check.require(a[name].shape == expected_origins.shape,
                      f"Array length {name}: {label}")
    check.equal(a["origin"], expected_origins, f"Complete issued-origin coverage: {label}")
    check.equal(a["horizon"], expected_horizons, f"Issued horizon coverage: {label}")
    check.equal(a["horizon_index"], np.tile(np.arange(len(hs)), len(finite_origins)),
                f"Horizon indices: {label}")
    check.equal(a["target"], expected_origins + expected_horizons,
                f"Future target indices: {label}")
    truths = np.full(len(expected_origins), np.nan)
    in_range = a["target"] < n
    truths[in_range] = v[a["target"][in_range]]
    check.close(a["truth"], truths, f"Targets reconstructed from raw stream: {label}",
                rtol=0, atol=0)
    check.equal(a["evaluation"], a["origin"] >= start, f"Evaluation flags: {label}")
    check.require(np.isfinite(a["baseline"]).all() and
                  np.isfinite(a["correction"]).all() and
                  np.isfinite(a["prediction"]).all(), f"Finite issued components: {label}")
    selected = []
    for hi, h in enumerate(hs):
        use = (a["horizon"] == h) & (a["origin"] < start) & (a["target"] < start)
        use &= np.isfinite(truths)
        candidate_losses = [float(np.mean((truths[use] - (a["baseline"][use] +
                                           g * a["correction"][use])) ** 2))
                            for g in cfg["gamma_grid"]]
        check.require(use.any(), f"Calibration availability: {label}/{h}")
        g = cfg["gamma_grid"][int(np.argmin(candidate_losses))] if s["method"] != "ridge" else 0.0
        selected.append(g)
        cal = s["calibration"][hi]
        check.require(cal["horizon"] == h and cal["rows"] == int(use.sum()),
                      f"Calibration count: {label}/{h}")
        check.require(cal["latest_target"] == int(a["target"][use].max()) < start,
                      f"Calibration maturity: {label}/{h}")
        check.equal([r["gamma"] for r in cal["trials"]], cfg["gamma_grid"],
                    f"Gamma grid: {label}/{h}")
        check.close([r["mse"] for r in cal["trials"]], candidate_losses,
                    f"Independent calibration losses: {label}/{h}")
        check.require(cal["gamma"] == g, f"First-minimum calibration choice: {label}/{h}")
    selected = np.asarray(selected)
    check.equal(s["gamma"], selected, f"Selected weights: {label}")
    expected_gamma = np.where(a["origin"] >= start, selected[a["horizon_index"]], 0.0)
    check.equal(a["gamma"], expected_gamma, f"No evaluation retuning: {label}")
    check.close(a["prediction"], a["baseline"] + expected_gamma * a["correction"],
                f"Forecast component mixing: {label}", rtol=1e-12, atol=1e-12)
    check.equal(a["prediction"][expected_gamma == 0], a["baseline"][expected_gamma == 0],
                f"Exact zero-correction baseline: {label}")
    valid = (a["origin"] >= start) & np.isfinite(truths)
    losses = []
    for hi, h in enumerate(hs):
        use = valid & (a["horizon"] == h)
        errors = truths[use] - a["prediction"][use]
        check.require(len(errors) > 0, f"Scorable evaluation horizon: {label}/{h}")
        mse, mae = float(np.mean(errors ** 2)), float(np.mean(np.abs(errors)))
        losses.append(mse / task["scale"])
        reported = s["per_horizon"][hi]
        check.require(reported["horizon"] == h and reported["count"] == len(errors),
                      f"Scored coverage: {label}/{h}")
        check.close([reported["mse"], reported["mae"], reported["scaled_mse"]],
                    [mse, mae, losses[-1]], f"Independent horizon losses: {label}/{h}")
    loss = float(np.mean(losses))
    check.close(s["mean_scaled_mse"], loss, f"Equal-horizon task loss: {label}")
    check.equal([t["time"] for t in tr], np.arange(fit, n), f"Complete arrival trace: {label}")
    for record in tr:
        t = record["time"]
        check.require(bool(record["evaluation"]) == (t >= start), f"Trace phase: {label}/{t}")
        matured = sum(t-h > initial[-1] and t-h >= length-1 and np.isfinite(v[t]) and
                      np.isfinite(v[t-h-length+1:t-h+1]).all() for h in hs)
        check.require(record["matured_count"] == matured, f"Matured update count: {label}/{t}")
        check.require(record["latest_supervised_target"] == (t if matured else -1),
                      f"No future supervised label: {label}/{t}")
        components = [record[k] for k in ("update_seconds", "prediction_seconds", "service_seconds")]
        check.require(np.isfinite(components).all() and min(components) >= 0 and
                      components[2] + 1e-9 >= components[0] + components[1],
                      f"Complete service accounting: {label}/{t}")
    disabled = np.asarray([t["disabled"] for t in tr], dtype=bool)
    check.require(not np.any(np.diff(disabled.astype(int)) < 0), f"Permanent framework fallback: {label}")
    check.equal(a["framework_fallback"], disabled[a["origin"] - fit], f"Fallback flags: {label}")
    fallback_count = int(np.sum(a["framework_fallback"][valid]))
    check.require(s["framework_fallback_forecasts"] == fallback_count, f"Fallback count: {label}")
    evaluation = [t for t in tr if t["time"] >= start]
    for output, field in (("evaluation_service", "service_seconds"),
                          ("evaluation_update", "update_seconds"),
                          ("evaluation_prediction", "prediction_seconds")):
        calculated = timing_summary([t[field] for t in evaluation])
        for name, value in calculated.items():
            check.close(s[output][name], value, f"Timing {output}/{name}: {label}")
    check.require(s["evaluation_observations"] == n-start, f"Observation count: {label}")
    over = sum(t["service_seconds"] > cfg["observation_allowance_seconds"] for t in evaluation)
    check.require(s["allowance_overruns"] == over, f"Allowance overruns: {label}")
    check.close(s["calibration_service_seconds"],
                sum(t["service_seconds"] for t in tr if t["time"] < start),
                f"Calibration cost: {label}")
    cap_violations = []
    samples = s["state_samples"]
    check.require(samples[0]["time"] == fit-1 and samples[-1]["time"] == n-1,
                  f"State snapshot endpoints: {label}")
    check.require(samples[-1]["summary"] == s["final_state"], f"Final-state snapshot: {label}")
    for sample in samples:
        state = sample["summary"]
        issues = []
        if any(x > budget["replay_size"] for x in state["replay_rows_per_horizon"]):
            issues.append("replay")
        if state.get("observed_input_rows", 0) > budget["replay_size"]:
            issues.append("observed_inputs")
        if state.get("total_geometry_references", 0) > budget["reference_budget"]:
            issues.append("geometry_references")
        if any(x > budget["reference_budget"] for x in state.get("kernel_supports_per_horizon", [])):
            issues.append("kernel_support")
        if issues:
            cap_violations.append({**{k:s[k] for k in ("task_id","kind","collection","method","budget")},
                                   "time": sample["time"], "violations": issues})
        check.require(state["owned_numpy_bytes"] >= 0, f"Nonnegative state bytes: {label}")
        if "reference_counts" in state:
            check.require(sum(state["reference_counts"]) == state["total_geometry_references"],
                          f"Reference-count identity: {label}/{sample['time']}")
            check.equal(state["anchor_sha256"], samples[0]["summary"]["anchor_sha256"],
                        f"Initial-anchor retention: {label}/{sample['time']}")
        if s["method"] == "fixed_kahm":
            check.equal(state["reference_sha256"], samples[0]["summary"]["reference_sha256"],
                        f"Fixed geometry stayed fixed: {label}/{sample['time']}")
    for pool in s["threadpools"]:
        check.require(pool["num_threads"] == 1, f"One numerical thread: {label}")
    check.require(s["cpu_affinity"] is None or len(s["cpu_affinity"]) == 1,
                  f"One worker CPU: {label}")
    refresh = [(t["time"], t["update"].get("local_model"),
                t["update"].get("reference_changes", 0),
                t["update"].get("geometry_refreshed", False))
               for t in tr if "local_model" in t["update"]]
    return {"key":key, "summary":s, "arrays":a, "loss":loss,
            "scored":int(valid.sum()), "fallbacks":fallback_count,
            "caps":cap_violations, "refresh":refresh,
            "p99":float(np.quantile([t["service_seconds"] for t in evaluation], .99))}


def audit(results, analysis, check):
    cfg = read_json(HERE / "config.json")
    check.equal(cfg["methods"], METHODS, "Fixed six-method roster")
    check.equal(cfg["gamma_grid"], [0, .25, .5, 1], "Declared calibration grid")
    freeze_path = HERE / "DEVELOPMENT_FREEZE.json"
    freeze = read_json(freeze_path)
    for name, record in freeze["files"].items():
        path = (HERE / name).resolve()
        check.require(path.is_relative_to(HERE.resolve()), f"Scoped freeze path: {name}")
        check.require(digest(path) == record["sha256"], f"Frozen file hash: {name}")
    check.require("audit.py" in freeze["files"], "Independent auditor included in pre-run freeze")
    manifest, tasks = prepared_inputs(check)
    expected = {(b, t, m) for b in cfg["budgets"] for t in tasks for m in METHODS}
    directories = sorted(p.parent for p in results.glob("*/summary.json"))
    check.require(len(directories) == len(expected) == 144, "Complete 144-run roster")
    runs = {}
    for directory in directories:
        header = read_json(directory / "summary.json")
        check.require(header["task_id"] in tasks, f"Unknown stream: {directory.name}")
        run = audit_run(directory, tasks[header["task_id"]], cfg, check)
        check.require(run["key"] not in runs, f"Duplicate run: {run['key']}")
        runs[run["key"]] = run
    check.require(set(runs) == expected, "Exact task/method/tier identities")
    execution = read_json(results / "execution.json")
    check.require(execution["freeze_sha256"] == digest(freeze_path), "Execution freeze identity")
    check.require({(b,t,m) for t,m,b in execution["roster"]} == expected,
                  "Execution roster identity")
    check.require(not execution["failed"], "No omitted failed worker executions")
    baseline_max_difference = 0.0
    for budget in cfg["budgets"]:
        for name in tasks:
            base = runs[(budget,name,"ridge")]
            for method in METHODS[1:]:
                other = runs[(budget,name,method)]
                aa, bb = base["arrays"], other["arrays"]
                check.equal(aa["origin"], bb["origin"], f"Shared prediction origins: {budget}/{name}/{method}")
                healthy = (aa["framework_fallback"] == 0) & (bb["framework_fallback"] == 0)
                check.close(aa["baseline"][healthy], bb["baseline"][healthy],
                            f"Shared raw baseline: {budget}/{name}/{method}", rtol=1e-8, atol=1e-8)
                if healthy.any():
                    baseline_max_difference = max(baseline_max_difference,
                        float(np.max(np.abs(aa["baseline"][healthy]-bb["baseline"][healthy]))))
            reference = runs[(budget,name,"adaptive_kahm")]
            for method in ADAPTIVE[1:]:
                other = runs[(budget,name,method)]
                if reference["summary"]["framework_errors"] or other["summary"]["framework_errors"]:
                    continue  # Failures remain visible and block technical qualification.
                check.require(reference["refresh"] == other["refresh"],
                              f"Matched adaptive refresh schedule: {budget}/{name}/{method}")
                left, right = reference["summary"]["state_samples"], other["summary"]["state_samples"]
                check.equal([r["time"] for r in left], [r["time"] for r in right],
                            f"Matched state-snapshot times: {budget}/{name}/{method}")
                for x,y in zip(left,right):
                    for field in ("reference_counts", "reference_sha256", "anchor_sha256"):
                        check.equal(x["summary"][field], y["summary"][field],
                                    f"Matched dictionary {field}: {budget}/{name}/{method}/{x['time']}")
    collections = sorted({t["collection"] for t in tasks.values()})
    aggregate = {}
    for budget in cfg["budgets"]:
        for method in METHODS:
            for collection in collections:
                chosen = [runs[(budget,t,method)]["loss"] for t in tasks
                          if tasks[t]["collection"] == collection]
                aggregate[(budget,collection,method)] = float(np.mean(chosen))
            aggregate[(budget,"natural_equal_collection",method)] = float(np.mean(
                [aggregate[(budget,c,method)] for c in NATURAL]))
    with (analysis / "collection_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    check.require(len(rows) == len(aggregate), "Complete collection-metric roster")
    for row in rows:
        key = row["budget"], row["collection"], row["method"]
        check.close(float(row["mean_scaled_mse"]), aggregate[key], f"Independent collection loss: {key}")
    def advantage(budget, collection, comparator):
        numerator = aggregate[(budget,collection,"adaptive_kahm")]
        denominator = aggregate[(budget,collection,comparator)]
        check.require(denominator > 0, f"Positive comparator loss: {budget}/{collection}/{comparator}")
        return 1.0 - numerator / denominator
    cap_violations = [c for run in runs.values() for c in run["caps"]]
    errors = sum(len(run["summary"]["framework_errors"]) for run in runs.values())
    technical = not cap_violations and errors == 0
    g = cfg["advancement_gate"]
    decisions = []
    detail = []
    for budget in cfg["budgets"]:
        mechanism = True
        for comparator in g["comparators"]:
            seeds = [t for t in tasks if tasks[t]["collection"] == "synthetic_recurring"]
            wins = sum(runs[(budget,t,"adaptive_kahm")]["loss"] < runs[(budget,t,comparator)]["loss"] for t in seeds)
            mechanism &= (advantage(budget,"synthetic_recurring",comparator) >= g["recurring_min_gain"]
                          and wins >= g["recurring_min_seed_wins"])
        mechanism &= all(advantage(budget,"synthetic_recurring",m) >=
                         g["recurring_min_gain_vs_fixed_and_ridge"] for m in ("fixed_kahm","ridge"))
        nat_aggregate = all(advantage(budget,"natural_equal_collection",m) >=
                            g["natural_aggregate_min_gain"] for m in g["comparators"])
        one_source = any(all(advantage(budget,c,m) >= g["natural_one_source_min_gain"]
                             for m in g["comparators"]) for c in NATURAL)
        no_harm = all(advantage(budget,c,m) >= -g["natural_max_other_source_harm"]
                      for m in g["comparators"] for c in NATURAL)
        natural = nat_aggregate and one_source and no_harm
        kahm = [run for (b,t,m),run in runs.items() if b == budget and m == "adaptive_kahm"]
        resource = all(r["p99"] <= g["maximum_p99_service_seconds"] for r in kahm)
        own_failures = any(r["summary"]["framework_errors"] for r in kahm)
        proceed = mechanism and natural and resource and not own_failures and technical
        decision = ("TECHNICAL_BLOCK" if not technical else
                    "PROCEED" if proceed else "REFINE" if mechanism else "DEFER")
        decisions.append(decision)
        detail.append({"budget":budget,"decision":decision,"mechanism_gate":bool(mechanism),
                       "natural_gate":bool(natural),"natural_aggregate_gate":bool(nat_aggregate),
                       "same_source_beats_all_comparators":bool(one_source),
                       "no_source_harm_above_five_percent":bool(no_harm),
                       "p99_allowance_gate":bool(resource),"zero_framework_failures":not own_failures})
    decision = ("TECHNICAL_BLOCK" if not technical else
                "PROCEED" if "PROCEED" in decisions else
                "REFINE" if "REFINE" in decisions else "DEFER")
    published = read_json(analysis / "decision.json")
    check.require(published["decision"] == decision, "Independent overall development decision")
    check.require(published["technical_validity"] == technical, "Independent global technical block")
    check.require(published["cap_violations"] == cap_violations, "Recorded state cap violations")
    for independent, reported in zip(detail,published["capacity_tiers"]):
        for key, expected_value in independent.items():
            check.require(reported[key] == expected_value,
                          f"Independent {independent['budget']} gate component: {key}")
    scored = sum(r["scored"] for r in runs.values())
    fallbacks = sum(r["fallbacks"] for r in runs.values())
    check.require(published["scored_forecasts"] == scored, "Total scored forecasts")
    check.require(published["framework_errors"] == errors, "Total recorded framework errors")
    check.require(published["framework_fallback_forecasts"] == fallbacks, "Total framework fallback forecasts")
    check.require(published["config_sha256"] == digest(HERE / "config.json"), "Analysis configuration identity")
    return {"status":"passed", "checks":check.count,"run_count":len(runs),
            "scored_forecasts":scored,"decision":decision,"technical_validity":technical,
            "cap_violations":cap_violations,"framework_errors":errors,
            "framework_fallback_forecasts":fallbacks,
            "maximum_shared_baseline_absolute_difference":baseline_max_difference,
            "baseline_relative_tolerance":1e-8,"baseline_absolute_tolerance":1e-8,
            "freeze_sha256":digest(freeze_path),"config_sha256":digest(HERE / "config.json"),
            "analysis_decision_sha256":digest(analysis / "decision.json"),
            "scope":"Independent saved-evidence reconstruction of targets, chronology, calibration, losses, timing summaries, capacity snapshots, shared baseline/reference parity, aggregation and development gates. No model fitting or independent experimental replication."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--analysis", type=Path, default=HERE / "analysis")
    p.add_argument("--output", type=Path, default=HERE / "analysis/audit.json")
    args = p.parse_args()
    check = Checks()
    try:
        result = audit(args.results, args.analysis, check)
    except Exception as error:
        result = {"status":"failed","checks":check.count,
                  "error":f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
