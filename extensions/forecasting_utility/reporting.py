"""Report the attempted frozen study without converting fit failures to outcomes.

This reporting-only extension was added after execution revealed two archive
tasks on which the inherited geometry could not fit its requested clusters.
It never fits a model, changes a forecast, replaces a task, or edits frozen
scientific sources. See REPORTING_AMENDMENT.md for the outcome-aware scope.

The original audit remains authoritative and retains ``complete=False``. The
complete synthetic factorial is summarized as specified. Archive comparisons
on common support are explicitly post hoc diagnostics; full-roster means are
defined only for procedures that produced forecasts for every archive task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import analysis, data
from .experiment import FREEZE, POLICIES, task_roster, verify_freeze


ROOT = Path(__file__).resolve().parent
METRICS = ["test_mse", "scaled_mse", "pipeline_seconds", "model_pipeline_seconds",
           "model_cpu_seconds", "serialized_model_bytes"]
AUDIT_FILES = ("task_methods.csv", "validation_trials.csv", "episode_losses.csv",
               "horizon_losses.csv", "correction_decomposition.csv", "timing_nodes.csv")
COMMON_SCOPE = "Post hoc common-support diagnostic; excludes tasks lacking any declared method or policy"
FULL_SCOPE = "Original full archive roster; mean defined only when all declared tasks have forecasts"


def _records(run_dir, execution):
    return [json.loads((run_dir / "tasks" / task["task_id"] / "result.json").read_text())
            for task in execution["expected_tasks"]]


def _status_frame(records, cfg):
    rows = []
    for record in records:
        task = record["task"]
        if [row["method"] for row in record["methods"]] != cfg["methods"]:
            raise ValueError("An attempted task does not retain the complete method roster")
        if set(record["policies"]) != set(POLICIES):
            raise ValueError("An attempted task does not retain the complete policy roster")
        base = {"task_id": task["task_id"], "phase": task["phase"],
                "source_id": task.get("source_id", ""), "series_id": task.get("series_id", ""),
                "law": task.get("law", ""), "seed": task["seed"],
                "training_episodes": task.get("training_episodes", 0), "landmarks": task["landmarks"]}
        for item in record["methods"]:
            rows.append({**base, "kind": "method", "method": item["method"],
                         "status": item["status"], "error": item.get("error", "")})
        for name, item in record["policies"].items():
            rows.append({**base, "kind": "policy", "method": name,
                         "status": item["status"], "error": item.get("error", "")})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["task_id", "method"]).any():
        raise ValueError("Duplicate task/procedure status")
    if not frame.status.isin(["ok", "failed"]).all():
        raise ValueError("Attempted tasks contain unfinished procedure statuses")
    frame["succeeded"] = frame.status == "ok"
    return frame


def _coverage(frame, keys):
    coverage = frame.groupby(keys, sort=False).agg(
        attempted_tasks=("task_id", "size"), successful_tasks=("succeeded", "sum")).reset_index()
    coverage["failed_tasks"] = coverage.attempted_tasks - coverage.successful_tasks
    coverage["coverage_fraction"] = coverage.successful_tasks / coverage.attempted_tasks
    return coverage


def _source_means(frame):
    series = frame.groupby(["source_id", "series_id", "method"], sort=False)[METRICS].mean().reset_index()
    sources = series.groupby(["source_id", "method"], sort=False)[METRICS].mean().reset_index()
    overall = sources.groupby("method", sort=False)[METRICS].mean().reset_index()
    return series, sources, overall


def _check_synthetic_pairing(records, run_dir, cfg):
    """Audit saved stream hashes and nested targets without generating any data."""
    seen = {}
    nested = {}
    for record in records:
        task = record["task"]
        if task["phase"] != "synthetic":
            continue
        metadata = record["data_metadata"]
        hashes = metadata["arrays_sha256"]
        for split_index, split in enumerate(("train", "validation", "test")):
            expected_seed = int(np.random.SeedSequence([
                task["seed"], cfg["synthetic"]["laws"].index(task["law"]), split_index, 731,
            ]).generate_state(1)[0])
            if metadata[split]["seed"] != expected_seed:
                raise ValueError("Synthetic split seed differs from the frozen derivation")
            for component in ("X", "Y", "groups", "origins"):
                key = (task["law"], task["seed"], split, component,
                       task["training_episodes"] if split == "train" else None)
                value = hashes[f"{component}_{split}"]
                if key in seen and seen[key] != value:
                    raise ValueError("Paired synthetic tasks do not share their required data stream")
                seen[key] = value
        for component in ("X", "Y", "groups", "origins"):
            if hashes[f"{component}_refit"] != hashes[f"{component}_train"]:
                raise ValueError("Synthetic no-refit arrays differ from their training arrays")
        key = (task["law"], task["seed"], task["training_episodes"])
        if key not in nested:
            with np.load(run_dir / "tasks" / task["task_id"] / "targets.npz", allow_pickle=False) as saved:
                nested[key] = {name: saved[f"{name}_train"] for name in ("Y", "groups", "origins")}
    for law in cfg["synthetic"]["laws"]:
        for seed in cfg["synthetic"]["repetition_seeds"]:
            largest = nested[(law, seed, max(cfg["synthetic"]["training_episodes"]))]
            for count in cfg["synthetic"]["training_episodes"]:
                smaller = nested[(law, seed, count)]
                keep = largest["groups"] < count
                for name in smaller:
                    if not np.array_equal(smaller[name], largest[name][keep]):
                        raise ValueError("Saved synthetic targets/groups/origins violate nested training samples")


def summarize(run_dir, output_dir=None):
    """Audit and write failure-aware tables; return their truthful summary.

    All 168 task attempts are required. Existing files produced by the frozen
    audit must remain byte-identical. New summaries go only into ``report/``.
    This function may write reports; ``verify_report`` is the read-only gate.
    """
    run_dir = Path(run_dir).resolve()
    output = (run_dir / "report" if output_dir is None else Path(output_dir).resolve())
    original = run_dir / "analysis"
    if output == original or original in output.parents:
        raise ValueError("Failure-aware reports must be separate from the original analysis directory")
    before = {path.name: data.digest(path) for path in original.iterdir() if path.is_file()} if original.exists() else {}
    audited = analysis.summarize(run_dir)
    if any(data.digest(original / name) != digest for name, digest in before.items()):
        raise ValueError("The frozen audit changed an existing original-analysis artifact")
    if audited["missing_tasks"] or audited["observed_tasks"] != audited["expected_tasks"]:
        raise ValueError("All prespecified task attempts are required before this report")
    if audited["expected_tasks"] != 168:
        raise ValueError("This reporting amendment is for the declared 168-task study")
    cfg = data.config()
    execution = json.loads((run_dir / "execution.json").read_text())
    records = _records(run_dir, execution)
    statuses = _status_frame(records, cfg)
    _check_synthetic_pairing(records, run_dir, cfg)
    methods = cfg["methods"] + list(POLICIES)
    frame = pd.read_csv(original / "task_methods.csv")
    if frame.duplicated(["task_id", "method"]).any():
        raise ValueError("Duplicate successful task/procedure rows")
    successful_pairs = set(map(tuple, statuses.loc[statuses.succeeded, ["task_id", "method"]].to_numpy()))
    if set(map(tuple, frame[["task_id", "method"]].to_numpy())) != successful_pairs:
        raise ValueError("Successful forecast table differs from the complete attempt accounting")
    if not np.isfinite(frame[METRICS].to_numpy(float)).all():
        raise ValueError("Successful outcomes contain nonfinite loss/cost values")

    synthetic = frame[frame.phase == "synthetic"].copy()
    synthetic_tasks = [task for task in execution["expected_tasks"] if task["phase"] == "synthetic"]
    if len(synthetic_tasks) != 144 or len(synthetic) != 144 * len(methods):
        raise ValueError("The complete prespecified synthetic factorial is required")
    if not statuses.loc[statuses.phase == "synthetic", "succeeded"].all():
        raise ValueError("Synthetic failure cannot be silently converted to a complete factorial")

    archive_status = statuses[statuses.phase == "natural"].copy()
    common_ids = archive_status.groupby("task_id", sort=False).succeeded.all()
    common_ids = set(common_ids.index[common_ids])
    archive = frame[frame.phase == "natural"].copy()
    common = archive[archive.task_id.isin(common_ids)].copy()
    if any(set(group.method) != set(methods) for _, group in common.groupby("task_id")):
        raise ValueError("Common-support tasks do not contain every declared procedure")
    task_rows = pd.DataFrame([{
        "task_id": task["task_id"], "phase": task["phase"], "source_id": task.get("source_id", ""),
        "series_id": task.get("series_id", ""), "landmarks": task["landmarks"],
        "all_procedures_succeeded": bool(next(record["complete"] for record in records if record["task"] == task)),
        "archive_common_support": task["task_id"] in common_ids,
    } for task in execution["expected_tasks"]])
    archive_tasks = task_rows[task_rows.phase == "natural"]
    common_tasks = archive_tasks[archive_tasks.archive_common_support]
    support = archive_tasks.groupby("source_id", sort=False).agg(
        expected_tasks=("task_id", "size"), expected_series=("series_id", "nunique")).reset_index()
    retained = common_tasks.groupby("source_id", sort=False).agg(
        common_support_tasks=("task_id", "size"), common_support_series=("series_id", "nunique")).reset_index()
    support = support.merge(retained, on="source_id", how="left").fillna(0)
    support["excluded_tasks"] = support.expected_tasks - support.common_support_tasks
    support["excluded_series"] = support.expected_series - support.common_support_series
    if len(common_ids) != 22 or len(common_tasks[["source_id", "series_id"]].drop_duplicates()) != 11:
        raise ValueError("Common support differs from the explicitly recorded 22-task/11-series amendment")
    if sorted(support.common_support_series.tolist()) != [2, 3, 3, 3]:
        raise ValueError("Archive source coverage differs from the reporting amendment")

    output.mkdir(parents=True, exist_ok=True)
    for name in AUDIT_FILES:
        shutil.copyfile(original / name, output / name)
    shutil.copyfile(original / "summary.json", output / "original_audit_summary.json")
    statuses.to_csv(output / "procedure_status.csv", index=False)
    statuses[~statuses.succeeded].to_csv(output / "failures.csv", index=False)
    task_rows.to_csv(output / "task_coverage.csv", index=False)
    coverage = _coverage(statuses, ["phase", "kind", "method"])
    coverage.to_csv(output / "coverage_by_method.csv", index=False)
    archive_coverage = _coverage(archive_status, ["kind", "method"])
    archive_coverage.to_csv(output / "archive_coverage_by_method.csv", index=False)
    source_coverage = _coverage(archive_status, ["source_id", "kind", "method"])
    source_coverage.to_csv(output / "archive_coverage_by_source_method.csv", index=False)
    support.to_csv(output / "archive_support.csv", index=False)
    common_tasks.to_csv(output / "archive_common_support_tasks.csv", index=False)
    archive_tasks[~archive_tasks.archive_common_support].to_csv(output / "archive_excluded_tasks.csv", index=False)
    common.to_csv(output / "archive_common_support_task_methods.csv", index=False)

    cell_keys = ["law", "training_episodes", "landmarks", "method"]
    syn_summary = synthetic.groupby(cell_keys, sort=False)[METRICS].mean().reset_index()
    syn_summary.to_csv(output / "synthetic_methods.csv", index=False)
    contrasts = []
    for (law, count, landmarks), group in synthetic.groupby(cell_keys[:-1], sort=False):
        table = group.pivot(index="seed", columns="method", values="test_mse")
        if set(table.index) != set(cfg["synthetic"]["repetition_seeds"]):
            raise ValueError("Primary synthetic contrast lacks a declared complete repetition")
        for control in ("raw_rbf", "concat_kahm"):
            contrasts.append({"law": law, "training_episodes": int(count), "landmarks": int(landmarks),
                              "control": control, "additive_mean_mse": float(table.add_kahm.mean()),
                              "control_mean_mse": float(table[control].mean()),
                              **analysis.paired_interval(table[control] - table.add_kahm, family_size=24)})
    if len(contrasts) != 24:
        raise ValueError("Primary synthetic contrast family is not the declared 24 comparisons")
    analysis.write_csv(output / "synthetic_primary_contrasts.csv", contrasts)

    series, sources, overall = _source_means(common)
    for name, table in (("series", series), ("sources", sources), ("methods", overall)):
        table["analysis_scope"] = COMMON_SCOPE
        table.to_csv(output / f"archive_{name}.csv", index=False)
        table.to_csv(output / f"archive_common_support_{name}.csv", index=False)
    complete_archive_methods = set(archive_coverage.loc[archive_coverage.failed_tasks == 0, "method"])
    full_frame = archive[archive.method.isin(complete_archive_methods)]
    full_series, full_sources, full_means = _source_means(full_frame)
    for name, table in (("series", full_series), ("sources", full_sources), ("baselines", full_means)):
        table["analysis_scope"] = FULL_SCOPE
        table.to_csv(output / f"archive_full_roster_{name}.csv", index=False)
    full_rows = []
    for method in methods:
        complete = method in complete_archive_methods
        row = {"method": method, "mean_defined": complete, "analysis_scope": FULL_SCOPE,
               "undefined_reason": "" if complete else "Missing forecasts from declared fit/policy failures; no imputation or replacement"}
        row.update({metric: float(full_means.loc[full_means.method == method, metric].iloc[0]) if complete else None
                    for metric in METRICS})
        full_rows.append(row)
    analysis.write_csv(output / "archive_full_roster_methods.csv", full_rows)

    source_pivot = sources.pivot(index="source_id", columns="method", values="scaled_mse")
    leave_one_out = []
    for omitted in source_pivot.index:
        rest = source_pivot.drop(index=omitted).mean()
        leave_one_out.append({"omitted_source": omitted, "raw_rbf": rest.raw_rbf,
                              "add_kahm": rest.add_kahm,
                              "relative_change_pct": 100 * (rest.add_kahm / rest.raw_rbf - 1),
                              "analysis_scope": COMMON_SCOPE})
    analysis.write_csv(output / "archive_leave_one_source_out.csv", leave_one_out)
    reported = pd.concat([synthetic, common], ignore_index=True)
    gamma = reported[reported.method.str.startswith("add_")].groupby(
        ["phase", "method", "gamma"], dropna=False).size().reset_index(name="tasks")
    gamma.to_csv(output / "gamma_choices.csv", index=False)
    policy = reported[reported.method.isin(POLICIES)].groupby(
        ["phase", "method", "selected_method"]).size().reset_index(name="tasks")
    policy.to_csv(output / "policy_choices.csv", index=False)
    warnings = [{"task_id": record["task"]["task_id"], **warning}
                for record in records for warning in record.get("warnings", [])]
    analysis.write_csv(output / "warnings.csv", warnings)

    standalone = statuses[statuses.kind == "method"]
    policies = statuses[statuses.kind == "policy"]
    summary = {
        **audited, "schema_version": 1, "report_type": "failure_aware_post_execution_reporting",
        "verification_passed": True, "all_tasks_attempted": True,
        "all_methods_succeeded": bool(standalone.succeeded.all()),
        "all_policies_succeeded": bool(policies.succeeded.all()),
        "complete": bool(statuses.succeeded.all()),
        "attempted_tasks": len(records), "completed_tasks": sum(bool(record["complete"]) for record in records),
        "attempted_methods": len(standalone), "successful_methods": int(standalone.succeeded.sum()),
        "failed_method_count": int((~standalone.succeeded).sum()),
        "successful_standalone_forecasts": int(standalone.succeeded.sum()),
        "attempted_policies": len(policies), "successful_policies": int(policies.succeeded.sum()),
        "failed_policy_count": int((~policies.succeeded).sum()),
        "synthetic_tasks": len(synthetic_tasks), "archive_tasks": len(archive_tasks),
        "archive_common_support_tasks": len(common_ids),
        "archive_common_support_series": len(common_tasks[["source_id", "series_id"]].drop_duplicates()),
        "archive_common_support_sources": len(support),
        "archive_common_support_task_ids": sorted(common_ids),
        "archive_excluded_task_ids": sorted(set(archive_tasks.task_id) - common_ids),
        "common_support_by_source": support.to_dict(orient="records"),
        "archive_full_roster_kahm_mean": None,
        "archive_full_roster_means": full_rows,
        "archive_full_roster_complete_methods": sorted(complete_archive_methods),
        "synthetic_methods": syn_summary.to_dict(orient="records"),
        "synthetic_primary_contrasts": contrasts,
        "archive_methods": overall.to_dict(orient="records"),
        "archive_sources": sources.to_dict(orient="records"),
        "archive_leave_one_source_out": leave_one_out,
        "gamma_choices": gamma.to_dict(orient="records"),
        "policy_choices": policy.to_dict(orient="records"),
        "warnings_count": len(warnings),
        "procedure_failures": statuses.loc[~statuses.succeeded].to_dict(orient="records"),
        "synthetic_inference": "Unchanged prespecified 24 paired-Student contrasts, 12 complete repetitions per cell, Bonferroni simultaneous95% under the stated normal approximation; no independent-window inference.",
        "archive_inference": COMMON_SCOPE + "; four source collections, 11 retained series, 22 tasks. Full24-task KAHM and geometry-policy means are undefined. No generalization or equivalence inference follows from fit-success conditioning.",
        "failure_scope": "Eight explicit inherited-geometry fit failures and four dependent policy failures on two capacities of one selected Car Parts series. No fit was retried, repaired, replaced, or assigned a forecast by this reporting module.",
        "shared_data_audit": "Saved synthetic input/target/group/origin hashes agree across paired capacities and evaluation data sizes; saved training targets/groups/origins are nested. No study data were regenerated.",
        "original_audit_preserved": True,
    }
    data.write_json(output / "summary.json", summary)
    report_paths = [path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"]
    source_paths = [run_dir / "execution.json", *[original / name for name in (*AUDIT_FILES, "summary.json")]]
    if (run_dir / "environment.json").is_file():
        source_paths.append(run_dir / "environment.json")
    for record in records:
        directory = run_dir / "tasks" / record["task"]["task_id"]
        source_paths.extend([directory / "result.json", directory / "targets.npz"])
        source_paths.extend(directory / method["prediction_file"] for method in record["methods"] if method["status"] == "ok")
    manifest = {
        "schema_version": 1, "execution_freeze_sha256": data.digest(FREEZE),
        "reporting_source_sha256": data.digest(__file__),
        "run_files": {str(path.relative_to(run_dir)): data.digest(path) for path in source_paths},
        "report_files": {str(path.relative_to(output)): data.digest(path) for path in report_paths},
    }
    data.write_json(output / "manifest.json", manifest)
    return summary


def verify_report(run_dir, report_dir=None):
    """Read-only integrity/coverage gate; success does not mean every fit succeeded."""
    run_dir = Path(run_dir).resolve()
    output = run_dir / "report" if report_dir is None else Path(report_dir).resolve()
    frozen = verify_freeze()
    execution = json.loads((run_dir / "execution.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if (execution["freeze"] != frozen or execution["freeze_sha256"] != data.digest(FREEZE)
            or manifest["execution_freeze_sha256"] != data.digest(FREEZE)):
        raise ValueError("Report belongs to another execution freeze")
    if manifest["reporting_source_sha256"] != data.digest(__file__):
        raise ValueError("Failure-aware reporting source differs from the audited source")
    for base, files in ((run_dir, manifest["run_files"]), (output, manifest["report_files"])):
        for name, expected in files.items():
            path = (base / name).resolve()
            if base not in path.parents or data.digest(path) != expected:
                raise ValueError(f"Reporting evidence checksum mismatch: {name}")
    cfg = data.config()
    if execution["expected_tasks"] != task_roster(cfg):
        raise ValueError("Execution task roster differs from the frozen study")
    records = _records(run_dir, execution)
    statuses = _status_frame(records, cfg)
    summary = json.loads((output / "summary.json").read_text())
    if (not summary["verification_passed"] or not summary["all_tasks_attempted"]
            or summary["observed_tasks"] != len(records) or len(records) != 168):
        raise ValueError("Report does not verify all declared task attempts")
    for kind, attempted, successful, failed in (("method", "attempted_methods", "successful_methods", "failed_method_count"),
                                               ("policy", "attempted_policies", "successful_policies", "failed_policy_count")):
        subset = statuses[statuses.kind == kind]
        if (summary[attempted] != len(subset) or summary[successful] != int(subset.succeeded.sum())
                or summary[failed] != int((~subset.succeeded).sum())):
            raise ValueError("Report coverage differs from the preserved attempt records")
    if summary["complete"] != bool(statuses.succeeded.all()):
        raise ValueError("Report falsely marks failed study procedures complete")
    original = json.loads((run_dir / "analysis/summary.json").read_text())
    if original["complete"] != summary["complete"]:
        raise ValueError("Original audit failure status was not preserved")
    return summary


build_report = summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "utility_v1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="Check existing reports and evidence without writing files")
    args = parser.parse_args()
    result = (verify_report(args.run_dir, args.output_dir) if args.verify_only
              else summarize(args.run_dir, args.output_dir))
    print(json.dumps({key: result[key] for key in (
        "verification_passed", "all_tasks_attempted", "all_methods_succeeded", "complete",
        "observed_tasks", "successful_methods", "failed_method_count", "failed_policy_count",
        "archive_common_support_tasks", "archive_full_roster_kahm_mean",
    )}, indent=2))
    raise SystemExit(0 if result["verification_passed"] and result["all_tasks_attempted"] else 1)


if __name__ == "__main__":
    main()
