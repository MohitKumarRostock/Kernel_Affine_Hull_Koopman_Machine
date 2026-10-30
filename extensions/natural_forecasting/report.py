"""Build manuscript tables and figures from a completed, frozen experiment.

This postprocessor never fits a forecaster, chooses a configuration, predicts a
diagnostic benefit, or recalibrates a diagnostic. It reads the saved analysis
decisions and metrics. Confirmation files are read only after the confirmation
freeze and a successful final VALIDATION.json exist. Plotting should be run
after the measured benchmark workers have finished.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
VERSION = "natural_forecasting_report_v1"
PHASES = ("development", "confirmation")
DIAGNOSTICS = ("metadata", "metadata_geometry", "metadata_probe", "metadata_probe_geometry")
GEOMETRY_FEATURES = {"kahm_gate_entropy", "log_affine_projection_residual"}
METHOD_LABELS = {
    "raw_ridge": "Raw ridge", "raw_rbf": "Raw RBF",
    "affine_rbf": "Affine local", "locality_rbf": "Locality local",
    "kahm_rbf": "KAHM local", "raw_affine_rbf": "Raw + affine",
    "raw_locality_rbf": "Raw + locality", "raw_kahm_rbf": "Raw + KAHM",
    "seasonal_naive": "Seasonal naive", "holt": "Holt",
    "hist_gradient": "Histogram boosting", "mlp": "MLP",
}
POLICY_ORDER = (
    "always_raw", "always_kahm", "validation_pair", "validation_raw_portfolio",
    "validation_kahm_family", "validation_full_portfolio", "cheap_probe",
    *DIAGNOSTICS,
)
POLICY_LABELS = {
    "always_raw": "Always raw", "always_kahm": "Always raw + KAHM",
    "validation_pair": "Validation: matched pair",
    "validation_raw_portfolio": "Validation: raw portfolio",
    "validation_kahm_family": "Validation: KAHM family",
    "validation_full_portfolio": "Validation: full portfolio",
    "cheap_probe": "Cheap probe", "metadata": "Metadata",
    "metadata_geometry": "Metadata + geometry", "metadata_probe": "Metadata + probe",
    "metadata_probe_geometry": "Metadata + probe + geometry",
}
FAMILY_LABELS = {
    "uci_electricity_load": "Electricity", "nn5_atm_cash": "NN5",
    "tourism_competition": "Tourism", "fred_md_macroeconomics": "FRED-MD",
    "caltrans_bay_area_traffic": "Traffic", "expsmooth_hospital": "Hospital",
    "saugeen_river_flow": "Saugeen", "us_births_mosaicdata": "US births",
}
METRIC_FIELDS = (
    "scaled_test_mse", "total_policy_wall_seconds", "total_policy_cpu_seconds",
    "benefit_vs_always_raw", "harm_vs_always_raw", "practical_harm_vs_always_raw",
    "practical_benefit_vs_always_raw", "oracle_excess_scaled_loss",
    "selected_kahm_pair_member", "decision_wall_seconds", "decision_cpu_seconds",
    "maximum_measured_node_peak_rss_bytes", "serialized_model_bytes",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def digest_value(value):
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _finite(value, *, nonnegative=False):
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError("A saved report quantity is nonfinite or invalid")
    return result


def _weights(rows, keys):
    """Use the already frozen hierarchy for extra descriptive metadata only."""
    result = np.zeros(len(rows), dtype=float)

    def assign(indices, depth, mass):
        if depth == len(keys):
            result[indices] = mass / len(indices)
            return
        groups = defaultdict(list)
        for index in indices:
            groups[rows[index][keys[depth]]].append(index)
        for indices_in_group in groups.values():
            assign(indices_in_group, depth + 1, mass / len(groups))

    if not rows:
        raise ValueError("Cannot summarize an empty set of saved rows")
    assign(list(range(len(rows))), 0, 1.0)
    return result


def _range(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return {"minimum": min(values), "maximum": max(values)}


def _gated_inputs(run_root):
    """Check the stage gate before opening any confirmation outcome file."""
    run_root = Path(run_root)
    for filename in ("initial_freeze.json", "confirmation_freeze.json", "frozen_diagnostic.json", "VALIDATION.json"):
        if not (run_root / filename).is_file():
            raise RuntimeError("Reporting requires the completed experiment and " + filename)
    validation = read_json(run_root / "VALIDATION.json")
    if validation.get("status") != "passed" or not validation.get("frozen_protocol_intact"):
        raise RuntimeError("The saved final validation has not passed")
    initial = read_json(run_root / "initial_freeze.json")
    confirmation_freeze = read_json(run_root / "confirmation_freeze.json")
    frozen = read_json(run_root / "frozen_diagnostic.json")
    if digest_file(run_root / "frozen_diagnostic.json") != confirmation_freeze["diagnostic_sha256"]:
        raise ValueError("Frozen diagnostic file differs from the confirmation freeze")
    copy = dict(frozen)
    if digest_value({key: value for key, value in copy.items() if key != "freeze_sha256"}) != frozen["freeze_sha256"]:
        raise ValueError("Frozen diagnostic contents changed")
    if digest_value(frozen["spec"]) != frozen["spec_sha256"]:
        raise ValueError("Frozen analysis specification changed")
    for freeze in (initial, confirmation_freeze):
        if digest_file(run_root / "task_roster.json") != freeze["roster_sha256"]:
            raise ValueError("Task roster differs from the saved freeze")
        for name, expected in freeze["code_sha256"].items():
            if digest_file(ROOT / name) != expected:
                raise ValueError("Frozen scientific file changed: " + name)
    if initial["code_sha256"] != confirmation_freeze["code_sha256"]:
        raise ValueError("Scientific code differs between the two stage freezes")
    roster = read_json(run_root / "task_roster.json")
    expected = {phase: {row["task_id"] for row in roster["tasks"] if row["role"] == phase} for phase in PHASES}
    reports, records, input_hashes = {}, {}, {}
    for filename in ("initial_freeze.json", "confirmation_freeze.json", "frozen_diagnostic.json", "VALIDATION.json", "task_roster.json"):
        input_hashes[filename] = digest_file(run_root / filename)
    freeze_time = datetime.fromisoformat(confirmation_freeze["created_utc"])
    for phase in PHASES:
        report_path = run_root / (phase + "_analysis.json")
        reports[phase] = read_json(report_path)
        input_hashes[report_path.name] = digest_file(report_path)
        if reports[phase]["spec_sha256"] != frozen["spec_sha256"]:
            raise ValueError("Saved analysis uses a different specification")
        if phase == "confirmation" and reports[phase]["freeze_sha256"] != frozen["freeze_sha256"]:
            raise ValueError("Saved confirmation analysis uses a different diagnostic freeze")
        records[phase] = []
        for path in sorted((run_root / phase).glob("*.json")):
            record = read_json(path)
            if record.get("status") != "complete" or record.get("phase") != phase:
                raise ValueError("Incomplete or incorrectly staged task result: " + path.name)
            if phase == "confirmation" and datetime.fromisoformat(record["created_utc"]) < freeze_time:
                raise ValueError("A confirmation task predates the confirmation freeze")
            if phase == "development":
                expected_hash = confirmation_freeze["development_result_sha256"].get(path.name)
                if expected_hash is None or digest_file(path) != expected_hash:
                    raise ValueError("Development evidence changed after diagnostic freezing")
            records[phase].append(record)
            input_hashes[phase + "/" + path.name] = digest_file(path)
        observed = [row["task_id"] for row in records[phase]]
        if len(observed) != len(set(observed)) or set(observed) != expected[phase]:
            raise ValueError("Saved result task roster is incomplete, duplicated or changed")
        policy_ids = {row["task_id"] for row in reports[phase]["task_policy_rows"]}
        if policy_ids != expected[phase]:
            raise ValueError("Saved analysis and task-result rosters differ")
    return {"reports": reports, "records": records, "frozen": frozen, "initial": initial,
            "confirmation_freeze": confirmation_freeze, "roster": roster,
            "validation": validation, "input_sha256": input_hashes}


def _node_cost(record, ids):
    nodes = {row["id"]: row for row in record["cost_nodes"]}
    ids = set(ids)
    if len(nodes) != len(record["cost_nodes"]) or not ids.issubset(nodes):
        raise ValueError("Invalid saved cost graph")
    selected = [nodes[key] for key in sorted(ids)]
    return {
        "wall_seconds": sum(_finite(row["seconds"], nonnegative=True) for row in selected),
        "cpu_seconds": sum(_finite(row["process_cpu_seconds"], nonnegative=True) for row in selected)
            if all(row.get("process_cpu_seconds") is not None for row in selected) else None,
        "peak_rss_bytes": max((_finite(row["peak_rss_bytes"], nonnegative=True) for row in selected), default=0.0)
            if all(row.get("peak_rss_bytes") is not None for row in selected) else None,
        "required_cost_node_ids": sorted(ids),
    }


def _sum_costs(costs):
    return {
        "wall_seconds": sum(row["wall_seconds"] for row in costs),
        "cpu_seconds": sum(row["cpu_seconds"] for row in costs)
            if all(row["cpu_seconds"] is not None for row in costs) else None,
        "maximum_process_high_water_rss_bytes": max((row["peak_rss_bytes"] for row in costs), default=0)
            if all(row["peak_rss_bytes"] is not None for row in costs) else None,
    }


def offline_costs(records, frozen):
    """Sum explicit acquisition dependencies, without reusing future labels."""
    config = frozen["spec"]["config"]
    pair = (config["raw_candidate"], config["kahm_candidate"])
    by_variant = {name: [] for name in DIAGNOSTICS}
    label_costs, joint_costs = [], []
    for record in records:
        candidates = {row["candidate_id"]: row for row in record["candidates"]}
        label_ids = set()
        for name in pair:
            label_ids.update(candidates[name]["selection_cost_node_ids"])
            label_ids.update(candidates[name]["final_cost_node_ids"])
        labels = _node_cost(record, label_ids)
        label_costs.append(labels)
        joint_ids = set(label_ids)
        for name in DIAGNOSTICS:
            features = set(frozen["models"][name]["features"])
            ids = label_ids | set(record["feature_cost_node_ids"]["metadata"])
            if features & GEOMETRY_FEATURES:
                ids |= set(record["feature_cost_node_ids"]["geometry"])
            if "probe_log_ratio" in features:
                ids |= set(record["probe"]["cost_node_ids"])
            joint_ids |= ids
            by_variant[name].append({"task_id": record["task_id"], "source_family": record["source_family"],
                                     **_node_cost(record, ids),
                                     "additional_feature_wall_seconds": _node_cost(record, ids - label_ids)["wall_seconds"]})
        joint_costs.append(_node_cost(record, joint_ids))
    variants = {}
    for name, rows in by_variant.items():
        acquisition = _sum_costs(rows)
        fit = frozen["offline_fit_by_model"][name]
        variants[name] = {
            "matched_label_and_feature_acquisition_task_sum": acquisition,
            "additional_feature_wall_seconds_beyond_matched_labels": sum(row["additional_feature_wall_seconds"] for row in rows),
            "ridge_fit": dict(fit),
            "acquisition_plus_ridge_fit_wall_seconds": acquisition["wall_seconds"] + fit["wall_seconds"],
            "acquisition_plus_ridge_fit_cpu_seconds": acquisition["cpu_seconds"] + fit["cpu_seconds"]
                if acquisition["cpu_seconds"] is not None else None,
            "per_development_task": rows,
        }
    joint = _sum_costs(joint_costs)
    return {
        "scope": "Offline evidence acquisition plus diagnostic fitting. Within each development task, union the complete matched raw/KAHM searches, final refits/forecasts/model serialization and the variant's metadata/probe/geometry dependencies. Sum standalone task costs across development tasks; no cross-task caching credit is assumed. This can conservatively repeat shared preparation across seeds/tasks. Additional study orchestration, evaluation and artifact-storage overhead remains in the actual campaign task intervals reported separately. No amortization over an invented future deployment count is used.",
        "development_tasks": len(records),
        "development_source_families": len({row["source_family"] for row in records}),
        "matched_pair": list(pair),
        "shared_matched_label_acquisition_task_sum": _sum_costs(label_costs),
        "by_variant": variants,
        "joint_all_variants_acquisition_task_sum": joint,
        "joint_all_variants_ridge_fit_wall_seconds": frozen["offline_fit_wall_seconds"],
        "joint_all_variants_ridge_fit_cpu_seconds": frozen["offline_fit_cpu_seconds"],
        "joint_all_variants_acquisition_plus_fit_wall_seconds": joint["wall_seconds"] + frozen["offline_fit_wall_seconds"],
        "joint_all_variants_acquisition_plus_fit_cpu_seconds": joint["cpu_seconds"] + frozen["offline_fit_cpu_seconds"]
            if joint["cpu_seconds"] is not None else None,
        "joint_scope": "All variants can share the same development outcomes and feature/probe records. The four per-variant totals must not be added to estimate a joint training campaign.",
    }


def _candidate_diagnostics(records, candidate_id, hierarchy):
    selected = []
    warning_messages = {"search": Counter(), "refit": Counter()}
    boundary = defaultdict(lambda: {"choices": 0, "lower_endpoint_choices": 0, "upper_endpoint_choices": 0,
                                    "single_value_grid_choices": 0, "grid_sets": set(), "selected_values": Counter()})
    dimension_fields = ("input_dimension", "target_dimension", "n_training_rows", "feature_width", "basis_width",
                        "requested_landmarks", "actual_landmarks", "retained_nystrom_rank",
                        "decoder_coefficient_count", "coefficient_count", "iterations_realized", "max_iterations")
    dimensions = defaultdict(list)
    tasks_with_warning = search_events = refit_events = iteration_limit_tasks = failed_trials = 0
    for record in records:
        candidate = next(row for row in record["candidates"] if row["candidate_id"] == candidate_id)
        metadata = candidate.get("model_metadata", {})
        search_warnings = candidate.get("warnings", [])
        refit_warnings = metadata.get("warnings", [])
        tasks_with_warning += bool(search_warnings or refit_warnings)
        search_events += len(search_warnings)
        refit_events += len(refit_warnings)
        warning_messages["search"].update(search_warnings)
        warning_messages["refit"].update(refit_warnings)
        failed_trials += sum(row.get("status") != "ok" for row in candidate["trials"])
        if metadata.get("iterations_realized") is not None and metadata.get("max_iterations") is not None:
            iteration_limit_tasks += metadata["iterations_realized"] >= metadata["max_iterations"]
        for field in dimension_fields:
            if metadata.get(field) is not None:
                dimensions[field].append(metadata[field])
        dimensions["serialized_model_bytes"].append(candidate["serialized_model_bytes"])
        for parameter, value in candidate["selected_parameters"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            grid = sorted({float(trial["parameters"][parameter]) for trial in candidate["trials"]
                           if parameter in trial.get("parameters", {}) and not isinstance(trial["parameters"][parameter], bool)})
            if not grid or float(value) not in grid:
                raise ValueError("Selected hyperparameter differs from the saved search grid")
            entry = boundary[parameter]
            entry["choices"] += 1
            entry["grid_sets"].add(tuple(grid))
            entry["selected_values"][format(float(value), ".12g")] += 1
            if len(grid) == 1:
                entry["single_value_grid_choices"] += 1
            else:
                entry["lower_endpoint_choices"] += float(value) == grid[0]
                entry["upper_endpoint_choices"] += float(value) == grid[-1]
        selected.append({**{key: record[key] for key in hierarchy},
                         "launch_to_result_seconds": candidate.get("launch_to_result_seconds"),
                         "child_wall_seconds": candidate.get("child_wall_seconds"),
                         "child_cpu_seconds": candidate.get("child_cpu_seconds"),
                         "whole_worker_peak_rss_kib": candidate.get("process_peak_rss_kib")})
    weights = _weights(selected, hierarchy)
    elapsed = {key: float(weights @ np.asarray([row[key] for row in selected]))
               if all(row[key] is not None for row in selected) else None
               for key in ("launch_to_result_seconds", "child_wall_seconds", "child_cpu_seconds", "whole_worker_peak_rss_kib")}
    boundaries = {}
    for parameter, row in boundary.items():
        boundaries[parameter] = {**row, "grid_sets": [list(values) for values in sorted(row["grid_sets"])],
                                 "selected_values": dict(sorted(row["selected_values"].items()))}
    return {
        "task_fits": len(records), "tasks_with_recorded_warnings": tasks_with_warning,
        "search_warning_events": search_events, "refit_warning_events": refit_events,
        "warning_messages": {stage: dict(sorted(counts.items())) for stage, counts in warning_messages.items()},
        "saved_failed_search_trials": failed_trials, "refits_reaching_recorded_iteration_limit": iteration_limit_tasks,
        "numeric_parameter_endpoint_counts": boundaries,
        "dimension_ranges": {key: _range(values) for key, values in dimensions.items()},
        "equal_source_family_worker_measurements": elapsed,
        "warning_scope": "Recorded search warnings and selected-refit metadata warnings are counted separately. Counts are task-fit descriptions, not independent replications or evidence that retuning would improve performance.",
        "boundary_scope": "Endpoints refer to the declared numerical grid, not a post hoc expanded grid. Every selection from a two-point grid is necessarily an endpoint. No reported result is retuned.",
    }


def _cost_components(records, candidate_id, hierarchy):
    components = ("data_and_load", "search_compute", "selection_cleanup", "final_refit", "forecast", "serialization", "other")
    values = []
    for record in records:
        candidate = next(row for row in record["candidates"] if row["candidate_id"] == candidate_id)
        nodes = {row["id"]: row for row in record["cost_nodes"]}
        required = set(candidate["selection_cost_node_ids"]) | set(candidate["final_cost_node_ids"])
        by_component = {component: [] for component in components}
        for identifier in required:
            node = nodes[identifier]
            kind = node.get("kind", "")
            if identifier in {"data:prepare", "data:load"}:
                component = "data_and_load"
            elif kind in {"search_fit_and_validation", "geometry_features_and_shared_alpha_search"}:
                component = "search_compute"
            elif kind == "selection_bookkeeping_and_cleanup":
                component = "selection_cleanup"
            elif kind == "selected_final_refit":
                component = "final_refit"
            elif kind == "forecast_batch":
                component = "forecast"
            elif kind == "model_serialization_and_metadata":
                component = "serialization"
            else:
                component = "other"
            by_component[component].append(identifier)
        value = {key: record[key] for key in hierarchy}
        for component, identifiers in by_component.items():
            cost = _node_cost(record, identifiers)
            value[component + "_wall_seconds"] = cost["wall_seconds"]
            value[component + "_cpu_seconds"] = cost["cpu_seconds"]
        total = _node_cost(record, required)
        value["total_scientific_wall_seconds"] = total["wall_seconds"]
        value["total_scientific_cpu_seconds"] = total["cpu_seconds"]
        value["launch_to_result_seconds"] = candidate.get("launch_to_result_seconds")
        value["whole_child_wall_seconds"] = candidate.get("child_wall_seconds")
        value["whole_child_cpu_seconds"] = candidate.get("child_cpu_seconds")
        if not math.isclose(sum(value[name + "_wall_seconds"] for name in components), total["wall_seconds"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Method cost decomposition does not reconcile")
        values.append(value)
    weights = _weights(values, hierarchy)
    fields = [key for key in values[0] if key not in hierarchy]
    return {field: float(weights @ np.asarray([row[field] for row in values]))
            if all(row[field] is not None for row in values) else None for field in fields}


def _prediction_inventory(run_root, records, input_hashes):
    by_phase, checked = {}, 0
    for phase in PHASES:
        task_origins = scalar_targets = scalar_predictions = candidate_arrays = 0
        horizons, origin_counts = [], []
        for row in records[phase]:
            path = Path(run_root) / phase / row["predictions_file"]
            if path.parent.resolve() != (Path(run_root) / phase).resolve():
                raise ValueError("Prediction artifact must reside in its task-result directory")
            checksum = digest_file(path)
            if checksum != row["predictions_sha256"]:
                raise ValueError("Saved prediction artifact checksum changed")
            input_hashes[phase + "/" + path.name] = checksum
            with np.load(path, allow_pickle=False) as artifact:
                shape = artifact["target"].shape
                if len(shape) != 2 or shape[1] != row["horizon"] or artifact["origin"].shape != (shape[0],):
                    raise ValueError("Unexpected saved forecast dimensions")
                horizons.append(shape[1]); origin_counts.append(shape[0])
                task_origins += shape[0]
                scalar_targets += math.prod(shape)
                for candidate in row["candidates"]:
                    name = candidate["candidate_id"]
                    if artifact[name].shape != shape or artifact[name + "__origin_losses"].shape != (shape[0],):
                        raise ValueError("Saved candidate predictions have inconsistent dimensions")
                    scalar_predictions += math.prod(shape)
                    candidate_arrays += 1
                    checked += 1
        by_phase[phase] = {"candidate_forecast_arrays": candidate_arrays,
            "task_origin_records": task_origins, "scalar_target_records": scalar_targets,
            "scalar_candidate_predictions": scalar_predictions,
            "forecast_horizon_range": _range(horizons), "evaluation_origins_per_task_range": _range(origin_counts),
            "scope": "Stored task/seed/horizon records are counted for artifact completeness; repeated or overlapping targets are not independent observations."}
    return {"by_phase": by_phase, "candidate_arrays_checked": checked,
            "scope": "Prediction files were checksum-verified and their dimensions inspected. Residual recomputation and equality to frozen source targets/origins are reported from the completed runner VALIDATION, without rerunning inference here."}


def _elapsed_summary(records, initial, confirmation_freeze, log_paths):
    record_events = {(phase, row["task_id"]): row for phase in PHASES for row in records[phase]}
    events, parsed_files = {}, []
    for path in log_paths:
        path = Path(path)
        if not path.is_file():
            raise ValueError("Missing requested benchmark log: " + str(path))
        count = 0
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("event") != "task_complete":
                continue
            key = (event["phase"], event["task_id"])
            if key not in record_events:
                raise ValueError("Benchmark log contains an undeclared task")
            seconds = _finite(event["seconds"], nonnegative=True)
            if not math.isclose(seconds, record_events[key]["task_elapsed_seconds"], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Log elapsed interval differs from its saved task record")
            if key in events and events[key] != seconds:
                raise ValueError("Conflicting elapsed intervals for the same task")
            events[key] = seconds
            count += 1
        parsed_files.append({"filename": path.name, "sha256": digest_file(path), "task_events": count})
    by_phase = {}
    for phase in PHASES:
        phase_events = {key: seconds for key, seconds in events.items() if key[0] == phase}
        by_phase[phase] = {
            "sum_recorded_task_elapsed_seconds": sum(row["task_elapsed_seconds"] for row in records[phase]),
            "saved_task_count": len(records[phase]), "distinct_logged_task_count": len(phase_events),
            "sum_distinct_logged_task_elapsed_seconds": sum(phase_events.values()) if phase_events else None,
            "log_coverage_complete": len(phase_events) == len(records[phase]),
        }
    imports = {}
    for name, freeze in (("initial_freeze_process", initial), ("confirmation_freeze_process", confirmation_freeze)):
        env = freeze.get("environment", {})
        imports[name] = {key: env.get(key) for key in ("shared_library_import_wall_seconds", "shared_library_import_cpu_seconds", "import_scope")}
    return {"by_phase": by_phase, "log_files": parsed_files, "library_initialization_measurements": imports,
        "sum_recorded_task_elapsed_seconds": sum(item["sum_recorded_task_elapsed_seconds"] for item in by_phase.values()),
        "sum_distinct_logged_task_elapsed_seconds": sum(events.values()) if events else None,
        "logged_task_count": len(events), "log_coverage_complete": len(events) == len(record_events),
        "scope": "Actual recorded task intervals include every candidate and probe plus orchestration and saved prediction export. Their sum is distinct from a selector's counterfactual dependency cost and from total wall time spanning preparation, imports, pauses and author analysis. Duplicate log events are counted once; unlogged tasks are not imputed into the log total."}


def _test_evidence(validation, supplied):
    if supplied is not None:
        evidence = read_json(supplied) if isinstance(supplied, (str, Path)) else dict(supplied)
    else:
        evidence = validation.get("unit_tests", {})
        if not isinstance(evidence, dict):
            evidence = {"tests_passed": evidence} if isinstance(evidence, int) else {}
        for key in ("unit_tests_passed", "tests_passed", "tests_run", "test_count"):
            if key in validation:
                evidence[key] = validation[key]
    count = next((evidence[key] for key in ("tests_run", "tests_passed", "unit_tests_passed", "test_count", "count") if key in evidence), None)
    if count is not None and (isinstance(count, bool) or int(count) != count or count < 0):
        raise ValueError("Invalid supplied unit-test count")
    return {"reported_count": count, "evidence": evidence,
            "scope": "Copied from explicit saved test evidence; the report builder does not rerun tests or invent a count."}


def _check_saved_aggregates(report, hierarchy):
    by_policy = defaultdict(list)
    for row in report["task_policy_rows"]:
        by_policy[row["policy"]].append(row)
    for name, rows in by_policy.items():
        saved = report["policies"][name]
        if saved["tasks"] != len(rows) or saved["source_families"] != len({row["source_family"] for row in rows}):
            raise ValueError("Saved summary task/source counts disagree with saved policy rows")
        groups = [(rows, saved["equal_source_family_metrics"])]
        groups += [([row for row in rows if row["source_family"] == family], metrics)
                   for family, metrics in saved["by_source_family"].items()]
        groups += [([row for row in rows if row["budget_id"] == budget], metrics)
                   for budget, metrics in saved["by_budget"].items()]
        for group, metrics in groups:
            weights = _weights(group, hierarchy)
            for field in METRIC_FIELDS:
                values = [row.get(field) for row in group]
                expected = float(weights @ np.asarray(values)) if all(value is not None for value in values) else None
                observed = metrics.get(field)
                if expected is None:
                    if observed is not None:
                        raise ValueError("Saved summary fills a missing measured quantity")
                elif observed is None or not math.isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("Saved source-weighted aggregate disagrees with its policy rows: " + name + "/" + field)


def _phase_summary(phase, report, records, frozen):
    methods = frozen["spec"]["config"]["full_portfolio"]
    hierarchy = frozen["spec"]["weighting_hierarchy"]
    expected = {row["task_id"] for row in records}
    by_id = {row["task_id"]: row for row in records}
    for record in records:
        if {row["candidate_id"] for row in record["candidates"]} != set(methods):
            raise ValueError("Incomplete candidate roster in saved task results")
    per_policy = defaultdict(list)
    for row in report["task_policy_rows"]:
        record = by_id[row["task_id"]]
        candidate = next(item for item in record["candidates"] if item["candidate_id"] == row["chosen_candidate"])
        if not math.isclose(row["test_mse"], candidate["test_loss"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Saved policy loss differs from the selected candidate")
        if not math.isclose(row["scaled_test_mse"], candidate["test_loss"] / record["loss_scale"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Saved policy loss uses a different scale")
        costs = _node_cost(record, row["required_cost_node_ids"])
        if not math.isclose(row["total_policy_wall_seconds"], costs["wall_seconds"] + row["decision_wall_seconds"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Saved policy wall cost differs from recorded dependencies")
        if costs["cpu_seconds"] is not None:
            if not math.isclose(row["total_policy_cpu_seconds"], costs["cpu_seconds"] + row["decision_cpu_seconds"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Saved policy CPU cost differs from recorded dependencies")
        per_policy[row["policy"]].append(row["task_id"])
    expected_policies = {"standalone_" + name for name in methods} | set(POLICY_ORDER)
    if set(per_policy) != expected_policies or set(report["policies"]) != expected_policies:
        raise ValueError("A predeclared method or policy is missing from the saved analysis")
    if any(len(ids) != len(expected) or set(ids) != expected for ids in per_policy.values()):
        raise ValueError("A saved policy has missing or duplicated task decisions")
    _check_saved_aggregates(report, hierarchy)
    budgets = sorted({row["budget_id"] for row in records})
    method_summaries = {}
    for name in methods:
        method_summaries[name] = {
            **report["policies"]["standalone_" + name], "label": METHOD_LABELS.get(name, name),
            "fit_diagnostics": _candidate_diagnostics(records, name, hierarchy),
            "cost_components": _cost_components(records, name, hierarchy),
            "fit_diagnostics_by_budget": {budget: _candidate_diagnostics(
                [row for row in records if row["budget_id"] == budget], name, hierarchy) for budget in budgets},
        }
    result = {
        "scope": report["scope"], "tasks": len(records),
        "source_families": sorted({row["source_family"] for row in records}),
        "series": len({(row["dataset_id"], row["series_id"]) for row in records}),
        "budgets": budgets, "methods": method_summaries,
        "policies": {name: {**report["policies"][name], "label": POLICY_LABELS[name]} for name in POLICY_ORDER},
        "calibration": report["calibration"], "paired_policy_comparisons": report["paired_policy_comparisons"],
        "independence": report["independence"], "generalization_limit": report["generalization_limit"],
        "cost_scope": report["cost_scope"],
    }
    result["matched_method_comparisons"] = {}
    candidate_id = "raw_kahm_rbf"
    for control_id in ("raw_rbf", "raw_affine_rbf", "raw_locality_rbf"):
        candidate, control = method_summaries[candidate_id], method_summaries[control_id]

        def contrast(left, right):
            candidate_loss, control_loss = left["scaled_test_mse"], right["scaled_test_mse"]
            candidate_cost, control_cost = left["total_policy_wall_seconds"], right["total_policy_wall_seconds"]
            return {"candidate_scaled_mse": candidate_loss, "control_scaled_mse": control_loss,
                "loss_advantage": control_loss - candidate_loss,
                "relative_reduction_in_aggregated_scaled_mse": (control_loss - candidate_loss) / control_loss if control_loss > 0 else None,
                "candidate_wall_seconds": candidate_cost, "control_wall_seconds": control_cost,
                "additional_wall_seconds": candidate_cost - control_cost,
                "wall_cost_ratio": candidate_cost / control_cost if control_cost > 0 else None}

        by_family = {family: contrast(candidate["by_source_family"][family], control["by_source_family"][family]) for family in result["source_families"]}
        result["matched_method_comparisons"][candidate_id + "_vs_" + control_id] = {
            "candidate_id": candidate_id, "control_id": control_id,
            "equal_source_family_metrics": contrast(candidate["equal_source_family_metrics"], control["equal_source_family_metrics"]),
            "by_source_family": by_family,
            "by_budget": {budget: contrast(candidate["by_budget"][budget], control["by_budget"][budget]) for budget in budgets},
            "families_favoring_candidate": sum(value["loss_advantage"] > 0 for value in by_family.values()),
            "families_favoring_control": sum(value["loss_advantage"] < 0 for value in by_family.values()),
            "interpretation": "Positive loss advantage favors raw + KAHM. Relative reduction is the ratio of aggregated loss differences to aggregated control loss, not the mean task-level percentage. Family counts are descriptive; all methods use the same complete task roster.",
        }
    return result


def _csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty report CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_csvs(directory, summary):
    method_rows, policy_rows, source_rows, cost_rows = [], [], [], []
    for phase in PHASES:
        values = summary["phases"][phase]
        for kind, items in (("method", values["methods"]), ("policy", values["policies"])):
            for name, item in items.items():
                row = {"phase": phase, "id": name, "label": item["label"],
                       "source_families": item["source_families"], "tasks": item["tasks"],
                       **{key: item["equal_source_family_metrics"].get(key) for key in METRIC_FIELDS}}
                if kind == "method":
                    diag = item["fit_diagnostics"]
                    row.update({key: diag[key] for key in ("tasks_with_recorded_warnings", "search_warning_events",
                        "refit_warning_events", "saved_failed_search_trials", "refits_reaching_recorded_iteration_limit")})
                    row.update(diag["equal_source_family_worker_measurements"])
                    method_rows.append(row)
                    cost_rows.append({"phase": phase, "candidate_id": name, "label": item["label"],
                        "source_families": item["source_families"], "tasks": item["tasks"], **item["cost_components"]})
                else:
                    policy_rows.append(row)
                for family, metrics in item["by_source_family"].items():
                    source_rows.append({"phase": phase, "source_family": family,
                        "source_label": FAMILY_LABELS.get(family, family), "kind": kind,
                        "id": name, "label": item["label"], **{key: metrics.get(key) for key in METRIC_FIELDS}})
    _csv(directory / "method_summary.csv", method_rows)
    _csv(directory / "policy_summary.csv", policy_rows)
    _csv(directory / "source_summary.csv", source_rows)
    _csv(directory / "cost_components.csv", cost_rows)


def _tex(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in str(value))


def _fmt(value, digits=4):
    if value is None:
        return "--"
    return format(float(value), "." + str(digits) + "g")


def _table(headers, rows, caption, label, *, long=False):
    columns = "l" + "r" * (len(headers) - 1)
    header = " & ".join(_tex(value) for value in headers) + r" \\" + "\n"
    body = "".join(" & ".join(_tex(value) for value in row) + r" \\" + "\n" for row in rows)
    if long:
        return (r"{\small\setlength{\tabcolsep}{4pt}" + "\n" + r"\begin{longtable}{" + columns + "}\n" +
            r"\caption{" + caption + r"}\label{" + label + r"}\\" + "\n" +
            r"\toprule" + "\n" + header + r"\midrule" + "\n" + r"\endfirsthead" + "\n" +
            r"\toprule" + "\n" + header + r"\midrule" + "\n" + r"\endhead" + "\n" +
            body + r"\bottomrule" + "\n" + r"\end{longtable}" + "\n}\n")
    return (r"\begin{table}[t]" + "\n" + r"\centering\small\setlength{\tabcolsep}{4pt}" + "\n" +
        r"\caption{" + caption + r"}\label{" + label + "}\n" +
        r"\begin{tabular}{" + columns + "}\n" + r"\toprule" + "\n" + header +
        r"\midrule" + "\n" + body + r"\bottomrule" + "\n" +
        r"\end{tabular}" + "\n" + r"\end{table}" + "\n")


def _all_items(phase_summary):
    return list(phase_summary["methods"].items()) + list(phase_summary["policies"].items())


def _write_tables(directory, summary, frozen, offline, test_evidence):
    confirmation = summary["phases"]["confirmation"]
    families = len(confirmation["source_families"])
    rows = []
    for name, item in confirmation["methods"].items():
        metrics = item["equal_source_family_metrics"]
        rows.append([item["label"], _fmt(metrics["scaled_test_mse"]), _fmt(metrics["total_policy_wall_seconds"]),
                     _fmt(metrics["total_policy_cpu_seconds"]),
                     _fmt(metrics["maximum_measured_node_peak_rss_bytes"] / 2**20) if metrics["maximum_measured_node_peak_rss_bytes"] is not None else "--",
                     _fmt(metrics["serialized_model_bytes"] / 2**10) if metrics["serialized_model_bytes"] is not None else "--"])
    caption = ("Standalone confirmation methods, equally weighted across " + str(families) +
        " original source families using the frozen within-source hierarchy. Loss is training-scale-normalized MSE. "
        "Wall and CPU seconds cover complete warm-library scientific pipelines. RSS is the averaged maximum recorded process high-water mark, "
        "not incremental memory; model size is serialized bytes in KiB. No uncertainty intervals are inferred from dependent tasks.")
    (directory / "method_table.tex").write_text(_table(["Method", "Scaled MSE", "Wall (s)", "CPU (s)", "RSS (MiB)", "Model (KiB)"], rows, caption, "tab:method-summary"))
    rows = []
    for name in POLICY_ORDER:
        item = confirmation["policies"][name]
        metrics = item["equal_source_family_metrics"]
        rows.append([item["label"], _fmt(metrics["scaled_test_mse"]), _fmt(metrics["total_policy_wall_seconds"]),
                     _fmt(metrics["benefit_vs_always_raw"]), _fmt(100 * metrics["practical_harm_vs_always_raw"], 3),
                     _fmt(100 * metrics["selected_kahm_pair_member"], 3)])
    caption = ("Frozen-policy confirmation outcomes over " + str(families) +
        " source families with the same hierarchical weighting. Gain is raw-RBF scaled MSE minus chosen loss. "
        "Harm is the weighted fraction with loss more than 5\\% above raw RBF. The final column counts selection of the "
        "specific raw-plus-KAHM pair member, including in portfolios. Wall cost includes all required probes/searches and decision inference, "
        "then the chosen final refit, forecast and model serialization. Offline evidence acquisition and diagnostic fitting are separate.")
    (directory / "policy_table.tex").write_text(_table(["Policy", "Scaled MSE", "Wall (s)", "Gain", "Harm (%)", "Raw+K (%)"], rows, caption, "tab:policy-summary"))
    cleanup_rows = []
    for name in ("raw_rbf", "raw_kahm_rbf"):
        item = confirmation["methods"][name]
        components = item["cost_components"]
        total = components["total_scientific_wall_seconds"]
        cleanup = components["selection_cleanup_wall_seconds"]
        cleanup_rows.append([item["label"], _fmt(total, 5), _fmt(cleanup, 5),
            _fmt(100 * cleanup / total, 4), _fmt(components["launch_to_result_seconds"], 5)])
    (directory / "cleanup_cost_table.tex").write_text(_table(
        ["Method", "Total (s)", "Cleanup (s)", "Cleanup (%)", "Launch (s)"], cleanup_rows,
        "Confirmation cost transparency for the primary matched pair, with equal source-family weighting. "
        "Total is the complete warm-library scientific pipeline. Cleanup is its measured selection-overhead component, "
        "including explicit garbage collection; the percentage is the ratio of aggregated component and total times. "
        "Launch separately reports the measured isolated-child launch-to-result interval. "
        "These are implementation-specific costs, not intrinsic kernel arithmetic or universal complexity ratios.",
        "tab:cleanup-costs"))
    source_tables, capacity_tables = [], []
    for phase in PHASES:
        values = summary["phases"][phase]
        source_names = values["source_families"]
        for metric, metric_label, suffix in (("scaled_test_mse", "scaled MSE", "loss"), ("total_policy_wall_seconds", "warm-library wall seconds", "cost")):
            rows = [[item["label"]] + [_fmt(item["by_source_family"][family][metric]) for family in source_names]
                    + [_fmt(item["equal_source_family_metrics"][metric])] for _, item in _all_items(values)]
            caption = (phase.capitalize() + " " + metric_label + " by original source family. "
                "The final column gives equal-source aggregation; source columns retain the declared within-source weights. "
                "Development diagnostics use leave-one-source-family-out fits; confirmation diagnostics use the frozen development fit. "
                "Always-policy rows duplicate their corresponding standalone methods deliberately.")
            source_tables.append(_table(["Method or policy"] + [FAMILY_LABELS.get(name, name) for name in source_names] + ["Equal-source"], rows, caption, "tab:source-" + phase + "-" + suffix, long=True))
        budgets = values["budgets"]
        headers = ["Method or policy"]
        for budget in budgets:
            headers += [budget + " MSE", budget + " wall (s)"]
        rows = []
        for _, item in _all_items(values):
            cells = [item["label"]]
            for budget in budgets:
                metrics = item["by_budget"][budget]
                cells += [_fmt(metrics["scaled_test_mse"]), _fmt(metrics["total_policy_wall_seconds"])]
            rows.append(cells)
        caption = (phase.capitalize() + " capacity comparison. Each capacity changes specified training-window and model-size caps; "
            "it is not an equal-time budget. Values are equal-source weighted and retain all prescribed context--horizon settings and seeds.")
        capacity_tables.append(_table(headers, rows, caption, "tab:capacity-" + phase, long=True))
    (directory / "full_source_table.tex").write_text("\n".join(source_tables))
    (directory / "capacity_table.tex").write_text("\n".join(capacity_tables))
    diagnostic_tables = []
    rows = []
    for phase in PHASES:
        for name in DIAGNOSTICS:
            metrics = summary["phases"][phase]["calibration"][name]
            rows.append([phase.capitalize() + ": " + POLICY_LABELS[name],
                _fmt(metrics["weighted_log_benefit_rmse"]), _fmt(metrics["weighted_log_benefit_bias"]),
                _fmt(100 * metrics["weighted_sign_accuracy"], 3),
                _fmt(100 * metrics["weighted_practical_benefit_classification_accuracy"], 3)])
    diagnostic_tables.append(_table(["Diagnostic", "Log RMSE", "Log bias", "Sign (%)", "5% class (%)"], rows,
        "Estimated-benefit diagnostics with frozen hierarchical weights. These are descriptive log-benefit prediction errors and classification accuracies; "
        "they are not calibrated probabilities or interval coverage. No recalibration is fitted on confirmation data.", "tab:diagnostic-calibration", long=True))
    rows = []
    for name in DIAGNOSTICS:
        value = offline["by_variant"][name]
        acquisition = value["matched_label_and_feature_acquisition_task_sum"]
        rows.append([POLICY_LABELS[name], _fmt(acquisition["wall_seconds"]), _fmt(acquisition["cpu_seconds"]),
            _fmt(1000 * value["ridge_fit"]["wall_seconds"]), _fmt(value["acquisition_plus_ridge_fit_wall_seconds"])])
    diagnostic_tables.append(_table(["Variant", "Acquire (s)", "CPU (s)", "Ridge (ms)", "Total (s)"], rows,
        "One-off diagnostic development costs. Acquisition sums the union of both matched predictors' complete search/final nodes and each variant's "
        "metadata/probe/geometry dependencies within every development task. Summing standalone tasks can conservatively repeat preparation across seeds/tasks. "
        "Acquire and total denote wall time; CPU covers acquisition. Do not add the four rows: their developmental labels and some features can be shared. "
        "Ridge fitting is measured separately; no deployment-count amortization is assumed.", "tab:offline-costs", long=True))
    feature_labels = {"log_n_train_windows": "log(1 + training windows)", "log_horizon_ratio": "log(1 + horizon/history)",
        "lag1_autocorrelation": "Lag-one autocorrelation", "kahm_gate_entropy": "KAHM gate entropy",
        "log_affine_projection_residual": "log(1 + affine residual)", "probe_log_ratio": "Probe log-loss ratio"}
    feature_order = list(dict.fromkeys(feature for name in DIAGNOSTICS for feature in frozen["models"][name]["features"]))
    rows = [["Intercept", "--", "--"] + [_fmt(frozen["models"][name]["intercept"], 5) for name in DIAGNOSTICS]]
    for feature in feature_order:
        models = [(name, model, model["features"].index(feature)) for name, model in frozen["models"].items() if feature in model["features"]]
        _, first, index = models[0]
        cells = [feature_labels.get(feature, feature), _fmt(first["mean"][index], 5), _fmt(first["scale"][index], 5)]
        for name in DIAGNOSTICS:
            model = frozen["models"][name]
            cells.append(_fmt(model["coefficients"][model["features"].index(feature)], 5) if feature in model["features"] else "--")
        rows.append(cells)
    diagnostic_tables.append(_table(["Feature", "Mean", "Scale", "Meta", "Meta+G", "Meta+P", "Meta+P+G"], rows,
        "Frozen development-only standardization and ridge coefficients, rounded for display. G denotes geometry and P the cheap probe. "
        "An absent feature is shown by a dash. Predictions use the unrounded values in the frozen diagnostic JSON and the fixed zero decision threshold.", "tab:diagnostic-coefficients", long=True))
    rows, boundary_rows = [], []
    for phase in PHASES:
        for name, item in summary["phases"][phase]["methods"].items():
            diag = item["fit_diagnostics"]
            rows.append([phase.capitalize() + ": " + item["label"], diag["task_fits"], diag["tasks_with_recorded_warnings"],
                         diag["search_warning_events"], diag["refit_warning_events"], diag["refits_reaching_recorded_iteration_limit"]])
            for parameter, counts in diag["numeric_parameter_endpoint_counts"].items():
                grids = "; ".join(",".join(_fmt(value, 3) for value in grid) for grid in counts["grid_sets"])
                boundary_rows.append([phase.capitalize() + ": " + item["label"], parameter, counts["choices"],
                    counts["lower_endpoint_choices"], counts["upper_endpoint_choices"], grids])
    diagnostic_tables.append(_table(["Method", "Fits", "Warned", "Search", "Refit", "At cap"], rows,
        "Recorded fitting behavior. Search and selected-refit warning events are separated. Counts describe task fits, which include dependent seeds and settings. "
        "Warned counts fits with recorded warnings; at cap counts selected refits reaching the recorded iteration limit. "
        "Full warning text is retained in the JSON report. No warning-triggered post hoc retuning is applied.", "tab:fit-warnings", long=True))
    diagnostic_tables.append(_table(["Method", "Parameter", "Choices", "Lower", "Upper", "Declared grid"], boundary_rows,
        "Selection at numerical search-grid endpoints. Counts are descriptive; a two-point grid necessarily chooses an endpoint. "
        "These observations do not establish the benefit of a larger grid, and confirmation outcomes are not used to retune any method.", "tab:grid-boundaries", long=True))
    (directory / "diagnostic_details.tex").write_text("\n".join(diagnostic_tables))
    validation = summary["validation"]
    audit = summary["prediction_inventory"]
    total_tasks = sum(summary["phases"][phase]["tasks"] for phase in PHASES)
    rows = [["Final runner validation", validation["status"]],
            ["Development original source families", len(summary["phases"]["development"]["source_families"])],
            ["Confirmation original source families", families],
            ["Development / confirmation task runs", str(summary["phases"]["development"]["tasks"]) + " / " + str(confirmation["tasks"])],
            ["Total task runs", total_tasks],
            ["Saved candidate forecast arrays / dimensions checked", audit["candidate_arrays_checked"]],
            ["Candidate losses independently recomputed by runner", validation["predictions_recomputed"]],
            ["Targets and origins equal frozen source windows", str(validation.get("source_targets_and_origins_equal", False))],
            ["Historical tracked files verified unchanged", validation["historical_tracked_files_unchanged"]],
            ["Scientific protocol and both stage freezes intact", str(validation["frozen_protocol_intact"])],
            ["Unit tests in supplied evidence", test_evidence["reported_count"] if test_evidence["reported_count"] is not None else "Not recorded in supplied evidence"]]
    (directory / "audit_table.tex").write_text(_table(["Audit item", "Recorded result"], rows,
        "Completed experiment audit. Counts of forecasts, tasks and tests serve reproducibility checks; they do not increase the number of independent source families. "
        "The report builder reads existing predictions and validation evidence without rerunning forecasting or diagnostic inference.", "tab:validation-audit"))


def _method_color(name):
    if "kahm" in name:
        return "#882255"
    if "affine" in name:
        return "#D55E00"
    if "locality" in name:
        return "#009E73"
    if name.startswith("raw"):
        return "#0072B2"
    return "#555555"


def _annotate_identifiers(ax, coordinates):
    """Place short identifiers with deterministic offsets, without moving data."""
    ax.figure.canvas.draw()
    boxes = []
    candidates = [(8, 8), (8, -13), (-15, 8), (-15, -13), (19, 19), (19, -24),
                  (-26, 19), (-26, -24), (32, 5), (-38, 5), (4, 31), (4, -36)]
    for identifier, x, y in coordinates:
        point = ax.transData.transform((x, y))
        best = None
        for dx, dy in candidates:
            center = point + np.asarray([dx, dy]) * ax.figure.dpi / 72
            inside = ax.bbox.x0 + 9 < center[0] < ax.bbox.x1 - 9 and ax.bbox.y0 + 9 < center[1] < ax.bbox.y1 - 9
            overlaps = sum(abs(center[0] - old[0]) < 18 and abs(center[1] - old[1]) < 15 for old in boxes)
            score = (not inside, overlaps, dx * dx + dy * dy)
            if best is None or score < best[0]:
                best = (score, dx, dy, center)
        _, dx, dy, center = best
        boxes.append(center)
        ax.annotate(identifier, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha="center", va="center", color="#202020",
                    arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.45},
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.85})


def _save_figure(fig, path, plt):
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": VERSION, "CreationDate": None, "ModDate": None})
    plt.close(fig)


def _write_figures(directory, summary, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    directory.mkdir(parents=True, exist_ok=True)
    confirmation = summary["phases"]["confirmation"]
    n_families = len(confirmation["source_families"])
    key_selectors = ("always_raw", "always_kahm", "validation_pair", "validation_raw_portfolio",
        "validation_full_portfolio", "cheap_probe", "metadata_probe", "metadata_probe_geometry")
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 7.0))
    fig.subplots_adjust(left=0.13, right=0.60, top=0.90, bottom=0.15, hspace=0.46)
    plot_scales = {}
    for ax, kind, items, title in (
        (axes[0], "methods", list(confirmation["methods"].items()), "(a) Standalone methods"),
        (axes[1], "policies", [(name, confirmation["policies"][name]) for name in key_selectors], "(b) Selection policies"),
    ):
        coordinates, handles, labels = [], [], []
        costs = [item["equal_source_family_metrics"]["total_policy_wall_seconds"] for _, item in items]
        losses = [item["equal_source_family_metrics"]["scaled_test_mse"] for _, item in items]
        xscale = "log" if min(costs) > 0 else "linear"
        yscale = "log" if min(losses) > 0 else "linear"
        ax.set_xscale(xscale); ax.set_yscale(yscale)
        for index, (name, item) in enumerate(items):
            identifier = str(index + 1) if kind == "methods" else chr(65 + index)
            metrics = item["equal_source_family_metrics"]
            x, y = metrics["total_policy_wall_seconds"], metrics["scaled_test_mse"]
            color = _method_color(name) if kind == "methods" else ("#882255" if name == "metadata_probe_geometry" else "#0072B2")
            ax.scatter([x], [y], s=34, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            coordinates.append((identifier, x, y))
            handles.append(Line2D([], [], color=color, marker="$" + identifier + "$", markersize=8, linestyle="None"))
            labels.append(item["label"])
        ax.margins(x=0.23, y=0.25)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Complete warm-library wall seconds" + (" (log)" if xscale == "log" else ""))
        ax.set_ylabel("Source-weighted scaled MSE" + (" (log)" if yscale == "log" else ""))
        ax.grid(True, which="major", color="#dddddd", linewidth=0.55, zorder=0)
        _annotate_identifiers(ax, coordinates)
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.02), frameon=False,
                  handletextpad=0.7, borderaxespad=0, labelspacing=0.5)
        plot_scales[kind] = {"x": xscale, "y": yscale}
    fig.suptitle("Confirmation: equal weighting of " + str(n_families) + " original source families", fontsize=10.5)
    fig.text(0.13, 0.025, "Lower loss and lower cost are preferable. Costs include required selection work.\nOffline diagnostic development and process/library setup are separate; no uncertainty bars.", fontsize=8)
    _save_figure(fig, directory / "accuracy_cost.pdf", plt)

    families = confirmation["source_families"]
    kahm = confirmation["methods"]["raw_kahm_rbf"]["by_source_family"]
    contrasts = (("raw_rbf", "vs raw RBF", "#0072B2", "o"),
                 ("raw_affine_rbf", "vs raw + affine", "#D55E00", "s"),
                 ("raw_locality_rbf", "vs raw + locality", "#009E73", "^"))
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    y = np.arange(len(families))
    source_gains = {}
    for index, (control, label, color, marker) in enumerate(contrasts):
        gains = [confirmation["methods"][control]["by_source_family"][family]["scaled_test_mse"] - kahm[family]["scaled_test_mse"] for family in families]
        source_gains[control] = dict(zip(families, gains))
        ax.scatter(gains, y + (index - 1) * 0.18, color=color, marker=marker, s=45, label=label, zorder=3)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_yticks(y, [FAMILY_LABELS.get(family, family) for family in families])
    ax.invert_yaxis()
    ax.set_xlabel("Scaled-MSE advantage of raw + KAHM (control loss minus KAHM loss)")
    ax.set_title("Matched-source confirmation comparisons (" + str(n_families) + " original sources)", loc="left")
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.14, bottom=0.28, top=0.86)
    fig.text(0.14, 0.015, "Positive values favor raw + KAHM. Points are within-source weighted paired gains; no confidence intervals.", fontsize=8)
    _save_figure(fig, directory / "source_gains.pdf", plt)

    task_rows = reports["confirmation"]["task_policy_rows"]
    diagnostic_rows = [row for row in task_rows if row["policy"] in DIAGNOSTICS]
    observed = [row["observed_pair_log_benefit"] for row in diagnostic_rows]
    predicted = [row["predicted_pair_log_benefit"] for row in diagnostic_rows]
    lower, upper = min(observed + predicted + [0.0]), max(observed + predicted + [0.0])
    span = max(upper - lower, 0.1)
    limits = (lower - 0.06 * span, upper + 0.06 * span)
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#999933", "#555555")
    colors = {family: palette[index % len(palette)] for index, family in enumerate(families)}
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.5), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, DIAGNOSTICS):
        for family in families:
            rows = [row for row in diagnostic_rows if row["policy"] == name and row["source_family"] == family]
            ax.scatter([row["observed_pair_log_benefit"] for row in rows],
                       [row["predicted_pair_log_benefit"] for row in rows],
                       color=colors[family], s=18, alpha=0.40, edgecolor="none", zorder=2)
            mean = confirmation["calibration"][name]["by_source_family"][family]
            ax.scatter([mean["mean_observed_log_benefit"]], [mean["mean_predicted_log_benefit"]],
                       color=colors[family], marker="D", s=48, edgecolor="black", linewidth=0.5, zorder=4)
        ax.plot(limits, limits, linestyle="--", color="#777777", linewidth=0.7, zorder=1)
        ax.axvline(0, color="#bbbbbb", linewidth=0.6, zorder=0)
        ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=0)
        ax.set_xlim(limits); ax.set_ylim(limits)
        ax.set_title(POLICY_LABELS[name], fontsize=9.5, loc="left")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[-1]:
        ax.set_xlabel("Observed matched-pair log benefit", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Frozen predicted log benefit", fontsize=8.5)
    handles = [Line2D([], [], color=colors[family], marker="o", linestyle="None", markersize=5,
                      label=FAMILY_LABELS.get(family, family)) for family in families]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.057), ncol=min(4, len(families)), frameon=False)
    fig.suptitle("Frozen diagnostics on " + str(n_families) + " reserved source families", fontsize=10.5, y=0.98)
    fig.text(0.5, 0.018, "Small points: dependent task runs; diamonds: source means. Dashed line: exact prediction.\nPositive benefit favors raw + KAHM. No fitted recalibration or uncertainty bands.", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.90, hspace=0.28, wspace=0.17)
    _save_figure(fig, directory / "diagnostic_predictions.pdf", plt)
    return {"accuracy_cost_axis_scales": plot_scales, "source_gains": source_gains,
            "diagnostic_common_axis_limits": list(limits), "source_families": n_families,
            "matplotlib_version": matplotlib.__version__,
            "scope": "All plotted losses, costs and predictions come from saved analysis outputs. Source gains subtract identically weighted saved method means. No forecaster or diagnostic is rerun; no inferential error bars are constructed."}


def build_report(run_root, reports_dir=None, manuscript_dir=None, *, test_summary=None, log_paths=None, make_figures=True):
    """Build all deliverables after validation; return the complete summary."""
    run_root = Path(run_root)
    reports_dir = Path(reports_dir) if reports_dir is not None else run_root / "reports"
    manuscript_dir = Path(manuscript_dir) if manuscript_dir is not None else ROOT / "manuscript" / "generated"
    evidence = _gated_inputs(run_root)
    reports, records, frozen = evidence["reports"], evidence["records"], evidence["frozen"]
    if test_summary is None and (run_root / "UNIT_TESTS.json").is_file():
        test_summary = run_root / "UNIT_TESTS.json"
    tests = _test_evidence(evidence["validation"], test_summary)
    if isinstance(test_summary, (str, Path)):
        tests["evidence_sha256"] = digest_file(test_summary)
        tests["evidence_filename"] = Path(test_summary).name
    if log_paths is None:
        log_paths = sorted(run_root.glob("*.log"))
    prediction_inventory = _prediction_inventory(run_root, records, evidence["input_sha256"])
    if prediction_inventory["candidate_arrays_checked"] != evidence["validation"]["predictions_recomputed"]:
        raise ValueError("Saved validation prediction count differs from actual prediction artifacts")
    phases = {phase: _phase_summary(phase, reports[phase], records[phase], frozen) for phase in PHASES}
    if set(phases["development"]["source_families"]) & set(phases["confirmation"]["source_families"]):
        raise ValueError("Source families overlap between development and confirmation")
    offline = offline_costs(records["development"], frozen)
    elapsed = _elapsed_summary(records, evidence["initial"], evidence["confirmation_freeze"], log_paths)
    offline["actual_campaign_task_intervals"] = elapsed
    summary = {
        "format": VERSION, "reporter_sha256": digest_file(__file__),
        "postprocessing_environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "input_sha256": evidence["input_sha256"], "frozen_analysis_spec_sha256": frozen["spec_sha256"],
        "frozen_diagnostic_sha256": frozen["freeze_sha256"], "phases": phases,
        "validation": evidence["validation"], "unit_test_evidence": tests,
        "prediction_inventory": prediction_inventory, "actual_campaign_task_intervals": elapsed,
        "offline_diagnostic_costs": {key: value for key, value in offline.items() if key != "actual_campaign_task_intervals"},
        "reporting_rules": {
            "outcomes": "Saved source-weighted scaled MSE and warm-library scientific-pipeline costs; no new tuning, forecasting, diagnostic inference or recalibration.",
            "dependencies": "The chosen pipeline pays every recorded dependency and scalar decision cost. Offline developmental outcome acquisition is distinct from fitting a ridge to already available labels.",
            "uncertainty": "Four original confirmation families provide bounded descriptive evidence. No confidence intervals or p-values are derived from dependent windows, horizons, series or seeds.",
            "memory": "Reported RSS is a maximum process high-water mark at required nodes, including earlier process history and interpreter/input memory; source summaries average that task-level quantity.",
            "cost_components": "Component CSV uses the same frozen source hierarchy. Data/load, search computation, cleanup, refit, forecast and serialization reconcile to the standalone scientific total; whole-child and launch-to-result intervals are separate measured quantities.",
            "warnings": "Warnings and numerical-grid endpoint choices describe the frozen fits. They do not trigger post hoc retuning or task removal.",
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    _write_csvs(reports_dir, summary)
    _write_tables(manuscript_dir, summary, frozen, offline, tests)
    summary["figure_metadata"] = _write_figures(manuscript_dir / "figures", summary, reports) if make_figures else {"status": "not requested"}
    write_json(reports_dir / "offline_costs.json", offline)
    write_json(reports_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "natural_v1")
    parser.add_argument("--reports-dir", type=Path, help="Default: RUN/reports")
    parser.add_argument("--manuscript-dir", type=Path, help="Generated TeX directory; PDFs go in its figures subdirectory")
    parser.add_argument("--test-summary", type=Path, help="Explicit saved unit-test evidence; default RUN/UNIT_TESTS.json if present")
    parser.add_argument("--log", type=Path, action="append", dest="log_paths", help="Benchmark JSON-lines log, repeatable; default RUN/*.log")
    parser.add_argument("--no-figures", action="store_true", help="Build numerical and TeX artifacts only")
    args = parser.parse_args(argv)
    summary = build_report(args.run_root, args.reports_dir, args.manuscript_dir,
        test_summary=args.test_summary, log_paths=args.log_paths, make_figures=not args.no_figures)
    print(json.dumps({"status": "complete", "format": summary["format"],
        "confirmation_source_families": len(summary["phases"]["confirmation"]["source_families"]),
        "candidate_forecast_arrays": summary["prediction_inventory"]["candidate_arrays_checked"]}, indent=2))


if __name__ == "__main__":
    main()
