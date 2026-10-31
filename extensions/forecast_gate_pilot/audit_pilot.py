#!/usr/bin/env python3
"""Read-only numerical audit of saved exploratory pilot evidence.

Checks hashes, roster completeness, causal window construction, finite and
shape-correct forecasts, loss reconstruction, recorded final selection, and
exact baseline recovery for zero correction. Does not refit every candidate,
certify global optimality, or create independent predictive evidence.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    prefix = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True).encode("utf-8")
    return hashlib.sha256(prefix + b"\0" + value.tobytes()).hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def close(left, right, atol=1e-12, rtol=1e-10):
    return finite_number(left) and finite_number(right) and bool(np.isclose(left, right, atol=atol, rtol=rtol))


def audit(output):
    output = Path(output).resolve()
    failures = []
    notes = []
    counts = {"hash_checks": 0, "window_checks": 0, "prediction_checks": 0,
              "loss_checks": 0, "selection_checks": 0, "zero_correction_checks": 0,
              "cost_checks": 0, "optimization_checks": 0}

    def check(condition, message, category=None):
        if category is not None:
            counts[category] += 1
        if not bool(condition):
            failures.append(message)

    def check_hash(path, expected, label):
        check(path.is_file() and sha256_file(path) == expected,
              f"Hash mismatch or missing file: {label}", "hash_checks")

    try:
        protocol = read_json(output / "protocol.json")
        freeze = read_json(output / "freeze.json")
        summary = read_json(output / "summary.json")
    except Exception as exc:
        return {"passed": False, "failures": [f"Cannot read required run metadata: {exc}"],
                "counts": counts, "scope": __doc__}
    check_hash(output / "protocol.json", freeze["protocol_sha256"], "protocol.json")
    check(freeze["source_sha256"] == protocol["source_sha256"], "Freeze/source hash rosters disagree")
    for name, expected in freeze["source_sha256"].items():
        check_hash(output / "source_snapshot" / name, expected, f"source_snapshot/{name}")
    check(protocol.get("not_confirmatory") is True, "Protocol must retain exploratory status")
    check(freeze.get("observed_D_outcomes_before_freeze") is False,
          "Freeze does not affirm no D outcomes before protocol")
    expected_ids = {task["task_id"] for task in protocol["tasks"]}
    actual_dirs = {path.name for path in (output / "tasks").iterdir() if path.is_dir()}
    check(actual_dirs == expected_ids, f"Task roster mismatch: expected {sorted(expected_ids)}; found {sorted(actual_dirs)}")
    methods = list(protocol["methods"])
    horizons = np.array(protocol["generator"]["offsets"], dtype=np.int64)
    origins = np.array(protocol["generator"]["origins_zero_based"], dtype=np.int64)
    history = int(protocol["generator"]["history"])
    task_losses = {}
    available_forecasts = 0
    complete_tasks = 0
    fallback_count = 0
    for task in protocol["tasks"]:
        task_id = task["task_id"]
        directory = output / "tasks" / task_id
        if not directory.is_dir():
            continue
        try:
            record = read_json(directory / "task.json")
            diagnostics = read_json(directory / "diagnostics.json")
            with np.load(directory / "inputs.npz", allow_pickle=False) as loaded:
                inputs = {key: loaded[key] for key in loaded.files}
            with np.load(directory / "predictions.npz", allow_pickle=False) as loaded:
                predictions = {key: loaded[key] for key in loaded.files}
        except Exception as exc:
            failures.append(f"{task_id}: cannot read complete task: {exc}")
            continue
        check(record["task_id"] == task_id and record["law"] == task["law"] and record["seed"] == task["seed"],
              f"{task_id}: task identity mismatch")
        check(diagnostics.get("config") == protocol["model_config"],
              f"{task_id}: used model configuration differs from frozen protocol")
        check(record["status"] == "completed", f"{task_id}: status {record['status']}")
        complete_tasks += int(record["status"] == "completed")
        for filename, key in [("inputs.npz", "inputs_file_sha256"),
                              ("predictions.npz", "prediction_file_sha256"),
                              ("diagnostics.json", "diagnostics_file_sha256")]:
            check_hash(directory / filename, record[key], f"{task_id}/{filename}")
        for key, expected in record["prediction_array_sha256"].items():
            check(key in predictions and array_hash(predictions[key]) == expected,
                  f"{task_id}: array hash mismatch {key}", "hash_checks")
        for split in ("A", "B", "C", "D"):
            prefix = f"{task_id}/{split}"
            split_record = record["split_records"][split]
            for key, expected in split_record["array_sha256"].items():
                check(array_hash(inputs[f"{split}_{key}"]) == expected,
                      f"{prefix}: array hash mismatch {key}", "hash_checks")
            episode_values = inputs[f"{split}_episodes"]
            check(array_hash(episode_values) == split_record["episodes_sha256"],
                  f"{prefix}: episode hash mismatch", "hash_checks")
            expected_episodes = int(protocol["split_episodes"][split])
            check(episode_values.shape == (expected_episodes, protocol["generator"]["episode_length"]),
                  f"{prefix}: episode shape mismatch", "window_checks")
            check(np.isfinite(episode_values).all(), f"{prefix}: nonfinite episodes", "window_checks")
            row_indices = inputs[f"{split}_row_indices"]
            group_ids = inputs[f"{split}_groups"]
            row_origins = inputs[f"{split}_origins"]
            expected_groups = row_indices // len(origins)
            expected_origins = origins[row_indices % len(origins)]
            check(np.array_equal(group_ids, expected_groups), f"{prefix}: episode group IDs mismatch", "window_checks")
            check(np.array_equal(row_origins, expected_origins), f"{prefix}: origin IDs mismatch", "window_checks")
            check(len(np.unique(row_indices)) == len(row_indices), f"{prefix}: duplicate capped rows", "window_checks")
            total = expected_episodes * len(origins)
            cap = protocol["split_row_caps"][split]
            expected_rows = total if cap is None else min(total, int(cap))
            check(len(row_indices) == expected_rows, f"{prefix}: wrong number of rows", "window_checks")
            check(np.all((row_indices >= 0) & (row_indices < total)), f"{prefix}: invalid row index", "window_checks")
            x = inputs[f"{split}_X"]
            y = inputs[f"{split}_Y"]
            expected_x = np.array([episode_values[g, t-history+1:t+1]
                                   for g, t in zip(group_ids, row_origins)])
            expected_y = np.array([episode_values[g, t+horizons]
                                   for g, t in zip(group_ids, row_origins)])
            check(np.array_equal(x, expected_x), f"{prefix}: history construction mismatch", "window_checks")
            check(np.array_equal(y, expected_y), f"{prefix}: target construction mismatch", "window_checks")
            if split == "D":
                check(np.array_equal(row_indices, np.arange(total)), f"{prefix}: evaluation rows incomplete", "window_checks")
        expected_prediction_keys = {f"{split}__{method}" for split in ("C", "D") for method in methods}
        check(set(predictions) == expected_prediction_keys,
              f"{task_id}: prediction roster mismatch; missing {sorted(expected_prediction_keys-set(predictions))}, unexpected {sorted(set(predictions)-expected_prediction_keys)}")
        task_losses[task_id] = {}
        for method in methods:
            key_d, key_c = f"D__{method}", f"C__{method}"
            if key_d not in predictions or key_c not in predictions:
                continue
            prediction_d, prediction_c = predictions[key_d], predictions[key_c]
            valid = True
            for split, pred in (("C", prediction_c), ("D", prediction_d)):
                shape_valid = pred.shape == inputs[f"{split}_Y"].shape
                finite_valid = np.isfinite(pred).all()
                check(shape_valid, f"{task_id}/{method}: {split} shape mismatch", "prediction_checks")
                check(finite_valid, f"{task_id}/{method}: {split} nonfinite prediction", "prediction_checks")
                valid = valid and shape_valid and finite_valid
            if not valid:
                continue
            available_forecasts += 1
            squared_error = (inputs["D_Y"] - prediction_d) ** 2
            episode_horizon = np.array([squared_error[inputs["D_groups"] == group].mean(axis=0)
                                        for group in np.unique(inputs["D_groups"])])
            horizon_loss = episode_horizon.mean(axis=0)
            mse = float(horizon_loss.mean())
            c_mse = float(np.mean((inputs["C_Y"] - prediction_c) ** 2))
            metrics = record["metrics"].get(method, {})
            check(close(mse, metrics.get("mse")), f"{task_id}/{method}: D MSE mismatch", "loss_checks")
            check(close(c_mse, metrics.get("C_mse")), f"{task_id}/{method}: C MSE mismatch", "loss_checks")
            for index, horizon in enumerate(horizons):
                check(close(float(horizon_loss[index]), metrics.get("horizon_mse", {}).get(str(horizon))),
                      f"{task_id}/{method}: horizon {horizon} MSE mismatch", "loss_checks")
            check(np.allclose(episode_horizon.mean(axis=1), metrics.get("episode_mse", []), rtol=1e-10, atol=1e-12),
                  f"{task_id}/{method}: episode MSE mismatch", "loss_checks")
            task_losses[task_id][method] = {"mse": mse, "horizons": horizon_loss, "C_mse": c_mse}
            if method == "persistence":
                for split in ("C", "D"):
                    expected = np.repeat(inputs[f"{split}_X"][:, -1:], len(horizons), axis=1)
                    check(np.array_equal(expected, predictions[f"{split}__{method}"]),
                          f"{task_id}: persistence mismatch on {split}", "prediction_checks")
                continue
            method_diagnostics = diagnostics.get("methods", {}).get(method)
            check(isinstance(method_diagnostics, dict), f"{task_id}/{method}: missing method diagnostics")
            if not isinstance(method_diagnostics, dict):
                continue
            check(close(c_mse, method_diagnostics.get("selected_c_mse")),
                  f"{task_id}/{method}: overall selected C diagnostic loss mismatch", "selection_checks")
            dependencies = method_diagnostics.get("cost_dependencies", [])
            check(len(dependencies) == len(set(dependencies)),
                  f"{task_id}/{method}: duplicate cost dependency", "cost_checks")
            nodes = diagnostics.get("cost_nodes", {})
            for measure in ("wall_seconds", "cpu_seconds"):
                required = [nodes.get(name, {}).get(measure) for name in dependencies]
                check(all(finite_number(value) and value >= 0 for value in required),
                      f"{task_id}/{method}: invalid/missing {measure} dependency", "cost_checks")
                if all(finite_number(value) for value in required):
                    check(close(sum(required), method_diagnostics.get("charged_" + measure)),
                          f"{task_id}/{method}: charged {measure} mismatch", "cost_checks")
            fallback_count += len(method_diagnostics.get("fallbacks", []))
            if method == "raw_ridge":
                baseline_candidates = method_diagnostics.get("candidates", [])
                check(len(baseline_candidates) == len(protocol["model_config"]["baseline_alpha_grid"]),
                      f"{task_id}: incomplete baseline ridge grid", "selection_checks")
                if baseline_candidates:
                    baseline_best = min(baseline_candidates, key=lambda value: value["inner_a_mse"])
                    check(method_diagnostics.get("selected_alpha") == baseline_best["alpha"],
                          f"{task_id}: baseline ridge is not first recorded inner-A minimum", "selection_checks")
                fitting_groups = set(method_diagnostics.get("inner_training_groups", []))
                validation_groups = set(method_diagnostics.get("inner_validation_groups", []))
                check(bool(fitting_groups) and bool(validation_groups) and not fitting_groups.intersection(validation_groups),
                      f"{task_id}: baseline inner-A episode leakage or empty split", "selection_checks")
            elif method == "raw_rbf" and not method_diagnostics.get("fallback"):
                rbf_candidates = method_diagnostics.get("candidates", [])
                expected_count = (len(protocol["model_config"]["baseline_alpha_grid"])
                                  * len(protocol["model_config"]["raw_rbf_bandwidth_factors"]))
                check(len(rbf_candidates) == expected_count, f"{task_id}: incomplete raw RBF candidate grid", "selection_checks")
                if rbf_candidates:
                    rbf_best = min(rbf_candidates, key=lambda value: value["b_mse"])
                    check(method_diagnostics.get("selected") == rbf_best,
                          f"{task_id}: raw RBF is not first recorded B minimum", "selection_checks")
            horizon_diagnostics = method_diagnostics.get("horizons")
            if method.startswith(("fixed_", "direct_")):
                check(isinstance(horizon_diagnostics, list) and len(horizon_diagnostics) == len(horizons),
                      f"{task_id}/{method}: missing per-horizon selection evidence")
                if not isinstance(horizon_diagnostics, list):
                    continue
                for index, horizon_diag in enumerate(horizon_diagnostics):
                    candidates = horizon_diag.get("candidates", [])
                    selected = horizon_diag.get("selected", {})
                    for optimization in horizon_diag.get("optimizations", []):
                        evaluations = optimization.get("evaluations", [])
                        check(optimization.get("nfev") == len(evaluations),
                              f"{task_id}/{method}/{index}: optimizer evaluation count mismatch", "optimization_checks")
                        check(len(evaluations) <= protocol["model_config"]["max_objective_evaluations"],
                              f"{task_id}/{method}/{index}: optimizer exceeded declared evaluation budget", "optimization_checks")
                        eta = np.asarray(optimization.get("eta", []), dtype=float)
                        bound = np.log(protocol["model_config"]["temperature_bound"])
                        check(eta.shape == (len(protocol["model_config"]["blocks"]),)
                              and np.isfinite(eta).all() and np.all(np.abs(eta) <= bound + 1e-12),
                              f"{task_id}/{method}/{index}: invalid retained gate temperature", "optimization_checks")
                        initial = optimization.get("initial_normalized_b_mse")
                        retained = optimization.get("best_normalized_b_mse")
                        if finite_number(initial) and finite_number(retained):
                            check(retained <= initial + 1e-12,
                                  f"{task_id}/{method}/{index}: optimizer discarded better identity incumbent", "optimization_checks")
                    check(bool(candidates), f"{task_id}/{method}/{index}: missing C candidate roster", "selection_checks")
                    if not candidates:
                        continue
                    if selected.get("fallback") != "shared_bank_failure":
                        expected_candidates = {"zero"}.union({
                            f"alpha={float(alpha):g},gamma={float(gamma):g}"
                            for alpha in protocol["model_config"]["alpha_grid"]
                            for gamma in protocol["model_config"]["gamma_grid"] if float(gamma) != 0.0
                        })
                        ids = [candidate.get("candidate_id") for candidate in candidates]
                        check(len(ids) == len(expected_candidates) and set(ids) == expected_candidates,
                              f"{task_id}/{method}/{index}: incomplete C candidate grid", "selection_checks")
                    check(candidates[0].get("gamma") == 0,
                          f"{task_id}/{method}/{index}: zero correction is not the first candidate", "selection_checks")
                    finite_candidates = [candidate for candidate in candidates if finite_number(candidate.get("c_mse"))]
                    check(bool(finite_candidates), f"{task_id}/{method}/{index}: no finite C candidate", "selection_checks")
                    if not finite_candidates:
                        continue
                    # Python min preserves the first exact minimum, matching
                    # strict incumbent replacement in the registered policy.
                    best = min(finite_candidates, key=lambda value: value["c_mse"])
                    check(selected.get("candidate_id") == best.get("candidate_id"),
                          f"{task_id}/{method}/{index}: selected candidate is not first strict C minimum", "selection_checks")
                    selected_loss = float(np.mean((inputs["C_Y"][:, index] - prediction_c[:, index]) ** 2))
                    check(close(selected_loss, selected.get("c_mse")),
                          f"{task_id}/{method}/{index}: selected C loss mismatch", "selection_checks")
                    check(close(selected.get("c_mse"), best.get("c_mse")),
                          f"{task_id}/{method}/{index}: selected/minimum recorded C losses mismatch", "selection_checks")
                    if selected.get("gamma") == 0:
                        for split in ("C", "D"):
                            baseline_key = f"{split}__raw_ridge"
                            if baseline_key not in predictions:
                                check(False, f"{task_id}/{method}: missing zero-correction baseline", "zero_correction_checks")
                            else:
                                check(np.array_equal(predictions[f"{split}__{method}"][:, index], predictions[baseline_key][:, index]),
                                      f"{task_id}/{method}/{index}: gamma=0 fails exact {split} baseline recovery", "zero_correction_checks")
    check(available_forecasts == len(protocol["tasks"]) * len(methods), "Incomplete forecast roster")
    check(summary["available_forecasts"] == available_forecasts, "Summary forecast count mismatch")
    check(summary["complete_task_count"] == complete_tasks, "Summary complete-task count mismatch")
    for aggregate in summary["law_method_aggregates"]:
        matching = [task for task in protocol["tasks"] if task["law"] == aggregate["law"]]
        values = [task_losses.get(task["task_id"], {}).get(aggregate["method"]) for task in matching]
        complete = len(values) == len(protocol["seeds"]) and all(value is not None for value in values)
        check(aggregate["complete"] == complete, f"Aggregate completeness mismatch: {aggregate['law']}/{aggregate['method']}")
        if complete:
            expected = float(np.mean([value["mse"] for value in values]))
            check(close(expected, aggregate["mean_mse"]), f"Aggregate MSE mismatch: {aggregate['law']}/{aggregate['method']}", "loss_checks")
        else:
            check(aggregate["mean_mse"] is None, "Incomplete aggregate must remain undefined")
    try:
        with (output / "task_metrics.csv").open(newline="", encoding="utf-8") as file:
            table = list(csv.DictReader(file))
        check(len(table) == len(protocol["tasks"]) * len(methods), "Task metrics CSV roster length mismatch")
        check(len({(row["task_id"], row["method"]) for row in table}) == len(table), "Duplicate task metrics CSV entries")
        for row in table:
            value = task_losses.get(row["task_id"], {}).get(row["method"])
            if value is not None:
                check(close(float(row["mse"]), value["mse"]), f"CSV MSE mismatch {row['task_id']}/{row['method']}", "loss_checks")
    except Exception as exc:
        failures.append(f"Cannot audit task_metrics.csv: {exc}")
    notes.extend([
        "Saved losses and recorded selection are reconstructed; individual candidate models are not independently refitted.",
        "Hash agreement verifies supplied evidence consistency, not independent predictive replication.",
        "This is an exploratory known-law pilot; no population confidence intervals or new-source claims are justified.",
    ])
    report = {"passed": not failures, "mode": protocol["mode"],
              "study_status": protocol["study_status"], "not_confirmatory": True,
              "task_count": len(protocol["tasks"]), "complete_task_count": complete_tasks,
              "available_forecasts": available_forecasts,
              "expected_forecasts": len(protocol["tasks"]) * len(methods),
              "recorded_fallback_events": fallback_count,
              "counts": counts, "failures": failures, "notes": notes}
    (output / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit(args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
