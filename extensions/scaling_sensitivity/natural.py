"""Paired, post-hoc sensitivity of the frozen natural forecasting pipelines.

Only the geometry input scaler changes. The raw skip/reference, feature and
target standardizers, stable KAHM query, seeds, training caps, alpha grid and
Nyström construction retain the original definitions. Standard arms delegate
their entire fit system to the frozen implementation. Min--max arms replace
the geometry scaler and keep the raw skip independently standardized.

This is sensitivity on already inspected sources, not new confirmation.
Every method runs in its own child, with one active child and one BLAS thread.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import datetime as dt
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import resource
import sys
import time
import traceback
import warnings

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import numpy as np
from threadpoolctl import threadpool_limits

from extensions.natural_forecasting import models as original_models
from extensions.natural_forecasting import run as original_run
from extensions.predictive_geometry.study import geometry as original_geometry
from extensions.predictive_geometry.study import heads as original_heads

ROOT = Path(__file__).resolve().parent
ORIGINAL_RUN = REPOSITORY / "extensions/natural_forecasting/runs/natural_v1"
DEFAULT_OUTPUT = ROOT / "results/natural"
GEOMETRIC_METHODS = ("raw_affine_rbf", "raw_locality_rbf", "raw_kahm_rbf")
SCALERS = ("standard", "minmax")
METHODS = (("raw_ridge", "standard"), ("raw_rbf", "standard")) + tuple(
    (method, scaling) for method in GEOMETRIC_METHODS for scaling in SCALERS)


def identifier(method, scaling):
    return method if method in ("raw_ridge", "raw_rbf") else f"{method}__{scaling}"


class MinMaxLocalFeatureBank(original_models.LocalFeatureBank):
    """Frozen local bank with a training-only, unclipped geometry scaler.

    Columns with exactly zero training range map to zero for every geometry
query, matching the earlier controlled min--max convention. The separate
raw skip preserves its original standardization, including those columns.
"""

    def fit(self, x):
        x = original_models._matrix(x)
        self.minimum_ = x.min(axis=0)
        self.maximum_ = x.max(axis=0)
        self.range_ = self.maximum_ - self.minimum_
        self.constant_ = self.range_ == 0.0
        result = super().fit(x)
        # These fields describe the effective geometry coordinates for metadata;
        # scale_inputs uses the explicit min--max formula below throughout fit.
        self.mean_ = self.minimum_ + 0.5 * self.range_
        self.scale_ = np.where(self.constant_, 1.0, 0.5 * self.range_)
        return result

    def scale_inputs(self, x):
        if not hasattr(self, "minimum_"):
            raise RuntimeError("Fit the bank before transforming")
        x = original_models._matrix(x)
        if x.shape[1] != len(self.minimum_):
            raise ValueError("Query dimension differs from fitted inputs")
        z = 2.0 * (x - self.minimum_) / np.where(self.constant_, 1.0, self.range_) - 1.0
        z[:, self.constant_] = 0.0
        return z

    def metadata(self):
        value = super().metadata()
        value.update(scaling="minmax", clip=False, training_minimum=self.minimum_.tolist(),
                     training_maximum=self.maximum_.tolist(),
                     constant_columns=self.constant_.tolist(), constant_range_tolerance=0.0,
                     constant_column_rule="zero for every geometry query")
        return value


class ForecastRegressor(original_models.ForecastRegressor):
    """Sensitivity adapter; original standard fits remain the exact original path."""

    def __init__(self, *args, geometry_scaling="standard", **kwargs):
        if geometry_scaling not in SCALERS:
            raise ValueError("Only the two frozen scaler arms are permitted")
        self.geometry_scaling = geometry_scaling
        super().__init__(*args, **kwargs)
        if self.representation == "raw" and geometry_scaling != "standard":
            raise ValueError("Raw references keep their frozen standardization")

    def _fit_system(self, x, y):
        if self.geometry_scaling == "standard":
            return super()._fit_system(x, y)
        # This is the original _fit_system with exactly two scientific changes:
        # MinMaxLocalFeatureBank, and raw skip scales independent of the bank.
        x = original_models._matrix(x)
        y = np.asarray(y, dtype=float)
        self.single_target_ = y.ndim == 1
        if self.single_target_:
            y = y[:, None]
        y = original_models._matrix(y, name="target")
        if len(x) != len(y):
            raise ValueError("Training inputs and targets have different row counts")
        self.n_training_rows_, self.n_features_in_ = x.shape
        self.n_targets_ = y.shape[1]
        self.target_mean_, self.target_scale_ = original_models._standardizer(y, 1e-12)
        standardized_target = (y - self.target_mean_) / self.target_scale_
        gate = {"affine_local": "affine", "locality_local": "locality", "kahm_local": "kahm"}[self.representation]
        self.bank_ = MinMaxLocalFeatureBank(gate, seed=self.seed, query_mode=self.query_mode,
                                          **self.geometry_parameters).fit(x)
        self.input_mean_, self.input_scale_ = original_models._standardizer(x, 1e-12)
        features, self.geometry_diagnostics_ = self.bank_.transform(x, diagnostics=True)
        self.feature_names_ = ([f"gate_{j}" for j in range(self.bank_.n_gates_)]
                               + [f"local_{j}_{k}" for j, model in enumerate(self.bank_.models_)
                                  for k in range(model.basis.shape[1])])
        if self.raw_skip:
            features = np.column_stack(((x - self.input_mean_) / self.input_scale_, features))
            self.feature_names_ = [f"raw_{j}" for j in range(x.shape[1])] + self.feature_names_
        self.feature_width_ = features.shape[1]
        self.basis_ = original_heads._bases(features, self.head, self.head_seed,
                                           self.n_landmarks, (self.bandwidth_multiplier,))[0]
        basis_values = self.basis_.transform(features)
        self.basis_mean_, self.basis_scale_ = original_heads.fit_standardizer(basis_values)
        design = (basis_values - self.basis_mean_) / self.basis_scale_
        self.standardized_target_mean_ = standardized_target.mean(axis=0)
        centered_target = standardized_target - self.standardized_target_mean_
        gram = (design.T @ design) / len(design)
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        rhs = eigenvectors.T @ ((design.T @ centered_target) / len(design))
        self.basis_width_ = design.shape[1]
        self.retained_nystrom_rank_ = (0 if self.basis_.inverse_sqrt is None
                                      else self.basis_.inverse_sqrt.shape[1])
        self.diagnostic_features_ = {
            "training_rows": float(len(x)), "input_dimension": float(x.shape[1]),
            "feature_dimension": float(self.feature_width_),
            "retained_basis_dimension": float(self.basis_width_),
            **self.geometry_diagnostics_["diagnostic_features"],
        }
        return eigenvalues, eigenvectors, rhs

    def metadata(self):
        value = super().metadata()
        value.update(geometry_scaling=self.geometry_scaling,
                     raw_skip_scaling="standard" if self.raw_skip else None,
                     post_hoc_sensitivity=True)
        if value["geometry"] is not None:
            value["geometry"].setdefault("scaling", self.geometry_scaling)
            value["geometry"].setdefault("clip", False)
        return value


def scaler_snapshot(model, fitting_x, query_x):
    """Audit-only state and out-of-range diagnostics; never fits a predictor."""
    minimum, maximum = fitting_x.min(axis=0), fitting_x.max(axis=0)
    constant = (maximum - minimum) == 0.0
    outside = (query_x < minimum) | (query_x > maximum)
    minmax_z = 2.0 * (query_x - minimum) / np.where(constant, 1.0, maximum - minimum) - 1.0
    minmax_z[:, constant] = 0.0
    bank = model.bank_
    state = {
        "mode": model.geometry_scaling if bank is not None else "standard",
        "training_rows": len(fitting_x),
        "training_input_sha256": original_geometry._array_hash(fitting_x),
        "training_minimum": minimum.tolist(), "training_maximum": maximum.tolist(),
        "constant_columns": constant.tolist(), "constant_range_tolerance": 0.0,
        "effective_center": (model.input_mean_ if bank is None else bank.mean_).tolist(),
        "effective_scale": (model.input_scale_ if bank is None else bank.scale_).tolist(),
        "clip": False,
        "raw_input_mean": model.input_mean_.tolist(),
        "raw_input_scale": model.input_scale_.tolist(),
        "target_mean": model.target_mean_.tolist(), "target_scale": model.target_scale_.tolist(),
        "geometry_model_effective_ranks": [] if bank is None else [m.basis.shape[1] for m in bank.models_],
    }
    diagnostic = {
        "query_rows": len(query_x), "query_coordinates": int(query_x.size),
        "outside_training_range_coordinates": int(outside.sum()),
        "outside_training_range_coordinate_fraction": float(outside.mean()),
        "outside_training_range_rows": int(outside.any(axis=1).sum()),
        "outside_training_range_row_fraction": float(outside.any(axis=1).mean()),
        "maximum_abs_minmax_coordinate": float(np.abs(minmax_z).max()),
        "scope": "Range relative to the fitting input histories; no clipping is applied.",
    }
    return state, diagnostic


def construct(method, scaling, alpha, task, protocol):
    representation, head, raw_skip = original_run.GEOMETRIC[method]
    return ForecastRegressor(representation=representation, head=head, raw_skip=raw_skip,
        alpha=alpha, n_landmarks=task["capacity_regime"]["n_landmarks"],
        bandwidth_multiplier=1.0, seed=task["seed"], head_seed=task["seed"] + 1000,
        geometry_parameters=protocol["geometry"], query_mode=protocol["query_mode"],
        geometry_scaling=scaling)


def run_method(task, arrays, method, scaling, protocol):
    """Complete standalone search/refit/forecast with separately charged audits."""
    candidate_id = identifier(method, scaling)
    nodes, all_warnings = [], []
    selection_started, selection_cpu = time.perf_counter(), time.process_time()
    with original_run.measured(nodes, candidate_id + ":search", "geometry_features_and_shared_alpha_search",
                               alphas=protocol["head_search"]["alphas"]):
        model = construct(method, scaling, protocol["head_search"]["alphas"][0], task, protocol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trials = model.fit_select(arrays["X_train"], arrays["Y_train"],
                arrays["X_validation"], arrays["Y_validation"], protocol["head_search"]["alphas"])
        all_warnings.extend(str(w.message) for w in caught)
        winner, best = float(model.alpha), min(float(t["validation_mse"]) for t in trials)
    audit_started, audit_cpu = time.perf_counter(), time.process_time()
    search_scaler, validation_outside = scaler_snapshot(model, arrays["X_train"], arrays["X_validation"])
    search_audit_seconds, search_audit_cpu = time.perf_counter() - audit_started, time.process_time() - audit_cpu
    del model
    gc.collect()
    nodes.append({"id": candidate_id + ":selection_overhead", "kind": "selection_bookkeeping_and_cleanup",
        "seconds": max(0.0, time.perf_counter() - selection_started - nodes[0]["seconds"] - search_audit_seconds),
        "process_cpu_seconds": max(0.0, time.process_time() - selection_cpu - nodes[0]["process_cpu_seconds"] - search_audit_cpu),
        "peak_rss_bytes": original_run.peak_rss_bytes()})
    with original_run.measured(nodes, candidate_id + ":refit", "selected_final_refit"):
        model = construct(method, scaling, winner, task, protocol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(arrays["X_refit"], arrays["Y_refit"])
        all_warnings.extend(str(w.message) for w in caught)
    with original_run.measured(nodes, candidate_id + ":forecast", "forecast_batch", n_origins=len(arrays["X_test"])):
        prediction = model.predict(arrays["X_test"])
    loss = original_run.finite_loss(prediction, arrays["Y_test"])
    with original_run.measured(nodes, candidate_id + ":serialization", "model_serialization_and_metadata"):
        metadata = model.metadata()
        serialized = pickle.dumps(model, protocol=5)
    audit_started, audit_cpu = time.perf_counter(), time.process_time()
    refit_scaler, test_outside = scaler_snapshot(model, arrays["X_refit"], arrays["X_test"])
    refit_audit_seconds, refit_audit_cpu = time.perf_counter() - audit_started, time.process_time() - audit_cpu
    candidate = {
        "candidate_id": candidate_id, "base_method": method, "geometry_scaling": scaling,
        "status": "ok", "selected_parameters": {"alpha": winner}, "validation_loss": best,
        "test_loss": loss, "trials": trials,
        "cost_nodes": nodes, "cost_seconds": sum(n["seconds"] for n in nodes),
        "cost_process_cpu_seconds": sum(n["process_cpu_seconds"] for n in nodes),
        "cost_scope": "Warm-library standalone alpha search, selection cleanup, selected pretest refit, forecast and serialization; source preparation/load, process launch and additional sensitivity audits reported separately.",
        "search_audit_seconds": search_audit_seconds, "refit_audit_seconds": refit_audit_seconds,
        "audit_diagnostic_seconds": search_audit_seconds + refit_audit_seconds,
        "audit_diagnostic_process_cpu_seconds": search_audit_cpu + refit_audit_cpu,
        "serialized_model_bytes": len(serialized), "model_sha256": hashlib.sha256(serialized).hexdigest(),
        "model_metadata": metadata, "search_scaler_metadata": search_scaler,
        "refit_scaler_metadata": refit_scaler, "validation_out_of_range": validation_outside,
        "test_out_of_range": test_outside, "warnings": all_warnings,
        "tuning_fit_max_target_index": int((arrays["origin_train"] + task["horizon"]).max()),
        "selection_max_target_index": int((arrays["origin_validation"] + task["horizon"]).max()),
        "refit_max_target_index": int((arrays["origin_refit"] + task["horizon"]).max()),
        "test_min_target_index": int((arrays["origin_test"] + 1).min()),
        "test_max_target_index": int((arrays["origin_test"] + task["horizon"]).max()),
    }
    return {"candidate": candidate, "prediction": prediction,
            "origin_losses": np.mean((prediction - arrays["Y_test"]) ** 2, axis=1)}


def child_entry(connection, task, arrays, method, scaling, protocol):
    started, cpu = time.perf_counter(), time.process_time()
    try:
        with threadpool_limits(limits=1):
            result = run_method(task, arrays, method, scaling, protocol)
        result["candidate"].update(child_wall_seconds=time.perf_counter() - started,
            child_cpu_seconds=time.process_time() - cpu,
            process_peak_rss_bytes=original_run.peak_rss_bytes())
        connection.send({"ok": True, "result": result})
    except BaseException as error:
        connection.send({"ok": False, "error": repr(error), "traceback": traceback.format_exc(),
            "child_wall_seconds": time.perf_counter() - started,
            "child_cpu_seconds": time.process_time() - cpu,
            "process_peak_rss_bytes": original_run.peak_rss_bytes()})
    finally:
        connection.close()


def isolated(task, arrays, method, scaling, protocol):
    context = mp.get_context("fork" if sys.platform == "linux" else "spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=child_entry, args=(child, task, arrays, method, scaling, protocol))
    started = time.perf_counter()
    process.start()
    child.close()
    timeout = protocol["resources"]["per_method_timeout_seconds"]
    if parent.poll(timeout):
        try:
            result = parent.recv()
        except EOFError:
            result = {"ok": False, "error": "worker exited without result"}
        process.join(5)
    else:
        process.terminate()
        process.join(5)
        result = {"ok": False, "error": f"timeout after {timeout} seconds", "timeout": True}
    if process.is_alive():
        process.kill()
        process.join(5)
    result["launch_to_result_seconds"] = time.perf_counter() - started
    result["exit_code"] = process.exitcode
    parent.close()
    return result


def archived_parity(task, method, candidate, prediction):
    name = hashlib.sha256(task["task_id"].encode()).hexdigest()[:24]
    path = ORIGINAL_RUN / task["role"] / (name + ".json")
    record = json.loads(path.read_text())
    old = next(row for row in record["candidates"] if row["candidate_id"] == method)
    artifact = path.parent / record["predictions_file"]
    if original_run.digest_file(artifact) != record["predictions_sha256"]:
        raise RuntimeError("Original prediction archive failed checksum")
    with np.load(artifact, allow_pickle=False) as arrays:
        archived = arrays[method]
    maximum = float(np.max(np.abs(prediction - archived)))
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(archived))))
    same_alpha = candidate["selected_parameters"] == old["selected_parameters"]
    value = {"maximum_abs_prediction_difference": maximum,
        "prediction_tolerance": tolerance, "predictions_within_tolerance": maximum <= tolerance,
        "predictions_exactly_equal": bool(np.array_equal(prediction, archived)),
        "selected_parameters_equal": same_alpha,
        "archived_test_loss": old["test_loss"],
        "test_loss_abs_difference": abs(candidate["test_loss"] - old["test_loss"]),
        "archive_json": str(path.relative_to(REPOSITORY)),
        "archive_json_sha256": original_run.digest_file(path),
        "archive_predictions_sha256": record["predictions_sha256"]}
    if maximum > tolerance or not same_alpha:
        raise RuntimeError(f"Standard-arm parity failed: {method}: {value}")
    return value


def run(output=DEFAULT_OUTPUT):
    from extensions.scaling_sensitivity.common import verify_freeze
    freeze = verify_freeze()
    output = Path(output)
    protocol = original_run.load_protocol()
    roster_file = ORIGINAL_RUN / "task_roster.json"
    roster = json.loads(roster_file.read_text())
    tasks = roster["tasks"]
    if len(tasks) != 208 or any(not task["supported"] for task in tasks):
        raise RuntimeError("The bounded study requires all 208 original supported tasks")
    execution_path = output / "execution_environment.json"
    execution = {"created_utc": original_run.now(), "environment": original_run.environment(),
        "protocol_freeze_sha256": original_run.digest_file(ROOT / "protocol_freeze.json"),
        "source_roster_sha256": original_run.digest_file(roster_file),
        "task_count": len(tasks), "candidate_count_per_task": len(METHODS),
        "scope": "Post-hoc sensitivity on all previously inspected development and confirmation sources; no diagnostic selector refit.",
        "execution": "One fresh child per method; one active method; one BLAS thread. Task/method order is independent of outcomes."}
    if execution_path.exists():
        saved = json.loads(execution_path.read_text())
        for key in ("protocol_freeze_sha256", "source_roster_sha256"):
            if saved[key] != execution[key]:
                raise RuntimeError("Cannot resume a changed frozen execution")
    else:
        original_run.write_json(execution_path, execution)
    for index, task in enumerate(tasks):
        verify_freeze()
        name = hashlib.sha256(task["task_id"].encode()).hexdigest()[:24]
        path = output / task["role"] / (name + ".json")
        artifact = path.with_suffix(".npz")
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["status"] == "complete" and saved["protocol_freeze_sha256"] == execution["protocol_freeze_sha256"]
                    and saved["predictions_sha256"] == original_run.digest_file(artifact)):
                continue
            raise RuntimeError(f"Existing incomplete or changed result requires audit, not replacement: {path}")
        task_start = time.perf_counter()
        load_start, load_cpu = time.perf_counter(), time.process_time()
        arrays = original_run.load_arrays(ORIGINAL_RUN, task)
        load_seconds, load_cpu_seconds = time.perf_counter() - load_start, time.process_time() - load_cpu
        order = list(METHODS)
        entropy = int(hashlib.sha256(("scaling_sensitivity_v1|" + task["task_id"]).encode()).hexdigest()[:8], 16)
        np.random.default_rng(entropy).shuffle(order)
        record = {"task_id": task["task_id"], "phase": task["role"],
            "source_family": task["family_id"], "dataset_id": task["source_id"],
            "series_id": task["series_id"], "horizon": task["horizon"], "history_length": task["context"],
            "budget_id": task["budget_id"], "seed": task["seed"], "loss_scale": task["loss_scale"],
            "source_task_cache_path": str((ORIGINAL_RUN / task["cache_path"]).relative_to(REPOSITORY)),
            "source_task_cache_sha256": task["cache_sha256"],
            "protocol_freeze_sha256": execution["protocol_freeze_sha256"],
            "execution_order": [identifier(*entry) for entry in order],
            "data_load_seconds": load_seconds, "data_load_cpu_seconds": load_cpu_seconds,
            "historical_data_preparation_seconds": task["data_preparation_seconds"],
            "data_cost_scope": "Original prepared arrays reused without retiming source preparation; current hash/load measured separately. Historical preparation allocation retained only as provenance.",
            "candidates": [], "failures": []}
        predictions = {"target": arrays["Y_test"], "origin": arrays["origin_test"], "loss_scale": np.array(task["loss_scale"])}
        for method, scaling in order:
            candidate_id = identifier(method, scaling)
            result = isolated(task, arrays, method, scaling, protocol)
            if not result["ok"]:
                record["failures"].append({"candidate_id": candidate_id, **result})
                break
            payload = result["result"]
            candidate = payload["candidate"]
            candidate["launch_to_result_seconds"] = result["launch_to_result_seconds"]
            if scaling == "standard":
                try:
                    candidate["archived_parity"] = archived_parity(task, method, candidate, payload["prediction"])
                except Exception as error:
                    record["failures"].append({"candidate_id": candidate_id, "error": repr(error)})
            record["candidates"].append(candidate)
            predictions[candidate_id] = payload["prediction"]
            predictions[candidate_id + "__origin_losses"] = payload["origin_losses"]
            if record["failures"]:
                break
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, **predictions)
        record.update(predictions_file=artifact.name, predictions_sha256=original_run.digest_file(artifact),
            task_elapsed_seconds=time.perf_counter() - task_start,
            status="complete" if not record["failures"] and len(record["candidates"]) == len(METHODS) else "incomplete")
        original_run.write_json(path, record)
        print(json.dumps({"event": "task_complete", "index": index + 1, "total": len(tasks),
            "task_id": task["task_id"], "status": record["status"],
            "seconds": record["task_elapsed_seconds"], "failures": len(record["failures"])}), flush=True)
        if record["status"] != "complete":
            raise RuntimeError(f"Incomplete task retained for audit: {path}")
    verify_freeze()
    return {"tasks": len(tasks), "method_runs": len(tasks) * len(METHODS), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output)), flush=True)


if __name__ == "__main__":
    main()
