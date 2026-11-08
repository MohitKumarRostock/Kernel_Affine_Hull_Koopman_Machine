"""Post-hoc VDP numerical robustness check on the completed phase-4 samples.

The only geometry change is an algebraically invariant row rescaling of the
Gaussian query kernel BEFORE multiplying by the unchanged AE inverse B.  All
training AEs, local references, PCA bases, partitions, and input scalers remain
frozen.  Heads are refit on stabilized training features and retuned on the same
validation split.  The existing phase-4 test observations are reused, so this
is a descriptive sensitivity analysis, not a new confirmation experiment.

No frozen source is edited.  A process-local context manager restores the
original query routine after every stabilized transform, including exceptions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

import geometry
from datasets import make_forecast_split
from followup_capacity import verify_capacity_freeze
from geometry import GeometryBank
from heads import fit_predict, grouped_losses


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "numerical_sensitivity_config.json"
FREEZE_FILE = ROOT / "numerical_sensitivity_freeze.json"


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes()).hexdigest()


def _reconstruct_log_rows(log_kernel, ae):
    """Normalize a common row factor before the unchanged signed-weight map."""
    maximum = log_kernel.max(axis=1, keepdims=True)
    kernel = np.exp(log_kernel - maximum)
    weights = kernel @ ae["B"]
    normalizers = weights.sum(axis=1, keepdims=True)
    safe = normalizers.copy()
    safe[safe == 0] = 1.0  # Preserve the upstream exact-zero convention.
    normalized = weights / safe
    reconstructed = (ae["y_data"] @ normalized.T).T
    return reconstructed, normalized, normalizers, maximum


def stabilized_kahm_reconstruction(Z, ae, collect_diagnostics=False):
    """Upstream arithmetic except exp(ell) is replaced by exp(ell-max(ell)).

    The upstream squared-distance expression is used exactly, without clipping
    or changing its covariance metric.  Positive row factors cancel in the
    subsequent normalization in exact arithmetic.  This prevents all Gaussian
    kernel entries underflowing before that cancellation can occur.
    """
    distance_matrix = geometry._upstream("distance_matrix").distance_matrix
    query = np.asarray(Z, dtype=float).T
    encoded = ae["PC"].T @ query
    precision_sqrt = np.asarray(ae["sqrt_weights_matrix"], dtype=float)
    encoded_references = np.asarray(ae["y_data_n_subspace"], dtype=float)
    n, rank = query.shape[1], encoded.shape[0]
    reconstructed = np.zeros((query.shape[0], n), dtype=float)
    stats = {
        "weight_rows": 0, "weight_entries": 0, "zero_normalizers": 0,
        "negative_weight_entries": 0, "rows_with_negative_weights": 0,
        "normalizer_min": float("inf"), "normalizer_abs_min": float("inf"),
        "normalizer_max": float("-inf"), "weight_l1_sum": 0.0, "weight_l1_max": 0.0,
        "original_all_zero_kernel_rows": 0,
        "kernel_log_row_max_min": float("inf"), "kernel_log_row_max_max": float("-inf"),
    }
    for start in range(0, n, 1000):
        stop = min(n, start + 1000)
        # This is the exact expression in the pinned upstream kxa_matrix.
        transformed_query = precision_sqrt @ encoded[:, start:stop]
        transformed_references = precision_sqrt @ encoded_references
        distances = distance_matrix(transformed_query.T, transformed_references.T)
        log_kernel = -(0.5 / rank) * distances
        if not np.isfinite(log_kernel).all():
            raise FloatingPointError("Nonfinite Gaussian log kernel; no substitute geometry is permitted")
        estimate, weights, normalizers, row_maximum = _reconstruct_log_rows(log_kernel, ae)
        if not np.isfinite(estimate).all() or not np.isfinite(weights).all():
            raise FloatingPointError("Nonfinite row-rescaled reconstruction")
        reconstructed[:, start:stop] = estimate.T
        if collect_diagnostics:
            l1 = np.abs(weights).sum(axis=1)
            stats["weight_rows"] += len(weights)
            stats["weight_entries"] += weights.size
            stats["zero_normalizers"] += int(np.count_nonzero(normalizers == 0))
            stats["negative_weight_entries"] += int(np.count_nonzero(weights < 0))
            stats["rows_with_negative_weights"] += int(np.count_nonzero(np.any(weights < 0, axis=1)))
            stats["normalizer_min"] = min(stats["normalizer_min"], float(normalizers.min()))
            stats["normalizer_abs_min"] = min(stats["normalizer_abs_min"], float(np.abs(normalizers).min()))
            stats["normalizer_max"] = max(stats["normalizer_max"], float(normalizers.max()))
            stats["weight_l1_sum"] += float(l1.sum())
            stats["weight_l1_max"] = max(stats["weight_l1_max"], float(l1.max()))
            with np.errstate(under="ignore"):
                stats["original_all_zero_kernel_rows"] += int(np.count_nonzero(np.exp(row_maximum) == 0))
            stats["kernel_log_row_max_min"] = min(stats["kernel_log_row_max_min"], float(row_maximum.min()))
            stats["kernel_log_row_max_max"] = max(stats["kernel_log_row_max_max"], float(row_maximum.max()))
    return reconstructed.T, stats if collect_diagnostics else None


@contextmanager
def stabilized_queries():
    """Temporary process-local substitution; never mutate a fitted bank or file."""
    original = geometry._kahm_reconstruction
    calls = []

    def instrumented(Z, ae, collect_diagnostics=False):
        result, diagnostics = stabilized_kahm_reconstruction(Z, ae, collect_diagnostics)
        if diagnostics is not None:
            calls.append(dict(diagnostics))
        return result, diagnostics

    geometry._kahm_reconstruction = instrumented
    try:
        yield calls
    finally:
        geometry._kahm_reconstruction = original


def self_tests():
    """Numerical checks on artificial inputs unrelated to the study streams."""
    rng = np.random.default_rng(991731)
    reference = rng.normal(size=(4, 36))
    ae = geometry._upstream("autoencoder").autoencoder(reference, 2)
    query = rng.normal(scale=.7, size=(1007, 4))
    original, original_stats = geometry._kahm_reconstruction(query, ae, True)
    corrected, corrected_stats = stabilized_kahm_reconstruction(query, ae, True)
    assert original_stats["zero_normalizers"] == 0
    np.testing.assert_allclose(corrected, original, rtol=1e-9, atol=1e-11)
    encoded = ae["PC"].T @ query[:31].T
    S = ae["sqrt_weights_matrix"]
    distances = geometry._upstream("distance_matrix").distance_matrix(
        (S @ encoded).T, (S @ ae["y_data_n_subspace"]).T)
    logs = -(0.5 / encoded.shape[0]) * distances
    first = _reconstruct_log_rows(logs, ae)[0]
    offsets = rng.uniform(-100, 100, size=(len(logs), 1))
    second = _reconstruct_log_rows(logs + offsets, ae)[0]
    np.testing.assert_allclose(first, second, rtol=1e-9, atol=1e-11)
    far = np.broadcast_to(reference.mean(axis=1) + 1000 * ae["PC"][:, 0], (9, 4)).copy()
    _, far_original_stats = geometry._kahm_reconstruction(far, ae, True)
    far_corrected, far_stats = stabilized_kahm_reconstruction(far, ae, True)
    assert far_original_stats["zero_normalizers"] == 9 and far_stats["zero_normalizers"] == 0
    assert np.isfinite(far_corrected).all()
    original_function = geometry._kahm_reconstruction
    try:
        with stabilized_queries():
            assert geometry._kahm_reconstruction is not original_function
            raise RuntimeError("deliberate restoration test")
    except RuntimeError:
        pass
    assert geometry._kahm_reconstruction is original_function
    return {
        "status": "pass", "relative_tolerance": 1e-9, "absolute_tolerance": 1e-11,
        "nonunderflow_maximum_absolute_reconstruction_difference": float(np.max(np.abs(original-corrected))),
        "common_log_row_offset_maximum_absolute_difference": float(np.max(np.abs(first-second))),
        "artificial_underflow_zero_normalizers_before": far_original_stats["zero_normalizers"],
        "artificial_underflow_zero_normalizers_after": far_stats["zero_normalizers"],
        "context_restored_after_exception": True,
        "scope": "artificial arithmetic checks; no forecasting performance evaluated",
    }


def _load_config():
    config = json.loads(CONFIG_FILE.read_text())
    if (config.get("phase") != "posthoc_numerical_sensitivity"
            or config.get("system") != "vanderpol" or config.get("repetitions") != 24
            or tuple(config.get("features", [])) != ("kahm:local", "kahm:scores")
            or tuple(config.get("landmark_counts", [])) != (64, 256)):
        raise ValueError("Sensitivity configuration differs from the declared all-24 VDP check")
    return config


def _source_directory(rep):
    return ROOT / "results_capacity" / f"r{rep:02d}_forecast_van_der_pol"


def freeze_plan():
    """Record the exact robustness plan before evaluating stabilized study features."""
    if FREEZE_FILE.exists():
        raise RuntimeError("A numerical sensitivity freeze already exists; do not overwrite it")
    verify_capacity_freeze()
    config = _load_config()
    arithmetic_checks = self_tests()
    source_paths = ["numerical_sensitivity.py", "numerical_sensitivity_config.json",
                    "followup_freeze.json", "protocol_freeze.json", "config.json", "followup_config.json",
                    "geometry.py", "heads.py", "datasets.py", "followup_capacity.py",
                    "vendor/upstream/distance_matrix.py", "vendor/upstream/kxa_matrix.py",
                    "vendor/upstream/autoencoder_filtering_extended.py"]
    inputs = {}
    for rep in range(config["repetitions"]):
        directory = _source_directory(rep)
        source = json.loads((directory / "result.json").read_text())
        if source.get("status") != "complete":
            raise RuntimeError(f"Source capacity job is incomplete: {directory.name}")
        for name in ("result.json", "bank.pkl", "unit_losses.npz"):
            path = directory / name
            inputs[str(path.relative_to(ROOT))] = _hash_file(path)
    frozen = {
        "version": 1, "created_system_utc": datetime.now(timezone.utc).isoformat(),
        "type": "internal post-hoc numerical robustness plan; reused stage2 test data, not new confirmation",
        "stabilized_study_jobs_completed_at_freeze": len(list((ROOT / "results_numerical_sensitivity").glob("*/result.json"))),
        "sha256": {name: _hash_file(ROOT / name) for name in source_paths},
        "input_sha256": inputs, "arithmetic_checks": arithmetic_checks,
    }
    if frozen["stabilized_study_jobs_completed_at_freeze"] != 0:
        raise RuntimeError("Cannot claim a pre-evaluation freeze after sensitivity results exist")
    _write_json(FREEZE_FILE, frozen)
    return frozen


def verify_sensitivity_freeze(rep=None):
    verify_capacity_freeze()
    if not FREEZE_FILE.is_file():
        raise RuntimeError("Create the recorded numerical-sensitivity freeze before study evaluation")
    frozen = json.loads(FREEZE_FILE.read_text())
    for name, expected in frozen["sha256"].items():
        if _hash_file(ROOT / name) != expected:
            raise RuntimeError(f"Frozen sensitivity dependency changed: {name}")
    for name, expected in frozen["input_sha256"].items():
        if rep is not None and _source_directory(rep).name not in name:
            continue
        if _hash_file(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source result changed: {name}")
    return frozen


def _signature(frozen, rep):
    inputs = {key: value for key, value in frozen["input_sha256"].items() if _source_directory(rep).name in key}
    payload = {"freeze_sha256": _hash_file(FREEZE_FILE), "source_sha256": frozen["sha256"], "input_sha256": inputs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _call_summary(calls):
    return {
        "calls": len(calls), "query_cluster_pairs": sum(s["weight_rows"] for s in calls),
        "original_all_zero_kernel_rows": sum(s["original_all_zero_kernel_rows"] for s in calls),
        "remaining_zero_normalizers": sum(s["zero_normalizers"] for s in calls),
        "kernel_log_row_max_min": min(s["kernel_log_row_max_min"] for s in calls),
        "kernel_log_row_max_max": max(s["kernel_log_row_max_max"] for s in calls),
        "normalizer_scope": "normalizer after algebraically cancelling a positive kernel-row factor",
    }


def run_job(rep, config):
    started = time.perf_counter()
    job_id = f"r{rep:02d}_forecast_van_der_pol"
    destination = ROOT / "results_numerical_sensitivity" / job_id
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.json"
    try:
        frozen = verify_sensitivity_freeze(rep)
        signature, provenance = _signature(frozen, rep)
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("status") == "complete":
                if previous.get("execution_signature") != signature:
                    raise RuntimeError("Refusing stale completed numerical sensitivity result")
                return {"job_id": job_id, "status": "already_complete"}
        source_directory = _source_directory(rep)
        source = json.loads((source_directory / "result.json").read_text())
        base = json.loads((ROOT / "config.json").read_text())
        fc = base["forecast"]
        splits = [make_forecast_split("vanderpol", fc[name+"_episodes"], source["seeds"][name],
                                      history=fc["history"], horizons=tuple(fc["horizons"]), stride=fc["stride"])
                  for name in ("train", "validation", "test")]
        for split, expected in zip(splits, source["split_metadata"]):
            if _array_hash(split["X"]) != expected["X_sha256"] or _array_hash(split["Y"]) != expected["Y_sha256"]:
                raise RuntimeError("Regenerated observations differ from the frozen phase4 split")
        train, validation, test = splits
        with np.load(source_directory / "unit_losses.npz") as archive:
            np.testing.assert_array_equal(test["Y"], archive["Y"])
            np.testing.assert_array_equal(test["groups"], archive["groups"])
        bank = GeometryBank.load(source_directory / "bank.pkl")
        before_state = pickle.dumps(bank.state_dict(), protocol=5)
        stabilized_features = []
        split_diagnostics = []
        for name, split in zip(("train", "validation", "test"), splits):
            original_features = bank.transform(split["X"], diagnostics=True)
            original_diagnostics = bank.last_diagnostics_
            with stabilized_queries() as calls:
                changed = bank.transform(split["X"], diagnostics=True)
            corrected_diagnostics = bank.last_diagnostics_
            differences = {}
            for feature in ("kahm:phi", "kahm:phi_logz", "kahm:scores", "kahm:local"):
                difference = changed[feature] - original_features[feature]
                norm = float(np.linalg.norm(original_features[feature]))
                differences[feature] = {
                    "maximum_absolute": float(np.max(np.abs(difference))),
                    "coordinate_rms": float(np.sqrt(np.mean(difference**2))),
                    "relative_frobenius": float(np.linalg.norm(difference) / norm) if norm > 0 else None,
                }
            # Unchanged controls are a direct isolation check.
            for control in ("raw", "affine:phi", "affine:local", "centroid:phi"):
                np.testing.assert_array_equal(changed[control], original_features[control])
            split_diagnostics.append({
                "split": name, "n_rows": len(split["X"]),
                "original_weight_diagnostics": original_diagnostics["weight_diagnostics"],
                "stabilized_weight_diagnostics": corrected_diagnostics["weight_diagnostics"],
                "kernel_rescaling": _call_summary(calls), "feature_differences": differences,
                "unchanged_controls_array_equal": True,
            })
            stabilized_features.append({feature: changed[feature] for feature in config["features"]})
            del original_features, changed
        if pickle.dumps(bank.state_dict(), protocol=5) != before_state:
            raise RuntimeError("Numerical sensitivity mutated the frozen bank")
        head_config = dict(base["heads"])
        head_config.pop("n_landmarks", None)
        original_rows = {(row["feature"], row["landmarks"]): row for row in source["records"] if row["head"] == "rbf"}
        arrays = {"Y": test["Y"], "groups": test["groups"]}
        records = []
        for landmarks in config["landmark_counts"]:
            for feature in config["features"]:
                tick = time.perf_counter()
                prediction, selected, fitted = fit_predict(
                    stabilized_features[0][feature], train["Y"], stabilized_features[1][feature], validation["Y"],
                    stabilized_features[2][feature], head="rbf", seed=source["seeds"]["head"],
                    bounds=train["bounds"], separate_targets=False, validation_groups=validation["groups"],
                    n_landmarks=landmarks, **head_config)
                losses = grouped_losses(prediction, test["Y"], test["groups"])
                suffix = feature.replace(":", "__") + f"__m{landmarks}"
                loss_key, prediction_key = "loss__"+suffix, "prediction__"+suffix
                arrays[loss_key], arrays[prediction_key] = losses, prediction
                original = original_rows[feature, landmarks]
                raw = original_rows["raw", landmarks]
                affine = original_rows["affine:local", landmarks]
                mse = float(losses.mean())
                records.append({
                    "phase": config["phase"], "replicate": rep, "setting_id": "forecast_van_der_pol",
                    "variant": "log_kernel_row_max_rescaling", "feature": feature, "head": "rbf",
                    "landmarks": landmarks, "test_mse": mse, "test_mse_by_target": losses.mean(axis=0).tolist(),
                    "validation_mse": float(np.mean([s["validation_mse_by_target"] for s in selected])),
                    "selected": selected, "original_test_mse": original["test_mse"],
                    "raw_baseline_mse": raw["test_mse"], "affine_local_baseline_mse": affine["test_mse"],
                    "original_minus_stabilized_mse": original["test_mse"] - mse,
                    "raw_minus_stabilized_mse": raw["test_mse"] - mse,
                    "affine_local_minus_stabilized_mse": affine["test_mse"] - mse,
                    "unit_loss_file": str((destination/"unit_losses.npz").relative_to(ROOT)),
                    "unit_loss_key": loss_key, "prediction_key": prediction_key,
                    "source_capacity_result": str((source_directory/"result.json").relative_to(ROOT)),
                    "source_capacity_unit_loss_file": original["unit_loss_file"],
                    "source_capacity_unit_loss_key": original["unit_loss_key"],
                    "n_test_units": len(losses), "unit_kind": "episode",
                    "fit_predict_seconds": time.perf_counter()-tick,
                })
                del fitted
        np.savez_compressed(destination/"unit_losses.npz", **arrays)
        verify_sensitivity_freeze(rep)
        result = {
            "status": "complete", "job_id": job_id, "phase": config["phase"], "replicate": rep,
            "execution_signature": signature, "provenance": provenance, "records": records,
            "split_diagnostics": split_diagnostics, "frozen_bank_state_unchanged": True,
            "source_seed_ids": source["seeds"], "total_seconds": time.perf_counter()-started,
            "scope": "post-hoc numerical sensitivity with unchanged training bank and reused phase4 train/validation/test data; heads retuned only on validation; no new confirmatory inference",
        }
        _write_json(result_path, result)
        return {"job_id": job_id, "status": "complete", "seconds": result["total_seconds"]}
    except Exception as error:
        failure = {"status": "failed", "job_id": job_id, "replicate": rep,
                   "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        _write_json(destination/f"failure_{time.time_ns()}.json", failure)
        return {key: failure[key] for key in ("status", "job_id", "error_type", "error")}


def collect(config, frozen):
    import pandas as pd
    jobs = []
    for rep in range(config["repetitions"]):
        path = ROOT/"results_numerical_sensitivity"/f"r{rep:02d}_forecast_van_der_pol"/"result.json"
        result = json.loads(path.read_text())
        expected, _ = _signature(frozen, rep)
        if result.get("status") != "complete" or result.get("execution_signature") != expected or len(result["records"]) != 4:
            raise RuntimeError(f"Incomplete or stale numerical sensitivity job: {path.parent.name}")
        jobs.append(result)
    records = [row for job in jobs for row in job["records"]]
    _write_json(ROOT/"numerical_sensitivity_records.json", records)
    frame = pd.DataFrame(records)
    export = frame.copy()
    for name in ("selected", "test_mse_by_target"):
        export[name] = export[name].map(json.dumps)
    export.to_csv(ROOT/"numerical_sensitivity_records.csv", index=False)
    summary = []
    for (feature, landmarks), rows in frame.groupby(["feature", "landmarks"], sort=True):
        summary.append({
            "feature": feature, "landmarks": int(landmarks), "repetitions": len(rows),
            "original_mean_mse": float(rows.original_test_mse.mean()),
            "stabilized_mean_mse": float(rows.test_mse.mean()),
            "stabilized_sd_mse": float(rows.test_mse.std(ddof=1)),
            "raw_mean_mse": float(rows.raw_baseline_mse.mean()),
            "affine_local_mean_mse": float(rows.affine_local_baseline_mse.mean()),
            "mean_original_minus_stabilized": float(rows.original_minus_stabilized_mse.mean()),
            "mean_raw_minus_stabilized": float(rows.raw_minus_stabilized_mse.mean()),
            "mean_affine_local_minus_stabilized": float(rows.affine_local_minus_stabilized_mse.mean()),
            "stabilized_beats_raw_repetitions": int((rows.raw_minus_stabilized_mse>0).sum()),
            "stabilized_beats_affine_local_repetitions": int((rows.affine_local_minus_stabilized_mse>0).sum()),
        })
    diagnostics = {}
    for split in ("train", "validation", "test"):
        values = [next(x for x in job["split_diagnostics"] if x["split"]==split) for job in jobs]
        total = sum(v["original_weight_diagnostics"]["weight_rows"] for v in values)
        diagnostics[split] = {
            "query_cluster_pairs": total,
            "original_zero_normalizers": sum(v["original_weight_diagnostics"]["zero_normalizers"] for v in values),
            "stabilized_zero_normalizers": sum(v["stabilized_weight_diagnostics"]["zero_normalizers"] for v in values),
            "original_all_zero_gaussian_rows": sum(v["kernel_rescaling"]["original_all_zero_kernel_rows"] for v in values),
            "maximum_feature_difference": {feature: max(v["feature_differences"][feature]["maximum_absolute"] for v in values)
                                           for feature in ("kahm:phi", "kahm:phi_logz", "kahm:scores", "kahm:local")},
        }
    output = {"scope": "descriptive post-hoc sensitivity on reused phase4 data; no confidence intervals or new confirmation claim",
              "methods": summary, "normalizer_diagnostics": diagnostics,
              "freeze_sha256": _hash_file(FREEZE_FILE), "completed_jobs": len(jobs), "failed_jobs": 0}
    _write_json(ROOT/"numerical_sensitivity_summary.json", output)
    pd.DataFrame(summary).to_csv(ROOT/"numerical_sensitivity_summary.csv", index=False)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_tests(), indent=2), flush=True)
        return
    if args.freeze:
        frozen = freeze_plan()
        print(json.dumps({"status": "frozen", "created_system_utc": frozen["created_system_utc"],
                          "source_files": len(frozen["sha256"]), "input_files": len(frozen["input_sha256"]),
                          "arithmetic_checks": frozen["arithmetic_checks"]}), flush=True)
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = _load_config()
    frozen = verify_sensitivity_freeze()
    if args.collect:
        print(json.dumps({"status": "collected", "records": collect(config, frozen)}), flush=True)
        return
    failures = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, rep, config) for rep in range(config["repetitions"])]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
            print(json.dumps({"completed": completed, "total": config["repetitions"], **result}), flush=True)
    if failures:
        print(json.dumps({"failed_jobs": len(failures), "seconds": time.perf_counter()-started}), flush=True)
        raise SystemExit(1)
    frozen = verify_sensitivity_freeze()
    count = collect(config, frozen)
    print(json.dumps({"completed_jobs": config["repetitions"], "failed_jobs": 0,
                      "records": count, "seconds": time.perf_counter()-started}), flush=True)


if __name__ == "__main__":
    main()
