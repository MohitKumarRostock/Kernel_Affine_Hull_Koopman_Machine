"""Bounded post-hoc geometry-scaling sensitivity on the capacity-study streams.

The original scientific snapshot is imported without edits.  Both scaler arms
reuse all 72 original data draws, both decoder capacities, all four original
representations, and one precisely matched locality-coordinate control.  The
raw representation stays training-standardized in both geometry arms.  Nothing
in this module selects a scaler or a representation using test outcomes.

Full predictions and model states are local run artifacts.  The compact release
contains episode losses, method records, execution provenance, and an audit of
every saved prediction.  The original seeds regenerate the complete artifacts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
HISTORICAL = REPOSITORY / "extensions" / "predictive_geometry" / "study"
if str(HISTORICAL) not in sys.path:
    sys.path.insert(0, str(HISTORICAL))

from datasets import make_forecast_split
from followup_capacity import _load_config, child_seed
from geometry import GeometryBank
from heads import fit_predict, grouped_losses

from .common import verify_freeze

SCALERS = ("standard", "minmax")
FEATURES = ("raw", "affine:local", "kahm:local", "kahm:scores", "locality:local")
ORIGINAL_FEATURES = ("raw", "affine:local", "kahm:local", "kahm:scores")
SYSTEM_IDS = {"nonlinear_ar": "nonlinear_ar", "vanderpol": "van_der_pol", "duffing": "duffing"}
DEFAULT_OUTPUT = ROOT / "runs" / "scaling_v1" / "controlled"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_hash(value):
    """Use the historical capacity runner's array digest convention."""
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_historical_freezes():
    """Check unchanged freeze hashes against compact, lossless archive bytes.

    The historical runner expects materialized JSON records.  This repository
    deliberately compresses those records.  Resolve their recorded storage
    paths without changing the source snapshot or weakening any old digest.
    """
    snapshot = json.loads((HISTORICAL / "SNAPSHOT.json").read_text())["files"]
    extension = HISTORICAL.parent
    for filename in ("protocol_freeze.json", "followup_freeze.json"):
        freeze = json.loads((HISTORICAL / filename).read_text())
        for logical_name, expected in freeze["sha256"].items():
            if logical_name in snapshot:
                path = extension / snapshot[logical_name]["path"]
            elif logical_name == "records.json":
                path = extension / "archive" / "stage1" / "records.json.gz"
            else:
                raise RuntimeError(f"No historical compact source mapping: {logical_name}")
            data = path.read_bytes()
            if path.suffix == ".gz":
                data = gzip.decompress(data)
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Historical frozen content changed: {logical_name}")
        if filename == "followup_freeze.json":
            if freeze["original_protocol_freeze_sha256"] != file_hash(HISTORICAL / "protocol_freeze.json"):
                raise RuntimeError("Original protocol freeze is not the follow-up's recorded dependency")
    return {name: file_hash(HISTORICAL / name)
            for name in ("protocol_freeze.json", "followup_freeze.json", "SNAPSHOT.json")}


def execution_metadata():
    frozen = verify_freeze()
    historical = verify_historical_freezes()
    # The manifest itself is not a scientific dependency of its own hash list.
    frozen_digest = file_hash(ROOT / "protocol_freeze.json")
    return {
        "freeze_sha256": frozen_digest,
        "protocol_sha256": frozen["protocol_sha256"],
        "source_sha256": frozen["sha256"],
        "historical_freezes": historical,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        "blas_threads": 1,
    }


def locality_coordinates(bank, x, transformed):
    """Match the natural study's residual-plus-locality gate exactly.

    ``native_affine:phi`` from the unchanged GeometryBank is proportional to
    exp[-(||z-affine(z)||^2/residual_scale + 0.1||z-center||^2/scale)].
    It is concatenated with the same affinity-gated effective PCA coordinates
    as the original local representations.  This is an added control, not an
    archived capacity-study method, and has no claimed historical parity.
    """
    z = bank._scale_inputs(x)
    phi = transformed["native_affine:phi"]
    gated = []
    offset = 0
    for block in bank.blocks_:
        for index, model in enumerate(block["models"]):
            coordinate = (z - model.center) @ model.basis
            gated.append(phi[:, offset + index:offset + index + 1] * coordinate)
        offset += block["count"]
    return np.concatenate([phi] + gated, axis=1)


def exceedance(bank, x):
    """Report future values beyond the training range without clipping them."""
    x = np.asarray(x, float)
    varying = ~bank.constant_features_
    span = np.where(varying, bank.input_range_, 1.0)
    z = 2.0 * (x - bank.input_min_) / span - 1.0
    z[:, ~varying] = 0.0
    outside = (z < -1.0) | (z > 1.0)
    # Compare original values as well, because constant training coordinates are
    # mapped to zero by the historical scaler even if they vary at evaluation.
    outside_original = (x < bank.input_min_) | (x > bank.input_max_)
    return {
        "n_rows": int(len(x)), "n_coordinates": int(x.shape[1]),
        "constant_training_coordinates": int(np.count_nonzero(~varying)),
        "outside_coordinate_count": int(np.count_nonzero(outside)),
        "outside_coordinate_fraction": float(outside.mean()),
        "outside_row_count": int(np.count_nonzero(np.any(outside, axis=1))),
        "outside_row_fraction": float(np.mean(np.any(outside, axis=1))),
        "original_outside_coordinate_count": int(np.count_nonzero(outside_original)),
        "constant_coordinate_changed_count": int(np.count_nonzero(outside_original[:, ~varying])),
        "minmax_minimum": float(z.min()), "minmax_maximum": float(z.max()),
        "clipped_coordinates": 0,
    }


def _archive_records():
    path = REPOSITORY / "extensions" / "predictive_geometry" / "archive" / "capacity" / "capacity_records.json"
    records = json.loads(path.read_text())
    return {(row["replicate"], row["setting_id"], row["feature"], row["landmarks"]): row
            for row in records if row["feature"] in ORIGINAL_FEATURES}


def historical_parity(record, prediction, losses, historical_record, historical_arrays=None):
    """Require numerical parity; report actual errors, not only a boolean."""
    actual = np.asarray(record["test_mse_by_target"])
    expected = np.asarray(historical_record["test_mse_by_target"])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    # Alpha/bandwidth choices and effective dimensions are discrete checks.
    for current, previous in zip(record["selected"], historical_record["selected"], strict=True):
        for key in ("alpha", "bandwidth_multiplier", "basis_dimension"):
            if current[key] != previous[key]:
                raise AssertionError(f"Historical selected parameter changed: {key}")
        for key in ("bandwidth_sq", "validation_mse_by_target"):
            np.testing.assert_allclose(current[key], previous[key], rtol=1e-10, atol=1e-12)
    output = {
        "status": "pass", "compact_record_checked": True,
        "test_mse_absolute_error": abs(record["test_mse"] - historical_record["test_mse"]),
        "target_mse_max_absolute_error": float(np.max(np.abs(actual - expected))),
        "full_prediction_checked": historical_arrays is not None,
    }
    if historical_arrays is not None:
        expected_pred = historical_arrays[historical_record["prediction_key"]]
        expected_losses = historical_arrays[historical_record["unit_loss_key"]]
        np.testing.assert_allclose(prediction, expected_pred, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(losses, expected_losses, rtol=1e-9, atol=1e-11)
        output["prediction_max_absolute_error"] = float(np.max(np.abs(prediction - expected_pred)))
        output["episode_loss_max_absolute_error"] = float(np.max(np.abs(losses - expected_losses)))
    return output


def _run_job(output, historical_results, rep, system):
    """Run one fully paired job; its predictive values are never printed."""
    started = time.perf_counter()
    output = Path(output)
    job_id = f"r{rep:02d}_forecast_{SYSTEM_IDS[system]}"
    destination = output / "jobs" / job_id
    destination.mkdir(parents=True, exist_ok=True)
    try:
        meta = execution_metadata()
        config, base = _load_config()
        signature = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        result_path = destination / "result.json"
        if result_path.is_file():
            previous = json.loads(result_path.read_text())
            if previous.get("status") == "complete":
                if previous.get("execution_signature") != signature:
                    raise RuntimeError("Refusing to reuse a completed job with changed source or runtime")
                if historical_results is not None and not previous.get("full_historical_parity_requested"):
                    raise RuntimeError("Completed job did not include requested full historical parity")
                for filename, digest in previous["artifacts_sha256"].items():
                    if file_hash(destination / filename) != digest:
                        raise RuntimeError(f"Completed artifact changed: {filename}")
                return {"job_id": job_id, "status": "already_complete"}
        fc = base["forecast"]
        seeds = {name: child_seed(config, rep, system, stream)
                 for name, stream in (("train", 0), ("validation", 1), ("test", 2), ("bank", 10), ("head", 11))}
        splits = [make_forecast_split(system, fc[name + "_episodes"], seeds[name],
                                      history=fc["history"], horizons=tuple(fc["horizons"]), stride=fc["stride"])
                  for name in ("train", "validation", "test")]
        train, validation, test = splits
        split_metadata = [{"split": name, "metadata": split["metadata"],
                           "X_sha256": array_hash(split["X"]), "Y_sha256": array_hash(split["Y"]),
                           "group_sha256": array_hash(split["groups"])}
                          for name, split in zip(("train", "validation", "test"), splits, strict=True)]
        original_records = _archive_records()
        original_arrays = None
        historical_evidence = None
        if historical_results is not None:
            original_directory = Path(historical_results) / job_id
            original_result_path = original_directory / "result.json"
            original_arrays_path = original_directory / "unit_losses.npz"
            original_result = json.loads(original_result_path.read_text())
            if original_result["seeds"] != seeds:
                raise AssertionError("Original data/fit seeds differ")
            for new, old in zip(split_metadata, original_result["split_metadata"], strict=True):
                for key in ("X_sha256", "Y_sha256", "group_sha256"):
                    if new[key] != old[key]:
                        raise AssertionError(f"Original {new['split']} array changed: {key}")
            original_arrays = dict(np.load(original_arrays_path))
            np.testing.assert_array_equal(test["Y"], original_arrays["Y"])
            np.testing.assert_array_equal(test["groups"], original_arrays["groups"])
            historical_evidence = {
                "result_json_sha256": file_hash(original_result_path),
                "unit_losses_npz_sha256": file_hash(original_arrays_path),
                "data_hashes_match": True,
                "source": "Original capacity-study full-review run, identified by content hashes; regenerated from frozen source and seeds",
            }
        predictions = {"Y": test["Y"], "groups": test["groups"]}
        unit_losses = {}
        records, arms = [], {}
        head_config = dict(base["heads"])
        head_config.pop("n_landmarks", None)
        # Alternating the first arm by repetition avoids always favoring one
        # scaler in warm-cache execution diagnostics.  This affects no RNG.
        arm_order = SCALERS if rep % 2 == 0 else tuple(reversed(SCALERS))
        for scaling in arm_order:
            arm_started = time.perf_counter()
            geometry_config = dict(base["geometry"], scaling=scaling)
            bank = GeometryBank(seed=seeds["bank"], **geometry_config).fit(train["X"])
            features = []
            for index, split in enumerate(splits):
                transformed = bank.transform(split["X"], diagnostics=index == 2)
                transformed["locality:local"] = locality_coordinates(bank, split["X"], transformed)
                features.append({feature: transformed[feature] for feature in FEATURES})
            # Full fitted banks allow rechecking nonlinear geometry diagnostics.
            bank.save(destination / f"bank__{scaling}.pkl")
            arms[scaling] = {
                "geometry": bank.metadata(), "test_geometry_diagnostics": bank.last_diagnostics_,
                "range_exceedance": {name: exceedance(bank, split["X"])
                                     for name, split in zip(("train", "validation", "test"), splits, strict=True)},
                "fit_and_transform_seconds": time.perf_counter() - arm_started,
            }
            for landmarks in config["landmark_counts"]:
                for feature in FEATURES:
                    tick = time.perf_counter()
                    prediction, selected, fitted = fit_predict(
                        features[0][feature], train["Y"], features[1][feature], validation["Y"], features[2][feature],
                        head="rbf", seed=seeds["head"], bounds=train["bounds"], separate_targets=False,
                        validation_groups=validation["groups"], n_landmarks=landmarks, **head_config)
                    losses = grouped_losses(prediction, test["Y"], test["groups"])
                    suffix = f"{scaling}__{feature.replace(':', '__')}__m{landmarks}"
                    prediction_key, loss_key = "prediction__" + suffix, "loss__" + suffix
                    predictions[prediction_key], unit_losses[loss_key] = prediction, losses
                    record = {
                        "phase": "post_hoc_scaling_sensitivity", "replicate": rep, "job_id": job_id,
                        "system": system, "setting_id": "forecast_" + SYSTEM_IDS[system],
                        "scaling": scaling, "feature": feature, "head": "rbf", "landmarks": int(landmarks),
                        "test_mse": float(losses.mean()), "test_mse_by_target": losses.mean(axis=0).tolist(),
                        "selected": selected,
                        "validation_mse": float(np.mean([item["validation_mse_by_target"] for item in selected])),
                        "prediction_file": f"jobs/{job_id}/predictions.npz", "prediction_key": prediction_key,
                        "unit_loss_file": f"jobs/{job_id}/unit_losses.npz", "unit_loss_key": loss_key,
                        "input_feature_dimension": int(features[0][feature].shape[1]),
                        "selected_basis_dimension": int(selected[0]["basis_dimension"]),
                        "n_train_rows": int(len(train["X"])), "n_train_episodes": fc["train_episodes"],
                        "n_validation_episodes": fc["validation_episodes"], "n_test_units": int(len(losses)),
                        "unit_kind": "episode", "fit_predict_seconds": time.perf_counter() - tick,
                        "bank_seed": seeds["bank"], "head_seed": seeds["head"],
                        "historical_parity": None,
                    }
                    if scaling == "standard" and feature in ORIGINAL_FEATURES:
                        original = original_records[(rep, record["setting_id"], feature, landmarks)]
                        record["historical_parity"] = historical_parity(record, prediction, losses, original, original_arrays)
                    records.append(record)
                    del fitted
            del bank, features
        raw_parity = []
        for landmarks in config["landmark_counts"]:
            first = predictions[f"prediction__standard__raw__m{landmarks}"]
            second = predictions[f"prediction__minmax__raw__m{landmarks}"]
            np.testing.assert_array_equal(first, second)
            raw_parity.append({"landmarks": int(landmarks), "exact_prediction_equality": True,
                               "prediction_max_absolute_error": 0.0})
        np.savez_compressed(destination / "predictions.npz", **predictions)
        np.savez_compressed(destination / "unit_losses.npz", **unit_losses)
        artifact_names = ["predictions.npz", "unit_losses.npz"]
        artifact_names += [f"bank__{scaler}.pkl{suffix}" for scaler in SCALERS for suffix in ("", ".json")]
        verify_freeze()
        result = {
            "status": "complete", "phase": "post_hoc_scaling_sensitivity", "job_id": job_id,
            "replicate": rep, "system": system, "setting_id": "forecast_" + SYSTEM_IDS[system],
            "execution_signature": signature, "execution": meta, "seeds": seeds,
            "split_metadata": split_metadata, "arm_execution_order": list(arm_order),
            "arms": arms, "records": records, "raw_cross_arm_parity": raw_parity,
            "full_historical_parity_requested": historical_results is not None,
            "historical_evidence": historical_evidence,
            "artifacts_sha256": {name: file_hash(destination / name) for name in artifact_names},
            "total_seconds": time.perf_counter() - started,
            "scope": {
                "data": "Reused original 24 complete data/fit repetitions per law; no new independent confirmation",
                "scaler": "Train-only geometry standardization versus unclipped coordinatewise minmax [-1,1]; original raw path stays standardized",
                "targets": "Original units and original known-support output clipping [-1,1] are unchanged",
                "head_selection": "Original alpha/bandwidth grid and validation-only selection, independently within each fixed arm",
                "locality_control": "Added native-affine residual-plus-locality gates with the same effective PCA coordinates; not previously archived",
                "inference": "Descriptive paired sensitivity across original repetitions; no new confirmatory family or scaler selector",
                "timings": "Shared-bank execution diagnostics, not standalone end-to-end comparative costs",
            },
        }
        write_json(result_path, result)
        return {"job_id": job_id, "status": "complete", "seconds": result["total_seconds"]}
    except Exception as error:
        failure = {"status": "failed", "job_id": job_id, "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc()}
        write_json(destination / f"failure_{time.time_ns()}.json", failure)
        return {"job_id": job_id, "status": "failed", "error_type": type(error).__name__, "error": str(error)}


def run_job(output, historical_results, rep, system):
    with threadpool_limits(limits=1):
        return _run_job(output, historical_results, rep, system)


def collect(output):
    """Require every declared job and retain all method outcomes in one file."""
    output = Path(output)
    config, _ = _load_config()
    meta = execution_metadata()
    signature = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    records, jobs = [], []
    for rep in range(config["repetitions"]):
        for system in config["systems"]:
            job_id = f"r{rep:02d}_forecast_{SYSTEM_IDS[system]}"
            path = output / "jobs" / job_id / "result.json"
            result = json.loads(path.read_text())
            if result["status"] != "complete" or result["execution_signature"] != signature:
                raise RuntimeError(f"Missing or stale result: {job_id}")
            expected = {(scaling, feature, landmarks) for scaling in SCALERS for feature in FEATURES
                        for landmarks in config["landmark_counts"]}
            actual = {(row["scaling"], row["feature"], row["landmarks"]) for row in result["records"]}
            if len(result["records"]) != len(expected) or actual != expected:
                raise RuntimeError(f"Incomplete or duplicated method cells: {job_id}")
            for name, expected_hash in result["artifacts_sha256"].items():
                if file_hash(path.parent / name) != expected_hash:
                    raise RuntimeError(f"Changed artifact: {job_id}/{name}")
            records.extend(result["records"])
            jobs.append({"job_id": job_id, "result_sha256": file_hash(path),
                         "total_seconds": result["total_seconds"],
                         "artifacts_sha256": result["artifacts_sha256"]})
    write_json(output / "records.json", records)
    write_json(output / "run_manifest.json", {
        "status": "complete", "execution": meta, "jobs": jobs,
        "n_jobs": len(jobs), "n_method_records": len(records),
        "records_sha256": file_hash(output / "records.json"),
        "scope": "Post-hoc scaling sensitivity on the original capacity-study streams",
    })
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--historical-results", type=Path,
                        help="Optional original results_capacity directory for full-prediction parity")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.workers != 1:
        parser.error("The fixed sensitivity protocol requires exactly one worker")
    execution_metadata()
    if args.collect:
        print(json.dumps({"status": "collected", "records": collect(args.output)}), flush=True)
        return
    config, _ = _load_config()
    jobs = [(rep, system) for rep in range(config["repetitions"]) for system in config["systems"]]
    failures = []
    started = time.perf_counter()
    for rep, system in jobs:
        outcome = run_job(args.output, args.historical_results, rep, system)
        print(json.dumps(outcome), flush=True)
        if outcome["status"] == "failed":
            failures.append(outcome)
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; incomplete outcomes are preserved, not omitted")
    count = collect(args.output)
    print(json.dumps({"status": "complete", "jobs": len(jobs), "records": count,
                      "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
