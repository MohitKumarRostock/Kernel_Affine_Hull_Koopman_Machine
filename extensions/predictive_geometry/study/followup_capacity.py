"""Fresh capacity follow-up; requires both original and new protocol freezes.

Uses the unchanged dataset, geometry, and head implementations.  Phase-code 4
streams are separate from first-stage pilots, confirmation, and natural-data
fits.  Each task/repetition shares one bank and all observations across four
feature representations and two landmark counts.  Logs contain completion
metadata, never predictive outcomes.  This file does not generate data on import.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from datasets import make_forecast_split
from geometry import GeometryBank
from heads import fit_predict, grouped_losses
from run_study import verify_freeze


ROOT = Path(__file__).resolve().parent
SYSTEM_IDS = {"nonlinear_ar": "nonlinear_ar", "vanderpol": "van_der_pol", "duffing": "duffing"}
REQUIRED_SYSTEMS = ("nonlinear_ar", "vanderpol", "duffing")
REQUIRED_FEATURES = ("raw", "affine:local", "kahm:local", "kahm:scores")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes()).hexdigest()


def verify_capacity_freeze():
    """Check the unchanged first-stage freeze and every new frozen dependency."""
    verify_freeze()
    path = ROOT / "followup_freeze.json"
    if not path.is_file():
        raise RuntimeError("Capacity confirmation requires followup_freeze.json before data generation")
    frozen = json.loads(path.read_text())
    hashes = frozen.get("sha256")
    if not isinstance(hashes, dict):
        raise RuntimeError("Follow-up freeze must contain a sha256 mapping")
    for required in ("followup_capacity.py", "followup_config.json"):
        if required not in hashes:
            raise RuntimeError(f"Missing required follow-up source hash: {required}")
    for name, expected in hashes.items():
        file_path = ROOT / name
        if not file_path.is_file() or hashlib.sha256(file_path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Follow-up frozen file changed or is missing: {name}")
    original_hash = hashlib.sha256((ROOT / "protocol_freeze.json").read_bytes()).hexdigest()
    recorded_original = frozen.get("original_protocol_freeze_sha256", hashes.get("protocol_freeze.json"))
    if recorded_original is None or recorded_original != original_hash:
        raise RuntimeError("Follow-up freeze must record the unchanged original protocol_freeze.json hash")
    return frozen


def _load_config():
    config = json.loads((ROOT / "followup_config.json").read_text())
    if config.get("phase") != "capacity_confirmation" or config.get("phase_code") != 4:
        raise ValueError("The follow-up phase must be capacity_confirmation with phase_code 4")
    if config.get("master_seed") != 20261030 or config.get("repetitions") != 24:
        raise ValueError("This declared follow-up requires master_seed 20261030 and 24 repetitions")
    if tuple(config.get("systems", [])) != REQUIRED_SYSTEMS:
        raise ValueError("Follow-up systems or their seed-defining order differ from the declared protocol")
    if tuple(config.get("landmark_counts", [])) != (64, 256):
        raise ValueError("Declared landmark counts are 64 and 256")
    if tuple(config.get("features", [])) != REQUIRED_FEATURES:
        raise ValueError("Follow-up feature list differs from the declared protocol")
    base_path = ROOT / config.get("base_config", "config.json")
    base = json.loads(base_path.read_text())
    fc = base["forecast"]
    if (fc["train_episodes"], fc["validation_episodes"], fc["test_episodes"]) != (128, 128, 512):
        raise ValueError("The follow-up requires the declared 128/128/512 episode sizes")
    return config, base


def _execution_metadata(frozen, config):
    original = json.loads((ROOT / "protocol_freeze.json").read_text())
    paths = set(original["sha256"]) | set(frozen["sha256"])
    paths.update({"followup_freeze.json", "protocol_freeze.json", config.get("base_config", "config.json")})
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(paths)}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return hashes, signature


def child_seed(config, rep, system, stream):
    family = 100 + config["systems"].index(system)
    sequence = np.random.SeedSequence([config["master_seed"], 4, rep, family, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _job_id(rep, system):
    return f"r{rep:02d}_forecast_{SYSTEM_IDS[system]}"


def run_job(config, base, rep, system):
    started = time.perf_counter()
    job_id = _job_id(rep, system)
    destination = ROOT / "results_capacity" / job_id
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.json"
    try:
        frozen = verify_capacity_freeze()
        execution_hashes, signature = _execution_metadata(frozen, config)
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("status") == "complete":
                if previous.get("execution_signature") != signature:
                    raise RuntimeError(f"Refusing stale completed capacity results for {job_id}")
                return {"job_id": job_id, "status": "already_complete"}
        fc = base["forecast"]
        seeds = {name: child_seed(config, rep, system, stream)
                 for name, stream in (("train", 0), ("validation", 1), ("test", 2), ("bank", 10), ("head", 11))}
        splits = [make_forecast_split(system, fc[name + "_episodes"], seeds[name],
                                      history=fc["history"], horizons=tuple(fc["horizons"]), stride=fc["stride"])
                  for name in ("train", "validation", "test")]
        train, validation, test = splits
        bank = GeometryBank(seed=seeds["bank"], **base["geometry"]).fit(train["X"])
        features = []
        for index, split in enumerate(splits):
            generated = bank.transform(split["X"], diagnostics=index == 2)
            features.append({feature: generated[feature] for feature in config["features"]})
            del generated
        geometry_diagnostics = bank.last_diagnostics_
        bank.save(destination / "bank.pkl")
        loss_arrays = {"Y": test["Y"], "groups": test["groups"]}
        records = []
        head_config = dict(base["heads"])
        head_config.pop("n_landmarks", None)
        setting_id = "forecast_" + SYSTEM_IDS[system]
        unit_loss_file = str((destination / "unit_losses.npz").relative_to(ROOT))
        for landmarks in config["landmark_counts"]:
            for feature in config["features"]:
                tick = time.perf_counter()
                prediction, selected, fitted = fit_predict(
                    features[0][feature], train["Y"], features[1][feature], validation["Y"], features[2][feature],
                    head="rbf", seed=seeds["head"], bounds=train["bounds"], separate_targets=False,
                    validation_groups=validation["groups"], n_landmarks=landmarks, **head_config)
                losses = grouped_losses(prediction, test["Y"], test["groups"])
                suffix = feature.replace(":", "__") + f"__m{landmarks}"
                loss_key, prediction_key = "loss__" + suffix, "prediction__" + suffix
                loss_arrays[loss_key], loss_arrays[prediction_key] = losses, prediction
                records.append({
                    "phase": "capacity_confirmation", "replicate": rep, "setting_id": setting_id,
                    "feature": feature, "head": "rbf", "landmarks": int(landmarks),
                    "test_mse": float(losses.mean()), "test_mse_by_target": losses.mean(axis=0).tolist(),
                    "selected": selected,
                    "validation_mse": float(np.mean([item["validation_mse_by_target"] for item in selected])),
                    "unit_loss_file": unit_loss_file, "unit_loss_key": loss_key, "prediction_key": prediction_key,
                    "input_feature_dimension": int(features[0][feature].shape[1]),
                    "selected_basis_dimension": int(selected[0]["basis_dimension"]),
                    "n_train_rows": len(train["X"]), "n_train_episodes": fc["train_episodes"],
                    "n_validation_episodes": fc["validation_episodes"], "n_test_units": len(losses),
                    "unit_kind": "episode", "fit_predict_seconds": time.perf_counter() - tick,
                    "bank_seed": seeds["bank"], "head_seed": seeds["head"],
                })
                # Predictions, selected parameters, and deterministic seeds are retained;
                # large flexible-head model objects are intentionally not archived.
                del fitted
        # A separately declared practical rule chooses among the eight complete
        # pipelines by their joint-horizon validation loss alone.  Iteration order
        # gives deterministic ties: 64 landmarks first, then declared feature order.
        chosen = min(records, key=lambda row: row["validation_mse"])
        selector_loss_key = "loss__validation_selector"
        selector_prediction_key = "prediction__validation_selector"
        loss_arrays[selector_loss_key] = loss_arrays[chosen["unit_loss_key"]].copy()
        loss_arrays[selector_prediction_key] = loss_arrays[chosen["prediction_key"]].copy()
        selector_record = dict(chosen,
                               feature="validation_selector", head="selected", landmarks=0,
                               selected={"pipeline_feature": chosen["feature"],
                                         "landmarks": chosen["landmarks"],
                                         "parameters": chosen["selected"]},
                               unit_loss_key=selector_loss_key, prediction_key=selector_prediction_key,
                               fit_predict_seconds=sum(row["fit_predict_seconds"] for row in records))
        records.append(selector_record)
        np.savez_compressed(destination / "unit_losses.npz", **loss_arrays)
        split_metadata = [{"metadata": split["metadata"], "X_sha256": _array_hash(split["X"]),
                           "Y_sha256": _array_hash(split["Y"]), "group_sha256": _array_hash(split["groups"])}
                          for split in splits]
        verify_capacity_freeze()
        result = {
            "status": "complete", "phase": "capacity_confirmation", "replicate": rep,
            "job_id": job_id, "system": system, "setting_id": setting_id,
            "execution_signature": signature, "execution_sha256": execution_hashes,
            "seeds": seeds, "split_metadata": split_metadata,
            "geometry": bank.metadata(), "test_geometry_diagnostics": geometry_diagnostics,
            "records": records, "total_seconds": time.perf_counter() - started,
            "scope": {
                "evaluation_unit": "independent test episode, averaging its origins and the three horizons",
                "replication_unit": "independent complete training/validation/test draw",
                "head": "training-standardized linear and Nystrom coordinates followed by validation-selected ridge",
                "landmark_comparison": "same seed and training rows across feature methods; different landmark counts are not guaranteed nested by upstream choice()",
                "timings": "shared-bank execution diagnostics, not separate end-to-end cost comparisons",
                "model_archive": "full geometry bank; selected head parameters and exact predictions, without large fitted flexible-head states",
                "secondary_selector": "minimum joint-horizon validation MSE among all eight pipelines; test outcomes do not enter selection",
            },
        }
        _write_json(result_path, result)
        return {"job_id": job_id, "status": "complete", "seconds": result["total_seconds"]}
    except Exception as error:
        failure = {"status": "failed", "phase": "capacity_confirmation", "replicate": rep,
                   "job_id": job_id, "system": system, "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc()}
        _write_json(destination / f"failure_{time.time_ns()}.json", failure)
        return {"job_id": job_id, "status": "failed", "error_type": type(error).__name__, "error": str(error)}


def collect(config, frozen):
    """Aggregate only a complete, matching set of all declared follow-up jobs."""
    import pandas as pd
    _, signature = _execution_metadata(frozen, config)
    records = []
    for rep in range(config["repetitions"]):
        for system in config["systems"]:
            path = ROOT / "results_capacity" / _job_id(rep, system) / "result.json"
            if not path.is_file():
                raise RuntimeError(f"Capacity follow-up is incomplete: {path.parent.name}")
            result = json.loads(path.read_text())
            if result.get("status") != "complete" or result.get("execution_signature") != signature:
                raise RuntimeError(f"Incomplete or stale capacity job: {path.parent.name}")
            expected_rows = len(config["features"]) * len(config["landmark_counts"]) + 1
            if len(result["records"]) != expected_rows:
                raise RuntimeError(f"Missing method outcomes in {path.parent.name}")
            records.extend(result["records"])
    _write_json(ROOT / "capacity_records.json", records)
    frame = pd.DataFrame(records)
    for column in ("selected", "test_mse_by_target"):
        frame[column] = frame[column].map(json.dumps)
    frame.to_csv(ROOT / "capacity_records.csv", index=False)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    frozen = verify_capacity_freeze()
    config, base = _load_config()
    if args.collect:
        print(json.dumps({"status": "collected", "records": collect(config, frozen)}), flush=True)
        return
    jobs = [(rep, system) for rep in range(config["repetitions"]) for system in config["systems"]]
    failures = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, config, base, rep, system): (rep, system) for rep, system in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            rep, system = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"job_id": _job_id(rep, system), "status": "failed",
                          "error_type": type(error).__name__, "error": str(error)}
            if result["status"] == "failed":
                failures.append(result)
            print(json.dumps({"completed": completed, "total": len(jobs), **result}), flush=True)
    if failures:
        print(json.dumps({"phase": config["phase"], "completed_jobs": len(jobs),
                          "failed_jobs": len(failures), "seconds": time.perf_counter() - started}), flush=True)
        raise SystemExit(1)
    frozen = verify_capacity_freeze()
    count = collect(config, frozen)
    print(json.dumps({"phase": config["phase"], "completed_jobs": len(jobs), "failed_jobs": 0,
                      "records": count, "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
