"""Run the frozen, paired KAHM mechanism study. See PROTOCOL.txt.

No outcome is used to choose the geometric bank. Hyperparameters use only the
validation split. Confirmation and natural runs require a matching freeze file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np

from datasets import make_forecast_split, make_branch_split, branch_targets, make_airfoil_split
from geometry import GeometryBank
from heads import fit_predict, grouped_losses, quadratic

ROOT = Path(__file__).resolve().parent
PRIMARY_FEATURES = {"raw", "kahm:phi", "affine:phi", "kahm:phi_logz", "kahm:local"}
SELECTOR_CANDIDATES = {(f, "ridge") for f in (
    "raw", "kahm:phi", "affine:phi", "kahm:phi_logz", "affine:phi_logz",
    "kahm:local", "affine:local", "kahm:phi_raw", "affine:phi_raw")}
SELECTOR_CANDIDATES.add(("raw", "rbf"))
SYSTEM_IDS = {"nonlinear_ar": "nonlinear_ar", "vanderpol": "van_der_pol", "duffing": "duffing"}
LAM_IDS = {0.0: "0", 0.5: "0p5", 1.0: "1"}


def pipelines():
    items = [("raw", "ridge"), ("raw", "rbf"), ("raw_quadratic", "ridge")]
    for geometry in ("kahm", "affine"):
        for variant in ("phi", "phi_logz", "scores", "local"):
            items.extend([(f"{geometry}:{variant}", "ridge"), (f"{geometry}:{variant}", "rbf")])
        for variant in ("radial_angle", "radial", "angle", "phi_raw", "logz"):
            items.append((f"{geometry}:{variant}", "ridge"))
    items.extend([("centroid:phi", "ridge"), ("centroid:phi_logz", "ridge"),
                  ("centroid:local", "ridge"), ("native_rbf:phi", "ridge"),
                  ("native_affine:phi", "ridge")])
    return items


def child_seed(config, phase, rep, family, stream):
    code = {"pilot": 1, "confirmation": 2, "natural": 3}[phase]
    seq = np.random.SeedSequence([config["master_seed"], code, rep, family, stream])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def array_hash(X):
    X = np.ascontiguousarray(X)
    return hashlib.sha256(str(X.shape).encode() + X.dtype.str.encode() + X.tobytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _subset(split, n):
    result = dict(split)
    for name in ("X", "S", "T", "E", "groups"):
        result[name] = split[name][:n]
    result["metadata"] = dict(split["metadata"], n_rows=n, nested_training_subset=True)
    return result


def make_splits(config, phase, rep, task):
    kind = task["kind"]
    if kind == "forecast":
        system = task["system"]
        family = 100 + config["forecast"]["systems"].index(system)
        fc = config["forecast"]
        output = []
        for index, name in enumerate(("train", "validation", "test")):
            output.append(make_forecast_split(
                system, fc[name + "_episodes"], child_seed(config, phase, rep, family, index),
                history=fc["history"], horizons=tuple(fc["horizons"]), stride=fc["stride"]))
        return output, None, child_seed(config, phase, rep, family, 10)
    if kind == "branch":
        bc, output = config["branch"], []
        # Same latent points/noise across curvatures and nested train sizes.
        for index, n in enumerate((max(bc["train_sizes"]), bc["validation_size"], bc["test_size"])):
            split = make_branch_split(n, child_seed(config, phase, rep, 200, index),
                                      curvature=task["curvature"], separation=bc["separation"], noise=bc["noise"])
            if index == 0:
                split = _subset(split, task["n_train"])
            pairs = [branch_targets(split, lam) for lam in bc["lambdas"]]
            split["Y"] = np.column_stack([p[0][:, 0] for p in pairs])
            split["M"] = np.column_stack([p[1][:, 0] for p in pairs])
            output.append(split)
        return output, bc["lambdas"], child_seed(config, phase, rep, 200, 10 + task["n_train"])
    output = [make_airfoil_split(name) for name in ("train", "validation", "test")]
    return output, None, child_seed(config, phase, rep, 300, 10)


def run_job(config, phase, rep, task):
    start = time.perf_counter()
    execution_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ("config.json", "run_study.py", "geometry.py", "heads.py",
                                     "datasets.py", "mechanism_diagnostics.py", "upstream_manifest.json")}
    signature = hashlib.sha256(json.dumps(execution_hashes, sort_keys=True).encode()).hexdigest()
    job_id = f"r{rep:02d}_{task['id']}"
    destination = ROOT / "results" / phase / job_id
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("status") == "complete":
            if previous.get("execution_signature") != signature:
                raise RuntimeError(f"Refusing stale completed results for {job_id}; preserve and invalidate explicitly")
            return {"job_id": job_id, "status": "already_complete"}
    try:
        splits, lambdas, bank_seed = make_splits(config, phase, rep, task)
        train, validation, test = splits
        bank = GeometryBank(seed=bank_seed, **config["geometry"]).fit(train["X"])
        features = []
        for index, split in enumerate(splits):
            features.append(bank.transform(split["X"], diagnostics=index == 2))
            features[-1]["raw_quadratic"] = quadratic(features[-1]["raw"])
        diagnostics = bank.last_diagnostics_
        bank.save(destination / "bank.pkl")
        mechanism = {}
        if task["kind"] == "branch":
            try:
                from mechanism_diagnostics import geometry_diagnostics
                mechanism = geometry_diagnostics(bank, train, test, branch=True)
            except ImportError:
                raise RuntimeError("Declared branch diagnostic module is missing")
        records, loss_arrays, saved_heads, selector_cache = [], {}, {}, {}
        primary_predictions = {}
        for pipeline_index, (feature, head) in enumerate(pipelines()):
            head_start = time.perf_counter()
            pred, selected, fitted = fit_predict(
                features[0][feature], train["Y"], features[1][feature], validation["Y"], features[2][feature],
                head=head, seed=child_seed(config, phase, rep, 400, 1),
                bounds=train["bounds"], separate_targets=lambdas is not None,
                validation_groups=validation["groups"], **config["heads"])
            unit_loss = grouped_losses(pred, test["Y"], test["groups"])
            if (feature, head) in SELECTOR_CANDIDATES:
                selector_cache[(feature, head)] = (pred, fitted)
            save_primary = feature in PRIMARY_FEATURES and head == "ridge"
            if save_primary:
                saved_heads[feature] = fitted
                primary_predictions[feature] = pred
            targets = list(enumerate(lambdas)) if lambdas is not None else [(None, None)]
            for column, lam in targets:
                setting_id = (f"{task['id']}_l{LAM_IDS[lam]}" if lam is not None else task["id"])
                loss = unit_loss[:, column:column + 1] if column is not None else unit_loss
                selection = selected[column:column + 1] if column is not None else selected
                is_primary_cell = task["kind"] == "forecast" or (task["kind"] == "branch" and task["n_train"] == 1024)
                row = {"phase": phase, "replicate": rep, "setting_id": setting_id,
                       "feature": feature, "head": head, "test_mse": float(loss.mean()),
                       "test_mse_by_target": loss.mean(axis=0).tolist(),
                       "validation_mse": float(np.mean([s["validation_mse_by_target"] for s in selection])),
                       "selected": selection, "primary_cell": is_primary_cell,
                       "input_feature_dimension": int(features[0][feature].shape[1]),
                       "n_train_rows": len(train["X"]), "n_test_units": len(loss),
                       "unit_kind": test["metadata"].get("independence_unit", "operating_setting_group"),
                       "conditional_mean_mse": None, "unit_loss_file": None, "unit_loss_key": None,
                       "fit_predict_seconds": time.perf_counter() - head_start,
                       "diagnostics": str((destination / "result.json").relative_to(ROOT))}
                if column is not None:
                    row["conditional_mean_mse"] = float(np.mean((pred[:, column] - test["M"][:, column]) ** 2))
                if save_primary:
                    key = feature.replace(":", "__") + ("_l" + LAM_IDS[lam] if lam is not None else "")
                    loss_arrays[key] = loss
                    row["unit_loss_file"] = str((destination / "unit_losses.npz").relative_to(ROOT))
                    row["unit_loss_key"] = key
                records.append(row)
        # A prespecified practical selector: choose among ten complete pipelines
        # by validation loss alone, separately for distinct branch target laws.
        # This is a secondary evaluation on the untouched test split.
        selector_states = []
        selector_prediction = np.empty_like(test["Y"])
        setting_ids = list(dict.fromkeys(row["setting_id"] for row in records))
        for setting_index, setting_id in enumerate(setting_ids):
            candidates = [row for row in records if row["setting_id"] == setting_id
                          and (row["feature"], row["head"]) in SELECTOR_CANDIDATES]
            chosen = min(candidates, key=lambda row: row["validation_mse"])
            choice = (chosen["feature"], chosen["head"])
            pred, fitted = selector_cache[choice]
            columns = slice(setting_index, setting_index + 1) if lambdas is not None else slice(None)
            selector_prediction[:, columns] = pred[:, columns]
            loss = grouped_losses(pred[:, columns], test["Y"][:, columns], test["groups"])
            key = "validation_selector" + ("_l" + LAM_IDS[lambdas[setting_index]] if lambdas is not None else "")
            loss_arrays[key] = loss
            row = dict(chosen, feature="validation_selector", head="selected",
                       selected={"pipeline_feature": choice[0], "pipeline_head": choice[1],
                                 "parameters": chosen["selected"]},
                       unit_loss_key=key,
                       unit_loss_file=str((destination / "unit_losses.npz").relative_to(ROOT)),
                       primary_cell=False)
            records.append(row)
            selector_states.append({"setting_id": setting_id, "feature": choice[0],
                                    "head": choice[1], "fitted_head": fitted})
        saved_heads["validation_selector"] = selector_states
        primary_predictions["validation_selector"] = selector_prediction
        loss_arrays["test_Y"] = test["Y"]
        loss_arrays["test_groups"] = test["groups"]
        for name, prediction in primary_predictions.items():
            loss_arrays["prediction__" + name.replace(":", "__")] = prediction
        np.savez_compressed(destination / "unit_losses.npz", **loss_arrays)
        with (destination / "primary_heads.pkl").open("wb") as handle:
            pickle.dump(saved_heads, handle, protocol=5)
        metadata = [{"metadata": s["metadata"], "X_sha256": array_hash(s["X"]),
                     "Y_sha256": array_hash(s["Y"]), "group_sha256": array_hash(s["groups"])} for s in splits]
        result = {"status": "complete", "phase": phase, "replicate": rep, "task": task,
                  "execution_signature": signature, "execution_sha256": execution_hashes,
                  "job_id": job_id, "bank_seed": bank_seed, "split_metadata": metadata,
                  "geometry": bank.metadata(), "test_geometry_diagnostics": diagnostics,
                  "mechanism_diagnostics": mechanism, "records": records,
                  "total_seconds": time.perf_counter() - start}
        write_json(result_path, result)
        return {"job_id": job_id, "status": "complete", "seconds": result["total_seconds"]}
    except Exception:
        failure = {"status": "failed", "phase": phase, "replicate": rep,
                   "task": task, "traceback": traceback.format_exc()}
        write_json(destination / "failure.json", failure)
        return {"job_id": job_id, **failure}


def tasks_for(config, phase):
    if phase == "natural":
        return [{"kind": "natural", "id": "airfoil_grouped"}]
    tasks = [{"kind": "forecast", "system": system, "id": "forecast_" + SYSTEM_IDS[system]}
             for system in config["forecast"]["systems"]]
    for c in config["branch"]["curvatures"]:
        for n in config["branch"]["train_sizes"]:
            tasks.append({"kind": "branch", "curvature": c, "n_train": n,
                          "id": f"branch_c{c}_n{n}"})
    return tasks


def verify_freeze():
    freeze_path = ROOT / "protocol_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("Confirmation requires a frozen protocol")
    frozen = json.loads(freeze_path.read_text())
    for path, expected in frozen["sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen file changed: {path}")


def collect():
    import pandas as pd
    records = []
    for path in sorted((ROOT / "results").glob("*/*/result.json")):
        records.extend(json.loads(path.read_text())["records"])
    write_json(ROOT / "records.json", records)
    frame = pd.DataFrame(records)
    for column in ("selected", "test_mse_by_target"):
        if column in frame:
            frame[column] = frame[column].map(json.dumps)
    frame.to_csv(ROOT / "records.csv", index=False)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirmation", "natural"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        print(json.dumps({"records": collect()}), flush=True)
        return
    if not args.phase:
        parser.error("--phase required unless --collect")
    config = json.loads((ROOT / "config.json").read_text())
    if args.phase != "pilot":
        verify_freeze()
    repetitions = config[args.phase + "_repetitions"]
    jobs = [(config, args.phase, rep, task) for rep in range(repetitions) for task in tasks_for(config, args.phase)]
    failures = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, *job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
            print(json.dumps({"completed": index, "total": len(jobs), **result}), flush=True)
    n = collect()
    print(json.dumps({"phase": args.phase, "seconds": time.perf_counter() - started,
                      "all_phase_record_count": n, "failed_jobs": len(failures)}), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
