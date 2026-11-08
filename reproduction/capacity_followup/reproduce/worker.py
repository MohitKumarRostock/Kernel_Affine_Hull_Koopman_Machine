"""Execution adapters for unchanged historical modules.

Only path resolution and execution bookkeeping live here. Scientific source
bytes, data laws, seeds, fitting, grids, and contrasts remain in study/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
from pathlib import Path
import sys

from .io import ROOT, bytes_from, copy_file, digest, materialize, now, read_json, verify_archive, write_json


def activate(work):
    work = Path(work).resolve()
    sys.path.insert(0, str(work))
    return work


def sensitivity_module(work):
    work = activate(work)
    module = importlib.import_module("numerical_sensitivity")
    module.FREEZE_FILE = work / "sensitivity_execution_manifest.json"
    return module


def sensitivity_initializer(work):
    sensitivity_module(work)


def sensitivity_job(rep, config):
    import numerical_sensitivity
    return numerical_sensitivity.run_job(rep, config)


def make_sensitivity_manifest(module, execution_path):
    """Bind this rerun to its own newly generated phase4 files.

    This is deliberately not the historical numerical_sensitivity_freeze.json,
    which stays unchanged. No claim of new preregistration is made.
    """
    module.verify_capacity_freeze()
    config = module._load_config()
    if module.FREEZE_FILE.exists():
        module.verify_sensitivity_freeze()
        return read_json(module.FREEZE_FILE)
    historical = read_json(module.ROOT / "numerical_sensitivity_freeze.json")
    hashes = {name: digest(module.ROOT / name) for name in historical["sha256"]}
    inputs = {}
    for rep in range(config["repetitions"]):
        directory = module._source_directory(rep)
        if read_json(directory / "result.json").get("status") != "complete":
            raise ValueError(f"Incomplete newly generated capacity job: {directory}")
        for name in ("result.json", "bank.pkl", "unit_losses.npz"):
            inputs[str((directory / name).relative_to(module.ROOT))] = digest(directory / name)
    count = len(list((module.ROOT / "results_numerical_sensitivity").glob("*/result.json")))
    if count:
        raise ValueError("Sensitivity results exist without their current execution manifest")
    value = {"version": 1, "created_system_utc": now(),
             "type": "computational reproduction: execution-linked numerical sensitivity provenance; not a scientific preregistration",
             "historical_freeze_sha256": digest(module.ROOT / "numerical_sensitivity_freeze.json"),
             "reproduction_execution_manifest_path": str(Path(execution_path).resolve()),
             "reproduction_execution_manifest_sha256_at_binding": digest(execution_path),
             "scope": "Reuses this reproduction's phase4 observations and fitted banks. The original published seeds are regenerated, not fresh inferential evidence.",
             "stabilized_study_jobs_completed_at_freeze": 0,
             "sha256": hashes, "input_sha256": inputs,
             "arithmetic_checks": module.self_tests()}
    write_json(module.FREEZE_FILE, value)
    return value


def run_sensitivity(work, execution, workers):
    module = sensitivity_module(work)
    frozen = make_sensitivity_manifest(module, execution)
    config = module._load_config()
    failures = []
    with ProcessPoolExecutor(max_workers=workers, initializer=sensitivity_initializer,
                             initargs=(str(work),)) as pool:
        futures = [pool.submit(sensitivity_job, rep, config) for rep in range(config["repetitions"])]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
            print(json.dumps({"completed": completed, "total": len(futures), **result}), flush=True)
    if failures:
        raise RuntimeError(f"Numerical sensitivity failed in {len(failures)} jobs")
    frozen = module.verify_sensitivity_freeze()
    print(json.dumps({"sensitivity_records": module.collect(config, frozen),
                      "provenance": str(module.FREEZE_FILE)}), flush=True)


def relocate_capacity_sources(module, work):
    """Resolve historical absolute paths by content, without changing any file.

    Old records were hashed when the margin was chosen. Their exact local copy
    is the only accepted relocation. Scientific calculations remain untouched.
    """
    original = module.check_margin_file

    def checked(path):
        data, changes, _ = original(path)
        checks = []
        for source in data["first_stage_sources"]:
            candidate = Path(work) / Path(source["path"]).name
            actual = digest(candidate)
            if actual != source["sha256"]:
                raise ValueError("Relocated historical margin source does not match its original hash")
            checks.append({"path": str(candidate.resolve()), "historical_absolute_path": source["path"],
                           "frozen_sha256": source["sha256"], "actual_sha256": actual,
                           "unchanged": True, "resolution": "exact-byte local relocation of historical evidence"})
        return data, changes, checks

    module.check_margin_file = checked
    return original


def prediction_audit(records, work, stage1=False, expected_records=None, expected_files=None):
    """Independent squared-residual and group-average calculation from saved arrays."""
    from collections import Counter, defaultdict
    import numpy as np
    if stage1:
        required_methods = {("raw", "ridge"), ("kahm:phi", "ridge"),
                            ("kahm:phi_logz", "ridge"), ("kahm:local", "ridge"),
                            ("affine:phi", "ridge"), ("validation_selector", "selected")}
        required = [r for r in records if (r.get("feature"), r.get("head")) in required_methods]
        coverage = Counter((r["feature"], r["head"]) for r in required)
        if set(coverage) != required_methods or any(count != 384 for count in coverage.values()):
            raise ValueError("First-stage audit requires 384 outcomes for each of its six archived prediction methods")
        if any(not r.get("unit_loss_file") or not r.get("unit_loss_key") for r in required):
            raise ValueError("A required first-stage prediction record lacks loss-file metadata")
        expected_records, expected_files = 2304, 192
    elif expected_records is not None:
        if len(records) != expected_records or any(
            not r.get("unit_loss_file") or not r.get("unit_loss_key") or not r.get("prediction_key") for r in records
        ):
            raise ValueError("Required prediction records or prediction/loss metadata are missing")
    by_file = defaultdict(list)
    for row in records:
        if row.get("unit_loss_file"):
            by_file[Path(work) / row["unit_loss_file"]].append(row)
    checked, max_array, max_mean = 0, 0., 0.
    for path, rows in by_file.items():
        with np.load(path, allow_pickle=False) as arrays:
            Y = arrays["test_Y" if stage1 else "Y"]
            groups = arrays["test_groups" if stage1 else "groups"]
            order = np.argsort(groups, kind="stable")
            ordered_groups = groups[order]
            starts = np.r_[0, np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1]
            counts = np.diff(np.r_[starts, len(order)])
            for row in rows:
                if stage1:
                    pred = arrays["prediction__" + row["feature"].replace(":", "__")]
                    target = Y
                    if row["setting_id"].startswith("branch_"):
                        column = {"0": 0, "0p5": 1, "1": 2}[row["setting_id"].rsplit("_l", 1)[1]]
                        pred, target = pred[:, column:column+1], Y[:, column:column+1]
                else:
                    pred, target = arrays[row["prediction_key"]], Y
                squared = (pred - target) ** 2
                # This independently uses sorted contiguous-group reductions;
                # the fitted head's loss function uses inverse labels/add.at.
                losses = np.add.reduceat(squared[order], starts, axis=0) / counts[:, None]
                stored = arrays[row["unit_loss_key"]]
                if losses.shape != stored.shape:
                    raise ValueError(f"Prediction-loss shape mismatch: {path}")
                array_error = float(np.max(np.abs(losses - stored)))
                mean_error = abs(float(losses.mean()) - row["test_mse"])
                if not np.allclose(losses, stored, rtol=1e-11, atol=1e-12) or mean_error > 1e-10:
                    raise ValueError(f"Prediction-loss mismatch: {path}/{row['unit_loss_key']}")
                max_array, max_mean = max(max_array, array_error), max(max_mean, mean_error)
                checked += 1
    if expected_records is not None and checked != expected_records:
        raise ValueError(f"Prediction audit checked {checked}, expected {expected_records}")
    if expected_files is not None and len(by_file) != expected_files:
        raise ValueError(f"Prediction audit checked {len(by_file)} files, expected {expected_files}")
    return {"prediction_and_loss_records_checked": checked, "files_checked": len(by_file),
            "max_absolute_unit_loss_difference": max_array,
            "max_absolute_record_mse_difference": max_mean, "mismatches": [],
            "required_records": expected_records, "required_files": expected_files,
            "method": "Independent squared residuals, stable group sorting and reduceat means; no call to the fitted-head loss function."}


def compare_records(observed, reference, key_fields):
    import numpy as np
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(str(row.get(field)) for field in key_fields)
            if key in result:
                raise ValueError(f"Duplicate comparison key: {key}")
            result[key] = row
        return result
    actual, expected = index(observed), index(reference)
    if set(actual) != set(expected):
        raise ValueError("Reproduced and archived method grids differ")
    diffs = [abs(actual[k]["test_mse"] - expected[k]["test_mse"]) for k in expected]
    within = [np.isclose(actual[k]["test_mse"], expected[k]["test_mse"], rtol=1e-6, atol=1e-10) for k in expected]
    return {"matched_records": len(diffs), "maximum_absolute_test_mse_difference": max(diffs, default=0.),
            "within_rtol_1e_minus6_atol_1e_minus10": int(sum(within)),
            "all_within_numerical_comparison_tolerance": bool(all(within)),
            "scope": "Numerical reproduction diagnostic; timings and machine-dependent metadata are not compared."}


def require_numerical_reproduction(comparisons):
    failures = [name for name in ("stage1", "capacity", "numerical_sensitivity")
                if not comparisons[name]["all_within_numerical_comparison_tolerance"]]
    if failures:
        raise ValueError("Execution outputs were preserved, but numerical reproduction exceeds the declared tolerance in: " + ", ".join(failures))


def export_mechanism(first, destination):
    rows = []
    for path in sorted((first / "results/confirmation").glob("*/result.json")):
        job = read_json(path)
        rows.append({"job_id": job["job_id"], "phase": job["phase"], "replicate": job["replicate"],
                     "bank_setting_id": job["task"]["id"],
                     "test_geometry_diagnostics": job["test_geometry_diagnostics"],
                     "mechanism_diagnostics": job["mechanism_diagnostics"]})
    if len(rows) != 168:
        raise ValueError("Expected 168 complete stage1 confirmation banks")
    write_json(destination / "mechanism_records.json", rows)


def numerical_audit(capacity, destination):
    import numpy as np
    module = sensitivity_module(capacity)
    frozen = module.verify_sensitivity_freeze()
    rows = read_json(capacity / "numerical_sensitivity_records.json")
    value = prediction_audit(rows, capacity, expected_records=96, expected_files=24)
    totals = {"query_cluster_pairs": 0, "original_exact_zero_normalizers": 0,
              "stabilized_exact_zero_normalizers": 0, "original_all_zero_gaussian_rows": 0}
    base = read_json(capacity / "config.json")["forecast"]
    for rep in range(24):
        source = read_json(module._source_directory(rep) / "result.json")
        test = module.make_forecast_split("vanderpol", base["test_episodes"], source["seeds"]["test"],
                                          history=base["history"], horizons=tuple(base["horizons"]), stride=base["stride"])
        bank = module.GeometryBank.load(module._source_directory(rep) / "bank.pkl")
        bank.transform(test["X"], diagnostics=True)
        before = bank.last_diagnostics_["weight_diagnostics"]
        with module.stabilized_queries() as calls:
            bank.transform(test["X"], diagnostics=True)
        after = bank.last_diagnostics_["weight_diagnostics"]
        totals["query_cluster_pairs"] += before["weight_rows"]
        totals["original_exact_zero_normalizers"] += before["zero_normalizers"]
        totals["stabilized_exact_zero_normalizers"] += after["zero_normalizers"]
        totals["original_all_zero_gaussian_rows"] += sum(x["original_all_zero_kernel_rows"] for x in calls)
    methods = []
    for feature in ("kahm:local", "kahm:scores"):
        for width in (64, 256):
            sub = [r for r in rows if r["feature"] == feature and r["landmarks"] == width]
            if len(sub) != 24:
                raise ValueError("Incomplete sensitivity grid")
            mean = lambda name: float(np.mean([r[name] for r in sub]))
            methods.append({"feature": feature, "landmarks": width, "replicates": len(sub),
                            "original_mean_mse": mean("original_test_mse"), "stabilized_mean_mse": mean("test_mse"),
                            "mean_original_minus_stabilized_mse": mean("original_minus_stabilized_mse"),
                            "max_absolute_replicate_mse_change": max(abs(r["original_minus_stabilized_mse"]) for r in sub),
                            "mean_raw_minus_stabilized_mse": mean("raw_minus_stabilized_mse"),
                            "mean_affine_local_minus_stabilized_mse": mean("affine_local_minus_stabilized_mse"),
                            "wins_vs_raw": sum(r["raw_minus_stabilized_mse"] > 0 for r in sub),
                            "wins_vs_affine_local": sum(r["affine_local_minus_stabilized_mse"] > 0 for r in sub)})
    value.update(scope="Descriptive post-hoc numerical sensitivity on regenerated published phase4 observations; no new confirmation.",
                 completed_jobs=24, source_files_and_input_files_verified=len(frozen["sha256"]) + len(frozen["input_sha256"]),
                 source_hash_mismatches=[], independently_recomputed_test_normalizers=totals,
                 normalizer_audit_method="Re-evaluated frozen numerical query routines on regenerated test splits; independent of archived diagnostic counts.",
                 methods=methods, sensitivity_execution_manifest_sha256=digest(module.FREEZE_FILE))
    write_json(destination / "numerical_sensitivity_audit.json", value)
    return value


def audit_export(run):
    run = Path(run).resolve()
    first, capacity, out = run / "stage1_work", run / "capacity_work", run / "results"
    activate(first)
    aa = importlib.import_module("audit_analysis")
    ac = importlib.import_module("audit_capacity")
    ac.ROOT = capacity
    stage1 = out / "stage1"
    rows, sources = aa.read_records([first / "records.json"])
    pilots, pilot_sources = aa.read_records([first / "pilot_records.json"])
    # Pilot wall-times are new on every rerun; preserve the OLD margin evidence.
    analysis_rows = [r for r in rows if r["phase"] != "pilot"] + pilots
    result1 = aa.analyze(analysis_rows, sources + pilot_sources, first / "margins.json", stage1,
                         check_units=True, study_root=first)
    first_predictions = prediction_audit([r for r in rows if r["phase"] != "pilot"], first, stage1=True)
    write_json(stage1 / "prediction_reproduction_audit.json", first_predictions)
    copy_file(first / "records.json", stage1 / "records.json")
    copy_file(first / "margins.json", stage1 / "margins.json")
    # These exact historical bytes verify already frozen margins, while fresh
    # records above continue to drive every outcome/contrast/figure.
    copy_file(ROOT / "archive/stage1/records.json.gz", stage1 / "margin_provenance_records.json.gz")
    copy_file(ROOT / "study/pilot_records.json.gz", stage1 / "margin_provenance_pilots.json.gz")
    export_mechanism(first, stage1)
    capacity_out = out / "capacity"
    cap_rows, cap_sources = aa.read_records([capacity / "capacity_records.json"])
    original = relocate_capacity_sources(ac, capacity)
    try:
        result2 = ac.analyze(cap_rows, cap_sources, capacity / "capacity_margins.json", capacity_out, check_units=True)
    finally:
        ac.check_margin_file = original
    if result2["prediction_loss_records_checked"] != 648:
        raise ValueError("Capacity audit must verify all 648 stored prediction records")
    for name in ("capacity_records.json", "capacity_margins.json"):
        copy_file(capacity / name, capacity_out / name)
    sens = out / "numerical_sensitivity"
    for name in ("numerical_sensitivity_records.json", "numerical_sensitivity_summary.json", "numerical_sensitivity_summary.csv"):
        copy_file(capacity / name, sens / name)
    # Scientific modules imported above do not load geometry, so its first import
    # can bind safely to the capacity workspace that owns these bank states.
    result3 = numerical_audit(capacity, sens)
    compare = {
        "stage1": compare_records(read_json(first / "records.json"), read_json(ROOT / "archive/stage1/records.json.gz"),
                                   ("phase", "replicate", "setting_id", "feature", "head")),
        "capacity": compare_records(read_json(capacity / "capacity_records.json"), read_json(ROOT / "archive/capacity/capacity_records.json"),
                                     ("phase", "replicate", "setting_id", "feature", "head", "landmarks")),
        "numerical_sensitivity": compare_records(read_json(capacity / "numerical_sensitivity_records.json"), read_json(ROOT / "archive/numerical_sensitivity/numerical_sensitivity_records.json"),
                                                  ("phase", "replicate", "setting_id", "feature", "head", "landmarks")),
        "pilot_margin_provenance": "Original byte-verified pilot records supply the frozen margins; rerun pilot MSEs are separately included in the stage1 comparison.",
        "statistical_scope": "Regeneration of the same published random streams; no additional independent replicates.",
    }
    write_json(run / "reproduction_comparison.json", compare)
    write_json(run / "audit_summary.json", {"stage1": result1, "capacity": result2,
                                           "stage1_predictions": first_predictions,
                                           "sensitivity_predictions_checked": result3["prediction_and_loss_records_checked"],
                                           "archive_hashes": verify_archive()})
    require_numerical_reproduction(compare)
    print(json.dumps({"audit": "complete", "comparisons": compare}), flush=True)


def archive_audit(work):
    work = Path(work).resolve()
    if not work.exists():
        materialize(work)
    activate(work)
    aa, ac = importlib.import_module("audit_analysis"), importlib.import_module("audit_capacity")
    import run_study, followup_capacity
    run_study.verify_freeze()
    followup_capacity.verify_capacity_freeze()
    rows, sources = aa.read_records([work / "records.json"])
    first = aa.analyze(rows, sources, work / "margins.json", work / "stage1", check_units=False)
    copy_file(ROOT / "archive/capacity/capacity_records.json", work / "capacity_records.json")
    cap_rows, cap_sources = aa.read_records([work / "capacity_records.json"])
    original = relocate_capacity_sources(ac, work)
    try:
        second = ac.analyze(cap_rows, cap_sources, work / "capacity_margins.json", work / "capacity", check_units=False)
    finally:
        ac.check_margin_file = original
    value = {"archive_hashes": verify_archive(), "stage1": first, "capacity": second,
             "scope": "All 29 and 12 paired contrasts recomputed from compact historical records. Full prediction auditing requires the full rerun because large states and predictions are not distributed in Git."}
    write_json(work / "archive_audit.json", value)
    print(json.dumps(value), flush=True)


def smoke(work):
    """Small, explicitly nonconfirmatory checks of all three generator families."""
    activate(work)
    import numpy as np
    import audit_analysis, audit_capacity, numerical_sensitivity, run_study, followup_capacity
    from datasets import make_forecast_split, make_branch_split, branch_targets, make_airfoil_split
    from geometry import GeometryBank
    from heads import fit_predict, grouped_losses
    run_study.verify_freeze()
    followup_capacity.verify_capacity_freeze()
    arithmetic = {"stage1": audit_analysis.selftest(), "capacity": audit_capacity.moments(np.arange(1., 25.), .05),
                  "stabilization": numerical_sensitivity.self_tests()}
    rows = []
    for index, system in enumerate(("nonlinear_ar", "vanderpol", "duffing", "branch", "airfoil")):
        if system in ("nonlinear_ar", "vanderpol", "duffing"):
            splits = [make_forecast_split(system, n, 700000 + 20*index + j) for j, n in enumerate((12, 6, 8))]
        elif system == "branch":
            splits = [make_branch_split(n, 700000 + 20*index + j) for j, n in enumerate((96, 48, 64))]
            for split in splits:
                split["Y"] = branch_targets(split, .5)[0]
        else:
            splits = [make_airfoil_split(name) for name in ("train", "validation", "test")]
            # One small grouped subset per split suffices for an execution check.
            for split in splits:
                take = np.isin(split["groups"], np.unique(split["groups"])[:5])
                for key in ("X", "Y", "groups"):
                    split[key] = split[key][take]
        bank = GeometryBank(seed=800000+index, cluster_counts=(2, 4), rank=2, max_refs=12, max_fit=180).fit(splits[0]["X"])
        transformed = [bank.transform(split["X"]) for split in splits]
        candidates = []
        for feature in ("raw", "affine:local", "kahm:local", "kahm:scores"):
            for head in ("ridge", "rbf"):
                pred, selected, fitted = fit_predict(transformed[0][feature], splits[0]["Y"],
                    transformed[1][feature], splits[1]["Y"], transformed[2][feature],
                    head=head, seed=900000+index, bounds=splits[0]["bounds"],
                    validation_groups=splits[1]["groups"], n_landmarks=8,
                    alphas=(1e-4, .01), bandwidth_multipliers=(.5, 2.))
                np.testing.assert_allclose(pred, fitted.predict(transformed[2][feature]), rtol=1e-12, atol=1e-12)
                loss = grouped_losses(pred, splits[2]["Y"], splits[2]["groups"])
                candidates.append({"phase": "smoke", "system": system, "feature": feature, "head": head,
                                   "test_mse": float(loss.mean()),
                                   "validation_mse": float(np.mean(selected[0]["validation_mse_by_target"]))})
        chosen = min(candidates, key=lambda r: r["validation_mse"])
        rows.extend(candidates)
        print(json.dumps({"smoke_system": system, "pipelines": len(candidates), "status": "pass"}), flush=True)
    write_json(Path(work) / "smoke_results.json", {"scope": "Reduced execution checks only, not research results or a reproduction of reported means.",
                                                 "arithmetic": arithmetic, "records": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sensitivity", "audit-export", "archive-audit", "smoke"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == "sensitivity": run_sensitivity(work, args.execution, args.workers)
    elif args.command == "audit-export": audit_export(work)
    elif args.command == "archive-audit": archive_audit(work)
    else: smoke(work)


if __name__ == "__main__":
    main()
