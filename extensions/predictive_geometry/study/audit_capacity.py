"""Independent analysis of the prospectively frozen, data-motivated follow-up.

This imports only the first-stage audit reader, not fitting/generation code.
The twelve-comparison family is separate from the first-stage 29-family.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

from audit_analysis import read_records, sha256_file, record_hash, write_json, write_csv, sorted_replicates


ROOT = Path(__file__).resolve().parent
SETTINGS = ("forecast_nonlinear_ar", "forecast_van_der_pol", "forecast_duffing")
WIDTHS = (64, 256)
FEATURES = ("raw", "affine:local", "kahm:local", "kahm:scores")
CONTRASTS = (("raw", "raw_minus_kahm_local"), ("affine:local", "affine_local_minus_kahm_local"))
N = 24
FAMILY_SIZE = 12
ALPHA = 0.05
FRACTION = 0.05


def versions():
    return {name: sha256_file(ROOT / name) for name in
            ("CAPACITY_ANALYSIS_SPEC.txt", "audit_capacity.py", "audit_analysis.py")}


def value(row):
    number = float(row["test_mse"])
    if not math.isfinite(number) or number < 0:
        raise ValueError("A result MSE is nonfinite or negative")
    return number


def freeze(records, sources, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite frozen margins: {output}")
    if any(row["phase"] == "capacity_confirmation" for row in records):
        raise ValueError("Margin source contains capacity-confirmation results")
    cells = {}
    for setting in SETTINGS:
        rows = [row for row in records if (row["phase"], row["setting_id"], row["feature"], row["head"])
                == ("confirmation", setting, "raw", "rbf")]
        ids = sorted_replicates({row["replicate"] for row in rows})
        if len(rows) != N or len(ids) != N:
            raise ValueError(f"{setting}: need exactly {N} first-stage raw/RBF outcomes")
        rows.sort(key=lambda row: ids.index(row["replicate"]))
        mses = [value(row) for row in rows]
        cells[setting] = {"margin": FRACTION * float(np.mean(mses)),
                          "first_stage_raw_rbf_mean_mse": float(np.mean(mses)),
                          "first_stage_mses": mses, "first_stage_replicates": ids,
                          "first_stage_record_sha256": [record_hash(row) for row in rows]}
    payload = {"format": "kahm_capacity_margins_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "data_motivated_followup": True, "confirmation_phase": "capacity_confirmation",
               "confirmation_replicates": N, "primary_family_size": FAMILY_SIZE,
               "family_alpha": ALPHA, "margin_fraction": FRACTION,
               "same_margin_at_both_widths": True, "cells": cells,
               "source_versions": versions(), "first_stage_sources": sources,
               "no_joint_95_percent_claim_with_stage1": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return {"status": "capacity_margins_frozen", "output": str(output), "sha256": sha256_file(output),
            "margins": {key: row["margin"] for key, row in cells.items()}}


def key(row):
    return (str(row["replicate"]), row["setting_id"], row["feature"], row["head"], int(row["landmarks"]))


def moments(differences, margin):
    x = np.asarray(differences, dtype=float)
    if x.shape != (N,) or not np.isfinite(x).all():
        raise ValueError("Each primary follow-up contrast needs 24 finite paired outcomes")
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(N)
    critical = float(student_t.ppf(1 - ALPHA / (2 * FAMILY_SIZE), N - 1))
    lower, upper = mean - critical * se, mean + critical * se
    classification = ("practical_benefit" if lower > margin else
                      "positive_practical_size_unresolved" if lower > 0 else
                      "harm" if upper < 0 else "direction_unresolved")
    return {"n_replicates": N, "mean_improvement": mean, "sd_improvement": sd,
            "se_improvement": se, "simultaneous_ci_lower": lower, "simultaneous_ci_upper": upper,
            "practical_margin": float(margin), "classification": classification,
            "within_prespecified_practical_band": bool(lower >= -margin and upper <= margin),
            "critical_t": critical, "degrees_of_freedom": N - 1,
            "family_size": FAMILY_SIZE, "family_alpha": ALPHA,
            "minimum_replicate_improvement": float(x.min()),
            "maximum_replicate_improvement": float(x.max()),
            "positive_replicates": int(np.sum(x > 0)), "zero_variance": sd == 0}


def check_margin_file(path):
    data = json.loads(Path(path).read_text())
    if data.get("format") != "kahm_capacity_margins_v1" or data.get("primary_family_size") != FAMILY_SIZE:
        raise ValueError("Unexpected capacity margin format or comparison family")
    if data.get("confirmation_replicates") != N or set(data["cells"]) != set(SETTINGS):
        raise ValueError("Frozen follow-up settings or replicate count changed")
    for setting, cell in data["cells"].items():
        mses = np.asarray(cell["first_stage_mses"], dtype=float)
        if mses.shape != (N,) or not np.isfinite(mses).all() or np.any(mses < 0):
            raise ValueError(f"Invalid first-stage source MSEs for {setting}")
        if not math.isclose(float(cell["margin"]), FRACTION * float(mses.mean()), rel_tol=1e-14, abs_tol=1e-15):
            raise ValueError(f"Margin arithmetic mismatch for {setting}")
    changes = [{"file": name, "frozen": expected, "current": sha256_file(ROOT / name)}
               for name, expected in data["source_versions"].items()
               if sha256_file(ROOT / name) != expected]
    source_checks = []
    for source in data["first_stage_sources"]:
        path = Path(source["path"])
        actual = sha256_file(path) if path.exists() else None
        source_checks.append({"path": str(path), "frozen_sha256": source["sha256"],
                              "actual_sha256": actual, "unchanged": actual == source["sha256"]})
    return data, changes, source_checks


def verify_predictions(rows):
    by_file = defaultdict(list)
    for row in rows:
        file = Path(row["unit_loss_file"])
        if not file.is_absolute():
            file = ROOT / file
        by_file[file.resolve()].append(row)
    checked = 0
    max_array_error = max_mean_error = 0.0
    errors, sources = [], []
    for path, file_rows in sorted(by_file.items(), key=lambda pair: str(pair[0])):
        with np.load(path, allow_pickle=False) as arrays:
            target = np.asarray(arrays["Y"], dtype=float)
            groups = arrays["groups"]
            order = np.argsort(groups, kind="stable")
            ordered = groups[order]
            starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
            counts = np.diff(np.r_[starts, len(order)])
            for row in file_rows:
                pred = np.asarray(arrays[row["prediction_key"]], dtype=float)
                stored = np.asarray(arrays[row["unit_loss_key"]], dtype=float)
                if pred.shape != target.shape:
                    raise ValueError(f"Prediction/target shape mismatch in {path}")
                squared = (pred - target)**2
                reproduced = np.add.reduceat(squared[order], starts, axis=0) / counts[:, None]
                if reproduced.shape != stored.shape or not np.isfinite(reproduced).all():
                    raise ValueError(f"Invalid group-loss shape or values in {path}")
                array_error = float(np.max(np.abs(reproduced - stored)))
                mean_error = abs(float(reproduced.mean()) - value(row))
                max_array_error = max(max_array_error, array_error)
                max_mean_error = max(max_mean_error, mean_error)
                if not np.allclose(reproduced, stored, rtol=1e-11, atol=1e-12) or not math.isclose(
                    float(reproduced.mean()), value(row), rel_tol=1e-11, abs_tol=1e-12
                ):
                    errors.append({"record_key": list(key(row)), "array_error": array_error, "mean_error": mean_error})
                checked += 1
        sources.append({"path": str(path), "sha256": sha256_file(path), "records": len(file_rows)})
    return {"records_checked": checked, "files_checked": len(by_file),
            "max_absolute_unit_loss_difference": max_array_error,
            "max_absolute_mse_difference": max_mean_error, "mismatches": errors,
            "method": "Independent squared residuals and sorted-group reduceat means from archived predictions, Y, and groups.",
            "sources": sources}


def analyze(records, sources, margins_path, out, check_units):
    margins, changes, stage1_checks = check_margin_file(margins_path)
    all_rows = [row for row in records if row["phase"] == "capacity_confirmation"]
    rows = [row for row in all_rows if row["head"] == "rbf"]
    selector_rows = [row for row in all_rows if row["head"] == "selected"
                     and row["feature"] == "validation_selector" and int(row["landmarks"]) == 0]
    if len(all_rows) != len(rows) + len(selector_rows):
        raise ValueError("Unrecognized follow-up method row")
    index = {}
    for row in rows:
        k = key(row)
        if k in index:
            raise ValueError(f"Duplicate follow-up record {k}")
        index[k] = row
    expected_ids = {str(i) for i in range(N)}
    for setting in SETTINGS:
        for feature in FEATURES:
            for width in WIDTHS:
                ids = {k[0] for k in index if k[1:] == (setting, feature, "rbf", width)}
                if ids != expected_ids:
                    raise ValueError(f"Incomplete or mismatched follow-up replicates at {setting}/{feature}/{width}")
                for rep in expected_ids:
                    value(index[(rep, setting, feature, "rbf", width)])
    if len(rows) != N * len(SETTINGS) * len(FEATURES) * len(WIDTHS):
        raise ValueError("Follow-up contains unexpected rows outside its declared 576-outcome grid")
    if len(selector_rows) != N * len(SETTINGS):
        raise ValueError("Follow-up needs exactly 72 prespecified secondary-selector rows")
    for setting in SETTINGS:
        selected_here = [row for row in selector_rows if row["setting_id"] == setting]
        if len(selected_here) != N or {row["replicate"] for row in selected_here} != expected_ids:
            raise ValueError(f"Incomplete or duplicate secondary-selector rows for {setting}")
        for row in selected_here:
            value(row)
    contrasts, pairs, summaries = [], [], []
    ids = sorted_replicates(expected_ids)
    for setting in SETTINGS:
        for width in WIDTHS:
            for feature in FEATURES:
                values = [value(index[(rep, setting, feature, "rbf", width)]) for rep in ids]
                summaries.append({"setting_id": setting, "landmarks": width, "feature": feature,
                                  "head": "rbf", "n_replicates": N, "mean_test_mse": float(np.mean(values)),
                                  "sd_test_mse": float(np.std(values, ddof=1)),
                                  "minimum_test_mse": float(np.min(values)), "maximum_test_mse": float(np.max(values))})
            candidate = [value(index[(rep, setting, "kahm:local", "rbf", width)]) for rep in ids]
            for control_feature, code in CONTRASTS:
                control = [value(index[(rep, setting, control_feature, "rbf", width)]) for rep in ids]
                diffs = np.asarray(control) - np.asarray(candidate)
                contrast_id = f"{setting}__m{width}__{code}"
                contrasts.append({"contrast_id": contrast_id, "setting_id": setting, "landmarks": width,
                                  "control_feature": control_feature, "candidate_feature": "kahm:local", "head": "rbf",
                                  "control_mean_test_mse": float(np.mean(control)),
                                  "candidate_mean_test_mse": float(np.mean(candidate)),
                                  **moments(diffs, float(margins["cells"][setting]["margin"]))})
                for rep, before, after, diff in zip(ids, control, candidate, diffs):
                    pairs.append({"contrast_id": contrast_id, "replicate": rep, "control_mse": float(before),
                                  "candidate_mse": float(after), "improvement": float(diff)})
    if len(contrasts) != FAMILY_SIZE:
        raise AssertionError("Capacity primary family changed")
    for setting in SETTINGS:
        selected_here = [value(row) for row in selector_rows if row["setting_id"] == setting]
        summaries.append({"setting_id": setting, "landmarks": 0, "feature": "validation_selector",
                          "head": "selected", "n_replicates": N,
                          "mean_test_mse": float(np.mean(selected_here)),
                          "sd_test_mse": float(np.std(selected_here, ddof=1)),
                          "minimum_test_mse": float(np.min(selected_here)),
                          "maximum_test_mse": float(np.max(selected_here))})
    unit_audit = verify_predictions(all_rows) if check_units else {"performed": False}
    if unit_audit.get("mismatches"):
        raise ValueError("Capacity prediction/loss audit failed")
    counts = defaultdict(int)
    for row in contrasts:
        counts[row["classification"]] += 1
    result = {"format": "kahm_capacity_independent_analysis_v1", "status": "complete_primary_family",
              "created_utc": datetime.now(timezone.utc).isoformat(), "data_motivated_followup": True,
              "primary_family_size": FAMILY_SIZE, "n_replicates": N, "family_alpha": ALPHA,
              "interval": "Bonferroni paired-replicate Student t; exact under a paired-normal model, otherwise approximate.",
              "no_joint_95_percent_claim_with_stage1": True, "not_distribution_free": True,
              "classification_counts": dict(counts), "source_versions": versions(),
              "source_changes_since_margin_freeze": changes,
              "stage1_source_checks": stage1_checks, "input_sources": sources,
              "frozen_margins_sha256": sha256_file(Path(margins_path)),
              "unit_loss_audit": unit_audit, "contrasts": contrasts,
              "scope": "Fresh finite-generator replicates under two specified finite Nyström budgets; not an unrestricted decoder or universal representation comparison."}
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "capacity_independent_analysis.json", result)
    write_csv(out / "capacity_primary_contrasts.csv", contrasts)
    write_csv(out / "capacity_paired_differences.csv", pairs)
    write_csv(out / "capacity_method_summary.csv", summaries)
    return {"status": result["status"], "output_dir": str(out), "contrasts": len(contrasts),
            "classification_counts": dict(counts), "source_changes": changes,
            "stage1_sources_unchanged": all(item["unchanged"] for item in stage1_checks),
            "prediction_loss_records_checked": unit_audit.get("records_checked", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze")
    p.add_argument("--stage1-records", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("analyze")
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--margins", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--check-unit-losses", action="store_true")
    commands.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        r = moments(np.arange(1.0, 25.0), 0.05)
        assert r["mean_improvement"] == 12.5 and math.isclose(r["sd_improvement"]**2, 50.0, rel_tol=1e-14)
        result = {"status": "arithmetic_selftest_passed", "family_size": FAMILY_SIZE, "critical_t": r["critical_t"]}
    elif args.command == "freeze":
        rows, sources = read_records(args.stage1_records)
        result = freeze(rows, sources, args.output)
    else:
        rows, sources = read_records(args.records)
        result = analyze(rows, sources, args.margins, args.output_dir, args.check_unit_losses)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, OSError, KeyError) as error:
        print(f"CAPACITY AUDIT ERROR: {error}", file=sys.stderr)
        sys.exit(2)
