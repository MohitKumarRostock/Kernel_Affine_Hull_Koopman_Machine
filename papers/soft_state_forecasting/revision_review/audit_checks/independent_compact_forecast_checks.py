#!/usr/bin/env python3
"""Independently recompute six primary contrasts from the compact supplement.

Only NumPy/SciPy and the Python standard library are used. No supplied model,
analysis, or verification module is imported. This compact audit reconstructs
the primary statistics from numeric forecasts and A targets. The broader
independent_saved_forecast_checks.py additionally checks the full input/window
and recorded-selection/resource evidence from the pinned public transport.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import t

LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
SEEDS = tuple(range(946100, 946124))
NOISES = (0.0, 0.1)
METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "kahm_departure",
           "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy")
POLICY = {"ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded"}


def mean(values):
    values = list(values)
    return math.fsum(float(x) for x in values) / len(values)


def digest(array):
    array = np.ascontiguousarray(array)
    metadata = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(metadata + b"\0" + array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True,
                        help="Extracted saved_forecast_supplement directory containing MANIFEST.json")
    parser.add_argument("--output", type=Path, required=True, help="JSON result outside the evidence directory")
    args = parser.parse_args()
    root, output = args.evidence_dir.resolve(), args.output.resolve()
    assert not output.is_relative_to(root), "Write results outside the immutable evidence bundle"

    def safe(relative):
        path = root / relative
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert path.resolve().is_relative_to(root) and not path.is_symlink()
        return path

    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert manifest["schema"] == "kahm-saved-forecast-manifest-v1"
    names = [row["path"] for row in manifest["files"]]
    assert len(set(names)) == len(names) == 399
    for row in manifest["files"]:
        data = safe(row["path"]).read_bytes()
        assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts} == set(names) | {"MANIFEST.json"}
    tasks = json.loads((root / "tasks.json").read_text())
    roster = {(x["law"], x["noise_level"], x["seed"]) for x in tasks}
    assert len(tasks) == len(roster) == 192 and roster == set(itertools.product(LAWS, NOISES, SEEDS))
    losses, normalization_maxdiff = {}, 0.0
    for task in tasks:
        key = (task["law"], task["noise_level"], task["seed"])
        with np.load(safe(task["normalization_file"]), allow_pickle=False) as a:
            targets, scale_record = a["A_Y"], a["loss_scale"]
        assert targets.shape == (512, 3) and scale_record.shape == (3,)
        assert digest(targets) == task["source_arrays_sha256"]["A_Y"]
        assert digest(scale_record) == task["source_arrays_sha256"]["loss_scale"]
        target_means = [mean(targets[:, h]) for h in range(3)]
        scales = [max(mean((float(y) - target_means[h]) ** 2 for y in targets[:, h]), 1e-12) for h in range(3)]
        normalization_maxdiff = max(normalization_maxdiff, max(abs(float(a) - b) for a, b in zip(scale_record, scales)))
        assert np.allclose(scale_record, scales, rtol=1e-10, atol=1e-12)
        prediction_path = safe(task["forecast_file"])
        assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == task["source_prediction_file_sha256"]
        with np.load(prediction_path, allow_pickle=False) as a:
            assert set(a.files) == {"Y", *METHODS}
            y = a["Y"]
            assert y.shape == (1664, 3) and np.isfinite(y).all()
            assert digest(y) == task["source_arrays_sha256"]["D_Y"]
            selected = task["selected_policy_method"]
            assert selected in POLICY and np.array_equal(a["nonkahm_policy"], a[selected])
            for method in METHODS:
                prediction = a[method]
                assert prediction.shape == y.shape and np.isfinite(prediction).all()
                assert digest(prediction) == task["source_prediction_arrays_sha256"][method]
                error = prediction - y
                losses[key + (method,)] = mean(mean(float(v) ** 2 for v in error[:, h]) / scales[h] for h in range(3))
    rep = {}
    for scope, laws in (("nonlinear", LAWS[1:]), ("linear", ("linear_ar",))):
        for method in METHODS:
            rep[(scope, method)] = [mean(losses[(law, noise, seed, method)] for law in laws for noise in NOISES) for seed in SEEDS]
    critical = float(t.isf(0.05 / 6, 23))
    calculated = []
    for scope, method in [("nonlinear", m) for m in ("common", "rbf_departure", "raw_rbf", "raw_hgb", "nonkahm_policy")] + [("linear", "ridge")]:
        factor = 0.97 if scope == "nonlinear" else 1.05
        candidate, comparator = rep[(scope, "kahm_departure")], rep[(scope, method)]
        gaps = [factor * b - a for a, b in zip(candidate, comparator)]
        center = mean(gaps)
        sd = math.sqrt(math.fsum((v - center) ** 2 for v in gaps) / 23)
        se = sd / math.sqrt(24)
        lower = center - critical * se
        p = float(t.sf(center / se, 23))
        calculated.append({
            "contrast_id": "nonlinear_vs_" + method if scope == "nonlinear" else "linear_noninferiority_vs_ridge",
            "candidate_mean_nmse": mean(candidate), "comparator_mean_nmse": mean(comparator),
            "point_relative_gain": 1 - mean(candidate) / mean(comparator), "margin_gap_mean": center,
            "margin_gap_sample_sd": sd, "margin_gap_se": se, "critical_t": critical,
            "simultaneous_one_sided_lower": lower, "p_one_sided": p, "p_bonferroni": min(1., 6 * p),
            "positive_margin_repetitions": sum(v > 0 for v in gaps), "criterion_established": lower > 0,
        })
    with (root / "reported/primary_contrasts.csv").open(newline="") as f:
        recorded = {row["contrast_id"]: row for row in csv.DictReader(f)}
    assert set(recorded) == {row["contrast_id"] for row in calculated}
    maxdiff, numeric_checks = 0., 0
    for row in calculated:
        other = recorded[row["contrast_id"]]
        for field, value in row.items():
            if isinstance(value, bool):
                assert str(value) == other[field]
            elif isinstance(value, str):
                assert value == other[field]
            else:
                delta = abs(float(other[field]) - value)
                maxdiff = max(maxdiff, delta)
                assert math.isclose(float(other[field]), value, rel_tol=1e-10, abs_tol=1e-12), (field, other[field], value)
                numeric_checks += 1
    result = {"status": "passed", "scope": "Independent arithmetic for the six original primary contrasts from the compact numeric supplement; no fitting, model-state replay, feature-bound or resource reproduction.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "manifest_files_checked": 399, "manifest_bytes_checked": sum(row["bytes"] for row in manifest["files"]),
        "task_suites": 192, "forecast_arrays": 1920, "forecast_values": 9584640,
        "independent_repetition_aggregates_per_contrast": 24, "selected_policy_forecast_identities_checked": 192,
        "normalization_maximum_absolute_difference": normalization_maxdiff,
        "primary_numeric_fields_checked": numeric_checks, "primary_maximum_absolute_difference": maxdiff,
        "established_contrasts": [row["contrast_id"] for row in calculated if row["criterion_established"]],
        "primary_contrasts": calculated}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "primary_contrasts"}, indent=2))


if __name__ == "__main__":
    main()
