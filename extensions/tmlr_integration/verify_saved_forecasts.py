"""Recompute the reconstruction-departure results from numeric forecast arrays.

The verifier uses NumPy and SciPy without importing any forecasting model. It
checks the complete experimental roster, training-target variance scales,
forecast errors, repetition aggregates, and six simultaneous margin contrasts.
All array loads disable pickle deserialization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
NONLINEAR_LAWS = ("nonlinear_ar", "vanderpol", "duffing")
SEEDS = tuple(range(946100, 946124))
NOISE_LEVELS = (0.0, 0.1)
HORIZONS = (1, 4, 12)
METHODS = (
    "ridge", "raw_rbf", "raw_hgb", "persistence", "common",
    "kahm_departure", "rbf_departure", "kahm_unbounded",
    "rbf_unbounded", "nonkahm_policy",
)
COMPARATORS = ("common", "rbf_departure", "raw_rbf", "raw_hgb", "nonkahm_policy")
NONKAHM_METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded")
VARIANCE_FLOOR = 1e-12
RTOL = 1e-10
ATOL = 1e-12
TABLE_KEYS = {
    "task_metrics": ("task_id", "method"),
    "horizon_metrics": ("task_id", "method", "horizon"),
    "condition_metrics": ("law", "noise_level", "method"),
    "condition_horizon_metrics": ("law", "noise_level", "method", "horizon"),
    "replicate_metrics": ("seed", "scope", "method"),
    "replicate_contrasts": ("contrast_id", "seed"),
    "primary_contrasts": ("contrast_id",),
    "overview_metrics": ("scope", "method"),
}


class VerificationError(ValueError):
    """A supplied record is incomplete or disagrees with recomputation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash array type, shape, and C-order values using the source convention."""
    array = np.ascontiguousarray(array)
    descriptor = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + array.tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise VerificationError(f"Nonfinite JSON value {value} in {path.name}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def safe_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Empty archive path")
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, f"Unsafe archive path: {relative}")
    result = root / value
    require(not result.is_symlink(), f"Symbolic links are unsupported: {relative}")
    require(result.resolve().is_relative_to(root.resolve()), f"Archive path escapes its root: {relative}")
    return result


def numeric_matrix(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    require(value.shape == shape, f"{label}: shape {value.shape} differs from {shape}")
    require(value.dtype.kind in "fiu", f"{label}: nonnumeric array")
    result = np.asarray(value, dtype=np.float64)
    require(bool(np.isfinite(result).all()), f"{label}: nonfinite values")
    return result


def assert_close(actual: float, expected: float, label: str) -> None:
    require(math.isfinite(actual) and math.isfinite(expected), f"{label}: nonfinite quantity")
    require(math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
            f"{label}: recomputed {actual:.17g}, recorded {expected:.17g}")


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "MANIFEST.json")
    require(manifest.get("schema") == "kahm-saved-forecast-manifest-v1", "Unknown manifest schema")
    entries = manifest.get("files")
    require(isinstance(entries, list) and bool(entries), "Empty file manifest")
    listed: set[str] = set()
    total_bytes = 0
    for entry in entries:
        relative = entry["path"]
        require(relative not in listed and relative != "MANIFEST.json", f"Duplicate manifest entry: {relative}")
        listed.add(relative)
        path = safe_path(root, relative)
        require(path.is_file(), f"Missing file: {relative}")
        require(path.stat().st_size == entry["bytes"], f"File size differs: {relative}")
        require(sha256(path) == entry["sha256"], f"SHA256 differs: {relative}")
        total_bytes += path.stat().st_size
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path != root / "MANIFEST.json"
    }
    require(actual == listed, f"Unlisted or missing files: {sorted(actual ^ listed)}")
    return {"files": len(entries), "bytes": total_bytes}


def verify_design(design: dict[str, Any]) -> None:
    expected = {
        "schema": "kahm-saved-forecast-study-v1",
        "laws": list(LAWS), "nonlinear_laws": list(NONLINEAR_LAWS),
        "seeds": list(SEEDS), "noise_levels": list(NOISE_LEVELS),
        "horizons": list(HORIZONS), "methods": list(METHODS),
        "training_target_rows": 512, "evaluation_origins": 1664,
        "variance_floor": VARIANCE_FLOOR, "variance_ddof": 0,
        "primary_candidate": "kahm_departure",
        "primary_comparators": list(COMPARATORS), "family_alpha": 0.05,
        "family_size": 6, "superiority_margin": 0.03,
        "linear_noninferiority_margin": 0.05,
    }
    for key, value in expected.items():
        require(design.get(key) == value, f"Study specification differs: {key}")


def compare_table(root: Path, name: str, recomputed: list[dict[str, Any]]) -> int:
    """Compare every column of an archived numerical analysis table."""
    with (root / "reported" / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        recorded = list(csv.DictReader(stream))
    require(len(recorded) == len(recomputed), f"{name}: row count differs")
    keys = TABLE_KEYS[name]
    def row_key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[key]) for key in keys)
    expected_lookup = {row_key(row): row for row in recorded}
    actual_lookup = {row_key(row): row for row in recomputed}
    require(len(expected_lookup) == len(recorded), f"{name}: duplicate recorded rows")
    require(len(actual_lookup) == len(recomputed), f"{name}: duplicate computed rows")
    require(set(expected_lookup) == set(actual_lookup), f"{name}: row identities differ")
    checks = 0
    for identity, row in actual_lookup.items():
        saved = expected_lookup[identity]
        require(set(saved) == set(row), f"{name}: column names differ")
        for column, value in row.items():
            label = f"{name}/{identity}/{column}"
            if isinstance(value, bool):
                require(saved[column] == str(value), f"{label}: Boolean differs")
            elif isinstance(value, (int, np.integer)):
                require(float(saved[column]) == int(value), f"{label}: integer differs")
            elif isinstance(value, (float, np.floating)):
                assert_close(float(value), float(saved[column]), label)
            elif value is None:
                require(saved[column] == "", f"{label}: expected an empty value")
            else:
                require(saved[column] == str(value), f"{label}: text differs")
            checks += 1
    return checks


def recompute(root: Path) -> dict[str, list[dict[str, Any]]]:
    design = read_json(root / "study.json")
    verify_design(design)
    roster = read_json(root / "tasks.json")
    require(isinstance(roster, list), "Task roster must be a list")
    expected = set(itertools.product(LAWS, NOISE_LEVELS, SEEDS))
    seen: set[tuple[str, float, int]] = set()
    task_ids: set[str] = set()
    scores: dict[tuple[str, float, int, str], dict[str, Any]] = {}
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_KEYS}
    for task in roster:
        law, noise, seed = task["law"], task["noise_level"], task["seed"]
        key = (law, noise, seed)
        require(key in expected and key not in seen, f"Unexpected or repeated task: {key}")
        seen.add(key)
        task_id = task["task_id"]
        require(task_id not in task_ids, f"Repeated task identifier: {task_id}")
        task_ids.add(task_id)
        require(task["repetition"] == SEEDS.index(seed), f"Repetition differs: {task_id}")
        calibration = safe_path(root, task["normalization_file"])
        with np.load(calibration, allow_pickle=False) as arrays:
            require(set(arrays.files) == {"A_Y", "loss_scale"}, f"{task_id}: calibration array roster differs")
            a_original, scale_original = arrays["A_Y"], arrays["loss_scale"]
            require(array_sha256(a_original) == task["source_arrays_sha256"]["A_Y"], f"{task_id}: A target digest differs")
            require(array_sha256(scale_original) == task["source_arrays_sha256"]["loss_scale"], f"{task_id}: scale digest differs")
            a = numeric_matrix(a_original, (512, 3), f"{task_id}/A_Y")
            saved_scale = numeric_matrix(scale_original, (3,), f"{task_id}/loss_scale")
        variance = np.var(a, axis=0, ddof=0)
        scale = np.maximum(variance, VARIANCE_FLOOR)
        for column in range(3):
            assert_close(float(scale[column]), float(saved_scale[column]), f"{task_id}/training variance/{column}")
        path = safe_path(root, task["forecast_file"])
        require(sha256(path) == task["source_prediction_file_sha256"], f"{task_id}: prediction archive changed")
        with np.load(path, allow_pickle=False) as arrays:
            require(set(arrays.files) == {"Y", *METHODS}, f"{task_id}: forecast array roster differs")
            require(array_sha256(arrays["Y"]) == task["source_arrays_sha256"]["D_Y"], f"{task_id}: D target digest differs")
            y = numeric_matrix(arrays["Y"], (1664, 3), f"{task_id}/Y")
            predictions = {}
            for method in METHODS:
                original = arrays[method]
                require(array_sha256(original) == task["source_prediction_arrays_sha256"][method], f"{task_id}/{method}: source-array digest differs")
                predictions[method] = numeric_matrix(original, y.shape, f"{task_id}/{method}")
        selected = task["selected_policy_method"]
        require(selected in NONKAHM_METHODS, f"{task_id}: invalid non-KAHM selection")
        require(np.array_equal(predictions["nonkahm_policy"], predictions[selected]), f"{task_id}: policy forecast differs from selected method")
        metadata = {"task_id": task_id, "law": law, "noise_level": noise,
                    "seed": seed, "repetition": task["repetition"]}
        for method, prediction in predictions.items():
            error = prediction - y
            with np.errstate(over="raise", invalid="raise"):
                mse = np.mean(error * error, axis=0)
                nmse = mse / scale
            require(bool(np.isfinite(nmse).all()), f"{task_id}/{method}: nonfinite loss")
            record = {"nmse": float(np.mean(nmse)), "raw_mse": float(np.mean(mse)),
                      "max_abs_error": float(np.max(np.abs(error))),
                      "horizon_nmse": nmse, "horizon_raw_mse": mse}
            scores[(law, noise, seed, method)] = record
            tables["task_metrics"].append({
                **metadata, "method": method,
                **{k: record[k] for k in ("nmse", "raw_mse", "max_abs_error")},
                "n_origins": 1664, "n_horizons": 3, "n_target_cells": 4992,
                "selected_policy_method": selected if method == "nonkahm_policy" else "",
            })
            for column, horizon in enumerate(HORIZONS):
                tables["horizon_metrics"].append({
                    **metadata, "method": method, "horizon": horizon,
                    "raw_mse": float(mse[column]), "nmse": float(nmse[column]),
                    "A_target_variance": float(variance[column]), "loss_scale": float(scale[column]),
                    "variance_floor_used": bool(variance[column] < VARIANCE_FLOOR),
                    "max_abs_error": float(np.max(np.abs(error[:, column]))), "n_origins": 1664,
                })
    require(seen == expected, f"Incomplete task roster: {len(seen)} of {len(expected)}")

    for law, noise, method in itertools.product(LAWS, NOISE_LEVELS, METHODS):
        records = [scores[(law, noise, seed, method)] for seed in SEEDS]
        values = np.array([record["nmse"] for record in records])
        tables["condition_metrics"].append({
            "law": law, "noise_level": noise, "method": method,
            "mean_nmse": float(values.mean()), "sample_sd_nmse": float(values.std(ddof=1)),
            "mean_raw_mse": float(np.mean([record["raw_mse"] for record in records])),
            "n_repetitions": 24, "inference": "descriptive",
        })
        for column, horizon in enumerate(HORIZONS):
            values = np.array([record["horizon_nmse"][column] for record in records])
            tables["condition_horizon_metrics"].append({
                "law": law, "noise_level": noise, "method": method, "horizon": horizon,
                "mean_nmse": float(values.mean()), "sample_sd_nmse": float(values.std(ddof=1)),
                "mean_raw_mse": float(np.mean([record["horizon_raw_mse"][column] for record in records])),
                "n_repetitions": 24, "inference": "descriptive",
            })
    repetitions: dict[tuple[str, str], np.ndarray] = {}
    for scope, laws in (("nonlinear", NONLINEAR_LAWS), ("linear", ("linear_ar",)), ("all_descriptive", LAWS)):
        for method in METHODS:
            values = []
            for seed in SEEDS:
                value = float(np.mean([scores[(law, noise, seed, method)]["nmse"]
                                       for law, noise in itertools.product(laws, NOISE_LEVELS)]))
                values.append(value)
                tables["replicate_metrics"].append({
                    "seed": seed, "scope": scope, "method": method, "nmse": value,
                    "n_conditions": 2 * len(laws), "equal_horizon_weight": True,
                    "equal_condition_weight": True,
                })
            values_array = np.asarray(values)
            repetitions[(scope, method)] = values_array
            tables["overview_metrics"].append({
                "scope": scope, "method": method, "mean_nmse": float(values_array.mean()),
                "sample_sd_nmse": float(values_array.std(ddof=1)), "n_repetitions": 24,
            })
    cases = [("nonlinear", method, 0.97, 0.03, f"nonlinear_vs_{method}") for method in COMPARATORS]
    cases.append(("linear", "ridge", 1.05, 0.05, "linear_noninferiority_vs_ridge"))
    critical = float(student_t.ppf(1.0 - 0.05 / 6, 23))
    for scope, method, factor, margin, name in cases:
        candidate = repetitions[(scope, "kahm_departure")]
        comparator = repetitions[(scope, method)]
        gaps = factor * comparator - candidate
        mean = float(gaps.mean())
        sd = float(gaps.std(ddof=1))
        se = sd / math.sqrt(24)
        lower = mean - critical * se
        p_value = float(student_t.sf(mean / se, 23)) if se else (0.0 if mean > 0 else (1.0 if mean < 0 else 0.5))
        candidate_mean, comparator_mean = float(candidate.mean()), float(comparator.mean())
        tables["primary_contrasts"].append({
            "contrast_id": name, "scope": scope, "candidate": "kahm_departure", "comparator": method,
            "n_repetitions": 24, "df": 23, "margin": margin, "comparator_factor": factor,
            "candidate_mean_nmse": candidate_mean, "comparator_mean_nmse": comparator_mean,
            "point_relative_gain": 1 - candidate_mean / comparator_mean if comparator_mean > 0 else None,
            "margin_gap_mean": mean, "margin_gap_sample_sd": sd, "margin_gap_se": se,
            "critical_t": critical, "simultaneous_one_sided_lower": lower,
            "family_alpha": 0.05, "family_size": 6, "p_one_sided": p_value,
            "p_bonferroni": min(1.0, 6 * p_value),
            "positive_margin_repetitions": int(np.sum(gaps > 0)),
            "criterion_established": bool(lower > 0),
            "criterion": "useful_superiority" if scope == "nonlinear" else "linear_noninferiority",
        })
        for index, seed in enumerate(SEEDS):
            tables["replicate_contrasts"].append({
                "contrast_id": name, "scope": scope, "seed": seed,
                "candidate": "kahm_departure", "comparator": method,
                "candidate_nmse": float(candidate[index]), "comparator_nmse": float(comparator[index]),
                "comparator_factor": factor, "margin_gap": float(gaps[index]),
            })
    return tables


def verify(root: Path, write_results: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    integrity = verify_manifest(root)
    tables = recompute(root)
    checks = sum(compare_table(root, name, rows) for name, rows in tables.items())
    criteria = tables["primary_contrasts"]
    established = [row["contrast_id"] for row in criteria if row["criterion_established"]]
    reported = read_json(root / "reported" / "statistical_result.json")
    require(reported["task_suites"] == 192 and reported["independent_repetitions"] == 24, "Reported design count differs")
    require(reported["established_contrasts"] == established, "Reported established comparisons differ")
    require(reported["all_primary_criteria_established"] == (len(established) == 6), "Reported statistical conclusion differs")
    result = {
        "verification": "passed", "manifest_files_checked": integrity["files"],
        "manifest_bytes_checked": integrity["bytes"], "task_suites": 192,
        "forecast_arrays": 1920, "forecast_values": 192 * 10 * 1664 * 3,
        "independent_repetitions": 24, "reported_table_cells_checked": checks,
        "reported_table_rows": {name: len(rows) for name, rows in tables.items()},
        "established_contrasts": established,
        "all_primary_criteria_established": len(established) == 6,
        "primary_contrasts": criteria,
        "scope": "Saved numeric forecasts and statistical analysis; model fitting, selection, feature bounds, and timing are not rerun.",
    }
    if write_results is not None:
        write_results = write_results.resolve()
        require(not write_results.exists(), "The result output directory must be new")
        require(not write_results.is_relative_to(root), "Write recomputed results outside the evidence directory")
        write_results.mkdir(parents=True)
        for name, rows in tables.items():
            with (write_results / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (write_results / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing MANIFEST.json, study.json, forecasts, and normalization arrays")
    parser.add_argument("--write-results", type=Path, help="New directory outside the evidence bundle for recomputed CSV tables")
    arguments = parser.parse_args()
    try:
        result = verify(arguments.evidence_dir, arguments.write_results)
    except (VerificationError, OSError, ValueError, KeyError, FloatingPointError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
