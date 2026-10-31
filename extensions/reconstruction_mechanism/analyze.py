"""Independent analysis of the frozen reconstruction-departure experiment.

This module does not import the forecasting implementation. It reads the entire
declared roster, validates saved targets and predictions, and recomputes losses
and the six prespecified paired-repetition contrasts. No successful subset is
used for primary inference. See PROTOCOL.md for the scope of the fixed decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


SCHEMA = "reconstruction-mechanism-analysis-v1"
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
NONKAHM_METHODS = (
    "ridge", "raw_rbf", "raw_hgb", "persistence", "common",
    "rbf_departure", "rbf_unbounded",
)
PRIMARY_CANDIDATE = "kahm_departure"
PRIMARY_COMPARATORS = (
    "common", "rbf_departure", "raw_rbf", "raw_hgb", "nonkahm_policy",
)
FAMILY_ALPHA = 0.05
SUPERIORITY_MARGIN = 0.03
LINEAR_NONINFERIORITY_MARGIN = 0.05
VARIANCE_FLOOR = 1e-12
MAX_ERROR_RATIO = 5.0
BOUND_TOLERANCE = 1e-10
GAMMA_GRID = (0.0, 0.25, 0.5, 1.0)
EXPECTED_A_ROWS = 512
EXPECTED_D_ROWS = 1664


class InvalidEvidence(ValueError):
    """Evidence is incomplete or inconsistent with the frozen study."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise InvalidEvidence(f"Non-finite JSON constant {value} in {path}")

    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_path(path: Path, study_dir: Path) -> str:
    try:
        return path.resolve().relative_to(study_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def input_path(study_dir: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise InvalidEvidence("Task input must be a nonempty study-relative path")
    candidate = (study_dir / value).resolve()
    try:
        candidate.relative_to(study_dir.resolve())
    except ValueError as error:
        raise InvalidEvidence(f"Input path escapes study directory: {value}") from error
    if Path(value).is_absolute():
        raise InvalidEvidence(f"Task input must be relative: {value}")
    return candidate


def _configuration_value(config: dict[str, Any], name: str,
                         *aliases: str) -> Any:
    """Allow documented field placement, never conflicting duplicate values."""
    found = []
    for mapping in (config, config.get("analysis", {})):
        if not isinstance(mapping, dict):
            raise InvalidEvidence("config.analysis must be an object")
        for key in (name, *aliases):
            if key in mapping:
                found.append(mapping[key])
    if not found:
        raise InvalidEvidence(f"Missing frozen configuration field: {name}")
    if any(value != found[0] for value in found[1:]):
        raise InvalidEvidence(f"Conflicting frozen configuration field: {name}")
    return found[0]


def validate_config(config: dict[str, Any]) -> None:
    """Reject a different scientific design rather than silently changing it."""
    if not isinstance(config, dict):
        raise InvalidEvidence("Frozen configuration must be an object")
    expected = {
        "laws": list(LAWS), "seeds": list(SEEDS),
        "noise_levels": list(NOISE_LEVELS), "horizons": list(HORIZONS),
        "methods": list(METHODS), "primary_candidate": PRIMARY_CANDIDATE,
        "primary_comparators": list(PRIMARY_COMPARATORS),
        "family_alpha": FAMILY_ALPHA,
        "superiority_margin": SUPERIORITY_MARGIN,
        "linear_noninferiority_margin": LINEAR_NONINFERIORITY_MARGIN,
        "variance_floor": VARIANCE_FLOOR,
        "max_error_ratio": MAX_ERROR_RATIO,
    }
    aliases = {"horizons": ("offsets",),
               "primary_comparators": ("primary_controls",)}
    for name, wanted in expected.items():
        actual = _configuration_value(config, name, *aliases.get(name, ()))
        if actual != wanted:
            raise InvalidEvidence(
                f"Frozen {name} differs from the analysis protocol: "
                f"{actual!r} != {wanted!r}"
            )
    if "nonlinear_laws" in config and config["nonlinear_laws"] != list(NONLINEAR_LAWS):
        raise InvalidEvidence("Nonlinear law roster differs from protocol")
    if "family_size" in config.get("analysis", {}):
        if config["analysis"]["family_size"] != 6:
            raise InvalidEvidence("Primary family must contain exactly six contrasts")


def verify_freeze(study_dir: Path) -> list[dict[str, str]]:
    """Verify the local file freeze without importing the experiment runner."""
    freeze_path = study_dir / "freeze.json"
    freeze = read_json(freeze_path)
    files = freeze.get("files")
    if not isinstance(files, list) or not files:
        raise InvalidEvidence("Local freeze must contain a nonempty files list")
    repository = study_dir.parent.parent.resolve()
    checked = []
    paths = set()
    for record in files:
        if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
            raise InvalidEvidence("Invalid local freeze file record")
        relative = record["path"]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise InvalidEvidence("Freeze paths must be repository-relative")
        path = (repository / relative).resolve()
        try:
            path.relative_to(repository)
        except ValueError as error:
            raise InvalidEvidence(f"Frozen path escapes repository: {relative}") from error
        if relative in paths:
            raise InvalidEvidence(f"Repeated frozen file: {relative}")
        paths.add(relative)
        actual = sha256(path)
        if actual != record["sha256"]:
            raise InvalidEvidence(f"Frozen file changed: {relative}")
        checked.append({"path": relative_path(path, study_dir), "sha256": actual})
    required = [study_dir / name for name in
                ("config.json", "PROTOCOL.md", "analyze.py", "prepared_inputs/manifest.json")]
    for path in required:
        relative = path.relative_to(repository).as_posix()
        if relative not in paths:
            raise InvalidEvidence(f"Required scientific file omitted from freeze: {relative}")
    checked.append({"path": "freeze.json", "sha256": sha256(freeze_path)})
    return checked


def validate_roster(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("development_only", False):
        raise InvalidEvidence("Evaluation manifest must be a non-development object")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise InvalidEvidence("Prepared manifest must contain a tasks list")
    expected = set(itertools.product(LAWS, NOISE_LEVELS, SEEDS))
    found: set[tuple[str, float, int]] = set()
    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise InvalidEvidence("Manifest task is not an object")
        task_id = task.get("task_id")
        if (not isinstance(task_id, str) or not task_id
                or task_id in task_ids or Path(task_id).name != task_id
                or task_id in (".", "..")):
            raise InvalidEvidence(f"Invalid or repeated task identifier: {task_id!r}")
        if isinstance(task.get("seed"), bool):
            raise InvalidEvidence(f"Boolean seed for {task_id}")
        seed = int(task["seed"])
        if seed != task["seed"]:
            raise InvalidEvidence(f"Nonintegral seed for {task_id}")
        key = (str(task["law"]), float(task["noise_level"]), seed)
        if key not in expected or key in found:
            raise InvalidEvidence(f"Unexpected or repeated law/noise/seed: {key}")
        if "input" not in task or "loss_scale" not in task:
            raise InvalidEvidence(f"Missing input or normalization metadata for {task_id}")
        if task.get("development_only", False):
            raise InvalidEvidence(f"Development-only task in evaluation roster: {task_id}")
        found.add(key)
        task_ids.add(task_id)
    if found != expected:
        missing = sorted(expected - found)
        raise InvalidEvidence(f"Incomplete roster: {len(found)}/{len(expected)}; missing {missing}")
    return sorted(tasks, key=lambda row: (
        SEEDS.index(int(row["seed"])), LAWS.index(row["law"]),
        NOISE_LEVELS.index(float(row["noise_level"])),
    ))


def finite_matrix(value: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise InvalidEvidence(f"{label}: shape {array.shape} != {shape}")
    if not np.isfinite(array).all():
        raise InvalidEvidence(f"{label}: non-finite values")
    return array


def flatten_scalars(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows = []
        for key in sorted(value):
            rows.extend(flatten_scalars(value[key], f"{prefix}.{key}" if prefix else key))
        return rows
    if isinstance(value, list):
        rows = []
        for index, element in enumerate(value):
            rows.extend(flatten_scalars(element, f"{prefix}[{index}]"))
        return rows
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidEvidence(f"Non-finite diagnostic/resource value at {prefix}")
        return [(prefix, value)]
    raise InvalidEvidence(f"Unsupported diagnostic/resource value at {prefix}")


def _meta(task: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": task["task_id"], "law": task["law"],
            "noise_level": float(task["noise_level"]), "seed": int(task["seed"]),
            "repetition": task.get("repetition", SEEDS.index(int(task["seed"])))}


def _gamma_map(fit: dict[str, Any], selected: str) -> dict[str, list[float] | None]:
    saved = fit.get("gamma_by_horizon")
    if not isinstance(saved, dict):
        raise InvalidEvidence("Missing fit gamma_by_horizon map")
    result: dict[str, list[float] | None] = {"ridge": None}
    for method in METHODS:
        if method in ("ridge", "nonkahm_policy"):
            continue
        values = np.asarray(saved.get(method, []), dtype=float)
        if values.shape != (len(HORIZONS),) or not np.isfinite(values).all():
            raise InvalidEvidence(f"Invalid C-selected gamma for {method}")
        if any(float(value) not in GAMMA_GRID for value in values):
            raise InvalidEvidence(f"Undeclared C-selected gamma for {method}")
        result[method] = values.tolist()
    result["nonkahm_policy"] = result[selected]
    return result


def _bound_rows(diagnostics: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    bounds = diagnostics.get("feature_bounds")
    if not isinstance(bounds, dict):
        raise InvalidEvidence("Missing reserved-prediction feature-bound diagnostics")
    rows = []
    for block in ("common", "kahm_departure", "rbf_departure"):
        info = bounds.get(block)
        if not isinstance(info, dict):
            raise InvalidEvidence(f"Missing bounded block diagnostics: {block}")
        for field in ("finite", "max_norm", "radius", "violations"):
            if field not in info:
                raise InvalidEvidence(f"Missing {block} bound field: {field}")
        if not isinstance(info["finite"], bool):
            raise InvalidEvidence(f"Non-Boolean finite flag for {block}")
        maximum, radius = float(info["max_norm"]), float(info["radius"])
        violations = int(info["violations"])
        width = int(info.get("width", 0))
        if (not math.isfinite(maximum) or not math.isfinite(radius)
                or maximum < 0 or radius <= 0 or violations < 0
                or violations != info["violations"] or width <= 0
                or width != info.get("width") or info.get("rows") != EXPECTED_D_ROWS
                or not math.isclose(radius, math.sqrt(width), rel_tol=1e-12, abs_tol=0)):
            raise InvalidEvidence(f"Invalid norm-bound values for {block}")
        check = bool(info["finite"] and violations == 0
                     and maximum <= radius * (1 + BOUND_TOLERANCE))
        rows.append({**metadata, "block": block, "finite": info["finite"], "width": width,
                     "max_norm": maximum, "radius": radius,
                     "violations": violations, "bound_pass": check,
                     "evidence": "model-reported norms; consistency checked independently"})
    return rows


def _readout_rows(diagnostics: dict[str, Any], metadata: dict[str, Any],
                  predictions: dict[str, np.ndarray],
                  gamma: dict[str, list[float] | None]) -> list[dict[str, Any]]:
    """Check reported fitted-state bounds against actual saved D corrections."""
    summaries = diagnostics.get("bounds")
    if not isinstance(summaries, dict):
        raise InvalidEvidence("Missing readout-bound diagnostics")
    rows = []
    for method in ("common", "kahm_departure", "rbf_departure"):
        info = summaries.get(method)
        if not isinstance(info, dict) or info.get("global_bound") is not True:
            raise InvalidEvidence(f"Missing bounded readout diagnostics: {method}")
        arrays = {}
        for name in ("response_mean_by_horizon", "centered_response_rms_by_horizon",
                     "alpha_by_horizon", "coefficient_norm_by_horizon",
                     "coefficient_norm_bound_by_horizon", "readout_bound_by_horizon",
                     "correction_bound_by_horizon", "max_abs_correction_by_horizon"):
            value = np.asarray(info.get(name, []), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise InvalidEvidence(f"Invalid {method} readout-bound field: {name}")
            arrays[name] = value
        alpha = arrays["alpha_by_horizon"]
        rms = arrays["centered_response_rms_by_horizon"]
        coefficient_norm = arrays["coefficient_norm_by_horizon"]
        radius, mean_norm = float(info["radius"]), float(info["training_feature_mean_norm"])
        common_radius = float(diagnostics["feature_bounds"]["common"]["radius"])
        expected_radius = common_radius if method == "common" else math.sqrt(
            common_radius**2 + float(diagnostics["feature_bounds"][method]["radius"])**2)
        if (np.any(alpha <= 0) or np.any(rms < 0) or np.any(coefficient_norm < 0)
                or not math.isfinite(radius) or not math.isfinite(mean_norm)
                or radius <= 0 or mean_norm < 0
                or not math.isclose(radius, expected_radius, rel_tol=1e-12, abs_tol=0)):
            raise InvalidEvidence(f"Invalid nonnegative readout-bound parameters: {method}")
        coefficient_bound = rms / (2 * np.sqrt(alpha))
        readout_bound = np.abs(arrays["response_mean_by_horizon"]) + (radius + mean_norm) * coefficient_bound
        correction_bound = np.asarray(gamma[method]) * readout_bound
        actual = np.max(np.abs(predictions[method] - predictions["ridge"]), axis=0)
        for name, expected in (("coefficient_norm_bound_by_horizon", coefficient_bound),
                               ("readout_bound_by_horizon", readout_bound),
                               ("correction_bound_by_horizon", correction_bound),
                               ("max_abs_correction_by_horizon", actual)):
            if not np.allclose(arrays[name], expected, rtol=1e-10, atol=1e-12):
                raise InvalidEvidence(f"Inconsistent {method} readout-bound field: {name}")
        coefficient_ok = coefficient_norm <= coefficient_bound * (1 + 1e-10) + 1e-12
        correction_ok = actual <= correction_bound * (1 + 1e-10) + 1e-12
        if (info.get("coefficient_bound_violations") != int(np.sum(~coefficient_ok))
                or info.get("violations") != int(np.sum(~correction_ok))):
            raise InvalidEvidence(f"Inconsistent {method} readout violation counts")
        for j, horizon in enumerate(HORIZONS):
            rows.append({**metadata, "method": method, "horizon": horizon,
                         "feature_radius": radius, "training_feature_mean_norm": mean_norm,
                         "alpha": float(alpha[j]), "gamma": gamma[method][j],
                         "coefficient_norm": float(coefficient_norm[j]),
                         "coefficient_norm_bound": float(coefficient_bound[j]),
                         "max_abs_correction": float(actual[j]),
                         "correction_bound": float(correction_bound[j]),
                         "coefficient_bound_pass": bool(coefficient_ok[j]),
                         "correction_bound_pass": bool(correction_ok[j]),
                         "training_mean_bound_pass": mean_norm <= radius * (1 + 1e-10) + 1e-12})
    return rows


def load_task(study_dir: Path, results_dir: Path,
              task: dict[str, Any]) -> dict[str, Any]:
    metadata = _meta(task)
    prepared = input_path(study_dir, task["input"])
    directory = results_dir / task["task_id"]
    prediction_file = directory / "predictions.npz"
    summary_file = directory / "summary.json"
    summary = read_json(summary_file)
    if not isinstance(summary, dict):
        raise InvalidEvidence("Task summary must be an object")
    if summary.get("status") != "completed":
        raise InvalidEvidence(f"Task has status {summary.get('status')!r}")
    saved_task = summary.get("task")
    if not isinstance(saved_task, dict):
        raise InvalidEvidence("Task summary lacks identity metadata")
    for key in ("task_id", "law", "seed", "noise_level"):
        if saved_task.get(key) != task[key]:
            raise InvalidEvidence(f"Summary task identity mismatch: {key}")
    digest = sha256(prepared)
    if summary.get("development_only", False):
        raise InvalidEvidence("Development result in evaluation evidence")
    if summary.get("input_sha256") != digest:
        raise InvalidEvidence("Summary input identity differs from prepared input")
    if summary.get("freeze_sha256") != sha256(study_dir / "freeze.json"):
        raise InvalidEvidence("Task was not executed under the current exact freeze")
    for key in ("sha256", "input_sha256", "prepared_sha256"):
        if key in task and task[key] != digest:
            raise InvalidEvidence(f"Prepared input hash mismatch: {key}")
    with np.load(prepared, allow_pickle=False) as data:
        train_targets = finite_matrix(data["A_Y"], (EXPECTED_A_ROWS, 3), "A_Y")
        test_targets = finite_matrix(data["D_Y"], (EXPECTED_D_ROWS, 3), "D_Y")
        raw_variance = np.var(train_targets, axis=0, ddof=0)
        loss_scale = np.maximum(raw_variance, VARIANCE_FLOOR)
        declared_scale = np.asarray(task["loss_scale"], dtype=float)
        if declared_scale.shape != (3,) or not np.allclose(
                declared_scale, loss_scale, rtol=1e-12, atol=0):
            raise InvalidEvidence("Manifest A-target variance differs from prepared A_Y")
        if "offsets" in data.files and not np.array_equal(data["offsets"], HORIZONS):
            raise InvalidEvidence("Prepared horizon order differs from protocol")
        if "D_groups" in data.files:
            groups, counts = np.unique(data["D_groups"], return_counts=True)
            if len(groups) != 128 or not np.all(counts == 13):
                raise InvalidEvidence("D episode/origin accounting differs from protocol")
    with np.load(prediction_file, allow_pickle=False) as data:
        expected_keys = {"Y", *METHODS}
        if set(data.files) != expected_keys:
            raise InvalidEvidence(
                f"Prediction keys differ; missing={expected_keys-set(data.files)}, "
                f"extra={set(data.files)-expected_keys}"
            )
        y = finite_matrix(data["Y"], (EXPECTED_D_ROWS, 3), "saved Y")
        if not np.array_equal(y, test_targets):
            raise InvalidEvidence("Saved prediction Y differs from prepared D_Y")
        predictions = {method: finite_matrix(data[method], y.shape, method)
                       for method in METHODS}
    fit = summary.get("fit_diagnostics")
    pred_diagnostics = summary.get("prediction_diagnostics")
    if not isinstance(fit, dict) or not isinstance(pred_diagnostics, dict):
        raise InvalidEvidence("Missing fitting or prediction diagnostics")
    method_failures = fit.get("method_failures")
    if not isinstance(method_failures, dict):
        raise InvalidEvidence("Missing whole-method failure accounting")
    if method_failures:
        raise InvalidEvidence(f"Wholly failed methods: {method_failures}")
    recoveries = pred_diagnostics.get("recovered_prediction_cells")
    if not isinstance(recoveries, dict) or not recoveries:
        raise InvalidEvidence("Missing final prediction-cell recovery accounting")
    if set(recoveries) != set(METHODS):
        raise InvalidEvidence("Incomplete prediction-cell recovery accounting")
    for method, count in recoveries.items():
        if (method not in METHODS or isinstance(count, bool) or not isinstance(count, int)
                or count < 0):
            raise InvalidEvidence("Invalid final prediction-cell recovery accounting")
        if count > 0:
            raise InvalidEvidence(f"Final prediction-cell recovery for {method}: {count}")
    policy = fit.get("policy", {})
    selected = policy.get("selected_method")
    if selected not in NONKAHM_METHODS:
        raise InvalidEvidence(f"Undeclared C-selected policy method: {selected!r}")
    if policy.get("selection_split") != "C":
        raise InvalidEvidence("Policy selection split is not C")
    candidates = policy.get("candidates", [])
    if [row.get("method") for row in candidates] != list(NONKAHM_METHODS):
        raise InvalidEvidence("C policy candidate roster/order differs from protocol")
    scores = np.asarray([row["normalized_mse"] for row in candidates], dtype=float)
    if (not np.isfinite(scores).all() or np.any(scores < 0)
            or selected != NONKAHM_METHODS[int(np.argmin(scores))]):
        raise InvalidEvidence("Saved C policy choice is inconsistent with its declared scores")
    if not np.allclose(policy.get("A_target_variance", []), loss_scale, rtol=1e-12, atol=0):
        raise InvalidEvidence("Policy normalization differs from A target variances")
    if not np.array_equal(predictions["nonkahm_policy"], predictions[selected]):
        raise InvalidEvidence("Saved policy predictions differ from C-selected complete arm")
    gamma = _gamma_map(fit, selected)
    bound_rows = _bound_rows(pred_diagnostics, metadata)
    readout_rows = _readout_rows(pred_diagnostics, metadata, predictions, gamma)
    for method in METHODS:
        if gamma[method] is not None:
            for j, value in enumerate(gamma[method]):
                if value == 0 and not np.array_equal(predictions[method][:, j], predictions["ridge"][:, j]):
                    raise InvalidEvidence(f"Zero-gamma forecast differs from baseline: {method}, horizon {HORIZONS[j]}")

    task_rows, horizon_rows, activation_rows = [], [], []
    metrics: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        error = predictions[method] - y
        with np.errstate(over="raise", invalid="raise"):
            mse = np.mean(np.square(error), axis=0)
            normalized = mse / loss_scale
        if not np.isfinite(mse).all() or not np.isfinite(normalized).all():
            raise InvalidEvidence(f"Non-finite recomputed error for {method}")
        maximum = float(np.max(np.abs(error)))
        metrics[method] = {"nmse": float(np.mean(normalized)),
                           "raw_mse": float(np.mean(mse)), "max_abs_error": maximum,
                           "horizon_nmse": normalized.tolist(),
                           "horizon_raw_mse": mse.tolist()}
        task_rows.append({**metadata, "method": method, **{
            key: metrics[method][key] for key in ("nmse", "raw_mse", "max_abs_error")},
            "n_origins": len(y), "n_horizons": 3, "n_target_cells": y.size,
            "selected_policy_method": selected if method == "nonkahm_policy" else ""})
        for j, horizon in enumerate(HORIZONS):
            horizon_rows.append({**metadata, "method": method, "horizon": horizon,
                                 "raw_mse": float(mse[j]), "nmse": float(normalized[j]),
                                 "A_target_variance": float(raw_variance[j]),
                                 "loss_scale": float(loss_scale[j]),
                                 "variance_floor_used": bool(raw_variance[j] < VARIANCE_FLOOR),
                                 "max_abs_error": float(np.max(np.abs(error[:, j]))),
                                 "n_origins": len(y)})
            value = None if gamma[method] is None else gamma[method][j]
            activation_rows.append({**metadata, "method": method, "horizon": horizon,
                                    "gamma": value, "correction_active":
                                    None if value is None else value > 0,
                                    "selected_policy_method": selected if method == "nonkahm_policy" else ""})
    candidate_max = metrics[PRIMARY_CANDIDATE]["max_abs_error"]
    ridge_max = metrics["ridge"]["max_abs_error"]
    max_error_pass = bool(candidate_max <= MAX_ERROR_RATIO * ridge_max)
    error_ratio = candidate_max / ridge_max if ridge_max > 0 else (
        1.0 if candidate_max == 0 else None)
    stability = {**metadata, "candidate_max_abs_error": candidate_max,
                 "ridge_max_abs_error": ridge_max, "max_error_ratio": error_ratio,
                 "ratio_defined": ridge_max > 0,
                 "ratio_limit": MAX_ERROR_RATIO, "max_error_pass": max_error_pass,
                 "feature_bounds_pass": all(row["bound_pass"] for row in bound_rows),
                 "readout_bounds_pass": all(row["coefficient_bound_pass"]
                                             and row["correction_bound_pass"]
                                             and row["training_mean_bound_pass"] for row in readout_rows)}
    resource_rows = [{**metadata, "resource": key, "value": value}
                     for key, value in flatten_scalars(summary.get("resources", {}))]
    if not resource_rows:
        raise InvalidEvidence("Missing resource accounting")
    diagnostic_rows = []
    for source, mapping in (("fit", fit), ("prediction", pred_diagnostics)):
        # Keep all numerical safety/failure accounting without coupling analysis
        # to the model's detailed coefficient or optimization-log schema.
        for key, value in flatten_scalars(mapping):
            if any(token in key.lower() for token in
                   ("fail", "fallback", "recover", "guard", "denominator")):
                diagnostic_rows.append({**metadata, "stage": source,
                                        "diagnostic": key, "value": value})
    hashes = [{"path": relative_path(path, study_dir), "sha256": value}
              for path, value in ((prepared, digest),
                                  (prediction_file, sha256(prediction_file)),
                                  (summary_file, sha256(summary_file)))]
    return {"task": task, "metrics": metrics, "task_rows": task_rows,
            "horizon_rows": horizon_rows, "activation_rows": activation_rows,
            "bound_rows": bound_rows, "readout_rows": readout_rows, "stability": stability,
            "resource_rows": resource_rows, "diagnostic_rows": diagnostic_rows,
            "hashes": hashes, "policy_method": selected}


def paired_margin_contrast(candidate: np.ndarray, comparator: np.ndarray, *,
                           comparator_factor: float, contrast_id: str,
                           comparator_name: str, scope: str,
                           margin: float) -> tuple[dict[str, Any], np.ndarray]:
    """Compute the frozen one-sided t bound from independent paired repeats."""
    candidate, comparator = np.asarray(candidate, float), np.asarray(comparator, float)
    if candidate.shape != (len(SEEDS),) or comparator.shape != candidate.shape:
        raise InvalidEvidence("Primary contrast must contain exactly 24 paired repetitions")
    if not np.isfinite(candidate).all() or not np.isfinite(comparator).all():
        raise InvalidEvidence("Non-finite primary repetition aggregate")
    if np.any(candidate < 0) or np.any(comparator < 0):
        raise InvalidEvidence("Forecast loss cannot be negative")
    differences = comparator_factor * comparator - candidate
    n = len(differences)
    average = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    se = sd / math.sqrt(n)
    critical = float(student_t.ppf(1 - FAMILY_ALPHA / 6, n - 1))
    lower = average - critical * se
    comparator_mean, candidate_mean = float(np.mean(comparator)), float(np.mean(candidate))
    relative_gain = (1 - candidate_mean / comparator_mean) if comparator_mean > 0 else None
    if se > 0:
        p_one_sided = float(student_t.sf(average / se, n - 1))
    else:
        p_one_sided = 0.0 if average > 0 else (1.0 if average < 0 else 0.5)
    return ({"contrast_id": contrast_id, "scope": scope,
             "candidate": PRIMARY_CANDIDATE, "comparator": comparator_name,
             "n_repetitions": n, "df": n - 1, "margin": margin,
             "comparator_factor": comparator_factor,
             "candidate_mean_nmse": candidate_mean,
             "comparator_mean_nmse": comparator_mean,
             "point_relative_gain": relative_gain,
             "margin_gap_mean": average, "margin_gap_sample_sd": sd,
             "margin_gap_se": se, "critical_t": critical,
             "simultaneous_one_sided_lower": lower,
             "family_alpha": FAMILY_ALPHA, "family_size": 6,
             "p_one_sided": p_one_sided,
             "p_bonferroni": min(1.0, 6 * p_one_sided),
             "positive_margin_repetitions": int(np.sum(differences > 0)),
             "criterion_established": bool(lower > 0),
             "criterion": "useful_superiority" if scope == "nonlinear" else "linear_noninferiority"},
            differences)


def aggregate_complete(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """This is called only after every frozen task has validated successfully."""
    lookup = {(item["task"]["law"], float(item["task"]["noise_level"]),
               int(item["task"]["seed"])): item for item in tasks}
    expected = set(itertools.product(LAWS, NOISE_LEVELS, SEEDS))
    if set(lookup) != expected or len(tasks) != len(expected):
        raise InvalidEvidence("Cannot aggregate an incomplete or repeated roster")
    condition_rows, condition_horizon_rows = [], []
    for law, noise, method in itertools.product(LAWS, NOISE_LEVELS, METHODS):
        rows = [lookup[(law, noise, seed)]["metrics"][method] for seed in SEEDS]
        values = np.array([row["nmse"] for row in rows])
        condition_rows.append({"law": law, "noise_level": noise, "method": method,
                               "mean_nmse": float(np.mean(values)),
                               "sample_sd_nmse": float(np.std(values, ddof=1)),
                               "mean_raw_mse": float(np.mean([row["raw_mse"] for row in rows])),
                               "n_repetitions": len(SEEDS), "inference": "descriptive"})
        for j, horizon in enumerate(HORIZONS):
            values = np.array([row["horizon_nmse"][j] for row in rows])
            condition_horizon_rows.append({
                "law": law, "noise_level": noise, "method": method, "horizon": horizon,
                "mean_nmse": float(np.mean(values)),
                "sample_sd_nmse": float(np.std(values, ddof=1)),
                "mean_raw_mse": float(np.mean([row["horizon_raw_mse"][j] for row in rows])),
                "n_repetitions": len(SEEDS), "inference": "descriptive"})

    replicate_rows = []
    values_by_scope: dict[str, dict[str, list[float]]] = {}
    for scope, laws in (("nonlinear", NONLINEAR_LAWS), ("linear", ("linear_ar",)),
                        ("all_descriptive", LAWS)):
        values_by_scope[scope] = {method: [] for method in METHODS}
        for seed, method in itertools.product(SEEDS, METHODS):
            rows = [lookup[(law, noise, seed)]["metrics"][method]
                    for law, noise in itertools.product(laws, NOISE_LEVELS)]
            value = float(np.mean([row["nmse"] for row in rows]))
            values_by_scope[scope][method].append(value)
            replicate_rows.append({"seed": seed, "scope": scope, "method": method,
                                   "nmse": value, "n_conditions": len(rows),
                                   "equal_horizon_weight": True,
                                   "equal_condition_weight": True})

    contrast_rows, replicate_contrast_rows = [], []
    cases = [("nonlinear", comparator, 1 - SUPERIORITY_MARGIN,
              SUPERIORITY_MARGIN, f"nonlinear_vs_{comparator}")
             for comparator in PRIMARY_COMPARATORS]
    cases.append(("linear", "ridge", 1 + LINEAR_NONINFERIORITY_MARGIN,
                  LINEAR_NONINFERIORITY_MARGIN, "linear_noninferiority_vs_ridge"))
    for scope, comparator, factor, margin, identifier in cases:
        candidate_values = np.array(values_by_scope[scope][PRIMARY_CANDIDATE])
        comparator_values = np.array(values_by_scope[scope][comparator])
        row, differences = paired_margin_contrast(
            candidate_values, comparator_values, comparator_factor=factor,
            contrast_id=identifier, comparator_name=comparator, scope=scope, margin=margin)
        contrast_rows.append(row)
        for j, seed in enumerate(SEEDS):
            replicate_contrast_rows.append({
                "contrast_id": identifier, "scope": scope, "seed": seed,
                "candidate": PRIMARY_CANDIDATE, "comparator": comparator,
                "candidate_nmse": float(candidate_values[j]),
                "comparator_nmse": float(comparator_values[j]),
                "comparator_factor": factor, "margin_gap": float(differences[j])})

    overview_rows = []
    for scope, methods in values_by_scope.items():
        for method, values in methods.items():
            overview_rows.append({"scope": scope, "method": method,
                                  "mean_nmse": float(np.mean(values)),
                                  "sample_sd_nmse": float(np.std(values, ddof=1)),
                                  "n_repetitions": len(SEEDS)})
    return {"condition_metrics": condition_rows,
            "condition_horizon_metrics": condition_horizon_rows,
            "replicate_metrics": replicate_rows,
            "primary_contrasts": contrast_rows,
            "replicate_contrasts": replicate_contrast_rows,
            "overview_metrics": overview_rows}


def _base_decision() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "analysis_utc": datetime.now(timezone.utc).isoformat(),
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_comparators": list(PRIMARY_COMPARATORS),
        "expected_tasks": len(LAWS) * len(NOISE_LEVELS) * len(SEEDS),
        "independent_repetitions": len(SEEDS),
        "family_alpha": FAMILY_ALPHA, "family_size": 6,
        "interval": "Bonferroni six one-sided paired-repetition t lower bounds, df=23",
        "independence_unit": "repetition aggregate, not windows or paired-noise tasks",
        "superiority_margin": SUPERIORITY_MARGIN,
        "linear_noninferiority_margin": LINEAR_NONINFERIORITY_MARGIN,
        "loss_normalization": "A retained-target variance by horizon, ddof=0, floor=1e-12",
        "feature_bound_evidence": "model-reported diagnostic norms, independently consistency-checked",
        "claim_scope": (
            "Fixed known numerical laws and the declared observed-noise levels. "
            "Fresh draws are not new application evidence. The 3%/5% margins "
            "and 5x maximum-error screen are research-allocation choices. "
            "Failure to pass does not prove equivalence or universal impossibility."
        ),
    }


def analyze(study_dir: Path, results_dir: Path, output_dir: Path) -> dict[str, Any]:
    study_dir, results_dir, output_dir = (
        path.resolve() for path in (study_dir, results_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = _base_decision()
    errors: list[dict[str, str]] = []
    valid: list[dict[str, Any]] = []
    config_path = study_dir / "config.json"
    manifest_path = study_dir / "prepared_inputs" / "manifest.json"
    source_paths = [Path(__file__).resolve(), study_dir / "PROTOCOL.md",
                    config_path, manifest_path]
    source_hashes = [{"path": relative_path(path, study_dir), "sha256": sha256(path)}
                     for path in source_paths if path.is_file()]
    try:
        validate_config(read_json(config_path))
        source_hashes.extend(verify_freeze(study_dir))
        roster = validate_roster(read_json(manifest_path))
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append({"task_id": "STUDY", "error": f"{type(error).__name__}: {error}"})
        roster = []
    declared_ids = {task["task_id"] for task in roster}
    if roster and results_dir.is_dir():
        extra = sorted(path.name for path in results_dir.iterdir()
                       if path.is_dir() and path.name not in declared_ids
                       and ((path / "summary.json").exists()
                            or (path / "predictions.npz").exists()))
        if extra:
            errors.append({"task_id": "STUDY", "error": f"Undeclared result task directories: {extra}"})
    for task in roster:
        try:
            valid.append(load_task(study_dir, results_dir, task))
        except (OSError, ValueError, KeyError, TypeError, OverflowError,
                FloatingPointError) as error:
            errors.append({"task_id": task["task_id"],
                           "error": f"{type(error).__name__}: {error}"})

    decision.update({"completed_valid_tasks": len(valid), "task_errors": errors,
                     "technical_complete": not errors and len(valid) == decision["expected_tasks"]})
    write_json(output_dir / "task_validation.json", {
        "expected_tasks": decision["expected_tasks"], "valid_tasks": len(valid),
        "errors": errors, "successful_subset_inference": False,
    })
    # Stale aggregate files cannot survive a later incomplete verification run.
    table_names = (
        "task_metrics", "horizon_metrics", "activation", "feature_bounds", "readout_bounds",
        "stability", "resources", "safety_diagnostics", "condition_metrics",
        "condition_horizon_metrics", "replicate_metrics", "primary_contrasts",
        "replicate_contrasts", "overview_metrics",
    )
    for name in table_names:
        path = output_dir / f"{name}.csv"
        if path.exists():
            path.unlink()
    if not decision["technical_complete"]:
        decision.update({"decision": "TECHNICAL_BLOCK", "statistical_gate_pass": None,
                         "stability_gate_pass": None, "reason":
                         "The complete frozen evidence roster did not validate; no partial efficacy aggregation was performed."})
    else:
        tables = aggregate_complete(valid)
        source_names = {
            "task_metrics": "task_rows", "horizon_metrics": "horizon_rows",
            "activation": "activation_rows", "feature_bounds": "bound_rows",
            "readout_bounds": "readout_rows",
            "resources": "resource_rows", "safety_diagnostics": "diagnostic_rows",
        }
        for target, source in source_names.items():
            tables[target] = [row for item in valid for row in item[source]]
        tables["stability"] = [item["stability"] for item in valid]
        for name, rows in tables.items():
            write_csv(output_dir / f"{name}.csv", rows)
        statistical = all(row["criterion_established"] for row in tables["primary_contrasts"])
        stable = all(row["max_error_pass"] and row["feature_bounds_pass"] and row["readout_bounds_pass"]
                     for row in tables["stability"])
        failed_contrasts = [row["contrast_id"] for row in tables["primary_contrasts"]
                            if not row["criterion_established"]]
        failed_stability = [row["task_id"] for row in tables["stability"]
                            if not row["max_error_pass"] or not row["feature_bounds_pass"] or not row["readout_bounds_pass"]]
        active = [row for row in tables["activation"]
                  if row["method"] == PRIMARY_CANDIDATE and row["law"] in NONLINEAR_LAWS]
        decision.update({
            "decision": "ADVANCE_TO_APPLICATION" if statistical and stable else "STOP_FORECASTING_PERFORMANCE_BRANCH",
            "statistical_gate_pass": statistical, "stability_gate_pass": stable,
            "failed_contrasts": failed_contrasts, "failed_stability_tasks": failed_stability,
            "primary_contrasts": tables["primary_contrasts"],
            "primary_nonlinear_active_horizon_cells": sum(row["correction_active"] for row in active),
            "primary_nonlinear_horizon_cells": len(active),
            "activation_is_a_gate": False, "resource_cost_is_a_gate": False,
            "scored_target_cells_per_method": len(valid) * EXPECTED_D_ROWS * len(HORIZONS),
            "point_estimates": tables["overview_metrics"],
            "reason": (
                "All six margin contrasts and all declared stability screens passed; undertake a separate reserved application study."
                if statistical and stable else
                "The fixed-budget experiment did not establish every required useful-effect/noninferiority and stability condition; stop this forecasting-performance branch."
            ),
        })
    write_json(output_dir / "decision.json", decision)
    analyzed = source_hashes + [row for item in valid for row in item["hashes"]]
    outputs = [{"path": relative_path(path, study_dir), "sha256": sha256(path)}
               for path in sorted(output_dir.iterdir())
               if path.is_file() and path.name != "analysis_manifest.json"]
    write_json(output_dir / "analysis_manifest.json", {
        "schema_version": SCHEMA, "decision": decision["decision"],
        "analyzed_files": analyzed, "output_files": outputs,
        "model_imported": False, "successful_subset_inference": False,
    })
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--results", type=Path, help="Task result root; default STUDY/results")
    parser.add_argument("--output", type=Path, help="Analysis output root; default STUDY/analysis")
    args = parser.parse_args()
    result = analyze(args.study_dir, args.results or args.study_dir / "results",
                     args.output or args.study_dir / "analysis")
    print(json.dumps({"decision": result["decision"],
                      "completed_valid_tasks": result["completed_valid_tasks"],
                      "expected_tasks": result["expected_tasks"],
                      "failed_contrasts": result.get("failed_contrasts", []),
                      "task_errors": result["task_errors"]}, allow_nan=False))
    return 2 if result["decision"] == "TECHNICAL_BLOCK" else 0


if __name__ == "__main__":
    sys.exit(main())
