"""Explicit standard/min--max geometry derivative of the pinned ten methods.

Only the geometry history scaler differs between arms. The baseline history
scaler, target units, downstream feature scalers, radial maps, fitted-head
grids, B selection and C calibration are inherited unchanged. No target from
the final forecast origin is accepted by either interface.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from kahm_adapter import ORIGINAL_METHODS, array_sha256, json_safe, verify_vendor


ROOT = Path(__file__).resolve().parents[1]
DERIVATIVE_ROOT = Path(__file__).resolve().parent / "vendor_geometry"
VENDOR_ROOT = DERIVATIVE_ROOT / "kahm_geometry"
GEOMETRY_ARMS = ("standard", "minmax", "tanh_z_1", "tanh_z_0p5", "tanh_z_2", "tanh_raw")
TANH_MULTIPLIERS = {"tanh_z_1": 1.0, "tanh_z_0p5": 0.5, "tanh_z_2": 2.0}
GEOMETRY_METHODS = ("common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded")
ADAPTER_VERSION = "kahm-corrected-dimension-geometry-adapter-v1"


def _matrix(value, label):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite nonempty matrix")
    return value


def verify_derivative():
    """Verify the isolated corrected runtime and its immutable source lineage."""
    original = verify_vendor()
    manifest_path = ROOT / "provenance/FORECASTING_RUNTIME_MANIFEST.json"
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if manifest["source_commit"] != original["commit"]:
        raise ValueError("Corrected runtime source commit differs")
    if manifest["source_vendor_manifest_sha256"] != original["manifest_sha256"]:
        raise ValueError("Corrected runtime source manifest differs")
    for row in manifest["files"] + manifest["historical_reference_files"]:
        relative = Path(row["path"])
        path = (ROOT / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Corrected source escapes its root")
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Corrected source identity failed: {relative}")
    policy = ROOT.parent / "policy/kahm_dimension_policy.py"
    policy_sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    if policy_sha != manifest["shared_dimension_policy_sha256"]:
        raise ValueError("Shared dimension policy identity differs")
    reference_manifest = ROOT / "reference_current/code/vendor_geometry/DERIVATIVE_MANIFEST.json"
    if hashlib.sha256(reference_manifest.read_bytes()).hexdigest() != manifest["source_current_derivative_manifest_sha256"]:
        raise ValueError("Historical derivative source identity differs")
    return {**original, "corrected_runtime_manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "corrected_runtime_files_verified": len(manifest["files"]),
            "shared_dimension_policy_sha256": policy_sha,
            "default_dimension_policy": "native", "default_rank_cap": 20,
            "scope": "Construction correction only; no rescoring of archived studies."}


def _implementation():
    provenance = verify_derivative()
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    module = importlib.import_module("extensions.reconstruction_mechanism.models")
    expected = VENDOR_ROOT / "extensions/reconstruction_mechanism/models.py"
    if Path(module.__file__).resolve() != expected.resolve():
        raise ImportError("An incompatible KAHM tree was already imported; use a separate worker process")
    for name, loaded in tuple(sys.modules.items()):
        if name.startswith("extensions.") and getattr(loaded, "__file__", None):
            if not Path(loaded.__file__).resolve().is_relative_to(VENDOR_ROOT.resolve()):
                raise ImportError(f"Mixed original and derived module trees: {name}")
    if tuple(module.METHODS) != ORIGINAL_METHODS:
        raise ValueError("Original method roster changed")
    return module, provenance


def fit_scaler_audit(xa, arm):
    if arm not in GEOMETRY_ARMS:
        raise ValueError("Unknown geometry arm")
    mean, raw_std = xa.mean(axis=0), xa.std(axis=0)
    minimum, maximum = xa.min(axis=0), xa.max(axis=0)
    span = maximum - minimum
    return {"arm": arm, "fitting_split": "A", "fit_before_bank_row_cap": True,
            "fit_rows": int(len(xa)), "minimum": minimum, "maximum": maximum,
            "range": span, "constant": span == 0.0, "mean": mean,
            "raw_standard_deviation": raw_std,
            "legacy_standard_scale": np.where(raw_std > 1e-12, raw_std, 1.0),
            "clip": False, "constant_minmax_rule": "zero for every geometry query",
            "tanh_multiplier": TANH_MULTIPLIERS.get(arm),
            "tanh_standard_deviation_floor": 1e-12, "standard_deviation_ddof": 0,
            "tanh_post_rescale": False,
            "tanh_constant_rule": "historical standard scale=1 when std<=1e-12; later query differences are retained",
            "tanh_raw_fitted_location_scale": False}


def scaled_histories(x, scaler, arm=None):
    arm = scaler["arm"] if arm is None else arm
    if arm == "standard":
        return (x - scaler["mean"]) / scaler["legacy_standard_scale"]
    if arm in TANH_MULTIPLIERS:
        return np.tanh(TANH_MULTIPLIERS[arm] * ((x - scaler["mean"]) / scaler["legacy_standard_scale"]))
    if arm == "tanh_raw":
        return np.tanh(x)
    if arm != "minmax":
        raise ValueError("Unknown geometry arm")
    z = 2.0 * ((x - scaler["minimum"]) / np.where(scaler["constant"], 1.0, scaler["range"])) - 1.0
    z[:, scaler["constant"]] = 0.0
    return z


def geometry_query_diagnostics(x, scaler):
    outside = (x < scaler["minimum"]) | (x > scaler["maximum"])
    z = scaled_histories(x, scaler)
    standard = scaled_histories(x, scaler, "standard")
    minmax = scaled_histories(x, scaler, "minmax")
    delta = minmax - standard
    if not np.isfinite(z).all():
        raise FloatingPointError("Nonfinite audited geometry transformation")
    is_tanh = scaler["arm"].startswith("tanh_")
    argument = (x if scaler["arm"] == "tanh_raw" else
                TANH_MULTIPLIERS[scaler["arm"]] * standard) if is_tanh else None
    z32 = z.astype(np.float32)
    absolute = np.abs(z)
    column_std = z.std(axis=0)
    column_range = np.ptp(z, axis=0)
    td = {"is_tanh": is_tanh, "coordinate_count": int(z.size),
        "argument_minimum": None if argument is None else float(argument.min()),
        "argument_maximum": None if argument is None else float(argument.max()),
        "abs_ge_0p99_count": int(np.count_nonzero(absolute >= .99)),
        "abs_ge_0p999_count": int(np.count_nonzero(absolute >= .999)),
        "abs_ge_0p999999_count": int(np.count_nonzero(absolute >= .999999)),
        "abs_eq_1_count": int(np.count_nonzero(absolute == 1.)),
        "rows_any_abs_eq_1": int(np.count_nonzero(np.any(absolute == 1., axis=1))),
        "unique_rows_float64": int(len(np.unique(z, axis=0))),
        "unique_rows_float32": int(len(np.unique(z32, axis=0))),
        "zero_range_columns_float64": int(np.count_nonzero(column_range == 0.)),
        "zero_range_columns_float32": int(np.count_nonzero(np.ptp(z32, axis=0) == 0.)),
        "column_population_std": column_std, "column_range": column_range,
        "maximum_column_std": float(column_std.max()),
        "minimum_column_std": float(column_std.min()),
        "all_finite": True, "post_tanh_rescale": False if is_tanh else None,
        "interpretation": "Boundary counts diagnose tanh saturation only for tanh arms; unique rows and zero ranges are exact floating-point diagnostics."}
    return json_safe({"rows": len(x), "columns": x.shape[1],
        "transformation_diagnostics": td,
        "outside_A_range_coordinates": int(outside.sum()),
        "outside_A_range_coordinate_fraction": float(outside.mean()),
        "outside_A_range_rows": int(outside.any(axis=1).sum()),
        "outside_A_range_row_fraction": float(outside.any(axis=1).mean()),
        "constant_A_columns": int(scaler["constant"].sum()),
        "queries_differing_from_A_constant": int(outside[:, scaler["constant"]].sum()),
        "geometry_minimum": float(z.min()), "geometry_maximum": float(z.max()),
        "geometry_max_abs": float(np.abs(z).max()),
        "geometry_array_sha256": array_sha256(z),
        "standard_minus_minmax_comparison": {
            "maximum_absolute_difference": float(np.abs(delta).max()),
            "root_mean_squared_difference": float(np.sqrt(np.mean(delta * delta))),
            "identical_coordinates": int(np.count_nonzero(delta == 0.0))},
        "scope": "A-frozen geometry-input coordinate diagnostic; no target observations or fitting."})


@dataclass
class FittedScalingTask:
    suite: Any
    horizons: tuple[int, ...]
    arm: str
    scaler_audit: dict
    fit_summary: dict

    @property
    def method_names(self):
        return ORIGINAL_METHODS

    def predict(self, x):
        x = _matrix(x, "query histories")
        wall, cpu = time.perf_counter(), time.process_time()
        with threadpool_limits(limits=1):
            result = self.suite.predict(x)
        measured = {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu}
        predictions = result["predictions"]
        uncalibrated = result["uncalibrated_predictions"]
        if tuple(predictions) != ORIGINAL_METHODS or tuple(uncalibrated) != GEOMETRY_METHODS:
            raise ValueError("Prediction roster changed")
        for method, value in {**predictions, **{f"uncalibrated:{k}": v for k, v in uncalibrated.items()}}.items():
            if value.shape != (len(x), len(self.horizons)) or not np.isfinite(value).all():
                raise FloatingPointError(f"Incomplete prediction: {method}")
        return {"predictions": predictions, "uncalibrated_predictions": uncalibrated,
                "diagnostics": json_safe(result["diagnostics"]),
                "geometry_query_diagnostics": geometry_query_diagnostics(x, self.scaler_audit),
                "resources": json_safe({**result["resources"],
                    "original_suite_predict_including_diagnostics": measured}),
                "metadata": {"adapter_version": ADAPTER_VERSION, "geometry_scaling": self.arm,
                             "no_query_targets_accepted": True,
                             "uncalibrated_scope": "Already computed baseline + selected fitted head, before C gamma blending; includes source emergency recovery if any."}}


def fit_scaling_suite(a_x, a_y, b_x, b_y, c_x, c_y, *, seed: int,
                      horizons: Sequence[int], arm: str, config: dict | None = None) -> FittedScalingTask:
    arrays = [_matrix(x, name) for x, name in zip((a_x,a_y,b_x,b_y,c_x,c_y),
                                                  ("A_X","A_Y","B_X","B_Y","C_X","C_Y"))]
    horizons = tuple(int(h) for h in horizons)
    if arm not in GEOMETRY_ARMS or not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError("Invalid geometry arm or horizons")
    if any(y.shape[1] != len(horizons) for y in arrays[1::2]):
        raise ValueError("Target width differs from declared horizons")
    module, provenance = _implementation()
    effective = module.dimension_config(module.DEFAULT_CONFIG, config)
    wall, cpu = time.perf_counter(), time.process_time()
    with threadpool_limits(limits=1):
        runtime_threads = threadpool_info()
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        suite = module.fit_suite(*arrays, seed=int(seed), config=effective, geometry_scaling=arm)
    elapsed = {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu}
    audit_scaler = fit_scaler_audit(arrays[0], arm)
    if suite.geometry is not None:
        bank_scaler = suite.geometry.bank.get("geometry_scaler")
        if arm == "standard" and bank_scaler is not None:
            raise AssertionError("Standard bank unexpectedly retained an alternative scaler")
        if arm == "minmax":
            if bank_scaler is None or bank_scaler["fit_rows"] != len(arrays[0]):
                raise AssertionError("Min--max bank did not fit all A histories")
            for key in ("minimum", "maximum", "range", "constant"):
                np.testing.assert_array_equal(bank_scaler[key], audit_scaler[key])
    parameters = {}
    for method in ORIGINAL_METHODS:
        b = suite.diagnostics["B"].get(method, {})
        parameters[method] = {"selected_hyperparameters": b.get("selected"),
            "feature_width": b.get("feature_width"),
            "gamma_by_horizon": suite.diagnostics["gamma_by_horizon"].get(method),
            "fit_failure": suite.diagnostics["method_failures"].get(method)}
    summary = json_safe({"adapter_version": ADAPTER_VERSION, "upstream": provenance,
        "geometry_scaling": audit_scaler, "horizons": horizons, "history": arrays[0].shape[1],
        "seed": int(seed), "A_B_C_rows": [len(arrays[i]) for i in (0,2,4)],
        "input_array_sha256": {name: array_sha256(x) for name,x in zip(("A_X","A_Y","B_X","B_Y","C_X","C_Y"),arrays)},
        "model_config": effective, "original_methods": ORIGINAL_METHODS,
        "selected_policy": suite.selected_policy, "method_parameters": parameters,
        "fit_diagnostics": suite.diagnostics, "resources": suite.resources,
        "original_suite_fit_including_diagnostics": elapsed,
        "geometry_diagnostics": {s: geometry_query_diagnostics(arrays[i],audit_scaler) for s,i in (("A",0),("B",2),("C",4))},
        "runtime_threadpool_libraries": runtime_threads, "runtime_cpu_affinity": affinity,
        "runtime_packages": {name: importlib.metadata.version(name) for name in ("numpy","scipy","scikit-learn","threadpoolctl")},
        "scope": "Corrected construction: A fit, B penalty selection, C correction calibration; targets unscaled; no post-C refit. No study results are produced by this adapter."})
    return FittedScalingTask(suite, horizons, arm, audit_scaler, summary)
