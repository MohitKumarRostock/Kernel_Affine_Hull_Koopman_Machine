"""Small adapter around the unmodified pinned ten-method forecasting suite.

The fit interface accepts only causal A/B/C windows. The caller owns temporal
splits, target maturity, row sampling, selection of series, and the protocol
freeze. The prediction interface accepts histories only. Extra naive methods
are separate, uncalibrated query rules and do not change the original policy.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

PINNED_COMMIT = "7dda356138675934aab5e327ec667bde0a5eda79"
ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "vendor/kahm_pinned"
ORIGINAL_METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common",
                    "kahm_departure", "rbf_departure", "kahm_unbounded",
                    "rbf_unbounded", "nonkahm_policy")
ADAPTER_VERSION = "kahm-adoption-adapter-v1"


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def json_safe(value: Any) -> Any:
    """Produce strict JSON data without replacing nonfinite scientific values."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite diagnostic value must be handled explicitly by the study runner")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported summary value: {type(value).__name__}")


def verify_vendor() -> dict[str, Any]:
    """Check the vendored subset before importing its executable modules."""
    manifest_path = VENDOR_ROOT.parent / "VENDOR_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["commit"] != PINNED_COMMIT:
        raise ValueError("Vendor commit differs from the adapter's pinned implementation")
    for row in manifest["files"]:
        path = VENDOR_ROOT / row["path"]
        if not path.resolve().is_relative_to(VENDOR_ROOT.resolve()):
            raise ValueError("Vendor path escapes its root")
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Vendor source identity differs: {row['path']}")
    return {"commit": PINNED_COMMIT, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "files_verified": len(manifest["files"])}


def _upstream():
    provenance = verify_vendor()
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    module = importlib.import_module("extensions.reconstruction_mechanism.models")
    expected = VENDOR_ROOT / "extensions/reconstruction_mechanism/models.py"
    if Path(module.__file__).resolve() != expected.resolve():
        raise ImportError("KAHM module was already imported from a different source tree")
    if tuple(module.METHODS) != ORIGINAL_METHODS:
        raise ValueError("Original method roster differs")
    return module, provenance


def default_model_config() -> dict[str, Any]:
    """Return a copy of the pinned principal study's exact model defaults."""
    module, _ = _upstream()
    return copy.deepcopy(module.DEFAULT_CONFIG)


def _matrix(value, label):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a nonempty finite numeric matrix")
    return value


@dataclass
class FittedTask:
    """Frozen original suite and declared query-only baseline metadata."""
    suite: Any
    horizons: tuple[int, ...]
    seasonal_period: int | None
    include_naive: bool
    fit_summary: dict[str, Any]

    @property
    def method_names(self):
        extras = () if not self.include_naive else (("naive_raw",) if self.seasonal_period is None else ("naive_raw", "seasonal_naive_raw"))
        return ORIGINAL_METHODS + extras

    def serialized_state(self) -> bytes:
        """Serialize this newly fitted, local state; never loads outside pickles."""
        return gzip.compress(pickle.dumps(self, protocol=5), compresslevel=6, mtime=0)

    def predict(self, x):
        """Predict all methods from histories; no target input, fitting or selection."""
        x = _matrix(x, "query histories")
        wall, cpu = time.perf_counter(), time.process_time()
        with threadpool_limits(limits=1):
            result = self.suite.predict(x)
        measured = {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu}
        predictions = dict(result["predictions"])
        extra_resources = {}
        if self.include_naive:
            rules = {"naive_raw": lambda: np.repeat(x[:, -1:], len(self.horizons), axis=1)}
            if self.seasonal_period is not None:
                indices = x.shape[1] - self.seasonal_period + (np.array(self.horizons) - 1) % self.seasonal_period
                rules["seasonal_naive_raw"] = lambda: x[:, indices].copy()
            for name, rule in rules.items():
                start_wall, start_cpu = time.perf_counter(), time.process_time()
                predictions[name] = rule()
                metadata = {"horizons": self.horizons, "seasonal_period": self.seasonal_period if name == "seasonal_naive_raw" else None}
                extra_resources[name] = {
                    "fit_and_selection": {"wall_seconds": 0., "cpu_seconds": 0., "nodes": []},
                    "predict": {"wall_seconds": time.perf_counter() - start_wall,
                                "cpu_seconds": time.process_time() - start_cpu, "nodes": ["fixed_query_rule"]},
                    "state": {"pickle_bytes": len(pickle.dumps(metadata, protocol=5)), "array_buffer_bytes": 0},
                    "scope": "Fixed uncalibrated rule on the supplied history; no fitting, C selection, or upstream-policy membership.",
                }
        if tuple(predictions) != self.method_names:
            raise ValueError("Prediction method roster differs")
        for method, prediction in predictions.items():
            if prediction.shape != (len(x), len(self.horizons)) or not np.isfinite(prediction).all():
                raise FloatingPointError(f"Incomplete/nonfinite predictions for {method}")
        return {"predictions": predictions, "diagnostics": json_safe(result["diagnostics"]),
                "resources": json_safe({**result["resources"],
                    "per_method": {**result["resources"]["per_method"], **extra_resources},
                    "original_suite_predict_including_diagnostics": measured}),
                "metadata": {"adapter_version": ADAPTER_VERSION, "horizons": list(self.horizons),
                             "original_policy_unchanged": True, "no_query_targets_accepted": True}}


def fit_original_suite(a_x, a_y, b_x, b_y, c_x, c_y, *, seed: int,
                       horizons: Sequence[int], seasonal_period: int | None = None,
                       include_naive: bool = True, config: dict[str, Any] | None = None) -> FittedTask:
    """Fit the original ten methods on A/B/C, leaving all final targets external.

    The model configuration is passed directly to the unmodified upstream
    implementation. No A+B or A+B+C refit is performed. B selects each
    horizon's hyperparameters; C selects gamma and the original seven-arm
    non-KAHM policy. Extra naive rules are never policy candidates.
    """
    arrays = [_matrix(x, label) for x, label in zip((a_x, a_y, b_x, b_y, c_x, c_y),
                                                   ("A_X", "A_Y", "B_X", "B_Y", "C_X", "C_Y"))]
    horizons = tuple(int(h) for h in horizons)
    if not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError("Horizons must be distinct positive integer offsets")
    if any(y.shape[1] != len(horizons) for y in arrays[1::2]):
        raise ValueError("Target columns must match the declared horizon offsets")
    if seasonal_period is not None:
        seasonal_period = int(seasonal_period)
        if seasonal_period < 1 or seasonal_period > arrays[0].shape[1]:
            raise ValueError("The seasonal-naive period must fit inside every history")
    module, provenance = _upstream()
    effective = copy.deepcopy({**module.DEFAULT_CONFIG, **(config or {})})
    input_hashes = {label: array_sha256(array) for label, array in zip(("A_X", "A_Y", "B_X", "B_Y", "C_X", "C_Y"), arrays)}
    wall, cpu = time.perf_counter(), time.process_time()
    with threadpool_limits(limits=1):
        suite = module.fit_suite(*arrays, seed=int(seed), config=effective)
    elapsed = {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu}
    method_parameters = {}
    for method in ORIGINAL_METHODS:
        b = suite.diagnostics["B"].get(method, {})
        method_parameters[method] = {"selected_hyperparameters": b.get("selected"),
                                     "feature_width": b.get("feature_width"),
                                     "gamma_by_horizon": suite.diagnostics["gamma_by_horizon"].get(method),
                                     "fit_failure": suite.diagnostics["method_failures"].get(method)}
    summary = json_safe({
        "adapter_version": ADAPTER_VERSION, "upstream": provenance,
        "horizons": horizons, "history": arrays[0].shape[1], "seed": int(seed),
        "A_B_C_rows": [len(arrays[i]) for i in (0, 2, 4)], "input_array_sha256": input_hashes,
        "model_config": effective, "original_methods": ORIGINAL_METHODS,
        "extra_naive_rules": bool(include_naive), "seasonal_period": seasonal_period,
        "selected_policy": suite.selected_policy, "method_parameters": method_parameters,
        "fit_diagnostics": suite.diagnostics, "resources": suite.resources,
        "original_suite_fit_including_diagnostics": elapsed,
        "runtime_packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        "scope": "Fixed fitted geometry and decoder; A fitting, B hyperparameter selection, C calibration/policy; one numerical thread; no post-C refit or query-target input.",
    })
    return FittedTask(suite, horizons, seasonal_period, bool(include_naive), summary)
