"""Serialization helpers and immutable-source verification for the corrected runtime.

Forecast entry points are principal_adapter.fit_principal_suite and
centered_adapter.fit_centered_suite. The former unmodified fit entry point
is retained only in reference_current for explicit historical verification.
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
ADAPTER_VERSION = "kahm-dimension-policy-source-utilities-v1"


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

