"""Locate the one shared dimension-policy module in this standalone package.

The forecasting configuration retains the historical key ``rank`` for the
requested cap. It is a cap, not the effective rank of every local fit.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

STEP1_ROOT = Path(__file__).resolve().parents[5]
if str(STEP1_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP1_ROOT))

from policy import kahm_dimension_policy as _policy

if Path(_policy.__file__).resolve() != (STEP1_ROOT / "policy/kahm_dimension_policy.py").resolve():
    raise ImportError("An incompatible dimension-policy module was already imported")

DEFAULT_KAHM_CAP = _policy.DEFAULT_KAHM_CAP
dimension_request = _policy.dimension_request
finalize_dimension_audit = _policy.finalize_dimension_audit
fit_native_pca = _policy.fit_native_pca


def dimension_config(defaults, supplied=None):
    """Merge configuration and validate the cap without silently coercing it."""
    config = copy.deepcopy({**defaults, **(supplied or {})})
    if "rank_cap" in config:
        raise ValueError("Use config['rank'] for the requested cap; rank_cap is not a configuration key")
    dimension_request(1, 2, cap=config["rank"], policy=config["dimension_policy"])
    return config


def historical_rank2_config():
    """Explicit replay of the old cap-two and extra ambient-minus-one rule."""
    return {"rank": 2, "dimension_policy": "legacy_normal_departure"}
