"""Corrected construction entry point for the manuscript's principal suite.

This delegates the existing ten-method A/B/C bounded experiment to its actual
runtime. It does not generate tasks, release terminal targets, score forecasts,
or revise the archived experiment. Query methods accept histories only.
"""
from __future__ import annotations

from pathlib import Path
import sys

STEP1_ROOT = Path(__file__).resolve().parents[2]
if str(STEP1_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP1_ROOT))

from policy.kahm_dimension_policy import DEFAULT_KAHM_CAP, dimension_request
from geometry_adapter import fit_scaling_suite


def fit_principal_suite(a_x, a_y, b_x, b_y, c_x, c_y, *, seed, horizons,
                        rank_cap=DEFAULT_KAHM_CAP, dimension_policy="native",
                        config=None):
    """Fit standard-scaled principal methods with an explicitly declared cap."""
    dimension_request(1, 2, cap=rank_cap, policy=dimension_policy)
    supplied = dict(config or {})
    if "rank" in supplied or "dimension_policy" in supplied:
        raise ValueError("Pass rank_cap and dimension_policy as explicit adapter arguments")
    supplied.update(rank=rank_cap, dimension_policy=dimension_policy)
    return fit_scaling_suite(a_x, a_y, b_x, b_y, c_x, c_y, seed=seed,
                             horizons=horizons, arm="standard", config=supplied)
