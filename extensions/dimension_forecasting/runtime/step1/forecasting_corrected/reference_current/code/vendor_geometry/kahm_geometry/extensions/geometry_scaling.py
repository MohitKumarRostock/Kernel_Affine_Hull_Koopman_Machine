"""Training-fitted geometry scaling; targets and raw heads are untouched.

Tanh standardization uses the historical population-standard-deviation rule.
There is no second standardization after applying the nonlinear map.
"""
from __future__ import annotations

import numpy as np

GEOMETRY_SCALERS = ("standard", "minmax", "tanh_z_1", "tanh_z_0p5", "tanh_z_2", "tanh_raw")
TANH_MULTIPLIERS = {"tanh_z_1": 1.0, "tanh_z_0p5": 0.5, "tanh_z_2": 2.0}


def fit_minmax_geometry(x):
    """Fit on every supplied A history before reference/partition subsampling."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Need finite nonempty A histories")
    minimum, maximum = x.min(axis=0), x.max(axis=0)
    span = maximum - minimum
    if not np.isfinite(span).all():
        raise FloatingPointError("Nonfinite A coordinate range")
    return {"mode": "minmax", "minimum": minimum, "maximum": maximum,
            "range": span, "constant": span == 0.0,
            "clip": False, "fit_rows": int(len(x))}


def transform_minmax_geometry(x, state):
    """Use the frozen training range, with no query clipping.

    Exactly constant training coordinates map to zero for every query,
    including later observations different from that training constant.
    """
    if state["mode"] != "minmax" or state["clip"] is not False:
        raise ValueError("Only the fixed unclipped min--max arm is supported")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(state["minimum"]):
        raise ValueError("Geometry query width differs from fitted scaler")
    span = np.where(state["constant"], 1.0, state["range"])
    z = 2.0 * ((x - state["minimum"]) / span) - 1.0
    z[:, state["constant"]] = 0.0
    if not np.isfinite(z).all():
        raise FloatingPointError("Nonfinite min--max geometry query")
    return z


def fit_geometry_scaler(x, mode):
    """Fit only on all supplied training histories, before bank subsampling."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Need finite nonempty training histories")
    if mode == "minmax":
        return fit_minmax_geometry(x)
    if mode == "tanh_raw":
        return {"mode": mode, "fit_rows": int(len(x)), "width": int(x.shape[1]),
                "fitted_location_scale": False, "post_tanh_rescale": False}
    if mode not in TANH_MULTIPLIERS:
        raise ValueError("Unsupported alternative geometry scaler")
    mean, raw_std = x.mean(axis=0), x.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(raw_std).all():
        raise FloatingPointError("Nonfinite training geometry statistics")
    return {"mode": mode, "fit_rows": int(len(x)), "width": int(x.shape[1]),
            "mean": mean, "raw_standard_deviation": raw_std,
            "scale": np.where(raw_std > 1e-12, raw_std, 1.0),
            "standard_deviation_floor": 1e-12, "ddof": 0,
            "multiplier": TANH_MULTIPLIERS[mode], "fitted_location_scale": True,
            "post_tanh_rescale": False}


def transform_geometry(x, state):
    """Apply the frozen transformation without clipping or restandardization."""
    if state["mode"] == "minmax":
        return transform_minmax_geometry(x, state)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != state["width"] or not np.isfinite(x).all():
        raise ValueError("Need finite geometry queries of the fitted width")
    if state.get("post_tanh_rescale") is not False:
        raise ValueError("Tanh output must not be restandardized")
    if state["mode"] == "tanh_raw":
        argument = x
    elif state["mode"] in TANH_MULTIPLIERS:
        if state["multiplier"] != TANH_MULTIPLIERS[state["mode"]]:
            raise ValueError("Tanh multiplier differs from declared mode")
        argument = state["multiplier"] * ((x - state["mean"]) / state["scale"])
    else:
        raise ValueError("Unknown geometry transformation")
    if not np.isfinite(argument).all():
        raise FloatingPointError("Nonfinite tanh argument")
    z = np.tanh(argument)
    if not np.isfinite(z).all() or np.any(np.abs(z) > 1.0):
        raise FloatingPointError("Invalid tanh output")
    return z
