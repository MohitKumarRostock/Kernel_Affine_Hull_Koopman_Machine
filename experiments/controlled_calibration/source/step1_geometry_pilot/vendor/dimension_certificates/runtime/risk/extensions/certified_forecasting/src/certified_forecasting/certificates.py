"""Finite-sample risk certificates; independent episodes are sampling units.

Every function assumes that the model, representation, bound, and family were
fixed without inspecting the certification episodes. Delta is the failure
probability of THIS call; callers must budget it across all reported choices.
The empirical-Bernstein formula is Maurer and Pontil (2009), Theorem 4, applied
also to the complementary variable for a lower bound.
"""
from __future__ import annotations

import math
import numpy as np


def mean_interval(values, bound: float, delta: float, method: str = "bernstein"):
    """One-sided lower and upper endpoints, EACH with failure probability delta.

    Simultaneous coverage of both endpoints therefore costs 2*delta. The
    declared bound is a population assumption, not an estimated sample maximum.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("Pass one scalar score per independent episode, as a 1D vector")
    if x.size < 2:
        raise ValueError("At least two independent scores are required")
    if not 0 < delta < 1 or not np.isfinite(bound) or bound <= 0:
        raise ValueError("Need 0 < delta < 1 and finite positive bound")
    if not np.all(np.isfinite(x)) or np.any(x < -1e-12) or np.any(x > bound + 1e-10):
        raise ValueError("Scores must be finite and lie in the declared [0,bound]")
    avg, variance = float(x.mean()), float(x.var(ddof=1))
    if method == "bernstein":
        logterm = math.log(2.0 / delta)
        radius = math.sqrt(2.0 * variance * logterm / x.size)
        radius += 7.0 * bound * logterm / (3.0 * (x.size - 1))
    elif method == "hoeffding":
        radius = bound * math.sqrt(math.log(1.0 / delta) / (2.0 * x.size))
    else:
        raise ValueError("method must be bernstein or hoeffding")
    return {
        "mean": avg, "variance": variance, "radius": radius,
        "lower": max(0.0, avg - radius),
        "upper": min(bound, avg + radius),
        "n_independent": int(x.size), "bound": float(bound),
        "delta_per_tail": float(delta), "method": method,
    }


def episode_mse(y_true, y_pred):
    """Per-episode per-horizon scalar MSE, input shape (episode,origin,horizon)."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if y.shape != pred.shape or y.ndim != 3:
        raise ValueError("Expected matching (episode,origin,horizon) arrays")
    return np.mean((y - pred) ** 2, axis=1)


def pair_witness_scores(features, targets, kappa, block_slices=None):
    """Disjoint episode pairs, same origin in each episode, one score/horizon.

    Features: (n,origins,p), targets: (n,origins,H), kappa: scalar or (H,).
    If block_slices is supplied, use the maximum Euclidean block distance;
    this matches a SUM of block Frobenius/column norms in the decoder.
    W = average_origins [(|Y-Y'|-kappa*||F-F'||)_+^2 / 2].
    Its expectation lower-bounds each kappa-Lipschitz predictor's MSE. If
    targets have diameter D, the score lies in [0,D^2/2]. An odd last episode
    is discarded deterministically. Episodes must be independent; origins
    within each episode may overlap and be arbitrarily dependent.
    """
    f, y = np.asarray(features, float), np.asarray(targets, float)
    if f.ndim != 3 or y.ndim != 3 or f.shape[:2] != y.shape[:2]:
        raise ValueError("Expected compatible episode/origin axes")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite inputs")
    k = np.broadcast_to(np.asarray(kappa, float), (y.shape[-1],))
    if np.any(k < 0) or not np.all(np.isfinite(k)):
        raise ValueError("Lipschitz budgets must be finite and nonnegative")
    end = 2 * (len(y) // 2)
    if end < 4:
        raise ValueError("Need at least four independent episodes")
    difference = f[:end:2] - f[1:end:2]
    if block_slices is None:
        distance = np.linalg.norm(difference, axis=-1)
    else:
        if not block_slices:
            raise ValueError("At least one feature block is required")
        covered = []
        for block in block_slices:
            if not isinstance(block, slice) or block.step not in (None,1):
                raise ValueError("Blocks must be contiguous slices")
            start = 0 if block.start is None else block.start
            stop = f.shape[-1] if block.stop is None else block.stop
            if not 0 <= start < stop <= f.shape[-1]:
                raise ValueError("Invalid or empty feature block")
            covered.extend(range(start,stop))
        if covered != list(range(f.shape[-1])):
            raise ValueError("Blocks must partition all feature coordinates exactly once")
        distance = np.max(np.stack([
            np.linalg.norm(difference[..., block], axis=-1)
            for block in block_slices
        ]), axis=0)
    gap = np.abs(y[:end:2] - y[1:end:2]) - distance[..., None] * k
    return 0.5 * np.mean(np.maximum(gap, 0.0) ** 2, axis=1)


def certify_forecast(features, targets, predictions, kappa, target_diameter,
                     delta_upper, delta_lower, method="bernstein", block_slices=None):
    """Matched MSE lower/upper bounds with separately budgeted one-sided tails.

    The lower endpoint is for the entire Lipschitz class at the frozen kappa;
    the upper endpoint is for this fitted predictor, provided it belongs to
    that class. Calling code verifies the model norm and records provenance.
    """
    losses = episode_mse(targets, predictions)
    witnesses = pair_witness_scores(features, targets, kappa, block_slices)
    h_count = losses.shape[1]
    d = np.broadcast_to(np.asarray(target_diameter, float), (h_count,))
    if not np.all(np.isfinite(d)) or np.any(d<=0):
        raise ValueError("Target diameters must be finite and positive")
    results = []
    for h in range(h_count):
        upper = mean_interval(losses[:, h], d[h] ** 2, delta_upper, method)
        lower = mean_interval(witnesses[:, h], d[h] ** 2 / 2.0, delta_lower, method)
        results.append({
            "mse_upper": upper["upper"], "mse_lower": lower["lower"],
            "rmse_upper": math.sqrt(upper["upper"]),
            "rmse_lower": math.sqrt(lower["lower"]),
            "empirical_rmse": math.sqrt(upper["mean"]),
            "upper_details": upper, "lower_details": lower,
        })
    return results, losses, witnesses


def paired_improvement(old_losses, new_losses, bound, delta, method="bernstein"):
    """LCB for E[old loss - new loss] on NEW independent validation episodes.

    Each loss is in [0,bound], hence shifted differences are in [0,2*bound].
    A strictly positive lower bound certifies improvement. The selection
    family must be fixed before these episodes or use a fresh block per step.
    """
    old, new = np.asarray(old_losses, float), np.asarray(new_losses, float)
    if old.shape != new.shape or old.ndim != 1:
        raise ValueError("Loss arrays must match and contain one scalar per independent episode")
    if np.any(old < 0) or np.any(new < 0) or np.any(old > bound) or np.any(new > bound):
        raise ValueError("Loss outside bound")
    result = mean_interval(old - new + bound, 2.0 * bound, delta, method)
    result["improvement_lower"] = result["lower"] - bound
    result["empirical_improvement"] = result["mean"] - bound
    return result
