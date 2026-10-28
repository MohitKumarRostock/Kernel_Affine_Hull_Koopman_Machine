"""Independent bounded episodes for explicit, testable risk guarantees.

Episodes are independent draws. Windows within an episode overlap and are
never asserted to be independent. Bounds below follow from the generators.
"""
from __future__ import annotations

import numpy as np


SYSTEM_BOUNDS = {
    "harmonic": (-1.0, 1.0),
    "quasiperiodic": (-1.0, 1.0),
    "nonlinear_ar": (-1.0, 1.0),
    "logistic": (-1.0, 1.0),
}


def make_episodes(system, n_episodes, length, seed):
    """Return ``(episodes, time, 1)`` observations, with independent rows.

    harmonic: amplitude 0.85, frequency uniform on [0.12,0.28], random phase,
      and additive independent U[-0.03,0.03] observation noise.
    quasiperiodic: two cosines with amplitudes .55 and .30, independent phases,
      base frequency U[.10,.22], incommensurate ratio sqrt(2), plus U[-.03,.03].
    nonlinear_ar: y[t+1] = tanh(1.25*y[t] - .65*y[t-1] + .25*sin(3*y[t])
      + U[-.10,.10]); initial values U[-.8,.8], 80 burn-in transitions.
    logistic: u[t+1]=r*u[t]*(1-u[t]), r~U[3.65,3.95], u[0]~U[.05,.95],
      40 burn-in transitions; y=2u-1. The random parameter is per episode.

    The stationary-risk assumption is deliberately unnecessary: calibration
    and deployment use the same finite-episode generation/origin protocol.
    """
    n, length = int(n_episodes), int(length)
    if n < 1 or length < 2:
        raise ValueError("Need at least one episode of length at least two.")
    if system not in SYSTEM_BOUNDS:
        raise ValueError(f"Unknown system {system!r}; choose {tuple(SYSTEM_BOUNDS)}.")
    rng = np.random.default_rng(seed)
    t = np.arange(length)[None, :]
    if system == "harmonic":
        phase = rng.uniform(0, 2 * np.pi, (n, 1))
        frequency = rng.uniform(0.12, 0.28, (n, 1))
        y = 0.85 * np.cos(phase + frequency * t)
        y += rng.uniform(-0.03, 0.03, size=(n, length))
    elif system == "quasiperiodic":
        phase = rng.uniform(0, 2 * np.pi, (n, 2))
        frequency = rng.uniform(0.10, 0.22, (n, 1))
        y = (0.55 * np.cos(phase[:, :1] + frequency * t)
             + 0.30 * np.cos(phase[:, 1:] + np.sqrt(2.0) * frequency * t))
        y += rng.uniform(-0.03, 0.03, size=(n, length))
    elif system == "nonlinear_ar":
        burn = 80
        series = np.zeros((n, length + burn))
        series[:, :2] = rng.uniform(-0.8, 0.8, size=(n, 2))
        innovations = rng.uniform(-0.10, 0.10, size=(n, length + burn - 2))
        for j in range(2, length + burn):
            last, before = series[:, j - 1], series[:, j - 2]
            series[:, j] = np.tanh(1.25 * last - 0.65 * before
                                  + 0.25 * np.sin(3 * last) + innovations[:, j - 2])
        y = series[:, burn:]
    else:
        burn = 40
        u = rng.uniform(0.05, 0.95, size=n)
        rate = rng.uniform(3.65, 3.95, size=n)
        y = np.empty((n, length))
        for j in range(length + burn):
            u = rate * u * (1.0 - u)
            if j >= burn:
                y[:, j - burn] = 2.0 * u - 1.0
    assert np.isfinite(y).all() and np.max(np.abs(y)) <= 1.0
    return y[:, :, None]


def histories_targets(episodes, history, horizons, origin_stride=1,
                      first_origin=None):
    """Build causal scalar histories and raw future targets.

    Returns X shape ``(n, origins, history)`` and Y shape ``(n, origins, H)``.
    Histories run oldest-to-newest. Origins share support across all horizons.
    ``first_origin`` permits identical origins when comparing history lengths.
    It must be fixed without using calibration targets.
    """
    episodes = np.asarray(episodes, dtype=float)
    if episodes.ndim == 2:
        episodes = episodes[:, :, None]
    if episodes.ndim != 3 or episodes.shape[2] != 1:
        raise ValueError("This generator currently supports scalar time series.")
    if not np.isfinite(episodes).all():
        raise ValueError("Episodes must be finite.")
    history, stride = int(history), int(origin_stride)
    horizons = np.asarray(horizons, dtype=int)
    if history < 1 or stride < 1 or horizons.ndim != 1 or len(horizons) < 1 or min(horizons) < 1:
        raise ValueError("history, stride, and forecast horizons must be positive.")
    first = history - 1 if first_origin is None else int(first_origin)
    if first < history - 1:
        raise ValueError("first_origin does not allow the requested causal history.")
    origins = np.arange(first, episodes.shape[1] - int(max(horizons)), stride)
    if len(origins) == 0:
        raise ValueError("Episodes are too short for the requested history and horizons.")
    history_indices = origins[:, None] - np.arange(history - 1, -1, -1)[None, :]
    target_indices = origins[:, None] + horizons[None, :]
    scalar = episodes[:, :, 0]
    return scalar[:, history_indices], scalar[:, target_indices]

