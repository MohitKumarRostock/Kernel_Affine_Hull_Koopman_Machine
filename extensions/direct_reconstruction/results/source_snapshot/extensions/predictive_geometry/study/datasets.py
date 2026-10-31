"""Frozen data laws for the KAHM geometry study.

Forecasting data reproduce the supplied upstream episode laws.  A row in a
forecasting design matrix is an origin, not an independent replicate: all
origins from one episode share a group identifier.  Callers must allocate
distinct seeds to training, validation, and test splits.

The branch experiment deliberately reuses one X and one noise vector across
target mixtures.  S and T are diagnostic latent variables, never training
partition labels.  There is no input noise in this experiment.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parent
_UPSTREAM_ROOT = _ROOT / "vendor/upstream/extensions/certified_forecasting"
_UPSTREAM_DATA = _UPSTREAM_ROOT / "src/certified_forecasting/data.py"
_UPSTREAM_OSCILLATORS = _UPSTREAM_ROOT / "experiments/run_oscillators.py"

# Import only the standalone upstream data module, not the experiment runner.
_SPEC = importlib.util.spec_from_file_location("_kahm_upstream_data", _UPSTREAM_DATA)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load the vendored generator at {_UPSTREAM_DATA}")
_DATA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DATA)
histories_targets = _DATA.histories_targets

EPISODE_LENGTH = 72
FORECAST_SYSTEMS = (
    "harmonic", "quasiperiodic", "nonlinear_ar", "logistic", "vanderpol", "duffing"
)
DEFAULT_SPLIT_EPISODES = {"train": 128, "validation": 128, "test": 512}
SYSTEM_BOUNDS = {system: (-1.0, 1.0) for system in FORECAST_SYSTEMS}
AIRFOIL_SPLIT_SEED = 2026103007
AIRFOIL_SHA256 = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if integer < 1 or integer != value:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def oscillator_episodes(system, n_episodes, length, seed, *, dt=0.025,
                        observation_interval=0.1, burn_time=10.0):
    """Exact standalone copy of the upstream float64 RK4 episode generator.

    Van der Pol: dx=v; dv=(1-x**2)*v-x.
    Duffing: dx=v; dv=-.2*v+x-x**3+.3*cos(1.2*t+phase).
    Each episode has independent x0,v0~Uniform[-2,2] and phase~U[0,2pi].
    The observation is .95*tanh(x)+U[-.02,.02], after a ten-unit burn-in.
    Noise is drawn before integration, preserving matched randomness when dt
    changes.  This is a fixed numerical law, not exact continuous ODE data.
    """
    n, length = int(n_episodes), int(length)
    if system not in {"vanderpol", "duffing"} or n < 1 or length < 2:
        raise ValueError("Invalid oscillator name, episode count, or length")
    if dt <= 0 or observation_interval <= 0 or burn_time < 0:
        raise ValueError("Integration times must be positive (burn can be zero)")
    steps_between = int(round(observation_interval / dt))
    burn_steps = int(round(burn_time / dt))
    if (steps_between < 1
            or not np.isclose(steps_between * dt, observation_interval, rtol=0, atol=1e-12)
            or not np.isclose(burn_steps * dt, burn_time, rtol=0, atol=1e-12)):
        raise ValueError("Observation interval and burn time must be integer multiples of dt")
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-2.0, 2.0, size=(n, 2))
    x, velocity = initial[:, 0].copy(), initial[:, 1].copy()
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n)
    noise = rng.uniform(-0.02, 0.02, size=(n, length))
    observed = np.empty((n, length))

    def acceleration(position, speed, t):
        if system == "vanderpol":
            return (1.0 - position * position) * speed - position
        return (-0.2 * speed + position - position**3
                + 0.3 * np.cos(1.2 * t + phase))

    total_steps = burn_steps + (length - 1) * steps_between
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for step in range(total_steps + 1):
            if step >= burn_steps and (step - burn_steps) % steps_between == 0:
                index = (step - burn_steps) // steps_between
                observed[:, index] = 0.95 * np.tanh(x) + noise[:, index]
            if step == total_steps:
                break
            t = step * dt
            k1x = velocity
            k1v = acceleration(x, velocity, t)
            k2x = velocity + 0.5 * dt * k1v
            k2v = acceleration(x + 0.5 * dt * k1x, k2x, t + 0.5 * dt)
            k3x = velocity + 0.5 * dt * k2v
            k3v = acceleration(x + 0.5 * dt * k2x, k3x, t + 0.5 * dt)
            k4x = velocity + dt * k3v
            k4v = acceleration(x + dt * k3x, k4x, t + dt)
            x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            velocity = velocity + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            if not np.isfinite(x).all() or not np.isfinite(velocity).all():
                raise FloatingPointError(f"{system}: non-finite state at integration step {step}")
    if not np.isfinite(observed).all() or np.any(np.abs(observed) > 0.97 + 1e-14):
        raise FloatingPointError("Generated sensor data violate the mathematical range")
    return observed[:, :, None]


def generate_episodes(system, n_episodes, length, seed):
    """Return (episodes, time, 1) using the exact inherited numerical laws."""
    if system not in FORECAST_SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; choose {FORECAST_SYSTEMS}")
    n_episodes = _positive_integer(n_episodes, "n_episodes")
    length = _positive_integer(length, "length")
    if length < 2:
        raise ValueError("length must be at least two")
    if system in {"vanderpol", "duffing"}:
        return oscillator_episodes(system, n_episodes, length, seed)
    return _DATA.make_episodes(system, n_episodes, length, seed)


def make_forecast_split(system, n_episodes, seed, history=12,
                        horizons=(1, 4, 12), stride=4) -> dict[str, Any]:
    """Build one split of independent length-72 forecasting episodes.

    X has shape (n_episodes*n_origins, history); Y has shape
    (n_episodes*n_origins, len(horizons)).  Rows are episode-major, then
    chronological origin order.  ``groups`` identifies the episode of each
    row.  Defaults give 13 origins per episode, at t=11,15,...,59.  Bounds
    concern the observed targets and are [-1,1] for all six systems.
    """
    n_episodes = _positive_integer(n_episodes, "n_episodes")
    history = _positive_integer(history, "history")
    stride = _positive_integer(stride, "stride")
    horizons = tuple(_positive_integer(h, "horizon") for h in horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be a nonempty sequence of distinct positive integers")
    episodes = generate_episodes(system, n_episodes, EPISODE_LENGTH, seed)
    x_episode, y_episode = histories_targets(
        episodes, history, horizons, origin_stride=stride
    )
    n_origins = x_episode.shape[1]
    origins = np.arange(history - 1, EPISODE_LENGTH - max(horizons), stride,
                        dtype=np.int64)
    x = np.ascontiguousarray(x_episode.reshape(-1, history))
    y = np.ascontiguousarray(y_episode.reshape(-1, len(horizons)))
    groups = np.repeat(np.arange(n_episodes, dtype=np.int64), n_origins)
    source = (_UPSTREAM_OSCILLATORS if system in {"vanderpol", "duffing"}
              else _UPSTREAM_DATA)
    metadata = {
        "system": system,
        "seed": int(seed),
        "n_episodes": n_episodes,
        "n_origins": n_origins,
        "n_rows": len(x),
        "episode_length": EPISODE_LENGTH,
        "history": history,
        "horizons": list(horizons),
        "origin_stride": stride,
        "origins": origins.tolist(),
        "episode_shape": list(episodes.shape),
        "X_episode_shape": list(x_episode.shape),
        "Y_episode_shape": list(y_episode.shape),
        "independence_unit": "episode",
        "within_episode_rows_independent": False,
        "generator_source": str(source.relative_to(_ROOT)),
        "generator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    if system in {"vanderpol", "duffing"}:
        metadata["integrator"] = {
            "method": "float64 RK4", "dt": 0.025,
            "observation_interval": 0.1, "burn_time": 10.0,
            "sensor": "0.95*tanh(position)+Uniform[-0.02,0.02]",
        }
    elif system == "nonlinear_ar":
        metadata["burn_in_stored_values"] = 80
    elif system == "logistic":
        metadata["burn_in"] = "upstream 40 burn-in transitions; preserve upstream indexing"
    return {
        "X": x, "Y": y, "groups": groups,
        "bounds": SYSTEM_BOUNDS[system],
        "n_episodes": n_episodes, "n_origins": n_origins,
        "origins": origins, "metadata": metadata,
    }


def make_branch_split(n, seed, curvature=1, separation=0.5,
                      noise=0.1) -> dict[str, Any]:
    """Draw independent points on two noise-free cosine branches.

    S is uniform on {-1,+1}; T is independently uniform on [-1,1].
    X=(T, curvature*cos(pi*T)+separation*S), with curvature 0 or 1.
    E is independent uniform noise on [-noise,noise], drawn once and reused
    by ``branch_targets`` for every lambda.  S/T/E are returned only for
    target construction and explicitly designated mechanism diagnostics.
    """
    n = _positive_integer(n, "n")
    curvature, separation, noise = float(curvature), float(separation), float(noise)
    if not np.isfinite([curvature, separation, noise]).all():
        raise ValueError("branch parameters must be finite")
    if curvature not in (0.0, 1.0):
        raise ValueError("curvature must be 0 or 1 in the declared experiment")
    if separation <= 0 or noise < 0:
        raise ValueError("separation must be positive and noise nonnegative")
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1.0, 1.0]), size=n)
    t = rng.uniform(-1.0, 1.0, size=n)
    e = rng.uniform(-noise, noise, size=n)
    x = np.column_stack((t, curvature * np.cos(np.pi * t) + separation * s))
    return {
        "X": x,
        "S": s, "T": t, "E": e,
        "groups": np.arange(n, dtype=np.int64),
        "bounds": (-2.0 - noise, 2.0 + noise),
        "metadata": {
            "system": "cosine_branches", "seed": int(seed), "n_rows": n,
            "curvature": curvature, "separation": separation,
            "noise_half_width": noise, "noise_variance": noise * noise / 3.0,
            "signal_variance_all_lambda": 1.0,
            "input_noise": 0.0,
            "independence_unit": "row",
            "latent_partition_labels_for_primary_fitting": False,
            "target_law": "sqrt(1-lambda)*S+sqrt(3*lambda)*T+E",
        },
    }


def branch_targets(data: dict[str, Any], lam) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy Y and noiseless m as (n,1) columns, without new RNG draws.

    m=sqrt(1-lambda)*S+sqrt(3*lambda)*T has variance one for all lambda in
    [0,1].  Because X identifies S and T exactly when separation is positive,
    the raw-input Bayes MSE is noise**2/3.  The common target range is
    [-(2+noise), 2+noise], hence [-2.1,2.1] with the default noise.
    """
    lam = float(lam)
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be finite and in [0,1]")
    s, t, e = (np.asarray(data[key], dtype=float) for key in ("S", "T", "E"))
    if s.ndim != 1 or s.shape != t.shape or s.shape != e.shape:
        raise ValueError("S, T, and E must be equal-length one-dimensional arrays")
    if not all(np.isfinite(value).all() for value in (s, t, e)):
        raise ValueError("S, T, and E must be finite")
    m = np.sqrt(1.0 - lam) * s + np.sqrt(3.0 * lam) * t
    return (m + e)[:, None], m[:, None]


def make_airfoil_split(split: str) -> dict[str, Any]:
    """Load one prespecified split of the optional Airfoil regression check.

    This is static sound-pressure regression, not forecasting.  Frequencies
    measured at the same four non-frequency operating settings stay together.
    The sorted 106 setting groups are permuted once with seed 2026103007;
    groups 0:64, 64:85, and 85:106 in that permutation form train, validation,
    and test.  No target values enter grouping or allocation.  The file hash
    is checked on every load.  No test performance has been used to choose
    this split.  Callers should freeze their protocol before evaluating test.

    X contains the five original features in UCI order.  Y is an (n,1)
    column in dB.  No transformation is fitted here.  There is no asserted
    population-wide target bound; ``bounds`` is therefore None.  A group
    split avoids sharing these settings but does not establish iid sampling
    from a broader physical population.
    """
    if split == "val":
        split = "validation"
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation (or val), or test")
    path = _ROOT / "airfoil_self_noise.dat"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AIRFOIL_SHA256:
        raise ValueError("Airfoil bytes differ from the recorded source snapshot")
    array = np.loadtxt(path)
    if array.shape != (1503, 6) or not np.isfinite(array).all():
        raise ValueError("Airfoil file must contain 1503 finite rows and 6 columns")
    _, group_ids = np.unique(array[:, 1:5], axis=0, return_inverse=True)
    n_groups = int(group_ids.max()) + 1
    if n_groups != 106:
        raise ValueError("Airfoil operating-setting grouping changed")
    permutation = np.random.default_rng(AIRFOIL_SPLIT_SEED).permutation(n_groups)
    chosen = {
        "train": permutation[:64],
        "validation": permutation[64:85],
        "test": permutation[85:],
    }[split]
    row_indices = np.flatnonzero(np.isin(group_ids, chosen))
    return {
        "X": np.ascontiguousarray(array[row_indices, :5]),
        "Y": np.ascontiguousarray(array[row_indices, 5:6]),
        "groups": group_ids[row_indices].astype(np.int64, copy=False),
        "row_indices": row_indices,
        "bounds": None,
        "metadata": {
            "system": "airfoil_static_regression",
            "split": split,
            "split_seed": AIRFOIL_SPLIT_SEED,
            "n_rows": len(row_indices),
            "n_groups": len(chosen),
            "group_ids": chosen.tolist(),
            "group_definition": "exact joint values of the four non-frequency input columns",
            "target_units": "scaled sound-pressure level, dB",
            "feature_names": [
                "frequency_hz", "attack_angle_degrees", "chord_length_m",
                "free_stream_velocity_m_per_s", "suction_side_displacement_thickness_m",
            ],
            "data_sha256": AIRFOIL_SHA256,
            "source_doi": "10.24432/C5VW2C",
            "source_url": "https://archive.ics.uci.edu/dataset/291/airfoil+self+noise",
            "source_mirror": (
                "https://raw.githubusercontent.com/DhannveenUppal7/Airfoil-Self-Noise/"
                "a152c1d1ca7d7a4b68c420071393692b621a55c0/airfoil_self_noise.dat"
            ),
            "license": "CC BY 4.0, according to the UCI source page",
            "statistical_scope": "descriptive fixed-cohort grouped holdout; no asserted iid physical population",
        },
    }
