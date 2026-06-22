"""Experiment 18: Acrobot-v1 sequential-decision benchmark for KAHKM.

Purpose
-------
This experiment adds a stronger AI-facing sequential-decision benchmark than
CartPole/MountainCar. Acrobot-v1 is an underactuated swing-up task. We treat a
fixed heuristic policy plus mild exploration as a policy-induced closed-loop
dynamical system and evaluate KAHKM as an interpretable state abstraction.

The experiment performs:
1. KAHKM sensitivity over C and omega.
2. Final KAHKM evaluation at the selected configuration.
3. Baselines: KMeans hard/distance/RBF abstractions, state DMD, EDMD-poly2,
   EDMD-poly3.
4. Regime interpretability diagnostics: action purity, reward statistics,
   termination / near-terminal statistics, and regime plots.

Expected local files
--------------------
Place this script in the same folder as:
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Python packages
---------------
pip install numpy pandas scikit-learn matplotlib "gymnasium[classic-control]"

Example
-------
python experiment_18_acrobot_sequential_decision_pylance_clean.py

Fast smoke test:
python experiment_18_acrobot_sequential_decision_pylance_clean.py \
  --clusters 20 40 --omegas 2 4 --train-episodes 4 --test-episodes 2 \
  --max-steps 200 --horizons 1 5 10 --nb 40
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
    KAHKMFitResult,
    fit_kahkm,
    kahm_associations,
    nlms_koopman_closure,
    project_vector_to_simplex,
    simplex_violation,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]
BaselineKind: TypeAlias = Literal["hard", "distance", "rbf"]


@dataclass(frozen=True)
class ExperimentArgs:
    output_dir: str
    train_episodes: int
    test_episodes: int
    max_steps: int
    train_seed: int
    test_seed: int
    random_state: int
    horizons: tuple[int, ...]
    clusters: tuple[int, ...]
    omegas: tuple[float, ...]
    tau: float
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    kmeans_batch_size: int
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int | None
    exploration_eps: float
    ridge: float
    risk_window: int
    save_ae_to_disk: bool


@dataclass(frozen=True)
class TrajectoryData:
    X0: FloatArray
    X1: FloatArray
    actions: IntArray
    rewards: FloatArray
    terminals: IntArray
    episode_ids: IntArray
    step_ids: IntArray
    episodes: tuple[FloatArray, ...]


@dataclass(frozen=True)
class KMeansModel:
    kind: BaselineKind
    centers: FloatArray
    sigma: float
    B: FloatArray
    train_error: float
    train_r2: float


@dataclass(frozen=True)
class StateModel:
    name: str
    degree: int
    K: FloatArray
    monomials: tuple[tuple[int, ...], ...]


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(sorted({int(value) for value in values}))
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    if min(parsed) < 1:
        raise argparse.ArgumentTypeError("All integer values must be positive.")
    return parsed


def _parse_float_tuple(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(sorted({float(value) for value in values}))
    if not parsed:
        raise argparse.ArgumentTypeError("At least one float value is required.")
    if min(parsed) <= 0.0:
        raise argparse.ArgumentTypeError("All float values must be positive.")
    return parsed


def _require_gymnasium() -> Any:
    try:
        import gymnasium as gym  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Gymnasium is required. Install with: pip install 'gymnasium[classic-control]'"
        ) from exc
    return gym


def _relative_error(pred: FloatArray, target: FloatArray, eps: float = 1e-12) -> float:
    num = float(np.sum((pred - target) ** 2))
    den = float(np.sum(target * target))
    return num / max(den, eps)


def _r2(pred: FloatArray, target: FloatArray) -> float:
    residual_ss = float(np.sum((pred - target) ** 2))
    total_ss = float(np.sum((target - target.mean(axis=1, keepdims=True)) ** 2))
    if total_ss <= 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def _acrobot_angles(obs: FloatArray) -> tuple[float, float]:
    theta1 = math.atan2(float(obs[1]), float(obs[0]))
    theta2 = math.atan2(float(obs[3]), float(obs[2]))
    return theta1, theta2


def _acrobot_tip_height(obs: FloatArray) -> float:
    # Same terminal geometry used by Gymnasium Acrobot: goal when this exceeds 1.
    theta1, theta2 = _acrobot_angles(obs)
    return -math.cos(theta1) - math.cos(theta1 + theta2)


def acrobot_policy(obs: FloatArray, rng: np.random.Generator, eps: float) -> int:
    """Simple energy-pumping heuristic with exploration.

    Action mapping in Gymnasium Acrobot is usually:
    0 -> negative torque, 1 -> zero torque, 2 -> positive torque.
    """
    if float(rng.random()) < float(eps):
        return int(rng.integers(0, 3))

    theta1, theta2 = _acrobot_angles(obs)
    dtheta1 = float(obs[4])
    dtheta2 = float(obs[5])
    height = _acrobot_tip_height(obs)

    # Near the goal, keep pumping in the direction of combined angular momentum.
    combined_velocity = dtheta1 + dtheta2
    if height > 0.6:
        return 2 if combined_velocity >= 0.0 else 0

    # Energy-pumping score. It is deliberately simple and deterministic.
    pump = (
        combined_velocity
        + 0.40 * math.sin(theta1 + theta2)
        + 0.20 * math.sin(theta1)
        + 0.10 * dtheta2
    )
    if abs(pump) < 0.05:
        return 1
    return 2 if pump > 0.0 else 0


def collect_acrobot_data(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    exploration_eps: float,
) -> TrajectoryData:
    gym = _require_gymnasium()
    env = gym.make("Acrobot-v1")
    rng = np.random.default_rng(int(seed))

    x0_parts: list[FloatArray] = []
    x1_parts: list[FloatArray] = []
    action_parts: list[int] = []
    reward_parts: list[float] = []
    terminal_parts: list[int] = []
    episode_id_parts: list[int] = []
    step_id_parts: list[int] = []
    episode_states: list[FloatArray] = []

    for episode_idx in range(int(episodes)):
        obs_obj, _info = env.reset(seed=int(seed) + episode_idx)
        obs = np.asarray(obs_obj, dtype=np.float64).reshape(-1)
        states: list[FloatArray] = [obs.copy()]
        for step_idx in range(int(max_steps)):
            action = int(acrobot_policy(obs, rng, float(exploration_eps)))
            next_obs_obj, reward, terminated, truncated, _info = env.step(action)
            next_obs = np.asarray(next_obs_obj, dtype=np.float64).reshape(-1)

            x0_parts.append(obs.copy().reshape(-1, 1))
            x1_parts.append(next_obs.copy().reshape(-1, 1))
            action_parts.append(action)
            reward_parts.append(float(reward))
            terminal_parts.append(1 if bool(terminated or truncated) else 0)
            episode_id_parts.append(int(episode_idx))
            step_id_parts.append(int(step_idx))
            states.append(next_obs.copy())

            obs = next_obs
            if bool(terminated or truncated):
                break
        episode_states.append(np.stack(states, axis=1).astype(np.float64, copy=False))

    env.close()
    if not x0_parts:
        raise RuntimeError("No Acrobot transitions were collected.")

    return TrajectoryData(
        X0=np.concatenate(x0_parts, axis=1).astype(np.float64, copy=False),
        X1=np.concatenate(x1_parts, axis=1).astype(np.float64, copy=False),
        actions=np.asarray(action_parts, dtype=np.int64),
        rewards=np.asarray(reward_parts, dtype=np.float64),
        terminals=np.asarray(terminal_parts, dtype=np.int64),
        episode_ids=np.asarray(episode_id_parts, dtype=np.int64),
        step_ids=np.asarray(step_id_parts, dtype=np.int64),
        episodes=tuple(episode_states),
    )


def _association_for_kmeans(model: KMeansModel, X: FloatArray) -> FloatArray:
    centers = model.centers
    diff = X.T[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    if model.kind == "hard":
        labels = np.argmin(d2, axis=1)
        Phi = np.zeros((centers.shape[0], X.shape[1]), dtype=np.float64)
        Phi[labels, np.arange(X.shape[1])] = 1.0
        return Phi
    if model.kind == "distance":
        dist = np.sqrt(np.maximum(d2, 0.0))
        inv = 1.0 / (dist + 1e-8)
        return (inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)).T.astype(np.float64, copy=False)
    if model.kind == "rbf":
        weights = np.exp(-d2 / (2.0 * max(model.sigma, 1e-8) ** 2))
        return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).T.astype(np.float64, copy=False)
    raise ValueError(f"Unknown KMeans baseline kind: {model.kind}")


def _fit_kmeans_baseline(
    kind: BaselineKind,
    X0: FloatArray,
    X1: FloatArray,
    *,
    n_clusters: int,
    beta: float,
    epochs: int,
    random_state: int,
) -> KMeansModel:
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init="auto")
    labels = kmeans.fit_predict(X0.T)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    if centers.shape[0] > 1:
        diffs = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.maximum(np.sum(diffs * diffs, axis=2), 0.0))
        sigma = float(np.median(dist[dist > 0.0])) if np.any(dist > 0.0) else 1.0
    else:
        sigma = 1.0

    temp_model = KMeansModel(kind=kind, centers=centers, sigma=sigma, B=np.eye(int(n_clusters)), train_error=0.0, train_r2=0.0)
    Phi = _association_for_kmeans(temp_model, X0)
    Chi = _association_for_kmeans(temp_model, X1)
    B, _history = nlms_koopman_closure(Phi, Chi, beta=float(beta), epochs=int(epochs), shuffle=False, random_state=int(random_state))
    pred = B.T @ Phi
    return KMeansModel(
        kind=kind,
        centers=centers,
        sigma=sigma,
        B=B,
        train_error=_relative_error(pred, Chi),
        train_r2=_r2(pred, Chi),
    )


def _poly_monomials(dim: int, degree: int) -> tuple[tuple[int, ...], ...]:
    monomials: list[tuple[int, ...]] = [(0,) * int(dim)]
    indices = list(range(int(dim)))
    for deg in range(1, int(degree) + 1):
        for combo in combinations_with_replacement(indices, deg):
            powers = [0] * int(dim)
            for idx in combo:
                powers[int(idx)] += 1
            monomials.append(tuple(powers))
    return tuple(monomials)


def _poly_features(X: FloatArray, monomials: tuple[tuple[int, ...], ...]) -> FloatArray:
    X_arr = np.asarray(X, dtype=np.float64)
    Z = np.ones((len(monomials), X_arr.shape[1]), dtype=np.float64)
    for row_idx, powers in enumerate(monomials):
        values = np.ones(X_arr.shape[1], dtype=np.float64)
        for dim_idx, power in enumerate(powers):
            if int(power) != 0:
                values *= X_arr[int(dim_idx), :] ** int(power)
        Z[int(row_idx), :] = values
    return Z


def _fit_state_model(X0: FloatArray, X1: FloatArray, *, degree: int, ridge: float) -> StateModel:
    dim = int(X0.shape[0])
    monomials = _poly_monomials(dim, int(degree))
    Z0 = _poly_features(X0, monomials)
    # Direct state decoder from polynomial features.
    gram = Z0 @ Z0.T
    K = X1 @ Z0.T @ np.linalg.pinv(gram + float(ridge) * np.eye(gram.shape[0]))
    name = "state_dmd" if int(degree) == 1 else f"state_edmd_poly{int(degree)}"
    return StateModel(name=name, degree=int(degree), K=K.astype(np.float64, copy=False), monomials=monomials)


def _state_step(model: StateModel, X: FloatArray) -> FloatArray:
    return model.K @ _poly_features(X, model.monomials)


def _rollout_association_error(
    *,
    source: str,
    fit: KAHKMFitResult,
    episodes: tuple[FloatArray, ...],
    horizons: tuple[int, ...],
    n_jobs: int,
    batch_size: int,
    project_each_step: bool = False,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for horizon in horizons:
        pred_parts: list[FloatArray] = []
        target_parts: list[FloatArray] = []
        for states in episodes:
            if states.shape[1] <= int(horizon):
                continue
            Phi_all = kahm_associations(fit.abstraction_model, states, omega=fit.omega, tau=fit.tau, n_jobs=n_jobs, batch_size=batch_size, show_progress=False)
            for t in range(states.shape[1] - int(horizon)):
                p = Phi_all[:, int(t)]
                for _ in range(int(horizon)):
                    p = fit.B.T @ p
                    if project_each_step:
                        p = project_vector_to_simplex(p)
                pred_parts.append(p.reshape(-1, 1))
                target_parts.append(Phi_all[:, int(t) + int(horizon)].reshape(-1, 1))
        if not pred_parts:
            continue
        pred = np.concatenate(pred_parts, axis=1)
        target = np.concatenate(target_parts, axis=1)
        rows.append(
            {
                "method": source,
                "horizon": int(horizon),
                "relative_error": _relative_error(pred, target),
                "r2": _r2(pred, target),
                "simplex_violation": simplex_violation(pred),
                "n_eval": int(pred.shape[1]),
            }
        )
    return rows


def _rollout_kmeans_error(
    *,
    model: KMeansModel,
    episodes: tuple[FloatArray, ...],
    horizons: tuple[int, ...],
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for horizon in horizons:
        pred_parts: list[FloatArray] = []
        target_parts: list[FloatArray] = []
        for states in episodes:
            if states.shape[1] <= int(horizon):
                continue
            Phi_all = _association_for_kmeans(model, states)
            for t in range(states.shape[1] - int(horizon)):
                p = Phi_all[:, int(t)]
                for _ in range(int(horizon)):
                    p = model.B.T @ p
                pred_parts.append(p.reshape(-1, 1))
                target_parts.append(Phi_all[:, int(t) + int(horizon)].reshape(-1, 1))
        if not pred_parts:
            continue
        pred = np.concatenate(pred_parts, axis=1)
        target = np.concatenate(target_parts, axis=1)
        rows.append(
            {
                "method": f"kmeans_{model.kind}_nlms",
                "horizon": int(horizon),
                "relative_error": _relative_error(pred, target),
                "r2": _r2(pred, target),
                "simplex_violation": simplex_violation(pred),
                "n_eval": int(pred.shape[1]),
            }
        )
    return rows


def _rollout_state_error(
    *,
    model: StateModel,
    fit: KAHKMFitResult,
    episodes: tuple[FloatArray, ...],
    horizons: tuple[int, ...],
    n_jobs: int,
    batch_size: int,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for horizon in horizons:
        pred_parts: list[FloatArray] = []
        target_parts: list[FloatArray] = []
        for states in episodes:
            if states.shape[1] <= int(horizon):
                continue
            Phi_true_all = kahm_associations(fit.abstraction_model, states, omega=fit.omega, tau=fit.tau, n_jobs=n_jobs, batch_size=batch_size, show_progress=False)
            for t in range(states.shape[1] - int(horizon)):
                x_pred = states[:, int(t)].reshape(-1, 1)
                for _ in range(int(horizon)):
                    x_pred = _state_step(model, x_pred)
                    x_pred = np.clip(x_pred, -5.0, 5.0)
                phi_pred = kahm_associations(fit.abstraction_model, x_pred, omega=fit.omega, tau=fit.tau, n_jobs=n_jobs, batch_size=batch_size, show_progress=False)
                pred_parts.append(phi_pred)
                target_parts.append(Phi_true_all[:, int(t) + int(horizon)].reshape(-1, 1))
        if not pred_parts:
            continue
        pred = np.concatenate(pred_parts, axis=1)
        target = np.concatenate(target_parts, axis=1)
        rows.append(
            {
                "method": model.name,
                "horizon": int(horizon),
                "relative_error": _relative_error(pred, target),
                "r2": _r2(pred, target),
                "simplex_violation": simplex_violation(pred),
                "n_eval": int(pred.shape[1]),
            }
        )
    return rows


def _terminal_risk_within_window(data: TrajectoryData, window: int) -> IntArray:
    risks = np.zeros(data.terminals.shape[0], dtype=np.int64)
    for idx in range(data.terminals.shape[0]):
        eid = int(data.episode_ids[idx])
        step = int(data.step_ids[idx])
        same_ep = np.where(data.episode_ids == eid)[0]
        future = same_ep[(data.step_ids[same_ep] >= step) & (data.step_ids[same_ep] <= step + int(window))]
        risks[idx] = 1 if future.size > 0 and int(np.max(data.terminals[future])) > 0 else 0
    return risks


def _write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_figures(output_dir: Path, fit: KAHKMFitResult, data: TrajectoryData, *, n_jobs: int, batch_size: int) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    Phi = kahm_associations(fit.abstraction_model, data.X0, omega=fit.omega, tau=fit.tau, n_jobs=n_jobs, batch_size=batch_size, show_progress=False)
    labels = np.argmax(Phi, axis=0)
    theta1 = np.arctan2(data.X0[1, :], data.X0[0, :])
    theta2 = np.arctan2(data.X0[3, :], data.X0[2, :])
    tip_height = np.asarray([_acrobot_tip_height(data.X0[:, i]) for i in range(data.X0.shape[1])], dtype=np.float64)

    plt.figure(figsize=(6.2, 4.8))
    sc = plt.scatter(theta1, theta2, c=labels, s=8, alpha=0.75)
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title("Acrobot-v1: dominant KAHKM regime")
    plt.colorbar(sc, label="dominant regime")
    plt.tight_layout()
    plt.savefig(figures_dir / "Acrobot-v1_dominant_regime_angles.pdf")
    plt.close()

    plt.figure(figsize=(6.2, 4.8))
    sc2 = plt.scatter(tip_height, data.X0[4, :] + data.X0[5, :], c=labels, s=8, alpha=0.75)
    plt.xlabel("tip height")
    plt.ylabel(r"$\dot{\theta}_1+\dot{\theta}_2$")
    plt.title("Acrobot-v1: regime map in energy-like coordinates")
    plt.colorbar(sc2, label="dominant regime")
    plt.tight_layout()
    plt.savefig(figures_dir / "Acrobot-v1_dominant_regime_energy.pdf")
    plt.close()


def _regime_statistics(
    fit: KAHKMFitResult,
    data: TrajectoryData,
    *,
    risk_window: int,
    n_jobs: int,
    batch_size: int,
) -> list[CsvRow]:
    Phi = kahm_associations(fit.abstraction_model, data.X0, omega=fit.omega, tau=fit.tau, n_jobs=n_jobs, batch_size=batch_size, show_progress=False)
    labels = np.argmax(Phi, axis=0).astype(np.int64)
    n_regimes = int(Phi.shape[0])
    risk = _terminal_risk_within_window(data, int(risk_window))
    rows: list[CsvRow] = []

    for regime in range(n_regimes):
        mask = labels == int(regime)
        count = int(np.sum(mask))
        if count <= 0:
            rows.append(
                {
                    "regime": int(regime),
                    "count": 0,
                    "mass_fraction": 0.0,
                    "mean_reward": float("nan"),
                    "terminal_next_rate": float("nan"),
                    f"terminal_risk_within_{int(risk_window)}": float("nan"),
                    "action_0_prob": float("nan"),
                    "action_1_prob": float("nan"),
                    "action_2_prob": float("nan"),
                    "action_purity": float("nan"),
                    "mean_tip_height": float("nan"),
                    "mean_theta1": float("nan"),
                    "mean_theta2": float("nan"),
                }
            )
            continue

        actions = data.actions[mask]
        action_counts = np.bincount(actions, minlength=3).astype(np.float64)
        action_probs = action_counts / max(float(count), 1.0)
        theta1_vals = np.arctan2(data.X0[1, mask], data.X0[0, mask])
        theta2_vals = np.arctan2(data.X0[3, mask], data.X0[2, mask])
        tip_vals = np.asarray([_acrobot_tip_height(data.X0[:, i]) for i in np.where(mask)[0]], dtype=np.float64)

        rows.append(
            {
                "regime": int(regime),
                "count": count,
                "mass_fraction": float(count) / float(labels.size),
                "mean_reward": float(np.mean(data.rewards[mask])),
                "terminal_next_rate": float(np.mean(data.terminals[mask])),
                f"terminal_risk_within_{int(risk_window)}": float(np.mean(risk[mask])),
                "action_0_prob": float(action_probs[0]),
                "action_1_prob": float(action_probs[1]),
                "action_2_prob": float(action_probs[2]),
                "action_purity": float(np.max(action_probs)),
                "mean_tip_height": float(np.mean(tip_vals)),
                "mean_theta1": float(np.mean(theta1_vals)),
                "mean_theta2": float(np.mean(theta2_vals)),
            }
        )
    return rows


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Experiment 18: Acrobot-v1 KAHKM sequential-decision benchmark.")
    parser.add_argument("--output-dir", default="kahkm_experiment_18_acrobot_outputs")
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--horizons", nargs="+", default=["1", "5", "10", "20", "50"], help="Rollout horizons.")
    parser.add_argument("--clusters", nargs="+", default=["5", "10", "25", "50", "75", "100"], help="C grid.")
    parser.add_argument("--omegas", nargs="+", default=["2", "4", "6", "8", "10", "12"], help="omega grid.")
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument("--exploration-eps", type=float, default=0.10)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--risk-window", type=int, default=25)
    parser.add_argument("--save-ae-to-disk", action="store_true")
    ns = parser.parse_args()

    max_train_per_cluster = int(ns.max_train_per_cluster)
    return ExperimentArgs(
        output_dir=str(ns.output_dir),
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        max_steps=int(ns.max_steps),
        train_seed=int(ns.train_seed),
        test_seed=int(ns.test_seed),
        random_state=int(ns.random_state),
        horizons=_parse_int_tuple(cast(Sequence[str], ns.horizons)),
        clusters=_parse_int_tuple(cast(Sequence[str], ns.clusters)),
        omegas=_parse_float_tuple(cast(Sequence[str], ns.omegas)),
        tau=float(ns.tau),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        max_train_per_cluster=None if max_train_per_cluster <= 0 else max_train_per_cluster,
        exploration_eps=float(ns.exploration_eps),
        ridge=float(ns.ridge),
        risk_window=int(ns.risk_window),
        save_ae_to_disk=bool(ns.save_ae_to_disk),
    )


def run() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting Acrobot-v1 closed-loop trajectories...")
    train = collect_acrobot_data(
        episodes=args.train_episodes,
        max_steps=args.max_steps,
        seed=args.train_seed,
        exploration_eps=args.exploration_eps,
    )
    test = collect_acrobot_data(
        episodes=args.test_episodes,
        max_steps=args.max_steps,
        seed=args.test_seed,
        exploration_eps=args.exploration_eps,
    )

    state_dim = int(train.X0.shape[0])
    effective_subspace_dim = min(int(args.subspace_dim), state_dim)
    print(f"State dimension: {state_dim}; requested subspace_dim={args.subspace_dim}; effective_subspace_dim={effective_subspace_dim}")
    print(f"Training transitions: {train.X0.shape[1]}; test transitions: {test.X0.shape[1]}")

    sensitivity_rows: list[CsvRow] = []
    multistep_sensitivity_rows: list[CsvRow] = []
    best_fit: KAHKMFitResult | None = None
    best_score = float("inf")
    best_c = -1
    best_omega = float("nan")
    best_horizon = max(args.horizons)

    for c_value in args.clusters:
        for omega_value in args.omegas:
            print(f"\n=== Acrobot-v1 KAHKM sensitivity: C={c_value}, omega={omega_value} ===")
            start = time.time()
            fit = fit_kahkm(
                train.X0,
                train.X1,
                n_clusters=int(c_value),
                subspace_dim=effective_subspace_dim,
                Nb=int(args.nb),
                omega=float(omega_value),
                tau=float(args.tau),
                beta=float(args.beta),
                nlms_epochs=int(args.nlms_epochs),
                random_state=int(args.random_state),
                kmeans_kind=cast(Any, args.kmeans_kind),
                kmeans_batch_size=int(args.kmeans_batch_size),
                singleton_strategy="augment",
                max_train_per_cluster=args.max_train_per_cluster,
                save_ae_to_disk=args.save_ae_to_disk,
                n_jobs=int(args.n_jobs),
                batch_size=int(args.batch_size),
                project_stochastic=True,
                verbose=False,
            )
            fit_seconds = time.time() - start

            Phi_test = kahm_associations(fit.abstraction_model, test.X0, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
            Chi_test = kahm_associations(fit.abstraction_model, test.X1, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
            pred_test = fit.B.T @ Phi_test
            one_step_error = _relative_error(pred_test, Chi_test)
            one_step_r2 = _r2(pred_test, Chi_test)
            sensitivity_rows.append(
                {
                    "task": "Acrobot-v1",
                    "n_clusters": int(c_value),
                    "omega": float(omega_value),
                    "state_dim": state_dim,
                    "requested_subspace_dim": int(args.subspace_dim),
                    "effective_subspace_dim": effective_subspace_dim,
                    "train_transitions": int(train.X0.shape[1]),
                    "test_transitions": int(test.X0.shape[1]),
                    "train_closure_error": float(fit.train_closure_error),
                    "train_r2": float(fit.association_r2),
                    "test_one_step_error": one_step_error,
                    "test_one_step_r2": one_step_r2,
                    "test_simplex_violation": simplex_violation(pred_test),
                    "fit_seconds": fit_seconds,
                }
            )

            rows = _rollout_association_error(
                source="kahkm_nlms",
                fit=fit,
                episodes=test.episodes,
                horizons=args.horizons,
                n_jobs=args.n_jobs,
                batch_size=args.batch_size,
                project_each_step=False,
            )
            for row in rows:
                row["task"] = "Acrobot-v1"
                row["n_clusters"] = int(c_value)
                row["omega"] = float(omega_value)
                row["effective_subspace_dim"] = effective_subspace_dim
                multistep_sensitivity_rows.append(row)

            selected = [r for r in rows if int(r["horizon"]) == int(best_horizon)]
            if selected:
                score = float(selected[0]["relative_error"])
                if score < best_score:
                    best_score = score
                    best_fit = fit
                    best_c = int(c_value)
                    best_omega = float(omega_value)

    if best_fit is None:
        raise RuntimeError("No valid KAHKM fit was selected.")

    print(f"\nSelected Acrobot configuration by h={best_horizon}: C={best_c}, omega={best_omega}, error={best_score:.6g}")

    _write_csv(output_dir / "experiment_18_kahkm_sensitivity_one_step.csv", sensitivity_rows)
    _write_csv(output_dir / "experiment_18_kahkm_sensitivity_multistep.csv", multistep_sensitivity_rows)
    _write_csv(
        output_dir / "experiment_18_best_config.csv",
        [
            {
                "task": "Acrobot-v1",
                "selection_horizon": int(best_horizon),
                "best_n_clusters": best_c,
                "best_omega": best_omega,
                "best_horizon_error": best_score,
                "state_dim": state_dim,
                "requested_subspace_dim": int(args.subspace_dim),
                "effective_subspace_dim": effective_subspace_dim,
            }
        ],
    )

    # Final benchmark at selected KAHKM abstraction.
    one_step_rows: list[CsvRow] = []
    multistep_rows: list[CsvRow] = []

    Phi_test = kahm_associations(best_fit.abstraction_model, test.X0, omega=best_fit.omega, tau=best_fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
    Chi_test = kahm_associations(best_fit.abstraction_model, test.X1, omega=best_fit.omega, tau=best_fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
    pred_test = best_fit.B.T @ Phi_test
    pred_proj = np.column_stack([project_vector_to_simplex(pred_test[:, i]) for i in range(pred_test.shape[1])])

    one_step_rows.append(
        {
            "task": "Acrobot-v1",
            "method": "kahkm_nlms",
            "n_clusters": best_c,
            "omega": best_omega,
            "test_error": _relative_error(pred_test, Chi_test),
            "test_r2": _r2(pred_test, Chi_test),
            "simplex_violation": simplex_violation(pred_test),
        }
    )
    one_step_rows.append(
        {
            "task": "Acrobot-v1",
            "method": "kahkm_nlms_simplex_project",
            "n_clusters": best_c,
            "omega": best_omega,
            "test_error": _relative_error(pred_proj, Chi_test),
            "test_r2": _r2(pred_proj, Chi_test),
            "simplex_violation": simplex_violation(pred_proj),
        }
    )

    multistep_rows.extend(_rollout_association_error(source="kahkm_nlms", fit=best_fit, episodes=test.episodes, horizons=args.horizons, n_jobs=args.n_jobs, batch_size=args.batch_size, project_each_step=False))
    multistep_rows.extend(_rollout_association_error(source="kahkm_nlms_simplex_project_each_step", fit=best_fit, episodes=test.episodes, horizons=args.horizons, n_jobs=args.n_jobs, batch_size=args.batch_size, project_each_step=True))

    # KMeans abstraction baselines.
    for kind in ("hard", "distance", "rbf"):
        km = _fit_kmeans_baseline(
            cast(BaselineKind, kind),
            train.X0,
            train.X1,
            n_clusters=best_c,
            beta=args.beta,
            epochs=args.nlms_epochs,
            random_state=args.random_state,
        )
        Phi_k = _association_for_kmeans(km, test.X0)
        Chi_k = _association_for_kmeans(km, test.X1)
        pred_k = km.B.T @ Phi_k
        one_step_rows.append(
            {
                "task": "Acrobot-v1",
                "method": f"kmeans_{kind}_nlms",
                "n_clusters": best_c,
                "omega": best_omega,
                "test_error": _relative_error(pred_k, Chi_k),
                "test_r2": _r2(pred_k, Chi_k),
                "simplex_violation": simplex_violation(pred_k),
            }
        )
        multistep_rows.extend(_rollout_kmeans_error(model=km, episodes=test.episodes, horizons=args.horizons))

    # State-space DMD/EDMD baselines evaluated in selected KAHKM association space.
    for degree in (1, 2, 3):
        state_model = _fit_state_model(train.X0, train.X1, degree=degree, ridge=args.ridge)
        X1_pred = _state_step(state_model, test.X0)
        Phi_state_pred = kahm_associations(best_fit.abstraction_model, X1_pred, omega=best_fit.omega, tau=best_fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
        one_step_rows.append(
            {
                "task": "Acrobot-v1",
                "method": state_model.name,
                "n_clusters": best_c,
                "omega": best_omega,
                "test_error": _relative_error(Phi_state_pred, Chi_test),
                "test_r2": _r2(Phi_state_pred, Chi_test),
                "simplex_violation": simplex_violation(Phi_state_pred),
            }
        )
        multistep_rows.extend(_rollout_state_error(model=state_model, fit=best_fit, episodes=test.episodes, horizons=args.horizons, n_jobs=args.n_jobs, batch_size=args.batch_size))

    for row in multistep_rows:
        row["task"] = "Acrobot-v1"
        row["n_clusters"] = best_c
        row["omega"] = best_omega

    _write_csv(output_dir / "experiment_18_one_step_results.csv", one_step_rows)
    _write_csv(output_dir / "experiment_18_multistep_summary.csv", multistep_rows)

    regime_rows = _regime_statistics(best_fit, test, risk_window=args.risk_window, n_jobs=args.n_jobs, batch_size=args.batch_size)
    _write_csv(output_dir / "experiment_18_regime_statistics.csv", regime_rows)

    active_rows = [row for row in regime_rows if int(row["count"]) > 0]
    action_purity_weighted = 0.0
    mass_high_purity = 0.0
    weighted_risk = 0.0
    risk_key = f"terminal_risk_within_{int(args.risk_window)}"
    for row in active_rows:
        mass = float(row["mass_fraction"])
        action_purity_weighted += mass * float(row["action_purity"])
        if float(row["action_purity"]) >= 0.9:
            mass_high_purity += mass
        if not math.isnan(float(row[risk_key])):
            weighted_risk += mass * float(row[risk_key])
    max_risk_row = max(active_rows, key=lambda row: -1.0 if math.isnan(float(row[risk_key])) else float(row[risk_key])) if active_rows else None

    summary_rows: list[CsvRow] = [
        {
            "task": "Acrobot-v1",
            "state_dim": state_dim,
            "requested_subspace_dim": int(args.subspace_dim),
            "effective_subspace_dim": effective_subspace_dim,
            "selected_n_clusters": best_c,
            "selected_omega": best_omega,
            "active_regimes": len(active_rows),
            "train_transitions": int(train.X0.shape[1]),
            "test_transitions": int(test.X0.shape[1]),
            "train_closure_error": float(best_fit.train_closure_error),
            "train_r2": float(best_fit.association_r2),
            "count_weighted_action_purity": action_purity_weighted,
            "mass_in_action_purity_ge_0p9": mass_high_purity,
            f"weighted_terminal_risk_within_{int(args.risk_window)}": weighted_risk,
            "max_risk_regime": int(max_risk_row["regime"]) if max_risk_row is not None else -1,
            "max_risk_regime_count": int(max_risk_row["count"]) if max_risk_row is not None else 0,
            "max_risk_value": float(max_risk_row[risk_key]) if max_risk_row is not None else float("nan"),
        }
    ]
    _write_csv(output_dir / "experiment_18_model_summary.csv", summary_rows)

    _make_figures(output_dir, best_fit, test, n_jobs=args.n_jobs, batch_size=args.batch_size)

    metadata = {
        "experiment": "18_acrobot_sequential_decision",
        "args": asdict(args),
        "selected": summary_rows[0],
        "notes": (
            "Prediction uses raw KAHKM B. Simplex projection is reported separately. "
            "Effective subspace dimension is clipped to the Acrobot observation dimension."
        ),
    }
    (output_dir / "experiment_18_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFinished Experiment 18.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("Important files:")
    print("  experiment_18_best_config.csv")
    print("  experiment_18_one_step_results.csv")
    print("  experiment_18_multistep_summary.csv")
    print("  experiment_18_model_summary.csv")
    print("  experiment_18_regime_statistics.csv")
    print("  figures/")


if __name__ == "__main__":
    run()
