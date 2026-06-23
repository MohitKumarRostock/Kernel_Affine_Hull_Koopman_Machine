"""Experiment 16: AI-facing closed-loop Gymnasium state-abstraction benchmark.

Purpose
-------
This experiment evaluates KAHKM as an interpretable state-abstraction / latent
closed-loop dynamics model on standard reinforcement-learning control tasks.
The current KAHKM implementation learns autonomous dynamics X_{t+1}=F(X_t), so
we use fixed policies and model the induced closed-loop dynamics F_pi.

Tasks
-----
- CartPole-v1
- MountainCar-v0

Methods
-------
1. kahkm_nlms: KAHKM folding abstraction with NLMS Koopman closure.
2. kmeans_hard_nlms: hard KMeans regime abstraction with NLMS closure.
3. kmeans_distance_nlms: inverse-distance soft KMeans abstraction with NLMS closure.
4. kmeans_rbf_nlms: RBF-soft KMeans abstraction with NLMS closure.
5. state_dmd: state-space linear DMD rollout, evaluated after mapping predicted
   states through the learned KAHKM association map.
6. state_edmd_poly2 / state_edmd_poly3: polynomial EDMD state rollouts, also
   evaluated in KAHKM association space.

Interpretation
--------------
The KMeans methods test whether KAHKM improves over simpler regime abstractions.
The state-space Koopman baselines test whether a conventional state predictor
can match the KAHKM association rollout on the same policy-induced dynamics.

Expected local files
--------------------
Place this script in the same folder as:
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Python packages
---------------
pip install numpy scikit-learn matplotlib gymnasium[classic-control]

Run
---
python experiment_16_ai_classic_control_closed_loop_pylance_clean.py \
  --tasks CartPole-v1 MountainCar-v0 \
  --n-clusters 100 \
  --omega 2 \
  --subspace-dim 20 \
  --train-episodes 8 \
  --test-episodes 4 \
  --horizons 1 5 10 20 50



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
from typing import Any, Callable, Literal, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
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
TaskName: TypeAlias = Literal["CartPole-v1", "MountainCar-v0"]
BaselineKind: TypeAlias = Literal["hard", "distance", "rbf"]
PolicyFn: TypeAlias = Callable[[FloatArray, np.random.Generator, float], int]


@dataclass(frozen=True)
class TaskConfig:
    env_id: TaskName
    n_clusters: int
    omega: float
    tau: float
    max_steps: int
    exploration_eps: float


@dataclass(frozen=True)
class ExperimentArgs:
    tasks: tuple[TaskName, ...]
    train_episodes: int
    test_episodes: int
    max_steps: int
    train_seed: int
    test_seed: int
    horizons: tuple[int, ...]
    n_clusters: int | None
    omega: float | None
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
    output_dir: str
    save_ae_to_disk: bool
    ridge: float
    exploration_eps: float | None
    random_state: int


@dataclass(frozen=True)
class TrajectoryData:
    X0: FloatArray
    X1: FloatArray
    actions: IntArray
    rewards: FloatArray
    terminals: IntArray
    episodes: tuple[FloatArray, ...]


@dataclass(frozen=True)
class KMeansAbstraction:
    kind: BaselineKind
    centers: FloatArray
    sigma: float
    B: FloatArray
    train_error: float
    train_r2: float


@dataclass(frozen=True)
class StateKoopmanModel:
    name: str
    degree: int
    K: FloatArray
    monomials: tuple[tuple[int, ...], ...]


DEFAULT_TASK_CONFIGS: dict[TaskName, TaskConfig] = {
    "CartPole-v1": TaskConfig("CartPole-v1", n_clusters=20, omega=1.0, tau=1e-6, max_steps=250, exploration_eps=0.05),
    "MountainCar-v0": TaskConfig("MountainCar-v0", n_clusters=100, omega=1.0, tau=1e-6, max_steps=200, exploration_eps=0.05),
}


def _parse_tasks(values: Sequence[str]) -> tuple[TaskName, ...]:
    parsed: list[TaskName] = []
    for value in values:
        if value not in {"CartPole-v1", "MountainCar-v0"}:
            raise argparse.ArgumentTypeError("Tasks must be CartPole-v1 and/or MountainCar-v0.")
        parsed.append(cast(TaskName, value))
    if not parsed:
        raise argparse.ArgumentTypeError("At least one task is required.")
    return tuple(parsed)


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one horizon is required.")
    if min(parsed) < 1:
        raise argparse.ArgumentTypeError("All horizons must be positive.")
    return tuple(sorted(set(parsed)))


def _require_gymnasium() -> Any:
    try:
        import gymnasium as gym  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Gymnasium is required for Experiment 16. Install with: "
            "pip install 'gymnasium[classic-control]'"
        ) from exc
    return gym


def cartpole_policy(obs: FloatArray, rng: np.random.Generator, eps: float) -> int:
    if float(rng.random()) < float(eps):
        return int(rng.integers(0, 2))
    x, x_dot, theta, theta_dot = (float(v) for v in obs[:4])
    score = theta + 0.25 * theta_dot + 0.01 * x + 0.10 * x_dot
    return 1 if score > 0.0 else 0


def mountaincar_policy(obs: FloatArray, rng: np.random.Generator, eps: float) -> int:
    if float(rng.random()) < float(eps):
        return int(rng.integers(0, 3))
    position = float(obs[0])
    velocity = float(obs[1])
    if position > 0.45:
        return 2
    return 2 if velocity >= 0.0 else 0


def _policy_for_task(task: TaskName) -> PolicyFn:
    if task == "CartPole-v1":
        return cartpole_policy
    if task == "MountainCar-v0":
        return mountaincar_policy
    raise ValueError(f"Unknown task: {task}")


def collect_closed_loop_data(
    task: TaskName,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    exploration_eps: float,
) -> TrajectoryData:
    gym = _require_gymnasium()
    env = gym.make(task)
    policy = _policy_for_task(task)
    rng = np.random.default_rng(int(seed))

    x0_parts: list[FloatArray] = []
    x1_parts: list[FloatArray] = []
    action_parts: list[int] = []
    reward_parts: list[float] = []
    terminal_parts: list[int] = []
    episode_states: list[FloatArray] = []

    for episode_idx in range(int(episodes)):
        obs_obj, _info = env.reset(seed=int(seed) + episode_idx)
        obs = np.asarray(obs_obj, dtype=np.float64).reshape(-1)
        states: list[FloatArray] = [obs.copy()]
        for _step in range(int(max_steps)):
            action = int(policy(obs, rng, float(exploration_eps)))
            next_obs_obj, reward, terminated, truncated, _info = env.step(action)
            next_obs = np.asarray(next_obs_obj, dtype=np.float64).reshape(-1)
            x0_parts.append(obs.copy().reshape(-1, 1))
            x1_parts.append(next_obs.copy().reshape(-1, 1))
            action_parts.append(action)
            reward_parts.append(float(reward))
            terminal_parts.append(1 if bool(terminated or truncated) else 0)
            states.append(next_obs.copy())
            obs = next_obs
            if bool(terminated or truncated):
                break
        episode_states.append(np.stack(states, axis=1).astype(np.float64, copy=False))
    env.close()

    if not x0_parts:
        raise RuntimeError(f"No transitions collected for {task}.")

    return TrajectoryData(
        X0=np.concatenate(x0_parts, axis=1).astype(np.float64, copy=False),
        X1=np.concatenate(x1_parts, axis=1).astype(np.float64, copy=False),
        actions=np.asarray(action_parts, dtype=np.int64),
        rewards=np.asarray(reward_parts, dtype=np.float64),
        terminals=np.asarray(terminal_parts, dtype=np.int64),
        episodes=tuple(episode_states),
    )


def _relative_error(pred: FloatArray, target: FloatArray, eps: float = 1e-12) -> float:
    numerator = float(np.sum((target - pred) ** 2))
    denominator = float(np.sum(target * target))
    return numerator / max(denominator, eps)


def _r2(pred: FloatArray, target: FloatArray) -> float:
    residual_ss = float(np.sum((pred - target) ** 2))
    total_ss = float(np.sum((target - target.mean(axis=1, keepdims=True)) ** 2))
    if total_ss <= 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def _nearest_distances(X: FloatArray, centers: FloatArray) -> FloatArray:
    X_t = np.asarray(X.T, dtype=np.float64)
    C_t = np.asarray(centers.T, dtype=np.float64)
    x2 = np.sum(X_t * X_t, axis=1, keepdims=True)
    c2 = np.sum(C_t * C_t, axis=1, keepdims=True).T
    d2 = np.maximum(x2 + c2 - 2.0 * X_t @ C_t.T, 0.0)
    return np.sqrt(d2).T.astype(np.float64, copy=False)


def _kmeans_features(X: FloatArray, centers: FloatArray, *, kind: BaselineKind, sigma: float) -> FloatArray:
    d = _nearest_distances(X, centers)
    if kind == "hard":
        labels = np.argmin(d, axis=0)
        P = np.zeros((int(centers.shape[1]), int(X.shape[1])), dtype=np.float64)
        P[labels, np.arange(int(X.shape[1]))] = 1.0
        return P
    if kind == "distance":
        weights = 1.0 / (d + 1e-8)
        return (weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)).astype(np.float64, copy=False)
    if kind == "rbf":
        sigma_f = max(float(sigma), 1e-8)
        weights = np.exp(-0.5 * (d / sigma_f) ** 2)
        return (weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)).astype(np.float64, copy=False)
    raise ValueError(f"Unknown KMeans feature kind: {kind}")


def fit_kmeans_abstraction(
    X0: FloatArray,
    X1: FloatArray,
    *,
    n_clusters: int,
    kind: BaselineKind,
    beta: float,
    epochs: int,
    random_state: int,
) -> KMeansAbstraction:
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init="auto")
    kmeans.fit(np.asarray(X0.T, dtype=np.float64))
    centers = np.asarray(kmeans.cluster_centers_.T, dtype=np.float64)
    center_dist = _nearest_distances(centers, centers)
    nonzero = center_dist[center_dist > 1e-12]
    sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    Phi = _kmeans_features(X0, centers, kind=kind, sigma=sigma)
    Chi = _kmeans_features(X1, centers, kind=kind, sigma=sigma)
    B, _history = nlms_koopman_closure(Phi, Chi, beta=float(beta), epochs=int(epochs), shuffle=False, random_state=int(random_state))
    pred = np.asarray(B.T @ Phi, dtype=np.float64)
    return KMeansAbstraction(kind=kind, centers=centers, sigma=sigma, B=np.asarray(B, dtype=np.float64), train_error=_relative_error(pred, Chi), train_r2=_r2(pred, Chi))


def evaluate_kmeans_one_step(model: KMeansAbstraction, X0: FloatArray, X1: FloatArray) -> tuple[float, float, float]:
    Phi = _kmeans_features(X0, model.centers, kind=model.kind, sigma=model.sigma)
    Chi = _kmeans_features(X1, model.centers, kind=model.kind, sigma=model.sigma)
    pred = np.asarray(model.B.T @ Phi, dtype=np.float64)
    return _relative_error(pred, Chi), _r2(pred, Chi), simplex_violation(pred)


def _monomial_index(dim: int, degree: int) -> tuple[tuple[int, ...], ...]:
    terms: list[tuple[int, ...]] = [tuple()]
    for deg in range(1, int(degree) + 1):
        terms.extend(tuple(combo) for combo in combinations_with_replacement(range(int(dim)), deg))
    return tuple(terms)


def _poly_features(X: FloatArray, monomials: tuple[tuple[int, ...], ...]) -> FloatArray:
    X_arr = np.asarray(X, dtype=np.float64)
    N = int(X_arr.shape[1])
    Z = np.empty((len(monomials), N), dtype=np.float64)
    for row, combo in enumerate(monomials):
        if not combo:
            Z[row, :] = 1.0
        else:
            values = np.ones(N, dtype=np.float64)
            for idx in combo:
                values *= X_arr[int(idx), :]
            Z[row, :] = values
    return Z


def fit_state_koopman(X0: FloatArray, X1: FloatArray, *, degree: int, ridge: float) -> StateKoopmanModel:
    dim = int(X0.shape[0])
    monomials = _monomial_index(dim, int(degree))
    Z0 = _poly_features(X0, monomials)
    Z1 = _poly_features(X1, monomials)
    gram = Z0 @ Z0.T
    reg = float(ridge) * np.eye(int(gram.shape[0]), dtype=np.float64)
    right = Z0 @ Z1.T
    K = np.linalg.solve(gram + reg, right).T
    name = "state_dmd" if int(degree) == 1 else f"state_edmd_poly{int(degree)}"
    return StateKoopmanModel(name=name, degree=int(degree), K=np.asarray(K, dtype=np.float64), monomials=monomials)


def rollout_state_model(model: StateKoopmanModel, X_start: FloatArray, horizon: int) -> FloatArray:
    Z = _poly_features(X_start, model.monomials)
    for _ in range(int(horizon)):
        Z = np.asarray(model.K @ Z, dtype=np.float64)
    # Monomial order is constant, x_0, x_1, ... after the first row.
    dim = int(X_start.shape[0])
    return np.asarray(Z[1 : dim + 1, :], dtype=np.float64)


def _episode_start_target_pairs(episodes: tuple[FloatArray, ...], horizon: int) -> tuple[FloatArray, FloatArray]:
    starts: list[FloatArray] = []
    targets: list[FloatArray] = []
    for states in episodes:
        T = int(states.shape[1]) - 1
        if T < int(horizon):
            continue
        starts.append(states[:, : T - int(horizon) + 1])
        targets.append(states[:, int(horizon) : T + 1])
    if not starts:
        raise RuntimeError(f"No episode segments are long enough for horizon {horizon}.")
    return np.concatenate(starts, axis=1).astype(np.float64, copy=False), np.concatenate(targets, axis=1).astype(np.float64, copy=False)


def evaluate_association_rollout(
    *,
    method_name: str,
    horizon: int,
    start_phi: FloatArray,
    target_phi: FloatArray,
    B: FloatArray,
    project_each_step: bool = False,
) -> CsvRow:
    pred = np.asarray(start_phi, dtype=np.float64).copy()
    for _ in range(int(horizon)):
        pred = np.asarray(B.T @ pred, dtype=np.float64)
        if project_each_step:
            pred = np.apply_along_axis(project_vector_to_simplex, axis=0, arr=pred).astype(np.float64, copy=False)
    return {
        "method": method_name,
        "horizon": int(horizon),
        "metric_space": "own_association",
        "relative_error": _relative_error(pred, target_phi),
        "association_r2": _r2(pred, target_phi),
        "simplex_violation": simplex_violation(pred),
    }


def write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[CsvRow], keys: tuple[str, ...], value_keys: tuple[str, ...]) -> list[CsvRow]:
    groups: dict[tuple[CsvValue, ...], list[CsvRow]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    out: list[CsvRow] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        new_row: CsvRow = {keys[i]: key[i] for i in range(len(keys))}
        for value_key in value_keys:
            vals = np.asarray([float(r[value_key]) for r in group_rows], dtype=np.float64)
            new_row[f"{value_key}_mean"] = float(np.mean(vals))
            new_row[f"{value_key}_std"] = float(np.std(vals, ddof=0))
        new_row["n"] = int(len(group_rows))
        out.append(new_row)
    return out


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Experiment 16: Gymnasium closed-loop AI state-abstraction benchmark.")
    parser.add_argument("--tasks", nargs="+", default=["CartPole-v1", "MountainCar-v0"])
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses task default.")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=1000)
    parser.add_argument("--horizons", nargs="+", default=["1", "5", "10", "20", "50"])
    parser.add_argument("--n-clusters", type=int, default=None, help="Override task default.")
    parser.add_argument("--omega", type=float, default=None, help="Override task default.")
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", type=str, default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--kmeans-batch-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="kahkm_experiment_16_outputs")
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--no-save-ae-to-disk", action="store_true", help="Compatibility flag; disables saving autoencoder objects.")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--exploration-eps", type=float, default=None, help="Override task default.")
    parser.add_argument("--random-state", type=int, default=0)
    ns = parser.parse_args()
    max_train_per_cluster = None if int(ns.max_train_per_cluster) <= 0 else int(ns.max_train_per_cluster)
    return ExperimentArgs(
        tasks=_parse_tasks(cast(Sequence[str], ns.tasks)),
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        max_steps=int(ns.max_steps),
        train_seed=int(ns.train_seed),
        test_seed=int(ns.test_seed),
        horizons=_parse_int_tuple(cast(Sequence[str], ns.horizons)),
        n_clusters=None if ns.n_clusters is None else int(ns.n_clusters),
        omega=None if ns.omega is None else float(ns.omega),
        tau=float(ns.tau),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        max_train_per_cluster=max_train_per_cluster,
        output_dir=str(ns.output_dir),
        save_ae_to_disk=bool(ns.save_ae_to_disk) and not bool(getattr(ns, "no_save_ae_to_disk", False)),
        ridge=float(ns.ridge),
        exploration_eps=None if ns.exploration_eps is None else float(ns.exploration_eps),
        random_state=int(ns.random_state),
    )


def run() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    one_step_rows: list[CsvRow] = []
    multistep_rows: list[CsvRow] = []
    fit_rows: list[CsvRow] = []

    start_all = time.time()
    for task in args.tasks:
        cfg = DEFAULT_TASK_CONFIGS[task]
        n_clusters = int(args.n_clusters if args.n_clusters is not None else cfg.n_clusters)
        omega = float(args.omega if args.omega is not None else cfg.omega)
        max_steps = int(args.max_steps if args.max_steps > 0 else cfg.max_steps)
        eps = float(args.exploration_eps if args.exploration_eps is not None else cfg.exploration_eps)

        print(f"\n=== {task}: collecting closed-loop trajectories ===")
        train = collect_closed_loop_data(task, episodes=args.train_episodes, max_steps=max_steps, seed=args.train_seed, exploration_eps=eps)
        test = collect_closed_loop_data(task, episodes=args.test_episodes, max_steps=max_steps, seed=args.test_seed, exploration_eps=eps)
        print(f"train transitions={train.X0.shape[1]}, test transitions={test.X0.shape[1]}, dim={train.X0.shape[0]}")

        print(f"=== {task}: fitting KAHKM abstraction ===")
        t0 = time.time()
        fit = fit_kahkm(
            train.X0,
            train.X1,
            n_clusters=n_clusters,
            subspace_dim=args.subspace_dim,
            Nb=args.nb,
            omega=omega,
            tau=args.tau,
            beta=args.beta,
            nlms_epochs=args.nlms_epochs,
            random_state=args.random_state,
            kmeans_kind=cast(Any, args.kmeans_kind),
            kmeans_batch_size=args.kmeans_batch_size,
            max_train_per_cluster=args.max_train_per_cluster,
            save_ae_to_disk=args.save_ae_to_disk,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            project_stochastic=True,
            preload_classifier_after_fit=False,
            verbose=True,
        )
        fit_seconds = time.time() - t0
        Phi_test = kahm_associations(fit.abstraction_model, test.X0, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
        Chi_test = kahm_associations(fit.abstraction_model, test.X1, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
        pred_test = np.asarray(fit.B.T @ Phi_test, dtype=np.float64)
        one_step_rows.append({
            "task": task,
            "method": "kahkm_nlms",
            "metric_space": "own_association",
            "train_error": float(fit.train_closure_error),
            "train_r2": float(fit.association_r2),
            "test_error": _relative_error(pred_test, Chi_test),
            "test_r2": _r2(pred_test, Chi_test),
            "simplex_violation": simplex_violation(pred_test),
        })
        fit_rows.append({
            "task": task,
            "method": "kahkm_nlms",
            "n_clusters": n_clusters,
            "omega": omega,
            "tau": float(args.tau),
            "train_transitions": int(train.X0.shape[1]),
            "test_transitions": int(test.X0.shape[1]),
            "fit_seconds": float(fit_seconds),
            "train_error": float(fit.train_closure_error),
            "test_error": _relative_error(pred_test, Chi_test),
        })

        print(f"=== {task}: fitting KMeans abstraction baselines ===")
        kmeans_models: list[KMeansAbstraction] = []
        for kind in ("hard", "distance", "rbf"):
            model = fit_kmeans_abstraction(
                train.X0,
                train.X1,
                n_clusters=n_clusters,
                kind=cast(BaselineKind, kind),
                beta=args.beta,
                epochs=args.nlms_epochs,
                random_state=args.random_state,
            )
            kmeans_models.append(model)
            test_error, test_r2, test_violation = evaluate_kmeans_one_step(model, test.X0, test.X1)
            one_step_rows.append({
                "task": task,
                "method": f"kmeans_{kind}_nlms",
                "metric_space": "own_association",
                "train_error": float(model.train_error),
                "train_r2": float(model.train_r2),
                "test_error": float(test_error),
                "test_r2": float(test_r2),
                "simplex_violation": float(test_violation),
            })

        print(f"=== {task}: fitting state DMD/EDMD baselines ===")
        state_models = [
            fit_state_koopman(train.X0, train.X1, degree=1, ridge=args.ridge),
            fit_state_koopman(train.X0, train.X1, degree=2, ridge=args.ridge),
            fit_state_koopman(train.X0, train.X1, degree=3, ridge=args.ridge),
        ]
        for model in state_models:
            X1_pred = rollout_state_model(model, test.X0, horizon=1)
            Psi_pred = kahm_associations(fit.abstraction_model, X1_pred, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
            one_step_rows.append({
                "task": task,
                "method": model.name,
                "metric_space": "kahkm_association_of_predicted_state",
                "train_error": float("nan"),
                "train_r2": float("nan"),
                "test_error": _relative_error(Psi_pred, Chi_test),
                "test_r2": _r2(Psi_pred, Chi_test),
                "simplex_violation": simplex_violation(Psi_pred),
            })

        print(f"=== {task}: multi-step rollouts ===")
        for horizon in args.horizons:
            try:
                X_start, X_target = _episode_start_target_pairs(test.episodes, int(horizon))
            except RuntimeError:
                continue
            Psi_start_kahkm = kahm_associations(fit.abstraction_model, X_start, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
            Psi_target_kahkm = kahm_associations(fit.abstraction_model, X_target, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
            row = evaluate_association_rollout(
                method_name="kahkm_nlms",
                horizon=int(horizon),
                start_phi=Psi_start_kahkm,
                target_phi=Psi_target_kahkm,
                B=np.asarray(fit.B, dtype=np.float64),
            )
            row["task"] = task
            row["metric_space"] = "kahkm_association"
            multistep_rows.append(row)

            row_proj = evaluate_association_rollout(
                method_name="kahkm_nlms_simplex_project_each_step",
                horizon=int(horizon),
                start_phi=Psi_start_kahkm,
                target_phi=Psi_target_kahkm,
                B=np.asarray(fit.B, dtype=np.float64),
                project_each_step=True,
            )
            row_proj["task"] = task
            row_proj["metric_space"] = "kahkm_association"
            multistep_rows.append(row_proj)

            for km_model in kmeans_models:
                Psi_start = _kmeans_features(X_start, km_model.centers, kind=km_model.kind, sigma=km_model.sigma)
                Psi_target = _kmeans_features(X_target, km_model.centers, kind=km_model.kind, sigma=km_model.sigma)
                km_row = evaluate_association_rollout(
                    method_name=f"kmeans_{km_model.kind}_nlms",
                    horizon=int(horizon),
                    start_phi=Psi_start,
                    target_phi=Psi_target,
                    B=km_model.B,
                )
                km_row["task"] = task
                multistep_rows.append(km_row)

            for state_model in state_models:
                X_pred = rollout_state_model(state_model, X_start, horizon=int(horizon))
                Psi_pred = kahm_associations(fit.abstraction_model, X_pred, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
                multistep_rows.append({
                    "task": task,
                    "method": state_model.name,
                    "horizon": int(horizon),
                    "metric_space": "kahkm_association_of_predicted_state",
                    "relative_error": _relative_error(Psi_pred, Psi_target_kahkm),
                    "association_r2": _r2(Psi_pred, Psi_target_kahkm),
                    "simplex_violation": simplex_violation(Psi_pred),
                })

    write_csv(out_dir / "experiment_16_fit_summary.csv", fit_rows)
    write_csv(out_dir / "experiment_16_one_step_results.csv", one_step_rows)
    write_csv(out_dir / "experiment_16_multistep_results.csv", multistep_rows)
    write_csv(
        out_dir / "experiment_16_multistep_summary.csv",
        summarize(multistep_rows, keys=("task", "method", "metric_space", "horizon"), value_keys=("relative_error", "association_r2", "simplex_violation")),
    )
    metadata = asdict(args)
    metadata["total_seconds"] = float(time.time() - start_all)
    metadata["note"] = "Closed-loop Gymnasium policy-induced dynamics; KAHKM autonomous model learns F_pi."
    with (out_dir / "experiment_16_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nWrote outputs to:")
    for path in [
        out_dir / "experiment_16_fit_summary.csv",
        out_dir / "experiment_16_one_step_results.csv",
        out_dir / "experiment_16_multistep_results.csv",
        out_dir / "experiment_16_multistep_summary.csv",
        out_dir / "experiment_16_metadata.json",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    if not math.isfinite(1.0):
        raise RuntimeError("Unexpected math failure.")
    run()
