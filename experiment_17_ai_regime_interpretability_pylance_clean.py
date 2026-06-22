"""Experiment 17: AI-facing KAHKM regime interpretability diagnostics.

Purpose
-------
This experiment turns KAHKM into an interpretable state-abstraction analysis for
Gymnasium classic-control tasks. It trains a KAHKM closed-loop abstraction under
a fixed policy, then reports regime-level statistics that are natural in AI/RL:

- dominant-regime state summaries,
- action and reward summaries per regime,
- terminal/failure risk per regime,
- empirical dominant-regime transition matrices,
- learned row-stochastic transition matrix for interpretation,
- phase-space scatter plots colored by dominant regime.

This complements the quantitative Experiment 16 by supporting the manuscript's
claim that KAHKM observables are interpretable regime-association coordinates.

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
python experiment_17_ai_regime_interpretability_pylance_clean.py \
  --tasks CartPole-v1 MountainCar-v0 \
  --n-clusters 100 \
  --omega 2 \
  --subspace-dim 20 \
  --train-episodes 8 \
  --test-episodes 4 


"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
    fit_kahkm,
    kahm_associations,
    project_rows_to_simplex,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]
TaskName: TypeAlias = Literal["CartPole-v1", "MountainCar-v0"]
PolicyFn: TypeAlias = Callable[[FloatArray, np.random.Generator, float], int]


@dataclass(frozen=True)
class TaskConfig:
    env_id: TaskName
    n_clusters: int
    omega: float
    tau: float
    max_steps: int
    exploration_eps: float
    scatter_x_index: int
    scatter_y_index: int
    scatter_x_label: str
    scatter_y_label: str


@dataclass(frozen=True)
class ExperimentArgs:
    tasks: tuple[TaskName, ...]
    train_episodes: int
    test_episodes: int
    max_steps: int
    train_seed: int
    test_seed: int
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
    exploration_eps: float | None
    random_state: int
    risk_horizon: int


@dataclass(frozen=True)
class TransitionData:
    X0: FloatArray
    X1: FloatArray
    actions: IntArray
    rewards: FloatArray
    terminated: IntArray
    truncated: IntArray
    episode_id: IntArray
    step_id: IntArray
    steps_until_end: IntArray
    episodes: tuple[FloatArray, ...]


DEFAULT_TASK_CONFIGS: dict[TaskName, TaskConfig] = {
    "CartPole-v1": TaskConfig(
        env_id="CartPole-v1",
        n_clusters=20,
        omega=8.0,
        tau=1e-6,
        max_steps=250,
        exploration_eps=0.05,
        scatter_x_index=0,
        scatter_y_index=2,
        scatter_x_label="cart position",
        scatter_y_label="pole angle",
    ),
    "MountainCar-v0": TaskConfig(
        env_id="MountainCar-v0",
        n_clusters=20,
        omega=4.0,
        tau=1e-6,
        max_steps=200,
        exploration_eps=0.05,
        scatter_x_index=0,
        scatter_y_index=1,
        scatter_x_label="position",
        scatter_y_label="velocity",
    ),
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


def _require_gymnasium() -> Any:
    try:
        import gymnasium as gym  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Gymnasium is required for Experiment 17. Install with: "
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


def collect_data(
    task: TaskName,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    exploration_eps: float,
) -> TransitionData:
    gym = _require_gymnasium()
    env = gym.make(task)
    policy = _policy_for_task(task)
    rng = np.random.default_rng(int(seed))

    x0_parts: list[FloatArray] = []
    x1_parts: list[FloatArray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminated_flags: list[int] = []
    truncated_flags: list[int] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    steps_until_end_all: list[int] = []
    episode_states: list[FloatArray] = []

    for episode_idx in range(int(episodes)):
        local_start = len(actions)
        obs_obj, _info = env.reset(seed=int(seed) + episode_idx)
        obs = np.asarray(obs_obj, dtype=np.float64).reshape(-1)
        states: list[FloatArray] = [obs.copy()]
        local_steps = 0
        for step_idx in range(int(max_steps)):
            action = int(policy(obs, rng, float(exploration_eps)))
            next_obs_obj, reward, terminated, truncated, _info = env.step(action)
            next_obs = np.asarray(next_obs_obj, dtype=np.float64).reshape(-1)
            x0_parts.append(obs.copy().reshape(-1, 1))
            x1_parts.append(next_obs.copy().reshape(-1, 1))
            actions.append(action)
            rewards.append(float(reward))
            terminated_flags.append(1 if bool(terminated) else 0)
            truncated_flags.append(1 if bool(truncated) else 0)
            episode_ids.append(episode_idx)
            step_ids.append(step_idx)
            states.append(next_obs.copy())
            obs = next_obs
            local_steps += 1
            if bool(terminated or truncated):
                break
        for j in range(local_steps):
            steps_until_end_all.append(local_steps - j)
        if len(steps_until_end_all) != len(actions):
            raise RuntimeError("Internal episode accounting error.")
        episode_states.append(np.stack(states, axis=1).astype(np.float64, copy=False))
    env.close()

    if not x0_parts:
        raise RuntimeError(f"No transitions collected for {task}.")

    return TransitionData(
        X0=np.concatenate(x0_parts, axis=1).astype(np.float64, copy=False),
        X1=np.concatenate(x1_parts, axis=1).astype(np.float64, copy=False),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float64),
        terminated=np.asarray(terminated_flags, dtype=np.int64),
        truncated=np.asarray(truncated_flags, dtype=np.int64),
        episode_id=np.asarray(episode_ids, dtype=np.int64),
        step_id=np.asarray(step_ids, dtype=np.int64),
        steps_until_end=np.asarray(steps_until_end_all, dtype=np.int64),
        episodes=tuple(episode_states),
    )


def _safe_mean(values: FloatArray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values: FloatArray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values, ddof=0))


def empirical_transition_matrix(labels0: IntArray, labels1: IntArray, n_regimes: int) -> FloatArray:
    counts = np.zeros((int(n_regimes), int(n_regimes)), dtype=np.float64)
    for src, dst in zip(labels0.tolist(), labels1.tolist(), strict=False):
        counts[int(src), int(dst)] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, np.maximum(row_sums, 1.0), out=np.zeros_like(counts), where=row_sums > 0.0)


def regime_statistics(
    *,
    task: TaskName,
    data: TransitionData,
    Psi0: FloatArray,
    Psi1: FloatArray,
    B_interpretive: FloatArray,
    risk_horizon: int,
) -> tuple[list[CsvRow], FloatArray, FloatArray]:
    labels0 = np.argmax(Psi0, axis=0).astype(np.int64, copy=False)
    labels1 = np.argmax(Psi1, axis=0).astype(np.int64, copy=False)
    n_regimes = int(Psi0.shape[0])
    empirical_T = empirical_transition_matrix(labels0, labels1, n_regimes)
    learned_T = np.asarray(B_interpretive, dtype=np.float64)
    rows: list[CsvRow] = []
    risk = np.logical_and(data.steps_until_end <= int(risk_horizon), data.terminated > 0).astype(np.float64)

    action_values = sorted(set(int(v) for v in data.actions.tolist()))
    for regime in range(n_regimes):
        mask = labels0 == int(regime)
        count = int(np.sum(mask))
        row: CsvRow = {
            "task": task,
            "regime": int(regime),
            "count": count,
            "mass_fraction": float(count / max(1, int(labels0.size))),
            "mean_membership": _safe_mean(Psi0[int(regime), mask].astype(np.float64, copy=False)),
            "reward_mean": _safe_mean(data.rewards[mask].astype(np.float64, copy=False)),
            "termination_rate": _safe_mean(data.terminated[mask].astype(np.float64, copy=False)),
            "truncation_rate": _safe_mean(data.truncated[mask].astype(np.float64, copy=False)),
            f"terminal_risk_within_{int(risk_horizon)}": _safe_mean(risk[mask].astype(np.float64, copy=False)),
            "empirical_persistence": float(empirical_T[regime, regime]),
            "learned_persistence": float(learned_T[regime, regime]) if learned_T.shape[0] > regime else float("nan"),
        }
        for dim in range(int(data.X0.shape[0])):
            values = data.X0[dim, mask].astype(np.float64, copy=False)
            row[f"state{dim}_mean"] = _safe_mean(values)
            row[f"state{dim}_std"] = _safe_std(values)
        for action in action_values:
            if count == 0:
                row[f"action_{action}_rate"] = float("nan")
            else:
                row[f"action_{action}_rate"] = float(np.mean((data.actions[mask] == int(action)).astype(np.float64)))
        rows.append(row)
    return rows, empirical_T, learned_T


def write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_task_figures(
    *,
    task: TaskName,
    cfg: TaskConfig,
    out_dir: Path,
    data: TransitionData,
    Psi0: FloatArray,
    empirical_T: FloatArray,
    learned_T: FloatArray,
    stats_rows: Sequence[CsvRow],
    risk_horizon: int,
) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = np.argmax(Psi0, axis=0)

    plt.figure(figsize=(7.0, 5.0))
    plt.scatter(
        data.X0[int(cfg.scatter_x_index), :],
        data.X0[int(cfg.scatter_y_index), :],
        c=labels,
        s=8,
        alpha=0.75,
    )
    plt.xlabel(cfg.scatter_x_label)
    plt.ylabel(cfg.scatter_y_label)
    plt.title(f"{task}: dominant KAHKM regime")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{task}_dominant_regime_scatter.pdf")
    plt.close()

    plt.figure(figsize=(6.0, 5.0))
    plt.imshow(empirical_T, aspect="auto")
    plt.colorbar(label="empirical probability")
    plt.xlabel("next dominant regime")
    plt.ylabel("current dominant regime")
    plt.title(f"{task}: empirical dominant-regime transitions")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{task}_empirical_transition_matrix.pdf")
    plt.close()

    plt.figure(figsize=(6.0, 5.0))
    plt.imshow(learned_T, aspect="auto")
    plt.colorbar(label="projected row-stochastic coefficient")
    plt.xlabel("next regime")
    plt.ylabel("current regime")
    plt.title(f"{task}: learned interpretive transition matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{task}_learned_transition_matrix.pdf")
    plt.close()

    risk_key = f"terminal_risk_within_{int(risk_horizon)}"
    regimes = [int(row["regime"]) for row in stats_rows]
    risks = [float(row[risk_key]) for row in stats_rows]
    plt.figure(figsize=(7.0, 4.0))
    plt.bar(regimes, risks)
    plt.xlabel("dominant regime")
    plt.ylabel(f"terminal risk within {int(risk_horizon)} steps")
    plt.title(f"{task}: regime-level terminal risk")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{task}_terminal_risk_by_regime.pdf")
    plt.close()


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Experiment 17: KAHKM regime interpretability for Gymnasium tasks.")
    parser.add_argument("--tasks", nargs="+", default=["CartPole-v1", "MountainCar-v0"])
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--test-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses task default.")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=2000)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=80)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", type=str, default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--kmeans-batch-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="kahkm_experiment_17_outputs")
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--exploration-eps", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--risk-horizon", type=int, default=25)
    ns = parser.parse_args()
    max_train_per_cluster = None if int(ns.max_train_per_cluster) <= 0 else int(ns.max_train_per_cluster)
    return ExperimentArgs(
        tasks=_parse_tasks(cast(Sequence[str], ns.tasks)),
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        max_steps=int(ns.max_steps),
        train_seed=int(ns.train_seed),
        test_seed=int(ns.test_seed),
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
        save_ae_to_disk=bool(ns.save_ae_to_disk),
        exploration_eps=None if ns.exploration_eps is None else float(ns.exploration_eps),
        random_state=int(ns.random_state),
        risk_horizon=int(ns.risk_horizon),
    )


def run() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_all = time.time()
    all_stats: list[CsvRow] = []
    model_rows: list[CsvRow] = []

    for task in args.tasks:
        cfg = DEFAULT_TASK_CONFIGS[task]
        n_clusters = int(args.n_clusters if args.n_clusters is not None else cfg.n_clusters)
        omega = float(args.omega if args.omega is not None else cfg.omega)
        max_steps = int(args.max_steps if args.max_steps > 0 else cfg.max_steps)
        eps = float(args.exploration_eps if args.exploration_eps is not None else cfg.exploration_eps)

        print(f"\n=== {task}: collecting training and interpretation trajectories ===")
        train = collect_data(task, episodes=args.train_episodes, max_steps=max_steps, seed=args.train_seed, exploration_eps=eps)
        test = collect_data(task, episodes=args.test_episodes, max_steps=max_steps, seed=args.test_seed, exploration_eps=eps)

        print(f"=== {task}: fitting KAHKM ===")
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
        B_interpretive = np.asarray(fit.B_stochastic if fit.B_stochastic is not None else project_rows_to_simplex(fit.B), dtype=np.float64)

        print(f"=== {task}: computing regime diagnostics ===")
        Psi0 = kahm_associations(fit.abstraction_model, test.X0, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
        Psi1 = kahm_associations(fit.abstraction_model, test.X1, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
        stats_rows, empirical_T, learned_T = regime_statistics(
            task=task,
            data=test,
            Psi0=Psi0,
            Psi1=Psi1,
            B_interpretive=B_interpretive,
            risk_horizon=args.risk_horizon,
        )
        all_stats.extend(stats_rows)
        np.savetxt(out_dir / f"{task}_empirical_transition_matrix.csv", empirical_T, delimiter=",")
        np.savetxt(out_dir / f"{task}_learned_transition_matrix.csv", learned_T, delimiter=",")
        plot_task_figures(
            task=task,
            cfg=cfg,
            out_dir=out_dir,
            data=test,
            Psi0=Psi0,
            empirical_T=empirical_T,
            learned_T=learned_T,
            stats_rows=stats_rows,
            risk_horizon=args.risk_horizon,
        )
        model_rows.append({
            "task": task,
            "n_clusters": n_clusters,
            "omega": omega,
            "tau": float(args.tau),
            "train_transitions": int(train.X0.shape[1]),
            "test_transitions": int(test.X0.shape[1]),
            "fit_seconds": float(fit_seconds),
            "train_closure_error": float(fit.train_closure_error),
            "train_r2": float(fit.association_r2),
            "mean_empirical_persistence": float(np.mean(np.diag(empirical_T))),
            "mean_learned_persistence": float(np.mean(np.diag(learned_T))),
        })

    write_csv(out_dir / "experiment_17_model_summary.csv", model_rows)
    write_csv(out_dir / "experiment_17_regime_statistics.csv", all_stats)
    metadata = asdict(args)
    metadata["total_seconds"] = float(time.time() - start_all)
    metadata["note"] = "Regime interpretability diagnostics for closed-loop Gymnasium policy-induced dynamics."
    with (out_dir / "experiment_17_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nWrote outputs to:")
    for path in [
        out_dir / "experiment_17_model_summary.csv",
        out_dir / "experiment_17_regime_statistics.csv",
        out_dir / "figures",
        out_dir / "experiment_17_metadata.json",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    run()
