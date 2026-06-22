"""Experiment 14: training-trajectory-count sensitivity for KAHKM generalization.

Purpose
-------
Experiment 13 trained one KAHKM model per system on several complete training
trajectories and evaluated on unseen trajectories. This experiment asks how many
training trajectories are needed for robust unseen-trajectory generalization.

For each system and each requested training-trajectory count K, the script trains
on the first K seeds from --train-pool-seeds and evaluates on all --test-seeds.

System-specific default hyperparameters
---------------------------------------
- Duffing:      C=15, omega=12
- Van der Pol:  C=20, omega=4

Expected local files
--------------------
Place this script in the same folder as:

- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Run
---
python experiment_14_training_trajectory_count_pylance_clean.py

Fast smoke test:
python experiment_14_training_trajectory_count_pylance_clean.py --systems duffing --train-counts 1 2 --train-pool-seeds 0 1 --test-seeds 100 --n-steps-train 700 --n-steps-test 500 --horizons 1 10 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
    fit_kahkm,
    kahm_associations,
    make_duffing_snapshots,
    simplex_violation,
)

FloatArray: TypeAlias = NDArray[np.float64]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]
SystemName: TypeAlias = Literal["duffing", "vanderpol"]


@dataclass(frozen=True)
class SystemConfig:
    name: SystemName
    dt: float
    n_clusters: int
    omega: float
    tau: float


@dataclass(frozen=True)
class ExperimentArgs:
    systems: tuple[SystemName, ...]
    train_pool_seeds: tuple[int, ...]
    train_counts: tuple[int, ...]
    test_seeds: tuple[int, ...]
    n_steps_train: int
    n_steps_test: int
    horizons: tuple[int, ...]
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    batch_size: int
    n_jobs: int
    output_dir: str
    save_ae_to_disk: bool
    max_train_per_cluster: int | None
    vanderpol_mu: float


SYSTEM_CONFIGS: dict[SystemName, SystemConfig] = {
    "duffing": SystemConfig(name="duffing", dt=0.03, n_clusters=15, omega=12.0, tau=1e-6),
    "vanderpol": SystemConfig(name="vanderpol", dt=0.02, n_clusters=20, omega=4.0, tau=1e-6),
}


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def _parse_system_tuple(values: Sequence[str]) -> tuple[SystemName, ...]:
    parsed: list[SystemName] = []
    for value in values:
        if value not in {"duffing", "vanderpol"}:
            raise argparse.ArgumentTypeError("systems must contain only 'duffing' and/or 'vanderpol'.")
        parsed.append(value)  # type: ignore[arg-type]
    if not parsed:
        raise argparse.ArgumentTypeError("At least one system is required.")
    return tuple(parsed)


def make_vanderpol_snapshots(
    n_steps: int = 1600,
    dt: float = 0.02,
    seed: int = 0,
    mu: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Generate snapshot pairs from the Van der Pol oscillator."""
    rng = np.random.default_rng(int(seed))

    def f(z: FloatArray) -> FloatArray:
        q = float(z[0])
        p = float(z[1])
        return np.array([p, float(mu) * (1.0 - q * q) * p - q], dtype=np.float64)

    def rk4_step(z: FloatArray) -> FloatArray:
        k1 = f(z)
        k2 = f(z + 0.5 * float(dt) * k1)
        k3 = f(z + 0.5 * float(dt) * k2)
        k4 = f(z + float(dt) * k3)
        return np.asarray(z + (float(dt) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)

    z = np.array([2.0, 0.0], dtype=np.float64) + 0.05 * rng.normal(size=2)
    states = np.empty((2, int(n_steps) + 1), dtype=np.float64)
    states[:, 0] = z
    for t in range(int(n_steps)):
        z = rk4_step(z) + 1e-5 * rng.normal(size=2)
        states[:, t + 1] = z
    return states[:, :-1], states[:, 1:]


def make_snapshots(
    system: SystemName,
    *,
    n_steps: int,
    dt: float,
    seed: int,
    vanderpol_mu: float,
) -> tuple[FloatArray, FloatArray]:
    if system == "duffing":
        X0, X1 = make_duffing_snapshots(n_steps=int(n_steps), dt=float(dt), seed=int(seed))
        return np.asarray(X0, dtype=np.float64), np.asarray(X1, dtype=np.float64)
    if system == "vanderpol":
        return make_vanderpol_snapshots(n_steps=int(n_steps), dt=float(dt), seed=int(seed), mu=float(vanderpol_mu))
    raise ValueError(f"Unknown system: {system}")


def make_concatenated_training_data(
    system: SystemName,
    *,
    train_seeds: tuple[int, ...],
    n_steps_train: int,
    dt: float,
    vanderpol_mu: float,
) -> tuple[FloatArray, FloatArray]:
    X0_parts: list[FloatArray] = []
    X1_parts: list[FloatArray] = []
    for seed in train_seeds:
        X0, X1 = make_snapshots(
            system,
            n_steps=int(n_steps_train),
            dt=float(dt),
            seed=int(seed),
            vanderpol_mu=float(vanderpol_mu),
        )
        X0_parts.append(X0)
        X1_parts.append(X1)
    return (
        np.concatenate(X0_parts, axis=1).astype(np.float64, copy=False),
        np.concatenate(X1_parts, axis=1).astype(np.float64, copy=False),
    )


def _relative_error(pred: FloatArray, target: FloatArray, eps: float = 1e-12) -> float:
    numerator = float(np.sum((target - pred) ** 2))
    denominator = float(np.sum(target * target))
    return numerator / max(denominator, eps)


def _association_r2(pred: FloatArray, target: FloatArray) -> float:
    residual_ss = float(np.sum((target - pred) ** 2))
    total_ss = float(np.sum((target - target.mean(axis=1, keepdims=True)) ** 2))
    if total_ss <= 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def _rollout_associations(B: FloatArray, Phi0: FloatArray, horizon: int) -> FloatArray:
    pred = np.array(Phi0, dtype=np.float64, copy=True)
    BT = np.asarray(B.T, dtype=np.float64)
    for _ in range(int(horizon)):
        pred = np.asarray(BT @ pred, dtype=np.float64)
    return pred


def _evaluate_test_trajectory(
    *,
    system: SystemName,
    train_count: int,
    train_seeds_used: tuple[int, ...],
    test_seed: int,
    fit_result: object,
    X0_test: FloatArray,
    X1_test: FloatArray,
    horizons: tuple[int, ...],
    n_jobs: int,
    batch_size: int,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    abstraction_model = getattr(fit_result, "abstraction_model")
    B = np.asarray(getattr(fit_result, "B"), dtype=np.float64)
    omega = float(getattr(fit_result, "omega"))
    tau = float(getattr(fit_result, "tau"))

    n_test = int(X0_test.shape[1])
    for horizon in horizons:
        h = int(horizon)
        if h <= 0 or h > n_test:
            continue
        n_start = n_test - h + 1
        X_start = X0_test[:, :n_start]
        X_target = X1_test[:, h - 1 : h - 1 + n_start]
        Phi_start = np.asarray(
            kahm_associations(
                abstraction_model,
                X_start,
                omega=omega,
                tau=tau,
                n_jobs=int(n_jobs),
                batch_size=int(batch_size),
                show_progress=False,
            ),
            dtype=np.float64,
        )
        Phi_target = np.asarray(
            kahm_associations(
                abstraction_model,
                X_target,
                omega=omega,
                tau=tau,
                n_jobs=int(n_jobs),
                batch_size=int(batch_size),
                show_progress=False,
            ),
            dtype=np.float64,
        )
        pred = _rollout_associations(B, Phi_start, h)
        rows.append(
            {
                "system": system,
                "train_count": int(train_count),
                "train_seeds_used": " ".join(str(seed) for seed in train_seeds_used),
                "test_seed": int(test_seed),
                "horizon": h,
                "n_eval_starts": int(n_start),
                "relative_association_error": _relative_error(pred, Phi_target),
                "association_r2": _association_r2(pred, Phi_target),
                "simplex_violation": simplex_violation(pred),
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _write_csv(path: Path, rows: Sequence[CsvRow], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_test_rows(rows_in: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[tuple[str, int, int], list[CsvRow]] = {}
    for row in rows_in:
        key = (str(row["system"]), int(row["train_count"]), int(row["horizon"]))
        groups.setdefault(key, []).append(row)

    out: list[CsvRow] = []
    for (system, train_count, horizon), rows in sorted(groups.items()):
        errors = [float(row["relative_association_error"]) for row in rows]
        r2s = [float(row["association_r2"]) for row in rows]
        violations = [float(row["simplex_violation"]) for row in rows]
        train_seed_label = str(rows[0]["train_seeds_used"]) if rows else ""
        out.append(
            {
                "system": system,
                "train_count": int(train_count),
                "train_seeds_used": train_seed_label,
                "horizon": int(horizon),
                "n_test_trajectories": len(rows),
                "relative_association_error_mean": _mean(errors),
                "relative_association_error_std": _std(errors),
                "relative_association_error_min": float(min(errors)) if errors else float("nan"),
                "relative_association_error_max": float(max(errors)) if errors else float("nan"),
                "association_r2_mean": _mean(r2s),
                "association_r2_std": _std(r2s),
                "simplex_violation_mean": _mean(violations),
                "simplex_violation_max": float(max(violations)) if violations else float("nan"),
            }
        )
    return out


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Experiment 14: training-trajectory-count sensitivity for tuned KAHKM.")
    parser.add_argument("--systems", nargs="+", default=("duffing", "vanderpol"), help="Systems: duffing vanderpol")
    parser.add_argument("--train-pool-seeds", nargs="+", default=("0", "1", "2", "3", "4"))
    parser.add_argument("--train-counts", nargs="+", default=("1", "2", "3", "5"))
    parser.add_argument("--test-seeds", nargs="+", default=("100", "101", "102"))
    parser.add_argument("--n-steps-train", type=int, default=1200)
    parser.add_argument("--n-steps-test", type=int, default=800)
    parser.add_argument("--horizons", nargs="+", default=("1", "10", "50", "100", "200"))
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", type=str, default="full")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="kahkm_experiment_14_outputs")
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--max-train-per-cluster", type=int, default=None)
    parser.add_argument("--vanderpol-mu", type=float, default=1.0)
    ns = parser.parse_args()

    systems = _parse_system_tuple([str(v) for v in ns.systems])
    train_pool_seeds = _parse_int_tuple([str(v) for v in ns.train_pool_seeds])
    train_counts = _parse_int_tuple([str(v) for v in ns.train_counts])
    test_seeds = _parse_int_tuple([str(v) for v in ns.test_seeds])
    horizons = _parse_int_tuple([str(v) for v in ns.horizons])

    if int(ns.n_steps_train) <= 0 or int(ns.n_steps_test) <= 0:
        raise ValueError("n_steps_train and n_steps_test must be positive.")
    if max(horizons) > int(ns.n_steps_test):
        raise ValueError("The largest horizon cannot exceed n_steps_test.")
    if min(train_counts) <= 0:
        raise ValueError("train_counts must be positive.")
    if max(train_counts) > len(train_pool_seeds):
        raise ValueError("The largest train_count cannot exceed the number of train_pool_seeds.")

    return ExperimentArgs(
        systems=systems,
        train_pool_seeds=train_pool_seeds,
        train_counts=tuple(sorted(set(train_counts))),
        test_seeds=test_seeds,
        n_steps_train=int(ns.n_steps_train),
        n_steps_test=int(ns.n_steps_test),
        horizons=horizons,
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        kmeans_kind=str(ns.kmeans_kind),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        output_dir=str(ns.output_dir),
        save_ae_to_disk=bool(ns.save_ae_to_disk),
        max_train_per_cluster=None if ns.max_train_per_cluster is None else int(ns.max_train_per_cluster),
        vanderpol_mu=float(ns.vanderpol_mu),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_all = time.time()

    fit_rows: list[CsvRow] = []
    test_rows: list[CsvRow] = []

    for system in args.systems:
        cfg = SYSTEM_CONFIGS[system]
        for train_count in args.train_counts:
            train_seeds_used = tuple(args.train_pool_seeds[: int(train_count)])
            print(
                f"\n[{system}] K={train_count}, train_seeds={train_seeds_used}, "
                f"test_seeds={args.test_seeds}, C={cfg.n_clusters}, omega={cfg.omega}"
            )
            X0_train, X1_train = make_concatenated_training_data(
                system,
                train_seeds=train_seeds_used,
                n_steps_train=args.n_steps_train,
                dt=cfg.dt,
                vanderpol_mu=args.vanderpol_mu,
            )
            fit_start = time.time()
            fit = fit_kahkm(
                X0_train,
                X1_train,
                n_clusters=cfg.n_clusters,
                subspace_dim=args.subspace_dim,
                Nb=args.nb,
                omega=cfg.omega,
                tau=cfg.tau,
                beta=args.beta,
                nlms_epochs=args.nlms_epochs,
                random_state=int(train_seeds_used[0]),
                kmeans_kind=args.kmeans_kind,  # type: ignore[arg-type]
                max_train_per_cluster=args.max_train_per_cluster,
                save_ae_to_disk=args.save_ae_to_disk,
                n_jobs=args.n_jobs,
                batch_size=args.batch_size,
                project_stochastic=False,
                verbose=False,
            )
            fit_seconds = time.time() - fit_start
            fit_rows.append(
                {
                    "system": system,
                    "train_count": int(train_count),
                    "train_seeds_used": " ".join(str(seed) for seed in train_seeds_used),
                    "n_train_snapshots": int(X0_train.shape[1]),
                    "n_clusters": int(cfg.n_clusters),
                    "omega": float(cfg.omega),
                    "tau": float(cfg.tau),
                    "train_closure_error": float(getattr(fit, "train_closure_error")),
                    "train_association_r2": float(getattr(fit, "association_r2")),
                    "fit_seconds": float(fit_seconds),
                }
            )
            print(
                f"  train_err={float(getattr(fit, 'train_closure_error')):.6g}, "
                f"train_R2={float(getattr(fit, 'association_r2')):.6g}, fit_seconds={fit_seconds:.2f}"
            )

            for test_seed in args.test_seeds:
                X0_test, X1_test = make_snapshots(
                    system,
                    n_steps=args.n_steps_test,
                    dt=cfg.dt,
                    seed=int(test_seed),
                    vanderpol_mu=args.vanderpol_mu,
                )
                rows = _evaluate_test_trajectory(
                    system=system,
                    train_count=int(train_count),
                    train_seeds_used=train_seeds_used,
                    test_seed=int(test_seed),
                    fit_result=fit,
                    X0_test=X0_test,
                    X1_test=X1_test,
                    horizons=args.horizons,
                    n_jobs=args.n_jobs,
                    batch_size=args.batch_size,
                )
                test_rows.extend(rows)
                h1 = next((row for row in rows if int(row["horizon"]) == 1), None)
                hmax = next((row for row in rows if int(row["horizon"]) == max(args.horizons)), None)
                h1_error = float(h1["relative_association_error"]) if h1 is not None else float("nan")
                hmax_error = float(hmax["relative_association_error"]) if hmax is not None else float("nan")
                print(f"  test_seed={test_seed}: h1_err={h1_error:.6g}, h{max(args.horizons)}_err={hmax_error:.6g}")

    fit_fields = [
        "system",
        "train_count",
        "train_seeds_used",
        "n_train_snapshots",
        "n_clusters",
        "omega",
        "tau",
        "train_closure_error",
        "train_association_r2",
        "fit_seconds",
    ]
    test_fields = [
        "system",
        "train_count",
        "train_seeds_used",
        "test_seed",
        "horizon",
        "n_eval_starts",
        "relative_association_error",
        "association_r2",
        "simplex_violation",
    ]
    summary_fields = [
        "system",
        "train_count",
        "train_seeds_used",
        "horizon",
        "n_test_trajectories",
        "relative_association_error_mean",
        "relative_association_error_std",
        "relative_association_error_min",
        "relative_association_error_max",
        "association_r2_mean",
        "association_r2_std",
        "simplex_violation_mean",
        "simplex_violation_max",
    ]

    summary_rows = _summarize_test_rows(test_rows)
    _write_csv(output_dir / "experiment_14_fit_summary.csv", fit_rows, fit_fields)
    _write_csv(output_dir / "experiment_14_test_raw_results.csv", test_rows, test_fields)
    _write_csv(output_dir / "experiment_14_test_summary.csv", summary_rows, summary_fields)

    metadata = asdict(args)
    metadata["system_configs"] = {name: asdict(cfg) for name, cfg in SYSTEM_CONFIGS.items()}
    metadata["purpose"] = "Train on prefixes of train_pool_seeds; evaluate each model on complete unseen trajectories."
    metadata["total_seconds"] = float(time.time() - start_all)
    with (output_dir / "experiment_14_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone. Wrote outputs to:")
    print(f"  {output_dir / 'experiment_14_fit_summary.csv'}")
    print(f"  {output_dir / 'experiment_14_test_raw_results.csv'}")
    print(f"  {output_dir / 'experiment_14_test_summary.csv'}")


if __name__ == "__main__":
    main()
