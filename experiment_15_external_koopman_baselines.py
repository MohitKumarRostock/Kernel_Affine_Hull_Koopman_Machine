"""Experiment 15: external Koopman baselines evaluated in KAHKM association space.

Purpose
-------
The previous experiments compared KAHKM against KMeans-style abstractions and
tested KAHKM robustness/generalization. This experiment adds external Koopman
state-space baselines:

- kahkm_nlms: KAHKM association rollout using the learned B matrix.
- state_dmd: linear DMD/least-squares state rollout, then mapped through KAHKM Ψ.
- state_edmd_poly2: polynomial EDMD degree 2 state-coordinate rollout, then mapped through KAHKM Ψ.
- state_edmd_poly3: polynomial EDMD degree 3 state-coordinate rollout, then mapped through KAHKM Ψ.

The comparison metric is association-space error against the true KAHKM
association of the future state. This keeps the target aligned with the paper:
regime-association Koopman closure, not universal raw-state prediction.

Expected local files
--------------------
Place this script in the same folder as:

- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Run
---
python experiment_15_external_koopman_baselines_pylance_clean.py

Fast smoke test:
python experiment_15_external_koopman_baselines_pylance_clean.py --systems duffing --train-seeds 0 --test-seeds 100 --n-steps-train 700 --n-steps-test 500 --horizons 1 10 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias

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
MethodName: TypeAlias = Literal[
    "kahkm_nlms",
    "state_dmd",
    "state_edmd_poly2",
    "state_edmd_poly3",
]


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
    train_seeds: tuple[int, ...]
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
    ridge: float


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
    x0_parts: list[FloatArray] = []
    x1_parts: list[FloatArray] = []
    for seed in train_seeds:
        X0, X1 = make_snapshots(
            system,
            n_steps=int(n_steps_train),
            dt=float(dt),
            seed=int(seed),
            vanderpol_mu=float(vanderpol_mu),
        )
        x0_parts.append(X0)
        x1_parts.append(X1)
    return (
        np.concatenate(x0_parts, axis=1).astype(np.float64, copy=False),
        np.concatenate(x1_parts, axis=1).astype(np.float64, copy=False),
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


def _state_relative_error(pred: FloatArray, target: FloatArray, eps: float = 1e-12) -> float:
    numerator = float(np.sum((target - pred) ** 2))
    denominator = float(np.sum(target * target))
    return numerator / max(denominator, eps)


def _fit_linear_map(X0: FloatArray, X1: FloatArray, ridge: float) -> FloatArray:
    """Fit A such that X1 ≈ A @ X0."""
    X0_arr = np.asarray(X0, dtype=np.float64)
    X1_arr = np.asarray(X1, dtype=np.float64)
    gram = X0_arr @ X0_arr.T
    reg = float(ridge) * np.eye(int(gram.shape[0]), dtype=np.float64)
    right = X0_arr @ X1_arr.T
    solved = np.linalg.solve(gram + reg, right)
    return np.asarray(solved.T, dtype=np.float64)


def _poly_features_2d(X: FloatArray, degree: int) -> FloatArray:
    """Polynomial features for 2D states, with state coordinates in rows 1 and 2."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.shape[0] != 2:
        raise ValueError("Polynomial EDMD baseline currently expects 2D states.")
    q = X_arr[0, :]
    p = X_arr[1, :]
    features: list[FloatArray] = [
        np.ones_like(q, dtype=np.float64).reshape(1, -1),
        q.reshape(1, -1),
        p.reshape(1, -1),
    ]
    if int(degree) >= 2:
        features.extend(
            [
                (q * q).reshape(1, -1),
                (q * p).reshape(1, -1),
                (p * p).reshape(1, -1),
            ]
        )
    if int(degree) >= 3:
        features.extend(
            [
                (q * q * q).reshape(1, -1),
                (q * q * p).reshape(1, -1),
                (q * p * p).reshape(1, -1),
                (p * p * p).reshape(1, -1),
            ]
        )
    return np.concatenate(features, axis=0).astype(np.float64, copy=False)


def _fit_lifted_operator(Z0: FloatArray, Z1: FloatArray, ridge: float) -> FloatArray:
    """Fit K such that Z1 ≈ K @ Z0."""
    Z0_arr = np.asarray(Z0, dtype=np.float64)
    Z1_arr = np.asarray(Z1, dtype=np.float64)
    gram = Z0_arr @ Z0_arr.T
    reg = float(ridge) * np.eye(int(gram.shape[0]), dtype=np.float64)
    right = Z0_arr @ Z1_arr.T
    solved = np.linalg.solve(gram + reg, right)
    return np.asarray(solved.T, dtype=np.float64)


def _rollout_linear_state(A: FloatArray, X0: FloatArray, horizon: int) -> FloatArray:
    pred = np.asarray(X0, dtype=np.float64).copy()
    A_arr = np.asarray(A, dtype=np.float64)
    for _ in range(int(horizon)):
        pred = np.asarray(A_arr @ pred, dtype=np.float64)
    return pred


def _rollout_lifted_state(K: FloatArray, X0: FloatArray, degree: int, horizon: int) -> FloatArray:
    z = _poly_features_2d(X0, int(degree))
    K_arr = np.asarray(K, dtype=np.float64)
    for _ in range(int(horizon)):
        z = np.asarray(K_arr @ z, dtype=np.float64)
    return np.asarray(z[1:3, :], dtype=np.float64)


def _rollout_kahkm_associations(B: FloatArray, Phi0: FloatArray, horizon: int) -> FloatArray:
    pred = np.asarray(Phi0, dtype=np.float64).copy()
    BT = np.asarray(B.T, dtype=np.float64)
    for _ in range(int(horizon)):
        pred = np.asarray(BT @ pred, dtype=np.float64)
    return pred


def _method_associations(
    *,
    method: MethodName,
    B_kahkm: FloatArray,
    A_dmd: FloatArray,
    K_poly2: FloatArray,
    K_poly3: FloatArray,
    Phi_start: FloatArray,
    X_start: FloatArray,
    abstraction_model: Any,
    omega: float,
    tau: float,
    horizon: int,
    n_jobs: int,
    batch_size: int,
) -> tuple[FloatArray, FloatArray | None]:
    """Return predicted associations and optional predicted states."""
    if method == "kahkm_nlms":
        return _rollout_kahkm_associations(B_kahkm, Phi_start, int(horizon)), None

    if method == "state_dmd":
        state_pred = _rollout_linear_state(A_dmd, X_start, int(horizon))
    elif method == "state_edmd_poly2":
        state_pred = _rollout_lifted_state(K_poly2, X_start, degree=2, horizon=int(horizon))
    elif method == "state_edmd_poly3":
        state_pred = _rollout_lifted_state(K_poly3, X_start, degree=3, horizon=int(horizon))
    else:
        raise ValueError(f"Unknown method: {method}")

    psi_pred = np.asarray(
        kahm_associations(
            abstraction_model,
            state_pred,
            omega=float(omega),
            tau=float(tau),
            n_jobs=int(n_jobs),
            batch_size=int(batch_size),
            show_progress=False,
        ),
        dtype=np.float64,
    )
    return psi_pred, np.asarray(state_pred, dtype=np.float64)


def _evaluate_test_trajectory(
    *,
    system: SystemName,
    train_seeds_used: tuple[int, ...],
    test_seed: int,
    abstraction_model: Any,
    omega: float,
    tau: float,
    B_kahkm: FloatArray,
    A_dmd: FloatArray,
    K_poly2: FloatArray,
    K_poly3: FloatArray,
    X0_test: FloatArray,
    X1_test: FloatArray,
    horizons: tuple[int, ...],
    n_jobs: int,
    batch_size: int,
) -> list[CsvRow]:
    methods: tuple[MethodName, ...] = ("kahkm_nlms", "state_dmd", "state_edmd_poly2", "state_edmd_poly3")
    rows: list[CsvRow] = []
    n_test = int(X0_test.shape[1])

    for horizon in horizons:
        h = int(horizon)
        if h <= 0 or h > n_test:
            continue

        n_start = n_test - h + 1
        X_start = np.asarray(X0_test[:, :n_start], dtype=np.float64)
        X_target = np.asarray(X1_test[:, h - 1 : h - 1 + n_start], dtype=np.float64)

        Phi_start = np.asarray(
            kahm_associations(
                abstraction_model,
                X_start,
                omega=float(omega),
                tau=float(tau),
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
                omega=float(omega),
                tau=float(tau),
                n_jobs=int(n_jobs),
                batch_size=int(batch_size),
                show_progress=False,
            ),
            dtype=np.float64,
        )

        for method in methods:
            Phi_pred, X_pred = _method_associations(
                method=method,
                B_kahkm=B_kahkm,
                A_dmd=A_dmd,
                K_poly2=K_poly2,
                K_poly3=K_poly3,
                Phi_start=Phi_start,
                X_start=X_start,
                abstraction_model=abstraction_model,
                omega=float(omega),
                tau=float(tau),
                horizon=h,
                n_jobs=int(n_jobs),
                batch_size=int(batch_size),
            )
            state_error = float("nan") if X_pred is None else _state_relative_error(X_pred, X_target)
            rows.append(
                {
                    "system": system,
                    "method": method,
                    "train_seeds_used": " ".join(str(seed) for seed in train_seeds_used),
                    "test_seed": int(test_seed),
                    "horizon": h,
                    "n_eval_starts": int(n_start),
                    "relative_association_error": _relative_error(Phi_pred, Phi_target),
                    "association_r2": _association_r2(Phi_pred, Phi_target),
                    "simplex_violation": simplex_violation(Phi_pred),
                    "state_relative_error": state_error,
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
    groups: dict[tuple[str, str, int], list[CsvRow]] = {}
    for row in rows_in:
        key = (str(row["system"]), str(row["method"]), int(row["horizon"]))
        groups.setdefault(key, []).append(row)

    out: list[CsvRow] = []
    for (system, method, horizon), rows in sorted(groups.items()):
        errors = [float(row["relative_association_error"]) for row in rows]
        r2s = [float(row["association_r2"]) for row in rows]
        violations = [float(row["simplex_violation"]) for row in rows]
        state_errors = [float(row["state_relative_error"]) for row in rows if np.isfinite(float(row["state_relative_error"]))]
        train_seed_label = str(rows[0]["train_seeds_used"]) if rows else ""
        out.append(
            {
                "system": system,
                "method": method,
                "train_seeds_used": train_seed_label,
                "horizon": int(horizon),
                "n_test_trajectories": len({int(row["test_seed"]) for row in rows}),
                "relative_association_error_mean": _mean(errors),
                "relative_association_error_std": _std(errors),
                "relative_association_error_min": float(min(errors)) if errors else float("nan"),
                "relative_association_error_max": float(max(errors)) if errors else float("nan"),
                "association_r2_mean": _mean(r2s),
                "association_r2_std": _std(r2s),
                "simplex_violation_mean": _mean(violations),
                "simplex_violation_max": float(max(violations)) if violations else float("nan"),
                "state_relative_error_mean": _mean(state_errors),
                "state_relative_error_std": _std(state_errors),
            }
        )
    return out


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Experiment 15: external Koopman baselines in KAHKM association space.")
    parser.add_argument("--systems", nargs="+", default=("duffing", "vanderpol"), help="Systems: duffing vanderpol")
    parser.add_argument("--train-seeds", nargs="+", default=("0", "1", "2"))
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
    parser.add_argument("--output-dir", type=str, default="kahkm_experiment_15_outputs")
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--max-train-per-cluster", type=int, default=None)
    parser.add_argument("--vanderpol-mu", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-8)
    ns = parser.parse_args()

    systems = _parse_system_tuple([str(v) for v in ns.systems])
    train_seeds = _parse_int_tuple([str(v) for v in ns.train_seeds])
    test_seeds = _parse_int_tuple([str(v) for v in ns.test_seeds])
    horizons = _parse_int_tuple([str(v) for v in ns.horizons])

    if int(ns.n_steps_train) <= 0 or int(ns.n_steps_test) <= 0:
        raise ValueError("n_steps_train and n_steps_test must be positive.")
    if max(horizons) > int(ns.n_steps_test):
        raise ValueError("The largest horizon cannot exceed n_steps_test.")
    if float(ns.ridge) < 0.0:
        raise ValueError("ridge must be nonnegative.")

    return ExperimentArgs(
        systems=systems,
        train_seeds=train_seeds,
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
        ridge=float(ns.ridge),
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
        print(
            f"\n[{system}] train_seeds={args.train_seeds}, test_seeds={args.test_seeds}, "
            f"C={cfg.n_clusters}, omega={cfg.omega}"
        )
        X0_train, X1_train = make_concatenated_training_data(
            system,
            train_seeds=args.train_seeds,
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
            random_state=int(args.train_seeds[0]),
            kmeans_kind=args.kmeans_kind,  # type: ignore[arg-type]
            max_train_per_cluster=args.max_train_per_cluster,
            save_ae_to_disk=args.save_ae_to_disk,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            project_stochastic=False,
            verbose=False,
        )
        kahkm_fit_seconds = time.time() - fit_start
        abstraction_model = getattr(fit, "abstraction_model")
        B_kahkm = np.asarray(getattr(fit, "B"), dtype=np.float64)

        baseline_start = time.time()
        A_dmd = _fit_linear_map(X0_train, X1_train, ridge=args.ridge)
        Z0_poly2 = _poly_features_2d(X0_train, degree=2)
        Z1_poly2 = _poly_features_2d(X1_train, degree=2)
        Z0_poly3 = _poly_features_2d(X0_train, degree=3)
        Z1_poly3 = _poly_features_2d(X1_train, degree=3)
        K_poly2 = _fit_lifted_operator(Z0_poly2, Z1_poly2, ridge=args.ridge)
        K_poly3 = _fit_lifted_operator(Z0_poly3, Z1_poly3, ridge=args.ridge)
        baseline_fit_seconds = time.time() - baseline_start

        fit_rows.append(
            {
                "system": system,
                "train_seeds_used": " ".join(str(seed) for seed in args.train_seeds),
                "n_train_snapshots": int(X0_train.shape[1]),
                "n_clusters": int(cfg.n_clusters),
                "omega": float(cfg.omega),
                "tau": float(cfg.tau),
                "kahkm_train_closure_error": float(getattr(fit, "train_closure_error")),
                "kahkm_train_association_r2": float(getattr(fit, "association_r2")),
                "state_dmd_one_step_train_error": _state_relative_error(A_dmd @ X0_train, X1_train),
                "state_edmd_poly2_one_step_train_error": _state_relative_error((_fit_lifted_operator(Z0_poly2, Z1_poly2, ridge=args.ridge) @ Z0_poly2)[1:3, :], X1_train),
                "state_edmd_poly3_one_step_train_error": _state_relative_error((_fit_lifted_operator(Z0_poly3, Z1_poly3, ridge=args.ridge) @ Z0_poly3)[1:3, :], X1_train),
                "kahkm_fit_seconds": float(kahkm_fit_seconds),
                "baseline_fit_seconds": float(baseline_fit_seconds),
            }
        )
        print(
            f"  KAHKM train_err={float(getattr(fit, 'train_closure_error')):.6g}, "
            f"KAHKM train_R2={float(getattr(fit, 'association_r2')):.6g}"
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
                train_seeds_used=args.train_seeds,
                test_seed=int(test_seed),
                abstraction_model=abstraction_model,
                omega=cfg.omega,
                tau=cfg.tau,
                B_kahkm=B_kahkm,
                A_dmd=A_dmd,
                K_poly2=K_poly2,
                K_poly3=K_poly3,
                X0_test=X0_test,
                X1_test=X1_test,
                horizons=args.horizons,
                n_jobs=args.n_jobs,
                batch_size=args.batch_size,
            )
            test_rows.extend(rows)
            hmax = max(args.horizons)
            key_rows = [row for row in rows if int(row["horizon"]) == int(hmax)]
            msg_parts = []
            for row in key_rows:
                msg_parts.append(f"{row['method']}={float(row['relative_association_error']):.4g}")
            print(f"  test_seed={test_seed}, h{hmax}: " + ", ".join(msg_parts))

    fit_fields = [
        "system",
        "train_seeds_used",
        "n_train_snapshots",
        "n_clusters",
        "omega",
        "tau",
        "kahkm_train_closure_error",
        "kahkm_train_association_r2",
        "state_dmd_one_step_train_error",
        "state_edmd_poly2_one_step_train_error",
        "state_edmd_poly3_one_step_train_error",
        "kahkm_fit_seconds",
        "baseline_fit_seconds",
    ]
    test_fields = [
        "system",
        "method",
        "train_seeds_used",
        "test_seed",
        "horizon",
        "n_eval_starts",
        "relative_association_error",
        "association_r2",
        "simplex_violation",
        "state_relative_error",
    ]
    summary_fields = [
        "system",
        "method",
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
        "state_relative_error_mean",
        "state_relative_error_std",
    ]

    summary_rows = _summarize_test_rows(test_rows)
    _write_csv(output_dir / "experiment_15_fit_summary.csv", fit_rows, fit_fields)
    _write_csv(output_dir / "experiment_15_test_raw_results.csv", test_rows, test_fields)
    _write_csv(output_dir / "experiment_15_test_summary.csv", summary_rows, summary_fields)

    metadata = asdict(args)
    metadata["system_configs"] = {name: asdict(cfg) for name, cfg in SYSTEM_CONFIGS.items()}
    metadata["purpose"] = (
        "Compare KAHKM association rollout with state-space DMD and polynomial EDMD baselines, "
        "all evaluated in KAHKM association space on unseen trajectories."
    )
    metadata["total_seconds"] = float(time.time() - start_all)
    with (output_dir / "experiment_15_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone. Wrote outputs to:")
    print(f"  {output_dir / 'experiment_15_fit_summary.csv'}")
    print(f"  {output_dir / 'experiment_15_test_raw_results.csv'}")
    print(f"  {output_dir / 'experiment_15_test_summary.csv'}")
    print(f"  {output_dir / 'experiment_15_metadata.json'}")


if __name__ == "__main__":
    main()
