"""Experiment 12: observation-noise robustness for tuned KAHKM closure.

Purpose
-------
This experiment checks whether the tuned KAHKM abstraction and NLMS closure are
robust when the training snapshots are noisy. The clean trajectory is generated
first. Gaussian observation noise is then added only to the training snapshot
pairs (X0_train, X1_train). The learned KAHKM model is evaluated on the clean
held-out suffix of the trajectory.

System-specific hyperparameters
-------------------------------
- Duffing:      C=15, omega=12
- Van der Pol:  C=20, omega=4

Noise convention
----------------
For each coordinate, noise standard deviation is

    noise_level * std(training coordinate).

Thus noise_level=0.02 means 2% coordinate-wise observation noise relative to the
training trajectory scale.

Expected local files
--------------------
Place this script in the same folder as:

- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Run
---
python experiment_12_noise_robustness_pylance_clean.py

Fast smoke test:
python experiment_12_noise_robustness_pylance_clean.py --systems duffing --random-states 0 --noise-levels 0 0.01 --n-steps 700 --horizons 1 10 50
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
    n_steps: int
    train_fraction: float
    noise_levels: tuple[float, ...]
    random_states: tuple[int, ...]
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


def _parse_float_tuple(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(float(v) for v in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one float value is required.")
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


def make_vanderpol_snapshots(n_steps: int = 1600, dt: float = 0.02, seed: int = 0, mu: float = 1.0) -> tuple[FloatArray, FloatArray]:
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


def make_snapshots(system: SystemName, *, n_steps: int, dt: float, seed: int, vanderpol_mu: float) -> tuple[FloatArray, FloatArray]:
    if system == "duffing":
        X0, X1 = make_duffing_snapshots(n_steps=int(n_steps), dt=float(dt), seed=int(seed))
        return np.asarray(X0, dtype=np.float64), np.asarray(X1, dtype=np.float64)
    if system == "vanderpol":
        return make_vanderpol_snapshots(n_steps=int(n_steps), dt=float(dt), seed=int(seed), mu=float(vanderpol_mu))
    raise ValueError(f"Unknown system: {system}")


def add_coordinate_noise(X: FloatArray, *, noise_level: float, rng: np.random.Generator) -> FloatArray:
    """Add coordinate-wise Gaussian noise scaled by the coordinate std of X."""
    X_arr = np.asarray(X, dtype=np.float64)
    level = float(noise_level)
    if level <= 0.0:
        return np.array(X_arr, dtype=np.float64, copy=True)
    scale = np.std(X_arr, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    noise = rng.normal(loc=0.0, scale=level * scale, size=X_arr.shape)
    return np.asarray(X_arr + noise, dtype=np.float64)


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


def _multistep_rows(
    *,
    system: SystemName,
    seed: int,
    noise_level: float,
    fit_result: object,
    X0_test_clean: FloatArray,
    X1_test_clean: FloatArray,
    horizons: tuple[int, ...],
    n_jobs: int,
    batch_size: int,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    abstraction_model = getattr(fit_result, "abstraction_model")
    B = np.asarray(getattr(fit_result, "B"), dtype=np.float64)
    omega = float(getattr(fit_result, "omega"))
    tau = float(getattr(fit_result, "tau"))

    n_test = int(X0_test_clean.shape[1])
    for horizon in horizons:
        h = int(horizon)
        if h <= 0 or h > n_test:
            continue
        n_start = n_test - h + 1
        X_start = X0_test_clean[:, :n_start]
        X_target = X1_test_clean[:, h - 1 : h - 1 + n_start]
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
                "seed": int(seed),
                "noise_level": float(noise_level),
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


def _summarize_one_step(raw_rows: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[tuple[str, float], list[CsvRow]] = {}
    for row in raw_rows:
        key = (str(row["system"]), float(row["noise_level"]))
        groups.setdefault(key, []).append(row)

    out: list[CsvRow] = []
    for (system, noise_level), rows in sorted(groups.items()):
        train_errors = [float(row["train_closure_error_noisy_train"]) for row in rows]
        clean_test_errors = [float(row["clean_test_closure_error"]) for row in rows]
        clean_test_r2 = [float(row["clean_test_association_r2"]) for row in rows]
        clean_simplex = [float(row["clean_test_simplex_violation"]) for row in rows]
        out.append(
            {
                "system": system,
                "noise_level": noise_level,
                "n_runs": len(rows),
                "train_closure_error_noisy_train_mean": _mean(train_errors),
                "train_closure_error_noisy_train_std": _std(train_errors),
                "clean_test_closure_error_mean": _mean(clean_test_errors),
                "clean_test_closure_error_std": _std(clean_test_errors),
                "clean_test_association_r2_mean": _mean(clean_test_r2),
                "clean_test_association_r2_std": _std(clean_test_r2),
                "clean_test_simplex_violation_mean": _mean(clean_simplex),
                "clean_test_simplex_violation_max": float(max(clean_simplex)) if clean_simplex else float("nan"),
            }
        )
    return out


def _summarize_multistep(rows_in: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[tuple[str, float, int], list[CsvRow]] = {}
    for row in rows_in:
        key = (str(row["system"]), float(row["noise_level"]), int(row["horizon"]))
        groups.setdefault(key, []).append(row)

    out: list[CsvRow] = []
    for (system, noise_level, horizon), rows in sorted(groups.items()):
        errors = [float(row["relative_association_error"]) for row in rows]
        r2s = [float(row["association_r2"]) for row in rows]
        violations = [float(row["simplex_violation"]) for row in rows]
        out.append(
            {
                "system": system,
                "noise_level": noise_level,
                "horizon": horizon,
                "n_runs": len(rows),
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
    parser = argparse.ArgumentParser(description="Experiment 12: observation-noise robustness for tuned KAHKM.")
    parser.add_argument("--systems", nargs="+", default=("duffing", "vanderpol"), help="Systems: duffing vanderpol")
    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--noise-levels", nargs="+", default=("0", "0.005", "0.01", "0.02", "0.05"))
    parser.add_argument("--random-states", nargs="+", default=("0", "1", "2"))
    parser.add_argument("--horizons", nargs="+", default=("1", "10", "50", "100", "200"))
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", type=str, default="full")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="kahkm_experiment_12_outputs")
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--max-train-per-cluster", type=int, default=None)
    parser.add_argument("--vanderpol-mu", type=float, default=1.0)
    ns = parser.parse_args()

    systems = _parse_system_tuple([str(v) for v in ns.systems])
    noise_levels = _parse_float_tuple([str(v) for v in ns.noise_levels])
    random_states = _parse_int_tuple([str(v) for v in ns.random_states])
    horizons = _parse_int_tuple([str(v) for v in ns.horizons])
    train_fraction = float(ns.train_fraction)
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    for noise_level in noise_levels:
        if noise_level < 0.0:
            raise ValueError("noise levels must be nonnegative.")

    return ExperimentArgs(
        systems=systems,
        n_steps=int(ns.n_steps),
        train_fraction=train_fraction,
        noise_levels=noise_levels,
        random_states=random_states,
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

    raw_rows: list[CsvRow] = []
    multistep_rows: list[CsvRow] = []
    start_all = time.time()

    for system in args.systems:
        cfg = SYSTEM_CONFIGS[system]
        for seed in args.random_states:
            X0, X1 = make_snapshots(
                system,
                n_steps=args.n_steps,
                dt=cfg.dt,
                seed=int(seed),
                vanderpol_mu=args.vanderpol_mu,
            )
            n_total = int(X0.shape[1])
            split = int(round(float(args.train_fraction) * n_total))
            max_horizon = max(args.horizons)
            if n_total - split < max_horizon:
                raise ValueError(
                    f"Not enough test samples for system={system}, train_fraction={args.train_fraction}, "
                    f"n_steps={n_total}, max_horizon={max_horizon}."
                )
            X0_train_clean = X0[:, :split]
            X1_train_clean = X1[:, :split]
            X0_test_clean = X0[:, split:]
            X1_test_clean = X1[:, split:]

            for noise_level in args.noise_levels:
                noise_seed = 1_000_003 * int(seed) + int(round(1_000_000.0 * float(noise_level))) + 17
                rng = np.random.default_rng(noise_seed)
                X0_train = add_coordinate_noise(X0_train_clean, noise_level=float(noise_level), rng=rng)
                X1_train = add_coordinate_noise(X1_train_clean, noise_level=float(noise_level), rng=rng)

                print(
                    f"\n[{system}] seed={seed}, noise={noise_level}, "
                    f"n_train={X0_train.shape[1]}, n_test={X0_test_clean.shape[1]}, "
                    f"C={cfg.n_clusters}, omega={cfg.omega}"
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
                    random_state=int(seed),
                    kmeans_kind=args.kmeans_kind,  # type: ignore[arg-type]
                    max_train_per_cluster=args.max_train_per_cluster,
                    save_ae_to_disk=args.save_ae_to_disk,
                    n_jobs=args.n_jobs,
                    batch_size=args.batch_size,
                    project_stochastic=False,
                    verbose=False,
                )
                fit_seconds = time.time() - fit_start

                ms_rows = _multistep_rows(
                    system=system,
                    seed=int(seed),
                    noise_level=float(noise_level),
                    fit_result=fit,
                    X0_test_clean=X0_test_clean,
                    X1_test_clean=X1_test_clean,
                    horizons=args.horizons,
                    n_jobs=args.n_jobs,
                    batch_size=args.batch_size,
                )
                multistep_rows.extend(ms_rows)
                one_step = next(row for row in ms_rows if int(row["horizon"]) == 1)
                raw_rows.append(
                    {
                        "system": system,
                        "seed": int(seed),
                        "noise_level": float(noise_level),
                        "n_steps": int(args.n_steps),
                        "train_fraction": float(args.train_fraction),
                        "n_train": int(X0_train.shape[1]),
                        "n_test": int(X0_test_clean.shape[1]),
                        "n_clusters": int(cfg.n_clusters),
                        "omega": float(cfg.omega),
                        "tau": float(cfg.tau),
                        "train_closure_error_noisy_train": float(getattr(fit, "train_closure_error")),
                        "train_association_r2_noisy_train": float(getattr(fit, "association_r2")),
                        "clean_test_closure_error": float(one_step["relative_association_error"]),
                        "clean_test_association_r2": float(one_step["association_r2"]),
                        "clean_test_simplex_violation": float(one_step["simplex_violation"]),
                        "fit_seconds": float(fit_seconds),
                    }
                )
                print(
                    f"  noisy_train_err={float(getattr(fit, 'train_closure_error')):.6g}, "
                    f"clean_test_err={float(one_step['relative_association_error']):.6g}, "
                    f"clean_test_R2={float(one_step['association_r2']):.6g}, "
                    f"fit_seconds={fit_seconds:.2f}"
                )

    raw_fields = [
        "system",
        "seed",
        "noise_level",
        "n_steps",
        "train_fraction",
        "n_train",
        "n_test",
        "n_clusters",
        "omega",
        "tau",
        "train_closure_error_noisy_train",
        "train_association_r2_noisy_train",
        "clean_test_closure_error",
        "clean_test_association_r2",
        "clean_test_simplex_violation",
        "fit_seconds",
    ]
    multistep_fields = [
        "system",
        "seed",
        "noise_level",
        "horizon",
        "n_eval_starts",
        "relative_association_error",
        "association_r2",
        "simplex_violation",
    ]
    one_step_summary = _summarize_one_step(raw_rows)
    multistep_summary = _summarize_multistep(multistep_rows)

    _write_csv(output_dir / "experiment_12_raw_results.csv", raw_rows, raw_fields)
    _write_csv(output_dir / "experiment_12_multistep_raw_results.csv", multistep_rows, multistep_fields)
    _write_csv(
        output_dir / "experiment_12_one_step_summary.csv",
        one_step_summary,
        [
            "system",
            "noise_level",
            "n_runs",
            "train_closure_error_noisy_train_mean",
            "train_closure_error_noisy_train_std",
            "clean_test_closure_error_mean",
            "clean_test_closure_error_std",
            "clean_test_association_r2_mean",
            "clean_test_association_r2_std",
            "clean_test_simplex_violation_mean",
            "clean_test_simplex_violation_max",
        ],
    )
    _write_csv(
        output_dir / "experiment_12_multistep_summary.csv",
        multistep_summary,
        [
            "system",
            "noise_level",
            "horizon",
            "n_runs",
            "relative_association_error_mean",
            "relative_association_error_std",
            "relative_association_error_min",
            "relative_association_error_max",
            "association_r2_mean",
            "association_r2_std",
            "simplex_violation_mean",
            "simplex_violation_max",
        ],
    )
    metadata = asdict(args)
    metadata["system_configs"] = {name: asdict(cfg) for name, cfg in SYSTEM_CONFIGS.items()}
    metadata["noise_convention"] = "training-only Gaussian observation noise; sigma_j = noise_level * std_j(training coordinate); evaluated on clean held-out trajectory"
    metadata["total_seconds"] = float(time.time() - start_all)
    with (output_dir / "experiment_12_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone. Wrote outputs to:")
    print(f"  {output_dir / 'experiment_12_raw_results.csv'}")
    print(f"  {output_dir / 'experiment_12_one_step_summary.csv'}")
    print(f"  {output_dir / 'experiment_12_multistep_summary.csv'}")


if __name__ == "__main__":
    main()
