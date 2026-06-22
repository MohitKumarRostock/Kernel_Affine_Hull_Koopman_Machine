"""Experiment 05: KAHKM closure-estimator comparison on Duffing.

Purpose
-------
This experiment keeps the KAHM/KAHKM abstraction fixed and compares several
finite Koopman closure estimators on the same learned association coordinates:

1. manuscript NLMS closure from fit_kahkm(...)
2. ordinary least-squares closure
3. ridge-regularized least-squares closures

This isolates the closure-estimation question from the abstraction question.
It does NOT compare KAHKM to other dictionaries; it compares estimators for the
same KAHKM dictionary.

Expected local files
--------------------
Place this script in the same folder as:

- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Run
---
python experiment_05_duffing_closure_estimators.py

Fast smoke test:
python experiment_05_duffing_closure_estimators.py --random-states 0 --n-steps 600 --horizons 1 2 5 10 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

# Ensure imports resolve when running from the script's folder.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
    fit_kahkm,
    kahm_associations,
    make_duffing_snapshots,
    project_rows_to_simplex,
    simplex_violation,
)

FloatArray: TypeAlias = NDArray[np.float64]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]


@dataclass(frozen=True)
class ExperimentArgs:
    n_steps: int
    dt: float
    train_fraction: float
    random_states: tuple[int, ...]
    n_clusters: int
    subspace_dim: int
    nb: int
    omega: float
    tau: float
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    batch_size: int
    n_jobs: int
    horizons: tuple[int, ...]
    ridge_lambdas: tuple[float, ...]
    output_dir: str
    save_ae_to_disk: bool
    max_train_per_cluster: int | None


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


def _spectral_radius(B: FloatArray) -> float:
    eigvals = np.linalg.eigvals(B)
    return float(np.max(np.abs(eigvals)))


def _ols_closure(Phi: FloatArray, Chi: FloatArray) -> FloatArray:
    """Return B for Chi ~= B.T @ Phi via ordinary least squares."""
    ZX = Phi.T
    ZY = Chi.T
    B, _, _, _ = np.linalg.lstsq(ZX, ZY, rcond=None)
    return np.asarray(B, dtype=np.float64)


def _ridge_closure(Phi: FloatArray, Chi: FloatArray, lam: float) -> FloatArray:
    """Return B for Chi ~= B.T @ Phi via ridge regression."""
    if lam < 0.0 or not math.isfinite(lam):
        raise ValueError("Ridge lambda must be finite and nonnegative.")
    ZX = Phi.T
    ZY = Chi.T
    c_count = int(Phi.shape[0])
    gram = ZX.T @ ZX
    rhs = ZX.T @ ZY
    B = np.linalg.solve(gram + float(lam) * np.eye(c_count, dtype=np.float64), rhs)
    return np.asarray(B, dtype=np.float64)


def _one_step_row(estimator: str, B: FloatArray, Phi: FloatArray, Chi: FloatArray, split_name: str) -> CsvRow:
    pred = B.T @ Phi
    return {
        "estimator": estimator,
        "split": split_name,
        "closure_error": _relative_error(pred, Chi),
        "association_r2": _association_r2(pred, Chi),
        "simplex_violation": simplex_violation(pred),
        "spectral_radius": _spectral_radius(B),
        "fro_norm": float(np.linalg.norm(B, ord="fro")),
        "min_entry": float(np.min(B)),
        "max_entry": float(np.max(B)),
        "row_sum_min": float(np.min(np.sum(B, axis=1))),
        "row_sum_max": float(np.max(np.sum(B, axis=1))),
    }


def _multistep_row(
    estimator: str,
    B: FloatArray,
    Psi_states: FloatArray,
    *,
    start_min: int,
    start_max_exclusive: int,
    horizon: int,
) -> CsvRow:
    """Evaluate Psi_{t+h} ~= (B.T)^h Psi_t over test starting indices."""
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be positive.")

    max_start_allowed = min(int(start_max_exclusive), int(Psi_states.shape[1]) - h)
    if max_start_allowed <= int(start_min):
        return {
            "estimator": estimator,
            "horizon": h,
            "n_starts": 0,
            "relative_association_error": float("nan"),
            "association_r2": float("nan"),
            "simplex_violation": float("nan"),
        }

    starts = np.arange(int(start_min), max_start_allowed, dtype=np.int64)
    current = Psi_states[:, starts]
    target = Psi_states[:, starts + h]
    Bh = np.linalg.matrix_power(B.T, h)
    pred = Bh @ current
    return {
        "estimator": estimator,
        "horizon": h,
        "n_starts": int(starts.size),
        "relative_association_error": _relative_error(pred, target),
        "association_r2": _association_r2(pred, target),
        "simplex_violation": simplex_violation(pred),
    }


def _write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_multistep(rows: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[tuple[str, int], list[CsvRow]] = {}
    for row in rows:
        estimator = str(row["estimator"])
        horizon = int(row["horizon"])
        groups.setdefault((estimator, horizon), []).append(row)

    summary: list[CsvRow] = []
    for (estimator, horizon), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        err = np.array([float(r["relative_association_error"]) for r in group_rows], dtype=np.float64)
        r2 = np.array([float(r["association_r2"]) for r in group_rows], dtype=np.float64)
        violation = np.array([float(r["simplex_violation"]) for r in group_rows], dtype=np.float64)
        summary.append(
            {
                "estimator": estimator,
                "horizon": horizon,
                "n_runs": int(len(group_rows)),
                "relative_association_error_mean": float(np.nanmean(err)),
                "relative_association_error_std": float(np.nanstd(err, ddof=1)) if len(group_rows) > 1 else 0.0,
                "relative_association_error_min": float(np.nanmin(err)),
                "relative_association_error_max": float(np.nanmax(err)),
                "association_r2_mean": float(np.nanmean(r2)),
                "association_r2_std": float(np.nanstd(r2, ddof=1)) if len(group_rows) > 1 else 0.0,
                "simplex_violation_mean": float(np.nanmean(violation)),
                "simplex_violation_max": float(np.nanmax(violation)),
            }
        )
    return summary


def _summarize_one_step(rows: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[tuple[str, str], list[CsvRow]] = {}
    for row in rows:
        estimator = str(row["estimator"])
        split = str(row["split"])
        groups.setdefault((estimator, split), []).append(row)

    summary: list[CsvRow] = []
    for (estimator, split), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        err = np.array([float(r["closure_error"]) for r in group_rows], dtype=np.float64)
        r2 = np.array([float(r["association_r2"]) for r in group_rows], dtype=np.float64)
        violation = np.array([float(r["simplex_violation"]) for r in group_rows], dtype=np.float64)
        rho = np.array([float(r["spectral_radius"]) for r in group_rows], dtype=np.float64)
        summary.append(
            {
                "estimator": estimator,
                "split": split,
                "n_runs": int(len(group_rows)),
                "closure_error_mean": float(np.nanmean(err)),
                "closure_error_std": float(np.nanstd(err, ddof=1)) if len(group_rows) > 1 else 0.0,
                "association_r2_mean": float(np.nanmean(r2)),
                "association_r2_std": float(np.nanstd(r2, ddof=1)) if len(group_rows) > 1 else 0.0,
                "simplex_violation_mean": float(np.nanmean(violation)),
                "simplex_violation_max": float(np.nanmax(violation)),
                "spectral_radius_mean": float(np.nanmean(rho)),
                "spectral_radius_max": float(np.nanmax(rho)),
            }
        )
    return summary


def run_experiment(args: ExperimentArgs) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    one_step_rows: list[CsvRow] = []
    multistep_rows: list[CsvRow] = []
    run_rows: list[CsvRow] = []

    start_all = time.time()
    for seed in args.random_states:
        print(f"\n=== Seed {seed} ===")
        seed_start = time.time()
        X0, X1 = make_duffing_snapshots(n_steps=args.n_steps, dt=args.dt, seed=seed)
        n_pairs = int(X0.shape[1])
        split = int(args.train_fraction * n_pairs)
        if split <= 1 or split >= n_pairs - 1:
            raise ValueError("train_fraction gives an invalid train/test split.")

        X0_train = X0[:, :split]
        X1_train = X1[:, :split]

        fit = fit_kahkm(
            X0_train,
            X1_train,
            n_clusters=args.n_clusters,
            subspace_dim=args.subspace_dim,
            Nb=args.nb,
            omega=args.omega,
            tau=args.tau,
            beta=args.beta,
            nlms_epochs=args.nlms_epochs,
            random_state=seed,
            kmeans_kind=args.kmeans_kind,  # type: ignore[arg-type]
            save_ae_to_disk=args.save_ae_to_disk,
            max_train_per_cluster=args.max_train_per_cluster,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            project_stochastic=True,
            verbose=True,
        )

        # Build full state trajectory x_0, ..., x_N from snapshot pairs and evaluate one KAHKM association path.
        states = np.concatenate([X0[:, [0]], X1], axis=1)
        Psi_states = kahm_associations(
            fit.abstraction_model,
            states,
            omega=fit.omega,
            tau=fit.tau,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            show_progress=False,
        )

        Phi_train = Psi_states[:, :split]
        Chi_train = Psi_states[:, 1 : split + 1]
        Phi_test = Psi_states[:, split:n_pairs]
        Chi_test = Psi_states[:, split + 1 : n_pairs + 1]

        estimators: dict[str, FloatArray] = {
            "nlms": np.asarray(fit.B, dtype=np.float64),
            "ols": _ols_closure(Phi_train, Chi_train),
            "row_stochastic_nlms": project_rows_to_simplex(np.asarray(fit.B, dtype=np.float64)),
        }
        for lam in args.ridge_lambdas:
            estimators[f"ridge_{lam:g}"] = _ridge_closure(Phi_train, Chi_train, lam)

        for estimator_name, B in estimators.items():
            train_row = _one_step_row(estimator_name, B, Phi_train, Chi_train, "train")
            test_row = _one_step_row(estimator_name, B, Phi_test, Chi_test, "test")
            train_row["seed"] = int(seed)
            test_row["seed"] = int(seed)
            one_step_rows.extend([train_row, test_row])

            for horizon in args.horizons:
                ms_row = _multistep_row(
                    estimator_name,
                    B,
                    Psi_states,
                    start_min=split,
                    start_max_exclusive=n_pairs,
                    horizon=horizon,
                )
                ms_row["seed"] = int(seed)
                multistep_rows.append(ms_row)

        run_rows.append(
            {
                "seed": int(seed),
                "n_pairs": n_pairs,
                "n_train": split,
                "n_test": n_pairs - split,
                "effective_regimes": int(fit.abstraction_model.get("n_clusters", args.n_clusters)),
                "nlms_final_train_error": float(fit.nlms_history[-1]) if fit.nlms_history else float("nan"),
                "fit_and_eval_seconds": float(time.time() - seed_start),
            }
        )
        print(f"Seed {seed} finished in {time.time() - seed_start:.2f} seconds.")

    one_step_summary = _summarize_one_step(one_step_rows)
    multistep_summary = _summarize_multistep(multistep_rows)

    _write_csv(output_dir / "experiment_05_one_step_raw.csv", one_step_rows)
    _write_csv(output_dir / "experiment_05_one_step_summary.csv", one_step_summary)
    _write_csv(output_dir / "experiment_05_multistep_raw.csv", multistep_rows)
    _write_csv(output_dir / "experiment_05_multistep_summary.csv", multistep_summary)
    _write_csv(output_dir / "experiment_05_run_diagnostics.csv", run_rows)

    metadata = asdict(args)
    metadata["total_seconds"] = float(time.time() - start_all)
    metadata["notes"] = (
        "Same KAHKM abstraction for all estimators within each seed. "
        "Estimator comparison isolates closure learning, not dictionary quality."
    )
    with (output_dir / "experiment_05_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nWrote outputs to:")
    for name in [
        "experiment_05_one_step_summary.csv",
        "experiment_05_multistep_summary.csv",
        "experiment_05_run_diagnostics.csv",
        "experiment_05_metadata.json",
    ]:
        print(f"  {output_dir / name}")

    # Compact terminal preview.
    print("\nOne-step test summary preview:")
    for row in one_step_summary:
        if row["split"] == "test":
            print(
                f"  {row['estimator']}: "
                f"err={float(row['closure_error_mean']):.6g}, "
                f"r2={float(row['association_r2_mean']):.6g}, "
                f"rho={float(row['spectral_radius_mean']):.6g}"
            )


def parse_args(argv: Sequence[str] | None = None) -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Experiment 05: KAHKM closure-estimator comparison on Duffing.")
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--random-states", nargs="+", default=["0", "1", "2"], help="Random seeds.")
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--omega", type=float, default=12.0)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", type=str, default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--horizons", nargs="+", default=["1", "2", "5", "10", "20", "50", "100", "200"])
    parser.add_argument("--ridge-lambdas", nargs="+", default=["1e-10", "1e-8", "1e-6", "1e-4", "1e-2"])
    parser.add_argument("--output-dir", type=str, default="kahkm_experiment_05_outputs")
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--max-train-per-cluster", type=int, default=None)
    ns = parser.parse_args(argv)

    return ExperimentArgs(
        n_steps=int(ns.n_steps),
        dt=float(ns.dt),
        train_fraction=float(ns.train_fraction),
        random_states=_parse_int_tuple(tuple(str(v) for v in ns.random_states)),
        n_clusters=int(ns.n_clusters),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        omega=float(ns.omega),
        tau=float(ns.tau),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        kmeans_kind=str(ns.kmeans_kind),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        horizons=_parse_int_tuple(tuple(str(v) for v in ns.horizons)),
        ridge_lambdas=_parse_float_tuple(tuple(str(v) for v in ns.ridge_lambdas)),
        output_dir=str(ns.output_dir),
        save_ae_to_disk=bool(ns.save_ae_to_disk),
        max_train_per_cluster=None if ns.max_train_per_cluster is None else int(ns.max_train_per_cluster),
    )


if __name__ == "__main__":
    run_experiment(parse_args())
