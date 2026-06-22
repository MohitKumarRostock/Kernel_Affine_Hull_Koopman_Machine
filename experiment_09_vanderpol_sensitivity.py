#!/usr/bin/env python3
"""Experiment 09: Van der Pol KAHKM hyperparameter sensitivity.

Purpose
-------
Experiment 07/08 showed that KAHKM is the strongest tested abstraction on
Van der Pol, but long-horizon rollouts drift substantially when using the
Duffing-selected defaults C=15, omega=12. This experiment tunes the KAHKM
abstraction itself for Van der Pol by sweeping:

- n_clusters: abstraction granularity C
- omega: association sharpness

For each configuration and seed, the script reports:

1. one-step held-out closure error and association R^2;
2. raw-matrix multi-step association errors at requested horizons;
3. simplex violation of raw rollouts;
4. basic spectral and matrix diagnostics.

Expected local files
--------------------
Place this script in the same folder as:
  - kernel_affine_hull_koopman_machines.py
  - parallel_autoencoders.py
  - combine_multiple_autoencoders_extended.py

Run
---
python experiment_09_vanderpol_sensitivity_pylance_clean.py

Fast smoke test
---------------
python experiment_09_vanderpol_sensitivity_pylance_clean.py --clusters 10 15 --omegas 8 12 --random-states 0 --n-steps 600 --horizons 1 2 5 10 20
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
from typing import Literal, Mapping, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
    evaluate_kahkm,
    fit_kahkm,
    kahm_associations,
    simplex_violation,
)

FloatArray: TypeAlias = NDArray[np.float64]
KMeansKind: TypeAlias = Literal["auto", "full", "minibatch"]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]


@dataclass(frozen=True)
class ExperimentArgs:
    output_dir: Path
    n_steps: int
    dt: float
    mu: float
    train_fraction: float
    clusters: list[int]
    omegas: list[float]
    random_states: list[int]
    tau: float
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    kmeans_kind: KMeansKind
    batch_size: int
    n_jobs: int
    horizons: list[int]
    save_ae_to_disk: bool
    max_train_per_cluster: int | None
    verbose_fit: bool


def make_vanderpol_snapshots(n_steps: int = 1600, dt: float = 0.02, seed: int = 0, mu: float = 1.0) -> tuple[FloatArray, FloatArray]:
    """Generate snapshot pairs from the Van der Pol oscillator.

    State z = [q, p]. Continuous-time dynamics:
        q' = p,
        p' = mu * (1 - q^2) * p - q.
    Integration uses RK4 plus tiny process noise to avoid duplicate samples.
    """
    rng = np.random.default_rng(int(seed))

    def f(z: FloatArray) -> FloatArray:
        q = float(z[0])
        p = float(z[1])
        return np.array([p, float(mu) * (1.0 - q * q) * p - q], dtype=np.float64)

    def rk4_step(z: FloatArray) -> FloatArray:
        dt_f = float(dt)
        k1 = f(z)
        k2 = f(np.asarray(z + 0.5 * dt_f * k1, dtype=np.float64))
        k3 = f(np.asarray(z + 0.5 * dt_f * k2, dtype=np.float64))
        k4 = f(np.asarray(z + dt_f * k3, dtype=np.float64))
        return np.asarray(z + (dt_f / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)

    z = np.array([2.0, 0.0], dtype=np.float64) + 0.05 * rng.normal(size=2)
    states = np.empty((2, int(n_steps) + 1), dtype=np.float64)
    states[:, 0] = z
    for t in range(int(n_steps)):
        z = rk4_step(z) + 1e-5 * rng.normal(size=2)
        states[:, t + 1] = z
    return states[:, :-1], states[:, 1:]


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return out


def _positive_float(value: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return out


def _object_to_int_list(value: object, *, name: str) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parsed = [int(item) for item in value]
        if not parsed:
            raise ValueError(f"{name} must contain at least one value.")
        return parsed
    raise TypeError(f"{name} must be a sequence of integers.")


def _object_to_float_list(value: object, *, name: str) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parsed = [float(item) for item in value]
        if not parsed:
            raise ValueError(f"{name} must contain at least one value.")
        return parsed
    raise TypeError(f"{name} must be a sequence of floats.")


def _parse_kmeans_kind(value: object) -> KMeansKind:
    raw = str(value)
    if raw not in {"auto", "full", "minibatch"}:
        raise ValueError("--kmeans-kind must be one of: auto, full, minibatch")
    return cast(KMeansKind, raw)


def _relative_sq_error(pred: FloatArray, target: FloatArray, eps: float = 1e-12) -> float:
    num = float(np.sum((target - pred) ** 2))
    den = float(np.sum(target * target))
    return num / max(den, eps)


def _association_r2(pred: FloatArray, target: FloatArray) -> float:
    residual_ss = float(np.sum((target - pred) ** 2))
    total_ss = float(np.sum((target - target.mean(axis=1, keepdims=True)) ** 2))
    if total_ss <= 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def _safe_spectral_radius(matrix: FloatArray) -> float:
    vals = np.linalg.eigvals(np.asarray(matrix, dtype=np.float64))
    return float(np.max(np.abs(vals)))


def _rollout_rows(
    *,
    B: FloatArray,
    psi_sequence: FloatArray,
    horizons: Sequence[int],
) -> list[CsvRow]:
    """Roll out raw B in association space and compute horizon diagnostics."""
    psi = np.asarray(psi_sequence, dtype=np.float64)
    matrix = np.asarray(B, dtype=np.float64)
    c_count, seq_len = psi.shape
    if matrix.shape != (c_count, c_count):
        raise ValueError(f"B has shape {matrix.shape}; expected {(c_count, c_count)}.")
    if not horizons:
        raise ValueError("At least one horizon is required.")
    max_h = int(max(horizons))
    if max_h >= seq_len:
        raise ValueError(f"Maximum horizon {max_h} must be smaller than sequence length {seq_len}.")

    rows: list[CsvRow] = []
    for horizon_obj in horizons:
        horizon = int(horizon_obj)
        pred = psi[:, : seq_len - horizon].copy()
        for _ in range(horizon):
            pred = np.asarray(matrix.T @ pred, dtype=np.float64)
        target = psi[:, horizon:]
        rows.append(
            {
                "horizon": horizon,
                "n_start_points": int(target.shape[1]),
                "relative_association_error": _relative_sq_error(pred, target),
                "association_r2": _association_r2(pred, target),
                "simplex_violation": float(simplex_violation(pred)),
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return float(math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)))


def write_csv(path: Path, rows: Sequence[Mapping[str, CsvValue]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Experiment 09: Van der Pol KAHKM sensitivity over C and omega.")
    parser.add_argument("--output-dir", type=Path, default=Path("kahkm_experiment_09_outputs"))
    parser.add_argument("--n-steps", type=_positive_int, default=1600)
    parser.add_argument("--dt", type=_positive_float, default=0.02)
    parser.add_argument("--mu", type=_positive_float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--clusters", nargs="+", type=_positive_int, default=[10, 15, 20, 25])
    parser.add_argument("--omegas", nargs="+", type=_positive_float, default=[4.0, 8.0, 12.0, 16.0])
    parser.add_argument("--random-states", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tau", type=_positive_float, default=1e-6)
    parser.add_argument("--subspace-dim", type=_positive_int, default=4)
    parser.add_argument("--nb", type=_positive_int, default=100)
    parser.add_argument("--beta", type=_positive_float, default=0.1)
    parser.add_argument("--nlms-epochs", type=_positive_int, default=20)
    parser.add_argument("--kmeans-kind", choices=["auto", "full", "minibatch"], default="full")
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--horizons", nargs="+", type=_positive_int, default=[1, 2, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--max-train-per-cluster", type=_positive_int, default=None)
    parser.add_argument("--verbose-fit", action="store_true")

    namespace = parser.parse_args()
    if not (0.0 < float(namespace.train_fraction) < 1.0):
        raise ValueError("--train-fraction must lie in (0, 1).")
    return ExperimentArgs(
        output_dir=Path(namespace.output_dir),
        n_steps=int(namespace.n_steps),
        dt=float(namespace.dt),
        mu=float(namespace.mu),
        train_fraction=float(namespace.train_fraction),
        clusters=_object_to_int_list(namespace.clusters, name="--clusters"),
        omegas=_object_to_float_list(namespace.omegas, name="--omegas"),
        random_states=_object_to_int_list(namespace.random_states, name="--random-states"),
        tau=float(namespace.tau),
        subspace_dim=int(namespace.subspace_dim),
        nb=int(namespace.nb),
        beta=float(namespace.beta),
        nlms_epochs=int(namespace.nlms_epochs),
        kmeans_kind=_parse_kmeans_kind(namespace.kmeans_kind),
        batch_size=int(namespace.batch_size),
        n_jobs=int(namespace.n_jobs),
        horizons=_object_to_int_list(namespace.horizons, name="--horizons"),
        save_ae_to_disk=bool(namespace.save_ae_to_disk),
        max_train_per_cluster=None if namespace.max_train_per_cluster is None else int(namespace.max_train_per_cluster),
        verbose_fit=bool(namespace.verbose_fit),
    )


def _metadata_args(args: ExperimentArgs) -> dict[str, object]:
    data: dict[str, object] = dict(asdict(args))
    data["output_dir"] = str(args.output_dir)
    return data


def _summarize_one_step(rows: Sequence[CsvRow]) -> list[CsvRow]:
    ok_rows = [row for row in rows if str(row["status"]) == "ok"]
    keys = sorted({(int(row["n_clusters"]), float(row["omega"])) for row in ok_rows})
    summary: list[CsvRow] = []
    for n_clusters, omega in keys:
        group = [row for row in ok_rows if int(row["n_clusters"]) == n_clusters and float(row["omega"]) == omega]
        test_errs = [float(row["test_closure_error"]) for row in group]
        test_r2s = [float(row["test_association_r2"]) for row in group]
        train_errs = [float(row["train_closure_error"]) for row in group]
        viols = [float(row["raw_predicted_simplex_violation"] ) for row in group]
        summary.append(
            {
                "n_clusters": n_clusters,
                "omega": omega,
                "n_runs": len(group),
                "train_closure_error_mean": _mean(train_errs),
                "train_closure_error_std": _std(train_errs),
                "test_closure_error_mean": _mean(test_errs),
                "test_closure_error_std": _std(test_errs),
                "test_closure_error_min": min(test_errs),
                "test_closure_error_max": max(test_errs),
                "test_association_r2_mean": _mean(test_r2s),
                "test_association_r2_std": _std(test_r2s),
                "test_association_r2_min": min(test_r2s),
                "raw_predicted_simplex_violation_mean": _mean(viols),
                "raw_predicted_simplex_violation_max": max(viols),
            }
        )
    return sorted(summary, key=lambda row: float(row["test_closure_error_mean"]))


def _summarize_multistep(rows: Sequence[CsvRow]) -> list[CsvRow]:
    keys = sorted({(int(row["n_clusters"]), float(row["omega"]), int(row["horizon"])) for row in rows})
    summary: list[CsvRow] = []
    for n_clusters, omega, horizon in keys:
        group = [
            row
            for row in rows
            if int(row["n_clusters"]) == n_clusters and float(row["omega"]) == omega and int(row["horizon"]) == horizon
        ]
        errs = [float(row["relative_association_error"]) for row in group]
        r2s = [float(row["association_r2"]) for row in group]
        viols = [float(row["simplex_violation"]) for row in group]
        summary.append(
            {
                "n_clusters": n_clusters,
                "omega": omega,
                "horizon": horizon,
                "n_runs": len(group),
                "relative_association_error_mean": _mean(errs),
                "relative_association_error_std": _std(errs),
                "relative_association_error_min": min(errs),
                "relative_association_error_max": max(errs),
                "association_r2_mean": _mean(r2s),
                "association_r2_std": _std(r2s),
                "simplex_violation_mean": _mean(viols),
                "simplex_violation_max": max(viols),
            }
        )
    return sorted(summary, key=lambda row: (int(row["horizon"]), float(row["relative_association_error_mean"])))


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    one_step_rows: list[CsvRow] = []
    multistep_rows: list[CsvRow] = []
    failure_rows: list[CsvRow] = []

    total = len(args.clusters) * len(args.omegas) * len(args.random_states)
    job_i = 0
    for n_clusters in args.clusters:
        for omega in args.omegas:
            for seed in args.random_states:
                job_i += 1
                print(f"\n=== Experiment 09 job {job_i}/{total}: C={n_clusters}, omega={omega:g}, seed={seed} ===")
                job_start = time.time()
                try:
                    X0, X1 = make_vanderpol_snapshots(n_steps=args.n_steps, dt=args.dt, seed=seed, mu=args.mu)
                    split = int(args.train_fraction * X0.shape[1])
                    if split <= 1 or split >= X0.shape[1] - 1:
                        raise ValueError("--train-fraction gives an invalid train/test split.")
                    X0_train, X1_train = X0[:, :split], X1[:, :split]
                    X0_test, X1_test = X0[:, split:], X1[:, split:]
                    test_state_sequence = np.concatenate([X0_test, X1_test[:, -1:]], axis=1)

                    fit_start = time.time()
                    fit = fit_kahkm(
                        X0_train,
                        X1_train,
                        n_clusters=int(n_clusters),
                        subspace_dim=args.subspace_dim,
                        Nb=args.nb,
                        omega=float(omega),
                        tau=args.tau,
                        beta=args.beta,
                        nlms_epochs=args.nlms_epochs,
                        random_state=int(seed),
                        kmeans_kind=args.kmeans_kind,
                        save_ae_to_disk=args.save_ae_to_disk,
                        max_train_per_cluster=args.max_train_per_cluster,
                        n_jobs=args.n_jobs,
                        batch_size=args.batch_size,
                        project_stochastic=True,
                        verbose=args.verbose_fit,
                    )
                    fit_seconds = time.time() - fit_start

                    eval_start = time.time()
                    eval_result = evaluate_kahkm(
                        fit,
                        X0_test,
                        X1_test,
                        n_jobs=args.n_jobs,
                        batch_size=args.batch_size,
                        show_progress=False,
                    )
                    psi_test_sequence = kahm_associations(
                        fit.abstraction_model,
                        test_state_sequence,
                        omega=fit.omega,
                        tau=fit.tau,
                        n_jobs=args.n_jobs,
                        batch_size=args.batch_size,
                        show_progress=False,
                    )
                    eval_seconds = time.time() - eval_start

                    B_raw = np.asarray(fit.B, dtype=np.float64)
                    one_step_rows.append(
                        {
                            "experiment": "09_vanderpol_kahkm_sensitivity",
                            "status": "ok",
                            "seed": int(seed),
                            "n_steps": args.n_steps,
                            "dt": args.dt,
                            "mu": args.mu,
                            "train_fraction": args.train_fraction,
                            "n_train": int(X0_train.shape[1]),
                            "n_test": int(X0_test.shape[1]),
                            "state_dim": int(X0.shape[0]),
                            "n_clusters": int(n_clusters),
                            "effective_regimes": int(B_raw.shape[0]),
                            "subspace_dim": args.subspace_dim,
                            "Nb": args.nb,
                            "omega": float(omega),
                            "tau": args.tau,
                            "beta": args.beta,
                            "nlms_epochs": args.nlms_epochs,
                            "train_closure_error": float(fit.train_closure_error),
                            "train_association_r2": float(fit.association_r2),
                            "test_closure_error": float(eval_result.closure_error),
                            "test_association_r2": float(eval_result.association_r2),
                            "raw_predicted_simplex_violation": float(eval_result.simplex_violation_raw),
                            "projected_stochastic_simplex_violation": float(eval_result.simplex_violation_stochastic)
                            if eval_result.simplex_violation_stochastic is not None
                            else float("nan"),
                            "nlms_history_first": float(fit.nlms_history[0]) if fit.nlms_history else float("nan"),
                            "nlms_history_last": float(fit.nlms_history[-1]) if fit.nlms_history else float("nan"),
                            "B_raw_spectral_radius": _safe_spectral_radius(B_raw),
                            "B_raw_min_entry": float(np.min(B_raw)),
                            "B_raw_max_entry": float(np.max(B_raw)),
                            "B_raw_row_sum_min": float(np.min(np.sum(B_raw, axis=1))),
                            "B_raw_row_sum_max": float(np.max(np.sum(B_raw, axis=1))),
                            "fit_seconds": float(fit_seconds),
                            "eval_seconds": float(eval_seconds),
                            "total_job_seconds": float(time.time() - job_start),
                            "error_message": "",
                        }
                    )

                    for row in _rollout_rows(B=B_raw, psi_sequence=psi_test_sequence, horizons=args.horizons):
                        multistep_rows.append(
                            {
                                "experiment": "09_vanderpol_kahkm_sensitivity",
                                "seed": int(seed),
                                "n_steps": args.n_steps,
                                "dt": args.dt,
                                "mu": args.mu,
                                "n_clusters": int(n_clusters),
                                "effective_regimes": int(B_raw.shape[0]),
                                "omega": float(omega),
                                "tau": args.tau,
                                "beta": args.beta,
                                "nlms_epochs": args.nlms_epochs,
                                **row,
                            }
                        )
                    print(
                        f"OK: test_err={float(eval_result.closure_error):.6g}, "
                        f"test_R2={float(eval_result.association_r2):.6g}, "
                        f"fit={fit_seconds:.2f}s, eval={eval_seconds:.2f}s"
                    )
                except Exception as exc:  # noqa: BLE001 - experiments should continue after one failed grid point.
                    msg = f"{type(exc).__name__}: {exc}"
                    print(f"FAILED: {msg}")
                    failure_rows.append(
                        {
                            "experiment": "09_vanderpol_kahkm_sensitivity",
                            "status": "failed",
                            "seed": int(seed),
                            "n_steps": args.n_steps,
                            "dt": args.dt,
                            "mu": args.mu,
                            "n_clusters": int(n_clusters),
                            "omega": float(omega),
                            "total_job_seconds": float(time.time() - job_start),
                            "error_message": msg,
                        }
                    )

    one_step_summary = _summarize_one_step(one_step_rows)
    multistep_summary = _summarize_multistep(multistep_rows)

    one_step_path = args.output_dir / "experiment_09_one_step_raw_results.csv"
    multistep_path = args.output_dir / "experiment_09_multistep_raw_results.csv"
    one_step_summary_path = args.output_dir / "experiment_09_one_step_summary.csv"
    multistep_summary_path = args.output_dir / "experiment_09_multistep_summary.csv"
    failure_path = args.output_dir / "experiment_09_failures.csv"
    metadata_path = args.output_dir / "experiment_09_metadata.json"

    if one_step_rows:
        write_csv(one_step_path, one_step_rows)
    if multistep_rows:
        write_csv(multistep_path, multistep_rows)
    if one_step_summary:
        write_csv(one_step_summary_path, one_step_summary)
    if multistep_summary:
        write_csv(multistep_summary_path, multistep_summary)
    if failure_rows:
        write_csv(failure_path, failure_rows)

    metadata = {
        "experiment": "09_vanderpol_kahkm_sensitivity",
        "description": "Van der Pol KAHKM sensitivity over number of regimes and association sharpness.",
        "args": _metadata_args(args),
        "n_successful_jobs": len(one_step_rows),
        "n_failed_jobs": len(failure_rows),
        "output_files": {
            "one_step_raw_results": str(one_step_path),
            "multistep_raw_results": str(multistep_path),
            "one_step_summary": str(one_step_summary_path),
            "multistep_summary": str(multistep_summary_path),
            "failures": str(failure_path) if failure_rows else "",
        },
        "total_seconds": time.time() - t0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Experiment 09 complete ===")
    print(f"Successful jobs: {len(one_step_rows)}")
    print(f"Failed jobs:     {len(failure_rows)}")
    print(f"One-step raw:    {one_step_path}")
    print(f"Multi-step raw:  {multistep_path}")
    print(f"One-step summary:{one_step_summary_path}")
    print(f"Multi-step sum.: {multistep_summary_path}")
    print(f"Metadata:        {metadata_path}")
    if one_step_summary:
        print("\nBest one-step configurations by mean test closure error:")
        for row in one_step_summary[: min(8, len(one_step_summary))]:
            print(
                f"  C={int(row['n_clusters']):>2}, omega={float(row['omega']):>5g}: "
                f"test_err={float(row['test_closure_error_mean']):.6g} +/- {float(row['test_closure_error_std']):.3g}, "
                f"R2={float(row['test_association_r2_mean']):.6g}, "
                f"simplex_max={float(row['raw_predicted_simplex_violation_max']):.3g}"
            )


if __name__ == "__main__":
    main()
