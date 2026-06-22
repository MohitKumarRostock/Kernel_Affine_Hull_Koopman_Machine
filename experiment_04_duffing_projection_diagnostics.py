#!/usr/bin/env python3
"""
Experiment 04: simplex-projection diagnostics for multi-step KAHKM rollouts.

Purpose
-------
Experiment 03 showed that the raw Koopman closure matrix can give accurate
multi-step association predictions, while the row-stochastic matrix projection
can drift over longer horizons. This experiment separates two effects:

1. Propagating with the raw closure matrix B.
2. Keeping predictions on the simplex by projecting the predicted association
   vector after each step, without modifying B.
3. Replacing B by its row-stochastic projection.

The main question is whether simplex feasibility can be enforced at the
prediction-vector level without sacrificing the raw Koopman rollout accuracy.

Expected files
--------------
Run this script in the same folder as:
  - kernel_affine_hull_koopman_machines.py
  - parallel_autoencoders.py
  - combine_multiple_autoencoders_extended.py

Example
-------
python experiment_04_duffing_projection_diagnostics.py

Fast smoke test
---------------
python experiment_04_duffing_projection_diagnostics.py --random-states 0 --n-steps 600 --horizons 1 2 5 10 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence, TypeAlias, cast

import numpy as np

from kernel_affine_hull_koopman_machines import (
    fit_kahkm,
    kahm_associations,
    make_duffing_snapshots,
    simplex_violation,
)

KMeansKind: TypeAlias = Literal["auto", "full", "minibatch"]
RolloutMode: TypeAlias = Literal[
    "raw_free",
    "raw_clip_renorm_each_step",
    "raw_simplex_project_each_step",
    "row_stochastic_free",
]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]


@dataclass(frozen=True)
class ExperimentArgs:
    output_dir: Path
    n_steps: int
    dt: float
    data_seed: int
    train_fraction: float
    n_clusters: int
    omega: float
    tau: float
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    random_states: list[int]
    kmeans_kind: KMeansKind
    batch_size: int
    n_jobs: int
    horizons: list[int]
    save_ae_to_disk: bool
    verbose_fit: bool


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
        return [int(item) for item in value]
    raise TypeError(f"{name} must be a sequence of integers.")


def _parse_kmeans_kind(value: object) -> KMeansKind:
    raw = str(value)
    if raw not in {"auto", "full", "minibatch"}:
        raise ValueError("--kmeans-kind must be one of: auto, full, minibatch")
    return cast(KMeansKind, raw)


def _relative_sq_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.sum((target - pred) ** 2))
    den = float(np.sum(target * target))
    return num / max(den, eps)


def _association_r2(pred: np.ndarray, target: np.ndarray) -> float:
    residual_ss = float(np.sum((target - pred) ** 2))
    total_ss = float(np.sum((target - target.mean(axis=1, keepdims=True)) ** 2))
    if total_ss <= 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def _safe_spectral_radius(matrix: np.ndarray) -> float:
    vals = np.linalg.eigvals(np.asarray(matrix, dtype=np.float64))
    return float(np.max(np.abs(vals)))


def clip_and_renormalize_columns(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip negative association entries to zero and renormalize each column."""
    out = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    sums = np.sum(out, axis=0, keepdims=True)
    bad = sums <= eps
    if np.any(bad):
        out[:, np.ravel(bad)] = 1.0 / float(out.shape[0])
        sums = np.sum(out, axis=0, keepdims=True)
    return out / sums


def project_columns_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of each column onto the probability simplex.

    Solves, independently for each column v,
        argmin_p ||p - v||_2 subject to p >= 0 and sum(p) = 1.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("values must be shaped (C, N).")
    c_count = int(v.shape[0])
    u = np.sort(v, axis=0)[::-1, :]
    cssv = np.cumsum(u, axis=0) - 1.0
    ind = np.arange(1, c_count + 1, dtype=np.float64).reshape(-1, 1)
    cond = u - cssv / ind > 0.0
    rho = np.sum(cond, axis=0) - 1
    rho = np.maximum(rho, 0).astype(np.int64, copy=False)
    theta = cssv[rho, np.arange(v.shape[1])] / (rho.astype(np.float64) + 1.0)
    return np.maximum(v - theta.reshape(1, -1), 0.0)


def _propagate_one_step(pred: np.ndarray, *, matrix: np.ndarray, mode: RolloutMode) -> np.ndarray:
    next_pred = matrix.T @ pred
    if mode == "raw_clip_renorm_each_step":
        return clip_and_renormalize_columns(next_pred)
    if mode == "raw_simplex_project_each_step":
        return project_columns_to_simplex(next_pred)
    return next_pred


def _rollout_errors(
    *,
    matrix: np.ndarray,
    psi_sequence: np.ndarray,
    horizons: Sequence[int],
    mode: RolloutMode,
) -> list[CsvRow]:
    psi = np.asarray(psi_sequence, dtype=np.float64)
    B = np.asarray(matrix, dtype=np.float64)
    c_count, seq_len = psi.shape
    expected_shape = (c_count, c_count)
    if B.shape != expected_shape:
        raise ValueError(f"B has shape {B.shape}, expected {expected_shape}")
    if not horizons:
        raise ValueError("At least one horizon is required.")
    max_h = max(horizons)
    if max_h >= seq_len:
        raise ValueError(f"Maximum horizon {max_h} must be smaller than psi sequence length {seq_len}.")

    out: list[CsvRow] = []
    for horizon in horizons:
        pred = psi[:, : seq_len - horizon].copy()
        for _ in range(horizon):
            pred = _propagate_one_step(pred, matrix=B, mode=mode)
        target = psi[:, horizon:]
        out.append(
            {
                "rollout_mode": mode,
                "horizon": horizon,
                "n_start_points": int(target.shape[1]),
                "relative_association_error": _relative_sq_error(pred, target),
                "association_r2": _association_r2(pred, target),
                "simplex_violation": float(simplex_violation(pred)),
            }
        )
    return out


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
    parser = argparse.ArgumentParser(description="Experiment 04: simplex-projection diagnostics for KAHKM rollouts.")
    parser.add_argument("--output-dir", type=Path, default=Path("kahkm_experiment_04_outputs"))
    parser.add_argument("--n-steps", type=_positive_int, default=1200)
    parser.add_argument("--dt", type=_positive_float, default=0.03)
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.75)

    # Defaults chosen from Experiment 02 and reused in Experiment 03.
    parser.add_argument("--n-clusters", type=_positive_int, default=15)
    parser.add_argument("--omega", type=_positive_float, default=12.0)
    parser.add_argument("--tau", type=_positive_float, default=1e-6)
    parser.add_argument("--subspace-dim", type=_positive_int, default=4)
    parser.add_argument("--nb", type=_positive_int, default=100)
    parser.add_argument("--beta", type=_positive_float, default=0.1)
    parser.add_argument("--nlms-epochs", type=_positive_int, default=20)
    parser.add_argument("--random-states", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--kmeans-kind", choices=["auto", "full", "minibatch"], default="auto")
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--horizons", type=_positive_int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--verbose-fit", action="store_true")

    namespace = parser.parse_args()
    return ExperimentArgs(
        output_dir=Path(namespace.output_dir),
        n_steps=int(namespace.n_steps),
        dt=float(namespace.dt),
        data_seed=int(namespace.data_seed),
        train_fraction=float(namespace.train_fraction),
        n_clusters=int(namespace.n_clusters),
        omega=float(namespace.omega),
        tau=float(namespace.tau),
        subspace_dim=int(namespace.subspace_dim),
        nb=int(namespace.nb),
        beta=float(namespace.beta),
        nlms_epochs=int(namespace.nlms_epochs),
        random_states=_object_to_int_list(namespace.random_states, name="--random-states"),
        kmeans_kind=_parse_kmeans_kind(namespace.kmeans_kind),
        batch_size=int(namespace.batch_size),
        n_jobs=int(namespace.n_jobs),
        horizons=_object_to_int_list(namespace.horizons, name="--horizons"),
        save_ae_to_disk=bool(namespace.save_ae_to_disk),
        verbose_fit=bool(namespace.verbose_fit),
    )


def _metadata_args(args: ExperimentArgs) -> dict[str, object]:
    data: dict[str, object] = dict(asdict(args))
    data["output_dir"] = str(args.output_dir)
    return data


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must lie in (0, 1).")

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X0, X1 = make_duffing_snapshots(n_steps=args.n_steps, dt=args.dt, seed=args.data_seed)
    split = int(args.train_fraction * X0.shape[1])
    X0_train, X1_train = X0[:, :split], X1[:, :split]
    X0_test, X1_test = X0[:, split:], X1[:, split:]
    test_state_sequence = np.concatenate([X0_test, X1_test[:, -1:]], axis=1)

    raw_rows: list[CsvRow] = []
    fit_rows: list[CsvRow] = []

    total_runs = len(args.random_states)
    for run_i, random_state in enumerate(args.random_states, start=1):
        print(f"\n=== Experiment 04 run {run_i}/{total_runs}: random_state={random_state} ===")
        fit_start = time.time()
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
            random_state=random_state,
            kmeans_kind=args.kmeans_kind,
            save_ae_to_disk=args.save_ae_to_disk,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            project_stochastic=True,
            verbose=args.verbose_fit,
        )
        fit_seconds = time.time() - fit_start

        assoc_start = time.time()
        psi_test_sequence = kahm_associations(
            fit.abstraction_model,
            test_state_sequence,
            omega=fit.omega,
            tau=fit.tau,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            show_progress=False,
        )
        assoc_seconds = time.time() - assoc_start

        raw_B = np.asarray(fit.B, dtype=np.float64)
        stochastic_B = None if fit.B_stochastic is None else np.asarray(fit.B_stochastic, dtype=np.float64)

        rollout_specs: list[tuple[RolloutMode, np.ndarray]] = [
            ("raw_free", raw_B),
            ("raw_clip_renorm_each_step", raw_B),
            ("raw_simplex_project_each_step", raw_B),
        ]
        if stochastic_B is not None:
            rollout_specs.append(("row_stochastic_free", stochastic_B))

        for mode, matrix in rollout_specs:
            rollout_rows = _rollout_errors(
                matrix=matrix,
                psi_sequence=psi_test_sequence,
                horizons=args.horizons,
                mode=mode,
            )
            for row in rollout_rows:
                raw_rows.append(
                    {
                        "experiment": "04_duffing_kahkm_projection_diagnostics",
                        "random_state": random_state,
                        "n_steps": args.n_steps,
                        "dt": args.dt,
                        "data_seed": args.data_seed,
                        "train_fraction": args.train_fraction,
                        "n_train": int(X0_train.shape[1]),
                        "n_test_snapshots": int(X0_test.shape[1]),
                        "state_dim": int(X0.shape[0]),
                        "n_clusters": args.n_clusters,
                        "effective_regimes": int(raw_B.shape[0]),
                        "subspace_dim": args.subspace_dim,
                        "Nb": args.nb,
                        "omega": args.omega,
                        "tau": args.tau,
                        "beta": args.beta,
                        "nlms_epochs": args.nlms_epochs,
                        **row,
                    }
                )

        fit_rows.append(
            {
                "experiment": "04_duffing_kahkm_projection_diagnostics",
                "random_state": random_state,
                "n_clusters": args.n_clusters,
                "effective_regimes": int(raw_B.shape[0]),
                "omega": args.omega,
                "tau": args.tau,
                "train_closure_error": float(fit.train_closure_error),
                "train_association_r2": float(fit.association_r2),
                "nlms_history_first": float(fit.nlms_history[0]) if fit.nlms_history else float("nan"),
                "nlms_history_last": float(fit.nlms_history[-1]) if fit.nlms_history else float("nan"),
                "B_raw_spectral_radius": _safe_spectral_radius(raw_B),
                "B_raw_min_entry": float(np.min(raw_B)),
                "B_raw_max_entry": float(np.max(raw_B)),
                "B_raw_row_sum_min": float(np.min(np.sum(raw_B, axis=1))),
                "B_raw_row_sum_max": float(np.max(np.sum(raw_B, axis=1))),
                "B_stochastic_spectral_radius": _safe_spectral_radius(stochastic_B) if stochastic_B is not None else float("nan"),
                "B_stochastic_min_entry": float(np.min(stochastic_B)) if stochastic_B is not None else float("nan"),
                "B_stochastic_max_entry": float(np.max(stochastic_B)) if stochastic_B is not None else float("nan"),
                "B_stochastic_row_sum_min": float(np.min(np.sum(stochastic_B, axis=1))) if stochastic_B is not None else float("nan"),
                "B_stochastic_row_sum_max": float(np.max(np.sum(stochastic_B, axis=1))) if stochastic_B is not None else float("nan"),
                "fit_seconds": float(fit_seconds),
                "association_eval_seconds": float(assoc_seconds),
            }
        )

    summary_rows: list[CsvRow] = []
    keys = sorted(
        {(str(row["rollout_mode"]), int(row["horizon"])) for row in raw_rows},
        key=lambda item: (item[0], item[1]),
    )
    for mode, horizon in keys:
        group = [row for row in raw_rows if str(row["rollout_mode"]) == mode and int(row["horizon"]) == horizon]
        errs = [float(row["relative_association_error"]) for row in group]
        r2s = [float(row["association_r2"]) for row in group]
        viols = [float(row["simplex_violation"]) for row in group]
        summary_rows.append(
            {
                "rollout_mode": mode,
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

    raw_path = args.output_dir / "experiment_04_projection_raw_results.csv"
    fit_path = args.output_dir / "experiment_04_fit_diagnostics.csv"
    summary_path = args.output_dir / "experiment_04_projection_summary.csv"
    metadata_path = args.output_dir / "experiment_04_metadata.json"

    write_csv(raw_path, raw_rows)
    write_csv(fit_path, fit_rows)
    write_csv(summary_path, summary_rows)
    metadata = {
        "experiment": "04_duffing_kahkm_projection_diagnostics",
        "description": "Compares raw matrix rollout, per-step prediction-vector simplex enforcement, and row-stochastic matrix projection.",
        "args": _metadata_args(args),
        "output_files": {
            "raw_results": str(raw_path),
            "fit_diagnostics": str(fit_path),
            "summary": str(summary_path),
        },
        "total_seconds": time.time() - t0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Experiment 04 complete ===")
    print(f"Raw results:      {raw_path}")
    print(f"Fit diagnostics:  {fit_path}")
    print(f"Summary:          {summary_path}")
    print(f"Metadata:         {metadata_path}")
    print("\nKey summary rows:")
    for row in summary_rows:
        mode = str(row["rollout_mode"])
        horizon = int(row["horizon"])
        err_mean = float(row["relative_association_error_mean"])
        err_std = float(row["relative_association_error_std"])
        r2_mean = float(row["association_r2_mean"])
        simplex_max = float(row["simplex_violation_max"])
        print(
            f"  {mode:>30} h={horizon:>3}: "
            f"err={err_mean:.6g} +/- {err_std:.3g}, "
            f"R2={r2_mean:.6g}, "
            f"simplex_max={simplex_max:.3g}"
        )


if __name__ == "__main__":
    main()
