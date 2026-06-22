#!/usr/bin/env python3
"""
Experiment 03: multi-step KAHKM association rollout on Duffing oscillator.

Purpose
-------
Experiment 01 tested one-step held-out closure.
Experiment 02 tested sensitivity over C and omega.
This experiment tests whether the selected abstraction remains useful when the
learned abstract Koopman matrix is rolled forward for multiple steps:

    psi_hat_{t+h} = (B.T)^h psi_t

The metric is relative squared error in association space, not state-space error.
This matches the KAHKM manuscript's regime-level Koopman-abstraction claim.

Expected files
--------------
Run this script in the same folder as:
  - kernel_affine_hull_koopman_machines.py
  - parallel_autoencoders.py
  - combine_multiple_autoencoders_extended.py

Example
-------
python experiment_03_duffing_multistep_pylance_clean.py

Fast smoke test
---------------
python experiment_03_duffing_multistep_pylance_clean.py --random-states 0 --n-steps 600 --horizons 1 2 5 10 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast

import numpy as np

from kernel_affine_hull_koopman_machines import (
    fit_kahkm,
    kahm_associations,
    make_duffing_snapshots,
    simplex_violation,
)

KMeansKind: TypeAlias = Literal["auto", "full", "minibatch"]
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
    """Convert argparse list-like values to list[int] without using int(object)."""
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


def _rollout_errors(
    *,
    matrix: np.ndarray,
    psi_sequence: np.ndarray,
    horizons: Sequence[int],
    matrix_label: str,
) -> list[CsvRow]:
    """Compute free-run h-step errors from a sequence psi(x_t).

    Parameters
    ----------
    matrix:
        Closure matrix B shaped (C, C). Predictions use B.T @ psi.
    psi_sequence:
        Association sequence shaped (C, T + 1). For horizon h, starts are
        columns 0 .. T-h and targets are columns h .. T.
    horizons:
        Positive rollout horizons.
    matrix_label:
        Label recorded in the output CSV, for example "raw" or
        "row_stochastic".
    """
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
            pred = B.T @ pred
        target = psi[:, horizon:]
        out.append(
            {
                "matrix": matrix_label,
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
    parser = argparse.ArgumentParser(description="Experiment 03: multi-step KAHKM association rollout on Duffing.")
    parser.add_argument("--output-dir", type=Path, default=Path("kahkm_experiment_03_outputs"))
    parser.add_argument("--n-steps", type=_positive_int, default=1200)
    parser.add_argument("--dt", type=_positive_float, default=0.03)
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.75)

    # Defaults chosen from Experiment 02: best mean held-out closure error.
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

    # Full state sequence x_split, ..., x_T for the held-out segment.
    test_state_sequence = np.concatenate([X0_test, X1_test[:, -1:]], axis=1)

    raw_rows: list[CsvRow] = []
    fit_rows: list[CsvRow] = []

    total_runs = len(args.random_states)
    for run_i, random_state in enumerate(args.random_states, start=1):
        print(f"\n=== Experiment 03 run {run_i}/{total_runs}: random_state={random_state} ===")
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

        matrices: list[tuple[str, np.ndarray]] = [("raw", np.asarray(fit.B, dtype=np.float64))]
        if fit.B_stochastic is not None:
            matrices.append(("row_stochastic", np.asarray(fit.B_stochastic, dtype=np.float64)))

        for matrix_label, matrix in matrices:
            rollout_rows = _rollout_errors(
                matrix=matrix,
                psi_sequence=psi_test_sequence,
                horizons=args.horizons,
                matrix_label=matrix_label,
            )
            for row in rollout_rows:
                raw_rows.append(
                    {
                        "experiment": "03_duffing_kahkm_multistep_association_rollout",
                        "random_state": random_state,
                        "n_steps": args.n_steps,
                        "dt": args.dt,
                        "data_seed": args.data_seed,
                        "train_fraction": args.train_fraction,
                        "n_train": int(X0_train.shape[1]),
                        "n_test_snapshots": int(X0_test.shape[1]),
                        "state_dim": int(X0.shape[0]),
                        "n_clusters": args.n_clusters,
                        "effective_regimes": int(fit.B.shape[0]),
                        "subspace_dim": args.subspace_dim,
                        "Nb": args.nb,
                        "omega": args.omega,
                        "tau": args.tau,
                        "beta": args.beta,
                        "nlms_epochs": args.nlms_epochs,
                        **row,
                    }
                )

        raw_B = np.asarray(fit.B, dtype=np.float64)
        stochastic_B = None if fit.B_stochastic is None else np.asarray(fit.B_stochastic, dtype=np.float64)
        fit_rows.append(
            {
                "experiment": "03_duffing_kahkm_multistep_association_rollout",
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
        {(str(row["matrix"]), int(row["horizon"])) for row in raw_rows},
        key=lambda item: (item[0], item[1]),
    )
    for matrix_label, horizon in keys:
        group = [row for row in raw_rows if str(row["matrix"]) == matrix_label and int(row["horizon"]) == horizon]
        errs = [float(row["relative_association_error"]) for row in group]
        r2s = [float(row["association_r2"]) for row in group]
        viols = [float(row["simplex_violation"]) for row in group]
        summary_rows.append(
            {
                "matrix": matrix_label,
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

    raw_path = args.output_dir / "experiment_03_multistep_raw_results.csv"
    fit_path = args.output_dir / "experiment_03_fit_diagnostics.csv"
    summary_path = args.output_dir / "experiment_03_multistep_summary.csv"
    metadata_path = args.output_dir / "experiment_03_metadata.json"

    write_csv(raw_path, raw_rows)
    write_csv(fit_path, fit_rows)
    write_csv(summary_path, summary_rows)
    metadata = {
        "experiment": "03_duffing_kahkm_multistep_association_rollout",
        "description": "Free-run h-step prediction in KAHKM association space using raw and row-stochastic closure matrices.",
        "args": _metadata_args(args),
        "output_files": {
            "raw_results": str(raw_path),
            "fit_diagnostics": str(fit_path),
            "summary": str(summary_path),
        },
        "total_seconds": time.time() - t0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Experiment 03 complete ===")
    print(f"Raw results:      {raw_path}")
    print(f"Fit diagnostics:  {fit_path}")
    print(f"Summary:          {summary_path}")
    print(f"Metadata:         {metadata_path}")
    print("\nKey summary rows:")
    for row in summary_rows:
        matrix_label = str(row["matrix"])
        horizon = int(row["horizon"])
        err_mean = float(row["relative_association_error_mean"])
        err_std = float(row["relative_association_error_std"])
        r2_mean = float(row["association_r2_mean"])
        simplex_max = float(row["simplex_violation_max"])
        print(
            f"  {matrix_label:>14} h={horizon:>3}: "
            f"err={err_mean:.6g} +/- {err_std:.3g}, "
            f"R2={r2_mean:.6g}, "
            f"simplex_max={simplex_max:.3g}"
        )


if __name__ == "__main__":
    main()
