#!/usr/bin/env python3
"""
Experiment 01: KAHKM one-step regime-closure on the Duffing oscillator.

Place this file in the same folder as:
  - kernel_affine_hull_koopman_machines.py
  - parallel_autoencoders.py
  - combine_multiple_autoencoders_extended.py

Then run, for example:
  python experiment_01_duffing_kahkm.py

For a quick smoke test:
  python experiment_01_duffing_kahkm.py --n-steps 500 --n-clusters 12 --nb 40 --nlms-epochs 8 --batch-size 128

Outputs are written to ./kahkm_experiment_01_outputs by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from kernel_affine_hull_koopman_machines import (
    evaluate_kahkm,
    fit_kahkm,
    make_duffing_snapshots,
)


def _float_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def matrix_summary(B: np.ndarray, prefix: str) -> dict[str, float]:
    """Small diagnostics for the learned closure matrix."""
    eigvals = np.linalg.eigvals(B)
    spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else float("nan")
    return {
        f"{prefix}_fro_norm": float(np.linalg.norm(B, ord="fro")),
        f"{prefix}_spectral_radius": spectral_radius,
        f"{prefix}_min_entry": float(np.min(B)),
        f"{prefix}_max_entry": float(np.max(B)),
        f"{prefix}_row_sum_min": float(np.min(B.sum(axis=1))),
        f"{prefix}_row_sum_max": float(np.max(B.sum(axis=1))),
    }


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 01: KAHKM one-step closure on Duffing oscillator."
    )

    # Dataset
    parser.add_argument("--n-steps", type=int, default=1200, help="Number of Duffing snapshot pairs.")
    parser.add_argument("--dt", type=float, default=0.03, help="RK4 time step used by the built-in Duffing generator.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for Duffing trajectory generation.")
    parser.add_argument("--train-fraction", type=float, default=0.75, help="Chronological train/test split fraction.")

    # KAHKM abstraction
    parser.add_argument("--n-clusters", type=int, default=25, help="Number of state-regime clusters C.")
    parser.add_argument("--subspace-dim", type=int, default=20, help="KAHM/OTFL subspace dimension.")
    parser.add_argument("--nb", type=int, default=100, help="Nb passed to parallel_autoencoders.")
    parser.add_argument("--omega", type=float, default=8.0, help="Association sharpness parameter omega.")
    parser.add_argument("--tau", type=float, default=1e-6, help="Positive association offset tau.")
    parser.add_argument("--kmeans-kind", choices=["auto", "full", "minibatch"], default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--singleton-strategy", choices=["augment", "merge"], default="augment")
    parser.add_argument("--singleton-aux-mix", type=float, default=0.05)
    parser.add_argument("--max-train-per-cluster", type=int, default=None)

    # Closure estimator
    parser.add_argument("--beta", type=float, default=0.1, help="NLMS step size; manuscript assumes 0 < beta < 1.")
    parser.add_argument("--nlms-epochs", type=int, default=20, help="Number of passes through training snapshots.")
    parser.add_argument("--no-project-stochastic", action="store_true", help="Disable row-simplex projection of B.")

    # Runtime/output
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for folding evaluation.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for folding evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("kahkm_experiment_01_outputs"))
    parser.add_argument("--save-ae-to-disk", action="store_true", help="Save AE shards to disk instead of keeping them in memory.")
    parser.add_argument("--ae-dir", type=Path, default=None, help="Optional AE cache directory if --save-ae-to-disk is used.")
    parser.add_argument("--overwrite-ae-dir", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")

    run_started = time.time()

    if verbose:
        print("\n=== Experiment 01: KAHKM Duffing one-step closure ===")
        print(f"Python: {sys.version.split()[0]} on {platform.platform()}")
        print(f"Output directory: {out_dir.resolve()}")

    # 1) Generate snapshot pairs X0, X1 with shape (D, N).
    X0, X1 = make_duffing_snapshots(n_steps=args.n_steps, dt=args.dt, seed=args.seed)
    split = int(args.train_fraction * X0.shape[1])
    X0_train, X1_train = X0[:, :split], X1[:, :split]
    X0_test, X1_test = X0[:, split:], X1[:, split:]

    if verbose:
        print("\nData")
        print(f"  X0 shape: {X0.shape}, X1 shape: {X1.shape}")
        print(f"  train snapshots: {X0_train.shape[1]}")
        print(f"  test snapshots:  {X0_test.shape[1]}")

    # 2) Fit KAHKM on the training segment.
    t_fit0 = time.time()
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
        random_state=args.seed,
        kmeans_kind=args.kmeans_kind,
        kmeans_batch_size=args.kmeans_batch_size,
        singleton_strategy=args.singleton_strategy,
        singleton_aux_mix=args.singleton_aux_mix,
        max_train_per_cluster=args.max_train_per_cluster,
        save_ae_to_disk=args.save_ae_to_disk,
        ae_dir=None if args.ae_dir is None else str(args.ae_dir),
        overwrite_ae_dir=args.overwrite_ae_dir,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        project_stochastic=not args.no_project_stochastic,
        verbose=verbose,
    )
    fit_seconds = time.time() - t_fit0

    # 3) Evaluate held-out one-step closure on the test segment.
    t_eval0 = time.time()
    test = evaluate_kahkm(
        fit,
        X0_test,
        X1_test,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        show_progress=verbose,
    )
    eval_seconds = time.time() - t_eval0

    # 4) Collect scalar diagnostics.
    effective_regimes = int(fit.B.shape[0])
    result: dict[str, Any] = {
        "experiment": "01_duffing_kahkm_one_step_closure",
        "n_steps": int(args.n_steps),
        "dt": float(args.dt),
        "seed": int(args.seed),
        "train_fraction": float(args.train_fraction),
        "n_train": int(X0_train.shape[1]),
        "n_test": int(X0_test.shape[1]),
        "state_dim": int(X0.shape[0]),
        "requested_regimes": int(args.n_clusters),
        "effective_regimes": effective_regimes,
        "subspace_dim": int(args.subspace_dim),
        "Nb": int(args.nb),
        "omega": float(args.omega),
        "tau": float(args.tau),
        "beta": float(args.beta),
        "nlms_epochs": int(args.nlms_epochs),
        "kmeans_kind": args.kmeans_kind,
        "batch_size": None if args.batch_size is None else int(args.batch_size),
        "n_jobs": int(args.n_jobs),
        "train_closure_error": float(fit.train_closure_error),
        "train_association_r2": _float_or_none(fit.association_r2),
        "test_closure_error": float(test.closure_error),
        "test_association_r2": _float_or_none(test.association_r2),
        "raw_predicted_simplex_violation": float(test.simplex_violation_raw),
        "projected_stochastic_simplex_violation": _float_or_none(test.simplex_violation_stochastic),
        "fit_seconds": float(fit_seconds),
        "eval_seconds": float(eval_seconds),
        "total_seconds": float(time.time() - run_started),
        "final_nlms_train_error": float(fit.nlms_history[-1]) if fit.nlms_history else None,
        "nlms_history": [float(v) for v in fit.nlms_history],
    }
    result.update(matrix_summary(fit.B, "B_raw"))
    if fit.B_stochastic is not None:
        result.update(matrix_summary(fit.B_stochastic, "B_stochastic"))

    # 5) Save outputs.
    json_path = out_dir / "experiment_01_results.json"
    csv_path = out_dir / "experiment_01_results.csv"
    npz_path = out_dir / "experiment_01_matrices.npz"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_single_row_csv(csv_path, result)
    np.savez_compressed(
        npz_path,
        B=fit.B,
        B_stochastic=np.array([]) if fit.B_stochastic is None else fit.B_stochastic,
        X0_train=X0_train,
        X1_train=X1_train,
        X0_test=X0_test,
        X1_test=X1_test,
        nlms_history=np.asarray(fit.nlms_history, dtype=np.float64),
    )

    # 6) Print a compact summary for copying into chat.
    print("\n=== Experiment 01 summary ===")
    print(f"Effective regimes: {effective_regimes}")
    print(f"Train closure error: {result['train_closure_error']:.8g}")
    print(f"Train association R^2: {result['train_association_r2']:.8g}")
    print(f"Test closure error:  {result['test_closure_error']:.8g}")
    print(f"Test association R^2: {result['test_association_r2']:.8g}")
    print(f"Raw simplex violation: {result['raw_predicted_simplex_violation']:.8g}")
    if result["projected_stochastic_simplex_violation"] is not None:
        print(f"Projected stochastic simplex violation: {result['projected_stochastic_simplex_violation']:.8g}")
    print(f"Raw B spectral radius: {result['B_raw_spectral_radius']:.8g}")
    if "B_stochastic_spectral_radius" in result:
        print(f"Stochastic-projected B spectral radius: {result['B_stochastic_spectral_radius']:.8g}")
    print(f"Fit seconds: {fit_seconds:.2f}")
    print(f"Eval seconds: {eval_seconds:.2f}")
    print("\nSaved files")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")


if __name__ == "__main__":
    main()
