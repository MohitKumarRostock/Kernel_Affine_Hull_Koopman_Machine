#!/usr/bin/env python3
"""
Experiment 02: KAHKM sensitivity study on the Duffing oscillator.

Purpose
-------
Experiment 01 showed that one fixed KAHKM configuration can close the Duffing
regime-association dynamics. Experiment 02 tests whether that result is robust
to the two main abstraction-design choices:

  - number of regimes C = n_clusters
  - association sharpness omega

Place this file in the same folder as:
  - kernel_affine_hull_koopman_machines.py
  - parallel_autoencoders.py
  - combine_multiple_autoencoders_extended.py

Run default grid:
  python experiment_02_duffing_sensitivity.py

Smaller smoke test:
  python experiment_02_duffing_sensitivity.py --clusters 15 25 --omegas 4 8 --seeds 0

Outputs are written to ./kahkm_experiment_02_outputs by default:
  - experiment_02_raw_results.csv
  - experiment_02_summary.csv
  - experiment_02_metadata.json

The most important columns are:
  test_closure_error_mean, test_closure_error_std,
  test_association_r2_mean, raw_simplex_violation_mean,
  B_raw_spectral_radius_mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import traceback
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


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_grid(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate completed runs by (n_clusters, omega)."""
    completed = [r for r in raw_rows if r.get("status") == "ok"]
    keys = sorted({(int(r["n_clusters"]), float(r["omega"])) for r in completed})
    metrics = [
        "train_closure_error",
        "test_closure_error",
        "train_association_r2",
        "test_association_r2",
        "raw_predicted_simplex_violation",
        "projected_stochastic_simplex_violation",
        "B_raw_spectral_radius",
        "B_raw_min_entry",
        "B_raw_max_entry",
        "fit_seconds",
        "eval_seconds",
        "total_seconds",
    ]
    summary: list[dict[str, Any]] = []
    for n_clusters, omega in keys:
        group = [r for r in completed if int(r["n_clusters"]) == n_clusters and float(r["omega"]) == omega]
        row: dict[str, Any] = {
            "n_clusters": n_clusters,
            "omega": omega,
            "n_completed": len(group),
        }
        if group:
            row["effective_regimes_min"] = min(int(r["effective_regimes"]) for r in group)
            row["effective_regimes_max"] = max(int(r["effective_regimes"]) for r in group)
        for metric in metrics:
            vals: list[float] = []
            for r in group:
                value = r.get(metric)
                if value is None or value == "":
                    continue
                try:
                    f = float(value)
                except Exception:
                    continue
                if np.isfinite(f):
                    vals.append(f)
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                row[f"{metric}_mean"] = float(np.mean(arr))
                row[f"{metric}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                row[f"{metric}_min"] = float(np.min(arr))
                row[f"{metric}_max"] = float(np.max(arr))
        summary.append(row)

    summary.sort(key=lambda r: (float(r.get("test_closure_error_mean", np.inf)), -float(r.get("test_association_r2_mean", -np.inf))))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 02: KAHKM sensitivity over C and omega on Duffing."
    )

    # Dataset
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--data-seed", type=int, default=1, help="Seed for the Duffing trajectory.")
    parser.add_argument("--train-fraction", type=float, default=0.75)

    # Sensitivity grid
    parser.add_argument("--clusters", type=int, nargs="+", default=[15, 25, 35])
    parser.add_argument("--omegas", type=float, nargs="+", default=[4.0, 8.0, 12.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds for KAHKM/KMeans.")

    # KAHKM abstraction
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--kmeans-kind", choices=["auto", "full", "minibatch"], default="auto")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--singleton-strategy", choices=["augment", "merge"], default="augment")
    parser.add_argument("--singleton-aux-mix", type=float, default=0.05)
    parser.add_argument("--max-train-per-cluster", type=int, default=None)

    # Closure estimator
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--no-project-stochastic", action="store_true")

    # Runtime/output
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("kahkm_experiment_02_outputs"))
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--ae-root", type=Path, default=Path("kahkm_experiment_02_ae_cache"))
    parser.add_argument("--strict", action="store_true", help="Stop immediately if one run fails.")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_ae_to_disk:
        args.ae_root.mkdir(parents=True, exist_ok=True)

    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")

    started = time.time()

    if verbose:
        print("\n=== Experiment 02: KAHKM Duffing sensitivity ===")
        print(f"Python: {sys.version.split()[0]} on {platform.platform()}")
        print(f"Output directory: {out_dir.resolve()}")
        print(f"Clusters: {args.clusters}")
        print(f"Omegas:   {args.omegas}")
        print(f"Seeds:    {args.seeds}")

    X0, X1 = make_duffing_snapshots(n_steps=args.n_steps, dt=args.dt, seed=args.data_seed)
    split = int(args.train_fraction * X0.shape[1])
    X0_train, X1_train = X0[:, :split], X1[:, :split]
    X0_test, X1_test = X0[:, split:], X1[:, split:]

    raw_rows: list[dict[str, Any]] = []
    total_runs = len(args.clusters) * len(args.omegas) * len(args.seeds)
    run_idx = 0

    for n_clusters in args.clusters:
        for omega in args.omegas:
            for seed in args.seeds:
                run_idx += 1
                run_started = time.time()
                base_row: dict[str, Any] = {
                    "experiment": "02_duffing_kahkm_sensitivity",
                    "run_index": run_idx,
                    "total_runs": total_runs,
                    "n_steps": int(args.n_steps),
                    "dt": float(args.dt),
                    "data_seed": int(args.data_seed),
                    "random_state": int(seed),
                    "train_fraction": float(args.train_fraction),
                    "n_train": int(X0_train.shape[1]),
                    "n_test": int(X0_test.shape[1]),
                    "state_dim": int(X0.shape[0]),
                    "n_clusters": int(n_clusters),
                    "subspace_dim": int(args.subspace_dim),
                    "Nb": int(args.nb),
                    "omega": float(omega),
                    "tau": float(args.tau),
                    "beta": float(args.beta),
                    "nlms_epochs": int(args.nlms_epochs),
                    "kmeans_kind": args.kmeans_kind,
                    "batch_size": None if args.batch_size is None else int(args.batch_size),
                    "n_jobs": int(args.n_jobs),
                }

                if verbose:
                    print(f"\n[{run_idx}/{total_runs}] C={n_clusters}, omega={omega}, seed={seed}")

                try:
                    ae_dir = None
                    if args.save_ae_to_disk:
                        ae_dir = args.ae_root / f"C{n_clusters}_omega{omega:g}_seed{seed}"

                    t_fit0 = time.time()
                    fit = fit_kahkm(
                        X0_train,
                        X1_train,
                        n_clusters=int(n_clusters),
                        subspace_dim=int(args.subspace_dim),
                        Nb=int(args.nb),
                        omega=float(omega),
                        tau=float(args.tau),
                        beta=float(args.beta),
                        nlms_epochs=int(args.nlms_epochs),
                        random_state=int(seed),
                        kmeans_kind=args.kmeans_kind,
                        kmeans_batch_size=int(args.kmeans_batch_size),
                        singleton_strategy=args.singleton_strategy,
                        singleton_aux_mix=float(args.singleton_aux_mix),
                        max_train_per_cluster=args.max_train_per_cluster,
                        save_ae_to_disk=bool(args.save_ae_to_disk),
                        ae_dir=None if ae_dir is None else str(ae_dir),
                        overwrite_ae_dir=True,
                        n_jobs=int(args.n_jobs),
                        batch_size=args.batch_size,
                        project_stochastic=not args.no_project_stochastic,
                        verbose=False,
                    )
                    fit_seconds = time.time() - t_fit0

                    t_eval0 = time.time()
                    test = evaluate_kahkm(
                        fit,
                        X0_test,
                        X1_test,
                        n_jobs=int(args.n_jobs),
                        batch_size=args.batch_size,
                        show_progress=False,
                    )
                    eval_seconds = time.time() - t_eval0

                    row = dict(base_row)
                    row.update(
                        {
                            "status": "ok",
                            "effective_regimes": int(fit.B.shape[0]),
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
                            "nlms_history_first": float(fit.nlms_history[0]) if fit.nlms_history else None,
                            "nlms_history_last": float(fit.nlms_history[-1]) if fit.nlms_history else None,
                            "nlms_history_monotone_nonincreasing": bool(
                                all(fit.nlms_history[i + 1] <= fit.nlms_history[i] + 1e-15 for i in range(len(fit.nlms_history) - 1))
                            ) if fit.nlms_history else None,
                        }
                    )
                    row.update(matrix_summary(fit.B, "B_raw"))
                    if fit.B_stochastic is not None:
                        row.update(matrix_summary(fit.B_stochastic, "B_stochastic"))

                    if verbose:
                        print(
                            f"  test_closure_error={row['test_closure_error']:.8g}, "
                            f"test_R2={row['test_association_r2']:.8g}, "
                            f"raw_simplex_violation={row['raw_predicted_simplex_violation']:.3g}"
                        )

                except Exception as exc:
                    row = dict(base_row)
                    row.update(
                        {
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                            "total_seconds": float(time.time() - run_started),
                        }
                    )
                    print(f"  FAILED: {type(exc).__name__}: {exc}")
                    if args.strict:
                        raw_rows.append(row)
                        write_rows_csv(out_dir / "experiment_02_raw_results.csv", raw_rows)
                        raise

                raw_rows.append(row)
                write_rows_csv(out_dir / "experiment_02_raw_results.csv", raw_rows)

    summary_rows = summarize_grid(raw_rows)
    write_rows_csv(out_dir / "experiment_02_summary.csv", summary_rows)

    metadata = {
        "experiment": "02_duffing_kahkm_sensitivity",
        "created_by_script": Path(__file__).name,
        "python": sys.version,
        "platform": platform.platform(),
        "n_steps": int(args.n_steps),
        "dt": float(args.dt),
        "data_seed": int(args.data_seed),
        "train_fraction": float(args.train_fraction),
        "clusters": [int(x) for x in args.clusters],
        "omegas": [float(x) for x in args.omegas],
        "seeds": [int(x) for x in args.seeds],
        "n_raw_rows": len(raw_rows),
        "n_success": sum(1 for r in raw_rows if r.get("status") == "ok"),
        "n_failed": sum(1 for r in raw_rows if r.get("status") != "ok"),
        "total_seconds": float(time.time() - started),
    }
    with (out_dir / "experiment_02_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Experiment 02 summary ===")
    print(f"Completed runs: {metadata['n_success']}/{metadata['n_raw_rows']}")
    if summary_rows:
        best = summary_rows[0]
        print("Best mean test closure setting:")
        print(f"  C={best['n_clusters']}, omega={best['omega']}")
        print(f"  test_closure_error_mean={best.get('test_closure_error_mean', float('nan')):.8g}")
        print(f"  test_closure_error_std={best.get('test_closure_error_std', float('nan')):.8g}")
        print(f"  test_association_r2_mean={best.get('test_association_r2_mean', float('nan')):.8g}")
    print("\nSaved files")
    print(f"  {out_dir / 'experiment_02_raw_results.csv'}")
    print(f"  {out_dir / 'experiment_02_summary.csv'}")
    print(f"  {out_dir / 'experiment_02_metadata.json'}")


if __name__ == "__main__":
    main()
