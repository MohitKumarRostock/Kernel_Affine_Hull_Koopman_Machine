"""Experiment 10: tuned Van der Pol abstraction ablation for KAHKM.

Purpose
-------
This experiment repeats the Van der Pol abstraction ablation using the
Van-der-Pol-tuned KAHKM hyperparameters selected by Experiment 09. It checks
whether the improvement from C=20, omega=4 remains when KAHKM is compared
against KMeans-style abstractions under the same abstraction granularity and
NLMS closure estimator.

Compared abstractions
---------------------
1. kahkm_folding_nlms
   The actual KAHKM association map from kernel_affine_hull_koopman_machines.py.

2. kmeans_hard_nlms
   Hard KMeans one-hot regime labels.

3. kmeans_distance_nlms
   A bounded prototype-distance analogue using the manuscript normalization:
       psi_c(x) = (1 - T_c(x) + tau)^omega / sum_l (...),
   where T_c is clipped normalized distance to centroid c.

4. kmeans_rbf_nlms
   Standard RBF/softmax associations to KMeans centroids.

All abstractions are evaluated with the same NLMS closure recursion and the same
one-step and multi-step association-space diagnostics.

Expected local files
--------------------
Place this script in the same folder as:

- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Run
---
python experiment_10_vanderpol_tuned_ablation_pylance_clean.py

Fast smoke test:
python experiment_10_vanderpol_tuned_ablation_pylance_clean.py --random-states 0 --n-steps 600 --horizons 1 2 5 10 20
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
from typing import Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
    fit_kahkm,
    kahm_associations,
    simplex_violation,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]
AbstractionName: TypeAlias = Literal[
    "kahkm_folding_nlms",
    "kmeans_hard_nlms",
    "kmeans_distance_nlms",
    "kmeans_rbf_nlms",
]


def make_vanderpol_snapshots(n_steps: int = 1600, dt: float = 0.02, seed: int = 0, mu: float = 1.0) -> tuple[FloatArray, FloatArray]:
    """Generate snapshot pairs from the Van der Pol oscillator.

    State z = [q, p]. The continuous-time dynamics are
        q' = p,
        p' = mu * (1 - q^2) * p - q,
    integrated by RK4 with tiny process noise to avoid duplicate samples.
    """
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

    # Start away from the limit cycle to include transient and cyclic behavior.
    z = np.array([2.0, 0.0], dtype=np.float64) + 0.05 * rng.normal(size=2)
    states = np.empty((2, int(n_steps) + 1), dtype=np.float64)
    states[:, 0] = z
    for t in range(int(n_steps)):
        z = rk4_step(z) + 1e-5 * rng.normal(size=2)
        states[:, t + 1] = z
    return states[:, :-1], states[:, 1:]


@dataclass(frozen=True)
class ExperimentArgs:
    n_steps: int
    dt: float
    mu: float
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
    output_dir: str
    save_ae_to_disk: bool
    max_train_per_cluster: int | None
    rbf_sigma_multiplier: float
    distance_scale_percentile: float


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
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


def _pairwise_squared_distances(X: FloatArray, centers: FloatArray) -> FloatArray:
    """Return distances shaped (C, N) for X shaped (D, N), centers shaped (D, C)."""
    x2 = np.sum(X * X, axis=0, keepdims=True)
    c2 = np.sum(centers * centers, axis=0, keepdims=True).T
    d2 = c2 + x2 - 2.0 * (centers.T @ X)
    return np.maximum(np.asarray(d2, dtype=np.float64), 0.0)


def _normalize_columns(P: FloatArray, eps: float = 1e-300) -> FloatArray:
    col_sums = np.sum(P, axis=0, keepdims=True)
    return P / np.maximum(col_sums, eps)


def _hard_kmeans_associations(X: FloatArray, centers: FloatArray) -> FloatArray:
    d2 = _pairwise_squared_distances(X, centers)
    labels = np.argmin(d2, axis=0)
    P = np.zeros((centers.shape[1], X.shape[1]), dtype=np.float64)
    P[labels, np.arange(X.shape[1])] = 1.0
    return P


def _distance_kmeans_associations(
    X: FloatArray,
    centers: FloatArray,
    *,
    scale: float,
    omega: float,
    tau: float,
) -> FloatArray:
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("Distance scale must be finite and positive.")
    distances = np.sqrt(_pairwise_squared_distances(X, centers))
    T = np.clip(distances / float(scale), 0.0, 1.0)
    affinity = np.power(1.0 - T + float(tau), float(omega))
    return _normalize_columns(np.asarray(affinity, dtype=np.float64))


def _rbf_kmeans_associations(
    X: FloatArray,
    centers: FloatArray,
    *,
    sigma: float,
) -> FloatArray:
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("RBF sigma must be finite and positive.")
    d2 = _pairwise_squared_distances(X, centers)
    logits = -0.5 * d2 / float(sigma * sigma)
    logits = logits - np.max(logits, axis=0, keepdims=True)
    weights = np.exp(logits)
    return _normalize_columns(np.asarray(weights, dtype=np.float64))


def _fit_baseline_kmeans(X_train: FloatArray, n_clusters: int, seed: int) -> tuple[FloatArray, IntArray, float, float]:
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init="auto")
    kmeans.fit(np.ascontiguousarray(X_train.T))
    centers = np.asarray(kmeans.cluster_centers_.T, dtype=np.float64)

    d2_train = _pairwise_squared_distances(X_train, centers)
    nearest_distances = np.sqrt(np.min(d2_train, axis=0))
    positive = nearest_distances[nearest_distances > 1e-12]
    if positive.size == 0:
        base_sigma = 1.0
        distance_scale = 1.0
    else:
        base_sigma = float(np.median(positive))
        distance_scale = float(np.percentile(positive, 95.0))
        base_sigma = max(base_sigma, 1e-12)
        distance_scale = max(distance_scale, 1e-12)
    return centers, np.asarray(kmeans.labels_, dtype=np.int64), base_sigma, distance_scale


def _nlms_closure(Phi: FloatArray, Chi: FloatArray, *, beta: float, epochs: int) -> tuple[FloatArray, tuple[float, ...]]:
    """Learn B for Chi ~= B.T @ Phi using the manuscript-style NLMS recursion."""
    if Phi.shape != Chi.shape:
        raise ValueError("Phi and Chi must have the same shape.")
    if beta <= 0.0 or beta >= 1.0 or not math.isfinite(beta):
        raise ValueError("beta must satisfy 0 < beta < 1.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    c_count, n_samples = Phi.shape
    B = np.eye(c_count, dtype=np.float64)
    history: list[float] = []
    for _epoch in range(int(epochs)):
        for sample_idx in range(int(n_samples)):
            phi = Phi[:, sample_idx]
            chi = Chi[:, sample_idx]
            residual = chi - B.T @ phi
            denom = 1.0 + float(beta) * float(phi @ phi)
            B += (float(beta) / denom) * np.outer(phi, residual)
        pred = B.T @ Phi
        history.append(_relative_error(pred, Chi))
    return B, tuple(history)


def _one_step_row(abstraction: str, B: FloatArray, Phi: FloatArray, Chi: FloatArray, split_name: str) -> CsvRow:
    pred = B.T @ Phi
    return {
        "abstraction": abstraction,
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
    abstraction: str,
    B: FloatArray,
    Psi_states: FloatArray,
    *,
    start_min: int,
    start_max_exclusive: int,
    horizon: int,
) -> CsvRow:
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be positive.")
    max_start_allowed = min(int(start_max_exclusive), int(Psi_states.shape[1]) - h)
    if max_start_allowed <= int(start_min):
        return {
            "abstraction": abstraction,
            "horizon": h,
            "n_starts": 0,
            "relative_association_error": float("nan"),
            "association_r2": float("nan"),
            "simplex_violation": float("nan"),
        }
    starts = np.arange(int(start_min), max_start_allowed, dtype=np.int64)
    current = Psi_states[:, starts]
    target = Psi_states[:, starts + h]
    pred = np.linalg.matrix_power(B.T, h) @ current
    return {
        "abstraction": abstraction,
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


def _summarize_one_step(rows: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[tuple[str, str], list[CsvRow]] = {}
    for row in rows:
        key = (str(row["abstraction"]), str(row["split"]))
        groups.setdefault(key, []).append(row)

    summary: list[CsvRow] = []
    for (abstraction, split), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        err = np.array([float(r["closure_error"]) for r in group_rows], dtype=np.float64)
        r2 = np.array([float(r["association_r2"]) for r in group_rows], dtype=np.float64)
        violation = np.array([float(r["simplex_violation"]) for r in group_rows], dtype=np.float64)
        rho = np.array([float(r["spectral_radius"]) for r in group_rows], dtype=np.float64)
        summary.append(
            {
                "abstraction": abstraction,
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


def _summarize_multistep(rows: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[tuple[str, int], list[CsvRow]] = {}
    for row in rows:
        key = (str(row["abstraction"]), int(row["horizon"]))
        groups.setdefault(key, []).append(row)

    summary: list[CsvRow] = []
    for (abstraction, horizon), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        err = np.array([float(r["relative_association_error"]) for r in group_rows], dtype=np.float64)
        r2 = np.array([float(r["association_r2"]) for r in group_rows], dtype=np.float64)
        violation = np.array([float(r["simplex_violation"]) for r in group_rows], dtype=np.float64)
        summary.append(
            {
                "abstraction": abstraction,
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
        X0, X1 = make_vanderpol_snapshots(n_steps=args.n_steps, dt=args.dt, seed=seed, mu=args.mu)
        X0 = np.asarray(X0, dtype=np.float64)
        X1 = np.asarray(X1, dtype=np.float64)
        n_pairs = int(X0.shape[1])
        split = int(args.train_fraction * n_pairs)
        if split <= 1 or split >= n_pairs - 1:
            raise ValueError("train_fraction gives an invalid train/test split.")

        X0_train = X0[:, :split]
        X1_train = X1[:, :split]
        states = np.concatenate([X0[:, [0]], X1], axis=1)

        print("Training actual KAHKM abstraction...")
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
            project_stochastic=False,
            verbose=True,
        )
        kahkm_states = kahm_associations(
            fit.abstraction_model,
            states,
            omega=fit.omega,
            tau=fit.tau,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            show_progress=False,
        )

        print("Training KMeans baseline abstractions...")
        centers, _labels, base_sigma, base_distance_scale = _fit_baseline_kmeans(X0_train, args.n_clusters, seed)
        sigma = max(float(args.rbf_sigma_multiplier) * base_sigma, 1e-12)
        distance_scale = base_distance_scale
        if args.distance_scale_percentile != 95.0:
            nearest = np.sqrt(np.min(_pairwise_squared_distances(X0_train, centers), axis=0))
            positive = nearest[nearest > 1e-12]
            if positive.size > 0:
                distance_scale = max(float(np.percentile(positive, float(args.distance_scale_percentile))), 1e-12)

        abstractions: dict[str, tuple[FloatArray, FloatArray, tuple[float, ...]]] = {}

        # KAHKM uses the closure learned by the implementation.
        abstractions["kahkm_folding_nlms"] = (kahkm_states, np.asarray(fit.B, dtype=np.float64), tuple(fit.nlms_history))

        hard_states = _hard_kmeans_associations(states, centers)
        Phi_hard = hard_states[:, :split]
        Chi_hard = hard_states[:, 1 : split + 1]
        B_hard, hist_hard = _nlms_closure(Phi_hard, Chi_hard, beta=args.beta, epochs=args.nlms_epochs)
        abstractions["kmeans_hard_nlms"] = (hard_states, B_hard, hist_hard)

        distance_states = _distance_kmeans_associations(
            states,
            centers,
            scale=distance_scale,
            omega=args.omega,
            tau=args.tau,
        )
        Phi_distance = distance_states[:, :split]
        Chi_distance = distance_states[:, 1 : split + 1]
        B_distance, hist_distance = _nlms_closure(Phi_distance, Chi_distance, beta=args.beta, epochs=args.nlms_epochs)
        abstractions["kmeans_distance_nlms"] = (distance_states, B_distance, hist_distance)

        rbf_states = _rbf_kmeans_associations(states, centers, sigma=sigma)
        Phi_rbf = rbf_states[:, :split]
        Chi_rbf = rbf_states[:, 1 : split + 1]
        B_rbf, hist_rbf = _nlms_closure(Phi_rbf, Chi_rbf, beta=args.beta, epochs=args.nlms_epochs)
        abstractions["kmeans_rbf_nlms"] = (rbf_states, B_rbf, hist_rbf)

        for abstraction_name, (Psi_states, B, history) in abstractions.items():
            Phi_train = Psi_states[:, :split]
            Chi_train = Psi_states[:, 1 : split + 1]
            Phi_test = Psi_states[:, split:n_pairs]
            Chi_test = Psi_states[:, split + 1 : n_pairs + 1]

            train_row = _one_step_row(abstraction_name, B, Phi_train, Chi_train, "train")
            test_row = _one_step_row(abstraction_name, B, Phi_test, Chi_test, "test")
            train_row["seed"] = int(seed)
            test_row["seed"] = int(seed)
            one_step_rows.extend([train_row, test_row])

            for horizon in args.horizons:
                ms_row = _multistep_row(
                    abstraction_name,
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
                    "abstraction": abstraction_name,
                    "n_pairs": n_pairs,
                    "n_train": split,
                    "n_test": n_pairs - split,
                    "effective_regimes": int(args.n_clusters),
                    "final_train_error": float(history[-1]) if history else float("nan"),
                    "monotone_history": int(all(history[i] <= history[i - 1] + 1e-15 for i in range(1, len(history)))),
                    "sigma_rbf": float(sigma),
                    "distance_scale": float(distance_scale),
                    "seed_seconds": float(time.time() - seed_start),
                }
            )

        print(f"Seed {seed} finished in {time.time() - seed_start:.2f} seconds.")

    one_step_summary = _summarize_one_step(one_step_rows)
    multistep_summary = _summarize_multistep(multistep_rows)

    _write_csv(output_dir / "experiment_10_one_step_raw.csv", one_step_rows)
    _write_csv(output_dir / "experiment_10_one_step_summary.csv", one_step_summary)
    _write_csv(output_dir / "experiment_10_multistep_raw.csv", multistep_rows)
    _write_csv(output_dir / "experiment_10_multistep_summary.csv", multistep_summary)
    _write_csv(output_dir / "experiment_10_run_diagnostics.csv", run_rows)

    metadata = asdict(args)
    metadata["total_seconds"] = float(time.time() - start_all)
    metadata["notes"] = (
        "Tuned Van der Pol abstraction ablation: actual KAHKM folding versus KMeans hard, "
        "bounded distance-normalized, and RBF-soft simplex associations using the "
        "C=20, omega=4 setting selected by Experiment 09. The closure estimator is "
        "NLMS for every abstraction."
    )
    with (output_dir / "experiment_10_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nWrote outputs to:")
    for name in [
        "experiment_10_one_step_summary.csv",
        "experiment_10_multistep_summary.csv",
        "experiment_10_run_diagnostics.csv",
        "experiment_10_metadata.json",
    ]:
        print(f"  {output_dir / name}")

    print("\nOne-step test summary preview:")
    for row in one_step_summary:
        if row["split"] == "test":
            print(
                f"  {row['abstraction']}: "
                f"err={float(row['closure_error_mean']):.6g}, "
                f"r2={float(row['association_r2_mean']):.6g}, "
                f"rho={float(row['spectral_radius_mean']):.6g}"
            )


def parse_args(argv: Sequence[str] | None = None) -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Experiment 10: tuned Van der Pol abstraction ablation for KAHKM.")
    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--random-states", nargs="+", default=["0", "1", "2"], help="Random seeds.")
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--omega", type=float, default=4.0)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", type=str, default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--horizons", nargs="+", default=["1", "2", "5", "10", "20", "50", "100", "200"])
    parser.add_argument("--output-dir", type=str, default="kahkm_experiment_10_outputs")
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--max-train-per-cluster", type=int, default=None)
    parser.add_argument("--rbf-sigma-multiplier", type=float, default=1.0)
    parser.add_argument("--distance-scale-percentile", type=float, default=95.0)
    ns = parser.parse_args(argv)

    if not (0.0 < float(ns.train_fraction) < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")
    if float(ns.rbf_sigma_multiplier) <= 0.0:
        raise ValueError("--rbf-sigma-multiplier must be positive.")
    if not (0.0 < float(ns.distance_scale_percentile) <= 100.0):
        raise ValueError("--distance-scale-percentile must be in (0, 100].")

    return ExperimentArgs(
        n_steps=int(ns.n_steps),
        dt=float(ns.dt),
        mu=float(ns.mu),
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
        output_dir=str(ns.output_dir),
        save_ae_to_disk=bool(ns.save_ae_to_disk),
        max_train_per_cluster=None if ns.max_train_per_cluster is None else int(ns.max_train_per_cluster),
        rbf_sigma_multiplier=float(ns.rbf_sigma_multiplier),
        distance_scale_percentile=float(ns.distance_scale_percentile),
    )


if __name__ == "__main__":
    run_experiment(parse_args())
