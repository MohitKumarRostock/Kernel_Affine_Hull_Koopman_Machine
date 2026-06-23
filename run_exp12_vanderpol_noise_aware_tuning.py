#!/usr/bin/env python3
"""Noise-aware Van der Pol tuning for KAHKM.

Place this file in the same directory as:
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Then run the manuscript-grade sweep:

    python3 run_exp12_vanderpol_noise_aware_tuning.py

This performs a noise-aware Van der Pol grid over:
- C in {10, 15, 20, 25, 30, 40}
- omega in {0.25, 0.5, 1, 2, 4}
- noise in {0, 0.005, 0.01, 0.02, 0.05}
- seeds in {0, 1, 2}

The training snapshots are corrupted by coordinate-wise Gaussian observation noise:
    sigma_j = noise_level * std_j(training coordinate).

The held-out evaluation trajectory remains clean.

Default robust-selection objective:
    score = mean over seeds and noise levels of
            0.25 E50 + 0.25 E100 + 0.50 E200

Useful pilot run:
    python3 run_exp12_vanderpol_noise_aware_tuning.py \
      --clusters 10 15 20 \
      --omegas 0.25 0.5 1 \
      --noise-levels 0 0.02 0.05 \
      --random-states 0

Resume interrupted runs:
    python3 run_exp12_vanderpol_noise_aware_tuning.py --resume

Outputs:
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_noise_tuning_raw.csv
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_noise_tuning_summary_by_config_noise_horizon.csv
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_noise_tuning_robust_selection.csv
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_noise_tuning_best_by_noise.csv
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_noise_tuning_metadata.json
- kahkm_exp12_vanderpol_noise_aware_tuning_results.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kernel_affine_hull_koopman_machines import (  # noqa: E402
    fit_kahkm,
    kahm_associations,
    simplex_violation,
)


FloatArray: TypeAlias = NDArray[np.float64]
CsvCell: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvCell]
RawCsvRow: TypeAlias = dict[str, str]
KMeansKind: TypeAlias = Literal["auto", "full", "minibatch"]


@dataclass(frozen=True)
class RunConfig:
    n_clusters: int
    omega: float
    noise_level: float
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    n_steps: int
    dt: float
    mu: float
    train_fraction: float
    clusters: tuple[int, ...]
    omegas: tuple[float, ...]
    noise_levels: tuple[float, ...]
    random_states: tuple[int, ...]
    horizons: tuple[int, ...]
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    tau: float
    kmeans_kind: str
    kmeans_batch_size: int
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int | None
    output_dir: str
    selection_weights: dict[int, float]


def parse_ints(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def parse_floats(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one float is required.")
    return parsed


def parse_selection_weights(text: str) -> dict[int, float]:
    """Parse selection weights such as '50:0.25,100:0.25,200:0.5'."""
    out: dict[int, float] = {}
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise argparse.ArgumentTypeError(
                "Selection weights must have form horizon:weight,horizon:weight."
            )
        horizon_text, weight_text = token.split(":", 1)
        horizon = int(horizon_text.strip())
        weight = float(weight_text.strip())
        if horizon <= 0:
            raise argparse.ArgumentTypeError("Selection horizons must be positive.")
        if weight < 0.0:
            raise argparse.ArgumentTypeError("Selection weights must be nonnegative.")
        out[horizon] = weight
    if not out:
        raise argparse.ArgumentTypeError("At least one selection weight is required.")
    total = sum(out.values())
    if total <= 0.0:
        raise argparse.ArgumentTypeError("The sum of selection weights must be positive.")
    return {horizon: weight / total for horizon, weight in out.items()}


def _float_or_nan(value: str | int | float | None) -> float:
    if value is None:
        return math.nan
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return math.nan
        try:
            return float(text)
        except ValueError:
            return math.nan
    return float(value)


def _int_or_zero(value: str | int | float | None) -> int:
    number = _float_or_nan(value)
    if not math.isfinite(number):
        return 0
    return int(number)


def _read_csv(path: Path) -> list[RawCsvRow]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cell(row: RawCsvRow, key: str, default: str = "") -> str:
    return row[key] if key in row else default


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Noise-aware Van der Pol KAHKM tuning."
    )

    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.75)

    parser.add_argument(
        "--clusters",
        nargs="+",
        default=["10", "15", "20", "25", "30", "40"],
        help="Cluster grid. Default: 10 15 20 25 30 40.",
    )
    parser.add_argument(
        "--omegas",
        nargs="+",
        default=["0.25", "0.5", "1", "2", "4"],
        help="Omega grid. Default: 0.25 0.5 1 2 4.",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        default=["0", "0.005", "0.01", "0.02", "0.05"],
        help="Training-only noise levels. Default: 0 0.005 0.01 0.02 0.05.",
    )
    parser.add_argument(
        "--random-states",
        nargs="+",
        default=["0", "1", "2"],
        help="Random states. Default: 0 1 2.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["1", "10", "50", "100", "200"],
        help="Evaluation horizons. Default: 1 10 50 100 200.",
    )

    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--kmeans-kind", type=str, default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--max-train-per-cluster",
        type=int,
        default=0,
        help="0 means no per-cluster cap; otherwise passed to fit_kahkm.",
    )

    parser.add_argument(
        "--selection-weights",
        default="50:0.25,100:0.25,200:0.5",
        help="Weighted long-horizon objective. Default: 50:0.25,100:0.25,200:0.5.",
    )
    parser.add_argument(
        "--output-dir",
        default="kahkm_exp12_vanderpol_noise_aware_tuning",
    )

    ns = parser.parse_args()

    train_fraction = float(ns.train_fraction)
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie strictly between 0 and 1.")

    horizons = tuple(sorted(set(parse_ints([str(value) for value in ns.horizons]))))
    selection_weights = parse_selection_weights(str(ns.selection_weights))
    missing_horizons = sorted(set(selection_weights.keys()) - set(horizons))
    if missing_horizons:
        raise ValueError(
            "Every selection-weight horizon must also be included in --horizons. Missing: "
            + ", ".join(str(value) for value in missing_horizons)
        )

    max_train_per_cluster_raw = int(ns.max_train_per_cluster)
    max_train_per_cluster = None if max_train_per_cluster_raw <= 0 else max_train_per_cluster_raw

    return ExperimentConfig(
        n_steps=int(ns.n_steps),
        dt=float(ns.dt),
        mu=float(ns.mu),
        train_fraction=train_fraction,
        clusters=parse_ints([str(value) for value in ns.clusters]),
        omegas=parse_floats([str(value) for value in ns.omegas]),
        noise_levels=parse_floats([str(value) for value in ns.noise_levels]),
        random_states=parse_ints([str(value) for value in ns.random_states]),
        horizons=horizons,
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        tau=float(ns.tau),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        max_train_per_cluster=max_train_per_cluster,
        output_dir=str(ns.output_dir),
        selection_weights=selection_weights,
    )


def make_vanderpol_snapshots(
    *,
    n_steps: int,
    dt: float,
    seed: int,
    mu: float,
) -> tuple[FloatArray, FloatArray]:
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
        return np.asarray(
            z + (float(dt) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            dtype=np.float64,
        )

    z = np.array([2.0, 0.0], dtype=np.float64) + 0.05 * rng.normal(size=2)
    states = np.empty((2, int(n_steps) + 1), dtype=np.float64)
    states[:, 0] = z
    for t in range(int(n_steps)):
        z = rk4_step(z) + 1e-5 * rng.normal(size=2)
        states[:, t + 1] = z

    return states[:, :-1], states[:, 1:]


def add_coordinate_noise(
    X: FloatArray,
    *,
    noise_level: float,
    rng: np.random.Generator,
) -> FloatArray:
    X_arr = np.asarray(X, dtype=np.float64)
    level = float(noise_level)
    if level <= 0.0:
        return np.array(X_arr, dtype=np.float64, copy=True)

    scale = np.std(X_arr, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    return np.asarray(
        X_arr + rng.normal(loc=0.0, scale=level * scale, size=X_arr.shape),
        dtype=np.float64,
    )


def relative_error(pred: FloatArray, target: FloatArray, eps: float = 1e-12) -> float:
    numerator = float(np.sum((target - pred) ** 2))
    denominator = float(np.sum(target * target))
    return numerator / max(denominator, eps)


def association_r2(pred: FloatArray, target: FloatArray) -> float:
    residual_ss = float(np.sum((target - pred) ** 2))
    total_ss = float(np.sum((target - target.mean(axis=1, keepdims=True)) ** 2))
    if total_ss <= 0.0:
        return math.nan
    return 1.0 - residual_ss / total_ss


def rollout(B: FloatArray, Phi0: FloatArray, horizon: int) -> FloatArray:
    pred = np.array(Phi0, dtype=np.float64, copy=True)
    BT = np.asarray(B.T, dtype=np.float64)
    for _ in range(int(horizon)):
        pred = np.asarray(BT @ pred, dtype=np.float64)
    return pred


def evaluate_horizons(
    *,
    fit_result: Any,
    X0_test_clean: FloatArray,
    X1_test_clean: FloatArray,
    horizons: Sequence[int],
    n_jobs: int,
    batch_size: int,
) -> list[tuple[int, int, float, float, float]]:
    abstraction_model = getattr(fit_result, "abstraction_model")
    B = np.asarray(getattr(fit_result, "B"), dtype=np.float64)
    omega = float(getattr(fit_result, "omega"))
    tau = float(getattr(fit_result, "tau"))

    n_test = int(X0_test_clean.shape[1])
    rows: list[tuple[int, int, float, float, float]] = []

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

        pred = rollout(B, Phi_start, h)
        rows.append(
            (
                h,
                int(n_start),
                relative_error(pred, Phi_target),
                association_r2(pred, Phi_target),
                float(simplex_violation(pred)),
            )
        )

    return rows


def load_completed_keys(raw_path: Path) -> set[RunConfig]:
    completed: set[RunConfig] = set()
    for row in _read_csv(raw_path):
        completed.add(
            RunConfig(
                n_clusters=_int_or_zero(_cell(row, "n_clusters")),
                omega=_float_or_nan(_cell(row, "omega")),
                noise_level=_float_or_nan(_cell(row, "noise_level")),
                seed=_int_or_zero(_cell(row, "seed")),
            )
        )
    return completed


def append_raw_rows(path: Path, rows: Sequence[CsvRow]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize(raw_path: Path, output_dir: Path, selection_weights: dict[int, float]) -> None:
    raw = _read_csv(raw_path)

    groups: dict[tuple[int, float, float, int], list[RawCsvRow]] = defaultdict(list)
    for row in raw:
        key = (
            _int_or_zero(_cell(row, "n_clusters")),
            _float_or_nan(_cell(row, "omega")),
            _float_or_nan(_cell(row, "noise_level")),
            _int_or_zero(_cell(row, "horizon")),
        )
        groups[key].append(row)

    summary_rows: list[CsvRow] = []
    for (n_clusters, omega, noise_level, horizon), rows in sorted(groups.items()):
        errors = [_float_or_nan(_cell(row, "relative_association_error")) for row in rows]
        r2s = [_float_or_nan(_cell(row, "association_r2")) for row in rows]
        violations = [_float_or_nan(_cell(row, "simplex_violation")) for row in rows]
        fit_seconds = [_float_or_nan(_cell(row, "fit_seconds")) for row in rows]

        errors = [value for value in errors if math.isfinite(value)]
        r2s = [value for value in r2s if math.isfinite(value)]
        violations = [value for value in violations if math.isfinite(value)]
        fit_seconds = [value for value in fit_seconds if math.isfinite(value)]

        if not errors:
            continue

        summary_rows.append(
            {
                "n_clusters": n_clusters,
                "omega": omega,
                "noise_level": noise_level,
                "horizon": horizon,
                "n_runs": len(errors),
                "relative_association_error_mean": mean(errors),
                "relative_association_error_std": pstdev(errors) if len(errors) > 1 else 0.0,
                "relative_association_error_min": min(errors),
                "relative_association_error_max": max(errors),
                "association_r2_mean": mean(r2s) if r2s else math.nan,
                "association_r2_std": pstdev(r2s) if len(r2s) > 1 else 0.0,
                "simplex_violation_mean": mean(violations) if violations else math.nan,
                "simplex_violation_max": max(violations) if violations else math.nan,
                "fit_seconds_mean": mean(fit_seconds) if fit_seconds else math.nan,
            }
        )

    _write_csv(output_dir / "experiment_12_noise_tuning_summary_by_config_noise_horizon.csv", summary_rows)

    # Weighted robust objective by configuration.
    by_config_noise: dict[tuple[int, float, float], dict[int, float]] = defaultdict(dict)
    for row in summary_rows:
        n_clusters = int(row["n_clusters"])
        omega = float(row["omega"])
        noise_level = float(row["noise_level"])
        horizon = int(row["horizon"])
        error = float(row["relative_association_error_mean"])
        by_config_noise[(n_clusters, omega, noise_level)][horizon] = error

    config_scores: dict[tuple[int, float], list[float]] = defaultdict(list)
    config_noise_scores: list[CsvRow] = []

    for (n_clusters, omega, noise_level), horizon_error in sorted(by_config_noise.items()):
        if not all(horizon in horizon_error for horizon in selection_weights):
            continue
        score = sum(selection_weights[horizon] * horizon_error[horizon] for horizon in selection_weights)
        config_scores[(n_clusters, omega)].append(score)
        config_noise_scores.append(
            {
                "n_clusters": n_clusters,
                "omega": omega,
                "noise_level": noise_level,
                "weighted_long_horizon_score": score,
                "E50": horizon_error.get(50, math.nan),
                "E100": horizon_error.get(100, math.nan),
                "E200": horizon_error.get(200, math.nan),
            }
        )

    _write_csv(output_dir / "experiment_12_noise_tuning_config_noise_scores.csv", config_noise_scores)

    selection_rows: list[CsvRow] = []
    for (n_clusters, omega), scores in sorted(config_scores.items()):
        if not scores:
            continue

        # Collect useful E200 statistics over noise levels.
        e200_values: list[float] = []
        e50_values: list[float] = []
        e100_values: list[float] = []
        for (config_c, config_omega, _noise), horizon_error in by_config_noise.items():
            if config_c == n_clusters and abs(config_omega - omega) < 1e-12:
                if 50 in horizon_error:
                    e50_values.append(horizon_error[50])
                if 100 in horizon_error:
                    e100_values.append(horizon_error[100])
                if 200 in horizon_error:
                    e200_values.append(horizon_error[200])

        selection_rows.append(
            {
                "rank": 0,
                "n_clusters": n_clusters,
                "omega": omega,
                "robust_weighted_score_mean_over_noise": mean(scores),
                "robust_weighted_score_std_over_noise": pstdev(scores) if len(scores) > 1 else 0.0,
                "E50_mean_over_noise": mean(e50_values) if e50_values else math.nan,
                "E100_mean_over_noise": mean(e100_values) if e100_values else math.nan,
                "E200_mean_over_noise": mean(e200_values) if e200_values else math.nan,
                "E200_max_over_noise": max(e200_values) if e200_values else math.nan,
                "n_noise_levels": len(scores),
                "selection_weights": ",".join(f"{h}:{w:.6g}" for h, w in sorted(selection_weights.items())),
            }
        )

    selection_rows.sort(key=lambda row: float(row["robust_weighted_score_mean_over_noise"]))
    ranked_selection: list[CsvRow] = []
    for rank, row in enumerate(selection_rows, start=1):
        updated = dict(row)
        updated["rank"] = rank
        ranked_selection.append(updated)

    _write_csv(output_dir / "experiment_12_noise_tuning_robust_selection.csv", ranked_selection)

    # Best configuration by noise level.
    best_by_noise_rows: list[CsvRow] = []
    by_noise: dict[float, list[CsvRow]] = defaultdict(list)
    for row in config_noise_scores:
        by_noise[float(row["noise_level"])].append(row)

    for noise_level, rows in sorted(by_noise.items()):
        rows_sorted = sorted(rows, key=lambda row: float(row["weighted_long_horizon_score"]))
        best = rows_sorted[0]
        best_by_noise_rows.append(
            {
                "noise_level": noise_level,
                "n_clusters": best["n_clusters"],
                "omega": best["omega"],
                "weighted_long_horizon_score": best["weighted_long_horizon_score"],
                "E50": best["E50"],
                "E100": best["E100"],
                "E200": best["E200"],
            }
        )

    _write_csv(output_dir / "experiment_12_noise_tuning_best_by_noise.csv", best_by_noise_rows)


def make_zip(output_dir: Path, zip_name: str) -> Path:
    zip_base = output_dir.parent / zip_name
    zip_path = Path(str(zip_base) + ".zip")
    if zip_path.exists():
        zip_path.unlink()

    staging_dir = output_dir.parent / f"{zip_name}_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(output_dir, staging_dir / output_dir.name)
    shutil.make_archive(str(zip_base), "zip", root_dir=staging_dir)
    shutil.rmtree(staging_dir)
    return zip_path


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "experiment_12_noise_tuning_raw.csv"
    completed = load_completed_keys(raw_path)

    total_jobs = (
        len(config.clusters)
        * len(config.omegas)
        * len(config.noise_levels)
        * len(config.random_states)
    )
    job_index = 0
    start_all = time.time()

    print("\nNoise-aware Van der Pol tuning")
    print(f"Output directory: {output_dir}")
    print(f"Total grid jobs: {total_jobs}")
    print(f"Already completed jobs in raw CSV: {len(completed)}")

    for n_clusters in config.clusters:
        for omega in config.omegas:
            for noise_level in config.noise_levels:
                for seed in config.random_states:
                    job_index += 1
                    run_key = RunConfig(
                        n_clusters=int(n_clusters),
                        omega=float(omega),
                        noise_level=float(noise_level),
                        seed=int(seed),
                    )
                    if run_key in completed:
                        print(
                            f"[{job_index}/{total_jobs}] skip completed "
                            f"C={n_clusters}, omega={omega}, noise={noise_level}, seed={seed}"
                        )
                        continue

                    print(
                        f"\n[{job_index}/{total_jobs}] fit "
                        f"C={n_clusters}, omega={omega}, noise={noise_level}, seed={seed}"
                    )

                    X0, X1 = make_vanderpol_snapshots(
                        n_steps=config.n_steps,
                        dt=config.dt,
                        seed=int(seed),
                        mu=config.mu,
                    )
                    n_total = int(X0.shape[1])
                    split = int(round(float(config.train_fraction) * n_total))
                    if n_total - split < max(config.horizons):
                        raise ValueError("Not enough clean test samples for the requested max horizon.")

                    X0_train_clean = X0[:, :split]
                    X1_train_clean = X1[:, :split]
                    X0_test_clean = X0[:, split:]
                    X1_test_clean = X1[:, split:]

                    noise_seed = (
                        1_000_003 * int(seed)
                        + 10_007 * int(n_clusters)
                        + int(round(1_000_000.0 * float(noise_level)))
                        + int(round(10_000.0 * float(omega)))
                        + 17
                    )
                    rng = np.random.default_rng(noise_seed)
                    X0_train = add_coordinate_noise(
                        X0_train_clean,
                        noise_level=float(noise_level),
                        rng=rng,
                    )
                    X1_train = add_coordinate_noise(
                        X1_train_clean,
                        noise_level=float(noise_level),
                        rng=rng,
                    )

                    fit_start = time.time()
                    fit = fit_kahkm(
                        X0_train,
                        X1_train,
                        n_clusters=int(n_clusters),
                        subspace_dim=int(config.subspace_dim),
                        Nb=int(config.nb),
                        omega=float(omega),
                        tau=float(config.tau),
                        beta=float(config.beta),
                        nlms_epochs=int(config.nlms_epochs),
                        random_state=int(seed),
                        kmeans_kind=config.kmeans_kind,  # type: ignore[arg-type]
                        kmeans_batch_size=int(config.kmeans_batch_size),
                        max_train_per_cluster=config.max_train_per_cluster,
                        save_ae_to_disk=False,
                        n_jobs=int(config.n_jobs),
                        batch_size=int(config.batch_size),
                        project_stochastic=False,
                        preload_classifier_after_fit=False,
                        verbose=False,
                    )
                    fit_seconds = time.time() - fit_start

                    eval_rows = evaluate_horizons(
                        fit_result=fit,
                        X0_test_clean=X0_test_clean,
                        X1_test_clean=X1_test_clean,
                        horizons=config.horizons,
                        n_jobs=int(config.n_jobs),
                        batch_size=int(config.batch_size),
                    )

                    csv_rows: list[CsvRow] = []
                    for horizon, n_eval_starts, error, r2, violation in eval_rows:
                        csv_rows.append(
                            {
                                "n_clusters": int(n_clusters),
                                "omega": float(omega),
                                "noise_level": float(noise_level),
                                "seed": int(seed),
                                "horizon": int(horizon),
                                "n_eval_starts": int(n_eval_starts),
                                "relative_association_error": float(error),
                                "association_r2": float(r2),
                                "simplex_violation": float(violation),
                                "train_closure_error_noisy_train": float(getattr(fit, "train_closure_error")),
                                "train_association_r2_noisy_train": float(getattr(fit, "association_r2")),
                                "fit_seconds": float(fit_seconds),
                            }
                        )

                    append_raw_rows(raw_path, csv_rows)
                    completed.add(run_key)

                    e200 = next((row["relative_association_error"] for row in csv_rows if row["horizon"] == 200), math.nan)
                    print(
                        f"  done: train_err={float(getattr(fit, 'train_closure_error')):.6g}, "
                        f"E200={float(e200):.6g}, fit_seconds={fit_seconds:.2f}"
                    )

                    # Refresh summaries after every completed fit, so interrupted runs are usable.
                    summarize(raw_path, output_dir, config.selection_weights)

    metadata = {
        "experiment": "noise-aware Van der Pol KAHKM tuning",
        "config": {
            key: value
            for key, value in asdict(config).items()
            if key != "selection_weights"
        },
        "selection_weights": config.selection_weights,
        "noise_convention": (
            "training-only coordinate-wise Gaussian observation noise; "
            "sigma_j = noise_level * std_j(training coordinate); clean held-out evaluation"
        ),
        "total_jobs": total_jobs,
        "completed_jobs": len(completed),
        "total_seconds": time.time() - start_all,
    }
    with (output_dir / "experiment_12_noise_tuning_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summarize(raw_path, output_dir, config.selection_weights)
    zip_path = make_zip(output_dir, "kahkm_exp12_vanderpol_noise_aware_tuning_results")

    print("\nFinished noise-aware Van der Pol tuning.")
    print(f"Raw results: {raw_path.resolve()}")
    print(f"Robust selection: {(output_dir / 'experiment_12_noise_tuning_robust_selection.csv').resolve()}")
    print(f"Best by noise: {(output_dir / 'experiment_12_noise_tuning_best_by_noise.csv').resolve()}")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload the ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
