#!/usr/bin/env python3
"""Duffing KAHKM tuning experiment for manuscript upgrade.

Purpose
-------
This script performs a systematic KAHKM fragmentation/sharpness tuning sweep for
the damped unforced Duffing oscillator. It is intended as the next manuscript
upgrade experiment after the Van der Pol tuning updates.

It tests whether the original Duffing KAHKM setting can be improved by tuning:
- number of regimes C,
- association sharpness omega.

Default grid:
- C in {8, 10, 12, 15, 20, 25, 30, 40}
- omega in {2, 4, 6, 8, 10, 12, 16, 20}

The old Duffing setting C=15, omega=12 is included in the grid.

Default selection objective:
- minimize mean relative association error over horizons {1, 10, 50, 100, 200}

Manuscript-compatible closure settings:
- beta = 0.1
- nlms_epochs = 20
- Nb = 100

Place this file in the same directory as:
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Pilot run:
    python3 run_exp_duffing_tuning.py --random-states 0

Manuscript-grade run:
    python3 run_exp_duffing_tuning.py

Resume interrupted run:
    python3 run_exp_duffing_tuning.py --resume

Outputs:
- kahkm_exp_duffing_tuning/experiment_duffing_tuning_raw.csv
- kahkm_exp_duffing_tuning/experiment_duffing_tuning_summary.csv
- kahkm_exp_duffing_tuning/experiment_duffing_selected_config_by_mean_horizon.csv
- kahkm_exp_duffing_tuning/experiment_duffing_best_by_horizon.csv
- kahkm_exp_duffing_tuning/experiment_duffing_top_selected_config_summary.csv
- kahkm_exp_duffing_tuning_results.zip
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
from typing import Any, Sequence, TypeAlias

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


@dataclass(frozen=True)
class RunKey:
    n_clusters: int
    omega: float
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    n_steps: int
    dt: float
    train_fraction: float
    clusters: tuple[int, ...]
    omegas: tuple[float, ...]
    random_states: tuple[int, ...]
    horizons: tuple[int, ...]
    selection_horizons: tuple[int, ...]
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
    process_noise: float
    output_dir: str
    resume: bool


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
    parser = argparse.ArgumentParser(description="Duffing KAHKM C/omega tuning.")

    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--process-noise", type=float, default=1e-5)

    parser.add_argument(
        "--clusters",
        nargs="+",
        default=["8", "10", "12", "15", "20", "25", "30", "40"],
        help="Cluster grid. Default: 8 10 12 15 20 25 30 40.",
    )
    parser.add_argument(
        "--omegas",
        nargs="+",
        default=["2", "4", "6", "8", "10", "12", "16", "20"],
        help="Omega grid. Default: 2 4 6 8 10 12 16 20.",
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
    parser.add_argument(
        "--selection-horizons",
        nargs="+",
        default=["1", "10", "50", "100", "200"],
        help="Horizons used for selection. Default: 1 10 50 100 200.",
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
        help="0 means no cap; otherwise passed to fit_kahkm.",
    )
    parser.add_argument("--output-dir", default="kahkm_exp_duffing_tuning")
    parser.add_argument("--resume", action="store_true")

    ns = parser.parse_args()

    train_fraction = float(ns.train_fraction)
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie strictly between 0 and 1.")

    horizons = tuple(sorted(set(parse_ints([str(value) for value in ns.horizons]))))
    selection_horizons = tuple(sorted(set(parse_ints([str(value) for value in ns.selection_horizons]))))
    missing = sorted(set(selection_horizons) - set(horizons))
    if missing:
        raise ValueError(
            "--selection-horizons must be included in --horizons. Missing: "
            + ", ".join(str(value) for value in missing)
        )

    max_train_per_cluster_raw = int(ns.max_train_per_cluster)
    max_train_per_cluster = None if max_train_per_cluster_raw <= 0 else max_train_per_cluster_raw

    return ExperimentConfig(
        n_steps=int(ns.n_steps),
        dt=float(ns.dt),
        train_fraction=train_fraction,
        clusters=tuple(sorted(set(parse_ints([str(value) for value in ns.clusters])))),
        omegas=tuple(sorted(set(parse_floats([str(value) for value in ns.omegas])))),
        random_states=parse_ints([str(value) for value in ns.random_states]),
        horizons=horizons,
        selection_horizons=selection_horizons,
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
        process_noise=float(ns.process_noise),
        output_dir=str(ns.output_dir),
        resume=bool(ns.resume),
    )


def make_duffing_snapshots(
    *,
    n_steps: int,
    dt: float,
    seed: int,
    process_noise: float,
) -> tuple[FloatArray, FloatArray]:
    """Unforced damped Duffing: qdot=p, pdot=-0.25p+q-q^3."""
    rng = np.random.default_rng(int(seed))

    def f(z: FloatArray) -> FloatArray:
        q = float(z[0])
        p = float(z[1])
        return np.array([p, -0.25 * p + q - q**3], dtype=np.float64)

    def rk4_step(z: FloatArray) -> FloatArray:
        k1 = f(z)
        k2 = f(z + 0.5 * float(dt) * k1)
        k3 = f(z + 0.5 * float(dt) * k2)
        k4 = f(z + float(dt) * k3)
        return np.asarray(
            z + (float(dt) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            dtype=np.float64,
        )

    # Mix initial conditions across the two wells and transition region.
    initial_pool = np.array(
        [
            [-1.6, 0.4],
            [-1.2, -0.6],
            [-0.4, 0.9],
            [0.3, -1.0],
            [1.1, 0.5],
            [1.6, -0.4],
        ],
        dtype=np.float64,
    )
    base = initial_pool[int(seed) % int(initial_pool.shape[0])]
    z = np.asarray(base + 0.05 * rng.normal(size=2), dtype=np.float64)

    states = np.empty((2, int(n_steps) + 1), dtype=np.float64)
    states[:, 0] = z
    for t in range(int(n_steps)):
        z = rk4_step(z) + float(process_noise) * rng.normal(size=2)
        states[:, t + 1] = z

    return states[:, :-1], states[:, 1:]


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
    X0_test: FloatArray,
    X1_test: FloatArray,
    horizons: Sequence[int],
    n_jobs: int,
    batch_size: int,
) -> list[CsvRow]:
    abstraction_model = getattr(fit_result, "abstraction_model")
    B = np.asarray(getattr(fit_result, "B"), dtype=np.float64)
    omega = float(getattr(fit_result, "omega"))
    tau = float(getattr(fit_result, "tau"))

    n_test = int(X0_test.shape[1])
    rows: list[CsvRow] = []

    for horizon in horizons:
        h = int(horizon)
        if h <= 0 or h > n_test:
            continue

        n_start = n_test - h + 1
        X_start = X0_test[:, :n_start]
        X_target = X1_test[:, h - 1 : h - 1 + n_start]

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
            {
                "horizon": h,
                "n_eval_starts": n_start,
                "relative_association_error": relative_error(pred, Phi_target),
                "association_r2": association_r2(pred, Phi_target),
                "simplex_violation": float(simplex_violation(pred)),
            }
        )

    return rows


def load_completed_keys(raw_path: Path) -> set[RunKey]:
    keys: set[RunKey] = set()
    for row in _read_csv(raw_path):
        keys.add(
            RunKey(
                n_clusters=_int_or_zero(_cell(row, "n_clusters")),
                omega=_float_or_nan(_cell(row, "omega")),
                seed=_int_or_zero(_cell(row, "seed")),
            )
        )
    return keys


def append_raw(path: Path, rows: Sequence[CsvRow]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize(raw_path: Path, output_dir: Path, selection_horizons: Sequence[int]) -> None:
    raw_rows = _read_csv(raw_path)

    grouped: dict[tuple[int, float, int], list[RawCsvRow]] = defaultdict(list)
    for row in raw_rows:
        grouped[
            (
                _int_or_zero(_cell(row, "n_clusters")),
                _float_or_nan(_cell(row, "omega")),
                _int_or_zero(_cell(row, "horizon")),
            )
        ].append(row)

    summary_rows: list[CsvRow] = []
    for (n_clusters, omega, horizon), rows in sorted(grouped.items()):
        errors = [_float_or_nan(_cell(row, "relative_association_error")) for row in rows]
        r2s = [_float_or_nan(_cell(row, "association_r2")) for row in rows]
        violations = [_float_or_nan(_cell(row, "simplex_violation")) for row in rows]
        train_errors = [_float_or_nan(_cell(row, "train_closure_error")) for row in rows]
        fit_seconds = [_float_or_nan(_cell(row, "fit_seconds")) for row in rows]

        errors = [value for value in errors if math.isfinite(value)]
        r2s = [value for value in r2s if math.isfinite(value)]
        violations = [value for value in violations if math.isfinite(value)]
        train_errors = [value for value in train_errors if math.isfinite(value)]
        fit_seconds = [value for value in fit_seconds if math.isfinite(value)]

        if not errors:
            continue

        summary_rows.append(
            {
                "n_clusters": n_clusters,
                "omega": omega,
                "horizon": horizon,
                "n_runs": len(errors),
                "relative_association_error_mean": mean(errors),
                "relative_association_error_std": pstdev(errors) if len(errors) > 1 else 0.0,
                "association_r2_mean": mean(r2s) if r2s else math.nan,
                "association_r2_std": pstdev(r2s) if len(r2s) > 1 else 0.0,
                "simplex_violation_mean": mean(violations) if violations else math.nan,
                "simplex_violation_max": max(violations) if violations else math.nan,
                "train_closure_error_mean": mean(train_errors) if train_errors else math.nan,
                "fit_seconds_mean": mean(fit_seconds) if fit_seconds else math.nan,
            }
        )

    _write_csv(output_dir / "experiment_duffing_tuning_summary.csv", summary_rows)

    # Selection by mean over requested horizons.
    by_config: dict[tuple[int, float], dict[int, RawCsvRow]] = defaultdict(dict)
    for row in _read_csv(output_dir / "experiment_duffing_tuning_summary.csv"):
        by_config[
            (_int_or_zero(_cell(row, "n_clusters")), _float_or_nan(_cell(row, "omega")))
        ][_int_or_zero(_cell(row, "horizon"))] = row

    selection_rows: list[CsvRow] = []
    wanted = tuple(int(h) for h in selection_horizons)

    for (n_clusters, omega), horizon_rows in sorted(by_config.items()):
        if not all(h in horizon_rows for h in wanted):
            continue
        errors = [
            _float_or_nan(_cell(horizon_rows[h], "relative_association_error_mean"))
            for h in wanted
        ]
        if not all(math.isfinite(value) for value in errors):
            continue

        selection_rows.append(
            {
                "rank": 0,
                "n_clusters": n_clusters,
                "omega": omega,
                "selection_objective_mean_error": mean(errors),
                "selection_horizons": " ".join(str(h) for h in wanted),
                "E1": _float_or_nan(_cell(horizon_rows.get(1, {}), "relative_association_error_mean")),
                "E10": _float_or_nan(_cell(horizon_rows.get(10, {}), "relative_association_error_mean")),
                "E50": _float_or_nan(_cell(horizon_rows.get(50, {}), "relative_association_error_mean")),
                "E100": _float_or_nan(_cell(horizon_rows.get(100, {}), "relative_association_error_mean")),
                "E200": _float_or_nan(_cell(horizon_rows.get(200, {}), "relative_association_error_mean")),
                "E200_std": _float_or_nan(_cell(horizon_rows.get(200, {}), "relative_association_error_std")),
                "train_closure_error_mean_h1row": _float_or_nan(_cell(horizon_rows.get(1, {}), "train_closure_error_mean")),
                "n_runs": _int_or_zero(_cell(horizon_rows.get(wanted[0], {}), "n_runs")),
            }
        )

    selection_rows.sort(key=lambda row: float(row["selection_objective_mean_error"]))
    ranked_rows: list[CsvRow] = []
    for rank, row in enumerate(selection_rows, start=1):
        updated = dict(row)
        updated["rank"] = rank
        ranked_rows.append(updated)

    _write_csv(output_dir / "experiment_duffing_selected_config_by_mean_horizon.csv", ranked_rows)
    _write_csv(output_dir / "experiment_duffing_top_selected_config_summary.csv", ranked_rows[:1])

    # Best per horizon.
    best_by_horizon: list[CsvRow] = []
    for horizon in sorted({int(row["horizon"]) for row in summary_rows}):
        rows = [row for row in summary_rows if int(row["horizon"]) == horizon]
        rows.sort(key=lambda row: float(row["relative_association_error_mean"]))
        if not rows:
            continue
        best = rows[0]
        best_by_horizon.append(
            {
                "horizon": horizon,
                "n_clusters": int(best["n_clusters"]),
                "omega": float(best["omega"]),
                "relative_association_error_mean": float(best["relative_association_error_mean"]),
                "relative_association_error_std": float(best["relative_association_error_std"]),
                "association_r2_mean": float(best["association_r2_mean"]),
                "association_r2_std": float(best["association_r2_std"]),
                "n_runs": int(best["n_runs"]),
            }
        )

    _write_csv(output_dir / "experiment_duffing_best_by_horizon.csv", best_by_horizon)


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

    raw_path = output_dir / "experiment_duffing_tuning_raw.csv"
    completed = load_completed_keys(raw_path) if config.resume else set()

    total_jobs = len(config.clusters) * len(config.omegas) * len(config.random_states)
    job_index = 0
    start_all = time.time()

    print("\nDuffing KAHKM tuning")
    print(f"Output directory: {output_dir}")
    print(f"Total grid jobs: {total_jobs}")
    print(f"Resume: {config.resume}; completed keys loaded: {len(completed)}")

    for n_clusters in config.clusters:
        for omega in config.omegas:
            for seed in config.random_states:
                job_index += 1
                key = RunKey(n_clusters=int(n_clusters), omega=float(omega), seed=int(seed))
                if key in completed:
                    print(
                        f"[{job_index}/{total_jobs}] skip completed "
                        f"C={n_clusters}, omega={omega}, seed={seed}"
                    )
                    continue

                print(
                    f"\n[{job_index}/{total_jobs}] fit "
                    f"C={n_clusters}, omega={omega}, seed={seed}"
                )

                X0, X1 = make_duffing_snapshots(
                    n_steps=int(config.n_steps),
                    dt=float(config.dt),
                    seed=int(seed),
                    process_noise=float(config.process_noise),
                )
                n_total = int(X0.shape[1])
                split = int(round(float(config.train_fraction) * n_total))
                if n_total - split < max(config.horizons):
                    raise ValueError("Not enough test samples for the requested maximum horizon.")

                X0_train = X0[:, :split]
                X1_train = X1[:, :split]
                X0_test = X0[:, split:]
                X1_test = X1[:, split:]

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
                    X0_test=X0_test,
                    X1_test=X1_test,
                    horizons=config.horizons,
                    n_jobs=int(config.n_jobs),
                    batch_size=int(config.batch_size),
                )

                csv_rows: list[CsvRow] = []
                for row in eval_rows:
                    csv_rows.append(
                        {
                            "n_clusters": int(n_clusters),
                            "omega": float(omega),
                            "seed": int(seed),
                            "horizon": int(row["horizon"]),
                            "n_eval_starts": int(row["n_eval_starts"]),
                            "relative_association_error": float(row["relative_association_error"]),
                            "association_r2": float(row["association_r2"]),
                            "simplex_violation": float(row["simplex_violation"]),
                            "train_closure_error": float(getattr(fit, "train_closure_error")),
                            "train_association_r2": float(getattr(fit, "association_r2")),
                            "fit_seconds": float(fit_seconds),
                        }
                    )

                append_raw(raw_path, csv_rows)
                summarize(raw_path, output_dir, config.selection_horizons)
                completed.add(key)

                e200 = next(
                    (
                        float(row["relative_association_error"])
                        for row in csv_rows
                        if int(row["horizon"]) == 200
                    ),
                    math.nan,
                )
                print(
                    f"  done: train_err={float(getattr(fit, 'train_closure_error')):.6g}, "
                    f"E200={e200:.6g}, fit_seconds={fit_seconds:.2f}"
                )

    metadata = {
        "experiment": "Duffing KAHKM C/omega tuning",
        "config": asdict(config),
        "old_default_reference": {"n_clusters": 15, "omega": 12.0},
        "total_jobs": total_jobs,
        "completed_jobs": len(completed),
        "total_seconds": time.time() - start_all,
    }
    with (output_dir / "experiment_duffing_tuning_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summarize(raw_path, output_dir, config.selection_horizons)
    zip_path = make_zip(output_dir, "kahkm_exp_duffing_tuning_results")

    print("\nFinished Duffing tuning.")
    print(f"Raw results: {raw_path.resolve()}")
    print(f"Selected config: {(output_dir / 'experiment_duffing_selected_config_by_mean_horizon.csv').resolve()}")
    print(f"Best by horizon: {(output_dir / 'experiment_duffing_best_by_horizon.csv').resolve()}")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload the ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
