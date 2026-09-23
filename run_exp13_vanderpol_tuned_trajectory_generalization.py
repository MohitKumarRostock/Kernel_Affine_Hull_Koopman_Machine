#!/usr/bin/env python3
"""Van der Pol trajectory-generalization upgrade experiment.

Purpose
-------
This script reruns the Van der Pol trajectory/data-size generalization diagnostic
under tuned KAHKM settings. It is designed as the next manuscript-upgrade
experiment after the noise-aware Van der Pol tuning.

It compares three KAHKM configurations by default:

1. robust_tuned:
      C = 25, omega = 4
   selected by the centered noise-aware Van der Pol tuning.

2. clean_tuned:
      C = 25, omega = 6
   selected by the centered multi-horizon Van der Pol tuning with
   retained-variation validation.

3. old_default:
      C = 20, omega = 4.0
   the older Van der Pol setting, retained as a reference.

The experiment increases the number of training trajectories and evaluates
clean held-out multi-step association rollouts on separate test trajectories.

Place this file in the same directory as:
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Manuscript-grade run:
    python3 run_exp13_vanderpol_tuned_trajectory_generalization.py

Pilot run:
    python3 run_exp13_vanderpol_tuned_trajectory_generalization.py \
      --configs robust_tuned clean_tuned \
      --train-counts 1 3 \
      --replicates 0 \
      --test-seeds 100

Resume interrupted run:
    python3 run_exp13_vanderpol_tuned_trajectory_generalization.py --resume

Outputs:
- kahkm_exp13_vanderpol_tuned_trajectory_generalization/
- kahkm_exp13_vanderpol_tuned_trajectory_generalization_results.zip
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
ConfigName: TypeAlias = Literal["robust_tuned", "clean_tuned", "old_default"]


@dataclass(frozen=True)
class KAHKMConfig:
    name: ConfigName
    n_clusters: int
    omega: float


@dataclass(frozen=True)
class RunKey:
    config_name: str
    n_train_trajectories: int
    replicate: int


@dataclass(frozen=True)
class ExperimentConfig:
    configs: tuple[KAHKMConfig, ...]
    train_counts: tuple[int, ...]
    replicates: tuple[int, ...]
    train_seed_pool_start: int
    test_seeds: tuple[int, ...]
    n_steps_train: int
    n_steps_test: int
    dt: float
    mu: float
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
    resume: bool


AVAILABLE_CONFIGS: dict[str, KAHKMConfig] = {
    "robust_tuned": KAHKMConfig("robust_tuned", n_clusters=25, omega=4.0),
    "clean_tuned": KAHKMConfig("clean_tuned", n_clusters=25, omega=6.0),
    "old_default": KAHKMConfig("old_default", n_clusters=20, omega=4.0),
}


def parse_ints(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def parse_config_names(values: Sequence[str]) -> tuple[KAHKMConfig, ...]:
    configs: list[KAHKMConfig] = []
    for value in values:
        name = str(value)
        if name not in AVAILABLE_CONFIGS:
            raise argparse.ArgumentTypeError(
                "Unknown config. Available: " + ", ".join(sorted(AVAILABLE_CONFIGS))
            )
        configs.append(AVAILABLE_CONFIGS[name])
    if not configs:
        raise argparse.ArgumentTypeError("At least one config is required.")
    return tuple(configs)


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
        description="Van der Pol tuned KAHKM trajectory-generalization experiment."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["robust_tuned", "clean_tuned", "old_default"],
        help="Configs to compare: robust_tuned clean_tuned old_default.",
    )
    parser.add_argument(
        "--train-counts",
        nargs="+",
        default=["1", "2", "3", "5", "8"],
        help="Number of training trajectories to concatenate.",
    )
    parser.add_argument(
        "--replicates",
        nargs="+",
        default=["0", "1", "2"],
        help="Replicate IDs. Each replicate uses a shifted block of training seeds.",
    )
    parser.add_argument("--train-seed-pool-start", type=int, default=0)
    parser.add_argument(
        "--test-seeds",
        nargs="+",
        default=["100", "101", "102"],
        help="Held-out test trajectory seeds.",
    )

    parser.add_argument("--n-steps-train", type=int, default=900)
    parser.add_argument("--n-steps-test", type=int, default=800)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["1", "10", "50", "100", "200"],
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
    parser.add_argument(
        "--output-dir",
        default="kahkm_exp13_vanderpol_tuned_trajectory_generalization",
    )
    parser.add_argument("--resume", action="store_true")

    ns = parser.parse_args()

    max_train_per_cluster_raw = int(ns.max_train_per_cluster)
    max_train_per_cluster = None if max_train_per_cluster_raw <= 0 else max_train_per_cluster_raw

    return ExperimentConfig(
        configs=parse_config_names([str(value) for value in ns.configs]),
        train_counts=tuple(sorted(set(parse_ints([str(value) for value in ns.train_counts])))),
        replicates=parse_ints([str(value) for value in ns.replicates]),
        train_seed_pool_start=int(ns.train_seed_pool_start),
        test_seeds=parse_ints([str(value) for value in ns.test_seeds]),
        n_steps_train=int(ns.n_steps_train),
        n_steps_test=int(ns.n_steps_test),
        dt=float(ns.dt),
        mu=float(ns.mu),
        horizons=tuple(sorted(set(parse_ints([str(value) for value in ns.horizons])))),
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
        resume=bool(ns.resume),
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


def concatenate_training_trajectories(
    *,
    seeds: Sequence[int],
    n_steps_train: int,
    dt: float,
    mu: float,
) -> tuple[FloatArray, FloatArray]:
    x0_parts: list[FloatArray] = []
    x1_parts: list[FloatArray] = []
    for seed in seeds:
        X0, X1 = make_vanderpol_snapshots(
            n_steps=int(n_steps_train),
            dt=float(dt),
            seed=int(seed),
            mu=float(mu),
        )
        x0_parts.append(X0)
        x1_parts.append(X1)
    return (
        np.concatenate(x0_parts, axis=1).astype(np.float64, copy=False),
        np.concatenate(x1_parts, axis=1).astype(np.float64, copy=False),
    )


def evaluate_on_test_trajectories(
    *,
    fit_result: Any,
    test_seeds: Sequence[int],
    n_steps_test: int,
    dt: float,
    mu: float,
    horizons: Sequence[int],
    n_jobs: int,
    batch_size: int,
) -> list[CsvRow]:
    abstraction_model = getattr(fit_result, "abstraction_model")
    B = np.asarray(getattr(fit_result, "B"), dtype=np.float64)
    omega = float(getattr(fit_result, "omega"))
    tau = float(getattr(fit_result, "tau"))

    rows: list[CsvRow] = []

    for horizon in horizons:
        h = int(horizon)
        pred_parts: list[FloatArray] = []
        target_parts: list[FloatArray] = []
        n_eval_total = 0

        for seed in test_seeds:
            X0_test, X1_test = make_vanderpol_snapshots(
                n_steps=int(n_steps_test),
                dt=float(dt),
                seed=int(seed),
                mu=float(mu),
            )
            n_test = int(X0_test.shape[1])
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

            pred_parts.append(rollout(B, Phi_start, h))
            target_parts.append(Phi_target)
            n_eval_total += n_start

        if not pred_parts:
            continue

        pred_all = np.concatenate(pred_parts, axis=1)
        target_all = np.concatenate(target_parts, axis=1)
        rows.append(
            {
                "horizon": h,
                "n_eval_starts": n_eval_total,
                "relative_association_error": relative_error(pred_all, target_all),
                "association_r2": association_r2(pred_all, target_all),
                "simplex_violation": float(simplex_violation(pred_all)),
            }
        )

    return rows


def completed_keys(raw_path: Path) -> set[RunKey]:
    keys: set[RunKey] = set()
    for row in _read_csv(raw_path):
        keys.add(
            RunKey(
                config_name=_cell(row, "config_name"),
                n_train_trajectories=_int_or_zero(_cell(row, "n_train_trajectories")),
                replicate=_int_or_zero(_cell(row, "replicate")),
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


def summarize(raw_path: Path, output_dir: Path) -> None:
    raw = _read_csv(raw_path)

    groups: dict[tuple[str, int, int], list[RawCsvRow]] = defaultdict(list)
    for row in raw:
        key = (
            _cell(row, "config_name"),
            _int_or_zero(_cell(row, "n_train_trajectories")),
            _int_or_zero(_cell(row, "horizon")),
        )
        groups[key].append(row)

    summary_rows: list[CsvRow] = []
    for (config_name, train_count, horizon), rows in sorted(groups.items()):
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

        first = rows[0]
        summary_rows.append(
            {
                "config_name": config_name,
                "n_clusters": _int_or_zero(_cell(first, "n_clusters")),
                "omega": _float_or_nan(_cell(first, "omega")),
                "n_train_trajectories": train_count,
                "horizon": horizon,
                "n_runs": len(errors),
                "relative_association_error_mean": mean(errors),
                "relative_association_error_std": pstdev(errors) if len(errors) > 1 else 0.0,
                "association_r2_mean": mean(r2s) if r2s else math.nan,
                "association_r2_std": pstdev(r2s) if len(r2s) > 1 else 0.0,
                "simplex_violation_mean": mean(violations) if violations else math.nan,
                "simplex_violation_max": max(violations) if violations else math.nan,
                "fit_seconds_mean": mean(fit_seconds) if fit_seconds else math.nan,
            }
        )

    _write_csv(output_dir / "experiment_13_tuned_trajectory_summary.csv", summary_rows)

    # Manuscript table: one row per config/train_count with E50/E100/E200.
    by_config_count: dict[tuple[str, int], dict[int, RawCsvRow]] = defaultdict(dict)
    for row in _read_csv(output_dir / "experiment_13_tuned_trajectory_summary.csv"):
        by_config_count[
            (_cell(row, "config_name"), _int_or_zero(_cell(row, "n_train_trajectories")))
        ][_int_or_zero(_cell(row, "horizon"))] = row

    table_rows: list[CsvRow] = []
    for (config_name, train_count), horizon_rows in sorted(by_config_count.items()):
        sample = next(iter(horizon_rows.values()))
        table_rows.append(
            {
                "config_name": config_name,
                "n_clusters": _int_or_zero(_cell(sample, "n_clusters")),
                "omega": _float_or_nan(_cell(sample, "omega")),
                "n_train_trajectories": train_count,
                "E1_mean": _float_or_nan(_cell(horizon_rows.get(1, {}), "relative_association_error_mean")),
                "E1_std": _float_or_nan(_cell(horizon_rows.get(1, {}), "relative_association_error_std")),
                "E50_mean": _float_or_nan(_cell(horizon_rows.get(50, {}), "relative_association_error_mean")),
                "E50_std": _float_or_nan(_cell(horizon_rows.get(50, {}), "relative_association_error_std")),
                "E100_mean": _float_or_nan(_cell(horizon_rows.get(100, {}), "relative_association_error_mean")),
                "E100_std": _float_or_nan(_cell(horizon_rows.get(100, {}), "relative_association_error_std")),
                "E200_mean": _float_or_nan(_cell(horizon_rows.get(200, {}), "relative_association_error_mean")),
                "E200_std": _float_or_nan(_cell(horizon_rows.get(200, {}), "relative_association_error_std")),
            }
        )

    _write_csv(output_dir / "experiment_13_tuned_trajectory_table_values.csv", table_rows)

    # Best config per train_count by E200.
    best_rows: list[CsvRow] = []
    by_count: dict[int, list[CsvRow]] = defaultdict(list)
    for row in table_rows:
        by_count[int(row["n_train_trajectories"])].append(row)

    for train_count, rows in sorted(by_count.items()):
        candidates = [row for row in rows if math.isfinite(float(row["E200_mean"]))]
        if not candidates:
            continue
        candidates.sort(key=lambda row: float(row["E200_mean"]))
        best = candidates[0]
        best_rows.append(
            {
                "n_train_trajectories": train_count,
                "best_config_name": str(best["config_name"]),
                "n_clusters": int(best["n_clusters"]),
                "omega": float(best["omega"]),
                "E200_mean": float(best["E200_mean"]),
                "E200_std": float(best["E200_std"]),
                "E100_mean": float(best["E100_mean"]),
                "E50_mean": float(best["E50_mean"]),
            }
        )

    _write_csv(output_dir / "experiment_13_tuned_trajectory_best_by_train_count.csv", best_rows)


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


def train_seeds_for_replicate(
    *,
    train_seed_pool_start: int,
    replicate: int,
    n_train_trajectories: int,
) -> tuple[int, ...]:
    # Non-overlapping blocks by replicate, with enough space for the largest default train count.
    start = int(train_seed_pool_start) + 1000 * int(replicate)
    return tuple(start + i for i in range(int(n_train_trajectories)))


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "experiment_13_tuned_trajectory_raw.csv"
    done = completed_keys(raw_path) if config.resume else set()

    total_jobs = len(config.configs) * len(config.train_counts) * len(config.replicates)
    job_index = 0
    start_all = time.time()

    print("\nVan der Pol tuned trajectory-generalization experiment")
    print(f"Output directory: {output_dir}")
    print(f"Total jobs: {total_jobs}")
    print(f"Resume: {config.resume}; completed keys loaded: {len(done)}")

    for kahkm_config in config.configs:
        for train_count in config.train_counts:
            for replicate in config.replicates:
                job_index += 1
                key = RunKey(
                    config_name=kahkm_config.name,
                    n_train_trajectories=int(train_count),
                    replicate=int(replicate),
                )
                if key in done:
                    print(
                        f"[{job_index}/{total_jobs}] skip {kahkm_config.name}, "
                        f"train_count={train_count}, replicate={replicate}"
                    )
                    continue

                train_seeds = train_seeds_for_replicate(
                    train_seed_pool_start=config.train_seed_pool_start,
                    replicate=int(replicate),
                    n_train_trajectories=int(train_count),
                )

                print(
                    f"\n[{job_index}/{total_jobs}] fit {kahkm_config.name}: "
                    f"C={kahkm_config.n_clusters}, omega={kahkm_config.omega}, "
                    f"train_count={train_count}, replicate={replicate}, "
                    f"train_seeds={train_seeds}"
                )

                X0_train, X1_train = concatenate_training_trajectories(
                    seeds=train_seeds,
                    n_steps_train=int(config.n_steps_train),
                    dt=float(config.dt),
                    mu=float(config.mu),
                )

                fit_start = time.time()
                fit = fit_kahkm(
                    X0_train,
                    X1_train,
                    n_clusters=int(kahkm_config.n_clusters),
                    subspace_dim=int(config.subspace_dim),
                    Nb=int(config.nb),
                    omega=float(kahkm_config.omega),
                    tau=float(config.tau),
                    beta=float(config.beta),
                    nlms_epochs=int(config.nlms_epochs),
                    random_state=int(replicate),
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

                eval_rows = evaluate_on_test_trajectories(
                    fit_result=fit,
                    test_seeds=config.test_seeds,
                    n_steps_test=int(config.n_steps_test),
                    dt=float(config.dt),
                    mu=float(config.mu),
                    horizons=config.horizons,
                    n_jobs=int(config.n_jobs),
                    batch_size=int(config.batch_size),
                )

                csv_rows: list[CsvRow] = []
                for row in eval_rows:
                    csv_rows.append(
                        {
                            "config_name": kahkm_config.name,
                            "n_clusters": kahkm_config.n_clusters,
                            "omega": kahkm_config.omega,
                            "n_train_trajectories": int(train_count),
                            "replicate": int(replicate),
                            "train_seeds": " ".join(str(seed) for seed in train_seeds),
                            "test_seeds": " ".join(str(seed) for seed in config.test_seeds),
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
                summarize(raw_path, output_dir)

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
        "experiment": "Van der Pol tuned trajectory-generalization",
        "config": {
            key: value for key, value in asdict(config).items() if key != "configs"
        },
        "configs": [asdict(kahkm_config) for kahkm_config in config.configs],
        "total_jobs": total_jobs,
        "total_seconds": time.time() - start_all,
    }
    with (output_dir / "experiment_13_tuned_trajectory_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summarize(raw_path, output_dir)
    zip_path = make_zip(output_dir, "kahkm_exp13_vanderpol_tuned_trajectory_generalization_results")

    print("\nFinished Van der Pol tuned trajectory-generalization experiment.")
    print(f"Raw results: {raw_path.resolve()}")
    print(f"Summary: {(output_dir / 'experiment_13_tuned_trajectory_summary.csv').resolve()}")
    print(f"Table values: {(output_dir / 'experiment_13_tuned_trajectory_table_values.csv').resolve()}")
    print(f"Best by train count: {(output_dir / 'experiment_13_tuned_trajectory_best_by_train_count.csv').resolve()}")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload the ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
