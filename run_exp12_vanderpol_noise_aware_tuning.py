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

Legacy robust-selection objective:
    score = mean over seeds and noise levels of
            0.25 E50 + 0.25 E100 + 0.50 E200

Centered/non-collapse robust selection:
    For each seed and training-noise level, compute
        0.25 (1 - R2_50) + 0.25 (1 - R2_100) + 0.50 (1 - R2_200).
    Average this centered score over training-noise levels for each seed,
    then form a one-standard-error set across seeds. Within that set,
    prefer the configuration with the largest clean held-out normalized
    association variation; normalized effective rank is secondary.

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
  (legacy uncentered-error ranking, preserved for provenance)
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_noise_tuning_best_by_noise.csv
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_centered_robust_ranking.csv
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_centered_one_se_finalists.csv
- kahkm_exp12_vanderpol_noise_aware_tuning/experiment_12_centered_retvar_selected_config.csv
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
    association_variation_diagnostics,
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
    zip_name: str
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


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = mean(values)
    return math.sqrt(
        sum((value - mean_value) ** 2 for value in values)
        / (len(values) - 1)
    )


def _standard_error(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return _sample_std(values) / math.sqrt(len(values))


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
    parser.add_argument(
        "--zip-name",
        default="kahkm_exp12_vanderpol_noise_aware_tuning_results",
        help=(
            "ZIP filename without the .zip suffix. The default preserves "
            "the original repository behavior."
        ),
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
        zip_name=str(ns.zip_name),
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

    if file_exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, [])

        if existing_header != fieldnames:
            raise RuntimeError(
                "Existing Experiment 12 raw CSV uses a different schema. "
                "Do not append retained-variation rows to a legacy raw file. "
                "Use a new --output-dir for the updated experiment. "
                f"Existing columns: {existing_header}; "
                f"new columns: {fieldnames}"
            )

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
        rho_vars = [
            _float_or_nan(_cell(row, "target_normalized_association_variation"))
            for row in rows
        ]
        effective_ranks = [
            _float_or_nan(_cell(row, "target_association_effective_rank"))
            for row in rows
        ]
        rho_ranks = [
            _float_or_nan(_cell(row, "target_normalized_association_effective_rank"))
            for row in rows
        ]

        errors = [value for value in errors if math.isfinite(value)]
        r2s = [value for value in r2s if math.isfinite(value)]
        violations = [value for value in violations if math.isfinite(value)]
        fit_seconds = [value for value in fit_seconds if math.isfinite(value)]
        rho_vars = [value for value in rho_vars if math.isfinite(value)]
        effective_ranks = [value for value in effective_ranks if math.isfinite(value)]
        rho_ranks = [value for value in rho_ranks if math.isfinite(value)]

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
                "target_normalized_association_variation_mean": (
                    mean(rho_vars) if rho_vars else math.nan
                ),
                "target_normalized_association_variation_std": (
                    pstdev(rho_vars) if len(rho_vars) > 1 else (0.0 if rho_vars else math.nan)
                ),
                "target_association_effective_rank_mean": (
                    mean(effective_ranks) if effective_ranks else math.nan
                ),
                "target_association_effective_rank_std": (
                    pstdev(effective_ranks)
                    if len(effective_ranks) > 1
                    else (0.0 if effective_ranks else math.nan)
                ),
                "target_normalized_association_effective_rank_mean": (
                    mean(rho_ranks) if rho_ranks else math.nan
                ),
                "target_normalized_association_effective_rank_std": (
                    pstdev(rho_ranks) if len(rho_ranks) > 1 else (0.0 if rho_ranks else math.nan)
                ),
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

    # -----------------------------------------------------------------
    # Centered robust selection.
    #
    # Independent units for the one-SE rule are the random-state seeds.
    # For each seed, first compute the weighted centered error at each
    # training-noise level, then average those scores over noise levels.
    # -----------------------------------------------------------------
    per_seed_noise_r2: dict[
        tuple[int, float, int, float],
        dict[int, float],
    ] = defaultdict(dict)

    for row in raw:
        horizon = _int_or_zero(_cell(row, "horizon"))
        if horizon not in selection_weights:
            continue

        r2 = _float_or_nan(_cell(row, "association_r2"))
        if not math.isfinite(r2):
            continue

        key = (
            _int_or_zero(_cell(row, "n_clusters")),
            _float_or_nan(_cell(row, "omega")),
            _int_or_zero(_cell(row, "seed")),
            _float_or_nan(_cell(row, "noise_level")),
        )
        per_seed_noise_r2[key][horizon] = r2

    seed_noise_scores: dict[
        tuple[int, float, int],
        list[float],
    ] = defaultdict(list)

    for (
        n_clusters,
        omega,
        seed,
        _noise_level,
    ), horizon_r2 in per_seed_noise_r2.items():
        if not all(horizon in horizon_r2 for horizon in selection_weights):
            continue

        centered_score = sum(
            selection_weights[horizon] * (1.0 - horizon_r2[horizon])
            for horizon in selection_weights
        )
        seed_noise_scores[(n_clusters, omega, seed)].append(centered_score)

    centered_scores_by_config: dict[
        tuple[int, float],
        list[float],
    ] = defaultdict(list)

    for (n_clusters, omega, _seed), noise_scores in seed_noise_scores.items():
        if not noise_scores:
            continue
        centered_scores_by_config[(n_clusters, omega)].append(
            mean(noise_scores)
        )

    # Retained variation is a model-level diagnostic. Each run repeats the
    # same value on all horizon rows, so deduplicate by
    # (C, omega, noise_level, seed) before aggregating by configuration.
    run_retvar: dict[
        tuple[int, float, float, int],
        tuple[float, float, float],
    ] = {}

    for row in raw:
        rho_var = _float_or_nan(
            _cell(row, "target_normalized_association_variation")
        )
        effective_rank = _float_or_nan(
            _cell(row, "target_association_effective_rank")
        )
        rho_rank = _float_or_nan(
            _cell(row, "target_normalized_association_effective_rank")
        )

        if not (
            math.isfinite(rho_var)
            and math.isfinite(effective_rank)
            and math.isfinite(rho_rank)
        ):
            continue

        run_key = (
            _int_or_zero(_cell(row, "n_clusters")),
            _float_or_nan(_cell(row, "omega")),
            _float_or_nan(_cell(row, "noise_level")),
            _int_or_zero(_cell(row, "seed")),
        )
        run_retvar[run_key] = (
            rho_var,
            effective_rank,
            rho_rank,
        )

    retvar_by_config: dict[
        tuple[int, float],
        dict[str, list[float]],
    ] = defaultdict(
        lambda: {
            "rho_var": [],
            "effective_rank": [],
            "rho_rank": [],
        }
    )

    for (
        n_clusters,
        omega,
        _noise_level,
        _seed,
    ), (
        rho_var,
        effective_rank,
        rho_rank,
    ) in run_retvar.items():
        bucket = retvar_by_config[(n_clusters, omega)]
        bucket["rho_var"].append(rho_var)
        bucket["effective_rank"].append(effective_rank)
        bucket["rho_rank"].append(rho_rank)

    centered_rows: list[CsvRow] = []

    for (n_clusters, omega), seed_scores in centered_scores_by_config.items():
        if not seed_scores:
            continue

        centered_mean = mean(seed_scores)
        centered_sd = _sample_std(seed_scores)
        centered_se = _standard_error(seed_scores)

        diagnostics = retvar_by_config.get(
            (n_clusters, omega),
            {
                "rho_var": [],
                "effective_rank": [],
                "rho_rank": [],
            },
        )
        rho_vars = diagnostics["rho_var"]
        effective_ranks = diagnostics["effective_rank"]
        rho_ranks = diagnostics["rho_rank"]

        centered_rows.append(
            {
                "centered_rank": 0,
                "n_clusters": n_clusters,
                "omega": omega,
                "n_seeds": len(seed_scores),
                "n_noise_levels_per_seed": (
                    len(
                        seed_noise_scores.get(
                            (n_clusters, omega, next(
                                seed
                                for (config_c, config_omega, seed)
                                in seed_noise_scores
                                if config_c == n_clusters
                                and abs(config_omega - omega) < 1e-12
                            )),
                            [],
                        )
                    )
                    if seed_scores
                    else 0
                ),
                "centered_robust_score_mean": centered_mean,
                "centered_robust_score_sd": centered_sd,
                "centered_robust_score_se": centered_se,
                "mean_robust_multihorizon_r2": 1.0 - centered_mean,
                "target_normalized_association_variation_mean": (
                    mean(rho_vars) if rho_vars else math.nan
                ),
                "target_normalized_association_variation_std": (
                    _sample_std(rho_vars) if rho_vars else math.nan
                ),
                "target_association_effective_rank_mean": (
                    mean(effective_ranks) if effective_ranks else math.nan
                ),
                "target_association_effective_rank_std": (
                    _sample_std(effective_ranks) if effective_ranks else math.nan
                ),
                "target_normalized_association_effective_rank_mean": (
                    mean(rho_ranks) if rho_ranks else math.nan
                ),
                "target_normalized_association_effective_rank_std": (
                    _sample_std(rho_ranks) if rho_ranks else math.nan
                ),
                "n_retained_variation_runs": len(rho_vars),
                "selection_weights": ",".join(
                    f"{h}:{w:.6g}"
                    for h, w in sorted(selection_weights.items())
                ),
                "inside_centered_one_se_set": 0,
                "selected_by_centered_retvar_rule": 0,
            }
        )

    if centered_rows:
        centered_rows.sort(
            key=lambda row: float(row["centered_robust_score_mean"])
        )

        best_mean = float(
            centered_rows[0]["centered_robust_score_mean"]
        )
        best_se = float(
            centered_rows[0]["centered_robust_score_se"]
        )
        one_se_limit = best_mean + best_se

        for rank, row in enumerate(centered_rows, start=1):
            row["centered_rank"] = rank
            row["centered_best_mean_score"] = best_mean
            row["centered_best_standard_error"] = best_se
            row["centered_one_se_limit"] = one_se_limit
            row["inside_centered_one_se_set"] = int(
                float(row["centered_robust_score_mean"])
                <= one_se_limit
            )

        finalists = [
            dict(row)
            for row in centered_rows
            if int(row["inside_centered_one_se_set"]) == 1
        ]

        def _retvar_sort_key(
            row: CsvRow,
        ) -> tuple[float, float, float, int, float]:
            rho_var = float(
                row["target_normalized_association_variation_mean"]
            )
            rho_rank = float(
                row[
                    "target_normalized_association_effective_rank_mean"
                ]
            )
            centered_score = float(
                row["centered_robust_score_mean"]
            )

            if not math.isfinite(rho_var):
                rho_var = -math.inf
            if not math.isfinite(rho_rank):
                rho_rank = -math.inf

            return (
                -rho_var,
                -rho_rank,
                centered_score,
                int(row["n_clusters"]),
                float(row["omega"]),
            )

        finalists.sort(key=_retvar_sort_key)

        # If the centered one-SE set contains a unique member, it is selected
        # even when retained-variation fields are unavailable. If there are
        # multiple finalists, retained variation must be available to break
        # the tie; otherwise we fail loudly rather than silently reverting to
        # the legacy objective.
        if len(finalists) == 1:
            selected_c = int(finalists[0]["n_clusters"])
            selected_omega = float(finalists[0]["omega"])
        else:
            if not math.isfinite(
                float(
                    finalists[0][
                        "target_normalized_association_variation_mean"
                    ]
                )
            ):
                raise RuntimeError(
                    "Multiple configurations are inside the centered robust "
                    "one-SE set, but retained-variation diagnostics are not "
                    "available. Rerun those finalists with the updated "
                    "Experiment 12 script in a new output directory."
                )
            selected_c = int(finalists[0]["n_clusters"])
            selected_omega = float(finalists[0]["omega"])

        for row in centered_rows:
            row["selected_by_centered_retvar_rule"] = int(
                int(row["n_clusters"]) == selected_c
                and abs(float(row["omega"]) - selected_omega) < 1e-12
            )

        finalists = [
            dict(row)
            for row in centered_rows
            if int(row["inside_centered_one_se_set"]) == 1
        ]
        finalists.sort(key=_retvar_sort_key)

        for rank, row in enumerate(finalists, start=1):
            row["retained_variation_rank_within_one_se"] = rank

        selected_rows = [
            dict(row)
            for row in centered_rows
            if int(row["selected_by_centered_retvar_rule"]) == 1
        ]

        _write_csv(
            output_dir / "experiment_12_centered_robust_ranking.csv",
            centered_rows,
        )
        _write_csv(
            output_dir / "experiment_12_centered_one_se_finalists.csv",
            finalists,
        )
        _write_csv(
            output_dir / "experiment_12_centered_retvar_selected_config.csv",
            selected_rows,
        )

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

                    # Model-level non-collapse diagnostics on the complete
                    # clean held-out one-step target association set.
                    target_associations = np.asarray(
                        kahm_associations(
                            getattr(fit, "abstraction_model"),
                            X1_test_clean,
                            omega=float(getattr(fit, "omega")),
                            tau=float(getattr(fit, "tau")),
                            n_jobs=int(config.n_jobs),
                            batch_size=int(config.batch_size),
                            show_progress=False,
                        ),
                        dtype=np.float64,
                    )
                    target_variation = association_variation_diagnostics(
                        target_associations
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
                                "target_association_variation": float(
                                    target_variation["association_variation"]
                                ),
                                "target_max_variation_given_mean": float(
                                    target_variation["max_variation_given_mean"]
                                ),
                                "target_normalized_association_variation": float(
                                    target_variation[
                                        "normalized_association_variation"
                                    ]
                                ),
                                "target_association_effective_rank": float(
                                    target_variation[
                                        "association_effective_rank"
                                    ]
                                ),
                                "target_normalized_association_effective_rank": float(
                                    target_variation[
                                        "normalized_association_effective_rank"
                                    ]
                                ),
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
                        f"E200={float(e200):.6g}, "
                        f"rho_var={float(target_variation['normalized_association_variation']):.6g}, "
                        f"rho_rank={float(target_variation['normalized_association_effective_rank']):.6g}, "
                        f"fit_seconds={fit_seconds:.2f}"
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
        "centered_selection": (
            "per seed: weighted mean of 1-R^2 over configured selection horizons "
            "at each training-noise level; average over noise levels; one-SE set "
            "across seeds; retained variation used only within that set"
        ),
        "retained_variation": (
            "computed on the complete clean held-out X1 association set for each fitted model"
        ),
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
    zip_path = make_zip(output_dir, config.zip_name)

    print("\nFinished noise-aware Van der Pol tuning.")
    print(f"Raw results: {raw_path.resolve()}")
    print(f"Legacy robust selection: {(output_dir / 'experiment_12_noise_tuning_robust_selection.csv').resolve()}")
    print(f"Centered robust ranking: {(output_dir / 'experiment_12_centered_robust_ranking.csv').resolve()}")
    print(f"Centered one-SE finalists: {(output_dir / 'experiment_12_centered_one_se_finalists.csv').resolve()}")
    print(f"Centered/retained-variation selected config: {(output_dir / 'experiment_12_centered_retvar_selected_config.csv').resolve()}")
    print(f"Best by noise: {(output_dir / 'experiment_12_noise_tuning_best_by_noise.csv').resolve()}")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload the ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
