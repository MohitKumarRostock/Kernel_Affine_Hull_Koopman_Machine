#!/usr/bin/env python3
"""Final Acrobot three-seed validation for manuscript upgrade.

Purpose
-------
This is the final high-priority validation experiment for the manuscript.

Earlier tuning found a strong Acrobot improvement with:
    C = 150, omega = 0.5

but that result was seed-0 only. This script validates that selected
configuration over three seeds and aggregates the output into manuscript-ready
CSV files.

It runs:
    experiment_18_acrobot_sequential_decision_pylance_clean.py

with the selected Acrobot configuration:
    clusters = 150
    omegas   = 0.5

Default manuscript-grade run:
    python3 run_exp18_acrobot_selected_3seed_validation.py

Dry run:
    python3 run_exp18_acrobot_selected_3seed_validation.py --dry-run

Resume / skip existing completed folders:
    python3 run_exp18_acrobot_selected_3seed_validation.py --skip-existing

Place this file in the same directory as:
- experiment_18_acrobot_sequential_decision_pylance_clean.py
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Outputs:
- kahkm_exp18_acrobot_selected_3seed_validation_results.zip
- kahkm_exp18_acrobot_selected_3seed_validation_aggregate/
    - experiment_18_selected_seed_summary.csv
    - experiment_18_selected_one_step_aggregate.csv
    - experiment_18_selected_multistep_aggregate.csv
    - experiment_18_selected_interpretability_aggregate.csv
    - experiment_18_selected_kahkm_table_values.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence, TypeAlias


CsvCell: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvCell]
RawCsvRow: TypeAlias = dict[str, str]


@dataclass(frozen=True)
class RunSpec:
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class RunnerConfig:
    python_executable: str
    script: str
    seeds: tuple[int, ...]
    train_episodes: int
    test_episodes: int
    max_steps: int
    horizons: tuple[int, ...]
    n_clusters: int
    omega: float
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    exploration_eps: float
    risk_window: int
    n_jobs: int
    batch_size: int
    kmeans_kind: str
    kmeans_batch_size: int
    max_train_per_cluster: int
    test_seed_base: int
    output_root: str
    output_prefix: str
    zip_name: str
    dry_run: bool
    skip_existing: bool


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


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


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(
        description="Validate selected Acrobot KAHKM configuration over three seeds."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--script", default="experiment_18_acrobot_sequential_decision_pylance_clean.py")
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])

    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--horizons", nargs="+", default=["1", "5", "10", "20", "50"])

    parser.add_argument("--n-clusters", type=int, default=150)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)

    parser.add_argument("--exploration-eps", type=float, default=0.10)
    parser.add_argument("--risk-window", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--kmeans-kind", default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)

    parser.add_argument("--test-seed-base", type=int, default=100)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--output-prefix", default="kahkm_exp18_acrobot_selected_seed")
    parser.add_argument("--zip-name", default="kahkm_exp18_acrobot_selected_3seed_validation_results")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    ns = parser.parse_args()

    return RunnerConfig(
        python_executable=str(ns.python),
        script=str(ns.script),
        seeds=_parse_int_tuple([str(value) for value in ns.seeds]),
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        max_steps=int(ns.max_steps),
        horizons=_parse_int_tuple([str(value) for value in ns.horizons]),
        n_clusters=int(ns.n_clusters),
        omega=float(ns.omega),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        exploration_eps=float(ns.exploration_eps),
        risk_window=int(ns.risk_window),
        n_jobs=int(ns.n_jobs),
        batch_size=int(ns.batch_size),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        max_train_per_cluster=int(ns.max_train_per_cluster),
        test_seed_base=int(ns.test_seed_base),
        output_root=str(ns.output_root),
        output_prefix=str(ns.output_prefix),
        zip_name=str(ns.zip_name),
        dry_run=bool(ns.dry_run),
        skip_existing=bool(ns.skip_existing),
    )


def output_complete(output_dir: Path) -> bool:
    required = [
        "experiment_18_best_config.csv",
        "experiment_18_one_step_results.csv",
        "experiment_18_multistep_summary.csv",
        "experiment_18_model_summary.csv",
        "experiment_18_regime_statistics.csv",
        "experiment_18_metadata.json",
    ]
    return all((output_dir / name).exists() for name in required)


def build_command(config: RunnerConfig, spec: RunSpec) -> list[str]:
    command = [
        config.python_executable,
        str(Path(config.script)),
        "--clusters",
        str(config.n_clusters),
        "--omegas",
        str(config.omega),
        "--train-episodes",
        str(config.train_episodes),
        "--test-episodes",
        str(config.test_episodes),
        "--max-steps",
        str(config.max_steps),
        "--horizons",
        *[str(horizon) for horizon in config.horizons],
        "--subspace-dim",
        str(config.subspace_dim),
        "--nb",
        str(config.nb),
        "--beta",
        str(config.beta),
        "--nlms-epochs",
        str(config.nlms_epochs),
        "--exploration-eps",
        str(config.exploration_eps),
        "--risk-window",
        str(config.risk_window),
        "--n-jobs",
        str(config.n_jobs),
        "--batch-size",
        str(config.batch_size),
        "--kmeans-kind",
        str(config.kmeans_kind),
        "--kmeans-batch-size",
        str(config.kmeans_batch_size),
        "--max-train-per-cluster",
        str(config.max_train_per_cluster),
        "--train-seed",
        str(spec.seed),
        "--test-seed",
        str(config.test_seed_base + spec.seed),
        "--random-state",
        str(spec.seed),
        "--output-dir",
        str(spec.output_dir),
    ]
    return command


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print("\n" + "=" * 96)
    print(printable)
    print("=" * 96, flush=True)

    if dry_run:
        return

    subprocess.run(list(command), check=True)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    return mean(finite), pstdev(finite) if len(finite) > 1 else 0.0


def aggregate_results(run_specs: Sequence[RunSpec], aggregate_dir: Path) -> None:
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[CsvRow] = []
    one_step_rows_all: list[tuple[int, RawCsvRow]] = []
    multistep_rows_all: list[tuple[int, RawCsvRow]] = []
    model_rows_all: list[tuple[int, RawCsvRow]] = []

    for spec in run_specs:
        best = _read_csv(spec.output_dir / "experiment_18_best_config.csv")
        one_step = _read_csv(spec.output_dir / "experiment_18_one_step_results.csv")
        multistep = _read_csv(spec.output_dir / "experiment_18_multistep_summary.csv")
        model = _read_csv(spec.output_dir / "experiment_18_model_summary.csv")

        best_row = best[0] if best else {}
        kahkm_one = next((row for row in one_step if _cell(row, "method") == "kahkm_nlms"), {})
        kahkm_h50 = next(
            (
                row for row in multistep
                if _cell(row, "method") == "kahkm_nlms"
                and _int_or_zero(_cell(row, "horizon")) == 50
            ),
            {},
        )
        model_row = model[0] if model else {}

        seed_rows.append(
            {
                "seed": spec.seed,
                "output_dir": str(spec.output_dir),
                "selected_C": _cell(best_row, "C", _cell(best_row, "n_clusters")),
                "selected_omega": _cell(best_row, "omega"),
                "selection_score": _cell(best_row, "score", _cell(best_row, "mean_relative_error")),
                "kahkm_E1": _cell(kahkm_one, "test_error", _cell(kahkm_one, "relative_error")),
                "kahkm_R2_1": _cell(kahkm_one, "test_r2", _cell(kahkm_one, "association_r2")),
                "kahkm_E50": _cell(kahkm_h50, "relative_error"),
                "kahkm_R2_50": _cell(kahkm_h50, "association_r2"),
                "active_regimes": _cell(model_row, "active_regimes"),
                "action_purity": _cell(model_row, "action_purity"),
                "high_purity_mass": _cell(model_row, "high_purity_mass"),
                "learned_persistence": _cell(model_row, "learned_persistence"),
                "max_terminal_risk": _cell(model_row, "max_terminal_risk"),
                "max_risk_regime": _cell(model_row, "max_risk_regime"),
                "max_risk_regime_support": _cell(model_row, "max_risk_regime_support"),
                "weighted_terminal_risk": _cell(model_row, "weighted_terminal_risk"),
            }
        )

        one_step_rows_all.extend((spec.seed, row) for row in one_step)
        multistep_rows_all.extend((spec.seed, row) for row in multistep)
        model_rows_all.extend((spec.seed, row) for row in model)

    _write_csv(aggregate_dir / "experiment_18_selected_seed_summary.csv", seed_rows)

    # Aggregate one-step by method.
    one_step_groups: dict[str, list[RawCsvRow]] = defaultdict(list)
    for _seed, row in one_step_rows_all:
        one_step_groups[_cell(row, "method")].append(row)

    one_step_agg: list[CsvRow] = []
    for method, rows in sorted(one_step_groups.items()):
        errors = [_float_or_nan(_cell(row, "test_error", _cell(row, "relative_error"))) for row in rows]
        r2s = [_float_or_nan(_cell(row, "test_r2", _cell(row, "association_r2"))) for row in rows]
        violations = [_float_or_nan(_cell(row, "simplex_violation")) for row in rows]
        err_mean, err_std = mean_std(errors)
        r2_mean, r2_std = mean_std(r2s)
        violation_mean, violation_std = mean_std(violations)
        one_step_agg.append(
            {
                "method": method,
                "test_error_mean": err_mean,
                "test_error_std": err_std,
                "test_r2_mean": r2_mean,
                "test_r2_std": r2_std,
                "simplex_violation_mean": violation_mean,
                "simplex_violation_std": violation_std,
                "n": len(rows),
            }
        )

    _write_csv(aggregate_dir / "experiment_18_selected_one_step_aggregate.csv", one_step_agg)

    # Aggregate multistep by method/horizon.
    multistep_groups: dict[tuple[str, int], list[RawCsvRow]] = defaultdict(list)
    for _seed, row in multistep_rows_all:
        multistep_groups[
            (_cell(row, "method"), _int_or_zero(_cell(row, "horizon")))
        ].append(row)

    multistep_agg: list[CsvRow] = []
    for (method, horizon), rows in sorted(multistep_groups.items(), key=lambda item: (item[0][0], item[0][1])):
        errors = [_float_or_nan(_cell(row, "relative_error")) for row in rows]
        r2s = [_float_or_nan(_cell(row, "association_r2")) for row in rows]
        violations = [_float_or_nan(_cell(row, "simplex_violation")) for row in rows]
        err_mean, err_std = mean_std(errors)
        r2_mean, r2_std = mean_std(r2s)
        violation_mean, violation_std = mean_std(violations)
        multistep_agg.append(
            {
                "method": method,
                "horizon": horizon,
                "relative_error_mean": err_mean,
                "relative_error_std": err_std,
                "association_r2_mean": r2_mean,
                "association_r2_std": r2_std,
                "simplex_violation_mean": violation_mean,
                "simplex_violation_std": violation_std,
                "n": len(rows),
            }
        )

    _write_csv(aggregate_dir / "experiment_18_selected_multistep_aggregate.csv", multistep_agg)

    # Aggregate interpretability/model summary.
    interp_keys = [
        "active_regimes",
        "train_closure_error",
        "train_association_r2",
        "action_purity",
        "high_purity_mass",
        "learned_persistence",
        "max_terminal_risk",
        "max_risk_regime_support",
        "weighted_terminal_risk",
    ]
    interp_rows: list[CsvRow] = []
    for key in interp_keys:
        values = [
            _float_or_nan(_cell(row, key))
            for _seed, row in model_rows_all
            if _cell(row, key) != ""
        ]
        value_mean, value_std = mean_std(values)
        interp_rows.append(
            {
                "metric": key,
                "mean": value_mean,
                "std": value_std,
                "n": len([value for value in values if math.isfinite(value)]),
            }
        )

    _write_csv(aggregate_dir / "experiment_18_selected_interpretability_aggregate.csv", interp_rows)

    # KAHKM manuscript table row.
    kahkm_horizon_rows = [
        row for row in multistep_agg
        if row["method"] == "kahkm_nlms"
    ]
    by_horizon = {int(row["horizon"]): row for row in kahkm_horizon_rows}
    kahkm_one = next((row for row in one_step_agg if row["method"] == "kahkm_nlms"), None)

    table_rows: list[CsvRow] = []
    if kahkm_one is not None:
        table_rows.append(
            {
                "task": "Acrobot-v1",
                "n_clusters": 150,
                "omega": 0.5,
                "E1_mean": float(kahkm_one["test_error_mean"]),
                "E1_std": float(kahkm_one["test_error_std"]),
                "R2_1_mean": float(kahkm_one["test_r2_mean"]),
                "R2_1_std": float(kahkm_one["test_r2_std"]),
                "E50_mean": float(by_horizon.get(50, {}).get("relative_error_mean", math.nan)),
                "E50_std": float(by_horizon.get(50, {}).get("relative_error_std", math.nan)),
                "R2_50_mean": float(by_horizon.get(50, {}).get("association_r2_mean", math.nan)),
                "R2_50_std": float(by_horizon.get(50, {}).get("association_r2_std", math.nan)),
                "n": int(kahkm_one["n"]),
            }
        )

    _write_csv(aggregate_dir / "experiment_18_selected_kahkm_table_values.csv", table_rows)


def make_zip(
    *,
    output_root: Path,
    zip_name: str,
    run_specs: Sequence[RunSpec],
    aggregate_dir: Path,
) -> Path:
    zip_base = output_root / zip_name
    zip_path = Path(str(zip_base) + ".zip")
    if zip_path.exists():
        zip_path.unlink()

    staging_dir = output_root / f"{zip_name}_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for spec in run_specs:
        destination = staging_dir / spec.output_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(spec.output_dir, destination)

    if aggregate_dir.exists():
        shutil.copytree(aggregate_dir, staging_dir / aggregate_dir.name)

    shutil.make_archive(str(zip_base), "zip", root_dir=staging_dir)
    shutil.rmtree(staging_dir)
    return zip_path


def main() -> None:
    config = parse_args()
    script_path = Path(config.script)

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find {script_path}. Run this file from the directory containing "
            "experiment_18_acrobot_sequential_decision_pylance_clean.py, or pass --script."
        )

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_specs = [
        RunSpec(
            seed=seed,
            output_dir=output_root / f"{config.output_prefix}{seed}",
        )
        for seed in config.seeds
    ]

    start = time.time()
    for spec in run_specs:
        if config.skip_existing and output_complete(spec.output_dir):
            print(f"Skipping existing completed output folder: {spec.output_dir}")
            continue

        command = build_command(config, spec)
        run_command(command, dry_run=config.dry_run)

    if config.dry_run:
        print("\nDry run finished. No experiments were executed.")
        return

    incomplete = [str(spec.output_dir) for spec in run_specs if not output_complete(spec.output_dir)]
    if incomplete:
        raise RuntimeError(
            "Some output folders are incomplete:\n" + "\n".join(incomplete)
        )

    aggregate_dir = output_root / "kahkm_exp18_acrobot_selected_3seed_validation_aggregate"
    if aggregate_dir.exists():
        shutil.rmtree(aggregate_dir)
    aggregate_results(run_specs, aggregate_dir)

    metadata = {
        "runner": "run_exp18_acrobot_selected_3seed_validation.py",
        "purpose": "Validate selected Acrobot C=150, omega=0.5 configuration over multiple seeds.",
        "runner_config": asdict(config),
        "total_seconds": time.time() - start,
    }
    with (aggregate_dir / "experiment_18_selected_runner_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    zip_path = make_zip(
        output_root=output_root,
        zip_name=config.zip_name,
        run_specs=run_specs,
        aggregate_dir=aggregate_dir,
    )

    print("\nFinished Acrobot selected-config validation.")
    print(f"Aggregate directory: {aggregate_dir.resolve()}")
    print(f"KAHKM table values: {(aggregate_dir / 'experiment_18_selected_kahkm_table_values.csv').resolve()}")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload the ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
