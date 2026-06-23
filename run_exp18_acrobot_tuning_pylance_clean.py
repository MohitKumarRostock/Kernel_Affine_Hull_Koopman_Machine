#!/usr/bin/env python3
"""Run Acrobot-v1 Experiment 18 tuning sweeps.

Place this file in the same directory as:
- experiment_18_acrobot_sequential_decision_pylance_clean.py
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Pilot run, one seed:

    python3 run_exp18_acrobot_tuning_pylance_clean.py

Manuscript-grade run, three seeds:

    python3 run_exp18_acrobot_tuning_pylance_clean.py --seeds 0 1 2

Dry run:

    python3 run_exp18_acrobot_tuning_pylance_clean.py --dry-run

The default tuning grid is:
- C in {20, 40, 50, 75, 100, 150}
- omega in {0.5, 1, 2, 4, 8}

Manuscript-compatible closure settings:
- beta = 0.1
- nlms_epochs = 20
- subspace_dim = 20
- Nb = 100

At the end, this script creates:
- kahkm_exp18_acrobot_tuning_results.zip
- kahkm_exp18_acrobot_tuning_aggregate/experiment_18_seed_summary.csv
- kahkm_exp18_acrobot_tuning_aggregate/experiment_18_kahkm_horizon_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
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


def _parse_seed_list(values: Sequence[str]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


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
    """Pylance-clean conversion of CSV values to float."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Acrobot-v1 KAHKM Experiment 18 tuning sweeps."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child runs. Default: current interpreter.",
    )
    parser.add_argument(
        "--script",
        default="experiment_18_acrobot_sequential_decision_pylance_clean.py",
        help="Experiment 18 script filename.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0"],
        help="Seeds to run. Default: 0. Use '--seeds 0 1 2' for manuscript-grade results.",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=["20", "40", "50", "75", "100", "150"],
        help="Cluster grid. Default: 20 40 50 75 100 150.",
    )
    parser.add_argument(
        "--omegas",
        nargs="+",
        default=["0.5", "1", "2", "4", "8"],
        help="Omega grid. Default: 0.5 1 2 4 8.",
    )
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["1", "5", "10", "20", "50"],
        help="Rollout horizons. Default: 1 5 10 20 50.",
    )
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
    parser.add_argument(
        "--test-seed-base",
        type=int,
        default=100,
        help="Test seed = test_seed_base + seed. Default: 100.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Directory in which result folders will be created.",
    )
    parser.add_argument(
        "--output-prefix",
        default="kahkm_exp18_acrobot_tuned_seed",
        help="Prefix for per-seed output folders.",
    )
    parser.add_argument(
        "--zip-name",
        default="kahkm_exp18_acrobot_tuning_results",
        help="Name of ZIP file without .zip suffix.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if experiment_18_model_summary.csv already exists in its output folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print("\n" + "=" * 96)
    print(printable)
    print("=" * 96, flush=True)

    if dry_run:
        return

    subprocess.run(command, check=True)


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


def build_aggregate(run_specs: Sequence[RunSpec], aggregate_dir: Path) -> None:
    seed_rows: list[CsvRow] = []

    for spec in run_specs:
        best = _read_csv(spec.output_dir / "experiment_18_best_config.csv")
        model = _read_csv(spec.output_dir / "experiment_18_model_summary.csv")
        one_step = _read_csv(spec.output_dir / "experiment_18_one_step_results.csv")
        multistep = _read_csv(spec.output_dir / "experiment_18_multistep_summary.csv")

        best_row = best[0] if best else {}
        model_row = model[0] if model else {}

        kahkm_one = next(
            (row for row in one_step if _cell(row, "method") == "kahkm_nlms"),
            {},
        )
        kahkm_h50 = next(
            (
                row for row in multistep
                if _cell(row, "method") == "kahkm_nlms" and _cell(row, "horizon") == "50"
            ),
            {},
        )

        seed_rows.append(
            {
                "seed": spec.seed,
                "output_dir": str(spec.output_dir),
                "selected_C": _cell(best_row, "C", _cell(best_row, "n_clusters")),
                "selected_omega": _cell(best_row, "omega"),
                "best_score": _cell(best_row, "score", _cell(best_row, "mean_relative_error")),
                "kahkm_E1": _cell(kahkm_one, "test_error", _cell(kahkm_one, "relative_error")),
                "kahkm_R2_1": _cell(kahkm_one, "test_r2", _cell(kahkm_one, "association_r2")),
                "kahkm_E50": _cell(kahkm_h50, "relative_error"),
                "kahkm_R2_50": _cell(kahkm_h50, "association_r2"),
                "active_regimes": _cell(model_row, "active_regimes"),
                "action_purity": _cell(model_row, "action_purity"),
                "learned_persistence": _cell(model_row, "learned_persistence"),
                "max_terminal_risk": _cell(model_row, "max_terminal_risk"),
                "max_risk_regime": _cell(model_row, "max_risk_regime"),
            }
        )

    _write_csv(aggregate_dir / "experiment_18_seed_summary.csv", seed_rows)

    horizon_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for spec in run_specs:
        multistep = _read_csv(spec.output_dir / "experiment_18_multistep_summary.csv")
        for row in multistep:
            if _cell(row, "method") != "kahkm_nlms":
                continue
            horizon = _cell(row, "horizon")
            err = _float_or_nan(row.get("relative_error"))
            r2 = _float_or_nan(row.get("association_r2"))
            horizon_values[horizon].append((err, r2))

    horizon_rows: list[CsvRow] = []
    for horizon in sorted(horizon_values, key=lambda x: int(float(x))):
        vals = horizon_values[horizon]
        errs = [v[0] for v in vals if math.isfinite(v[0])]
        r2s = [v[1] for v in vals if math.isfinite(v[1])]

        if not errs or not r2s:
            continue

        horizon_rows.append(
            {
                "method": "kahkm_nlms",
                "horizon": int(float(horizon)),
                "relative_error_mean": mean(errs),
                "relative_error_std": pstdev(errs) if len(errs) > 1 else 0.0,
                "association_r2_mean": mean(r2s),
                "association_r2_std": pstdev(r2s) if len(r2s) > 1 else 0.0,
                "n": len(errs),
            }
        )

    _write_csv(aggregate_dir / "experiment_18_kahkm_horizon_summary.csv", horizon_rows)


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
    args = parse_args()

    seeds = _parse_seed_list(args.seeds)
    script_path = Path(str(args.script))
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find {script_path}. Run this script from the directory containing "
            "experiment_18_acrobot_sequential_decision_pylance_clean.py, or pass --script."
        )

    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    run_specs = [
        RunSpec(
            seed=seed,
            output_dir=output_root / f"{args.output_prefix}{seed}",
        )
        for seed in seeds
    ]

    for spec in run_specs:
        if args.skip_existing and output_complete(spec.output_dir):
            print(f"Skipping existing completed output folder: {spec.output_dir}")
            continue

        command = [
            str(args.python),
            str(script_path),
            "--clusters",
            *[str(v) for v in args.clusters],
            "--omegas",
            *[str(v) for v in args.omegas],
            "--train-episodes",
            str(args.train_episodes),
            "--test-episodes",
            str(args.test_episodes),
            "--max-steps",
            str(args.max_steps),
            "--horizons",
            *[str(v) for v in args.horizons],
            "--subspace-dim",
            str(args.subspace_dim),
            "--nb",
            str(args.nb),
            "--beta",
            str(args.beta),
            "--nlms-epochs",
            str(args.nlms_epochs),
            "--exploration-eps",
            str(args.exploration_eps),
            "--risk-window",
            str(args.risk_window),
            "--n-jobs",
            str(args.n_jobs),
            "--batch-size",
            str(args.batch_size),
            "--kmeans-kind",
            str(args.kmeans_kind),
            "--kmeans-batch-size",
            str(args.kmeans_batch_size),
            "--max-train-per-cluster",
            str(args.max_train_per_cluster),
            "--train-seed",
            str(spec.seed),
            "--test-seed",
            str(args.test_seed_base + spec.seed),
            "--random-state",
            str(spec.seed),
            "--output-dir",
            str(spec.output_dir),
        ]
        run_command(command, dry_run=bool(args.dry_run))

    if args.dry_run:
        print("\nDry run finished. No experiments were executed.")
        return

    incomplete = [str(spec.output_dir) for spec in run_specs if not output_complete(spec.output_dir)]
    if incomplete:
        raise RuntimeError(
            "Some expected output folders are incomplete:\n" + "\n".join(incomplete)
        )

    aggregate_dir = output_root / "kahkm_exp18_acrobot_tuning_aggregate"
    if aggregate_dir.exists():
        shutil.rmtree(aggregate_dir)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    build_aggregate(run_specs, aggregate_dir)

    zip_path = make_zip(
        output_root=output_root,
        zip_name=str(args.zip_name),
        run_specs=run_specs,
        aggregate_dir=aggregate_dir,
    )

    print("\nFinished Acrobot tuning experiments.")
    print(f"Aggregate CSVs: {aggregate_dir.resolve()}")
    print(f"Created ZIP: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
