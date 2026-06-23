#!/usr/bin/env python3
"""Run tuned Experiment 17 interpretability configurations for KAHKM Classic Control.

Place this file in the same directory as:
- experiment_17_ai_regime_interpretability_pylance_clean.py
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Then run:

    python3 run_exp17_tuned_interpretability.py

Default tuned configurations:
- CartPole-v1:     C=20,  omega=1
- MountainCar-v0:  C=100, omega=1

Manuscript-compatible NLMS settings:
- beta = 0.1
- nlms_epochs = 20

By default this script runs seed 0 only, matching the immediate interpretability update
request. To run multiple seeds:

    python3 run_exp17_tuned_interpretability.py --seeds 0 1 2

At the end it creates:
    kahkm_exp17_tuned_interpretability_results.zip

Upload that ZIP for manuscript-table/text updating.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TaskRun:
    task: str
    n_clusters: int
    omega: float
    output_prefix: str


def _parse_seed_list(values: Sequence[str]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tuned KAHKM Experiment 17 Classic Control interpretability configurations."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child runs. Default: current interpreter.",
    )
    parser.add_argument(
        "--script",
        default="experiment_17_ai_regime_interpretability_pylance_clean.py",
        help="Experiment 17 script filename.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0"],
        help="Training/random seeds to run. Default: 0.",
    )
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument(
        "--test-seed-base",
        type=int,
        default=2000,
        help="Test seed offset. Test seed = test_seed_base + seed.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Directory in which result folders will be created.",
    )
    parser.add_argument(
        "--zip-name",
        default="kahkm_exp17_tuned_interpretability_results",
        help="Name of ZIP file without .zip suffix.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if its output directory already exists and is non-empty.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print("\n" + "=" * 80)
    print(printable)
    print("=" * 80, flush=True)

    if dry_run:
        return

    subprocess.run(command, check=True)


def directory_has_outputs(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(item.is_file() for item in path.rglob("*"))


def main() -> None:
    args = parse_args()
    seeds = _parse_seed_list(args.seeds)

    script_path = Path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find {script_path}. Run this script from the directory containing "
            "experiment_17_ai_regime_interpretability_pylance_clean.py, or pass --script."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    task_runs = (
        TaskRun(
            task="CartPole-v1",
            n_clusters=20,
            omega=1.0,
            output_prefix="kahkm_exp17_cartpole_tuned_seed",
        ),
        TaskRun(
            task="MountainCar-v0",
            n_clusters=100,
            omega=1.0,
            output_prefix="kahkm_exp17_mountaincar_tuned_seed",
        ),
    )

    output_dirs: list[Path] = []

    for task_run in task_runs:
        print(
            f"\nRunning tuned interpretability configuration for {task_run.task}: "
            f"C={task_run.n_clusters}, omega={task_run.omega}"
        )

        for seed in seeds:
            output_dir = output_root / f"{task_run.output_prefix}{seed}"
            output_dirs.append(output_dir)

            if args.skip_existing and directory_has_outputs(output_dir):
                print(f"Skipping existing non-empty result directory: {output_dir}")
                continue

            command = [
                args.python,
                str(script_path),
                "--tasks",
                task_run.task,
                "--n-clusters",
                str(task_run.n_clusters),
                "--omega",
                str(task_run.omega),
                "--subspace-dim",
                str(args.subspace_dim),
                "--train-episodes",
                str(args.train_episodes),
                "--test-episodes",
                str(args.test_episodes),
                "--beta",
                str(args.beta),
                "--nlms-epochs",
                str(args.nlms_epochs),
                "--train-seed",
                str(seed),
                "--test-seed",
                str(args.test_seed_base + seed),
                "--random-state",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            run_command(command, dry_run=bool(args.dry_run))

    if args.dry_run:
        print("\nDry run finished. No ZIP created.")
        return

    missing_or_empty = [str(path) for path in output_dirs if not directory_has_outputs(path)]
    if missing_or_empty:
        raise RuntimeError(
            "Some expected output folders are missing or empty:\n"
            + "\n".join(missing_or_empty)
        )

    zip_base = output_root / args.zip_name
    zip_path = Path(str(zip_base) + ".zip")
    if zip_path.exists():
        zip_path.unlink()

    staging_dir = output_root / f"{args.zip_name}_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for path in output_dirs:
        destination = staging_dir / path.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(path, destination)

    shutil.make_archive(str(zip_base), "zip", root_dir=staging_dir)
    shutil.rmtree(staging_dir)

    print("\nFinished all Experiment 17 tuned interpretability runs.")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload this ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
