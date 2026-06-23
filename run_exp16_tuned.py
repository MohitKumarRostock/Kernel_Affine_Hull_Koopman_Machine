#!/usr/bin/env python3
"""Run tuned Experiment 16 configurations for KAHKM Classic Control.

Place this file in the same directory as:
- experiment_16_ai_classic_control_closed_loop_pylance_clean.py
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Then run:

    python3 run_exp16_tuned.py

The script runs three seeds for:
- CartPole-v1 with C=20, omega=1
- MountainCar-v0 with C=100, omega=1

It uses manuscript-compatible NLMS settings:
- beta = 0.1
- nlms_epochs = 20

At the end it creates:
    kahkm_exp16_tuned_results.zip

Upload that ZIP for manuscript-table updating.
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
        description="Run tuned KAHKM Experiment 16 Classic Control configurations."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child runs. Default: current interpreter.",
    )
    parser.add_argument(
        "--script",
        default="experiment_16_ai_classic_control_closed_loop_pylance_clean.py",
        help="Experiment 16 script filename.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0", "1", "2"],
        help="Training/random seeds to run. Default: 0 1 2.",
    )
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["1", "5", "10", "20", "50"],
        help="Horizons passed to Experiment 16.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Directory in which result folders will be created.",
    )
    parser.add_argument(
        "--zip-name",
        default="kahkm_exp16_tuned_results",
        help="Name of ZIP file without .zip suffix.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if its output directory already contains experiment_16_multistep_summary.csv.",
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


def main() -> None:
    args = parse_args()
    seeds = _parse_seed_list(args.seeds)

    script_path = Path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find {script_path}. Run this script from the directory containing "
            "experiment_16_ai_classic_control_closed_loop_pylance_clean.py, or pass --script."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    task_runs = (
        TaskRun(
            task="CartPole-v1",
            n_clusters=20,
            omega=1.0,
            output_prefix="kahkm_exp16_cartpole_tuned_seed",
        ),
        TaskRun(
            task="MountainCar-v0",
            n_clusters=100,
            omega=1.0,
            output_prefix="kahkm_exp16_mountaincar_tuned_seed",
        ),
    )

    output_dirs: list[Path] = []

    for task_run in task_runs:
        print(
            f"\nRunning tuned configuration for {task_run.task}: "
            f"C={task_run.n_clusters}, omega={task_run.omega}"
        )

        for seed in seeds:
            output_dir = output_root / f"{task_run.output_prefix}{seed}"
            output_dirs.append(output_dir)

            done_file = output_dir / "experiment_16_multistep_summary.csv"
            if args.skip_existing and done_file.exists():
                print(f"Skipping existing result directory: {output_dir}")
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
                "--horizons",
                *[str(h) for h in args.horizons],
                "--beta",
                str(args.beta),
                "--nlms-epochs",
                str(args.nlms_epochs),
                "--train-seed",
                str(seed),
                "--test-seed",
                str(1000 + seed),
                "--random-state",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            run_command(command, dry_run=bool(args.dry_run))

    if args.dry_run:
        print("\nDry run finished. No ZIP created.")
        return

    missing = [
        str(path)
        for path in output_dirs
        if not (path / "experiment_16_multistep_summary.csv").exists()
    ]
    if missing:
        raise RuntimeError(
            "Some expected output folders do not contain experiment_16_multistep_summary.csv:\n"
            + "\n".join(missing)
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

    print("\nFinished all runs.")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload this ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
