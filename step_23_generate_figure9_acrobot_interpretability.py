#!/usr/bin/env python3
"""Generate manuscript Figure 9 from the selected Acrobot-v1 three-seed validation.

Figure 9 in the manuscript is the two-panel Acrobot interpretability figure:

    figures/fig_acrobot_dominant_regime_angles.pdf
    figures/fig_acrobot_dominant_regime_energy.pdf

It is derived from the selected Experiment 18 validation:

    run_exp18_acrobot_selected_3seed_validation.py

The selected model is:

    C = 150
    omega = 0.5

The validation is over seeds 0, 1, and 2. The plotted panels are representative
diagnostic plots copied from one selected-validation seed, seed 0 by default,
while Table 14 reports the three-seed aggregate interpretability values.

Place this script in the same directory as the experiment scripts and run:

    python step_23_generate_figure9_acrobot_interpretability.py

To reuse already-generated Experiment 18 selected-validation outputs:

    python step_23_generate_figure9_acrobot_interpretability.py --no-run

Outputs:
    figures/fig_acrobot_dominant_regime_angles.pdf
    figures/fig_acrobot_dominant_regime_energy.pdf
    kahkm_figure9_acrobot_interpretability/fig_acrobot_dominant_regime_angles.pdf
    kahkm_figure9_acrobot_interpretability/fig_acrobot_dominant_regime_energy.pdf
    kahkm_figure9_acrobot_interpretability/figure9_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    output_dir: Path
    figure_dir: Path
    python_executable: str
    runner_script: str
    experiment_script: str
    no_run: bool
    skip_existing: bool
    seeds: tuple[int, ...]
    representative_seed: int
    n_clusters: int
    omega: float
    output_prefix: str
    train_episodes: int
    test_episodes: int
    max_steps: int
    horizons: tuple[int, ...]
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


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 9 from selected Acrobot-v1 three-seed validation outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_figure9_acrobot_interpretability"),
        help="Directory for copied panels and metadata.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where manuscript figure PDFs are written.",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument(
        "--runner-script",
        default="run_exp18_acrobot_selected_3seed_validation.py",
        help="Selected three-seed validation runner.",
    )
    parser.add_argument(
        "--experiment-script",
        default="experiment_18_acrobot_sequential_decision_pylance_clean.py",
        help="Experiment 18 script called by the selected-validation runner.",
    )
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Pass --skip-existing to the selected-validation runner. Default: enabled.",
    )
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument(
        "--representative-seed",
        type=int,
        default=0,
        help="Seed whose diagnostic panels are copied into the manuscript. Default: 0.",
    )
    parser.add_argument("--n-clusters", type=int, default=150)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--output-prefix", default="kahkm_exp18_acrobot_selected_seed")
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--horizons", nargs="+", default=["1", "5", "10", "20", "50"])
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--exploration-eps", type=float, default=0.1)
    parser.add_argument("--risk-window", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--kmeans-kind", default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument("--test-seed-base", type=int, default=100)

    ns = parser.parse_args()
    source_dir = Path(ns.source_dir).resolve()

    output_dir = Path(ns.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir

    figure_dir = Path(ns.figure_dir)
    if not figure_dir.is_absolute():
        figure_dir = source_dir / figure_dir

    seeds = _parse_int_tuple(tuple(str(value) for value in ns.seeds))
    representative_seed = int(ns.representative_seed)
    if representative_seed not in seeds:
        raise ValueError("--representative-seed must be included in --seeds.")

    return CliArgs(
        source_dir=source_dir,
        output_dir=output_dir,
        figure_dir=figure_dir,
        python_executable=str(ns.python_executable),
        runner_script=str(ns.runner_script),
        experiment_script=str(ns.experiment_script),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        seeds=seeds,
        representative_seed=representative_seed,
        n_clusters=int(ns.n_clusters),
        omega=float(ns.omega),
        output_prefix=str(ns.output_prefix),
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        max_steps=int(ns.max_steps),
        horizons=_parse_int_tuple(tuple(str(value) for value in ns.horizons)),
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
    )


def run_selected_validation(args: CliArgs) -> None:
    if args.no_run:
        return

    runner_path = args.source_dir / args.runner_script
    if not runner_path.exists():
        raise FileNotFoundError(f"Could not find selected-validation runner: {runner_path}")

    command = [
        args.python_executable,
        str(runner_path),
        "--python",
        args.python_executable,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--n-clusters",
        str(args.n_clusters),
        "--omega",
        str(args.omega),
        "--output-prefix",
        args.output_prefix,
        "--script",
        args.experiment_script,
        "--train-episodes",
        str(args.train_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--max-steps",
        str(args.max_steps),
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
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
        args.kmeans_kind,
        "--kmeans-batch-size",
        str(args.kmeans_batch_size),
        "--max-train-per-cluster",
        str(args.max_train_per_cluster),
        "--test-seed-base",
        str(args.test_seed_base),
        "--output-root",
        str(args.source_dir),
    ]
    if args.skip_existing:
        command.append("--skip-existing")

    print("Running selected Acrobot three-seed validation:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: str | None, *, context: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Missing numeric value for {context}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {context}: {value!r}")
    return result


def seed_dir(args: CliArgs, seed: int) -> Path:
    return args.source_dir / f"{args.output_prefix}{seed}"


def verify_selected_outputs(args: CliArgs) -> None:
    for seed in args.seeds:
        run_dir = seed_dir(args, seed)
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Missing selected-validation output directory for seed {seed}: {run_dir}. "
                "Run without --no-run first."
            )

        best_config = run_dir / "experiment_18_best_config.csv"
        if best_config.exists():
            rows = _read_csv(best_config)
            if rows:
                row = rows[0]
                c_value = row.get("best_n_clusters") or row.get("n_clusters") or row.get("C")
                omega_value = row.get("best_omega") or row.get("omega")
                if c_value is not None and int(float(c_value)) != args.n_clusters:
                    raise ValueError(f"{best_config} has C={c_value}, expected {args.n_clusters}.")
                if omega_value is not None and abs(float(omega_value) - args.omega) > 1e-12:
                    raise ValueError(f"{best_config} has omega={omega_value}, expected {args.omega}.")

        model_summary = run_dir / "experiment_18_model_summary.csv"
        if model_summary.exists():
            rows = _read_csv(model_summary)
            if rows:
                row = rows[0]
                c_value = row.get("n_clusters") or row.get("C")
                omega_value = row.get("omega")
                if c_value is not None and int(float(c_value)) != args.n_clusters:
                    raise ValueError(f"{model_summary} has C={c_value}, expected {args.n_clusters}.")
                if omega_value is not None and abs(float(omega_value) - args.omega) > 1e-12:
                    raise ValueError(f"{model_summary} has omega={omega_value}, expected {args.omega}.")


def copy_panels(args: CliArgs) -> tuple[Path, Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rep_dir = seed_dir(args, args.representative_seed)
    source_angles = rep_dir / "figures" / "Acrobot-v1_dominant_regime_angles.pdf"
    source_energy = rep_dir / "figures" / "Acrobot-v1_dominant_regime_energy.pdf"

    if not source_angles.exists():
        raise FileNotFoundError(f"Missing representative angles panel: {source_angles}")
    if not source_energy.exists():
        raise FileNotFoundError(f"Missing representative energy panel: {source_energy}")

    manuscript_angles = args.figure_dir / "fig_acrobot_dominant_regime_angles.pdf"
    manuscript_energy = args.figure_dir / "fig_acrobot_dominant_regime_energy.pdf"
    local_angles = args.output_dir / "fig_acrobot_dominant_regime_angles.pdf"
    local_energy = args.output_dir / "fig_acrobot_dominant_regime_energy.pdf"

    shutil.copy2(source_angles, manuscript_angles)
    shutil.copy2(source_energy, manuscript_energy)
    shutil.copy2(source_angles, local_angles)
    shutil.copy2(source_energy, local_energy)

    return manuscript_angles, manuscript_energy, local_angles, local_energy


def write_metadata(
    args: CliArgs,
    manuscript_angles: Path,
    manuscript_energy: Path,
    local_angles: Path,
    local_energy: Path,
) -> Path:
    aggregate_dir = args.source_dir / "kahkm_exp18_acrobot_selected_3seed_validation_aggregate"
    metadata = {
        "figure": "Figure 9",
        "manuscript_files": {
            "angles_panel": str(manuscript_angles),
            "energy_panel": str(manuscript_energy),
        },
        "local_copies": {
            "angles_panel": str(local_angles),
            "energy_panel": str(local_energy),
        },
        "source_runner": args.runner_script,
        "source_experiment": args.experiment_script,
        "source_seed_directories": [str(seed_dir(args, seed)) for seed in args.seeds],
        "representative_seed": args.representative_seed,
        "aggregate_directory": str(aggregate_dir),
        "selected_model": {
            "task": "Acrobot-v1",
            "n_clusters": args.n_clusters,
            "omega": args.omega,
            "validated_seeds": list(args.seeds),
        },
        "settings": {
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "max_steps": args.max_steps,
            "horizons": list(args.horizons),
            "subspace_dim": args.subspace_dim,
            "nb": args.nb,
            "beta": args.beta,
            "nlms_epochs": args.nlms_epochs,
            "exploration_eps": args.exploration_eps,
            "risk_window": args.risk_window,
            "kmeans_kind": args.kmeans_kind,
        },
        "interpretation": (
            "Representative Acrobot-v1 panels for the selected C=150, omega=0.5 model. "
            "The model selection and validation are over three seeds; the figure displays the representative seed specified above."
        ),
    }
    path = args.output_dir / "figure9_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


def main() -> None:
    args = parse_args()
    run_selected_validation(args)
    verify_selected_outputs(args)
    manuscript_angles, manuscript_energy, local_angles, local_energy = copy_panels(args)
    metadata_path = write_metadata(args, manuscript_angles, manuscript_energy, local_angles, local_energy)

    print("Wrote:")
    for path in (manuscript_angles, manuscript_energy, local_angles, local_energy, metadata_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
