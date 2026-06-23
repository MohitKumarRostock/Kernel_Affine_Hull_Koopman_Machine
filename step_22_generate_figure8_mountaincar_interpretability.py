#!/usr/bin/env python3
"""Generate manuscript Figure 8 from tuned Experiment 17 MountainCar interpretability outputs.

Figure 8 in the manuscript is assembled from two PDF panels:

    figures/fig_mountaincar_dominant_regime.pdf
    figures/fig_mountaincar_terminal_risk.pdf

They are derived from Experiment 17 tuned interpretability output for the selected
MountainCar model:

    task  = MountainCar-v0
    C     = 100
    omega = 1
    seed  = 0 by default

The source Experiment 17 runner is:

    run_exp17_tuned_interpretability.py

and it creates the source files:

    kahkm_exp17_mountaincar_tuned_seed0/figures/MountainCar-v0_dominant_regime_scatter.pdf
    kahkm_exp17_mountaincar_tuned_seed0/figures/MountainCar-v0_terminal_risk_by_regime.pdf

Place this script in the same directory as the experiment scripts and run:

    python step_22_generate_figure8_mountaincar_interpretability.py

To reuse existing Experiment 17 outputs:

    python step_22_generate_figure8_mountaincar_interpretability.py --no-run

Outputs:
    figures/fig_mountaincar_dominant_regime.pdf
    figures/fig_mountaincar_terminal_risk.pdf
    kahkm_figure8_mountaincar_interpretability/fig_mountaincar_dominant_regime.pdf
    kahkm_figure8_mountaincar_interpretability/fig_mountaincar_terminal_risk.pdf
    kahkm_figure8_mountaincar_interpretability/figure8_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    runner_script: str
    output_root: Path
    output_dir: Path
    figure_dir: Path
    python_executable: str
    no_run: bool
    skip_existing: bool
    seed: int
    train_episodes: int
    test_episodes: int
    subspace_dim: int
    beta: float
    nlms_epochs: int
    test_seed_base: int


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 8 from tuned Experiment 17 MountainCar interpretability outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--runner-script", default="run_exp17_tuned_interpretability.py")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd(),
        help="Directory containing/receiving kahkm_exp17_* output folders. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_figure8_mountaincar_interpretability"),
        help="Directory for copied panels and metadata.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Manuscript figure directory.",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Pass --skip-existing to the Experiment 17 tuned runner. Default: enabled.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--test-seed-base", type=int, default=2000)

    ns = parser.parse_args()
    source_dir = Path(ns.source_dir).resolve()
    output_root = Path(ns.output_root)
    if not output_root.is_absolute():
        output_root = source_dir / output_root
    output_dir = Path(ns.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir
    figure_dir = Path(ns.figure_dir)
    if not figure_dir.is_absolute():
        figure_dir = source_dir / figure_dir

    return CliArgs(
        source_dir=source_dir,
        runner_script=str(ns.runner_script),
        output_root=output_root,
        output_dir=output_dir,
        figure_dir=figure_dir,
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        seed=int(ns.seed),
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        subspace_dim=int(ns.subspace_dim),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        test_seed_base=int(ns.test_seed_base),
    )


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return

    runner_path = args.source_dir / args.runner_script
    if not runner_path.exists():
        raise FileNotFoundError(f"Could not find Experiment 17 tuned runner: {runner_path}")

    command = [
        args.python_executable,
        str(runner_path),
        "--python",
        args.python_executable,
        "--seeds",
        str(args.seed),
        "--train-episodes",
        str(args.train_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--subspace-dim",
        str(args.subspace_dim),
        "--beta",
        str(args.beta),
        "--nlms-epochs",
        str(args.nlms_epochs),
        "--test-seed-base",
        str(args.test_seed_base),
        "--output-root",
        str(args.output_root),
    ]
    if args.skip_existing:
        command.append("--skip-existing")

    print("Running tuned Experiment 17 interpretability wrapper:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_run_dir(args: CliArgs) -> Path:
    return args.output_root / f"kahkm_exp17_mountaincar_tuned_seed{args.seed}"


def copy_panels(args: CliArgs) -> dict[str, str]:
    run_dir = source_run_dir(args)
    source_fig_dir = run_dir / "figures"
    if not source_fig_dir.exists():
        raise FileNotFoundError(f"Missing Experiment 17 source figure directory: {source_fig_dir}")

    source_dominant = source_fig_dir / "MountainCar-v0_dominant_regime_scatter.pdf"
    source_risk = source_fig_dir / "MountainCar-v0_terminal_risk_by_regime.pdf"

    missing = [path for path in (source_dominant, source_risk) if not path.exists()]
    if missing:
        available = sorted(path.name for path in source_fig_dir.glob("*.pdf"))
        raise FileNotFoundError(
            "Missing expected MountainCar source panels:\n"
            + "\n".join(str(path) for path in missing)
            + "\nAvailable PDF files:\n"
            + "\n".join(available)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    output_dominant = args.output_dir / "fig_mountaincar_dominant_regime.pdf"
    output_risk = args.output_dir / "fig_mountaincar_terminal_risk.pdf"
    manuscript_dominant = args.figure_dir / "fig_mountaincar_dominant_regime.pdf"
    manuscript_risk = args.figure_dir / "fig_mountaincar_terminal_risk.pdf"

    for src, dst in (
        (source_dominant, output_dominant),
        (source_risk, output_risk),
        (source_dominant, manuscript_dominant),
        (source_risk, manuscript_risk),
    ):
        shutil.copy2(src, dst)

    return {
        "source_dominant": str(source_dominant),
        "source_risk": str(source_risk),
        "output_dominant": str(output_dominant),
        "output_risk": str(output_risk),
        "manuscript_dominant": str(manuscript_dominant),
        "manuscript_risk": str(manuscript_risk),
    }


def write_metadata(args: CliArgs, copied: dict[str, str]) -> Path:
    run_dir = source_run_dir(args)
    model_summary_path = run_dir / "experiment_17_model_summary.csv"
    regime_stats_path = run_dir / "experiment_17_regime_statistics.csv"
    model_rows = _read_csv(model_summary_path)
    regime_rows = _read_csv(regime_stats_path)

    metadata: dict[str, Any] = {
        "figure": "Figure 8",
        "manuscript_panels": [
            "figures/fig_mountaincar_dominant_regime.pdf",
            "figures/fig_mountaincar_terminal_risk.pdf",
        ],
        "source_runner": args.runner_script,
        "source_run_dir": str(run_dir),
        "source_model_summary": str(model_summary_path),
        "source_regime_statistics": str(regime_stats_path),
        "copied_files": copied,
        "selected_model": {
            "task": "MountainCar-v0",
            "n_clusters": 100,
            "omega": 1.0,
            "seed": args.seed,
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "beta": args.beta,
            "nlms_epochs": args.nlms_epochs,
        },
        "model_summary_rows": model_rows,
        "n_regime_stat_rows": len(regime_rows),
        "interpretation": (
            "Figure 8 visualizes the tuned MountainCar KAHKM abstraction by dominant regime "
            "and by regime-level terminal-risk statistics."
        ),
    }

    path = args.output_dir / "figure8_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


def main() -> None:
    args = parse_args()
    run_experiment(args)
    copied = copy_panels(args)
    metadata_path = write_metadata(args, copied)

    print("Wrote:")
    for key in ("manuscript_dominant", "manuscript_risk", "output_dominant", "output_risk"):
        print(f"  {copied[key]}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()
