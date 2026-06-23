#!/usr/bin/env python3
"""Generate manuscript Figure 7 from Experiment 16A Classic Control sensitivity outputs.

Figure 7 in the manuscript is:

    figures/fig_ai_cluster_sensitivity.pdf

It is derived from Experiment 16A:

    experiment_16a_ai_hyperparameter_sensitivity_pylance_clean.py

The figure summarizes the expanded regime-count grid for the AI-facing Classic
Control experiments. For each task and each regime count C, the script selects
the omega value that minimizes the KAHKM h=50 relative association error, then
plots both the corresponding one-step error E_1 and long-horizon error E_50.

This makes the over-fragmentation diagnostic explicit: increasing C may help a
long-horizon CartPole curve, but excessive fragmentation can degrade one-step
closure or MountainCar performance.

Place this script in the same directory as the experiment scripts and run:

    python step_21_generate_figure7_ai_cluster_sensitivity.py

Outputs:
    figures/fig_ai_cluster_sensitivity.pdf
    kahkm_figure7_ai_cluster_sensitivity/fig_ai_cluster_sensitivity.png
    kahkm_figure7_ai_cluster_sensitivity/figure7_plot_data.csv
    kahkm_figure7_ai_cluster_sensitivity/figure7_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_TASKS: Final[tuple[str, ...]] = ("CartPole-v1", "MountainCar-v0")
DEFAULT_CLUSTERS: Final[tuple[int, ...]] = (10, 15, 20, 30, 50, 80, 100)
DEFAULT_OMEGAS: Final[tuple[float, ...]] = (2.0, 4.0, 8.0, 12.0)
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 5, 10, 20, 50)

TASK_LABELS: Final[dict[str, str]] = {
    "CartPole-v1": "CartPole",
    "MountainCar-v0": "MountainCar",
}

MARKERS: Final[dict[tuple[str, int], str]] = {
    ("CartPole-v1", 1): "o",
    ("CartPole-v1", 50): "s",
    ("MountainCar-v0", 1): "^",
    ("MountainCar-v0", 50): "D",
}

LINESTYLES: Final[dict[int, str]] = {
    1: "--",
    50: "-",
}


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    experiment_script: str
    experiment16_script: str
    experiment_output_dir: Path
    output_dir: Path
    figure_dir: Path
    python_executable: str
    no_run: bool
    skip_existing: bool
    tasks: tuple[str, ...]
    clusters: tuple[int, ...]
    omegas: tuple[float, ...]
    horizons: tuple[int, ...]
    selection_horizon: int
    train_episodes: int
    test_episodes: int
    max_steps: int
    train_seed: int
    test_seed: int
    random_state: int
    nb: int
    subspace_dim: int
    beta: float
    nlms_epochs: int
    tau: float
    batch_size: int
    n_jobs: int
    kmeans_kind: str
    kmeans_batch_size: int
    max_train_per_cluster: int
    ridge: float
    continue_on_error: bool
    legend_loc: str


@dataclass(frozen=True)
class SelectedClusterPoint:
    task: str
    n_clusters: int
    omega: float
    e1: float
    r2_1: float
    e50: float
    r2_50: float


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def _parse_float_tuple(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(float(v) for v in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return parsed


def _parse_task_tuple(values: Sequence[str]) -> tuple[str, ...]:
    parsed = tuple(str(v) for v in values)
    allowed = set(DEFAULT_TASKS)
    invalid = [value for value in parsed if value not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported task(s): {invalid}. Expected subset of {sorted(allowed)}.")
    if not parsed:
        raise argparse.ArgumentTypeError("At least one task is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 7 from Experiment 16A Classic Control sensitivity outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment-script",
        default="experiment_16a_ai_hyperparameter_sensitivity_pylance_clean.py",
        help="Experiment 16A sensitivity wrapper.",
    )
    parser.add_argument(
        "--experiment16-script",
        default="experiment_16_ai_classic_control_closed_loop_pylance_clean.py",
        help="Experiment 16 script called by the Experiment 16A sensitivity wrapper.",
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=Path("kahkm_experiment_16a_sensitivity_outputs"),
        help="Output directory used by Experiment 16A.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_figure7_ai_cluster_sensitivity"),
        help="Directory for plot data, PNG preview, and metadata.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where the manuscript PDF figure is written.",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip Experiment 16A if the required summary CSVs already exist. Default: enabled.",
    )

    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--clusters", nargs="+", default=[str(v) for v in DEFAULT_CLUSTERS])
    parser.add_argument("--omegas", nargs="+", default=[str(v) for v in DEFAULT_OMEGAS])
    parser.add_argument("--horizons", nargs="+", default=[str(v) for v in DEFAULT_HORIZONS])
    parser.add_argument(
        "--selection-horizon",
        type=int,
        default=50,
        help="For each C, choose omega by KAHKM relative error at this horizon. Default: 50.",
    )
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--kmeans-kind", type=str, default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--legend-loc",
        default="best",
        help="Matplotlib legend location. Default: best.",
    )

    ns = parser.parse_args()
    source_dir = Path(ns.source_dir).resolve()

    experiment_output_dir = Path(ns.experiment_output_dir)
    if not experiment_output_dir.is_absolute():
        experiment_output_dir = source_dir / experiment_output_dir

    output_dir = Path(ns.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir

    figure_dir = Path(ns.figure_dir)
    if not figure_dir.is_absolute():
        figure_dir = source_dir / figure_dir

    horizons = _parse_int_tuple(tuple(str(v) for v in ns.horizons))
    selection_horizon = int(ns.selection_horizon)
    if selection_horizon not in horizons:
        raise ValueError("--selection-horizon must be included in --horizons.")

    return CliArgs(
        source_dir=source_dir,
        experiment_script=str(ns.experiment_script),
        experiment16_script=str(ns.experiment16_script),
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        figure_dir=figure_dir,
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        tasks=_parse_task_tuple(tuple(str(v) for v in ns.tasks)),
        clusters=_parse_int_tuple(tuple(str(v) for v in ns.clusters)),
        omegas=_parse_float_tuple(tuple(str(v) for v in ns.omegas)),
        horizons=horizons,
        selection_horizon=selection_horizon,
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        max_steps=int(ns.max_steps),
        train_seed=int(ns.train_seed),
        test_seed=int(ns.test_seed),
        random_state=int(ns.random_state),
        nb=int(ns.nb),
        subspace_dim=int(ns.subspace_dim),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        tau=float(ns.tau),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        max_train_per_cluster=int(ns.max_train_per_cluster),
        ridge=float(ns.ridge),
        continue_on_error=bool(ns.continue_on_error),
        legend_loc=str(ns.legend_loc),
    )


def one_step_path(args: CliArgs) -> Path:
    return args.experiment_output_dir / "experiment_16a_kahkm_one_step_summary.csv"


def multistep_path(args: CliArgs) -> Path:
    return args.experiment_output_dir / "experiment_16a_kahkm_multistep_summary.csv"


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return

    expected_one = one_step_path(args)
    expected_multi = multistep_path(args)
    if args.skip_existing and expected_one.exists() and expected_multi.exists():
        print(f"Skipping Experiment 16A because summaries already exist: {expected_one}, {expected_multi}")
        return

    script_path = args.source_dir / args.experiment_script
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find Experiment 16A script: {script_path}")

    command = [
        args.python_executable,
        str(script_path),
        "--experiment16-script",
        str(args.source_dir / args.experiment16_script),
        "--output-dir",
        str(args.experiment_output_dir),
        "--tasks",
        *list(args.tasks),
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
        "--train-seed",
        str(args.train_seed),
        "--test-seed",
        str(args.test_seed),
        "--random-state",
        str(args.random_state),
        "--selection-horizon",
        str(args.selection_horizon),
        "--nb",
        str(args.nb),
        "--subspace-dim",
        str(args.subspace_dim),
        "--beta",
        str(args.beta),
        "--nlms-epochs",
        str(args.nlms_epochs),
        "--tau",
        str(args.tau),
        "--batch-size",
        str(args.batch_size),
        "--n-jobs",
        str(args.n_jobs),
        "--kmeans-kind",
        str(args.kmeans_kind),
        "--kmeans-batch-size",
        str(args.kmeans_batch_size),
        "--max-train-per-cluster",
        str(args.max_train_per_cluster),
        "--ridge",
        str(args.ridge),
    ]
    if args.continue_on_error:
        command.append("--continue-on-error")

    print("Running Experiment 16A Classic Control sensitivity:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: str | None, *, context: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Missing value for {context}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for {context}: {value!r}")
    return result


def _int_from_row(row: dict[str, str], name: str) -> int:
    return int(float(str(row.get(name, "nan"))))


def _float_from_row(row: dict[str, str], name: str) -> float:
    return _finite_float(row.get(name), context=f"{name} in row {row}")


def select_points(args: CliArgs) -> list[SelectedClusterPoint]:
    one_path = one_step_path(args)
    multi_path = multistep_path(args)
    if not one_path.exists() or not multi_path.exists():
        raise FileNotFoundError(
            f"Missing Experiment 16A summaries. Expected {one_path} and {multi_path}. "
            "Run without --no-run, or pass --experiment-output-dir."
        )

    one_rows = [
        row for row in _read_csv(one_path)
        if row.get("task") in args.tasks
        and row.get("method") == "kahkm_nlms"
    ]
    multi_rows = [
        row for row in _read_csv(multi_path)
        if row.get("task") in args.tasks
        and row.get("method") == "kahkm_nlms"
        and _int_from_row(row, "horizon") == args.selection_horizon
    ]

    points: list[SelectedClusterPoint] = []
    for task in args.tasks:
        for c_value in args.clusters:
            candidates = [
                row for row in multi_rows
                if row.get("task") == task and _int_from_row(row, "n_clusters") == c_value
            ]
            if not candidates:
                raise ValueError(f"No Experiment 16A h={args.selection_horizon} candidates for {task}, C={c_value}.")
            best = min(candidates, key=lambda row: _float_from_row(row, "relative_error"))
            omega = _float_from_row(best, "omega")

            one_matches = [
                row for row in one_rows
                if row.get("task") == task
                and _int_from_row(row, "n_clusters") == c_value
                and abs(_float_from_row(row, "omega") - omega) <= 1e-12
            ]
            if not one_matches:
                raise ValueError(f"No matching one-step row for {task}, C={c_value}, omega={omega:g}.")
            one = one_matches[0]

            points.append(
                SelectedClusterPoint(
                    task=task,
                    n_clusters=c_value,
                    omega=omega,
                    e1=_float_from_row(one, "test_error"),
                    r2_1=_float_from_row(one, "test_r2"),
                    e50=_float_from_row(best, "relative_error"),
                    r2_50=_float_from_row(best, "association_r2"),
                )
            )
    return points


def write_plot_data(points: Sequence[SelectedClusterPoint], args: CliArgs) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "figure7_plot_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "task",
            "task_label",
            "C",
            "selected_omega",
            "E1",
            "R2_1",
            f"E{args.selection_horizon}",
            f"R2_{args.selection_horizon}",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in sorted(points, key=lambda item: (item.task, item.n_clusters)):
            writer.writerow(
                {
                    "task": point.task,
                    "task_label": TASK_LABELS.get(point.task, point.task),
                    "C": str(point.n_clusters),
                    "selected_omega": f"{point.omega:.17g}",
                    "E1": f"{point.e1:.17g}",
                    "R2_1": f"{point.r2_1:.17g}",
                    f"E{args.selection_horizon}": f"{point.e50:.17g}",
                    f"R2_{args.selection_horizon}": f"{point.r2_50:.17g}",
                }
            )
    return path


def make_figure(points: Sequence[SelectedClusterPoint], args: CliArgs) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.1, 4.6))

    for task in args.tasks:
        task_points = sorted([point for point in points if point.task == task], key=lambda item: item.n_clusters)
        xs = [point.n_clusters for point in task_points]

        for horizon in (1, args.selection_horizon):
            ys = [point.e1 if horizon == 1 else point.e50 for point in task_points]
            marker = MARKERS.get((task, horizon), "o")
            linestyle = LINESTYLES.get(horizon, "-")
            label = f"{TASK_LABELS.get(task, task)} $E_{{{horizon}}}$"
            ax.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=4.8,
                label=label,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of regimes $C$")
    ax.set_ylabel("Relative association error")
    ax.set_title("Classic Control regime-count sensitivity")
    ax.set_xticks(list(args.clusters))
    ax.set_xticklabels([str(value) for value in args.clusters])
    ax.grid(True, which="major", linewidth=0.5, alpha=0.30)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)
    ax.legend(frameon=False, fontsize=8.5, loc=args.legend_loc)
    fig.tight_layout()

    pdf_path = args.figure_dir / "fig_ai_cluster_sensitivity.pdf"
    png_path = args.output_dir / "fig_ai_cluster_sensitivity.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_metadata(args: CliArgs, data_path: Path, pdf_path: Path, png_path: Path) -> Path:
    metadata = {
        "figure": "Figure 7",
        "manuscript_file": "figures/fig_ai_cluster_sensitivity.pdf",
        "source_experiment": args.experiment_script,
        "source_one_step_summary": str(one_step_path(args)),
        "source_multistep_summary": str(multistep_path(args)),
        "data_csv": str(data_path),
        "pdf": str(pdf_path),
        "png_preview": str(png_path),
        "selection_rule": f"For each task and C, select omega minimizing kahkm_nlms relative_error at h={args.selection_horizon}; plot that configuration's E1 and E{args.selection_horizon}.",
        "settings": {
            "tasks": list(args.tasks),
            "clusters": list(args.clusters),
            "omegas": list(args.omegas),
            "horizons": list(args.horizons),
            "selection_horizon": args.selection_horizon,
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "max_steps": args.max_steps,
            "train_seed": args.train_seed,
            "test_seed": args.test_seed,
            "random_state": args.random_state,
            "nb": args.nb,
            "subspace_dim": args.subspace_dim,
            "beta": args.beta,
            "nlms_epochs": args.nlms_epochs,
            "tau": args.tau,
            "kmeans_kind": args.kmeans_kind,
        },
    }
    path = args.output_dir / "figure7_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


def main() -> None:
    args = parse_args()
    run_experiment(args)
    points = select_points(args)
    data_path = write_plot_data(points, args)
    pdf_path, png_path = make_figure(points, args)
    metadata_path = write_metadata(args, data_path, pdf_path, png_path)

    print("Wrote:")
    for path in (data_path, pdf_path, png_path, metadata_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
