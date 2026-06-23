#!/usr/bin/env python3
"""Generate manuscript Figure 4 from Experiment 14 training-trajectory-count outputs.

Figure 4 in the manuscript is:

    figures/fig_train_count.pdf

It is derived from Experiment 14:

    experiment_14_training_trajectory_count.py

Place this script in the same directory as the experiment scripts and run:

    python step_18_generate_figure4_train_count.py

By default, the script runs/reuses the default unseen-trajectory generalization
diagnostic:

    systems          = duffing vanderpol
    train counts     = 1 2 3 5
    train pool seeds = 0 1 2 3 4
    test seeds       = 100 101 102
    horizons         = 1 10 50 100 200

The manuscript plot is intentionally compact and shows the long-horizon
association error E_200 versus the number of training trajectories. The full
summary for all horizons is preserved in figure4_plot_data.csv.

Outputs:
    figures/fig_train_count.pdf
    kahkm_figure4_train_count/fig_train_count.png
    kahkm_figure4_train_count/figure4_plot_data.csv
    kahkm_figure4_train_count/figure4_metadata.json
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


DEFAULT_SYSTEMS: Final[tuple[str, ...]] = ("duffing", "vanderpol")
DEFAULT_TRAIN_POOL_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
DEFAULT_TRAIN_COUNTS: Final[tuple[int, ...]] = (1, 2, 3, 5)
DEFAULT_TEST_SEEDS: Final[tuple[int, ...]] = (100, 101, 102)
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 10, 50, 100, 200)

SYSTEM_LABELS: Final[dict[str, str]] = {
    "duffing": "Duffing",
    "vanderpol": "Van der Pol",
}

MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D", "v", "P", "X")


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    experiment_script: str
    experiment_output_dir: Path
    output_dir: Path
    figure_dir: Path
    python_executable: str
    no_run: bool
    skip_existing: bool
    systems: tuple[str, ...]
    train_pool_seeds: tuple[int, ...]
    train_counts: tuple[int, ...]
    test_seeds: tuple[int, ...]
    horizons: tuple[int, ...]
    plot_horizon: int
    n_steps_train: int
    n_steps_test: int
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int | None
    vanderpol_mu: float
    legend_loc: str


@dataclass(frozen=True)
class PlotRow:
    system: str
    train_count: int
    train_seeds_used: str
    horizon: int
    error_mean: float
    error_std: float
    r2_mean: float
    r2_std: float
    n_test_trajectories: int


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def _parse_system_tuple(values: Sequence[str]) -> tuple[str, ...]:
    parsed = tuple(str(value).strip().lower() for value in values)
    allowed = set(DEFAULT_SYSTEMS)
    invalid = [value for value in parsed if value not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown system(s): {invalid}. Expected subset of {sorted(allowed)}.")
    if not parsed:
        raise argparse.ArgumentTypeError("At least one system is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 4 from Experiment 14 training-count outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-script", default="experiment_14_training_trajectory_count.py")
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=Path("kahkm_experiment_14_outputs"),
        help="Output directory used by Experiment 14.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_figure4_train_count"),
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
        help="Skip Experiment 14 if the required summary CSV already exists. Default: enabled.",
    )

    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--train-pool-seeds", nargs="+", default=[str(v) for v in DEFAULT_TRAIN_POOL_SEEDS])
    parser.add_argument("--train-counts", nargs="+", default=[str(v) for v in DEFAULT_TRAIN_COUNTS])
    parser.add_argument("--test-seeds", nargs="+", default=[str(v) for v in DEFAULT_TEST_SEEDS])
    parser.add_argument("--horizons", nargs="+", default=[str(v) for v in DEFAULT_HORIZONS])
    parser.add_argument(
        "--plot-horizon",
        type=int,
        default=200,
        help="Horizon shown in the manuscript figure. Default: 200.",
    )
    parser.add_argument("--n-steps-train", type=int, default=1200)
    parser.add_argument("--n-steps-test", type=int, default=800)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", type=str, default="full")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=None)
    parser.add_argument("--vanderpol-mu", type=float, default=1.0)
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
    plot_horizon = int(ns.plot_horizon)
    if plot_horizon not in horizons:
        raise ValueError("--plot-horizon must be one of --horizons.")

    return CliArgs(
        source_dir=source_dir,
        experiment_script=str(ns.experiment_script),
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        figure_dir=figure_dir,
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        systems=_parse_system_tuple(tuple(str(v) for v in ns.systems)),
        train_pool_seeds=_parse_int_tuple(tuple(str(v) for v in ns.train_pool_seeds)),
        train_counts=_parse_int_tuple(tuple(str(v) for v in ns.train_counts)),
        test_seeds=_parse_int_tuple(tuple(str(v) for v in ns.test_seeds)),
        horizons=horizons,
        plot_horizon=plot_horizon,
        n_steps_train=int(ns.n_steps_train),
        n_steps_test=int(ns.n_steps_test),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        kmeans_kind=str(ns.kmeans_kind),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        max_train_per_cluster=None if ns.max_train_per_cluster is None else int(ns.max_train_per_cluster),
        vanderpol_mu=float(ns.vanderpol_mu),
        legend_loc=str(ns.legend_loc),
    )


def summary_path(args: CliArgs) -> Path:
    return args.experiment_output_dir / "experiment_14_test_summary.csv"


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return

    expected = summary_path(args)
    if args.skip_existing and expected.exists():
        print(f"Skipping Experiment 14 because summary already exists: {expected}")
        return

    script_path = args.source_dir / args.experiment_script
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find experiment script: {script_path}")

    command = [
        args.python_executable,
        str(script_path),
        "--output-dir",
        str(args.experiment_output_dir),
        "--systems",
        *args.systems,
        "--train-pool-seeds",
        *[str(seed) for seed in args.train_pool_seeds],
        "--train-counts",
        *[str(count) for count in args.train_counts],
        "--test-seeds",
        *[str(seed) for seed in args.test_seeds],
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
        "--n-steps-train",
        str(args.n_steps_train),
        "--n-steps-test",
        str(args.n_steps_test),
        "--subspace-dim",
        str(args.subspace_dim),
        "--nb",
        str(args.nb),
        "--beta",
        str(args.beta),
        "--nlms-epochs",
        str(args.nlms_epochs),
        "--kmeans-kind",
        args.kmeans_kind,
        "--batch-size",
        str(args.batch_size),
        "--n-jobs",
        str(args.n_jobs),
        "--vanderpol-mu",
        str(args.vanderpol_mu),
    ]
    if args.max_train_per_cluster is not None:
        command.extend(["--max-train-per-cluster", str(args.max_train_per_cluster)])

    print("Running Experiment 14:")
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


def load_plot_rows(args: CliArgs) -> list[PlotRow]:
    path = summary_path(args)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run without --no-run, or pass --experiment-output-dir to existing Experiment 14 outputs."
        )

    wanted_systems = set(args.systems)
    wanted_train_counts = set(args.train_counts)
    wanted_horizons = set(args.horizons)
    rows: list[PlotRow] = []

    for raw in _read_csv(path):
        system = str(raw.get("system", "")).lower()
        if system not in wanted_systems:
            continue
        train_count = int(float(str(raw.get("train_count", "nan"))))
        if train_count not in wanted_train_counts:
            continue
        horizon = int(float(str(raw.get("horizon", "nan"))))
        if horizon not in wanted_horizons:
            continue
        rows.append(
            PlotRow(
                system=system,
                train_count=train_count,
                train_seeds_used=str(raw.get("train_seeds_used", "")),
                horizon=horizon,
                error_mean=_finite_float(
                    raw.get("relative_association_error_mean"),
                    context=f"{system} train_count={train_count} horizon={horizon} error mean",
                ),
                error_std=_finite_float(
                    raw.get("relative_association_error_std", "0"),
                    context=f"{system} train_count={train_count} horizon={horizon} error std",
                ),
                r2_mean=_finite_float(
                    raw.get("association_r2_mean", "nan"),
                    context=f"{system} train_count={train_count} horizon={horizon} R2 mean",
                ),
                r2_std=_finite_float(
                    raw.get("association_r2_std", "0"),
                    context=f"{system} train_count={train_count} horizon={horizon} R2 std",
                ),
                n_test_trajectories=int(float(str(raw.get("n_test_trajectories", "0")))),
            )
        )

    expected = {
        (system, count, horizon)
        for system in args.systems
        for count in args.train_counts
        for horizon in args.horizons
    }
    observed = {(row.system, row.train_count, row.horizon) for row in rows}
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(f"Missing expected train-count summary rows: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return rows


def write_plot_data(rows: Sequence[PlotRow], args: CliArgs) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "figure4_plot_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "system",
            "system_label",
            "train_count",
            "train_seeds_used",
            "horizon",
            "error_mean",
            "error_std",
            "r2_mean",
            "r2_std",
            "n_test_trajectories",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.system, item.train_count, item.horizon)):
            writer.writerow(
                {
                    "system": row.system,
                    "system_label": SYSTEM_LABELS.get(row.system, row.system),
                    "train_count": str(row.train_count),
                    "train_seeds_used": row.train_seeds_used,
                    "horizon": str(row.horizon),
                    "error_mean": f"{row.error_mean:.17g}",
                    "error_std": f"{row.error_std:.17g}",
                    "r2_mean": f"{row.r2_mean:.17g}",
                    "r2_std": f"{row.r2_std:.17g}",
                    "n_test_trajectories": str(row.n_test_trajectories),
                }
            )
    return path


def make_figure(rows: Sequence[PlotRow], args: CliArgs) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    plot_rows = [row for row in rows if row.horizon == args.plot_horizon]

    for idx, system in enumerate(args.systems):
        series = sorted([row for row in plot_rows if row.system == system], key=lambda item: item.train_count)
        if not series:
            continue
        xs = [row.train_count for row in series]
        ys = [row.error_mean for row in series]
        ax.plot(
            xs,
            ys,
            marker=MARKERS[idx % len(MARKERS)],
            linewidth=2.0,
            markersize=5.0,
            label=f"{SYSTEM_LABELS.get(system, system)} $E_{{{args.plot_horizon}}}$",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of training trajectories")
    ax.set_ylabel(f"Relative association error $E_{{{args.plot_horizon}}}$")
    ax.set_title("Default unseen-trajectory generalization")
    ax.set_xticks(list(args.train_counts))
    ax.grid(True, which="major", linewidth=0.5, alpha=0.30)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)
    ax.legend(frameon=False, fontsize=8.5, loc=args.legend_loc)
    fig.tight_layout()

    pdf_path = args.figure_dir / "fig_train_count.pdf"
    png_path = args.output_dir / "fig_train_count.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_metadata(args: CliArgs, data_path: Path, pdf_path: Path, png_path: Path) -> Path:
    metadata = {
        "figure": "Figure 4",
        "manuscript_file": "figures/fig_train_count.pdf",
        "source_experiment": args.experiment_script,
        "source_summary": str(summary_path(args)),
        "data_csv": str(data_path),
        "pdf": str(pdf_path),
        "png_preview": str(png_path),
        "plot_horizon": args.plot_horizon,
        "settings": {
            "systems": list(args.systems),
            "train_pool_seeds": list(args.train_pool_seeds),
            "train_counts": list(args.train_counts),
            "test_seeds": list(args.test_seeds),
            "horizons": list(args.horizons),
            "n_steps_train": args.n_steps_train,
            "n_steps_test": args.n_steps_test,
            "subspace_dim": args.subspace_dim,
            "nb": args.nb,
            "beta": args.beta,
            "nlms_epochs": args.nlms_epochs,
            "kmeans_kind": args.kmeans_kind,
            "vanderpol_mu": args.vanderpol_mu,
        },
        "interpretation": "Default diagnostic plot of E200 versus the number of training trajectories for Duffing and Van der Pol.",
    }
    path = args.output_dir / "figure4_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


def main() -> None:
    args = parse_args()
    run_experiment(args)
    rows = load_plot_rows(args)
    data_path = write_plot_data(rows, args)
    pdf_path, png_path = make_figure(rows, args)
    metadata_path = write_metadata(args, data_path, pdf_path, png_path)

    print("Wrote:")
    for path in (data_path, pdf_path, png_path, metadata_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
