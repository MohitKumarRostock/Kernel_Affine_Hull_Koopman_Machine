#!/usr/bin/env python3
"""Generate a cleaner manuscript Figure 1 from Duffing abstraction-ablation outputs.

Figure 1 in the manuscript is:

    figures/fig_duffing_ablation.pdf

It is derived from Experiment 06:

    experiment_06_duffing_abstraction_ablation.py

Place this script in the same directory as the experiment scripts and run:

    python step_15_generate_figure1_duffing_ablation_CLEAN_LOWER_RIGHT.py

By default the script uses the manuscript/default diagnostic settings for Figure 1:

    C = 15
    omega = 12
    horizons = 1 2 5 10 20 50 100 200
    random states = 0 1 2

The script will run Experiment 06 unless --no-run is used or --skip-existing
finds an existing multistep summary.

Outputs:
    figures/fig_duffing_ablation.pdf
    kahkm_figure1_duffing_ablation_clean/fig_duffing_ablation_clean.pdf
    kahkm_figure1_duffing_ablation_clean/fig_duffing_ablation_clean.png
    kahkm_figure1_duffing_ablation/figure1_plot_data.csv
    kahkm_figure1_duffing_ablation/figure1_metadata.json
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


DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 2, 5, 10, 20, 50, 100, 200)
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

ABSTRACTION_LABELS: Final[dict[str, str]] = {
    "kahkm_folding_nlms": "KAHKM folding",
    "kmeans_rbf_nlms": "KMeans RBF",
    "kmeans_distance_nlms": "KMeans distance",
    "kmeans_hard_nlms": "KMeans hard",
}

ABSTRACTION_ORDER: Final[tuple[str, ...]] = (
    "kahkm_folding_nlms",
    "kmeans_rbf_nlms",
    "kmeans_distance_nlms",
    "kmeans_hard_nlms",
)

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
    n_clusters: int
    omega: float
    horizons: tuple[int, ...]
    seeds: tuple[int, ...]
    n_steps: int
    dt: float
    train_fraction: float
    subspace_dim: int
    nb: int
    tau: float
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int | None
    rbf_sigma_multiplier: float
    distance_scale_percentile: float
    uncertainty: str
    legend_outside: bool


@dataclass(frozen=True)
class PlotRow:
    abstraction: str
    horizon: int
    error_mean: float
    error_std: float
    r2_mean: float
    n_runs: int


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 1 from Duffing abstraction-ablation results."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-script", default="experiment_06_duffing_abstraction_ablation.py")
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=Path("kahkm_exp06_duffing_ablation"),
        help="Output directory used by Experiment 06.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_figure1_duffing_ablation_clean"),
        help="Directory for plot data, clean PDF/PNG preview, and metadata.",
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
        help="Skip running Experiment 06 if the required summary CSV already exists.",
    )

    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--omega", type=float, default=12.0)
    parser.add_argument("--horizons", nargs="+", default=[str(v) for v in DEFAULT_HORIZONS])
    parser.add_argument("--random-states", nargs="+", default=[str(v) for v in DEFAULT_SEEDS])
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", default="full", choices=("auto", "full", "minibatch"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=None)
    parser.add_argument("--rbf-sigma-multiplier", type=float, default=1.0)
    parser.add_argument("--distance-scale-percentile", type=float, default=95.0)
    parser.add_argument(
        "--uncertainty",
        choices=("none", "band", "errorbar"),
        default="none",
        help=(
            "How to display cross-seed uncertainty. Default 'none' keeps the manuscript figure clean; "
            "the standard deviations remain in figure1_plot_data.csv. Use 'band' or 'errorbar' for diagnostics."
        ),
    )
    parser.add_argument(
        "--legend-outside",
        action="store_true",
        default=False,
        help="Place the legend above the axes. Default: disabled; the manuscript version uses a lower-right legend.",
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

    return CliArgs(
        source_dir=source_dir,
        experiment_script=str(ns.experiment_script),
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        figure_dir=figure_dir,
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        n_clusters=int(ns.n_clusters),
        omega=float(ns.omega),
        horizons=_parse_int_tuple(tuple(str(v) for v in ns.horizons)),
        seeds=_parse_int_tuple(tuple(str(v) for v in ns.random_states)),
        n_steps=int(ns.n_steps),
        dt=float(ns.dt),
        train_fraction=float(ns.train_fraction),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        tau=float(ns.tau),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        kmeans_kind=str(ns.kmeans_kind),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        max_train_per_cluster=None if ns.max_train_per_cluster is None else int(ns.max_train_per_cluster),
        rbf_sigma_multiplier=float(ns.rbf_sigma_multiplier),
        distance_scale_percentile=float(ns.distance_scale_percentile),
        uncertainty=str(ns.uncertainty),
        legend_outside=bool(ns.legend_outside),
    )


def summary_path(args: CliArgs) -> Path:
    return args.experiment_output_dir / "experiment_06_multistep_summary.csv"


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return

    expected = summary_path(args)
    if args.skip_existing and expected.exists():
        print(f"Skipping Experiment 06 because summary already exists: {expected}")
        return

    script_path = args.source_dir / args.experiment_script
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find experiment script: {script_path}")

    command = [
        args.python_executable,
        str(script_path),
        "--output-dir",
        str(args.experiment_output_dir),
        "--n-clusters",
        str(args.n_clusters),
        "--omega",
        str(args.omega),
        "--horizons",
        *[str(h) for h in args.horizons],
        "--random-states",
        *[str(seed) for seed in args.seeds],
        "--n-steps",
        str(args.n_steps),
        "--dt",
        str(args.dt),
        "--train-fraction",
        str(args.train_fraction),
        "--subspace-dim",
        str(args.subspace_dim),
        "--nb",
        str(args.nb),
        "--tau",
        str(args.tau),
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
        "--rbf-sigma-multiplier",
        str(args.rbf_sigma_multiplier),
        "--distance-scale-percentile",
        str(args.distance_scale_percentile),
    ]
    if args.max_train_per_cluster is not None:
        command.extend(["--max-train-per-cluster", str(args.max_train_per_cluster)])

    print("Running Experiment 06:")
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
            f"Missing {path}. Run without --no-run, or pass --experiment-output-dir to existing Experiment 06 outputs."
        )

    rows: list[PlotRow] = []
    horizon_set = set(args.horizons)
    for raw in _read_csv(path):
        abstraction = str(raw.get("abstraction", ""))
        horizon = int(float(str(raw.get("horizon", "nan"))))
        if horizon not in horizon_set:
            continue
        rows.append(
            PlotRow(
                abstraction=abstraction,
                horizon=horizon,
                error_mean=_finite_float(
                    raw.get("relative_association_error_mean"),
                    context=f"{abstraction} horizon {horizon} error mean",
                ),
                error_std=_finite_float(
                    raw.get("relative_association_error_std", "0"),
                    context=f"{abstraction} horizon {horizon} error std",
                ),
                r2_mean=_finite_float(
                    raw.get("association_r2_mean", "nan"),
                    context=f"{abstraction} horizon {horizon} R2 mean",
                ),
                n_runs=int(float(str(raw.get("n_runs", "0")))),
            )
        )

    if not rows:
        raise ValueError(f"No usable rows found in {path}")

    abstractions = {row.abstraction for row in rows}
    missing = [name for name in ABSTRACTION_ORDER if name not in abstractions]
    if missing:
        print(f"Warning: missing expected abstractions in {path}: {missing}")

    return rows


def write_plot_data(rows: Sequence[PlotRow], args: CliArgs) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "figure1_plot_data.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["abstraction", "label", "horizon", "error_mean", "error_std", "r2_mean", "n_runs"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                ABSTRACTION_ORDER.index(item.abstraction) if item.abstraction in ABSTRACTION_ORDER else 999,
                item.horizon,
            ),
        ):
            writer.writerow(
                {
                    "abstraction": row.abstraction,
                    "label": ABSTRACTION_LABELS.get(row.abstraction, row.abstraction),
                    "horizon": str(row.horizon),
                    "error_mean": f"{row.error_mean:.17g}",
                    "error_std": f"{row.error_std:.17g}",
                    "r2_mean": f"{row.r2_mean:.17g}",
                    "n_runs": str(row.n_runs),
                }
            )
    return output


def make_figure(rows: Sequence[PlotRow], args: CliArgs) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    # Clean manuscript version: mean curves are emphasized, uncertainty remains
    # available in figure1_plot_data.csv. Error bars can be restored by passing
    # --uncertainty errorbar; bands can be restored with --uncertainty band.
    fig, ax = plt.subplots(figsize=(7.3, 4.8))

    row_by_abs: dict[str, list[PlotRow]] = {}
    for row in rows:
        row_by_abs.setdefault(row.abstraction, []).append(row)

    ordered_names = [name for name in ABSTRACTION_ORDER if name in row_by_abs]
    ordered_names.extend(sorted(name for name in row_by_abs if name not in set(ordered_names)))

    for idx, abstraction in enumerate(ordered_names):
        series = sorted(row_by_abs[abstraction], key=lambda item: item.horizon)
        xs = [item.horizon for item in series]
        ys = [item.error_mean for item in series]
        yerr = [item.error_std for item in series]
        marker = MARKERS[idx % len(MARKERS)]
        label = ABSTRACTION_LABELS.get(abstraction, abstraction)

        if args.uncertainty == "errorbar":
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker=marker,
                linewidth=1.8,
                markersize=4.2,
                capsize=2.0,
                elinewidth=0.8,
                label=label,
            )
            continue

        (line,) = ax.plot(
            xs,
            ys,
            marker=marker,
            linewidth=2.0,
            markersize=4.5,
            label=label,
        )

        if args.uncertainty == "band":
            lower: list[float] = []
            upper: list[float] = []
            for mean_value, std_value in zip(ys, yerr):
                lower.append(max(mean_value - std_value, mean_value * 0.35, 1.0e-12))
                upper.append(max(mean_value + std_value, mean_value * 1.05))
            ax.fill_between(xs, lower, upper, alpha=0.12, linewidth=0, color=line.get_color())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Prediction horizon $h$")
    ax.set_ylabel("Relative association error $E_h$")
    ax.set_title("Duffing abstraction ablation")
    ax.grid(True, which="major", linewidth=0.5, alpha=0.30)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)

    if args.legend_outside:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=8.5,
            columnspacing=1.2,
            handlelength=1.8,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    else:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        fig.tight_layout()

    pdf_path = args.figure_dir / "fig_duffing_ablation.pdf"
    clean_pdf_path = args.output_dir / "fig_duffing_ablation_clean.pdf"
    png_path = args.output_dir / "fig_duffing_ablation_clean.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(clean_pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_metadata(args: CliArgs, data_path: Path, pdf_path: Path, png_path: Path) -> Path:
    metadata = {
        "figure": "Figure 1",
        "manuscript_file": "figures/fig_duffing_ablation.pdf",
        "source_experiment": args.experiment_script,
        "source_summary": str(summary_path(args)),
        "data_csv": str(data_path),
        "pdf": str(pdf_path),
        "png_preview": str(png_path),
        "settings": {
            "n_clusters": args.n_clusters,
            "omega": args.omega,
            "horizons": list(args.horizons),
            "seeds": list(args.seeds),
            "n_steps": args.n_steps,
            "dt": args.dt,
            "train_fraction": args.train_fraction,
            "subspace_dim": args.subspace_dim,
            "nb": args.nb,
            "tau": args.tau,
            "beta": args.beta,
            "nlms_epochs": args.nlms_epochs,
            "kmeans_kind": args.kmeans_kind,
            "uncertainty_display": args.uncertainty,
            "legend_outside": args.legend_outside,
        },
    }
    path = args.output_dir / "figure1_metadata.json"
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
