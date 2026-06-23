#!/usr/bin/env python3
"""Generate manuscript Figure 3 from Experiment 12 default noise-robustness outputs.

Figure 3 in the manuscript is:

    figures/fig_noise_robustness.pdf

It is derived from Experiment 12:

    experiment_12_noise_robustness.py

Place this script in the same directory as the experiment scripts and run:

    python step_17_generate_figure3_noise_robustness_CLEAN_LOWER_RIGHT.py

By default, the script runs/reuses the default diagnostic noise experiment:

    systems       = duffing vanderpol
    noise levels  = 0 0.005 0.01 0.02 0.05
    random states = 0 1 2
    horizons      = 1 10 50 100 200

The manuscript plot is intentionally compact: it compares one-step and long-horizon
association error, E_1 and E_200, for Duffing and Van der Pol. The full multistep
summary remains available in the plot-data CSV.

Outputs:
    figures/fig_noise_robustness.pdf
    kahkm_figure3_noise_robustness/fig_noise_robustness.png
    kahkm_figure3_noise_robustness/figure3_plot_data.csv
    kahkm_figure3_noise_robustness/figure3_metadata.json
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


DEFAULT_NOISE_LEVELS: Final[tuple[float, ...]] = (0.0, 0.005, 0.01, 0.02, 0.05)
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 10, 50, 100, 200)
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
DEFAULT_SYSTEMS: Final[tuple[str, ...]] = ("duffing", "vanderpol")

SYSTEM_LABELS: Final[dict[str, str]] = {
    "duffing": "Duffing",
    "vanderpol": "Van der Pol",
}

HORIZON_LABELS: Final[dict[int, str]] = {
    1: "$E_1$",
    200: "$E_{200}$",
}

MARKERS: Final[dict[tuple[str, int], str]] = {
    ("duffing", 1): "o",
    ("duffing", 200): "s",
    ("vanderpol", 1): "^",
    ("vanderpol", 200): "D",
}

LINESTYLES: Final[dict[int, str]] = {
    1: "--",
    200: "-",
}


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
    noise_levels: tuple[float, ...]
    random_states: tuple[int, ...]
    horizons: tuple[int, ...]
    plot_horizons: tuple[int, ...]
    n_steps: int
    train_fraction: float
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
    noise_level: float
    horizon: int
    error_mean: float
    error_std: float
    r2_mean: float
    n_runs: int


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def _parse_float_tuple(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
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
        description="Generate manuscript Figure 3 from Experiment 12 default noise-robustness outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-script", default="experiment_12_noise_robustness.py")
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=Path("kahkm_experiment_12_outputs"),
        help="Output directory used by Experiment 12.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_figure3_noise_robustness"),
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
        help="Skip Experiment 12 if the required summary CSV already exists. Default: enabled.",
    )

    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--noise-levels", nargs="+", default=[str(v) for v in DEFAULT_NOISE_LEVELS])
    parser.add_argument("--random-states", nargs="+", default=[str(v) for v in DEFAULT_SEEDS])
    parser.add_argument("--horizons", nargs="+", default=[str(v) for v in DEFAULT_HORIZONS])
    parser.add_argument(
        "--plot-horizons",
        nargs="+",
        default=["1", "200"],
        help="Horizons shown in the compact manuscript plot. Default: 1 200.",
    )
    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", default="full")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=None)
    parser.add_argument("--vanderpol-mu", type=float, default=1.0)
    parser.add_argument(
        "--legend-loc",
        default="lower right",
        help="Matplotlib legend location. Default: lower right.",
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

    horizons = _parse_int_tuple(tuple(str(value) for value in ns.horizons))
    plot_horizons = _parse_int_tuple(tuple(str(value) for value in ns.plot_horizons))
    missing_plot_horizons = sorted(set(plot_horizons).difference(horizons))
    if missing_plot_horizons:
        raise ValueError(f"--plot-horizons must be included in --horizons; missing {missing_plot_horizons}")

    return CliArgs(
        source_dir=source_dir,
        experiment_script=str(ns.experiment_script),
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        figure_dir=figure_dir,
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        systems=_parse_system_tuple(tuple(str(value) for value in ns.systems)),
        noise_levels=_parse_float_tuple(tuple(str(value) for value in ns.noise_levels)),
        random_states=_parse_int_tuple(tuple(str(value) for value in ns.random_states)),
        horizons=horizons,
        plot_horizons=plot_horizons,
        n_steps=int(ns.n_steps),
        train_fraction=float(ns.train_fraction),
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
    return args.experiment_output_dir / "experiment_12_multistep_summary.csv"


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return

    expected = summary_path(args)
    if args.skip_existing and expected.exists():
        print(f"Skipping Experiment 12 because summary already exists: {expected}")
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
        "--noise-levels",
        *[str(value) for value in args.noise_levels],
        "--random-states",
        *[str(seed) for seed in args.random_states],
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
        "--n-steps",
        str(args.n_steps),
        "--train-fraction",
        str(args.train_fraction),
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

    print("Running Experiment 12:")
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
            f"Missing {path}. Run without --no-run, or pass --experiment-output-dir to existing Experiment 12 outputs."
        )

    wanted_systems = set(args.systems)
    wanted_horizons = set(args.horizons)
    wanted_noise = set(round(value, 12) for value in args.noise_levels)

    rows: list[PlotRow] = []
    for raw in _read_csv(path):
        system = str(raw.get("system", "")).lower()
        if system not in wanted_systems:
            continue
        horizon = int(float(str(raw.get("horizon", "nan"))))
        if horizon not in wanted_horizons:
            continue
        noise_level = round(_finite_float(raw.get("noise_level"), context="noise_level"), 12)
        if noise_level not in wanted_noise:
            continue
        rows.append(
            PlotRow(
                system=system,
                noise_level=float(noise_level),
                horizon=horizon,
                error_mean=_finite_float(
                    raw.get("relative_association_error_mean"),
                    context=f"{system} noise={noise_level:g} horizon={horizon} error mean",
                ),
                error_std=_finite_float(
                    raw.get("relative_association_error_std", "0"),
                    context=f"{system} noise={noise_level:g} horizon={horizon} error std",
                ),
                r2_mean=_finite_float(
                    raw.get("association_r2_mean", "nan"),
                    context=f"{system} noise={noise_level:g} horizon={horizon} R2 mean",
                ),
                n_runs=int(float(str(raw.get("n_runs", "0")))),
            )
        )

    expected = {(system, round(noise, 12), horizon) for system in args.systems for noise in args.noise_levels for horizon in args.horizons}
    observed = {(row.system, round(row.noise_level, 12), row.horizon) for row in rows}
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(f"Missing expected noise-summary rows: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return rows


def write_plot_data(rows: Sequence[PlotRow], args: CliArgs) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "figure3_plot_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "system",
            "system_label",
            "noise_level",
            "noise_percent",
            "horizon",
            "error_mean",
            "error_std",
            "r2_mean",
            "n_runs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.system, item.noise_level, item.horizon)):
            writer.writerow(
                {
                    "system": row.system,
                    "system_label": SYSTEM_LABELS.get(row.system, row.system),
                    "noise_level": f"{row.noise_level:.17g}",
                    "noise_percent": f"{100.0 * row.noise_level:.6g}",
                    "horizon": str(row.horizon),
                    "error_mean": f"{row.error_mean:.17g}",
                    "error_std": f"{row.error_std:.17g}",
                    "r2_mean": f"{row.r2_mean:.17g}",
                    "n_runs": str(row.n_runs),
                }
            )
    return path


def make_figure(rows: Sequence[PlotRow], args: CliArgs) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.3, 4.6))

    # Clean manuscript version: legend in the lower-right corner to reduce clutter.
    lookup: dict[tuple[str, int], list[PlotRow]] = {}
    for row in rows:
        if row.horizon in set(args.plot_horizons):
            lookup.setdefault((row.system, row.horizon), []).append(row)

    for system in args.systems:
        for horizon in args.plot_horizons:
            series = sorted(lookup.get((system, horizon), []), key=lambda item: item.noise_level)
            if not series:
                continue
            xs = [100.0 * item.noise_level for item in series]
            ys = [item.error_mean for item in series]
            marker = MARKERS.get((system, horizon), "o")
            linestyle = LINESTYLES.get(horizon, "-")
            label = f"{SYSTEM_LABELS.get(system, system)} {HORIZON_LABELS.get(horizon, f'E_{horizon}')}"
            ax.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=4.8,
                label=label,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Training noise level (% of coordinate standard deviation)")
    ax.set_ylabel("Relative association error $E_h$")
    ax.set_title("Default noise robustness diagnostic")
    ax.set_xticks([100.0 * value for value in args.noise_levels])
    ax.set_xticklabels([f"{100.0 * value:g}" for value in args.noise_levels])
    ax.grid(True, which="major", linewidth=0.5, alpha=0.30)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)
    ax.legend(frameon=False, fontsize=8.5, loc=args.legend_loc)
    fig.tight_layout()

    pdf_path = args.figure_dir / "fig_noise_robustness.pdf"
    png_path = args.output_dir / "fig_noise_robustness.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_metadata(args: CliArgs, data_path: Path, pdf_path: Path, png_path: Path) -> Path:
    metadata = {
        "figure": "Figure 3",
        "manuscript_file": "figures/fig_noise_robustness.pdf",
        "source_experiment": args.experiment_script,
        "source_summary": str(summary_path(args)),
        "data_csv": str(data_path),
        "pdf": str(pdf_path),
        "png_preview": str(png_path),
        "plot_horizons": list(args.plot_horizons),
        "settings": {
            "systems": list(args.systems),
            "noise_levels": list(args.noise_levels),
            "random_states": list(args.random_states),
            "horizons": list(args.horizons),
            "n_steps": args.n_steps,
            "train_fraction": args.train_fraction,
            "subspace_dim": args.subspace_dim,
            "nb": args.nb,
            "beta": args.beta,
            "nlms_epochs": args.nlms_epochs,
            "kmeans_kind": args.kmeans_kind,
            "vanderpol_mu": args.vanderpol_mu,
        },
        "interpretation": "Compact plot of E1 and E200 for default Duffing and Van der Pol abstractions under training-only observation noise.",
    }
    path = args.output_dir / "figure3_metadata.json"
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
