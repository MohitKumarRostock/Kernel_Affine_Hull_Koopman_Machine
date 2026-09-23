#!/usr/bin/env python3
"""Generate manuscript Figure 6 from the default Van der Pol external-baseline diagnostic.

Figure 6 in the manuscript is:

    figures/fig_external_vanderpol.pdf

It is derived from Experiment 15:

    experiment_15_external_koopman_baselines.py

This figure is the default diagnostic plot, not the tuned Table 10 comparison.
The tuned Table 10 Van der Pol values use the retained-variation/noise-aware setting C=25, omega=4; this figure uses the
Experiment 15 default Van der Pol abstraction, C=20, omega=4, matching the current
manuscript caption.

Place this script in the same directory as the experiment scripts and run:

    python step_20_generate_figure6_external_vanderpol.py

Outputs:
    figures/fig_external_vanderpol.pdf
    kahkm_figure6_external_vanderpol/fig_external_vanderpol.png
    kahkm_figure6_external_vanderpol/figure6_plot_data.csv
    kahkm_figure6_external_vanderpol/figure6_metadata.json
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


DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 10, 50, 100, 200)
DEFAULT_TRAIN_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
DEFAULT_TEST_SEEDS: Final[tuple[int, ...]] = (100, 101, 102)

METHOD_ORDER: Final[tuple[str, ...]] = (
    "kahkm_nlms",
    "state_dmd",
    "state_edmd_poly2",
    "state_edmd_poly3",
)

METHOD_LABELS: Final[dict[str, str]] = {
    "kahkm_nlms": "KAHKM + NLMS",
    "state_dmd": "State DMD",
    "state_edmd_poly2": "State EDMD poly2",
    "state_edmd_poly3": "State EDMD poly3",
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
    train_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    horizons: tuple[int, ...]
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
    ridge: float
    vanderpol_mu: float
    legend_loc: str


@dataclass(frozen=True)
class PlotRow:
    method: str
    horizon: int
    error_mean: float
    error_std: float
    r2_mean: float
    r2_std: float
    n_runs: int


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 6 from the default Van der Pol external-baseline diagnostic."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-script", default="experiment_15_external_koopman_baselines.py")
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=Path("kahkm_exp15_default_external_vanderpol"),
        help="Output directory used by Experiment 15 for this default Van der Pol figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_figure6_external_vanderpol"),
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
        help="Skip Experiment 15 if the required summary CSV already exists. Default: enabled.",
    )

    parser.add_argument("--train-seeds", nargs="+", default=[str(v) for v in DEFAULT_TRAIN_SEEDS])
    parser.add_argument("--test-seeds", nargs="+", default=[str(v) for v in DEFAULT_TEST_SEEDS])
    parser.add_argument("--horizons", nargs="+", default=[str(v) for v in DEFAULT_HORIZONS])
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
    parser.add_argument("--ridge", type=float, default=1e-8)
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

    return CliArgs(
        source_dir=source_dir,
        experiment_script=str(ns.experiment_script),
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        figure_dir=figure_dir,
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        train_seeds=_parse_int_tuple(tuple(str(v) for v in ns.train_seeds)),
        test_seeds=_parse_int_tuple(tuple(str(v) for v in ns.test_seeds)),
        horizons=_parse_int_tuple(tuple(str(v) for v in ns.horizons)),
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
        ridge=float(ns.ridge),
        vanderpol_mu=float(ns.vanderpol_mu),
        legend_loc=str(ns.legend_loc),
    )


def summary_path(args: CliArgs) -> Path:
    return args.experiment_output_dir / "experiment_15_test_summary.csv"


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return

    expected = summary_path(args)
    if args.skip_existing and expected.exists():
        print(f"Skipping Experiment 15 because summary already exists: {expected}")
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
        "vanderpol",
        "--train-seeds",
        *[str(seed) for seed in args.train_seeds],
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
        "--ridge",
        str(args.ridge),
    ]
    if args.max_train_per_cluster is not None:
        command.extend(["--max-train-per-cluster", str(args.max_train_per_cluster)])

    print("Running Experiment 15 Van der Pol default external-baseline diagnostic:")
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


def _row_value(row: dict[str, str], names: Sequence[str], *, context: str) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            return _finite_float(row[name], context=f"{context} column {name}")
    raise ValueError(f"Missing any of columns {list(names)} for {context}. Available columns: {sorted(row)}")


def load_plot_rows(args: CliArgs) -> list[PlotRow]:
    path = summary_path(args)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run without --no-run, or pass --experiment-output-dir to existing Experiment 15 outputs."
        )

    wanted_horizons = set(args.horizons)
    rows: list[PlotRow] = []
    for raw in _read_csv(path):
        if str(raw.get("system", "")).lower() != "vanderpol":
            continue
        method = str(raw.get("method", ""))
        if method not in METHOD_ORDER:
            continue
        horizon = int(float(str(raw.get("horizon", "nan"))))
        if horizon not in wanted_horizons:
            continue
        rows.append(
            PlotRow(
                method=method,
                horizon=horizon,
                error_mean=_row_value(
                    raw,
                    ("relative_association_error_mean", "association_error_mean", "relative_error_mean"),
                    context=f"{method} h={horizon} error mean",
                ),
                error_std=_row_value(
                    raw,
                    ("relative_association_error_std", "association_error_std", "relative_error_std"),
                    context=f"{method} h={horizon} error std",
                ),
                r2_mean=_row_value(
                    raw,
                    ("association_r2_mean", "r2_mean", "test_r2_mean"),
                    context=f"{method} h={horizon} R2 mean",
                ),
                r2_std=_row_value(
                    raw,
                    ("association_r2_std", "r2_std", "test_r2_std"),
                    context=f"{method} h={horizon} R2 std",
                ),
                n_runs=int(float(str(raw.get("n_runs", raw.get("n_test_trajectories", "0"))))),
            )
        )

    expected = {(method, horizon) for method in METHOD_ORDER for horizon in args.horizons}
    observed = {(row.method, row.horizon) for row in rows}
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(f"Missing expected Van der Pol external-baseline rows: {missing}")

    return rows


def write_plot_data(rows: Sequence[PlotRow], args: CliArgs) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "figure6_plot_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "method_label",
            "horizon",
            "error_mean",
            "error_std",
            "r2_mean",
            "r2_std",
            "n_runs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (METHOD_ORDER.index(item.method), item.horizon),
        ):
            writer.writerow(
                {
                    "method": row.method,
                    "method_label": METHOD_LABELS.get(row.method, row.method),
                    "horizon": str(row.horizon),
                    "error_mean": f"{row.error_mean:.17g}",
                    "error_std": f"{row.error_std:.17g}",
                    "r2_mean": f"{row.r2_mean:.17g}",
                    "r2_std": f"{row.r2_std:.17g}",
                    "n_runs": str(row.n_runs),
                }
            )
    return path


def make_figure(rows: Sequence[PlotRow], args: CliArgs) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for idx, method in enumerate(METHOD_ORDER):
        series = sorted([row for row in rows if row.method == method], key=lambda item: item.horizon)
        xs = [row.horizon for row in series]
        ys = [row.error_mean for row in series]
        ax.plot(
            xs,
            ys,
            marker=MARKERS[idx % len(MARKERS)],
            linewidth=2.0,
            markersize=4.8,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Prediction horizon $h$")
    ax.set_ylabel("Relative association error $E_h$")
    ax.set_title("Van der Pol external Koopman baselines")
    ax.grid(True, which="major", linewidth=0.5, alpha=0.30)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)
    ax.legend(frameon=False, fontsize=8.5, loc=args.legend_loc)
    fig.tight_layout()

    pdf_path = args.figure_dir / "fig_external_vanderpol.pdf"
    png_path = args.output_dir / "fig_external_vanderpol.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_metadata(args: CliArgs, data_path: Path, pdf_path: Path, png_path: Path) -> Path:
    metadata = {
        "figure": "Figure 6",
        "manuscript_file": "figures/fig_external_vanderpol.pdf",
        "source_experiment": args.experiment_script,
        "source_summary": str(summary_path(args)),
        "data_csv": str(data_path),
        "pdf": str(pdf_path),
        "png_preview": str(png_path),
        "diagnostic": "default external-baseline plot",
        "note": "This figure uses Experiment 15 default Van der Pol KAHKM settings C=20, omega=4. Table 10 uses the retained-variation/noise-aware tuned Van der Pol setting C=25, omega=4.",
        "settings": {
            "system": "vanderpol",
            "default_n_clusters": 20,
            "default_omega": 4,
            "train_seeds": list(args.train_seeds),
            "test_seeds": list(args.test_seeds),
            "horizons": list(args.horizons),
            "n_steps_train": args.n_steps_train,
            "n_steps_test": args.n_steps_test,
            "subspace_dim": args.subspace_dim,
            "nb": args.nb,
            "beta": args.beta,
            "nlms_epochs": args.nlms_epochs,
            "kmeans_kind": args.kmeans_kind,
            "ridge": args.ridge,
            "vanderpol_mu": args.vanderpol_mu,
        },
    }
    path = args.output_dir / "figure6_metadata.json"
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
