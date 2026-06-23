#!/usr/bin/env python3
"""Generate manuscript Table 11 from tuned Classic Control Experiment 16 outputs.

Place this script in the same directory as the experiment scripts and run:

    python step_10_generate_table11_classic_control.py

By default, the script runs/reuses:

    run_exp16_tuned.py

Then it aggregates the per-seed Experiment 16 result folders:

    kahkm_exp16_cartpole_tuned_seed0/
    kahkm_exp16_cartpole_tuned_seed1/
    kahkm_exp16_cartpole_tuned_seed2/
    kahkm_exp16_mountaincar_tuned_seed0/
    kahkm_exp16_mountaincar_tuned_seed1/
    kahkm_exp16_mountaincar_tuned_seed2/

Expected per-seed input file:

    experiment_16_multistep_results.csv

Outputs are written to:

    kahkm_table11_classic_control/table11_classic_control_values.csv
    kahkm_table11_classic_control/table11_classic_control_values.md
    kahkm_table11_classic_control/table11_classic_control_values.tex
    kahkm_table11_classic_control/table11_classic_control_metadata.json

Use --no-run to only format already generated Experiment 16 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence


TASK_CONFIGS: Final[dict[str, tuple[int, float]]] = {
    "CartPole-v1": (20, 1.0),
    "MountainCar-v0": (100, 1.0),
}
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 5, 10, 20, 50)


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    experiment_output_dir: Path
    output_dir: Path
    python_executable: str
    runner_script: str
    no_run: bool
    skip_existing: bool
    seeds: tuple[int, ...]
    horizons: tuple[int, ...]


@dataclass(frozen=True)
class SeedMetrics:
    task: str
    seed_label: str
    e1: float
    r2_1: float
    e50: float
    r2_50: float
    mean_e: float


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float


@dataclass(frozen=True)
class TableRow:
    task: str
    n_clusters: int
    omega: float
    e1: SummaryStats
    r2_1: SummaryStats
    e50: SummaryStats
    r2_50: SummaryStats
    mean_e: SummaryStats


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Table 11 from tuned Classic Control Experiment 16 outputs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing run_exp16_tuned.py and experiment scripts. Default: current directory.",
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing kahkm_exp16_*_tuned_seed* output folders. "
            "Default: --source-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table11_classic_control"),
        help="Output directory for generated table files.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable used to run the Experiment 16 wrapper.",
    )
    parser.add_argument(
        "--runner-script",
        default="run_exp16_tuned.py",
        help="Experiment 16 tuned runner script filename.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run the experiment wrapper; only aggregate existing outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Pass --skip-existing to the Experiment 16 wrapper. Default: enabled.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0", "1", "2"],
        help="Seeds expected in the tuned result directories. Default: 0 1 2.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=[str(value) for value in DEFAULT_HORIZONS],
        help="Horizons used for the mean-error column. Default: 1 5 10 20 50.",
    )
    namespace = parser.parse_args()
    source_dir = Path(namespace.source_dir).resolve()
    experiment_output_dir = Path(namespace.experiment_output_dir).resolve() if namespace.experiment_output_dir else source_dir
    output_dir = Path(namespace.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir
    return CliArgs(
        source_dir=source_dir,
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        python_executable=str(namespace.python_executable),
        runner_script=str(namespace.runner_script),
        no_run=bool(namespace.no_run),
        skip_existing=bool(namespace.skip_existing),
        seeds=_parse_int_tuple(namespace.seeds),
        horizons=_parse_int_tuple(namespace.horizons),
    )


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return
    runner_path = args.source_dir / args.runner_script
    if not runner_path.exists():
        raise FileNotFoundError(
            f"Could not find {runner_path}. Place this script in the experiment folder, "
            "or pass --source-dir/--runner-script."
        )
    command = [
        args.python_executable,
        str(runner_path),
        "--output-root",
        str(args.experiment_output_dir),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    print("Running Experiment 16 tuned wrapper:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return result


def _task_seed_dirs(root: Path, task: str, seeds: tuple[int, ...]) -> list[Path]:
    prefix = "cartpole" if task == "CartPole-v1" else "mountaincar"
    dirs: list[Path] = []
    for seed in seeds:
        expected = root / f"kahkm_exp16_{prefix}_tuned_seed{seed}"
        if expected.exists():
            dirs.append(expected)
    if dirs:
        return dirs
    pattern = f"kahkm_exp16_{prefix}_tuned_seed*"
    return sorted(path for path in root.glob(pattern) if path.is_dir())


def load_seed_metrics(args: CliArgs) -> list[SeedMetrics]:
    metrics: list[SeedMetrics] = []
    horizon_set = set(args.horizons)
    if 1 not in horizon_set or 50 not in horizon_set:
        raise ValueError("Table 11 requires horizons 1 and 50.")

    for task in TASK_CONFIGS:
        dirs = _task_seed_dirs(args.experiment_output_dir, task, args.seeds)
        if not dirs:
            raise FileNotFoundError(
                f"No tuned Experiment 16 output directories found for {task} in {args.experiment_output_dir}."
            )
        for run_dir in dirs:
            csv_path = run_dir / "experiment_16_multistep_results.csv"
            if not csv_path.exists():
                fallback = run_dir / "experiment_16_multistep_summary.csv"
                if fallback.exists():
                    csv_path = fallback
                else:
                    raise FileNotFoundError(f"Missing multistep CSV in {run_dir}")
            rows = _read_csv(csv_path)
            selected: dict[int, dict[str, str]] = {}
            for row in rows:
                if row.get("task") != task:
                    continue
                if row.get("method") != "kahkm_nlms":
                    continue
                horizon = int(float(row.get("horizon", "nan")))
                if horizon not in horizon_set:
                    continue
                selected[horizon] = row
            missing = sorted(horizon_set.difference(selected))
            if missing:
                raise ValueError(f"Missing KAHKM horizons {missing} in {csv_path}")
            e_values = [_safe_float(selected[h]["relative_error"]) for h in args.horizons]
            seed_label = run_dir.name.rsplit("seed", 1)[-1]
            metrics.append(
                SeedMetrics(
                    task=task,
                    seed_label=seed_label,
                    e1=_safe_float(selected[1]["relative_error"]),
                    r2_1=_safe_float(selected[1]["association_r2"]),
                    e50=_safe_float(selected[50]["relative_error"]),
                    r2_50=_safe_float(selected[50]["association_r2"]),
                    mean_e=float(statistics.fmean(e_values)),
                )
            )
    return metrics


def _sample_stats(values: Sequence[float]) -> SummaryStats:
    """Return mean and sample standard deviation across seed-level values."""
    if not values:
        raise ValueError("Cannot summarize an empty list.")
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return SummaryStats(mean=mean, std=std)


def summarize_metrics(metrics: Sequence[SeedMetrics]) -> list[TableRow]:
    rows: list[TableRow] = []
    for task, (n_clusters, omega) in TASK_CONFIGS.items():
        task_metrics = [item for item in metrics if item.task == task]
        if not task_metrics:
            raise ValueError(f"No metrics found for {task}")
        rows.append(
            TableRow(
                task=task,
                n_clusters=n_clusters,
                omega=omega,
                e1=_sample_stats([item.e1 for item in task_metrics]),
                r2_1=_sample_stats([item.r2_1 for item in task_metrics]),
                e50=_sample_stats([item.e50 for item in task_metrics]),
                r2_50=_sample_stats([item.r2_50 for item in task_metrics]),
                mean_e=_sample_stats([item.mean_e for item in task_metrics]),
            )
        )
    return rows


def _format_pm(stats: SummaryStats, decimals: int) -> str:
    return f"{stats.mean:.{decimals}f}±{stats.std:.{decimals}f}"


def _format_omega(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _decimals_for_error(stats: SummaryStats) -> int:
    return 5 if abs(stats.mean) < 0.01 else 4


def table_row_to_csv(row: TableRow) -> dict[str, str]:
    return {
        "task": row.task,
        "C": str(row.n_clusters),
        "omega": _format_omega(row.omega),
        "E1": _format_pm(row.e1, _decimals_for_error(row.e1)),
        "R2_1": _format_pm(row.r2_1, 3),
        "E50": _format_pm(row.e50, _decimals_for_error(row.e50)),
        "R2_50": _format_pm(row.r2_50, 3),
        "mean_E_1_50": _format_pm(row.mean_e, _decimals_for_error(row.mean_e)),
        "E1_mean": f"{row.e1.mean:.17g}",
        "E1_std": f"{row.e1.std:.17g}",
        "R2_1_mean": f"{row.r2_1.mean:.17g}",
        "R2_1_std": f"{row.r2_1.std:.17g}",
        "E50_mean": f"{row.e50.mean:.17g}",
        "E50_std": f"{row.e50.std:.17g}",
        "R2_50_mean": f"{row.r2_50.mean:.17g}",
        "R2_50_std": f"{row.r2_50.std:.17g}",
        "mean_E_1_50_mean": f"{row.mean_e.mean:.17g}",
        "mean_E_1_50_std": f"{row.mean_e.std:.17g}",
    }


def _latex_pm(stats: SummaryStats, decimals: int) -> str:
    return f"${stats.mean:.{decimals}f}\\pm{stats.std:.{decimals}f}$"


def table_row_to_latex_cells(row: TableRow) -> list[str]:
    return [
        row.task,
        str(row.n_clusters),
        _format_omega(row.omega),
        _latex_pm(row.e1, _decimals_for_error(row.e1)),
        _latex_pm(row.r2_1, 3),
        _latex_pm(row.e50, _decimals_for_error(row.e50)),
        _latex_pm(row.r2_50, 3),
        _latex_pm(row.mean_e, _decimals_for_error(row.mean_e)),
    ]


def write_table_files(rows: Sequence[TableRow], metrics: Sequence[SeedMetrics], args: CliArgs) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = [table_row_to_csv(row) for row in rows]
    csv_path = args.output_dir / "table11_classic_control_values.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path = args.output_dir / "table11_classic_control_values.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Task | C | omega | E1 | R2_1 | E50 | R2_50 | mean E1:50 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in csv_rows:
            handle.write(
                f"| {row['task']} | {row['C']} | {row['omega']} | {row['E1']} | "
                f"{row['R2_1']} | {row['E50']} | {row['R2_50']} | {row['mean_E_1_50']} |\n"
            )

    tex_path = args.output_dir / "table11_classic_control_values.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by step_10_generate_table11_classic_control.py\n")
        handle.write("\\begin{table}[t]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write(
            "\\caption{Tuned KAHKM Classic Control closure results. Configurations were selected by mean relative association error over "
            "$h\\in\\{1,5,10,20,50\\}$ using $\\beta=0.1$, 20 NLMS epochs, eight training episodes, four test episodes, and three seeds. "
            "Entries are mean $\\pm$ standard deviation across seeds.}\n"
        )
        handle.write("\\label{tab:classic_control_c}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{lrrccccc}\n")
        handle.write("\\toprule\n")
        handle.write("Task & $C$ & $\\omega$ & $E_1$ & $R^2_1$ & $E_{50}$ & $R^2_{50}$ & $\\overline{E}_{1:50}$ \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(table_row_to_latex_cells(row)) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}%\n")
        handle.write("}\n")
        handle.write("\\end{table}\n")

    metadata = {
        "source_dir": str(args.source_dir),
        "experiment_output_dir": str(args.experiment_output_dir),
        "runner_script": args.runner_script,
        "no_run": args.no_run,
        "seeds": list(args.seeds),
        "horizons": list(args.horizons),
        "method": "kahkm_nlms",
        "std_convention": "sample standard deviation across seed-level values, ddof=1",
        "seed_level_records": [item.__dict__ for item in metrics],
    }
    with (args.output_dir / "table11_classic_control_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Wrote:")
    for path in (csv_path, md_path, tex_path, args.output_dir / "table11_classic_control_metadata.json"):
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    run_experiment(args)
    metrics = load_seed_metrics(args)
    rows = summarize_metrics(metrics)
    write_table_files(rows, metrics, args)


if __name__ == "__main__":
    main()
