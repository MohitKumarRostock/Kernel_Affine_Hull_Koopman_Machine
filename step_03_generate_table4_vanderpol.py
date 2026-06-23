#!/usr/bin/env python3
"""Generate manuscript Table 4 values for the Van der Pol KAHKM tuning sweep.

Place this file in the same folder as the experiment scripts, especially:

    run_exp09_vanderpol_tuning.py
    experiment_09_vanderpol_sensitivity.py

The script can either run the underlying tuning runner or reuse existing results.
It then reads the Experiment 09 CSV outputs and emits the manuscript-style
Table 4 files.

Manuscript Table 4 definition
-----------------------------
System: Van der Pol, mu = 1, dt = 0.02.
Grid: C in {10, 15, 20, 25, 30, 40, 50} and omega in {0.5, 1, 2, 4, 6, 8, 12}.
Seeds: 0, 1, 2.
Selection objective: mean relative association error over horizons
{1, 10, 50, 100, 200}.
Displayed columns: rank, C, omega, mean E_h, E50, E100, E200.

Examples
--------
Run the full underlying experiment and generate Table 4:

    python step_03_generate_table4_vanderpol.py

Reuse existing Experiment 09 output only:

    python step_03_generate_table4_vanderpol.py --no-run

Use a custom source folder:

    python step_03_generate_table4_vanderpol.py --source-dir ./Archiv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence, cast


DEFAULT_CLUSTERS: tuple[int, ...] = (10, 15, 20, 25, 30, 40, 50)
DEFAULT_OMEGAS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_RANDOM_STATES: tuple[int, ...] = (0, 1, 2)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 10, 50, 100, 200)
DISPLAY_HORIZONS: tuple[int, ...] = (50, 100, 200)
RUNNER_NAME = "run_exp09_vanderpol_tuning.py"
RAW_CSV_NAME = "experiment_09_multistep_raw_results.csv"
SUMMARY_CSV_NAME = "experiment_09_multistep_summary.csv"

CsvRow = dict[str, str]
HorizonStatsMap = dict[int, "HorizonStats"]
RawGroupedMap = dict[tuple[int, float, int], dict[int, float]]


@dataclass(frozen=True)
class Config:
    source_dir: Path
    runner_script: Path
    experiment_output_dir: Path
    table_output_dir: Path
    run_experiment: bool
    force_rerun: bool
    python_executable: str
    clusters: tuple[int, ...]
    omegas: tuple[float, ...]
    random_states: tuple[int, ...]
    horizons: tuple[int, ...]
    top_k: int
    n_steps: int
    dt: float
    mu: float
    train_fraction: float
    tau: float
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int
    mean_std_mode: str


@dataclass(frozen=True)
class HorizonStats:
    mean_value: float
    std_value: float
    n_runs: int


@dataclass(frozen=True)
class ConfigStats:
    n_runs: int
    mean_eh_mean: float
    mean_eh_std: float
    e1_mean: float
    e1_std: float
    e10_mean: float
    e10_std: float
    e50_mean: float
    e50_std: float
    e100_mean: float
    e100_std: float
    e200_mean: float
    e200_std: float


@dataclass(frozen=True)
class CandidateRow:
    c: int
    omega: float
    stats: ConfigStats


@dataclass(frozen=True)
class TableRow:
    rank: int
    c: int
    omega: float
    stats: ConfigStats

    def to_csv_row(self) -> CsvRow:
        return {
            "rank": str(self.rank),
            "C": str(self.c),
            "omega": omega_label(self.omega),
            "n_runs": str(self.stats.n_runs),
            "mean_Eh_mean": format_float(self.stats.mean_eh_mean),
            "mean_Eh_std": format_float(self.stats.mean_eh_std),
            "E50_mean": format_float(self.stats.e50_mean),
            "E50_std": format_float(self.stats.e50_std),
            "E100_mean": format_float(self.stats.e100_mean),
            "E100_std": format_float(self.stats.e100_std),
            "E200_mean": format_float(self.stats.e200_mean),
            "E200_std": format_float(self.stats.e200_std),
            "E1_mean_audit": format_float(self.stats.e1_mean),
            "E1_std_audit": format_float(self.stats.e1_std),
            "E10_mean_audit": format_float(self.stats.e10_mean),
            "E10_std_audit": format_float(self.stats.e10_std),
            "mean_Eh_formatted": sci_pm(self.stats.mean_eh_mean, self.stats.mean_eh_std),
            "E50_formatted": sci_pm(self.stats.e50_mean, self.stats.e50_std),
            "E100_formatted": sci_pm(self.stats.e100_mean, self.stats.e100_std),
            "E200_formatted": sci_pm(self.stats.e200_mean, self.stats.e200_std),
        }


class CliArgs(argparse.Namespace):
    source_dir: str
    runner_script: str
    experiment_output_dir: str
    table_output_dir: str
    run_experiment: bool
    force_rerun: bool
    python_executable: str
    clusters: list[str]
    omegas: list[str]
    random_states: list[str]
    horizons: list[str]
    top_k: int
    n_steps: int
    dt: float
    mu: float
    train_fraction: float
    tau: float
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int
    mean_std_mode: str


def parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if len(parsed) == 0:
        raise ValueError("At least one integer value is required.")
    return parsed


def parse_float_tuple(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) == 0:
        raise ValueError("At least one floating-point value is required.")
    return parsed


def safe_float(text: str | None) -> float:
    if text is None:
        return math.nan
    stripped = text.strip()
    if stripped == "":
        return math.nan
    try:
        return float(stripped)
    except ValueError:
        return math.nan


def safe_int(text: str | None) -> int:
    value = safe_float(text)
    if not math.isfinite(value):
        return 0
    return int(value)


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.16g}"


def omega_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def sci_pm(mean_value: float, std_value: float, digits: int = 2) -> str:
    if not math.isfinite(mean_value):
        return "NA"
    if mean_value == 0.0:
        return f"(0.{'0' * digits} +/- {std_value:.{digits}f}) x 10^0"
    exponent = int(math.floor(math.log10(abs(mean_value))))
    scale = 10.0**exponent
    mantissa = mean_value / scale
    std_mantissa = std_value / scale if math.isfinite(std_value) else math.nan
    return f"({mantissa:.{digits}f} +/- {std_mantissa:.{digits}f}) x 10^{exponent}"


def latex_sci_pm(mean_value: float, std_value: float, digits: int = 2) -> str:
    if not math.isfinite(mean_value):
        return "NA"
    if mean_value == 0.0:
        return rf"$(0.{'0' * digits} \pm {std_value:.{digits}f}) \times 10^{{0}}$"
    exponent = int(math.floor(math.log10(abs(mean_value))))
    scale = 10.0**exponent
    mantissa = mean_value / scale
    std_mantissa = std_value / scale if math.isfinite(std_value) else math.nan
    return rf"$({mantissa:.{digits}f} \pm {std_mantissa:.{digits}f}) \times 10^{{{exponent}}}$"


def read_csv(path: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            rows.append({str(key): "" if value is None else value for key, value in raw_row.items()})
    return rows


def write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    if len(rows) == 0:
        raise RuntimeError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run/reformat the Van der Pol tuning experiment and emit manuscript Table 4 values."
    )
    parser.add_argument("--source-dir", default=".", help="Folder containing run_exp09_vanderpol_tuning.py.")
    parser.add_argument("--runner-script", default=RUNNER_NAME)
    parser.add_argument("--experiment-output-dir", default="kahkm_exp09_vanderpol_tuning")
    parser.add_argument("--table-output-dir", default="kahkm_table4_vanderpol")

    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run", dest="run_experiment", action="store_true", default=True)
    run_group.add_argument("--no-run", dest="run_experiment", action="store_false")

    parser.add_argument("--force-rerun", action="store_true", help="Rerun even if summary CSV already exists.")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--clusters", nargs="+", default=[str(x) for x in DEFAULT_CLUSTERS])
    parser.add_argument("--omegas", nargs="+", default=[omega_label(x) for x in DEFAULT_OMEGAS])
    parser.add_argument("--random-states", nargs="+", default=[str(x) for x in DEFAULT_RANDOM_STATES])
    parser.add_argument("--horizons", nargs="+", default=[str(x) for x in DEFAULT_HORIZONS])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mean-std-mode",
        choices=["horizon-mean", "seed-mean"],
        default="horizon-mean",
        help=(
            "How to compute the standard deviation displayed with mean E_h. "
            "horizon-mean averages the per-horizon summary stds. seed-mean uses raw rows "
            "to compute std across per-seed horizon means when raw results are available."
        ),
    )

    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    return parser


def resolve_path(base_dir: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_config(args: CliArgs) -> Config:
    source_dir = Path(args.source_dir).expanduser().resolve()
    runner_script = resolve_path(source_dir, args.runner_script)
    experiment_output_dir = resolve_path(source_dir, args.experiment_output_dir)
    table_output_dir = resolve_path(Path.cwd(), args.table_output_dir)
    horizons = tuple(sorted(set(parse_int_tuple(args.horizons))))
    missing_horizons = sorted(set(DEFAULT_HORIZONS) - set(horizons))
    if missing_horizons:
        raise ValueError(f"Table 4 requires horizons {DEFAULT_HORIZONS}; missing {missing_horizons}.")

    return Config(
        source_dir=source_dir,
        runner_script=runner_script,
        experiment_output_dir=experiment_output_dir,
        table_output_dir=table_output_dir,
        run_experiment=args.run_experiment,
        force_rerun=args.force_rerun,
        python_executable=args.python_executable,
        clusters=tuple(sorted(set(parse_int_tuple(args.clusters)))),
        omegas=tuple(sorted(set(parse_float_tuple(args.omegas)))),
        random_states=parse_int_tuple(args.random_states),
        horizons=horizons,
        top_k=args.top_k,
        n_steps=args.n_steps,
        dt=args.dt,
        mu=args.mu,
        train_fraction=args.train_fraction,
        tau=args.tau,
        subspace_dim=args.subspace_dim,
        nb=args.nb,
        beta=args.beta,
        nlms_epochs=args.nlms_epochs,
        kmeans_kind=args.kmeans_kind,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        max_train_per_cluster=args.max_train_per_cluster,
        mean_std_mode=args.mean_std_mode,
    )


def run_underlying_experiment(config: Config) -> None:
    summary_path = config.experiment_output_dir / SUMMARY_CSV_NAME
    if summary_path.exists() and not config.force_rerun:
        print(f"Reusing existing Experiment 09 summary: {summary_path}")
        return

    if not config.runner_script.exists():
        raise FileNotFoundError(
            f"Could not find {config.runner_script}. Put this script in the experiment folder "
            "or pass --source-dir/--runner-script."
        )

    command: list[str] = [
        config.python_executable,
        str(config.runner_script),
        "--output-dir",
        str(config.experiment_output_dir),
        "--n-steps",
        str(config.n_steps),
        "--dt",
        str(config.dt),
        "--mu",
        str(config.mu),
        "--train-fraction",
        str(config.train_fraction),
        "--clusters",
        *[str(value) for value in config.clusters],
        "--omegas",
        *[omega_label(value) for value in config.omegas],
        "--random-states",
        *[str(value) for value in config.random_states],
        "--tau",
        str(config.tau),
        "--subspace-dim",
        str(config.subspace_dim),
        "--nb",
        str(config.nb),
        "--beta",
        str(config.beta),
        "--nlms-epochs",
        str(config.nlms_epochs),
        "--kmeans-kind",
        config.kmeans_kind,
        "--batch-size",
        str(config.batch_size),
        "--n-jobs",
        str(config.n_jobs),
        "--horizons",
        *[str(value) for value in config.horizons],
        "--selection-horizons",
        *[str(value) for value in DEFAULT_HORIZONS],
    ]
    if config.max_train_per_cluster > 0:
        command.extend(["--max-train-per-cluster", str(config.max_train_per_cluster)])

    print("Running underlying Van der Pol tuning experiment:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(config.source_dir), check=True)


def required(row: Mapping[str, str], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise KeyError(f"Missing required column: {key}")
    return value


def read_horizon_stats(summary_path: Path) -> dict[tuple[int, float], HorizonStatsMap]:
    rows = read_csv(summary_path)
    by_config: dict[tuple[int, float], HorizonStatsMap] = defaultdict(dict)
    for row in rows:
        c = safe_int(required(row, "n_clusters"))
        omega = safe_float(required(row, "omega"))
        horizon = safe_int(required(row, "horizon"))
        mean_value = safe_float(required(row, "relative_association_error_mean"))
        std_value = safe_float(required(row, "relative_association_error_std"))
        n_runs = safe_int(row.get("n_runs", "0"))
        if math.isfinite(mean_value):
            by_config[(c, omega)][horizon] = HorizonStats(mean_value, std_value, n_runs)
    return dict(by_config)


def read_raw_grouped(raw_path: Path) -> RawGroupedMap:
    grouped: RawGroupedMap = defaultdict(dict)
    if not raw_path.exists():
        return {}
    for row in read_csv(raw_path):
        c = safe_int(row.get("n_clusters"))
        omega = safe_float(row.get("omega"))
        seed = safe_int(row.get("seed"))
        horizon = safe_int(row.get("horizon"))
        error = safe_float(row.get("relative_association_error"))
        if math.isfinite(error):
            grouped[(c, omega, seed)][horizon] = error
    return dict(grouped)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def mean_std_for_config(
    config: Config,
    c: int,
    omega: float,
    horizon_map: HorizonStatsMap,
    raw_grouped: RawGroupedMap,
) -> tuple[float, float]:
    horizon_means = [horizon_map[h].mean_value for h in DEFAULT_HORIZONS]
    mean_eh_mean = mean(horizon_means)

    if config.mean_std_mode == "seed-mean" and raw_grouped:
        per_seed_means: list[float] = []
        for key, errors_by_horizon in sorted(raw_grouped.items()):
            raw_c, raw_omega, _seed = key
            if raw_c != c:
                continue
            if not math.isclose(raw_omega, omega, rel_tol=0.0, abs_tol=1.0e-12):
                continue
            if all(horizon in errors_by_horizon for horizon in DEFAULT_HORIZONS):
                per_seed_means.append(mean([errors_by_horizon[h] for h in DEFAULT_HORIZONS]))
        if per_seed_means:
            return mean_eh_mean, sample_std(per_seed_means)

    horizon_stds = [horizon_map[h].std_value for h in DEFAULT_HORIZONS]
    finite_stds = [value for value in horizon_stds if math.isfinite(value)]
    mean_eh_std = mean(finite_stds) if finite_stds else math.nan
    return mean_eh_mean, mean_eh_std


def stats_for_config(
    config: Config,
    c: int,
    omega: float,
    horizon_map: HorizonStatsMap,
    raw_grouped: RawGroupedMap,
) -> ConfigStats:
    missing = [horizon for horizon in DEFAULT_HORIZONS if horizon not in horizon_map]
    if missing:
        raise KeyError(f"Missing horizons {missing} for C={c}, omega={omega}.")

    mean_eh_mean, mean_eh_std = mean_std_for_config(config, c, omega, horizon_map, raw_grouped)
    n_runs = horizon_map[DEFAULT_HORIZONS[0]].n_runs
    return ConfigStats(
        n_runs=n_runs,
        mean_eh_mean=mean_eh_mean,
        mean_eh_std=mean_eh_std,
        e1_mean=horizon_map[1].mean_value,
        e1_std=horizon_map[1].std_value,
        e10_mean=horizon_map[10].mean_value,
        e10_std=horizon_map[10].std_value,
        e50_mean=horizon_map[50].mean_value,
        e50_std=horizon_map[50].std_value,
        e100_mean=horizon_map[100].mean_value,
        e100_std=horizon_map[100].std_value,
        e200_mean=horizon_map[200].mean_value,
        e200_std=horizon_map[200].std_value,
    )


def build_table_rows(config: Config) -> list[TableRow]:
    summary_path = config.experiment_output_dir / SUMMARY_CSV_NAME
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary results: {summary_path}. Run without --no-run, or provide --experiment-output-dir."
        )
    raw_grouped = read_raw_grouped(config.experiment_output_dir / RAW_CSV_NAME)
    by_config = read_horizon_stats(summary_path)

    candidates: list[CandidateRow] = []
    for (c, omega), horizon_map in sorted(by_config.items()):
        try:
            stats = stats_for_config(config, c, omega, horizon_map, raw_grouped)
        except KeyError:
            continue
        candidates.append(CandidateRow(c=c, omega=omega, stats=stats))

    if not candidates:
        raise RuntimeError("No complete configurations found for Table 4.")
    candidates.sort(key=lambda candidate: candidate.stats.mean_eh_mean)

    rows: list[TableRow] = []
    for rank, candidate in enumerate(candidates[: config.top_k], start=1):
        rows.append(TableRow(rank=rank, c=candidate.c, omega=candidate.omega, stats=candidate.stats))
    return rows


def write_markdown(path: Path, rows: Sequence[TableRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Rank | C | omega | mean E_h | E50 | E100 | E200 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.rank),
                    str(row.c),
                    omega_label(row.omega),
                    sci_pm(row.stats.mean_eh_mean, row.stats.mean_eh_std),
                    sci_pm(row.stats.e50_mean, row.stats.e50_std),
                    sci_pm(row.stats.e100_mean, row.stats.e100_std),
                    sci_pm(row.stats.e200_mean, row.stats.e200_std),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: Sequence[TableRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        'Rank & $C$ & $\\omega$ & mean $E_h$ & $E_{50}$ & $E_{100}$ & $E_{200}$ \\\\',
        r"\midrule",
    ]
    for row in rows:
        values = [
            str(row.rank),
            str(row.c),
            omega_label(row.omega),
            latex_sci_pm(row.stats.mean_eh_mean, row.stats.mean_eh_std),
            latex_sci_pm(row.stats.e50_mean, row.stats.e50_std),
            latex_sci_pm(row.stats.e100_mean, row.stats.e100_std),
            latex_sci_pm(row.stats.e200_mean, row.stats.e200_std),
        ]
        lines.append(" & ".join(values) + ' \\\\')
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(path: Path, config: Config, rows: Sequence[TableRow]) -> None:
    metadata: dict[str, str | int | float | bool | list[int] | list[float]] = {
        "table": "Table 4",
        "description": "Expanded Van der Pol tuning over fragmentation and association sharpness.",
        "ranking_objective": "mean relative association error over h={1,10,50,100,200}",
        "source_dir": str(config.source_dir),
        "runner_script": str(config.runner_script),
        "experiment_output_dir": str(config.experiment_output_dir),
        "table_output_dir": str(config.table_output_dir),
        "run_experiment": config.run_experiment,
        "force_rerun": config.force_rerun,
        "clusters": list(config.clusters),
        "omegas": list(config.omegas),
        "random_states": list(config.random_states),
        "horizons": list(config.horizons),
        "top_k": config.top_k,
        "mean_std_mode": config.mean_std_mode,
        "n_steps": config.n_steps,
        "dt": config.dt,
        "mu": config.mu,
        "train_fraction": config.train_fraction,
        "tau": config.tau,
        "subspace_dim": config.subspace_dim,
        "nb": config.nb,
        "beta": config.beta,
        "nlms_epochs": config.nlms_epochs,
        "kmeans_kind": config.kmeans_kind,
        "batch_size": config.batch_size,
        "n_jobs": config.n_jobs,
        "max_train_per_cluster": config.max_train_per_cluster,
        "rows_written": len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def print_preview(md_path: Path) -> None:
    print("\nMarkdown preview:\n")
    print(md_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = build_parser()
    args = cast(CliArgs, parser.parse_args())
    config = resolve_config(args)

    if config.run_experiment:
        run_underlying_experiment(config)

    rows = build_table_rows(config)
    output_dir = config.table_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "table4_vanderpol_values.csv"
    md_path = output_dir / "table4_vanderpol_values.md"
    tex_path = output_dir / "table4_vanderpol_values.tex"
    metadata_path = output_dir / "table4_vanderpol_metadata.json"

    write_csv(csv_path, [row.to_csv_row() for row in rows])
    write_markdown(md_path, rows)
    write_latex(tex_path, rows)
    write_metadata(metadata_path, config, rows)

    print("\nGenerated Table 4 files:")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"  LaTeX:    {tex_path}")
    print(f"  Metadata: {metadata_path}")
    print_preview(md_path)


if __name__ == "__main__":
    main()
