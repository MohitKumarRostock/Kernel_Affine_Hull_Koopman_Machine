#!/usr/bin/env python3
"""Generate manuscript Table 8 (noise-aware Van der Pol tuning).

Place this script in the same directory as the experiment scripts and run:

    python step_07_generate_table8_noise_aware.py

It runs or reuses:
    run_exp12_vanderpol_noise_aware_tuning.py

Outputs are written to:
    kahkm_table8_noise_aware/table8_noise_aware_values.csv
    kahkm_table8_noise_aware/table8_noise_aware_values.md
    kahkm_table8_noise_aware/table8_noise_aware_values.tex
    kahkm_table8_noise_aware/table8_noise_aware_metadata.json
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
from typing import Sequence


HORIZONS: tuple[int, int, int] = (50, 100, 200)


@dataclass(frozen=True)
class MetricCell:
    mean: float
    std: float

    def csv_mean(self) -> str:
        return format_float(self.mean)

    def csv_std(self) -> str:
        return format_float(self.std)

    def table_cell(self) -> str:
        return mean_std_scientific_cell(self.mean, self.std)


@dataclass(frozen=True)
class TableRow:
    noise_level: float
    e50: MetricCell
    e100: MetricCell
    e200: MetricCell

    def as_csv_row(self) -> dict[str, str]:
        return {
            "training_noise": noise_label(self.noise_level),
            "noise_level": format_float(self.noise_level),
            "E50_mean": self.e50.csv_mean(),
            "E50_std": self.e50.csv_std(),
            "E100_mean": self.e100.csv_mean(),
            "E100_std": self.e100.csv_std(),
            "E200_mean": self.e200.csv_mean(),
            "E200_std": self.e200.csv_std(),
            "E50_table_cell": self.e50.table_cell(),
            "E100_table_cell": self.e100.table_cell(),
            "E200_table_cell": self.e200.table_cell(),
        }


@dataclass(frozen=True)
class SelectedConfig:
    n_clusters: int
    omega: float
    source: str


class CliArgs(argparse.Namespace):
    experiment_dir: str
    experiment_output_dir: str
    output_dir: str
    no_run: bool
    force: bool
    python_executable: str
    selected_n_clusters: int | None
    selected_omega: float | None


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Table 8 from the noise-aware Van der Pol tuning sweep."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=".",
        help="Directory containing run_exp12_vanderpol_noise_aware_tuning.py. Defaults to the current directory.",
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=str,
        default="kahkm_exp12_vanderpol_noise_aware_tuning",
        help="Output directory used by run_exp12_vanderpol_noise_aware_tuning.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kahkm_table8_noise_aware",
        help="Directory for the generated Table 8 files.",
    )
    parser.add_argument(
        "--selected-n-clusters",
        type=int,
        default=None,
        help="Optional override for selected C. If omitted, the rank-1 robust-selection row is used.",
    )
    parser.add_argument(
        "--selected-omega",
        type=float,
        default=None,
        help="Optional override for selected omega. If omitted, the rank-1 robust-selection row is used.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run the experiment; only format existing CSV outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the experiment even if summary CSVs already exist.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to run the experiment script.",
    )
    return parser.parse_args(argv, namespace=CliArgs())


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.17g}"


def parse_float(text: str, *, field_name: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse {field_name}={text!r} as float") from exc


def parse_int(text: str, *, field_name: str) -> int:
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Could not parse {field_name}={text!r} as int") from exc


def noise_label(noise_level: float) -> str:
    percent = 100.0 * noise_level
    if abs(percent - round(percent)) < 1.0e-12:
        return f"{int(round(percent))}%"
    return f"{percent:.2f}".rstrip("0").rstrip(".") + "%"


def mean_std_scientific_cell(mean_value: float, std_value: float) -> str:
    if math.isnan(mean_value) or math.isnan(std_value):
        return "nan"
    if mean_value == 0.0:
        return rf"$(0.00\pm{std_value:.2f})$"
    abs_mean = abs(mean_value)
    exponent = int(math.floor(math.log10(abs_mean)))
    scale = 10.0 ** exponent
    mean_scaled = mean_value / scale
    std_scaled = std_value / scale
    return rf"$({mean_scaled:.2f}\pm{std_scaled:.2f})\times10^{{{exponent}}}$"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def run_or_reuse_experiment(
    experiment_dir: Path,
    experiment_output_dir: Path,
    python_executable: str,
    *,
    force: bool,
    no_run: bool,
) -> None:
    script_path = experiment_dir / "run_exp12_vanderpol_noise_aware_tuning.py"
    summary_path = experiment_output_dir / "experiment_12_noise_tuning_summary_by_config_noise_horizon.csv"
    selection_path = experiment_output_dir / "experiment_12_noise_tuning_robust_selection.csv"

    if no_run:
        require_file(summary_path)
        require_file(selection_path)
        return

    require_file(script_path)
    if summary_path.exists() and selection_path.exists() and not force:
        return

    command = [python_executable, str(script_path), "--output-dir", str(experiment_output_dir)]
    subprocess.run(command, cwd=str(experiment_dir), check=True)


def selected_config_from_outputs(
    selection_path: Path,
    selected_n_clusters: int | None,
    selected_omega: float | None,
) -> SelectedConfig:
    if (selected_n_clusters is None) != (selected_omega is None):
        raise ValueError("Pass both --selected-n-clusters and --selected-omega, or pass neither.")
    if selected_n_clusters is not None and selected_omega is not None:
        return SelectedConfig(selected_n_clusters, selected_omega, "command-line override")

    rows = read_csv_rows(selection_path)
    if not rows:
        raise ValueError(f"No rows found in robust-selection file: {selection_path}")
    sorted_rows = sorted(rows, key=lambda row: parse_int(row["rank"], field_name="rank"))
    best = sorted_rows[0]
    return SelectedConfig(
        n_clusters=parse_int(best["n_clusters"], field_name="n_clusters"),
        omega=parse_float(best["omega"], field_name="omega"),
        source="rank-1 robust-selection row",
    )


def collect_table_rows(summary_path: Path, selected: SelectedConfig) -> list[TableRow]:
    rows = read_csv_rows(summary_path)
    grouped: dict[float, dict[int, MetricCell]] = {}

    for row in rows:
        n_clusters = parse_int(row["n_clusters"], field_name="n_clusters")
        omega = parse_float(row["omega"], field_name="omega")
        horizon = parse_int(row["horizon"], field_name="horizon")
        if n_clusters != selected.n_clusters:
            continue
        if abs(omega - selected.omega) > 1.0e-12:
            continue
        if horizon not in HORIZONS:
            continue
        noise_level = parse_float(row["noise_level"], field_name="noise_level")
        mean_value = parse_float(row["relative_association_error_mean"], field_name="relative_association_error_mean")
        std_value = parse_float(row["relative_association_error_std"], field_name="relative_association_error_std")
        grouped.setdefault(noise_level, {})[horizon] = MetricCell(mean=mean_value, std=std_value)

    table_rows: list[TableRow] = []
    for noise_level in sorted(grouped):
        horizon_cells = grouped[noise_level]
        missing = [horizon for horizon in HORIZONS if horizon not in horizon_cells]
        if missing:
            raise KeyError(f"Missing horizons {missing} for noise level {noise_level:g}")
        table_rows.append(
            TableRow(
                noise_level=noise_level,
                e50=horizon_cells[50],
                e100=horizon_cells[100],
                e200=horizon_cells[200],
            )
        )

    if not table_rows:
        raise ValueError(
            f"No Table 8 rows found for C={selected.n_clusters}, omega={selected.omega:g} in {summary_path}"
        )
    return table_rows


def write_csv(path: Path, rows: list[TableRow]) -> None:
    csv_rows = [row.as_csv_row() for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def write_markdown(path: Path, rows: list[TableRow]) -> None:
    lines = [
        "| Training noise | $E_{50}$ | $E_{100}$ | $E_{200}$ |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        csv_row = row.as_csv_row()
        lines.append(
            f"| {csv_row['training_noise']} | {csv_row['E50_table_cell']} | "
            f"{csv_row['E100_table_cell']} | {csv_row['E200_table_cell']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[TableRow], selected: SelectedConfig) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Noise-aware Van der Pol tuning with the robust selected configuration $C={selected.n_clusters}$, $\omega={selected.omega:g}$. Gaussian noise is added only to training snapshots and evaluation is performed on clean held-out snapshots. Values are mean $\pm$ standard deviation over three random states.}}",
        r"\label{tab:vdp_noise_tuned}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Training noise & $E_{50}$ & $E_{100}$ & $E_{200}$ \\",
        r"\midrule",
    ]
    for row in rows:
        csv_row = row.as_csv_row()
        lines.append(
            f"{csv_row['training_noise']} & {csv_row['E50_table_cell']} & "
            f"{csv_row['E100_table_cell']} & {csv_row['E200_table_cell']} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir).resolve()
    experiment_output_dir = Path(args.experiment_output_dir)
    if not experiment_output_dir.is_absolute():
        experiment_output_dir = experiment_dir / experiment_output_dir
    output_dir = Path(args.output_dir).resolve()

    run_or_reuse_experiment(
        experiment_dir,
        experiment_output_dir,
        args.python_executable,
        force=args.force,
        no_run=args.no_run,
    )

    selection_path = experiment_output_dir / "experiment_12_noise_tuning_robust_selection.csv"
    summary_path = experiment_output_dir / "experiment_12_noise_tuning_summary_by_config_noise_horizon.csv"
    require_file(selection_path)
    require_file(summary_path)

    selected = selected_config_from_outputs(selection_path, args.selected_n_clusters, args.selected_omega)
    table_rows = collect_table_rows(summary_path, selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "table8_noise_aware_values.csv", table_rows)
    write_markdown(output_dir / "table8_noise_aware_values.md", table_rows)
    write_latex(output_dir / "table8_noise_aware_values.tex", table_rows, selected)

    metadata = {
        "table": "Table 8",
        "description": "Noise-aware Van der Pol tuning for the robust selected KAHKM configuration.",
        "experiment_script": "run_exp12_vanderpol_noise_aware_tuning.py",
        "experiment_output_dir": str(experiment_output_dir),
        "selected_n_clusters": selected.n_clusters,
        "selected_omega": selected.omega,
        "selected_config_source": selected.source,
        "horizons": list(HORIZONS),
        "inputs": {
            "summary_by_config_noise_horizon": str(summary_path),
            "robust_selection": str(selection_path),
        },
    }
    (output_dir / "table8_noise_aware_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Selected C={selected.n_clusters}, omega={selected.omega:g} from {selected.source}.")
    print(f"Wrote Table 8 files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
