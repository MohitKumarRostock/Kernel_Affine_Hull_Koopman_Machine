#!/usr/bin/env python3
"""Generate manuscript Table 9 from Experiment 13 outputs.

Table 9: Tuned Van der Pol unseen-trajectory generalization.

Place this script in the same directory as the experiment scripts and run:

    python step_08_generate_table9_tuned_trajectory.py

By default, it runs/reuses:

    run_exp13_vanderpol_tuned_trajectory_generalization.py

and reads:

    kahkm_exp13_vanderpol_tuned_trajectory_generalization/
        experiment_13_tuned_trajectory_table_values.csv

Use --no-run to format existing experiment outputs only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


EXPERIMENT_RUNNER = "run_exp13_vanderpol_tuned_trajectory_generalization.py"
EXPERIMENT_OUTPUT_DIR = "kahkm_exp13_vanderpol_tuned_trajectory_generalization"
TABLE_OUTPUT_DIR = "kahkm_table9_tuned_trajectory"
TABLE_VALUES_FILE = "experiment_13_tuned_trajectory_table_values.csv"

CONFIG_ROBUST = "robust_tuned"
CONFIG_CLEAN = "clean_tuned"
CONFIG_DEFAULT = "old_default"


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    experiment_output_dir: Path
    output_dir: Path
    runner: str
    python_executable: str
    no_run: bool
    resume: bool
    n_jobs: int | None


@dataclass(frozen=True)
class ExperimentRow:
    config_name: str
    n_clusters: int
    omega: float
    n_train_trajectories: int
    E50_mean: float
    E50_std: float
    E100_mean: float
    E100_std: float
    E200_mean: float
    E200_std: float


@dataclass(frozen=True)
class TableRow:
    training_trajectories: int
    robust_E50_mean: float
    robust_E50_std: float
    robust_E100_mean: float
    robust_E100_std: float
    robust_E200_mean: float
    robust_E200_std: float
    clean_tuned_E200_mean: float
    clean_tuned_E200_std: float
    default_E200_mean: float
    default_E200_std: float

    @property
    def robust_E50_cell(self) -> str:
        return format_mean_std_scientific(self.robust_E50_mean, self.robust_E50_std)

    @property
    def robust_E100_cell(self) -> str:
        return format_mean_std_scientific(self.robust_E100_mean, self.robust_E100_std)

    @property
    def robust_E200_cell(self) -> str:
        return format_mean_std_scientific(self.robust_E200_mean, self.robust_E200_std)

    @property
    def clean_tuned_E200_cell(self) -> str:
        return format_mean_std_scientific(self.clean_tuned_E200_mean, self.clean_tuned_E200_std)

    @property
    def default_E200_cell(self) -> str:
        return format_mean_std_scientific(self.default_E200_mean, self.default_E200_std)


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate manuscript Table 9 from Experiment 13 Van der Pol trajectory-generalization outputs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=script_dir,
        help="Directory containing the experiment scripts. Default: this script's directory.",
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing Experiment 13 outputs. Default: "
            f"<source-dir>/{EXPERIMENT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for generated Table 9 files. Default: <source-dir>/{TABLE_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--runner",
        default=EXPERIMENT_RUNNER,
        help=f"Experiment runner script. Default: {EXPERIMENT_RUNNER}.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to run the experiment runner.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run Experiment 13; only format existing outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to the Experiment 13 runner.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Optional --n-jobs value passed to the Experiment 13 runner.",
    )

    ns = parser.parse_args(argv)
    source_dir = Path(ns.source_dir).resolve()
    experiment_output_dir = (
        Path(ns.experiment_output_dir).resolve()
        if ns.experiment_output_dir is not None
        else source_dir / EXPERIMENT_OUTPUT_DIR
    )
    output_dir = (
        Path(ns.output_dir).resolve()
        if ns.output_dir is not None
        else source_dir / TABLE_OUTPUT_DIR
    )

    return CliArgs(
        source_dir=source_dir,
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        runner=str(ns.runner),
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        resume=bool(ns.resume),
        n_jobs=ns.n_jobs if ns.n_jobs is None else int(ns.n_jobs),
    )


def run_experiment(args: CliArgs) -> None:
    runner_path = args.source_dir / args.runner
    if not runner_path.exists():
        raise FileNotFoundError(f"Experiment runner not found: {runner_path}")

    command = [
        args.python_executable,
        str(runner_path),
        "--output-dir",
        str(args.experiment_output_dir),
    ]
    if args.resume:
        command.append("--resume")
    if args.n_jobs is not None:
        command.extend(["--n-jobs", str(args.n_jobs)])

    print("Running Experiment 13:")
    print(" ".join(command))
    subprocess.run(command, cwd=args.source_dir, check=True)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric column {key!r} from value {value!r}") from exc
    if not math.isfinite(number):
        return math.nan
    return number


def parse_int(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key, "0")))


def load_experiment_rows(table_values_path: Path) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    for row in read_csv_dicts(table_values_path):
        rows.append(
            ExperimentRow(
                config_name=row.get("config_name", ""),
                n_clusters=parse_int(row, "n_clusters"),
                omega=parse_float(row, "omega"),
                n_train_trajectories=parse_int(row, "n_train_trajectories"),
                E50_mean=parse_float(row, "E50_mean"),
                E50_std=parse_float(row, "E50_std"),
                E100_mean=parse_float(row, "E100_mean"),
                E100_std=parse_float(row, "E100_std"),
                E200_mean=parse_float(row, "E200_mean"),
                E200_std=parse_float(row, "E200_std"),
            )
        )
    return rows


def format_mean_std_scientific(mean_value: float, std_value: float) -> str:
    if not math.isfinite(mean_value) or not math.isfinite(std_value):
        return "--"
    if mean_value == 0.0:
        return f"$({mean_value:.2f}\\pm{std_value:.2f})$"
    exponent = int(math.floor(math.log10(abs(mean_value))))
    scale = 10.0 ** exponent
    mean_scaled = mean_value / scale
    std_scaled = std_value / scale
    return f"$({mean_scaled:.2f}\\pm{std_scaled:.2f})\\times10^{{{exponent}}}$"


def build_table_rows(experiment_rows: Sequence[ExperimentRow]) -> list[TableRow]:
    indexed: dict[tuple[str, int], ExperimentRow] = {}
    for row in experiment_rows:
        indexed[(row.config_name, row.n_train_trajectories)] = row

    train_counts = sorted(
        count for config, count in indexed if config == CONFIG_ROBUST
    )
    table_rows: list[TableRow] = []

    for train_count in train_counts:
        missing = [
            name
            for name in (CONFIG_ROBUST, CONFIG_CLEAN, CONFIG_DEFAULT)
            if (name, train_count) not in indexed
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"Missing Experiment 13 rows for train_count={train_count}: {missing_text}"
            )

        robust = indexed[(CONFIG_ROBUST, train_count)]
        clean = indexed[(CONFIG_CLEAN, train_count)]
        default = indexed[(CONFIG_DEFAULT, train_count)]
        table_rows.append(
            TableRow(
                training_trajectories=train_count,
                robust_E50_mean=robust.E50_mean,
                robust_E50_std=robust.E50_std,
                robust_E100_mean=robust.E100_mean,
                robust_E100_std=robust.E100_std,
                robust_E200_mean=robust.E200_mean,
                robust_E200_std=robust.E200_std,
                clean_tuned_E200_mean=clean.E200_mean,
                clean_tuned_E200_std=clean.E200_std,
                default_E200_mean=default.E200_mean,
                default_E200_std=default.E200_std,
            )
        )

    return table_rows


def table_row_to_csv(row: TableRow) -> dict[str, str]:
    return {
        "training_trajectories": str(row.training_trajectories),
        "robust_E50_mean": repr(row.robust_E50_mean),
        "robust_E50_std": repr(row.robust_E50_std),
        "robust_E100_mean": repr(row.robust_E100_mean),
        "robust_E100_std": repr(row.robust_E100_std),
        "robust_E200_mean": repr(row.robust_E200_mean),
        "robust_E200_std": repr(row.robust_E200_std),
        "clean_tuned_E200_mean": repr(row.clean_tuned_E200_mean),
        "clean_tuned_E200_std": repr(row.clean_tuned_E200_std),
        "default_E200_mean": repr(row.default_E200_mean),
        "default_E200_std": repr(row.default_E200_std),
        "robust_E50_table_cell": row.robust_E50_cell,
        "robust_E100_table_cell": row.robust_E100_cell,
        "robust_E200_table_cell": row.robust_E200_cell,
        "clean_tuned_E200_table_cell": row.clean_tuned_E200_cell,
        "default_E200_table_cell": row.default_E200_cell,
    }


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        "# Table 9: Tuned Van der Pol unseen-trajectory generalization",
        "",
        "| Training trajectories | Robust E50 | Robust E100 | Robust E200 | Clean-tuned E200 | Default E200 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.training_trajectories} | "
            f"{row.robust_E50_cell} | "
            f"{row.robust_E100_cell} | "
            f"{row.robust_E200_cell} | "
            f"{row.clean_tuned_E200_cell} | "
            f"{row.default_E200_cell} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_latex(rows: Sequence[TableRow]) -> str:
    body_lines: list[str] = []
    for row in rows:
        body_lines.append(
            f"{row.training_trajectories} & "
            f"{row.robust_E50_cell} & "
            f"{row.robust_E100_cell} & "
            f"{row.robust_E200_cell} & "
            f"{row.clean_tuned_E200_cell} & "
            f"{row.default_E200_cell} \\\\" 
        )
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Tuned Van der Pol unseen-trajectory generalization. "
        "The robust setting $C=10,\\omega=0.25$ is compared with the clean-tuned "
        "setting $C=15,\\omega=0.5$ and the default setting $C=20,\\omega=4$. "
        "Values are mean $\\pm$ standard deviation over three independent "
        "training-seed blocks and three unseen test trajectories.}\n"
        "\\label{tab:vanderpol_tuned_trajectory}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{cccccc}\n"
        "\\toprule\n"
        "Training trajectories & Robust $E_{50}$ & Robust $E_{100}$ & Robust $E_{200}$ & "
        "Clean-tuned $E_{200}$ & Default $E_{200}$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )


def write_metadata(path: Path, args: CliArgs, table_values_path: Path, rows: Sequence[TableRow]) -> None:
    metadata = {
        "table": "Table 9",
        "description": "Tuned Van der Pol unseen-trajectory generalization",
        "experiment_runner": args.runner,
        "experiment_output_dir": str(args.experiment_output_dir),
        "source_table_values_csv": str(table_values_path),
        "configs": {
            "robust": "robust_tuned: C=10, omega=0.25",
            "clean_tuned": "clean_tuned: C=15, omega=0.5",
            "default": "old_default: C=20, omega=4",
        },
        "training_trajectories": [row.training_trajectories for row in rows],
        "outputs": {
            "csv": str(args.output_dir / "table9_tuned_trajectory_values.csv"),
            "markdown": str(args.output_dir / "table9_tuned_trajectory_values.md"),
            "latex": str(args.output_dir / "table9_tuned_trajectory_values.tex"),
        },
        "cli_args": {
            "source_dir": str(args.source_dir),
            "no_run": args.no_run,
            "resume": args.resume,
            "n_jobs": args.n_jobs,
        },
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.no_run:
        run_experiment(args)

    table_values_path = args.experiment_output_dir / TABLE_VALUES_FILE
    experiment_rows = load_experiment_rows(table_values_path)
    table_rows = build_table_rows(experiment_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = [table_row_to_csv(row) for row in table_rows]
    write_csv(args.output_dir / "table9_tuned_trajectory_values.csv", csv_rows)
    write_markdown(args.output_dir / "table9_tuned_trajectory_values.md", table_rows)
    latex = make_latex(table_rows)
    (args.output_dir / "table9_tuned_trajectory_values.tex").write_text(latex, encoding="utf-8")
    write_metadata(args.output_dir / "table9_tuned_trajectory_metadata.json", args, table_values_path, table_rows)

    print("Generated Table 9 files:")
    print(args.output_dir / "table9_tuned_trajectory_values.csv")
    print(args.output_dir / "table9_tuned_trajectory_values.md")
    print(args.output_dir / "table9_tuned_trajectory_values.tex")
    print(args.output_dir / "table9_tuned_trajectory_metadata.json")


if __name__ == "__main__":
    main()
