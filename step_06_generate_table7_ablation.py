#!/usr/bin/env python3
"""Generate manuscript Table 7 (abstraction ablation).

Place this script in the same directory as the experiment scripts and run:

    python step_06_generate_table7_ablation.py

It runs or reuses:
    experiment_06_duffing_abstraction_ablation.py
    experiment_07_vanderpol_abstraction_ablation.py

Outputs are written to:
    kahkm_table7_ablation/table7_ablation_values.csv
    kahkm_table7_ablation/table7_ablation_values.md
    kahkm_table7_ablation/table7_ablation_values.tex
    kahkm_table7_ablation/table7_ablation_metadata.json
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


@dataclass(frozen=True)
class TableRow:
    system: str
    abstraction_key: str
    abstraction_label: str
    test_e1: float
    test_r2: float
    e200: float

    def as_csv_row(self) -> dict[str, str]:
        return {
            "system": self.system,
            "abstraction": self.abstraction_label,
            "abstraction_key": self.abstraction_key,
            "test_E1_mean": format_float(self.test_e1),
            "test_R2_mean": format_float(self.test_r2),
            "E200_mean": format_float(self.e200),
            "test_E1_table_cell": error_cell(self.test_e1),
            "test_R2_table_cell": r2_cell(self.test_r2),
            "E200_table_cell": error_cell(self.e200),
        }


@dataclass(frozen=True)
class ExperimentSpec:
    system: str
    script_name: str
    output_dir: Path
    one_step_summary_name: str
    multistep_summary_name: str
    extra_args: tuple[str, ...]


ABSTRACTION_ORDER: tuple[tuple[str, str], ...] = (
    ("kahkm_folding_nlms", "KAHKM folding"),
    ("kmeans_rbf_nlms", "KMeans RBF"),
    ("kmeans_distance_nlms", "KMeans distance"),
    ("kmeans_hard_nlms", "KMeans hard"),
)


class CliArgs(argparse.Namespace):
    experiment_dir: str
    output_dir: str
    no_run: bool
    force: bool
    python_executable: str
    duffing_output_dir: str
    vanderpol_output_dir: str
    vanderpol_n_clusters: int
    vanderpol_omega: float


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Table 7 from Experiments 06 and 07."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=".",
        help="Directory containing the experiment_06 and experiment_07 scripts. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kahkm_table7_ablation",
        help="Directory for the generated Table 7 files.",
    )
    parser.add_argument(
        "--duffing-output-dir",
        type=str,
        default="kahkm_experiment_06_outputs",
        help="Output directory used by experiment_06_duffing_abstraction_ablation.py.",
    )
    parser.add_argument(
        "--vanderpol-output-dir",
        type=str,
        default="kahkm_experiment_07_outputs",
        help="Output directory used by experiment_07_vanderpol_abstraction_ablation.py.",
    )
    parser.add_argument(
        "--vanderpol-n-clusters",
        type=int,
        default=20,
        help="Van der Pol diagnostic number of regimes. Manuscript default diagnostic setting is C=20.",
    )
    parser.add_argument(
        "--vanderpol-omega",
        type=float,
        default=4.0,
        help="Van der Pol diagnostic association sharpness. Manuscript default diagnostic setting is omega=4.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run experiments; only format existing CSV outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run experiments even if their summary CSVs already exist.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to run the experiment scripts.",
    )
    return parser.parse_args(argv, namespace=CliArgs())


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.17g}"


def r2_cell(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def error_cell(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    exponent = int(math.floor(math.log10(abs_value)))
    mantissa = value / (10.0 ** exponent)
    mantissa_text = f"{mantissa:.2f}".rstrip("0").rstrip(".")
    return rf"${mantissa_text}\times10^{{{exponent}}}$"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def run_experiment(spec: ExperimentSpec, experiment_dir: Path, python_executable: str, *, force: bool, no_run: bool) -> None:
    script_path = experiment_dir / spec.script_name
    require_file(script_path)
    one_step_path = spec.output_dir / spec.one_step_summary_name
    multistep_path = spec.output_dir / spec.multistep_summary_name
    if no_run:
        require_file(one_step_path)
        require_file(multistep_path)
        return
    if one_step_path.exists() and multistep_path.exists() and not force:
        return
    command = [
        python_executable,
        str(script_path),
        "--output-dir",
        str(spec.output_dir),
        *spec.extra_args,
    ]
    subprocess.run(command, cwd=str(experiment_dir), check=True)


def find_one_step_metric(rows: list[dict[str, str]], abstraction_key: str) -> tuple[float, float]:
    for row in rows:
        if row.get("abstraction") == abstraction_key and row.get("split") == "test":
            return float(row["closure_error_mean"]), float(row["association_r2_mean"])
    raise KeyError(f"Missing test one-step summary row for abstraction {abstraction_key!r}")


def find_e200(rows: list[dict[str, str]], abstraction_key: str) -> float:
    for row in rows:
        if row.get("abstraction") == abstraction_key and int(float(row.get("horizon", "-1"))) == 200:
            return float(row["relative_association_error_mean"])
    raise KeyError(f"Missing h=200 multistep summary row for abstraction {abstraction_key!r}")


def collect_table_rows(spec: ExperimentSpec) -> list[TableRow]:
    one_step_path = spec.output_dir / spec.one_step_summary_name
    multistep_path = spec.output_dir / spec.multistep_summary_name
    require_file(one_step_path)
    require_file(multistep_path)
    one_step_rows = read_csv_rows(one_step_path)
    multistep_rows = read_csv_rows(multistep_path)
    table_rows: list[TableRow] = []
    for abstraction_key, abstraction_label in ABSTRACTION_ORDER:
        test_e1, test_r2 = find_one_step_metric(one_step_rows, abstraction_key)
        e200 = find_e200(multistep_rows, abstraction_key)
        table_rows.append(
            TableRow(
                system=spec.system,
                abstraction_key=abstraction_key,
                abstraction_label=abstraction_label,
                test_e1=test_e1,
                test_r2=test_r2,
                e200=e200,
            )
        )
    return table_rows


def write_csv(path: Path, rows: list[TableRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [row.as_csv_row() for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def write_markdown(path: Path, rows: list[TableRow]) -> None:
    lines = [
        "| System | Abstraction | test $E_1$ | test $R^2$ | $E_{200}$ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        csv_row = row.as_csv_row()
        lines.append(
            f"| {row.system} | {row.abstraction_label} | {csv_row['test_E1_table_cell']} | "
            f"{csv_row['test_R2_table_cell']} | {csv_row['E200_table_cell']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[TableRow]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Abstraction ablation. All methods use the same closure estimator; only the association map changes.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"System & Abstraction & test $E_1$ & test $R^2$ & $E_{200}$ \\",
        r"\midrule",
    ]
    previous_system = ""
    for row in rows:
        if previous_system and previous_system != row.system:
            lines.append(r"\midrule")
        csv_row = row.as_csv_row()
        lines.append(
            f"{row.system} & {row.abstraction_label} & {csv_row['test_E1_table_cell']} & "
            f"{csv_row['test_R2_table_cell']} & {csv_row['E200_table_cell']} " + r"\\"
        )
        previous_system = row.system
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    duffing_output_dir = Path(args.duffing_output_dir)
    vanderpol_output_dir = Path(args.vanderpol_output_dir)
    if not duffing_output_dir.is_absolute():
        duffing_output_dir = experiment_dir / duffing_output_dir
    if not vanderpol_output_dir.is_absolute():
        vanderpol_output_dir = experiment_dir / vanderpol_output_dir

    specs = [
        ExperimentSpec(
            system="Duffing",
            script_name="experiment_06_duffing_abstraction_ablation.py",
            output_dir=duffing_output_dir,
            one_step_summary_name="experiment_06_one_step_summary.csv",
            multistep_summary_name="experiment_06_multistep_summary.csv",
            extra_args=(),
        ),
        ExperimentSpec(
            system="Van der Pol",
            script_name="experiment_07_vanderpol_abstraction_ablation.py",
            output_dir=vanderpol_output_dir,
            one_step_summary_name="experiment_07_one_step_summary.csv",
            multistep_summary_name="experiment_07_multistep_summary.csv",
            extra_args=(
                "--n-clusters",
                str(args.vanderpol_n_clusters),
                "--omega",
                str(args.vanderpol_omega),
            ),
        ),
    ]

    for spec in specs:
        run_experiment(
            spec,
            experiment_dir,
            args.python_executable,
            force=args.force,
            no_run=args.no_run,
        )

    table_rows: list[TableRow] = []
    for spec in specs:
        table_rows.extend(collect_table_rows(spec))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "table7_ablation_values.csv", table_rows)
    write_markdown(output_dir / "table7_ablation_values.md", table_rows)
    write_latex(output_dir / "table7_ablation_values.tex", table_rows)

    metadata = {
        "table": "Table 7",
        "description": "Abstraction ablation: KAHKM folding versus KMeans simplex-coordinate baselines.",
        "experiments": [
            {
                **asdict(spec),
                "output_dir": str(spec.output_dir),
                "extra_args": list(spec.extra_args),
            }
            for spec in specs
        ],
        "notes": [
            "Duffing uses Experiment 06 defaults.",
            "Van der Pol is run with the manuscript diagnostic default C and omega unless overridden.",
            "Only the association map changes; all rows use the NLMS closure estimator inside the experiment scripts.",
        ],
    }
    (output_dir / "table7_ablation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote Table 7 files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
