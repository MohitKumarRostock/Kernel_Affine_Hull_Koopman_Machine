#!/usr/bin/env python3
"""Generate manuscript Table 6: Duffing closure-estimator comparison.

Place this wrapper in the same folder as the manuscript experiment scripts. It
runs, or reuses, Experiment 05 and formats the table values used in the paper:

  experiment_05_duffing_closure_estimators.py

Outputs
-------
By default, writes:

  kahkm_table6_estimators/table6_estimators_values.csv
  kahkm_table6_estimators/table6_estimators_values.md
  kahkm_table6_estimators/table6_estimators_values.tex
  kahkm_table6_estimators/table6_estimators_metadata.json

Usage
-----
From the folder containing the experiment scripts:

  python step_05_generate_table6_estimators.py

Reuse existing wrapper outputs:

  python step_05_generate_table6_estimators.py --no-run

Reuse explicit summary files:

  python step_05_generate_table6_estimators.py --no-run \
    --one-step-summary kahkm_table6_estimators/exp05_duffing_estimators/experiment_05_one_step_summary.csv \
    --multistep-summary kahkm_table6_estimators/exp05_duffing_estimators/experiment_05_multistep_summary.csv
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
from typing import Mapping, Sequence, TypeAlias, cast

CsvRow: TypeAlias = dict[str, str]

ESTIMATOR_ORDER: tuple[str, str, str, str, str] = (
    "nlms",
    "ols",
    "ridge_0.0001",
    "ridge_0.01",
    "row_stochastic_nlms",
)

ESTIMATOR_LABELS: dict[str, str] = {
    "nlms": "NLMS",
    "ols": "OLS",
    "ridge_0.0001": r"Ridge $10^{-4}$",
    "ridge_0.01": r"Ridge $10^{-2}$",
    "row_stochastic_nlms": "Row-stochastic NLMS",
}


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    output_dir: Path
    no_run: bool
    one_step_summary: Path | None
    multistep_summary: Path | None
    python_executable: str
    force: bool


@dataclass(frozen=True)
class Metric:
    mean: float
    std: float


@dataclass(frozen=True)
class TableRow:
    estimator_key: str
    estimator_label: str
    test_e1: Metric
    test_r2: Metric
    e200: Metric

    def to_csv_row(self) -> CsvRow:
        return {
            "estimator": self.estimator_key,
            "estimator_label": self.estimator_label.replace("$", ""),
            "test_E1_mean": _float_to_csv(self.test_e1.mean),
            "test_E1_std": _float_to_csv(self.test_e1.std),
            "test_R2_mean": _float_to_csv(self.test_r2.mean),
            "test_R2_std": _float_to_csv(self.test_r2.std),
            "E200_mean": _float_to_csv(self.e200.mean),
            "E200_std": _float_to_csv(self.e200.std),
            "test_E1_table_cell": _format_sci(self.test_e1.mean, digits=2),
            "test_R2_table_cell": _format_decimal(self.test_r2.mean, digits=4),
            "E200_table_cell": _format_sci(self.e200.mean, digits=3),
        }


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Generate manuscript Table 6 closure-estimator comparison.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing the manuscript experiment scripts. Default: folder containing this wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table6_estimators"),
        help="Output folder for wrapper results and, by default, experiment outputs.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not execute Experiment 05; only read existing summary CSV files.",
    )
    parser.add_argument(
        "--one-step-summary",
        type=Path,
        default=None,
        help="Optional explicit path to experiment_05_one_step_summary.csv.",
    )
    parser.add_argument(
        "--multistep-summary",
        type=Path,
        default=None,
        help="Optional explicit path to experiment_05_multistep_summary.csv.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to run the underlying experiment script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run Experiment 05 even when expected summary CSVs already exist.",
    )
    namespace = parser.parse_args()
    return CliArgs(
        source_dir=Path(cast(Path, namespace.source_dir)).resolve(),
        output_dir=Path(cast(Path, namespace.output_dir)),
        no_run=bool(namespace.no_run),
        one_step_summary=None if namespace.one_step_summary is None else Path(cast(Path, namespace.one_step_summary)),
        multistep_summary=None if namespace.multistep_summary is None else Path(cast(Path, namespace.multistep_summary)),
        python_executable=str(namespace.python_executable),
        force=bool(namespace.force),
    )


def _float_to_csv(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.17g}"


def _format_sci(value: float, *, digits: int = 3) -> str:
    if not math.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10.0 ** exponent)
    coefficient_text = f"{coefficient:.{digits - 1}f}".rstrip("0").rstrip(".")
    if "." not in coefficient_text and digits <= 2:
        coefficient_text = f"{coefficient:.1f}".rstrip("0").rstrip(".")
        if "." not in coefficient_text:
            coefficient_text += ".0"
    return rf"${coefficient_text}\times10^{{{exponent}}}$"


def _format_decimal(value: float, *, digits: int = 4) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}")
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {printable}")


def _expected_paths(args: CliArgs) -> tuple[Path, Path, Path]:
    exp_dir = args.output_dir / "exp05_duffing_estimators"
    one_step = exp_dir / "experiment_05_one_step_summary.csv"
    multistep = exp_dir / "experiment_05_multistep_summary.csv"
    return exp_dir, one_step, multistep


def _run_experiment(args: CliArgs) -> tuple[Path, Path]:
    script = args.source_dir / "experiment_05_duffing_closure_estimators.py"
    _require_file(script, "Duffing closure-estimator experiment script")
    exp_dir, one_step, multistep = _expected_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.force or not one_step.exists() or not multistep.exists():
        _run_command([args.python_executable, str(script), "--output-dir", str(exp_dir)], cwd=args.source_dir)
    else:
        print(f"Reusing Experiment 05 summaries: {one_step}, {multistep}")
    return one_step, multistep


def _first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _resolve_summaries(args: CliArgs) -> tuple[Path, Path]:
    _, default_one_step, default_multistep = _expected_paths(args)
    if args.no_run:
        one_step = args.one_step_summary or _first_existing(
            [
                default_one_step,
                args.source_dir / "kahkm_experiment_05_outputs" / "experiment_05_one_step_summary.csv",
            ]
        )
        multistep = args.multistep_summary or _first_existing(
            [
                default_multistep,
                args.source_dir / "kahkm_experiment_05_outputs" / "experiment_05_multistep_summary.csv",
            ]
        )
    else:
        one_step, multistep = _run_experiment(args)
    _require_file(one_step, "Experiment 05 one-step summary")
    _require_file(multistep, "Experiment 05 multistep summary")
    return one_step, multistep


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _validate_columns(path: Path, rows: Sequence[Mapping[str, str]], required: set[str]) -> None:
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise ValueError(f"File {path} is missing required columns: {missing}")


def _one_step_metric(rows: Sequence[Mapping[str, str]], *, estimator: str, column: str, std_column: str, path: Path) -> Metric:
    matches = [row for row in rows if row.get("estimator") == estimator and row.get("split") == "test"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one test row for estimator={estimator!r} in {path}; found {len(matches)}")
    row = matches[0]
    return Metric(mean=float(str(row[column])), std=float(str(row[std_column])))


def _multistep_metric(rows: Sequence[Mapping[str, str]], *, estimator: str, horizon: int, path: Path) -> Metric:
    matches = [
        row
        for row in rows
        if row.get("estimator") == estimator and int(float(str(row.get("horizon", "nan")))) == horizon
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one horizon={horizon} row for estimator={estimator!r} in {path}; found {len(matches)}")
    row = matches[0]
    return Metric(
        mean=float(str(row["relative_association_error_mean"])),
        std=float(str(row["relative_association_error_std"])),
    )


def _build_rows(one_step_rows: Sequence[Mapping[str, str]], multistep_rows: Sequence[Mapping[str, str]], *, one_step_path: Path, multistep_path: Path) -> list[TableRow]:
    _validate_columns(
        one_step_path,
        one_step_rows,
        {
            "estimator",
            "split",
            "closure_error_mean",
            "closure_error_std",
            "association_r2_mean",
            "association_r2_std",
        },
    )
    _validate_columns(
        multistep_path,
        multistep_rows,
        {"estimator", "horizon", "relative_association_error_mean", "relative_association_error_std"},
    )
    rows: list[TableRow] = []
    for estimator in ESTIMATOR_ORDER:
        rows.append(
            TableRow(
                estimator_key=estimator,
                estimator_label=ESTIMATOR_LABELS[estimator],
                test_e1=_one_step_metric(
                    one_step_rows,
                    estimator=estimator,
                    column="closure_error_mean",
                    std_column="closure_error_std",
                    path=one_step_path,
                ),
                test_r2=_one_step_metric(
                    one_step_rows,
                    estimator=estimator,
                    column="association_r2_mean",
                    std_column="association_r2_std",
                    path=one_step_path,
                ),
                e200=_multistep_metric(multistep_rows, estimator=estimator, horizon=200, path=multistep_path),
            )
        )
    return rows


def _write_csv(path: Path, rows: Sequence[TableRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [row.to_csv_row() for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def _write_markdown(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        "# Table 6: Closure-estimator comparison on Duffing",
        "",
        "The table reports test one-step closure error, test association R2, and h=200 association rollout error.",
        "",
        "| Estimator | test E1 | test R2 | E200 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.estimator_label.replace("$", ""),
                    _format_sci(row.test_e1.mean, digits=2).replace("$", ""),
                    _format_decimal(row.test_r2.mean, digits=4),
                    _format_sci(row.e200.mean, digits=3).replace("$", ""),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Closure-estimator comparison on Duffing using fixed KAHKM features.}",
        r"\label{tab:estimators}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        "Estimator & test $E_1$ & test $R^2$ & $E_{200}$ " + r"\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.estimator_label} & "
            f"{_format_sci(row.test_e1.mean, digits=2)} & "
            f"{_format_decimal(row.test_r2.mean, digits=4)} & "
            f"{_format_sci(row.e200.mean, digits=3)} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metadata(path: Path, args: CliArgs, *, one_step_summary: Path, multistep_summary: Path) -> None:
    metadata = {
        "table": "Table 6",
        "description": "Closure-estimator comparison on Duffing using fixed KAHKM features.",
        "source_script": "experiment_05_duffing_closure_estimators.py",
        "manuscript_convention": {
            "system": "Duffing",
            "settings": "Experiment 05 defaults: C=15, omega=12, beta=0.1, 20 NLMS epochs, seeds 0,1,2.",
            "reported_cells": "test E1 from one-step summary, test R2 from one-step summary, E200 from multistep summary.",
            "estimators": list(ESTIMATOR_ORDER),
        },
        "input_summaries": {"one_step": str(one_step_summary), "multistep": str(multistep_summary)},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(args).items()},
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _print_rows(rows: Sequence[TableRow]) -> None:
    print("\nTable 6 values:")
    for row in rows:
        print(
            f"  {row.estimator_key:22s} | "
            f"test E1={_format_sci(row.test_e1.mean, digits=2):>18s} | "
            f"test R2={_format_decimal(row.test_r2.mean, digits=4):>7s} | "
            f"E200={_format_sci(row.e200.mean, digits=3):>18s}"
        )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    one_step_summary, multistep_summary = _resolve_summaries(args)
    one_step_rows = _read_csv(one_step_summary)
    multistep_rows = _read_csv(multistep_summary)
    table_rows = _build_rows(one_step_rows, multistep_rows, one_step_path=one_step_summary, multistep_path=multistep_summary)

    csv_path = args.output_dir / "table6_estimators_values.csv"
    md_path = args.output_dir / "table6_estimators_values.md"
    tex_path = args.output_dir / "table6_estimators_values.tex"
    metadata_path = args.output_dir / "table6_estimators_metadata.json"

    _write_csv(csv_path, table_rows)
    _write_markdown(md_path, table_rows)
    _write_latex(tex_path, table_rows)
    _write_metadata(metadata_path, args, one_step_summary=one_step_summary, multistep_summary=multistep_summary)
    _print_rows(table_rows)

    print("\nWrote:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {tex_path}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()
