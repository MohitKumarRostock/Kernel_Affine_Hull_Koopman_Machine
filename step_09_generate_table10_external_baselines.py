#!/usr/bin/env python3
"""Generate manuscript Table 10: external Koopman baselines.

Place this script in the same directory as the experiment scripts and run:

    python step_09_generate_table10_external_baselines.py

By default it runs/reuses the tuned Experiment 15 runners:

    run_exp15_duffing_tuned_external_baselines.py
    run_exp15_vanderpol_tuned_external_baselines.py

and then combines their table-value CSVs into one manuscript-ready Table 10.

To only format existing outputs:

    python step_09_generate_table10_external_baselines.py --no-run

Expected runner outputs:

    kahkm_exp15_duffing_tuned_external_baselines/experiment_15_tuned_duffing_table_values.csv
    kahkm_exp15_vanderpol_tuned_external_baselines/experiment_15_tuned_vanderpol_table_values.csv

Generated outputs:

    kahkm_table10_external_baselines/table10_external_baselines_values.csv
    kahkm_table10_external_baselines/table10_external_baselines_values.md
    kahkm_table10_external_baselines/table10_external_baselines_values.tex
    kahkm_table10_external_baselines/table10_external_baselines_metadata.json
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


RawRow = dict[str, str]
StringRow = dict[str, str]


METHOD_ORDER: tuple[str, ...] = (
    "kahkm_nlms",
    "state_dmd",
    "state_edmd_poly2",
    "state_edmd_poly3",
)

METHOD_LABELS: dict[str, str] = {
    "kahkm_nlms": "KAHKM + NLMS",
    "state_dmd": "State DMD",
    "state_edmd_poly2": "State EDMD poly2",
    "state_edmd_poly3": "State EDMD poly3",
}

SYSTEM_LABELS: dict[str, str] = {
    "duffing": "Duffing",
    "vanderpol": "Van der Pol",
}


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    output_dir: Path
    python_executable: str
    no_run: bool
    skip_existing: bool
    duffing_runner: str
    vanderpol_runner: str
    duffing_output_dir: Path
    vanderpol_output_dir: Path
    digits: int


@dataclass(frozen=True)
class TableRow:
    system_key: str
    system_label: str
    method_key: str
    method_label: str
    e1_mean: float
    e50_mean: float
    e200_mean: float
    e1_cell: str
    e50_cell: str
    e200_cell: str

    def to_csv_row(self) -> StringRow:
        return {
            "system": self.system_label,
            "method": self.method_label,
            "method_key": self.method_key,
            "E1_mean": _float_for_csv(self.e1_mean),
            "E50_mean": _float_for_csv(self.e50_mean),
            "E200_mean": _float_for_csv(self.e200_mean),
            "E1_table_cell": self.e1_cell,
            "E50_table_cell": self.e50_cell,
            "E200_table_cell": self.e200_cell,
        }


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Table 10 from tuned Experiment 15 outputs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the experiment scripts. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table10_external_baselines"),
        help="Directory for generated Table 10 files.",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--no-run", action="store_true", help="Only format existing outputs.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Pass --skip-existing to tuned Experiment 15 runners. Enabled by default.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Do not pass --skip-existing to the tuned Experiment 15 runners.",
    )
    parser.add_argument(
        "--duffing-runner",
        default="run_exp15_duffing_tuned_external_baselines.py",
    )
    parser.add_argument(
        "--vanderpol-runner",
        default="run_exp15_vanderpol_tuned_external_baselines.py",
    )
    parser.add_argument(
        "--duffing-output-dir",
        type=Path,
        default=Path("kahkm_exp15_duffing_tuned_external_baselines"),
    )
    parser.add_argument(
        "--vanderpol-output-dir",
        type=Path,
        default=Path("kahkm_exp15_vanderpol_tuned_external_baselines"),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Mantissa digits in scientific table cells. Default: 2.",
    )

    ns = parser.parse_args(argv)
    skip_existing = bool(ns.skip_existing) and not bool(ns.force_rerun)

    return CliArgs(
        source_dir=Path(ns.source_dir),
        output_dir=Path(ns.output_dir),
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        skip_existing=skip_existing,
        duffing_runner=str(ns.duffing_runner),
        vanderpol_runner=str(ns.vanderpol_runner),
        duffing_output_dir=Path(ns.duffing_output_dir),
        vanderpol_output_dir=Path(ns.vanderpol_output_dir),
        digits=int(ns.digits),
    )


def _read_csv(path: Path) -> list[RawRow]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[StringRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_from_row(row: RawRow, key: str) -> float:
    text = row.get(key, "").strip()
    if not text:
        return math.nan
    return float(text)


def _float_for_csv(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.17g}"


def _scientific_cell(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return f"${mantissa:.{digits}f}\\times10^{{{exponent}}}$"


def _runner_command(args: CliArgs, runner_name: str) -> list[str]:
    command = [args.python_executable, runner_name]
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def _run_runner(args: CliArgs, runner_name: str) -> None:
    script_path = args.source_dir / runner_name
    if not script_path.exists():
        raise FileNotFoundError(f"Runner script not found: {script_path}")
    command = _runner_command(args, runner_name)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def run_experiments(args: CliArgs) -> None:
    if args.no_run:
        return
    _run_runner(args, args.duffing_runner)
    _run_runner(args, args.vanderpol_runner)


def _table_values_path(source_dir: Path, output_dir: Path, system_key: str) -> Path:
    filename = f"experiment_15_tuned_{system_key}_table_values.csv"
    return source_dir / output_dir / filename


def _rows_for_system(
    *,
    source_dir: Path,
    output_dir: Path,
    system_key: str,
    digits: int,
) -> list[TableRow]:
    csv_path = _table_values_path(source_dir, output_dir, system_key)
    raw_rows = _read_csv(csv_path)
    by_method = {row.get("method", "").strip(): row for row in raw_rows}
    rows: list[TableRow] = []

    for method_key in METHOD_ORDER:
        if method_key not in by_method:
            raise KeyError(f"Method {method_key!r} not found in {csv_path}")
        raw = by_method[method_key]
        e1 = _float_from_row(raw, "E1_mean")
        e50 = _float_from_row(raw, "E50_mean")
        e200 = _float_from_row(raw, "E200_mean")
        rows.append(
            TableRow(
                system_key=system_key,
                system_label=SYSTEM_LABELS[system_key],
                method_key=method_key,
                method_label=METHOD_LABELS[method_key],
                e1_mean=e1,
                e50_mean=e50,
                e200_mean=e200,
                e1_cell=_scientific_cell(e1, digits),
                e50_cell=_scientific_cell(e50, digits),
                e200_cell=_scientific_cell(e200, digits),
            )
        )
    return rows


def build_table_rows(args: CliArgs) -> list[TableRow]:
    rows: list[TableRow] = []
    rows.extend(
        _rows_for_system(
            source_dir=args.source_dir,
            output_dir=args.duffing_output_dir,
            system_key="duffing",
            digits=args.digits,
        )
    )
    rows.extend(
        _rows_for_system(
            source_dir=args.source_dir,
            output_dir=args.vanderpol_output_dir,
            system_key="vanderpol",
            digits=args.digits,
        )
    )
    return rows


def write_markdown(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        "| System | Method | $E_1$ | $E_{50}$ | $E_{200}$ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.system_label} | {row.method_label} | "
            f"{row.e1_cell} | {row.e50_cell} | {row.e200_cell} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{External Koopman baselines evaluated in KAHKM association space on unseen trajectories. Duffing rows use the tuned KAHKM association map $C=10,\omega=2$; Van der Pol rows use the noise-aware tuned KAHKM association map $C=10,\omega=0.25$. All rows are averaged over three train/test trajectories.}",
        r"\label{tab:external_baselines}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"System & Method & $E_1$ & $E_{50}$ & $E_{200}$ \\",
        r"\midrule",
    ]
    last_system = ""
    for row in rows:
        if last_system == "Duffing" and row.system_label == "Van der Pol":
            lines.append(r"\midrule")
        lines.append(
            f"{row.system_label} & {row.method_label} & "
            f"{row.e1_cell} & {row.e50_cell} & {row.e200_cell} \\\\"
        )
        last_system = row.system_label
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(path: Path, args: CliArgs, rows: Sequence[TableRow]) -> None:
    payload = {
        "table": "Table 10",
        "description": "External Koopman baselines evaluated in KAHKM association space.",
        "arguments": asdict(args) | {
            "source_dir": str(args.source_dir),
            "output_dir": str(args.output_dir),
            "duffing_output_dir": str(args.duffing_output_dir),
            "vanderpol_output_dir": str(args.vanderpol_output_dir),
        },
        "input_files": [
            str(_table_values_path(args.source_dir, args.duffing_output_dir, "duffing")),
            str(_table_values_path(args.source_dir, args.vanderpol_output_dir, "vanderpol")),
        ],
        "n_rows": len(rows),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiments(args)
    rows = build_table_rows(args)
    csv_rows = [row.to_csv_row() for row in rows]

    _write_csv(args.output_dir / "table10_external_baselines_values.csv", csv_rows)
    write_markdown(args.output_dir / "table10_external_baselines_values.md", rows)
    write_latex(args.output_dir / "table10_external_baselines_values.tex", rows)
    write_metadata(args.output_dir / "table10_external_baselines_metadata.json", args, rows)

    print(f"Wrote {args.output_dir / 'table10_external_baselines_values.csv'}")
    print(f"Wrote {args.output_dir / 'table10_external_baselines_values.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
