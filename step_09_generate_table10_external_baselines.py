#!/usr/bin/env python3
"""Generate manuscript Table 10: centered external Koopman baselines.

This generator combines the tuned Experiment 15 outputs for Duffing and
Van der Pol into a manuscript-facing comparison based on centered association
R^2. The uncentered relative association errors E_h are retained in the
machine-readable CSV and metadata for provenance.

Default Experiment 15 sources
-----------------------------
Duffing:
    kahkm_exp15_duffing_tuned_external_baselines/
        experiment_15_tuned_duffing_table_values.csv
        experiment_15_tuned_duffing_fit_table.csv
    tuned association map: (C, omega) = (10, 2)

Van der Pol:
    kahkm_exp15_vanderpol_retvar_external_baselines/
        experiment_15_tuned_vanderpol_table_values.csv
        experiment_15_tuned_vanderpol_fit_table.csv
    retained-variation / noise-aware association map: (C, omega) = (25, 4)

Generated outputs
-----------------
    kahkm_table10_centered_external_baselines/
        table10_external_baselines_values.csv
        table10_external_baselines_values.md
        table10_external_baselines_values.tex
        table10_external_baselines_metadata.json

Use --no-run to format already-computed Experiment 15 outputs without rerunning
either experiment.
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
    expected_duffing_c: int
    expected_duffing_omega: float
    expected_vanderpol_c: int
    expected_vanderpol_omega: float
    r2_digits: int
    e_digits: int


@dataclass(frozen=True)
class TableRow:
    system_key: str
    system_label: str
    method_key: str
    method_label: str

    r2_1_mean: float
    r2_1_std: float
    r2_50_mean: float
    r2_50_std: float
    r2_200_mean: float
    r2_200_std: float

    e1_mean: float
    e1_std: float
    e50_mean: float
    e50_std: float
    e200_mean: float
    e200_std: float

    r2_1_cell: str
    r2_50_cell: str
    r2_200_cell: str

    def to_csv_row(self, *, e_digits: int) -> StringRow:
        return {
            "system": self.system_label,
            "system_key": self.system_key,
            "method": self.method_label,
            "method_key": self.method_key,
            "R2_1_mean": _float_for_csv(self.r2_1_mean),
            "R2_1_std": _float_for_csv(self.r2_1_std),
            "R2_50_mean": _float_for_csv(self.r2_50_mean),
            "R2_50_std": _float_for_csv(self.r2_50_std),
            "R2_200_mean": _float_for_csv(self.r2_200_mean),
            "R2_200_std": _float_for_csv(self.r2_200_std),
            "R2_1_table_cell": self.r2_1_cell,
            "R2_50_table_cell": self.r2_50_cell,
            "R2_200_table_cell": self.r2_200_cell,
            # Legacy uncentered relative-association errors retained for provenance.
            "E1_mean": _float_for_csv(self.e1_mean),
            "E1_std": _float_for_csv(self.e1_std),
            "E50_mean": _float_for_csv(self.e50_mean),
            "E50_std": _float_for_csv(self.e50_std),
            "E200_mean": _float_for_csv(self.e200_mean),
            "E200_std": _float_for_csv(self.e200_std),
            "E1_table_cell": _mean_std_scientific_cell(self.e1_mean, self.e1_std, e_digits),
            "E50_table_cell": _mean_std_scientific_cell(self.e50_mean, self.e50_std, e_digits),
            "E200_table_cell": _mean_std_scientific_cell(self.e200_mean, self.e200_std, e_digits),
        }


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate centered manuscript Table 10 from tuned Experiment 15 outputs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing experiment scripts and outputs. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table10_centered_external_baselines"),
        help="Directory for generated centered Table 10 files.",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--no-run", action="store_true", help="Only format existing outputs.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Pass --skip-existing to Experiment 15 runners. Enabled by default.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Do not pass --skip-existing to the Experiment 15 runners.",
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
        default=Path("kahkm_exp15_vanderpol_retvar_external_baselines"),
    )

    parser.add_argument("--expected-duffing-c", type=int, default=10)
    parser.add_argument("--expected-duffing-omega", type=float, default=2.0)
    parser.add_argument("--expected-vanderpol-c", type=int, default=25)
    parser.add_argument("--expected-vanderpol-omega", type=float, default=4.0)

    parser.add_argument(
        "--r2-digits",
        type=int,
        default=4,
        help="Decimal places for mean±SD R^2 cells. Default: 4.",
    )
    parser.add_argument(
        "--e-digits",
        type=int,
        default=2,
        help="Mantissa digits for provenance E_h mean±SD cells in the generated CSV. Default: 2.",
    )

    ns = parser.parse_args(argv)
    skip_existing = bool(ns.skip_existing) and not bool(ns.force_rerun)

    source_dir = Path(ns.source_dir).resolve()
    output_dir = Path(ns.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir

    return CliArgs(
        source_dir=source_dir,
        output_dir=output_dir,
        python_executable=str(ns.python_executable),
        no_run=bool(ns.no_run),
        skip_existing=skip_existing,
        duffing_runner=str(ns.duffing_runner),
        vanderpol_runner=str(ns.vanderpol_runner),
        duffing_output_dir=Path(ns.duffing_output_dir),
        vanderpol_output_dir=Path(ns.vanderpol_output_dir),
        expected_duffing_c=int(ns.expected_duffing_c),
        expected_duffing_omega=float(ns.expected_duffing_omega),
        expected_vanderpol_c=int(ns.expected_vanderpol_c),
        expected_vanderpol_omega=float(ns.expected_vanderpol_omega),
        r2_digits=int(ns.r2_digits),
        e_digits=int(ns.e_digits),
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


def _finite_float_from_row(row: RawRow, key: str, *, context: str) -> float:
    text = row.get(key, "").strip()
    if not text:
        raise ValueError(
            f"Missing required column {key!r} for {context}. "
            f"Available columns: {sorted(row)}"
        )
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} for {context}: {text!r}")
    return value


def _finite_int_from_row(row: RawRow, key: str, *, context: str) -> int:
    return int(round(_finite_float_from_row(row, key, context=context)))


def _float_for_csv(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.17g}"


def _r2_cell(mean_value: float, std_value: float, digits: int) -> str:
    return f"${mean_value:.{digits}f}\\pm{std_value:.{digits}f}$"


def _mean_std_scientific_cell(mean_value: float, std_value: float, digits: int) -> str:
    if not math.isfinite(mean_value) or not math.isfinite(std_value):
        return "--"
    scale_value = max(abs(mean_value), abs(std_value))
    if scale_value == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(scale_value)))
    scale = 10.0 ** exponent
    return (
        f"$({mean_value / scale:.{digits}f}\\pm"
        f"{std_value / scale:.{digits}f})\\times10^{{{exponent}}}$"
    )


def _resolve_under_source(source_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else source_dir / path


def _table_values_path(source_dir: Path, output_dir: Path, system_key: str) -> Path:
    filename = f"experiment_15_tuned_{system_key}_table_values.csv"
    return _resolve_under_source(source_dir, output_dir) / filename


def _fit_table_path(source_dir: Path, output_dir: Path, system_key: str) -> Path:
    filename = f"experiment_15_tuned_{system_key}_fit_table.csv"
    return _resolve_under_source(source_dir, output_dir) / filename


def _runner_command(
    args: CliArgs,
    *,
    runner_name: str,
    output_dir: Path,
) -> list[str]:
    command = [
        args.python_executable,
        str(args.source_dir / runner_name),
        "--output-dir",
        str(_resolve_under_source(args.source_dir, output_dir)),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def _run_runner(
    args: CliArgs,
    *,
    runner_name: str,
    output_dir: Path,
) -> None:
    script_path = args.source_dir / runner_name
    if not script_path.exists():
        raise FileNotFoundError(f"Runner script not found: {script_path}")
    command = _runner_command(args, runner_name=runner_name, output_dir=output_dir)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def run_experiments(args: CliArgs) -> None:
    if args.no_run:
        return
    _run_runner(
        args,
        runner_name=args.duffing_runner,
        output_dir=args.duffing_output_dir,
    )
    _run_runner(
        args,
        runner_name=args.vanderpol_runner,
        output_dir=args.vanderpol_output_dir,
    )


def _validate_system_config(
    args: CliArgs,
    *,
    system_key: str,
    output_dir: Path,
    expected_c: int,
    expected_omega: float,
) -> dict[str, object]:
    fit_path = _fit_table_path(args.source_dir, output_dir, system_key)
    rows = _read_csv(fit_path)
    if not rows:
        raise ValueError(f"No fit-summary rows found in {fit_path}")

    observed_pairs: set[tuple[int, float]] = set()
    for index, row in enumerate(rows):
        context = f"{system_key} fit row {index + 1}"
        c = _finite_int_from_row(row, "n_clusters", context=context)
        omega = _finite_float_from_row(row, "omega", context=context)
        observed_pairs.add((c, omega))

    if len(observed_pairs) != 1:
        raise ValueError(
            f"Expected one {system_key} association-map configuration in {fit_path}, "
            f"found {sorted(observed_pairs)}"
        )

    observed_c, observed_omega = next(iter(observed_pairs))
    if observed_c != int(expected_c) or not math.isclose(
        observed_omega,
        float(expected_omega),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{system_key} configuration mismatch: expected "
            f"(C, omega)=({expected_c}, {expected_omega:g}), observed "
            f"({observed_c}, {observed_omega:g}) in {fit_path}"
        )

    return {
        "system": system_key,
        "fit_table": str(fit_path),
        "n_fit_rows": len(rows),
        "n_clusters": observed_c,
        "omega": observed_omega,
    }


def validate_input_configs(args: CliArgs) -> dict[str, dict[str, object]]:
    return {
        "duffing": _validate_system_config(
            args,
            system_key="duffing",
            output_dir=args.duffing_output_dir,
            expected_c=args.expected_duffing_c,
            expected_omega=args.expected_duffing_omega,
        ),
        "vanderpol": _validate_system_config(
            args,
            system_key="vanderpol",
            output_dir=args.vanderpol_output_dir,
            expected_c=args.expected_vanderpol_c,
            expected_omega=args.expected_vanderpol_omega,
        ),
    }


def _rows_for_system(
    *,
    args: CliArgs,
    output_dir: Path,
    system_key: str,
) -> list[TableRow]:
    csv_path = _table_values_path(args.source_dir, output_dir, system_key)
    raw_rows = _read_csv(csv_path)
    by_method = {row.get("method", "").strip(): row for row in raw_rows}

    missing = [method for method in METHOD_ORDER if method not in by_method]
    if missing:
        raise KeyError(f"Methods {missing!r} not found in {csv_path}")

    rows: list[TableRow] = []
    for method_key in METHOD_ORDER:
        raw = by_method[method_key]
        context = f"{system_key}/{method_key}"

        r2_1_mean = _finite_float_from_row(raw, "R2_1_mean", context=context)
        r2_1_std = _finite_float_from_row(raw, "R2_1_std", context=context)
        r2_50_mean = _finite_float_from_row(raw, "R2_50_mean", context=context)
        r2_50_std = _finite_float_from_row(raw, "R2_50_std", context=context)
        r2_200_mean = _finite_float_from_row(raw, "R2_200_mean", context=context)
        r2_200_std = _finite_float_from_row(raw, "R2_200_std", context=context)

        e1_mean = _finite_float_from_row(raw, "E1_mean", context=context)
        e1_std = _finite_float_from_row(raw, "E1_std", context=context)
        e50_mean = _finite_float_from_row(raw, "E50_mean", context=context)
        e50_std = _finite_float_from_row(raw, "E50_std", context=context)
        e200_mean = _finite_float_from_row(raw, "E200_mean", context=context)
        e200_std = _finite_float_from_row(raw, "E200_std", context=context)

        rows.append(
            TableRow(
                system_key=system_key,
                system_label=SYSTEM_LABELS[system_key],
                method_key=method_key,
                method_label=METHOD_LABELS[method_key],
                r2_1_mean=r2_1_mean,
                r2_1_std=r2_1_std,
                r2_50_mean=r2_50_mean,
                r2_50_std=r2_50_std,
                r2_200_mean=r2_200_mean,
                r2_200_std=r2_200_std,
                e1_mean=e1_mean,
                e1_std=e1_std,
                e50_mean=e50_mean,
                e50_std=e50_std,
                e200_mean=e200_mean,
                e200_std=e200_std,
                r2_1_cell=_r2_cell(r2_1_mean, r2_1_std, args.r2_digits),
                r2_50_cell=_r2_cell(r2_50_mean, r2_50_std, args.r2_digits),
                r2_200_cell=_r2_cell(r2_200_mean, r2_200_std, args.r2_digits),
            )
        )
    return rows


def build_table_rows(args: CliArgs) -> list[TableRow]:
    rows: list[TableRow] = []
    rows.extend(
        _rows_for_system(
            args=args,
            output_dir=args.duffing_output_dir,
            system_key="duffing",
        )
    )
    rows.extend(
        _rows_for_system(
            args=args,
            output_dir=args.vanderpol_output_dir,
            system_key="vanderpol",
        )
    )
    return rows


def write_markdown(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        "| System | Method | $R^2_1$ | $R^2_{50}$ | $R^2_{200}$ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.system_label} | {row.method_label} | "
            f"{row.r2_1_cell} | {row.r2_50_cell} | {row.r2_200_cell} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: Sequence[TableRow], args: CliArgs) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{External Koopman baselines evaluated in KAHKM association space "
            r"on unseen trajectories using centered association $R^2$. "
            rf"Duffing rows use the tuned KAHKM association map $(C,\omega)=({args.expected_duffing_c},{args.expected_duffing_omega:g})$; "
            rf"Van der Pol rows use the retained-variation/noise-aware map $(C,\omega)=({args.expected_vanderpol_c},{args.expected_vanderpol_omega:g})$. "
            r"Entries are mean $\pm$ standard deviation over three train/test trajectory pairs. "
            r"The corresponding uncentered relative association errors are retained in the generated CSV for provenance.}"
        ),
        r"\label{tab:external_baselines}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"System & Method & $R^2_1$ & $R^2_{50}$ & $R^2_{200}$ \\",
        r"\midrule",
    ]
    last_system = ""
    for row in rows:
        if last_system == "Duffing" and row.system_label == "Van der Pol":
            lines.append(r"\midrule")
        lines.append(
            f"{row.system_label} & {row.method_label} & "
            f"{row.r2_1_cell} & {row.r2_50_cell} & {row.r2_200_cell} \\\\"
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


def write_metadata(
    path: Path,
    args: CliArgs,
    rows: Sequence[TableRow],
    validated_configs: dict[str, dict[str, object]],
) -> None:
    payload = {
        "table": "Table 10",
        "description": (
            "Centered external Koopman-baseline comparison in KAHKM association space; "
            "legacy uncentered E_h values retained in the generated CSV."
        ),
        "metric": {
            "manuscript_columns": ["R2_1", "R2_50", "R2_200"],
            "aggregation": "mean ± standard deviation over the Experiment 15 train/test trajectories",
            "provenance_columns": [
                "E1_mean",
                "E1_std",
                "E50_mean",
                "E50_std",
                "E200_mean",
                "E200_std",
            ],
        },
        "expected_configs": {
            "duffing": {
                "C": args.expected_duffing_c,
                "omega": args.expected_duffing_omega,
            },
            "vanderpol": {
                "C": args.expected_vanderpol_c,
                "omega": args.expected_vanderpol_omega,
            },
        },
        "validated_configs": validated_configs,
        "arguments": asdict(args) | {
            "source_dir": str(args.source_dir),
            "output_dir": str(args.output_dir),
            "duffing_output_dir": str(args.duffing_output_dir),
            "vanderpol_output_dir": str(args.vanderpol_output_dir),
        },
        "input_files": {
            "duffing_table_values": str(
                _table_values_path(args.source_dir, args.duffing_output_dir, "duffing")
            ),
            "duffing_fit_table": str(
                _fit_table_path(args.source_dir, args.duffing_output_dir, "duffing")
            ),
            "vanderpol_table_values": str(
                _table_values_path(args.source_dir, args.vanderpol_output_dir, "vanderpol")
            ),
            "vanderpol_fit_table": str(
                _fit_table_path(args.source_dir, args.vanderpol_output_dir, "vanderpol")
            ),
        },
        "n_rows": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiments(args)
    validated_configs = validate_input_configs(args)
    rows = build_table_rows(args)
    csv_rows = [row.to_csv_row(e_digits=args.e_digits) for row in rows]

    csv_path = args.output_dir / "table10_external_baselines_values.csv"
    md_path = args.output_dir / "table10_external_baselines_values.md"
    tex_path = args.output_dir / "table10_external_baselines_values.tex"
    metadata_path = args.output_dir / "table10_external_baselines_metadata.json"

    _write_csv(csv_path, csv_rows)
    write_markdown(md_path, rows)
    write_latex(tex_path, rows, args)
    write_metadata(metadata_path, args, rows, validated_configs)

    print("Wrote:")
    for path in (csv_path, md_path, tex_path, metadata_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
