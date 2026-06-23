#!/usr/bin/env python3
"""Generate manuscript Table 15 from Experiment 19 residual-verified spectra outputs.

Place this script in the same directory as the experiment scripts and run:

    python step_14_generate_table15_residual_spectra.py

By default, the script runs/reuses:

    experiment_19_residual_verified_kahkm_spectra.py

with the manuscript spectral-diagnostic configuration:

    systems: Duffing and Van der Pol
    operators: NLMS and ridge least squares
    train seeds: 0, 1, 2
    test seeds: 100, 101, 102
    per-seed train snapshots: 1200   -> 3600 total train snapshots
    per-seed test snapshots: 800     -> 2400 total test snapshots
    spectral subsample: 800 held-out snapshot pairs
    Duffing tuned abstraction: C=10, omega=2
    Van der Pol robust tuned abstraction: C=10, omega=0.25
    ridge parameter: 1e-8

It then reads:

    kahkm_experiment_19_residual_verified_spectra/experiment_19_system_summary.csv

and writes:

    kahkm_table15_residual_spectra/table15_residual_spectra_values.csv
    kahkm_table15_residual_spectra/table15_residual_spectra_values.md
    kahkm_table15_residual_spectra/table15_residual_spectra_values.tex
    kahkm_table15_residual_spectra/table15_residual_spectra_metadata.json

Use --no-run to only format existing Experiment 19 outputs.
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
from typing import Final, Sequence


SYSTEM_ORDER: Final[tuple[str, ...]] = ("duffing", "vanderpol")
OPERATOR_ORDER: Final[tuple[str, ...]] = ("nlms", "ridge_ls")
SYSTEM_LABELS: Final[dict[str, str]] = {
    "duffing": "Duffing",
    "vanderpol": "Van der Pol",
}
OPERATOR_LABELS: Final[dict[str, str]] = {
    "nlms": "NLMS",
    "ridge_ls": "Ridge-LS",
}
REQUIRED_THRESHOLDS: Final[tuple[float, ...]] = (1e-2, 5e-2, 1e-1)


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    experiment_output_dir: Path
    output_dir: Path
    python_executable: str
    runner_script: str
    no_run: bool
    skip_existing: bool
    with_plots: bool
    systems: tuple[str, ...]
    operators: tuple[str, ...]
    train_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    n_steps_train: int
    n_steps_test: int
    duffing_c: int
    duffing_omega: float
    vanderpol_c: int
    vanderpol_omega: float
    tau: float
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    ridge: float
    spectral_subsample: int
    residual_thresholds: tuple[float, ...]
    random_state: int
    kmeans_kind: str
    kmeans_batch_size: int
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int


@dataclass(frozen=True)
class TableRow:
    system: str
    operator: str
    test_e1: float
    test_r2: float
    spectral_radius: float
    min_residual: float
    median_residual: float
    modes_le_1e_2: int
    modes_le_5e_2: int
    modes_le_1e_1: int


def parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def parse_float_tuple(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one float is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Table 15 from residual-verified KAHKM spectra outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=None,
        help="Experiment 19 output directory. Default: <source-dir>/kahkm_experiment_19_residual_verified_spectra.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table15_residual_spectra"),
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--runner-script", default="experiment_19_residual_verified_kahkm_spectra.py")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--with-plots", action="store_true", help="Do not pass --no-plots to Experiment 19.")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEM_ORDER))
    parser.add_argument("--operators", nargs="+", default=list(OPERATOR_ORDER))
    parser.add_argument("--train-seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--test-seeds", nargs="+", default=["100", "101", "102"])
    parser.add_argument("--n-steps-train", type=int, default=1200)
    parser.add_argument("--n-steps-test", type=int, default=800)
    parser.add_argument("--duffing-c", type=int, default=10)
    parser.add_argument("--duffing-omega", type=float, default=2.0)
    parser.add_argument("--vanderpol-c", type=int, default=10)
    parser.add_argument("--vanderpol-omega", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--spectral-subsample", type=int, default=800)
    parser.add_argument("--residual-thresholds", nargs="+", default=["1e-3", "1e-2", "5e-2", "1e-1"])
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--kmeans-kind", default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)

    ns = parser.parse_args()
    source_dir = Path(ns.source_dir).resolve()
    experiment_output_dir = (
        Path(ns.experiment_output_dir).resolve()
        if ns.experiment_output_dir
        else source_dir / "kahkm_experiment_19_residual_verified_spectra"
    )
    output_dir = Path(ns.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir

    systems = tuple(str(item) for item in ns.systems)
    operators = tuple(str(item) for item in ns.operators)
    unknown_systems = sorted(set(systems).difference(SYSTEM_LABELS))
    unknown_operators = sorted(set(operators).difference(OPERATOR_LABELS))
    if unknown_systems:
        raise ValueError(f"Unsupported systems for Table 15: {unknown_systems}")
    if unknown_operators:
        raise ValueError(f"Unsupported operators for Table 15: {unknown_operators}")

    return CliArgs(
        source_dir=source_dir,
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        python_executable=str(ns.python_executable),
        runner_script=str(ns.runner_script),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        with_plots=bool(ns.with_plots),
        systems=systems,
        operators=operators,
        train_seeds=parse_int_tuple(ns.train_seeds),
        test_seeds=parse_int_tuple(ns.test_seeds),
        n_steps_train=int(ns.n_steps_train),
        n_steps_test=int(ns.n_steps_test),
        duffing_c=int(ns.duffing_c),
        duffing_omega=float(ns.duffing_omega),
        vanderpol_c=int(ns.vanderpol_c),
        vanderpol_omega=float(ns.vanderpol_omega),
        tau=float(ns.tau),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        ridge=float(ns.ridge),
        spectral_subsample=int(ns.spectral_subsample),
        residual_thresholds=parse_float_tuple(ns.residual_thresholds),
        random_state=int(ns.random_state),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        max_train_per_cluster=int(ns.max_train_per_cluster),
    )


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return
    runner = args.source_dir / args.runner_script
    if not runner.exists():
        raise FileNotFoundError(f"Could not find runner script: {runner}")
    command = [
        args.python_executable,
        str(runner),
        "--systems",
        *args.systems,
        "--operators",
        *args.operators,
        "--train-seeds",
        *[str(seed) for seed in args.train_seeds],
        "--test-seeds",
        *[str(seed) for seed in args.test_seeds],
        "--n-steps-train",
        str(args.n_steps_train),
        "--n-steps-test",
        str(args.n_steps_test),
        "--output-dir",
        str(args.experiment_output_dir),
        "--duffing-c",
        str(args.duffing_c),
        "--duffing-omega",
        str(args.duffing_omega),
        "--vanderpol-c",
        str(args.vanderpol_c),
        "--vanderpol-omega",
        str(args.vanderpol_omega),
        "--tau",
        str(args.tau),
        "--subspace-dim",
        str(args.subspace_dim),
        "--nb",
        str(args.nb),
        "--beta",
        str(args.beta),
        "--nlms-epochs",
        str(args.nlms_epochs),
        "--ridge",
        str(args.ridge),
        "--spectral-subsample",
        str(args.spectral_subsample),
        "--residual-thresholds",
        *[str(threshold) for threshold in args.residual_thresholds],
        "--random-state",
        str(args.random_state),
        "--kmeans-kind",
        args.kmeans_kind,
        "--kmeans-batch-size",
        str(args.kmeans_batch_size),
        "--batch-size",
        str(args.batch_size),
        "--n-jobs",
        str(args.n_jobs),
        "--max-train-per-cluster",
        str(args.max_train_per_cluster),
    ]
    if not args.with_plots:
        command.append("--no-plots")
    if args.skip_existing:
        command.append("--skip-existing")
    print("Running Experiment 19 residual spectra:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str, *, context: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing column {key!r} for {context}. Available columns: {sorted(row.keys())}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value in {key!r} for {context}: {value!r}")
    return result


def finite_int(row: dict[str, str], key: str, *, context: str) -> int:
    return int(round(finite_float(row, key, context=context)))


def threshold_key(threshold: float) -> str:
    return f"modes_with_residual_le_{threshold:g}"


def find_summary_row(rows: Sequence[dict[str, str]], *, system: str, operator: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("system") == system and row.get("operator") == operator]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one summary row for {system}/{operator}, found {len(matches)}.")
    return matches[0]


def build_table_rows(args: CliArgs) -> list[TableRow]:
    rows = read_csv_rows(args.experiment_output_dir / "experiment_19_system_summary.csv")
    table_rows: list[TableRow] = []
    for system in SYSTEM_ORDER:
        if system not in args.systems:
            continue
        for operator in OPERATOR_ORDER:
            if operator not in args.operators:
                continue
            row = find_summary_row(rows, system=system, operator=operator)
            context = f"{system}/{operator}"
            table_rows.append(
                TableRow(
                    system=system,
                    operator=operator,
                    test_e1=finite_float(row, "test_closure_error", context=context),
                    test_r2=finite_float(row, "test_association_r2", context=context),
                    spectral_radius=finite_float(row, "spectral_radius", context=context),
                    min_residual=finite_float(row, "min_rkhs_residual", context=context),
                    median_residual=finite_float(row, "median_rkhs_residual", context=context),
                    modes_le_1e_2=finite_int(row, threshold_key(1e-2), context=context),
                    modes_le_5e_2=finite_int(row, threshold_key(5e-2), context=context),
                    modes_le_1e_1=finite_int(row, threshold_key(1e-1), context=context),
                )
            )
    return table_rows


def sci_plain(value: float) -> str:
    return f"{value:.2e}"


def sci_latex(value: float) -> str:
    if value == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


def table_row_to_csv(row: TableRow) -> dict[str, str]:
    return {
        "system": SYSTEM_LABELS[row.system],
        "operator": OPERATOR_LABELS[row.operator],
        "test_E1": sci_plain(row.test_e1),
        "test_R2": f"{row.test_r2:.4f}",
        "spectral_radius": f"{row.spectral_radius:.3f}",
        "min_residual": sci_plain(row.min_residual),
        "median_residual": sci_plain(row.median_residual),
        "modes_le_1e_2": str(row.modes_le_1e_2),
        "modes_le_5e_2": str(row.modes_le_5e_2),
        "modes_le_1e_1": str(row.modes_le_1e_1),
        "test_E1_value": f"{row.test_e1:.17g}",
        "test_R2_value": f"{row.test_r2:.17g}",
        "spectral_radius_value": f"{row.spectral_radius:.17g}",
        "min_residual_value": f"{row.min_residual:.17g}",
        "median_residual_value": f"{row.median_residual:.17g}",
    }


def table_row_to_latex_cells(row: TableRow) -> list[str]:
    return [
        SYSTEM_LABELS[row.system],
        OPERATOR_LABELS[row.operator],
        sci_latex(row.test_e1),
        f"${row.test_r2:.4f}$",
        f"${row.spectral_radius:.3f}$",
        sci_latex(row.min_residual),
        sci_latex(row.median_residual),
        str(row.modes_le_1e_2),
        str(row.modes_le_5e_2),
        str(row.modes_le_1e_1),
    ]


def write_outputs(rows: Sequence[TableRow], args: CliArgs) -> None:
    if not rows:
        raise ValueError("No table rows to write.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = [table_row_to_csv(row) for row in rows]

    csv_path = args.output_dir / "table15_residual_spectra_values.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path = args.output_dir / "table15_residual_spectra_values.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| System | Operator | Test E1 | Test R2 | rho(M) | min residual | median residual | modes <= 1e-2 | modes <= 5e-2 | modes <= 1e-1 |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in csv_rows:
            handle.write(
                f"| {row['system']} | {row['operator']} | {row['test_E1']} | {row['test_R2']} | "
                f"{row['spectral_radius']} | {row['min_residual']} | {row['median_residual']} | "
                f"{row['modes_le_1e_2']} | {row['modes_le_5e_2']} | {row['modes_le_1e_1']} |\n"
            )

    tex_path = args.output_dir / "table15_residual_spectra_values.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by step_14_generate_table15_residual_spectra.py\n")
        handle.write("\\begin{table}[t]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write(
            "\\caption{Residual-verified KAHKM spectra on held-out snapshot pairs. Residuals use the finite-rank KAHM kernel and the feature residual in \\eqref{eq:feature_spectral_residual}. The counts report how many of the ten feature-space modes have residual below the indicated threshold.}\n"
        )
        handle.write("\\label{tab:residual_verified_spectra}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{llcccccccc}\n")
        handle.write("\\toprule\n")
        handle.write(
            "System & Operator & Test $E_1$ & Test $R^2$ & $\\rho(M)$ & min res. & median res. & modes $\\le 10^{-2}$ & modes $\\le 5\\times10^{-2}$ & modes $\\le 10^{-1}$ \\\\\n"
        )
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
        "skip_existing": args.skip_existing,
        "systems": list(args.systems),
        "operators": list(args.operators),
        "train_seeds": list(args.train_seeds),
        "test_seeds": list(args.test_seeds),
        "total_train_snapshots": args.n_steps_train * len(args.train_seeds),
        "total_test_snapshots": args.n_steps_test * len(args.test_seeds),
        "duffing": {"C": args.duffing_c, "omega": args.duffing_omega},
        "vanderpol": {"C": args.vanderpol_c, "omega": args.vanderpol_omega},
        "spectral_subsample": args.spectral_subsample,
        "ridge": args.ridge,
        "residual_thresholds": list(args.residual_thresholds),
        "table_rows": [asdict(row) for row in rows],
    }
    metadata_path = args.output_dir / "table15_residual_spectra_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Wrote:")
    for path in (csv_path, md_path, tex_path, metadata_path):
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    run_experiment(args)
    rows = build_table_rows(args)
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
