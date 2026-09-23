#!/usr/bin/env python3
"""Generate manuscript Table 15 from Experiment 19 R3 spectral outputs.

Place this script in the same directory as the experiment scripts and run:

    python step_14_generate_table15_residual_spectra.py

By default, the script runs/reuses:

    experiment_19_residual_verified_kahkm_spectra.py

with the manuscript R3 spectral-diagnostic configuration:

    systems: Duffing and Van der Pol
    operators: NLMS and ridge least squares
    train seeds: 0, 1, 2
    test seeds: 100, 101, 102
    per-seed train snapshots: 1200   -> 3600 total train snapshots
    per-seed test snapshots: 800     -> 2400 total test snapshots
    spectral subsample: 800 held-out snapshot pairs
    Duffing tuned abstraction: C=10, omega=2
    Van der Pol noise-aware tuned abstraction: C=25, omega=4
    ridge-LS operator ridge: 1e-8
    held-out representation ridge: 1e-8
    training feature-rank relative tolerance: 1e-10

Experiment 19 constructs the training-supported basis Q_Phi from the uncentered
SVD of Phi_train, diagonalizes M_red = Q_Phi^T M Q_Phi, represents each lifted
candidate by held-out kernel sections, and reports the manuscript R3 quantities

    eta_rep = ||Phi_spec v - w||^2 / ||w||^2
    rho     = ||Chi_spec v - lambda Phi_spec v||^2 / ||Chi_spec v||^2.

It then reads:

    kahkm_experiment_19_r3_residual_verified_spectra/
        experiment_19_system_summary.csv

and writes:

    kahkm_table15_r3_residual_spectra/table15_residual_spectra_values.csv
    kahkm_table15_r3_residual_spectra/table15_residual_spectra_values.md
    kahkm_table15_r3_residual_spectra/table15_residual_spectra_values.tex
    kahkm_table15_r3_residual_spectra/table15_residual_spectra_metadata.json

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
    representation_ridge: float
    rank_rtol: float
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
    ambient_num_regimes: int
    effective_feature_rank: int
    num_valid_r3_modes: int
    reduced_spectral_radius: float
    min_r3_rho: float
    median_r3_rho: float
    median_eta_rep: float
    max_eta_rep: float
    max_formula_abs_diff: float
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
        description="Generate manuscript Table 15 from R3 residual-verified KAHKM spectra outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=None,
        help=(
            "Experiment 19 R3 output directory. Default: "
            "<source-dir>/kahkm_experiment_19_r3_residual_verified_spectra."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table15_r3_residual_spectra"),
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
    parser.add_argument("--vanderpol-c", type=int, default=25)
    parser.add_argument("--vanderpol-omega", type=float, default=4.0)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--representation-ridge", type=float, default=1e-8)
    parser.add_argument("--rank-rtol", type=float, default=1e-10)
    parser.add_argument("--spectral-subsample", type=int, default=800)
    parser.add_argument("--residual-thresholds", nargs="+", default=["1e-3", "1e-2", "5e-2", "1e-1"])
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--kmeans-kind", default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)

    ns = parser.parse_args()
    if float(ns.representation_ridge) < 0.0:
        parser.error("--representation-ridge must be nonnegative.")
    if not (0.0 < float(ns.rank_rtol) < 1.0):
        parser.error("--rank-rtol must lie strictly between 0 and 1.")

    source_dir = Path(ns.source_dir).resolve()
    experiment_output_dir = (
        Path(ns.experiment_output_dir).resolve()
        if ns.experiment_output_dir
        else source_dir / "kahkm_experiment_19_r3_residual_verified_spectra"
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
        representation_ridge=float(ns.representation_ridge),
        rank_rtol=float(ns.rank_rtol),
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
        "--representation-ridge",
        str(args.representation_ridge),
        "--rank-rtol",
        str(args.rank_rtol),
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
    print("Running Experiment 19 R3 residual spectra:")
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
    return f"modes_with_r3_rho_le_{threshold:g}"


def find_summary_row(rows: Sequence[dict[str, str]], *, system: str, operator: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("system") == system and row.get("operator") == operator]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one summary row for {system}/{operator}, found {len(matches)}.")
    return matches[0]


def expected_config(args: CliArgs, system: str) -> tuple[int, float]:
    if system == "duffing":
        return args.duffing_c, args.duffing_omega
    if system == "vanderpol":
        return args.vanderpol_c, args.vanderpol_omega
    raise ValueError(f"Unknown system: {system}")


def validate_experiment_config(args: CliArgs) -> dict[str, object]:
    config_path = args.experiment_output_dir / "experiment_19_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Experiment 19 config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment 19 config must be a JSON object: {config_path}")

    expected_scalars: dict[str, float | int] = {
        "duffing_c": args.duffing_c,
        "duffing_omega": args.duffing_omega,
        "vanderpol_c": args.vanderpol_c,
        "vanderpol_omega": args.vanderpol_omega,
        "ridge": args.ridge,
        "representation_ridge": args.representation_ridge,
        "rank_rtol": args.rank_rtol,
        "spectral_subsample": args.spectral_subsample,
    }
    for key, expected in expected_scalars.items():
        if key not in payload:
            raise ValueError(f"Experiment 19 config is missing {key!r}: {config_path}")
        actual = float(payload[key])
        if not math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"Experiment 19 config mismatch for {key}: expected {expected}, found {payload[key]}."
            )

    configured_systems = tuple(str(item) for item in payload.get("systems", []))
    configured_operators = tuple(str(item) for item in payload.get("operators", []))
    if configured_systems != args.systems:
        raise ValueError(
            f"Experiment 19 systems mismatch: expected {args.systems}, found {configured_systems}."
        )
    if configured_operators != args.operators:
        raise ValueError(
            f"Experiment 19 operators mismatch: expected {args.operators}, found {configured_operators}."
        )
    return payload


def build_table_rows(args: CliArgs) -> tuple[list[TableRow], dict[str, object]]:
    experiment_config = validate_experiment_config(args)
    rows = read_csv_rows(args.experiment_output_dir / "experiment_19_system_summary.csv")
    table_rows: list[TableRow] = []
    for system in SYSTEM_ORDER:
        if system not in args.systems:
            continue
        expected_c, expected_omega = expected_config(args, system)
        for operator in OPERATOR_ORDER:
            if operator not in args.operators:
                continue
            row = find_summary_row(rows, system=system, operator=operator)
            context = f"{system}/{operator}"

            actual_c = finite_int(row, "n_clusters", context=context)
            actual_omega = finite_float(row, "omega", context=context)
            if actual_c != expected_c or not math.isclose(
                actual_omega, expected_omega, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(
                    f"Configuration mismatch for {context}: expected C={expected_c}, omega={expected_omega:g}; "
                    f"found C={actual_c}, omega={actual_omega:g}."
                )

            ambient_c = finite_int(row, "ambient_num_regimes", context=context)
            effective_rank = finite_int(row, "effective_feature_rank", context=context)
            valid_modes = finite_int(row, "num_valid_r3_modes", context=context)
            if ambient_c != actual_c:
                raise ValueError(
                    f"Ambient regime count mismatch for {context}: n_clusters={actual_c}, ambient={ambient_c}."
                )
            if not (1 <= effective_rank <= ambient_c):
                raise ValueError(
                    f"Invalid effective feature rank for {context}: r_Phi={effective_rank}, C={ambient_c}."
                )
            if not (0 <= valid_modes <= effective_rank):
                raise ValueError(
                    f"Invalid number of valid R3 modes for {context}: {valid_modes} for r_Phi={effective_rank}."
                )

            threshold_counts = {
                threshold: finite_int(row, threshold_key(threshold), context=context)
                for threshold in REQUIRED_THRESHOLDS
            }
            for threshold, count in threshold_counts.items():
                if not (0 <= count <= valid_modes):
                    raise ValueError(
                        f"Invalid R3 threshold count for {context} at {threshold:g}: {count}/{valid_modes}."
                    )

            table_rows.append(
                TableRow(
                    system=system,
                    operator=operator,
                    test_e1=finite_float(row, "test_closure_error", context=context),
                    test_r2=finite_float(row, "test_association_r2", context=context),
                    ambient_num_regimes=ambient_c,
                    effective_feature_rank=effective_rank,
                    num_valid_r3_modes=valid_modes,
                    reduced_spectral_radius=finite_float(row, "reduced_spectral_radius", context=context),
                    min_r3_rho=finite_float(row, "min_r3_residual_rho", context=context),
                    median_r3_rho=finite_float(row, "median_r3_residual_rho", context=context),
                    median_eta_rep=finite_float(row, "median_representation_defect_eta", context=context),
                    max_eta_rep=finite_float(row, "max_representation_defect_eta", context=context),
                    max_formula_abs_diff=finite_float(
                        row, "max_r3_feature_matrix_formula_abs_diff", context=context
                    ),
                    modes_le_1e_2=threshold_counts[1e-2],
                    modes_le_5e_2=threshold_counts[5e-2],
                    modes_le_1e_1=threshold_counts[1e-1],
                )
            )
    return table_rows, experiment_config


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
        "effective_rank_over_C": f"{row.effective_feature_rank}/{row.ambient_num_regimes}",
        "reduced_spectral_radius": f"{row.reduced_spectral_radius:.3f}",
        "min_r3_rho": sci_plain(row.min_r3_rho),
        "median_r3_rho": sci_plain(row.median_r3_rho),
        "max_eta_rep": sci_plain(row.max_eta_rep),
        "modes_rho_le_1e_2": str(row.modes_le_1e_2),
        "modes_rho_le_5e_2": str(row.modes_le_5e_2),
        "modes_rho_le_1e_1": str(row.modes_le_1e_1),
        "test_E1_value": f"{row.test_e1:.17g}",
        "test_R2_value": f"{row.test_r2:.17g}",
        "ambient_num_regimes": str(row.ambient_num_regimes),
        "effective_feature_rank": str(row.effective_feature_rank),
        "num_valid_r3_modes": str(row.num_valid_r3_modes),
        "reduced_spectral_radius_value": f"{row.reduced_spectral_radius:.17g}",
        "min_r3_rho_value": f"{row.min_r3_rho:.17g}",
        "median_r3_rho_value": f"{row.median_r3_rho:.17g}",
        "median_eta_rep_value": f"{row.median_eta_rep:.17g}",
        "max_eta_rep_value": f"{row.max_eta_rep:.17g}",
        "max_formula_abs_diff_value": f"{row.max_formula_abs_diff:.17g}",
    }


def table_row_to_latex_cells(row: TableRow) -> list[str]:
    return [
        SYSTEM_LABELS[row.system],
        OPERATOR_LABELS[row.operator],
        sci_latex(row.test_e1),
        f"${row.test_r2:.4f}$",
        f"${row.effective_feature_rank}/{row.ambient_num_regimes}$",
        f"${row.reduced_spectral_radius:.3f}$",
        sci_latex(row.min_r3_rho),
        sci_latex(row.median_r3_rho),
        sci_latex(row.max_eta_rep),
        str(row.modes_le_1e_2),
        str(row.modes_le_5e_2),
        str(row.modes_le_1e_1),
    ]


def write_outputs(
    rows: Sequence[TableRow],
    args: CliArgs,
    experiment_config: dict[str, object],
) -> None:
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
        handle.write(
            "| System | Operator | Test E1 | Test R2 | r_Phi/C | spr(M_red) | min R3 rho | "
            "median R3 rho | max eta_rep | modes rho <= 1e-2 | modes rho <= 5e-2 | modes rho <= 1e-1 |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in csv_rows:
            handle.write(
                f"| {row['system']} | {row['operator']} | {row['test_E1']} | {row['test_R2']} | "
                f"{row['effective_rank_over_C']} | {row['reduced_spectral_radius']} | "
                f"{row['min_r3_rho']} | {row['median_r3_rho']} | {row['max_eta_rep']} | "
                f"{row['modes_rho_le_1e_2']} | {row['modes_rho_le_5e_2']} | "
                f"{row['modes_rho_le_1e_1']} |\n"
            )

    tex_path = args.output_dir / "table15_residual_spectra_values.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by step_14_generate_table15_residual_spectra.py\n")
        handle.write("\\begin{table}[t]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write(
            "\\caption{Rank-aware R3 spectral verification on held-out snapshot pairs. "
            "$r_\\Phi/C$ reports the training-supported coefficient rank relative to the ambient "
            "number of regimes. Candidate modes are eigenpairs of the reduced operator "
            "$M_{\\mathrm{red}}=Q_\\Phi^\\top M Q_\\Phi$. The R3 residual is the successor-normalized "
            "squared residual $\\rho$, and $\\eta_{\\mathrm{rep}}$ is the held-out representation "
            "defect; the table reports its maximum across verified candidates. Counts report the "
            "number of the $r_\\Phi$ valid reduced-space candidates below each R3 threshold.}\n"
        )
        handle.write("\\label{tab:residual_verified_spectra}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{llcccccccccc}\n")
        handle.write("\\toprule\n")
        handle.write(
            "System & Operator & Test $E_1$ & Test $R^2$ & $r_\\Phi/C$ & "
            "$\\mathrm{spr}(M_{\\mathrm{red}})$ & min $\\rho$ & median $\\rho$ & "
            "max $\\eta_{\\mathrm{rep}}$ & $N_{\\rho\\le10^{-2}}$ & "
            "$N_{\\rho\\le5\\times10^{-2}}$ & $N_{\\rho\\le10^{-1}}$ \\\\\n"
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
        "ridge_operator": args.ridge,
        "representation_ridge": args.representation_ridge,
        "rank_rtol": args.rank_rtol,
        "residual_thresholds": list(args.residual_thresholds),
        "displayed_thresholds": list(REQUIRED_THRESHOLDS),
        "r3_definition": {
            "feature_basis": "uncentered training Phi SVD; retain sigma_j > rank_rtol * sigma_max",
            "reduced_operator": "M_red = Q_Phi.T @ M @ Q_Phi",
            "representation_defect_eta": "||Phi_spec v - w||^2 / ||w||^2",
            "spectral_residual_rho": "||Chi_spec v - lambda Phi_spec v||^2 / ||Chi_spec v||^2",
        },
        "experiment_config": experiment_config,
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
    rows, experiment_config = build_table_rows(args)
    write_outputs(rows, args, experiment_config)


if __name__ == "__main__":
    main()