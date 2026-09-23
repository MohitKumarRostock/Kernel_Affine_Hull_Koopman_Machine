#!/usr/bin/env python3
"""Generate manuscript Table 9 from centered Experiment 13 outputs.

Table 9: centered multi-horizon Van der Pol unseen-trajectory generalization.

The generator is aligned with the retained-variation/centered-selection update.
It compares the three manuscript Experiment 13 configurations:

    robust_tuned: C=25, omega=4
    clean_tuned:  C=25, omega=6
    old_default:  C=20, omega=4

The manuscript-facing score is the mean association R^2 across horizons
{1, 10, 50, 100, 200}. The paired columns report matched-replicate
differences in the corresponding centered error

    Delta E_clean-robust = E_clean - E_robust
    Delta E_ref-robust   = E_old_default - E_robust

so a positive paired difference means that robust_tuned has the lower centered
error for that comparison. Paired differences are reported as mean +/- SE;
configuration-level mean multihorizon R^2 is reported as mean +/- SD across the
three replicate training-seed blocks.

By default, the script runs/reuses:

    run_exp13_vanderpol_tuned_trajectory_generalization.py

and reads:

    kahkm_exp13_retvar_updated_configs/
        experiment_13_centered_multihorizon_summary.csv
        experiment_13_centered_paired_comparisons.csv

Legacy Experiment 13 E_h summaries remain untouched for provenance.

Use --no-run to format existing centered Experiment 13 outputs only.
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


EXPERIMENT_RUNNER: Final[str] = "run_exp13_vanderpol_tuned_trajectory_generalization.py"
EXPERIMENT_OUTPUT_DIR: Final[str] = "kahkm_exp13_retvar_updated_configs"
TABLE_OUTPUT_DIR: Final[str] = "kahkm_table9_centered_tuned_trajectory"
CENTERED_SUMMARY_FILE: Final[str] = "experiment_13_centered_multihorizon_summary.csv"
PAIRED_COMPARISONS_FILE: Final[str] = "experiment_13_centered_paired_comparisons.csv"

CONFIG_ROBUST: Final[str] = "robust_tuned"
CONFIG_CLEAN: Final[str] = "clean_tuned"
CONFIG_REFERENCE: Final[str] = "old_default"

EXPECTED_CONFIGS: Final[dict[str, tuple[int, float]]] = {
    CONFIG_ROBUST: (25, 4.0),
    CONFIG_CLEAN: (25, 6.0),
    CONFIG_REFERENCE: (20, 4.0),
}
EXPECTED_SELECTION_HORIZONS: Final[tuple[int, ...]] = (1, 10, 50, 100, 200)
EXPECTED_TRAIN_COUNTS: Final[tuple[int, ...]] = (1, 2, 3, 5, 8)
EXPECTED_REPLICATES: Final[int] = 3


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
class CenteredSummaryRow:
    config_name: str
    n_clusters: int
    omega: float
    n_train_trajectories: int
    selection_horizons: tuple[int, ...]
    n_replicates: int
    mean_centered_error: float
    mean_multihorizon_r2: float
    centered_error_sd: float
    centered_error_se: float


@dataclass(frozen=True)
class PairedComparisonRow:
    n_train_trajectories: int
    config_a: str
    config_b: str
    difference_definition: str
    n_paired_replicates: int
    mean_paired_difference: float
    paired_difference_sd: float
    paired_difference_se: float
    replicate_0_difference: float
    replicate_1_difference: float
    replicate_2_difference: float


@dataclass(frozen=True)
class TableRow:
    training_trajectories: int

    robust_r2_mean: float
    robust_r2_sd: float
    robust_centered_error_mean: float

    clean_r2_mean: float
    clean_r2_sd: float
    clean_centered_error_mean: float

    reference_r2_mean: float
    reference_r2_sd: float
    reference_centered_error_mean: float

    clean_minus_robust_mean: float
    clean_minus_robust_sd: float
    clean_minus_robust_se: float

    reference_minus_robust_mean: float
    reference_minus_robust_sd: float
    reference_minus_robust_se: float

    @property
    def robust_r2_cell(self) -> str:
        return format_mean_sd(self.robust_r2_mean, self.robust_r2_sd)

    @property
    def clean_r2_cell(self) -> str:
        return format_mean_sd(self.clean_r2_mean, self.clean_r2_sd)

    @property
    def reference_r2_cell(self) -> str:
        return format_mean_sd(self.reference_r2_mean, self.reference_r2_sd)

    @property
    def clean_minus_robust_cell(self) -> str:
        return format_signed_mean_se(self.clean_minus_robust_mean, self.clean_minus_robust_se)

    @property
    def reference_minus_robust_cell(self) -> str:
        return format_signed_mean_se(self.reference_minus_robust_mean, self.reference_minus_robust_se)


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Generate centered manuscript Table 9 from Experiment 13 "
            "Van der Pol trajectory-generalization outputs."
        )
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
            "Directory containing centered Experiment 13 outputs. Default: "
            f"<source-dir>/{EXPERIMENT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for generated centered Table 9 files. Default: <source-dir>/{TABLE_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--runner",
        default=EXPERIMENT_RUNNER,
        help=f"Experiment runner script. Default: {EXPERIMENT_RUNNER}.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to run the Experiment 13 runner.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run Experiment 13; only format existing centered outputs.",
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

    print("Running centered Experiment 13:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=args.source_dir, check=True)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Required CSV is empty: {path}")
    return rows


def parse_float(row: dict[str, str], key: str, *, context: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing numeric column {key!r} for {context}.")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric column {key!r}={value!r} for {context}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric column {key!r}={value!r} for {context}.")
    return number


def parse_int(row: dict[str, str], key: str, *, context: str) -> int:
    return int(round(parse_float(row, key, context=context)))


def parse_horizons(value: str, *, context: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(part) for part in value.split())
    except ValueError as exc:
        raise ValueError(f"Could not parse selection_horizons={value!r} for {context}.") from exc
    if not horizons:
        raise ValueError(f"Empty selection_horizons for {context}.")
    return horizons


def load_centered_summary(path: Path) -> list[CenteredSummaryRow]:
    rows: list[CenteredSummaryRow] = []
    for raw in read_csv_dicts(path):
        config_name = raw.get("config_name", "")
        count_text = raw.get("n_train_trajectories", "?")
        context = f"centered summary {config_name}/N={count_text}"
        rows.append(
            CenteredSummaryRow(
                config_name=config_name,
                n_clusters=parse_int(raw, "n_clusters", context=context),
                omega=parse_float(raw, "omega", context=context),
                n_train_trajectories=parse_int(raw, "n_train_trajectories", context=context),
                selection_horizons=parse_horizons(
                    raw.get("selection_horizons", ""), context=context
                ),
                n_replicates=parse_int(raw, "n_replicates", context=context),
                mean_centered_error=parse_float(raw, "mean_centered_error", context=context),
                mean_multihorizon_r2=parse_float(raw, "mean_multihorizon_r2", context=context),
                centered_error_sd=parse_float(raw, "centered_error_sd", context=context),
                centered_error_se=parse_float(raw, "centered_error_se", context=context),
            )
        )
    return rows


def load_paired_comparisons(path: Path) -> list[PairedComparisonRow]:
    rows: list[PairedComparisonRow] = []
    for raw in read_csv_dicts(path):
        config_a = raw.get("config_a", "")
        config_b = raw.get("config_b", "")
        count_text = raw.get("n_train_trajectories", "?")
        context = f"paired comparison {config_a}-{config_b}/N={count_text}"
        rows.append(
            PairedComparisonRow(
                n_train_trajectories=parse_int(raw, "n_train_trajectories", context=context),
                config_a=config_a,
                config_b=config_b,
                difference_definition=raw.get("difference_definition", ""),
                n_paired_replicates=parse_int(raw, "n_paired_replicates", context=context),
                mean_paired_difference=parse_float(raw, "mean_paired_difference", context=context),
                paired_difference_sd=parse_float(raw, "paired_difference_sd", context=context),
                paired_difference_se=parse_float(raw, "paired_difference_se", context=context),
                replicate_0_difference=parse_float(raw, "replicate_0_difference", context=context),
                replicate_1_difference=parse_float(raw, "replicate_1_difference", context=context),
                replicate_2_difference=parse_float(raw, "replicate_2_difference", context=context),
            )
        )
    return rows


def validate_centered_summary(rows: Sequence[CenteredSummaryRow]) -> None:
    indexed: dict[tuple[str, int], CenteredSummaryRow] = {}
    for row in rows:
        if row.config_name not in EXPECTED_CONFIGS:
            raise ValueError(f"Unexpected Experiment 13 config in centered summary: {row.config_name!r}.")
        key = (row.config_name, row.n_train_trajectories)
        if key in indexed:
            raise ValueError(f"Duplicate centered summary row for {key}.")
        indexed[key] = row

        expected_c, expected_omega = EXPECTED_CONFIGS[row.config_name]
        if row.n_clusters != expected_c or not math.isclose(
            row.omega, expected_omega, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Centered summary config mismatch for {row.config_name}: "
                f"found C={row.n_clusters}, omega={row.omega}; "
                f"expected C={expected_c}, omega={expected_omega}."
            )
        if row.selection_horizons != EXPECTED_SELECTION_HORIZONS:
            raise ValueError(
                f"Selection-horizon mismatch for {key}: found {row.selection_horizons}, "
                f"expected {EXPECTED_SELECTION_HORIZONS}."
            )
        if row.n_replicates != EXPECTED_REPLICATES:
            raise ValueError(
                f"Replicate-count mismatch for {key}: found {row.n_replicates}, "
                f"expected {EXPECTED_REPLICATES}."
            )

        # Since mean_multihorizon_r2 is the mean of R^2 over the same horizons,
        # it must equal 1 - mean_centered_error up to roundoff.
        expected_r2 = 1.0 - row.mean_centered_error
        if not math.isclose(
            row.mean_multihorizon_r2, expected_r2, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Centered summary identity failed for {key}: "
                f"mean_multihorizon_r2={row.mean_multihorizon_r2}, "
                f"1-mean_centered_error={expected_r2}."
            )

    expected_keys = {
        (config, count)
        for config in EXPECTED_CONFIGS
        for count in EXPECTED_TRAIN_COUNTS
    }
    actual_keys = set(indexed)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Centered summary protocol mismatch. Missing={missing}; extra={extra}."
        )


def validate_paired_comparisons(rows: Sequence[PairedComparisonRow]) -> None:
    expected_pairs = {
        (CONFIG_CLEAN, CONFIG_REFERENCE),
        (CONFIG_CLEAN, CONFIG_ROBUST),
        (CONFIG_REFERENCE, CONFIG_ROBUST),
    }
    indexed: dict[tuple[int, str, str], PairedComparisonRow] = {}

    for row in rows:
        key = (row.n_train_trajectories, row.config_a, row.config_b)
        if key in indexed:
            raise ValueError(f"Duplicate centered paired-comparison row for {key}.")
        indexed[key] = row

        if row.n_train_trajectories not in EXPECTED_TRAIN_COUNTS:
            raise ValueError(f"Unexpected training count in paired comparisons: {row.n_train_trajectories}.")
        if (row.config_a, row.config_b) not in expected_pairs:
            raise ValueError(
                f"Unexpected config pair in centered paired comparisons: "
                f"{row.config_a!r}, {row.config_b!r}."
            )
        if row.n_paired_replicates != EXPECTED_REPLICATES:
            raise ValueError(
                f"Paired replicate-count mismatch for {key}: "
                f"found {row.n_paired_replicates}, expected {EXPECTED_REPLICATES}."
            )
        expected_definition = (
            "mean_centered_error(config_a) - mean_centered_error(config_b)"
        )
        if row.difference_definition != expected_definition:
            raise ValueError(
                f"Unexpected difference definition for {key}: "
                f"{row.difference_definition!r}."
            )

    expected_keys = {
        (count, config_a, config_b)
        for count in EXPECTED_TRAIN_COUNTS
        for config_a, config_b in expected_pairs
    }
    actual_keys = set(indexed)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Paired-comparison protocol mismatch. Missing={missing}; extra={extra}."
        )


def format_mean_sd(mean_value: float, sd_value: float) -> str:
    return f"${mean_value:.4f}\\pm{sd_value:.4f}$"


def format_signed_mean_se(mean_value: float, se_value: float) -> str:
    return f"${mean_value:+.4f}\\pm{se_value:.4f}$"


def build_table_rows(
    summary_rows: Sequence[CenteredSummaryRow],
    paired_rows: Sequence[PairedComparisonRow],
) -> list[TableRow]:
    summary_index = {
        (row.config_name, row.n_train_trajectories): row for row in summary_rows
    }
    paired_index = {
        (row.n_train_trajectories, row.config_a, row.config_b): row
        for row in paired_rows
    }

    table_rows: list[TableRow] = []
    for train_count in EXPECTED_TRAIN_COUNTS:
        robust = summary_index[(CONFIG_ROBUST, train_count)]
        clean = summary_index[(CONFIG_CLEAN, train_count)]
        reference = summary_index[(CONFIG_REFERENCE, train_count)]

        clean_vs_robust = paired_index[
            (train_count, CONFIG_CLEAN, CONFIG_ROBUST)
        ]
        reference_vs_robust = paired_index[
            (train_count, CONFIG_REFERENCE, CONFIG_ROBUST)
        ]

        table_rows.append(
            TableRow(
                training_trajectories=train_count,
                robust_r2_mean=robust.mean_multihorizon_r2,
                robust_r2_sd=robust.centered_error_sd,
                robust_centered_error_mean=robust.mean_centered_error,
                clean_r2_mean=clean.mean_multihorizon_r2,
                clean_r2_sd=clean.centered_error_sd,
                clean_centered_error_mean=clean.mean_centered_error,
                reference_r2_mean=reference.mean_multihorizon_r2,
                reference_r2_sd=reference.centered_error_sd,
                reference_centered_error_mean=reference.mean_centered_error,
                clean_minus_robust_mean=clean_vs_robust.mean_paired_difference,
                clean_minus_robust_sd=clean_vs_robust.paired_difference_sd,
                clean_minus_robust_se=clean_vs_robust.paired_difference_se,
                reference_minus_robust_mean=reference_vs_robust.mean_paired_difference,
                reference_minus_robust_sd=reference_vs_robust.paired_difference_sd,
                reference_minus_robust_se=reference_vs_robust.paired_difference_se,
            )
        )
    return table_rows


def table_row_to_csv(row: TableRow) -> dict[str, str]:
    return {
        "training_trajectories": str(row.training_trajectories),
        "robust_mean_multihorizon_r2": repr(row.robust_r2_mean),
        "robust_r2_sd": repr(row.robust_r2_sd),
        "robust_mean_centered_error": repr(row.robust_centered_error_mean),
        "clean_mean_multihorizon_r2": repr(row.clean_r2_mean),
        "clean_r2_sd": repr(row.clean_r2_sd),
        "clean_mean_centered_error": repr(row.clean_centered_error_mean),
        "reference_mean_multihorizon_r2": repr(row.reference_r2_mean),
        "reference_r2_sd": repr(row.reference_r2_sd),
        "reference_mean_centered_error": repr(row.reference_centered_error_mean),
        "clean_minus_robust_centered_error_mean": repr(row.clean_minus_robust_mean),
        "clean_minus_robust_centered_error_sd": repr(row.clean_minus_robust_sd),
        "clean_minus_robust_centered_error_se": repr(row.clean_minus_robust_se),
        "reference_minus_robust_centered_error_mean": repr(row.reference_minus_robust_mean),
        "reference_minus_robust_centered_error_sd": repr(row.reference_minus_robust_sd),
        "reference_minus_robust_centered_error_se": repr(row.reference_minus_robust_se),
        "robust_r2_table_cell": row.robust_r2_cell,
        "clean_r2_table_cell": row.clean_r2_cell,
        "reference_r2_table_cell": row.reference_r2_cell,
        "clean_minus_robust_table_cell": row.clean_minus_robust_cell,
        "reference_minus_robust_table_cell": row.reference_minus_robust_cell,
    }


def paired_row_to_csv(row: PairedComparisonRow) -> dict[str, str]:
    return {
        "n_train_trajectories": str(row.n_train_trajectories),
        "config_a": row.config_a,
        "config_b": row.config_b,
        "difference_definition": row.difference_definition,
        "n_paired_replicates": str(row.n_paired_replicates),
        "mean_paired_difference": repr(row.mean_paired_difference),
        "paired_difference_sd": repr(row.paired_difference_sd),
        "paired_difference_se": repr(row.paired_difference_se),
        "replicate_0_difference": repr(row.replicate_0_difference),
        "replicate_1_difference": repr(row.replicate_1_difference),
        "replicate_2_difference": repr(row.replicate_2_difference),
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
        "# Table 9: Centered multi-horizon Van der Pol unseen-trajectory generalization",
        "",
        (
            "| Training trajectories | Robust mean R2 | Clean mean R2 | Reference mean R2 | "
            "Delta E clean-robust | Delta E reference-robust |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.training_trajectories} | "
            f"{row.robust_r2_cell} | "
            f"{row.clean_r2_cell} | "
            f"{row.reference_r2_cell} | "
            f"{row.clean_minus_robust_cell} | "
            f"{row.reference_minus_robust_cell} |"
        )
    lines.extend(
        [
            "",
            (
                "Mean R2 is the mean association R2 over horizons "
                "{1, 10, 50, 100, 200}; configuration entries are mean +/- SD "
                "over three replicate training-seed blocks."
            ),
            (
                "Paired Delta E columns are mean +/- SE over matched replicates, "
                "with Delta E = centered error(comparator) - centered error(robust). "
                "Positive values therefore indicate lower centered error for robust_tuned."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_latex(rows: Sequence[TableRow]) -> str:
    body_lines = [
        (
            f"{row.training_trajectories} & "
            f"{row.robust_r2_cell} & "
            f"{row.clean_r2_cell} & "
            f"{row.reference_r2_cell} & "
            f"{row.clean_minus_robust_cell} & "
            f"{row.reference_minus_robust_cell} \\\\"
        )
        for row in rows
    ]
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Centered multi-horizon Van der Pol unseen-trajectory generalization. "
        "The noise-aware robust setting $(C,\\omega)=(25,4)$ is compared with the "
        "clean-tuned setting $(25,6)$ and the older reference setting $(20,4)$. "
        "For each configuration, $\\overline{R^2}$ is the mean association $R^2$ over "
        "$h\\in\\{1,10,50,100,200\\}$ and is reported as mean $\\pm$ standard deviation "
        "over three replicate training-seed blocks. Paired columns report "
        "$\\Delta E=E_{\\mathrm{comp}}-E_{\\mathrm{robust}}$ as mean $\\pm$ standard error "
        "over the same matched replicates; positive $\\Delta E$ therefore means lower "
        "centered multi-horizon error for the robust setting. With three paired replicates, "
        "these differences are reported descriptively rather than as formal significance tests.}\n"
        "\\label{tab:vanderpol_tuned_trajectory}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{cccccc}\n"
        "\\toprule\n"
        "Training trajectories & Robust $\\overline{R^2}$ & Clean $\\overline{R^2}$ & "
        "Reference $\\overline{R^2}$ & $\\Delta E_{\\mathrm{clean-robust}}$ & "
        "$\\Delta E_{\\mathrm{ref-robust}}$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )


def write_metadata(
    path: Path,
    args: CliArgs,
    summary_path: Path,
    paired_path: Path,
    table_rows: Sequence[TableRow],
    paired_rows: Sequence[PairedComparisonRow],
) -> None:
    metadata = {
        "table": "Table 9",
        "description": "Centered multi-horizon Van der Pol unseen-trajectory generalization",
        "experiment_runner": args.runner,
        "experiment_output_dir": str(args.experiment_output_dir),
        "source_centered_summary_csv": str(summary_path),
        "source_centered_paired_comparisons_csv": str(paired_path),
        "configs": {
            "robust_tuned": {"C": 25, "omega": 4.0},
            "clean_tuned": {"C": 25, "omega": 6.0},
            "old_default_reference": {"C": 20, "omega": 4.0},
        },
        "selection_horizons": list(EXPECTED_SELECTION_HORIZONS),
        "n_replicates": EXPECTED_REPLICATES,
        "uncertainty_convention": {
            "configuration_mean_multihorizon_r2": "mean +/- SD across replicate training-seed blocks",
            "paired_centered_error_differences": "mean +/- SE across matched replicates",
        },
        "paired_difference_definition": (
            "centered error(comparator) - centered error(robust_tuned); "
            "positive values indicate lower centered error for robust_tuned"
        ),
        "statistical_scope": (
            "Three paired replicates; differences are descriptive and no formal "
            "significance test is encoded by this generator."
        ),
        "training_trajectories": list(EXPECTED_TRAIN_COUNTS),
        "outputs": {
            "csv": str(args.output_dir / "table9_tuned_trajectory_values.csv"),
            "paired_comparisons_csv": str(
                args.output_dir / "table9_tuned_trajectory_paired_comparisons.csv"
            ),
            "markdown": str(args.output_dir / "table9_tuned_trajectory_values.md"),
            "latex": str(args.output_dir / "table9_tuned_trajectory_values.tex"),
        },
        "cli_args": {
            "source_dir": str(args.source_dir),
            "no_run": args.no_run,
            "resume": args.resume,
            "n_jobs": args.n_jobs,
        },
        "table_rows": [asdict(row) for row in table_rows],
        "all_paired_comparisons": [asdict(row) for row in paired_rows],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.no_run:
        run_experiment(args)

    centered_summary_path = args.experiment_output_dir / CENTERED_SUMMARY_FILE
    paired_comparisons_path = args.experiment_output_dir / PAIRED_COMPARISONS_FILE

    summary_rows = load_centered_summary(centered_summary_path)
    paired_rows = load_paired_comparisons(paired_comparisons_path)
    validate_centered_summary(summary_rows)
    validate_paired_comparisons(paired_rows)
    table_rows = build_table_rows(summary_rows, paired_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_csv = args.output_dir / "table9_tuned_trajectory_values.csv"
    paired_csv = args.output_dir / "table9_tuned_trajectory_paired_comparisons.csv"
    markdown_path = args.output_dir / "table9_tuned_trajectory_values.md"
    latex_path = args.output_dir / "table9_tuned_trajectory_values.tex"
    metadata_path = args.output_dir / "table9_tuned_trajectory_metadata.json"

    write_csv(table_csv, [table_row_to_csv(row) for row in table_rows])
    write_csv(paired_csv, [paired_row_to_csv(row) for row in paired_rows])
    write_markdown(markdown_path, table_rows)
    latex_path.write_text(make_latex(table_rows), encoding="utf-8")
    write_metadata(
        metadata_path,
        args,
        centered_summary_path,
        paired_comparisons_path,
        table_rows,
        paired_rows,
    )

    print("Generated centered Table 9 files:")
    for path in (table_csv, paired_csv, markdown_path, latex_path, metadata_path):
        print(path)


if __name__ == "__main__":
    main()
