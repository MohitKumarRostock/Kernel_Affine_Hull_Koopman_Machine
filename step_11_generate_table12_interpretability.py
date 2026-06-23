#!/usr/bin/env python3
"""Generate manuscript Table 12 from tuned Classic Control Experiment 17 outputs.

Place this script in the same directory as the experiment scripts and run:

    python step_11_generate_table12_interpretability.py

By default, the script runs/reuses:

    run_exp17_tuned_interpretability.py

and aggregates the tuned seed-0 interpretability folders:

    kahkm_exp17_cartpole_tuned_seed0/
    kahkm_exp17_mountaincar_tuned_seed0/

Expected input files inside each folder:

    experiment_17_model_summary.csv
    experiment_17_regime_statistics.csv

Outputs are written to:

    kahkm_table12_interpretability/table12_interpretability_values.csv
    kahkm_table12_interpretability/table12_interpretability_values.md
    kahkm_table12_interpretability/table12_interpretability_values.tex
    kahkm_table12_interpretability/table12_interpretability_metadata.json

Use --no-run to only format already generated Experiment 17 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence


TASK_CONFIGS: Final[dict[str, tuple[int, float]]] = {
    "CartPole-v1": (20, 1.0),
    "MountainCar-v0": (100, 1.0),
}


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    experiment_output_dir: Path
    output_dir: Path
    python_executable: str
    runner_script: str
    no_run: bool
    skip_existing: bool
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class SeedDiagnostic:
    task: str
    seed_label: str
    n_clusters: int
    omega: float
    active_count: int
    total_count: int
    train_error: float
    train_r2: float
    action_purity: float
    learned_persistence: float
    max_terminal_risk: float


@dataclass(frozen=True)
class TableRow:
    task: str
    n_clusters: int
    omega: float
    active: str
    train_error: float
    train_r2: float
    action_purity: float
    learned_persistence: float
    max_terminal_risk: float


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Table 12 from tuned Classic Control Experiment 17 outputs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing run_exp17_tuned_interpretability.py. Default: current directory.",
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=None,
        help="Directory containing kahkm_exp17_*_tuned_seed* folders. Default: --source-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table12_interpretability"),
        help="Output directory for generated table files.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable used to run the Experiment 17 wrapper.",
    )
    parser.add_argument(
        "--runner-script",
        default="run_exp17_tuned_interpretability.py",
        help="Experiment 17 tuned runner script filename.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run the experiment wrapper; only aggregate existing outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Pass --skip-existing to the Experiment 17 wrapper. Default: enabled.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0"],
        help="Seeds expected in the tuned interpretability result directories. Default: 0.",
    )
    namespace = parser.parse_args()
    source_dir = Path(namespace.source_dir).resolve()
    experiment_output_dir = Path(namespace.experiment_output_dir).resolve() if namespace.experiment_output_dir else source_dir
    output_dir = Path(namespace.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir
    return CliArgs(
        source_dir=source_dir,
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        python_executable=str(namespace.python_executable),
        runner_script=str(namespace.runner_script),
        no_run=bool(namespace.no_run),
        skip_existing=bool(namespace.skip_existing),
        seeds=_parse_int_tuple(namespace.seeds),
    )


def run_experiment(args: CliArgs) -> None:
    if args.no_run:
        return
    runner_path = args.source_dir / args.runner_script
    if not runner_path.exists():
        raise FileNotFoundError(
            f"Could not find {runner_path}. Place this script in the experiment folder, "
            "or pass --source-dir/--runner-script."
        )
    command = [
        args.python_executable,
        str(runner_path),
        "--output-root",
        str(args.experiment_output_dir),
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    print("Running Experiment 17 tuned interpretability wrapper:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return result


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _task_seed_dirs(root: Path, task: str, seeds: tuple[int, ...]) -> list[Path]:
    prefix = "cartpole" if task == "CartPole-v1" else "mountaincar"
    dirs: list[Path] = []
    for seed in seeds:
        expected = root / f"kahkm_exp17_{prefix}_tuned_seed{seed}"
        if expected.exists():
            dirs.append(expected)
    if dirs:
        return dirs
    pattern = f"kahkm_exp17_{prefix}_tuned_seed*"
    return sorted(path for path in root.glob(pattern) if path.is_dir())


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("Value and weight sequences must have the same length.")
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Cannot compute weighted mean with non-positive total weight.")
    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight)


def _action_purity(row: dict[str, str]) -> float | None:
    rates: list[float] = []
    for key, value in row.items():
        if not key.startswith("action_") or not key.endswith("_rate"):
            continue
        parsed = _maybe_float(value)
        if parsed is not None:
            rates.append(parsed)
    if not rates:
        return None
    return float(max(rates))


def _terminal_risk_key(rows: Sequence[dict[str, str]]) -> str:
    for row in rows:
        for key in row.keys():
            if key.startswith("terminal_risk_within_"):
                return key
    raise ValueError("Could not find a terminal_risk_within_* column in regime statistics.")


def _diagnostic_from_folder(task: str, run_dir: Path) -> SeedDiagnostic:
    model_path = run_dir / "experiment_17_model_summary.csv"
    stats_path = run_dir / "experiment_17_regime_statistics.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}")

    model_rows = [row for row in _read_csv(model_path) if row.get("task") == task]
    if len(model_rows) != 1:
        raise ValueError(f"Expected exactly one model-summary row for {task} in {model_path}; found {len(model_rows)}")
    model = model_rows[0]

    stats_rows = [row for row in _read_csv(stats_path) if row.get("task") == task]
    if not stats_rows:
        raise ValueError(f"No regime-statistics rows found for {task} in {stats_path}")

    n_clusters = int(float(model.get("n_clusters", "nan")))
    omega = _safe_float(model.get("omega", "nan"))
    risk_key = _terminal_risk_key(stats_rows)

    active_rows: list[dict[str, str]] = []
    for row in stats_rows:
        count = int(float(row.get("count", "0")))
        if count > 0:
            active_rows.append(row)
    if not active_rows:
        raise ValueError(f"No active regimes found for {task} in {stats_path}")

    weights = [float(int(float(row["count"]))) for row in active_rows]

    purity_values: list[float] = []
    purity_weights: list[float] = []
    learned_values: list[float] = []
    learned_weights: list[float] = []
    risk_values: list[float] = []

    for row, weight in zip(active_rows, weights):
        purity = _action_purity(row)
        if purity is not None:
            purity_values.append(purity)
            purity_weights.append(weight)
        learned = _maybe_float(row.get("learned_persistence"))
        if learned is not None:
            learned_values.append(learned)
            learned_weights.append(weight)
        risk = _maybe_float(row.get(risk_key))
        if risk is not None:
            risk_values.append(risk)

    if not purity_values:
        raise ValueError(f"No finite action-rate values found for {task} in {stats_path}")
    if not learned_values:
        raise ValueError(f"No finite learned-persistence values found for {task} in {stats_path}")
    if not risk_values:
        raise ValueError(f"No finite terminal-risk values found for {task} in {stats_path}")

    seed_label = run_dir.name.rsplit("seed", 1)[-1]
    return SeedDiagnostic(
        task=task,
        seed_label=seed_label,
        n_clusters=n_clusters,
        omega=omega,
        active_count=len(active_rows),
        total_count=n_clusters,
        train_error=_safe_float(model.get("train_closure_error", "nan")),
        train_r2=_safe_float(model.get("train_r2", "nan")),
        action_purity=_weighted_mean(purity_values, purity_weights),
        learned_persistence=_weighted_mean(learned_values, learned_weights),
        max_terminal_risk=max(risk_values),
    )


def load_seed_diagnostics(args: CliArgs) -> list[SeedDiagnostic]:
    diagnostics: list[SeedDiagnostic] = []
    for task in TASK_CONFIGS:
        dirs = _task_seed_dirs(args.experiment_output_dir, task, args.seeds)
        if not dirs:
            raise FileNotFoundError(
                f"No tuned Experiment 17 output directories found for {task} in {args.experiment_output_dir}."
            )
        for run_dir in dirs:
            diagnostics.append(_diagnostic_from_folder(task, run_dir))
    return diagnostics


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty list.")
    return float(statistics.fmean(values))


def summarize_diagnostics(diagnostics: Sequence[SeedDiagnostic]) -> list[TableRow]:
    rows: list[TableRow] = []
    for task, (expected_c, expected_omega) in TASK_CONFIGS.items():
        task_diags = [item for item in diagnostics if item.task == task]
        if not task_diags:
            raise ValueError(f"No diagnostics found for {task}")
        n_clusters = int(round(_mean([float(item.n_clusters) for item in task_diags])))
        omega = _mean([item.omega for item in task_diags])
        if n_clusters != expected_c or abs(omega - expected_omega) > 1e-12:
            raise ValueError(
                f"Unexpected tuned configuration for {task}: C={n_clusters}, omega={omega}. "
                f"Expected C={expected_c}, omega={expected_omega}."
            )
        active_mean = _mean([float(item.active_count) for item in task_diags])
        active_text = f"{int(round(active_mean))}/{n_clusters}"
        rows.append(
            TableRow(
                task=task,
                n_clusters=n_clusters,
                omega=omega,
                active=active_text,
                train_error=_mean([item.train_error for item in task_diags]),
                train_r2=_mean([item.train_r2 for item in task_diags]),
                action_purity=_mean([item.action_purity for item in task_diags]),
                learned_persistence=_mean([item.learned_persistence for item in task_diags]),
                max_terminal_risk=max(item.max_terminal_risk for item in task_diags),
            )
        )
    return rows


def _format_omega(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _format_train_error(value: float) -> str:
    return f"{value:.5f}" if abs(value) < 0.01 else f"{value:.4f}"


def table_row_to_csv(row: TableRow) -> dict[str, str]:
    return {
        "task": row.task,
        "C": str(row.n_clusters),
        "omega": _format_omega(row.omega),
        "active": row.active,
        "train_error": _format_train_error(row.train_error),
        "train_R2": f"{row.train_r2:.4f}",
        "action_purity": f"{row.action_purity:.3f}",
        "learned_persistence": f"{row.learned_persistence:.3f}",
        "max_terminal_risk": f"{row.max_terminal_risk:.3f}",
        "train_error_raw": f"{row.train_error:.17g}",
        "train_R2_raw": f"{row.train_r2:.17g}",
        "action_purity_raw": f"{row.action_purity:.17g}",
        "learned_persistence_raw": f"{row.learned_persistence:.17g}",
        "max_terminal_risk_raw": f"{row.max_terminal_risk:.17g}",
    }


def table_row_to_latex_cells(row: TableRow) -> list[str]:
    return [
        row.task,
        str(row.n_clusters),
        _format_omega(row.omega),
        row.active,
        _format_train_error(row.train_error),
        f"{row.train_r2:.4f}",
        f"{row.action_purity:.3f}",
        f"{row.learned_persistence:.3f}",
        f"{row.max_terminal_risk:.3f}",
    ]


def write_table_files(rows: Sequence[TableRow], diagnostics: Sequence[SeedDiagnostic], args: CliArgs) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = [table_row_to_csv(row) for row in rows]

    csv_path = args.output_dir / "table12_interpretability_values.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path = args.output_dir / "table12_interpretability_values.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Task | C | omega | Active | Train error | Train R2 | Action purity | Learned pers. | Max terminal risk |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in csv_rows:
            handle.write(
                f"| {row['task']} | {row['C']} | {row['omega']} | {row['active']} | "
                f"{row['train_error']} | {row['train_R2']} | {row['action_purity']} | "
                f"{row['learned_persistence']} | {row['max_terminal_risk']} |\n"
            )

    tex_path = args.output_dir / "table12_interpretability_values.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by step_11_generate_table12_interpretability.py\n")
        handle.write("\\begin{table}[t]\n")
        handle.write("\\centering\n")
        handle.write("\\scriptsize\n")
        handle.write(
            "\\caption{Regime-level interpretability diagnostics for the tuned Classic Control abstractions. "
            "Action purity and learned persistence are sample-count-weighted averages over active regimes. "
            "The terminal-risk column reports the maximum empirical risk of reaching a terminal state within 25 steps over active regimes and should be interpreted diagnostically, especially for small-support regimes.}\n"
        )
        handle.write("\\label{tab:classic_control_interpretability}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{lrrrrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write(
            "Task & $C$ & $\\omega$ & Active & Train error & Train $R^2$ & Action purity & Learned pers. & Max terminal risk \\\\\n"
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
        "seeds": list(args.seeds),
        "task_configs": {task: {"C": cfg[0], "omega": cfg[1]} for task, cfg in TASK_CONFIGS.items()},
        "aggregation": "default seed 0; if multiple seeds are supplied, table values are averaged across seed-level diagnostics except max terminal risk, which is maximized.",
        "seed_level_diagnostics": [asdict(item) for item in diagnostics],
    }
    with (args.output_dir / "table12_interpretability_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Wrote:")
    for path in (csv_path, md_path, tex_path, args.output_dir / "table12_interpretability_metadata.json"):
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    run_experiment(args)
    diagnostics = load_seed_diagnostics(args)
    rows = summarize_diagnostics(diagnostics)
    write_table_files(rows, diagnostics, args)


if __name__ == "__main__":
    main()
