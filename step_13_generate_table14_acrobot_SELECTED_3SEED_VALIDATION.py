#!/usr/bin/env python3
"""Generate manuscript Table 14 from selected Acrobot three-seed validation outputs.

This wrapper treats Table 14 as a selected-model validation result, not as a
standalone interpretability run.  It uses the same selected Acrobot setting as
Table 13:

    Acrobot-v1, C = 150, omega = 0.5, seeds = 0, 1, 2

By default it runs/reuses:

    run_exp18_acrobot_selected_3seed_validation.py

and then aggregates the *seed-level* files:

    kahkm_exp18_acrobot_selected_seed0/experiment_18_model_summary.csv
    kahkm_exp18_acrobot_selected_seed1/experiment_18_model_summary.csv
    kahkm_exp18_acrobot_selected_seed2/experiment_18_model_summary.csv

The script intentionally does not depend on the runner's aggregate
interpretability CSV, because older aggregate files/scripts may contain stale
or differently named interpretability columns.  Instead it recomputes Table 14
from the validated selected-model seed outputs.

Usage from the folder containing the experiment scripts:

    python step_13_generate_table14_acrobot_SELECTED_3SEED_VALIDATION.py --skip-existing

To aggregate already generated selected-validation outputs only:

    python step_13_generate_table14_acrobot_SELECTED_3SEED_VALIDATION.py --no-run

Outputs:

    kahkm_table14_acrobot_interpretability_selected/
        table14_acrobot_interpretability_values.csv
        table14_acrobot_interpretability_values.md
        table14_acrobot_interpretability_values.tex
        table14_acrobot_interpretability_metadata.json
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
from typing import Final, Literal, Sequence


SELECTED_N_CLUSTERS: Final[int] = 150
SELECTED_OMEGA: Final[float] = 0.5
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 5, 10, 20, 50)
DEFAULT_RISK_WINDOW: Final[int] = 25

StdConvention = Literal["population", "sample"]


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    experiment_output_root: Path
    output_dir: Path
    python_executable: str
    runner_script: str
    no_run: bool
    skip_existing: bool
    seeds: tuple[int, ...]
    horizons: tuple[int, ...]
    n_clusters: int
    omega: float
    train_episodes: int
    test_episodes: int
    max_steps: int
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    exploration_eps: float
    risk_window: int
    n_jobs: int
    batch_size: int
    kmeans_kind: str
    kmeans_batch_size: int
    max_train_per_cluster: int
    test_seed_base: int
    output_prefix: str
    zip_name: str
    std_convention: StdConvention
    allow_config_mismatch: bool


@dataclass(frozen=True)
class MetricSpec:
    label: str
    aliases: tuple[str, ...]
    decimals: int
    suffix: str = ""


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    n: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class TableRow:
    statistic: str
    value_plain: str
    value_latex: str
    mean: float
    std: float
    n: int
    seed_values: tuple[float, ...]


METRIC_SPECS: Final[tuple[MetricSpec, ...]] = (
    MetricSpec("Active regimes", ("active_regimes",), 1, " out of 150"),
    MetricSpec("Train closure error", ("train_closure_error",), 4),
    MetricSpec("Train association $R^2$", ("train_association_r2", "train_r2", "association_r2"), 4),
    MetricSpec(
        "Count-weighted action purity",
        ("count_weighted_action_purity", "action_purity"),
        4,
    ),
    MetricSpec(
        "Mass in regimes with action purity $\\ge 0.9$",
        ("mass_in_action_purity_ge_0p9", "high_purity_mass"),
        4,
    ),
    MetricSpec(
        "Weighted terminal risk within 25 steps",
        ("weighted_terminal_risk_within_25", "weighted_terminal_risk"),
        4,
    ),
    MetricSpec(
        "Terminal risk of highest-risk supported regime",
        ("max_risk_value", "max_terminal_risk"),
        4,
    ),
    MetricSpec(
        "Support of highest-risk supported regime",
        ("max_risk_regime_count", "max_risk_regime_support"),
        1,
    ),
)


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def _parse_std_convention(value: str) -> StdConvention:
    normalized = value.strip().lower()
    if normalized not in {"population", "sample"}:
        raise argparse.ArgumentTypeError("std convention must be 'population' or 'sample'")
    return normalized  # type: ignore[return-value]


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate Table 14 from selected Acrobot three-seed validation outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment-output-root",
        type=Path,
        default=None,
        help="Root containing/receiving kahkm_exp18_acrobot_selected_seed* directories. Default: --source-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table14_acrobot_interpretability_selected"),
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--runner-script", default="run_exp18_acrobot_selected_3seed_validation.py")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seeds", nargs="+", default=[str(seed) for seed in DEFAULT_SEEDS])
    parser.add_argument("--horizons", nargs="+", default=[str(horizon) for horizon in DEFAULT_HORIZONS])
    parser.add_argument("--n-clusters", type=int, default=SELECTED_N_CLUSTERS)
    parser.add_argument("--omega", type=float, default=SELECTED_OMEGA)
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--exploration-eps", type=float, default=0.10)
    parser.add_argument("--risk-window", type=int, default=DEFAULT_RISK_WINDOW)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--kmeans-kind", default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument("--test-seed-base", type=int, default=100)
    parser.add_argument("--output-prefix", default="kahkm_exp18_acrobot_selected_seed")
    parser.add_argument("--zip-name", default="kahkm_exp18_acrobot_selected_3seed_validation_results")
    parser.add_argument(
        "--std-convention",
        type=_parse_std_convention,
        default="population",
        help=(
            "Standard deviation convention across the three validation seeds. "
            "Default: population, matching the Experiment 18 selected-validation aggregate."
        ),
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Do not fail if seed outputs do not report C=150 and omega=0.5.",
    )
    ns = parser.parse_args()

    source_dir = Path(ns.source_dir).resolve()
    output_root = Path(ns.experiment_output_root).resolve() if ns.experiment_output_root else source_dir
    output_dir = Path(ns.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir

    return CliArgs(
        source_dir=source_dir,
        experiment_output_root=output_root,
        output_dir=output_dir,
        python_executable=str(ns.python_executable),
        runner_script=str(ns.runner_script),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        seeds=_parse_int_tuple([str(value) for value in ns.seeds]),
        horizons=_parse_int_tuple([str(value) for value in ns.horizons]),
        n_clusters=int(ns.n_clusters),
        omega=float(ns.omega),
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        max_steps=int(ns.max_steps),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        exploration_eps=float(ns.exploration_eps),
        risk_window=int(ns.risk_window),
        n_jobs=int(ns.n_jobs),
        batch_size=int(ns.batch_size),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        max_train_per_cluster=int(ns.max_train_per_cluster),
        test_seed_base=int(ns.test_seed_base),
        output_prefix=str(ns.output_prefix),
        zip_name=str(ns.zip_name),
        std_convention=ns.std_convention,
        allow_config_mismatch=bool(ns.allow_config_mismatch),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: str | None, *, context: str) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"Missing numeric value for {context}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite numeric value for {context}: {value!r}")
    return parsed


def _optional_finite_float(row: dict[str, str], aliases: Sequence[str], *, context: str) -> float:
    for key in aliases:
        value = row.get(key)
        if value is None or value.strip() == "":
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    alias_text = ", ".join(aliases)
    raise ValueError(f"Could not find finite value for {context}. Tried: {alias_text}")


def _seed_output_dir(args: CliArgs, seed: int) -> Path:
    return args.experiment_output_root / f"{args.output_prefix}{seed}"


def run_selected_validation(args: CliArgs) -> None:
    if args.no_run:
        return
    runner_path = args.source_dir / args.runner_script
    if not runner_path.exists():
        raise FileNotFoundError(
            f"Could not find {runner_path}. Place this wrapper in the experiment folder, "
            "or pass --source-dir/--runner-script."
        )
    command = [
        args.python_executable,
        str(runner_path),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--train-episodes",
        str(args.train_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--max-steps",
        str(args.max_steps),
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
        "--n-clusters",
        str(args.n_clusters),
        "--omega",
        str(args.omega),
        "--subspace-dim",
        str(args.subspace_dim),
        "--nb",
        str(args.nb),
        "--beta",
        str(args.beta),
        "--nlms-epochs",
        str(args.nlms_epochs),
        "--exploration-eps",
        str(args.exploration_eps),
        "--risk-window",
        str(args.risk_window),
        "--n-jobs",
        str(args.n_jobs),
        "--batch-size",
        str(args.batch_size),
        "--kmeans-kind",
        str(args.kmeans_kind),
        "--kmeans-batch-size",
        str(args.kmeans_batch_size),
        "--max-train-per-cluster",
        str(args.max_train_per_cluster),
        "--test-seed-base",
        str(args.test_seed_base),
        "--output-root",
        str(args.experiment_output_root),
        "--output-prefix",
        args.output_prefix,
        "--zip-name",
        args.zip_name,
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    print("Running selected Acrobot three-seed validation:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)


def load_seed_model_summaries(args: CliArgs) -> list[tuple[int, dict[str, str]]]:
    summaries: list[tuple[int, dict[str, str]]] = []
    for seed in args.seeds:
        seed_dir = _seed_output_dir(args, seed)
        summary_path = seed_dir / "experiment_18_model_summary.csv"
        rows = _read_csv(summary_path)
        if not rows:
            raise ValueError(f"No rows found in {summary_path}")
        row = rows[0]
        if not args.allow_config_mismatch:
            observed_c = _optional_finite_float(
                row,
                ("selected_n_clusters", "n_clusters", "C"),
                context=f"selected C for seed {seed}",
            )
            observed_omega = _optional_finite_float(
                row,
                ("selected_omega", "omega"),
                context=f"selected omega for seed {seed}",
            )
            if int(round(observed_c)) != args.n_clusters or not math.isclose(observed_omega, args.omega, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Seed {seed} is not the requested selected model. "
                    f"Expected C={args.n_clusters}, omega={args.omega}; "
                    f"found C={observed_c}, omega={observed_omega} in {summary_path}."
                )
        summaries.append((seed, row))
    return summaries


def summarize(values: Sequence[float], convention: StdConvention) -> MetricSummary:
    finite = tuple(value for value in values if math.isfinite(value))
    if not finite:
        raise ValueError("Cannot summarize an empty/non-finite value list.")
    mean_value = float(statistics.fmean(finite))
    if len(finite) == 1:
        std_value = 0.0
    elif convention == "sample":
        std_value = float(statistics.stdev(finite))
    else:
        std_value = float(statistics.pstdev(finite))
    return MetricSummary(mean=mean_value, std=std_value, n=len(finite), values=finite)


def format_plain(summary: MetricSummary, decimals: int, suffix: str) -> str:
    return f"{summary.mean:.{decimals}f}±{summary.std:.{decimals}f}{suffix}"


def format_latex(summary: MetricSummary, decimals: int, suffix: str) -> str:
    return f"${summary.mean:.{decimals}f}\\pm{summary.std:.{decimals}f}${suffix}"


def build_table_rows(seed_summaries: Sequence[tuple[int, dict[str, str]]], args: CliArgs) -> list[TableRow]:
    rows: list[TableRow] = []
    for spec in METRIC_SPECS:
        values = [
            _optional_finite_float(row, spec.aliases, context=f"{spec.label} for seed {seed}")
            for seed, row in seed_summaries
        ]
        summary = summarize(values, args.std_convention)
        rows.append(
            TableRow(
                statistic=spec.label,
                value_plain=format_plain(summary, spec.decimals, spec.suffix),
                value_latex=format_latex(summary, spec.decimals, spec.suffix),
                mean=summary.mean,
                std=summary.std,
                n=summary.n,
                seed_values=summary.values,
            )
        )
    return rows


def rows_to_csv(rows: Sequence[TableRow]) -> list[dict[str, str]]:
    return [
        {
            "statistic": row.statistic,
            "value": row.value_plain,
            "mean": f"{row.mean:.17g}",
            "std": f"{row.std:.17g}",
            "n": str(row.n),
            "seed_values": ";".join(f"{value:.17g}" for value in row.seed_values),
        }
        for row in rows
    ]


def markdown_table(rows: Sequence[TableRow]) -> str:
    lines = ["| Statistic | Value |", "|---|---:|"]
    for row in rows:
        lines.append(f"| {row.statistic} | {row.value_plain} |")
    return "\n".join(lines) + "\n"


def latex_table(rows: Sequence[TableRow]) -> str:
    lines = [
        "% Auto-generated by step_13_generate_table14_acrobot_SELECTED_3SEED_VALIDATION.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Acrobot-v1 regime-level interpretability diagnostics for the selected KAHKM model, validated over three seeds. Terminal risk denotes empirical probability of termination within 25 steps; entries are mean $\pm$ standard deviation where applicable.}",
        r"\label{tab:acrobot_interpretability}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lr}",
        r"\toprule",
        "Statistic & Value \\\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row.statistic} & {row.value_latex} " + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(rows: Sequence[TableRow], seed_summaries: Sequence[tuple[int, dict[str, str]]], args: CliArgs) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "table14_acrobot_interpretability_values.csv"
    md_path = args.output_dir / "table14_acrobot_interpretability_values.md"
    tex_path = args.output_dir / "table14_acrobot_interpretability_values.tex"
    meta_path = args.output_dir / "table14_acrobot_interpretability_metadata.json"

    _write_csv(csv_path, rows_to_csv(rows))
    md_path.write_text(markdown_table(rows), encoding="utf-8")
    tex_path.write_text(latex_table(rows), encoding="utf-8")

    metadata = {
        "table": "Table 14",
        "description": "Acrobot-v1 regime-level interpretability diagnostics for selected C=150, omega=0.5 model validated over three seeds.",
        "selected_n_clusters": args.n_clusters,
        "selected_omega": args.omega,
        "seeds": list(args.seeds),
        "risk_window": args.risk_window,
        "std_convention": args.std_convention,
        "source_runner": args.runner_script,
        "source_seed_dirs": [str(_seed_output_dir(args, seed)) for seed in args.seeds],
        "cli_args": {
            **asdict(args),
            "source_dir": str(args.source_dir),
            "experiment_output_root": str(args.experiment_output_root),
            "output_dir": str(args.output_dir),
        },
        "seed_model_summary_rows": [
            {"seed": seed, "row": row} for seed, row in seed_summaries
        ],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Wrote:")
    for path in (csv_path, md_path, tex_path, meta_path):
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    run_selected_validation(args)
    seed_summaries = load_seed_model_summaries(args)
    rows = build_table_rows(seed_summaries, args)
    write_outputs(rows, seed_summaries, args)


if __name__ == "__main__":
    main()
