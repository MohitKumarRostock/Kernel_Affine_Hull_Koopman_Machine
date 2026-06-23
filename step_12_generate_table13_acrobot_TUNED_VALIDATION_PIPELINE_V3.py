#!/usr/bin/env python3
"""Generate manuscript Table 13 from the tuned Acrobot-v1 validation pipeline.

This is the careful Table 13 wrapper.

Manuscript logic
----------------
Table 13 is not a generic Acrobot formatter.  It is the benchmark table after
Acrobot tuning selected the KAHKM configuration

    C = 150, omega = 0.5

and the selected configuration was then validated over three seeds.  Therefore
this script explicitly runs/reuses

    run_exp18_acrobot_selected_3seed_validation.py

with --n-clusters 150 --omega 0.5, and then aggregates the *per-seed*
Experiment 18 outputs directly.  It does not trust stale aggregate R2 columns:
R2_50 is read from each seed's experiment_18_multistep_summary.csv using robust
column aliases (association_r2, test_r2, r2, etc.) and then aggregated, so the
final table cannot contain nan±nan when per-seed values are finite.

Optional provenance check
-------------------------
Use --run-tuning-first if you also want this wrapper to run the Acrobot tuning
runner before the selected validation.  By default this provenance run uses seed
0, because the manuscript selected C=150, omega=0.5 from the tuning stage before
performing the three-seed selected validation.

Typical use, from the experiment-script folder:

    python step_12_generate_table13_acrobot_TUNED_VALIDATION_PIPELINE_V3.py --skip-existing

Reuse already generated selected-validation outputs only:

    python step_12_generate_table13_acrobot_TUNED_VALIDATION_PIPELINE_V3.py --no-run

Full provenance check plus selected validation:

    python step_12_generate_table13_acrobot_TUNED_VALIDATION_PIPELINE_V3.py \
        --run-tuning-first --skip-existing

Outputs:
    kahkm_table13_acrobot_benchmark_tuned/table13_acrobot_benchmark_values.csv
    kahkm_table13_acrobot_benchmark_tuned/table13_acrobot_benchmark_values.md
    kahkm_table13_acrobot_benchmark_tuned/table13_acrobot_benchmark_values.tex
    kahkm_table13_acrobot_benchmark_tuned/table13_acrobot_benchmark_metadata.json
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


SELECTED_C: Final[int] = 150
SELECTED_OMEGA: Final[float] = 0.5
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 5, 10, 20, 50)


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    output_root: Path
    output_dir: Path
    python_executable: str
    selected_runner: str
    tuning_runner: str
    experiment_script: str
    no_run: bool
    skip_existing: bool
    run_tuning_first: bool
    verify_tuning_selection: bool
    selected_output_prefix: str
    tuning_output_prefix: str
    seeds: tuple[int, ...]
    tuning_seeds: tuple[int, ...]
    horizons: tuple[int, ...]
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


@dataclass(frozen=True)
class MethodSpec:
    display_name: str
    one_step_method: str
    multistep_method: str


@dataclass(frozen=True)
class MetricPair:
    mean: float
    std: float


@dataclass(frozen=True)
class TableRow:
    method: str
    e1: MetricPair
    r2_1: MetricPair
    e50: MetricPair
    r2_50: MetricPair


METHOD_SPECS: Final[tuple[MethodSpec, ...]] = (
    MethodSpec("KAHKM + NLMS", "kahkm_nlms", "kahkm_nlms"),
    MethodSpec(
        "KAHKM + simplex projection",
        "kahkm_nlms_simplex_project",
        "kahkm_nlms_simplex_project_each_step",
    ),
    MethodSpec("KMeans RBF + NLMS", "kmeans_rbf_nlms", "kmeans_rbf_nlms"),
    MethodSpec("KMeans distance + NLMS", "kmeans_distance_nlms", "kmeans_distance_nlms"),
    MethodSpec("KMeans hard + NLMS", "kmeans_hard_nlms", "kmeans_hard_nlms"),
    MethodSpec("State DMD", "state_dmd", "state_dmd"),
    MethodSpec("State EDMD poly2", "state_edmd_poly2", "state_edmd_poly2"),
    MethodSpec("State EDMD poly3", "state_edmd_poly3", "state_edmd_poly3"),
)


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Table 13 from tuned Acrobot selected-validation outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table13_acrobot_benchmark_tuned"),
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument(
        "--selected-runner",
        default="run_exp18_acrobot_selected_3seed_validation.py",
    )
    parser.add_argument(
        "--tuning-runner",
        default="run_exp18_acrobot_tuning_pylance_clean.py",
    )
    parser.add_argument(
        "--experiment-script",
        default="experiment_18_acrobot_sequential_decision_pylance_clean.py",
    )
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--run-tuning-first",
        action="store_true",
        help="Run the Acrobot tuning runner before selected validation as a provenance check.",
    )
    parser.add_argument(
        "--no-verify-tuning-selection",
        action="store_true",
        help="Do not require the tuning provenance output to contain C=150, omega=0.5.",
    )
    parser.add_argument(
        "--selected-output-prefix",
        default="kahkm_exp18_acrobot_selected_seed",
    )
    parser.add_argument(
        "--tuning-output-prefix",
        default="kahkm_exp18_acrobot_tuned_seed",
    )
    parser.add_argument("--seeds", nargs="+", default=[str(v) for v in DEFAULT_SEEDS])
    parser.add_argument(
        "--tuning-seeds",
        nargs="+",
        default=["0"],
        help="Seeds used only for optional tuning provenance check. Default: 0.",
    )
    parser.add_argument("--horizons", nargs="+", default=[str(v) for v in DEFAULT_HORIZONS])
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--exploration-eps", type=float, default=0.10)
    parser.add_argument("--risk-window", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--kmeans-kind", default="full")
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument("--test-seed-base", type=int, default=100)

    ns = parser.parse_args()
    source_dir = Path(ns.source_dir).resolve()
    output_root = Path(ns.output_root).resolve() if ns.output_root is not None else source_dir
    output_dir = Path(ns.output_dir)
    if not output_dir.is_absolute():
        output_dir = source_dir / output_dir
    return CliArgs(
        source_dir=source_dir,
        output_root=output_root,
        output_dir=output_dir,
        python_executable=str(ns.python_executable),
        selected_runner=str(ns.selected_runner),
        tuning_runner=str(ns.tuning_runner),
        experiment_script=str(ns.experiment_script),
        no_run=bool(ns.no_run),
        skip_existing=bool(ns.skip_existing),
        run_tuning_first=bool(ns.run_tuning_first),
        verify_tuning_selection=not bool(ns.no_verify_tuning_selection),
        selected_output_prefix=str(ns.selected_output_prefix),
        tuning_output_prefix=str(ns.tuning_output_prefix),
        seeds=_parse_int_tuple([str(v) for v in ns.seeds]),
        tuning_seeds=_parse_int_tuple([str(v) for v in ns.tuning_seeds]),
        horizons=_parse_int_tuple([str(v) for v in ns.horizons]),
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
    )


def _print_and_run(command: Sequence[str], *, cwd: Path) -> None:
    print("\n" + "=" * 96)
    print(" ".join(command))
    print("=" * 96, flush=True)
    subprocess.run(list(command), cwd=str(cwd), check=True)


def _runner_common_args(args: CliArgs) -> list[str]:
    return [
        "--script",
        args.experiment_script,
        "--train-episodes",
        str(args.train_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--max-steps",
        str(args.max_steps),
        "--horizons",
        *[str(h) for h in args.horizons],
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
        args.kmeans_kind,
        "--kmeans-batch-size",
        str(args.kmeans_batch_size),
        "--max-train-per-cluster",
        str(args.max_train_per_cluster),
        "--test-seed-base",
        str(args.test_seed_base),
        "--output-root",
        str(args.output_root),
    ]


def run_optional_tuning(args: CliArgs) -> None:
    if args.no_run or not args.run_tuning_first:
        return
    runner_path = args.source_dir / args.tuning_runner
    if not runner_path.exists():
        raise FileNotFoundError(f"Missing tuning runner: {runner_path}")
    command = [
        args.python_executable,
        args.tuning_runner,
        "--python",
        args.python_executable,
        "--seeds",
        *[str(seed) for seed in args.tuning_seeds],
        "--clusters",
        "20",
        "40",
        "50",
        "75",
        "100",
        "150",
        "--omegas",
        "0.5",
        "1",
        "2",
        "4",
        "8",
        "--output-prefix",
        args.tuning_output_prefix,
        *_runner_common_args(args),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    _print_and_run(command, cwd=args.source_dir)


def _first_present_float(row: dict[str, str], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = optional_float(row.get(key, ""))
        if value is not None:
            return value
    return None


def _extract_config_from_row(row: dict[str, str]) -> tuple[int, float] | None:
    """Extract C and omega from either aggregate rows or per-seed best-config rows.

    The original Experiment 18 tuning runner writes per-seed best-config files with
    columns `best_n_clusters` and `best_omega`, but some aggregate builders look
    for `C`, `n_clusters`, and `omega`, which can leave aggregate selected columns
    blank.  For provenance verification we therefore accept all known aliases and
    use per-seed best-config files as the source of truth when the aggregate row is
    incomplete.
    """
    c_value = _first_present_float(
        row,
        (
            "selected_C",
            "selected_n_clusters",
            "best_C",
            "best_n_clusters",
            "C",
            "n_clusters",
            "clusters",
        ),
    )
    omega_value = _first_present_float(
        row,
        (
            "selected_omega",
            "best_omega",
            "omega",
        ),
    )
    if c_value is None or omega_value is None:
        return None
    return int(round(c_value)), float(omega_value)


def _tuning_provenance_candidates(args: CliArgs) -> list[tuple[str, int, float]]:
    candidates: list[tuple[str, int, float]] = []

    aggregate = args.output_root / "kahkm_exp18_acrobot_tuning_aggregate" / "experiment_18_seed_summary.csv"
    if aggregate.exists():
        for row in read_csv(aggregate):
            config = _extract_config_from_row(row)
            if config is not None:
                c_value, omega_value = config
                seed_label = row.get("seed", "?")
                candidates.append((f"{aggregate} seed={seed_label}", c_value, omega_value))

    for seed in args.tuning_seeds:
        best_path = args.output_root / f"{args.tuning_output_prefix}{seed}" / "experiment_18_best_config.csv"
        if not best_path.exists():
            continue
        for row in read_csv(best_path):
            config = _extract_config_from_row(row)
            if config is not None:
                c_value, omega_value = config
                candidates.append((str(best_path), c_value, omega_value))

    return candidates


def verify_tuning_selection_if_available(args: CliArgs) -> None:
    if not args.verify_tuning_selection:
        return

    aggregate = args.output_root / "kahkm_exp18_acrobot_tuning_aggregate" / "experiment_18_seed_summary.csv"
    candidates = _tuning_provenance_candidates(args)

    if not candidates:
        if args.run_tuning_first:
            raise ValueError(
                "Tuning provenance was expected, but no parseable selected configuration was found. "
                f"Checked aggregate {aggregate} and per-seed best-config files with prefix "
                f"{args.tuning_output_prefix!r}. This usually means the aggregate has blank "
                "selected_C/selected_omega columns; inspect experiment_18_best_config.csv."
            )
        return

    matched = any(
        c_value == SELECTED_C and abs(omega_value - SELECTED_OMEGA) <= 1e-12
        for _, c_value, omega_value in candidates
    )
    if not matched:
        found = "; ".join(
            f"{source}: C={c_value}, omega={omega_value:g}"
            for source, c_value, omega_value in candidates
        )
        raise ValueError(
            "Tuning provenance did not select the manuscript configuration "
            f"C={SELECTED_C}, omega={SELECTED_OMEGA}. Found: {found}. "
            "Do not update Table 13 until this tuning/selection discrepancy is resolved."
        )

    # Informative only: the run can proceed even if the aggregate row was blank,
    # because the per-seed best-config file is the authoritative provenance file.
    print(
        f"Verified Acrobot tuning provenance contains C={SELECTED_C}, omega={SELECTED_OMEGA}.",
        flush=True,
    )


def run_selected_validation(args: CliArgs) -> None:
    if args.no_run:
        return
    runner_path = args.source_dir / args.selected_runner
    if not runner_path.exists():
        raise FileNotFoundError(f"Missing selected-validation runner: {runner_path}")
    command = [
        args.python_executable,
        args.selected_runner,
        "--python",
        args.python_executable,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--n-clusters",
        str(SELECTED_C),
        "--omega",
        str(SELECTED_OMEGA),
        "--output-prefix",
        args.selected_output_prefix,
        *_runner_common_args(args),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    _print_and_run(command, cwd=args.source_dir)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def finite_float(value: str | None, *, context: str) -> float:
    result = optional_float(value)
    if result is None:
        raise ValueError(f"Missing or non-finite numeric value for {context}: {value!r}")
    return result


def finite_from_aliases(row: dict[str, str], aliases: Sequence[str], *, context: str) -> float:
    """Return the first finite numeric value found under any accepted column name.

    Experiment 18 has used several R^2 column names across script revisions:
    association_r2, test_r2, r2, and r2_score.  Table 13 is generated from
    selected per-seed validation outputs, so the wrapper should be robust to
    these naming aliases while still refusing genuinely missing/non-finite data.
    """
    for name in aliases:
        if name in row:
            value = optional_float(row.get(name))
            if value is not None:
                return value
    available = ", ".join(sorted(row.keys()))
    alias_text = ", ".join(aliases)
    raise ValueError(
        f"Missing or non-finite numeric value for {context}. "
        f"Tried aliases [{alias_text}]. Available columns: [{available}]."
    )


ERROR_ALIASES: Final[tuple[str, ...]] = (
    "relative_error",
    "test_error",
    "error",
    "association_error",
    "test_relative_error",
)
R2_ALIASES: Final[tuple[str, ...]] = (
    "association_r2",
    "test_r2",
    "r2",
    "R2",
    "r2_score",
    "association_R2",
    "test_association_r2",
    "relative_r2",
)


def selected_run_dirs(args: CliArgs) -> list[Path]:
    dirs = [args.output_root / f"{args.selected_output_prefix}{seed}" for seed in args.seeds]
    missing = [path for path in dirs if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing selected-validation seed output folders:\n"
            f"{missing_text}\n"
            "Run without --no-run, pass the correct --output-root/--selected-output-prefix, "
            "or regenerate Experiment 18 selected validation."
        )
    return dirs


def mean_pstdev(values: Sequence[float]) -> MetricPair:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) != len(values) or not finite:
        raise ValueError(f"Cannot summarize non-finite or empty values: {values}")
    return MetricPair(
        mean=float(statistics.fmean(finite)),
        std=float(statistics.pstdev(finite)) if len(finite) > 1 else 0.0,
    )


def validate_selected_config(run_dir: Path) -> None:
    best_path = run_dir / "experiment_18_best_config.csv"
    if not best_path.exists():
        return
    rows = read_csv(best_path)
    if not rows:
        raise ValueError(f"Best-config CSV is empty: {best_path}")
    row = rows[0]
    c_value = optional_float(row.get("best_n_clusters", row.get("C", row.get("n_clusters", ""))))
    omega_value = optional_float(row.get("best_omega", row.get("omega", "")))
    if c_value is None or omega_value is None:
        return
    if int(round(c_value)) != SELECTED_C or abs(omega_value - SELECTED_OMEGA) > 1e-12:
        raise ValueError(
            f"{best_path} reports C={c_value}, omega={omega_value}, but Table 13 "
            f"requires the tuned selected configuration C={SELECTED_C}, omega={SELECTED_OMEGA}."
        )


def collect_seed_metric_values(args: CliArgs, spec: MethodSpec) -> tuple[list[float], list[float], list[float], list[float]]:
    e1_values: list[float] = []
    r2_1_values: list[float] = []
    e50_values: list[float] = []
    r2_50_values: list[float] = []

    for run_dir in selected_run_dirs(args):
        validate_selected_config(run_dir)
        one_rows = read_csv(run_dir / "experiment_18_one_step_results.csv")
        multi_rows = read_csv(run_dir / "experiment_18_multistep_summary.csv")

        one = next((row for row in one_rows if row.get("method", "") == spec.one_step_method), None)
        if one is None:
            raise ValueError(f"Missing one-step method {spec.one_step_method!r} in {run_dir}")
        multi = next(
            (
                row for row in multi_rows
                if row.get("method", "") == spec.multistep_method
                and int(float(row.get("horizon", "nan"))) == 50
            ),
            None,
        )
        if multi is None:
            raise ValueError(f"Missing h=50 method {spec.multistep_method!r} in {run_dir}")

        e1_values.append(
            finite_from_aliases(one, ERROR_ALIASES, context=f"{run_dir} {spec.one_step_method} E1")
        )
        r2_1_values.append(
            finite_from_aliases(one, R2_ALIASES, context=f"{run_dir} {spec.one_step_method} R2_1")
        )
        e50_values.append(
            finite_from_aliases(multi, ERROR_ALIASES, context=f"{run_dir} {spec.multistep_method} E50")
        )
        r2_50_values.append(
            finite_from_aliases(multi, R2_ALIASES, context=f"{run_dir} {spec.multistep_method} R2_50")
        )

    return e1_values, r2_1_values, e50_values, r2_50_values


def build_table_rows(args: CliArgs) -> list[TableRow]:
    rows: list[TableRow] = []
    for spec in METHOD_SPECS:
        e1_values, r2_1_values, e50_values, r2_50_values = collect_seed_metric_values(args, spec)
        rows.append(
            TableRow(
                method=spec.display_name,
                e1=mean_pstdev(e1_values),
                r2_1=mean_pstdev(r2_1_values),
                e50=mean_pstdev(e50_values),
                r2_50=mean_pstdev(r2_50_values),
            )
        )
    return rows


def pm_text(value: MetricPair, decimals: int = 4) -> str:
    return f"{value.mean:.{decimals}f}±{value.std:.{decimals}f}"


def pm_latex(value: MetricPair, decimals: int = 4) -> str:
    return f"${value.mean:.{decimals}f}\\pm{value.std:.{decimals}f}$"


def table_row_to_csv(row: TableRow) -> dict[str, str]:
    return {
        "method": row.method,
        "E1": pm_text(row.e1),
        "R2_1": pm_text(row.r2_1),
        "E50": pm_text(row.e50),
        "R2_50": pm_text(row.r2_50),
        "E1_mean": f"{row.e1.mean:.17g}",
        "E1_std": f"{row.e1.std:.17g}",
        "R2_1_mean": f"{row.r2_1.mean:.17g}",
        "R2_1_std": f"{row.r2_1.std:.17g}",
        "E50_mean": f"{row.e50.mean:.17g}",
        "E50_std": f"{row.e50.std:.17g}",
        "R2_50_mean": f"{row.r2_50.mean:.17g}",
        "R2_50_std": f"{row.r2_50.std:.17g}",
    }


def markdown_table(rows: Sequence[TableRow]) -> str:
    lines = [
        "| Method | E1 | R2_1 | E50 | R2_50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.method} | {pm_text(row.e1)} | {pm_text(row.r2_1)} | "
            f"{pm_text(row.e50)} | {pm_text(row.r2_50)} |"
        )
    return "\n".join(lines) + "\n"


def latex_table(rows: Sequence[TableRow]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Tuned Acrobot-v1 sequential-decision benchmark validated over three seeds. The selected KAHKM configuration is $C=150,\omega=0.5$ with effective subspace dimension 6. Errors are relative association errors measured in the selected KAHKM association space; entries are mean $\pm$ standard deviation.}",
        r"\label{tab:acrobot_baselines}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        "Method & $E_1$ & $R^2_1$ & $E_{50}$ & $R^2_{50}$ " + r"\\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            row.method,
            pm_latex(row.e1),
            pm_latex(row.r2_1),
            pm_latex(row.e50),
            pm_latex(row.r2_50),
        ]
        lines.append(" & ".join(cells) + " " + r"\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def write_outputs(args: CliArgs, rows: Sequence[TableRow]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = [table_row_to_csv(row) for row in rows]
    csv_path = args.output_dir / "table13_acrobot_benchmark_values.csv"
    md_path = args.output_dir / "table13_acrobot_benchmark_values.md"
    tex_path = args.output_dir / "table13_acrobot_benchmark_values.tex"
    metadata_path = args.output_dir / "table13_acrobot_benchmark_metadata.json"

    write_csv(csv_path, csv_rows)
    md_path.write_text(markdown_table(rows), encoding="utf-8")
    tex_path.write_text(latex_table(rows), encoding="utf-8")
    metadata = {
        "table": "Table 13",
        "description": "Tuned Acrobot-v1 selected-validation benchmark.",
        "selected_config": {"C": SELECTED_C, "omega": SELECTED_OMEGA},
        "selection_provenance": (
            "C=150, omega=0.5 is the manuscript-selected Acrobot configuration from the tuning stage; "
            "this wrapper validates it over seeds and aggregates per-seed outputs directly."
        ),
        "std_convention": "population standard deviation across seed-level values, matching the Experiment 18 aggregate runner",
        "tuning_provenance_verification": "accepts best_n_clusters/best_omega from per-seed best-config files as authoritative when aggregate selected_C/selected_omega fields are blank",
        "aggregation_source": "per-seed experiment_18_one_step_results.csv and experiment_18_multistep_summary.csv",
        "numeric_column_aliases": {"errors": list(ERROR_ALIASES), "r2": list(R2_ALIASES)},
        "cli_args": {
            **asdict(args),
            "source_dir": str(args.source_dir),
            "output_root": str(args.output_root),
            "output_dir": str(args.output_dir),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote:")
    for path in (csv_path, md_path, tex_path, metadata_path):
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    if 1 not in args.horizons or 50 not in args.horizons:
        raise ValueError("Table 13 requires horizons 1 and 50.")
    run_optional_tuning(args)
    verify_tuning_selection_if_available(args)
    run_selected_validation(args)
    rows = build_table_rows(args)
    write_outputs(args, rows)


if __name__ == "__main__":
    main()
