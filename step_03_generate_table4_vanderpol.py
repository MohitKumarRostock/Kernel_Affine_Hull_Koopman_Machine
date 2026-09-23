#!/usr/bin/env python3
"""Generate manuscript Table 4 from centered/retained-variation Experiment 09 outputs.

Table 4 documents the clean Van der Pol representation-selection rule used by
the manuscript. The primary selection objective is centered multi-horizon error

    E_c = mean_h (1 - R_h^2),   h in {1, 10, 50, 100, 200}.

The one-standard-error set is formed using the SE of the lowest-mean centered
error. Retained variation is then used only within that finalist set, with
normalized effective rank as the secondary tie-break.

The manuscript-facing table contains the one-SE finalists. The full centered
grid ranking is exported as a companion CSV. Legacy uncentered E_h values are
merged into the machine-readable outputs when an Experiment 09 multistep
summary is available, but they do not affect selection.

Expected centered Experiment 09 files
-------------------------------------
- experiment_09_centered_one_se_ranking.csv
- experiment_09_centered_one_se_finalists.csv
- experiment_09_centered_retvar_selected_config.csv

Typical formatting-only use with the validated result directory:

    python step_03_generate_table4_vanderpol.py \
        --no-run \
        --experiment-output-dir kahkm_exp09_retvar_fullgrid_selection

Generated outputs
-----------------
- kahkm_table4_centered_vanderpol/table4_vanderpol_values.csv
- kahkm_table4_centered_vanderpol/table4_vanderpol_values.md
- kahkm_table4_centered_vanderpol/table4_vanderpol_values.tex
- kahkm_table4_centered_vanderpol/table4_vanderpol_full_centered_ranking.csv
- kahkm_table4_centered_vanderpol/table4_vanderpol_metadata.json
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


DEFAULT_CLUSTERS: tuple[int, ...] = (10, 15, 20, 25, 30, 40, 50)
DEFAULT_OMEGAS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_RANDOM_STATES: tuple[int, ...] = (0, 1, 2)
SELECTION_HORIZONS: tuple[int, ...] = (1, 10, 50, 100, 200)

EXPECTED_SELECTED: tuple[int, float] = (25, 6.0)
EXPECTED_FINALISTS: tuple[tuple[int, float], ...] = (
    (25, 6.0),
    (30, 6.0),
    (20, 4.0),
)

RUNNER_NAME = "run_exp09_vanderpol_tuning.py"
CENTERED_RANKING_NAME = "experiment_09_centered_one_se_ranking.csv"
FINALISTS_NAME = "experiment_09_centered_one_se_finalists.csv"
SELECTED_NAME = "experiment_09_centered_retvar_selected_config.csv"
LEGACY_SUMMARY_NAME = "experiment_09_multistep_summary.csv"

RawRow = dict[str, str]
StringRow = dict[str, str]


@dataclass(frozen=True)
class CliArgs:
    source_dir: Path
    runner_script: Path
    experiment_output_dir: Path
    table_output_dir: Path
    run_experiment: bool
    force_rerun: bool
    python_executable: str
    clusters: tuple[int, ...]
    omegas: tuple[float, ...]
    random_states: tuple[int, ...]
    horizons: tuple[int, ...]
    n_steps: int
    dt: float
    mu: float
    train_fraction: float
    tau: float
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    kmeans_kind: str
    batch_size: int
    n_jobs: int
    max_train_per_cluster: int
    legacy_summary: Path | None
    expected_selected_c: int
    expected_selected_omega: float


@dataclass(frozen=True)
class CenteredRow:
    centered_rank: int
    n_clusters: int
    omega: float
    selection_horizons: tuple[int, ...]
    n_runs: int
    mean_centered_error: float
    mean_multihorizon_r2: float
    centered_error_sd: float
    centered_error_se: float
    rho_var_mean: float
    rho_var_std: float
    effective_rank_mean: float
    effective_rank_std: float
    rho_rank_mean: float
    rho_rank_std: float
    inside_one_se: int
    selected: int
    best_mean_error: float
    best_standard_error: float
    one_se_limit: float
    retvar_rank_within_one_se: int | None


@dataclass(frozen=True)
class LegacyHorizonStats:
    e_mean: float
    e_std: float
    r2_mean: float
    r2_std: float


@dataclass(frozen=True)
class TableRow:
    centered: CenteredRow
    legacy: dict[int, LegacyHorizonStats]

    @property
    def selected_label(self) -> str:
        return "yes" if self.centered.selected == 1 else "--"

    def to_csv_row(self) -> StringRow:
        c = self.centered
        row: StringRow = {
            "centered_rank": str(c.centered_rank),
            "C": str(c.n_clusters),
            "omega": omega_label(c.omega),
            "selection_horizons": " ".join(str(h) for h in c.selection_horizons),
            "n_runs": str(c.n_runs),
            "mean_multihorizon_R2": format_float(c.mean_multihorizon_r2),
            "mean_centered_error": format_float(c.mean_centered_error),
            "centered_error_sd": format_float(c.centered_error_sd),
            "centered_error_se": format_float(c.centered_error_se),
            "rho_var_mean": format_optional_float(c.rho_var_mean),
            "rho_var_std": format_optional_float(c.rho_var_std),
            "effective_rank_mean": format_optional_float(c.effective_rank_mean),
            "effective_rank_std": format_optional_float(c.effective_rank_std),
            "rho_rank_mean": format_optional_float(c.rho_rank_mean),
            "rho_rank_std": format_optional_float(c.rho_rank_std),
            "inside_centered_one_se_set": str(c.inside_one_se),
            "selected_by_centered_retvar_rule": str(c.selected),
            "retained_variation_rank_within_one_se": (
                "" if c.retvar_rank_within_one_se is None
                else str(c.retvar_rank_within_one_se)
            ),
            "centered_best_mean_error": format_float(c.best_mean_error),
            "centered_best_standard_error": format_float(c.best_standard_error),
            "centered_one_se_limit": format_float(c.one_se_limit),
            "mean_multihorizon_R2_table_cell": f"${c.mean_multihorizon_r2:.4f}$",
            "centered_error_se_table_cell": (
                f"${c.mean_centered_error:.4f}\\pm{c.centered_error_se:.4f}$"
            ),
            "rho_var_table_cell": optional_pm_cell(c.rho_var_mean, c.rho_var_std, 4),
            "effective_rank_table_cell": optional_pm_cell(
                c.effective_rank_mean, c.effective_rank_std, 2
            ),
            "rho_rank_table_cell": optional_pm_cell(c.rho_rank_mean, c.rho_rank_std, 4),
            "selected_table_cell": self.selected_label,
        }

        # Legacy uncentered/centered per-horizon summaries are provenance only.
        for horizon in SELECTION_HORIZONS:
            stats = self.legacy.get(horizon)
            for suffix, value in (
                ("E_mean", math.nan if stats is None else stats.e_mean),
                ("E_std", math.nan if stats is None else stats.e_std),
                ("R2_mean", math.nan if stats is None else stats.r2_mean),
                ("R2_std", math.nan if stats is None else stats.r2_std),
            ):
                row[f"legacy_h{horizon}_{suffix}"] = format_optional_float(value)

        return row


def parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise ValueError("At least one integer value is required.")
    return parsed


def parse_float_tuple(values: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError("At least one floating-point value is required.")
    return parsed


def omega_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Expected finite float, received {value!r}")
    return f"{value:.17g}"


def format_optional_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.17g}"


def optional_pm_cell(mean_value: float, std_value: float, digits: int) -> str:
    if not math.isfinite(mean_value) or not math.isfinite(std_value):
        return "--"
    return f"${mean_value:.{digits}f}\\pm{std_value:.{digits}f}$"


def _finite_float(row: RawRow, key: str, *, context: str) -> float:
    text = row.get(key, "").strip()
    if not text:
        raise ValueError(f"Missing {key!r} for {context}. Available: {sorted(row)}")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} for {context}: {text!r}")
    return value


def _optional_float(row: RawRow, key: str) -> float:
    text = row.get(key, "").strip()
    if not text:
        return math.nan
    value = float(text)
    return value if math.isfinite(value) else math.nan


def _finite_int(row: RawRow, key: str, *, context: str) -> int:
    return int(round(_finite_float(row, key, context=context)))


def _parse_horizons(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.replace(",", " ").split())


def read_csv(path: Path) -> list[RawRow]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[StringRow]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate centered/retained-variation manuscript Table 4 from Experiment 09."
        )
    )
    parser.add_argument("--source-dir", default=".")
    parser.add_argument("--runner-script", default=RUNNER_NAME)
    parser.add_argument(
        "--experiment-output-dir",
        default="kahkm_exp09_vanderpol_tuning",
    )
    parser.add_argument(
        "--table-output-dir",
        default="kahkm_table4_centered_vanderpol",
    )

    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--run", dest="run_experiment", action="store_true", default=True
    )
    run_group.add_argument(
        "--no-run", dest="run_experiment", action="store_false"
    )

    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)

    parser.add_argument("--clusters", nargs="+", default=[str(v) for v in DEFAULT_CLUSTERS])
    parser.add_argument("--omegas", nargs="+", default=[omega_label(v) for v in DEFAULT_OMEGAS])
    parser.add_argument(
        "--random-states",
        nargs="+",
        default=[str(v) for v in DEFAULT_RANDOM_STATES],
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=[str(v) for v in SELECTION_HORIZONS],
    )

    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument(
        "--kmeans-kind",
        default="full",
        choices=["auto", "full", "minibatch"],
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)

    parser.add_argument(
        "--legacy-summary",
        default=None,
        help=(
            "Optional Experiment 09 multistep summary used only to append legacy E_h/R^2 "
            "provenance columns. If omitted, the generator searches the centered result "
            "directory and then <source-dir>/kahkm_exp09_vanderpol_tuning."
        ),
    )
    parser.add_argument("--expected-selected-c", type=int, default=EXPECTED_SELECTED[0])
    parser.add_argument(
        "--expected-selected-omega",
        type=float,
        default=EXPECTED_SELECTED[1],
    )
    return parser


def resolve_args(ns: argparse.Namespace) -> CliArgs:
    source_dir = Path(ns.source_dir).expanduser().resolve()
    experiment_output_dir = resolve_path(source_dir, ns.experiment_output_dir)
    table_output_dir = resolve_path(Path.cwd(), ns.table_output_dir)
    runner_script = resolve_path(source_dir, ns.runner_script)

    clusters = tuple(sorted(set(parse_int_tuple(ns.clusters))))
    omegas = tuple(sorted(set(parse_float_tuple(ns.omegas))))
    random_states = parse_int_tuple(ns.random_states)
    horizons = tuple(sorted(set(parse_int_tuple(ns.horizons))))

    if clusters != DEFAULT_CLUSTERS:
        raise ValueError(
            f"Table 4 expects cluster grid {DEFAULT_CLUSTERS}, received {clusters}."
        )
    if omegas != DEFAULT_OMEGAS:
        raise ValueError(
            f"Table 4 expects omega grid {DEFAULT_OMEGAS}, received {omegas}."
        )
    if horizons != SELECTION_HORIZONS:
        raise ValueError(
            f"Table 4 expects selection horizons {SELECTION_HORIZONS}, received {horizons}."
        )
    if tuple(sorted(random_states)) != DEFAULT_RANDOM_STATES:
        raise ValueError(
            f"Table 4 expects seeds {DEFAULT_RANDOM_STATES}, received {random_states}."
        )

    legacy_summary = None
    if ns.legacy_summary is not None:
        legacy_summary = resolve_path(source_dir, ns.legacy_summary)

    return CliArgs(
        source_dir=source_dir,
        runner_script=runner_script,
        experiment_output_dir=experiment_output_dir,
        table_output_dir=table_output_dir,
        run_experiment=bool(ns.run_experiment),
        force_rerun=bool(ns.force_rerun),
        python_executable=str(ns.python_executable),
        clusters=clusters,
        omegas=omegas,
        random_states=random_states,
        horizons=horizons,
        n_steps=int(ns.n_steps),
        dt=float(ns.dt),
        mu=float(ns.mu),
        train_fraction=float(ns.train_fraction),
        tau=float(ns.tau),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        kmeans_kind=str(ns.kmeans_kind),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        max_train_per_cluster=int(ns.max_train_per_cluster),
        legacy_summary=legacy_summary,
        expected_selected_c=int(ns.expected_selected_c),
        expected_selected_omega=float(ns.expected_selected_omega),
    )


def centered_files_complete(output_dir: Path) -> bool:
    return all(
        (output_dir / name).exists()
        for name in (CENTERED_RANKING_NAME, FINALISTS_NAME, SELECTED_NAME)
    )


def run_underlying_experiment(args: CliArgs) -> None:
    if centered_files_complete(args.experiment_output_dir) and not args.force_rerun:
        print(f"Reusing centered Experiment 09 outputs: {args.experiment_output_dir}")
        return

    if not args.runner_script.exists():
        raise FileNotFoundError(f"Could not find runner script: {args.runner_script}")

    command = [
        args.python_executable,
        str(args.runner_script),
        "--output-dir",
        str(args.experiment_output_dir),
        "--n-steps",
        str(args.n_steps),
        "--dt",
        str(args.dt),
        "--mu",
        str(args.mu),
        "--train-fraction",
        str(args.train_fraction),
        "--clusters",
        *[str(value) for value in args.clusters],
        "--omegas",
        *[omega_label(value) for value in args.omegas],
        "--random-states",
        *[str(value) for value in args.random_states],
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
        "--kmeans-kind",
        args.kmeans_kind,
        "--batch-size",
        str(args.batch_size),
        "--n-jobs",
        str(args.n_jobs),
        "--horizons",
        *[str(value) for value in args.horizons],
        "--selection-horizons",
        *[str(value) for value in SELECTION_HORIZONS],
    ]
    if args.max_train_per_cluster > 0:
        command.extend(["--max-train-per-cluster", str(args.max_train_per_cluster)])

    print("Running centered Van der Pol Experiment 09:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(args.source_dir), check=True)

    if not centered_files_complete(args.experiment_output_dir):
        raise RuntimeError(
            "Experiment 09 completed without the centered selection outputs required "
            "for Table 4."
        )


def parse_centered_row(row: RawRow, *, finalist: bool = False) -> CenteredRow:
    context = (
        f"C={row.get('n_clusters','?')}, omega={row.get('omega','?')}"
    )
    horizons = _parse_horizons(row.get("selection_horizons", ""))
    retvar_rank: int | None = None
    if finalist:
        retvar_rank = _finite_int(
            row, "retained_variation_rank_within_one_se", context=context
        )

    return CenteredRow(
        centered_rank=_finite_int(row, "centered_rank", context=context),
        n_clusters=_finite_int(row, "n_clusters", context=context),
        omega=_finite_float(row, "omega", context=context),
        selection_horizons=horizons,
        n_runs=_finite_int(row, "n_runs", context=context),
        mean_centered_error=_finite_float(
            row, "mean_centered_error", context=context
        ),
        mean_multihorizon_r2=_finite_float(
            row, "mean_multihorizon_r2", context=context
        ),
        centered_error_sd=_finite_float(
            row, "centered_error_sd", context=context
        ),
        centered_error_se=_finite_float(
            row, "centered_error_se", context=context
        ),
        rho_var_mean=_optional_float(
            row, "target_normalized_association_variation_mean"
        ),
        rho_var_std=_optional_float(
            row, "target_normalized_association_variation_std"
        ),
        effective_rank_mean=_optional_float(
            row, "target_association_effective_rank_mean"
        ),
        effective_rank_std=_optional_float(
            row, "target_association_effective_rank_std"
        ),
        rho_rank_mean=_optional_float(
            row, "target_normalized_association_effective_rank_mean"
        ),
        rho_rank_std=_optional_float(
            row, "target_normalized_association_effective_rank_std"
        ),
        inside_one_se=_finite_int(
            row, "inside_centered_one_se_set", context=context
        ),
        selected=_finite_int(
            row, "selected_by_centered_retvar_rule", context=context
        ),
        best_mean_error=_finite_float(
            row, "centered_best_mean_error", context=context
        ),
        best_standard_error=_finite_float(
            row, "centered_best_standard_error", context=context
        ),
        one_se_limit=_finite_float(
            row, "centered_one_se_limit", context=context
        ),
        retvar_rank_within_one_se=retvar_rank,
    )


def row_key(row: CenteredRow) -> tuple[int, float]:
    return (row.n_clusters, row.omega)


def assert_close(a: float, b: float, label: str, *, atol: float = 1e-12) -> None:
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{label} mismatch: {a:.17g} vs {b:.17g}")


def validate_centered_outputs(
    ranking_rows: list[CenteredRow],
    finalist_rows: list[CenteredRow],
    selected_rows: list[CenteredRow],
    args: CliArgs,
) -> None:
    expected_grid = {
        (c, omega)
        for c in DEFAULT_CLUSTERS
        for omega in DEFAULT_OMEGAS
    }
    observed_grid = {row_key(row) for row in ranking_rows}
    if observed_grid != expected_grid:
        missing = sorted(expected_grid - observed_grid)
        extra = sorted(observed_grid - expected_grid)
        raise ValueError(
            f"Centered ranking grid mismatch; missing={missing}, extra={extra}."
        )
    if len(ranking_rows) != len(expected_grid):
        raise ValueError(
            f"Expected {len(expected_grid)} centered ranking rows, got {len(ranking_rows)}."
        )

    ranks = sorted(row.centered_rank for row in ranking_rows)
    if ranks != list(range(1, len(ranking_rows) + 1)):
        raise ValueError("Centered ranks are not a complete 1..N sequence.")

    for row in ranking_rows:
        if row.selection_horizons != SELECTION_HORIZONS:
            raise ValueError(
                f"Unexpected selection horizons for {row_key(row)}: "
                f"{row.selection_horizons}"
            )
        if row.n_runs != len(DEFAULT_RANDOM_STATES):
            raise ValueError(
                f"Expected three runs for {row_key(row)}, got {row.n_runs}."
            )
        assert_close(
            row.mean_multihorizon_r2,
            1.0 - row.mean_centered_error,
            f"R2/error identity for {row_key(row)}",
        )

    best = min(ranking_rows, key=lambda row: row.mean_centered_error)
    one_se_limit = best.mean_centered_error + best.centered_error_se

    for row in ranking_rows:
        assert_close(
            row.best_mean_error,
            best.mean_centered_error,
            f"best mean for {row_key(row)}",
        )
        assert_close(
            row.best_standard_error,
            best.centered_error_se,
            f"best SE for {row_key(row)}",
        )
        assert_close(
            row.one_se_limit,
            one_se_limit,
            f"one-SE limit for {row_key(row)}",
        )
        expected_inside = int(row.mean_centered_error <= one_se_limit + 1e-15)
        if row.inside_one_se != expected_inside:
            raise ValueError(
                f"one-SE membership mismatch for {row_key(row)}."
            )

    inside_from_ranking = {
        row_key(row) for row in ranking_rows if row.inside_one_se == 1
    }
    finalist_keys = {row_key(row) for row in finalist_rows}
    if finalist_keys != inside_from_ranking:
        raise ValueError(
            "Finalist CSV does not match the one-SE set from the centered ranking."
        )

    expected_finalist_set = set(EXPECTED_FINALISTS)
    if finalist_keys != expected_finalist_set:
        raise ValueError(
            f"Unexpected one-SE finalist set: {sorted(finalist_keys)}; "
            f"expected {sorted(expected_finalist_set)}."
        )

    ranking_by_key = {row_key(row): row for row in ranking_rows}
    for finalist in finalist_rows:
        ranking = ranking_by_key[row_key(finalist)]
        for attr in (
            "centered_rank",
            "mean_centered_error",
            "mean_multihorizon_r2",
            "centered_error_sd",
            "centered_error_se",
            "rho_var_mean",
            "rho_var_std",
            "effective_rank_mean",
            "effective_rank_std",
            "rho_rank_mean",
            "rho_rank_std",
            "inside_one_se",
            "selected",
            "best_mean_error",
            "best_standard_error",
            "one_se_limit",
        ):
            left = getattr(finalist, attr)
            right = getattr(ranking, attr)
            if isinstance(left, float):
                if math.isnan(left) and math.isnan(right):
                    continue
                assert_close(left, right, f"{attr} for {row_key(finalist)}")
            elif left != right:
                raise ValueError(
                    f"{attr} mismatch for {row_key(finalist)}: {left} vs {right}"
                )

    recomputed_finalists = sorted(
        finalist_rows,
        key=lambda row: (
            -row.rho_var_mean,
            -row.rho_rank_mean,
            row.mean_centered_error,
            row.n_clusters,
            row.omega,
        ),
    )
    for rank, row in enumerate(recomputed_finalists, start=1):
        if row.retvar_rank_within_one_se != rank:
            raise ValueError(
                f"Retained-variation finalist rank mismatch for {row_key(row)}."
            )

    if len(selected_rows) != 1:
        raise ValueError(
            f"Expected exactly one selected configuration row, got {len(selected_rows)}."
        )
    selected = selected_rows[0]
    expected_selected = (
        args.expected_selected_c,
        args.expected_selected_omega,
    )
    if row_key(selected) != expected_selected:
        raise ValueError(
            f"Selected configuration mismatch: observed {row_key(selected)}, "
            f"expected {expected_selected}."
        )
    if selected.selected != 1 or selected.inside_one_se != 1:
        raise ValueError("Selected row is not marked as selected inside the one-SE set.")

    recomputed_selected = row_key(recomputed_finalists[0])
    if recomputed_selected != expected_selected:
        raise ValueError(
            f"Retained-variation rule recomputes selection {recomputed_selected}, "
            f"expected {expected_selected}."
        )

    selected_from_ranking = [
        row for row in ranking_rows if row.selected == 1
    ]
    if len(selected_from_ranking) != 1 or row_key(selected_from_ranking[0]) != expected_selected:
        raise ValueError(
            "Centered ranking does not contain exactly one expected selected row."
        )


def find_legacy_summary(args: CliArgs) -> Path | None:
    if args.legacy_summary is not None:
        if not args.legacy_summary.exists():
            raise FileNotFoundError(
                f"Explicit legacy summary does not exist: {args.legacy_summary}"
            )
        return args.legacy_summary

    candidates = (
        args.experiment_output_dir / LEGACY_SUMMARY_NAME,
        args.source_dir / "kahkm_exp09_vanderpol_tuning" / LEGACY_SUMMARY_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_legacy_summary(
    path: Path | None,
) -> dict[tuple[int, float], dict[int, LegacyHorizonStats]]:
    if path is None:
        return {}

    grouped: dict[tuple[int, float], dict[int, LegacyHorizonStats]] = {}
    for raw in read_csv(path):
        context = f"legacy row in {path.name}"
        c = _finite_int(raw, "n_clusters", context=context)
        omega = _finite_float(raw, "omega", context=context)
        horizon = _finite_int(raw, "horizon", context=context)
        if horizon not in SELECTION_HORIZONS:
            continue
        grouped.setdefault((c, omega), {})[horizon] = LegacyHorizonStats(
            e_mean=_finite_float(
                raw, "relative_association_error_mean", context=context
            ),
            e_std=_finite_float(
                raw, "relative_association_error_std", context=context
            ),
            r2_mean=_finite_float(raw, "association_r2_mean", context=context),
            r2_std=_finite_float(raw, "association_r2_std", context=context),
        )
    return grouped


def build_table_rows(
    finalists: list[CenteredRow],
    legacy: dict[tuple[int, float], dict[int, LegacyHorizonStats]],
) -> list[TableRow]:
    # Preserve centered ranking order in the manuscript table.
    return [
        TableRow(
            centered=row,
            legacy=legacy.get(row_key(row), {}),
        )
        for row in sorted(finalists, key=lambda item: item.centered_rank)
    ]


def full_ranking_csv_rows(
    ranking: list[CenteredRow],
    legacy: dict[tuple[int, float], dict[int, LegacyHorizonStats]],
) -> list[StringRow]:
    return [
        TableRow(centered=row, legacy=legacy.get(row_key(row), {})).to_csv_row()
        for row in sorted(ranking, key=lambda item: item.centered_rank)
    ]


def write_markdown(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        "# Table 4: centered Van der Pol representation selection",
        "",
        "| Rank | C | omega | mean R2 | centered error ± SE | rho_var | r_eff | rho_rank | selected |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        c = row.centered
        lines.append(
            f"| {c.centered_rank} | {c.n_clusters} | {omega_label(c.omega)} | "
            f"{c.mean_multihorizon_r2:.4f} | "
            f"{c.mean_centered_error:.4f} ± {c.centered_error_se:.4f} | "
            f"{c.rho_var_mean:.4f} | {c.effective_rank_mean:.2f} | "
            f"{c.rho_rank_mean:.4f} | {row.selected_label} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Centered Van der Pol representation selection. Configurations "
            r"are ranked by mean centered error over $h\in\{1,10,50,100,200\}$. "
            r"The one-standard-error set uses the standard error of the minimum-error "
            r"configuration; only those finalists are shown. Within this set, normalized "
            r"retained variation $\rho_{\mathrm{var}}$ is the primary representation "
            r"tie-break and normalized effective rank $\rho_{\mathrm{rank}}$ is secondary. "
            r"The selected clean configuration is $(C,\omega)=(25,6)$. Entries for "
            r"centered error are mean $\pm$ standard error over three random states; "
            r"$r_{\mathrm{eff}}$ is the entropy effective rank of held-out target "
            r"associations.}"
        ),
        r"\label{tab:vanderpol_tuning}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rrrrrrrrc}",
        r"\toprule",
        (
            r"Rank & $C$ & $\omega$ & $\overline{R^2}$ & "
            r"centered error $\pm$ SE & $\rho_{\mathrm{var}}$ & "
            r"$r_{\mathrm{eff}}$ & $\rho_{\mathrm{rank}}$ & Selected \\"
        ),
        r"\midrule",
    ]

    for row in rows:
        c = row.centered
        lines.append(
            f"{c.centered_rank} & {c.n_clusters} & {omega_label(c.omega)} & "
            f"${c.mean_multihorizon_r2:.4f}$ & "
            f"${c.mean_centered_error:.4f}\\pm{c.centered_error_se:.4f}$ & "
            f"${c.rho_var_mean:.4f}$ & ${c.effective_rank_mean:.2f}$ & "
            f"${c.rho_rank_mean:.4f}$ & {row.selected_label} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    path: Path,
    args: CliArgs,
    ranking: list[CenteredRow],
    finalists: list[CenteredRow],
    selected: CenteredRow,
    legacy_summary: Path | None,
) -> None:
    payload = {
        "table": "Table 4",
        "description": "Centered one-SE and retained-variation Van der Pol representation selection.",
        "selection_rule": {
            "primary_objective": "mean centered error mean_h(1 - R_h^2)",
            "selection_horizons": list(SELECTION_HORIZONS),
            "one_se_rule": (
                "one-SE limit = minimum mean centered error + SE of minimum-error configuration"
            ),
            "within_one_se_primary_tiebreak": "largest normalized retained variation rho_var",
            "within_one_se_secondary_tiebreak": "largest normalized effective rank rho_rank",
        },
        "expected_selected": {
            "C": args.expected_selected_c,
            "omega": args.expected_selected_omega,
        },
        "selected": asdict(selected),
        "one_se_finalists": [asdict(row) for row in finalists],
        "full_centered_ranking_rows": len(ranking),
        "grid": {
            "clusters": list(args.clusters),
            "omegas": list(args.omegas),
            "random_states": list(args.random_states),
        },
        "inputs": {
            "centered_ranking": str(
                args.experiment_output_dir / CENTERED_RANKING_NAME
            ),
            "one_se_finalists": str(
                args.experiment_output_dir / FINALISTS_NAME
            ),
            "selected_config": str(
                args.experiment_output_dir / SELECTED_NAME
            ),
            "legacy_multistep_summary": (
                None if legacy_summary is None else str(legacy_summary)
            ),
        },
        "legacy_metrics_role": (
            "provenance only; legacy E_h values do not enter the centered one-SE/"
            "retained-variation selection rule"
        ),
        "runner": str(args.runner_script),
        "experiment_output_dir": str(args.experiment_output_dir),
        "run_experiment": args.run_experiment,
        "force_rerun": args.force_rerun,
        "experiment_parameters": {
            "n_steps": args.n_steps,
            "dt": args.dt,
            "mu": args.mu,
            "train_fraction": args.train_fraction,
            "tau": args.tau,
            "subspace_dim": args.subspace_dim,
            "Nb": args.nb,
            "beta": args.beta,
            "nlms_epochs": args.nlms_epochs,
            "kmeans_kind": args.kmeans_kind,
            "batch_size": args.batch_size,
            "n_jobs": args.n_jobs,
            "max_train_per_cluster": args.max_train_per_cluster,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = resolve_args(parser.parse_args())

    if args.run_experiment:
        run_underlying_experiment(args)

    ranking_path = args.experiment_output_dir / CENTERED_RANKING_NAME
    finalists_path = args.experiment_output_dir / FINALISTS_NAME
    selected_path = args.experiment_output_dir / SELECTED_NAME

    ranking = [parse_centered_row(row) for row in read_csv(ranking_path)]
    finalists = [
        parse_centered_row(row, finalist=True)
        for row in read_csv(finalists_path)
    ]
    selected_rows = [
        parse_centered_row(row)
        for row in read_csv(selected_path)
    ]

    validate_centered_outputs(ranking, finalists, selected_rows, args)
    selected = selected_rows[0]

    legacy_summary_path = find_legacy_summary(args)
    legacy = load_legacy_summary(legacy_summary_path)
    table_rows = build_table_rows(finalists, legacy)

    args.table_output_dir.mkdir(parents=True, exist_ok=True)
    values_path = args.table_output_dir / "table4_vanderpol_values.csv"
    md_path = args.table_output_dir / "table4_vanderpol_values.md"
    tex_path = args.table_output_dir / "table4_vanderpol_values.tex"
    full_ranking_path = (
        args.table_output_dir / "table4_vanderpol_full_centered_ranking.csv"
    )
    metadata_path = args.table_output_dir / "table4_vanderpol_metadata.json"

    write_csv(values_path, [row.to_csv_row() for row in table_rows])
    write_csv(full_ranking_path, full_ranking_csv_rows(ranking, legacy))
    write_markdown(md_path, table_rows)
    write_latex(tex_path, table_rows)
    write_metadata(
        metadata_path,
        args,
        ranking,
        finalists,
        selected,
        legacy_summary_path,
    )

    best = min(ranking, key=lambda row: row.mean_centered_error)
    print(
        f"Centered best: C={best.n_clusters}, omega={best.omega:g}, "
        f"mean error={best.mean_centered_error:.6f}, "
        f"SE={best.centered_error_se:.6f}, "
        f"one-SE limit={best.one_se_limit:.6f}."
    )
    print(
        f"Retained-variation selected: C={selected.n_clusters}, "
        f"omega={selected.omega:g}, rho_var={selected.rho_var_mean:.6f}, "
        f"rho_rank={selected.rho_rank_mean:.6f}."
    )
    print("Wrote:")
    for output in (
        values_path,
        md_path,
        tex_path,
        full_ranking_path,
        metadata_path,
    ):
        print(f"  {output}")
    if legacy_summary_path is None:
        print(
            "Legacy Experiment 09 multistep summary was not found; "
            "legacy E_h provenance columns were left blank."
        )
    else:
        print(f"Legacy E_h provenance: {legacy_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
