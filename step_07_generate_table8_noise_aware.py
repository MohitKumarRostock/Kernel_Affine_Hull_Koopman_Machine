#!/usr/bin/env python3
"""Generate manuscript Table 8: centered noise-aware Van der Pol tuning.

The manuscript-facing table reports centered association R^2 for the
retained-variation-selected noise-aware Van der Pol configuration, together
with the per-noise weighted centered score used by Experiment 12.

Selection provenance
--------------------
The selected configuration is read from:

    experiment_12_centered_retvar_selected_config.csv

The default expected selection is (C, omega) = (25, 4). Experiment 12 uses
weights 0.25, 0.25, and 0.50 on centered errors (1 - R^2) at horizons
50, 100, and 200, respectively, then averages the per-seed score across the
training-noise levels.

The printed table contains:
    training noise, R^2_50, R^2_100, R^2_200, S_noise

where

    S_noise = 0.25(1-R^2_50) + 0.25(1-R^2_100) + 0.50(1-R^2_200).

Machine-readable CSV/metadata also retain the legacy E_h statistics and the
retained-variation/effective-rank diagnostics.

Use --no-run to format already-computed Experiment 12 outputs.
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


HORIZONS: tuple[int, int, int] = (50, 100, 200)
EXPECTED_WEIGHTS: dict[int, float] = {50: 0.25, 100: 0.25, 200: 0.50}
EXPECTED_C = 25
EXPECTED_OMEGA = 4.0


@dataclass(frozen=True)
class MetricCell:
    mean: float
    std: float

    def table_cell(self, digits: int = 4) -> str:
        return f"${self.mean:.{digits}f}\\pm{self.std:.{digits}f}$"


@dataclass(frozen=True)
class RetainedVariationDiagnostics:
    rho_var_mean: float
    rho_var_std: float
    effective_rank_mean: float
    effective_rank_std: float
    rho_rank_mean: float
    rho_rank_std: float


@dataclass(frozen=True)
class SelectedConfig:
    n_clusters: int
    omega: float
    source: str
    centered_rank: int
    n_seeds: int
    n_noise_levels_per_seed: int
    centered_robust_score_mean: float
    centered_robust_score_sd: float
    centered_robust_score_se: float
    mean_robust_multihorizon_r2: float
    selection_weights: str
    inside_centered_one_se_set: int
    selected_by_centered_retvar_rule: int
    aggregate_retvar: RetainedVariationDiagnostics


@dataclass(frozen=True)
class TableRow:
    noise_level: float
    r2_50: MetricCell
    r2_100: MetricCell
    r2_200: MetricCell
    weighted_centered_score: float

    e50: MetricCell
    e100: MetricCell
    e200: MetricCell

    retvar: RetainedVariationDiagnostics

    def as_csv_row(self) -> dict[str, str]:
        return {
            "training_noise": noise_label(self.noise_level),
            "noise_level": format_float(self.noise_level),

            "R2_50_mean": format_float(self.r2_50.mean),
            "R2_50_std": format_float(self.r2_50.std),
            "R2_100_mean": format_float(self.r2_100.mean),
            "R2_100_std": format_float(self.r2_100.std),
            "R2_200_mean": format_float(self.r2_200.mean),
            "R2_200_std": format_float(self.r2_200.std),
            "weighted_centered_score": format_float(self.weighted_centered_score),

            "R2_50_table_cell": self.r2_50.table_cell(),
            "R2_100_table_cell": self.r2_100.table_cell(),
            "R2_200_table_cell": self.r2_200.table_cell(),

            # Legacy uncentered relative association error, retained for provenance.
            "E50_mean": format_float(self.e50.mean),
            "E50_std": format_float(self.e50.std),
            "E100_mean": format_float(self.e100.mean),
            "E100_std": format_float(self.e100.std),
            "E200_mean": format_float(self.e200.mean),
            "E200_std": format_float(self.e200.std),

            # Retained-variation diagnostics for the clean held-out target
            # association representation under this training-noise fit.
            "rho_var_mean": format_float(self.retvar.rho_var_mean),
            "rho_var_std": format_float(self.retvar.rho_var_std),
            "effective_rank_mean": format_float(self.retvar.effective_rank_mean),
            "effective_rank_std": format_float(self.retvar.effective_rank_std),
            "rho_rank_mean": format_float(self.retvar.rho_rank_mean),
            "rho_rank_std": format_float(self.retvar.rho_rank_std),
        }


@dataclass(frozen=True)
class CliArgs:
    experiment_dir: Path
    experiment_output_dir: Path
    output_dir: Path
    no_run: bool
    force: bool
    python_executable: str
    expected_n_clusters: int
    expected_omega: float


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate centered manuscript Table 8 from Experiment 12 outputs."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("."),
        help="Directory containing run_exp12_vanderpol_noise_aware_tuning.py.",
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=Path("kahkm_exp12_retvar_C25_w4"),
        help=(
            "Selected Experiment 12 retained-variation output directory. "
            "Default: kahkm_exp12_retvar_C25_w4."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kahkm_table8_centered_noise_aware"),
        help="Directory for generated centered Table 8 files.",
    )
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--expected-n-clusters", type=int, default=EXPECTED_C)
    parser.add_argument("--expected-omega", type=float, default=EXPECTED_OMEGA)

    ns = parser.parse_args(argv)
    experiment_dir = Path(ns.experiment_dir).resolve()
    experiment_output_dir = Path(ns.experiment_output_dir)
    if not experiment_output_dir.is_absolute():
        experiment_output_dir = experiment_dir / experiment_output_dir
    output_dir = Path(ns.output_dir)
    if not output_dir.is_absolute():
        output_dir = experiment_dir / output_dir

    return CliArgs(
        experiment_dir=experiment_dir,
        experiment_output_dir=experiment_output_dir,
        output_dir=output_dir,
        no_run=bool(ns.no_run),
        force=bool(ns.force),
        python_executable=str(ns.python_executable),
        expected_n_clusters=int(ns.expected_n_clusters),
        expected_omega=float(ns.expected_omega),
    )


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.17g}"


def finite_float(row: dict[str, str], key: str, *, context: str) -> float:
    text = row.get(key, "").strip()
    if not text:
        raise ValueError(f"Missing {key!r} for {context}. Available columns: {sorted(row)}")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} for {context}: {text!r}")
    return value


def finite_int(row: dict[str, str], key: str, *, context: str) -> int:
    return int(round(finite_float(row, key, context=context)))


def noise_label(noise_level: float) -> str:
    percent = 100.0 * noise_level
    if abs(percent - round(percent)) < 1e-12:
        return f"{int(round(percent))}%"
    return f"{percent:.2f}".rstrip("0").rstrip(".") + "%"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def run_or_reuse_experiment(args: CliArgs) -> None:
    summary_path = (
        args.experiment_output_dir
        / "experiment_12_noise_tuning_summary_by_config_noise_horizon.csv"
    )
    selection_path = (
        args.experiment_output_dir
        / "experiment_12_centered_retvar_selected_config.csv"
    )

    if args.no_run:
        require_file(summary_path)
        require_file(selection_path)
        return

    script_path = args.experiment_dir / "run_exp12_vanderpol_noise_aware_tuning.py"
    require_file(script_path)

    if summary_path.exists() and selection_path.exists() and not args.force:
        return

    command = [
        args.python_executable,
        str(script_path),
        "--output-dir",
        str(args.experiment_output_dir),
    ]
    subprocess.run(command, cwd=str(args.experiment_dir), check=True)


def parse_selection_weights(text: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        h_text, weight_text = part.split(":", maxsplit=1)
        result[int(h_text)] = float(weight_text)
    return result


def load_selected_config(path: Path, args: CliArgs) -> SelectedConfig:
    rows = read_csv_rows(path)
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one selected configuration in {path}, found {len(rows)}."
        )
    row = rows[0]
    context = "Experiment 12 centered retained-variation selection"

    n_clusters = finite_int(row, "n_clusters", context=context)
    omega = finite_float(row, "omega", context=context)

    if n_clusters != args.expected_n_clusters or not math.isclose(
        omega, args.expected_omega, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Selected Experiment 12 configuration mismatch: expected "
            f"(C, omega)=({args.expected_n_clusters}, {args.expected_omega:g}), "
            f"observed ({n_clusters}, {omega:g})."
        )

    selected_flag = finite_int(row, "selected_by_centered_retvar_rule", context=context)
    inside_flag = finite_int(row, "inside_centered_one_se_set", context=context)
    if selected_flag != 1 or inside_flag != 1:
        raise ValueError(
            "Selected-config row must be inside the centered one-SE set and marked "
            "selected_by_centered_retvar_rule=1."
        )

    selection_weights = row.get("selection_weights", "").strip()
    parsed_weights = parse_selection_weights(selection_weights)
    if set(parsed_weights) != set(EXPECTED_WEIGHTS):
        raise ValueError(
            f"Unexpected selection horizons/weights in {path}: {selection_weights!r}"
        )
    for horizon, expected in EXPECTED_WEIGHTS.items():
        if not math.isclose(
            parsed_weights[horizon], expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Unexpected weight for h={horizon}: "
                f"{parsed_weights[horizon]} vs expected {expected}"
            )

    aggregate_retvar = RetainedVariationDiagnostics(
        rho_var_mean=finite_float(
            row, "target_normalized_association_variation_mean", context=context
        ),
        rho_var_std=finite_float(
            row, "target_normalized_association_variation_std", context=context
        ),
        effective_rank_mean=finite_float(
            row, "target_association_effective_rank_mean", context=context
        ),
        effective_rank_std=finite_float(
            row, "target_association_effective_rank_std", context=context
        ),
        rho_rank_mean=finite_float(
            row, "target_normalized_association_effective_rank_mean", context=context
        ),
        rho_rank_std=finite_float(
            row, "target_normalized_association_effective_rank_std", context=context
        ),
    )

    return SelectedConfig(
        n_clusters=n_clusters,
        omega=omega,
        source="experiment_12_centered_retvar_selected_config.csv",
        centered_rank=finite_int(row, "centered_rank", context=context),
        n_seeds=finite_int(row, "n_seeds", context=context),
        n_noise_levels_per_seed=finite_int(
            row, "n_noise_levels_per_seed", context=context
        ),
        centered_robust_score_mean=finite_float(
            row, "centered_robust_score_mean", context=context
        ),
        centered_robust_score_sd=finite_float(
            row, "centered_robust_score_sd", context=context
        ),
        centered_robust_score_se=finite_float(
            row, "centered_robust_score_se", context=context
        ),
        mean_robust_multihorizon_r2=finite_float(
            row, "mean_robust_multihorizon_r2", context=context
        ),
        selection_weights=selection_weights,
        inside_centered_one_se_set=inside_flag,
        selected_by_centered_retvar_rule=selected_flag,
        aggregate_retvar=aggregate_retvar,
    )


def _retvar_from_summary_row(
    row: dict[str, str], *, context: str
) -> RetainedVariationDiagnostics:
    return RetainedVariationDiagnostics(
        rho_var_mean=finite_float(
            row, "target_normalized_association_variation_mean", context=context
        ),
        rho_var_std=finite_float(
            row, "target_normalized_association_variation_std", context=context
        ),
        effective_rank_mean=finite_float(
            row, "target_association_effective_rank_mean", context=context
        ),
        effective_rank_std=finite_float(
            row, "target_association_effective_rank_std", context=context
        ),
        rho_rank_mean=finite_float(
            row, "target_normalized_association_effective_rank_mean", context=context
        ),
        rho_rank_std=finite_float(
            row, "target_normalized_association_effective_rank_std", context=context
        ),
    )


def _same_retvar(
    left: RetainedVariationDiagnostics,
    right: RetainedVariationDiagnostics,
    *,
    atol: float = 1e-12,
) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=atol)
        for a, b in zip(
            asdict(left).values(),
            asdict(right).values(),
            strict=True,
        )
    )


def collect_table_rows(summary_path: Path, selected: SelectedConfig) -> list[TableRow]:
    rows = read_csv_rows(summary_path)

    # noise -> horizon -> source row
    grouped: dict[float, dict[int, dict[str, str]]] = {}
    for row in rows:
        context = "Experiment 12 noise/horizon summary"
        n_clusters = finite_int(row, "n_clusters", context=context)
        omega = finite_float(row, "omega", context=context)
        horizon = finite_int(row, "horizon", context=context)

        if n_clusters != selected.n_clusters:
            continue
        if not math.isclose(omega, selected.omega, rel_tol=0.0, abs_tol=1e-12):
            continue
        if horizon not in HORIZONS:
            continue

        noise_level = finite_float(row, "noise_level", context=context)
        grouped.setdefault(noise_level, {})[horizon] = row

    table_rows: list[TableRow] = []
    for noise_level in sorted(grouped):
        horizon_rows = grouped[noise_level]
        missing = [h for h in HORIZONS if h not in horizon_rows]
        if missing:
            raise ValueError(
                f"Missing horizons {missing} for selected configuration at "
                f"noise={noise_level:g}."
            )

        def metric(horizon: int, mean_key: str, std_key: str) -> MetricCell:
            row = horizon_rows[horizon]
            context = f"noise={noise_level:g}, horizon={horizon}"
            return MetricCell(
                mean=finite_float(row, mean_key, context=context),
                std=finite_float(row, std_key, context=context),
            )

        r2_50 = metric(50, "association_r2_mean", "association_r2_std")
        r2_100 = metric(100, "association_r2_mean", "association_r2_std")
        r2_200 = metric(200, "association_r2_mean", "association_r2_std")

        e50 = metric(
            50, "relative_association_error_mean", "relative_association_error_std"
        )
        e100 = metric(
            100, "relative_association_error_mean", "relative_association_error_std"
        )
        e200 = metric(
            200, "relative_association_error_mean", "relative_association_error_std"
        )

        per_horizon_retvar = [
            _retvar_from_summary_row(
                horizon_rows[h],
                context=f"noise={noise_level:g}, horizon={h}",
            )
            for h in HORIZONS
        ]
        retvar = per_horizon_retvar[0]
        if not all(_same_retvar(retvar, other) for other in per_horizon_retvar[1:]):
            raise ValueError(
                "Retained-variation diagnostics differ across horizons for "
                f"noise={noise_level:g}; expected one representation diagnostic per fit."
            )

        score = (
            EXPECTED_WEIGHTS[50] * (1.0 - r2_50.mean)
            + EXPECTED_WEIGHTS[100] * (1.0 - r2_100.mean)
            + EXPECTED_WEIGHTS[200] * (1.0 - r2_200.mean)
        )

        table_rows.append(
            TableRow(
                noise_level=noise_level,
                r2_50=r2_50,
                r2_100=r2_100,
                r2_200=r2_200,
                weighted_centered_score=score,
                e50=e50,
                e100=e100,
                e200=e200,
                retvar=retvar,
            )
        )

    if not table_rows:
        raise ValueError(
            f"No rows found for selected configuration "
            f"C={selected.n_clusters}, omega={selected.omega:g}."
        )

    if len(table_rows) != selected.n_noise_levels_per_seed:
        raise ValueError(
            f"Expected {selected.n_noise_levels_per_seed} training-noise levels from "
            f"the selection record, found {len(table_rows)}."
        )

    # Linearity check: mean of the per-noise mean weighted scores must reproduce
    # the selected centered robust score mean.
    score_mean = sum(row.weighted_centered_score for row in table_rows) / len(table_rows)
    if not math.isclose(
        score_mean,
        selected.centered_robust_score_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Per-noise centered-score means do not reproduce the Experiment 12 "
            f"selection score: {score_mean:.17g} vs "
            f"{selected.centered_robust_score_mean:.17g}."
        )

    return table_rows


def write_csv(path: Path, rows: Sequence[TableRow]) -> None:
    csv_rows = [row.as_csv_row() for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def write_markdown(path: Path, rows: Sequence[TableRow]) -> None:
    lines = [
        "| Training noise | $R^2_{50}$ | $R^2_{100}$ | $R^2_{200}$ | $S_{\\rm noise}$ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {noise_label(row.noise_level)} | "
            f"{row.r2_50.table_cell()} | "
            f"{row.r2_100.table_cell()} | "
            f"{row.r2_200.table_cell()} | "
            f"{row.weighted_centered_score:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(
    path: Path,
    rows: Sequence[TableRow],
    selected: SelectedConfig,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            rf"\caption{{Centered noise-aware Van der Pol evaluation for the "
            rf"retained-variation-selected configuration $(C,\omega)="
            rf"({selected.n_clusters},{selected.omega:g})$. Gaussian observation "
            r"noise is added only to the training snapshots; evaluation uses clean "
            r"held-out trajectories. Entries in the three $R^2$ columns are mean "
            rf"$\pm$ standard deviation over {selected.n_seeds} random states. "
            r"$S_{\mathrm{noise}}=0.25(1-R^2_{50})+0.25(1-R^2_{100})"
            r"+0.50(1-R^2_{200})$ is the noise-specific centered score; the "
            r"selection rule averages the corresponding per-seed score across all "
            r"noise levels and applies the centered one-SE/retained-variation rule.}"
        ),
        r"\label{tab:vdp_noise_tuned}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Training noise & $R^2_{50}$ & $R^2_{100}$ & $R^2_{200}$ & $S_{\mathrm{noise}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{noise_label(row.noise_level)} & "
            f"{row.r2_50.table_cell()} & "
            f"{row.r2_100.table_cell()} & "
            f"{row.r2_200.table_cell()} & "
            f"${row.weighted_centered_score:.4f}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    path: Path,
    args: CliArgs,
    selected: SelectedConfig,
    table_rows: Sequence[TableRow],
    summary_path: Path,
    selection_path: Path,
) -> None:
    payload = {
        "table": "Table 8",
        "description": "Centered noise-aware Van der Pol tuning for the retained-variation-selected configuration.",
        "experiment_script": "run_exp12_vanderpol_noise_aware_tuning.py",
        "experiment_output_dir": str(args.experiment_output_dir),
        "selection": asdict(selected),
        "expected_selection": {
            "C": args.expected_n_clusters,
            "omega": args.expected_omega,
        },
        "selection_weights": EXPECTED_WEIGHTS,
        "printed_metrics": [
            "R2_50_mean_std",
            "R2_100_mean_std",
            "R2_200_mean_std",
            "weighted_centered_score",
        ],
        "provenance_metrics_in_csv": [
            "E50_mean_std",
            "E100_mean_std",
            "E200_mean_std",
            "rho_var_mean_std",
            "effective_rank_mean_std",
            "rho_rank_mean_std",
        ],
        "inputs": {
            "summary_by_config_noise_horizon": str(summary_path),
            "centered_retvar_selected_config": str(selection_path),
        },
        "rows": [asdict(row) for row in table_rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_or_reuse_experiment(args)

    selection_path = (
        args.experiment_output_dir
        / "experiment_12_centered_retvar_selected_config.csv"
    )
    summary_path = (
        args.experiment_output_dir
        / "experiment_12_noise_tuning_summary_by_config_noise_horizon.csv"
    )

    selected = load_selected_config(selection_path, args)
    table_rows = collect_table_rows(summary_path, selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "table8_noise_aware_values.csv"
    md_path = args.output_dir / "table8_noise_aware_values.md"
    tex_path = args.output_dir / "table8_noise_aware_values.tex"
    metadata_path = args.output_dir / "table8_noise_aware_metadata.json"

    write_csv(csv_path, table_rows)
    write_markdown(md_path, table_rows)
    write_latex(tex_path, table_rows, selected)
    write_metadata(
        metadata_path,
        args,
        selected,
        table_rows,
        summary_path,
        selection_path,
    )

    print(
        f"Selected C={selected.n_clusters}, omega={selected.omega:g} "
        f"from {selected.source}."
    )
    print(
        "Centered robust score: "
        f"{selected.centered_robust_score_mean:.6f} "
        f"± {selected.centered_robust_score_sd:.6f} SD; "
        f"mean robust multi-horizon R2="
        f"{selected.mean_robust_multihorizon_r2:.6f}."
    )
    print("Wrote:")
    for output in (csv_path, md_path, tex_path, metadata_path):
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
