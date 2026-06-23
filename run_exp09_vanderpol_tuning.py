#!/usr/bin/env python3
"""Run Van der Pol Experiment 09 tuning for KAHKM.

Place this file in the same directory as:
- experiment_09_vanderpol_sensitivity.py
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Pilot run, one seed:

    python3 run_exp09_vanderpol_tuning.py

Manuscript-grade run, three seeds:

    python3 run_exp09_vanderpol_tuning.py --random-states 0 1 2



Default tuning grid:
- C in {10, 15, 20, 25, 30, 40, 50}
- omega in {0.5, 1, 2, 4, 6, 8, 12}

Default selection objective:
- minimize mean relative association error over horizons {1, 10, 50, 100, 200}

Manuscript-compatible closure settings:
- beta = 0.1
- nlms_epochs = 20
- subspace_dim = 4
- Nb = 100

At the end, this script creates:
- kahkm_exp09_vanderpol_tuning_results.zip
- kahkm_exp09_vanderpol_tuning/experiment_09_selected_config_by_mean_horizon.csv
- kahkm_exp09_vanderpol_tuning/experiment_09_best_by_horizon.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias


CsvCell: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvCell]
RawCsvRow: TypeAlias = dict[str, str]


@dataclass(frozen=True)
class ConfigKey:
    n_clusters: int
    omega: float


def _read_csv(path: Path) -> list[RawCsvRow]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cell(row: RawCsvRow, key: str, default: str = "") -> str:
    return row[key] if key in row else default


def _float_or_nan(value: str | int | float | None) -> float:
    if value is None:
        return math.nan
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return math.nan
        try:
            return float(text)
        except ValueError:
            return math.nan
    return float(value)


def _int_or_zero(value: str | int | float | None) -> int:
    number = _float_or_nan(value)
    if not math.isfinite(number):
        return 0
    return int(number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Van der Pol KAHKM Experiment 09 tuning."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for the child run. Default: current interpreter.",
    )
    parser.add_argument(
        "--script",
        default="experiment_09_vanderpol_sensitivity.py",
        help="Experiment 09 script filename.",
    )
    parser.add_argument(
        "--output-dir",
        default="kahkm_exp09_vanderpol_tuning",
        help="Output directory for Experiment 09 results.",
    )
    parser.add_argument(
        "--zip-name",
        default="kahkm_exp09_vanderpol_tuning_results",
        help="Name of ZIP file without .zip suffix.",
    )

    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.75)

    parser.add_argument(
        "--clusters",
        nargs="+",
        default=["10", "15", "20", "25", "30", "40", "50"],
        help="Cluster grid. Default: 10 15 20 25 30 40 50.",
    )
    parser.add_argument(
        "--omegas",
        nargs="+",
        default=["0.5", "1", "2", "4", "6", "8", "12"],
        help="Omega grid. Default: 0.5 1 2 4 6 8 12.",
    )
    parser.add_argument(
        "--random-states",
        nargs="+",
        default=["0", "1", "2"],
        help="Random states passed to Experiment 09. Default: 0. Use 0 1 2 for manuscript-grade results.",
    )

    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--kmeans-kind", default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["1", "10", "50", "100", "200"],
        help="Horizons passed to Experiment 09. Default: 1 10 50 100 200.",
    )
    parser.add_argument(
        "--selection-horizons",
        nargs="+",
        default=["1", "10", "50", "100", "200"],
        help="Horizons used in the selection objective. Default: 1 10 50 100 200.",
    )
    parser.add_argument(
        "--max-train-per-cluster",
        type=int,
        default=0,
        help="0 omits the flag; otherwise passed to Experiment 09.",
    )
    parser.add_argument("--save-ae-to-disk", action="store_true")
    parser.add_argument("--verbose-fit", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the subprocess if Experiment 09 summary files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing it.",
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print("\n" + "=" * 96)
    print(printable)
    print("=" * 96, flush=True)

    if dry_run:
        return

    subprocess.run(command, check=True)


def build_command(args: argparse.Namespace, script_path: Path, output_dir: Path) -> list[str]:
    command = [
        str(args.python),
        str(script_path),
        "--output-dir",
        str(output_dir),
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
        *[str(value) for value in args.omegas],
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
        str(args.kmeans_kind),
        "--batch-size",
        str(args.batch_size),
        "--n-jobs",
        str(args.n_jobs),
        "--horizons",
        *[str(value) for value in args.horizons],
    ]
    if int(args.max_train_per_cluster) > 0:
        command.extend(["--max-train-per-cluster", str(args.max_train_per_cluster)])
    if bool(args.save_ae_to_disk):
        command.append("--save-ae-to-disk")
    if bool(args.verbose_fit):
        command.append("--verbose-fit")
    return command


def output_complete(output_dir: Path) -> bool:
    required = [
        "experiment_09_one_step_summary.csv",
        "experiment_09_multistep_summary.csv",
        "experiment_09_metadata.json",
    ]
    return all((output_dir / name).exists() for name in required)


def select_best_configs(output_dir: Path, selection_horizons: Sequence[str]) -> None:
    multistep_path = output_dir / "experiment_09_multistep_summary.csv"
    one_step_path = output_dir / "experiment_09_one_step_summary.csv"

    multistep_rows = _read_csv(multistep_path)
    one_step_rows = _read_csv(one_step_path)

    if not multistep_rows:
        raise RuntimeError(f"No multistep rows found in {multistep_path}")

    wanted_horizons = {_int_or_zero(value) for value in selection_horizons}
    grouped_errors: dict[ConfigKey, list[float]] = defaultdict(list)
    grouped_r2: dict[ConfigKey, list[float]] = defaultdict(list)
    grouped_horizon_map: dict[ConfigKey, dict[int, float]] = defaultdict(dict)

    for row in multistep_rows:
        horizon = _int_or_zero(_cell(row, "horizon"))
        if horizon not in wanted_horizons:
            continue

        key = ConfigKey(
            n_clusters=_int_or_zero(_cell(row, "n_clusters")),
            omega=_float_or_nan(_cell(row, "omega")),
        )
        err = _float_or_nan(_cell(row, "relative_association_error_mean"))
        r2 = _float_or_nan(_cell(row, "association_r2_mean"))

        if math.isfinite(err):
            grouped_errors[key].append(err)
            grouped_horizon_map[key][horizon] = err
        if math.isfinite(r2):
            grouped_r2[key].append(r2)

    selection_rows: list[CsvRow] = []
    for key, errors in grouped_errors.items():
        if len(errors) != len(wanted_horizons):
            continue

        horizon_errors = grouped_horizon_map[key]
        selection_rows.append(
            {
                "rank": 0,
                "n_clusters": key.n_clusters,
                "omega": key.omega,
                "selection_objective_mean_error": sum(errors) / len(errors),
                "selection_horizons": " ".join(str(h) for h in sorted(wanted_horizons)),
                "E1": horizon_errors.get(1, math.nan),
                "E10": horizon_errors.get(10, math.nan),
                "E50": horizon_errors.get(50, math.nan),
                "E100": horizon_errors.get(100, math.nan),
                "E200": horizon_errors.get(200, math.nan),
                "mean_association_r2_over_selection_horizons": (
                    sum(grouped_r2[key]) / len(grouped_r2[key]) if grouped_r2[key] else math.nan
                ),
                "n_selection_horizons_present": len(errors),
            }
        )

    selection_rows.sort(key=lambda row: float(row["selection_objective_mean_error"]))
    ranked_rows: list[CsvRow] = []
    for idx, row in enumerate(selection_rows, start=1):
        ranked = dict(row)
        ranked["rank"] = idx
        ranked_rows.append(ranked)

    _write_csv(output_dir / "experiment_09_selected_config_by_mean_horizon.csv", ranked_rows)

    best_by_horizon: list[CsvRow] = []
    for horizon in sorted(wanted_horizons):
        candidates = [
            row for row in multistep_rows
            if _int_or_zero(_cell(row, "horizon")) == horizon
        ]
        candidates.sort(key=lambda row: _float_or_nan(_cell(row, "relative_association_error_mean")))
        if not candidates:
            continue
        best = candidates[0]
        best_by_horizon.append(
            {
                "horizon": horizon,
                "n_clusters": _int_or_zero(_cell(best, "n_clusters")),
                "omega": _float_or_nan(_cell(best, "omega")),
                "relative_association_error_mean": _float_or_nan(_cell(best, "relative_association_error_mean")),
                "relative_association_error_std": _float_or_nan(_cell(best, "relative_association_error_std")),
                "association_r2_mean": _float_or_nan(_cell(best, "association_r2_mean")),
                "association_r2_std": _float_or_nan(_cell(best, "association_r2_std")),
                "n_runs": _int_or_zero(_cell(best, "n_runs")),
            }
        )
    _write_csv(output_dir / "experiment_09_best_by_horizon.csv", best_by_horizon)

    # Attach one-step information for the selected top configuration, if available.
    if ranked_rows:
        top = ranked_rows[0]
        top_c = int(top["n_clusters"])
        top_omega = float(top["omega"])

        one_step_match = next(
            (
                row for row in one_step_rows
                if _int_or_zero(_cell(row, "n_clusters")) == top_c
                and abs(_float_or_nan(_cell(row, "omega")) - top_omega) < 1e-12
            ),
            None,
        )

        top_summary: list[CsvRow] = [dict(top)]
        if one_step_match is not None:
            top_summary[0]["one_step_test_closure_error_mean"] = _float_or_nan(
                _cell(one_step_match, "test_closure_error_mean")
            )
            top_summary[0]["one_step_test_closure_error_std"] = _float_or_nan(
                _cell(one_step_match, "test_closure_error_std")
            )
            top_summary[0]["one_step_test_association_r2_mean"] = _float_or_nan(
                _cell(one_step_match, "test_association_r2_mean")
            )
            top_summary[0]["one_step_test_association_r2_std"] = _float_or_nan(
                _cell(one_step_match, "test_association_r2_std")
            )
        _write_csv(output_dir / "experiment_09_top_selected_config_summary.csv", top_summary)


def make_zip(output_dir: Path, zip_name: str) -> Path:
    zip_base = output_dir.parent / zip_name
    zip_path = Path(str(zip_base) + ".zip")
    if zip_path.exists():
        zip_path.unlink()

    staging_dir = output_dir.parent / f"{zip_name}_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    destination = staging_dir / output_dir.name
    shutil.copytree(output_dir, destination)

    shutil.make_archive(str(zip_base), "zip", root_dir=staging_dir)
    shutil.rmtree(staging_dir)
    return zip_path


def main() -> None:
    args = parse_args()

    script_path = Path(str(args.script))
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find {script_path}. Run this script from the directory containing "
            "experiment_09_vanderpol_sensitivity.py, or pass --script."
        )

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args, script_path=script_path, output_dir=output_dir)

    if args.skip_existing and output_complete(output_dir):
        print(f"Skipping child run because summary files already exist in {output_dir}")
    else:
        run_command(command, dry_run=bool(args.dry_run))

    if args.dry_run:
        print("\nDry run finished. No experiment was executed and no ZIP was created.")
        return

    if not output_complete(output_dir):
        raise RuntimeError(
            "Experiment 09 output is incomplete. Expected summary files were not found in "
            f"{output_dir}"
        )

    select_best_configs(output_dir, selection_horizons=[str(v) for v in args.selection_horizons])
    zip_path = make_zip(output_dir, zip_name=str(args.zip_name))

    print("\nFinished Van der Pol tuning.")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Selected config CSV: {(output_dir / 'experiment_09_selected_config_by_mean_horizon.csv').resolve()}")
    print(f"Best by horizon CSV: {(output_dir / 'experiment_09_best_by_horizon.csv').resolve()}")
    print(f"Top selected summary CSV: {(output_dir / 'experiment_09_top_selected_config_summary.csv').resolve()}")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload the ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
