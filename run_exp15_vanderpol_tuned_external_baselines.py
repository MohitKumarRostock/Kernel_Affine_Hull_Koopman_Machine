#!/usr/bin/env python3
"""Run Experiment 15 external Koopman baselines with tuned Van der Pol KAHKM settings.

Fixed version
-------------
This version fixes the ModuleNotFoundError that occurs when the temporary patched
Experiment 15 script is executed from inside the output directory. It injects the
project root into sys.path before the patched script imports local modules such as:

    kernel_affine_hull_koopman_machines.py

Purpose
-------
This runner reruns:
    experiment_15_external_koopman_baselines.py

with the centered noise-aware Van der Pol KAHKM configuration:
    C = 25, omega = 4

The original Experiment 15 file is NOT modified.

Place this file in the same directory as:
- experiment_15_external_koopman_baselines.py
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Default run:
    python3 run_exp15_vanderpol_tuned_external_baselines_fixed.py

Dry run:
    python3 run_exp15_vanderpol_tuned_external_baselines_fixed.py --dry-run

Outputs:
- kahkm_exp15_vanderpol_retvar_external_baselines/
- experiment_15_tuned_vanderpol_table_values.csv
- experiment_15_tuned_vanderpol_method_ranking.csv
  (legacy uncentered-error ranking, preserved for provenance)
- experiment_15_tuned_vanderpol_centered_method_ranking.csv
- kahkm_exp15_vanderpol_retvar_external_baselines_results.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, TypeAlias


CsvCell: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvCell]
RawCsvRow: TypeAlias = dict[str, str]


@dataclass(frozen=True)
class RunnerConfig:
    python_executable: str
    script: str
    output_dir: str
    zip_name: str
    systems: tuple[str, ...]
    train_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    n_steps_train: int
    n_steps_test: int
    horizons: tuple[int, ...]
    subspace_dim: int
    nb: int
    beta: float
    nlms_epochs: int
    batch_size: int
    n_jobs: int
    kmeans_kind: str
    max_train_per_cluster: int | None
    ridge: float
    vanderpol_c: int
    vanderpol_omega: float
    dry_run: bool
    skip_existing: bool


def _parse_int_tuple(values: Sequence[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def _parse_str_tuple(values: Sequence[str]) -> tuple[str, ...]:
    parsed = tuple(str(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return parsed


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


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(
        description="Run tuned Van der Pol Experiment 15 external Koopman baselines."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--script", default="experiment_15_external_koopman_baselines.py")
    parser.add_argument("--output-dir", default="kahkm_exp15_vanderpol_retvar_external_baselines")
    parser.add_argument("--zip-name", default="kahkm_exp15_vanderpol_retvar_external_baselines_results")

    parser.add_argument("--systems", nargs="+", default=["vanderpol"])
    parser.add_argument("--train-seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--test-seeds", nargs="+", default=["100", "101", "102"])
    parser.add_argument("--n-steps-train", type=int, default=1200)
    parser.add_argument("--n-steps-test", type=int, default=800)
    parser.add_argument("--horizons", nargs="+", default=["1", "10", "50", "100", "200"])

    parser.add_argument("--subspace-dim", type=int, default=4)
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--kmeans-kind", default="full", choices=["auto", "full", "minibatch"])
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-8)

    parser.add_argument("--vanderpol-c", type=int, default=25)
    parser.add_argument("--vanderpol-omega", type=float, default=4.0)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    ns = parser.parse_args()
    max_train_per_cluster = (
        None if int(ns.max_train_per_cluster) <= 0 else int(ns.max_train_per_cluster)
    )

    return RunnerConfig(
        python_executable=str(ns.python),
        script=str(ns.script),
        output_dir=str(ns.output_dir),
        zip_name=str(ns.zip_name),
        systems=_parse_str_tuple([str(value) for value in ns.systems]),
        train_seeds=_parse_int_tuple([str(value) for value in ns.train_seeds]),
        test_seeds=_parse_int_tuple([str(value) for value in ns.test_seeds]),
        n_steps_train=int(ns.n_steps_train),
        n_steps_test=int(ns.n_steps_test),
        horizons=_parse_int_tuple([str(value) for value in ns.horizons]),
        subspace_dim=int(ns.subspace_dim),
        nb=int(ns.nb),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        kmeans_kind=str(ns.kmeans_kind),
        max_train_per_cluster=max_train_per_cluster,
        ridge=float(ns.ridge),
        vanderpol_c=int(ns.vanderpol_c),
        vanderpol_omega=float(ns.vanderpol_omega),
        dry_run=bool(ns.dry_run),
        skip_existing=bool(ns.skip_existing),
    )


def make_patched_script(
    *,
    source_script: Path,
    output_dir: Path,
    project_root: Path,
    vanderpol_c: int,
    vanderpol_omega: float,
) -> Path:
    text = source_script.read_text(encoding="utf-8")

    # Inject project root before the script's local imports.
    injection = (
        "import sys\n"
        f"PROJECT_ROOT_FOR_PATCHED_RUN = {str(project_root.resolve())!r}\n"
        "if PROJECT_ROOT_FOR_PATCHED_RUN not in sys.path:\n"
        "    sys.path.insert(0, PROJECT_ROOT_FOR_PATCHED_RUN)\n"
    )

    marker = "from kernel_affine_hull_koopman_machines import"
    if marker in text and "PROJECT_ROOT_FOR_PATCHED_RUN" not in text:
        text = text.replace(marker, injection + "\n" + marker, 1)

    old = (
        '"vanderpol": SystemConfig(name="vanderpol", dt=0.02, '
        "n_clusters=20, omega=4.0, tau=1e-6),"
    )
    new = (
        '"vanderpol": SystemConfig(name="vanderpol", dt=0.02, '
        f"n_clusters={int(vanderpol_c)}, omega={float(vanderpol_omega)!r}, tau=1e-6),"
    )

    if old not in text:
        lines = text.splitlines()
        replaced = False
        new_lines: list[str] = []
        for line in lines:
            if (
                '"vanderpol": SystemConfig(' in line
                and "n_clusters=" in line
                and "omega=" in line
            ):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(
                    indent
                    + '"vanderpol": SystemConfig(name="vanderpol", dt=0.02, '
                    + f"n_clusters={int(vanderpol_c)}, omega={float(vanderpol_omega)!r}, tau=1e-6),"
                )
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            raise RuntimeError(
                "Could not locate the Van der Pol SystemConfig line in Experiment 15."
            )
        text = "\n".join(new_lines) + "\n"
    else:
        text = text.replace(old, new)

    patched_path = output_dir / "_patched_experiment_15_vanderpol_tuned.py"
    patched_path.write_text(text, encoding="utf-8")
    return patched_path


def output_complete(output_dir: Path) -> bool:
    required = [
        "experiment_15_fit_summary.csv",
        "experiment_15_test_raw_results.csv",
        "experiment_15_test_summary.csv",
        "experiment_15_metadata.json",
    ]
    return all((output_dir / name).exists() for name in required)


def build_command(config: RunnerConfig, patched_script: Path) -> list[str]:
    command = [
        config.python_executable,
        str(patched_script),
        "--systems",
        *list(config.systems),
        "--train-seeds",
        *[str(seed) for seed in config.train_seeds],
        "--test-seeds",
        *[str(seed) for seed in config.test_seeds],
        "--n-steps-train",
        str(config.n_steps_train),
        "--n-steps-test",
        str(config.n_steps_test),
        "--horizons",
        *[str(horizon) for horizon in config.horizons],
        "--subspace-dim",
        str(config.subspace_dim),
        "--nb",
        str(config.nb),
        "--beta",
        str(config.beta),
        "--nlms-epochs",
        str(config.nlms_epochs),
        "--batch-size",
        str(config.batch_size),
        "--n-jobs",
        str(config.n_jobs),
        "--kmeans-kind",
        str(config.kmeans_kind),
        "--ridge",
        str(config.ridge),
        "--output-dir",
        str(config.output_dir),
    ]
    if config.max_train_per_cluster is not None:
        command.extend(["--max-train-per-cluster", str(config.max_train_per_cluster)])
    return command


def run_command(command: Sequence[str], *, dry_run: bool, project_root: Path) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print("\n" + "=" * 96)
    print(printable)
    print("=" * 96, flush=True)
    if dry_run:
        return

    env = dict(os.environ)
    project_root_text = str(project_root.resolve())
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        project_root_text if not old_pythonpath else project_root_text + os.pathsep + old_pythonpath
    )

    subprocess.run(list(command), check=True, cwd=str(project_root.resolve()), env=env)


def summarize_for_manuscript(output_dir: Path) -> None:
    summary_rows = _read_csv(output_dir / "experiment_15_test_summary.csv")
    fit_rows = _read_csv(output_dir / "experiment_15_fit_summary.csv")

    if not summary_rows:
        raise RuntimeError("No Experiment 15 test summary rows found.")

    wanted_methods = [
        "kahkm_nlms",
        "state_dmd",
        "state_edmd_poly2",
        "state_edmd_poly3",
    ]

    table_rows: list[CsvRow] = []
    for method in wanted_methods:
        h1 = next(
            (
                row for row in summary_rows
                if _cell(row, "system") == "vanderpol"
                and _cell(row, "method") == method
                and _int_or_zero(_cell(row, "horizon")) == 1
            ),
            None,
        )
        h50 = next(
            (
                row for row in summary_rows
                if _cell(row, "system") == "vanderpol"
                and _cell(row, "method") == method
                and _int_or_zero(_cell(row, "horizon")) == 50
            ),
            None,
        )
        h200 = next(
            (
                row for row in summary_rows
                if _cell(row, "system") == "vanderpol"
                and _cell(row, "method") == method
                and _int_or_zero(_cell(row, "horizon")) == 200
            ),
            None,
        )
        if h1 is None and h50 is None and h200 is None:
            continue

        table_rows.append(
            {
                "system": "vanderpol",
                "method": method,
                "E1_mean": _float_or_nan(_cell(h1, "relative_association_error_mean")) if h1 else math.nan,
                "E1_std": _float_or_nan(_cell(h1, "relative_association_error_std")) if h1 else math.nan,
                "R2_1_mean": _float_or_nan(_cell(h1, "association_r2_mean")) if h1 else math.nan,
                "R2_1_std": _float_or_nan(_cell(h1, "association_r2_std")) if h1 else math.nan,
                "E50_mean": _float_or_nan(_cell(h50, "relative_association_error_mean")) if h50 else math.nan,
                "E50_std": _float_or_nan(_cell(h50, "relative_association_error_std")) if h50 else math.nan,
                "R2_50_mean": _float_or_nan(_cell(h50, "association_r2_mean")) if h50 else math.nan,
                "R2_50_std": _float_or_nan(_cell(h50, "association_r2_std")) if h50 else math.nan,
                "E200_mean": _float_or_nan(_cell(h200, "relative_association_error_mean")) if h200 else math.nan,
                "E200_std": _float_or_nan(_cell(h200, "relative_association_error_std")) if h200 else math.nan,
                "R2_200_mean": _float_or_nan(_cell(h200, "association_r2_mean")) if h200 else math.nan,
                "R2_200_std": _float_or_nan(_cell(h200, "association_r2_std")) if h200 else math.nan,
            }
        )

    _write_csv(output_dir / "experiment_15_tuned_vanderpol_table_values.csv", table_rows)

    ranking_rows: list[CsvRow] = []
    for horizon in (1, 10, 50, 100, 200):
        rows_for_h = [
            row for row in summary_rows
            if _cell(row, "system") == "vanderpol"
            and _int_or_zero(_cell(row, "horizon")) == horizon
        ]
        rows_for_h.sort(
            key=lambda row: _float_or_nan(_cell(row, "relative_association_error_mean"))
        )
        for rank, row in enumerate(rows_for_h, start=1):
            ranking_rows.append(
                {
                    "horizon": horizon,
                    "rank": rank,
                    "method": _cell(row, "method"),
                    "relative_association_error_mean": _float_or_nan(
                        _cell(row, "relative_association_error_mean")
                    ),
                    "relative_association_error_std": _float_or_nan(
                        _cell(row, "relative_association_error_std")
                    ),
                    "association_r2_mean": _float_or_nan(_cell(row, "association_r2_mean")),
                    "association_r2_std": _float_or_nan(_cell(row, "association_r2_std")),
                }
            )

    _write_csv(output_dir / "experiment_15_tuned_vanderpol_method_ranking.csv", ranking_rows)

    # Centered ranking: higher association R^2 is better. This avoids ranking
    # methods by an uncentered target norm that can favor low-contrast
    # association representations. The legacy relative-error ranking above is
    # intentionally preserved for provenance.
    centered_ranking_rows: list[CsvRow] = []
    for horizon in (1, 10, 50, 100, 200):
        rows_for_h = [
            row
            for row in summary_rows
            if _cell(row, "system") == "vanderpol"
            and _int_or_zero(_cell(row, "horizon")) == horizon
        ]

        rows_for_h = [
            row
            for row in rows_for_h
            if math.isfinite(
                _float_or_nan(_cell(row, "association_r2_mean"))
            )
        ]

        rows_for_h.sort(
            key=lambda row: _float_or_nan(
                _cell(row, "association_r2_mean")
            ),
            reverse=True,
        )

        for rank, row in enumerate(rows_for_h, start=1):
            r2_mean = _float_or_nan(
                _cell(row, "association_r2_mean")
            )
            centered_ranking_rows.append(
                {
                    "horizon": horizon,
                    "rank": rank,
                    "method": _cell(row, "method"),
                    "association_r2_mean": r2_mean,
                    "association_r2_std": _float_or_nan(
                        _cell(row, "association_r2_std")
                    ),
                    "centered_error_mean": 1.0 - r2_mean,
                    "relative_association_error_mean": _float_or_nan(
                        _cell(row, "relative_association_error_mean")
                    ),
                    "relative_association_error_std": _float_or_nan(
                        _cell(row, "relative_association_error_std")
                    ),
                }
            )

    _write_csv(
        output_dir
        / "experiment_15_tuned_vanderpol_centered_method_ranking.csv",
        centered_ranking_rows,
    )

    fit_summary: list[CsvRow] = []
    for row in fit_rows:
        if _cell(row, "system") != "vanderpol":
            continue
        fit_summary.append(
            {
                "system": _cell(row, "system"),
                "train_seeds_used": _cell(row, "train_seeds_used"),
                "n_train_snapshots": _int_or_zero(_cell(row, "n_train_snapshots")),
                "n_clusters": _int_or_zero(_cell(row, "n_clusters")),
                "omega": _float_or_nan(_cell(row, "omega")),
                "kahkm_train_closure_error": _float_or_nan(
                    _cell(row, "kahkm_train_closure_error")
                ),
                "kahkm_train_association_r2": _float_or_nan(
                    _cell(row, "kahkm_train_association_r2")
                ),
                "kahkm_fit_seconds": _float_or_nan(_cell(row, "kahkm_fit_seconds")),
                "baseline_fit_seconds": _float_or_nan(_cell(row, "baseline_fit_seconds")),
            }
        )
    _write_csv(output_dir / "experiment_15_tuned_vanderpol_fit_table.csv", fit_summary)


def make_zip(output_dir: Path, zip_name: str) -> Path:
    zip_base = output_dir.parent / zip_name
    zip_path = Path(str(zip_base) + ".zip")
    if zip_path.exists():
        zip_path.unlink()

    staging_dir = output_dir.parent / f"{zip_name}_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(output_dir, staging_dir / output_dir.name)
    shutil.make_archive(str(zip_base), "zip", root_dir=staging_dir)
    shutil.rmtree(staging_dir)
    return zip_path


def main() -> None:
    config = parse_args()
    source_script = Path(config.script).resolve()
    if not source_script.exists():
        raise FileNotFoundError(
            f"Could not find {source_script}. Run this file from the directory containing "
            "experiment_15_external_koopman_baselines.py, or pass --script."
        )

    project_root = source_script.parent
    output_dir = (project_root / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    patched_script = make_patched_script(
        source_script=source_script,
        output_dir=output_dir,
        project_root=project_root,
        vanderpol_c=config.vanderpol_c,
        vanderpol_omega=config.vanderpol_omega,
    )

    command = build_command(
        RunnerConfig(
            **{**asdict(config), "script": str(source_script), "output_dir": str(output_dir)}
        ),
        patched_script,
    )

    start = time.time()
    if config.skip_existing and output_complete(output_dir):
        print(f"Skipping child run because Experiment 15 outputs already exist in {output_dir}")
    else:
        run_command(command, dry_run=config.dry_run, project_root=project_root)

    if config.dry_run:
        print("\nDry run finished. No experiments were executed.")
        return

    if not output_complete(output_dir):
        raise RuntimeError(
            f"Experiment 15 output is incomplete in {output_dir}. "
            "Expected fit, raw, summary, and metadata CSV/JSON files."
        )

    summarize_for_manuscript(output_dir)

    runner_metadata = {
        "runner": Path(__file__).name,
        "purpose": (
            "Rerun external Koopman baselines with the retained-variation/"
            "noise-aware Van der Pol KAHKM configuration C=25, omega=4."
        ),
        "runner_config": asdict(config),
        "project_root": str(project_root),
        "base_experiment_script": config.script,
        "runtime_patch": {
            "deterministic": True,
            "transient_output": str(patched_script),
            "inject_project_root_for_local_imports": True,
            "vanderpol_n_clusters": config.vanderpol_c,
            "vanderpol_omega": config.vanderpol_omega,
            "note": (
                "The transient patched script is generated deterministically "
                "from the committed base experiment by this committed runner "
                "and is not required as a tracked reproducibility artifact."
            ),
        },
        "total_seconds_runner_including_child": time.time() - start,
    }
    with (output_dir / "experiment_15_tuned_runner_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(runner_metadata, handle, indent=2)

    zip_path = make_zip(output_dir, zip_name=config.zip_name)

    print("\nFinished tuned Van der Pol external-baseline run.")
    print(f"Output directory: {output_dir}")
    print(f"Table values: {output_dir / 'experiment_15_tuned_vanderpol_table_values.csv'}")
    print(f"Legacy method ranking: {output_dir / 'experiment_15_tuned_vanderpol_method_ranking.csv'}")
    print(f"Centered method ranking: {output_dir / 'experiment_15_tuned_vanderpol_centered_method_ranking.csv'}")
    print(f"Created ZIP: {zip_path.resolve()}")
    print("Upload the ZIP for manuscript updating.")


if __name__ == "__main__":
    main()
