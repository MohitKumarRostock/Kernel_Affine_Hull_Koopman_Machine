"""
Experiment 16A: AI-facing KAHKM hyperparameter sensitivity for closed-loop Gymnasium tasks.

This wrapper repeatedly runs:

    experiment_16_ai_classic_control_closed_loop_pylance_clean.py

for a grid of KAHKM free parameters:

    C = n_clusters
    omega = association sharpness

It aggregates the KAHKM rows from Experiment 16 and recommends one setting per task.

Why this script exists
----------------------
Previous dynamical-system experiments showed that KAHKM performance depends on C and omega.
For AI-facing Gymnasium experiments, a single default value is not enough for a strong paper.
This script provides a transparent sensitivity/tuning stage before running the final benchmark
and the interpretability/regime-transition analysis.

Required files in the same folder
---------------------------------
- experiment_16_ai_classic_control_closed_loop_pylance_clean.py
- kernel_affine_hull_koopman_machines.py
- parallel_autoencoders.py
- combine_multiple_autoencoders_extended.py

Required package
----------------
pip install "gymnasium[classic-control]"

Typical run
-----------
python experiment_16a_ai_hyperparameter_sensitivity_pylance_clean.py

Fast smoke test
---------------
python experiment_16a_ai_hyperparameter_sensitivity_pylance_clean.py \
    --tasks CartPole-v1 \
    --clusters 10 15 \
    --omegas 4 8 \
    --train-episodes 4 \
    --test-episodes 2 \
    --max-steps 120 \
    --horizons 1 5 10 \
    --nb 40

Outputs
-------
Each grid point is saved in a separate subdirectory, and aggregate CSVs are written to:

kahkm_experiment_16a_sensitivity_outputs/experiment_16a_kahkm_one_step_summary.csv
kahkm_experiment_16a_sensitivity_outputs/experiment_16a_kahkm_multistep_summary.csv
kahkm_experiment_16a_sensitivity_outputs/experiment_16a_best_by_task.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

CsvValue: TypeAlias = str | int | float
CsvRow: TypeAlias = dict[str, CsvValue]

VALID_TASKS = ("CartPole-v1", "MountainCar-v0")


@dataclass(frozen=True)
class Args:
    experiment16_script: str
    output_dir: str
    tasks: tuple[str, ...]
    clusters: tuple[int, ...]
    omegas: tuple[float, ...]
    train_episodes: int
    test_episodes: int
    max_steps: int
    horizons: tuple[int, ...]
    train_seed: int
    test_seed: int
    random_state: int
    selection_horizon: int
    nb: int
    subspace_dim: int
    beta: float
    nlms_epochs: int
    tau: float
    batch_size: int
    n_jobs: int
    kmeans_kind: str
    kmeans_batch_size: int
    max_train_per_cluster: int
    ridge: float
    save_ae_to_disk: bool
    continue_on_error: bool


def _parse_str_tuple(values: list[str]) -> tuple[str, ...]:
    cleaned = tuple(str(v) for v in values)
    if not cleaned:
        raise argparse.ArgumentTypeError("At least one value is required.")
    for task in cleaned:
        if task not in VALID_TASKS:
            raise argparse.ArgumentTypeError(f"Unsupported task {task!r}; choose from {VALID_TASKS}.")
    return cleaned


def _parse_int_tuple(values: list[str]) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if not out:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    if any(v <= 0 for v in out):
        raise argparse.ArgumentTypeError("All integer values must be positive.")
    return out


def _parse_float_tuple(values: list[str]) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not out:
        raise argparse.ArgumentTypeError("At least one float is required.")
    if any((not math.isfinite(v)) or v <= 0.0 for v in out):
        raise argparse.ArgumentTypeError("All float values must be finite and positive.")
    return out


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="KAHKM AI-facing C/omega sensitivity wrapper for Experiment 16.")
    parser.add_argument("--experiment16-script", type=str, default="experiment_16_ai_classic_control_closed_loop_pylance_clean.py")
    parser.add_argument("--output-dir", type=str, default="kahkm_experiment_16a_sensitivity_outputs")
    parser.add_argument("--tasks", nargs="+", default=["CartPole-v1", "MountainCar-v0"])
    parser.add_argument("--clusters", nargs="+", default=["10", "15", "20", "30", "50", "80", "100"])
    parser.add_argument("--omegas", nargs="+", default=["2", "4", "8", "12"])
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses Experiment 16 task defaults.")
    parser.add_argument("--horizons", nargs="+", default=["1", "5", "10", "20", "50"])
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--selection-horizon", type=int, default=50, help="Pick best setting by KAHKM error at this horizon; fallback to max available horizon.")
    parser.add_argument("--nb", type=int, default=100)
    parser.add_argument("--subspace-dim", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nlms-epochs", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--kmeans-kind", type=str, default="full", choices=("auto", "full", "minibatch"))
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-per-cluster", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--no-save-ae-to-disk", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    ns = parser.parse_args()
    return Args(
        experiment16_script=str(ns.experiment16_script),
        output_dir=str(ns.output_dir),
        tasks=_parse_str_tuple([str(x) for x in ns.tasks]),
        clusters=_parse_int_tuple([str(x) for x in ns.clusters]),
        omegas=_parse_float_tuple([str(x) for x in ns.omegas]),
        train_episodes=int(ns.train_episodes),
        test_episodes=int(ns.test_episodes),
        max_steps=int(ns.max_steps),
        horizons=_parse_int_tuple([str(x) for x in ns.horizons]),
        train_seed=int(ns.train_seed),
        test_seed=int(ns.test_seed),
        random_state=int(ns.random_state),
        selection_horizon=int(ns.selection_horizon),
        nb=int(ns.nb),
        subspace_dim=int(ns.subspace_dim),
        beta=float(ns.beta),
        nlms_epochs=int(ns.nlms_epochs),
        tau=float(ns.tau),
        batch_size=int(ns.batch_size),
        n_jobs=int(ns.n_jobs),
        kmeans_kind=str(ns.kmeans_kind),
        kmeans_batch_size=int(ns.kmeans_batch_size),
        max_train_per_cluster=int(ns.max_train_per_cluster),
        ridge=float(ns.ridge),
        save_ae_to_disk=not bool(ns.no_save_ae_to_disk),
        continue_on_error=bool(ns.continue_on_error),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _build_command(args: Args, *, task: str, c_value: int, omega: float, out_dir: Path) -> list[str]:
    command = [
        sys.executable,
        args.experiment16_script,
        "--output-dir",
        str(out_dir),
        "--tasks",
        task,
        "--train-episodes",
        str(args.train_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--max-steps",
        str(args.max_steps),
        "--horizons",
        *[str(h) for h in args.horizons],
        "--train-seed",
        str(args.train_seed),
        "--test-seed",
        str(args.test_seed),
        "--random-state",
        str(args.random_state),
        "--n-clusters",
        str(c_value),
        "--omega",
        str(omega),
        "--nb",
        str(args.nb),
        "--subspace-dim",
        str(args.subspace_dim),
        "--beta",
        str(args.beta),
        "--nlms-epochs",
        str(args.nlms_epochs),
        "--tau",
        str(args.tau),
        "--batch-size",
        str(args.batch_size),
        "--n-jobs",
        str(args.n_jobs),
        "--kmeans-kind",
        args.kmeans_kind,
        "--kmeans-batch-size",
        str(args.kmeans_batch_size),
        "--max-train-per-cluster",
        str(args.max_train_per_cluster),
        "--ridge",
        str(args.ridge),
    ]
    if not args.save_ae_to_disk:
        command.append("--no-save-ae-to-disk")
    return command


def _combo_dir_name(task: str, c_value: int, omega: float) -> str:
    omega_s = str(omega).replace(".", "p").replace("-", "m")
    return f"{task}_C{c_value}_omega{omega_s}"


def _select_best(multistep_rows: list[CsvRow], one_step_rows: list[CsvRow], selection_horizon: int) -> list[CsvRow]:
    tasks = sorted({str(row["task"]) for row in one_step_rows})
    best_rows: list[CsvRow] = []
    for task in tasks:
        candidates = [row for row in multistep_rows if str(row["task"]) == task and str(row["method"]) == "kahkm_nlms"]
        if not candidates:
            candidates_one = [row for row in one_step_rows if str(row["task"]) == task and str(row["method"]) == "kahkm_nlms"]
            best_one = min(candidates_one, key=lambda r: float(r["test_error"]))
            best_rows.append({
                "task": task,
                "selection_metric": "one_step_test_error",
                "selection_horizon": 1,
                "n_clusters": int(best_one["n_clusters"]),
                "omega": float(best_one["omega"]),
                "selected_error": float(best_one["test_error"]),
                "selected_r2": float(best_one["test_r2"]),
            })
            continue

        exact = [row for row in candidates if int(row["horizon"]) == selection_horizon]
        if exact:
            eligible = exact
            horizon_used = selection_horizon
        else:
            horizon_used = max(int(row["horizon"]) for row in candidates)
            eligible = [row for row in candidates if int(row["horizon"]) == horizon_used]
        best = min(eligible, key=lambda r: float(r["relative_error"]))
        best_rows.append({
            "task": task,
            "selection_metric": "kahkm_multistep_relative_error",
            "selection_horizon": int(horizon_used),
            "n_clusters": int(best["n_clusters"]),
            "omega": float(best["omega"]),
            "selected_error": float(best["relative_error"]),
            "selected_r2": float(best["association_r2"]),
        })
    return best_rows


def run() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    script_path = Path(args.experiment16_script)
    if not script_path.exists():
        alt = Path(__file__).resolve().parent / args.experiment16_script
        if alt.exists():
            script_path = alt
        else:
            raise FileNotFoundError(
                f"Could not find {args.experiment16_script!r}. Put this wrapper in the same folder as Experiment 16 "
                "or pass --experiment16-script /path/to/experiment_16_ai_classic_control_closed_loop_pylance_clean.py."
            )

    one_step_summary: list[CsvRow] = []
    multistep_summary: list[CsvRow] = []
    combo_rows: list[CsvRow] = []
    start = time.time()

    for task in args.tasks:
        for c_value in args.clusters:
            for omega in args.omegas:
                combo_name = _combo_dir_name(task, int(c_value), float(omega))
                combo_dir = root / "runs" / combo_name
                command = _build_command(args, task=task, c_value=int(c_value), omega=float(omega), out_dir=combo_dir)
                command[1] = str(script_path)
                print(f"\n=== Running {task}, C={c_value}, omega={omega} ===")
                status = "ok"
                t0 = time.time()
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as exc:
                    status = f"failed:{exc.returncode}"
                    if not args.continue_on_error:
                        raise
                elapsed = time.time() - t0
                combo_rows.append({
                    "task": task,
                    "n_clusters": int(c_value),
                    "omega": float(omega),
                    "status": status,
                    "seconds": float(elapsed),
                    "output_dir": str(combo_dir),
                })
                if status != "ok":
                    continue

                one_rows = _read_csv(combo_dir / "experiment_16_one_step_results.csv")
                for row in one_rows:
                    if row.get("method") != "kahkm_nlms":
                        continue
                    one_step_summary.append({
                        "task": str(row.get("task", task)),
                        "n_clusters": int(c_value),
                        "omega": float(omega),
                        "method": "kahkm_nlms",
                        "train_error": _float(row, "train_error"),
                        "train_r2": _float(row, "train_r2"),
                        "test_error": _float(row, "test_error"),
                        "test_r2": _float(row, "test_r2"),
                        "simplex_violation": _float(row, "simplex_violation"),
                    })

                multi_rows = _read_csv(combo_dir / "experiment_16_multistep_summary.csv")
                for row in multi_rows:
                    method = str(row.get("method", ""))
                    if method not in {"kahkm_nlms", "kahkm_nlms_simplex_project_each_step"}:
                        continue
                    multistep_summary.append({
                        "task": str(row.get("task", task)),
                        "n_clusters": int(c_value),
                        "omega": float(omega),
                        "method": method,
                        "metric_space": str(row.get("metric_space", "kahkm_association")),
                        "horizon": _int(row, "horizon"),
                        "relative_error": _float(row, "relative_error_mean"),
                        "association_r2": _float(row, "association_r2_mean"),
                        "simplex_violation": _float(row, "simplex_violation_mean"),
                    })

                _write_csv(root / "experiment_16a_combo_index.csv", combo_rows)
                if one_step_summary:
                    _write_csv(root / "experiment_16a_kahkm_one_step_summary.csv", one_step_summary)
                if multistep_summary:
                    _write_csv(root / "experiment_16a_kahkm_multistep_summary.csv", multistep_summary)

    if not one_step_summary:
        raise RuntimeError("No successful KAHKM one-step results were collected.")
    _write_csv(root / "experiment_16a_combo_index.csv", combo_rows)
    _write_csv(root / "experiment_16a_kahkm_one_step_summary.csv", one_step_summary)
    if multistep_summary:
        _write_csv(root / "experiment_16a_kahkm_multistep_summary.csv", multistep_summary)
        best_rows = _select_best(multistep_summary, one_step_summary, args.selection_horizon)
    else:
        best_rows = _select_best([], one_step_summary, args.selection_horizon)
    _write_csv(root / "experiment_16a_best_by_task.csv", best_rows)

    metadata = asdict(args)
    metadata["total_seconds"] = float(time.time() - start)
    metadata["note"] = "Grid sensitivity over KAHKM n_clusters C and omega for closed-loop Gymnasium tasks."
    with (root / "experiment_16a_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nBest settings by task:")
    for row in best_rows:
        print(
            f"  {row['task']}: C={row['n_clusters']}, omega={row['omega']}, "
            f"metric={row['selection_metric']}, horizon={row['selection_horizon']}, error={row['selected_error']}"
        )
    print("\nWrote aggregate outputs to:")
    for path in [
        root / "experiment_16a_combo_index.csv",
        root / "experiment_16a_kahkm_one_step_summary.csv",
        root / "experiment_16a_kahkm_multistep_summary.csv",
        root / "experiment_16a_best_by_task.csv",
        root / "experiment_16a_metadata.json",
    ]:
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    run()
