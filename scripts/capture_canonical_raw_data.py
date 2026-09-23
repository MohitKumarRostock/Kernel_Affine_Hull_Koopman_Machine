#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys

sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = ROOT / "reproduction" / "raw_data" / "canonical_input_plan.json"
DEFAULT_CAPTURE_DIR = ROOT / "reproduction" / "raw_data" / "canonical"

sys.path.insert(0, str(ROOT))

from scripts.raw_data_capture import save_raw_dataset, verify_raw_dataset  # noqa: E402


def slug_float(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def array_prefix(dataset_id: str) -> str:
    return dataset_id.replace("-", "_").replace(".", "p")


def add_arrays(target: dict[str, np.ndarray], dataset_id: str, arrays: dict[str, Any]) -> list[str]:
    prefix = array_prefix(dataset_id)
    keys: list[str] = []
    for name, value in arrays.items():
        key = f"{prefix}__{name}"
        if key in target:
            raise RuntimeError(f"Duplicate array key: {key}")
        target[key] = np.ascontiguousarray(np.asarray(value))
        keys.append(key)
    return keys


def pack_episodes(episodes: Iterable[np.ndarray]) -> dict[str, np.ndarray]:
    eps = [np.asarray(x, dtype=np.float64) for x in episodes]
    if not eps:
        return {
            "episode_states": np.empty((0, 0), dtype=np.float64),
            "episode_offsets": np.asarray([0], dtype=np.int64),
        }

    state_dim = int(eps[0].shape[0])
    for episode in eps:
        if episode.ndim != 2 or int(episode.shape[0]) != state_dim:
            raise ValueError("Episode arrays must have shape (state_dim, n_states).")

    offsets = [0]
    for episode in eps:
        offsets.append(offsets[-1] + int(episode.shape[1]))

    return {
        "episode_states": np.concatenate(eps, axis=1),
        "episode_offsets": np.asarray(offsets, dtype=np.int64),
    }


def argparse_default(source_path: Path, flag: str, fallback: Any = None) -> Any:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        try:
            name = ast.literal_eval(node.args[0])
        except Exception:
            continue
        if name != flag:
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    return ast.literal_eval(kw.value)
                except Exception:
                    return fallback
    return fallback


def assignment_expression(source_path: Path, variable: str) -> tuple[str, list[str]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    candidates: list[ast.expr] = []
    for node in ast.walk(tree):
        value: ast.expr | None = None
        names: list[str] = []
        if isinstance(node, ast.Assign):
            value = node.value
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
            names = [node.target.id]
        if variable in names and value is not None:
            candidates.append(value)

    if not candidates:
        raise RuntimeError(f"Could not find assignment to {variable} in {source_path}")

    expr = candidates[-1]
    deps = sorted({
        node.id
        for node in ast.walk(expr)
        if isinstance(node, ast.Name)
        and node.id not in {"int", "float", "round", "abs", "max", "min"}
    })
    return ast.unparse(expr), deps


def eval_noise_seed(expr: str, **variables: Any) -> int:
    env = {
        "__builtins__": {},
        "int": int,
        "float": float,
        "round": round,
        "abs": abs,
        "max": max,
        "min": min,
    }
    env.update(variables)
    return int(eval(expr, env, {}))


def make_plan() -> dict[str, Any]:
    plans: list[dict[str, Any]] = []

    def add(
        family: str,
        dataset_id: str,
        *,
        system_or_task: str,
        role: str,
        seed: int,
        params: dict[str, Any],
        results: list[str],
    ) -> None:
        plans.append({
            "family_archive": family,
            "dataset_id": dataset_id,
            "system_or_task": system_or_task,
            "role": role,
            "seed": int(seed),
            "generation_parameters": params,
            "covered_result_ids": results,
        })

    # 11 Duffing clean trajectories.
    for seed in range(5):
        add(
            "duffing_clean",
            f"duffing_n1200_seed{seed}",
            system_or_task="duffing",
            role="clean_trajectory",
            seed=seed,
            params={"n_steps": 1200, "dt": 0.03},
            results=["R01", "R02", "R05", "R06", "R10", "R11", "R12", "R21", "R22"],
        )
    for seed in range(3):
        add(
            "duffing_clean",
            f"duffing_n1600_seed{seed}",
            system_or_task="duffing",
            role="clean_trajectory",
            seed=seed,
            params={"n_steps": 1600, "dt": 0.03},
            results=["R07", "R08"],
        )

    # Expanded Duffing tuning uses process_noise=1e-5 and must be archived separately.
    for seed in range(3):
        add(
            "duffing_tuning_process_noise",
            f"duffing_tuning_n1600_seed{seed}_process_noise1em05",
            system_or_task="duffing",
            role="tuning_trajectory_with_process_noise",
            seed=seed,
            params={
                "n_steps": 1600,
                "dt": 0.03,
                "process_noise": 1e-5,
            },
            results=["R04", "R11", "R21", "R22"],
        )

    for seed in (100, 101, 102):
        add(
            "duffing_clean",
            f"duffing_n800_seed{seed}",
            system_or_task="duffing",
            role="clean_heldout_trajectory",
            seed=seed,
            params={"n_steps": 800, "dt": 0.03},
            results=["R10", "R11", "R12", "R21", "R22"],
        )

    # 35 Van der Pol clean trajectories.
    for seed in range(3):
        add(
            "vanderpol_clean",
            f"vdp_n1600_seed{seed}",
            system_or_task="vanderpol",
            role="clean_trajectory",
            seed=seed,
            params={"n_steps": 1600, "dt": 0.02, "mu": 1.0},
            results=["R01", "R03", "R04", "R06", "R07", "R08"],
        )
    for seed in range(5):
        add(
            "vanderpol_clean",
            f"vdp_n1200_seed{seed}",
            system_or_task="vanderpol",
            role="clean_training_trajectory",
            seed=seed,
            params={"n_steps": 1200, "dt": 0.02, "mu": 1.0},
            results=["R10", "R11", "R13", "R21", "R22"],
        )
    for seed in (100, 101, 102):
        add(
            "vanderpol_clean",
            f"vdp_n800_seed{seed}",
            system_or_task="vanderpol",
            role="clean_heldout_trajectory",
            seed=seed,
            params={"n_steps": 800, "dt": 0.02, "mu": 1.0},
            results=["R09", "R10", "R11", "R13", "R21", "R22"],
        )
    for replicate, base in enumerate((0, 1000, 2000)):
        for seed in range(base, base + 8):
            add(
                "vanderpol_clean",
                f"vdp_n900_seed{seed}",
                system_or_task="vanderpol",
                role="generalization_training_trajectory",
                seed=seed,
                params={
                    "n_steps": 900,
                    "dt": 0.02,
                    "mu": 1.0,
                    "replicate": replicate,
                },
                results=["R09"],
            )

    # 480 noisy training subdatasets:
    # 30 Experiment-12 reference-map inputs + 450 noise-aware tuning inputs.
    noise_levels = (0.0, 0.005, 0.01, 0.02, 0.05)
    for system in ("duffing", "vanderpol"):
        for seed in range(3):
            for level in noise_levels:
                add(
                    "noisy_training",
                    f"noise_ref_{system}_seed{seed}_level{slug_float(level)}",
                    system_or_task=system,
                    role="noisy_training_reference_map",
                    seed=seed,
                    params={
                        "n_steps": 1600,
                        "noise_level": level,
                        "noise_recipe": "experiment_12_noise_robustness.py",
                    },
                    results=["R08"],
                )

    for seed in range(3):
        for n_clusters in (10, 15, 20, 25, 30, 40):
            for omega in (0.25, 0.5, 1.0, 2.0, 4.0):
                for level in noise_levels:
                    add(
                        "noisy_training",
                        f"noise_aware_vdp_seed{seed}_C{n_clusters}_omega{slug_float(omega)}_level{slug_float(level)}",
                        system_or_task="vanderpol",
                        role="noise_aware_tuning_training",
                        seed=seed,
                        params={
                            "n_steps": 1600,
                            "dt": 0.02,
                            "mu": 1.0,
                            "n_clusters": n_clusters,
                            "omega": omega,
                            "noise_level": level,
                            "noise_recipe": "run_exp12_vanderpol_noise_aware_tuning.py",
                        },
                        results=["R07"],
                    )

    # 14 unique Experiment-16 task datasets.
    for task in ("CartPole-v1", "MountainCar-v0"):
        for seed in (0, 1, 2):
            add(
                "classic_control_exp16",
                f"exp16_{task}_train_seed{seed}",
                system_or_task=task,
                role="train",
                seed=seed,
                params={"episodes": 8, "collector": "experiment_16"},
                results=["R14", "R15"],
            )
            add(
                "classic_control_exp16",
                f"exp16_{task}_test_seed{1000 + seed}",
                system_or_task=task,
                role="test",
                seed=1000 + seed,
                params={"episodes": 4, "collector": "experiment_16"},
                results=["R14"],
            )
        add(
            "classic_control_exp16",
            f"exp16_{task}_test_seed100",
            system_or_task=task,
            role="test",
            seed=100,
            params={"episodes": 4, "collector": "experiment_16"},
            results=["R15"],
        )

    # 4 Experiment-17 interpretability datasets.
    for task in ("CartPole-v1", "MountainCar-v0"):
        add(
            "classic_control_exp17",
            f"exp17_{task}_train_seed0",
            system_or_task=task,
            role="train",
            seed=0,
            params={"episodes": 8, "collector": "experiment_17"},
            results=["R17", "R18"],
        )
        add(
            "classic_control_exp17",
            f"exp17_{task}_test_seed2000",
            system_or_task=task,
            role="test",
            seed=2000,
            params={"episodes": 4, "collector": "experiment_17"},
            results=["R17", "R18"],
        )

    # 6 selected Acrobot validation datasets.
    for seed in (0, 1, 2):
        add(
            "acrobot_exp18",
            f"acrobot_train_seed{seed}",
            system_or_task="Acrobot-v1",
            role="train",
            seed=seed,
            params={
                "episodes": 12,
                "max_steps": 500,
                "exploration_eps": 0.1,
                "collector": "experiment_18",
            },
            results=["R16", "R17", "R19", "R20"],
        )
        add(
            "acrobot_exp18",
            f"acrobot_test_seed{100 + seed}",
            system_or_task="Acrobot-v1",
            role="test",
            seed=100 + seed,
            params={
                "episodes": 6,
                "max_steps": 500,
                "exploration_eps": 0.1,
                "collector": "experiment_18",
            },
            results=["R16", "R17", "R19", "R20"],
        )

    ref_expr, ref_deps = assignment_expression(
        ROOT / "experiment_12_noise_robustness.py", "noise_seed"
    )
    aware_expr, aware_deps = assignment_expression(
        ROOT / "run_exp12_vanderpol_noise_aware_tuning.py", "noise_seed"
    )
    expected_aware_deps = {"n_clusters", "noise_level", "omega", "seed"}
    if set(aware_deps) != expected_aware_deps:
        raise RuntimeError(
            f"Unexpected noise-aware noise-seed dependencies: {aware_deps}"
        )

    counts: dict[str, int] = {}
    for row in plans:
        counts[row["family_archive"]] = counts.get(row["family_archive"], 0) + 1

    expected_counts = {
        "duffing_clean": 11,
        "duffing_tuning_process_noise": 3,
        "vanderpol_clean": 35,
        "noisy_training": 480,
        "classic_control_exp16": 14,
        "classic_control_exp17": 4,
        "acrobot_exp18": 6,
    }
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected family counts: {counts}")

    covered = [f"R{i:02d}" for i in range(1, 23)]
    return {
        "plan_version": 1,
        "family_archive_count": 7,
        "subdataset_count": len(plans),
        "counts_by_family_archive": counts,
        "covered_result_ids": covered,
        "excluded_documentary_result_ids": ["R23"],
        "noise_seed_dependencies": {
            "reference_noise_robustness": {
                "expression": ref_expr,
                "dependencies": ref_deps,
            },
            "noise_aware_tuning": {
                "expression": aware_expr,
                "dependencies": aware_deps,
            },
        },
        "subdatasets": plans,
    }


def write_plan(plan: dict[str, Any]) -> None:
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_generator_equivalence() -> None:
    import experiment_02_duffing_sensitivity as d1
    import run_exp_duffing_tuning as d2
    import experiment_07_vanderpol_abstraction_ablation as v1
    import run_exp12_vanderpol_noise_aware_tuning as v2
    import run_exp13_vanderpol_tuned_trajectory_generalization as v3

    clean0_a, clean1_a = d1.make_duffing_snapshots(n_steps=32, dt=0.03, seed=2)
    clean0_b, clean1_b = d1.make_duffing_snapshots(n_steps=32, dt=0.03, seed=2)

    if not (
        np.array_equal(clean0_a, clean0_b)
        and np.array_equal(clean1_a, clean1_b)
    ):
        raise RuntimeError("Clean Duffing generator is not deterministic.")

    process_noise = float(
        argparse_default(
            ROOT / "run_exp_duffing_tuning.py",
            "--process-noise",
            0.0,
        )
    )

    if process_noise != 1e-5:
        raise RuntimeError(
            "Archived Duffing tuning recipe is expected to use "
            f"process_noise=1e-5, found {process_noise}."
        )

    noisy0_a, noisy1_a = d2.make_duffing_snapshots(
        n_steps=32,
        dt=0.03,
        seed=2,
        process_noise=process_noise,
    )
    noisy0_b, noisy1_b = d2.make_duffing_snapshots(
        n_steps=32,
        dt=0.03,
        seed=2,
        process_noise=process_noise,
    )

    if not (
        np.array_equal(noisy0_a, noisy0_b)
        and np.array_equal(noisy1_a, noisy1_b)
    ):
        raise RuntimeError("Duffing process-noise tuning generator is not deterministic.")

    if (
        np.array_equal(clean0_a, noisy0_a)
        and np.array_equal(clean1_a, noisy1_a)
    ):
        raise RuntimeError(
            "Duffing process-noise tuning trajectory unexpectedly equals "
            "the clean trajectory."
        )

    a0, a1 = v1.make_vanderpol_snapshots(n_steps=32, dt=0.02, seed=2, mu=1.0)
    b0, b1 = v2.make_vanderpol_snapshots(n_steps=32, dt=0.02, seed=2, mu=1.0)
    c0, c1 = v3.make_vanderpol_snapshots(n_steps=32, dt=0.02, seed=2, mu=1.0)
    if not (
        np.array_equal(a0, b0)
        and np.array_equal(a1, b1)
        and np.array_equal(a0, c0)
        and np.array_equal(a1, c1)
    ):
        raise RuntimeError("Van der Pol generator implementations are not equivalent.")

    print("PASS: canonical generator determinism and recipe separation")


def capture_family(
    *,
    output_dir: Path,
    family: str,
    arrays: dict[str, np.ndarray],
    subdatasets: list[dict[str, Any]],
    source_script: str,
) -> None:
    npz_path, meta_path = save_raw_dataset(
        root=ROOT,
        output_dir=output_dir,
        dataset_id=family,
        role="canonical_raw_input_family",
        source_script=source_script,
        generator_seed=None,
        arrays=arrays,
        generation_parameters={
            "logical_subdataset_count": len(subdatasets),
        },
        extra_metadata={
            "logical_subdatasets": subdatasets,
        },
    )
    verify_raw_dataset(npz_path, meta_path)
    print(f"PASS: {family} ({len(subdatasets)} logical subdatasets)")


def capture_clean_oscillators(plan: dict[str, Any], output_dir: Path) -> None:
    import experiment_02_duffing_sensitivity as d
    import experiment_07_vanderpol_abstraction_ablation as v
    import run_exp_duffing_tuning as d_tuning

    for family, generator in (
        ("duffing_clean", d.make_duffing_snapshots),
        ("vanderpol_clean", v.make_vanderpol_snapshots),
    ):
        rows = [x for x in plan["subdatasets"] if x["family_archive"] == family]
        arrays: dict[str, np.ndarray] = []
        arrays = {}
        enriched: list[dict[str, Any]] = []

        for row in rows:
            p = row["generation_parameters"]
            if family == "duffing_clean":
                X0, X1 = generator(
                    n_steps=int(p["n_steps"]),
                    dt=float(p["dt"]),
                    seed=int(row["seed"]),
                )
            else:
                X0, X1 = generator(
                    n_steps=int(p["n_steps"]),
                    dt=float(p["dt"]),
                    seed=int(row["seed"]),
                    mu=float(p["mu"]),
                )

            item = dict(row)
            item["array_keys"] = add_arrays(
                arrays,
                row["dataset_id"],
                {"X0": X0, "X1": X1},
            )
            enriched.append(item)

        capture_family(
            output_dir=output_dir,
            family=family,
            arrays=arrays,
            subdatasets=enriched,
            source_script=(
                "experiment_02_duffing_sensitivity.py"
                if family == "duffing_clean"
                else "experiment_07_vanderpol_abstraction_ablation.py"
            ),
        )

    tuning_rows = [
        x
        for x in plan["subdatasets"]
        if x["family_archive"] == "duffing_tuning_process_noise"
    ]
    tuning_arrays: dict[str, np.ndarray] = {}
    tuning_enriched: list[dict[str, Any]] = []

    for row in tuning_rows:
        p = row["generation_parameters"]
        X0, X1 = d_tuning.make_duffing_snapshots(
            n_steps=int(p["n_steps"]),
            dt=float(p["dt"]),
            seed=int(row["seed"]),
            process_noise=float(p["process_noise"]),
        )
        item = dict(row)
        item["array_keys"] = add_arrays(
            tuning_arrays,
            row["dataset_id"],
            {"X0": X0, "X1": X1},
        )
        tuning_enriched.append(item)

    capture_family(
        output_dir=output_dir,
        family="duffing_tuning_process_noise",
        arrays=tuning_arrays,
        subdatasets=tuning_enriched,
        source_script="run_exp_duffing_tuning.py",
    )


def capture_noisy_training(plan: dict[str, Any], output_dir: Path) -> None:
    import experiment_12_noise_robustness as ref
    import run_exp12_vanderpol_noise_aware_tuning as aware

    ref_expr = plan["noise_seed_dependencies"]["reference_noise_robustness"]["expression"]
    aware_expr = plan["noise_seed_dependencies"]["noise_aware_tuning"]["expression"]

    ref_fraction = float(
        argparse_default(ROOT / "experiment_12_noise_robustness.py", "--train-fraction", 0.75)
    )
    aware_fraction = float(
        argparse_default(ROOT / "run_exp12_vanderpol_noise_aware_tuning.py", "--train-fraction", 0.75)
    )

    rows = [x for x in plan["subdatasets"] if x["family_archive"] == "noisy_training"]
    arrays: dict[str, np.ndarray] = {}
    enriched: list[dict[str, Any]] = []

    for row in rows:
        p = row["generation_parameters"]
        level = float(p["noise_level"])
        seed = int(row["seed"])

        if p["noise_recipe"] == "experiment_12_noise_robustness.py":
            system = str(row["system_or_task"])
            dt = 0.03 if system == "duffing" else 0.02
            X0, X1 = ref.make_snapshots(
                system,
                n_steps=1600,
                dt=dt,
                seed=seed,
                vanderpol_mu=1.0,
            )
            split = int(ref_fraction * X0.shape[1])
            noise_seed = eval_noise_seed(
                ref_expr,
                seed=seed,
                noise_level=level,
            )
            rng = np.random.default_rng(noise_seed)
            noisy0 = ref.add_coordinate_noise(
                X0[:, :split], noise_level=level, rng=rng
            )
            noisy1 = ref.add_coordinate_noise(
                X1[:, :split], noise_level=level, rng=rng
            )

        else:
            n_clusters = int(p["n_clusters"])
            omega = float(p["omega"])
            X0, X1 = aware.make_vanderpol_snapshots(
                n_steps=1600,
                dt=0.02,
                seed=seed,
                mu=1.0,
            )
            split = int(aware_fraction * X0.shape[1])
            noise_seed = eval_noise_seed(
                aware_expr,
                seed=seed,
                n_clusters=n_clusters,
                omega=omega,
                noise_level=level,
            )
            rng = np.random.default_rng(noise_seed)
            noisy0 = aware.add_coordinate_noise(
                X0[:, :split], noise_level=level, rng=rng
            )
            noisy1 = aware.add_coordinate_noise(
                X1[:, :split], noise_level=level, rng=rng
            )

        item = dict(row)
        item["noise_seed"] = int(noise_seed)
        item["array_keys"] = add_arrays(
            arrays,
            row["dataset_id"],
            {"X0_train_noisy": noisy0, "X1_train_noisy": noisy1},
        )
        enriched.append(item)

    capture_family(
        output_dir=output_dir,
        family="noisy_training",
        arrays=arrays,
        subdatasets=enriched,
        source_script=(
            "experiment_12_noise_robustness.py; "
            "run_exp12_vanderpol_noise_aware_tuning.py"
        ),
    )


def capture_exp16(plan: dict[str, Any], output_dir: Path) -> None:
    import experiment_16_ai_classic_control_closed_loop_pylance_clean as e16

    rows = [x for x in plan["subdatasets"] if x["family_archive"] == "classic_control_exp16"]
    arrays: dict[str, np.ndarray] = {}
    enriched: list[dict[str, Any]] = []

    for row in rows:
        task = str(row["system_or_task"])
        cfg = e16.DEFAULT_TASK_CONFIGS[task]
        data = e16.collect_closed_loop_data(
            task,
            episodes=int(row["generation_parameters"]["episodes"]),
            max_steps=int(cfg.max_steps),
            seed=int(row["seed"]),
            exploration_eps=float(cfg.exploration_eps),
        )
        payload = {
            "X0": data.X0,
            "X1": data.X1,
            "actions": data.actions,
            "rewards": data.rewards,
            "terminals": data.terminals,
            **pack_episodes(data.episodes),
        }
        item = dict(row)
        item["generation_parameters"] = {
            **item["generation_parameters"],
            "max_steps": int(cfg.max_steps),
            "exploration_eps": float(cfg.exploration_eps),
        }
        item["array_keys"] = add_arrays(arrays, row["dataset_id"], payload)
        enriched.append(item)

    capture_family(
        output_dir=output_dir,
        family="classic_control_exp16",
        arrays=arrays,
        subdatasets=enriched,
        source_script="experiment_16_ai_classic_control_closed_loop_pylance_clean.py",
    )


def capture_exp17(plan: dict[str, Any], output_dir: Path) -> None:
    import experiment_17_ai_regime_interpretability_pylance_clean as e17

    rows = [x for x in plan["subdatasets"] if x["family_archive"] == "classic_control_exp17"]
    arrays: dict[str, np.ndarray] = {}
    enriched: list[dict[str, Any]] = []

    for row in rows:
        task = str(row["system_or_task"])
        cfg = e17.DEFAULT_TASK_CONFIGS[task]
        data = e17.collect_data(
            task,
            episodes=int(row["generation_parameters"]["episodes"]),
            max_steps=int(cfg.max_steps),
            seed=int(row["seed"]),
            exploration_eps=float(cfg.exploration_eps),
        )
        payload = {
            "X0": data.X0,
            "X1": data.X1,
            "actions": data.actions,
            "rewards": data.rewards,
            "terminated": data.terminated,
            "truncated": data.truncated,
            "episode_id": data.episode_id,
            "step_id": data.step_id,
            "steps_until_end": data.steps_until_end,
            **pack_episodes(data.episodes),
        }
        item = dict(row)
        item["generation_parameters"] = {
            **item["generation_parameters"],
            "max_steps": int(cfg.max_steps),
            "exploration_eps": float(cfg.exploration_eps),
        }
        item["array_keys"] = add_arrays(arrays, row["dataset_id"], payload)
        enriched.append(item)

    capture_family(
        output_dir=output_dir,
        family="classic_control_exp17",
        arrays=arrays,
        subdatasets=enriched,
        source_script="experiment_17_ai_regime_interpretability_pylance_clean.py",
    )


def capture_acrobot(plan: dict[str, Any], output_dir: Path) -> None:
    import experiment_18_acrobot_sequential_decision_pylance_clean as e18

    rows = [x for x in plan["subdatasets"] if x["family_archive"] == "acrobot_exp18"]
    arrays: dict[str, np.ndarray] = {}
    enriched: list[dict[str, Any]] = []

    for row in rows:
        p = row["generation_parameters"]
        data = e18.collect_acrobot_data(
            episodes=int(p["episodes"]),
            max_steps=int(p["max_steps"]),
            seed=int(row["seed"]),
            exploration_eps=float(p["exploration_eps"]),
        )
        payload = {
            "X0": data.X0,
            "X1": data.X1,
            "actions": data.actions,
            "rewards": data.rewards,
            "terminals": data.terminals,
            "episode_ids": data.episode_ids,
            "step_ids": data.step_ids,
            **pack_episodes(data.episodes),
        }
        item = dict(row)
        item["array_keys"] = add_arrays(arrays, row["dataset_id"], payload)
        enriched.append(item)

    capture_family(
        output_dir=output_dir,
        family="acrobot_exp18",
        arrays=arrays,
        subdatasets=enriched,
        source_script="experiment_18_acrobot_sequential_decision_pylance_clean.py",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the deduplicated canonical raw scientific inputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = make_plan()
    if int(plan["subdataset_count"]) != 553:
        raise RuntimeError(f"Expected 553 logical subdatasets, got {plan['subdataset_count']}")

    write_plan(plan)
    validate_generator_equivalence()

    print(f"Family archives: {plan['family_archive_count']}")
    print(f"Canonical subdatasets: {plan['subdataset_count']}")
    print("Covered retained empirical results: R01-R22")

    if args.plan_only:
        print(f"Wrote plan: {PLAN_PATH}")
        return

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite only for an intentional recapture."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_clean_oscillators(plan, output_dir)
    capture_noisy_training(plan, output_dir)
    capture_exp16(plan, output_dir)
    capture_exp17(plan, output_dir)
    capture_acrobot(plan, output_dir)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_raw_data_manifest.py"),
            "--raw-dir",
            str(output_dir),
            "--require-nonempty",
            "--require-clean-captures",
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"PASS: canonical raw-data campaign captured in {output_dir}")


if __name__ == "__main__":
    main()
