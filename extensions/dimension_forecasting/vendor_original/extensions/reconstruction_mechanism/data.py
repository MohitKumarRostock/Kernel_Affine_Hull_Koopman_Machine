"""Prepare independent, paired-noise episodes without fitting any forecaster.

The three nonlinear numerical laws are imported unchanged from the archived
study.  Window indexing follows direct_reconstruction/run.py explicitly, so the
published arrays can also be checked without importing any forecasting code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from extensions.predictive_geometry.study.datasets import generate_episodes

SPLITS = ("A", "B", "C", "D")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    prefix = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True).encode()
    return hashlib.sha256(prefix + b"\0" + value.tobytes()).hexdigest()


def seed_from(parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def make_windows(episodes, history, horizons, stride):
    episodes = np.asarray(episodes, dtype=np.float64)
    if episodes.ndim != 2 or not np.isfinite(episodes).all():
        raise ValueError("Expected finite scalar episode matrix")
    horizons = np.asarray(horizons, dtype=np.int64)
    origins = np.arange(history - 1, episodes.shape[1] - int(horizons.max()),
                        stride, dtype=np.int64)
    history_index = origins[:, None] - np.arange(history - 1, -1, -1)[None, :]
    target_index = origins[:, None] + horizons[None, :]
    count = len(episodes) * len(origins)
    return {
        "X": np.ascontiguousarray(episodes[:, history_index].reshape(count, history)),
        "Y": np.ascontiguousarray(episodes[:, target_index].reshape(count, len(horizons))),
        "groups": np.repeat(np.arange(len(episodes), dtype=np.int64), len(origins)),
        "origins": np.tile(origins, len(episodes)),
        "row_indices": np.arange(count, dtype=np.int64),
    }


def linear_episodes(n, length, seed, config):
    burn = int(config["burn_in_stored_values"])
    total = burn + length
    rng = np.random.default_rng(seed)
    y = np.empty((n, total), dtype=np.float64)
    y[:, :2] = np.asarray(config["initial_values"])
    a, b = config["coefficients"]
    for t in range(2, total):
        y[:, t] = a * y[:, t - 1] + b * y[:, t - 2] + rng.normal(
            0., config["innovation_std"], size=n)
    return y[:, burn:]


def prepare_pair(config, law, seed, repetition):
    law_index = config["laws"].index(law)
    salts = config["seed_salts"]
    base, noise, retained, records = {}, {}, {}, {}
    for split_index, split in enumerate(SPLITS):
        n = config["split_episodes"][split]
        generation_parts = [seed, law_index, split_index, salts["generation"]]
        noise_parts = [seed, law_index, split_index, salts["unit_noise"]]
        cap_parts = [seed, law_index, split_index, salts["caps"]]
        generation_seed = seed_from(generation_parts)
        base[split] = (linear_episodes(n, config["episode_length"], generation_seed,
                                       config["linear_ar"])
                       if law == "linear_ar" else generate_episodes(
                           law, n, config["episode_length"], generation_seed)[:, :, 0])
        noise[split] = np.random.default_rng(seed_from(noise_parts)).standard_normal(
            base[split].shape)
        full_rows = n * len(range(config["history"] - 1,
                                  config["episode_length"] - max(config["horizons"]),
                                  config["stride"]))
        cap = config["split_caps"][split]
        retained[split] = (np.arange(full_rows, dtype=np.int64) if cap is None or cap >= full_rows
                           else np.sort(np.random.default_rng(seed_from(cap_parts)).choice(
                               full_rows, cap, replace=False)))
        records[split] = {
            "generation_seed_components": generation_parts,
            "generation_seed": generation_seed,
            "unit_noise_seed_components": noise_parts,
            "unit_noise_seed": seed_from(noise_parts),
            "cap_seed_components": cap_parts,
            "cap_seed": seed_from(cap_parts),
            "episodes": n, "uncapped_rows": full_rows,
            "retained_rows": len(retained[split]),
            "base_episode_sha256": array_hash(base[split]),
            "unit_noise_sha256": array_hash(noise[split]),
            "retained_row_indices_sha256": array_hash(retained[split]),
        }
    amplitude = float(np.std(base["A"], ddof=0))
    fit_parts = [seed, law_index, salts["fit"]]
    for noise_index, level in enumerate(config["noise_levels"]):
        arrays = {"offsets": np.asarray(config["horizons"], dtype=np.int64),
                  "noise_std": np.asarray(float(level) * amplitude),
                  "A_base_episode_std": np.asarray(amplitude)}
        for split in SPLITS:
            augmented = base[split] + float(level) * amplitude * noise[split]
            windows = make_windows(augmented, config["history"], config["horizons"],
                                   config["stride"])
            arrays.update({f"{split}_{key}": np.ascontiguousarray(value[retained[split]])
                           for key, value in windows.items()})
            arrays[f"{split}_episodes"] = augmented
            arrays[f"{split}_base_episodes"] = base[split]
            arrays[f"{split}_unit_noise"] = noise[split]
        scale = np.maximum(np.var(arrays["A_Y"], axis=0, ddof=0),
                           config["analysis"]["variance_floor"])
        arrays["loss_scale"] = scale
        task_id = f"{law}__noise{noise_index}__seed{seed}"
        yield arrays, {
            "task_id": task_id, "law": law, "seed": seed, "repetition": repetition,
            "noise_level": float(level), "noise_index": noise_index,
            "noise_std": float(level) * amplitude, "A_base_episode_std": amplitude,
            "fit_seed": seed_from(fit_parts), "fit_seed_components": fit_parts,
            "loss_scale": scale.tolist(), "splits": records,
            "array_sha256": {key: array_hash(value) for key, value in arrays.items()},
        }


def prepare(study_dir=ROOT, *, output=None, development=False):
    study_dir = Path(study_dir).resolve()
    config = json.loads((study_dir / "config.json").read_text())
    output = Path(output).resolve() if output else study_dir / "prepared_inputs"
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Will not replace prepared inputs: {output}")
    if not development and (study_dir / "freeze.json").exists():
        raise FileExistsError("Reserved inputs cannot be prepared after the freeze")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tasks").mkdir(exist_ok=True)
    seeds = [config["development_seed"]] if development else config["seeds"]
    tasks = []
    for repetition, seed in enumerate(seeds):
        for law in config["laws"]:
            for arrays, task in prepare_pair(config, law, seed, repetition):
                path = output / "tasks" / (task["task_id"] + ".npz")
                if path.exists():
                    raise FileExistsError(path)
                np.savez_compressed(path, **arrays)
                task["input"] = str(path.relative_to(study_dir)) if path.is_relative_to(study_dir) else str(path)
                task["sha256"] = sha256_file(path)
                tasks.append(task)
        print(json.dumps({"prepared_repetition": repetition, "seed": seed,
                          "total_tasks_prepared": len(tasks)}), flush=True)
    manifest = {
        "schema_version": config["schema_version"], "created_utc": utcnow(),
        "development_only": bool(development), "config_sha256": sha256_file(study_dir / "config.json"),
        "tasks": tasks, "task_count": len(tasks),
        "evaluation_status": "No models fitted; no reserved forecast performance computed",
        "independence": "Independent repetition aggregates; disjoint episodes across A/B/C/D; origins within episodes dependent; noise levels paired",
        "augmentation": "eta * std(all unaugmented A episode values, ddof=0) * independent standard Gaussian noise; every observed history and target augmented",
        "law_scope": "Three previously investigated nonlinear numerical laws and a linear Gaussian negative control; new independent realizations, not new domains",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    prepare(args.study_dir, output=args.output, development=args.development)
