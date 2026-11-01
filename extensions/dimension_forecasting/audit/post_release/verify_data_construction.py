#!/usr/bin/env python3
"""Independently check staged seed, episode, window and A-only scale construction.

Only pinned observation-law generators are replayed. The study preparation,
forecaster, and primary analysis modules are not imported. D is never read or
regenerated unless the final global target release and its predecessor seals
already exist and their hashes verify.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_design_config import declared_seeds
from verify_task_artifacts import array_hash, sha, read, arrays


class DataAudit:
    def __init__(self):
        self.counts = Counter()

    def require(self, condition, message):
        self.counts["assertions"] += 1
        if not condition:
            raise AssertionError(message)

    def exact(self, actual, expected, message):
        self.require(np.array_equal(actual, expected), message)
        self.counts["exact_array_comparisons"] += 1


def checked_file_map(root, mapping, audit, label):
    audit.require(isinstance(mapping, dict) and bool(mapping), label + " empty file map")
    for relative, digest in mapping.items():
        path = (root / relative).resolve()
        audit.require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                      and path.is_relative_to(root), label + " unsafe path")
        audit.require(sha(path) == digest, label + " file hash: " + relative)
    audit.counts[label + "_files_verified"] += len(mapping)


def gates(root, released, audit):
    freeze = read(root / "freeze.json")
    audit.require(freeze["status"] == "frozen", "Source is not frozen")
    audit.require(freeze["config_sha256"] == sha(root / "config.json"), "Frozen config differs")
    checked_file_map(root, freeze["files"], audit, "freeze")
    abc = read(root / "prepared/abc_manifest.json")
    audit.require(abc["freeze_sha256"] == sha(root / "freeze.json")
                  and abc["config_sha256"] == sha(root / "config.json"), "ABC predecessor identity")
    audit.require(abc["D_episodes_generated"] is False and abc["D_targets_materialized"] is False,
                  "ABC phase generated D")
    audit.require(abc["seed_manifest_sha256"] == sha(root / "prepared/seed_manifest.json"), "Seed manifest identity")
    checked_file_map(root, abc["base_files"], audit, "ABC_base")
    d_manifest = None
    if released:
        fit, pred, release = [read(root / name) for name in
                              ("FIT_LOCK.json", "PREDICTION_LOCK.json", "TARGET_RELEASE.json")]
        for label, record in (("fit", fit), ("prediction", pred)):
            audit.require(record["status"] == "sealed", label + " incomplete")
            audit.require(record["freeze_sha256"] == sha(root / "freeze.json")
                          and record["config_sha256"] == sha(root / "config.json"), label + " source predecessor")
            checked_file_map(root, record["files"], audit, label + "_lock")
        audit.require(release["status"] == "sealed" and release["phase"] == "target_release", "No global target release")
        audit.require(pred["fit_lock_sha256"] == sha(root / "FIT_LOCK.json"), "Prediction fit predecessor")
        audit.require(release["prediction_lock_sha256"] == sha(root / "PREDICTION_LOCK.json"), "Target prediction predecessor")
        audit.require(release["fit_lock_sha256"] == sha(root / "FIT_LOCK.json"), "Target fit predecessor")
        audit.require(release["freeze_sha256"] == sha(root / "freeze.json")
                      and release["config_sha256"] == sha(root / "config.json"), "Target source predecessor")
        checked_file_map(root, release["files"], audit, "target_release")
        d_manifest = read(root / "prepared/d_manifest.json")
        audit.require(d_manifest["freeze_sha256"] == sha(root / "freeze.json")
                      and d_manifest["config_sha256"] == sha(root / "config.json")
                      and d_manifest["fit_lock_sha256"] == sha(root / "FIT_LOCK.json")
                      and d_manifest["ABC_manifest_sha256"] == sha(root / "prepared/abc_manifest.json"),
                      "D manifest predecessors")
        audit.require(d_manifest["generation_after_complete_fit_lock"] is True
                      and d_manifest["targets_reserved_until_global_prediction_lock"] is True,
                      "D stage declarations")
        audit.require(pred["D_manifest_sha256"] == sha(root / "prepared/d_manifest.json"), "Prediction D predecessor")
        audit.require(release["D_manifest_sha256"] == sha(root / "prepared/d_manifest.json"), "Target D predecessor")
        audit.require(fit["ABC_manifest_sha256"] == sha(root / "prepared/abc_manifest.json"), "Fit ABC predecessor")
        audit.require(fit["D_generated"] is False and pred["D_scoring_started"] is False
                      and release["targets_read_for_scoring_before_release"] is False, "Phase access declarations")
        expected_ids = {task["task_id"] for task in abc["tasks"]}
        for label, record in (("fit", fit), ("prediction", pred), ("release", release)):
            audit.require(record["task_count"] == len(expected_ids) == len(record["task_ids"])
                          and set(record["task_ids"]) == expected_ids, label + " complete roster")
        dates = [freeze["created_utc"], abc["created_utc"], fit["created_utc"],
                 d_manifest["created_utc"], pred["created_utc"], release["created_utc"]]
        dates = [datetime.fromisoformat(value) for value in dates]
        audit.require(dates == sorted(dates) and len(set(dates)) == len(dates), "Global phase chronology")
        checked_file_map(root, d_manifest["base_files"], audit, "D_base")
    return freeze, abc, d_manifest


def independent_window_rows(episode, history, horizons, stride, retained):
    # Slice each selected history directly, rather than using the preparation
    # module's multidimensional advanced-index operation.
    origins = list(range(history - 1, episode.shape[1] - max(horizons), stride))
    groups = np.asarray([int(row) // len(origins) for row in retained], dtype=np.int64)
    at = np.asarray([origins[int(row) % len(origins)] for row in retained], dtype=np.int64)
    return {
        "X": np.stack([episode[g, t - history + 1:t + 1] for g, t in zip(groups, at)]),
        "Y": np.asarray([[episode[g, t + offset] for offset in horizons] for g, t in zip(groups, at)], dtype=np.float64),
        "groups": groups, "origins": at, "row_indices": np.asarray(retained, dtype=np.int64),
    }


def audit_data(root, *, released=False):
    root = Path(root).resolve()
    audit = DataAudit()
    freeze, abc, d_manifest = gates(root, released, audit)
    config = read(root / "config.json")
    records = declared_seeds(config)
    for record in records:
        if record["role"] == "repetition_identifier":
            record["role"] = "identifier"
    saved_seeds = read(root / "prepared/seed_manifest.json")
    key = lambda r: json.dumps(r, sort_keys=True)
    audit.require(sorted(map(key, records)) == sorted(map(key, saved_seeds["records"])), "Independent seed records differ")
    audit.require(saved_seeds["distinct_seed_count"] == len(records) == len({r["seed"] for r in records}), "Seed count/uniqueness")
    stream = {(r.get("repetition"), r.get("law"), r.get("split"), r["role"]): r for r in records}
    task_metadata = {}
    for repetition in range(config["repetitions"]):
        for law_index, law in enumerate(config["laws"]):
            for history in config["histories"]:
                for noise_index, noise in enumerate(config["noise_levels"]):
                    identity = f"{law}__h{history}__noise{noise_index}__rep{repetition:02d}"
                    task_metadata[identity] = dict(task_id=identity, law=law, law_index=law_index,
                        history=history, noise_index=noise_index, noise_level=float(noise), repetition=repetition,
                        seed=stream[repetition, None, None, "identifier"]["seed"],
                        fit_seed=stream[repetition, law, None, "fit"]["seed"])
    audit.require(len(task_metadata) == config["task_count"] == abc["task_count"], "Complete ABC task count")
    abc_tasks = {t["task_id"]: t for t in abc["tasks"]}
    audit.require(len(abc_tasks) == len(abc["tasks"]) and set(abc_tasks) == set(task_metadata), "ABC factorial roster")
    d_tasks = {t["task_id"]: t for t in d_manifest["tasks"]} if released else {}
    if released:
        audit.require(len(d_tasks) == len(d_manifest["tasks"]) == config["task_count"]
                      and set(d_tasks) == set(task_metadata), "D factorial roster")

    original = root / "vendor_original"
    original_manifest = read(original / "GENERATOR_SOURCE_MANIFEST.json")
    for record in original_manifest["files"]:
        audit.require(sha(original / record["path"]) == record["sha256"], "Pinned generator source hash")
    sys.path.insert(0, str(original))
    spec = importlib.util.spec_from_file_location("independent_pinned_observation_laws", original / "extensions/reconstruction_mechanism/data.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    source_dependencies = [name for name in sys.modules if name.startswith("extensions.")]
    audit.require(all(not any(word in name for word in ("models", "adapter", "forecaster")) for name in source_dependencies),
                  "No fitting runtime imported")

    phases = ("A", "B", "C", "D") if released else ("A", "B", "C")
    for repetition in range(config["repetitions"]):
        for law in config["laws"]:
            base_path = root / f"prepared/base_abc/{law}__rep{repetition:02d}.npz"
            raw = arrays(base_path)
            audit.require(set(raw) == {f"{s}_{kind}" for s in ("A", "B", "C") for kind in ("base_episodes", "unit_noise")}, "ABC raw array roster")
            if released:
                d_raw = arrays(root / f"prepared/withheld_d_base/{law}__rep{repetition:02d}.npz")
                audit.require(set(d_raw) == {"D_base_episodes", "D_unit_noise"}, "D raw array roster")
                raw.update(d_raw)
            retained = {}
            for split in phases:
                n = config["split_episodes"][split]
                generation_seed = stream[repetition, law, split, "generation"]["seed"]
                expected_base = (source.linear_episodes(n, config["max_episode_length"], generation_seed, config["linear_ar"])
                    if law == "linear_ar" else source.generate_episodes(law, n, config["max_episode_length"], generation_seed)[:, :, 0])
                expected_unit = np.random.default_rng(stream[repetition, law, split, "unit_noise"]["seed"]).standard_normal(expected_base.shape)
                audit.exact(raw[split + "_base_episodes"], expected_base, "Pinned generator replay")
                audit.exact(raw[split + "_unit_noise"], expected_unit, "Noise seed replay")
                count = n * config["origins_per_episode"]
                cap = config["split_caps"][split]
                retained[split] = (np.arange(count, dtype=np.int64) if cap is None or cap >= count else
                    np.sort(np.random.default_rng(stream[repetition, law, split, "rows"]["seed"]).choice(count, cap, replace=False)))
                audit.counts["split_generator_replays"] += 1
            for identity, meta in task_metadata.items():
                if meta["repetition"] != repetition or meta["law"] != law:
                    continue
                task = abc_tasks[identity]
                audit.require(all(task[k] == value for k, value in meta.items()), "ABC task metadata")
                audit.require(sha(root / task["input"]) == task["sha256"], "ABC archive hash")
                audit.require(task["base_file"] == base_path.relative_to(root).as_posix()
                              and task["base_file_sha256"] == sha(base_path), "ABC raw dependency")
                actual = arrays(root / task["input"])
                history = meta["history"]
                length = history + config["history_tail_length"]
                amplitude = float(np.std(raw["A_base_episodes"][:, :length], ddof=0))
                noise_sd = meta["noise_level"] * amplitude
                expected_arrays = {"offsets": np.asarray(config["horizons"], dtype=np.int64),
                                   "A_base_episode_std": np.asarray(amplitude), "noise_std": np.asarray(noise_sd)}
                audit.require(task["A_base_episode_std"] == amplitude and task["noise_std"] == noise_sd, "A-only amplitude metadata")
                for split in phases:
                    augmented = raw[split + "_base_episodes"][:, :length] + noise_sd * raw[split + "_unit_noise"][:, :length]
                    windows = independent_window_rows(augmented, history, config["horizons"], config["stride"], retained[split])
                    if split != "D":
                        expected_arrays.update({split + "_" + k: value for k, value in windows.items()})
                        logged = task["splits"][split]
                        expected_streams = {role: stream[repetition, law, split, role] for role in ("generation", "unit_noise", "rows")}
                        audit.require(logged["streams"] == expected_streams, "ABC generation stream provenance")
                        audit.require(logged["retained_row_indices_sha256"] == array_hash(retained[split])
                                      and logged["base_episode_sha256"] == array_hash(raw[split + "_base_episodes"])
                                      and logged["unit_noise_sha256"] == array_hash(raw[split + "_unit_noise"]), "ABC stream array provenance")
                    else:
                        dtask = d_tasks[identity]
                        audit.require(all(dtask[k] == value for k, value in meta.items()), "D task metadata")
                        for file_key in ("query", "target"):
                            audit.require(sha(root / dtask[file_key]) == dtask[file_key + "_sha256"], "D split archive identity")
                        query, target = arrays(root / dtask["query"]), arrays(root / dtask["target"])
                        audit.require(set(query) == {"D_X", "D_groups", "D_origins", "D_row_indices"}, "Target-free query roster")
                        audit.require(set(target) == {"D_Y", "D_groups", "D_origins", "D_row_indices"}, "Separated target roster")
                        for key, value in windows.items():
                            relevant = [query, target] if key not in ("X", "Y") else [query if key == "X" else target]
                            for saved in relevant:
                                audit.exact(saved["D_" + key], value, "Independent D window slicing")
                        for key in ("X", "Y", "groups"):
                            audit.require(dtask["D_" + key + "_sha256"] == array_hash(windows[key]), "D window hash provenance")
                        audit.require(dtask["A_base_episode_std"] == amplitude and dtask["rows"] == len(retained[split])
                                      and dtask["retained_row_indices_sha256"] == array_hash(retained[split]), "D A-amplitude/retention provenance")
                        audit.counts["D_target_joins"] += 1
                expected_arrays["loss_scale"] = np.maximum(np.var(expected_arrays["A_Y"], axis=0, ddof=0), config["model"]["target_variance_floor"])
                audit.require(set(actual) == set(expected_arrays) == set(task["array_sha256"]), "ABC exact array roster")
                for key, value in expected_arrays.items():
                    audit.exact(actual[key], value, "Independent ABC construction: " + key)
                    audit.require(task["array_sha256"][key] == array_hash(value), "ABC array hash")
                audit.exact(task["loss_scale"], expected_arrays["loss_scale"], "A-only metric variance metadata")
                audit.counts["ABC_tasks_verified"] += 1
        print(json.dumps({"audit": "data_construction", "phase": "released" if released else "abc", "repetition_complete": repetition}), flush=True)
    return {"schema": "kahm-full-forecast-independent-data-audit-v1", "status": "passed",
            "completed_utc": datetime.now(timezone.utc).isoformat(), "phase": "released" if released else "abc",
            "D_outcomes_opened": bool(released), "counts": dict(audit.counts), "issues": [],
            "source_freeze_sha256": sha(root / "freeze.json"), "config_sha256": sha(root / "config.json"),
            "ABC_manifest_sha256": sha(root / "prepared/abc_manifest.json"),
            "D_manifest_sha256": sha(root / "prepared/d_manifest.json") if released else None,
            "audit_source_sha256": sha(__file__),
            "generator_check": "Exact replay of pinned observation laws with independently derived seeds; window slicing and normalization reconstructed independently without preparation/model/analysis imports."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("abc", "released"), required=True)
    args = parser.parse_args()
    result = audit_data(args.root, released=args.phase == "released")
    path = args.root / "audit/post_release" / ("released_data_checks.json" if args.phase == "released" else "abc_data_checks.json")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
