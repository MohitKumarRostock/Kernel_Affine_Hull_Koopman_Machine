#!/usr/bin/env python3
"""Independently verify matched primary selections/effects using only stdlib.

This program does not import the study runner or analyzer. It reconstructs
validation-only selections from all raw JSON candidates, checks PenDigits
writer aggregation, and compares identities and values with exported tables.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

MODES = ("standard", "minmax", "minmax_clip", "raw")
GEOMETRIES = ("kahm", "rbf", "affine")
HEADS = ("nested", "frobenius", "koopman")
TASKS = ("harmonic", "quasiperiodic", "nonlinear_ar", "logistic", "vanderpol", "duffing", "pendigits")
SEEDS = tuple(range(5701, 5706))
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def classify(value):
    return "improved" if value < -TOL else "degraded" if value > TOL else "tie_within_1e-12"


def verify(study, analysis):
    errors, checks, hashes = defaultdict(float), Counter(), {}

    def close(a, b, category):
        a, b = float(a), float(b)
        assert math.isfinite(a) and math.isfinite(b), (category, a, b)
        error = abs(a - b)
        errors[category] = max(errors[category], error)
        checks[category] += 1
        assert error <= TOL, (category, a, b, error)

    def tracked(path):
        hashes[str(path)] = sha(path)
        return read(path)

    completion = tracked(study / "completion.json")
    assert (completion["scheduled_jobs"], completion["completed_jobs"], completion["failed_jobs"]) == (124, 124, 0)
    jobs = tracked(study / "schedule.json")
    assert len(jobs) == len({job["id"] for job in jobs}) == 124
    assert {row["id"] for row in completion["results"]} == {job["id"] for job in jobs}
    assert all(row["status"] == "completed" for row in completion["results"])
    groups, pen_records = defaultdict(list), {}
    raw_count = 0
    for job in jobs:
        mode, campaign = job["mode"], job["campaign"]
        assert mode in MODES
        member = None if campaign == "pendigits_v1" else f"{job['system']}_{job['seed']}"
        directory = study / "results" / mode / campaign
        if member is not None:
            directory /= member
        config_path = study / "frozen_configs" / mode / (campaign + ".json")
        cfg = tracked(config_path)
        assert sha(config_path) == job["config_sha256"]
        assert cfg["seeds"] == list(SEEDS)
        assert cfg["input_scaling_by_kind"] == {kind: mode for kind in GEOMETRIES}
        result_path = directory / "results.json"
        record = tracked(result_path)
        frozen = tracked(directory / "frozen_candidates.json")
        if member is None:
            pen_records[mode] = record
            frozen_by_member = {(row["seed"], row["coordinate"]):
                                {c["name"]: c for c in row["candidates"]} for row in frozen}
        else:
            frozen_by_member = {(job["seed"], -1): {c["name"]: c for c in frozen}}
        for index, candidate in enumerate(record["results"]):
            raw_count += 1
            seed = candidate["seed"] if member is None else job["seed"]
            coordinate = candidate["coordinate"] if member is None else -1
            task = "pendigits" if member is None else job["system"]
            assert candidate["validation_mse"] == frozen_by_member[(seed, coordinate)][candidate["name"]]["validation_mse"]
            groups[(task, seed, coordinate)].append({
                **candidate, "mode": mode, "task": task, "seed": seed, "coordinate": coordinate,
                "test_values": candidate["cohort_writer_mean_mse"] if member is None else candidate["test_mse"],
                "horizons": cfg["horizons"], "raw_result_index": index,
                "raw_result_relative_path": str(result_path.relative_to(study)),
            })
    assert raw_count == 13440
    expected_groups = {(task, seed, coordinate) for task in TASKS for seed in SEEDS
                       for coordinate in ((0, 1) if task == "pendigits" else (-1,))}
    assert set(groups) == expected_groups
    selected = {}
    for group, candidates in groups.items():
        assert Counter(c["mode"] for c in candidates) == Counter({mode: 84 for mode in MODES})
        for head in HEADS:
            for geometry in GEOMETRIES:
                eligible = [c for c in candidates if c["family"] == geometry + "_" + head]
                assert len(eligible) == 32
                # The recorded three-horizon mean uses ordinary left-to-right
                # float additions; no calibration or test value enters this key.
                choice = min(eligible, key=lambda c: (
                    sum(c["validation_mse"]) / len(c["validation_mse"]),
                    c["n_features"], c["history"], MODES.index(c["mode"]), c["name"]))
                selected[group + (head, geometry)] = choice
    assert len(selected) == 360

    table_path = analysis / "selected_candidates.csv"
    hashes[str(table_path)] = sha(table_path)
    selected_rows = [row for row in rows(table_path) if row["analysis"] == "primary"]
    assert len(selected_rows) == 1080
    seen = set()
    identities = []
    for row in selected_rows:
        key = (row["task"], int(row["seed"]), int(row["coordinate"]), row["head"], row["geometry"])
        choice = selected[key]
        horizon = int(row["horizon"])
        assert key + (horizon,) not in seen
        seen.add(key + (horizon,))
        assert (row["mode"], row["candidate"], int(row["history"]), int(row["n_features"])) == (
            choice["mode"], choice["name"], choice["history"], choice["n_features"])
        assert row["family"] == choice["family"]
        assert row["source_result_relative_path"] == choice["raw_result_relative_path"]
        hi = choice["horizons"].index(horizon)
        close(row["validation_mse"], choice["validation_mse"][hi], "selected_validation")
        close(row["test_mse"], choice["test_values"][hi], "selected_test")
        close(row["test_rmse"], math.sqrt(choice["test_values"][hi]), "selected_rmse")
    for key, choice in sorted(selected.items()):
        identities.append(dict(zip(("task", "seed", "coordinate", "head", "geometry"), key)) | {
            "mode": choice["mode"], "candidate": choice["name"], "history": choice["history"],
            "n_features": choice["n_features"], "raw_result_index": choice["raw_result_index"],
            "raw_result_relative_path": choice["raw_result_relative_path"]})

    wanted = {(c["mode"], c["seed"], c["coordinate"], c["name"]): c
              for c in selected.values() if c["task"] == "pendigits"}
    writer_losses = defaultdict(dict)
    for mode in MODES:
        path = study / "results" / mode / "pendigits_v1/writer_losses.csv"
        hashes[str(path)] = sha(path)
        for row in rows(path):
            key = (mode, int(row["seed"]), int(row["coordinate"]), row["candidate"])
            if key in wanted:
                wh = (int(row["writer"]), int(row["horizon"]))
                assert wh not in writer_losses[key]
                value = float(row["writer_mean_squared_loss"])
                assert 0 <= value <= 4 and math.isfinite(value)
                writer_losses[key][wh] = value
    for key, choice in wanted.items():
        expected_writers = sorted({record["writer"] for record in pen_records[key[0]]["held_out_record_ids"]})
        assert len(expected_writers) == 13
        assert set(writer_losses[key]) == {(writer, h) for writer in expected_writers for h in choice["horizons"]}
        for hi, h in enumerate(choice["horizons"]):
            close(mean(writer_losses[key][(writer, h)] for writer in expected_writers),
                  choice["test_values"][hi], "independent_writer_means")

    pooled = {}
    for task in TASKS:
        for seed in SEEDS:
            for head in HEADS:
                for geometry in GEOMETRIES:
                    coordinates = (0, 1) if task == "pendigits" else (-1,)
                    choices = [selected[(task, seed, coordinate, head, geometry)] for coordinate in coordinates]
                    per_horizon = [mean(c["test_values"][hi] for c in choices)
                                   for hi in range(len(choices[0]["horizons"]))]
                    key = (task, seed, head, geometry)
                    for h, value in zip(choices[0]["horizons"], per_horizon):
                        pooled[key + (str(h),)] = value
                    pooled[key + ("all",)] = mean(per_horizon)
                    if task == "pendigits":
                        writers = sorted({w for w, _ in writer_losses[(choices[0]["mode"], seed, 0, choices[0]["name"])]})
                        pooled_from_writers = mean(mean(
                            writer_losses[(c["mode"], seed, c["coordinate"], c["name"])][(w, h)]
                            for c in choices for h in c["horizons"]) for w in writers)
                        close(pooled[key + ("all",)], pooled_from_writers, "coordinate_and_writer_pooling")
    table_path = analysis / "selected_pooled.csv"
    hashes[str(table_path)] = sha(table_path)
    pooled_rows = [r for r in rows(table_path) if r["analysis"] == "primary"]
    assert len(pooled_rows) == len(pooled) == 1260
    seen = set()
    for row in pooled_rows:
        key = (row["task"], int(row["seed"]), row["head"], row["geometry"], row["horizon"])
        assert key not in seen
        seen.add(key)
        close(row["test_mse"], pooled[key], "pooled_mse")
        close(row["test_rmse"], math.sqrt(pooled[key]), "pooled_rmse")
    assert seen == set(pooled)

    effects = {}
    for task in TASKS:
        for head in HEADS:
            for alternative in ("rbf", "affine"):
                kahm = [pooled[(task, seed, head, "kahm", "all")] for seed in SEEDS]
                other = [pooled[(task, seed, head, alternative, "all")] for seed in SEEDS]
                differences = [a - b for a, b in zip(kahm, other)]
                km, am = mean(kahm), mean(other)
                effects[(task, head, alternative)] = {
                    "task": task, "head": head, "alternative": alternative,
                    "n_seeds": 5, "seeds": list(SEEDS), "kahm_mse": km, "alternative_mse": am,
                    "mse_difference_kahm_minus_alternative": mean(differences),
                    "relative_mse_change_percent": 100.0 * (km / am - 1.0),
                    "paired_seed_difference_sd": statistics.stdev(differences),
                    "n_seeds_improved": sum(x < -TOL for x in differences),
                    "n_seeds_degraded": sum(x > TOL for x in differences),
                    "n_seeds_tied": sum(abs(x) <= TOL for x in differences),
                    "seed_effects": differences, "effect_sign": classify(mean(differences)),
                }
    table_path = analysis / "taskwise_comparisons.csv"
    hashes[str(table_path)] = sha(table_path)
    reported = [row for row in rows(table_path) if row["analysis"] == "primary" and row["horizon"] == "all"]
    assert len(reported) == len(effects) == 42
    seen = set()
    for row in reported:
        key = (row["task"], row["head"], row["alternative"])
        assert key not in seen
        seen.add(key)
        expected = effects[key]
        for name in ("kahm_mse", "alternative_mse", "mse_difference_kahm_minus_alternative",
                     "relative_mse_change_percent", "paired_seed_difference_sd"):
            close(row[name], expected[name], "effect_" + name)
        for name in ("n_seeds", "n_seeds_improved", "n_seeds_degraded", "n_seeds_tied"):
            assert int(row[name]) == expected[name]
        assert row["effect_sign"] == expected["effect_sign"]
        assert json.loads(row["seeds"]) == list(SEEDS)
        reported_seed_effects = json.loads(row["seed_effects"])
        assert len(reported_seed_effects) == 5
        for a, b in zip(reported_seed_effects, expected["seed_effects"]):
            close(a, b, "seed_effects")
    assert seen == set(effects)
    ordered = [effects[key] for key in sorted(effects)]
    return {
        "status": "passed", "independence": "Standard-library-only reconstruction; no analyzer/runner imports",
        "absolute_tolerance": TOL, "raw_candidates": raw_count, "raw_jobs": len(jobs),
        "primary_choices": len(selected), "primary_choices_match_exactly": True,
        "selected_horizon_rows": len(selected_rows), "pooled_horizon_rows": len(pooled_rows),
        "primary_taskwise_effects": len(effects), "all_effect_signs_match": True,
        "numeric_comparisons": dict(checks), "maximum_absolute_errors": dict(errors),
        "overall_maximum_absolute_error": max(errors.values()),
        "effect_counts": dict(Counter(row["effect_sign"] for row in ordered)),
        "all_42_effects": ordered,
        "adverse_kahm_cases": [row for row in ordered if row["effect_sign"] == "degraded"],
        "selected_identities_and_raw_indices": identities,
        "input_sha256": hashes, "verification_script_sha256": sha(Path(__file__)),
        "scope": "Independent descriptive primary selection/effect check; it does not claim joint significance across tasks or new PenDigits writers",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Verification output must be new")
    result = verify(args.study.resolve(), args.analysis.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "primary_choices", "primary_taskwise_effects",
          "overall_maximum_absolute_error", "effect_counts")}, indent=2))
    for row in result["adverse_kahm_cases"]:
        print(row["task"], row["head"], "vs", row["alternative"],
              f"{row['relative_mse_change_percent']:+.6f}%", "degraded seeds", row["n_seeds_degraded"])


if __name__ == "__main__":
    main()
