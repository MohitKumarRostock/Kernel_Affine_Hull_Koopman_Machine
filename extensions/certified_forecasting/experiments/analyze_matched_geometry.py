#!/usr/bin/env python3
"""Audit and analyse the frozen, matched KAHM/RBF/affine forecasting study.

No fitting takes place here. Primary selection minimizes development-writer or
development-episode MSE over preprocessing, history and retained dimension,
separately within each geometry and head. Six geometry contrasts per synthetic
replicate receive the reserved 0.01 calibration failure budget. Fixed-dimension
and head-selected comparisons are descriptive. All outcomes are retained.

    python experiments/analyze_matched_geometry.py --study NEW_STUDY --output NEW_ANALYSIS

The default requires the complete seven-task, four-mode, five-seed protocol.
--allow-smoke permits reduced configurations for software checks, and labels
the outputs accordingly. Missing or corrupt requested jobs always fail.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certified_forecasting.certificates import mean_interval, paired_improvement

MODES = ("standard", "minmax", "minmax_clip", "raw")
GEOMETRIES = ("kahm", "rbf", "affine")
HEADS = ("nested", "frobenius", "koopman")
FAMILIES = tuple(f"{geometry}_{head}" for geometry in GEOMETRIES for head in HEADS)
CONTROLS = ("ridge_ar", "polynomial", "polynomial_koopman", "persistence", "constant")
TASKS = ("harmonic", "quasiperiodic", "nonlinear_ar", "logistic", "vanderpol", "duffing", "pendigits")
SEEDS = (5701, 5702, 5703, 5704, 5705)
CALIBRATION_DELTA = 0.01 / 6
TOL = 1e-12

NOTES = {
    "primary": "Within each geometry and head, select preprocessing, history and retained dimension by mean validation MSE only; compare KAHM against RBF and affine within that head.",
    "tie_break": "mean validation MSE; fewer features; shorter history; mode order standard,minmax,minmax_clip,raw; lexical candidate name",
    "fixed_dimension": "Descriptive matched-dimension comparisons: select only history and preprocessing within each geometry/head/dimension.",
    "head_selection": "Secondary descriptive comparison allowing head choice by the same validation rule within each geometry.",
    "mse_summary": "Average MSE over horizons and fitting seeds within each task; PenDigits first averages coordinate MSE and equally weights eligible held-out writers. No physical-error average across tasks.",
    "relative_effect": "100*(mean KAHM MSE / mean alternative MSE - 1); negative means KAHM improves. This is a ratio of seed means, not a mean of seed ratios.",
    "calibration": "Six one-sided all-horizon paired-improvement lower bounds per synthetic system-seed, each at delta=0.01/6. Comparison loss is alternative minus KAHM, bounded in [-4,4]. Positive lower bounds certify improvement of the fitted validation-selected KAHM predictor.",
    "confidence_scope": "Conditional on development data, four complete arm families cost 0.04 plus the six paired comparisons cost 0.01 within one synthetic system-seed replicate. No joint 95% statement across tasks or fitting seeds is claimed.",
    "descriptive_variation": "Paired seed effects and sample SD describe fitting/episode-stream variation; no seed-level p-values or population claims are produced.",
    "pendigits": "The same fixed held-out writer cohort is reused by fitting seeds. Coordinate-specific validation selection precedes equal-coordinate and equal-writer pooling. Paired writer effects are descriptive. The existing candidate upper bounds require the explicitly unverified iid-new-writer assumption, with 0.0125 per mode and 0.05 over all modes, seeds, coordinates and candidate horizons.",
    "dimensions": "Equal feature count and equal numeric readout budgets do not imply equal prediction classes, runtime, storage cost or capacity across geometries or heads.",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def asset_path(study, value):
    """Resolve original absolute schedule paths or a relocated evidence tree."""
    path = Path(value)
    if not path.is_absolute():
        path = study / path
    for marker in ("results", "frozen_configs", "logs"):
        if marker in path.parts:
            relocated = study.joinpath(*path.parts[path.parts.index(marker):])
            if relocated.exists():
                return relocated
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing frozen study asset: {value}")


def select_candidate(candidates):
    """This function deliberately reads no calibration/test field."""
    if not candidates:
        raise ValueError("Cannot select from an empty candidate family")
    return min(candidates, key=lambda c: (
        float(np.mean(c["validation_mse"])), int(c["n_features"]), int(c["history"]),
        MODES.index(c["mode"]), c["name"],
    ))


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def sign(value):
    return "improved" if value < -TOL else "degraded" if value > TOL else "tie_within_1e-12"


def paired_seed_summary(kahm, alternative):
    kahm, alternative = np.asarray(kahm, float), np.asarray(alternative, float)
    if kahm.ndim != 1 or kahm.shape != alternative.shape or not len(kahm):
        raise ValueError("Seed observations must be paired one-dimensional arrays")
    if not np.isfinite(kahm).all() or not np.isfinite(alternative).all():
        raise ValueError("Non-finite losses")
    difference = kahm - alternative
    kmean, amean = float(kahm.mean()), float(alternative.mean())
    return {
        "n_seeds": len(kahm), "kahm_mse": kmean, "alternative_mse": amean,
        "mse_difference_kahm_minus_alternative": float(difference.mean()),
        "relative_mse_change_percent": 100.0 * (kmean / amean - 1.0) if amean > 0 else None,
        "paired_seed_difference_sd": sample_sd(difference),
        "n_seeds_improved": int(np.sum(difference < -TOL)),
        "n_seeds_degraded": int(np.sum(difference > TOL)),
        "n_seeds_tied": int(np.sum(np.abs(difference) <= TOL)),
        "seed_effects": difference.tolist(), "effect_sign": sign(float(difference.mean())),
    }


def assert_close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-12):
        raise ValueError(f"Stored score aggregation differs: {label}")


def validate_candidate_grid(candidates, config, mode):
    """Reject missing geometry/head/dimension cells and unequal scaler fits."""
    dimensions = np.cumsum(config["cluster_counts"]).tolist()
    expected = {(family, history, dimension) for family in FAMILIES
                for history in config["histories"] for dimension in dimensions}
    selected = [c for c in candidates if c["family"] in FAMILIES]
    actual = [(c["family"], c["history"], c["n_features"]) for c in selected]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Incomplete, duplicated or unmatched geometry/head/dimension grid")
    for history in config["histories"]:
        scalers = [c["input_scaling"] for c in selected if c["history"] == history]
        if any(scaler["mode"] != mode or scaler != scalers[0] for scaler in scalers):
            raise ValueError("The matched geometries did not use identical frozen input maps")
    expected_count = len(expected) + 5 * len(config["histories"]) + 2
    if len(candidates) != expected_count or len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Candidate schedule has a missing or duplicate control")
    return dimensions


def load_study(study, allow_smoke=False):
    manifest = read_json(study / "manifest.json")
    jobs = read_json(study / "schedule.json")
    if not jobs or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Empty or duplicated frozen job schedule")
    hashes = {"manifest.json": digest(study / "manifest.json"),
              "schedule.json": digest(study / "schedule.json")}
    candidates, archives, writer_arrays, definitions = [], {}, {}, {}
    seen_jobs, audit_counts = set(), defaultdict(int)
    control_reference, scaler_reference, training_reference, config_reference = {}, {}, {}, {}
    cohort_reference = None

    def record_hash(path):
        try:
            label = str(path.relative_to(study))
        except ValueError:
            label = str(path)
        hashes[label] = digest(path)

    for job in jobs:
        mode, campaign = job["mode"], job["campaign"]
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        cfg_path = asset_path(study, job["config_file"])
        cfg = read_json(cfg_path)
        if digest(cfg_path) != job["config_sha256"]:
            raise ValueError("Frozen configuration hash changed")
        record_hash(cfg_path)
        if not cfg.get("matched_geometry_heads"):
            raise ValueError("This is not a matched-head experiment")
        if cfg.get("input_scaling_by_kind") != dict.fromkeys(GEOMETRIES, mode):
            raise ValueError("Preprocessing choices are unequal across geometries")
        common_config = {k: v for k, v in cfg.items() if k != "input_scaling_by_kind"}
        if campaign in config_reference and config_reference[campaign] != common_config:
            raise ValueError("A non-preprocessing configuration changed between arms")
        config_reference.setdefault(campaign, common_config)
        if not allow_smoke and (tuple(cfg["seeds"]) != SEEDS or cfg["cluster_counts"] != [2, 4, 8, 16]):
            raise ValueError("The complete five-seed/four-prefix schedule is required")
        destination = asset_path(study, job["destination"])
        if (destination / "failure.json").exists():
            raise ValueError(f"Failed job remains in the requested study: {job['id']}")
        result_path, frozen_path = destination / "results.json", destination / "frozen_candidates.json"
        record, frozen = read_json(result_path), read_json(frozen_path)
        record_hash(result_path)
        record_hash(frozen_path)
        if campaign == "pendigits_v1":
            if not math.isclose(cfg["delta"], 0.0125, abs_tol=1e-15):
                raise ValueError("PenDigits arm does not use the frozen 0.0125 budget")
            if set(cfg["coordinates"]) != {0, 1}:
                raise ValueError("Both PenDigits coordinates are required")
            members = {(seed, coordinate): [] for seed in cfg["seeds"] for coordinate in cfg["coordinates"]}
            for candidate in record["results"]:
                members[(candidate["seed"], candidate["coordinate"])].append(candidate)
            frozen_by_member = {(f["seed"], f["coordinate"]): f for f in frozen}
            if set(frozen_by_member) != set(members):
                raise ValueError("PenDigits frozen family is incomplete")
            total_statements = sum(len(cs) * len(cfg["horizons"]) for cs in members.values())
            if record["n_upper_bound_statements"] != total_statements:
                raise ValueError("PenDigits simultaneous statement count changed")
            assert_close(record["per_statement_delta"], cfg["delta"] / total_statements, "PenDigits tail allocation")
            cohort = record["held_out_record_ids"]
            if cohort_reference is not None and cohort != cohort_reference:
                raise ValueError("Preprocessing arms used different held-out PenDigits records")
            cohort_reference = cohort
            writer_path = destination / "writer_losses.csv"
            record_hash(writer_path)
            values = defaultdict(dict)
            with writer_path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    key = (int(row["seed"]), int(row["coordinate"]), row["candidate"])
                    coordinate_key = (int(row["writer"]), int(row["horizon"]))
                    if coordinate_key in values[key]:
                        raise ValueError("Duplicate writer loss")
                    values[key][coordinate_key] = float(row["writer_mean_squared_loss"])
            for (seed, coordinate), cs in members.items():
                dimensions = validate_candidate_grid(cs, cfg, mode)
                frozen_candidates = {c["name"]: c for c in frozen_by_member[(seed, coordinate)]["candidates"]}
                training_ids = frozen_by_member[(seed, coordinate)]["training_record_ids"]
                if seed in training_reference and training_reference[seed] != training_ids:
                    raise ValueError("PenDigits fitting arms used different training-stroke subsamples")
                training_reference.setdefault(seed, training_ids)
                for candidate in cs:
                    if candidate["validation_mse"] != frozen_candidates[candidate["name"]]["validation_mse"]:
                        raise ValueError("PenDigits validation losses changed after candidate freeze")
                    key = (seed, coordinate, candidate["name"])
                    writer_ids = sorted({writer for writer, _ in values[key]})
                    if len(writer_ids) != record["n_held_out_writers"]:
                        raise ValueError("Missing eligible writer scores")
                    if set(values[key]) != {(w, h) for w in writer_ids for h in cfg["horizons"]}:
                        raise ValueError("Incomplete writer/horizon score grid")
                    array = np.array([[values[key][(w, h)] for h in cfg["horizons"]] for w in writer_ids])
                    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 4):
                        raise ValueError("Writer scores violate scalar loss bounds")
                    assert_close(array.mean(axis=0), candidate["cohort_writer_mean_mse"], "PenDigits writer means")
                    if candidate["family"] in CONTROLS:
                        control_key = ("pendigits", seed, coordinate, candidate["name"])
                        if control_key in control_reference and not np.array_equal(array, control_reference[control_key]):
                            raise ValueError("Ordinary PenDigits writer scores changed between preprocessing arms")
                        control_reference.setdefault(control_key, array.copy())
                    uid = ("pendigits", seed, coordinate, mode, candidate["name"])
                    writer_arrays[uid] = (writer_ids, array)
                    c = {**candidate, "task": "pendigits", "seed": seed, "coordinate": coordinate,
                         "mode": mode, "test_mse": candidate["cohort_writer_mean_mse"],
                         "horizons": cfg["horizons"], "uid": uid, "campaign": campaign,
                         "source_result_relative_path": str(result_path.relative_to(study)),
                         "source_archive_relative_path": str(writer_path.relative_to(study))}
                    candidates.append(c)
                member_key = ("pendigits", seed, coordinate, mode)
                if member_key in seen_jobs:
                    raise ValueError("Duplicate underlying PenDigits fit in schedule")
                seen_jobs.add(member_key)
                definitions[("pendigits", seed, coordinate)] = (tuple(dimensions), tuple(cfg["horizons"]))
            audit_counts["writer_loss_values"] += sum(len(v) for v in values.values())
        else:
            if not math.isclose(cfg["delta"], 0.01, abs_tol=1e-15):
                raise ValueError("Synthetic arm does not use the frozen 0.01 budget")
            task, seed = job["system"], job["seed"]
            if record["system"] != task or record["seed"] != seed:
                raise ValueError("Synthetic identity differs from schedule")
            streams = {"train": seed * 100 + 11, "validation": seed * 100 + 23,
                       "calibration": seed * 100 + 47, "test": seed * 100 + 89}
            if record["streams"] != streams:
                raise ValueError("Synthetic independent stream schedule differs")
            cs = record["results"]
            dimensions = validate_candidate_grid(cs, cfg, mode)
            expected_tail = cfg["delta"] * cfg["risk_delta_fraction"] / (2 * len(cs) * len(cfg["horizons"]))
            assert_close(record["tail_delta"], expected_tail, "synthetic family tail allocation")
            frozen_candidates = {c["name"]: c for c in frozen}
            if set(frozen_candidates) != {c["name"] for c in cs}:
                raise ValueError("Scored candidates differ from frozen candidate family")
            archive = destination / "episode_scores.npz"
            record_hash(archive)
            with zipfile.ZipFile(archive) as zipstream:
                bad = zipstream.testzip()
                if bad is not None:
                    raise ValueError(f"Archive CRC failure: {archive}: {bad}")
            expected_keys = {c["name"] + "__" + role for c in cs
                             for role in ("cal_loss", "pair_score", "test_loss")}
            with np.load(archive, allow_pickle=False) as data:
                if set(data.files) != expected_keys:
                    raise ValueError("Score archive has missing or extra candidates/roles")
                for candidate in cs:
                    if candidate["validation_mse"] != frozen_candidates[candidate["name"]]["validation_mse"]:
                        raise ValueError("Validation losses changed after candidate freeze")
                    for role, count, bound in (("cal_loss", cfg["n_calibration"], 4),
                                               ("pair_score", cfg["n_calibration"] // 2, 2),
                                               ("test_loss", cfg["n_test"], 4)):
                        array = data[candidate["name"] + "__" + role]
                        if array.dtype != np.float64 or array.shape != (count, len(cfg["horizons"])) or not np.isfinite(array).all():
                            raise ValueError("Score shape or finite-value check failed")
                        if np.any(array < -TOL) or np.any(array > bound + TOL):
                            raise ValueError("Episode scores violate the declared population range")
                        if role == "test_loss":
                            assert_close(array.mean(axis=0), candidate["test_mse"], "test means")
                        else:
                            kind = "upper" if role == "cal_loss" else "lower"
                            for hindex, certificate in enumerate(candidate["certificates"]):
                                expected = mean_interval(array[:, hindex], bound, record["tail_delta"], cfg["certificate_method"])
                                assert_close(expected[kind], certificate[f"mse_{kind}"], f"{kind} certificate")
                        if candidate["family"] in CONTROLS:
                            key = (task, seed, candidate["name"], role)
                            if key in control_reference and not np.array_equal(array, control_reference[key]):
                                raise ValueError("Ordinary control scores changed between preprocessing arms")
                            control_reference.setdefault(key, array.copy())
                        audit_counts["episode_score_arrays"] += 1
                        audit_counts["episode_score_values"] += array.size
                    c = {**candidate, "task": task, "seed": seed, "coordinate": -1,
                         "mode": mode, "horizons": cfg["horizons"],
                         "uid": (task, seed, -1, mode, candidate["name"]), "campaign": campaign,
                         "source_result_relative_path": str(result_path.relative_to(study)),
                         "source_archive_relative_path": str(archive.relative_to(study))}
                    candidates.append(c)
            archives[(task, seed, mode)] = archive
            member_key = (task, seed, -1, mode)
            if member_key in seen_jobs:
                raise ValueError("Duplicate underlying synthetic fit in schedule")
            seen_jobs.add(member_key)
            definitions[(task, seed, -1)] = (tuple(dimensions), tuple(cfg["horizons"]))
        audit_counts["jobs"] += 1

    modes = set(j["mode"] for j in jobs)
    tasks = set(c["task"] for c in candidates)
    seeds = set(c["seed"] for c in candidates)
    if not allow_smoke and (modes != set(MODES) or tasks != set(TASKS) or seeds != set(SEEDS)):
        raise ValueError("Missing required modes, tasks or fitting seeds")
    expected_jobs = {(task, seed, coordinate, mode) for task in tasks for seed in seeds
                     for coordinate in ((0, 1) if task == "pendigits" else (-1,)) for mode in modes}
    if seen_jobs != expected_jobs:
        raise ValueError("Incomplete Cartesian task/seed/coordinate/mode schedule")
    grouped = defaultdict(list)
    for c in candidates:
        grouped[(c["task"], c["seed"], c["coordinate"])].append(c)
        if c["family"] in CONTROLS:
            key = (c["task"], c["seed"], c["coordinate"], c["name"])
            vals = (c["validation_mse"], c["test_mse"])
            if key in scaler_reference and scaler_reference[key] != vals:
                raise ValueError("Ordinary control validation/test means changed across modes")
            scaler_reference.setdefault(key, vals)
    audit = {"status": "passed", "allow_smoke": allow_smoke, "counts": dict(audit_counts),
             "n_candidates": len(candidates), "modes": sorted(modes), "tasks": sorted(tasks),
             "seeds": sorted(seeds), "input_sha256": hashes,
             "checks": ["complete frozen schedule", "matched head/history/dimension grid",
                        "identical training scaler per history and mode across geometries",
                        "frozen validation losses", "archive CRC and all arrays",
                        "bounded scores", "recomputed synthetic certificate endpoints",
                        "ordinary control invariance across modes", "equal-writer loss aggregation"]}
    return manifest, grouped, archives, writer_arrays, definitions, audit


def make_selections(grouped, definitions):
    selections = []
    for key, cs in sorted(grouped.items()):
        dimensions, _ = definitions[key]
        for head in HEADS:
            for geometry in GEOMETRIES:
                family = f"{geometry}_{head}"
                eligible = [c for c in cs if c["family"] == family]
                selections.append({"analysis": "primary", "head": head, "dimension": "selected",
                                   "geometry": geometry, "candidate": select_candidate(eligible)})
                for dimension in dimensions:
                    selections.append({"analysis": "fixed_dimension", "head": head,
                                       "dimension": dimension, "geometry": geometry,
                                       "candidate": select_candidate([c for c in eligible if c["n_features"] == dimension])})
        for geometry in GEOMETRIES:
            selections.append({"analysis": "head_selected", "head": "selected", "dimension": "selected",
                               "geometry": geometry,
                               "candidate": select_candidate([c for c in cs if c["family"] in
                                                               [f"{geometry}_{head}" for head in HEADS]])})
        for family in (*CONTROLS, "kahm_hard", "kahm_duplicate"):
            selections.append({"analysis": "control", "head": family, "dimension": "selected",
                               "geometry": family, "candidate": select_candidate([c for c in cs if c["family"] == family])})
    return selections


def selection_rows(selections):
    rows = []
    for s in selections:
        c = s["candidate"]
        for hi, horizon in enumerate(c["horizons"]):
            rows.append({k: v for k, v in s.items() if k != "candidate"} | {
                "task": c["task"], "seed": c["seed"], "coordinate": c["coordinate"], "mode": c["mode"],
                "candidate": c["name"], "family": c["family"], "history": c["history"],
                "selection_scope": s["analysis"], "dimension_if_fixed": s["dimension"] if s["analysis"] == "fixed_dimension" else None,
                "campaign": c["campaign"], "source_result_relative_path": c["source_result_relative_path"],
                "source_archive_relative_path": c["source_archive_relative_path"],
                "n_features": c["n_features"], "horizon": horizon,
                "validation_mse": c["validation_mse"][hi], "test_mse": c["test_mse"][hi],
                "test_rmse": math.sqrt(c["test_mse"][hi]), "train_seconds": c.get("train_seconds"),
                "optimization_gap": c.get("optimization_gap"), "converged": c.get("converged"),
            })
    return rows


def pool_selections(selections):
    groups = defaultdict(list)
    for s in selections:
        c = s["candidate"]
        key = (c["task"], c["seed"], s["analysis"], s["head"], s["dimension"], s["geometry"])
        groups[key].append(c)
    pooled, members = [], {}
    for key, cs in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        task, seed, analysis, head, dimension, geometry = key
        if {c["coordinate"] for c in cs} != ({0, 1} if task == "pendigits" else {-1}):
            raise ValueError("Coordinate pooling is incomplete")
        if len({tuple(c["horizons"]) for c in cs}) != 1:
            raise ValueError("Coordinates have different forecast horizons")
        test = np.array([c["test_mse"] for c in cs]).mean(axis=0)
        validation = np.array([c["validation_mse"] for c in cs]).mean(axis=0)
        base = dict(zip(("task", "seed", "analysis", "head", "dimension", "geometry"), key))
        base.update(n_features_mean=float(np.mean([c["n_features"] for c in cs])),
                    selected_models=[{"coordinate": c["coordinate"], "mode": c["mode"], "candidate": c["name"],
                                      "n_features": c["n_features"], "history": c["history"]} for c in cs])
        for hi, horizon in enumerate(cs[0]["horizons"]):
            pooled.append({**base, "horizon": horizon, "validation_mse": float(validation[hi]),
                           "test_mse": float(test[hi]), "test_rmse": math.sqrt(test[hi])})
        pooled.append({**base, "horizon": "all", "validation_mse": float(validation.mean()),
                       "test_mse": float(test.mean()), "test_rmse": math.sqrt(float(test.mean()))})
        members[key] = cs
    return pooled, members


def summarize_pooled(pooled):
    groups = defaultdict(list)
    for row in pooled:
        key = tuple(row[k] for k in ("task", "analysis", "head", "dimension", "geometry", "horizon"))
        groups[key].append(row)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        rows.append(dict(zip(("task", "analysis", "head", "dimension", "geometry", "horizon"), key)) | {
            "n_seeds": len(rs), "mse_mean": float(np.mean([r["test_mse"] for r in rs])),
            "mse_seed_sd": sample_sd([r["test_mse"] for r in rs]),
            "rmse_pooled": math.sqrt(float(np.mean([r["test_mse"] for r in rs]))),
            "rmse_seed_mean": float(np.mean([r["test_rmse"] for r in rs])),
            "rmse_seed_sd": sample_sd([r["test_rmse"] for r in rs]),
            "feature_count_mean": float(np.mean([r["n_features_mean"] for r in rs])),
        })
    return rows


def compare(pooled, members, archives, writer_arrays):
    groups = defaultdict(dict)
    for row in pooled:
        if row["analysis"] == "control":
            continue
        key = tuple(row[k] for k in ("task", "seed", "analysis", "head", "dimension", "horizon"))
        groups[key][row["geometry"]] = row
    contrasts, certificates, writer_rows = [], [], []
    cache = {}

    def scores(c, role):
        if c["task"] == "pendigits":
            return writer_arrays[c["uid"]][1]
        key = (*c["uid"], role)
        if key not in cache:
            with np.load(archives[(c["task"], c["seed"], c["mode"])], allow_pickle=False) as data:
                cache[key] = data[c["name"] + "__" + role]
        return cache[key]

    for key, rs in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        if set(rs) != set(GEOMETRIES):
            raise ValueError("Missing geometry in a requested paired comparison")
        task, seed, analysis, head, dimension, horizon = key
        for alternative in ("rbf", "affine"):
            kmse, amse = rs["kahm"]["test_mse"], rs[alternative]["test_mse"]
            row = dict(zip(("task", "seed", "analysis", "head", "dimension", "horizon"), key)) | {
                "alternative": alternative, "kahm_mse": kmse, "alternative_mse": amse,
                "mse_difference_kahm_minus_alternative": kmse - amse,
                "relative_mse_change_percent": 100 * (kmse / amse - 1) if amse > 0 else None,
                "effect_sign": sign(kmse - amse), "kahm_models": rs["kahm"]["selected_models"],
                "alternative_models": rs[alternative]["selected_models"],
            }
            contrasts.append(row)
            if analysis != "primary" or horizon != "all":
                continue
            kcs = members[(task, seed, analysis, head, dimension, "kahm")]
            acs = members[(task, seed, analysis, head, dimension, alternative)]
            if task == "pendigits":
                kcs, acs = sorted(kcs, key=lambda c: c["coordinate"]), sorted(acs, key=lambda c: c["coordinate"])
                ids = writer_arrays[kcs[0]["uid"]][0]
                if any(writer_arrays[c["uid"]][0] != ids for c in kcs + acs):
                    raise ValueError("Paired PenDigits candidates use different writer orders")
                kloss = np.mean([scores(c, "test_loss") for c in kcs], axis=0).mean(axis=1)
                aloss = np.mean([scores(c, "test_loss") for c in acs], axis=0).mean(axis=1)
                for writer, kl, al in zip(ids, kloss, aloss):
                    writer_rows.append({"seed": seed, "head": head, "alternative": alternative,
                                        "writer": writer, "kahm_mse": float(kl), "alternative_mse": float(al),
                                        "mse_difference_kahm_minus_alternative": float(kl - al)})
                assert_close(kloss.mean(), kmse, "pooled PenDigits KAHM MSE")
                assert_close(aloss.mean(), amse, "pooled PenDigits alternative MSE")
            else:
                kloss = scores(kcs[0], "cal_loss").mean(axis=1)
                aloss = scores(acs[0], "cal_loss").mean(axis=1)
                certificate = paired_improvement(aloss, kloss, 4.0, CALIBRATION_DELTA, "bernstein")
                certificates.append({"task": task, "seed": seed, "head": head, "alternative": alternative,
                                     "kahm_mode": kcs[0]["mode"], "kahm_candidate": kcs[0]["name"],
                                     "alternative_mode": acs[0]["mode"], "alternative_candidate": acs[0]["name"],
                                     **certificate, "certified_kahm_improvement": certificate["improvement_lower"] > 0})
    summary_groups = defaultdict(list)
    for row in contrasts:
        key = tuple(row[k] for k in ("task", "analysis", "head", "dimension", "horizon", "alternative"))
        summary_groups[key].append(row)
    summaries = []
    for key, rs in sorted(summary_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        rs = sorted(rs, key=lambda r: r["seed"])
        summaries.append(dict(zip(("task", "analysis", "head", "dimension", "horizon", "alternative"), key)) | {
            **paired_seed_summary([r["kahm_mse"] for r in rs], [r["alternative_mse"] for r in rs]),
            "seeds": [r["seed"] for r in rs],
        })
    return contrasts, summaries, certificates, writer_rows


def analyse(study, output, allow_smoke=False):
    if output.exists():
        raise ValueError("Analysis output must be a new directory")
    manifest, grouped, archives, writer_arrays, definitions, audit = load_study(study, allow_smoke)
    selections = make_selections(grouped, definitions)
    pooled, members = pool_selections(selections)
    contrasts, summary_contrasts, certificates, writer_rows = compare(pooled, members, archives, writer_arrays)
    all_candidates = []
    for cs in grouped.values():
        for c in cs:
            for hi, horizon in enumerate(c["horizons"]):
                all_candidates.append({k: c.get(k) for k in ("task", "seed", "coordinate", "mode", "family", "history", "n_features",
                                                            "train_seconds", "optimization_gap", "converged")} | {
                    "candidate": c["name"], "horizon": horizon,
                    "validation_mse": c["validation_mse"][hi], "test_mse": c["test_mse"][hi],
                })
    tables = {"all_candidates.csv": all_candidates, "selected_candidates.csv": selection_rows(selections),
              "selected_pooled.csv": pooled, "taskwise_models.csv": summarize_pooled(pooled),
              "paired_effects.csv": contrasts, "taskwise_comparisons.csv": summary_contrasts,
              "calibration_comparisons.csv": certificates, "pendigits_writer_differences.csv": writer_rows}
    primary = [r for r in summary_contrasts if r["analysis"] == "primary" and r["horizon"] == "all"]
    summary = {"schema": "matched-geometry-analysis-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "allow_smoke": allow_smoke, "notes": NOTES, "n_candidates": audit["n_candidates"],
               "primary_taskwise_comparisons": primary,
               "n_primary_taskwise_comparisons": len(primary),
               "n_primary_mean_improved": sum(r["effect_sign"] == "improved" for r in primary),
               "n_primary_mean_degraded": sum(r["effect_sign"] == "degraded" for r in primary),
               "n_calibration_comparisons": len(certificates),
               "n_positive_calibration_lower_bounds": sum(r["certified_kahm_improvement"] for r in certificates),
               "positive_calibration_comparisons": [r for r in certificates if r["certified_kahm_improvement"]],
               "original_manifest_sha256": digest(study / "manifest.json"),
               "analysis_source_sha256": digest(Path(__file__)),
               "table_rows": {name: len(rows) for name, rows in tables.items()}}
    output.mkdir(parents=True)
    for name, rows in tables.items():
        write_csv(output / name, rows)
    index = {}
    for selection in selections:
        c = selection["candidate"]
        if selection["analysis"] not in ("primary", "head_selected") or c["task"] == "pendigits":
            continue
        key = c["source_archive_relative_path"] + "::" + c["name"]
        item = index.setdefault(key, {
            "task": c["task"], "seed": c["seed"], "mode": c["mode"], "candidate": c["name"],
            "family": c["family"], "history": c["history"], "n_features": c["n_features"],
            "source_archive_relative_path": c["source_archive_relative_path"],
            "source_result_relative_path": c["source_result_relative_path"],
            "array_keys": [c["name"] + "__" + role for role in ("cal_loss", "pair_score", "test_loss")],
            "selection_scopes": [],
        })
        item["selection_scopes"].append(selection["analysis"])
    write_json(output / "selected_score_index.json", list(index.values()))
    write_json(output / "summary.json", summary)
    write_json(output / "audit.json", audit)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    summary = analyse(args.study.resolve(), args.output.resolve(), args.allow_smoke)
    print(json.dumps({key: summary[key] for key in ("n_candidates", "n_primary_taskwise_comparisons",
                                                   "n_primary_mean_improved", "n_primary_mean_degraded",
                                                   "n_calibration_comparisons", "n_positive_calibration_lower_bounds")}, indent=2))


if __name__ == "__main__":
    main()
