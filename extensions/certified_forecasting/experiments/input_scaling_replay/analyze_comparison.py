#!/usr/bin/env python3
"""Independent, read-only analysis of the KAHM input-scaling comparison.

This script reads the original forecasting/PenDigits result schemas. It never
imports an experiment runner, fits a predictor, changes a manuscript, or writes
to the source repository. Primary contrast: minmax versus standard. Other arms
remain explicitly diagnostic. Candidate selection uses validation MSE only.

Usage after all requested runs finish::

    python scaling-investigation/analyze_comparison.py

Subset analyses require an explicit --campaigns and/or --modes argument; missing
requested repetitions cause an error rather than a favorable incomplete table.
Run --self-test before performance data are available to check arithmetic and
selection on artificial fixtures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
import sys
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "kahm-study/extensions/certified_forecasting"
SYNTHETIC_CAMPAIGNS = (
    "main_v1", "oscillators_v1", "sensitivity_v1_k1",
    "sensitivity_v1_k4", "sensitivity_v1_k16",
)
ALL_CAMPAIGNS = (*SYNTHETIC_CAMPAIGNS, "pendigits_v1")
DEFAULT_MODES = ("standard", "minmax", "minmax_clip", "raw")
ATOL, RTOL = 1e-12, 1e-11
Z95 = NormalDist().inv_cdf(0.975)
TIME_FIELDS = {"train_seconds", "fit_seconds", "elapsed_seconds"}
NOTES = {
    "primary_contrast": "minmax minus standard; negative MSE differences mean improvement",
    "other_contrasts": "minmax_clip and raw are diagnostic arms, not test-selected replacements",
    "selection": "Within each arm, family, seed, and (for PenDigits) coordinate, minimize validation MSE averaged across horizons. Main campaign uses original stable first-minimum tie behavior; other campaigns use feature count then name as tiebreaks.",
    "primary_outcome": "Arithmetic mean of test MSE across horizons and fitting seeds, reported separately per task. No cross-generator physical-error average is asserted.",
    "rmse_summary": "Per-horizon mean and sample SD of replicate RMSE; PenDigits pools coordinate MSE before the square root.",
    "synthetic_intervals": "Unadjusted approximate normal 95% intervals for paired mean test-loss differences, conditional on the fitted, validation-selected models. They are descriptive Monte Carlo uncertainty, not the manuscript's finite-sample certificates, and do not include training randomness.",
    "seed_variation": "All seed effect signs and sample SDs are retained. A separate seed SD summarizes complete-learning variation; it is not interchangeable with the conditional episode standard error.",
    "pendigits": "Equal-weighted eligible writers in the fixed held-out cohort; average coordinate MSE before RMSE. Writer differences are descriptive. No iid-writer population interval is computed. Repeated fitting seeds share the same test cohort.",
    "audit": "Original baseline replay and non-KAHM invariance exclude only timing fields. Both exact equality and tolerance agreement are recorded; maximum discrepancies are retained by metric.",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value
                             for key, value in row.items()})


def select_candidate(candidates, campaign):
    """Match published selection without consulting calibration or test losses."""
    if not candidates:
        raise ValueError("Cannot select from an empty family")
    if campaign == "main_v1":
        return min(candidates, key=lambda c: float(np.mean(c["validation_mse"])))
    return min(candidates, key=lambda c: (float(np.mean(c["validation_mse"])),
                                          c["n_features"], c["name"]))


def sign_name(difference):
    if abs(difference) <= ATOL:
        return "tie_within_1e-12"
    return "improved" if difference < 0 else "degraded"


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def paired_statistics(baseline, alternative, inferential=True):
    """Reduce independent units only; horizons must be averaged before this call."""
    baseline, alternative = np.asarray(baseline, float), np.asarray(alternative, float)
    if baseline.shape != alternative.shape or baseline.ndim != 1 or len(baseline) < 2:
        raise ValueError("Paired observations must be equal-length 1D arrays with n>=2")
    if not (np.isfinite(baseline).all() and np.isfinite(alternative).all()):
        raise ValueError("Non-finite paired losses")
    difference = alternative - baseline
    effect = float(np.mean(difference))
    standard_error = float(np.std(difference, ddof=1) / math.sqrt(len(difference))) if inferential else None
    return {
        "n_paired_units": len(difference),
        "standard_mse": float(np.mean(baseline)),
        "alternative_mse": float(np.mean(alternative)),
        "mse_difference_alternative_minus_standard": effect,
        "relative_mse_change_percent": 100 * effect / float(np.mean(baseline)) if np.mean(baseline) > 0 else None,
        "effect_sign": sign_name(effect),
        "paired_unit_difference_sd": sample_sd(difference),
        "conditional_standard_error": standard_error,
        "conditional_normal_95_lower": effect - Z95 * standard_error if inferential else None,
        "conditional_normal_95_upper": effect + Z95 * standard_error if inferential else None,
        "n_units_improved": int(np.sum(difference < -ATOL)),
        "n_units_degraded": int(np.sum(difference > ATOL)),
        "n_units_tied": int(np.sum(np.abs(difference) <= ATOL)),
    }


class Audit:
    """Independent recursive comparison retaining metric-level discrepancies."""
    def __init__(self):
        self.metrics = defaultdict(lambda: {"n_numeric_values": 0, "n_exact_differences": 0,
                                            "n_outside_tolerance": 0, "max_absolute_difference": 0.,
                                            "max_relative_difference": 0.})
        self.structural_mismatches = 0
        self.scalar_checks = 0
        self.examples = []

    def mismatch(self, where, message):
        self.structural_mismatches += 1
        if len(self.examples) < 30:
            self.examples.append({"where": where, "issue": message})

    def compare(self, observed, expected, where="", metric=""):
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                self.mismatch(where, "Expected mapping")
                return
            ek, ok = set(expected) - TIME_FIELDS, set(observed) - TIME_FIELDS
            if ek != ok:
                self.mismatch(where, f"Mapping keys differ: missing={sorted(ek-ok)}, extra={sorted(ok-ek)}")
            for key in sorted(ek & ok):
                self.compare(observed[key], expected[key], where + "/" + key, metric + "/" + key)
        elif isinstance(expected, (list, tuple)):
            if not isinstance(observed, (list, tuple)):
                self.mismatch(where, "Expected sequence")
                return
            if len(observed) != len(expected):
                self.mismatch(where, "Sequence lengths differ")
            for index, (left, right) in enumerate(zip(observed, expected)):
                self.compare(left, right, f"{where}/{index}", metric + "/*")
        elif isinstance(expected, (int, float, np.number)) and not isinstance(expected, (bool, np.bool_)):
            self.compare_array(np.asarray(observed), np.asarray(expected), where, metric)
        else:
            self.scalar_checks += 1
            if observed != expected:
                self.mismatch(where, f"Scalar values differ: {observed!r} versus {expected!r}")

    def compare_array(self, observed, expected, where, metric):
        try:
            left, right = np.asarray(observed, float), np.asarray(expected, float)
        except (ValueError, TypeError):
            self.mismatch(where, "Incompatible numeric data")
            return
        if left.shape != right.shape:
            self.mismatch(where, f"Array shapes differ: {left.shape} vs {right.shape}")
            return
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            self.mismatch(where, "Non-finite numeric data")
            return
        difference = np.abs(left - right)
        relative = difference / np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)
        outside = difference > ATOL + RTOL * np.abs(right)
        item = self.metrics[metric.lstrip("/")]
        item["n_numeric_values"] += int(difference.size)
        item["n_exact_differences"] += int(np.count_nonzero(difference))
        item["n_outside_tolerance"] += int(np.count_nonzero(outside))
        item["max_absolute_difference"] = max(item["max_absolute_difference"], float(np.max(difference, initial=0)))
        item["max_relative_difference"] = max(item["max_relative_difference"], float(np.max(relative, initial=0)))
        if np.any(outside) and len(self.examples) < 30:
            self.examples.append({"where": where, "issue": "Numerical difference outside tolerance",
                                  "maximum_absolute_difference": float(np.max(difference))})

    def result(self):
        metrics = dict(sorted(self.metrics.items()))
        outside = sum(x["n_outside_tolerance"] for x in metrics.values())
        exact = sum(x["n_exact_differences"] for x in metrics.values())
        return {
            "status": "mismatch" if self.structural_mismatches or outside else "exact" if exact == 0 else "within_tolerance",
            "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
            "excluded_fields": sorted(TIME_FIELDS), "scalar_checks": self.scalar_checks,
            "n_numeric_values": sum(x["n_numeric_values"] for x in metrics.values()),
            "n_exact_numeric_differences": exact, "n_numeric_outside_tolerance": outside,
            "n_structural_mismatches": self.structural_mismatches,
            "max_absolute_difference": max((x["max_absolute_difference"] for x in metrics.values()), default=0.),
            "metrics": metrics, "mismatch_examples": self.examples,
        }


def candidate_rows(candidate, context, horizons, loss_key="test_mse"):
    rows = []
    for index, horizon in enumerate(horizons):
        row = {**context, "family": candidate["family"], "candidate": candidate["name"],
               "history": candidate["history"], "n_features": candidate["n_features"],
               "horizon": horizon, "validation_mse": candidate["validation_mse"][index],
               "test_mse": candidate[loss_key][index],
               "test_rmse": float(math.sqrt(candidate[loss_key][index])),
               "optimization_gap": candidate.get("optimization_gap"),
               "converged": candidate.get("converged")}
        if "certificates" in candidate:
            row["rmse_lower"] = candidate["certificates"][index]["rmse_lower"]
            row["rmse_upper"] = candidate["certificates"][index]["rmse_upper"]
        if "conditional_iid_writer_rmse_upper" in candidate:
            row["conditional_iid_writer_rmse_upper"] = candidate["conditional_iid_writer_rmse_upper"][index]
        rows.append(row)
    return rows


def family_choices(record, campaign):
    grouped = defaultdict(list)
    for candidate in record["results"]:
        grouped[candidate["family"]].append(candidate)
    return {family: select_candidate(rows, campaign) for family, rows in grouped.items()}


def run_synthetic(args, campaign, cfg, outputs, replay_audit, invariant_audit):
    for system in cfg["systems"]:
        for seed in cfg["seeds"]:
            suffix = Path(campaign) / f"{system}_{seed}"
            records = {mode: read_json(args.results_root / mode / suffix / "results.json") for mode in args.modes}
            baseline = records["standard"]
            if not args.skip_original_replay:
                original = read_json(args.original_root / suffix / "results.json")
                replay_audit.compare(baseline, original, str(suffix))
            selections = {}
            for mode, record in records.items():
                if record["seed"] != seed or record["system"] != system:
                    raise ValueError(f"Record identity mismatch in {mode}/{suffix}")
                selections[mode] = family_choices(record, campaign)
                context = {"campaign": campaign, "system": system, "mode": mode, "seed": seed, "coordinate": "scalar"}
                for candidate in record["results"]:
                    rows = candidate_rows(candidate, context, cfg["horizons"])
                    selected = selections[mode][candidate["family"]]["name"] == candidate["name"]
                    for row in rows:
                        row["selected_by_validation"] = selected
                    outputs["all_candidates"].extend(rows)
                    if selected:
                        outputs["selected_candidates"].extend(rows)
                        outputs["selected_pooled_horizons"].extend(rows)
            standard_candidates = {c["name"]: c for c in baseline["results"]}
            with np.load(args.results_root / "standard" / suffix / "episode_scores.npz", allow_pickle=False) as archive:
                standard_arrays = {key: archive[key] for key in archive.files}
            for mode in args.modes:
                if mode == "standard":
                    continue
                record = records[mode]
                alt_candidates = {c["name"]: c for c in record["results"]}
                invariant_audit.compare(sorted(alt_candidates), sorted(standard_candidates), f"{mode}/{suffix}/candidate_names", "candidate_names")
                invariant_audit.compare(record["streams"], baseline["streams"], f"{mode}/{suffix}/streams", "streams")
                with np.load(args.results_root / mode / suffix / "episode_scores.npz", allow_pickle=False) as archive:
                    for name, standard_candidate in standard_candidates.items():
                        if not standard_candidate["family"].startswith("kahm"):
                            invariant_audit.compare(alt_candidates[name], standard_candidate, f"{mode}/{suffix}/{name}", "non_kahm_candidates")
                            for score_suffix in ("__test_loss", "__cal_loss", "__pair_score"):
                                key = name + score_suffix
                                invariant_audit.compare_array(archive[key], standard_arrays[key], f"{mode}/{suffix}/{key}", "non_kahm_arrays/" + score_suffix)
                    for family, standard_candidate in selections["standard"].items():
                        if not family.startswith("kahm"):
                            continue
                        alt_candidate = selections[mode][family]
                        left = standard_arrays[standard_candidate["name"] + "__test_loss"]
                        right = archive[alt_candidate["name"] + "__test_loss"]
                        if left.shape != (cfg["n_test"], len(cfg["horizons"])) or right.shape != left.shape:
                            raise ValueError(f"Unexpected test-score shape in {mode}/{suffix}/{family}")
                        if not np.allclose(left.mean(axis=0), standard_candidate["test_mse"], atol=ATOL, rtol=RTOL):
                            raise ValueError(f"Standard selected score/JSON mismatch: {suffix}/{family}")
                        if not np.allclose(right.mean(axis=0), alt_candidate["test_mse"], atol=ATOL, rtol=RTOL):
                            raise ValueError(f"Alternative selected score/JSON mismatch: {mode}/{suffix}/{family}")
                        context = {"campaign": campaign, "system": system, "mode": mode,
                                   "contrast_role": "primary" if mode == "minmax" else "diagnostic",
                                   "family": family, "seed": seed,
                                   "standard_candidate": standard_candidate["name"], "alternative_candidate": alt_candidate["name"],
                                   "paired_unit": "independent_test_episode", "interval_scope": "conditional_on_fitted_models_unadjusted"}
                        for index, horizon in enumerate(cfg["horizons"]):
                            outputs["paired_effects"].append({**context, "horizon": horizon,
                                                              **paired_statistics(left[:, index], right[:, index])})
                        outputs["paired_effects"].append({**context, "horizon": "all",
                                                          **paired_statistics(left.mean(axis=1), right.mean(axis=1))})


def writer_loss_map(path):
    result = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["seed"]), int(row["coordinate"]), row["candidate"],
                   int(row["writer"]), int(row["horizon"]))
            if key in result:
                raise ValueError(f"Duplicate writer-loss key {key}")
            result[key] = (float(row["writer_mean_squared_loss"]), int(row["n_eligible_strokes"]))
    return result


def run_pendigits(args, cfg, outputs, replay_audit, invariant_audit):
    campaign = "pendigits_v1"
    records = {mode: read_json(args.results_root / mode / campaign / "results.json") for mode in args.modes}
    baseline = records["standard"]
    if not args.skip_original_replay:
        original = read_json(args.original_root / "pendigits_v1b/results.json")
        replay_audit.compare(baseline, original, "pendigits_v1")
    loss_maps = {mode: writer_loss_map(args.results_root / mode / campaign / "writer_losses.csv") for mode in args.modes}
    if not args.skip_original_replay:
        original_losses = writer_loss_map(args.original_root / "pendigits_v1b/writer_losses.csv")
        replay_audit.compare(sorted(loss_maps["standard"]), sorted(original_losses), "pendigits_v1/writer_loss_keys", "writer_loss_keys")
        for key in original_losses.keys() & loss_maps["standard"].keys():
            replay_audit.compare(loss_maps["standard"][key], original_losses[key], "pendigits_v1/writer_loss/" + str(key), "writer_loss")
    choices, candidates_by_key = {}, {}
    for mode, record in records.items():
        candidates_by_key[mode] = {(c["seed"], c["coordinate"], c["name"]): c for c in record["results"]}
        choices[mode] = {}
        for seed in cfg["seeds"]:
            for coordinate in cfg["coordinates"]:
                subset = {"results": [c for c in record["results"] if c["seed"] == seed and c["coordinate"] == coordinate]}
                selection = family_choices(subset, campaign)
                choices[mode][(seed, coordinate)] = selection
                context = {"campaign": campaign, "system": "pendigits", "mode": mode, "seed": seed, "coordinate": coordinate}
                for candidate in subset["results"]:
                    selected = selection[candidate["family"]]["name"] == candidate["name"]
                    if selected != candidate["selected_before_test"]:
                        raise ValueError(f"PenDigits frozen selection mismatch: {mode}/{seed}/{coordinate}/{candidate['name']}")
                    rows = candidate_rows(candidate, context, cfg["horizons"], "cohort_writer_mean_mse")
                    for row in rows:
                        row["selected_by_validation"] = selected
                    outputs["all_candidates"].extend(rows)
                    if selected:
                        outputs["selected_candidates"].extend(rows)
            families = choices[mode][(seed, cfg["coordinates"][0])].keys()
            for family in families:
                chosen = [choices[mode][(seed, coordinate)][family] for coordinate in cfg["coordinates"]]
                mse = np.mean([c["cohort_writer_mean_mse"] for c in chosen], axis=0)
                validation = np.mean([c["validation_mse"] for c in chosen], axis=0)
                for index, horizon in enumerate(cfg["horizons"]):
                    outputs["selected_pooled_horizons"].append({"campaign": campaign, "system": "pendigits", "mode": mode,
                        "seed": seed, "coordinate": "pooled", "family": family, "candidate": [c["name"] for c in chosen],
                        "history": [c["history"] for c in chosen], "n_features": float(np.mean([c["n_features"] for c in chosen])),
                        "horizon": horizon, "validation_mse": float(validation[index]), "test_mse": float(mse[index]),
                        "test_rmse": float(math.sqrt(mse[index])), "selected_by_validation": True})
    for mode in args.modes:
        if mode == "standard":
            continue
        record = records[mode]
        for key in ("held_out_record_ids", "origins", "n_held_out_writers", "n_held_out_eligible_digits"):
            invariant_audit.compare(record[key], baseline[key], "pendigits_v1/" + mode + "/" + key, key)
        for key, standard_candidate in candidates_by_key["standard"].items():
            if not standard_candidate["family"].startswith("kahm"):
                invariant_audit.compare(candidates_by_key[mode][key], standard_candidate,
                                        f"pendigits_v1/{mode}/{key}", "non_kahm_candidates")
        for key, value in loss_maps["standard"].items():
            if not candidates_by_key["standard"][(key[0], key[1], key[2])]["family"].startswith("kahm"):
                invariant_audit.compare(loss_maps[mode][key], value, f"pendigits_v1/{mode}/writer_loss/{key}", "non_kahm_writer_losses")
        for seed in cfg["seeds"]:
            families = choices["standard"][(seed, cfg["coordinates"][0])].keys()
            for family in families:
                if not family.startswith("kahm"):
                    continue
                picked = {arm: [choices[arm][(seed, coordinate)][family] for coordinate in cfg["coordinates"]]
                          for arm in ("standard", mode)}
                writers = sorted({key[3] for key in loss_maps["standard"] if key[0] == seed})
                arrays = {}
                for arm in ("standard", mode):
                    coordinate_arrays = []
                    for coordinate, candidate in zip(cfg["coordinates"], picked[arm]):
                        coordinate_arrays.append([[loss_maps[arm][(seed, coordinate, candidate["name"], writer, horizon)][0]
                                                   for horizon in cfg["horizons"]] for writer in writers])
                    arrays[arm] = np.mean(coordinate_arrays, axis=0)
                    expected = np.mean([c["cohort_writer_mean_mse"] for c in picked[arm]], axis=0)
                    if not np.allclose(arrays[arm].mean(axis=0), expected, atol=ATOL, rtol=RTOL):
                        raise ValueError(f"PenDigits writer-score/JSON mismatch: {arm}/{seed}/{family}")
                context = {"campaign": campaign, "system": "pendigits", "mode": mode,
                           "contrast_role": "primary" if mode == "minmax" else "diagnostic", "family": family, "seed": seed,
                           "standard_candidate": [c["name"] for c in picked["standard"]],
                           "alternative_candidate": [c["name"] for c in picked[mode]],
                           "paired_unit": "fixed_cohort_writer", "interval_scope": "descriptive_no_iid_writer_assumption"}
                for index, horizon in enumerate([*cfg["horizons"], "all"]):
                    left = arrays["standard"][:, index] if horizon != "all" else arrays["standard"].mean(axis=1)
                    right = arrays[mode][:, index] if horizon != "all" else arrays[mode].mean(axis=1)
                    outputs["paired_effects"].append({**context, "horizon": horizon, **paired_statistics(left, right, inferential=False)})
                    for writer, value_left, value_right in zip(writers, left, right):
                        outputs["pendigits_writer_differences"].append({**context, "horizon": horizon, "writer": writer,
                            "standard_mse": float(value_left), "alternative_mse": float(value_right),
                            "mse_difference_alternative_minus_standard": float(value_right-value_left),
                            "effect_sign": sign_name(value_right-value_left)})


def aggregate_outputs(outputs):
    groups = defaultdict(list)
    for row in outputs["selected_pooled_horizons"]:
        groups[(row["campaign"], row["system"], row["mode"], row["family"], row["horizon"])].append(row)
    for (campaign, system, mode, family, horizon), rows in sorted(groups.items()):
        outputs["horizon_summary"].append({"campaign": campaign, "system": system, "mode": mode,
            "family": family, "horizon": horizon, "n_fitting_seeds": len(rows),
            "mean_test_rmse": float(np.mean([r["test_rmse"] for r in rows])),
            "sd_test_rmse_over_fitting_seeds": sample_sd([r["test_rmse"] for r in rows]),
            "mean_test_mse": float(np.mean([r["test_mse"] for r in rows])),
            "mean_validation_mse": float(np.mean([r["validation_mse"] for r in rows])),
            "mean_selected_features": float(np.mean([r["n_features"] for r in rows])),
            "seeds": [r["seed"] for r in rows]})
    paired_groups = defaultdict(list)
    for row in outputs["paired_effects"]:
        paired_groups[(row["campaign"], row["system"], row["mode"], row["family"], str(row["horizon"]))].append(row)
    for (campaign, system, mode, family, horizon), rows in sorted(paired_groups.items()):
        left = np.asarray([r["standard_mse"] for r in rows])
        right = np.asarray([r["alternative_mse"] for r in rows])
        differences = np.asarray([r["mse_difference_alternative_minus_standard"] for r in rows])
        standard_errors = [r["conditional_standard_error"] for r in rows]
        combined_se = math.sqrt(sum(x*x for x in standard_errors)) / len(rows) if all(x is not None for x in standard_errors) else None
        effect = float(differences.mean())
        summary = {"campaign": campaign, "system": system, "mode": mode,
                   "contrast_role": "primary" if mode == "minmax" else "diagnostic",
                   "family": family, "horizon": int(horizon) if horizon != "all" else "all",
                   "n_fitting_seeds": len(rows), "standard_mean_mse": float(left.mean()), "alternative_mean_mse": float(right.mean()),
                   "mse_difference_alternative_minus_standard": effect,
                   "relative_mse_change_percent": float(100*effect/left.mean()) if left.mean() > 0 else None,
                   "relative_mse_reduction_percent": float(-100*effect/left.mean()) if left.mean() > 0 else None,
                   "standard_mse_sd_over_seeds": sample_sd(left), "alternative_mse_sd_over_seeds": sample_sd(right),
                   "paired_mse_difference_sd_over_seeds": sample_sd(differences),
                   "conditional_standard_error": combined_se,
                   "conditional_normal_95_lower": effect-Z95*combined_se if combined_se is not None else None,
                   "conditional_normal_95_upper": effect+Z95*combined_se if combined_se is not None else None,
                   "n_seeds_improved": sum(r["effect_sign"] == "improved" for r in rows),
                   "n_seeds_degraded": sum(r["effect_sign"] == "degraded" for r in rows),
                   "n_seeds_tied": sum(r["effect_sign"] == "tie_within_1e-12" for r in rows),
                   "seeds": [r["seed"] for r in rows], "seed_effect_signs": [r["effect_sign"] for r in rows],
                   "seed_mse_differences": differences.tolist(), "interval_scope": rows[0]["interval_scope"]}
        outputs["comparison_by_horizon"].append(summary)
        if horizon == "all":
            outputs["taskwise_comparison"].append(summary)


def required_paths(args, configurations):
    required = []
    for campaign in args.campaigns:
        cfg = configurations[campaign]
        if campaign == "pendigits_v1":
            if not args.skip_original_replay:
                required.extend(args.original_root / "pendigits_v1b" / filename for filename in ("results.json", "writer_losses.csv"))
            for mode in args.modes:
                required.extend(args.results_root / mode / campaign / filename for filename in ("results.json", "writer_losses.csv"))
        else:
            for system in cfg["systems"]:
                for seed in cfg["seeds"]:
                    suffix = Path(campaign) / f"{system}_{seed}"
                    if not args.skip_original_replay:
                        required.append(args.original_root / suffix / "results.json")
                    for mode in args.modes:
                        required.extend(args.results_root / mode / suffix / filename for filename in ("results.json", "episode_scores.npz"))
    return required


def self_test():
    # Candidate selection must never favor the lower test error when validation is worse.
    candidates = [dict(name="first", n_features=30, validation_mse=[1., 1., 1.], test_mse=[0., 0., 0.]),
                  dict(name="second", n_features=6, validation_mse=[.9, .9, .9], test_mse=[2., 2., 2.])]
    assert select_candidate(candidates, "main_v1")["name"] == "second"
    candidates[0]["validation_mse"] = [.9, .9, .9]
    assert select_candidate(candidates, "main_v1")["name"] == "first"
    assert select_candidate(candidates, "oscillators_v1")["name"] == "second"
    # Correlated horizon differences cancel before the episode-level SE is computed.
    base = np.ones((4, 2))
    alt = base + np.asarray([[.3, -.1], [.1, .1], [-.1, .3], [.2, 0.]])
    paired = paired_statistics(base.mean(axis=1), alt.mean(axis=1))
    assert abs(paired["mse_difference_alternative_minus_standard"] - .1) < 1e-14
    assert paired["conditional_standard_error"] < 1e-14
    assert paired_statistics([1., 2.], [.5, 1.], inferential=False)["conditional_standard_error"] is None
    # Coordinate pooling must happen in MSE units, before square roots.
    assert math.isclose(float(np.sqrt(np.mean([.01, .09]))), math.sqrt(.05))
    assert not math.isclose(float(np.mean(np.sqrt([.01, .09]))), math.sqrt(.05))
    audit = Audit()
    audit.compare({"a": [1., 2.], "train_seconds": 999}, {"a": [1., 2.], "train_seconds": 1})
    assert audit.result()["status"] == "exact"
    audit.compare_array(np.array([1.]), np.array([1.+1e-13]), "fixture", "metric")
    assert audit.result()["status"] == "within_tolerance"
    audit.compare_array(np.array([1.]), np.array([2.]), "fixture", "metric")
    assert audit.result()["status"] == "mismatch"
    outputs = defaultdict(list)
    for seed, mean, se in [(1, -.1, .02), (2, -.3, .04)]:
        outputs["paired_effects"].append(dict(campaign="main_v1", system="fixture", mode="minmax", family="kahm_nested",
            horizon="all", seed=seed, standard_mse=1., alternative_mse=1.+mean,
            mse_difference_alternative_minus_standard=mean, conditional_standard_error=se,
            effect_sign="improved", interval_scope="fixture"))
    aggregate_outputs(outputs)
    summary = outputs["taskwise_comparison"][0]
    assert math.isclose(summary["relative_mse_reduction_percent"], 20.)
    assert math.isclose(summary["conditional_standard_error"], math.sqrt(.02**2+.04**2)/2)
    print("Self-test passed: validation-only selection, horizon pairing, coordinate pooling, audit exclusions, and task aggregation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument("--original-root", type=Path, default=DEFAULT_SOURCE / "results")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_SOURCE / "configs")
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=DEFAULT_MODES)
    parser.add_argument("--campaigns", nargs="+", default=list(ALL_CAMPAIGNS), choices=ALL_CAMPAIGNS)
    parser.add_argument("--skip-original-replay", action="store_true",
                        help="For genuinely fresh confirmation seeds with no retained original result; non-KAHM invariance is still checked")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if "standard" not in args.modes or len(args.modes) < 2:
        parser.error("Include standard and at least one alternative mode")
    configurations = {name: read_json(args.config_root / (name + ".json")) for name in args.campaigns}
    required = required_paths(args, configurations)
    missing = [str(path) for path in required if not path.is_file()]
    invalid_archives = [str(path) for path in required if path.suffix == ".npz" and path.is_file() and not zipfile.is_zipfile(path)]
    if missing or invalid_archives:
        print(json.dumps({"status": "incomplete_no_performance_analysis", "n_missing_required_files": len(missing),
                          "missing_files": missing, "n_invalid_score_archives": len(invalid_archives),
                          "invalid_score_archives": invalid_archives}, indent=2), file=sys.stderr)
        return 2
    outputs = defaultdict(list)
    baseline_audits, invariance_audits = {}, {}
    for campaign in args.campaigns:
        print(f"Analyzing complete campaign {campaign}", flush=True)
        baseline, invariance = Audit(), Audit()
        if campaign == "pendigits_v1":
            run_pendigits(args, configurations[campaign], outputs, baseline, invariance)
        else:
            run_synthetic(args, campaign, configurations[campaign], outputs, baseline, invariance)
        baseline_audits[campaign] = ({"status": "not_applicable_fresh_confirmation_seeds",
                                      "reason": "Explicit --skip-original-replay; no published counterpart for these seeds"}
                                     if args.skip_original_replay else baseline.result())
        invariance_audits[campaign] = invariance.result()
    aggregate_outputs(outputs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_csv(args.output / (name + ".csv"), rows)
    repair_paths = sorted({path.parent / "artifact_repair.json" for path in required
                           if path.name == "episode_scores.npz" and (path.parent / "artifact_repair.json").is_file()})
    repairs = [{"path": str(path), "record": read_json(path)} for path in repair_paths]
    audit = {"baseline_replay": baseline_audits, "non_kahm_invariance": invariance_audits,
             "score_archive_repairs": repairs, "notes": NOTES["audit"]}
    atomic_json(args.output / "audit.json", audit)
    primary = [row for row in outputs["taskwise_comparison"] if row["mode"] == "minmax" and row["family"] == "kahm_nested"]
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_sha256": sha256(__file__),
               "modes": args.modes, "campaigns": args.campaigns, "notes": NOTES,
               "primary_six_synthetic_tasks_and_pendigits": [r for r in primary if not r["campaign"].startswith("sensitivity")],
               "primary_sensitivity_tasks": [r for r in primary if r["campaign"].startswith("sensitivity")],
               "all_taskwise_comparisons": outputs["taskwise_comparison"],
               "all_horizon_summaries": outputs["horizon_summary"],
               "score_archive_repairs": repairs,
               "input_result_json_hashes": {str(path): sha256(path) for path in required if path.name == "results.json"},
               "input_config_sha256": {name: sha256(args.config_root / (name + ".json")) for name in args.campaigns},
               "output_row_counts": {name: len(rows) for name, rows in outputs.items()},
               "audit_status": {"baseline_replay": {name: value["status"] for name, value in baseline_audits.items()},
                                "non_kahm_invariance": {name: value["status"] for name, value in invariance_audits.items()}}}
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"output": str(args.output), "primary_results": primary,
                      "audit_status": summary["audit_status"], "row_counts": summary["output_row_counts"]}, indent=2))
    return 1 if any(value["status"] == "mismatch" for value in [*baseline_audits.values(), *invariance_audits.values()]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
