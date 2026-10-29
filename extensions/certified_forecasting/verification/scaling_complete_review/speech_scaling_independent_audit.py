"""Independent arithmetic audit of the descriptive speech scaling evidence.

This script imports neither the study runner nor its verifier. It reconstructs
reported statistics from the retained group losses, using Python's CSV parser
and sample-statistics functions. Numeric training archives are opened with
allow_pickle=False. It does not load predictive-model pickles or refit models.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np


EXTENSION = Path(__file__).resolve().parents[2]
REPOSITORY = EXTENSION.parents[1]
SEEDS = (8601, 8602, 8603)
FAMILIES = ("kahm", "rbf", "affine")
MODES = ("standard", "raw", "minmax", "minmax_clip")
HORIZONS = (1, 4, 12)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def average(values):
    values = list(values)
    return math.fsum(values) / len(values)


def audit(results, *, portable=False):
    errors = {}
    checks = {}

    def equal(label, actual, expected, tolerance=5e-10):
        error = abs(float(actual) - float(expected))
        if not math.isfinite(error) or error > tolerance:
            raise AssertionError((label, actual, expected, error))
        errors[label] = max(errors.get(label, 0.0), error)

    def same(label, actual, expected):
        if actual != expected:
            raise AssertionError((label, actual, expected))
        checks[label] = checks.get(label, 0) + 1

    protocol = read_json(results / "protocol_freeze.json")
    development_freeze = read_json(results / "development_freeze.json")
    completion = read_json(results / "completion.json")
    for relative, expected in protocol["source_hashes"].items():
        same("frozen_source_hashes", sha256(REPOSITORY / relative), expected)
    inherited = read_json(results / "inherited_source_freeze.json")
    for relative, expected in inherited["inherited_root_sources"].items():
        same("inherited_source_hashes", sha256(REPOSITORY / relative), expected)
    same("protocol_manifest_hash", sha256(results / "protocol_freeze.json"),
         development_freeze["protocol_freeze_sha256"])
    for filename, key in (("development_freeze.json", "development_freeze_sha256"),
                          ("results.json", "results_sha256"),
                          ("group_losses.csv.gz", "group_losses_sha256")):
        same("completion_hashes", sha256(results / filename), completion[key])

    clips = read_csv(results / "clips.csv")
    group_split = {}
    for clip in clips:
        group = clip["speaker"]
        if group in group_split:
            same("group_disjointness", group_split[group], clip["split"])
        group_split[group] = clip["split"]
    same("group_counts", {s: list(group_split.values()).count(s)
         for s in ("train", "validation", "calibration", "test")},
         {"train": 244, "validation": 163, "calibration": 978, "test": 245})
    test_groups = tuple(sorted(g for g, s in group_split.items() if s == "test"))
    scores = {}
    for row in read_csv(results / "group_losses.csv.gz"):
        key = (int(row["seed"]), row["family"], row["scaling"],
               int(row["horizon"]), row["group"])
        if key in scores:
            raise AssertionError(("duplicate group loss", key))
        value = float(row["mse"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise AssertionError(("invalid bounded loss", key, value))
        scores[key] = value
    expected_keys = set(itertools.product(SEEDS, FAMILIES, MODES, HORIZONS, test_groups))
    same("complete_score_grid", set(scores), expected_keys)
    means = {key: average(scores[(*key, g)] for g in test_groups)
             for key in itertools.product(SEEDS, FAMILIES, MODES, HORIZONS)}
    all_horizon_means = {
        key: average(means[(*key, h)] for h in HORIZONS)
        for key in itertools.product(SEEDS, FAMILIES, MODES)}

    if not portable:
        same("energy_archive_hash", sha256(results / "energy_observations.npz"),
             development_freeze["energy_sha256"])
        with np.load(results / "energy_observations.npz", allow_pickle=False) as archive:
            same("energy_archive_keys", set(archive.files), {"energy"})
            energy = archive["energy"]
        same("energy_shape", energy.shape, (len(clips), 100))
        if not np.isfinite(energy).all() or energy.min() < 0 or energy.max() > 1:
            raise AssertionError("Energy array is outside its deterministic range")

    selected = {}
    total_candidates = converged_candidates = 0
    max_gap = 0.0
    for seed in SEEDS:
        path = results / f"development_{seed}.json"
        same("development_hashes", sha256(path), development_freeze["development"][str(seed)])
        development = read_json(path)
        candidates = development["candidates"]
        total_candidates += len(candidates)
        converged_candidates += sum(bool(c["converged"]) for c in candidates)
        max_gap = max(max_gap, max(c["optimization_gap"] for c in candidates))
        candidate_keys = set()
        for c in candidates:
            p = c["parameters"]
            candidate_keys.add((c["family"], c["scaling"], c["history"],
                                p["sharpness"], p["stage"], p["kappa"], p["alpha"]))
            equal("validation_mean", c["validation_mean"], average(c["validation_mse"]))
            if c["coefficient_norm"] > p["kappa"] + 1e-10:
                raise AssertionError(("coefficient infeasibility", c["candidate"]))
        expected_candidates = {
            (family, mode, history, sharp, stage, kappa, ridge)
            for family in FAMILIES for mode in MODES for history in (4, 12)
            for sharp in ((1.0,) if family == "kahm" else (0.25, 1.0, 4.0))
            for stage in (1, 2, 3, 4) for kappa in (1.0, 4.0)
            for ridge in (0.0001, 0.01)}
        same("complete_candidate_grid", candidate_keys, expected_candidates)
        same("unique_candidates", len(candidates), len(candidate_keys))
        for c in development["selected"]:
            key = (seed, c["family"], c["scaling"])
            contenders = [r for r in candidates if
                          (r["family"], r["scaling"]) == key[1:]]
            winner = min(contenders, key=lambda r: (
                average(r["validation_mse"]), r["features"], r["history"], r["candidate"]))
            same("validation_only_selection", c["candidate"], winner["candidate"])
            selected[key] = c

        if not portable:
            with np.load(results / f"training_rows_{seed}.npz", allow_pickle=False) as archive:
                same("training_archive_keys", set(archive.files), {"clip_index", "origin_index"})
                clip_indices = archive["clip_index"]
                origin_indices = archive["origin_index"]
            same("training_row_count", (len(clip_indices), len(origin_indices)), (4096, 4096))
            if any(clips[int(i)]["split"] != "train" for i in clip_indices):
                raise AssertionError("A training row comes from a held-out split")
            if not np.all((0 <= origin_indices) & (origin_indices < 20)):
                raise AssertionError("Training origin index is out of range")
            origins = 11 + 4 * origin_indices
            geometry_indices = {}
            for bank in development["banks"]:
                identifier = bank["bank_id"]
                history = int(identifier.split("_m")[1].split("_")[0])
                x = energy[clip_indices[:, None], origins[:, None] - np.arange(history - 1, -1, -1)]
                metadata = bank["input_scaling"]
                if history in geometry_indices:
                    same("paired_geometry_subsamples", bank["fit_indices"], geometry_indices[history])
                geometry_indices[history] = bank["fit_indices"]
                same("geometry_row_count", len(bank["fit_indices"]), 1600)
                if metadata["mode"] == "standard":
                    equal("training_only_mean", np.max(np.abs(x.mean(0) - metadata["mean"])), 0)
                    scale = np.where(x.std(0) > 1e-12, x.std(0), 1.0)
                    equal("training_only_sd", np.max(np.abs(scale - metadata["scale"])), 0)
                elif metadata["mode"].startswith("minmax"):
                    lo, hi = x.min(0), x.max(0)
                    equal("training_only_min", np.max(np.abs(lo - metadata["training_min"])), 0)
                    equal("training_only_max", np.max(np.abs(hi - metadata["training_max"])), 0)
                    same("constant_coordinates", (lo == hi).tolist(), metadata["constant_features"])

    result_json = read_json(results / "results.json")
    same("descriptive_scope", (result_json["confidence_claims"], result_json["calibration_evaluated"]), (False, False))
    same("selected_result_count", len(result_json["results"]), 36)
    for row in result_json["results"]:
        key = (row["seed"], row["family"], row["scaling"])
        same("selected_candidate_identity", row["candidate"], selected[key]["candidate"])
        same("reported_test_groups", row["test_groups"], len(test_groups))
        for h, value in zip(HORIZONS, row["test_mse"]):
            equal("results_json_mse", value, means[(*key, h)])
        if not portable and row["scaling"].startswith("minmax"):
            held_out = [i for i, clip in enumerate(clips) if clip["split"] == "test"]
            forecast_origins = np.arange(11, 88, 4)
            offsets = forecast_origins[:, None] - np.arange(row["history"] - 1, -1, -1)
            histories = energy[held_out][:, offsets].reshape(-1, row["history"])
            settings = row["input_scaling"]
            lo, hi = np.asarray(settings["training_min"]), np.asarray(settings["training_max"])
            span = hi - lo
            normalized = 2 * (histories - lo) / np.where(span == 0, 1, span) - 1
            normalized[:, span == 0] = 0
            equal("out_of_range_coordinate_fraction", row["test_coordinate_fraction_outside_training_range"],
                  np.mean(np.abs(normalized) > 1))
            equal("out_of_range_history_fraction", row["test_history_fraction_outside_training_range"],
                  np.mean(np.any(np.abs(normalized) > 1, axis=1)))
            equal("unclipped_minimum", row["test_unclipped_min"], normalized.min())
            equal("unclipped_maximum", row["test_unclipped_max"], normalized.max())

    for row in read_csv(results / "selected_results.csv"):
        key = (int(row["seed"]), row["family"], row["scaling"], int(row["horizon"]))
        equal("selected_csv_mse", row["mse"], means[key])
        equal("selected_csv_rmse_db", row["rmse_db"], 60 * math.sqrt(means[key]))
    same("selected_csv_count", len(read_csv(results / "selected_results.csv")), 108)

    for row in read_csv(results / "summary.csv"):
        suffix = (row["family"], row["scaling"], int(row["horizon"]))
        values = [means[(s, *suffix)] for s in SEEDS]
        db_values = [60 * math.sqrt(v) for v in values]
        for column, expected in (("mean_mse", average(values)), ("sd_mse", statistics.stdev(values)),
                                 ("mean_rmse_db", average(db_values)), ("sd_rmse_db", statistics.stdev(db_values))):
            equal("summary_" + column, row[column], expected)
        same("summary_n_fits", int(row["n_fits"]), 3)
    same("summary_row_count", len(read_csv(results / "summary.csv")), 36)

    seed_deltas = []
    for row in read_csv(results / "paired_differences.csv"):
        seed, family, mode, horizon = int(row["seed"]), row["family"], row["scaling"], int(row["horizon"])
        differences = [scores[(seed, family, mode, horizon, g)] -
                       scores[(seed, family, "standard", horizon, g)] for g in test_groups]
        delta = average(differences)
        equal("paired_mean_difference", row["mean_mse_difference"], delta)
        equal("paired_conditional_group_se", row["conditional_group_se"],
              statistics.stdev(differences) / math.sqrt(len(test_groups)))
        equal("paired_relative_percent", row["relative_mse_change_percent"],
              100 * delta / means[(seed, family, "standard", horizon)])
        same("paired_groups_improved", int(row["groups_improved"]), sum(d < 0 for d in differences))
        same("paired_group_count", int(row["n_groups"]), len(test_groups))
    same("paired_row_count", len(read_csv(results / "paired_differences.csv")), 81)

    for row in read_csv(results / "all_horizon_selected.csv"):
        key = (int(row["seed"]), row["family"], row["scaling"])
        equal("all_horizon_selected_mse", row["mse"], all_horizon_means[key])
        same("all_horizon_selection", row["candidate"], selected[key]["candidate"])
    for row in read_csv(results / "all_horizon_summary.csv"):
        values = [all_horizon_means[(s, row["family"], row["scaling"])] for s in SEEDS]
        equal("all_horizon_mean", row["mean"], average(values))
        equal("all_horizon_sd", row["std"], statistics.stdev(values))
    same("all_horizon_summary_count", len(read_csv(results / "all_horizon_summary.csv")), 12)
    for seed, family, mode in itertools.product(SEEDS, FAMILIES, MODES[1:]):
        base = all_horizon_means[(seed, family, "standard")]
        changed = all_horizon_means[(seed, family, mode)]
        seed_deltas.append({"seed": seed, "family": family, "scaling": mode,
                            "mean_mse_difference": changed - base,
                            "relative_mse_change_percent": 100 * (changed - base) / base})

    family_findings = []
    paper_rows = []
    for family in FAMILIES:
        reference = average(all_horizon_means[(s, family, "standard")] for s in SEEDS)
        formatted_cells = []
        for mode in MODES:
            values = [all_horizon_means[(s, family, mode)] for s in SEEDS]
            family_findings.append({"family": family, "scaling": mode, "mean_mse": average(values),
                                    "fitting_sd": statistics.stdev(values),
                                    "relative_to_standard_percent": 100 * (average(values) - reference) / reference})
            formatted_cells.append(f"{1000 * average(values):.3f} ({1000 * statistics.stdev(values):.3f})")
        minmax = average(all_horizon_means[(s, family, "minmax")] for s in SEEDS)
        favorable = sum(all_horizon_means[(s, family, "minmax")] <
                        all_horizon_means[(s, family, "standard")] for s in SEEDS)
        effect = 100 * (minmax - reference) / reference
        formatted_cells += [f"${effect:+.2f}$", f"{favorable}/3"]
        display_name = {"kahm": "KAHM", "rbf": "RBF", "affine": "Affine hull"}[family]
        expected_line = display_name + " & " + " & ".join(formatted_cells) + r" \\"
        paper_rows.append(expected_line)
        comparisons = read_json(results / "all_horizon_comparisons.json")
        comparison = next(r for r in comparisons if r["family"] == family)
        equal("comparison_json_relative_percent", comparison["relative_minmax_vs_standard_percent"], effect)
        same("comparison_json_favorable_fits", comparison["favorable_fits"], favorable)
        for seed in SEEDS:
            base = all_horizon_means[(seed, family, "standard")]
            changed = all_horizon_means[(seed, family, "minmax")]
            equal("comparison_json_seed_percent", comparison["per_seed_relative_percent"][str(seed)],
                  100 * (changed - base) / base)
    table_lines = (results / "speech_scaling.tex").read_text().splitlines()
    for expected_line in paper_rows:
        same("paper_table_rounding", expected_line in table_lines, True)
    assets = read_json(results / "asset_manifest.json")
    for name, expected in assets["outputs"].items():
        same("derived_asset_hashes", sha256(results / name), expected)
    same("asset_source_score_hash", assets["loss_sha256"], sha256(results / "group_losses.csv.gz"))
    return {"status": "passed", "portable": portable,
            "audit_scope": "Independent reaggregation and training-input provenance; no model pickle loading or refitting",
            "score_count": len(scores), "selected_models": len(selected),
            "candidates": total_candidates, "converged_candidates": converged_candidates,
            "maximum_recorded_training_gap": max_gap,
            "test_groups": len(test_groups), "checks": checks, "max_absolute_errors": errors,
            "family_findings": family_findings, "all_horizon_seed_deltas": seed_deltas,
            "verified_paper_rows": paper_rows,
            "confidence_claims": False,
            "limitations": ["The reused test cohort does not constitute independent confirmation.",
                            "Conditional group standard errors are descriptive; fitting seeds are paired on the same cohort.",
                            "The loader derives the fixed observable for all splits, but calibration groups are not used for fitting, selection, or loss scoring."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=EXTENSION / "results/speech_scaling_v1")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--portable", action="store_true")
    args = parser.parse_args()
    report = audit(args.results, portable=args.portable)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "score_count", "selected_models", "candidates", "converged_candidates", "max_absolute_errors", "family_findings")}, indent=2))


if __name__ == "__main__":
    main()
