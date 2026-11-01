#!/usr/bin/env python3
"""Standalone score, endpoint, selection, and retained-array analysis.

Only standard libraries, NumPy, and Matplotlib are used. No KAHM source is
imported and no fitted model is unpickled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty scientific table")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        raise FileExistsError(path)
    with temporary.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    with temporary.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != keys:
            raise AssertionError("CSV header changed during export")
        reloaded = list(reader)
    if len(reloaded) != len(rows):
        raise AssertionError("CSV export lost rows")
    for source, parsed in zip(rows, reloaded, strict=True):
        expected = {key: "" if source.get(key) is None else str(source.get(key, "")) for key in keys}
        if parsed != expected:
            raise AssertionError("CSV export changed a cell")
    temporary.replace(path)


def close(a, b, label):
    if not np.allclose(a, b, rtol=2e-12, atol=2e-13, equal_nan=False):
        raise AssertionError(f"Independent reconstruction disagrees: {label}")


def interval(values, bound, delta):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise AssertionError("Invalid independent-unit scores")
    if x.min() < -1e-12 or x.max() > bound + 1e-10:
        raise AssertionError("A declared score bound is violated")
    mean, variance = float(x.mean()), float(x.var(ddof=1))
    radius = math.sqrt(2 * variance * math.log(2 / delta) / len(x))
    radius += 7 * bound * math.log(2 / delta) / (3 * (len(x) - 1))
    return {"mean": mean, "variance": variance, "radius": radius,
            "lower": max(0.0, mean - radius), "upper": min(bound, mean + radius)}


def selection_key(row):
    return (float(np.mean(row["validation_mse"])), row["n_features"], row["history"],
            row["cap"], row["id"])


def verify_seals(run):
    frozen = read(run / "freeze.json")
    if frozen.get("source_sha256", {}).get("analyze.py") != digest(Path(__file__)):
        raise AssertionError("Analyzer differs from the frozen source")
    expected_inputs = {"config.json", "PROTOCOL.md", "schedule.json", "manifest.json", "preflight.json"}
    if set(frozen.get("files", {})) != expected_inputs:
        raise AssertionError("Incomplete frozen-input roster")
    for name, value in frozen["files"].items():
        if digest(run / name) != value:
            raise AssertionError(f"Frozen input hash mismatch: {name}")
    preceding = None
    jobs = read(run / "schedule.json")
    stage_files = {
        "development": ("development.json", "fitted_models.pkl", "coefficients.npz", "validation_scores.npz"),
        "calibration": ("calibration.json", "calibration_scores.npz"),
        "test": ("test.json", "test_scores.npz"),
    }
    for stage in ("development", "calibration", "test"):
        seal = read(run / (stage + "_seal.json"))
        expected = set()
        for job in jobs:
            names = list(stage_files[stage]) + [stage + "_started.json", stage + "_complete.json"]
            if stage != "development" and job["full_prediction_audit"]:
                names.append(stage + "_predictions.npz")
            expected.update(f"jobs/{job['id']}/{name}" for name in names)
        if seal.get("stage") != stage or seal.get("n_jobs") != len(jobs) or set(seal.get("files", {})) != expected:
            raise AssertionError("Incomplete or rebound stage seal")
        if seal["freeze_sha256"] != digest(run / "freeze.json"):
            raise AssertionError("Stage belongs to a different source freeze")
        if seal["preceding_seal_sha256"] != preceding:
            raise AssertionError("Stage seal chain is broken")
        for name, value in seal["files"].items():
            if digest(run / name) != value:
                raise AssertionError(f"Sealed artifact hash mismatch: {name}")
        preceding = digest(run / (stage + "_seal.json"))


def analyze(run, output):
    verify_seals(run)
    cfg, schedule = read(run / "config.json"), read(run / "schedule.json")
    tail = cfg["delta"] / cfg["campaign_total_tails"]
    endpoint_rows, decision_rows, contrast_rows, selected_rows, ranks = [], [], [], [], []
    counters = {"repetitions": 0, "development_candidates": 0, "evaluated_model_aliases": 0,
                "risk_endpoints": 0, "improvement_endpoints": 0, "full_array_jobs": 0,
                "full_array_forecast_joins": 0, "local_rank_records": 0,
                "coefficient_budget_checks": 0, "prediction_range_checks": 0,
                "pair_witness_reconstructions": 0}
    fitting = []

    for job in schedule:
        folder = run / "jobs" / job["id"]
        development, calibration, test = (read(folder / (stage + ".json"))
                                          for stage in ("development", "calibration", "test"))
        counters["repetitions"] += 1
        fitting.append({"job": job["id"], "system": job["system"],
                        "development_seconds": development["elapsed_seconds"],
                        "calibration_seconds": calibration["elapsed_seconds"],
                        "test_seconds": test["elapsed_seconds"]})
        if calibration["tail_delta"] != tail:
            raise AssertionError("Incorrect endpoint allocation")
        with np.load(folder / "validation_scores.npz", allow_pickle=False) as validation:
            for row in development["candidates"]:
                values = validation[row["id"]]
                if values.shape != (cfg["n_validation"], len(cfg["horizons"])):
                    raise AssertionError("Incorrect validation unit count")
                close(values.mean(axis=0), row["validation_mse"], row["id"] + "/validation")
                counters["development_candidates"] += 1
        for arm, choice in development["choices"].items():
            family = "kahm" if arm in cfg["policies"] else arm
            eligible = [row for row in development["candidates"] if row["family"] == family]
            if arm in ("cap2", "native20"):
                eligible = [row for row in eligible if row["cap"] == (2 if arm == "cap2" else 20)]
            winner = min(eligible, key=selection_key)
            if winner["id"] != choice["winner_id"]:
                raise AssertionError("Validation selection mismatch")

        evaluation = {row["alias"]: row for row in development["evaluation"]}
        cal_by_alias = {row["alias"]: row for row in calibration["results"]}
        test_by_alias = {row["alias"]: row for row in test["results"]}
        if set(evaluation) != set(cal_by_alias) or set(evaluation) != set(test_by_alias):
            raise AssertionError("Evaluated family changed across stages")
        counters["evaluated_model_aliases"] += len(evaluation)
        with np.load(folder / "coefficients.npz", allow_pickle=False) as coefficients:
            for alias, row in evaluation.items():
                coef = coefficients[row["id"] + "__coefficient"]
                intercept = coefficients[row["id"] + "__intercept"]
                if coef.shape != (row["n_features"], len(cfg["horizons"])):
                    raise AssertionError("Coefficient dimension mismatch")
                if intercept.shape != (len(cfg["horizons"]),):
                    raise AssertionError("Intercept dimension mismatch")
                if row["family"] == "persistence":
                    norm = float(np.linalg.norm(coef, axis=0).max())
                elif row["blocks"] is None:
                    norm = float(np.linalg.norm(coef))
                else:
                    indices = [i for a, b in row["blocks"] for i in range(a, b)]
                    if indices != list(range(row["n_features"])):
                        raise AssertionError("Block slices do not partition the feature coordinates")
                    norm = sum(float(np.linalg.norm(coef[a:b])) for a, b in row["blocks"])
                close(norm, row["coefficient_norm"], alias + "/norm")
                if norm > row["kappa"] + 1e-9:
                    raise AssertionError("Coefficient budget violation")
                counters["coefficient_budget_checks"] += 1

        for arm in (*cfg["policies"], "rbf", "affine"):
            choice = development["choices"][arm]
            bank = development["banks"][choice["bank_id"]]
            for rank in bank["dimension_policy"]["local_models"]:
                if arm in cfg["policies"]:
                    initial = min(rank["requested_cap"], rank["ambient_dimension"],
                                  rank["local_reference_count"] - 1)
                    if initial != rank["initial_admissible_rank"]:
                        raise AssertionError("Native dimension request mismatch")
                    if not 1 <= rank["effective_rank"] <= initial:
                        raise AssertionError("Native stored rank outside its declared range")
                ranks.append({"job": job["id"], "system": job["system"], "arm": arm,
                              "history": choice["history"], "cap": choice["cap"],
                              "selected_prefix_stage": choice["winner_stage"],
                              "used_by_selected_forecast": rank["block_index"] < choice["winner_stage"],
                              "block_index": rank["block_index"],
                              "cluster_index": rank["regime_index"],
                              "ambient_dimension": rank["ambient_dimension"],
                              "local_reference_count": rank["local_reference_count"],
                              "initial_admissible_rank": rank["initial_admissible_rank"],
                              "effective_rank": rank["effective_rank"],
                              "rank_reduction_reasons": "|".join(rank["rank_reduction_reasons"])})
                counters["local_rank_records"] += 1

        with np.load(folder / "calibration_scores.npz", allow_pickle=False) as cs, \
             np.load(folder / "test_scores.npz", allow_pickle=False) as ts:
            if not np.array_equal(cs["pair_episode_ids"], np.arange(cfg["n_calibration"]).reshape(-1, 2)):
                raise AssertionError("Episode pairing is not the frozen disjoint pairing")
            for alias, row in evaluation.items():
                losses, witnesses, distance = (cs[alias + suffix] for suffix in
                                               ("__loss", "__witness", "__pair_distance"))
                if losses.shape != (cfg["n_calibration"], len(cfg["horizons"])):
                    raise AssertionError("Calibration loss sampling unit mismatch")
                if witnesses.shape != (cfg["n_calibration"] // 2, len(cfg["horizons"])):
                    raise AssertionError("Pair witness sampling unit mismatch")
                expected_witness = 0.5 * np.mean(np.maximum(
                    cs["pair_target_abs_difference"] - row["kappa"] * distance[..., None], 0.0) ** 2, axis=1)
                close(witnesses, expected_witness, alias + "/pair witness")
                counters["pair_witness_reconstructions"] += 1
                test_losses = ts[alias + "__loss"]
                if test_losses.shape != (cfg["n_test"], len(cfg["horizons"])):
                    raise AssertionError("Test sampling unit mismatch")
                close(test_losses.mean(axis=0), test_by_alias[alias]["mse"], alias + "/test MSE")
                close(losses.mean(axis=0), cal_by_alias[alias]["mse"], alias + "/calibration MSE")
                for held in (cal_by_alias[alias], test_by_alias[alias]):
                    diagnostics = held["diagnostics"]
                    if (diagnostics["prediction_nonfinite_count"] != 0
                            or diagnostics["prediction_min"] < -1 - 1e-12
                            or diagnostics["prediction_max"] > 1 + 1e-12):
                        raise AssertionError("Prediction range violation")
                    counters["prediction_range_checks"] += 1
                for index, horizon in enumerate(cfg["horizons"]):
                    upper = interval(losses[:, index], 4.0, tail)
                    lower = interval(witnesses[:, index], 2.0, tail)
                    reported = cal_by_alias[alias]["certificates"][index]
                    close(upper["upper"], reported["mse_upper"], alias + "/upper")
                    close(lower["lower"], reported["mse_lower"], alias + "/lower")
                    for key in ("mean", "variance", "radius"):
                        close(upper[key], reported["upper_details"][key], alias + "/upper " + key)
                        close(lower[key], reported["lower_details"][key], alias + "/lower " + key)
                    counters["risk_endpoints"] += 2
                    test_mse = float(test_losses[:, index].mean())
                    endpoint_rows.append({
                        "job": job["id"], "system": job["system"], "repetition": job["repetition"],
                        "alias": alias, "arm": row["arm"], "selected_forecast": row["selected_forecast"],
                        "horizon": horizon, "history": row["history"], "cap": row["cap"],
                        "prefix_stage": row["stage"], "n_features": row["n_features"],
                        "validation_mse": row["validation_mse"][index],
                        "calibration_mse": upper["mean"], "test_mse": test_mse,
                        "test_rmse": math.sqrt(test_mse),
                        "test_mse_se": float(test_losses[:, index].std(ddof=1) / np.sqrt(len(test_losses))),
                        "mse_lower": lower["lower"], "mse_upper": upper["upper"],
                        "endpoint_separation": upper["upper"] - lower["lower"],
                        "zero_lower": lower["lower"] == 0.0,
                        "upper_reaches_loss_bound": upper["upper"] == 4.0,
                        "test_mean_below_lower": test_mse < lower["lower"],
                        "test_mean_above_upper": test_mse > upper["upper"],
                        "tail_delta": tail,
                    })
                for tolerance in cfg["rmse_tolerances"]:
                    certificates = cal_by_alias[alias]["certificates"]
                    accepted = all(item["mse_upper"] <= tolerance**2 for item in certificates)
                    excluded = any(item["mse_lower"] > tolerance**2 for item in certificates)
                    decision = ("conflicting_bounds" if accepted and excluded else
                                "accepted" if accepted else "excluded" if excluded else "unresolved")
                    if cal_by_alias[alias]["decisions"][str(tolerance)] != decision:
                        raise AssertionError("Tolerance decision mismatch")
                    decision_rows.append({"job": job["id"], "system": job["system"],
                                          "alias": alias, "arm": row["arm"],
                                          "selected_forecast": row["selected_forecast"],
                                          "rmse_tolerance": tolerance, "decision": decision})
                if row["selected_forecast"]:
                    selected_rows.append({
                        "job": job["id"], "system": job["system"], "repetition": job["repetition"],
                        "arm": row["arm"], "history": row["history"], "cap": row["cap"],
                        "prefix_stage": row["stage"], "n_features": row["n_features"],
                        "validation_mean_mse": float(np.mean(row["validation_mse"])),
                        "test_mean_mse": float(test_losses.mean()),
                        **{f"test_mse_h{h}": float(test_losses[:, i].mean()) for i, h in enumerate(cfg["horizons"])},
                        "coefficient_norm": row["coefficient_norm"],
                        "optimization_gap": row["optimization_gap"],
                        "solver_converged": row["solver_converged"],
                    })
            for comparison in calibration["paired_improvements"]:
                i = cfg["horizons"].index(comparison["horizon"])
                old, new = comparison["old"], comparison["new"]
                raw_difference = cs[old + "__loss"][:, i] - cs[new + "__loss"][:, i]
                reconstructed = interval(raw_difference + 4.0, 8.0, tail)
                improvement_lower = reconstructed["lower"] - 4.0
                close(improvement_lower, comparison["improvement_lower"], comparison["arm"] + "/improvement")
                close(reconstructed["mean"] - 4.0, comparison["empirical_improvement"], comparison["arm"] + "/paired mean")
                counters["improvement_endpoints"] += 1
                difference = ts[old + "__loss"][:, i] - ts[new + "__loss"][:, i]
                contrast_rows.append({"job": job["id"], "system": job["system"],
                                      "repetition": job["repetition"], "type": comparison["type"],
                                      "contrast": comparison["arm"], "old": old, "new": new,
                                      "horizon": comparison["horizon"],
                                      "calibration_improvement": reconstructed["mean"] - 4.0,
                                      "improvement_lower": improvement_lower,
                                      "certified_positive_improvement": improvement_lower > 0,
                                      "test_mse_improvement": float(difference.mean()),
                                      "test_paired_difference_se": float(difference.std(ddof=1) / np.sqrt(len(difference)))})

            if job["full_prediction_audit"]:
                counters["full_array_jobs"] += 1
                for split, scores in (("calibration", cs), ("test", ts)):
                    with np.load(folder / (split + "_predictions.npz"), allow_pickle=False) as arrays:
                        targets = arrays["targets"]
                        if targets.shape != (cfg["n_" + split], 13, len(cfg["horizons"])):
                            raise AssertionError("Full target-array shape mismatch")
                        for alias, row in evaluation.items():
                            prediction = arrays[row["id"]]
                            close(np.mean((prediction - targets) ** 2, axis=1), scores[alias + "__loss"],
                                  alias + "/full-array forecast join")
                            if prediction.min() < -1 - 1e-12 or prediction.max() > 1 + 1e-12:
                                raise AssertionError("Retained full prediction violates range")
                            counters["full_array_forecast_joins"] += 1

    if counters["risk_endpoints"] != cfg["campaign_risk_tails"]:
        raise AssertionError("Incomplete campaign risk endpoint audit")
    if counters["improvement_endpoints"] != cfg["campaign_improvement_tails"]:
        raise AssertionError("Incomplete campaign improvement audit")
    if counters["full_array_jobs"] != len(cfg["systems"]):
        raise AssertionError("Incomplete prespecified full-array audit")
    policy_summary, comparison_summary, resolution = [], [], []
    for system in cfg["systems"]:
        name = system["name"]
        system_rows = [row for row in selected_rows if row["system"] == name]
        for arm in dict.fromkeys(row["arm"] for row in system_rows):
            chosen = [row for row in system_rows if row["arm"] == arm]
            values = np.asarray([row["test_mean_mse"] for row in chosen])
            policy_summary.append({
                "system": name, "arm": arm, "repetitions": len(chosen),
                "mean_test_mse": float(values.mean()), "sd_test_mse_over_repetitions": float(values.std(ddof=1)),
                **{f"mean_test_mse_h{h}": float(np.mean([r[f"test_mse_h{h}"] for r in chosen]))
                   for h in cfg["horizons"]},
                "selected_caps": ";".join(str(row["cap"]) for row in chosen),
                "selected_histories": ";".join(str(row["history"]) for row in chosen),
                "selected_prefix_stages": ";".join(str(row["prefix_stage"]) for row in chosen),
                "nonconverged_selected_fits": sum(not row["solver_converged"] for row in chosen),
            })
        for old, new in cfg["policy_contrasts"]:
            old_by_job = {row["job"]: row["test_mean_mse"] for row in system_rows if row["arm"] == old}
            new_by_job = {row["job"]: row["test_mean_mse"] for row in system_rows if row["arm"] == new}
            ordered = sorted(old_by_job)
            old_values = np.asarray([old_by_job[job] for job in ordered])
            new_values = np.asarray([new_by_job[job] for job in ordered])
            difference = new_values - old_values
            comparison_summary.append({
                "system": name, "old": old, "new": new, "repetitions": len(ordered),
                "old_mean_test_mse": float(old_values.mean()), "new_mean_test_mse": float(new_values.mean()),
                "mean_test_mse_change": float(difference.mean()),
                "percent_change_in_mean_test_mse": float(100 * (new_values.mean() / old_values.mean() - 1)),
                "favorable_repetitions": int(np.sum(difference < 0)),
                "equal_repetitions": int(np.sum(difference == 0)),
                "unfavorable_repetitions": int(np.sum(difference > 0)),
            })
        for arm in cfg["policies"]:
            for selected in (False, True):
                entries = [row for row in endpoint_rows if row["system"] == name and row["arm"] == arm
                           and (row["selected_forecast"] or not selected)]
                for tolerance in cfg["rmse_tolerances"]:
                    decisions = [row for row in decision_rows if row["system"] == name and row["arm"] == arm
                                 and row["rmse_tolerance"] == tolerance
                                 and (row["selected_forecast"] or not selected)]
                    resolution.append({"system": name, "arm": arm,
                                       "scope": "selected_forecast" if selected else "all_retained_prefixes",
                                       "rmse_tolerance": tolerance,
                                       "accepted": sum(row["decision"] == "accepted" for row in decisions),
                                       "excluded": sum(row["decision"] == "excluded" for row in decisions),
                                       "unresolved": sum(row["decision"] == "unresolved" for row in decisions),
                                       "conflicting_bounds": sum(row["decision"] == "conflicting_bounds" for row in decisions),
                                       "zero_lower_endpoints": sum(row["zero_lower"] for row in entries),
                                       "upper_endpoints_at_bound": sum(row["upper_reaches_loss_bound"] for row in entries),
                                       "mean_endpoint_separation": float(np.mean([row["endpoint_separation"] for row in entries]))})
    output.mkdir(parents=True, exist_ok=False)
    tables = {"endpoints": endpoint_rows, "decisions": decision_rows, "paired_contrasts": contrast_rows,
              "selected_forecasts": selected_rows, "selected_policy_summary": policy_summary,
              "policy_comparison_summary": comparison_summary, "certificate_resolution": resolution,
              "selected_bank_local_ranks": ranks, "stage_runtime": fitting}
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    summary = {
        "verification": {"passed": True, **counters},
        "confidence": {"delta": cfg["delta"], "scope": cfg["confidence_scope"],
                       "risk_endpoints": cfg["campaign_risk_tails"],
                       "improvement_endpoints": cfg["campaign_improvement_tails"], "tail_delta": tail},
        "forecasting": {"policy_summaries": policy_summary, "policy_comparisons": comparison_summary},
        "certificate_diagnostics": {
            "risk_lower_endpoints": len(endpoint_rows),
            "zero_lower_endpoints": sum(row["zero_lower"] for row in endpoint_rows),
            "upper_endpoints_at_bound": sum(row["upper_reaches_loss_bound"] for row in endpoint_rows),
            "test_mean_below_lower": sum(row["test_mean_below_lower"] for row in endpoint_rows),
            "test_mean_above_upper": sum(row["test_mean_above_upper"] for row in endpoint_rows),
            "positive_paired_improvement_endpoints": sum(row["certified_positive_improvement"] for row in contrast_rows),
            "conflicting_model_tolerance_decisions": sum(row["decision"] == "conflicting_bounds" for row in decision_rows),
            "interpretation": "Test means are independent finite-sample diagnostics, not the population class optimum or a coverage estimate",
        },
        "audit_scope": "Every selection and designated endpoint replayed from score arrays; full prediction-to-loss joins for the six prespecified repetition-zero jobs",
        "freeze_sha256": digest(run / "freeze.json"),
        "test_seal_sha256": digest(run / "test_seal.json"),
        "analysis_source_sha256": digest(Path(__file__)),
    }
    write_json(output / "summary.json", summary)
    plot_summary(output, cfg, policy_summary)
    write_json(output / "analysis_files.json", {str(path.name): digest(path) for path in sorted(output.iterdir())
                                                 if path.is_file()})
    return summary


def plot_summary(output, cfg, rows):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/kahm_risk_dimension_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.0), constrained_layout=True)
    colors = ["#667085", "#B54708", "#175CD3"]
    labels = ["Cap 2", "Native 20", "Selected cap"]
    for axis, system in zip(axes.flat, cfg["systems"]):
        selected = {row["arm"]: row for row in rows if row["system"] == system["name"]}
        values = [selected[arm]["mean_test_mse"] for arm in cfg["policies"]]
        spread = [selected[arm]["sd_test_mse_over_repetitions"] for arm in cfg["policies"]]
        axis.bar(np.arange(3), values, yerr=spread, color=colors, capsize=3)
        axis.set_xticks(np.arange(3), labels)
        axis.set_title(system["name"].replace("_", " ").title())
        axis.set_ylabel("Test MSE")
        axis.set_ylim(bottom=0)
    figure.suptitle("Validation-selected forecasts within each dimension policy\nBars average repetitions; error bars show their sample standard deviation", fontsize=11)
    figure.savefig(output / "forecasting_dimension_policies.pdf")
    figure.savefig(output / "forecasting_dimension_policies.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run.resolve(), args.output or args.run / "analysis")
    print(json.dumps(result["verification"], indent=2))


if __name__ == "__main__":
    main()
