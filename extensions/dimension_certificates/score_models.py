"""Evaluation of frozen feasible forecasts on fresh independent episodes."""
from __future__ import annotations

import pickle
import time

from common import (array_digest, flat, load_runtime, make_episodes, origins,
                    read_json, save_npz, utc, windows, write_json)
from fit_models import feature_matrix
import numpy as np

load_runtime()
from certified_forecasting.certificates import certify_forecast, episode_mse, paired_improvement


def selected_tail(details, side):
    return {key: details[key] for key in (
        "mean", "variance", "radius", side, "n_independent", "bound", "delta_per_tail", "method")}


def pair_distances(features, blocks):
    end = 2 * (len(features) // 2)
    difference = features[:end:2] - features[1:end:2]
    if blocks is None:
        return np.linalg.norm(difference, axis=-1)
    return np.max(np.stack([np.linalg.norm(difference[..., start:stop], axis=-1)
                            for start, stop in blocks]), axis=0)


def tolerance_decision(certificates, tolerance):
    threshold = tolerance ** 2
    accepted = all(item["mse_upper"] <= threshold for item in certificates)
    excluded = any(item["mse_lower"] > threshold for item in certificates)
    if accepted and excluded:
        return "conflicting_bounds"
    return "accepted" if accepted else "excluded" if excluded else "unresolved"


def planned_comparisons(development, cfg):
    comparisons = []
    for policy in cfg["policies"]:
        for stage in range(1, len(cfg["cluster_counts"])):
            comparisons.append({"type": "retained_prefix", "arm": policy,
                                "old": f"{policy}_s{stage}", "new": f"{policy}_s{stage+1}"})
    for old, new in cfg["policy_contrasts"]:
        comparisons.append({"type": "selected_policy", "arm": f"{old}_to_{new}",
                            "old": f"{old}_s{development['choices'][old]['winner_stage']}",
                            "new": f"{new}_s{development['choices'][new]['winner_stage']}"})
    if len(comparisons) * len(cfg["horizons"]) != cfg["improvement_tails_per_repetition"]:
        raise AssertionError("Planned comparison count changed")
    return comparisons


def evaluate(job, cfg, destination, split):
    if split not in ("calibration", "test"):
        raise ValueError("Unknown held-out split")
    started = time.perf_counter()
    development = read_json(destination / "development.json")
    # The workflow verifies the global model seal before loading these files.
    with (destination / "fitted_models.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    episodes = make_episodes(job, cfg, split)
    data = windows(episodes, cfg)
    targets = data[cfg["histories"][0]][1]
    n = len(targets)
    archive = {"episode_ids": np.arange(n), "origin_indices": origins(cfg),
               "horizons": np.asarray(cfg["horizons"])}
    if split == "calibration":
        pair_ids = np.arange(2 * (n // 2)).reshape(-1, 2)
        archive["pair_episode_ids"] = pair_ids
        archive["pair_target_abs_difference"] = np.abs(
            targets[pair_ids[:, 0]] - targets[pair_ids[:, 1]])
    predictions = {"targets": targets, "episode_ids": np.arange(n),
                   "origin_indices": origins(cfg), "horizons": np.asarray(cfg["horizons"])}
    bank_cache, fitted_cache, rows = {}, {}, []
    tail = cfg["delta"] / cfg["campaign_total_tails"]

    for row in development["evaluation"]:
        alias, identifier = row["alias"], row["id"]
        if identifier not in fitted_cache:
            X, Y = data[row["history"]]
            F = feature_matrix(row, payload, flat(X), bank_cache)
            predicted = payload["models"][identifier].predict(F).reshape(Y.shape)
            if not np.all(np.isfinite(predicted)) or np.max(np.abs(predicted)) > 1 + 1e-12:
                raise AssertionError("Forecast range or finiteness violation")
            if not np.all(np.isfinite(F)):
                raise AssertionError("Nonfinite features")
            loss = episode_mse(Y, predicted)
            information = {"prediction_min": float(predicted.min()),
                           "prediction_max": float(predicted.max()),
                           "prediction_nonfinite_count": int(np.sum(~np.isfinite(predicted))),
                           "prediction_count": int(predicted.size),
                           "prediction_sha256": array_digest(predicted),
                           "features_sha256": array_digest(F),
                           "feature_min": float(F.min()), "feature_max": float(F.max()),
                           "target_sha256": array_digest(Y),
                           "input_sha256": array_digest(X)}
            if row["bank_id"] is not None:
                bank = payload["banks"][row["bank_id"]]
                information["max_simplex_sum_error"] = max(
                    float(np.max(np.abs(F[:, block].sum(axis=1) - 1)))
                    for block in bank.block_slices_[:row["stage"]])
                if F.min() < -1e-12 or F.max() > 1 + 1e-12:
                    raise AssertionError("Soft-state coordinate outside simplex bounds")
            if split == "calibration":
                features = F.reshape(*X.shape[:2], F.shape[-1])
                blocks = None if row["blocks"] is None else [slice(a, b) for a, b in row["blocks"]]
                endpoints, checked_loss, witness = certify_forecast(
                    features, Y, predicted, row["kappa"], 2.0, tail, tail,
                    method=cfg["certificate_method"], block_slices=blocks)
                np.testing.assert_array_equal(loss, checked_loss)
                distance = pair_distances(features, row["blocks"])
                oracle = 0.5 * np.mean(np.maximum(
                    archive["pair_target_abs_difference"] - row["kappa"] * distance[..., None],
                    0.0) ** 2, axis=1)
                np.testing.assert_array_equal(witness, oracle)
                certificates = [{
                    "horizon": horizon,
                    "mse_upper": endpoint["mse_upper"], "mse_lower": endpoint["mse_lower"],
                    "rmse_upper": endpoint["rmse_upper"], "rmse_lower": endpoint["rmse_lower"],
                    "upper_details": selected_tail(endpoint["upper_details"], "upper"),
                    "lower_details": selected_tail(endpoint["lower_details"], "lower"),
                } for horizon, endpoint in zip(cfg["horizons"], endpoints)]
            else:
                witness, distance, certificates = None, None, None
            fitted_cache[identifier] = (loss, witness, distance, certificates, information)
            if job["full_prediction_audit"]:
                predictions[identifier] = predicted
        loss, witness, distance, certificates, information = fitted_cache[identifier]
        archive[alias + "__loss"] = loss
        result = {"alias": alias, "fitted_id": identifier, "arm": row["arm"],
                  "selected_forecast": row["selected_forecast"],
                  "mse": loss.mean(axis=0).tolist(),
                  "mse_se": (loss.std(axis=0, ddof=1) / np.sqrt(n)).tolist(),
                  "diagnostics": information}
        if split == "calibration":
            archive[alias + "__witness"] = witness
            archive[alias + "__pair_distance"] = distance
            result["certificates"] = certificates
            result["decisions"] = {str(tolerance): tolerance_decision(certificates, tolerance)
                                   for tolerance in cfg["rmse_tolerances"]}
        rows.append(result)

    improvements = []
    if split == "calibration":
        for contrast in planned_comparisons(development, cfg):
            for index, horizon in enumerate(cfg["horizons"]):
                values = paired_improvement(
                    archive[contrast["old"] + "__loss"][:, index],
                    archive[contrast["new"] + "__loss"][:, index],
                    4.0, tail, method=cfg["certificate_method"])
                improvements.append({**contrast, "horizon": horizon,
                                     "empirical_improvement": values["empirical_improvement"],
                                     "improvement_lower": values["improvement_lower"],
                                     **selected_tail(values, "lower")})
        if len(rows) * len(cfg["horizons"]) * 2 != cfg["risk_tails_per_repetition"]:
            raise AssertionError("Risk endpoint roster changed")
    save_npz(destination / (split + "_scores.npz"), archive)
    if job["full_prediction_audit"]:
        save_npz(destination / (split + "_predictions.npz"), predictions)
    write_json(destination / (split + ".json"), {
        "job": job, "split": split, "completed_utc": utc(),
        "elapsed_seconds": time.perf_counter() - started,
        "episode_array_sha256": array_digest(episodes),
        "target_array_sha256": array_digest(targets),
        "n_independent_episodes": n, "n_origins_per_episode": len(origins(cfg)),
        "origin_indices": origins(cfg).tolist(), "stream_seed": job["streams"][split],
        "full_prediction_arrays_retained": job["full_prediction_audit"],
        "tail_delta": tail if split == "calibration" else None,
        "results": rows, "paired_improvements": improvements,
        "confidence_scope": cfg["confidence_scope"] if split == "calibration" else
                            "Independent test diagnostics; no claim that finite test loss is population risk",
    })
    return {"job": job["id"], "split": split, "seconds": time.perf_counter() - started,
            "models": len(rows)}
