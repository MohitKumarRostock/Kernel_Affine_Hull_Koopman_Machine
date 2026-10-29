"""Independent accounting and saved-model replay for paired speech scaling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "speech_forecasting"))
import run_study as speech

SEEDS = (8601, 8602, 8603)
FAMILIES = ("kahm", "rbf", "affine")
MODES = ("standard", "raw", "minmax", "minmax_clip")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_losses(forecast, energy, clips):
    indices = clips.index[clips.split == "test"].to_numpy()
    groups = clips.loc[indices, "speaker"].to_numpy()
    origins = np.arange(11, 88, 4)
    histories = energy[indices][:, origins[:, None] - np.arange(forecast.history - 1, -1, -1)]
    targets = energy[indices][:, origins[:, None] + np.array([1, 4, 12])]
    prediction = forecast.predict(histories.reshape(-1, forecast.history)).reshape(len(indices), 20, 3)
    clip_losses = ((prediction - targets) ** 2).mean(axis=1)
    frame = pd.DataFrame(clip_losses, columns=[1, 4, 12])
    frame["group"] = groups
    return frame.groupby("group", sort=True)[[1, 4, 12]].mean()


def verify(root, portable=False):
    reference = EXTENSION / "results/speech_forecasting_v1_replay"
    results = json.loads((root / "results.json").read_text())
    if results["confidence_claims"] or results["calibration_evaluated"]:
        raise AssertionError("Descriptive follow-up has acquired a confidence claim")
    loss = pd.read_csv(root / "group_losses.csv.gz", dtype={"group": str}, float_precision="round_trip")
    if len(loss) != 36 * 245 * 3 or loss.duplicated(["seed", "family", "scaling", "group", "horizon"]).any():
        raise AssertionError("Incomplete or duplicate held-out scores")
    if not np.isfinite(loss.mse).all() or (loss.mse < 0).any() or (loss.mse > 1).any():
        raise AssertionError("Invalid bounded losses")
    freeze = json.loads((root / "development_freeze.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    assert digest(root / "group_losses.csv.gz") == completion["group_losses_sha256"]
    assert digest(root / "results.json") == completion["results_sha256"]
    assert digest(root / "development_freeze.json") == completion["development_freeze_sha256"]
    assert digest(root / "protocol_freeze.json") == freeze["protocol_freeze_sha256"]
    prior_scores = pd.read_csv(reference / "speaker_losses.csv.gz", dtype={"speaker": str}, float_precision="round_trip")
    prior_scores = prior_scores[prior_scores.split == "test"]
    errors = {"summary_mse": 0., "prior_candidate_mse": 0., "prior_selected_group_loss": 0.,
              "saved_model_group_loss": 0., "fitted_extrema": 0., "paired_difference": 0., "paired_se": 0.}
    candidate_count = 0
    prior_candidate_count = 0
    if not portable:
        assert digest(root / "energy_observations.npz") == freeze["energy_sha256"]
        energy = np.load(root / "energy_observations.npz")["energy"]
        clips = pd.read_csv(root / "clips.csv", dtype={"speaker": str})
    for seed in SEEDS:
        development = json.loads((root / f"development_{seed}.json").read_text())
        prior = json.loads((reference / f"development_{seed}.json").read_text())
        assert digest(root / f"development_{seed}.json") == freeze["development"][str(seed)]
        assert len(development["candidates"]) == 896
        candidate_count += len(development["candidates"])
        candidates = {r["candidate"]: r for r in development["candidates"]}
        for row in prior["candidates"]:
            if row["family"] not in FAMILIES:
                continue
            new = candidates[row["candidate"]]
            error = np.max(np.abs(np.asarray(row["validation_mse"]) - new["validation_mse"]))
            errors["prior_candidate_mse"] = max(errors["prior_candidate_mse"], float(error))
            prior_candidate_count += 1
        for family in FAMILIES:
            old = next(r for r in prior["selected"] if r["family"] == family)
            matching = next(r for r in development["selected"] if r["candidate"] == old["candidate"])
            new_losses = loss[(loss.seed == seed) & (loss.family == family) & (loss.scaling == matching["scaling"])]
            old_losses = prior_scores[(prior_scores.seed == seed) & (prior_scores.family == family)]
            merged = new_losses.merge(old_losses, left_on=["group", "horizon"], right_on=["speaker", "horizon"], suffixes=("_new", "_old"))
            assert len(merged) == 245 * 3
            errors["prior_selected_group_loss"] = max(errors["prior_selected_group_loss"], float(np.max(np.abs(merged.mse_new - merged.mse_old))))
        for row in development["selected"]:
            subset = [r for r in development["candidates"] if (r["family"], r["scaling"]) == (row["family"], row["scaling"])]
            expected = min(subset, key=lambda r: (r["validation_mean"], r["features"], r["history"], r["candidate"]))
            assert expected["candidate"] == row["candidate"]
        if not portable:
            model_path = root / f"selected_models_{seed}.pkl"
            assert digest(model_path) == freeze["models"][str(seed)]
            with model_path.open("rb") as stream:
                selected = pickle.load(stream)
            saved_rows = np.load(root / f"training_rows_{seed}.npz")
            clip_index, origin_index = speech.train_rows(clips, seed)
            np.testing.assert_array_equal(saved_rows["clip_index"], clip_index)
            np.testing.assert_array_equal(saved_rows["origin_index"], origin_index)
            for (family, mode), (_, forecast, row) in selected.items():
                if mode.startswith("minmax"):
                    h = np.arange(11, 88, 4)[origin_index, None] - np.arange(forecast.history - 1, -1, -1)
                    training = energy[clip_index[:, None], h]
                    lo_error = np.max(np.abs(training.min(axis=0) - forecast.feature.input_min_))
                    hi_error = np.max(np.abs(training.max(axis=0) - forecast.feature.input_max_))
                    errors["fitted_extrema"] = max(errors["fitted_extrema"], float(lo_error), float(hi_error))
                reproduced = independent_losses(forecast, energy, clips)
                archived = loss[(loss.seed == seed) & (loss.family == family) & (loss.scaling == mode)].pivot(index="group", columns="horizon", values="mse")
                np.testing.assert_array_equal(reproduced.index, archived.index)
                errors["saved_model_group_loss"] = max(errors["saved_model_group_loss"], float(np.max(np.abs(reproduced.to_numpy() - archived.to_numpy()))))
    for row in results["results"]:
        values = loss[(loss.seed == row["seed"]) & (loss.family == row["family"]) & (loss.scaling == row["scaling"])]
        means = values.groupby("horizon", sort=True).mse.mean().to_numpy()
        errors["summary_mse"] = max(errors["summary_mse"], float(np.max(np.abs(means - row["test_mse"]))))
    paired = pd.read_csv(root / "paired_differences.csv")
    for row in paired.itertuples(index=False):
        values = loss[(loss.seed == row.seed) & (loss.family == row.family) & (loss.horizon == row.horizon)]
        pivot = values.pivot(index="group", columns="scaling", values="mse")
        difference = pivot[row.scaling] - pivot.standard
        errors["paired_difference"] = max(errors["paired_difference"], abs(float(difference.mean()) - row.mean_mse_difference))
        errors["paired_se"] = max(errors["paired_se"], abs(float(difference.std(ddof=1) / np.sqrt(len(difference))) - row.conditional_group_se))
    if max(errors.values()) > 2e-10:
        raise AssertionError(f"Replay error exceeds numerical tolerance: {errors}")
    audit = {"status": "passed", "portable": portable, "candidates_checked": candidate_count,
        "prior_candidates_replayed": prior_candidate_count, "selected_models": 36,
        "group_horizon_losses": len(loss), "max_absolute_errors": errors,
        "confidence_claims": False, "calibration_evaluated": False}
    speech.write_json(root / ("portable_verification.json" if portable else "verification.json"), audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--portable", action="store_true", help="Audit archived scores without local derived arrays or models")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.results, args.portable), indent=2))


if __name__ == "__main__":
    main()
