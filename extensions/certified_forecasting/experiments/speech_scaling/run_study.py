"""Paired, descriptive scaling study on the existing speech recording cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "speech_forecasting"))
import run_study as speech
from certified_forecasting.features import NestedSoftFeatures
from certified_forecasting.models import BlockBoundedRidge

SEEDS = (8601, 8602, 8603)
FAMILIES = ("kahm", "rbf", "affine")
MODES = ("standard", "raw", "minmax", "minmax_clip")
ARCHIVE_HASH = "49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0"
REFERENCE = EXTENSION / "results/speech_forecasting_v1_replay"
digest, write_json = speech.digest, speech.write_json


def source_hashes():
    paths = [Path(__file__), HERE / "protocol.md", HERE / "verify_study.py",
             HERE.parent / "speech_forecasting/run_study.py",
             EXTENSION / "src/certified_forecasting/features.py",
             EXTENSION / "src/certified_forecasting/models.py"]
    parent = EXTENSION.parents[1]
    paths += [parent / "kernel_affine_hull_koopman_machines.py"]
    return {str(p.relative_to(parent)): digest(p) for p in paths}


def develop(energy, metadata, seed, output):
    selected, records, banks = {}, [], []
    clip_index, origin_index = speech.train_rows(metadata, seed)
    np.savez_compressed(output / f"training_rows_{seed}.npz",
                        clip_index=clip_index, origin_index=origin_index)
    val_idx = metadata.index[metadata.split == "validation"].to_numpy()
    val_groups = metadata.loc[val_idx, "speaker"].to_numpy()
    for history in (4, 12):
        indices = speech.ORIGINS[origin_index, None] - np.arange(history - 1, -1, -1)[None, :]
        x = energy[clip_index[:, None], indices]
        y = energy[clip_index[:, None], speech.ORIGINS[origin_index, None] + speech.HORIZONS]
        xv, yv = speech.histories(energy[val_idx], history)
        for family in FAMILIES:
            for mode in MODES:
                for sharpness in ((1.,) if family == "kahm" else (.25, 1., 4.)):
                    bank_id = f"{seed}_{family}_m{history}_{mode}_sharp{sharpness:g}"
                    started = time.perf_counter()
                    bank = NestedSoftFeatures(cluster_counts=(2, 4, 8, 16), kind=family,
                        seed=seed + history, omega=16, max_fit=1600,
                        max_train_per_cluster=24, subspace_dim=2,
                        sharpness=sharpness, input_scaling=mode).fit(x)
                    fit_time = time.perf_counter() - started
                    started = time.perf_counter()
                    f, fv = bank.transform(x), bank.transform(xv)
                    banks.append({"bank_id": bank_id, "input_scaling": bank.input_scaling_metadata(),
                        "fit_indices": bank.fit_indices_.tolist(), "geometry_fit_seconds": fit_time,
                        "development_transform_seconds": time.perf_counter() - started})
                    for stage, block in enumerate(bank.block_slices_, 1):
                        for kappa in (1., 4.):
                            for alpha in (1e-4, 1e-2):
                                started = time.perf_counter()
                                model = BlockBoundedRidge(alpha=alpha, kappa=kappa, bounds=(0, 1),
                                    block_slices=bank.block_slices_[:stage], tol=1e-7).fit(f[:, :block.stop], y)
                                elapsed = time.perf_counter() - started
                                if sum(np.linalg.norm(model.coef_[sl]) for sl in bank.block_slices_[:stage]) > kappa + 1e-10:
                                    raise AssertionError("Infeasible fitted coefficient budget")
                                _, losses = speech.group_losses(model.predict(fv[:, :block.stop]), yv, val_groups)
                                mse = losses.mean(axis=0)
                                name = f"{bank_id}_s{stage}_k{kappa:g}_a{alpha:g}"
                                record = {"seed": seed, "family": family, "scaling": mode, "candidate": name,
                                    "history": history, "features": block.stop, "validation_mse": mse.tolist(),
                                    "validation_mean": float(mse.mean()), "parameters": {"scaling": mode,
                                        "sharpness": sharpness, "stage": stage, "kappa": kappa, "alpha": alpha},
                                    "head_fit_seconds": elapsed, "bank_id": bank_id,
                                    "optimization_gap": float(model.optimization_gap_),
                                    "coefficient_norm": float(model.norm_sum_), "converged": bool(model.converged_)}
                                records.append(record)
                                key = (float(mse.mean()), block.stop, history, name)
                                arm = (family, mode)
                                if arm not in selected or key < selected[arm][0]:
                                    selected[arm] = (key, speech.Forecast(family, history, model, bank, stage=stage), record)
                    print(f"completed development bank {bank_id}", flush=True)
    if len(records) != 896 or len(selected) != 12 or len(banks) != 56:
        raise AssertionError("Incomplete prescribed development grid")
    write_json(output / f"development_{seed}.json", {"candidates": records, "banks": banks,
        "selected": [selected[(family, mode)][2] for family in FAMILIES for mode in MODES],
        "training_rows": len(clip_index)})
    return selected


def evaluate(energy, metadata, selected_by_seed, output):
    records, scores = [], []
    indices = metadata.index[metadata.split == "test"].to_numpy()
    test_groups = metadata.loc[indices, "speaker"].to_numpy()
    for seed in SEEDS:
        for family in FAMILIES:
            for mode in MODES:
                _, forecast, development = selected_by_seed[seed][(family, mode)]
                x, y = speech.histories(energy[indices], forecast.history)
                frozen_parameters = forecast.feature.input_scaling_metadata()
                started = time.perf_counter()
                prediction = forecast.predict(x)
                elapsed = time.perf_counter() - started
                groups, losses = speech.group_losses(prediction, y, test_groups)
                if not np.isfinite(losses).all() or np.any(losses < 0) or np.any(losses > 1):
                    raise AssertionError("Invalid group losses")
                if forecast.feature.input_scaling_metadata() != frozen_parameters:
                    raise AssertionError("Held-out transformation changed fitted parameters")
                diagnostics = {}
                if mode in ("minmax", "minmax_clip"):
                    z = forecast.feature._scale_inputs(x, clip=False)
                    diagnostics = {"test_coordinate_fraction_outside_training_range": float(np.mean(np.abs(z) > 1)),
                                   "test_history_fraction_outside_training_range": float(np.mean(np.any(np.abs(z) > 1, axis=1))),
                                   "test_unclipped_min": float(z.min()), "test_unclipped_max": float(z.max())}
                records.append({**development, "test_mse": losses.mean(axis=0).tolist(),
                    "test_groups": len(groups), "test_prediction_seconds": elapsed,
                    "test_prediction_rows": len(x), "input_scaling": frozen_parameters, **diagnostics})
                for group, values in zip(groups, losses):
                    for horizon, value in zip(speech.HORIZONS, values):
                        scores.append({"seed": seed, "family": family, "scaling": mode, "group": group,
                                       "horizon": int(horizon), "mse": float(value)})
                print(f"scored frozen test model {seed} {family} {mode}", flush=True)
    pd.DataFrame(scores).to_csv(output / "group_losses.csv.gz", index=False,
                                compression={"method": "gzip", "mtime": 0})
    write_json(output / "results.json", {"study_type": "descriptive follow-up on reused groups",
        "confidence_claims": False, "calibration_evaluated": False, "n_candidates": 2688,
        "n_selected": 36, "n_group_horizon_losses": len(scores), "results": records})


def summarize(output):
    records = json.loads((output / "results.json").read_text())["results"]
    scores = pd.read_csv(output / "group_losses.csv.gz", dtype={"group": str})
    flat = []
    for record in records:
        for horizon, mse in zip(speech.HORIZONS, record["test_mse"]):
            flat.append({"seed": record["seed"], "family": record["family"], "scaling": record["scaling"],
                "horizon": int(horizon), "mse": mse, "rmse_db": 60 * np.sqrt(mse),
                "history": record["history"], "features": record["features"]})
    frame = pd.DataFrame(flat)
    frame.to_csv(output / "selected_results.csv", index=False)
    summary = frame.groupby(["family", "scaling", "horizon"], sort=True).agg(
        mean_mse=("mse", "mean"), sd_mse=("mse", "std"), mean_rmse_db=("rmse_db", "mean"),
        sd_rmse_db=("rmse_db", "std"), n_fits=("seed", "size")).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    paired = []
    for (seed, family, horizon), part in scores.groupby(["seed", "family", "horizon"], sort=True):
        pivot = part.pivot(index="group", columns="scaling", values="mse")
        for mode in MODES[1:]:
            difference = pivot[mode] - pivot["standard"]
            paired.append({"seed": int(seed), "family": family, "horizon": int(horizon), "scaling": mode,
                "mean_mse_difference": float(difference.mean()),
                "conditional_group_se": float(difference.std(ddof=1) / np.sqrt(len(difference))),
                "relative_mse_change_percent": float(100 * difference.mean() / pivot["standard"].mean()),
                "groups_improved": int((difference < 0).sum()), "n_groups": len(difference)})
    pd.DataFrame(paired).to_csv(output / "paired_differences.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if digest(args.archive) != ARCHIVE_HASH:
        raise ValueError("Source archive differs from the fixed speech study")
    sources = source_hashes()
    write_json(args.output / "protocol_freeze.json", {"source_hashes": sources,
        "archive_sha256": ARCHIVE_HASH, "scope": "descriptive reused-cohort follow-up; no new certificates",
        "seeds": SEEDS, "families": FAMILIES, "scaling_modes": MODES, "n_candidates": 2688,
        "reference_development_hashes": {str(s): digest(REFERENCE / f"development_{s}.json") for s in SEEDS}})
    write_json(args.output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "platform": platform.platform(), "blas_threads": 1})
    try:
        with threadpool_limits(limits=1):
            energy, metadata, provenance = speech.prepare(args.archive, args.output)
            for filename in ("clips.csv", "speakers.csv"):
                if digest(args.output / filename) != digest(REFERENCE / filename):
                    raise AssertionError("Data eligibility or split differs from the reference")
            selected = {s: develop(energy, metadata, s, args.output) for s in SEEDS}
            freeze = {"protocol_freeze_sha256": digest(args.output / "protocol_freeze.json"),
                      "energy_sha256": digest(args.output / "energy_observations.npz"), "models": {}, "development": {}}
            for seed in SEEDS:
                path = args.output / f"selected_models_{seed}.pkl"
                with path.open("xb") as stream:
                    pickle.dump(selected[seed], stream, protocol=5)
                freeze["models"][str(seed)] = digest(path)
                freeze["development"][str(seed)] = digest(args.output / f"development_{seed}.json")
            write_json(args.output / "development_freeze.json", freeze)
            print("All 36 models frozen before any follow-up test loss is computed.", flush=True)
            evaluate(energy, metadata, selected, args.output)
            summarize(args.output)
            if sources != source_hashes():
                raise AssertionError("A study source changed during execution")
            write_json(args.output / "completion.json", {"status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "development_freeze_sha256": digest(args.output / "development_freeze.json"),
                "results_sha256": digest(args.output / "results.json"),
                "group_losses_sha256": digest(args.output / "group_losses.csv.gz")})
    except Exception:
        write_json(args.output / "failure.json", {"status": "failed", "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
