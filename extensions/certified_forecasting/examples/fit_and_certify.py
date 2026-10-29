#!/usr/bin/env python3
"""Fit a reusable soft-state forecaster and certify an independent episode risk.

NPZ training/calibration files must have a numeric 'episodes' array of shape
(n_episodes, time, 1). Calibration rows must be iid episodes, independent of
all model construction, and representative of the deployment episode law.
Overlapping windows within each episode are allowed. Known target bounds
must be supplied; observed sample extrema are not population bounds.

Prediction mode loads a Python pickle. Load only a model file you created or
otherwise trust: unpickling an untrusted file can execute arbitrary code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import scipy
import sklearn

from certified_forecasting.certificates import certify_forecast
from certified_forecasting.data import histories_targets, make_episodes
from certified_forecasting.features import NestedSoftFeatures
from certified_forecasting.models import BlockBoundedRidge


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_hash(episodes):
    array = np.ascontiguousarray(episodes, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_episodes(path, bounds):
    with np.load(path, allow_pickle=False) as stored:
        if "episodes" not in stored:
            raise ValueError(f"{path}: expected an 'episodes' array")
        episodes = np.asarray(stored["episodes"], dtype=np.float64)
    if episodes.ndim != 3 or episodes.shape[2] != 1 or not episodes.shape[0]:
        raise ValueError("episodes must have shape (n_episodes, time, 1)")
    if not np.isfinite(episodes).all():
        raise ValueError("Episode arrays must be finite")
    if np.any(episodes < bounds[0]) or np.any(episodes > bounds[1]):
        raise ValueError("Observed values violate the declared known target range")
    return episodes


def fit_export(train_path, calibration_path, output, *, history, horizons,
               target_bounds, rmse_tolerance, origin_stride=1, kappa=4.0,
               delta=0.05, seed=0, alpha=1e-4, kahm_input_scaling=None):
    """Fit four retained KAHM stages, certify them jointly, and save all fits.

    Even an unresolved run saves all models and all certificates. The returned
    selection is the smallest stage whose upper RMSE bound meets the requested
    tolerance at every horizon. No candidate is refitted after calibration.
    """
    train_path, calibration_path, output = map(Path, (train_path, calibration_path, output))
    bounds = tuple(float(v) for v in target_bounds)
    if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("Known target bounds must be finite and strictly ordered")
    if not np.isfinite(rmse_tolerance) or rmse_tolerance <= 0:
        raise ValueError("The RMSE tolerance must be positive and finite")
    if not 0 < delta < 1 or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("Need 0 < delta < 1 and a finite nonnegative norm budget")
    if train_path.resolve() == calibration_path.resolve():
        raise ValueError("Training and calibration must be separate independent episode files")
    input_hashes = {"train": file_hash(train_path), "calibration": file_hash(calibration_path)}
    if input_hashes["train"] == input_hashes["calibration"]:
        raise ValueError("Training and calibration files have identical contents")
    # Demo input files may already occupy output/inputs; no model output is overwritten.
    for filename in ("models.pkl", "certificate.json", "frozen_candidates.json"):
        if (output / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {output / filename}")
    output.mkdir(parents=True, exist_ok=True)
    train = load_episodes(train_path, bounds)
    X, Y = histories_targets(train, history, horizons, origin_stride)
    xf, yf = X.reshape(-1, X.shape[-1]), Y.reshape(-1, Y.shape[-1])
    feature = NestedSoftFeatures(
        cluster_counts=(2, 4, 8, 16), kind="kahm", seed=seed,
        omega=16.0, max_fit=1600, max_train_per_cluster=24,
        input_scaling=kahm_input_scaling,
    ).fit(xf)
    F = feature.transform(xf)
    models, frozen = [], []
    for stage, block in enumerate(feature.block_slices_, 1):
        model = BlockBoundedRidge(
            alpha=alpha, kappa=kappa, bounds=bounds,
            block_slices=feature.block_slices_[:stage],
        ).fit(F[:, :block.stop], yf)
        models.append(model)
        frozen.append({
            "stage": stage, "n_features": block.stop, "kappa": float(kappa),
            "sum_block_norms": model.norm_sum_,
            "optimization_gap": model.optimization_gap_,
            "optimizer_converged": bool(model.converged_),
        })
    frozen_record = {
        "history": int(history), "horizons": list(map(int, horizons)),
        "candidates": frozen, "frozen_before_loading_calibration_episodes": True,
    }
    if kahm_input_scaling is not None:
        frozen_record["input_scaling"] = feature.input_scaling_metadata()
    save_json(output / "frozen_candidates.json", frozen_record)

    # Load calibration episodes only after the candidate family is frozen.
    calibration = load_episodes(calibration_path, bounds)
    if len(calibration) < 4:
        raise ValueError("At least four independent calibration episodes are required")
    if episode_hash(train) == episode_hash(calibration):
        raise ValueError("Training and calibration arrays are identical")
    Xc, Yc = histories_targets(calibration, history, horizons, origin_stride)
    Fc = feature.transform(Xc.reshape(-1, Xc.shape[-1]))
    K, H = len(models), len(horizons)
    tail = delta / (2.0 * K * H)
    scores, descriptions = {}, []
    for stage, (model, description) in enumerate(zip(models, frozen), 1):
        stop = feature.block_slices_[stage - 1].stop
        stage_features = Fc[:, :stop]
        prediction = model.predict(stage_features).reshape(Yc.shape)
        certificate, losses, witnesses = certify_forecast(
            stage_features.reshape(*Xc.shape[:2], stop), Yc, prediction,
            kappa, bounds[1] - bounds[0], tail, tail,
            block_slices=feature.block_slices_[:stage],
        )
        descriptions.append({
            **description, "certificates": certificate,
            "meets_all_horizon_tolerances": all(
                c["rmse_upper"] <= rmse_tolerance for c in certificate
            ),
        })
        scores[f"stage{stage}_episode_losses"] = losses
        scores[f"stage{stage}_pair_witnesses"] = witnesses
    eligible = [c for c in descriptions if c["meets_all_horizon_tolerances"]]
    chosen = min(eligible, key=lambda c: c["n_features"]) if eligible else None
    report = {
        "status": "certified" if chosen is not None else "unresolved",
        "selected_stage": None if chosen is None else chosen["stage"],
        "history": int(history), "horizons": list(map(int, horizons)),
        "origin_stride": int(origin_stride), "target_bounds": list(bounds),
        "rmse_tolerance": float(rmse_tolerance), "delta": float(delta),
        "tail_delta": tail, "n_frozen_candidates": K,
        "n_calibration_episodes": len(calibration),
        "n_disjoint_calibration_pairs": len(calibration) // 2,
        "n_origins_per_calibration_episode": Xc.shape[1],
        "calibration_episode_length": calibration.shape[1],
        "input_file_sha256": input_hashes,
        "input_episode_array_sha256": {
            "train": episode_hash(train), "calibration": episode_hash(calibration),
        },
        "candidates": descriptions,
        "interpretation": (
            "Joint population mean-squared-error bounds over all frozen stages and "
            "horizons, conditional on independent representative calibration episodes. "
            "These are not prediction intervals for an individual future observation."
        ),
        "sampling_assumptions": [
            "Calibration episodes are iid and independent of training/model construction.",
            "New episodes follow the same law and forecast-origin protocol as calibration.",
            "The supplied target range is a known population bound, not an empirical range.",
            "Distinct file hashes cannot establish independence or rule out partial overlap.",
        ],
    }
    save_json(output / "certificate.json", report)
    np.savez_compressed(output / "calibration_scores.npz", **scores)
    bundle = {
        "schema": "certified-soft-state-model-v1",
        "feature_bank": feature, "models": models,
        "selected_stage": report["selected_stage"],
        "history": int(history), "horizons": list(map(int, horizons)),
        "target_bounds": bounds, "certificate": report,
    }
    with (output / "models.pkl").open("wb") as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    extension_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(path.relative_to(extension_root)): file_hash(path)
        for path in sorted((extension_root / "src").rglob("*.py"))
    }
    for path in sorted(extension_root.parents[1].glob("*.py")):
        source_hashes["parent/" + path.name] = file_hash(path)
    save_json(output / "provenance.json", {
        "example_script_sha256": file_hash(__file__),
        "models_pickle_sha256": file_hash(output / "models.pkl"),
        "seed": int(seed), "alpha": float(alpha), "python": sys.version,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "source_sha256": source_hashes,
    })
    return report


def load_model(path):
    """Load only a trusted model pickle produced by fit_export."""
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    if bundle.get("schema") != "certified-soft-state-model-v1":
        raise ValueError("Unsupported model schema")
    return bundle


def predict_history(bundle, histories, stage=None):
    """Forecast scalar histories ordered oldest-to-newest.

    Returns shape (queries, horizons). If no stage passed certification, a
    caller must explicitly choose a fitted stage; it remains uncertified at
    the requested tolerance. Even a certified stage has a population-risk
    guarantee, not an individual-query error guarantee.
    """
    x = np.asarray(histories, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != bundle["history"] or not np.isfinite(x).all():
        raise ValueError("histories must be finite and have the fitted history length")
    chosen = bundle["selected_stage"] if stage is None else int(stage)
    if chosen is None:
        raise ValueError("No stage met the tolerance; explicitly choose a fitted stage for an uncertified forecast")
    if not 1 <= chosen <= len(bundle["models"]):
        raise ValueError("Stage is outside the fitted candidate family")
    bounds = bundle["target_bounds"]
    if np.any(x < bounds[0]) or np.any(x > bounds[1]):
        raise ValueError("History observations violate the fitted target-range assumption")
    features = bundle["feature_bank"].transform(x, n_blocks=chosen)
    return bundle["models"][chosen - 1].predict(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 12])
    parser.add_argument("--origin-stride", type=int, default=4)
    parser.add_argument("--target-min", type=float)
    parser.add_argument("--target-max", type=float)
    parser.add_argument("--rmse-tolerance", type=float)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kahm-input-scaling", choices=["standard", "raw", "minmax", "minmax_clip"],
                        help="Training-fitted KAHM input map; default is training standardization")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-train-episodes", type=int, default=128)
    parser.add_argument("--demo-calibration-episodes", type=int, default=4096)
    parser.add_argument("--demo-length", type=int, default=72)
    parser.add_argument("--predict-model", type=Path,
                        help="Prediction mode: load only a trusted model pickle")
    parser.add_argument("--histories", type=Path,
                        help="Prediction mode: NPZ with numeric 'histories' array")
    parser.add_argument("--stage", type=int,
                        help="Prediction mode: explicitly choose a stage (may be uncertified)")
    args = parser.parse_args()
    if args.predict_model is not None:
        if args.histories is None or args.output.suffix != ".npz":
            parser.error("Prediction mode needs --histories and an --output ending in .npz")
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        bundle = load_model(args.predict_model)
        with np.load(args.histories, allow_pickle=False) as stored:
            if "histories" not in stored:
                raise ValueError("Prediction NPZ needs a 'histories' array")
            prediction = predict_history(bundle, stored["histories"], args.stage)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, predictions=prediction,
                            horizons=np.asarray(bundle["horizons"]))
        print(json.dumps({"predictions": str(args.output), "shape": list(prediction.shape)}))
        return
    if args.target_min is None or args.target_max is None or args.rmse_tolerance is None:
        parser.error("Fitting requires --target-min, --target-max, and --rmse-tolerance")
    if args.demo:
        if args.train is not None or args.calibration is not None:
            parser.error("--demo cannot be combined with --train or --calibration")
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Demo output must be a new or empty directory")
        inputs = args.output / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        args.train, args.calibration = inputs / "train.npz", inputs / "calibration.npz"
        np.savez_compressed(args.train, episodes=make_episodes(
            "harmonic", args.demo_train_episodes, args.demo_length, args.seed * 1000 + 101))
        np.savez_compressed(args.calibration, episodes=make_episodes(
            "harmonic", args.demo_calibration_episodes, args.demo_length, args.seed * 1000 + 202))
    elif args.train is None or args.calibration is None:
        parser.error("Supply --train and --calibration, or use --demo")
    report = fit_export(
        args.train, args.calibration, args.output,
        history=args.history, horizons=args.horizons,
        target_bounds=(args.target_min, args.target_max),
        rmse_tolerance=args.rmse_tolerance, origin_stride=args.origin_stride,
        kappa=args.kappa, delta=args.delta, seed=args.seed,
        kahm_input_scaling=args.kahm_input_scaling,
    )
    print(json.dumps({
        "status": report["status"], "selected_stage": report["selected_stage"],
        "certificate": str(args.output / "certificate.json"),
        "fitted_models": str(args.output / "models.pkl"),
        "n_independent_calibration_episodes": report["n_calibration_episodes"],
    }, indent=2))


if __name__ == "__main__":
    main()
