"""Check the reusable model against one archived stage2 selector outcome."""
import argparse
import json
from pathlib import Path
import numpy as np
from datasets import make_forecast_split
from followup_capacity import verify_capacity_freeze, child_seed, SYSTEM_IDS
from predictive_geometry_model import PredictiveGeometryRegressor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--system", choices=tuple(SYSTEM_IDS), default="vanderpol")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify_capacity_freeze()
    cfg = json.loads((root / "followup_config.json").read_text())
    base = json.loads((root / "config.json").read_text())
    fc = base["forecast"]
    splits = [make_forecast_split(args.system, fc[name + "_episodes"],
              child_seed(cfg, args.replicate, args.system, j), history=fc["history"],
              horizons=tuple(fc["horizons"]), stride=fc["stride"])
              for j, name in enumerate(("train", "validation", "test"))]
    train, val, test = splits
    model = PredictiveGeometryRegressor(
        seed=child_seed(cfg, args.replicate, args.system, 10),
        head_seed=child_seed(cfg, args.replicate, args.system, 11),
        bounds=train["bounds"], geometry_parameters=base["geometry"],
        landmark_counts=cfg["landmark_counts"], features=cfg["features"],
        alphas=base["heads"]["alphas"], bandwidth_multipliers=base["heads"]["bandwidth_multipliers"])
    model.fit(train["X"], train["Y"], val["X"], val["Y"], val["groups"])
    prediction = model.predict(test["X"])
    folder = root / "results_capacity" / f"r{args.replicate:02d}_forecast_{SYSTEM_IDS[args.system]}"
    with np.load(folder / "unit_losses.npz") as archived:
        expected = archived["prediction__validation_selector"]
        error = float(np.max(np.abs(prediction - expected)))
        assert np.allclose(prediction, expected, atol=1e-9, rtol=1e-9), error
    report = {"status": "passed", "system": args.system, "replicate": args.replicate,
              "maximum_absolute_prediction_difference": error, "selected": model.selected_}
    if args.save_model:
        destination = root / "models"
        destination.mkdir(exist_ok=True)
        model.save(destination / f"{args.system}_example.pkl")
        report["saved_model"] = str((destination / f"{args.system}_example.pkl").relative_to(root))
    (root / "analysis").mkdir(exist_ok=True)
    (root / "analysis" / f"wrapper_replay_{args.system}_r{args.replicate:02d}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
