"""Regenerate test inputs and replay archived first-stage bank/head predictions."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from geometry import GeometryBank
from run_study import make_splits, verify_freeze, array_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirmation", "natural"), default="confirmation")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--task", default="forecast_duffing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify_freeze()
    folder = root / "results" / args.phase / f"r{args.replicate:02d}_{args.task}"
    info = json.loads((folder / "result.json").read_text())
    config = json.loads((root / "config.json").read_text())
    splits, lambdas, _ = make_splits(config, args.phase, args.replicate, info["task"])
    test = splits[2]
    assert array_hash(test["X"]) == info["split_metadata"][2]["X_sha256"]
    assert array_hash(test["Y"]) == info["split_metadata"][2]["Y_sha256"]
    bank = GeometryBank.load(folder / "bank.pkl")
    features = bank.transform(test["X"])
    with (folder / "primary_heads.pkl").open("rb") as handle:
        models = pickle.load(handle)
    checks = []
    with np.load(folder / "unit_losses.npz") as archived:
        for feature, fitted in models.items():
            if feature == "validation_selector":
                prediction = np.empty_like(test["Y"])
                for j, item in enumerate(fitted):
                    values = item["fitted_head"].predict(features[item["feature"]])
                    columns = slice(j, j + 1) if lambdas is not None else slice(None)
                    prediction[:, columns] = values[:, columns]
            else:
                prediction = fitted.predict(features[feature])
            expected = archived["prediction__" + feature.replace(":", "__")]
            error = float(np.max(np.abs(prediction - expected)))
            if not np.allclose(prediction, expected, atol=1e-9, rtol=1e-9):
                raise AssertionError(f"Replay mismatch for {feature}: {error}")
            checks.append({"feature": feature, "maximum_absolute_error": error})
    print(json.dumps({"status": "passed", "phase": args.phase, "replicate": args.replicate,
                      "task": args.task, "checks": checks}, indent=2))
