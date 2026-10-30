#!/usr/bin/env python3
"""A small, reproducible introduction to additive KAHM forecasting.

From the repository root::

    python examples/forecasting_utility.py
    python examples/forecasting_utility.py --csv /tmp/toy_forecasts.csv

Requires the packages in extensions/forecasting_utility/requirements.txt.
The generated harmonic series is an API illustration, not a research benchmark.
All settings below are fixed; validation selects the declared penalties/mixing
weights, and the chronological test block is evaluated after that selection.
The selected training-only fits are retained without a train+validation refit.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

# Support `python examples/forecasting_utility.py` without installing the repo.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from extensions.forecasting_utility.models import (  # noqa: E402
    AdditiveForecastRegressor,
    RawForecastRegressor,
)


SEED = 23
HISTORY = 12
HORIZONS = (1, 3, 6)
ALPHAS = (0.01, 1.0, 100.0)
GAMMAS = (0.0, 0.25, 0.5, 1.0)
GEOMETRY = {
    "cluster_counts": (2, 4), "rank": 2, "max_refs": 12,
    "max_fit": 96, "omega": 16.0, "tau": 1e-6,
}


def windows(values, target_start, target_stop):
    """Keep every target inside its chronological block; history stays causal."""
    first_origin = max(HISTORY - 1, target_start - 1)
    origins = np.arange(first_origin, target_stop - max(HORIZONS))
    inputs = values[origins[:, None] - HISTORY + 1 + np.arange(HISTORY)]
    targets = values[origins[:, None] + np.asarray(HORIZONS)]
    return inputs, targets, origins


def mse(prediction, target):
    return float(np.mean((prediction - target) ** 2))


def write_forecasts(path, origins, targets, predictions):
    """Write one row per test origin/horizon, without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["origin_index", "horizon", "target_index", "target", *predictions])
        for row, origin in enumerate(origins):
            for column, horizon in enumerate(HORIZONS):
                writer.writerow([
                    int(origin), horizon, int(origin + horizon), targets[row, column],
                    *[prediction[row, column] for prediction in predictions.values()],
                ])


def run(csv_path=None):
    if csv_path is not None and csv_path.exists():
        raise FileExistsError(f"Choose a new CSV path; this file already exists: {csv_path}")

    rng = np.random.default_rng(SEED)
    time = np.arange(360)
    values = (np.sin(2 * np.pi * time / 24)
              + 0.25 * np.sin(2 * np.pi * time / 9 + 0.4)
              + 0.06 * rng.standard_normal(len(time)))
    x_train, y_train, _ = windows(values, 0, 216)
    x_validation, y_validation, _ = windows(values, 216, 288)
    x_test, y_test, test_origins = windows(values, 288, len(values))

    # Select one raw backbone. Every correction receives this exact fitted object.
    raw = RawForecastRegressor(head="rbf", n_landmarks=16,
                               seed=SEED, head_seed=SEED + 1000)
    raw_trials = raw.fit_select(x_train, y_train, x_validation, y_validation, ALPHAS)
    raw_train = raw.predict(x_train)
    raw_validation = raw.predict(x_validation)
    validation_losses = {
        "persistence": mse(np.repeat(x_validation[:, -1:], len(HORIZONS), axis=1), y_validation),
        "raw_rbf": min(trial["validation_mse"] for trial in raw_trials),
    }
    corrections = {}
    for gate in ("kahm", "affine", "locality"):
        model = AdditiveForecastRegressor(gate, seed=SEED, geometry_parameters=GEOMETRY)
        trials = model.fit_select(
            x_train, y_train, x_validation, y_validation, ALPHAS, GAMMAS,
            raw_model=raw, raw_train_prediction=raw_train,
            raw_validation_prediction=raw_validation,
        )
        name = f"raw_plus_{gate}"
        corrections[name] = model
        validation_losses[name] = min(trial["validation_mse"] for trial in trials)

    # No test target has entered fitting or selection. Share one raw test query.
    raw_test = raw.predict(x_test)
    predictions = {
        "persistence": np.repeat(x_test[:, -1:], len(HORIZONS), axis=1),
        "raw_rbf": raw_test,
    }
    for name, model in corrections.items():
        predictions[name] = model.predict(x_test, raw_prediction=raw_test)
        baseline = model.predict(x_test, gamma=0, raw_prediction=raw_test)
        if not np.array_equal(baseline, raw_test):
            raise AssertionError("Gamma zero must preserve the actual raw forecast exactly")

    print("Toy forecasting quickstart: one generated harmonic series")
    print("Chronological targets; training-only fits; validation selection; no final refit.")
    print(f"Rows: train={len(x_train)}, validation={len(x_validation)}, test={len(x_test)}")
    print(f"History={HISTORY}; horizons={HORIZONS}; shared raw alpha={raw.alpha:g}\n")
    print(f"{'Method':<22} {'Validation MSE':>15} {'Test MSE':>12} {'Correction alpha':>18} {'Gamma':>7}")
    for name, prediction in predictions.items():
        model = corrections.get(name)
        alpha = "unused" if model is not None and model.gamma == 0 else (
            f"{model.alpha:g}" if model is not None else "-")
        gamma = f"{model.gamma:g}" if model is not None else "-"
        print(f"{name:<22} {validation_losses[name]:>15.6g} {mse(prediction, y_test):>12.6g} {alpha:>18} {gamma:>7}")
    print("\nAll three gamma-zero predictions match the shared raw prediction exactly.")
    print("This small example demonstrates the API; it is not evidence of a general accuracy advantage.")
    if csv_path is not None:
        write_forecasts(csv_path, test_origins, y_test, predictions)
        print(f"Test forecasts written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="Optional new CSV file containing every test forecast and target")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.csv)


if __name__ == "__main__":
    main()
