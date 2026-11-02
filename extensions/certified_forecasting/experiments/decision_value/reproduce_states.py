#!/usr/bin/env python3
"""Reconstruct the decision study's omitted numeric archives from frozen seeds.

This utility does not refit models or generate a new experiment. It reconstructs
the original training, validation, and calibration streams, then requires each
compressed archive to match the SHA-256 recorded by that experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    config = json.loads((args.results / "protocol.json").read_text())
    models = json.loads((args.results / "frozen_models.json").read_text())
    expected = {item["path"]: item["sha256"] for item in
                json.loads((args.results / "summary.json").read_text())["files"]}
    chosen = {(m["regime"], m["repetition"]): m for m in models if m["prefix"] == 1}
    if len(chosen) != len(config["regimes"]) * config["n_repetitions"]:
        raise ValueError("Incomplete frozen model schedule.")
    args.output.mkdir(parents=True)
    development, checks = {}, {}
    for regime in config["regimes"]:
        arrays = {}
        for phase, field, length in (
            ("train", "training_seed", config["n_train"]),
            ("validation", "validation_seed", config["n_validation"]),
            ("calibration", "calibration_seed", max(config["calibration_sizes"])),
        ):
            array = np.empty((config["n_repetitions"], length), dtype=np.uint8)
            for repetition in range(config["n_repetitions"]):
                rng = np.random.default_rng(chosen[(regime, repetition)][field])
                s = (rng.random(length) < config["rare_probability"]).astype(np.uint8)
                z, r, noise = rng.integers(0, 2, size=(3, length), dtype=np.uint8)
                array[repetition] = 8 * s + 4 * z + 2 * r + noise
            arrays[phase] = array
        development[regime + "_train"] = arrays["train"]
        development[regime + "_validation"] = arrays["validation"]
        np.savez_compressed(args.output / (regime + "_calibration_states.npz"),
                            states=arrays["calibration"])
    np.savez_compressed(args.output / "development_states.npz", **development)
    for path in sorted(args.output.glob("*.npz")):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checks[path.name] = {"sha256": actual, "matches_original": actual == expected[path.name]}
    report = {"status": "passed" if all(c["matches_original"] for c in checks.values()) else "failed",
              "archives": checks, "scope": "Exact replay of the original frozen streams; no new data or fits."}
    (args.output / "reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit("Archive hashes differ; use the recorded NumPy environment and inspect the replay.")


if __name__ == "__main__":
    main()
