#!/usr/bin/env python3
"""Compare a production replay with the frozen scaling-study implementation.

Example (from the extension directory)::

    python scripts/verify_scaling_production_replay.py \
      --standard-reference STUDY/results/standard/main_v1/harmonic_1701 \
      --minmax-reference STUDY/results/minmax/main_v1/harmonic_1701 \
      --standard-replay REPLAY/legacy_full/harmonic_1701 \
      --minmax-replay REPLAY/minmax_full/harmonic_1701 \
      --output REPLAY/production_replay.json

Only elapsed timing and the explicitly added min-max scaler metadata are
excluded. Every scientific JSON value and every float64 score must agree
exactly; no numerical tolerance is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical(value, *, scaling_metadata=False):
    excluded = {"train_seconds", "elapsed_seconds"}
    if scaling_metadata:
        excluded.add("input_scaling")
    if isinstance(value, dict):
        return {key: canonical(item, scaling_metadata=scaling_metadata)
                for key, item in value.items() if key not in excluded}
    if isinstance(value, list):
        return [canonical(item, scaling_metadata=scaling_metadata) for item in value]
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference, replay, mode):
    result = {"mode": mode, "reference": str(reference), "replay": str(replay),
              "excluded_fields": ["train_seconds", "elapsed_seconds"],
              "files": {}}
    if mode == "minmax":
        result["excluded_fields"].append("input_scaling")
    for name in ("results.json", "frozen_candidates.json"):
        expected = json.loads((reference / name).read_text())
        actual = json.loads((replay / name).read_text())
        if canonical(expected, scaling_metadata=(mode == "minmax")) != canonical(
                actual, scaling_metadata=(mode == "minmax")):
            raise AssertionError(f"Non-timing scientific values differ: {mode}/{name}")
        result["files"][name] = {"scientific_values_exact": True,
                                  "reference_sha256": sha256(reference / name),
                                  "replay_sha256": sha256(replay / name)}
    frozen = json.loads((replay / "frozen_candidates.json").read_text())
    if mode == "minmax":
        scaled = [candidate for candidate in frozen if candidate["family"].startswith("kahm")]
        controls = [candidate for candidate in frozen if not candidate["family"].startswith("kahm")]
        assert all(candidate["input_scaling"]["mode"] == mode for candidate in scaled)
        assert all("input_scaling" not in candidate for candidate in controls)
        result["explicit_scaler_metadata_candidates"] = len(scaled)
    else:
        assert all("input_scaling" not in candidate for candidate in frozen)
        result["legacy_output_schema_preserved"] = True
    name = "episode_scores.npz"
    values = 0
    with np.load(reference / name, allow_pickle=False) as expected, np.load(
            replay / name, allow_pickle=False) as actual:
        assert set(expected.files) == set(actual.files)
        for key in expected.files:
            left, right = expected[key], actual[key]
            assert left.dtype == right.dtype == np.dtype("float64"), key
            assert np.isfinite(right).all(), key
            assert np.array_equal(left, right), key
            values += right.size
        arrays = len(expected.files)
    result["files"][name] = {"arrays": arrays, "values": values, "all_values_exact": True,
                              "reference_sha256": sha256(reference / name),
                              "replay_sha256": sha256(replay / name)}
    result["status"] = "pass"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("standard-reference", "minmax-reference", "standard-replay", "minmax-replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    record = {"scope": "Full harmonic_1701 production replays in legacy and explicit min-max modes",
              "comparisons": [compare(getattr(args, mode + "_reference"),
                                      getattr(args, mode + "_replay"), mode)
                              for mode in ("standard", "minmax")], "status": "pass"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "pass", "arrays": sum(item["files"]["episode_scores.npz"]["arrays"]
                                                     for item in record["comparisons"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
