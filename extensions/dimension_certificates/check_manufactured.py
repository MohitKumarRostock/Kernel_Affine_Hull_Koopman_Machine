#!/usr/bin/env python3
"""Exercise complete study wiring on labelled manufactured observations.

This check does not invoke any scheduled law or production stream. It checks
selection, nested-bank identity, actual local ranks, prediction/score joins,
budget feasibility, and every designated endpoint for one small fixture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from common import ROOT, read_json, write_json
import numpy as np
import fit_models
import score_models
from analyze import interval, close


def manufactured(job, cfg, split):
    seeds = {"train": 900001, "validation": 900002, "calibration": 900003, "test": 900004}
    rng = np.random.default_rng(seeds[split])
    return rng.uniform(-0.8, 0.8, size=(cfg["n_" + split], cfg["episode_length"], 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = read_json(ROOT / "config.json")
    cfg.update(n_train=40, n_validation=20, n_calibration=48, n_test=24)
    job = {"id": "manufactured_fixture", "system": "manufactured_fixture", "repetition": 0,
           "streams": {"train": 900001, "validation": 900002, "calibration": 900003, "test": 900004},
           "geometry_seeds": {"4": 910004, "12": 910012}, "full_prediction_audit": True}
    fit_models.make_episodes = manufactured
    score_models.make_episodes = manufactured
    started = time.perf_counter()
    development_result = fit_models.fit_development(job, cfg, output)
    calibration_result = score_models.evaluate(job, cfg, output, "calibration")
    test_result = score_models.evaluate(job, cfg, output, "test")
    development = read_json(output / "development.json")
    calibration = read_json(output / "calibration.json")
    assertions = 0
    with np.load(output / "coefficients.npz", allow_pickle=False) as coefficients:
        for row in development["evaluation"]:
            coefficient = coefficients[row["id"] + "__coefficient"]
            assert coefficient.shape == (row["n_features"], 3)
            assertions += 1
    for arm in (*cfg["policies"], "rbf", "affine"):
        rows = [row for row in development["evaluation"] if row["arm"] == arm]
        assert len({row["bank_id"] for row in rows}) == 1
        assert [row["n_features"] for row in rows] == [2, 6, 14, 30]
        assertions += 2
    tail = cfg["delta"] / cfg["campaign_total_tails"]
    for split in ("calibration", "test"):
        with np.load(output / (split + "_scores.npz"), allow_pickle=False) as scores, \
             np.load(output / (split + "_predictions.npz"), allow_pickle=False) as arrays:
            for row in development["evaluation"]:
                alias = row["alias"]
                target, prediction = arrays["targets"], arrays[row["id"]]
                close(np.mean((target - prediction)**2, axis=1), scores[alias + "__loss"], alias)
                assertions += 1
            if split == "calibration":
                for result in calibration["results"]:
                    alias = result["alias"]
                    for h, endpoint in enumerate(result["certificates"]):
                        upper = interval(scores[alias + "__loss"][:, h], 4.0, tail)
                        lower = interval(scores[alias + "__witness"][:, h], 2.0, tail)
                        close(upper["upper"], endpoint["mse_upper"], alias + " upper")
                        close(lower["lower"], endpoint["mse_lower"], alias + " lower")
                        assertions += 2
                for comparison in calibration["paired_improvements"]:
                    h = cfg["horizons"].index(comparison["horizon"])
                    values = scores[comparison["old"] + "__loss"][:, h] - scores[comparison["new"] + "__loss"][:, h]
                    checked = interval(values + 4.0, 8.0, tail)
                    close(checked["lower"] - 4.0, comparison["improvement_lower"], "paired lower")
                    assertions += 1
    rank_values = [r["effective_rank"] for bank in development["banks"].values()
                   if bank["kind"] == "kahm" for r in bank["dimension_policy"]["local_models"]]
    result = {"passed": True, "scope": "Manufactured independent uniform arrays; no production law or seed used",
              "assertions": assertions, "development": development_result,
              "calibration": calibration_result, "test": test_result,
              "local_KAHM_rank_records": len(rank_values), "actual_ranks": sorted(set(rank_values)),
              "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
