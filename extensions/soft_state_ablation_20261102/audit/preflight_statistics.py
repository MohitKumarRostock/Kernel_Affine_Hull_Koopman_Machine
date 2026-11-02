#!/usr/bin/env python3
"""Deterministic artificial checks of the frozen statistical unit and signs."""
from __future__ import annotations
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def run():
    spec = importlib.util.spec_from_file_location("analysis_under_test", ROOT / "analysis_code/analyze_soft_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_config(json.loads((ROOT / "config.json").read_text()))
    checks = 1
    rows = []
    for rep in range(32):
        for law_index, law in enumerate(module.LAWS):
            for noise in (0, 1):
                g = .2 + .01 * law_index + .001 * noise + .0001 * rep
                p = .8 * g
                k = .93 * p if law != "linear_ar" else 1.1 * p
                r = .99 * p
                for view in module.VIEWS:
                    for arm, value in zip(module.ARMS, (g, p, k, r)):
                        value *= 1 if view == "shared" else 1.02
                        rows.append({"task_id": f"artificial_{rep}_{law_index}_{noise}",
                            "repetition": rep, "law": law, "noise_index": noise,
                            "forecast_key": f"{view}__{arm}", "mse": 2 * value, "smse": value})
    result = module.repeated_loss_tables(rows)
    primary = result["primary_contrasts.csv"]
    assert [r["criterion_established"] for r in primary] == [True, True, True, False]
    checks += 1
    assert len(result["repetition_losses.csv"]) == 32 * 3 * 8
    assert len(result["primary_paired_repetitions.csv"]) == 32 * 4
    checks += 2
    for row in primary:
        assert (row["n_repetitions"], row["df"], row["family_size"], row["family_alpha"], row["view"]) == (32, 31, 4, .05, "shared")
        pair = [r["margin_difference"] for r in result["primary_paired_repetitions.csv"] if r["contrast_id"] == row["contrast_id"]]
        mean = math.fsum(pair) / 32
        variance = math.fsum((x - mean) ** 2 for x in pair) / 31
        bound = mean + float(t.isf(.0125, 31)) * math.sqrt(variance / 32)
        assert math.isclose(row["simultaneous_one_sided_upper"], bound, rel_tol=1e-12, abs_tol=1e-14)
        checks += 2
    # The linear arm has a deliberately adverse enrichment contrast. Keep its
    # population separate and verify the nonlinear aggregation explicitly.
    first_rep_g = next(r["smse"] for r in result["repetition_losses.csv"]
                       if r["panel"] == "nonlinear" and r["forecast_key"] == "shared__G" and r["repetition"] == 0)
    expected_g = math.fsum(.2 + .01 * law + .001 * noise for law in (1, 2, 3) for noise in (0, 1)) / 6
    assert math.isclose(first_rep_g, expected_g, rel_tol=0, abs_tol=1e-14)
    checks += 1
    for bad in (rows[:-1], rows + [rows[0]]):
        try:
            module.repeated_loss_tables(bad)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Incomplete/duplicate factorial roster was accepted")
    # An exactly zero margin-adjusted difference must not establish a claim.
    tied = [dict(row) for row in rows]
    by_task = {(r["task_id"], r["forecast_key"]): r for r in tied}
    for row in tied:
        if row["forecast_key"] == "shared__P" and row["law"] != "linear_ar":
            row["smse"] = .97 * by_task[row["task_id"], "shared__G"]["smse"]
            row["mse"] = 2 * row["smse"]
    exact = module.repeated_loss_tables(tied)["primary_contrasts.csv"][0]
    # Aggregate-before-contrast arithmetic can leave machine-scale residuals;
    # the decision still must not become a spurious strict rejection.
    assert abs(exact["margin_difference_mean"]) < 1e-15
    assert exact["criterion_established"] is False
    checks += 2
    return {"status": "passed", "production_data_used": False, "checks": checks,
            "scope": "Artificial scalar losses: population units, weighting, contrast signs, family, strict margin and missing/duplicate rejection",
            "critical_t": float(t.isf(.0125, 31)), "first_fixture_decisions": [r["criterion_established"] for r in primary]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
