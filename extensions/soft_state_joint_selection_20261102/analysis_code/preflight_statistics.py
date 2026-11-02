#!/usr/bin/env python3
"""Artificial scalar-loss checks; never read or generate production targets."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def run(root=ROOT):
    root = Path(root)
    spec = importlib.util.spec_from_file_location("joint_analysis_under_test",
                                                root / "analysis_code/analyze_soft_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((root / "config.json").read_text())
    module.validate_config(cfg)
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
                    for arm, base in zip(module.ARMS, (g, p, k, r)):
                        multiplier = {"shared": 1., "independent": 1.02,
                                      "standard_shared": 1.04, "standard_independent": 1.}[view]
                        if view == "independent" and arm == "K":
                            multiplier = .8 if law != "linear_ar" else 1.2
                        value = base * multiplier
                        rows.append({"task_id": f"artificial_{rep}_{law_index}_{noise}",
                            "repetition": rep, "law": law, "noise_index": noise,
                            "forecast_key": f"{view}__{arm}", "mse": 2 * value, "smse": value})
    result = module.repeated_loss_tables(rows)
    primary = result["primary_contrasts.csv"]
    decisions = [r["criterion_established"] for r in primary]
    assert decisions == [True, True, True, False, True, False]
    checks += 1
    assert len(result["repetition_losses.csv"]) == 32 * 3 * 16
    assert len(result["primary_paired_repetitions.csv"]) == 32 * 6
    assert len(result["secondary_descriptive.csv"]) == 2 * (16 + 8 + 8)
    assert len(result["optimization_repetition_pairs.csv"]) == 32 * 2 * 2 * 4
    assert len(result["optimization_summary.csv"]) == 2 * 2 * 4
    checks += 5

    # Use independent scalar arithmetic and the full artificial input roster;
    # do not reuse analyzer-produced differences to derive the expected bounds.
    for claim, row in zip(module.CLAIMS, primary):
        cid, panel, av, a, bv, b, factor, _ = claim
        laws = ("linear_ar",) if panel == "linear" else ("nonlinear_ar", "vanderpol", "duffing")
        left, right, differences = [], [], []
        for rep in range(32):
            a_values = [x["smse"] for x in rows if x["repetition"] == rep
                        and x["law"] in laws and x["forecast_key"] == av + "__" + a]
            b_values = [x["smse"] for x in rows if x["repetition"] == rep
                        and x["law"] in laws and x["forecast_key"] == bv + "__" + b]
            aa, bb = math.fsum(a_values) / len(a_values), math.fsum(b_values) / len(b_values)
            left.append(aa)
            right.append(bb)
            differences.append(aa - factor * bb)
        mean = math.fsum(differences) / 32
        variance = math.fsum((x - mean) ** 2 for x in differences) / 31
        bound = mean + float(t.isf(.05 / 6, 31)) * math.sqrt(variance / 32)
        assert math.isclose(row["simultaneous_one_sided_upper"], bound, rel_tol=1e-11, abs_tol=1e-14)
        assert (row["n_repetitions"], row["df"], row["family_size"], row["family_alpha"]) == (32, 31, 6, .05)
        assert row["candidate_key"] == av + "__" + a and row["comparator_key"] == bv + "__" + b
        gain = 100 * (1 - math.fsum(left) / math.fsum(right))
        assert math.isclose(row["point_relative_gain_percent"], gain, rel_tol=1e-12, abs_tol=1e-12)
        checks += 4

    first = next(r["smse"] for r in result["repetition_losses.csv"]
                 if r["panel"] == "nonlinear" and r["forecast_key"] == "shared__G" and r["repetition"] == 0)
    expected = math.fsum(.2 + .01 * law + .001 * noise for law in (1, 2, 3) for noise in (0, 1)) / 6
    assert math.isclose(first, expected, rel_tol=0, abs_tol=1e-14)
    checks += 1

    nonfinite = copy.deepcopy(rows)
    nonfinite[0]["smse"] = float("nan")
    negative_raw = copy.deepcopy(rows)
    negative_raw[0]["mse"] = -1.
    for bad in (rows[:-1], rows + [rows[0]], nonfinite, negative_raw):
        try:
            module.repeated_loss_tables(bad)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid factorial losses were accepted")

    # Exactly representable per-repetition boundary: 97 - .97*100 == 0.
    boundary = copy.deepcopy(rows)
    for row in boundary:
        if row["law"] != "linear_ar" and row["forecast_key"] in ("shared__G", "shared__P"):
            row["smse"] = 100. if row["forecast_key"] == "shared__G" else 97.
            row["mse"] = 2 * row["smse"]
    exact = module.repeated_loss_tables(boundary)["primary_contrasts.csv"][0]
    assert exact["margin_difference_mean"] == 0 and exact["simultaneous_one_sided_upper"] == 0
    assert exact["criterion_established"] is False
    checks += 2

    # Perfect zero losses leave ratio gains undefined; they never establish
    # strict superiority or noninferiority from a zero upper margin bound.
    zeros = [{**row, "mse": 0., "smse": 0.} for row in rows]
    zero_primary = module.repeated_loss_tables(zeros)["primary_contrasts.csv"]
    assert all(x["point_relative_gain_percent"] is None and not x["criterion_established"] for x in zero_primary)
    checks += 1
    for field, value in (("master_seed", 2026110203), ("repetitions", 33),
                         ("input_scaling_order", ["minmax", "standard", "tanh_z_1"])):
        bad_config = copy.deepcopy(cfg)
        bad_config[field] = value
        try:
            module.validate_config(bad_config)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Changed fixed analysis specification was accepted")
    return {"status": "passed", "production_data_used": False, "checks": checks,
            "scope": "Artificial scalar losses: repetition weighting, cross-context pairing, six-claim bounds, signs, boundary, undefined ratios, malformed roster/config rejection",
            "critical_t": float(t.isf(.05 / 6, 31)), "first_fixture_decisions": decisions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
