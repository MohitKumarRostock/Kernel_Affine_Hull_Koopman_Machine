#!/usr/bin/env python3
"""Reconstruct the decision study without importing its runner or certificates.

Every recorded endpoint is recomputed from retained atom indices. A vertex
enumeration independently checks the runner's greedy mass-interval program.
Finite-support enumeration supplies exact class and fitted-predictor risks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def close(actual, expected, name, tolerance=2e-11):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{name}: {actual} != {expected}")


def atom_table(config, regime):
    atoms = []
    for s, z, r, noise in itertools.product((0, 1), (-1, 1), (-1, 1), (-1, 1)):
        driver = z if regime == "repairable" else r
        atoms.append({"s": s, "z": z,
            "p": (config["rare_probability"] if s else 1-config["rare_probability"])/8,
            "y": config["signal_amplitude"]*s*driver+config["noise_amplitude"]*noise,
            "f": [1., 1.-s, float(s), (1+s*z)/2, (1-s*z)/2]})
    return atoms


def scores_and_oracles(atoms, cost, kappa):
    block_ends = [(0, 1)] + ([(1, 3)] if cost >= 3 else []) + ([(3, 5)] if cost >= 5 else [])
    score = np.empty((16, 16))
    distance = np.empty((16, 16))
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms):
            d = max(math.sqrt(sum((a["f"][k]-b["f"][k])**2 for k in range(start, stop)))
                    for start, stop in block_ends)
            distance[i, j] = d
            score[i, j] = max(abs(a["y"]-b["y"])-kappa*d, 0)**2/2
    keys = [tuple(a["f"][:cost]) for a in atoms]
    means = {key: sum(a["p"]*a["y"] for a, k in zip(atoms, keys) if k == key)/
                    sum(a["p"] for a, k in zip(atoms, keys) if k == key)
             for key in set(keys)}
    mean_vector = np.array([means[key] for key in keys])
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms):
            if abs(mean_vector[i]-mean_vector[j]) > kappa*distance[i, j]+1e-12:
                raise AssertionError("Conditional mean is not in the Lipschitz class")
    # A feasible affine realization shows equality also for the fitted model class.
    design = np.array([a["f"][:cost]+[1.] for a in atoms])
    coefficient, *_ = np.linalg.lstsq(design, mean_vector, rcond=None)
    norm = sum(np.linalg.norm(coefficient[start:stop]) for start, stop in block_ends)
    if norm > kappa+1e-10 or np.max(np.abs(design@coefficient-mean_vector)) > 1e-10:
        raise AssertionError("Conditional mean lacks the claimed feasible affine realization")
    risk = sum(a["p"]*(a["y"]-m)**2 for a, m in zip(atoms, mean_vector))
    ordinary = sum(a["p"]*b["p"]*score[i, j] for i, a in enumerate(atoms)
                   for j, b in enumerate(atoms))
    conditional = {}
    for name, labels in (("state", [a["s"] for a in atoms]),
                         ("complete_history", [2*a["s"]+(a["z"]+1)//2 for a in atoms])):
        value = 0.0
        for stratum in sorted(set(labels)):
            indices = [i for i, label in enumerate(labels) if label == stratum]
            mass = sum(atoms[i]["p"] for i in indices)
            value += sum(atoms[i]["p"]*atoms[j]["p"]*score[i, j]
                         for i in indices for j in indices)/mass
        conditional[name] = value
    return score, risk, ordinary, conditional


def bernstein(values, bound, delta):
    n = len(values)
    mean = float(np.sum(values)/n)
    variance = float(np.sum((values-mean)**2)/(n-1))
    radius = math.sqrt(2*variance*math.log(2/delta)/n)+7*bound*math.log(2/delta)/(3*(n-1))
    return {"mean": mean, "variance": variance, "radius": radius,
            "lower": max(0., mean-radius), "upper": min(bound, mean+radius)}


def check_interval(recorded, values, bound, delta, name):
    expected = bernstein(values, bound, delta)
    for field, number in expected.items():
        close(recorded[field], number, name+"/"+field)
    if recorded["n_independent"] != len(values):
        raise AssertionError("Incorrect independent score count")
    close(recorded["bound"], bound, name+"/bound")
    close(recorded["delta_per_tail"], delta, name+"/tail")
    return expected


def vertex_minimum(lower, upper, objectives):
    """Enumerate all vertices: all but at most one mass lies on its box bound."""
    best = math.inf
    for free in range(len(lower)):
        fixed = [j for j in range(len(lower)) if j != free]
        for choice in itertools.product((0, 1), repeat=len(fixed)):
            p = lower.copy()
            for j, select_upper in zip(fixed, choice):
                p[j] = upper[j] if select_upper else lower[j]
            p[free] = 1-sum(p[j] for j in fixed)
            if lower[free]-1e-12 <= p[free] <= upper[free]+1e-12:
                best = min(best, float(np.dot(p, objectives)))
    if not math.isfinite(best):
        raise AssertionError("No probability-polytope vertex")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.results
    config = json.loads((folder/"protocol.json").read_text())
    models = json.loads((folder/"frozen_models.json").read_text())
    recorded_oracle = json.loads((folder/"population_oracle.json").read_text())
    summary = json.loads((folder/"summary.json").read_text())
    for entry in summary["files"]:
        actual = hashlib.sha256((folder/entry["path"]).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise AssertionError("Retained output hash mismatch: "+entry["path"])
    freeze = json.loads((folder/"precalibration_manifest.json").read_text())
    model_freeze = json.loads((folder/"frozen_models_manifest.json").read_text())
    if not (datetime.fromisoformat(freeze["frozen_at_utc"]) <=
            datetime.fromisoformat(model_freeze["frozen_at_utc"]) <=
            datetime.fromisoformat(summary["completed_at_utc"])):
        raise AssertionError("Invalid freeze chronology")
    close(config["upper_budget"]+config["unconditional_lower_budget"]+
          config["stratified_lower_budget"], config["confidence_delta"], "total failure budget")
    development = np.load(folder/"development_states.npz", allow_pickle=False)
    calibration = {name: np.load(folder/(name+"_calibration_states.npz"), allow_pickle=False)["states"]
                   for name in config["regimes"]}
    atoms = {name: atom_table(config, name) for name in config["regimes"]}
    lookup = {}
    exact_risks, exact_witnesses = {}, {}
    for regime in config["regimes"]:
        for prefix, cost in enumerate(config["retained_costs"], start=1):
            matrix, risk, ordinary, conditional = scores_and_oracles(atoms[regime], cost, config["kappa"])
            lookup[(regime, prefix)] = matrix
            exact_risks[(regime, prefix)] = risk
            exact_witnesses[(regime, prefix)] = (ordinary, conditional)
            rec = recorded_oracle[regime]["prefixes"][prefix-1]
            close(rec["optimal_risk"], risk, "oracle risk")
            close(rec["unconditional_witness"], ordinary, "oracle ordinary witness")
            for name, value in conditional.items():
                close(rec["stratified_witnesses"][name]["witness"], value, "oracle stratified witness")
    model_map, stream_seeds = {}, set()
    largest_gap = 0.0
    for model in models:
        regime, repetition, prefix = model["regime"], model["repetition"], model["prefix"]
        model_map[(regime, repetition, prefix)] = model
        cost = model["cost"]
        f = np.array([a["f"][:cost] for a in atoms[regime]])
        y = np.array([a["y"] for a in atoms[regime]])
        weights = np.array(model["coef"])
        predicted = np.clip(f@weights+model["intercept"], *config["target_bounds"])
        for number, expected in zip(model["atom_predictions"], predicted):
            close(number, expected, "frozen prediction")
        block_weights = [weights[:1]] + ([weights[1:3]] if cost >= 3 else []) + ([weights[3:5]] if cost >= 5 else [])
        norm = sum(float(np.linalg.norm(w)) for w in block_weights)
        close(model["norm_sum"], norm, "coefficient norm")
        if norm > config["kappa"]+1e-10:
            raise AssertionError("Infeasible model")
        loss = (predicted-y)**2
        pop_risk = sum(a["p"]*v for a, v in zip(atoms[regime], loss))
        close(model["population_mse"], pop_risk, "fitted population risk")
        if pop_risk+1e-10 < exact_risks[(regime, prefix)]:
            raise AssertionError("Fitted risk smaller than asserted class optimum")
        for phase in ("train", "validation"):
            indices = development[regime+"_"+phase][repetition]
            close(model["training_mse" if phase == "train" else "validation_mse"],
                  np.mean(loss[indices]), "development risk")
        largest_gap = max(largest_gap, model["optimization_gap"])
        if prefix == 1:
            for phase, seed_field in (("train", "training_seed"), ("validation", "validation_seed"),
                                      ("calibration", "calibration_seed")):
                seed = model[seed_field]
                if seed in stream_seeds:
                    raise AssertionError("Reused stream seed")
                stream_seeds.add(seed)
                raw = (calibration[regime][repetition] if phase == "calibration"
                       else development[regime+"_"+phase][repetition])
                rng = np.random.default_rng(seed)
                s = (rng.random(len(raw)) < config["rare_probability"]).astype(np.uint8)
                z, r, noise = rng.integers(0, 2, size=(3, len(raw)), dtype=np.uint8)
                if not np.array_equal(raw, 8*s+4*z+2*r+noise):
                    raise AssertionError("Saved raw states differ from declared independent stream")
    n_looks, n_prefixes = len(config["calibration_sizes"]), len(config["retained_costs"])
    d_upper = config["upper_budget"]/(n_looks*n_prefixes)
    d_lower = config["unconditional_lower_budget"]/(n_looks*n_prefixes)
    eta = config["stratified_lower_budget"]/(n_looks*n_prefixes*len(config["partitions"]))
    bound = (config["target_bounds"][1]-config["target_bounds"][0])**2
    endpoint_map = {}
    with (folder/"endpoints.csv").open() as f:
        for row in csv.DictReader(f):
            endpoint_map[(row["regime"], int(row["repetition"]), int(row["n_calibration"]),
                          int(row["prefix"]))] = row
    n_verified, n_linear_programs = 0, 0
    failed_events = set()
    with gzip.open(folder/"interval_details.jsonl.gz", "rt") as f:
        for line in f:
            detail = json.loads(line)
            regime, repetition = detail["regime"], detail["repetition"]
            n, prefix = detail["n_calibration"], detail["prefix"]
            key = (regime, repetition, n, prefix)
            row = endpoint_map[key]
            model = model_map[(regime, repetition, prefix)]
            raw = calibration[regime][repetition, :n]
            y = np.array([a["y"] for a in atoms[regime]])
            predictions = np.array(model["atom_predictions"])
            matrix = lookup[(regime, prefix)]
            upper = check_interval(detail["upper"], (predictions[raw]-y[raw])**2,
                                   bound, d_upper, "upper")
            ordinary = check_interval(detail["unconditional"], matrix[raw[0::2], raw[1::2]],
                                      bound/2, d_lower, "ordinary lower")
            close(row["upper"], upper["upper"], "endpoint CSV upper")
            close(row["lower_unconditional"], ordinary["lower"], "endpoint CSV lower")
            conditional_endpoints = []
            for name, strat in detail["stratified"].items():
                labels = np.array([a["s"] if name == "state" else
                                   2*a["s"]+(a["z"]+1)//2 for a in atoms[regime]])
                n_strata = int(labels.max())+1
                counts = np.array([(labels[raw] == s).sum() for s in range(n_strata)])
                if counts.tolist() != strat["counts"]:
                    raise AssertionError("Incorrect stratum count")
                rad = math.sqrt(math.log(4*n_strata/eta)/(2*n))
                p_lo, p_hi = np.maximum(0, counts/n-rad), np.minimum(1, counts/n+rad)
                close(rad, strat["mass_radius"], "mass radius")
                for recorded, expected in ((strat["mass_lower"], p_lo), (strat["mass_upper"], p_hi)):
                    for number, value in zip(recorded, expected):
                        close(number, value, "mass interval")
                lo = []
                for s, record in enumerate(strat["conditional_intervals"]):
                    items = raw[labels[raw] == s]
                    pairs = len(items)//2
                    if pairs != record["n_pairs"] or len(items) != record["count"]:
                        raise AssertionError("Incorrect conditional pair count")
                    if pairs < 2:
                        lo.append(0.)
                        close(record["lower"], 0., "empty conditional lower")
                    else:
                        values = matrix[items[:2*pairs:2], items[1:2*pairs:2]]
                        interval = check_interval(record, values, bound/2, eta/(2*n_strata),
                                                  "conditional lower")
                        lo.append(interval["lower"])
                expected = vertex_minimum(p_lo, p_hi, np.asarray(lo))
                close(strat["lower"], expected, "mass LP optimum")
                mass = np.array(strat["minimizing_mass"])
                close(mass.sum(), 1., "minimizing probability sum")
                if np.any(mass < p_lo-1e-10) or np.any(mass > p_hi+1e-10):
                    raise AssertionError("Infeasible minimizing masses")
                close(mass@lo, expected, "greedy solution objective")
                close(row["lower_"+name], expected, "conditional endpoint CSV")
                conditional_endpoints.append(expected)
                n_linear_programs += 1
            stratified = max(conditional_endpoints)
            close(row["lower_stratified"], stratified, "combined stratified endpoint")
            if (upper["upper"]+1e-10 < model["population_mse"] or
                    max(ordinary["lower"], stratified) > exact_risks[(regime, prefix)]+1e-10):
                failed_events.add((regime, repetition))
            n_verified += 1
    if n_verified != len(endpoint_map) or n_verified != 2*config["n_repetitions"]*n_looks*n_prefixes:
        raise AssertionError("Incomplete endpoint family")
    threshold = config["rmse_tolerance"]**2
    summary_counts = Counter()
    with (folder/"decisions.csv").open() as f:
        for decision in csv.DictReader(f):
            regime, repetition, n = decision["regime"], int(decision["repetition"]), int(decision["n_calibration"])
            method = decision["lower_method"]
            rows = [endpoint_map[(regime, repetition, n, prefix)] for prefix in range(1, n_prefixes+1)]
            accepted = [int(r["cost"]) for r in rows if float(r["upper"]) <= threshold]
            nonexcluded = [int(r["cost"]) for r in rows if float(r["lower_"+method]) <= threshold]
            validation = [int(r["cost"]) for r in rows if float(r["validation_mse"]) <= threshold]
            up, lo, val = min(accepted, default=None), min(nonexcluded, default=None), min(validation, default=None)
            for field, value in (("upper_cost", up), ("lower_cost", lo), ("validation_cost", val)):
                actual = None if decision[field] == "" else int(decision[field])
                if actual != value:
                    raise AssertionError("Incorrect selected or excluded cost")
            optimum = up is not None and up == lo
            infeasible = lo is None
            if (decision["minimum_cost_certified"] == "True") != optimum or (
                    decision["infeasibility_certified"] == "True") != infeasible:
                raise AssertionError("Incorrect final class decision")
            for field, value in (("minimum_cost_certificates", optimum), ("infeasibility_certificates", infeasible),
                                 ("upper_acceptances", up is not None), ("validation_acceptances", val is not None)):
                summary_counts[(regime, n, method, field)] += int(value)
    for row in summary["decision_summary"]:
        for field in ("minimum_cost_certificates", "infeasibility_certificates", "upper_acceptances", "validation_acceptances"):
            if row[field] != summary_counts[(row["regime"], row["n_calibration"], row["lower_method"], field)]:
                raise AssertionError("Incorrect aggregate decision count")
    if len(failed_events) != len(summary["coverage_failures"]):
        raise AssertionError("Incorrect empirical simultaneous coverage count")
    report = {"status": "passed", "verifier": "independent score formulas, finite-support enumeration, and LP vertex enumeration; runner not imported",
              "models_verified": len(models), "endpoint_records_verified": n_verified,
              "mass_linear_programs_verified": n_linear_programs, "unique_data_streams": len(stream_seeds),
              "raw_states_reconstructed_from_seeds": True, "retained_file_hashes_verified": True,
              "oracle_class_risks_and_witnesses_verified": True, "all_fitted_norms_feasible": True,
              "largest_recorded_optimization_gap": largest_gap,
              "simultaneous_coverage_failures": len(failed_events),
              "confidence_scope": config["confidence_scope"]}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
