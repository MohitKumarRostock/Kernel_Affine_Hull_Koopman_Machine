#!/usr/bin/env python3
"""Run the frozen positive-tolerance decision experiment.

Training and validation are completed and recorded before any calibration
stream is generated. Every prescribed repetition and sample size is retained.
The discrete atom indices and fitted coefficients reconstruct every score.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from certified_forecasting.certificates import mean_interval
from certified_forecasting.models import BlockBoundedRidge


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def support(config, regime):
    """Index = 8*S + 4*z_bit + 2*r_bit + n_bit, with sign = 2*bit-1."""
    index = np.arange(16, dtype=np.uint8)
    s = (index // 8).astype(float)
    z = (2 * ((index // 4) % 2).astype(int) - 1).astype(float)
    r = (2 * ((index // 2) % 2).astype(int) - 1).astype(float)
    noise = (2 * (index % 2).astype(int) - 1).astype(float)
    driver = z if regime == "repairable" else r
    y = config["signal_amplitude"] * s * driver + config["noise_amplitude"] * noise
    probability = np.where(s == 1, config["rare_probability"],
                           1 - config["rare_probability"]) / 8
    features = np.column_stack((np.ones(16), 1-s, s, (1+s*z)/2, (1-s*z)/2))
    partitions = {"state": s.astype(int),
                  "complete_history": (2*s+(z+1)/2).astype(int)}
    return {"index": index, "s": s, "z": z, "r": r, "noise": noise,
            "y": y, "p": probability, "features": features, "partitions": partitions}


def generate_indices(seed, n, probability):
    rng = np.random.default_rng(seed)
    s = (rng.random(n) < probability).astype(np.uint8)
    signs = rng.integers(0, 2, size=(3, n), dtype=np.uint8)
    return 8*s + 4*signs[0] + 2*signs[1] + signs[2]


def blocks(prefix):
    return (slice(0, 1), slice(1, 3), slice(3, 5))[:prefix+1]


def witness_matrix(atoms, prefix, kappa):
    f, y = atoms["features"], atoms["y"]
    distances = [np.linalg.norm(f[:, None, b]-f[None, :, b], axis=-1)
                 for b in blocks(prefix)]
    distance = np.maximum.reduce(distances)
    return 0.5 * np.maximum(np.abs(y[:, None]-y[None, :])-kappa*distance, 0)**2


def pair_values(indices, matrix):
    usable = len(indices) // 2 * 2
    return matrix[indices[:usable:2], indices[1:usable:2]]


def minimum_mass_weighted_value(lower, upper, values):
    """Solve the bounded-simplex linear program by the sorted greedy rule."""
    mass = lower.copy()
    remainder = 1-float(mass.sum())
    for index in np.argsort(values, kind="stable"):
        addition = min(float(upper[index]-mass[index]), max(0.0, remainder))
        mass[index] += addition
        remainder -= addition
    if abs(remainder) > 1e-9:
        raise ValueError("Empty or inaccurate stratum probability polytope")
    return float(mass @ values), mass


def stratified_interval(indices, labels, matrix, pair_bound, eta):
    n_strata = int(labels.max())+1
    n = len(indices)
    counts = np.bincount(labels[indices], minlength=n_strata)
    mass_radius = math.sqrt(math.log(4*n_strata/eta)/(2*n))
    mass_lower = np.maximum(0, counts/n-mass_radius)
    mass_upper = np.minimum(1, counts/n+mass_radius)
    lower, details = [], []
    for stratum in range(n_strata):
        chosen = indices[labels[indices] == stratum]
        values = pair_values(chosen, matrix)
        detail = {"stratum": stratum, "count": int(len(chosen)),
                  "n_pairs": int(len(values))}
        if len(values) < 2:
            detail.update(lower=0.0, insufficient_pairs=True)
        else:
            detail.update(mean_interval(values, pair_bound, eta/(2*n_strata)))
            detail["insufficient_pairs"] = False
        lower.append(detail["lower"])
        details.append(detail)
    endpoint, minimizer = minimum_mass_weighted_value(
        mass_lower, mass_upper, np.asarray(lower))
    return {"lower": endpoint, "eta": eta, "mass_radius": mass_radius,
            "counts": counts.tolist(), "mass_lower": mass_lower.tolist(),
            "mass_upper": mass_upper.tolist(), "minimizing_mass": minimizer.tolist(),
            "conditional_intervals": details}


def oracle_record(config, regime, atoms, matrices):
    result = {"regime": regime, "atoms": [], "prefixes": []}
    for i in range(16):
        result["atoms"].append({"index": i, "S": int(atoms["s"][i]),
            "Z": int(atoms["z"][i]), "R": int(atoms["r"][i]),
            "N": int(atoms["noise"][i]), "probability": float(atoms["p"][i]),
            "target": float(atoms["y"][i]), "features": atoms["features"][i].tolist()})
    for prefix, matrix in enumerate(matrices):
        record = {"prefix": prefix+1, "cost": config["retained_costs"][prefix],
                  "optimal_risk": config["oracle_risks"][regime][prefix],
                  "unconditional_witness": float(atoms["p"] @ matrix @ atoms["p"]),
                  "stratified_witnesses": {}}
        for name, labels in atoms["partitions"].items():
            conditional, weighted = [], 0.0
            for label in range(int(labels.max())+1):
                idx = np.flatnonzero(labels == label)
                probability = float(atoms["p"][idx].sum())
                normalized = atoms["p"][idx]/probability
                witness = float(normalized @ matrix[np.ix_(idx, idx)] @ normalized)
                weighted += probability*witness
                conditional.append({"stratum": label, "mass": probability,
                                    "witness": witness})
            record["stratified_witnesses"][name] = {"witness": weighted,
                                                    "strata": conditional}
        result["prefixes"].append(record)
    return result


def status(lower, upper, threshold):
    return "accepted" if upper <= threshold else "excluded" if lower > threshold else "unresolved"


def min_cost(costs, mask):
    eligible = [cost for cost, keep in zip(costs, mask) if keep]
    return min(eligible) if eligible else None  # JSON null denotes +infinity.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; retain it and choose a new directory.")
    args.output.mkdir(parents=True)
    config = json.loads(args.protocol.read_text())
    write_json(args.output / "protocol.json", config)
    source_paths = [Path(__file__).resolve(), args.protocol.resolve(),
                    ROOT / "src/certified_forecasting/models.py",
                    ROOT / "src/certified_forecasting/certificates.py"]
    manifest = {"frozen_at_utc": now(), "protocol_sha256": sha256(args.protocol),
                "calibration_generated": False,
                "python": platform.python_version(), "numpy": np.__version__,
                "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
                            for p in source_paths]}
    write_json(args.output / "precalibration_manifest.json", manifest)
    print("Protocol and source manifest frozen; fitting independent training samples.", flush=True)
    n_models = len(config["retained_costs"])
    n_looks = len(config["calibration_sizes"])
    n_partitions = len(config["partitions"])
    upper_tail = config["upper_budget"]/(n_models*n_looks)
    lower_tail = config["unconditional_lower_budget"]/(n_models*n_looks)
    eta = config["stratified_lower_budget"]/(n_models*n_looks*n_partitions)
    loss_bound = (config["target_bounds"][1]-config["target_bounds"][0])**2
    pair_bound = loss_bound/2
    threshold = config["rmse_tolerance"]**2
    atoms_all = {regime: support(config, regime) for regime in config["regimes"]}
    matrices_all = {regime: [witness_matrix(atoms, j, config["kappa"])
                            for j in range(n_models)]
                    for regime, atoms in atoms_all.items()}
    oracle = {regime: oracle_record(config, regime, atoms_all[regime], matrices_all[regime])
              for regime in config["regimes"]}
    write_json(args.output / "population_oracle.json", oracle)
    models, fit_rows, development_states = {}, [], {}
    for regime_index, regime in enumerate(config["regimes"]):
        atoms = atoms_all[regime]
        train_states = np.empty((config["n_repetitions"], config["n_train"]), dtype=np.uint8)
        validation_states = np.empty((config["n_repetitions"], config["n_validation"]), dtype=np.uint8)
        for repetition in range(config["n_repetitions"]):
            seed = config["seed_start"]+repetition
            base = 100000000*regime_index+100*seed
            train = generate_indices(base+11, config["n_train"], config["rare_probability"])
            validation = generate_indices(base+23, config["n_validation"], config["rare_probability"])
            train_states[repetition], validation_states[repetition] = train, validation
            fitted = []
            for prefix, cost in enumerate(config["retained_costs"]):
                model = BlockBoundedRidge(alpha=0, kappa=config["kappa"],
                    bounds=config["target_bounds"], block_slices=blocks(prefix),
                    tol=config["decoder_tolerance"], max_iter=config["decoder_max_iter"])
                model.fit(atoms["features"][train, :cost], atoms["y"][train])
                if model.norm_sum_ > config["kappa"]+1e-9:
                    raise ValueError("Infeasible fitted model")
                atom_predictions = model.predict(atoms["features"][:, :cost]).ravel()
                atom_losses = (atom_predictions-atoms["y"])**2
                record = {"regime": regime, "repetition": repetition, "seed": seed,
                    "training_seed": base+11, "validation_seed": base+23,
                    "calibration_seed": base+47, "prefix": prefix+1, "cost": cost,
                    "coef": model.coef_.ravel().tolist(),
                    "intercept": float(model.intercept_[0]),
                    "block_norms": model.block_norms_.tolist(), "norm_sum": model.norm_sum_,
                    "optimization_gap": model.optimization_gap_, "n_iter": model.n_iter_,
                    "converged": bool(model.converged_),
                    "training_mse": float(np.mean(atom_losses[train])),
                    "validation_mse": float(np.mean(atom_losses[validation])),
                    "population_mse": float(atoms["p"] @ atom_losses),
                    "atom_predictions": atom_predictions.tolist()}
                fit_rows.append(record)
                fitted.append(record)
            models[(regime, repetition)] = fitted
        development_states[regime+"_train"] = train_states
        development_states[regime+"_validation"] = validation_states
    np.savez_compressed(args.output / "development_states.npz", **development_states)
    write_json(args.output / "frozen_models.json", fit_rows)
    model_manifest = {"frozen_at_utc": now(), "calibration_generated": False,
        "models_sha256": sha256(args.output / "frozen_models.json"),
        "development_states_sha256": sha256(args.output / "development_states.npz"),
        "n_models": len(fit_rows), "nonconverged": sum(not row["converged"] for row in fit_rows)}
    write_json(args.output / "frozen_models_manifest.json", model_manifest)
    print(f"All {len(fit_rows)} fitted models frozen; generating calibration streams.", flush=True)
    endpoint_rows, decision_rows = [], []
    coverage = defaultdict(lambda: {"all_upper_valid": True, "all_unconditional_valid": True,
                                    "all_stratified_valid": True})
    with gzip.open(args.output / "interval_details.jsonl.gz", "wt") as details_file:
        for regime in config["regimes"]:
            atoms, matrices = atoms_all[regime], matrices_all[regime]
            calibration_states = np.empty((config["n_repetitions"], max(config["calibration_sizes"])),
                                          dtype=np.uint8)
            for repetition in range(config["n_repetitions"]):
                fitted = models[(regime, repetition)]
                states = generate_indices(fitted[0]["calibration_seed"],
                    max(config["calibration_sizes"]), config["rare_probability"])
                calibration_states[repetition] = states
                validation_cost = min_cost(config["retained_costs"],
                    [m["validation_mse"] <= threshold for m in fitted])
                for n in config["calibration_sizes"]:
                    indices = states[:n]
                    current = []
                    for prefix, cost in enumerate(config["retained_costs"]):
                        model, matrix = fitted[prefix], matrices[prefix]
                        atom_predictions = np.asarray(model["atom_predictions"])
                        losses = (atom_predictions[indices]-atoms["y"][indices])**2
                        upper = mean_interval(losses, loss_bound, upper_tail)
                        ordinary = mean_interval(pair_values(indices, matrix), pair_bound, lower_tail)
                        stratified = {name: stratified_interval(indices, labels, matrix, pair_bound, eta)
                                      for name, labels in atoms["partitions"].items()}
                        lo_strat = max(interval["lower"] for interval in stratified.values())
                        row = {"regime": regime, "repetition": repetition, "seed": model["seed"],
                            "n_calibration": n, "prefix": prefix+1, "cost": cost,
                            "optimal_mse": config["oracle_risks"][regime][prefix],
                            "population_mse": model["population_mse"],
                            "validation_mse": model["validation_mse"],
                            "empirical_mse": upper["mean"], "upper": upper["upper"],
                            "lower_unconditional": ordinary["lower"],
                            "lower_state": stratified["state"]["lower"],
                            "lower_complete_history": stratified["complete_history"]["lower"],
                            "lower_stratified": lo_strat,
                            "status_unconditional": status(ordinary["lower"], upper["upper"], threshold),
                            "status_stratified": status(lo_strat, upper["upper"], threshold)}
                        endpoint_rows.append(row)
                        current.append(row)
                        cov = coverage[(regime, repetition)]
                        cov["all_upper_valid"] &= upper["upper"]+1e-12 >= model["population_mse"]
                        cov["all_unconditional_valid"] &= ordinary["lower"] <= row["optimal_mse"]+1e-12
                        cov["all_stratified_valid"] &= lo_strat <= row["optimal_mse"]+1e-12
                        detail = {"regime": regime, "repetition": repetition, "n_calibration": n,
                                  "prefix": prefix+1, "upper": upper, "unconditional": ordinary,
                                  "stratified": stratified}
                        details_file.write(json.dumps(detail, allow_nan=False)+"\n")
                    c_up = min_cost(config["retained_costs"], [row["upper"] <= threshold for row in current])
                    for method in ("unconditional", "stratified"):
                        c_low = min_cost(config["retained_costs"],
                            [row["lower_"+method] <= threshold for row in current])
                        decision_rows.append({"regime": regime, "repetition": repetition,
                            "seed": fitted[0]["seed"], "n_calibration": n, "lower_method": method,
                            "validation_cost": validation_cost, "upper_cost": c_up,
                            "lower_cost": c_low,
                            "minimum_cost_certified": c_up is not None and c_low == c_up,
                            "infeasibility_certified": c_low is None,
                            "accepted_classes": sum(row["upper"] <= threshold for row in current),
                            "excluded_classes": sum(row["lower_"+method] > threshold for row in current),
                            "unresolved_classes": sum(row["status_"+method] == "unresolved" for row in current)})
            np.savez_compressed(args.output / (regime+"_calibration_states.npz"), states=calibration_states)
            print(f"Completed {regime}: all {config['n_repetitions']} repetitions and {n_looks} sizes.", flush=True)
    write_csv(args.output / "endpoints.csv", endpoint_rows)
    write_csv(args.output / "decisions.csv", decision_rows)
    coverage_rows = [dict(regime=key[0], repetition=key[1], **values,
                          all_valid=all(values.values())) for key, values in coverage.items()]
    write_csv(args.output / "coverage.csv", coverage_rows)
    summaries = []
    for regime in config["regimes"]:
        for n in config["calibration_sizes"]:
            for method in ("unconditional", "stratified"):
                rows = [row for row in decision_rows if row["regime"] == regime
                        and row["n_calibration"] == n and row["lower_method"] == method]
                summaries.append({"regime": regime, "n_calibration": n, "lower_method": method,
                    "n_repetitions": len(rows),
                    "validation_acceptances": sum(row["validation_cost"] is not None for row in rows),
                    "upper_acceptances": sum(row["upper_cost"] is not None for row in rows),
                    "minimum_cost_certificates": sum(row["minimum_cost_certified"] for row in rows),
                    "infeasibility_certificates": sum(row["infeasibility_certified"] for row in rows),
                    "excluded_classes": sum(row["excluded_classes"] for row in rows),
                    "unresolved_classes": sum(row["unresolved_classes"] for row in rows)})
    write_csv(args.output / "decision_summary.csv", summaries)
    output_manifest = [{"path": str(path.relative_to(args.output)), "sha256": sha256(path),
                        "bytes": path.stat().st_size}
                       for path in sorted(args.output.iterdir()) if path.is_file()]
    summary = {"completed_at_utc": now(), "status": "completed", "n_repetitions_per_regime":
        config["n_repetitions"], "n_models": len(fit_rows), "n_endpoints": len(endpoint_rows),
        "n_decision_rows": len(decision_rows), "n_nonconverged_fits": model_manifest["nonconverged"],
        "largest_optimization_gap": max(row["optimization_gap"] for row in fit_rows),
        "largest_coefficient_norm": max(row["norm_sum"] for row in fit_rows),
        "coverage_failures": [row for row in coverage_rows if not row["all_valid"]],
        "confidence_scope": config["confidence_scope"], "decision_summary": summaries,
        "raw_state_encoding": "uint8: 8*S+4*((Z+1)/2)+2*((R+1)/2)+(N+1)/2; rows are repetitions in seed order",
        "null_cost_meaning": "positive infinity (no eligible class or fitted predictor)",
        "files": output_manifest}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "n_models": len(fit_rows),
                      "n_endpoints": len(endpoint_rows), "coverage_failures": len(summary["coverage_failures"]),
                      "n_nonconverged_fits": model_manifest["nonconverged"],
                      "decision_summary": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
