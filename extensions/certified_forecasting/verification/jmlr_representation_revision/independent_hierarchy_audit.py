#!/usr/bin/env python3
"""Independent numerical reconstruction of the learned-KAHM hierarchy study.

No project or study module is imported. KAHM features are evaluated from the
portable numerical banks, and signed-edge QP duals are evaluated directly in
extended precision. Confidence endpoints and decisions are reconstructed
separately. Numerical guards are not formal interval arithmetic.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from scipy.optimize import linprog


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def read_json_lines(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__array__" in value:
        result = np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])
        require(result.dtype.kind in "fiub", "Non-numerical frozen array")
        return result
    if "__kahm_block__" in value:
        return decode(value["__kahm_block__"])
    if "__slice__" in value:
        return slice(*value["__slice__"])
    return {key: decode(item) for key, item in value.items()}


class Checks:
    def __init__(self):
        self.count = 0
        self.maximum = {}

    def close(self, actual, recorded, category, atol=5e-9, rtol=1e-10):
        actual, recorded = np.asarray(actual), np.asarray(recorded)
        require(actual.shape == recorded.shape,
                f"{category}: shapes {actual.shape} and {recorded.shape}")
        difference = float(np.max(np.abs(actual-recorded), initial=0.))
        self.maximum[category] = max(self.maximum.get(category, 0.), difference)
        self.count += actual.size
        require(np.allclose(actual, recorded, atol=atol, rtol=rtol),
                f"{category}: maximum absolute discrepancy {difference}")


def draw(seed, count, regime):
    """Reconstruct the declared independent episode law, with no study import."""
    require(regime in {"repairable", "irreducible"}, "Unknown regime")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1., 1.]), (count, 2))
    x = signs * (.9 + rng.uniform(-.1, .1, (count, 2)))
    if regime == "repairable":
        source = x
    else:
        source = rng.choice(np.array([-1., 1.]), (count, 2))
        source *= .9 + rng.uniform(-.1, .1, (count, 2))
    y = .8 * np.prod(source, axis=1) + rng.uniform(-.05, .05, count)
    return x, y


def features(bank, x):
    """Evaluate the folding formula using direct coordinate differences."""
    require(bank["input_scaling_"] == "minmax", "Unexpected frozen scaler")
    span = np.where(bank["constant_features_"], 1., bank["input_range_"])
    z = 2*(x-bank["input_min_"])/span-1
    z[:, bank["constant_features_"]] = 0
    z = z.T
    blocks = []
    for block in bank["blocks_"]:
        distances = []
        for autoencoders in block["model_"]["classifier"]:
            individual = []
            for ae in autoencoders:
                output = np.empty_like(z)
                reference = (ae["sqrt_weights_matrix"] @ ae["y_data_n_subspace"]).T
                for start in range(0, z.shape[1], 1000):
                    stop = min(z.shape[1], start+1000)
                    projected = ae["PC"].T @ z[:, start:stop]
                    query = (ae["sqrt_weights_matrix"] @ projected).T
                    differences = query[:, None, :] - reference[None, :, :]
                    squared = np.sum(differences**2, axis=2)
                    kernel = np.exp(-squared/(2*projected.shape[0]))
                    weights = kernel @ ae["B"]
                    normalizer = weights.sum(axis=1, keepdims=True)
                    normalizer[normalizer == 0] = 1.
                    output[:, start:stop] = ae["y_data"] @ (weights/normalizer).T
                radial = 1-np.exp(-np.linalg.norm(output-z, axis=0))
                denominator = np.linalg.norm(output, axis=0)*np.linalg.norm(z, axis=0)
                cosine = np.divide(np.sum(output*z, axis=0), denominator,
                                   out=np.zeros(z.shape[1]), where=denominator != 0)
                angular = np.arccos(np.clip(cosine, -1, 1))/np.pi
                individual.append(np.hypot(radial, angular)/np.sqrt(2))
            distances.append(np.min(individual, axis=0))
        score = np.array(distances).T
        log_weight = block["omega"]*np.log(1-np.clip(score, 0, 1)+block["tau"])
        weight = np.exp(log_weight-log_weight.max(axis=1, keepdims=True))
        blocks.append(weight/weight.sum(axis=1, keepdims=True))
    return np.column_stack(blocks)


def block_distances(x, slices):
    x = np.asarray(x, dtype=np.longdouble)
    difference = x[:, None, :] - x[None, :, :]
    return np.maximum.reduce([
        np.sqrt(np.sum(difference[..., block]**2, axis=-1)) for block in slices])


def bernoulli_kl(q, p):
    if q == 0:
        return -math.log1p(-p)
    if q == 1:
        return -math.log(p)
    if p <= 0 or p >= 1:
        return math.inf
    return q*math.log(q/p) + (1-q)*(math.log1p(-q)-math.log1p(-p))


def lower_kl(mean, count, bound, alpha):
    require(count > 0 and bound > 0 and 0 < alpha < 1, "Invalid KL arguments")
    q = min(1., max(0., mean/bound))
    if q == 0:
        return 0.
    target = math.log(1/alpha)/count
    if q == 1:
        return bound*math.exp(-target)
    left, right = 0., q
    for _ in range(100):
        middle = (left+right)/2
        if bernoulli_kl(q, middle) > target:
            left = middle
        else:
            right = middle
    return bound*(left+right)/2


def empirical_bernstein(values, bound, alpha):
    values = np.asarray(values, dtype=float)
    require(len(values) >= 2 and bound > 0 and 0 < alpha < 1, "Invalid EB arguments")
    mean = float(np.mean(values))
    radius = math.sqrt(2*float(np.var(values, ddof=1))*math.log(2/alpha)/len(values))
    radius += 7*bound*math.log(2/alpha)/(3*(len(values)-1))
    return max(0., mean-radius), min(bound, mean+radius)


def pair_scores(x, y, slices, kappa):
    end = len(y)//2*2
    delta = np.asarray(x[:end:2], dtype=np.longdouble)-np.asarray(x[1:end:2], dtype=np.longdouble)
    distance = np.maximum.reduce([np.sqrt(np.sum(delta[:, block]**2, axis=1)) for block in slices])
    target = np.abs(np.asarray(y[:end:2], dtype=np.longdouble)-np.asarray(y[1:end:2], dtype=np.longdouble))
    return np.asarray(np.maximum(target-kappa*distance, 0)**2/2, dtype=float)


def score_bound(size, diameter):
    require(size >= 2, "A block needs two observations")
    return diameter**2*(size*size//4)/(size*(size-1))


def signed_dual_value(x, y, slices, kappa, edge_dual):
    """Evaluate weak duality without running or importing an optimizer."""
    y = np.asarray(y, dtype=np.longdouble)
    edge_dual = np.asarray(edge_dual, dtype=np.longdouble)
    left, right = np.triu_indices(len(y), 1)
    require(edge_dual.shape == left.shape and np.isfinite(edge_dual).all(),
            "Invalid signed-edge dual")
    cap = np.longdouble(kappa)*block_distances(x, slices)[left, right]
    divergence = np.zeros(len(y), dtype=np.longdouble)
    np.add.at(divergence, left, edge_dual)
    np.add.at(divergence, right, -edge_dual)
    linear = np.sum(edge_dual*(y[left]-y[right]))
    penalty = np.sum(cap*np.abs(edge_dual))
    quadratic = np.sum(divergence**2)/4
    return float(linear-penalty-quadratic), float(linear), float(penalty), float(quadratic)


def audit_qp(x, y, slices, kappa, record, checks, diameter):
    """Validate an archived bracket with independent weak-duality arithmetic."""
    size = len(y)
    require(int(record["block_size"]) == size, "QP block size differs")
    raw, linear, penalty, quadratic = signed_dual_value(
        x, y, slices, kappa, record["signed_dual"])
    yld = np.asarray(y, dtype=np.longdouble)
    fitted = np.asarray(record["primal"], dtype=np.longdouble)
    require(fitted.shape == yld.shape and np.isfinite(fitted).all(), "Invalid QP primal")
    distances = block_distances(x, slices)
    violation = float(max(np.longdouble(0), np.max(
        np.abs(fitted[:, None]-fitted[None, :])-np.longdouble(kappa)*distances)))
    require(violation <= 5e-10, f"Primal violates Lipschitz constraint: {violation}")
    primal_sse = float(np.sum((yld-fitted)**2))
    guard = 1e-11*(1+abs(linear)+abs(penalty)+abs(quadratic))
    expected_low = max(0., raw-guard)
    expected_high = primal_sse+1e-11*(1+abs(primal_sse))
    checks.close(raw, record["dual_raw"], "QP signed dual", atol=5e-10)
    checks.close(guard, record["dual_guard"], "QP dual guard", atol=1e-17)
    checks.close(expected_low, record["sse_lower"], "QP guarded lower SSE", atol=5e-10)
    checks.close(expected_high, record["sse_upper"], "QP guarded upper SSE", atol=5e-10)
    checks.close(violation, record["feasibility_residual"], "QP primal residual", atol=5e-12)
    lower, upper = float(record["lower"]), float(record["upper"])
    checks.close([lower, upper, upper-lower],
                 [record["sse_lower"]/(size-1), record["sse_upper"]/(size-1), record["gap"]],
                 "QP normalization and gap", atol=5e-12)
    require(0 <= lower <= upper+5e-10, "Invalid QP bracket ordering")
    require(lower <= score_bound(size, diameter)+5e-10, "Lower score exceeds sharp range")
    require(lower <= max(0., raw)/(size-1)+5e-10, "Lower score exceeds feasible dual")
    if size == 2:
        exact = pair_scores(x, y, slices, kappa)[0]
        require(lower <= exact+5e-10 <= upper+1e-9, "Size-two bracket misses exact pair score")
    if np.max(distances) == 0 or kappa == 0:
        exact = float(np.sum((yld-yld.mean())**2))/(size-1)
        require(lower <= exact+5e-10 <= upper+1e-9, "Collapsed-feature bracket misses variance")
    return {"gap": upper-lower, "violation": violation,
            "solver_converged": bool(record["converged"]), "lower": lower}


def audit(results, output):
    require(not output.exists(), "Refusing to overwrite an existing audit")
    root = Path(__file__).resolve().parents[2]
    repository = root.parents[1]
    config = read_json(results/"protocol.json")
    freeze = read_json(results/"protocol_freeze.json")
    started = read_json(results/"calibration_started.json")
    summary = read_json(results/"run_summary.json")
    checks = Checks()
    require(freeze["calibration_generated"] is False, "Freeze reports prior calibration")
    require(started["calibration_generated_before_this_point"] is False,
            "Start record reports earlier calibration")
    require(freeze["frozen_at_utc"] < started["started_at_utc"] < summary["completed_at_utc"],
            "Recorded freeze/calibration order differs")
    require(started["protocol_freeze_sha256"] == digest(results/"protocol_freeze.json"),
            "Calibration start did not bind this freeze")
    for field, name in [("protocol_sha256", "protocol.json"),
                        ("source_snapshot_sha256", "source_snapshot.tar.gz"),
                        ("source_manifest_sha256", "source_manifest.json")]:
        require(freeze[field] == digest(results/name), "Frozen file hash differs: "+name)
    for name, sha in summary["files_sha256"].items():
        require(digest(results/name) == sha, "Completed evidence changed: "+name)
    manifest = read_json(results/"source_manifest.json")
    source_current_mismatches = []
    development_log_references = []
    with tarfile.open(results/"source_snapshot.tar.gz", "r:gz") as archive:
        for source in manifest["files"]:
            content = archive.extractfile(source["path"]).read()
            require(len(content) == source["bytes"], "Source archive byte count differs")
            require(hashlib.sha256(content).hexdigest() == source["sha256"], "Source archive hash differs")
            if source["path"].endswith(".py"):
                parsed = ast.parse(content)
                for node in ast.walk(parsed):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if any(isinstance(value, ast.Constant) and isinstance(value.value, str)
                               and value.value == "development_benchmark.json" for value in ast.walk(node)):
                            development_log_references.append({"path": source["path"], "function": node.name})
            if not (repository/source["path"]).exists() or digest(repository/source["path"]) != source["sha256"]:
                source_current_mismatches.append(source["path"])

    models_path = root/freeze["models_path"]
    require(digest(models_path) == freeze["models_sha256"], "Frozen-bank bytes differ")
    development = models_path.parent/"development_freeze.json"
    require(digest(development) == freeze["archived_development_freeze_sha256"],
            "Archived development freeze differs")
    require(read_json(development)["frozen_models_sha256"] == freeze["models_sha256"],
            "Banks differ from original development freeze")
    models = read_gzip(models_path)
    expected_banks = [(regime, repetition) for regime in config["regimes"]
                      for repetition in range(config["repetitions_per_regime"])]
    require([(row["regime"], row["repetition"]) for row in models] == expected_banks,
            "Study did not retain all declared banks in order")
    require(config["regimes"] == ["repairable", "irreducible"] and len(models) == 40,
            "Unexpected frozen bank family")
    require(config["prefixes"] == [1, 2] and config["retained_costs"] == [2, 6],
            "Unexpected retained family")
    require(config["calibration_sizes"] == [256, 1024, 4096]
            and config["block_sizes"] == [2, 4, 8, 16]
            and config["full_erm_sizes"] == [256], "Declared method grid changed")
    require(config["target_bounds"] == [-.85, .85]
            and config["kappa"] == 2 and config["mse_tolerance"] == .1,
            "Risk specification differs")
    expected_counts = {"upper": 40*2*3, "block": 40*2*3*4,
                       "stratified": 40*2*3*2, "full_erm": 40*2}
    expected_counts["total"] = sum(expected_counts.values())
    require(config["endpoint_counts"] == expected_counts, "Invalid confidence allocation counts")
    eta = config["confidence_delta"]/expected_counts["total"]
    checks.close(eta*expected_counts["total"], .05, "Total confidence allocation", atol=1e-15)
    checks.close(eta, summary["eta_per_endpoint"], "Per-endpoint allocation", atol=1e-18)
    maximum_n = max(config["calibration_sizes"])
    diameter = config["target_bounds"][1]-config["target_bounds"][0]
    loss_bound = diameter**2
    old_seeds = {seed for row in models for seed in row["data_seeds"].values()}
    development_seeds = {202610291701, 202610291702, 202610291801, 202610291802}
    samples, selected_norms, fresh_seeds = {}, [], set()
    model_lookup = {(row["regime"], row["repetition"]): row for row in models}
    for item in read_json_lines(results/"samples.jsonl.gz"):
        key = item["regime"], item["repetition"]
        require(key in model_lookup and key not in samples, "Unexpected or duplicate sample bank")
        regime, repetition = key
        record = model_lookup[key]
        seed = config["data_seed_base"]+100*config["regimes"].index(regime)+repetition
        require(seed == item["seed"] and seed not in old_seeds | development_seeds | fresh_seeds,
                "Calibration seed reused or changed")
        fresh_seeds.add(seed)
        x, y = draw(seed, maximum_n, regime)
        f = np.asarray(item["features"], dtype=float)
        predictions = np.asarray(item["predictions"], dtype=float)
        for name, actual in [("x", x), ("y", y)]:
            checks.close(actual, item[name], "Episode law replay", atol=0, rtol=0)
            require(array_digest(actual) == item["hashes"][name], "Episode hash differs")
        for name, actual in [("features", f), ("predictions", predictions)]:
            require(array_digest(actual) == item["hashes"][name], "Stored numerical array hash differs")
        require(np.all(y >= -.85) and np.all(y <= .85), "Target bound violated")
        bank = decode(record["bank"])
        require(bank["block_slices_"] == [slice(0, 2), slice(2, 6)], "Unexpected bank blocks")
        independently_evaluated = features(bank, x)
        checks.close(independently_evaluated, f, "Independent KAHM feature calculation", atol=2e-10)
        require(np.all(f >= 0) and np.all(f <= 1), "Soft coordinates outside unit interval")
        checks.close(f[:, :2].sum(1), np.ones(maximum_n), "First simplex block")
        checks.close(f[:, 2:].sum(1), np.ones(maximum_n), "Second simplex block")
        for position, model in enumerate(record["models"]):
            selected = model["selected"]
            width = config["retained_costs"][position]
            require(selected["cost"] == width, "Selected head cost differs")
            coefficients = np.asarray(selected["coefficients"], dtype=float)
            norm = sum(np.linalg.norm(coefficients[block]) for block in bank["block_slices_"][:position+1])
            require(norm <= config["kappa"]+1e-12, "Selected head exceeds common sensitivity budget")
            selected_norms.append(float(norm))
            predicted = np.clip(f[:, :width]@coefficients+selected["intercept"], -.85, .85)
            checks.close(predicted, predictions[:, position], "Selected predictor replay", atol=2e-12)
        labels = {}
        for name, block in zip(config["partitions"], bank["block_slices_"]):
            labels[name] = np.argmax(f[:, block], axis=1)
            checks.close(labels[name], item["labels"][name], "Frozen partition labels", atol=0, rtol=0)
        samples[key] = {"x": x, "y": y, "features": f, "predictions": predictions,
                        "labels": labels, "slices": bank["block_slices_"]}
    require(set(samples) == set(expected_banks), "Missing calibration bank")
    print(f"Reconstructed all {len(samples)} banks and fresh episode streams", flush=True)

    score_sets, qp_diagnostics = {}, {}
    qp_count = 0
    unconverged_count = 0
    fallback_batches = 0
    max_gap, max_violation = 0., 0.
    for row in read_json_lines(results/"qp_certificates.jsonl.gz"):
        key = row["regime"], row["repetition"]
        sample = samples[key]
        prefix = int(row["prefix"])
        width = config["retained_costs"][prefix-1]
        require(row["cost"] == width and row["start"] == 0, "QP family or observation order differs")
        checks.close(row["eta"], eta, "QP allocated failure probability", atol=1e-18)
        size = int(row["block_size"])
        if row["kind"] == "block":
            require(row["n"] == maximum_n and size in [4, 8, 16], "Unexpected hierarchy QP")
            count = maximum_n//size
            method = f"block_{size}"
        else:
            require(row["kind"] == "full_erm" and row["n"] == size == 256,
                    "Unexpected empirical-minimum QP")
            count, method = 1, "full_erm"
        index = (*key, prefix, method)
        require(index not in score_sets, "Repeated QP certificate family")
        lower_values, gaps, convergence = [], [], []
        require(len(row["lower"]) == count, "Missing QP programs")
        for position in range(count):
            start = position*size
            stop = start+size
            fields = ["lower", "upper", "gap", "sse_lower", "sse_upper", "dual_raw", "dual_guard",
                      "signed_dual", "primal", "iterations", "converged", "feasibility_residual"]
            record = {name: row[name][position] for name in fields}
            record["block_size"] = size
            checked = audit_qp(sample["features"][start:stop, :width], sample["y"][start:stop],
                               sample["slices"][:prefix], config["kappa"], record, checks, diameter)
            lower_values.append(checked["lower"])
            gaps.append(checked["gap"])
            convergence.append(checked["solver_converged"])
            max_gap = max(max_gap, checked["gap"])
            max_violation = max(max_violation, checked["violation"])
            qp_count += 1
            unconverged_count += int(not checked["solver_converged"])
        score_sets[index] = np.asarray(lower_values)
        qp_diagnostics[index] = {"gaps": np.asarray(gaps), "converged": np.asarray(convergence)}
        fallback_batches += int(row["arithmetic_fallback"] is not None)
        if len(score_sets) % 80 == 0:
            print(f"Independently checked {len(score_sets)} QP batches and {qp_count} programs", flush=True)
    require(len(score_sets) == 40*2*4, "Missing declared QP batches")
    require(qp_count == 40*2*(maximum_n//4+maximum_n//8+maximum_n//16+1), "QP count differs")
    require(qp_count == summary["qp_count"] and unconverged_count == summary["unconverged_qp_count"],
            "QP diagnostic counts differ")
    require(fallback_batches == summary["arithmetic_fallback_batches"], "Arithmetic fallback count differs")
    checks.close(max_gap, summary["maximum_normalized_gap"], "Maximum normalized gap", atol=5e-12)
    checks.close(max_violation, summary["maximum_feasibility_residual"], "Maximum primal residual", atol=5e-12)

    stratified_lookup = {}
    for row in read_json_lines(results/"stratified_details.jsonl.gz"):
        key = row["regime"], row["repetition"]
        prefix, n, partition = int(row["prefix"]), int(row["n"]), row["partition"]
        index = (*key, prefix, n, partition)
        require(index not in stratified_lookup, "Duplicate stratified endpoint")
        sample = samples[key]
        width = config["retained_costs"][prefix-1]
        labels = sample["labels"][partition][:n]
        classes = [2, 4][config["partitions"].index(partition)]
        counts = np.bincount(labels, minlength=classes)
        radius = math.sqrt(math.log(4*classes/eta)/(2*n))
        lo, hi = np.maximum(0, counts/n-radius), np.minimum(1, counts/n+radius)
        checks.close(counts, row["counts"], "Stratum counts", atol=0, rtol=0)
        checks.close(radius, row["mass_radius"], "Mass interval radius", atol=1e-14)
        checks.close(lo, row["mass_lower"], "Mass interval lower", atol=1e-14)
        checks.close(hi, row["mass_upper"], "Mass interval upper", atol=1e-14)
        conditional, total_pairs = [], 0
        for label in range(classes):
            selected = np.flatnonzero(labels == label)
            scores = pair_scores(sample["features"][selected, :width], sample["y"][selected],
                                 sample["slices"][:prefix], config["kappa"])
            pair_mean = float(scores.mean()) if len(scores) else 0.
            lower = 0. if len(scores) < 2 else lower_kl(pair_mean, len(scores), loss_bound/2, eta/(2*classes))
            detail = row["conditional"][label]
            require(detail["stratum"] == label and detail["count"] == len(selected)
                    and detail["pairs"] == len(scores), "Stratum pairing differs")
            checks.close(pair_mean, detail["mean"], "Conditional pair score mean", atol=5e-12)
            checks.close(lower, detail["lower"], "Conditional KL lower endpoint", atol=5e-12)
            checks.close([detail["eta"], detail["bound"]], [eta/(2*classes), loss_bound/2],
                         "Conditional allocation and range", atol=1e-14)
            conditional.append(lower)
            total_pairs += len(scores)
        # Positive objective scaling preserves the optimizer. It avoids HiGHS
        # treating the tiny conditional endpoints in unhelpful partitions as
        # numerically indistinguishable under its default dual tolerance.
        objective_scale = max(max(conditional), 1e-100)
        program = linprog(np.asarray(conditional)/objective_scale,
                          A_eq=np.ones((1, classes)), b_eq=[1.],
                          bounds=list(zip(lo, hi)), method="highs",
                          options={"dual_feasibility_tolerance": 1e-10,
                                   "primal_feasibility_tolerance": 1e-10})
        require(program.success, "Independent mass linear program failed")
        minimum = float(program.fun)*objective_scale
        lower = max(0., minimum-1e-12)
        checks.close(lower, row["lower"], "Independent mass LP endpoint", atol=5e-11)
        mass = np.asarray(row["minimizing_mass"])
        require(np.all(mass >= lo-1e-12) and np.all(mass <= hi+1e-12)
                and abs(float(mass.sum())-1) < 1e-12, "Recorded mass minimizer is infeasible")
        checks.close(mass@conditional, minimum, "Recorded mass LP optimum", atol=5e-11)
        stratified_lookup[index] = {"lower": lower, "pairs": total_pairs}
    require(len(stratified_lookup) == expected_counts["stratified"], "Missing stratified endpoints")
    print(f"Recomputed {len(stratified_lookup)} stratified mass linear programs", flush=True)

    with (results/"endpoints.csv").open(newline="") as stream:
        endpoint_rows = list(csv.DictReader(stream))
    require(len(endpoint_rows) == expected_counts["total"] == summary["n_endpoints"],
            "Missing or extra confidence endpoints")
    verified_endpoints = {}
    irreducible_class_risk = .8**2*(.9**2+.1**2/3)**2+.05**2/3
    known_risk_lower_checks = 0
    known_risk_lower_violations = 0
    for row in endpoint_rows:
        key = row["regime"], int(row["repetition"])
        prefix, n, method = int(row["prefix"]), int(row["n"]), row["method"]
        index = (*key, prefix, n, method)
        require(index not in verified_endpoints, "Duplicate confidence endpoint")
        sample = samples[key]
        width = config["retained_costs"][prefix-1]
        checks.close(float(row["eta"]), eta, "Endpoint failure allocation", atol=1e-18)
        require(int(row["cost"]) == width, "Endpoint cost differs")
        if method == "fitted_upper":
            losses = (sample["y"][:n]-sample["predictions"][:n, prefix-1])**2
            _, endpoint = empirical_bernstein(losses, loss_bound, eta)
            mean, bound, count, block_size = float(losses.mean()), loss_bound, n, 1
            checks.close(endpoint, float(row["upper"]), "Fitted-risk upper endpoint", atol=5e-12)
            verified_endpoints[index] = {"upper": endpoint}
        elif method.startswith("block_"):
            size = int(method.split("_")[1])
            if size == 2:
                scores = pair_scores(sample["features"][:n, :width], sample["y"][:n],
                                     sample["slices"][:prefix], config["kappa"])
                maximum_gap, failed = 0., 0
            else:
                scores = score_sets[*key, prefix, method][:n//size]
                diagnostic = qp_diagnostics[*key, prefix, method]
                maximum_gap = float(diagnostic["gaps"][:n//size].max())
                failed = int((~diagnostic["converged"][:n//size]).sum())
            mean, bound, count, block_size = float(scores.mean()), score_bound(size, diameter), len(scores), size
            endpoint = lower_kl(mean, count, bound, eta)
            checks.close(endpoint, float(row["lower"]), "Block KL lower endpoint", atol=5e-12)
            checks.close(maximum_gap, float(row["maximum_gap"]), "Block endpoint optimization gap", atol=5e-12)
            require(failed == int(row["unconverged_programs"]), "Unconverged program count differs")
            verified_endpoints[index] = {"lower": endpoint}
        elif method.startswith("stratified_"):
            partition = method[len("stratified_"):]
            detail = stratified_lookup[*key, prefix, n, partition]
            endpoint = detail["lower"]
            mean, bound, count, block_size = None, loss_bound/2, detail["pairs"], 2
            checks.close(endpoint, float(row["lower"]), "Stratified endpoint CSV", atol=5e-11)
            verified_endpoints[index] = {"lower": endpoint}
        elif method == "full_erm":
            require(n in config["full_erm_sizes"], "ERM comparison claimed at uncomputed sample size")
            mean = float(score_sets[*key, prefix, method][0])*(n-1)/n
            bound, count, block_size = loss_bound, n, n
            endpoint = lower_kl(mean, count, bound, eta)
            checks.close(endpoint, float(row["lower"]), "Full empirical-minimum endpoint", atol=5e-12)
            diagnostic = qp_diagnostics[*key, prefix, method]
            checks.close(float(diagnostic["gaps"][0])*(n-1)/n, float(row["maximum_gap"]),
                         "Full empirical-minimum gap normalization", atol=5e-12)
            require(int(not diagnostic["converged"][0]) == int(row["unconverged_programs"]),
                    "ERM stopping-limit status differs")
            verified_endpoints[index] = {"lower": endpoint}
        else:
            raise AssertionError("Unknown endpoint method: "+method)
        require(int(row["n_independent"]) == count and int(row["block_size"]) == block_size,
                "Wrong independent sample count or block size")
        checks.close(bound, float(row["score_bound"]), "Endpoint range bound", atol=1e-14)
        if mean is not None:
            checks.close(mean, float(row["score_mean"]), "Endpoint score mean", atol=5e-12)
        if key[0] == "irreducible" and method != "fitted_upper":
            known_risk_lower_checks += 1
            known_risk_lower_violations += int(endpoint > irreducible_class_risk)

    with (results/"decisions.csv").open(newline="") as stream:
        decision_rows = list(csv.DictReader(stream))
    decision_map = {(row["regime"], int(row["repetition"]), int(row["n"]), row["method"]): row
                    for row in decision_rows}
    require(len(decision_map) == len(decision_rows) == summary["n_decisions"] == 880,
            "Missing or duplicate decision records")
    outcomes = {}
    for key in expected_banks:
        for n in config["calibration_sizes"]:
            uppers = [verified_endpoints[*key, prefix, n, "fitted_upper"]["upper"]
                      for prefix in config["prefixes"]]
            accepted = [cost for cost, upper in zip(config["retained_costs"], uppers) if upper <= .1]
            cost_up = min(accepted) if accepted else None
            alternatives = {f"block_{b}": [f"block_{b}"] for b in config["block_sizes"]}
            alternatives["best_block"] = [f"block_{b}" for b in config["block_sizes"]]
            alternatives["stratified_pair"] = ["stratified_"+name for name in config["partitions"]]
            if n in config["full_erm_sizes"]:
                alternatives["full_erm"] = ["full_erm"]
            alternatives["all_available"] = sorted({source for sources in alternatives.values() for source in sources})
            for method, sources in alternatives.items():
                row = decision_map[*key, n, method]
                lowers = [max(verified_endpoints[*key, prefix, n, source]["lower"] for source in sources)
                          for prefix in config["prefixes"]]
                compatible = [cost for cost, lower in zip(config["retained_costs"], lowers) if lower <= .1]
                cost_low = min(compatible) if compatible else None
                computed = {"upper_choice": cost_up, "cost_lower": cost_low,
                            "minimum_cost_certified": cost_up is not None and cost_up == cost_low,
                            "all_classes_excluded": cost_low is None,
                            "upper_accepted": cost_up is not None}
                for name, value in computed.items():
                    require(row[name] == ("" if value is None else str(value)), "Decision differs: "+name)
                checks.close(lowers, [float(row["lower_cost2"]), float(row["lower_cost6"])],
                             "Decision class lower endpoints", atol=5e-11)
                category = key[0], n, method
                outcome = outcomes.setdefault(category, {"upper_acceptances": 0,
                    "minimum_cost_certificates": 0, "joint_infeasibility_certificates": 0,
                    "unresolved": 0})
                outcome["upper_acceptances"] += int(computed["upper_accepted"])
                outcome["minimum_cost_certificates"] += int(computed["minimum_cost_certified"])
                outcome["joint_infeasibility_certificates"] += int(computed["all_classes_excluded"])
                outcome["unresolved"] += int(not computed["minimum_cost_certified"] and not computed["all_classes_excluded"])

    report = {"passed": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "No project imports: direct KAHM feature and selected-head replay; fresh episode replay; long-double signed dual and primal feasibility; independent confidence arithmetic, mass linear programs, and all class decisions.",
              "frozen_banks": len(models), "fresh_calibration_seeds": len(fresh_seeds),
              "fresh_calibration_observations": len(samples)*maximum_n,
              "selected_predictors": len(selected_norms), "maximum_selected_coefficient_norm": max(selected_norms),
              "qp_programs": qp_count, "qp_stopping_limit_cases": unconverged_count,
              "arithmetic_fallback_batches": fallback_batches, "maximum_normalized_qp_gap": max_gap,
              "maximum_primal_feasibility_residual": max_violation,
              "confidence_endpoints": len(verified_endpoints), "stratified_mass_programs": len(stratified_lookup),
              "class_decisions": len(decision_rows), "total_failure_allocation": eta*len(verified_endpoints),
              "irreducible_known_class_risk": irreducible_class_risk,
              "irreducible_lower_endpoints_compared_with_known_class_risk": known_risk_lower_checks,
              "irreducible_observed_lower_coverage_violations": known_risk_lower_violations,
              "floating_scalar_checks": checks.count, "maximum_absolute_discrepancies": checks.maximum,
              "archived_sources_with_current_checkout_changes": source_current_mismatches,
              "current_executable_sources_match_frozen_snapshot": not any(
                  name.endswith(".py") for name in source_current_mismatches),
              "frozen_code_development_log_literal_references": development_log_references,
              "outcomes": [{"regime": regime, "n": n, "method": method, **values}
                           for (regime, n, method), values in sorted(outcomes.items())],
              "run_summary_sha256": digest(results/"run_summary.json"),
              "limitations": ["Numerical verification, not formal interval arithmetic.",
                "Frozen source and timestamps document recorded execution, not external preregistration.",
                "New calibration conditional on reused KAHM banks; no new training replication.",
                "The controlled-law endpoints do not establish natural-task prevalence or universal KAHM superiority.",
                "Full empirical minimization is evaluated only at n=256; larger sample budgets are uncomputed."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({name: report[name] for name in ["passed", "frozen_banks", "fresh_calibration_observations",
        "qp_programs", "qp_stopping_limit_cases", "confidence_endpoints", "stratified_mass_programs",
        "class_decisions", "maximum_normalized_qp_gap", "maximum_primal_feasibility_residual"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.results, arguments.output)
