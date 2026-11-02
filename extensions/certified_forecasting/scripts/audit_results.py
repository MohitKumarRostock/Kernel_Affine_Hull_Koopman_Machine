"""Reconstruct every retained forecasting certificate and audit provenance.

This script does not treat finite test-set error as population truth, and never
calls agreement with that test error a coverage result. It independently
recomputes the stated confidence formulas from retained episode scores.

Example (after the campaign and optional replay have completed)::

    python scripts/audit_results.py --campaign results/main_v1 \
        --config configs/main_v1.json --replay results/reproduction_check \
        --output results/audits/main_v1_audit.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXTENSION_ROOT.parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Checks:
    def __init__(self):
        self.failures = []
        self.n_checks = 0
        self.maximum_numeric_difference = 0.0

    def require(self, condition, label):
        self.n_checks += 1
        if not bool(condition):
            self.failures.append(label)

    def compare(self, observed, expected, label):
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                self.require(False, label + ": expected mapping")
                return
            self.require(set(observed) == set(expected), label + ": mapping keys differ")
            for key in sorted(set(expected).intersection(observed)):
                self.compare(observed[key], expected[key], f"{label}/{key}")
        elif isinstance(expected, (list, tuple)):
            if not isinstance(observed, (list, tuple)):
                self.require(False, label + ": expected sequence")
                return
            self.require(len(observed) == len(expected), label + ": lengths differ")
            for i, (left, right) in enumerate(zip(observed, expected)):
                self.compare(left, right, f"{label}/{i}")
        elif isinstance(expected, (int, float, np.number)) and not isinstance(expected, (bool, np.bool_)):
            try:
                left, right = float(observed), float(expected)
                finite = math.isfinite(left) and math.isfinite(right)
                self.require(finite, label + ": non-finite number")
                if finite:
                    difference = abs(left - right)
                    self.maximum_numeric_difference = max(self.maximum_numeric_difference, difference)
                    self.require(math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-12),
                                 label + f": numerical mismatch {left!r} vs {right!r}")
            except (TypeError, ValueError):
                self.require(False, label + ": incompatible numeric value")
        else:
            self.require(observed == expected, label + ": values differ")


def confidence_formula(values, bound, delta, method):
    """Independent formula implementation; does not import certificates.py."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    average = float(np.mean(x))
    variance = float(np.sum((x - average)**2) / (n - 1))
    if method == "bernstein":
        logarithm = math.log(2.0 / delta)
        radius = math.sqrt(2.0 * variance * logarithm / n)
        radius += (7.0 * bound * logarithm) / (3.0 * (n - 1))
    elif method == "hoeffding":
        radius = bound * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    else:
        raise ValueError(f"Unsupported confidence method {method!r}")
    return {
        "mean": average, "variance": variance, "radius": radius,
        "lower": max(0.0, average - radius),
        "upper": min(bound, average + radius),
        "n_independent": n, "bound": float(bound),
        "delta_per_tail": float(delta), "method": method,
    }


def expected_names(configuration):
    names, pairs = [], []
    for history in configuration["histories"]:
        for kind in configuration["feature_kinds"]:
            previous = None
            for stage in range(1, len(configuration["cluster_counts"]) + 1):
                name = f"{kind}_nested_m{history}_s{stage}"
                names.append(name)
                if previous is not None and kind == "kahm":
                    pairs.append((previous, name))
                previous = name
            if kind == "kahm":
                names.extend(f"kahm_{family}_m{history}" for family in
                             ("frobenius", "duplicate", "hard", "koopman"))
        names.extend(f"{family}_m{history}" for family in
                     ("ridge_ar", "polynomial", "polynomial_koopman"))
    return names + ["persistence", "constant"], pairs


def verify_manifest(campaign, configuration_path, checks):
    manifest = read_json(campaign / "manifest.json")
    configuration = read_json(configuration_path)
    checks.compare(manifest["configuration"], configuration, "manifest/configuration")
    checks.compare(manifest["configuration_sha256"], sha256(configuration_path),
                   "manifest/configuration_sha256")
    source_hashes = manifest["source_sha256"]
    checks.require("experiments/run_forecasting.py" in source_hashes,
                   "manifest must hash the experiment runner")
    checks.require("parent/kernel_affine_hull_koopman_machines.py" in source_hashes,
                   "manifest must hash the actual parent KAHM implementation")
    required_helpers = ["autoencoder.py", "autoencoder_filtering_extended.py",
                        "combine_multiple_autoencoders_extended.py", "dim_reduce.py",
                        "distance_matrix.py", "kernel_regularized_least_squares.py",
                        "kxa_matrix.py", "kxx_matrix.py", "parallel_autoencoders.py"]
    for helper in required_helpers:
        checks.require("parent/" + helper in source_hashes,
                       "manifest must hash parent helper " + helper)
    for name, expected in source_hashes.items():
        source = (REPOSITORY_ROOT / name[len("parent/"):]
                  if name.startswith("parent/") else EXTENSION_ROOT / name)
        checks.require(source.is_file(), "manifest source missing: " + name)
        if source.is_file():
            checks.compare(sha256(source), expected, "manifest/source/" + name)
    return manifest, configuration


def check_norm_metadata(candidate, checks, label):
    budget = np.asarray(candidate["kappa"], dtype=float)
    checks.require(np.isfinite(budget).all() and np.all(budget >= 0),
                   label + ": finite nonnegative Lipschitz budget")
    if candidate["metric"] == "block_max":
        checks.require("norm_sum" in candidate, label + ": missing block norm diagnostic")
        if "norm_sum" in candidate:
            checks.require(candidate["norm_sum"] <= float(budget) + 1e-8,
                           label + ": sum of block norms exceeds declared budget")
    elif "frobenius_norm" in candidate:
        checks.require(candidate["frobenius_norm"] <= float(budget) + 1e-8,
                       label + ": Frobenius norm exceeds declared budget")
    if "optimization_gap" in candidate:
        checks.require(math.isfinite(candidate["optimization_gap"])
                       and candidate["optimization_gap"] >= 0,
                       label + ": invalid optimization gap")
    if candidate["family"] == "persistence":
        checks.require(np.all(budget == 1.0), label + ": persistence budget must be one")
    if candidate["family"] == "constant":
        checks.require(np.all(budget == 0.0), label + ": constant budget must be zero")


def audit_run(directory, cfg, expected_system, expected_seed, checks):
    name = directory.name
    for filename in ("results.json", "frozen_candidates.json", "episode_scores.npz"):
        checks.require((directory / filename).is_file(), f"{name}: missing {filename}")
    if not all((directory / f).is_file() for f in
               ("results.json", "frozen_candidates.json", "episode_scores.npz")):
        return {"run": name, "status": "missing_artifacts"}
    record = read_json(directory / "results.json")
    frozen = read_json(directory / "frozen_candidates.json")
    names, pairs = expected_names(cfg)
    results = record["results"]
    H, K = cfg["horizons"], len(names)
    tail = cfg["delta"] * cfg["risk_delta_fraction"] / (2 * K * len(H))
    pair_tail = (cfg["delta"] * (1.0 - cfg["risk_delta_fraction"]) / (len(pairs) * len(H))
                 if pairs else 0.0)
    checks.compare(record["system"], expected_system, name + "/system")
    checks.compare(record["seed"], expected_seed, name + "/seed")
    checks.compare(record["n_candidates"], K, name + "/candidate_count")
    checks.compare([r["name"] for r in results], names, name + "/candidate_names")
    checks.compare([r["name"] for r in frozen], names, name + "/frozen_names")
    checks.compare(record["n_independent_calibration_episodes"], cfg["n_calibration"], name + "/n_calibration")
    checks.compare(record["n_disjoint_calibration_pairs"], cfg["n_calibration"] // 2, name + "/n_pairs")
    checks.compare(record["delta"], cfg["delta"], name + "/delta")
    checks.compare(record["tail_delta"], tail, name + "/tail_delta")
    stream_offsets = {"train": 11, "validation": 23, "calibration": 47, "test": 89}
    checks.compare(record["streams"], {s: expected_seed * 100 + offset for s, offset in stream_offsets.items()},
                   name + "/streams")
    risk_budget = 2 * K * len(H) * tail
    gate_budget = len(pairs) * len(H) * pair_tail
    checks.require(risk_budget + gate_budget <= cfg["delta"] + 1e-15,
                   name + ": joint failure-probability allocation exceeded")
    frozen_by_name = {r["name"]: r for r in frozen}
    data = np.load(directory / "episode_scores.npz", allow_pickle=False)
    expected_keys = {candidate + suffix for candidate in names for suffix in
                     ("__cal_loss", "__pair_score", "__test_loss")}
    checks.require(set(data.files) == expected_keys, name + ": episode-score keys differ")
    overlap_count = 0
    nonconverged = []
    for candidate in results:
        candidate_name = candidate["name"]
        label = name + "/" + candidate_name
        frozen_description = frozen_by_name[candidate_name]
        checks.compare({k: candidate[k] for k in frozen_description}, frozen_description, label + "/frozen")
        check_norm_metadata(candidate, checks, label)
        if candidate.get("converged") is False:
            nonconverged.append(candidate_name)
        losses, witnesses, test_losses = (data[candidate_name + suffix] for suffix in
                                          ("__cal_loss", "__pair_score", "__test_loss"))
        for array, n, upper, kind in ((losses, cfg["n_calibration"], 4.0, "calibration"),
                                      (witnesses, cfg["n_calibration"] // 2, 2.0, "pairs"),
                                      (test_losses, cfg["n_test"], 4.0, "test")):
            checks.require(array.shape == (n, len(H)), label + "/" + kind + ": shape mismatch")
            checks.require(np.isfinite(array).all() and np.all(array >= -1e-12)
                           and np.all(array <= upper + 1e-12),
                           label + "/" + kind + ": scores outside known bounded range")
        checks.compare(len(candidate["certificates"]), len(H), label + "/certificate_count")
        for h in range(len(H)):
            upper = confidence_formula(losses[:, h], 4.0, tail, cfg["certificate_method"])
            lower = confidence_formula(witnesses[:, h], 2.0, tail, cfg["certificate_method"])
            certificate = {
                "mse_upper": upper["upper"], "mse_lower": lower["lower"],
                "rmse_upper": math.sqrt(upper["upper"]), "rmse_lower": math.sqrt(lower["lower"]),
                "empirical_rmse": math.sqrt(upper["mean"]),
                "upper_details": upper, "lower_details": lower,
            }
            checks.compare(candidate["certificates"][h], certificate, label + f"/h{H[h]}")
            overlap_count += int(certificate["mse_lower"] <= certificate["mse_upper"])
        checks.compare(candidate["test_mse"], test_losses.mean(axis=0).tolist(), label + "/test_mean")
        checks.compare(candidate["test_mse_se"],
                       (test_losses.std(axis=0, ddof=1) / math.sqrt(len(test_losses))).tolist(),
                       label + "/test_standard_error")
    expected_gates = []
    for old, new in pairs:
        for h_index, horizon in enumerate(H):
            shifted = data[old + "__cal_loss"][:, h_index] - data[new + "__cal_loss"][:, h_index] + 4.0
            interval = confidence_formula(shifted, 8.0, pair_tail, cfg["certificate_method"])
            interval["improvement_lower"] = interval["lower"] - 4.0
            interval["empirical_improvement"] = interval["mean"] - 4.0
            expected_gates.append({"old": old, "new": new, "horizon": horizon, **interval})
    checks.compare(record["paired_improvements"], expected_gates, name + "/paired_gates")
    selected_validation = min(results, key=lambda r: (np.mean(r["validation_mse"]), r["n_features"]))["name"]
    checks.compare(record["selected_by_validation"], selected_validation, name + "/validation_selection")
    expected_selections = []
    for tolerance in cfg["rmse_tolerances"]:
        for family in ("all", "kahm_nested"):
            eligible = [r for r in results if family == "all" or r["family"] == family]
            passed = [r for r in eligible if all(ci["rmse_upper"] <= tolerance for ci in r["certificates"])]
            chosen = (min(passed, key=lambda r: (r["n_features"], r["history"], np.mean(r["validation_mse"])))
                      if passed else None)
            expected_selections.append({"tolerance": tolerance, "family": family,
                                        "selected": None if chosen is None else chosen["name"],
                                        "n_features": None if chosen is None else chosen["n_features"],
                                        "status": "certified" if chosen is not None else "unresolved"})
    checks.compare(record["certified_selections"], expected_selections, name + "/certified_selections")
    data.close()
    return {
        "run": name, "candidate_horizon_certificates_recomputed": K * len(H),
        "one_sided_risk_endpoints_recomputed": 2 * K * len(H),
        "paired_improvement_statements_recomputed": len(expected_gates),
        "independent_calibration_episodes": cfg["n_calibration"],
        "independent_disjoint_pairs": cfg["n_calibration"] // 2,
        "allocated_risk_failure_probability": risk_budget,
        "allocated_refinement_failure_probability": gate_budget,
        "nonconverged_decoders": nonconverged,
        "noncrossing_sandwiches": overlap_count,
        "results_sha256": sha256(directory / "results.json"),
        "episode_scores_sha256": sha256(directory / "episode_scores.npz"),
    }


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items()
                if k not in {"train_seconds", "elapsed_seconds"}}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def audit_replay(original, replay, checks):
    source_manifest = read_json(original / "manifest.json")
    repeat_manifest = read_json(replay / "manifest.json")
    for key in ("configuration", "configuration_sha256", "source_sha256", "python", "numpy", "scipy", "sklearn"):
        checks.compare(repeat_manifest[key], source_manifest[key], "replay/manifest/" + key)
    repeated_runs = sorted(p.parent for p in replay.glob("*/results.json"))
    checks.require(bool(repeated_runs), "replay must contain a completed run")
    results = []
    for repeated in repeated_runs:
        first = original / repeated.name
        checks.require((first / "results.json").is_file(), "replay original missing: " + repeated.name)
        if not (first / "results.json").is_file():
            continue
        original_results = without_timing(read_json(first / "results.json"))
        repeated_results = without_timing(read_json(repeated / "results.json"))
        checks.compare(repeated_results, original_results, "replay/results/" + repeated.name)
        original_frozen = without_timing(read_json(first / "frozen_candidates.json"))
        repeated_frozen = without_timing(read_json(repeated / "frozen_candidates.json"))
        checks.compare(repeated_frozen, original_frozen, "replay/frozen/" + repeated.name)
        exact_json = (original_results == repeated_results and original_frozen == repeated_frozen)
        array_count, scalar_count, max_difference, exact_arrays = 0, 0, 0.0, True
        with np.load(first / "episode_scores.npz", allow_pickle=False) as before, \
             np.load(repeated / "episode_scores.npz", allow_pickle=False) as after:
            checks.require(set(before.files) == set(after.files), "replay score keys differ")
            for key in sorted(set(before.files).intersection(after.files)):
                left, right = before[key], after[key]
                checks.require(left.shape == right.shape, "replay score shapes differ: " + key)
                if left.shape != right.shape:
                    exact_arrays = False
                    continue
                difference = float(np.max(np.abs(left - right), initial=0.0))
                max_difference = max(max_difference, difference)
                equal = np.array_equal(left, right)
                exact_arrays = exact_arrays and equal
                checks.require(np.allclose(left, right, rtol=1e-11, atol=1e-12),
                               "replay score values differ: " + key)
                array_count += 1
                scalar_count += left.size
        results.append({
            "run": repeated.name, "all_nontiming_json_exactly_equal": exact_json,
            "all_score_arrays_exactly_equal": bool(exact_arrays),
            "score_arrays_compared": array_count, "score_scalars_compared": scalar_count,
            "maximum_score_absolute_difference": max_difference,
            "original_results_sha256": sha256(first / "results.json"),
            "replayed_results_sha256": sha256(repeated / "results.json"),
            "replayed_episode_scores_sha256": sha256(repeated / "episode_scores.npz"),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=Path("results/main_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/main_v1.json"))
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/audits/main_v1_audit.json"))
    args = parser.parse_args()
    checks = Checks()
    manifest, cfg = verify_manifest(args.campaign, args.config, checks)
    expected = {(system, seed): f"{system}_{seed}" for system in cfg["systems"] for seed in cfg["seeds"]}
    observed = {p.name for p in args.campaign.iterdir() if p.is_dir()}
    checks.require(observed == set(expected.values()), "Campaign run directories differ from the frozen full schedule")
    failures = sorted(str(p.relative_to(args.campaign)) for p in args.campaign.rglob("failure.json"))
    checks.require(not failures, "Campaign contains recorded failed attempts")
    runs = []
    for (system, seed), name in expected.items():
        directory = args.campaign / name
        if directory.is_dir():
            runs.append(audit_run(directory, cfg, system, seed, checks))
        else:
            checks.require(False, "Scheduled run missing: " + name)
    replay_report = audit_replay(args.campaign, args.replay, checks) if args.replay else []
    output = {
        "status": "PASS" if not checks.failures else "FAIL",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha256(__file__),
        "campaign_manifest_sha256": sha256(args.campaign / "manifest.json"),
        "configuration_sha256": manifest["configuration_sha256"],
        "scheduled_runs": len(expected), "run_directories_present": len(observed),
        "completed_runs_audited": sum("candidate_horizon_certificates_recomputed" in run for run in runs),
        "recorded_failures": failures,
        "candidate_horizon_certificates_recomputed": sum(run.get("candidate_horizon_certificates_recomputed", 0) for run in runs),
        "one_sided_risk_endpoints_recomputed": sum(run.get("one_sided_risk_endpoints_recomputed", 0) for run in runs),
        "paired_improvement_statements_recomputed": sum(run.get("paired_improvement_statements_recomputed", 0) for run in runs),
        "checks_performed": checks.n_checks,
        "maximum_formula_numeric_difference": checks.maximum_numeric_difference,
        "failures": checks.failures, "runs": runs, "replay": replay_report,
        "scope": [
            "Risk endpoints and paired gates are recomputed from retained per-episode scores.",
            "Known target and forecast range [-1,1] gives loss bound4 and pair-score bound2.",
            "Counts and confidence allocations refer to independent episodes, not overlapping windows.",
            "Direct decoder norm feasibility is checked against frozen retained norm diagnostics.",
            "Koopman composed norms are retained budgets; only replayed models are independently refitted here.",
            "The test-set means and standard errors are arithmetic checks, not estimates of certificate coverage.",
            "This audit does not independently establish scientific novelty or prove theorem correctness.",
        ],
    }
    atomic_json(args.output, output)
    print(json.dumps({key: output[key] for key in
                      ("status", "scheduled_runs", "completed_runs_audited",
                       "candidate_horizon_certificates_recomputed", "one_sided_risk_endpoints_recomputed",
                       "paired_improvement_statements_recomputed", "checks_performed")}, indent=2))
    if checks.failures:
        print("\n".join(checks.failures[:20]), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
