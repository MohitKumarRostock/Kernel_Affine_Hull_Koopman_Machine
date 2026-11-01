#!/usr/bin/env python3
"""Independent seed, grid and causal-index checks without generating episodes."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(parts):
    return int(np.random.SeedSequence([int(x) for x in parts]).generate_state(1)[0])


def declared_seeds(config):
    records = []
    master = config["master_seed"]
    for repetition in range(config["repetitions"]):
        parts = [master, repetition, 1000]
        records.append({"role": "repetition_identifier", "repetition": repetition,
                        "components": parts, "seed": seed(parts)})
        for law_index, law in enumerate(config["laws"]):
            parts = [master, repetition, law_index, 1004]
            records.append({"role": "fit", "repetition": repetition, "law": law,
                            "components": parts, "seed": seed(parts)})
            for split_index, split in enumerate(("A", "B", "C", "D")):
                for purpose, purpose_id in (("generation", 1001), ("unit_noise", 1002), ("rows", 1003)):
                    parts = [master, repetition, law_index, split_index, purpose_id]
                    records.append({"role": purpose, "repetition": repetition, "law": law, "split": split,
                                    "components": parts, "seed": seed(parts)})
    return records


def main(root):
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    original = json.loads((root / "vendor_original/extensions/reconstruction_mechanism/config.json").read_text())
    issues, assertions = [], 0

    def check(condition, label, context=None):
        nonlocal assertions
        assertions += 1
        if not bool(condition):
            issues.append({"check": label, "context": context})

    for key, expected in {
        "master_seed": 2026110103, "repetitions": 24,
        "laws": ["linear_ar", "nonlinear_ar", "vanderpol", "duffing"],
        "nonlinear_laws": ["nonlinear_ar", "vanderpol", "duffing"],
        "histories": [12], "primary_history": 12, "noise_levels": [0., .1],
        "horizons": [1, 4, 12], "stride": 4, "max_episode_length": 72,
        "history_tail_length": 60, "origins_per_episode": 13, "task_count": 192,
        "split_episodes": {"A": 64, "B": 32, "C": 32, "D": 128},
        "split_caps": {"A": 512, "B": 256, "C": 256, "D": None},
        "split_indices": {"A": 0, "B": 1, "C": 2, "D": 3},
        "seed_purposes": {"identifier": 1000, "generation": 1001, "unit_noise": 1002, "rows": 1003, "fit": 1004},
        "native_cap_order": [1, 2, 4, 8, 12, 20],
        "native_candidate_order": ["native1", "native2", "native4", "native8", "native12", "native20"],
        "geometry_scaling": "standard",
        "methods": ["ridge", "raw_rbf", "raw_hgb", "persistence", "common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy"],
    }.items():
        check(config[key] == expected, "final declared design", key)
    expected_candidates = [("historical2", 2, "legacy_normal_departure")] + [(f"native{x}", x, "native") for x in (1, 2, 4, 8, 12, 20)]
    observed = [(r["candidate_id"], r["rank_cap"], r["dimension_policy"]) for r in config["candidates"]]
    check(observed == expected_candidates, "exact fixed and selectable candidate roster")
    tree = ast.parse((root / "code/dimension_worker.py").read_text())
    declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "CANDIDATES" for t in node.targets)]
    check(len(declarations) == 1 and list(ast.literal_eval(declarations[0])) == expected_candidates,
          "worker candidate definitions equal frozen config")
    check(config["linear_ar"] == original["linear_ar"], "original linear numerical law retained")
    for key in ("alpha_grid", "gamma_grid", "blocks", "max_references", "bank_row_cap", "relative_denominator_tolerance", "feature_scale_floor", "target_variance_floor", "raw_rbf_landmarks", "raw_rbf_bandwidth_factors", "raw_hgb_learning_rates", "raw_hgb_max_leaf_nodes", "raw_hgb_max_iter", "raw_hgb_l2_regularization"):
        check(config["model"][key] == original["model"][key], "retained source primitive configuration", key)
    check(config["analysis"]["primary_family_size"] == 6
          and config["analysis"]["family_alpha"] == .05
          and config["analysis"]["primary_context"] == "selected"
          and config["analysis"]["superiority_margin"] == .03
          and config["analysis"]["linear_noninferiority_margin"] == .05
          and config["analysis"]["max_error_ratio"] == 5.
          and config["analysis"]["independent_unit"] == "repetition", "original six-margin inference, selected context only")
    check(config["selection"]["split"] == "B" and config["selection"]["prediction_mode"] == "gamma1"
          and config["selection"]["freeze_before_C"] is True, "pre-calibration cap selection role")

    streams = declared_seeds(config)
    seeds = [record["seed"] for record in streams]
    check(len(seeds) == len(set(seeds)) == 1272 == config["seed_derivation"]["distinct_seed_count"],
          "all1272 intended-independent stream identifiers are unique")
    old_streams = set(original["seeds"] + [original["development_seed"]])
    for old_seed in original["seeds"] + [original["development_seed"]]:
        for law_index in range(len(original["laws"])):
            old_streams.add(seed([old_seed, law_index, original["seed_salts"]["fit"]]))
            for split_index in range(4):
                for purpose in ("generation", "unit_noise", "caps"):
                    old_streams.add(seed([old_seed, law_index, split_index, original["seed_salts"][purpose]]))
    old_config = {**config, "master_seed": 2026110102, "repetitions": 16}
    old_streams |= {r["seed"] for r in declared_seeds(old_config)}
    excluded = json.loads((root / "readiness/prior_seed_exclusions.json").read_text())
    check(old_streams == {r["seed"] for r in excluded["records"]}
          and len(old_streams) == excluded["unique_excluded_seed_count"] == 2173,
          "independent original and Step2 exclusion-set reconstruction")
    for record in excluded["records"]:
        if "components" in record and len(record["components"]) > 1:
            check(seed(record["components"]) == record["seed"], "prior seed derivation identity")
    collisions = sorted(set(seeds) & old_streams)
    check(not collisions, "no derived seed reuses an archived study/development stream", collisions)

    window_records = []
    for d in config["histories"]:
        length = d + config["history_tail_length"]
        origins = np.arange(d-1, length-max(config["horizons"]), config["stride"])
        history_indices = origins[:, None] - np.arange(d-1, -1, -1)[None, :]
        target_indices = origins[:, None] + np.array(config["horizons"])[None, :]
        check(len(origins) == 13 and int(origins[0]) == d-1 and int(origins[-1]) == d+47,
              "same13 origins at each declared width", d)
        check(history_indices.min() == 0 and np.array_equal(history_indices[:, -1], origins)
              and np.all(history_indices <= origins[:, None]) and np.all(target_indices > origins[:, None])
              and target_indices.max() == length-1, "all histories causal and horizons share support", d)
        rows = {split: min(n*len(origins), config["split_caps"][split] or n*len(origins))
                for split, n in config["split_episodes"].items()}
        check(rows == {"A": 512, "B": 256, "C": 256, "D": 1664}, "equal declared row counts at both widths", d)
        window_records.append({"history": d, "prefix_length": length, "origins": origins.tolist(), "retained_rows": rows})

    result = {"schema": "kahm-full-forecast-independent-design-check-v1",
              "status": "passed" if not issues else "failed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "No episode generation and no production input or outcome reads.",
              "audit_source_sha256": sha(__file__), "config_sha256": sha(config_path),
              "worker_source_sha256": sha(root / "code/dimension_worker.py"),
              "counts": {"assertions": assertions, "new_unique_streams": len(set(seeds)), "archived_identifiers_and_streams": len(old_streams), "stream_collisions": len(collisions)},
              "seed_records_sha256": hashlib.sha256(json.dumps(streams, sort_keys=True).encode()).hexdigest(),
              "window_records": window_records, "issues": issues}
    output = root / "audit/design_config_checks.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.root.resolve()))
