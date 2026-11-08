#!/usr/bin/env python3
"""Replay the complete finite-cohort class-versus-head diagnosis offline.

The original independent auditor verifies stored witnesses using exact rational
arithmetic. This entry point does not fit a predictor or invoke an optimizer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_package():
    manifest = read_json(ROOT / "MANIFEST.json")
    require(manifest["schema"] == "class-diagnosis-reviewer-files-v1", "Unknown package manifest")
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Missing or unsafe package path: " + relative)
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                "Package checksum mismatch: " + relative)
    return len(manifest["files"])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(output):
    verified_files = verify_package()
    # These two original modules contain the independent arithmetic checker.
    # Their import does not run historical protocol gates or scientific fits.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "source"))
    import audit37 as audit
    import numpy as np

    require(sha(ROOT / "source/independent_math35.py") == audit.INDEPENDENT_MATH_SHA256,
            "Original independent arithmetic source differs")
    manifest, input_check = audit.verify_inputs()
    protocol = read_json(ROOT / "protocol/protocol.json")
    protocol_check = audit.audit_protocol(protocol, manifest)
    result = read_json(ROOT / "records/result.json")
    require(result["schema"] == "step37-full-class-diagnostic-result-v1"
            and result["status"] == "COMPLETE" and result["error"] is None
            and result["prior"] == audit.PRIOR_HASHES and result["scope"] == protocol["scientific_scope"],
            "Original result identity, completion, or scope differs")
    require(result["input_manifest_sha256"] == sha(ROOT / "inputs/source_manifest37.json")
            and result["protocol_sha256"] == sha(ROOT / "protocol/protocol.json")
            and result["protocol_seal_sha256"] == sha(ROOT / "protocol/original_seal.json"),
            "Original result is not bound to the supplied inputs and protocol")
    require(result["new_prediction_model_fits"] == result["new_forecast_predictions"] == 0,
            "A finite-cohort diagnostic was misreported as a new fitted forecast")
    ordered = [(task, name) for task in audit.TASKS for name in audit.map_names()]
    rows = result["class_rows"]
    require([(row["task"], row["map"]) for row in rows] == ordered
            and result["completed_classes"] == result["expected_classes"] == 54
            and result["inherited_heads"] == len(result["head_rows"]) == 108
            and result["missing_classes"] == [], "The complete class/head roster differs")

    cohorts, metadata = {}, {}
    common_ids = None
    for task in audit.TASKS:
        spec = manifest["tasks"][task]
        cohort = read_json(ROOT / spec["cohort_path"])
        ids = cohort["ids"]
        require(cohort["name"] == "decision" and cohort["target"] == task
                and len(ids) == len(set(ids)) == 172
                and all(isinstance(value, str) and value.startswith("decision:") for value in ids),
                "Unexpected cohort identity or observation roster")
        require(len(cohort["y"]) == 172
                and all(isinstance(value, str) and str(audit.q(value)) == value for value in cohort["y"]),
                "Original targets are not canonical exact rationals")
        common_ids = ids if common_ids is None else common_ids
        require(ids == common_ids, "Related targets do not share the same observations")
        targets = list(map(audit.q, cohort["y"]))
        floor = audit.original_target_floor(targets)
        multiplier = audit.q(spec["epsilon_physical_mse"]) / audit.q(spec["epsilon"])
        metadata[task] = {"decision_n": 172, "epsilon": spec["epsilon"],
                          "epsilon_physical_mse": spec["epsilon_physical_mse"],
                          "physical_mse_multiplier": str(multiplier),
                          "targets_outside_interval": floor["outside_count"],
                          "range_floor_mean": str(floor["mean"])}
        cohorts[task] = {"ids": ids, "y": targets, "physical_mse_multiplier": multiplier}
    require(metadata == result["task_metadata"], "Original target/range-floor metadata differs")

    checks, head_rows, class_table, head_table = [], [], [], []
    outcomes = {task: {"adequate_classes": 0, "inadequate_classes": 0, "unresolved_classes": 0,
                      "classes_with_both_heads_failing": 0, "heads_with_positive_excess": 0,
                      "adequate_target_envelopes": 0, "adequate_constants": 0}
                for task in audit.TASKS}
    for row in rows:
        task, name = row["task"], row["map"]
        spec = manifest["tasks"][task]
        mapping = next(item for item in spec["maps"] if item["map"] == name)
        relative = f"output/classes/{task}/{name}.json"
        path = ROOT / "records/classes" / task / (name + ".json")
        require(row["receipt_path"] == relative and row["receipt_sha256"] == sha(path),
                "Class receipt identity differs from the original result")
        record = read_json(path)
        features = np.load(ROOT / mapping["feature_path"], allow_pickle=False)
        require(features.dtype == np.float64 and features.shape == (172, mapping["dimension"])
                and np.isfinite(features).all(), "Stored feature dtype, shape, or finiteness differs")
        binding = {"input_manifest_sha256": result["input_manifest_sha256"],
                   "cohort_path": spec["cohort_path"], "cohort_sha256": sha(ROOT / spec["cohort_path"]),
                   "feature_path": mapping["feature_path"], "feature_sha256": mapping["feature_sha256"],
                   "map_payload_sha256": mapping["map_payload_sha256"],
                   "prior_model_seal_sha256": audit.PRIOR_HASHES["model_seal_sha256"]}
        expected = {"task": task, "map": name, "feature_id": mapping["feature_id"],
                    "class_id": mapping["class_id"], "epsilon": spec["epsilon"],
                    "input_binding": binding, "settings": protocol["numerics"]["settings"]}
        checked = audit.audit_diagnostic(record, features, cohorts[task]["y"], cohorts[task]["ids"],
                                         [mapping["heads"][recipe] for recipe in audit.RECIPES], expected)
        require(record["kappa"] == "2", "The sensitivity budget differs")
        rebuilt = {"task": task, "map": name, "family": mapping["family"], "scaling": mapping["scaling"],
                   "dimension": mapping["dimension"], "n": 172, "feature_id": mapping["feature_id"],
                   "class_id": mapping["class_id"], "epsilon": spec["epsilon"],
                   "epsilon_physical_mse": spec["epsilon_physical_mse"],
                   "physical_mse_multiplier": str(cohorts[task]["physical_mse_multiplier"]),
                   "class_risk": record["class_risk"], "range_floor_mean": record["range_floor"]["mean"],
                   "new_primal_mean": record["new_primal_mean"], "diagnosis": record["diagnosis"],
                   "gap_resolved": record["gap_resolved"],
                   "existing_eligible_head_count": record["existing_eligible_head_count"],
                   "new_finite_cohort_adequacy": record["new_finite_cohort_adequacy"],
                   "solver_status": record["numerical"]["status"],
                   "solver_iterations": record["numerical"]["iterations"],
                   "solver_fallback_reason": record["numerical"]["fallback_reason"],
                   "pair_count": record["distance"]["pair_count"],
                   "nonredundant_pair_count": len(record["distance"]["nonredundant_pair_indices"]),
                   "zero_distance_pair_count": record["distance"]["zero_distance_pair_count"],
                   "nonzero_dual_count": len(record["dual"]["signed_multipliers"]),
                   "selected_primal_source": record["selected_primal_source"], "timing": record["timing"],
                   "receipt_path": relative, "receipt_sha256": sha(path)}
        require(row == rebuilt, "Aggregate class result differs from its verified receipt")
        checked["receipt_sha256"] = sha(path)
        checks.append(checked)
        outcome = outcomes[task]
        outcome[{"ADEQUATE": "adequate_classes", "INADEQUATE": "inadequate_classes",
                 "UNRESOLVED": "unresolved_classes"}[checked["diagnosis"]]] += 1
        outcome["classes_with_both_heads_failing"] += int(all(h["verified_ineligible"] for h in record["inherited_heads"]))
        for candidate in record["primal_candidates"]:
            field = {"clipped_target_envelope": "adequate_target_envelopes", "clipped_mean_constant": "adequate_constants"}.get(candidate["source"])
            if field:
                outcome[field] += int(audit.q(candidate["mean"]) <= audit.q(record["epsilon"]))
        class_table.append({"target": task, "map": name, "dimension": mapping["dimension"],
                            "epsilon": record["epsilon"], "class_lower": str(checked["class_risk"]["lower"]),
                            "class_upper": str(checked["class_risk"]["upper"]),
                            "class_width": str(checked["class_risk"]["width"]), "diagnosis": checked["diagnosis"]})
        for head in record["inherited_heads"]:
            head_rows.append({"task": task, "map": name, "family": mapping["family"],
                              "scaling": mapping["scaling"], "epsilon": record["epsilon"],
                              "class_diagnosis": record["diagnosis"], "class_risk": record["class_risk"], **head})
            outcome["heads_with_positive_excess"] += int(head["positive_certified_excess"])
            head_table.append({"target": task, "map": name, "recipe": head["recipe"],
                               "risk_lower": head["decision_risk"]["lower"], "risk_upper": head["decision_risk"]["upper"],
                               "excess_lower": head["excess_risk"]["lower"], "excess_upper": head["excess_risk"]["upper"],
                               "verified_ineligible": head["verified_ineligible"],
                               "positive_certified_excess": head["positive_certified_excess"]})
        if len(checks) % 6 == 0:
            print(f"Verified {len(checks)}/54 classes", flush=True)
    require(head_rows == result["head_rows"], "Aggregate head/excess rows differ from verified receipts")
    expected_paths = {ROOT / "records/classes" / task / (name + ".json") for task, name in ordered}
    require(set((ROOT / "records/classes").rglob("*.json")) == expected_paths, "Unreported class receipt")
    totals = {"classes_verified": len(checks), "inherited_head_excess_intervals_verified": len(head_rows)}
    for field in ("distance_pairs_verified", "primal_candidates_verified", "primal_pair_constraints_verified",
                  "shortest_path_envelopes_verified", "signed_dual_edges_verified"):
        totals[field] = sum(check[field] for check in checks)
    for field in next(iter(outcomes.values())):
        totals[field] = sum(outcome[field] for outcome in outcomes.values())
    require(totals["adequate_classes"] == 54 and totals["classes_with_both_heads_failing"] == 39
            and totals["heads_with_positive_excess"] == 108 and totals["adequate_target_envelopes"] == 54
            and totals["adequate_constants"] == 0, "Displayed diagnostic findings differ")
    summary = {"schema": "class-diagnosis-numerical-replay-v1", "status": "PASS",
               "package_files_verified": verified_files, "original_result_sha256": sha(ROOT / "records/result.json"),
               "assurance": audit.ASSURANCE, "totals": totals, "by_target": outcomes,
               "maximum_class_risk_width": str(max(check["class_risk"]["width"] for check in checks)),
               "maximum_class_risk_width_decimal": format(float(max(check["class_risk"]["width"] for check in checks)), ".15g"),
               "scope": {"finite_cohort_only": True, "stored_binary64_features_treated_as_exact": True,
                         "original_targets_unclipped": True, "new_model_fits": 0, "optimizer_invocations": 0,
                         "new_forecasts": 0, "upstream_feature_error_enclosed": False,
                         "head_membership_and_risk_proofs": "Inherited from the bound public-screening audit; original head proof payloads are not recomputed here",
                         "head_excess_arithmetic": "Recomputed independently for all 108 inherited heads",
                         "historical_execution_attestation": False},
               "input_binding_checks": input_check, "protocol_semantics": protocol_check,
               "class_checks": checks}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(audit.canonical(summary), indent=2, sort_keys=True) + "\n")
    write_csv(output / "class_risk.csv", class_table)
    write_csv(output / "head_excess.csv", head_table)
    print(json.dumps({"status": "PASS", "totals": totals, "output": str(output)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the replay report and exact-value CSV tables")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output != ROOT and not output.is_relative_to(ROOT / "inputs")
            and not output.is_relative_to(ROOT / "records") and not output.is_relative_to(ROOT / "source")
            and not output.is_relative_to(ROOT / "protocol"), "Replay output would overwrite frozen package files")
    try:
        replay(output)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAIL", "error_type": type(exc).__name__, "message": str(exc)}
        (output / "summary.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
