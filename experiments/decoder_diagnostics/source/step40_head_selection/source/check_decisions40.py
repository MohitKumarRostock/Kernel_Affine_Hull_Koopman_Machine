"""Artificial rational fixtures for Step40's mixed-comparator decisions.

No dataset, feature transform, numerical solver, or scientific fitting is used.
The expected decisions below follow from hand-specified exact intervals.
"""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import decisions40 as d


ROOT = Path(__file__).resolve().parents[1]


def interval(lo, hi=None):
    lo = F(lo)
    hi = lo if hi is None else F(hi)
    return {"lower": str(lo), "upper": str(hi), "width": str(hi - lo)}


def pair(name, affine, qp_operational, qp_lower, qp_upper):
    return [
        {"candidate_id": name + "::affine", "map": name, "recipe": "affine",
         "ridge_lambda": None, "extension_weight": None,
         "pooled_physical_risk": interval(affine), "pooled_optimizer_risk": None},
        {"candidate_id": name + "::qp_l0_w05", "map": name, "recipe": "qp_regularized",
         "ridge_lambda": "0", "extension_weight": "1/2",
         "pooled_physical_risk": interval(qp_operational),
         "pooled_optimizer_risk": interval(qp_lower, qp_upper)},
    ]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reject(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Malformed or out-of-scope artificial input was accepted")


def main():
    started = time.perf_counter()
    if (ROOT / "protocol_seal40.json").exists():
        raise RuntimeError("Artificial fixture evidence cannot be replaced after the protocol freeze")
    fixtures, checks, rejected = {}, [], []
    names = ["map_a", "map_b"]
    rows = pair("map_a", 4, 6, 5, 7) + pair("map_b", 2, 4, 3, 5)
    original = deepcopy(rows)
    certificate = d.mixed_decision_certificate(rows, "map_b::affine", expected_maps=names)
    fixtures["strict_affine_winner"] = {"rows": rows, "expected_maps": names, "certificate": certificate}
    require(rows == original, "Decision helper mutated its source risk rows")
    exclusion = certificate["candidate_exclusion"]
    require(exclusion["survivor_ids"] == ["map_b::affine"] and exclusion["excluded_count"] == 3,
            "Known strict affine winner did not exclude the other three candidates")
    require(exclusion["excluded_fraction"] == "3/4" and exclusion["fitting_work_avoided"] is False,
            "Exclusion count or compute-scope statement is wrong")
    require(certificate["selected_winner"]["strict_margin_lower"] == "1" and
            certificate["selected_winner"]["status"] == "CERTIFIED_UNIQUE_MIXED_COMPARATOR_WINNER",
            "Known unique winner was not certified")
    require(certificate["head_family_resolution"]["difference"] == interval(1, 3),
            "Family-minimum difference enclosure is incorrect")
    require(all(row["mixed_difference"] == interval(1, 3) and row["status"] == "AFFINE_STRICTLY_LOWER_RISK"
                for row in certificate["same_map_comparisons"]),
            "Same-map positive differences should favor affine")
    require(certificate["affine_training_optimizer_certified"] is False and
            certificate["computational_savings_claim"] is False and certificate["test_labels_used"] is False,
            "Scope claims drifted")
    checks.extend([
        "strict_exclusion_and_selected_winner_follow_exact_endpoints",
        "family_minimum_bounds_and_same_map_positive_differences_are_correct",
        "input_rows_are_not_mutated",
        "affine_optimizer_and_computational_savings_claims_are_false",
    ])

    equal_rows = pair("map_a", 4, 6, 5, 7) + pair("map_b", 2, "5/2", 2, 3)
    equality = d.mixed_decision_certificate(equal_rows, "map_b::affine", expected_maps=names)
    fixtures["boundary_equality"] = {"rows": equal_rows, "expected_maps": names, "certificate": equality}
    require(equality["candidate_exclusion"]["survivor_ids"] == ["map_b::affine", "map_b::qp_l0_w05"] and
            equality["selected_winner"]["strict_margin_lower"] == "0" and
            equality["selected_winner"]["status"] == "UNRESOLVED",
            "Equality at the exclusion threshold must remain unresolved")
    require(equality["same_map_comparisons"][1]["status"] == "NO_STRICT_SEPARATION",
            "Zero lower difference is not a strict advantage")
    checks.append("equality_is_retained_and_never_promoted_to_strict_separation")

    tied_rows = pair("map_a", 2, 4, 3, 5) + pair("map_b", 2, 5, 4, 6)
    tied = d.mixed_decision_certificate(tied_rows, "map_a::affine", expected_maps=names)
    fixtures["family_resolved_configuration_tied"] = {"rows": tied_rows, "expected_maps": names, "certificate": tied}
    require(tied["selected_winner"]["status"] == "UNRESOLVED" and
            tied["head_family_resolution"]["status"] == "AFFINE_FAMILY_STRICTLY_LOWER_RISK" and
            tied["candidate_exclusion"]["survivor_count"] == 2,
            "Family resolution must not require a unique within-family configuration")
    checks.append("head_family_can_resolve_when_configuration_tie_remains")

    qp_rows = pair("map_a", 8, 3, 2, 4) + pair("map_b", 10, 5, 5, 6)
    qp = d.mixed_decision_certificate(qp_rows, "map_a::qp_l0_w05", expected_maps=names)
    fixtures["strict_qp_winner"] = {"rows": qp_rows, "expected_maps": names, "certificate": qp}
    require(qp["selected_winner"]["status"] == "CERTIFIED_UNIQUE_MIXED_COMPARATOR_WINNER" and
            qp["head_family_resolution"]["difference"] == interval(-6, -4) and
            qp["head_family_resolution"]["status"] == "QP_FAMILY_STRICTLY_LOWER_RISK",
            "Known QP family advantage was not enclosed correctly")
    require(all(row["status"] == "QP_STRICTLY_LOWER_RISK" for row in qp["same_map_comparisons"]),
            "Negative same-map differences should favor QP")
    checks.append("negative_same_map_and_family_differences_favor_qp")

    wrong_selected = d.mixed_decision_certificate(rows, "map_a::affine", expected_maps=names)
    fixtures["declared_selection_differs_from_unique_survivor"] = {
        "rows": rows, "expected_maps": names, "certificate": wrong_selected,
        "intentional_invalid_upstream_selection": True,
    }
    require(wrong_selected["candidate_exclusion"]["unique_survivor_id"] == "map_b::affine" and
            wrong_selected["candidate_exclusion"]["unique_survivor_agrees_with_operational_selection"] is False and
            wrong_selected["selected_winner"]["status"] == "UNRESOLVED",
            "Another candidate's unique survivor status must not certify the supplied selection")
    checks.append("unique_survivor_of_another_candidate_does_not_certify_selected_id")

    tiny = F(1, 10 ** 100)
    tiny_rows = pair("map_tiny", 1, 1 + tiny, 1 + tiny, 1 + 2 * tiny)
    precision = d.mixed_decision_certificate(tiny_rows, "map_tiny::affine", expected_maps=["map_tiny"])
    fixtures["exact_sub_float_strictness"] = {"rows": tiny_rows, "expected_maps": ["map_tiny"], "certificate": precision}
    require(F(precision["selected_winner"]["strict_margin_lower"]) == tiny and
            precision["selected_winner"]["status"] == "CERTIFIED_UNIQUE_MIXED_COMPARATOR_WINNER",
            "Strict rational margins must not be rounded to binary64 zero")
    checks.append("strictness_survives_margins_smaller_than_binary64_resolution")

    wide_rows = pair("map_a", 4, 6, 0, 20) + pair("map_b", 2, 4, 0, 20)
    wide = d.mixed_decision_certificate(wide_rows, "map_b::affine", expected_maps=names)
    fixtures["valid_wide_intervals"] = {"rows": wide_rows, "expected_maps": names, "certificate": wide}
    require(wide["selected_winner"]["status"] == "UNRESOLVED" and
            wide["candidate_exclusion"]["survivor_count"] == 3 and
            wide["head_family_resolution"]["status"] == "NO_STRICT_SEPARATION",
            "Finite wide intervals must be retained rather than forced to resolve")
    checks.append("valid_wide_intervals_return_unresolved_without_refitting")

    permuted = d.mixed_decision_certificate(list(reversed(rows)), "map_b::affine", expected_maps=names)
    require(permuted == certificate, "Source row order changed the fixed-roster decision")
    changed = deepcopy(rows)
    changed[0]["pooled_physical_risk"] = interval("9/2")
    changed_result = d.mixed_decision_certificate(changed, "map_b::affine", expected_maps=names)
    require(changed_result["source_comparator_rows_digest"] != certificate["source_comparator_rows_digest"],
            "Exact comparator-input digest did not bind changed risk")
    checks.extend(["declared_map_order_makes_candidate_input_permutations_equivalent",
                   "comparator_projection_digest_binds_exact_risks"])

    test_rows = deepcopy(rows)
    for row in test_rows:
        row["physical_test_risk"] = row.pop("pooled_physical_risk")
        row["physical_optimizer_risk"] = row.pop("pooled_optimizer_risk")
    test_pairs = d.paired_head_comparisons(test_rows, expected_maps=names,
                operational_risk_key="physical_test_risk", qp_optimizer_risk_key="physical_optimizer_risk",
                cohort_role="test")
    fixtures["descriptive_test_pairs"] = {"rows": test_rows, "expected_maps": names, "paired_result": test_pairs}
    require(test_pairs["pairs"] == certificate["same_map_comparisons"] and
            test_pairs["test_reselection_allowed"] is False and test_pairs["selection_performed_by_this_function"] is False,
            "Predeclared test contrasts must remain descriptive")
    checks.append("same_map_test_reporting_cannot_reselect_the_deployed_head")

    def mutation(name, modify):
        altered = deepcopy(rows)
        modify(altered)
        reject(lambda: d.mixed_decision_certificate(altered, "map_b::affine", expected_maps=names))
        rejected.append(name)

    mutation("duplicate_candidate_id", lambda r: r[0].update(candidate_id=r[1]["candidate_id"]))
    mutation("outside_map", lambda r: r[0].update(map="unfrozen_map"))
    mutation("ordinary_candidate_in_mixed_collection", lambda r: r[0].update(recipe="baseline"))
    mutation("duplicate_family_for_map", lambda r: r[1].update(recipe="affine"))
    mutation("nonzero_qp_regularization", lambda r: r[1].update(ridge_lambda="1/4"))
    mutation("nonmidpoint_qp_extension", lambda r: r[1].update(extension_weight="0"))
    mutation("missing_qp_optimizer_interval", lambda r: r[1].update(pooled_optimizer_risk=None))
    mutation("qp_interval_does_not_enclose_operational", lambda r: r[1].update(pooled_optimizer_risk=interval(7, 8)))
    mutation("unsupported_affine_optimizer_interval", lambda r: r[0].update(pooled_optimizer_risk=interval(4)))
    mutation("unsupported_affine_qp_regularization", lambda r: r[0].update(ridge_lambda="0"))
    mutation("unsupported_affine_extension_weight", lambda r: r[0].update(extension_weight="1/2"))
    mutation("affine_is_not_a_point", lambda r: r[0].update(pooled_physical_risk=interval(3, 4)))
    mutation("negative_mse", lambda r: r[0].update(pooled_physical_risk=interval(-1)))
    mutation("reversed_interval", lambda r: r[1].update(pooled_optimizer_risk={"lower": "7", "upper": "5", "width": "-2"}))
    mutation("inconsistent_width", lambda r: r[1].update(pooled_optimizer_risk={"lower": "5", "upper": "7", "width": "3"}))
    mutation("float_endpoint", lambda r: r[0].update(pooled_physical_risk={"lower": 4.0, "upper": 4.0, "width": "0"}))
    mutation("boolean_endpoint", lambda r: r[0].update(pooled_physical_risk={"lower": True, "upper": True, "width": "0"}))
    mutation("missing_operational_interval", lambda r: r[0].pop("pooled_physical_risk"))
    reject(lambda: d.mixed_decision_certificate(rows[:-1], "map_b::affine", expected_maps=names))
    rejected.append("missing_candidate")
    reject(lambda: d.mixed_decision_certificate(rows, "missing::affine", expected_maps=names))
    rejected.append("selected_id_outside_collection")
    reject(lambda: d.mixed_decision_certificate(rows, "map_b::affine", expected_maps=["map_a", "map_a"]))
    rejected.append("duplicated_expected_map")
    reject(lambda: d.mixed_decision_certificate(rows, "map_b::affine", expected_maps=[]))
    rejected.append("empty_expected_map_roster")
    reject(lambda: d.paired_head_comparisons(rows, expected_maps=names, cohort_role="unseen"))
    rejected.append("undeclared_cohort_role")
    checks.append("malformed_scope_roster_and_interval_mutations_are_rejected")

    source_paths = [ROOT / "source/decisions40.py", Path(__file__)]
    result = {
        "schema": "step40-artificial-decision-checks-v1",
        "status": "PASS",
        "artificial_only": True,
        "scientific_data_accessed": False,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        "checks": checks,
        "check_count": len(checks),
        "rejected_mutations": rejected,
        "rejected_mutation_count": len(rejected),
        "fixtures": fixtures,
        "scientific_data_values_accessed": 0,
        "scientific_model_fits": 0,
        "seconds": time.perf_counter() - started,
    }
    path = ROOT / "tests/artificial_decisions40.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.write_bytes(payload)
    print(json.dumps({"status": "PASS", "checks": len(checks), "rejected_mutations": len(rejected),
                      "artifact": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
