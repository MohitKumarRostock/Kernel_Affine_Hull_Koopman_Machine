"""Artificial-only independent and adversarial checks for Step40 mathematics and mixed decisions."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import time

import audit40 as a

ROOT = Path(__file__).resolve().parents[1]


def main():
    if (ROOT / "protocol_seal40.json").exists():
        raise RuntimeError("Artificial audit evidence cannot change after the protocol freeze")
    start = time.perf_counter()
    fixture_path = ROOT / "tests/learner_fixture40.json"
    x = a.read(fixture_path)
    a.require(x["schema"] == "step39-artificial-learner-fixture-v1" and x["scientific_data_accessed"] is False,
              "Only the explicitly artificial fixture is permitted")
    checks = {}

    def passed(name, condition=True):
        a.require(condition, name)
        checks[name] = True

    def rejected(name, operation):
        try:
            operation()
        except (AssertionError, ValueError, KeyError, TypeError, ArithmeticError):
            checks[name] = True
            return
        raise AssertionError("Mutation escaped independent checking: " + name)

    z, y, ids = x["training_z"], x["training_y"], x["training_ids"]
    a.audit_fit_wrapper(x["fits"]["l0"], z, y, ids)
    passed("current_unregularized_objective_wrapper_and_core_dual")
    geometry = a.a38.audit_distances(x["distances"], z, x["query_z"], ids, x["query_ids"])
    for name, head in x["heads"].items():
        if name not in ("affine", "qp_l0_w05"):
            continue
        fit = None if name == "affine" else x["fits"]["l0" if name == "envelope_all" else name.split("_")[1]]
        a.audit_head40(head, fit, z, y, ids)
        prediction = x["predictions"][name]
        a.audit_prediction40(prediction, head, x["query_z"], x["query_ids"], x["distances"], geometry)
        a.a38.audit_loss(prediction["predictions"], x["original_loss_targets"], x["losses"][name])
        if name.startswith("qp_"):
            expanded = x["optimizer_predictions"][name]
            a.audit_optimizer_expansion(expanded, prediction, head, fit)
            a.a38.audit_loss(expanded["predictions"], x["original_loss_targets"], x["optimizer_losses"][name])
    passed("current_midpoint_qp_and_affine_predictions_original_unclipped_losses")
    analytic = x["analytic"]
    fit = analytic["fit"]
    a.audit_fit_wrapper(fit, analytic["training_z"], analytic["training_y"], analytic["training_ids"])
    effective = list(map(F, fit["effective_targets"]))
    optimum = [max(F(0), min(F(1), value)) for value in effective]
    # All pair distances are at least 3 and kappa=2: box constraints alone
    # determine the optimum in this deliberately separable fixture.
    passed("independent_closed_form_box_projection_optimum",
           list(map(F, analytic["exact_optimum"])) == optimum and optimum == [F(1, 24), F(5, 12), F(1)])
    center = sum(map(F, analytic["training_y"]), F(0)) / 3
    objective = sum(((v - F(t)) ** 2 + (v - center) ** 2 for v, t in zip(optimum, analytic["training_y"])), F(0)) / 3
    passed("independent_regularized_objective_with_out_of_range_targets",
           objective == F(523, 864) == F(analytic["exact_objective"]))
    jl, ju = a.box(fit["regularized_objective"])
    passed("verified_objective_bracket_contains_closed_form_optimum", jl <= objective <= ju)
    candidate = next(r for r in fit["core_fit"]["primal_candidates"] if r["source"] == fit["selected_qp_source"])
    distance_sq = sum(((F(v) - star) ** 2 for v, star in zip(candidate["values"], optimum)), F(0))
    passed("strong_convexity_optimizer_distance_bound", distance_sq <= F(fit["optimizer_prediction_bound"]["squared_upper"]))
    geometry2 = a.a38.audit_distances(analytic["distances"], analytic["training_z"], analytic["query_z"],
                                     analytic["training_ids"], analytic["query_ids"])
    for name, head in analytic["heads"].items():
        prediction = analytic["predictions"][name]
        a.audit_head40(head, fit, analytic["training_z"], analytic["training_y"], analytic["training_ids"])
        a.audit_prediction40(prediction, head, analytic["query_z"], analytic["query_ids"], analytic["distances"], geometry2)
        expanded = analytic["optimizer_predictions"][name]
        a.audit_optimizer_expansion(expanded, prediction, head, fit)
        low, high = a.box(expanded["predictions"][0])
        passed("exact_optimum_prediction_contained_" + name, low <= F(analytic["exact_optimal_predictions"][name]) <= high)
    passed("clip_after_mixture_is_not_clip_before_mixture",
           F(analytic["exact_optimal_predictions"]["w05"]) == F(11, 48) != F(1, 2))
    for case in x["sqrt_cases"]:
        a.audit_sqrt(case["receipt"], case["radicand"])
    passed("zero_exact_irrational_and_tiny_square_root_bounds")

    bad = deepcopy(x["fits"]["l1"]); bad["regularization_center"] = "0"
    rejected("reject_wrong_training_mean", lambda: a.audit_fit_wrapper(bad, z, y, ids))
    bad = deepcopy(x["fits"]["l025"]); bad["objective_additive_constant"] = "0"
    rejected("reject_missing_objective_constant", lambda: a.audit_fit_wrapper(bad, z, y, ids))
    bad = deepcopy(x["fits"]["l1"]); bad["effective_targets"][0] = "1"
    rejected("reject_changed_pseudo_targets", lambda: a.audit_fit_wrapper(bad, z, y, ids))
    bad = deepcopy(x["fits"]["l1"]); bad["original_target_risks"][bad["selected_qp_source"]]["original_target_mse"] = "0"
    rejected("reject_regularized_risk_as_original_mse", lambda: a.audit_fit_wrapper(bad, z, y, ids))
    bad = deepcopy(x["fits"]["l1"]); bad["core_fit"]["dual"]["distance_endpoint"] = "lower"; bad["core_fit_digest"] = a.digest(bad["core_fit"])
    rejected("reject_wrong_dual_distance_endpoint", lambda: a.audit_fit_wrapper(bad, z, y, ids))
    bad = deepcopy(x["fits"]["l1"]); bad["optimizer_prediction_bound"]["squared_upper"] = "0"
    rejected("reject_forgotten_strong_convexity_factor", lambda: a.audit_fit_wrapper(bad, z, y, ids))
    bad = deepcopy(x["fits"]["l1"]); bad["optimizer_prediction_bound"]["uniform_upper"] = "0"
    rejected("reject_understated_optimizer_discrepancy", lambda: a.audit_fit_wrapper(bad, z, y, ids))
    bad_head = deepcopy(x["heads"]["qp_l0_w05"]); bad_head["extension_weight"] = "3/2"
    rejected("reject_nonconvex_extension_weight", lambda: a.audit_head40(bad_head, x["fits"]["l0"], z, y, ids))
    bad_head = deepcopy(x["heads"]["qp_l0_w05"]); bad_head["clipping_order"] = "before_mixture"
    rejected("reject_wrong_extension_clipping_order", lambda: a.audit_head40(bad_head, x["fits"]["l0"], z, y, ids))
    bad_head = deepcopy(x["heads"]["qp_l0_w05"]); bad_head["anchor_values"][0] = "1"
    rejected("reject_unbound_training_anchor_values", lambda: a.audit_head40(bad_head, x["fits"]["l0"], z, y, ids))
    bad_pred = deepcopy(analytic["predictions"]["w05"])
    bad_pred["predictions"][0] = {"lower": "1/2", "upper": "1/2", "width": "0"}
    rejected("reject_clip_before_mixture_prediction",
             lambda: a.audit_prediction40(bad_pred, analytic["heads"]["w05"], analytic["query_z"], analytic["query_ids"], analytic["distances"], geometry2))
    bad_expansion = deepcopy(analytic["optimizer_predictions"]["w05"])
    bad_expansion["predictions"] = deepcopy(analytic["predictions"]["w05"]["predictions"])
    rejected("reject_omitted_optimizer_expansion",
             lambda: a.audit_optimizer_expansion(bad_expansion, analytic["predictions"]["w05"], analytic["heads"]["w05"], analytic["fit"]))
    bad_sqrt = deepcopy(x["sqrt_cases"][2]["receipt"])
    bad_sqrt["floor_scaled"] = str(int(bad_sqrt["floor_scaled"]) + 1)
    rejected("reject_rounded_up_root_floor", lambda: a.audit_sqrt(bad_sqrt, x["sqrt_cases"][2]["radicand"]))

    point = lambda value: {"lower": str(F(value)), "upper": str(F(value)), "width": "0"}
    passed("unequal_fold_sizes_pool_sse_not_mean_of_fold_means",
           a.pooled_box([{"n": 1, "risk": point(1)}, {"n": 3, "risk": point(9)}]) == point(7))
    def final_row(cid, value, *, interval=None):
        return {"candidate_id": cid, "head_id": "head:" + cid, "head_sha256": "sha:" + cid,
                "test_predictions_digest": "pred:" + cid, "physical_test_risk": point(value) if interval is None else interval}
    new, reference = final_row("affine_a", 19), final_row("qp_a", 20)
    ordinary, affine = final_row("ordinary", 19), deepcopy(new)
    true_gate = {"reference_positive_risk": True, "at_least_five_percent_mse_improvement": True,
                 "no_worse_than_selected_ordinary": True, "no_worse_than_selected_affine": True,
                 "ordinary_comparison_reason": "INTERVAL_NONINFERIORITY",
                 "affine_comparison_reason": "SAME_FINAL_CANDIDATE_PAYLOAD_EXACT_EQUALITY",
                 "joint_same_as_affine": True, "joint_superiority_over_affine_from_identity": False,
                 "numerical_criterion_pass": True, "relative_gain_status": "DEFINED", "population_noninferiority_claim": False}
    a.audit_practical_gate(new, reference, ordinary, affine, true_gate)
    passed("five_percent_mse_equality_and_identical_affine_boundary")
    rejected("reject_zero_denominator_as_relative_gain_success",
             lambda: a.audit_practical_gate(final_row("z", 0), final_row("q", 0), final_row("o", 0), final_row("a", 0), true_gate))
    rejected("reject_primary_success_when_ordinary_baseline_fails",
             lambda: a.audit_practical_gate(new, reference, final_row("ordinary", 18), affine, true_gate))
    rejected("reject_primary_success_when_affine_guard_fails",
             lambda: a.audit_practical_gate(new, reference, ordinary, final_row("affine_b", 18), true_gate))
    for key in ("candidate_id", "head_id", "head_sha256", "test_predictions_digest"):
        altered = deepcopy(affine); altered[key] += "-different"
        rejected("reject_false_payload_identity_" + key,
                 lambda altered=altered: a.audit_practical_gate(new, reference, ordinary, altered, true_gate))
    wide_new = final_row("same", 0, interval={"lower": "18", "upper": "19", "width": "1"})
    a.audit_practical_gate(wide_new, reference, ordinary, deepcopy(wide_new), true_gate)
    passed("same_bound_final_payload_proves_equality_even_with_nonzero_width")
    passed("independent_byte_csv_scanner_quotes_and_commas",
           a.csv_cells(b'"a,b",c,"d""e"\r\n') == [b"a,b", b"c", b'd"e'])
    passed("independent_byte_csv_scanner_semicolon",
           a.csv_cells(b'16/12/2006;17:24:00;?\n', ";") == [b"16/12/2006", b"17:24:00", b"?"])
    rejected("reject_unescaped_internal_quote", lambda: a.csv_cells(b'"a"b",c\n'))

    household_header = b"Date;Time;Global_active_power;a;b;c;d;e;f\n"
    household_lines = [b"31/12/2008;23:00:00;NEVER_PARSE;0;0;0;0;0;0\n",
                       b"01/01/2009;00:00:00; ? ;0;0;0;0;0;0\n",
                       b"01/01/2009;01:00:00;ALSO_NOT_NUMERIC;0;0;0;0;0;0\n"]
    household_projection = a.independent_calendar_projection(
        "household_power2009", {"household": household_header + b"".join(household_lines)})
    passed("independent_household_metadata_never_parses_measurement_tokens",
           household_projection["timestamp_rows"] == 3 and
           household_projection["hourly_target_status_counts"] == {"declared_missing": 1, "not_declared_missing": 2} and
           household_projection["rows"][1]["declared_target_missing"] is True and
           household_projection["rows"][0]["records"][0]["byte_start"] == len(household_header) and
           household_projection["target_values_decoded_or_parsed"] == 0)
    jena_header = ("Date Time,p (mbar),T (degC)," + ",".join("unit" + str(i) for i in range(18)) + ",W/m²\n").encode("latin-1")
    jena_row = lambda stamp: (stamp + ",0,UNPARSED," + ",".join("0" for _ in range(19)) + "\n").encode("ascii")
    jena_projection = a.independent_calendar_projection("jena_temperature2016", {
        "jena2016a": jena_header + jena_row("01.01.2016 01:00:00"),
        "jena2016b": jena_header + jena_row("01.07.2016 01:00:00"),
    })
    passed("independent_jena_latin1_headers_multiple_archives_timestamp_only",
           jena_projection["timestamp_rows"] == 2 and jena_projection["headers"][-1] == "W/m²" and
           jena_projection["target_missing_status_inspected"] is False and
           jena_projection["rows"][1]["records"][0]["source_id"] == "jena2016b" and
           jena_projection["target_values_decoded_or_parsed"] == 0)
    rejected("reject_jena_calendar_cadence_violation",
             lambda: a.source_timestamp("jena_temperature2016", [b"01.01.2016 01:05:00"]))

    decisions_path = ROOT / "tests/artificial_decisions40.json"
    decisions = a.read(decisions_path)
    a.require(decisions["status"] == "PASS" and decisions["artificial_only"] is True and
              decisions["scientific_data_values_accessed"] == 0 and decisions["scientific_model_fits"] == 0,
              "Decision fixture was not explicitly artificial")
    for label, case in decisions["fixtures"].items():
        if "certificate" in case:
            certificate = case["certificate"]
            a.audit_decision_certificate(case["rows"], certificate["operational_selected_id"], certificate, case["expected_maps"])
        else:
            a.audit_paired_heads(case["rows"], case["paired_result"], names=case["expected_maps"],
                operational_key="physical_test_risk", optimizer_key="physical_optimizer_risk", cohort_role="test")
        passed("independent_mixed_decision_fixture_" + label)
    case = decisions["fixtures"]["strict_affine_winner"]
    valid = case["certificate"]
    mutations = {
        "false_affine_optimizer_claim": lambda r: r.update(affine_training_optimizer_certified=True),
        "false_compute_savings_claim": lambda r: r.update(computational_savings_claim=True),
        "false_native_transform_enclosure": lambda r: r.update(native_transform_error_enclosed=True),
        "future_performance_claim": lambda r: r.update(future_response_coverage_claim=True),
        "wrong_selected_margin": lambda r: r["selected_winner"].update(strict_margin_lower="0"),
        "wrong_survivor_count": lambda r: r["candidate_exclusion"].update(survivor_count=2),
        "wrong_exclusion_margin": lambda r: r["candidate_exclusion"]["excluded"][0].update(strict_exclusion_margin="999"),
        "wrong_family_minimum": lambda r: r["head_family_resolution"]["family_minimum_risks"].update(affine=point(0)),
        "wrong_pair_sign": lambda r: r["same_map_comparisons"][0].update(mixed_difference=point(-1)),
        "false_changed_selection": lambda r: r.update(operational_selection_changed=True),
        "wrong_input_digest": lambda r: r.update(source_comparator_rows_digest="0" * 64),
    }
    for label, mutation in mutations.items():
        bad = deepcopy(valid); mutation(bad)
        rejected("reject_mixed_certificate_" + label,
                 lambda bad=bad: a.audit_decision_certificate(case["rows"], valid["operational_selected_id"], bad, case["expected_maps"]))
    malformed = deepcopy(case["rows"])
    malformed[0]["pooled_optimizer_risk"] = point(1)
    rejected("reject_attached_affine_optimizer_interval", lambda: a.audit_decision_certificate(
        malformed, valid["operational_selected_id"], valid, case["expected_maps"]))
    malformed = deepcopy(case["rows"])
    malformed[1]["extension_weight"] = "0"
    rejected("reject_nonmidpoint_qp_comparator", lambda: a.audit_decision_certificate(
        malformed, valid["operational_selected_id"], valid, case["expected_maps"]))
    malformed = deepcopy(case["rows"])
    malformed[0]["pooled_physical_risk"]["lower"] = 4.0
    rejected("reject_float_decision_endpoint", lambda: a.audit_decision_certificate(
        malformed, valid["operational_selected_id"], valid, case["expected_maps"]))

    result = {"schema": "step40-independent-artificial-audit-v1", "status": "PASS",
              "artificial_only": True, "scientific_data_accessed": False,
              "fixture_path": "tests/learner_fixture40.json", "fixture_sha256": a.sha(fixture_path),
              "fixture_provenance_sha256": a.sha(ROOT / "tests/inherited_learner_fixture40.json"),
              "decision_fixture_sha256": a.sha(decisions_path), "check_count": len(checks), "checks": checks,
              "source_sha256": {"source/audit40.py": a.sha(ROOT / "source/audit40.py"),
                                "source/check_audit40.py": a.sha(Path(__file__)),
                                "source/audit38.py": a.sha(ROOT / "source/audit38.py"),
                                "source/audit_math37.py": a.sha(ROOT / "source/audit_math37.py"),
                                "source/independent_math35.py": a.sha(ROOT / "source/independent_math35.py")},
              "wall_seconds": time.perf_counter() - start}
    a.write(ROOT / "tests/artificial_audit40.json", result)
    print(json.dumps({"status": "PASS", "checks": len(checks), "scientific_data_accessed": False}))


if __name__ == "__main__":
    main()
