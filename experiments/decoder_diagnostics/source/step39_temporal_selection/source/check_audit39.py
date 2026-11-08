"""Artificial-only independent and adversarial checks for Step39 mathematics."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import time

import audit39 as a

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    fixture_path = ROOT / "tests/learner_fixture39.json"
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
    for token, fit in x["fits"].items():
        a.audit_fit_wrapper(fit, z, y, ids)
    passed("three_regularized_objective_wrappers_and_core_duals")
    geometry = a.a38.audit_distances(x["distances"], z, x["query_z"], ids, x["query_ids"])
    for name, head in x["heads"].items():
        fit = None if name == "affine" else x["fits"]["l0" if name == "envelope_all" else name.split("_")[1]]
        a.audit_head39(head, fit, z, y, ids)
        prediction = x["predictions"][name]
        a.audit_prediction39(prediction, head, x["query_z"], x["query_ids"], x["distances"], geometry)
        a.a38.audit_loss(prediction["predictions"], x["original_loss_targets"], x["losses"][name])
        if name.startswith("qp_"):
            expanded = x["optimizer_predictions"][name]
            a.audit_optimizer_expansion(expanded, prediction, head, fit)
            a.a38.audit_loss(expanded["predictions"], x["original_loss_targets"], x["optimizer_losses"][name])
    passed("all_weights_controls_predictions_and_original_unclipped_losses")
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
        a.audit_head39(head, fit, analytic["training_z"], analytic["training_y"], analytic["training_ids"])
        a.audit_prediction39(prediction, head, analytic["query_z"], analytic["query_ids"], analytic["distances"], geometry2)
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
    rejected("reject_nonconvex_extension_weight", lambda: a.audit_head39(bad_head, x["fits"]["l0"], z, y, ids))
    bad_head = deepcopy(x["heads"]["qp_l0_w05"]); bad_head["clipping_order"] = "before_mixture"
    rejected("reject_wrong_extension_clipping_order", lambda: a.audit_head39(bad_head, x["fits"]["l0"], z, y, ids))
    bad_head = deepcopy(x["heads"]["qp_l0_w05"]); bad_head["anchor_values"][0] = "1"
    rejected("reject_unbound_training_anchor_values", lambda: a.audit_head39(bad_head, x["fits"]["l0"], z, y, ids))
    bad_pred = deepcopy(analytic["predictions"]["w05"])
    bad_pred["predictions"][0] = {"lower": "1/2", "upper": "1/2", "width": "0"}
    rejected("reject_clip_before_mixture_prediction",
             lambda: a.audit_prediction39(bad_pred, analytic["heads"]["w05"], analytic["query_z"], analytic["query_ids"], analytic["distances"], geometry2))
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
    true_gate = {"reference_positive_risk": True, "at_least_five_percent_mse_improvement": True,
                 "no_worse_than_selected_ordinary": True, "numerical_criterion_pass": True,
                 "relative_gain_status": "DEFINED", "population_noninferiority_claim": False}
    a.audit_practical_gate(point(19), point(20), point(19), true_gate)
    passed("five_percent_mse_equality_boundary")
    rejected("reject_zero_denominator_as_relative_gain_success",
             lambda: a.audit_practical_gate(point(0), point(0), point(0), true_gate))
    rejected("reject_primary_success_when_ordinary_baseline_fails",
             lambda: a.audit_practical_gate(point(19), point(20), point(18), true_gate))
    passed("independent_byte_csv_scanner_quotes_and_commas",
           a.csv_cells(b'"a,b",c,"d""e"\r\n') == [b"a,b", b"c", b'd"e'])

    result = {"schema": "step39-independent-artificial-audit-v1", "status": "PASS",
              "artificial_only": True, "scientific_data_accessed": False,
              "fixture_sha256": a.sha(fixture_path), "check_count": len(checks), "checks": checks,
              "source_sha256": {"source/audit39.py": a.sha(ROOT / "source/audit39.py"),
                                "source/check_audit39.py": a.sha(Path(__file__)),
                                "source/audit38.py": a.sha(ROOT / "source/audit38.py"),
                                "source/audit_math37.py": a.sha(ROOT / "source/audit_math37.py"),
                                "source/independent_math35.py": a.sha(ROOT / "source/independent_math35.py")},
              "wall_seconds": time.perf_counter() - start}
    a.write(ROOT / "tests/artificial_audit39.json", result)
    print(json.dumps({"status": "PASS", "checks": len(checks), "scientific_data_accessed": False}))


if __name__ == "__main__":
    main()
