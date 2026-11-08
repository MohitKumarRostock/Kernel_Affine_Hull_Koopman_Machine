"""Artificial-only checks of the new Step 39 objective and prediction bounds."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import learners39 as m


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.canonical_bytes(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expect_reject(action):
    try:
        action()
    except (ValueError, ArithmeticError):
        return True
    raise AssertionError("Invalid artificial input was accepted")


def main():
    started = time.perf_counter()
    checks = []
    # Synthetic training coordinates include a duplicate to exercise exact
    # feasibility where the paired fitted anchor values must be equal.
    z = np.asarray([[i / 32, ((i * 7) % 19) / 64, ((i * 11) % 23) / 32]
                    for i in range(40)], dtype=np.float64)
    z[-1] = z[0]
    y = [F((i * 13) % 37, 36) for i in range(40)]
    ids = ["artificial_train_" + str(i) for i in range(len(y))]
    feature_id = "artificial_feature39"
    lambda_tokens = {"0": "l0", "1/4": "l025", "1": "l1"}
    weight_tokens = {"0": "w0", "1/2": "w05", "1": "w1"}
    fits = {}
    heads = {}
    for lam in m.RIDGE_GRID:
        token = lambda_tokens[lam]
        fit = m.fit_regularized(z, y, ids=ids, feature_id=feature_id,
                                ridge_lambda=lam, input_binding={"artificial": True})
        fits[token] = fit
        assert fit["core_fit"]["target_digest"] == fit["effective_target_digest"]
        assert fit["original_target_digest"] == m.content_digest(y)
        assert F(fit["regularized_objective"]["lower"]) <= F(fit["regularized_objective"]["upper"])
        bound = fit["optimizer_prediction_bound"]
        assert F(bound["uniform_upper"]) ** 2 >= F(bound["squared_upper"])
        assert F(bound["squared_upper"]) == len(y) * F(fit["core_fit"]["class_risk"]["width"])
        for weight in m.EXTENSION_GRID:
            name = "qp_" + token + "_" + weight_tokens[weight]
            heads[name] = m.make_extension(
                fit, z, anchor_indices=list(range(len(y))), recipe="qp_regularized",
                extension_weight=weight, fit_path="artificial/" + token + ".json",
                fit_sha256=m.digest(fit),
            )
    checks.extend([
        "all_three_regularized_objectives_have_exact_original_to_effective_conversion",
        "original_target_risks_are_separate_from_regularized_lower_bounds",
        "duplicate_anchor_coordinates_remain_feasible_for_all_lambda",
        "optimization_to_prediction_bound_accounts_for_objective_strong_convexity",
    ])
    assert fits["l0"]["effective_targets"] == list(map(str, y))
    assert F(fits["l0"]["objective_additive_constant"]) == 0
    assert fits["l0"]["regularized_objective"] == fits["l0"]["core_fit"]["class_risk"]
    checks.append("lambda_zero_reduces_exactly_to_inherited_objective")

    heads["envelope_all"] = m.make_extension(
        fits["l0"], z, anchor_indices=list(range(len(y))), recipe="envelope_all",
        extension_weight="1/2", fit_path="artificial/l0.json", fit_sha256=m.digest(fits["l0"]),
    )
    affine = m.fit_affine(z, y, feature_id=feature_id, training_binding={"artificial": True})
    heads["affine"] = affine["head"]
    queries = np.asarray([z[0], z[18], [.1, .2, .3], [-1., -1., -1.], [2., 2., 2.]], dtype=np.float64)
    query_ids = ["artificial_query_" + str(i) for i in range(len(queries))]
    distance = m.query_distances(z, queries, anchor_ids=ids, query_ids=query_ids, feature_id=feature_id)
    predictions = {name: m.evaluate_head(head, queries, ids=query_ids, distance=distance)
                   for name, head in heads.items()}
    expanded = {}
    for name, head in heads.items():
        prediction = predictions[name]
        for interval in prediction["predictions"]:
            assert 0 <= F(interval["lower"]) <= F(interval["upper"]) <= 1
            assert F(interval["width"]) <= F(2, 1 << 80)
        if head.get("schema") == m.HEAD_SCHEMA:
            value = F(head["anchor_values"][0])
            interval = prediction["predictions"][0]
            assert F(interval["lower"]) <= value <= F(interval["upper"])
        if head.get("recipe") == "qp_regularized":
            fit = fits[lambda_tokens[head["ridge_lambda"]]]
            expanded[name] = m.expand_optimizer_predictions(prediction, head, fit)
            for fixed, widened in zip(prediction["predictions"], expanded[name]["predictions"]):
                assert 0 <= F(widened["lower"]) <= F(fixed["lower"]) <= F(fixed["upper"]) <= F(widened["upper"]) <= 1
    checks.extend([
        "all_nine_qp_weights_and_both_controls_use_shared_label_free_query_geometry",
        "all_extension_weights_interpolate_feasible_training_anchors",
        "weighted_fixed_prediction_enclosure_width_respects_80_bit_bound",
        "optimization_expansion_contains_fixed_head_interval_and_respects_output_range",
        "affine_control_uses_unchanged_inherited_prediction_path",
    ])

    outside = [F(-1, 3), F(3, 2), F(1, 2), F(2), F(-1)]
    losses = {name: m.score(prediction["predictions"], outside) for name, prediction in predictions.items()}
    expanded_losses = {name: m.score(prediction["predictions"], outside) for name, prediction in expanded.items()}
    assert all(F(loss["lower"]) > 0 for loss in losses.values())
    for name, loss in expanded_losses.items():
        assert F(loss["lower"]) <= F(losses[name]["lower"]) <= F(losses[name]["upper"]) <= F(loss["upper"])
    checks.append("original_outside_range_labels_are_not_clipped_in_fixed_or_expanded_losses")

    # Analytic optimum: all feature separations exceed the maximum possible
    # value difference, so the only active restrictions are scalar [0,1] boxes.
    # This checks the certificate against a known solution, independently of
    # the optimizer's returned status or its dual coordinates.
    az = np.asarray([[0.], [3.], [6.]], dtype=np.float64)
    ay = [F(-1, 2), F(1, 4), F(2)]
    aids = ["analytic_" + str(i) for i in range(3)]
    analytic_fit = m.fit_regularized(az, ay, ids=aids, feature_id="analytic_feature39",
                                   ridge_lambda="1", input_binding={"artificial": True})
    exact_optimum = [m.clip(F(t)) for t in analytic_fit["effective_targets"]]
    center = sum(ay, F(0)) / len(ay)
    exact_objective = sum(((v - yv) ** 2 + (v - center) ** 2
                           for v, yv in zip(exact_optimum, ay)), F(0)) / len(ay)
    bracket = analytic_fit["regularized_objective"]
    assert F(bracket["lower"]) <= exact_objective <= F(bracket["upper"])
    chosen = next(c for c in analytic_fit["core_fit"]["primal_candidates"]
                  if c["source"] == analytic_fit["selected_qp_source"])
    fitted = list(map(F, chosen["values"]))
    exact_squared_discrepancy = sum(((a - b) ** 2 for a, b in zip(fitted, exact_optimum)), F(0))
    assert exact_squared_discrepancy <= F(analytic_fit["optimizer_prediction_bound"]["squared_upper"])
    aq = np.asarray([[1.5]], dtype=np.float64)
    aqids = ["analytic_query"]
    adist = m.query_distances(az, aq, anchor_ids=aids, query_ids=aqids, feature_id="analytic_feature39")
    analytical_heads = {}
    analytical_predictions = {}
    analytical_expanded = {}
    exact_optimal_predictions = {}
    for weight in m.EXTENSION_GRID:
        head = m.make_extension(analytic_fit, az, anchor_indices=list(range(3)), recipe="qp_regularized",
                                extension_weight=weight, fit_path="artificial/analytic.json",
                                fit_sha256=m.digest(analytic_fit))
        prediction = m.evaluate_head(head, aq, ids=aqids, distance=adist)
        expansion = m.expand_optimizer_predictions(prediction, head, analytic_fit)
        lower = max(v - 2 * abs(F(3, 2) - F(3 * i)) for i, v in enumerate(exact_optimum))
        upper = min(v + 2 * abs(F(3, 2) - F(3 * i)) for i, v in enumerate(exact_optimum))
        value = m.clip((1 - F(weight)) * lower + F(weight) * upper)
        interval = expansion["predictions"][0]
        assert F(interval["lower"]) <= value <= F(interval["upper"])
        token = weight_tokens[weight]
        analytical_heads[token] = head
        analytical_predictions[token] = prediction
        analytical_expanded[token] = expansion
        exact_optimal_predictions[token] = value
    checks.extend([
        "known_box_constrained_optimum_lies_inside_regularized_objective_bracket",
        "known_optimizer_anchor_discrepancy_obeys_exact_bound",
        "known_exact_optimum_predictions_lie_inside_all_expanded_weighted_intervals",
    ])

    # Midpoint of unclipped cones for a single anchor is its anchor value at
    # every query. Averaging the separately clipped cones would give 1/2 here,
    # which would silently define a different and incorrect implementation.
    sz = np.asarray([[0.]], dtype=np.float64)
    sfit = m.fit_regularized(sz, [F(3, 4)], ids=["single"], feature_id="single_feature39",
                             ridge_lambda="0", input_binding={"artificial": True})
    shead = m.make_extension(sfit, sz, anchor_indices=[0], recipe="qp_regularized", extension_weight="1/2",
                             fit_path="artificial/single.json", fit_sha256=m.digest(sfit))
    sq = np.asarray([[1.]], dtype=np.float64)
    sdistance = m.query_distances(sz, sq, anchor_ids=["single"], query_ids=["far_single"], feature_id="single_feature39")
    spred = m.evaluate_head(shead, sq, ids=["far_single"], distance=sdistance)
    assert spred["predictions"][0] == m.box(F(3, 4), F(3, 4))
    checks.append("raw_envelopes_are_mixed_before_clipping")

    sqrt_cases = [F(0), F(1), F(2), F(1, 3), F(1, 10 ** 30)]
    roots = []
    for value in sqrt_cases:
        receipt = m.sqrt_enclosure(value)
        lower = F(receipt["interval"]["lower"])
        upper = F(receipt["interval"]["upper"])
        assert lower * lower <= value <= upper * upper
        assert upper - lower <= F(1, 1 << 80)
        roots.append({"radicand": value, "receipt": receipt})
    expect_reject(lambda: m.sqrt_enclosure(-1))
    checks.append("dyadic_square_root_bounds_cover_zero_exact_irrational_and_tiny_cases")

    forbidden = deepcopy(fits["l0"])
    forbidden["role"] = "assessment_diagnostic"
    forbidden["head_promotion_allowed"] = False
    expect_reject(lambda: m.make_extension(forbidden, z, anchor_indices=list(range(len(y))),
                  recipe="qp_regularized", extension_weight="1/2", fit_path="forbidden", fit_sha256=m.digest(forbidden)))
    expect_reject(lambda: m.fit_regularized(z, y, ids=ids, feature_id=feature_id, ridge_lambda="-1"))
    expect_reject(lambda: m.make_extension(fits["l0"], z, anchor_indices=list(range(len(y))), recipe="qp_regularized",
                  extension_weight="2", fit_path="invalid", fit_sha256=m.digest(fits["l0"])))
    expect_reject(lambda: m.make_extension(fits["l1"], z, anchor_indices=list(range(len(y))), recipe="envelope_all",
                  extension_weight="1/2", fit_path="invalid", fit_sha256=m.digest(fits["l1"])))
    expect_reject(lambda: m.make_extension(fits["l0"], z, anchor_indices=list(range(len(y))), recipe="qp_regularized",
                  extension_weight="1/2", fit_path="invalid", fit_sha256="0" * 64))
    expect_reject(lambda: m.expand_optimizer_predictions(predictions["envelope_all"], heads["envelope_all"], fits["l0"]))
    expect_reject(lambda: m.expand_optimizer_predictions(predictions["qp_l0_w0"], heads["qp_l0_w05"], fits["l0"]))
    checks.extend([
        "assessment_witness_cannot_be_promoted",
        "negative_regularization_and_out_of_range_extension_weights_rejected",
        "envelope_control_restriction_and_fit_identity_enforced",
        "optimizer_expansion_rejects_envelope_or_substituted_head",
    ])

    fixture = {
        "schema": "step39-artificial-learner-fixture-v1",
        "scientific_data_accessed": False,
        "training_z": z.tolist(), "training_y": y, "training_ids": ids,
        "fits": fits, "affine_fit": affine, "heads": heads,
        "query_z": queries.tolist(), "query_ids": query_ids, "distances": distance,
        "predictions": predictions, "optimizer_predictions": expanded,
        "original_loss_targets": outside, "losses": losses, "optimizer_losses": expanded_losses,
        "analytic": {
            "training_z": az.tolist(), "training_y": ay, "training_ids": aids,
            "fit": analytic_fit, "exact_optimum": exact_optimum, "exact_objective": exact_objective,
            "query_z": aq.tolist(), "query_ids": aqids, "distances": adist,
            "heads": analytical_heads, "predictions": analytical_predictions,
            "optimizer_predictions": analytical_expanded, "exact_optimal_predictions": exact_optimal_predictions,
        },
        "single_anchor": {"fit": sfit, "head": shead, "distances": sdistance, "prediction": spred},
        "sqrt_cases": roots,
    }
    fixture_path = ROOT / "tests/learner_fixture39.json"
    write(fixture_path, fixture)
    sources = ["source/" + name for name in (
        "check_learners39.py", "learners39.py", "learners38.py", "full_graph37.py", "finite_feature35.py",
    )]
    receipt = {
        "schema": "step39-artificial-producer-check-v1", "status": "PASS",
        "checks": checks, "check_count": len(checks), "scientific_data_accessed": False,
        "artificial_only": True, "source_sha256": {name: sha(ROOT / name) for name in sources},
        "fixture_path": "tests/learner_fixture39.json", "fixture_sha256": sha(fixture_path),
        "wall_seconds": time.perf_counter() - started,
    }
    write(ROOT / "tests/artificial_learners39.json", receipt)
    print(json.dumps({"status": "PASS", "checks": len(checks), "seconds": receipt["wall_seconds"],
                      "fixture_sha256": receipt["fixture_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
