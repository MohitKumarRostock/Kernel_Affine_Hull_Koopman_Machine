"""Artificial-only Step36 preflight: exact pair mathematics and online policy.

The examples in this file are generated here and do not open public numeric
data, cohort receipts, model outcomes, or any reserved assessment records.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import random

import finite_feature35 as ff
import triggered_policy36 as tp


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, KeyError, TypeError):
        return True
    raise AssertionError("Invalid artificial input was accepted")


def polygon_oracle(yi, yj, radius, interval):
    """Independent exact two-dimensional feasible-face enumeration."""
    low, high = map(F, interval)
    yi, yj, radius = map(F, (yi, yj, radius))
    lines = [(F(1), F(0), high), (F(-1), F(0), -low),
             (F(0), F(1), high), (F(0), F(-1), -low),
             (F(1), F(-1), radius), (F(-1), F(1), radius)]
    candidates = set()

    def add(a, b):
        if all(c * a + d * b <= e for c, d, e in lines):
            candidates.add((a, b))

    add(yi, yj)
    for a, b, c in lines:
        step = (a * yi + b * yj - c) / (a * a + b * b)
        add(yi - step * a, yj - step * b)
    for (a, b, c), (d, e, f) in combinations(lines, 2):
        determinant = a * e - b * d
        if determinant:
            add((c * e - b * f) / determinant, (a * f - c * d) / determinant)
    require(bool(candidates), "Artificial polygon unexpectedly empty")
    chosen = min(candidates, key=lambda v: ((v[0] - yi) ** 2 + (v[1] - yj) ** 2, v))
    return chosen, (chosen[0] - yi) ** 2 + (chosen[1] - yj) ** 2


def scalar_checks():
    count = 0
    targets = [F(-3), F(-1), F(-1, 4), F(0), F(1, 4), F(1, 2), F(1), F(5, 4), F(2), F(4)]
    for interval in [(F(0), F(1)), (F(-2), F(4)), (F(1, 3), F(7, 9))]:
        for yi, yj, radius in product(targets, targets, [F(0), F(1, 16), F(1, 4), F(1), F(2), F(7)]):
            result = tp.two_point_box_optimum(yi, yj, radius, output_interval=interval)
            values, sse = polygon_oracle(yi, yj, radius, interval)
            require(tuple(map(F, result["values"])) == values and F(result["sse"]) == sse,
                    "Closed form disagrees with independent polygon oracle")
            require(F(result["dual_sse"]) == sse, "Pair scalar dual equality failed")
            require(F(result["improvement_sse"]) >= 0, "Negative range-floor improvement")
            count += 1
    invalid = sum([
        rejects(tp.two_point_box_optimum, 0, 1, -1),
        rejects(tp.two_point_box_optimum, 0, 1, 0, output_interval=(1, 1)),
        rejects(tp.two_point_box_optimum, True, 1, 0),
    ])
    return {"independent_polygon_agreements": count, "invalid_scalar_inputs_rejected": invalid}


def payload_checks(fixtures):
    bounds = []
    for d in (1, 4, 24):
        bound = tp.head_cost_lower_bound("affine", feature_id="artificial:map", dimension=d,
                                         map_payload_bytes=103)
        fit = ff.make_affine_head([F(i - d, 3 * d) for i in range(d)], F(-11, 17),
                                  feature_id="artificial:map")
        checked = tp.check_realized_head_cost(bound, fit["head"])
        fixtures["payload_examples"].append({"bound": bound, "head": fit["head"], "expected": checked})
        require(checked["deployment_payload_bytes"] >= bound["deployment_payload_lower_bound_bytes"],
                "Affine byte lower bound failed")
        zeros = ff.make_affine_head([0] * d, 0, feature_id="artificial:map")["head"]
        require(tp.head_payload_bytes(zeros) == bound["head_payload_lower_bound_bytes"],
                "Zero affine template is not exact serialized minimum")
        bounds.append(bound)
    features, targets, ids = [[0, F(1, 3)], [1, F(2, 3)]], [F(1, 5), F(4, 5)], ["train-a", "train-b"]
    bound = tp.head_cost_lower_bound("extension32", feature_id="artificial:map", dimension=2,
        map_payload_bytes=103, anchor_features=features, anchor_ids=ids)
    fit = ff.fit_anchor_decoder(features, targets, feature_id="artificial:map", ids=ids,
                                max_updates=20, check_every=20)
    checked = tp.check_realized_head_cost(bound, fit["head"])
    fixtures["payload_examples"].append({"bound": bound, "head": fit["head"], "expected": checked})
    require(checked["head_payload_bytes"] == len((json.dumps(fit["head"], sort_keys=True,
        separators=(",", ":"), allow_nan=False) + "\n").encode()), "Canonical bytes disagree")
    require(checked["head_payload_bytes"] >= bound["head_payload_lower_bound_bytes"],
             "Extension byte lower bound failed")
    changed = dict(fit["head"])
    changed["anchor_ids"] = ["wrong-a", "train-b"]
    invalid = sum([
        rejects(tp.check_realized_head_cost, bound, changed),
        rejects(tp.head_cost_lower_bound, "affine", feature_id="x", dimension=0, map_payload_bytes=1),
        rejects(tp.head_cost_lower_bound, "affine", feature_id="x", dimension=1, map_payload_bytes=-1),
    ])
    return {"actual_head_byte_bounds_verified": len(bounds) + 1,
            "zero_affine_templates_attain_bound": len(bounds), "invalid_payload_inputs_rejected": invalid}


def audit_pair_receipt(receipt):
    z = [[F(value) for value in row] for row in receipt["features_exact"]]
    y = list(map(F, receipt["targets_exact"]))
    low, high = map(F, receipt["output_interval"])
    kap = F(receipt["kappa"])
    floor = sum(((min(high, max(low, value)) - value) ** 2 for value in y), F(0))
    require(F(receipt["range_floor_sse"]) == floor, "Range floor differs")
    matched = set()
    improvement = F(0)
    for pair in receipt["selected_pairs"]:
        i, j = pair["indices"]
        require(i < j and i not in matched and j not in matched, "Pair matching is not disjoint")
        matched.update((i, j))
        squared = sum(((a - b) ** 2 for a, b in zip(z[i], z[j])), F(0))
        dl, du = F(pair["distance"]["lower"]), F(pair["distance"]["upper"])
        require(dl >= 0 and dl * dl <= squared <= du * du,
                "Stored-coordinate distance enclosure failed")
        require(F(pair["radius_upper"]) == kap * du, "Pair radius differs")
        _, sse = polygon_oracle(y[i], y[j], kap * du, (low, high))
        gain = sse - (min(high, max(low, y[i])) - y[i]) ** 2 - (min(high, max(low, y[j])) - y[j]) ** 2
        require(F(pair["optimum"]["sse"]) == sse and F(pair["optimum"]["improvement_sse"]) == gain,
                "Pair optimum or singleton subtraction differs")
        improvement += gain
    require(F(receipt["lower_sse"]) == floor + improvement and
            F(receipt["lower_mean"]) == (floor + improvement) / len(y),
            "Whole-cohort floor and disjoint-pair aggregation failed")
    require(receipt["matching"]["selected_indices"] == [p["indices"] for p in receipt["selected_pairs"]],
            "Matching and exact witnesses differ")


def pair_checks(fixtures):
    receipts = []
    unmatched = tp.pair_class_lower([[0], [0], [0]], [-1, 2, 5], feature_id="artificial:odd")
    audit_pair_receipt(unmatched)
    require(F(unmatched["lower_mean"]) == F(41, 6), "Unmatched range floor was lost or double-counted")
    receipts.append(unmatched)
    no_edges = tp.pair_class_lower([[0], [100], [200], [300]], [-2, -1, 2, 3],
                                    feature_id="artificial:far")
    audit_pair_receipt(no_edges)
    require(no_edges["work"]["exact_selected_pair_distance_evaluations"] == 0 and
             F(no_edges["lower_mean"]) == F(5, 2), "No-positive-edge fallback differs from common range floor")
    receipts.append(no_edges)
    singleton = tp.pair_class_lower([[F(2, 3)]], [2], feature_id="artificial:single")
    audit_pair_receipt(singleton)
    require(F(singleton["lower_mean"]) == 1, "Singleton range floor failed")
    receipts.append(singleton)
    capped = tp.pair_class_lower([[100 * (i // 2), 0] for i in range(192)],
        [i % 2 for i in range(192)], feature_id="artificial:cap")
    audit_pair_receipt(capped)
    require(len(capped["selected_pairs"]) == 64 and
             capped["work"]["approximate_distance_evaluations"] == 192 * 191 // 2 and
             F(capped["lower_mean"]) == F(1, 6), "Frozen selected-pair work cap failed")
    receipts.append(capped)
    tied = tp.pair_class_lower([[0], [0], [0], [0]], [0, 1, 0, 1],
        ids=["d", "c", "b", "a"], feature_id="artificial:ties")
    repeated = tp.pair_class_lower([[0], [0], [0], [0]], [0, 1, 0, 1],
        ids=["d", "c", "b", "a"], feature_id="artificial:ties")
    require(tied == repeated and tied["matching"]["selected_indices"] == [[2, 3], [0, 1]],
             "Lexical deterministic edge ties failed")
    audit_pair_receipt(tied)
    receipts.append(tied)
    rng = random.Random(36001)
    for index in range(24):
        n = rng.randrange(2, 13)
        z = [[F(rng.randrange(-4, 5), 8), F(rng.randrange(-4, 5), 8)] for _ in range(n)]
        y = [F(rng.randrange(-4, 13), 8) for _ in range(n)]
        receipt = tp.pair_class_lower(z, y, feature_id="artificial:random:" + str(index))
        audit_pair_receipt(receipt)
        # Every constant in the range belongs to the class and gives an upper witness.
        constant = min(F(1), max(F(0), sum(y, F(0)) / n))
        require(F(receipt["lower_mean"]) <= sum(((value - constant) ** 2 for value in y), F(0)) / n,
                 "Pair lower exceeds a feasible constant upper witness")
        receipts.append(receipt)
    invalid = sum([
        rejects(tp.pair_class_lower, [[0], [1]], [0], feature_id="x"),
        rejects(tp.pair_class_lower, [[0], [1]], [0, 1], feature_id="x", ids=["same", "same"]),
        rejects(tp.pair_class_lower, [[0], [1]], [0, 1], feature_id="x", max_pairs=65),
        rejects(tp.pair_class_lower, [[0], [1]], [0, 1], feature_id="x", bits=79),
    ])
    fixtures["pair_receipts"] = receipts
    return {"independently_recomputed_pair_receipts": len(receipts),
        "deterministic_ties_verified": True, "selected_pair_cap_attained": 64,
        "invalid_pair_inputs_rejected": invalid,
        "artificial_receipt_digests": [ff.digest(r) for r in receipts]}


def artificial_bounds(name, map_bytes=100):
    fid = "artificial:" + name
    affine = tp.head_cost_lower_bound("affine", feature_id=fid, dimension=1,
                                      map_payload_bytes=map_bytes)
    extension = tp.head_cost_lower_bound("extension32", feature_id=fid, dimension=1,
        map_payload_bytes=map_bytes, anchor_features=[[0], [1]], anchor_ids=["a", "b"])
    return affine, extension


def artificial_row(name, recipe, cost, upper, lower=None):
    upper = F(upper)
    lower = upper if lower is None else F(lower)
    return {"candidate_id": name + "::" + recipe, "map": name, "recipe": recipe,
        "head_id": "artificial-head:" + name + ":" + recipe,
        "deployment_payload_bytes": cost,
        "decision_risk": {"lower": str(lower), "upper": str(upper), "width": str(upper - lower)}}


def policy_checks():
    a, e = artificial_bounds("a")
    # Adequacy prevents certification; it is not by itself a cost-pruning rule.
    state = tp.PolicyState("1/10", mode="triggered")
    row = artificial_row("a", "affine", 10000, "1/20")
    require(state.affine_decision("a", a)["action"] == "FIT", "First cheap head not offered")
    state.observe(row)
    decision = state.extension_decision("a", e, row)
    require(decision["action"] == "FIT" and decision["reason"] == "AFFINE_ALREADY_VERIFIED_ADEQUATE",
            "Affine adequacy incorrectly dropped a potentially cheaper extension")
    smaller = artificial_row("a", "extension32", 9000, "1/25")
    state.observe(smaller)
    require(state.selection()["candidate_id"] == smaller["candidate_id"], "Cheaper extension was not retained")
    # Equality of byte bound is not sufficient to remove a potential tie winner.
    tie_state = tp.PolicyState("1/10")
    tied_cost = a["deployment_payload_lower_bound_bytes"]
    tie_state.observe(artificial_row("z", "affine", tied_cost, "1/20"))
    require(tie_state.affine_decision("a", a)["action"] == "FIT", "Equal-cost head was incorrectly pruned")
    tie_state.observe(artificial_row("a", "affine", tied_cost, "1/20"))
    require(tie_state.selection()["candidate_id"] == "a::affine", "Lexical final tie failed")
    pruned_state = tp.PolicyState("1/10")
    pruned_state.observe(artificial_row("z", "affine", tied_cost - 1, "1/20"))
    require(pruned_state.affine_decision("a", a)["action"] == "COST_PRUNE", "Strict cost pruning failed")
    # Pair lower strictly exceeds epsilon; exact equality must fall through.
    cert = tp.pair_class_lower([[0], [0]], [0, 1], feature_id=a["feature_id"])
    excludes = tp.PolicyState("1/10", mode="triggered")
    bad = artificial_row("a", "affine", 10000, "1/2")
    excludes.observe(bad)
    require(excludes.extension_decision("a", e, bad)["action"] == "CERTIFY", "Eligible trigger did not fire")
    require(excludes.observe_certificate("a", cert)["action"] == "CLASS_PRUNE", "Valid class exclusion failed")
    equality = tp.PolicyState("1/4", mode="triggered")
    equality.observe(bad)
    require(equality.extension_decision("a", e, bad)["action"] == "CERTIFY", "Equality example did not trigger")
    require(equality.observe_certificate("a", cert)["action"] == "FIT", "Non-strict class lower incorrectly pruned")
    # Budget exhaustion falls back to fitting, and no unobserved affine is a trigger premise.
    budget = tp.PolicyState("1/10", mode="triggered", max_certificate_calls=1)
    budget.observe(bad)
    budget.extension_decision("a", e, bad)
    budget.observe_certificate("a", cert)
    a2, e2 = artificial_bounds("b")
    bad2 = artificial_row("b", "affine", 10000, "1/2")
    budget.observe(bad2)
    require(budget.extension_decision("b", e2, bad2)["action"] == "FIT", "Exhausted budget did not fall back to fit")
    empty = tp.PolicyState("1/10", mode="triggered")
    require(empty.extension_decision("b", e2)["action"] == "FIT", "Unobserved affine caused a certificate call")
    require(empty.certificate_calls == 0, "Unobserved affine consumed the pair budget")
    # Range floor is common to both algorithms and is not attributed to pair certificates.
    floor = tp.range_floor([-1, 2])
    for mode in ("ordinary", "triggered"):
        ranged = tp.PolicyState("1/10", mode=mode)
        ranged.set_range_floor(floor)
        require(ranged.range_floor_prunes(), "Common range-only skip failed")
        require(ranged.extension_decision("b", e2)["action"] == "RANGE_PRUNE" and
                 ranged.certificate_calls == 0, "Common range skip was attributed to pair screening")
    require(tp.map_order({"b": 10, "a": 10, "c": 5}) == ["c", "a", "b"],
             "Prefit encoder-byte ordering failed")
    invalid = sum([
        rejects(tp.PolicyState, "-1"),
        rejects(tp.PolicyState, "1", max_certificate_calls=7),
        rejects(excludes.observe_certificate, "a", cert),
    ])
    return {"affine_adequacy_alone_does_not_prune_extension": True,
        "strict_byte_bound_and_full_ties_verified": True,
        "strict_class_exclusion_and_equality_fallback_verified": True,
        "budget_exhaustion_and_missing_affine_fall_back_to_fit": True,
        "common_range_skip_attributed_equally": True,
        "invalid_policy_inputs_rejected": invalid}


def panel_equivalence_checks():
    """Finite synthetic panels with actual valid pair bounds and cost lower bounds."""
    rng = random.Random(36002)
    panels, exclusions, cost_skips, budget_fallbacks, abstentions = 0, 0, 0, 0, 0
    for panel_index in range(120):
        epsilon = F(rng.randrange(0, 11), 40)
        entries = []
        for i in range(10):
            name = f"m{i:02d}"
            map_bytes = rng.randrange(0, 8) * 100
            bound_a, bound_e = artificial_bounds(name, map_bytes)
            # At identical features with y=(0,1), every class head has mean loss >=1/4.
            identical = bool(rng.randrange(2))
            features = [[0], [0 if identical else 1]]
            certificate = tp.pair_class_lower(features, [0, 1], feature_id=bound_a["feature_id"])
            valid_floor = F(certificate["lower_mean"])
            rows = []
            for recipe, bound in [("affine", bound_a), ("extension32", bound_e)]:
                upper = valid_floor + F(rng.randrange(0, 21), 40)
                cost = bound["deployment_payload_lower_bound_bytes"] + rng.randrange(0, 8) * 100
                rows.append(artificial_row(name, recipe, cost, upper))
            entries.append((name, map_bytes, bound_a, bound_e, rows, certificate))
        entries.sort(key=lambda item: (item[1], item[0]))
        all_rows = [row for item in entries for row in item[4]]
        eligible = [row for row in all_rows if F(row["decision_risk"]["upper"]) <= epsilon]
        exhaustive = min(eligible, key=tp.candidate_key, default=None)
        abstentions += exhaustive is None
        observed_ids = {}
        for mode in ("ordinary", "triggered"):
            state = tp.PolicyState(epsilon, mode=mode)
            state.set_range_floor(tp.range_floor([0, 1]))
            observed_ids[mode] = set()
            for name, _, bound_a, bound_e, rows, cert in entries:
                row_a = None
                first = state.affine_decision(name, bound_a)
                if first["action"] == "FIT":
                    row_a = rows[0]
                    state.observe(row_a)
                    observed_ids[mode].add(row_a["candidate_id"])
                else:
                    cost_skips += 1
                second = state.extension_decision(name, bound_e, row_a)
                if second["action"] == "CERTIFY":
                    second = state.observe_certificate(name, cert)
                    exclusions += second["action"] == "CLASS_PRUNE"
                elif second["action"] == "COST_PRUNE":
                    cost_skips += 1
                elif second["reason"] == "PAIR_CERTIFICATE_CALL_BUDGET_EXHAUSTED":
                    budget_fallbacks += 1
                if second["action"] == "FIT":
                    state.observe(rows[1])
                    observed_ids[mode].add(rows[1]["candidate_id"])
            selected = state.selection()
            require((None if selected is None else selected["candidate_id"]) ==
                    (None if exhaustive is None else exhaustive["candidate_id"]),
                    "Online policy changed the exhaustive fixed-tolerance choice")
            require(state.certificate_calls <= 6 and state.exact_pair_distances <= 384,
                    "Global exact-distance work cap failed")
        require(observed_ids["triggered"] <= observed_ids["ordinary"],
                "Triggered policy evaluated a head omitted by ordinary policy")
        panels += 1
    require(exclusions > 0 and cost_skips > 0 and budget_fallbacks > 0 and abstentions > 0,
             "Artificial panels missed a required nontrivial policy branch")
    return {"random_artificial_panels": panels, "exhaustive_ordinary_triggered_choice_agreements": panels,
        "triggered_fit_set_subset_ordinary": True, "class_exclusions_exercised": exclusions,
        "cost_prunes_exercised": cost_skips, "budget_fallbacks_exercised": budget_fallbacks,
        "exhaustive_abstentions_exercised": abstentions}


def main():
    fixtures = {"schema": "step36-artificial-policy-witnesses-v1", "artificial_only": True,
                "public_numeric_values_read": False, "payload_examples": [], "pair_receipts": []}
    checks = {"scalar": scalar_checks(), "payload": payload_checks(fixtures), "pairs": pair_checks(fixtures),
              "policy": policy_checks(), "finite_panel_equivalence": panel_equivalence_checks()}
    files = [Path(__file__), Path(tp.__file__), Path(ff.__file__)]
    witness_path = ROOT / "tests" / "artificial_policy_witnesses36.json"
    witness_path.parent.mkdir(parents=True, exist_ok=True)
    witness_path.write_text(json.dumps(fixtures, sort_keys=True, indent=2) + "\n")
    receipt = {"schema": "step36-artificial-policy-preflight-v1", "status": "PASS",
        "public_numeric_values_read": False, "reserved_patient_data_read": False,
        "actual_scientific_models_fit": False, "artificial_training_heads_fit": True,
        "witness_path": str(witness_path.relative_to(ROOT)),
        "witness_sha256": hashlib.sha256(witness_path.read_bytes()).hexdigest(),
        "checks": checks, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "claim_scope": "Artificial arithmetic and decision-logic verification only; no empirical utility result."}
    target = ROOT / "tests" / "artificial_policy_preflight36.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "path": str(target), "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
