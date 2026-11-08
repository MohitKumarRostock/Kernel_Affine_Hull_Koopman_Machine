"""Independent Step37 exact full-graph certificate and provenance verification.

Scientific producers are not imported. Numerical optimization status is never a
proof premise. Primal feasibility and signed weak-dual values are reconstructed
from exact stored coordinates, original targets and the retained witnesses.
Preflight and selfcheck modes never parse scientific numerical arrays.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
from math import isfinite, lcm
from pathlib import Path
import sys
import time
import traceback

import independent_math35 as old

ROOT = Path(__file__).resolve().parents[1]
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
INDEPENDENT_MATH_SHA256 = "3f440f427614ac230f4a09ac062d5bfc18eaa47babdaaff1c3c9b04b2c330308"
TASKS = ("casual", "registered", "cnt")
RECIPES = ("affine", "extension32")
PRIMARY_SETTINGS = {"max_iter":20000,"eps_abs":1e-8,"eps_rel":1e-8,
    "polishing":True,"polish_refine_iter":3,"check_termination":25,
    "scaling":10,"rho":0.1,"sigma":1e-6,"alpha":1.6,
    "adaptive_rho":True,"adaptive_rho_interval":50,"adaptive_rho_tolerance":5.0,
    "warm_starting":False,"scaled_termination":False,"time_limit":30.0,"verbose":False}
PRIOR_HASHES = {
    "result_sha256": "d52cc6ba7412ae507df473b3d4a4da9216e661470ac268c7ad69f1ce7d7d8471",
    "protocol_sha256": "4377b08b9653ea36d46afa77b1957ae4952bba83463272286e7fa9f4e572450c",
    "protocol_seal_sha256": "5e095bdd88140b8e80ac3b918661de209aeef7a9377118715cdd8ebe1635074a",
    "model_seal_sha256": "df0a1a36a80469f465f86cadd97a90af4f3b3060ed8b9256b41c161158010c9e",
    "postrun_audit_sha256": "be75862c90d9031d9ebf1bccc1374f34fe53e81815cb6bec236deace5bc23757",
}


class AuditError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def q(value):
    require(not isinstance(value, bool), "Boolean used as an exact rational")
    return F.from_float(value) if isinstance(value, float) else F(value)


def canonical(value):
    return old.canon(value)


def digest(value):
    return old.digest(value)


def digest_newline(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False)+"\n").encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return old.read_json(path)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(canonical(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False) + "\n")


def box(record, expected=None):
    low, high = q(record["lower"]), q(record["upper"])
    require(low <= high, "Reversed risk interval")
    if "width" in record:
        require(q(record["width"]) == high-low, "Incorrect exact interval width")
    if expected is not None:
        require((low, high) == expected, "Risk interval differs from exact reconstruction")
    return low, high


def clip(value, low, high):
    return min(high, max(low, value))


def exact_matrix(values):
    require(isinstance(values, (list, tuple)) and len(values) > 0,
            "Expected a nonempty exact feature matrix")
    dimension = len(values[0])
    require(dimension > 0 and all(len(row) == dimension for row in values),
            "Inconsistent feature dimension")
    return [[q(value) for value in row] for row in values]


def verify_sqrt_floor(squared_distance, floor, bits):
    """Check integer floor inequalities, independently of producer sqrt code."""
    require(type(floor) is int and floor >= 0 and type(bits) is int and 1 <= bits <= 512,
            "Invalid compact square-root floor or precision")
    value = q(squared_distance)
    require(value >= 0, "Negative squared feature distance")
    scale = 1 << bits
    numerator = value.numerator*scale*scale
    denominator = value.denominator
    require(floor*floor*denominator <= numerator < (floor+1)*(floor+1)*denominator,
            "Retained integer is not the exact scaled square-root floor")
    lower = F(floor, scale)
    upper = lower if floor*floor*denominator == numerator else F(floor+1, scale)
    return lower, upper


def complete_distances(features, ids, sqrt_floors, bits=80):
    """Independently bind every compact lexicographic pair to stored coordinates."""
    z = exact_matrix(features)
    n = len(z)
    require(isinstance(ids, list) and len(ids) == n and len(set(ids)) == n and
            all(isinstance(name, str) and name for name in ids),
            "Invalid exact cohort identifier roster")
    pairs = list(combinations(range(n), 2))
    require(len(sqrt_floors) == len(pairs), "Incomplete complete-graph distance roster")
    # Reconstruct one exact denominator, then use only integer differences.
    # This independently certifies the stored float rationals without any
    # floating distance calculation or the producer's isqrt implementation.
    denominator = lcm(*(value.denominator for row in z for value in row))
    integer_z = [[value.numerator*(denominator//value.denominator) for value in row] for row in z]
    rows, lookup, exact_zero_pairs, conservative_zero_pairs = [], {}, 0, 0
    for index, ((i, j), floor) in enumerate(zip(pairs, sqrt_floors)):
        squared_distance = F(sum((a-b)**2 for a,b in zip(integer_z[i],integer_z[j])), denominator**2)
        lower, upper = verify_sqrt_floor(squared_distance, floor, bits)
        row = {"indices": (i, j), "squared": squared_distance,
               "lower": lower, "upper": upper, "sqrt_floor": floor}
        rows.append(row); lookup[i, j] = row
        exact_zero_pairs += squared_distance == 0
        conservative_zero_pairs += lower == 0
    groups = {}
    for i, row in enumerate(z):
        groups.setdefault(tuple(row), []).append(i)
    return {"features": z, "ids": ids, "n": n, "dimension": len(z[0]),
            "pairs": rows, "lookup": lookup, "pair_count": len(rows), "bits": bits,
            "exact_zero_pairs": exact_zero_pairs,
            "conservative_zero_pairs": conservative_zero_pairs,
            "duplicate_feature_groups": [indices for indices in groups.values() if len(indices) > 1],
            "unique_feature_rows": len(groups)}


def original_target_floor(targets, output_interval=("0", "1")):
    y = list(map(q, targets))
    low, high = map(q, output_interval)
    require(y and low < high, "Invalid original targets or output interval")
    projected = [clip(value, low, high) for value in y]
    sse = sum(((v-t)**2 for v, t in zip(projected, y)), F(0))
    return {"values": projected, "sse": sse, "mean": sse/len(y),
            "outside_count": sum(not low <= value <= high for value in y)}


def verify_primal(values, targets, graph, kappa="2", output_interval=("0", "1")):
    """Certify a global Lipschitz/range extension exists for the finite witness."""
    y, v = list(map(q, targets)), list(map(q, values))
    low, high = map(q, output_interval)
    kap = q(kappa)
    n = graph["n"]
    require(kap >= 0 and low < high and len(v) == len(y) == n and
            all(low <= value <= high for value in v),
            "Primal witness length, range or sensitivity differs")
    value_denominator = lcm(*(value.denominator for value in v))
    integers = [value.numerator*(value_denominator//value.denominator) for value in v]
    radius_denominator = (1 << graph["bits"])*kap.denominator
    common = lcm(value_denominator, radius_denominator)
    value_factor, radius_factor = common//value_denominator, common//radius_denominator
    minimum_slack = None
    for edge in graph["pairs"]:
        i, j = edge["indices"]
        difference = abs(integers[i]-integers[j])
        slack_integer = kap.numerator*edge["sqrt_floor"]*radius_factor-difference*value_factor
        require(slack_integer >= 0, "Primal violates a conservative full-graph constraint")
        # Independent direct check of the actual stored-coordinate premise.
        squared = edge["squared"]
        require(difference*difference*kap.denominator**2*squared.denominator <=
                kap.numerator**2*squared.numerator*value_denominator**2,
                "Primal violates exact squared Lipschitz feasibility")
        minimum_slack = slack_integer if minimum_slack is None else min(minimum_slack, slack_integer)
    for indices in graph["duplicate_feature_groups"]:
        require(len({v[i] for i in indices}) == 1,
                "Exact duplicate features received different witness values")
    sse = sum(((value-target)**2 for value, target in zip(v, y)), F(0))
    return {"values": v, "sse": sse, "mean": sse/n,
            "minimum_pair_slack": F(minimum_slack or 0, common),
            "pair_checks": graph["pair_count"], "global_extension_exists": True,
            "cohort_only_diagnostic_witness": True}


def verify_dual(targets, signed_edges, graph, kappa="2", output_interval=("0", "1"), grid_bits=40):
    """Reconstruct a range-aware weak-dual lower objective from arbitrary multipliers.

    A signed multiplier lambda for i<j contributes lambda*(v_i-v_j)
    minus |lambda|*kappa*d_plus. Taking exact independent box minima yields
    a lower objective on the relaxed distance-upper feasible set, hence also
    on the exact stored-feature class. No numerical optimizer claim is needed.
    """
    y = list(map(q, targets)); low, high = map(q, output_interval); kap = q(kappa)
    require(len(y) == graph["n"] and kap >= 0 and low < high,
            "Invalid weak-dual cohort or class")
    require(type(grid_bits) is int and 1 <= grid_bits <= 512,
            "Invalid signed-multiplier grid")
    incidence, penalty, seen = [F(0)]*len(y), F(0), set()
    for i, j, value in signed_edges:
        require(type(i) is int and type(j) is int and (i, j) in graph["lookup"] and (i, j) not in seen,
                "Unknown or repeated signed-dual edge")
        seen.add((i, j))
        lam = q(value)
        require((lam*(1 << grid_bits)).denominator == 1,
                "Signed multiplier is not on the frozen exact grid")
        incidence[i] += lam; incidence[j] -= lam
        penalty += abs(lam)*kap*graph["lookup"][i, j]["upper"]
    minimizers = [clip(target-linear/2, low, high) for target, linear in zip(y, incidence)]
    terms = [(value-target)**2+linear*value for target, linear, value in zip(y, incidence, minimizers)]
    raw = sum(terms, F(0))-penalty
    # The all-zero multiplier witness yields the exact range-only floor.
    floor = original_target_floor(y, (low, high))["sse"]
    lower = max(raw, floor)
    return {"incidence": incidence, "box_minimizers": minimizers,
            "coordinate_minima": terms, "penalty": penalty, "unclamped_sse": raw,
            "range_floor_sse": floor, "sse": lower, "mean": lower/len(y),
            "signed_edge_count": len(seen), "zero_multiplier_fallback_selected": floor > raw}


def three_way(lower, upper, epsilon):
    lower, upper, epsilon = map(q, (lower, upper, epsilon))
    require(0 <= lower <= upper and epsilon >= 0, "Invalid class-risk bracket or tolerance")
    if lower > epsilon:
        return "CLASS_INADEQUATE"
    if upper <= epsilon:
        return "CLASS_ADEQUATE"
    return "UNRESOLVED"


def head_excess(head_risk, class_risk):
    hlo, hhi = box(head_risk); clo, chi = box(class_risk)
    require(0 <= hlo <= hhi and 0 <= clo <= chi and clo <= hhi,
            "Inherited head or class interval is incompatible")
    lower, upper = max(F(0), hlo-chi), max(F(0), hhi-clo)
    return {"lower": lower, "upper": upper, "width": upper-lower}


def exact_integer(value):
    require(type(value) is int or (isinstance(value, str) and str(int(value)) == value),
            "A compact integer is not canonical")
    return int(value)


def quantized_integer(value, bits=40):
    # Fraction.__round__ implements exact nearest/ties-to-even. This does not
    # call the producer's quotient/remainder quantizer.
    return round(q(value)*(1 << bits))


def verify_binary64_list(values, expected_length):
    require(isinstance(values, list) and len(values) == expected_length,
            "Numerical proposal length differs")
    result = list(map(q, values))
    for original, value in zip(values, result):
        require(isinstance(original, str) and str(value) == original,
                "Numerical proposal is not a canonical exact rational")
        numeric = float(value)
        require(isfinite(numeric) and F.from_float(numeric) == value,
                "Retained proposal is not an exact finite binary64 value")
    return result


def class_identity(feature_id, kappa="2", output_interval=("0", "1")):
    require(isinstance(feature_id, str) and feature_id, "Missing feature identity")
    descriptor = {"schema":"step35-stored-feature-decoder-class-v1",
        "feature_id":feature_id,"metric":"Euclidean_distance_on_supplied_feature_coordinates",
        "decoder_class":"all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space",
        "kappa":str(q(kappa)),"output_interval":[str(q(v)) for v in output_interval],
        "assurance":ASSURANCE}
    return "class35:"+digest(descriptor)


def verify_min_envelope(seed, values, visit_order, weights):
    """Prove a shortest-path envelope without replaying producer Dijkstra.

    A feasible value below every seed is no greater than any path-derived
    bound. Conversely each retained value must be attained by an original
    seed or an earlier visited predecessor, proving equality to a finite
    seed-plus-path value. These two inequalities prove the exact envelope.
    """
    n = len(seed)
    require(len(values) == n and len(weights) == n and
            sorted(visit_order) == list(range(n)) and all(type(i) is int for i in visit_order),
            "Invalid shortest-path witness roster")
    require(all(value <= start for value, start in zip(values, seed)),
            "Minorant envelope exceeds an initial value")
    for i,j in combinations(range(n),2):
        require(abs(values[i]-values[j]) <= weights[i][j],
                "Envelope is not globally feasible on the conservative graph")
    seen = []
    for i in visit_order:
        require(values[i] == seed[i] or any(values[i] == values[j]+weights[j][i] for j in seen),
                "Shortest-path envelope is not attained by a seed/predecessor path")
        seen.append(i)
    return True


def verify_candidate(record, raw_seed, targets, graph, kappa):
    values = list(map(q, record["values"]))
    checked = verify_primal(values, targets, graph, kappa)
    require(q(record["sse"]) == checked["sse"] and q(record["mean"]) == checked["mean"],
            "Saved candidate original-label objective differs")
    if record["source"] == "clipped_mean_constant":
        constant = clip(sum(targets,F(0))/len(targets),F(0),F(1))
        require(q(record["value"]) == constant and values == [constant]*len(targets),
                "Constant fallback is not the exact clipped target mean")
        return checked
    require(record["source"] in ("clipped_target_envelope", "solver_envelope") and raw_seed is not None,
            "Unknown repaired-primal proposal")
    require(record["proposal_grid_bits"] == 40,
            "Primal seed grid differs from frozen 40 bits")
    seed = [quantized_integer(clip(q(v),F(0),F(1))) for v in raw_seed]
    require(record["proposal_grid_integers"] == seed,
            "Retained quantized/clipped primal seed differs")
    kap = q(kappa)
    scale = (1 << graph["bits"])*kap.denominator
    require(exact_integer(record["envelope_scale"]) == scale and scale % (1 << 40) == 0,
            "Conservative envelope integer scale differs")
    seeds = [value*(scale//(1 << 40)) for value in seed]
    low = [exact_integer(value) for value in record["minorant_grid"]]
    high = [exact_integer(value) for value in record["majorant_grid"]]
    require(len(low) == len(high) == len(seeds) and
            all(0 <= a <= start <= b <= scale for a,start,b in zip(low,seeds,high)),
            "Primal envelopes have invalid range/order")
    weights = [[0]*graph["n"] for _ in range(graph["n"])]
    for edge in graph["pairs"]:
        i,j = edge["indices"]
        weights[i][j] = weights[j][i] = kap.numerator*edge["sqrt_floor"]
    verify_min_envelope(seeds,low,record["minorant_visit_order"],weights)
    verify_min_envelope([scale-v for v in seeds],[scale-v for v in high],
                        record["majorant_visit_order"],weights)
    lo = [F(v,scale) for v in low]
    difference = [F(b-a,scale) for a,b in zip(low,high)]
    denom = sum((value**2 for value in difference),F(0))
    theta = F(0) if denom == 0 else clip(
        sum((delta*(target-start) for delta,target,start in zip(difference,targets,lo)),F(0))/denom,F(0),F(1))
    require(q(record["theta"]) == theta and
            values == [start+theta*delta for start,delta in zip(lo,difference)],
            "Primal is not the declared exact least-squares best convex blend")
    checked["shortest_path_envelopes_independently_proved"] = True
    return checked


def audit_diagnostic(record, z, y, ids, heads, expected=None):
    """Verify one complete-graph receipt using only independently supplied z/y."""
    expected = expected or {}
    z = z.tolist() if hasattr(z,"tolist") else z
    targets = list(map(q,y)); n = len(targets)
    require(record["schema"] == "step37-full-graph-class-diagnostic-v1" and
            record["assurance"] == ASSURANCE and record["output_interval"] == ["0","1"] and
            record["n"] == n and record["ids"] == ids and len(z) == n,
            "Diagnostic schema, assurance or cohort differs")
    for name in ("task","map","feature_id","class_id","epsilon","input_binding"):
        if name in expected:
            require(record[name] == expected[name],"Diagnostic binding differs: "+name)
    kap, epsilon = q(record["kappa"]), q(record["epsilon"])
    require(kap >= 0 and epsilon >= 0 and
            record["class_id"] == class_identity(record["feature_id"],kap),
            "Stored-feature class identity differs")
    require(record["targets_digest"] == digest_newline([str(v) for v in targets]),
            "Original targets digest differs")
    distance = record["distance"]
    require(distance["schema"] == "step37-complete-exact-distance-graph-v1" and
            distance["bits"] == 80 and distance["pair_order"] == "lexicographic_i_then_j_i_lt_j" and
            distance["coordinates_interpreted_as_exact_binary64"] is True and
            distance["omitted_qp_pairs_are_redundant_given_output_interval"] is True,
            "Complete-graph distance premise differs")
    graph = complete_distances(z,ids,[exact_integer(value) for value in distance["floor_scaled"]],distance["bits"])
    require(record["dimension"] == graph["dimension"] and distance["pair_count"] == graph["pair_count"],
            "Complete-graph dimension or pair count differs")
    exponent = max(value.denominator.bit_length()-1 for row in graph["features"] for value in row)
    require(all(value.denominator == 1 << (value.denominator.bit_length()-1)
                for row in graph["features"] for value in row) and
            distance["common_binary_exponent"] == exponent,
            "Stored feature dyadic denominator metadata differs")
    exact_indices = [i for i,edge in enumerate(graph["pairs"]) if edge["lower"] == edge["upper"]]
    active_indices = [i for i,edge in enumerate(graph["pairs"]) if kap*edge["lower"] < 1]
    require(distance["exact_pair_indices"] == exact_indices and
            distance["nonredundant_pair_indices"] == active_indices and
            distance["zero_distance_pair_count"] == graph["exact_zero_pairs"] and
            distance["kappa_numerator"] == kap.numerator and
            exact_integer(distance["constraint_integer_scale"]) == (1 << 80)*kap.denominator,
            "Exact/active/zero pair or constraint-scale metadata differs")
    numerical = record["numerical"]
    require(numerical["solver"] == "OSQP" and numerical["version"] == "1.0.5" and
            numerical["settings"] == expected.get("settings",PRIMARY_SETTINGS) and
            numerical["objective"] == "unhalved_SSE_without_constant_P2I_q_minus2y" and
            numerical["box_duals_used_in_certificate"] is False and
            numerical["constraint_lower_bounds_rounded_to_binary64_for_proposal_only"] is True,
            "Numerical proposal settings or semantics differ")
    raw_primal = None if numerical["raw_primal_binary64"] is None else verify_binary64_list(numerical["raw_primal_binary64"],n)
    raw_dual = None if numerical["raw_edge_dual_binary64"] is None else verify_binary64_list(numerical["raw_edge_dual_binary64"],len(active_indices))
    require(type(numerical["iterations"]) is int and 0 <= numerical["iterations"] <= numerical["settings"]["max_iter"] and
            isfinite(numerical["proposal_wall_seconds"]) and numerical["proposal_wall_seconds"] >= 0,
            "Invalid numerical iteration/timing metadata")
    if raw_primal is None or raw_dual is None:
        require(isinstance(numerical["fallback_reason"],str) and numerical["fallback_reason"],
                "Missing/nonfinite numerical proposal has no fallback reason")
    else:
        require(numerical["fallback_reason"] is None,"Unexpected fallback with complete finite proposals")
    sources = ["clipped_target_envelope"]+(["solver_envelope"] if raw_primal is not None else [])+["clipped_mean_constant"]
    candidates = record["primal_candidates"]
    require([row["source"] for row in candidates] == sources,
            "A predeclared feasible-primal candidate was omitted or added")
    candidate_checks = [verify_candidate(row, targets if row["source"] == "clipped_target_envelope" else raw_primal,
                                         targets,graph,kap) for row in candidates]
    chosen = min(candidates,key=lambda row:(q(row["sse"]),row["source"]))
    require(record["selected_primal_source"] == chosen["source"] and
            q(record["new_primal_mean"]) == q(chosen["mean"]),
            "Declared feasible-primal minimization/tie rule differs")
    saved_dual = record["dual"]
    require(saved_dual["grid_bits"] == 40 and saved_dual["distance_endpoint"] == "upper" and
            saved_dual["valid_for_any_finite_signed_multipliers"] is True,
            "Signed weak-dual precision or endpoint differs")
    expected_multipliers = []
    if raw_dual is not None:
        for index,value in zip(active_indices,raw_dual):
            lam = F(quantized_integer(value),1 << 40)
            if lam:
                expected_multipliers.append({"pair_index":index,"value":str(lam)})
    require(saved_dual["signed_multipliers"] == expected_multipliers,
            "Signed weak-dual multipliers differ from the frozen rounding rule")
    signed_edges = [(*graph["pairs"][row["pair_index"]]["indices"],q(row["value"])) for row in expected_multipliers]
    dual = verify_dual(targets,signed_edges,graph,kap)
    for saved_key, rebuilt_key in (("incidence","incidence"),("box_minimizers","box_minimizers"),
                                    ("coordinate_terms","coordinate_minima")):
        require(list(map(q,saved_dual[saved_key])) == dual[rebuilt_key],
                "Exact weak-dual coordinate reconstruction differs: "+saved_key)
    require(q(saved_dual["penalty"]) == dual["penalty"] and
            q(saved_dual["sse"]) == dual["unclamped_sse"] and
            q(saved_dual["mean"]) == dual["unclamped_sse"]/n,
            "Exact weak-dual penalty/objective differs")
    floor = original_target_floor(targets)
    require(list(map(q,record["range_floor"]["singleton_sse"])) == [(v-t)**2 for v,t in zip(floor["values"],targets)] and
            q(record["range_floor"]["sse"]) == floor["sse"] and q(record["range_floor"]["mean"]) == floor["mean"],
            "Common range floor changed the original labels")
    require(len(heads) == 2 and [row["recipe"] for row in heads] == list(RECIPES),
            "Inherited head roster differs")
    for head in heads:
        require(head["feature_id"] == record["feature_id"] and head["class_id"] == record["class_id"] and
                head["assurance"] == ASSURANCE,"Inherited head belongs to a different class")
        box(head["decision_risk"])
    best_head = min(heads,key=lambda row:(q(row["decision_risk"]["upper"]),row["candidate_id"]))
    new_upper = q(chosen["mean"]); head_upper = q(best_head["decision_risk"]["upper"])
    lower,upper = dual["mean"],min(new_upper,head_upper)
    box(record["class_risk"],(lower,upper))
    upper_source = chosen["source"] if new_upper <= head_upper else "inherited_head:"+best_head["candidate_id"]
    lower_source = "rounded_edge_dual" if dual["unclamped_sse"] >= floor["sse"] else "common_range_floor"
    require(record["class_risk"]["upper_source"] == upper_source and record["class_risk"]["lower_source"] == lower_source,
            "Class bound source attribution differs")
    status = three_way(lower,upper,epsilon).removeprefix("CLASS_")
    require(record["diagnosis"] == status and q(record["gap_target_mean"]) == F(1,100000000) and
            record["gap_resolved"] is (upper-lower <= F(1,100000000)),
            "Class diagnosis or frozen precision flag differs")
    inherited = record["inherited_heads"]
    require(len(inherited) == len(heads),"Inherited head summary roster differs")
    excesses = []
    for saved,head in zip(inherited,heads):
        for key in ("candidate_id","head_id","recipe","decision_risk","decision_prediction_sha256"):
            require(saved[key] == head[key],"Inherited achieved-risk proof binding differs: "+key)
        excess = head_excess(head["decision_risk"],record["class_risk"])
        box(saved["excess_risk"],(excess["lower"],excess["upper"]))
        hlo,hhi = box(head["decision_risk"])
        require(saved["verified_eligible"] is (hhi <= epsilon) and
                saved["verified_ineligible"] is (hlo > epsilon) and
                saved["positive_certified_excess"] is (hlo > upper),
                "Inherited head eligibility/excess claim differs")
        excesses.append({"candidate_id":head["candidate_id"],**excess})
    eligible_count = sum(q(head["decision_risk"]["upper"]) <= epsilon for head in heads)
    require(record["existing_eligible_head_count"] == eligible_count and
            record["new_finite_cohort_adequacy"] is (status == "ADEQUATE" and eligible_count == 0),
            "New finite-cohort adequacy attribution differs")
    scope = record["scope"]
    require(scope == {"diagnostic_uses_decision_labels":True,"assignment_is_a_trained_prediction_head":False,
        "new_forecast_predictions_computed":0,"native_transform_error_enclosed":False,
        "population_risk_claim":False,"future_response_coverage_claim":False,
        "causal_optimizer_failure_claim":False,"reserved_patients_accessed":0},
        "Diagnostic witness assurance exceeded its frozen scope")
    return {"status":"PASS","task":record["task"],"map":record["map"],"n":n,
        "dimension":graph["dimension"],"distance_pairs_verified":graph["pair_count"],
        "primal_candidates_verified":len(candidates),"primal_pair_constraints_verified":graph["pair_count"]*len(candidates),
        "shortest_path_envelopes_verified":2*(len(candidates)-1),
        "signed_dual_edges_verified":dual["signed_edge_count"],"outside_range_target_count":floor["outside_count"],
        "exact_duplicate_feature_pairs":graph["exact_zero_pairs"],
        "conservative_zero_radius_pairs":graph["conservative_zero_pairs"],
        "duplicate_feature_groups":graph["duplicate_feature_groups"],"unique_feature_rows":graph["unique_feature_rows"],
        "epsilon":epsilon,"range_floor_mean":floor["mean"],"class_risk":{"lower":lower,"upper":upper,"width":upper-lower},
        "diagnosis":status,"upper_source":upper_source,"lower_source":lower_source,
        "new_finite_cohort_adequacy":record["new_finite_cohort_adequacy"],"gap_resolved":record["gap_resolved"],
        "inherited_head_excess":excesses,"numerical_status_is_not_a_proof_premise":True}


def map_names():
    result = ["raw_"+scale for scale in ("minmax01","standard")]
    result += ["nystrom_"+scale+"_w"+str(width) for scale in ("minmax01","standard") for width in (8,24)]
    result += ["kahm_"+scale+"_cap"+str(cap)+"_p"+str(prefix)
               for scale in ("minmax01","standard") for cap in (4,12,20) for prefix in (1,2)]
    return sorted(result)


def checked_path(relative, root=ROOT):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            "Unsafe path in frozen receipt")
    resolved = (root/path).resolve()
    require(resolved.is_relative_to(root.resolve()),"Frozen path escapes Step37 root")
    return root/path


def candidate_prefix(path):
    # Deliberately decode only candidate metadata. The remainder contains a
    # target/range-floor receipt and is not parsed before the protocol gate.
    text = Path(path).read_text()
    prefix = '{"candidate_rows":'
    require(text.startswith(prefix),"Unexpected prior canonical exhaustive record")
    rows,end = json.JSONDecoder().raw_decode(text,len(prefix))
    require(isinstance(rows,list) and text[end:end+1] == ",", "Invalid candidate metadata prefix")
    require(all(not any(key.startswith("test") for key in row) for row in rows),
            "Pretest candidate metadata contains a test outcome")
    return rows


def verify_inputs():
    """Independent metadata/opaque-hash audit; no feature/label values parsed."""
    manifest_path = ROOT/"inputs/source_manifest37.json"
    manifest = read_json(manifest_path)
    require(manifest["schema"] == "step37-decision-input-manifest-v1" and
            manifest["prior"] == PRIOR_HASHES and manifest["task_order"] == list(TASKS) and
            manifest["map_order"] == map_names() and set(manifest["tasks"]) == set(TASKS) and
            manifest["class"] == {"kappa":"2","prediction_interval":["0","1"],"assurance":ASSURANCE},
            "Frozen input/target/map/class scope differs")
    require(manifest["preparation_source_sha256"] == sha(ROOT/"source/prepare_inputs37.py"),
            "Opaque input preparation source differs from its recorded hash")
    require(manifest["copied_feature_array_count"] == 54 and manifest["candidate_metadata_count"] == 108 and
            manifest["prior_result_hashed_opaquely_only"] is True,
            "Opaque preparation count/premise differs")
    access = manifest["prefreeze_access"]
    for key in ("decision_feature_array_payloads_numerically_parsed","decision_cohort_target_payloads_numerically_parsed",
                "exhaustive_record_target_payloads_numerically_parsed","training_or_test_feature_arrays_opened",
                "raw_source_archives_opened","new_prediction_model_fits","reserved_patient_arrays_opened"):
        require(type(access[key]) is int and access[key] == 0,"Prefreeze scope violation: "+key)
    require(access["numpy_headers_only_inspected"] is True and access["new_unseen_dataset_claim"] is False and
            access["candidate_metadata_decoded_from_pretest_prefix_only"] is True,
            "Prefreeze access interpretation differs")
    files = manifest["files"]
    require(len(files) == 71 and len({row["path"] for row in files}) == 71,
            "Input whitelist size or uniqueness differs")
    registry = {row["path"]:row for row in files}
    actual = {str(path.relative_to(ROOT)) for path in (ROOT/"inputs").rglob("*") if path.is_file()}
    require(actual == set(registry)|{"inputs/source_manifest37.json"},
            "Unlisted or missing Step37 input")
    for row in files:
        path = checked_path(row["path"])
        require(path.relative_to(ROOT).parts[0] == "inputs" and path.is_file() and
                sha(path) == row["sha256"] == row["source_sha256"] and
                path.stat().st_size == row["bytes"] and row["byte_identical_copy"] is True,
                "Opaque input identity differs: "+row["path"])
    prior_paths = {"protocol_sha256":"inputs/prior/protocol36.json",
        "protocol_seal_sha256":"inputs/prior/protocol_seal36.json",
        "model_seal_sha256":"inputs/prior/model_seal36.json",
        "postrun_audit_sha256":"inputs/prior/postrun_audit36.json"}
    for key,path in prior_paths.items():
        require(registry[path]["sha256"] == PRIOR_HASHES[key],"Prior immutable evidence hash differs")
    prior_protocol = read_json(ROOT/prior_paths["protocol_sha256"])
    prior_seal = read_json(ROOT/prior_paths["protocol_seal_sha256"])
    prior_model = read_json(ROOT/prior_paths["model_seal_sha256"])
    prior_audit = read_json(ROOT/prior_paths["postrun_audit_sha256"])
    require(prior_audit["status"] == "PASS" and
            all(prior_audit[key] == PRIOR_HASHES[key] for key in ("result_sha256","protocol_seal_sha256","model_seal_sha256")) and
            prior_model["protocol_seal_sha256"] == PRIOR_HASHES["protocol_seal_sha256"],
            "Prior audited prediction/membership inheritance is not bound to the expected experiment")
    require(prior_protocol["tasks"] == list(TASKS) and
            prior_protocol["class"]["kappa"] == "2" and prior_protocol["class"]["prediction_interval"] == ["0","1"] and
            prior_protocol["class"]["assurance"] == ASSURANCE,
            "Prior admissible decoder class or target roster differs")
    for row in files:
        if row["bound_in_prior_pretest_model_seal"]:
            require(prior_model["bound_files"].get(row["source_relative_path"]) == row["sha256"],
                    "Copied input is not the exact prior pretest-sealed file")
        if row["bound_in_prior_protocol_seal"]:
            require(prior_seal["bound_files"].get(row["source_relative_path"]) == row["sha256"],
                    "Copied input is not the exact prior protocol-sealed file")
    expected_files = {"inputs/prior/protocol36.json","inputs/prior/protocol_seal36.json",
                      "inputs/prior/model_seal36.json","inputs/prior/postrun_audit36.json",
                      "inputs/prior/public_source36.json"}
    head_fields = {"candidate_id","task","map","recipe","feature_id","class_id","head_id",
        "feature_width","family","scaling","decision_risk","deployment_payload_bytes",
        "head_sha256","fit_sha256","decision_prediction_sha256","fit_status",
        "projection_applied","assurance","head_path","fit_receipt_path","decision_prediction_path"}
    bound_heads = 0
    for task in TASKS:
        spec = manifest["tasks"][task]
        paths = {"cohort_path":(f"inputs/decision/{task}/cohort.json",f"output/cohorts/{task}/decision.json","decision_cohort_opaque"),
            "training_plan_path":(f"inputs/decision/{task}/training_plan36.json",f"output/tasks/{task}/training_plan36.json","training_derived_threshold_metadata"),
            "map_manifest_path":(f"inputs/decision/{task}/map_manifest36.json",f"output/tasks/{task}/map_manifest36.json","frozen_map_identity_metadata"),
            "exhaustive_path":(f"inputs/decision/{task}/exhaustive_r00.json",f"output/search/{task}/exhaustive_r00.json","pretest_exhaustive_record_opaque")}
        for key,(relative,prior_relative,role) in paths.items():
            require(spec[key] == relative and registry[relative]["role"] == role and
                    registry[relative]["source_relative_path"] == prior_relative and
                    registry[relative]["bound_in_prior_pretest_model_seal"] is True,
                    "Task source path, role or prior binding differs")
            expected_files.add(relative)
        plan = read_json(ROOT/spec["training_plan_path"])
        maps = read_json(ROOT/spec["map_manifest_path"])
        prior_rows = candidate_prefix(ROOT/spec["exhaustive_path"])
        require(len(prior_rows) == 36 and len({row["candidate_id"] for row in prior_rows}) == 36,
                "Prior pretest fitted-head roster is incomplete")
        original = {row["candidate_id"]:row for row in prior_rows}
        require(spec["expected_n"] == prior_audit["cohorts"][task]["decision"] == 172 and
                spec["epsilon"] == plan["epsilon"] == prior_audit["exact_training_thresholds"][task] and
                spec["epsilon_physical_mse"] == plan["epsilon_physical_mse"] and q(spec["epsilon"]) > 0 and
                plan["task"] == task,
                "Decision cohort size or unchanged training-derived tolerance differs")
        require([row["map"] for row in spec["maps"]] == map_names() and set(maps) == set(map_names()),
                "A representation class is omitted, substituted or reordered")
        for item in spec["maps"]:
            name = item["map"]; path = f"inputs/decision/{task}/features/{name}.npy"
            file = registry[path]
            require(item["feature_path"] == path and file["role"] == "decision_features_opaque" and
                    file["source_relative_path"] == f"output/features/{task}/decision/{name}.npy" and
                    file["bound_in_prior_pretest_model_seal"] is True and
                    item["feature_sha256"] == file["sha256"] and
                    file["numpy_header"]["shape"] == [172,item["dimension"]] and
                    file["numpy_header"]["payload_values_parsed"] is False,
                    "Stored feature matrix scope/identity metadata differs")
            expected_files.add(path)
            require(set(item["heads"]) == set(RECIPES) and maps[name]["feature_id"] == item["feature_id"] and
                    maps[name]["sha256"] == item["map_payload_sha256"] and
                    item["class_id"] == class_identity(item["feature_id"]),
                    "Feature payload/decoder class identity differs")
            for recipe in RECIPES:
                head = item["heads"][recipe]
                require(set(head) == head_fields and head["candidate_id"] == name+"::"+recipe and
                        head == {key:original[head["candidate_id"]][key] for key in head_fields} and
                        head["task"] == task and head["map"] == name and head["recipe"] == recipe and
                        head["feature_id"] == item["feature_id"] and head["class_id"] == item["class_id"] and
                        head["feature_width"] == item["dimension"] and head["family"] == item["family"] and
                        head["scaling"] == item["scaling"] and head["assurance"] == ASSURANCE,
                        "Inherited head metadata differs from the same-class pretest proof row")
                lo,hi = box(head["decision_risk"])
                require(0 <= lo <= hi,"Invalid inherited achieved-risk enclosure")
                for hash_key,path_key in (("head_sha256","head_path"),("fit_sha256","fit_receipt_path"),
                                           ("decision_prediction_sha256","decision_prediction_path")):
                    require(prior_model["bound_files"].get(head[path_key]) == head[hash_key],
                            "Inherited head membership/prediction proof is outside prior audited seal")
                bound_heads += 1
    require(set(registry) == expected_files and bound_heads == 108,
            "Opaque input whitelist or inherited proof roster differs")
    return manifest,{"status":"PASS","copied_inputs_verified":len(files),"decision_feature_files_hashed_opaquely":54,
        "decision_feature_values_parsed":False,"decision_target_values_parsed":False,
        "prior_same_class_prediction_proof_bindings_verified":bound_heads,
        "prior_proof_payloads_reopened":False,"training_or_test_feature_arrays_opened":False,
        "raw_source_archive_opened":False,"new_prediction_model_fits":0,"prior_result_sha256":PRIOR_HASHES["result_sha256"],
        "prior_membership_and_achieved_risk_assurance":"Inherited from the fixed Step36 independent PASS; all new interval arithmetic independently verified"}


def audit_protocol(protocol, manifest):
    require(protocol["schema"] == "step37-prospective-diagnostic-protocol-v1" and protocol["step"] == 37 and
            protocol["tasks"] == list(TASKS) and protocol["map_order"] == map_names() and
            protocol["expected_classes"] == 54 and protocol["inherited_heads"] == 108 and
            protocol["cohort"] == "previously_opened_step36_decision" and protocol["prior"] == PRIOR_HASHES and
            protocol["input_manifest_sha256"] == sha(ROOT/"inputs/source_manifest37.json"),
            "Step37 protocol roster/prior/input binding differs")
    require(protocol["class"] == {"kappa":"2","prediction_interval":["0","1"],"assurance":ASSURANCE,
        "feature_ids_and_class_ids_unchanged":True,"full_pairwise_constraints":True,
        "norm":"Euclidean","loss":"original_unclipped_target_squared_error"},
        "The frozen sensitivity, range, norm or objective changed")
    scope = protocol["scientific_scope"]
    for key in ("exploratory_analysis","features_and_targets_previously_opened_in_step36",
                "frozen_before_new_step37_numerical_diagnostic","diagnostic_uses_decision_labels",
                "diagnostic_is_not_future_forecast_evidence"):
        require(scope[key] is True,"Missing diagnostic scope premise: "+key)
    for key in ("new_independent_validation","training_or_test_arrays_allowed","raw_source_archive_allowed",
                "reserved_patient_arrays_allowed","new_prediction_model_fits_allowed","native_transform_error_enclosed"):
        require(scope[key] is False,"Unauthorized protocol scientific scope: "+key)
    require(scope["g3"] == "NOT_ESTABLISHED" and scope["g4"] == "NOT_STARTED" and
            scope["natural_study"] == "NO_GO","Prior study gate was promoted by exploratory diagnosis")
    for task in TASKS:
        require(protocol["thresholds"][task] == {"epsilon":manifest["tasks"][task]["epsilon"],
            "epsilon_physical_mse":manifest["tasks"][task]["epsilon_physical_mse"],
            "rule":"unchanged_training_derived_step36_tolerance"},
            "Frozen training-derived tolerance was changed")
    require(protocol["representation_controls"] == {"both_scalings":["minmax01","standard"],
        "kahm_native_caps":[4,12,20],"kahm_realized_dimension_rule":"min(cap,D,N_local-1)",
        "kahm_prefixes":[1,2],"maps_refit":False,"maps_added_or_removed":False},
        "Fixed representation controls differ")
    numerical = protocol["numerics"]
    require(numerical["solver"] == "OSQP" and numerical["version"] == "1.0.5" and numerical["algebra"] == "builtin" and
            numerical["settings"] == PRIMARY_SETTINGS and numerical["distance_bits"] == 80 and
            numerical["proposal_and_dual_grid_bits"] == 40 and
            numerical["objective"] == "unhalved_SSE_without_constant_P2I_q_minus2y" and
            numerical["qp_constraints"] == "box_I_plus_all_nonredundant_pair_rows" and
            numerical["omit_pair_only_if"] == "kappa_times_exact_lower_distance_ge_output_interval_width" and
            numerical["solver_status_is_not_a_proof"] is True and numerical["unfinished_finite_vectors_are_usable"] is True and
            numerical["adaptive_retries_allowed"] is False and
            numerical["numerical_exception_policy"] == "record_then_fixed_exact_fallback" and
            numerical["proposal_rounding"] == "nearest_dyadic_ties_even",
            "Frozen numerical proposal/fallback protocol differs")
    require(protocol["primal"] == {"candidate_order":["clipped_target_envelope","solver_envelope_if_finite","clipped_mean_constant"],
        "envelopes":"exact_complete_graph_multisource_Dijkstra_for_minorant_and_majorant",
        "repair":"exact_best_convex_blend_against_original_targets","proposal_targets_clipped_only_for_range":True,
        "loss_targets_clipped":False,"select":"minimum_exact_SSE_then_lexicographic_source",
        "class_upper":"min_new_verified_primal_mean_and_best_inherited_admissible_head_upper"},
        "Exact primal candidate/repair/selection protocol differs")
    require(protocol["dual"] == {"signed_edges":"rounded_OSQP_edge_duals_box_duals_unused",
        "endpoint":"upper_outward_distance","scalar_minimizer":"clip(y-Btranspose_lambda/2,I)",
        "lower":"max_common_range_floor_and_signed_weak_dual_mean","absent_nonfinite_fallback":"all_edge_multipliers_zero"},
        "Exact signed weak-dual protocol differs")
    require(protocol["decision"] == {"adequate":"upper_le_epsilon","inadequate":"lower_gt_epsilon","otherwise":"UNRESOLVED",
        "gap_target_mean":"1/100000000","gap_resolution_is_separate_from_threshold_decision":True,
        "head_excess":"[max(0,head_lower-class_upper),max(0,head_upper-class_lower)]",
        "head_inadequacy_requires":"head_lower_gt_epsilon",
        "new_finite_cohort_adequacy":"class_adequate_and_neither_inherited_head_verified_eligible"},
        "Three-way threshold/excess interpretation differs")
    execution = protocol["execution"]
    require(execution["single_primary_pass"] is True and execution["task_order"] == list(TASKS) and
            execution["map_order"] == map_names() and execution["global_wall_budget_seconds"] == 1800.0 and
            execution["budget_checked_between_classes"] is True and
            execution["hard_arithmetic_or_provenance_failure"] == "stop_and_retain_partial_evidence_no_rerun" and
            execution["numerical_failure"] == "retain_class_with_fixed_exact_fallback_no_retry" and
            execution["budget_stop"] == "retain_all_completed_classes_and_explicit_missing_roster" and
            execution["new_heldout_forecasts"] == 0 and execution["no_outcome_driven_setting_changes"] is True and
            execution["input_loader_requires_independent_preexecution_pass"] is True,
            "Primary-pass/budget/gate protocol differs")
    reporting = protocol["reporting"]
    for key in ("all_classes_and_heads_required","missing_or_failed_classes_must_be_explicit",
                "publish_exact_rational_brackets","publish_independent_receipt_audit","report_finite_cohort_scope",
                "prior_step36_test_metrics_and_decisions_unchanged"):
        require(reporting[key] is True,"Missing transparent-reporting commitment: "+key)
    for key in ("causal_optimizer_failure_claim","out_of_sample_improvement_claim","new_lipschitz_QP_novelty_claim",
                "new_exact_QP_certificate_paradigm_claim"):
        require(reporting[key] is False,"Unsupported frozen reporting claim: "+key)
    require(protocol["environment"]["packages"]["osqp"] == "1.0.5" and
            protocol["environment"]["numerical_threads"] == 1,
            "Frozen numerical environment differs")
    require(protocol["required_artificial_checks"] == ["tests/artificial_data_preflight37.json",
        "tests/artificial_producer_preflight37.json","audit/artificial_audit_preflight37.json"],
        "Required artificial gate roster differs")
    return {"unchanged_class_and_thresholds":True,"complete_class_roster":True,"single_pass_without_retries":True,
        "original_targets_preserved":True,"diagnostic_scope_preserved":True,"prior_study_gates_unchanged":True}


def verify_seal():
    path = ROOT/"protocol_seal37.json"
    require(path.is_file(),"Protocol/source/input seal is required before numerical access")
    seal = read_json(path)
    require(seal["schema"] == "step37-protocol-source-input-seal-v1" and
            seal["protocol_sha256"] == sha(ROOT/"protocol37.json") and
            seal["input_manifest_sha256"] == sha(ROOT/"inputs/source_manifest37.json") and
            seal["scientific_numerical_arrays_accessed_by_step37_before_seal"] == 0 and
            seal["all_prior_step36_outcomes_already_known"] is True and seal["exploratory_not_new_validation"] is True,
            "Protocol seal identity or scope differs")
    bounds = seal["bound_files"]
    require(isinstance(bounds,dict) and bounds,"Empty protocol/source/input seal")
    for relative,expected in bounds.items():
        path = checked_path(relative)
        require(path.is_file() and sha(path) == expected,"Frozen file changed: "+relative)
    required = {"protocol37.json","inputs/source_manifest37.json"}
    required.update(str(path.relative_to(ROOT)) for path in (ROOT/"source").glob("*.py"))
    required.update(str(path.relative_to(ROOT)) for path in (ROOT/"inputs").rglob("*") if path.is_file())
    required.update(("tests/artificial_data_preflight37.json","tests/artificial_producer_preflight37.json",
                     "audit/artificial_audit_preflight37.json"))
    require(required <= set(bounds),"Protocol seal omits a required source/input/check")
    require(sha(ROOT/"source/independent_math35.py") == INDEPENDENT_MATH_SHA256,
            "Inherited generic independent arithmetic source changed")
    return seal


def audit_events():
    path = ROOT/"output/access_and_execution37.jsonl"
    require(path.is_file(),"Missing retained protocol/access/execution event chain")
    records=[];previous=None;last_time=-1;last_utc=None
    for line in path.read_text().splitlines():
        record=json.loads(line);saved=record["event_sha256"]
        unsigned={key:value for key,value in record.items() if key != "event_sha256"}
        require(saved == digest_newline(unsigned) and record["previous_event_sha256"] == previous,
                "Retained execution event-chain digest differs")
        stamp = datetime.fromisoformat(record["utc"])
        require(type(record["monotonic_ns"]) is int and record["monotonic_ns"] >= last_time and
                (last_utc is None or stamp >= last_utc),"Execution event ordering differs")
        previous=saved;last_time=record["monotonic_ns"];last_utc=stamp;records.append(record)
    require(records and records[0]["event"] == "protocol_sealed" and
            records[0]["protocol_seal_sha256"] == sha(ROOT/"protocol_seal37.json") and
            records[0]["protocol_sha256"] == sha(ROOT/"protocol37.json"),
            "Execution event chain does not begin with this protocol freeze")
    return records


def append_audit_event(name,**fields):
    records=audit_events()
    record={"event":name,"utc":datetime.now(timezone.utc).isoformat(),"monotonic_ns":time.monotonic_ns(),
        "previous_event_sha256":records[-1]["event_sha256"],**fields}
    record["event_sha256"]=digest_newline(record)
    with (ROOT/"output/access_and_execution37.jsonl").open("a") as handle:
        handle.write(json.dumps(record,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")


def artificial_fixtures():
    receipt = read_json(ROOT/"tests/artificial_producer_preflight37.json")
    require(receipt["status"] == "PASS" and receipt["artificial_only"] is True and
            receipt["scientific_inputs_opened"] == 0,"Producer artificial checks did not pass")
    checks=[]
    for relative,expected in sorted(receipt["fixture_sha256"].items()):
        path=checked_path(relative)
        require(relative.startswith("tests/producer_fixture_") and path.suffix == ".json" and sha(path) == expected,
                "Artificial fixture identity differs")
        fixture=read_json(path)
        checked=audit_diagnostic(fixture["record"],fixture["z_binary64"],fixture["y"],fixture["ids"],
                                 fixture["heads"],fixture["expected"])
        require(canonical(checked) == fixture["independent_check"],"Retained artificial independent check differs")
        checks.append({"name":fixture["name"],"status":"PASS","fixture_sha256":expected,
                       "distance_pairs_verified":checked["distance_pairs_verified"]})
    require(len(checks) == receipt["fixtures"] == 8,"Artificial fixture roster differs")
    return checks


def artificial_preflight():
    require(not (ROOT/"protocol_seal37.json").exists(),"Artificial preflight preparation cannot edit a frozen experiment")
    checks=independent_selfcheck();manifest,input_check=verify_inputs()
    protocol_check=audit_protocol(read_json(ROOT/"protocol37.json"),manifest)
    fixtures=artificial_fixtures()
    result={"schema":"step37-independent-artificial-preflight-v1","status":"PASS",
        "utc":datetime.now(timezone.utc).isoformat(),"selfchecks":checks,"producer_fixtures_independently_verified":fixtures,
        "input_provenance":input_check,"protocol_semantics":protocol_check,
        "scientific_feature_or_label_arrays_read":False,"source_sha256":{
            "source/audit37.py":sha(ROOT/"source/audit37.py"),
            "source/independent_math35.py":sha(ROOT/"source/independent_math35.py")}}
    write_json(ROOT/"audit/artificial_audit_preflight37.json",result)
    return result


def preflight():
    require(not (ROOT/"audit/preexecution_gate37.json").exists(),"Independent preexecution gate already exists")
    require(not (ROOT/"output/primary_started37.json").exists() and not (ROOT/"output/result37.json").exists(),
            "Primary numerical execution already began before independent preflight")
    seal=verify_seal();manifest,input_check=verify_inputs()
    protocol=read_json(ROOT/"protocol37.json");protocol_checks=audit_protocol(protocol,manifest)
    selfchecks=independent_selfcheck();fixtures=artificial_fixtures()
    for relative in protocol["required_artificial_checks"]:
        receipt=read_json(ROOT/relative)
        require(receipt["status"] == "PASS" and relative in seal["bound_files"],"Artificial preflight not sealed/PASS")
        for source,expected in receipt["source_sha256"].items():
            require(sha(checked_path(source)) == expected,"Artificial source assurance became stale")
    events=audit_events()
    require([row["event"] for row in events] == ["protocol_sealed"],
            "Unexpected scientific activity before the independent gate")
    result={"schema":"step37-independent-preexecution-gate-v1","status":"PASS",
        "utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":sha(ROOT/"protocol37.json"),
        "protocol_seal_sha256":sha(ROOT/"protocol_seal37.json"),
        "input_manifest_sha256":sha(ROOT/"inputs/source_manifest37.json"),"sealed_file_count":len(seal["bound_files"]),
        "input_provenance":input_check,"protocol_semantics":protocol_checks,"selfchecks":selfchecks,
        "producer_fixtures_independently_verified":fixtures,"scientific_feature_or_label_arrays_read":False,
        "source_sha256":{"source/audit37.py":sha(ROOT/"source/audit37.py"),
                         "source/independent_math35.py":sha(ROOT/"source/independent_math35.py")}}
    write_json(ROOT/"audit/preexecution_gate37.json",result)
    append_audit_event("independent_preexecution_pass",protocol_seal_sha256=sha(ROOT/"protocol_seal37.json"),
                       preexecution_gate_sha256=sha(ROOT/"audit/preexecution_gate37.json"))
    return result


def postrun():
    """Reconstruct every completed class and audit the retained single pass."""
    started=time.perf_counter()
    seal=verify_seal();manifest,input_check=verify_inputs()
    protocol=read_json(ROOT/"protocol37.json");audit_protocol(protocol,manifest)
    gate_path=ROOT/"audit/preexecution_gate37.json"
    gate=read_json(gate_path)
    require(gate["status"] == "PASS" and gate["protocol_seal_sha256"] == sha(ROOT/"protocol_seal37.json") and
            gate["input_manifest_sha256"] == sha(ROOT/"inputs/source_manifest37.json") and
            gate["scientific_feature_or_label_arrays_read"] is False and
            gate["source_sha256"]["source/audit37.py"] == sha(ROOT/"source/audit37.py"),
            "Independent numerical-access gate is missing or stale")
    result_path=ROOT/"output/result37.json";result=read_json(result_path)
    require(result["schema"] == "step37-full-class-diagnostic-result-v1" and result["step"] == 37 and
            result["status"] in ("COMPLETE","BUDGET_STOP","STOPPED_ERROR") and
            result["prior"] == PRIOR_HASHES and result["scope"] == protocol["scientific_scope"],
            "Primary result schema, prior or scope differs")
    for field,relative in (("protocol_sha256","protocol37.json"),("protocol_seal_sha256","protocol_seal37.json"),
                           ("preexecution_gate_sha256","audit/preexecution_gate37.json"),
                           ("input_manifest_sha256","inputs/source_manifest37.json")):
        require(result[field] == sha(ROOT/relative),"Primary result provenance differs: "+field)
    require(result["new_prediction_model_fits"] == 0 and result["new_forecast_predictions"] == 0 and
            result["reserved_patients_accessed"] == 0 and result["prior_test_results_unchanged"] is True,
            "Primary result claims unauthorized new training/forecasts/test changes")
    ordered=[(task,name) for task in TASKS for name in map_names()]
    rows=result["class_rows"];completed=[(row["task"],row["map"]) for row in rows]
    require(completed == ordered[:len(rows)] and len(set(completed)) == len(rows) and
            result["completed_classes"] == len(rows) and result["expected_classes"] == 54 and
            result["inherited_heads"] == 2*len(rows) == len(result["head_rows"]) and
            result["missing_classes"] == [{"task":task,"map":name} for task,name in ordered[len(rows):]],
            "Completed prefix, missing roster or head count differs")
    if result["status"] == "COMPLETE":
        require(len(rows) == 54 and result["error"] is None,"Complete result omitted a class or retained an error")
    else:
        require(len(rows) < 54,"Stopped result has no explicit missing class")
        require((result["error"] is not None) is (result["status"] == "STOPPED_ERROR"),
                "Stopped-result error disposition differs")
    require(isfinite(result["total_wall_seconds"]) and result["total_wall_seconds"] >= 0,
            "Invalid retained primary wall time")
    events=audit_events();names=[event["event"] for event in events]
    require(names[:4] == ["protocol_sealed","independent_preexecution_pass","primary_started","decision_input_access_start"] and
            names[-1] == "primary_finished" and names.count("primary_started") == 1 and
            names.count("decision_input_access_start") == 1 and names.count("primary_finished") == 1,
            "Input/primary/gate ordering or single-pass execution differs")
    require(events[1]["preexecution_gate_sha256"] == sha(gate_path) and
            events[2]["preexecution_gate_sha256"] == sha(gate_path),"Execution did not use this independent gate")
    marker=read_json(ROOT/"output/primary_started37.json")
    require(marker["single_primary_pass"] is True and marker["protocol_seal_sha256"] == result["protocol_seal_sha256"] and
            events[2]["primary_marker_sha256"] == sha(ROOT/"output/primary_started37.json"),
            "Single-primary marker binding differs")
    require(events[-1]["result_sha256"] == sha(result_path) and events[-1]["status"] == result["status"] and
            events[-1]["completed_classes"] == len(rows) and events[-1]["missing_classes"] == 54-len(rows),
            "Final event is not bound to this complete/partial result")
    require(events[3]["protocol_seal_sha256"] == result["protocol_seal_sha256"] and
            events[3]["input_manifest_sha256"] == result["input_manifest_sha256"] and
            events[3]["input_role"] == "previously_opened_step36_decision_only" and
            events[3]["raw_archive_accessed"] is False and events[3]["training_or_test_feature_arrays_accessed"] is False and
            events[3]["reserved_patient_arrays_accessed"] is False,
            "Input-access event scope differs")
    input_access_path=ROOT/"output/input_access37.json"
    has_inputs=input_access_path.is_file()
    if has_inputs:
        access=read_json(input_access_path)
        require(result["input_access_sha256"] == sha(input_access_path) and access["status"] == "PASS" and
                access["schema"] == "step37-gated-decision-input-access-v1" and
                access["protocol_seal_sha256"] == result["protocol_seal_sha256"] and
                access["input_manifest_sha256"] == result["input_manifest_sha256"] and
                access["decision_feature_arrays_loaded"] == 54 and access["common_decision_roster"] is True and
                access["targets_clipped"] is False and access["training_or_test_feature_arrays_accessed"] is False and
                access["raw_source_archive_accessed"] is False and access["representation_or_decoder_payloads_loaded"] is False and
                access["new_prediction_model_fits"] == 0 and access["reserved_patient_arrays_accessed"] == 0 and
                access["new_unseen_dataset_claim"] is False,
                "Gated input-access proof differs")
        require(names[4] == "decision_input_access_complete" and
                events[4]["receipt_sha256"] == sha(input_access_path) and
                events[4]["decision_feature_arrays_loaded"] == 54 and events[4]["common_decision_n"] == 172 and
                events[4]["training_or_test_feature_arrays_accessed"] is False,
                "Completed gated access event differs")
        event_cursor=5
    else:
        require(result["status"] == "STOPPED_ERROR" and not rows and result["input_access_sha256"] is None,
                "A diagnostic ran without completed gated input receipt")
        event_cursor=4
    # Numerical scientific parsing starts only here, after the independent
    # gate, current seal, result and input provenance have all been checked.
    cohorts={};metadata={};reference_ids=None
    if has_inputs:
        for task in TASKS:
            spec=manifest["tasks"][task];cohort=read_json(ROOT/spec["cohort_path"])
            ids=cohort["ids"]
            require(cohort["name"] == "decision" and cohort["target"] == task and
                    len(ids) == len(set(ids)) == 172 and all(isinstance(name,str) and name.startswith("decision:") for name in ids),
                    "Numerical audit attempted a non-decision or inconsistent cohort")
            require(isinstance(cohort["y"],list) and len(cohort["y"]) == 172 and
                    all(isinstance(value,str) and str(q(value)) == value for value in cohort["y"]),
                    "Original decision labels are not canonical exact rationals")
            targets=list(map(q,cohort["y"]))
            if reference_ids is None:reference_ids=ids
            require(ids == reference_ids,"Target cohorts use different decision row rosters")
            floor=original_target_floor(targets)
            multiplier=q(spec["epsilon_physical_mse"])/q(spec["epsilon"])
            metadata[task]={"decision_n":172,"epsilon":spec["epsilon"],"epsilon_physical_mse":spec["epsilon_physical_mse"],
                "physical_mse_multiplier":str(multiplier),"targets_outside_interval":floor["outside_count"],
                "range_floor_mean":str(floor["mean"])}
            cohorts[task]={"ids":ids,"y":targets,"physical_mse_multiplier":multiplier}
            require(access["tasks"][task] == {"decision_n":172,"maps":18,"epsilon":spec["epsilon"],
                "cohort_sha256":sha(ROOT/spec["cohort_path"])},"Gated task-access summary differs")
        require(result["task_metadata"] == metadata,"Primary target/range-floor/physical-unit summary differs")
    else:
        require(result["task_metadata"] == {},"Input failure nevertheless retained numerical task summaries")
    all_head_rows=[];audited=[];paths=[]
    # NumPy is imported only after the source/manifest/gate access checks.
    import numpy as np
    for row in rows:
        task,name=row["task"],row["map"];spec=manifest["tasks"][task]
        mapping=next(item for item in spec["maps"] if item["map"] == name)
        relative=f"output/classes/{task}/{name}.json";path=ROOT/relative;paths.append(relative)
        require(row["receipt_path"] == relative and path.is_file() and row["receipt_sha256"] == sha(path),
                "Class receipt identity differs from primary summary")
        record=read_json(path)
        z=np.load(ROOT/mapping["feature_path"],allow_pickle=False)
        require(z.dtype == np.float64 and z.shape == (172,mapping["dimension"]) and np.isfinite(z).all(),
                "Stored decision feature payload dtype/shape/finiteness differs")
        binding={"input_manifest_sha256":result["input_manifest_sha256"],
            "cohort_path":spec["cohort_path"],"cohort_sha256":sha(ROOT/spec["cohort_path"]),
            "feature_path":mapping["feature_path"],"feature_sha256":mapping["feature_sha256"],
            "map_payload_sha256":mapping["map_payload_sha256"],"prior_model_seal_sha256":PRIOR_HASHES["model_seal_sha256"]}
        expected={"task":task,"map":name,"feature_id":mapping["feature_id"],"class_id":mapping["class_id"],
            "epsilon":spec["epsilon"],"input_binding":binding,"settings":protocol["numerics"]["settings"]}
        checked=audit_diagnostic(record,z,cohorts[task]["y"],cohorts[task]["ids"],
                                [mapping["heads"][recipe] for recipe in RECIPES],expected)
        require(record["kappa"] == "2","Primary sensitivity differs")
        rebuilt={"task":task,"map":name,"family":mapping["family"],"scaling":mapping["scaling"],
            "dimension":mapping["dimension"],"n":172,"feature_id":mapping["feature_id"],"class_id":mapping["class_id"],
            "epsilon":spec["epsilon"],"epsilon_physical_mse":spec["epsilon_physical_mse"],
            "physical_mse_multiplier":str(cohorts[task]["physical_mse_multiplier"]),
            "class_risk":record["class_risk"],"range_floor_mean":record["range_floor"]["mean"],
            "new_primal_mean":record["new_primal_mean"],"diagnosis":record["diagnosis"],"gap_resolved":record["gap_resolved"],
            "existing_eligible_head_count":record["existing_eligible_head_count"],
            "new_finite_cohort_adequacy":record["new_finite_cohort_adequacy"],
            "solver_status":record["numerical"]["status"],"solver_iterations":record["numerical"]["iterations"],
            "solver_fallback_reason":record["numerical"]["fallback_reason"],"pair_count":record["distance"]["pair_count"],
            "nonredundant_pair_count":len(record["distance"]["nonredundant_pair_indices"]),
            "zero_distance_pair_count":record["distance"]["zero_distance_pair_count"],
            "nonzero_dual_count":len(record["dual"]["signed_multipliers"]),
            "selected_primal_source":record["selected_primal_source"],"timing":record["timing"],
            "receipt_path":relative,"receipt_sha256":sha(path)}
        require(row == rebuilt,"Primary class summary differs from exact audited receipt")
        for head in record["inherited_heads"]:
            all_head_rows.append({"task":task,"map":name,"family":mapping["family"],"scaling":mapping["scaling"],
                "epsilon":record["epsilon"],"class_diagnosis":record["diagnosis"],"class_risk":record["class_risk"],**head})
        require(event_cursor+1 < len(events) and
                events[event_cursor]["event"] == "class_diagnostic_started" and
                events[event_cursor+1]["event"] == "class_diagnostic_completed",
                "Completed class has no single started/completed event pair")
        begin,end=events[event_cursor:event_cursor+2]
        require(begin["task"] == end["task"] == task and begin["map"] == end["map"] == name and
                begin["feature_id"] == mapping["feature_id"] and begin["class_id"] == mapping["class_id"] and
                end["receipt_path"] == relative and end["receipt_sha256"] == sha(path) and end["diagnosis"] == record["diagnosis"],
                "Class event identity/digest differs")
        event_cursor+=2;checked["receipt_sha256"]=sha(path);audited.append(checked)
        print(json.dumps({"independently_audited_classes":len(audited),"task":task,"map":name,
            "diagnosis":checked["diagnosis"],"status":"PASS"}),flush=True)
    require(result["head_rows"] == all_head_rows,"Primary inherited head/excess rows differ")
    actual_paths={str(path.relative_to(ROOT)) for path in (ROOT/"output/classes").rglob("*.json")}
    require(actual_paths == set(paths),"Unreported or missing primary class receipt")
    tail=events[event_cursor:-1]
    if result["status"] == "STOPPED_ERROR":
        require(tail and tail[-1]["event"] == "primary_stopped_error" and tail[-1]["error"] == result["error"],
                "Stopped error is not retained in execution evidence")
        if len(tail) == 2:
            task,name=ordered[len(rows)]
            require(tail[0]["event"] == "class_diagnostic_started" and tail[0]["task"] == task and tail[0]["map"] == name,
                    "Failed class is not the next fixed class")
        else:require(len(tail) == 1,"Unexpected retry/extra event after partial execution")
    else:
        require(not tail,"Unexpected retry or extra execution event")
    totals={"classes_verified":len(audited),"inherited_heads_verified":len(all_head_rows),
        "distance_pairs_verified":sum(row["distance_pairs_verified"] for row in audited),
        "primal_pair_constraints_verified":sum(row["primal_pair_constraints_verified"] for row in audited),
        "shortest_path_envelopes_verified":sum(row["shortest_path_envelopes_verified"] for row in audited),
        "signed_dual_edges_verified":sum(row["signed_dual_edges_verified"] for row in audited)}
    outcome={task:{"ADEQUATE":sum(row["task"] == task and row["diagnosis"] == "ADEQUATE" for row in audited),
        "INADEQUATE":sum(row["task"] == task and row["diagnosis"] == "INADEQUATE" for row in audited),
        "UNRESOLVED":sum(row["task"] == task and row["diagnosis"] == "UNRESOLVED" for row in audited),
        "new_finite_cohort_adequacy":sum(row["task"] == task and row["new_finite_cohort_adequacy"] for row in audited),
        "gap_resolved":sum(row["task"] == task and row["gap_resolved"] for row in audited)} for task in TASKS}
    audit={"schema":"step37-independent-postrun-audit-v1","status":"PASS",
        "utc":datetime.now(timezone.utc).isoformat(),"primary_status":result["status"],
        "result_sha256":sha(result_path),"protocol_sha256":result["protocol_sha256"],
        "protocol_seal_sha256":result["protocol_seal_sha256"],"preexecution_gate_sha256":sha(gate_path),
        "input_manifest_sha256":result["input_manifest_sha256"],"prior":PRIOR_HASHES,
        "source_sha256":{"source/audit37.py":sha(ROOT/"source/audit37.py"),
                         "source/independent_math35.py":sha(ROOT/"source/independent_math35.py")},
        "input_provenance":input_check,"totals":totals,"outcomes":outcome,"class_checks":audited,
        "missing_classes":result["missing_classes"],"all_reported_classes_verified":True,
        "single_primary_pass_and_gated_order_verified":True,"original_targets_unclipped":True,
        "all_pair_primal_constraints_independently_verified":True,"all_signed_dual_arithmetic_independently_verified":True,
        "prior_head_membership_and_risk_proofs_inherited_from_step36_audit":True,
        "new_head_excess_arithmetic_independently_verified":True,"numerical_solver_status_used_as_proof":False,
        "training_or_test_feature_arrays_opened":False,"raw_source_archive_opened":False,"new_model_fits":0,
        "reserved_patient_arrays_opened":0,"event_count":len(events),"event_chain_sha256":sha(ROOT/"output/access_and_execution37.jsonl"),
        "audit_wall_seconds":time.perf_counter()-started,
        "scope":"Exploratory finite decision-cohort class-risk certificates conditional on exact stored features; no new forecasting/generalization guarantee"}
    write_json(ROOT/"audit/postrun_audit37.json",audit)
    return audit


def independent_selfcheck():
    require(sha(Path(old.__file__)) == INDEPENDENT_MATH_SHA256,
            "The inherited independent arithmetic checker changed")
    checks = dict(old.independent_selfcheck())
    # Compact floor validation uses exact integer inequalities, not sqrt.
    for value, floor, bits in ((F(0), 0, 80), (F(9, 4), 3 << 79, 80),
                               (F(2), 181, 7), (F(1, 3), 73, 7)):
        lo, hi = verify_sqrt_floor(value, floor, bits)
        require(lo*lo <= value <= hi*hi and hi-lo <= F(1, 1 << bits),
                "Independent compact floor check failed")
    checks["compact_integer_floor_inequalities"] = True
    for value, floor, bits in ((F(2), 180, 7), (F(2), 182, 7)):
        try:
            verify_sqrt_floor(value, floor, bits)
        except AuditError:
            pass
        else:
            raise AuditError("Wrong sqrt-floor fixture was accepted")
    checks["incorrect_floor_rejected"] = True
    graph = complete_distances([[0], [0]], ["a", "b"], [0])
    primal = verify_primal([F(1, 2), F(1, 2)], [0, 1], graph)
    dual = verify_dual([0, 1], [(0, 1, -1)], graph)
    require(primal["sse"] == dual["sse"] == F(1, 2),
            "Exact duplicate-class optimum check failed")
    checks["duplicate_feature_primal_dual_equality"] = True
    try:
        verify_primal([0, 1], [0, 1], graph)
    except AuditError:
        pass
    else:
        raise AuditError("An infeasible duplicate-feature witness was accepted")
    checks["duplicate_inconsistency_rejected"] = True
    outside_primal = verify_primal([F(1, 2)]*2, [-1, 2], graph)
    outside_dual = verify_dual([-1, 2], [(0, 1, -3)], graph)
    require(outside_primal["sse"] == outside_dual["sse"] == F(9, 2) and
            original_target_floor([-1, 2])["sse"] == 2,
            "Original outside-range targets were changed in the objective")
    checks["outside_range_labels_retained"] = True
    chain = complete_distances([[0], [F(1, 4)], [F(1, 2)]],
        ["left", "middle", "right"], [1 << 78, 1 << 79, 1 << 78])
    chain_primal = verify_primal([F(1,4), F(1,2), F(1,4)], [0,1,0], chain, kappa="1")
    chain_dual = verify_dual([0,1,0], [(0,1,F(-1,2)), (1,2,F(1,2))], chain, kappa="1")
    require(chain_primal["sse"] == chain_dual["sse"] == F(3,8),
            "Intersecting graph constraints or signed incidence failed")
    checks["three_point_intersecting_constraints"] = True
    wide_graph = complete_distances([[i] for i in range(40)],
        [str(i) for i in range(40)], [(j-i)*(1 << 80) for i,j in combinations(range(40),2)])
    wide = verify_primal([F(i % 2) for i in range(40)], [F(i % 2) for i in range(40)], wide_graph)
    require(wide["sse"] == 0 and wide["pair_checks"] == 780,
            "Full graph incorrectly retained the inherited 32-point limit")
    checks["full_graph_above_32_rows"] = True
    require(three_way(0, 1, 1) == "CLASS_ADEQUATE" and
            three_way(1, 2, 1) == "UNRESOLVED" and
            three_way(F(3, 2), 2, 1) == "CLASS_INADEQUATE",
            "Three-way strict/equality decision differs")
    checks["three_way_boundary_rules"] = True
    require(head_excess({"lower":"3/4","upper":"1"},
            {"lower":"1/4","upper":"1/2"}) ==
            {"lower":F(1,4),"upper":F(3,4),"width":F(1,2)},
            "Inherited-head excess interval arithmetic differs")
    checks["excess_risk_interval"] = True
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["selfcheck","artificial","preflight","postrun"])
    args = parser.parse_args()
    try:
        if args.mode == "selfcheck":
            output={"status":"PASS","selfchecks":independent_selfcheck(),
                    "scientific_feature_or_label_arrays_read":False}
        elif args.mode == "artificial":
            output=artificial_preflight()
        elif args.mode == "preflight":
            output=preflight()
        else:
            output=postrun()
        summary={key:output[key] for key in ("status","schema","primary_status","protocol_seal_sha256",
                 "result_sha256","totals","outcomes","scientific_feature_or_label_arrays_read") if key in output}
        if args.mode == "selfcheck":summary["selfchecks"]=output["selfchecks"]
        print(json.dumps(canonical(summary),sort_keys=True),flush=True)
    except Exception as exc:
        failure={"status":"FAIL","mode":args.mode,"utc":datetime.now(timezone.utc).isoformat(),
            "error_type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),
            "source_sha256":sha(Path(__file__))}
        write_json(ROOT/("audit/"+args.mode+"_audit_failure37.json"),failure)
        print(json.dumps(failure,sort_keys=True),flush=True)
        raise SystemExit(1)
