"""Step36 fixed-tolerance search and bounded disjoint-pair class screening.

No files or scientific datasets are opened here.  Selection uses binary64
distances only to choose a reproducible sparse graph; every reported lower
bound is then proved using exact rational arithmetic on the supplied stored
coordinates.  The exact-distance work cap does not bound floating-point graph
construction, serialization, total CPU time, or wall time.
"""
from __future__ import annotations

from fractions import Fraction as F
import json
from math import isfinite

import finite_feature35 as ff


ASSURANCE = ff.ASSURANCE
NEIGHBORS = 4
MAX_PAIRS = 64
DISTANCE_BITS = 80
MAX_CERTIFICATE_CALLS = 6
PAIR_SCHEMA = "step36-disjoint-pair-class-lower-v1"
PAYLOAD_SCHEMA = "step36-prefit-head-payload-bound-v1"
POLICY_SCHEMA = "step36-fixed-tolerance-search-trace-v1"


class PolicyError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise PolicyError(message)


def _integer(value, name, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum and
             (maximum is None or value <= maximum), "Invalid " + name)
    return value


def _text(value, name):
    _require(isinstance(value, str) and bool(value), "Invalid " + name)
    return value


def _interval(values):
    _require(isinstance(values, (list, tuple)) and len(values) == 2,
             "Expected two interval endpoints")
    low, high = map(ff.rational, values)
    _require(low < high, "Predictor interval must have positive width")
    return low, high


def _clip(value, low, high):
    return min(high, max(low, value))


def _matrix(values):
    _require(isinstance(values, (list, tuple)) and len(values) > 0,
             "Expected a nonempty feature matrix")
    _require(isinstance(values[0], (list, tuple)) and len(values[0]) > 0,
             "Expected positive feature dimension")
    dimension = len(values[0])
    _require(all(isinstance(row, (list, tuple)) and len(row) == dimension
                 for row in values), "Inconsistent feature dimension")
    return [[ff.rational(x) for x in row] for row in values]


def _names(ids, n):
    if ids is None:
        return [f"q{i:06d}" for i in range(n)]
    _require(isinstance(ids, (list, tuple)) and len(ids) == n,
             "Identifier roster has wrong length")
    names = [_text(value, "query identifier") for value in ids]
    _require(len(set(names)) == n, "Duplicate query identifiers")
    return names


def head_payload_bytes(head):
    """Canonical deployed-head bytes, including the required trailing newline."""
    return len((json.dumps(ff.canonical(head), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))


def head_cost_lower_bound(kind, *, feature_id, dimension, map_payload_bytes,
        anchor_features=None, anchor_ids=None, kappa="2", output_interval=("0", "1")):
    """Minimum canonical bytes conditional on all already-known head fields.

    Each unknown rational value must serialize as a nonempty JSON string.
    Replacing it by the one-character string "0" gives a byte lower bound.
    The template need not be an admissible fitted head: it only bounds string
    lengths.  All known anchor coordinates, identifiers and schema fields are
    retained exactly, so no fitted or shadow-fitted coefficient size is used.
    """
    _text(feature_id, "feature identifier")
    _integer(dimension, "dimension", 1)
    _integer(map_payload_bytes, "map payload byte count")
    kap = ff.rational(kappa)
    _require(kap >= 0, "Negative sensitivity")
    low, high = _interval(output_interval)
    aliases = {"affine": "clipped_affine", "clipped_affine": "clipped_affine",
        "extension32": "symmetric_lipschitz_extension",
        "extension": "symmetric_lipschitz_extension",
        "symmetric_lipschitz_extension": "symmetric_lipschitz_extension"}
    _require(kind in aliases, "Unknown head recipe")
    kind = aliases[kind]
    head = {"schema": ff.HEAD_SCHEMA, "kind": kind, "feature_id": feature_id,
        "class_id": ff.class_identity(feature_id, kap, (low, high)),
        "dimension": dimension, "kappa": str(kap),
        "output_interval": [str(low), str(high)], "assurance": ASSURANCE}
    if kind == "clipped_affine":
        _require(anchor_features is None and anchor_ids is None,
                 "Affine byte template cannot include anchor fields")
        head.update(coefficients=["0"] * dimension, intercept="0")
        unknown = ["coefficients", "intercept"]
    else:
        anchors = _matrix(anchor_features)
        _require(len(anchors[0]) == dimension, "Anchor dimension differs")
        names = _names(anchor_ids, len(anchors))
        head.update(anchor_ids=names, anchor_features=ff.canonical(anchors),
            anchor_values=["0"] * len(anchors),
            formula="clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)")
        unknown = ["anchor_values"]
    head_bytes = head_payload_bytes(head)
    return {"schema": PAYLOAD_SCHEMA, "kind": kind, "feature_id": feature_id,
        "class_id": head["class_id"], "dimension": dimension,
        "map_payload_bytes": map_payload_bytes,
        "head_payload_lower_bound_bytes": head_bytes,
        "deployment_payload_lower_bound_bytes": map_payload_bytes + head_bytes,
        "template_head": head, "unknown_fields": unknown,
        "serialization": "UTF8_sorted_compact_ensure_ascii_JSON_plus_one_newline",
        "uses_fitted_unknown_values": False}


def check_realized_head_cost(bound, head):
    """Verify a fitted head matches its prefit template and its proven byte bound."""
    _require(bound.get("schema") == PAYLOAD_SCHEMA, "Wrong byte-bound schema")
    actual = ff.canonical(head)
    template = bound["template_head"]
    _require(isinstance(actual, dict) and set(actual) == set(template),
             "Realized head schema fields differ from prefit template")
    masked = dict(actual)
    for field in bound["unknown_fields"]:
        value = actual[field]
        if isinstance(template[field], list):
            _require(isinstance(value, list) and len(value) == len(template[field]),
                     "Realized unknown field has different shape")
            _require(all(isinstance(x, str) and str(ff.rational(x)) == x for x in value),
                     "Unknown head values must be canonical rational strings")
        else:
            _require(isinstance(value, str) and str(ff.rational(value)) == value,
                     "Unknown head value must be a canonical rational string")
        masked[field] = template[field]
    _require(masked == template, "Known head fields differ from prefit template")
    lower = head_payload_bytes(template)
    _require(lower == bound["head_payload_lower_bound_bytes"] and
             lower + bound["map_payload_bytes"] == bound["deployment_payload_lower_bound_bytes"],
             "Stored byte lower bound is inconsistent")
    actual_bytes = head_payload_bytes(actual)
    _require(actual_bytes >= lower, "Realized head violates minimum payload bytes")
    return {"head_payload_bytes": actual_bytes,
        "deployment_payload_bytes": bound["map_payload_bytes"] + actual_bytes,
        "head_payload_lower_bound_bytes": lower,
        "deployment_payload_lower_bound_bytes": bound["deployment_payload_lower_bound_bytes"],
        "verified": True}


def range_floor(targets, *, ids=None, output_interval=("0", "1")):
    """Exact common range-only lower bound, available equally to both policies."""
    _require(isinstance(targets, (list, tuple)) and len(targets) > 0,
             "Expected a nonempty target roster")
    values = [ff.rational(y) for y in targets]
    names = _names(ids, len(values))
    low, high = _interval(output_interval)
    clipped = [_clip(y, low, high) for y in values]
    floors = [(y - z) ** 2 for y, z in zip(values, clipped)]
    total = sum(floors, F(0))
    return ff.canonical({"schema": "step36-range-only-floor-v1", "n": len(values),
        "query_ids": names, "targets_exact": values, "output_interval": [low, high],
        "clipped_targets_for_floor_only": clipped, "singleton_floor_sse": floors,
        "range_floor_sse": total, "range_floor_mean": total / len(values),
        "targets_clipped_in_loss": False, "target_digest": ff.digest(values)})


def two_point_box_optimum(target_i, target_j, radius, *, output_interval=("0", "1")):
    """Exact squared-loss optimum for two box-valued predictions within radius.

    The returned scalar multiplier proves the same objective by weak duality;
    its equality with the feasible primal objective is checked exactly here.
    Original targets are used in every squared-loss term, including targets
    outside the predictor range.
    """
    yi, yj, r = map(ff.rational, (target_i, target_j, radius))
    _require(r >= 0, "Negative pair constraint radius")
    low, high = _interval(output_interval)
    qi, qj = _clip(yi, low, high), _clip(yj, low, high)
    singleton = (qi - yi) ** 2 + (qj - yj) ** 2
    if abs(qi - qj) <= r:
        vi, vj, signed = qi, qj, F(0)
        branch = "independent_range_projections"
    else:
        i_high = qi > qj
        yh, yl = (yi, yj) if i_high else (yj, yi)
        t0 = (yh + yl - r) / 2
        t = _clip(t0, low, high - r)
        vh, vl = t + r, t
        if t0 <= low:
            multiplier = 2 * (yh - (low + r))
        elif t0 >= high - r:
            multiplier = 2 * ((high - r) - yl)
        else:
            multiplier = yh - yl - r
        _require(multiplier >= 0, "Pair dual multiplier has wrong sign")
        vi, vj = (vh, vl) if i_high else (vl, vh)
        signed = multiplier if i_high else -multiplier
        branch = "active_difference_constraint"
    primal = (vi - yi) ** 2 + (vj - yj) ** 2
    wi, wj = _clip(yi - signed / 2, low, high), _clip(yj + signed / 2, low, high)
    terms = [(wi - yi) ** 2 + signed * wi, (wj - yj) ** 2 - signed * wj]
    penalty = abs(signed) * r
    dual = sum(terms, F(0)) - penalty
    _require(low <= vi <= high and low <= vj <= high and abs(vi - vj) <= r,
             "Pair primal is infeasible")
    _require([wi, wj] == [vi, vj] and dual == primal and primal >= singleton,
             "Exact pair primal-dual equality failed")
    return ff.canonical({"values": [vi, vj], "sse": primal,
        "singleton_floor_sse": singleton, "improvement_sse": primal - singleton,
        "signed_multiplier": signed, "dual_box_minimizers": [wi, wj],
        "dual_coordinate_minima": terms, "dual_penalty": penalty,
        "dual_sse": dual, "branch": branch})


def _approximate_matching(z, y, names, kap, low, high, *, neighbors, max_pairs):
    # Numerical proposals never enter the exact lower-bound arithmetic.
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    n = len(z)
    matrix = np.asarray([[float(x) for x in row] for row in z], dtype=np.float64)
    clipped = [float(_clip(value, low, high)) for value in y]
    kappa_float = float(kap)
    _require(np.isfinite(matrix).all() and all(isfinite(v) for v in clipped)
             and isfinite(kappa_float), "Nonfinite approximate matching input")
    distances = squareform(pdist(matrix, metric="euclidean"))
    _require(np.isfinite(distances).all(), "Nonfinite approximate matching distance")
    nearest, edges = [], set()
    for i in range(n):
        roster = sorted((j for j in range(n) if i != j),
                        key=lambda j: (float(distances[i, j]), names[j], j))[:min(neighbors, n - 1)]
        nearest.append({"index": i, "query_id": names[i], "neighbor_indices": roster,
                        "neighbor_ids": [names[j] for j in roster]})
        edges.update((min(i, j), max(i, j)) for j in roster)
    candidates = []
    for i, j in sorted(edges):
        distance = float(distances[i, j])
        violation = max(0.0, abs(clipped[i] - clipped[j]) - kappa_float * distance)
        _require(isfinite(violation), "Nonfinite approximate violation")
        candidates.append({"indices": [i, j], "query_ids": [names[i], names[j]],
            "approximate_distance_exact_binary64": ff.exact_float(distance),
            "approximate_clipped_target_violation_exact_binary64": ff.exact_float(violation)})
    positive = [edge for edge in candidates
                if ff.rational(edge["approximate_clipped_target_violation_exact_binary64"]) > 0]
    positive.sort(key=lambda edge: (
        -ff.rational(edge["approximate_clipped_target_violation_exact_binary64"]),
        tuple(sorted(edge["query_ids"])), tuple(edge["indices"])))
    matched, selected = set(), []
    for edge in positive:
        i, j = edge["indices"]
        if i not in matched and j not in matched:
            selected.append([i, j])
            matched.update((i, j))
            if len(selected) == max_pairs:
                break
    return {"algorithm": "full_binary64_pdist_stable_kNN_union_positive_violation_greedy_matching",
        "k": neighbors, "max_pairs": max_pairs, "neighbors": nearest,
        "neighbor_tie_order": "approximate_distance_then_neighbor_query_id_then_index",
        "edge_rank_order": "descending_positive_approximate_violation_then_lexically_sorted_query_ids_then_indices",
        "candidate_edges": candidates,
        "ranked_positive_edges": [edge["indices"] for edge in positive],
        "selected_indices": selected,
        "approximate_unordered_distance_evaluations": n * (n - 1) // 2,
        "matching_uses_assessment_targets": True,
        "matching_is_not_part_of_lower_bound_validity_proof": True}


def pair_class_lower(features, targets, *, feature_id, ids=None, kappa="2",
        output_interval=("0", "1"), neighbors=NEIGHBORS, max_pairs=MAX_PAIRS,
        bits=DISTANCE_BITS):
    """Valid all-cohort class lower bound from at most64 disjoint exact pairs."""
    _text(feature_id, "feature identifier")
    _integer(neighbors, "neighbor count", 1, NEIGHBORS)
    _integer(max_pairs, "selected-pair cap", 1, MAX_PAIRS)
    _integer(bits, "distance bits", DISTANCE_BITS, DISTANCE_BITS)
    z = _matrix(features)
    n = len(z)
    _require(isinstance(targets, (list, tuple)) and len(targets) == n,
             "Target roster has wrong length")
    y = [ff.rational(value) for value in targets]
    names = _names(ids, n)
    kap = ff.rational(kappa)
    _require(kap >= 0, "Negative sensitivity")
    low, high = _interval(output_interval)
    floor = range_floor(y, ids=names, output_interval=(low, high))
    matching = _approximate_matching(z, y, names, kap, low, high,
                                    neighbors=neighbors, max_pairs=max_pairs)
    pairs, total_improvement = [], F(0)
    for i, j in matching["selected_indices"]:
        squared = sum(((left - right) ** 2 for left, right in zip(z[i], z[j])), F(0))
        distance = ff.sqrt_enclosure(squared, bits=bits)
        radius = kap * ff.rational(distance["upper"])
        optimum = two_point_box_optimum(y[i], y[j], radius, output_interval=(low, high))
        total_improvement += ff.rational(optimum["improvement_sse"])
        pairs.append({"indices": [i, j], "query_ids": [names[i], names[j]],
            "distance": distance, "radius_upper": str(radius), "optimum": optimum})
    lower_sse = ff.rational(floor["range_floor_sse"]) + total_improvement
    descriptor = ff.class_descriptor(feature_id, kap, (low, high))
    return ff.canonical({"schema": PAIR_SCHEMA,
        "assurance": ASSURANCE, "feature_id": feature_id,
        "class_id": ff.class_identity(feature_id, kap, (low, high)),
        "class_descriptor": descriptor, "n": n, "dimension": len(z[0]),
        "query_ids": names, "features_exact": z, "targets_exact": y,
        "feature_digest": ff.digest(z), "target_digest": ff.digest(y),
        "input_digest": ff.digest({"ids": names, "features": z, "targets": y, "class": descriptor}),
        "kappa": kap, "output_interval": [low, high],
        "range_floor_sse": floor["range_floor_sse"],
        "range_floor_mean": floor["range_floor_mean"],
        "singleton_floor_sse": floor["singleton_floor_sse"],
        "selected_pairs": pairs, "pair_improvement_sse": total_improvement,
        "lower_sse": lower_sse, "lower_mean": lower_sse / n,
        "matching": matching,
        "work": {"approximate_distance_evaluations": n * (n - 1) // 2,
            "exact_selected_pair_distance_evaluations": len(pairs),
            "selected_pairs": len(pairs), "distance_bits": bits,
            "max_pairs": max_pairs, "neighbors": neighbors,
            "full_approximate_distance_matrix_computed": True,
            "exact_distance_cap_is_not_a_wall_time_bound": True},
        "witness_uses_assessment_targets": True, "targets_clipped_in_loss": False,
        "targets_outside_predictor_range": sum(not low <= value <= high for value in y),
        "native_transform_error_enclosed": False, "native_pipeline_roundoff_certified": False,
        "population_risk_claim": False, "future_response_coverage_claim": False,
        "raw_input_lipschitz_claim": False, "deployable_head_returned": False})


def candidate_key(row):
    """Frozen eligible-candidate order: bytes, exact risk upper, lexical ID."""
    _integer(row["deployment_payload_bytes"], "candidate payload bytes")
    _text(row["candidate_id"], "candidate identifier")
    lower = ff.rational(row["decision_risk"]["lower"])
    upper = ff.rational(row["decision_risk"]["upper"])
    _require(0 <= lower <= upper, "Invalid candidate risk enclosure")
    if "width" in row["decision_risk"]:
        _require(ff.rational(row["decision_risk"]["width"]) == upper - lower,
                 "Incorrect candidate risk enclosure width")
    return row["deployment_payload_bytes"], upper, row["candidate_id"]


def map_order(map_payload_bytes):
    """Pre-fit map order; input is a name -> known encoder-byte-count mapping."""
    for name, count in map_payload_bytes.items():
        _text(name, "map name")
        _integer(count, "map payload bytes")
    return sorted(map_payload_bytes, key=lambda name: (map_payload_bytes[name], name))


class PolicyState:
    """Online state with a fixed target tolerance and at most six pair calls.

    The caller must first offer each map's cheap affine head, then its extension.
    Head fits, exact evaluation, and map generation remain with the caller.
    Pair receipts are lower bounds for the entire class; the only extension
    omission using them is lower_mean > fixed epsilon, strictly.
    """
    def __init__(self, epsilon, *, mode="ordinary", max_certificate_calls=MAX_CERTIFICATE_CALLS):
        self.epsilon = ff.rational(epsilon)
        _require(self.epsilon >= 0, "Negative fixed tolerance")
        _require(mode in ("ordinary", "triggered"), "Unknown policy mode")
        _integer(max_certificate_calls, "certificate-call cap", 0, MAX_CERTIFICATE_CALLS)
        self.mode = mode
        self.max_certificate_calls = max_certificate_calls
        self.certificate_calls = 0
        self.exact_pair_distances = 0
        self.approximate_pair_distances = 0
        self._incumbent = None
        self._events = []
        self._observed = {}
        self._pending_certificate = None
        self._range_floor = None

    def _summary(self):
        if self._incumbent is None:
            return None
        row = self._incumbent
        return {key: row[key] for key in
                ("candidate_id", "map", "recipe", "deployment_payload_bytes", "decision_risk", "head_id")}

    def _event(self, event):
        result = ff.canonical({"sequence": len(self._events), **event})
        self._events.append(result)
        return result

    def set_range_floor(self, receipt):
        _require(self._range_floor is None and not self._events,
                 "The common range floor must be set before policy actions")
        _require(receipt.get("schema") == "step36-range-only-floor-v1", "Wrong range-floor schema")
        lower = ff.rational(receipt["range_floor_mean"])
        _require(lower >= 0, "Negative range floor")
        self._range_floor = lower
        return self._event({"kind": "range_floor", "lower_mean": lower,
            "epsilon": self.epsilon, "prunes_all": lower > self.epsilon,
            "range_floor_digest": ff.digest(receipt), "shared_by_both_policies": True})

    def range_floor_prunes(self):
        return self._range_floor is not None and self._range_floor > self.epsilon

    def _cost_prunes(self, bound):
        _require(bound.get("schema") == PAYLOAD_SCHEMA, "Wrong byte-bound schema")
        lower = _integer(bound["deployment_payload_lower_bound_bytes"], "deployment byte lower bound")
        return self._incumbent is not None and lower > self._incumbent["deployment_payload_bytes"]

    def affine_decision(self, map_name, bound):
        _require(self._pending_certificate is None, "Unconsumed class-certificate call")
        _text(map_name, "map name")
        _require(bound["kind"] == "clipped_affine", "Affine decision received a different recipe")
        prune = self._cost_prunes(bound)
        return self._event({"kind": "affine_decision", "map": map_name,
            "action": "COST_PRUNE" if prune else "FIT",
            "reason": "STRICT_BYTE_LOWER_BOUND" if prune else "CHEAP_AFFINE_FIRST",
            "bound": bound, "incumbent_before": self._summary(),
            "certificate_calls_before": self.certificate_calls})

    def observe(self, row):
        _require(self._pending_certificate is None, "Consume certificate before observing a head")
        _text(row["map"], "map name")
        _text(row["recipe"], "recipe")
        _text(row["head_id"], "head identifier")
        key = candidate_key(row)
        _require(row["candidate_id"] not in self._observed, "Candidate observed twice")
        record = ff.canonical(row)
        before = self._summary()
        eligible = key[1] <= self.epsilon
        updated = eligible and (self._incumbent is None or key < candidate_key(self._incumbent))
        self._observed[row["candidate_id"]] = record
        if updated:
            self._incumbent = record
        return self._event({"kind": "observe", "row": record, "eligible": eligible,
            "incumbent_before": before, "incumbent_after": self._summary(),
            "incumbent_updated": updated})

    def extension_decision(self, map_name, bound, affine_row_or_none=None):
        _require(self._pending_certificate is None, "Unconsumed class-certificate call")
        _text(map_name, "map name")
        _require(bound["kind"] == "symmetric_lipschitz_extension", "Extension decision received a different recipe")
        before_calls = self.certificate_calls
        affine_id = None
        affine_eligible = None
        if affine_row_or_none is not None:
            affine_id = affine_row_or_none["candidate_id"]
            _require(affine_id in self._observed and
                     self._observed[affine_id] == ff.canonical(affine_row_or_none) and
                     affine_row_or_none["map"] == map_name and affine_row_or_none["recipe"] == "affine",
                     "Affine trigger premise was not observed for this map")
            affine_eligible = candidate_key(affine_row_or_none)[1] <= self.epsilon
        if self.range_floor_prunes():
            action, reason = "RANGE_PRUNE", "COMMON_RANGE_FLOOR_EXCEEDS_FIXED_TOLERANCE"
        elif self._cost_prunes(bound):
            action, reason = "COST_PRUNE", "STRICT_BYTE_LOWER_BOUND"
        elif affine_eligible:
            action, reason = "FIT", "AFFINE_ALREADY_VERIFIED_ADEQUATE"
        elif self.mode == "ordinary":
            action, reason = "FIT", "ORDINARY_SEARCH_NO_PAIR_SCREEN"
        elif affine_row_or_none is None:
            action, reason = "FIT", "AFFINE_FEASIBILITY_NOT_OBSERVED"
        elif self.certificate_calls >= self.max_certificate_calls:
            action, reason = "FIT", "PAIR_CERTIFICATE_CALL_BUDGET_EXHAUSTED"
        else:
            action, reason = "CERTIFY", "AFFINE_NOT_VERIFIED_ADEQUATE_AND_EXTENSION_COST_COMPETITIVE"
            self.certificate_calls += 1
            self._pending_certificate = {"map": map_name, "bound": bound,
                                         "call_number": self.certificate_calls}
        return self._event({"kind": "extension_decision", "map": map_name,
            "action": action, "reason": reason, "bound": bound,
            "incumbent_before": self._summary(), "affine_candidate_id": affine_id,
            "affine_verified_adequate": affine_eligible,
            "certificate_calls_before": before_calls,
            "certificate_calls_after": self.certificate_calls})

    def observe_certificate(self, map_name, certificate):
        pending = self._pending_certificate
        _require(pending is not None and pending["map"] == map_name,
                 "No certificate call is pending for this map")
        _require(certificate.get("schema") == PAIR_SCHEMA and
                 certificate["feature_id"] == pending["bound"]["feature_id"] and
                 certificate["class_id"] == pending["bound"]["class_id"],
                 "Class certificate is not bound to the pending head class")
        work = certificate["work"]
        exact_count = _integer(work["exact_selected_pair_distance_evaluations"],
                               "exact pair distance count", 0, MAX_PAIRS)
        approximate_count = _integer(work["approximate_distance_evaluations"],
                                     "approximate pair distance count")
        _require(exact_count == len(certificate["selected_pairs"]) and
                 work["distance_bits"] == DISTANCE_BITS and
                 work["neighbors"] == NEIGHBORS and work["max_pairs"] == MAX_PAIRS,
                 "Class certificate used a different frozen work recipe")
        lower = ff.rational(certificate["lower_mean"])
        _require(lower >= 0, "Negative class-risk lower bound")
        self.exact_pair_distances += exact_count
        self.approximate_pair_distances += approximate_count
        _require(self.exact_pair_distances <= self.max_certificate_calls * MAX_PAIRS,
                 "Global selected-pair work cap exceeded")
        self._pending_certificate = None
        return self._event({"kind": "certificate_result", "map": map_name,
            "action": "CLASS_PRUNE" if lower > self.epsilon else "FIT",
            "reason": "PAIR_LOWER_EXCEEDS_FIXED_TOLERANCE" if lower > self.epsilon else "PAIR_LOWER_DOES_NOT_EXCLUDE",
            "call_number": pending["call_number"], "lower_mean": lower,
            "epsilon": self.epsilon, "feature_id": certificate["feature_id"],
            "class_id": certificate["class_id"], "certificate_digest": ff.digest(certificate),
            "exact_selected_pair_distance_evaluations": exact_count,
            "approximate_distance_evaluations": approximate_count,
            "incumbent_before": self._summary()})

    def selection(self):
        _require(self._pending_certificate is None, "Unconsumed class-certificate call")
        return None if self._incumbent is None else dict(self._incumbent)

    def trace(self):
        _require(self._pending_certificate is None, "Unconsumed class-certificate call")
        return ff.canonical({"schema": POLICY_SCHEMA, "mode": self.mode,
            "epsilon": self.epsilon,
            "eligibility_rule": "exact_decision_risk_upper_le_fixed_epsilon",
            "selection_order": ["deployment_payload_bytes", "exact_decision_risk_upper", "candidate_id_lexical"],
            "cost_pruning_rule": "prefit_byte_lower_bound_strictly_gt_incumbent_actual_bytes",
            "max_certificate_calls": self.max_certificate_calls,
            "certificate_calls": self.certificate_calls,
            "max_pairs_per_call": MAX_PAIRS,
            "exact_selected_pair_distance_evaluations": self.exact_pair_distances,
            "approximate_distance_evaluations": self.approximate_pair_distances,
            "exact_selected_pair_distance_cap": self.max_certificate_calls * MAX_PAIRS,
            "wall_time_regret_bound_claim": False,
            "events": list(self._events), "selection": self.selection()})
