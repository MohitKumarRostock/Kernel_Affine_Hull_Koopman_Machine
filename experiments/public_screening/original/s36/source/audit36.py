"""Independent Step36 audit; scientific producers are never imported for proof checks.

This module reuses the byte-identical Step35 independent arithmetic checker for
the retained generic head and prediction formats. Step36 pair witnesses are
checked by a separate exact projection onto the two-point feasible polygon.
Dataset, policy and provenance checks are added once the frozen schema is set.
Preflight/selfchecks must never inspect new public numerical values.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
import hashlib
import io
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import traceback
import zipfile

for _thread_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_key] = "1"

import independent_math35 as old


ROOT = Path(__file__).resolve().parents[1]
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
INDEPENDENT_MATH_SHA256 = "3f440f427614ac230f4a09ac062d5bfc18eaa47babdaaff1c3c9b04b2c330308"


class AuditError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def q(value):
    require(not isinstance(value, bool), "Boolean used as a rational")
    return F.from_float(value) if isinstance(value, float) else F(value)


def canon(value):
    return old.canon(value)


def digest(value):
    return old.digest(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return old.read_json(path)


def write_json(path, value):
    return old.write_json(path, value)


def interval(record, expected=None):
    return old.interval(record, expected)


def exact_pair_polygon(target_i, target_j, radius, output_interval=("0", "1")):
    """Independent exact closest point in the 2D box/difference polygon.

    Unlike the producer's clipping-based closed form, this verifier enumerates
    every possible minimizing face: the interior target, orthogonal projections
    onto six supporting lines, and intersections of each nonparallel line pair.
    A strictly convex squared-distance objective has its unique minimizer among
    these feasible candidates, including the degenerate radius-zero segment.
    """
    yi, yj, r = q(target_i), q(target_j), q(radius)
    lo, hi = map(q, output_interval)
    require(lo < hi and r >= 0, "Invalid pair box or radius")
    # Each inequality is a*x+b*y <= c.
    inequalities = [(F(1), F(0), hi), (F(-1), F(0), -lo),
                    (F(0), F(1), hi), (F(0), F(-1), -lo),
                    (F(1), F(-1), r), (F(-1), F(1), r)]
    candidates = set()

    def add(x, y):
        if all(a*x+b*y <= c for a, b, c in inequalities):
            candidates.add((x, y))

    add(yi, yj)
    for a, b, c in inequalities:
        step = (a*yi+b*yj-c)/(a*a+b*b)
        add(yi-step*a, yj-step*b)
    for (a, b, c), (d, e, f) in combinations(inequalities, 2):
        determinant = a*e-b*d
        if determinant:
            add((c*e-b*f)/determinant, (a*f-c*d)/determinant)
    require(candidates, "Pair polygon has no feasible candidate")
    selected = min(candidates, key=lambda v: ((v[0]-yi)**2+(v[1]-yj)**2, v))
    sse = (selected[0]-yi)**2+(selected[1]-yj)**2
    return {"values": list(selected), "sse": sse,
            "candidate_count": len(candidates), "method": "exact_polygon_face_projection"}


def deployed_json_bytes(value):
    """The frozen compact UTF-8 JSON format used for deployed head payloads."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def audit_head_cost_bound(bound, head=None):
    """Verify fixed schema fields and the minimal unknown rational-string sizes."""
    require(bound["schema"] == "step36-prefit-head-payload-bound-v1" and
            bound["uses_fitted_unknown_values"] is False,
            "Wrong byte lower-bound schema or information boundary")
    template = bound["template_head"]
    kind, dimension = bound["kind"], bound["dimension"]
    require(type(dimension) is int and dimension > 0 and
            type(bound["map_payload_bytes"]) is int and bound["map_payload_bytes"] >= 0,
            "Invalid prefit dimension or map bytes")
    require(template["schema"] == "step35-stored-feature-head-v1" and
            template["kind"] == kind and template["dimension"] == dimension and
            template["feature_id"] == bound["feature_id"] and
            template["class_id"] == bound["class_id"] and template["assurance"] == ASSURANCE,
            "Prefit known head schema fields differ")
    cls = old.descriptor(template["feature_id"], q(template["kappa"]),
                         tuple(map(q, template["output_interval"])))
    require(template["class_id"] == "class35:"+digest(cls),
            "Prefit class identity differs")
    common = {"schema", "kind", "feature_id", "class_id", "dimension", "kappa",
              "output_interval", "assurance"}
    if kind == "clipped_affine":
        require(bound["unknown_fields"] == ["coefficients", "intercept"] and
                template["coefficients"] == ["0"]*dimension and template["intercept"] == "0" and
                set(template) == common | {"coefficients", "intercept"},
                "Affine minimal-byte template differs")
    else:
        require(kind == "symmetric_lipschitz_extension" and
                bound["unknown_fields"] == ["anchor_values"] and
                set(template) == common | {"anchor_ids", "anchor_features", "anchor_values", "formula"},
                "Extension minimal-byte template differs")
        anchors = old.matrix(template["anchor_features"])
        require(len(anchors[0]) == dimension and
                len(template["anchor_ids"]) == len(set(template["anchor_ids"])) == len(anchors) and
                template["anchor_values"] == ["0"]*len(anchors) and
                template["formula"] == "clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)",
                "Known extension anchor roster, formula or unknown-value template differs")
    lower = len(deployed_json_bytes(template))
    require(bound["serialization"] == "UTF8_sorted_compact_ensure_ascii_JSON_plus_one_newline" and
            bound["head_payload_lower_bound_bytes"] == lower and
            bound["deployment_payload_lower_bound_bytes"] == lower+bound["map_payload_bytes"],
            "Canonical prefit byte count differs")
    if head is not None:
        require(set(head) == set(template), "Realized deployment head changes schema")
        masked = dict(head)
        for field in bound["unknown_fields"]:
            values = head[field] if isinstance(template[field], list) else [head[field]]
            if isinstance(template[field], list):
                require(isinstance(head[field], list) and len(head[field]) == len(template[field]),
                        "Realized unknown-value shape differs")
            require(all(isinstance(v, str) and len(v) >= 1 and str(q(v)) == v for v in values),
                    "Head parameter is not a nonempty canonical rational string")
            masked[field] = template[field]
        require(masked == template and len(deployed_json_bytes(head)) >= lower,
                "Realized head violates its prefit template bound")
    return {"kind": kind, "head_lower_bytes": lower,
            "deployment_lower_bytes": bound["deployment_payload_lower_bound_bytes"]}


def audit_range_floor(record):
    require(record["schema"] == "step36-range-only-floor-v1" and
            record["targets_clipped_in_loss"] is False,
            "Wrong common range-floor schema")
    y = list(map(q, record["targets_exact"]))
    low, high = map(q, record["output_interval"])
    require(low < high and len(y) == record["n"] == len(record["query_ids"]) and
            len(set(record["query_ids"])) == len(y), "Range-floor source roster differs")
    clipped = [old.clip(v, low, high) for v in y]
    floors = [(v-w)**2 for v, w in zip(y, clipped)]
    total = sum(floors, F(0))
    require(record["clipped_targets_for_floor_only"] == list(map(str, clipped)) and
            record["singleton_floor_sse"] == list(map(str, floors)) and
            q(record["range_floor_sse"]) == total and
            q(record["range_floor_mean"]) == total/len(y) and record["target_digest"] == digest(y),
            "Range-floor exact arithmetic differs")
    return total/len(y)


def audit_matching(record, z, y, ids, kap, low, high):
    """Reproduce the frozen floating proposal rule; separate from the exact proof."""
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    n = len(z)
    matching = record["matching"]
    k, cap = matching["k"], matching["max_pairs"]
    require(matching["algorithm"] == "full_binary64_pdist_stable_kNN_union_positive_violation_greedy_matching",
            "Pair proposal algorithm differs")
    float_z = np.asarray([[float(v) for v in row] for row in z], dtype=np.float64)
    distances = squareform(pdist(float_z, metric="euclidean"))
    require(np.isfinite(distances).all(), "Nonfinite matching distance")
    neighbors, edges = [], set()
    for i in range(n):
        nearest = sorted((j for j in range(n) if i != j),
                         key=lambda j: (float(distances[i, j]), ids[j], j))[:min(k, n-1)]
        neighbors.append({"index": i, "query_id": ids[i], "neighbor_indices": nearest,
                          "neighbor_ids": [ids[j] for j in nearest]})
        edges.update((min(i, j), max(i, j)) for j in nearest)
    require(matching["neighbors"] == neighbors, "Frozen nearest-neighbor proposal roster differs")
    clipped = [float(old.clip(v, low, high)) for v in y]
    candidates = []
    for i, j in sorted(edges):
        distance = float(distances[i, j])
        violation = max(0.0, abs(clipped[i]-clipped[j])-float(kap)*distance)
        candidates.append({"indices": [i, j], "query_ids": [ids[i], ids[j]],
            "approximate_distance_exact_binary64": str(F.from_float(distance)),
            "approximate_clipped_target_violation_exact_binary64": str(F.from_float(violation))})
    require(matching["candidate_edges"] == candidates, "Approximate candidate edge values differ")
    positive = [r for r in candidates if q(r["approximate_clipped_target_violation_exact_binary64"]) > 0]
    positive.sort(key=lambda r: (-q(r["approximate_clipped_target_violation_exact_binary64"]),
                                tuple(sorted(r["query_ids"])), tuple(r["indices"])))
    require(matching["ranked_positive_edges"] == [r["indices"] for r in positive],
            "Frozen approximate violation ranking differs")
    selected, used = [], set()
    for edge in positive:
        i, j = edge["indices"]
        if i not in used and j not in used:
            selected.append([i, j]); used.update((i, j))
            if len(selected) == cap:
                break
    require(matching["selected_indices"] == selected and
            matching["approximate_unordered_distance_evaluations"] == n*(n-1)//2 and
            record["work"]["approximate_distance_evaluations"] == n*(n-1)//2,
            "Greedy matching, cap, or charged floating-distance work differs")
    return len(selected)


def audit_pair_certificate(record):
    """Check a whole-cohort lower witness without trusting pair optimization."""
    require(record["schema"] == "step36-disjoint-pair-class-lower-v1",
            "Unexpected pair-certificate schema")
    z = old.matrix(record["features_exact"])
    ids = record["query_ids"]
    y = list(map(q, record["targets_exact"]))
    n, d = len(z), len(z[0])
    require(n == len(y) == len(ids) and len(set(ids)) == n,
            "Pair certificate source roster is inconsistent")
    kap = q(record["kappa"])
    low, high = map(q, record["output_interval"])
    require(kap >= 0 and low < high, "Invalid pair decoder class")
    descriptor = old.descriptor(record["feature_id"], kap, (low, high))
    require(record["class_id"] == "class35:" + digest(descriptor) and
            record["class_descriptor"] == descriptor and
            record["feature_digest"] == digest(z) and record["target_digest"] == digest(y) and
            record["input_digest"] == digest({"ids": ids, "features": z, "targets": y, "class": descriptor}) and
            record["n"] == n and record["dimension"] == d,
            "Pair certificate class identity differs")
    floor_rows = [(old.clip(t, low, high)-t)**2 for t in y]
    floor_sse = sum(floor_rows, F(0))
    require(record["singleton_floor_sse"] == list(map(str, floor_rows)) and
            q(record["range_floor_sse"]) == floor_sse and
            q(record["range_floor_mean"]) == floor_sse/n,
            "Whole-cohort range floor differs")
    used, improvements, selection = set(), F(0), []
    for pair in record["selected_pairs"]:
        i, j = pair["indices"]
        require(type(i) is int and type(j) is int and 0 <= i < j < n,
                "Invalid ordered pair indices")
        require(i not in used and j not in used, "Selected pairs are not disjoint")
        used.update((i, j)); selection.append([i, j])
        require(pair["query_ids"] == [ids[i], ids[j]], "Selected pair identity differs")
        dist = pair["distance"]
        squared = sum(((a-b)**2 for a, b in zip(z[i], z[j])), F(0))
        lower, upper, integer_floor = old.exact_sqrt(squared, dist["bits"])
        require(dist["bits"] == 80 and q(dist["squared_exact"]) == squared and
                q(dist["lower"]) == lower and q(dist["upper"]) == upper and
                q(dist["width"]) == upper-lower and
                int(dist["scaled_sqrt_floor"]) == integer_floor,
                "Selected pair outward distance witness differs")
        radius = kap*upper
        require(q(pair["radius_upper"]) == radius,
                "Pair relaxation used an incorrect distance endpoint")
        optimum = exact_pair_polygon(y[i], y[j], radius, (low, high))
        retained = pair["optimum"]
        retained_values = list(map(q, retained["values"]))
        require(retained_values == optimum["values"] and
                q(retained["sse"]) == optimum["sse"],
                "Analytic pair optimum differs from independent polygon projection")
        singleton_floor = floor_rows[i]+floor_rows[j]
        improvement = optimum["sse"]-singleton_floor
        require(improvement >= 0 and
                q(retained["singleton_floor_sse"]) == singleton_floor and
                q(retained["improvement_sse"]) == improvement,
                "Pair improvement or singleton-floor accounting differs")
        # A signed multiplier imposes s*(v_i-v_j) <= |s|*radius.
        # The box is minimized independently and original targets are retained.
        multiplier = q(retained["signed_multiplier"])
        vi = old.clip(y[i]-multiplier/2, low, high)
        vj = old.clip(y[j]+multiplier/2, low, high)
        dual_sse = ((vi-y[i])**2+multiplier*vi+
                    (vj-y[j])**2-multiplier*vj-abs(multiplier)*radius)
        dual_terms = [(vi-y[i])**2+multiplier*vi, (vj-y[j])**2-multiplier*vj]
        require(q(retained["dual_sse"]) == dual_sse == optimum["sse"] and
                retained["dual_box_minimizers"] == list(map(str, [vi, vj])) and
                retained["dual_coordinate_minima"] == list(map(str, dual_terms)) and
                q(retained["dual_penalty"]) == abs(multiplier)*radius,
                "Pair scalar dual does not prove the retained optimum")
        improvements += improvement
    lower_sse = floor_sse+improvements
    require(q(record["pair_improvement_sse"]) == improvements and
            q(record["lower_sse"]) == lower_sse and
            q(record["lower_mean"]) == lower_sse/n,
            "Disjoint pair bound double counts floors or uses the wrong denominator")
    matching, work = record["matching"], record["work"]
    require(type(matching["max_pairs"]) is int and 0 <= matching["max_pairs"] <= 64 and
            type(matching["k"]) is int and 1 <= matching["k"] <= 4,
            "Pair-matching budget is invalid")
    require(len(selection) <= min(matching["max_pairs"], n//2) and matching["selected_indices"] == selection,
            "Matching roster or pair budget differs")
    require(work["exact_selected_pair_distance_evaluations"] == work["selected_pairs"] == len(selection) and
            work["distance_bits"] == 80 and work["max_pairs"] == matching["max_pairs"] and
            work["neighbors"] == matching["k"] and
            work["full_approximate_distance_matrix_computed"] is True and
            work["exact_distance_cap_is_not_a_wall_time_bound"] is True,
            "Exact pair distance count differs")
    audit_matching(record, z, y, ids, kap, low, high)
    for flag in ("population_risk_claim", "future_response_coverage_claim",
                 "native_transform_error_enclosed", "native_pipeline_roundoff_certified",
                 "raw_input_lipschitz_claim", "deployable_head_returned", "targets_clipped_in_loss"):
        if flag in record:
            require(record[flag] is False, "Unsupported pair assurance: "+flag)
    if "assurance" in record:
        require(record["assurance"] == ASSURANCE, "Pair assurance scope differs")
    require(record["targets_outside_predictor_range"] == sum(not low <= v <= high for v in y),
            "Unclipped target range count differs")
    return {"n": n, "dimension": d, "class_id": record["class_id"],
            "pair_count": len(selection), "range_floor": floor_sse/n,
            "pair_improvement": improvements/n, "lower": lower_sse/n}


def audit_protocol(p):
    require(p["step"] == 36 and p["seed"] == 3601 and
            p["tasks"] == ["casual", "registered", "cnt"],
            "Frozen Step36 task roster or seed differs")
    data = p["data"]
    expected_data = {"uci_id": 275, "member": "hour.csv", "origin_hour": 23,
        "horizon_hours": 24, "lag_hours": 24, "partition_basis": "target timestamp",
        "train_start": "2011-01-01", "train_end": "2011-12-31",
        "decision_start": "2012-01-01", "decision_end": "2012-06-30",
        "test_start": "2012-07-01", "test_end": "2012-12-31",
        "weather_calendar_or_other_count_features": False,
        "use_future_observed_weather": False, "common_roster_across_targets": True,
        "rolling_observed_history": True, "targets_clipped": False,
        "minimum_training_rows": 32}
    require(all(data[k] == v for k, v in expected_data.items()),
            "Frozen causal public-data task or chronology differs")
    exposure = p["pre_freeze_exposure"]
    require(exposure["programmatic_measurement_parsing"] is False and
            exposure["public_model_fits"] == 0 and
            exposure["incidental_training_snippet_seen"] is True and
            exposure["decision_or_test_measurements_seen"] is False and
            exposure["broad_all_values_unseen_claim"] is False,
            "Pre-freeze exposure disclosure differs")
    expected_representations = {"dimension": 24, "scalings": ["minmax01", "standard"],
        "nystrom_widths": [8, 24], "kahm_caps": [4, 12, 20], "kahm_prefixes": [1, 2],
        "cluster_counts": [8, 16], "max_fit": 512, "max_references": 24,
        "tau": 1e-6, "omega": 2.0, "probe_count": 32, "nystrom_eigen_floor": 1e-12,
        "temporal_selection": "equal_row_stratum_midpoints",
        "metric_normalization": "reciprocal_median_positive_pair_distance"}
    require(p["representations"] == expected_representations,
            "Unchanged 18-map factory recipe differs")
    cls, heads, cert = p["class"], p["heads"], p["certificate"]
    require(cls["kappa"] == "2" and cls["prediction_interval"] == ["0", "1"] and
            cls["assurance"] == ASSURANCE,
            "Frozen decoder class or conditional assurance differs")
    require(heads["recipes"] == ["affine", "extension32"] and
            q(heads["ridge_lambda"]) == F(1, 1000000) and heads["anchor_count"] == 32 and
            heads["extension_max_coordinate_attempts"] == 4000 and
            heads["multiplier_grid_bits"] == 40 and heads["check_every"] == 20 and
            q(heads["gap_sse"]) == F(1, 10000000),
            "Frozen training head recipes or optimization budget differs")
    require(cert["distance_bits"] == 80 and cert["neighbors"] == 4 and
            cert["max_pairs_per_call"] == 64 and cert["max_calls_per_target_run"] == 6 and
            cert["approximate_distance_work_counted"] is True and
            cert["no_full_block_QP_escalation"] is True,
            "Triggered certificate arithmetic or work budget differs")
    require(q(p["selection"]["epsilon_factor"]) == F(19, 20) and
            p["selection"]["tolerance_not_tightened_from_step35_outcomes"] is True and
            p["selection"]["no_eligible_action"] == "ABSTAIN",
            "Threshold or abstention recipe differs")
    require(p["search"]["head_order"] == ["affine", "extension32"] and
            p["search"]["modes"] == ["exhaustive", "ordinary", "triggered"] and
            p["search"]["no_retrospective_costs_or_head_cache"] is True,
            "Search order or counterfactual information boundary differs")
    timing = p["timing"]
    require(timing["ordinary_triggered_repetitions"] == 5 and
            timing["exhaustive_repetitions"] == 1 and
            timing["exhaustive_executed_first_per_target"] is True and
            timing["canonical_results_unavailable_to_timed_policy"] is True and
            timing["independent_head_fits_each_execution"] is True and
            timing["no_deployment_latency_or_runtime_size_claim"] is True,
            "Frozen matched timing counterfactual differs")
    execution = p["execution"]
    require(execution["primary_experimental_runs"] == 1 and
            execution["predeclared_timing_repetitions_are_not_adaptive_refits"] is True and
            execution["overall_wall_budget_seconds"] == 3600 and execution["threads"] == 1 and
            execution["reserved_patient_arrays_read"] is False and
            execution["reserved_patients_accessed"] == 0 and
            execution["reserved_patients_total"] == 5557 and
            execution["inherited_study_statuses"] == {"G3": "NOT_ESTABLISHED", "G4": "NOT_STARTED", "natural_study": "NO_GO"},
            "Execution budget or inherited sealed-study boundary differs")
    for flag in ("native_transform_error_enclosed", "population_risk_claim", "future_response_coverage_claim"):
        require(cls[flag] is False, "Unsupported class assurance: "+flag)
    require(p["report"]["all108_canonical_head_outcomes_retained"] is True and
            p["report"]["test_no_reselection"] is True and
            p["report"]["single_external_dataset_with_correlated_targets"] is True and
            p["report"]["keep_step35_negative_result"] is True and
            p["report"]["generic_safe_screening_novelty_claim"] is False,
            "Frozen reporting scope differs")
    return {"one_external_dataset_three_correlated_targets": True,
        "full_24hour_causal_histories_and_target_timestamp_split": True,
        "same_18_maps_and_two_heads_per_map": True,
        "ordinary_cost_and_range_rules_matched": True,
        "strict_bounds_preserve_equal_cost_candidates": True,
        "incremental_fits_and_paired_timing_both_required_for_utility": True,
        "stored_feature_assurance_only": True,
        "incidental_training_snippet_honestly_disclosed": True,
        "reserved_patient_cohort_remains_sealed": True}


def audit_cohorts(root, protocol):
    """Independently rebuild each causal task only after all model/test seals."""
    import numpy as np
    source = read_json(root/"inputs/public_source36.json")
    archive = root/protocol["data"]["archive"]
    require(sha(archive) == source["archive_sha256"], "Public archive identity differs")
    observed, ordered_stamps = {}, []
    missing_tokens = {"", "na", "nan", "+nan", "-nan", "null", "none",
                      "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    targets = protocol["tasks"]
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open("hour.csv") as handle:
            for row in csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig")):
                stamp = datetime.fromisoformat(row["dteday"])+timedelta(hours=int(row["hr"]))
                require(stamp not in observed and (not ordered_stamps or stamp > ordered_stamps[-1]),
                        "Public source timestamps duplicate or change order")
                values = []
                for target in targets:
                    token = row[target]
                    if token.strip().lower() in missing_tokens:
                        values.append(None)
                    else:
                        value = int(token)
                        require(str(value) == token and 0 <= value <= 2**53,
                                "Source count is not an exact nonnegative float64 integer")
                        values.append(value)
                if all(v is not None for v in values):
                    require(values[0]+values[1] == values[2], "Algebraic target relation differs")
                observed[stamp] = values
                ordered_stamps.append(stamp)
    metadata = read_json(root/"inputs/metadata_preflight36.json")
    require(metadata["timestamp_rows"] == len(observed) and
            metadata["all_timestamp_sha256"] == hashlib.sha256(deployed_json_bytes(
                [s.isoformat(timespec="hours") for s in ordered_stamps])).hexdigest() and
            metadata["archive_sha256"] == source["archive_sha256"],
            "Full postrun source differs from timestamp-only preflight")
    cohorts = {target: {} for target in targets}
    expected = {target: {} for target in targets}
    numeric_rows = {}
    for role in ("train", "decision", "test"):
        first = datetime.fromisoformat(protocol["data"][role+"_start"])
        stop = datetime.fromisoformat(protocol["data"][role+"_end"])+timedelta(days=1)
        numeric_rows[role] = sum(first <= s < stop for s in observed)
        daily = [first+timedelta(days=i, hours=23) for i in range((stop-first).days)]
        selected, origins, lag_stamps = [], [], []
        absent = {"required_timestamp_absent": 0, "missing_lag_or_target": 0}
        details = {"target_absent": 0, "lag_absent_with_target_present": 0}
        for target_stamp in daily:
            origin = target_stamp-timedelta(hours=24)
            lags = [origin-timedelta(hours=23-j) for j in range(24)]
            required = lags+[target_stamp]
            if any(s not in observed for s in required):
                absent["required_timestamp_absent"] += 1
                details["target_absent" if target_stamp not in observed else "lag_absent_with_target_present"] += 1
                continue
            if any(any(v is None for v in observed[s]) for s in required):
                absent["missing_lag_or_target"] += 1
                continue
            selected.append(target_stamp); origins.append(origin); lag_stamps.append(lags)
        ids = [role+":"+s.isoformat(timespec="hours") for s in selected]
        target_times = [s.isoformat(timespec="hours") for s in selected]
        origin_times = [s.isoformat(timespec="hours") for s in origins]
        for column, target in enumerate(targets):
            saved = read_json(root/"output/cohorts"/target/(role+".json"))
            raw_y = [observed[s][column] for s in selected]
            X = np.asarray([[observed[s][column] for s in stamps] for stamps in lag_stamps], dtype=np.float64)
            require(saved["name"] == role and saved["target"] == target and
                    saved["ids"] == ids and saved["origin_timestamps"] == origin_times and
                    saved["target_timestamps"] == target_times and saved["raw_targets"] == raw_y,
                    "Independent common causal roster differs: "+target+"/"+role)
            lag_path = root/saved["lag_matrix_path"]
            require(sha(lag_path) == saved["lag_matrix_sha256"], "Saved lag matrix hash differs")
            retained = np.load(lag_path, allow_pickle=False)
            require(retained.dtype == np.float64 and np.array_equal(retained, X),
                    "Saved own-target lag matrix differs from independent source reconstruction")
            roster = saved["roster"]
            require(roster["candidate_daily_targets"] == len(daily) and
                    roster["eligible"] == len(selected) and roster["excluded"] == absent and
                    roster["absent_timestamp_details"] == details and
                    roster["target_id_sha256"] == hashlib.sha256(deployed_json_bytes(target_times)).hexdigest() and
                    roster["common_three_target_roster"] is True and
                    roster["own_target_lags_only"] is True and roster["lag_dimension"] == 24 and
                    roster["origin_hour"] == 23 and roster["horizon_hours"] == 24,
                    "Calendar complete-case accounting differs")
            cohorts[target][role] = saved
            expected[target][role] = {"X": X, "raw_targets": raw_y}
    development = read_json(root/"output/development_data36.json")
    test = read_json(root/"output/test_data36.json")
    require(development["numeric_rows_by_partition"] == {**numeric_rows, "test": 0} and
            development["test_numeric_values_parsed"] is False and
            development["test_missingness_inspected"] is False and
            development["post_development_payload_skipped_uninterpreted"] is True and
            test["numeric_rows_by_partition"] == numeric_rows and
            test["test_numeric_values_parsed"] is True and test["test_missingness_inspected"] is True and
            development["normalizations"] == test["normalizations"],
            "Declared numeric read barrier or unchanged normalization differs")
    epsilons, scales = {}, {}
    for target in targets:
        raw_train = expected[target]["train"]["raw_targets"]
        require(len(raw_train) >= 32, "Frozen minimum training sample count not met")
        low, high = min(raw_train), max(raw_train)
        span = high-low
        require(span > 0, "Degenerate target normalization")
        normalization = development["normalizations"][target]
        require(normalization["target"] == target and
                normalization["train_target_min"] == low and
                normalization["train_target_max"] == high and
                normalization["train_target_range"] == span and
                normalization["physical_mse_multiplier"] == span*span and
                normalization["targets_clipped"] is False and
                normalization["normalization_source"] == "eligible_2011_training_targets_only",
                "Training-only target normalization differs")
        for role in ("train", "decision", "test"):
            saved = cohorts[target][role]
            y = [F(v-low, span) for v in expected[target][role]["raw_targets"]]
            persistence = [F(int(v)-low, span) for v in expected[target][role]["X"][:, -1]]
            require(saved["y"] == list(map(str, y)) and saved["persistence"] == list(map(str, persistence)) and
                    saved["roster"]["target_outside_prediction_range_count"] == sum(not 0 <= v <= 1 for v in y),
                    "Exact targets/persistence or unchanged out-of-range labels differ")
            expected[target][role].update(y=y, persistence=persistence)
        train = expected[target]["train"]
        persist_mse = sum(((p-y)**2 for p, y in zip(train["persistence"], train["y"])), F(0))/len(train["y"])
        epsilons[target] = F(19, 20)*persist_mse
        scales[target] = {"train_min": low, "train_max": high, "range": span,
                          "train_persistence_mse": persist_mse}
    return cohorts, expected, epsilons, scales


def candidate_key(row):
    lower, upper = interval(row["decision_risk"])
    require(lower >= 0 and type(row["deployment_payload_bytes"]) is int,
            "Invalid candidate cost/risk")
    return row["deployment_payload_bytes"], upper, row["candidate_id"]


def selection(rows, epsilon):
    return min((row for row in rows if candidate_key(row)[1] <= epsilon),
               key=candidate_key, default=None)


def candidate_summary(row):
    return None if row is None else {key: row[key] for key in
        ("candidate_id", "map", "recipe", "deployment_payload_bytes", "decision_risk", "head_id")}


IDENTITY_FIELDS = ("candidate_id", "task", "map", "recipe", "feature_id", "class_id", "head_id",
    "feature_width", "family", "scaling", "map_payload_bytes", "map_numeric_bytes",
    "head_payload_bytes", "deployment_payload_bytes", "decision_risk", "head_sha256",
    "fit_sha256", "decision_prediction_sha256", "fit_status", "projection_applied", "assurance")


def same_candidate_identity(left, right):
    require(all(left[k] == right[k] for k in IDENTITY_FIELDS),
            "Repeated fit/evaluation identity differs from exhaustive canonical head: "+left["candidate_id"])


def audit_policy_trace(trace, order, bounds, canonical_rows, pair_registry, floor_receipt):
    """Replay every decision from information available before each head fit."""
    require(trace["schema"] == "step36-fixed-tolerance-search-trace-v1" and
            trace["mode"] in ("ordinary", "triggered") and trace["max_certificate_calls"] == 6 and
            trace["max_pairs_per_call"] == 64 and trace["exact_selected_pair_distance_cap"] == 384 and
            trace["wall_time_regret_bound_claim"] is False,
            "Online policy trace schema or frozen budgets differ")
    epsilon, mode = q(trace["epsilon"]), trace["mode"]
    floor = audit_range_floor(floor_receipt)
    events, position = trace["events"], 0
    require(all(e["sequence"] == i for i, e in enumerate(events)), "Policy event sequence differs")
    incumbent, observed, calls, exact_pairs, approximate_pairs = None, {}, 0, 0, 0
    cost_pruned, class_pruned = [], []

    def take(kind):
        nonlocal position
        require(position < len(events) and events[position]["kind"] == kind,
                "Unexpected policy transition; expected "+kind)
        record = events[position]; position += 1
        return record

    def observe(candidate_id):
        nonlocal incumbent
        event = take("observe")
        row = event["row"]
        require(row["candidate_id"] == candidate_id and candidate_id not in observed,
                "Policy fit roster differs or repeats a head")
        same_candidate_identity(row, canonical_rows[candidate_id])
        before = candidate_summary(incumbent)
        eligible = candidate_key(row)[1] <= epsilon
        updated = eligible and (incumbent is None or candidate_key(row) < candidate_key(incumbent))
        if updated:
            incumbent = row
        require(event["eligible"] == eligible and event["incumbent_before"] == before and
                event["incumbent_after"] == candidate_summary(incumbent) and
                event["incumbent_updated"] == updated,
                "Observed candidate eligibility or incumbent transition differs")
        observed[candidate_id] = row
        return row

    first = take("range_floor")
    require(q(first["lower_mean"]) == floor and q(first["epsilon"]) == epsilon and
            first["prunes_all"] == (floor > epsilon) and
            first["range_floor_digest"] == digest(floor_receipt) and
            first["shared_by_both_policies"] is True,
            "Common ordinary/triggered range-floor premise differs")
    if floor <= epsilon:
        for name in order:
            affine_bound = bounds[name]["affine"]
            extension_bound = bounds[name]["extension32"]
            for bound in (affine_bound, extension_bound):
                audit_head_cost_bound(bound)
            event = take("affine_decision")
            prune = incumbent is not None and affine_bound["deployment_payload_lower_bound_bytes"] > incumbent["deployment_payload_bytes"]
            require(event["map"] == name and event["bound"] == affine_bound and
                    event["incumbent_before"] == candidate_summary(incumbent) and
                    event["certificate_calls_before"] == calls and
                    event["action"] == ("COST_PRUNE" if prune else "FIT") and
                    event["reason"] == ("STRICT_BYTE_LOWER_BOUND" if prune else "CHEAP_AFFINE_FIRST"),
                    "Affine prefit cost decision differs")
            affine = None
            if prune:
                cost_pruned.append(name+"::affine")
            else:
                affine = observe(name+"::affine")
            event = take("extension_decision")
            before_calls = calls
            affine_eligible = None if affine is None else candidate_key(affine)[1] <= epsilon
            if incumbent is not None and extension_bound["deployment_payload_lower_bound_bytes"] > incumbent["deployment_payload_bytes"]:
                action, reason = "COST_PRUNE", "STRICT_BYTE_LOWER_BOUND"
            elif affine_eligible:
                action, reason = "FIT", "AFFINE_ALREADY_VERIFIED_ADEQUATE"
            elif mode == "ordinary":
                action, reason = "FIT", "ORDINARY_SEARCH_NO_PAIR_SCREEN"
            elif affine is None:
                action, reason = "FIT", "AFFINE_FEASIBILITY_NOT_OBSERVED"
            elif calls >= 6:
                action, reason = "FIT", "PAIR_CERTIFICATE_CALL_BUDGET_EXHAUSTED"
            else:
                action, reason = "CERTIFY", "AFFINE_NOT_VERIFIED_ADEQUATE_AND_EXTENSION_COST_COMPETITIVE"
                calls += 1
            require(event["map"] == name and event["bound"] == extension_bound and
                    event["incumbent_before"] == candidate_summary(incumbent) and
                    event["affine_candidate_id"] == (None if affine is None else affine["candidate_id"]) and
                    event["affine_verified_adequate"] == affine_eligible and
                    event["certificate_calls_before"] == before_calls and
                    event["certificate_calls_after"] == calls and
                    event["action"] == action and event["reason"] == reason,
                    "Extension prefit cost/adequacy/trigger decision differs")
            if action == "CERTIFY":
                outcome = take("certificate_result")
                certificate = pair_registry[outcome["certificate_digest"]]
                result = audit_pair_certificate(certificate)
                require(certificate["feature_id"] == extension_bound["feature_id"] and
                        certificate["class_id"] == extension_bound["class_id"] and
                        certificate["query_ids"] == floor_receipt["query_ids"] and
                        certificate["targets_exact"] == floor_receipt["targets_exact"] and
                        certificate["matching"]["k"] == 4 and certificate["matching"]["max_pairs"] == 64,
                        "Triggered pair certificate source/cohort/budget differs")
                exact_count = result["pair_count"]
                approximate_count = result["n"]*(result["n"]-1)//2
                exact_pairs += exact_count; approximate_pairs += approximate_count
                action = "CLASS_PRUNE" if result["lower"] > epsilon else "FIT"
                reason = "PAIR_LOWER_EXCEEDS_FIXED_TOLERANCE" if action == "CLASS_PRUNE" else "PAIR_LOWER_DOES_NOT_EXCLUDE"
                require(outcome["map"] == name and outcome["action"] == action and outcome["reason"] == reason and
                        outcome["call_number"] == calls and q(outcome["lower_mean"]) == result["lower"] and
                        q(outcome["epsilon"]) == epsilon and outcome["feature_id"] == certificate["feature_id"] and
                        outcome["class_id"] == certificate["class_id"] and
                        outcome["exact_selected_pair_distance_evaluations"] == exact_count and
                        outcome["approximate_distance_evaluations"] == approximate_count and
                        outcome["incumbent_before"] == candidate_summary(incumbent),
                        "Class-certificate result transition differs")
            if action == "FIT":
                observe(name+"::extension32")
            elif action == "COST_PRUNE":
                cost_pruned.append(name+"::extension32")
            else:
                require(action == "CLASS_PRUNE", "Unrecognized policy outcome")
                class_pruned.append(name+"::extension32")
    require(position == len(events), "Unconsumed or out-of-order policy trace events")
    require(trace["certificate_calls"] == calls and
            trace["exact_selected_pair_distance_evaluations"] == exact_pairs and
            trace["approximate_distance_evaluations"] == approximate_pairs and
            exact_pairs <= 384 and calls <= 6 and
            trace["selection"] == incumbent,
            "Final policy selection or certificate cost totals differ")
    exhaustive = selection(canonical_rows.values(), epsilon)
    require(candidate_summary(incumbent) == candidate_summary(exhaustive),
            "Policy failed exhaustive same-threshold/cost selection preservation")
    return {"mode": mode, "selection": candidate_summary(incumbent),
        "fits": len(observed), "fitted_ids": list(observed), "cost_pruned_ids": cost_pruned,
        "class_pruned_ids": class_pruned, "certificate_calls": calls,
        "exact_selected_pair_distances": exact_pairs, "approximate_pair_distances": approximate_pairs,
        "range_pruned_all": floor > epsilon}


def audit_artificial(root):
    inherited = read_json(root/"inputs/inherited_components36.json")
    for item in inherited["files"]:
        require(item["unchanged"] is True and sha(root/item["path"]) == item["sha256"],
                "Inherited validated component changed")
    require(inherited["previous_scientific_arrays_copied"] is False,
            "Historical arrays were introduced into the new pilot")
    arithmetic = read_json(root/"analysis/finite_feature_preflight35.json")
    require(arithmetic["status"] in ("PASS", "PASSED") and
            arithmetic["actual_public_data_rows_loaded"] == arithmetic["reserved_rows_opened"] == 0,
            "Inherited arithmetic artificial preflight did not pass")
    for filename, checksum in arithmetic["source_sha256"].items():
        require(sha(root/"source"/filename) == checksum,
                "Inherited arithmetic preflight refers to another source")
    inherited_counts = old.audit_receipt_collection(arithmetic)[0]
    representations = read_json(root/"tests/artificial_representation_preflight35.json")
    require(representations["status"] in ("PASS", "PASSED") and
            representations["factory_metadata"]["factory_source_sha256"] == sha(root/"source/representations35.py") and
            representations["public_data_values_read"] == representations["patient_data_values_read"] == 0,
            "Inherited representation preflight identity or zero-access scope differs")
    policy = read_json(root/"tests/artificial_policy_preflight36.json")
    require(policy["status"] == "PASS" and policy["public_numeric_values_read"] is False and
            policy["reserved_patient_data_read"] is False and
            policy["actual_scientific_models_fit"] is False,
            "Artificial policy preflight failed or read actual scientific values")
    for path, checksum in policy["source_sha256"].items():
        require(sha(root/path) == checksum, "Policy preflight refers to a different source revision")
    witnesses = read_json(root/"tests/artificial_policy_witnesses36.json")
    require(witnesses["schema"] == "step36-artificial-policy-witnesses-v1" and
            witnesses["artificial_only"] is True and witnesses["public_numeric_values_read"] is False,
            "Artificial proof fixture scope differs")
    require(policy["witness_path"] == "tests/artificial_policy_witnesses36.json" and
            policy["witness_sha256"] == sha(root/policy["witness_path"]),
            "Artificial witness file changed after executed preflight")
    pair_checks = [audit_pair_certificate(r) for r in witnesses["pair_receipts"]]
    require(len(pair_checks) == 29 and
            [digest(r) for r in witnesses["pair_receipts"]] == policy["checks"]["pairs"]["artificial_receipt_digests"],
            "Retained pair fixture roster differs from executed preflight")
    for example in witnesses["payload_examples"]:
        result = audit_head_cost_bound(example["bound"], example["head"])
        expected = example["expected"]
        size = len(deployed_json_bytes(example["head"]))
        require(expected["verified"] is True and expected["head_payload_bytes"] == size and
                expected["deployment_payload_bytes"] == size+example["bound"]["map_payload_bytes"] and
                expected["head_payload_lower_bound_bytes"] == result["head_lower_bytes"],
                "Retained artificial payload byte result differs")
    require(len(witnesses["payload_examples"]) == 4, "Artificial payload fixture roster differs")
    data = read_json(root/"tests/artificial_data_preflight36.json")
    require(data["status"] == "PASS" and data["public_archive_numeric_values_read"] is False and
            data["test_count_values_parsed"] == 0 and data["scientific_model_fits"] == 0 and
            all(data["checks"].values()),
            "Artificial causal data/barrier checks failed")
    data_source_hashes = data["source_sha256"]
    if isinstance(data_source_hashes, str):
        data_source_hashes = {"source/data36.py": data_source_hashes}
    for path, checksum in data_source_hashes.items():
        require(sha(root/path) == checksum, "Data preflight refers to a different source revision")
    runner = read_json(root/"tests/artificial_run_preflight36.json")
    require(runner["status"] == "PASS" and runner["artificial_only"] is True and
            runner["public_measurement_values_opened"] is False and all(runner["checks"].values()),
            "Artificial actual-fit runner integration checks failed")
    for filename, checksum in runner["source_sha256"].items():
        require(sha(root/"source"/filename) == checksum, "Runner preflight refers to a different source revision")
    return {"inherited_arithmetic_receipt_counts": inherited_counts,
            "independent_pair_proofs": len(pair_checks),
            "independent_prefit_payload_bounds": len(witnesses["payload_examples"]),
            "scalar_polygon_cases": policy["checks"]["scalar"]["independent_polygon_agreements"],
            "artificial_finite_panels": policy["checks"]["finite_panel_equivalence"]["random_artificial_panels"],
            "loader_barrier_checks": len(data["checks"]), "runner_integration_checks": len(runner["checks"])}


def audit_models(root, protocol, cohorts, expected, rows_by_task):
    """Replay stored maps and verify canonical head/projection/prediction proofs."""
    import numpy as np
    from scipy.spatial.distance import pdist
    import representations35 as replay
    replay.initialize_runtime()
    features, manifest_by_task, bounds_by_task, summaries = {}, {}, {}, {}
    counts = {"maps_replayed": 0, "feature_cohorts_replayed": 0,
              "affine_fit_proofs": 0, "extension_fit_proofs": 0, "prediction_proofs": 0}
    for task in protocol["tasks"]:
        folder = root/"output/tasks"/task
        metadata = read_json(folder/"representations36.json")
        manifest = read_json(folder/"map_manifest36.json")
        bounds = read_json(folder/"head_cost_bounds36.json")
        names = metadata["map_order"]
        require(len(names) == len(set(names)) == len(manifest) == 18,
                "Frozen 18-map roster differs")
        require(metadata["config"] == protocol["representations"] and
                metadata["factory_source_sha256"] == sha(root/"source/representations35.py") and
                metadata["forecast_labels_used"] is False and metadata["no_heldout_input_or_label_used"] is True,
                "Training-only representation factory binding differs")
        train_X = expected[task]["train"]["X"]
        require(metadata["training_shape"] == list(train_X.shape), "Map training matrix shape differs")
        probes = [((2*j+1)*len(train_X))//64 for j in range(32)]
        anchors = probes
        require(metadata["probe_indices"] == probes, "Frozen metric probe roster differs")
        for mode, scaler in metadata["scalers"].items():
            minimum, maximum = train_X.min(axis=0), train_X.max(axis=0)
            constant = maximum == minimum
            offset = minimum if mode == "minmax01" else train_X.mean(axis=0)
            divisor = maximum-minimum if mode == "minmax01" else train_X.std(axis=0, ddof=0)
            divisor = np.where(constant, 1.0, divisor)
            require(np.array_equal(np.asarray(scaler["offset"]), offset) and
                    np.array_equal(np.asarray(scaler["divisor"]), divisor) and
                    scaler["constant_mask"] == constant.tolist() and scaler["training_rows"] == len(train_X) and
                    scaler["heldout_clipping"] is False,
                    "Map scaler does not equal the complete training-only fit")
        metadata_maps = {row["name"]: row for row in metadata["maps"]}
        features[task] = {}
        local_rank_records = 0
        for name in names:
            info, meta = manifest[name], metadata_maps[name]
            path = root/info["path"]
            checksum, size = sha(path), path.stat().st_size
            require(checksum == info["sha256"] == meta["payload_sha256"] and
                    size == info["payload_bytes"] == meta["payload_bytes"] and
                    info["feature_id"].endswith(checksum),
                    "Map deployment hash, size or exact feature identity differs")
            restored = replay.deserialize_map(path.read_bytes())
            require(restored.name == name and replay.numeric_bytes(restored) == info["numeric_bytes"],
                    "Restored map identity or numeric byte count differs")
            unscaled = restored.core.transform(train_X[probes])
            distances = pdist(unscaled, metric="euclidean")
            positive = distances[distances > 0]
            metric = 1.0/float(np.median(positive)) if len(positive) else 1.0
            require(float(restored.metric_scale) == metric == meta["metric_scale"] and
                    meta["positive_probe_pair_count"] == len(positive),
                    "Map metric scalar is not the frozen training-only median-distance recipe")
            features[task][name] = {}
            for role in ("train", "decision", "test"):
                array = np.load(root/"output/features"/task/role/(name+".npy"), allow_pickle=False)
                require(array.dtype == np.float64 and np.isfinite(array).all() and
                        array.shape == (len(cohorts[task][role]["ids"]), meta["actual_width"]),
                        "Stored feature shape/dtype/finiteness differs")
                replayed = replay.transform_one(restored, expected[task][role]["X"])
                require(np.array_equal(array, replayed),
                        "Sealed operational map payload does not bitwise reproduce features: "+task+"/"+name+"/"+role)
                features[task][name][role] = [[str(F.from_float(float(v))) for v in row] for row in array]
                counts["feature_cohorts_replayed"] += 1
            counts["maps_replayed"] += 1
            if meta["family"] == "nystrom":
                width = meta["requested_width"]
                require(meta["landmark_indices"] == [((2*j+1)*len(train_X))//(2*width) for j in range(width)],
                        "Nyström landmark roster is not the frozen training subset")
            elif meta["family"] == "kahm":
                for local in meta["all_local_models"]:
                    expected_initial = min(meta["cap"], 24, local["local_reference_count"]-1)
                    require(local["requested_cap"] == meta["cap"] and
                            local["initial_admissible_rank"] == expected_initial and
                            0 <= local["effective_rank"] <= expected_initial and
                            local["ambient_dimension"] == 24,
                            "Native local dimension cap or recorded effective rank differs")
                    local_rank_records += 1
            for recipe in ("affine", "extension32"):
                bound = bounds[name][recipe]
                audit_head_cost_bound(bound)
                require(bound["feature_id"] == info["feature_id"] and
                        bound["map_payload_bytes"] == size and bound["dimension"] == meta["actual_width"],
                        "Prefit byte bound is not bound to its frozen representation")
                if recipe == "extension32":
                    template = bound["template_head"]
                    require(template["anchor_ids"] == [cohorts[task]["train"]["ids"][i] for i in anchors] and
                            template["anchor_features"] == [features[task][name]["train"][i] for i in anchors],
                            "Prefit extension template used unknown/nontraining anchor fields")
        rows = rows_by_task[task]
        expected_ids = {name+"::"+recipe for name in names for recipe in ("affine", "extension32")}
        require(set(rows) == expected_ids and len(rows) == 36, "Canonical finite head panel is incomplete")
        for candidate_id, row in rows.items():
            name, recipe = candidate_id.split("::")
            info = manifest[name]
            head_folder = root/"output/heads"/task/(name+"__"+recipe)
            head = read_json(head_folder/"head.json")
            fit = read_json(head_folder/"fit.json")
            require(head == fit["head"] and row["head_id"] == fit["head_id"] and
                    row["feature_id"] == fit["feature_id"] == head["feature_id"] == info["feature_id"] and
                    row["class_id"] == fit["class_id"] == head["class_id"],
                    "Canonical deployed/fitted head binding differs")
            require(row["task"] == task and row["map"] == name and row["recipe"] == recipe and
                    row["head_sha256"] == sha(head_folder/"head.json") and
                    row["fit_sha256"] == sha(head_folder/"fit.json") and
                    row["decision_prediction_sha256"] == sha(head_folder/"decision.json"),
                    "Canonical fitted/decision proof digest differs")
            head_bytes = (head_folder/"head.json").stat().st_size
            require(row["head_payload_bytes"] == head_bytes and
                    row["map_payload_bytes"] == info["payload_bytes"] and
                    row["deployment_payload_bytes"] == head_bytes+info["payload_bytes"],
                    "Canonical complete deployment payload byte count differs")
            audit_head_cost_bound(bounds[name][recipe], head)
            old.audit_head(head)
            if recipe == "affine":
                old.audit_affine_record(fit)
                train_z = old.matrix(features[task][name]["train"])
                center = [sum((row_z[j] for row_z in train_z), F(0))/len(train_z) for j in range(len(train_z[0]))]
                require(fit["projection"]["center_features"] == list(map(str, center)) and
                        q(fit["projection"]["center_target"]) == sum(map(q, cohorts[task]["train"]["y"]), F(0))/len(train_z) and
                        q(fit["training_binding"]["ridge_lambda"]) == F(1, 1000000),
                        "Affine training-only exact center or ridge recipe differs")
                counts["affine_fit_proofs"] += 1
            else:
                old.audit_qp(fit)
                require(head["anchor_ids"] == [cohorts[task]["train"]["ids"][i] for i in anchors] and
                        head["anchor_features"] == [features[task][name]["train"][i] for i in anchors] and
                        fit["targets_exact"] == [cohorts[task]["train"]["y"][i] for i in anchors] and
                        fit["training_binding"]["anchor_indices"] == anchors and
                        fit["work"]["max_updates"] == 4000,
                        "Conservative extension fitted a different anchor/target roster or budget")
                counts["extension_fit_proofs"] += 1
            for role in ("decision", "test"):
                receipt = read_json(head_folder/(role+".json"))
                old.audit_prediction(receipt)
                require(receipt["head"] == head and receipt["query_ids"] == cohorts[task][role]["ids"] and
                        receipt["features_exact"] == features[task][name][role] and
                        receipt["targets_exact"] == cohorts[task][role]["y"],
                        "Canonical prediction proof is not on the common stored feature/target cohort")
                interval(row[role+"_risk"], interval(receipt["loss"]))
                counts["prediction_proofs"] += 1
        manifest_by_task[task] = manifest
        bounds_by_task[task] = bounds
        summaries[task] = {"maps": len(names), "heads": len(rows), "local_rank_records_checked": local_rank_records}
    return features, manifest_by_task, bounds_by_task, summaries, counts


def policy_selection_summary(row):
    if row is None:
        return {"candidate_id": None, "head_id": None, "abstained": True}
    return {"candidate_id": row["candidate_id"], "head_id": row["head_id"], "abstained": False,
            "deployment_payload_bytes": row["deployment_payload_bytes"], "decision_risk": row["decision_risk"]}


def audit_searches(root, protocol, result, cohorts, epsilons, scales, features, manifests, bounds, rows_by_task):
    checks, actual_fits, total_calls, total_exact_distances = {}, 0, 0, 0
    search_execution_order = []
    frozen = read_json(root/"output/policies36.json")
    require(frozen["test_values_parsed"] is False, "Global policy was not frozen before test")
    for task_index, task in enumerate(protocol["tasks"]):
        rows, epsilon = rows_by_task[task], epsilons[task]
        task_result = result["tasks"][task]
        frozen_task = read_json(root/"output/tasks"/task/"policy36.json")
        require(frozen["tasks"][task] == frozen_task, "All-target and per-target frozen policy differ")
        unsealed = json.loads(json.dumps(task_result))
        test_time = unsealed["timing"].pop("test_evaluation_and_retention_seconds")
        require(test_time >= 0 and unsealed == frozen_task,
                "A pre-test policy, threshold, timing or selection changed after its seal")
        plan = read_json(root/"output/tasks"/task/"training_plan36.json")
        anchors = [((2*j+1)*len(cohorts[task]["train"]["ids"]))//64 for j in range(32)]
        require(q(plan["epsilon"]) == epsilon and q(task_result["epsilon"]) == epsilon and
                q(plan["train_persistence_mse"]) == scales[task]["train_persistence_mse"] and
                q(plan["epsilon_physical_mse"]) == epsilon*scales[task]["range"]**2 and
                plan["anchor_indices"] == anchors and
                plan["anchor_ids"] == [cohorts[task]["train"]["ids"][i] for i in anchors] and
                plan["test_values_parsed"] is False,
                "Training-only tolerance/anchor plan differs")
        order = sorted(manifests[task], key=lambda name: (manifests[task][name]["payload_bytes"], name))
        require(task_result["map_order"] == order, "Policy map order uses unavailable/postfit costs")
        expected_pairs = [["ordinary", "triggered"] if (task_index+r)%2 == 0 else ["triggered", "ordinary"] for r in range(5)]
        expected_runs = [("exhaustive", 0)]+[(mode, r) for r, modes in enumerate(expected_pairs) for mode in modes]
        expected_paths = ["output/search/"+task+"/"+mode+"_r%02d.json" % r for mode, r in expected_runs]
        require(task_result["paired_execution_order"] == expected_pairs and
                task_result["search_paths"] == expected_paths,
                "Fixed alternating matched execution roster differs")
        canonical_choice = selection(rows.values(), epsilon)
        chosen = policy_selection_summary(canonical_choice)
        require(task_result["exhaustive_selection"] == chosen,
                "Frozen threshold/cost choice differs from independent exhaustive reconstruction")
        accuracy = min(rows.values(), key=lambda row: (q(row["decision_risk"]["upper"]), row["deployment_payload_bytes"], row["candidate_id"]))
        require(task_result["ordinary_mse_minimizer"] == policy_selection_summary(accuracy),
                "Descriptive decision-MSE comparator changed after test")
        records, results = {}, {}
        for (mode, r), path in zip(expected_runs, expected_paths):
            record = read_json(root/path)
            require(record["schema"] == "step36-actual-search-execution-v1" and
                    record["task"] == task and record["mode"] == mode and record["repetition"] == r and
                    record["map_order"] == order and q(record["epsilon"]) == epsilon and
                    record["fresh_actual_head_fits"] is True and
                    record["canonical_exhaustive_results_available_inside_policy"] is False and
                    record["evidence_disk_io_inside_timing"] is False and
                    record["canonical_identity_checked_after_timing"] == (mode != "exhaustive") and
                    record["canonical_prediction_proofs_retained_once"] is True,
                    "Actual fresh-fit execution scope differs")
            require(record["trace_sha256"] == hashlib.sha256(deployed_json_bytes(record["trace"])).hexdigest() and
                    record["range_floor_sha256"] == hashlib.sha256(deployed_json_bytes(record["range_floor"])).hexdigest(),
                    "Timed trace or common-floor digest changed during retention")
            floor = audit_range_floor(record["range_floor"])
            require(record["range_floor"]["query_ids"] == cohorts[task]["decision"]["ids"] and
                    record["range_floor"]["targets_exact"] == cohorts[task]["decision"]["y"] and
                    q(task_result["range_floor_mean"]) == floor,
                    "Common range-only premise uses a different decision roster")
            registry = {}
            for item in record["certificates"]:
                certificate = read_json(root/item["path"])
                key = digest(certificate)
                require(item["sha256"] == sha(root/item["path"]) and item["certificate_digest"] == key and
                        item["lower_mean"] == certificate["lower_mean"] and item["work"] == certificate["work"] and
                        certificate["features_exact"] == features[task][item["map"]]["decision"] and
                        certificate["feature_id"] == manifests[task][item["map"]]["feature_id"],
                        "Retained pair proof does not match timed/source feature identity")
                require(key not in registry, "Duplicate certificate receipt within one execution")
                registry[key] = certificate
            ids = [row["candidate_id"] for row in record["candidate_rows"]]
            require(len(ids) == len(set(ids)), "Actual execution fitted a head twice")
            for row in record["candidate_rows"]:
                same_candidate_identity(row, rows[row["candidate_id"]])
            if mode == "exhaustive":
                expected_ids = [name+"::"+recipe for name in order for recipe in ("affine", "extension32")]
                require(ids == expected_ids and not registry and
                        record["trace"]["candidate_order"] == expected_ids and
                        record["trace"]["range_floor_used_to_prune"] is False and
                        candidate_summary(record["trace"]["selection"]) == candidate_summary(canonical_choice),
                        "Canonical exhaustive execution did not evaluate the complete frozen head panel")
                policy_result = {"fits": 36, "fitted_ids": ids, "cost_pruned_ids": [],
                    "class_pruned_ids": [], "range_pruned_all": False,
                    "certificate_calls": 0, "exact_selected_pair_distances": 0,
                    "approximate_pair_distances": 0}
            else:
                policy_result = audit_policy_trace(record["trace"], order, bounds[task], rows, registry, record["range_floor"])
                require(policy_result["fitted_ids"] == ids,
                        "Retained actual fit list differs from replayed policy transitions")
            counts = record["counts"]
            require(counts == {"fit_calls": policy_result["fits"],
                "affine_fit_calls": sum(row["recipe"] == "affine" for row in record["candidate_rows"]),
                "extension_fit_calls": sum(row["recipe"] == "extension32" for row in record["candidate_rows"]),
                "cost_pruned_heads": len(policy_result["cost_pruned_ids"]),
                "class_pruned_heads": len(policy_result["class_pruned_ids"]),
                "range_pruned_heads": 36 if policy_result["range_pruned_all"] else 0} and
                counts["fit_calls"]+counts["cost_pruned_heads"]+counts["class_pruned_heads"]+counts["range_pruned_heads"] == 36,
                "Actual fit/prune work accounting differs")
            require(record["selection"] == chosen and record["elapsed_seconds"] >= 0 and
                    record["certificate_seconds"] == sum(item["call_seconds_including_receipt_and_policy_hash"] for item in record["certificates"]) and
                    record["certificate_seconds"] <= record["elapsed_seconds"] and
                    len(registry) == policy_result["certificate_calls"],
                    "Choice preservation or observed policy work accounting differs")
            records[mode, r] = record; results[mode, r] = policy_result
            actual_fits += counts["fit_calls"]
            total_calls += policy_result["certificate_calls"]
            total_exact_distances += policy_result["exact_selected_pair_distances"]
            search_execution_order.append((task, mode, r, path))
        ordinary, triggered = [records["ordinary", r] for r in range(5)], [records["triggered", r] for r in range(5)]
        ordinary_times = [row["elapsed_seconds"] for row in ordinary]
        triggered_times = [row["elapsed_seconds"] for row in triggered]
        differences = [a-b for a, b in zip(ordinary_times, triggered_times)]
        avoided = [a["counts"]["fit_calls"]-b["counts"]["fit_calls"] for a, b in zip(ordinary, triggered)]
        require(len(set(avoided)) == 1 and avoided[0] >= 0,
                "Additional head fits avoided change across deterministic executions")
        for r in range(5):
            require(set(results["triggered", r]["fitted_ids"]) <= set(results["ordinary", r]["fitted_ids"]),
                    "Triggered policy fit a new head absent from ordinary search")
        utility = avoided[0] > 0 and sorted(differences)[2] > 0
        statistics_expected = {
            "safety_pass": True, "exhaustive_selection": chosen, "exhaustive_fit_calls": 36,
            "ordinary_fit_calls_each_repetition": [row["counts"]["fit_calls"] for row in ordinary],
            "triggered_fit_calls_each_repetition": [row["counts"]["fit_calls"] for row in triggered],
            "additional_head_fits_avoided_each_repetition": avoided,
            "ordinary_seconds": ordinary_times, "triggered_seconds": triggered_times,
            "paired_ordinary_minus_triggered_seconds": differences,
            "median_paired_seconds_saved": sorted(differences)[2],
            "ordinary_median_seconds": sorted(ordinary_times)[2],
            "triggered_median_seconds": sorted(triggered_times)[2],
            "ordinary_seconds_range": [min(ordinary_times), max(ordinary_times)],
            "triggered_seconds_range": [min(triggered_times), max(triggered_times)],
            "triggered_certificate_calls_each_repetition": [results["triggered", r]["certificate_calls"] for r in range(5)],
            "triggered_class_pruned_heads_each_repetition": [len(results["triggered", r]["class_pruned_ids"]) for r in range(5)],
            "triggered_exact_pair_distances_each_repetition": [results["triggered", r]["exact_selected_pair_distances"] for r in range(5)],
            "positive_incremental_certificate_utility": utility}
        stats = task_result["search_statistics"]
        require(all(stats[k] == v for k, v in statistics_expected.items()),
                "Frozen matched timing/fit-saving/utility summary differs")
        checks[task] = {"selection": chosen, "epsilon": str(epsilon),
            "ordinary_fits": ordinary[0]["counts"]["fit_calls"],
            "triggered_fits": triggered[0]["counts"]["fit_calls"],
            "additional_fits_avoided": avoided[0], "median_paired_seconds_saved": sorted(differences)[2],
            "certificate_calls_per_triggered_run": results["triggered", 0]["certificate_calls"],
            "positive_incremental_certificate_utility": utility}
    consistent = all(row["positive_incremental_certificate_utility"] for row in checks.values())
    require(result["actual_search_head_fits"] == actual_fits and
            result["all_task_safety_pass"] == frozen["all_task_safety_pass"] is True and
            result["consistent_panel_incremental_utility"] == frozen["consistent_panel_incremental_utility"] == consistent,
            "All-task headline work/selection/utility claims differ")
    return checks, {"actual_search_head_fits": actual_fits, "pair_certificate_calls": total_calls,
                    "exact_selected_pair_distances": total_exact_distances,
                    "consistent_panel_incremental_utility": consistent}, search_execution_order


def audit_baselines(root, protocol, result, cohorts, expected):
    import numpy as np
    import pickle
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = {(row["task"], row["baseline"]): row for row in result["baseline_rows"]}
    require(len(rows) == 9 and set(rows) == {(task, name) for task in protocol["tasks"] for name in protocol["baselines"]["names"]},
            "Descriptive baseline roster differs")
    count = 0
    for task in protocol["tasks"]:
        path = root/"output/baselines"/task/"hist_gradient_boosting.pickle"
        with path.open("rb") as handle:
            restored = pickle.load(handle)
        model, scaler = restored["model"], restored["scaler"]
        require(isinstance(model, HistGradientBoostingRegressor) and model.n_features_in_ == 24,
                "Retained empirical baseline payload differs")
        params = model.get_params()
        require(all(params[k] == v for k, v in protocol["baselines"]["hist_gradient_boosting_parameters"].items()),
                "Empirical baseline used untuned-but-unfrozen parameters")
        train_X = expected[task]["train"]["X"]
        low, high = train_X.min(axis=0), train_X.max(axis=0)
        divisor = np.where(high == low, 1.0, high-low)
        require(np.array_equal(scaler.offset, low) and np.array_equal(scaler.divisor, divisor),
                "Empirical baseline scaler is not the frozen training-only MinMax scaler")
        mean = sum(expected[task]["train"]["y"], F(0))/len(expected[task]["train"]["y"])
        for role in ("decision", "test"):
            predictions = {"persistence": expected[task][role]["persistence"],
                "training_mean": [mean]*len(expected[task][role]["y"]),
                "hist_gradient_boosting": [F.from_float(float(v)) for v in model.predict(scaler.transform(expected[task][role]["X"]))]}
            for name, values in predictions.items():
                row = rows[task, name]
                receipt_path = root/row[role+"_prediction_path"]
                receipt = read_json(receipt_path)
                require(receipt["schema"] == "step36-stored-baseline-predictions-v1" and
                        receipt["baseline"] == name and receipt["query_ids"] == cohorts[task][role]["ids"] and
                        receipt["targets_exact"] == cohorts[task][role]["y"] and
                        receipt["predictions_exact"] == list(map(str, values)) and
                        receipt["class_membership_certified"] is False and
                        receipt["assurance"] == "EXACT_LOSS_OF_STORED_BASELINE_PREDICTIONS" and
                        receipt["population_risk_claim"] is False,
                        "Empirical baseline replay or target binding differs")
                boxes = [{"lower": str(v), "upper": str(v), "width": "0"} for v in values]
                old.audit_loss(boxes, expected[task][role]["y"], receipt["loss"])
                interval(row[role+"_risk"], interval(receipt["loss"]))
                require(row["class_membership_certified"] is False and
                        row["deployment_payload_bytes"] == (path.stat().st_size if name == "hist_gradient_boosting" else None),
                        "Descriptive baseline payload or class claim differs")
                if role == "test":
                    require(row["test_prediction_sha256"] == sha(receipt_path), "Baseline test proof hash differs")
                count += 1
    return {"trained_baseline_payloads_replayed": 3, "prediction_and_loss_receipts_verified": count,
            "no_baseline_refits_performed_by_audit": True}


def audit_provenance(root, protocol, result, expected_search_order):
    protocol_seal = old.verify_seal(root/"protocol_seal36.json", root)
    model_seal = old.verify_seal(root/"output/model_seal36.json", root)
    gate = read_json(root/"audit/preexecution_gate36.json")
    require(gate["status"] == "PASS" and
            gate["protocol_seal_sha256"] == sha(root/"protocol_seal36.json") and
            model_seal["protocol_seal_sha256"] == sha(root/"protocol_seal36.json") and
            model_seal["all_models_and_policies_frozen"] is True and
            model_seal["test_numeric_values_parsed"] is False,
            "Independent execution gate or all-target model barrier differs")
    require(result["protocol_sha256"] == sha(root/"protocol36.json") and
            result["protocol_seal_sha256"] == sha(root/"protocol_seal36.json") and
            result["model_seal_sha256"] == sha(root/"output/model_seal36.json"),
            "Result hash binding differs")
    required = {"output/policies36.json", "output/development_data36.json", "output/primary_started36.json"}
    for task in protocol["tasks"]:
        required.update({"output/tasks/"+task+"/policy36.json", "output/tasks/"+task+"/training_plan36.json",
                         "output/tasks/"+task+"/map_manifest36.json", "output/tasks/"+task+"/head_cost_bounds36.json"})
    required.update(path for _, _, _, path in expected_search_order)
    require(required <= set(model_seal["bound_files"]), "All-target model seal omitted a required pretest decision")
    require(all("/test/" not in path and not path.endswith(("/test.json", "_test.json", "test_data36.json"))
                for path in model_seal["bound_files"]), "Test artifacts already existed at model seal")
    events = old.audit_event_chain(root/"output/access_and_execution36.jsonl")
    names = [e["event"] for e in events]
    for name in ("primary_start", "development_numeric_access_start", "development_numeric_access_complete",
                 "models_and_policies_sealed", "test_numeric_access_start", "test_numeric_access_complete", "primary_complete"):
        require(names.count(name) == 1, "Execution event missing or duplicated: "+name)
    require(names.index("primary_start") < names.index("development_numeric_access_start") <
            names.index("development_numeric_access_complete") < names.index("models_and_policies_sealed") <
            names.index("test_numeric_access_start") < names.index("test_numeric_access_complete") < names.index("primary_complete"),
            "Development/model/test barrier order differs")
    require(datetime.fromisoformat(gate["utc"]) <= datetime.fromisoformat(events[names.index("primary_start")]["utc"]),
            "Primary run began before independent preflight")
    starts = [e for e in events if e["event"] == "search_execution_start"]
    completes = [e for e in events if e["event"] == "search_execution_complete"]
    require(len(starts) == len(completes) == 33, "Actual search execution count differs")
    for i, ((task, mode, repetition, path), start, complete) in enumerate(zip(expected_search_order, starts, completes)):
        require((start["task"], start["mode"], start["repetition"]) == (task, mode, repetition) and
                (complete["task"], complete["mode"], complete["repetition"]) == (task, mode, repetition) and
                start["monotonic_ns"] < complete["monotonic_ns"] < events[names.index("models_and_policies_sealed")]["monotonic_ns"] and
                (i == 0 or completes[i-1]["monotonic_ns"] < start["monotonic_ns"]),
                "Actual alternating measured-execution order differs")
        require(complete["search_path"] == path and complete["search_sha256"] == sha(root/path) and
                complete["elapsed_seconds"] == read_json(root/path)["elapsed_seconds"],
                "Execution event references another search record")
    completed_tasks = [e for e in events if e["event"] == "task_policies_complete"]
    require([e["task"] for e in completed_tasks] == protocol["tasks"] and
            all(e["monotonic_ns"] < events[names.index("models_and_policies_sealed")]["monotonic_ns"] and
                e["test_values_parsed"] is False for e in completed_tasks),
            "A task selected after test or was omitted from the all-target barrier")
    sealed_event = events[names.index("models_and_policies_sealed")]
    test_start = events[names.index("test_numeric_access_start")]
    require(sealed_event["model_seal_sha256"] == test_start["model_seal_sha256"] == sha(root/"output/model_seal36.json") and
            test_start["all_policy_and_model_choices_frozen"] is True and
            events[names.index("primary_complete")]["result_sha256"] == sha(root/"output/result36.json"),
            "Event-chain test barrier or final result hash binding differs")
    require(result["primary_run_count"] == 1 and result["canonical_prediction_heads_fitted"] == 108 and
            result["baseline_models_fitted"] == 3 and result["adaptive_public_refits"] == 0 and
            result["predeclared_actual_timing_repetitions"] == 5 and
            result["reserved_patient_arrays_read"] is False and result["reserved_patients_accessed"] == 0 and
            result["inherited_study_statuses"] == protocol["execution"]["inherited_study_statuses"] and
            result["assurance"] == ASSURANCE,
            "Primary run count, sealed-study scope or assurance changed")
    return {"protocol_files_sealed": len(protocol_seal["bound_files"]),
            "pretest_model_policy_files_sealed": len(model_seal["bound_files"]),
            "execution_events_verified": len(events), "actual_ordered_search_executions": len(starts),
            "all_targets_selected_before_any_test_counts": True,
            "local_hash_chain_is_not_external_trusted_timestamp": True}


def independent_selfcheck():
    require(sha(Path(old.__file__)) == INDEPENDENT_MATH_SHA256,
            "Independent inherited arithmetic checker changed")
    checks = dict(old.independent_selfcheck())
    cases = [
        ((0, 1, 0, (0, 1)), ([F(1, 2), F(1, 2)], F(1, 2))),
        ((0, 1, F(1, 4), (0, 1)), ([F(3, 8), F(5, 8)], F(9, 32))),
        ((-1, 2, 0, (0, 1)), ([F(1, 2), F(1, 2)], F(9, 2))),
        ((-1, 2, 2, (0, 1)), ([F(0), F(1)], F(2))),
        ((-4, -3, F(1, 2), (0, 1)), ([F(0), F(0)], F(25))),
        ((2, 5, F(1, 4), (0, 1)), ([F(1), F(1)], F(17))),
        ((-3, 6, 2, (-2, 4)), ([F(1, 2), F(5, 2)], F(49, 2))),
    ]
    for index, (arguments, expected) in enumerate(cases):
        result = exact_pair_polygon(*arguments)
        require((result["values"], result["sse"]) == expected,
                "Independent exact pair check differs: " + str(index))
    checks["pair_polygon_closed_and_degenerate_cases"] = len(cases)
    return checks


def preflight(root, sealed=True):
    require(not (root/"output/development_data36.json").exists() and
            not (root/"output/test_data36.json").exists() and
            not (root/"output/primary_started36.json").exists(),
            "A primary numerical execution already started before preflight")
    protocol = read_json(root/"protocol36.json")
    protocol_checks = audit_protocol(protocol)
    for path in sorted((root/"source").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    artifacts = audit_artificial(root)
    metadata = read_json(root/"inputs/metadata_preflight36.json")
    require(metadata["status"] == "PASS" and metadata["count_values_numerically_parsed"] == 0 and
            metadata["test_count_values_numerically_parsed"] == 0 and
            metadata["count_missingness_inspected"] is False and
            metadata["target_payload_tokens_split"] is False and metadata["scientific_model_fits"] == 0,
            "Timestamp-only preflight crossed the numerical value boundary")
    source = read_json(root/"inputs/public_source36.json")
    require(sha(root/protocol["data"]["archive"]) == source["archive_sha256"] == metadata["archive_sha256"],
            "Opaque public archive identity changed")
    result = {"schema": "step36-independent-preexecution-gate-v1", "status": "PASS" if sealed else "PRESEAL_REVIEW_PASS",
        "utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(root/"protocol36.json"),
        "independent_math_selfchecks": independent_selfcheck(), "artificial_proof_checks": artifacts,
        "protocol_semantic_checks": protocol_checks,
        "source_review": {"causal_common_calendar_roster_and_test_barrier": "PASS",
            "closed_form_pair_optimum_independent_polygon_and_dual_check": "PASS",
            "whole_cohort_range_floor_plus_disjoint_improvements": "PASS",
            "strict_prefit_payload_bound_and_equal_cost_ties": "PASS",
            "actual_fresh_fits_and_matched_counterfactual_information": "PASS",
            "shared_encoder_feature_template_costs_separate_from_policy_time": "PASS",
            "fixed_alternation_and_budget_exhaustion_fallback": "PASS",
            "all_target_model_selection_seal_before_test_counts": "PASS"},
        "public_bike_numeric_values_parsed_by_this_audit": False,
        "reserved_patient_arrays_accessed": 0,
        "limitations": ["File seals and a local hash chain are reproducible provenance, not external trusted timestamps.",
            "The preflight checks frozen source and artificial fixtures; actual witnesses and utility require the postrun audit.",
            "A few training rows appeared incidentally in earlier metadata-search output; this is disclosed and is not a claim that every public value was unseen."]}
    if sealed:
        seal = old.verify_seal(root/"protocol_seal36.json", root)
        required = {"protocol36.json", "source/audit36.py", "source/independent_math35.py", "source/data36.py",
            "source/run36.py", "source/models36.py", "source/triggered_policy36.py", "source/finite_feature35.py",
            "source/representations35.py", "tests/artificial_policy_preflight36.json", "tests/artificial_policy_witnesses36.json",
            "tests/artificial_data_preflight36.json", "tests/artificial_run_preflight36.json",
            "inputs/metadata_preflight36.json", "inputs/prior_exposure36.json", protocol["data"]["archive"]}
        require(required <= set(seal["bound_files"]), "Frozen protocol seal omitted required audit/source/proof/data inputs")
        require(seal["programmatic_public_measurements_parsed_before_seal"] is False and
                seal["public_model_fits_before_seal"] == 0 and
                seal["incidental_training_snippet_disclosed"] is True and
                seal["decision_test_measurements_seen_before_seal"] is False,
                "Sealed exposure declaration differs")
        result.update(protocol_seal_sha256=sha(root/"protocol_seal36.json"), sealed_file_count=len(seal["bound_files"]))
    return result


def postrun(root):
    started = datetime.now(timezone.utc)
    old.verify_seal(root/"protocol_seal36.json", root)
    old.verify_seal(root/"output/model_seal36.json", root)
    require((root/"output/test_data36.json").exists(), "Postrun audit must not open test values before the test barrier")
    protocol = read_json(root/"protocol36.json")
    protocol_checks = audit_protocol(protocol)
    result = read_json(root/"output/result36.json")
    require(result["status"] == "PRIMARY_COMPLETED_PENDING_INDEPENDENT_AUDIT" and len(result["candidate_rows"]) == 108,
            "Primary run is incomplete or canonical panel size differs")
    rows_by_task = {task: {row["candidate_id"]: row for row in result["candidate_rows"] if row["task"] == task}
                    for task in protocol["tasks"]}
    print(json.dumps({"audit_stage": "independent_causal_data_reconstruction"}), flush=True)
    cohorts, expected, epsilons, scales = audit_cohorts(root, protocol)
    print(json.dumps({"audit_stage": "map_head_and_prediction_proofs"}), flush=True)
    features, manifests, bounds, model_summaries, proof_counts = audit_models(root, protocol, cohorts, expected, rows_by_task)
    print(json.dumps({"audit_stage": "online_search_policy_and_timing"}), flush=True)
    search_checks, cost_totals, order = audit_searches(root, protocol, result, cohorts, epsilons, scales, features, manifests, bounds, rows_by_task)
    baselines = audit_baselines(root, protocol, result, cohorts, expected)
    provenance = audit_provenance(root, protocol, result, order)
    return {"schema": "step36-independent-postrun-audit-v1", "status": "PASS",
        "started_utc": started.isoformat(), "completed_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": sha(root/"source/audit36.py"), "independent_math_sha256": sha(root/"source/independent_math35.py"),
        "result_sha256": sha(root/"output/result36.json"),
        "protocol_seal_sha256": sha(root/"protocol_seal36.json"), "model_seal_sha256": sha(root/"output/model_seal36.json"),
        "protocol_semantic_checks": protocol_checks, "provenance_checks": provenance,
        "cohorts": {task: {role: len(value["ids"]) for role, value in task_cohorts.items()} for task, task_cohorts in cohorts.items()},
        "exact_training_thresholds": {task: str(value) for task, value in epsilons.items()},
        "model_checks": model_summaries, "proof_counts": proof_counts, "baseline_checks": baselines,
        "search_checks": search_checks, "search_cost_totals": cost_totals,
        "assurance": ASSURANCE, "reserved_patient_arrays_accessed": 0,
        "scientific_models_refitted_by_audit": 0,
        "limitations": ["Stored payload replay establishes operational reproducibility, not ideal-real native-transform enclosure.",
            "Repeated fit identities are independently matched to canonical proofs; the frozen runner and retained local trace establish the actual execution record.",
            "One external dataset with three algebraically related targets; no independent-dataset or population inference.",
            "Timing is conditional search time after shared encoders/features/templates and excludes disk evidence writes.",
            "Generic safe pruning or exact pair optimization alone is not an established novelty claim."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["selfcheck", "review", "preflight", "postrun"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        if args.mode == "selfcheck":
            report = {"status": "PASS", "selfchecks": independent_selfcheck(), "public_numeric_values_read": False}
        elif args.mode == "review":
            report = preflight(args.root, sealed=False)
        elif args.mode == "preflight":
            report = preflight(args.root)
            write_json(args.root/"audit/preexecution_gate36.json", report)
        else:
            report = postrun(args.root)
            write_json(args.root/"audit/postrun_audit36.json", report)
        print(json.dumps(canon(report), sort_keys=True), flush=True)
    except Exception as error:
        failure = {"status": "FAIL", "mode": args.mode, "utc": datetime.now(timezone.utc).isoformat(),
                   "error": repr(error), "traceback": traceback.format_exc(),
                   "no_scientific_state_modified_by_audit": True}
        if args.mode in ("preflight", "postrun"):
            write_json(args.root/("audit/"+args.mode+"_failure36.json"), failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        raise
