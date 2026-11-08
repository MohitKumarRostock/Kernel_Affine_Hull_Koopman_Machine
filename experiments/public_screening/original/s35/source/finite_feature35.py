"""Exact finite-cohort certificates conditional on stored feature coordinates.

This pure module never opens files, fits a feature transform, or imports a
model/runtime library.  A supplied binary64 number is interpreted as its exact
dyadic value, not as an enclosure for an ideal-real native transform.  The
caller must bind the representation state, data roles, and feature receipts.

The fitting QP uses conservative LOWER distances in its constraints and dual.
The assessed class lower certificate uses UPPER distances in its dual.  The
range-aware dual remains valid when observed targets lie outside the declared
PREDICTOR range; targets are never clipped or substituted in squared loss.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
from math import isfinite, isqrt


ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
HEAD_SCHEMA = "step35-stored-feature-head-v1"
CLASS_SCHEMA = "step35-stored-feature-decoder-class-v1"


class FeatureCertificateError(ValueError):
    pass


def rational(value):
    """Interpret Python float inputs as exact stored binary64 dyadics."""
    if isinstance(value, bool):
        raise FeatureCertificateError("Boolean is not an explicit numeric coordinate")
    if isinstance(value, float):
        if not isfinite(value):
            raise FeatureCertificateError("Nonfinite binary64 coordinate")
        return F.from_float(value)
    if not isinstance(value, (str, int, F)):
        raise FeatureCertificateError("Require rational strings, integers, Fraction or finite Python float")
    try:
        return F(value)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise FeatureCertificateError("Invalid explicit rational") from error


def exact_float(value):
    if not isinstance(value, float) or not isfinite(value):
        raise FeatureCertificateError("exact_float requires a finite Python binary64 value")
    return str(F.from_float(value))


def canonical(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, float):
        return exact_float(value)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise FeatureCertificateError(f"{name} must be an integer in [{low},{high}]")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise FeatureCertificateError(name + " must be a nonempty string")
    return value


def _features(value, *, maximum_rows=None):
    if not isinstance(value, (list, tuple)) or not value:
        raise FeatureCertificateError("Require a nonempty feature matrix")
    if maximum_rows is not None and len(value) > maximum_rows:
        raise FeatureCertificateError("Finite QP row cap exceeded")
    if not isinstance(value[0], (list, tuple)) or not value[0]:
        raise FeatureCertificateError("Require a positive feature dimension")
    d = len(value[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != d for row in value):
        raise FeatureCertificateError("Feature rows must have one common dimension")
    return [[rational(v) for v in row] for row in value]


def _ids(value, n):
    if value is None:
        return [f"q{i:06d}" for i in range(n)]
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise FeatureCertificateError("Input identifier roster has the wrong length")
    names = [_text(v, "query id") for v in value]
    if len(set(names)) != n:
        raise FeatureCertificateError("Input identifiers must be unique within a cohort")
    return names


def _targets(values, n):
    if not isinstance(values, (list, tuple)) or len(values) != n:
        raise FeatureCertificateError("Target roster has the wrong length")
    return [rational(v) for v in values]


def _range(output_interval):
    if not isinstance(output_interval, (list, tuple)) or len(output_interval) != 2:
        raise FeatureCertificateError("Require an explicit two-endpoint predictor range")
    low, high = map(rational, output_interval)
    if not low < high:
        raise FeatureCertificateError("Predictor range must be nonempty")
    return low, high


def _clip(value, low, high):
    return max(low, min(high, value))


def _box(lower, upper):
    if lower > upper:
        raise FeatureCertificateError("Reversed enclosure")
    return {"lower": lower, "upper": upper, "width": upper-lower}


def _read_box(value):
    try:
        lo, hi = rational(value["lower"]), rational(value["upper"])
    except (KeyError, TypeError) as error:
        raise FeatureCertificateError("Malformed enclosure") from error
    if lo > hi or ("width" in value and rational(value["width"]) != hi-lo):
        raise FeatureCertificateError("Invalid enclosure ordering or width")
    return lo, hi


def sqrt_enclosure(value, *, bits=80):
    """Outward dyadic square-root bounds proved by exact integer arithmetic."""
    _integer(bits, "distance bits", 1, 512)
    value = rational(value)
    if value < 0:
        raise FeatureCertificateError("A squared distance cannot be negative")
    denominator = 1 << bits
    root = isqrt((value.numerator << (2*bits)) // value.denominator)
    low = F(root, denominator)
    high = low if low*low == value else F(root+1, denominator)
    if not low*low <= value <= high*high:
        raise ArithmeticError("Integer square-root enclosure failed")
    return canonical({**_box(low, high), "squared_exact": value, "bits": bits,
                      "scaled_sqrt_floor": root})


def _squared_distance(left, right):
    return sum(((a-b)**2 for a, b in zip(left, right)), F(0))


def distance_table(features, *, bits=80, ids=None):
    """Complete unordered distance table for at most32 stored feature rows."""
    z = _features(features, maximum_rows=32)
    names = _ids(ids, len(z))
    _integer(bits, "distance bits", 1, 512)
    pairs = []
    for i, j in combinations(range(len(z)), 2):
        box = sqrt_enclosure(_squared_distance(z[i], z[j]), bits=bits)
        pairs.append({"indices": [i, j], "query_ids": [names[i], names[j]],
            "distance": {k: box[k] for k in ["lower", "upper", "width"]},
            "squared_exact": box["squared_exact"], "scaled_sqrt_floor": box["scaled_sqrt_floor"],
            "bits": bits})
    return canonical({"schema": "step35-stored-feature-distances-v1", "assurance": ASSURANCE,
        "n": len(z), "dimension": len(z[0]), "query_ids": names, "features_exact": z,
        "feature_digest": digest(z), "input_digest": digest({"ids": names, "features": z}),
        "bits": bits, "pairs": pairs, "pair_count": len(pairs),
        "distance_evaluations": len(pairs), "native_transform_error_enclosed": False})


def _distances(features, ids, supplied, bits):
    _integer(bits, "distance bits", 1, 512)
    created = supplied is None
    table = distance_table(features, bits=bits, ids=ids) if created else supplied
    if not isinstance(table, dict):
        raise FeatureCertificateError("Expected a complete distance-table receipt")
    if (table.get("feature_digest") != digest(features) or
        table.get("input_digest") != digest({"ids": ids, "features": features}) or
        table.get("features_exact") != canonical(features) or table.get("query_ids") != ids or
        table.get("n") != len(features) or table.get("dimension") != len(features[0]) or
        table.get("bits") != bits):
        raise FeatureCertificateError("Distance table is bound to another feature roster")
    pairs = list(combinations(range(len(features)), 2))
    if not isinstance(table.get("pairs"), list) or len(table["pairs"]) != len(pairs):
        raise FeatureCertificateError("Complete unordered distance roster is required")
    values = {}
    for (i, j), record in zip(pairs, table["pairs"]):
        if record.get("indices") != [i, j] or record.get("query_ids") != [ids[i], ids[j]]:
            raise FeatureCertificateError("Distance order or source identity differs")
        lo, hi = _read_box(record["distance"])
        squared = _squared_distance(features[i], features[j])
        if lo < 0 or rational(record["squared_exact"]) != squared or not lo*lo <= squared <= hi*hi:
            raise FeatureCertificateError("Distance bounds do not enclose the exact stored-feature distance")
        values[i, j] = (lo, hi)
    return table, pairs, values, created


def class_descriptor(feature_id, kappa="2", output_interval=("0", "1")):
    feature_id = _text(feature_id, "feature source id")
    kap = rational(kappa)
    low, high = _range(output_interval)
    if kap < 0:
        raise FeatureCertificateError("Sensitivity cannot be negative")
    return canonical({"schema": CLASS_SCHEMA, "feature_id": feature_id,
        "metric": "Euclidean_distance_on_supplied_feature_coordinates",
        "decoder_class": "all_range_valued_kappa_Lipschitz_functions_on_Euclidean_feature_space",
        "kappa": kap, "output_interval": [low, high], "assurance": ASSURANCE})


def class_identity(feature_id, kappa="2", output_interval=("0", "1")):
    return "class35:" + digest(class_descriptor(feature_id, kappa, output_interval))


def head_identity(head):
    return "head35:" + digest(head)


def _conjugate_minimum(target, incidence, low, high):
    minimizer = _clip(target-incidence/2, low, high)
    value = (minimizer-target)**2 + incidence*minimizer
    return value, minimizer


def _dual(targets, multipliers, distances, kappa, endpoint, low, high):
    incidence = [F(0) for _ in targets]
    edges = []
    penalty = F(0)
    index = 0 if endpoint == "lower" else 1
    for (i, j), multiplier in sorted(multipliers.items()):
        if multiplier:
            distance = distances[i, j][index]
            incidence[i] += multiplier
            incidence[j] -= multiplier
            penalty += kappa*abs(multiplier)*distance
            edges.append({"indices": [i, j], "lambda": multiplier,
                "distance_endpoint": distance, "constraint_radius": kappa*distance})
    terms = [_conjugate_minimum(y, s, low, high) for y, s in zip(targets, incidence)]
    raw = sum((v for v, _ in terms), F(0))-penalty
    return {"edges": edges, "incidence": incidence,
        "box_minimizers": [v for _, v in terms], "coordinate_minima": [v for v, _ in terms],
        "penalty": penalty, "unclamped_sse": raw, "sse": max(F(0), raw),
        "mean": max(F(0), raw)/len(targets), "distance_endpoint": endpoint,
        "range_constraints_used_in_dual": True,
        "formula": "sum_i min_{v in I}((v-y_i)^2+s_i*v)-kappa*sum_edges(abs(lambda)*distance_endpoint)",
        "targets_clipped_in_loss": False}


def _primal(values, targets, distances, kappa, low, high, *, selected_from, contraction=None):
    if len(values) != len(targets) or any(not low <= v <= high for v in values):
        raise ArithmeticError("Primal vector violates shape or range")
    slack = []
    for (i, j), (dlo, _) in sorted(distances.items()):
        value = kappa*dlo-abs(values[i]-values[j])
        if value < 0:
            raise ArithmeticError("Conservative primal pair constraint violated")
        slack.append(value)
    sse = sum(((v-y)**2 for v, y in zip(values, targets)), F(0))
    return {"values": list(values), "sse": sse, "mean": sse/len(targets),
        "selected_from": selected_from, "contraction": contraction,
        "all_pair_constraints_verified": True, "pair_checks": len(distances),
        "minimum_pair_slack": min(slack, default=F(0))}


def _contracted_primal(targets, incidence, distances, kappa, low, high, iteration):
    proposal = [_clip(y-s/2, low, high) for y, s in zip(targets, incidence)]
    center = sum(proposal, F(0))/len(proposal)
    factor, limiting_pair = F(1), None
    for (i, j), (dlo, _) in sorted(distances.items()):
        difference = abs(proposal[i]-proposal[j])
        if difference and kappa*dlo/difference < factor:
            factor, limiting_pair = kappa*dlo/difference, [i, j]
    values = [center+factor*(v-center) for v in proposal]
    return _primal(values, targets, distances, kappa, low, high,
        selected_from="contracted_box_dual_residual", contraction={"proposal": proposal,
            "incidence": list(incidence), "center": center, "factor": factor,
            "limiting_pair": limiting_pair, "coordinate_updates": iteration})


def _positive_root(a, b, radius, low, high):
    """Smallest positive root clip(a-t/2)-clip(b+t/2)=radius."""
    previous = F(0)
    difference = _clip(a, low, high)-_clip(b, low, high)
    if not difference > radius:
        raise ArithmeticError("Positive-coordinate root was requested without a violation")
    breakpoints = sorted(set(v for v in [2*(a-high), 2*(a-low), 2*(low-b), 2*(high-b)] if v > 0))
    for value in breakpoints:
        current = _clip(a-value/2, low, high)-_clip(b+value/2, low, high)
        if current <= radius:
            root = previous+(value-previous)*(difference-radius)/(difference-current)
            if _clip(a-root/2, low, high)-_clip(b+root/2, low, high) != radius:
                raise ArithmeticError("Piecewise coordinate root failed exact substitution")
            return root
        previous, difference = value, current
    raise ArithmeticError("Box-coordinate root was not bracketed by its breakpoints")


def _coordinate_optimum(yi, yj, si, sj, old, radius, low, high):
    a, b = yi-(si-old)/2, yj-(sj+old)/2
    difference = _clip(a, low, high)-_clip(b, low, high)
    if difference > radius:
        return _positive_root(a, b, radius, low, high)
    if difference < -radius:
        return -_positive_root(b, a, radius, low, high)
    return F(0)


def _coordinate_change(yi, yj, si, sj, old, proposal, radius, low, high):
    delta = proposal-old
    before = _conjugate_minimum(yi, si, low, high)[0]+_conjugate_minimum(yj, sj, low, high)[0]
    after = _conjugate_minimum(yi, si+delta, low, high)[0]+_conjugate_minimum(yj, sj-delta, low, high)[0]
    return after-before-radius*(abs(proposal)-abs(old))


def _solve(features, targets, *, feature_id, ids, kappa, output_interval, distances,
           distance_bits, max_updates, check_every, gap_sse, grid_bits, fallback_reason, kind):
    z = _features(features, maximum_rows=32)
    names = _ids(ids, len(z))
    y = _targets(targets, len(z))
    kap = rational(kappa)
    low, high = _range(output_interval)
    descriptor = class_descriptor(feature_id, kap, (low, high))
    _integer(max_updates, "maximum coordinate updates", 0, 100000)
    _integer(check_every, "checkpoint frequency", 1, 100000)
    _integer(grid_bits, "multiplier grid bits", 1, 128)
    gap_target = rational(gap_sse)
    if gap_target < 0:
        raise FeatureCertificateError("Optimization gap target cannot be negative")
    if fallback_reason is not None:
        _text(fallback_reason, "fallback reason")
    elif max_updates == 0:
        fallback_reason = "ZERO_UPDATE_BUDGET"
    table, pairs, distance_values, created_distances = _distances(z, names, distances, distance_bits)
    endpoint = "lower" if kind == "training" else "upper"
    distance_index = 0 if endpoint == "lower" else 1
    constant = _clip(sum(y, F(0))/len(y), low, high)
    primal = _primal([constant]*len(y), y, distance_values, kap, low, high,
        selected_from="clipped_empirical_constant_mean")
    constant_primal = dict(primal)
    multipliers = {pair: F(0) for pair in pairs}
    incidence = [F(0) for _ in y]
    grid = 1 << grid_bits
    initialization = []
    initialization_attempts=initialization_accepted=0
    if fallback_reason is None:
        clipped_y = [_clip(t, low, high) for t in y]
        ranked = sorted(pairs, key=lambda ij: (-max(F(0),abs(clipped_y[ij[0]]-clipped_y[ij[1]])-
            kap*distance_values[ij][distance_index]),ij))
        used = set()
        for i, j in ranked:
            if initialization_attempts>=max_updates:
                break
            if i in used or j in used:
                continue
            radius = kap*distance_values[i,j][distance_index]
            if abs(clipped_y[i]-clipped_y[j])<=radius:
                break
            initialization_attempts+=1
            proposal = _coordinate_optimum(y[i],y[j],incidence[i],incidence[j],F(0),radius,low,high)
            proposal = F(proposal.numerator*grid//proposal.denominator,grid)
            change = _coordinate_change(y[i],y[j],incidence[i],incidence[j],F(0),proposal,radius,low,high)
            if proposal and change >= 0:
                multipliers[i,j]=proposal
                incidence[i]+=proposal
                incidence[j]-=proposal
                used.update((i,j))
                initialization_accepted+=1
                initialization.append({"indices":[i,j],"lambda":proposal,"exact_dual_change":change})
    dual = _dual(y,multipliers,distance_values,kap,endpoint,low,high)
    checkpoints=[]
    def checkpoint(iteration):
        nonlocal dual,primal,incidence
        previous_dual,previous_primal=dual["sse"],primal["sse"]
        dual=_dual(y,multipliers,distance_values,kap,endpoint,low,high)
        if dual["incidence"] != incidence or dual["sse"] < previous_dual:
            raise ArithmeticError("Exact dual reconstruction or monotonicity failed")
        if fallback_reason is None:
            candidate=_contracted_primal(y,incidence,distance_values,kap,low,high,iteration)
            if candidate["sse"] < primal["sse"]:
                primal=candidate
        if not 0 <= dual["sse"] <= primal["sse"] <= previous_primal:
            raise ArithmeticError("Primal-dual enclosure order failed")
        checkpoints.append({"coordinate_updates":iteration,"dual_sse":dual["sse"],
            "primal_sse":primal["sse"],"gap_sse":primal["sse"]-dual["sse"],
            "active_edges":len(dual["edges"]),"primal_selected_from":primal["selected_from"]})
    checkpoint(initialization_attempts)
    updates=initialization_attempts
    accepted=initialization_accepted
    cyclic_updates=0
    while fallback_reason is None and pairs and updates < max_updates and primal["sse"]-dual["sse"] > gap_target:
        i,j=pairs[cyclic_updates%len(pairs)]
        old=multipliers[i,j]
        radius=kap*distance_values[i,j][distance_index]
        proposal=_coordinate_optimum(y[i],y[j],incidence[i],incidence[j],old,radius,low,high)
        proposal=F(proposal.numerator*grid//proposal.denominator,grid)
        change=_coordinate_change(y[i],y[j],incidence[i],incidence[j],old,proposal,radius,low,high)
        if change >= 0:
            delta=proposal-old
            multipliers[i,j]=proposal
            incidence[i]+=delta
            incidence[j]-=delta
            accepted+=1
        updates+=1
        cyclic_updates+=1
        if updates%check_every==0 or updates==max_updates:
            checkpoint(updates)
    if checkpoints[-1]["coordinate_updates"] != updates:
        checkpoint(updates)
    gap=primal["sse"]-dual["sse"]
    status=("VALID_CONSTANT_PRIMAL_ZERO_EDGE_DUAL_FALLBACK" if fallback_reason is not None else
        "VERIFIED_BRACKET_TO_DECLARED_GAP" if gap <= gap_target else "VERIFIED_BOUNDED_UNFINISHED")
    record={"schema":"step35-stored-feature-finite-qp-v1","kind":kind,"status":status,
        "assurance":ASSURANCE,"feature_id":feature_id,"class_descriptor":descriptor,
        "class_id":"class35:"+digest(descriptor),"dimension":len(z[0]),"n":len(z),
        "query_ids":names,"features_exact":z,"targets_exact":y,"feature_digest":digest(z),
        "target_digest":digest(y),"input_digest":digest({"ids":names,"features":z,"targets":y,"class":descriptor}),
        "kappa":kap,"output_interval":[low,high],"targets_clipped":False,
        "targets_outside_predictor_range":sum(not low<=t<=high for t in y),
        "distance_table":table,"primal":primal,"dual":dual,
        "optimization":{"status":status,"gap_sse":gap,"gap_target_sse":gap_target,
            "gap_resolved":fallback_reason is None and gap<=gap_target,
            "objective_bracket_sse":_box(dual["sse"],primal["sse"]),
            "objective_bracket_mean":_box(dual["mean"],primal["mean"])},
        "constant_witness":{"value":constant,"sse":constant_primal["sse"],"mean":constant_primal["mean"]},
        "work":{"max_updates":max_updates,"coordinate_updates":updates,"accepted_updates":accepted,
            "initialization_coordinate_attempts":initialization_attempts,"cyclic_coordinate_attempts":cyclic_updates,
            "initialization_included_in_coordinate_budget":True,
            "check_every":check_every,"grid_bits":grid_bits,"checkpoints":checkpoints,
            "initialization_edges":initialization,"distance_sqrt_evaluations":len(pairs) if created_distances else 0,
            "distance_squared_verifications":len(pairs),"pair_count":len(pairs),
            "full_coordinate_trace_retained":False,
            "coordinate_rule":"Exact piecewise box-aware one-coordinate optimum, floor to the declared dyadic grid, accept only a nonnegative exact dual change.",
            "independent_audit_requirement":"Verify final primal and dual witnesses directly; validity does not depend on trusting the optimizer or its unretained transition trace."},
        "fallback_reason":fallback_reason,"native_transform_error_enclosed":False,
        "native_pipeline_roundoff_certified":False,"population_risk_claim":False,
        "future_response_coverage_claim":False,"raw_input_lipschitz_claim":False}
    if kind=="class":
        record["class_risk"]=_box(dual["mean"],primal["mean"])
        record["witness_uses_assessment_targets"]=True
        record["deployable_head_returned"]=False
    else:
        head={"schema":HEAD_SCHEMA,"kind":"symmetric_lipschitz_extension",
            "feature_id":feature_id,"class_id":record["class_id"],"dimension":len(z[0]),
            "kappa":kap,"output_interval":[low,high],"assurance":ASSURANCE,
            "anchor_ids":names,"anchor_features":z,"anchor_values":primal["values"],
            "formula":"clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)"}
        record["head"]=head
        record["head_id"]=head_identity(head)
        record["membership"]={"status":"VERIFIED_GLOBAL_FEATURE_SPACE_LIPSCHITZ_AND_RANGE",
            "anchor_consistency_verified":True,"interpolation_verified":True,
            "global_feature_space_lipschitz_requires_anchor_consistency":False,
            "proof":"Distance cones, finite maxima/minima, their arithmetic mean and clipping preserve kappa-Lipschitz continuity. Conservative pair feasibility gives interpolation at the exact stored anchors."}
        record["training_only_requires_parent_roster_binding"]=True
        record["full_class_lower_bound_claim"]=False
    return canonical(record)


def fit_anchor_decoder(features,targets,*,feature_id,ids=None,kappa="2",output_interval=("0","1"),
        distances=None,distance_bits=80,max_updates=4000,check_every=20,gap_sse="1/10000000",grid_bits=40,fallback_reason=None):
    return _solve(features,targets,feature_id=feature_id,ids=ids,kappa=kappa,output_interval=output_interval,
        distances=distances,distance_bits=distance_bits,max_updates=max_updates,check_every=check_every,
        gap_sse=gap_sse,grid_bits=grid_bits,fallback_reason=fallback_reason,kind="training")


def class_risk_certificate(features,targets,*,feature_id,ids=None,kappa="2",output_interval=("0","1"),
        distances=None,distance_bits=80,max_updates=4000,check_every=20,gap_sse="1/10000000",grid_bits=40,fallback_reason=None):
    return _solve(features,targets,feature_id=feature_id,ids=ids,kappa=kappa,output_interval=output_interval,
        distances=distances,distance_bits=distance_bits,max_updates=max_updates,check_every=check_every,
        gap_sse=gap_sse,grid_bits=grid_bits,fallback_reason=fallback_reason,kind="class")


def make_affine_head(coefficients,intercept,*,feature_id,kappa="2",output_interval=("0","1"),
        project=True,center_features=None,center_target=None,bits=80):
    _integer(bits,"distance bits",1,512)
    if not isinstance(coefficients,(list,tuple)) or not coefficients:
        raise FeatureCertificateError("Require a nonempty coefficient vector")
    beta=[rational(v) for v in coefficients]
    b=rational(intercept)
    kap=rational(kappa)
    low,high=_range(output_interval)
    descriptor=class_descriptor(feature_id,kap,(low,high))
    if type(project) is not bool:
        raise FeatureCertificateError("Projection switch must be boolean")
    squared=sum((v*v for v in beta),F(0))
    original_beta,original_intercept=list(beta),b
    factor=F(1)
    if squared>kap*kap:
        if not project:
            raise FeatureCertificateError("Affine coefficient norm exceeds the class sensitivity")
        upper=rational(sqrt_enclosure(squared,bits=bits)["upper"])
        factor=kap/upper
        beta=[factor*v for v in beta]
    if (center_features is None)!=(center_target is None):
        raise FeatureCertificateError("Both training center fields are required together")
    center=None
    if center_features is not None:
        if not isinstance(center_features,(list,tuple)) or len(center_features)!=len(beta):
            raise FeatureCertificateError("Affine training center has wrong dimension")
        center=[rational(v) for v in center_features]
        center_y=rational(center_target)
        b=center_y-sum((v*x for v,x in zip(beta,center)),F(0))
    actual_squared=sum((v*v for v in beta),F(0))
    if actual_squared>kap*kap:
        raise ArithmeticError("Exact affine norm projection failed")
    head={"schema":HEAD_SCHEMA,"kind":"clipped_affine","feature_id":feature_id,
        "class_id":"class35:"+digest(descriptor),"dimension":len(beta),"kappa":kap,
        "output_interval":[low,high],"assurance":ASSURANCE,
        "coefficients":beta,"intercept":b}
    return canonical({"schema":"step35-admissible-affine-head-v1","head":head,"head_id":head_identity(head),
        "class_id":head["class_id"],"feature_id":feature_id,
        "membership":{"status":"VERIFIED_EXACT_AFFINE_CLASS_MEMBERSHIP","coefficient_squared_norm":actual_squared,
            "kappa_squared":kap*kap,"proof":"Cauchy-Schwarz followed by nonexpansive range clipping."},
        "projection":{"source_coefficients":original_beta,"source_intercept":original_intercept,
            "source_squared_norm":squared,"factor":factor,"projected":factor!=1,
            "center_features":center,"center_target":None if center is None else center_y,
            "intercept_recentered":center is not None},
        "native_transform_error_enclosed":False,"population_risk_claim":False})


def _head(value):
    if not isinstance(value,dict):
        raise FeatureCertificateError("Head must be a dictionary")
    wrapper=value if "head" in value else None
    head=value["head"] if wrapper is not None else value
    if not isinstance(head,dict) or head.get("schema")!=HEAD_SCHEMA or head.get("assurance")!=ASSURANCE:
        raise FeatureCertificateError("Unknown head schema or assurance level")
    feature_id=_text(head.get("feature_id"),"feature source id")
    kap=rational(head["kappa"])
    low,high=_range(head["output_interval"])
    if head.get("class_id")!=class_identity(feature_id,kap,(low,high)):
        raise FeatureCertificateError("Head class identity does not match its declared metric/range/sensitivity")
    head_id=head_identity(head)
    if wrapper is not None and wrapper.get("head_id")!=head_id:
        raise FeatureCertificateError("Head content identity differs")
    dimension=head.get("dimension")
    if type(dimension) is not int or dimension<1:
        raise FeatureCertificateError("Invalid head feature dimension")
    kind=head.get("kind")
    if kind=="clipped_affine":
        beta=[rational(v) for v in head["coefficients"]]
        if len(beta)!=dimension or sum((b*b for b in beta),F(0))>kap*kap:
            raise FeatureCertificateError("Affine head does not satisfy exact norm membership")
        rational(head["intercept"])
    elif kind=="symmetric_lipschitz_extension":
        anchors=_features(head["anchor_features"],maximum_rows=32)
        _ids(head["anchor_ids"],len(anchors))
        values=_targets(head["anchor_values"],len(anchors))
        if len(anchors[0])!=dimension or any(not low<=v<=high for v in values):
            raise FeatureCertificateError("Extension anchor shape or range differs")
    else:
        raise FeatureCertificateError("Unknown head formula")
    return head,head_id,kap,low,high


def loss_bounds(predictions,targets):
    if not isinstance(predictions,(list,tuple)) or not predictions:
        raise FeatureCertificateError("Require nonempty prediction enclosures")
    y=_targets(targets,len(predictions))
    losses=[]
    for value,target in zip(predictions,y):
        lo,hi=_read_box(value)
        a,b=lo-target,hi-target
        lower=F(0) if a<=0<=b else min(a*a,b*b)
        upper=max(a*a,b*b)
        losses.append(_box(lower,upper))
    lower=sum((v["lower"] for v in losses),F(0))/len(losses)
    upper=sum((v["upper"] for v in losses),F(0))/len(losses)
    return canonical({**_box(lower,upper),"n":len(losses),"per_row":losses,"targets_clipped":False})


def _prediction_record(head,head_id,z,ids,targets,rows,*,distance_evaluations,bits,status):
    predictions=[row["prediction"] for row in rows]
    y=None if targets is None else _targets(targets,len(z))
    losses=None if y is None else loss_bounds(predictions,y)
    if losses is not None:
        for row,loss in zip(rows,losses["per_row"]):
            row["squared_loss"]=loss
    return canonical({"schema":"step35-stored-feature-prediction-receipt-v1","status":status,
        "assurance":ASSURANCE,"feature_id":head["feature_id"],"class_id":head["class_id"],
        "head_id":head_id,"head":head,"query_ids":ids,"n":len(z),"dimension":len(z[0]),
        "features_exact":z,"feature_digest":digest(z),"query_roster_digest":digest({"ids":ids,"features":z}),
        "targets_exact":y,"target_digest":None if y is None else digest(y),
        "predictions":predictions,"rows":rows,"loss":losses,
        "distance_evaluations":distance_evaluations,"distance_bits":bits,
        "targets_outside_predictor_range":None if y is None else sum(not rational(head["output_interval"][0])<=v<=rational(head["output_interval"][1]) for v in y),
        "targets_clipped":False,"native_transform_error_enclosed":False,
        "native_pipeline_roundoff_certified":False,"population_risk_claim":False,
        "future_response_coverage_claim":False,"raw_input_lipschitz_claim":False})


def evaluate_affine(head,query_features,targets=None,*,ids=None):
    head,head_id,kap,low,high=_head(head)
    if head["kind"]!="clipped_affine":
        raise FeatureCertificateError("Expected a clipped affine head")
    z=_features(query_features)
    names=_ids(ids,len(z))
    if len(z[0])!=head["dimension"]:
        raise FeatureCertificateError("Query dimension differs from frozen affine head")
    beta=[rational(v) for v in head["coefficients"]]
    intercept=rational(head["intercept"])
    rows=[]
    for i,vector in enumerate(z):
        raw=intercept+sum((b*v for b,v in zip(beta,vector)),F(0))
        value=_clip(raw,low,high)
        rows.append({"query_index":i,"query_id":names[i],"raw_prediction":_box(raw,raw),
            "prediction":_box(value,value)})
    return _prediction_record(head,head_id,z,names,targets,rows,distance_evaluations=0,bits=None,
        status="VERIFIED_EXACT_STORED_FEATURE_AFFINE_PREDICTIONS")


def evaluate_extension(head,query_features,targets=None,*,ids=None,bits=80):
    head,head_id,kap,low,high=_head(head)
    if head["kind"]!="symmetric_lipschitz_extension":
        raise FeatureCertificateError("Expected a symmetric Lipschitz extension head")
    _integer(bits,"distance bits",1,512)
    z=_features(query_features)
    names=_ids(ids,len(z))
    if len(z[0])!=head["dimension"]:
        raise FeatureCertificateError("Query dimension differs from frozen extension")
    anchors=_features(head["anchor_features"],maximum_rows=32)
    values=[rational(v) for v in head["anchor_values"]]
    rows=[]
    for i,vector in enumerate(z):
        distances=[];lower_cones=[];upper_cones=[]
        for j,(anchor,value) in enumerate(zip(anchors,values)):
            enclosure=sqrt_enclosure(_squared_distance(vector,anchor),bits=bits)
            dlo,dhi=rational(enclosure["lower"]),rational(enclosure["upper"])
            lower_cones.append((value-kap*dhi,value-kap*dlo))
            upper_cones.append((value+kap*dlo,value+kap*dhi))
            distances.append({"anchor_index":j,"anchor_id":head["anchor_ids"][j],
                "distance":{k:enclosure[k] for k in ["lower","upper","width"]},
                "squared_exact":enclosure["squared_exact"],"scaled_sqrt_floor":enclosure["scaled_sqrt_floor"]})
        lower=(max(v[0] for v in lower_cones),max(v[1] for v in lower_cones))
        upper=(min(v[0] for v in upper_cones),min(v[1] for v in upper_cones))
        raw=((lower[0]+upper[0])/2,(lower[1]+upper[1])/2)
        prediction=(_clip(raw[0],low,high),_clip(raw[1],low,high))
        rows.append({"query_index":i,"query_id":names[i],"anchor_distances":distances,
            "lower_envelope":_box(*lower),"upper_envelope":_box(*upper),
            "raw_prediction":_box(*raw),"prediction":_box(*prediction)})
    return _prediction_record(head,head_id,z,names,targets,rows,distance_evaluations=len(z)*len(anchors),bits=bits,
        status="VERIFIED_STORED_FEATURE_EXTENSION_ENCLOSURES")


def fallback_prediction(head,query_features,targets=None,*,ids=None,reason):
    head,head_id,kap,low,high=_head(head)
    _text(reason,"fallback reason")
    z=_features(query_features)
    names=_ids(ids,len(z))
    if len(z[0])!=head["dimension"]:
        raise FeatureCertificateError("Fallback query feature dimension differs")
    rows=[{"query_index":i,"query_id":names[i],"prediction":_box(low,high)} for i in range(len(z))]
    result=_prediction_record(head,head_id,z,names,targets,rows,distance_evaluations=0,bits=None,
        status="VALID_PREDICTOR_RANGE_FALLBACK")
    result["fallback_reason"]=reason
    return result


def aggregate_class_risk(block_certificates,global_head_receipts=(),*,include_diagnostic_constant=False):
    """Combine disjoint lower witnesses, never independently fitted block uppers.

    Input certificates/prediction enclosures are premises whose direct proof
    checks belong to the independent audit. This routine verifies identities,
    cohort concatenation and the aggregation algebra. Supplied global in-class
    frozen head losses provide union uppers. An optional single diagnostic
    constant uses the assessment targets and is not a training-only head.
    """
    if not isinstance(block_certificates,(list,tuple)) or not block_certificates:
        raise FeatureCertificateError("Require a nonempty ordered disjoint block roster")
    if type(include_diagnostic_constant) is not bool:
        raise FeatureCertificateError("Diagnostic constant switch must be boolean")
    if not isinstance(global_head_receipts,(list,tuple)):
        raise FeatureCertificateError("Global head receipts must be an explicit sequence")
    first=block_certificates[0]
    feature_id,class_id=first["feature_id"],first["class_id"]
    low,high=_range(first["output_interval"])
    kap=rational(first["kappa"])
    if class_id!=class_identity(feature_id,kap,(low,high)):
        raise FeatureCertificateError("Block class identity differs")
    ids=[];z=[];y=[];lower_sse=F(0)
    block_rows=[]
    for block in block_certificates:
        if (block.get("kind")!="class" or block.get("feature_id")!=feature_id or
            block.get("class_id")!=class_id or block.get("assurance")!=ASSURANCE):
            raise FeatureCertificateError("Blocks belong to different classes or assurance levels")
        bz=_features(block["features_exact"],maximum_rows=32)
        by=_targets(block["targets_exact"],len(bz));bids=_ids(block["query_ids"],len(bz))
        blo,bhi=_read_box(block["class_risk"])
        if blo<0 or blo!=rational(block["dual"]["sse"])/len(bz):
            raise FeatureCertificateError("Block lower bound is not its retained dual mean")
        if block["feature_digest"]!=digest(bz) or block["target_digest"]!=digest(by):
            raise FeatureCertificateError("Block feature or target digest differs")
        lower_sse+=len(bz)*blo;ids+=bids;z+=bz;y+=by
        block_rows.append({"n":len(bz),"query_ids":bids,"lower":blo,"certificate_digest":digest(block)})
    if len(set(ids))!=len(ids):
        raise FeatureCertificateError("Aggregation blocks overlap in query identifiers")
    _features(z)
    n=len(y)
    candidates=[]
    if include_diagnostic_constant:
        center=_clip(sum(y,F(0))/n,low,high)
        constant_loss=sum(((center-t)**2 for t in y),F(0))/n
        candidates.append({"kind":"single_global_diagnostic_constant_witness","value":center,
            "upper":constant_loss,"uses_assessment_targets":True,"training_only_head":False})
    for receipt in global_head_receipts:
        head,head_id,_,_,_=_head(receipt["head"])
        if (receipt.get("assurance")!=ASSURANCE or receipt.get("class_id")!=class_id or
            head["class_id"]!=class_id or receipt.get("head_id")!=head_id or receipt.get("query_ids")!=ids or
            receipt.get("features_exact")!=canonical(z) or receipt.get("targets_exact")!=canonical(y)):
            raise FeatureCertificateError("Global head receipt is not on the exact ordered block union")
        loss=loss_bounds(receipt["predictions"],y)
        if any(not low<=_read_box(p)[0]<=_read_box(p)[1]<=high for p in receipt["predictions"]):
            raise FeatureCertificateError("Global head prediction enclosure exceeds its declared range")
        if receipt.get("loss")!=loss:
            raise FeatureCertificateError("Global head loss does not match retained prediction enclosures")
        candidates.append({"kind":"single_global_frozen_head","head_id":head_id,
            "upper":rational(loss["upper"]),"prediction_receipt_digest":digest(receipt)})
    if not candidates:
        raise FeatureCertificateError("Supply a frozen global head receipt for the union upper")
    upper=min(v["upper"] for v in candidates)
    lower=lower_sse/n
    if not 0<=lower<=upper:
        raise ArithmeticError("Aggregated lower and common-witness upper contradict")
    return canonical({"schema":"step35-aggregated-stored-feature-class-risk-v1",
        "status":"VERIFIED_AGGREGATION_OF_RETAINED_CLASS_AND_HEAD_WITNESSES",
        "assurance":ASSURANCE,"feature_id":feature_id,"class_id":class_id,"n":n,
        "query_ids":ids,"feature_digest":digest(z),"target_digest":digest(y),
        "class_risk":_box(lower,upper),"blocks":block_rows,"global_upper_candidates":candidates,
        "diagnostic_assessment_constant_included":include_diagnostic_constant,
        "block_primals_averaged_for_union_upper":False,
        "input_certificate_and_prediction_enclosures_require_independent_verification":True,
        "targets_clipped":False,"population_risk_claim":False,"future_response_coverage_claim":False})
