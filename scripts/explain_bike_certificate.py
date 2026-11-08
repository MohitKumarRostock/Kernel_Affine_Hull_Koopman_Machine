#!/usr/bin/env python3
"""Explain one stored Bike class certificate; no fitting or full certificate replay.

Run from a repository checkout with Python 3.9+ (standard library only):
    python3 scripts/explain_bike_certificate.py
    python3 scripts/explain_bike_certificate.py --json

The interval and decision comparisons below use exact rational arithmetic.
Feasibility, dual bounds and fitted-predictor membership are inherited from the
recorded study, not independently proved by this explanatory reader.
"""

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys


GROUP = "experiments/class_diagnosis"
TASK = "registered"
MAP = "kahm_minmax01_cap4_p1"
PATHS = {
    "certificate": f"{GROUP}/records/classes/{TASK}/{MAP}.json",
    "protocol": f"{GROUP}/protocol/protocol.json",
    "prior_protocol": f"{GROUP}/inputs/prior/protocol36.json",
    "cohort": f"{GROUP}/inputs/decision/{TASK}/cohort.json",
}
PREDICTOR_NAMES = {
    "affine": "Clipped affine ridge fit",
    "extension32": "Clipped Lipschitz extension fit (32 training anchors)",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    """Reject floating-point input so all decision arithmetic remains exact."""
    require(type(value) in (str, int), "Expected an exact integer/fraction string")
    return Fraction(value)


def approximate(value):
    """Decimals are for display only, never for a decision."""
    value = rational(value)
    with localcontext() as context:
        context.prec = 30
        return format(Decimal(value.numerator) / Decimal(value.denominator), ".10g")


def checked_interval(record, name, require_width=True):
    lower, upper = rational(record["lower"]), rational(record["upper"])
    require(0 <= lower <= upper, f"Invalid nonnegative interval: {name}")
    if require_width:
        require(rational(record["width"]) == upper - lower,
                f"Inconsistent interval width: {name}")
    return lower, upper


def read_inputs(root):
    """Check selected files against the repository's own content manifest."""
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["algorithm"] == "sha256", "Unsupported manifest algorithm")
    records, integrity = {}, []
    for name, relative in PATHS.items():
        matches = [entry for entry in manifest["files"] if entry["path"] == relative]
        require(len(matches) == 1, f"Expected one manifest entry for {relative}")
        expected = matches[0]
        raw = (root / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(len(raw) == expected["bytes"], f"Size mismatch: {relative}")
        require(digest == expected["sha256"], f"SHA256 mismatch: {relative}")
        records[name] = json.loads(raw.decode("utf-8"))
        integrity.append({"path": relative, "bytes": len(raw), "sha256": digest})
    return records, integrity


def explain(root):
    records, integrity = read_inputs(root)
    cert, protocol = records["certificate"], records["protocol"]
    prior, cohort = records["prior_protocol"], records["cohort"]
    require(cert["schema"] == "step37-full-graph-class-diagnostic-v1",
            "Unsupported certificate schema")
    require(cert["task"] == TASK and cert["map"] == MAP, "Unexpected case identity")
    require(cohort["target"] == TASK and cohort["name"] == "decision",
            "Unexpected assessment cohort")
    require(cert["n"] == len(cohort["ids"]) == len(cohort["y"]) == 172,
            "Unexpected cohort size")
    require(cert["ids"] == cohort["ids"] and len(set(cohort["ids"])) == cert["n"],
            "Certificate/cohort row identities do not agree")
    require(cert["dimension"] == 8, "Unexpected feature dimension")
    binding = cert["input_binding"]
    require(f"{GROUP}/{binding['cohort_path']}" == PATHS["cohort"],
            "Unexpected bound cohort path")
    cohort_hash = next(item["sha256"] for item in integrity
                       if item["path"] == PATHS["cohort"])
    require(binding["cohort_sha256"] == cohort_hash, "Cohort binding mismatch")

    # Confirm the class and loss used by this particular example. Targets may
    # lie outside the prediction interval; they must not be clipped in the loss.
    require(rational(cert["kappa"]) == rational(protocol["class"]["kappa"]) == 2,
            "Unexpected Lipschitz budget")
    require([rational(v) for v in cert["output_interval"]] == [Fraction(0), Fraction(1)],
            "Unexpected output interval")
    require(cert["output_interval"] == protocol["class"]["prediction_interval"],
            "Protocol/certificate output intervals differ")
    require(protocol["class"]["norm"] == "Euclidean", "Unexpected metric")
    require(protocol["class"]["loss"] == "original_unclipped_target_squared_error",
            "Unexpected loss")
    require(prior["data"]["targets_clipped"] is False, "Targets must remain unclipped")
    require(prior["data"]["horizon_hours"] == cohort["roster"]["horizon_hours"] == 24,
            "Unexpected forecasting horizon")
    require(rational(prior["selection"]["epsilon_factor"]) == Fraction(19, 20),
            "Unexpected threshold rule")
    require(prior["heads"]["anchor_count"] == 32, "Unexpected extension fit")
    outside = sum(not 0 <= rational(y) <= 1 for y in cohort["y"])
    require(outside == cohort["roster"]["target_outside_prediction_range_count"],
            "Target range count does not match the roster")

    epsilon = rational(cert["epsilon"])
    require(epsilon > 0 and epsilon == rational(protocol["thresholds"][TASK]["epsilon"]),
            "Threshold does not match the protocol")
    lower, upper = checked_interval(cert["class_risk"], "class optimum")
    decision = "ADEQUATE" if upper <= epsilon else (
        "INADEQUATE" if lower > epsilon else "UNRESOLVED")
    require(decision == cert["diagnosis"] == "ADEQUATE",
            "This example no longer has the expected class-adequacy decision")

    predictors = []
    require(sorted(h["recipe"] for h in cert["inherited_heads"]) == sorted(PREDICTOR_NAMES),
            "Unexpected fitted-predictor roster")
    for head in cert["inherited_heads"]:
        h_lower, h_upper = checked_interval(head["decision_risk"], head["recipe"])
        # The excess interval compares a fitted predictor's risk with the
        # best risk in its declared class, conditional on inherited membership.
        e_lower, e_upper = max(Fraction(0), h_lower - upper), max(Fraction(0), h_upper - lower)
        stored_excess = checked_interval(head["excess_risk"], "excess risk", require_width=False)
        require(stored_excess == (e_lower, e_upper), "Stored excess interval does not agree")
        meets, fails = h_upper <= epsilon, h_lower > epsilon
        require(head["verified_eligible"] is meets and head["verified_ineligible"] is fails,
                "Stored fitted-predictor threshold flags do not agree")
        require(head["positive_certified_excess"] is (e_lower > 0),
                "Stored positive-excess flag does not agree")
        require(fails, "This example no longer has the expected fitted-predictor failure")
        predictors.append({
            "name": PREDICTOR_NAMES[head["recipe"]],
            "stored_fields": {key: head[key] for key in (
                "recipe", "candidate_id", "decision_risk", "excess_risk",
                "verified_eligible", "verified_ineligible", "positive_certified_excess")},
            "computed": {"meets_requirement": meets, "fails_requirement": fails,
                         "excess_lower": str(e_lower), "excess_upper": str(e_upper)},
        })

    scope = cert["scope"]
    require(scope["diagnostic_uses_decision_labels"] is True
            and scope["population_risk_claim"] is False
            and scope["future_response_coverage_claim"] is False
            and scope["assignment_is_a_trained_prediction_head"] is False,
            "Unexpected certificate scope")
    return {
        "mode": "stored_certificate_interpretation",
        "check_status": "CONSISTENT",
        "question": "Can a decoder class meet a requirement when its fitted predictors fail?",
        "case": {"dataset": prior["data"]["dataset"], "target": TASK,
                 "horizon_hours": prior["data"]["horizon_hours"], "map": MAP,
                 "observations": cert["n"], "feature_dimension": cert["dimension"],
                 "first_target": cohort["roster"]["first_target"],
                 "last_target": cohort["roster"]["last_target"],
                 "targets_outside_output_interval": outside},
        "decoder_class": {"metric": protocol["class"]["norm"], "kappa": cert["kappa"],
                          "output_interval": cert["output_interval"],
                          "assurance": cert["assurance"]},
        "loss": "Mean squared error on normalized, unclipped cohort targets",
        "target_normalization": prior["data"]["target_normalization"],
        "threshold": {"exact": cert["epsilon"], "approximate": approximate(cert["epsilon"]),
                      "rule": "0.95 times exact normalized training persistence MSE"},
        "class_risk": cert["class_risk"],
        "class_risk_meaning": (
            "An interval enclosing the smallest average squared error among all admissible "
            "decoders on this cohort. The lower endpoint rules out smaller errors; "
            "the feasible upper endpoint shows an error attainable at that level or below."
        ),
        "excess_risk_meaning": "A fitted predictor's cohort risk minus the best risk in its declared class.",
        "stored_class_diagnosis": cert["diagnosis"], "computed_class_diagnosis": decision,
        "predictors": predictors,
        "content_integrity": {"reference": "Repository MANIFEST.json", "checked_files": integrity},
        "independently_checked_here": [
            "Selected file sizes and SHA256 digests against the repository manifest",
            "Case, cohort identity, declared class and threshold consistency",
            "Exact interval widths, threshold comparisons and excess-risk arithmetic",
        ],
        "inherited_not_recomputed_here": [
            "All pairwise feature distances and feasibility of predictions under the class constraints (primal checks)",
            "The lower-bound calculation using signed edge multipliers (dual witness)",
            "Fitted-predictor class membership and achieved-risk enclosures",
            "Training persistence MSE used to choose the stored threshold",
        ],
        "outside_the_guarantee": "Numerical error in constructing the stored feature coordinates is not enclosed.",
        "interpretation": [
            "The stored feasible class witness meets the requirement on these 172 observations.",
            "Both fitted predictors fail that same requirement; their failure does not exclude the class.",
            "The class witness uses assessment targets and is not a predictor fitted on training data.",
            "This is a fixed-cohort capacity diagnosis, without a population or future-forecast guarantee.",
            "The example does not establish the cause of fit failure or superiority of this representation.",
        ],
        "fuller_replay_command": "python3 experiments/class_diagnosis/replay.py --output reproduction_class_diagnosis",
        "fuller_replay_scope": (
            "Rechecks pairwise constraints, lower-bound witnesses and excess arithmetic for all 54 classes; "
            "fitted-predictor membership/risk proofs remain inherited. No fitting or optimizer rerun."
        ),
    }


def print_explanation(result):
    case, threshold, bounds = result["case"], result["threshold"], result["class_risk"]
    print(result["question"])
    print("Mode: interpreting stored evidence; selected-file integrity and exact arithmetic checks.")
    print(f"\nCase: {case['dataset']}, registered rentals, {case['horizon_hours']} hours ahead.")
    print(f"Fixed cohort: {case['observations']} observations, {case['first_target']} to {case['last_target']}.")
    print("Representation: Kernel Affine Hull Machine (KAHM) features.")
    print(f"  Stored map: {case['map']} ({case['feature_dimension']} coordinates).")
    print("Class: all Euclidean 2-Lipschitz scalar decoders into [0,1].")
    print("A distance d between feature vectors allows an output difference of at most 2d.")
    print("Risk: mean squared error after training-minimum/range target normalization.")
    print(f"Targets are not clipped; {case['targets_outside_output_interval']} lie outside [0,1].")
    print(f"\nRequirement: risk <= {threshold['exact']} (approximately {threshold['approximate']}).")
    print(f"Threshold rule: {threshold['rule']}.")
    print("All decisions below use exact fractions. Displayed interval decimals are approximate.")
    print(f"\nStored bounds on best class risk: [{approximate(bounds['lower'])}, {approximate(bounds['upper'])}].")
    print("  This interval encloses the smallest average squared error among all admissible decoders.")
    print("  Its lower endpoint rules out smaller errors on this cohort.")
    print("  Its feasible upper endpoint shows an error attainable at that level or below.")
    print(f"  Upper <= threshold: {result['computed_class_diagnosis']} on this cohort.")
    print(f"  Stored lower/upper sources: {bounds['lower_source']} / {bounds['upper_source']}.")
    print(f"  Interval width: approximately {approximate(bounds['width'])}.")
    print("Excess risk is a fitted predictor's error minus the best risk in the class.")
    for predictor in result["predictors"]:
        fields = predictor["stored_fields"]
        risk, excess = fields["decision_risk"], fields["excess_risk"]
        print(f"\n{predictor['name']}:")
        print(f"  Risk: [{approximate(risk['lower'])}, {approximate(risk['upper'])}].")
        print("  Lower > threshold: FAILS the requirement on this cohort.")
        print(f"  Excess above the class optimum: [{approximate(excess['lower'])}, {approximate(excess['upper'])}].")
    print("\nWhat this means:")
    for line in result["interpretation"]:
        print(f"  - {line}")
    print("\nChecked here (against the repository's manifest and stored records):")
    for line in result["independently_checked_here"]:
        print(f"  - {line}")
    print("Inherited from the recorded study; not re-proved by this reader:")
    for line in result["inherited_not_recomputed_here"]:
        print(f"  - {line}")
    print("\nThe certificate treats stored binary64 feature coordinates as exact.")
    print(result["outside_the_guarantee"])
    print("Record field 'verified_eligible' means meeting the threshold, not class membership.")
    print("\nFor the fuller existing class-diagnosis replay (install repository requirements first):")
    print(f"  {result['fuller_replay_command']}")
    print(f"  {result['fuller_replay_scope']}")
    print("Use --json for exact endpoints, raw decision fields and source-file hashes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Print exact machine-readable results")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root (default: parent of this script's directory)")
    args = parser.parse_args()
    try:
        result = explain(args.repo_root.resolve())
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_explanation(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
