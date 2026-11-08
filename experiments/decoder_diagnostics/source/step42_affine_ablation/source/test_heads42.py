"""Artificial-only validation of the matched Step 42 head factorial.

Checks use analytic ridge solutions, exact center/norm invariants, an active
projection counterexample, an inactive-projection rounding example, and
byte-for-byte comparisons with the copied inherited evaluator.  No public
measurement, stored real feature array, or real fitted payload is opened.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction as F
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import heads42 as h


ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(__file__).resolve().parent / "legacy42"
LAMBDAS = ("1/1000000", "1/10000", "1/100", "1", "100")
CHECKS: list[str] = []
COUNTS = {"synthetic_base_solves": 0, "synthetic_legacy_solves": 0,
          "factorial_variants": 0, "legacy_bridge_packets": 0}


def check(condition, label):
    if not bool(condition):
        raise AssertionError(label)
    CHECKS.append(label)


def rejects(call, label, errors=(ValueError, ArithmeticError)):
    try:
        call()
    except errors:
        CHECKS.append(label)
    else:
        raise AssertionError(label + ": expected rejection")


def fit(z, y, **kwargs):
    result = h.fit_base(z, y, **kwargs)
    COUNTS["synthetic_base_solves"] += 1
    return result


def load_legacy():
    attribution = json.loads((LEGACY / "attribution42.json").read_text())
    for row in attribution["files"]:
        check(hashlib.sha256((LEGACY / row["file"]).read_bytes()).hexdigest() == row["sha256"],
              "Copied inherited source hash: " + row["file"])
    sys.path.insert(0, str(LEGACY))
    modules = tuple(importlib.import_module(name)
                    for name in ("finite_feature35", "affine41", "certificate41"))
    for module in modules:
        check(Path(module.__file__).resolve().parent == LEGACY.resolve(),
              "Inherited module imported only from copied source: " + module.__name__)
    return modules


def analytic_checks():
    # Centered two-point regression: beta=0.5/(1+lambda).  At lambda=1,
    # n*lambda=2 makes beta=1/4; using sum-loss lambda would give beta=1/3.
    z = np.asarray([[-1.0], [1.0]])
    base = fit(z, ["0", "1"], feature_id="synthetic42:analytic", ridge_lambda="1")
    check(base["coefficients"] == ["1/4"] and base["intercept"] == "1/2",
          "Analytic mean-loss ridge solution distinguishes n*lambda from lambda")
    check(base["penalty_multiplier"] == "2", "Analytic ridge penalty multiplier")
    duplicate = fit(np.tile(z, (3, 1)), ["0", "1"] * 3,
                    feature_id="synthetic42:duplicates", ridge_lambda="1")
    check(duplicate["coefficients"] == base["coefficients"]
          and duplicate["intercept"] == base["intercept"],
          "Replicating all training rows preserves a mean-loss ridge solution")
    shifted = fit(z + 8, ["3", "4"], feature_id="synthetic42:translated", ridge_lambda="1")
    check(shifted["coefficients"] == ["1/4"] and shifted["intercept"] == "3/2",
          "Unpenalized intercept handles exact feature and target translations")

    # A non-dyadic target mean makes exact recentering genuinely different,
    # even though the slope is below kappa and no projection is applied.
    rounding = fit([[0.0], [1.0], [3.0]], ["0", "1/3", "1"],
                   feature_id="synthetic42:rounding", ridge_lambda="1/1000000")
    off = h.make_variant(rounding, project=False, recenter=False, clip=False)
    on = h.make_variant(rounding, project=False, recenter=True, clip=False)
    check(F(rounding["coefficient_squared_norm"]) < 4,
          "Rounding-only recenter example has inactive projection")
    check(off["head"]["coefficients"] == on["head"]["coefficients"]
          and F(on["transformation"]["recenter_delta"]) != 0,
          "Inactive-projection recentering isolates an exact intercept correction")
    check(on["structural_claims"]["training_mean_affine_matches_target"],
          "Exact recentering restores the exact rational target mean")
    check(not off["structural_claims"]["training_mean_affine_matches_target"],
          "Binary64 intercept does not falsely claim exact rational centering")

    # Small feature spread demands slope near four.  Projection to two while
    # retaining b changes the prediction at the training mean substantially.
    active = fit([[0.0], [0.125], [0.25]], ["0", "1/2", "1"],
                 feature_id="synthetic42:active_projection", ridge_lambda="1/1000000")
    p0 = h.make_variant(active, project=False, recenter=False, clip=True)
    p1r0 = h.make_variant(active, project=True, recenter=False, clip=True)
    p1r1 = h.make_variant(active, project=True, recenter=True, clip=True)
    check(F(active["coefficient_squared_norm"]) > 4,
          "Projection fixture exceeds the inherited coefficient bound")
    check(not p0["structural_claims"]["norm_within_kappa"],
          "Projection-off oversize coefficients make no false kappa claim")
    check(p1r0["head"]["intercept"] == active["intercept"]
          and p1r0["transformation"]["recenter_delta"] == "0",
          "Recenter-off retains base intercept after active coefficient projection")
    check(p1r1["head"]["coefficients"] == ["2"]
          and F(p1r1["transformation"]["recenter_delta"]) > F(1, 8),
          "Recenter-on corrects a substantive projection-induced mean displacement")
    check(p1r1["transformation"]["training_mean_prediction_before_clipping"] == "1/2",
          "Projected and recentered affine training mean is exactly preserved")

    # A genuinely irrational norm exercises the upward sqrt rule; the lower
    # neighboring 80-bit dyadic must lie strictly below the exact norm.
    irrational = fit([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]],
                     ["-6", "6", "-6", "6"],
                     feature_id="synthetic42:irrational_norm", ridge_lambda="1/2")
    check(irrational["coefficients"] == ["3", "3"],
          "Analytic two-coordinate fixture has irrational Euclidean norm")
    shrunk = h.make_variant(irrational, project=True, recenter=True, clip=True)
    factor = F(shrunk["transformation"]["projection_factor"])
    upper = 2 / factor
    check((upper - F(1, 1 << 80)) ** 2 < 18 <= upper ** 2,
          "Projection uses the tight upward 80-bit dyadic norm bound")
    check(F(shrunk["transformation"]["coefficient_squared_norm"]) < 4,
          "Irrational-norm projection is exactly conservative")
    return active


def factorial_checks(legacy_affine, legacy_cert):
    rng = np.random.default_rng(4201)
    random_z = rng.normal(size=(9, 3)) * np.asarray([0.1, 2.0, 1.0])
    random_y = [str(F.from_float(float(v))) for v in rng.uniform(size=9)]
    fixtures = [
        ("active", np.asarray([[0.0], [0.125], [0.25]]), ["0", "1/2", "1"]),
        ("ordinary", np.asarray([[-1., 0.], [0., 1.], [1., 2.], [2., -1.]]),
         ["0", "1/3", "2/3", "1"]),
        ("constant", np.full((4, 2), 0.25), ["0", "1", "1/3", "2/3"]),
        ("random", random_z, random_y),
    ]
    for name, z, y in fixtures:
        fid = "synthetic42:factorial:" + name
        for lam in LAMBDAS:
            tag = name + "/lambda=" + lam
            base = fit(z, y, feature_id=fid, ridge_lambda=lam)
            all_variants = {}
            for project, recenter, clip in itertools.product((False, True), repeat=3):
                variant = h.make_variant(base, project=project, recenter=recenter, clip=clip)
                COUNTS["factorial_variants"] += 1
                all_variants[project, recenter, clip] = variant
                prepared = h.prepare_prediction(variant, target_min="-5/4", span="3/7")
                vtag = f"{tag}/P{int(project)}R{int(recenter)}C{int(clip)}"
                beta = [F(v) for v in variant["head"]["coefficients"]]
                b = F(variant["head"]["intercept"])
                norm = sum((v * v for v in beta), F(0))
                center = list(map(F, base["center_features"]))
                center_y = F(base["center_target"])
                check(variant["structural_claims"]["norm_within_kappa"] == (norm <= 4)
                      and (not project or norm <= 4), vtag + ": exact norm claim")
                check((not recenter and b == F(base["intercept"]))
                      or (recenter and b + sum((v * x for v, x in zip(beta, center)), F(0)) == center_y),
                      vtag + ": independent intercept switch semantics")
                check(variant["structural_claims"]["normalized_range"] == (["0", "1"] if clip else None),
                      vtag + ": boundedness disclosed only when clipping is enabled")
                for qindex, query in enumerate([np.zeros(z.shape[1]), z[0], z[-1] + 2]):
                    details = h.predict_details(prepared, query)
                    exact = h.exact_details(prepared, query)
                    value = h.predict_float(prepared, query)
                    # Independent evaluation with exact dyadic coordinates.
                    raw = b + sum((v * F.from_float(float(x)) for v, x in zip(beta, query)), F(0))
                    normalized = min(F(1), max(F(0), raw)) if clip else raw
                    physical = F(-5, 4) + F(3, 7) * normalized
                    check(exact["exact_physical"] == str(physical)
                          and h.exact_prediction(prepared, query) == physical
                          and details["reported_physical_hex"] == value.hex(),
                          vtag + f"/q{qindex}: exact reference and operational record agree")
                    check(details["clipping_active"] == (clip and float.fromhex(details["raw_normalized_hex"])
                                                            != float.fromhex(details["normalized_hex"])),
                          vtag + f"/q{qindex}: clipping activation is truthful")
            # Only C changes the range operation, never fitted coefficients/b.
            check(all(all_variants[p, r, False]["head"]["coefficients"]
                      == all_variants[p, r, True]["head"]["coefficients"]
                      and all_variants[p, r, False]["head"]["intercept"]
                      == all_variants[p, r, True]["head"]["intercept"]
                      for p, r in itertools.product((False, True), repeat=2)),
                  tag + ": clipping toggle never refits or recenters")
            if F(base["coefficient_squared_norm"]) <= 4:
                check(all(all_variants[False, r, c]["head_id"] == all_variants[True, r, c]["head_id"]
                          for r, c in itertools.product((False, True), repeat=2)),
                      tag + ": inactive projection leaves evaluated heads identical")
            old = legacy_affine.fit_affine(z, y, feature_id=fid, ridge_lambda=lam)
            COUNTS["synthetic_legacy_solves"] += 1
            anchor = all_variants[True, True, True]
            bridge = h.as_legacy_head(anchor)
            check(bridge == old["head"], tag + ": all-on branch exactly reproduces inherited head")
            old_deploy = legacy_cert.make_deployment(
                bridge, target_min="-5/4", span="3/7", map_sha256="0" * 64,
                map_bytes=1, candidate_id="synthetic42:" + tag)
            old_prepared = legacy_cert.prepare_deployment(old_deploy)
            new_prepared = h.prepare_prediction(anchor, target_min="-5/4", span="3/7")
            for qindex, query in enumerate([np.zeros(z.shape[1]), z[0], z[-1] + 2]):
                value = h.predict_float(new_prepared, query)
                check(value.hex() == legacy_cert.predict_float(old_prepared, query).hex()
                      and h.exact_prediction(new_prepared, query) == legacy_cert.exact_prediction(old_prepared, query),
                      tag + f"/q{qindex}: inherited physical evaluation is byte-identical")
                qid = f"synthetic:{tag}:{qindex}"
                packet = legacy_cert.produce_certificate(old_prepared, query, value, query_id=qid)
                verified = legacy_cert.verify_certificate(old_prepared, query, packet,
                    expected_query_id=qid, reported_prediction=value)
                COUNTS["legacy_bridge_packets"] += 1
                check(verified["status"] == "PASS" and verified["target_label_accessed"] is False,
                      tag + f"/q{qindex}: unchanged certificate checker accepts bridged head")


def boundary_and_rejection_checks(active, legacy_cert):
    clipped = h.make_variant(active, project=True, recenter=True, clip=True)
    free = h.make_variant(active, project=True, recenter=True, clip=False)
    pclip = h.prepare_prediction(clipped, target_min="-5/4", span="3/7")
    pfree = h.prepare_prediction(free, target_min="-5/4", span="3/7")
    for query, expected, side in [([-10.], F(-5, 4), "below"), ([10.], F(-5, 4) + F(3, 7), "above")]:
        check(h.exact_prediction(pclip, query) == expected
              and h.predict_details(pclip, query)["clipping_side"] == side,
              "Clipped extrapolation reaches the exact physical " + side + " endpoint")
        check(h.exact_prediction(pfree, query) != expected
              and not h.predict_details(pfree, query)["clipping_active"],
              "Unclipped extrapolation retains its unbounded " + side + " prediction")
    # Disabled projection is a legal diagnostic head even outside old class;
    # the old-class bridge must reject it rather than fabricating a certificate.
    oversize = h.make_variant(active, project=False, recenter=True, clip=True)
    h.prepare_prediction(oversize, target_min="0", span="1")
    rejects(lambda: h.as_legacy_head(oversize), "Legacy bridge rejects oversize coefficients")
    rejects(lambda: h.as_legacy_head(free), "Legacy bridge rejects removed range clipping")
    for project, recenter in [(True, False), (True, True)]:
        v = h.make_variant(active, project=project, recenter=recenter, clip=True)
        b = h.as_legacy_head(v)
        deployed = legacy_cert.make_deployment(b, target_min="0", span="1",
            map_sha256="a" * 64, map_bytes=1, candidate_id="synthetic42:bridge")
        prepared = legacy_cert.prepare_deployment(deployed)
        query = np.asarray([0.125])
        value = legacy_cert.predict_float(prepared, query)
        packet = legacy_cert.produce_certificate(prepared, query, value, query_id="trusted")
        check(legacy_cert.verify_certificate(prepared, query, packet,
              expected_query_id="trusted", reported_prediction=value)["status"] == "PASS",
              f"Legacy checker accepts a class-preserving P1R{int(recenter)}C1 variant")
        inaccurate = value + 0.001
        self_consistent_bad = legacy_cert.produce_certificate(prepared, query, inaccurate, query_id="trusted")
        rejects(lambda: legacy_cert.verify_certificate(prepared, query, self_consistent_bad,
                expected_query_id="trusted", reported_prediction=inaccurate),
                f"Legacy checker rejects self-consistent over-budget P1R{int(recenter)}C1 packet")
        rejects(lambda: legacy_cert.verify_certificate(prepared, query, packet,
                expected_query_id="other", reported_prediction=value),
                f"Legacy checker requires external query identity for P1R{int(recenter)}C1")
    corrupted = copy.deepcopy(active)
    corrupted["coefficients"][0] = "999"
    rejects(lambda: h.make_variant(corrupted, project=True, recenter=True, clip=True),
            "Mutated base coefficients without a matching identity reject")
    badhead = copy.deepcopy(clipped)
    badhead["head"]["intercept"] = "999"
    rejects(lambda: h.prepare_prediction(badhead, target_min="0", span="1"),
            "Mutated evaluated head without a matching identity rejects")
    false_claim = copy.deepcopy(oversize)
    false_claim["structural_claims"]["norm_within_kappa"] = True
    rejects(lambda: h.prepare_prediction(false_claim, target_min="0", span="1"),
            "False norm-membership claim rejects independently of head identity")
    rejects(lambda: h.make_variant(active, project="False", recenter=True, clip=True),
            "String projection switch cannot silently enable a factorial factor")
    for badlam in ["0", "-1", True]:
        rejects(lambda badlam=badlam: h.fit_base([[0.], [1.]], ["0", "1"],
                feature_id="synthetic42:invalid", ridge_lambda=badlam),
                "Invalid lambda rejects: " + repr(badlam))
    rejects(lambda: h.fit_base([[float("nan")]], ["0"], feature_id="synthetic42:invalid"),
            "Nonfinite training features reject")
    rejects(lambda: h.fit_base([[0.], [1.]], ["0"], feature_id="synthetic42:invalid"),
            "Training label count mismatch rejects")
    rejects(lambda: h.predict_float(pclip, [1., 2.]), "Wrong query dimension rejects")
    rejects(lambda: h.predict_float(pclip, [float("inf")]), "Nonfinite query rejects")
    rejects(lambda: h.prepare_prediction(clipped, target_min="0", span="0"),
            "Degenerate physical normalization rejects")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, affine, cert = load_legacy()
    active = analytic_checks()
    factorial_checks(affine, cert)
    boundary_and_rejection_checks(active, cert)
    example = h.make_variant(active, project=True, recenter=False, clip=False)
    record = {
        "schema": "step42-head-artificial-checks-v1",
        "status": "PASS", "checks_count": len(CHECKS), "checks": CHECKS,
        **COUNTS,
        "public_measurement_values_read": 0,
        "real_model_fits": 0,
        "scientific_files_from_step41_modified": False,
        "step41_bytecode_created": False,
        "source_sha256": hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "legacy_source_attribution_sha256": hashlib.sha256((LEGACY / "attribution42.json").read_bytes()).hexdigest(),
        "example_is_artificial": True,
        "example_base": active,
        "example_variant": example,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(record, handle, sort_keys=True, indent=2)
            handle.write("\n")
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("checks", "example_base", "example_variant")}, sort_keys=True))


if __name__ == "__main__":
    main()
