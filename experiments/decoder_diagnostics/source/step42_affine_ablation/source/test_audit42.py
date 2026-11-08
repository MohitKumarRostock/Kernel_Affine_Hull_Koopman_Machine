"""Artificial and adversarial checks of the independent Step 42 auditor."""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import sys

import audit42 as audit
import heads42 as producer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy42"))
import affine41 as legacy_fit
import certificate41 as legacy_certificate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=audit.ROOT/"analysis/artificial_audit_checks42.json")
    args = parser.parse_args()
    checks = []

    def passed(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejected(call, name):
        try:
            call()
        except audit.AuditError:
            checks.append(name)
        else:
            raise AssertionError("Auditor accepted: " + name)

    fixtures = [
        ("active_projection", np.array([[0.001, 0.10], [0.002, 0.11], [0.003, 0.12],
                                        [0.004, 0.13], [0.005, 0.14], [0.006, 0.15]]),
         [F(0), F(1, 10), F(2, 5), F(4, 5), F(9, 10), F(1)]),
        ("constant_rational_target", np.array([[-2., .5], [-1., .25], [0., .125],
                                               [1., .0625], [2., .03125]]), [F(1, 3)]*5),
        ("collinear_features", np.array([[-2., -4., 1.], [-1., -2., 1.],
                                         [0., 0., 1.], [1., 2., 1.], [2., 4., 1.]]),
         [F(0), F(1, 4), F(2, 5), F(3, 4), F(1)]),
        ("nonzero_center", np.array([[1000.1, -7.3], [1000.2, -7.1], [1000.3, -6.9],
                                     [1000.4, -6.7], [1000.5, -6.5]]),
         [F(1, 7), F(2, 7), F(4, 7), F(5, 7), F(6, 7)]),
    ]
    retained = []
    active = 0
    query_count = 0
    for label, z, y in fixtures:
        for lam in audit.LAMBDAS:
            binding = {"synthetic_fixture": label, "training_only": True}
            base = producer.fit_base(z, y, feature_id="synthetic:" + label,
                                     ridge_lambda=lam, training_binding=binding)
            audit.audit_base(base, z, y, feature_id="synthetic:" + label,
                             ridge_lambda=lam, training_binding=binding)
            checks.append("independent base solve and centers: " + label + "/" + lam)
            for project, recenter, clip in product((False, True), repeat=3):
                variant = producer.make_variant(base, project=project, recenter=recenter, clip=clip)
                head = audit.audit_variant(base, variant, project=project, recenter=recenter, clip=clip)
                checks.append(f"independent rational factorial: {label}/{lam}/{project}/{recenter}/{clip}")
                active += int(variant["transformation"]["projection_active"])
                prepared = producer.prepare_prediction(variant, target_min="-37/10", span="49/2")
                independently_prepared = audit.prepare_exact(head)
                queries = [z[0], z[-1], -100*z[0], 100*z[-1]]
                for row in queries:
                    expected = audit.prediction_reference(independently_prepared, row, "-37/10", "49/2")
                    reported = producer.predict_float(prepared, row)
                    exact = producer.exact_prediction(prepared, row)
                    passed(reported.hex() == expected["reported_physical_hex"] and
                           str(exact) == expected["exact_physical"],
                           f"integer reference and single-dot replay: {label}/{lam}/{project}/{recenter}/{clip}/{query_count}")
                    operational = producer.predict_details(prepared, row)
                    reference = producer.exact_details(prepared, row)
                    passed(operational["clipping_active"] == expected["operational_clipping_active"] and
                           operational["clipping_side"] == expected["operational_clipping_side"] and
                           reference["clipping_active"] == expected["exact_clipping_active"] and
                           reference["clipping_side"] == expected["exact_clipping_side"],
                           f"independent operational and exact clipping labels: {query_count}")
                    query_count += 1
                if lam == "1/1000000" and project and recenter and clip:
                    inherited = legacy_fit.fit_affine(z, y, feature_id=base["feature_id"],
                                                      ridge_lambda=lam, training_binding=binding)
                    passed(head["coefficients"] == inherited["head"]["coefficients"] and
                           head["intercept"] == inherited["head"]["intercept"],
                           "legacy exact head recovery: " + label)
                retained.append((base, variant, z, y, binding, project, recenter, clip))
    passed(active > 0, "Synthetic suite exercises actual norm contraction")

    # A scalar analytic example distinguishes n*lambda from an unnormalized
    # penalty or a penalized intercept without invoking the producer as oracle.
    simple = producer.fit_base(np.array([[-1.], [1.]]), [F(0), F(1)],
                               feature_id="synthetic:analytic", ridge_lambda="1")
    passed(simple["coefficients"] == ["1/4"] and simple["intercept"] == "1/2",
           "Analytic centered ridge checks n*lambda and unpenalized intercept")

    base, variant, z, y, binding, p, r, c = next(x for x in retained if x[5:] == (True, True, True))

    def rebased(mutant):
        mutant["base_id"] = "base42:" + audit.digest({k: v for k, v in mutant.items() if k != "base_id"})
        return mutant

    mutant = copy.deepcopy(base)
    mutant["coefficients"][0] = str(F(mutant["coefficients"][0]) + F(1, 1024))
    mutant["coefficient_squared_norm"] = str(sum((F(v)**2 for v in mutant["coefficients"]), F(0)))
    rebased(mutant)
    rejected(lambda: audit.audit_base(mutant, z, y, feature_id=base["feature_id"],
                                      ridge_lambda=base["ridge_lambda"], training_binding=binding),
             "Self-consistent changed ridge coefficient is rejected by fit replay")
    for key, value, name in [
        ("penalty_multiplier", base["ridge_lambda"], "Missing n in penalty is rejected"),
        ("center_target", str(F(base["center_target"]) + F(1, 100)), "Evaluation-contaminated target center is rejected"),
        ("training_optimality_claim", True, "Unsupported exact optimizer claim is rejected"),
    ]:
        altered = copy.deepcopy(base)
        altered[key] = value
        rebased(altered)
        rejected(lambda altered=altered: audit.audit_base(altered, z, y,
                    feature_id=base["feature_id"], ridge_lambda=base["ridge_lambda"], training_binding=binding), name)

    for field, value, name in [
        ("projection_factor", "1", "Incorrect projection factor is rejected"),
        ("recenter_delta", "0", "Incorrect recentering amount is rejected"),
    ]:
        altered = copy.deepcopy(variant)
        altered["transformation"][field] = value
        rejected(lambda altered=altered: audit.audit_variant(base, altered, project=p, recenter=r, clip=c), name)
    altered = copy.deepcopy(variant)
    altered["head"]["intercept"] = str(F(altered["head"]["intercept"]) + 1)
    altered["head_id"] = "head42:" + audit.digest(altered["head"])
    rejected(lambda: audit.audit_variant(base, altered, project=p, recenter=r, clip=c),
             "Self-consistent changed head intercept is rejected")
    altered = copy.deepcopy(variant)
    altered["structural_claims"]["population_risk_claim"] = True
    rejected(lambda: audit.audit_variant(base, altered, project=p, recenter=r, clip=c),
             "Population-risk overclaim is rejected")
    unclipped = producer.make_variant(base, project=True, recenter=True, clip=False)
    altered = copy.deepcopy(unclipped)
    altered["structural_claims"]["normalized_range"] = ["0", "1"]
    rejected(lambda: audit.audit_variant(base, altered, project=True, recenter=True, clip=False),
             "Unclipped variant cannot retain the old bounded-range claim")

    # Check packet interoperability only for actual admissible C=1 heads.
    valid_variant = variant
    old_head = producer.as_legacy_head(valid_variant)
    deployment = legacy_certificate.make_deployment(old_head, target_min="-37/10", span="49/2",
                    map_sha256="a"*64, map_bytes=123, candidate_id="synthetic:certified")
    pp = legacy_certificate.prepare_deployment(deployment)
    for index, row in enumerate([z[0], z[-1], -100*z[0], 100*z[-1]]):
        value = legacy_certificate.predict_float(pp, row)
        packet = legacy_certificate.produce_certificate(pp, row, value, query_id="q" + str(index))
        kwargs = dict(query_id="q" + str(index), reported_hex=value.hex(), target_min="-37/10", span="49/2",
                      map_sha256="a"*64, map_bytes=123, candidate_id="synthetic:certified")
        audit.audit_legacy_packet(valid_variant, deployment, packet, row, **kwargs)
        checks.append("Independent old certificate interoperability: q" + str(index))
        corrupt = copy.deepcopy(packet)
        corrupt["query_id"] = "wrong"
        rejected(lambda corrupt=corrupt, row=row, kwargs=kwargs:
                 audit.audit_legacy_packet(valid_variant, deployment, corrupt, row, **kwargs),
                 "Certificate binds expected query: q" + str(index))
        corrupt = copy.deepcopy(packet)
        corrupt["reference_interval_hex"][0] = float(-1e10).hex()
        rejected(lambda corrupt=corrupt, row=row, kwargs=kwargs:
                 audit.audit_legacy_packet(valid_variant, deployment, corrupt, row, **kwargs),
                 "Nontight reference interval is rejected: q" + str(index))

    # A valid norm-bounded clipped head can still evaluate poorly in floating
    # arithmetic; a self-consistent error packet must not bypass its budget.
    cancellation_base = copy.deepcopy(simple)
    cancellation_base.update({"feature_id": "synthetic:cancellation", "dimension": 3,
                              "coefficients": ["1", "1", "-1"], "intercept": "0",
                              "coefficient_squared_norm": "3", "center_features": ["0", "0", "0"],
                              "center_target": "0"})
    rebased(cancellation_base)
    cancellation = producer.make_variant(cancellation_base, project=True, recenter=False, clip=True)
    ch = producer.as_legacy_head(cancellation)
    cd = legacy_certificate.make_deployment(ch, target_min="0", span="1", map_sha256="b"*64,
                                           map_bytes=10, candidate_id="synthetic:cancellation")
    cp = legacy_certificate.prepare_deployment(cd)
    query = np.array([2.0**54, 1., 2.0**54])
    value = legacy_certificate.predict_float(cp, query)
    packet = legacy_certificate.produce_certificate(cp, query, value, query_id="cancel")
    passed(abs(F.from_float(value) - legacy_certificate.exact_prediction(cp, query)) > F(1, 2**40),
           "Cancellation fixture genuinely exceeds the certificate budget")
    rejected(lambda: audit.audit_legacy_packet(cancellation, cd, packet, query, query_id="cancel",
             reported_hex=value.hex(), target_min="0", span="1", map_sha256="b"*64,
             map_bytes=10, candidate_id="synthetic:cancellation"),
             "Self-consistent high-error packet is rejected by the budget")

    passed(audit.exact_score(["0x1.0000000000000p+0", "0x0.0p+0"], ["1/3", "2/3"], hex_values=True)
           ["mse"] == "4/9", "Physical squared losses use exact binary64 and rational labels")
    exact_rows = [{"candidate_id": "a", "map": "a", "pooled_physical_mse": str(1 + F(1, 2**60)),
                   "legacy_f3_deployment_bytes": 1},
                  {"candidate_id": "z", "map": "z", "pooled_physical_mse": "1",
                   "legacy_f3_deployment_bytes": 9}]
    passed(audit.choose(exact_rows)["candidate_id"] == "z", "Selection does not round exact risk ties to binary64")
    tied = [{"candidate_id": "a", "map": "a", "pooled_physical_mse": "1",
             "legacy_f3_deployment_bytes": 10},
            {"candidate_id": "z", "map": "z", "pooled_physical_mse": "1",
             "legacy_f3_deployment_bytes": 2}]
    passed(audit.choose(tied)["candidate_id"] == "a" and audit.choose(tied, legacy=True)["candidate_id"] == "z",
           "New lexical and inherited payload-size tie rules are distinguished")
    rejected(lambda: audit.unique_rows([{"candidate_id": "x"}, {"candidate_id": "x"}]),
             "Duplicate candidates cannot silently replace rows")
    rejected(lambda: audit.exact_score(["nan"], ["0"], hex_values=True),
             "Nonfinite reported predictions are rejected")
    passed(len(audit.expected_configurations()) == 720 and len(audit.expected_specs()) == 177,
           "Independent Cartesian and selector grid counts")
    for truth in (F(0), F(1, 3), F(1)):
        for value in (F(-3), F(-1, 3), F(1, 2), F(4, 3), F(3)):
            clipped = max(F(0), min(F(1), value))
            passed((clipped-truth)**2 <= (value-truth)**2, "Clipping cannot harm exact squared loss inside target range")
    passed((F(1)-F(3))**2 > (F(2)-F(3))**2, "Clipping can harm outside-range targets")
    record = {"schema": "step42-artificial-independent-auditor-checks-v1", "status": "PASS",
              "checks_count": len(checks), "checks": checks,
              "synthetic_base_fits": len(fixtures)*len(audit.LAMBDAS),
              "synthetic_variants": len(retained), "synthetic_prediction_comparisons": query_count,
              "real_step42_models_fitted": 0, "new_source_values_read": 0,
              "auditor_sha256": audit.sha(Path(audit.__file__)), "producer_sha256": audit.sha(Path(producer.__file__)),
              "test_source_sha256": audit.sha(Path(__file__))}
    audit.save(args.output, record)
    print(json.dumps({"status": "PASS", "checks": len(checks), "receipt": str(args.output)}))


if __name__ == "__main__":
    main()
