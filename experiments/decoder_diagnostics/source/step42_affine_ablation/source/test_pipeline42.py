"""Artificial component integration for the frozen Step 42 pipeline.

This is NOT a full provenance fixture, data-access rehearsal, or scientific audit
gate. No real input array, target, model, or result is loaded. It exercises the
pure head/prediction/scoring/pooling/selector/contrast/certificate components on
small generated matrices. Independent audit functions reconstruct their values.
"""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import copy
from fractions import Fraction as F
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import common42 as c
import heads42 as h
import audit42 as independent
import aggregate42 as aggregate
import protocol42

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "source/legacy42"
CHECKS = []
COUNTS = {"synthetic_base_fits": 0, "independent_synthetic_solve_replays": 0,
          "synthetic_legacy_fits": 0, "synthetic_variants": 0,
          "synthetic_query_predictions": 0, "synthetic_certificate_packets": 0}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CHECKS.append(label)


def rejects(call, label, exception=(ValueError, AssertionError)):
    try:
        call()
    except exception:
        CHECKS.append(label)
    else:
        raise AssertionError(label + ": expected rejection")


def load_legacy():
    sys.path.insert(0, str(LEGACY))
    modules = [importlib.import_module(x) for x in ("affine41", "certificate41")]
    for module in modules:
        check(Path(module.__file__).resolve().parent == LEGACY.resolve(), "Legacy component resolves to retained source")
    return modules


def generated_context(stage, stage_index):
    n_train = (6, 8, 10, 12)[stage_index]
    n_evaluation = (2, 3, 5, 4)[stage_index]
    targets = [F(i, n_train - 1) for i in range(n_train)]
    normalized_evaluation = [F(i + 1, n_evaluation + 1) for i in range(n_evaluation)]
    normalized_evaluation[0], normalized_evaluation[-1] = F(-1, 4), F(5, 4)
    normal = {"target_min": str(F(-5, 4) + F(stage_index, 10)),
              "span": str(F(3, 7) + F(stage_index, 8))}
    raw = [F(normal["target_min"]) + F(normal["span"]) * y for y in normalized_evaluation]
    training, evaluation = {}, {}
    for map_index, map_name in enumerate(c.maps()):
        dimension = 1 + map_index % 3
        scale = 0.125 if map_index % 3 == 0 else 1.0 + (map_index % 4) / 2
        offset = 4.0 if map_index % 2 else 0.0

        def feature(t):
            x = (t - (n_train - 1) / 2) / 16
            coordinates = [x, x * x + (t % 3) / 32, ((t * 5 + map_index) % 9 - 4) / 16]
            return [offset + scale * value for value in coordinates[:dimension]]

        training[map_name] = np.asarray([feature(i) for i in range(n_train)], dtype=np.float64)
        query_locations = np.linspace(-16, n_train + 16, n_evaluation)
        evaluation[map_name] = np.asarray([feature(float(i)) for i in query_locations], dtype=np.float64)
    return {"schema": "step42-artificial-component-input-v1", "stage": stage,
            "artificial": True, "normalization": normal,
            "train": {"y": targets, "features": training},
            "evaluation": {"ids": [f"artificial42:{stage}:q{i}" for i in range(n_evaluation)],
                           "y": normalized_evaluation, "raw_targets": raw, "features": evaluation}}


def stage_components(context, old_affine):
    normal = context["normalization"]
    bases, variants, predictions, scores = {}, {}, {}, []
    for cfg in c.configurations():
        name, key, cid = cfg["map"], cfg["base_key"], cfg["candidate_id"]
        Z = context["train"]["features"][name]
        fid = f"artificial42:{context['stage']}:{name}"
        if key not in bases:
            binding = {"artificial": True, "stage": context["stage"], "map_label": name,
                       "actual_feature_factory_used": False}
            base = h.fit_base(Z, context["train"]["y"], feature_id=fid,
                              ridge_lambda=cfg["ridge_lambda"], training_binding=binding)
            COUNTS["synthetic_base_fits"] += 1
            independent.audit_base(base, Z, context["train"]["y"], feature_id=fid,
                                   ridge_lambda=cfg["ridge_lambda"], training_binding=binding)
            COUNTS["independent_synthetic_solve_replays"] += 1
            check(True, "Independent synthetic base reconstruction: " + context["stage"] + "/" + key)
            bases[key] = base
        variant = h.make_variant(bases[key], project=cfg["project"], recenter=cfg["recenter"], clip=cfg["clip"])
        COUNTS["synthetic_variants"] += 1
        exact_head = independent.audit_variant(bases[key], variant, project=cfg["project"],
                                               recenter=cfg["recenter"], clip=cfg["clip"])
        prepared = h.prepare_prediction(variant, target_min=normal["target_min"], span=normal["span"])
        audit_prepared = independent.prepare_exact(exact_head)
        operational, references, clipping, exact_clipping = [], [], [], []
        for row in context["evaluation"]["features"][name]:
            op = h.predict_details(prepared, row)
            ex = h.exact_details(prepared, row)
            checked = independent.prediction_reference(audit_prepared, row, normal["target_min"], normal["span"])
            check(op["reported_physical_hex"] == checked["reported_physical_hex"]
                  and ex["exact_physical"] == checked["exact_physical"]
                  and op["clipping_active"] == checked["operational_clipping_active"]
                  and ex["clipping_active"] == checked["exact_clipping_active"]
                  and op["clipping_side"] == checked["operational_clipping_side"]
                  and ex["clipping_side"] == checked["exact_clipping_side"],
                  "Independent exact/operational prediction and activation: " + context["stage"] + "/" + cid)
            operational.append(op["reported_physical_hex"])
            references.append(ex["exact_physical"])
            clipping.append(op["clipping_active"])
            exact_clipping.append(ex["clipping_active"])
            COUNTS["synthetic_query_predictions"] += 1
        risk = c.score(operational, context["evaluation"]["raw_targets"], hex_predictions=True)
        erisk = c.score(references, context["evaluation"]["raw_targets"])
        check(risk == independent.exact_score(operational, context["evaluation"]["raw_targets"], hex_values=True)
              and erisk == independent.exact_score(references, context["evaluation"]["raw_targets"]),
              "Independent exact losses: " + context["stage"] + "/" + cid)
        predictions[cid] = {"candidate_id": cid, "reported_physical_hex": operational,
                            "exact_physical": references, "clipping_active": clipping,
                            "exact_clipping_active": exact_clipping}
        variants[cid] = variant
        scores.append({**cfg, "physical_risk": risk, "exact_reference_risk": erisk,
                       "projection_active": variant["transformation"]["projection_active"],
                       "clipping_count": sum(clipping)})
        if cfg["lambda_token"] == "1e-6" and cfg["project"] and cfg["recenter"] and cfg["clip"]:
            inherited = old_affine.fit_affine(Z, context["train"]["y"], feature_id=fid)
            COUNTS["synthetic_legacy_fits"] += 1
            check(h.as_legacy_head(variant) == inherited["head"],
                  "Inherited synthetic branch recovery: " + context["stage"] + "/" + name)
    check(len(bases) == 90 and len(variants) == len(predictions) == len(scores) == 720,
          "Complete artificial grid retains every configuration: " + context["stage"])
    return {"schema": "step42-artificial-component-stage-v1", "stage": context["stage"],
            "artificial": True, "full_stage_audit_gate": False, "bases": bases,
            "variants": variants, "predictions": predictions, "scores": scores}


def pooled_components(contexts, stages):
    folded = {name: {x["candidate_id"]: x for x in stages[name]["scores"]} for name in ("f1", "f2", "f3")}
    rows = []
    total_n = sum(len(contexts[name]["evaluation"]["ids"]) for name in folded)
    for cfg in c.configurations():
        cid = cfg["candidate_id"]
        sse = sum((F(folded[name][cid]["physical_risk"]["sse"]) for name in folded), F(0))
        esse = sum((F(folded[name][cid]["exact_reference_risk"]["sse"]) for name in folded), F(0))
        all_op = [value for name in folded for value in stages[name]["predictions"][cid]["reported_physical_hex"]]
        all_exact = [value for name in folded for value in stages[name]["predictions"][cid]["exact_physical"]]
        all_targets = [value for name in folded for value in contexts[name]["evaluation"]["raw_targets"]]
        risk = independent.exact_score(all_op, all_targets, hex_values=True)
        erisk = independent.exact_score(all_exact, all_targets)
        check(F(risk["sse"]) == sse and F(risk["mse"]) == sse / total_n
              and F(erisk["sse"]) == esse and F(erisk["mse"]) == esse / total_n,
              "Pooled losses equal independent concatenated-query scoring: " + cid)
        rows.append({**cfg, "n": total_n, "pooled_physical_mse": str(sse / total_n),
                     "pooled_exact_reference_mse": str(esse / total_n),
                     "physical_risk": risk, "exact_reference_risk": erisk,
                     "projection_active_fold_count": sum(folded[name][cid]["projection_active"] for name in folded),
                     "legacy_f3_deployment_bytes": 1000 + c.maps().index(cfg["map"])})
    specs = c.selector_specs()
    selected = {}
    for spec in specs:
        actual = c.select(rows, spec)
        pool = [x for x in rows if all(x[key] == value for key, value in spec["fixed"].items())]
        expected = independent.choose(pool, legacy=spec["label"] == "original41_replay")
        check(actual["candidate_id"] == expected["candidate_id"] and actual["pool_size"] == spec["pool_size"],
              "Independent selector on artificial pooled data: " + spec["label"])
        selected[spec["label"]] = actual
    tied = [{**row, "pooled_physical_mse": "1", "legacy_f3_deployment_bytes":
             (1 if row["map"] in c.maps()[-2:] else 100)} for row in reversed(rows)]
    for spec in specs:
        pool = [x for x in tied if all(x[key] == value for key, value in spec["fixed"].items())]
        expected = independent.choose(pool, legacy=spec["label"] == "original41_replay")
        check(c.select(tied, spec)["candidate_id"] == expected["candidate_id"],
              "Order-independent exact tie handling: " + spec["label"])
    expected = c.select(tied, specs[-1])["candidate_id"]
    check(expected.startswith(c.maps()[-2] + "::"), "Legacy ties honor bytes before map lexical order")
    damaged = rows[1:]
    rejects(lambda: c.select(damaged, specs[0]), "Selector rejects a missing member of its declared pool")
    # Fold means would prefer A, whereas pooled losses correctly prefer B.
    fold_a = [c.score(["0"], ["0"]), c.score(["2"] * 3, ["0"] * 3)]
    fold_b = [c.score(["3/2"], ["0"]), c.score(["3/2"] * 3, ["0"] * 3)]
    pooled_a = sum((F(x["sse"]) for x in fold_a), F(0)) / 4
    pooled_b = sum((F(x["sse"]) for x in fold_b), F(0)) / 4
    mean_a = sum((F(x["mse"]) for x in fold_a), F(0)) / 2
    mean_b = sum((F(x["mse"]) for x in fold_b), F(0)) / 2
    check(pooled_b < pooled_a and mean_a < mean_b and (pooled_a, pooled_b) == (F(3), F(9, 4)),
          "Unequal-fold sentinel distinguishes pooled observations from averaged fold MSE")
    return rows, selected


def contrast_components(contexts, stages, pooled):
    for name, rows in (("development", pooled), ("assessment", stages["final"]["scores"])):
        result = aggregate.contrast_context(rows, "2")
        check(all(len(x["pairs"]) == 360 for x in result["factor_contrasts"].values())
              and len(result["projection_recenter_interactions"]["rows"]) == 180
              and len(result["regularization_contrasts"]["rows"]) == 576,
              "Complete contrasts on actual artificial-model scores: " + name)
    analytic = []
    for cfg in c.configurations():
        p, r, clip = int(cfg["project"]), int(cfg["recenter"]), int(cfg["clip"])
        lam_index = [x[0] for x in c.LAMBDA_GRID].index(cfg["lambda_token"])
        value = F(20) + F(c.maps().index(cfg["map"]), 10) + F(lam_index, 5) + 3 * p - 2 * r + 5 * clip - 4 * p * r
        analytic.append({**cfg, "physical_risk": {"mse": str(value)},
                         "exact_reference_risk": {"mse": str(value)},
                         "projection_active": cfg["project"] and c.maps().index(cfg["map"]) % 2 == 0})
    result = aggregate.contrast_context(analytic, "10")
    factors = result["factor_contrasts"]
    check(factors["project"]["summary"]["exact_improved"] == 180
          and factors["project"]["summary"]["exact_worsened"] == 180
          and factors["project"]["summary"]["mean_mse_difference"] == "1",
          "Analytic projection contrasts recover sign counts and mean")
    check(factors["recenter"]["summary"]["exact_improved"] == 360
          and factors["recenter"]["summary"]["mean_mse_difference"] == "-4"
          and factors["clip"]["summary"]["exact_worsened"] == 360
          and factors["clip"]["summary"]["mean_mse_difference"] == "5",
          "Analytic recentering and clipping effects have correct directions")
    check(all(row["mse_difference"] == "-4" for row in result["projection_recenter_interactions"]["rows"]),
          "Projection/recentering interaction has the declared difference-of-differences sign")
    check(all(F(row["mse_difference"]) == F([x[0] for x in c.LAMBDA_GRID].index(row["lambda_token"]), 5)
              for row in result["regularization_contrasts"]["rows"]),
          "Regularization contrasts use the fixed legacy lambda reference")
    reference, epsilon = F(7), F(1, 10**10)
    boundary = aggregate.effect_summary([{"mse_difference": str(x * reference)} for x in (-epsilon, epsilon, 2 * epsilon)], reference)
    check(boundary["within_display_threshold"] == 2 and boundary["beyond_display_threshold_worsened"] == 1,
          "Descriptive near-zero threshold includes its exact boundary")
    check(aggregate.median([F(1, 3), F(1, 2)]) == F(5, 12), "Even-sized exact median remains rational")
    # Explicit paired counterexample: clipping cannot hurt an in-range label,
    # but can hurt a label outside the training target range.
    toy_context = {"synthetic": {"normalization": {"target_min": "0", "span": "1"},
                                  "evaluation": {"raw_targets": list(map(F, ["-2", "3", "1/2", "1/2"]))}}}
    toy_predictions = {"synthetic": {cfg["candidate_id"]: {"exact_physical":
                        ["0", "1", "0", "1"] if cfg["clip"] else ["-2", "3", "-2", "3"]}
                        for cfg in c.configurations()}}
    clipped = aggregate.clipping_range_diagnostic(toy_context, toy_predictions, c.configurations())
    check(all(x["inside_n"] == x["outside_n"] == 2
              and x["inside_exact_sse_difference"] == "-12"
              and x["outside_exact_sse_difference"] == "8" for x in clipped),
          "Clipping diagnostic separates in-range nonincrease from possible out-of-range harm")
    broken = copy.deepcopy(toy_predictions)
    first_on = next(x for x in c.configurations() if x["clip"])["candidate_id"]
    broken["synthetic"][first_on]["exact_physical"][2] = "4"
    rejects(lambda: aggregate.clipping_range_diagnostic(toy_context, broken, c.configurations()),
            "Clipping diagnostic rejects a false in-range nonincrease assertion", exception=RuntimeError)
    return {"analytic_contrasts": result, "clipping_counterexample": clipped[0], "threshold_boundary": boundary}


def bridge_components(context, final, selectors, certificate):
    labels = protocol42.protocol()["assurance"]["certificate_bridge_selectors"]
    aliases = {label: selectors[label]["candidate_id"] for label in labels}
    records = {}
    normal = context["normalization"]
    for cid in sorted(set(aliases.values())):
        variant = final["variants"][cid]
        name = cid.split("::")[0]
        map_hash = hashlib.sha256(("artificial:" + name).encode()).hexdigest()
        legacy_head = h.as_legacy_head(variant)
        deployment = certificate.make_deployment(legacy_head, target_min=normal["target_min"], span=normal["span"],
                                                 map_sha256=map_hash, map_bytes=1, candidate_id=cid)
        prepared = certificate.prepare_deployment(deployment)
        rows = []
        for i, row in enumerate(context["evaluation"]["features"][name]):
            query = context["evaluation"]["ids"][i]
            value = certificate.predict_float(prepared, row)
            expected = final["predictions"][cid]["reported_physical_hex"][i]
            check(value.hex() == expected, "Selected certificate bridge preserves operational prediction")
            packet = certificate.produce_certificate(prepared, row, value, query_id=query)
            accepted = certificate.verify_certificate(prepared, row, packet, expected_query_id=query, reported_prediction=value)
            audit = independent.audit_legacy_packet(variant, deployment, packet, row, query_id=query,
                reported_hex=expected, target_min=normal["target_min"], span=normal["span"],
                map_sha256=map_hash, map_bytes=1, candidate_id=cid)
            check(accepted["status"] == "PASS" and not accepted["target_label_accessed"]
                  and F(audit["absolute_error"]) <= F(audit["budget"]), "Selected packet passes unchanged and independent checkers")
            rows.append({"query_id": query, "reported_physical_hex": expected, "certificate": packet,
                         "verification": accepted, "independent_check": audit})
            COUNTS["synthetic_certificate_packets"] += 1
        records[cid] = {"deployment": deployment, "rows": rows}
    # Self-consistent incorrect output must be rejected specifically for budget,
    # while a wrong query identity must produce a different rejection category.
    cid = next(iter(records))
    record = records[cid]
    prepared = certificate.prepare_deployment(record["deployment"])
    name = cid.split("::")[0]
    row = context["evaluation"]["features"][name][0]
    query = context["evaluation"]["ids"][0]
    value = certificate.predict_float(prepared, row) + 0.01
    packet = certificate.produce_certificate(prepared, row, value, query_id=query)
    try:
        certificate.verify_certificate(prepared, row, packet, expected_query_id=query, reported_prediction=value)
    except certificate.CertificateError as exc:
        check(str(exc) == "Reported prediction exceeds the frozen discrepancy budget", "Correctly classified self-consistent over-budget rejection")
    else:
        raise AssertionError("Self-consistent inaccurate output was accepted")
    try:
        certificate.verify_certificate(prepared, row, packet, expected_query_id="wrong", reported_prediction=value)
    except certificate.CertificateError as exc:
        check(str(exc) != "Reported prediction exceeds the frozen discrepancy budget", "Unrelated packet error is distinguishable from over-budget rejection")
    else:
        raise AssertionError("Wrong external query identity was accepted")
    return {"aliases": aliases, "records": records, "artificial": True}


def serializable_context(context):
    return {"schema": context["schema"], "stage": context["stage"], "artificial": True,
            "normalization": context["normalization"],
            "train": {"y": [str(x) for x in context["train"]["y"]],
                      "features_binary64_hex": {k: [[float(x).hex() for x in row] for row in z]
                                                 for k, z in context["train"]["features"].items()}},
            "evaluation": {"ids": context["evaluation"]["ids"],
                           "raw_targets": [str(x) for x in context["evaluation"]["raw_targets"]],
                           "features_binary64_hex": {k: [[float(x).hex() for x in row] for row in z]
                                                      for k, z in context["evaluation"]["features"].items()}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/pipeline_artificial_checks42.json")
    parser.add_argument("--fixture", type=Path, default=ROOT / "analysis/component_fixture42")
    args = parser.parse_args()
    start = time.perf_counter()
    if args.output.exists() or args.fixture.exists():
        raise FileExistsError("Retain the existing component fixture and receipt")
    modules = (c, h, independent, aggregate, protocol42)
    before = {str(Path(x.__file__).relative_to(ROOT)): c.sha(x.__file__) for x in modules}
    old_affine, certificate = load_legacy()
    check(c.configurations() == independent.expected_configurations(), "Producer and independent complete factorial definitions agree")
    check(c.selector_specs() == independent.expected_specs() and len(c.selector_specs()) == 177,
          "Producer and independent 177 selector definitions agree")
    contexts = {stage: generated_context(stage, i) for i, stage in enumerate(("f1", "f2", "f3", "final"))}
    stages = {stage: stage_components(context, old_affine) for stage, context in contexts.items()}
    pooled, selectors = pooled_components(contexts, stages)
    contrast_checks = contrast_components(contexts, stages, pooled)
    bridge = bridge_components(contexts["final"], stages["final"], selectors, certificate)
    for path, expected in before.items():
        check(c.sha(ROOT / path) == expected, "Component source remained unchanged during this test: " + path)
    args.fixture.mkdir(parents=True)
    files = {}

    def retained(name, data):
        path = args.fixture / name
        raw = c.canonical_bytes(data)
        with path.open("xb") as stream:
            stream.write(raw)
        files[str(path.relative_to(ROOT))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    for stage in contexts:
        retained(stage + "_inputs.json", serializable_context(contexts[stage]))
        retained(stage + "_components.json", stages[stage])
    retained("pooled_and_selected.json", {"rows": pooled, "selectors": selectors, "artificial": True})
    retained("contrast_checks.json", contrast_checks)
    retained("certificate_bridge.json", bridge)
    record = {"schema": "step42-artificial-component-integration-v1", "status": "PASS",
              "scope": "Pure component integration, not a full real-provenance fixture, stage gate, controller rehearsal, or scientific evaluation",
              "public_measurements_read": 0, "real_model_fits": 0, "native_feature_factory_used": False,
              "full_provenance_or_stage_gates_fabricated": False, "checks_count": len(CHECKS),
              "checks": CHECKS, "counts": COUNTS, "selectors_exercised": len(selectors),
              "source_bindings": before, "test_source_sha256": c.sha(Path(__file__)),
              "fixture_files": files, "fixture_file_count": len(files),
              "scientific_outputs_written": False, "prior_step41_modified": False,
              "wall_seconds": time.perf_counter() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(c.canonical_bytes(record))
    print(json.dumps({k: v for k, v in record.items() if k not in ("checks", "fixture_files", "source_bindings")}, sort_keys=True))


if __name__ == "__main__":
    main()
