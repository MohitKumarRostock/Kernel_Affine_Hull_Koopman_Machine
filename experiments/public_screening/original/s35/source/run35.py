"""The one frozen Step35 public forecasting run, with test parsing last.

All real-data model choices are mechanically determined by protocol35.json.
The independent audit, rather than this producer, validates final witnesses.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import pickle
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import finite_feature35 as ff
import representations35 as rep
from data35 import ROOT, canonical_bytes, event, load_development, load_test, sha_file, verify_protocol_seal, write_json

START = None
DEADLINE = None
OUT = ROOT / "output"


def rel(path):
    return str(Path(path).relative_to(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(stage, **fields):
    print(json.dumps({"stage": stage, "elapsed_seconds": round(time.perf_counter()-START, 3), **fields}), flush=True)


def _timeout(signum, frame):
    raise TimeoutError("Frozen Step35 wall-clock budget expired")


@contextmanager
def phase_budget(seconds):
    remaining = DEADLINE-time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("Overall deadline passed")
    signal.setitimer(signal.ITIMER_REAL, min(seconds, remaining))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, max(0.001, DEADLINE-time.perf_counter()))


def save_cohort(cohort):
    path = OUT / "cohorts" / (cohort["name"] + ".json")
    numeric = OUT / "cohorts" / (cohort["name"] + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(numeric, cohort["X"], allow_pickle=False)
    value = {key: value for key, value in cohort.items() if key != "X"}
    value["y"] = [str(x) for x in value["y"]]
    value["persistence"] = [str(x) for x in value["persistence"]]
    value["lag_matrix_path"] = rel(numeric)
    value["lag_matrix_sha256"] = sha_file(numeric)
    write_json(path, value)
    return rel(path)


def save_features(name, role, values):
    path = OUT / "features" / role / (name + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    return rel(path)


def class_lower(name, feature_id, role, z, cohort, protocol):
    started = time.perf_counter()
    folder = OUT / "classes" / role / name
    folder.mkdir(parents=True, exist_ok=True)
    cfg = protocol["certificate"]
    block_paths, summaries, lower_sse = [], [], F(0)
    for index, first in enumerate(range(0, len(z), cfg["block_size"])):
        last = min(len(z), first+cfg["block_size"])
        certificate = ff.class_risk_certificate(z[first:last].tolist(), cohort["y"][first:last],
            feature_id=feature_id, ids=cohort["ids"][first:last], kappa=protocol["class"]["kappa"],
            output_interval=protocol["class"]["prediction_interval"], distance_bits=cfg["distance_bits"],
            max_updates=cfg["class_max_coordinate_attempts_per_block"], check_every=cfg["check_every"],
            gap_sse=cfg["gap_sse"], grid_bits=cfg["multiplier_grid_bits"])
        path = folder / ("block_%03d.json" % index)
        write_json(path, certificate)
        block_paths.append(rel(path))
        lower_sse += F(certificate["dual"]["sse"])
        summaries.append({"index": index, "n": certificate["n"], "status": certificate["status"],
                          "gap_sse": certificate["optimization"]["gap_sse"], "path": rel(path),
                          "certificate_sha256": sha_file(path), "work": {k: v for k, v in certificate["work"].items() if k not in ["checkpoints", "initialization_edges"]}})
    range_floor = sum((max(F(0), -y, y-1)**2 for y in cohort["y"]), F(0))/len(z)
    dual_mean = lower_sse/len(z)
    lower = max(range_floor, dual_mean)
    result = {"schema": "step35-whole-cohort-class-lower-v1", "map": name, "role": role,
              "feature_id": feature_id, "class_id": ff.class_identity(feature_id, protocol["class"]["kappa"], protocol["class"]["prediction_interval"]),
              "n": len(z), "lower": str(lower), "dual_lower": str(dual_mean), "range_only_floor": str(range_floor),
              "combination": "maximum, not addition", "blocks": block_paths, "block_summaries": summaries,
              "block_status_counts": dict(Counter(x["status"] for x in summaries)),
              "stage_seconds_including_witness_retention": time.perf_counter()-started,
              "assurance": ff.ASSURANCE, "population_risk_claim": False}
    path = folder / "lower.json"
    write_json(path, result)
    result["receipt_path"] = rel(path)
    return result


def fit_and_decide(name, representation, feature_id, recipe, train_z, decision_z, train, decision,
                   anchors, map_info, protocol, shadow):
    started = time.perf_counter()
    candidate_id = name + "::" + recipe
    folder = OUT / "heads" / (name + "__" + recipe)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = protocol["certificate"]
    fit_started = time.perf_counter()
    if recipe == "affine":
        y_float = np.asarray([float(y) for y in train["y"]], dtype=np.float64)
        xmean = train_z.mean(axis=0)
        ymean = y_float.mean()
        xc, yc = train_z-xmean, y_float-ymean
        lam = float(F(protocol["heads"]["affine"]["ridge_lambda"]))
        gram = xc.T @ xc + (len(xc)*lam)*np.eye(xc.shape[1])
        beta = np.linalg.solve(gram, xc.T @ yc)
        intercept = float(ymean-xmean @ beta)
        exact_center = [sum((F.from_float(float(v)) for v in train_z[:, column]), F(0))/len(train_z) for column in range(train_z.shape[1])]
        exact_target_center = sum(train["y"], F(0))/len(train["y"])
        fit = ff.make_affine_head(beta.tolist(), intercept, feature_id=feature_id,
            kappa=protocol["class"]["kappa"], output_interval=protocol["class"]["prediction_interval"],
            project=True, center_features=exact_center, center_target=exact_target_center, bits=cfg["distance_bits"])
        fit["training_binding"] = {"training_cohort_path": "output/cohorts/train.json", "n": len(train["y"]),
                                   "ridge_lambda": protocol["heads"]["affine"]["ridge_lambda"],
                                   "training_feature_path": "output/features/train/" + name + ".npy"}
    elif recipe == "extension32":
        fit = ff.fit_anchor_decoder(train_z[anchors].tolist(), [train["y"][i] for i in anchors],
            feature_id=feature_id, ids=[train["ids"][i] for i in anchors], kappa=protocol["class"]["kappa"],
            output_interval=protocol["class"]["prediction_interval"], distance_bits=cfg["distance_bits"],
            max_updates=cfg["head_max_coordinate_attempts"], check_every=cfg["check_every"],
            gap_sse=cfg["gap_sse"], grid_bits=cfg["multiplier_grid_bits"])
        fit["training_binding"] = {"training_cohort_path": "output/cohorts/train.json",
                                   "anchor_indices": [int(i) for i in anchors], "training_feature_path": "output/features/train/" + name + ".npy"}
    else:
        raise ValueError("Unfrozen head recipe")
    fit_seconds = time.perf_counter()-fit_started
    head_path, fit_path = folder / "head.json", folder / "fit.json"
    write_json(head_path, fit["head"])
    write_json(fit_path, fit)
    evaluation_started = time.perf_counter()
    evaluator = ff.evaluate_affine if recipe == "affine" else ff.evaluate_extension
    prediction = evaluator(fit["head"], decision_z.tolist(), decision["y"], ids=decision["ids"])
    prediction_path = folder / "decision.json"
    write_json(prediction_path, prediction)
    evaluation_seconds = time.perf_counter()-evaluation_started
    deployed = map_info["payload_bytes"] + head_path.stat().st_size
    row = {"candidate_id": candidate_id, "map": name, "recipe": recipe, "family": representation.family,
           "scaling": representation.scaling, "feature_width": representation.n_features_out_,
           "feature_id": feature_id, "head_id": fit["head_id"], "class_id": fit["class_id"],
           "map_path": map_info["path"], "map_payload_bytes": map_info["payload_bytes"],
           "map_numeric_bytes": map_info["numeric_bytes"], "head_path": rel(head_path),
           "head_payload_bytes": head_path.stat().st_size, "deployment_payload_bytes": deployed,
           "fit_path": rel(fit_path), "decision_prediction_path": rel(prediction_path),
           "decision_risk": {k: prediction["loss"][k] for k in ["lower", "upper", "width"]},
           "fit_seconds": fit_seconds, "decision_evaluation_seconds_including_retention": evaluation_seconds,
           "decoder_stage_seconds_including_retention": time.perf_counter()-started,
           "shadow_excluded_class": shadow, "fit_status": fit.get("status", fit["membership"]["status"]),
           "head_projection_applied": fit.get("projection", {}).get("projected", False),
           "assurance": ff.ASSURANCE}
    progress("decoder_complete", candidate_id=candidate_id, shadow=shadow,
             decision_mse=float(F(row["decision_risk"]["upper"])), payload_bytes=deployed)
    return row


def threshold_select(rows, epsilon):
    eligible = [row for row in rows if F(row["decision_risk"]["upper"]) <= epsilon]
    selected = min(eligible, key=lambda row: (row["deployment_payload_bytes"], F(row["decision_risk"]["upper"]), row["candidate_id"]), default=None)
    return {"candidate_id": None if selected is None else selected["candidate_id"],
            "eligible_candidate_count": len(eligible), "abstained": selected is None}


def action(selection, protocol):
    selected = selection["candidate_id"]
    if selected is None:
        return "ABSTAIN"
    incumbent = protocol["heads"]["incumbent"]
    if selected == incumbent:
        return "KEEP"
    if selected.split("::")[0] == incumbent.split("::")[0]:
        return "REPAIR_DECODER"
    return "CHANGE_REPRESENTATION"


def actual_aggregate(name, role, lower_record, rows):
    certificates = [read_json(ROOT / path) for path in lower_record["blocks"]]
    predictions = [read_json(ROOT / row[role + "_prediction_path"]) for row in rows if row["map"] == name]
    aggregate = ff.aggregate_class_risk(certificates, predictions, include_diagnostic_constant=False)
    if F(aggregate["class_risk"]["lower"]) != F(lower_record["lower"]):
        raise AssertionError("Range floor/dual aggregation identity changed")
    path = OUT / "classes" / role / name / "aggregate.json"
    write_json(path, aggregate)
    return {"map": name, "role": role, "class_id": aggregate["class_id"],
            "class_risk": aggregate["class_risk"], "range_only_floor": lower_record["range_only_floor"],
            "aggregate_path": rel(path), "lower_path": lower_record["receipt_path"],
            "block_status_counts": lower_record["block_status_counts"],
            "screen_stage_seconds": lower_record["stage_seconds_including_witness_retention"]}


def empirical_receipt(name, predictions, cohort):
    # Float predictions are treated as stored binary64 values, never as ideal-real fits.
    values = [F.from_float(float(x)) if not isinstance(x, F) else x for x in predictions]
    intervals = [{"lower": str(x), "upper": str(x), "width": "0"} for x in values]
    return {"schema": "step35-empirical-baseline-predictions-v1", "baseline": name,
            "query_ids": cohort["ids"], "predictions_exact": [str(x) for x in values],
            "targets_exact": [str(y) for y in cohort["y"]], "loss": ff.loss_bounds(intervals, cohort["y"]),
            "class_membership_certified": False, "population_risk_claim": False,
            "assurance": "EXACT_LOSS_OF_STORED_BASELINE_PREDICTIONS"}


def fit_baselines(train, decision, maps, protocol):
    started = time.perf_counter()
    directory = OUT / "baselines"
    directory.mkdir(parents=True, exist_ok=True)
    y = np.asarray([float(x) for x in train["y"]])
    scaler = maps["raw_minmax01"].core
    cfg = protocol["baselines"]["hist_gradient_boosting"]
    parameter_keys = ["loss", "learning_rate", "max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization", "max_features", "max_bins", "categorical_features", "monotonic_cst", "interaction_cst", "warm_start", "early_stopping", "random_state"]
    model = HistGradientBoostingRegressor(**{key: cfg[key] for key in parameter_keys})
    model.fit(scaler.transform(train["X"]), y)
    mean = sum(train["y"], F(0))/len(train["y"])
    model_path = directory / "hist_gradient_boosting.pickle"
    model_path.write_bytes(pickle.dumps({"model": model, "scaler": scaler}, protocol=5))
    write_json(directory / "training_mean.json", {"value": str(mean), "training_n": len(train["y"])})
    outputs = {"persistence": decision["persistence"], "training_mean": [mean]*len(decision["y"]),
               "hist_gradient_boosting": model.predict(scaler.transform(decision["X"])).tolist()}
    rows = []
    for name, predictions in outputs.items():
        receipt = empirical_receipt(name, predictions, decision)
        path = directory / (name + "_decision.json")
        write_json(path, receipt)
        rows.append({"baseline": name, "decision_risk": {k: receipt["loss"][k] for k in ["lower", "upper", "width"]},
                     "decision_prediction_path": rel(path),
                     "deployment_payload_bytes": model_path.stat().st_size if name == "hist_gradient_boosting" else (directory / "training_mean.json").stat().st_size if name == "training_mean" else 0,
                     "class_membership_certified": False})
    return model, scaler, mean, rows, time.perf_counter()-started


def main():
    global START, DEADLINE
    START = time.perf_counter()
    protocol = verify_protocol_seal()
    marker = OUT / "primary_started35.json"
    if marker.exists():
        raise RuntimeError("The single primary run was already started; rerun forbidden")
    write_json(marker, {"primary_run_number": 1, "started_utc": now(), "protocol_sha256": sha_file(ROOT / "protocol35.json"),
                        "protocol_seal_sha256": sha_file(ROOT / "protocol_seal35.json")})
    DEADLINE = START+protocol["execution"]["overall_wall_budget_seconds"]
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, protocol["execution"]["overall_wall_budget_seconds"])
    event("primary_start", primary_run_number=1, protocol_seal_sha256=sha_file(ROOT / "protocol_seal35.json"))
    train, decision, normalization = load_development()
    save_cohort(train); save_cohort(decision)
    persistence_train = sum(((p-y)**2 for p, y in zip(train["persistence"], train["y"])), F(0))/len(train["y"])
    epsilon = F(protocol["policy"]["epsilon_factor"])*persistence_train
    anchors = rep.temporal_indices(len(train["y"]), protocol["heads"]["extension32"]["anchor_count"])
    write_json(OUT / "training_plan35.json", {"train_persistence_mse": str(persistence_train), "epsilon": str(epsilon),
               "epsilon_physical_mse": str(epsilon*normalization["physical_mse_multiplier"]),
               "anchor_indices": anchors.tolist(), "anchor_ids": [train["ids"][i] for i in anchors],
               "test_values_parsed": False})
    progress("development_loaded", train_n=len(train["y"]), decision_n=len(decision["y"]), epsilon=float(epsilon))
    with phase_budget(protocol["execution"]["native_family_wall_budget_seconds"]):
        maps, representation_metadata = rep.fit_representations(train["X"], protocol["representations"], protocol["seed"])
    write_json(OUT / "representations35.json", representation_metadata)
    map_information, features, decision_classes = {}, {}, {}
    query_started = time.perf_counter()
    for name, representation in maps.items():
        path = OUT / "maps" / (name + ".pickle")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rep.serialize_map(representation))
        feature_id = "step35-map-sha256:" + sha_file(path)
        map_information[name] = {"path": rel(path), "sha256": sha_file(path), "feature_id": feature_id,
                                 "payload_bytes": path.stat().st_size, "numeric_bytes": rep.numeric_bytes(representation)}
        ztrain = rep.transform_one(representation, train["X"])
        zdecision = rep.transform_one(representation, decision["X"])
        save_features(name, "train", ztrain); save_features(name, "decision", zdecision)
        features[name] = {"train": ztrain, "decision": zdecision}
    representation_query_seconds = time.perf_counter()-query_started
    write_json(OUT / "map_manifest35.json", map_information)
    progress("all_representations_frozen", map_count=len(maps), physical_native_counts=representation_metadata["native_cost_accounting"])
    for name in maps:
        decision_classes[name] = class_lower(name, map_information[name]["feature_id"], "decision", features[name]["decision"], decision, protocol)
        progress("decision_class_complete", map=name, lower=float(F(decision_classes[name]["lower"])))
    excluded = [name for name in maps if F(decision_classes[name]["lower"]) > epsilon]
    decision_cert_seconds = sum(x["stage_seconds_including_witness_retention"] for x in decision_classes.values())
    screen = {"schema": "step35-predecoder-screen-v1", "epsilon": str(epsilon), "map_order": list(maps),
              "excluded_classes": excluded, "excluded_count": len(excluded),
              "lower_by_map": {name: row["lower"] for name, row in decision_classes.items()},
              "classes_evaluated_before_any_decoder_fit": True, "test_values_parsed": False,
              "decision_class_seconds": decision_cert_seconds}
    write_json(OUT / "screen35.json", screen)
    event("class_screen_frozen", excluded_classes=excluded, epsilon=str(epsilon), screen_sha256=sha_file(OUT / "screen35.json"))
    rows = []
    for shadow in [False, True]:
        if shadow:
            screened_choice = threshold_select(rows, epsilon)
            screened_choice["action"] = action(screened_choice, protocol)
            screened_record = {"selection": screened_choice, "candidate_ids": [r["candidate_id"] for r in rows],
                               "excluded_classes": excluded, "epsilon": str(epsilon), "test_values_parsed": False}
            write_json(OUT / "screened_policy35.json", screened_record)
            event("screened_policy_frozen", candidate_id=screened_choice["candidate_id"], policy_sha256=sha_file(OUT / "screened_policy35.json"))
        for name, representation in maps.items():
            if (name in excluded) != shadow:
                continue
            for recipe in protocol["heads"]["recipes_per_map"]:
                row = fit_and_decide(name, representation, map_information[name]["feature_id"], recipe,
                    features[name]["train"], features[name]["decision"], train, decision, anchors,
                    map_information[name], protocol, shadow)
                rows.append(row)
    order = {name: index for index, name in enumerate(maps)}
    rows.sort(key=lambda row: (order[row["map"]], protocol["heads"]["recipes_per_map"].index(row["recipe"])))
    exhaustive_choice = threshold_select(rows, epsilon)
    exhaustive_choice["action"] = action(exhaustive_choice, protocol)
    best = min(rows, key=lambda row: (F(row["decision_risk"]["upper"]), row["deployment_payload_bytes"], row["candidate_id"]))
    accuracy_choice = {"candidate_id": best["candidate_id"], "action": action({"candidate_id": best["candidate_id"]}, protocol)}
    if screened_choice["candidate_id"] != exhaustive_choice["candidate_id"]:
        raise AssertionError("Sound class screening changed the exhaustive tolerance/cost choice")
    for row in rows:
        if row["shadow_excluded_class"] and F(row["decision_risk"]["upper"]) <= epsilon:
            raise AssertionError("A screened class contains an achieved eligible head")
    avoided_rows = [row for row in rows if row["shadow_excluded_class"]]
    avoided_seconds = sum(row["decoder_stage_seconds_including_retention"] for row in avoided_rows)
    net_seconds = avoided_seconds-decision_cert_seconds
    policy = {"schema": "step35-frozen-policies-v1", "epsilon": str(epsilon), "screened": screened_choice,
              "exhaustive_same_rule": exhaustive_choice, "ordinary_mse_minimizer": accuracy_choice,
              "same_rule_choice_identical": True, "excluded_classes": excluded,
              "candidate_rows": rows, "all_36_heads_fitted_physically": len(rows) == 36,
              "avoided_decoder_fits_in_screened_policy": len(avoided_rows),
              "avoided_decoder_stage_seconds": avoided_seconds, "decision_class_seconds": decision_cert_seconds,
              "net_conditional_stage_seconds_saved": net_seconds,
              "positive_screening_utility_demonstrated": len(avoided_rows) > 0 and net_seconds > 0,
              "cost_claim_scope": "Measured stage accounting on one physical exhaustive-plus-screen run, not separately timed independent end-to-end pipelines",
              "test_values_parsed": False}
    write_json(OUT / "policy35.json", policy)
    decision_aggregates = [actual_aggregate(name, "decision", decision_classes[name], rows) for name in maps]
    write_json(OUT / "decision_class_summary35.json", decision_aggregates)
    model, baseline_scaler, train_mean, baseline_rows, baseline_fit_seconds = fit_baselines(train, decision, maps, protocol)
    write_json(OUT / "baseline_decisions35.json", baseline_rows)
    # Seal every pre-test output except the append-only execution log and the seal itself.
    bound_paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in ["access_and_execution35.jsonl", "model_seal35.json"])
    model_seal = {"schema": "step35-model-policy-seal-v1", "sealed_utc": now(),
                  "protocol_seal_sha256": sha_file(ROOT / "protocol_seal35.json"),
                  "all_models_and_policies_frozen": True, "test_numeric_values_parsed": False,
                  "bound_files": {rel(path): sha_file(path) for path in bound_paths}}
    write_json(OUT / "model_seal35.json", model_seal)
    event("models_and_policies_sealed", model_seal_sha256=sha_file(OUT / "model_seal35.json"),
          screened_choice=screened_choice["candidate_id"], ordinary_choice=accuracy_choice["candidate_id"])
    progress("models_and_policies_sealed", screened=screened_choice, ordinary=accuracy_choice,
             excluded_classes=len(excluded), net_conditional_stage_seconds_saved=net_seconds)
    test = load_test(normalization)
    save_cohort(test)
    test_start = time.perf_counter()
    test_aggregates = []
    for name, representation in maps.items():
        ztest = rep.transform_one(representation, test["X"])
        save_features(name, "test", ztest)
        for row in [r for r in rows if r["map"] == name]:
            head = read_json(ROOT / row["head_path"])
            evaluator = ff.evaluate_affine if row["recipe"] == "affine" else ff.evaluate_extension
            prediction = evaluator(head, ztest.tolist(), test["y"], ids=test["ids"])
            path = (ROOT / row["head_path"]).parent / "test.json"
            write_json(path, prediction)
            row["test_prediction_path"] = rel(path)
            row["test_risk"] = {key: prediction["loss"][key] for key in ["lower", "upper", "width"]}
            del prediction
        lower = class_lower(name, map_information[name]["feature_id"], "test", ztest, test, protocol)
        aggregate = actual_aggregate(name, "test", lower, rows)
        test_aggregates.append(aggregate)
        for row in [r for r in rows if r["map"] == name]:
            lo, hi = F(aggregate["class_risk"]["lower"]), F(aggregate["class_risk"]["upper"])
            rlo, rhi = F(row["test_risk"]["lower"]), F(row["test_risk"]["upper"])
            row["test_class_risk"] = aggregate["class_risk"]
            row["test_excess_risk"] = {"lower": str(max(F(0), rlo-hi)), "upper": str(max(F(0), rhi-lo))}
        progress("test_map_complete", map=name, class_lower=float(F(aggregate["class_risk"]["lower"])),
                 class_upper=float(F(aggregate["class_risk"]["upper"])))
    outputs = {"persistence": test["persistence"], "training_mean": [train_mean]*len(test["y"]),
               "hist_gradient_boosting": model.predict(baseline_scaler.transform(test["X"])).tolist()}
    for row in baseline_rows:
        receipt = empirical_receipt(row["baseline"], outputs[row["baseline"]], test)
        path = OUT / "baselines" / (row["baseline"] + "_test.json")
        write_json(path, receipt)
        row["test_prediction_path"] = rel(path)
        row["test_risk"] = {key: receipt["loss"][key] for key in ["lower", "upper", "width"]}
    test_seconds = time.perf_counter()-test_start
    result = {"schema": "step35-public-forecasting-result-v1", "status": "PRIMARY_COMPLETED_PENDING_INDEPENDENT_AUDIT",
              "completed_utc": now(), "protocol_sha256": sha_file(ROOT / "protocol35.json"),
              "protocol_seal_sha256": sha_file(ROOT / "protocol_seal35.json"), "model_seal_sha256": sha_file(OUT / "model_seal35.json"),
              "primary_run_count": 1, "assurance": ff.ASSURANCE,
              "data": {"train": train["roster"], "decision": decision["roster"], "test": test["roster"], "normalization": normalization},
              "epsilon": str(epsilon), "policy": {key: value for key, value in policy.items() if key != "candidate_rows"},
              "candidate_rows": rows, "baseline_rows": baseline_rows,
              "decision_classes": decision_aggregates, "test_classes": test_aggregates,
              "timing": {"representation_factory_seconds": representation_metadata["factory_total_seconds"],
                         "train_and_decision_representation_query_seconds": representation_query_seconds,
                         "decision_class_seconds": decision_cert_seconds, "baseline_fit_and_decision_seconds": baseline_fit_seconds,
                         "all_decoder_stages_seconds": sum(row["decoder_stage_seconds_including_retention"] for row in rows),
                         "test_evaluation_and_class_seconds": test_seconds,
                         "primary_wall_seconds": time.perf_counter()-START},
              "scientific_scope": protocol["scientific_scope"],
              "public_dataset_values_read": True, "scientific_prediction_heads_fitted": len(rows)+1,
              "reserved_patient_arrays_read": False, "reserved_patients_accessed": 0,
              "new_primary_fallbacks": 0, "adaptive_public_refits": 0,
              "native_physical_work": representation_metadata["native_cost_accounting"],
              "limitations": ["Single retrospectively complete-case public series and one temporal split",
                              "Finite-cohort stored-feature assurance; no population or future-response guarantee",
                              "Native ideal-real transform roundoff is outside the guarantee",
                              "Screening utility is relative to a fixed36-head training recipe panel",
                              "Cost timing is stage accounting, not independent reruns of competing policies"]}
    write_json(OUT / "result35.json", result)
    event("primary_complete", result_sha256=sha_file(OUT / "result35.json"), elapsed_seconds=time.perf_counter()-START)
    signal.setitimer(signal.ITIMER_REAL, 0)
    progress("PRIMARY_COMPLETE", result_sha256=sha_file(OUT / "result35.json"), positive_screening_utility=policy["positive_screening_utility_demonstrated"])


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if (OUT / "primary_started35.json").exists():
            failure = {"status": "PRIMARY_STOPPED_RETAIN_PARTIAL_NO_RERUN", "error_type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc(), "utc": now()}
            write_json(OUT / "primary_failure35.json", failure)
            event("primary_failed", error_type=type(error).__name__, message=str(error))
        raise
