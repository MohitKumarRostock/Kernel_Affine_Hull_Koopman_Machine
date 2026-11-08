"""Prospective Step36 execution: independent fresh policy fits, then sealed test.

The timed function accepts only training/decision inputs, frozen encoders,
prefit byte bounds, and the protocol. Canonical exhaustive results are used
only outside that function, after each measured execution has ended.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import signal
import statistics
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import finite_feature35 as ff
import representations35 as rep
import triggered_policy36 as policy
import models36 as models
import data36 as data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
START = None
DEADLINE = None
TIMING_FIELDS = {"fit_seconds", "exact_decision_evaluation_seconds", "fit_evaluate_and_identity_seconds"}


def rel(path):
    return str(Path(path).relative_to(ROOT))


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(stage, **fields):
    print(json.dumps({"stage": stage, "utc": now(), **fields}, sort_keys=True), flush=True)


def _timeout(signum, frame):
    raise TimeoutError("Frozen Step36 wall budget exceeded; retain partial run without rerun")


@contextmanager
def phase_budget(seconds):
    signal.setitimer(signal.ITIMER_REAL, min(seconds, max(0.001, DEADLINE-time.perf_counter())))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, max(0.001, DEADLINE-time.perf_counter()))


def save_cohort(task, cohort):
    path = OUT / "cohorts" / task / (cohort["name"] + ".json")
    numeric = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(numeric, cohort["X"], allow_pickle=False)
    record = {key: value for key, value in cohort.items() if key != "X"}
    record.update(y=[str(v) for v in cohort["y"]], persistence=[str(v) for v in cohort["persistence"]],
                  lag_matrix_path=rel(numeric), lag_matrix_sha256=data.sha_file(numeric))
    data.write_json(path, record)


def save_features(task, role, name, z):
    path = OUT / "features" / task / role / (name + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, z, allow_pickle=False)
    return rel(path)


def selected_summary(row):
    if row is None:
        return {"candidate_id": None, "head_id": None, "abstained": True}
    return {"candidate_id": row["candidate_id"], "head_id": row["head_id"], "abstained": False,
            "deployment_payload_bytes": row["deployment_payload_bytes"], "decision_risk": row["decision_risk"]}


def timed_policy(task, mode, repetition, map_order, map_info, features, bounds,
                 train, decision, anchors, epsilon, protocol):
    """Actual fit/evaluation only. No canonical reference argument or file reads."""
    started = time.perf_counter()
    bundles, certificates = [], []
    floor = policy.range_floor(decision["y"], ids=decision["ids"], output_interval=protocol["class"]["prediction_interval"])
    state = None if mode == "exhaustive" else policy.PolicyState(epsilon, mode=mode,
                        max_certificate_calls=protocol["certificate"]["max_calls_per_target_run"])
    if state is not None:
        state.set_range_floor(floor)
    counts = {"fit_calls": 0, "affine_fit_calls": 0, "extension_fit_calls": 0,
              "cost_pruned_heads": 0, "class_pruned_heads": 0, "range_pruned_heads": 0}

    def fit(name, recipe):
        bundle = models.fit_evaluate(task, name, recipe, features[name]["train"], features[name]["decision"],
                                    train, decision, map_info[name], anchors, protocol)
        check = policy.check_realized_head_cost(bounds[name][recipe], bundle["head"])
        for field in ("head_payload_bytes", "deployment_payload_bytes"):
            if check[field] != bundle["row"][field]:
                raise AssertionError("Realized head serialization and prefit bound disagree")
        bundles.append(bundle)
        counts["fit_calls"] += 1
        counts["affine_fit_calls" if recipe == "affine" else "extension_fit_calls"] += 1
        if state is not None:
            state.observe(bundle["row"])
        return bundle["row"]

    if mode == "exhaustive":
        for name in map_order:
            for recipe in protocol["heads"]["recipes"]:
                fit(name, recipe)
        rows = [bundle["row"] for bundle in bundles]
        eligible = [row for row in rows if F(row["decision_risk"]["upper"]) <= epsilon]
        selected = min(eligible, key=policy.candidate_key, default=None)
        trace = {"schema": "step36-exhaustive-search-trace-v1", "mode": mode,
                 "map_order": list(map_order), "candidate_order": [row["candidate_id"] for row in rows],
                 "range_floor_mean": floor["range_floor_mean"], "range_floor_used_to_prune": False,
                 "selection": ff.canonical(selected), "certificate_calls": 0,
                 "exact_selected_pair_distance_evaluations": 0, "approximate_distance_evaluations": 0}
    elif state.range_floor_prunes():
        counts["range_pruned_heads"] = 2*len(map_order)
        selected, trace = state.selection(), state.trace()
    else:
        for name in map_order:
            affine = None
            choice = state.affine_decision(name, bounds[name]["affine"])
            if choice["action"] == "FIT":
                affine = fit(name, "affine")
            elif choice["action"] == "COST_PRUNE":
                counts["cost_pruned_heads"] += 1
            else:
                raise AssertionError("Undeclared affine policy action")
            choice = state.extension_decision(name, bounds[name]["extension32"], affine)
            if choice["action"] == "CERTIFY":
                call_start = time.perf_counter()
                receipt = policy.pair_class_lower(features[name]["decision"].tolist(), decision["y"],
                    feature_id=map_info[name]["feature_id"], ids=decision["ids"],
                    kappa=protocol["class"]["kappa"], output_interval=protocol["class"]["prediction_interval"],
                    neighbors=protocol["certificate"]["neighbors"], max_pairs=protocol["certificate"]["max_pairs_per_call"],
                    bits=protocol["certificate"]["distance_bits"])
                choice = state.observe_certificate(name, receipt)
                certificates.append({"map": name, "receipt": receipt,
                                     "call_seconds_including_receipt_and_policy_hash": time.perf_counter()-call_start})
            action = choice["action"]
            if action == "FIT":
                fit(name, "extension32")
            elif action == "COST_PRUNE":
                counts["cost_pruned_heads"] += 1
            elif action == "CLASS_PRUNE":
                counts["class_pruned_heads"] += 1
            elif action == "RANGE_PRUNE":
                counts["range_pruned_heads"] += 1
            else:
                raise AssertionError("Undeclared extension policy action")
        selected, trace = state.selection(), state.trace()
    if counts["fit_calls"] + counts["cost_pruned_heads"] + counts["class_pruned_heads"] + counts["range_pruned_heads"] != 2*len(map_order):
        raise AssertionError("Search work accounting does not cover frozen panel")
    record = {"schema": "step36-actual-search-execution-v1", "task": task, "mode": mode,
              "repetition": repetition, "epsilon": str(epsilon), "map_order": list(map_order),
              "selection": selected_summary(selected), "counts": counts,
              "candidate_rows": [bundle["row"] for bundle in bundles], "trace": trace,
              "range_floor": floor, "range_floor_sha256": models.object_sha(floor),
              "trace_sha256": models.object_sha(trace),
              "fresh_actual_head_fits": True, "canonical_exhaustive_results_available_inside_policy": False,
              "evidence_disk_io_inside_timing": False,
              "certificate_seconds": sum(item["call_seconds_including_receipt_and_policy_hash"] for item in certificates)}
    # Include complete measured-record serialization; evidence writes follow after timing.
    models.canonical_bytes(record)
    record["elapsed_seconds"] = time.perf_counter()-started
    return record, bundles, certificates


def retain_execution(record, bundles, certificates, canonical=None):
    task, mode, r = record["task"], record["mode"], record["repetition"]
    for bundle in bundles:
        row = bundle["row"]
        directory = OUT / "heads" / task / (row["map"] + "__" + row["recipe"])
        if canonical is None:
            for kind in ("head", "fit", "decision"):
                path = directory / (kind + ".json")
                data.write_json(path, bundle[kind])
                expected = row[{"head": "head_sha256", "fit": "fit_sha256", "decision": "decision_prediction_sha256"}[kind]]
                if data.sha_file(path) != expected:
                    raise AssertionError("Retained proof differs from timed digest")
        else:
            reference = canonical[row["candidate_id"]]
            for key, value in row.items():
                if key not in TIMING_FIELDS and reference[key] != value:
                    raise AssertionError("Fresh policy fit differs from canonical exhaustive: " + row["candidate_id"] + ":" + key)
        row.update(head_path=rel(directory / "head.json"), fit_receipt_path=rel(directory / "fit.json"),
                   decision_prediction_path=rel(directory / "decision.json"))
    summaries = []
    for index, item in enumerate(certificates):
        path = OUT / "pairs" / task / f"{mode}_r{r:02}" / (item["map"] + ".json")
        data.write_json(path, item["receipt"])
        summaries.append({"map": item["map"], "path": rel(path), "sha256": data.sha_file(path),
                          "certificate_digest": ff.digest(item["receipt"]),
                          "lower_mean": item["receipt"]["lower_mean"], "work": item["receipt"]["work"],
                          "call_seconds_including_receipt_and_policy_hash": item["call_seconds_including_receipt_and_policy_hash"]})
    record["certificates"] = summaries
    record["canonical_identity_checked_after_timing"] = canonical is not None
    record["canonical_prediction_proofs_retained_once"] = True
    path = OUT / "search" / task / f"{mode}_r{r:02}.json"
    data.write_json(path, record)
    return rel(path)


def task_statistics(exhaustive, executions):
    ordinary = sorted((row for row in executions if row["mode"] == "ordinary"), key=lambda x:x["repetition"])
    triggered = sorted((row for row in executions if row["mode"] == "triggered"), key=lambda x:x["repetition"])
    safety = all(row["selection"] == exhaustive["selection"] for row in executions)
    if not safety:
        raise AssertionError("Search changed the exhaustive frozen-rule choice")
    diffs = [o["elapsed_seconds"]-c["elapsed_seconds"] for o,c in zip(ordinary,triggered)]
    added_avoided = [o["counts"]["fit_calls"]-c["counts"]["fit_calls"] for o,c in zip(ordinary,triggered)]
    if len(set(added_avoided)) != 1:
        raise AssertionError("Deterministic fit savings changed across timing repetitions")
    return {"safety_pass": safety, "exhaustive_selection": exhaustive["selection"],
            "exhaustive_fit_calls": exhaustive["counts"]["fit_calls"],
            "ordinary_fit_calls_each_repetition": [r["counts"]["fit_calls"] for r in ordinary],
            "triggered_fit_calls_each_repetition": [r["counts"]["fit_calls"] for r in triggered],
            "additional_head_fits_avoided_each_repetition": added_avoided,
            "ordinary_seconds": [r["elapsed_seconds"] for r in ordinary],
            "triggered_seconds": [r["elapsed_seconds"] for r in triggered],
            "paired_ordinary_minus_triggered_seconds": diffs,
            "median_paired_seconds_saved": statistics.median(diffs),
            "ordinary_median_seconds": statistics.median(r["elapsed_seconds"] for r in ordinary),
            "triggered_median_seconds": statistics.median(r["elapsed_seconds"] for r in triggered),
            "ordinary_seconds_range": [min(r["elapsed_seconds"] for r in ordinary),max(r["elapsed_seconds"] for r in ordinary)],
            "triggered_seconds_range": [min(r["elapsed_seconds"] for r in triggered),max(r["elapsed_seconds"] for r in triggered)],
            "triggered_certificate_calls_each_repetition": [r["trace"]["certificate_calls"] for r in triggered],
            "triggered_class_pruned_heads_each_repetition": [r["counts"]["class_pruned_heads"] for r in triggered],
            "triggered_exact_pair_distances_each_repetition": [r["trace"]["exact_selected_pair_distance_evaluations"] for r in triggered],
            "positive_incremental_certificate_utility": safety and added_avoided[0]>0 and statistics.median(diffs)>0,
            "scope": "Five alternating matched timings on this one target, no iid timing inference"}


def main():
    global START, DEADLINE
    START = time.perf_counter()
    protocol = data.verify_protocol_seal()
    marker = OUT / "primary_started36.json"
    if marker.exists():
        raise RuntimeError("Single primary execution already started; rerun forbidden")
    data.write_json(marker, {"primary_run_number": 1, "started_utc": now(),
               "protocol_sha256": data.sha_file(ROOT / "protocol36.json"), "protocol_seal_sha256": data.sha_file(ROOT / "protocol_seal36.json")})
    DEADLINE = START+protocol["execution"]["overall_wall_budget_seconds"]
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, protocol["execution"]["overall_wall_budget_seconds"])
    data.event("primary_start", primary_run_number=1, protocol_seal_sha256=data.sha_file(ROOT / "protocol_seal36.json"))
    train_tasks, decision_tasks, normalizations = data.load_development()
    all_rows, all_baseline_rows, task_results, context = [], [], {}, {}
    for task_index, task in enumerate(protocol["tasks"]):
        train, decision = train_tasks[task], decision_tasks[task]
        save_cohort(task, train); save_cohort(task, decision)
        if len(train["y"]) < protocol["data"]["minimum_training_rows"]:
            raise RuntimeError("Insufficient training rows; retain frozen failure")
        persistence_mse = sum(((p-y)**2 for p,y in zip(train["persistence"],train["y"])),F(0))/len(train["y"])
        epsilon = F(protocol["selection"]["epsilon_factor"])*persistence_mse
        anchors = rep.temporal_indices(len(train["y"]), protocol["heads"]["anchor_count"])
        plan = {"task": task, "train_persistence_mse": str(persistence_mse), "epsilon": str(epsilon),
                "epsilon_physical_mse": str(epsilon*normalizations[task]["physical_mse_multiplier"]),
                "anchor_indices": anchors.tolist(), "anchor_ids": [train["ids"][i] for i in anchors], "test_values_parsed": False}
        data.write_json(OUT / "tasks" / task / "training_plan36.json", plan)
        progress("development_task", task=task, train_n=len(train["y"]),decision_n=len(decision["y"]),epsilon=float(epsilon))
        with phase_budget(protocol["execution"]["native_family_budget_seconds_per_target"]):
            maps, metadata = rep.fit_representations(train["X"], protocol["representations"], protocol["seed"])
        data.write_json(OUT / "tasks" / task / "representations36.json", metadata)
        map_info, features, bounds = {}, {}, {}
        cache_start = time.perf_counter()
        for name, representation in maps.items():
            path = OUT / "maps" / task / (name + ".pickle")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rep.serialize_map(representation))
            feature_id = "step36-" + task + "-map-sha256:" + data.sha_file(path)
            map_info[name] = {"path": rel(path), "sha256": data.sha_file(path), "feature_id": feature_id,
                              "payload_bytes": path.stat().st_size, "numeric_bytes": rep.numeric_bytes(representation)}
            features[name] = {}
            for role, cohort in (("train",train),("decision",decision)):
                z = rep.transform_one(representation, cohort["X"])
                save_features(task, role, name, z)
                z.flags.writeable = False
                features[name][role] = z
        cache_seconds = time.perf_counter()-cache_start
        data.write_json(OUT / "tasks" / task / "map_manifest36.json", map_info)
        template_start = time.perf_counter()
        for name in maps:
            kwargs = {"feature_id": map_info[name]["feature_id"], "dimension": int(features[name]["train"].shape[1]),
                      "map_payload_bytes": map_info[name]["payload_bytes"], "kappa": protocol["class"]["kappa"],
                      "output_interval": protocol["class"]["prediction_interval"]}
            bounds[name] = {"affine": policy.head_cost_lower_bound("affine", **kwargs),
                           "extension32": policy.head_cost_lower_bound("extension32", **kwargs,
                               anchor_features=features[name]["train"][anchors].tolist(),anchor_ids=[train["ids"][i] for i in anchors])}
        template_seconds = time.perf_counter()-template_start
        data.write_json(OUT / "tasks" / task / "head_cost_bounds36.json", bounds)
        order = policy.map_order({name: info["payload_bytes"] for name,info in map_info.items()})
        progress("representations_frozen",task=task,map_count=len(maps),factory_seconds=metadata["factory_total_seconds"])
        args = (order,map_info,features,bounds,train,decision,anchors,epsilon,protocol)
        data.event("search_execution_start",task=task,mode="exhaustive",repetition=0)
        exhaustive,bundles,certificates = timed_policy(task,"exhaustive",0,*args)
        exhaustive_path = retain_execution(exhaustive,bundles,certificates)
        data.event("search_execution_complete",task=task,mode="exhaustive",repetition=0,
                   search_path=exhaustive_path,search_sha256=data.sha_file(ROOT/exhaustive_path),elapsed_seconds=exhaustive["elapsed_seconds"])
        canonical = {row["candidate_id"]:row for row in exhaustive["candidate_rows"]}
        rows = list(canonical.values())
        del bundles, certificates
        progress("exhaustive_complete",task=task,choice=exhaustive["selection"],seconds=exhaustive["elapsed_seconds"])
        executions, search_paths, paired_order = [], [exhaustive_path], []
        for r in range(protocol["timing"]["ordinary_triggered_repetitions"]):
            modes = ("ordinary","triggered") if (task_index+r)%2==0 else ("triggered","ordinary")
            paired_order.append(list(modes))
            for mode in modes:
                data.event("search_execution_start",task=task,mode=mode,repetition=r)
                record,bundles,certificates = timed_policy(task,mode,r,*args)
                search_path=retain_execution(record,bundles,certificates,canonical)
                search_paths.append(search_path)
                data.event("search_execution_complete",task=task,mode=mode,repetition=r,
                           search_path=search_path,search_sha256=data.sha_file(ROOT/search_path),elapsed_seconds=record["elapsed_seconds"])
                executions.append(record)
                progress("policy_execution",task=task,mode=mode,repetition=r,seconds=record["elapsed_seconds"],
                         fits=record["counts"]["fit_calls"],pair_calls=record["trace"]["certificate_calls"],
                         class_pruned=record["counts"]["class_pruned_heads"])
                del bundles,certificates
        stats = task_statistics(exhaustive,executions)
        accuracy_choice = min(rows,key=lambda row:(F(row["decision_risk"]["upper"]),row["deployment_payload_bytes"],row["candidate_id"]))
        baseline = models.fit_baselines(train,decision,maps["raw_minmax01"],protocol)
        base_path = OUT / "baselines" / task / "hist_gradient_boosting.pickle"
        base_path.parent.mkdir(parents=True,exist_ok=True);base_path.write_bytes(baseline["payload"])
        base_rows=[]
        for name,receipt in baseline["decision_receipts"].items():
            path = base_path.parent / (name+"_decision.json");data.write_json(path,receipt)
            base_rows.append({"task":task,"baseline":name,"decision_prediction_path":rel(path),
                "decision_risk":{key:receipt["loss"][key] for key in ("lower","upper","width")},
                "deployment_payload_bytes":len(baseline["payload"]) if name=="hist_gradient_boosting" else None,
                "class_membership_certified":False})
        data.write_json(base_path.parent / "baseline_decisions36.json",base_rows)
        task_results[task] = {"task":task,"epsilon":str(epsilon),"training_plan":plan,"map_order":order,
             "search_paths":search_paths,"paired_execution_order":paired_order,"search_statistics":stats,
             "exhaustive_selection":exhaustive["selection"],"ordinary_mse_minimizer":selected_summary(accuracy_choice),
             "range_floor_mean":exhaustive["range_floor"]["range_floor_mean"],
             "timing":{"representation_factory_seconds":metadata["factory_total_seconds"],
                 "train_decision_feature_cache_and_map_retention_seconds":cache_seconds,
                 "head_template_construction_seconds":template_seconds,
                 "exhaustive_policy_seconds":exhaustive["elapsed_seconds"],
                 "baseline_fit_and_decision_seconds":baseline["fit_and_decision_seconds"]},
             "native_physical_work":metadata["native_cost_accounting"],"test_values_parsed_at_policy_choice":False}
        data.write_json(OUT / "tasks" / task / "policy36.json",task_results[task])
        all_rows.extend(rows);all_baseline_rows.extend(base_rows)
        context[task]={"maps":maps,"baseline":baseline,"rows":rows,"baseline_rows":base_rows}
        data.event("task_policies_complete",task=task,selection=exhaustive["selection"],
                   positive_incremental_utility=stats["positive_incremental_certificate_utility"],test_values_parsed=False)
    policy_result={"schema":"step36-global-frozen-policy-v1","tasks":task_results,
                   "all_task_safety_pass":all(t["search_statistics"]["safety_pass"] for t in task_results.values()),
                   "consistent_panel_incremental_utility":all(t["search_statistics"]["positive_incremental_certificate_utility"] for t in task_results.values()),
                   "test_values_parsed":False}
    data.write_json(OUT / "policies36.json",policy_result)
    bound_paths=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in ("access_and_execution36.jsonl","model_seal36.json"))
    data.write_json(OUT / "model_seal36.json",{"schema":"step36-all-target-model-policy-seal-v1","sealed_utc":now(),
         "protocol_seal_sha256":data.sha_file(ROOT / "protocol_seal36.json"),"all_models_and_policies_frozen":True,
         "test_numeric_values_parsed":False,"bound_files":{rel(path):data.sha_file(path) for path in bound_paths}})
    data.event("models_and_policies_sealed",model_seal_sha256=data.sha_file(OUT / "model_seal36.json"),all_targets=protocol["tasks"])
    progress("all_target_test_barrier_sealed",safety=policy_result["all_task_safety_pass"],incremental_utility=policy_result["consistent_panel_incremental_utility"])
    tests=data.load_test(normalizations)
    for task in protocol["tasks"]:
        test=tests[task];save_cohort(task,test);test_start=time.perf_counter()
        ctx=context[task]
        for name,representation in ctx["maps"].items():
            z=rep.transform_one(representation,test["X"]);save_features(task,"test",name,z)
            for row in (row for row in ctx["rows"] if row["map"]==name):
                head=json.loads((ROOT / row["head_path"]).read_text())
                evaluator=ff.evaluate_affine if row["recipe"]=="affine" else ff.evaluate_extension
                receipt=evaluator(head,z.tolist(),test["y"],ids=test["ids"])
                path=(ROOT / row["head_path"]).parent / "test.json";data.write_json(path,receipt)
                row.update(test_prediction_path=rel(path),test_prediction_sha256=data.sha_file(path),
                           test_risk={key:receipt["loss"][key] for key in ("lower","upper","width")})
        baseline=ctx["baseline"]
        predictions={"persistence":test["persistence"],"training_mean":[baseline["training_mean"]]*len(test["y"]),
                     "hist_gradient_boosting":baseline["model"].predict(baseline["scaler"].transform(test["X"])).tolist()}
        for row in ctx["baseline_rows"]:
            receipt=models.baseline_receipt(row["baseline"],predictions[row["baseline"]],test)
            path=OUT / "baselines" / task / (row["baseline"]+"_test.json");data.write_json(path,receipt)
            row.update(test_prediction_path=rel(path),test_prediction_sha256=data.sha_file(path),
                       test_risk={key:receipt["loss"][key] for key in ("lower","upper","width")})
        task_results[task]["timing"]["test_evaluation_and_retention_seconds"]=time.perf_counter()-test_start
        progress("test_task_complete",task=task,test_n=len(test["y"]),heads=len(ctx["rows"]))
    result={"schema":"step36-triggered-search-result-v1","status":"PRIMARY_COMPLETED_PENDING_INDEPENDENT_AUDIT",
        "completed_utc":now(),"protocol_sha256":data.sha_file(ROOT / "protocol36.json"),
        "protocol_seal_sha256":data.sha_file(ROOT / "protocol_seal36.json"),"model_seal_sha256":data.sha_file(OUT / "model_seal36.json"),
        "primary_run_count":1,"assurance":ff.ASSURANCE,"tasks":task_results,
        "candidate_rows":all_rows,"baseline_rows":all_baseline_rows,
        "data":{task:{"train":train_tasks[task]["roster"],"decision":decision_tasks[task]["roster"],
                       "test":tests[task]["roster"],"normalization":normalizations[task]} for task in protocol["tasks"]},
        "all_task_safety_pass":policy_result["all_task_safety_pass"],
        "consistent_panel_incremental_utility":policy_result["consistent_panel_incremental_utility"],
        "canonical_prediction_heads_fitted":len(all_rows),
        "actual_search_head_fits":sum(json.loads((ROOT / p).read_text())["counts"]["fit_calls"] for t in task_results.values() for p in t["search_paths"]),
        "baseline_models_fitted":3,"reserved_patient_arrays_read":False,"reserved_patients_accessed":0,
        "adaptive_public_refits":0,"predeclared_actual_timing_repetitions":5,"primary_wall_seconds":time.perf_counter()-START,
        "inherited_study_statuses":protocol["execution"]["inherited_study_statuses"],
        "limitations":["One dataset with three algebraically related targets and one temporal split",
             "Retrospective complete-case daily target roster, no imputation",
             "Finite-cohort assurance conditional on stored binary64 features as exact coordinates",
             "No population MSE, future response coverage, ideal-real native transform, deployment latency or runtime-size guarantee",
             "Timing is in-memory head search after shared representations, features and templates; evidence disk writes excluded",
             "No adaptive threshold tightening or new comparison model tuning"]}
    data.write_json(OUT / "result36.json",result)
    data.event("primary_complete",result_sha256=data.sha_file(OUT / "result36.json"),elapsed_seconds=time.perf_counter()-START)
    signal.setitimer(signal.ITIMER_REAL,0)
    progress("PRIMARY_COMPLETE",result_sha256=data.sha_file(OUT / "result36.json"),
             safety=result["all_task_safety_pass"],incremental_utility=result["consistent_panel_incremental_utility"])


if __name__=="__main__":
    try:
        with threadpool_limits(limits=1):
            main()
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        failure={"status":"PRIMARY_STOPPED_RETAIN_PARTIAL_NO_RERUN","utc":now(),"error":repr(exc),"traceback":traceback.format_exc()}
        data.write_json(OUT / "failure36.json",failure)
        data.event("primary_failure",error=repr(exc))
        raise
