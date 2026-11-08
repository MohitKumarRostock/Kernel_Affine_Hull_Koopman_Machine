"""Post-seal evaluation and fixed-reference reporting for the Step24 panel.

Importing this module performs no benchmark access, prediction, or fitting.
Every public stage checks the complete six-task fit seal before opening cached
test arrays or invoking the production test loader. Outputs are write-once;
completed stages may only be reused after their complete file manifest passes.
Timing is implemented separately, so its worker imports no evaluated features.
"""
from __future__ import annotations

from fractions import Fraction
import importlib.util
import os
from pathlib import Path, PurePosixPath
import re
import resource
import sys
import time

import common24 as C
import numpy as np
import data24 as D
import geometry24 as G
import models24 as M
import pipeline24 as P


_EVALUATED_IN_PROCESS = False
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_TEST_ARRAYS = ("test_X", "test_Y", "test_anchor_ids")


def _bits_equal(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _path_in(folder, relative):
    p = PurePosixPath(relative)
    if not isinstance(relative, str) or p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Invalid output manifest path")
    path = Path(folder) / str(p)
    if not path.resolve().is_relative_to(Path(folder).resolve()):
        raise ValueError("Output manifest escaped the task directory")
    return path


def _write_npz(path, arrays):
    """Commit one archive without ever replacing an existing artifact."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    # link() is exclusive even if another worker creates path after the check.
    os.link(temporary, path)
    temporary.unlink()


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _verify_completion(folder, name, fit_sha, task, required_files=()):
    receipt = C.read_json(Path(folder) / name)
    if (receipt.get("passed") is not True or receipt.get("task") != task
            or receipt.get("fit_seal_sha256") != fit_sha):
        raise RuntimeError("Completion receipt does not bind the current task/fit seal")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files or not set(required_files) <= set(files):
        raise RuntimeError("Incomplete output hash manifest")
    for relative, digest in files.items():
        if C.sha(_path_in(folder, relative)) != digest:
            raise RuntimeError("Completed output changed: " + relative)
    return receipt


def _task_contract(task):
    kind = C.task_kind(task)
    if kind == "hmm":
        spec = C.read_json(C.PARENT / "controlled/hmm_spec23.json")
        rows = int(spec["rows_per_replication"]["test"])
        return {"horizons": spec["horizons"], "primary_horizon": spec["primary_horizon"],
                "rows": rows, "dimension": int(spec["history_length"]),
                "anchor_ids": int(spec["role_codes"]["test"]) * 1_000_000 + np.arange(rows, dtype=np.int64)}
    spec = C.read_json(C.PARENT / "data/appliances_protocol23.json")
    lo, hi = spec["partitions"]["eligible_anchors_zero_based_half_open"]["test"]
    return {"horizons": spec["forecast"]["horizons_records"],
            "primary_horizon": spec["forecast"]["primary_horizon_records"],
            "rows": int(spec["partitions"]["eligible_anchor_counts"]["test"]),
            "dimension": int(spec["features"]["dimension"]),
            "anchor_ids": np.arange(lo, hi, dtype=np.int64)}


def _validate_test_data(task, data, *, expected=None):
    """Validate the common cohort; expected is a manufactured-only test seam."""
    contract = _task_contract(task) if expected is None else expected
    horizons = list(data["horizons"])
    if horizons != list(contract["horizons"]) or len(set(horizons)) != 3:
        raise ValueError("Test horizon order differs from the frozen three-horizon cohort")
    primary = horizons.index(contract["primary_horizon"])
    if data["primary_column"] != primary:
        raise ValueError("Test primary horizon column changed")
    X, Y, ids = (np.asarray(data[key]) for key in _TEST_ARRAYS)
    rows = int(contract["rows"])
    if X.shape != (rows, int(contract["dimension"])) or Y.shape != (rows, len(horizons)):
        raise ValueError("Test feature/target dimensions or common row roster changed")
    if X.dtype != np.float64 or Y.dtype != np.float64 or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Test arrays must retain finite binary64 source values")
    if ids.dtype.kind not in "iu" or ids.shape != (rows,) or not np.array_equal(ids, contract["anchor_ids"]):
        raise ValueError("Test anchor identities/order differ from the fixed cohort")
    if len(np.unique(ids)) != rows:
        raise ValueError("Duplicated test anchor")
    if C.task_kind(task) == "hmm":
        if not np.isin(X, [-1., -.5, .5, 1.]).all() or not np.isin(Y, [-1., -.5, .5, 1.]).all():
            raise ValueError("HMM arrays contain values outside the exact symbol alphabet")
    return data


def _load_test_cache(task, fit_seal=None):
    """Open or create the fixed test cohort only after the complete fit gate."""
    checked = C.check_fit_seal()
    if fit_seal is not None and checked != fit_seal:
        raise RuntimeError("Fit seal changed between gate checks")
    fit_sha = C.sha(C.RUN / "fit_seal.json")
    folder = C.RUN / task
    stored, receipt_path = folder / "test.npz", folder / "data_access_test.json"
    if stored.exists() != receipt_path.exists():
        raise RuntimeError("Incomplete test cache; no automatic regeneration or overwrite")
    if stored.exists():
        receipt = C.read_json(receipt_path)
        if (receipt.get("schema") != "step24-test-access-cache-v1"
                or receipt.get("task") != task or receipt.get("fit_seal_sha256") != fit_sha
                or receipt.get("test_npz_sha256") != C.sha(stored)
                or receipt.get("protocol_sha256") != checked["protocol_sha256"]):
            raise RuntimeError("Stale or changed test cache/access receipt")
        arrays = _read_npz(stored)
        if set(arrays) != set(_TEST_ARRAYS):
            raise RuntimeError("Unexpected test cache array roster")
        if receipt.get("array_sha256") != {key: C.array_sha(value) for key, value in arrays.items()}:
            raise RuntimeError("Retained test array hashes changed")
        data = {**arrays, "horizons": receipt["returned_horizons"],
                "primary_column": receipt["returned_primary_column"], "metadata": receipt["loader_metadata"]}
        return _validate_test_data(task, data)
    started = time.perf_counter_ns()
    data = D.load_test(task, gate_path=C.RUN / "fit_seal.json")
    load_ns = time.perf_counter_ns() - started
    _validate_test_data(task, data)
    arrays = {key: data[key] for key in _TEST_ARRAYS}
    _write_npz(stored, arrays)
    C.write_json(receipt_path, {
        "schema": "step24-test-access-cache-v1", "task": task, "created_utc": C.now(),
        "fit_seal_sha256": fit_sha, "protocol_sha256": checked["protocol_sha256"],
        "returned_horizons": data["horizons"], "returned_primary_column": data["primary_column"],
        "loader_metadata": data["metadata"], "test_npz_sha256": C.sha(stored),
        "array_sha256": {key: C.array_sha(value) for key, value in arrays.items()},
        "raw_test_loading_and_common_history_construction_nanoseconds": load_ns,
        "timing_scope": "Gate recheck, source integrity, test generation/parsing and common feature construction; no model-specific encoding",
        "all_six_fit_seal_checked_before_test_access": True, "ECG_or_reserve_access": False})
    return data


def _required_names(kind):
    result = {f"{group}_{family}" for group in ("P", "CS", "RAW") for family in ("ridge", "hgb")}
    result |= {"NYSTROM_ridge", "C_accuracy", "C_cost", "P_accuracy", "P_cost", "CS_accuracy", "CS_cost"}
    result |= {f"matched_{view}_{family}" for view in ("P", "CS", "LR0", "LRM", "LW") for family in ("ridge", "hgb")}
    result |= {"persistence", "training_mean"}
    if kind == "appliances":
        result.add("daily_seasonal")
    return result


def _validate_roster(task, selected, models, metadata, target_maps):
    kind, contract = C.task_kind(task), _task_contract(task)
    horizons = list(selected["horizons"])
    if horizons != list(contract["horizons"]) or selected["primary_horizon"] != contract["primary_horizon"]:
        raise ValueError("Selected horizon roster changed")
    hkeys = {str(h) for h in horizons}
    if set(models) != hkeys or set(metadata) != hkeys or set(target_maps) != hkeys:
        raise ValueError("Models, metadata and shared target maps require exactly all three horizons")
    ids, names = list(selected["selected_ids"]), selected["names"]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(cid, str) or not _SAFE_ID.fullmatch(cid) for cid in ids):
        raise ValueError("Invalid or duplicated selected model id")
    if set(names) != _required_names(kind) or set(ids) != {cid for cid in names.values() if cid is not None}:
        raise ValueError("Frozen named selections and deduplicated model roster differ")
    if any(cid is None and name not in {"P_cost", "CS_cost"} for name, cid in names.items()):
        raise ValueError("Only unqualified P_cost/CS_cost may be absent")
    if selected["source_protocol_sha256"] != C.sha(C.PARENT / "PROTOCOL.json"):
        raise ValueError("Selected roster protocol binding changed")
    for name in ("C_accuracy", "C_cost", "P_accuracy", "P_cost", "CS_accuracy", "CS_cost"):
        if selected["selectors"][name] != names[name]:
            raise ValueError("Named primary selector binding changed")
    primary = str(selected["primary_horizon"])
    first_specs = {cid: metadata[primary][cid]["source_spec"] for cid in ids}
    for h in horizons:
        key = str(h)
        if set(models[key]) != set(ids) or set(metadata[key]) != set(ids):
            raise ValueError("Secondary horizon did not preserve the exact selected candidate cohort")
        M._target_map(target_maps[key])
        for cid in ids:
            spec, pipe = metadata[key][cid]["source_spec"], models[key][cid]
            if spec != first_specs[cid] or spec.get("id") != cid or pipe.get("task_kind") != kind:
                raise ValueError("Transferred model specification/task changed")
            group = spec["group"]
            expected_mode = ("geometry" if group in {"P", "CS", "MATCHED_DIAGNOSTIC"}
                             else {"RAW": "raw", "NYSTROM": "nystrom", "UNFITTED": "baseline"}.get(group))
            if pipe.get("pipeline_kind") != expected_mode or expected_mode is None:
                raise ValueError("Model family does not match its frozen candidate specification")
            if expected_mode in {"geometry", "raw"} and pipe["head"].get("kind") != spec["family"]:
                raise ValueError("Readout family changed after selection")
            if expected_mode == "geometry":
                transform = pipe["transform"]
                if (transform["view"] != spec["view"]
                        or tuple(map(int, pipe["geometry"]["block_sizes"])) != tuple(P.block_sizes(spec["prefix"]))
                        or tuple(map(int, transform["block_sizes"])) != tuple(P.block_sizes(spec["prefix"]))):
                    raise ValueError("Selected representation or state prefix changed")
                if spec["view"] in {"P", "LRM", "LW"} and float(transform["omega"]) != spec["omega"]:
                    raise ValueError("Selected concentration changed")
            cost = metadata[key][cid]["cost"]
            for field in ("complete_numeric_bytes", "serialized_model_bytes"):
                if isinstance(cost[field], bool) or not isinstance(cost[field], int) or cost[field] < 0:
                    raise ValueError("Invalid complete deployment cost")
    for name, cid in names.items():
        if cid is None:
            continue
        spec = first_specs[cid]
        if name in {"C_accuracy", "C_cost"} and spec["group"] not in {"RAW", "NYSTROM", "UNFITTED"}:
            raise ValueError("Frozen comparator reference is not a comparator")
        if name in {"P_accuracy", "P_cost", "CS_accuracy", "CS_cost"} and spec["group"] != name.split("_")[0]:
            raise ValueError("Frozen P/CS selector has the wrong candidate group")
        if name.startswith("matched_"):
            _, view, family = name.split("_")
            if spec["view"] != view or spec["family"] != family:
                raise ValueError("Matched coordinate family changed")
            binding = selected["binding"]
            if any(spec[k] != binding[k] for k in ("scaling", "cap", "prefix")) or (view != "CS" and spec["omega"] != binding["omega"]):
                raise ValueError("Matched coordinates do not use the frozen P accuracy geometry")
    return horizons, ids


def _prediction_replay(pipe, prediction, independent, features, independent_features, target_map, label):
    """Audit a second full query and an independent readout on the same layout."""
    if (features is None) != (independent_features is None):
        raise ArithmeticError("Independent replay changed its feature contract")
    identical_features = None
    if features is not None:
        identical_features = _bits_equal(features, independent_features)
        if not identical_features:
            raise ArithmeticError("Repeated identical-layout query changed the exported feature matrix")
    hgb = pipe["pipeline_kind"] in {"geometry", "raw"} and pipe["head"]["kind"] == "hgb"
    if hgb and not _bits_equal(prediction, independent):
        raise ArithmeticError("HGB replay changed predictions on identical binary64 features")
    result = M.audit_export(prediction, independent, target_map, label)
    result.update(identical_feature_bits=identical_features,
                  prediction_bits_identical=_bits_equal(prediction, independent),
                  independent_head_or_kernel_replay=True,
                  changed_prediction_values=int(np.count_nonzero(np.asarray(prediction) != np.asarray(independent))))
    return result


def _independent_mse(y, prediction):
    # All terms are dyadic. This deliberately uses a different simple rational
    # accumulation from common24's common-exponent integer algorithm.
    total = sum(((Fraction.from_float(float(a)) - Fraction.from_float(float(b))) ** 2
                 for a, b in zip(y, prediction)), Fraction())
    return total / len(y)


def _risk_and_replay(y, prediction):
    record = C.risk_record(y, prediction)
    if record["mse_exact"] != _independent_mse(y, prediction):
        raise ArithmeticError("Independent exact dyadic metric replay failed")
    record["independent_exact_metric_replay_passed"] = True
    return record


def _appliances_block_risks(Y, horizons, predictions, anchor_ids):
    if len(anchor_ids) != 26 * 144 + 24 or Y.shape != (len(anchor_ids), 3):
        raise ValueError("Appliances descriptive panel requires all 3768 origins")
    if not np.array_equal(np.diff(anchor_ids), np.ones(len(anchor_ids) - 1, dtype=np.int64)):
        raise ValueError("Appliances blocks require chronological consecutive origins")
    rows = []
    for index, first in enumerate(range(0, len(anchor_ids), 144)):
        stop = min(first + 144, len(anchor_ids))
        rows.append({"block_index": index, "start_row_in_test": first, "stop_row_in_test": stop,
                     "rows": stop - first, "first_anchor_id": int(anchor_ids[first]),
                     "last_anchor_id": int(anchor_ids[stop - 1]), "final_remainder": index == 26,
                     "metrics": {str(h): {cid: C.risk_record(Y[first:stop, col], p[first:stop])
                                 for cid, p in predictions[str(h)].items()}
                                 for col, h in enumerate(horizons)}})
    return {"schema": "step24-appliances-descriptive-block-risk-v1", "blocks": rows,
            "full_blocks": 26, "full_block_origins": 144, "remainder_origins": 24,
            "all_origins": len(anchor_ids), "scope": "Paired descriptive risks on fixed dependent chronological blocks; no independent-sample or confidence-interval claim"}


def evaluate_task(task):
    global _EVALUATED_IN_PROCESS
    seal = C.check_fit_seal()
    C.task_kind(task)
    fit_sha, folder = C.sha(C.RUN / "fit_seal.json"), C.RUN / task
    completion = folder / "evaluation_complete.json"
    if completion.exists():
        _verify_completion(folder, completion.name, fit_sha, task, ("evaluation.json", "predictions.npz", "test.npz", "data_access_test.json"))
        C.log(task, "evaluation_already_complete")
        return C.read_json(folder / "evaluation.json")
    _EVALUATED_IN_PROCESS = True
    started = time.perf_counter()
    outputs = ["evaluation.json", "predictions.npz", "test_features.npz", "test_scores.npz",
               "test_feature_diagnostics.json", "representation_diagnostics_test.json", "paired_coordinate_diagnostics.json"]
    outputs += ["oracle.json", "oracle_arrays.npz"] if C.task_kind(task) == "hmm" else ["appliances_block_risks.json"]
    if any((folder / name).exists() for name in outputs):
        raise RuntimeError("Incomplete evaluation artifacts exist; no automatic overwrite or repeated outcome evaluation")
    selected = C.read_json(folder / "selected.json")
    models = C.read_pickle(folder / "selected_models.pkl")
    metadata = C.read_json(folder / "selected_model_metadata.json")
    maps = C.read_json(folder / "target_maps.json")
    horizons, ids = _validate_roster(task, selected, models, metadata, maps)
    data = _load_test_cache(task, seal)
    X, Y = data["test_X"], data["test_Y"]
    C.log(task, "sealed_test_evaluation_start", rows=len(X), horizons=horizons, selected_models=len(ids))
    G.initialize_runtime()
    metrics, replays, predictions, reloaded_costs = {}, {}, {}, {}
    feature_arrays, score_arrays, feature_keys, feature_info, representation = {}, {}, {}, {}, {}
    primary = str(selected["primary_horizon"])
    for col, h in enumerate(horizons):
        key = str(h)
        metrics[key], replays[key], predictions[key], reloaded_costs[key] = {}, {}, {}, {}
        for cid in ids:
            pipe = models[key][cid]
            prediction, features = P.predict(pipe, X, return_features=True)
            independent, reference_features = P.predict(pipe, X, reference=True, return_features=True)
            replays[key][cid] = _prediction_replay(pipe, prediction, independent, features, reference_features,
                                                 maps[key], f"{task}/h{h}/{cid} independent full-test replay")
            metrics[key][cid] = _risk_and_replay(Y[:, col], prediction)
            predictions[key][cid] = prediction
            cost = P.payload_cost(pipe)
            frozen_cost = metadata[key][cid]["cost"]
            if cost["complete_numeric_bytes"] != frozen_cost["complete_numeric_bytes"]:
                raise ArithmeticError("Selected complete numeric inventory changed after payload reload")
            reloaded_costs[key][cid] = {
                "complete_numeric_bytes": cost["complete_numeric_bytes"],
                "numeric_inventory_matches_frozen_selection": True,
                "frozen_serialized_model_bytes": frozen_cost["serialized_model_bytes"],
                "reserialized_model_bytes_after_reload": cost["serialized_model_bytes"],
                "serialization_byte_difference": cost["serialized_model_bytes"] - frozen_cost["serialized_model_bytes"],
                "scope": "Serialization length can reflect pickle memoization after reload; the fixed memory gate uses complete retained numeric bytes"}
            if features is None:
                feature_keys[cid] = None
            else:
                fkey = "f_" + cid
                if fkey in feature_arrays and not _bits_equal(features, feature_arrays[fkey]):
                    raise ArithmeticError("Transferred horizon changed its frozen raw-to-feature construction")
                feature_arrays.setdefault(fkey, features)
                feature_keys[cid] = fkey
                feature_info[cid] = {"array_key": fkey, "array_sha256": C.array_sha(features),
                                     "shape": list(features.shape), "retained_feature_array_bytes": features.nbytes,
                                     "minimum": float(features.min()), "maximum": float(features.max()),
                                     "zero_coordinates": int(np.count_nonzero(features == 0)),
                                     "pipeline_kind": pipe["pipeline_kind"]}
                if pipe["pipeline_kind"] == "geometry":
                    spec = metadata[key][cid]["source_spec"]
                    skey = f"s_{G.bank_key(spec['scaling'], spec['cap'])}_p{spec['prefix']}"
                    if skey not in score_arrays:
                        score_arrays[skey] = G.score_compact(pipe["geometry"], X)
                    diagnostic = G.representation_diagnostics(score_arrays[skey], pipe["transform"], features)
                    diagnostic.update(score_array_key=skey, feature_array_key=fkey)
                    if cid in representation and diagnostic != representation[cid]:
                        raise ArithmeticError("Geometry diagnostics changed between transferred horizons")
                    representation[cid] = diagnostic
        C.log(task, "test_horizon_evaluated", horizon=h, selected_models=len(ids))
    p_id = selected["names"]["P_accuracy"]
    p_spec = metadata[primary][p_id]["source_spec"]
    p_key = f"s_{G.bank_key(p_spec['scaling'], p_spec['cap'])}_p{p_spec['prefix']}"
    paired = G.paired_coordinate_diagnostics(score_arrays[p_key], P.block_sizes(p_spec["prefix"]), p_spec["omega"])
    paired.update(binding=selected["binding"], P_accuracy_id=p_id, score_array_key=p_key,
                  canonical_enclosure_claim=False)
    _write_npz(folder / "predictions.npz", {f"h{h}_{cid}": predictions[str(h)][cid] for h in horizons for cid in ids})
    _write_npz(folder / "test_features.npz", feature_arrays)
    _write_npz(folder / "test_scores.npz", score_arrays)
    C.write_json(folder / "test_feature_diagnostics.json", {"task": task, "models": feature_info,
                 "common_raw_feature_shape": list(X.shape), "shared_incoming_raw_feature_bytes": X.nbytes,
                 "transient_work_memory_measured": False,
                 "feature_storage_is_evaluation_evidence_not_timing_input": True})
    C.write_json(folder / "representation_diagnostics_test.json", {"task": task, "models": representation,
                 "canonical_enclosure_claim": False, "probability_floor_applied": False})
    C.write_json(folder / "paired_coordinate_diagnostics.json", paired)
    if C.task_kind(task) == "hmm":
        import oracle24 as O
        oracle, oracle_arrays = O.evaluate_oracle(X, Y, horizons, predictions)
        for h in horizons:
            for cid in ids:
                if C.fraction(oracle["horizons"][str(h)]["models"][cid]["realized_risk"]["mse_exact"]) != metrics[str(h)][cid]["mse_exact"]:
                    raise ArithmeticError("Oracle companion risk disagrees with operational risk")
        _write_npz(folder / "oracle_arrays.npz", oracle_arrays)
        C.write_json(folder / "oracle.json", oracle)
    else:
        C.write_json(folder / "appliances_block_risks.json", _appliances_block_risks(Y, horizons, predictions, data["test_anchor_ids"]))
    summary = {"schema": "step24-frozen-test-evaluation-v1", "task": task, "horizons": horizons,
               "primary_horizon": selected["primary_horizon"], "selected_ids": ids, "names": selected["names"],
               "metrics": metrics, "replays": replays, "feature_keys": feature_keys, "rows": len(X),
               "reloaded_deployment_costs": reloaded_costs,
               "test_anchor_ids_sha256": C.array_sha(data["test_anchor_ids"]),
               "fit_seal_sha256": fit_sha, "all_six_fit_seal_checked_before_test_access": True,
               "all_horizons_share_fixed_test_anchors": True, "test_outcome_selection": False,
               "canonical_enclosure_claim": False,
               "risk_scope": "Exact dyadic risk of retained binary64 targets and exported full-test-layout forecasts; no population guarantee",
               "risk_evaluation_layout": {"raw_rows": len(X), "raw_columns": X.shape[1],
                                          "geometry_native_batch_size": G.NATIVE_BATCH_SIZE},
               "evaluation_seconds": time.perf_counter() - started,
               "peak_process_rss_kib_including_runtime": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "ECG_or_reserve_access": False}
    C.write_json(folder / "evaluation.json", summary)
    files = outputs + ["test.npz", "data_access_test.json"]
    C.write_json(completion, {"schema": "step24-evaluation-complete-v1", "passed": True, "task": task,
                  "completed_utc": C.now(), "fit_seal_sha256": fit_sha, "process_id": os.getpid(),
                  "files": {name: C.sha(folder / name) for name in files},
                  "model_horizon_pairs": len(ids) * len(horizons), "canonical_enclosure_claim": False})
    C.log(task, "test_evaluation_complete", model_horizon_pairs=len(ids) * len(horizons), elapsed_seconds=time.perf_counter() - started)
    return C.jsonable(summary)


def time_task(task):
    """The timing module owns its separately launched, serial worker."""
    if _EVALUATED_IN_PROCESS:
        raise RuntimeError("Timing must run in a fresh process after evaluation")
    from timing24 import time_task as measure
    return measure(task)


def _policy_module():
    path = C.PARENT / "protocol/policy23.py"
    name = "step24_frozen_policy_for_reporting"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return sys.modules[name]


def _primary_gates(selected, metadata, metrics, timing):
    """Use only the frozen primary P_cost/CS_cost and the two fixed references."""
    h = str(selected["primary_horizon"])
    def observed(cid):
        if cid is None:
            return None
        return {"id": cid, "group": metadata[h][cid]["source_spec"]["group"],
                "test_mse_exact": C.fraction(metrics[h][cid]["mse_exact"]),
                "complete_numeric_bytes": metadata[h][cid]["cost"]["complete_numeric_bytes"],
                "median_warm_batch_nanoseconds_per_row": C.fraction(timing[h][cid]["median_warm_batch_nanoseconds_per_row"])}
    accuracy = observed(selected["names"]["C_accuracy"])
    cost = observed(selected["names"]["C_cost"])
    gate = _policy_module().observed_utility_gate
    results = {name: gate(observed(selected["names"][name]), accuracy, cost) for name in ("P_cost", "CS_cost")}
    for name, result in results.items():
        expected_group = name.split("_")[0]
        if result["candidate_id"] is not None and result["candidate_group"] != expected_group:
            raise ValueError("Primary P/CS gate candidate group changed")
    return results, {"C_accuracy": accuracy, "C_cost": cost}


def finalize_task(task):
    C.check_fit_seal()
    C.task_kind(task)
    folder, fit_sha = C.RUN / task, C.sha(C.RUN / "fit_seal.json")
    evaluation_done = _verify_completion(folder, "evaluation_complete.json", fit_sha, task, ("evaluation.json", "predictions.npz", "test.npz", "data_access_test.json"))
    timing_done = _verify_completion(folder, "timing_complete.json", fit_sha, task, ("timing.json",))
    if evaluation_done["process_id"] == timing_done["process_id"]:
        raise RuntimeError("Evaluation and timing did not use separate processes")
    completion = folder / "finalization_complete.json"
    if completion.exists():
        _verify_completion(folder, completion.name, fit_sha, task, ("results.json", "evaluation_complete.json", "timing_complete.json"))
        C.log(task, "finalization_already_complete")
        return C.read_json(folder / "results.json")
    if (folder / "results.json").exists():
        raise RuntimeError("Unsealed final results already exist; refusing overwrite")
    evaluation, timing = C.read_json(folder / "evaluation.json"), C.read_json(folder / "timing.json")
    selected, metadata = C.read_json(folder / "selected.json"), C.read_json(folder / "selected_model_metadata.json")
    horizons, ids = selected["horizons"], selected["selected_ids"]
    if (evaluation["task"] != task or timing["task"] != task or evaluation["horizons"] != horizons
            or evaluation["selected_ids"] != ids or evaluation["primary_horizon"] != selected["primary_horizon"]
            or set(timing["horizons"]) != {str(h) for h in horizons}):
        raise RuntimeError("Evaluation/timing changed the frozen task roster")
    for h in horizons:
        if set(evaluation["metrics"][str(h)]) != set(ids) or set(timing["horizons"][str(h)]) != set(ids):
            raise RuntimeError("Incomplete model/horizon timing or risk roster")
    gates, references = _primary_gates(selected, metadata, evaluation["metrics"], timing["horizons"])
    rows = {str(h): {cid: {"risk": evaluation["metrics"][str(h)][cid],
                           "cost": metadata[str(h)][cid]["cost"],
                           "timing": timing["horizons"][str(h)][cid],
                           "source_spec": metadata[str(h)][cid]["source_spec"]} for cid in ids} for h in horizons}
    summary = {"schema": "step24-final-task-results-v1", "task": task, "horizons": horizons,
               "primary_horizon": selected["primary_horizon"], "selected_ids": ids,
               "names": selected["names"], "models": rows, "primary_gates": gates,
               "primary_references": references, "fit_seal_sha256": fit_sha,
               "evaluation_complete_sha256": C.sha(folder / "evaluation_complete.json"),
               "timing_complete_sha256": C.sha(folder / "timing_complete.json"),
               "primary_only_utility_screen": True, "secondary_horizons_descriptive": True,
               "canonical_enclosure_claim": False, "post_test_model_selection": False,
               "screen_scope": "Frozen observed operational screen with exact 11/10 risk boundary, strict numeric-byte saving and inclusive median warm latency boundary",
               "cross_task_rule": "P_cost and CS_cost are separate procedures; consistent panel success requires all three gates on every one of five HMM replications and Appliances",
               "ECG_or_reserve_access": False}
    C.write_json(folder / "results.json", summary)
    C.write_json(completion, {"schema": "step24-finalization-complete-v1", "passed": True, "task": task,
                  "completed_utc": C.now(), "fit_seal_sha256": fit_sha, "process_id": os.getpid(),
                  "files": {name: C.sha(folder / name) for name in ("results.json", "evaluation_complete.json", "timing_complete.json")}})
    C.log(task, "task_finalized", primary_gates={name: row["all_three_pass"] for name, row in gates.items()})
    return C.jsonable(summary)
