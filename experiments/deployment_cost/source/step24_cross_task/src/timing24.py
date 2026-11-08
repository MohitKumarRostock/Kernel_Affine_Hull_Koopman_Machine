"""Fresh-process, raw-input deployment timing for the sealed Step 24 panel.

Importing this module does not open a test cache or invoke a model. Timed calls
always enter the complete deployed predictor with raw feature rows. Full-test
and singleton feature replays are diagnostic work performed after all clocks.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
from numbers import Integral
import os
from pathlib import Path
import pickle
import re
import resource
import time

_IMPORT_STARTED_NS = time.perf_counter_ns()
import common24 as C
import numpy as np
import geometry24 as G
import models24 as M
import pipeline24 as P
from threadpoolctl import threadpool_info
_IMPORT_FINISHED_NS = time.perf_counter_ns()

BATCH_ROWS = 512
SINGLE_ROWS = 32
REPEATS = 9
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_TIMING_PROCESS_USED = False


def _bits_equal(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def exact_median(values, denominator=1):
    """The exact median of nine integer nanosecond observations, then divide."""
    values = list(values)
    if (len(values) != REPEATS or any(isinstance(v, (bool, np.bool_))
            or not isinstance(v, Integral) or v < 0 for v in values)):
        raise ValueError("Timing requires nine nonnegative integer nanosecond observations")
    if isinstance(denominator, (bool, np.bool_)) or not isinstance(denominator, Integral) or denominator <= 0:
        raise ValueError("Timing denominator must be a positive integer")
    return Fraction(sorted(map(int, values))[REPEATS // 2], int(denominator))


def combination_order(models, repeat):
    """Odd one-based repeats ascend zero-padded horizon/model ids; even reverse."""
    if isinstance(repeat, (bool, np.bool_)) or not isinstance(repeat, Integral) or not 1 <= repeat <= REPEATS:
        raise ValueError("Repeat must be one of the nine one-based timing repeats")
    combinations = []
    if not isinstance(models, dict) or not models:
        raise ValueError("A nonempty horizon/model roster is required")
    for horizon, candidates in models.items():
        if not isinstance(horizon, str) or not horizon.isdecimal() or str(int(horizon)) != horizon or not 0 < int(horizon) < 1000:
            raise ValueError("Timing horizon keys must be canonical positive decimal strings below 1000")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError("A nonempty candidate roster is required for every horizon")
        for cid in candidates:
            if not isinstance(cid, str) or not _SAFE_ID.fullmatch(cid):
                raise ValueError("Unsafe timing model id")
            combinations.append((horizon, cid))
    return sorted(combinations, key=lambda item: f"h{int(item[0]):03d}_{item[1]}", reverse=repeat % 2 == 0)


def _prediction(value, rows):
    a = np.asarray(value)
    if a.dtype != np.float64 or a.shape != (rows,) or not np.isfinite(a).all():
        raise ArithmeticError("A timed predictor did not return finite scalar binary64 forecasts")
    return a


def measure_batches(models, X, *, predict=None, clock=None, progress=None):
    """Measure the frozen schedule; injectable functions are manufactured seams.

    Return (timing records, cold512 forecasts, first-pass singleton forecasts).
    No cached model-specific features enter predict. Validation and copies occur
    after the clock stops; the one warm-up call is deliberately untimed.
    """
    predict = P.predict if predict is None else predict
    clock = time.perf_counter_ns if clock is None else clock
    X = np.asarray(X)
    if X.dtype != np.float64 or X.ndim != 2 or X.shape[0] != BATCH_ROWS or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("Timing requires exactly the first 512 finite binary64 raw rows")
    order = combination_order(models, 1)
    records = {h: {} for h in models}
    cold = {h: {} for h in models}
    single = {h: {} for h in models}

    def measured(pipe, rows):
        before = clock()
        value = predict(pipe, rows)
        elapsed = clock() - before
        if isinstance(elapsed, (bool, np.bool_)) or not isinstance(elapsed, Integral) or elapsed < 0:
            raise ArithmeticError("Invalid monotonic nanosecond timer observation")
        return int(elapsed), _prediction(value, len(rows))

    for h, cid in order:
        elapsed, value = measured(models[h][cid], X)
        cold[h][cid] = value.copy()
        records[h][cid] = {"cold_batch_nanoseconds": elapsed,
                           "warm_batch_nanoseconds": [], "single_query_nanoseconds": []}
        warmed = _prediction(predict(models[h][cid], X), BATCH_ROWS)
        if not _bits_equal(cold[h][cid], warmed):
            raise ArithmeticError(f"Untimed warm-up changed same-layout forecasts: h{h}/{cid}")
    if progress is not None:
        progress("cold_and_warmup_complete", len(order))

    for repeat in range(1, REPEATS + 1):
        for h, cid in combination_order(models, repeat):
            elapsed, value = measured(models[h][cid], X)
            if not _bits_equal(cold[h][cid], value):
                raise ArithmeticError(f"Warm repeat {repeat} changed same-layout forecasts: h{h}/{cid}")
            records[h][cid]["warm_batch_nanoseconds"].append(elapsed)
        if progress is not None:
            progress("warm_repeat_complete", repeat)

    for repeat in range(1, REPEATS + 1):
        for h, cid in combination_order(models, repeat):
            predictions, observations = np.empty(SINGLE_ROWS, dtype=np.float64), []
            for row in range(SINGLE_ROWS):
                elapsed, value = measured(models[h][cid], X[row:row + 1])
                observations.append(elapsed)
                predictions[row] = value[0]
            if repeat == 1:
                single[h][cid] = predictions.copy()
            elif not _bits_equal(single[h][cid], predictions):
                raise ArithmeticError(f"Singleton repeat {repeat} changed same-layout forecasts: h{h}/{cid}")
            records[h][cid]["single_query_nanoseconds"].append(observations)
        if progress is not None:
            progress("single_query_repeat_complete", repeat)

    for h, cid in order:
        record = records[h][cid]
        totals = [sum(row) for row in record["single_query_nanoseconds"]]
        record.update(
            median_warm_batch_nanoseconds=exact_median(record["warm_batch_nanoseconds"]),
            median_warm_batch_nanoseconds_per_row=exact_median(record["warm_batch_nanoseconds"], BATCH_ROWS),
            single_pass_nanoseconds=totals,
            median_single_query_nanoseconds_per_row=exact_median(totals, SINGLE_ROWS),
            warm_and_cold512_bit_identical=True,
            single_query_repeats_bit_identical=True,
            timed_batch_calls=1 + REPEATS, untimed_warmup_calls=1,
            timed_single_query_calls=REPEATS * SINGLE_ROWS,
            batch_rows=BATCH_ROWS, single_query_rows=SINGLE_ROWS, repeats=REPEATS)
    return records, cold, single


def audit_layout(pipe, reference_prediction, actual_prediction, reference_features,
                 actual_features, target_map, label):
    """Distinguish numeric feature changes from unexplained head changes by row.

    Tiny feature shifts can cross tree thresholds. Such jumps are retained and
    described; an unchanged feature row cannot excuse a changed HGB forecast.
    Ridge/Nystrom must satisfy the already frozen export tolerance regardless
    of whether their upstream features changed.
    """
    a = np.asarray(reference_prediction)
    if a.ndim != 1 or not len(a):
        raise ValueError("A nonempty prediction vector is required")
    a, b = _prediction(a, len(a)), _prediction(actual_prediction, len(a))
    changed = a != b
    changed_bits = a.view(np.uint64) != b.view(np.uint64)
    error = np.abs(a - b)
    result = {"label": label, "rows": len(a), "passed": True,
              "prediction_bit_identical": bool(not np.any(changed_bits)),
              "prediction_numeric_changed_rows": np.flatnonzero(changed).tolist(),
              "prediction_bit_changed_rows": np.flatnonzero(changed_bits).tolist(),
              "max_abs_prediction_difference": float(np.max(error)),
              "max_relative_prediction_difference_with_unit_floor": float(np.max(error / np.maximum(1., np.abs(a)))),
              "features": None, "unchanged_feature_prediction_change_rows": [],
              "tree_jump_rows": [], "fixed_tolerance_audit": None,
              "scope": "Operational raw-input layout comparison; no canonical predictor or information-loss inference"}
    feature_changed = None
    feature_max = None
    if (reference_features is None) != (actual_features is None):
        raise ValueError("Both feature layouts must be available, or both omitted")
    if reference_features is not None:
        fa, fb = np.asarray(reference_features), np.asarray(actual_features)
        if (fa.dtype != np.float64 or fb.dtype != np.float64 or fa.ndim != 2 or fa.shape != fb.shape
                or fa.shape[0] != len(a) or fa.shape[1] < 1 or not np.isfinite(fa).all() or not np.isfinite(fb).all()):
            raise ValueError("Invalid equal-shape finite binary64 feature replay")
        differences = fa != fb
        feature_changed = np.any(differences, axis=1)
        bit_differences = fa.view(np.uint64) != fb.view(np.uint64)
        feature_max = np.max(np.abs(fa - fb), axis=1)
        result["features"] = {
            "shape": list(fa.shape),
            "numeric_changed_rows": np.flatnonzero(feature_changed).tolist(),
            "bit_changed_rows": np.flatnonzero(np.any(bit_differences, axis=1)).tolist(),
            "numeric_changed_coordinates": int(np.count_nonzero(differences)),
            "bit_changed_coordinates": int(np.count_nonzero(bit_differences)),
            "max_abs_difference": float(np.max(feature_max))}
        unchanged_bad = changed & ~feature_changed
        result["unchanged_feature_prediction_change_rows"] = np.flatnonzero(unchanged_bad).tolist()
    mode = pipe.get("pipeline_kind")
    family = pipe["head"]["kind"] if mode in {"geometry", "raw"} else mode
    result["family"] = family
    if family == "hgb":
        if feature_changed is None:
            raise ValueError("HGB batch-layout audit requires both explicit feature matrices")
        if result["unchanged_feature_prediction_change_rows"]:
            raise ArithmeticError(label + ": HGB prediction changed on numerically unchanged feature rows "
                                  + str(result["unchanged_feature_prediction_change_rows"]))
        result["tree_jump_rows"] = [
            {"row": int(i), "reference_prediction": float(a[i]), "actual_prediction": float(b[i]),
             "prediction_difference": float(b[i] - a[i]), "max_abs_feature_difference": float(feature_max[i])}
            for i in np.flatnonzero(changed)]
        result["changed_feature_tree_jumps_permitted_with_explicit_record"] = True
    elif family in {"ridge", "nystrom"}:
        result["fixed_tolerance_audit"] = M.audit_export(a, b, target_map, label)
    elif family == "baseline":
        if not _bits_equal(a, b):
            raise ArithmeticError(label + ": a fixed baseline changed with batch layout")
    else:
        raise ValueError("Unknown frozen deployment family")
    return result


def _require_fresh_process(evaluation_receipt, *, current_pid=None, already_used=False, evaluated_here=False):
    current_pid = os.getpid() if current_pid is None else current_pid
    pid = evaluation_receipt.get("process_id")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("Evaluation receipt has no valid worker process id")
    if pid == current_pid or already_used or evaluated_here:
        raise RuntimeError("Timing requires a fresh worker process, separate from evaluation and earlier timing")


def _require_no_partial_timing(folder):
    names = ("timing.json", "timing.json.pending", "timing_started.json",
             "timing_started.json.pending", "timing_complete.json.pending")
    if any((Path(folder) / name).exists() for name in names):
        raise RuntimeError("Incomplete or uncertain timing artifacts exist; refusing overwrite or automatic rerun")


def _active_study_workers():
    """Inspect runner command lines only; never read another process environment."""
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal() or int(directory.name) == os.getpid():
            continue
        try:
            args = (directory / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for index, arg in enumerate(args[:-1]):
            if arg == os.fsencode(C.ROOT / "src/run24.py") or arg.endswith(b"/step24_cross_task/src/run24.py"):
                stage = args[index + 1].decode("ascii", errors="replace")
                if stage in {"fit", "fit-panel", "evaluate", "time"}:
                    result.append({"process_id": int(directory.name), "stage": stage})
    return result


def _require_serial_worker():
    active = _active_study_workers()
    if active:
        raise RuntimeError("Timing requires no concurrent study fit/evaluation/timing workers: " + str(active))


def _check_loaded_cost(pipe, expected):
    loaded = P.payload_cost(pipe)
    restored = pickle.loads(pickle.dumps(pipe, protocol=5))
    reloaded = P.payload_cost(restored)
    target = expected["complete_numeric_bytes"]
    if loaded["complete_numeric_bytes"] != target or reloaded["complete_numeric_bytes"] != target:
        raise ArithmeticError("Complete numeric deployment storage changed after selected payload reload")
    return {"frozen_complete_numeric_bytes": target,
            "loaded_complete_numeric_bytes": loaded["complete_numeric_bytes"],
            "reloaded_complete_numeric_bytes": reloaded["complete_numeric_bytes"],
            "frozen_serialized_model_bytes": expected["serialized_model_bytes"],
            "loaded_reserialized_model_bytes": loaded["serialized_model_bytes"],
            "reloaded_reserialized_model_bytes": reloaded["serialized_model_bytes"],
            "numeric_storage_stable": True,
            "serialized_length_scope": "Reported separately; pickle memoization can change serialized length without changing numeric payload storage"}


def _feature_fingerprint(pipe):
    if pipe["pipeline_kind"] == "raw":
        return "raw"
    payload = {key: pipe[key] for key in ("geometry", "transform")}
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def _validate_evaluation_arrays(evaluation, selected, data, predictions, features):
    horizons, ids, rows = selected["horizons"], selected["selected_ids"], len(data["test_X"])
    if (evaluation.get("horizons") != horizons or evaluation.get("selected_ids") != ids
            or evaluation.get("primary_horizon") != selected["primary_horizon"]
            or evaluation.get("rows") != rows
            or set(evaluation.get("feature_keys", {})) != set(ids)
            or set(evaluation.get("metrics", {})) != {str(h) for h in horizons}
            or any(set(evaluation["metrics"][str(h)]) != set(ids) for h in horizons)):
        raise RuntimeError("Evaluation changed the frozen selected model/horizon/common-row roster")
    if set(predictions) != {f"h{h}_{cid}" for h in horizons for cid in ids}:
        raise RuntimeError("Committed full-test forecast roster is incomplete or contains extra models")
    for value in predictions.values():
        _prediction(value, rows)
    keys = {key for key in evaluation["feature_keys"].values() if key is not None}
    if set(features) != keys:
        raise RuntimeError("Committed full-test feature roster changed")
    for cid, key in evaluation["feature_keys"].items():
        if key is None:
            continue
        if key != "f_" + cid:
            raise RuntimeError("Full-test feature key does not match its selected candidate")
        a = features[key]
        if a.dtype != np.float64 or a.ndim != 2 or a.shape[0] != rows or not np.isfinite(a).all():
            raise RuntimeError("Invalid committed full-test feature matrix")
    if rows < BATCH_ROWS:
        raise RuntimeError("The fixed test cohort has fewer than 512 timing rows")


def time_task(task):
    """Execute one complete write-once timing worker after the six-task fit seal."""
    global _TIMING_PROCESS_USED
    task_started = time.perf_counter_ns()
    seal = C.check_fit_seal()  # This precedes every test/model/evaluation cache read.
    C.task_kind(task)
    import evaluate24 as E
    folder, fit_sha = C.RUN / task, C.sha(C.RUN / "fit_seal.json")
    evaluation_done = E._verify_completion(folder, "evaluation_complete.json", fit_sha, task,
        ("evaluation.json", "predictions.npz", "test_features.npz", "test.npz", "data_access_test.json"))
    completion = folder / "timing_complete.json"
    if completion.exists():
        done = E._verify_completion(folder, completion.name, fit_sha, task,
                                   ("timing.json", "timing_started.json", "evaluation_complete.json"))
        if done.get("process_id") == evaluation_done["process_id"]:
            raise RuntimeError("Completed timing reused its evaluation worker process")
        C.log(task, "timing_already_complete")
        return C.read_json(folder / "timing.json")
    _require_no_partial_timing(folder)
    _require_fresh_process(evaluation_done, already_used=_TIMING_PROCESS_USED,
                          evaluated_here=E._EVALUATED_IN_PROCESS)
    _require_serial_worker()
    lock = C.RUN / "timing_worker.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("An active or uncertain global timing worker lock already exists") from exc
    with os.fdopen(descriptor, "w") as stream:
        stream.write(f"task={task}\npid={os.getpid()}\nfit_seal_sha256={fit_sha}\n")
    _TIMING_PROCESS_USED = True
    C.write_json(folder / "timing_started.json", {
        "schema": "step24-timing-started-v1", "task": task, "started_utc": C.now(),
        "process_id": os.getpid(), "evaluation_process_id": evaluation_done["process_id"],
        "fit_seal_sha256": fit_sha, "evaluation_complete_sha256": C.sha(folder / "evaluation_complete.json"),
        "resume_policy": "Any partial timing attempt stops; no automatic remeasurement or overwrite"})

    startup = {"timing_module_dependency_import_nanoseconds": _IMPORT_FINISHED_NS - _IMPORT_STARTED_NS}
    before = time.perf_counter_ns()
    G.initialize_runtime()
    startup["geometry_runtime_initialization_nanoseconds"] = time.perf_counter_ns() - before
    pools = threadpool_info()
    if any(pool.get("num_threads") != 1 for pool in pools):
        raise RuntimeError("All native runtime pools must have exactly one thread before timing")
    before = time.perf_counter_ns()
    models = C.read_pickle(folder / "selected_models.pkl")
    startup["selected_models_pickle_load_nanoseconds"] = time.perf_counter_ns() - before
    selected = C.read_json(folder / "selected.json")
    metadata = C.read_json(folder / "selected_model_metadata.json")
    maps = C.read_json(folder / "target_maps.json")
    horizons, ids = E._validate_roster(task, selected, models, metadata, maps)
    evaluation = C.read_json(folder / "evaluation.json")
    if evaluation.get("task") != task or evaluation.get("fit_seal_sha256") != fit_sha:
        raise RuntimeError("Evaluation summary changed its task/fit binding")
    before = time.perf_counter_ns()
    data = E._load_test_cache(task, seal)
    startup["verified_test_cache_load_nanoseconds"] = time.perf_counter_ns() - before
    before = time.perf_counter_ns()
    predictions = E._read_npz(folder / "predictions.npz")
    feature_arrays = E._read_npz(folder / "test_features.npz")
    startup["committed_evaluation_evidence_load_nanoseconds"] = time.perf_counter_ns() - before
    _validate_evaluation_arrays(evaluation, selected, data, predictions, feature_arrays)
    costs = {h: {} for h in models}
    before = time.perf_counter_ns()
    for h, cid in combination_order(models, 1):
        costs[h][cid] = _check_loaded_cost(models[h][cid], metadata[h][cid]["cost"])
    startup["numeric_inventory_and_pickle_replay_nanoseconds"] = time.perf_counter_ns() - before
    X = data["test_X"]
    startup["timing_worker_preparation_before_first_predict_nanoseconds"] = time.perf_counter_ns() - task_started
    startup["scope"] = "Recorded imports/runtime initialization, model loading and verification; interpreter startup before timing-module import is outside this stopwatch"
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    C.log(task, "fresh_timing_started", process_id=os.getpid(), models=len(ids), horizons=horizons)
    timings, cold, singles = measure_batches(models, X[:BATCH_ROWS],
        progress=lambda event, count: C.log(task, event, count=count))
    rss_after_clocks = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _require_serial_worker()

    # Replaying features before the cold call would warm the tested model path.
    # These diagnostic-only caches are created strictly after every timing call.
    diagnostic_started = time.perf_counter_ns()
    layouts = {}
    for h, cid in combination_order(models, 1):
        pipe, fkey = models[h][cid], evaluation["feature_keys"][cid]
        explicit = pipe["pipeline_kind"] in {"geometry", "raw"}
        if explicit != (fkey is not None):
            raise RuntimeError("Explicit-head pipeline lacks its committed full-test features")
        full_prefix = batch_features = single_features = None
        if explicit:
            fingerprint = _feature_fingerprint(pipe)
            if fingerprint not in layouts:
                full_features = P.features(pipe, X)
                batch_features = P.features(pipe, X[:BATCH_ROWS])
                single_features = np.concatenate([P.features(pipe, X[i:i + 1]) for i in range(SINGLE_ROWS)])
                layouts[fingerprint] = (full_features, batch_features, single_features)
            full_features, batch_features, single_features = layouts[fingerprint]
            if not _bits_equal(full_features, feature_arrays[fkey]):
                raise ArithmeticError("Fresh-process full-test feature replay changed its committed same-layout values")
            full_prefix = full_features[:BATCH_ROWS]
        result = timings[h][cid]
        result["full_test_vs_512"] = audit_layout(pipe, predictions[f"h{h}_{cid}"][:BATCH_ROWS], cold[h][cid],
            full_prefix, batch_features, maps[h], f"{task}/h{h}/{cid}: full test versus first512")
        result["batch512_vs_single"] = audit_layout(pipe, cold[h][cid][:SINGLE_ROWS], singles[h][cid],
            None if batch_features is None else batch_features[:SINGLE_ROWS], single_features,
            maps[h], f"{task}/h{h}/{cid}: first512 versus singleton32")
        result["numeric_storage"] = costs[h][cid]
        result["same_layout_full_feature_replay_bit_identical"] = True if explicit else None
        result["timing_feature_output_bytes_512"] = None if batch_features is None else int(batch_features.nbytes)
        if M.numeric_inventory(pipe)["numeric_bytes"] != costs[h][cid]["loaded_complete_numeric_bytes"]:
            raise ArithmeticError("Timed deployed model mutated its retained numeric storage")
    startup["post_timing_feature_and_prediction_diagnostics_nanoseconds"] = time.perf_counter_ns() - diagnostic_started
    # Recheck both sealed fit payloads and every evaluation commit before writing.
    C.check_fit_seal()
    E._verify_completion(folder, "evaluation_complete.json", fit_sha, task,
        ("evaluation.json", "predictions.npz", "test_features.npz", "test.npz", "data_access_test.json"))
    access = C.read_json(folder / "data_access_test.json")
    summary = {
        "schema": "step24-fresh-raw-pipeline-timing-v1", "task": task,
        "horizons": timings, "primary_horizon": selected["primary_horizon"], "selected_ids": ids,
        "fit_seal_sha256": fit_sha, "evaluation_complete_sha256": C.sha(folder / "evaluation_complete.json"),
        "process_id": os.getpid(), "evaluation_process_id": evaluation_done["process_id"],
        "fresh_process": True, "all_six_fit_seal_checked_before_test_access": True,
        "timing_order": {"odd_repeats": [f"h{int(h):03d}_{cid}" for h, cid in combination_order(models, 1)],
                         "even_repeats": [f"h{int(h):03d}_{cid}" for h, cid in combination_order(models, 2)]},
        "timer": "time.perf_counter_ns", "batch_rows": BATCH_ROWS, "single_query_rows": SINGLE_ROWS, "repeats": REPEATS,
        "median_scope": "Exact median of nine integer total nanoseconds divided by 512; no float conversion for utility comparisons",
        "single_query_scope": "Nine passes in fixed first32 anchor order, individually timed calls; descriptive median pass total divided by 32",
        "pipeline_scope": "Raw rows through model-specific input scaling, geometry/kernel encoding, representation, head, target inverse and output map",
        "model_specific_features_used_as_timing_inputs": False,
        "feature_replays_started_after_all_timing_calls": True,
        "shared_incoming_raw_feature_bytes": int(X.nbytes),
        "first512_input_view_bytes": int(X[:BATCH_ROWS].nbytes),
        "common_raw_history_loading_and_construction_nanoseconds": access["raw_test_loading_and_common_history_construction_nanoseconds"],
        "preparation": startup, "threadpools": pools,
        "process_peak_rss_before_clock_kib": rss_before,
        "process_peak_rss_after_clock_kib": rss_after_clocks,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "Linux process high-water RSS includes runtime, all loaded models, common buffers, evaluation evidence and post-clock diagnostics; no per-model transient-work-memory claim",
        "transient_work_memory_measured": False, "concurrent_study_workers_observed": [],
        "worker_wall_nanoseconds": time.perf_counter_ns() - task_started,
        "canonical_enclosure_claim": False, "ECG_or_reserve_access": False}
    C.write_json(folder / "timing.json", summary)
    C.write_json(completion, {
        "schema": "step24-timing-complete-v1", "passed": True, "task": task,
        "completed_utc": C.now(), "fit_seal_sha256": fit_sha, "process_id": os.getpid(),
        "evaluation_process_id": evaluation_done["process_id"],
        "files": {name: C.sha(folder / name) for name in ("timing.json", "timing_started.json", "evaluation_complete.json")},
        "model_horizon_pairs": len(ids) * len(horizons), "canonical_enclosure_claim": False})
    lock.unlink()
    C.log(task, "fresh_timing_complete", model_horizon_pairs=len(ids) * len(horizons),
          worker_wall_seconds=(time.perf_counter_ns() - task_started) / 1e9)
    return C.jsonable(summary)
