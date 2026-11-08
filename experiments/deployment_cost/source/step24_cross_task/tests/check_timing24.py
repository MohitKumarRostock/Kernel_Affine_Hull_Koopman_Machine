"""Manufactured timing schedule, floating drift, and cache/access preflight.

No scientific loader, generator, fitted benchmark model or native geometry is
invoked. The only complete worker execution uses a temporary fake task cache
and pure manufactured predictor functions, with the six-task seal gate mocked.
"""
from __future__ import annotations

from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common24 as C
import evaluate24 as E
import models24 as M
import pipeline24 as P
import timing24 as T
import numpy as np

SOURCES = [Path(__file__).resolve(), *(ROOT / "src" / name for name in
    ("timing24.py", "common24.py", "evaluate24.py", "pipeline24.py", "models24.py", "geometry24.py", "data24.py")),
    C.PARENT / "PROTOCOL.json"]
IMPORT_HASHES = {str(path.relative_to(ROOT.parent)): C.sha(path) for path in SOURCES}


def rejects(function):
    try:
        function()
    except (AssertionError, ValueError, TypeError, ArithmeticError, RuntimeError, FileNotFoundError):
        return True
    raise AssertionError("A manufactured invalid timing case was accepted")


class Clock:
    def __init__(self):
        self.value, self.open, self.durations = 10**18, False, []

    def __call__(self):
        if self.open:
            duration = 2**53 + 7 + 3 * len(self.durations)
            self.durations.append(duration)
            self.value += duration
        self.open = not self.open
        return self.value


def main():
    output = ROOT / "tests/timing_preflight24.json"
    if output.exists():
        raise FileExistsError("Manufactured timing receipts are write-once")
    checks, details, invalid = [], {}, 0
    X = np.arange(512 * 2, dtype=np.float64).reshape(512, 2) / 1024
    models = {h: {cid: {"tag": h + cid, "offset": float(int(h))}
                  for cid in ("z", "a")} for h in ("16", "4", "1")}
    calls, clock = [], Clock()

    def predict(pipe, rows):
        calls.append((pipe["tag"], len(rows), float(rows[0, 0])))
        return (rows[:, 0] + pipe["offset"]).copy()

    records, cold, single = T.measure_batches(models, X, predict=predict, clock=clock)
    order = [(h, cid) for h in ("1", "4", "16") for cid in ("a", "z")]
    assert T.combination_order(models, 1) == order
    assert T.combination_order(models, 2) == list(reversed(order))
    expected_calls = [(h + cid, 512, 0.) for h, cid in order for _ in range(2)]
    for repeat in range(1, 10):
        expected_calls.extend((h + cid, 512, 0.) for h, cid in (order if repeat % 2 else list(reversed(order))))
    for repeat in range(1, 10):
        expected_calls.extend((h + cid, 1, float(X[row, 0]))
            for h, cid in (order if repeat % 2 else list(reversed(order))) for row in range(32))
    assert calls == expected_calls
    assert len(clock.durations) == len(order) * (1 + 9 + 9 * 32)
    for index, (h, cid) in enumerate(order):
        record = records[h][cid]
        assert record["cold_batch_nanoseconds"] == clock.durations[index]
        warm = []
        for repeat in range(1, 10):
            position = (order if repeat % 2 else list(reversed(order))).index((h, cid))
            warm.append(clock.durations[len(order) + (repeat - 1) * len(order) + position])
        assert record["warm_batch_nanoseconds"] == warm
        assert record["median_warm_batch_nanoseconds_per_row"] == Fraction(sorted(warm)[4], 512)
        assert record["median_single_query_nanoseconds_per_row"] == Fraction(sorted(record["single_pass_nanoseconds"])[4], 32)
        assert np.array_equal(cold[h][cid][:32], single[h][cid])
        assert len(record["single_query_nanoseconds"]) == 9 and all(len(row) == 32 for row in record["single_query_nanoseconds"])
    assert T.exact_median([9, 1, 5, 7, 3, 8, 2, 4, 6], 512) == Fraction(5, 512)
    for fn in (lambda: T.exact_median([1] * 8), lambda: T.exact_median([True] * 9),
               lambda: T.exact_median([1.] * 9), lambda: T.exact_median([1] * 9, 0),
               lambda: T.combination_order(models, 0), lambda: T.combination_order({"01": {"a": {}}}, 1)):
        assert rejects(fn); invalid += 1
    details["fake_schedule"] = {"model_horizon_pairs": len(order), "predict_calls": len(calls),
        "clocked_calls": len(clock.durations), "nanoseconds_exceed_2_power_53": True,
        "single_query_raw_observations_per_model": 288, "warm_batch_observations_per_model": 9}
    checks.append("Fake clock/predictor verifies every cold, untimed warm-up, nine alternating 512-row rounds and 288 singleton calls per pair; medians retain exact integers above 2^53")

    for fault in ("warmup", "warm_repeat", "singleton_repeat", "signed_zero"):
        state = {"batches": 0, "singletons": 0}
        def unstable(pipe, rows):
            key = "batches" if len(rows) == 512 else "singletons"
            state[key] += 1
            values = np.zeros(len(rows), dtype=np.float64)
            if ((fault == "warmup" and key == "batches" and state[key] == 2)
                    or (fault == "warm_repeat" and key == "batches" and state[key] == 3)
                    or (fault == "singleton_repeat" and key == "singletons" and state[key] == 33)):
                values[0] = 1e-15
            if fault == "signed_zero" and key == "batches" and state[key] == 2:
                values[0] = -0.
            return values
        assert rejects(lambda: T.measure_batches({"1": {"a": {}}}, X, predict=unstable, clock=Clock())); invalid += 1
    checks.append("Same-layout bit checks reject warm-up drift, warm-repeat drift, singleton repeat drift and signed-zero changes")

    target_map = {"offset": 0., "scale": 1.}
    tree = {"pipeline_kind": "geometry", "head": {"kind": "hgb"}}
    ridge = {"pipeline_kind": "raw", "head": {"kind": "ridge"}}
    base = np.array([1., 2., 3.]); fa = np.array([[0., 1.], [2., 3.], [4., 5.]])
    fb = fa.copy(); fb[0, 0] = 1e-15
    jumped = base.copy(); jumped[0] += .25
    accepted = T.audit_layout(tree, base, jumped, fa, fb, target_map, "manufactured changed-feature tree jump")
    assert accepted["features"]["numeric_changed_rows"] == [0]
    assert accepted["tree_jump_rows"] == [{"row": 0, "reference_prediction": 1., "actual_prediction": 1.25,
        "prediction_difference": .25, "max_abs_feature_difference": 1e-15}]
    unexplained = jumped.copy(); unexplained[1] += .25
    signed_features = fa.copy(); signed_features[0, 0] = -0.
    for fn in (
        lambda: T.audit_layout(tree, base, unexplained, fa, fb, target_map, "one unchanged row changed"),
        lambda: T.audit_layout(tree, base, jumped, fa, fa, target_map, "all features unchanged"),
        lambda: T.audit_layout(tree, base, jumped, fa, signed_features, target_map, "numeric equality despite sign bits"),
        lambda: T.audit_layout(ridge, base, jumped, fa, fb, target_map, "ridge excessive drift"),
        lambda: T.audit_layout({"pipeline_kind": "nystrom"}, base, jumped, None, None, target_map, "kernel excessive drift"),
    ):
        assert rejects(fn); invalid += 1
    small = base + 1e-12
    for pipe, features in ((ridge, fa), ({"pipeline_kind": "nystrom"}, None)):
        audit = T.audit_layout(pipe, base, small, features, features, target_map, "small fixed-tolerance change")
        assert audit["fixed_tolerance_audit"]["export_passed"]
    details["manufactured_tree_jump"] = accepted
    checks.append("Rowwise tree audit permits explicitly recorded changed-feature jumps but rejects changed predictions on any numerically unchanged row; ridge and Nystrom use frozen fidelity tolerances")

    owner = np.arange(10, dtype=np.float64)
    view_payload = {"array": owner[1:]}
    assert rejects(lambda: T._check_loaded_cost(view_payload, P.payload_cost(view_payload))); invalid += 1
    owning = {"array": owner.copy(), "value": np.float64(.25)}
    expected = P.payload_cost(owning); expected["serialized_model_bytes"] += 11
    stable = T._check_loaded_cost(owning, expected)
    assert stable["numeric_storage_stable"]
    assert stable["frozen_serialized_model_bytes"] != stable["loaded_reserialized_model_bytes"]
    details["pickle_numeric_storage"] = stable
    checks.append("Numeric inventory rejects owner bytes lost on serialization, requires frozen/loaded/reloaded byte equality, and reports serializer length changes separately")

    for kwargs in ({"current_pid": 42}, {"current_pid": 43, "already_used": True},
                   {"current_pid": 43, "evaluated_here": True}):
        assert rejects(lambda kwargs=kwargs: T._require_fresh_process({"process_id": 42}, **kwargs)); invalid += 1
    T._require_fresh_process({"process_id": 42}, current_pid=43)
    with tempfile.TemporaryDirectory(prefix="step24_timing_guards_") as temporary:
        folder = Path(temporary)
        for name in ("timing.json", "timing.json.pending", "timing_started.json", "timing_started.json.pending", "timing_complete.json.pending"):
            path = folder / name; path.write_text("manufactured incomplete state")
            assert rejects(lambda: T._require_no_partial_timing(folder)); invalid += 1
            path.unlink()
        T._require_no_partial_timing(folder)
    checks.append("Fresh-worker guard rejects evaluation-process reuse and prior timing in-process; each unfinished timing artifact blocks automatic rerun")

    # A complete temporary worker checks commit binding, post-clock diagnostics,
    # completed-result reuse and corruption refusal without invoking real gates.
    with tempfile.TemporaryDirectory(prefix="step24_timing_fake_worker_") as temporary:
        run = Path(temporary); folder = run / "hmm_0"; folder.mkdir()
        C.write_json(run / "fit_seal.json", {"manufactured": True})
        fit_sha = C.sha(run / "fit_seal.json")
        fixture_X = np.arange(513 * 2, dtype=np.float64).reshape(513, 2) / 2048
        fixture_data = {"test_X": fixture_X, "test_Y": np.zeros((513, 1)), "test_anchor_ids": np.arange(513),
                        "horizons": [1], "primary_column": 0}
        fixture_pipe = {"pipeline_kind": "raw", "head": {"kind": "ridge", "coef": np.array([.1, 0.]),
                        "intercept": np.float64(.25), "output_bounds": np.array([-1., 1.])}, "task_kind": "hmm"}
        fixture_models = {"1": {"fake": fixture_pipe}}
        selected = {"horizons": [1], "primary_horizon": 1, "selected_ids": ["fake"]}
        evaluation = {"task": "hmm_0", **selected, "rows": 513, "fit_seal_sha256": fit_sha,
                      "feature_keys": {"fake": "f_fake"}, "metrics": {"1": {"fake": {}}}}
        for name, content in (("selected.json", selected), ("evaluation.json", evaluation),
                ("selected_model_metadata.json", {"1": {"fake": {"cost": P.payload_cost(fixture_pipe)}}}),
                ("target_maps.json", {"1": target_map}),
                ("data_access_test.json", {"raw_test_loading_and_common_history_construction_nanoseconds": 123})):
            C.write_json(folder / name, content)
        C.write_pickle(folder / "selected_models.pkl", fixture_models)
        E._write_npz(folder / "test.npz", {key: fixture_data[key] for key in ("test_X", "test_Y", "test_anchor_ids")})
        E._write_npz(folder / "predictions.npz", {"h1_fake": fixture_X[:, 0] * .1 + .25})
        E._write_npz(folder / "test_features.npz", {"f_fake": fixture_X.copy()})
        committed = ("evaluation.json", "test.npz", "predictions.npz", "test_features.npz", "data_access_test.json")
        C.write_json(folder / "evaluation_complete.json", {"passed": True, "task": "hmm_0",
            "fit_seal_sha256": fit_sha, "process_id": os.getpid() + 100000,
            "files": {name: C.sha(folder / name) for name in committed}})
        events = []
        def fake_predict(pipe, rows):
            events.append("predict")
            return rows[:, 0] * .1 + .25
        def fake_features(pipe, rows):
            events.append("features")
            return rows.copy()
        with ExitStack() as stack:
            stack.enter_context(patch.object(C, "RUN", run))
            gate = stack.enter_context(patch.object(C, "check_fit_seal", return_value={"manufactured": True}))
            stack.enter_context(patch.object(C, "log", lambda *args, **kwargs: None))
            stack.enter_context(patch.object(E, "_EVALUATED_IN_PROCESS", False))
            stack.enter_context(patch.object(E, "_validate_roster", return_value=([1], ["fake"])))
            loader = stack.enter_context(patch.object(E, "_load_test_cache", return_value=fixture_data))
            stack.enter_context(patch.object(T, "_TIMING_PROCESS_USED", False))
            stack.enter_context(patch.object(T, "_require_serial_worker", lambda: None))
            stack.enter_context(patch.object(T, "threadpool_info", return_value=[]))
            stack.enter_context(patch.object(T.G, "initialize_runtime", lambda: None))
            stack.enter_context(patch.object(P, "predict", fake_predict))
            stack.enter_context(patch.object(P, "features", fake_features))
            result = T.time_task("hmm_0")
            assert events[:299] == ["predict"] * 299 and events[299:] == ["features"] * 34
            assert gate.call_count == 2 and loader.call_count == 1
            assert not (run / "timing_worker.lock").exists()
            assert result["horizons"]["1"]["fake"]["full_test_vs_512"]["prediction_bit_identical"]
            assert C.fraction(result["horizons"]["1"]["fake"]["median_warm_batch_nanoseconds_per_row"]) >= 0
            saved_bytes = (folder / "timing.json").read_bytes()
            assert T.time_task("hmm_0") == result
            assert loader.call_count == 1 and len(events) == 333
            (folder / "timing.json").write_bytes(saved_bytes + b" ")
            assert rejects(lambda: T.time_task("hmm_0")); invalid += 1
            assert loader.call_count == 1 and len(events) == 333
            (folder / "timing.json").write_bytes(saved_bytes)
            with patch.object(C, "check_fit_seal", side_effect=RuntimeError("manufactured unsealed fit gate")):
                assert rejects(lambda: T.time_task("hmm_0")); invalid += 1
            assert loader.call_count == 1 and len(events) == 333
        details["temporary_worker"] = {"predict_calls": 299, "feature_replay_calls_after_all_clocks": 34,
            "test_cache_loads": 1, "valid_completed_reuse": True, "corrupted_completion_rejected": True,
            "unsealed_fit_access_rejected_before_load": True}
    checks.append("Temporary fake worker seals timing outputs, defers all model feature replays until clocks finish, reuses verified completed timing without prediction, and refuses tampered completion or unsealed test access")

    final_hashes = {str(path.relative_to(ROOT.parent)): C.sha(path) for path in SOURCES}
    assert final_hashes == IMPORT_HASHES, "Preflight source changed while checks were running"
    receipt = {"schema": "step24-manufactured-timing-preflight-v1", "passed": True, "created_utc": C.now(),
        "source_sha256": final_hashes, "checks": checks, "details": details, "rejected_invalid_cases": invalid,
        "benchmark_trajectories_generated": 0, "benchmark_model_fits": 0, "source_csv_rows_parsed": 0,
        "scientific_test_features_or_models_accessed": False, "native_geometry_invoked": False,
        "ECG_or_reserve_access": False}
    C.write_json(output, receipt)
    print(json.dumps({"passed": True, "receipt": str(output), "checks": len(checks), "rejected_invalid_cases": invalid}))


if __name__ == "__main__":
    main()
