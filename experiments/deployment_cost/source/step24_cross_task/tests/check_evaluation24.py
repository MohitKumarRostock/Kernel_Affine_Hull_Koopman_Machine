"""Manufactured evaluation, cohort, cache, replay and fixed-reference checks.

No production test loader, benchmark RNG, CSV numeric parser or fitting routine
is invoked. The complete evaluator is exercised on five hand-authored HMM
histories, artificial encoder scores and hand-authored readout parameters.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import common24 as C
import data24 as D
import evaluate24 as E
import geometry24 as G
import models24 as M
import pipeline24 as P
import oracle24 as O


def rejects(call):
    try:
        call()
    except (ArithmeticError, AssertionError, FileExistsError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        return True
    return False


def manufactured_data():
    symbols = np.asarray([-1., -.5, .5, 1.])
    indices = np.asarray([[0, 3] * 16, [0, 1, 2, 3] * 8, [0] * 16 + [3] * 16,
                          [3, 2, 1, 0] * 8, [1] * 31 + [2]], dtype=np.int64)
    X = symbols[indices]
    Y = np.asarray([[-1., .5, 1.], [.5, -1., -.5], [1., -.5, .5], [-.5, 1., -1.], [1., .5, -.5]])
    return {"test_X": X, "test_Y": Y, "test_anchor_ids": np.arange(3_000_000, 3_000_005, dtype=np.int64),
            "horizons": [1, 4, 16], "primary_column": 1,
            "metadata": {"manufactured_only": True, "actual_benchmark_trajectories_generated": 0,
                         "actual_source_csv_rows_parsed": 0}}


def head(family, dimension):
    if family == "ridge":
        coef = np.zeros(dimension, dtype=np.float64)
        coef[0], coef[-1] = .125, -.25
        return {"kind": "ridge", "coef": coef, "intercept": np.float64(.0625), "output_bounds": np.array([-1., 1.])}
    # This handcrafted constant tree ensemble exercises both independent head
    # paths without fitting an estimator or accessing any source observation.
    return {"kind": "hgb", "baseline": np.float64(.125), "trees": (),
            "f_idx_map": np.zeros(dimension, dtype=np.uint32),
            "known_cat_bitsets": np.empty((0, 8), dtype=np.uint32),
            "target_offset": np.float64(0), "target_scale": np.float64(1),
            "output_bounds": np.array([-1., 1.])}


def manufactured_roster(data):
    specs, names = {}, {}
    binding = {"scaling": "standard", "cap": 2, "prefix": 2, "omega": 4.}
    for view in ("P", "CS", "LR0", "LRM", "LW"):
        for family in ("ridge", "hgb"):
            cid = view.lower() + "_" + family
            spec = {"id": cid, "index": len(specs), "view": view, "family": family,
                    "group": view if view in {"P", "CS"} else "MATCHED_DIAGNOSTIC", **binding}
            specs[cid] = spec
            names[f"matched_{view}_{family}"] = cid
            if view in {"P", "CS"}:
                names[f"{view}_{family}"] = cid
    for family in ("ridge", "hgb"):
        cid = "raw_" + family
        specs[cid] = {"id": cid, "index": len(specs), "group": "RAW", "view": "RAW", "family": family}
        names["RAW_" + family] = cid
    specs["nystrom_ridge"] = {"id": "nystrom_ridge", "index": len(specs), "group": "NYSTROM", "view": "NYSTROM", "family": "ridge"}
    names["NYSTROM_ridge"] = "nystrom_ridge"
    for baseline in ("persistence", "training_mean"):
        cid = "b_" + baseline
        specs[cid] = {"id": cid, "index": len(specs), "group": "UNFITTED", "view": "BASELINE", "family": "unfitted", "baseline_name": baseline}
        names[baseline] = cid
    names.update(C_accuracy="raw_ridge", C_cost="b_training_mean", P_accuracy="p_ridge", P_cost="p_ridge", CS_accuracy="cs_ridge", CS_cost=None)
    selected = {"horizons": data["horizons"], "primary_horizon": 4, "selected_ids": list(specs),
                "names": names, "binding": binding,
                "selectors": {key: names[key] for key in ("C_accuracy", "C_cost", "P_accuracy", "P_cost", "CS_accuracy", "CS_cost")},
                "source_protocol_sha256": C.sha(C.PARENT / "PROTOCOL.json")}
    models, metadata, maps = {}, {}, {}
    for h in data["horizons"]:
        key = str(h)
        models[key], metadata[key], maps[key] = {}, {}, {"offset": 0., "scale": 1.}
        for cid, spec in specs.items():
            group = spec["group"]
            if group in {"P", "CS", "MATCHED_DIAGNOSTIC"}:
                pipe = {"pipeline_kind": "geometry", "task_kind": "hmm",
                        "geometry": {"block_sizes": np.asarray([2], dtype=np.int64), "score_batch_size": np.int64(1000)},
                        "transform": G.view_payload(spec["view"], [2], binding["omega"]), "head": head(spec["family"], 2)}
            elif group == "RAW":
                pipe = {"pipeline_kind": "raw", "task_kind": "hmm", "head": head(spec["family"], 32)}
            elif group == "NYSTROM":
                pipe = {"pipeline_kind": "nystrom", "task_kind": "hmm", "model": {
                    "kind": "nystrom", "scaler": {"offset": np.zeros(32), "divisor": np.ones(32), "constant_mask": np.zeros(32, dtype=bool)},
                    "landmarks": data["test_X"][:2].copy(), "gamma": np.float64(.125), "coef": np.asarray([.125, -.25]),
                    "intercept": np.float64(.0625), "output_bounds": np.array([-1., 1.])}}
            else:
                pipe = P.baseline_pipeline(spec["baseline_name"], "hmm", h, .0625)
            models[key][cid] = pipe
            metadata[key][cid] = {"source_spec": deepcopy(spec), "cost": P.payload_cost(pipe)}
    return selected, models, metadata, maps


def main():
    out = PROJECT / "tests/evaluation_preflight24.json"
    if out.exists():
        raise FileExistsError("The manufactured evaluation receipt is write-once")
    data = manufactured_data()
    contract = {"horizons": [1, 4, 16], "primary_horizon": 4, "rows": 5, "dimension": 32,
                "anchor_ids": data["test_anchor_ids"]}
    selected, models, metadata, maps = manufactured_roster(data)
    checks, rejected = [], []
    old_evaluated = E._EVALUATED_IN_PROCESS
    with ExitStack() as stack:
        # A production loader/fitter call is a failing manufactured test.
        forbidden = AssertionError("Production benchmark access/fitting is forbidden in this preflight")
        stack.enter_context(patch.object(D, "load_test", side_effect=forbidden))
        stack.enter_context(patch.object(D, "load_development", side_effect=forbidden))
        stack.enter_context(patch.object(G, "fit_banks", side_effect=forbidden))
        stack.enter_context(patch.object(M, "fit_head", side_effect=forbidden))
        stack.enter_context(patch.object(E, "_task_contract", return_value=contract))
        E._validate_test_data("hmm_0", data)
        invalid = []
        for label, key, value in [
            ("missing target horizon", "test_Y", data["test_Y"][:, :2]),
            ("wrong feature dimension", "test_X", data["test_X"][:, :31]),
            ("reordered anchors", "test_anchor_ids", data["test_anchor_ids"][::-1]),
            ("float anchor ids", "test_anchor_ids", data["test_anchor_ids"].astype(float)),
            ("horizon order", "horizons", [4, 1, 16]),
            ("wrong primary column", "primary_column", 0),
            ("non-binary64 targets", "test_Y", data["test_Y"].astype(np.float32)),
            ("non-HMM feature", "test_X", np.zeros_like(data["test_X"])),
            ("nonfinite feature", "test_X", np.full_like(data["test_X"], np.nan)),
        ]:
            invalid.append((label, lambda key=key, value=value: E._validate_test_data("hmm_0", {**data, key: value})))
        E._validate_roster("hmm_0", selected, models, metadata, maps)
        for label, change in [
            ("duplicated selected id", lambda s, m, d, t: s["selected_ids"].append(s["selected_ids"][0])),
            ("omitted transferred model", lambda s, m, d, t: m["16"].pop("raw_ridge")),
            ("extra transferred model", lambda s, m, d, t: m["16"].update(extra=m["16"]["raw_ridge"])),
            ("missing horizon target map", lambda s, m, d, t: t.pop("1")),
            ("changed source specification", lambda s, m, d, t: d["16"]["p_ridge"]["source_spec"].update(omega=.5)),
            ("missing mandatory baseline", lambda s, m, d, t: s["names"].update(persistence=None)),
            ("changed selector reference", lambda s, m, d, t: s["selectors"].update(C_cost="raw_ridge")),
            ("changed matched geometry", lambda s, m, d, t: s["binding"].update(cap=8)),
            ("changed model head family", lambda s, m, d, t: m["1"]["p_ridge"]["head"].update(kind="hgb")),
        ]:
            s, m, d, t = deepcopy((selected, models, metadata, maps))
            change(s, m, d, t)
            invalid.append((label, lambda s=s, m=m, d=d, t=t: E._validate_roster("hmm_0", s, m, d, t)))
        for label, call in invalid:
            assert rejects(call), label
            rejected.append(label)
        checks.append("Common finite binary64 test cohort, exact anchor order, three-horizon maps and unchanged candidate/representation roster enforced")

        y = np.asarray([1., .5, -.5, -1., .5])
        prediction = np.asarray([.125, np.nextafter(.5, 1.), -.25, -.875, .0625])
        risk = E._risk_and_replay(y, prediction)
        expected = sum(((Fraction.from_float(float(a)) - Fraction.from_float(float(b))) ** 2 for a, b in zip(y, prediction)), Fraction()) / len(y)
        assert risk["mse_exact"] == expected
        F = np.arange(10, dtype=np.float64).reshape(5, 2)
        altered = prediction.copy(); altered[0] = np.nextafter(altered[0], 1.)
        # The fixed tolerance permits this ULP, while unchanged-feature HGB
        # requires an exact replay and must refuse it.
        assert M.audit_export(prediction, altered, maps["4"])["export_passed"]
        assert rejects(lambda: E._prediction_replay(models["4"]["p_hgb"], prediction, altered, F, F.copy(), maps["4"], "manufactured HGB"))
        rejected.append("changed HGB bits on unchanged features even within export tolerance")
        ridge_replay = E._prediction_replay(models["4"]["p_ridge"], prediction, altered, F, F.copy(), maps["4"], "manufactured ridge")
        assert ridge_replay["export_passed"] and not ridge_replay["prediction_bits_identical"]
        changedF = F.copy(); changedF[0, 0] = np.nextafter(0., 1.)
        assert rejects(lambda: E._prediction_replay(models["4"]["p_ridge"], prediction, prediction, F, changedF, maps["4"], "changed feature"))
        rejected.append("changed features in same-layout independent replay")
        assert not E._bits_equal(np.array([0.]), np.array([-0.]))
        checks.append("Independent rational MSE replay and fixed export tolerance checked; same-feature HGB requires bit identity, including signed zero")

        real_rows = 26 * 144 + 24
        real_Y = np.arange(real_rows * 3, dtype=np.float64).reshape(real_rows, 3) / 8
        real_predictions = {str(h): {"p": real_Y[:, col] + .125, "q": np.zeros(real_rows)} for col, h in enumerate([1, 6, 36])}
        blocks = E._appliances_block_risks(real_Y, [1, 6, 36], real_predictions, np.arange(15931, 15931 + real_rows))
        assert len(blocks["blocks"]) == 27 and [b["rows"] for b in blocks["blocks"]] == [144] * 26 + [24]
        for col, h in enumerate([1, 6, 36]):
            for cid, prediction in real_predictions[str(h)].items():
                weighted = sum((b["metrics"][str(h)][cid]["mse_exact"] * b["rows"] for b in blocks["blocks"]), Fraction()) / real_rows
                assert weighted == C.exact_mse(real_Y[:, col], prediction)
        assert rejects(lambda: E._appliances_block_risks(real_Y[:-1], [1, 6, 36], real_predictions, np.arange(real_rows - 1)))
        rejected.append("incomplete Appliances block roster")
        checks.append("All 26 complete 144-origin blocks and final 24 origins retain every model/horizon and reconstruct exact full-cohort risk")

        gate_selected = {"primary_horizon": 4, "names": {"C_accuracy": "a", "C_cost": "c", "P_cost": "p", "CS_cost": None}}
        gate_meta = {"4": {cid: {"source_spec": {"group": group}, "cost": {"complete_numeric_bytes": size}}
                            for cid, group, size in [("a", "RAW", 100), ("c", "UNFITTED", 20), ("p", "P", 19)]}}
        gate_risks = {"4": {cid: {"mse_exact": value} for cid, value in [("a", Fraction(10)), ("c", Fraction(50)), ("p", Fraction(11))]}}
        gate_timing = {"4": {cid: {"median_warm_batch_nanoseconds_per_row": Fraction(1001, 512)} for cid in ("a", "c", "p")}}
        gates, refs = E._primary_gates(gate_selected, gate_meta, gate_risks, gate_timing)
        assert gates["P_cost"]["all_three_pass"] and not gates["P_cost"]["cost_reference_accuracy_pass"]
        assert gates["P_cost"]["accuracy_reference_id"] == "a" and gates["P_cost"]["cost_reference_id"] == "c"
        assert refs["C_cost"]["test_mse_exact"] == 50 and gates["CS_cost"]["no_qualifying_candidate"] and not gates["CS_cost"]["all_three_pass"]
        gate_meta["4"]["p"]["cost"]["complete_numeric_bytes"] = 20
        assert not E._primary_gates(gate_selected, gate_meta, gate_risks, gate_timing)[0]["P_cost"]["memory_pass"]
        gate_meta["4"]["p"]["source_spec"]["group"] = "CS"
        assert rejects(lambda: E._primary_gates(gate_selected, gate_meta, gate_risks, gate_timing))
        rejected.append("P gate replaced by a CS candidate")
        checks.append("Exact primary utility boundaries retain both frozen references even when C_cost fails accuracy; absent CS stays failed and memory saving is strict")

        with TemporaryDirectory(prefix="step24_manufactured_evaluation_") as tmp:
            run = Path(tmp)
            folder = run / "hmm_0"; folder.mkdir()
            stack.enter_context(patch.object(C, "RUN", run))
            C.write_json(run / "fit_seal.json", {"manufactured_only": True})
            seal = {"protocol_sha256": C.sha(C.PARENT / "PROTOCOL.json"), "manufactured_only": True}
            with patch.object(C, "check_fit_seal", side_effect=RuntimeError("manufactured missing six-task seal")):
                assert rejects(lambda: E._load_test_cache("hmm_0"))
                assert not (folder / "test.npz").exists()
            rejected.append("test access before full fit seal")
            stack.enter_context(patch.object(C, "check_fit_seal", return_value=seal))
            with patch.object(D, "load_test", return_value=deepcopy(data)) as fake_loader:
                cache = E._load_test_cache("hmm_0", seal)
                assert fake_loader.call_count == 1
            assert E._bits_equal(cache["test_X"], data["test_X"])
            reused = E._load_test_cache("hmm_0", seal)
            assert E._bits_equal(reused["test_Y"], data["test_Y"])
            original_seal = (run / "fit_seal.json").read_bytes()
            (run / "fit_seal.json").write_text('{"manufactured_changed":true}\n')
            assert rejects(lambda: E._load_test_cache("hmm_0", seal))
            (run / "fit_seal.json").write_bytes(original_seal)
            rejected.append("test cache bound to a different fit seal digest")
            stored_bytes = (folder / "test.npz").read_bytes()
            with (folder / "test.npz").open("ab") as stream:
                stream.write(b"manufactured altered cache")
            assert rejects(lambda: E._load_test_cache("hmm_0", seal))
            (folder / "test.npz").write_bytes(stored_bytes)
            rejected.append("changed cached test artifact")
            receipt_bytes = (folder / "data_access_test.json").read_bytes()
            receipt = C.read_json(folder / "data_access_test.json")
            receipt["array_sha256"]["test_X"] = "0" * 64
            (folder / "data_access_test.json").write_text(json.dumps(receipt))
            assert rejects(lambda: E._load_test_cache("hmm_0", seal))
            (folder / "data_access_test.json").write_bytes(receipt_bytes)
            rejected.append("changed per-array cache receipt")
            assert rejects(lambda: E._write_npz(folder / "test.npz", {"x": np.ones(1)}))
            rejected.append("overwrite existing test artifact")
            checks.append("Test cache refuses pre-seal access, changed seal/file/array hashes and overwrite; intact cache reuses without invoking the loader")

            for name, value in [("selected.json", selected), ("selected_model_metadata.json", metadata), ("target_maps.json", maps)]:
                C.write_json(folder / name, value)
            C.write_pickle(folder / "selected_models.pkl", models)
            def manufactured_scores(payload, X):
                first = (np.asarray(X)[:, -1] + 1.) / 2.
                return np.column_stack([first, 1. - first])
            with patch.object(G, "initialize_runtime", return_value=None), patch.object(G, "score_compact", side_effect=manufactured_scores), patch.object(P, "predict", wraps=P.predict) as prediction_calls:
                evaluation = E.evaluate_task("hmm_0")
            assert prediction_calls.call_count == 2 * len(selected["selected_ids"]) * 3
            assert evaluation["rows"] == 5 and evaluation["horizons"] == [1, 4, 16]
            assert len(E._read_npz(folder / "predictions.npz")) == 3 * len(selected["selected_ids"])
            assert set(evaluation["metrics"]) == {"1", "4", "16"}
            oracle = C.read_json(folder / "oracle.json")
            assert oracle["rows"] == 5
            fit_sha = C.sha(run / "fit_seal.json")
            E._verify_completion(folder, "evaluation_complete.json", fit_sha, "hmm_0", ("evaluation.json", "oracle.json"))
            assert E.evaluate_task("hmm_0") == evaluation
            assert rejects(lambda: E.time_task("hmm_0"))
            rejected.append("timing launched in the evaluation process")
            checks.append("Complete manufactured evaluation executes all 15 selected models at all three horizons, independent readouts, exact risks, coordinate evidence and rational oracle without fitting")

            timing = {"task": "hmm_0", "horizons": {str(h): {cid: {"median_warm_batch_nanoseconds_per_row": Fraction(1001 + j, 512)}
                        for j, cid in enumerate(selected["selected_ids"])} for h in [1, 4, 16]}}
            C.write_json(folder / "timing.json", timing)
            C.write_json(folder / "timing_complete.json", {"passed": True, "task": "hmm_0", "fit_seal_sha256": fit_sha,
                         "process_id": os.getpid() + 1, "files": {"timing.json": C.sha(folder / "timing.json")}})
            final = E.finalize_task("hmm_0")
            assert final["primary_references"]["C_accuracy"]["id"] == selected["names"]["C_accuracy"]
            assert final["primary_references"]["C_cost"]["id"] == selected["names"]["C_cost"]
            assert final["primary_gates"]["CS_cost"]["no_qualifying_candidate"]
            assert final["primary_only_utility_screen"] is True
            assert E.finalize_task("hmm_0") == final
            E._verify_completion(folder, "finalization_complete.json", fit_sha, "hmm_0", ("results.json",))
            with (folder / "predictions.npz").open("ab") as stream:
                stream.write(b"manufactured altered evaluated artifact")
            assert rejects(lambda: E.evaluate_task("hmm_0"))
            assert rejects(lambda: E.finalize_task("hmm_0"))
            rejected.append("completed evaluation or finalization with changed prediction artifact")
            assert rejects(lambda: E._path_in(folder, "../outside"))
            rejected.append("output manifest path escape")
            checks.append("Finalization consumes only sealed risk/cost/timing records, records both fixed reference risks, and rejects changed completed evaluation artifacts")
    E._EVALUATED_IN_PROCESS = old_evaluated
    paths = [Path(module.__file__).resolve() for module in (C, D, E, G, M, P, O)]
    paths += [Path(__file__).resolve(), C.PARENT / "PROTOCOL.json", C.PARENT / "protocol/policy23.py",
              C.PARENT / "controlled/hmm23.py", C.PARENT / "controlled/hmm_spec23.json", C.PARENT / "data/appliances_protocol23.json"]
    receipt = {"schema": "step24-manufactured-evaluation-preflight-v1", "passed": True, "completed_utc": C.now(),
               "checks": checks, "check_groups": len(checks), "rejected_requests": rejected,
               "manufactured_test_histories": 5, "manufactured_selected_models": len(selected["selected_ids"]),
               "manufactured_model_horizon_pairs": 3 * len(selected["selected_ids"]),
               "actual_benchmark_trajectories_generated": 0, "actual_source_csv_rows_parsed": 0,
               "production_test_loader_invocations": 0, "model_fits": 0, "ECG_or_reserve_access": False,
               "timing_engine_checks": "Separate timing24 manufactured preflight; no benchmark clock panel runs here",
               "source_sha256": {str(path.relative_to(C.BASE)): C.sha(path) for path in paths}}
    C.write_json(out, receipt)
    print(json.dumps({"passed": True, "check_groups": len(checks), "rejections": len(rejected), "receipt": str(out)}))


if __name__ == "__main__":
    main()
