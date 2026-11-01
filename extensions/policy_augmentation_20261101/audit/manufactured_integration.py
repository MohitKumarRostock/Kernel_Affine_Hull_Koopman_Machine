#!/usr/bin/env python3
"""Manufactured-data integration of fitting, C policies and target-free queries.

No production generator or study seed is used.  Numerical budgets are reduced
only inside a temporary fixture root, retaining all three scalers, five alpha
values, five departure caps, fifteen extra-RBF maps and fifty output arrays.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import pipeline as p


def main():
    source_sha = p.digest(ROOT / "code/extra_rbf.py")
    config = copy.deepcopy(p.config(ROOT))
    config["manufactured_fixture"] = True
    config["model"].update(blocks=[2], bank_row_cap=64, max_references=12,
                           raw_rbf_landmarks=16, raw_hgb_max_iter=8,
                           raw_hgb_learning_rates=[0.1], raw_hgb_max_leaf_nodes=[7])
    assert len(config["model"]["alpha_grid"]) == 5
    rng = np.random.default_rng(70411893)
    ax, bx, cx, dx = [rng.normal(size=(n, 12)) for n in (64, 24, 20, 17)]
    coefficient = rng.normal(size=(12, 3)) * 0.2
    def target(x):
        nonlinearity = np.column_stack((np.sin(x[:, 0]), np.tanh(x[:, 1]), x[:, 2] * x[:, 3]))
        return x @ coefficient + 0.3 * nonlinearity + np.array([.1, -.2, .05])
    ay, by, cy = [target(x) for x in (ax, bx, cx)]
    loss_scale = np.maximum(np.var(ay, axis=0, ddof=0), config["model"]["target_variance_floor"])
    arrays = {"A_X": ax, "A_Y": ay, "A_groups": np.repeat(np.arange(16), 4),
              "B_X": bx, "B_Y": by, "C_X": cx, "C_Y": cy, "loss_scale": loss_scale}
    task = {"task_id": "manufactured_policy_integration", "input": "prepared/abc.npz", "fit_seed": 53912417}
    temporary = Path(tempfile.mkdtemp(prefix="kahm-policy-integration-"))
    p.write_json(temporary / "config.json", config)
    p.write_json(temporary / "freeze.json", {"status": "manufactured-fixture-only"})
    p.write_npz(temporary / task["input"], arrays)
    fit = p.worker_fit(temporary, task)
    result_dir = temporary / "results" / task["task_id"]
    summary = json.loads((result_dir / "fit_summary.json").read_text())
    with np.load(result_dir / "calibration_predictions.npz", allow_pickle=False) as saved:
        calibration = {key: saved[key] for key in saved.files}
    assert set(calibration) == p.expected_keys() and len(calibration) == 50
    policy_checks = {}
    for name, order in config["policy_orders"].items():
        # Compute loss/argmin without using choose_policy or stored scores.
        scores = []
        for identity in order:
            family, method = identity.split("__", 1)
            key = f"{family}__calibrated__{method}"
            scores.append(float(np.mean(np.mean((cy - calibration[key]) ** 2, axis=0) / loss_scale)))
        chosen = order[int(np.argmin(scores))]
        assert summary["policies"][name]["selected_identity"] == chosen
        stored = [r["score"] for r in summary["policies"][name]["candidates"]]
        np.testing.assert_array_equal(stored, scores)
        for mode in ("calibrated", "eta1"):
            family, method = chosen.split("__", 1)
            actual = "gamma1" if family == "legacy" and mode == "eta1" else mode
            np.testing.assert_array_equal(calibration[f"policy__{mode}__{name}"],
                                          calibration[f"{family}__{actual}__{method}"])
        policy_checks[name] = {"choice": chosen, "selected_C_score": min(scores)}
    assert policy_checks["P1"]["selected_C_score"] <= policy_checks["P0"]["selected_C_score"]
    assert policy_checks["P0plus"]["selected_C_score"] <= policy_checks["P0"]["selected_C_score"]

    state_path = result_dir / "state.pkl.gz"
    state = pickle.loads(gzip.decompress(state_path.read_bytes()))
    # Required pickle roundtrip and prediction without a target argument.
    roundtrip = pickle.loads(pickle.dumps(state, protocol=5))
    from preprocessing_model import fd
    assert fd.state_hash(roundtrip) == fd.state_hash(state)
    before = fd.state_hash(roundtrip)
    legacy, _ = roundtrip["legacy"].predict(dx)
    manual = p.compact_legacy(legacy)
    for family in ("crossfit", "extra"):
        forecast = roundtrip[family].predict(dx)
        for mode, key in (("calibrated", "predictions"), ("eta1", "eta1_predictions")):
            manual.update({f"{family}__{mode}__{name}": value for name, value in forecast[key].items()})
    # Manually forward fixed identities; never reselect on query values.
    for name, record in roundtrip["policies"].items():
        family, method = record["selected_identity"].split("__", 1)
        for mode in ("calibrated", "eta1"):
            actual = "gamma1" if family == "legacy" and mode == "eta1" else mode
            manual[f"policy__{mode}__{name}"] = manual[f"{family}__{actual}__{method}"].copy()
    for mode in ("calibrated", "eta1"):
        manual[f"policy__{mode}__expanded_nonkahm"] = manual[f"policy__{mode}__P0"].copy()
    assert len(manual) == 50 and set(manual) == p.expected_keys()
    assert fd.state_hash(roundtrip) == before

    query = {**task, "query": "prepared/query.npz"}
    p.write_npz(temporary / query["query"], {"D_X": dx})
    p.write_json(temporary / "FIT_LOCK.json", {"files": {str(state_path.relative_to(temporary)): p.digest(state_path)}})
    prediction = p.worker_predict(temporary, query)
    pred_dir = temporary / "predictions" / task["task_id"]
    with np.load(pred_dir / "predictions.npz", allow_pickle=False) as saved:
        assert set(saved.files) == p.expected_keys()
        for key in saved.files:
            np.testing.assert_array_equal(saved[key], manual[key])
    receipt = json.loads((pred_dir / "prediction_receipt.json").read_text())
    assert receipt["state_unchanged"] and receipt["targets_loaded"] is False
    # Reordered or augmented query batches must not define new centering stats.
    permutation = rng.permutation(len(dx))
    for family in ("crossfit", "extra"):
        original = roundtrip[family].predict(dx)["predictions"]
        reordered = roundtrip[family].predict(dx[permutation])["predictions"]
        expanded = roundtrip[family].predict(np.vstack((dx, 5 * dx[:2])))["predictions"]
        for name, value in original.items():
            np.testing.assert_allclose(reordered[name], value[permutation], atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(expanded[name][:len(dx)], value, atol=1e-11, rtol=1e-11)
    assert fd.state_hash(roundtrip) == before
    assert source_sha == p.digest(ROOT / "code/extra_rbf.py")
    from extra_rbf import manufactured_fixture
    extra_checks = manufactured_fixture()
    report = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "manufactured_data_only": True, "production_data_generated": False,
              "production_config_modified": False, "fixture_root": str(temporary),
              "fixture_source_sha256": p.digest(__file__), "extra_rbf_source_sha256": source_sha,
              "arrays_per_stage": 50, "A_B_C_query_rows": [64, 24, 20, 17],
              "A_episode_groups": 16, "fit": fit, "prediction": prediction,
              "independent_C_policy_argmin_and_forwarding": policy_checks,
              "nested_C_loss_nondegradation": True, "pickle_roundtrip": True,
              "independent_query_policy_forwarding": True, "target_free_prediction": True,
              "query_batch_invariance": True, "whole_state_query_immutability": True,
              "extra_rbf_source_unchanged": True, "extra_RBF_checks": extra_checks}
    output = ROOT / "readiness/manufactured_integration.json"
    p.write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
