#!/usr/bin/env python3
"""Manufactured regression checks for the separately corrected PCA verifier.

This program does not open production datasets, fitted states, predictions,
or result tables. Its receipt cannot certify the 192-task experiment.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def function_ast(path, name):
    return next(node for node in ast.parse(Path(path).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def assertion_calls(node):
    return [ast.dump(call, include_attributes=False)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "audit"
            and call.func.attr in ("require", "exact", "close")]


def expected_basis(q, cap):
    """SVD reference for manufactured data, independent of covariance eigh."""
    initial = min(cap, q.shape[1], len(q) - 1)
    if initial == 0:
        return np.empty((q.shape[1], 0)), initial
    centered = q - q.mean(axis=0)
    _, values, vh = np.linalg.svd(centered, full_matrices=False)
    positive = int(np.count_nonzero(values ** 2 / (len(q) - 1) > np.finfo(float).eps))
    if positive == 0:
        return np.empty((q.shape[1], 0)), initial
    rank = initial
    while True:
        p = vh[:min(rank, positive)].T
        if rank <= 1 or np.ptp(p.T @ q.T, axis=1).min() >= 1e-3:
            return p, initial
        rank -= 1


def case_bank(tx, ids, arm, cap, oracle):
    q = oracle.transformed(tx, arm)[ids].astype(np.float32).astype(np.float64)
    p, initial = expected_basis(q, cap)
    bank = {"models": [{"q": q, "p": p,
                        "dimension_audit": {"initial_admissible_rank": initial,
                                            "effective_rank": p.shape[1],
                                            "normal_direction_reserved": False}}]}
    records = [{"reference_indices": list(ids)}]
    return bank, records


def execute(module, tx, bank, records, oracle, arm, cap, audit=None):
    audit = module.Audit() if audit is None else audit
    module.selected_native_bank(tx, bank, records, oracle, audit,
                                f"manufactured/{arm}__cap{cap}")
    return audit


def expect_failure(call, exception, label):
    try:
        call()
    except exception as error:
        return {"case": label, "status": "expected_rejection",
                "exception": type(error).__name__, "message": str(error)}
    raise AssertionError(label + " unexpectedly succeeded")


def run():
    original_path = SOURCE_ROOT / "audit/verify_results.py"
    corrected_path = HERE / "verify_results.py"
    original = load("zero_rank_frozen_verifier", original_path)
    corrected = load("zero_rank_corrected_verifier", corrected_path)
    freeze = json.loads((SOURCE_ROOT / "freeze.json").read_text())
    for relative, expected in freeze["files"].items():
        assert digest(SOURCE_ROOT / relative) == expected, "Frozen file changed: " + relative

    original_tree, corrected_tree = [ast.parse(p.read_text())
                                     for p in (original_path, corrected_path)]
    original_function = function_ast(original_path, "selected_native_bank")
    corrected_function = function_ast(corrected_path, "selected_native_bank")
    assert assertion_calls(original_function) == assertion_calls(corrected_function), \
        "Existing validation assertions changed"
    for tree in (original_tree, corrected_tree):
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.FunctionDef)
                             and node.name == "selected_native_bank")]
    assert ast.dump(original_tree, include_attributes=False) == ast.dump(corrected_tree, include_attributes=False), \
        "Correction altered code outside selected_native_bank"

    # Load just the frozen preprocessing function; do not import its runtime.
    transformation = function_ast(SOURCE_ROOT / "audit/verify_crossfit.py", "transformed")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[transformation], type_ignores=[]),
                 "<frozen-transformation-only>", "exec"), namespace)
    oracle = SimpleNamespace(transformed=namespace["transformed"])
    rng = np.random.default_rng(2026110106)
    tx = rng.normal(size=(31, 4)) * np.array([0.3, 1.1, 2.2, 3.7])
    outcomes = []

    for arm in ("standard", "minmax", "tanh_z_1"):
        bank, records = case_bank(tx, [7], arm, 4, oracle)
        outcomes.append(expect_failure(
            lambda: execute(original, tx, bank, records, oracle, arm, 4),
            ZeroDivisionError, arm + "/original_single_reference_failure"))
        with np.errstate(all="raise"):
            audited = execute(corrected, tx, bank, records, oracle, arm, 4)
        assert audited.counts["selected_zero_admissible_rank_bases"] == 1
        assert audited.counts["selected_reference_eigendecompositions"] == 0
        assert not audited.pca_cache
        assert audited.counts["selected_local_bases_checked"] == 1
        outcomes.append({"case": arm + "/corrected_single_reference",
                         "status": "passed", "counts": dict(audited.counts),
                         "ranks": dict(audited.ranks)})

        malformed, bad_records = case_bank(tx, [7], arm, 4, oracle)
        malformed["models"][0]["p"] = np.eye(4)[:, :1]
        malformed["models"][0]["dimension_audit"]["effective_rank"] = 1
        outcomes.append(expect_failure(
            lambda: execute(corrected, tx, malformed, bad_records, oracle, arm, 4),
            AssertionError, arm + "/reject_nonzero_single_reference_basis"))

        wrong_dimension, wrong_records = case_bank(tx, [7], arm, 4, oracle)
        wrong_dimension["models"][0]["dimension_audit"]["normal_direction_reserved"] = True
        outcomes.append(expect_failure(
            lambda: execute(corrected, tx, wrong_dimension, wrong_records, oracle, arm, 4),
            AssertionError, arm + "/reject_wrong_single_reference_dimension_metadata"))

        constant = np.tile([1.0, -2.0, 3.0, 0.5], (8, 1))
        bank, records = case_bank(constant, list(range(6)), arm, 4, oracle)
        old = execute(original, constant, bank, records, oracle, arm, 4)
        new = execute(corrected, constant, bank, records, oracle, arm, 4)
        assert old.ranks == new.ranks
        assert new.counts["selected_zero_variance_rank_bases"] == 1
        assert new.counts["selected_reference_eigendecompositions"] == 1
        comparable = new.counts.copy()
        del comparable["selected_zero_variance_rank_bases"]
        assert comparable == old.counts
        outcomes.append({"case": arm + "/constant_multiple_references",
                         "status": "passed", "counts": dict(new.counts),
                         "original_behavior_preserved": True})

        for cap in (1, 2, 4, 12):
            bank, records = case_bank(tx, list(range(17)), arm, cap, oracle)
            old = execute(original, tx, bank, records, oracle, arm, cap)
            new = execute(corrected, tx, bank, records, oracle, arm, cap)
            assert old.counts == new.counts and old.ranks == new.ranks
            assert bank["models"][0]["p"].shape[1] > 0
            assert set(old.pca_cache) == set(new.pca_cache)
            for key in old.pca_cache:
                np.testing.assert_array_equal(old.pca_cache[key][0], new.pca_cache[key][0])
                assert old.pca_cache[key][1] == new.pca_cache[key][1]
            execute(corrected, tx, bank, records, oracle, arm, cap, audit=new)
            assert new.counts["selected_reference_eigendecompositions"] == 1
            assert new.counts["selected_local_bases_checked"] == 2
            outcomes.append({"case": f"{arm}/ordinary_rank_cap{cap}",
                             "status": "passed", "original_behavior_preserved": True,
                             "repeat_uses_cached_eigendecomposition": True,
                             "effective_rank": bank["models"][0]["p"].shape[1]})

    return {"status": "passed", "scope": "Manufactured selected-native-bank regression only",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_results_audited": False, "production_arrays_read": False,
            "frozen_files_hashes_verified": len(freeze["files"]),
            "all_code_outside_selected_native_bank_AST_identical": True,
            "all_existing_selected_native_bank_assertions_AST_identical": True,
            "case_count": len(outcomes),
            "case_status_counts": dict(Counter(o["status"] for o in outcomes)),
            "cases": outcomes,
            "source_sha256": {"original_verifier": digest(original_path),
                               "corrected_verifier": digest(corrected_path),
                               "frozen_oracle": digest(SOURCE_ROOT / "audit/verify_crossfit.py"),
                               "freeze": digest(SOURCE_ROOT / "freeze.json"),
                               "regression_program": digest(__file__)},
            "environment": {"python": sys.version, "numpy": np.__version__},
            "not_claimed": ["Completed production audit", "Verified experimental performance",
                            "Recovery of missing study arrays or states",
                            "Correction of the frozen manufactured-data oracle"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior regression receipts: " + str(args.output))
    try:
        receipt = run()
    except Exception:
        receipt = {"status": "failed", "scope": "Manufactured selected-native-bank regression only",
                   "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "traceback": traceback.format_exc(), "scientific_results_audited": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "case_count": receipt.get("case_count"),
                      "receipt": str(args.output), "traceback": receipt.get("traceback")}, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
