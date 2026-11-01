#!/usr/bin/env python3
"""Independent validity and exception-path audit of the ridge SVD amendment.

The only scientific arrays read are the exact saved training-design failure.
No B/C predictive score is calculated and no terminal data are opened.
The coefficient oracle is augmented least-squares QR, independent of SVD.
Temporary mocks exercise exception branches; they are never installed in a
production fitting or query process.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / "crossfit_residual"
sys.path.insert(0, str(ROOT / "code"))
import crossfit_model as model
import numerical_solver as solver

ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.assertions = 0

    def require(self, condition, label):
        self.assertions += 1
        if not bool(condition):
            raise AssertionError(label)

    def exact(self, actual, expected, label):
        self.assertions += 1
        np.testing.assert_array_equal(actual, expected, err_msg=label)


def relative_norm(value, reference):
    return float(np.linalg.norm(value) / max(float(np.linalg.norm(reference)), np.finfo(float).tiny))


def qr_ridge(centered, response, alpha):
    n, p = centered.shape
    augmented = np.vstack((centered, np.sqrt(n * alpha) * np.eye(p)))
    rhs = np.vstack((response, np.zeros((p, response.shape[1]))))
    q, r = np.linalg.qr(augmented, mode="reduced")
    return scipy.linalg.solve_triangular(r, q.T @ rhs, lower=False)


def expect_failure(function, expected_type, audit, label):
    try:
        function()
    except expected_type as error:
        audit.require(True, label)
        return {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        raise AssertionError(label + ": unexpected exception " + repr(error)) from error
    raise AssertionError(label + ": unexpectedly returned a result")


def event_check(audit, *, recovered):
    events = solver.events()
    audit.require(len(events) == 1, "Exactly one fallback event per failed original path")
    event = events[0]
    audit.require(event["status"] == ("recovered" if recovered else "failed"), "Explicit fallback outcome")
    return event


def run(evidence):
    audit = Audit()
    evidence = Path(evidence).resolve()
    metadata = json.loads((evidence / "metadata.json").read_text())
    saved = {name: np.load(evidence / name, allow_pickle=False) for name in
             ("x.npy", "y.npy", "centered.npy", "response.npy", "mean.npy", "ymean.npy")}
    for name, expected in metadata["saved_files"].items():
        audit.require(digest(evidence / name) == expected, "Saved failed-matrix file identity")
    x, y = saved["x.npy"], saved["y.npy"]
    centered, response = saved["centered.npy"], saved["response.npy"]
    audit.require(x.shape == (383, 182) and y.shape == (383, 3), "Exact failed design shape")
    audit.require(np.isfinite(x).all() and np.isfinite(y).all(), "Finite saved training data")
    audit.exact(x.mean(axis=0), saved["mean.npy"], "Original feature centering")
    audit.exact(y.mean(axis=0), saved["ymean.npy"], "Original response centering")
    audit.exact(x - x.mean(axis=0), centered, "Exact centered design")
    audit.exact(y - y.mean(axis=0), response, "Exact centered response")
    audit.require(hashlib.sha256(np.ascontiguousarray(centered).tobytes()).hexdigest()
                  == metadata["matrix_sha256"], "Exact failing matrix bytes")
    audit.require(tuple(metadata["stack"][0]["alphas"]) == ALPHAS, "Original penalty grid retained")
    original_freeze = json.loads((ORIGINAL / "freeze.json").read_text())
    source_before = {name: digest(ORIGINAL / name) for name in original_freeze["files"]}
    audit.require(source_before == original_freeze["files"], "Original frozen source unchanged before audit")
    runtime = model.fd.load_runtime()
    original_path = runtime._ridge_path
    numpy_svd = np.linalg.svd
    rng = np.random.default_rng(71863542)
    easy_x, easy_y = rng.normal(size=(79, 11)), rng.normal(size=(79, 3))
    with threadpool_limits(limits=1):
        expected_easy = original_path(easy_x, easy_y, ALPHAS)
        original_error = expect_failure(lambda: original_path(x, y, ALPHAS), np.linalg.LinAlgError,
                                        audit, "Original finite-design failure reproduced")
        provenance = solver.install(runtime)
        installed = runtime._ridge_path
        second_provenance = solver.install(runtime)
        audit.require(runtime._ridge_path is installed and second_provenance == provenance, "Idempotent scoped installation")
        audit.require(np.linalg.svd is numpy_svd, "Installation does not replace global NumPy SVD")
        solver.clear_events()
        with patch.object(solver.scipy.linalg, "svd", side_effect=AssertionError("Unexpected alternate solver")):
            actual_easy = runtime._ridge_path(easy_x, easy_y, ALPHAS)
        audit.require(len(actual_easy) == len(expected_easy) == len(ALPHAS), "Successful original head count")
        for actual, expected in zip(actual_easy, expected_easy):
            audit.require(set(actual) == set(expected), "Successful original head schema unchanged")
            for key in actual:
                audit.exact(actual[key], expected[key], "Successful original path bitwise identity: " + key)
        audit.require(solver.events() == [], "Successful original path has no fallback event")
        solver.clear_events()
        real_heads = runtime._ridge_path(x, y, ALPHAS)
        real_event = event_check(audit, recovered=True)
        u, singular, vh = scipy.linalg.svd(centered, full_matrices=False, lapack_driver="gesvd", check_finite=True)
        factorization = {
            "relative_reconstruction": relative_norm((u * singular) @ vh - centered, centered),
            "left_orthogonality_frobenius": float(np.linalg.norm(u.T @ u - np.eye(len(singular)))),
            "right_orthogonality_frobenius": float(np.linalg.norm(vh @ vh.T - np.eye(len(singular)))),
        }
        audit.require(np.isfinite(u).all() and np.isfinite(singular).all() and np.isfinite(vh).all(), "Finite alternate SVD factors")
        audit.require(factorization["relative_reconstruction"] < 1e-10, "Independent SVD reconstruction validity")
        audit.require(max(factorization["left_orthogonality_frobenius"], factorization["right_orthogonality_frobenius"]) < 1e-8,
                      "Independent SVD orthogonality validity")
        gram, rhs = centered.T @ centered, centered.T @ response
        checks = []
        for alpha, head in zip(ALPHAS, real_heads):
            coefficient = head["coef"]
            oracle = qr_ridge(centered, response, alpha)
            system = gram + len(centered) * alpha * np.eye(centered.shape[1])
            normal_residual = relative_norm(system @ coefficient - rhs, rhs)
            coefficient_difference = relative_norm(coefficient - oracle, oracle)
            fit_difference = relative_norm(centered @ (coefficient - oracle), centered @ oracle)
            audit.require(np.isfinite(coefficient).all(), "Finite recovered ridge coefficients")
            audit.require(normal_residual < 1e-10, "Independent ridge normal-equation residual")
            audit.require(coefficient_difference < 1e-8 and fit_difference < 1e-8, "Independent augmented-QR coefficient agreement")
            audit.exact(head["mean"], saved["mean.npy"], "Recovered feature centering unchanged")
            audit.exact(head["ymean"], saved["ymean.npy"], "Recovered response intercept unchanged")
            audit.exact(head["alpha_by_horizon"], np.full(3, alpha), "Recovered penalty unchanged")
            audit.require(head["training_rows"] == len(centered) and head["feature_width"] == centered.shape[1], "Recovered head allocation")
            expected_rank = int(np.count_nonzero(singular > max(centered.shape) * np.finfo(float).eps * singular[0]))
            audit.require(head["feature_rank"] == expected_rank, "Existing diagnostic rank tolerance unchanged")
            checks.append({"alpha": alpha, "relative_normal_equation_residual": normal_residual,
                           "relative_coefficient_difference_from_augmented_QR": coefficient_difference,
                           "relative_training_fit_difference_from_augmented_QR": fit_difference})
        # Exercise the actual original-exception branch on a well-conditioned
        # manufactured design. The temporary global mock belongs only to this
        # isolated audit process, never to the installed production adapter.
        solver.clear_events()
        with patch.object(np.linalg, "svd", side_effect=np.linalg.LinAlgError("audit forced original SVD nonconvergence")):
            forced_heads = runtime._ridge_path(easy_x, easy_y, ALPHAS)
        forced_event = event_check(audit, recovered=True)
        for alpha, head in zip(ALPHAS, forced_heads):
            oracle = qr_ridge(easy_x - easy_x.mean(axis=0), easy_y - easy_y.mean(axis=0), alpha)
            audit.require(relative_norm(head["coef"] - oracle, oracle) < 1e-10, "Forced-exception coefficients agree with QR")
        branch_checks = {}
        for exception in (ValueError("audit unrelated value error"), RuntimeError("audit unrelated runtime error")):
            solver.clear_events()
            with patch.object(np.linalg, "svd", side_effect=exception), patch.object(
                    solver.scipy.linalg, "svd", side_effect=AssertionError("An unrelated exception triggered retry")):
                branch_checks[type(exception).__name__] = expect_failure(
                    lambda: runtime._ridge_path(easy_x, easy_y, ALPHAS), type(exception), audit,
                    "Unrelated exceptions propagate without recovery")
            audit.require(solver.events() == [], "Unrelated failure is not a fallback event")
        for target in ("features", "responses"):
            solver.clear_events()
            bad_x, bad_y = easy_x.copy(), easy_y.copy()
            (bad_x if target == "features" else bad_y)[0, 0] = np.inf
            with patch.object(solver.scipy.linalg, "svd", side_effect=AssertionError("Nonfinite input reached alternate solver")):
                branch_checks["nonfinite_" + target] = expect_failure(
                    lambda: runtime._ridge_path(bad_x, bad_y, ALPHAS), ValueError, audit,
                    "Nonfinite input rejected")
            audit.require(solver.events() == [], "Nonfinite input is not a recoverable convergence event")
        solver.clear_events()
        with patch.object(np.linalg, "svd", side_effect=np.linalg.LinAlgError("audit forced original SVD failure")), patch.object(
                solver.scipy.linalg, "svd", side_effect=np.linalg.LinAlgError("audit forced gesvd failure")):
            branch_checks["alternate_driver_failure"] = expect_failure(
                lambda: runtime._ridge_path(easy_x, easy_y, ALPHAS), Exception, audit,
                "Alternate solver failure stops execution")
        failed_event = event_check(audit, recovered=False)
        solver.clear_events()
        k = min(easy_x.shape)
        invalid_factors = (np.zeros((len(easy_x), k)), np.ones(k), np.zeros((k, easy_x.shape[1])))
        with patch.object(np.linalg, "svd", side_effect=np.linalg.LinAlgError("audit forced original SVD failure")), patch.object(
                solver.scipy.linalg, "svd", return_value=invalid_factors):
            branch_checks["invalid_factors"] = expect_failure(
                lambda: runtime._ridge_path(easy_x, easy_y, ALPHAS), Exception, audit,
                "Numerically invalid alternate factors stop execution")
        invalid_event = event_check(audit, recovered=False)
        audit.require(np.linalg.svd is numpy_svd, "All temporary numerical mocks restored")
    audit.require({name: digest(ORIGINAL / name) for name in source_before} == source_before,
                  "Original frozen source unchanged after audit")
    for name, expected in metadata["saved_files"].items():
        audit.require(digest(evidence / name) == expected, "Saved failure evidence unchanged")
    return {"status": "passed", "assertions": audit.assertions,
            "scope": "One exact saved A-training failure plus manufactured exception-path probes; no B/C predictive scores or D data accessed",
            "matrix_shape": list(centered.shape), "matrix_sha256": metadata["matrix_sha256"],
            "original_failure": original_error, "installation": provenance,
            "factorization_checks": factorization, "independent_QR_ridge_checks": checks,
            "actual_recovery_event": real_event, "forced_recovery_event": forced_event,
            "expected_failed_recovery_event": failed_event, "expected_invalid_factor_event": invalid_event,
            "expected_exception_probes": branch_checks,
            "checks": {"successful_original_path_bitwise_equal": True, "scoped_idempotent_installation": True,
                       "same_objective_and_all_original_penalties": True, "independent_augmented_QR_agreement": True,
                       "unrelated_exceptions_propagate": True, "nonfinite_inputs_rejected": True,
                       "alternate_failure_stops": True, "invalid_factorization_stops": True,
                       "original_source_and_failure_evidence_unchanged": True},
            "source_sha256": {"code/numerical_solver.py": digest(ROOT / "code/numerical_solver.py"),
                              "audit/verify_numerical_amendment.py": digest(__file__)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path,
                        default=ORIGINAL / "readiness/svd_failure_duffing_rep01/failure_00")
    parser.add_argument("--output", type=Path, default=ROOT / "readiness/independent_numerical_amendment.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve audit attempts; choose a new output path")
    start = time.perf_counter()
    try:
        receipt = run(args.evidence)
    except Exception:
        receipt = {"status": "failed", "scope": "Independent numerical amendment validation; no terminal outcomes",
                   "traceback": traceback.format_exc()}
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
