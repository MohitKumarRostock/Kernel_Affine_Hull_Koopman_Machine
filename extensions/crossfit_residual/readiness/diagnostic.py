"""Preserve and diagnose the initial SVD failure without amending frozen code.

Only the already-prepared A/B arrays are read.  No fit output is saved into the
production results tree, no production file is overwritten, and no D is read.
The capture wrapper calls NumPy unchanged and re-raises its original error.
Driver replays are judged solely by numerical identities, never forecast loss.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import traceback

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import scipy
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "duffing__h12__noise0__rep01"
OUT = ROOT / "readiness" / "svd_failure_duffing_rep01"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    value = np.asarray(value)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def protected_hashes():
    frozen = json.loads((ROOT / "freeze.json").read_text())
    records = {}
    for category, directory in (("files", ROOT), ("inherited_files", ROOT.parent / "dimension_forecasting"),
                                ("previous_extension_files", ROOT.parent / "preprocessing_dimension")):
        for relative, expected in frozen[category].items():
            path = directory / relative
            actual = sha(path)
            if actual != expected:
                raise RuntimeError(f"Frozen file changed before diagnostic: {path}")
            records[str(path)] = actual
    for path in (ROOT / "freeze.json", ROOT / "prepared/abc_manifest.json",
                 ROOT / "results" / TASK_ID / "failure.json"):
        records[str(path)] = sha(path)
    for path in (ROOT / "execution").rglob("*" + TASK_ID + "*"):
        if path.is_file():
            records[str(path)] = sha(path)
    return records


def capture():
    OUT.mkdir(exist_ok=False)
    before = protected_hashes()
    if (ROOT / "prepared/d_manifest.json").exists():
        raise RuntimeError("This diagnostic was authorized before D generation")
    manifest = json.loads((ROOT / "prepared/abc_manifest.json").read_text())
    task = next(t for t in manifest["tasks"] if t["task_id"] == TASK_ID)
    assert task["fit_seed"] == 2073710093
    source = ROOT / task["input"]
    assert sha(source) == task["sha256"]
    before[str(source)] = sha(source)
    config = json.loads((ROOT / "config.json").read_text())
    sys.path.insert(0, str(ROOT / "code"))
    import crossfit_model as model

    original_svd = np.linalg.svd
    calls = 0
    captured = []

    def traced_svd(a, *args, **kwargs):
        nonlocal calls
        calls += 1
        try:
            return original_svd(a, *args, **kwargs)
        except np.linalg.LinAlgError:
            event_dir = OUT / f"failure_{len(captured):02d}"
            event_dir.mkdir()
            matrix = np.asarray(a)
            np.save(event_dir / "centered.npy", matrix, allow_pickle=False)
            frames = []
            frame = inspect.currentframe().f_back
            while frame is not None:
                entry = {"function": frame.f_code.co_name, "file": frame.f_code.co_filename,
                         "line": frame.f_lineno}
                local = frame.f_locals
                for name in ("tag", "prefix", "cap", "scaler", "key", "fold", "seed"):
                    value = local.get(name)
                    if isinstance(value, (str, int, float, bool)):
                        entry[name] = value
                for name in ("tx", "ty", "ax", "ay", "train_mask", "test_mask"):
                    if isinstance(local.get(name), np.ndarray):
                        entry[name + "_shape"] = list(local[name].shape)
                        entry[name + "_sha256"] = array_sha(local[name])
                if frame.f_code.co_name == "_ridge_path":
                    for name in ("x", "y", "mean", "ymean", "response"):
                        np.save(event_dir / (name + ".npy"), local[name], allow_pickle=False)
                    entry["alphas"] = [float(v) for v in local["alphas"]]
                frames.append(entry)
                frame = frame.f_back
            del frame
            metadata = {
                "utc": utc(), "svd_call_index": calls, "task_id": TASK_ID,
                "fit_seed": task["fit_seed"], "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                "finite": bool(np.isfinite(matrix).all()), "max_abs": float(np.max(np.abs(matrix))),
                "frobenius_norm": float(np.linalg.norm(matrix)),
                "min": float(matrix.min()), "max": float(matrix.max()),
                "c_contiguous": bool(matrix.flags.c_contiguous), "f_contiguous": bool(matrix.flags.f_contiguous),
                "kwargs": kwargs, "positional_args": list(args), "stack": frames,
                "matrix_sha256": array_sha(matrix), "traceback": traceback.format_exc(),
                "saved_files": {p.name: sha(p) for p in event_dir.glob("*.npy")},
            }
            write_json(event_dir / "metadata.json", metadata)
            captured.append(metadata)
            print(json.dumps({"capture": str(event_dir), "shape": metadata["shape"],
                              "finite": metadata["finite"], "context": frames[:4]}), flush=True)
            raise

    receipt = {"schema": "crossfit-svd-failure-diagnostic-v1", "started_utc": utc(),
               "task_id": TASK_ID, "fit_seed": task["fit_seed"],
               "source_abc_sha256": sha(source), "python": sys.version, "platform": platform.platform(),
               "numpy": np.__version__, "scipy": scipy.__version__,
               "read_arrays": ["A_X", "A_Y", "A_groups", "B_X", "B_Y"],
               "C_targets_read": False, "D_read_or_generated": False, "protected_before": before}
    try:
        with threadpool_limits(limits=1), np.load(source, allow_pickle=False) as data:
            receipt["threadpools"] = threadpool_info()
            ax, ay, groups = data["A_X"], data["A_Y"], data["A_groups"]
            bx, by = data["B_X"], data["B_Y"]
            np.linalg.svd = traced_svd
            print("Replaying legacy fit from unchanged A/B inputs", flush=True)
            legacy = model.previous.fit_primary(ax, ay, bx, by, seed=task["fit_seed"], config=config["model"])
            print("Replaying crossfit fit with NumPy SVD capture and no numerical fallback", flush=True)
            model.fit_crossfit(ax, ay, groups, bx, by, seed=task["fit_seed"], config=config["model"],
                               reference_primary=legacy)
            receipt["status"] = "unexpectedly_completed_without_svd_failure"
    except Exception as error:
        receipt["status"] = "failure_reproduced" if captured else "failed_before_svd_capture"
        receipt["exception_type"] = type(error).__name__
        receipt["exception"] = str(error)
        receipt["traceback"] = traceback.format_exc()
        raise
    finally:
        np.linalg.svd = original_svd
        after = {path: sha(path) for path in before}
        receipt["protected_after"] = after
        receipt["protected_unchanged"] = after == before
        receipt["svd_calls"] = calls
        receipt["captured_failures"] = len(captured)
        receipt["finished_utc"] = utc()
        write_json(OUT / "capture_receipt.json", receipt)
        if after != before:
            raise RuntimeError("A protected file changed during diagnostic")


def replay():
    matrix_dir = OUT / "failure_00"
    matrix = np.load(matrix_dir / "centered.npy", allow_pickle=False)
    response = np.load(matrix_dir / "response.npy", allow_pickle=False)
    metadata = json.loads((matrix_dir / "metadata.json").read_text())
    alphas = next(f["alphas"] for f in metadata["stack"] if f["function"] == "_ridge_path")
    if (OUT / "driver_replay.json").exists():
        raise RuntimeError("Refusing to overwrite driver evidence")
    receipt = {"schema": "crossfit-svd-driver-replay-v1", "utc": utc(),
               "matrix_sha256": array_sha(matrix), "matrix_shape": list(matrix.shape),
               "response_shape": list(response.shape), "finite_matrix": bool(np.isfinite(matrix).all()),
               "finite_response": bool(np.isfinite(response).all()),
               "numpy": np.__version__, "scipy": scipy.__version__, "alphas": alphas,
               "criterion": "Factorization and ridge normal equations only; no B/C/D forecast loss computed",
               "trials": []}
    successful = {}
    with threadpool_limits(limits=1):
        receipt["threadpools"] = threadpool_info()
        drivers = {
            "numpy_gesdd": lambda x: np.linalg.svd(x, full_matrices=False),
            "scipy_gesdd": lambda x: scipy.linalg.svd(x, full_matrices=False, lapack_driver="gesdd", check_finite=True),
            "scipy_gesvd": lambda x: scipy.linalg.svd(x, full_matrices=False, lapack_driver="gesvd", check_finite=True),
        }
        for label, function in drivers.items():
            for repeat in range(3):
                input_matrix = matrix.copy()
                trial = {"driver": label, "repeat": repeat, "input_sha256": array_sha(input_matrix)}
                try:
                    u, s, vt = function(input_matrix)
                    trial.update({"status": "success", "finite_factors": bool(all(np.isfinite(v).all() for v in (u, s, vt))),
                                  "smax": float(s.max()), "smin": float(s.min()),
                                  "numerical_rank": int(np.count_nonzero(s > max(matrix.shape) * np.finfo(float).eps * s[0])),
                                  "relative_reconstruction_error": float(np.linalg.norm((u * s) @ vt - matrix) / np.linalg.norm(matrix)),
                                  "left_orthogonality_frobenius": float(np.linalg.norm(u.T @ u - np.eye(len(s)))),
                                  "right_orthogonality_frobenius": float(np.linalg.norm(vt @ vt.T - np.eye(len(s)))),
                                  "singular_values_sha256": array_sha(s)})
                    if label not in successful:
                        successful[label] = (u, s, vt)
                        np.savez(OUT / (label + "_decomposition.npz"), u=u, singular=s, vt=vt)
                except np.linalg.LinAlgError as error:
                    trial.update(status="failure", exception=str(error))
                trial["input_unchanged"] = array_sha(input_matrix) == array_sha(matrix)
                receipt["trials"].append(trial)
                print(json.dumps(trial), flush=True)
        receipt["ridge_checks"] = []
        gram, rhs = matrix.T @ matrix, matrix.T @ response
        for label, (u, s, vt) in successful.items():
            for alpha in alphas:
                regularized = gram + len(matrix) * alpha * np.eye(matrix.shape[1])
                coefficient = vt.T @ ((s / (s ** 2 + len(matrix) * alpha))[:, None] * (u.T @ response))
                oracle = scipy.linalg.solve(regularized, rhs, assume_a="pos")
                residual = regularized @ coefficient - rhs
                row = {"driver": label, "alpha": alpha,
                       "relative_normal_equation_residual": float(np.linalg.norm(residual) / np.linalg.norm(rhs)),
                       "relative_coefficient_difference_from_spd_solve": float(np.linalg.norm(coefficient - oracle) / np.linalg.norm(oracle)),
                       "maximum_absolute_coefficient_difference": float(np.max(np.abs(coefficient - oracle))),
                       "relative_fitted_training_correction_difference": float(np.linalg.norm(matrix @ (coefficient - oracle)) / np.linalg.norm(matrix @ oracle)),
                       "finite_coefficient": bool(np.isfinite(coefficient).all())}
                receipt["ridge_checks"].append(row)
        if "scipy_gesdd" in successful and "scipy_gesvd" in successful:
            sd, sv = successful["scipy_gesdd"][1], successful["scipy_gesvd"][1]
            receipt["successful_driver_singular_max_abs_difference"] = float(np.max(np.abs(sd - sv)))
        receipt["matrix_column_mean_max_abs"] = float(np.max(np.abs(matrix.mean(0))))
        receipt["matrix_exact_zero_columns"] = int(np.count_nonzero(np.all(matrix == 0, axis=0)))
        receipt["response_column_mean_max_abs"] = float(np.max(np.abs(response.mean(0))))
    write_json(OUT / "driver_replay.json", receipt)
    print(str(OUT / "driver_replay.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "replay"])
    arguments = parser.parse_args()
    capture() if arguments.mode == "capture" else replay()
