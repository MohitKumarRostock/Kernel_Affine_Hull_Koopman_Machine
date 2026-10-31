"""Reproduce the two distinct defects observed in the frozen first attempt.

This is a post-observation numerical diagnostic, not a new efficacy run.
It reads original source, inputs, configuration and forecasts from the
preserved ZIP. It never rewrites that archive or any study result directory.

Run from the repository root:
    python -m extensions.adaptive_development.AMENDMENT_DIAGNOSTICS --check-amendment

The optional amendment checks replay only the four previously failing
prefixes and the two adverse forecast states; they are not timed benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import types
import zipfile

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import scipy
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
PREFIX = "extensions/adaptive_development/"
ARCHIVE_SHA256 = "6ec5bb7cae6336fb53d417b82d907c16b43fc25db2d99af32ce04647c20cdbe3"
ORIGINAL_CASES = (
    (20261101, "small", "fixed_kahm", 1279, 1),
    (20261101, "large", "adaptive_kahm", 1554, 12),
    (20261101, "large", "adaptive_affine", 1555, 12),
)
FAILURE_PREFIXES = (
    (20261101, "small", "fixed_kahm", 1279),
    (20261102, "small", "fixed_kahm", 1159),
    (20261103, "small", "fixed_kahm", 1160),
    (20261103, "small", "adaptive_kahm", 1685),
)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def archived_module(archive):
    source = archive.read(PREFIX + "models.py")
    name = "_kahm_archived_initial_attempt_models"
    module = types.ModuleType(name)
    module.__file__ = str(archive.filename) + "!/" + PREFIX + "models.py"
    sys.modules[name] = module  # dataclasses resolves the defining module here.
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def archived_task(archive, seed):
    name = f"synthetic_recurring__{seed}"
    manifest = json.loads(archive.read(PREFIX + "prepared_inputs/manifest.json"))
    task = dict(next(r for r in manifest["tasks"] if r["task_id"] == name))
    with np.load(io.BytesIO(archive.read(PREFIX + "prepared_inputs/" + task["filename"])),
                 allow_pickle=False) as arrays:
        task["values"] = arrays["values"].copy()
    return task


def replay(module, task, config, budget, method, stop):
    """Exact matured-pair chronology, without timing or result persistence."""
    values = task["values"]
    length, end = int(task["history"]), int(task["fit_end"])
    horizons = tuple(map(int, task["horizons"]))
    origins = np.arange(length - 1, end - max(horizons))
    x = np.asarray([values[o-length+1:o+1] for o in origins])
    y = np.asarray([values[o+np.asarray(horizons)] for o in origins])
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    last_initial_origin = int(origins[finite][-1])
    model = module.OnlineForecaster(
        method, **config["budgets"][budget], horizons=horizons,
        seed=config["model_seed"], config=config.get("model_config", {}))
    model.fit(x[finite], y[finite])
    error = None
    for now in range(end, stop + 1):
        matured = []
        for hi, horizon in enumerate(horizons):
            origin = now - horizon
            if origin <= last_initial_origin or origin < length - 1:
                continue
            history = values[origin-length+1:origin+1]
            if np.isfinite(values[now]) and np.isfinite(history).all():
                matured.append({"horizon_index": hi, "x": history.copy(),
                                "y": float(values[now]), "origin": origin, "target": now})
        observed = values[now-length+1:now+1].copy()
        observed = observed if np.isfinite(observed).all() else None
        try:
            model.update(matured, now, observed)
        except Exception as exc:
            error = {"time": now, "type": type(exc).__name__, "message": str(exc)}
            break
    return model, error, now


def objective(features, residual, coefficients, penalty):
    error = residual - features @ coefficients
    return float(error @ error + coefficients @ penalty @ coefficients)


def inspect_original(archive, module, config, case):
    seed, budget, method, stop, horizon = case
    task = archived_task(archive, seed)
    model, error, reached = replay(module, task, config, budget, method, stop)
    if reached != stop:
        raise RuntimeError(f"Unexpected early failure in {case}: {error}")
    index = tuple(task["horizons"]).index(horizon)
    head = model.heads[index]
    use = head.valid[:head.count]
    features = head.f[:head.count][use]
    x, y = head.x[:head.count][use], head.y[:head.count][use]
    residual = y - x @ head.raw_coef
    raw_query = task["values"][stop-task["history"]+1:stop+1]
    z = (raw_query - model.mean) / model.scale
    f, safe, feature_diagnostics = model._features(z[None, :])
    query = np.r_[1.0, f[0]]

    # Two independent fresh solves: with intercept in the augmented design,
    # and after eliminating the free intercept by centering. Neither forms
    # normal equations, accumulates historical moments or uses amended code.
    augmented = np.vstack((features, np.sqrt(head.geo_penalty)))
    rhs = np.r_[residual, np.zeros(features.shape[1])]
    direct = lstsq(augmented, rhs, cond=1e-15, lapack_driver="gelsd")[0]
    center, mean = features[:, 1:].mean(axis=0), float(residual.mean())
    centered_augmented = np.vstack((features[:, 1:] - center,
                                    np.sqrt(head.geo_penalty[1:, 1:])))
    centered_rhs = np.r_[residual - mean, np.zeros(features.shape[1] - 1)]
    slopes = lstsq(centered_augmented, centered_rhs, cond=1e-15,
                   lapack_driver="gelsd")[0]
    centered_coef = np.r_[mean - center @ slopes, slopes]
    accumulated = head.ff + head.geo_penalty
    fresh = features.T @ features + head.geo_penalty

    result_path = PREFIX + f"results/{task['task_id']}__{budget}__{method}/"
    summary = json.loads(archive.read(result_path + "summary.json"))
    with np.load(io.BytesIO(archive.read(result_path + "forecasts.npz")),
                 allow_pickle=False) as recorded:
        selected = (recorded["origin"] == stop) & (recorded["horizon"] == horizon)
        saved = {name: float(recorded[name][selected][0])
                 for name in ("baseline", "correction", "prediction", "gamma", "truth",
                              "framework_fallback")}
    return {
        "task_id": task["task_id"], "budget": budget, "method": method,
        "origin": stop, "horizon": horizon, "horizon_index": index,
        "original_update_error": error, "original_gamma": summary["gamma"],
        "original_recorded_forecast": saved,
        "feature_scale_min": float(model.feature_scale.min()),
        "feature_scale": model.feature_scale.tolist(),
        "query_feature_max_abs": float(np.abs(f).max()),
        "replay_feature_max_abs": float(np.abs(features[:, 1:]).max()),
        "query_feature_safe": bool(safe[0]), "feature_diagnostics": feature_diagnostics,
        "valid_replay_rows": int(use.sum()),
        "max_abs_accumulated_minus_fresh_ff": float(np.abs(head.ff - features.T @ features).max()),
        "accumulated_ridge_min_eigenvalue": float(np.linalg.eigvalsh((accumulated + accumulated.T)/2).min()),
        "fresh_ridge_min_eigenvalue": float(np.linalg.eigvalsh((fresh + fresh.T)/2).min()),
        "fresh_ridge_condition_number": float(np.linalg.cond(fresh)),
        "retained_original_correction": float(query @ head.geo_coef),
        "fresh_augmented_svd_correction": float(query @ direct),
        "fresh_centered_svd_correction": float(mean + (f[0] - center) @ slopes),
        "retained_original_objective": objective(features, residual, head.geo_coef, head.geo_penalty),
        "fresh_augmented_svd_objective": objective(features, residual, direct, head.geo_penalty),
        "fresh_centered_svd_objective": objective(features, residual, centered_coef, head.geo_penalty),
        "failure_state_qualification": (
            "The failed update had pushed current rows and solved the horizon-1 raw baseline; "
            "its correction coefficients are retained from the preceding successful solve. "
            "The framework's issued forecast used its recorded seasonal fallback."
            if error else "All updates through this forecast state succeeded."),
    }


def check_amendment(archive, config):
    from . import models
    checks = []
    for seed, budget, method, failure_time in FAILURE_PREFIXES:
        task = archived_task(archive, seed)
        stop = failure_time + 1
        model, error, reached = replay(models, task, config, budget, method, stop)
        if error:
            raise AssertionError(f"Amendment failed on a known failure prefix: {error}")
        query = task["values"][stop-task["history"]+1:stop+1]
        base, correction, _ = model.predict_components(query)
        if not np.isfinite(np.r_[base, correction]).all():
            raise AssertionError("Nonfinite amended prefix prediction")
        checks.append({"task_id": task["task_id"], "budget": budget, "method": method,
                       "previous_failure_time": failure_time, "completed_through": reached,
                       "errors": [], "finite_components": True})
    for seed, budget, method, stop, horizon in ORIGINAL_CASES[1:]:
        task = archived_task(archive, seed)
        model, error, reached = replay(models, task, config, budget, method, stop)
        if error:
            raise AssertionError(f"Amended extrapolation-state check failed: {error}")
        query = task["values"][stop-task["history"]+1:stop+1]
        base, correction, _ = model.predict_components(query)
        index = tuple(task["horizons"]).index(horizon)
        checks.append({"task_id": task["task_id"], "budget": budget, "method": method,
                       "origin": reached, "horizon": horizon,
                       "baseline": float(base[index]), "correction": float(correction[index]),
                       "qualification": "Adverse extrapolation remains; this is not an efficacy rerun."})
    return {"models_sha256": sha256(HERE / "models.py"), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=HERE / "archives/initial_attempt.zip")
    parser.add_argument("--output", type=Path, default=HERE / "AMENDMENT_DIAGNOSTICS.json")
    parser.add_argument("--check-amendment", action="store_true")
    args = parser.parse_args()
    archive_hash = sha256(args.archive)
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError("The original-attempt archive checksum does not match the preserved evidence")
    with zipfile.ZipFile(args.archive) as archive, threadpool_limits(limits=1):
        original, source_hash = archived_module(archive)
        config = json.loads(archive.read(PREFIX + "config.json"))
        report = {
            "status": "Post-observation numerical diagnosis; no efficacy selection or threshold changes.",
            "original_archive_sha256": archive_hash,
            "original_models_sha256": source_hash,
            "diagnostic_script_sha256": sha256(__file__),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__},
            "cases": [inspect_original(archive, original, config, case) for case in ORIGINAL_CASES],
        }
        if args.check_amendment:
            report["amendment_regression_checks"] = check_amendment(archive, config)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")
    for case in report["cases"]:
        print(case["method"], case["budget"], case["origin"], case["horizon"],
              "retained/fresh correction", case["retained_original_correction"],
              case["fresh_centered_svd_correction"],
              "moment drift", case["max_abs_accumulated_minus_fresh_ff"])


if __name__ == "__main__":
    main()
