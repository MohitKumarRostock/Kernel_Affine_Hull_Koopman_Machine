"""Deterministic development streams for the bounded adaptive KAHM pilot.

These inputs are development evidence.  The natural sources were previously
inspected, and the new synthetic streams implement a deliberately simple
regime hypothesis; neither is reserved source-transfer confirmation.

Only ``values`` and its declared history/target offsets may enter forecasting.
Synthetic regime labels and innovations are retained for auditing/analysis,
not passed as model covariates.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PREPARED = HERE / "prepared_inputs"
ORIGINAL = REPO_ROOT / "extensions" / "direct_reconstruction" / "data"
PARENT_COMMIT = "88e9c568ee983856804a2b2da41f6290f259b8cd"
STUDY = "bounded_adaptive_kahm_development_v1"
SYNTHETIC_SEEDS = (20261101, 20261102, 20261103, 20261104)
SMOKE_SEED = 20261099
SYNTHETIC_LENGTH = 2048
BURN_IN = 256
REGIME_PERIOD = 256
REGIME_PARAMETERS = {
    "A": {"mu": -0.75, "a": 0.30, "b": 0.35},
    "B": {"mu": 0.75, "a": 0.45, "b": -0.25},
}
REAL_TASKS = (
    "dominick_weekly__01", "dominick_weekly__02", "dominick_weekly__03",
    "sunspot_daily__01",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _array_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def _scale(values, fit_end, seasonal_lag):
    """Squared-difference scale using the initial fitting prefix only."""
    initial = np.asarray(values, dtype=np.float64)[:fit_end]
    for lag, name in ((seasonal_lag, "seasonal_mse"), (1, "first_difference_mse")):
        if lag < len(initial):
            delta = initial[lag:] - initial[:-lag]
            finite = delta[np.isfinite(delta)]
            candidate = float(np.mean(finite * finite)) if len(finite) else float("nan")
            if np.isfinite(candidate) and candidate > 0.0:
                return candidate, name, int(len(finite))
    return 1.0, "unit_fallback", 0


def _task(task_id, kind, collection, values, history, horizons, seasonal_lag, metadata):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    fit_end = 35 * n // 100
    eval_start = n // 2
    horizons = tuple(int(h) for h in horizons)
    if values.ndim != 1 or n < history + max(horizons) + 2:
        raise ValueError("Insufficient one-dimensional stream")
    if not (history + max(horizons) < fit_end < eval_start < n):
        raise ValueError("Invalid fitting/calibration/evaluation boundaries")
    scale, scale_method, scale_count = _scale(values, fit_end, seasonal_lag)
    metadata = {
        **metadata,
        "n": n,
        "finite_observations": int(np.isfinite(values).sum()),
        "raw_values_sha256": _array_hash(values),
        "boundary_convention": "Half-open initial fit [0,fit_end); calibration forecast origins [fit_end,eval_start); evaluation origins [eval_start,n).",
        "fit_end": fit_end,
        "eval_start": eval_start,
        "initial_complete_origins": [history - 1, fit_end - max(horizons) - 1],
        "initial_complete_row_count": fit_end - max(horizons) - history + 1,
        "initial_matured_origin_last_by_horizon": {str(h): fit_end - 1 - h for h in horizons},
        "calibration_scored_origin_last_by_horizon": {str(h): eval_start - 1 - h for h in horizons},
        "evaluation_scored_origin_last_by_horizon": {str(h): n - 1 - h for h in horizons},
        "evaluation_scored_origin_counts_if_finite": {str(h): n - h - eval_start for h in horizons},
        "scale_method": scale_method,
        "scale_pair_count": scale_count,
        "loss_scale": scale,
        "loss_scale_scope": "Initial fitting observations [0,fit_end), before calibration and evaluation.",
        "scale_rule": "Mean squared seasonal differences; if unavailable/nonpositive/nonfinite use mean squared first differences; if still invalid use 1.0. No square root.",
        "missingness_rule": "No imputation. Forecast only from finite history; score each horizon only when its individual target is finite. Availability and method failures remain distinct.",
    }
    return {
        "task_id": task_id, "kind": kind, "collection": collection,
        "values": values, "history": int(history), "horizons": horizons,
        "seasonal_lag": int(seasonal_lag), "fit_end": fit_end,
        "eval_start": eval_start, "scale": scale, "metadata": metadata,
    }


def make_synthetic(seed: int, regime: str, n: int = SYNTHETIC_LENGTH) -> dict:
    """Create a stationary or recurring stable nonlinear scalar AR stream.

    y[t] = mu[s] + a[s]*(y[t-1]-mu[s]) + .10*(y[t-2]-mu[s])
           + b[s]*tanh(1.5*(y[t-1]-mu[s])) + .20*epsilon[t].

    Both settings use identical innovations for a given seed. The burn-in
    always uses A. Recurring retained observations alternate A/B every 256
    points. No target, model fit, or error is used to choose these inputs.
    """
    if regime not in ("stationary", "recurring"):
        raise ValueError("regime must be stationary or recurring")
    if n < 128:
        raise ValueError("Synthetic stream must have at least 128 observations")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    innovations = rng.standard_normal(BURN_IN + int(n))
    y = np.empty(BURN_IN + int(n), dtype=np.float64)
    tags = np.zeros(BURN_IN + int(n), dtype=np.int8)
    previous, before_previous = -0.75, -0.75
    for j in range(len(y)):
        retained_time = j - BURN_IN
        tag = ((retained_time // REGIME_PERIOD) % 2) if regime == "recurring" and retained_time >= 0 else 0
        parameters = REGIME_PARAMETERS["B" if tag else "A"]
        mu, a, b = parameters["mu"], parameters["a"], parameters["b"]
        current = (mu + a * (previous - mu) + 0.10 * (before_previous - mu)
                   + b * np.tanh(1.5 * (previous - mu)) + 0.20 * innovations[j])
        y[j], tags[j] = current, tag
        before_previous, previous = previous, current
    retained = y[BURN_IN:]
    if not np.isfinite(retained).all():
        raise FloatingPointError("Synthetic generator produced nonfinite values")
    task = _task(
        f"synthetic_{regime}__{seed}", "synthetic", f"synthetic_{regime}",
        retained, 12, (1, 4, 12), 1,
        {
            "seed": int(seed), "generator": "paired_regime_tanh_ar_v1",
            "regime_schedule": regime, "burn_in": BURN_IN,
            "regime_period": REGIME_PERIOD, "regime_parameters": REGIME_PARAMETERS,
            "lag2_coefficient": 0.10, "tanh_scale": 1.5, "noise_standard_deviation": 0.20,
            "noise_distribution": "Independent standard normal innovations from NumPy PCG64, multiplied by 0.20.",
            "equation": "y_t = mu_s + a_s*(y_(t-1)-mu_s) + 0.10*(y_(t-2)-mu_s) + b_s*tanh(1.5*(y_(t-1)-mu_s)) + 0.20*epsilon_t",
            "observable_regime_qualification": "Regime changes alter the observable conditional mean and dynamics. Their identity is not perfectly observable under noise; no regime labels or transition warnings enter forecasting.",
            "contractivity_note": "The deterministic recurrence has sum of absolute lag derivatives at most 0.925 in A and 0.55 in B. This is a stability check, not a predictive advantage claim.",
            "evidence_status": "Fresh synthetic development repetitions from a newly specified simple law; not independent real-source confirmation.",
            "license": "CC-BY-4.0 under repository DATA_LICENSE",
            "excluded_smoke_seed": int(seed) == SMOKE_SEED,
        },
    )
    task["analysis_regimes"] = tags[BURN_IN:].copy()
    task["analysis_innovations"] = innovations[BURN_IN:].copy()
    return task


def _natural_tasks():
    source_manifest_path = ORIGINAL / "sources_manifest.json"
    manifest = json.loads(source_manifest_path.read_text())
    source_records = {record["source_key"]: record for record in manifest["sources"]}
    task_records = {record["task_id"]: record for record in manifest["tasks"]}
    tasks = []
    for name in REAL_TASKS:
        record = task_records[name]
        source = source_records[record["source_key"]]
        path = ORIGINAL / f"{name}.npz"
        observed_hash = sha256_file(path)
        if observed_hash != record["prepared_npz_sha256"]:
            raise ValueError(f"Original prepared source checksum differs: {name}")
        with np.load(path, allow_pickle=False) as arrays:
            values = arrays["raw_values"].astype(np.float64, copy=True)
            task = _task(
                name, "natural", record["source_key"], values,
                int(arrays["L"]), tuple(arrays["offsets"].tolist()), int(arrays["p"]),
                {
                    "series_id": record["series_id"],
                    "raw_start_index": int(arrays["raw_start_index"]),
                    "original_raw_length": record["original_raw_length"],
                    "source_npz": path.relative_to(REPO_ROOT).as_posix(),
                    "source_npz_sha256": observed_hash,
                    "source_manifest_sha256": sha256_file(source_manifest_path),
                    "source_archive_sha256": source["archive_sha256"],
                    "source_record_url": source["record_url"],
                    "source_doi": source["doi"],
                    "source_version": source["version"],
                    "source_title": source["title"],
                    "source_frequency": source["frequency"],
                    "creators": source["creators"],
                    "license": source["license"],
                    "license_url": source["license_url"],
                    "license_scope": source["license_scope"],
                    "evidence_status": "Previously inspected natural development source; selected intentionally using prior favorable/unfavorable KAHM findings, not new source-transfer confirmation.",
                    "selection_reason": "All three preserved Dominick series give the favorable development collection; the preserved Sunspot series gives an unfavorable collection. No new outcome-based series selection.",
                    "changes_from_parent": "Use every chronological raw observation, without old capped A/B/C/D window subsampling; new 35% initial-fit, 15% calibration, 50% online-evaluation schedule; initial-fit-only loss scale.",
                },
            )
        tasks.append(task)
    return tasks


def _write_npz(path, arrays):
    """Write fixed-order, fixed-timestamp little-endian NumPy members."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype.kind in "fi":
                value = value.astype(value.dtype.newbyteorder("<"), copy=False)
            member = io.BytesIO()
            np.lib.format.write_array(member, value, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def prepare(destination: Path = PREPARED):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tasks = [make_synthetic(seed, regime) for regime in ("stationary", "recurring") for seed in SYNTHETIC_SEEDS]
    tasks.extend(_natural_tasks())
    entries = []
    for task in tasks:
        meta = {key: value for key, value in task.items() if key not in ("values", "analysis_regimes", "analysis_innovations")}
        arrays = {
            "values": task["values"],
            "metadata_json": np.array(json.dumps(meta, sort_keys=True, separators=(",", ":"))),
        }
        for key in ("analysis_regimes", "analysis_innovations"):
            if key in task:
                arrays[key] = task[key]
        filename = task["task_id"] + ".npz"
        path = destination / filename
        _write_npz(path, arrays)
        entries.append({**meta, "filename": filename, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 1, "study": STUDY, "parent_commit": PARENT_COMMIT,
        "evidence_status": "Focused development study; no confirmation or population-transfer claim.",
        "task_count": len(entries), "synthetic_stream_count": 8, "natural_series_count": 4,
        "natural_collection_count": 2, "synthetic_seeds": list(SYNTHETIC_SEEDS),
        "excluded_smoke_seed": SMOKE_SEED,
        "model_input_restriction": "Only values and declared history/offset metadata enter models. analysis_regimes and analysis_innovations are audit-only.",
        "chronology": "At observed time t, score saved forecasts with target t, then release only matured labels, update, and issue immutable predictions for t+h. Calibration selection uses target times before eval_start.",
        "source_manifest_path": (ORIGINAL / "sources_manifest.json").relative_to(REPO_ROOT).as_posix(),
        "source_manifest_sha256": sha256_file(ORIGINAL / "sources_manifest.json"),
        "tasks": entries,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_tasks(directory: Path = PREPARED) -> list[dict]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    tasks = []
    for entry in manifest["tasks"]:
        path = directory / entry["filename"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Prepared stream checksum differs: {entry['task_id']}")
        with np.load(path, allow_pickle=False) as arrays:
            task = json.loads(str(arrays["metadata_json"]))
            task["values"] = arrays["values"].copy()
            task["horizons"] = tuple(task["horizons"])
            for key in ("analysis_regimes", "analysis_innovations"):
                if key in arrays:
                    task[key] = arrays[key].copy()
        if _array_hash(task["values"]) != task["metadata"]["raw_values_sha256"]:
            raise ValueError(f"Raw value checksum differs: {entry['task_id']}")
        tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Generate the fixed twelve development input streams")
    parser.add_argument("--directory", type=Path, default=PREPARED)
    args = parser.parse_args()
    manifest = prepare(args.directory) if args.prepare else json.loads((args.directory / "manifest.json").read_text())
    tasks = load_tasks(args.directory)
    print(json.dumps({"study": manifest["study"], "task_count": len(tasks),
                      "manifest": str(args.directory / "manifest.json"),
                      "manifest_sha256": sha256_file(args.directory / "manifest.json"),
                      "streams": [{"task_id": task["task_id"], "n": len(task["values"]),
                                   "fit_end": task["fit_end"], "eval_start": task["eval_start"],
                                   "scale": task["scale"]} for task in tasks]}, indent=2))


if __name__ == "__main__":
    main()
