"""Provenance, array storage, and isolated native-runtime loading."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parent
RISK = ROOT / "runtime" / "risk"
RISK_EXTENSION = RISK / "extensions" / "certified_forecasting"
STAGE_FILES = {
    "development": ("development.json", "fitted_models.pkl", "coefficients.npz", "validation_scores.npz"),
    "calibration": ("calibration.json", "calibration_scores.npz"),
    "test": ("test.json", "test_scores.npz"),
}


def load_runtime():
    for path in (RISK, RISK_EXTENSION / "src", RISK_EXTENSION / "experiments"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.shape).encode("ascii"))
    h.update(value.dtype.str.encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def sources():
    files = list((ROOT / "runtime").rglob("*.py")) + list(ROOT.glob("*.py"))
    files += [ROOT / name for name in ("config.json", "PROTOCOL.md", "runtime_sources.json", "requirements.txt")]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def seed_value(master, system_index, repetition, domain, extra=0):
    sequence = np.random.SeedSequence([master, system_index, repetition, domain, extra])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def geometry_seed(master, system_index, repetition, history):
    # Leave room for the native block-seed offsets inside the uint32 domain.
    return seed_value(master, system_index, repetition, 10, history) % (2**31)


def schedule(cfg):
    rows = []
    for index, system in enumerate(cfg["systems"]):
        for repetition in range(system["repetitions"]):
            rows.append({
                "id": f"{system['name']}_r{repetition:02d}",
                "system": system["name"], "system_index": index,
                "repetition": repetition,
                "streams": {split: seed_value(cfg["master_seed"], index, repetition, domain)
                            for domain, split in enumerate(
                                ("train", "validation", "calibration", "test"), 1)},
                "geometry_seeds": {str(history): geometry_seed(
                    cfg["master_seed"], index, repetition, history)
                    for history in cfg["histories"]},
                "full_prediction_audit": repetition == cfg["full_prediction_audit_repetition"],
            })
    if len(rows) != cfg["expected_repetitions"]:
        raise AssertionError("The frozen repetition count is inconsistent")
    values = [seed for row in rows for seed in row["streams"].values()]
    if len(values) != len(set(values)):
        raise AssertionError("Repeated split stream")
    return rows


def make_episodes(job, cfg, split):
    load_runtime()
    if job["system"] in ("vanderpol", "duffing"):
        from run_oscillators import oscillator_episodes as generator
    else:
        from certified_forecasting.data import make_episodes as generator
    result = generator(job["system"], cfg["n_" + split], cfg["episode_length"],
                       job["streams"][split])
    if not np.all(np.isfinite(result)) or np.max(np.abs(result)) > 1:
        raise AssertionError("Episode law violates its declared target range")
    return result


def windows(episodes, cfg):
    load_runtime()
    from certified_forecasting.data import histories_targets
    first = max(cfg["histories"]) - 1
    result = {history: histories_targets(episodes, history, cfg["horizons"],
                                          cfg["origin_stride"], first)
              for history in cfg["histories"]}
    first_targets = result[cfg["histories"][0]][1]
    for _, target in result.values():
        np.testing.assert_array_equal(target, first_targets)
    return result


def origins(cfg):
    return np.arange(max(cfg["histories"]) - 1,
                     cfg["episode_length"] - max(cfg["horizons"]), cfg["origin_stride"])


def flat(array):
    return array.reshape(-1, array.shape[-1])


def verify_freeze(output):
    output = Path(output)
    frozen = read_json(output / "freeze.json")
    required = {"config.json", "PROTOCOL.md", "schedule.json", "manifest.json", "preflight.json"}
    if set(frozen.get("files", {})) != required:
        raise RuntimeError("Incomplete frozen-input roster")
    for name, expected in frozen["files"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    if sources() != frozen["source_sha256"]:
        raise RuntimeError("Executable source or protocol changed after freeze")
    cfg, jobs = read_json(output / "config.json"), read_json(output / "schedule.json")
    if cfg != read_json(ROOT / "config.json") or jobs != schedule(cfg):
        raise RuntimeError("Frozen configuration or schedule is rebound")
    if digest(output / "PROTOCOL.md") != digest(ROOT / "PROTOCOL.md"):
        raise RuntimeError("Frozen protocol differs from its source")
    receipt = read_json(output / "preflight.json")
    if receipt.get("passed") is not True or receipt.get("source_sha256") != sources():
        raise RuntimeError("Preflight does not cover the frozen source")
    return cfg, jobs


def verify_seal(output, stage):
    output = Path(output)
    if stage not in STAGE_FILES:
        raise ValueError("Unknown stage seal")
    cfg, jobs = verify_freeze(output)
    seal = read_json(output / (stage + "_seal.json"))
    expected_files = set()
    for job in jobs:
        for name in stage_job_files(stage, job):
            expected_files.add(f"jobs/{job['id']}/{name}")
    if (seal.get("stage") != stage or seal.get("n_jobs") != len(jobs)
            or seal.get("freeze_sha256") != digest(output / "freeze.json")
            or set(seal.get("files", {})) != expected_files):
        raise RuntimeError(f"Incomplete or rebound {stage} seal")
    previous = None if stage == "development" else (
        "development" if stage == "calibration" else "calibration")
    expected_predecessor = None if previous is None else digest(output / (previous + "_seal.json"))
    if seal.get("preceding_seal_sha256") != expected_predecessor:
        raise RuntimeError(f"Broken {stage} predecessor chain")
    if previous is not None:
        verify_seal(output, previous)
    for name, expected in seal["files"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Sealed {stage} artifact changed: {name}")
    return seal


def stage_job_files(stage, job):
    files = list(STAGE_FILES[stage]) + [stage + "_started.json", stage + "_complete.json"]
    if stage in ("calibration", "test") and job["full_prediction_audit"]:
        files.append(stage + "_predictions.npz")
    return files
