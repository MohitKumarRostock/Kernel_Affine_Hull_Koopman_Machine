"""Export a compact controlled archive after every prediction has been audited.

This packaging utility does not fit models or calculate scientific results.
It copies unchanged result JSON and episode-loss NPZ bytes, retains the full
prediction audit attestation and digests, and leaves all full local artifacts
in place.  Predictions and pickle model states are reproducible local artifacts
rather than part of this compact public archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "status": "passed", "mode": "full_prediction_audit", "jobs": 72,
    "per_episode_loss_arrays_checked": 1440,
    "full_prediction_arrays_checked": 1440,
    "historical_standard_mses_checked": 576,
    "cross_arm_raw_arrays_checked": 144,
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_exact(source, destination):
    """Be idempotent for identical bytes and reject silent result replacement."""
    source, destination = Path(source), Path(destination)
    expected = digest(source)
    if destination.exists():
        if not destination.is_file() or digest(destination) != expected:
            raise RuntimeError(f"Refusing to replace different archived bytes: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if digest(destination) != expected:
        raise RuntimeError(f"Exported bytes differ: {destination}")
    return {"sha256": expected, "bytes": source.stat().st_size}


def export(full_root, output, audit_path=None):
    full_root, output = Path(full_root).resolve(), Path(output).resolve()
    audit_path = full_root / "FULL_PREDICTION_AUDIT.json" if audit_path is None else Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text())
    for name, expected in EXPECTED.items():
        if audit.get(name) != expected:
            raise ValueError(f"Full controlled prediction audit is incomplete: {name}")
    if full_root == output or full_root.is_relative_to(output) or output.is_relative_to(full_root):
        raise ValueError("The compact export must be separate from the full run")
    attested_sources = audit["source_files_sha256"]
    manifest = json.loads((full_root / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("n_jobs") != 72 or manifest.get("n_method_records") != 1440:
        raise ValueError("Full run manifest does not declare the complete fixed design")
    if digest(full_root / "records.json") != manifest["records_sha256"]:
        raise ValueError("Method records changed after collection")
    planned = [full_root / "records.json", full_root / "run_manifest.json"]
    for job in manifest["jobs"]:
        folder = full_root / "jobs" / job["job_id"]
        result_path = folder / "result.json"
        losses_path = folder / "unit_losses.npz"
        if digest(result_path) != job["result_sha256"]:
            raise ValueError(f"Job result changed after collection: {job['job_id']}")
        result = json.loads(result_path.read_text())
        if result["status"] != "complete" or len(result["records"]) != 20:
            raise ValueError(f"Incomplete job result: {job['job_id']}")
        for filename, expected in result["artifacts_sha256"].items():
            if digest(folder / filename) != expected:
                raise ValueError(f"Full run artifact changed: {job['job_id']}/{filename}")
        for source in (result_path, losses_path):
            relative = str(source.relative_to(ROOT))
            if attested_sources.get(relative) != digest(source):
                raise ValueError(f"Current result bytes lack a matching prediction audit: {relative}")
            planned.append(source)
    exported = {}
    for source in planned:
        relative = source.relative_to(full_root)
        exported[str(relative)] = copy_exact(source, output / relative)
    exported["FULL_PREDICTION_AUDIT.json"] = copy_exact(audit_path, output / "FULL_PREDICTION_AUDIT.json")
    retention = {
        "format": "controlled_scaling_compact_archive_v1", "status": "complete",
        "jobs": 72, "method_records": 1440, "episode_loss_arrays": 1440,
        "retained": "Unchanged complete per-job result JSON, per-episode loss arrays, method records, run manifest, and independent full-prediction audit",
        "local_reproducible_artifacts": "Full per-origin predictions and fitted geometry pickle states remain in the original local run and are excluded from this compact export",
        "verification_scope": "Compact replay verifies losses and summaries. The retained independent audit attests every original prediction-to-loss calculation; full verification requires the local full artifacts or a rerun of the fixed seeds",
        "full_run_root_relative_to_extension": str(full_root.relative_to(ROOT)),
        "prediction_audit_sha256": digest(audit_path),
        "files": exported,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "RETENTION.json"
    serialized = (json.dumps(retention, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if destination.exists() and destination.read_bytes() != serialized:
        raise RuntimeError("Refusing to replace a different compact archive retention manifest")
    destination.write_bytes(serialized)
    return {"status": "complete", "files": len(exported) + 1,
            "bytes": sum(entry["bytes"] for entry in exported.values()) + len(serialized)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=ROOT / "runs" / "scaling_v1" / "controlled")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "controlled")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.full_root, args.output, args.audit)), flush=True)


if __name__ == "__main__":
    main()
