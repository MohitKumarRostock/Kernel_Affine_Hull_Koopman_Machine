"""Accept an exact numerical replay to repair a damaged score archive.

This does not fit or select models. Both original and replay records must
already exist, and all non-timing numerical records must agree exactly.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()
                if k not in {"elapsed_seconds", "train_seconds"}}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original", type=Path)
    p.add_argument("replay", type=Path)
    a = p.parse_args()
    checks = {}
    for name in ["results.json", "frozen_candidates.json"]:
        checks[name] = clean(json.loads((a.original / name).read_text())) == clean(
            json.loads((a.replay / name).read_text()))
    old_manifest = json.loads((a.original / "intervention_manifest.json").read_text())
    new_manifest = json.loads((a.replay / "intervention_manifest.json").read_text())
    for key in ["mode", "configuration", "config_sha256", "protocol_sha256",
                "harness_sha256", "scaler_sha256", "source_sha256"]:
        checks[key] = old_manifest[key] == new_manifest[key]
    expected = 3 * json.loads((a.original / "results.json").read_text())["n_candidates"]
    replacement = a.replay / "episode_scores.npz"
    with zipfile.ZipFile(replacement) as z:
        checks["replay_zip_entries"] = len(z.infolist()) == expected
        checks["replay_crc"] = z.testzip() is None
    if not all(checks.values()):
        raise RuntimeError(f"Exact replay verification failed: {checks}")
    original = a.original / "episode_scores.npz"
    corrupt = a.original / "episode_scores.corrupt.npz"
    if corrupt.exists() or (a.original / "artifact_repair.json").exists():
        raise RuntimeError("Existing repair record; refusing overwrite")
    before = {"sha256": digest(original), "bytes": original.stat().st_size}
    temporary = a.original / "episode_scores.repair.tmp.npz"
    with replacement.open("rb") as source, temporary.open("xb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise RuntimeError("Copied replay archive failed CRC")
    original.rename(corrupt)
    temporary.replace(original)
    record = {
        "reason": "Truncated score archive detected by independent analysis despite normally completed numerical run.",
        "repair": "Exact unchanged source/configuration/seed replay; all original non-timing result and frozen-candidate fields agree.",
        "original_corrupt_archive": str(corrupt),
        "corrupt_sha256": before["sha256"], "corrupt_bytes": before["bytes"],
        "repaired_sha256": digest(original), "repaired_bytes": original.stat().st_size,
        "replay_directory": str(a.replay), "checks": checks,
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    (a.original / "artifact_repair.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
