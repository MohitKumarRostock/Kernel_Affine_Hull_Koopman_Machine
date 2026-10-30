"""Create an internal, auditable protocol freeze before confirmation execution."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    destination = ROOT / "protocol_freeze.json"
    if destination.exists():
        raise RuntimeError("Refusing to overwrite the protocol freeze")
    if list((ROOT / "results/confirmation").glob("*/result.json")):
        raise RuntimeError("Confirmation results already exist")
    if list((ROOT / "results/natural").glob("*/result.json")):
        raise RuntimeError("Natural-data evaluation already exists")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name)
                                for name in ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib")},
                   "blas_threads_during_experiments": 1, "worker_processes": 4}
    (ROOT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    paths = list(ROOT.glob("*.py")) + list((ROOT / "vendor").rglob("*.py"))
    paths += [ROOT / name for name in ("config.json", "PROTOCOL.txt", "ANALYSIS_SPEC.txt",
              "THEORETICAL_NOTE.txt", "DATA_NOTE.txt", "margins.json", "pilot_records.json",
              "environment.json", "upstream_manifest.json", "airfoil_self_noise.dat", "vendor/upstream/LICENSE")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    result = {"version": 1, "created_system_utc": datetime.now(timezone.utc).isoformat(),
              "verified_clock_utc": sys.argv[1] if len(sys.argv) > 1 else None,
              "type": "internal protocol freeze; not externally preregistered",
              "confirmation_results_at_freeze": 0, "natural_results_at_freeze": 0,
              "pilot_outcome_changes": "none; pilots set prespecified practical margins and verified execution",
              "sha256": hashes}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"frozen_files": len(hashes), "freeze_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))
