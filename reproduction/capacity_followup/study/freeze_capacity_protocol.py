"""Freeze the data-motivated second-stage protocol before fresh draws."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from run_study import verify_freeze

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    verify_freeze()
    path = ROOT / "followup_freeze.json"
    if path.exists() or list((ROOT / "results_capacity").glob("*/result.json")):
        raise RuntimeError("A follow-up freeze or completed result already exists")
    names = ("freeze_capacity_protocol.py", "followup_capacity.py", "followup_config.json",
             "FOLLOWUP_PROTOCOL.txt", "CAPACITY_ANALYSIS_SPEC.txt", "audit_capacity.py",
             "capacity_margins.json", "protocol_freeze.json", "records.json")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    result = {"version": 1, "created_system_utc": datetime.now(timezone.utc).isoformat(),
              "type": "data-motivated follow-up frozen before fresh stage2 data; not external preregistration",
              "stage2_completed_jobs_at_freeze": 0,
              "original_protocol_freeze_sha256": hashes["protocol_freeze.json"],
              "pre_data_analysis_amendment": "Added secondary validation selector after numerical margins were frozen; numerical margins unchanged; final analysis source hashes pinned here.",
              "sha256": hashes}
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"frozen_files": len(hashes), "freeze_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
