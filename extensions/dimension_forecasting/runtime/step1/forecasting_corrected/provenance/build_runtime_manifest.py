#!/usr/bin/env python3
"""Record the completed derivative; verification never invokes this writer."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(directory):
    return [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]


if __name__ == "__main__":
    record = {
        "schema": "kahm-forecasting-dimension-correction-v1",
        "source_commit": "7dda356138675934aab5e327ec667bde0a5eda79",
        "native_source_commit": "d6d74d1953a4310b73065a556df816e5ab5e3ebc",
        "source_vendor_manifest_sha256": sha(ROOT / "vendor/VENDOR_MANIFEST.json"),
        "source_current_derivative_manifest_sha256": sha(
            ROOT / "reference_current/code/vendor_geometry/DERIVATIVE_MANIFEST.json"),
        "source_copy_manifest_sha256": sha(ROOT / "provenance/SOURCE_COPY_MANIFEST.json"),
        "shared_dimension_policy_sha256": sha(ROOT.parent / "policy/kahm_dimension_policy.py"),
        "files": inventory(ROOT / "code"),
        "historical_reference_files": inventory(ROOT / "reference_current"),
        "default_requested_cap": 20,
        "default_policy": "native",
        "historical_ablation": {"rank": 2, "dimension_policy": "legacy_normal_departure"},
        "scope": "Isolated construction correction and fabricated integration fixtures only; no empirical study rerun.",
    }
    (ROOT / "provenance/FORECASTING_RUNTIME_MANIFEST.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runtime_files": len(record["files"]),
                      "historical_reference_files": len(record["historical_reference_files"])}))
