#!/usr/bin/env python3
"""Confirm PenDigits scaling over new fitting seeds on the same test cohort.

This scratch investigation does not edit any repository source or result.
Use one new process per mode so the recorded fitted-bank registry is complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "kahm-study"
EXTENSION = REPO / "extensions" / "certified_forecasting"
sys.path[:0] = [str(HERE), str(EXTENSION / "src"), str(EXTENSION / "experiments")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["standard", "minmax", "minmax_clip", "raw"], required=True)
    args = parser.parse_args()
    protocol = HERE / "pendigits_confirmation_protocol.json"
    if not protocol.is_file():
        raise RuntimeError("Freeze pendigits_confirmation_protocol.json before launching confirmation.")
    output = HERE / "confirmation_results" / args.mode / "pendigits_v1"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    launch_file = output.with_name(output.name + "_launch.json")
    if launch_file.exists():
        raise RuntimeError(f"Refusing to overwrite launch record: {launch_file}")

    import scaling_features as adapter
    import run_forecasting
    run_forecasting.NestedSoftFeatures = adapter.make_feature_class(args.mode)
    import run_pendigits

    config = HERE / "confirmation_configs" / "pendigits_v1.json"
    prepared = HERE / "data_cache" / "pendigits" / "pendigits_original.npz"
    sources = sorted(REPO.glob("*.py"))
    sources += sorted((EXTENSION / "src").glob("**/*.py"))
    sources += [EXTENSION / "experiments" / name for name in
                ["run_forecasting.py", "run_pendigits.py", "prepare_pendigits.py"]]
    manifest = {
        "mode": args.mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "protocol_sha256": sha256(protocol),
        "adapter_sha256": sha256(HERE / "scaling_features.py"),
        "wrapper_sha256": sha256(__file__),
        "confirmation_config_sha256": sha256(config),
        "original_config_sha256": sha256(EXTENSION / "configs" / "pendigits_v1.json"),
        "cohort_scope": "New fitting seeds on the same held-out writer cohort; not new-cohort replication.",
        "prepared_data_sha256": sha256(prepared),
        "original_source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sources},
        "change": "Fitting seeds4701-4703 and campaign label; paired arms differ only in KAHM input preprocessing. Original writer partitions, target scaling, candidates, validation selection and evaluation are unchanged.",
        "output": str(output),
        "thread_environment": {key: os.environ[key] for key in
                               ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
    }
    write_json(launch_file, manifest)
    print(f"START mode={args.mode} output={output}", flush=True)
    try:
        summary = run_pendigits.run(config, prepared, output)
    except Exception:
        if output.exists():
            (output / "FAILED_SCALING.txt").write_text(traceback.format_exc())
        raise
    diagnostics = [bank.scaling_metadata() for bank in adapter.FITTED_BANKS]
    write_json(output / "scaling_diagnostics.json", diagnostics)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["fitted_bank_count"] = len(diagnostics)
    manifest["diagnostics_sha256"] = sha256(output / "scaling_diagnostics.json")
    write_json(output / "intervention_manifest.json", manifest)
    kahm = next(row for row in summary["families"] if row["family"] == "kahm_nested")
    print(json.dumps({"mode": args.mode, "kahm_nested": kahm,
                      "elapsed_seconds": summary["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
