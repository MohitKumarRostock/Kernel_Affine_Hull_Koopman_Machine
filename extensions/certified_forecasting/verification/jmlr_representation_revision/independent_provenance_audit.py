#!/usr/bin/env python3
"""Check the disclosed ancillary-log exception without importing project code."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def audit(results, output):
    require(not output.exists(), "Refusing to overwrite a provenance audit")
    root = Path(__file__).resolve().parents[2]
    repository = root.parents[1]
    freeze = read_json(results/"protocol_freeze.json")
    summary = read_json(results/"run_summary.json")
    incident = read_json(results/"provenance_incident.json")
    manifest = read_json(results/"source_manifest.json")["files"]
    allowed = "extensions/certified_forecasting/experiments/learned_kahm_hierarchy/development_benchmark.json"
    require(incident["changed_path"] == allowed, "Unexpected provenance exception")
    require(summary["source_unchanged"] is False and summary["executable_sources_unchanged"] is True,
            "Summary obscures the distinction between log and executable changes")
    require(summary["ancillary_source_changes"] == [allowed], "Summary exception list differs")
    for key, name in [("protocol_sha256", "protocol.json"),
                      ("source_snapshot_sha256", "source_snapshot.tar.gz"),
                      ("source_manifest_sha256", "source_manifest.json")]:
        require(digest(results/name) == freeze[key], "Frozen artifact changed: "+name)
    require(digest(root/freeze["models_path"]) == freeze["models_sha256"], "Frozen banks changed")
    changes, references = [], []
    with tarfile.open(results/"source_snapshot.tar.gz", "r:gz") as archive:
        for item in manifest:
            content = archive.extractfile(item["path"]).read()
            require(hashlib.sha256(content).hexdigest() == item["sha256"], "Source snapshot content differs")
            if digest(repository/item["path"]) != item["sha256"]:
                changes.append(item["path"])
            if item["path"].endswith(".py"):
                parsed = ast.parse(content)
                for node in ast.walk(parsed):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for value in ast.walk(node):
                            if isinstance(value, ast.Constant) and value.value == "development_benchmark.json":
                                references.append({"path": item["path"], "function": node.name})
        frozen_log = archive.extractfile(allowed).read()
    require(changes == [allowed], "Other execution dependencies changed: "+repr(changes))
    require(references == [{"path": "extensions/certified_forecasting/experiments/learned_kahm_hierarchy/run.py",
                            "function": "source_paths"}], "Unexpected executable reference to auxiliary log")
    preserved_frozen = results/"ancillary_log_incident/frozen_development_benchmark.json"
    preserved_completed = results/"ancillary_log_incident/completed_development_benchmark.json"
    require(preserved_frozen.read_bytes() == frozen_log, "Preserved frozen log differs from snapshot")
    require(digest(preserved_frozen) == incident["frozen_sha256"], "Frozen log hash differs")
    require(digest(preserved_completed) == incident["completed_sha256"] == digest(repository/allowed),
            "Completed log hash differs")
    before, after = json.loads(frozen_log), read_json(preserved_completed)
    require(len(before["rows"]) == incident["frozen_rows"] == 20, "Frozen log row count differs")
    require(len(after["rows"]) == incident["completed_rows"] == 28, "Completed log row count differs")
    require(after["rows"][:20] == before["rows"], "Existing development rows were changed")
    benchmark_seeds = sorted({row["seed"] for row in before["rows"]})
    require(benchmark_seeds == [202610291701, 202610291702], "Frozen benchmark seed disclosure differs")
    for name, sha in incident["pre_finalization_result_hashes"].items():
        require(digest(results/name) == sha, "An already completed output changed: "+name)
    finalizer = root/"experiments/learned_kahm_hierarchy/finalize_saved_results.py"
    require(digest(finalizer) == incident["finalizer_sha256"], "Finalizer source differs")
    for field in ["confirmation_rerun", "confirmation_data_regenerated",
                  "model_or_endpoint_reselection", "frozen_snapshot_modified"]:
        require(incident[field] is False, "Incident reports a substantive rerun or revision")
    report = {"passed": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "frozen_files_preserved": True, "all_executable_dependencies_match_frozen_snapshot": True,
              "source_changes": changes, "auxiliary_log_executable_references": references,
              "preserved_frozen_development_rows": 20, "preserved_completed_development_rows": 28,
              "frozen_log_already_contains_both_development_seeds": benchmark_seeds,
              "all_pre_finalization_result_hashes_unchanged": True,
              "scope": "Independent hash, archive, AST-reference, log-prefix, and output-preservation checks; numerical endpoints are reconstructed in independent_hierarchy_audit.json.",
              "limitations": ["File hashes and timestamps are provenance records, not external attestations.",
                  "The original frozen narrative incorrectly reported early termination of the development benchmark; the exception record preserves the correction.",
                  "Concurrent development execution can affect timings, so these are not isolated runtime benchmarks."],
              "incident_sha256": digest(results/"provenance_incident.json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.results, args.output)
