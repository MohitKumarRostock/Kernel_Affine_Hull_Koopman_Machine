#!/usr/bin/env python3
"""Test reproduction provenance gates using only fabricated execution records."""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import traceback


HERE = Path(__file__).resolve().parent
EXTENSIONS = HERE.parents[2]
SOURCE = EXTENSIONS / "crossfit_residual_reproduction_20261101"
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run():
    auditor_path = SOURCE / "audit/verify_reproduction.py"
    auditor = load("reproduction_provenance_fixture_auditor", auditor_path)
    sys.path.insert(0, str(SOURCE / "code"))
    pipeline = load("crossfit_production_audit_pipeline", SOURCE / "code/pipeline.py")
    config = json.loads((SOURCE / "config.json").read_text())
    original_freeze = json.loads((EXTENSIONS / "crossfit_residual_v2/freeze.json").read_text())
    metadata = config["computational_reproduction"]
    base = datetime.fromisoformat(metadata["created_utc"])
    at = lambda n: (base + timedelta(seconds=n)).isoformat()
    freeze = {"data_generated": False, "D_generated": False,
              "numerical_amendment": None, "public_preregistration": False,
              "created_utc": at(1),
              "files": dict(metadata["original_source_files"]),
              "inherited_files": original_freeze["inherited_files"],
              "previous_extension_files": original_freeze["previous_extension_files"]}
    freeze["files"].update({name: value["sha256"]
                             for name, value in metadata["historical_documents"].items()})
    freeze["files"]["audit/verify_reproduction.py"] = sha(auditor_path)
    abc = {"schema": "crossfit-residual-abc", "D_generated": False, "created_utc": at(2)}
    results = []
    with tempfile.TemporaryDirectory(prefix="kahm-provenance-unit-") as directory:
        root = Path(directory)
        seeds = pipeline.seed_records(config)
        write(root / "prepared/seed_manifest.json", {"records": seeds, "seed_count": len(seeds),
                                                     "D_streams_reserved_but_not_generated": True})
        for name, index in (("FIT_LOCK.json", 3), ("prepared/d_manifest.json", 4),
                            ("PREDICTION_LOCK.json", 5), ("TARGET_RELEASE.json", 6)):
            write(root / name, {"created_utc": at(index)})
        audit = auditor.Audit()
        receipt = auditor.reproduction_provenance(root, SOURCE, config, freeze, abc, audit)
        assert receipt["original_completed_run_authenticated"] is False
        assert receipt["additional_independent_repetitions"] == 0
        results.append({"case": "consistent_manufactured_provenance", "status": "passed",
                        "assertions": audit.counts["assertions"]})

        def rejected(label, cfg=None, fz=None, manifest=None):
            try:
                auditor.reproduction_provenance(root, SOURCE, cfg or config, fz or freeze,
                                               manifest or abc, auditor.Audit())
            except AssertionError as error:
                results.append({"case": label, "status": "expected_rejection", "message": str(error)})
            else:
                raise AssertionError(label + " unexpectedly accepted")

        changed = copy.deepcopy(config)
        changed["model"]["alpha_grid"][0] *= 2
        rejected("scientific_hyperparameter_change", cfg=changed)
        changed = copy.deepcopy(config)
        changed["master_seed"] += 1
        rejected("cohort_seed_change", cfg=changed)
        changed = copy.deepcopy(config)
        changed["computational_reproduction"]["original_completed_run_authenticated"] = True
        rejected("false_original_authentication", cfg=changed)
        changed = copy.deepcopy(config)
        changed["computational_reproduction"]["additional_independent_repetitions"] = 24
        rejected("false_additional_independent_repetitions", cfg=changed)
        changed = copy.deepcopy(config)
        changed["computational_reproduction"]["historical_outputs_known_before_reproduction"] = False
        rejected("concealed_prior_output_knowledge", cfg=changed)
        changed = copy.deepcopy(config)
        changed["computational_reproduction"]["original_freeze_sha256"] = "0" * 64
        rejected("wrong_original_freeze_identity", cfg=changed)
        changed = copy.deepcopy(freeze)
        changed["files"]["analysis_code/analyze.py"] = "0" * 64
        rejected("changed_analyzer_in_fresh_freeze", fz=changed)
        changed = copy.deepcopy(freeze)
        changed["inherited_files"] = {}
        rejected("changed_inherited_source_roster", fz=changed)
        changed = dict(abc, reuse={"exact_original_bytes": True})
        rejected("false_original_ABC_reuse", manifest=changed)
        write(root / "PREDICTION_LOCK.json", {"created_utc": at(7)})
        rejected("target_release_before_prediction_seal")
        write(root / "PREDICTION_LOCK.json", {"created_utc": at(5)})
        bad_seed = copy.deepcopy(seeds)
        bad_seed[0]["seed"] += 1
        write(root / "prepared/seed_manifest.json", {"records": bad_seed, "seed_count": len(seeds),
                                                     "D_streams_reserved_but_not_generated": True})
        rejected("changed_recorded_seed_stream")

    old_tree = ast.parse((HERE / "verify_results.py").read_text())
    new_tree = ast.parse(auditor_path.read_text())
    old_functions = {node.name: ast.dump(node, include_attributes=False) for node in old_tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    new_functions = {node.name: ast.dump(node, include_attributes=False) for node in new_tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    preserved = sorted(set(old_functions) - {"amendment_provenance", "run"})
    assert all(old_functions[name] == new_functions[name] for name in preserved)
    assert set(new_functions) - set(old_functions) == {"reproduction_provenance"}
    assert set(old_functions) - set(new_functions) == {"amendment_provenance"}
    results.append({"case": "numerical_and_selection_functions_unchanged", "status": "passed",
                    "AST_identical_functions_and_classes": preserved})
    return {"status": "passed", "scope": "Provenance unit fixture with fabricated execution timestamps and manifests",
            "completed_utc": datetime.now(timezone.utc).isoformat(), "cases": results,
            "case_count": len(results), "scientific_data_generated": False,
            "production_results_audited": False, "fabricated_records_retained_as_production": False,
            "source_sha256": {"auditor": sha(auditor_path), "fixture": sha(__file__),
                               "source_config": sha(SOURCE / "config.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve fixture receipts: " + str(args.output))
    try:
        receipt = run()
    except Exception:
        receipt = {"status": "failed", "traceback": traceback.format_exc(),
                   "scope": "Manufactured provenance unit fixture", "production_results_audited": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "case_count": receipt.get("case_count"),
                      "receipt": str(args.output), "traceback": receipt.get("traceback")}, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
