"""Gate-enforced loading of only the retained Step36 decision-cohort inputs.

The model-facing loader accepts no path arguments. It reads the exact fixed
input whitelist after a protocol/source seal and matching independent PASS.
Training/test features, raw count archives, representation payloads and full
decoder payloads are not present in this input tree and are never followed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from prepare_inputs37 import ASSURANCE, HEAD_FIELDS, RECIPES, TASKS, map_names, extract_candidate_prefix, minimal_heads


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "inputs/source_manifest37.json"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def event(name, **fields):
    path = ROOT/"output/access_and_execution37.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if path.exists():
        lines = path.read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            previous = last.pop("event_sha256")
            if hashlib.sha256(canonical_bytes(last)).hexdigest() != previous:
                raise RuntimeError("Existing Step37 event-chain tail changed")
    record = {"event": name, "utc": datetime.now(timezone.utc).isoformat(),
              "monotonic_ns": time.monotonic_ns(), "previous_event_sha256": previous, **fields}
    record["event_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    with path.open("ab") as handle:
        handle.write(canonical_bytes(record)); handle.flush()
    return record


def _relative(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError("Unsafe frozen input path")
    return path


def _verify_bound_files(seal):
    bounds = seal.get("bound_files")
    if not isinstance(bounds, dict) or not bounds:
        raise RuntimeError("Frozen seal has no bound files")
    for relative, expected in bounds.items():
        path = ROOT/_relative(relative)
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError("Frozen protocol/source/input changed: " + relative)


def verify_input_manifest():
    """Metadata and opaque hash checks only; safe to call before numerical access."""
    manifest = json.loads((ROOT/MANIFEST_PATH).read_text())
    if (manifest["schema"] != "step37-decision-input-manifest-v1"
            or manifest["task_order"] != list(TASKS) or manifest["map_order"] != list(map_names())
            or manifest["class"] != {"kappa": "2", "prediction_interval": ["0", "1"], "assurance": ASSURANCE}
            or set(manifest["tasks"]) != set(TASKS)):
        raise RuntimeError("Frozen decision/class roster differs")
    items = manifest["files"]
    names = [item["path"] for item in items]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate input whitelist path")
    expected = set(names) | {MANIFEST_PATH}
    actual = {str(p.relative_to(ROOT)) for p in (ROOT/"inputs").rglob("*") if p.is_file()}
    if actual != expected:
        raise RuntimeError("Unwhitelisted or missing Step37 input file")
    allowed_roles = {"prior_protocol", "prior_protocol_seal", "prior_model_seal", "prior_independent_audit",
                     "prior_public_source_metadata", "decision_cohort_opaque", "training_derived_threshold_metadata",
                     "frozen_map_identity_metadata", "pretest_exhaustive_record_opaque", "decision_features_opaque"}
    for item in items:
        relative = _relative(item["path"])
        if relative.parts[0] != "inputs" or item["role"] not in allowed_roles:
            raise RuntimeError("Input role or path lies outside the approved scope")
        path = ROOT/relative
        if (sha_file(path) != item["sha256"] or item["source_sha256"] != item["sha256"]
                or path.stat().st_size != item["bytes"] or item["byte_identical_copy"] is not True):
            raise RuntimeError("Staged input identity differs")
    registry = {item["path"]: item for item in items}
    prior_audit = json.loads((ROOT/"inputs/prior/postrun_audit36.json").read_text())
    prior_model = json.loads((ROOT/"inputs/prior/model_seal36.json").read_text())
    prior_protocol_seal = json.loads((ROOT/"inputs/prior/protocol_seal36.json").read_text())
    prior_protocol = json.loads((ROOT/"inputs/prior/protocol36.json").read_text())
    prior = manifest["prior"]
    if (prior_audit["status"] != "PASS" or prior_audit["result_sha256"] != prior["result_sha256"]
            or prior_audit["protocol_seal_sha256"] != prior["protocol_seal_sha256"]
            or prior_audit["model_seal_sha256"] != prior["model_seal_sha256"]
            or prior_model["protocol_seal_sha256"] != prior["protocol_seal_sha256"]):
        raise RuntimeError("Prior independent evidence bindings differ")
    if (prior_protocol["tasks"] != list(TASKS) or prior_protocol["class"]["kappa"] != "2"
            or prior_protocol["class"]["prediction_interval"] != ["0", "1"]
            or prior_protocol["class"]["assurance"] != ASSURANCE):
        raise RuntimeError("Prior declared decoder class differs")
    for key, relative in (("protocol_sha256", "inputs/prior/protocol36.json"),
                          ("protocol_seal_sha256", "inputs/prior/protocol_seal36.json"),
                          ("model_seal_sha256", "inputs/prior/model_seal36.json"),
                          ("postrun_audit_sha256", "inputs/prior/postrun_audit36.json")):
        if registry[relative]["sha256"] != prior[key]:
            raise RuntimeError("Prior copied provenance digest differs")
    for item in items:
        if item["bound_in_prior_pretest_model_seal"] and prior_model["bound_files"].get(item["source_relative_path"]) != item["sha256"]:
            raise RuntimeError("Copied decision input is not the prior pre-test sealed file")
        if item["bound_in_prior_protocol_seal"] and prior_protocol_seal["bound_files"].get(item["source_relative_path"]) != item["sha256"]:
            raise RuntimeError("Copied prior protocol input identity differs")
    for task in TASKS:
        spec = manifest["tasks"][task]
        maps = spec["maps"]
        if [item["map"] for item in maps] != list(map_names()) or len(maps) != 18:
            raise RuntimeError("A fixed map was omitted or substituted")
        for key, role in (("cohort_path", "decision_cohort_opaque"),
                          ("training_plan_path", "training_derived_threshold_metadata"),
                          ("map_manifest_path", "frozen_map_identity_metadata"),
                          ("exhaustive_path", "pretest_exhaustive_record_opaque")):
            if spec[key] not in registry or registry[spec[key]]["role"] != role:
                raise RuntimeError("Task input path does not match its declared role")
        if int(spec["expected_n"]) <= 0 or F(spec["epsilon"]) < 0:
            raise RuntimeError("Invalid decision cohort or tolerance")
        for item in maps:
            if (item["feature_path"] not in registry
                    or registry[item["feature_path"]]["role"] != "decision_features_opaque"
                    or item["feature_sha256"] != registry[item["feature_path"]]["sha256"]
                    or set(item["heads"]) != set(RECIPES)):
                raise RuntimeError("Map input identity or head roster differs")
            for recipe in RECIPES:
                row = item["heads"][recipe]
                if (set(row) != set(HEAD_FIELDS) or row["candidate_id"] != item["map"]+"::"+recipe
                        or row["task"] != task or row["map"] != item["map"] or row["recipe"] != recipe
                        or row["feature_id"] != item["feature_id"] or row["class_id"] != item["class_id"]
                        or row["feature_width"] != item["dimension"] or row["assurance"] != ASSURANCE):
                    raise RuntimeError("Prior candidate is bound to a different decision class")
                for hash_key, path_key in (("head_sha256", "head_path"), ("fit_sha256", "fit_receipt_path"),
                                           ("decision_prediction_sha256", "decision_prediction_path")):
                    if prior_model["bound_files"].get(row[path_key]) != row[hash_key]:
                        raise RuntimeError("Prior head/prediction proof hash is outside the prior pre-test seal")
    return manifest


def verify_protocol_seal():
    path = ROOT/"protocol_seal37.json"
    if not path.is_file():
        raise RuntimeError("Step37 protocol seal required before decision numerical access")
    seal = json.loads(path.read_text())
    _verify_bound_files(seal)
    required = {"protocol37.json", "source/data37.py", "source/prepare_inputs37.py", MANIFEST_PATH}
    if not required <= set(seal["bound_files"]):
        raise RuntimeError("Protocol seal omits required loader/source/input metadata")
    manifest = verify_input_manifest()
    if not {item["path"] for item in manifest["files"]} <= set(seal["bound_files"]):
        raise RuntimeError("Protocol seal omits a whitelisted decision input")
    gate_path = ROOT/"audit/preexecution_gate37.json"
    gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
    if gate.get("status") != "PASS" or gate.get("protocol_seal_sha256") != sha_file(path):
        raise RuntimeError("Independent Step37 preexecution PASS required before numerical access")
    protocol = json.loads((ROOT/"protocol37.json").read_text())
    if protocol.get("step") != 37 or protocol.get("tasks") != list(TASKS):
        raise RuntimeError("Sealed protocol is not the declared Step37 three-target diagnostic")
    return protocol


def _load_task(task):
    # Internal entry point also enforces the gate: callers cannot bypass it by
    # invoking a private helper before load_inputs().
    verify_protocol_seal()
    manifest = verify_input_manifest()
    spec = manifest["tasks"][task]
    cohort = json.loads((ROOT/spec["cohort_path"]).read_text())
    if cohort["name"] != "decision" or cohort["target"] != task:
        raise RuntimeError("Attempted to load a non-decision cohort")
    ids = cohort["ids"]
    if len(ids) != spec["expected_n"] or len(set(ids)) != len(ids) or any(not i.startswith("decision:") for i in ids):
        raise RuntimeError("Decision identifier roster differs")
    y_strings = cohort["y"]
    if len(y_strings) != len(ids) or any(not isinstance(value, str) for value in y_strings):
        raise RuntimeError("Invalid exact decision-target roster")
    y = [F(value) for value in y_strings]
    if list(map(str, y)) != y_strings:
        raise RuntimeError("Decision targets are not canonical exact rationals")
    # raw_targets, persistence, lag_matrix_path and historical lag values are not
    # inputs to this diagnostic; in particular the lag matrix is never opened.
    plan = json.loads((ROOT/spec["training_plan_path"]).read_text())
    epsilon = F(spec["epsilon"])
    if plan["epsilon"] != spec["epsilon"] or plan["epsilon_physical_mse"] != spec["epsilon_physical_mse"]:
        raise RuntimeError("Training-derived threshold changed")
    originals = {row["candidate_id"]: row for row in minimal_heads(extract_candidate_prefix(ROOT/spec["exhaustive_path"]))}
    maps = {}
    for item in spec["maps"]:
        if any(item["heads"][recipe] != originals[item["map"]+"::"+recipe] for recipe in RECIPES):
            raise RuntimeError("Extracted candidate metadata differs from prior sealed exhaustive record")
        z = np.load(ROOT/item["feature_path"], allow_pickle=False)
        if z.dtype != np.float64 or z.shape != (len(ids), item["dimension"]) or not np.isfinite(z).all():
            raise RuntimeError("Stored decision feature matrix is invalid")
        z.flags.writeable = False
        maps[item["map"]] = {"map": item["map"], "z": z, "feature_id": item["feature_id"],
            "class_id": item["class_id"], "dimension": item["dimension"], "family": item["family"],
            "scaling": item["scaling"], "heads": item["heads"],
            "feature_path": item["feature_path"], "feature_sha256": item["feature_sha256"],
            "map_payload_sha256": item["map_payload_sha256"]}
    return {"task": task, "cohort": {"name": "decision", "target": task, "ids": ids, "y": y,
                "target_timestamps": cohort.get("target_timestamps"), "n": len(ids),
                "path": spec["cohort_path"], "sha256": sha_file(ROOT/spec["cohort_path"])},
            "epsilon": epsilon, "epsilon_physical_mse": F(spec["epsilon_physical_mse"]),
            "physical_mse_multiplier": None if epsilon == 0 else F(spec["epsilon_physical_mse"])/epsilon,
            "maps": maps}


def load_inputs():
    verify_protocol_seal()
    event("decision_input_access_start", protocol_seal_sha256=sha_file(ROOT/"protocol_seal37.json"),
          input_manifest_sha256=sha_file(ROOT/MANIFEST_PATH), task_order=list(TASKS),
          input_role="previously_opened_step36_decision_only", raw_archive_accessed=False,
          training_or_test_feature_arrays_accessed=False, reserved_patient_arrays_accessed=False)
    tasks = {task: _load_task(task) for task in TASKS}
    reference_ids = tasks[TASKS[0]]["cohort"]["ids"]
    if any(tasks[task]["cohort"]["ids"] != reference_ids for task in TASKS):
        raise RuntimeError("The three decision target rosters are not identical")
    receipt = {"schema": "step37-gated-decision-input-access-v1", "status": "PASS",
        "protocol_seal_sha256": sha_file(ROOT/"protocol_seal37.json"),
        "input_manifest_sha256": sha_file(ROOT/MANIFEST_PATH),
        "tasks": {task: {"decision_n": tasks[task]["cohort"]["n"], "maps": len(tasks[task]["maps"]),
                          "epsilon": str(tasks[task]["epsilon"]),
                          "cohort_sha256": tasks[task]["cohort"]["sha256"]} for task in TASKS},
        "decision_feature_arrays_loaded": sum(len(item["maps"]) for item in tasks.values()),
        "common_decision_roster": True, "targets_clipped": False,
        "training_or_test_feature_arrays_accessed": False, "raw_source_archive_accessed": False,
        "representation_or_decoder_payloads_loaded": False, "new_prediction_model_fits": 0,
        "reserved_patient_arrays_accessed": 0, "new_unseen_dataset_claim": False,
        "diagnostic_scope": "exploratory_on_previously_opened_decision_data"}
    path = ROOT/"output/input_access37.json"
    write_json(path, receipt)
    event("decision_input_access_complete", receipt_sha256=sha_file(path),
          decision_feature_arrays_loaded=receipt["decision_feature_arrays_loaded"],
          common_decision_n=len(reference_ids), training_or_test_feature_arrays_accessed=False)
    return tasks
