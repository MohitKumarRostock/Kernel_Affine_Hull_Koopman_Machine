"""Prepare opaque, whitelisted Step36 decision inputs for the Step37 diagnostic.

This preparation step never loads a feature array, decodes a decision-cohort
target payload, opens a raw archive, or follows a training/test data path.
Only pre-test candidate summaries, already-reported thresholds and provenance
metadata are interpreted. NumPy headers are metadata; array payloads are copied
and hashed without numerical interpretation. Preparation is forbidden after the
Step37 protocol seal exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent / "step36_triggered_search"
TASKS = ("casual", "registered", "cnt")
RECIPES = ("affine", "extension32")
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"
EXPECTED_PRIOR = {
    "result_sha256": "d52cc6ba7412ae507df473b3d4a4da9216e661470ac268c7ad69f1ce7d7d8471",
    "protocol_sha256": "4377b08b9653ea36d46afa77b1957ae4952bba83463272286e7fa9f4e572450c",
    "protocol_seal_sha256": "5e095bdd88140b8e80ac3b918661de209aeef7a9377118715cdd8ebe1635074a",
    "model_seal_sha256": "df0a1a36a80469f465f86cadd97a90af4f3b3060ed8b9256b41c161158010c9e",
    "postrun_audit_sha256": "be75862c90d9031d9ebf1bccc1374f34fe53e81815cb6bec236deace5bc23757",
}
HEAD_FIELDS = (
    "candidate_id", "task", "map", "recipe", "feature_id", "class_id", "head_id",
    "feature_width", "family", "scaling", "decision_risk", "deployment_payload_bytes",
    "head_sha256", "fit_sha256", "decision_prediction_sha256", "fit_status",
    "projection_applied", "assurance", "head_path", "fit_receipt_path",
    "decision_prediction_path",
)


def map_names():
    names = ["raw_" + scaling for scaling in ("minmax01", "standard")]
    names += ["nystrom_" + scaling + "_w" + str(width)
              for scaling in ("minmax01", "standard") for width in (8, 24)]
    names += ["kahm_" + scaling + "_cap" + str(cap) + "_p" + str(prefix)
              for scaling in ("minmax01", "standard") for cap in (4, 12, 20) for prefix in (1, 2)]
    return tuple(sorted(names))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def extract_candidate_prefix(path):
    """Decode only the first, pre-test candidate_rows value of canonical JSON.

    The rest of the exhaustive record includes its range-floor target receipt.
    It is deliberately not passed through a JSON parser in this preparation.
    """
    text = Path(path).read_bytes().decode("utf-8")
    prefix = '{"candidate_rows":'
    if not text.startswith(prefix):
        raise RuntimeError("Unexpected canonical exhaustive-record prefix")
    rows, end = json.JSONDecoder().raw_decode(text, len(prefix))
    if not isinstance(rows, list) or text[end:end+1] != ",":
        raise RuntimeError("Malformed exhaustive candidate prefix")
    if any(any(key.startswith("test") for key in row) for row in rows):
        raise RuntimeError("A pre-test candidate row unexpectedly contains test outcomes")
    return rows


def minimal_heads(rows):
    result = []
    for row in rows:
        if not isinstance(row, dict) or not set(HEAD_FIELDS) <= set(row):
            raise RuntimeError("Incomplete prior candidate metadata")
        item = {key: row[key] for key in HEAD_FIELDS}
        low, high, width = (Fraction(item["decision_risk"][key]) for key in ("lower", "upper", "width"))
        if not 0 <= low <= high or width != high-low:
            raise RuntimeError("Invalid previously audited risk interval")
        if item["assurance"] != ASSURANCE:
            raise RuntimeError("Prior assurance differs")
        result.append(item)
    return result


def numpy_header(path):
    with Path(path).open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise RuntimeError("Unsupported prior NumPy file version")
        payload_offset = handle.tell()
    return {"npy_version": list(version), "shape": list(shape), "fortran_order": bool(fortran),
            "dtype": np.dtype(dtype).str, "payload_offset": payload_offset,
            "payload_values_parsed": False}


def prepare(source_root=SOURCE_ROOT):
    source_root = Path(source_root).resolve()
    if (ROOT/"protocol_seal37.json").exists():
        raise RuntimeError("Step37 is sealed; input preparation cannot mutate frozen inputs")
    manifest_path = ROOT/"inputs/source_manifest37.json"
    if manifest_path.exists():
        raise RuntimeError("Step37 input manifest already exists; do not silently replace it")
    prior_paths = {
        "result_sha256": "output/result36.json", "protocol_sha256": "protocol36.json",
        "protocol_seal_sha256": "protocol_seal36.json", "model_seal_sha256": "output/model_seal36.json",
        "postrun_audit_sha256": "audit/postrun_audit36.json",
    }
    prior = {key: sha_file(source_root/relative) for key, relative in prior_paths.items()}
    if prior != EXPECTED_PRIOR:
        raise RuntimeError("Step36 retained evidence identity differs")
    # Hashing the final result is opaque; none of its test metrics are parsed.
    audit = json.loads((source_root/"audit/postrun_audit36.json").read_text())
    model_seal = json.loads((source_root/"output/model_seal36.json").read_text())
    protocol_seal = json.loads((source_root/"protocol_seal36.json").read_text())
    protocol = json.loads((source_root/"protocol36.json").read_text())
    if (audit["status"] != "PASS" or audit["result_sha256"] != prior["result_sha256"]
            or audit["protocol_seal_sha256"] != prior["protocol_seal_sha256"]
            or audit["model_seal_sha256"] != prior["model_seal_sha256"]
            or model_seal["protocol_seal_sha256"] != prior["protocol_seal_sha256"]):
        raise RuntimeError("Step36 independent audit and seals disagree")
    if (protocol["tasks"] != list(TASKS) or protocol["class"]["kappa"] != "2"
            or protocol["class"]["prediction_interval"] != ["0", "1"]
            or protocol["class"]["assurance"] != ASSURANCE):
        raise RuntimeError("Step36 target or class scope differs")
    files = []

    def copy(relative, destination, role, *, pretest_bound=False, protocol_bound=False):
        source = source_root/relative
        expected = sha_file(source)
        if pretest_bound and model_seal["bound_files"].get(relative) != expected:
            raise RuntimeError("Decision input was not bound in prior pre-test model seal: " + relative)
        if protocol_bound and protocol_seal["bound_files"].get(relative) != expected:
            raise RuntimeError("Prior protocol input binding differs: " + relative)
        local = ROOT/destination
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and sha_file(local) != expected:
            raise RuntimeError("An existing staged input differs: " + destination)
        local.write_bytes(source.read_bytes())
        if sha_file(local) != expected:
            raise RuntimeError("Opaque copied input differs")
        item = {"path": destination, "sha256": expected, "bytes": local.stat().st_size,
                "role": role, "source_relative_path": relative, "source_sha256": expected,
                "byte_identical_copy": True, "bound_in_prior_pretest_model_seal": pretest_bound,
                "bound_in_prior_protocol_seal": protocol_bound}
        files.append(item)
        return item

    provenance_copies = (
        ("protocol36.json", "inputs/prior/protocol36.json", "prior_protocol", False, True),
        ("protocol_seal36.json", "inputs/prior/protocol_seal36.json", "prior_protocol_seal", False, False),
        ("output/model_seal36.json", "inputs/prior/model_seal36.json", "prior_model_seal", False, False),
        ("audit/postrun_audit36.json", "inputs/prior/postrun_audit36.json", "prior_independent_audit", False, False),
        ("inputs/public_source36.json", "inputs/prior/public_source36.json", "prior_public_source_metadata", False, True),
    )
    for relative, destination, role, pretest_bound, protocol_bound in provenance_copies:
        copy(relative, destination, role, pretest_bound=pretest_bound, protocol_bound=protocol_bound)

    tasks = {}
    for task in TASKS:
        cohort_path = f"inputs/decision/{task}/cohort.json"
        plan_path = f"inputs/decision/{task}/training_plan36.json"
        map_manifest_path = f"inputs/decision/{task}/map_manifest36.json"
        exhaustive_path = f"inputs/decision/{task}/exhaustive_r00.json"
        copy(f"output/cohorts/{task}/decision.json", cohort_path, "decision_cohort_opaque", pretest_bound=True)
        copy(f"output/tasks/{task}/training_plan36.json", plan_path, "training_derived_threshold_metadata", pretest_bound=True)
        copy(f"output/tasks/{task}/map_manifest36.json", map_manifest_path, "frozen_map_identity_metadata", pretest_bound=True)
        copy(f"output/search/{task}/exhaustive_r00.json", exhaustive_path, "pretest_exhaustive_record_opaque", pretest_bound=True)
        plan = json.loads((ROOT/plan_path).read_text())
        map_manifest = json.loads((ROOT/map_manifest_path).read_text())
        rows = minimal_heads(extract_candidate_prefix(ROOT/exhaustive_path))
        if set(map_manifest) != set(map_names()) or len(rows) != 36:
            raise RuntimeError("Incomplete prior fixed representation/head roster")
        expected_ids = {name+"::"+recipe for name in map_names() for recipe in RECIPES}
        if {row["candidate_id"] for row in rows} != expected_ids or any(row["task"] != task for row in rows):
            raise RuntimeError("Prior candidate identities differ")
        by_id = {row["candidate_id"]: row for row in rows}
        for row in rows:
            for hash_key, path_key in (("head_sha256", "head_path"), ("fit_sha256", "fit_receipt_path"),
                                       ("decision_prediction_sha256", "decision_prediction_path")):
                if model_seal["bound_files"].get(row[path_key]) != row[hash_key]:
                    raise RuntimeError("Prior fitted-head/prediction proof is not bound by the pre-test seal")
        if plan["task"] != task or plan["epsilon"] != audit["exact_training_thresholds"][task]:
            raise RuntimeError("Frozen training-derived threshold differs from prior audit")
        maps = []
        for name in map_names():
            heads = {recipe: by_id[name+"::"+recipe] for recipe in RECIPES}
            affine = heads["affine"]
            if (heads["extension32"]["feature_id"] != affine["feature_id"]
                    or heads["extension32"]["class_id"] != affine["class_id"]
                    or map_manifest[name]["feature_id"] != affine["feature_id"]):
                raise RuntimeError("Prior feature/class identities differ across heads")
            feature_path = f"inputs/decision/{task}/features/{name}.npy"
            item = copy(f"output/features/{task}/decision/{name}.npy", feature_path,
                        "decision_features_opaque", pretest_bound=True)
            header = numpy_header(ROOT/feature_path)
            expected_n = audit["cohorts"][task]["decision"]
            if header["shape"] != [expected_n, affine["feature_width"]] or np.dtype(header["dtype"]) != np.dtype(np.float64):
                raise RuntimeError("Opaque feature header differs from frozen metadata")
            item["numpy_header"] = header
            maps.append({"map": name, "feature_path": feature_path, "feature_sha256": item["sha256"],
                         "feature_id": affine["feature_id"], "class_id": affine["class_id"],
                         "dimension": affine["feature_width"], "family": affine["family"],
                         "scaling": affine["scaling"], "map_payload_sha256": map_manifest[name]["sha256"],
                         "heads": heads})
        tasks[task] = {"cohort_path": cohort_path, "training_plan_path": plan_path,
                       "map_manifest_path": map_manifest_path, "exhaustive_path": exhaustive_path,
                       "epsilon": plan["epsilon"], "epsilon_physical_mse": plan["epsilon_physical_mse"],
                       "expected_n": expected_n, "maps": maps}
    record = {"schema": "step37-decision-input-manifest-v1", "status": "OPAQUE_INPUTS_PREPARED",
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "prior": prior,
        "prior_source_root_at_preparation": str(source_root), "prior_result_hashed_opaquely_only": True,
        "files": sorted(files, key=lambda item: item["path"]), "task_order": list(TASKS),
        "map_order": list(map_names()), "tasks": tasks,
        "class": {"kappa": "2", "prediction_interval": ["0", "1"], "assurance": ASSURANCE},
        "copied_feature_array_count": 54, "candidate_metadata_count": 108,
        "scientific_scope": "Exploratory diagnosis on the already-open Step36 decision cohorts; no new test or independent-data replication",
        "prefreeze_access": {
            "prior_head_risks_and_training_threshold_metadata_interpreted": True,
            "candidate_metadata_decoded_from_pretest_prefix_only": True,
            "decision_feature_array_payloads_numerically_parsed": 0,
            "decision_cohort_target_payloads_numerically_parsed": 0,
            "exhaustive_record_target_payloads_numerically_parsed": 0,
            "numpy_headers_only_inspected": True,
            "training_or_test_feature_arrays_opened": 0, "raw_source_archives_opened": 0,
            "new_prediction_model_fits": 0, "reserved_patient_arrays_opened": 0,
            "prior_step36_outcomes_already_known": True,
            "new_unseen_dataset_claim": False,
        },
        "preparation_source_sha256": sha_file(Path(__file__)),
    }
    write_json(manifest_path, record)
    actual = {str(path.relative_to(ROOT)) for path in (ROOT/"inputs").rglob("*") if path.is_file()}
    if actual != {item["path"] for item in files} | {"inputs/source_manifest37.json"}:
        raise RuntimeError("Unwhitelisted file in staged Step37 inputs")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    result = prepare(args.source_root)
    print(json.dumps({"status": result["status"], "copied_files": len(result["files"]),
        "decision_feature_arrays": result["copied_feature_array_count"],
        "candidate_metadata_rows": result["candidate_metadata_count"],
        "feature_or_target_payloads_parsed": 0, "manifest_sha256": sha_file(ROOT/"inputs/source_manifest37.json")}))
