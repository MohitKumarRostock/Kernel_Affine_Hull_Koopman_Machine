"""Artificial-only tests of the Step37 decision-input boundary and loader.

This fixture creates synthetic decision labels/features. No retained Step36
cohort, feature array, model payload, raw archive or test data is opened.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

import data37 as data
import prepare_inputs37 as prep


ROOT = Path(__file__).resolve().parents[1]


def make_fixture(root, *, cohort_role="decision", dtype=np.float64):
    root = Path(root)
    records, model_bounds, tasks = [], {}, {}

    def add(relative, payload, role, original, *, model_bound=False, protocol_bound=False, array=False):
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if array:
            np.save(path, payload, allow_pickle=False)
        else:
            data.write_json(path, payload)
        checksum = data.sha_file(path)
        item = {"path": relative, "sha256": checksum, "source_sha256": checksum,
            "source_relative_path": original, "bytes": path.stat().st_size, "role": role,
            "byte_identical_copy": True, "bound_in_prior_pretest_model_seal": model_bound,
            "bound_in_prior_protocol_seal": protocol_bound}
        records.append(item)
        if model_bound:
            model_bounds[original] = checksum
        return checksum

    for task in prep.TASKS:
        ids = ["decision:artificial-%02d" % i for i in range(4)]
        rows, maps, original_maps = [], [], {}
        for name in prep.map_names():
            d = 24 if name.startswith("raw_") or "_w24" in name or name.endswith("_p2") else 8
            feature_id = "artificial-feature:" + task + ":" + name
            class_id = "artificial-class:" + task + ":" + name
            family = name.split("_")[0]
            scaling = "minmax01" if "minmax01" in name else "standard"
            map_sha = hashlib.sha256(feature_id.encode()).hexdigest()
            heads = {}
            for recipe in prep.RECIPES:
                cid = name+"::"+recipe
                base = f"output/heads/{task}/{name}__{recipe}/"
                fields = {"candidate_id": cid, "task": task, "map": name, "recipe": recipe,
                    "feature_id": feature_id, "class_id": class_id,
                    "head_id": "artificial-head:"+task+":"+cid,
                    "feature_width": d, "family": family, "scaling": scaling,
                    "decision_risk": {"lower": "1/8", "upper": "1/4", "width": "1/8"},
                    "deployment_payload_bytes": 1000,
                    "head_sha256": hashlib.sha256((task+cid+"head").encode()).hexdigest(),
                    "fit_sha256": hashlib.sha256((task+cid+"fit").encode()).hexdigest(),
                    "decision_prediction_sha256": hashlib.sha256((task+cid+"decision").encode()).hexdigest(),
                    "fit_status": "ARTIFICIAL", "projection_applied": False,
                    "assurance": prep.ASSURANCE, "head_path": base+"head.json",
                    "fit_receipt_path": base+"fit.json", "decision_prediction_path": base+"decision.json"}
                assert set(fields) == set(prep.HEAD_FIELDS)
                rows.append(fields); heads[recipe] = fields
                for hash_key, path_key in (("head_sha256", "head_path"), ("fit_sha256", "fit_receipt_path"),
                                           ("decision_prediction_sha256", "decision_prediction_path")):
                    model_bounds[fields[path_key]] = fields[hash_key]
            feature_path = f"inputs/decision/{task}/features/{name}.npy"
            matrix = np.arange(4*d, dtype=dtype).reshape(4, d)/8
            checksum = add(feature_path, matrix, "decision_features_opaque",
                           f"output/features/{task}/decision/{name}.npy", model_bound=True, array=True)
            maps.append({"map": name, "feature_path": feature_path, "feature_sha256": checksum,
                "feature_id": feature_id, "class_id": class_id, "dimension": d,
                "family": family, "scaling": scaling, "map_payload_sha256": map_sha, "heads": heads})
            original_maps[name] = {"feature_id": feature_id, "sha256": map_sha}
        cohort_path = f"inputs/decision/{task}/cohort.json"
        plan_path = f"inputs/decision/{task}/training_plan36.json"
        map_path = f"inputs/decision/{task}/map_manifest36.json"
        exhaustive_path = f"inputs/decision/{task}/exhaustive_r00.json"
        add(cohort_path, {"name": cohort_role, "target": task, "ids": ids,
                         "y": ["-1/2", "0", "2/3", "3/2"],
                         "raw_targets": ["UNUSED"]*4, "persistence": ["UNUSED"]*4,
                         "lag_matrix_path": "/MUST_NOT_OPEN_IGNORED_LAG_MATRIX.npy"},
            "decision_cohort_opaque", f"output/cohorts/{task}/decision.json", model_bound=True)
        add(plan_path, {"task": task, "epsilon": "1/10", "epsilon_physical_mse": "10"},
            "training_derived_threshold_metadata", f"output/tasks/{task}/training_plan36.json", model_bound=True)
        add(map_path, original_maps, "frozen_map_identity_metadata", f"output/tasks/{task}/map_manifest36.json", model_bound=True)
        add(exhaustive_path, {"candidate_rows": rows, "range_floor": {"targets_exact": ["NEVER_DECODE_AS_NUMBERS"]}},
            "pretest_exhaustive_record_opaque", f"output/search/{task}/exhaustive_r00.json", model_bound=True)
        tasks[task] = {"cohort_path": cohort_path, "training_plan_path": plan_path,
                       "map_manifest_path": map_path, "exhaustive_path": exhaustive_path,
                       "epsilon": "1/10", "epsilon_physical_mse": "10", "expected_n": 4, "maps": maps}
    prior_protocol = {"tasks": list(prep.TASKS), "class": {"kappa": "2", "prediction_interval": ["0", "1"],
                                                        "assurance": prep.ASSURANCE}}
    protocol_sha = add("inputs/prior/protocol36.json", prior_protocol, "prior_protocol", "protocol36.json", protocol_bound=True)
    source_sha = add("inputs/prior/public_source36.json", {"artificial_only": True}, "prior_public_source_metadata",
                     "inputs/public_source36.json", protocol_bound=True)
    prior_protocol_seal = {"bound_files": {"protocol36.json": protocol_sha, "inputs/public_source36.json": source_sha}}
    protocol_seal_sha = add("inputs/prior/protocol_seal36.json", prior_protocol_seal, "prior_protocol_seal", "protocol_seal36.json")
    prior_model = {"protocol_seal_sha256": protocol_seal_sha, "bound_files": model_bounds}
    model_sha = add("inputs/prior/model_seal36.json", prior_model, "prior_model_seal", "output/model_seal36.json")
    prior_audit = {"status": "PASS", "result_sha256": "0"*64, "model_seal_sha256": model_sha,
                   "protocol_seal_sha256": protocol_seal_sha, "artificial_only": True}
    audit_sha = add("inputs/prior/postrun_audit36.json", prior_audit, "prior_independent_audit", "audit/postrun_audit36.json")
    manifest = {"schema": "step37-decision-input-manifest-v1", "task_order": list(prep.TASKS),
                "map_order": list(prep.map_names()), "class": prior_protocol["class"],
                "files": sorted(records, key=lambda item: item["path"]), "tasks": tasks,
                "prior": {"result_sha256": "0"*64, "protocol_sha256": protocol_sha,
                          "protocol_seal_sha256": protocol_seal_sha, "model_seal_sha256": model_sha,
                          "postrun_audit_sha256": audit_sha}, "artificial_only": True}
    data.write_json(root/data.MANIFEST_PATH, manifest)
    data.write_json(root/"protocol37.json", {"step": 37, "tasks": list(prep.TASKS), "artificial_only": True})
    for filename in ("data37.py", "prepare_inputs37.py"):
        path = root/"source"/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/"source"/filename).read_bytes())
    paths = [root/"protocol37.json", root/data.MANIFEST_PATH,
             root/"source/data37.py", root/"source/prepare_inputs37.py"]
    paths += [root/item["path"] for item in records]
    seal = {"bound_files": {str(path.relative_to(root)): data.sha_file(path) for path in paths}, "artificial_only": True}
    data.write_json(root/"protocol_seal37.json", seal)
    data.write_json(root/"audit/preexecution_gate37.json", {"status": "PASS", "artificial_only": True,
                   "protocol_seal_sha256": data.sha_file(root/"protocol_seal37.json")})
    return manifest


def main():
    checks = {}
    original_root, original_np_load = data.ROOT, np.load
    opened = []

    def spy_load(path, *args, **kwargs):
        opened.append(str(Path(path)))
        return original_np_load(path, *args, **kwargs)

    def blocked(call, phrase):
        opened.clear()
        try:
            call()
        except (RuntimeError, ValueError) as error:
            assert phrase in str(error), repr(error)
        else:
            raise AssertionError("Expected guarded access to fail")
        assert opened == [], opened

    np.load = spy_load
    try:
        with tempfile.TemporaryDirectory(prefix="step37-artificial-data-") as temporary:
            root = Path(temporary)
            manifest = make_fixture(root)
            data.ROOT = root
            seal_path, gate_path = root/"protocol_seal37.json", root/"audit/preexecution_gate37.json"
            seal_bytes, gate_bytes = seal_path.read_bytes(), gate_path.read_bytes()
            seal_path.unlink()
            blocked(data.load_inputs, "protocol seal required")
            checks["missing_protocol_seal_blocks_public_loader_before_array_access"] = True
            blocked(lambda: data._load_task("casual"), "protocol seal required")
            checks["missing_protocol_seal_blocks_internal_loader_before_array_access"] = True
            seal_path.write_bytes(seal_bytes)
            gate_path.unlink()
            blocked(data.load_inputs, "Independent Step37 preexecution PASS required")
            checks["missing_independent_pass_blocks_array_access"] = True
            data.write_json(gate_path, {"status": "PASS", "protocol_seal_sha256": "f"*64})
            blocked(data.load_inputs, "Independent Step37 preexecution PASS required")
            checks["pass_for_another_seal_blocks_array_access"] = True
            gate_path.write_bytes(gate_bytes)
            feature_path = root/manifest["tasks"]["casual"]["maps"][0]["feature_path"]
            feature_bytes = feature_path.read_bytes()
            feature_path.write_bytes(feature_bytes+b"tamper")
            blocked(data.load_inputs, "Frozen protocol/source/input changed")
            checks["changed_bound_input_blocks_before_array_access"] = True
            feature_path.write_bytes(feature_bytes)
            extra = root/"inputs/test_values.npy"; extra.write_bytes(b"unused")
            blocked(data.load_inputs, "Unwhitelisted or missing Step37 input file")
            checks["unwhitelisted_test_input_blocks_before_array_access"] = True
            extra.unlink()
            opened.clear()
            loaded = data.load_inputs()
            expected_paths = {str(root/item["path"]) for item in manifest["files"] if item["role"] == "decision_features_opaque"}
            checks["only_54_whitelisted_feature_arrays_opened"] = len(opened) == 54 and set(opened) == expected_paths
            checks["common_three_target_decision_ids_preserved"] = all(loaded[t]["cohort"]["ids"] == loaded["casual"]["cohort"]["ids"] for t in prep.TASKS)
            checks["out_of_range_targets_retained_exactly"] = all(loaded[t]["cohort"]["y"] == [F(-1, 2), F(0), F(2, 3), F(3, 2)] for t in prep.TASKS)
            checks["frozen_threshold_and_physical_scale_preserved"] = all(loaded[t]["epsilon"] == F(1, 10) and loaded[t]["physical_mse_multiplier"] == 100 for t in prep.TASKS)
            checks["feature_arrays_returned_read_only"] = all(not item["z"].flags.writeable for t in loaded.values() for item in t["maps"].values())
            checks["unused_lag_paths_and_exhaustive_target_payload_not_interpreted"] = True
            access = json.loads((root/"output/input_access37.json").read_text())
            checks["gated_access_receipt_records_decision_only_scope"] = (access["decision_feature_arrays_loaded"] == 54 and access["targets_clipped"] is False
                and access["training_or_test_feature_arrays_accessed"] is False and access["raw_source_archive_accessed"] is False)
        with tempfile.TemporaryDirectory(prefix="step37-artificial-role-") as temporary:
            data.ROOT = Path(temporary); make_fixture(data.ROOT, cohort_role="train")
            blocked(data.load_inputs, "non-decision cohort")
            checks["nondecision_cohort_role_rejected_before_array_access"] = True
        with tempfile.TemporaryDirectory(prefix="step37-artificial-dtype-") as temporary:
            data.ROOT = Path(temporary); make_fixture(data.ROOT, dtype=np.float32)
            opened.clear()
            try:
                data.load_inputs()
            except RuntimeError as error:
                assert "feature matrix is invalid" in str(error), repr(error)
            else:
                raise AssertionError("A different stored coordinate dtype was accepted")
            checks["nonbinary64_features_rejected"] = len(opened) == 1
    finally:
        data.ROOT, np.load = original_root, original_np_load
    if not all(checks.values()):
        raise AssertionError(checks)
    result = {"schema": "step37-artificial-data-preflight-v1", "status": "PASS", "artificial_only": True,
              "checks": checks, "actual_decision_feature_arrays_loaded": 0,
              "actual_decision_target_payloads_parsed": 0, "training_or_test_arrays_opened": 0,
              "raw_source_archives_opened": 0, "new_prediction_model_fits": 0, "reserved_patient_arrays_opened": 0,
              "source_sha256": {"source/"+name: data.sha_file(ROOT/"source"/name)
                                 for name in ("data37.py", "prepare_inputs37.py", "check_data37.py")}}
    data.write_json(ROOT/"tests/artificial_data_preflight37.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
