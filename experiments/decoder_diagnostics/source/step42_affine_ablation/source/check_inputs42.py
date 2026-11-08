"""Independent checks of Step 42's copied and concatenated Step 41 inputs.

No preparation helper is imported. All checks are against the retained original
package manifest, original receipts/rosters, and independently reconstructed rows.
The check concerns faithful reuse of already exposed data, not new validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction as Q
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_HASH = "41b7194fed979c50b71b4af58b4ca8ac99c398afca4c9986e0b5f5c50d86d687"
MANIFEST_HASH = "208db57f86830ae63868d5889ffe7426eb046183e073fbfa9998e6a5b4ec56d1"
TASK_COUNTS = {
    "household_power2010": {"f1": (144, 30), "f2": (174, 31), "f3": (205, 24), "final": (229, 81)},
    "jena_temperature2017": {"f1": (149, 30), "f2": (179, 31), "f3": (210, 31), "final": (241, 91)},
}


def hash_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def canon(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


class Check:
    def __init__(self, root):
        self.root = root
        self.count = 0
        self.groups = {}
        self.file_cache = {}
        self.arrays = {}
        self.raw_cache = {}

    def require(self, condition, description, group="contract"):
        self.count += 1
        self.groups[group] = self.groups.get(group, 0) + 1
        if not condition:
            raise ValueError(description)

    def path(self, relative):
        rel = Path(relative)
        self.require(not rel.is_absolute() and ".." not in rel.parts and bool(rel.parts),
                     "Unsafe retained path", "paths")
        return self.root / rel

    def file(self, ref):
        path = self.path(ref["path"])
        if ref["path"] not in self.file_cache:
            self.file_cache[ref["path"]] = (hash_file(path), path.stat().st_size)
        actual, length = self.file_cache[ref["path"]]
        self.require(actual == ref["sha256"] and ("bytes" not in ref or length == ref["bytes"]),
                     "Local input binding mismatch: " + ref["path"], "file_hashes")
        return path

    def matrix(self, ref):
        path = self.file(ref)
        key = str(path)
        if key not in self.arrays:
            self.arrays[key] = np.load(path, allow_pickle=False)
        matrix = self.arrays[key]
        self.require(matrix.ndim == 2 and matrix.dtype == np.dtype("float64") and np.isfinite(matrix).all(),
                     "A retained matrix is not finite binary64", "matrices")
        return matrix

    def original(self, path):
        info = self.inputs["source_files"][path]
        self.require(info["mode"] == "byte_identical_copy", "JSON source not retained exactly", "provenance")
        return read(self.path(info["local_path"]))

    def run(self):
        self.inputs = read(self.root / "inputs/manifest42.json")
        m = self.inputs
        self.require(m["schema"] == "step42-fixed-input-manifest-v1", "Unknown input schema")
        self.require(m["study_role"] == "EXPOSED_PERIOD_DIAGNOSTIC" and m["prior_outcomes_already_exposed"]
                     and not m["new_primary_generalization_claim"], "Exposed-period scope is missing")
        self.require(all(m[x] == 0 for x in ("new_source_measurement_tokens_decoded", "feature_transforms_performed",
                                             "map_fits_performed", "head_fits_performed")), "Preparation claims new fitting")
        self.require(m["source_archive_sha256"] == ARCHIVE_HASH and m["source_archive_manifest_sha256"] == MANIFEST_HASH,
                     "Wrong inherited package identity")
        prior_manifest_path = self.root / "inputs/prior41/MANIFEST41.json"
        self.require(hash_file(prior_manifest_path) == MANIFEST_HASH, "Retained prior manifest changed", "provenance")
        package = read(prior_manifest_path)
        source_catalog = {x["path"]: x for x in package["files"]}
        self.require(len(source_catalog) == len(package["files"]) == 7082, "Prior manifest closure changed", "provenance")
        receipt = read(self.root / "inputs/prior41/step41_package_verification41.json")
        self.require(receipt["status"] == "PASS" and receipt["archive_sha256"] == ARCHIVE_HASH
                     and receipt["manifest_sha256"] == MANIFEST_HASH and receipt["scientific_pass_gates"] == 24,
                     "Prior package check does not bind expected complete record", "provenance")
        self.require(hash_file(self.root / "source/prepare42.py") == m["preparation_source_sha256"],
                     "Preparation source changed", "provenance")
        actual_files = {str(p.relative_to(self.root)) for p in (self.root / "inputs").rglob("*") if p.is_file()}
        self.require(actual_files == set(m["files"]) | {"inputs/manifest42.json"}, "Prepared input file closure changed", "provenance")
        for name, binding in m["files"].items():
            self.file({"path": name, **binding})
        reconstruction_count = 0
        for name, info in m["source_files"].items():
            expected = source_catalog[name]
            self.require(info["source_sha256"] == expected["sha256"] and info["source_bytes"] == expected["bytes"],
                         "Source binding not anchored in original package: " + name, "provenance")
            if info["mode"] == "byte_identical_copy":
                self.file({"path": info["local_path"], "sha256": expected["sha256"], "bytes": expected["bytes"]})
            elif info["mode"] == "exact_one_row_npy_reconstruction":
                array = self.matrix({"path": info["assembly_path"], **m["files"][info["assembly_path"]]})
                i = info["assembly_row"]
                self.require(type(i) is int and 0 <= i < len(array), "Invalid assembly row", "reconstructed_rows")
                stream = io.BytesIO()
                np.save(stream, array[i:i + 1], allow_pickle=False)
                raw = stream.getvalue()
                self.require(hashlib.sha256(raw).hexdigest() == expected["sha256"] and len(raw) == expected["bytes"],
                             "Assembled row does not reproduce exact prior NPY bytes: " + name, "reconstructed_rows")
                reconstruction_count += 1
            else:
                raise ValueError("Unknown input reuse operation")
        protocol = self.original("protocol41.json")
        protocol_seal = self.original("protocol_seal41.json")
        audit = self.original("audit/postrun_audit41.json")
        result = self.original("output/result41.json")
        self.require(audit["status"] == "PASS" and audit["primary_status"] == result["primary_status"] == "NOT_ESTABLISHED",
                     "Prior negative outcome changed", "provenance")
        self.require(audit["result_sha256"] == m["source_result_sha256"] == package["result_sha256"]
                     and audit["protocol_seal_sha256"] == m["source_protocol_seal_sha256"] == package["protocol_seal_sha256"],
                     "Prior scientific ancestry mismatch", "provenance")
        self.require(protocol_seal["protocol_sha256"] == source_catalog["protocol41.json"]["sha256"],
                     "Prior protocol binding differs", "provenance")
        for name, expected in audit["scientific_gates"].items():
            gate = self.original(name)
            self.require(source_catalog[name]["sha256"] == expected and gate["status"] == "PASS",
                         "Prior scientific gate changed", "provenance")
        self.require(m["map_order"] == protocol["maps"] and len(m["map_order"]) == 18 and len(set(m["map_order"])) == 18,
                     "Fixed map panel changed")
        self.require(set(m["tasks"]) == set(TASK_COUNTS), "Task panel changed")
        self.roster = self.original("inputs/data41/calendar_roster41.json")
        totals = {"training_rows_including_reuse": 0, "development_rows": 0, "assessment_rows": 0,
                  "training_feature_matrices": 0, "evaluation_feature_matrices": 0}
        for task, expected_stages in TASK_COUNTS.items():
            stages = m["tasks"][task]["stages"]
            self.require(set(stages) == set(expected_stages), "Fitting stages changed")
            for stage, expected_counts in expected_stages.items():
                spec = stages[stage]
                train, evaluation = self.stage(task, stage, spec, expected_counts)
                totals["training_rows_including_reuse"] += train["n"]
                totals["assessment_rows" if stage == "final" else "development_rows"] += evaluation["n"]
                totals["training_feature_matrices"] += 18
                totals["evaluation_feature_matrices"] += 18
        self.require(reconstruction_count == (81 + 91) * 19, "Unexpected count of original one-row arrays", "reconstructed_rows")
        return {"schema": "step42-independent-input-audit-v1", "status": "PASS", "checks_count": self.count,
                "checks_by_group": self.groups, "input_manifest_sha256": hash_file(self.root / "inputs/manifest42.json"),
                "source_archive_sha256": ARCHIVE_HASH, "source_manifest_sha256": MANIFEST_HASH,
                "source_protocol_seal_sha256": m["source_protocol_seal_sha256"], "source_result_sha256": m["source_result_sha256"],
                "source_files_bound": len(m["source_files"]), "prepared_files_checked": len(m["files"]),
                "one_row_npy_files_exactly_reconstructed": reconstruction_count,
                "unique_raw_queries_checked_across_roles": len(self.raw_cache), "totals": totals,
                "study_role": "EXPOSED_PERIOD_DIAGNOSTIC", "new_generalization_claim": False,
                "new_source_measurement_tokens_decoded": 0, "feature_transforms_or_fits": 0,
                "prior_record_modified": False, "checker_sha256": hash_file(Path(__file__)),
                "source_archive_scope": "Original archive is identified by its verified SHA; the minimal head diagnosis reuses the original package-bound stored matrices rather than redecoding raw measurements.",
                "utc": datetime.now(timezone.utc).isoformat()}

    def stage(self, task, stage, spec, expected_counts):
        self.require(spec["task"] == task and spec["stage"] == stage and spec["map_order"] == self.inputs["map_order"],
                     "Stage identity changed")
        normal = read(self.file(spec["normalization"]))
        lo, hi, span = Q(normal["target_min"]), Q(normal["target_max"]), Q(normal["span"])
        self.require(normal["task"] == task and normal["stage"] == stage and normal["fitted_partition"] == "train"
                     and not normal["future_data_used"] and hi > lo and span == hi - lo,
                     "Training-only normalization violated", "normalization")
        cohorts = []
        for position, role in enumerate(("train", "evaluation")):
            info = spec[role]
            cohort = read(self.file(info["cohort"]))
            X = self.matrix(info["X"])
            expected_role = "train" if role == "train" else ("assessment" if stage == "final" else "validation")
            self.require(cohort["task"] == task and cohort["stage"] == stage and cohort["name"] == expected_role
                         and cohort["study_role"] == "EXPOSED_PERIOD_DIAGNOSTIC", "Cohort role changed", "cohort_identity")
            self.require(cohort["n"] == info["n"] == expected_counts[position] == len(cohort["ids"])
                         and X.shape == (cohort["n"], 24), "Cohort shape changed", "cohort_identity")
            self.require(info["ids"] == cohort["ids"] == sorted(set(cohort["ids"])), "Query order or uniqueness changed", "cohort_identity")
            self.require(cohort["X_path"] == info["X"]["path"] and cohort["X_sha256"] == info["X"]["sha256"]
                         and cohort["normalization_path"] == spec["normalization"]["path"]
                         and cohort["normalization_sha256"] == spec["normalization"]["sha256"],
                         "Cohort input binding changed", "cohort_identity")
            wanted = (self.roster["tasks"][task]["test"]["queries"] if expected_role == "assessment"
                      else self.roster["tasks"][task]["stages"][stage][expected_role]["queries"])
            self.require(cohort["ids"] == [x["id"] for x in wanted]
                         and cohort["target_timestamps"] == [x["target_timestamp"] for x in wanted]
                         and cohort["origin_timestamps"] == [x["origin_timestamp"] for x in wanted],
                         "Cohort differs from frozen prior eligibility roster", "chronology")
            self.require(not cohort["target_values_clipped"] and not cohort["other_measurement_predictors_used"],
                         "Target filtering or extra predictors introduced", "cohort_identity")
            self.require(len(cohort["y"]) == len(cohort["raw_targets"]) == len(cohort["lag_timestamps"]) == cohort["n"],
                         "Target or lag row count changed", "cohort_identity")
            for i, query in enumerate(wanted):
                target = datetime.fromisoformat(query["target_timestamp"])
                origin = datetime.fromisoformat(query["origin_timestamp"])
                lags = [datetime.fromisoformat(x) for x in cohort["lag_timestamps"][i]]
                self.require(target.tzinfo is None and target.hour == 23 and target.minute == target.second == 0
                             and origin == target - timedelta(hours=24)
                             and lags == [origin - timedelta(hours=k) for k in range(23, -1, -1)],
                             "Forecast origin/horizon/lag chronology changed", "chronology")
                raw = Q(cohort["raw_targets"][i])
                self.require(Q(cohort["y"][i]) == (raw - lo) / span,
                             "Exact target normalization changed", "normalization")
                self.require(float(Q(cohort["raw_persistence"][i])).hex() == float(X[i, -1]).hex()
                             and Q(cohort["persistence"][i]) == (Q(cohort["raw_persistence"][i]) - lo) / span,
                             "Raw-input endpoint and persistence disagree", "raw_values")
                cached = (str(raw), X[i].view(np.uint64).tobytes())
                key = (task, query["id"])
                if key in self.raw_cache:
                    self.require(self.raw_cache[key] == cached, "Same query changes raw data across chronological stages", "raw_values")
                else:
                    self.raw_cache[key] = cached
            if expected_role != "assessment":
                old = self.original(cohort["source_cohort_path"])
                for key in ("task", "stage", "name", "n", "ids", "target_timestamps", "origin_timestamps",
                            "lag_timestamps", "raw_targets", "y", "raw_persistence", "persistence", "normalization_sha256"):
                    self.require(old[key] == cohort[key], "Copied cohort differs in " + key, "cohort_identity")
                source = self.inputs["source_files"][cohort["source_cohort_path"]]
                self.require(source["source_sha256"] == cohort["source_cohort_sha256"]
                             and old["X_sha256"] == info["X"]["sha256"], "Original cohort source binding changed", "provenance")
            cohorts.append(cohort)
        train, evaluation = cohorts
        self.require(min(map(Q, train["raw_targets"])) == lo and max(map(Q, train["raw_targets"])) == hi,
                     "Normalization was not determined by the training targets", "normalization")
        self.require(hashlib.sha256(canon(train["ids"])).hexdigest() == normal["train_target_ids_sha256"]
                     and hashlib.sha256(canon(train["raw_targets"])).hexdigest() == normal["raw_train_targets_sha256"],
                     "Normalization training membership binding changed", "normalization")
        self.require(set(train["ids"]).isdisjoint(evaluation["ids"])
                     and max(train["target_timestamps"]) <= min(evaluation["origin_timestamps"]),
                     "Future evaluation labels enter a fitted stage", "chronology")
        prior_fit = read(self.file(spec["inherited_fit_manifest"]))
        inherited_predictions = read(self.file(spec["inherited_predictions"]))
        self.require(inherited_predictions["query_ids"] == evaluation["ids"], "Inherited predictions have different query order", "prediction_context")
        self.require(prior_fit["normalization"] == normal and prior_fit["training_n"] == train["n"],
                     "Fixed feature factory was fitted on different data", "map_bindings")
        for name in spec["map_order"]:
            item = spec["maps"][name]
            old = prior_fit["maps"][name]
            oldrow = next(x for x in prior_fit["candidate_rows"] if x["map"] == name)
            for field in ("name", "family", "scaling", "dimension", "feature_id", "map_payload_bytes"):
                self.require(item[field] == old[field], "Inherited map metadata changed: " + field, "map_bindings")
            self.require(item["map"]["sha256"] == old["map_sha256"]
                         and item["training_features"]["sha256"] == old["training_features"]["sha256"]
                         and item["original_deployment_payload_bytes"] == oldrow["deployment_payload_bytes"]
                         and item["original_static_deployment_bytes"] == oldrow["static_deployment_bytes"],
                         "Map, training features, or original payload changed", "map_bindings")
            tr = self.matrix(item["training_features"])
            ev = self.matrix(item["evaluation_features"])
            self.require(tr.shape == (train["n"], item["dimension"]) and ev.shape == (evaluation["n"], item["dimension"]),
                         "Fixed map matrix dimensions changed", "matrices")
            oldfit = read(self.file(item["inherited_fit"]))
            oldhead = read(self.file(item["inherited_head"]))
            self.require(oldfit["head"] == oldhead and oldfit["feature_id"] == item["feature_id"]
                         and oldfit["training_binding"]["feature_sha256"] == item["training_features"]["sha256"]
                         and oldfit["training_binding"]["cohort_sha256"] == train["source_cohort_sha256"]
                         and oldfit["training_binding"]["map_sha256"] == item["map"]["sha256"],
                         "Inherited fit does not bind these exact training features", "map_bindings")
            for field, oldfield in (("inherited_fit", "fit_sha256"), ("inherited_head", "head_sha256"),
                                    ("inherited_deployment", "deployment_sha256")):
                self.require(item[field]["sha256"] == oldrow[oldfield], "Original fitted artifact changed", "map_bindings")
            if stage != "final":
                self.require(item["evaluation_features"]["sha256"] == inherited_predictions["maps"][name]["features"]["sha256"],
                             "Original validation features changed", "map_bindings")
        if stage == "final":
            self.assessment(task, spec, evaluation, inherited_predictions)
        return train, evaluation

    def assessment(self, task, spec, cohort, assembled):
        final_fit_hash = self.inputs["source_files"][f"output/{task}/final/fit_seal41.json"]["source_sha256"]
        primary = self.original(f"output/{task}/primary_result41.json")
        self.require(primary["query_ids"] == cohort["ids"] and primary["raw_targets"] == cohort["raw_targets"],
                     "Original assessed target set changed", "assessment")
        prediction_lists = {}
        X = self.matrix(spec["evaluation"]["X"])
        for i, query in enumerate(cohort["source_queries"]):
            history = read(self.file(query["history"]))
            target = read(self.file(query["target"]))
            bundle = read(self.file(query["prediction"]))
            seal = read(self.file(query["prediction_seal"]))
            qid = cohort["ids"][i]
            self.require(query["query_index"] == history["query_index"] == target["query_index"] == bundle["query_index"] == seal["query_index"] == i
                         and query["query_id"] == history["query_id"] == target["query_id"] == bundle["query_id"] == seal["query_id"] == qid,
                         "Original prediction/target/history identity mismatch", "assessment")
            self.require(target["raw_target"] == cohort["raw_targets"][i] and target["y"] == cohort["y"][i]
                         and target["target_timestamp"] == history["target_timestamp"] == cohort["target_timestamps"][i]
                         and target["origin_timestamp"] == history["origin_timestamp"] == cohort["origin_timestamps"][i],
                         "Original assessment labels or dates changed", "assessment")
            self.require(history["history_timestamps"] == cohort["lag_timestamps"][i]
                         and np.array_equal(np.asarray([float(Q(x)) for x in history["history_values"]]).view(np.uint64), X[i].view(np.uint64)),
                         "Assembled raw history differs from retained exact values", "raw_values")
            self.require(bundle["model_seal_sha256"] == history["model_seal_sha256"] == target["model_seal_sha256"] == seal["model_seal_sha256"] == final_fit_hash
                         and bundle["history_receipt_sha256"] == seal["history_receipt_sha256"] == query["history"]["sha256"]
                         and seal["prediction_sha256"] == query["prediction"]["sha256"]
                         and target["prediction_seal_sha256"] == query["prediction_seal"]["sha256"],
                         "Original chronological model/prediction/target binding changed", "assessment")
            self.require(not history["current_target_value_decoded"] and not seal["current_target_value_decoded"],
                         "Original chronology receipt contradicts prediction-before-label ordering", "assessment")
            rawsource = self.inputs["source_files"][history["X_path"]]
            self.require(rawsource["assembly_path"] == spec["evaluation"]["X"]["path"] and rawsource["assembly_row"] == i,
                         "Raw input row assembly order changed", "assessment")
            for name in spec["map_order"]:
                source = bundle["maps"][name]["features"]
                assembled_source = self.inputs["source_files"][source["path"]]
                evref = spec["maps"][name]["evaluation_features"]
                self.require(assembled_source["source_sha256"] == source["sha256"]
                             and assembled_source["assembly_path"] == evref["path"] and assembled_source["assembly_row"] == i
                             and evref["source_rows"][i]["query_id"] == qid and evref["source_rows"][i]["source_path"] == source["path"],
                             "Feature query assembly differs from original deployment", "assessment")
            for cid, entries in bundle["predictions"].items():
                self.require(len(entries) == 1 and entries[0]["query_id"] == qid,
                             "Original prediction row identity mismatch", "prediction_context")
                prediction_lists.setdefault(cid, []).extend(entries)
        self.require(prediction_lists == assembled["predictions"] and not assembled["recomputed_predictions"],
                     "Assembled original prediction values changed", "prediction_context")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = Check(args.root.resolve()).run()
    output = args.output or args.root / "audit/input_audit42.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canon(report))
    print(json.dumps({"status": report["status"], "checks": report["checks_count"],
                      "copied_source_files": report["source_files_bound"],
                      "reconstructed_one_query_arrays": report["one_row_npy_files_exactly_reconstructed"],
                      "totals": report["totals"], "receipt": str(output)}))


if __name__ == "__main__":
    main()
