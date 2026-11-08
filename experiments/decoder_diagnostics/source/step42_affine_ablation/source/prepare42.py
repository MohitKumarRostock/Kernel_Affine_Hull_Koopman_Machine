"""Reuse audited Step 41 stored data for the exposed-period Step 42 diagnosis.

This module never imports or runs the Step 41 data-access state machine, feature
factory, or fitting code. It copies retained matrices and concatenates retained
one-query rows. Original data, fitted maps, selections, and outcomes are immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "step41_certified_affine"
TASKS = ("household_power2010", "jena_temperature2017")
STAGES = ("f1", "f2", "f3", "final")
EXPECTED_ARCHIVE_SHA256 = "41b7194fed979c50b71b4af58b4ca8ac99c398afca4c9986e0b5f5c50d86d687"
EXPECTED_MANIFEST_SHA256 = "208db57f86830ae63868d5889ffe7426eb046183e073fbfa9998e6a5b4ec56d1"
EXPECTED_PROTOCOL_SEAL_SHA256 = "0b564ce79d83b27b01d6d87ba48fb97721d689bc9865aebc29b0c7ce4a6d9cf3"
EXPECTED_RESULT_SHA256 = "c7f7f25da8b2b26a9b7efd3c90a230da5bd66e6e4a2ddfb320eb255a45a04cf3"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def safe_relative(text):
    path = Path(text)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Unsafe relative path: " + text)
    return path


def array_bytes(array):
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


class Preparation:
    def __init__(self, prior, destination):
        self.prior = prior.resolve()
        self.destination = destination.resolve()
        if self.destination != ROOT / "inputs":
            raise ValueError("This prepared input contract uses ROOT/inputs")
        self.staging = ROOT / "inputs.preparing42"
        if self.destination.exists() or self.staging.exists():
            raise FileExistsError("Inputs or interrupted preparation already exist; retain and inspect them")
        self.staging.mkdir(parents=True)
        self.source_files = {}
        self.generated_files = {}
        manifest_path = self.prior.parent / "MANIFEST41.json"
        receipt_path = self.prior.parent / "step41_package_verification41.json"
        archive_path = self.prior.parent / "step41_complete_package.zip"
        if digest_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise ValueError("The authoritative Step 41 package manifest changed")
        self.archive_manifest = read_json(manifest_path)
        self.receipt = read_json(receipt_path)
        if self.receipt["status"] != "PASS" or self.receipt["archive_sha256"] != EXPECTED_ARCHIVE_SHA256:
            raise ValueError("The inherited complete package lacks the expected PASS")
        if self.receipt["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
            raise ValueError("The inherited archive receipt binds a different manifest")
        if digest_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
            raise ValueError("The inherited complete archive changed")
        if self.archive_manifest["root_prefix"] != "step41_certified_affine/":
            raise ValueError("Unexpected inherited archive root")
        self.catalog = {item["path"]: item for item in self.archive_manifest["files"]}
        if len(self.catalog) != len(self.archive_manifest["files"]):
            raise ValueError("Duplicate source manifest paths")
        self.external_sources = {}
        for path, local in [(manifest_path, "inputs/prior41/MANIFEST41.json"),
                            (receipt_path, "inputs/prior41/step41_package_verification41.json")]:
            self.write_bytes(local, path.read_bytes())
            self.external_sources[local] = {"sha256": digest_file(path), "bytes": path.stat().st_size}

    def target(self, local):
        path = safe_relative(local)
        if path.parts[0] != "inputs":
            raise ValueError("Preparation may only write its owned input tree")
        return self.staging.joinpath(*path.parts[1:])

    def write_bytes(self, local, raw):
        path = self.target(local)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        return {"path": local, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def write_json(self, local, data):
        ref = self.write_bytes(local, canonical_bytes(data))
        self.generated_files[local] = ref
        return ref

    def source(self, relative):
        rel = str(safe_relative(relative))
        record = self.catalog.get(rel)
        if record is None:
            raise ValueError("Source absent from Step 41 package manifest: " + rel)
        path = self.prior / rel
        if path.stat().st_size != record["bytes"] or digest_file(path) != record["sha256"]:
            raise ValueError("Source differs from the audited Step 41 package: " + rel)
        return path, record

    def copy_source(self, relative, local=None):
        path, record = self.source(relative)
        local = local or "inputs/prior41/" + relative
        previous = self.source_files.get(relative)
        if previous is not None:
            if previous.get("local_path") != local:
                raise ValueError("One retained source was assigned two local destinations")
            return {"path": local, "sha256": record["sha256"], "bytes": record["bytes"]}
        ref = self.write_bytes(local, path.read_bytes())
        self.source_files[relative] = {"source_sha256": record["sha256"], "source_bytes": record["bytes"],
                                       "local_path": local, "mode": "byte_identical_copy"}
        return ref

    def retain_json(self, relative):
        ref = self.copy_source(relative)
        return read_json(self.target(ref["path"])), ref

    def assemble(self, local, members):
        arrays, bindings = [], []
        for index, item in enumerate(members):
            relative, query_id = item
            path, record = self.source(relative)
            z = np.load(path, allow_pickle=False)
            if z.dtype != np.dtype("float64") or z.ndim != 2 or z.shape[0] != 1 or not np.isfinite(z).all():
                raise ValueError("Invalid inherited one-query feature row")
            if hashlib.sha256(array_bytes(z)).hexdigest() != record["sha256"]:
                raise ValueError("The original NPY encoding is not exactly reconstructible")
            arrays.append(z)
            bindings.append({"row": index, "query_id": query_id, "source_path": relative,
                             "source_sha256": record["sha256"], "source_bytes": record["bytes"]})
            if relative in self.source_files:
                raise ValueError("Unexpected duplicated assembly source")
            self.source_files[relative] = {"source_sha256": record["sha256"], "source_bytes": record["bytes"],
                                           "mode": "exact_one_row_npy_reconstruction",
                                           "assembly_path": local, "assembly_row": index}
        combined = np.concatenate(arrays, axis=0)
        ref = self.write_bytes(local, array_bytes(combined))
        self.generated_files[local] = ref
        return {**ref, "shape": list(combined.shape), "dtype": str(combined.dtype), "source_rows": bindings}

    def cohort(self, task, stage, role):
        relative = f"output/{task}/{stage}/cohorts/{role}.json"
        original, old_ref = self.retain_json(relative)
        base = f"inputs/data42/{task}/{stage}"
        Xref = self.copy_source(original["X_path"], base + f"/cohorts/{role}_X.npy")
        new = {**original, "schema": "step42-reused-causal-cohort-v1", "study_role": "EXPOSED_PERIOD_DIAGNOSTIC",
               "source_cohort_path": relative, "source_cohort_sha256": old_ref["sha256"],
               "X_path": Xref["path"], "X_sha256": Xref["sha256"],
               "normalization_path": base + "/normalization.json"}
        cref = self.write_json(base + f"/cohorts/{role}.json", new)
        return {"cohort": cref, "X": Xref, "n": new["n"], "ids": new["ids"]}

    def prepare(self):
        protocol, _ = self.retain_json("protocol41.json")
        seal, _ = self.retain_json("protocol_seal41.json")
        audit, _ = self.retain_json("audit/postrun_audit41.json")
        result, _ = self.retain_json("output/result41.json")
        roster, _ = self.retain_json("inputs/data41/calendar_roster41.json")
        self.copy_source("output/execution_completed41.json")
        self.copy_source("output/access_and_execution41.jsonl")
        if audit["status"] != "PASS" or audit["primary_status"] != "NOT_ESTABLISHED":
            raise ValueError("Unexpected inherited scientific status")
        for path in audit["scientific_gates"]:
            self.copy_source(path)
        self.copy_source("source/affine41.py")
        self.copy_source("source/certificate41.py")
        self.copy_source("source/data41.py")
        self.copy_source("source/common41.py")
        self.copy_source("source/models41.py")
        self.copy_source("source/run41.py")
        map_names = protocol["maps"]
        tasks = {}
        for task in TASKS:
            self.copy_source(f"inputs/data41/{task}/public_source41.json")
            selected, selref = self.retain_json(f"output/{task}/selection41.json")
            self.copy_source(f"output/{task}/selection_seal41.json")
            primary, primary_ref = self.retain_json(f"output/{task}/primary_result41.json")
            self.copy_source(f"output/{task}/test/completion_seal41.json")
            stages = {}
            for stage in STAGES:
                prior_base = f"output/{task}/{stage}"
                base = f"inputs/data42/{task}/{stage}"
                fit, fitref = self.retain_json(prior_base + "/fit_manifest41.json")
                self.copy_source(prior_base + "/fit_seal41.json")
                nref = self.copy_source(prior_base + "/normalization41.json", base + "/normalization.json")
                train = self.cohort(task, stage, "train")
                self.copy_source(prior_base + "/data_access_train41.json")
                if stage != "final":
                    evaluation = self.cohort(task, stage, "validation")
                    self.copy_source(prior_base + "/data_access_validation41.json")
                    bundle, prediction_ref = self.retain_json(prior_base + "/validation_predictions41.json")
                    _, score_ref = self.retain_json(prior_base + "/validation_scores41.json")
                    self.copy_source(prior_base + "/completion_seal41.json")
                    evaluation_role = "validation"
                    assessment_features = None
                else:
                    evaluation, prediction_ref, assessment_features = self.assessment(task, primary, fit)
                    score_ref = primary_ref
                    evaluation_role = "assessment"
                maps = {}
                for name in map_names:
                    meta = fit["maps"][name]
                    row = next(x for x in fit["candidate_rows"] if x["map"] == name)
                    mapref = self.copy_source(meta["map_path"], base + f"/maps/{name}.pkl")
                    tr = self.copy_source(meta["training_features"]["path"], base + f"/features/train/{name}.npy")
                    if stage != "final":
                        er = self.copy_source(bundle["maps"][name]["features"]["path"], base + f"/features/validation/{name}.npy")
                    else:
                        er = assessment_features[name]
                    oldfit = self.copy_source(row["fit_path"])
                    oldhead = self.copy_source(row["head_path"])
                    olddeployment = self.copy_source(row["deployment_path"])
                    maps[name] = {"name": name, "family": meta["family"], "scaling": meta["scaling"],
                                  "dimension": meta["dimension"], "feature_id": meta["feature_id"],
                                  "map": mapref, "map_payload_bytes": meta["map_payload_bytes"],
                                  "original_deployment_payload_bytes": row["deployment_payload_bytes"],
                                  "original_static_deployment_bytes": row["static_deployment_bytes"],
                                  "training_features": tr, "evaluation_features": er,
                                  "inherited_fit": oldfit, "inherited_head": oldhead,
                                  "inherited_deployment": olddeployment}
                stages[stage] = {"task": task, "stage": stage, "evaluation_role": evaluation_role,
                                 "train": train, "evaluation": evaluation, "normalization": nref,
                                 "maps": maps, "map_order": map_names,
                                 "inherited_fit_manifest": fitref,
                                 "inherited_predictions": prediction_ref,
                                 "inherited_scores": score_ref, "inherited_selection": selref}
            tasks[task] = {"stages": stages, "prior_primary_result": primary_ref,
                           "prior_selection": selref, "assessment_n": primary["n"]}
        files = {}
        for path in sorted(self.staging.rglob("*")):
            if path.is_file():
                relative = "inputs/" + str(path.relative_to(self.staging))
                files[relative] = {"sha256": digest_file(path), "bytes": path.stat().st_size}
        manifest = {
            "schema": "step42-fixed-input-manifest-v1", "status": "PREPARED_PENDING_INDEPENDENT_INPUT_CHECK",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "study_role": "EXPOSED_PERIOD_DIAGNOSTIC", "prior_outcomes_already_exposed": True,
            "new_primary_generalization_claim": False, "new_source_measurement_tokens_decoded": 0,
            "feature_transforms_performed": 0, "map_fits_performed": 0, "head_fits_performed": 0,
            "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "source_archive_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "source_protocol_seal_sha256": EXPECTED_PROTOCOL_SEAL_SHA256,
            "source_result_sha256": EXPECTED_RESULT_SHA256,
            "source_audit_sha256": self.archive_manifest["independent_audit_sha256"],
            "source_files": self.source_files, "files": files, "tasks": tasks,
            "external_source_records": self.external_sources,
            "map_order": map_names, "preparation_source_sha256": digest_file(Path(__file__)),
            "scope": "Stored matrices and existing revealed targets are reused. No raw archive is decoded, no map is refitted or transformed, and no new head is fitted by input preparation.",
        }
        self.write_json("inputs/manifest42.json", manifest)
        self.staging.replace(self.destination)
        return manifest

    def assessment(self, task, primary, fit):
        base = f"inputs/data42/{task}/final"
        rows = primary["queries"]
        queries, raw_y, norm_y, raw_persistence, norm_persistence, Xmembers = [], [], [], [], [], []
        lag_timestamps, source_queries = [], []
        features = {name: [] for name in fit["maps"]}
        predictions = {}
        normal = fit["normalization"]
        for index, item in enumerate(rows):
            bundle, pref = self.retain_json(item["prediction_path"])
            target, tref = self.retain_json(item["target_path"])
            history, href = self.retain_json(f"output/{task}/test/history/{index:03d}.json")
            seal, sref = self.retain_json(target["prediction_seal_path"])
            query_id = item["id"]
            if not (query_id == bundle["query_id"] == target["query_id"] == history["query_id"]):
                raise ValueError("Inherited assessment query identity mismatch")
            if not (index == bundle["query_index"] == target["query_index"] == history["query_index"]):
                raise ValueError("Inherited assessment query order mismatch")
            queries.append({"id": query_id, "task": task, "target_timestamp": target["target_timestamp"],
                            "origin_timestamp": target["origin_timestamp"]})
            raw_y.append(target["raw_target"])
            norm_y.append(target["y"])
            raw_persistence.append(history["raw_persistence"])
            norm_persistence.append(history["persistence"])
            lag_timestamps.append(history["history_timestamps"])
            Xmembers.append((history["X_path"], query_id))
            source_queries.append({"query_id": query_id, "query_index": index,
                                   "prediction": pref, "target": tref, "history": href, "prediction_seal": sref})
            for name in features:
                features[name].append((bundle["maps"][name]["features"]["path"], query_id))
            for cid, values in bundle["predictions"].items():
                if len(values) != 1 or values[0]["query_id"] != query_id:
                    raise ValueError("Invalid inherited one-query prediction record")
                predictions.setdefault(cid, []).extend(values)
        Xref = self.assemble(base + "/cohorts/assessment_X.npy", Xmembers)
        cohort = {"schema": "step42-reused-causal-cohort-v1", "study_role": "EXPOSED_PERIOD_DIAGNOSTIC",
                  "task": task, "stage": "final", "name": "assessment", "n": len(rows),
                  "ids": [q["id"] for q in queries], "target_timestamps": [q["target_timestamp"] for q in queries],
                  "origin_timestamps": [q["origin_timestamp"] for q in queries], "lag_timestamps": lag_timestamps,
                  "raw_targets": raw_y, "y": norm_y, "raw_persistence": raw_persistence, "persistence": norm_persistence,
                  "X_path": Xref["path"], "X_sha256": Xref["sha256"],
                  "normalization_path": base + "/normalization.json",
                  "normalization_sha256": self.catalog[f"output/{task}/final/normalization41.json"]["sha256"],
                  "source_queries": source_queries, "target_values_clipped": False,
                  "other_measurement_predictors_used": False, "targets_already_revealed_in_step41": True}
        cref = self.write_json(base + "/cohorts/assessment.json", cohort)
        feature_refs = {name: self.assemble(base + f"/features/assessment/{name}.npy", members)
                        for name, members in features.items()}
        pref = self.write_json(base + "/inherited_predictions42.json", {
            "schema": "step42-assembled-prior-predictions-v1", "task": task, "stage": "final",
            "query_ids": cohort["ids"], "predictions": predictions,
            "source_bundles": [x["prediction"] for x in source_queries],
            "recomputed_predictions": False})
        return {"cohort": cref, "X": Xref, "n": len(rows), "ids": cohort["ids"]}, pref, feature_refs


def load_stage(task, stage, root=ROOT):
    """Load the already prepared fixed matrices; does not mutate any file.

    train/evaluation contain ids, exact Fraction y/raw_targets, X and
    features={map_name: Z}; maps retain source feature IDs and payload metadata.
    """
    root = Path(root)
    manifest = read_json(root / "inputs/manifest42.json")
    if task not in TASKS or stage not in STAGES:
        raise ValueError("Unknown task or stage")
    spec = manifest["tasks"][task]["stages"][stage]

    def checked_path(ref):
        path = root / safe_relative(ref["path"])
        if digest_file(path) != ref["sha256"]:
            raise ValueError("Prepared input changed: " + ref["path"])
        return path

    cohorts = {}
    for role, feature_key in (("train", "training_features"), ("evaluation", "evaluation_features")):
        info = spec[role]
        record = read_json(checked_path(info["cohort"]))
        X = np.load(checked_path(info["X"]), allow_pickle=False)
        X.flags.writeable = False
        matrices = {}
        for name in spec["map_order"]:
            z = np.load(checked_path(spec["maps"][name][feature_key]), allow_pickle=False)
            z.flags.writeable = False
            matrices[name] = z
        cohorts[role] = {**record, "y": [Fraction(x) for x in record["y"]],
                          "raw_targets": [Fraction(x) for x in record["raw_targets"]],
                          "X": X, "features": matrices,
                          "path": info["cohort"]["path"], "sha256": info["cohort"]["sha256"]}
    normalization = read_json(checked_path(spec["normalization"]))
    return {"task": task, "stage": stage, "evaluation_role": spec["evaluation_role"],
            **cohorts, "normalization": normalization, "maps": spec["maps"], "map_order": spec["map_order"],
            "inherited_predictions": read_json(checked_path(spec["inherited_predictions"])),
            "inherited_fit_manifest": read_json(checked_path(spec["inherited_fit_manifest"])),
            "inherited_scores": read_json(checked_path(spec["inherited_scores"])),
            "inherited_selection": read_json(checked_path(spec["inherited_selection"])),
            "spec": spec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    args = parser.parse_args()
    m = Preparation(args.prior_root, ROOT / "inputs").prepare()
    print(json.dumps({"status": m["status"], "source_files": len(m["source_files"]),
                      "prepared_files": len(m["files"]), "tasks": list(m["tasks"]),
                      "map_fits": 0, "head_fits": 0, "feature_transforms": 0,
                      "new_source_measurement_tokens_decoded": 0}))


if __name__ == "__main__":
    main()
