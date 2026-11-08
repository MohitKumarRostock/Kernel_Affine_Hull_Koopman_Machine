"""Manufactured-only component integration of the actual Step 43 producer.

This is not a scientific access-gate rehearsal.  It never creates a fake audit
PASS.  A single declared task label is used inside an explicitly artificial
retained fixture; all source values, cohorts and gate responses are manufactured.
Actual native models, serialization, validation, selection, assessment and
misbinding code are executed.  The natural source decode boundary is blocked.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT"):
    os.environ[_name] = "1"

from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
from fractions import Fraction as F
import gzip
import hashlib
import math
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np

import common43 as c
import data43 as d
import model43 as m
import run43 as r


REAL_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REAL_ROOT / "analysis/producer_fixture43"
RECEIPT = REAL_ROOT / "analysis/producer_artificial_checks43.json"
SCOPE = "ARTIFICIAL_COMPONENT_ONLY"
TASK = c.TASKS[0]
IDENTITY = hashlib.sha256(b"Step43 ARTIFICIAL_COMPONENT_ONLY producer integration").hexdigest()
GROUPS = Counter()
ORIGINALS = {}


def check(value, group):
    if not value:
        raise AssertionError(group)
    GROUPS[group] += 1


def exact_risk(values, targets, *, hex_values=True):
    if hex_values:
        predictions = [F.from_float(float.fromhex(x)) for x in values]
    else:
        predictions = [F(x) for x in values]
    check(len(predictions) == len(targets) and len(targets) > 0, "independent_risk_shape")
    differences = [p - y for p, y in zip(predictions, targets)]
    sse = sum((x*x for x in differences), F(0))
    n = len(differences)
    return {"n": n, "sse": str(sse), "mse": str(sse/n),
            "rmse": math.sqrt(float(sse/n)),
            "mae": str(sum(map(abs, differences), F(0))/n)}


def exact_head(deployment, coordinates):
    head = deployment["head"]
    affine = F(head["intercept"]) + sum(
        (F(b)*F.from_float(float(z)) for b, z in zip(head["coefficients"], coordinates)), F(0))
    return F(deployment["target_min"]) + F(deployment["span"])*min(F(1), max(F(0), affine))


def stored_array(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float64), allow_pickle=False)
    return c.file_ref(path)


def load_reference(reference):
    path = FIXTURE / reference["path"]
    check(c.sha_file(path) == reference["sha256"] and path.stat().st_size == reference["bytes"],
          "retained_artificial_reference_integrity")
    return path


class Tee:
    def __init__(self, first, second):
        self.first, self.second = first, second
    def write(self, value):
        self.first.write(value)
        self.first.flush()
        return self.second.write(value)
    def flush(self):
        self.first.flush()
        self.second.flush()


class ArtificialData:
    def __init__(self):
        self.events = []
        self.training_loaded = set()
        self.validation_loaded = set()
        self.next_query = 0
        self.pending = None
        self.gate_calls = []
        self.natural_decode_attempts = 0
        rng = np.random.default_rng(430344)
        start = datetime(2001, 1, 1)
        self.values = {}
        for hour in range(61*24):
            stamp = start + timedelta(hours=hour)
            value = (F(100) + F(hour, 96) + F(int(rng.integers(-2048, 2049)), 512)
                     + F(3*(hour % 24), 16))
            self.values[stamp.isoformat(timespec="seconds")] = value
        self.queries = []
        for index in range(56):
            target = datetime(2001, 1, 5, 23) + timedelta(days=index)
            origin = target - timedelta(hours=24)
            self.queries.append({
                "id": f"ARTIFICIAL_COMPONENT_ONLY:{TASK}:{target.isoformat(timespec='seconds')}",
                "task": TASK, "target_timestamp": target.isoformat(timespec="seconds"),
                "origin_timestamp": origin.isoformat(timespec="seconds")})
        self.sizes = {"f1": 32, "f2": 39, "f3": 46, "final": 53}
        self.cohorts = {}
        self.assessment = self.queries[53:]
        c.write_json(FIXTURE / "manufactured_series43.json", {
            "schema": "step43-manufactured-series-v1", "scope": SCOPE,
            "generator_seed": 430344, "natural_measurements": False, "values": self.values})
        for stage, n in self.sizes.items():
            normal_targets = [self.values[q["target_timestamp"]] for q in self.queries[:n]]
            lo, hi = min(normal_targets), max(normal_targets)
            normal = {"schema": "step43-training-target-normalization-v1", "scope": SCOPE,
                      "task": TASK, "stage": stage, "fitted_partition": "train",
                      "target_min": lo, "target_max": hi, "span": hi-lo,
                      "future_data_used": False, "target_values_clipped": False}
            c.write_json(FIXTURE / f"output/data43/{TASK}/{stage}/normalization43.json", normal)
            self.cohorts[stage, "train"] = self.make_cohort(stage, "train", self.queries[:n], normal)
            if stage != "final":
                self.cohorts[stage, "validation"] = self.make_cohort(
                    stage, "validation", self.queries[n:n+7], normal)
        c.write_json(FIXTURE / "fixture_manifest43.json", {
            "schema": "step43-producer-manufactured-fixture-v1", "scope": SCOPE,
            "scientific_audit_gate": False, "task_label": TASK, "seeds": list(c.SEEDS),
            "development_folds": [{"stage": s, "training_rows": self.sizes[s], "validation_rows": 7}
                                  for s in ("f1", "f2", "f3")],
            "final_training_rows": 53, "assessment_queries": self.assessment,
            "protocol_identity_is_artificial": True, "artificial_protocol_identifier": IDENTITY,
            "all_values_manufactured": True, "real_numeric_decode_boundary_blocked": True,
            "data_access_stub_scope": "Interface order/schema checks only; genuine data gates tested separately"})

    def history_values(self, query):
        origin = datetime.fromisoformat(query["origin_timestamp"])
        return [self.values[(origin-timedelta(hours=23-j)).isoformat(timespec="seconds")] for j in range(24)]

    def make_cohort(self, stage, partition, queries, normal):
        histories = [self.history_values(query) for query in queries]
        targets = [self.values[q["target_timestamp"]] for q in queries]
        base = FIXTURE / f"output/data43/{TASK}/{stage}"
        Xref = stored_array(base / f"{partition}_X.npy", histories)
        access_path = base / f"{partition}_access43.json"
        c.write_json(access_path, {"schema": "step43-artificial-access-v1", "scope": SCOPE,
                                 "task": TASK, "stage": stage, "partition": partition,
                                 "natural_values_decoded": 0})
        normal_ref = c.file_ref(base / "normalization43.json")
        record = {"schema": "step43-causal-cohort-v1", "scope": SCOPE,
                  "task": TASK, "stage": stage, "partition": partition,
                  "protocol_seal_sha256": IDENTITY,
                  "n": len(queries), "ids": [q["id"] for q in queries], "queries": queries,
                  "X": Xref, "raw_targets": targets,
                  "y": [(x-normal["target_min"])/normal["span"] for x in targets],
                  "raw_persistence": [row[-1] for row in histories],
                  "persistence": [(row[-1]-normal["target_min"])/normal["span"] for row in histories],
                  "normalization": normal_ref, "access": c.file_ref(access_path),
                  "other_measurement_predictors_used": False, "target_values_clipped": False}
        path = base / f"{partition}.json"
        c.write_json(path, record)
        X = np.asarray(histories, dtype=np.float64)
        X.flags.writeable = False
        return {**record, "X": X, "normalization": normal,
                "path": path.relative_to(FIXTURE).as_posix(), "sha256": c.sha_file(path)}

    def record_event(self, kind, **fields):
        event = {"index": len(self.events), "scope": SCOPE, "kind": kind, **fields}
        self.events.append(event)
        c.event(kind, log="output/artificial_interface_events43.jsonl", scope=SCOPE, **fields)

    def forbid_decode(self, *args, **kwargs):
        self.natural_decode_attempts += 1
        raise AssertionError("Natural measurement decoding is forbidden in this artificial component test")

    def training(self, task, stage):
        check(task == TASK and stage in self.sizes and stage not in self.training_loaded,
              "training_cohort_interface_once")
        check((FIXTURE / f"output/{task}/{stage}/stage_started43.json").is_file(),
              "producer_stage_marker_precedes_training")
        if stage == "final":
            roster = c.read_json(FIXTURE / "output/selected_roster43.json")
            check(len(roster["tasks"][TASK]["selections"]) == 18, "selection_precedes_final_training")
        self.training_loaded.add(stage)
        self.record_event("artificial_training_cohort_supplied", task=task, stage=stage)
        return self.cohorts[stage, "train"]

    def validation(self, task, stage):
        check(task == TASK and stage in ("f1", "f2", "f3") and stage not in self.validation_loaded,
              "validation_cohort_interface_once")
        seal = c.read_json(FIXTURE / f"output/{task}/{stage}/fit_seal43.json")
        check(seal["schema"] == "step43-stage-fit-seal-v1" and seal["task"] == task
              and seal["stage"] == stage and seal["seeds"] == list(c.SEEDS)
              and seal["protocol_seal_sha256"] == IDENTITY, "all_seed_fit_seal_before_validation")
        c.verify_bindings(seal["bound_files"])
        for seed in c.SEEDS:
            bank = c.read_json(FIXTURE / f"output/{task}/{stage}/seed_{seed}/fits43.json")
            check(len(bank["affine_rows"]) == 90 and len(bank["baseline_rows"]) == 44,
                  "complete_grid_before_validation")
            check(not (FIXTURE / f"output/{task}/{stage}/seed_{seed}/validation_predictions43.json").exists(),
                  "no_validation_predictions_before_validation_interface")
        self.validation_loaded.add(stage)
        self.record_event("artificial_validation_cohort_supplied", task=task, stage=stage,
                          fit_seal_sha256=c.sha_file(FIXTURE / f"output/{task}/{stage}/fit_seal43.json"))
        return self.cohorts[stage, "validation"]

    def require_gate(self, relative, **bindings):
        check(relative == "audit/preassessment43.json", "only_expected_component_gate_stub")
        check(bindings.get("protocol_seal_sha256") == IDENTITY, "component_gate_artificial_identity")
        selection = FIXTURE / "output/selection_seal43.json"
        check(selection.is_file() and bindings.get("selection_seal_sha256") == c.sha_file(selection),
              "selection_seal_precedes_component_assessment")
        check(not (FIXTURE / "audit").exists(), "no_fake_scientific_PASS_gate_created")
        self.gate_calls.append({"relative": relative, "bindings": bindings, "scope": SCOPE})
        return {"scope": SCOPE, "scientific_audit_gate": False, **bindings}

    def history(self, task, index):
        check(task == TASK and index == self.next_query and self.pending is None,
              "chronological_assessment_history_interface")
        check(bool(self.gate_calls), "component_gate_precedes_history_interface")
        query = self.assessment[index]
        raw_values = self.history_values(query)
        X = np.asarray([[float(x) for x in raw_values]], dtype=np.float64)
        X.flags.writeable = False
        base = FIXTURE / f"output/data43/{task}/assessment/queries/{index:04d}"
        Xref = stored_array(base / "history_X.npy", X)
        normal = self.cohorts["final", "train"]["normalization"]
        record = {"schema": "step43-causal-assessment-history-v1", "scope": SCOPE,
                  "task": task, "query_id": query["id"], "query_index": index,
                  "target_timestamp": query["target_timestamp"], "origin_timestamp": query["origin_timestamp"],
                  "protocol_seal_sha256": IDENTITY, "X": Xref,
                  "raw_values": raw_values, "raw_persistence": raw_values[-1],
                  "persistence": (raw_values[-1]-normal["target_min"])/normal["span"],
                  "current_target_decoded": False}
        path = base / "history43.json"
        c.write_json(path, record)
        self.pending = {"index": index, "history": c.file_ref(path), "query": query}
        self.record_event("artificial_history_supplied", task=task, query_index=index, query_id=query["id"],
                          current_target_supplied=False)
        return {**record, "X": X, "raw_X": X, "normalization": normal,
                "history_receipt_path": path.relative_to(FIXTURE).as_posix(),
                "history_receipt_sha256": c.sha_file(path)}

    def reveal(self, task, index, prediction_path):
        check(task == TASK and index == self.next_query and self.pending["index"] == index,
              "target_interface_after_current_history")
        expected_path = f"output/assessment/{task}/queries/{index:04d}/predictions43.json"
        check(prediction_path == expected_path, "prediction_commit_path")
        prediction = c.read_json(FIXTURE / prediction_path)
        selected = c.read_json(FIXTURE / "output/selected_roster43.json")["tasks"][task]["selections"]
        query = self.assessment[index]
        check(prediction["schema"] == "step43-query-predictions-v1" and prediction["query_id"] == query["id"]
              and prediction["query_index"] == index and prediction["task"] == task
              and prediction["protocol_seal_sha256"] == IDENTITY
              and prediction["history_receipt_sha256"] == self.pending["history"]["sha256"]
              and prediction["history"] == self.pending["history"], "prediction_commit_identity")
        check(set(prediction["predictions"]) == set(selected) and len(selected) == 18,
              "all_18_aliases_committed_before_target")
        aliases = []
        expected_evaluations = set()
        for alias, selection in selected.items():
            reported = prediction["predictions"][alias]
            check(reported["candidate_id"] == selection["candidate_id"], "selected_alias_identity_before_target")
            value = float.fromhex(reported["reported_physical_hex"])
            check(math.isfinite(value) and value.hex() == reported["reported_physical_hex"],
                  "canonical_finite_reported_prediction")
            bank = c.read_json(FIXTURE / f"output/{task}/final/seed_{selection['seed']}/fits43.json")
            cid = selection["candidate_id"]
            if cid.startswith("baseline::"):
                baseline = next(row for row in bank["baseline_rows"] if row["candidate_id"] == cid)
                eid = "baseline::" + c.digest({"candidate_id": cid, "pickle_sha256": baseline["pickle_sha256"]})
            else:
                heads = c.read_json(load_reference(bank["heads"]))
                eid = "affine::" + heads[cid]["deployment"]["deployment_id"]
            check(reported["evaluation_id"] == eid, "exact_evaluation_identity_before_target")
            expected_evaluations.add(eid)
            evaluation = prediction["evaluations"][eid]
            check(reported["reported_physical_hex"] == evaluation["reported_physical_hex"],
                  "shared_alias_reported_value")
            check(alias in evaluation["aliases"], "evaluation_alias_membership")
        check(set(prediction["evaluations"]) == expected_evaluations, "one_evaluation_per_actual_identity")
        for evaluation in prediction["evaluations"].values():
            aliases.extend(evaluation["aliases"])
        check(sorted(aliases) == sorted(selected), "deduplicated_evaluations_partition_18_aliases")
        check(len(expected_evaluations) < 18, "all_map_alias_dedup_exercised")
        self.record_event("all_18_artificial_predictions_observed_before_target", task=task, query_index=index,
                          prediction=c.file_ref(FIXTURE / prediction_path), evaluations=len(expected_evaluations))
        raw_target = self.values[query["target_timestamp"]]
        normal = self.cohorts["final", "train"]["normalization"]
        path = FIXTURE / f"output/data43/{task}/assessment/queries/{index:04d}/target43.json"
        record = {"schema": "step43-revealed-assessment-target-v1", "scope": SCOPE,
                  "task": task, "query_id": query["id"], "query_index": index,
                  "raw_target": raw_target, "y": (raw_target-normal["target_min"])/normal["span"],
                  "prediction": c.file_ref(FIXTURE / prediction_path), "natural_measurement_decoded": False}
        c.write_json(path, record)
        self.record_event("artificial_target_supplied", task=task, query_index=index, target=c.file_ref(path))
        self.next_query += 1
        self.pending = None
        return {**record, "path": path.relative_to(FIXTURE).as_posix(), "sha256": c.sha_file(path)}


def patch_component_root():
    for name in ("file_ref", "seal_files", "verify_bindings", "event", "protocol_identity", "require_gate"):
        ORIGINALS[name] = getattr(c, name)
    c.ROOT = FIXTURE
    c.TASKS = (TASK,)
    c.file_ref = lambda path, **kwargs: ORIGINALS["file_ref"](path, root=FIXTURE)
    c.seal_files = lambda path, files, **kwargs: ORIGINALS["seal_files"](path, files, root=FIXTURE, **kwargs)
    c.verify_bindings = lambda bindings, **kwargs: ORIGINALS["verify_bindings"](bindings, root=FIXTURE)
    c.event = lambda kind, **kwargs: ORIGINALS["event"](kind, root=FIXTURE, **kwargs)
    c.protocol_identity = lambda **kwargs: IDENTITY
    r.ROOT, r.OUT = FIXTURE, FIXTURE / "output"


def verify_validation(data):
    expected_configs = {row["candidate_id"]: row for row in m.affine_configs()}
    validated = 0
    for stage in ("f1", "f2", "f3"):
        cohort = data.cohorts[stage, "validation"]
        for seed in c.SEEDS:
            base = FIXTURE / f"output/{TASK}/{stage}/seed_{seed}"
            bank = c.read_json(base / "fits43.json")
            predictions = c.read_json(base / "validation_predictions43.json")
            scores = c.read_json(base / "validation_scores43.json")
            check(bank["affine_rows"] == m.affine_configs(), "actual_affine_grid_roundtrip")
            check(len(bank["maps"]) == 18 and len(bank["baseline_rows"]) == 44,
                  "actual_native_and_baseline_grid_roundtrip")
            check(predictions["query_ids"] == cohort["ids"] and scores["n"] == 7,
                  "validation_query_roster_roundtrip")
            check(predictions["validation_cohort"] == c.file_ref(FIXTURE / cohort["path"]),
                  "validation_exact_cohort_binding")
            check(scores["prediction_sha256"] == c.sha_file(base / "validation_predictions43.json"),
                  "score_prediction_binding")
            check(len(scores["candidate_rows"]) == len(predictions["candidate_predictions"]) == 134,
                  "all_134_candidates_scored")
            heads = c.read_json(load_reference(bank["heads"]))
            for cid, prediction in predictions["candidate_predictions"].items():
                score = scores["candidate_rows"][cid]
                check(score["physical_risk"] == exact_risk(prediction["reported_physical_hex"], cohort["raw_targets"]),
                      "independent_validation_reported_loss")
                if cid in expected_configs:
                    check(score["configuration"] == expected_configs[cid], "validation_affine_identity")
                    Z = np.load(load_reference(predictions["maps"][expected_configs[cid]["map"]]), allow_pickle=False)
                    refs = [exact_head(heads[cid]["deployment"], z) for z in Z]
                    check(prediction["exact_physical"] == list(map(str, refs)),
                          "independent_validation_exact_affine_reference")
                    check(score["exact_reference_risk"] == exact_risk(refs, cohort["raw_targets"], hex_values=False),
                          "independent_validation_reference_loss")
                else:
                    check(prediction["exact_physical"] is None and score["exact_reference_risk"] is None,
                          "baseline_has_no_affine_certificate_claim")
                validated += 1
    return validated


def verify_selection(data):
    roster = c.read_json(FIXTURE / "output/selected_roster43.json")
    check(set(roster["tasks"]) == {TASK} and roster["assessment_targets_accessed"] == 0,
          "artificial_single_task_development_only_roster")
    check(len(roster["tasks"][TASK]["selections"]) == 18, "selected_all_18_policy_seed_aliases")
    counts = {"raw": 10, "nystrom": 20, "kahm": 60, "all_map": 90, "ordinary": 44, "ridge": 5}
    for seed in c.SEEDS:
        folds = [c.read_json(FIXTURE / f"output/{TASK}/{stage}/seed_{seed}/validation_scores43.json")
                 for stage in ("f1", "f2", "f3")]
        result = c.read_json(FIXTURE / f"output/{TASK}/selection_seed_{seed}43.json")
        independent = {}
        for cid in folds[0]["candidate_rows"]:
            parts = [fold["candidate_rows"][cid]["physical_risk"] for fold in folds]
            n = sum(row["n"] for row in parts)
            sse = sum((F(row["sse"]) for row in parts), F(0))
            sae = sum((F(row["mae"])*row["n"] for row in parts), F(0))
            expected = {"n": n, "sse": str(sse), "mse": str(sse/n),
                        "rmse": math.sqrt(float(sse/n)), "mae": str(sae/n)}
            check(result["candidate_rows"][cid]["physical_risk"] == expected,
                  "independent_pooled_development_risk")
            independent[cid] = F(expected["mse"])
        for policy in c.POLICIES:
            if policy in ("raw", "nystrom", "kahm"):
                pool = [cid for cid in independent if cid.startswith(policy+"_")]
            elif policy == "all_map":
                pool = [cid for cid in independent if not cid.startswith("baseline::")]
            elif policy == "ordinary":
                pool = [cid for cid in independent if cid.startswith("baseline::")]
            else:
                pool = [cid for cid in independent if cid.startswith("baseline::ridge_")]
            choice = min(pool, key=lambda cid: (independent[cid], cid))
            check(len(pool) == counts[policy], "independent_fixed_selection_pool")
            check(result["selections"][policy]["candidate_id"] == choice
                  and result["selections"][policy]["pool_size"] == len(pool),
                  "independent_selection_and_tie_break")
            check(roster["tasks"][TASK]["selections"][f"{seed}::{policy}"]["candidate_id"] == choice,
                  "pooled_selection_global_roster_identity")
        check(result["development_n"] == 21, "development_selection_count")
    return roster


def verify_assessment(data, roster):
    result = c.read_json(FIXTURE / f"output/assessment/{TASK}/assessment43.json")
    check(result["n"] == data.next_query == 3 and data.pending is None, "assessment_completed_once")
    check(set(result["monthly_results"]) == {"2001-02", "2001-03"}, "monthly_assessment_partition_exercised")
    targets = [data.values[q["target_timestamp"]] for q in data.assessment]
    queries = [c.read_json(load_reference(row["prediction"])) for row in result["queries"]]
    unique_affine_evaluations = 0
    for query in queries:
        for evaluation in query["evaluations"].values():
            if evaluation["recipe"] == "affine":
                Z = np.load(load_reference(evaluation["features"]), allow_pickle=False)[0]
                reference = exact_head(evaluation["deployment"], Z)
                honest = evaluation["assay"]["cases"]["binary64"]
                check(F(honest["exact_physical"]) == reference, "independent_assessment_exact_reference")
                reported = F.from_float(float.fromhex(evaluation["reported_physical_hex"]))
                check(F(honest["absolute_physical_error"]) == abs(reported-reference),
                      "independent_assessment_absolute_error")
                budget = F(evaluation["deployment"]["span"]) / (1 << 40)
                check(honest["accepted"] == (abs(reported-reference) <= budget),
                      "independent_assessment_numerical_budget")
                unique_affine_evaluations += 1
    for alias in roster["tasks"][TASK]["selections"]:
        reported = [query["predictions"][alias]["reported_physical_hex"] for query in queries]
        check(result["policy_results"][alias]["physical_risk"] == exact_risk(reported, targets),
              "independent_assessment_policy_risk")
        for month in result["monthly_results"]:
            indices = [i for i, query in enumerate(data.assessment) if query["target_timestamp"].startswith(month)]
            check(result["monthly_results"][month][alias] == exact_risk(
                [reported[i] for i in indices], [targets[i] for i in indices]),
                "independent_monthly_policy_risk")
    raw = result["policy_results"]["3901::raw"]["physical_risk"]
    for policy in ("ordinary", "ridge"):
        other = result["policy_results"]["3901::"+policy]["physical_risk"]
        comparison = result["primary_comparisons"][policy]
        a, b = F(raw["sse"]), F(other["sse"])
        check(comparison == {"raw_risk": raw, "reference_risk": other,
                             "mse_ratio": str(a/b) if b else None, "within_five_percent_mse": 20*a <= 21*b},
              "independent_primary_exact_inequality")
    check(result["certificate_summary"]["binary64"]["packets"] == unique_affine_evaluations,
          "certificate_counts_deduplicated_actual_evaluations")
    for index in range(3):
        sequence = [e["kind"] for e in data.events if e.get("query_index") == index]
        check(sequence == ["artificial_history_supplied",
                           "all_18_artificial_predictions_observed_before_target",
                           "artificial_target_supplied"], "commit_before_target_interface_sequence")
    misbinding = c.read_json(FIXTURE / f"output/assessment/{TASK}/misbinding43.json")
    expected = {eid for eid, value in queries[0]["evaluations"].items() if value["recipe"] == "affine"}
    check(set(misbinding["records"]) == expected, "all_selected_unique_heads_misbinding_exercised")
    check(misbinding["performed_after_assessment"] == c.file_ref(
        FIXTURE / f"output/assessment/{TASK}/assessment43.json"), "misbinding_after_assessment_binding")
    return {"assessment_queries": 3, "forecast_aliases_per_query": 18,
            "unique_affine_query_evaluations": unique_affine_evaluations,
            "selected_misbinding_heads": len(expected)}


def run():
    patch_component_root()
    data = ArtificialData()
    d._decode_authorized_values = data.forbid_decode
    d.load_training, d.load_validation = data.training, data.validation
    d.assessment_queries = lambda task: list(data.assessment)
    d.load_assessment_history, d.reveal_assessment_target = data.history, data.reveal
    c.require_gate = data.require_gate
    for stage in ("f1", "f2", "f3"):
        r.stage(TASK, stage)
    validated = verify_validation(data)
    r.select()
    roster = verify_selection(data)
    r.stage(TASK, "final")
    for seed in c.SEEDS:
        bank = c.read_json(FIXTURE / f"output/{TASK}/final/seed_{seed}/fits43.json")
        wanted = {roster["tasks"][TASK]["selections"][f"{seed}::{policy}"]["candidate_id"] for policy in c.POLICIES}
        actual = {row["candidate_id"] for row in bank["affine_rows"]+bank["baseline_rows"]}
        check(actual == wanted and len(bank["affine_rows"]) == 3, "final_fit_only_selected_identities")
    r.seal_selection()
    r.assess(TASK)
    r.misbinding(TASK)
    metrics = verify_assessment(data, roster)
    check(data.natural_decode_attempts == 0, "real_numeric_decode_boundary_never_called")
    check(not (FIXTURE / "audit").exists(), "no_fake_scientific_audit_files")
    check(not (REAL_ROOT / "output/data43").exists(), "no_real_numeric_outputs_created")
    c.write_json(FIXTURE / "component_gate_stub_calls43.json", {
        "scope": SCOPE, "scientific_audit_gate": False, "calls": data.gate_calls})
    return {"development_fit_banks": 9, "final_selected_fit_banks": 3,
            "development_affine_fits": 810, "development_baseline_fits": 396,
            "independently_verified_validation_candidates": validated,
            "natural_measurement_values_decoded": 0, "real_model_fits": 0,
            "all_model_fits_used_manufactured_values": True, **metrics}


def main():
    if FIXTURE.exists() or RECEIPT.exists():
        raise FileExistsError("Retain existing producer artificial evidence; no silent replacement")
    if (REAL_ROOT / "protocol_seal43.json").exists():
        raise RuntimeError("Producer integration must complete before the final scientific source freeze")
    FIXTURE.mkdir(parents=True)
    start = time.perf_counter()
    source_names = ("source/common43.py", "source/data43.py", "source/model43.py", "source/run43.py",
                    "source/test_producer43.py")
    before = {name: c.sha_file(REAL_ROOT / name) for name in source_names}
    manifest_before = c.sha_file(REAL_ROOT / "inputs/data43/manifest43.json")
    with (FIXTURE / "stdout.log").open("x") as out, (FIXTURE / "stderr.log").open("x") as err:
        with redirect_stdout(Tee(out, sys.stdout)), redirect_stderr(Tee(err, sys.stderr)):
            try:
                metrics = run()
                check(before == {name: c.sha_file(REAL_ROOT / name) for name in source_names},
                      "tested_source_bytes_unchanged_during_component_run")
                check(c.sha_file(REAL_ROOT / "inputs/data43/manifest43.json") == manifest_before,
                      "real_data_manifest_unchanged")
                receipt = {"schema": "step43-producer-artificial-checks-v1", "status": "PASS", "scope": SCOPE,
                           "utc": c.utc_now(), "wall_seconds": time.perf_counter()-start,
                           "checks": sum(GROUPS.values()), "groups": dict(sorted(GROUPS.items())),
                           "source_bindings": before, "real_data_manifest_sha256_unchanged": manifest_before,
                           "fixture_root": FIXTURE.relative_to(REAL_ROOT).as_posix(),
                           "no_fake_scientific_PASS_gates": True,
                           "controller_and_real_data_gates_rehearsed": False, **metrics}
                c.write_json(RECEIPT, receipt)
                print(c.canonical_bytes(receipt).decode(), end="", flush=True)
            except Exception as failure:
                record = {"schema": "step43-producer-artificial-failure-v1", "status": "FAIL", "scope": SCOPE,
                          "type": type(failure).__name__, "reason": str(failure),
                          "traceback": traceback.format_exc(), "source_bindings": before,
                          "checks_completed": sum(GROUPS.values()), "wall_seconds": time.perf_counter()-start}
                c.write_json(FIXTURE / "failure43.json", record)
                print(c.canonical_bytes(record).decode(), end="", flush=True)
                raise


if __name__ == "__main__":
    main()
