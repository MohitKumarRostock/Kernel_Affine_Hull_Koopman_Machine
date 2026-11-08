"""Synthetic end-to-end runner integration, isolated from all public datasets.

Actual fit/validation/selection/final-refit/test/aggregation producer functions
are used.  Only the data provider, evidence root, and scientific access gates
are replaced.  Independent auditor APIs check the resulting models, forecasts,
losses and selections before each synthetic PASS gate is written.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

from datetime import datetime, timedelta
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import time

import numpy as np

import common40 as c
import learners39 as m
import protocol40 as plan
import run40 as run
import audit40 as audit


SOURCE_ROOT = Path(__file__).resolve().parents[1]
TASK = "artificial_task40"
COUNTS = {"f1": 64, "f2": 72, "f3": 80, "final": 88}
VALIDATION_COUNTS = {"f1": 1, "f2": 2, "f3": 3}


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    names = ("check_runner40.py", "run40.py", "common40.py", "protocol40.py", "decisions40.py", "learners39.py",
             "learners38.py", "full_graph37.py", "finite_feature35.py", "representations35.py")
    return {"source/" + name: sha(SOURCE_ROOT / "source" / name) for name in names}


def raw_value(stamp):
    hour = int((stamp - datetime(2017, 1, 1)).total_seconds() // 3600)
    return F(2000 + (hour * 17) % 271 + (hour * hour) % 19 + hour // 24, 10)


def query_for(index, *, test=False):
    target = datetime(2017, 9, 1, 23) + timedelta(days=index) if test else datetime(2017, 1, 2, 23) + timedelta(days=index)
    origin = target - timedelta(hours=24)
    lags = [origin - timedelta(hours=23 - j) for j in range(24)]
    return {"id": "synthetic_" + target.isoformat(timespec="seconds"),
            "target_timestamp": target.isoformat(timespec="seconds"),
            "origin_timestamp": origin.isoformat(timespec="seconds"),
            "lag_timestamps": [v.isoformat(timespec="seconds") for v in lags]}


def verify_seal(path):
    record = c.read(path)
    for relative, wanted in record["bound_files"].items():
        require(c.sha(c.ROOT / relative) == wanted, "Synthetic sealed artifact changed")
    return record


class SyntheticData:
    """Deterministic artificial inputs; target access follows real stage order."""
    def __init__(self, root, protocol):
        self.root = root
        self.protocol = protocol
        self.cohorts = {}
        self.norms = {}
        self.pending = None
        self.revealed = []
        self.histories = {}
        self.events = []
        self.official_measurement_accesses = 0

    def task_ids(self):
        return (TASK,)

    def event(self, name, **fields):
        record = {"event": name, "utc": c.utc(), "previous_event_sha256":
                  self.events[-1]["event_sha256"] if self.events else None,
                  "artificial_only": True, **fields}
        record["event_sha256"] = c.digest(record)
        self.events.append(record)
        path = self.root / "output/access_and_execution40.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            stream.write(c.canonical_bytes(record))
        return record

    def gate(self, relative):
        record = c.read(self.root / relative)
        require(record["status"] == "PASS" and record["artificial_only"] is True and
                record["protocol_seal_sha256"] == c.sha(self.root / "protocol_seal40.json"),
                "Synthetic provider lacks independently checked stage gate")
        return record

    def make_cohort(self, task, stage, partition):
        require(task == TASK and stage in COUNTS, "Synthetic task/stage differs")
        key = (stage, partition)
        require(key not in self.cohorts, "Synthetic cohort was requested twice")
        if partition == "train":
            indices = range(COUNTS[stage])
        else:
            indices = range(COUNTS[stage], COUNTS[stage] + VALIDATION_COUNTS[stage])
        queries = [query_for(i) for i in indices]
        X = np.asarray([[float(raw_value(datetime.fromisoformat(stamp))) for stamp in q["lag_timestamps"]]
                        for q in queries], dtype=np.float64)
        raw_targets = [raw_value(datetime.fromisoformat(q["target_timestamp"])) for q in queries]
        raw_persistence = [raw_value(datetime.fromisoformat(q["origin_timestamp"])) for q in queries]
        base = self.root / f"output/{task}/{stage}"
        if partition == "train":
            low, high = min(raw_targets), max(raw_targets)
            require(low < high, "Synthetic target span must be positive")
            normal = {"schema": "step40-artificial-training-normalization-v1", "task": task, "stage": stage,
                      "minimum": str(low), "maximum": str(high), "span": str(high - low),
                      "source_role": "train_only", "n": len(raw_targets)}
            self.norms[stage] = normal
            c.save(base / "normalization40.json", normal)
        normal = self.norms[stage]
        low, span = F(normal["minimum"]), F(normal["span"])
        y = [(v - low) / span for v in raw_targets]
        persistence = [(v - low) / span for v in raw_persistence]
        path = base / f"cohorts/{partition}.json"
        array_path = base / f"cohorts/{partition}_X.npy"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        with array_path.open("xb") as stream:
            np.save(stream, X, allow_pickle=False)
        record = {"schema": "step40-artificial-cohort-v1", "task": task, "stage": stage, "partition": partition,
                  "n": len(y), "ids": [q["id"] for q in queries], "y": list(map(str, y)),
                  "raw_targets": list(map(str, raw_targets)), "persistence": list(map(str, persistence)),
                  "raw_persistence": list(map(str, raw_persistence)),
                  "target_timestamps": [q["target_timestamp"] for q in queries],
                  "origin_timestamps": [q["origin_timestamp"] for q in queries],
                  "lag_timestamps": [q["lag_timestamps"] for q in queries],
                  "X_path": c.relative(array_path), "X_sha256": c.sha(array_path),
                  "artificial_only": True, "official_measurement_accesses": 0}
        ref = c.save(path, record)
        cohort = {**record, "X": X, "y": y, "persistence": persistence,
                  "normalization": normal, "path": ref["path"], "sha256": ref["sha256"]}
        self.cohorts[key] = cohort
        self.event("artificial_" + partition + "_cohort_supplied", task=task, stage=stage, n=len(y))
        return cohort

    def load_training(self, task, stage):
        if stage == "final":
            self.gate(f"audit/{task}/selection_gate40.json")
            verify_seal(self.root / f"output/{task}/selection_seal40.json")
        else:
            self.gate("audit/preexecution_gate40.json")
            if stage != "f1":
                previous = "f1" if stage == "f2" else "f2"
                self.gate(f"audit/{task}/{previous}/validation_gate40.json")
        return self.make_cohort(task, stage, "train")

    def load_validation(self, task, stage):
        gate = self.gate(f"audit/{task}/{stage}/fit_gate40.json")
        path = self.root / f"output/{task}/{stage}/fit_seal40.json"
        require(gate["fit_seal_sha256"] == c.sha(path), "Synthetic validation gate has another fitted model")
        verify_seal(path)
        return self.make_cohort(task, stage, "validation")

    def test_queries(self, task):
        self.gate(f"audit/{task}/pretest_gate40.json")
        return [query_for(i, test=True) for i in range(2)]

    def load_test_history(self, task, query):
        self.gate(f"audit/{task}/pretest_gate40.json")
        index = len(self.revealed)
        require(self.pending is None and query == query_for(index, test=True), "Synthetic test query reordered or replayed")
        X = np.asarray([[float(raw_value(datetime.fromisoformat(stamp))) for stamp in query["lag_timestamps"]]], dtype=np.float64)
        normal = self.norms["final"]
        raw = raw_value(datetime.fromisoformat(query["origin_timestamp"]))
        persistence = (raw - F(normal["minimum"])) / F(normal["span"])
        path = self.root / f"output/{task}/test/history/{index:03d}.json"
        record = {"schema": "step40-artificial-history-v1", "query_id": query["id"],
                  "origin_timestamp": query["origin_timestamp"], "target_timestamp": query["target_timestamp"],
                  "history_timestamps": query["lag_timestamps"], "current_target_value_decoded": False,
                  "persistence": str(persistence), "raw_persistence": str(raw), "artificial_only": True}
        ref = c.save(path, record)
        history = {**record, "X": X, "persistence": persistence, "normalization": normal,
                   "history_receipt_path": ref["path"], "history_receipt_sha256": ref["sha256"]}
        self.pending = query
        self.histories[query["id"]] = history
        self.event("artificial_history_supplied", task=task, query_id=query["id"], target_value_accessed=False)
        return history

    def reveal_test_target(self, task, query, prediction_path, prediction_sha256):
        require(self.pending == query, "Synthetic target revealed without current history")
        path = self.root / prediction_path
        require(path.is_file() and c.sha(path) == prediction_sha256, "Synthetic target revealed before prediction byte seal")
        prediction = c.read(path)
        require(prediction["query_id"] == query["id"] and
                set(prediction["predictions"]) == set(self.protocol["evaluation"]["prediction_ids"]) and
                len(prediction["optimizer_predictions"]) == 18 and
                prediction["sealed_before_current_target_reveal"] is True,
                "Synthetic target revealed without complete fixed forecast panel")
        raw = raw_value(datetime.fromisoformat(query["target_timestamp"]))
        normal = self.norms["final"]
        value = (raw - F(normal["minimum"])) / F(normal["span"])
        record = {"schema": "step40-artificial-target-v1", "query_id": query["id"],
                  "raw_target": str(raw), "y": str(value), "target_value_clipped": False,
                  "prediction_path": prediction_path, "prediction_sha256": prediction_sha256,
                  "artificial_only": True}
        ref = c.save(self.root / f"output/{task}/test/targets/{len(self.revealed):03d}.json", record)
        self.revealed.append(record)
        self.pending = None
        self.event("artificial_target_revealed", task=task, query_id=query["id"], prediction_sha256=prediction_sha256)
        return {**record, "y": value, "path": ref["path"], "sha256": ref["sha256"]}


def protocol_fixture():
    names = plan.maps()
    configs = c.head_configs(names)
    baselines = plan.baselines()
    return {"schema": "step40-artificial-runner-protocol-v1", "artificial_only": True, "seed": 3901,
            "tasks": [TASK], "maps": names, "representations": run.rep.DEFAULT_CONFIG,
            "heads": {"configurations": configs, "solver_settings": m.old.g.SOLVER_SETTINGS,
                      "affine_ridge_lambda": "1/1000000"},
            "baselines": {"configurations": baselines},
            "selection": {"folds": ["f1", "f2", "f3"], "selectors": list(c.SELECTORS)},
            "evaluation": {"prediction_ids": [row["candidate_id"] for row in configs + baselines],
                           "months_by_task": {TASK: ["2017-09"]}},
            "execution": {"native_representation_budget_seconds": 300,
                          "phase_budget_seconds": {"fit": 1800, "validate": 1800, "select": 120, "test": 1800, "aggregate": 120}}}


def gate(path, **fields):
    value = {"schema": "step40-artificial-independent-gate-v1", "status": "PASS", "artificial_only": True,
             "protocol_seal_sha256": c.sha(c.ROOT / "protocol_seal40.json"),
             "scope": "Synthetic producer integration; no official data or scientific gates", **fields}
    c.save(path, value)
    return value


def score_and_selection_edge_cases(protocol, root):
    # Two unequal-sized folds with different physical spans produce opposite
    # rankings under correct physical pooling and incorrect normalized pooling.
    a_norm = [c.box(F(1, 10), F(1, 10)), c.box(F(9, 10), F(9, 10))]
    b_norm = [c.box(F(1, 5), F(1, 5)), c.box(F(4, 5), F(4, 5))]
    weights = [1, 3]
    spans = [F(10), F(1)]
    physical = []
    normalized = []
    for values in (a_norm, b_norm):
        physical.append(c.pooled([{"n": n, "risk": c.scale_box(r, s * s)} for r, n, s in zip(values, weights, spans)]))
        normalized.append(c.pooled([{"n": n, "risk": r} for r, n in zip(values, weights)]))
    require(F(physical[0]["upper"]) == F(127, 40) and F(physical[1]["upper"]) == F(28, 5), "Physical pooled SSE/count is incorrect")
    require(F(normalized[0]["upper"]) > F(normalized[1]["upper"]), "Synthetic normalization counterexample did not distinguish rankings")
    rows = [{"candidate_id": name, "recipe": recipe, "family": "kahm",
             "ridge_lambda": "0" if recipe == "qp_regularized" else None,
             "extension_weight": "1/2" if recipe == "qp_regularized" else None,
             "pooled_physical_risk": risk, "last_fold_physical_risk": last_risk,
             "selection_payload_bytes": 100}
            for name, recipe, risk, last_risk in (("affine_winner", "affine", physical[0], physical[1]),
                                                 ("qp_midpoint", "qp_regularized", physical[1], physical[0]))]
    require(c.select(rows, "joint_rolling")["candidate_id"] == "affine_winner", "Physical selector missed the affine winner")
    require(c.select(rows, "qp_rolling")["candidate_id"] == "qp_midpoint", "QP-only selector pool changed")
    require(c.select(rows, "affine_rolling")["candidate_id"] == "affine_winner", "Affine-only selector pool changed")
    require(c.select(rows, "joint_last_fold")["candidate_id"] == "qp_midpoint", "Last-fold selector ignored its declared score")
    for reverse in (False, True):
        union_rows = deepcopy(rows)
        if reverse:
            union_rows[0]["pooled_physical_risk"], union_rows[1]["pooled_physical_risk"] = (
                union_rows[1]["pooled_physical_risk"], union_rows[0]["pooled_physical_risk"])
        joint_id = c.select(union_rows, "joint_rolling")["candidate_id"]
        joint_row = next(row for row in union_rows if row["candidate_id"] == joint_id)
        family_selector = "affine_rolling" if joint_row["recipe"] == "affine" else "qp_rolling"
        require(joint_id == c.select(union_rows, family_selector)["candidate_id"],
                "Union selection must equal the winner of its chosen head family")
    size_rows = deepcopy(rows)
    for row in size_rows:
        row["pooled_physical_risk"] = c.box(1, 1)
    size_rows[0]["selection_payload_bytes"] = 101
    require(c.select(size_rows, "joint_rolling")["candidate_id"] == "qp_midpoint", "Payload-size tie breaker failed")
    size_rows[0]["selection_payload_bytes"] = 100
    require(c.select(size_rows, "joint_rolling")["candidate_id"] == "affine_winner", "Lexical candidate tie breaker failed")
    # An algebraic score fixture exercises the actual runner's strict
    # stability decision at exactly zero margin. All risks equal one, as for
    # the elementary prediction-zero/target-one squared-loss construction.
    # This is a selector unit fixture, not another fitted forecast experiment.
    tie_task = "artificial_selection_tie40"
    tie_scores = {}
    for stage, n in VALIDATION_COUNTS.items():
        tie_rows = []
        for cfg in protocol["heads"]["configurations"] + protocol["baselines"]["configurations"]:
            baseline = "kind" in cfg
            row = {**cfg, "recipe": "baseline" if baseline else cfg["recipe"],
                   "family": "ordinary" if baseline else cfg["map"].split("_")[0],
                   "baseline_kind": cfg.get("kind"), "deployment_payload_bytes": 100,
                   "physical_validation_risk": c.box(1, 1),
                   "physical_optimizer_risk": c.box(1, 1) if cfg.get("recipe") == "qp_regularized" else None}
            tie_rows.append(row)
        record = {"schema": "step40-validation-scores-v1", "task": tie_task, "stage": stage,
                  "n": n, "candidate_rows": tie_rows, "artificial_algebraic_scores_only": True}
        base = root / f"output/{tie_task}/{stage}"
        ref = c.save(base / "validation_scores40.json", record)
        c.save(base / "completion_seal40.json", {"schema": "step40-artificial-algebraic-completion-v1",
               "artificial_only": True, "bound_files": {ref["path"]: ref["sha256"]}})
        gate(root / f"audit/{tie_task}/{stage}/validation_gate40.json", task=tie_task, stage=stage,
             completion_seal_sha256=c.sha(base / "completion_seal40.json"),
             known_loss_construction="prediction0_target1_all_candidates")
        tie_scores[stage] = record
    run.select(protocol, tie_task)
    tie_selection = c.read(root / f"output/{tie_task}/selection40.json")
    audit.audit_selection_values(tie_scores, tie_selection)
    decision = tie_selection["mixed_decision_certificate"]
    require(decision["selected_winner"]["strict_margin_lower"] == "0" and
            decision["selected_winner"]["status"] == "UNRESOLVED" and
            decision["candidate_exclusion"]["survivor_count"] == 36 and
            decision["head_family_resolution"]["status"] == "NO_STRICT_SEPARATION",
            "A zero-margin mixed comparison was labelled uniquely resolved")
    corrupted = deepcopy(tie_selection)
    corrupted["mixed_decision_certificate"]["selected_winner"]["status"] = "CERTIFIED_UNIQUE_MIXED_COMPARATOR_WINNER"
    try:
        audit.audit_selection_values(tie_scores, corrupted)
    except AssertionError:
        false_uniqueness_rejected = True
    else:
        false_uniqueness_rejected = False
    require(false_uniqueness_rejected, "Independent selection verifier accepted false uniqueness")
    def bound_row(name, lo, hi=None):
        return {"candidate_id": name, "head_id": "artificial-head:" + name,
                "head_sha256": c.digest({"head": name}),
                "test_predictions_digest": c.digest({"predictions": name}),
                "physical_test_risk": c.box(lo, lo if hi is None else hi)}
    new = bound_row("new", F(19, 20))
    reference, ordinary, affine = [bound_row(name, 1) for name in ("qp", "ordinary", "affine")]
    boundary = c.practical_gate(new, reference, ordinary, affine)
    audit.audit_practical_gate(new, reference, ordinary, affine, boundary)
    require(boundary["numerical_criterion_pass"], "Exact five-percent boundary failed")
    zero = bound_row("zero", 0)
    zero_gate = c.practical_gate(zero, zero, zero, zero)
    audit.audit_practical_gate(zero, zero, zero, zero, zero_gate)
    require(not zero_gate["numerical_criterion_pass"], "Zero reference risk became a relative-gain success")
    stronger_affine = bound_row("stronger_affine", F(94, 100))
    guarded = c.practical_gate(new, reference, ordinary, stronger_affine)
    audit.audit_practical_gate(new, reference, ordinary, stronger_affine, guarded)
    require(guarded["at_least_five_percent_mse_improvement"] and guarded["no_worse_than_selected_ordinary"] and
            not guarded["no_worse_than_selected_affine"] and not guarded["numerical_criterion_pass"],
            "The new affine guard did not affect the primary criterion")
    same = bound_row("same_affine", F(9, 10), F(19, 20))
    identity_gate = c.practical_gate(same, reference, ordinary, deepcopy(same))
    audit.audit_practical_gate(same, reference, ordinary, deepcopy(same), identity_gate)
    require(identity_gate["numerical_criterion_pass"] and identity_gate["joint_same_as_affine"] and
            identity_gate["affine_comparison_reason"] == "SAME_FINAL_CANDIDATE_PAYLOAD_EXACT_EQUALITY" and
            identity_gate["joint_superiority_over_affine_from_identity"] is False,
            "Identical final payloads with a nonzero-width enclosure did not certify equality")
    identity_mutations = []
    for field in ("candidate_id", "head_id", "head_sha256", "test_predictions_digest"):
        altered = deepcopy(same)
        altered[field] += "_different"
        rejected_gate = c.practical_gate(same, reference, ordinary, altered)
        audit.audit_practical_gate(same, reference, ordinary, altered, rejected_gate)
        require(not rejected_gate["joint_same_as_affine"] and not rejected_gate["no_worse_than_selected_affine"],
                "Changed final identity field falsely proved equality: " + field)
        identity_mutations.append(field)
    return {"physical_risks": physical, "incorrect_normalized_risks": normalized,
            "selected_affine_in_joint_pool": c.select(rows, "joint_rolling"),
            "last_fold_control_uses_last_fold": True, "size_then_lexical_ties_checked": True,
            "joint_union_winner_equals_chosen_family_winner": True,
            "zero_margin_is_unresolved": True, "false_uniqueness_rejected": false_uniqueness_rejected,
            "algebraic_tie_fixture": f"output/{tie_task}/selection40.json", "positive_denominator_required": True,
            "affine_no_worse_guard_required": True,
            "same_payload_with_nonzero_width_proves_equality": True,
            "same_payload_does_not_prove_joint_superiority": True,
            "false_identity_mutations_rejected": identity_mutations}


def main():
    if (SOURCE_ROOT / "protocol_seal40.json").exists():
        raise RuntimeError("Cannot update artificial fixtures after scientific protocol freeze")
    started = time.perf_counter()
    (SOURCE_ROOT / "tmp").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="artificial_runner40_", dir=SOURCE_ROOT / "tmp"))
    print(json.dumps({"artificial_root": str(root), "scientific_data_accessed": False}), flush=True)
    hashes = source_hashes()
    auditor_hash = sha(SOURCE_ROOT / "source/audit40.py")
    prior_source = SOURCE_ROOT / "inputs/prior/protocol39.json"
    prior_destination = root / "inputs/prior/protocol39.json"
    prior_destination.parent.mkdir(parents=True, exist_ok=True)
    prior_destination.write_bytes(prior_source.read_bytes())
    c.ROOT = root
    run.ROOT = root
    run.OUT = root / "output"
    audit.ROOT = root
    protocol = protocol_fixture()
    c.save(root / "protocol40.json", protocol)
    c.save(root / "protocol_seal40.json", {"schema": "step40-artificial-protocol-seal-v1", "artificial_only": True,
           "protocol_sha256": c.sha(root / "protocol40.json"), "bound_files": {"protocol40.json": c.sha(root / "protocol40.json")}})
    fake = SyntheticData(root, protocol)
    run.d = fake
    gate(root / "audit/preexecution_gate40.json", synthetic_source_formula="raw_value(stamp)")
    stages = {}
    scores = {}
    for stage in ("f1", "f2", "f3"):
        run.fit(protocol, TASK, stage)
        training = fake.cohorts[(stage, "train")]
        context = audit.load_context(TASK, stage)
        checked_fit = audit.audit_stage_models(protocol, TASK, stage, training, context)
        fitseal = root / f"output/{TASK}/{stage}/fit_seal40.json"
        verify_seal(fitseal)
        gate(root / f"audit/{TASK}/{stage}/fit_gate40.json", task=TASK, stage=stage,
             fit_seal_sha256=c.sha(fitseal), model_audit=checked_fit)
        run.validate(protocol, TASK, stage)
        cohort = fake.cohorts[(stage, "validation")]
        base = root / f"output/{TASK}/{stage}"
        bundle = c.read(base / "validation_predictions40.json")
        checked_pred = audit.audit_prediction_bundle(protocol, TASK, stage, bundle, context,
                                                     cohort["X"], cohort["ids"], cohort["persistence"])
        scores[stage] = c.read(base / "validation_scores40.json")
        checked_loss = audit.audit_validation_values(protocol, TASK, stage, scores[stage], bundle,
                                                     context["manifest"], cohort)
        verify_seal(base / "completion_seal40.json")
        gate(root / f"audit/{TASK}/{stage}/validation_gate40.json", task=TASK, stage=stage,
             completion_seal_sha256=c.sha(base / "completion_seal40.json"),
             prediction_audit=checked_pred, loss_audit=checked_loss)
        stages[stage] = {"training_n": len(training["y"]), "validation_n": len(cohort["y"]),
                         "maps": checked_fit["maps"], "heads": checked_fit["heads"],
                         "regularized_fits": len(checked_fit["regularized_fits"]),
                         "predictions": checked_pred["head_predictions"] + checked_pred["baseline_predictions"]}
        print(json.dumps({"artificial_stage": stage, "status": "PASS"}), flush=True)
    run.select(protocol, TASK)
    selection = c.read(root / f"output/{TASK}/selection40.json")
    checked_selection = audit.audit_selection_values(scores, selection)
    joint_id = selection["selections"]["joint_rolling"]["candidate_id"]
    joint_row = next(row for row in selection["candidate_rows"] if row["candidate_id"] == joint_id)
    family_selector = "affine_rolling" if joint_row["recipe"] == "affine" else "qp_rolling"
    require(joint_id == selection["selections"][family_selector]["candidate_id"],
            "Full-panel union selection differs from its chosen family's selection")
    require(selection["fold_score_sha256"] == {s: c.sha(root / f"output/{TASK}/{s}/validation_scores40.json") for s in scores},
            "Selection source score hashes differ")
    gate(root / f"audit/{TASK}/selection_gate40.json", task=TASK,
         selection_seal_sha256=c.sha(root / f"output/{TASK}/selection_seal40.json"), selection_audit=checked_selection)
    run.fit(protocol, TASK, "final")
    final_training = fake.cohorts[("final", "train")]
    context = audit.load_context(TASK, "final")
    checked_final = audit.audit_stage_models(protocol, TASK, "final", final_training, context)
    fitseal = root / f"output/{TASK}/final/fit_seal40.json"
    verify_seal(fitseal)
    gate(root / f"audit/{TASK}/final/fit_gate40.json", task=TASK, stage="final", fit_seal_sha256=c.sha(fitseal), model_audit=checked_final)
    gate(root / f"audit/{TASK}/pretest_gate40.json", task=TASK, fit_seal_sha256=c.sha(fitseal),
         selection_seal_sha256=c.sha(root / f"output/{TASK}/selection_seal40.json"))
    selected_before = {name: value["candidate_id"] for name, value in selection["selections"].items()}
    run.test(protocol, TASK)
    queries = [query_for(i, test=True) for i in range(2)]
    all_predictions = {cid: [] for cid in protocol["evaluation"]["prediction_ids"]}
    all_optimizer = {row["candidate_id"]: [] for row in context["manifest"]["candidate_rows"] if row["recipe"] == "qp_regularized"}
    test_audits = []
    for index, query in enumerate(queries):
        bundle = c.read(root / f"output/{TASK}/test/predictions/q{index:03d}.json")
        history = fake.histories[query["id"]]
        test_audits.append(audit.audit_prediction_bundle(protocol, TASK, "final", bundle, context,
                                                         history["X"], [query["id"]], [history["persistence"]]))
        for cid in all_predictions:
            all_predictions[cid].append(bundle["predictions"][cid]["predictions"][0])
        for cid in all_optimizer:
            all_optimizer[cid].append(bundle["optimizer_predictions"][cid]["predictions"][0])
    result_path = root / f"output/{TASK}/test/result40.json"
    result = c.read(result_path)
    primary = result["primary"]
    y = [F(row["y"]) for row in fake.revealed]
    span_squared = F(final_training["normalization"]["span"]) ** 2
    require([row["candidate_id"] for row in primary["candidate_rows"]] == protocol["evaluation"]["prediction_ids"], "Test candidate roster/order differs")
    for row in primary["candidate_rows"]:
        cid = row["candidate_id"]
        audit.a38.audit_loss(all_predictions[cid], y, row["test_risk"])
        require(row["physical_test_risk"] == audit.scaled_box(row["test_risk"], span_squared), "Test physical loss conversion differs")
        require(row["test_predictions_digest"] == audit.digest(all_predictions[cid]), "Test prediction digest differs")
        if cid in all_optimizer:
            optimizer_risk = audit.scaled_box(row["physical_optimizer_risk"], F(1) / span_squared)
            require(optimizer_risk == audit.loss_box(all_optimizer[cid], y), "Test optimizer risk differs from independent interval loss")
            require(row["optimizer_predictions_digest"] == audit.digest(all_optimizer[cid]), "Test optimizer prediction digest differs")
        else:
            require(row["physical_optimizer_risk"] is None, "Unsupported optimizer loss was attached")
        require(row["monthly"]["2017-09"]["risk"] == row["test_risk"] and
                row["monthly"]["2017-09"]["physical_risk"] == row["physical_test_risk"], "Monthly synthetic test partition differs")
    require({name: value["candidate_id"] for name, value in primary["selected"].items()} == selected_before,
            "Test outcomes changed a frozen selected configuration")
    selected = primary["selected"]
    audit.audit_practical_gate(selected["joint_rolling"], selected["qp_rolling"],
                               selected["ordinary_rolling"], selected["affine_rolling"], primary["primary_gate"])
    require(primary["development_decision_certificate"] == selection["mixed_decision_certificate"],
            "Test report altered the frozen development certificate")
    audit.audit_paired_heads([row for row in primary["candidate_rows"] if row["recipe"] != "baseline"],
                            primary["test_same_map_comparisons"], names=protocol["maps"],
                            operational_key="physical_test_risk", optimizer_key="physical_optimizer_risk",
                            cohort_role="test")
    require(len(fake.revealed) == 2 and fake.pending is None and fake.official_measurement_accesses == 0,
            "Synthetic test did not consume exactly two postprediction labels")
    gate(root / f"audit/{TASK}/postrun_audit40.json", task=TASK, result_sha256=c.sha(result_path),
         verified_test_predictions=test_audits, verified_candidate_losses=45, verified_optimizer_losses=18)
    run.aggregate(protocol)
    aggregate = c.read(root / "output/result40.json")
    require(aggregate["practical_transfer_success"] is primary["primary_gate"]["numerical_criterion_pass"], "Synthetic aggregate gate differs")
    try:
        run.fit(protocol, TASK, "f1")
    except FileExistsError:
        replay_rejected = True
    else:
        replay_rejected = False
    require(replay_rejected, "Write-once runner allowed an artificial fitting-stage replay")
    for index, event in enumerate(fake.events):
        unsigned = {k: v for k, v in event.items() if k != "event_sha256"}
        require(event["event_sha256"] == c.digest(unsigned) and event["previous_event_sha256"] ==
                (fake.events[index - 1]["event_sha256"] if index else None), "Synthetic event chain differs")
    for query in queries:
        written = next(i for i, e in enumerate(fake.events) if e["event"] == "test_prediction_receipt_written" and e["query_id"] == query["id"])
        revealed = next(i for i, e in enumerate(fake.events) if e["event"] == "artificial_target_revealed" and e["query_id"] == query["id"])
        require(written < revealed, "A test label preceded its complete forecast receipt")
    edges = score_and_selection_edge_cases(protocol, root)
    require(hashes == source_hashes(), "Producer source changed during artificial integration")
    require(auditor_hash == sha(SOURCE_ROOT / "source/audit40.py"),
            "Independent auditor source changed during artificial integration")
    artifacts = [{"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(root.rglob("*")) if path.is_file()]
    fixture_manifest = {"schema": "step40-artificial-runner-fixture-manifest-v1", "scientific_data_accessed": False,
                        "artificial_root": str(root), "files": artifacts}
    manifest_path = root / "fixture_manifest40.json"
    manifest_path.write_bytes(c.canonical_bytes(fixture_manifest))
    receipt = {"schema": "step40-artificial-runner-check-v1", "status": "PASS", "artificial_only": True,
               "scientific_data_accessed": False, "official_measurement_accesses": 0,
               "source_sha256": hashes, "independent_audit_source_sha256": auditor_hash,
               "baseline_recipe_source_sha256": sha(prior_destination),
               "fixture_root": str(root.relative_to(SOURCE_ROOT)), "fixture_manifest_sha256": sha(manifest_path),
               "fixture_files": len(artifacts), "stages": stages,
               "final_training_n": len(final_training["y"]), "test_n": 2,
               "total_qp_training_fits": 18 * 4, "total_affine_training_fits": 18 * 4,
               "total_certified_heads": 36 * 4,
               "validation_prediction_count": 45 * sum(VALIDATION_COUNTS.values()), "test_prediction_count": 45 * 2,
               "all_nine_selections_independently_checked": True,
               "all_mixed_decision_certificates_independently_checked": True,
               "all_same_map_test_contrasts_independently_checked": True,
               "all_exact_optimizer_enclosures_independently_checked": True,
               "all_physical_losses_independently_checked": True,
               "source_fold_and_final_refit_bindings_checked": True,
               "prediction_before_label_order_checked": True, "write_once_replay_rejected": replay_rejected,
               "edge_cases": edges, "wall_seconds": time.perf_counter() - started}
    (SOURCE_ROOT / "tests/runner_preflight40.json").write_bytes(c.canonical_bytes(receipt))
    print(json.dumps({"status": "PASS", "scientific_data_accessed": False, "fixture_root": receipt["fixture_root"],
                      "fixture_files": len(artifacts), "seconds": receipt["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
