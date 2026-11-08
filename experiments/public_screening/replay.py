#!/usr/bin/env python3
"""Recompute the fixed Beijing and Bike screening evidence without refitting."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from fractions import Fraction as F
import gzip
import hashlib
import importlib.util
import io
from itertools import combinations
import json
from math import isqrt
from pathlib import Path
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parent
ASSURANCE = "CONDITIONAL_ON_STORED_EXACT_FEATURE_COORDINATES"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def packed(name):
    return load(ROOT / "witnesses" / (name + ".json.gz"))


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def box(lo, hi):
    return {"lower": str(lo), "upper": str(hi), "width": str(hi - lo)}


def clip(value, low, high):
    return min(high, max(low, value))


def import_auditor(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The archived auditors are independent of the scientific certificate producers.
sys.path.insert(0, str(ROOT / "original/s36/source"))
audit36 = import_auditor("public_screening_audit36", ROOT / "original/s36/source/audit36.py")
audit35 = audit36.old


class DyadicMetric:
    """Exact squared distances using a common integer denominator."""

    def __init__(self, values):
        self.values = [[F(v) for v in row] for row in values]
        require(self.values and len(self.values[0]) > 0, "Empty feature matrix")
        width = len(self.values[0])
        require(all(len(row) == width for row in self.values), "Ragged feature matrix")
        denominators = [v.denominator for row in self.values for v in row]
        require(all(d & (d - 1) == 0 for d in denominators), "Features are not dyadic")
        denominator = max(denominators)
        self.square_denominator = denominator * denominator
        self.rows = [[v.numerator * (denominator // v.denominator) for v in row]
                     for row in self.values]

    def squared(self, i, j):
        return F(sum((a - b) ** 2 for a, b in zip(self.rows[i], self.rows[j])),
                 self.square_denominator)


def sqrt_bounds(value, bits):
    require(value >= 0 and 1 <= bits <= 512, "Invalid distance precision")
    denominator = 1 << bits
    integer = isqrt((value.numerator << (2 * bits)) // value.denominator)
    lower = F(integer, denominator)
    upper = lower if lower * lower == value else F(integer + 1, denominator)
    require(lower * lower <= value <= upper * upper, "Distance enclosure failed")
    return lower, upper, integer


def rebuild_distances(record):
    values, ids = record["features_exact"], record["query_ids"]
    metric = DyadicMetric(values)
    bits = record["distance_table_replay"]["bits"]
    pairs = []
    for i, j in combinations(range(len(values)), 2):
        squared = metric.squared(i, j)
        lo, hi, integer = sqrt_bounds(squared, bits)
        pairs.append({"indices": [i, j], "query_ids": [ids[i], ids[j]],
                      "distance": box(lo, hi), "squared_exact": str(squared),
                      "scaled_sqrt_floor": integer, "bits": bits})
    table = {"schema": "step35-stored-feature-distances-v1", "assurance": ASSURANCE,
             "n": len(values), "dimension": len(values[0]), "query_ids": ids,
             "features_exact": values, "feature_digest": audit35.digest(values),
             "input_digest": audit35.digest({"ids": ids, "features": values}),
             "bits": bits, "pairs": pairs, "pair_count": len(pairs),
             "distance_evaluations": len(pairs), "native_transform_error_enclosed": False}
    require(sha_bytes(encoded(table).rstrip(b"\n")) ==
            record["distance_table_replay"]["canonical_sha256"],
            "Reconstructed distance-table digest differs")
    restored = {key: value for key, value in record.items() if key != "distance_table_replay"}
    restored["distance_table"] = table
    return restored


def rebuild_prediction(head, cohort, bits):
    hid, kappa, low, high = audit35.audit_head(head)
    features, targets, ids = (cohort[k] for k in ("features_exact", "targets_exact", "query_ids"))
    y = [F(v) for v in targets]
    n = len(features)
    require(n == len(y) == len(ids) and len(set(ids)) == n, "Prediction roster differs")
    require(all(len(vector) == head["dimension"] for vector in features),
            "Prediction feature dimension differs from head")
    rows = []
    if head["kind"] == "clipped_affine":
        beta, intercept = [F(v) for v in head["coefficients"]], F(head["intercept"])
        for i, vector in enumerate(features):
            raw = intercept + sum((b * F(v) for b, v in zip(beta, vector)), F(0))
            value = clip(raw, low, high)
            rows.append({"query_index": i, "query_id": ids[i],
                         "raw_prediction": box(raw, raw), "prediction": box(value, value)})
        distance_count = 0
        status = "VERIFIED_EXACT_STORED_FEATURE_AFFINE_PREDICTIONS"
        require(bits is None, "Unexpected affine distance precision")
    else:
        anchors = head["anchor_features"]
        values = [F(v) for v in head["anchor_values"]]
        metric = DyadicMetric(features + anchors)
        require(bits == 80, "Extension distance precision differs")
        for i in range(n):
            distances, lower_cones, upper_cones = [], [], []
            for j, value in enumerate(values):
                squared = metric.squared(i, n + j)
                dlo, dhi, integer = sqrt_bounds(squared, bits)
                distances.append({"anchor_index": j, "anchor_id": head["anchor_ids"][j],
                                  "distance": box(dlo, dhi), "squared_exact": str(squared),
                                  "scaled_sqrt_floor": integer})
                lower_cones.append((value - kappa * dhi, value - kappa * dlo))
                upper_cones.append((value + kappa * dlo, value + kappa * dhi))
            lower = tuple(max(c[k] for c in lower_cones) for k in (0, 1))
            upper = tuple(min(c[k] for c in upper_cones) for k in (0, 1))
            raw = tuple((lower[k] + upper[k]) / 2 for k in (0, 1))
            prediction = tuple(clip(v, low, high) for v in raw)
            rows.append({"query_index": i, "query_id": ids[i], "anchor_distances": distances,
                         "lower_envelope": box(*lower), "upper_envelope": box(*upper),
                         "raw_prediction": box(*raw), "prediction": box(*prediction)})
        distance_count = n * len(anchors)
        status = "VERIFIED_STORED_FEATURE_EXTENSION_ENCLOSURES"
    losses = []
    for row, target in zip(rows, y):
        lo, hi = F(row["prediction"]["lower"]) - target, F(row["prediction"]["upper"]) - target
        lower = F(0) if lo <= 0 <= hi else min(lo * lo, hi * hi)
        upper = max(lo * lo, hi * hi)
        loss = box(lower, upper)
        losses.append(loss)
        row["squared_loss"] = loss
    loss = {**box(sum((F(v["lower"]) for v in losses), F(0)) / n,
                  sum((F(v["upper"]) for v in losses), F(0)) / n),
            "n": n, "per_row": losses, "targets_clipped": False}
    return {"schema": "step35-stored-feature-prediction-receipt-v1", "status": status,
            "assurance": ASSURANCE, "feature_id": head["feature_id"], "class_id": head["class_id"],
            "head_id": hid, "head": head, "query_ids": ids, "n": n, "dimension": len(features[0]),
            "features_exact": features, "feature_digest": audit35.digest(features),
            "query_roster_digest": audit35.digest({"ids": ids, "features": features}),
            "targets_exact": targets, "target_digest": audit35.digest(targets),
            "predictions": [r["prediction"] for r in rows], "rows": rows, "loss": loss,
            "distance_evaluations": distance_count, "distance_bits": bits,
            "targets_outside_predictor_range": sum(not low <= t <= high for t in y),
            "targets_clipped": False, "native_transform_error_enclosed": False,
            "native_pipeline_roundoff_certified": False, "population_risk_claim": False,
            "future_response_coverage_claim": False, "raw_input_lipschitz_claim": False}


def verify_cohorts(study, protocol):
    """Bind each retained causal cohort and its threshold to public source rows."""
    root = ROOT / "original" / study
    number = study[1:]
    source = load(root / "inputs" / ("public_source" + number + ".json"))
    archive = root / protocol["data"]["archive"]
    require(sha_bytes(archive.read_bytes()) == source["archive_sha256"], "Public archive digest differs")
    tasks = ["pm25"] if study == "s35" else protocol["tasks"]
    observed = {}
    with zipfile.ZipFile(archive) as zipped:
        member = "PRSA_data_2010.1.1-2014.12.31.csv" if study == "s35" else "hour.csv"
        with zipped.open(member) as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")):
                if study == "s35":
                    stamp = datetime(*(int(row[k]) for k in ("year", "month", "day", "hour")))
                    values = [None if row["pm2.5"] == "NA" else int(row["pm2.5"])]
                else:
                    stamp = datetime.fromisoformat(row["dteday"]) + timedelta(hours=int(row["hr"]))
                    values = [int(row[t]) for t in tasks]
                    require(values[0] + values[1] == values[2], "Bike target relationship differs")
                require(stamp not in observed, "Duplicate source timestamp")
                observed[stamp] = values
    cohorts = {task: {} for task in tasks}
    for role in ("train", "decision", "test"):
        if study == "s35":
            years = protocol["data"][role + "_target_years"]
            first, stop = datetime(min(years), 1, 1), datetime(max(years) + 1, 1, 1)
        else:
            first = datetime.fromisoformat(protocol["data"][role + "_start"])
            stop = datetime.fromisoformat(protocol["data"][role + "_end"]) + timedelta(days=1)
        selected, histories, origins = [], [], []
        exclusions = {"required_timestamp_absent": 0, "missing_lag_or_target": 0}
        for i in range((stop - first).days):
            target = first + timedelta(days=i, hours=23)
            origin = target - timedelta(hours=24)
            history = [origin - timedelta(hours=23 - j) for j in range(24)]
            required = history + [target]
            if any(s not in observed for s in required):
                exclusions["required_timestamp_absent"] += 1
                continue
            if any(any(v is None for v in observed[s]) for s in required):
                exclusions["missing_lag_or_target"] += 1
                continue
            selected.append(target)
            histories.append(history)
            origins.append(origin)
        for column, task in enumerate(tasks):
            folder = root / "cohorts" if study == "s35" else root / "cohorts" / task
            saved = load(folder / (role + ".json"))
            require(saved["ids"] == [role + ":" + s.isoformat(timespec="hours") for s in selected]
                    and saved["target_timestamps"] == [s.isoformat(timespec="hours") for s in selected]
                    and saved["origin_timestamps"] == [s.isoformat(timespec="hours") for s in origins]
                    and saved["raw_targets"] == [observed[s][column] for s in selected],
                    "Causal cohort source reconstruction differs")
            matrix_path = folder / (role + ".npy")
            lag_values = np.load(matrix_path, allow_pickle=False)
            expected = np.asarray([[observed[s][column] for s in history] for history in histories], dtype=np.float64)
            require(sha_bytes(matrix_path.read_bytes()) == saved["lag_matrix_sha256"]
                    and lag_values.dtype == np.float64 and np.array_equal(lag_values, expected),
                    "Causal lag matrix differs")
            require(saved["roster"]["excluded"] == exclusions and saved["roster"]["eligible"] == len(selected)
                    and saved["roster"]["candidate_daily_targets"] == (stop - first).days,
                    "Complete-case roster accounting differs")
            cohorts[task][role] = saved
    epsilons, scales = {}, {}
    for task in tasks:
        train = cohorts[task]["train"]
        low, high = min(train["raw_targets"]), max(train["raw_targets"])
        span = high - low
        require(span > 0, "Degenerate normalization")
        for role, saved in cohorts[task].items():
            folder = root / "cohorts" if study == "s35" else root / "cohorts" / task
            lag_values = np.load(folder / (role + ".npy"), allow_pickle=False)
            require(saved["y"] == [str(F(v - low, span)) for v in saved["raw_targets"]]
                    and saved["persistence"] == [str(F(int(v) - low, span)) for v in lag_values[:, -1]],
                    "Training-only normalization or persistence differs")
            require(saved["roster"]["target_outside_prediction_range_count"] ==
                    sum(not 0 <= F(v) <= 1 for v in saved["y"]), "Target range accounting differs")
        persistence_mse = sum(((F(a) - F(b)) ** 2 for a, b in zip(train["persistence"], train["y"])), F(0)) / len(train["y"])
        epsilons[task] = F(19, 20) * persistence_mse
        scales[task] = {"train_min": low, "train_max": high, "range": span,
                        "train_persistence_mse": persistence_mse}
    return cohorts, epsilons, scales


def replay_heads(study, cohort_records, source_digests):
    data = packed(study + "_heads")
    rows = {(row.get("task", "pm25"), row["candidate_id"]): row for row in data["candidate_rows"]}
    predictions, count, distances = {}, 0, 0
    for key, record in data["heads"].items():
        task, name = record["task"], record["map"]
        row = rows[task, record["candidate_id"]]
        head = record["head"]
        hid, _, _, _ = audit35.audit_head(head)
        require(hid == row["head_id"] and head["feature_id"] == row["feature_id"]
                and head["class_id"] == row["class_id"], "Frozen head identity differs")
        require(len(encoded(head)) == row["head_payload_bytes"]
                and row["deployment_payload_bytes"] == row["map_payload_bytes"] + row["head_payload_bytes"],
                "Serialized head size or complete payload sum differs")
        if head["kind"] == "symmetric_lipschitz_extension":
            train_ids = cohort_records[task]["train"]["ids"]
            anchors = [((2 * j + 1) * len(train_ids)) // 64 for j in range(32)]
            require(head["anchor_ids"] == [train_ids[i] for i in anchors], "Training-only anchor roster differs")
        for role in ("decision", "test"):
            cohort = data["cohorts"][task + "::" + name][role]
            reference = record["expected"][role]
            require(cohort["query_ids"] == cohort_records[task][role]["ids"]
                    and cohort["targets_exact"] == cohort_records[task][role]["y"],
                    "Head predictions use another target cohort")
            require(cohort["feature_id"] == head["feature_id"]
                    and cohort["feature_digest"] == audit35.digest(cohort["features_exact"])
                    and cohort["target_digest"] == audit35.digest(cohort["targets_exact"]),
                    "Stored feature/target identity differs")
            prediction = rebuild_prediction(head, cohort, reference["distance_bits"])
            loss = {k: v for k, v in prediction["loss"].items() if k != "per_row"}
            require(prediction["predictions"] == reference["predictions"] and loss == reference["loss"]
                    and prediction["head_id"] == reference["head_id"],
                    "Exact prediction or squared-loss enclosure differs")
            audit35.interval(row[role + "_risk"], audit35.interval(loss))
            original_path = row[role + "_prediction_path"]
            require(sha_bytes(encoded(prediction)) == source_digests[study, original_path],
                    "Reconstructed complete prediction receipt differs from source bytes")
            digest = audit35.digest(prediction)
            predictions[digest] = {"head_id": hid, "class_id": head["class_id"],
                                   "loss": loss, **cohort}
            count += 1
            distances += prediction["distance_evaluations"]
        if len(predictions) % 24 == 0:
            print(study + ": verified " + str(len(predictions)) + " prediction receipts", flush=True)
    return data, predictions, {"heads": len(rows), "prediction_and_loss_receipts": count,
                               "exact_query_anchor_distances": distances}


def replay_beijing(data, predictions, epsilon, source_digests):
    witnesses = packed("s35_class_witnesses")
    policies = packed("s35_policy")
    result = packed("s35_reference_result")
    require(data["candidate_rows"] == result["candidate_rows"], "Beijing result panel differs")
    blocks, pair_count = {}, 0
    for path, compact in witnesses["blocks"].items():
        block = rebuild_distances(compact)
        require(sha_bytes(encoded(block)) == source_digests["s35", path], "Restored block differs from source bytes")
        check = audit35.audit_qp(block)
        require(block["kind"] == "class" and block["work"]["max_updates"] == 400, "Block role or work budget differs")
        blocks[path] = {"certificate_digest": audit35.digest(block), "record": compact, "check": check}
        pair_count += block["distance_table"]["pair_count"]
    lower_by_role, aggregates = {"decision": {}, "test": {}}, {}
    for path, lower in witnesses["lowers"].items():
        role, name = lower["role"], lower["map"]
        cohort = data["cohorts"]["pm25::" + name][role]
        ids, values, targets, total = [], [], [], F(0)
        for i, block_path in enumerate(lower["blocks"]):
            entry = blocks[block_path]
            block = entry["record"]
            start, stop = i * 32, min((i + 1) * 32, len(cohort["query_ids"]))
            require(block["query_ids"] == cohort["query_ids"][start:stop]
                    and block["features_exact"] == cohort["features_exact"][start:stop]
                    and block["targets_exact"] == cohort["targets_exact"][start:stop]
                    and block["feature_id"] == cohort["feature_id"], "Block partition differs")
            ids.extend(block["query_ids"])
            values.extend(block["features_exact"])
            targets.extend(block["targets_exact"])
            total += entry["check"]["lower_sse"]
        n = len(ids)
        require(n == len(set(ids)) == lower["n"] == len(cohort["query_ids"]), "Block union is incomplete")
        floor = sum((max(F(0), -F(v), F(v) - 1) ** 2 for v in targets), F(0)) / n
        total /= n
        require(F(lower["dual_lower"]) == total and F(lower["range_only_floor"]) == floor
                and F(lower["lower"]) == max(total, floor), "Whole-cohort lower bound differs")
        lower_by_role[role][name] = max(total, floor)
        aggregate_path = path.rsplit("/", 1)[0] + "/aggregate.json"
        aggregate = witnesses["aggregates"][aggregate_path]
        require(aggregate["query_ids"] == ids and aggregate["n"] == n
                and aggregate["feature_digest"] == audit35.digest(values)
                and aggregate["target_digest"] == audit35.digest(targets)
                and aggregate["block_primals_averaged_for_union_upper"] is False,
                "Aggregate cohort identity differs")
        require(len(aggregate["blocks"]) == len(lower["blocks"]), "Aggregate block count differs")
        for block_ref, block_path in zip(aggregate["blocks"], lower["blocks"]):
            entry = blocks[block_path]
            block = entry["record"]
            require(block_ref["certificate_digest"] == entry["certificate_digest"]
                    and block_ref["n"] == block["n"] and block_ref["query_ids"] == block["query_ids"]
                    and F(block_ref["lower"]) == entry["check"]["lower_sse"] / block["n"],
                    "Aggregate block reference differs")
        uppers = []
        for candidate in aggregate["global_upper_candidates"]:
            require(candidate["kind"] == "single_global_frozen_head", "Unexpected upper witness")
            prediction = predictions[candidate["prediction_receipt_digest"]]
            require(prediction["head_id"] == candidate["head_id"]
                    and prediction["class_id"] == aggregate["class_id"]
                    and prediction["query_ids"] == ids and prediction["features_exact"] == values
                    and prediction["targets_exact"] == targets,
                    "Global upper witness is not common to the entire cohort")
            upper = F(prediction["loss"]["upper"])
            require(upper == F(candidate["upper"]), "Aggregate upper value differs")
            uppers.append(upper)
        audit35.interval(aggregate["class_risk"], (total, min(uppers)))
        aggregates[aggregate_path] = aggregate
    for key in ("decision_classes", "test_classes"):
        for row in result[key]:
            require(row["class_risk"] == aggregates[row["aggregate_path"]]["class_risk"],
                    "Reference class interval differs")
    policy, screen = policies["policy35"], policies["screen35"]
    require({k: v for k, v in policy.items() if k != "candidate_rows"} == result["policy"],
            "Beijing policy summary differs")
    rows = data["candidate_rows"]
    names = list(dict.fromkeys(row["map"] for row in rows))
    excluded = [name for name in names if lower_by_role["decision"][name] > epsilon]
    require(F(result["epsilon"]) == F(policy["epsilon"]) == F(screen["epsilon"]) == epsilon,
            "Training threshold differs")
    require(excluded == policy["excluded_classes"] == screen["excluded_classes"]
            and screen["excluded_count"] == len(excluded)
            and screen["lower_by_map"] == {name: str(lower_by_role["decision"][name]) for name in names},
            "Screening decisions differ")
    eligible = [row for row in rows if F(row["decision_risk"]["upper"]) <= epsilon]
    key = lambda row: (row["deployment_payload_bytes"], F(row["decision_risk"]["upper"]), row["candidate_id"])
    exhaustive = min(eligible, key=key, default=None)
    screened = min((row for row in eligible if row["map"] not in excluded), key=key, default=None)
    accuracy = min(rows, key=lambda row: (F(row["decision_risk"]["upper"]), row["deployment_payload_bytes"], row["candidate_id"]))
    identifier = lambda row: None if row is None else row["candidate_id"]
    choices = {"screened": identifier(screened), "exhaustive_same_rule": identifier(exhaustive),
               "ordinary_mse_minimizer": identifier(accuracy)}
    require(all(policy[k]["candidate_id"] == value for k, value in choices.items())
            and choices["screened"] == choices["exhaustive_same_rule"]
            and all(row["map"] not in excluded for row in eligible), "Screened selection differs")
    frozen = policies["screened_policy35"]
    require(frozen["selection"]["candidate_id"] == choices["screened"]
            and frozen["candidate_ids"] == [row["candidate_id"] for row in rows if row["map"] not in excluded],
            "Frozen candidate roster differs")
    avoided = [row for row in rows if row["map"] in excluded]
    avoided_time = sum(row["decoder_stage_seconds_including_retention"] for row in avoided)
    certificate_time = sum(witnesses["lowers"]["output/classes/decision/" + name + "/lower.json"]
                           ["stage_seconds_including_witness_retention"] for name in names)
    utility = bool(avoided) and avoided_time > certificate_time
    require(policy["avoided_decoder_fits_in_screened_policy"] == len(avoided)
            and policy["avoided_decoder_stage_seconds"] == avoided_time
            and policy["decision_class_seconds"] == certificate_time
            and policy["net_conditional_stage_seconds_saved"] == avoided_time - certificate_time
            and policy["positive_screening_utility_demonstrated"] == utility,
            "Screening cost or utility calculation differs")
    return {"class_blocks": len(blocks), "exact_block_pair_distances": pair_count,
            "whole_cohort_class_intervals": len(aggregates), "decision_classes": len(names),
            "excluded_classes": excluded, "eligible_heads": len(eligible), "choices": choices,
            "avoided_decoder_fits": len(avoided), "epsilon": str(epsilon),
            "recorded_certificate_seconds": certificate_time,
            "recorded_net_stage_seconds_saved": avoided_time - certificate_time,
            "positive_screening_utility_demonstrated": utility}


def replay_bike(data, cohorts, epsilons, scales):
    policy, result = packed("s36_policy"), packed("s36_reference_result")
    require(data["candidate_rows"] == result["candidate_rows"], "Bike result panel differs")
    searches, pairs = packed("s36_search"), packed("s36_pairs")
    rows = {task: {row["candidate_id"]: row for row in data["candidate_rows"] if row["task"] == task}
            for task in cohorts}
    features, manifests, bounds = {}, {}, {}
    for task, task_rows in rows.items():
        features[task], manifests[task], bounds[task] = {}, {}, {}
        for row in task_rows.values():
            name = row["map"]
            features[task][name] = {role: data["cohorts"][task + "::" + name][role]["features_exact"]
                                    for role in ("decision", "test")}
            manifests[task][name] = {"payload_bytes": row["map_payload_bytes"], "feature_id": row["feature_id"]}
        trace = searches["output/search/" + task + "/ordinary_r00.json"]["trace"]
        for event in trace["events"]:
            if event["kind"] in ("affine_decision", "extension_decision"):
                name = event["map"]
                recipe = "affine" if event["kind"] == "affine_decision" else "extension32"
                bounds[task].setdefault(name, {})[recipe] = event["bound"]
                head = data["heads"][task + "::" + name + "::" + recipe]["head"]
                audit36.audit_head_cost_bound(event["bound"], head)
    virtual = dict(searches)
    virtual["output/policies36.json"] = policy
    for task in cohorts:
        virtual["output/tasks/" + task + "/policy36.json"] = policy["tasks"][task]
        virtual["output/tasks/" + task + "/training_plan36.json"] = policy["tasks"][task]["training_plan"]
    for path, digest in pairs["aliases"].items():
        receipt = pairs["receipts"][digest]
        require(sha_bytes(encoded(receipt).rstrip(b"\n")) == digest, "Compressed pair alias digest differs")
        virtual[path] = receipt

    # Relocate archived input records without changing the independent audit.
    # Every reconstructed file is serialized in its original canonical format.
    root = ROOT / "original/s36"
    original_read, original_sha = audit36.read_json, audit36.sha

    def archive_read(path):
        relative = Path(path).relative_to(root).as_posix()
        return virtual[relative] if relative in virtual else original_read(path)

    def archive_sha(path):
        relative = Path(path).relative_to(root).as_posix()
        return sha_bytes(encoded(virtual[relative])) if relative in virtual else original_sha(path)

    audit36.read_json, audit36.sha = archive_read, archive_sha
    try:
        checks, totals, order = audit36.audit_searches(
            root, load(root / "protocol36.json"), result, cohorts, epsilons, scales,
            features, manifests, bounds, rows)
    finally:
        audit36.read_json, audit36.sha = original_read, original_sha
    require(set(path for _, _, _, path in order) == set(searches), "Search execution roster incomplete")
    return {"distinct_pair_certificates": len(pairs["receipts"]),
            "retained_pair_invocations": len(pairs["aliases"]), "search_executions": len(order),
            "tasks": checks, "totals": totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = load(ROOT / "witnesses/source_inventory.json")
    names = {"step35_public_forecasting": "s35", "step36_triggered_search": "s36"}
    source_digests = {(names[row["study"]], row["path"]): row["sha256"] for row in inventory}
    report = {"status": "RUNNING", "assurance": ASSURANCE,
              "scope": "Fixed-cohort source, certificate, prediction, and policy replay; no new model fits or timing measurements."}
    report["arithmetic_selfchecks"] = audit36.independent_selfcheck()
    for study in ("s35", "s36"):
        print(study + ": reconstructing source cohorts and training thresholds", flush=True)
        protocol = load(ROOT / "original" / study / ("protocol" + study[1:] + ".json"))
        cohorts, epsilons, scales = verify_cohorts(study, protocol)
        print(study + ": recomputing stored-feature predictions and exact loss bounds", flush=True)
        data, predictions, head_checks = replay_heads(study, cohorts, source_digests)
        if study == "s35":
            print(study + ": checking block primal/dual witnesses and screening decisions", flush=True)
            checks = replay_beijing(data, predictions, epsilons["pm25"], source_digests)
            name = "beijing_pm25"
        else:
            print(study + ": checking pair certificates and each search-policy transition", flush=True)
            checks = replay_bike(data, cohorts, epsilons, scales)
            name = "bike_sharing"
        report[name] = {"cohort_sizes": {task: {role: len(record["ids"]) for role, record in roles.items()}
                                        for task, roles in cohorts.items()},
                        "head_checks": head_checks, "screening_checks": checks}
    report["status"] = "PASS"
    report["limitations"] = [
        "Representation transforms are conditional on the retained exact dyadic feature coordinates; native feature-generation roundoff is not recertified.",
        "Head byte counts are reconstructed exactly; representation payload byte counts remain inherited inputs to the archived policy.",
        "Historical timing aggregates are recomputed from retained execution records, without a new timing campaign or independent attestation of execution times.",
        "Three Bike targets share one dataset and satisfy cnt = casual + registered; they are not independent dataset replications.",
        "The replay does not refit encoders or heads, verify every original lifecycle event, or establish population-risk or future-response coverage."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit35.canon(report), indent=2) + "\n", encoding="utf-8")
    print("Public screening: PASS. Result: " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
