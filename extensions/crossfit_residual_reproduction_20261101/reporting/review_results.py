#!/usr/bin/env python3
"""Read-only scientific arithmetic review after release and a passing audit.

The script prints JSON to stdout. It does not modify source, results, fitted
states, forecasts, or documents, and refuses to read analysis tables before
the separately recorded computational-reproduction audit has passed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t


PRIMARY = (
    ("crossfit_KAHM_vs_shared_bias", "nonlinear", "crossfit__calibrated__shared_bias", 0.97),
    ("crossfit_KAHM_vs_independent_RBF", "nonlinear", "crossfit__calibrated__crossfit_rbf", 0.97),
    ("crossfit_KAHM_vs_expanded_nonKAHM", "nonlinear", "policy__calibrated__expanded_nonkahm", 0.97),
    ("linear_crossfit_KAHM_vs_ridge", "linear", "legacy__calibrated__ridge", 1.05),
)
CANDIDATE = "crossfit__calibrated__crossfit_kahm"
PANELS = ("nonlinear", "linear", "all_descriptive")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=2e-10, abs_tol=2e-12),
            label + f": {actual!r} != {expected!r}")


def flag(value):
    require(value in ("True", "False"), "Invalid CSV boolean")
    return value == "True"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def paired(left, right):
    require(len(left) == len(right) == 24, "Paired inference requires exactly 24 repetition units")
    differences = [a - b for a, b in zip(left, right)]
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(24)
    critical = float(student_t.ppf(0.975, 23))
    denominator = statistics.mean(right)
    return {"left_mean_smse": statistics.mean(left), "right_mean_smse": denominator,
            "relative_gain_percent": 100 * (1 - statistics.mean(left) / denominator) if denominator else None,
            "mean_difference": mean, "standard_error": se,
            "lower95": mean - critical * se, "upper95": mean + critical * se,
            "better_repetitions": sum(d < 0 for d in differences),
            "ties": sum(d == 0 for d in differences),
            "worse_repetitions": sum(d > 0 for d in differences)}


def selection_summaries(rows, *, calibration):
    grouped = defaultdict(list)
    for row in rows:
        if calibration and not flag(row["selected"]):
            continue
        panel = "linear" if row["law"] == "linear_ar" else "nonlinear"
        for group_panel in (panel, "all_descriptive"):
            if calibration:
                if row["scope"] not in ("crossfit", "expanded_policy"):
                    continue
                key = (group_panel, row["scope"], row["method"], row["horizon"])
            else:
                if row["scope"] not in ("crossfit", "insample", "full_common", "half",
                                        "fold0", "fold1", "fold2", "fold3"):
                    continue
                key = (group_panel, row["scope"], row["method"])
            grouped[key].append(row)
    summaries = []
    for key, members in sorted(grouped.items()):
        record = dict(zip(("panel", "scope", "method", "horizon") if calibration
                          else ("panel", "scope", "method"), key))
        record["selection_units"] = len(members)
        if calibration:
            record["candidate_counts"] = dict(sorted(Counter(r["candidate"] for r in members).items()))
        else:
            record["requested_cap_counts"] = dict(sorted(Counter(r["cap"] for r in members).items()))
            record["scaler_counts"] = dict(sorted(Counter(r["scaler"] for r in members).items()))
            record["joint_candidate_counts"] = dict(sorted(Counter(r["candidate_id"] for r in members).items()))
        summaries.append(record)
    return summaries


def run(root, audit_path):
    root = Path(root).resolve()
    audit_path = Path(audit_path).resolve()
    require((root / "TARGET_RELEASE.json").is_file(), "Target release is required")
    require(read(root / "TARGET_RELEASE.json")["status"] == "released", "Targets are not released")
    audit = read(audit_path)
    require(audit["status"] == "passed" and audit["mode"] == "computational_reproduction",
            "A passing computational-reproduction audit is required before result review")
    require(audit["counts"]["complete_tasks_verified"] == 192, "All 192 tasks must already be audited")
    for name, expected in audit["seals"].items():
        require(sha(root / name) == expected, "Audited seal changed: " + name)
    analysis = root / "analysis"
    summary_path = analysis / "analysis_summary.json"
    require(sha(summary_path) == audit["analysis_summary_sha256"], "Analysis summary differs from passing audit")
    summary = read(summary_path)
    require(summary["status"] == "completed" and summary["task_count"] == 192,
            "Complete 192-task analysis required")
    cfg = read(root / "config.json")
    provenance = cfg["computational_reproduction"]
    require(provenance["same_cohort_not_independent"] is True
            and provenance["additional_independent_repetitions"] == 0
            and provenance["original_completed_run_authenticated"] is False,
            "Reproduction interpretation metadata differs")
    # No result table is opened until the preceding release/audit gates pass.
    tables = {}
    for name, record in summary["tables"].items():
        path = analysis / name
        require(sha(path) == record["sha256"], "Analysis table changed: " + name)
        tables[name] = read_csv(path)
        require(len(tables[name]) == record["rows"], "Analysis row count differs: " + name)
    vectors = defaultdict(dict)
    for row in tables["repetition_losses.csv"]:
        key, rep = (row["panel"], row["forecast_key"]), int(row["repetition"])
        require(rep not in vectors[key], "Duplicate repetition loss")
        vectors[key][rep] = float(row["smse"])
    require(len(vectors) == 3 * 42, "Expected 42 forecast views in each of three panels")
    def vector(panel, forecast):
        members = vectors[panel, forecast]
        require(set(members) == set(range(24)), "Incomplete repetition vector")
        values = [members[r] for r in range(24)]
        require(all(math.isfinite(value) and value >= 0 for value in values), "Invalid repetition loss")
        return values

    rows = {r["contrast_id"]: r for r in tables["primary_contrasts.csv"]}
    require(set(rows) == {case[0] for case in PRIMARY}, "Four separate primary claims required")
    critical = float(student_t.ppf(1 - 0.05 / 4, 23))
    primary = []
    for identifier, panel, comparator_key, factor in PRIMARY:
        left, right = vector(panel, CANDIDATE), vector(panel, comparator_key)
        row = rows[identifier]
        require(row["panel"] == panel and row["candidate_key"] == CANDIDATE
                and row["comparator_key"] == comparator_key, "Primary contrast identity differs")
        gap = [factor * b - a for a, b in zip(left, right)]
        mean, sd = statistics.mean(gap), statistics.stdev(gap)
        se = sd / math.sqrt(24)
        lower = mean - critical * se
        p = float(student_t.sf(mean / se, 23)) if se else (0.0 if mean > 0 else 1.0 if mean < 0 else 0.5)
        base = paired(left, right)
        calculated = {"candidate_mean_smse": base["left_mean_smse"],
                      "comparator_mean_smse": base["right_mean_smse"],
                      "point_relative_gain_percent": base["relative_gain_percent"],
                      "margin_gap_mean": mean, "margin_gap_sample_sd": sd, "margin_gap_se": se,
                      "critical_t": critical, "simultaneous_one_sided_lower": lower,
                      "p_one_sided": p, "p_bonferroni": min(1.0, 4 * p)}
        for name, value in calculated.items():
            if value is None:
                require(row[name] == "", "Undefined relative gain must remain empty")
            else:
                close(row[name], value, identifier + "/" + name)
        require(flag(row["criterion_established"]) == (lower > 0), "Primary criterion flag differs")
        require(int(row["positive_margin_repetitions"]) == sum(g > 0 for g in gap), "Margin-win count differs")
        primary.append({"contrast_id": identifier, "panel": panel, "comparator_key": comparator_key,
                        "criterion": "3% superiority" if factor == 0.97 else "5% noninferiority",
                        **calculated, "criterion_established": lower > 0,
                        "strict_loss_improvement_repetitions": base["better_repetitions"],
                        "positive_margin_repetitions": sum(g > 0 for g in gap)})
    require(sum(row["criterion_established"] for row in primary) == summary["primary_comparisons_established"],
            "Primary established-count summary differs")

    descriptive = []
    for row in tables["descriptive_contrasts.csv"]:
        calculated = paired(vector(row["panel"], row["left_key"]), vector(row["panel"], row["right_key"]))
        for name, value in calculated.items():
            if value is None:
                require(row[name] == "", "Undefined descriptive relative gain must remain empty")
            else:
                close(row[name], value, row["contrast_id"] + "/" + name)
        # The displayed set is chosen by contrast identity, before seeing outcomes.
        show = (row["contrast_id"].endswith("__calibrated")
                or row["scope"] in ("common_allocation_or_bias", "calibration_same_selection"))
        if show:
            descriptive.append({key: row[key] for key in ("contrast_id", "panel", "scope", "left_key", "right_key")}
                               | calculated)

    stability = tables["stability_screens.csv"]
    require(len(stability) == 192, "Expected 192 task stability screens")
    for row in stability:
        require(flag(row["screen_pass"]) == (float(row["maximum_error_ratio"]) <= 5), "Stability flag differs")
    largest = max(stability, key=lambda row: float(row["maximum_error_ratio"]))
    close(summary["maximum_selected_error_ratio"], largest["maximum_error_ratio"], "Maximum error ratio")
    failures = [row for row in stability if not flag(row["screen_pass"])]
    require(summary["maximum_error_screens_passed"] == (not failures), "Stability summary differs")

    ledger_counts = Counter()
    recovery_tasks = []
    error_maxima = Counter()
    for row in sorted(stability, key=lambda item: item["task_id"]):
        task_id = row["task_id"]
        ledger = read(root / "results" / task_id / "numerical_solver_events.json")
        events = ledger["events"]
        require(ledger["fallback_count"] == len(events), "Fallback event count differs")
        require(all(event["status"] == "recovered" and event["recovered"] is True for event in events),
                "Unrecovered numerical fallback")
        require(ledger["recovered_count"] == len(events) and ledger["failed_count"] == 0,
                "Numerical ledger status differs")
        ledger_counts.update(tasks=1, fallback_attempts=len(events), recovered=len(events), failed=0)
        if events:
            recovery_tasks.append({"task_id": task_id, "law": row["law"], "fallback_count": len(events)})
        for event in events:
            for name in ("relative_reconstruction_error", "left_orthogonality_frobenius", "right_orthogonality_frobenius"):
                error_maxima[name] = max(error_maxima[name], event["checks"][name])
            error_maxima["relative_normal_equation_residual"] = max(
                error_maxima["relative_normal_equation_residual"],
                max(record["relative_normal_equation_residual"] for record in event["ridge_checks"]))
    require(ledger_counts["tasks"] == audit["counts"]["numerical_solver_installed_tasks"] == 192,
            "Audited solver task count differs")
    require(ledger_counts["fallback_attempts"] == audit["counts"].get("numerical_fallback_attempts", 0)
            and ledger_counts["recovered"] == audit["counts"].get("numerical_fallback_recoveries", 0),
            "Audited fallback totals differ")

    return {"status": "passed", "review_scope": "Independent arithmetic review of already audited results; no refitting or selection",
            "reproduction_scope": {"same_declared_repetition_units": 24, "additional_independent_repetitions": 0,
                                   "original_completed_run_authenticated": False,
                                   "scope": "Four synthetic laws, history 12, horizons 1/4/12 and the frozen finite procedure"},
            "primary": primary, "primary_claims_verified": len(primary),
            "descriptive_contrasts_verified": len(tables["descriptive_contrasts.csv"]),
            "descriptive_scope": "Unadjusted 95% paired-t intervals; matched RBF is secondary; own B/C choices may differ by target source",
            "selected_descriptive_contrasts": descriptive,
            "stability": {"tasks": 192, "threshold": 5, "maximum_ratio_task": largest, "failed_tasks": failures,
                          "interpretation": "Maximum-error screen; does not establish predictive superiority"},
            "B_selection_frequencies": selection_summaries(tables["B_selections.csv"], calibration=False),
            "C_selection_frequencies": selection_summaries(tables["C_scores_choices.csv"], calibration=True),
            "selected_rank_counts": audit["selected_rank_counts"],
            "rank_scope": "Occurrences in independently checked selected geometry banks, not independent tasks or preferred-dimension evidence",
            "numerical_ledger": {"counts": dict(ledger_counts), "tasks_with_recoveries": recovery_tasks,
                                 "maximum_recorded_numerical_errors": dict(error_maxima),
                                 "contexts": audit["numerical_recovery_contexts"],
                                 "scope": "New reproduction execution only"},
            "audit_counts": audit["counts"], "audit_limitations": audit["not_claimed"],
            "evidence_sha256": {"analysis_summary": sha(summary_path), "passing_audit": sha(audit_path),
                                "review_script": sha(__file__)},
            "tables_hash_verified": len(tables)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.audit), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
