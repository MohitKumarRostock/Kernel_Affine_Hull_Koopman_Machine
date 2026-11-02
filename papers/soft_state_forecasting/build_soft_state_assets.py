#!/usr/bin/env python3
"""Build reporting assets from the completed, independently audited ablation.

This is a reporting program, not part of the frozen scientific pipeline. It
imports neither that pipeline nor its analyzer/audit, never opens episode,
target, model-state or forecast arrays, and performs no fitting or selection.
It checks released CSVs against their hashes and independently recomputes the
four prespecified paired bounds from the saved repetition summaries. Run only
after analysis/analysis_summary.json and verification/independent_audit.json
are complete. An existing output directory is never overwritten.

Example, from the repository root::

    python papers/soft_state_forecasting/build_soft_state_assets.py

The default output is generated/soft_state_ablation beside this file. TeX
tables contain tabular environments only; callers supply captions and labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent
DEFAULT_STUDY = REPOSITORY / "extensions/soft_state_ablation_20261102"
N = 32
TASKS = 256
FAMILY_ALPHA = 0.05
ARMS = ("G", "P", "K", "R")
VIEWS = ("shared", "independent")
KEYS = tuple(f"{view}__{arm}" for view in VIEWS for arm in ARMS)
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
CAPS = (1, 2, 4, 8, 12)
HORIZONS = (1, 4, 12)
PANELS = {"nonlinear": LAWS[1:], "linear": LAWS[:1], "all_descriptive": LAWS}
CLAIMS = (
    ("P_vs_G_nonlinear", "nonlinear", "P", "G", 0.97, "3% superiority", "PG"),
    ("K_vs_P_nonlinear", "nonlinear", "K", "P", 0.97, "3% superiority", "KP"),
    ("K_vs_R_nonlinear", "nonlinear", "K", "R", 0.97, "3% superiority", "KR"),
    ("K_vs_P_linear", "linear", "K", "P", 1.05, "5% noninferiority", "LinearKP"),
)
EXPECTED_ROWS = {
    "task_losses.csv": 2048,
    "horizon_losses.csv": 6144,
    "repetition_losses.csv": 768,
    "panel_summary.csv": 24,
    "condition_summary.csv": 64,
    "primary_contrasts.csv": 4,
    "primary_paired_repetitions.csv": 128,
    "secondary_descriptive.csv": 24,
    "selection_by_task.csv": 2048,
    "resource_by_task.csv": None,
    "basis_dimensions.csv": 17920,
    "feature_widths.csv": 5120,
    "B_candidate_scores.csv": 76800,
    "C_diagnostic_losses.csv": 6144,
    "horizon_summary.csv": 72,
    "cap_selection_frequencies.csv": 120,
    "resource_summary.csv": None,
}
LAW_LABEL = {"linear_ar": "Linear AR", "nonlinear_ar": "Nonlinear AR",
             "vanderpol": "Van der Pol", "duffing": "Duffing"}
VIEW_LABEL = {"shared": "Shared", "independent": "Independent"}
PANEL_LABEL = {"nonlinear": "Nonlinear", "linear": "Linear"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    number = float(value)
    require(math.isfinite(number), f"Non-finite reporting value: {value!r}")
    return number


def near(actual, expected, label):
    require(math.isclose(finite(actual), finite(expected), rel_tol=3e-10, abs_tol=3e-13),
            f"Reporting verification failed at {label}: {actual!r} != {expected!r}")


def boolean(value):
    if isinstance(value, bool):
        return value
    require(value in ("True", "False"), f"Invalid Boolean: {value!r}")
    return value == "True"


def mean(values):
    values = list(values)
    require(bool(values), "Empty reporting aggregate")
    return math.fsum(values) / len(values)


def sample_sd(values):
    values = list(values)
    mu = mean(values)
    require(len(values) > 1, "Sample SD needs multiple repetitions")
    return math.sqrt(math.fsum((x - mu) ** 2 for x in values) / (len(values) - 1))


def index_rows(rows, keys, name):
    result = {tuple(str(row[k]) for k in keys): row for row in rows}
    require(len(result) == len(rows), "Duplicate records in " + name)
    return result


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Artifact timestamp lacks timezone")
    return parsed


class Inputs:
    def __init__(self):
        self.files = {}

    def raw(self, path, name):
        data = Path(path).read_bytes()
        record = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        require(name not in self.files or self.files[name] == record,
                "Input changed during reporting: " + name)
        self.files[name] = record
        return data

    def json(self, path, name):
        def bad_constant(value):
            raise ValueError("Non-finite JSON constant: " + value)
        return json.loads(self.raw(path, name), parse_constant=bad_constant)

    def digest(self, name):
        return self.files[name]["sha256"]

    def csv(self, path, name):
        text = self.raw(path, name).decode("utf-8")
        return list(csv.DictReader(io.StringIO(text, newline="")))


def validate_evidence(study, analysis, audit_path, inputs):
    """Check source/phase metadata, completed audit, and every released table.

    The separate audit, not this display builder, verifies the underlying
    episode/model/forecast array hashes and independently replays predictions.
    """
    documents = {}
    for name in ("config.json", "freeze.json", "PUBLICATION.json", "FIT_LOCK.json",
                 "PREDICTION_LOCK.json", "TARGET_RELEASE.json",
                 "prepared/abc_manifest.json", "prepared/d_manifest.json"):
        documents[name] = inputs.json(study / name, "study/" + name)
    config, freeze = documents["config.json"], documents["freeze.json"]
    require(config["master_seed"] == 2026110203 and
            not config.get("manufactured_fixture", False), "Production cohort required")
    require(config["repetitions"] == N and config["task_count"] == TASKS,
            "Fixed complete cohort is required; partial reporting is forbidden")
    require(tuple(config["laws"]) == LAWS and tuple(config["horizons"]) == HORIZONS and
            config["histories"] == [12] and config["noise_levels"] == [0.0, 0.1] and
            tuple(config["native_cap_order"]) == CAPS, "Study condition roster changed")
    require(config["arms"] == list(ARMS) and config["contexts"] == list(VIEWS),
            "Arm/cap-context roster changed")
    plan = config["analysis"]
    require(plan["primary_context"] == "shared" and plan["independent_unit"] == "repetition"
            and plan["family_alpha"] == FAMILY_ALPHA and plan["primary_family_size"] == 4,
            "Primary family or inferential unit changed")
    require(plan["primary_claims"] == [
        {"candidate": a, "population": panel, "reference": b, "reference_factor": factor}
        for _, panel, a, b, factor, _, _ in CLAIMS], "Primary contrast roster changed")
    require(freeze["status"] == "frozen" and freeze["data_generated"] is False and
            freeze["D_generated"] is False and
            freeze["config_sha256"] == inputs.digest("study/config.json"),
            "Pre-generation source freeze does not match the config")
    # Only source/protocol files are in this frozen map; do not open data arrays.
    for relative, expected in freeze["files"].items():
        rel = Path(relative)
        path = (study / rel).resolve()
        require(not rel.is_absolute() and ".." not in rel.parts and
                path.is_relative_to(study), "Frozen source path leaves study root")
        require(path.suffix not in (".npz", ".npy"), "Unexpected array in source map")
        inputs.raw(path, "study/" + relative)
        require(inputs.digest("study/" + relative) == expected,
                "Frozen source mismatch: " + relative)

    summary = inputs.json(analysis / "analysis_summary.json", "analysis/analysis_summary.json")
    audit = inputs.json(audit_path, "verification/independent_audit.json")
    require(summary["schema"] == "soft-state-ablation-analysis-v1" and
            summary["status"] == "complete", "Scientific analysis is not complete")
    require(audit["schema"] == "independent-soft-state-audit-v1" and
            audit["status"] == "passed", "Completed independent audit required")
    for doc in (summary, audit):
        require(doc["tasks"] == TASKS and doc["independent_repetitions"] == N,
                "Incomplete analysis/audit population")
        for key, file in (("config_sha256", "config.json"), ("freeze_sha256", "freeze.json")):
            require(doc[key] == inputs.digest("study/" + file), "Analysis/audit source link: " + key)
    require(summary["primary_family_size"] == 4 and summary["forecast_arrays"] == 2048,
            "Incomplete analysis family/forecast roster")
    require(audit["D_forecast_arrays_replayed"] == 2048 and
            audit["C_forecast_arrays_replayed"] == 2048 and
            audit["B_penalty_path_arrays_replayed"] == 25600,
            "Independent replay did not cover every declared path")
    require(audit["analysis_summary_sha256"] == inputs.digest("analysis/analysis_summary.json"),
            "Audit is not bound to this analysis summary")
    analyzer = "analysis_code/analyze_soft_state.py"
    require(summary["analyzer_sha256"] == inputs.digest("study/" + analyzer) == freeze["files"][analyzer],
            "Analysis was not produced by the frozen analyzer")
    for name in ("verify_soft_state.py", "independent_oracle.py"):
        rel = "audit/" + name
        require(audit["source_hashes"][name] == inputs.digest("study/" + rel) == freeze["files"][rel],
                "Independent audit source mismatch: " + name)

    publication = documents["PUBLICATION.json"]
    abc, terminal = documents["prepared/abc_manifest.json"], documents["prepared/d_manifest.json"]
    fits, predictions, release = (documents[x] for x in
                                  ("FIT_LOCK.json", "PREDICTION_LOCK.json", "TARGET_RELEASE.json"))
    require(publication["schema"] == "kahm-soft-state-publication-v1" and
            publication["status"] == "published" and publication["source_files_verified"] is True and
            publication["freeze_sha256"] == inputs.digest("study/freeze.json") and
            publication["repository"] == config["source"]["repository"] and bool(publication["branch"]),
            "Source publication receipt does not bind this study")
    for key in ("source_commit", "source_tree_sha"):
        require(re.fullmatch(r"[0-9a-f]{40}", publication[key]) is not None, "Invalid Git identity: " + key)
    require(timestamp(freeze["created_utc"]) <= timestamp(publication["published_utc"]) <
            timestamp(abc["created_utc"]), "Source publication must precede production data")
    chronology = [freeze, abc, fits, terminal, predictions, release, summary, audit]
    require(all(timestamp(a["created_utc"]) <= timestamp(b["created_utc"])
                for a, b in zip(chronology, chronology[1:])), "Recorded phase order is inconsistent")
    roster = {(r, law, noise) for r in range(N) for law in LAWS for noise in (0, 1)}
    for manifest in (abc, terminal):
        require(len(manifest["tasks"]) == TASKS and
                len({row["task_id"] for row in manifest["tasks"]}) == TASKS and
                {(row["repetition"], row["law"], row["noise_index"]) for row in manifest["tasks"]} == roster,
                "Prepared manifest has an incomplete or duplicate task roster")
        require(manifest["publication_sha256"] == inputs.digest("study/PUBLICATION.json"),
                "Prepared manifest publication link mismatch")
    task_ids = {row["task_id"] for row in abc["tasks"]}
    require(task_ids == {row["task_id"] for row in terminal["tasks"]}, "ABC/D task identities differ")
    for document in (abc, terminal, fits, predictions):
        require(document["config_sha256"] == inputs.digest("study/config.json") and
                document["freeze_sha256"] == inputs.digest("study/freeze.json"), "Stage source links differ")
    for document, phase, manifest_name in (
            (fits, "fits", "prepared/abc_manifest.json"),
            (predictions, "predictions", "prepared/d_manifest.json")):
        require(document["status"] == "sealed" and document["phase"] == phase and
                document["task_count"] == TASKS and len(document["task_ids"]) == TASKS and
                set(document["task_ids"]) == task_ids and
                document["manifest_sha256"] == inputs.digest("study/" + manifest_name),
                "Stage seal does not cover the complete prepared population")
    require(terminal["fit_lock_sha256"] == inputs.digest("study/FIT_LOCK.json") and
            terminal["ABC_manifest_sha256"] == inputs.digest("study/prepared/abc_manifest.json"),
            "D preparation is not bound to the global fit seal")
    require(release["status"] == "released" and release["task_count"] == TASKS,
            "Scoring has not been released for the complete study")
    for field, filename in (("fit_lock_sha256", "FIT_LOCK.json"),
                            ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                            ("d_manifest_sha256", "prepared/d_manifest.json")):
        require(release[field] == inputs.digest("study/" + filename), "Release hash link mismatch: " + field)
    for field, filename in (("fit_lock_sha256", "FIT_LOCK.json"),
                            ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                            ("target_release_sha256", "TARGET_RELEASE.json")):
        require(summary[field] == inputs.digest("study/" + filename), "Analysis hash link mismatch: " + field)

    require(set(summary["tables"]) == set(EXPECTED_ROWS), "Released analysis table roster changed")
    tables = {}
    for name, count in EXPECTED_ROWS.items():
        rows = inputs.csv(analysis / name, "analysis/" + name)
        expected = summary["tables"][name]
        require(inputs.digest("analysis/" + name) == expected["sha256"], "Released table hash mismatch: " + name)
        require(len(rows) == expected["rows"] and bool(rows) and (count is None or len(rows) == count),
                "Released table row count mismatch: " + name)
        tables[name] = rows
    return config, publication, summary, audit, tables, task_ids


def recompute_statistics(tables, summary, audit, task_ids):
    """Bookkeeping and fixed statistics over saved losses; no forecast scoring."""
    task = index_rows(tables["task_losses.csv"], ("repetition", "law", "noise_index", "forecast_key"), "task losses")
    require({row["task_id"] for row in task.values()} == task_ids, "Analysis task identities differ from seals")
    expected_task_keys = {(str(rep), law, str(noise), key) for rep in range(N)
                          for law in LAWS for noise in (0, 1) for key in KEYS}
    require(set(task) == expected_task_keys, "Task loss factorial roster changed")
    repeated = index_rows(tables["repetition_losses.csv"], ("panel", "repetition", "forecast_key"), "repetition losses")
    require(set(repeated) == {(panel, str(rep), key) for panel in PANELS for rep in range(N) for key in KEYS},
            "Repetition loss factorial roster changed")
    vectors = {}
    for panel, laws in PANELS.items():
        for key in KEYS:
            values = []
            for rep in range(N):
                row = repeated[panel, str(rep), key]
                require(int(row["condition_count"]) == 2 * len(laws), "Incorrect condition weighting")
                saved = finite(row["smse"])
                require(saved >= 0, "Negative squared-error loss")
                aggregate = mean(finite(task[str(rep), law, str(noise), key]["smse"])
                                 for law in laws for noise in (0, 1))
                near(saved, aggregate, f"repetition/{panel}/{key}/{rep}")
                values.append(saved)
            vectors[panel, key] = values

    panels = index_rows(tables["panel_summary.csv"], ("panel", "forecast_key"), "panel summary")
    summary_panels = index_rows(summary["panel_summary"], ("panel", "forecast_key"), "JSON panel summary")
    require(set(panels) == set(summary_panels) == set(vectors), "Panel summary roster changed")
    for key, values in vectors.items():
        for row in (panels[key], summary_panels[key]):
            require(int(row["n_repetitions"]) == N, "Panel count is not independent repetitions")
            near(row["mean_smse"], mean(values), "panel mean " + str(key))
            near(row["repetition_sd"], sample_sd(values), "panel SD " + str(key))

    conditions = index_rows(tables["condition_summary.csv"], ("law", "noise_index", "forecast_key"), "condition summary")
    require(set(conditions) == {(law, str(noise), key) for law in LAWS for noise in (0, 1) for key in KEYS},
            "Condition summary roster changed")
    for (law, noise, key), row in conditions.items():
        values = [finite(task[str(rep), law, noise, key]["smse"]) for rep in range(N)]
        require(int(row["n_repetitions"]) == N and "descriptive" in row["inference"],
                "Condition summaries must remain descriptive")
        near(row["mean_smse"], mean(values), "condition mean")
        near(row["repetition_sd"], sample_sd(values), "condition SD")

    paired = index_rows(tables["primary_paired_repetitions.csv"], ("contrast_id", "repetition"), "paired primary rows")
    require(set(paired) == {(claim[0], str(rep)) for claim in CLAIMS for rep in range(N)},
            "Primary paired roster changed")
    copies = [index_rows(rows, ("contrast_id",), label) for rows, label in (
        (tables["primary_contrasts.csv"], "primary CSV"),
        (summary["primary_contrasts"], "primary analysis JSON"),
        (audit["primary_contrasts"], "primary audit JSON"))]
    require(all(set(copy) == {(claim[0],) for claim in CLAIMS} for copy in copies), "Incomplete primary family")
    critical = float(student_t.isf(FAMILY_ALPHA / 4, N - 1))
    primary = []
    for cid, panel, candidate, comparator, factor, margin, _ in CLAIMS:
        left, right = vectors[panel, "shared__" + candidate], vectors[panel, "shared__" + comparator]
        difference = [a - factor * b for a, b in zip(left, right)]
        for rep, (a, b, delta) in enumerate(zip(left, right, difference)):
            row = paired[cid, str(rep)]
            for key, value in (("candidate_smse", a), ("comparator_smse", b), ("margin_difference", delta)):
                near(row[key], value, f"paired/{cid}/{rep}/{key}")
        mu, sd = mean(difference), sample_sd(difference)
        se = sd / math.sqrt(N)
        upper = mu + critical * se
        pvalue = float(student_t.cdf(mu / se, N - 1)) if se else (0.0 if mu < 0 else 1.0 if mu > 0 else 0.5)
        require(mean(right) > 0, "Relative gain has a nonpositive denominator")
        computed = {"contrast_id": cid, "view": "shared", "panel": panel,
            "candidate": candidate, "comparator": comparator, "margin": margin,
            "comparator_factor": factor, "candidate_mean_smse": mean(left),
            "comparator_mean_smse": mean(right),
            "point_relative_gain_percent": 100 * (1 - mean(left) / mean(right)),
            "margin_difference_mean": mu, "margin_difference_sd": sd, "margin_difference_se": se,
            "critical_t": critical, "simultaneous_one_sided_upper": upper,
            "criterion_established": upper < 0, "p_one_sided": pvalue,
            "p_bonferroni": min(1.0, 4 * pvalue), "family_alpha": FAMILY_ALPHA,
            "family_size": 4, "n_repetitions": N, "df": N - 1,
            "strict_D_win_repetitions": sum(a < b for a, b in zip(left, right))}
        for number, copy in enumerate(copies):
            row = copy[cid,]
            for key, expected in computed.items():
                if number == 2 and key not in row:  # Audit retains a smaller statistics schema.
                    require(key in {"view", "margin", "family_alpha", "family_size", "n_repetitions", "df",
                                    "strict_D_win_repetitions"}, "Missing audited primary field: " + key)
                    continue
                if isinstance(expected, bool):
                    require(boolean(row[key]) == expected, "Primary strict decision changed: " + cid)
                elif isinstance(expected, (int, float)):
                    near(row[key], expected, cid + "/" + key)
                else:
                    require(row[key] == expected, "Primary label changed: " + cid + "/" + key)
        primary.append(computed)
    established = sum(row["criterion_established"] for row in primary)
    require(summary["criteria_established"] == audit["criteria_established"] == established,
            "Analysis/audit primary decisions disagree")

    secondary = index_rows(tables["secondary_descriptive.csv"], ("panel", "candidate", "comparator"), "secondary contrasts")
    expected_secondary = {(panel, f"{view}__{a}", f"{view}__{b}") for panel in ("nonlinear", "linear")
                          for view in VIEWS for a, b in (("P", "G"), ("K", "P"), ("K", "R"), ("R", "P"))}
    expected_secondary |= {(panel, "independent__" + arm, "shared__" + arm)
                           for panel in ("nonlinear", "linear") for arm in ARMS}
    require(set(secondary) == expected_secondary, "Secondary descriptive roster changed")
    q95 = float(student_t.isf(0.025, N - 1))
    for (panel, candidate, comparator), row in secondary.items():
        left, right = vectors[panel, candidate], vectors[panel, comparator]
        delta = [a - b for a, b in zip(left, right)]
        mu, sd = mean(delta), sample_sd(delta)
        se = sd / math.sqrt(N)
        for key, value in (("candidate_mean_smse", mean(left)), ("comparator_mean_smse", mean(right)),
                           ("difference_mean", mu), ("difference_sd", sd), ("difference_se", se),
                           ("descriptive_95_lower", mu - q95 * se), ("descriptive_95_upper", mu + q95 * se),
                           ("point_relative_gain_percent", 100 * (1 - mean(left) / mean(right))),
                           ("strict_D_win_repetitions", sum(a < b for a, b in zip(left, right)))):
            near(row[key], value, "secondary/" + key)
        require(int(row["n_repetitions"]) == N and "unadjusted" in row["scope"],
                "Secondary intervals must remain descriptive and unadjusted")

    horizon_rows = index_rows(tables["horizon_losses.csv"],
                             ("repetition", "law", "noise_index", "forecast_key", "horizon"), "horizon losses")
    require(set(horizon_rows) == {(*key, str(h)) for key in expected_task_keys for h in HORIZONS},
            "Horizon loss roster changed")
    horizons = index_rows(tables["horizon_summary.csv"], ("panel", "forecast_key", "horizon"), "horizon summary")
    require(set(horizons) == {(panel, key, str(h)) for panel in PANELS for key in KEYS for h in HORIZONS},
            "Horizon summary roster changed")
    for (panel, key, h), row in horizons.items():
        values = [mean(finite(horizon_rows[str(rep), law, str(noise), key, h]["smse"])
                       for law in PANELS[panel] for noise in (0, 1)) for rep in range(N)]
        near(row["mean_smse"], mean(values), "horizon mean")
        require(int(row["n_repetitions"]) == N, "Horizon summary repetition count changed")
    return primary, panels, conditions, secondary, horizons


def validate_reporting_summaries(tables, task_ids):
    selected = index_rows(tables["selection_by_task.csv"], ("task_id", "view", "arm"), "selection by task")
    require(set(selected) == {(tid, view, arm) for tid in task_ids for view in VIEWS for arm in ARMS},
            "Selection roster changed")
    for tid in task_ids:
        shared_caps = {int(selected[tid, "shared", arm]["cap"]) for arm in ARMS}
        require(len(shared_caps) == 1 and shared_caps.issubset(CAPS), "Primary geometry is not shared")
        for field in ("cap", "feature_width", "centered_feature_rank", "B_mean_smse"):
            near(selected[tid, "shared", "P"][field], selected[tid, "independent", "P"][field],
                 "P must select the same cap in both views")
        for view in VIEWS:
            for arm in ARMS:
                row = selected[tid, view, arm]
                require(int(row["cap"]) in CAPS and 0 <= int(row["centered_feature_rank"]) <= int(row["feature_width"]),
                        "Invalid selected native cap or feature rank")
            if view == "shared":
                require(int(selected[tid, view, "K"]["feature_width"]) ==
                        int(selected[tid, view, "R"]["feature_width"]), "Matched K/R width differs")
    frequencies = index_rows(tables["cap_selection_frequencies.csv"], ("panel", "view", "arm", "cap"), "cap frequencies")
    require(set(frequencies) == {(panel, view, arm, str(cap)) for panel in PANELS
                                for view in VIEWS for arm in ARMS for cap in CAPS}, "Cap frequency roster changed")
    for (panel, view, arm, cap), row in frequencies.items():
        relevant = [r for r in selected.values() if r["law"] in PANELS[panel] and
                    r["view"] == view and r["arm"] == arm]
        count = sum(int(r["cap"]) == int(cap) for r in relevant)
        require(int(row["task_count"]) == count and int(row["total_tasks"]) == len(relevant) == N * 2 * len(PANELS[panel]),
                "Cap frequency count mismatch")
        near(row["fraction"], count / len(relevant), "cap fraction")

    resource_rows = index_rows(tables["resource_by_task.csv"], ("task_id", "resource"), "resource by task")
    resources = index_rows(tables["resource_summary.csv"], ("resource",), "resource summary")
    require(set(resource_rows) == {(tid, key[0]) for tid in task_ids for key in resources},
            "Resource measurement roster changed")
    for (resource,), row in resources.items():
        values = [finite(resource_rows[tid, resource]["value"]) for tid in task_ids]
        require(int(row["n_tasks"]) == TASKS and min(values) >= 0, "Invalid measured resource population")
        for key, value in (("sum", math.fsum(values)), ("mean", mean(values)), ("median", statistics.median(values)),
                           ("minimum", min(values)), ("maximum", max(values))):
            near(row[key], value, "resource/" + resource + "/" + key)
    return selected, frequencies, resources


class Assets:
    def __init__(self, root):
        self.root = root
        self.outputs = {}
        self.numeric_cells = []
        self.macros = {}

    def record(self, name):
        path = self.root / name
        self.outputs[name] = {"bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write(self, name, text):
        (self.root / name).write_text(text.rstrip() + "\n", encoding="utf-8")
        self.record(name)

    def json(self, name, value):
        self.write(name, json.dumps(value, indent=2, sort_keys=True, allow_nan=False))

    def number(self, value, key, *, decimals=None, significant=6):
        value = finite(value)
        text = f"{value:.{decimals}f}" if decimals is not None else f"{value:.{significant}g}"
        # Never render a nonzero bound as an unsigned rounded zero.
        if value != 0 and float(text) == 0:
            text = f"{value:.2e}"
        if "e" in text.lower():
            mantissa, exponent = text.lower().split("e")
            display = r"\ensuremath{" + mantissa + r"\times 10^{" + str(int(exponent)) + "}}"
        elif text.startswith("-"):
            display = r"\ensuremath{" + text + "}"
        else:
            display = text
        self.numeric_cells.append({"key": key, "value": value, "display": display})
        return display

    def macro(self, name, value, *, decimals=None, significant=6):
        if isinstance(value, str):
            display = value
        elif isinstance(value, int):
            display = str(value)
            self.numeric_cells.append({"key": "macro/" + name, "value": value, "display": display})
        else:
            display = self.number(value, "macro/" + name, decimals=decimals, significant=significant)
        self.macros[name] = {"value": value, "display": display}

    def table(self, name, columns, headers, rows, note):
        lines = ["% Generated by build_soft_state_assets.py; do not edit.", "% " + note,
                 r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule",
                 " & ".join(headers) + r" \\", r"\midrule"]
        for row in rows:
            if row is None:
                lines.append(r"\addlinespace")
            else:
                require(len(row) == len(headers), "Wrong display table column count: " + name)
                lines.append(" & ".join(str(value) for value in row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write(name + ".tex", "\n".join(lines))


def make_plot(assets, conditions):
    """Descriptive condition heterogeneity; no additional confidence claims."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/kahm-softstate-report-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    order = [(law, noise) for law in (*LAWS[1:], LAWS[0]) for noise in (0, 1)]
    labels = [LAW_LABEL[law] + (r", $\eta=0$" if noise == 0 else r", $\eta=0.1$") for law, noise in order]

    def gains(candidate, comparator):
        values = []
        for law, noise in order:
            a = finite(conditions[law, str(noise), "shared__" + candidate]["mean_smse"])
            b = finite(conditions[law, str(noise), "shared__" + comparator]["mean_smse"])
            require(b > 0, "Condition gain has a nonpositive denominator")
            value = 100 * (1 - a / b)
            values.append(value)
            assets.numeric_cells.append({"key": f"plot/{law}/{noise}/{candidate}_vs_{comparator}", "value": value})
        return values

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 9, "axes.labelsize": 8.2,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.hashsalt": "soft_state_ablation_20261102"}):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), sharey=True,
                                 gridspec_kw={"width_ratios": [0.8, 1.2]})
        y = np.arange(len(order))
        axes[0].scatter(gains("P", "G"), y, color="#2166ac", marker="o", s=24, label="P vs G", zorder=3)
        axes[1].scatter(gains("K", "P"), y - 0.12, color="#00796b", marker="o", s=24, label="K vs P", zorder=3)
        axes[1].scatter(gains("K", "R"), y + 0.12, color="#b45f06", marker="D", s=21, label="K vs R", zorder=3)
        axes[0].set_title("Local position versus memberships")
        axes[1].set_title("Reconstruction enrichment")
        for ax in axes:
            ax.axvline(0, color="0.35", linewidth=0.8, linestyle="--", zorder=1)
            ax.axhline(5.5, color="0.75", linewidth=0.7, zorder=1)
            ax.grid(axis="x", color="0.9", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_xlabel("Relative gain (%)")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False,
                      ncol=2, handletextpad=0.35, columnspacing=1.0)
            ax.margins(x=0.12)
        axes[0].set_yticks(y, labels)
        axes[0].set_ylim(len(order) - 0.5, -0.5)
        fig.subplots_adjust(left=0.245, right=0.99, bottom=0.24, top=0.91, wspace=0.24)
        for extension in ("pdf", "svg", "png"):
            name = "condition_gains." + extension
            kwargs = {"dpi": 220} if extension == "png" else {}
            if extension == "pdf":
                kwargs["metadata"] = {"CreationDate": None, "ModDate": None,
                    "Title": "Shared-cap soft-state ablation: descriptive condition gains"}
            elif extension == "svg":
                kwargs["metadata"] = {"Date": None}
            fig.savefig(assets.root / name, **kwargs)
            assets.record(name)
        plt.close(fig)


def build_displays(assets, config, publication, summary, audit, primary, panels,
                   conditions, secondary, horizons, selected, frequencies, resources):
    num = assets.number
    rows = []
    for view in VIEWS:
        if rows:
            rows.append(None)
        for arm in ARMS:
            row = [VIEW_LABEL[view], arm]
            for panel in ("nonlinear", "linear"):
                source = panels[panel, view + "__" + arm]
                row += [num(source[key], f"aggregate/{view}/{arm}/{panel}/{key}")
                        for key in ("mean_smse", "repetition_sd")]
            rows.append(row)
    assets.table("aggregate_table", "llrrrr", ["Cap rule", "Arm", "Nonlinear mean", "SD", "Linear mean", "SD"], rows,
                 "D SMSE; SD across 32 complete repetitions, not a confidence interval. Independent caps are secondary.")

    rows = []
    for row in primary:
        cid = row["contrast_id"]
        rows.append([PANEL_LABEL[row["panel"]], row["candidate"] + " vs " + row["comparator"],
                     r"3\% sup." if row["panel"] == "nonlinear" else r"5\% NI",
                     num(row["point_relative_gain_percent"], cid + "/gain", decimals=2),
                     num(row["margin_difference_mean"], cid + "/mean", significant=4),
                     num(row["simultaneous_one_sided_upper"], cid + "/upper", significant=4),
                     "Yes" if row["criterion_established"] else "No"])
    assets.table("primary_table", "lllrrrc", ["Population", "Contrast", "Criterion", r"Gain (\%)", r"$\bar d$", "Upper bound", "Met"], rows,
                 "Shared caps; d=L_candidate-factor*L_reference. Four Bonferroni one-sided paired-t upper bounds; met iff upper<0.")

    rows = []
    for view in VIEWS:
        if rows:
            rows.append(None)
        for law in (*LAWS[1:], LAWS[0]):
            for noise in (0, 1):
                rows.append([VIEW_LABEL[view], LAW_LABEL[law], "0" if noise == 0 else "0.1", *[
                    num(conditions[law, str(noise), view + "__" + arm]["mean_smse"],
                        f"condition/{view}/{law}/{noise}/{arm}") for arm in ARMS]])
    assets.table("condition_table", "llrrrrr", ["Cap rule", "Law", r"$\eta$", "G", "P", "K", "R"], rows,
                 "Descriptive D SMSE by law/noise; each mean uses all 32 repetitions and averages the three horizons.")

    rows = []
    for view in VIEWS:
        if rows:
            rows.append(None)
        for arm in ARMS:
            selection = [row for row in selected.values() if row["view"] == view and row["arm"] == arm]
            ranges = []
            for field in ("feature_width", "centered_feature_rank"):
                values = [int(row[field]) for row in selection]
                ranges.append(str(min(values)) if min(values) == max(values) else f"{min(values)}--{max(values)}")
            rows.append([VIEW_LABEL[view], arm, *[
                str(frequencies["all_descriptive", view, arm, str(cap)]["task_count"]) for cap in CAPS], *ranges])
    assets.table("selection_table", "llrrrrrcc", ["Cap rule", "Arm", "1", "2", "4", "8", "12", "Width range", "Rank range"], rows,
                 "B-selected cap counts across all 256 task conditions (not independent inferential units); width excludes intercept; centered feature rank on A.")

    rows = []
    for panel in ("nonlinear", "linear"):
        if rows:
            rows.append(None)
        for view in VIEWS:
            for arm in ARMS:
                rows.append([PANEL_LABEL[panel], VIEW_LABEL[view], arm, *[
                    str(frequencies[panel, view, arm, str(cap)]["task_count"]) for cap in CAPS]])
    assets.table("selection_by_panel_table", "lllrrrrr", ["Population", "Cap rule", "Arm", "1", "2", "4", "8", "12"], rows,
                 "Descriptive cap counts: 192 nonlinear and 64 linear task conditions, each arising from 32 repetitions.")

    specifications = [
        ("Whole fit wall time (s)", "fit/total_wall_seconds", 1, True),
        ("Whole fit CPU time (s)", "fit/total_cpu_seconds", 1, True),
        ("D prediction wall time (s)", "prediction/wall_seconds", 1, True),
        ("D prediction CPU time (s)", "prediction/cpu_seconds", 1, True),
        ("Peak worker RSS (MiB)", "fit/peak_rss_KiB", 1024, False),
        ("All-candidate NPZ states (MiB)", "fit/state_npz_bytes", 1024 ** 2, True),
        ("All-candidate JSON states (MiB)", "fit/state_json_bytes", 1024 ** 2, True),
    ]
    rows = []
    for label, key, divisor, summed in specifications:
        row = resources[key,]
        rows.append([label, num(finite(row["sum"]) / divisor, "resource/" + key + "/sum", decimals=2) if summed else "---",
                     num(finite(row["median"]) / divisor, "resource/" + key + "/median", decimals=2),
                     num(finite(row["maximum"]) / divisor, "resource/" + key + "/maximum", decimals=2)])
    assets.table("resource_table", "lrrr", ["Measured quantity", "Task sum", "Task median", "Task maximum"], rows,
                 "Complete search/all four arms. Summed worker wall time is not parallel elapsed time; RSS is not concurrent aggregate memory; state sizes retain all candidates.")

    rows = []
    for panel in ("nonlinear", "linear"):
        if rows:
            rows.append(None)
        pairs = [(view, f"{view}__{a}", f"{view}__{b}") for view in VIEWS
                 for a, b in (("P", "G"), ("K", "P"), ("K", "R"), ("R", "P"))]
        pairs += [("cross", "independent__" + arm, "shared__" + arm) for arm in ARMS]
        for view, candidate, comparator in pairs:
            row = secondary[panel, candidate, comparator]
            cell = f"secondary/{panel}/{candidate}_vs_{comparator}/"
            contrast = candidate.split("__")[1] + " vs " + comparator.split("__")[1]
            if view == "cross":
                contrast = candidate.split("__")[1]
            rows.append([PANEL_LABEL[panel], "Ind. vs shared" if view == "cross" else VIEW_LABEL[view], contrast,
                         num(row["point_relative_gain_percent"], cell + "gain", decimals=2),
                         *[num(row[key], cell + key, significant=4) for key in
                           ("difference_mean", "descriptive_95_lower", "descriptive_95_upper")]])
    assets.table("secondary_table", "lllrrrr", ["Population", "Cap rule", "Arm contrast", r"Gain (\%)", r"$\bar\Delta$", r"95\% lower", r"95\% upper"], rows,
                 "Unadjusted descriptive paired intervals; Delta=candidate-reference, with no practical margin. Cross-view rows compare independent minus shared for the same arm.")

    rows = []
    for view in VIEWS:
        if rows:
            rows.append(None)
        for arm in ARMS:
            rows.append([VIEW_LABEL[view], arm, *[
                num(horizons[panel, view + "__" + arm, str(h)]["mean_smse"], f"horizon/{view}/{arm}/{panel}/{h}")
                for panel in ("nonlinear", "linear") for h in HORIZONS]])
    assets.table("horizon_table", "llrrrrrr", ["Cap rule", "Arm", "NL $h=1$", "NL $h=4$", "NL $h=12$",
                                               "L $h=1$", "L $h=4$", "L $h=12$"], rows,
                 "Descriptive horizon-specific D SMSE; NL=nonlinear, L=linear. Per-horizon A target variances normalize loss.")

    assets.macro("SARepetitions", N)
    assets.macro("SATasks", TASKS)
    assets.macro("SAFamilySize", 4)
    assets.macro("SACriteriaEstablished", summary["criteria_established"])
    assets.macro("SACriticalT", primary[0]["critical_t"], decimals=6)
    assets.macro("SASolverRecoveries", audit["numerical_solver_recoveries"])
    assets.macro("SAAuditChecks", audit["checks"])
    for row, claim in zip(primary, CLAIMS):
        suffix = claim[-1]
        assets.macro("SAGain" + suffix, row["point_relative_gain_percent"], decimals=2)
        assets.macro("SAGap" + suffix, row["margin_difference_mean"], significant=6)
        assets.macro("SAUpper" + suffix, row["simultaneous_one_sided_upper"], significant=6)
        assets.macro("SAEstablished" + suffix, "Yes" if row["criterion_established"] else "No")
    for view in VIEWS:
        for panel in ("nonlinear", "linear"):
            for arm in ARMS:
                assets.macro("SALoss" + view.title() + panel.title() + arm,
                             finite(panels[panel, view + "__" + arm]["mean_smse"]))
    for macro, key, statistic, divisor in (
            ("SAFitWallHours", "fit/total_wall_seconds", "sum", 3600),
            ("SAFitCPUHours", "fit/total_cpu_seconds", "sum", 3600),
            ("SAPredictionWallSeconds", "prediction/wall_seconds", "sum", 1),
            ("SAPeakWorkerRSSMiB", "fit/peak_rss_KiB", "maximum", 1024),
            ("SAStateNPZMiB", "fit/state_npz_bytes", "sum", 1024 ** 2)):
        assets.macro(macro, finite(resources[key,][statistic]) / divisor, decimals=2)
    assets.write("numbers.tex", "% Generated from completed, independently audited evidence; do not edit.\n" +
                 "\n".join("\\newcommand{\\" + name + "}{" + meta["display"] + "}" for name, meta in assets.macros.items()))
    make_plot(assets, conditions)

    metrics = {"schema": "kahm-soft-state-report-metrics-v1", "source_commit": publication["source_commit"],
               "tasks": TASKS, "independent_repetitions": N, "primary": primary,
               "panel_summary": [panels[key] for key in sorted(panels)],
               "condition_summary": [conditions[key] for key in sorted(conditions)],
               "resource_summary": [resources[key] for key in sorted(resources)],
               "macros": assets.macros,
               "statistical_scope": "Four prespecified shared-cap one-sided upper bounds; all other displays descriptive.",
               "computation_scope": "Whole search with reused geometry; no isolated arm cost or equal-compute claim.",
               "analysis_limitations": summary["limitations"], "independent_audit_scope": audit["scope"]}
    assets.json("report_metrics.json", metrics)
    assets.write("README.md", """# Audited soft-state ablation displays

These files are generated by `build_soft_state_assets.py` from the completed
frozen analysis and its completed independent audit. The builder fits no model,
selects no procedure, and reads no raw targets, forecast arrays, or model states.

## Integration and captions

Input `numbers.tex` once in each document, replacing the preproduction manual
definitions of `SARepetitions` and `SATasks`. Tables provide only `tabular`;
place them inside the document's table environment with a caption and label.
No table suppresses a failed primary criterion. Full-precision values and the
macro contract are in `report_metrics.json`; the manifest records input/output
hashes and formatted numerical values.

- `primary_table.tex`: all four prespecified shared-cap comparisons. The margin
  difference is candidate loss minus 0.97 times reference loss for nonlinear
  superiority, and candidate minus 1.05 times reference for linear
  noninferiority. A strictly negative Bonferroni one-sided upper bound meets
  the corresponding criterion. Gain is 100 times one minus the ratio of mean
  losses. Neither the mean difference nor its bound is a percentage.
- `aggregate_table.tex`: D SMSE means and SD across 32 complete repetitions,
  with six law/noise conditions averaged per nonlinear repetition and two per
  linear repetition. Independent-cap rows are secondary. SD is not a confidence
  interval; test windows and 256 task conditions are not independent units.
- `condition_table.tex` and `horizon_table.tex`: descriptive D SMSE means with
  equal prespecified weighting; these are not additional confirmatory tests.
- `selection_table.tex`: cap frequencies across all 256 task conditions,
  together with selected feature-width and centered-A-rank ranges, excluding
  the unpenalized intercept. `selection_by_panel_table.tex` separates the
  nonlinear and linear counts. Repeated shared rows reflect one P-selected
  bank, not four independent geometry choices. Cap 20 is an alias of 12 at
  history 12 and was not an extra search candidate.
- `secondary_table.tex`: all 24 planned descriptive contrasts. The ordinary
  paired 95% intervals are unadjusted, use candidate minus comparator without
  the primary practical margin, and cannot establish failed primary claims.
- `resource_table.tex`: the measured complete search with shared geometry and
  all four arms. Summed task wall time is not parallel elapsed time. Peak worker
  RSS is not concurrent aggregate memory. Stored NPZ/JSON states include every
  candidate, not just a minimal deployed predictor. C diagnostics are included
  in whole-fit totals; prediction rows cover D prediction. Production analysis,
  independent audit, and this reporting builder are outside these fit costs.
- `condition_gains.pdf` (also SVG/PNG): descriptive shared-cap gains in each
  law/noise condition, calculated from condition mean losses. Positive values
  favor the first arm; the dashed line is zero gain. The last two rows are the
  linear controls. Horizontal scales differ between panels. No confidence
  interval or extra condition-specific superiority test is shown.

All forecasts are direct horizon-specific affine readouts of the selected
representations. These results do not evaluate autonomous state propagation,
establish Koopman closure, or demonstrate performance in new application
domains. Source publication precedes this cohort's generation; earlier studies
on the same laws informed the questions. Recorded stage order and hashes are
procedural evidence, not cryptographic target isolation. The separate audit
replays features, coefficients, predictions, and losses within frozen numerical
tolerances; it does not independently reimplement physical integration or
partition k-means.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--analysis", type=Path, help="Completed analysis directory; defaults to STUDY/analysis")
    parser.add_argument("--audit", type=Path, help="Passed independent audit JSON; defaults to STUDY/verification/independent_audit.json")
    parser.add_argument("--output", type=Path, default=HERE / "generated/soft_state_ablation",
                        help="New output directory only; existing directories are never overwritten")
    args = parser.parse_args()
    study = args.study_root.resolve()
    analysis = (args.analysis or study / "analysis").resolve()
    audit_path = (args.audit or study / "verification/independent_audit.json").resolve()
    output = args.output.resolve()
    require(not output.exists(), "Output already exists; choose a new reporting-only directory: " + str(output))
    require(not output.is_relative_to(study) and not study.is_relative_to(output),
            "Reporting output must not overlap frozen scientific evidence")
    require(not analysis.is_relative_to(output) and not audit_path.is_relative_to(output),
            "Reporting output must not contain its evidence inputs")
    inputs = Inputs()
    inputs.raw(Path(__file__), "reporting/build_soft_state_assets.py")
    config, publication, summary, audit, tables, task_ids = validate_evidence(study, analysis, audit_path, inputs)
    primary, panels, conditions, secondary, horizons = recompute_statistics(tables, summary, audit, task_ids)
    selected, frequencies, resources = validate_reporting_summaries(tables, task_ids)
    # Validation above is read-only. Commit displays atomically only after every
    # completed-evidence check has succeeded; no partial output counts as done.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".soft_state_ablation-report-", dir=output.parent))
    try:
        assets = Assets(staging)
        build_displays(assets, config, publication, summary, audit, primary, panels, conditions,
                       secondary, horizons, selected, frequencies, resources)
        manifest = {"schema": "kahm-soft-state-manuscript-assets-v1", "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(), "reporting_only": True,
            "source_commit": publication["source_commit"], "source_tree_sha": publication["source_tree_sha"],
            "freeze_sha256": inputs.digest("study/freeze.json"),
            "analysis_summary_sha256": inputs.digest("analysis/analysis_summary.json"),
            "independent_audit_sha256": inputs.digest("verification/independent_audit.json"),
            "inputs": inputs.files, "outputs": assets.outputs, "numeric_cells": assets.numeric_cells,
            "primary_recomputed": primary,
            "checks": {"complete_frozen_analysis_and_independent_audit": True,
                "publication_and_recorded_stage_hash_links": True, "all_17_table_hashes_and_rows": True,
                "complete_task_to_repetition_aggregation": True, "four_primary_bounds_from_paired_repetitions": True,
                "primary_matches_csv_analysis_and_independent_audit": True,
                "descriptive_panel_condition_horizon_and_secondary_aggregates": True,
                "selection_and_resource_summaries": True},
            "scope": "Metadata/CSV verification and display arithmetic only. Raw array replay is performed by the bound independent audit.",
            "environment": {"python": platform.python_version(),
                            **{name: version(name) for name in ("numpy", "scipy", "matplotlib")}}}
        # The manifest lists every generated payload; it cannot list its own hash.
        (staging / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        require(not output.exists(), "Output appeared while building; refusing overwrite")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"status": "complete", "reporting_only": True, "output": str(output),
                      "primary_criteria_established": summary["criteria_established"],
                      "payload_files": len(manifest["outputs"])}))


if __name__ == "__main__":
    main()
