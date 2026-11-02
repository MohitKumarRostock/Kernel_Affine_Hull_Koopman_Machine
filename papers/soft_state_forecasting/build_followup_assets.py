#!/usr/bin/env python3
"""Build follow-up manuscript displays from existing, audited study evidence.

With a full repository checkout, run this script without arguments.  Otherwise
pass --preprocessing-root and --crossfit-root.  The committed generated assets
are sufficient for LaTeX compilation; regeneration requires the study tables,
source freezes, phase seals, and audit receipts.  This script neither fits a
model nor reruns the scientific analysis or production audits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def one(records, **query):
    found = [r for r in records if all(r[k] == str(v) for k, v in query.items())]
    require(len(found) == 1, f"Expected one row for {query}; found {len(found)}")
    return found[0]


def close(left, right, context):
    require(math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-13),
            f"Numerical inconsistency: {context}: {left} != {right}")


def discover(extension):
    return next((p / "extensions" / extension for p in HERE.parents
                 if (p / "extensions" / extension / "analysis/analysis_summary.json").is_file()), None)


class Study:
    def __init__(self, root, out, prefix, audit_path, auditor_path):
        self.root, self.out, self.prefix = root.resolve(), out, prefix
        self.inputs, self.outputs, self.macros, self.cells, self.cache = {}, {}, {}, [], {}
        self.summary = self.read_json("analysis/analysis_summary.json")
        self.config = self.read_json("config.json")
        self.freeze = self.read_json("freeze.json")
        self.audit = self.read_json(audit_path)
        require(self.summary["status"] == "completed", "Incomplete analysis")
        require(self.audit["status"] == "passed", "Production audit did not pass")
        require(self.freeze["status"] == "frozen", "Source is not frozen")
        require(self.summary["task_count"] == self.config["task_count"], "Task count mismatch")
        require(self.summary["primary_family_size"] == self.config["analysis"]["primary_family_size"],
                "Primary family mismatch")
        close(self.config["analysis"]["family_alpha"], 0.05, "family alpha")
        self.check_hash("config.json", self.freeze["config_sha256"])
        for key, rel in [("config_sha256", "config.json"), ("freeze_sha256", "freeze.json")]:
            self.check_hash(rel, self.summary["integrity"][key])
        self.check_hash("analysis/analysis_summary.json", self.audit["analysis_summary_sha256"])
        self.check_hash(auditor_path, self.audit["audit_source_sha256"])
        for rel, digest in self.freeze["files"].items():
            self.check_hash(rel, digest)
        for rel, digest in self.audit["seals"].items():
            self.check_hash(rel, digest)
        require(self.summary["integrity"]["verified_before_terminal_target_access"],
                "Analysis release gate not verified")
        require(self.summary["maximum_error_screens_passed"], "Recorded stability screen failed")
        for key, count in self.audit["counts"].items():
            if key.endswith("bound_violations"):
                require(count == 0, f"Recorded bound violation: {key}")
        self.metrics = {
            "schema": "followup_manuscript_metrics_v1", "study_directory": self.root.name,
            "scope": "Display extraction from existing audited results; no fitting or new hypothesis tests",
            "audit_receipt": audit_path, "audit_counts": self.audit["counts"],
            "audit_limitations": self.audit["not_claimed"],
            "primary_family_size": self.summary["primary_family_size"],
            "primary_comparisons_established": self.summary["primary_comparisons_established"],
        }

    def check_hash(self, rel, expected):
        actual = sha(self.root / rel)
        require(actual == expected, f"Hash mismatch: {self.root / rel}")
        self.inputs[rel] = actual

    def read_json(self, rel):
        path = self.root / rel
        self.inputs[rel] = sha(path)
        return json.loads(path.read_text())

    def rows(self, name):
        if name not in self.cache:
            rel = "analysis/" + name + ".csv"
            meta = self.summary["tables"][name + ".csv"]
            self.check_hash(rel, meta["sha256"])
            with (self.root / rel).open(newline="") as stream:
                self.cache[name] = list(csv.DictReader(stream))
            require(len(self.cache[name]) == meta["rows"], f"Row count mismatch: {name}")
        return self.cache[name]

    def fmt(self, value, precision=6, key=""):
        value = float(value)
        require(math.isfinite(value), f"Nonfinite display: {key}")
        text = f"{value:.{precision}f}"
        self.cells.append({"key": key, "value": value, "display": text,
                           "decimal_places": precision})
        return text

    def macro(self, suffix, value, meaning, source, precision=2, integer=False):
        name = self.prefix + suffix
        require(re.fullmatch(r"[A-Za-z]+", name), f"Invalid TeX macro: {name}")
        display = f"{int(value):,}" if integer else self.fmt(value, precision, name)
        self.macros[name] = {"value": value, "display": display, "meaning": meaning,
                             "source": source,
                             "rounding": "integer" if integer else f"{precision} decimal places"}

    def write(self, name, content):
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / name
        path.write_text(content.rstrip() + "\n")
        self.outputs[name] = sha(path)

    def table(self, name, columns, header, lines):
        self.write(name, "\n".join([
            r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule", header + r" \\",
            r"\midrule", *[line + r" \\" for line in lines], r"\bottomrule", r"\end{tabular}"]))

    def common_macros(self):
        c, s = self.config, self.summary
        values = {
            "Tasks": (c["task_count"], "Prescribed task suites"),
            "Repetitions": (c["repetitions"], "Repetition units in this study's declared cohort"),
            "History": (c["primary_history"], "History width"),
            "Origins": (c["origins_per_episode"], "Origins per full episode"),
            "PrimaryCount": (s["primary_family_size"], "Original simultaneous primary family size"),
            "PrimaryEstablished": (s["primary_comparisons_established"], "Established original primary criteria"),
            "ArraysPerTask": (s["forecast_arrays_per_task"], "Retained forecast arrays per task"),
            "TotalArrays": (c["task_count"] * s["forecast_arrays_per_task"], "Retained forecast arrays across all tasks"),
        }
        values.update({"Episodes" + k: (v, "Prescribed " + k + " episodes")
                       for k, v in c["split_episodes"].items()})
        values.update({"Rows" + k: (v, "Retained " + k + " histories")
                       for k, v in c["split_caps"].items() if v is not None})
        values["RowsD"] = (c["split_episodes"]["D"] * c["origins_per_episode"], "Retained D histories")
        for suffix, (value, meaning) in values.items():
            self.macro(suffix, value, meaning, ["config.json", "analysis/analysis_summary.json"], integer=True)
        self.macro("MaxErrorRatio", s["maximum_selected_error_ratio"],
                   "Maximum selected task error ratio against raw ridge", "analysis/analysis_summary.json", 4)

    def primary_table(self, roster):
        rows = self.rows("primary_contrasts")
        require({r["contrast_id"] for r in rows} == {x[0] for x in roster}, "Unexpected primary roster")
        passed = 0
        lines = []
        for cid, label in roster:
            r = one(rows, contrast_id=cid)
            require(int(r["n_repetitions"]) == self.config["repetitions"], "Wrong inferential sample size")
            require(int(r["family_size"]) == self.summary["primary_family_size"], "Wrong primary family")
            close(r["point_relative_gain_percent"],
                  100 * (1 - float(r["candidate_mean_smse"]) / float(r["comparator_mean_smse"])), cid)
            close(r["margin_gap_mean"], float(r["comparator_factor"]) * float(r["comparator_mean_smse"])
                  - float(r["candidate_mean_smse"]), cid + " margin")
            close(r["simultaneous_one_sided_lower"], float(r["margin_gap_mean"])
                  - float(r["critical_t"]) * float(r["margin_gap_se"]), cid + " bound")
            established = float(r["simultaneous_one_sided_lower"]) > 0
            require(established == (r["criterion_established"] == "True"), "Primary decision mismatch")
            passed += established
            lines.append(" & ".join([label, self.fmt(r["candidate_mean_smse"], 6, cid + " candidate"),
                self.fmt(r["comparator_mean_smse"], 6, cid + " comparator"),
                self.fmt(r["point_relative_gain_percent"], 2, cid + " gain"),
                self.fmt(r["simultaneous_one_sided_lower"], 6, cid + " lower"),
                "Yes" if established else "No"]))
        require(passed == self.summary["primary_comparisons_established"], "Primary pass count mismatch")
        self.table("primary_table.tex", r"p{0.32\linewidth}rrrrc",
                   r"Comparison & $L_c$ & $L_m$ & Gain (\%) & Lower gap & Pass", lines)
        self.metrics["primary_contrasts"] = rows
        return {r["contrast_id"]: r for r in rows}

    def finish(self):
        self.write("numbers.tex", "\n".join("\\newcommand{\\" + name + "}{" + item["display"] + "}"
                                              for name, item in self.macros.items()))
        self.metrics["macros"] = self.macros
        self.write("metrics.json", json.dumps(self.metrics, indent=2, sort_keys=True))
        manifest = {
            "schema": "followup_manuscript_assets_v1", "study_directory": self.root.name,
            "builder_sha256": sha(Path(__file__)), "inputs": self.inputs,
            "outputs": self.outputs, "numeric_cells": self.cells,
            "source_freeze_sha256": self.inputs["freeze.json"],
            "verification_scope": "Checks existing source/phase/table/audit hash consistency and displayed arithmetic. Does not rerun production audit or authenticate unavailable original records.",
        }
        (self.out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {"study": self.root.name, "input_hashes_verified_or_bound": len(self.inputs),
                "analysis_tables_hash_verified": len(self.cache),
                "generated_files": len(self.outputs) + 1, "macros": len(self.macros)}


def build_preprocessing(study):
    s = study
    s.common_macros()
    p = s.primary_table([
        ("selected_KAHM_vs_nonKAHM_policy", "Selected KAHM / policy"),
        ("selected_KAHM_vs_selected_RBF", "Selected KAHM / RBF"),
        ("tanh_KAHM_vs_standard_KAHM", "Tanh KAHM / standard KAHM"),
        ("linear_selected_KAHM_vs_ridge", "Selected KAHM / ridge (linear)"),
    ])
    for suffix, cid in [("PolicyPenalty", "selected_KAHM_vs_nonKAHM_policy"),
                         ("RBFPenalty", "selected_KAHM_vs_selected_RBF"),
                         ("TanhPenalty", "tanh_KAHM_vs_standard_KAHM"),
                         ("LinearPenalty", "linear_selected_KAHM_vs_ridge")]:
        s.macro(suffix, -float(p[cid]["point_relative_gain_percent"]),
                "Candidate percent excess loss relative to the stated comparator", "analysis/primary_contrasts.csv#" + cid)
    s.macro("NonlinearLoss", float(p["selected_KAHM_vs_selected_RBF"]["candidate_mean_smse"]),
            "Selected calibrated KAHM nonlinear SMSE", "analysis/primary_contrasts.csv", 6)
    panels, contrasts = s.rows("fixed_common_panel_summary"), s.rows("fixed_common_contrasts")
    cells, lines, intervals = [], [], []
    for pair, common, label in [("ss", "standard", "Standard / standard"),
                                ("mm", "minmax", "Min--max / min--max"),
                                ("tt", "tanh_z_1", "Tanh / tanh"),
                                ("tm", "tanh_z_1", "Tanh / min--max"),
                                ("mt", "minmax", "Min--max / tanh")]:
        def loss(context, method):
            return float(one(panels, panel="nonlinear", mode="calibrated", context=context, method=method)["smse"])
        base, kahm, rbf = loss(common, "shared_bias"), loss(pair, "kahm"), loss(pair, "rbf")
        rb = one(contrasts, contrast_id=f"fixed_common_{pair}_KAHM_minus_shared_bias", panel="nonlinear", mode="calibrated")
        rr = one(contrasts, contrast_id=f"fixed_common_{pair}_KAHM_minus_independently_selected_RBF", panel="nonlinear", mode="calibrated")
        close(rb["left_mean_smse"], kahm, pair + " KAHM")
        close(rr["right_mean_smse"], rbf, pair + " RBF")
        close(rb["right_mean_smse"], base, pair + " bias")
        cells.append({"pair": pair, "common_scaler": common, "label": label, "bias_smse": base,
                      "kahm_smse": kahm, "rbf_smse": rbf, "versus_bias": rb, "versus_rbf": rr})
        lines.append(" & ".join([label, s.fmt(base), s.fmt(kahm), s.fmt(rbf),
                                 s.fmt(rb["relative_gain_percent"], 2), s.fmt(rr["relative_gain_percent"], 2)]))
        intervals.append(" & ".join([label, s.fmt(rr["mean_difference"]),
            "[" + s.fmt(rr["lower95"]) + ", " + s.fmt(rr["upper95"]) + "]", rr["better_repetitions"]]))
    s.table("fixed_common_table.tex", r"p{0.29\linewidth}rrrrr",
            r"Common / departure scale & Bias & KAHM & RBF & Gain/bias (\%) & Gain/RBF (\%)", lines)
    s.table("fixed_common_intervals.tex", r"p{0.35\linewidth}rrr",
            r"Common / departure scale & $L_K-L_R$ & Descriptive 95\% interval & KAHM wins", intervals)
    s.metrics["fixed_common_nonlinear_cells"] = cells
    for suffix, comparator, extreme in [("FixedBiasGainMin", "versus_bias", min), ("FixedBiasGainMax", "versus_bias", max),
                                        ("FixedRBFGainMin", "versus_rbf", min), ("FixedRBFGainMax", "versus_rbf", max)]:
        s.macro(suffix, extreme(float(c[comparator]["relative_gain_percent"]) for c in cells),
                "Range endpoint across all five prespecified fixed-common scaler cells; descriptive",
                "analysis/fixed_common_contrasts.csv")
    for suffix, key in [("AuditAssertions", "assertions"), ("AuditBases", "primary_local_bases_independently_reconstructed"),
                        ("AuditRawTasks", "raw_D_tasks_recomputed"), ("AuditRawArrays", "raw_D_prediction_arrays_recomputed"),
                        ("AuditLossRows", "task_loss_table_records_verified")]:
        s.macro(suffix, s.audit["counts"][key], "Preprocessing audit count: " + key,
                "audit/final_results_audit.json", integer=True)
    m = s.config["model"]
    def grid(xs): return "$\\{" + ",".join(f"{v:g}" for v in xs) + "\\}$"
    def power(value):
        mantissa, exponent = f"{value:e}".split("e")
        lead = "" if float(mantissa) == 1 else f"{float(mantissa):g}\\times "
        return "$" + lead + "10^{" + str(int(exponent)) + "}$"
    s.table("model_grid_table.tex", r"p{0.43\linewidth}p{0.51\linewidth}", "Setting & Frozen value", [
        "Ridge penalties, each horizon & " + grid(m["alpha_grid"]),
        "Calibration multipliers, each horizon & " + grid(m["gamma_grid"]),
        "Preprocessing-study native caps & " + grid(s.config["native_cap_order"]),
        "Requested partition counts & " + grid(m["blocks"]),
        f"References per local model / bank rows & At most {m['max_references']} / {m['bank_row_cap']}",
        f"Raw RBF landmark count & At most {m['raw_rbf_landmarks']}",
        "Raw RBF bandwidth factors & " + grid(m["raw_rbf_bandwidth_factors"]),
        "HGB learning rates / leaf counts & " + grid(m["raw_hgb_learning_rates"]) + " / " + grid(m["raw_hgb_max_leaf_nodes"]),
        f"HGB iterations / L2 penalty & {m['raw_hgb_max_iter']} / {m['raw_hgb_l2_regularization']:g}",
        "Feature-scale / target-variance floors & " + power(m['feature_scale_floor']) + " / " + power(m['target_variance_floor']),
    ])
    s.metrics["cohort_status"] = "Fresh repetitions under previously studied synthetic laws; separate cohort and primary family from dimension study"
    return s.finish()


def build_crossfit(study):
    s = study
    s.common_macros()
    provenance = s.audit["computational_reproduction_provenance"]
    require(s.audit["mode"] == "computational_reproduction", "Wrong reproduction audit mode")
    for key in ["original_completed_run_authenticated", "original_array_byte_identity_claimed",
                "original_execution_chronology_authenticated"]:
        require(provenance[key] is False, "Unsupported original-run authentication")
    require(provenance["additional_independent_repetitions"] == 0, "Reproduction cannot add independent repetitions")
    require(provenance["same_declared_seed_streams_verified"] and provenance["scientific_configuration_and_source_identical"],
            "Reproduction source/cohort equivalence is unverified")
    require(provenance["historical_outputs_known_before_reproduction"], "Reproduction chronology scope omitted")
    s.metrics["computational_reproduction_provenance"] = provenance
    meta = s.config["computational_reproduction"]
    original = s.read_json(meta["original_source_manifest"])
    s.check_hash(meta["original_source_manifest"], meta["original_source_manifest_sha256"])
    require(original["original_source_commit"] == provenance["original_source_commit"], "Original source identity mismatch")
    publication = s.read_json("PUBLISHED_SOURCE.json")
    s.check_hash("freeze.json", publication["freeze_sha256"])
    require(publication["published_source_commit"] == publication["verified_remote_head"], "Publication receipt mismatch")
    require(publication["scientific_data_generated_at_verification"] is False, "Wrong publication chronology")
    s.metrics["reproduction_source_publication"] = publication
    review = s.read_json("verification/results_review.json")
    require(review["status"] == "passed", "Independent arithmetic review did not pass")
    for key, rel in [("analysis_summary", "analysis/analysis_summary.json"),
                     ("passing_audit", "verification/reproduction_audit.json"),
                     ("review_script", "reporting/review_results.py")]:
        s.check_hash(rel, review["evidence_sha256"][key])
    s.metrics["numerical_recovery_ledger"] = review["numerical_ledger"]
    p = s.primary_table([
        ("crossfit_KAHM_vs_shared_bias", "OOF KAHM / shared bias"),
        ("crossfit_KAHM_vs_independent_RBF", "OOF KAHM / independent RBF"),
        ("crossfit_KAHM_vs_expanded_nonKAHM", "OOF KAHM / expanded policy"),
        ("linear_crossfit_KAHM_vs_ridge", "OOF KAHM / ridge (linear)"),
    ])
    for suffix, cid, sign in [("GainBias", "crossfit_KAHM_vs_shared_bias", 1),
                              ("GainRBF", "crossfit_KAHM_vs_independent_RBF", 1),
                              ("PolicyPenalty", "crossfit_KAHM_vs_expanded_nonKAHM", -1),
                              ("LinearPenalty", "linear_crossfit_KAHM_vs_ridge", -1)]:
        s.macro(suffix, sign * float(p[cid]["point_relative_gain_percent"]),
                "Percent gain (or named penalty) relative to the primary comparator", "analysis/primary_contrasts.csv#" + cid)
    desc = s.rows("descriptive_contrasts")
    chosen = [("crossfit_minus_insample_kahm__calibrated", "OOF KAHM / in-sample KAHM"),
              ("insample_KAHM_minus_shared_bias__calibrated", "In-sample KAHM / shared bias"),
              ("insample_KAHM_minus_insample_rbf__calibrated", "In-sample KAHM / in-sample RBF"),
              ("crossfit_KAHM_minus_crossfit_matched_rbf__calibrated", "OOF KAHM / matched RBF"),
              ("full_minus_half_common", "Full-A common / half-A common"),
              ("calibrated_minus_unit__crossfit__crossfit_kahm", "Calibrated OOF KAHM / unit weight")]
    d, lines, short = {}, [], []
    for cid, label in chosen:
        r = one(desc, contrast_id=cid, panel="nonlinear")
        d[cid] = r
        close(r["relative_gain_percent"], 100 * (1 - float(r["left_mean_smse"]) / float(r["right_mean_smse"])), cid)
        close(r["mean_difference"], float(r["left_mean_smse"]) - float(r["right_mean_smse"]), cid)
        line = " & ".join([label, s.fmt(r["relative_gain_percent"], 2), s.fmt(r["mean_difference"]),
                            "[" + s.fmt(r["lower95"]) + ", " + s.fmt(r["upper95"]) + "]"])
        lines.append(line)
        if len(lines) <= 4: short.append(line)
    header = r"Candidate / comparator & Gain (\%) & $L_c-L_m$ & Descriptive 95\% interval"
    s.table("secondary_table.tex", r"p{0.39\linewidth}rrr", header, short)
    s.table("secondary_detail_table.tex", r"p{0.39\linewidth}rrr", header, lines)
    s.metrics["selected_descriptive_contrasts"] = d
    for suffix, cid, sign in [("ISGainBias", "insample_KAHM_minus_shared_bias__calibrated", 1),
                              ("ISGainRBF", "insample_KAHM_minus_insample_rbf__calibrated", 1),
                              ("OOFvsISPenalty", "crossfit_minus_insample_kahm__calibrated", -1),
                              ("MatchedRBFGain", "crossfit_KAHM_minus_crossfit_matched_rbf__calibrated", 1),
                              ("FullCommonGain", "full_minus_half_common", 1),
                              ("CalibrationGain", "calibrated_minus_unit__crossfit__crossfit_kahm", 1)]:
        s.macro(suffix, sign * float(d[cid]["relative_gain_percent"]),
                "Secondary nonlinear percent gain (or named penalty); original descriptive comparison",
                "analysis/descriptive_contrasts.csv#" + cid)
    oo = d["crossfit_minus_insample_kahm__calibrated"]
    for suffix, key in [("OOFvsISDifference", "mean_difference"), ("OOFvsISLower", "lower95"), ("OOFvsISUpper", "upper95")]:
        s.macro(suffix, float(oo[key]), "OOF minus in-sample KAHM nonlinear SMSE; descriptive interval",
                "analysis/descriptive_contrasts.csv#crossfit_minus_insample_kahm__calibrated", 6)
    s.macro("ISWinningRepetitions", int(oo["worse_repetitions"]), "Repetition aggregates with lower in-sample than OOF KAHM loss",
            "analysis/descriptive_contrasts.csv", integer=True)
    panels = s.rows("panel_summary")
    def loss(family, method):
        return float(one(panels, panel="nonlinear", family=family, method=method, mode="calibrated")["smse"])
    is_loss, policy = loss("crossfit", "insample_kahm"), loss("policy", "expanded_nonkahm")
    s.macro("ISLoss", is_loss, "Calibrated in-sample KAHM nonlinear SMSE", "analysis/panel_summary.csv", 6)
    s.macro("ISPolicyPenalty", 100 * (is_loss / policy - 1),
            "In-sample KAHM percent excess loss relative to expanded policy; point description only, no added hypothesis test",
            "analysis/panel_summary.csv")
    s.metrics["insample_vs_policy_point_description"] = {"insample_smse": is_loss, "expanded_policy_smse": policy,
        "excess_percent": 100 * (is_loss / policy - 1), "inference": "None; point description only"}
    selections, choices = s.rows("B_selections"), s.rows("C_scores_choices")
    selection_lines, calibration_lines, settings = [], [], {}
    for scope, method, label in [("full_common", "common", "Full-A common"), ("crossfit", "kahm", "OOF KAHM"),
                                 ("crossfit", "rbf", "OOF RBF"), ("insample", "kahm", "In-sample KAHM")]:
        selected = [r for r in selections if r["scope"] == scope and r["method"] == method]
        require(len(selected) == s.config["task_count"], "Selected procedure roster mismatch")
        caps = Counter(int(r["cap"]) for r in selected)
        scalers = Counter(r["scaler"] for r in selected)
        settings[label] = {"cap_counts": dict(caps), "scaler_counts": dict(scalers)}
        selection_lines.append(" & ".join([label, *[str(caps[c]) for c in s.config["native_cap_order"]],
                                           *[str(scalers[c]) for c in s.config["scaler_order"]]]))
    for method, label in [("crossfit_kahm", "OOF KAHM"), ("crossfit_rbf", "OOF RBF"), ("insample_kahm", "In-sample KAHM")]:
        selected = [r for r in choices if r["scope"] == "crossfit" and r["method"] == method and r["selected"] == "True"]
        require(len(selected) == s.config["task_count"] * len(s.config["horizons"]), "Calibration roster mismatch")
        counts = Counter(float(r["candidate"]) for r in selected)
        settings[label]["eta_counts"] = dict(counts)
        calibration_lines.append(" & ".join([label, *[str(counts[e]) for e in s.config["model"]["gamma_grid"]]]))
    s.table("selection_table.tex", "lrrrrrrrr", r"Procedure & Cap 1 & 2 & 4 & 8 & 12 & Standard & Min--max & Tanh", selection_lines)
    s.table("calibration_table.tex", "lrrrr", r"Procedure & $\eta=0$ & $1/4$ & $1/2$ & $1$", calibration_lines)
    s.metrics["selection_frequencies"] = settings
    s.macro("KAHMTanhSelections", settings["OOF KAHM"]["scaler_counts"].get("tanh_z_1", 0),
            "OOF KAHM B selections using tanh; frequency is not a causal effect", "analysis/B_selections.csv", integer=True)
    s.macro("KAHMEtaZero", settings["OOF KAHM"]["eta_counts"].get(0.0, 0),
            "OOF KAHM task-horizon C choices returning shared-bias baseline", "analysis/C_scores_choices.csv", integer=True)
    s.macro("CalibrationCells", s.config["task_count"] * len(s.config["horizons"]),
            "Task-horizon calibration choices per correction procedure", "config.json", integer=True)
    for suffix, key in [("AuditAssertions", "assertions"), ("AuditBases", "selected_local_bases_checked"),
                        ("AuditArrays", "raw_D_forecast_arrays_scored"), ("ZeroRanks", "selected_zero_admissible_rank_bases"),
                        ("SVDRecoveries", "numerical_fallback_recoveries"), ("SVDFailures", "numerical_fallback_failures_in_completed_run")]:
        s.macro(suffix, s.audit["counts"][key], "Reproduction audit count: " + key,
                "verification/reproduction_audit.json", integer=True)
    s.macro("RecoveryTasks", len(review["numerical_ledger"]["tasks_with_recoveries"]),
            "Reproduction tasks with recorded SVD recoveries", "verification/results_review.json", integer=True)
    s.macro("AdditionalRepetitions", provenance["additional_independent_repetitions"],
            "Additional independent repetition units contributed by this computational reproduction",
            "verification/reproduction_audit.json", integer=True)
    return s.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-root", type=Path)
    parser.add_argument("--crossfit-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    pp = args.preprocessing_root or discover("preprocessing_dimension")
    cf = args.crossfit_root or discover("crossfit_residual_reproduction_20261101")
    if pp is None or cf is None:
        parser.error("Pass both study roots when the repository evidence is not available beside this manuscript.")
    preprocessing = Study(pp, args.output_root / "preprocessing_dimension", "PP",
                          "audit/final_results_audit.json", "audit/verify_results.py")
    crossfit = Study(cf, args.output_root / "crossfit_residual", "CFR",
                     "verification/reproduction_audit.json", "audit/verify_reproduction.py")
    require(preprocessing.config["model"] == crossfit.config["model"], "Inherited model grids differ")
    require(preprocessing.config["master_seed"] != crossfit.config["master_seed"], "Distinct study cohorts were conflated")
    results = [build_preprocessing(preprocessing), build_crossfit(crossfit)]
    print(json.dumps({"status": "completed", "studies": results}, indent=2))


if __name__ == "__main__":
    main()
