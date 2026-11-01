#!/usr/bin/env python3
"""Create the final six-page report from completed, independently audited results.

This reporting program never fits, selects, scores raw forecasts, or changes any
scientific artifact. It refuses to read analysis tables until the production
reproduction audit passes and its seals and analysis-summary hash agree.
The PDF authoring marker was run when this builder was first authored; do not
run it again merely to execute the builder or correct its report layout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
import statistics
from urllib.parse import quote
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COMMIT = "f36e7c797f37d7d073dae42603b3770543d9fb43"
REPRODUCTION_SOURCE_COMMIT = "77553bdcb6cc16fe22862b0727068f86be1d7cde"
REPOSITORY = "https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine"
EXTENSION = "extensions/crossfit_residual_reproduction_20261101"
DEFAULT_RESULTS_REF = "research/kahm-crossfit-residual-2026-11-01"
PRIMARY_ORDER = (
    "crossfit_KAHM_vs_shared_bias", "crossfit_KAHM_vs_independent_RBF",
    "crossfit_KAHM_vs_expanded_nonKAHM", "linear_crossfit_KAHM_vs_ridge",
)
KAHM = "crossfit__calibrated__crossfit_kahm"
INSAMPLE = "crossfit__calibrated__insample_kahm"
BIAS = "crossfit__calibrated__shared_bias"
RBF = "crossfit__calibrated__crossfit_rbf"
POLICY = "policy__calibrated__expanded_nonkahm"
CAPS = (1, 2, 4, 8, 12)
SCALERS = ("standard", "minmax", "tanh_z_1")
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
LAW_LABELS = {"linear_ar": "Linear AR", "nonlinear_ar": "Nonlinear AR",
              "vanderpol": "Van der Pol", "duffing": "Duffing"}


def require(valid, message):
    if not valid:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def flag(value):
    require(value in (True, False, "True", "False"), "Invalid boolean in result table")
    return value is True or value == "True"


def number(value):
    out = float(value)
    require(math.isfinite(out), "A report value is nonfinite")
    return out


def fixed(value, digits=5):
    return f"{number(value):.{digits}f}"


def signed(value, digits=2):
    return f"{number(value):+.{digits}f}"


def scientific(value):
    return f"{number(value):+.3e}"


def direction(gain):
    return "reduced" if gain >= 0 else "increased"


def load_final_evidence(root, audit_path, review_path=None):
    """Gate all result access on the final passing audit and exact hashes."""
    root, audit_path = Path(root).resolve(), Path(audit_path).resolve()
    require(audit_path.is_file(), "Final reproduction audit is missing; no report was created")
    audit = read(audit_path)
    require(audit.get("status") == "passed" and audit.get("mode") == "computational_reproduction",
            "Report requires a passing production computational-reproduction audit")
    provenance = audit["computational_reproduction_provenance"]
    require(provenance["original_completed_run_authenticated"] is False
            and provenance["additional_independent_repetitions"] == 0
            and provenance["same_declared_seed_streams_verified"] is True,
            "Audit does not establish the declared reproduction scope")
    counts = audit["counts"]
    require(counts["complete_tasks_verified"] == 192
            and counts["raw_D_forecast_arrays_scored"] == 8064
            and counts["raw_D_horizon_losses_scored"] == 24192,
            "Incomplete final production audit")
    for name, digest in audit["seals"].items():
        require(sha(root / name) == digest, "An audited execution seal changed: " + name)
    summary_path = root / "analysis/analysis_summary.json"
    require(summary_path.is_file() and sha(summary_path) == audit["analysis_summary_sha256"],
            "Analysis summary is missing or differs from the passing audit")
    summary = read(summary_path)
    require(summary["status"] == "completed" and summary["task_count"] == 192,
            "Final analysis is incomplete")
    require(summary["integrity"]["config_sha256"] == sha(root / "config.json"),
            "Scientific configuration changed after analysis")
    tables = {}
    for name, record in summary["tables"].items():
        path = (root / "analysis" / name).resolve()
        require(path.parent == (root / "analysis").resolve(), "Unconfined analysis filename")
        require(sha(path) == record["sha256"], "An audited analysis table changed: " + name)
        with path.open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
        require(len(tables[name]) == record["rows"], "Analysis table row count differs: " + name)
    review = None
    if review_path is not None and Path(review_path).is_file():
        review = read(review_path)
        require(review["status"] == "passed"
                and review["evidence_sha256"]["analysis_summary"] == sha(summary_path)
                and review["evidence_sha256"]["passing_audit"] == sha(audit_path),
                "Optional independent results review does not bind current evidence")
    return audit, summary, tables, review


def derive_metrics(root, audit_path, audit, summary, tables, review):
    primary = {r["contrast_id"]: r for r in tables["primary_contrasts.csv"]}
    require(set(primary) == set(PRIMARY_ORDER), "Primary comparison roster differs")
    descriptive = {(r["panel"], r["contrast_id"]): r for r in tables["descriptive_contrasts.csv"]}
    lookup = {(r["panel"], r["forecast_key"], int(r["repetition"])): number(r["smse"])
              for r in tables["repetition_losses.csv"]}
    require(len(lookup) == len(tables["repetition_losses.csv"]), "Duplicated repetition identity")
    vector = lambda panel, key: [lookup[panel, key, i] for i in range(24)]
    oof, insample = vector("nonlinear", KAHM), vector("nonlinear", INSAMPLE)
    differences = [a - b for a, b in zip(oof, insample)]
    ablation = descriptive["nonlinear", "crossfit_minus_insample_kahm__calibrated"]
    require(math.isclose(statistics.mean(differences), number(ablation["mean_difference"]),
                         rel_tol=2e-10, abs_tol=2e-12), "Ablation mean does not match paired losses")
    require(sum(v < 0 for v in differences) == int(ablation["better_repetitions"])
            and sum(v > 0 for v in differences) == int(ablation["worse_repetitions"]),
            "Ablation improvement counts disagree")
    selections = {}
    for scope, method in (("crossfit", "kahm"), ("crossfit", "rbf"),
                          ("insample", "kahm"), ("insample", "rbf")):
        selected = [r for r in tables["B_selections.csv"]
                    if r["scope"] == scope and r["method"] == method]
        require(len(selected) == 192 and len({r["task_id"] for r in selected}) == 192,
                "Incomplete correction-selection roster")
        selections[f"{scope}/{method}"] = {
            "requested_cap_counts": dict(sorted(Counter(int(r["cap"]) for r in selected).items())),
            "scaler_counts": dict(Counter(r["scaler"] for r in selected)), "tasks": len(selected),
        }
    rank_counts = Counter()
    for key, count in audit["selected_rank_counts"].items():
        rank_counts[int(key.rsplit("/rank", 1)[1])] += int(count)
    require(sum(rank_counts.values()) == audit["counts"]["selected_local_bases_checked"],
            "Selected-basis rank accounting differs")
    conditional = {(r["law"], number(r["noise_level"]), r["forecast_key"]): number(r["smse"])
                   for r in tables["condition_summary.csv"]}
    conditions = []
    for law in LAWS:
        for noise in (0.0, 0.1):
            loss = conditional[law, noise, KAHM]
            conditions.append({"law": law, "noise": noise, "kahm_smse": loss,
                "gain_vs_bias_percent": 100 * (1 - loss / conditional[law, noise, BIAS]),
                "gain_vs_rbf_percent": 100 * (1 - loss / conditional[law, noise, RBF]),
                "gain_vs_policy_percent": 100 * (1 - loss / conditional[law, noise, POLICY])})
    resources = [json.loads(r["fit_resources_json"]) for r in tables["task_resources.csv"]]
    require(len(resources) == 192, "Resource task roster differs")
    resource_summary = {
        "fit_worker_wall_seconds_sum": sum(number(r["total_wall_seconds"]) for r in resources),
        "fit_worker_cpu_seconds_sum": sum(number(r["total_cpu_seconds"]) for r in resources),
        "peak_worker_rss_MiB": max(number(r["peak_rss_KiB"]) for r in resources) / 1024,
        "complete_task_state_gzip_bytes_sum": sum(int(r["state_gzip_bytes"]) for r in resources),
        "prediction_worker_wall_seconds_sum": sum(number(r["prediction_wall_seconds"])
                                                  for r in tables["task_resources.csv"]),
        "independent_audit_wall_seconds": number(audit["wall_seconds"]),
    }
    evidence = {"analysis_summary": sha(root / "analysis/analysis_summary.json"),
                "reproduction_audit": sha(audit_path), "config": sha(root / "config.json"),
                "freeze": sha(root / "freeze.json"), "report_builder": sha(__file__),
                "tables": {name: record["sha256"] for name, record in summary["tables"].items()}}
    return {"schema": "kahm-crossfit-reproduction-report-metrics-v1", "status": "audited_evidence_loaded",
            "original_source_commit": ORIGINAL_COMMIT, "reproduction_source_commit": REPRODUCTION_SOURCE_COMMIT,
            "same_declared_repetition_units": 24, "additional_independent_repetitions": 0,
            "original_completed_run_authenticated": False,
            "primary": [primary[key] for key in PRIMARY_ORDER],
            "selected_descriptive": {key: descriptive["nonlinear", key] for key in (
                "crossfit_minus_insample_kahm__calibrated", "crossfit_minus_insample_rbf__calibrated",
                "crossfit_minus_insample_matched_rbf__calibrated", "full_minus_half_common",
                "full_minus_half_shared_bias", "crossfit_KAHM_minus_crossfit_matched_rbf__calibrated")},
            "nonlinear_paired_OOF_minus_insample_KAHM": differences,
            "correction_selections_all_192_tasks": selections,
            "selected_native_rank_occurrences": dict(sorted(rank_counts.items())),
            "selected_native_rank_scope": "Occurrences in audited selected common/correction banks, not unique tasks",
            "condition_results": conditions, "resources": resource_summary,
            "stability": {"all_screens_passed": summary["maximum_error_screens_passed"],
                          "maximum_error_ratio": summary["maximum_selected_error_ratio"],
                          "failed_tasks": summary["failed_stability_tasks"]},
            "audit_counts": audit["counts"], "audit_not_claimed": audit["not_claimed"],
            "independent_results_review_included": review is not None,
            "evidence_sha256": evidence}


def chart_paired_differences(values):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(7.1, 2.25), dpi=180)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    positions = list(range(1, 25))
    colors = ["#B66B35" if value > 0 else "#147A86" for value in values]
    axis.bar(positions, values, width=.72, color=colors, linewidth=0)
    axis.axhline(0, color="#334B59", lw=.8)
    mean = statistics.mean(values)
    axis.axhline(mean, color="#10283F", lw=1.1, ls="--", label=f"Mean difference {mean:+.5f}")
    axis.set_xlim(.2, 24.8)
    axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
    axis.set_xlabel("Original cohort repetition", fontsize=9)
    axis.set_ylabel("OOF - in-sample SMSE", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#ADBCC4")
    axis.legend(frameon=False, fontsize=8, loc="best")
    axis.grid(axis="y", alpha=.14)
    axis.set_axisbelow(True)
    figure.subplots_adjust(left=.115, right=.985, bottom=.22, top=.94)
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=180, facecolor="white")
    output.seek(0)
    return output


def build_pdf(output, metrics, audit, summary, tables, results_ref):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                   TableStyle, PageBreak, Image, KeepTogether)

    fonts = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                           ("DejaVuSerif", "DejaVuSerif.ttf"), ("DejaVuSerif-Bold", "DejaVuSerif-Bold.ttf"),
                           ("DejaVuMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                  italic="DejaVu", boldItalic="DejaVu-Bold")
    pdfmetrics.registerFontFamily("DejaVuSerif", normal="DejaVuSerif", bold="DejaVuSerif-Bold",
                                  italic="DejaVuSerif", boldItalic="DejaVuSerif-Bold")
    navy, teal, gray = colors.HexColor("#10283F"), colors.HexColor("#116C78"), colors.HexColor("#52636F")
    pale, rule = colors.HexColor("#F1F5F7"), colors.HexColor("#D3DEE4")
    width = A4[0] - 90
    styles = {
        "title": ParagraphStyle("title", fontName="DejaVuSerif-Bold", fontSize=22, leading=27, textColor=navy, spaceAfter=8),
        "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=11, leading=15, textColor=gray, spaceAfter=13),
        "kicker": ParagraphStyle("kicker", fontName="DejaVu-Bold", fontSize=8, leading=11, textColor=teal, spaceAfter=8),
        "h1": ParagraphStyle("h1", fontName="DejaVuSerif-Bold", fontSize=16, leading=21, textColor=navy, spaceAfter=12),
        "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=10, leading=14, textColor=navy, spaceBefore=9, spaceAfter=5),
        "body": ParagraphStyle("body", fontName="DejaVuSerif", fontSize=9.25, leading=13.2, textColor=navy, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="DejaVu", fontSize=7.65, leading=10.8, textColor=gray, spaceAfter=7),
        "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=7.7, leading=10.5, textColor=navy),
        "cellhead": ParagraphStyle("cellhead", fontName="DejaVu-Bold", fontSize=7.5, leading=10.2, textColor=navy),
        "equation": ParagraphStyle("equation", fontName="DejaVuSerif", fontSize=10, leading=15, alignment=TA_CENTER,
                                    textColor=navy, spaceAfter=9, spaceBefore=4),
    }
    story = []

    def p(text, style="body"):
        return Paragraph(text, styles[style])

    def add(text, style="body"):
        story.append(p(text, style))

    def table(headers, records, widths, *, first_column_bold=False):
        data = [[p(escape(str(value)), "cellhead") for value in headers]]
        for record in records:
            data.append([p(escape(str(value)), "cellhead" if first_column_bold and index == 0 else "cell")
                         for index, value in enumerate(record)])
        result = Table(data, colWidths=[width * fraction for fraction in widths], hAlign="LEFT", repeatRows=1)
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), pale), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), .8, rule),
            ("LINEBELOW", (0, 1), (-1, -1), .35, rule),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(result)
        story.append(Spacer(1, 7))

    primary = {r["contrast_id"]: r for r in metrics["primary"]}
    base, rbf, policy, linear = [primary[k] for k in PRIMARY_ORDER]
    ablation = metrics["selected_descriptive"]["crossfit_minus_insample_kahm__calibrated"]
    gain_base = number(base["point_relative_gain_percent"])
    gain_rbf = number(rbf["point_relative_gain_percent"])
    gain_policy = number(policy["point_relative_gain_percent"])
    passed = sum(flag(row["criterion_established"]) for row in primary.values())
    date = datetime.fromisoformat(audit["completed_utc"]).astimezone(ZoneInfo("Europe/Vienna")).date().isoformat()

    # PAGE 1: outcome-led abstract and the complete primary family.
    add("RESEARCH REPORT  |  COMPUTATIONAL REPRODUCTION  |  " + date, "kicker")
    add("Cross-fitted residual corrections for KAHM forecasting", "title")
    add("A reproduction of the declared simulation cohort with a full-training common predictor", "subtitle")
    add("Abstract", "h2")
    add(f"We reconstructed a published finite forecasting procedure using its prescribed simulation streams and "
        f"independently audited the resulting computation. Across 24 reused repetition units, cross-fitted KAHM "
        f"{direction(gain_base)} nonlinear standardized mean-squared error by {abs(gain_base):.2f}% relative to "
        f"its identical common-plus-bias baseline. The point gains against independently selected cross-fitted RBF "
        f"and the expanded non-KAHM policy were {gain_rbf:+.2f}% and {gain_policy:+.2f}%, respectively. "
        f"{passed} of the four prespecified margin criteria were established. The study evaluates residual "
        f"transfer from fold-trained common models to a common predictor fitted on all training episodes. "
        f"It contributes no additional independent repetitions and does not authenticate the unavailable original execution. [1,2]")
    add("1. Primary results", "h2")
    rows = []
    names = ["Common + shared bias (nonlinear)", "Independent OOF RBF (nonlinear)",
             "Expanded non-KAHM policy (nonlinear)", "Raw ridge (linear)"]
    for name, key in zip(names, PRIMARY_ORDER):
        row = primary[key]
        rows.append([name, f"{fixed(row['candidate_mean_smse'])} / {fixed(row['comparator_mean_smse'])}",
                     signed(row["point_relative_gain_percent"]) + "%", scientific(row["simultaneous_one_sided_lower"]),
                     "Established" if flag(row["criterion_established"]) else "Not established"])
    table(["Reference / panel", "Mean SMSE: KAHM / reference", "Point gain", "Adjusted lower margin gap", "Criterion"],
          rows, [.265, .235, .13, .205, .165])
    add("Positive gains favor cross-fitted KAHM. The first three criteria require at least 3% lower nonlinear loss; "
        "the linear criterion permits at most 5% excess loss over ridge. Each lower bound uses the paired "
        "repetition margin and Bonferroni allocation 0.05/4; a criterion is established only if the bound is positive. [3]", "small")
    add("Interpretation", "h2")
    if flag(policy["criterion_established"]):
        add("The expanded-policy comparison establishes the declared practical margin within this simulation design. "
            "Its scope is the complete calibrated and selected procedure on the specified laws, rather than a general "
            "ordering of KAHM and all non-KAHM forecasting methods. Each remaining primary comparison is reported separately.")
    else:
        add("The expanded-policy comparison does not establish the declared practical advantage. A gain over the "
            "shared baseline identifies useful correction behavior, when present, but cannot replace that broader "
            "comparison. Failure to establish a superiority margin also does not establish equivalence.")
    add("Evidence status. All 192 tasks and 8,064 retained forecast arrays passed the independent reproduction audit. "
        "The audit binds this execution's sources, stage seals, fitted records, forecasts and analysis; the original "
        "completed-run records were not recovered. [4]", "small")
    story.append(PageBreak())

    # PAGE 2: architecture, selection and estimand.
    add("2. Procedure and inferential target", "h1")
    add("The common predictor combines the inherited raw ridge forecast with a common-geometry residual head. "
        "Its final deployment version, g<sub>A</sub>, is fitted using all 64 A episodes and selected on B. Four "
        "whole-episode folds each fit a separate common predictor using the remaining 48 episodes, with independent "
        "B selection of scaler, native cap and penalties. Every A row receives exactly one prediction from a model "
        "that excludes its whole episode. [1,2]")
    add("r<sub>i</sub><sup>OOF</sup> = y<sub>i</sub> - g<sub>-k(i)</sub>(x<sub>i</sub>)<br/>"
        "r<sub>i</sub><sup>IS</sup> = y<sub>i</sub> - g<sub>A</sub>(x<sub>i</sub>)", "equation")
    add("The residual target changes between out-of-fold (OOF) and in-sample (IS) arms. Their deployment common "
        "predictor, B residual-mean bias b, full-A feature candidates and 512 correction rows are shared. Correction "
        "features and responses are centered on A; the residual head uses slopes without an added response-mean "
        "intercept. The forecast is", "body")
    add("f<sub>m</sub>(x) = g<sub>A</sub>(x) + b + &eta;<sub>m</sub> d<sub>m</sub>(x)", "equation")
    table(["Block", "Episodes / retained rows", "Role"], [
        ["A", "64 / 512", "Fit common models, geometry and correction slopes"],
        ["B", "32 / 256", "Select common/correction settings and estimate shared bias"],
        ["C", "32 / 256", "Select per-horizon multipliers and expanded policy family"],
        ["D", "128 / 1,664", "Terminal evaluation after fitting and prediction seals"],
    ], [.12, .27, .61])
    add("KAHM and RBF independently search standardization, min-max scaling and tanh of standardized histories; "
        "caps {1,2,4,8,12}; and ridge penalties {0.0001,0.001,0.01,0.1,1}. C selects each horizon's multiplier "
        "from {0,0.25,0.5,1}. Zero exactly recovers the common-plus-bias baseline. Matched RBF uses KAHM's selected "
        "scaler and cap with its own penalty and multiplier, and remains a secondary mechanism comparison.")
    add("The expanded policy chooses on C among ridge, raw RBF, boosted trees, persistence, common geometry, "
        "bounded/unbounded RBF departure, the shared-bias baseline, and OOF/in-sample RBF corrections. "
        "It excludes KAHM and KAHM-matched RBF; it is a selected policy, not a D-outcome oracle.", "small")
    add("Loss and uncertainty", "h2")
    add("Each task/horizon MSE is divided by the corresponding full-A target variance, floored at 10<super>-12</super>. "
        "Horizons 1,4,12, noise levels 0 and 0.1, and eligible laws receive equal weight. The nonlinear repetition "
        "aggregate uses nonlinear AR, Van der Pol and Duffing; the linear aggregate uses linear AR. Inference uses "
        "24 repetition aggregates, not the dependent origins, folds, 192 tasks or 8,064 arrays.")
    add("g<sub>r</sub> = 0.97 L<sub>reference,r</sub> - L<sub>KAHM,r</sub> "
        "(superiority)<br/>g<sub>r</sub> = 1.05 L<sub>ridge,r</sub> - L<sub>KAHM,r</sub> "
        "(noninferiority)<br/>lower = mean(g) - t<sub>0.9875,23</sub> sd(g)/&radic;24", "equation")
    add("These are approximate paired-t inferences. Repeated B/C reuse is part of development. Episode exclusion "
        "concerns learning on a fixed generated dataset: the inherited added-noise scale is derived from A. "
        "Phase separation is procedural and does not make this reused cohort an independent confirmation "
        "or provide cryptographic target isolation.", "small")
    story.append(PageBreak())

    # PAGE 3: paired residual ablation and common-strength control.
    add("3. Residual-target ablation", "h1")
    add("The OOF-versus-in-sample comparison holds the deployment baseline and candidate feature construction fixed. "
        "Each arm retains its own B/C choices, so the contrast evaluates complete procedures rather than forcing "
        "identical selected correction parameters. Positive differences below mean higher OOF error.")
    rows = []
    for label, kind in (("KAHM", "kahm"), ("Independent RBF", "rbf"), ("Matched RBF", "matched_rbf")):
        row = metrics["selected_descriptive"][f"crossfit_minus_insample_{kind}__calibrated"]
        rows.append([label, f"{fixed(row['left_mean_smse'])} / {fixed(row['right_mean_smse'])}",
                     scientific(row["mean_difference"]),
                     f"[{scientific(row['lower95'])}, {scientific(row['upper95'])}]",
                     f"{int(row['better_repetitions'])} / {int(row['ties'])} / {int(row['worse_repetitions'])}"])
    table(["Correction", "Mean SMSE: OOF / IS", "OOF - IS", "Descriptive 95% interval", "OOF wins / ties / losses"],
          rows, [.18, .22, .145, .285, .17])
    add("Nonlinear panel; 24 paired repetition aggregates. Intervals are unadjusted two-sided paired-t intervals "
        "and are descriptive, separate from the four-claim simultaneous family. [3]", "small")
    story.append(Image(chart_paired_differences(metrics["nonlinear_paired_OOF_minus_insample_KAHM"]),
                       width=width, height=width * 2.25 / 7.1))
    add("Figure 1. Paired nonlinear SMSE differences for calibrated KAHM. Orange bars favor in-sample targets; "
        "teal bars favor OOF targets. Repetition labels refer to the reused original cohort, not newly added samples.", "small")
    add(f"In-sample KAHM had lower nonlinear loss in {int(ablation['worse_repetitions'])} of 24 repetition aggregates, "
        f"OOF KAHM had lower loss in {int(ablation['better_repetitions'])}, and {int(ablation['ties'])} were tied. "
        f"The mean paired OOF-minus-IS difference was {scientific(ablation['mean_difference'])}. The "
        "fold models use 48 episodes while the deployment model uses 64 and may select different settings. "
        "OOF residuals are consequently errors of the fold procedures, not exact errors of g<sub>A</sub>. "
        "Cross-fitting protects episode exclusion; beneficial residual transfer remains an empirical question.")
    add("Common-training allocation", "h2")
    rows = []
    for label, key in (("Common only", "full_minus_half_common"), ("Common + own B bias", "full_minus_half_shared_bias")):
        row = metrics["selected_descriptive"][key]
        rows.append([label, f"{fixed(row['left_mean_smse'])} / {fixed(row['right_mean_smse'])}",
                     signed(row["relative_gain_percent"]) + "%", f"{int(row['better_repetitions'])}/24"])
    table(["Allocation control", "Full-A / half-A SMSE", "Gain of full A", "Full-A wins"], rows, [.32, .30, .20, .18])
    add("The half-training common model uses the first 32 A episode identities and its own B selection. This "
        "diagnostic assesses training allocation; it does not rerun the earlier half-split residual-correction experiment.", "small")
    story.append(PageBreak())

    # PAGE 4: finite dimension/scaler choices and condition heterogeneity.
    add("4. Dimension, preprocessing and heterogeneity", "h1")
    add("The native initial dimension is min(cap, d, N<sub>local</sub> - 1), followed by positive-variance and "
        "projected-range reductions. A requested cap is an upper limit, not the realized rank of every local model. "
        "No extra normal direction is reserved. At history dimension d = 12, the inherited cap-20 candidate is "
        "equivalent to cap 12; this experiment does not evaluate realized dimension 20.")
    labels = [("OOF KAHM", "crossfit/kahm"), ("IS KAHM", "insample/kahm"),
              ("OOF RBF", "crossfit/rbf"), ("IS RBF", "insample/rbf")]
    rows = [[label, *[metrics["correction_selections_all_192_tasks"][key]["requested_cap_counts"].get(cap, 0)
                       for cap in CAPS]] for label, key in labels]
    table(["B-selected correction", "Cap 1", "Cap 2", "Cap 4", "Cap 8", "Cap 12"],
          rows, [.31, .138, .138, .138, .138, .138])
    rows = [[label, *[metrics["correction_selections_all_192_tasks"][key]["scaler_counts"].get(scaler, 0)
                       for scaler in SCALERS]] for label, key in labels]
    table(["B-selected correction", "Standard", "Min-max", "Tanh(z)"], rows, [.40, .20, .20, .20])
    ranks = metrics["selected_native_rank_occurrences"]
    rank_min, rank_max = min(ranks), max(ranks)
    add(f"Each selection row above totals 192 tasks. Across all audited selected common/correction banks, "
        f"{sum(ranks.values()):,} local-basis occurrences had realized ranks from {rank_min} to {rank_max}; "
        f"{ranks.get(0, 0):,} were rank zero. These are bank occurrences, not independent observations or a "
        "ranking of preferred dimensions. Tanh(z) uses A-fitted mean and scale; the table measures how often "
        "validation selected it. It does not isolate its causal contribution or establish that tanh is universally superior.", "small")
    add("Condition-level point gains", "h2")
    rows = [[f"{LAW_LABELS[r['law']]} / {r['noise']:g}", fixed(r["kahm_smse"]),
             signed(r["gain_vs_bias_percent"]) + "%", signed(r["gain_vs_rbf_percent"]) + "%",
             signed(r["gain_vs_policy_percent"]) + "%"] for r in metrics["condition_results"]]
    table(["Law / added noise", "KAHM SMSE", "vs bias", "vs OOF RBF", "vs policy"],
          rows, [.32, .17, .17, .17, .17])
    add("Calibrated cross-fitted KAHM; equal horizon weighting and means over the 24 reused repetitions. "
        "Positive entries favor KAHM. These descriptive cells show heterogeneity and do not replace the "
        "prespecified nonlinear or linear primary comparisons. [3]", "small")
    story.append(PageBreak())

    # PAGE 5: numerical verification, disclosed history and measured cost scope.
    add("5. Numerical verification and computational scope", "h1")
    counts = audit["counts"]
    add("The independent verifier first checked target release and the current source, fit and prediction seals. "
        "It then checked development selections, episode exclusion, shared-state identities, native bases and "
        "selected ridge coefficients, calibration and policy forwarding, and every retained D array's loss and "
        "analysis-table joins. The resulting verification concerns the reconstructed execution. [4]")
    add("The native PCA and selected ridge calculations are independently replayed; the inherited raw feature-map "
        "implementation is reused. The audit does not refit every unselected candidate or independently regenerate "
        "every terminal forecast.", "small")
    table(["Audited quantity", "Verified count / value"], [
        ["Complete task suites", f"{counts['complete_tasks_verified']:,}"],
        ["Whole-episode folds", f"{counts['whole_episode_folds_checked']:,}"],
        ["D forecast arrays / horizon losses", f"{counts['raw_D_forecast_arrays_scored']:,} / {counts['raw_D_horizon_losses_scored']:,}"],
        ["Selected local-basis occurrences", f"{counts['selected_local_bases_checked']:,}"],
        ["Independently recomputed primary bounds", f"{counts['primary_bounds_independently_recomputed']:,}"],
        ["Numerical solver fallback attempts / recoveries", f"{counts.get('numerical_fallback_attempts',0):,} / {counts.get('numerical_fallback_recoveries',0):,}"],
        ["Unrecovered fallback events in the completed reproduction", f"{counts.get('numerical_fallback_failures_in_completed_run',0):,}"],
        ["Maximum KAHM-to-ridge absolute-error ratio", fixed(summary['maximum_selected_error_ratio'], 3)],
    ], [.70, .30])
    screen = "All 192 maximum-error screens passed" if summary["maximum_error_screens_passed"] else \
             f"{len(summary['failed_stability_tasks'])} maximum-error screens failed"
    add(screen + ". The screen threshold is five; passing it does not establish a superiority margin. "
        "Guard activations and native rank-zero fits are recorded components of the numerical procedure.", "small")
    add("Failure history and verifier scope", "h2")
    add("The published original attempt documented a finite-matrix ridge SVD failure before D generation. Its "
        "amended rule retains the original ridge solver and permits checked SciPy gesvd recovery only after "
        "LinAlgError, with fixed reconstruction, orthogonality and normal-equation tests. The reproduction uses "
        "that identical rule. The historical singleton-PCA verifier defect is separate: its correction handles "
        "zero admissible rank before division by N - 1 and preserves the rank/reference checks. Manufactured "
        "regression cases established the correction's behavior; this is not a recovered original failure receipt. [1,5]")
    resource = metrics["resources"]
    table(["Measured resource scope", "Value"], [
        ["Sum of fitting-worker elapsed times", f"{resource['fit_worker_wall_seconds_sum']/60:.1f} min"],
        ["Sum of fitting-worker CPU times", f"{resource['fit_worker_cpu_seconds_sum']/60:.1f} min"],
        ["Largest recorded worker peak resident memory", f"{resource['peak_worker_rss_MiB']:.0f} MiB"],
        ["Complete compressed task-state archives", f"{resource['complete_task_state_gzip_bytes_sum']/(1024**3):.3f} GiB"],
        ["Independent reproduction audit elapsed time", f"{resource['independent_audit_wall_seconds']/60:.1f} min"],
    ], [.70, .30])
    add("Fitting costs include folds, finite searches, ablations, controls and calibration. Summed worker times "
        "are not campaign wall time or isolated deployment latency. Stored task-state bytes include the retained "
        "suite and should not be interpreted as the size of one selected deployment model.", "small")
    story.append(PageBreak())

    # PAGE 6: measured implication, limits and precise reproducibility sources.
    add("6. Interpretation and research implication", "h1")
    add("Three questions must remain separate: whether a correction improves its identical shared baseline; "
        "whether KAHM clears the matched-budget independent RBF margin; and whether the complete procedure "
        "clears the expanded non-KAHM policy margin. The primary table answers all three and the linear "
        "noninferiority question. Secondary matched-RBF or in-sample findings cannot substitute for a failed primary criterion.")
    if number(ablation["mean_difference"]) > 0 and not flag(policy["criterion_established"]):
        add("The observed residual-target comparison points to deployment alignment as the next mechanistic question. "
            "OOF corrections learn errors of independently selected 48-episode common models and are applied to "
            "a separately selected 64-episode model. A future, independently evaluated study could hold the "
            "forecasting budget and comparators fixed while comparing explicitly aligned residual-generation "
            "and deployment rules. This is a hypothesis suggested by the current contrast, not an explanation "
            "proven by it. Further scaler or cap searching alone is not an established remedy.")
    elif flag(policy["criterion_established"]):
        add("The expanded-policy margin motivates evaluation beyond the present cohort. A useful follow-up would "
            "freeze the procedure and comparator budgets before collecting independent simulations or natural "
            "forecasting tasks. Such work would test whether the practical advantage extends beyond the four "
            "laws and finite settings examined here; the present reproduction cannot answer that question.")
    else:
        add("The next research question is which aspects of the residual representation and deployment procedure "
            "account for the observed conditional gains. A new study should retain the independent RBF and "
            "expanded-policy comparators, freeze any altered architecture before evaluation, and use an "
            "independent cohort. The present reproduction provides computational evidence for the recorded "
            "procedure, rather than a basis for selecting further settings on these terminal outcomes.")
    add("Scope of the evidence", "h2")
    add("The study is restricted to four synthetic laws, history width 12, three horizons and finite candidate "
        "menus. Overlapping fold fits and within-episode origins are dependent. B and C are reused development "
        "sets; paired-t uncertainty is approximate. A bounded correction controls magnitude rather than accuracy. "
        "The evidence establishes neither general natural-data suitability nor risk-certificate coverage.")
    add("The original completed numerical records were unavailable. All A/B/C and D arrays and fitted states "
        "were regenerated from the declared streams under the published computational source. The report "
        "therefore describes a retrospective computational reproduction with 24 reused inferential units and "
        "zero added independent repetitions. It authenticates its own new records and chronology; it cannot "
        "authenticate the missing original records, even if rounded summaries agree.")
    add("Reproducibility sources", "h2")
    encoded_ref = quote(results_ref, safe="/")
    references = [
        ("1", "Original frozen scientific protocol", f"{REPOSITORY}/blob/{ORIGINAL_COMMIT}/extensions/crossfit_residual_v2/PROTOCOL.md"),
        ("2", "Published reproduction protocol and source", f"{REPOSITORY}/blob/{REPRODUCTION_SOURCE_COMMIT}/{EXTENSION}/PROTOCOL.md"),
        ("3", "Reproduced results and numerical-archive index", f"{REPOSITORY}/blob/{encoded_ref}/{EXTENSION}/reporting/RESULTS.md"),
        ("4", "Passing independent reproduction audit", f"{REPOSITORY}/blob/{encoded_ref}/{EXTENSION}/verification/reproduction_audit.json"),
        ("5", "Original checked-SVD numerical amendment", f"{REPOSITORY}/blob/{ORIGINAL_COMMIT}/extensions/crossfit_residual_v2/NUMERICAL_AMENDMENT.md"),
    ]
    for number_, label, url in references:
        add(f'[{number_}] <link href="{escape(url, quote=True)}" color="#116C78">{escape(label)}</link>.', "small")
    add("Complete per-task and per-horizon tables are distributed in Analysis.zip; the repository index provides "
        "the results synthesis, selected tables and numerical-archive information. The accompanying ReportMetrics "
        "JSON records full source, audit, analysis-summary and table hashes, the exact plotted differences, and "
        "every value used in this report. The byte identities below bind the measured evidence.", "small")
    for label, key in (("Analysis summary SHA-256", "analysis_summary"), ("Reproduction audit SHA-256", "reproduction_audit")):
        add(f"{label}<br/><font name='DejaVuMono' size='6.5'>{metrics['evidence_sha256'][key]}</font>", "small")

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(rule)
        canvas.setLineWidth(.5)
        canvas.line(45, 38, A4[0] - 45, 38)
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(gray)
        canvas.drawString(45, 26, "KAHM residual study  |  Computational reproduction")
        canvas.drawRightString(A4[0] - 45, 26, f"{document.page} / 6")
        if document.page > 1:
            canvas.setFont("DejaVu", 7)
            canvas.drawString(45, A4[1] - 28, "FULL-TRAINING COMMON PREDICTOR  |  SAME DECLARED SIMULATION COHORT")
        canvas.restoreState()

    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=45, rightMargin=45,
                                 topMargin=49, bottomMargin=49,
                                 title="Cross-fitted residual corrections for KAHM forecasting",
                                 author="Research report", subject="Audited computational reproduction",
                                 pageCompression=1)
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--results-ref", default=DEFAULT_RESULTS_REF)
    parser.add_argument("--replace-output", action="store_true",
                        help="Permit replacing only the PDF and its report-metrics sidecar during visual QA")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    audit_path = (args.audit or root / "verification/reproduction_audit.json").resolve()
    review_path = (args.review or root / "reporting/results_review.json").resolve()
    metrics_path = (args.metrics or output.with_name("KAHM_Crossfit_Reproduction_ReportMetrics.json")).resolve()
    require(output.suffix.lower() == ".pdf", "The report output must be a PDF")
    require(not output.is_relative_to(root / "analysis") and not output.is_relative_to(root / "verification"),
            "Reporting output may not overwrite scientific evidence")
    require(output != metrics_path, "PDF and metrics output paths must differ")
    if not args.replace_output:
        require(not output.exists() and not metrics_path.exists(), "Report output already exists")
    audit, summary, tables, review = load_final_evidence(root, audit_path, review_path)
    metrics = derive_metrics(root, audit_path, audit, summary, tables, review)
    metrics["results_repository_ref"] = args.results_ref
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(output, metrics, audit, summary, tables, args.results_ref)
    from pypdf import PdfReader
    pages = len(PdfReader(str(output)).pages)
    require(pages == 6, f"Expected exactly six pages, found {pages}; visually review and correct layout")
    metrics.update(status="report_created_pending_visual_QA", pdf_pages=pages, pdf_sha256=sha(output),
                   pdf_file=output.name)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": metrics["status"], "pages": pages, "pdf": str(output),
                      "metrics": str(metrics_path), "audit_sha256": metrics["evidence_sha256"]["reproduction_audit"]}, indent=2))


if __name__ == "__main__":
    main()
