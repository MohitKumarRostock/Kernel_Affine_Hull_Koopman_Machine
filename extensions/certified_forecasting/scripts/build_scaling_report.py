#!/usr/bin/env python3
"""Build a standalone, data-driven scientific comparison report."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
    KeepTogether,
)


ROOT = Path(__file__).resolve().parents[1] / "results/input_scaling_study"
OUT = Path.cwd() / "scaling_report"
BASE = "https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine"
SHA = "a8bcf6c8016499e1a4b64f7dd4196af43c60bd78"
TASKS = ["harmonic", "quasiperiodic", "nonlinear_ar", "logistic", "vanderpol", "duffing", "pendigits"]
LABELS = {"harmonic": "Harmonic", "quasiperiodic": "Quasiperiodic",
          "nonlinear_ar": "Nonlinear AR", "logistic": "Logistic",
          "vanderpol": "Van der Pol", "duffing": "Duffing", "pendigits": "PenDigits"}
MODES = {"standard": "Standardization", "minmax": "Min-max",
         "minmax_clip": "Clipped min-max", "raw": "Bounded raw"}
FAMILIES = {"kahm_nested": "Retained soft blocks", "kahm_frobenius": "Frobenius decoder",
            "kahm_duplicate": "Duplicate-block ablation", "kahm_hard": "Hard assignment",
            "kahm_koopman": "Koopman rollout"}
NAVY = colors.HexColor("#183248")
TEAL = colors.HexColor("#087f8c")
LIGHT = colors.HexColor("#edf4f7")
GRAY = colors.HexColor("#52616b")


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    return float(row[key])


def pct(value):
    return f"{value:+.2f}%"


def body(text, style="Body"):
    return Paragraph(text, STYLES[style])


def table(data, widths, *, font_size=8.5):
    converted = []
    for ri, row in enumerate(data):
        style = "TableHead" if ri == 0 else "TableBody"
        converted.append([body(escape(str(cell)), style) for cell in row])
    t = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#bacbd5")),
    ]))
    return t


def task_rows(source, mode="minmax", family="kahm_nested", primary=True):
    found = [r for r in source if r["mode"] == mode and r["family"] == family
             and r["horizon"] == "all"]
    if primary:
        found = [r for r in found if r["campaign"] in {"main_v1", "oscillators_v1", "pendigits_v1"}]
    return {r["system"]: r for r in found}


def plot_effects(original, fresh):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    y = np.arange(len(TASKS))
    values = [fnum(original[s], "relative_mse_change_percent") for s in TASKS]
    ax.barh(y, values, height=.48,
            color=["#168a80" if v < 0 else "#c06a51" for v in values],
            label="Original matched schedule")
    for i, task in enumerate(TASKS):
        if task in fresh:
            ax.plot(fnum(fresh[task], "relative_mse_change_percent"), i,
                    marker="D", color="#1b3f6b", markersize=5,
                    label="Fresh fitting seeds" if i == 0 else None)
    ax.axvline(0, color="#71808a", linewidth=.8)
    ax.set_yticks(y, [LABELS[s] for s in TASKS])
    ax.invert_yaxis()
    ax.set_xlabel("Relative change in mean test MSE (%) - negative is better")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(pad=.9)
    path = OUT / "scaling_effects.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def page_frame(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd8df"))
    canvas.line(45, height - 39, width - 45, height - 39)
    canvas.setFont("DejaVu", 7.4)
    canvas.setFillColor(GRAY)
    canvas.drawString(45, height - 30, "RESEARCH NOTE  /  KAHM INPUT SCALING")
    canvas.drawRightString(width - 45, height - 30, "29 October 2026")
    canvas.drawString(45, 27, "Paired preprocessing investigation | original targets and evaluation units preserved")
    canvas.drawRightString(width - 45, 27, str(doc.page))
    canvas.restoreState()


def main():
    global ROOT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--decision", type=Path, help="Optional disposition/interpretation JSON")
    args = parser.parse_args()
    ROOT, OUT = args.study_root, args.output
    OUT.mkdir(parents=True, exist_ok=True)
    original = rows(ROOT / "analysis_original/taskwise_comparison.csv")
    fresh = rows(ROOT / "analysis_confirmation/taskwise_comparison.csv")
    horizons = rows(ROOT / "analysis_original/horizon_summary.csv")
    primary = task_rows(original)
    confirm = task_rows(fresh)
    if set(primary) != set(TASKS) or set(confirm) != set(TASKS):
        raise RuntimeError("A complete original and fresh comparison is required")
    decision = json.loads((args.decision or ROOT / "report_decision.json").read_text())
    audit = json.loads((ROOT / "analysis_original/audit.json").read_text())
    fresh_audit = json.loads((ROOT / "analysis_confirmation/audit.json").read_text())
    comparison_plot = plot_effects(primary, confirm)
    story = []

    # Page 1: decision and complete primary outcomes.
    story += [body("Does [-1, 1] input scaling improve KAHM forecasting?", "Title"),
              body(escape(decision["lead"]), "Lead"),
              body(escape(decision["action"])), Spacer(1, 8)]
    story.append(body("Primary KAHM block forecaster: original matched experiments", "Heading"))
    data = [["Task", "Standard MSE", "Min-max MSE", "MSE change", "Seeds improved"]]
    for s in TASKS:
        r = primary[s]
        data.append([LABELS[s], f"{fnum(r, 'standard_mean_mse'):.6f}",
                     f"{fnum(r, 'alternative_mean_mse'):.6f}",
                     pct(fnum(r, "relative_mse_change_percent")),
                     f"{r['n_seeds_improved']}/{r['n_fitting_seeds']}"])
    story += [table(data, [109, 99, 99, 99, 99]), Spacer(1, 6),
              body("MSE is averaged over the three horizons and all scheduled fitting seeds within each task. "
                   "Selection uses validation data separately within each arm. Negative changes indicate improvement. "
                   "PenDigits pools the two coordinates in MSE units and gives each eligible writer equal weight.", "Small"),
              Image(str(comparison_plot), width=500, height=194),
              body("The chart compares changes within each task; it is not a pooled physical error across different systems. "
                   "Fresh PenDigits fitting seeds reuse the original held-out writer cohort; the simulated tasks use new episodes.", "Small"),
              PageBreak()]

    # Page 2: intervention and experimental controls.
    story += [body("What was changed, and what was held fixed", "Title"),
              body("The original histories are already bounded by the task definitions. The manuscript's implementation "
                   "then standardizes every history coordinate using its training mean and standard deviation before "
                   "building geometric dictionaries. The intervention changes the input to KAHM itself; the future "
                   "forecast target, output clipping, and reported error units remain unchanged. [1, 2]"),
              table([["Arm", "Frozen transformation applied to KAHM inputs"],
                     ["Standard", "Original training mean / standard-deviation normalization."],
                     ["Min-max (primary)", "Training-only featurewise minimum and maximum map each nonconstant coordinate to [-1, 1]. Future values may extrapolate beyond this interval."],
                     ["Clipped min-max", "The same fitted map, followed by clipping to [-1, 1] for every query."],
                     ["Bounded raw", "Use the existing bounded observations directly, with no standardization."]], [120, 385]),
              Spacer(1, 9),
              body("The min-max formula", "Heading"),
              body("z_j = 2 (x_j - training_min_j) / (training_max_j - training_min_j) - 1", "Formula"),
              body("A coordinate constant in training is mapped to zero. Fitted extrema use all training histories, "
                   "before the original seeded geometry subsample. The same fitted transformation is used for training, "
                   "validation, calibration, test, and one-step shifted histories for Koopman fits. No extrema are fitted "
                   "to validation or test data."),
              body("Complete scope", "Heading"),
              table([["Campaign", "Original schedule per arm"],
                     ["Four bounded systems", "4 systems x 5 seeds; histories 4 and 12; horizons 1, 4, 12."],
                     ["Nonlinear oscillators", "Van der Pol and Duffing x 3 seeds; same histories and horizons."],
                     ["Budget sensitivity", "4 systems x 3 seeds x budgets 1, 4, 16; history 12."],
                     ["PenDigits", "2 coordinates x 3 seeds; histories 4 and 8; horizons 1, 2, 4; 13 eligible test writers."],
                     ["Fresh confirmation", "6 simulated systems x 3 new seeds (4701-4703), plus 2 PenDigits coordinates x those seeds; all four fixed arms. PenDigits reuses the test writers."]], [137, 368]),
              Spacer(1, 7),
              body("All original candidate models are retained, including the KAHM soft, hard, Frobenius, duplicate, "
                   "and rollout variants. RBF, affine-hull, autoregressive, polynomial, persistence, and constant controls "
                   "are unchanged. Dictionary sizes, local-point limits, sharpness, decoder ridge, coefficient budgets, "
                   "random streams, origins, and validation rules are preserved."),
              body("The analytic two-atom risk calibration and duplicate-invariance illustration do not learn KAHM "
                   "input geometry, so an input-scaling intervention does not apply to them."), PageBreak()]

    # Page 3: direct comparison in the same RMSE units as the manuscript.
    story += [body("Horizon-specific forecasting errors", "Title"),
              body("Validation-selected retained KAHM blocks. Entries are mean RMSE across fitting seeds, followed "
                   "by sample standard deviation in parentheses. These are descriptive test estimates, not prediction-risk certificates."),
              Spacer(1, 6)]
    data = [["Task / horizon", "Standard", "Min-max", "Clipped min-max", "Bounded raw"]]
    lookup = {(r["system"], r["mode"], r["horizon"]): r for r in horizons
              if r["family"] == "kahm_nested" and r["campaign"] in
              {"main_v1", "oscillators_v1", "pendigits_v1"}}
    for task in TASKS:
        for h in ([1, 2, 4] if task == "pendigits" else [1, 4, 12]):
            values = []
            for mode in ["standard", "minmax", "minmax_clip", "raw"]:
                r = lookup[(task, mode, str(h))]
                values.append(f"{fnum(r, 'mean_test_rmse'):.4f} ({fnum(r, 'sd_test_rmse_over_fitting_seeds'):.4f})")
            data.append([f"{LABELS[task]} / {h}", *values])
    story += [table(data, [113, 98, 98, 98, 98]), Spacer(1, 8),
              body("Synthetic task horizons are observations ahead. For Van der Pol and Duffing, horizons 1, 4, "
                   "and 12 correspond to 0.1, 0.4, and 1.2 time units. PenDigits horizons 1, 2, and 4 correspond "
                   "to 0.1, 0.2, and 0.4 seconds under the dataset's documented sampling interval. [1]", "Small"),
              body("PenDigits uses the same held-out writer cohort across fitting seeds. Its between-seed standard "
                   "deviation measures fitting variability, not uncertainty over independently sampled test cohorts.", "Small"),
              PageBreak()]

    # Page 4: confirmation.
    story += [body("Fresh-seed confirmation", "Title"),
              body("The original matched comparison reuses the manuscript's recorded random streams. To check "
                   "whether the result survives new data and new fits, a separate campaign froze all four arms and "
                   "three unused fitting seeds before generating its evaluation data. The simulator definitions, "
                   "candidate models, and sample sizes were unchanged. A separately frozen PenDigits follow-up "
                   "uses the same new fitting seeds to draw new training-stroke subsamples, with the original validation "
                   "and held-out writers retained."),
              body(escape(decision["confirmation_interpretation"])), Spacer(1, 6)]
    data = [["Task", "Standard MSE", "Min-max MSE", "MSE change", "Seeds improved"]]
    for s in TASKS:
        r = confirm[s]
        data.append([LABELS[s], f"{fnum(r, 'standard_mean_mse'):.6f}",
                     f"{fnum(r, 'alternative_mean_mse'):.6f}",
                     pct(fnum(r, "relative_mse_change_percent")),
                     f"{r['n_seeds_improved']}/{r['n_fitting_seeds']}"])
    story += [table(data, [109, 99, 99, 99, 99]), Spacer(1, 11),
              body("Prespecified alternatives on the new seeds", "Heading")]
    data = [["Task", "Min-max", "Clipped min-max", "Bounded raw"]]
    alternatives = {m: task_rows(fresh, m) for m in ["minmax", "minmax_clip", "raw"]}
    for s in TASKS:
        data.append([LABELS[s], *[pct(fnum(alternatives[m][s], "relative_mse_change_percent"))
                                  for m in ["minmax", "minmax_clip", "raw"]]])
    story += [table(data, [151, 118, 118, 118]), Spacer(1, 6),
              body("All entries are relative changes in mean all-horizon MSE versus the same standard arm. "
                   "These alternatives were fixed before confirmation; no test-set search for a preferred scaling "
                   "variant was performed. PenDigits confirmation concerns fitting variability on the same writer cohort, "
                   "not generalization to a second cohort.", "Small"),
              body("How uncertainty is represented", "Heading"),
              body("Every fitting-seed effect and its sign are retained in the evidence tables. Paired episode "
                   "differences quantify Monte Carlo uncertainty conditional on each fitted model. Those conditional "
                   "intervals do not account for the randomness of learning a new model. The separate between-seed "
                   "variation must therefore remain visible. Three confirmation fits per task give useful replication "
                   "evidence, but cannot establish that an improvement occurs on every future fit or dataset."),
              body("The manuscript's finite-sample risk endpoints are recomputed for each arm's frozen family. "
                   "Their original failure allocations are not one simultaneous guarantee over all scaling arms, "
                   "tasks, and repetitions. Comparative test-error summaries are kept distinct from those certificates."),
              PageBreak()]

    # Page 5: sensitivity and ablations, including all regressions.
    story += [body("Sensitivity and model-family differences", "Title"),
              body("The original budget study uses three shared fitting seeds (2701-2703) and history length 12. "
                   "The following table retains every budget and system. Each entry compares the validation-selected "
                   "KAHM block model with the standard arm at the same budget."), Spacer(1, 6)]
    data = [["Task", "Budget", "Standard MSE", "Min-max change", "Clipped change", "Raw change"]]
    for task in TASKS[:4]:
        for budget in [1, 4, 16]:
            campaign = f"sensitivity_v1_k{budget}"
            rs = {r["mode"]: r for r in original if r["campaign"] == campaign
                  and r["system"] == task and r["family"] == "kahm_nested" and r["horizon"] == "all"}
            data.append([LABELS[task], budget,
                         f"{fnum(rs['minmax'], 'standard_mean_mse'):.6f}",
                         *[pct(fnum(rs[m], "relative_mse_change_percent"))
                           for m in ["minmax", "minmax_clip", "raw"]]])
    story += [table(data, [100, 49, 89, 89, 89, 89]), Spacer(1, 10),
              body("Other KAHM variants: original main, oscillator, and PenDigits schedules", "Heading"),
              body("The table reports how many of the seven tasks improve in mean all-horizon MSE for each "
                   "KAHM family. It prevents a benefit for soft coordinates from being generalized to hard assignment "
                   "or every decoder."), Spacer(1, 4)]
    data = [["KAHM family", "Min-max", "Clipped min-max", "Bounded raw"]]
    for fam, label in FAMILIES.items():
        vals = []
        for mode in ["minmax", "minmax_clip", "raw"]:
            rr = task_rows(original, mode, fam)
            vals.append(f"{sum(fnum(r, 'mse_difference_alternative_minus_standard') < 0 for r in rr.values())}/{len(rr)} tasks")
        data.append([label, *vals])
    story += [table(data, [178, 109, 109, 109]), Spacer(1, 8),
              body(escape(decision["limitations_by_task"])), PageBreak()]

    # Page 6: mechanism, validation, provenance and interpretation.
    story += [body("Why scaling matters, and how the results were checked", "Title"),
              body("Scale changes the fitted KAHM, even when covariance normalization preserves the Gaussian "
                   "Gram matrix. The inherited local solver initializes its regularization using the mean squared "
                   "magnitude of its reconstruction inputs and updates it using reconstruction error. The folding "
                   "score also contains 1 - exp(-Euclidean reconstruction distance), plus an origin-dependent angular "
                   "term. Changing magnitude or centering can therefore change both the fitted reconstruction and "
                   "the soft state weights. [3, 4]"),
              body("This explains why a scale intervention is plausible; it does not prove that [-1, 1] is optimal. "
                   "A separate small mechanism check produced identical Gaussian Gram matrices under uniform "
                   "rescaling but different regularization and folding values. Forecasting conclusions come from "
                   "the complete paired experiments, not that illustrative check."),
              body("Verification", "Heading"),
              body(escape(decision["verification_summary"])),
              body("The reference source commit is <font name='DejaVuMono' size='8'>" + SHA + "</font>. "
                   "The prepared PenDigits data exactly match the original recorded SHA-256. Every run retains "
                   "its configuration, source and intervention hashes, all candidate summaries, frozen validation "
                   "decisions, and full double-precision loss arrays. No failed attempt is silently removed."),
              body("Repository and manuscript disposition", "Heading"),
              body(escape(decision["repository_disposition"])),
              body("Interpretive limits", "Heading"),
              body("This is a follow-up to an already available draft. The original-stream replay is a matched "
                   "descriptive comparison, and the new simulation seeds provide fresh confirmation. PenDigits "
                   "remains a fixed, writer-disjoint cohort of 13 eligible writers; its documentation does not establish "
                   "independent random sampling of a writer population. Training-seed repetitions do not create new "
                   "writer cohorts. All performance claims concern the supplied tasks and fixed hyperparameters; "
                   "stronger autoregressive or polynomial comparators can remain more accurate."),
              body("Sources and reproducibility references", "Heading")]
    references = [
        ("1", "Supplied research draft, 28 October 2026: methods, experiment protocols, real-data evaluation, and retained tables.", None),
        ("2", "Original geometric feature preprocessing and retained dictionaries.", f"{BASE}/blob/{SHA}/extensions/certified_forecasting/src/certified_forecasting/features.py"),
        ("3", "Inherited KAHM regularization implementation.", f"{BASE}/blob/{SHA}/kernel_regularized_least_squares.py"),
        ("4", "Inherited KAHM reconstruction and folding score.", f"{BASE}/blob/{SHA}/autoencoder_filtering_extended.py"),
        ("5", "Original forecasting configurations, runners, and reproducibility documentation.", f"{BASE}/tree/{SHA}/extensions/certified_forecasting"),
        ("6", "Alpaydin and Alimoglu, Pen-Based Recognition of Handwritten Digits, UCI Machine Learning Repository (1996), DOI 10.24432/C5MG6K.", "https://doi.org/10.24432/C5MG6K"),
        ("7", "Complete comparison tables, frozen protocols, audit results, and reproduction scripts accompany this report.", None),
    ]
    for n, label, url in references:
        value = f"[{n}] {escape(label)}"
        if url:
            value += f" <link href='{escape(url, quote=True)}' color='#087f8c'>Source</link>"
        story.append(body(value, "Small"))
    path = OUT / "KAHM_scaling_investigation.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=45, leftMargin=45,
                            topMargin=52, bottomMargin=43,
                            title="KAHM input scaling: paired replication and fresh confirmation",
                            author="Research investigation prepared for the manuscript authors")
    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    print(path)


FONTROOT = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DejaVu", str(FONTROOT / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuBold", str(FONTROOT / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONTROOT / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVuBold")
STYLES = {
    "Title": ParagraphStyle("Title", fontName="DejaVuBold", fontSize=19, leading=24,
                             textColor=NAVY, spaceAfter=12),
    "Lead": ParagraphStyle("Lead", fontName="DejaVuBold", fontSize=10.1, leading=14.5,
                            textColor=NAVY, spaceAfter=8),
    "Heading": ParagraphStyle("Heading", fontName="DejaVuBold", fontSize=11.2, leading=15,
                               textColor=TEAL, spaceBefore=5, spaceAfter=7, keepWithNext=True),
    "Body": ParagraphStyle("Body", fontName="DejaVu", fontSize=9.2, leading=13.2,
                            textColor=NAVY, spaceAfter=8),
    "Small": ParagraphStyle("Small", fontName="DejaVu", fontSize=7.7, leading=10.5,
                             textColor=GRAY, spaceAfter=6),
    "Formula": ParagraphStyle("Formula", fontName="DejaVuMono", fontSize=8.2, leading=12,
                               textColor=NAVY, spaceAfter=8, leftIndent=8),
    "TableBody": ParagraphStyle("TableBody", fontName="DejaVu", fontSize=8.0, leading=10.5,
                                 textColor=NAVY),
    "TableHead": ParagraphStyle("TableHead", fontName="DejaVuBold", fontSize=8.1, leading=10.5,
                                 textColor=colors.white),
}


if __name__ == "__main__":
    main()
