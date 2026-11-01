#!/usr/bin/env python3
"""Render the joint report strictly from the two independently audited studies.

This reporting-only program does not fit, select, calibrate or change a model.
It refuses absent/unverified metrics and never renders placeholder results.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kahm-report-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                               PageBreak, PageTemplate, Paragraph, Spacer,
                               Table, TableStyle)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
SOURCE_COMMIT = "b1915813e98a59572a4bbb800adc2bd1f3fccc1b"
SYNTHETIC_SOURCE_COMMIT = "e41b6e4fb902e49c8438f1adf2a3e98cb429cddd"
SOURCE_BRANCH = "research/kahm-policy-augmentation-2026-11-01"
M4_COMMIT = "6c1067e5a57161249b17289a565178dc7a3fb3ca"
REPO = "https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine"
POLICIES = ("P0", "P1", "P0plus")
LABELS = {"P0": "P0", "P1": "P1", "P0plus": "P0+"}
NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#087F8C")
ORANGE = colors.HexColor("#C56B31")
INK = colors.HexColor("#222B35")
MUTED = colors.HexColor("#52606D")
PALE = colors.HexColor("#EFF4F7")
LINE = colors.HexColor("#CCD8DF")
WIDTH, HEIGHT = A4
MARGIN = 46
CONTENT_WIDTH = WIDTH - 2*MARGIN


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, digits=6):
    value = float(value)
    require(math.isfinite(value), "Nonfinite report value")
    if value and (abs(value) < 10**(-digits) or abs(value) >= 1e6):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def percent(value, digits=2, signed=True):
    if 0 < abs(float(value)) < 0.01:
        digits = max(digits, 4)
    return (f"{float(value):+.{digits}f}%" if signed else f"{float(value):.{digits}f}%")


def gain(candidate, comparator):
    return 100*(1-float(candidate)/float(comparator))


def result_phrase(g, digits=2):
    return f"{abs(g):.{digits}f}% {'lower' if g > 0 else 'higher' if g < 0 else 'equal'}"


def one(rows, **fields):
    chosen = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    require(len(chosen) == 1, "Expected one reporting row: " + repr(fields))
    return chosen[0]


def optional_one(rows, **fields):
    chosen = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    require(len(chosen) <= 1, "Ambiguous reporting row: " + repr(fields))
    return chosen[0] if chosen else None


def register_fonts():
    folder = Path("/usr/share/fonts/truetype/dejavu")
    fonts = {"Text": "DejaVuSerif.ttf", "TextBold": "DejaVuSerif-Bold.ttf",
             "TextItalic": "DejaVuSerif.ttf", "Sans": "DejaVuSans.ttf",
             "SansBold": "DejaVuSans-Bold.ttf", "Mono": "DejaVuSansMono.ttf"}
    for name, filename in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(folder/filename)))
    pdfmetrics.registerFontFamily("Text", normal="Text", bold="TextBold", italic="TextItalic", boldItalic="TextBold")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansBold", italic="Sans", boldItalic="SansBold")


def styles():
    return {
        "title": ParagraphStyle("Title", fontName="SansBold", fontSize=23, leading=28, textColor=NAVY, spaceAfter=8),
        "subtitle": ParagraphStyle("Subtitle", fontName="Sans", fontSize=12.0, leading=16.5, textColor=MUTED, spaceAfter=13),
        "section": ParagraphStyle("Section", fontName="SansBold", fontSize=17.5, leading=22, textColor=NAVY, spaceAfter=12),
        "h2": ParagraphStyle("H2", fontName="SansBold", fontSize=11.3, leading=15, textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("Body", fontName="Text", fontSize=9.8, leading=14.0, textColor=INK, spaceAfter=8.2),
        "small": ParagraphStyle("Small", fontName="Text", fontSize=8.8, leading=12.3, textColor=INK, spaceAfter=6.5),
        "caption": ParagraphStyle("Caption", fontName="Sans", fontSize=8.2, leading=11.3, textColor=MUTED, spaceAfter=10),
        "table": ParagraphStyle("TableCell", fontName="Sans", fontSize=8.4, leading=11.3, textColor=INK),
        "table_small": ParagraphStyle("TableCellSmall", fontName="Sans", fontSize=7.8, leading=10.5, textColor=INK),
        "th": ParagraphStyle("TableHeading", fontName="SansBold", fontSize=8.3, leading=11.0, textColor=colors.white),
        "callout": ParagraphStyle("Callout", fontName="SansBold", fontSize=11.2, leading=15.4, textColor=NAVY),
        "mono": ParagraphStyle("Mono", fontName="Mono", fontSize=8.0, leading=11.2, textColor=MUTED, spaceAfter=6),
        "reference": ParagraphStyle("Reference", fontName="Sans", fontSize=8.2, leading=11.5, textColor=INK, spaceAfter=8),
    }


class Report:
    def __init__(self, metrics, output):
        self.m = metrics
        self.s = metrics["synthetic"]
        self.n = metrics["natural"]
        self.output = Path(output)
        self.styles = styles()
        self.story = []
        self.page_titles = []

    def p(self, text, style="body"):
        return Paragraph(text, self.styles[style])

    def add(self, text, style="body"):
        self.story.append(self.p(text, style))

    def section(self, title, first=False):
        if self.story:
            self.story.append(PageBreak())
        self.page_titles.append(title)
        if not first:
            self.add(title, "section")

    def h2(self, text):
        self.add(text, "h2")

    def table(self, headers, rows, widths, small=False):
        style = "table_small" if small else "table"
        data = [[self.p(str(v), "th") for v in headers]]
        data += [[self.p(str(v), style) for v in row] for row in rows]
        widths = [w*CONTENT_WIDTH/sum(widths) for w in widths]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), NAVY),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 7),
            ("RIGHTPADDING", (0,0), (-1,-1), 7),
            ("TOPPADDING", (0,0), (-1,0), 8),
            ("BOTTOMPADDING", (0,0), (-1,0), 8),
            ("TOPPADDING", (0,1), (-1,-1), 5),
            ("BOTTOMPADDING", (0,1), (-1,-1), 5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE]),
            ("LINEBELOW", (0,-1), (-1,-1), .55, LINE),
        ]))
        self.story.extend([t, Spacer(1,9)])

    def box(self, text):
        t = Table([[self.p(text,"callout")]], colWidths=[CONTENT_WIDTH])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE),
                               ("BOX",(0,0),(-1,-1),.55,LINE),
                               ("LEFTPADDING",(0,0),(-1,-1),13),
                               ("RIGHTPADDING",(0,0),(-1,-1),13),
                               ("TOPPADDING",(0,0),(-1,-1),11),
                               ("BOTTOMPADDING",(0,0),(-1,-1),11)]))
        self.story.extend([t, Spacer(1,10)])

    def syn_loss(self, panel, procedure):
        return one(self.s["panel_means"], panel=panel, procedure=procedure)["mean_smse"]

    def nat_loss(self, panel, procedure, metric="SMSE"):
        return one(self.n["frequency_panel_means"], panel=panel, procedure=procedure)[metric]

    def cap_record(self, cohort, procedure):
        return one(cohort["cap_scaler_counts"], subset="all_tasks", procedure=procedure)

    def draw_figure(self):
        directory = self.output.parent / "report_intermediates"
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / "policy_relative_smse.png"
        panels = [("Fresh simulations: nonlinear", [self.syn_loss("nonlinear", p) for p in POLICIES]),
                  ("Natural benchmark: overall", [self.nat_loss("Overall", p) for p in POLICIES])]
        plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.labelcolor":"#344452",
                             "text.color":"#223344", "axes.edgecolor":"#B9C6CF"})
        fig, axes = plt.subplots(1, 2, figsize=(7.15,2.65), layout="constrained")
        for ax, (title, values) in zip(axes, panels):
            indexed = np.asarray(values)/values[0]*100
            span = max(float(np.ptp(indexed)), 1.0)
            ax.set_xlim(min(indexed)-.40*span, max(indexed)+.60*span)
            ax.axvline(100, color="#A6B5BF", ls="--", lw=1.0, zorder=1)
            for j,(value,color) in enumerate(zip(indexed,["#17324D","#087F8C","#C56B31"])):
                ax.scatter([value],[2-j],s=52,c=color,zorder=3,edgecolors="white",linewidths=.6)
                ax.annotate(f"{value:.2f}",(value,2-j),xytext=(7,0),textcoords="offset points",
                            va="center",ha="left",fontsize=8.7,color=color)
            ax.set_yticks([2,1,0],["P0","P1: + KAHM","P0+: + RBF"])
            ax.set_ylim(-.5,2.65)
            ax.set_title(title,loc="left",fontweight="bold",fontsize=9.6,pad=9)
            ax.set_xlabel("SMSE relative to P0 = 100",fontsize=8.9,labelpad=7)
            ax.grid(axis="x",color="#E7EDF1",lw=.55)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(axis="y",length=0,pad=7)
            ax.tick_params(axis="x",labelsize=8)
        fig.savefig(filename,dpi=240,bbox_inches="tight",facecolor="white")
        fig.savefig(directory/"policy_relative_smse.pdf",bbox_inches="tight",facecolor="white")
        plt.close(fig)
        return filename

    def build_pages(self):
        s, n = self.s, self.n
        prim = s["primary_results"]
        nonlinear = [r for r in prim if r["panel"] == "nonlinear"]
        linear = [r for r in prim if r["panel"] == "linear"]
        nsup = sum(r["criterion_established"] for r in nonlinear)
        nlin = sum(r["criterion_established"] for r in linear)
        sg = gain(self.syn_loss("nonlinear","P1"),self.syn_loss("nonlinear","P0"))
        sgplus = gain(self.syn_loss("nonlinear","P1"),self.syn_loss("nonlinear","P0plus"))
        ng = gain(self.nat_loss("Overall","P1"),self.nat_loss("Overall","P0"))
        ngplus = gain(self.nat_loss("Overall","P1"),self.nat_loss("Overall","P0plus"))
        natural_selected = [r for r in n["roster_and_policy_choices"] if r["P1_selected_KAHM"]]
        headline = ("An incremental synthetic benefit with limited external transfer" if nsup == 2 else
                    "Partial synthetic support without a general forecasting advantage" if nsup == 1 else
                    "The prespecified policy advantage is not established")

        # Page 1: principal outcome and precise scope.
        self.section("Results and scope", first=True)
        self.add("KAHM as an optional<br/>forecasting component", "title")
        self.add("A fresh policy augmentation study with a matched search control<br/>Research report | 1 November 2026", "subtitle")
        self.box(headline)
        self.h2("Abstract")
        self.add(
            "We evaluate whether adding a bounded, centered in-sample KAHM residual correction improves a "
            "strong validation-selected forecasting policy. P0 is the established non-KAHM menu; P1 adds "
            "the KAHM correction; P0+ adds a raw-history Nystr\u00f6m RBF residual family with the same number "
            "of additional validation scores. All policies retain training-only feature construction, "
            "B-selected preprocessing and dimensions, and C-selected calibration and model family. "
            f"On 24 fresh simulation repetitions, P1 has {result_phrase(sg)} nonlinear standardized MSE "
            f"than P0 and {result_phrase(sgplus)} than P0+. The study establishes {nsup} of the two "
            f"prespecified 3% superiority criteria and {nlin} of the two 5% linear noninferiority criteria. "
            f"On 48 M4 series outside the recorded prior pilot, from two frequency panels, policy loss is {result_phrase(ng)} "
            f"than P0; KAHM is selected for {len(natural_selected)} series. Natural-series findings are "
            "descriptive benchmark-development evidence. They are analyzed separately from the simulations.")
        figure = self.draw_figure()
        self.story.append(Image(str(figure),width=CONTENT_WIDTH,height=CONTENT_WIDTH*2.65/7.15))
        self.add("Figure 1. Calibrated policy losses, each panel indexed to its own P0. Lower values indicate "
                 "lower loss. Points are descriptive; the four adjusted synthetic decisions appear in Table 3. "
                 "The panels use different evaluation designs and are not pooled.","caption")
        self.add("The experiment tests incremental value at the level of a complete selection procedure. "
                 "Improvement over one shared common forecast remains a separate, conditional mechanism result. "
                 "The presence of an optional KAHM candidate does not guarantee a smaller terminal loss.","small")

        # Page 2: scientific design and exact information roles.
        self.section("1. Policies, candidate selection and data roles")
        self.add("The scientific question concerns a deployable choice made before future outcomes are known. "
                 "P0 is selected from credible non-KAHM candidates. The two augmented menus differ by one "
                 "additional candidate family, so their comparison addresses the value of KAHM beyond an "
                 "equally sized increase in validation search opportunities. Selection itself can overfit a "
                 "finite development sample [3].")
        self.table(["Policy", "Additional family", "Simulation / natural family count"], [
            ["P0", "Established non-KAHM candidates", "10 / 11"],
            ["P1", "P0 + centered in-sample KAHM residual correction", "11 / 12"],
            ["P0+", "P0 + bounded raw-history Nystr\u00f6m RBF residual correction", "11 / 12"],
        ], [48,330,125])
        self.add("P0 includes ridge, raw Nystr\u00f6m RBF, histogram gradient boosting, persistence, common geometry, "
                 "bounded and unbounded local-RBF departures, common plus shared bias, and in-sample local-RBF "
                 "correction. The simulation menu also includes an episode cross-fitted RBF correction. The natural "
                 "menu omits that family and adds seasonal naive and drift forecasts; no independent episode "
                 "folds are invented from a single observed series.","small")
        self.table(["Role", "Information used", "Fresh simulations", "Natural series"], [
            ["A: fit", "Feature maps, transformations, coefficients; fixed loss scale", "64 episodes; 512 rows", "256-512 recent windows; realized minimum 291"],
            ["B: tune", "Penalties, cap/scaler; one shared residual bias", "32 episodes; 256 rows", "64 origins"],
            ["C: choose", "Correction multiplier and one complete family", "32 episodes; 256 rows", "64 origins"],
            ["D: evaluate", "Frozen predictions, then released targets", "128 episodes; 1,664 rows", "32 rolling origins"],
        ], [65,190,123,125], small=True)
        self.h2("Chronology and information available at a forecast origin")
        self.add("All horizons are {1, 4, 12} and the tabular history length is 12. Hyperparameters and coefficients "
                 "are not refitted after C. In the natural panel, twelve intervening origins ensure that all targets "
                 "from one block are mature before the next block's first origin; target observations are disjoint "
                 "between blocks. Later D contexts may include earlier D outcomes already observed at that origin [4].")
        self.add("The simulation cohort uses independent episode splits, with 13 retained candidate origins per "
                 "episode before row caps. The natural design uses one continuous record per series with overlapping "
                 "histories within each block. These different sources of dependence are respected in their "
                 "different analysis rules.","small")

        self.section("2. Residual construction and complete tuning grids")
        self.h2("A shared baseline and a centered correction")
        self.add("The common predictor g is fitted on all A rows and tuned on B. A single shared bias is the B "
                 "mean of y - g(x). The correction regresses the in-sample A residuals on bounded features centered "
                 "at their A mean. Its response-mean intercept is omitted. The forecast is g(x) + b + eta times "
                 "the centered feature slope; eta = 0 recovers g(x) + b exactly.","small")
        self.table(["Component", "Frozen settings and search", "Selection role"], [
            ["History transforms", "Standard, min-max, tanh(z); tanh multiplier 1", "Fit on A; choose on B"],
            ["New local corrections", "Native caps {1, 2, 4, 8, 12}; each of the three transforms", "One cap/scaler across horizons"],
            ["Legacy geometry", "Caps {1, 2, 4, 8, 12, 20}; common/departure transform pairs (s,s), (m,m), (t,t), (t,m), (m,t)", "B; cap 20 duplicates 12"],
            ["Ridge penalties", "{0.0001, 0.001, 0.01, 0.1, 1}", "B, separately per horizon"],
            ["Calibration", "{0, 0.25, 0.5, 1}; legacy blends toward ridge, new residual families toward common + bias", "C, separately per horizon"],
            ["Raw RBF", "64 landmarks; bandwidth factors {0.5, 1, 2}; three history transforms", "B; inherited complete head"],
            ["Extra raw-RBF residual", "64 landmarks; bandwidth factors {0.25, 0.5, 1, 2, 4}; three history transforms", "B; one new family on C"],
            ["Gradient boosting", "Learning rates {0.05, 0.1}; leaves {7, 15}; 100 iterations; L2 penalty 1", "B; per-horizon candidate"],
            ["Geometry reference bank", "Partitions {2, 4, 8}; at most 24 references per local model; bank fit cap 512 rows", "A only; settings fixed"],
        ], [108,285,110],small=True)
        self.add("Table 1. Complete selected grids for the inherited comparison and the new residual families. "
                 "s, m and t denote standard, min-max and tanh. Persistence, seasonal naive and drift have no "
                 "fitted search grid; the two direct natural controls are specified in Section 5.","caption")
        self.add("For every correction map, B chooses the ridge penalty separately per horizon and then chooses "
                 "one map using the mean normalized loss across horizons. KAHM and the extra raw-RBF family each "
                 "receive 15 feature configurations and 75 B scores per horizon, followed by one extra C family "
                 "slot. Family selection uses calibrated C loss. Ties prefer the first candidate in the frozen "
                 "order, with existing P0 candidates preceding an added family.","small")
        self.add("Min-max queries are not clipped. Tanh is applied to A-standardized histories with multiplier one "
                 "and no subsequent history standardization. Post-map feature scaling and radial bounding are "
                 "separate operations. Bandwidth factors multiply the median squared landmark distance. The "
                 "feature scale floor is 10<super>-6</super>, the target-variance floor is 10<super>-12</super>, "
                 "and the inherited relative normalizer guard is 10<super>-10</super>.","small")

        # Page 3: full confirmatory family.
        self.section("3. Fresh simulations: the four primary criteria")
        self.add("The cohort comprises linear AR, nonlinear AR, Van der Pol and Duffing laws, each at added-noise "
                 "levels 0 and 0.1, with 24 new paired repetitions: 192 tasks. This is a new cohort under the "
                 "previously studied laws, with master seed 2026110191. The independent inferential unit is the "
                 "repetition; origins, horizons and task cells within a repetition are not additional replications.")
        self.table(["Policy", "Nonlinear mean SMSE", "Linear mean SMSE"], [
            [LABELS[p], number(self.syn_loss("nonlinear",p)), number(self.syn_loss("linear",p))] for p in POLICIES
        ], [70,215,218])
        self.add("Table 2. Horizon losses are divided by the corresponding A-target population variance, floored "
                 "at 10<super>-12</super>. Horizons, noise levels and laws are averaged equally within each repetition; "
                 "reported values then average the 24 repetition summaries. Nonlinear and linear panels are separate.","caption")
        criterion_rows = []
        for row in prim:
            label = ("Nonlinear: " if row["panel"] == "nonlinear" else "Linear: ")+"P1 / "+LABELS[row["comparator"]]
            criterion_rows.append([label, row["margin"], percent(row["point_relative_gain_percent"]),
                                   number(row["simultaneous_one_sided_upper"],6),
                                   "Established" if row["criterion_established"] else "Not established"])
        self.table(["Comparison", "Required margin", "Point gain", "Adjusted upper bound", "Decision"],
                   criterion_rows, [130,104,72,102,95],small=True)
        self.add("Table 3. Positive point gains mean lower P1 loss. The upper bound concerns the margin-adjusted "
                 "loss difference, not the percentage gain. Every frozen criterion is reported.","caption")
        self.h2("Decision rule")
        self.add("For a nonlinear comparison, define the repetition difference as P1 loss minus 0.97 times "
                 "comparator loss. For linear noninferiority use P1 loss minus 1.05 times comparator loss. "
                 "The reported endpoint is the mean difference plus the 98.75th percentile of a Student-t "
                 "distribution with 23 degrees of freedom times its standard error. A criterion is established "
                 "only if this endpoint is strictly negative. "
                 f"The common critical value is {number(prim[0]['critical_t'],6)}. Bonferroni adjustment "
                 "covers exactly the four prespecified claims.")
        self.box(f"{s['criteria_established']} of 4 criteria established: {nsup}/2 nonlinear superiority "
                 f"criteria and {nlin}/2 linear noninferiority criteria.")
        self.add("Failure to establish a 3% advantage does not demonstrate equality or rule out a smaller effect. "
                 "Passing linear noninferiority addresses its declared loss margin and does not establish "
                 "nonlinear superiority. The Student-t calculations are approximate finite-repetition inference "
                 "under the stated simulation population; they are not distribution-free certificates.","small")

        # Page 4: full condition table and selected subgroups.
        self.section("4. Synthetic heterogeneity and optional selection")
        self.add("Condition summaries describe how the complete policies behave across the fixed laws and noise "
                 "levels. They are not additional confirmatory tests and are not used to change the menus, "
                 "dimension grid, calibration rule or primary interpretation.")
        cell_rows=[]
        law_names={"linear_ar":"Linear AR","nonlinear_ar":"Nonlinear AR","vanderpol":"Van der Pol","duffing":"Duffing"}
        details=s["task_selection_details"]
        for law in ("linear_ar","nonlinear_ar","vanderpol","duffing"):
            for noise in (0.0,0.1):
                vals={p:one(s["condition_means"],law=law,noise_level=noise,procedure=p)["mean_smse"] for p in POLICIES}
                count=sum(r["P1_selected_KAHM"] for r in details if r["law"]==law and r["noise_level"]==noise)
                cell_rows.append([law_names[law]+f"<br/>noise {noise:g}", number(vals["P0"],5),number(vals["P1"],5),
                                  number(vals["P0plus"],5),percent(gain(vals["P1"],vals["P0"])),f"{count}/24"])
        self.table(["Law / noise", "P0", "P1", "P0+", "P1 gain vs P0", "KAHM chosen"],
                   cell_rows,[112,75,75,75,91,75],small=True)
        self.add("Table 4. Mean SMSE in each law/noise cell, averaged over horizons and 24 repetitions. "
                 "KAHM choice counts refer to C selection by P1. The same task can have distinct cap, scaler "
                 "and penalty choices for its KAHM and non-KAHM candidates.","caption")
        self.h2("What changes when KAHM is selected?")
        for panel,label in (("nonlinear","Nonlinear panel"),("linear","Linear panel")):
            total = 144 if panel=="nonlinear" else 48
            row=optional_one(s["conditional_descriptive"],panel=panel,policy="P1",selected_identity="crossfit__insample_kahm")
            if row:
                count=row["task_count"];wins=row["strict_D_win_count"];ties=row["equal_D_count"]
                text=(f"<b>{label}.</b> KAHM is selected in {count}/{total} tasks. Within that C-defined subset, "
                      f"P1 wins against P0 on {wins} terminal task losses, ties on {ties}, and loses on {count-wins-ties}. "
                      f"Its mean loss is {result_phrase(gain(row['mean_D_smse'],row['mean_P0_D_smse']))} than P0. "
                      "The subset is descriptive and cannot replace the full-policy criterion.")
            else:
                text=f"<b>{label}.</b> KAHM is selected in 0/{total} tasks; P1 forwards the P0 choice throughout this panel."
            self.add(text,"small")
        sgbase=gain(self.syn_loss("nonlinear","in_sample_KAHM"),self.syn_loss("nonlinear","shared_common_plus_bias"))
        sglocal=gain(self.syn_loss("nonlinear","in_sample_KAHM"),self.syn_loss("nonlinear","in_sample_local_RBF"))
        self.add(f"As a standalone correction in the nonlinear panel, in-sample KAHM has {result_phrase(sgbase)} "
                 f"SMSE than its shared common-plus-bias baseline and {result_phrase(sglocal)} than the "
                 "independently selected local-RBF correction. These are descriptive mechanism comparisons. "
                 "A useful correction to a fixed baseline need not improve a stronger policy that can choose "
                 "among different complete forecasts.","small")

        # Page 5: natural benchmark, complete policies and every selected case.
        self.section("5. Natural series: external benchmark development")
        self.add("The observed-data panel contains 24 weekly and 24 hourly series selected by a fixed identity "
                 "hash and length eligibility from the pinned M4 Train files [2]. Seventy-two previously used "
                 "M4 IDs were excluded. Official M4 Test files were not downloaded. Each series contributes "
                 "32 rolling origins at horizons {1, 4, 12}; this is not the official M4 evaluation. Anonymous "
                 "series can share underlying sources, so no independent-series confidence interval or "
                 "population-level significance claim is made.")
        natural_rows=[]
        for policy in POLICIES:
            natural_rows.append([LABELS[policy],number(self.nat_loss("Weekly",policy),5),number(self.nat_loss("Hourly",policy),5),
                                 number(self.nat_loss("Overall",policy),5),number(self.nat_loss("Overall",policy,"frequency_MASE"),4)])
        self.table(["Policy","Weekly SMSE","Hourly SMSE","Overall SMSE","Frequency MASE"],
                   natural_rows,[48,112,112,112,119])
        self.add("Table 5. Each frequency mean weights its 24 series equally; the overall mean weights the two "
                 "frequencies equally. Frequency MASE uses A-only mean absolute differences at lag 1 for weekly "
                 "and 24 for hourly series. Seasonal MASE, reported in the metrics record, uses lags 52 and 24. "
                 "Denominators are floored at 10<super>-12</super> and are never fitted on D.","caption")
        self.add(f"P1 has {result_phrase(ng)} overall SMSE than P0 and {result_phrase(ngplus)} than P0+. "
                 f"KAHM is selected for {len(natural_selected)}/48 series. The remaining P1 forecasts use "
                 "the selected P0 family with unchanged parameters.")
        selected_rows=[]
        selection_index={r["task_id"]:r for r in n["task_selection_details"]}
        for row in natural_selected:
            sel=selection_index[row["task_id"]]["in_sample_KAHM"]
            a,b=row["policy_losses"]["P1"]["SMSE"],row["policy_losses"]["P0"]["SMSE"]
            selected_rows.append([row["series_id"]+"<br/>"+row["frequency"],sel["selected_cap"],
                                  {"standard":"Standard","minmax":"Min-max","tanh_z_1":"Tanh"}[sel["selected_scaler"]],
                                  number(b,5),number(a,5),percent(gain(a,b))])
        if selected_rows:
            self.table(["Selected series","Cap","Scaler","P0 SMSE","P1 SMSE","P1 gain"],selected_rows,
                       [100,40,85,90,90,98],small=True)
            self.add("Table 6. Every natural-series case in which P1 selects KAHM. These outcomes describe "
                     "a subset chosen on C; D losses do not define the subset or revise the selection rule.","caption")
        natbase=gain(self.nat_loss("Overall","in_sample_KAHM"),self.nat_loss("Overall","shared_common_plus_bias"))
        natlocal=gain(self.nat_loss("Overall","in_sample_KAHM"),self.nat_loss("Overall","in_sample_local_RBF"))
        self.h2("Conditional correction value and policy value")
        self.add(f"Across all 48 series, standalone KAHM correction has {result_phrase(natbase)} SMSE than the "
                 f"identical common-plus-bias baseline and {result_phrase(natlocal)} than local-RBF correction. "
                 "The policy comparison asks a broader question because P0 may instead select a raw method, "
                 "persistence, drift or seasonal naive. The fixed-baseline result therefore does not establish "
                 "that adding KAHM improves the available forecasting policy.","small")
        self.add("Seasonal naive accesses its declared lag from the observed history; drift uses the earliest "
                 "observation in A's retained history interval and the current observation. All forecasting "
                 "parameters remain fixed after C, while each origin receives the observations then available. "
                 "The short terminal period, two frequency panels and fixed finite menus limit external scope.","small")

        # Page 6: all requested cap and preprocessing accountability.
        self.section("6. Validation-selected dimensions and preprocessing")
        self.add("The experiment carries forward the corrected dimension policy and earlier validation-based "
                 "dimension studies. A requested cap c initializes each local model at min(c, 12, N<sub>c</sub>-1); "
                 "the inherited positive-variance and projected-range rules can reduce the realized rank. "
                 "The native default cap 20 is equivalent to cap 12 at ambient history dimension 12. The "
                 "legacy search retains this duplicate and resolves ties toward 12; the new corrections "
                 "search the five distinct caps.")
        caprows=[]
        for cohort,name in ((s,"Simulation"),(n,"Natural")):
            for proc,label in (("common","Common"),("in_sample_KAHM","KAHM"),("in_sample_local_RBF","Local RBF")):
                row=self.cap_record(cohort,proc)
                count=row["cap_counts"];sc=row["scaler_counts"]
                require(set(count)<=set(map(str,(1,2,4,8,12))),"Unexpected selected cap in report")
                caprows.append([name+"<br/>"+label,*[count.get(str(cap),0) for cap in (1,2,4,8,12)],
                                sc.get("standard",0),sc.get("minmax",0),sc.get("tanh_z_1",0)])
        self.table(["Cohort / head","1","2","4","8","12","Std","Min-max","Tanh"],caprows,
                   [119,40,40,40,40,40,56,66,62],small=True)
        self.add("Table 7. B-selected requested caps and transforms across all tasks: 192 simulation tasks "
                 "and 48 natural tasks per head. These are candidate settings even where P1 ultimately "
                 "chooses a different complete family. They are not a ranking of transforms on D.","caption")
        self.h2("Realized local ranks")
        for cohort,label in ((s,"Simulation"),(n,"Natural")):
            row=one(cohort["selected_local_rank_summary"],subset="all_tasks",procedure="in_sample_KAHM")
            counts_text=", ".join(f"{rank}: {count}" for rank,count in sorted(row["rank_counts"].items(),key=lambda kv:int(kv[0])))
            er=row["effective_rank"]
            self.add(f"<b>{label} KAHM.</b> {row['local_model_occurrences']:,} local-model occurrences in selected "
                     f"banks; realized rank {er['minimum']:g}-{er['maximum']:g}, mean {er['mean']:.2f}. "
                     f"Counts by realized rank (rank: occurrences): {counts_text}. "
                     f"Zero-rank fallbacks: {row['rank_zero_fallback_occurrences']}.","small")
        widthrows=[]
        for cohort,label in ((s,"Simulation"),(n,"Natural")):
            vals=[]
            for measure in ("KAHM_feature_width","local_RBF_feature_width","extra_RBF_feature_width"):
                v=one(cohort["feature_width_summary"],subset="all_tasks",measure=measure)["values"]
                vals.append(f"{v['minimum']:g}-{v['maximum']:g}<br/>mean {v['mean']:.1f}")
            widthrows.append([label,*vals])
        self.table(["Cohort","KAHM features","Local RBF features","Extra raw-RBF features"],widthrows,
                   [95,125,130,153],small=True)
        self.add("Table 8. Widths of the selected correction designs. The extra family includes raw history "
                 "plus up to 64 Nystr\u00f6m directions, followed by feature scaling and radial bounding. Actual "
                 "width and effective rank depend on the fitted data.","caption")
        self.add("No transform receives a presumption of superiority. The cap and scaler summaries describe "
                 "the frozen B-selection procedure; terminal D loss never selects a dimension.","small")

        # Page 7: verification and costs.
        self.section("7. Computation, verification and limitations")
        costrows=[]
        def cs(cohort,key,field="sum"):
            return cohort["resource_summary"][key][field]
        measures=[("Fit worker wall time, summed", "fit_wall_seconds", "sum", "seconds"),
                  ("Fit worker CPU time, summed", "fit_cpu_seconds", "sum", "seconds"),
                  ("Prediction worker wall time, summed", "prediction_wall_seconds", "sum", "seconds"),
                  ("Extra-RBF fit wall time, summed", "extra_fit_wall_seconds", "sum", "seconds"),
                  ("Largest individual worker RSS", "peak_worker_rss_MiB", "maximum", "MiB"),
                  ("Complete compressed fitting evidence", "state_gzip_bytes", "sum", "MiB")]
        for label,key,field,unit in measures:
            vals=[cs(cohort,key,field) for cohort in (s,n)]
            if key=="state_gzip_bytes": vals=[v/(1024**2) for v in vals]
            costrows.append([label,*[f"{v:,.2f} {unit}" for v in vals]])
        self.table(["Measurement","Simulations (192 tasks)","Natural (48 tasks)"],costrows,[235,134,134],small=True)
        self.add("Table 9. Durations sum individual worker measurements; concurrent execution means these "
                 "are not elapsed stage times. Peak worker RSS includes interpreter and retained process state. "
                 "The compressed fitting evidence stores candidates, diagnostics and audit material and is "
                 "not a minimal deployed model size. All costs include the declared rejected searches and "
                 "diagnostic procedures appropriate to their recorded scope.","caption")
        self.h2("Independent saved-record verification")
        sf,nf=s["numerical_recovery"],n["numerical_recovery"]
        self.add("Both independent audits passed. The simulation audit covers 192 tasks and 9,600 retained "
                 "forecast matrices; the natural audit covers 48 series and 2,016 matrices. Recorded hashes "
                 "link source/configuration freezes, fitted states, prediction seals and released targets. "
                 "The audits reconstruct loss denominators, B and C choices, complete policy forwarding and "
                 "reported aggregates. Manufactured tests independently checked ridge normal equations, "
                 "zero-correction recovery, query-batch invariance and exact in-sample adapter parity.")
        self.add(f"The inherited numerical policy first attempts the original ridge SVD and invokes a logged "
                 f"alternative driver only on a linear-algebra exception. Simulation fitting recorded "
                 f"{sf['fallback_count']} invocation(s), {sf['recovered_count']} recovered and {sf['unrecovered_count']} "
                 f"unrecovered; natural fitting recorded {nf['fallback_count']}, {nf['recovered_count']} and "
                 f"{nf['unrecovered_count']}, respectively. Requested dimensions, realized ranks and native "
                 "fallbacks are retained separately from these numerical-solver events.","small")
        self.h2("Limits of the evidence")
        self.add("The design was informed by earlier studies and was fixed before this cohort's outcomes; it "
                 "is not a public preregistration. Fresh simulation draws extend evidence within four known "
                 "laws. They do not establish performance over a population of real forecasting tasks. "
                 "Natural series were selected from available benchmark metadata, and unknown dependencies "
                 "between series prevent treating 48 cases as 48 independent source populations.","small")
        self.add("Equal added score counts do not match computation, feature width or statistical capacity. "
                 "The experimental suite shares fitted objects, so total timings do not identify every isolated "
                 "policy's marginal runtime. The ridge correction has a deterministic magnitude bound, but "
                 "that bound does not certify lower forecasting risk. Procedural target masking and saved-array "
                 "checks do not constitute cryptographic isolation or a complete independent reimplementation "
                 "of every feature map.","small")

        # Page 8: defensible claim and stable reproducibility references.
        self.section("8. Scientific interpretation and reproducibility")
        if nsup==2:
            claim=("Under the declared synthetic laws and finite menus, adding the specified KAHM family "
                   "improves the complete selection policy by the prespecified margin relative to both controls. "
                   "The natural benchmark limits transfer of that conclusion. KAHM's forecasting value remains "
                   "conditional on the data, geometry, incumbent predictor and selection procedure.")
        elif nsup==1:
            claim=("The primary synthetic evidence supports only one of the two required superiority "
                   "comparisons. This is partial support for the optional component, with a substantial "
                   "qualification from the external benchmark. The results support studying geometry-dependent "
                   "residual features without asserting a general forecasting advantage.")
        else:
            claim=("The strongest defensible position remains a geometry-dependent nonlinear feature family "
                   "with conditional value as a bounded, calibrated residual correction. The new experiment "
                   "does not establish the prespecified incremental advantage of adding that correction to "
                   "a strong forecasting policy. Fixed-common improvements and stable correction magnitudes "
                   "remain narrower findings with different estimands.")
        self.box(claim)
        self.add("The distinction matters for the manuscript's central argument. A representation may add "
                 "nonconstant directions beyond a shared coordinate basis, yet the available alternatives and "
                 "finite-sample selection error determine whether those directions improve the forecast actually "
                 "chosen. Retaining unsuccessful policy comparisons clarifies that boundary and supports a "
                 "more precise scientific contribution.")
        self.add("A justified continuation is a declared exploratory analysis of selection stability and residual "
                 "geometry using these completed records. Any diagnostic or revised acceptance rule developed "
                 "from this evidence requires another reserved evaluation. The current D outcomes must not be "
                 "reused as untouched confirmation for a policy designed after inspecting them.","small")
        self.h2("Reproducibility record")
        self.add("The simulation source was published before terminal D generation. The natural source was "
                 "frozen locally before numerical parsing and published after its evaluation. The combined source "
                 "commit below contains both study runners and configurations; it is not a public preregistration. "
                 "The retained extensions contain the protocols, grids, source manifests, sealed predictions, "
                 "audit programs and analysis tables. Reporting reads verified metrics without refitting a model "
                 "or changing a decision.","small")
        self.add("Branch: " + SOURCE_BRANCH,"mono")
        self.add("Combined source commit: " + SOURCE_COMMIT,"mono")
        self.add("M4 input snapshot commit: " + M4_COMMIT,"mono")
        self.add("Extensions: policy_augmentation_20261101 and<br/>policy_augmentation_natural_20261101", "mono")
        self.h2("References and data provenance")
        refs=[
            ("[1] KAHM forecasting research repository. Frozen scientific source and study protocols. ",
             REPO+"/tree/"+SOURCE_COMMIT, "Pinned study source"),
            ("[2] M4 Competition. M4-methods repository, pinned Train data and metadata snapshot. ",
             "https://github.com/Mcompetitions/M4-methods/tree/"+M4_COMMIT,"M4 source snapshot"),
            ("[3] Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent "
             "selection bias in performance evaluation. Journal of Machine Learning Research, 11, 2079-2107. ",
             "https://www.jmlr.org/papers/v11/cawley10a.html", "Article and paper"),
            ("[4] Hyndman, R. J. and Athanasopoulos, G. Forecasting: Principles and Practice, third edition. "
             "Chapter 5: The forecaster's toolbox; simple forecasting methods, accuracy evaluation and time-series cross-validation. ",
             "https://otexts.com/fpp3/toolbox.html", "Open textbook"),
        ]
        for text,url,label in refs:
            self.add(escape(text)+f'<link href="{escape(url)}" color="#087F8C">{escape(label)}</link>.',"reference")
        self.add("Evidence source: KAHM_Policy_Augmentation_Metrics.json. Every numerical statement in this "
                 "report is drawn from the verified record or from explicitly labeled arithmetic on its tables. "
                 "Complete input hashes are retained with that record.","caption")

    def render(self):
        self.build_pages()
        titles=self.page_titles
        def decorate(canvas,doc):
            canvas.saveState()
            canvas.setStrokeColor(LINE);canvas.setLineWidth(.6)
            canvas.line(MARGIN,HEIGHT-35,WIDTH-MARGIN,HEIGHT-35)
            canvas.setFont("SansBold",7.5);canvas.setFillColor(NAVY)
            canvas.drawString(MARGIN,HEIGHT-26,"KAHM  /  FORECASTING POLICY AUGMENTATION")
            canvas.setFont("Sans",7.5);canvas.setFillColor(MUTED)
            canvas.drawRightString(WIDTH-MARGIN,HEIGHT-26,"RESEARCH REPORT  |  1 NOVEMBER 2026")
            canvas.line(MARGIN,37,WIDTH-MARGIN,37)
            canvas.setFont("Sans",7.5)
            title=titles[doc.page-1] if doc.page<=len(titles) else "Report continuation"
            canvas.drawString(MARGIN,25,title)
            canvas.drawRightString(WIDTH-MARGIN,25,f"{doc.page} / 9")
            canvas.restoreState()
        self.output.parent.mkdir(parents=True,exist_ok=True)
        doc=BaseDocTemplate(str(self.output),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
                            topMargin=53,bottomMargin=49,title="KAHM as an optional forecasting component",
                            author="KAHM forecasting research",subject="Verified policy augmentation research report")
        frame=Frame(MARGIN,49,CONTENT_WIDTH,HEIGHT-102,leftPadding=0,rightPadding=0,bottomPadding=0,topPadding=0)
        doc.addPageTemplates(PageTemplate(id="report",frames=frame,onPage=decorate))
        doc.build(self.story)
        pdf=PdfReader(str(self.output))
        require(len(pdf.pages)==9,f"Report pagination changed: expected9, observed{len(pdf.pages)}")
        text="\n".join(page.extract_text() or "" for page in pdf.pages)
        for token in ("PLACEHOLDER","TODO","NaN","None","Report continuation"):
            require(token not in text,"Unresolved report token: "+token)
        return {"status":"created", "pages":len(pdf.pages), "bytes":self.output.stat().st_size,
                "pdf_sha256":digest(self.output),"output":str(self.output),"requires_visual_QA":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics",type=Path,default=WORKSPACE/"deliverables/KAHM_Policy_Augmentation_Metrics.json")
    parser.add_argument("--output",type=Path,default=WORKSPACE/"deliverables/KAHM_Policy_Augmentation_Research_Report.pdf")
    args=parser.parse_args()
    require(args.metrics.is_file(),"Verified metrics are not available; refusing placeholder rendering")
    metrics=json.loads(args.metrics.read_text())
    require(metrics.get("schema")=="kahm-verified-policy-report-metrics-v1" and
            metrics.get("status")=="verified_report_metrics","Metrics are not independently verified")
    for cohort in ("synthetic","natural"):
        require(metrics[cohort]["verification"]["status"]=="passed","Missing passing audit: "+cohort)
    require(metrics["synthetic"]["design"]["tasks"]==192 and metrics["natural"]["design"]["series_count"]==48,
            "Unexpected study sizes")
    register_fonts()
    result=Report(metrics,args.output.resolve()).render()
    result.update(metrics_sha256=digest(args.metrics),builder_sha256=digest(__file__))
    print(json.dumps(result))


if __name__=="__main__":
    main()
