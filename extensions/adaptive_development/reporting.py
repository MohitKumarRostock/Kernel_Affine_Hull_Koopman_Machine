"""Post-run presentation only; does not fit models or change frozen decisions.

Reads the frozen analyzer's outputs and report_content.json. Generates the
standalone PDF and scientific figures used in the development report.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kahm_adaptive_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                               Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
METHODS = ["ridge", "fixed_kahm", "adaptive_kahm", "adaptive_affine", "adaptive_rbf", "kernel_rr"]
LABELS = {"ridge":"Raw ridge", "fixed_kahm":"Fixed KAHM", "adaptive_kahm":"Adaptive KAHM",
          "adaptive_affine":"Adaptive affine", "adaptive_rbf":"Adaptive RBF-local",
          "kernel_rr":"Sliding-window KRR"}
COLLECTIONS = ["synthetic_stationary", "synthetic_recurring", "dominick_weekly", "sunspot_daily", "natural_equal_collection"]
COLLECTION_LABELS = ["Stationary", "Recurring", "Dominick", "Sunspot", "Natural mean"]


def rows(name):
    with (HERE/"analysis"/name).open(newline="") as f:
        return list(csv.DictReader(f))


def figures():
    out=HERE/"analysis"/"figures"
    out.mkdir(exist_ok=True)
    collection_rows=rows("collection_metrics.csv")
    lookup={(r["budget"],r["collection"],r["method"]):float(r["mean_scaled_mse"]) for r in collection_rows}
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False,
                        "axes.spines.right":False, "axes.titleweight":"bold", "savefig.dpi":180})
    fig,axes=plt.subplots(1,3,figsize=(7.3,3.0),sharey=True)
    for ax,collection,title in zip(axes,["synthetic_recurring","dominick_weekly","sunspot_daily"],
                                  ["Recurring regimes", "Dominick", "Sunspot"]):
        for offset,budget,color,label in [(-.16,"small","#245b89","Small: 64 / 128"),(.16,"large","#c37b29","Large: 128 / 256")]:
            v=[lookup[(budget,collection,m)] for m in METHODS]
            ax.scatter(v,np.arange(len(METHODS))+offset,s=36,color=color,label=label,zorder=3)
        if collection=="synthetic_recurring":
            ax.set_xscale("log")
        else:
            ax.set_xlim(left=0)
        ax.set_title(title,pad=10)
        ax.grid(axis="x",color="#e0e5e9",lw=.7)
        ax.set_xlabel("Scaled MSE"+(" (log scale)" if collection=="synthetic_recurring" else ""))
        ax.set_yticks(np.arange(len(METHODS)),[LABELS[m] for m in METHODS])
    axes[0].invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=8,loc="lower center",
               bbox_to_anchor=(.60,.015),ncol=2)
    fig.subplots_adjust(left=.23,right=.985,bottom=.32,wspace=.25,top=.85)
    fig.savefig(out/"collection_errors.png",facecolor="white")
    fig.savefig(out/"collection_errors.pdf",facecolor="white")
    plt.close(fig)

    task=rows("task_metrics.csv")
    recurring=[r for r in task if r["collection"]=="synthetic_recurring"]
    seeds=sorted({r["task_id"] for r in recurring})
    loss={(r["budget"],r["task_id"],r["method"]):float(r["scaled_mse"]) for r in recurring}
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.2),sharey=True)
    palette=["#73818e","#9d77a9","#b33133","#318c82","#b6802c","#245b89"]
    for ax,budget in zip(axes,["small","large"]):
        for method,color in zip(METHODS,palette):
            ax.plot(range(1,5),[loss[(budget,s,method)] for s in seeds],marker="o",lw=1.4,
                    ms=4,label=LABELS[method],color=color)
        ax.set_title(budget.capitalize()+" capacity tier")
        ax.set_xticks(range(1,5),[s.rsplit("__",1)[1][-2:] for s in seeds])
        ax.set_xlabel("Synthetic seed suffix (202611xx)")
        ax.grid(axis="y",color="#e0e5e9",lw=.7)
    axes[0].set_ylabel("Mean scaled MSE (lower is better)")
    if max(loss.values())/min(loss.values())>50:
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Mean scaled MSE (log scale)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.29,wspace=.15)
    fig.savefig(out/"recurring_seed_results.png",facecolor="white")
    fig.savefig(out/"recurring_seed_results.pdf",facecolor="white")
    plt.close(fig)
    return out


def create_report(output):
    figdir=figures()
    content=json.loads((HERE/"report_content.json").read_text())
    font_root=Path("/usr/share/fonts/truetype/dejavu")
    if (font_root/"DejaVuSans.ttf").exists():
        pdfmetrics.registerFont(TTFont("StudySans",str(font_root/"DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("StudySans-Bold",str(font_root/"DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("StudySans",normal="StudySans",bold="StudySans-Bold",italic="StudySans",boldItalic="StudySans-Bold")
        font,bold="StudySans","StudySans-Bold"
    else:
        font,bold="Helvetica","Helvetica-Bold"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="StudyTitle",fontName=bold,fontSize=24,leading=28,textColor=colors.HexColor("#163a56"),spaceAfter=14))
    styles.add(ParagraphStyle(name="StudyH1",fontName=bold,fontSize=17,leading=21,textColor=colors.HexColor("#163a56"),spaceAfter=12))
    styles.add(ParagraphStyle(name="StudyH2",fontName=bold,fontSize=11,leading=15,textColor=colors.HexColor("#163a56"),spaceBefore=9,spaceAfter=6))
    styles.add(ParagraphStyle(name="StudyBody",fontName=font,fontSize=9.3,leading=13.6,spaceAfter=8))
    styles.add(ParagraphStyle(name="StudySmall",fontName=font,fontSize=7.8,leading=10.5,spaceAfter=6,textColor=colors.HexColor("#4b5864")))
    styles.add(ParagraphStyle(name="StudyCell",fontName=font,fontSize=7.6,leading=10,alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="StudyHeader",fontName=bold,fontSize=7.6,leading=10,textColor=colors.white))
    width=524
    story=[]
    def p(text,style="StudyBody"):
        return Paragraph(text,styles[style])
    def table(data,widths=None):
        cells=[[p(str(x),"StudyHeader" if i==0 else "StudyCell") for x in row] for i,row in enumerate(data)]
        t=Table(cells,colWidths=widths or [width/len(data[0])]*len(data[0]),hAlign="LEFT",repeatRows=1)
        spec=[("BACKGROUND",(0,0),(-1,0),colors.HexColor("#204c6b")),("VALIGN",(0,0),(-1,-1),"TOP"),
              ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
              ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
              ("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#204c6b"))]
        for i in range(1,len(data)):
            if "Adaptive KAHM" in str(data[i][0]):
                spec.append(("BACKGROUND",(0,i),(-1,i),colors.HexColor("#e3eef5")))
            elif i%2==0:
                spec.append(("BACKGROUND",(0,i),(-1,i),colors.HexColor("#f1f4f6")))
        t.setStyle(TableStyle(spec))
        return t
    cm=rows("collection_metrics.csv")
    cm_lookup={(r["budget"],r["collection"],r["method"]):float(r["mean_scaled_mse"]) for r in cm}
    tr=rows("task_metrics.csv")
    for page_index,page in enumerate(content["pages"]):
        if page_index:
            story.append(PageBreak())
        story.append(p(page["title"],"StudyTitle" if page_index==0 else "StudyH1"))
        for block in page["blocks"]:
            kind=block["type"]
            if kind in ("p","small","h2"):
                story.append(p(block["text"],{"p":"StudyBody","small":"StudySmall","h2":"StudyH2"}[kind]))
            elif kind=="table":
                story.append(table(block["data"],block.get("widths")))
                story.append(Spacer(1,8))
            elif kind=="accuracy":
                budget=block["budget"]
                best={c:min(cm_lookup[(budget,c,m)] for m in METHODS) for c in COLLECTIONS}
                data=[["Method"]+COLLECTION_LABELS]
                for m in METHODS:
                    vals=[]
                    for c in COLLECTIONS:
                        x=cm_lookup[(budget,c,m)]
                        s=f"{x:.3e}" if abs(x)>=10000 else f"{x:.4f}"
                        vals.append("<b>"+s+"</b>" if np.isclose(x,best[c],rtol=1e-12,atol=1e-14) else s)
                    data.append([LABELS[m]]+vals)
                story.append(table(data,[144,76,76,76,76,76]))
                story.append(Spacer(1,9))
            elif kind=="resources":
                data=[["Method / tier","Worst p99<br/>(ms)","Maximum<br/>(ms)","Owned NumPy<br/>(KiB)","Worker RSS<br/>(MiB)"]]
                for m in METHODS:
                    for b in ("small","large"):
                        subset=[r for r in tr if r["method"]==m and r["budget"]==b]
                        data.append([LABELS[m]+" / "+b,
                                     f"{max(float(r['p99_service_ms']) for r in subset):.2f}",
                                     f"{max(float(r['max_service_ms']) for r in subset):.2f}",
                                     f"{max(float(r['max_observed_owned_numpy_bytes']) for r in subset)/1024:.1f}",
                                     f"{max(float(r['process_peak_rss_kib']) for r in subset)/1024:.1f}"])
                story.append(table(data,[204,70,70,90,90]))
                story.append(Spacer(1,9))
            elif kind=="figure":
                path=figdir/block["file"]
                iw,ih=ImageReader(str(path)).getSize()
                w=block.get("width",width)
                story.append(Image(str(path),width=w,height=w*ih/iw,hAlign="LEFT"))
                if block.get("caption"):
                    story.append(p(block["caption"],"StudySmall"))
            elif kind=="space":
                story.append(Spacer(1,block.get("height",8)))
            else:
                raise ValueError(kind)
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#ccd5dc"))
        canvas.line(44,40,568,40)
        canvas.setFont(font,7)
        canvas.setFillColor(colors.HexColor("#53616d"))
        canvas.drawString(44,28,"KAHM | Focused development study | 31 October 2026")
        canvas.drawRightString(568,28,str(doc.page))
        canvas.restoreState()
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=(612,792),leftMargin=44,rightMargin=44,topMargin=44,bottomMargin=54,
                          title="KAHM: focused development study and manuscript decision",
                          author="Development study prepared for Mohit Kumar",pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({"report":str(output),"bytes":output.stat().st_size,"figures":str(figdir)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    create_report(args.output)
