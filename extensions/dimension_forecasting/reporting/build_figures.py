#!/usr/bin/env python3
"""Scientific figures from the complete, audited saved-forecast analysis."""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/kahm_step3_report_matplotlib")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def figures(root, output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/"analysis/analysis_summary.json").read_text())
    audit=json.loads((root/"audit/post_release/final_results_audit.json").read_text())
    assert summary["status"]=="completed" and audit["status"]=="passed"
    def table(name):
        path=root/"analysis"/(name+".csv")
        assert hashlib.sha256(path.read_bytes()).hexdigest()==summary["tables"][name+".csv"]["sha256"]
        return pd.read_csv(path)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":"#243746","text.color":"#243746"})
    repetitions=table("repetition_losses")
    subset=repetitions[(repetitions.panel=="nonlinear")&(repetitions.method=="kahm_departure")&(repetitions["mode"]=="calibrated")]
    names=["context_historical2","context_native20","context_selected"]
    matrix=np.array([subset[subset.view==name].sort_values("repetition").smse.to_numpy() for name in names]).T
    assert matrix.shape==(24,3)
    figure,axes=plt.subplots(1,2,figsize=(9.9,3.25),gridspec_kw={"width_ratios":[1.,1.25]},constrained_layout=True)
    for row in matrix: axes[0].plot(range(3),row,color="#d1d8dd",linewidth=.75,zorder=1)
    colors=["#697586","#ba5b25","#146b85"]
    axes[0].scatter(range(3),matrix.mean(axis=0),s=70,c=colors,zorder=3)
    axes[0].plot(range(3),matrix.mean(axis=0),color="#223f54",linewidth=1.7,zorder=2)
    axes[0].set_xticks(range(3),["Cap 2","Cap 20","B-selected"])
    axes[0].set_ylabel("Nonlinear mean standardized MSE")
    axes[0].set_title("Dimension policy",fontweight="bold")
    axes[0].grid(axis="y",alpha=.16)
    selection=table("cap_selections")
    selected=selection[(selection.method=="kahm_departure")&(selection.law!="linear_ar")]
    caps=[1,2,4,8,12,20];counts=[int((selected.selected_cap==cap).sum()) for cap in caps]
    assert sum(counts)==144
    axes[1].bar(range(6),counts,color="#146b85",width=.65)
    axes[1].set_xticks(range(6),[str(c) for c in caps]);axes[1].set_xlabel("Requested native cap")
    axes[1].set_ylabel("Nonlinear task suites (144 total)")
    axes[1].set_title("Validation-selected KAHM caps",fontweight="bold")
    for x,y in enumerate(counts): axes[1].text(x,y+.8,str(y),ha="center",va="bottom",fontsize=8)
    axes[1].set_ylim(0,max(counts)*1.17);axes[1].grid(axis="y",alpha=.16);axes[1].set_axisbelow(True)
    figure.savefig(output/"forecast_dimension_selection.png",dpi=210);plt.close(figure)
    primary=table("primary_contrasts")
    primary=primary[primary.panel=="nonlinear"].reset_index(drop=True)
    labels={"common":"Common features","rbf_departure":"RBF reconstruction","raw_rbf":"Raw RBF regression","raw_hgb":"Boosted trees","nonkahm_policy":"Non-KAHM policy","ridge":"Linear ridge control"}
    figure,axis=plt.subplots(figsize=(8.6,3.35),constrained_layout=True)
    for index,row in primary.iterrows():
        color="#146b85" if row.criterion_established else "#b65e36"
        axis.plot([row.simultaneous_one_sided_lower,row.margin_gap_mean],[index,index],color=color,linewidth=2)
        axis.plot(row.simultaneous_one_sided_lower,index,"|",color=color,markersize=11,markeredgewidth=1.6)
        axis.scatter(row.margin_gap_mean,index,c=color,s=35,zorder=3)
    axis.axvline(0,color="#526474",linewidth=1,linestyle="--")
    axis.set_yticks(range(len(primary)),[labels[m] for m in primary.comparator]);axis.invert_yaxis()
    axis.set_xlabel("Paired margin gap (standardized MSE units)")
    axis.set_title("Five nonlinear superiority comparisons",fontweight="bold")
    axis.grid(axis="x",alpha=.18)
    figure.savefig(output/"primary_margin_bounds.png",dpi=210);plt.close(figure)
    return [str(p) for p in sorted(output.glob("*.png"))]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("root",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();print(json.dumps(figures(args.root,args.output)))
