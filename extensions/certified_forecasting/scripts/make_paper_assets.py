"""Create paper tables/figures only from retained, completed experiment files."""
from pathlib import Path
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
TABLE=ROOT/"paper/tables"
FIG=ROOT/"paper/figures"
OUT=ROOT/"results/analysis"
SYSTEMS=["harmonic","quasiperiodic","nonlinear_ar","logistic"]
LABELS={"harmonic":"Harmonic","quasiperiodic":"Quasiperiodic",
        "nonlinear_ar":"Nonlinear AR","logistic":"Logistic"}
FAMILIES=["kahm_nested","rbf_nested","affine_nested","kahm_hard",
          "ridge_ar","polynomial","kahm_koopman","polynomial_koopman","persistence"]
NAMES={"kahm_nested":"KAHM blocks","rbf_nested":"RBF blocks",
       "affine_nested":"Affine-hull blocks","kahm_hard":"Hard KAHM states",
       "ridge_ar":"Ridge AR","polynomial":"Quadratic regression",
       "kahm_koopman":"KAHM rollout","polynomial_koopman":"Polynomial rollout",
       "persistence":"Persistence","constant":"Constant"}
COLORS={"pair":"#126782","center":"#e07a3f","upper":"#76519c","risk":"#303030"}


def savefig(fig,name):
    fig.savefig(FIG/(name+".pdf"),bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(FIG/(name+".png"),dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    for p in [TABLE,FIG,OUT]:p.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.titlesize":10.5,"axes.labelsize":10,
                         "xtick.labelsize":9.5,"ytick.labelsize":9.5,
                         "legend.fontsize":9.5,"pdf.fonttype":42,"ps.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    runs=[json.loads(p.read_text()) for p in sorted((ROOT/"results/main_v1").glob("*/results.json"))]
    if len(runs)!=20:raise RuntimeError("Expected all 20 frozen main runs")
    selected=[]
    all_rows=[]
    for r in runs:
        for family in sorted(set(c["family"] for c in r["results"])):
            c=min((c for c in r["results"] if c["family"]==family),key=lambda c:np.mean(c["validation_mse"]))
            for hi,h in enumerate([1,4,12]):
                selected.append({"system":r["system"],"seed":r["seed"],"family":family,
                                 "candidate":c["name"],"horizon":h,"test_rmse":np.sqrt(c["test_mse"][hi]),
                                 "rmse_upper":c["certificates"][hi]["rmse_upper"],
                                 "rmse_lower":c["certificates"][hi]["rmse_lower"],
                                 "n_features":c["n_features"],"history":c["history"]})
        for c in r["results"]:
            for hi,h in enumerate([1,4,12]):
                all_rows.append({"system":r["system"],"seed":r["seed"],"family":c["family"],
                                 "name":c["name"],"history":c["history"],"n_features":c["n_features"],
                                 "horizon":h,"test_rmse":np.sqrt(c["test_mse"][hi]),
                                 "rmse_upper":c["certificates"][hi]["rmse_upper"],
                                 "rmse_lower":c["certificates"][hi]["rmse_lower"]})
    selected=pd.DataFrame(selected);all_rows=pd.DataFrame(all_rows)
    selected.to_csv(OUT/"validation_selected.csv",index=False)
    all_rows.to_csv(OUT/"all_candidates.csv",index=False)
    g=selected.groupby(["system","family","horizon"]).agg(mean=("test_rmse","mean"),sd=("test_rmse","std"),upper=("rmse_upper","mean"))
    g.to_csv(OUT/"forecast_summary.csv")

    lines=[r"\begin{table}[t]",r"\centering\small",r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{lrrrrrrrr}",r"\toprule",
           r"& \multicolumn{2}{c}{Harmonic} & \multicolumn{2}{c}{Quasiperiodic} & \multicolumn{2}{c}{Nonlinear AR} & \multicolumn{2}{c}{Logistic}\\",
           r"Method & $h=1$ & $h=12$ & $h=1$ & $h=12$ & $h=1$ & $h=12$ & $h=1$ & $h=12$\\",r"\midrule"]
    for fam in FAMILIES:
        vals=[g.loc[(sys,fam,h)] for sys in SYSTEMS for h in [1,12]]
        cells=[r"\shortstack[r]{"+f"{v['mean']:.3f}"+r"\\("+f"{v['sd']:.3f}"+r")}" for v in vals]
        lines.append(NAMES[fam]+" & "+" & ".join(cells)+r"\\[2pt]")
    lines.extend([r"\bottomrule",r"\end{tabular}",
       r"\caption{Independent-test RMSE: mean over five complete fitting repetitions, with sample standard deviation below in parentheses. One candidate per family is selected by validation MSE across all three horizons before calibration or test scoring. The archive includes $h=4$, all candidates, and every confidence endpoint. Fitting variation differs from uncertainty conditional on a fitted model; these test errors are descriptive estimates.}",
       r"\label{tab:forecast}",r"\end{table}"])
    (TABLE/"forecast_results.tex").write_text("\n".join(lines)+"\n")

    selections=[]
    for r in runs:
        for s in r["certified_selections"]:selections.append({"system":r["system"],"seed":r["seed"],**s})
    s=pd.DataFrame(selections);s.to_csv(OUT/"selections.csv",index=False)
    lines=[r"\begin{table}[t]",r"\centering\small",r"\begin{tabular}{lcccccc}",r"\toprule",
           r"& \multicolumn{3}{c}{KAHM retained family} & \multicolumn{3}{c}{All candidates}\\",
           r"System & $\epsilon=.20$ & $.35$ & $.50$ & $\epsilon=.20$ & $.35$ & $.50$\\",r"\midrule"]
    for sys in SYSTEMS:
        vals=[]
        for fam in ["kahm_nested","all"]:
            for eps in [.2,.35,.5]:
                z=s[(s.system==sys)&(s.family==fam)&(s.tolerance==eps)]
                p=z[z.status=="certified"]
                med="--" if len(p)==0 else str(int(p.n_features.median()))
                vals.append(f"{len(p)}/5 ({med})")
        lines.append(LABELS[sys]+" & "+" & ".join(vals)+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}",
       r"\caption{Runs certifying the same RMSE tolerance at all horizons $1,4,12$. Parentheses give median retained feature dimension among successful least-dimension selections; `--' means no model passed. Cost is a representation-size proxy, not measured runtime. Each run has a joint 95\% event covering its complete frozen family and improvement comparisons.}",
       r"\label{tab:selection}",r"\end{table}"])
    (TABLE/"certified_selection.tex").write_text("\n".join(lines)+"\n")

    # Known-risk comparisons use MSE; no implicit RMSE/MSE change in the plot.
    known_path=ROOT/"results/known_risk_v1/trials.csv"
    known=pd.read_csv(known_path if known_path.exists() else known_path.with_suffix(".csv.gz"))
    exact=known[(known.p==.5)&(known.kappa==1.)]
    fig,axes=plt.subplots(1,2,figsize=(6,3.1),sharey=True)
    for ax,a in zip(axes,[0.,.6]):
        z=exact[exact.softness==a]
        means=z.groupby("n")[["pair_lower","centered_lower","fitted_upper","oracle_mse"]].mean()
        n=means.index.to_numpy()
        for col,label,color,mark in [("pair_lower","Pair lower",COLORS["pair"],"o"),
                                      ("centered_lower","Centered lower",COLORS["center"],"s"),
                                      ("fitted_upper","Fitted upper",COLORS["upper"],"^")]:
            ax.plot(n,means[col],label=label,color=color,marker=mark,linewidth=1.8)
        ax.axhline(means.oracle_mse.iloc[0],color=COLORS["risk"],linestyle="--",label="Exact oracle risk")
        ax.set_xscale("log",base=2);ax.set_xticks(n,labels=[str(i) for i in n]);ax.grid(alpha=.18)
        ax.set_xlabel("Calibration observations")
        ax.set_title("Collapsed ($a=0$)" if a==0 else "Partially separated ($a=0.6$)")
    axes[0].set_ylabel("Mean squared error")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0.17,1,1));savefig(fig,"known_risk")

    fig,axes=plt.subplots(2,2,figsize=(6,4.8),sharex=True)
    for ax,sys in zip(axes.flat,SYSTEMS):
        z=all_rows[(all_rows.system==sys)&(all_rows.family=="kahm_nested")&(all_rows.history==12)&(all_rows.horizon==4)]
        means=z.groupby("n_features")[["rmse_lower","test_rmse","rmse_upper"]].mean()
        for col,label,color,mark in [("rmse_lower","Class lower",COLORS["pair"],"v"),
                                    ("test_rmse","Independent test",COLORS["risk"],"o"),
                                    ("rmse_upper","Fitted upper",COLORS["upper"],"^")]:
            ax.plot(means.index,means[col],label=label,color=color,marker=mark,linewidth=1.7)
        ax.set_title(LABELS[sys]);ax.set_xticks([2,6,14,30]);ax.set_ylim(bottom=0);ax.grid(alpha=.18)
        ax.set_ylabel("RMSE, horizon 4")
    for ax in axes[-1,:]:ax.set_xlabel("Retained coordinates")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0.065,1,1));savefig(fig,"refinement")

    # Exact risks illustrate the effect of duplicating one feature block.
    copies=np.arange(1,9)
    euclid=np.maximum(2-np.sqrt(2*copies),0)**2/4
    block=np.full_like(copies,(2-np.sqrt(2))**2/4,dtype=float)
    fig,ax=plt.subplots(figsize=(6,2.8))
    ax.plot(copies,euclid,"o-",color=COLORS["center"],label="Ordinary Frobenius budget")
    ax.plot(copies,block,"s-",color=COLORS["pair"],label="Sum of block norms")
    ax.set(xlabel="Number of exact copies of the same block",ylabel="Exact minimum MSE",xticks=copies)
    ax.legend();ax.grid(alpha=.18);fig.tight_layout();savefig(fig,"duplicate_invariance")

    improvements=[{"system":r["system"],"seed":r["seed"],**v} for r in runs for v in r["paired_improvements"]]
    imp=pd.DataFrame(improvements);imp.to_csv(OUT/"paired_improvements.csv",index=False)
    pos=imp.groupby(["system","horizon"]).improvement_lower.apply(lambda x:int((x>0).sum()))
    lines=[r"\begin{table}[t]",r"\centering\small",r"\begin{tabular}{lrrr}",r"\toprule",
           r"System & $h=1$ & $h=4$ & $h=12$\\",r"\midrule"]
    for sys in SYSTEMS:lines.append(LABELS[sys]+" & "+" & ".join(f"{pos[sys,h]}/30" for h in [1,4,12])+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}",
                  r"\caption{Certified positive risk reductions for adjacent retained KAHM prefixes: three refinements, two history lengths, and five seeds per horizon. The other comparisons are inconclusive; they are not certified degradations.}",r"\label{tab:improvements}",r"\end{table}"])
    (TABLE/"improvements.tex").write_text("\n".join(lines)+"\n")
    maxgap=max(c.get("optimization_gap",0) for r in runs for c in r["results"])
    summary={"completed_main_runs":len(runs),"main_candidates":sum(r["n_candidates"] for r in runs),
             "main_candidate_horizon_pairs":len(all_rows),
             "primary_positive_lower_count":int(((all_rows.family=="kahm_nested")&(all_rows.rmse_lower>0)).sum()),
             "primary_lower_count":int((all_rows.family=="kahm_nested").sum()),
             "certified_positive_improvements":int((imp.improvement_lower>0).sum()),
             "total_improvement_comparisons":len(imp),
             "largest_numerical_optimization_gap":maxgap,
             "main_total_seconds":sum(r["elapsed_seconds"] for r in runs),
             "known_risk_trials":len(known),"known_risk_joint_failures":int((known.joint_valid==0).sum()),
             "confidence_scope":"Joint within each frozen replicate; no simultaneous statement over all 20 experiments"}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
