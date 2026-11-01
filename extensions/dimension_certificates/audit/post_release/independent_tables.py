"""Complete risk-study table reconstruction from verified sufficient statistics."""
import math
from pathlib import Path
import numpy as np
from independent_scores import read,arrays,moments,POLICIES,ARMS,HORIZONS

KEYS={
 "endpoints.csv":("job","alias","horizon"),
 "decisions.csv":("job","alias","rmse_tolerance"),
 "paired_contrasts.csv":("job","old","new","horizon"),
 "selected_forecasts.csv":("job","arm"),
 "selected_policy_summary.csv":("system","arm"),
 "policy_comparison_summary.csv":("system","old","new"),
 "certificate_resolution.csv":("system","arm","scope","rmse_tolerance"),
 "selected_bank_local_ranks.csv":("job","arm","block_index","cluster_index"),
 "stage_runtime.csv":("job",),
}


def job_tables(folder,cfg):
    development=read(folder/"development.json");cal=read(folder/"calibration.json");test=read(folder/"test.json")
    cs=arrays(folder/"calibration_scores.npz");ts=arrays(folder/"test_scores.npz")
    job=development["job"];tail=cfg["delta"]/cfg["campaign_total_tails"]
    tables={name:[] for name in ("endpoints.csv","decisions.csv","paired_contrasts.csv",
                                "selected_forecasts.csv","selected_bank_local_ranks.csv","stage_runtime.csv")}
    tables["stage_runtime.csv"].append({"job":job["id"],"system":job["system"],"development_seconds":development["elapsed_seconds"],
                                       "calibration_seconds":cal["elapsed_seconds"],"test_seconds":test["elapsed_seconds"]})
    for row in development["evaluation"]:
        alias=row["alias"];loss=cs[alias+"__loss"];witness=cs[alias+"__witness"];testloss=ts[alias+"__loss"]
        endpoints=[]
        for i,h in enumerate(HORIZONS):
            upper=moments(loss[:,i],4.,tail);lower=moments(witness[:,i],2.,tail)
            tmse=float(np.mean(testloss[:,i]))
            entry={"job":job["id"],"system":job["system"],"repetition":job["repetition"],"alias":alias,"arm":row["arm"],
                   "selected_forecast":row["selected_forecast"],"horizon":h,"history":row["history"],"cap":row["cap"],
                   "prefix_stage":row["stage"],"n_features":row["n_features"],"validation_mse":row["validation_mse"][i],
                   "calibration_mse":upper["mean"],"test_mse":tmse,"test_rmse":math.sqrt(tmse),
                   "test_mse_se":float(np.std(testloss[:,i],ddof=1)/math.sqrt(len(testloss))),
                   "mse_lower":lower["lower"],"mse_upper":upper["upper"],"endpoint_separation":upper["upper"]-lower["lower"],
                   "zero_lower":lower["lower"]==0.,"upper_reaches_loss_bound":upper["upper"]==4.,
                   "test_mean_below_lower":tmse<lower["lower"],"test_mean_above_upper":tmse>upper["upper"],"tail_delta":tail}
            tables["endpoints.csv"].append(entry);endpoints.append(entry)
        for tolerance in cfg["rmse_tolerances"]:
            accepted=all(r["mse_upper"]<=tolerance**2 for r in endpoints)
            excluded=any(r["mse_lower"]>tolerance**2 for r in endpoints)
            decision="conflicting_bounds" if accepted and excluded else "accepted" if accepted else "excluded" if excluded else "unresolved"
            tables["decisions.csv"].append({"job":job["id"],"system":job["system"],"alias":alias,"arm":row["arm"],
                                           "selected_forecast":row["selected_forecast"],"rmse_tolerance":tolerance,"decision":decision})
        if row["selected_forecast"]:
            tables["selected_forecasts.csv"].append({"job":job["id"],"system":job["system"],"repetition":job["repetition"],
                "arm":row["arm"],"history":row["history"],"cap":row["cap"],"prefix_stage":row["stage"],"n_features":row["n_features"],
                "validation_mean_mse":float(np.mean(row["validation_mse"])),"test_mean_mse":float(np.mean(testloss)),
                **{f"test_mse_h{h}":float(np.mean(testloss[:,i])) for i,h in enumerate(HORIZONS)},
                "coefficient_norm":row["coefficient_norm"],"optimization_gap":row["optimization_gap"],"solver_converged":row["solver_converged"]})
    contrasts=[("retained_prefix",policy,f"{policy}_s{s}",f"{policy}_s{s+1}") for policy in POLICIES for s in (1,2,3)]
    contrasts += [("selected_policy",f"{old}_to_{new}",f"{old}_s{development['choices'][old]['winner_stage']}",
                    f"{new}_s{development['choices'][new]['winner_stage']}") for old,new in cfg["policy_contrasts"]]
    for kind,label,old,new in contrasts:
        for i,h in enumerate(HORIZONS):
            delta=cs[old+"__loss"][:,i]-cs[new+"__loss"][:,i];oracle=moments(delta+4.,8.,tail)
            difference=ts[old+"__loss"][:,i]-ts[new+"__loss"][:,i]
            tables["paired_contrasts.csv"].append({"job":job["id"],"system":job["system"],"repetition":job["repetition"],
                "type":kind,"contrast":label,"old":old,"new":new,"horizon":h,
                "calibration_improvement":oracle["mean"]-4.,"improvement_lower":oracle["lower"]-4.,
                "certified_positive_improvement":oracle["lower"]-4.>0,
                "test_mse_improvement":float(np.mean(difference)),
                "test_paired_difference_se":float(np.std(difference,ddof=1)/math.sqrt(len(difference)))})
    for arm in ARMS:
        choice=development["choices"][arm];bank=development["banks"][choice["bank_id"]]
        for rank in bank["dimension_policy"]["local_models"]:
            tables["selected_bank_local_ranks.csv"].append({"job":job["id"],"system":job["system"],"arm":arm,
                "history":choice["history"],"cap":choice["cap"],"selected_prefix_stage":choice["winner_stage"],
                "used_by_selected_forecast":rank["block_index"]<choice["winner_stage"],"block_index":rank["block_index"],
                "cluster_index":rank["regime_index"],"ambient_dimension":rank["ambient_dimension"],
                "local_reference_count":rank["local_reference_count"],"initial_admissible_rank":rank["initial_admissible_rank"],
                "effective_rank":rank["effective_rank"],"rank_reduction_reasons":"|".join(rank["rank_reduction_reasons"])})
    return tables


def aggregate_tables(tables,cfg):
    summaries=[];contrasts=[];resolution=[]
    selected=tables["selected_forecasts.csv"];ends=tables["endpoints.csv"];decisions=tables["decisions.csv"]
    for system in cfg["systems"]:
        name=system["name"];system_rows=[r for r in selected if r["system"]==name]
        for arm in (*ARMS,"ridge_ar","polynomial","persistence","constant"):
            rows=[r for r in system_rows if r["arm"]==arm];values=[r["test_mean_mse"] for r in rows]
            summaries.append({"system":name,"arm":arm,"repetitions":len(rows),"mean_test_mse":float(np.mean(values)),
                "sd_test_mse_over_repetitions":float(np.std(values,ddof=1)),
                **{f"mean_test_mse_h{h}":float(np.mean([r[f"test_mse_h{h}"] for r in rows])) for h in HORIZONS},
                "selected_caps":";".join(str(r["cap"]) for r in rows),
                "selected_histories":";".join(str(r["history"]) for r in rows),
                "selected_prefix_stages":";".join(str(r["prefix_stage"]) for r in rows),
                "nonconverged_selected_fits":sum(not r["solver_converged"] for r in rows)})
        for old,new in cfg["policy_contrasts"]:
            old_values={r["job"]:r["test_mean_mse"] for r in system_rows if r["arm"]==old}
            new_values={r["job"]:r["test_mean_mse"] for r in system_rows if r["arm"]==new}
            order=sorted(old_values);x=np.array([old_values[j] for j in order]);y=np.array([new_values[j] for j in order])
            delta=y-x
            contrasts.append({"system":name,"old":old,"new":new,"repetitions":len(order),
                "old_mean_test_mse":float(np.mean(x)),"new_mean_test_mse":float(np.mean(y)),
                "mean_test_mse_change":float(np.mean(delta)),"percent_change_in_mean_test_mse":float(100*(np.mean(y)/np.mean(x)-1)),
                "favorable_repetitions":int(np.sum(delta<0)),"equal_repetitions":int(np.sum(delta==0)),
                "unfavorable_repetitions":int(np.sum(delta>0))})
        for arm in POLICIES:
            for only_selected in (False,True):
                entries=[r for r in ends if r["system"]==name and r["arm"]==arm and (r["selected_forecast"] or not only_selected)]
                for tolerance in cfg["rmse_tolerances"]:
                    rows=[r for r in decisions if r["system"]==name and r["arm"]==arm and r["rmse_tolerance"]==tolerance
                          and (r["selected_forecast"] or not only_selected)]
                    resolution.append({"system":name,"arm":arm,"scope":"selected_forecast" if only_selected else "all_retained_prefixes",
                        "rmse_tolerance":tolerance,"accepted":sum(r["decision"]=="accepted" for r in rows),
                        "excluded":sum(r["decision"]=="excluded" for r in rows),"unresolved":sum(r["decision"]=="unresolved" for r in rows),
                        "conflicting_bounds":sum(r["decision"]=="conflicting_bounds" for r in rows),
                        "zero_lower_endpoints":sum(r["zero_lower"] for r in entries),
                        "upper_endpoints_at_bound":sum(r["upper_reaches_loss_bound"] for r in entries),
                        "mean_endpoint_separation":float(np.mean([r["endpoint_separation"] for r in entries]))})
    tables["selected_policy_summary.csv"]=summaries;tables["policy_comparison_summary.csv"]=contrasts
    tables["certificate_resolution.csv"]=resolution
    return tables
