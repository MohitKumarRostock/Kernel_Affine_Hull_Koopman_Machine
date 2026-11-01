"""Independent complete table reconstruction; no primary analyzer imports."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import stdtr, stdtrit
from verify_task_artifacts import CANDIDATES, NATIVE, METHODS, GEOMETRIC, SHARED, FULL, NONKAHM, CONTEXTS, MODES, blend

H=(1,4,12)
META=("task_id","law","history","repetition","seed","noise_index","noise_level")
PANELS={"nonlinear":("nonlinear_ar","vanderpol","duffing"),"linear":("linear_ar",),
        "all_descriptive":("linear_ar","nonlinear_ar","vanderpol","duffing")}
VIEWS=tuple((c,m) for c,_,_ in CANDIDATES for m in METHODS)
VIEWS+=tuple(("shared",m) for m in SHARED)
VIEWS+=tuple(("context_"+c,m) for c in CONTEXTS for m in FULL)
VIEWS+=tuple(("matched",m) for m in GEOMETRIC)
CAP={c:v for c,v,_ in CANDIDATES}
POLICY={c:p for c,_,p in CANDIDATES}
SECONDARY=(
 ("selected_minus_historical_KAHM","context_selected","kahm_departure","context_historical2","kahm_departure"),
 ("native20_minus_historical_KAHM","context_native20","kahm_departure","context_historical2","kahm_departure"),
 ("selected_minus_native20_KAHM","context_selected","kahm_departure","context_native20","kahm_departure"),
 ("selected_KAHM_minus_selected_RBF","context_selected","kahm_departure","context_selected","rbf_departure"),
 ("selected_KAHM_minus_selected_common","context_selected","kahm_departure","context_selected","common"),
 ("matched_KAHM_minus_common","matched","kahm_departure","matched","common"),
 ("matched_KAHM_minus_RBF","matched","kahm_departure","matched","rbf_departure"),
 ("selected_bounded_minus_unbounded_KAHM","context_selected","kahm_departure","context_selected","kahm_unbounded"),
 ("matched_bounded_minus_unbounded_KAHM","matched","kahm_departure","matched","kahm_unbounded"),
 ("native20_KAHM_minus_RBF","context_native20","kahm_departure","context_native20","rbf_departure"),
 ("historical_KAHM_minus_RBF","context_historical2","kahm_departure","context_historical2","rbf_departure"))


def metadata(record):
    return {key:record[key] for key in META}


def selected_source(view,method,fit):
    if view in CAP or view=="shared":return view
    if view=="matched":return fit["selections"]["kahm_departure"]["selected_candidate_id"]
    context=view[len("context_"):]
    if method=="nonkahm_policy":method=fit["contexts"][context]["selected_policy"]
    if method in SHARED:return "shared"
    return fit["selections"][method]["selected_candidate_id"] if context=="selected" else context


def score_task(task,target,predictions,scale,fit):
    horizon,losses,screen=[],[],[]
    baseline_max=float(abs(target-predictions["shared__calibrated__ridge"]).max())
    for view,method in VIEWS:
        for mode in MODES:
            error=target-predictions[f"{view}__{mode}__{method}"]
            mse=np.array([np.dot(error[:,i],error[:,i])/len(error) for i in range(3)])
            identity={**metadata(task),"view":view,"method":method,"mode":mode,
                      "source_candidate":selected_source(view,method,fit)}
            horizon.extend({**identity,"horizon":h,"D_rows":len(target),"A_target_variance":float(scale[i]),
                            "mse":float(mse[i]),"smse":float(mse[i]/scale[i])} for i,h in enumerate(H))
            losses.append({**identity,"mse":math.fsum(mse)/3,"smse":math.fsum(mse/scale)/3})
            if view.startswith("context_") and method=="kahm_departure" and mode=="calibrated":
                maximum=float(abs(error).max());ratio=maximum/baseline_max if baseline_max else (0. if not maximum else math.inf)
                screen.append({**identity,"maximum_absolute_error":maximum,"ridge_maximum_absolute_error":baseline_max,
                               "maximum_error_ratio":ratio,"screen_pass":ratio<=5.})
    return horizon,losses,screen


def means(rows,keys,values=("smse","mse"),count="n_records"):
    frame=pd.DataFrame(rows);groups=frame.groupby(list(keys),sort=True,dropna=False)
    result=groups[list(values)].mean();result.insert(0,count,groups.size())
    return result.reset_index().to_dict("records")


def scalar_moments(values):
    values=[float(v) for v in values];n=len(values);mean=math.fsum(values)/n
    sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(n-1))
    return mean,sd,sd/math.sqrt(n)


def outcome_tables(horizon,losses):
    tables={"horizon_losses.csv":horizon,"task_losses.csv":losses}
    tables["condition_summary.csv"]=means(losses,("law","noise_level","view","method","mode"))
    tables["condition_horizon_summary.csv"]=means(horizon,("law","noise_level","view","method","mode","horizon"))
    reps=[];panelh=[]
    for panel,laws in PANELS.items():
        reps.extend({"panel":panel,**r} for r in means([x for x in losses if x["law"] in laws],
                    ("repetition","view","method","mode"),count="n_conditions"))
        panelh.extend({"panel":panel,**r} for r in means([x for x in horizon if x["law"] in laws],
                    ("view","method","mode","horizon")))
    tables["repetition_losses.csv"]=reps
    tables["panel_summary.csv"]=means(reps,("panel","view","method","mode"))
    tables["panel_horizon_summary.csv"]=panelh
    index={(r["panel"],r["repetition"],r["view"],r["method"],r["mode"]):r for r in reps}
    primary=[];primary_detail=[]
    comparisons=[("nonlinear",m,.97,"nonlinear_vs_"+m) for m in ("common","rbf_departure","raw_rbf","raw_hgb","nonkahm_policy")]
    comparisons += [("linear","ridge",1.05,"linear_noninferiority_vs_ridge")]
    for panel,comparator,factor,cid in comparisons:
        candidate=[index[panel,r,"context_selected","kahm_departure","calibrated"]["smse"] for r in range(24)]
        control=[index[panel,r,"context_selected",comparator,"calibrated"]["smse"] for r in range(24)]
        gaps=[factor*b-c for b,c in zip(control,candidate)]
        gap,sd,se=scalar_moments(gaps);critical=float(stdtrit(23,1-.05/6))
        c,b=math.fsum(candidate)/24,math.fsum(control)/24
        p=float(stdtr(23,-gap/se)) if se else (0. if gap>0 else 1. if gap<0 else .5)
        lower=gap-critical*se
        primary.append({"contrast_id":cid,"panel":panel,"context":"selected","candidate":"kahm_departure",
                        "comparator":comparator,"candidate_mean_smse":c,"comparator_mean_smse":b,
                        "point_relative_gain_percent":100*(1-c/b) if b else None,
                        "comparator_factor":factor,"margin":.03 if panel=="nonlinear" else .05,
                        "n_repetitions":24,"df":23,"family_alpha":.05,"family_size":6,
                        "margin_gap_mean":gap,"margin_gap_sample_sd":sd,"margin_gap_se":se,
                        "critical_t":critical,"simultaneous_one_sided_lower":lower,"p_one_sided":p,
                        "p_bonferroni":min(1.,6*p),"positive_margin_repetitions":sum(v>0 for v in gaps),
                        "criterion_established":lower>0})
        primary_detail.extend({"contrast_id":cid,"panel":panel,"repetition":r,"candidate_smse":candidate[r],
                               "comparator_smse":control[r],"comparator_factor":factor,"margin_gap":gaps[r]} for r in range(24))
    tables["primary_contrasts.csv"]=primary;tables["primary_repetition_contrasts.csv"]=primary_detail
    secondary=[];secondary_detail=[]
    for panel in PANELS:
        for mode in MODES:
            for cid,lv,lm,rv,rm in SECONDARY:
                left=[index[panel,r,lv,lm,mode]["smse"] for r in range(24)]
                right=[index[panel,r,rv,rm,mode]["smse"] for r in range(24)]
                delta=np.array(left)-np.array(right);point,sd,se=scalar_moments(delta);critical=float(stdtrit(23,.975))
                l,r=math.fsum(left)/24,math.fsum(right)/24
                identity={"contrast_id":cid,"panel":panel,"mode":mode,"left_view":lv,"left_method":lm,"right_view":rv,"right_method":rm}
                secondary.append({**identity,"inference":"secondary_unadjusted_paired_t","n_repetitions":24,
                                  "left_mean_smse":l,"right_mean_smse":r,"relative_gain_percent":100*(1-l/r),
                                  "mean_difference":point,"standard_error":se,"lower95":point-critical*se,"upper95":point+critical*se,
                                  "better_repetitions":int(sum(delta<0)),"ties":int(sum(delta==0)),"worse_repetitions":int(sum(delta>0))})
                secondary_detail.extend({**identity,"repetition":r,"left_smse":left[r],"right_smse":right[r],"difference":float(delta[r])} for r in range(24))
    tables["secondary_contrasts.csv"]=secondary;tables["secondary_repetition_contrasts.csv"]=secondary_detail
    taskindex=defaultdict(dict)
    for row in losses:taskindex[row["task_id"],row["mode"]][row["view"],row["method"]]=row
    decomp=[]
    for (_,mode),rows in taskindex.items():
        oldk=rows["historical2","kahm_departure"]["smse"];oldc=rows["historical2","common"]["smse"]
        for view in ("native20","matched"):
            k,c=rows[view,"kahm_departure"],rows[view,"common"]
            complete=k["smse"]-oldk;common=c["smse"]-oldc
            increment=(k["smse"]-c["smse"])-(oldk-oldc)
            decomp.append({**metadata(k),"mode":mode,"view":view,"complete_change":complete,"common_change":common,
                           "departure_increment_change":increment,"historical_departure_increment":oldk-oldc,
                           "new_departure_increment":k["smse"]-c["smse"],"identity_residual":complete-common-increment})
    tables["dimension_decomposition.csv"]=decomp
    values=("complete_change","common_change","departure_increment_change","historical_departure_increment","new_departure_increment","identity_residual")
    tables["decomposition_summary.csv"]=[{"panel":panel,**r} for panel,laws in PANELS.items()
                for r in means([x for x in decomp if x["law"] in laws],("view","mode"),values)]
    return tables


def diagnostic_tables(task,fit,pre,abc,validation,calibration,basis,query,predictions,config):
    tables=defaultdict(list);identity=metadata(task)
    variance=np.maximum(abc["A_Y"].var(axis=0),1e-12)
    for method in GEOMETRIC:
        scores=[]
        for candidate in NATIVE:
            mse=np.mean((abc["B_Y"]-validation[f"{candidate}__{method}"])**2,axis=0)
            score=float(np.mean(mse/variance));scores.append(score)
            tables["B_cap_scores.csv"].append({**identity,"method":method,"candidate_id":candidate,"requested_cap":CAP[candidate],"B_smse":score,
                                              **{f"B_mse_h{h}":float(mse[i]) for i,h in enumerate(H)}})
        i=int(np.argmin(scores))
        tables["cap_selections.csv"].append({**identity,"method":method,"selected_candidate":NATIVE[i],"selected_cap":CAP[NATIVE[i]],
                    "B_smse":scores[i],"exact_minimum_ties":sum(v==scores[i] for v in scores),"runner_up_gap":sorted(scores)[1]-scores[i]})
    gammas={}
    for candidate,record in list(fit["candidates"].items())+[("shared",fit["shared"])]:
        diag=record["fit_diagnostics"];methods=SHARED if candidate=="shared" else METHODS
        for method in methods:
            if method!="persistence":
                choices=diag["B"][method];trials=np.array([r["mse_by_horizon"] for r in choices["candidates"]])
                indices=trials.argmin(axis=0);mse=np.mean((abc["B_Y"]-validation[f"{candidate}__{method}"])**2,axis=0)
                for i,h in enumerate(H):
                    selected=choices["candidates"][indices[i]]
                    tables["B_tuning_choices.csv"].append({**identity,"candidate_id":candidate,"method":method,"horizon":h,"selected_index":int(indices[i]),
                        "alpha":selected.get("alpha"),"bandwidth_factor":selected.get("bandwidth_factor"),
                        "learning_rate":selected.get("learning_rate"),"max_leaf_nodes":selected.get("max_leaf_nodes"),"B_mse":float(mse[i])})
            if method=="ridge":gammas[candidate,method]=np.zeros(3);continue
            grid=np.array(config["model"]["gamma_grid"]);base=calibration["shared__calibrated__ridge"]
            raw=calibration[f"{candidate}__gamma1__{method}"]
            trials=np.array([np.mean((abc["C_Y"]-blend(base,raw,[g]*3))**2,axis=0) for g in grid])
            indices=trials.argmin(axis=0);gamma=grid[indices];gammas[candidate,method]=gamma
            for i,h in enumerate(H):
                tables["C_gamma_choices.csv"].append({**identity,"candidate_id":candidate,"method":method,"horizon":h,
                                                    "gamma":float(gamma[i]),"C_mse":float(trials[indices[i],i])})
        if candidate=="shared":continue
        local=diag["local_dimensions"];ranks=[]
        for j,item in enumerate(local):
            q,p=basis[f"{candidate}__model{j}__q"],basis[f"{candidate}__model{j}__p"]
            rank=p.shape[1];n,d=q.shape;ranks.append(rank)
            tables["local_dimensions.csv"].append({**identity,"candidate_id":candidate,"model_index":j,"requested_cap":CAP[candidate],"policy":POLICY[candidate],
                "local_reference_count":n,"initial_admissible_rank":min(CAP[candidate],d-int(POLICY[candidate]=="legacy_normal_departure"),n-1),
                "effective_rank":rank,"fallback_applied":item["fallback_applied"],"rank_reduction_reasons":item["rank_reduction_reasons"],
                "initial_limit_reasons":item["initial_limit_reasons"],"range_reduction_steps":item["range_reduction_steps"],"native_failure":item["native_failure"]})
        widths=record["feature_widths"]
        tables["geometry_summary_by_task.csv"].append({**identity,"candidate_id":candidate,"n_local_models":len(ranks),"requested_cap":CAP[candidate],
                    "minimum_rank":min(ranks),"mean_rank":float(np.mean(ranks)),"maximum_rank":max(ranks),"full_ambient_models":ranks.count(12),
                    "zero_rank_models":ranks.count(0),"common_width":len(ranks)+sum(ranks),"departure_width":len(ranks)*12,
                    "stacked_basis_rank":diag["bank"]["stack_diagnostics"]["numerical_rank"]})
        for method in GEOMETRIC:
            bound=record["bounds"][method];alpha=np.array(bound["alpha_by_horizon"])
            limit=np.array(bound["centered_response_rms_by_horizon"])/(2*np.sqrt(alpha))
            coefficients=np.array(bound["coefficient_norm_by_horizon"])
            bounded=not method.endswith("unbounded");ratio=None
            if bounded:
                radius=math.sqrt(widths["full_readout"][method])
                readout=np.abs(bound["response_mean_by_horizon"])+(radius+bound["training_feature_mean_norm"])*limit
                correction=gammas[candidate,method]*readout
                maximum=np.max(abs(predictions[f"{candidate}__calibrated__{method}"]-predictions["shared__calibrated__ridge"]),axis=0)
                ratio=float(np.max(np.divide(maximum,correction,out=np.zeros(3),where=correction!=0)))
            violations=int(np.count_nonzero(coefficients>limit*(1+1e-10)+1e-12))
            tables["bound_checks.csv"].append({**identity,"candidate_id":candidate,"method":method,"bounded":bounded,"coefficient_bound_violations":violations,
                    "maximum_coefficient_bound_ratio":float(np.max(np.divide(coefficients,limit,out=np.zeros(3),where=limit!=0))),
                    "maximum_D_correction_bound_ratio":ratio,"D_correction_bound_violations":0 if bounded else None})
        stages={"A":diag["features_A_B"]["A"],"B":diag["features_A_B"]["B"],"C":record["calibration"]["query_diagnostics"],"D":query["diagnostics"][candidate]}
        for stage,record in stages.items():
            for method in GEOMETRIC:
                f=record["method_features"][method];kind=method.split("_")[0]
                tables["feature_checks.csv"].append({**identity,"candidate_id":candidate,"stage":stage,"method":method,"bounded":not method.endswith("unbounded"),
                    "maximum_norm":f["max_norm"],"radius":f.get("radius"),"bound_violations":f.get("violations"),
                    "fallback_rows":record.get(kind+"_fallback_count",0),
                    "factor_failure_models":sum("factor_failure" in r for r in record["local_models"] if r.get("kind")==kind)})
    for context in CONTEXTS:
        losses=[float(np.mean(np.mean((abc["C_Y"]-calibration[f"context_{context}__calibrated__{m}"])**2,axis=0)/variance)) for m in NONKAHM]
        tables["C_policy_choices.csv"].append({**identity,"context":context,"selected_method":NONKAHM[int(np.argmin(losses))],"C_smse":min(losses)})
        for method,record in fit["procedure_resources"]["per_context"][context].items():
            nodes=record["fit_and_selection"]["node_ids"];registry=fit["procedure_resources"]["node_registry"]
            tables["procedure_resources.csv"].append({**identity,"context":context,"method":method,
                "fit_and_selection_wall_seconds":sum(registry[n]["wall_seconds"] for n in nodes),
                "fit_and_selection_cpu_seconds":sum(registry[n]["cpu_seconds"] for n in nodes),
                "deployment_pickle_bytes":record["state"]["pickle_bytes"],"deployment_array_bytes":record["state"]["array_buffer_bytes"],
                "accounting":"source dependency nodes; selected geometric procedures include six-cap search"})
    tables["task_resources.csv"].append({**identity,"fit_wall_seconds":fit["resources"]["task_fit_wall_seconds"],"fit_cpu_seconds":fit["resources"]["task_fit_cpu_seconds"],
                    "query_wall_seconds":query["resources"]["total_wall_seconds"],"query_cpu_seconds":query["resources"]["total_cpu_seconds"],
                    "peak_rss_kib":fit["resources"]["process_peak_rss_kib"]})
    return dict(tables)


def derived_tables(tables):
    rows=tables["cap_selections.csv"];count=[]
    for law in PANELS["all_descriptive"]:
        for noise in (0.,.1):
            for method in GEOMETRIC:
                group=[r for r in rows if r["law"]==law and r["noise_level"]==noise and r["method"]==method]
                count.extend({"law":law,"noise_level":noise,"method":method,"cap":cap,"count":sum(r["selected_cap"]==cap for r in group)} for cap in (1,2,4,8,12,20))
    tables["cap_selection_counts.csv"]=count
    tables["resource_summary.csv"]=means(tables["procedure_resources.csv"],("context","method"),
        ("fit_and_selection_wall_seconds","fit_and_selection_cpu_seconds","deployment_pickle_bytes","deployment_array_bytes"))
    return tables


TABLE_KEYS={
 "horizon_losses.csv":("task_id","view","method","mode","horizon"),
 "task_losses.csv":("task_id","view","method","mode"),
 "condition_summary.csv":("law","noise_level","view","method","mode"),
 "condition_horizon_summary.csv":("law","noise_level","view","method","mode","horizon"),
 "repetition_losses.csv":("panel","repetition","view","method","mode"),
 "panel_summary.csv":("panel","view","method","mode"),
 "panel_horizon_summary.csv":("panel","view","method","mode","horizon"),
 "primary_contrasts.csv":("contrast_id",),
 "primary_repetition_contrasts.csv":("contrast_id","repetition"),
 "secondary_contrasts.csv":("contrast_id","panel","mode"),
 "secondary_repetition_contrasts.csv":("contrast_id","panel","mode","repetition"),
 "dimension_decomposition.csv":("task_id","mode","view"),
 "decomposition_summary.csv":("panel","view","mode"),
 "B_cap_scores.csv":("task_id","method","candidate_id"),
 "cap_selections.csv":("task_id","method"),
 "B_tuning_choices.csv":("task_id","candidate_id","method","horizon"),
 "C_gamma_choices.csv":("task_id","candidate_id","method","horizon"),
 "C_policy_choices.csv":("task_id","context"),
 "local_dimensions.csv":("task_id","candidate_id","model_index"),
 "geometry_summary_by_task.csv":("task_id","candidate_id"),
 "bound_checks.csv":("task_id","candidate_id","method"),
 "feature_checks.csv":("task_id","candidate_id","stage","method"),
 "procedure_resources.csv":("task_id","context","method"),
 "task_resources.csv":("task_id",),
 "stability_screens.csv":("task_id","view"),
 "cap_selection_counts.csv":("law","noise_level","method","cap"),
 "resource_summary.csv":("context","method"),
}


def table_comparison(directory, expected_tables, audit):
    """Compare every parsed CSV field; no sampled rows and no rounded comparison."""
    if set(expected_tables) != set(TABLE_KEYS):
        raise AssertionError("Independent complete table roster differs")
    observed_files = {path.name for path in Path(directory).glob("*.csv")}
    audit.require(observed_files == set(TABLE_KEYS), "Exported complete table roster")
    result = {}
    for name, expected in expected_tables.items():
        with (Path(directory)/name).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            actual = list(reader)
        fields = set(expected[0])
        audit.require(set(header) == fields and len(header) == len(fields), "Table field roster", name)
        audit.require(len(actual) == len(expected), "Table row count", (name, len(actual), len(expected)))
        kinds = {}
        for field in fields:
            samples = [row[field] for row in expected if row[field] is not None]
            if not samples:
                kinds[field] = "none"
            elif all(isinstance(value, (bool, np.bool_)) for value in samples):
                kinds[field] = "bool"
            elif all(isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) for value in samples):
                kinds[field] = "number"
            elif all(isinstance(value, (list, dict, tuple)) for value in samples):
                kinds[field] = "json"
            else:
                kinds[field] = "string"

        def parse(field, value):
            if value == "":
                return None
            kind = kinds[field]
            if kind == "number":
                return float(value)
            if kind == "bool":
                if value not in ("True", "False"):
                    raise AssertionError("Invalid CSV boolean")
                return value == "True"
            if kind == "json":
                return json.loads(value)
            return value

        actual_index = {}
        for row in actual:
            audit.require(None not in row and all(value is not None for value in row.values()), "Complete CSV row", name)
            normalized = {field: parse(field, row[field]) for field in fields}
            key = tuple(normalized[field] for field in TABLE_KEYS[name])
            audit.require(key not in actual_index, "Duplicate table key", (name, key))
            actual_index[key] = normalized
        expected_keys = {tuple(row[field] for field in TABLE_KEYS[name]) for row in expected}
        audit.require(len(expected_keys) == len(expected) and set(actual_index) == expected_keys, "Exact table key roster", name)
        absolute = 0.
        numeric_cells = 0
        exact_cells = 0
        for row in expected:
            key = tuple(row[field] for field in TABLE_KEYS[name])
            found = actual_index[key]
            for field, wanted in row.items():
                observed = found[field]
                context = (name, key, field, observed, wanted)
                if wanted is None:
                    audit.require(observed is None, "Expected null table cell", context)
                    exact_cells += 1
                elif kinds[field] == "number":
                    audit.require(observed is not None and math.isfinite(observed) and math.isfinite(float(wanted))
                                  and math.isclose(observed, float(wanted), rel_tol=2e-11, abs_tol=2e-13),
                                  "Independent numeric table cell", context)
                    absolute = max(absolute, abs(observed-float(wanted)))
                    numeric_cells += 1
                else:
                    audit.require(observed == wanted, "Independent exact table cell", context)
                    exact_cells += 1
        audit.counts["export_numeric_cells_verified"] += numeric_cells
        audit.counts["export_exact_cells_verified"] += exact_cells
        audit.counts["export_rows_verified"] += len(actual)
        audit.counts["export_tables_verified"] += 1
        result[name] = {"rows": len(actual), "columns": len(fields), "numeric_cells": numeric_cells,
                        "exact_cells": exact_cells, "maximum_absolute_numeric_difference": absolute}
    return result
