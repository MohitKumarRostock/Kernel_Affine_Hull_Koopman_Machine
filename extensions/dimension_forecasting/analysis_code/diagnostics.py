"""Independent checks of selection, dimensions, bounds and recorded costs."""
from __future__ import annotations
import math
import numpy as np
from analysis_core import (CANDIDATES, NATIVE, CONTEXTS, GEOMETRY, RAW, NONKAHM,
    HORIZONS, META, VIEWS, MODES, array_hash, blend, close, finite, require)


def diagnose(meta, fit, pre, validation, calibration, basis, query, forecasts, arrays, config):
    identity={k:meta[k] for k in META}
    tables={name:[] for name in ("B_cap_scores.csv","cap_selections.csv","B_tuning_choices.csv",
        "C_gamma_choices.csv","C_policy_choices.csv","local_dimensions.csv","geometry_summary_by_task.csv",
        "bound_checks.csv","feature_checks.csv","procedure_resources.csv","task_resources.csv")}
    ay,by,cy=arrays["A_Y"],arrays["B_Y"],arrays["C_Y"]
    variance=np.maximum(np.var(ay,axis=0,ddof=0),1e-12)
    for value in (fit["A_target_variance"],pre["A_target_variance"],validation["A_target_variance"]):
        close(value,variance,"A variance",rtol=0,atol=0)
    require(fit["status"]==query["status"]=="completed","Incomplete fit/query")
    require(pre["C_arrays_loaded"] is False and pre["D_generated_or_loaded"] is False,"Selection stage violation")
    require(query["D_target_file_opened"] is False,"Query accessed terminal targets")
    require(fit["selections"]==pre["selections"],"Cap selection changed after pre-C seal")
    require(set(fit["selections"])==set(GEOMETRY),"Cap method roster differs")
    require(set(fit["candidates"])==set(CANDIDATES) and set(fit["contexts"])==set(CONTEXTS),"Fit roster differs")
    expected_b={f"{c}__{m}" for c in CANDIDATES for m in ("ridge",)+GEOMETRY}|{f"shared__{m}" for m in RAW}|{"A_target_variance"}
    require(set(validation)==expected_b,"B forecast roster differs")
    expected_c={f"{v}__{mode}__{m}" for v,m in VIEWS for mode in MODES}
    require(set(calibration)==expected_c,"C forecast roster differs")
    for key,value in calibration.items(): finite(value,cy.shape,key)
    for key in ("historical_native2_identity","native12_native20_identity","reference_identity_across_candidates"):
        require(fit[key]["status"]=="passed","Construction identity check failed: "+key)
    for state,record in query["state_identity"].items():
        require(record["unchanged"] and record["before_sha256"]==record["after_sha256"],"Query changed state: "+state)
    require(set(query["state_identity"])==set(CANDIDATES)|{"shared"},"State identity roster differs")

    for method in GEOMETRY:
        selection=pre["selections"][method]
        require(selection["selection_split"]=="B" and selection["calibration_used"] is False,"Cap selection role differs")
        require(tuple(selection["candidate_order"])==NATIVE,"Native order differs")
        require(selection["B_target_sha256"]==array_hash(by),"Cap target hash differs")
        scores=[]
        for j,candidate in enumerate(NATIVE):
            prediction=finite(validation[f"{candidate}__{method}"],by.shape,"B prediction")
            losses=np.mean((by-prediction)**2,axis=0); score=float(np.mean(losses/variance)); scores.append(score)
            saved=selection["candidates"][j]
            require(saved["candidate_id"]==candidate and saved["candidate_index"]==j,"Cap record order differs")
            require(saved["requested_cap"]==CANDIDATES[candidate][0] and saved["dimension_policy"]=="native","Cap record policy differs")
            require(saved["B_prediction_sha256"]==array_hash(prediction),"B forecast hash differs")
            close(saved["mse_by_horizon"],losses,"B cap losses"); close(saved["normalized_score"],score,"B cap score")
            tables["B_cap_scores.csv"].append({**identity,"method":method,"candidate_id":candidate,"requested_cap":CANDIDATES[candidate][0],"B_smse":score,
                **{f"B_mse_h{h}":float(losses[k]) for k,h in enumerate(HORIZONS)}})
        chosen=int(np.argmin(scores))
        require(selection["selected_index"]==chosen and selection["selected_candidate_id"]==NATIVE[chosen] and selection["selected_cap"]==CANDIDATES[NATIVE[chosen]][0],"Cap is not first B minimum")
        tables["cap_selections.csv"].append({**identity,"method":method,"selected_candidate":NATIVE[chosen],"selected_cap":selection["selected_cap"],
            "B_smse":scores[chosen],"exact_minimum_ties":sum(s==scores[chosen] for s in scores),"runner_up_gap":sorted(scores)[1]-scores[chosen]})

    alpha_grid=config["model"]["alpha_grid"]; gamma_grid=config["model"]["gamma_grid"]
    gammas={}; references=[]; expected_basis=set()
    all_records={**fit["candidates"],"shared":fit["shared"]}
    for candidate,record in all_records.items():
        diag=record["fit_diagnostics"]
        require(diag["calibration_used"] is False,"A/B fitting used C")
        methods=RAW if candidate=="shared" else ("ridge",)+GEOMETRY
        if candidate=="shared":
            require(not diag["raw_grid_failures"],"Raw grid failure")
            require(record["calibration"]["shared_pre_C_state_sha256"]==pre["shared_state_sha256"]==record["calibration"]["shared_post_C_state_sha256"],"C changed shared A/B state")
        else:
            require(record["calibration"]["candidate_pre_C_state_sha256"]==pre["candidate_state_sha256"][candidate]==record["calibration"]["candidate_post_C_state_sha256"],"C changed candidate A/B state")
        for method in methods:
            b_prediction=finite(validation[f"{candidate}__{method}"],by.shape,"B selected prediction")
            selected_mse=np.mean((by-b_prediction)**2,axis=0)
            if method!="persistence":
                b=diag["B"][method]
                require(b["fit_split"]=="A" and b["selection_split"]=="B","B model role differs")
                trials=np.asarray([x["mse_by_horizon"] for x in b["candidates"]],float)
                require(trials.shape[1:]==(3,) and np.isfinite(trials).all(),"Invalid B trial losses")
                if method=="raw_rbf":
                    require([(x["bandwidth_factor"],x["alpha"]) for x in b["candidates"]]==[(v,a) for v in (.5,1.,2.) for a in alpha_grid],"Raw RBF grid differs")
                    require(not b["failures"] and len(b["landmark_indices"])==64,"Raw RBF fit failure or landmark count")
                elif method=="raw_hgb":
                    require([(x["learning_rate"],x["max_leaf_nodes"]) for x in b["candidates"]]==[(r,l) for r in (.05,.1) for l in (7,15)],"HGB grid differs")
                    require(not b["failures"] and all(x["max_iter"]==100 and x["l2_regularization"]==1 and not x["early_stopping"] for x in b["candidates"]),"HGB fixed settings differ")
                else:
                    require([x["alpha"] for x in b["candidates"]]==alpha_grid,"Ridge grid differs")
                for j,h in enumerate(HORIZONS):
                    chosen=int(np.argmin(trials[:,j])); saved=b["selected"][j]
                    require(saved["candidate_index"]==chosen,"B selected model is not first minimum")
                    close(saved["mse"],selected_mse[j],"B selected MSE")
                    close(trials[chosen,j],selected_mse[j],"B trial minimum")
                    tables["B_tuning_choices.csv"].append({**identity,"candidate_id":candidate,"method":method,"horizon":h,"selected_index":chosen,
                        "alpha":saved.get("alpha"),"bandwidth_factor":saved.get("bandwidth_factor"),"learning_rate":saved.get("learning_rate"),
                        "max_leaf_nodes":saved.get("max_leaf_nodes"),"B_mse":float(selected_mse[j])})
            else:
                require(np.array_equal(b_prediction,np.repeat(arrays["B_X"][:,-1:],3,axis=1)),"Persistence B formula differs")
            raw=calibration[f"{candidate}__gamma1__{method}"]; calibrated=calibration[f"{candidate}__calibrated__{method}"]
            base=calibration["shared__calibrated__ridge"]
            if method=="ridge":
                require(np.array_equal(raw,base) and np.array_equal(calibrated,base),"C ridge differs")
                gammas[candidate,method]=[0.,0.,0.]
            else:
                saved=record["calibration"]["C"][method]
                require(saved["selection_split"]=="C" and [x["gamma"] for x in saved["candidates"]]==gamma_grid,"C gamma grid/role differs")
                losses=np.array([np.mean((cy-blend(base,raw,[g]*3))**2,axis=0) for g in gamma_grid])
                close([x["mse_by_horizon"] for x in saved["candidates"]],losses,"C gamma losses")
                chosen=np.argmin(losses,axis=0); gamma=[gamma_grid[int(i)] for i in chosen]
                require(saved["selected_indices"]==chosen.tolist() and saved["gamma_by_horizon"]==gamma,"Gamma is not first C minimum")
                require(np.array_equal(calibrated,blend(base,raw,gamma)),"C blend differs")
                require(np.array_equal(forecasts[f"{candidate}__calibrated__{method}"],blend(forecasts["shared__calibrated__ridge"],forecasts[f"{candidate}__gamma1__{method}"],gamma)),"D blend differs")
                gammas[candidate,method]=gamma
                for j,h in enumerate(HORIZONS):
                    tables["C_gamma_choices.csv"].append({**identity,"candidate_id":candidate,"method":method,"horizon":h,"gamma":gamma[j],"C_mse":float(losses[chosen[j],j])})
        if candidate=="shared": continue
        cap,policy=CANDIDATES[candidate]
        require(record["rank_cap"]==cap and record["dimension_policy"]==policy,"Fitted policy differs")
        local=diag["local_dimensions"]; ranks=[]
        for j,item in enumerate(local):
            qkey,pkey=f"{candidate}__model{j}__q",f"{candidate}__model{j}__p"; expected_basis.update((qkey,pkey))
            q,p=basis[qkey],basis[pkey]; n,d=q.shape; rank=p.shape[1]
            require(d==12 and p.shape==(12,rank) and np.isfinite(q).all() and np.isfinite(p).all(),"Local array dimensions differ")
            initial=min(cap,d-int(policy=="legacy_normal_departure"),n-1)
            require(item["effective_rank"]==rank and item["initial_admissible_rank"]==initial and 0<=rank<=initial,"Local admissible rank differs")
            require(item["local_reference_count"]==n and item["requested_cap"]==cap and item["policy"]==policy,"Local dimension metadata differs")
            require(item["reference_sha256"]==array_hash(q) and item["basis_sha256"]==array_hash(p),"Local array hashes differ")
            require(item["policy_source_sha256"]==config["source"]["corrected_policy_sha256"],"Dimension policy source differs")
            require(rank==initial or item["rank_reduction_reasons"],"Unexplained rank reduction")
            if candidate=="historical2": references.append(q)
            else: require(j<len(references) and np.array_equal(q,references[j]),"Cap changed reference selection")
            ranks.append(rank)
            tables["local_dimensions.csv"].append({**identity,"candidate_id":candidate,"model_index":j,"requested_cap":cap,"policy":policy,
                "local_reference_count":n,"initial_admissible_rank":initial,"effective_rank":rank,"fallback_applied":item["fallback_applied"],
                "rank_reduction_reasons":item["rank_reduction_reasons"],"initial_limit_reasons":item["initial_limit_reasons"],
                "range_reduction_steps":item["range_reduction_steps"],"native_failure":item["native_failure"]})
        require(len(local)==len(references),"Local model count differs")
        widths=record["feature_widths"]
        require(widths["common_geometry"]==len(ranks)+sum(ranks) and widths["kahm_departure_geometry"]==widths["rbf_departure_geometry"]==len(ranks)*12,"Geometry feature widths differ")
        tables["geometry_summary_by_task.csv"].append({**identity,"candidate_id":candidate,"n_local_models":len(ranks),"requested_cap":cap,
            "minimum_rank":min(ranks),"mean_rank":float(np.mean(ranks)),"maximum_rank":max(ranks),"full_ambient_models":ranks.count(12),"zero_rank_models":ranks.count(0),
            "common_width":widths["common_geometry"],"departure_width":widths["kahm_departure_geometry"],
            "stacked_basis_rank":diag["bank"]["stack_diagnostics"]["numerical_rank"]})
        for method in GEOMETRY:
            bound=record["bounds"][method]; alpha=np.asarray(bound["alpha_by_horizon"])
            coefficient=np.asarray(bound["centered_response_rms_by_horizon"])/(2*np.sqrt(alpha))
            close(bound["coefficient_norm_bound_by_horizon"],coefficient,"Coefficient bound formula")
            actual=np.asarray(bound["coefficient_norm_by_horizon"])
            violation=int(np.count_nonzero(actual>coefficient*(1+1e-10)+1e-12))
            require(violation==bound["coefficient_bound_violations"]==0,"Coefficient bound violation")
            bounded=not method.endswith("unbounded"); limit=None; correction_ratio=None
            if bounded:
                radius=math.sqrt(widths["full_readout"][method]); close(bound["radius"],radius,"Feature radius")
                require(bound["training_feature_mean_norm"]<=radius*(1+1e-10)+1e-12,"Training feature mean exceeds radius")
                readout=np.abs(bound["response_mean_by_horizon"])+(radius+bound["training_feature_mean_norm"])*coefficient
                limit=np.asarray(gammas[candidate,method])*readout
                close(bound["readout_bound_by_horizon"],readout,"Readout bound formula"); close(bound["correction_bound_by_horizon"],limit,"Correction bound formula")
                maximum=np.max(np.abs(forecasts[f"{candidate}__calibrated__{method}"]-forecasts["shared__calibrated__ridge"]),axis=0)
                require(np.all(maximum<=limit*(1+1e-10)+1e-12),"D correction exceeds deterministic limit")
                require(not np.any((limit==0)&(maximum!=0)),"Nonzero correction at zero limit")
                correction_ratio=float(np.max(np.divide(maximum,limit,out=np.zeros_like(maximum),where=limit!=0)))
            else:
                require(bound["global_bound"] is False and bound["correction_bound_by_horizon"] is None,"Unbounded procedure claims global bound")
            tables["bound_checks.csv"].append({**identity,"candidate_id":candidate,"method":method,"bounded":bounded,
                "coefficient_bound_violations":violation,"maximum_coefficient_bound_ratio":float(np.max(np.divide(actual,coefficient,out=np.zeros_like(actual),where=coefficient!=0))),
                "maximum_D_correction_bound_ratio":correction_ratio,"D_correction_bound_violations":0 if bounded else None})
        stages={"A":diag["features_A_B"]["A"],"B":diag["features_A_B"]["B"],"C":record["calibration"]["query_diagnostics"],"D":query["diagnostics"][candidate]}
        for stage,data in stages.items():
            for method in GEOMETRY:
                feature=data["method_features"][method]
                require(feature["finite"],"Nonfinite features")
                bounded=not method.endswith("unbounded")
                if bounded: require(feature["violations"]==0,"Feature bound violation")
                kind=method.split("_")[0]
                tables["feature_checks.csv"].append({**identity,"candidate_id":candidate,"stage":stage,"method":method,"bounded":bounded,
                    "maximum_norm":feature["max_norm"],"radius":feature.get("radius"),"bound_violations":feature.get("violations"),
                    "fallback_rows":data.get(kind+"_fallback_count",0),
                    "factor_failure_models":sum("factor_failure" in x for x in data["local_models"] if x.get("kind")==kind)})
    require(set(basis)==expected_basis,"Local basis archive roster differs")
    require(fit["matched_candidate_id"]==pre["selections"]["kahm_departure"]["selected_candidate_id"],"Matched bank differs")
    for context,saved in fit["contexts"].items():
        mapping={m:pre["selections"][m]["selected_candidate_id"] if context=="selected" else context for m in GEOMETRY}
        require(saved["method_to_candidate"]==mapping,"Context cap mapping differs")
        policy=saved["policy"]
        require(tuple(policy["candidate_order"])==NONKAHM and policy["selection_split"]=="C" and policy["cap_choices_already_frozen_on_B"] is True,"Policy candidate scope/role differs")
        losses=[float(np.mean(np.mean((cy-calibration[f"context_{context}__calibrated__{m}"])**2,axis=0)/variance)) for m in NONKAHM]
        close([x["normalized_mse"] for x in policy["candidates"]],losses,"C policy losses")
        chosen=NONKAHM[int(np.argmin(losses))]
        require(saved["selected_policy"]==policy["selected_method"]==chosen,"Policy is not first C minimum")
        tables["C_policy_choices.csv"].append({**identity,"context":context,"selected_method":chosen,"C_smse":min(losses)})
        for method,resource in fit["procedure_resources"]["per_context"][context].items():
            measured=resource["fit_and_selection"]; registry=fit["procedure_resources"]["node_registry"]
            require(len(measured["node_ids"])==len(set(measured["node_ids"])),"Repeated charged resource node")
            for unit in ("wall_seconds","cpu_seconds"):
                close(measured[unit],sum(registry[k][unit] for k in measured["node_ids"]),"Resource node sum")
            tables["procedure_resources.csv"].append({**identity,"context":context,"method":method,
                "fit_and_selection_wall_seconds":measured["wall_seconds"],"fit_and_selection_cpu_seconds":measured["cpu_seconds"],
                "deployment_pickle_bytes":resource["state"]["pickle_bytes"],"deployment_array_bytes":resource["state"]["array_buffer_bytes"],
                "accounting":"source dependency nodes; selected geometric procedures include six-cap search"})
    tables["task_resources.csv"].append({**identity,"fit_wall_seconds":fit["resources"]["task_fit_wall_seconds"],
        "fit_cpu_seconds":fit["resources"]["task_fit_cpu_seconds"],"query_wall_seconds":query["resources"]["total_wall_seconds"],
        "query_cpu_seconds":query["resources"]["total_cpu_seconds"],"peak_rss_kib":fit["resources"]["process_peak_rss_kib"]})
    return tables,variance
