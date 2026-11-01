"""Chronological residuals, a B-frozen shared bias, and centered geometry.

The historical ridge path already centers both its design and response.  We
reuse its slope coefficients but explicitly omit its response mean when
predicting the new geometric correction.  The optional latest-origin legacy
replay restores that mean and reuses the same fitted seven-alpha paths.
No evaluation-target argument exists in this module's fit or predict APIs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
import time
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from geometry_adapter import _implementation, array_sha256, fit_scaler_audit, json_safe
from fixed_common_adapter import (CommonForecast, GeometryFailure, bounded_features,
    calibrated, check_geometry, input_diagnostics, select_head, state_hash)

METHODS = ("common", "shared_bias", "centered_kahm", "centered_rbf")
GEOMETRIC = METHODS[2:]
LEGACY_METHODS = ("common", "common_kahm", "common_rbf", "common_bias",
                  "common_kahm_extended", "common_rbf_extended", "early_common")
ARMS = ("standard", "minmax", "tanh_z_1", "tanh_z_0p5", "tanh_z_2", "tanh_raw")
ALPHAS = (1e-4, .001, .01, .1, 1., 10., 100.)
GAMMAS = (0., .25, .5, 1.)
STAGES = ("F", "V", "R", "B", "C", "A", "P1", "P2", "P3")


def slope_predict(head, bounded):
    """No free constant term: centered bounded features times ridge slopes."""
    out = (bounded - head["mean"]) @ head["coef"]
    if not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite centered geometric correction")
    return out


def fixed_alpha_refit(m, x, y, selected):
    grid = sorted(set(float(v) for v in selected))
    path = m._ridge_path(x, y, grid)
    return m._combine_heads(path, [grid.index(float(v)) for v in selected])


def choose_path(m, path, features_b, yb, base_b, *, centered, count=7):
    candidates = np.stack([base_b + (slope_predict(h, features_b) if centered else
                                  m._ridge_predict(h, features_b)) for h in path[:count]])
    losses = np.array([m._mse(yb, p) for p in candidates])
    indices = np.argmin(losses, axis=0)
    head = m._combine_heads(path[:count], indices)
    return head, candidates, {
        "fit_split":"R", "selection_split":"B", "alpha_grid":list(ALPHAS[:count]),
        "mse_by_alpha_horizon":losses.tolist(), "selected_indices":indices.tolist(),
        "selected_alpha_by_horizon":head["alpha_by_horizon"].tolist(),
        "feature_width":int(head["feature_width"]), "feature_rank":int(head["feature_rank"]),
        "centered_geometry":bool(centered), "tie_rule":"first argmin in declared alpha order"}


def centered_bound(head, eta):
    radius = math.sqrt(head["feature_width"])
    mean_norm = float(np.linalg.norm(head["mean"]))
    centered_radius = radius + mean_norm
    coefficient_bound = head["response_rms"] / (2*np.sqrt(head["alpha_by_horizon"]))
    coefficient_norm = np.linalg.norm(head["coef"], axis=0)
    readout_bound = centered_radius * coefficient_bound
    return {"global_bound":True, "radius_before_centering":radius,
        "bounded_training_feature_mean_norm":mean_norm, "centered_radius":centered_radius,
        "alpha_by_horizon":head["alpha_by_horizon"].tolist(),
        "centered_response_rms_by_horizon":head["response_rms"].tolist(),
        "coefficient_norm_by_horizon":coefficient_norm.tolist(),
        "coefficient_norm_bound_by_horizon":coefficient_bound.tolist(),
        "coefficient_bound_violations":int(np.count_nonzero(coefficient_norm > coefficient_bound*(1+1e-10)+1e-12)),
        "readout_bound_by_horizon":readout_bound.tolist(),
        "correction_bound_by_horizon":(np.abs(eta)*readout_bound).tolist(),
        "scope":"Nonconstant geometry only, relative to the frozen shared-bias predictor; centering follows radial bounding."}


def validate_inputs(m, xs, ys, origins, folds, horizons):
    if set(xs)!=set(STAGES) or set(ys)!=set(STAGES) or set(origins)!=set(STAGES):
        raise ValueError("Require exactly nine declared training stages")
    x = {s:m._matrix(xs[s],"X_"+s) for s in STAGES}
    y = {s:m._matrix(ys[s],"Y_"+s) for s in STAGES}
    o = {s:np.asarray(origins[s]) for s in STAGES}
    horizons = np.asarray(horizons,dtype=int)
    if horizons.shape!=(3,) or np.any(horizons<=0) or len(set(horizons))!=3:
        raise ValueError("Require three distinct positive horizons")
    if len({v.shape[1] for v in x.values()})!=1:
        raise ValueError("History widths differ")
    for s in STAGES:
        if len(x[s])!=len(y[s]) or y[s].shape[1]!=3 or o[s].shape!=(len(x[s]),):
            raise ValueError("Stage matrix dimensions differ")
        if not np.issubdtype(o[s].dtype,np.integer) or np.any(np.diff(o[s])<=0):
            raise ValueError("Origins must be increasing integers")
    fold_ids = np.asarray(folds)
    expected = np.empty(len(x["R"]),dtype=int)
    chronology = {}
    for j,ids in enumerate(np.array_split(np.arange(len(expected)),3),1):
        if len(ids)<2:
            raise ValueError("Each chronological residual block needs at least two rows")
        expected[ids]=j
        stage=f"P{j}"
        first=int(o["R"][ids[0]])
        eligible=o["A"]+int(np.max(horizons))<=first
        for values,name in ((x,"X"),(y,"Y"),(o,"origins")):
            if not np.array_equal(values[stage],values["A"][eligible]):
                raise ValueError(f"{stage} must contain exactly every mature A window: {name}")
        if len(x[stage])>512 or len(x[stage])<2:
            raise ValueError("Prefix cap must be inactive and at least two rows available")
        last=int(np.max(o[stage])+np.max(horizons))
        v_last=int(np.max(o["V"])+np.max(horizons))
        if last>first or v_last>first:
            raise ValueError("Prefix fitting or V tuning used an unavailable target")
        chronology[stage]={"first_R_origin":first,"last_fit_target":last,
            "last_V_target":v_last,"fit_maturity_margin":first-last,
            "V_maturity_margin":first-v_last,"fit_rows":len(x[stage]),"R_rows":len(ids)}
    if not np.array_equal(fold_ids,expected):
        raise ValueError("Residual fold identifiers differ from consecutive array_split groups")
    if len(x["A"])>512:
        raise ValueError("A fitting cap must be inactive")
    return x,y,o,expected,tuple(int(h) for h in horizons),chronology


@dataclass
class FittedCentered:
    base: CommonForecast
    early: CommonForecast
    prefixes: list
    departure_scalers: dict
    heads: dict
    bias: np.ndarray
    bias_gamma: np.ndarray
    etas: dict
    scaler_audit: dict
    fit_summary: dict
    bounds: dict
    fit_arrays: dict
    legacy_heads: dict
    legacy_gammas: dict

    def predictive_state(self):
        return {"base":self.base,"early":self.early,"prefixes":self.prefixes,
            "departure_scalers":self.departure_scalers,"heads":self.heads,
            "bias":self.bias,"bias_gamma":self.bias_gamma,"etas":self.etas,
            "legacy_heads":self.legacy_heads,"legacy_gammas":self.legacy_gammas}

    def state_hash(self):
        return state_hash(self.predictive_state())

    def shared_bias_hash(self):
        return state_hash({"base":self.base,"bias":self.bias,"bias_gamma":self.bias_gamma})

    def predict(self,x,*,fold_diagnostics=False):
        m,_=_implementation()
        x=m._matrix(x,"query histories")
        nodes={}
        with threadpool_limits(limits=1):
            common,raw,geometry=self.base.evaluate(x,nodes,"query")
            shared=m._blend(common,common+self.bias,self.bias_gamma)
            predictions={"common":common.copy(),"shared_bias":shared.copy()}
            eta1={"common":common.copy(),"shared_bias":shared.copy()}
            features,feature_records,bound_records={},{},{}
            for kind in ("kahm","rbf"):
                method="centered_"+kind
                features[kind]=bounded_features(raw[kind],self.departure_scalers[kind],nodes,f"centered_{kind}_transform_query")
                with m.timed(nodes,f"centered_{kind}_predict_query"):
                    correction=slope_predict(self.heads[kind],features[kind])
                    eta1[method]=shared+correction
                    predictions[method]=m._blend(shared,eta1[method],self.etas[method])
                centered=features[kind]-self.heads[kind]["mean"]
                feature_records[method]=m._feature_summary(centered,self.bounds[method]["centered_radius"])
                limit=np.asarray(self.bounds[method]["correction_bound_by_horizon"])
                observed=np.max(np.abs(predictions[method]-shared),axis=0)
                bound_records[method]={**self.bounds[method],"max_abs_correction_by_horizon":observed.tolist(),
                    "violations":int(np.count_nonzero(observed>limit*(1+1e-10)+1e-12)),
                    "mean_unscaled_correction_by_horizon":correction.mean(axis=0).tolist()}
            legacy,legacy1={},{}
            if self.legacy_heads:
                legacy["common"]=common.copy();legacy1["common"]=common.copy()
                for method in LEGACY_METHODS[1:-1]:
                    if method=="common_bias":
                        candidate=common+self.bias
                    else:
                        kind="kahm" if "kahm" in method else "rbf"
                        candidate=common+m._ridge_predict(self.legacy_heads[method],features[kind])
                    legacy1[method]=candidate
                    legacy[method]=m._blend(common,candidate,self.legacy_gammas[method])
            transfer={}
            if fold_diagnostics or self.legacy_heads:
                early,_,_=self.early.evaluate(x,nodes,"early_query")
                transfer["early_common"]=early
                if self.legacy_heads:
                    legacy["early_common"]=early.copy();legacy1["early_common"]=early.copy()
            if fold_diagnostics:
                transfer["fold_common"]=np.stack([p.evaluate(x,nodes,f"P{j}_query")[0]
                    for j,p in enumerate(self.prefixes,1)])
        for container,names in ((predictions,METHODS),(eta1,METHODS),(legacy,LEGACY_METHODS),(legacy1,LEGACY_METHODS)):
            if not container and names==LEGACY_METHODS: continue
            if tuple(container)!=names or any(v.shape!=common.shape or not np.isfinite(v).all() for v in container.values()):
                raise FloatingPointError("Incomplete/nonfinite forecast roster; no silent replacement")
        return {"predictions":predictions,"eta1_predictions":eta1,
            "legacy_predictions":legacy,"legacy_gamma1_predictions":legacy1,"transfer":transfer,
            "diagnostics":json_safe({"geometry":geometry,"geometry_inputs":input_diagnostics(x,self.scaler_audit),
                "centered_feature_bounds":feature_records,"bounds":bound_records,
                "recoveries":{method:0 for method in METHODS}}),"resources":{"predict_nodes":nodes}}


def fit_centered_suite(xs,ys,origins,fold_id_R,*,seed,arm,horizons,legacy_replay=False,config=None):
    if arm not in ARMS: raise ValueError("Unknown geometry scaler")
    m,provenance=_implementation()
    # Imported historical helpers call their own module-global core function.
    import fixed_common_adapter as historical
    historical._MODULE=m
    xs,ys,origins,folds,horizons,chronology=validate_inputs(m,xs,ys,origins,fold_id_R,horizons)
    config=m.dimension_config(m.DEFAULT_CONFIG,config)
    nodes,raw_maps,geometry_records,base_selection={},{},{},{}
    start_w,start_c=time.perf_counter(),time.process_time()
    with threadpool_limits(limits=1):
        runtime_threads=threadpool_info()
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None
        with m.timed(nodes,"base_input_scaler_F"):
            input_scaler=m._scaler_fit(xs["F"],config["feature_scale_floor"])
        z={s:m._standardize(input_scaler,xs[s]) for s in ("F","V")}
        raw_head,base_selection["raw_ridge"]=select_head(z["F"],ys["F"],z["V"],ys["V"],np.zeros_like(ys["V"]),
            config["alpha_grid"],nodes,"base_raw","F","V")
        raw_forecast={s:m._ridge_predict(raw_head,z[s]) for s in ("F","V")}
        bank=m._InstrumentedBank(seed=int(seed),config=config,geometry_scaling=arm).fit(xs["F"])
        nodes.update(bank.nodes)
        if any("factor_error" in r for r in bank.diagnostics["local_models"]):
            raise GeometryFailure("F_factor_fit",bank.diagnostics)
        bank_state=bank.state("kahm")
        for model in bank_state["models"]: model.pop("reference_distance",None)
        for s in ("F","V"):
            raw_maps[s],geometry_records[s]=m._raw_feature_maps(bank_state,xs[s],nodes,s)
            check_geometry(geometry_records[s],s)
        common_scaler=m._scaler_fit(raw_maps["F"]["common"],config["feature_scale_floor"])
        common_features={s:bounded_features(raw_maps[s]["common"],common_scaler,nodes,"base_common_transform_"+s) for s in ("F","V")}
        common_head,base_selection["common_head"]=select_head(common_features["F"],ys["F"]-raw_forecast["F"],
            common_features["V"],ys["V"],raw_forecast["V"],config["alpha_grid"],nodes,"base_common","F","V")
        base_gamma,_,base_selection["common_calibration"]=calibrated(ys["V"],raw_forecast["V"],
            raw_forecast["V"]+m._ridge_predict(common_head,common_features["V"]),GAMMAS,nodes,"base_common","V")
        early=CommonForecast(input_scaler,raw_head,bank_state,common_scaler,common_head,base_gamma)
        early_hash=state_hash(early)
        transform_hash=state_hash({"input_scaler":input_scaler,"bank":bank_state,"common_scaler":common_scaler})

        def refit(stage):
            with m.timed(nodes,"readout_refit_"+stage):
                zz=m._standardize(input_scaler,xs[stage])
                rh=fixed_alpha_refit(m,zz,ys[stage],raw_head["alpha_by_horizon"])
                rr=m._ridge_predict(rh,zz)
                maps,record=m._raw_feature_maps(bank_state,xs[stage],nodes,stage)
                check_geometry(record,stage);geometry_records[stage]=record
                ff=bounded_features(maps["common"],common_scaler,nodes,"refit_common_transform_"+stage)
                ch=fixed_alpha_refit(m,ff,ys[stage]-rr,common_head["alpha_by_horizon"])
            return CommonForecast(input_scaler,rh,bank_state,common_scaler,ch,base_gamma)

        prefixes=[refit(f"P{j}") for j in (1,2,3)]
        prefix_hashes=[state_hash(p) for p in prefixes]
        oof=np.empty_like(ys["R"])
        for j,prefix in enumerate(prefixes,1):
            ids=np.flatnonzero(folds==j)
            oof[ids],_,geometry_records[f"P{j}_assigned_R"]=prefix.evaluate(xs["R"][ids],nodes,f"P{j}_assigned_R")
        base=refit("A")
        base_before=state_hash(base)
        bases,early_predictions={},{}
        for s in ("R","B","C"):
            bases[s],raw_maps[s],geometry_records[s]=base.evaluate(xs[s],nodes,s)
            early_predictions[s]=early.evaluate(xs[s],nodes,"early_"+s)[0]
        residual=ys["R"]-oof
        bias=residual.mean(axis=0)
        bias_gamma,shared_b,bias_selection=calibrated(ys["B"],bases["B"],bases["B"]+bias,GAMMAS,nodes,"shared_bias","B")
        shared={"B":shared_b,"R":m._blend(bases["R"],bases["R"]+bias,bias_gamma),
                "C":m._blend(bases["C"],bases["C"]+bias,bias_gamma)}
        shared_hash=state_hash({"base":base,"bias":bias,"bias_gamma":bias_gamma})
        scalers,heads,etas,selection,calibration,bounds,feature_records={},{},{},{},{},{},{}
        legacy_heads,legacy_gammas,legacy_selection,legacy_calibration={},{},{},{}
        arrays={"R_oof_common":oof,"R_oof_residual":residual,"R_final_common":bases["R"],
            "R_early_common":early_predictions["R"],"fold_id_R":folds,"origins_R":origins["R"],
            "B_bias_candidates":np.stack([m._blend(bases["B"],bases["B"]+bias,np.full(3,g)) for g in GAMMAS])}
        for s in ("B","C"):
            arrays[s+"_final_common"]=bases[s];arrays[s+"_shared_bias"]=shared[s]
            arrays[s+"_early_common"]=early_predictions[s]
            arrays["fold_common_"+s]=np.stack([p.evaluate(xs[s],nodes,f"P{j}_{s}")[0] for j,p in enumerate(prefixes,1)])
        for kind in ("kahm","rbf"):
            method="centered_"+kind
            scalers[kind]=m._scaler_fit(raw_maps["R"][kind],config["feature_scale_floor"])
            features={s:bounded_features(raw_maps[s][kind],scalers[kind],nodes,f"departure_{kind}_transform_{s}") for s in ("R","B","C")}
            with m.timed(nodes,f"departure_{kind}_seven_alpha_fit_R"):
                path=m._ridge_path(features["R"],residual,ALPHAS)
            with m.timed(nodes,f"centered_{kind}_select_B"):
                heads[kind],arrays["B_"+method+"_candidates"],selection[method]=choose_path(m,path,features["B"],ys["B"],shared["B"],centered=True)
            candidate=shared["C"]+slope_predict(heads[kind],features["C"])
            etas[method],_,calibration[method]=calibrated(ys["C"],shared["C"],candidate,GAMMAS,nodes,method,"C")
            bounds[method]=centered_bound(heads[kind],etas[method])
            feature_records[method]={"bounded_training_feature_mean":heads[kind]["mean"].tolist(),
                "residual_training_mean_by_horizon":heads[kind]["ymean"].tolist(),
                "centered_residual_training_rms_by_horizon":heads[kind]["response_rms"].tolist(),
                "centered_feature_training_mean_max_abs":float(np.max(np.abs((features["R"]-heads[kind]["mean"]).mean(axis=0)))),
                "separate_intercept":False,"zero_mean_scope":"R training rows only; query means need not vanish.",
                "stages":{s:{"bounded":m._feature_summary(features[s],math.sqrt(features[s].shape[1])),
                    "centered":m._feature_summary(features[s]-heads[kind]["mean"],bounds[method]["centered_radius"])} for s in features}}
            for s in ("R","B","C"):
                correction=slope_predict(heads[kind],features[s])
                arrays[s+"_"+method+"_correction"]=correction
                feature_records[method]["stages"][s]["mean_unscaled_correction_by_horizon"]=correction.mean(axis=0).tolist()
            if legacy_replay:
                for suffix,count in (("",5),("_extended",7)):
                    name="common_"+kind+suffix
                    legacy_heads[name],candidates,legacy_selection[name]=choose_path(m,path,features["B"],ys["B"],bases["B"],centered=False,count=count)
                    if count==7: arrays["B_legacy_"+kind+"_candidates"]=candidates
                    candidate=bases["C"]+m._ridge_predict(legacy_heads[name],features["C"])
                    legacy_gammas[name],_,legacy_calibration[name]=calibrated(ys["C"],bases["C"],candidate,GAMMAS,nodes,"legacy_"+name,"C")
        if legacy_replay:
            legacy_gammas["common_bias"],_,legacy_calibration["common_bias"]=calibrated(ys["C"],bases["C"],bases["C"]+bias,GAMMAS,nodes,"legacy_bias","C")
        after=state_hash(base)
        if base_before!=after or early_hash!=state_hash(early) or prefix_hashes!=[state_hash(p) for p in prefixes]:
            raise AssertionError("Residual fitting mutated a common/prefix model")
        if transform_hash!=state_hash({"input_scaler":input_scaler,"bank":bank_state,"common_scaler":common_scaler}):
            raise AssertionError("F-fitted transforms changed")
        if shared_hash!=state_hash({"base":base,"bias":bias,"bias_gamma":bias_gamma}):
            raise AssertionError("Geometry selection changed shared bias")
    scaler_audit=fit_scaler_audit(xs["F"],arm);scaler_audit["fitting_split"]="F"
    if arm=="minmax":
        actual=base.bank["geometry_scaler"]
        if actual["fit_rows"]!=len(xs["F"]): raise AssertionError("Scaler fitted on wrong stage")
        for key in ("minimum","maximum","range","constant"):
            np.testing.assert_array_equal(actual[key],scaler_audit[key])
    if arm.startswith("tanh_"):
        from extensions.geometry_scaling import transform_geometry
        from geometry_adapter import scaled_histories
        actual=base.bank["geometry_scaler"]
        if actual["fit_rows"]!=len(xs["F"]) or actual["mode"]!=arm:
            raise AssertionError("Tanh scaler fitted on wrong stage or mode")
        if actual["post_tanh_rescale"] is not False:
            raise AssertionError("Tanh output unexpectedly rescaled")
        if arm != "tanh_raw":
            for key,other in (("mean","mean"),("raw_standard_deviation","raw_standard_deviation"),("scale","legacy_standard_scale")):
                np.testing.assert_array_equal(actual[key],scaler_audit[other])
        for stage in ("F","R","B","C"):
            np.testing.assert_array_equal(transform_geometry(xs[stage],actual),scaled_histories(xs[stage],scaler_audit))
    base_selection.update(raw_ridge_alpha_by_horizon=raw_head["alpha_by_horizon"].tolist(),
        common_alpha_by_horizon=common_head["alpha_by_horizon"].tolist(),common_gamma_by_horizon=base_gamma.tolist())
    summary=json_safe({"schema":"centered-geometry-dimension-corrected-v1","arm":arm,"seed":int(seed),"horizons":horizons,
        "upstream":provenance,"model_config":config,"alpha_grid":ALPHAS,"gamma_grid":GAMMAS,"methods":METHODS,
        "stage_rows":{s:len(xs[s]) for s in STAGES},"history":xs["F"].shape[1],"chronology":chronology,
        "base_selection":base_selection,"bias_selection":bias_selection,"head_selection":selection,
        "correction_calibration":calibration,"bias_R_mean_residual":bias,"bias_gamma_by_horizon":bias_gamma,
        "shared_bias_by_horizon":bias*bias_gamma,"base_state_hashes":{"before_residual_fit":base_before,"after_residual_fit":after},
        "early_state_sha256":early_hash,"prefix_state_sha256":prefix_hashes,"F_transform_sha256":transform_hash,
        "shared_bias_state_sha256":shared_hash,"early_raw_ridge_sha256":state_hash(early.raw_state()),
        "final_raw_ridge_sha256":state_hash(base.raw_state()),"geometry_scaling":scaler_audit,
        "geometry_inputs":{s:input_diagnostics(xs[s],scaler_audit) for s in STAGES},
        "bank":bank.diagnostics,"raw_geometry":geometry_records,"centered_feature_diagnostics":feature_records,
        "common_feature_scaler_F":common_scaler,"departure_feature_scalers_R":scalers,
        "bounds":bounds,"common_feature_width":int(common_head["feature_width"]),
        "departure_feature_widths":{kind:int(heads[kind]["feature_width"]) for kind in heads},
        "legacy":{"enabled":bool(legacy_replay),"methods":LEGACY_METHODS if legacy_replay else (),
            "head_selection":legacy_selection,"correction_calibration":legacy_calibration,
            "gamma_by_horizon":legacy_gammas,"scope":"Recovery replay using the same fitted slopes; not a new primary comparison."},
        "fit_nodes":nodes,"whole_fit_including_diagnostics":{"wall_seconds":time.perf_counter()-start_w,"cpu_seconds":time.process_time()-start_c},
        "runtime_threadpool_libraries":runtime_threads,"runtime_cpu_affinity":affinity,
        "tuning_reuse":"B first selects shared bias gamma; after freezing it, B selects geometric alpha. C selects geometric eta only. B is intentionally reused.",
        "centered_fit_identity":"The inherited ridge path centers bounded R features and R OOF residual targets. Subtracting any fixed shared bias before response centering gives the same target. Predictions omit ymean explicitly; no new intercept is fitted.",
        "failure_policy":"Ordinary factor/head/nonfinite failures stop the task; inherited rank-zero and guarded reconstruction cases remain defined and separately logged.",
        "scope":"F-fitted geometry/scalers and V-selected common hyperparameters; chronological P1/P2/P3 residuals; A-refitted common; fixed shared bias; centered geometry; no post-C refit."})
    return FittedCentered(base,early,prefixes,scalers,heads,bias,bias_gamma,etas,scaler_audit,summary,bounds,arrays,legacy_heads,legacy_gammas)
