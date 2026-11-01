"""Causal staged common forecast plus departure-only residual corrections.

F fits the raw ridge, geometry bank and common-feature statistics. V tunes
the raw ridge penalty, common penalty and common shrinkage sequentially.
The base is then frozen. R fits departure-feature statistics and residual
heads, B selects their penalties, and C calibrates added corrections.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import os
import pickle
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from geometry_adapter import (_implementation, array_sha256, fit_scaler_audit,
                              geometry_query_diagnostics, json_safe)

METHODS = ("common", "common_kahm", "common_rbf", "common_bias")
ADDED = METHODS[1:]
ARMS = ("standard", "minmax")
_MODULE = None


def core():
    global _MODULE
    if _MODULE is None:
        _MODULE, _ = _implementation()
    return _MODULE


def state_hash(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


class GeometryFailure(FloatingPointError):
    def __init__(self, stage, details):
        self.details = {"stage": stage, "geometry_diagnostics": json_safe(details)}
        super().__init__(f"Nonfinite geometry/factor failure at {stage}; task is not silently replaced by the base")


def check_geometry(record, stage):
    # Rank-zero maps and inherited guarded normalized reconstruction/zero
    # departures remain defined, logged behavior. The inherited guard combines
    # finite-value and relative-denominator checks without separating reasons.
    # Invalid raw coordinates/gates/kernels and missing factors are task errors.
    if any(record.get(k) for k in ("coordinate_fallback_rows", "gate_fallback_rows", "kernel_fallback_rows")) or any(
            "factor_failure" in item for item in record.get("local_models", [])):
        raise GeometryFailure(stage, record)


def select_head(xfit, yfit, xselect, yselect, base_select, alphas, nodes, prefix, fit_stage, select_stage):
    m = core()
    with m.timed(nodes, f"{prefix}_fit_{fit_stage}"):
        heads = m._ridge_path(xfit, yfit, alphas)
    with m.timed(nodes, f"{prefix}_select_{select_stage}"):
        losses = np.array([m._mse(yselect, base_select + m._ridge_predict(head, xselect)) for head in heads])
        indices = np.argmin(losses, axis=0)
        head = m._combine_heads(heads, indices)
    return head, {"fit_split": fit_stage, "selection_split": select_stage,
        "alpha_grid": list(alphas), "mse_by_alpha_horizon": losses.tolist(),
        "selected_alpha_by_horizon": head["alpha_by_horizon"].tolist(),
        "selected_indices": indices.tolist(), "feature_width": xfit.shape[1],
        "feature_rank": head["feature_rank"], "tie_rule": "first argmin in declared alpha order"}


def calibrated(y, base, candidate, gammas, nodes, prefix, stage):
    m = core()
    with m.timed(nodes, f"{prefix}_calibrate_{stage}"):
        gamma, forecast, record = m._select_gamma(y, base, candidate, gammas)
    record["selection_split"] = stage
    record["gamma_grid"] = list(gammas)
    record["mse_by_horizon_gamma"] = np.array([v["mse_by_horizon"] for v in record["candidates"]]).T.tolist()
    return gamma, forecast, record


def bounded_features(raw, scaler, nodes, name):
    m = core()
    with m.timed(nodes, name):
        return m.radial_bound(m._standardize(scaler, raw))


def input_diagnostics(x, scaler_audit):
    record = geometry_query_diagnostics(x, scaler_audit)
    return {key.replace("_A_", "_F_"): value for key, value in record.items()} | {
        "scope": "Frozen F history-scaler diagnostic; no R/B/C labels or terminal observations fit this map."}


@dataclass
class CommonForecast:
    input_scaler: dict
    raw_head: dict
    bank: dict
    common_scaler: dict
    common_head: dict
    gamma: np.ndarray

    def raw_state(self):
        return {"input_scaler": self.input_scaler, "raw_head": self.raw_head}

    def evaluate(self, x, nodes, stage):
        m = core()
        with m.timed(nodes, f"base_raw_transform_{stage}"):
            z = m._standardize(self.input_scaler, x)
        with m.timed(nodes, f"base_raw_predict_{stage}"):
            raw_forecast = m._ridge_predict(self.raw_head, z)
        raw_features, diagnostic = m._raw_feature_maps(self.bank, x, nodes, stage)
        check_geometry(diagnostic, stage)
        common = bounded_features(raw_features["common"], self.common_scaler, nodes, f"base_common_transform_{stage}")
        with m.timed(nodes, f"base_common_predict_{stage}"):
            candidate = raw_forecast + m._ridge_predict(self.common_head, common)
            forecast = m._blend(raw_forecast, candidate, self.gamma)
        diagnostic["common_feature_bounds"] = m._feature_summary(common, math.sqrt(common.shape[1]))
        return forecast, raw_features, diagnostic


def base_nodes(nodes, stages):
    return {name for name in nodes if any(name.endswith("_"+stage) for stage in stages)
            and (name.startswith("base_") or name.startswith("common_raw_"))}


def fit_dependencies(nodes, method):
    required = {"bank_structure"} | base_nodes(nodes, ("F", "V"))
    if method == "common":
        return required & set(nodes)
    phases = ("R", "C") if method == "common_bias" else ("R", "B", "C")
    required |= base_nodes(nodes, phases) | {"residual_formation_R"}
    if method == "common_bias":
        required |= {name for name in nodes if name.startswith("bias_")}
    else:
        kind = "kahm" if method == "common_kahm" else "rbf"
        required |= {"bank_projected_metric"}
        if kind == "kahm":
            required.add("bank_kahm_factor")
        required |= {f"projected_kernel_{s}" for s in phases} | {f"{kind}_raw_{s}" for s in phases}
        required |= {name for name in nodes if name.startswith(f"departure_{kind}_") or name.startswith(f"correction_{kind}_")}
    return required & set(nodes)


def predict_dependencies(nodes, method):
    required = base_nodes(nodes, ("query",))
    if method == "common_bias":
        required |= {name for name in nodes if name.startswith("bias_")}
    elif method != "common":
        kind = "kahm" if method == "common_kahm" else "rbf"
        required |= {"projected_kernel_query", f"{kind}_raw_query"}
        required |= {name for name in nodes if name.startswith(f"departure_{kind}_") or name.startswith(f"correction_{kind}_")}
    return required & set(nodes)


def selected_state(base, scalers, heads, bias, gammas, method):
    bank = dict(base.bank)
    if method in ("common", "common_bias"):
        excluded = {"q", "whitening", "uq", "inverse", "lambda_original", "reference_distance", "factor_error"}
    elif method == "common_rbf":
        excluded = {"inverse", "lambda_original", "reference_distance", "factor_error"}
    else:
        excluded = {"reference_distance"}
    bank["models"] = [{k:v for k,v in model.items() if k not in excluded} for model in base.bank["models"]]
    result = {"input_scaler":base.input_scaler,"raw_head":base.raw_head,"bank":bank,
              "common_scaler":base.common_scaler,"common_head":base.common_head,"base_gamma":base.gamma}
    if method == "common_bias":
        result.update(bias=bias, gamma=gammas[method])
    elif method != "common":
        kind = "kahm" if method == "common_kahm" else "rbf"
        result.update(departure_scaler=scalers[kind], head=heads[kind], gamma=gammas[method])
    return result


@dataclass
class FittedFixedCommon:
    base: CommonForecast
    departure_scalers: dict
    heads: dict
    bias: np.ndarray
    gammas: dict
    scaler_audit: dict
    fit_summary: dict
    bounds: dict

    def base_hash(self):
        return state_hash(self.base)

    def predict(self, x):
        m = core()
        x = m._matrix(x, "query histories")
        nodes = {}
        with threadpool_limits(limits=1):
            common, raw, geometry = self.base.evaluate(x, nodes, "query")
            uncalibrated, predictions = {"common":common.copy()}, {"common":common.copy()}
            feature_diagnostics = {}
            for kind in ("kahm", "rbf"):
                method = "common_"+kind
                features = bounded_features(raw[kind], self.departure_scalers[kind], nodes, f"departure_{kind}_transform_query")
                with m.timed(nodes, f"correction_{kind}_predict_query"):
                    uncalibrated[method] = common + m._ridge_predict(self.heads[kind], features)
                with m.timed(nodes, f"correction_{kind}_blend_query"):
                    predictions[method] = m._blend(common, uncalibrated[method], self.gammas[method])
                feature_diagnostics[method] = m._feature_summary(features, math.sqrt(features.shape[1]))
            with m.timed(nodes, "bias_predict_query"):
                uncalibrated["common_bias"] = common + self.bias
            with m.timed(nodes, "bias_blend_query"):
                predictions["common_bias"] = m._blend(common, uncalibrated["common_bias"], self.gammas["common_bias"])
        for container in (predictions, uncalibrated):
            if tuple(container) != METHODS or any(v.shape!=common.shape or not np.isfinite(v).all() for v in container.values()):
                raise FloatingPointError("Incomplete or nonfinite forecast; no silent common fallback")
        bound_diagnostics = {}
        for method in ADDED:
            observed = np.max(np.abs(predictions[method]-common), axis=0)
            limits = np.asarray(self.bounds[method]["correction_bound_by_horizon"])
            bound_diagnostics[method] = {**self.bounds[method], "max_abs_correction_by_horizon":observed.tolist(),
                "violations":int(np.count_nonzero(observed > limits*(1+1e-10)+1e-12))}
        return {"predictions":predictions,"uncalibrated_predictions":uncalibrated,
            "diagnostics":json_safe({"geometry":geometry,"geometry_inputs":input_diagnostics(x,self.scaler_audit),
                "departure_feature_bounds":feature_diagnostics,"bounds":bound_diagnostics,
                "recoveries":{method:0 for method in METHODS}}),
            "resources":json_safe({"predict_nodes":nodes,
                "per_method":{method:{"predict":m._node_cost(nodes,predict_dependencies(nodes,method))} for method in METHODS}})}


def fit_fixed_common_suite(xf,yf,xv,yv,xr,yr,xb,yb,xc,yc,*,seed,arm,horizons):
    """Fit the five-stage procedure; no terminal target argument exists."""
    if arm not in ARMS:
        raise ValueError("Only standard and unclipped minmax are supported")
    m, provenance = _implementation()
    global _MODULE
    _MODULE = m
    values = [m._matrix(x,name) for x,name in zip((xf,yf,xv,yv,xr,yr,xb,yb,xc,yc),
        ("X_F","Y_F","X_V","Y_V","X_R","Y_R","X_B","Y_B","X_C","Y_C"))]
    xs = dict(zip(("F","V","R","B","C"),values[::2]))
    ys = dict(zip(("F","V","R","B","C"),values[1::2]))
    horizons = tuple(int(h) for h in horizons)
    if len(horizons)!=3 or any(h<=0 for h in horizons) or len(set(horizons))!=3:
        raise ValueError("Require three declared positive offsets")
    if len({x.shape[1] for x in xs.values()})!=1 or any(len(xs[s])!=len(ys[s]) or ys[s].shape[1]!=3 for s in xs):
        raise ValueError("Stage matrix shapes disagree")
    config = copy.deepcopy(m.DEFAULT_CONFIG)
    alphas, gamma_grid = config["alpha_grid"], config["gamma_grid"]
    nodes, geometry_records, feature_records, base_selection = {}, {}, {}, {}
    start_w,start_c = time.perf_counter(),time.process_time()
    with threadpool_limits(limits=1):
        runtime_threads = threadpool_info()
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None
        with m.timed(nodes,"base_input_scaler_F"):
            input_scaler = m._scaler_fit(xs["F"],config["feature_scale_floor"])
        z = {}
        for s in ("F","V"):
            with m.timed(nodes,f"base_raw_transform_{s}"):
                z[s] = m._standardize(input_scaler,xs[s])
        raw_head,base_selection["raw_ridge"] = select_head(z["F"],ys["F"],z["V"],ys["V"],np.zeros_like(ys["V"]),
                                                          alphas,nodes,"base_raw","F","V")
        raw_forecasts = {}
        for s in ("F","V"):
            with m.timed(nodes,f"base_raw_predict_{s}"):
                raw_forecasts[s] = m._ridge_predict(raw_head,z[s])
        bank = m._InstrumentedBank(seed=int(seed),config=config,geometry_scaling=arm).fit(xs["F"])
        nodes.update(bank.nodes)
        if any("factor_error" in r for r in bank.diagnostics["local_models"]):
            raise GeometryFailure("F_factor_fit",bank.diagnostics)
        bank_state = bank.state("kahm")
        for model in bank_state["models"]:
            model.pop("reference_distance",None)
        raw_maps = {}
        for s in ("F","V"):
            raw_maps[s],geometry_records[s] = m._raw_feature_maps(bank_state,xs[s],nodes,s)
            check_geometry(geometry_records[s],s)
        with m.timed(nodes,"base_common_scaler_F"):
            common_scaler = m._scaler_fit(raw_maps["F"]["common"],config["feature_scale_floor"])
        common_features = {s:bounded_features(raw_maps[s]["common"],common_scaler,nodes,f"base_common_transform_{s}") for s in ("F","V")}
        with m.timed(nodes,"base_common_residual_F"):
            common_residual = ys["F"]-raw_forecasts["F"]
        common_head,base_selection["common_head"] = select_head(common_features["F"],common_residual,
            common_features["V"],ys["V"],raw_forecasts["V"],alphas,nodes,"base_common","F","V")
        with m.timed(nodes,"base_common_predict_V"):
            common_candidate_v = raw_forecasts["V"]+m._ridge_predict(common_head,common_features["V"])
        base_gamma,_,base_selection["common_calibration"] = calibrated(ys["V"],raw_forecasts["V"],common_candidate_v,
            gamma_grid,nodes,"base_common","V")
        base = CommonForecast(input_scaler,raw_head,bank_state,common_scaler,common_head,base_gamma)
        base_before = state_hash(base)
        raw_ridge_hash = state_hash(base.raw_state())
        # Nothing below changes the frozen base or its F-only bank/scalers.
        bases = {}
        for s in ("R","B","C"):
            bases[s],raw_maps[s],geometry_records[s] = base.evaluate(xs[s],nodes,s)
        with m.timed(nodes,"residual_formation_R"):
            residual_r = ys["R"]-bases["R"]
        scalers,heads,head_selection,gammas,calibration,bounds = {},{},{},{},{},{}
        for kind in ("kahm","rbf"):
            method = "common_"+kind
            with m.timed(nodes,f"departure_{kind}_scaler_R"):
                scalers[kind] = m._scaler_fit(raw_maps["R"][kind],config["feature_scale_floor"])
            features = {s:bounded_features(raw_maps[s][kind],scalers[kind],nodes,f"departure_{kind}_transform_{s}") for s in ("R","B","C")}
            feature_records[kind] = {s:m._feature_summary(features[s],math.sqrt(features[s].shape[1])) for s in features}
            heads[kind],head_selection[method] = select_head(features["R"],residual_r,features["B"],ys["B"],bases["B"],
                alphas,nodes,f"correction_{kind}","R","B")
            with m.timed(nodes,f"correction_{kind}_candidate_C"):
                candidate_c = bases["C"]+m._ridge_predict(heads[kind],features["C"])
            gammas[method],_,calibration[method] = calibrated(ys["C"],bases["C"],candidate_c,gamma_grid,nodes,f"correction_{kind}","C")
            bounds[method] = m._readout_bound(heads[kind],math.sqrt(features["R"].shape[1]),gammas[method])
        with m.timed(nodes,"bias_fit_R"):
            bias = residual_r.mean(axis=0)
        with m.timed(nodes,"bias_candidate_C"):
            bias_candidate = bases["C"]+bias
        gammas["common_bias"],_,calibration["common_bias"] = calibrated(ys["C"],bases["C"],bias_candidate,gamma_grid,nodes,"bias","C")
        bounds["common_bias"] = {"global_bound":True,"correction_bound_by_horizon":(np.abs(bias)*gammas["common_bias"]).tolist(),
                                   "scope":"Exact magnitude of the constant added correction"}
        base_after = state_hash(base)
        if base_before!=base_after:
            raise AssertionError("Residual fitting mutated the common forecast")
    elapsed = {"wall_seconds":time.perf_counter()-start_w,"cpu_seconds":time.process_time()-start_c}
    scaler_audit = fit_scaler_audit(xs["F"],arm)
    scaler_audit["fitting_split"] = "F"
    if arm=="minmax":
        actual = base.bank["geometry_scaler"]
        if actual["fit_rows"]!=len(xs["F"]):
            raise AssertionError("History scaler used a wrong stage")
        for key in ("minimum","maximum","range","constant"):
            np.testing.assert_array_equal(actual[key],scaler_audit[key])
    base_selection.update(raw_ridge_alpha_by_horizon=raw_head["alpha_by_horizon"].tolist(),
        common_alpha_by_horizon=common_head["alpha_by_horizon"].tolist(),common_gamma_by_horizon=base_gamma.tolist(),
        scope="Sequential raw-penalty, common-penalty and common-gamma tuning on the same V block; not independent base calibration.")
    resources = {"fit_nodes":nodes,"per_method":{method:{
        "fit_and_selection":m._node_cost(nodes,fit_dependencies(nodes,method)),
        "state":m._measure_state(selected_state(base,scalers,heads,bias,gammas,method))} for method in METHODS},
        "accounting":"Required shared nodes charged once per method; unused F/V departure calculations from the shared feature routine are not charged. Selected-state bytes are dependency dictionaries, not process RSS or exported fitted models."}
    summary = json_safe({"schema":"fixed-common-adapter-v1","arm":arm,"seed":int(seed),"horizons":horizons,
        "upstream":provenance,"model_config":config,"methods":METHODS,
        "stage_rows":{s:len(xs[s]) for s in xs},"history":xs["F"].shape[1],
        "input_array_sha256":{name:array_sha256(value) for name,value in zip(
            ("X_F","Y_F","X_V","Y_V","X_R","Y_R","X_B","Y_B","X_C","Y_C"),values)},
        "base_selection":base_selection,"head_selection":head_selection,"correction_calibration":calibration,
        "base_state_hashes":{"before_residual_fit":base_before,"after_residual_fit":base_after},
        "raw_ridge_state_sha256":raw_ridge_hash,
        "geometry_scaling":scaler_audit,"geometry_inputs":{s:input_diagnostics(xs[s],scaler_audit) for s in xs},
        "bank":bank.diagnostics,"raw_geometry":geometry_records,"departure_feature_bounds":feature_records,
        "common_feature_scaler_F":common_scaler,"departure_feature_scalers_R":scalers,
        "departure_features_only":True,"departure_feature_widths":{k:int(v["feature_width"]) for k,v in heads.items()},
        "common_feature_width":int(common_head["feature_width"]),"bias_R_mean_residual":bias,
        "bounds":bounds,"base_common_bound":m._readout_bound(common_head,math.sqrt(common_head["feature_width"]),base_gamma),
        "resources":resources,"whole_fit_including_diagnostics":elapsed,
        "runtime_threadpool_libraries":runtime_threads,"runtime_cpu_affinity":affinity,
        "failure_policy":"Ordinary factor/head/nonfinite-query failures stop the task. Inherited rank-zero maps and guarded normalized-reconstruction/zero-departure rules remain defined and separately logged; the source guard combines finite-value and relative-denominator checks without separately identifying their causes.",
        "scope":"F/V-frozen common forecast, departure-only R residual heads, B penalties, C added-correction calibration; no post-C refit and no terminal target input."})
    return FittedFixedCommon(base,scalers,heads,bias,gammas,scaler_audit,summary,bounds)
