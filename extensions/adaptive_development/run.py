"""Causal multi-horizon replay, immutable forecasts, and serial subprocess runs.

Only this harness reads the complete raw input. Model calls receive prefix
training pairs, newly matured labels, and the currently observed history.
Calibration selects a fixed mixing weight from previously issued components;
the updating algorithm is then evaluated without changing its decision rules.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
import zipfile

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .data import load_tasks, make_synthetic
from .models import OnlineForecaster

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("ridge", "fixed_kahm", "adaptive_kahm", "adaptive_affine",
           "adaptive_rbf", "kernel_rr")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def save_npz(path, arrays):
    """Use deterministic ZIP metadata; timings themselves are measured values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, values in sorted(arrays.items()):
            b = io.BytesIO()
            np.save(b, np.asarray(values), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, b.getvalue())


def seasonal_prediction(values, origin, horizon, period):
    index = origin + horizon - period * int(np.ceil(horizon / period))
    if index >= 0 and np.isfinite(values[index]):
        return float(values[index])
    previous = np.flatnonzero(np.isfinite(values[:origin + 1]))
    return float(values[previous[-1]]) if len(previous) else 0.0


def initial_pairs(task):
    values = np.asarray(task["values"], dtype=float)
    length, end = int(task["history"]), int(task["fit_end"])
    horizons = np.asarray(task["horizons"], dtype=int)
    origins = np.arange(length - 1, end - int(max(horizons)))
    x = np.asarray([values[o-length+1:o+1] for o in origins])
    y = np.asarray([values[o+horizons] for o in origins])
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if finite.sum() < 12:
        raise ValueError("Insufficient finite, completely matured initial pairs")
    assert np.max(origins[finite, None] + horizons) < end
    return x[finite], y[finite], origins[finite]


def quantiles(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {k: None for k in ("mean", "median", "p95", "p99", "maximum")}
    return dict(zip(("mean", "median", "p95", "p99", "maximum"),
                    map(float, (x.mean(), np.median(x), np.quantile(x,.95),
                                np.quantile(x,.99), x.max()))))


def execute(task, method, budget_name, config, *, output=None, timing_only=False):
    """Return every issued component and arrival trace; never retune on evaluation."""
    v = np.asarray(task["values"], dtype=float)
    length = int(task["history"])
    horizons = tuple(map(int, task["horizons"]))
    fit_end, eval_start = int(task["fit_end"]), int(task["eval_start"])
    period, scale = int(task["seasonal_lag"]), float(task["scale"])
    assert 0 < scale and length < fit_end < eval_start < len(v)
    budget = config["budgets"][budget_name]
    model = OnlineForecaster(method=method, horizons=horizons,
                             seed=config["model_seed"],
                             config=config.get("model_config", {}), **budget)
    x0, y0, origins0 = initial_pairs(task)
    started = utc_now()
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(x0, y0)
    initial_seconds = time.perf_counter() - start
    initial_state = jsonable(model.state_summary())
    rows, pending, trace, errors = [], defaultdict(list), [], []
    gamma = np.zeros(len(horizons), dtype=float)
    calibration = None
    disabled = False
    state_samples = [{"time": fit_end-1, "summary": initial_state}]
    geometry_refreshes = []
    last_initial_origin = int(origins0[-1])
    for t in range(fit_end, len(v)):
        # Select solely from labels that arrived strictly before evaluation.
        if t == eval_start:
            selection = []
            for hi, h in enumerate(horizons):
                eligible = [r for r in rows if r[2] == hi and r[0] < eval_start
                            and r[1] < eval_start and np.isfinite(r[6])]
                trials = []
                for g in config["gamma_grid"]:
                    loss = float(np.mean([(r[6] - (r[3] + g*r[4]))**2
                                          for r in eligible])) if eligible else None
                    trials.append({"gamma": g, "mse": loss})
                best = min(range(len(trials)),
                           key=lambda i: float("inf") if trials[i]["mse"] is None
                           else trials[i]["mse"])
                gamma[hi] = trials[best]["gamma"] if method != "ridge" else 0.0
                selection.append({"horizon": h, "rows": len(eligible),
                                  "latest_target": max((r[1] for r in eligible),
                                                       default=None),
                                  "trials": trials, "gamma": gamma[hi]})
            calibration = selection
        # An issued forecast receives its truth only on target arrival.
        for index in pending.pop(t, []):
            rows[index][6] = float(v[t])
        service_start = time.perf_counter()
        matured = []
        for hi, h in enumerate(horizons):
            origin = t-h
            if origin <= last_initial_origin or origin < length-1:
                continue
            hist = v[origin-length+1:origin+1]
            if np.isfinite(v[t]) and np.isfinite(hist).all():
                matured.append({"horizon_index": hi, "x": hist.copy(),
                                "y": float(v[t]), "origin": origin,
                                "target": t})
        observed_x = v[t-length+1:t+1]
        observed_x = observed_x.copy() if np.isfinite(observed_x).all() else None
        update_start = time.perf_counter()
        diag = {}
        if not disabled:
            try:
                diag = model.update(matured, now=t, observed_x=observed_x) or {}
            except Exception as exc:
                errors.append({"time": t, "stage": "update", "error": repr(exc),
                               "traceback": traceback.format_exc()})
                disabled = True
        update_seconds = time.perf_counter()-update_start
        prediction_start = time.perf_counter()
        pred_diag = {}
        if observed_x is not None:
            fallback = np.array([seasonal_prediction(v,t,h,period) for h in horizons])
            if not disabled:
                try:
                    base, corr, pred_diag = model.predict_components(observed_x)
                    base, corr = np.asarray(base,float), np.asarray(corr,float)
                    if base.shape != (len(horizons),) or corr.shape != base.shape:
                        raise ValueError("Model returned incorrect horizon shape")
                    if not np.isfinite(base).all() or not np.isfinite(corr).all():
                        raise FloatingPointError("Nonfinite prediction components")
                except Exception as exc:
                    errors.append({"time":t,"stage":"prediction","error":repr(exc),
                                   "traceback":traceback.format_exc()})
                    disabled = True
                    base, corr = fallback, np.zeros_like(fallback)
            else:
                base, corr = fallback, np.zeros_like(fallback)
            prediction = base + gamma*corr
        else:
            base = corr = prediction = None
        prediction_seconds = time.perf_counter()-prediction_start
        service_seconds = time.perf_counter()-service_start
        # Persistence, result formatting, and audit-state inspection are excluded
        # from model service timings; they do not affect any model decision.
        if prediction is not None:
            for hi,h in enumerate(horizons):
                target = t+h
                row = [t,target,hi,float(base[hi]),float(corr[hi]),
                       float(prediction[hi]),float("nan"),float(gamma[hi]),
                       int(t>=eval_start),int(disabled)]
                if target < len(v):
                    pending[target].append(len(rows))
                rows.append(row)
        trace.append({"time":t,"evaluation":t>=eval_start,
                      "matured_count":len(matured),
                      "latest_supervised_target":max((r["target"] for r in matured),default=-1),
                      "update_seconds":update_seconds,
                      "prediction_seconds":prediction_seconds,
                      "service_seconds":service_seconds,"disabled":disabled,
                      "update":jsonable(diag),"prediction":jsonable(pred_diag)})
        if (t-fit_end)%64==0 or t==len(v)-1:
            state_samples.append({"time":t,"summary":jsonable(model.state_summary())})
    arr = np.asarray(rows,dtype=float)
    cols = ["origin","target","horizon_index","baseline","correction",
            "prediction","truth","gamma","evaluation","framework_fallback"]
    arrays = {name:arr[:,i].astype(int) if i in (0,1,2,8,9) else arr[:,i]
              for i,name in enumerate(cols)}
    arrays["horizon"] = np.asarray(horizons,dtype=int)[arrays["horizon_index"]]
    valid = arrays["evaluation"].astype(bool)&np.isfinite(arrays["truth"])
    per_horizon = []
    for h in horizons:
        use = valid & (arrays["horizon"]==h)
        e = arrays["truth"][use]-arrays["prediction"][use]
        per_horizon.append({"horizon":h,"count":int(use.sum()),
                            "mse":float(np.mean(e**2)) if len(e) else None,
                            "scaled_mse":float(np.mean(e**2)/scale) if len(e) else None,
                            "mae":float(np.mean(np.abs(e))) if len(e) else None})
    evaluation_trace = [a for a in trace if a["evaluation"]]
    allowance = float(config["observation_allowance_seconds"])
    summary = {"schema_version":1,"task_id":task["task_id"],"kind":task["kind"],
               "collection":task["collection"],"method":method,"budget":budget_name,
               "budget_parameters":budget,"started_at":started,"finished_at":utc_now(),
               "history":length,"horizons":horizons,"seasonal_lag":period,
               "fit_end":fit_end,"evaluation_start":eval_start,"raw_length":len(v),
               "initial_training_rows":len(x0),"initial_latest_target":int(origins0[-1]+max(horizons)),
               "scale":scale,"calibration":calibration,"gamma":gamma,
               "per_horizon":per_horizon,
               "mean_scaled_mse":float(np.mean([p["scaled_mse"] for p in per_horizon]))
                   if all(p["scaled_mse"] is not None for p in per_horizon) else None,
               "initialization_seconds":initial_seconds,
               "calibration_service_seconds":sum(a["service_seconds"] for a in trace if not a["evaluation"]),
               "evaluation_service":quantiles([a["service_seconds"] for a in evaluation_trace]),
               "evaluation_update":quantiles([a["update_seconds"] for a in evaluation_trace]),
               "evaluation_prediction":quantiles([a["prediction_seconds"] for a in evaluation_trace]),
               "evaluation_observations":len(evaluation_trace),
               "allowance_seconds":allowance,
               "allowance_overruns":sum(a["service_seconds"]>allowance for a in evaluation_trace),
               "framework_errors":errors,"framework_fallback_forecasts":int(arrays["framework_fallback"][valid].sum()),
               "final_state":jsonable(model.state_summary()),"state_samples":state_samples,
               "process_peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               "process_memory_scope":"Peak RSS includes Python, libraries, source arrays and evidence buffers; not incremental model memory.",
               "threadpools":threadpool_info(),"cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
               "environment":{"python":sys.version,"numpy":np.__version__,"platform":platform.platform()},
               "timing_only":bool(timing_only)}
    if output is not None:
        output=Path(output)
        output.mkdir(parents=True,exist_ok=True)
        save_npz(output/"forecasts.npz",arrays)
        write_json(output/"trace.json",trace)
        summary["forecasts_sha256"]=hashlib.sha256((output/"forecasts.npz").read_bytes()).hexdigest()
        summary["trace_sha256"]=hashlib.sha256((output/"trace.json").read_bytes()).hexdigest()
        write_json(output/"summary.json",summary)
    return summary,arrays,trace


def config_read():
    return json.loads((HERE/"config.json").read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task")
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--budget",choices=("small","large"))
    p.add_argument("--output",type=Path,default=HERE/"results")
    p.add_argument("--all",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--pin-cpu",action="store_true")
    args=p.parse_args()
    cfg=config_read()
    if args.pin_cpu and hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    if args.all:
        if not (HERE/"DEVELOPMENT_FREEZE.json").exists():
            raise SystemExit("Freeze protocol/source/inputs before the reported run")
        check_freeze()
        tasks=load_tasks()
        roster=[(t["task_id"],m,b) for t in tasks for b in cfg["budgets"] for m in METHODS]
        # Fixed shuffle prevents a method always occupying one wall-clock order.
        rng=np.random.default_rng(20261031)
        roster=[roster[i] for i in rng.permutation(len(roster))]
        args.output.mkdir(parents=True,exist_ok=True)
        execution={"started_at":utc_now(),"roster":roster,"completed":[],"failed":[],
                   "freeze_sha256":hashlib.sha256((HERE/"DEVELOPMENT_FREEZE.json").read_bytes()).hexdigest()}
        for index,(task,method,budget) in enumerate(roster,1):
            dest=args.output/f"{task}__{budget}__{method}"
            if args.resume and (dest/"summary.json").exists():
                execution["completed"].append([task,method,budget,"already_present"])
                continue
            cmd=[sys.executable,"-m","extensions.adaptive_development.run","--task",task,
                 "--method",method,"--budget",budget,"--output",str(dest),"--pin-cpu"]
            done=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
            entry={"task":task,"method":method,"budget":budget,"returncode":done.returncode}
            if done.returncode:
                dest.mkdir(parents=True,exist_ok=True)
                (dest/"worker_error.log").write_text(done.stdout+done.stderr)
                execution["failed"].append(entry)
            else:
                execution["completed"].append(entry)
            write_json(args.output/"execution.json",execution)
            print(json.dumps({"progress":index,"total":len(roster),**entry}),flush=True)
        execution["finished_at"]=utc_now()
        write_json(args.output/"execution.json",execution)
        return
    if args.smoke:
        task=make_synthetic(20261099,"recurring",n=768)
        summary,_,_=execute(task,args.method or "adaptive_kahm",args.budget or "small",cfg,
                            output=args.output,timing_only=True)
        print(json.dumps({"timing_only":True,"method":summary["method"],
                          "initialization_seconds":summary["initialization_seconds"],
                          "service":summary["evaluation_service"],
                          "framework_error_count":len(summary["framework_errors"]),
                          "final_state":summary["final_state"]},default=jsonable))
        return
    tasks={t["task_id"]:t for t in load_tasks()}
    if not args.task or not args.method or not args.budget:
        p.error("Supply --task, --method and --budget, or --all / --smoke")
    summary,_,_=execute(tasks[args.task],args.method,args.budget,cfg,output=args.output)
    print(json.dumps({"task":args.task,"method":args.method,"budget":args.budget,
                      "status":"completed","framework_errors":len(summary["framework_errors"])}))


def check_freeze():
    frozen=json.loads((HERE/"DEVELOPMENT_FREEZE.json").read_text())
    for name,metadata in frozen["files"].items():
        p=HERE/name
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=metadata["sha256"]:
            raise RuntimeError(f"Frozen scientific file changed: {name}")


if __name__=="__main__":
    main()
