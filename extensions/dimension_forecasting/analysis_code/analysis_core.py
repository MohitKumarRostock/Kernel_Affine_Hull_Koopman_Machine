"""Saved-forecast analysis of dimension-selected reconstruction procedures.

No function imports a forecasting implementation, fits a model, or selects a
procedure on terminal outcomes. Independent repetition aggregates, rather than
overlapping histories, are the units of the primary inference.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
NONLINEAR = LAWS[1:]
PANELS = {"nonlinear": NONLINEAR, "linear": LAWS[:1], "all_descriptive": LAWS}
HORIZONS = (1, 4, 12)
METHODS = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common",
           "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded", "nonkahm_policy")
GEOMETRY = ("common", "kahm_departure", "rbf_departure", "kahm_unbounded", "rbf_unbounded")
RAW = ("ridge", "raw_rbf", "raw_hgb", "persistence")
NONKAHM = ("ridge", "raw_rbf", "raw_hgb", "persistence", "common", "rbf_departure", "rbf_unbounded")
CANDIDATES = {"historical2": (2, "legacy_normal_departure"), "native1": (1, "native"),
              "native2": (2, "native"), "native4": (4, "native"), "native8": (8, "native"),
              "native12": (12, "native"), "native20": (20, "native")}
NATIVE = tuple(c for c in CANDIDATES if c != "historical2")
CONTEXTS = ("historical2", "native20", "selected")
MODES = ("calibrated", "gamma1")
COMPARATORS = ("common", "rbf_departure", "raw_rbf", "raw_hgb", "nonkahm_policy")
META = ("task_id", "law", "history", "repetition", "seed", "noise_index", "noise_level")
VIEWS = tuple((c, m) for c in CANDIDATES for m in ("ridge",) + GEOMETRY)
VIEWS += tuple(("shared", m) for m in RAW)
VIEWS += tuple(("context_" + c, m) for c in CONTEXTS for m in METHODS)
VIEWS += tuple(("matched", m) for m in GEOMETRY)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def close(left, right, name, *, rtol=1e-12, atol=1e-13):
    require(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False), "Mismatch: " + name)


def finite(value, shape, name):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in "fiu" and np.isfinite(result).all(), "Invalid array: " + name)
    return result


def grouped(rows, fields):
    output = defaultdict(list)
    for row in rows:
        output[tuple(row[field] for field in fields)].append(row)
    return output


def means(rows, fields, values=("smse", "mse")):
    return [{**dict(zip(fields, key)), "n_records": len(group),
             **{name: float(np.mean([r[name] for r in group])) for name in values}}
            for key, group in sorted(grouped(rows, fields).items())]


def blend(base, raw, gamma):
    result = base.copy()
    for j, g in enumerate(gamma):
        if g != 0:
            result[:, j] = base[:, j] + g * (raw[:, j] - base[:, j])
    return result


def source(view, method, selections, contexts):
    if view in CANDIDATES or view == "shared":
        return view
    if view == "matched":
        return selections["kahm_departure"]["selected_candidate_id"]
    context = view.removeprefix("context_")
    if method == "nonkahm_policy":
        method = contexts[context]["selected_policy"]
    if method in RAW:
        return "shared"
    return selections[method]["selected_candidate_id"] if context == "selected" else context


def score_task(meta, targets, predictions, scale, selections, contexts):
    targets = finite(targets, (1664, 3), "D targets")
    scale = finite(scale, (3,), "A scale")
    require(np.all(scale >= 1e-12), "A scale below variance floor")
    expected = {f"{view}__{mode}__{method}" for view, method in VIEWS for mode in MODES}
    require(set(predictions) == expected, "Complete forecast roster differs")
    baseline = predictions["shared__calibrated__ridge"]
    for key, value in predictions.items():
        finite(value, targets.shape, key)
        if key.endswith("__ridge"):
            require(np.array_equal(value, baseline), "Shared ridge differs")
    for mode in MODES:
        for method in ("ridge",) + GEOMETRY:
            for left, right in (("historical2", "native2"), ("native12", "native20")):
                require(np.array_equal(predictions[f"{left}__{mode}__{method}"], predictions[f"{right}__{mode}__{method}"]), "Admissible-rank identity differs")
        for view, method in VIEWS:
            if view not in CANDIDATES and view != "shared":
                selected_method = contexts[view.removeprefix("context_")]["selected_policy"] if method == "nonkahm_policy" else method
                key = f"{source(view, method, selections, contexts)}__{mode}__{selected_method}"
                require(np.array_equal(predictions[f"{view}__{mode}__{method}"], predictions[key]), "Declared forecast view differs: " + view + "/" + method)
    horizon, task, stability = [], [], []
    ridge_max_error = float(np.max(np.abs(targets - baseline)))
    for view, method in VIEWS:
        for mode in MODES:
            prediction = predictions[f"{view}__{mode}__{method}"]
            error = targets - prediction
            mse = np.mean(error**2, axis=0)
            require(np.isfinite(mse).all(), "Nonfinite loss")
            identity = {**{k: meta[k] for k in META}, "view": view, "method": method, "mode": mode,
                        "source_candidate": source(view, method, selections, contexts)}
            for j, h in enumerate(HORIZONS):
                horizon.append({**identity, "horizon": h, "D_rows": len(targets),
                                "A_target_variance": float(scale[j]), "mse": float(mse[j]), "smse": float(mse[j]/scale[j])})
            task.append({**identity, "mse": float(np.mean(mse)), "smse": float(np.mean(mse/scale))})
            if view.startswith("context_") and method == "kahm_departure" and mode == "calibrated":
                maximum = float(np.max(np.abs(error)))
                ratio = maximum / ridge_max_error if ridge_max_error else (0. if maximum == 0 else math.inf)
                require(math.isfinite(ratio), "Undefined maximum-error screen")
                stability.append({**identity, "maximum_absolute_error": maximum, "ridge_maximum_absolute_error": ridge_max_error,
                                  "maximum_error_ratio": ratio, "screen_pass": ratio <= 5.})
    return horizon, task, stability


def repetition_table(task_rows, repetitions=24):
    expected = {(law, noise, rep) for law in LAWS for noise in (0, 1) for rep in range(repetitions)}
    first = [r for r in task_rows if r["view"] == "shared" and r["method"] == "ridge" and r["mode"] == "calibrated"]
    require(len(first) == len(expected) and {(r["law"], r["noise_index"], r["repetition"]) for r in first} == expected, "Incomplete task roster")
    require(len(task_rows) == len(expected)*len(VIEWS)*len(MODES), "Incomplete view roster")
    output = []
    fields = ("repetition", "view", "method", "mode")
    for panel, laws in PANELS.items():
        rows = [r for r in task_rows if r["law"] in laws]
        for key, group in sorted(grouped(rows, fields).items()):
            require(len(group) == 2*len(laws) and len({(r["law"], r["noise_index"]) for r in group}) == len(group), "Repeated or incomplete panel")
            output.append({**dict(zip(fields, key)), "panel": panel, "n_conditions": len(group),
                           "smse": float(np.mean([r["smse"] for r in group])), "mse": float(np.mean([r["mse"] for r in group]))})
    return output


def margin_contrast(candidate, comparator, *, factor, identifier, panel, comparator_name):
    candidate, comparator = np.asarray(candidate, float), np.asarray(comparator, float)
    require(candidate.shape == comparator.shape == (24,), "Primary inference requires 24 paired repetitions")
    require(np.isfinite(candidate).all() and np.isfinite(comparator).all() and np.all(candidate >= 0) and np.all(comparator >= 0), "Invalid loss aggregate")
    gaps = factor * comparator - candidate
    sd = float(np.std(gaps, ddof=1)); se = sd / math.sqrt(24)
    critical = float(student_t.ppf(1-.05/6, 23))
    gap = float(np.mean(gaps)); lower = gap - critical * se
    c, b = float(np.mean(candidate)), float(np.mean(comparator))
    p = float(student_t.sf(gap/se, 23)) if se else (0. if gap > 0 else (1. if gap < 0 else .5))
    return {"contrast_id": identifier, "panel": panel, "context": "selected", "candidate": "kahm_departure",
            "comparator": comparator_name, "candidate_mean_smse": c, "comparator_mean_smse": b,
            "point_relative_gain_percent": 100*(1-c/b) if b else None,
            "comparator_factor": factor, "margin": .03 if panel == "nonlinear" else .05,
            "n_repetitions": 24, "df": 23, "family_alpha": .05, "family_size": 6,
            "margin_gap_mean": gap, "margin_gap_sample_sd": sd, "margin_gap_se": se,
            "critical_t": critical, "simultaneous_one_sided_lower": lower,
            "p_one_sided": p, "p_bonferroni": min(1.,6*p),
            "positive_margin_repetitions": int(np.count_nonzero(gaps > 0)),
            "criterion_established": bool(lower > 0)}, gaps


def primary_tables(repetition_rows):
    lookup = {(r["panel"], r["view"], r["method"], r["mode"], r["repetition"]): r["smse"] for r in repetition_rows}
    summary, detail = [], []
    cases = [("nonlinear", m, .97, "nonlinear_vs_"+m) for m in COMPARATORS]
    cases.append(("linear", "ridge", 1.05, "linear_noninferiority_vs_ridge"))
    for panel, comparator, factor, identifier in cases:
        candidate = np.array([lookup[panel, "context_selected", "kahm_departure", "calibrated", rep] for rep in range(24)])
        control = np.array([lookup[panel, "context_selected", comparator, "calibrated", rep] for rep in range(24)])
        row, gaps = margin_contrast(candidate, control, factor=factor, identifier=identifier, panel=panel, comparator_name=comparator)
        summary.append(row)
        detail.extend({"contrast_id": identifier, "panel": panel, "repetition": rep,
                       "candidate_smse": float(candidate[rep]), "comparator_smse": float(control[rep]),
                       "comparator_factor": factor, "margin_gap": float(gaps[rep])} for rep in range(24))
    return summary, detail


SECONDARY = (
    ("selected_minus_historical_KAHM", "context_selected", "kahm_departure", "context_historical2", "kahm_departure"),
    ("native20_minus_historical_KAHM", "context_native20", "kahm_departure", "context_historical2", "kahm_departure"),
    ("selected_minus_native20_KAHM", "context_selected", "kahm_departure", "context_native20", "kahm_departure"),
    ("selected_KAHM_minus_selected_RBF", "context_selected", "kahm_departure", "context_selected", "rbf_departure"),
    ("selected_KAHM_minus_selected_common", "context_selected", "kahm_departure", "context_selected", "common"),
    ("matched_KAHM_minus_common", "matched", "kahm_departure", "matched", "common"),
    ("matched_KAHM_minus_RBF", "matched", "kahm_departure", "matched", "rbf_departure"),
    ("selected_bounded_minus_unbounded_KAHM", "context_selected", "kahm_departure", "context_selected", "kahm_unbounded"),
    ("matched_bounded_minus_unbounded_KAHM", "matched", "kahm_departure", "matched", "kahm_unbounded"),
    ("native20_KAHM_minus_RBF", "context_native20", "kahm_departure", "context_native20", "rbf_departure"),
    ("historical_KAHM_minus_RBF", "context_historical2", "kahm_departure", "context_historical2", "rbf_departure"),
)


def secondary_tables(repetition_rows):
    lookup = {(r["panel"], r["view"], r["method"], r["mode"], r["repetition"]): r["smse"] for r in repetition_rows}
    summary, detail = [], []
    for panel in PANELS:
        for mode in MODES:
            for identifier, lv, lm, rv, rm in SECONDARY:
                left = np.array([lookup[panel,lv,lm,mode,rep] for rep in range(24)])
                right = np.array([lookup[panel,rv,rm,mode,rep] for rep in range(24)])
                delta = left-right; se = float(np.std(delta,ddof=1)/np.sqrt(24)); critical = float(student_t.ppf(.975,23))
                point = float(np.mean(delta))
                identity = {"contrast_id": identifier, "panel": panel, "mode": mode, "left_view": lv, "left_method": lm, "right_view": rv, "right_method": rm}
                summary.append({**identity, "inference": "secondary_unadjusted_paired_t", "n_repetitions":24,
                                "left_mean_smse":float(np.mean(left)),"right_mean_smse":float(np.mean(right)),
                                "relative_gain_percent":float(100*(1-np.mean(left)/np.mean(right))),
                                "mean_difference":point,"standard_error":se,"lower95":point-critical*se,"upper95":point+critical*se,
                                "better_repetitions":int(np.count_nonzero(delta<0)),"ties":int(np.count_nonzero(delta==0)),"worse_repetitions":int(np.count_nonzero(delta>0))})
                detail.extend({**identity,"repetition":rep,"left_smse":float(left[rep]),"right_smse":float(right[rep]),"difference":float(delta[rep])} for rep in range(24))
    return summary, detail


def decompositions(task_rows):
    result=[]
    for (_, mode), group in sorted(grouped(task_rows,("task_id","mode")).items()):
        idx={(r["view"],r["method"]):r for r in group}
        old_common=idx["historical2","common"]["smse"]
        old_kahm=idx["historical2","kahm_departure"]["smse"]
        for view in ("native20","matched"):
            common=idx[view,"common"]; kahm=idx[view,"kahm_departure"]
            complete=kahm["smse"]-old_kahm
            shared=common["smse"]-old_common
            increment=(kahm["smse"]-common["smse"])-(old_kahm-old_common)
            close(complete,shared+increment,"Additive mechanism decomposition")
            result.append({**{k:kahm[k] for k in META},"mode":mode,"view":view,
                           "complete_change":complete,"common_change":shared,"departure_increment_change":increment,
                           "historical_departure_increment":old_kahm-old_common,
                           "new_departure_increment":kahm["smse"]-common["smse"],"identity_residual":complete-shared-increment})
    return result


def all_outcome_tables(horizon_rows, task_rows):
    reps=repetition_table(task_rows)
    primary, primary_detail=primary_tables(reps)
    secondary, secondary_detail=secondary_tables(reps)
    decomp=decompositions(task_rows)
    panel_horizon=[]; panel_decomp=[]
    for panel,laws in PANELS.items():
        panel_horizon.extend({"panel":panel,**r} for r in means([x for x in horizon_rows if x["law"] in laws],("view","method","mode","horizon")))
        panel_decomp.extend({"panel":panel,**r} for r in means([x for x in decomp if x["law"] in laws],("view","mode"),("complete_change","common_change","departure_increment_change","historical_departure_increment","new_departure_increment","identity_residual")))
    return {"horizon_losses.csv":horizon_rows,"task_losses.csv":task_rows,
            "condition_summary.csv":means(task_rows,("law","noise_level","view","method","mode")),
            "condition_horizon_summary.csv":means(horizon_rows,("law","noise_level","view","method","mode","horizon")),
            "repetition_losses.csv":reps,"panel_summary.csv":means(reps,("panel","view","method","mode")),
            "panel_horizon_summary.csv":panel_horizon,"primary_contrasts.csv":primary,"primary_repetition_contrasts.csv":primary_detail,
            "secondary_contrasts.csv":secondary,"secondary_repetition_contrasts.csv":secondary_detail,
            "dimension_decomposition.csv":decomp,"decomposition_summary.csv":panel_decomp}


def export_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    def cell(value):
        if isinstance(value,(dict,list,tuple)):
            return json.dumps(value,sort_keys=True,separators=(",",":"))
        return "" if value is None else str(value)
    expected=[[cell(row.get(key)) for key in fields] for row in rows]
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for values in expected:
            writer.writerow(dict(zip(fields,values)))
        stream.flush(); os.fsync(stream.fileno())
    with temporary.open(newline="",encoding="utf-8") as stream:
        saved=list(csv.reader(stream))
    require(saved==[fields]+expected,"CSV header, row or cell readback differs: "+str(path))
    expected_digest=digest(temporary)
    temporary.replace(path)
    require(digest(path)==expected_digest,"Atomic CSV installation differs: "+str(path))


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    require(temporary.read_bytes()==data and json.loads(temporary.read_text())==value,"JSON readback differs: "+str(path))
    temporary.replace(path)
    require(path.read_bytes()==data,"Atomic JSON installation differs: "+str(path))
