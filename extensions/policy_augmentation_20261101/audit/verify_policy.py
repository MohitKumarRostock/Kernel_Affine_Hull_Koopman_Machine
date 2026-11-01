#!/usr/bin/env python3
"""Independent audit of optional KAHM and equal-score-count search policies.

Recomputes selection and terminal errors from retained arrays; independently
checks selected extra RBF features and centered ridge normal equations. KAHM
native basis metadata and orthogonality are checked, but its inherited feature
map and every earlier fitting algorithm are not independently reimplemented.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t as student
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
P0 = ("legacy__ridge", "legacy__raw_rbf", "legacy__raw_hgb", "legacy__persistence",
      "legacy__common", "legacy__rbf_departure", "legacy__rbf_unbounded",
      "crossfit__shared_bias", "crossfit__crossfit_rbf", "crossfit__insample_rbf")
ORDERS = {"P0": P0, "P1": P0 + ("crossfit__insample_kahm",), "P0plus": P0 + ("extra__extra_rbf",)}
ALPHAS = np.array([.0001, .001, .01, .1, 1.])
ETAS = np.array([0., .25, .5, 1.])
SCALERS = ("standard", "minmax", "tanh_z_1")
CAPS = (1, 2, 4, 8, 12)
FACTORS = (.25, .5, 1., 2., 4.)
LAWS = ("linear_ar", "nonlinear_ar", "vanderpol", "duffing")
CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def near(left, right, message, *, exact=False, rtol=3e-10, atol=3e-12):
    left, right = np.asarray(left), np.asarray(right)
    check(left.shape == right.shape, message + ": shape")
    check(np.array_equal(left, right) if exact else np.allclose(left, right, rtol=rtol, atol=atol), message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(piece)
    return state.hexdigest()


def ahash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def verified_files(base, mapping):
    base = Path(base).resolve()
    for relative, expected in mapping.items():
        path = (base / relative).resolve()
        check(not Path(relative).is_absolute() and path.is_relative_to(base), "Unconfined sealed file")
        check(path.is_file() and digest(path) == expected, "Sealed file differs: " + str(path))


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def array_key(identity, mode):
    family, name = identity.split("__", 1)
    if family == "legacy" and mode == "eta1":
        mode = "gamma1"
    return family + "__" + mode + "__" + name


def mix(base, raw, eta):
    result = base.copy()
    for h, multiplier in enumerate(eta):
        if multiplier != 0:
            result[:, h] = base[:, h] + multiplier * (raw[:, h] - base[:, h])
    return result


def loss(prediction, targets):
    delta = np.asarray(prediction) - targets
    check(delta.shape == targets.shape and np.isfinite(delta).all(), "Invalid loss arrays")
    return np.einsum("ij,ij->j", delta, delta) / len(targets)


def verify_policy_selection(calibration, targets, scale, policies, predictions=None):
    """Public pure oracle used by manufactured preflight and the full audit."""
    scores_by_policy = {}
    for name, order in ORDERS.items():
        record = policies[name]
        check(record["selection_split"] == "C", "Policy selection was not C")
        check([x["identity"] for x in record["candidates"]] == list(order), "Candidate menu/order changed")
        scores = [float(np.mean(loss(calibration[array_key(identity, "calibrated")], targets) / scale))
                  for identity in order]
        choice = int(np.argmin(scores))
        check(record["selected_identity"] == order[choice], "Policy did not select first C minimum")
        if "C_Y_sha256" in record:
            check(record["C_Y_sha256"] == ahash(targets), "Policy target hash differs")
        for row, score, identity in zip(record["candidates"], scores, order):
            near(row["score"], score, "Policy C loss")
            near(row["mse_by_horizon"], loss(calibration[array_key(identity, "calibrated")], targets), "C MSE")
            check(row["C_prediction_sha256"] == ahash(calibration[array_key(identity, "calibrated")]),
                  "C forecast hash differs")
        scores_by_policy[name] = scores[choice]
        for mode in ("calibrated", "eta1"):
            key = f"policy__{mode}__{name}"
            source = array_key(order[choice], mode)
            near(calibration[key], calibration[source], "C policy exact forwarding", exact=True)
            if predictions is not None:
                near(predictions[key], predictions[source], "D policy exact forwarding", exact=True)
    for name in ("P1", "P0plus"):
        check(scores_by_policy[name] <= scores_by_policy["P0"] + 1e-14, "Nested C loss increased")
    for mode in ("calibrated", "eta1"):
        near(calibration[f"policy__{mode}__P0"], calibration[f"policy__{mode}__expanded_nonkahm"],
             "C original policy alias", exact=True)
        if predictions is not None:
            near(predictions[f"policy__{mode}__P0"], predictions[f"policy__{mode}__expanded_nonkahm"],
                 "D original policy alias", exact=True)
    return scores_by_policy


def audit_b_family(selection, validation, prefix, targets, scale, expected_order, label):
    records = selection["candidates"]
    check(len(records) == 15 and len(ALPHAS) * len(records) == 75, label + " score count")
    check([r["candidate_id"] for r in records] == expected_order, label + " candidate order")
    check(selection["selection_split"] == "B", label + " selection split")
    scores = []
    for record in records:
        key = prefix + record["candidate_id"]
        paths = validation[key + "__alpha_paths"]
        check(paths.shape == (5, len(targets), targets.shape[1]), label + " alpha paths shape")
        losses = np.stack([loss(p, targets) for p in paths])
        near(losses, record["B_mse_by_alpha_horizon"], label + " all alpha losses")
        indices = np.argmin(losses, axis=0)
        near(indices, record["selected_alpha_indices"], label + " alpha argmin", exact=True)
        near(ALPHAS[indices], record["selected_alpha_by_horizon"], label + " selected alphas", exact=True)
        selected = np.column_stack([paths[indices[h], :, h] for h in range(targets.shape[1])])
        near(selected, validation[key], label + " selected B forecast", exact=True)
        near(loss(selected, targets), record["mse_by_horizon"], label + " selected loss")
        score = float(np.mean(loss(selected, targets) / scale))
        near(score, record["score"], label + " normalized candidate loss")
        scores.append(score)
        check(ahash(paths) == record["B_alpha_prediction_sha256"], label + " alpha paths hash")
        check(ahash(selected) == record["B_prediction_sha256"], label + " B forecast hash")
    chosen = int(np.argmin(scores))
    check(selection["selected_index"] == chosen and selection["selected_candidate"] == expected_order[chosen],
          label + " joint feature choice")


def scaled_histories(x, extra):
    arm = extra.selections["selected_scaler"]
    audit = extra.history_audit
    if arm == "standard":
        return (x - extra.raw_scaler["mean"]) / extra.raw_scaler["scale"]
    if arm == "tanh_z_1":
        return np.tanh((x - audit["mean"]) / audit["legacy_standard_scale"])
    check(arm == "minmax", "Undeclared extra scaler")
    z = 2 * (x - audit["minimum"]) / np.where(audit["constant"], 1., audit["range"]) - 1
    z[:, audit["constant"]] = 0
    return z


def extra_features(x, extra):
    """Independent NumPy expression of the declared selected Nyström map."""
    state = extra.feature_state
    z = scaled_histories(x, extra)
    distances = cdist(z, state["landmarks"], metric="sqeuclidean")
    kernel = np.exp(-distances / (2 * state["bandwidth_squared"]))
    raw = np.column_stack([z, kernel @ state["transform"]])
    normalized = (raw - state["scaler"]["mean"]) / state["scaler"]["scale"]
    width = normalized.shape[1]
    # All retained production features are finite and moderate. This algebraic
    # form differs from the implementation's overflow-safe row rescaling.
    bounded = np.sqrt(width) * normalized / np.sqrt(width + np.sum(normalized * normalized, axis=1, keepdims=True))
    check(np.isfinite(bounded).all(), "Nonfinite independent extra features")
    return bounded


def audit_extra_state(suite, abc, c_arrays, d_arrays, queries):
    extra, crossfit = suite["extra"], suite["crossfit"]
    head = extra.head
    near(extra.bias, crossfit.bias, "Shared exact bias", exact=True)
    residual = abc["A_Y"] - crossfit.training_arrays["A_full_common"]
    check(extra.selections["residual_target_sha256"] == ahash(residual), "Wrong extra residual target")
    feature = extra_features(abc["A_X"], extra)
    near(feature.mean(axis=0), head["mean"], "Selected extra feature centering")
    centered = feature - feature.mean(axis=0)
    response = residual - residual.mean(axis=0)
    n, width = centered.shape
    check(width == head["feature_width"], "Feature width differs")
    near(np.mean(centered @ head["coef"], axis=0), np.zeros(3), "Extra training mean correction")
    near(np.sqrt(np.mean(response * response, axis=0)), head["response_rms"], "Residual RMS")
    gram, cross = centered.T @ centered, centered.T @ response
    for h, alpha in enumerate(head["alpha_by_horizon"]):
        oracle = np.linalg.solve(gram + n * alpha * np.eye(width), cross[:, h])
        near(oracle, head["coef"][:, h], "Selected extra ridge normal equation", rtol=2e-7, atol=2e-9)
    coefficient = np.linalg.norm(head["coef"], axis=0)
    coefficient_limit = head["response_rms"] / (2 * np.sqrt(head["alpha_by_horizon"]))
    check(np.all(coefficient <= coefficient_limit * (1 + 1e-9) + 1e-11), "Selected extra coefficient bound")
    radius = math.sqrt(width) + np.linalg.norm(head["mean"])
    bound = np.abs(extra.eta) * radius * coefficient_limit
    for split, x, arrays in (("C", abc["C_X"], c_arrays), ("D", queries, d_arrays)):
        shared = arrays["crossfit__eta1__shared_bias"]
        raw = shared + (extra_features(x, extra) - head["mean"]) @ head["coef"]
        near(raw, arrays["extra__eta1__extra_rbf"], "Independent extra " + split + " raw predictions",
             rtol=2e-8, atol=2e-10)
        pred = mix(shared, raw, extra.eta)
        near(pred, arrays["extra__calibrated__extra_rbf"], "Independent extra " + split + " predictions",
             rtol=2e-8, atol=2e-10)
        check(np.all(np.max(np.abs(pred-shared), axis=0) <= bound * (1+1e-8) + 1e-10), "Extra correction bound")
    # Audit selected native local dimensions and orthonormality. Rank-zero
    # singleton/constant fallbacks have zero-column matrices and are explicit.
    basis_count = 0
    for bank in (crossfit.common.bank, crossfit.departures["insample_kahm"].bank):
        for local in bank["models"]:
            p, q, record = local["p"], local["q"], local["dimension_audit"]
            rank, ambient = p.shape[1], p.shape[0]
            check(record["effective_rank"] == rank, "Incorrect stored local rank")
            check(record["normal_direction_reserved"] is False and record["policy"] == "native", "Wrong native policy")
            expected = min(record["requested_cap"], ambient, record["local_reference_count"]-1)
            check(record["initial_admissible_rank"] == expected and 0 <= rank <= expected, "Invalid native rank cap")
            check(ambient == 12 and record["local_reference_count"] == len(q), "Native dimensions/reference count")
            near(p.T @ p, np.eye(rank), "Selected basis orthogonality", rtol=1e-8, atol=1e-8)
            basis_count += 1
    return basis_count


def full_audit(root, analysis, output):
    root, analysis, output = Path(root).resolve(), Path(analysis).resolve(), Path(output).resolve()
    check(not output.exists(), "Audit output exists")
    config = read(root / "config.json")
    check(config["master_seed"] == 2026110191 and config["repetitions"] == 24 and config["task_count"] == 192,
          "Fresh cohort scope differs")
    check(config["analysis"]["primary_family_size"] == 4 and config["analysis"]["family_alpha"] == .05,
          "Primary family differs")
    freeze = read(root / "freeze.json")
    check(freeze["status"] == "frozen" and freeze["config_sha256"] == digest(root / "config.json"), "Source freeze")
    verified_files(root, freeze["files"])
    verified_files(root.parent / "dimension_forecasting", freeze["inherited_files"])
    verified_files(root.parent / "preprocessing_dimension", freeze["previous_extension_files"])
    check(config["policy_orders"] == {name: list(order) for name, order in ORDERS.items()}, "Policy menus changed")
    seed_manifest = read(root / "prepared/seed_manifest.json")
    records = seed_manifest["records"]
    seeds = [r["seed"] for r in records]
    check(len(seeds) == len(set(seeds)) == 1272, "Seed roster/collision")
    prior = read(root / "readiness/prior_seed_exclusions.json")
    check(not (set(seeds) & {r["seed"] for r in prior["records"]}), "Seed overlaps prior studies")
    for record in records:
        check(record["components"][0] == 2026110191, "Incorrect master seed component")
        check(int(np.random.SeedSequence(record["components"]).generate_state(1)[0]) == record["seed"],
              "Seed derivation differs")
    fit_lock, pred_lock = read(root / "FIT_LOCK.json"), read(root / "PREDICTION_LOCK.json")
    for lock, phase in ((fit_lock, "fits"), (pred_lock, "predictions")):
        check(lock["status"] == "sealed" and lock["phase"] == phase and lock["task_count"] == 192, "Invalid seal")
        check(lock["config_sha256"] == digest(root / "config.json") and
              lock["freeze_sha256"] == digest(root / "freeze.json"), "Seal source link")
        verified_files(root, lock["files"])
    release = read(root / "TARGET_RELEASE.json")
    check(release["status"] == "released", "D scoring not released")
    for key, name in (("fit_lock_sha256", "FIT_LOCK.json"), ("prediction_lock_sha256", "PREDICTION_LOCK.json"),
                      ("d_manifest_sha256", "prepared/d_manifest.json")):
        check(release[key] == digest(root/name), "Release linkage differs")
    a_manifest, d_manifest = read(root/"prepared/abc_manifest.json"), read(root/"prepared/d_manifest.json")
    check(utc(freeze["created_utc"]) <= utc(a_manifest["created_utc"]) <= utc(fit_lock["created_utc"]) <=
          utc(d_manifest["created_utc"]) <= utc(pred_lock["created_utc"]) <= utc(release["created_utc"]), "Stage chronology")
    abc_tasks, d_tasks = ({x["task_id"]: x for x in m["tasks"]} for m in (a_manifest,d_manifest))
    check(set(abc_tasks) == set(d_tasks) and len(d_tasks) == 192, "Missing tasks")
    check({(x["law"], x["noise_index"], x["repetition"]) for x in d_tasks.values()} ==
          {(law, noise, rep) for law in LAWS for noise in (0,1) for rep in range(24)}, "Factorial changed")
    # The prediction entry point must use already selected policies, never call
    # the policy selector, and report targets_loaded=False. This is procedural
    # source verification, not evidence of access-control isolation.
    syntax = ast.parse((root/"code/pipeline.py").read_text())
    prediction_fn = next(node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "worker_predict")
    calls = {node.func.id for node in ast.walk(prediction_fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    check("choose_policy" not in calls and "choose_policies" not in calls, "D entry point selects a policy")
    for folder in (root/"code", root.parent/"preprocessing_dimension/code", root.parent/"dimension_forecasting/code"):
        sys.path.insert(0, str(folder))
    import focused_dimension_model as fd
    fd.load_runtime()
    import crossfit_model, extra_rbf  # Trusted frozen types required for saved-state loading.
    summary = read(analysis/"analysis_summary.json")
    for name, record in summary["tables"].items():
        check(digest(analysis/name) == record["sha256"], "Analysis table digest")
    with (analysis/"task_losses.csv").open() as stream:
        reported = {(r["task_id"],r["forecast_key"]):r for r in csv.DictReader(stream)}
    with (analysis/"primary_results.csv").open() as stream:
        primary = {r["contrast_id"]:r for r in csv.DictReader(stream)}
    check(len(reported) == 192*50 and len(primary) == 4, "Analysis roster")
    values, selected_counts = {}, {name:{} for name in ORDERS}
    bases = 0
    for index, task_id in enumerate(sorted(d_tasks)):
        atask, dtask = abc_tasks[task_id], d_tasks[task_id]
        check(digest(root/atask["input"]) == atask["sha256"], "ABC file digest")
        for key in ("query", "target"):
            check(digest(root/dtask[key]) == dtask[key+"_sha256"], "D input digest")
        with np.load(root/atask["input"], allow_pickle=False) as z:
            abc = {key:z[key] for key in z.files}
        with np.load(root/dtask["query"], allow_pickle=False) as z:
            check(not any("Y" in key for key in z.files), "Targets in D query file")
            query = z["D_X"]
        with np.load(root/dtask["target"], allow_pickle=False) as z:
            target = z["D_Y"]
        scale = np.maximum(np.var(abc["A_Y"],axis=0), 1e-12)
        near(scale, abc["loss_scale"], "A-only denominator", exact=True)
        result = root/"results"/task_id
        fit = read(result/"fit_summary.json")
        check(fit["task"] == atask and fit["status"] == "completed", "Fit receipt")
        pre = read(result/"pre_c_selection.json")
        check(pre["C_loaded"] is False, "C loaded before B selection receipt")
        receipt = read(root/"predictions"/task_id/"prediction_receipt.json")
        check(receipt["task"] == dtask and receipt["targets_loaded"] is False and receipt["state_unchanged"] is True,
              "Target-free prediction receipt")
        check(utc(fit["completed_utc"]) <= utc(fit_lock["created_utc"]), "Fit after seal")
        check(utc(receipt["utc"]) <= utc(pred_lock["created_utc"]), "Prediction after seal")
        with np.load(result/"calibration_predictions.npz",allow_pickle=False) as z:
            cp = {key:z[key] for key in z.files}
        with np.load(root/"predictions"/task_id/"predictions.npz",allow_pickle=False) as z:
            dp = {key:z[key] for key in z.files}
        check(set(cp) == set(dp) and len(cp) == 50, "C/D forecast roster")
        verify_policy_selection(cp, abc["C_Y"], scale, fit["policies"], dp)
        with np.load(result/"validation_predictions.npz",allow_pickle=False) as z:
            validation = {key:z[key] for key in z.files}
        # KAHM candidate state keys contain a suffix __kahm, so map its saved
        # keys to the same direct candidate view used by the extra family.
        kahm_selection = fit["crossfit"]["selections"]["target_modes"]["insample"]["kahm"]
        kahm_validation = {}
        for record in kahm_selection["candidates"]:
            source = "crossfit__"+record["state_key"]
            for suffix in ("", "__alpha_paths"):
                kahm_validation[record["candidate_id"]+suffix] = validation[source+suffix]
        audit_b_family(kahm_selection,kahm_validation,"",abc["B_Y"],scale,
                       [f"{arm}__cap{cap}" for cap in CAPS for arm in SCALERS],"KAHM")
        audit_b_family(fit["extra"]["selections"],validation,"extra__",abc["B_Y"],scale,
                       [f"{arm}__bw{factor:g}" for factor in FACTORS for arm in SCALERS],"Extra RBF")
        check(fit["extra"]["diagnostics"]["B_candidate_scores_per_horizon"] == 75,"Extra counted search")
        for family, method, record in (("crossfit","insample_kahm",fit["crossfit"]["calibration"]["choices"]["insample_kahm"]),
                                       ("extra","extra_rbf",fit["extra"]["calibration"])):
            raw = cp[f"{family}__eta1__{method}"]
            base = cp["crossfit__eta1__shared_bias"]
            losses = np.stack([loss(mix(base,raw,np.repeat(eta,3)),abc["C_Y"]) for eta in ETAS])
            eta = ETAS[np.argmin(losses,axis=0)]
            near(eta,record["selected_eta"],"C eta argmin",exact=True)
            near(mix(base,raw,eta),cp[f"{family}__calibrated__{method}"],"C eta forwarding",exact=True)
            near(mix(dp["crossfit__eta1__shared_bias"],dp[f"{family}__eta1__{method}"],eta),
                 dp[f"{family}__calibrated__{method}"],"D eta unchanged",exact=True)
        suite = pickle.loads(gzip.decompress((result/"state.pkl.gz").read_bytes()))
        check(suite["policies"] == fit["policies"], "Serialized policies differ from receipt")
        with threadpool_limits(limits=1):
            bases += audit_extra_state(suite,abc,cp,dp,query)
        for key,prediction in dp.items():
            score = float(np.mean(loss(prediction,target)/scale))
            row = reported[task_id,key]
            near(score,float(row["smse"]),"Reported terminal loss")
            near(float(np.mean(loss(prediction,target))),float(row["mse"]),"Reported raw terminal loss")
            values[dtask["law"],dtask["noise_index"],dtask["repetition"],key] = score
        for policy,record in fit["policies"].items():
            identity=record["selected_identity"]
            selected_counts[policy][identity]=selected_counts[policy].get(identity,0)+1
        if (index+1)%16==0:
            print(json.dumps({"verified_tasks":index+1,"total_tasks":192,"assertions":CHECKS}),flush=True)
    results=[]
    critical=float(student.ppf(1-.05/4,23))
    for panel,laws,factor,label in (("nonlinear",LAWS[1:],.97,"3% superiority"),("linear",LAWS[:1],1.05,"5% noninferiority")):
        for comparator in ("P0","P0plus"):
            candidate=np.array([np.mean([values[law,noise,rep,"policy__calibrated__P1"] for law in laws for noise in (0,1)]) for rep in range(24)])
            other=np.array([np.mean([values[law,noise,rep,"policy__calibrated__"+comparator] for law in laws for noise in (0,1)]) for rep in range(24)])
            margin=candidate-factor*other
            upper=float(margin.mean()+critical*margin.std(ddof=1)/np.sqrt(24))
            identity=f"P1_vs_{comparator}_{panel}"
            row=primary[identity]
            near(upper,float(row["simultaneous_one_sided_upper"]),"Independent simultaneous bound")
            near(candidate.mean(),float(row["candidate_mean_smse"]),"Independent candidate mean")
            near(other.mean(),float(row["comparator_mean_smse"]),"Independent comparator mean")
            check((row["criterion_established"]=="True")==bool(upper<0),"Independent primary decision")
            results.append({"contrast_id":identity,"upper_bound":upper,"criterion_established":bool(upper<0)})
    value={"status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),"assertions":CHECKS,
           "tasks":192,"forecast_arrays":192*50,"independent_repetitions":24,"selected_local_bases":bases,
           "four_primary_decisions":results,"selection_counts":selected_counts,"audit_sha256":digest(__file__),
           "analysis_summary_sha256":digest(analysis/"analysis_summary.json"),"freeze_sha256":digest(root/"freeze.json"),
           "scope":["All source/seal/array digests; 192 task roster and chronology links.",
                    "Every new-policy C selection, both added families' 75 B scores per horizon, eta argmins, exact C/D forwarding.",
                    "Independent selected extra RBF feature formulas and centered normal equations; selected native dimension metadata and orthogonality.",
                    "Every saved D forecast loss and all four simultaneous primary bounds from repetition aggregates."],
           "limits":["Does not independently reimplement KAHM raw feature maps, all legacy fits, every unselected model, or native PCA eigenspaces.",
                     "Execution seals document a procedural sequence, not access-control isolation.",
                     "The selected extra forecast is replayed relative to the saved shared baseline; the shared baseline is not independently regenerated."]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    return value


def manufactured_policy_check():
    rng=np.random.default_rng(723199)
    target=rng.normal(size=(11,3));scale=np.array([.5,1.,2.])
    arrays={};policies={}
    for identity in (*P0,"crossfit__insample_kahm","extra__extra_rbf"):
        pred=rng.normal(size=target.shape)
        for mode in ("calibrated","eta1"):
            arrays[array_key(identity,mode)]=pred.copy()
    # Force an exact best-score tie between old and new families. P0 must win
    # according to the nested order, including a zero-length marginal gain.
    for identity in (P0[0],"crossfit__insample_kahm","extra__extra_rbf"):
        for mode in ("calibrated","eta1"):
            arrays[array_key(identity,mode)]=target.copy()
    for name,order in ORDERS.items():
        rows=[]
        for identity in order:
            pred=arrays[array_key(identity,"calibrated")]
            mse=loss(pred,target)
            rows.append({"identity":identity,"score":float(np.mean(mse/scale)),"mse_by_horizon":mse.tolist(),
                         "C_prediction_sha256":ahash(pred)})
        selected=order[int(np.argmin([r["score"] for r in rows]))]
        policies[name]={"selected_identity":selected,"candidates":rows,"selection_split":"C","C_Y_sha256":ahash(target)}
        for mode in ("calibrated","eta1"):
            arrays[f"policy__{mode}__{name}"]=arrays[array_key(selected,mode)].copy()
    for mode in ("calibrated","eta1"):
        arrays[f"policy__{mode}__expanded_nonkahm"]=arrays[f"policy__{mode}__P0"].copy()
    verify_policy_selection(arrays,target,scale,policies,arrays)
    check(all(p["selected_identity"]==P0[0] for p in policies.values()),"First-tie oracle failed")
    corrupted=dict(arrays)
    corrupted["policy__calibrated__P1"]=arrays["policy__calibrated__P1"]+1
    caught=False
    try: verify_policy_selection(corrupted,target,scale,policies,arrays)
    except AssertionError: caught=True
    check(caught,"Forwarding corruption was not rejected")
    return {"status":"passed","manufactured_only":True,"nested_menus":True,"first_tie_rule":True,
            "corrupted_forwarding_rejected":True,"assertions":CHECKS}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--analysis",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--manufactured",action="store_true")
    args=parser.parse_args()
    if args.manufactured:
        value=manufactured_policy_check()
    else:
        value=full_audit(args.root,args.analysis or args.root/"analysis",args.output or args.root/"verification/policy_audit.json")
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
