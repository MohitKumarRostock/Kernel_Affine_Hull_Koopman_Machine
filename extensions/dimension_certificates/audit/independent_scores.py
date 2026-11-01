#!/usr/bin/env python3
"""Independent score and certificate reconstruction without fitting imports.

The fixture command reads only labelled manufactured observations. The reusable
verify_job function is called on production only by a separately gated audit
after the complete test-stage seal is released.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np

POLICIES = ("cap2", "native20", "selected_cap")
ARMS = POLICIES + ("rbf", "affine")
REST = ("ridge_ar", "polynomial", "persistence", "constant")
CAPS = (1,2,4,8,12,20)
HORIZONS = (1,4,12)
POLICY_SHA = "c03d9a1923b5d9c8ade8a4198fd06e2b24c6add9bf0e0992c469f90d7621efa2"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode("ascii")+value.dtype.str.encode("ascii")+value.tobytes()).hexdigest()


class Audit:
    def __init__(self):
        self.counts=Counter()

    def require(self,value,label,context=None):
        self.counts["assertions"]+=1
        if not bool(value):
            raise AssertionError(label+(" :: "+str(context) if context is not None else ""))

    def equal(self,a,b,label,context=None):
        self.require(np.array_equal(a,b),label,context)
        self.counts["exact_array_comparisons"]+=1

    def close(self,a,b,label,context=None):
        a,b=np.asarray(a,float),np.asarray(b,float)
        self.require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                     and np.allclose(a,b,rtol=2e-11,atol=5e-13),label,context)
        self.counts["numeric_comparisons"]+=int(a.size)


def moments(values,bound,tail):
    # Scalar compensated summation is independent of the fitting/analyzer code.
    x=[float(v) for v in values]
    if len(x)<2 or not all(math.isfinite(v) and -1e-12<=v<=bound+1e-10 for v in x):
        raise AssertionError("Invalid independent-unit score range")
    n=len(x);mean=math.fsum(x)/n
    variance=math.fsum((v-mean)**2 for v in x)/(n-1)
    log=math.log(2/tail)
    radius=math.sqrt(2*variance*log/n)+7*bound*log/(3*(n-1))
    return dict(mean=mean,variance=variance,radius=radius,lower=max(0.,mean-radius),
                upper=min(bound,mean+radius),n_independent=n,bound=bound,delta_per_tail=tail,method="bernstein")


def selection_key(row):
    return (float(np.mean(row["validation_mse"])),row["n_features"],row["history"],row["cap"],row["id"])


def verify_job(folder,cfg,*,fixture=False,audit=None):
    audit=Audit() if audit is None else audit
    folder=Path(folder)
    development=read(folder/"development.json")
    job=development["job"]
    audit.require((job["system"]=="manufactured_fixture")==fixture,"fixture and production roles differ")
    candidates=development["candidates"]
    expected_ids={f"kahm_m{h}_c{c}_s{s}" for h in (4,12) for c in CAPS for s in (1,2,3,4)}
    expected_ids|={f"{kind}_m{h}_c2_s{s}" for kind in ("rbf","affine") for h in (4,12) for s in (1,2,3,4)}
    expected_ids|={f"{kind}_m{h}" for kind in ("ridge_ar","polynomial") for h in (4,12)}
    expected_ids|={"persistence","constant"}
    audit.require(len(candidates)==70 and {r["id"] for r in candidates}==expected_ids,
                  "complete seventy-candidate development family",job["id"])
    validation=arrays(folder/"validation_scores.npz")
    audit.require(set(validation)==expected_ids|{"episode_ids","origin_indices","horizons"},
                  "complete validation array family")
    by_id={r["id"]:r for r in candidates}
    for row in candidates:
        values=validation[row["id"]]
        audit.require(values.shape==(cfg["n_validation"],3) and np.isfinite(values).all()
                      and values.min()>=0 and values.max()<=4,"episode-level bounded validation scores",row["id"])
        audit.close(values.mean(axis=0),row["validation_mse"],"all candidate validation scores reproduced",row["id"])
        audit.counts["validation_candidates"]+=1
    audit.require(set(development["choices"])==set(ARMS+REST),"complete nine-arm selection family")
    for arm,choice in development["choices"].items():
        family="kahm" if arm in POLICIES else arm
        eligible=[r for r in candidates if r["family"]==family]
        if arm in ("cap2","native20"):
            eligible=[r for r in eligible if r["cap"]==(2 if arm=="cap2" else 20)]
        winner=min(eligible,key=selection_key)
        audit.require(winner["id"]==choice["winner_id"] and list(selection_key(winner))==choice["selection_key"],
                      "best-prefix validation-only bank/forecast choice",arm)
        audit.counts["validation_choices"]+=1
    rows=development["evaluation"]
    expected_aliases={f"{arm}_s{s}" for arm in ARMS for s in (1,2,3,4)}|set(REST)
    audit.require(len(rows)==24 and {r["alias"] for r in rows}==expected_aliases,
                  "fixed twenty-four-alias evaluation family")
    evaluation={r["alias"]:r for r in rows}
    for arm in ARMS:
        choice=development["choices"][arm]
        for stage,width in zip((1,2,3,4),(2,6,14,30)):
            row=evaluation[f"{arm}_s{stage}"]
            audit.require(row["bank_id"]==choice["bank_id"] and row["id"]==f"{choice['bank_id']}_s{stage}"
                          and row["stage"]==stage and row["n_features"]==width
                          and row["selected_forecast"]==(stage==choice["winner_stage"]),
                          "all four nested prefixes use one validation-chosen bank",[arm,stage])
            audit.require(all(row[k]==by_id[row["id"]][k] for k in by_id[row["id"]]),
                          "evaluation alias retains exact development candidate",[arm,stage])
    for key,bank in development["banks"].items():
        d=bank["dimension_policy"]
        audit.require(bank["input_scaling"]["mode"]=="standard" and bank["input_scaling"]["clip"] is False,
                      "declared fixed standard input scaling",key)
        audit.require(d["geometry_kind"]==bank["kind"] and d["requested_cap"]==bank["cap"]
                      and d["ambient_dimension"]==bank["history"] and len(d["local_models"])==30,
                      "complete bank dimension metadata",key)
        if bank["kind"]=="kahm":
            audit.require(d["shared_policy_source_sha256"]==POLICY_SHA and d["policy"]=="native",
                          "native risk policy source identity",key)
            for rank in d["local_models"]:
                initial=min(bank["cap"],bank["history"],rank["local_reference_count"]-1)
                audit.require(rank["policy_source_sha256"]==POLICY_SHA and rank["requested_cap"]==bank["cap"]
                              and rank["ambient_dimension"]==bank["history"] and rank["initial_admissible_rank"]==initial
                              and 1<=rank["effective_rank"]<=initial and rank["fallback_applied"] is False
                              and rank["native_failure"] is None and rank["status"]=="fitted",
                              "native rank request and realized successful state",key)
                audit.counts["native_rank_records"]+=1
    for history in (4,12):
        banks=[development["banks"][f"kahm_m{history}_c{cap}"] for cap in CAPS if cap>=history]
        for field in ("training_feature_sha256","validation_feature_sha256"):
            audit.require(len({b[field] for b in banks})==1,"ambient-limited caps have identical actual features",[history,field])
    coefficients=arrays(folder/"coefficients.npz")
    selected_ids={r["id"] for r in rows}
    audit.require(set(coefficients)=={i+s for i in selected_ids for s in ("__coefficient","__intercept")},
                  "retained actual coefficient/intercept family")
    for row in rows:
        coef=coefficients[row["id"]+"__coefficient"];intercept=coefficients[row["id"]+"__intercept"]
        audit.require(coef.shape==(row["n_features"],3) and intercept.shape==(3,)
                      and np.isfinite(coef).all() and np.isfinite(intercept).all(),"finite head dimensions",row["alias"])
        if row["family"]=="persistence":
            expected=np.zeros_like(coef);expected[-1]=1
            audit.equal(coef,expected,"persistence latest-value coefficient")
            value=float(np.linalg.norm(coef,axis=0).max())
            audit.require(row["kappa"]==1,"persistence Lipschitz budget")
        elif row["blocks"] is None:
            value=float(np.linalg.norm(coef))
        else:
            audit.require([i for start,stop in row["blocks"] for i in range(start,stop)]==list(range(len(coef))),
                          "retained blocks exactly partition head")
            value=math.fsum(float(np.linalg.norm(coef[start:stop])) for start,stop in row["blocks"])
        audit.close(value,row["coefficient_norm"],"actual coefficient budget metadata",row["alias"])
        audit.require(value<=row["kappa"]+1e-9 and (row["family"]!="constant" or row["kappa"]==0),
                      "actual head is in the frozen certified class",row["alias"])
        audit.counts["coefficient_budget_checks"]+=1
    tail=cfg["delta"]/cfg["campaign_total_tails"]
    held={}
    for split in ("calibration","test"):
        record=read(folder/(split+".json"));score=arrays(folder/(split+"_scores.npz"))
        held[split]=(record,score)
        n=cfg["n_"+split]
        audit.require(record["job"]==job and record["split"]==split and record["n_independent_episodes"]==n
                      and record["n_origins_per_episode"]==13 and record["stream_seed"]==job["streams"][split],
                      "held-out stage sampling and identity",split)
        audit.equal(score["episode_ids"],np.arange(n),"independent episode identity",split)
        audit.equal(score["origin_indices"],np.arange(11,60,4),"common thirteen causal origins",split)
        audit.equal(score["horizons"],HORIZONS,"three forecast horizons",split)
        results={r["alias"]:r for r in record["results"]}
        audit.require(len(record["results"])==24 and set(results)==expected_aliases,"complete held-out alias family",split)
        expected={a+"__loss" for a in expected_aliases}|{"episode_ids","origin_indices","horizons"}
        if split=="calibration":
            expected|={a+s for a in expected_aliases for s in ("__witness","__pair_distance")}
            expected|={"pair_episode_ids","pair_target_abs_difference"}
            audit.equal(score["pair_episode_ids"],np.arange(n).reshape(-1,2),"fixed disjoint episode pairing")
            target_diff=score["pair_target_abs_difference"]
            audit.require(target_diff.shape==(n//2,13,3) and np.isfinite(target_diff).all()
                          and target_diff.min()>=0 and target_diff.max()<=2,"bounded per-origin target differences")
            audit.require(record["tail_delta"]==tail,"fixed campaign tail allocation")
        audit.require(set(score)==expected,"complete sufficient-statistics array roster",split)
        full=arrays(folder/(split+"_predictions.npz")) if job["full_prediction_audit"] else None
        if full is not None:
            audit.require(set(full)==selected_ids|{"targets","episode_ids","origin_indices","horizons"},
                          "full prediction subset retains all selected unique forecasts",split)
            target=full["targets"]
            audit.require(target.shape==(n,13,3) and np.isfinite(target).all() and abs(target).max()<=1,
                          "full bounded target array",split)
            audit.require(array_hash(target)==record["target_array_sha256"],"full target provenance hash",split)
            if fixture:
                episodes=np.random.default_rng(job["streams"][split]).uniform(-.8,.8,size=(n,72,1))
                oracle=np.stack([episodes[:,np.arange(11,60,4)+h,0] for h in HORIZONS],axis=2)
                audit.equal(target,oracle,"manufactured target indexing from independently sliced episodes",split)
                audit.require(array_hash(episodes)==record["episode_array_sha256"],"manufactured base episode identity",split)
            if split=="calibration":
                audit.equal(np.abs(target[::2]-target[1::2]),score["pair_target_abs_difference"],
                            "full prediction subset independently reconstructs pair target differences")
        for alias,row in evaluation.items():
            result=results[alias];loss=score[alias+"__loss"]
            audit.require(loss.shape==(n,3) and np.isfinite(loss).all() and loss.min()>=0 and loss.max()<=4,
                          "bounded per-episode loss", [split,alias])
            audit.close(loss.mean(axis=0),result["mse"],"held-out means use episodes",[split,alias])
            audit.close(loss.std(axis=0,ddof=1)/math.sqrt(n),result["mse_se"],"held-out SE uses episodes",[split,alias])
            diag=result["diagnostics"]
            audit.require(diag["prediction_nonfinite_count"]==0 and -1<=diag["prediction_min"]<=diag["prediction_max"]<=1
                          and diag["prediction_count"]==n*13*3,"prediction range/count diagnostic",[split,alias])
            if full is not None:
                pred=full[row["id"]]
                audit.require(pred.shape==target.shape and np.isfinite(pred).all() and abs(pred).max()<=1
                              and array_hash(pred)==diag["prediction_sha256"],"full prediction provenance/range",[split,alias])
                audit.equal(np.mean((pred-target)**2,axis=1),loss,"exact full prediction-to-episode-loss join",[split,alias])
                audit.counts["full_prediction_loss_joins"]+=1
            if split=="calibration":
                distance=score[alias+"__pair_distance"];witness=score[alias+"__witness"]
                audit.require(distance.shape==(n//2,13) and np.isfinite(distance).all() and distance.min()>=0,
                              "finite disjoint-pair feature distances",alias)
                oracle=.5*np.mean(np.maximum(target_diff-row["kappa"]*distance[:,:,None],0)**2,axis=1)
                audit.equal(witness,oracle,"independent pair-witness reconstruction",alias)
                audit.counts["pair_witness_reconstructions"]+=1
                for index,horizon in enumerate(HORIZONS):
                    endpoint=result["certificates"][index]
                    audit.require(endpoint["horizon"]==horizon,"certificate horizon identity",alias)
                    for side,values,bound,details_key in (("upper",loss[:,index],4.,"upper_details"),
                                                         ("lower",witness[:,index],2.,"lower_details")):
                        oracle=moments(values,bound,tail);detail=endpoint[details_key]
                        audit.require(set(detail)=={"mean","variance","radius",side,"n_independent","bound","delta_per_tail","method"},
                                      "only designated tail is reported",[alias,horizon,side])
                        for key in ("mean","variance","radius",side):
                            audit.close(oracle[key],detail[key],"independent Bernstein statistic",[alias,horizon,side,key])
                        for key in ("n_independent","bound","delta_per_tail","method"):
                            audit.require(detail[key]==oracle[key],"certificate sampling/allocated tail identity",[alias,horizon,side,key])
                        audit.close(endpoint["mse_"+side],oracle[side],"risk endpoint",[alias,horizon,side])
                        audit.close(endpoint["rmse_"+side],math.sqrt(oracle[side]),"RMSE endpoint",[alias,horizon,side])
                        audit.counts["risk_endpoints"]+=1
                for tolerance in cfg["rmse_tolerances"]:
                    accepted=all(e["mse_upper"]<=tolerance**2 for e in result["certificates"])
                    excluded=any(e["mse_lower"]>tolerance**2 for e in result["certificates"])
                    expected="conflicting_bounds" if accepted and excluded else "accepted" if accepted else "excluded" if excluded else "unresolved"
                    audit.require(result["decisions"][str(tolerance)]==expected,"simultaneous tolerance decision",[alias,tolerance])
    comparisons=[(f"{p}_s{s}",f"{p}_s{s+1}") for p in POLICIES for s in (1,2,3)]
    comparisons += [(f"{old}_s{development['choices'][old]['winner_stage']}",
                     f"{new}_s{development['choices'][new]['winner_stage']}")
                    for old,new in (("cap2","native20"),("cap2","selected_cap"),("native20","selected_cap"))]
    reported=held["calibration"][0]["paired_improvements"];score=held["calibration"][1]
    audit.require(len(reported)==36 and [(r["old"],r["new"],r["horizon"]) for r in reported]
                  ==[(old,new,h) for old,new in comparisons for h in HORIZONS],
                  "fixed thirty-six improvement-tail family")
    for row in reported:
        h=HORIZONS.index(row["horizon"])
        delta=score[row["old"]+"__loss"][:,h]-score[row["new"]+"__loss"][:,h]
        oracle=moments(delta+4,8.,tail)
        audit.close(row["improvement_lower"],oracle["lower"]-4,"independent paired lower endpoint")
        audit.close(row["empirical_improvement"],float(np.mean(delta)),"independent paired improvement mean")
        for key in ("mean","variance","radius","lower"):
            audit.close(row[key],oracle[key],"shifted improvement statistics",key)
        audit.require("upper" not in row and row["delta_per_tail"]==tail and row["bound"]==8
                      and row["n_independent"]==cfg["n_calibration"],"only designated bounded paired tail")
        audit.counts["improvement_endpoints"]+=1
    audit.counts["jobs_verified"]+=1
    return {"job":job,"choices":development["choices"],"evaluation":evaluation,
            "calibration":held["calibration"][0],"test":held["test"][0]}


def main(root):
    cfg=read(root/"config.json")
    cfg.update(n_train=40,n_validation=20,n_calibration=48,n_test=24)
    audit=Audit()
    verify_job(root/"verification/manufactured_v1",cfg,fixture=True,audit=audit)
    result={"status":"passed","passed":True,"completed_utc":datetime.now(timezone.utc).isoformat(),
            "manufactured_only":True,"counts":dict(audit.counts),"audit_source_sha256":sha(__file__),
            "scope":"Independent selection, nested-family, actual coefficient budget, pair witness, all designated endpoint, full forecast-loss and target-index reconstruction without fitting or primary-analyzer imports."}
    (root/"audit/manufactured_score_checks.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"passed","counts":dict(audit.counts)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root.resolve())
