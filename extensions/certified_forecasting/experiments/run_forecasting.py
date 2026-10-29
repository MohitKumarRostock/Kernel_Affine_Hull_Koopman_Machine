"""Frozen-candidate independent-episode forecasting experiment.

Run from the extension root with PYTHONPATH=src. A run directory is immutable:
rerunning a completed run requires a different --output directory. Failures
are recorded and raised, never silently discarded from aggregate results.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
import sklearn

from certified_forecasting.data import make_episodes, histories_targets, SYSTEM_BOUNDS
from certified_forecasting.features import NestedSoftFeatures
from certified_forecasting.models import (
    BoundedRidge, BlockBoundedRidge, PolynomialMap, Persistence, KoopmanRidge,
)
from certified_forecasting.certificates import (
    certify_forecast, episode_mse, paired_improvement,
)


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_npz_atomic(path, arrays):
    """Publish a score archive only after the entire temporary file is flushed.

    A failed write leaves its .tmp file for diagnosis and never exposes a
    truncated file under the completed archive name. Existing archives are
    not overwritten.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite score archive {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def flat(X):
    return X.reshape(-1, X.shape[-1])


@dataclass
class Candidate:
    name: str
    family: str
    history: int
    transform: object
    model: object
    n_features: int
    n_states: int
    kappa: object
    blocks: object = None
    koopman: bool = False
    train_seconds: float = 0.0
    validation_mse: object = None

    def predict(self, F, horizons):
        return self.model.predict(F, horizons) if self.koopman else self.model.predict(F)


def build_candidates(train, validation, cfg, seed):
    H = cfg["horizons"]
    first = max(cfg["histories"]) - 1
    candidates, pair_names = [], []
    training_arrays, validation_arrays = {}, {}
    for history in cfg["histories"]:
        X, Y = histories_targets(train, history, H, cfg["origin_stride"], first)
        Xv, Yv = histories_targets(validation, history, H, cfg["origin_stride"], first)
        training_arrays[history] = (X, Y)
        validation_arrays[history] = (Xv, Yv)
        xf, yf = flat(X), flat(Y)
        # Targets are y[t+h], while EDMD is trained on a one-step feature shift.
        origins = np.arange(first, train.shape[1] - max(H), cfg["origin_stride"])
        next_indices = origins[:, None] + 1 - np.arange(history-1, -1, -1)[None, :]
        Xnext = train[:, :, 0][:, next_indices]
        current_y = xf[:, -1:]

        for kind in cfg["feature_kinds"]:
            started = time.perf_counter()
            first_kind_candidate = len(candidates)
            bank = NestedSoftFeatures(
                cluster_counts=cfg["cluster_counts"], kind=kind,
                seed=seed + history, omega=cfg["omega"],
                max_fit=cfg["max_geometry_inputs"],
                max_train_per_cluster=cfg["max_train_per_cluster"],
                input_scaling=cfg.get("kahm_input_scaling", "standard") if kind == "kahm" else None,
            ).fit(xf)
            F = bank.transform(xf)
            geometry_seconds = time.perf_counter() - started
            previous = None
            for stage, block in enumerate(bank.block_slices_, 1):
                start_fit = time.perf_counter()
                stop = block.stop
                blocks = bank.block_slices_[:stage]
                model = BlockBoundedRidge(
                    alpha=cfg["alpha"], kappa=cfg["kappa"],
                    bounds=(-1,1), block_slices=blocks,
                ).fit(F[:, :stop], yf)
                name = f"{kind}_nested_m{history}_s{stage}"
                transform = lambda x, b=bank, s=stage: b.transform(x, n_blocks=s)
                candidates.append(Candidate(
                    name, f"{kind}_nested", history, transform, model,
                    stop, stop, cfg["kappa"], blocks,
                    train_seconds=geometry_seconds + time.perf_counter()-start_fit,
                ))
                if previous is not None and kind == "kahm":
                    pair_names.append((previous, name))
                previous = name

            if kind == "kahm":
                # Ordinary Frobenius budget is an explicitly different class.
                start_fit = time.perf_counter()
                frob = BoundedRidge(cfg["alpha"], cfg["kappa"], (-1,1)).fit(F,yf)
                candidates.append(Candidate(
                    f"kahm_frobenius_m{history}", "kahm_frobenius", history,
                    bank.transform, frob, F.shape[1], F.shape[1], cfg["kappa"],
                    train_seconds=geometry_seconds+time.perf_counter()-start_fit,
                ))
                # Exact duplicate blocks leave the block-budget class unchanged.
                b0 = bank.block_slices_[0]
                count = len(bank.block_slices_)
                dup = np.tile(F[:, b0], (1,count))
                dupblocks = [slice(i*b0.stop,(i+1)*b0.stop) for i in range(count)]
                duplicate_model = BlockBoundedRidge(
                    0.0, cfg["kappa"], (-1,1), block_slices=dupblocks
                ).fit(dup,yf)
                duptransform = lambda x,b=bank,c=count: np.tile(b.transform(x,n_blocks=1),(1,c))
                candidates.append(Candidate(
                    f"kahm_duplicate_m{history}", "kahm_duplicate", history,
                    duptransform, duplicate_model, dup.shape[1], b0.stop,
                    cfg["kappa"], dupblocks,
                ))
                # Last-block hard state assignment: same geometry, lost softness.
                last = bank.block_slices_[-1]
                def hard_transform(x, b=bank, sl=last):
                    probs = b.transform(x)[:,sl]
                    return np.eye(probs.shape[1])[np.argmax(probs,axis=1)]
                hard = hard_transform(xf)
                hard_model = BoundedRidge(cfg["alpha"], cfg["kappa"], (-1,1)).fit(hard,yf)
                candidates.append(Candidate(
                    f"kahm_hard_m{history}", "kahm_hard", history,
                    hard_transform, hard_model, hard.shape[1], hard.shape[1], cfg["kappa"],
                ))
                start_fit = time.perf_counter()
                Fn = bank.transform(flat(Xnext))
                koop = KoopmanRidge(alpha=cfg["alpha"],bounds=(-1,1)).fit(F,Fn,current_y)
                candidates.append(Candidate(
                    f"kahm_koopman_m{history}", "kahm_koopman", history,
                    bank.transform, koop, F.shape[1],F.shape[1],
                    koop.lipschitz_for(H), koopman=True,
                    train_seconds=geometry_seconds+time.perf_counter()-start_fit,
                ))
                if "kahm_input_scaling" in cfg:
                    # Explicit new configurations retain the frozen scaler;
                    # inherited configurations keep their original output schema.
                    for candidate in candidates[first_kind_candidate:]:
                        candidate.input_scaling = bank.input_scaling_metadata()

        start_fit=time.perf_counter()
        ar = BoundedRidge(cfg["alpha"], cfg["kappa"],(-1,1)).fit(xf,yf)
        candidates.append(Candidate(
            f"ridge_ar_m{history}","ridge_ar",history,lambda x:x,ar,history,0,
            cfg["kappa"],train_seconds=time.perf_counter()-start_fit,
        ))
        start_fit=time.perf_counter()
        poly=PolynomialMap(degree=2).fit(xf)
        Fp=poly.transform(xf)
        pm=BoundedRidge(cfg["alpha"],cfg["kappa"],(-1,1)).fit(Fp,yf)
        candidates.append(Candidate(
            f"polynomial_m{history}","polynomial",history,poly.transform,pm,Fp.shape[1],0,
            cfg["kappa"],train_seconds=time.perf_counter()-start_fit,
        ))
        start_fit=time.perf_counter()
        pk=KoopmanRidge(alpha=cfg["alpha"],bounds=(-1,1)).fit(Fp,poly.transform(flat(Xnext)),current_y)
        candidates.append(Candidate(
            f"polynomial_koopman_m{history}","polynomial_koopman",history,
            poly.transform,pk,Fp.shape[1],0,pk.lipschitz_for(H),koopman=True,
            train_seconds=time.perf_counter()-start_fit,
        ))

    history=cfg["histories"][0]
    X,Y=training_arrays[history]
    candidates.append(Candidate(
        "persistence","persistence",history,lambda x:x,
        Persistence().fit(flat(X),flat(Y)),history,0,np.ones(len(H)),
    ))
    constant = BoundedRidge(cfg["alpha"],0.,(-1,1)).fit(np.zeros((len(flat(X)),1)),flat(Y))
    candidates.append(Candidate(
        "constant","constant",history,lambda x:np.zeros((len(x),1)),
        constant,1,1,0.,
    ))

    for c in candidates:
        Xv,Yv=validation_arrays[c.history]
        Fv=c.transform(flat(Xv))
        pred=c.predict(Fv,H).reshape(Yv.shape)
        c.validation_mse=episode_mse(Yv,pred).mean(axis=0)
    return candidates,pair_names


def candidate_description(c):
    desc={"name":c.name,"family":c.family,"history":c.history,
          "n_features":c.n_features,"n_states":c.n_states,
          "kappa":np.asarray(c.kappa).tolist(),
          "metric":"block_max" if c.blocks is not None else "euclidean",
          "validation_mse":c.validation_mse.tolist(),
          "train_seconds":c.train_seconds}
    for name in ["norm_sum_","frobenius_norm_","optimization_gap_","n_iter_","converged_"]:
        if hasattr(c.model,name):
            value=getattr(c.model,name)
            desc[name.rstrip("_")]=value.item() if isinstance(value,np.generic) else value
    if hasattr(c, "input_scaling"):
        desc["input_scaling"] = c.input_scaling
    return desc


def run_one(cfg,system,seed,destination):
    start=time.perf_counter()
    # Seed domains are explicit and independent across development/calibration/test.
    streams={"train":seed*100+11,"validation":seed*100+23,
             "calibration":seed*100+47,"test":seed*100+89}
    train=make_episodes(system,cfg["n_train"],cfg["episode_length"],streams["train"])
    validation=make_episodes(system,cfg["n_validation"],cfg["episode_length"],streams["validation"])
    candidates,pair_names=build_candidates(train,validation,cfg,seed)
    description=[candidate_description(c) for c in candidates]
    write_json(destination/"frozen_candidates.json",description)
    # Only now are calibration and test episodes generated.
    calibration=make_episodes(system,cfg["n_calibration"],cfg["episode_length"],streams["calibration"])
    test=make_episodes(system,cfg["n_test"],cfg["episode_length"],streams["test"])
    H=cfg["horizons"]
    first=max(cfg["histories"])-1
    arrays={}
    for m in cfg["histories"]:
        arrays[m]=(histories_targets(calibration,m,H,cfg["origin_stride"],first),
                   histories_targets(test,m,H,cfg["origin_stride"],first))
    K=len(candidates)
    tail=cfg["delta"]*cfg["risk_delta_fraction"]/(2*K*len(H))
    results=[]
    scores={}
    for c in candidates:
        (Xc,Yc),(Xt,Yt)=arrays[c.history]
        F=c.transform(flat(Xc))
        prediction=c.predict(F,H).reshape(Yc.shape)
        certificate,losses,witnesses=certify_forecast(
            F.reshape(*Xc.shape[:2],F.shape[-1]),Yc,prediction,c.kappa,2.,
            tail,tail,method=cfg["certificate_method"],block_slices=c.blocks,
        )
        pred_test=c.predict(c.transform(flat(Xt)),H).reshape(Yt.shape)
        test_losses=episode_mse(Yt,pred_test)
        result=candidate_description(c)
        result["certificates"]=certificate
        result["test_mse"]=test_losses.mean(axis=0).tolist()
        result["test_mse_se"]=(test_losses.std(axis=0,ddof=1)/np.sqrt(len(test_losses))).tolist()
        results.append(result)
        scores[c.name+"__cal_loss"]=losses
        scores[c.name+"__pair_score"]=witnesses
        scores[c.name+"__test_loss"]=test_losses
    improvements=[]
    if pair_names:
        pair_delta=cfg["delta"]*(1-cfg["risk_delta_fraction"])/(len(pair_names)*len(H))
        for old,new in pair_names:
            for h_idx,h in enumerate(H):
                ci=paired_improvement(scores[old+"__cal_loss"][:,h_idx],
                                      scores[new+"__cal_loss"][:,h_idx],4.,pair_delta,
                                      method=cfg["certificate_method"])
                improvements.append({"old":old,"new":new,"horizon":h,**ci})
    selected_validation=min(results,key=lambda r:(np.mean(r["validation_mse"]),r["n_features"]))["name"]
    selections=[]
    for tolerance in cfg["rmse_tolerances"]:
        for family in ["all","kahm_nested"]:
            eligible=[r for r in results if family=="all" or r["family"]==family]
            passed=[r for r in eligible if all(ci["rmse_upper"]<=tolerance for ci in r["certificates"])]
            chosen=min(passed,key=lambda r:(r["n_features"],r["history"],np.mean(r["validation_mse"]))) if passed else None
            selections.append({"tolerance":tolerance,"family":family,
                               "selected":None if chosen is None else chosen["name"],
                               "n_features":None if chosen is None else chosen["n_features"],
                               "status":"certified" if chosen is not None else "unresolved"})
    record={"system":system,"seed":seed,"streams":streams,"n_candidates":K,
            "n_independent_calibration_episodes":len(calibration),
            "n_disjoint_calibration_pairs":len(calibration)//2,
            "confidence_scope":"one complete frozen family and its paired refinements within this replicate",
            "delta":cfg["delta"],"tail_delta":tail,"results":results,
            "paired_improvements":improvements,"selected_by_validation":selected_validation,
            "certified_selections":selections,"elapsed_seconds":time.perf_counter()-start}
    if cfg.get("save_episode_scores",True):
        # Full double-precision per-episode scores let every certificate be recomputed.
        write_npz_atomic(destination/"episode_scores.npz",scores)
    # A completed result is published after its requested score archive.
    write_json(destination/"results.json",record)
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("configs/main_v1.json"))
    p.add_argument("--output",type=Path,default=Path("results/main_v1"))
    p.add_argument("--system",default=None)
    p.add_argument("--seed",type=int,default=None)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    config_hash=hashlib.sha256(args.config.read_bytes()).hexdigest()
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:
        head="unavailable"
    manifest={"configuration":cfg,"configuration_sha256":config_hash,"base_git_commit":head,
              "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
              "scipy":scipy.__version__,"sklearn":sklearn.__version__,
              "source_sha256":{str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(Path("src").rglob("*.py"))}}
    # Include this runner and the actual imported parent implementation/helpers.
    parent_repository=Path(__file__).resolve().parents[3]
    manifest["source_sha256"]["experiments/run_forecasting.py"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for source in sorted(parent_repository.glob("*.py")):
        manifest["source_sha256"]["parent/"+source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path=args.output/"manifest.json"
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old["configuration_sha256"]!=config_hash or old["source_sha256"]!=manifest["source_sha256"]:
            raise RuntimeError("Configuration/source changed; use a new campaign output directory")
    else:
        write_json(manifest_path,manifest)
    for system in ([args.system] if args.system else cfg["systems"]):
        for seed in ([args.seed] if args.seed is not None else cfg["seeds"]):
            destination=args.output/f"{system}_{seed}"
            if destination.exists():
                raise RuntimeError(f"Refusing to overwrite run {destination}")
            destination.mkdir()
            print(f"START {system} seed={seed}",flush=True)
            try:
                record=run_one(cfg,system,seed,destination)
                print(f"DONE {system} seed={seed} candidates={record['n_candidates']} seconds={record['elapsed_seconds']:.1f}",flush=True)
            except Exception as error:
                write_json(destination/"failure.json",{"error":repr(error),"traceback":traceback.format_exc()})
                raise


if __name__=="__main__":
    main()
