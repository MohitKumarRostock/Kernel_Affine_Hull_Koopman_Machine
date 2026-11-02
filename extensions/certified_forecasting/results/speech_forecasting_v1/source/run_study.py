"""Prospective speaker-separated forecasting under the adjacent protocol.

The raw provider archive stays outside the repository. Development is completed
and hashed for all seeds before any calibration or test loss is evaluated.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import platform
import time
import traceback
import wave
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist
import sklearn
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from certified_forecasting.features import NestedSoftFeatures
from certified_forecasting.models import BlockBoundedRidge, BoundedRidge


HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parents[1]
ORIGINS = np.arange(11, 88, 4)
HORIZONS = np.array([1, 4, 12])
SEEDS = (8601, 8602, 8603)
FAMILIES = ("kahm", "rbf", "affine", "ar", "quadratic", "kernel", "persistence", "constant")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def energy_observable(pcm):
    pcm = np.asarray(pcm)
    if pcm.dtype != np.int16 or pcm.shape != (16000,):
        raise ValueError("Expected one mono, signed 16-bit, one-second clip.")
    frames = pcm.astype(np.float64).reshape(100, 160) / 32768.0
    power = np.mean(frames * frames, axis=1)
    return np.clip((10 * np.log10(np.maximum(power, 1e-6)) + 60) / 60, 0, 1)


def prepare(archive, output):
    rows, curves, excluded = [], [], []
    with zipfile.ZipFile(archive) as source:
        names = sorted(n for n in source.namelist()
                       if n.startswith("mini_speech_commands/") and n.endswith(".wav"))
        for name in names:
            path = Path(name)
            if "_nohash_" not in path.stem:
                excluded.append({"path": name, "reason": "missing documented identifier"})
                continue
            with wave.open(io.BytesIO(source.read(name)), "rb") as audio:
                info = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes())
                if info != (1, 2, 16000, 16000):
                    excluded.append({"path": name, "reason": "format or duration", "format": info})
                    continue
                pcm = np.frombuffer(audio.readframes(16000), dtype="<i2")
            curves.append(energy_observable(pcm))
            rows.append({"path": name, "word": path.parent.name,
                         "speaker": path.stem.split("_nohash_")[0]})
    metadata = pd.DataFrame(rows)
    speakers = np.sort(metadata.speaker.unique())
    permutation = np.random.default_rng(8602026).permutation(speakers)
    n = len(permutation)
    cut = np.cumsum([0, int(.15 * n), int(.10 * n), int(.60 * n), n - int(.15*n) - int(.10*n) - int(.60*n)])
    assignment = {}
    for split, lo, hi in zip(("train", "validation", "calibration", "test"), cut[:-1], cut[1:]):
        assignment.update({str(s): split for s in permutation[lo:hi]})
    metadata["split"] = metadata.speaker.map(assignment)
    metadata.to_csv(output / "clips.csv", index=False)
    pd.DataFrame({"speaker": speakers, "split": [assignment[s] for s in speakers]}).to_csv(
        output / "speakers.csv", index=False)
    values = np.asarray(curves, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise AssertionError("Invalid bounded observable.")
    np.savez_compressed(output / "energy_observations.npz", energy=values)
    provenance = {"archive_sha256": digest(archive), "archive_bytes": Path(archive).stat().st_size,
                  "source_wav_count": len(names), "eligible_clips": len(rows),
                  "eligible_speakers": n, "excluded": excluded,
                  "groups": metadata.groupby("split").speaker.nunique().to_dict(),
                  "clips": metadata.groupby("split").size().to_dict(),
                  "protocol_sha256": digest(HERE / "protocol.md"),
                  "source_url": "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                  "license": "CC BY 4.0", "attribution": "Speech Commands, Pete Warden (2018); TensorFlow Mini Speech Commands subset"}
    write_json(output / "data_provenance.json", provenance)
    return values, metadata, provenance


def histories(energy, history):
    indices = ORIGINS[:, None] - np.arange(history - 1, -1, -1)[None, :]
    return energy[:, indices].reshape(-1, history), energy[:, ORIGINS[:, None] + HORIZONS].reshape(-1, 3)


def group_losses(prediction, target, speakers):
    losses = np.mean((prediction - target).reshape(len(speakers), len(ORIGINS), 3)**2, axis=1)
    unique, inverse = np.unique(speakers, return_inverse=True)
    total = np.zeros((len(unique), 3))
    np.add.at(total, inverse, losses)
    total /= np.bincount(inverse)[:, None]
    return unique, total


def train_rows(metadata, seed):
    rng = np.random.default_rng(seed)
    eligible = metadata.index[metadata.split == "train"].to_numpy()
    groups = {s: metadata.index[(metadata.split == "train") & (metadata.speaker == s)].to_numpy()
              for s in sorted(metadata.loc[eligible, "speaker"].unique())}
    keys = list(groups)
    count = min(4096, len(eligible) * len(ORIGINS))
    chosen = rng.integers(len(keys), size=count)
    clips = np.array([rng.choice(groups[keys[int(j)]]) for j in chosen])
    origins = rng.integers(len(ORIGINS), size=count)
    return clips, origins


@dataclass
class Forecast:
    family: str
    history: int
    model: object = None
    feature: object = None
    scaler: object = None
    stage: int = 0
    mean: object = None

    def predict(self, x):
        if self.family == "persistence":
            return np.repeat(x[:, -1:], 3, axis=1)
        if self.family == "constant":
            return np.tile(self.mean, (len(x), 1))
        if self.family in ("kahm", "rbf", "affine"):
            f = self.feature.transform(x, n_blocks=self.stage)
        else:
            f = self.scaler.transform(x)
            if self.feature is not None:
                f = self.feature.transform(f)
        return np.clip(self.model.predict(f), 0, 1)


def develop(energy, metadata, seed, output):
    selected, records, banks = {}, [], []
    clip_index, origin_index = train_rows(metadata, seed)
    np.savez_compressed(output / f"training_rows_{seed}.npz", clip_index=clip_index, origin_index=origin_index)
    val_idx = metadata.index[metadata.split == "validation"].to_numpy()
    val_speakers = metadata.loc[val_idx, "speaker"].to_numpy()
    training = {}
    validation = {}
    for m in (4, 12):
        indices = ORIGINS[origin_index, None] - np.arange(m - 1, -1, -1)[None, :]
        x = energy[clip_index[:, None], indices]
        y = energy[clip_index[:, None], ORIGINS[origin_index, None] + HORIZONS]
        training[m] = x, y
        validation[m] = histories(energy[val_idx], m)

    def retain(forecast, name, dimensions, prediction, parameters, fit_seconds, bank_id=None):
        _, losses = group_losses(prediction, validation[forecast.history][1], val_speakers)
        mse = losses.mean(axis=0)
        model = forecast.model
        gap = float(getattr(model, "optimization_gap_", 0.0)) if model is not None else 0.0
        norm = float(getattr(model, "norm_sum_", getattr(model, "frobenius_norm_", 0.0))) if model is not None else 0.0
        row = {"seed": seed, "family": forecast.family, "candidate": name,
               "history": forecast.history, "features": dimensions,
               "validation_mse": mse.tolist(), "validation_mean": float(mse.mean()),
               "parameters": parameters, "head_fit_seconds": fit_seconds,
               "bank_id": bank_id, "optimization_gap": gap, "coefficient_norm": norm,
               "converged": bool(getattr(model, "converged_", True))}
        records.append(row)
        key = (float(mse.mean()), dimensions, forecast.history, name)
        if forecast.family not in selected or key < selected[forecast.family][0]:
            selected[forecast.family] = (key, forecast, row)

    for m in (4, 12):
        x, y = training[m]
        xv, _ = validation[m]
        for kind in ("kahm", "rbf", "affine"):
            for scaling in ("standard", "raw"):
                for sharpness in ((1.0,) if kind == "kahm" else (.25, 1., 4.)):
                    bank_id = f"{seed}_{kind}_m{m}_{scaling}_sharp{sharpness:g}"
                    started = time.perf_counter()
                    bank = NestedSoftFeatures(cluster_counts=(2,4,8,16), kind=kind,
                        seed=seed+m, omega=16, max_fit=1600, max_train_per_cluster=24,
                        subspace_dim=2, sharpness=sharpness, input_scaling=scaling).fit(x)
                    fit_time = time.perf_counter() - started
                    started = time.perf_counter()
                    f, fv = bank.transform(x), bank.transform(xv)
                    transform_time = time.perf_counter() - started
                    banks.append({"bank_id": bank_id, "geometry_fit_seconds": fit_time,
                                  "development_transform_seconds": transform_time})
                    for stage, block in enumerate(bank.block_slices_, 1):
                        for kappa in (1., 4.):
                            for alpha in (1e-4, 1e-2):
                                started = time.perf_counter()
                                model = BlockBoundedRidge(alpha=alpha, kappa=kappa, bounds=(0,1),
                                    block_slices=bank.block_slices_[:stage], tol=1e-7).fit(f[:,:block.stop], y)
                                elapsed = time.perf_counter() - started
                                if sum(np.linalg.norm(model.coef_[sl]) for sl in bank.block_slices_[:stage]) > kappa + 1e-10:
                                    raise AssertionError("Infeasible fitted coefficient budget")
                                name = f"{bank_id}_s{stage}_k{kappa:g}_a{alpha:g}"
                                retain(Forecast(kind,m,model,bank,stage=stage), name, block.stop,
                                       model.predict(fv[:,:block.stop]),
                                       {"scaling":scaling,"sharpness":sharpness,"stage":stage,"kappa":kappa,"alpha":alpha},
                                       elapsed, bank_id)
                    print(f"completed development bank {bank_id}", flush=True)

        scaler = StandardScaler().fit(x)
        z, zv = scaler.transform(x), scaler.transform(xv)
        for kind in ("ar", "quadratic"):
            feature = None if kind == "ar" else PolynomialFeatures(2, include_bias=False).fit(z)
            f = z if feature is None else feature.transform(z)
            fv = zv if feature is None else feature.transform(zv)
            for alpha in (1e-6,1e-4,1e-2,1e-1):
                started = time.perf_counter()
                model = BoundedRidge(alpha=alpha, bounds=(0,1)).fit(f,y)
                elapsed = time.perf_counter() - started
                retain(Forecast(kind,m,model,feature,scaler), f"{seed}_{kind}_m{m}_a{alpha:g}", f.shape[1],
                       model.predict(fv), {"alpha":alpha}, elapsed)
        rng = np.random.default_rng(seed + 91000 + m)
        sample = z[rng.choice(len(z), min(512,len(z)), replace=False)]
        distances = pdist(sample, metric="sqeuclidean")
        positive = distances[distances > 0]
        median = float(np.median(positive)) if len(positive) else 1.0
        for multiplier in (.25,1.,4.):
            gamma = multiplier / median
            started = time.perf_counter()
            feature = Nystroem(kernel="rbf", gamma=gamma, n_components=128, random_state=seed+m).fit(z)
            f, fv = feature.transform(z), feature.transform(zv)
            bank_id = f"{seed}_kernel_m{m}_g{multiplier:g}"
            banks.append({"bank_id":bank_id,"geometry_fit_seconds":time.perf_counter()-started,
                          "development_transform_seconds":0.0})
            for alpha in (1e-4,1e-2,1e-1):
                started = time.perf_counter()
                model = BoundedRidge(alpha=alpha,bounds=(0,1)).fit(f,y)
                elapsed = time.perf_counter()-started
                retain(Forecast("kernel",m,model,feature,scaler), f"{bank_id}_a{alpha:g}",128,
                       model.predict(fv),{"alpha":alpha,"gamma_multiplier":multiplier,"gamma":gamma},elapsed,bank_id)
        retain(Forecast("persistence",m),f"{seed}_persistence_m{m}",1,np.repeat(xv[:,-1:],3,axis=1),{},0.)
        mean = y.mean(axis=0)
        retain(Forecast("constant",m,mean=mean),f"{seed}_constant_m{m}",1,np.tile(mean,(len(xv),1)),{},0.)
    if set(selected) != set(FAMILIES) or len(records) != 486:
        raise AssertionError(f"Incomplete development family: {len(records)} candidates")
    write_json(output/f"development_{seed}.json",{"candidates":records,"banks":banks,
        "selected":[selected[k][2] for k in FAMILIES],"training_rows":len(clip_index)})
    return selected


def kl_upper(mean, n, alpha):
    if not 0 <= mean <= 1 or n < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bounded-mean endpoint arguments")
    if mean == 1:
        return 1.0
    c = math.log(1/alpha) / n
    if mean == 0:
        return -math.expm1(-c)
    lower, upper = mean, 1.0
    for _ in range(80):
        p = (lower+upper)/2
        if p == 1:
            break
        divergence = mean*math.log(mean/p)+(1-mean)*math.log((1-mean)/(1-p))
        if divergence > c:
            upper = p
        else:
            lower = p
    return upper


def evaluate(energy, metadata, selected_by_seed, output):
    alpha = .05 / (len(SEEDS)*len(FAMILIES)*len(HORIZONS))
    records, scores = [], []
    for seed, selected in selected_by_seed.items():
        for family in FAMILIES:
            _, forecast, development = selected[family]
            entry = {**development,"upper_tail_alpha":alpha}
            for split in ("calibration","test"):
                indices = metadata.index[metadata.split == split].to_numpy()
                x, y = histories(energy[indices], forecast.history)
                started = time.perf_counter()
                prediction = forecast.predict(x)
                elapsed = time.perf_counter()-started
                groups, losses = group_losses(prediction, y, metadata.loc[indices,"speaker"].to_numpy())
                if not np.isfinite(losses).all() or np.any(losses < 0) or np.any(losses > 1):
                    raise AssertionError("Invalid speaker losses")
                entry[split+"_mse"] = losses.mean(axis=0).tolist()
                entry[split+"_speakers"] = len(groups)
                entry[split+"_prediction_seconds"] = elapsed
                entry[split+"_prediction_rows"] = len(x)
                if split == "calibration":
                    entry["mse_upper"] = [kl_upper(float(q),len(groups),alpha) for q in losses.mean(axis=0)]
                for speaker, row in zip(groups, losses):
                    for h, value in zip(HORIZONS,row):
                        scores.append({"seed":seed,"family":family,"split":split,"speaker":speaker,
                                       "horizon":int(h),"mse":float(value)})
            entry["rmse_upper"] = np.sqrt(entry["mse_upper"]).tolist()
            entry["all_horizon_pass"] = {str(t):bool(max(entry["mse_upper"]) <= t*t) for t in (.1,.2,.3)}
            records.append(entry)
            print(f"scored frozen model {seed} {family}",flush=True)
    pd.DataFrame(scores).to_csv(output/"speaker_losses.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    write_json(output/"results.json",{"confidence":.95,"n_upper_statements":72,"upper_tail_alpha":alpha,
               "confidence_scope":"all reported selected families, seeds and horizons, conditional on development; iid speaker-record bundles assumed",
               "results":records})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    source_hashes = {str(p.relative_to(EXTENSION)):digest(p) for p in
        [Path(__file__),HERE/"protocol.md",EXTENSION/"src/certified_forecasting/features.py",
         EXTENSION/"src/certified_forecasting/models.py"]}
    environment = {"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                   "sklearn":sklearn.__version__,"platform":platform.platform(),"blas_threads":1,
                   "source_hashes":source_hashes}
    write_json(args.output/"environment.json",environment)
    try:
        with threadpool_limits(limits=1):
            energy, metadata, provenance = prepare(args.archive,args.output)
            print(json.dumps({"groups":provenance["groups"],"clips":provenance["clips"]}),flush=True)
            selected = {seed:develop(energy,metadata,seed,args.output) for seed in SEEDS}
            freeze = {"protocol_sha256":digest(HERE/"protocol.md"),"development":{str(s):digest(args.output/f"development_{s}.json") for s in SEEDS}}
            for seed in SEEDS:
                with (args.output/f"selected_models_{seed}.pkl").open("xb") as stream:
                    pickle.dump(selected[seed],stream,protocol=5)
                freeze[str(seed)+"_model_sha256"] = digest(args.output/f"selected_models_{seed}.pkl")
            write_json(args.output/"development_freeze.json",freeze)
            print("All development selections frozen before held-out loss evaluation.",flush=True)
            evaluate(energy,metadata,selected,args.output)
            write_json(args.output/"completion.json",{"status":"complete","elapsed_seconds":time.perf_counter()-started,
                       "development_freeze_sha256":digest(args.output/"development_freeze.json"),
                       "result_sha256":digest(args.output/"results.json"),"loss_sha256":digest(args.output/"speaker_losses.csv.gz")})
    except Exception:
        write_json(args.output/"failure.json",{"status":"failed","traceback":traceback.format_exc(),
                     "elapsed_seconds":time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
