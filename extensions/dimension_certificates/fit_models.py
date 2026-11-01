"""Training-only representation fits and independent validation selection."""
from __future__ import annotations

import pickle
import time

from common import (array_digest, flat, load_runtime, make_episodes, origins,
                    save_npz, utc, windows, write_json)
import numpy as np

load_runtime()
from certified_forecasting.features import NestedSoftFeatures
from certified_forecasting.models import BlockBoundedRidge, BoundedRidge, PolynomialMap, Persistence
from certified_forecasting.certificates import episode_mse


def selection_key(row):
    return (float(np.mean(row["validation_mse"])), row["n_features"], row["history"],
            row["cap"], row["id"])


def norm_metadata(model, blocks):
    coefficient = np.asarray(model.coef_)
    value = (sum(float(np.linalg.norm(coefficient[start:stop])) for start, stop in blocks)
             if blocks is not None else float(np.linalg.norm(coefficient)))
    return {"coefficient_norm": value,
            "coefficient_metric": "sum_block_frobenius" if blocks is not None else "frobenius",
            "optimization_gap": float(getattr(model, "optimization_gap_", 0.0)),
            "solver_iterations": int(getattr(model, "n_iter_", 0)),
            "solver_converged": bool(getattr(model, "converged_", True))}


def fit_development(job, cfg, destination):
    start = time.perf_counter()
    training = make_episodes(job, cfg, "train")
    validation = make_episodes(job, cfg, "validation")
    train_arrays, validation_arrays = windows(training, cfg), windows(validation, cfg)
    banks, transforms, models = {}, {}, {}
    rows, scores, bank_records = [], {}, {}

    def add(identifier, family, history, cap, bank_id, stage, transform, model,
            Fvalidation, targets, blocks, elapsed):
        prediction = model.predict(Fvalidation).reshape(targets.shape)
        losses = episode_mse(targets, prediction)
        row = {"id": identifier, "family": family, "history": history, "cap": cap,
               "bank_id": bank_id, "stage": stage, "n_features": Fvalidation.shape[1],
               "blocks": blocks, "kappa": cfg["kappa"],
               "validation_mse": losses.mean(axis=0).tolist(),
               "fit_seconds": elapsed, **norm_metadata(model, blocks)}
        if row["coefficient_norm"] > cfg["kappa"] + 1e-9:
            raise AssertionError("Infeasible fitted coefficient budget")
        rows.append(row)
        scores[identifier] = losses
        models[identifier] = model
        if transform is not None:
            transforms[identifier] = transform

    for history in cfg["histories"]:
        X, Y = train_arrays[history]
        Xv, Yv = validation_arrays[history]
        xf, yf, xv = flat(X), flat(Y), flat(Xv)
        for kind in ("kahm", "rbf", "affine"):
            for cap in cfg["caps"] if kind == "kahm" else [2]:
                identifier = f"{kind}_m{history}_c{cap}"
                started = time.perf_counter()
                bank = NestedSoftFeatures(
                    cluster_counts=cfg["cluster_counts"], kind=kind,
                    seed=job["geometry_seeds"][str(history)], omega=cfg["omega"],
                    max_fit=cfg["max_geometry_inputs"],
                    max_train_per_cluster=cfg["max_train_per_cluster"],
                    subspace_dim=None if kind == "kahm" and cap == 20 else cap,
                    input_scaling=cfg["input_scaling"],
                ).fit(xf)
                F, Fv = bank.transform(xf), bank.transform(xv)
                geometry_seconds = time.perf_counter() - started
                if not np.all(np.isfinite(F)) or not np.all(np.isfinite(Fv)):
                    raise AssertionError("Nonfinite representation")
                banks[identifier] = bank
                bank_records[identifier] = {
                    "id": identifier, "kind": kind, "history": history, "cap": cap,
                    "geometry_seconds": geometry_seconds,
                    "input_scaling": bank.input_scaling_metadata(),
                    "dimension_policy": bank.dimension_policy_metadata(),
                    "training_feature_sha256": array_digest(F),
                    "validation_feature_sha256": array_digest(Fv),
                    "training_feature_min": float(F.min()),
                    "training_feature_max": float(F.max()),
                }
                for stage, block in enumerate(bank.block_slices_, 1):
                    started = time.perf_counter()
                    slices = bank.block_slices_[:stage]
                    blocks = [[int(item.start), int(item.stop)] for item in slices]
                    model = BlockBoundedRidge(
                        alpha=cfg["alpha"], kappa=cfg["kappa"], bounds=(-1, 1),
                        block_slices=slices, tol=cfg["solver_tolerance"],
                        max_iter=cfg["solver_max_iterations"],
                    ).fit(F[:, :block.stop], yf)
                    add(identifier + f"_s{stage}", kind, history, cap, identifier, stage,
                        None, model, Fv[:, :block.stop], Yv, blocks,
                        geometry_seconds + time.perf_counter() - started)

        started = time.perf_counter()
        ar = BoundedRidge(cfg["alpha"], cfg["kappa"], (-1, 1)).fit(xf, yf)
        add(f"ridge_ar_m{history}", "ridge_ar", history, 0, None, None, None,
            ar, xv, Yv, None, time.perf_counter() - started)
        started = time.perf_counter()
        poly = PolynomialMap(degree=2).fit(xf)
        polynomial = BoundedRidge(cfg["alpha"], cfg["kappa"], (-1, 1)).fit(poly.transform(xf), yf)
        add(f"polynomial_m{history}", "polynomial", history, 0, None, None, poly,
            polynomial, poly.transform(xv), Yv, None, time.perf_counter() - started)

    history = cfg["histories"][0]
    X, Y = train_arrays[history]
    Xv, Yv = validation_arrays[history]
    persistence = Persistence().fit(flat(X), flat(Y))
    prediction = persistence.predict(flat(Xv)).reshape(Yv.shape)
    loss = episode_mse(Yv, prediction)
    rows.append({"id": "persistence", "family": "persistence", "history": history,
                 "cap": 0, "bank_id": None, "stage": None, "n_features": history,
                 "blocks": None, "kappa": 1.0, "validation_mse": loss.mean(axis=0).tolist(),
                 "fit_seconds": 0.0, "coefficient_norm": 1.0,
                 "coefficient_metric": "per_horizon_euclidean",
                 "optimization_gap": 0.0, "solver_iterations": 0, "solver_converged": True})
    models["persistence"], scores["persistence"] = persistence, loss
    constant = BoundedRidge(cfg["alpha"], 0.0, (-1, 1)).fit(
        np.zeros((len(flat(X)), 1)), flat(Y))
    add("constant", "constant", history, 0, None, None, None, constant,
        np.zeros((len(flat(Xv)), 1)), Yv, None, 0.0)
    rows[-1]["kappa"] = 0.0
    if len(rows) != cfg["development_candidates_per_repetition"]:
        raise AssertionError(f"Unexpected development count {len(rows)}")

    by_id = {row["id"]: row for row in rows}
    choices, evaluation = {}, []
    for arm in (*cfg["policies"], "rbf", "affine"):
        kind = "kahm" if arm in cfg["policies"] else arm
        eligible = [row for row in rows if row["family"] == kind]
        if arm in ("cap2", "native20"):
            cap = 2 if arm == "cap2" else 20
            eligible = [row for row in eligible if row["cap"] == cap]
        winner = min(eligible, key=selection_key)
        choices[arm] = {"bank_id": winner["bank_id"], "winner_id": winner["id"],
                        "winner_stage": winner["stage"], "history": winner["history"],
                        "cap": winner["cap"], "selection_key": list(selection_key(winner))}
        for stage in range(1, len(cfg["cluster_counts"]) + 1):
            fitted_id = winner["bank_id"] + f"_s{stage}"
            evaluation.append({"alias": f"{arm}_s{stage}", "arm": arm,
                               "selected_forecast": stage == winner["stage"],
                               **by_id[fitted_id]})
    for family in ("ridge_ar", "polynomial", "persistence", "constant"):
        winner = min((row for row in rows if row["family"] == family), key=selection_key)
        choices[family] = {"winner_id": winner["id"], "history": winner["history"],
                           "selection_key": list(selection_key(winner))}
        evaluation.append({"alias": family, "arm": family, "selected_forecast": True, **winner})
    if len(evaluation) != cfg["evaluated_models_per_repetition"]:
        raise AssertionError("Evaluation family count changed")

    # Identical admissible cap requests must produce identical native arrays.
    for history in cfg["histories"]:
        same = [bank_records[f"kahm_m{history}_c{cap}"] for cap in cfg["caps"] if cap >= history]
        for field in ("training_feature_sha256", "validation_feature_sha256"):
            if len({record[field] for record in same}) != 1:
                raise AssertionError("Ambient-limited KAHM caps differ")

    selected_ids = {row["id"] for row in evaluation}
    selected_banks = {row["bank_id"] for row in evaluation if row["bank_id"] is not None}
    payload = {"banks": {key: banks[key] for key in sorted(selected_banks)},
               "models": {key: models[key] for key in sorted(selected_ids)},
               "transforms": {key: transforms[key] for key in sorted(selected_ids) if key in transforms}}
    with (destination / "fitted_models.pkl").open("xb") as stream:
        pickle.dump(payload, stream, protocol=5)
    coefficients = {}
    for identifier in sorted(selected_ids):
        model = models[identifier]
        if identifier == "persistence":
            coef = np.zeros((by_id[identifier]["history"], len(cfg["horizons"])))
            coef[-1] = 1
            coefficients[identifier + "__coefficient"] = coef
            coefficients[identifier + "__intercept"] = np.zeros(len(cfg["horizons"]))
        else:
            coefficients[identifier + "__coefficient"] = np.asarray(model.coef_)
            coefficients[identifier + "__intercept"] = np.asarray(model.intercept_)
    save_npz(destination / "coefficients.npz", coefficients)
    save_npz(destination / "validation_scores.npz", {
        **scores, "episode_ids": np.arange(cfg["n_validation"]),
        "origin_indices": origins(cfg), "horizons": np.asarray(cfg["horizons"]),
    })
    write_json(destination / "development.json", {
        "job": job, "completed_utc": utc(), "elapsed_seconds": time.perf_counter() - start,
        "train_sha256": array_digest(training), "validation_sha256": array_digest(validation),
        "windows": {str(history): {"train_inputs_sha256": array_digest(x),
                                    "train_targets_sha256": array_digest(y),
                                    "validation_inputs_sha256": array_digest(validation_arrays[history][0]),
                                    "validation_targets_sha256": array_digest(validation_arrays[history][1])}
                    for history, (x, y) in train_arrays.items()},
        "banks": bank_records, "candidates": rows, "choices": choices,
        "evaluation": evaluation,
        "n_evaluation_aliases": len(evaluation), "n_distinct_stored_forecasts": len(selected_ids),
    })
    return {"job": job["id"], "seconds": time.perf_counter() - start,
            "development_candidates": len(rows), "evaluated_models": len(evaluation)}


def feature_matrix(row, payload, x, bank_cache):
    identifier = row["id"]
    if row["bank_id"] is not None:
        bank_id = row["bank_id"]
        if bank_id not in bank_cache:
            bank_cache[bank_id] = payload["banks"][bank_id].transform(x)
        return bank_cache[bank_id][:, :row["n_features"]]
    if row["family"] == "polynomial":
        return payload["transforms"][identifier].transform(x)
    if row["family"] == "constant":
        return np.zeros((len(x), 1))
    return x
