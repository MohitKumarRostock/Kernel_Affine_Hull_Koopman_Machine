"""Honest episode cross-fitting with a frozen full-A common predictor.

The immutable corrected KAHM runtime supplies every numerical primitive.
Every fold repeats common-model construction and B selection on its A episode
complement.  Fold target normalization also excludes held-out episodes.
Full-A departure features regress two declared targets, OOF and in-sample
residuals, around the same full-A common forecast and B-fitted shared bias.
No C or terminal target is accepted by the fitting entry points.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import numpy as np

PREVIOUS = Path(__file__).resolve().parents[2] / "preprocessing_dimension"
if str(PREVIOUS / "code") not in sys.path:
    sys.path.insert(0, str(PREVIOUS / "code"))
import preprocessing_model as previous
import fixed_common_secondary as secondary
fd = previous.fd

SCALERS = ("standard", "minmax", "tanh_z_1")
CAPS = (1, 2, 4, 8, 12)
TARGET_MODES = ("crossfit", "insample")
KINDS = ("kahm", "rbf")
SLOPE_METHODS = tuple(f"{mode}_{kind}" for mode in TARGET_MODES
                      for kind in ("kahm", "rbf", "matched_rbf"))
METHODS = ("common", "shared_bias", "half_common", "half_shared_bias", *SLOPE_METHODS)


def _candidate_id(scaler, cap):
    return f"{scaler}__cap{cap}"


def _checked_raw(bank, x, tag):
    m = fd.load_runtime()
    raw, diagnostics = m._raw_feature_maps(bank, x, {}, tag)
    secondary._guard_geometry(diagnostics, tag)
    if any(not np.isfinite(value).all() for value in raw.values()):
        raise FloatingPointError("Nonfinite cross-fit feature map at " + tag)
    return raw, diagnostics


def _score(prediction, target, variance):
    mse = np.mean((target - prediction) ** 2, axis=0)
    if not np.isfinite(mse).all():
        raise FloatingPointError("Nonfinite B loss")
    return float(np.mean(mse / variance)), mse


@dataclass
class CommonModel:
    scaler: str
    cap: int
    bank: dict
    feature_scaler: dict
    head: dict
    raw_scaler: dict
    raw_head: dict

    def predictive_state(self):
        return {"scaler": self.scaler, "cap": self.cap, "bank": self.bank,
                "feature_scaler": self.feature_scaler, "head": self.head,
                "raw_scaler": self.raw_scaler, "raw_head": self.raw_head}

    def state_hash(self):
        return fd.state_hash(self.predictive_state())

    def predict(self, x):
        m = fd.load_runtime()
        x = m._matrix(x, "common query histories")
        if x.shape[1] != len(self.raw_scaler["mean"]):
            raise ValueError("Common query history width differs")
        raw, diagnostic = _checked_raw(self.bank, x, "common_query")
        return self.predict_from_raw(x, raw), diagnostic

    def predict_from_raw(self, x, raw):
        m = fd.load_runtime()
        baseline = m._ridge_predict(self.raw_head, m._standardize(self.raw_scaler, x))
        features = secondary._bounded(m, raw["common"], self.feature_scaler)
        forecast = baseline + m._ridge_predict(self.head, features)
        if not np.isfinite(forecast).all():
            raise FloatingPointError("Nonfinite common forecast")
        return forecast


def fit_common_only(tx, ty, bx, by, *, seed, config, tag="common"):
    """Fit/select one entire common procedure using training rows and B only.

    No full-A, held-out fold, C, or terminal target is available here.  Every
    candidate uses the training subset's own scalers, bank and variance.
    Candidate order is cap first, then standard/minmax/tanh_z_1.
    """
    m = fd.load_runtime()
    tx, ty, bx, by = fd._fit_arrays(tx, ty, bx, by)
    start = time.perf_counter()
    base = fd.fit_baseline(tx, ty, bx, by, config=config)
    effective = base.config
    models, records, predictions, basis_arrays, banks_diagnostics = {}, [], {}, {}, {}
    predictions[f"{tag}__raw_ridge"] = base.predictions_b.copy()
    nodes = {}
    for cap in CAPS:
        for scaler in SCALERS:
            key = _candidate_id(scaler, cap)
            bank = m._InstrumentedBank(seed=int(seed), config=fd._config(cap, "native", config),
                                        geometry_scaling=scaler).fit(tx)
            if any(row.get("factor_error") for row in bank.diagnostics["local_models"]):
                raise FloatingPointError("Cross-fit common bank factor failure")
            state = bank.state("kahm")
            for local in state["models"]:
                local.pop("reference_distance", None)
            train_raw, train_diagnostics = _checked_raw(state, tx, tag + "/" + key + "/train")
            b_raw, b_diagnostics = _checked_raw(state, bx, tag + "/" + key + "/B")
            feature_scaler = m._scaler_fit(train_raw["common"], effective["feature_scale_floor"])
            ft = secondary._bounded(m, train_raw["common"], feature_scaler)
            fb = secondary._bounded(m, b_raw["common"], feature_scaler)
            head, head_record = m._select_ridge(ft, base.residual_a, fb, by, base.predictions_b,
                                               effective["alpha_grid"], nodes, tag + "/" + key)
            prediction = base.predictions_b + m._ridge_predict(head, fb)
            score, mse = _score(prediction, by, base.loss_scale)
            model = CommonModel(scaler, cap, state, feature_scaler, head,
                                base.input_scaler, base.head)
            models[key] = model
            predictions[f"{tag}__{key}__common"] = prediction
            records.append({"candidate_id": key, "scaler": scaler, "cap": cap,
                            "score": score, "mse_by_horizon": mse.tolist(),
                            "selected_alpha_by_horizon": head["alpha_by_horizon"].tolist(),
                            "head_selection": head_record,
                            "B_prediction_sha256": fd.array_hash(prediction),
                            "state_sha256": model.state_hash()})
            dimension_records = secondary._dimension_records(state, bank.diagnostics)
            for index, local in enumerate(state["models"]):
                prefix = f"{tag}__{key}__local{index}"
                basis_arrays[prefix + "__basis"] = local["p"].copy()
                basis_arrays[prefix + "__references"] = local["q"].copy()
            banks_diagnostics[key] = {
                "bank": bank.diagnostics, "geometry_scaler": state.get("geometry_scaler"),
                "standard_mean": state["mean"], "standard_scale": state["scale"],
                "common_feature_scaler": feature_scaler,
                "local_dimensions": dimension_records,
                "train_geometry": train_diagnostics, "B_geometry": b_diagnostics,
            }
    selected_index = int(np.argmin([record["score"] for record in records]))
    selected_record = records[selected_index]
    chosen = selected_record["candidate_id"]
    selection = {
        "selected_candidate": chosen, "selected_scaler": models[chosen].scaler,
        "selected_cap": models[chosen].cap,
        "selected_alpha_by_horizon": models[chosen].head["alpha_by_horizon"].tolist(),
        "raw_alpha_by_horizon": base.head["alpha_by_horizon"].tolist(),
        "selected_index": selected_index, "B_scores": records, "selection_split": "B",
        "common_gamma": 1.0,
        "training_target_variance": base.loss_scale.tolist(),
        "normalizer": "This common fit's training-target population variance only; no held-out A targets",
        "candidate_order": [record["candidate_id"] for record in records],
        "tie_rule": "First minimum: cap ascending, then standard/minmax/tanh_z_1",
        "input_hashes": copy.deepcopy(base.input_hashes),
        "selected_state_sha256": models[chosen].state_hash(),
    }
    diagnostics = secondary.json_safe({
        "tag": tag, "training_rows": len(tx), "B_rows": len(bx), "seed": int(seed),
        "banks": banks_diagnostics, "raw_ridge": base.diagnostics,
        "baseline_fit_nodes": base.nodes, "common_head_nodes": nodes,
        "fit_wall_seconds": time.perf_counter() - start,
        "fit_splits": ["supplied training subset", "B selection"],
        "all_history_and_feature_scalers_fit_on_training_subset": True,
    })
    return models[chosen], selection, predictions, basis_arrays, diagnostics


@dataclass
class DepartureState:
    scaler: str
    cap: int
    kind: str
    bank: dict
    feature_scaler: dict
    head: dict
    mode: str
    mean_training_correction: np.ndarray


@dataclass
class CrossfitSuite:
    common: CommonModel
    half_common: CommonModel
    fold_states: dict
    bias: np.ndarray
    half_bias: np.ndarray
    departures: dict
    departure_candidates: dict
    selections: dict
    training_arrays: dict
    validation_predictions: dict
    basis_arrays: dict
    config: dict
    diagnostics: dict
    etas: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def frozen_state(self):
        """Pre-C state, excluding any mutable legacy-primary calibration."""
        return {"common": self.common.predictive_state(),
                "half_common": self.half_common.predictive_state(),
                "fold_states": {key: value.predictive_state() for key, value in self.fold_states.items()},
                "bias": self.bias, "half_bias": self.half_bias,
                "departures": self.departures, "departure_candidates": self.departure_candidates,
                "selections": self.selections, "config": self.config,
                "OOF_common": self.training_arrays["A_oof_common"],
                "OOF_residual": self.training_arrays["A_oof_residual"],
                "insample_residual": self.training_arrays["A_insample_residual"]}

    def frozen_hash(self):
        return fd.state_hash(self.frozen_state())

    def _uncalibrated(self, x):
        m = fd.load_runtime()
        x = m._matrix(x, "cross-fit query histories")
        if x.shape[1] != len(self.common.raw_scaler["mean"]):
            raise ValueError("Cross-fit query history width differs")
        cache, geometry = {}, {}
        common_key = _candidate_id(self.common.scaler, self.common.cap)
        cache[common_key], geometry[common_key] = _checked_raw(self.common.bank, x, "full_common_query")
        common = self.common.predict_from_raw(x, cache[common_key])
        half, half_geometry = self.half_common.predict(x)
        shared = common + self.bias
        predictions = {"common": common, "shared_bias": shared,
                       "half_common": half, "half_shared_bias": half + self.half_bias}
        feature_checks = {}
        for method in SLOPE_METHODS:
            state = self.departures[method]
            key = _candidate_id(state.scaler, state.cap)
            if key not in cache:
                cache[key], geometry[key] = _checked_raw(state.bank, x, key + "/query")
            features = secondary._bounded(m, cache[key][state.kind], state.feature_scaler)
            predictions[method] = shared + secondary._slope(state.head, features)
            feature_checks[method] = m._feature_summary(features, np.sqrt(features.shape[1]))
        if tuple(predictions) != METHODS:
            raise AssertionError("Cross-fit method roster differs from declared order")
        if any(value.shape != common.shape or not np.isfinite(value).all() for value in predictions.values()):
            raise FloatingPointError("Invalid cross-fit query forecast")
        return predictions, {"geometry": geometry, "half_common_geometry": half_geometry,
                             "feature_bounds": feature_checks}

    def calibrate(self, cx, cy):
        """Select only the six centered correction etas from C."""
        if self.etas:
            raise RuntimeError("Cross-fit suite already calibrated")
        m = fd.load_runtime()
        cx, cy = m._matrix(cx, "C histories"), m._matrix(cy, "C targets")
        if len(cx) != len(cy) or cy.shape[1] != len(self.bias):
            raise ValueError("C histories/targets differ from fitted shape")
        before = self.frozen_hash()
        raw, diagnostics = self._uncalibrated(cx)
        output = {key: value.copy() for key, value in raw.items()}
        records, etas = {}, {}
        for method in SLOPE_METHODS:
            eta, output[method], record = m._select_gamma(cy, raw["shared_bias"], raw[method], self.config["gamma_grid"])
            etas[method] = eta
            records[method] = {**record, "selected_eta": eta.tolist(), "selection_split": "C"}
        if self.frozen_hash() != before:
            raise AssertionError("C selection changed cross-fit development state")
        self.etas = etas
        self.calibration = secondary.json_safe({
            "choices": records, "pre_C_hash": before, "post_C_hash": before,
            "C_X_sha256": fd.array_hash(cx), "C_Y_sha256": fd.array_hash(cy),
            "common_state_sha256": self.common.state_hash(),
            "bias_sha256": fd.array_hash(self.bias),
            "shared_baseline_sha256": fd.state_hash((self.common.predictive_state(), self.bias)),
            "uncalibrated_prediction_sha256": {key: fd.array_hash(value) for key, value in raw.items()},
            "calibrated_prediction_sha256": {key: fd.array_hash(value) for key, value in output.items()},
            "query": diagnostics,
        })
        return {"predictions": output, "eta1_predictions": raw, "records": self.calibration}

    def predict(self, x):
        """Fixed-state prediction from histories only; no fitting or selection."""
        if set(self.etas) != set(SLOPE_METHODS):
            raise RuntimeError("Calibrate the cross-fit suite before predicting")
        m = fd.load_runtime()
        before = self.frozen_hash()
        raw, diagnostics = self._uncalibrated(x)
        output = {key: value.copy() for key, value in raw.items()}
        bounds = {}
        for method in SLOPE_METHODS:
            state = self.departures[method]
            output[method] = m._blend(raw["shared_bias"], raw[method], self.etas[method])
            bound = secondary._centered_bound(state.head, self.etas[method])
            observed = np.max(np.abs(output[method] - raw["shared_bias"]), axis=0)
            limit = np.asarray(bound["correction_bound_by_horizon"])
            violations = int(np.count_nonzero(observed > limit * (1 + 1e-10) + 1e-12))
            if violations:
                raise AssertionError("Cross-fit correction exceeds the centered readout bound")
            bounds[method] = {**bound, "observed_max_abs_correction_by_horizon": observed.tolist(),
                              "query_bound_violations": violations}
        if self.frozen_hash() != before:
            raise AssertionError("Cross-fit query changed development state")
        diagnostics.update(bounds=bounds, frozen_state_before=before, frozen_state_after=before,
                           common_state_sha256=self.common.state_hash(), bias_sha256=fd.array_hash(self.bias),
                           shared_baseline_sha256=fd.state_hash((self.common.predictive_state(), self.bias)),
                           etas={key: value.tolist() for key, value in self.etas.items()})
        return {"predictions": output, "eta1_predictions": raw,
                "diagnostics": secondary.json_safe(diagnostics)}


def fit_crossfit(ax, ay, groups_a, bx, by, *, seed, config, reference_primary):
    """Fit the declared four-fold/common/correction procedure using A and B."""
    start = time.perf_counter()
    m = fd.load_runtime()
    ax, ay, bx, by = fd._fit_arrays(ax, ay, bx, by)
    groups = np.asarray(groups_a)
    if groups.shape != (len(ax),) or not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("A episode IDs must be an integer array aligned with A rows")
    unique = np.unique(groups)
    if len(unique) < 8 or len(unique) % 4 or np.any(unique < 0):
        raise ValueError("Require at least eight nonnegative A episode IDs, divisible by four")
    fold_ids = groups % 4
    if set(np.unique(fold_ids)) != {0, 1, 2, 3}:
        raise ValueError("Original A episode IDs modulo four must populate every fold")
    actual_hashes = {key: fd.array_hash(value) for key, value in
                     (("A_X", ax), ("A_Y", ay), ("B_X", bx), ("B_Y", by))}
    if actual_hashes != reference_primary.diagnostics["input_hashes"]:
        raise ValueError("Reference primary was fitted on different A/B data")
    if reference_primary.calibration or reference_primary.gammas:
        raise ValueError("Construct cross-fit suite before any reference-primary C calibration")
    full_key = reference_primary.selections["selected"]["common"]
    full_candidate = reference_primary.candidates[full_key]
    full_bank = reference_primary.banks[full_candidate.common_bank]
    full_scaler = full_bank.diagnostics["arm"]
    base = reference_primary.shared.baseline_fit
    common = CommonModel(full_scaler, full_candidate.cap, full_bank.bank,
                         full_bank.scalers["common"], full_candidate.heads["common"],
                         base.input_scaler, base.head)
    full_common_hash = common.state_hash()
    common_a, common_a_diag = common.predict(ax)
    common_b, common_b_diag = common.predict(bx)
    np.testing.assert_array_equal(common_b, full_candidate.b_predictions["common"])
    bias = np.mean(by - common_b, axis=0)
    shared_b = common_b + bias
    shared_hash = fd.state_hash((common.predictive_state(), bias))
    full_selection = {
        "selected_candidate": full_key, "selected_scaler": full_scaler,
        "selected_cap": full_candidate.cap,
        "selected_alpha_by_horizon": common.head["alpha_by_horizon"].tolist(),
        "raw_alpha_by_horizon": common.raw_head["alpha_by_horizon"].tolist(),
        "training_target_variance": base.loss_scale.tolist(),
        "selection_split": "B", "common_gamma": 1.0,
        "source": "reference_primary.selections['selected']['common'], before C",
        "state_sha256": full_common_hash, "B_prediction_sha256": fd.array_hash(common_b),
        "training_episode_ids": unique.tolist(), "training_row_indices": list(range(len(ax))),
    }

    oof = np.empty_like(ay)
    assignment_count = np.zeros(len(ax), dtype=np.int64)
    fold_states, fold_records, validation, basis_arrays, fit_diagnostics = {}, [], {}, {}, {}
    for fold in range(4):
        train_mask, heldout_mask = fold_ids != fold, fold_ids == fold
        train_rows, heldout_rows = np.flatnonzero(train_mask), np.flatnonzero(heldout_mask)
        train_groups, heldout_groups = np.unique(groups[train_mask]), np.unique(groups[heldout_mask])
        if len(train_rows) < 2 or len(heldout_rows) < 1 or np.intersect1d(train_groups, heldout_groups).size:
            raise AssertionError("Invalid independent-episode fold")
        tag = f"fold{fold}"
        model, selection, vp, ba, diagnostic = fit_common_only(ax[train_mask], ay[train_mask], bx, by,
                                                                seed=seed, config=config, tag=tag)
        before = model.state_hash()
        forecast, query_diagnostics = model.predict(ax[heldout_mask])
        if model.state_hash() != before:
            raise AssertionError("OOF query changed its fold model")
        oof[heldout_mask] = forecast
        assignment_count[heldout_mask] += 1
        fold_states[tag] = model
        selection.update(fold_id=fold, fold_assignment="original_episode_id modulo 4",
            training_episode_ids=train_groups.tolist(), heldout_episode_ids=heldout_groups.tolist(),
            training_row_indices=train_rows.tolist(), heldout_row_indices=heldout_rows.tolist(),
            training_group_sha256=fd.array_hash(groups[train_mask]),
            heldout_group_sha256=fd.array_hash(groups[heldout_mask]),
            training_rows_sha256=fd.array_hash(train_rows), heldout_rows_sha256=fd.array_hash(heldout_rows),
            heldout_history_sha256=fd.array_hash(ax[heldout_mask]),
            OOF_prediction_sha256=fd.array_hash(forecast),
            heldout_targets_accepted_by_fit=False, state_before_OOF=before, state_after_OOF=before)
        fold_records.append(selection)
        validation.update(vp)
        basis_arrays.update(ba)
        fit_diagnostics[tag] = {"fit": diagnostic, "OOF_query": query_diagnostics}
    if not np.all(assignment_count == 1) or not np.isfinite(oof).all():
        raise AssertionError("Every A row must have exactly one finite OOF prediction")

    half_groups = unique[:len(unique)//2]
    half_mask = np.isin(groups, half_groups)
    half_common, half_selection, vp, ba, half_diagnostics = fit_common_only(
        ax[half_mask], ay[half_mask], bx, by, seed=seed, config=config, tag="half")
    half_b, half_b_diag = half_common.predict(bx)
    half_bias = np.mean(by - half_b, axis=0)
    half_selection.update(training_episode_ids=half_groups.tolist(),
        training_row_indices=np.flatnonzero(half_mask).tolist(),
        state_sha256=half_common.state_hash(), B_prediction_sha256=fd.array_hash(half_b),
        shared_bias=half_bias.tolist(), bias_fit_split="B")
    validation.update(vp)
    basis_arrays.update(ba)
    fit_diagnostics["half"] = {"fit": half_diagnostics, "B_query": half_b_diag}

    residuals = {"crossfit": ay - oof, "insample": ay - common_a}
    training_arrays = {
        "A_oof_common": oof, "A_oof_residual": residuals["crossfit"],
        "A_full_common": common_a, "A_insample_residual": residuals["insample"],
        "A_fold_id": fold_ids.copy(), "A_groups": groups.copy(),
        "A_oof_assignment_count": assignment_count,
        "B_common": common_b, "B_shared_bias": shared_b,
        "B_half_common": half_b, "B_half_shared_bias": half_b + half_bias,
    }

    features, raw_diagnostics = {}, {}
    for cap in CAPS:
        for scaler in SCALERS:
            key = _candidate_id(scaler, cap)
            bank = reference_primary.banks[previous.bank_id(scaler, cap)]
            maps = {}
            for split, x in (("A", ax), ("B", bx)):
                maps[split], diagnostic = _checked_raw(bank.bank, x, "full_departure/" + key + "/" + split)
                raw_diagnostics[key + "/" + split] = diagnostic
            for kind in KINDS:
                # The full-A geometry and post-map scaler are exactly those in
                # the source primary fit; no fold feature alignment is needed.
                feature_scaler = bank.scalers[kind]
                oracle = m._scaler_fit(maps["A"][kind], base.config["feature_scale_floor"])
                np.testing.assert_array_equal(feature_scaler["mean"], oracle["mean"])
                np.testing.assert_array_equal(feature_scaler["scale"], oracle["scale"])
                features[(scaler, cap, kind)] = (
                    bank.bank, feature_scaler,
                    secondary._bounded(m, maps["A"][kind], feature_scaler),
                    secondary._bounded(m, maps["B"][kind], feature_scaler))

    departure_candidates, selected_departures, mode_selections = {}, {}, {}
    for mode in TARGET_MODES:
        candidate_records = {kind: [] for kind in KINDS}
        for cap in CAPS:
            for scaler in SCALERS:
                candidate_key = _candidate_id(scaler, cap)
                for kind in KINDS:
                    bank, feature_scaler, fa, fb = features[(scaler, cap, kind)]
                    path = m._ridge_path(fa, residuals[mode], base.config["alpha_grid"])
                    alpha_predictions = np.stack([shared_b + secondary._slope(head, fb) for head in path])
                    losses = np.mean((by[None, :, :] - alpha_predictions)**2, axis=1)
                    alpha_indices = np.argmin(losses, axis=0)
                    head = m._combine_heads(path, alpha_indices)
                    prediction = shared_b + secondary._slope(head, fb)
                    score, mse = _score(prediction, by, base.loss_scale)
                    mean_correction = np.mean(secondary._slope(head, fa), axis=0)
                    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(residuals[mode]))))
                    if np.max(np.abs(mean_correction)) > tolerance:
                        raise AssertionError("Centered A correction has a nonzero training mean")
                    secondary._centered_bound(head, np.ones(ay.shape[1]))
                    state = DepartureState(scaler, cap, kind, bank, feature_scaler, head,
                                           mode, mean_correction)
                    key = f"{mode}__{candidate_key}__{kind}"
                    departure_candidates[key] = state
                    validation[key] = prediction
                    validation[key + "__alpha_paths"] = alpha_predictions
                    candidate_records[kind].append({
                        "candidate_id": candidate_key, "state_key": key, "scaler": scaler, "cap": cap,
                        "score": score, "mse_by_horizon": mse.tolist(),
                        "selected_alpha_by_horizon": head["alpha_by_horizon"].tolist(),
                        "selected_alpha_indices": alpha_indices.tolist(),
                        "B_mse_by_alpha_horizon": losses.tolist(),
                        "B_prediction_sha256": fd.array_hash(prediction),
                        "B_alpha_prediction_sha256": fd.array_hash(alpha_predictions),
                        "feature_width": int(head["feature_width"]),
                        "A_mean_centered_correction": mean_correction.tolist(),
                        "A_mean_centered_feature_max_abs": float(np.max(np.abs(np.mean(fa-head["mean"],axis=0)))),
                        "residual_target_sha256": fd.array_hash(residuals[mode]),
                        "shared_baseline_sha256": shared_hash,
                    })
        mode_selections[mode] = {}
        for kind in KINDS:
            index = int(np.argmin([record["score"] for record in candidate_records[kind]]))
            chosen = candidate_records[kind][index]
            selected_departures[f"{mode}_{kind}"] = departure_candidates[chosen["state_key"]]
            mode_selections[mode][kind] = {
                "selected_candidate": chosen["candidate_id"],
                "selected_scaler": chosen["scaler"], "selected_cap": chosen["cap"],
                "selected_alpha_by_horizon": chosen["selected_alpha_by_horizon"],
                "selected_index": index, "candidates": candidate_records[kind],
                "selection_split": "B", "candidate_order": [r["candidate_id"] for r in candidate_records[kind]],
                "tie_rule": "First minimum: cap ascending, then standard/minmax/tanh_z_1",
            }
        kahm_choice = mode_selections[mode]["kahm"]
        matched_key = f"{mode}__{kahm_choice['selected_candidate']}__rbf"
        matched_state = departure_candidates[matched_key]
        selected_departures[f"{mode}_matched_rbf"] = matched_state
        mode_selections[mode]["matched_rbf"] = {
            "selected_candidate": kahm_choice["selected_candidate"],
            "selected_scaler": matched_state.scaler, "selected_cap": matched_state.cap,
            "selected_alpha_by_horizon": matched_state.head["alpha_by_horizon"].tolist(),
            "state_key": matched_key,
            "selection_rule": "KAHM-selected scaler/cap; RBF's own B-selected per-horizon penalty",
            "shared_baseline_sha256": shared_hash,
        }

    if common.state_hash() != full_common_hash or fd.state_hash((common.predictive_state(), bias)) != shared_hash:
        raise AssertionError("Cross-fit correction fitting changed full-A common/shared bias")
    selections = {
        "full_common": full_selection, "half_common": half_selection,
        "folds": fold_records, "target_modes": mode_selections,
        "shared_bias": {"value": bias.tolist(), "fit_split": "B", "gamma": 1.0,
                        "formula": "mean_B(Y_B - full_A_common_B)", "state_sha256": shared_hash},
        "departure_normalizer": "Full A target population variance, floored by target_variance_floor",
        "A_target_variance": base.loss_scale.tolist(),
    }
    diagnostics = secondary.json_safe({
        "schema": "episode-crossfit-full-common-v1", "seed": int(seed),
        "fold_assignment": "original A episode IDs modulo 4", "number_of_folds": 4,
        "input_hashes": actual_hashes, "groups_A_sha256": fd.array_hash(groups),
        "A_rows": len(ax), "B_rows": len(bx), "A_episodes": len(unique),
        "OOF_assignment_counts": {str(n): int(np.count_nonzero(assignment_count == n)) for n in np.unique(assignment_count)},
        "OOF_fold_row_counts": {str(k): int(np.count_nonzero(fold_ids == k)) for k in range(4)},
        "OOF_fold_episode_counts": {str(k): int(np.count_nonzero(unique % 4 == k)) for k in range(4)},
        "fold_tuning": "Independent complement-only fit and normalization, B selection in each fold",
        "fold_fit_seeds": {str(k): int(seed) for k in range(4)},
        "full_A_feature_state_reused_from_reference_primary": True,
        "fold_outputs_used_as_features": False,
        "full_common_state_sha256": full_common_hash, "shared_baseline_sha256": shared_hash,
        "half_common_state_sha256": half_common.state_hash(),
        "full_common_A_geometry": common_a_diag, "full_common_B_geometry": common_b_diag,
        "common_fits": fit_diagnostics, "full_A_departure_geometry": raw_diagnostics,
        "residual_target_hashes": {key: fd.array_hash(value) for key, value in residuals.items()},
        "centered_target_modes": list(TARGET_MODES), "methods": list(METHODS),
        "bias_target_modes_share_exactly_one_state": True,
        "interpretation": "OOF residuals of complement-trained common procedures are transferred to the separately fitted full-A common predictor",
        "fit_wall_seconds": time.perf_counter() - start,
    })
    return CrossfitSuite(common, half_common, fold_states, bias, half_bias,
                        selected_departures, departure_candidates, selections, training_arrays,
                        validation, basis_arrays, copy.deepcopy(base.config), diagnostics)


def manufactured_fixture():
    """Small smoke and state-isolation checks; independent audit is separate."""
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(62819047)
    ax, bx, cx, query = [rng.normal(size=(n, 12)) for n in (64, 24, 20, 17)]
    groups = np.repeat(np.arange(16, dtype=np.int64), 4)
    def target(x):
        return np.column_stack((x[:,0] + .2*np.sin(x[:,1]), x[:,2]*x[:,3], np.cos(x[:,4])))
    ay, by, cy = [target(x) + .03*rng.normal(size=(len(x),3)) for x in (ax,bx,cx)]
    m = fd.load_runtime()
    config = {key: copy.deepcopy(value) for key,value in m.DEFAULT_CONFIG.items()
              if key not in ("rank","dimension_policy")}
    config.update(blocks=[2],max_references=24,bank_row_cap=64,alpha_grid=[1e-4,.01,1.],
                  raw_rbf_landmarks=16,raw_rbf_bandwidth_factors=[1.],
                  raw_hgb_learning_rates=[.1],raw_hgb_max_leaf_nodes=[7],raw_hgb_max_iter=8)
    with threadpool_limits(limits=1):
        reference = previous.fit_primary(ax,ay,bx,by,seed=76102,config=config)
        suite = fit_crossfit(ax,ay,groups,bx,by,seed=76102,config=config,reference_primary=reference)
        assert np.all(suite.training_arrays["A_oof_assignment_count"] == 1)
        for fold in range(4):
            ids = groups % 4 == fold
            expected,_ = suite.fold_states[f"fold{fold}"].predict(ax[ids])
            np.testing.assert_array_equal(expected,suite.training_arrays["A_oof_common"][ids])
        before = suite.frozen_hash()
        reference.calibrate(cx,cy)
        assert suite.frozen_hash() == before
        alternate = copy.deepcopy(suite)
        suite.calibrate(cx,cy)
        alternate.calibrate(cx,cy*100+3)
        assert suite.frozen_hash() == alternate.frozen_hash() == before
        before_all = fd.state_hash(suite)
        result = suite.predict(query)
        assert before_all == fd.state_hash(suite)
        assert tuple(result["predictions"]) == METHODS
        assert all(value.shape == (17,3) for value in result["predictions"].values())
        for mode in TARGET_MODES:
            kahm,matched = suite.departures[f"{mode}_kahm"],suite.departures[f"{mode}_matched_rbf"]
            assert (kahm.scaler,kahm.cap)==(matched.scaler,matched.cap)
        zero = copy.deepcopy(suite)
        zero.etas = {key: np.zeros(3) for key in SLOPE_METHODS}
        z = zero.predict(query)
        for key in SLOPE_METHODS:
            np.testing.assert_array_equal(z["predictions"][key],z["predictions"]["shared_bias"])
    return {"status":"passed","scope":"Manufactured inputs only, no production outcomes",
            "method_count":len(METHODS),"fold_count":len(suite.fold_states),
            "validation_arrays":len(suite.validation_predictions),
            "basis_arrays":len(suite.basis_arrays),"training_arrays":len(suite.training_arrays),
            "OOF_assignment_exact_once":True,"legacy_C_calibration_isolation":True,
            "C_target_isolation":True,"prediction_no_mutation":True,
            "matched_RBF_geometry_identity":True,"eta_zero_identity":True}


if __name__ == "__main__":
    import json
    print(json.dumps(manufactured_fixture(),indent=2))
