"""Secondary preprocessing mechanism with an honest, shared common forecast.

P and R are disjoint sets of A episodes.  P fits every history scaler and bank,
the raw ridge and common readout; B selects common penalties and caps.  R
targets are never used by the common predictor, including its selection loss
normalizer.  R fits centered departure slopes.  B estimates one shared bias
per common row and selects departure penalties/caps; C selects eta only.

This is an independent-episode analogue of the archived chronological
centered adapter, not a replay of its nine-stage natural-series procedure.
The immutable corrected runtime supplies all numerical primitives.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
import time

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "dimension_forecasting"
if str(SOURCE_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "code"))
import focused_dimension_model as fd


SCALERS = ("standard", "minmax", "tanh_z_1")
PAIRS = {
    "ss": ("standard", "standard"),
    "mm": ("minmax", "minmax"),
    "tt": ("tanh_z_1", "tanh_z_1"),
    "tm": ("tanh_z_1", "minmax"),
    "mt": ("minmax", "tanh_z_1"),
}
KINDS = ("kahm", "rbf")
SLOPE_METHODS = tuple(f"{pair}__{kind}" for pair in PAIRS
                      for kind in ("kahm", "rbf", "matched_rbf"))
METHODS = (tuple(f"common__{arm}" for arm in SCALERS)
           + tuple(f"shared_bias__{arm}" for arm in SCALERS)
           + SLOPE_METHODS)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _guard_geometry(record, stage):
    """Unexpected feature/factor failures stop; inherited guards stay logged."""
    bad = [key for key in ("coordinate_fallback_rows", "gate_fallback_rows",
                          "kernel_fallback_rows") if record.get(key)]
    failed = [r for r in record.get("local_models", []) if "factor_failure" in r]
    if bad or failed:
        raise FloatingPointError(f"Secondary geometry failure at {stage}: {bad}; {failed}")


def _bounded(m, raw, scaler):
    return m.radial_bound(m._standardize(scaler, raw))


def _slope(head, features):
    correction = (features - head["mean"]) @ head["coef"]
    if not np.isfinite(correction).all():
        raise FloatingPointError("Nonfinite centered secondary correction")
    return correction


def _centered_bound(head, eta):
    radius = math.sqrt(head["feature_width"])
    centered_radius = radius + float(np.linalg.norm(head["mean"]))
    coefficient_norm = np.linalg.norm(head["coef"], axis=0)
    coefficient_limit = head["response_rms"] / (2 * np.sqrt(head["alpha_by_horizon"]))
    violations = int(np.count_nonzero(coefficient_norm > coefficient_limit * (1 + 1e-10) + 1e-12))
    if violations:
        raise AssertionError("Secondary ridge coefficient bound violated")
    return {
        "radius_before_centering": radius,
        "centered_radius": centered_radius,
        "coefficient_norm_by_horizon": coefficient_norm.tolist(),
        "coefficient_norm_bound_by_horizon": coefficient_limit.tolist(),
        "coefficient_bound_violations": violations,
        "correction_bound_by_horizon": (np.abs(eta) * centered_radius * coefficient_limit).tolist(),
        "scope": "Centered bounded departure correction relative to shared common plus bias",
    }


def _cap_score(m, prediction, targets, loss_scale):
    by_horizon = m._mse(targets, prediction)
    return float(np.mean(by_horizon / loss_scale)), by_horizon


def _dimension_records(bank_state, bank_diagnostics):
    records = []
    diagnostics = bank_diagnostics["local_models"]
    for i, model in enumerate(bank_state["models"]):
        record = copy.deepcopy(model["dimension_audit"])
        record.update(model_index=i, block_index=int(model["block"]),
                      stored_basis_shape=list(model["p"].shape),
                      stored_reference_shape=list(model["q"].shape),
                      basis_sha256=fd.array_hash(model["p"]),
                      reference_sha256=fd.array_hash(model["q"]),
                      reference_indices=diagnostics[i]["reference_indices"])
        if record["effective_rank"] != model["p"].shape[1]:
            raise AssertionError("Secondary stored basis differs from dimension audit")
        if record["normal_direction_reserved"]:
            raise AssertionError("Secondary native bank reserved an extra normal dimension")
        records.append(record)
    return records


@dataclass
class CommonRow:
    scaler: str
    cap: int
    bank: dict
    feature_scaler: dict
    head: dict
    raw_scaler: dict
    raw_head: dict
    bias: np.ndarray

    def frozen_hash(self):
        return fd.state_hash(self)


@dataclass
class DepartureState:
    common_scaler: str
    departure_scaler: str
    cap: int
    kind: str
    bank: dict
    feature_scaler: dict
    head: dict
    mean_training_correction: np.ndarray


@dataclass
class SecondarySuite:
    common: dict
    departures: dict
    selections: dict
    config: dict
    diagnostics: dict
    etas: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def frozen_state(self):
        """Everything determined before C; excludes C eta/calibration records."""
        return {"common": self.common, "departures": self.departures,
                "selections": self.selections, "config": self.config}

    def frozen_hash(self):
        return fd.state_hash(self.frozen_state())

    def _uncalibrated(self, x):
        m = fd.load_runtime()
        x = m._matrix(x, "secondary query histories")
        if x.shape[1] != len(self.common[SCALERS[0]].raw_scaler["mean"]):
            raise ValueError("Secondary query width differs from fitted state")
        cache, geometry, predictions = {}, {}, {}
        nodes = {}

        def raw_features(arm, cap, bank):
            key = f"{arm}__cap{cap}"
            if key not in cache:
                raw, record = m._raw_feature_maps(bank, x, nodes, key)
                _guard_geometry(record, "query/" + key)
                cache[key] = raw
                geometry[key] = record
            return cache[key]

        raw_row = self.common[SCALERS[0]]
        raw_forecast = m._ridge_predict(raw_row.raw_head, m._standardize(raw_row.raw_scaler, x))
        for arm in SCALERS:
            row = self.common[arm]
            raw = raw_features(arm, row.cap, row.bank)
            features = _bounded(m, raw["common"], row.feature_scaler)
            common = raw_forecast + m._ridge_predict(row.head, features)
            predictions[f"common__{arm}"] = common
            predictions[f"shared_bias__{arm}"] = common + row.bias
        features_summary = {}
        for key in SLOPE_METHODS:
            state = self.departures[key]
            raw = raw_features(state.departure_scaler, state.cap, state.bank)
            features = _bounded(m, raw[state.kind], state.feature_scaler)
            shared = predictions[f"shared_bias__{state.common_scaler}"]
            predictions[key] = shared + _slope(state.head, features)
            features_summary[key] = m._feature_summary(features, math.sqrt(features.shape[1]))
        if set(predictions) != set(METHODS):
            raise AssertionError("Incomplete secondary prediction roster")
        if any(p.shape != raw_forecast.shape or not np.isfinite(p).all() for p in predictions.values()):
            raise FloatingPointError("Invalid secondary forecast")
        return {key: predictions[key] for key in METHODS}, {
            "geometry": geometry, "bounded_features": features_summary, "nodes": nodes,
        }

    def calibrate(self, cx, cy):
        """C selects eta after every P/R/B model and choice has been frozen."""
        if self.etas:
            raise RuntimeError("Secondary suite already calibrated; preserve its frozen C decision")
        m = fd.load_runtime()
        cx, cy = m._matrix(cx, "secondary C histories"), m._matrix(cy, "secondary C targets")
        if len(cx) != len(cy):
            raise ValueError("Secondary C history and target counts differ")
        before = self.frozen_hash()
        raw, diagnostics = self._uncalibrated(cx)
        out = {key: value.copy() for key, value in raw.items()}
        records, etas = {}, {}
        for key in SLOPE_METHODS:
            state = self.departures[key]
            shared = raw[f"shared_bias__{state.common_scaler}"]
            eta, out[key], record = m._select_gamma(cy, shared, raw[key], self.config["gamma_grid"])
            etas[key] = eta
            records[key] = {**record, "selection_split": "C", "selected_eta": eta.tolist()}
        if self.frozen_hash() != before:
            raise AssertionError("Secondary C calibration changed P/R/B state")
        self.etas = etas
        self.calibration = json_safe({
            "selection_split": "C", "C_X_sha256": fd.array_hash(cx), "C_Y_sha256": fd.array_hash(cy),
            "frozen_state_before": before, "frozen_state_after": before,
            "choices": records, "geometry": diagnostics["geometry"],
            "uncalibrated_prediction_sha256": {key: fd.array_hash(value) for key, value in raw.items()},
            "calibrated_prediction_sha256": {key: fd.array_hash(value) for key, value in out.items()},
        })
        return {"records": self.calibration, "predictions": out,
                "eta1_predictions": raw}

    def predict(self, x):
        """Fixed-state query; accepts histories alone and never fits or selects."""
        if set(self.etas) != set(SLOPE_METHODS):
            raise RuntimeError("Calibrate the secondary suite before prediction")
        m = fd.load_runtime()
        before = self.frozen_hash()
        raw, diagnostics = self._uncalibrated(x)
        out = {key: value.copy() for key, value in raw.items()}
        bounds = {}
        for key in SLOPE_METHODS:
            state = self.departures[key]
            shared = raw[f"shared_bias__{state.common_scaler}"]
            out[key] = m._blend(shared, raw[key], self.etas[key])
            bound = _centered_bound(state.head, self.etas[key])
            observed = np.max(np.abs(out[key] - shared), axis=0)
            violations = int(np.count_nonzero(observed > np.asarray(bound["correction_bound_by_horizon"]) * (1 + 1e-10) + 1e-12))
            if violations:
                raise AssertionError("Secondary query correction bound violated")
            bounds[key] = {**bound, "observed_max_abs_correction_by_horizon": observed.tolist(),
                           "query_bound_violations": violations}
        if self.frozen_hash() != before:
            raise AssertionError("Secondary query changed P/R/B state")
        diagnostics.update(bounds=bounds, frozen_state_before=before, frozen_state_after=before,
                           shared_baseline_hashes={arm: row.frozen_hash() for arm, row in self.common.items()},
                           etas={key: eta.tolist() for key, eta in self.etas.items()})
        return {"predictions": out, "eta1_predictions": raw,
                "diagnostics": json_safe(diagnostics)}


def fit_secondary(ax, ay, groups_a, bx, by, *, seed, config=None,
                  cap_order=(1, 2, 4, 8, 12, 20)):
    """Fit the fully specified secondary P/R/B experiment; C is not accepted."""
    start = time.perf_counter()
    m = fd.load_runtime()
    ax, ay, bx, by = fd._fit_arrays(ax, ay, bx, by)
    groups = np.asarray(groups_a)
    if groups.ndim != 1 or len(groups) != len(ax):
        raise ValueError("A episode labels must align with A rows")
    unique = np.unique(groups)
    if len(unique) < 4 or len(unique) % 2:
        raise ValueError("Secondary requires an even number of at least four A episodes")
    cap_order = tuple(int(cap) for cap in cap_order)
    if tuple(sorted(set(cap_order))) != cap_order or not cap_order or cap_order[0] < 1:
        raise ValueError("Declare distinct positive caps in increasing order")
    p_ids, r_ids = np.split(unique, 2)
    p_mask, r_mask = np.isin(groups, p_ids), np.isin(groups, r_ids)
    if np.any(p_mask & r_mask) or not np.all(p_mask | r_mask):
        raise AssertionError("Secondary P/R episode split is invalid")
    px, py, rx, ry = ax[p_mask], ay[p_mask], ax[r_mask], ay[r_mask]
    base = fd.fit_baseline(px, py, bx, by, config=config)
    effective = fd._config(20, "native", config)
    # Common cap selection must not inspect R targets even through normalization.
    loss_scale = base.loss_scale.copy()
    common_candidates, banks, raw_maps, bank_diagnostics = {}, {}, {}, {}
    common_records, selections, fit_nodes = {}, {}, {}
    common_hashes = {}
    for arm in SCALERS:
        records = []
        for cap in cap_order:
            key = (arm, cap)
            conf = fd._config(cap, "native", config)
            bank = m._InstrumentedBank(seed=int(seed), config=conf, geometry_scaling=arm).fit(px)
            failed = [record for record in bank.diagnostics["local_models"] if record.get("factor_error")]
            if failed:
                raise FloatingPointError("Secondary P-bank factor failure: " + repr(failed))
            state = bank.state("kahm")
            for local in state["models"]:
                local.pop("reference_distance", None)
            banks[key] = state
            maps, guards = {}, {}
            for stage, x in (("P", px), ("R", rx), ("B", bx)):
                maps[stage], guards[stage] = m._raw_feature_maps(state, x, {}, stage)
                _guard_geometry(guards[stage], f"{arm}/cap{cap}/{stage}")
            raw_maps[key] = maps
            feature_scaler = m._scaler_fit(maps["P"]["common"], effective["feature_scale_floor"])
            fp = _bounded(m, maps["P"]["common"], feature_scaler)
            fb = _bounded(m, maps["B"]["common"], feature_scaler)
            head, head_record = m._select_ridge(fp, base.residual_a, fb, by, base.predictions_b,
                effective["alpha_grid"], fit_nodes, f"common_{arm}_{cap}")
            pred_b = base.predictions_b + m._ridge_predict(head, fb)
            score, mse = _cap_score(m, pred_b, by, loss_scale)
            common_candidates[key] = {"head": head, "feature_scaler": feature_scaler,
                                      "pred_b": pred_b}
            records.append({"cap": cap, "score": score, "mse_by_horizon": mse.tolist(),
                            "head_selection": head_record, "B_prediction_sha256": fd.array_hash(pred_b)})
            bank_diagnostics[f"{arm}__cap{cap}"] = {
                "geometry_scaler": state.get("geometry_scaler"),
                "dimension_records": _dimension_records(state, bank.diagnostics),
                "bank": bank.diagnostics, "geometry_P_R_B": guards,
            }
        chosen = records[int(np.argmin([record["score"] for record in records]))]["cap"]
        selections[arm] = {"common_cap": chosen, "common_candidates": records,
                            "common_selection_split": "B", "common_gamma": 1.0}
    # A saturated cap has exact parity at d<=12, including the separately fitted bank.
    identity = {}
    if ax.shape[1] <= 12 and 12 in cap_order and 20 in cap_order:
        for arm in SCALERS:
            for stage in ("P", "R", "B"):
                for block in ("common", "kahm", "rbf"):
                    np.testing.assert_array_equal(raw_maps[(arm, 12)][stage][block],
                                                  raw_maps[(arm, 20)][stage][block])
            identity[arm] = "cap12_cap20_exact_raw_feature_identity"

    common_rows, common_predictions, residuals = {}, {}, {}
    for arm in SCALERS:
        cap = selections[arm]["common_cap"]
        key = (arm, cap)
        selected = common_candidates[key]
        pred_b = selected["pred_b"]
        raw_r = m._ridge_predict(base.head, m._standardize(base.input_scaler, rx))
        feature_r = _bounded(m, raw_maps[key]["R"]["common"], selected["feature_scaler"])
        pred_r = raw_r + m._ridge_predict(selected["head"], feature_r)
        bias = np.mean(by - pred_b, axis=0)
        row = CommonRow(arm, cap, banks[key], selected["feature_scaler"], selected["head"],
                        base.input_scaler, base.head, bias)
        common_rows[arm] = row
        common_predictions[arm] = {"R": pred_r, "B": pred_b, "shared_B": pred_b + bias}
        residuals[arm] = ry - pred_r
        common_hashes[arm] = row.frozen_hash()
        common_records[arm] = {
            "common_cap": cap, "bias": bias.tolist(), "bias_fit_split": "B",
            "shared_bias_formula": "mean_B(Y_B - common_B)", "bias_gamma": 1.0,
            "common_frozen_state_sha256": common_hashes[arm],
            "R_common_prediction_sha256": fd.array_hash(pred_r),
            "R_honest_residual_sha256": fd.array_hash(residuals[arm]),
            "R_common_target_access": False,
            "B_shared_prediction_sha256": fd.array_hash(pred_b + bias),
        }
    departures, pair_selections = {}, {}
    feature_cache = {}
    for pair, (common_arm, departure_arm) in PAIRS.items():
        shared_b = common_predictions[common_arm]["shared_B"]
        residual = residuals[common_arm]
        candidate_states, records_by_kind = {}, {kind: [] for kind in KINDS}
        for cap in cap_order:
            for kind in KINDS:
                cache_key = (departure_arm, cap, kind)
                if cache_key not in feature_cache:
                    maps = raw_maps[(departure_arm, cap)]
                    scaler = m._scaler_fit(maps["R"][kind], effective["feature_scale_floor"])
                    feature_cache[cache_key] = (scaler, _bounded(m, maps["R"][kind], scaler),
                                               _bounded(m, maps["B"][kind], scaler))
                scaler, fr, fb = feature_cache[cache_key]
                path = m._ridge_path(fr, residual, effective["alpha_grid"])
                alpha_predictions = [shared_b + _slope(head, fb) for head in path]
                losses = np.array([m._mse(by, prediction) for prediction in alpha_predictions])
                indices = np.argmin(losses, axis=0)
                head = m._combine_heads(path, indices)
                pred_b = shared_b + _slope(head, fb)
                score, mse = _cap_score(m, pred_b, by, loss_scale)
                mean_correction = np.mean(_slope(head, fr), axis=0)
                tolerance = 1e-10 * max(1.0, float(np.max(np.abs(residual))))
                if np.max(np.abs(mean_correction)) > tolerance:
                    raise AssertionError("Secondary centered training correction has nonzero mean")
                _centered_bound(head, np.ones(ay.shape[1]))
                candidate_states[(cap, kind)] = DepartureState(common_arm, departure_arm, cap, kind,
                    banks[(departure_arm, cap)], scaler, head, mean_correction)
                records_by_kind[kind].append({
                    "cap": cap, "score": score, "mse_by_horizon": mse.tolist(),
                    "alpha_grid": list(effective["alpha_grid"]),
                    "B_mse_by_alpha_horizon": losses.tolist(),
                    "alpha_indices_by_horizon": indices.tolist(),
                    "selected_alpha_by_horizon": head["alpha_by_horizon"].tolist(),
                    "B_prediction_sha256": fd.array_hash(pred_b),
                    "R_mean_centered_correction": mean_correction.tolist(),
                    "feature_width": int(head["feature_width"]),
                    "mean_centered_R_feature_max_abs": float(np.max(np.abs(np.mean(fr-head["mean"], axis=0)))),
                })
        choices = {}
        for kind in KINDS:
            chosen = records_by_kind[kind][int(np.argmin([r["score"] for r in records_by_kind[kind]]))]["cap"]
            choices[kind] = chosen
            departures[f"{pair}__{kind}"] = candidate_states[(chosen, kind)]
        departures[f"{pair}__matched_rbf"] = candidate_states[(choices["kahm"], "rbf")]
        pair_selections[pair] = {
            "common_scaler": common_arm, "departure_scaler": departure_arm,
            "common_cap": common_rows[common_arm].cap,
            "selected_departure_caps": choices,
            "matched_rbf_departure_cap": choices["kahm"],
            "candidates": records_by_kind,
            "shared_baseline_sha256": common_hashes[common_arm],
        }
    if common_hashes != {arm: row.frozen_hash() for arm, row in common_rows.items()}:
        raise AssertionError("Departure fitting modified a frozen common row")
    selections = {"common": selections, "departure_pairs": pair_selections,
                  "normalizer": "P target population variance, floored by target_variance_floor",
                  "P_target_variance": loss_scale.tolist(),
                  "tie_rule": "First minimum in declared ascending cap/alpha order"}
    diagnostics = json_safe({
        "schema": "preprocessing-dimension-fixed-common-secondary-v1",
        "seed": int(seed), "cap_order": cap_order, "pairs": PAIRS, "methods": METHODS,
        "P_episode_ids": p_ids, "R_episode_ids": r_ids,
        "P_rows": int(np.sum(p_mask)), "R_rows": int(np.sum(r_mask)), "B_rows": len(bx),
        "input_hashes": {name: fd.array_hash(value) for name, value in
                         (("A_X", ax), ("A_Y", ay), ("groups_A", groups), ("B_X", bx), ("B_Y", by))},
        "P_target_access_only_for_common": True,
        "history_scaler_and_bank_fit_split": "P",
        "common_feature_scaler_fit_split": "P", "departure_feature_scaler_fit_split": "R",
        "departure_training_target": "R_Y minus frozen P/B common forecast evaluated on R histories",
        "shared_bias_fit_split": "B", "shared_bias_has_selected_multiplier": False,
        "departure_centering": "Radial-bound R features, then subtract their R mean; no response intercept",
        "common_rows": common_records, "banks": bank_diagnostics,
        "native12_native20_identity": identity,
        "fit_nodes": fit_nodes, "fit_wall_seconds": time.perf_counter() - start,
        "scope": "Secondary independent-episode mechanism; not chronological natural-series adoption evidence",
    })
    return SecondarySuite(common_rows, departures, selections, effective, diagnostics)


def manufactured_fixture():
    """Focused checks for shared-common, centering, and C/query isolation."""
    from unittest.mock import patch
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(27640319)
    ax = rng.normal(size=(96, 6))
    ay = np.column_stack((ax[:, 0] + .1*rng.normal(size=96),
                          np.sin(ax[:, 1]) + .1*rng.normal(size=96),
                          ax[:, 2]*ax[:, 3] + .1*rng.normal(size=96)))
    bx, by = rng.normal(size=(40, 6)), rng.normal(size=(40, 3))
    cx, cy = rng.normal(size=(32, 6)), rng.normal(size=(32, 3))
    query = rng.normal(size=(17, 6))
    groups = np.repeat(np.arange(12), 8)
    config = {"blocks": [2], "max_references": 12}
    with threadpool_limits(limits=1):
        suite = fit_secondary(ax, ay, groups, bx, by, seed=10974, config=config, cap_order=(1, 2, 20))
        frozen = suite.frozen_hash()
        alternate = copy.deepcopy(suite)
        for row in alternate.common.values():
            assert row.frozen_hash() == suite.common[row.scaler].frozen_hash()
        out_c = suite.calibrate(cx, cy)
        alternate.calibrate(cx, 100*cy + 3)
        assert suite.frozen_hash() == alternate.frozen_hash() == frozen
        assert suite.selections == alternate.selections
        m = fd.load_runtime()
        before = fd.state_hash(suite)
        with patch.object(m._InstrumentedBank, "fit", side_effect=AssertionError("query must not fit")):
            out = suite.predict(query)
        assert fd.state_hash(suite) == before
        assert set(out["predictions"]) == set(METHODS)
        assert all(value.shape == (17, 3) for value in out["predictions"].values())
        for pair, (common_arm, _) in PAIRS.items():
            assert suite.selections["departure_pairs"][pair]["shared_baseline_sha256"] == suite.common[common_arm].frozen_hash()
            for kind in ("kahm", "rbf", "matched_rbf"):
                key = f"{pair}__{kind}"
                assert np.max(np.abs(suite.departures[key].mean_training_correction)) < 1e-10
        # Slope-only predictions must ignore the fitted residual response mean.
        mean_shifted = copy.deepcopy(suite)
        for state in mean_shifted.departures.values():
            state.head["ymean"] = state.head["ymean"] + 123.0
        shifted_out = mean_shifted.predict(query)
        for key in METHODS:
            np.testing.assert_array_equal(out["predictions"][key], shifted_out["predictions"][key])
        # Every eta-zero candidate reduces exactly to its shared-bias row.
        zero = copy.deepcopy(suite)
        zero.etas = {key: np.zeros(3) for key in SLOPE_METHODS}
        zero_out = zero.predict(query)
        for key in SLOPE_METHODS:
            arm = zero.departures[key].common_scaler
            np.testing.assert_array_equal(zero_out["predictions"][key], zero_out["predictions"][f"shared_bias__{arm}"])
        # Changing all R targets cannot change any common row, cap, or shared bias.
        ay_changed = ay.copy()
        ay_changed[groups >= 6] = 50*ay_changed[groups >= 6] + 7
        refit = fit_secondary(ax, ay_changed, groups, bx, by, seed=10974, config=config, cap_order=(1, 2, 20))
        for arm in SCALERS:
            assert suite.common[arm].frozen_hash() == refit.common[arm].frozen_hash()
            assert suite.selections["common"][arm] == refit.selections["common"][arm]
    return {"status": "passed", "scope": "Manufactured inputs only; no production outcomes",
            "method_count": len(METHODS), "fixture_caps": [1, 2, 20],
            "shared_common_and_bias": True, "R_target_common_isolation": True,
            "C_target_fit_isolation": True, "query_no_fit_or_mutation": True,
            "centered_mean_zero": True, "response_intercept_omitted": True,
            "eta_zero_identity": True,
            "C_prediction_count": len(out_c["predictions"])}


if __name__ == "__main__":
    import json
    print(json.dumps(manufactured_fixture(), indent=2))
