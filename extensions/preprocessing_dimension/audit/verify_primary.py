#!/usr/bin/env python3
"""Independent manufactured-input gates for the preprocessing adapter.

No observation-law data, study seed, or terminal scientific outcome is used.
Oracles reconstruct predictions, ranks, transforms, and selection minima from
retained arrays rather than relying on the adapter's diagnostic assertions.
"""
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import preprocessing_model as pm
import fixed_common_secondary as secondary
fd = pm.fd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_equal(a, b):
    np.testing.assert_array_equal(a, b)


def assert_close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)


def radial_oracle(raw, state):
    z = (raw - state["mean"]) / state["scale"]
    width = z.shape[1]
    return np.sqrt(width) * z / np.sqrt(width + np.sum(z*z, axis=1, keepdims=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=ROOT / "readiness/primary_adapter_audit.json")
    parser.add_argument("--secondary-receipt", type=Path, default=ROOT / "readiness/secondary_adapter_audit.json")
    args = parser.parse_args()
    if args.receipt.exists() or args.secondary_receipt.exists():
        raise FileExistsError("Preserve prior verification receipts; choose fresh paths")
    primary_path = ROOT / "code/preprocessing_model.py"
    secondary_path = ROOT / "code/fixed_common_secondary.py"
    initial_hashes = {"primary": sha256(primary_path), "secondary": sha256(secondary_path)}
    m = fd.load_runtime()
    from extensions.geometry_scaling import fit_geometry_scaler, transform_geometry
    rng = np.random.default_rng(83710429)
    ax, bx, cx, query = [rng.normal(size=(n, 12)) for n in (64, 24, 20, 17)]
    def target(x):
        return np.column_stack((x[:, 0] + .2*np.sin(x[:, 1]),
                                x[:, 2]*x[:, 3] + .15*x[:, 1],
                                np.cos(x[:, 4]) + .1*x[:, 0]))
    ay, by, cy = [target(x) + .03*rng.normal(size=(len(x), 3)) for x in (ax, bx, cx)]
    config = {key: copy.deepcopy(value) for key, value in m.DEFAULT_CONFIG.items()
              if key not in ("rank", "dimension_policy")}
    config.update(blocks=[2], max_references=24, bank_row_cap=64,
                  alpha_grid=[1e-4, .01, 1.0], raw_rbf_landmarks=16,
                  raw_rbf_bandwidth_factors=[.5, 1.0],
                  raw_hgb_learning_rates=[.1], raw_hgb_max_leaf_nodes=[7], raw_hgb_max_iter=8)
    counts, notes = {}, {}
    with threadpool_limits(limits=1):
        # Independently observe the training arrays seen by every bank.
        bank_inputs = []
        original_fit = m._InstrumentedBank.fit
        def watched_fit(self, x):
            assert_equal(x, ax)
            bank_inputs.append((self.geometry_scaling, self.config["rank"], fd.array_hash(x)))
            return original_fit(self, x)
        with patch.object(m._InstrumentedBank, "fit", new=watched_fit):
            suite = pm.fit_primary(ax, ay, bx, by, seed=710963, config=config)
        assert len(bank_inputs) == 18
        assert len(suite.candidates) == 30
        counts["bank_training_input_checks"] = len(bank_inputs)
        counts["complete_candidates"] = len(suite.candidates)

        # Actual stored native bases, including full ambient rank >2.
        rank_checks, full_rank = 0, 0
        for key, bank in suite.banks.items():
            cap = bank.diagnostics["cap"]
            for local in bank.bank["models"]:
                audit = local["dimension_audit"]
                assert audit["initial_admissible_rank"] == min(cap, 12, len(local["q"])-1)
                assert audit["effective_rank"] == local["p"].shape[1] == local["rank"]
                assert local["rank"] <= audit["initial_admissible_rank"]
                assert audit["policy"] == "native" and not audit["normal_direction_reserved"]
                rank_checks += 1
                full_rank += int(local["rank"] == 12)
        assert full_rank > 0
        counts.update(native_basis_checks=rank_checks, full_ambient_rank_models=full_rank)

        # Training-only geometry scalers reconstructed without adapter helpers.
        expected_mean, expected_std = ax.mean(axis=0), ax.std(axis=0, ddof=0)
        expected_minimum, expected_maximum = ax.min(axis=0), ax.max(axis=0)
        scaler_checks = 0
        for arm in pm.SCALERS:
            for cap in pm.CAPS:
                state = suite.banks[pm.bank_id(arm, cap)].bank
                if arm == "standard":
                    assert_equal(state["mean"], expected_mean)
                    assert_equal(state["scale"], np.where(expected_std > 1e-12, expected_std, 1.0))
                else:
                    fitted = state["geometry_scaler"]
                    assert fitted["fit_rows"] == len(ax)
                    if arm == "minmax":
                        assert_equal(fitted["minimum"], expected_minimum)
                        assert_equal(fitted["maximum"], expected_maximum)
                        expected = 2*(query-expected_minimum)/(expected_maximum-expected_minimum)-1
                        assert fitted["clip"] is False
                    else:
                        assert_equal(fitted["mean"], expected_mean)
                        assert_equal(fitted["scale"], np.where(expected_std > 1e-12, expected_std, 1.0))
                        assert fitted["ddof"] == 0 and fitted["multiplier"] == 1
                        assert fitted["post_tanh_rescale"] is False
                        expected = np.tanh((query-expected_mean)/np.where(expected_std > 1e-12, expected_std, 1.0))
                    assert_close(transform_geometry(query, fitted), expected)
                scaler_checks += 1
        # A constant coordinate plus a later query that differs from it.
        scale_train = np.array([[1., 7., -3.], [2., 7., 0.], [4., 7., 3.], [5., 7., 5.]])
        scale_query = np.array([[20., 15., -30.], [-9., -8., 20.]])
        mm = fit_geometry_scaler(scale_train, "minmax")
        mm_expected = np.column_stack((2*(scale_query[:,0]-1)/4-1, np.zeros(2),
                                       2*(scale_query[:,2]+3)/8-1))
        assert_equal(transform_geometry(scale_query, mm), mm_expected)
        assert np.max(np.abs(mm_expected)) > 1
        th = fit_geometry_scaler(scale_train, "tanh_z_1")
        sd = scale_train.std(axis=0, ddof=0)
        assert_equal(transform_geometry(scale_query, th),
                     np.tanh((scale_query-scale_train.mean(axis=0))/np.where(sd > 1e-12, sd, 1.0)))
        assert np.all(transform_geometry(scale_query, th)[:,1] != 0)
        counts["scaler_checks"] = scaler_checks + 2

        # Independently reconstruct all B scores and context-specific choices.
        loss_scale = np.maximum(np.var(ay, axis=0, ddof=0), config["target_variance_floor"])
        oracle_scores = {}
        for key, candidate in suite.candidates.items():
            oracle_scores[key] = {}
            for method, prediction in candidate.b_predictions.items():
                score = float(np.mean(np.mean((by-prediction)**2, axis=0)/loss_scale))
                assert score == suite.b_scores[key][method]
                oracle_scores[key][method] = score
        ordered_pairs = ("standard", "minmax", "tanh", "hybrid_tanh_minmax", "hybrid_minmax_tanh")
        ordered_caps = (1, 2, 4, 8, 12, 20)
        candidates = [f"{pair}__cap{cap}" for cap in ordered_caps for pair in ordered_pairs]
        choice_checks = 0
        for context, choices in suite.selections.items():
            for method in pm.GEOMETRY:
                eligible = list(candidates)
                if context in ("cap2", "cap20"):
                    eligible = [key for key in eligible if key.endswith("__"+context)]
                elif context in ordered_pairs:
                    eligible = [key for key in eligible if key.split("__cap")[0] == context]
                if method == "common" and context in ("selected", "cap2", "cap20"):
                    eligible = [key for key in eligible if key.split("__cap")[0] in ("standard", "minmax", "tanh")]
                expected = min(eligible, key=lambda key: oracle_scores[key][method])
                assert choices[method] == expected
                choice_checks += 1
        counts["independent_B_selection_checks"] = choice_checks

        # Independently rebuild hybrid vectors and predictions on new histories.
        raw_query, shared_query, _ = suite.predict_uncalibrated(query)
        hybrid_checks = 0
        for pair in ("hybrid_tanh_minmax", "hybrid_minmax_tanh"):
            for cap in ordered_caps:
                key = pm.candidate_id(pair, cap)
                candidate = suite.candidates[key]
                cb, db = suite.banks[candidate.common_bank], suite.banks[candidate.departure_bank]
                rawc = m._raw_feature_maps(cb.bank, query, {}, "oracle_common")[0]
                rawd = m._raw_feature_maps(db.bank, query, {}, "oracle_departure")[0]
                common_features = radial_oracle(rawc["common"], cb.scalers["common"])
                for method in pm.GEOMETRY:
                    if method == "common":
                        features = common_features
                    else:
                        kind = method.split("_")[0]
                        departure_features = ((rawd[kind]-db.scalers[kind]["mean"])/db.scalers[kind]["scale"]
                            if method.endswith("unbounded") else radial_oracle(rawd[kind], db.scalers[kind]))
                        features = np.column_stack((common_features, departure_features))
                    head = candidate.heads[method]
                    expected = shared_query["ridge"] + head["ymean"] + (features-head["mean"])@head["coef"]
                    assert_close(raw_query[key][method], expected)
                    hybrid_checks += 1
                diag_pair = "tanh" if pair == "hybrid_tanh_minmax" else "minmax"
                assert_equal(raw_query[key]["common"], raw_query[pm.candidate_id(diag_pair, cap)]["common"])
        counts["hybrid_prediction_oracles"] = hybrid_checks

        # The full d=12 cap12/cap20 identity holds in state and predictions.
        identity_checks = 0
        for pair in ordered_pairs:
            for method in pm.GEOMETRY:
                assert_equal(raw_query[pm.candidate_id(pair,12)][method], raw_query[pm.candidate_id(pair,20)][method])
                identity_checks += 1
        counts["cap12_cap20_query_identities"] = identity_checks

        # C cannot modify the selected A/B procedure; changing C may only change eta/policy.
        development_before = fd.state_hash((suite.shared, suite.banks, suite.candidates, suite.selections,
                                           suite.raw_rbf_states, suite.raw_rbf_selected_scaler))
        alternate = copy.deepcopy(suite)
        suite.calibrate(cx, cy)
        alternate.calibrate(cx, cy*23+5)
        for fitted in (suite, alternate):
            assert fd.state_hash((fitted.shared, fitted.banks, fitted.candidates, fitted.selections,
                                  fitted.raw_rbf_states, fitted.raw_rbf_selected_scaler)) == development_before
        assert suite.selections == alternate.selections
        notes["C_isolation"] = "Different C targets leave all A/B states and choices byte-identical"

        # Standard diagonal predictions agree exactly with the published adapter.
        predictions, _ = suite.predict(query)
        parity_checks = 0
        for cap in (2,20):
            old = fd.fit_candidate(ax,ay,bx,by,candidate_id=f"oracle{cap}",rank_cap=cap,
                                   dimension_policy="native",seed=710963,config=config,
                                   baseline_fit=suite.shared.baseline_fit)
            old_cal,_,_ = fd.calibrate(old,cx,cy)
            old_out = old_cal.predict(query)
            for method in pm.GEOMETRY:
                assert_equal(predictions[f"candidate__standard__cap{cap}__calibrated__{method}"],old_out["predictions"][method])
                assert_equal(predictions[f"candidate__standard__cap{cap}__gamma1__{method}"],old_out["uncalibrated_predictions"][method])
                parity_checks += 2
        counts["published_standard_parity_arrays"] = parity_checks

        before = fd.state_hash(suite)
        with patch.object(m._InstrumentedBank,"fit",side_effect=AssertionError("query must not fit banks")), \
             patch.object(m,"_ridge_path",side_effect=AssertionError("query must not fit heads")):
            nofit_out,_ = suite.predict(query)
        assert fd.state_hash(suite) == before
        for key in predictions:
            assert_equal(nofit_out[key],predictions[key])
        notes["query_isolation"] = "Prediction fits neither banks nor heads and preserves serialized state"
        assert tuple(inspect.signature(pm.PrimarySuite.predict).parameters) == ("self","x")

        # Eta/gamma zero is an exact recovery of the shared raw ridge.
        zero = copy.deepcopy(suite)
        zero.gammas = {key:{method:np.zeros(3) for method in pm.GEOMETRY} for key in zero.candidates}
        zero.shared_gammas = {method:np.zeros(3) for method in fd.SHARED_METHODS[1:]}
        zero_out,_ = zero.predict(query)
        zero_checks = 0
        for key,prediction in zero_out.items():
            if "__calibrated__" in key:
                assert_equal(prediction,shared_query["ridge"])
                zero_checks += 1
        counts["gamma_zero_identities"] = zero_checks

        # Real exact ties: constant responses make every candidate score zero.
        tied = pm.fit_primary(ax,np.zeros_like(ay),bx,np.zeros_like(by),seed=710963,config=config)
        for context, choices in tied.selections.items():
            expected_pair = context if context in ordered_pairs else "standard"
            expected_cap = int(context[3:]) if context in ("cap2","cap20") else 1
            for method in pm.GEOMETRY:
                assert choices[method] == f"{expected_pair}__cap{expected_cap}"
        assert tied.raw_rbf_selected_scaler == "standard"
        tied.calibrate(cx,np.zeros_like(cy))
        assert all(record["selected_method"] == "ridge" for record in tied.policies.values())
        notes["exact_tie_order"] = "Cap ascending, then declared pair order; ridge first in policy ties"

        # A finite fallback must not hide an unexpected invalid-coordinate event.
        bank = suite.banks[pm.bank_id("standard",2)].bank
        raw,diag = m._raw_feature_maps(bank,query,{},"injected_guard")
        diag = copy.deepcopy(diag)
        diag["coordinate_fallback_rows"] = [{"model":0,"rows":[0]}]
        with patch.object(m,"_raw_feature_maps",return_value=(raw,diag)):
            try:
                pm.checked_raw(bank,query,{},"injected_guard")
            except FloatingPointError:
                pass
            else:
                raise AssertionError("Unexpected geometry guard was silently accepted")
        notes["unexpected_guard"] = "Injected finite coordinate fallback stops the adapter"

        secondary_result = secondary.manufactured_fixture()
    if initial_hashes != {"primary":sha256(primary_path),"secondary":sha256(secondary_path)}:
        raise RuntimeError("Implementation changed while verification was running; rerun on final source")
    result = {"status":"passed", "completed_utc":datetime.now(timezone.utc).isoformat(),
              "scope":"Manufactured inputs only; no scientific production data or outcomes",
              "implementation_sha256":initial_hashes, "audit_sha256":sha256(Path(__file__)),
              "fixture_dimensions":{"A":[64,12],"B":[24,12],"C":[20,12],"query":[17,12]},
              "fixture_model_overrides":config,"checks":counts,"notes":notes,
              "runtime_provenance":fd.runtime_provenance()}
    secondary_result.update(completed_utc=result["completed_utc"],
                            implementation_sha256=initial_hashes["secondary"],
                            audit_sha256=result["audit_sha256"])
    for path,payload in ((args.receipt,result),(args.secondary_receipt,secondary_result)):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x") as handle:
            json.dump(payload,handle,indent=2,sort_keys=True,allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status":"passed","checks":counts,
                      "primary_receipt":str(args.receipt),"secondary_receipt":str(args.secondary_receipt)},indent=2))


if __name__ == "__main__":
    main()
