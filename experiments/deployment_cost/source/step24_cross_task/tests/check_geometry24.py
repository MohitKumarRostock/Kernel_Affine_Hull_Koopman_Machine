"""Manufactured-only geometry and representation preflight for Step 24.

The only fitted geometries use synthetic Gaussian reference features with a
fixed constant axis. No HMM benchmark generator or Appliances file is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import geometry24 as M
import models24 as Models
import numpy as np


def rejects(fn):
    try:
        fn()
    except (ValueError, TypeError, ArithmeticError, RuntimeError):
        return True
    raise AssertionError("Manufactured invalid input was accepted")


def plain_payload(value):
    if isinstance(value, np.ndarray):
        assert value.dtype.kind in "fbiu" and value.dtype != object
    elif isinstance(value, dict):
        assert all(isinstance(k, str) for k in value)
        for child in value.values():plain_payload(child)
    elif isinstance(value, (tuple, list)):
        for child in value:plain_payload(child)
    elif value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return
    else:
        raise AssertionError("Hidden nonprimitive payload: " + str(type(value)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def high_precision_row(scores, omega):
    """Independent 110-digit canonical arithmetic on exact binary64 inputs."""
    with localcontext() as ctx:
        ctx.prec = 110
        tau = Decimal.from_float(M.TAU)
        w = Decimal.from_float(float(omega))
        c = [min(Decimal(1), max(Decimal(0), Decimal.from_float(float(s)))) for s in scores]
        u = [Decimal(1) - v + tau for v in c]
        v = [a.ln() for a in u]
        powered = [ctx.power(a, w) for a in u]
        total = sum(powered)
        p = [a / total for a in powered]
        H = M.walsh_contrasts(len(scores))
        r = [sum(Decimal(int(sign)) * item for sign, item in zip(row, v)) / len(scores) for row in H]
        mass = total.ln()
        lw = [w * item for item in v]
        return {"P": p, "LR0": r + [Decimal(0)], "LRM": r + [mass], "LW": lw, "CS": c}


def main():
    out = ROOT / "tests/geometry_preflight24_v2.json"
    if out.exists():raise FileExistsError("Manufactured geometry receipts are write-once")
    started = time.perf_counter()
    G = M.initialize_runtime()
    checks = []
    rng = np.random.Generator(np.random.PCG64(240320241))
    max_high_precision = {view: 0.0 for view in ("P", "CS", "LR0", "LRM", "LW")}
    hp_rows = 0
    pickle_inventory_checks = 0
    underflow_cases = []
    for width in (2, 4, 8, 16):
        H = M.walsh_contrasts(width).astype(np.int64)
        assert np.array_equal(H @ np.ones(width, dtype=np.int64), np.zeros(width - 1, dtype=np.int64))
        assert np.array_equal(H @ H.T, width * np.eye(width - 1, dtype=np.int64))
        assert np.array_equal(H.T @ H, width * np.eye(width, dtype=np.int64) - np.ones((width, width), dtype=np.int64))
        scores = np.vstack((np.zeros(width), np.full(width, .5), np.ones(width),
                            np.linspace(0, 1, width), np.linspace(1, 0, width),
                            rng.uniform(0, 1, width), np.linspace(-.25, 1.25, width)))
        for omega in (.5, 4., 64.):
            views = {}
            for view in ("P", "CS", "LR0", "LRM", "LW"):
                payload = M.view_payload(view, (width,), omega)
                plain_payload(payload)
                restored = pickle.loads(pickle.dumps(payload, protocol=5))
                assert (Models.numeric_inventory(payload)["numeric_bytes"]
                        == Models.numeric_inventory(restored)["numeric_bytes"])
                pickle_inventory_checks += 1
                if view in ("LR0", "LRM"):
                    assert payload["walsh"][0].flags.owndata
                    assert payload["walsh"][0].nbytes == width * (width - 1)
                F = M.transform_scores(scores, payload)
                assert np.array_equal(F, M.transform_scores(scores, restored))
                assert F.shape == scores.shape and F.dtype == np.float64 and np.isfinite(F).all()
                M.representation_diagnostics(scores, payload, F)
                views[view] = F
                if view == "CS":assert "tau" not in payload and "omega" not in payload
                if view == "LR0":assert "omega" not in payload
            assert np.array_equal(views["P"], G.associations(scores, (width,), omega))
            assert np.array_equal(views["CS"], np.clip(scores, 0, 1))
            assert np.array_equal(views["LR0"][:, :-1], views["LRM"][:, :-1])
            assert np.array_equal(views["LR0"][:3], np.zeros((3, width)))
            paired = M.paired_coordinate_diagnostics(scores, (width,), omega)
            assert paired["shared_LR0_LRM_contrasts_bit_identical"]
            assert paired["max_P_from_LR0_reconstruction_error"] < 2e-12
            assert paired["max_CS_from_LRM_reconstruction_error"] < 2e-12
            assert paired["max_CS_from_LW_reconstruction_error"] < 2e-12
            for row in (3, 5, 6):
                exact = high_precision_row(scores[row], omega)
                hp_rows += 1
                for view, truth in exact.items():
                    deviation = max(abs(float(a) - float(b)) for a, b in zip(views[view][row], truth))
                    max_high_precision[view] = max(max_high_precision[view], deviation)
                    assert deviation < 2e-12, (width, omega, view, deviation)
            if omega == 64.:
                exact = high_precision_row(scores[3], omega)
                assert min(exact["P"]) > 0
                assert np.count_nonzero(views["P"][3] == 0) >= 1
                underflow_cases.append({"width": width, "rounded_zeros": int(np.count_nonzero(views["P"][3] == 0)),
                                        "canonical_smallest_probability_decimal": str(min(exact["P"]))})
    checks += ["Walsh identities for all four widths including J=16",
               "All five views preserve declared shape and finite operational values",
               "Operational P is bit-identical to the inherited stable associations implementation",
               "Shared LR0/LRM contrasts are bit-identical; uniform blocks give exact zero contrasts",
               "110-digit canonical checks bound manufactured floating discrepancies, without a global enclosure claim",
               "omega64 deliberately preserves rounded zero probabilities while canonical probabilities remain positive",
               "Unused tau/omega parameters are omitted from CS/LR0 payloads"]

    # Multiple blocks, arbitrary leading batch axes, and frozen sign-sum order.
    multiple = rng.uniform(-.1, 1.1, (2, 3, 30))
    for view in ("P", "CS", "LR0", "LRM", "LW"):
        payload = M.view_payload(view, (2, 4, 8, 16), 4.)
        shaped = M.transform_scores(multiple, payload)
        flattened = M.transform_scores(multiple.reshape(6, 30), payload).reshape(2, 3, 30)
        assert np.array_equal(shaped, flattened)
    checks.append("Multi-block concatenation and leading batch axes preserve exact coordinate order")

    # Exactly one inherited family fit, captured to compare the real API export.
    U = rng.normal(0.0, 1.0, (512, 32))
    U[:, -1] = 7.25
    X = rng.normal(0.2, 1.4, (1031, 32))
    X[:, -1] = rng.normal(200, 50, len(X))
    captured = []
    original_fit = G.fit_family
    def capture(*args, **kwargs):
        family = original_fit(*args, **kwargs)
        captured.append(family)
        return family
    G.fit_family = capture
    try:
        banks, evidence = M.fit_banks(U, 240320242)
    finally:
        G.fit_family = original_fit
    assert len(captured) == 1
    family = captured[0]
    assert len(banks) == 6 and set(banks) == {M.bank_key(s, c) for s in M.SCALINGS for c in M.CAPS}
    assert evidence["unused_regrouped_bank_count"] == 6
    assert evidence["unused_regrouped_local_autoencoder_fits"] == 180
    assert evidence["learned_local_autoencoder_fits"] == 180
    assert evidence["native_total_local_autoencoder_fits"] == 360
    assert evidence["unused_regrouped_controls_evaluated"] is False
    assert family["metadata"]["fit_indices"] == list(range(512))
    max_native_export_error = 0.0
    max_shape_drift = 0.0
    rank_checks = 0
    reference_checks = 0
    for scaling in M.SCALINGS:
        reference_bank = family["learned"][(scaling, 20)]
        for cap in M.CAPS:
            native = family["learned"][(scaling, cap)]
            bank = banks[M.bank_key(scaling, cap)]
            plain_payload(bank)
            assert "audit" not in bank and "fit_indices" not in bank
            expected = G.score(native, X)
            actual = M.score_bank(bank, X)
            deviation = float(np.max(np.abs(expected - actual)))
            max_native_export_error = max(max_native_export_error, deviation)
            assert np.array_equal(actual, expected), (scaling, cap, deviation)
            # Timing uses512-row batches, so retain evidence about batch shape.
            smaller = M.score_bank(bank, X[:512])
            max_shape_drift = max(max_shape_drift, float(np.max(np.abs(smaller - actual[:512]))))
            for prefix in (2, 6, 14, 30):
                compact = M.compact_bank(bank, prefix)
                plain_payload(compact)
                deployment = {"geometry": compact,
                              "transform": M.view_payload("LRM", M.blocks_for_prefix(prefix), 4.)}
                restored = pickle.loads(pickle.dumps(deployment, protocol=5))
                assert (Models.numeric_inventory(deployment)["numeric_bytes"]
                        == Models.numeric_inventory(restored)["numeric_bytes"])
                pickle_inventory_checks += 1
                assert np.array_equal(M.score_compact(restored["geometry"], X[:37]),
                                      M.score_compact(compact, X[:37]))
                assert tuple(compact["block_sizes"]) == M.blocks_for_prefix(prefix)
                assert np.array_equal(M.score_compact(compact, X[:37]),
                                      G.score(G.compact(native, len(M.blocks_for_prefix(prefix))), X[:37]))
                S = actual[:37, :prefix]
                assert np.array_equal(M.transform_scores(S, M.view_payload("P", M.blocks_for_prefix(prefix), 4.)),
                                      G.associations(S, M.blocks_for_prefix(prefix), 4.))
            for bi, block in enumerate(native.blocks):
                for si, group in enumerate(block["classifier"]):
                    ae = group[0]
                    audit = ae["dimension_audit"]
                    assert np.array_equal(ae["y_data"], reference_bank.blocks[bi]["classifier"][si][0]["y_data"])
                    reference_checks += 1
                    assert audit["requested_cap"] == cap
                    assert audit["initial_admissible_rank"] == min(cap, 32, ae["y_data"].shape[1] - 1)
                    assert 1 <= ae["PC"].shape[1] <= audit["initial_admissible_rank"]
                    assert ae["PC"].shape[1] == audit["effective_rank"]
                    assert audit["fallback_applied"] is False
                    assert ae["y_data"].shape[1] <= 24
                    rank_checks += 1
            scaled = M.scale_input(bank["scaler"], X)
            assert np.array_equal(scaled[:, -1], np.zeros(len(X)))
            assert np.array_equal(scaled, native.scaler.transform(X))
            if scaling == "minmax01":assert np.any((scaled[:, :-1] < 0) | (scaled[:, :-1] > 1))
    checks += ["Six frozen learned banks fit only512 manufactured reference rows; all unused regrouping is counted",
               "All six compact encoders exactly match native scores for identical1031-row layouts",
               "Every 2/6/14/30 prefix exactly matches the inherited compact native path",
               "Local caps/ranks/refcounts are recorded and reference arrays match across caps",
               "Support-constant axes map to zero and held-out MinMax inputs remain unclipped",
               "Exported banks contain only primitive/NumPy payloads, with full retained query arrays and centers"]
    checks.append("Every view and complete compact encoder-plus-LRM payload keeps the same numeric storage and forecasts after pickle; Walsh rows own exactly their retained data")

    invalid = [
        lambda: M.fit_banks(np.zeros((511, 32)), 0),
        lambda: M.fit_banks(np.zeros((512, 32)), True),
        lambda: M.blocks_for_prefix(8),
        lambda: M.walsh_contrasts(3),
        lambda: M.view_payload("BAD", (2,), 4.),
        lambda: M.view_payload("P", (2,), None),
        lambda: M.view_payload("P", (2,), 0.),
        lambda: M.view_payload("P", (2,), 129.),
        lambda: M.view_payload("P", (2,), 4., 0.),
        lambda: M.view_payload("P", (2,), 4., float("nan")),
        lambda: M.transform_scores(np.array([[float("nan"), 0.]]), M.view_payload("P", (2,), 4.)),
        lambda: M.transform_scores(np.zeros((2, 3)), M.view_payload("P", (2,), 4.)),
        lambda: M.score_bank(banks["minmax01_2"], np.zeros((1, 31))),
        lambda: M.representation_diagnostics(np.zeros((2, 2)), M.view_payload("P", (2,), 4.), np.zeros((2, 2))),
    ]
    for fn in invalid:assert rejects(fn)
    checks.append("Nonfinite, malformed, unauthorized-roster and cached-feature mismatch inputs are rejected")

    receipt = {"schema": "step24-geometry-manufactured-preflight-v2",
        "passed": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "high_precision_rows_checked": hp_rows,
        "maximum_110digit_reference_errors": max_high_precision,
        "intentional_probability_underflow_cases": underflow_cases,
        "manufactured_reference_shape": [512, 32], "manufactured_query_shape": [1031, 32],
        "manufactured_family_fits": 1, "manufactured_local_autoencoder_fits": 360,
        "manufactured_unused_regrouped_local_fits": 180,
        "native_export_max_abs_error_identical_layout": max_native_export_error,
        "native_batch_shape_drift_max_abs": max_shape_drift,
        "local_rank_checks": rank_checks, "same_reference_checks": reference_checks,
        "pickle_inventory_checks": pickle_inventory_checks,
        "invalid_requests_rejected": len(invalid),
        "scientific_benchmark_trajectories_generated": 0, "scientific_benchmark_model_fits": 0,
        "real_dataset_rows_parsed": 0, "ECG_or_reserve_access": False,
        "wall_seconds": time.perf_counter() - started,
        "source_sha256": {**M.source_provenance(), "tests/check_geometry24.py": sha(Path(__file__).resolve()),
                          "src/models24.py": sha(ROOT / "src/models24.py")},
        "scope": "Manufactured implementation/rounding/export checks only. No canonical numerical enclosure or forecasting result."}
    with out.open("x") as f:json.dump(receipt, f, indent=2, allow_nan=False);f.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":main()
