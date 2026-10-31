"""Algebra and numerical witnesses for the reconstruction-feature study.

Uses fabricated reference banks and target vectors only. No forecasting
evaluation data, outcomes, or model selection are accessed. Numerical
witnesses supplement the analytic statements in ALGEBRA.md; sample ranks
are not treated as proofs of function-space equality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extensions.direct_reconstruction.model import SharedBank, _guarded_reconstruction


def bounded_radial(values: np.ndarray, radius: float) -> np.ndarray:
    """Evaluate v/sqrt(1+||v||^2/B^2) without squaring huge finite v."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or min(v.shape) == 0 or not np.isfinite(v).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("A finite matrix and positive finite radius are required")
    maximum = np.max(np.abs(v), axis=1)
    small = maximum <= radius
    answer = np.empty_like(v)
    scaled = v[small] / radius
    answer[small] = v[small] / np.sqrt(1.0 + np.sum(scaled * scaled, axis=1))[:, None]
    if np.any(~small):
        maximum_large = maximum[~small]
        scaled = v[~small] / maximum_large[:, None]
        divisor = np.sqrt((radius / maximum_large) ** 2 + np.sum(scaled * scaled, axis=1))
        answer[~small] = scaled * (radius / divisor)[:, None]
    if not np.isfinite(answer).all():
        raise FloatingPointError("The bounded transform produced nonfinite values")
    return answer


def normalized_block(values: np.ndarray, fit_values: np.ndarray) -> np.ndarray:
    center = fit_values.mean(axis=0)
    scale = np.maximum(fit_values.std(axis=0), 1e-6)
    return bounded_radial((values - center) / scale, np.sqrt(values.shape[1]))


def numeric_rank(matrix: np.ndarray) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    return int(np.count_nonzero(singular > tolerance))


def residual_to_span(target: np.ndarray, basis: np.ndarray) -> float:
    coefficient, *_ = np.linalg.lstsq(basis, target, rcond=None)
    return float(np.max(np.abs(target - basis @ coefficient)))


def make_fixture(references: list[float]) -> dict:
    q = np.asarray(references, dtype=float)[:, None]
    model = {"q": q, "p": np.ones((1, 1)), "rank": 1, "center": q.mean(axis=0)}
    factor_record = {}
    # Use the actual inherited covariance and reference-energy iteration.
    SharedBank._fit_kahm_factor(model, factor_record)
    precision = float(model["whitening"][0, 0] ** 2)
    kernel = np.exp(-0.5 * precision * (q - q.T) ** 2)
    c = model["inverse"]
    c_ones = c @ np.ones(len(q))
    c_q = c @ q
    off_diagonal_row_sum = float(np.max(np.sum(kernel - np.eye(len(q)), axis=1)))
    return {
        "q": q, "model": model, "factor": factor_record,
        "kernel": kernel, "precision": precision,
        "c": c, "c_ones": c_ones, "c_q": c_q,
        "neumann_condition": bool(model["lambda_original"] + 1 > 2 * off_diagonal_row_sum),
        "off_diagonal_row_sum": off_diagonal_row_sum,
    }


def reconstructions(fixture: dict, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    q, c = fixture["q"], fixture["c"]
    log_kernel = -0.5 * fixture["precision"] * (x - q.T) ** 2
    # The common positive row factor cancels in both normalized maps.
    k = np.exp(log_kernel - log_kernel.max(axis=1, keepdims=True))
    w = k @ c
    alpha = w / w.sum(axis=1, keepdims=True)
    kahm = alpha @ q
    rbf = (k @ q) / k.sum(axis=1, keepdims=True)
    inherited, guard = _guarded_reconstruction(k, c, q, 1e-10)
    if not np.allclose(inherited, kahm, rtol=2e-14, atol=2e-14):
        raise AssertionError("Algebraic and inherited reconstruction differ")
    return {"x": x, "k": k, "alpha": alpha, "kahm": kahm, "rbf": rbf,
            "affine": x.copy(), "d_kahm": kahm - x, "d_rbf": rbf - x,
            "guard": guard}


def fixture_summary(fixture: dict, x: np.ndarray) -> dict:
    value = reconstructions(fixture, x)
    common = np.column_stack([np.ones(len(x)), value["x"]])
    k_augmented = np.column_stack([common, value["d_kahm"]])
    r_augmented = np.column_stack([common, value["d_rbf"]])
    both = np.column_stack([common, value["d_kahm"], value["d_rbf"]])

    # Fabricated y is an algebraic coordinate vector, not a future series.
    y = np.linspace(-0.4, 1.1, len(fixture["q"])) ** 2 - 0.2
    direct = value["alpha"] @ y
    numerator = value["k"] @ np.linalg.solve(
        fixture["kernel"] + fixture["model"]["lambda_original"] * np.eye(len(y)), y)
    denominator = value["k"] @ np.linalg.solve(
        fixture["kernel"] + fixture["model"]["lambda_original"] * np.eye(len(y)), np.ones(len(y)))
    normalized_krr = numerator / denominator
    affine_y = 1.7 * fixture["q"][:, 0] - 0.3
    affine_identity_left = value["alpha"] @ affine_y - (1.7 * value["x"][:, 0] - 0.3)
    affine_identity_right = 1.7 * value["d_kahm"][:, 0]

    bounded_common = normalized_block(common, common)
    bounded_k = normalized_block(value["d_kahm"], value["d_kahm"])
    bounded_r = normalized_block(value["d_rbf"], value["d_rbf"])
    bounded_r_basis = np.column_stack([np.ones(len(x)), bounded_common, bounded_r])
    bounded_both = np.column_stack([np.ones(len(x)), bounded_common, bounded_r, bounded_k])

    triple = reconstructions(fixture, np.array([0.0, 0.5, 1.0]))
    curvature = float(triple["d_kahm"][0, 0] - 2 * triple["d_kahm"][1, 0] + triple["d_kahm"][2, 0])
    return {
        "references": fixture["q"][:, 0].tolist(),
        "precision": fixture["precision"],
        "source_regularizer": fixture["model"]["lambda_original"],
        "source_regularizer_iterations": fixture["factor"]["iterations"],
        "source_regularizer_converged": fixture["factor"]["converged"],
        "row_sums_of_reference_kernel": fixture["kernel"].sum(axis=1).tolist(),
        "inverse_times_one": fixture["c_ones"].tolist(),
        "neumann_positivity_condition_passed": fixture["neumann_condition"],
        "largest_offdiagonal_row_sum": fixture["off_diagonal_row_sum"],
        "minimum_relative_denominator_on_grid": value["guard"]["minimum_relative_denominator"],
        "maximum_normalized_weight_sum_error": float(np.max(np.abs(value["alpha"].sum(axis=1) - 1))),
        "normalized_krr_identity_max_abs_error": float(np.max(np.abs(direct - normalized_krr))),
        "affine_target_identity_max_abs_error": float(np.max(np.abs(affine_identity_left - affine_identity_right))),
        "departure_second_difference_at_0_half_1": curvature,
        "sample_rank_common": numeric_rank(common),
        "sample_rank_common_plus_kahm": numeric_rank(k_augmented),
        "sample_rank_common_plus_rbf": numeric_rank(r_augmented),
        "sample_rank_common_plus_both": numeric_rank(both),
        "raw_kahm_departure_residual_to_common_span": residual_to_span(value["d_kahm"], common),
        "raw_kahm_departure_residual_to_common_plus_rbf_span": residual_to_span(value["d_kahm"], r_augmented),
        "bounded_kahm_departure_residual_to_common_plus_rbf_span": residual_to_span(bounded_k, bounded_r_basis),
        "bounded_combined_sample_rank": numeric_rank(bounded_both),
        "sample_rank_scope": "Numerical witness on the declared fabricated grid; not a proof about every fitted study bank.",
    }


def stability_witness() -> dict:
    rng = np.random.default_rng(104729)
    raw_fit = rng.normal(size=(31, 7))
    raw_query = np.vstack([rng.normal(size=(113, 7)), np.full((1, 7), 1e300),
                           np.array([[1e300, -1e300, 0, 1e300, -1e300, 0, 1e300]])])
    train_blocks, query_blocks = [], []
    for columns in (slice(0, 3), slice(3, 7)):
        train_blocks.append(normalized_block(raw_fit[:, columns], raw_fit[:, columns]))
        query_blocks.append(normalized_block(raw_query[:, columns], raw_fit[:, columns]))
    phi = np.column_stack(train_blocks)
    query = np.column_stack(query_blocks)
    residual = rng.normal(size=len(phi))
    center, mean_target = phi.mean(axis=0), residual.mean()
    phi_c, target_c = phi - center, residual - mean_target
    alpha = 0.01
    penalty = len(phi) * alpha
    coefficient = np.linalg.solve(phi_c.T @ phi_c + penalty * np.eye(phi.shape[1]),
                                  phi_c.T @ target_c)
    centered_correction = (query - center) @ coefficient
    coefficient_bound = np.linalg.norm(target_c) / (2 * np.sqrt(penalty))
    pointwise_bound = coefficient_bound * np.linalg.norm(query - center, axis=1)
    uniform_bound = abs(mean_target) + (np.sqrt(7) + np.linalg.norm(center)) * coefficient_bound
    measured = np.max(np.abs(mean_target + centered_correction))
    if np.max(np.linalg.norm(query, axis=1)) > np.sqrt(7) * (1 + 2e-14):
        raise AssertionError("Blockwise feature bound failed")
    if np.any(np.abs(centered_correction) > pointwise_bound + 2e-14):
        raise AssertionError("Ridge correction bound failed")
    if np.linalg.norm(coefficient) > coefficient_bound + 2e-14:
        raise AssertionError("Ridge coefficient bound failed")
    if measured > uniform_bound + 2e-14:
        raise AssertionError("Uniform correction bound failed")
    return {
        "scope": "Fabricated matrices and residuals only, with finite queries up to 1e300; no forecasting task or accuracy claim.",
        "training_rows": len(phi), "query_rows": len(query), "feature_width": phi.shape[1],
        "block_widths": [3, 4], "feature_norm_bound": float(np.sqrt(7)),
        "maximum_query_feature_norm": float(np.max(np.linalg.norm(query, axis=1))),
        "penalty_convention": "mean squared residual + alpha times squared coefficient norm; free intercept",
        "alpha": alpha, "equivalent_sum_loss_penalty": penalty,
        "coefficient_norm": float(np.linalg.norm(coefficient)),
        "coefficient_norm_bound": float(coefficient_bound),
        "training_feature_mean_norm": float(np.linalg.norm(center)),
        "training_residual_mean": float(mean_target),
        "training_centered_residual_rms": float(np.linalg.norm(target_c) / np.sqrt(len(phi))),
        "maximum_abs_query_correction": float(measured),
        "uniform_abs_correction_bound": float(uniform_bound),
        "maximum_pointwise_bound_violation": float(np.max(np.abs(centered_correction) - pointwise_bound)),
        "all_query_features_finite": bool(np.isfinite(query).all()),
    }


def run() -> dict:
    grid = np.linspace(-3.0, 4.0, 81)
    two = make_fixture([-1.0, 1.0])
    three = make_fixture([0.0, 1.0, 2.0])
    nonsymmetric = make_fixture([-1.0, 0.0, 2.0])
    summaries = {"two_reference_degeneracy": fixture_summary(two, grid),
                 "three_reference_distinctness": fixture_summary(three, grid),
                 "nonsymmetric_reference_witness": fixture_summary(nonsymmetric, grid)}
    values = reconstructions(two, grid)
    rho = float(two["kernel"][0, 1])
    lam = two["model"]["lambda_original"]
    multiplier = (1 + rho + lam) / (1 - rho + lam)
    degeneracy_error = float(np.max(np.abs(values["kahm"] - multiplier * values["rbf"])))
    departure_error = float(np.max(np.abs(values["d_kahm"] - multiplier * values["d_rbf"] - (multiplier - 1) * values["x"])))
    summaries["two_reference_degeneracy"].update({
        "exact_reparameterization_multiplier": multiplier,
        "reconstruction_identity_max_abs_error": degeneracy_error,
        "departure_identity_max_abs_error": departure_error,
    })
    q = nonsymmetric["q"][:, 0]
    pair_sums = [float(q[i] + q[j]) for i in range(len(q)) for j in range(i, len(q))]
    summaries["nonsymmetric_reference_witness"]["analytic_distinctness_conditions"] = {
        "unordered_reference_pair_sums": sorted(pair_sums),
        "all_unordered_reference_pair_sums_distinct": len(set(pair_sums)) == len(pair_sums),
        "reference_kernel_row_sums_nonconstant": bool(np.ptp(nonsymmetric["kernel"].sum(axis=1)) > 1e-6),
        "positive_inverse_times_one_by_sufficient_neumann_condition": nonsymmetric["neumann_condition"],
        "proof_location": "ALGEBRA.md, Proposition 4",
    }
    for fixture in summaries.values():
        if fixture["normalized_krr_identity_max_abs_error"] > 1e-12 or fixture["affine_target_identity_max_abs_error"] > 1e-12:
            raise AssertionError("An algebraic identity failed")
    if max(degeneracy_error, departure_error) > 1e-12:
        raise AssertionError("Symmetric-bank reparameterization failed")
    for name in ("three_reference_distinctness", "nonsymmetric_reference_witness"):
        if not summaries[name]["neumann_positivity_condition_passed"]:
            raise AssertionError("Fixture does not satisfy the analytic positivity condition")
        if summaries[name]["raw_kahm_departure_residual_to_common_plus_rbf_span"] < 1e-6:
            raise AssertionError("Distinctness witness is numerically unresolved")
    source_paths = [Path(__file__), ROOT / "extensions/direct_reconstruction/model.py",
                    ROOT / "extensions/predictive_geometry/study/vendor/upstream/kxx_matrix.py"]
    return {
        "schema_version": 1,
        "status": "passed",
        "algebra_decision": "GO_TO_ONE_MATCHED_FORECAST_STUDY",
        "decision_scope": "No universal structural redundancy: an explicit valid one-bank construction differs from affine and same-kernel RBF reconstruction features. This does not establish predictive value or distinctness for every fitted bank.",
        "forecast_evaluation_inputs_accessed": False,
        "forecast_model_fits": 0,
        "query_grid": {"start": -3.0, "stop": 4.0, "count": 81},
        "fixtures": summaries,
        "bounded_feature_and_ridge_witness": stability_witness(),
        "required_attribution_caveats": [
            "Transferring normalized KAHM weights directly to target values is exactly a normalized kernel-ridge ratio, not a distinct predictor to beat.",
            "Some symmetric banks give identical augmented raw feature spans after reparameterization.",
            "Isotropic ridge and separate bounded feature transforms need not be invariant under those reparameterizations.",
            "The actual tested object is the fixed reconstruction-plus-normalization-plus-decoder procedure; numerical sample ranks do not prove global function-space equality.",
            "Bounded corrections do not imply forecasting accuracy or a globally bounded raw baseline.",
        ],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("algebra_results.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "algebra_decision": result["algebra_decision"],
                      "output": str(args.output), "forecast_model_fits": 0}, indent=2))


if __name__ == "__main__":
    main()
