"""Small independent diagnostics for the KAHM reconstruction manuscript.

No project implementation or saved experiment results are imported.
These floating-point calculations diagnose formulas; they are not proofs.
Run: python audit_method_theory.py
"""

from __future__ import annotations

import json
from pathlib import Path
import platform

import numpy as np


SEED = 260731
RNG = np.random.default_rng(SEED)
TOL = 2e-9


def local_geometry(references):
    q = np.asarray(references, dtype=float)
    if q.ndim == 1:
        q = q[:, None]
    center = q.mean(axis=0)
    _, singular, vt = np.linalg.svd(q - center, full_matrices=False)
    rank = int(np.sum(singular > 1e-11))
    p = vt[:rank].T
    projected = (q - center) @ p
    covariance = projected.T @ projected / (len(q) - 1)
    precision = np.linalg.inv(covariance)

    def kernels(queries):
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        difference = (queries[:, None, :] - q[None, :, :]) @ p
        log_kernel = -np.einsum("...i,ij,...j->...", difference, precision, difference) / (2 * rank)
        # This common positive row rescaling cancels in every tested ratio.
        # At reference queries the diagonal kernel is one, so the Gram is unchanged.
        return np.exp(log_kernel - log_kernel.max(axis=1, keepdims=True))

    return q, center, p, kernels, kernels(q)


def reference_regularizer(q, gram):
    energy = float(np.mean(q**2))
    lam = 2.5 * energy
    for iteration in range(100):
        inverse = np.linalg.inv(gram + lam * np.eye(len(q)))
        proposal = 2 * energy + float(np.mean((q - gram @ inverse @ q) ** 2))
        if abs(proposal - lam) <= 1e-3 or iteration == 99:
            return lam, inverse, iteration + 1
        lam = proposal
    raise AssertionError("Unreachable")


def maps(q, inverse, k):
    signed = k @ inverse
    normalizer = signed.sum(axis=1)
    if np.any(abs(normalizer) < 1e-12):
        raise ArithmeticError("A test query is too close to a singular normalization")
    weights = signed / normalizer[:, None]
    kahm = weights @ q
    rbf = (k / k.sum(axis=1)[:, None]) @ q
    return kahm, rbf, weights


def normalized_identity_checks():
    banks = [np.array([-1.0, 1.0]), np.array([-1.0, 0.0, 2.0]),
             RNG.normal(size=(6, 2)), RNG.normal(size=(8, 3))]
    transfer_errors, affine_errors, scaling_errors, rho_errors, sum_errors = [], [], [], [], []
    negative_rows = 0
    for reference in banks:
        q, _, _, kernel, gram = local_geometry(reference)
        lam = 1.7
        inverse = np.linalg.inv(gram + lam * np.eye(len(q)))
        queries = RNG.normal(size=(60, q.shape[1])) * 2
        k = kernel(queries)
        kahm, _, weights = maps(q, inverse, k)
        negative_rows += int(np.count_nonzero(np.min(weights, axis=1) < -1e-12))
        targets = RNG.normal(size=len(q))
        direct = weights @ targets
        ratio = (k @ np.linalg.solve(gram + lam * np.eye(len(q)), targets)) / (
            k @ np.linalg.solve(gram + lam * np.eye(len(q)), np.ones(len(q))))
        transfer_errors.extend(abs(direct - ratio))
        b = RNG.normal(size=q.shape[1])
        offset = 3.2
        affine_errors.extend(abs(weights @ (q @ b + offset) - (queries @ b + offset)
                                 - (kahm - queries) @ b))
        sum_errors.extend(abs(weights.sum(axis=1) - 1))
        raw_weights = k @ inverse
        rho = abs(raw_weights.sum(axis=1)) / np.abs(raw_weights).sum(axis=1)
        for factor in [1e-100, 1e-4, 10.0, 1e100]:
            scaled = (k * factor) @ inverse
            scaling_errors.extend(abs(scaled / scaled.sum(axis=1)[:, None] - weights).ravel())
            scaled_rho = abs(scaled.sum(axis=1)) / np.abs(scaled).sum(axis=1)
            rho_errors.extend(abs(scaled_rho - rho))
    maxima = {
        "max_target_transfer_error": float(max(transfer_errors)),
        "max_affine_target_identity_error": float(max(affine_errors)),
        "max_weight_sum_error": float(max(sum_errors)),
        "max_common_scaling_error": float(max(scaling_errors)),
        "max_relative_denominator_scaling_error": float(max(rho_errors)),
    }
    return {"banks": len(banks), "queries": 240, "rows_with_negative_weights": negative_rows,
            **maxima, "passed": bool(max(maxima.values()) < TOL)}


def equal_span_checks():
    square = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    redundant = np.column_stack([square, 2 * square[:, 0] - 3 * square[:, 1], np.ones(4)])
    results = []
    for name, references in [("two references", [-1.0, 1.0]),
                             ("square", square), ("redundant square coordinates", redundant)]:
        q, center, principal, kernel, gram = local_geometry(references)
        inverse = np.linalg.inv(gram + 1.3 * np.eye(len(q)))
        constant = float(gram.sum(axis=1).mean())
        a = 1 / (constant + 1.3)
        augmented = np.column_stack([q, np.ones(len(q))])
        coefficients = np.linalg.lstsq(augmented, inverse @ q, rcond=None)[0]
        b, c = coefficients[:-1], coefficients[-1]
        queries = RNG.normal(size=(120, q.shape[1])) * 2
        kahm, rbf, _ = maps(q, inverse, kernel(queries))
        affine = center + (queries - center) @ principal @ principal.T
        gates = 1 / (1 + np.sum((queries - center)**2, axis=1))
        common = np.column_stack([gates, gates[:, None] * ((queries - center) @ principal)])
        feature_k = np.column_stack([common, gates[:, None] * (kahm - affine)])
        feature_r = np.column_stack([common, gates[:, None] * (rbf - affine)])
        k_from_r = feature_r @ np.linalg.lstsq(feature_r, feature_k, rcond=None)[0]
        r_from_k = feature_k @ np.linalg.lstsq(feature_k, feature_r, rcond=None)[0]
        row = {
            "bank": name,
            "max_row_sum_deviation": float(np.max(abs(gram.sum(axis=1) - constant))),
            "max_invariance_residual": float(np.max(abs(augmented @ coefficients - inverse @ q))),
            "max_explicit_map_identity_error": float(np.max(abs(kahm - (rbf @ b + c) / a))),
            "max_K_from_R_span_residual": float(np.max(abs(feature_k - k_from_r))),
            "max_R_from_K_span_residual": float(np.max(abs(feature_r - r_from_k))),
            "K_sampled_rank": int(np.linalg.matrix_rank(feature_k, tol=1e-10)),
            "R_sampled_rank": int(np.linalg.matrix_rank(feature_r, tol=1e-10)),
        }
        row["passed"] = bool(max(value for key, value in row.items() if key.startswith("max_")) < TOL)
        results.append(row)
    return {"banks": len(results), "queries_per_bank": 120,
            "passed": all(row["passed"] for row in results), "details": results}


def explicit_examples():
    q2, _, _, kernel2, gram2 = local_geometry([-1.0, 1.0])
    lam2, c2, iterations2 = reference_regularizer(q2, gram2)
    z2 = np.linspace(-3, 4, 81)[:, None]
    k2, r2, _ = maps(q2, c2, kernel2(z2))
    rho = float(gram2[0, 1])
    multiplier = (1 + rho + lam2) / (1 - rho + lam2)
    q3, _, _, kernel3, gram3 = local_geometry([-1.0, 0.0, 2.0])
    lam3, c3, iterations3 = reference_regularizer(q3, gram3)
    z3 = np.linspace(-3, 4, 81)[:, None]
    k3, r3, _ = maps(q3, c3, kernel3(z3))
    design = np.column_stack([np.ones(81), z3[:, 0], (r3 - z3)[:, 0]])
    departure = (k3 - z3)[:, 0]
    discrepancy = float(np.max(abs(departure - design @ np.linalg.lstsq(design, departure, rcond=None)[0])))
    e_norm = float(np.max(np.sum(abs(gram3 - np.eye(3)), axis=1)))
    v = c3 @ np.ones(3)
    results = {
        "two_reference_lambda": float(lam2), "two_reference_inverse_evaluations": iterations2,
        "two_reference_multiplier": float(multiplier),
        "two_reference_identity_max_error": float(np.max(abs(k2 - multiplier * r2))),
        "irregular_lambda": float(lam3), "irregular_inverse_evaluations": iterations3,
        "irregular_E_infinity_norm": e_norm,
        "irregular_minimum_C_times_ones": float(v.min()),
        "irregular_Neumann_condition_margin": float(1 + lam3 - 2 * e_norm),
        "irregular_sampled_span_max_residual": discrepancy,
        "illustration_only": "A nonzero sampled residual is not a proof of global function-space distinctness.",
    }
    results["passed"] = bool(
        abs(lam2 - 2.6519665429866075) < TOL
        and abs(multiplier - 1.224037566462905) < TOL
        and results["two_reference_identity_max_error"] < TOL
        and abs(lam3 - 4.498274743339644) < TOL
        and abs(e_norm - 1.2314905926823394) < TOL
        and v.min() > 0 and 1 + lam3 > 2 * e_norm
        and abs(discrepancy - 0.05355670997078987) < TOL
    )
    return results


def radial(v):
    v = np.asarray(v, dtype=float)
    p = v.shape[-1]
    return v / np.sqrt(1 + np.sum(v**2, axis=-1, keepdims=True) / p)


def bounded_features(raw):
    p = raw.shape[1]
    if p == 1:
        return radial(raw)
    split = p // 2
    return np.column_stack([radial(raw[:, :split]), radial(raw[:, split:])])


def radial_checks():
    excess_norms, excess_distances = [], []
    pairs = 0
    for p in [1, 2, 5, 14]:
        for scale in [1e-6, 1.0, 1e6]:
            x = scale * RNG.normal(size=(40, p))
            y = scale * RNG.normal(size=(40, p))
            tx, ty = radial(x), radial(y)
            excess_norms.extend(np.linalg.norm(tx, axis=1) - np.sqrt(p))
            excess_distances.extend(np.linalg.norm(tx - ty, axis=1) - np.linalg.norm(x - y, axis=1))
            pairs += len(x)
    result = {"pairs": pairs, "max_norm_bound_violation": float(max(0.0, max(excess_norms))),
              "max_nonexpansiveness_violation": float(max(0.0, max(excess_distances)))}
    result["passed"] = max(result["max_norm_bound_violation"], result["max_nonexpansiveness_violation"]) < TOL
    return result


def ridge_checks():
    coefficient_excess, correction_excess, estimator_errors = [], [], []
    fits = 0
    for n in [1, 2, 7, 40]:
        for p in [1, 3, 12]:
            features = bounded_features(RNG.normal(size=(n, p)) + 0.7)
            e = 3 + RNG.normal(size=n)
            centered = features - features.mean(axis=0)
            e_centered = e - e.mean()
            sigma = float(np.sqrt(np.mean(e_centered**2)))
            queries = bounded_features(5 * RNG.normal(size=(32, p)))
            for alpha in [1e-4, 1e-2, 1.0]:
                coefficient = np.linalg.solve(centered.T @ centered + n * alpha * np.eye(p),
                                              centered.T @ e_centered)
                augmented_x = np.vstack([centered / np.sqrt(n), np.sqrt(alpha) * np.eye(p)])
                augmented_y = np.concatenate([e_centered / np.sqrt(n), np.zeros(p)])
                independent = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
                estimator_errors.append(float(np.max(abs(coefficient - independent))))
                coefficient_excess.append(float(np.linalg.norm(coefficient) - sigma / (2 * np.sqrt(alpha))))
                intercept = float(e.mean() - features.mean(axis=0) @ coefficient)
                magnitude = abs(float(e.mean())) + (np.sqrt(p) + np.linalg.norm(features.mean(axis=0))) * sigma / (2 * np.sqrt(alpha))
                for gamma in [-2.0, 0.0, 0.25, 1.0]:
                    values = gamma * (intercept + queries @ coefficient)
                    correction_excess.extend(abs(values) - abs(gamma) * magnitude)
                fits += 1
    # An attaining example for the general bounded-feature proposition.
    alpha, sigma = 0.25, 2.0
    x = np.array([[-np.sqrt(alpha)], [np.sqrt(alpha)]])
    e = np.array([3 - sigma, 3 + sigma])
    w = np.linalg.solve(x.T @ x + 2 * alpha * np.eye(1), x.T @ (e - e.mean()))
    result = {
        "fits": fits, "queries_per_fit": 32, "gamma_values": [-2.0, 0.0, 0.25, 1.0],
        "includes_single_observation_fit": True,
        "max_solver_formulation_discrepancy": float(max(estimator_errors)),
        "max_coefficient_bound_violation": float(max(0.0, max(coefficient_excess))),
        "max_correction_bound_violation": float(max(0.0, max(correction_excess))),
        "attaining_example_coefficient_bound_error": float(abs(w[0] - sigma / (2 * np.sqrt(alpha)))),
        "attaining_example_correction_bound_error": float(abs((3 + w[0]) - (3 + sigma / (2 * np.sqrt(alpha))))),
    }
    result["passed"] = all(value < TOL for key, value in result.items()
                           if key.startswith("max_") or key.endswith("_error"))
    return result


def main():
    checks = {
        "normalized_kernel_identities": normalized_identity_checks(),
        "equal_augmented_raw_spans": equal_span_checks(),
        "reported_algebra_examples": explicit_examples(),
        "radial_map": radial_checks(),
        "centered_ridge_bound": ridge_checks(),
    }
    result = {
        "scope": "Small independent mathematical and numerical diagnostics, without forecasting replay",
        "seed": SEED, "absolute_tolerance": TOL,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "limitations": [
            "Floating-point diagnostics do not constitute formal proofs or verified arithmetic.",
            "Sampled span checks do not establish global function-space equality or distinctness.",
            "The full forecasting implementation, experiment protocol, and statistical conclusions are not reproduced here.",
            "Raw span checks intentionally exclude standardization, radial transformations, ridge penalties, and fallback boundaries.",
        ],
        "checks": checks,
        "all_checks_passed": all(check["passed"] for check in checks.values()),
    }
    output = Path(__file__).with_name("method_theory_checks.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
