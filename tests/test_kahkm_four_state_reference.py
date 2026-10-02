"""Deterministic reference-model tests, not a sampling or coverage study.

Source: arXiv:2609.32652v1, Section 5.5, Eq. (131), and Table 2.
The rational reference independently constructs the four-state population.
Its conditional-mean risk is an exact lower bound here because the current
coordinates are constant within each reference class. Tests separately
check attainment by B and feasibility under the supported norm budgets.
"""

from fractions import Fraction
import math
import unittest

import numpy as np

from kahkm_four_state_reference import four_state_reference

P_VALUES = (0.5, 0.625, 0.8, 31 / 32, 1.0)
DYNAMICS = ("identity", "conflicting_successors")


def rational_population(p, dynamics):
    """Exact sums over four equiprobable states; no implementation helpers."""
    p = Fraction(float(p))
    q = 1 - p
    v, u = (p, q), (q, p)
    current = (v, v, u, u)
    future = current if dynamics == "identity" else (v, u, u, u)
    labels = (0, 0, 1, 1)
    means = tuple(
        tuple((future[2*c][j] + future[2*c + 1][j]) / 2 for j in range(2))
        for c in range(2)
    )
    f = sum((1 - current[i][labels[i]])**2 for i in range(4)) / 4
    risk = sum(
        (future[i][j] - means[labels[i]][j])**2
        for i in range(4) for j in range(2)
    ) / 4
    global_mean = tuple(sum(row[j] for row in current) / 4 for j in range(2))
    variance = sum(
        (current[i][j] - global_mean[j])**2
        for i in range(4) for j in range(2)
    ) / 4
    return float(f), float(risk), float(variance)


class FourStateReferenceTests(unittest.TestCase):
    def reference(self, **changes):
        params = dict(p=31 / 32, dynamics="conflicting_successors", kappa=math.sqrt(2))
        params.update(changes)
        return four_state_reference(**params)

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-15)

    def test_state_order_and_uniform_law(self):
        r = self.reference()
        self.assertEqual(r.reference_id, "four_state_uniform_reference_v1")
        self.assertEqual(r.state_names, ("a", "b", "d", "e"))
        self.assertEqual(r.states, ((1., 1.), (1., -1.), (-1., 1.), (-1., -1.)))
        self.assertEqual(r.probabilities, (0.25,) * 4)
        self.assertEqual(r.labels, (0, 0, 1, 1))
        self.assertEqual(r.p, 31 / 32)
        self.assertEqual(r.kappa, math.sqrt(2))
        self.assertEqual(r.dynamics, "conflicting_successors")

    def test_representation_is_fixed_across_dynamics(self):
        for p in P_VALUES:
            with self.subTest(p=p):
                first = self.reference(p=p, dynamics="identity")
                second = self.reference(p=p)
                expected = ((p, 1-p), (p, 1-p), (1-p, p), (1-p, p))
                self.assertEqual(first.phi, expected)
                self.assertEqual(second.phi, expected)
                for name in ("states", "labels", "probabilities", "assignment_loss",
                             "current_coordinate_variance"):
                    self.assertEqual(getattr(first, name), getattr(second, name))

    def test_deterministic_successor_maps(self):
        for dynamics, indices in (("identity", (0, 1, 2, 3)),
                                  ("conflicting_successors", (0, 2, 2, 2))):
            with self.subTest(dynamics=dynamics):
                r = self.reference(dynamics=dynamics)
                self.assertEqual(r.successor_indices, indices)
                self.assertEqual(r.successors, tuple(r.phi[j] for j in indices))

    def test_identity_predictions_and_zero_risk(self):
        for p in P_VALUES:
            with self.subTest(p=p):
                r = self.reference(p=p, dynamics="identity", kappa=1.0)
                self.assertEqual(r.optimal_B, ((1., 0.), (0., 1.)))
                self.assert_close(np.array(r.phi) @ np.array(r.optimal_B), r.successors)
                self.assertEqual(r.exact_optimal_mse, 0.0)
                self.assertEqual(r.exact_optimal_rmse, 0.0)
                self.assertEqual(r.population_certificate_limit, 0.0)
                self.assertEqual(r.sufficient_kappa, 1.0)

    def test_conflicting_predictions_and_matrix_orientation(self):
        for p in P_VALUES:
            with self.subTest(p=p):
                r = self.reference(p=p)
                B = np.array(r.optimal_B)
                expected = np.array([[.5, .5], [.5, .5], [1-p, p], [1-p, p]])
                self.assert_close(np.array(r.phi) @ B, expected)
                for phi, prediction in zip(r.phi, expected):
                    self.assert_close(B.T @ np.array(phi), prediction)

    def test_population_formulas_against_rational_sums(self):
        for dynamics in DYNAMICS:
            for p in P_VALUES:
                with self.subTest(dynamics=dynamics, p=p):
                    r = self.reference(p=p, dynamics=dynamics)
                    f, risk, variance = rational_population(p, dynamics)
                    self.assert_close(r.assignment_loss, f)
                    self.assert_close(r.sigma_res_sq, risk)
                    self.assert_close(r.exact_optimal_mse, risk)
                    self.assert_close(r.exact_optimal_rmse, math.sqrt(risk))
                    self.assert_close(r.current_coordinate_variance, variance)

    def test_attained_vector_risk_and_normal_equations(self):
        for dynamics in DYNAMICS:
            for p in P_VALUES:
                with self.subTest(dynamics=dynamics, p=p):
                    r = self.reference(p=p, dynamics=dynamics)
                    X, Z, B = map(np.array, (r.phi, r.successors, r.optimal_B))
                    weights = np.array(r.probabilities)
                    residual = Z - X @ B
                    risk = np.sum(weights * np.sum(residual**2, axis=1))
                    self.assert_close(risk, r.exact_optimal_mse)
                    self.assert_close(X.T @ (weights[:, None] * residual), np.zeros((2, 2)))

    def test_risk_decomposition_for_fixed_alternative_matrices(self):
        candidates = (
            np.zeros((2, 2)), np.eye(2), np.full((2, 2), .5),
            np.array([[-1., 2.], [3., -.25]]),
        )
        for dynamics in DYNAMICS:
            for p in P_VALUES:
                r = self.reference(p=p, dynamics=dynamics)
                X, Z, optimum = map(np.array, (r.phi, r.successors, r.optimal_B))
                weights = np.array(r.probabilities)
                for index, B in enumerate(candidates):
                    with self.subTest(dynamics=dynamics, p=p, candidate=index):
                        risk = np.sum(weights * np.sum((Z - X @ B)**2, axis=1))
                        penalty = np.sum(weights * np.sum((X @ (B - optimum))**2, axis=1))
                        self.assert_close(risk, r.exact_optimal_mse + penalty)
                        self.assertGreaterEqual(risk + 5e-15, r.exact_optimal_mse)

    def test_spectral_norm_feasibility_and_frobenius_bound(self):
        for p in P_VALUES:
            with self.subTest(p=p):
                r = self.reference(p=p)
                B = np.array(r.optimal_B)
                spectral = float(np.linalg.norm(B, ord=2))
                frobenius = float(np.linalg.norm(B, ord="fro"))
                self.assert_close(frobenius**2, p*p - p + 1.5)
                self.assert_close(r.sufficient_kappa, frobenius)
                self.assertLessEqual(spectral, r.sufficient_kappa + 5e-15)
                self.assertLessEqual(r.sufficient_kappa, math.sqrt(1.5) + 5e-15)
                identity = self.reference(p=p, dynamics="identity", kappa=1.0)
                self.assertEqual(float(np.linalg.norm(identity.optimal_B, ord=2)), 1.0)

    def test_documented_budget_boundary(self):
        for dynamics in DYNAMICS:
            for p in P_VALUES:
                with self.subTest(dynamics=dynamics, p=p):
                    threshold = self.reference(p=p, dynamics=dynamics).sufficient_kappa
                    at_boundary = self.reference(p=p, dynamics=dynamics, kappa=threshold)
                    self.assertEqual(at_boundary.kappa, threshold)
                    with self.assertRaises(NotImplementedError):
                        self.reference(p=p, dynamics=dynamics,
                                       kappa=math.nextafter(threshold, 0.0))
        # These are unsupported by this reference, not asserted to be infeasible.
        for dynamics in DYNAMICS:
            with self.subTest(dynamics=dynamics), self.assertRaises(NotImplementedError):
                self.reference(dynamics=dynamics, kappa=0.0)

    def test_table_two_values(self):
        r = self.reference()
        self.assertEqual(r.assignment_loss, 1 / 1024)
        self.assertEqual(r.sigma_res_sq, 225 / 1024)
        self.assertEqual(r.exact_optimal_mse, 225 / 1024)
        self.assertEqual(r.exact_optimal_rmse, 15 / 32)
        self.assertEqual(r.current_coordinate_variance, 225 / 512)
        self.assert_close(r.population_certificate_limit, 13 / 32)
        self.assert_close(r.exact_optimal_rmse - r.population_certificate_limit, 1 / 16)

    def test_positive_optimum_with_zero_population_certificate(self):
        r = self.reference(p=0.8)
        self.assert_close(r.exact_optimal_rmse, 0.3)
        self.assertGreater(r.exact_optimal_mse, 0.0)
        self.assertGreater(r.current_coordinate_variance, 0.0)
        self.assertEqual(r.population_certificate_limit, 0.0)

    def test_endpoint_limits(self):
        for dynamics in DYNAMICS:
            with self.subTest(dynamics=dynamics):
                soft = self.reference(p=.5, dynamics=dynamics)
                self.assertEqual(soft.phi, ((.5, .5),) * 4)
                self.assertEqual(soft.assignment_loss, .25)
                self.assertEqual(soft.exact_optimal_mse, 0.0)
                self.assertEqual(soft.current_coordinate_variance, 0.0)
        hard = self.reference(p=1.0)
        self.assertEqual(hard.assignment_loss, 0.0)
        self.assertEqual(hard.current_coordinate_variance, .5)
        self.assertEqual(hard.exact_optimal_mse, .25)
        self.assertEqual(hard.exact_optimal_rmse, .5)
        self.assertEqual(hard.population_certificate_limit, .5)

    def test_supported_larger_budgets_preserve_optimum(self):
        results = [self.reference(kappa=k) for k in (math.sqrt(2), 2., 4., 32.)]
        limits = []
        for r in results:
            self.assertEqual(r.exact_optimal_rmse, 15 / 32)
            expected = max(0., r.exact_optimal_rmse - r.kappa * math.sqrt(2*r.assignment_loss))
            self.assert_close(r.population_certificate_limit, expected)
            limits.append(r.population_certificate_limit)
        self.assertEqual(limits, sorted(limits, reverse=True))
        self.assertEqual(limits[-1], 0.0)

    def test_constant_predictor_comparison(self):
        for p in P_VALUES:
            with self.subTest(p=p):
                r = self.reference(p=p)
                Z = np.array(r.successors)
                mean = np.average(Z, axis=0, weights=r.probabilities)
                risk = np.average(np.sum((Z - mean)**2, axis=1), weights=r.probabilities)
                self.assert_close(risk, 3 * (2*p - 1)**2 / 8)
                self.assert_close(r.exact_optimal_mse, (2 / 3) * risk)

    def test_rejects_out_of_range_p(self):
        for value in (0., math.nextafter(.5, 0.), math.nextafter(1., math.inf), 2.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reference(p=value)

    def test_rejects_negative_kappa(self):
        for value in (-1., -1e-12):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reference(kappa=value)

    def test_rejects_nonfinite_parameters(self):
        for name in ("p", "kappa"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    self.reference(**{name: value})

    def test_rejects_invalid_numeric_types(self):
        for name in ("p", "kappa"):
            for value in (True, False, None, "0.8", 1j):
                with self.subTest(parameter=name, value=value), self.assertRaises(TypeError):
                    self.reference(**{name: value})

    def test_rejects_unknown_or_invalid_dynamics(self):
        for value in ("", "Identity", "other"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reference(dynamics=value)
        for value in (None, True, 1, ["identity"]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.reference(dynamics=value)


if __name__ == "__main__":
    unittest.main()
