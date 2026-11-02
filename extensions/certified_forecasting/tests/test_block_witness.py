"""Independent mathematical checks of normalized Lipschitz block minima.

The principal references are closed-form optima and the pathwise subset
inequality. The existing finite-support solver is an additional differential
check; it uses a different constraint system and objective normalization.
"""
import importlib.util
from itertools import combinations
from pathlib import Path
import sys
import unittest

import numpy as np

from certified_forecasting.block_witness import (
    block_score_bound,
    lipschitz_block_bracket,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "witness_core_for_block_tests", ROOT / "experiments/witness_comparison/core.py"
)
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class BlockWitnessTests(unittest.TestCase):
    def assert_contains(self, result, exact, tolerance=2e-9):
        self.assertLessEqual(result["lower"], exact + tolerance)
        self.assertGreaterEqual(result["upper"], exact - tolerance)
        self.assertLessEqual(result["gap"], 1e-7)
        self.assertLessEqual(result["feasibility_residual"], 1e-10)
        self.assertTrue(all(value >= 0 for value in result["dual_multipliers"]))

    def test_two_observations_equal_the_analytic_pair_score(self):
        # Includes a duplicated feature, zero target discrepancy, an active
        # constraint, and a pair whose target separation is representable.
        for distance in (0.0, 0.25, 1.0, 3.0):
            for kappa in (0.0, 0.4, 2.0):
                for targets in ((-1.0, 1.0), (0.3, 0.3)):
                    with self.subTest(distance=distance, kappa=kappa, y=targets):
                        result = lipschitz_block_bracket([0.0, distance], targets, kappa)
                        exact = 0.5 * max(abs(targets[1] - targets[0]) - kappa * distance, 0) ** 2
                        self.assert_contains(result, exact)

    def test_zero_budget_is_unbiased_sample_variance(self):
        # A free constant intercept makes the finite optimum sample variance
        # with denominator b-1, not the biased empirical variance.
        y = np.array([-0.9, 0.3, 0.3, 0.8, 1.0])
        x = np.array([[0.0, 1.0], [2.0, -1.0], [3.0, 0.0], [1.0, 7.0], [9.0, 9.0]])
        self.assert_contains(lipschitz_block_bracket(x, y, 0.0), y.var(ddof=1))
        self.assert_contains(lipschitz_block_bracket(np.ones((len(y), 2)), y, 100.0), y.var(ddof=1))

    def test_sharp_scalar_range_for_even_and_odd_blocks(self):
        for b in (2, 3, 4, 5, 8, 9):
            y = np.array([-0.75] * (b // 2) + [1.25] * (b - b // 2))
            exact = 4.0 * (b // 2) * (b - b // 2) / (b * (b - 1))
            self.assertAlmostEqual(block_score_bound(b, 2.0), exact, places=14)
            self.assert_contains(lipschitz_block_bracket(np.zeros((b, 1)), y, 1.0), exact)
        self.assertEqual(block_score_bound(3, 0.0), 0.0)

    def test_rare_conflict_count_formula(self):
        # Common atom (3,0); conflicting rare atoms (0,-1), (1,1).
        # If rare counts are a,c, exact SSE=a*c/(a+c), independently of
        # the common count. This covers missing atoms and unequal masses.
        for common, a, c in ((4, 0, 0), (4, 1, 0), (3, 1, 1), (2, 3, 1), (0, 2, 3), (7, 4, 2)):
            x = np.array([3.0] * common + [0.0] * a + [1.0] * c)
            y = np.array([0.0] * common + [-1.0] * a + [1.0] * c)
            exact = (a * c / (a + c) if a + c else 0.0) / (len(y) - 1)
            self.assert_contains(lipschitz_block_bracket(x, y, 1.0), exact)

    def test_pathwise_average_subset_inequality(self):
        # The theorem is pathwise, stronger than Monte Carlo monotonicity of
        # the expected score. Comparing lower(full) with upper(subsets)
        # makes the numerical check conservative up to the explicit tolerance.
        designs = (
            (np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([-1.0, 1.0, -1.0, 1.0, -1.0]), 1.0),
            (np.array([3.0, 3.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0, 1.0, -1.0]), 1.0),
        )
        for x, y, kappa in designs:
            full = lipschitz_block_bracket(x, y, kappa)
            for m in (2, 3, 4):
                subset_upper = [
                    lipschitz_block_bracket(x[list(s)], y[list(s)], kappa)["upper"]
                    for s in combinations(range(len(y)), m)
                ]
                self.assertGreaterEqual(full["lower"] + 2e-8, np.mean(subset_upper))

    def test_max_block_metric_and_exact_block_duplication(self):
        x = np.array([[0.0], [1.0]])
        y = np.array([-1.0, 1.0])
        original = lipschitz_block_bracket(x, y, 1.0)
        duplicated = lipschitz_block_bracket(np.hstack((x, x)), y, 1.0, [slice(0, 1), slice(1, 2)])
        euclidean = lipschitz_block_bracket(np.hstack((x, x)), y, 1.0)
        self.assert_contains(original, 0.5)
        self.assert_contains(duplicated, 0.5)
        self.assert_contains(euclidean, 0.5 * (2.0 - np.sqrt(2.0)) ** 2)
        self.assertGreater(duplicated["lower"], euclidean["upper"])

    def test_permutation_and_translation_invariance(self):
        x = np.array([-1.0, -0.5, 0.0, 0.0, 0.5, 1.0])
        y = np.array([-1.0, 1.0, -0.2, 0.2, 1.0, -1.0])
        permutation = [4, 1, 5, 3, 0, 2]
        original = lipschitz_block_bracket(x, y, 0.5)
        changed = lipschitz_block_bracket(x[permutation] + 3.25, y[permutation] - 7.125, 0.5)
        self.assertLessEqual(abs(original["lower"] - changed["lower"]), 1e-8)
        self.assertLessEqual(abs(original["upper"] - changed["upper"]), 1e-8)
        self.assertLessEqual(changed["feasibility_residual"], 1e-10)

    def test_agrees_with_existing_weighted_finite_support_solver(self):
        # The older solver enforces adjacent 1D constraints plus a target box;
        # the new solver enforces every pair and no box. Projection into the
        # common target interval makes their optimum values identical.
        counts_by_law = (
            [3, 5], [3, 5], [2, 3, 3], [1, 2, 1, 2, 2],
            [1, 2, 1, 1, 2, 1], [2, 1, 1, 2, 1, 1], [6, 1, 1], [3, 2, 3],
        )
        for law, counts in zip(CORE.laws(), counts_by_law):
            with self.subTest(law=law.name):
                x, y, _, _ = law.arrays()
                b = sum(counts)
                result = lipschitz_block_bracket(np.repeat(x, counts), np.repeat(y, counts), law.kappa)
                reference = CORE.solve_lipschitz(law, np.asarray(counts) / b)
                factor = b / (b - 1)
                self.assertLessEqual(result["lower"], factor * reference["upper"] + 1e-9)
                self.assertGreaterEqual(result["upper"], factor * reference["lower"] - 1e-9)
                self.assertLessEqual(result["gap"], 1e-7)

    def test_invalid_inputs_and_nonfinite_arithmetic_rejected(self):
        for features, targets, kappa, blocks in (
            ([[0]], [0], 1, None),
            ([[0], [1]], [0, np.nan], 1, None),
            ([[0], [np.inf]], [0, 1], 1, None),
            ([[0], [1]], [0, 1], -1, None),
            ([[0], [1]], [0, 1], np.inf, None),
            ([[0, 0], [1, 1]], [0, 1], 1, [slice(0, 1)]),
            ([[0, 0], [1, 1]], [0, 1], 1, [slice(0, 2), slice(1, 2)]),
        ):
            with self.assertRaises(ValueError):
                lipschitz_block_bracket(features, targets, kappa, blocks)
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaises(ValueError):
                lipschitz_block_bracket([[-1e200], [1e200]], [-1.0, 1.0], 1.0)
            with self.assertRaises((ValueError, ArithmeticError)):
                lipschitz_block_bracket([[0.0], [1.0]], [-1e200, 1e200], 1.0)
        for b, diameter in ((True, 1), (1, 1), (2.5, 1), (np.inf, 1), (2, -1), (2, np.inf), (2, 1e308)):
            with self.assertRaises(ValueError):
                block_score_bound(b, diameter)


if __name__ == "__main__":
    unittest.main()
