"""Mathematical checks of retrospective diagnostics; no experiment fits."""
import unittest

import numpy as np

from scripts.review_diagnostics import (
    paired_diagnostics, paired_interval, recover_from_gated_coordinates,
    stacked_basis_diagnostics,
)


class ReviewDiagnosticsTests(unittest.TestCase):
    def test_student_interval_matches_known_small_sample_calculation(self):
        # These five symmetric observations have mean 0, variance 5/2, SE sqrt(1/2).
        result = paired_interval([-2, -1, 0, 1, 2], 1)
        self.assertEqual(result["mean"], 0)
        self.assertAlmostEqual(result["sd"], np.sqrt(2.5))
        self.assertAlmostEqual(result["se"], np.sqrt(0.5))
        self.assertAlmostEqual(result["critical_t"], 2.7764451051977987)
        self.assertAlmostEqual(result["upper"], 2.7764451051977987 / np.sqrt(2))
        self.assertEqual(result["lower"], -result["upper"])

    def test_deletion_keeps_family_and_changes_degrees_of_freedom(self):
        result = paired_diagnostics([1, 2, 3, 4, 5], 12, 0.2)
        self.assertAlmostEqual(result["bias_corrected_sample_skewness"], 0)
        self.assertEqual(result["original"]["degrees_of_freedom"], 4)
        self.assertEqual(len(result["leave_one_out"]), 5)
        # Deleting the final observation leaves [1,2,3,4], mean 2.5, SE sqrt(5/12).
        last = result["leave_one_out"][-1]
        self.assertEqual(last["family_size"], 12)
        self.assertEqual(last["degrees_of_freedom"], 3)
        self.assertEqual(last["mean"], 2.5)
        self.assertAlmostEqual(last["se"], np.sqrt(5 / 12))
        scaled = paired_diagnostics(np.array([1, 2, 3, 4, 5]) * 1000, 12, 200)
        self.assertAlmostEqual(scaled["leave_one_out_min_lower"] / 1000, result["leave_one_out_min_lower"])

    def test_zero_variance_is_explicit_and_invalid_samples_are_rejected(self):
        result = paired_diagnostics([2, 2, 2, 2], 29, 1)
        self.assertIsNone(result["bias_corrected_sample_skewness"])
        self.assertEqual(result["leave_one_out_min_lower"], 2)
        for values in ([1, 2], [1, 2, np.nan], [[1, 2], [3, 4]]):
            with self.assertRaises(ValueError):
                paired_interval(values, 1)

    def test_positive_gates_and_spanning_bases_recover_shifted_inputs(self):
        bases = [np.array([[1.], [0.]]), np.array([[0.], [1.]])]
        centers = [np.array([2., -3.]), np.array([-1., 4.])]
        points = np.array([[1., 8.], [-4., 3.], [0.25, -0.5]])
        gates = np.array([[0.1, 0.9], [1e-40, 1.], [0.7, 0.3]])
        gated = np.concatenate([gates[:, i:i+1] * ((points - center) @ basis)
                                for i, (basis, center) in enumerate(zip(bases, centers))], axis=1)
        rank = stacked_basis_diagnostics(bases)
        self.assertEqual(rank["numerical_rank"], 2)
        self.assertEqual(rank["condition_number_2"], 1)
        np.testing.assert_allclose(recover_from_gated_coordinates(gates, gated, bases, centers), points, rtol=1e-14, atol=1e-14)
        gates[0, 0] = 0
        with self.assertRaises(ValueError):
            recover_from_gated_coordinates(gates, gated, bases, centers)

    def test_deficient_stack_has_an_unseen_direction_without_proving_gate_loss(self):
        bases = [np.array([[1.], [0.]]), np.array([[1.], [0.]])]
        summary = stacked_basis_diagnostics(bases)
        self.assertEqual(summary["numerical_rank"], 1)
        self.assertFalse(summary["full_numerical_column_rank"])
        self.assertIsNone(summary["condition_number_2"])
        # Fixed positive gates give the same coordinates for points differing along e2.
        points = np.array([[3., 0.], [3., 9.]])
        projections = np.concatenate([points @ basis for basis in bases], axis=1)
        np.testing.assert_array_equal(projections[0], projections[1])
        # This is a counterexample for constant gates, not an assertion about variable gates.
        with self.assertRaises(ValueError):
            recover_from_gated_coordinates(np.full((2, 2), 0.5), projections * 0.5, bases, [np.zeros(2), np.zeros(2)])

    def test_conditioning_agrees_with_two_nearly_aligned_unit_directions(self):
        theta = 0.1
        bases = [np.array([[1.], [0.]]), np.array([[np.cos(theta)], [np.sin(theta)]])]
        summary = stacked_basis_diagnostics(bases)
        # Eigenvalues of A^T A are 1 +/- cos(theta); kappa_2 = cot(theta/2).
        np.testing.assert_allclose(summary["singular_values"], np.sqrt([1 + np.cos(theta), 1 - np.cos(theta)]), rtol=1e-12)
        self.assertAlmostEqual(summary["condition_number_2"], 1 / np.tan(theta / 2), places=12)


if __name__ == "__main__":
    unittest.main()
