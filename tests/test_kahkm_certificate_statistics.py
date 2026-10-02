"""Deterministic statistics checks, not an independence or coverage study.

The independent reference uses exact rational arithmetic on small inputs.
Duplicated rows below are algebraic fixtures, not independent observations.
"""

from fractions import Fraction
import math
import unittest

import numpy as np

from kahkm_certificate_statistics import statistics_from_pairs
from kahkm_certificates import certificate_from_statistics


def rational_reference(phi, successors, labels):
    """Loop-based exact calculation, without NumPy grouping or module helpers."""
    M, C = len(phi), len(phi[0])
    current = [[Fraction(float(x)) for x in row] for row in phi]
    future = [[Fraction(float(x)) for x in row] for row in successors]
    counts, means, sses = [], [], []
    for c in range(C):
        indices = [i for i in range(M) if int(labels[i]) == c]
        counts.append(len(indices))
        mean = [
            sum((future[i][j] for i in indices), Fraction(0)) / len(indices)
            if indices else Fraction(1, C)
            for j in range(C)
        ]
        sse = sum(
            ((future[i][j] - mean[j]) ** 2 for i in indices for j in range(C)),
            Fraction(0),
        )
        means.append(tuple(float(x) for x in mean))
        sses.append(sse)
    f = sum((1 - current[i][int(labels[i])]) ** 2 for i in range(M)) / M
    return {
        "f_hat": float(f), "s_hat": float(sum(sses) / M),
        "class_counts": tuple(counts),
        "class_successor_means": tuple(means),
        "class_successor_sse": tuple(float(x) for x in sses),
    }


class PairStatisticsTests(unittest.TestCase):
    def example(self):
        # Dyadic coordinates make the inputs exactly representable in float64.
        return dict(
            phi=np.array([
                [.75, .25, 0], [.5, .25, .25], [.25, .5, .25],
                [.125, .25, .625], [0, .25, .75],
            ]),
            successors=np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [.25, .5, .25], [.5, 0, .5],
            ]),
            labels=np.array([0, 0, 0, 2, 2]),
        )

    def evaluate(self, **changes):
        data = dict(
            phi=[[1, 0], [0, 1]],
            successors=[[.75, .25], [.25, .75]],
            labels=[0, 1],
        )
        data.update(changes)
        return statistics_from_pairs(**data)

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

    def test_exact_rational_reference(self):
        data = self.example()
        actual = statistics_from_pairs(**data)
        expected = rational_reference(**data)
        self.assertEqual((actual.n_pairs, actual.n_classes), (5, 3))
        self.assertEqual(actual.class_counts, expected["class_counts"])
        for name in ("f_hat", "s_hat", "class_successor_means", "class_successor_sse"):
            with self.subTest(quantity=name):
                self.assert_close(getattr(actual, name), expected[name])
        self.assertEqual(actual.current_max_row_sum_error, 0.0)
        self.assertEqual(actual.successor_max_row_sum_error, 0.0)
        self.assertEqual(actual.simplex_sum_atol, 1e-12)

    def test_unequal_class_sizes_use_total_pair_count(self):
        result = statistics_from_pairs(
            phi=[[1, 0], [1, 0], [1, 0], [0, 1]],
            successors=[[1, 0], [1, 0], [0, 1], [0, 1]],
            labels=[0, 0, 0, 1],
        )
        self.assertEqual(result.class_counts, (3, 1))
        self.assertEqual(result.f_hat, 0.0)
        self.assert_close(result.class_successor_sse, (4 / 3, 0))
        self.assert_close(result.s_hat, 1 / 3)
        # Not division by M-1, and not an unweighted average over classes.
        self.assertNotAlmostEqual(result.s_hat, 4 / 9)
        self.assertNotAlmostEqual(result.s_hat, 2 / 9)

    def test_supplied_labels_not_coordinate_argmax(self):
        result = self.evaluate(
            phi=[[.25, .75], [.25, .75]],
            successors=[[1, 0], [0, 1]], labels=[0, 0],
        )
        self.assertEqual(result.class_counts, (2, 0))
        self.assertEqual(result.f_hat, 9 / 16)
        self.assertEqual(result.s_hat, 1 / 2)

    def test_empty_classes_keep_zero_contributions(self):
        result = statistics_from_pairs(
            phi=[[1, 0, 0], [1, 0, 0]],
            successors=[[1, 0, 0], [0, 0, 1]], labels=[0, 0],
        )
        self.assertEqual(result.class_counts, (2, 0, 0))
        self.assertEqual(result.class_successor_sse, (1.0, 0.0, 0.0))
        self.assertEqual(result.s_hat, .5)
        self.assert_close(result.class_successor_means, [
            [.5, 0, .5], [1 / 3] * 3, [1 / 3] * 3,
        ])

    def test_singletons_have_zero_residual_variance(self):
        result = self.evaluate()
        self.assertEqual(result.class_counts, (1, 1))
        self.assertEqual(result.class_successor_sse, (0.0, 0.0))
        self.assertEqual(result.s_hat, 0.0)
        self.assert_close(result.class_successor_means, [[.75, .25], [.25, .75]])

    def test_one_pair_is_supported(self):
        result = self.evaluate(phi=[[.75, .25]], successors=[[.5, .5]], labels=[1])
        self.assertEqual(result.n_pairs, 1)
        self.assertEqual(result.f_hat, 9 / 16)
        self.assertEqual(result.s_hat, 0.0)

    def test_one_coordinate_is_supported(self):
        result = statistics_from_pairs(
            phi=[[1], [1], [1]], successors=[[1], [1], [1]], labels=[0, 0, 0],
        )
        self.assertEqual((result.n_pairs, result.n_classes), (3, 1))
        self.assertEqual(result.class_successor_means, ((1.0,),))
        self.assertEqual((result.f_hat, result.s_hat), (0.0, 0.0))

    def test_row_permutation_invariance(self):
        data = self.example()
        first = statistics_from_pairs(**data)
        order = np.array([4, 0, 3, 1, 2])
        second = statistics_from_pairs(**{k: v[order] for k, v in data.items()})
        self.assertEqual(first.class_counts, second.class_counts)
        for name in ("f_hat", "s_hat", "class_successor_means", "class_successor_sse"):
            self.assert_close(getattr(first, name), getattr(second, name))

    def test_joint_coordinate_and_label_permutation(self):
        data = self.example()
        first = statistics_from_pairs(**data)
        order = np.array([2, 0, 1])
        inverse = np.argsort(order)
        second = statistics_from_pairs(
            phi=data["phi"][:, order], successors=data["successors"][:, order],
            labels=inverse[data["labels"]],
        )
        self.assert_close(second.f_hat, first.f_hat)
        self.assert_close(second.s_hat, first.s_hat)
        self.assert_close(second.class_counts, np.array(first.class_counts)[order])
        self.assert_close(second.class_successor_sse, np.array(first.class_successor_sse)[order])
        self.assert_close(second.class_successor_means,
                          np.array(first.class_successor_means)[order][:, order])

    def test_readonly_inputs_are_not_modified(self):
        data = self.example()
        snapshots = {k: v.copy() for k, v in data.items()}
        for value in data.values():
            value.setflags(write=False)
        statistics_from_pairs(**data)
        for name in data:
            np.testing.assert_array_equal(data[name], snapshots[name])

    def test_roundoff_is_recorded_without_renormalization(self):
        eps = 2.0 ** -42
        result = self.evaluate(
            phi=[[.5, .5 + eps]], successors=[[.25, .75]], labels=[1],
        )
        self.assertEqual(result.current_max_row_sum_error, eps)
        self.assertEqual(result.successor_max_row_sum_error, 0.0)
        self.assertEqual(result.f_hat, (.5 - eps) ** 2)

    def test_end_to_end_balanced_fixture(self):
        # One conflicting class and one constant-successor class.
        # Repetition fixes algebraic counts; it does not establish coverage.
        p, q = 31 / 32, 1 / 32
        result = statistics_from_pairs(
            phi=np.tile([[p, q], [p, q], [q, p], [q, p]], (1024, 1)),
            successors=np.tile([[p, q], [q, p], [q, p], [q, p]], (1024, 1)),
            labels=np.tile([0, 0, 1, 1], 1024),
        )
        self.assertEqual(result.n_pairs, 4096)
        self.assertEqual(result.f_hat, 1 / 1024)
        self.assertEqual(result.s_hat, 225 / 1024)
        actual = certificate_from_statistics(
            f_hat=result.f_hat, s_hat=result.s_hat, n_pairs=result.n_pairs,
            kappa=math.sqrt(2), delta=.05,
        )
        expected = certificate_from_statistics(
            f_hat=1 / 1024, s_hat=225 / 1024, n_pairs=4096,
            kappa=math.sqrt(2), delta=.05,
        )
        self.assertEqual(actual, expected)
        self.assertGreater(actual.L_kappa_delta, .30)
        self.assertLess(actual.L_kappa_delta, .31)

    def test_rejects_bad_coordinate_shapes(self):
        values = ([1, 0], [], np.empty((0, 2)), np.empty((2, 0)), np.ones((2, 2, 1)))
        for name in ("phi", "successors"):
            for value in values:
                with self.subTest(field=name, shape=np.shape(value)):
                    with self.assertRaises(ValueError):
                        self.evaluate(**{name: value})

    def test_rejects_mismatched_coordinate_shapes(self):
        for value in ([[1, 0]], [[1, 0, 0], [0, 1, 0]]):
            with self.subTest(shape=np.shape(value)), self.assertRaises(ValueError):
                self.evaluate(successors=value)

    def test_rejects_nonfinite_coordinates(self):
        for name in ("phi", "successors"):
            for bad in (math.nan, math.inf, -math.inf):
                with self.subTest(field=name, value=bad), self.assertRaises(ValueError):
                    self.evaluate(**{name: [[bad, 0], [0, 1]]})

    def test_rejects_out_of_range_and_nonunit_rows(self):
        values = (
            [[-1e-14, 1], [0, 1]], [[1 + 1e-14, 0], [0, 1]],
            [[.4, .4], [0, 1]], [[.5, .5 + 2e-12], [0, 1]],
        )
        for name in ("phi", "successors"):
            for value in values:
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    self.evaluate(**{name: value})

    def test_rejects_invalid_coordinate_types(self):
        values = (
            np.eye(2, dtype=bool), np.eye(2, dtype=complex),
            np.eye(2).astype(str), np.eye(2, dtype=object),
            np.ma.array(np.eye(2), mask=False),
        )
        for name in ("phi", "successors"):
            for value in values:
                with self.subTest(field=name, dtype=value.dtype), self.assertRaises(TypeError):
                    self.evaluate(**{name: value})

    def test_rejects_bad_label_shapes(self):
        for value in (0, [], [0], [[0], [1]], [0, 1, 0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(labels=value)

    def test_rejects_invalid_label_types(self):
        values = (
            [0.0, 1.0], [False, True], ["0", "1"], [0j, 1j],
            np.array([0, 1], dtype=object), np.ma.array([0, 1], mask=False),
        )
        for value in values:
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.evaluate(labels=value)

    def test_rejects_out_of_range_labels(self):
        for value in ([-1, 1], [0, 2], np.array([0, 2**64 - 1], dtype=np.uint64)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(labels=value)


if __name__ == "__main__":
    unittest.main()
