"""Deterministic checks of count-based empirical statistics.

Counts here are algebraic fixtures, not randomly sampled experimental evidence.
The rational reference uses weighted residual sums, not the compact n_a*n_b
formula. Expanded-pair checks use small counts only. Large-count checks never
allocate one row per observation. No statistical coverage claim is tested.
"""

from dataclasses import asdict, replace
from fractions import Fraction
from itertools import product
import json
import math
import unittest

import numpy as np

from kahkm_certificate_statistics import statistics_from_pairs
from kahkm_certificates import certificate_from_statistics
from kahkm_four_state_counts import statistics_from_state_counts
from kahkm_four_state_reference import four_state_reference


def rational_statistics(p, dynamics, counts):
    """Independent exact weighted sums over the four specified state types."""
    p = Fraction(float(p))
    q = 1 - p
    v, u = (p, q), (q, p)
    current = (v, v, u, u)
    future = current if dynamics == "identity" else (v, u, u, u)
    labels = (0, 0, 1, 1)
    counts = tuple(int(n) for n in counts)
    M = sum(counts)
    class_counts, means, sses = [], [], []
    for c in range(2):
        indices = [i for i in range(4) if labels[i] == c]
        n = sum(counts[i] for i in indices)
        mean = tuple(
            sum((counts[i] * future[i][j] for i in indices), Fraction(0)) / n
            if n else Fraction(1, 2)
            for j in range(2)
        )
        sse = sum(
            (counts[i] * (future[i][j] - mean[j])**2
             for i in indices for j in range(2)), Fraction(0),
        )
        class_counts.append(n)
        means.append(tuple(float(x) for x in mean))
        sses.append(sse)
    f = sum(counts[i] * (1-current[i][labels[i]])**2 for i in range(4)) / M
    return dict(
        f_hat=float(f), s_hat=float(sum(sses) / M),
        class_counts=tuple(class_counts),
        class_successor_means=tuple(means),
        class_successor_sse=tuple(float(x) for x in sses),
    )


def expanded_statistics(reference, counts):
    """Use the separately tested pair implementation for small fixtures."""
    indices = np.repeat(np.arange(4), counts)
    return statistics_from_pairs(
        phi=np.asarray(reference.phi)[indices],
        successors=np.asarray(reference.successors)[indices],
        labels=np.asarray(reference.labels)[indices],
    )


class FourStateCountTests(unittest.TestCase):
    def reference(self, p=31/32, dynamics="conflicting_successors"):
        return four_state_reference(p=p, dynamics=dynamics, kappa=math.sqrt(2))

    def evaluate(self, counts, **changes):
        return statistics_from_state_counts(
            reference=self.reference(**changes), state_counts=counts,
        )

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-15)

    def assert_statistics_equal(self, actual, expected):
        for name in ("n_pairs", "n_classes", "class_counts", "simplex_sum_atol"):
            self.assertEqual(getattr(actual, name), getattr(expected, name))
        for name in ("f_hat", "s_hat", "class_successor_means", "class_successor_sse",
                     "current_max_row_sum_error", "successor_max_row_sum_error"):
            self.assert_close(getattr(actual, name), getattr(expected, name))

    def test_small_count_grid_matches_expanded_pairs(self):
        # 80 nonempty count patterns x 4 p values x 2 dynamics = 640 cases.
        for dynamics in ("identity", "conflicting_successors"):
            for p in (.5, .8, 31/32, 1.):
                reference = self.reference(p=p, dynamics=dynamics)
                for counts in product(range(3), repeat=4):
                    if not sum(counts):
                        continue
                    with self.subTest(dynamics=dynamics, p=p, counts=counts):
                        actual = statistics_from_state_counts(
                            reference=reference, state_counts=counts,
                        )
                        self.assertEqual(actual.state_counts, counts)
                        self.assert_statistics_equal(
                            actual.statistics, expanded_statistics(reference, counts),
                        )

    def test_exact_rational_weighted_reference(self):
        cases = (
            (3, 1, 2, 0), (0, 5, 0, 7), (1000003, 700009, 1100009, 3),
            (2**51-3, 17, 2**50+1, 0), (0, 0, 2**53, 0),
        )
        for dynamics in ("identity", "conflicting_successors"):
            for p in (.5, .625, .8, 31/32, 1.):
                for counts in cases:
                    with self.subTest(dynamics=dynamics, p=p, counts=counts):
                        actual = self.evaluate(counts, p=p, dynamics=dynamics).statistics
                        expected = rational_statistics(p, dynamics, counts)
                        self.assertEqual(actual.n_pairs, sum(counts))
                        self.assertEqual(actual.class_counts, expected["class_counts"])
                        for name in ("f_hat", "s_hat", "class_successor_means",
                                     "class_successor_sse"):
                            self.assert_close(getattr(actual, name), expected[name])

    def test_end_to_end_certificates(self):
        reference = self.reference()
        for counts in ((1024,)*4, (1536, 768, 1024, 768)):
            with self.subTest(counts=counts):
                result = self.evaluate(counts)
                self.assertEqual(result.statistics_id, "four_state_count_statistics_v1")
                compact = result.statistics
                expanded = expanded_statistics(reference, counts)
                self.assert_statistics_equal(compact, expanded)
                bounds = [certificate_from_statistics(
                    f_hat=s.f_hat, s_hat=s.s_hat, n_pairs=s.n_pairs,
                    kappa=reference.kappa, delta=.05,
                ) for s in (compact, expanded)]
                for name in ("F_delta", "V_delta", "L_kappa_delta"):
                    self.assert_close(getattr(bounds[0], name), getattr(bounds[1], name))
                if counts == (1024,)*4:
                    self.assertEqual(compact.f_hat, 1/1024)
                    self.assertEqual(compact.s_hat, 225/1024)
                    self.assertGreater(bounds[0].L_kappa_delta, .30)
                    self.assertLess(bounds[0].L_kappa_delta, .31)

    def test_unbalanced_counts_are_not_replaced_by_population_weights(self):
        result = self.evaluate((9, 1, 0, 2), p=1.).statistics
        self.assertEqual(result.class_counts, (10, 2))
        self.assert_close(result.class_successor_means, ((.9, .1), (0., 1.)))
        self.assert_close(result.class_successor_sse, (1.8, 0.))
        self.assert_close(result.s_hat, .15)
        self.assertNotAlmostEqual(result.s_hat, self.reference(p=1.).sigma_res_sq)

    def test_replication_scaling_is_algebraic(self):
        counts = (7, 3, 2, 1)
        first = self.evaluate(counts).statistics
        for factor in (2, 37):
            with self.subTest(factor=factor):
                second = self.evaluate(tuple(factor*n for n in counts)).statistics
                self.assertEqual(second.n_pairs, factor*first.n_pairs)
                self.assertEqual(second.class_counts, tuple(factor*n for n in first.class_counts))
                for name in ("f_hat", "s_hat", "class_successor_means"):
                    self.assert_close(getattr(first, name), getattr(second, name))
                self.assert_close(second.class_successor_sse,
                                  factor*np.asarray(first.class_successor_sse))

    def test_swapping_conflicting_counts_changes_mean_not_variance(self):
        first = self.evaluate((9, 1, 3, 4)).statistics
        second = self.evaluate((1, 9, 3, 4)).statistics
        self.assertEqual(first.class_counts, second.class_counts)
        self.assert_close(first.f_hat, second.f_hat)
        self.assert_close(first.s_hat, second.s_hat)
        self.assert_close(first.class_successor_sse, second.class_successor_sse)
        self.assert_close(first.class_successor_means[0], second.class_successor_means[0][::-1])
        self.assertNotEqual(first.class_successor_means[0], second.class_successor_means[0])

    def test_redistributing_second_class_counts_changes_no_statistics(self):
        for dynamics in ("identity", "conflicting_successors"):
            with self.subTest(dynamics=dynamics):
                first = self.evaluate((7, 3, 9, 0), dynamics=dynamics)
                second = self.evaluate((7, 3, 2, 7), dynamics=dynamics)
                self.assertNotEqual(first.state_counts, second.state_counts)
                self.assertEqual(first.statistics, second.statistics)

    def test_single_state_samples_and_empty_classes(self):
        for dynamics in ("identity", "conflicting_successors"):
            reference = self.reference(dynamics=dynamics)
            for state in range(4):
                with self.subTest(dynamics=dynamics, state=state):
                    counts = tuple(3 if i == state else 0 for i in range(4))
                    result = self.evaluate(counts, dynamics=dynamics).statistics
                    occupied = reference.labels[state]
                    self.assertEqual(result.class_counts[occupied], 3)
                    self.assertEqual(result.class_counts[1-occupied], 0)
                    self.assertEqual(result.class_successor_means[occupied], reference.successors[state])
                    self.assertEqual(result.class_successor_means[1-occupied], (.5, .5))
                    self.assertEqual(result.s_hat, 0.)
                    self.assertEqual(result.class_successor_sse, (0., 0.))

    def test_identity_and_constant_representation_have_zero_variance(self):
        counts = (1000003, 17, 500009, 1)
        for p in (.5, .8, 31/32, 1.):
            with self.subTest(p=p):
                result = self.evaluate(counts, p=p, dynamics="identity").statistics
                self.assertEqual(result.s_hat, 0.)
                self.assertEqual(result.class_successor_sse, (0., 0.))
        constant = self.evaluate(counts, p=.5).statistics
        self.assertEqual(constant.f_hat, .25)
        self.assertEqual(constant.s_hat, 0.)
        self.assert_close(constant.class_successor_means, ((.5, .5), (.5, .5)))

    def test_hard_assignment_empirical_variance_extreme(self):
        result = self.evaluate((17, 17, 0, 0), p=1.).statistics
        self.assertEqual(result.f_hat, 0.)
        self.assertEqual(result.s_hat, .5)
        self.assertEqual(result.class_successor_sse, (17., 0.))
        self.assertEqual(result.class_successor_means, ((.5, .5), (.5, .5)))

    def test_integer_inputs_and_readonly_arrays(self):
        expected = self.evaluate((7, 3, 2, 1))
        readonly = np.array([7, 3, 2, 1], dtype=np.int64)
        readonly.setflags(write=False)
        for values in ([7, 3, 2, 1], np.array([7, 3, 2, 1], dtype=np.uint64), readonly):
            with self.subTest(input_type=type(values).__name__):
                result = self.evaluate(values)
                self.assertEqual(result, expected)
                self.assertTrue(all(type(n) is int for n in result.state_counts))
        np.testing.assert_array_equal(readonly, [7, 3, 2, 1])
        self.assertFalse(readonly.flags.writeable)

    def test_maximum_supported_total_without_row_expansion(self):
        result = self.evaluate((2**52, 2**52, 0, 0), p=1.).statistics
        self.assertEqual(result.n_pairs, 2**53)
        self.assertEqual(result.class_counts, (2**53, 0))
        self.assertEqual(result.s_hat, .5)
        self.assertEqual(result.class_successor_sse, (float(2**52), 0.))
        single = self.evaluate((2**53, 0, 0, 0), p=1.).statistics
        self.assertEqual(single.s_hat, 0.)

    def test_rejects_zero_negative_and_excessive_totals(self):
        cases = (
            (0, 0, 0, 0), (2, -1, 0, 0), (2**53+1, 0, 0, 0),
            (2**52, 2**52, 1, 0),
            np.array([2**63, 2**63, 0, 0], dtype=np.uint64),
        )
        for counts in cases:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.evaluate(counts)

    def test_rejects_wrong_lengths(self):
        for counts in ((), (1,), (1, 0, 0), (1, 0, 0, 0, 0)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.evaluate(counts)

    def test_rejects_noninteger_counts(self):
        cases = (
            [1., 0, 0, 0], [1, True, 0, 0], [1, np.bool_(False), 0, 0],
            ["1", 0, 0, 0], [1, None, 0, 0], [1+0j, 0, 0, 0],
            [Fraction(1), 0, 0, 0], [math.nan, 0, 0, 1], [math.inf, 0, 0, 1],
            np.array([1, 0, 0, 0], dtype=float),
            np.array([1, 0, 0, 0], dtype=bool),
        )
        for counts in cases:
            with self.subTest(counts=counts), self.assertRaises(TypeError):
                self.evaluate(counts)

    def test_rejects_nonsequence_and_nested_inputs(self):
        for counts in (None, True, 1, np.int64(1), np.array(1), [[1], [0], [0], [0]]):
            with self.subTest(counts=counts), self.assertRaises(TypeError):
                self.evaluate(counts)

    def test_rejects_masked_arrays(self):
        for mask in (False, [False, True, False, False]):
            with self.subTest(mask=mask), self.assertRaises(TypeError):
                self.evaluate(np.ma.array([1, 0, 0, 0], mask=mask))

    def test_rejects_wrong_reference_types(self):
        for reference in (None, 1, "reference", {}):
            with self.subTest(reference=reference), self.assertRaises(TypeError):
                statistics_from_state_counts(reference=reference, state_counts=(1, 1, 1, 1))

    def test_rejects_altered_reference_fields(self):
        reference = self.reference()
        changes = (
            dict(reference_id="unverified"), dict(probabilities=(.5, 0., .25, .25)),
            dict(labels=(1, 1, 0, 0)), dict(successor_indices=(0, 1, 2, 3)),
            dict(successors=reference.phi), dict(assignment_loss=0.),
            dict(exact_optimal_mse=0.), dict(population_certificate_limit=0.),
            dict(sufficient_kappa=0.),
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                statistics_from_state_counts(
                    reference=replace(reference, **change), state_counts=(1, 1, 1, 1),
                )

    def test_json_roundtrip_preserves_recomputable_evidence(self):
        reference = self.reference()
        result = self.evaluate((123, 57, 31, 9))
        saved = json.loads(json.dumps(
            dict(reference=asdict(reference), result=asdict(result)), allow_nan=False,
        ))
        self.assertEqual(saved["result"]["state_counts"], [123, 57, 31, 9])
        rebuilt_reference = four_state_reference(**{
            name: saved["reference"][name] for name in ("p", "dynamics", "kappa")
        })
        rebuilt = statistics_from_state_counts(
            reference=rebuilt_reference, state_counts=saved["result"]["state_counts"],
        )
        self.assertEqual(rebuilt_reference, reference)
        self.assertEqual(rebuilt, result)


if __name__ == "__main__":
    unittest.main()
