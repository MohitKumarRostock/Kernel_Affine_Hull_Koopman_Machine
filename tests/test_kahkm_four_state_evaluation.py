"""Deterministic evaluation checks, not a coverage or power experiment.

Count vectors are hand-constructed fixtures, not sampled campaign evidence.
One very unbalanced fixture exceeds the population optimum; this checks that
such outcomes are retained, not that the probabilistic guarantee is invalid.
Injected certificate values test comparison boundaries only. No new bound
or statistical claim is inferred from those injected values.
"""

from contextlib import ExitStack
from dataclasses import FrozenInstanceError, asdict, replace
import json
import math
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import numpy as np

import kahkm_four_state_evaluation as evaluation
from kahkm_certificates import certificate_from_statistics
from kahkm_four_state_reference import four_state_reference


class FourStateEvaluationTests(unittest.TestCase):
    def reference(self, p=31/32, dynamics="conflicting_successors"):
        return four_state_reference(p=p, dynamics=dynamics, kappa=math.sqrt(2))

    def evaluate(self, **changes):
        params = dict(reference=self.reference(), state_counts=(1024,)*4,
                      delta=.05, exclusion_tolerances=[.1, .3, .45],
                      comparison_atol=1e-12)
        params.update(changes)
        return evaluation.evaluate_four_state_counts(**params)

    def failure_case(self, **changes):
        params = dict(reference=self.reference(p=1.),
                      state_counts=(4096, 4096, 0, 0),
                      exclusion_tolerances=[.3, .5, .9])
        params.update(changes)
        return self.evaluate(**params)

    def assert_close(self, actual, expected):
        self.assertTrue(math.isclose(actual, expected, rel_tol=5e-13, abs_tol=5e-15),
                        msg=f"{actual!r} != {expected!r}")

    def test_balanced_example_and_component_integration(self):
        result = self.evaluate()
        expected = certificate_from_statistics(
            f_hat=1/1024, s_hat=225/1024, n_pairs=4096,
            kappa=math.sqrt(2), delta=.05,
        )
        self.assertEqual(result.evaluation_id, "four_state_original_iid_evaluation_v1")
        self.assertEqual(result.certificate_id, "original_iid_eq8_v1")
        self.assertEqual(result.reference, self.reference())
        self.assertEqual(result.count_statistics.state_counts, (1024,)*4)
        self.assertEqual(result.certificate, expected)
        self.assertTrue(result.strict_coverage)
        self.assertTrue(result.tolerance_aware_coverage)
        self.assertFalse(result.zero_certificate)
        self.assertEqual([d.excluded for d in result.exclusions], [True, True, False])
        self.assertTrue(all(d.exact_error_exceeds_tolerance for d in result.exclusions))
        self.assertFalse(any(d.erroneous_exclusion for d in result.exclusions))
        self.assert_close(result.population_bound_slack, 1/16)
        self.assert_close(result.gap_to_optimum, 15/32-expected.L_kappa_delta)

    def test_identity_is_not_falsely_excluded(self):
        for p in (.5, .8, 31/32, 1.):
            with self.subTest(p=p):
                result = self.evaluate(reference=self.reference(p=p, dynamics="identity"),
                                       exclusion_tolerances=[0., .3])
                self.assertTrue(result.zero_certificate)
                self.assertTrue(result.strict_coverage)
                self.assertEqual(result.gap_to_optimum, 0.)
                self.assertEqual(result.population_bound_slack, 0.)
                self.assertEqual(result.sampling_gap, 0.)
                for decision in result.exclusions:
                    self.assertFalse(decision.excluded)
                    self.assertFalse(decision.exact_error_exceeds_tolerance)
                    self.assertFalse(decision.erroneous_exclusion)

    def test_positive_optimum_with_inconclusive_certificate(self):
        reference = self.reference(p=.8)
        result = self.evaluate(reference=reference,
                              exclusion_tolerances=[0., reference.exact_optimal_rmse, 1.])
        self.assertTrue(result.zero_certificate)
        self.assertTrue(result.strict_coverage)
        self.assert_close(result.gap_to_optimum, .3)
        self.assertEqual(result.reference.population_certificate_limit, 0.)
        self.assertEqual(result.sampling_gap, 0.)
        self.assertFalse(any(d.excluded for d in result.exclusions))
        self.assertEqual([d.exact_error_exceeds_tolerance for d in result.exclusions],
                         [True, False, False])

    def test_missing_classes_and_singletons_are_retained(self):
        for counts in ((0, 0, 31, 9), (17, 0, 0, 0), (0, 1, 0, 0)):
            with self.subTest(counts=counts):
                result = self.evaluate(state_counts=counts)
                statistics = result.count_statistics.statistics
                self.assertEqual(result.count_statistics.state_counts, counts)
                self.assertEqual(statistics.n_pairs, sum(counts))
                self.assertEqual(statistics.class_counts, (sum(counts[:2]), sum(counts[2:])))
                self.assertEqual(statistics.s_hat, 0.)
                self.assertTrue(result.zero_certificate)
                self.assertTrue(result.strict_coverage)

    def test_exclusion_uses_strict_comparison(self):
        lower = self.evaluate().certificate.L_kappa_delta
        tolerances = [math.nextafter(lower, 0.), lower, math.nextafter(lower, math.inf)]
        result = self.evaluate(exclusion_tolerances=tolerances)
        self.assertEqual([d.tolerance for d in result.exclusions], tolerances)
        self.assertEqual([d.excluded for d in result.exclusions], [True, False, False])
        self.assertFalse(any(d.erroneous_exclusion for d in result.exclusions))

    def test_ground_truth_equality_and_erroneous_exclusion(self):
        optimum = self.reference(p=1.).exact_optimal_rmse
        tolerances = [math.nextafter(optimum, 0.), optimum,
                      math.nextafter(optimum, math.inf)]
        result = self.failure_case(exclusion_tolerances=tolerances)
        self.assertTrue(all(d.excluded for d in result.exclusions))
        self.assertEqual([d.exact_error_exceeds_tolerance for d in result.exclusions],
                         [True, False, False])
        self.assertEqual([d.erroneous_exclusion for d in result.exclusions],
                         [False, True, True])

    def test_real_count_fixture_preserves_negative_gaps_and_failure(self):
        result = self.failure_case()
        self.assertGreater(result.certificate.L_kappa_delta, .5)
        self.assertFalse(result.strict_coverage)
        self.assertFalse(result.tolerance_aware_coverage)
        self.assertFalse(result.zero_certificate)
        self.assertLess(result.gap_to_optimum, 0.)
        self.assertLess(result.sampling_gap, 0.)
        self.assertEqual(result.population_bound_slack, 0.)
        self.assertEqual(result.gap_to_optimum, result.sampling_gap)
        self.assertEqual([d.excluded for d in result.exclusions], [True, True, False])
        self.assertEqual([d.erroneous_exclusion for d in result.exclusions],
                         [False, True, False])
        self.assertEqual(result.count_statistics.state_counts, (4096, 4096, 0, 0))

    def test_comparison_tolerance_changes_only_its_named_coverage_check(self):
        strict = self.failure_case(comparison_atol=0.)
        # Deliberately large allowance tests isolation, not a campaign setting.
        allowance = 2*(strict.certificate.L_kappa_delta-strict.reference.exact_optimal_rmse)
        tolerant = self.failure_case(comparison_atol=allowance)
        self.assertFalse(strict.tolerance_aware_coverage)
        self.assertTrue(tolerant.tolerance_aware_coverage)
        self.assertEqual(tolerant, replace(strict, comparison_atol=allowance,
                                          tolerance_aware_coverage=True))
        self.assertFalse(tolerant.strict_coverage)
        self.assertTrue(tolerant.exclusions[1].erroneous_exclusion)

    def test_one_ulp_violation_is_not_hidden(self):
        baseline = self.evaluate()
        optimum = baseline.reference.exact_optimal_rmse
        injected = replace(baseline.certificate, L_kappa_delta=math.nextafter(optimum, math.inf))
        with patch.object(evaluation, "certificate_from_statistics", return_value=injected) as call:
            result = self.evaluate(exclusion_tolerances=[optimum])
        call.assert_called_once()
        self.assertFalse(result.strict_coverage)
        self.assertTrue(result.tolerance_aware_coverage)
        self.assertLess(result.gap_to_optimum, 0.)
        self.assertTrue(result.exclusions[0].excluded)
        self.assertTrue(result.exclusions[0].erroneous_exclusion)
        self.assertEqual(result.certificate, injected)

    def test_exact_coverage_boundary_includes_equality(self):
        baseline = self.evaluate()
        optimum = baseline.reference.exact_optimal_rmse
        injected = replace(baseline.certificate, L_kappa_delta=optimum)
        with patch.object(evaluation, "certificate_from_statistics", return_value=injected):
            result = self.evaluate(exclusion_tolerances=[optimum], comparison_atol=0.)
        self.assertTrue(result.strict_coverage)
        self.assertTrue(result.tolerance_aware_coverage)
        self.assertEqual(result.gap_to_optimum, 0.)
        self.assertFalse(result.exclusions[0].excluded)
        self.assertFalse(result.exclusions[0].erroneous_exclusion)

    def test_signed_gap_decomposition(self):
        gaps = []
        for dynamics in ("identity", "conflicting_successors"):
            for p in (.5, .8, 31/32, 1.):
                for counts in ((1024,)*4, (4096, 4096, 0, 0), (9, 1, 0, 2)):
                    with self.subTest(dynamics=dynamics, p=p, counts=counts):
                        result = self.evaluate(reference=self.reference(p=p, dynamics=dynamics),
                                               state_counts=counts)
                        self.assert_close(result.gap_to_optimum,
                                          result.population_bound_slack+result.sampling_gap)
                        self.assertGreaterEqual(result.population_bound_slack, 0.)
                        self.assertEqual(result.strict_coverage, result.gap_to_optimum >= 0.)
                        gaps.append(result.gap_to_optimum)
        self.assertTrue(any(gap < 0 for gap in gaps))
        self.assertTrue(any(gap > 0 for gap in gaps))

    def test_tolerance_order_and_numeric_normalization(self):
        tolerances = [1, .3, 0, .1]
        result = self.evaluate(exclusion_tolerances=tolerances, comparison_atol=0)
        self.assertEqual([d.tolerance for d in result.exclusions], tolerances)
        self.assertTrue(all(type(d.tolerance) is float for d in result.exclusions))
        self.assertIs(type(result.comparison_atol), float)
        self.assertEqual(result, self.evaluate(exclusion_tolerances=tuple(tolerances),
                                               comparison_atol=0.))

    def test_json_roundtrip_recomputes_success_and_failure_records(self):
        for original in (self.evaluate(), self.failure_case()):
            with self.subTest(strict_coverage=original.strict_coverage):
                saved = json.loads(json.dumps(asdict(original), allow_nan=False))
                reference = four_state_reference(**{
                    name: saved["reference"][name] for name in ("p", "dynamics", "kappa")
                })
                rebuilt = self.evaluate(
                    reference=reference, state_counts=saved["count_statistics"]["state_counts"],
                    delta=saved["certificate"]["delta"],
                    exclusion_tolerances=[d["tolerance"] for d in saved["exclusions"]],
                    comparison_atol=saved["comparison_atol"],
                )
                self.assertEqual(rebuilt, original)

    def test_result_is_immutable_and_inputs_are_not_modified(self):
        counts, tolerances = [1024]*4, [.1, .3, .45]
        result = self.evaluate(state_counts=counts, exclusion_tolerances=tolerances)
        self.assertEqual(counts, [1024]*4)
        self.assertEqual(tolerances, [.1, .3, .45])
        counts[0], tolerances[0] = 0, .9
        self.assertEqual(result.count_statistics.state_counts, (1024,)*4)
        self.assertEqual(result.exclusions[0].tolerance, .1)
        with self.assertRaises(FrozenInstanceError):
            result.strict_coverage = False
        with self.assertRaises(FrozenInstanceError):
            result.exclusions[0].excluded = False
        with self.assertRaises(FrozenInstanceError):
            result.certificate.L_kappa_delta = 0.

    def test_rejects_invalid_tolerance_containers_empty_and_duplicates(self):
        for value in (None, .1, ".1", {.1, .3}, np.array([.1, .3])):
            with self.subTest(value=repr(value)), self.assertRaises(TypeError):
                self.evaluate(exclusion_tolerances=value)
        for value in ([], (), [.1, .1], [0., -0.], [1, 1.]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(exclusion_tolerances=value)

    def test_rejects_invalid_tolerance_values(self):
        for value in (True, False, None, ".1", 1j, np.bool_(True)):
            with self.subTest(value=repr(value)), self.assertRaises(TypeError):
                self.evaluate(exclusion_tolerances=[value])
        for value in (-1e-12, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(exclusion_tolerances=[value])

    def test_rejects_invalid_comparison_tolerance_before_calculation(self):
        cases = [(x, TypeError) for x in (True, None, "0", 1j)]
        cases += [(x, ValueError) for x in (-1e-12, math.nan, math.inf, -math.inf)]
        for value, error in cases:
            with self.subTest(value=value):
                with patch.object(evaluation, "statistics_from_state_counts") as count_call:
                    with self.assertRaises(error):
                        self.evaluate(comparison_atol=value)
                    count_call.assert_not_called()

    def test_upstream_validation_is_preserved(self):
        cases = [({"delta": d}, ValueError) for d in (0., 1., -.1, math.nan, math.inf)]
        cases += [({"delta": d}, TypeError) for d in (True, ".05", None)]
        cases += [({"reference": None}, TypeError),
                  ({"reference": replace(self.reference(), probabilities=(.5, 0., .25, .25))}, ValueError),
                  ({"state_counts": (0, 0, 0, 0)}, ValueError),
                  ({"state_counts": (1., 0, 0, 0)}, TypeError)]
        for changes, error in cases:
            with self.subTest(changes=changes), self.assertRaises(error):
                self.evaluate(**changes)

    def test_component_failures_propagate_without_retry(self):
        failure = RuntimeError("injected count-statistics failure")
        with patch.object(evaluation, "statistics_from_state_counts", side_effect=failure) as count_call, \
             patch.object(evaluation, "certificate_from_statistics") as certificate_call:
            with self.assertRaises(RuntimeError) as caught:
                self.evaluate()
            self.assertIs(caught.exception, failure)
            count_call.assert_called_once()
            certificate_call.assert_not_called()
        failure = RuntimeError("injected certificate failure")
        with patch.object(evaluation, "statistics_from_state_counts",
                          wraps=evaluation.statistics_from_state_counts) as count_call, \
             patch.object(evaluation, "certificate_from_statistics", side_effect=failure) as certificate_call:
            with self.assertRaises(RuntimeError) as caught:
                self.evaluate()
            self.assertIs(caught.exception, failure)
            count_call.assert_called_once()
            certificate_call.assert_called_once()

    def test_no_sampling_during_import_or_evaluation(self):
        expected = self.evaluate()
        with ExitStack() as stack:
            spies = [stack.enter_context(patch.object(
                np.random, name, side_effect=AssertionError("Unexpected random sampling"),
            )) for name in ("SeedSequence", "PCG64", "Generator", "default_rng", "multinomial", "seed")]
            loaded = runpy.run_path(str(Path(evaluation.__file__).resolve()),
                                    run_name="_evaluation_import_check_")
            actual = loaded["evaluate_four_state_counts"](
                reference=self.reference(), state_counts=(1024,)*4,
                delta=.05, exclusion_tolerances=[.1, .3, .45], comparison_atol=1e-12,
            )
            self.assertEqual(asdict(actual), asdict(expected))
            self.assertEqual(self.evaluate(), expected)
            for spy in spies:
                spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
