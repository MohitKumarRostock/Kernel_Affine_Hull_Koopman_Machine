"""Deterministic implementation checks; not a statistical coverage study.

Reference: arXiv:2609.32652v1, Eq. (8) and the balanced example on p. 28.
The Decimal reference uses 80-digit arithmetic and a rearranged expression
for F_delta. It shares no helper functions with the implementation.
"""

from decimal import Decimal, localcontext
import math
import unittest

from kahkm_certificates import certificate_from_statistics


def decimal_reference(*, f_hat, s_hat, n_pairs, kappa, delta):
    """Evaluate the same mathematical bound using separate arithmetic."""
    with localcontext() as ctx:
        ctx.prec = 80
        # Preserve the exact binary inputs supplied to the implementation.
        f, s, k, d = (
            Decimal.from_float(float(x))
            for x in (f_hat, s_hat, kappa, delta)
        )
        zero, one, two = Decimal(0), Decimal(1), Decimal(2)
        r = (two / d).ln() / Decimal(n_pairs)

        # The positive root of F - f = sqrt(2*r*F), before clipping.
        F = min(one, ((r / two).sqrt() + (f + r / two).sqrt()) ** 2)
        V = max(zero, s - (two * r).sqrt())
        L = max(zero, V.sqrt() - k * (two * F).sqrt())

        return dict(zip(
            ("r_delta", "F_delta", "V_delta", "L_kappa_delta"),
            map(float, (r, F, V, L)),
        ))


class CertificateTests(unittest.TestCase):
    def parameters(self, **changes):
        values = dict(
            f_hat=1 / 1024,
            s_hat=225 / 1024,
            n_pairs=4096,
            kappa=math.sqrt(2),
            delta=0.05,
        )
        values.update(changes)
        return values

    def evaluate(self, **changes):
        return certificate_from_statistics(**self.parameters(**changes))

    def test_high_precision_reference(self):
        cases = [
            {},
            dict(f_hat=0.0, s_hat=0.25, n_pairs=100000),
            dict(f_hat=0.02, s_hat=0.4, kappa=0.75, delta=1e-12),
            dict(f_hat=1.0, s_hat=0.5, kappa=0.1, delta=0.95),
            dict(n_pairs=3, s_hat=0.0),
            dict(n_pairs=1000000, delta=5e-324),
        ]
        for changes in cases:
            params = self.parameters(**changes)
            actual = certificate_from_statistics(**params)
            for name, expected in decimal_reference(**params).items():
                with self.subTest(parameters=params, quantity=name):
                    self.assertTrue(math.isclose(
                        getattr(actual, name), expected,
                        rel_tol=5e-13, abs_tol=5e-15,
                    ), msg=f"{name}: {getattr(actual, name)} != {expected}")

    def test_balanced_example_excludes_point_three(self):
        # Specified counts, not Monte Carlo evidence of 95% coverage.
        result = self.evaluate()
        self.assertGreater(result.L_kappa_delta, 0.30)
        self.assertLess(result.L_kappa_delta, 13 / 32)
        self.assertLess(result.L_kappa_delta, 15 / 32)

    def test_zero_variance_is_inconclusive(self):
        result = self.evaluate(s_hat=0.0)
        self.assertEqual(result.V_delta, 0.0)
        self.assertEqual(result.L_kappa_delta, 0.0)

    def test_small_sample_is_inconclusive(self):
        result = self.evaluate(n_pairs=3, s_hat=0.5, kappa=0.0)
        self.assertEqual(result.V_delta, 0.0)
        self.assertEqual(result.L_kappa_delta, 0.0)

    def test_zero_assignment_loss_still_has_uncertainty(self):
        result = self.evaluate(f_hat=0.0)
        self.assertGreater(result.F_delta, 0.0)
        self.assertAlmostEqual(result.F_delta, 2 * result.r_delta, places=15)

    def test_assignment_upper_bound_is_capped(self):
        self.assertEqual(self.evaluate(f_hat=1.0).F_delta, 1.0)

    def test_zero_norm_budget(self):
        result = self.evaluate(kappa=0.0)
        self.assertEqual(result.L_kappa_delta, math.sqrt(result.V_delta))

    def test_large_norm_budget_truncates_at_zero(self):
        self.assertEqual(self.evaluate(kappa=100.0).L_kappa_delta, 0.0)

    def test_fixed_statistics_monotonicity(self):
        # Algebraic checks only: sample statistics normally change with M.
        cases = (
            ("n_pairs", [128, 1024, 4096, 65536], True),
            ("delta", [0.001, 0.01, 0.05, 0.2], True),
            ("s_hat", [0.0, 0.05, 0.25, 0.5], True),
            ("kappa", [0.0, 0.5, 1.0, 2.0], False),
            ("f_hat", [0.0, 0.001, 0.01, 0.1], False),
        )
        for name, values, increasing in cases:
            bounds = [
                self.evaluate(**{name: v}).L_kappa_delta
                for v in values
            ]
            with self.subTest(parameter=name):
                self.assertEqual(
                    bounds, sorted(bounds, reverse=not increasing)
                )

    def test_large_sample_approaches_known_limit(self):
        # Formula evaluation only; no sample of this size is generated.
        bound = self.evaluate(n_pairs=10**16).L_kappa_delta
        self.assertLess(bound, 13 / 32)
        self.assertAlmostEqual(bound, 13 / 32, delta=1e-6)

    def test_inputs_are_recorded(self):
        result = self.evaluate()
        for name, value in self.parameters().items():
            with self.subTest(parameter=name):
                self.assertEqual(getattr(result, name), value)

    def test_rejects_invalid_pair_counts(self):
        for value in (True, False, 4096.0, "4096", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.evaluate(n_pairs=value)
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(n_pairs=value)

    def test_rejects_nonfinite_scalars(self):
        for name in ("f_hat", "s_hat", "kappa", "delta"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(ValueError):
                        self.evaluate(**{name: value})

    def test_rejects_invalid_scalar_types(self):
        for name in ("f_hat", "s_hat", "kappa", "delta"):
            for value in (True, False, "0.05", None, 1j):
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(TypeError):
                        self.evaluate(**{name: value})

    def test_rejects_out_of_range_scalars(self):
        cases = (
            ("f_hat", [-1e-12, 1.0 + 1e-12]),
            ("s_hat", [-1e-12, 1.0 + 1e-12]),
            ("kappa", [-1e-12]),
            ("delta", [-0.1, 0.0, 1.0, 1.1]),
        )
        for name, values in cases:
            for value in values:
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(ValueError):
                        self.evaluate(**{name: value})


if __name__ == "__main__":
    unittest.main()
