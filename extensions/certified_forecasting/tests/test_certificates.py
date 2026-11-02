"""Mathematical invariants and adversarial misuse checks, using unittest."""
import unittest
import numpy as np
from certified_forecasting.certificates import mean_interval, pair_witness_scores, paired_improvement


class CertificateTests(unittest.TestCase):
    def test_constant_representation_exact_population_witness(self):
        # Enumerate all ordered pairs of a two-point target distribution.
        targets = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], float).reshape(8, 1, 1)
        f = np.zeros((8, 1, 2))
        w = pair_witness_scores(f, targets, 10.0)
        self.assertAlmostEqual(w.mean(), 1.0)

    def test_pairwise_bound_for_arbitrary_predictor_exhaustive(self):
        # Exactly enumerate the product distribution, not a Monte Carlo test.
        f0 = np.array([[-1.0], [0.0], [2.0]])
        y0 = np.array([-1.0, 0.7, 1.0])
        pairs = [(i,j) for i in range(3) for j in range(3)]
        f = np.array([f0[i] for pair in pairs for i in pair])[:, None, :]
        y = np.array([y0[i] for pair in pairs for i in pair])[:, None, None]
        for slope in [-1.0, -0.3, 0.0, 0.8]:
            pred = np.clip(slope * f0[:, 0] + 0.2, -1, 1)
            risk = np.mean((y0 - pred) ** 2)
            witness = pair_witness_scores(f, y, abs(slope)).mean()
            self.assertLessEqual(witness, risk + 1e-12)

    def test_nested_distance_witness_cannot_increase(self):
        rng = np.random.default_rng(8)
        f = rng.normal(size=(20, 5, 2))
        extra = rng.normal(size=(20, 5, 3))
        y = rng.uniform(-1, 1, size=(20, 5, 2))
        old = pair_witness_scores(f, y, .2)
        new = pair_witness_scores(np.concatenate((f, extra), axis=-1), y, .2)
        self.assertTrue(np.all(new <= old + 1e-14))

    def test_episode_count_not_window_count(self):
        f = np.zeros((12, 50, 1))
        y = np.zeros((12, 50, 3))
        self.assertEqual(pair_witness_scores(f,y,1).shape, (6,3))

    def test_reject_unbounded_scores(self):
        with self.assertRaises(ValueError):
            mean_interval([0,1.1], 1.0, .05)
        with self.assertRaises(ValueError):
            mean_interval([0,float('nan')], 1.0, .05)
        with self.assertRaises(ValueError):
            mean_interval([0], 1.0, .05)

    def test_no_false_positive_for_identical_models(self):
        x = np.linspace(0,1,100)
        r = paired_improvement(x,x,1,.05)
        self.assertLessEqual(r['improvement_lower'],0)


if __name__ == '__main__':
    unittest.main()
