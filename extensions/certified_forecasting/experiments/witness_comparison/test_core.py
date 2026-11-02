"""Substantive checks of class identity, dual bounds and confidence formulas."""
import math
import unittest
import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom
from core import (laws, grouped_objective, qp_values, solve_lipschitz, population,
                  affine_minimum, erm_kl_lower, centered_from_counts, mass_minimum)

class WitnessChecks(unittest.TestCase):
    def test_analytic_oracles_and_dual_certificates(self):
        expected=(1.,(1-.6/math.sqrt(2))**2,2/9,.54,19/75,.25,.025,0.)
        for law,truth in zip(laws(),expected):
            with self.subTest(law=law.name):
                result=solve_lipschitz(law,np.asarray(law.probabilities))
                self.assertLessEqual(result['lower'],truth+1e-13)
                self.assertGreaterEqual(result['upper'],truth-1e-13)
                self.assertLessEqual(result['gap'],1e-8)
                self.assertLessEqual(result['feasibility_residual'],1e-12)
                self.assertTrue(all(x>=0 for x in result['dual_multipliers']))

    def test_rare_pairing_and_affine_class_mismatch(self):
        rare=population(laws()[6])
        self.assertAlmostEqual(rare['pair_witness'],.0025,places=14)
        self.assertAlmostEqual(rare['stratified_witness'],.025,places=14)
        mismatch=population(laws()[7])
        self.assertLess(mismatch['oracle']['upper'],1e-9)
        self.assertAlmostEqual(mismatch['affine_minimum'],2/9,places=12)
        self.assertAlmostEqual(mismatch['pair_witness'],0.,places=14)

    def test_missing_atoms_and_extension(self):
        law=laws()[3]
        # Observing only two neighboring opposite targets: distance 1/2.
        result=solve_lipschitz(law,[.5,.5,0,0,0])
        self.assertLessEqual(result['lower'],.5625)
        self.assertGreaterEqual(result['upper'],.5625)
        self.assertEqual(result['support'],[-1.,-.5])
        # A single observed atom has zero empirical minimum without inventing
        # information about unsampled representation values.
        one=solve_lipschitz(law,[0,0,1,0,0])
        self.assertEqual(one['lower'],0.)
        self.assertLess(one['upper'],1e-9)

    def test_arbitrary_nonnegative_dual_is_lower(self):
        law=laws()[2]
        _,w,m,c,a,b=grouped_objective(law,[.2,.3,.5])
        optimum=solve_lipschitz(law,[.2,.3,.5])
        for scale in (0.,.1,1.,10.):
            lam=scale*np.arange(1,len(b)+1)
            lower,_,_,_,_=qp_values(w,m,c,a,b,np.zeros(len(w)),lam)
            self.assertLessEqual(lower,optimum['upper'])

    def test_exact_binomial_coverage_for_erm_and_centered(self):
        law=laws()[0]
        for n in (4,8,16,64):
            for p in (.02,.2,.5,.8,.98):
                truth=4*p*(1-p)
                failure_erm=failure_centered=0.
                for successes in range(n+1):
                    empirical=4*(successes/n)*(1-successes/n)
                    probability=binom.pmf(successes,n,p)
                    failure_erm+=probability*(erm_kl_lower(empirical,n)>truth+1e-12)
                    value,_=centered_from_counts(law,[n-successes,successes])
                    failure_centered+=probability*(value>truth+1e-12)
                self.assertLessEqual(failure_erm,.05+1e-12)
                self.assertLessEqual(failure_centered,.05+1e-12)

    def test_mass_minimizer_matches_independent_linear_program(self):
        for values,counts in (([0.,.2,.8],[10,20,70]),([.3,.1],[95,5]),
                              ([.4,.4,.1],[0,0,100])):
            n=sum(counts)
            value,masses,radius=mass_minimum(values,counts,n,.05)
            frequency=np.asarray(counts)/n
            bounds=list(zip(np.maximum(0,frequency-radius),np.minimum(1,frequency+radius)))
            reference=linprog(values,A_eq=[np.ones(len(values))],b_eq=[1.],bounds=bounds,method='highs')
            self.assertTrue(reference.success)
            self.assertAlmostEqual(value,max(0.,reference.fun-1e-12),places=11)
            self.assertAlmostEqual(sum(masses),1.,places=12)

    def test_prior_structural_correction_is_not_a_new_sd_estimate(self):
        n=4096; eta=.05/3
        lower_expectation=1-1/n-4*math.sqrt(math.log(2/eta)/(2*n))
        self.assertAlmostEqual(lower_expectation,.9030575149728274,places=13)

if __name__=='__main__':
    unittest.main()
