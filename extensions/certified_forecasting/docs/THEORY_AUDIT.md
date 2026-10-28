# Theory audit for the certified forecasting extension

This is an internal mathematical scope record. The results in `paper/sections/theory.tex`
have complete direct proofs; no literature-priority claim or acceptance claim is made.

## Core checked facts

1. Pair witness: `W = E[(||Y-Y'|| - kappa*d(U,U'))_+^2]/2 <= inf_g R(g)`
   for conditionally independent, identically distributed copies given a common
   variable A and all kappa-Lipschitz decoders. Conditional residual covariance is
   nonnegative after averaging: `E[r^T r' | A] = ||E[r|A]||^2`.
2. The coefficient 1/2 is correct for independent copies (including independent
   episodes at the same time). For arbitrary dependent within-episode pairs the
   same argument only yields coefficient 1/4. Never apply 1/2 to arbitrary pair
   matching without establishing the required conditional law.
3. Conditional-pair witness dominates the squared centered population bound only
   for the SAME conditioning variable and law. Pairing episodes at t conditions on
   t; it does not prove domination over the old reference-class c(X) certificate.
4. Deterministic two-atom law: exact oracle MSE is
   `p(1-p)(||y1-y0|| - kappa*||u1-u0||)_+^2`, attained by affine interpolation of
   optimally shrunk output separation; hence pair witness is sharp.
5. Upper/lower episode bounds use independent episode counts n and floor(n/2),
   not total windows. Each within-episode or within-pair average is one bounded
   random variable. Dependence between upper and lower statistics is harmless.
6. Empirical Bernstein one-sided radius is
   `sqrt(2*s2*log(2/alpha)/N) + 7*range*log(2/alpha)/(3*(N-1))`, with unbiased s2.
   Upper-loss range is B = diameter(target set)^2; pair-score range is B/2.
   For J models and H horizons, alpha=delta/(2*J*H) yields log(4*J*H/delta).
7. For a complete sandwich, the fitted forecaster must belong to the same
   Lipschitz comparator class. An unpenalized intercept does not alter Lipschitz
   constants. Output projection is nonexpansive. Block-sum Frobenius norm controls
   Lipschitz sensitivity in the block maximum metric.
8. Candidate construction and all reported budgets/horizons/families must be fixed
   before calibration. A fitted norm is permissible when calculated solely from
   development data. Comparing floors at different fitted norms changes classes.
9. Raw-target and prediction bounding assumptions are explicit. Clipping only
   forecasts does not bound unbounded targets. Gaussian noise is unbounded.
10. No stationarity is assumed inside episodes, but independent episodes must have
    the same distribution. Same common origin set for each compared candidate.

## Refinement

- Retaining old blocks gives class nesting. Independently refitting/replacing all
  existing blocks does not establish nestedness.
- Sum of block Frobenius norms and maximum block input distance give exact
  duplicate invariance. An ordinary concatenated Frobenius ball does not:
  duplicating simplex endpoints can turn positive approximation error into zero
  by lowering the norm cost without adding information.
- The necessary metric-change theorem concerns an old lower confidence bound on
  the PAIR WITNESS, not any lower confidence bound on oracle risk.
- The old pair event gives a deterministic necessary condition valid for every
  new map, even adaptively proposed maps. It does not certify a new map's risk.
- Pair obstruction decreasing is not evidence of actual improvement. Independent
  future noise can cause an irreducible obstruction; harmless or redundant
  refinement can also change geometry without improving useful information.
- Population infimum risk non-increase does not imply finite-sample fitted risk
  non-increase. Report actual fits and paired-loss uncertainty.
- Residual-correlation proposition has exact un-clipped quadratic expansion.
  It requires budget slack for the stated strict improvement and needs the old
  output already in the convex output set before claiming that separate output
  clipping preserves relative improvement.

## Confidence selection and sequential reuse

- Selecting cheapest candidate among a fixed jointly certified family is valid.
- It is only cheapest among candidates that pass the test, not all truly feasible
  candidates. Failure to certify does not prove excessive risk.
- If using both Hoeffding and empirical Bernstein then choosing the tighter,
  allocate confidence over both families, or prespecify the family.
- Paired-loss difference has range length 2B, not B; use this in EB correction.
- Genuine sequential proposal based on past calibration needs independent fresh
  episode blocks and alpha spending (or another proved adaptive procedure).
- The sequential gate is a separately stated option; do not say implemented if
  experiments use a frozen finite family only.

## Interpretation

- A risk confidence bound is neither a pointwise future-error bound nor a
  conditional prediction interval.
- The direct pair statistic avoids subtracting square roots of two empirical
  variances; it does not establish a new minimax rate. RMSE square roots can still
  behave slowly near zero.
- Exact two-atom tightness does not imply sharpness on all distributions.
- A small upper/lower RMSE gap bounds suboptimality; a large gap cannot be
  attributed specifically to optimization error.
- Scope is direct finite-horizon prediction on raw targets. No exact Koopman
  closure, latent Markov property, single-trajectory mixing bound, arbitrary
  nonstationary future guarantee, or neural dictionary priority is claimed.

## Sources and novelty positioning

The empirical Bernstein formula is from Maurer and Pontil (2009), arXiv:0907.3740.
Finite-family post-selection risk testing is established (e.g. Learn then Test).
Block norm / multiple kernel constructions and quadratic residual-risk geometry
are established tools. The manuscript should attribute those ingredients and
present the pair witness, its consequences, and their use in geometry-aware
forecast selection as the specific extension being studied. A targeted primary
search did not locate this exact population witness as a named previous theorem,
but that is not evidence of exhaustive novelty.
