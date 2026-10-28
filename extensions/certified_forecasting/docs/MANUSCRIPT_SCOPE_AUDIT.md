# Manuscript scope and implementation audit

Reviewed text-only on 28 October 2026. No generated manuscript artifacts were
compiled or changed during this audit. Files examined: `main.tex`, introduction,
related work, method, theory, experiments, real data, discussion, protocols,
optimization, exact-risk details, and budget-sensitivity sections. Implementation
checks included feature maps, constrained decoders, certificates, forecasting
selection, and the known-risk runner.

## Findings requiring wording alignment

1. **Artifact distribution:** the initial main-file provenance paragraph said that
   the per-episode archives and manuscript generators were all distributed in the
   accompanying repository directory. The intended release separates the public
   code/result JSON from the larger evidence archive and the author-review
   manuscript package. The wording must match that actual split. The protocol
   appendix already accommodates separately archived score files correctly.
2. **Computational comparison:** the initial related-work ending described
   comparisons with a fixed computational budget. The implemented study fixes
   configurations and transparently reports feature counts, but does not establish
   equal runtime/compute budgets across all methods. Replace that phrase with
   transparent representation-cost and computational accounting.
3. **Prior-paper credit:** the predecessor already includes a fitted-risk upper
   certificate (Remark 5.14) and independent-trajectory extensions (Appendix H).
   Explicitly credit these to avoid implying that upper-risk calibration or the
   episode aggregation idea is newly introduced here. The pairwise witness,
   retained-class refinement analysis, and implemented selection framework are
   the specific extension under study.

These findings were sent to the main manuscript owner; this audit does not itself
edit the manuscript sections that owner is updating.

## Mathematical and implementation alignment checked

- Main method correctly states that the target is raw future observations and
  that the implementation is direct multihorizon prediction, with iterated
  Koopman-style feature models kept as separate baselines.
- The retained feature vector is explicitly a product of simplices. Coordinate
  functions are retained exactly; no unsupported nested-partition claim appears.
- The block-sum Frobenius constraint and block-maximum input metric match the
  implementation. Matrix orientation is explicitly reconciled between row-wise
  training notation and column-vector theorem notation.
- Duplication invariance is limited to feasible classes, their population
  optima, and the pair witness. The method and experiments correctly disclose
  that quadratic ridge regularization can change under replication, and that
  the duplicate candidate uses a different zero-ridge fitting convention.
- The main experiment does not claim that every new block improves a learned
  predictor. It retains inconclusive adjacent comparisons and explicitly
  separates fixed-family experiments from optional sequential refinement.
- Common forecast origins across history lengths are correctly stated and match
  the runner. Independent-unit counts are episodes/pairs, not overlapping rows.
- Main-family counts agree: 40 candidates, three horizons, 20 learning runs,
  800 fits, 4800 selected one-sided risk endpoints, 480 KAHM candidate/horizon
  lower statements, and 360 adjacent KAHM comparisons.
- Failure allocation agrees with the code: 0.04 across 240 main risk tails and
  0.01 across 18 paired-improvement tails in each complete run.
- Known-risk exact oracle and fitted risks match the implementation. Its three
  directions share 0.05 within each trial; no global confidence across all
  Monte Carlo trials is claimed. The one observed violation is retained.
- Bounded simulation outputs follow from the stated definitions. No Gaussian
  innovation is silently treated as bounded, and the range assumption is not
  estimated from observed extrema.
- Numerical optimization gaps are correctly separated from statistical risk
  certificates and are not represented as formal floating-point verification.
- The code's candidate-cost rule is feature count, then history length, then
  validation MSE, matching the method. It is not claimed to measure runtime.
- Main empirical conclusions are task dependent and acknowledge stronger AR and
  polynomial results, including the real-data results.

## Real-data scope

The real-data section is appropriately conservative: it uses original raw
coordinates, documents inferred writer IDs and the reduced eligible-writer
count, states that iid writer sampling is unverified, reports the very loose
conditional Hoeffding bounds, and does not count individual strokes as
independent writers. Seed variation on the same cohort is explicitly not a
test-population confidence interval. The coordinate-pooled RMSE calculation and
its 720-statement confidence allocation agree with the runner.

The length restriction defines a retrospective eligible-stroke population.
Eligibility of a newly started stroke is not generally known online before it
reaches the required length. The existing restricted-population language is
therefore important and should not be removed or converted into an unrestricted
online deployment claim. The writer identifier remains a documented inference,
not a source field whose meaning has independently been established.

## Follow-up sensitivity scope

The budget-sensitivity section correctly labels the campaign as a follow-up
designed after observing the main results, uses new development/calibration/test
streams, holds data and feature dictionaries fixed across budgets within each
replicate, and allocates 0.05/3 to each of the three budget families. No global
confidence across all system/seed repetitions is asserted. Its conclusions are
limited to the tested budgets and fixed dictionary family.

## Overall conclusion

No concrete mathematical contradiction was found between the assembled method,
the theorems, and the implementation in the reviewed files. The principal
remaining changes are accurate distribution/provenance wording and explicit
credit for inherited upper-risk/episode ingredients. This is a targeted audit,
not exhaustive formal verification or a claim of publication acceptance.
