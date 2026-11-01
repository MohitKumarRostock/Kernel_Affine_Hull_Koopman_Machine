# Computational reproduction of cross-fitted residual corrections

## Scope and provenance

This execution reconstructs the declared simulation cohort of the cross-fitted
residual study from its published frozen implementation. The original amended
source is commit `f36e7c797f37d7d073dae42603b3770543d9fb43` in
`MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine`, under
`extensions/crossfit_residual_v2`. The completed original execution's development
arrays, fitted states, terminal forecasts, analysis tables, and stage seals were
not recovered. This reproduction cannot authenticate those unavailable records.
Prior assistant reports of aggregate results and an audit failure were known
before this reproduction was prepared.
The actual reproduction publication commit is recorded separately after its new
source freeze and publication; a commit cannot be self-bound in its own frozen
contents. The legacy `source.publication_commit` configuration field is retained
unchanged and does not identify this later reproduction publication.

The model implementation, numerical solver, pipeline, scientific analyzer, and
manufactured cross-fit oracle are copied byte-for-byte from the published source.
The full original scientific protocol is retained without alteration in
[audit/original_protocol.md](audit/original_protocol.md). Its historical execution
statements describe that earlier study. The current execution follows the scope
and provenance requirements in this document. The original numerical amendment
is preserved in [audit/original_numerical_amendment.md](audit/original_numerical_amendment.md).

## Scientific procedure

All scientific configuration values remain identical to the amended study.
The only top-level configuration changes remove `numerical_amendment`, whose
historical exact-byte reuse conditions cannot be satisfied without the original
records, and add `computational_reproduction`, which states the present execution
scope and binds the published source identities. The complete removed amendment
metadata remains nested within this provenance record. No method, model grid,
selection rule, loss, margin, seed, or inference rule changes.

The master seed is `2026110173`. The same declared generation, unit-noise,
retained-row, and fit streams specify 24 repetitions of four laws at two noise
levels, yielding 192 task suites. A/B/C/D contain 64/32/32/128 episodes; history
width is 12 and horizons are 1, 4, and 12. The independent units within the
simulation design remain the 24 repetitions. Replaying those same streams adds
zero independent repetitions, and results must not be pooled with the original
study as if they came from a second sample.

The new execution regenerates all A/B/C arrays, recomputes every fitted state,
and subsequently regenerates D from the declared streams. It does not reuse or
claim to restore original fitted states. Numerical or byte-level equality to the
unavailable original data and forecasts is not asserted. Any numerical comparison
with an earlier reported summary is described as computational consistency with
that summary, rather than authentication of the earlier execution history.

Four whole-episode folds create common-predictor out-of-fold residual targets.
The deployed common predictor uses all A; B selects all prescribed model settings
and the shared residual-mean bias; C selects correction multipliers and the
expanded non-KAHM policy. The in-sample residual ablation, independent RBF
correction, matched RBF diagnostic, half-training common diagnostics, and inherited
complete-procedure competitors are retained. Every task produces the original
42 forecast arrays. No selection is made from terminal D outcomes.

## Numerical and statistical identities

The copied `code/pipeline.py` installs `code/numerical_solver.py` unconditionally
in every fitting worker. Removing historical amendment metadata therefore does
not remove or alter the checked ridge recovery rule. The original ridge path
runs first; its `LinAlgError` alone triggers the same checked SciPy `gesvd`
fallback, with identical objectives, penalties, validity gates, and event logs.
All original and inherited numerical source hashes remain mandatory.

The byte-identical analyzer implements the four original primary comparisons:
3% superiority of calibrated cross-fit KAHM over the full-A common-plus-bias
baseline, independently selected cross-fit RBF, and expanded non-KAHM policy on
the nonlinear panel; and 5% noninferiority to raw ridge on the linear panel.
Each decision uses the original repetition-level margin gap and the simultaneous
one-sided t bound with family alpha 0.05 over four claims. These retained
calculations describe this reused simulation cohort; the reproduction is not
an independent confirmatory experiment or a new prospective preregistration.
The copied [analysis protocol](analysis_code/ANALYSIS_PROTOCOL.md) supplies the
unchanged formulas, weighting, tie rules, and descriptive diagnostics.

## Current execution gates and audit

A new source/configuration freeze must precede any regenerated A/B/C data.
All 192 fresh fits and calibration records must be complete and sealed before
regenerating D. All target-free D forecast arrays must be complete and sealed
before target release and scoring by the unchanged analyzer. These phase gates
record only the new execution; they do not reconstruct historical timings or
provide access-control isolation. Missing, failed, duplicate, nonfinite, or
shape-inconsistent tasks stop the corresponding stage and remain recorded.

The independent reproduction audit must verify scientific configuration
equivalence, all executable source identities, all current data and stage seals,
all required task and forecast rosters, selected native bases and coefficients,
B/C decisions, and raw-loss-to-table joins. Its provenance gate must explicitly
retain `original_completed_run_authenticated=false`. No unavailable historical
artifact may be replaced by an invented manifest, timestamp, state, or receipt.

A separate correction to the historical verifier handles the legitimate
one-reference-point native rank-zero case before covariance division. That
correction affects verification only. Its source and manufactured regression
receipt must be identified separately from the frozen scientific model. A
successful audit of this reproduction establishes the declared checks on its
newly retained records; it cannot certify the unavailable original execution.
