# Residual learning, donor predictors, and the final forecast

## Status and purpose

These notes give elementary squared-loss identities used to interpret the
target-geometry extension. They are consequences of conditional-expectation
projection, not a claim of a new general statistical theorem. They do not prove
that KAHM, cross-fitting, supervised metric learning, or an available zero
correction improves prediction.

The extension learns geometric corrections from either in-sample residuals or
out-of-fold residuals. The latter are produced by earlier, separately fitted
baseline predictors. Those residuals can be honest about what each donor
predictor knew while still differing from the future errors of the final
baseline. The identities below separate that issue from the existence of useful
residual signal and from the error of fitting a correction.

## Common notation and assumptions

Condition on the fitted development state. Let `Q` be one specified evaluation
law for a fresh pair `(X,Y)`, with `Y` a vector of `H` forecast targets. Let
`f0(X)` be the deployed baseline and `fd(X)` a fixed donor baseline. Let
`Z = Gamma(X)` be the frozen feature representation. All these functions may
have been learned from development data, but they are fixed in the expectations
below. Assume finite second moments for the targets and forecasts.

For a fixed positive-definite horizon-weight matrix `W`, define

\[
\langle a,b\rangle_W=a^\top Wb,\qquad
\|a\|_W^2=a^\top Wa,\qquad
R_W(f)=\mathbb E_Q\|Y-f(X)\|_W^2.
\]

The study's equal-horizon mean squared error corresponds to `W = I/H`; the
scalar case has `H=1` and `W=1`. Fixed positive horizon weights work identically.
This is a population calculation under a **common evaluation law**, not a claim
that chronological folds have identical distributions. It makes no
independence assertion about overlapping origins. It supplies no confidence
interval from the training residuals themselves.

## 1. Gain from any fixed fitted correction

Set

\[
r_0=Y-f_0(X),\qquad m_0(Z)=\mathbb E_Q[r_0\mid Z].
\]

For any fixed square-integrable fitted correction `q(Z)`,

\[
\boxed{
R_W(f_0)-R_W(f_0+q\circ\Gamma)
=\mathbb E_Q\|m_0(Z)\|_W^2
-\mathbb E_Q\|q(Z)-m_0(Z)\|_W^2.}
\tag{T1}
\]

**Proof.** Write `r0 = m0(Z) + epsilon`, where
`E_Q[epsilon | Z] = 0`. Conditional orthogonality removes the cross term in
`E_Q ||epsilon + m0(Z) - q(Z)||_W^2`. Subtract the resulting risk from the
analogous expansion with `q=0`.

Consequently the oracle residual signal is `E_Q ||m0||_W^2`, and a fitted
correction improves expected squared loss precisely when its error in learning
that signal is smaller than the signal itself. The statement is relative to
the specified baseline, representation, and evaluation law.

An equivalent identity is

\[
R_W(f_0)-R_W(f_0+q\circ\Gamma)
=2\mathbb E_Q\langle Y-f_0(X),q(Z)\rangle_W
-\mathbb E_Q\|q(Z)\|_W^2.
\tag{T2}
\]

Its empirical analogue can be verified exactly from saved predictions. That
verification describes realized prediction errors; it does not estimate the
oracle conditional mean or establish population improvement.

## 2. Transporting a donor correction to a different baseline

Define the donor's oracle correction and the conditional donor discrepancy by

\[
q_d(Z)=\mathbb E_Q[Y-f_d(X)\mid Z],\qquad
b_d(Z)=\mathbb E_Q[f_0(X)-f_d(X)\mid Z].
\]

Linearity of conditional expectation gives

\[
q_d(Z)=m_0(Z)+b_d(Z).
\tag{T3}
\]

If this donor-optimal correction is attached to the deployed baseline `f0`,
then (T1) gives the exact identity

\[
\boxed{
R_W(f_0)-R_W(f_0+q_d\circ\Gamma)
=\mathbb E_Q\|m_0(Z)\|_W^2
-\mathbb E_Q\|b_d(Z)\|_W^2.}
\tag{T4}
\]

Thus even an oracle correction for a donor predictor need not improve a
different final predictor. Donor-to-final mismatch can consume the available
residual signal. Honest prediction of held-out training observations by the
donor does not, by itself, remove this mismatch.

For a fitted donor correction `qhat = qd + a`, the exact extension is

\[
R_W(f_0)-R_W(f_0+\widehat q\circ\Gamma)
=\mathbb E_Q\|m_0\|_W^2
-\mathbb E_Q\|b_d+a\|_W^2.
\tag{T5}
\]

The final term contains the cross term `2 E_Q <bd,a>_W`. It must not be
replaced by the sum of two nonnegative penalties without an additional
orthogonality argument. Approximation error against the donor target may
reinforce or partially cancel donor-to-final mismatch.

Conditional Jensen's inequality supplies a useful bound:

\[
\mathbb E_Q\|b_d(Z)\|_W^2
\leq \mathbb E_Q\|f_0(X)-f_d(X)\|_W^2.
\tag{T6}
\]

The right side measures total disagreement between the two forecasts. It is
not equal in general to the conditional mismatch penalty on the left.

## 3. Several out-of-fold donors

Let the fixed donors be `f1,...,fK` and let deterministic mixture weights
`pi_k >= 0` sum to one. Under the same evaluation law `Q`, define

\[
\overline f(X)=\sum_{k=1}^K\pi_k f_k(X).
\]

The population correction minimizing

\[
\sum_{k=1}^K\pi_k\,
\mathbb E_Q\|Y-f_k(X)-q(Z)\|_W^2
\]

is `qbar(Z) = E_Q[Y-fbar(X) | Z]`. To see this, expand each donor around
`fbar`; the weighted donor deviations sum to zero, leaving an additive
donor-variance term that does not depend on `q`. Equations (T3)--(T6) then apply
with `fd=fbar`.

This result does **not** identify the empirical out-of-fold fit with that
population optimizer. Chronological folds may have different covariate or
target laws, different sample sizes, and dependent observations. Their group
frequencies may also correspond to different weights. When the donor index
depends on the observation, one must keep that joint law explicitly rather
than substituting a fixed-weight mean predictor. Distribution shift introduces
additional changes in the conditional residual target; the common-law
calculation does not control those changes.

## 4. What the extension can measure

The following are descriptive quantities available without fitting a new
representation after test outcomes have been inspected:

1. The held-out gain and its alignment/energy decomposition in (T2), computed
   from the final baseline and final corrected predictions on the same rows.
2. The RMS discrepancy between an out-of-fold donor prediction and the final
   baseline prediction on that donor's held-out training rows. The final base
   has seen those rows, so this is a development diagnostic, not an unbiased
   estimate of future prediction risk.
3. If fixed donors are additionally evaluated on a fresh common sample, their
   RMS disagreement with the final baseline on that sample. This estimates
   forecast disagreement under that sample's law; it still does not directly
   estimate `E_Q ||bd(Z)||_W^2` or `E_Q ||m0(Z)||_W^2`.

Report each quantity with its sample, units, aggregation, and scaling. Do not
label observed RMS disagreement an estimate of oracle residual structure.
Neither its presence nor absence establishes that cross-fitting caused a
forecast gain. The primary evidence remains the untouched evaluation of the
complete trained-and-selected procedure.

## 5. Scope for supervised KAHM geometry

The new metric is learned through a leave-group-out Gaussian kernel regression
proxy on development residuals. It is then used by matched KAHM, affine,
locality, and RBF geometric controls. This is supervised metric learning with
a shared proxy objective; it is not end-to-end optimization of the KAHM
forecast loss. The proxy can improve while the final geometric decoder does
not. The experiment measures that transfer rather than assuming it.

After fitting, a deterministic positive diagonal transformation changes the
geometry seen by a finite learner but adds no information to a history already
retained in full. Likewise, if `f0(X)=E_Q[Y | X]`, then
`E_Q[Y-f0(X) | Gamma(X)]=0`: no deterministic-history residual correction can
reduce its population squared-error risk.

Finally, including zero correction preserves an available forecast. A finite
validation procedure can still select a harmful correction. Supervised
geometry, residual learning, model selection, and any operational fallback
must be evaluated together. Existing class-risk certificates remain separate
results with their own fixed-class and calibration assumptions.

## Sources and relation to prior work

- The original revised prediction manuscript states the conditional-mean
  residual oracle and distinguishes information retention from finite-decoder
  usefulness. These notes make the donor-to-final distinction explicit.
- Weinberger and Tesauro (2007), *Metric Learning for Kernel Regression*,
  AISTATS, PMLR 2:612--619, learns distances by a regression objective. This is
  relevant prior art for the shared supervised metric proxy, not evidence that
  the new KAHM extension will improve forecasting.
  <https://proceedings.mlr.press/v2/weinberger07a.html>
- Bergmeir, Hyndman, and Koo (2018), *A note on the validity of cross-validation
  for evaluating autoregressive time series prediction*, Computational
  Statistics & Data Analysis 120:70--83, DOI 10.1016/j.csda.2017.11.003, gives
  conditions under which ordinary K-fold validation is valid for autoregressive
  prediction. The extension chooses forward residual folds without claiming
  that every random-fold time-series procedure is invalid.
  <https://robjhyndman.com/publications/cv-time-series/>
