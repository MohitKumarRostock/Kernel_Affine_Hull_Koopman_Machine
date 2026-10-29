# Mathematical foundations and implementation verification

This document states the mathematical conditions underlying the forecasting
certificates and relates the results to their implementation. Complete proofs
appear in the manuscript's theory and optimization sections. The statements
concern expected squared prediction error for a specified population of
histories and future observations.

## 1. Prediction task and admissible decoders

Let $X$ be an available history, let $Y\in\mathbb R^q$ be the target, and let
$U=\Gamma(X)$ be a measurable representation. A measurable pseudometric $d$
on the representation space may assign zero distance to distinct points. For
a fixed budget $\kappa\geq0$, the admissible decoders satisfy

$$
\|g(u)-g(v)\|_2\leq\kappa d(u,v).
$$

Their optimal risk is the infimum

$$
R^*_\kappa(\Gamma,d)
=\inf_g\mathbb E\|Y-g(\Gamma(X))\|_2^2.
$$

The representation, metric, budget, target definition, and origin distribution
specify the comparator class and its risk. Changing any of these quantities
changes the comparison. The finite-sample results condition on development
data and require the calibration episodes to be independent of all choices
made from those data.

## 2. Conditional pairwise obstruction

Draw a conditioning variable $A$ and then two independent copies
$(X,Y)$ and $(X',Y')$ from the same conditional distribution given $A$.
Both copies retain the original marginal law. Write $U=\Gamma(X)$,
$U'=\Gamma(X')$, and $z_+=\max\{z,0\}$. The pair witness is

$$
W_{\kappa,A}
=\frac12\mathbb E
 \left[(\|Y-Y'\|_2-\kappa d(U,U'))_+^2\right].
$$

If $Y$ has a finite second moment, then $W_{\kappa,A}\leq R^*_\kappa$.
To see why, take any admissible decoder with finite risk and let
$r=Y-g(U)$ and $r'=Y'-g(U')$. The triangle inequality gives

$$
(\|Y-Y'\|_2-\kappa d(U,U'))_+\leq\|r-r'\|_2.
$$

Conditional independence gives the cross-moment identity
$\mathbb E[r^\top r'\mid A]=\|\mathbb E[r\mid A]\|_2^2$, and hence

$$
\frac12\mathbb E\|r-r'\|_2^2
=\mathbb E\|r\|_2^2
 -\mathbb E\|\mathbb E[r\mid A]\|_2^2
\leq\mathbb E\|r\|_2^2.
$$

Taking the infimum proves the bound. A constant $A$ gives ordinary
independent pairing; matching independent episodes at a common time origin
corresponds to conditioning on that origin.

The factor $1/2$ depends on this pairing construction. For arbitrary dependent
pairs with the same marginal law, the generic residual inequality gives only

$$
\frac14\mathbb E
 [(\|Y-Y'\|_2-\kappa d(U,U'))_+^2]
\leq\mathbb E\|Y-g(U)\|_2^2.
$$

For example, with a constant representation, a symmetric target
$Y\in\{-1,1\}$, and the dependent pairing $Y'=-Y$, the expression using
$1/2$ equals two while the constant-zero predictor has risk one.
Nearest-neighbor matching need not preserve either the required conditional
independence or the original marginals.

### Relation to conditional variance

For Euclidean representation distance, also assume a finite second moment
for $U$. Define the averaged conditional variances

$$
\sigma^2=\mathbb E\|Y-\mathbb E[Y\mid A]\|_2^2,
\qquad
\tau^2=\mathbb E\|U-\mathbb E[U\mid A]\|_2^2.
$$

The $L^2$ triangle inequality and conditional independence imply

$$
W_{\kappa,A}\geq(\sigma-\kappa\tau)_+^2.
$$

This comparison uses the same variable $A$ and the same evaluation law on
both sides. It does not compare a common-origin witness with a variance
certificate conditioned on a different reference-class label.

### Exactness for two possible observations

If $(U,Y)$ equals $(u_0,y_0)$ with probability $p$ and $(u_1,y_1)$ otherwise,
ordinary independent pairing gives

$$
R^*_\kappa=W_\kappa
=p(1-p)(\|y_1-y_0\|_2-\kappa\|u_1-u_0\|_2)_+^2.
$$

An affine decoder attains this value by shrinking the target separation to
$\min\{\|y_1-y_0\|_2,\kappa\|u_1-u_0\|_2\}$ while preserving the
weighted target mean. Projection onto a nonempty closed convex set containing
the two targets preserves the attained risk and the Lipschitz budget.
Exactness in this example does not imply sharpness for every distribution.

## 3. Independent episodes and simultaneous risk bounds

The calibration units are independent and identically distributed complete
episodes. Within an episode, origins may overlap and observations may be
arbitrarily dependent or nonstationary. Every compared candidate uses the same
fixed origin set at a given horizon.

If the target and prediction lie in a known output set of diameter $D_h$,
write $B_h=D_h^2$. One episode-averaged loss lies in $[0,B_h]$. One witness
score, averaged over common origins in a disjoint episode pair, lies in
$[0,B_h/2]$. With $n$ episodes, the respective independent sample sizes are
$n$ and $K=\lfloor n/2\rfloor$. The final episode is excluded from pairing
when $n$ is odd. Additional windows do not increase either sample size.

For $N\geq2$ independent scalar observations in an interval of length $b$,
let $\bar V$ be their mean and
$s_V^2=(N-1)^{-1}\sum_i(V_i-\bar V)^2$. The one-sided empirical Bernstein
radius is

$$
\operatorname{rad}_{\rm EB}
=\sqrt{\frac{2s_V^2\log(2/\alpha)}{N}}
 +\frac{7b\log(2/\alpha)}{3(N-1)}.
$$

This is the rescaled form of Maurer and Pontil (2009), Theorem 4. Reflecting
the observations gives the lower direction. Each direction has failure
probability at most $\alpha$; asserting both directions requires allocating
to both events.

For $J$ frozen candidates and $H$ horizons, assigning
$\alpha=\delta/(2JH)$ to each reported lower or upper direction gives,
with probability at least $1-\delta$,

$$
L_{jh}\leq W_{jh}\leq R^*_{jh}\leq R_{jh}\leq U_{jh}
\quad\text{for every }j,h.
$$

The fitted decoder must belong to the same sensitivity class for the middle
comparison $R^*_{jh}\leq R_{jh}$. The upper and lower statistics may use the
same episodes; their mutual independence is unnecessary for the union bound.
The target bound is a population assumption. Clipping forecasts or observing
bounded training values does not establish that an otherwise unbounded target
has this range.

### Correspondence with the software

- `mean_interval` returns both endpoints and labels the failure probability
  **per tail**. `certify_forecast` reports the loss upper endpoint and witness
  lower endpoint, each with its separately supplied allocation.
- `pair_witness_scores` pairs adjacent episodes disjointly, averages within
  each pair over common origins, and discards an odd final episode.
- The score arrays have shape `(episode, origin, horizon)` for separate scalar
  targets. Vector-output losses in the theory sum coordinate errors; the
  PenDigits protocol instead certifies scalar coordinates separately and then
  averages their risks as explicitly specified in that protocol.
- `run_forecasting.py` allocates the risk fraction across twice the number of
  candidate--horizon combinations, and the remaining fraction across the
  declared adjacent-prefix comparisons. Multiple comparisons may share data
  because the validity argument uses a union bound.
- A prespecified Hoeffding radius may replace the empirical Bernstein radius.
  Selecting the tighter of the two after seeing data requires an allocation
  that covers both methods.

## 4. Retained blocks and sensitivity budgets

Let $\Phi_b$ be the retained feature blocks, and define the history
pseudometric

$$
d_s(x,x')=\max_{1\leq b\leq s}\|\Phi_b(x)-\Phi_b(x')\|_2.
$$

For an affine decoder $a+\sum_bW_b\Phi_b(x)$, the constraint
$\sum_b\|W_b\|_{\rm F}\leq\kappa$ implies a Lipschitz constant at most
$\kappa$ relative to $d_s$. An unpenalized intercept does not alter that
constant. Projection onto a nonempty closed convex output set is
nonexpansive, so clipping preserves it.

Appending a block without changing the earlier blocks preserves every old
prediction by assigning zero coefficients to the new block. The feasible
classes are therefore nested and their population infimum risks cannot
increase. At a common budget and pairing law, the witness cannot increase
because the maximum block distance cannot decrease.

If the appended block duplicates an existing one, merging their coefficient
matrices preserves predictions and cannot increase the sum of their norms.
The prediction class and witness are consequently unchanged. A single
Frobenius constraint on concatenated coefficients does not have this
replication invariance. The squared-Frobenius ridge penalty also changes
when coefficients are split among duplicate blocks; the numerical
comparison of duplicate optimum risks therefore uses zero ridge.

For a joint fit, the coefficient columns associated with any one horizon
have total block norm at most the joint budget. The implementation uses this
common declared budget for direct-decoder comparisons. For affine feature
rollouts it instead calculates the horizon-specific effective coefficient
from development data. If the row-feature transition is $z\mapsto zM+b$
and the output decoder is $z\mapsto zv+c$, the horizon-$h$ coefficient is
$M^hv$, with Euclidean Lipschitz bound $\|M^hv\|_2$. The affine offsets do
not affect sensitivity, and only the final output is clipped.

## 5. What refinement can establish

For old and new history pseudometrics $d_0,d_1$, a common positive budget,
and the same pairing law, put
$\Delta=(d_1(X,X')-d_0(X,X'))_+$. If $\Delta$ has a finite second moment,

$$
\sqrt{W_1}\geq
\left(\sqrt{W_0}-\frac{\kappa}{\sqrt2}\|\Delta\|_{L^2}\right)_+.
$$

Thus a new decoder with RMSE at most $\epsilon$ requires

$$
\|\Delta\|_{L^2}\geq
\frac{\sqrt2}{\kappa}(\sqrt{W_0}-\epsilon)_+.
$$

A nonnegative lower confidence bound on the old **pair witness** can replace
$W_0$ on the right. On that old event the necessary condition applies to every
new representation, including adaptively proposed maps. It neither certifies
a new map's risk nor validates an estimate of its distance increase. A lower
bound on oracle risk alone cannot replace a lower bound on the witness.

A separate population calculation gives a sufficient direction of improvement.
For an unprojected predictor $f_0$, residual $r=Y-f_0(X)$, and a new feature
block $\psi$, let $C=\mathbb E[r\psi^\top]$ and
$G=\mathbb E[\psi\psi^\top]$. Finite second moments make these quantities
well-defined. With positive remaining budget $s_0$, nonzero $C$, and

$$
\eta=\min\left\{
\frac{\|C\|_{\rm F}^2}{\operatorname{tr}(CGC^\top)},
\frac{s_0}{\|C\|_{\rm F}}\right\},
$$

appending $V=\eta C$ is feasible and decreases risk by
$2\eta\|C\|_{\rm F}^2-\eta^2\operatorname{tr}(CGC^\top)>0$.
The denominator is positive whenever $C\ne0$. Projection preserves this
relative improvement if both the target and the old forecast already lie
in the stated closed convex output set. Substituting training estimates for
the population moments does not itself preserve the population conclusion.

For two fitted forecasts, a direct comparison uses one loss difference per
independent episode. If each loss lies in $[0,B]$, the difference lies in
$[-B,B]$ and the empirical Bernstein range length is $2B$.
`paired_improvement` shifts this interval to $[0,2B]$, applies the lower
mean bound, and shifts back. Its final lower endpoint is additionally
bounded below by $-B$, which is the known minimum possible improvement.

For sequential refinement, candidates may depend on past results when each
stage uses fresh conditionally independent evaluation episodes and assigned
failure probabilities whose sum is at most the total budget. Repeated tuning
on one calibration set is outside that argument. A positive obstruction may
also reflect irreducible future noise, and a zero obstruction does not
establish the existence of an accurate predictor.

## 6. Convex fitting and numerical optimality

After eliminating the intercept, the finite training objective is

$$
J(W)=\operatorname{tr}(W^\top M W)-2\langle W,C\rangle_{\rm F}+c,
\qquad \nabla J(W)=2(MW-C),
$$

with $M=F_c^\top F_c/N+\lambda I$ positive semidefinite and
$C=F_c^\top Y_c/N$. Projection onto the sum-of-block-norms ball reduces to
Euclidean projection of the block-norm vector onto a nonnegative $\ell_1$
ball. The resulting norms are $(\|A_b\|_{\rm F}-\theta)_+$ for a common
threshold, followed by radial rescaling of each block.

At every feasible coefficient matrix, convexity and the support function of
that ball give

$$
0\leq J(W)-J^*
\leq\langle\nabla J(W),W\rangle_{\rm F}
 +\kappa\max_b\|\nabla_bJ(W)\|_{\rm F}.
$$

`BlockBoundedRidge` records this first-order gap. It measures the regularized
training objective before output clipping. Its floating-point value is a
numerical diagnostic, not an interval-arithmetic proof. The population
certificate depends on feasibility and independent evaluation, not on reaching
the training optimum. Class nesting also does not imply that independently
fitted finite-sample models have monotone risk.

## 7. Exact-risk benchmark and verification evidence

The benchmark uses $Z\sim\operatorname{Bernoulli}(p)$ and target $2Z-1$.
For deficit $c=(2-\kappa\sqrt2a)_+$ and training frequency $\widehat p$,
the selected empirical minimizer has fitted population risk

$$
R(\widehat f)=p(1-p)c^2+c^2(\widehat p-p)^2.
$$

The centered comparator uses two bounded-differences events, one for a lower
output-variance bound and one for an upper representation-variance bound.
Their failure probabilities sum to the centered bound's allocation. The
fitted upper, pair lower, and centered lower directions each receive one third
of the within-trial budget. No joint event across every Monte Carlo trial is
asserted.

`run_known_risk.py` requires at least four calibration observations and uses
$\lfloor n/2\rfloor$ disjoint pairs. An odd final observation still contributes
to the fitted-risk and centered bounds. The retained schedule
$64,256,1024,4096$ is entirely even; the odd-sample correction leaves its score
calculation unchanged.

The mathematical review combined direct proof checking with finite-law and
numerical consistency checks: vector-target conditional pairing, two-atom
affine attainment, the necessary expansion inequality, the residual-risk
identity, independently solved two-variable block-budget optima, the
primary-source empirical Bernstein formula, and exact binomial summation for
the centered comparator. Focused regression tests verify odd sample handling
and unchanged even-sample scores. These checks supplement the proofs and
software tests; they do not amount to a full experimental rerun or a
machine-checked proof.

## 8. Interpretation and attribution

Acceptance certifies an expected RMSE tolerance for the fitted model.
Exclusion concerns every decoder in the specified sensitivity class. An
inconclusive result proves neither excessive risk nor equality of predictors.
Selecting the least costly passing member of a jointly certified finite
family is valid, but need not find the cheapest member whose unknown true risk
meets the tolerance.

Risk confidence bounds are distinct from individual future-error guarantees
and conditional prediction intervals. The framework does not require exact
Koopman closure, a latent Markov model, or stationarity within episodes. It
also does not provide a mixing theorem for a single dependent trajectory or a
guarantee for an unspecified distribution shift. The variance comparison
establishes no new minimax rate; taking square roots of risk bounds can still
produce slow behavior near zero.

The empirical Bernstein inequality is attributed to Andreas Maurer and
Massimiliano Pontil, *Empirical Bernstein Bounds and Sample Variance
Penalization* (2009), [Theorem 4](https://arxiv.org/pdf/0907.3740).
Finite-family risk testing, block norm constraints, and quadratic-risk
expansions are established ingredients. The manuscript develops their use
with the conditional pair witness and retained geometric representations.
The verification presented here does not establish an exhaustive claim of
literature priority.
