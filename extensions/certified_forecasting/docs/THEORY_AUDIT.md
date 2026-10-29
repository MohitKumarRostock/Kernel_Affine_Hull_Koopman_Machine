# Mathematical foundations and implementation verification

This document states the mathematical conditions underlying the forecasting
certificates and relates the results to their implementation. Complete proofs
appear in the manuscript's theory and optimization sections and its appendix
on comparison certificates. The statements
concern expected squared prediction error for a specified population of
histories and future observations. The account below includes the substantive
revision of 29 October 2026; retained results from the earlier experiments are
identified separately from the additional protocols and derivations.

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

### Exact slack and equality conditions

For an admissible finite-risk decoder, put
$v=(\|Y-Y'\|_2-\kappa d(U,U'))_+$. The conditional variance identity
above gives the exact decomposition

$$
R(g\circ\Gamma)-W_{\kappa,A}
=\mathbb E\|\mathbb E[r\mid A]\|_2^2
 +\frac12\mathbb E\left[\|r-r'\|_2^2-v^2\right].
$$

Both terms are nonnegative. Equality therefore requires zero conditional
residual mean and $\|r-r'\|_2=v$ almost surely. On pairs with a nonzero
residual difference, the latter condition requires a saturated Lipschitz
constraint and aligned residual and prediction differences. This separates
conditional bias from geometric slack in the pair inequality.

Equality is not limited to two atoms. For $U=T\in\mathbb R$, absolute input
distance, and
$Y=b+\lambda T e$, where $\|e\|_2=1$, $\lambda\geq0$, and
$T$ has finite variance, ordinary independent pairing gives

$$
W_\kappa=R^*_\kappa
=(\lambda-\kappa)_+^2\operatorname{Var}(T).
$$

The affine forecast with mean $b+\lambda\mathbb E[T]e$ and slope
$\min(\lambda,\kappa)e$ attains this value, including on continuous
supports.

### Two distinct population gaps

Let $(U,Y)$ take the values $(3,0),(0,-1),(1,1)$ with probabilities
$1-2\varepsilon,\varepsilon,\varepsilon$, where
$0<\varepsilon<1/2$, and use $\kappa=1$ and absolute distance. Only
the two rare atoms violate the sensitivity constraint. Thus
$W_\kappa=\varepsilon^2$, whereas the feasible predictions
$g(0)=-1/2$, $g(1)=1/2$, and $g(3)=0$ attain
$R^*_\kappa=\varepsilon/2$. The rare pair alone gives the matching
risk lower bound. Consequently $W_\kappa/R^*_\kappa=2\varepsilon$
can approach zero despite independent pairing and fixed bounded supports.
The embedding $u\mapsto(u/3,1-u/3)$ in the two-coordinate simplex,
with budget $3/\sqrt2$, preserves this example.

A different gap arises when the comparison class contains nonlinear
Lipschitz decoders but fitting uses clipped affine decoders. Let
$x\in\{-1,0,1\}$ be uniform, set
$\Phi(x)=((x+1)/2,(1-x)/2)$ and $Y=x^2$, and use budget $\sqrt2$.
The Lipschitz optimum and witness are zero. Every clipped affine decoder is
monotone in $x$, so a conflicting adjacent target pair forces MSE at least
$1/6$. The unprojected affine optimum is the constant $2/3$, with risk
$2/9$. More calibration cannot eliminate either this class difference or
the preceding witness slack.

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

For $N\geq2$ independent, identically distributed scalar observations in an
interval of length $b$,
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

### Stratification with estimated population masses

Condition on development data and freeze a measurable history partition
$c:\mathcal X\to\{1,\ldots,M\}$. Write
$\pi_a=\Pr\{c(X)=a\}$ and let $W_a$ be the pair witness conditional on
stratum $a$, with $W_a=0$ for a zero-mass stratum. Conditional pairing gives

$$
W_{\rm str}=\sum_{a=1}^M\pi_aW_a\leq R^*_\kappa.
$$

The weights retain the original marginal risk; the target distribution is
not changed to equal weighting over strata. For a trajectory application,
one independently sampled origin from the prespecified common origin law
per independent episode supplies an iid calibration observation $(X_i,Y_i)$.
All dependent windows must not be treated as iid observations. A fitted-risk
upper bound can still average all prescribed origins within each episode;
the subsequent union bound needs no independence between these two statistics.

For $n$ such observations, let $N_a$ be the count in stratum $a$ and set
$K_a=\lfloor N_a/2\rfloor$. Pair in original sample-index order within
each stratum. The score

$$
Q_{a,r}=\frac12
 (\|Y_{a,2r-1}-Y_{a,2r}\|_2
   -\kappa d(U_{a,2r-1},U_{a,2r}))_+^2
$$

lies in $[0,B/2]$ when the target set has squared diameter $B$.
For allocated failures $\delta_p,\beta_1,\ldots,\beta_M>0$, define
$\ell_a=0$ when $K_a<2$; otherwise use the empirical Bernstein lower
endpoint with score range $B/2$ and failure probability $\beta_a$.
Set

$$
t_p=\sqrt{\frac{\log(2M/\delta_p)}{2n}},\qquad
p_a^-=(N_a/n-t_p)_+,\qquad
p_a^+=\min(1,N_a/n+t_p),
$$

and let $\mathcal P$ be the probability vectors satisfying
$p_a^-\leq p_a\leq p_a^+$. Then, with failure probability at most
$\delta_p+\sum_a\beta_a$,

$$
L_{\rm str}=\min_{p\in\mathcal P}\sum_a p_a\ell_a
\leq W_{\rm str}\leq R^*_\kappa.
$$

Conditional on the complete label vector, each stratum contains iid
observations from its conditional law. Disjoint pair scores are therefore
iid for every realized count. Integrating their conditional failure bounds
requires no union over possible counts. Two-sided Hoeffding bounds for the
indicator means supply the mass event. A union bound combines these events
without assuming that masses and score estimates are independent.

The set $\mathcal P$ contains the empirical frequencies and is nonempty.
Its minimum is obtained by starting at $p^-$ and allocating remaining mass
to increasing $\ell_a$ until the corresponding upper bounds are reached.
Known population masses remove the mass-estimation step, but are reported
as a separate comparison when the primary method estimates them. Several
frozen partitions may be combined by a maximum after allocating across their
score and mass events. A shared partition requires only one mass event.

If every positive-mass stratum has either a constant representation or a
two-point joint law, each conditional witness is its conditional optimum.
Their weighted sum equals the global optimum when a single admissible decoder
realizes all conditional optimal predictions. For scalar finite support this
compatibility is equivalent to all pairwise Lipschitz inequalities, including
cross-stratum pairs. The extension
$g(u)=\min_i\{v_i+\kappa d(u,u_i)\}$ proves sufficiency. The rare
three-atom example becomes exact with strata $\{3\}$ and $\{0,1\}$:
$W_{\rm str}=2\varepsilon(1/4)=\varepsilon/2$.
Stratification is not universally beneficial: singleton strata can erase
conflicts between incompatible conditional predictions. Small strata also
reduce the number of available pairs and increase uncertainty.

The additional `experiments/witness_comparison` and
`experiments/decision_value` protocols implement this iid-unit construction.
Their primary stratified endpoints estimate the masses and allocate to both
mass and conditional-score uncertainty. Neither counts dependent windows as
additional observations.

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

## 7. Retained exact-risk benchmark

The benchmark uses $Z\sim\operatorname{Bernoulli}(p)$ and target $2Z-1$.
For deficit $c=(2-\kappa\sqrt2a)_+$ and training frequency $\widehat p$,
the selected empirical minimizer has fitted population risk

$$
R(\widehat f)=p(1-p)c^2+c^2(\widehat p-p)^2.
$$

The original centered comparator uses two bounded-differences events, one for a lower
output-variance bound and one for an upper representation-variance bound.
Their failure probabilities sum to the centered bound's allocation. The
fitted upper, pair lower, and centered lower directions each receive one third
of the within-trial budget. No joint event across every Monte Carlo trial is
asserted. This retained comparator does not exploit a structurally constant
representation or the direct sample-standard-deviation bounds described
below. Its historical results do not establish dominance over every centered
certificate.

`run_known_risk.py` requires at least four calibration observations and uses
$\lfloor n/2\rfloor$ disjoint pairs. An odd final observation still contributes
to the fitted-risk and centered bounds. The retained schedule
$64,256,1024,4096$ is entirely even; the odd-sample correction leaves its score
calculation unchanged.

The earlier mathematical review checked vector-target conditional pairing,
two-atom affine attainment, necessary expansion, residual-risk identities,
two-variable block-budget optima, and the empirical Bernstein formula. Exact
binomial summation checked the original centered comparator. Focused tests
checked odd-sample handling and unchanged even-sample scores. These were
algebraic and targeted software checks, rather than a full rerun of the
original experimental collection. The additional studies retain separate
protocols, execution records, and results.

## 8. Comparison certificates and bounded-loss calibration

### A range-aware centered comparator

For scalar $U,Y$ with known interval widths $D_U,D_Y$, the Euclidean
variance obstruction can use direct standard-deviation concentration from
[Maurer and Pontil (2009), Theorem 10](https://arxiv.org/pdf/0907.3740).
For $n\geq2$ iid pairs, let $s_U,s_Y$ be square roots of unbiased sample
variances and let $\widehat v_Y=n^{-1}\sum_i(Y_i-\bar Y)^2$.
For total failure probability $\eta$, set $\alpha=\eta/3$ when
$D_U>0$, or $\alpha=\eta/2$ when $D_U=0$, and put
$c_\alpha=\sqrt{2\log(1/\alpha)/(n-1)}$. Define

$$
\sigma_Y^-=
\max\left\{(s_Y-D_Yc_\alpha)_+,
\sqrt{\left(\widehat v_Y-D_Y^2
             \sqrt{\frac{\log(1/\alpha)}{2n}}\right)_+}\right\},
$$

$$
\sigma_U^+=
\begin{cases}
\min(s_U+D_Uc_\alpha,D_U/2),&D_U>0,\\
0,&D_U=0.
\end{cases}
\qquad
L_{\rm ctr}=(\sigma_Y^- -\kappa\sigma_U^+)_+^2.
$$

The direct standard-deviation events each use failure probability $\alpha$.
The additional target lower bound follows from bounded differences:
replacing one observation changes
$\widehat v_Y=\inf_a n^{-1}\sum_i(Y_i-a)^2$ by at most $D_Y^2/n$,
and $\mathbb E\widehat v_Y=(1-1/n)\operatorname{Var}(Y)$.
The known range gives $\operatorname{Var}(U)\leq D_U^2/4$.
Thus $L_{\rm ctr}\leq R^*_\kappa$ except on an event of probability
at most $\eta$. Both target bounds are allocated before taking their maximum.
A structurally constant representation has variance exactly zero and requires
no variance-estimation penalty; sample constancy alone is insufficient.
The direct standard-deviation result also means that taking square roots of
a variance confidence bound is not the only available centered procedure.

### Relative-entropy inversion for a bounded mean

For $p,q\in[0,1]$, write
$\operatorname{kl}(q\Vert p)=q\log(q/p)+(1-q)\log((1-q)/(1-p))$,
with the usual continuous endpoint conventions. Let
$\operatorname{kl}_-(q,c)$ and $\operatorname{kl}_+(q,c)$ be the
smallest and largest $p$ for which $\operatorname{kl}(q\Vert p)\leq c$.
If $V_i\in[0,B]$ are iid, $B>0$, with mean $\mu$, each of

$$
B\operatorname{kl}_-\left(\bar V/B,\frac{\log(1/\eta)}n\right)
\leq\mu,
\qquad
\mu\leq
B\operatorname{kl}_+\left(\bar V/B,\frac{\log(1/\eta)}n\right)
$$

has failure probability at most $\eta$. The two directions require separate
allocations for a joint statement. This is established bounded-variable
Chernoff calibration, not a new concentration inequality. The proof uses
the exponential chord bound
$\mathbb E e^{tV_i/B}\leq1-\mu/B+(\mu/B)e^t$, followed by independence,
exponential Markov bounds, and scalar optimization. The scores need not be
Bernoulli. When $B=0$, both endpoints are zero. One bounded episode average
is one observation for this calculation.

### A lower confidence bound from the empirical minimum

For a fixed nonempty class $\mathcal G$ with losses in $[0,B]$, $B>0$,
define

$$
R^*=\inf_{g\in\mathcal G}\mathbb E\ell_g(X,Y),\qquad
a_n=\inf_{g\in\mathcal G}\frac1n\sum_i\ell_g(X_i,Y_i),
$$

using iid observations from the target law and assuming measurability of
the infima. If a computed dual value satisfies $d_n\leq a_n$, put
$q_n=\max(0,d_n/B)$. Then

$$
L_{\rm ERM}
=B\operatorname{kl}_-\left(q_n,\frac{\log(1/\eta)}n\right)
\leq R^*
$$

with failure probability at most $\eta$. No uniform complexity penalty is
needed for this direction. For every fixed population competitor, the
empirical infimum is below that competitor's empirical risk. More formally,
for $t\geq0$,

$$
\mathbb E e^{tn a_n/B}
\leq\inf_g\{1-R(g)/B+(R(g)/B)e^t\}^n
=\{1-R^*/B+(R^*/B)e^t\}^n.
$$

The argument does not require an attained population minimum. It lower-bounds
the risk of every class member, but supplies no upper risk or excess-risk
guarantee for a predictor fitted on these calibration observations. Empirical
minimum-risk comparisons are already discussed in
[Kumar and Kargaran (2026), Section 2.5](https://arxiv.org/html/2609.32652v3);
the present benchmark uses a certified numerical lower value and standard
relative-entropy inversion.

For scalar Lipschitz regression, the empirical minimum is the established
finite quadratic program described by
[Gottlieb, Kontorovich, and Krauthgamer (2017), Section III-A](https://www.wisdom.weizmann.ac.il/~robi/papers/GKK-MetricRegression-ITIT17-AuthorVersion.pdf).
Use only distinct observed inputs $u_j$, their positive empirical frequencies
$w_j$, target first moments $t_j=n^{-1}\sum_{i:U_i=u_j}Y_i$, and
$\nu=n^{-1}\sum_iY_i^2$. The objective is
$\nu+\sum_j(w_jv_j^2-2t_jv_j)$. Pairwise Lipschitz constraints and the
known output interval have the form $Av\leq b$. Scalar Lipschitz extension
and interval projection establish equivalence to the global clipped class.
For ordered real inputs with absolute distance, adjacent constraints imply
all pairwise constraints. For any nonnegative multiplier vector $\lambda$,

$$
d(\lambda)=\nu-\lambda^\top b
-\sum_j\frac{((A^\top\lambda)_j-2t_j)^2}{4w_j}
\leq a_n.
$$

This is the unconstrained quadratic minimum of the Lagrangian. A primal
objective is an upper bound on $a_n$ and cannot substitute for it. The
implementation records feasibility, a downward numerical guard on the dual
value, and the primal--dual gap. These floating-point checks are not
interval-arithmetic certificates.

## 9. Decisions, cost optimality, and interpretation

Acceptance certifies an expected RMSE tolerance for the fitted model.
Exclusion concerns every decoder in the specified sensitivity class. An
inconclusive result proves neither excessive risk nor equality of predictors.

For a finite family of candidate classes $\mathcal C_j$, let $c_j$ be
the prespecified cost, and require each fitted forecast to belong to its
class. Let $c^*$ be the minimum cost of a class containing a forecast that
meets all horizon tolerances $\epsilon_h$. The minimum of an empty set is
$+\infty$. On a joint event for all class lower bounds and fitted-risk upper
bounds, define

$$
\mathcal E=\{j:L_{jh}>\epsilon_h^2\text{ for some }h\},\qquad
\mathcal A=\{j:U_{jh}\leq\epsilon_h^2\text{ for all }h\}.
$$

Then

$$
c_-:=\min_{j\notin\mathcal E}c_j
\leq c^*
\leq c_+:=\min_{j\in\mathcal A}c_j.
$$

Every accepted fitted forecast is feasible, and every feasible class is
nonexcluded. These inclusions prove the interval. Finite equality
$c_-=c_+$ certifies minimum feasible cost within the stated family. If all
classes are excluded, the requirements are infeasible in their union. If the
endpoints differ, choosing the least costly accepted forecast establishes
only minimum cost among certified candidates. Exclusion by a larger
horizon-wise Lipschitz class remains valid for a smaller jointly constrained
fitted class. Retained feature count is a cost proxy, not measured runtime.

Risk confidence bounds are distinct from individual future-error guarantees
and conditional prediction intervals. The framework does not require exact
Koopman closure, a latent Markov model, or stationarity within episodes. It
also does not provide a mixing theorem for a single dependent trajectory or a
guarantee for an unspecified distribution shift. The variance comparison
establishes no new minimax rate; taking square roots of risk bounds can still
produce slow behavior near zero.

## 10. Verification and attribution of the additional results

The new derivations were checked against the implemented score ranges,
failure allocations, random stratum counts, constrained-mass minimum, and
quadratic dual. Independent exact rational calculations verified two examples
separating conditional-bias and geometric slack, a three-point aligned law
with $W=R^*=243/64$, and the rare-conflict law at $\varepsilon=1/20$
with $W=1/400$ and $W_{\rm str}=R^*=1/40$. A separate finite noisy
state law gave an unconditional witness $81/2500$, a compatible
state-conditioned witness $69/500$, and an incompatible finer-partition
witness $1/100$. These calculations verify the stated examples and the need
for compatibility; the proofs establish the general statements.

The benchmark's focused tests also cover analytically known optima,
unobserved support points and scalar extension, arbitrary nonnegative dual
multipliers, exact binomial coverage of the relative-entropy lower endpoint,
and the constrained-mass minimum against a linear-program solver. Execution
and aggregate empirical results belong to the individual study records and
the repository's scientific verification report. No finite test suite is a
machine-checked proof of distribution-free coverage.

The empirical Bernstein and direct standard-deviation inequalities are
attributed to Maurer and Pontil (2009), Theorems 4 and 10. Bounded-variable
Chernoff bounds, empirical-minimum lower calibration, scalar Lipschitz
regression, finite-family risk testing, and block norm constraints are
established ingredients. The paired-residual argument also has a close
antecedent in Akash et al. (2021), Lemma 2; the explicit relation is given in
the manuscript's comparison appendix and the
[literature audit](LITERATURE_AUDIT.md). The current development concerns the
conditional witness, its slack, estimated-mass stratification, retained
geometry, and the resulting finite-family decisions. This targeted review
does not establish an exhaustive literature-priority claim.
