# Statistical and numerical methods

## Prediction task and notation

The procedure learns and assesses a finite family of forecasting representations
using one regularly indexed scalar time series. It combines a lower bound on the
best risk in a declared decoder class with an upper bound on the risk of a
particular fitted decoder. An inadequate fit and an inadequate decoder class are
therefore different conclusions.

| Symbol | Definition |
| --- | --- |
| $V_t$ | Recorded scalar observation at time index $t$. |
| $a,b$ | Supplied support bounds, with $a<b$ and $a\leq V_t\leq b$ almost surely. |
| $Z_t=(V_t-a)/(b-a)$ | Normalized observation in $[0,1]$. |
| $T$ | Number of recorded observations. |
| $h$ | Number of observations in a prediction history. |
| $X_t=(Z_{t-h+1},\ldots,Z_t)$ | History available at forecast origin $t$. |
| $Y_t=Z_{t+1}$ | One-step prediction target. |
| $n_0$ | Number of raw observations in the development prefix. |
| $\mathcal D$ | Development data and any declared independent algorithmic randomness. |
| $q$ | Number of consecutive forecasting windows in each calibration block. |
| $K$ | Number of retained calibration blocks. |
| $r$ | Separation lag; each boundary omits $r-1$ raw observations. |
| $N=Kq$ | Total number of calibration windows. |
| $M$ | Number of frozen candidate representation–decoder packages. |
| $\Gamma_m$ | Encoder for candidate $m$, including its preprocessing and scaling. |
| $\kappa_m$ | Euclidean Lipschitz budget for candidate $m$'s decoder class. |
| $\mathcal G_m$ | Full class of $[0,1]$-valued decoders with Lipschitz constant at most $\kappa_m$. |
| $g$ | A decoder in the declared class. |
| $\widehat g_m$ | Decoder fitted using development data only. |
| $R_m(g)$ | Stationary-law mean squared prediction error of decoder $g$ through encoder $\Gamma_m$. |
| $R_m^*$ | Infimum of $R_m(g)$ over $g\in\mathcal G_m$. |
| $L_m,U_m$ | Reported lower class-risk and upper fitted-risk endpoints. |
| $\tau$ | Required normalized mean squared error. |
| $c_m$ | Number of scalar coordinates supplied by encoder $m$, excluding the decoder intercept. |
| $\delta_{\rm stat},\delta_{\rm dep}$ | Statistical and dependence failure allowances for one certification. |
| $\overline\beta(r)$ | Supplied deterministic upper bound on dependence at lag $r$. |
| $p$ | Flip probability in the controlled binary Markov experiment. It is not a feature dimension. |

Dependence on the realized development data is suppressed in $\Gamma_m$,
$\mathcal G_m$, $\widehat g_m$, and $R_m$ to keep the notation readable.
The mathematical notation uses $h$ for history length and $r$ for separation.
The archived implementation retains `d` for history length and `g` or
`gap_lag` for separation; these stored identifiers are distinct from the decoder
$g$ in the mathematical exposition.

Let $\Pi$ be the stationary law of $(X_t,Y_t)$. For a realized development result,
define

$$
R_m(g)=\mathbb E_{(X,Y)\sim\Pi}
\left[(Y-g(\Gamma_m(X)))^2\right],
\qquad R_m^*=\inf_{g\in\mathcal G_m}R_m(g).
$$

The evaluation draw in this definition is independent of development. The risk
averages over the stationary history–target law; it is not the conditional risk
of the very next observation given the particular observed history. A raw-unit
mean squared error tolerance is divided by $(b-a)^2$ before use.

Decimal input strings denote their exact recorded decimal values. Numerical
verification concerns this recorded process, not unknown unrounded physical
measurements.

## Assumptions and temporal layout

The normalized process is strictly stationary, including its initialization.
The support interval and the dependence bound are supplied scientific premises.
An observed training range or an estimated autocorrelation does not establish
these premises.

For probability laws $P$ and $Q$, use the total-variation convention
$\|P-Q\|_{\rm TV}=\sup_A|P(A)-Q(A)|$. Writing $\mathcal L$ for probability law,
the dependence coefficient is

$$
\beta(r)=\sup_s\left\|
\mathcal L(Z_{-\infty:s},Z_{s+r:\infty})-
\mathcal L(Z_{-\infty:s})\otimes\mathcal L(Z_{s+r:\infty})
\right\|_{\rm TV}.
$$

The supplied bound must satisfy $\beta(r)\leq\overline\beta(r)$. The history
length, development cutoff, block count, block size, and gaps are fixed
independently of observed series values. Encoders, scaling, decoder classes,
fitted witnesses, and candidate costs are fixed before calibration; their
construction may use development. Repeated calibration-guided changes to the
candidate family are not covered by the original finite-family allocation.

A raw block has length $\ell=h+q$. For $k=1,\ldots,K$, its first and last raw
indices are

$$
a_k=n_0+r+(k-1)(\ell+r-1),\qquad b_k=a_k+\ell-1.
$$

Its forecast origins are $t_{k,j}=a_k+h+j-2$, for $j=1,\ldots,q$; each target is
at $t_{k,j}+1$. Every history and target lies inside its raw block. The complete
layout requires

$$
n_0+K(h+q+r-1)\leq T.
$$

All development targets lie in the prefix $1{:}n_0$. Omitted gaps and trailing
observations are not used to revise the candidates. Overlap among windows inside
a calibration block is allowed. The statistical unit is the whole block.

## Declared representation and predictor family

The executable instance uses rational affine reconstruction scores. It differs
from the Gaussian Kernel Affine Hull Machine and radial/angular feature engine
used in the related forecasting studies.

For reference group $j=1,\ldots,J$, development determines a rational center
$\mu_j\in[0,1]^h$ and a coordinate index $k_j$. Let $e_{k_j}$ be that coordinate's
unit vector and define the affine reconstruction

$$
A_j(x)=\mu_j+e_{k_j}e_{k_j}^{\mathsf T}(x-\mu_j),\qquad
r_j^2(x)=\|x-A_j(x)\|_2^2.
$$

For positive rational parameters $s,\eta$ and positive integer $\omega$, define

$$
T_j(x)=\frac{r_j^2(x)}{s+r_j^2(x)},\qquad
\phi_j(x)=
\frac{(1-T_j(x)+\eta)^\omega}
{\sum_{j'=1}^J(1-T_{j'}(x)+\eta)^\omega}.
$$

The score scale is $s$, the positive affinity offset is $\eta$, and $\omega$
controls affinity contrast. Every denominator is positive. The memberships
$\phi_j$ sum to one; this normalization does not make them calibrated class
probabilities.

For a positive rational feature scale $\gamma$, the three encoders are

$$
\Gamma_{\rm raw}(x)=\gamma x,\qquad
\Gamma_{\rm mem}(x)=\gamma\phi(x),
$$

$$
\Gamma_{\rm loc}(x)=\gamma\operatorname{col}
\left(\phi(x),
\{\phi_j(x)(x_{k_j}-\mu_{j,k_j})\}_{j=1}^J\right).
$$

Here $\operatorname{col}$ concatenates coordinates. The costs are $h,J,2J$.
The experiment uses $h=6$, $J=2$, $\gamma=s=1$, $\eta=1/16$, $\omega=2$, and
$k_j=h$. Centers are development-history exemplars selected at deterministic
midpoint quantiles after sorting by the latest coordinate, with the complete
history as a tie breaker.

Each decoder class is

$$
\mathcal G_m=
\{g:\mathbb R^{c_m}\to[0,1]:
|g(u)-g(v)|\leq\kappa_m\|u-v\|_2\text{ for all }u,v\}.
$$

The fitted witness has the form
$\widehat g_m(u)=\operatorname{clip}_{[0,1]}(b_m+w_m^{\mathsf T}u)$, where $b_m$
is its intercept and $w_m$ is its coefficient vector. A development-only ridge
fit proposes coefficients. The stored coefficients are dyadic rational numbers;
exact verification checks $\|w_m\|_2^2\leq\kappa_m^2$. Clipping is nonexpansive,
so this proves membership on the whole feature space. The experiment uses
$\kappa_m=2$ and a mean-loss ridge penalty of $1/1000$.

The encoder need only be measurable for temporal calibration. Full-history
recovery, Markov sufficiency, and autonomous state evolution are separate
properties and are not asserted for these compact encoders.

## Pointwise numerical certificates

Flatten the $K$ blocks into $N=Kq$ feature–target pairs $(u_i,y_i)$, with
$u_i=\Gamma_m(x_i)$. For each fixed candidate, the empirical class optimum is

$$
a_m=\inf_{g\in\mathcal G_m}\frac1N\sum_{i=1}^{N}(y_i-g(u_i))^2.
$$

The numerical routines must return a nonnegative $\underline a_m\leq a_m$ and
an upper value $\overline r_m$ for the fitted predictor's empirical mean loss.
These inequalities are pointwise: they hold for every admissible combination of
development data and calibration blocks, including independent combinations
used in the statistical proof.

### Lower input

Select oriented edges $e=(i,j)$ and rational bounds
$c_e\geq\min\{1,\kappa_m\|u_i-u_j\|_2\}$. Bounds are rounded upward. The
prediction range makes a bound of one sufficient. Exact squared comparisons
verify smaller bounds without trusting a floating-point square root.

Let $B$ be the incidence matrix, with $+1$ at the first endpoint and $-1$ at the
second. For arbitrary signed rational edge multipliers $\lambda$, put
$z=B\lambda$ and $v_i^\lambda=\operatorname{clip}_{[0,1]}(y_i-z_i/2)$. The boxed
dual value is

$$
D_{\rm box}(\lambda)=
\sum_i\big[(y_i-v_i^\lambda)^2+z_i v_i^\lambda\big]
-\sum_e c_e|\lambda_e|\leq N a_m.
$$

For an admissible prediction vector $v$, every term
$\lambda_e(v_i-v_j)-c_e|\lambda_e|$ is nonpositive. Adding these terms to squared
loss and minimizing over $v\in[0,1]^N$ gives the displayed inequality.

A disjoint pair set $\mathcal P$ supplies the additional residual-sum bound

$$
P_{\mathcal P}=
\sum_{(i,j)\in\mathcal P}\frac{(|y_i-y_j|-c_{ij})_+^2}{2}
\leq N a_m,
$$

where $(x)_+=\max(0,x)$. The returned lower input is a downward dyadic rounding
of $\max\{0,D_{\rm box},P_{\mathcal P}\}/N$. Isolated and unpaired observations
remain in $N$. Overlapping graph and pair bounds are maximized, not added.

Graph selection and multiplier proposals may use calibration data because the
resulting bound remains pointwise valid. The implementation uses temporal and
single-coordinate sorted orders to propose a sparse graph, with two neighbors
per order and at most four orders. Finite floating-point coordinate sweeps
propose multipliers, which are rationalized and checked exactly. Convergence or
solver optimality is unnecessary for validity; sparse graphs and limited
optimization can still weaken informativeness.

### Upper input and endpoint arithmetic

Each fitted prediction is evaluated exactly from its normalized raw history.
For exact squared loss $\ell_i\in[0,1]$, round upward to the dyadic grid of width
$2^{-b_\ell}$ before averaging. Thus

$$
\overline r_m=\frac1N\sum_i2^{-b_\ell}\lceil2^{b_\ell}\ell_i\rceil,
\qquad
0\leq\overline r_m-\widehat r_m<2^{-b_\ell},
$$

where $\widehat r_m$ is the exact empirical mean loss. This controls denominator
growth while preserving the declared mathematical features and predictor.

The default precisions are 32 bits for individual loss rounding, 20 bits for
outward rounding of empirical inputs before KL inversion, and 24 bits for the
requested endpoint grid. Distance bounds use 32 bits; signed multipliers use
24 bits. Lower empirical output rounding uses 40 bits.

Natural logarithms are enclosed with exact rational series. For $x=2^k u$,
$1\leq u<2$, put $z=(u-1)/(u+1)$. For a positive integer $n$,

$$
2\sum_{j=0}^{n-1}\frac{z^{2j+1}}{2j+1}
\leq\log u\leq
2\sum_{j=0}^{n-1}\frac{z^{2j+1}}{2j+1}
+\frac{2z^{2n+1}}{(2n+1)(1-z^2)}.
$$

The same expression at $u=2$ bounds $\log2$. Interval addition with the correct
sign of $k$ bounds $\log x$. Endpoint bisection changes a bracket boundary only
after a proved comparison; unresolved comparisons retain a conservative outer
bracket. A separate checker verifies the endpoint inequality directly.

## Simultaneous temporal calibration

For $v,p_0\in[0,1]$, binary relative entropy is

$$
\operatorname{kl}(v\Vert p_0)=
v\log\frac{v}{p_0}+(1-v)\log\frac{1-v}{1-p_0}.
$$

The symbol $p_0$ is a candidate mean in this definition, distinct from the
experiment's flip probability $p$. Let $k_-(v,c)$ and $k_+(v,c)$ be the smallest
and largest $p_0$ satisfying $\operatorname{kl}(v\Vert p_0)\leq c$.

Allocate endpoint failure probabilities $\alpha_m^L,\alpha_m^U>0$ with
$\sum_m(\alpha_m^L+\alpha_m^U)\leq\delta_{\rm stat}$. The unrounded endpoints are

$$
L_m=k_-\left(\underline a_m,\frac{\log(1/\alpha_m^L)}{K}\right),\qquad
U_m=k_+\left(\overline r_m,\frac{\log(1/\alpha_m^U)}{K}\right).
$$

Their outward numerical enclosures obey the same guarantee:

$$
\Pr\!\left\{\forall m:\;
L_m\leq R_m^*\leq R_m(\widehat g_m)\leq U_m\right\}
\geq1-\delta_{\rm stat}-K\overline\beta(r).
$$

Under an independent-block reference law, conditioning on development fixes the
candidate classes and fitted decoders. Each block's average bounded loss has
mean $R_m(g)$. Bounded-mean KL calibration applies to the $K$ block losses.
For the lower endpoint, the pointwise empirical-infimum inequality compares
$\underline a_m$ to every fixed near-optimal decoder's empirical loss; an
attaining class optimizer is unnecessary. A union bound covers all $2M$
endpoints.

The total-variation distance from the actual development-and-block experiment
to this reference law is at most $K\overline\beta(r)$. It includes the boundary
from development to the first block. Transferring the whole simultaneous event
pays this dependence allowance once, with no extra factor $M$.

Coverage is unconditional over the complete experiment, under the supplied
scientific premises. It is not a separate conditional guarantee for every
realized development prefix. The independent-block law is a proof device, not
an additional data requirement. Statistical calibration uses $K$, whereas
empirical optimization and fitted-loss evaluation use all $N=Kq$ windows.

## Decisions and coordinate cost

For fixed tolerance $\tau$, candidate $m$ is accepted if $U_m\leq\tau$, excluded
if $L_m>\tau$, and otherwise unresolved. An unavailable input, premise, or
required numerical check produces no population certificate.

Define actual class feasibility by the existence of a decoder meeting the
requirement. The minimum feasible coordinate cost is

$$
c^*=\min\{c_m:\exists g\in\mathcal G_m,\ R_m(g)\leq\tau\}.
$$

With an empty minimum defined as $+\infty$, let

$$
c_- =\min_{m:L_m\leq\tau}c_m,\qquad
c_+ =\min_{m:U_m\leq\tau}c_m.
$$

On the same simultaneous confidence event, $c_-\leq c^*\leq c_+$. Equality
$c_-=c_+<\infty$ certifies minimum feasible cost in the declared family.
Otherwise the procedure reports the bracket and selects a least-cost accepted
candidate when one exists. An empty accepted set alone does not establish
infeasibility. The entire family is excluded only when every class is excluded.

Cost counts the declared scalar feature coordinates, including any algebraic
redundancy. It is not a claim about intrinsic state dimension, storage, latency,
or optimality over unexamined representations.

## Controlled experiment and population truth

The synthetic process starts with a fair binary state and changes state with
probability $p$ at each transition. Its stationary distribution is $(1/2,1/2)$,
and the stated total-variation convention gives

$$
\beta(r)=\tfrac12|1-2p|^r.
$$

The experiment crosses $p\in\{0.05,0.20,0.50,0.95\}$ with $K\in\{80,320\}$
at $q=4$. Additional matched settings use $q=1,16$ and zero or 32 multiplier
sweeps at $p=0.05,K=320$, against the default eight sweeps. Each of the twelve
settings contains 200 repetitions. The history length is six, development uses
512 observations, and $\tau=0.1$. The statistical allowance is $0.04$, the
dependence allowance is $0.01$, and each of the six endpoints receives
$\alpha_m^L=\alpha_m^U=1/150$. The gap is the smallest positive integer satisfying
$K\beta(r)\leq0.01$.

For a binary history $x=(x_1,\ldots,x_h)$, its stationary mass and next-target
conditional mean are

$$
\pi(x)=\tfrac12\prod_{j=2}^h
\left[p^{\mathbf1\{x_j\ne x_{j-1}\}}
(1-p)^{\mathbf1\{x_j=x_{j-1}\}}\right],\qquad
\mu(x)=p+(1-2p)x_h.
$$

The population oracle enumerates all $2^h=64$ histories with these exact
rational weights. Fitted risk is evaluated over that finite law, without an
additional finite test sample.

Histories with the same exact feature tuple are aggregated into feature atoms.
For atom $a$, let $w_a$ be its probability mass and $\mu_a$ its conditional target
mean. For a prediction $v_a$ on each atom,

$$
R=I+\sum_a w_a(v_a-\mu_a)^2,\qquad
I=\sum_a w_a\mu_a(1-\mu_a).
$$

If the atom means satisfy all Lipschitz constraints, the optimum is exactly
$I$. Otherwise a weighted boxed graph dual supplies a lower bound. A rational
prediction vector, conservatively shrunk toward $1/2$ and checked against every
pair, supplies an upper bound. Scalar Lipschitz extension followed by clipping
extends such a feasible finite vector to the full decoder domain. Nonzero
oracle intervals remain explicit; an ambiguous comparison is not called
successful coverage.

The raw and latest-local classes attain $p(1-p)$ for every development outcome
in this experiment. For the local encoder with scale one,

$$
x_h=\sum_j\phi_j(x)\mu_{j,h}
+\sum_j\phi_j(x)(x_h-\mu_{j,h}).
$$

The affine decoder for $p+(1-2p)x_h$ has coefficient norm at most
$2|1-2p|\leq2$ when $J=2$ and the centers are binary. This proves the required
class membership. Recovering this single predictive coordinate for the
first-order binary law does not establish full-history recovery.

## Simulation inference and observed limitations

The primary event is coverage of all three candidates' risk sandwiches in a
repetition. Definite violations, unresolved oracle comparisons, and unavailable
executions are reported separately and counted conservatively as failures for
the procedure-level Monte Carlo diagnostic. This diagnostic is stricter than
erroneous certificate issuance alone.

Each setting has 200 repetition units. A one-sided binomial failure bound uses
$\alpha_{\rm MC}=0.05/12=1/240$ per setting, yielding simultaneous 95% coverage
of the twelve failure bounds by a union bound. With zero counted failures, the
verified outward rational upper bound is $113377/4194304$, approximately
$0.02703118$. Secondary two-sided intervals are pointwise 95% Clopper–Pearson
intervals. Matched settings share trajectory streams and are not pooled as
2,400 independent trials.

The retained results contain 2,400 covered joint events and no unavailable or
oracle-unresolved events. Coverage is evidence about the specified simulation
laws under the independent-replicate pseudorandom simulation interpretation;
it is not a universal empirical guarantee.

Three practical limitations can be distinguished:

1. **Sparse empirical constraints.** At $p=0.50,K=320$, the raw candidate has
   mean empirical lower value $0.020929$ and mean population lower endpoint
   $0.004609$, although its true class risk is $0.25$. A post-hoc diagnostic of
   the saved replicate zero finds full empirical optimum $0.24016370666$,
   sparse-graph optimum $0.01959635417$, and returned lower value
   $0.01959634970$. Its missing equality links dominate the gap; the remaining
   numerical gap is approximately $4.47\times10^{-9}$. This exact diagnosis
   concerns that saved record, not every repetition.
2. **Statistical slack.** At $p=0.05,K=320$, memberships have mean empirical
   lower value $0.109827$ but mean population lower endpoint $0.062826$.
   Only two of 187 truly infeasible membership classes are excluded. Thus
   accepting a four-coordinate model rarely also proves that every cheaper
   class fails. Minimum cost is certified in one of 200 runs.
3. **Chosen-witness error.** The membership class is feasible in thirteen
   developments at $p=0.05$, but its fitted witness exceeds tolerance in all
   thirteen. Their mean class optimum is $0.095473$, while mean fitted risk is
   $0.110080$. Sharper confidence intervals alone cannot accept these fitted
   witnesses. This does not establish failure of every affine learner or
   isolate expressiveness from fitting and regularization effects.

The registered sweep comparison increases mean membership lower endpoints from
$0.03477$ to $0.06283$ to $0.06516$ with zero, eight, and 32 sweeps. Increasing
eight to 32 sweeps changes no acceptance or exclusion decision. The block-size
comparison keeps $K$ fixed while changing both $N$ and $T$, so it is not a
fixed-data-budget efficiency comparison.

The post-hoc graph and witness descriptions are labeled separately from the
prespecified experimental comparisons. No improved graph or fitted predictor
is substituted into the retained study.

## Scope

The guarantee requires the stated support, stationarity, dependence, temporal
design, and freezing assumptions. The implementation checks numerical
inequalities and declared data flow; it cannot authenticate historical
preregistration or infer these scientific premises from an arbitrary record.
Exact arithmetic is not proof-assistant verification or data provenance
authentication.

The results concern the declared rational encoder family and coordinate-cost
objective. They do not verify the separate Gaussian KAHM feature engine,
establish latent-state recovery or autonomous dynamics, guarantee informative
bounds for every finite record, or demonstrate general forecasting superiority.

The numerical evidence is in [results/summary.json](results/summary.json), with
the full repetition records in [results/runs.jsonl.gz](results/runs.jsonl.gz).
The explanatory calculations are recorded in
[results/posthoc_graph_diagnostic.json](results/posthoc_graph_diagnostic.json)
and [results/bottleneck_diagnostics.json](results/bottleneck_diagnostics.json).
