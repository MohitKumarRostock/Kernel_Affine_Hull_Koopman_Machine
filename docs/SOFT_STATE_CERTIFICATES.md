# Prediction-error certificates for soft state representations

## 1. Statistical model

Let $(X,X^+)$ have a specified transition law, let $c:\mathcal X\to\{1,\ldots,C\}$ be a fixed reference partition, and let $\phi:\mathcal X\to\Delta_C$ be a fixed measurable representation. Write $U=\phi(X)$, $Z=\phi(X^+)$, and $J=c(X)$. For a norm budget $\kappa\geq0$, define

$$
R_\kappa=\inf_{\|B\|_2\leq\kappa}\sqrt{\mathbb E\|Z-B^\top U\|_2^2}.
$$

Here $\|B\|_2$ is the spectral norm. $R_\kappa$ concerns one-step vector RMSE under the specified transition law. It is not the error of a particular fitted matrix.

The evaluation sample consists of $M$ independent, identically distributed pairs $(U_i,Z_i,J_i)$. The representation, reference map, and any externally supplied prototypes must be fixed independently of this sample (for example, using separate training and selection data). A sequence of adjacent transitions from one trajectory does not in general satisfy the independent-pair requirement.

## 2. Population obstruction

Define the within-reference-class successor variation, current-coordinate dispersion, and label deviation as

$$
\begin{aligned}
s&=\mathbb E\|Z-\mathbb E[Z\mid J]\|_2^2,\\
t&=\mathbb E\|U-\mathbb E[U\mid J]\|_2^2,\\
f&=\mathbb E(1-U_J)^2.
\end{aligned}
$$

The conditional mean is the minimum-MSE predictor using only $J$. Conditional centering and the reverse triangle inequality in $L^2$ yield

$$
R_\kappa\geq[\sqrt{s}-\kappa\sqrt{t}]_+
\geq[\sqrt{s}-\kappa\sqrt{2f}]_+,
$$

where $[a]_+=\max(0,a)$. The second inequality follows from $t\leq\mathbb E\|U-e_J\|_2^2\leq 2f$, with $e_J$ the one-hot reference vector.

For fixed simplex-valued class prototypes $h_j$, define $q=\tfrac12\mathbb E\|U-h_J\|_2^2$. The same conditional-mean optimality gives $t\leq2q$ and the prototype-based bound $[\sqrt{s}-\kappa\sqrt{2q}]_+$.

These inequalities quantify how much within-class successor variation can remain unexplained by a bounded linear map. They do not require that the coordinates originate from a particular reconstruction method.

## 3. Finite-sample lower confidence bounds

For each occupied reference class $j$, compute the sample means $\overline U_j$ and $\overline Z_j$. Define

$$
\begin{aligned}
\widehat s&=\frac1M\sum_i\|Z_i-\overline Z_{J_i}\|_2^2,\\
\widehat t&=\frac1M\sum_i\|U_i-\overline U_{J_i}\|_2^2,\\
\widehat f&=\frac1M\sum_i(1-U_{i,J_i})^2,\\
\widehat q&=\frac1{2M}\sum_i\|U_i-h_{J_i}\|_2^2 .
\end{aligned}
$$

All denominators are $M$; empty classes contribute zero, as do singleton classes to the centered statistics. For $0<\delta<1$, set $r=\log(2/\delta)/M$ and

$$
\begin{aligned}
V_\delta&=[\widehat s-\sqrt{2r}]_+,\\
F_\delta&=\min(1,\widehat f+r+\sqrt{r^2+2r\widehat f}),\\
T_\delta&=\min(1,\widehat t+\sqrt{2r}+C/M),\\
Q_\delta&=\min(1,\widehat q+r+\sqrt{r^2+2r\widehat q}).
\end{aligned}
$$

The implemented certificate formulas are

$$
\begin{aligned}
L_{\rm label}&=[\sqrt{V_\delta}-\kappa\sqrt{2F_\delta}]_+,\\
L_{\rm centered}&=[\sqrt{V_\delta}-\kappa\sqrt{T_\delta}]_+,\\
L_{\rm prototype}&=[\sqrt{V_\delta}-\kappa\sqrt{2Q_\delta}]_+.
\end{aligned}
$$

To see the confidence mechanism, the empirical centered dispersion is the minimum of the average squared residual over class-specific simplex means. Replacing one observation changes that minimum by at most $2/M$, since squared distances between simplex points are at most two. Its expected value is no larger than the corresponding population dispersion. For the current-coordinate statistic the downward bias is at most $C/M$: conditional on an occupied class of size $n$, the expected unnormalized residual sum is $(n-1)$ times the class variance, which is at most one. One-sided bounded-difference inequalities therefore give $V_\delta\leq s$ and $t\leq T_\delta$ with individual failure probabilities at most $\delta/2$. For a mean of independent losses in $[0,1]$, the one-sided lower-tail inequality $\Pr(\widehat f<f-\sqrt{2fr})\leq e^{-Mr}$, inverted for $f$, gives $f\leq F_\delta$ at the same confidence; the prototype loss has the same range and yields $q\leq Q_\delta$. Combining the successor event with the appropriate current-coordinate event gives the stated bounds by a union bound.

Each method has confidence at least $1-\delta$ under its assumptions. The event underlying each dispersion-based bound holds for every nonnegative $\kappa$ for the fixed representation and evaluation law. This is not a simultaneous guarantee across alternative representations, sample sizes, or evaluation laws selected from the data.

For comparison, if $\underline m_\kappa$ is a certified lower bound on the constrained minimum **empirical** MSE, a generic pointwise empirical-risk bound is

$$
L_{\rm ERM}=\sqrt{\left[\underline m_\kappa-(1+\kappa)^2
\sqrt{\frac{\log(1/\delta)}{2M}}\right]_+}.
$$

Indeed, the loss is bounded by $(1+\kappa)^2$ on the spectral-norm ball. The minimum empirical loss has expectation no larger than the minimum population loss and changes by at most $(1+\kappa)^2/M$ upon replacement of one pair. The one-sided bounded-difference inequality controls this minimum without requiring a union bound over fitted matrices. The empirical optimum must be lower-bounded, not replaced with the objective attained by an arbitrary feasible fitted model. That attained empirical objective is an upper bound on the empirical minimum. For several prespecified methods or pointwise ERM budgets, allocate the failure probabilities so that their sum does not exceed the desired joint failure probability.

## 4. Interpretation and scope

If $L>\epsilon$, the confidence statement excludes population RMSE at most $\epsilon$ for every predictor in the specified class. The strict inequality is required. $L=0$ is inconclusive. Larger spectral-norm limits weaken the dispersion-based certificates; a certificate at a budget smaller than a fitted matrix's norm does not apply to that matrix.

The finite-sample guarantee concerns the random evaluation dataset, not the conditional probability that any specific realized bound is correct. A fitted predictor's empirical error is neither the exact optimal population error nor a certified population upper bound without a separate statistical argument.

Under deterministic dynamics $F$, exact pointwise closure $\phi\circ F=B^\top\phi$ implies that the coordinate-function span is invariant under $g\mapsto g\circ F$. If $v^\top\phi=0$ as a function, the same relation holds after composition; the restricted operator is consequently well-defined on the coefficient space modulo the redundant directions. The kernel $k_\phi(x,y)=\phi(x)^\top\phi(y)$ induces the corresponding finite-dimensional function space. This statement is conditional on exact closure and is separate from approximate forecasting and finite-sample risk certification.

## 5. Certificate computation

The [primary implementation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/0bbb2bafa107892e32e221e882326890b14fe36e) contains `kahkm_certificate_statistics.py` and `kahkm_certificates.py`. The [four-method implementation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2/reproduction/posthoc_four_state_erm) contains `scripts/certificate_reanalysis.py`. Use the pinned source checkout specified in [Reproduction](SOFT_STATE_REPRODUCIBILITY.md).

The following illustrative calculation requires NumPy and the primary implementation at the checkout root. It illustrates a fixed two-class representation with independent uniform draws of four discrete states; no prediction matrix is fitted.

```python
import numpy as np
from kahkm_certificate_statistics import statistics_from_pairs
from kahkm_certificates import certificate_from_statistics

p = 31 / 32
h = np.array([[p, 1-p], [1-p, p]])
# States a,b have reference class 0; d,e have class 1.
# Conflicting successors have class indices 0,1,1,1.
current_class = np.array([0, 0, 1, 1])
next_class = np.array([0, 1, 1, 1])
rng = np.random.default_rng(42)
states = rng.integers(0, 4, size=32768)

stats = statistics_from_pairs(
    phi=h[current_class[states]],
    successors=h[next_class[states]],
    labels=current_class[states],
)
bound = certificate_from_statistics(
    f_hat=stats.f_hat, s_hat=stats.s_hat, n_pairs=stats.n_pairs,
    kappa=np.sqrt(2), delta=0.05,
)
print("Lower confidence bound:", bound.L_kappa_delta)
print("Exact population optimum:", p - 0.5)
```

The pair arrays have shape $(M,C)$, contain finite simplex coordinates, and use zero-based integer labels of shape $(M,)$. Current-state labels must not be inferred from successor states. The primary statistics function requires coordinates in $[0,1]$ and row sums within $10^{-12}$ of one; it does not silently clip or renormalize them.

## 6. Experimental designs and observations

### Controlled four-state distribution

The input law is uniform on four states $a,b,d,e$. The two reference classes are $\{a,b\}$ and $\{d,e\}$, with coordinates $h_1=(p,1-p)^\top$ and $h_2=(1-p,p)^\top$. In the identity case, $R_{\sqrt2}=0$. In the conflicting-successor case the successor classes of $a,b,d,e$ are respectively $1,2,2,2$. For the conflicting case, the column-prediction matrix $A=B^\top$ attaining the population class means is $A=\tfrac12 I+\tfrac12 h_2\mathbf1^\top$. For any $\alpha\in[0,1]$, the matrix $\alpha I+(1-\alpha)h_2\mathbf1^\top$ maps the simplex into itself and has spectral norm at most $\sqrt2$ (by the triangle inequality and $\|h_2\|_2\leq1$). Thus the population class-conditional mean predictor belongs to the constrained class. The empirical class means are attainable in the same manner with $\alpha=n_a/(n_a+n_b)$ when the first class is occupied, and any suitable $\alpha$ otherwise. Consequently

$$
R_{\sqrt2}=p-\tfrac12,\qquad
L_{\rm label,pop}=[3p-\tfrac52]_+.
$$

The population centered bound equals the optimum since $t=0$. The label-based population bound can vanish even when the optimum is positive. The empirical optimum at $\kappa=\sqrt2$ is $\widehat s$ in squared-error units. For conflicting successors it is

$$
\widehat s=\frac{2(2p-1)^2 n_a n_b}{M(n_a+n_b)}
$$

when $n_a+n_b>0$, and zero otherwise.

The [configuration](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/certificates/configs/four_state_final_v1.json) specifies 13 cases, nine sample sizes from 128 to 524,288, 1,000 independent count datasets per case and size, $\kappa=\sqrt2$, and six exclusion tolerances. The complete archive contains 117,000 count records. At $\delta=0.05$, 49,271 original label-based certificates are zero and none exceeds the analytical population optimum. These are observed frequencies, not proof of zero theoretical failure probability.

The four-method comparison uses the same count datasets with $\delta=0.0125$ per method, giving joint confidence of at least 95% within one prespecified evaluation dataset. The fixed prototypes are design-known. No numerical empirical-risk optimizer is needed for this construction. Selected mean lower bounds for $p=31/32$ (true optimum 0.46875) are:

| $M$ | Label | Centered | Fixed prototype | ERM |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.1289 | 0.0000 | 0.1468 | 0.0000 |
| 4,096 | 0.2825 | 0.0951 | 0.3126 | 0.2913 |
| 8,192 | 0.3228 | 0.1635 | 0.3593 | 0.3529 |
| 32,768 | 0.3671 | 0.2616 | 0.4144 | 0.4148 |

Source: [four-method output directory](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/78a1a5e7ea9d3a2fecd3b335bbe5a9f3f2237eb2/reproduction/posthoc_four_state_erm/outputs/historical_117k). Comparisons of realized lower bounds do not imply uniform dominance of one method.

### Van der Pol oscillator

The evaluation concerns the oscillator $\dot x_1=x_2$, $\dot x_2=(1-x_1^2)x_2-x_1$, with fourth-order Runge–Kutta integration at step size 0.02. A fixed 25-class KAHM model uses $\omega=128$ and $\tau=10^{-6}$. Its matrix is fitted using three 1,200-transition training trajectories, $\beta=0.1$, 20 ordered NLMS epochs and identity initialization. The reference label is the nearest stored state-space cluster center. The model and evaluation design are fixed independently of confirmatory data.

Each retained evaluation pair is obtained from a separate trajectory draw with an independent uniform time index in $\{0,\ldots,1199\}$. The initial state is sampled near $(2,0)$ with Gaussian standard deviation 0.05 per coordinate. Two transition laws use, respectively, Gaussian process noise with standard deviation $10^{-5}$ per coordinate after each integration step, or no process noise. The configuration specifies 32 datasets of 16,384 independent pairs per law, for 64 datasets in total.

The label-based bound is evaluated at $\delta=0.05$ and the spectral norm of the fixed fitted matrix, $\kappa=1.0203452494452896$.

| Evaluation law | Positive bounds | Mean lower bound | Mean fitted-matrix RMSE |
| --- | ---: | ---: | ---: |
| Matched stochastic | 32/32 | 0.08457 | 0.27631 |
| Deterministic | 32/32 | 0.08445 | 0.27596 |

Source: [mode_summary.csv](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/prospective_campaigns/dynamical_original_iid_confirmatory_v1_run001/summary_v1/mode_summary.csv). These results establish nonvacuous certificates for the specified representation, norm budget and sampling laws under the stated assumptions. They do not identify the minimum population RMSE or establish exact closure for the oscillator. See the [configuration](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/0bbb2bafa107892e32e221e882326890b14fe36e/reproduction/certificates/configs/dynamical_confirmatory_v1.json) and [reproduction guide](SOFT_STATE_REPRODUCIBILITY.md) for the data and verification procedures.

## 7. Reference

M. Kumar and S. Kargaran (2026). [Prediction Limits and Koopman Closure of Geometry-Induced Soft State Abstractions](https://arxiv.org/abs/2609.32652v3), arXiv:2609.32652. The definitions, formulas, assumptions and computational instructions above are sufficient to interpret the repository's certificate calculations independently of the article.
