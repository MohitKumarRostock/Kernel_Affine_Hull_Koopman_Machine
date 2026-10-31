# Mathematical background for reconstruction-based forecasting

This guide defines the local reconstruction maps, explains when their feature spans agree or differ, and derives the bound used for the forecast correction. The results concern specified feature maps and decoders. They do not imply that accurate reconstruction alone guarantees accurate forecasting.

All vectors in this guide are columns. A superscript $\top$ denotes transpose. The norm $\|\cdot\|$ is the Euclidean norm for vectors and its induced operator norm for matrices unless a different norm is specified. The span of a collection of scalar functions means all finite linear combinations of those functions with coefficients that do not depend on the query.

## 1. Local reconstruction maps

Consider a fixed bank of $M$ reference histories $q_1,\ldots,q_M\in\mathbb R^p$. Let $Q\in\mathbb R^{M\times p}$ have rows $q_i^\top$, let $\mu\in\mathbb R^p$ be the reference mean, and let $P\in\mathbb R^{p\times r}$ contain $r$ orthonormal principal directions. The matrix $W\in\mathbb R^{r\times r}$ is the fixed positive-definite precision matrix in the projected coordinates.

For a history $x\in\mathbb R^p$, define the projected difference $z_i(x)=P^\top(x-q_i)$ and the Gaussian kernel vector $k(x)\in\mathbb R^M$ by

$$
k_i(x)=\exp\!\left[-\frac{1}{2r}z_i(x)^\top Wz_i(x)\right].
$$

The Gram matrix $K\in\mathbb R^{M\times M}$ has entries $K_{ij}=k_i(q_j)$. Given a regularization parameter $\lambda>0$, set $C=(K+\lambda I_M)^{-1}$, where $I_M$ is the $M$-dimensional identity matrix. The Gaussian Gram matrix and $C$ are symmetric. Let $\mathbf1_M$ be the vector of $M$ ones.

The KAHM, normalized RBF, and affine reconstructions are, respectively,

$$
A_K(x)=\frac{Q^\top Ck(x)}{\mathbf1_M^\top Ck(x)},\qquad
A_R(x)=\frac{Q^\top k(x)}{\mathbf1_M^\top k(x)},\qquad
A_A(x)=\mu+PP^\top(x-\mu).
$$

The KAHM expression is defined wherever its denominator is nonzero. The RBF denominator is positive for every finite query in exact arithmetic. A common positive multiplier applied to all entries of $k(x)$ cancels from both ratios. This permits a numerically stable evaluation of the kernel row by subtracting its largest log-kernel value before exponentiation.

For multiple banks indexed by $j$, let $g_j(x)$ be a common scalar locality gate. The shared features contain $g_j(x)$ and the coordinates of $g_j(x)P_j^\top(x-\mu_j)$. The added reconstruction departures are

$$
d_{K,j}(x)=g_j(x)\{A_{K,j}(x)-A_{A,j}(x)\},\qquad
d_{R,j}(x)=g_j(x)\{A_{R,j}(x)-A_{A,j}(x)\}.
$$

Each departure has $p$ coordinates. The corresponding affine departure is identically zero. The forecasting experiment compares predictors with the same common features and different departure blocks.

## 2. Normalized target transfer

Where the signed denominator is nonzero, define the weight vector

$$
\omega(x)=\frac{Ck(x)}{\mathbf1_M^\top Ck(x)}.
$$

Then $\mathbf1_M^\top\omega(x)=1$ and $A_K(x)=Q^\top\omega(x)$. The weights sum to one but need not be nonnegative.

For reference targets $y\in\mathbb R^M$, direct transfer gives

$$
\omega(x)^\top y=
\frac{k(x)^\top(K+\lambda I_M)^{-1}y}
     {k(x)^\top(K+\lambda I_M)^{-1}\mathbf1_M}.
$$

The numerator is the dual kernel-ridge prediction fitted to $y$; the denominator is the same prediction fitted to the constant target. Both expressions use the displayed regularization parameter and no additional unpenalized intercept. Thus normalized target transfer is exactly a ratio of these two kernel-ridge predictions. It is generally different from the unnormalized numerator alone.

If the target is affine, $f(q)=q^\top b+c$, where $b\in\mathbb R^p$ and $c\in\mathbb R$, then

$$
\sum_{i=1}^{M}\omega_i(x)f(q_i)-f(x)
=\{A_K(x)-x\}^\top b.
$$

To obtain this identity, expand the left-hand side, use $\sum_i\omega_i=1$ to cancel the constants, and use $\sum_i\omega_iq_i=A_K(x)$. The identity applies to affine targets. It does not extend to an arbitrary nonlinear conditional mean without additional assumptions.

## 3. A sufficient condition for equal raw feature spans

Suppose $K\mathbf1_M=\kappa\mathbf1_M$ for a scalar $\kappa$, so the Gram matrix has constant row sums. Then

$$
C\mathbf1_M=a\mathbf1_M,\qquad a=(\kappa+\lambda)^{-1}>0.
$$

Assume also that the column span of $[\mathbf1_M,Q]$ is invariant under $C$. Consequently there exist $B\in\mathbb R^{p\times p}$ and $c\in\mathbb R^p$ such that

$$
CQ=QB+\mathbf1_Mc^\top.
$$

Substitution into the reconstruction formula gives

$$
A_K(x)=\frac{B^\top A_R(x)+c}{a}.
$$

Indeed, symmetry of $C$ gives $Q^\top Ck=B^\top Q^\top k+c\mathbf1_M^\top k$, while the denominator is $a\mathbf1_M^\top k$. Cancelling the positive common factor proves the formula.

The restriction of the invertible linear map $C$ to its invariant, finite-dimensional subspace is invertible. After removing dependent columns of the augmented reference coordinates, the displayed transformation therefore gives an invertible change of the active augmented coordinate functions. The scalar coordinate functions of $[1,A_K]$ and $[1,A_R]$ have equal spans.

For a common gate $g$, the departure relation is

$$
d_K=\frac{B^\top}{a}d_R+
\left(\frac{B^\top}{a}-I_p\right)gA_A+\frac{gc}{a}.
$$

Every coordinate of $gA_A=g\mu+P\{gP^\top(x-\mu)\}$ lies in the common-feature span. Thus adding the KAHM or RBF departure to those common features yields the same raw function span under the stated conditions.

For example, take two scalar references $-1$ and $1$, a unit gate, and $A_A(x)=x$. Write

$$
K=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix},\qquad
c_0=\frac{1+\rho+\lambda}{1-\rho+\lambda}.
$$

The constant vector and the reference vector are eigenvectors of $K$, with eigenvalues $1+\rho$ and $1-\rho$. Hence $A_K=c_0A_R$, and subtracting $x$ gives $d_K=c_0d_R+(c_0-1)x$. Therefore $\operatorname{span}(1,x,d_K)=\operatorname{span}(1,x,d_R)$.

Equal raw spans need not yield equal fitted predictions. An isotropic ridge penalty changes under a nonorthogonal coordinate transformation, and applying separate nonlinear transforms to the feature blocks can change the span relation itself. The empirical experiment consequently compares the complete normalized and regularized procedures.

## 4. An irregular reference set with distinct raw features

Consider three scalar references $q_1<q_2<q_3$ satisfying $q_1+q_3\ne2q_2$. Set $q=(q_1,q_2,q_3)^\top$, use the Gaussian kernel $k_i(x)=\exp[-t(x-q_i)^2/2]$ with $t>0$, and let $C=(K+\lambda I_3)^{-1}$. Assume the vector $v=C\mathbf1_3$ has strictly positive entries. Use a unit gate and the affine reconstruction $A_A(x)=x$.

Under these assumptions,

$$
A_K\notin\operatorname{span}(1,x,A_R),\qquad
d_K\notin\operatorname{span}(1,x,d_R).
$$

**Proof.** Let $u=Cq$. Suppose first that $A_K=cA_R+d$ for constants $c,d$. Cross-multiplying the two denominators yields

$$
\{k(x)^\top(u-dv)\}\{k(x)^\top\mathbf1_3\}
=c\{k(x)^\top q\}\{k(x)^\top v\}.
$$

The six products indexed by $i\le j$ have the form

$$
k_i(x)k_j(x)=e^{-tx^2}e^{-t(q_i^2+q_j^2)/2}e^{t(q_i+q_j)x}.
$$

All six pair sums $q_i+q_j$ are distinct under the spacing assumption. The corresponding real exponentials are linearly independent: differentiating a zero linear combination at zero through order five produces a Vandermonde system with distinct nodes, whose matrix is invertible. After cancelling $e^{-tx^2}$, coefficients of these six products can therefore be compared separately.

Writing $w=u-dv$, the diagonal and off-diagonal coefficients give

$$
w_i=cq_iv_i,\qquad w_i+w_j=c(q_iv_j+q_jv_i)\quad(i<j).
$$

If $c=0$, then $u=dv$. Invertibility of $C$ would imply $q=d\mathbf1_3$, contradicting distinct references. If $c\ne0$, substitution of the diagonal equations into the off-diagonal equations gives $(q_i-q_j)(v_i-v_j)=0$. Thus all entries of $v$ must be equal. The equation $(K+\lambda I_3)v=\mathbf1_3$ would then imply constant Gram row sums. This is impossible: the row sum at the middle reference is strictly larger than either endpoint row sum, because its remaining neighbor is closer than the opposite endpoint and the Gaussian kernel strictly decreases with distance. Thus no affine relation $A_K=cA_R+d$ exists.

Both reconstructions are bounded on the real line. The RBF map is a convex combination of the references. For KAHM,

$$
A_K(x)=\sum_{i=1}^{3}
\frac{k_i(x)v_i}{\sum_{j=1}^{3}k_j(x)v_j}\frac{u_i}{v_i}.
$$

The coefficients are positive and sum to one, so the map is bounded by the three finite values $u_i/v_i$. In an identity $A_K=a+bx+cA_R$, boundedness therefore forces $b=0$, which reduces to the excluded affine relation. Finally, any relation $d_K=a+bx+cd_R$ would imply $A_K=a+(b+1-c)x+cA_R$, also impossible. This proves both assertions.

The source-compatible example $q=(-1,0,2)^\top$ uses $t=3/7$ and $\lambda=4.498274743339644$. Positivity can be certified without inspecting inverse entries individually. Write $K=I_3+E$, let $r_E=\|E\|_\infty$ be the maximum absolute row sum, and put $a_0=1+\lambda$. If $a_0>2r_E$, then $C=a_0^{-1}\sum_{m=0}^{\infty}(-E/a_0)^m$. The matrix infinity norm is submultiplicative and $\|\mathbf1_3\|_\infty=1$, so the norm of the terms with $m\ge1$ is bounded by the geometric series $a_0^{-1}\sum_{m\ge1}(r_E/a_0)^m$. Consequently,

$$
\left\|C\mathbf1_3-a_0^{-1}\mathbf1_3\right\|_\infty
\le\frac{r_E}{a_0(a_0-r_E)}<a_0^{-1}.
$$

Here the vector infinity norm is the largest absolute entry. Every entry of $C\mathbf1_3$ is consequently positive. For the stated example, $r_E=1.2314905926823394$ and $a_0=5.498274743339644$, so the condition holds.

This example rules out universal redundancy of KAHM departures. It does not establish that every fitted bank has distinct features, or that distinctness yields a forecasting advantage.

## 5. Bounded features and a ridge correction bound

For a finite feature vector $v\in\mathbb R^d$ and a radius $B>0$, define

$$
T_B(v)=\frac{v}{\sqrt{1+\|v\|^2/B^2}}.
$$

Its norm is smaller than $B$. If $a_v=(1+\|v\|^2/B^2)^{-1/2}$, its derivative is

$$
DT_B(v)=a_vI_d-a_v^3vv^\top/B^2.
$$

The derivative acts by $a_v^3$ in the radial direction and by $a_v$ in every orthogonal direction. Both are at most one. Integrating the derivative along the line segment joining any two feature vectors bounds the norm of their transformed difference by the norm of the original difference. The map is therefore nonexpansive. This statement concerns its feature-vector input; preceding kernel evaluations and coordinate standardization require their own sensitivity analysis.

The experiment fits coordinate centers and scales on A only and applies the radial map separately to each feature block. A block of width $d_b$ has radius $B_b=\sqrt{d_b}$. The concatenated feature vector $\psi(x)\in\mathbb R^d$, where $d=\sum_bd_b$, satisfies

$$
\|\psi(x)\|\le R,\qquad R=\sqrt{\sum_bB_b^2}=\sqrt d.
$$

For one forecast horizon, let $e_i$ be the training residual of the raw-input baseline on A row $i$, and let $n$ be the number of fitting rows. The residual predictor minimizes

$$
\frac1n\sum_{i=1}^{n}\{e_i-a-\psi(x_i)^\top w\}^2+\alpha\|w\|^2,
\qquad \alpha>0,
$$

over an unpenalized intercept $a\in\mathbb R$ and coefficients $w\in\mathbb R^d$. Define the means $\bar e=n^{-1}\sum_ie_i$ and $\bar\psi=n^{-1}\sum_i\psi(x_i)$. Let $e_c\in\mathbb R^n$ have entries $e_i-\bar e$, and let $X_c\in\mathbb R^{n\times d}$ have rows $\{\psi(x_i)-\bar\psi\}^\top$.

Minimizing over the intercept gives $a=\bar e-\bar\psi^\top w$. Substitution and differentiation with respect to $w$ yield

$$
w=(X_c^\top X_c+n\alpha I_d)^{-1}X_c^\top e_c,
\qquad h(x)=\bar e+\{\psi(x)-\bar\psi\}^\top w.
$$

The factor $n\alpha$ is required because the objective uses the mean, rather than the sum, of squared residuals. To bound $w$, use a singular-value decomposition of $X_c$. For every singular value $s\ge0$,

$$
\frac{s}{s^2+n\alpha}\le\frac{1}{2\sqrt{n\alpha}},
$$

because $(s-\sqrt{n\alpha})^2\ge0$. Orthogonality of the singular vectors gives

$$
\|w\|\le\frac{\|e_c\|}{2\sqrt{n\alpha}}
=\frac{\sigma_e}{2\sqrt\alpha},\qquad
\sigma_e=\sqrt{\frac1n\sum_i(e_i-\bar e)^2}.
$$

Cauchy–Schwarz and the feature bound then imply

$$
|h(x)|\le|\bar e|+(R+\|\bar\psi\|)\frac{\sigma_e}{2\sqrt\alpha}
\le|\bar e|+\frac{R\sigma_e}{\sqrt\alpha}.
$$

If the complete forecast is the baseline plus $\gamma h(x)$, the correction bound is multiplied by $|\gamma|$. Each horizon uses its own selected penalty, residuals, and multiplier. The result is generic bounded-feature ridge algebra and applies to the matched common-feature and RBF predictors as well as KAHM. It controls correction magnitude, not forecasting accuracy, the raw baseline, or stability across geometry updates.

## 6. Numerical checks and scientific records

The [original algebra record](ALGEBRA.md) and `algebra_results.json` are part of the immutable prospective scientific checkpoint. Their numerical examples check exact identities and the stated bounds on fabricated inputs. Numerical ranks and finite-grid residuals supplement the analytic arguments; they do not replace proofs of global function-space statements.

The experiment also checks signed denominators before forming KAHM reconstructions. For unnormalized weights $w_0=Ck(x)$, its relative guard requires finite calculations and

$$
\frac{|\mathbf1_M^\top w_0|}{\|w_0\|_1}>\tau,
$$

where $\|w_0\|_1=\sum_i|w_{0,i}|$ and $\tau>0$ is the configured tolerance. This implies $\|\omega(x)\|_1<1/\tau$ and therefore $\|A_K(x)\|<\max_i\|q_i\|/\tau$. With $\tau=10^{-10}$, that guard is primarily a numerical safeguard; the radial feature map supplies the substantially smaller decoder correction bound.

The implemented fallback for an unsafe reconstruction is specified in the protocol and counted in the results. No such local fallback occurred in the principal experiment. These safeguards do not diagnose the earlier online extrapolation failures: a previously examined large KAHM forecast had a safe denominator.
