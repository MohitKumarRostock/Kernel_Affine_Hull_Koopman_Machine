# Scientific reference

## 1. Problem and notation

An observed scalar series supplies a history $x_t=(y_{t-d+1},\ldots,y_t)^\top$ and direct targets $y_{t+h}$ for horizons $h\in\mathcal H$. For the observed CO2 series the input instead consists of recent and seasonal changes, as specified in [Experimental protocols](PROTOCOLS.md). Let $z\in\mathbb R^d$ be the input transformed using training-only centering and scaling. A local reference partition $b$ has indices $\mathcal J_b$; the total number of local models is $J$. Model $j$ has reference matrix $Q_j$ (one history per row), mean $\mu_j$, and orthonormal retained principal directions $P_j\in\mathbb R^{d\times r_j}$. The stacked projection matrix is $\mathcal P=\operatorname{col}_j P_j^\top$, where $\operatorname{col}$ denotes vertical concatenation. All results about information recovery and affine function spaces concern fixed fitted reference models and exact arithmetic.

## 2. Reconstruction and membership features

For $r_j>0$, let $S_j$ be the projected sample covariance with divisor $n_j-1$. Define the kernel

$$
\kappa_j(z,z')=\exp\left[-\frac{(P_j^\top(z-z'))^\top S_j^{-1}P_j^\top(z-z')}{2r_j}\right].
$$

Write $k_j(z)$ for query-to-reference similarities, $G_j$ for the Gram matrix, and $C_j=(G_j+\lambda_j I)^{-1}$ for a positive reference-fitted regularizer. Three reconstructions use the same reference histories and bases:

$$
\mathcal A_j^K(z)=\frac{Q_j^\top C_jk_j(z)}{\mathbf1^\top C_jk_j(z)},\qquad
\mathcal A_j^R(z)=\frac{Q_j^\top k_j(z)}{\mathbf1^\top k_j(z)},\qquad
\mathcal A_j^A(z)=\mu_j+P_jP_j^\top(z-\mu_j).
$$

Here K is KAHM, R is positive-kernel averaging, and A is affine projection. A fourth control, `center`, returns $\mu_j$. Rank-zero models use the affine reconstruction for K and R. In numerical evaluation, common rescaling of query similarities prevents all similarities from underflowing; the signed KAHM quotient requires finite weights and a nonzero denominator satisfying $|\mathbf1^\top C_j k_j|/\|C_j k_j\|_1\ge 10^{-10}$. Failed checks use the affine reconstruction. A negative denominator alone is not a failure.

Given $a=\mathcal A_j^m(z)$, let $e=1-\exp(-\|z-a\|_2)$ and $\theta=\arccos(\operatorname{clip}(c,-1,1))/\pi$, where $c=z^\top a/(\|z\|_2\|a\|_2)$ if both norms are positive and $c=0$ otherwise. The score is $T_j^m=\sqrt{(e^2+\theta^2)/2}\in[0,1]$. Exact nonzero reconstruction gives score zero; exact reconstruction at zero gives $1/(2\sqrt2)$. Smaller scores give larger affinities. With offset $\tau>0$ and contrast $\omega>0$, define for $j\in\mathcal J_b$

$$
Z_b^m=\sum_{i\in\mathcal J_b}(1-T_i^m+\tau)^\omega,\qquad
\phi_j^m=(1-T_j^m+\tau)^\omega/Z_b^m,\qquad \zeta_b^m=\log Z_b^m.
$$

The memberships are positive and sum to one in each partition. They describe relative reconstruction affinity, not calibrated regime probabilities. Suppress superscript $m$ below and let $u_j=P_j^\top(z-\mu_j)$, $v_j=\phi_j u_j$.

| Label | Encoding | Width |
| --- | --- | --- |
| Mem | $\phi$ | $J$ |
| Norm | $(\phi,\zeta)$ | $J+B_0$ |
| Score | $T$ | $J$ |
| Local | $(\phi,\operatorname{col}_j v_j)$ | $J+\sum_j r_j$ |
| Sep | $(\phi,\operatorname{col}_j u_j)$ | $J+\sum_j r_j$ |
| Raw+Mem | $(\phi,z)$ | $J+d$ |
| Gated raw | $(\phi,\operatorname{col}_j\phi_j z)$ | $J+Jd$ |
| Raw | $z$ | $d$ |

$B_0$ is the number of partitions. Width counts stored coordinates, not independent dimensions. Norm and Score are included in the archived study; the executable weighting comparisons use Mem, Local, Sep, Raw+Mem, Gated raw and Raw.

## 3. Score recovery and input recovery

The memberships alone lose affinity scale. The log normalizer restores it exactly:

$$T_j=1+\tau-\exp((\log\phi_j+\zeta_b)/\omega).$$

The inverse is obtained directly by taking logarithms of the membership definition. For a partition of $q$ models, $\phi_j\ge \tau^\omega/[\tau^\omega+(q-1)(1+\tau)^\omega]$. This follows from the smallest numerator and largest competing affinities.

For a membership vector $p$ and stacked position blocks $w_j$, define

$$
A(p)=\operatorname{col}_j(p_jP_j^\top),\qquad
b(p,w)=\operatorname{col}_j(w_j+p_jP_j^\top\mu_j).
$$

When all memberships are positive and $\mathcal P$ has column rank $d$, $A(\phi)z=b(\phi,v)$ has a unique solution $z=A(\phi)^\dagger b(\phi,v)$, where $\dagger$ denotes the Moore–Penrose inverse. In particular, Local retains the complete input. For perturbed coordinates $(\widehat\phi,\widehat v)$ with $s(\widehat\phi)>0$, where $s(p)=\sigma_{\min}(A(p))$ (zero for rank deficiency),

$$
\|\widehat z-z\|_2\le
\frac{\|\widehat v-v\|_2+M_z\|\widehat\phi-\phi\|_2}{s(\widehat\phi)},\qquad
M_z=\max_j\|u_j\|_2.
$$

Indeed, the residual of $z$ in the perturbed system has blocks $(\widehat v_j-v_j)-(\widehat\phi_j-\phi_j)u_j$; multiplying by the pseudoinverse yields the bound. The spectral perturbation bound $s(\widehat\phi)\ge s(\phi)-\|\widehat\phi-\phi\|_2$ follows from the norm of the stacked perturbation. Also $s(\phi)\ge (\min_j\phi_j)\sigma_{\min}(\mathcal P)$. If every local basis has rank $d$, then $A(\phi)^\top A(\phi)=(\sum_j\phi_j^2)I_d$, so $s(\phi)^2\ge\sum_b1/|\mathcal J_b|$. Small memberships alone do not imply poor recovery conditioning. These bounds assume exact fixed bases and means and exact solution of the perturbed least-squares problem; they are not interval-arithmetic guarantees. Conditioning of this system is not necessarily the conditioning of every inverse of the complete encoding.

## 4. Affine accessibility and predictive alignment

Let $h(z)=\mathbb E[Y\mid Z=z]$, $\mathbb E[Y^2]<\infty$, and $R^*=\mathbb E\operatorname{Var}(Y\mid Z)$. For any measurable predictor $f$ with finite squared risk,

$$\mathbb E(Y-f(Z))^2=R^*+\mathbb E(h(Z)-f(Z))^2.$$

An affine decoder of Local has the form

$$f(z)=a+\sum_{j=1}^J\phi_j(z)[a_j+w_j^\top P_j^\top(z-\mu_j)].$$

The geometry determines gates; supervised fitting determines local prediction coefficients. Sep and Local determine one another with positive memberships; they have identical width but generally different affine predictor spaces. If $\mathcal P$ has full column rank, Sep and Raw+Mem generate identical affine spaces. Gated raw contains the Local space. Fitted coordinate standardization preserves affine spaces, but changes ridge penalties; equal width or affine dimension does not imply equal regularized complexity.

### Reconstruction-margin bound

Choose one partition of $q\ge2$ local models, and a measurable predictive assignment $s(z)$ in this partition. Suppose $h(z)=\ell_{s(z)}(z)+r(z)$, with $\ell_j(z)=c_j+\beta_j^\top(z-\mu_j)$. Define the projected law $\widetilde\ell_j(z)=c_j+\beta_j^\top P_jP_j^\top(z-\mu_j)$, the projection error $E_\perp=|\ell_s-\widetilde\ell_s|$, the disagreement $M=\max_{j\in\mathcal J_b}|\widetilde\ell_j-\widetilde\ell_s|$, and the reconstruction margin $\delta(z)=\min_{j\ne s}(T_j(z)-T_s(z))$ within this partition. The optimal affine Local risk $R_{\rm Local}$ satisfies

$$
\sqrt{R_{\rm Local}-R^*}\le \|r\|_{L^2}+\|E_\perp\|_{L^2}+\|(1-\phi_s)M\|_{L^2}.
$$

The candidate $\sum_{j\in\mathcal J_b}\phi_j\widetilde\ell_j$ belongs to the Local affine class (ignore all other partitions). Subtracting it from $h$ and using the normalization and triangle inequality gives the displayed bound. If $\delta\ge\delta_0>0$ almost surely, then the ratio $\phi_j/\phi_s\le(1-\delta_0/(1+\tau))^\omega\le\exp[-\omega\delta_0/(1+\tau)]$, so

$$\|(1-\phi_s)M\|_{L^2}\le(q-1)e^{-\omega\delta_0/(1+\tau)}\|M\|_{L^2}.$$

All norms are under the law of $Z$; finite right sides are required. A sufficient condition for strict improvement over raw affine prediction is that the squared upper bound is below $\inf_{a,b}\mathbb E[(h(Z)-a-b^\top Z)^2]$. Alignment is an assumption on the conditional prediction law and geometry, not a consequence of unsupervised reconstruction. Sharpening memberships reduces the gating-error upper bound under positive margins but does not ensure monotonicity of optimal or fitted risk. The result is a population affine approximation statement, not a finite-sample ridge guarantee.

### Exact matched-dimensional example

Let $u>v>0$, $u^2+v^2=2$. Draw an equiprobable regime $S\in\{1,2\}$ and independent signs $\xi_1,\xi_2$. Let $Z=(u\xi_1,v\xi_2)^\top$ in regime 1 and $(v\xi_1,u\xi_2)^\top$ in regime 2. Use the four histories in each regime as prescribed reference sets with means zero and rank cap one, so $P_1=e_1,P_2=e_2$. Use the same fixed positive KAHM regularizer for both models. In each model the projected samples are $\pm u$, each repeated twice, with projected sample variance $4u^2/3$. Writing $x=P_j^\top z$, the scalar reconstructed coordinate is

$$a_A(x)=x,\quad a_R(x)=u\tanh(3x/(4u)),\quad a_K(x)=\gamma u\tanh(3x/(4u)),$$

where $\gamma=[2(1+e^{-3/2})+\lambda]/[2(1-e^{-3/2})+\lambda]$. The constant and projected-sign Gram eigenvectors have eigenvalues $2(1\pm e^{-3/2})$, giving these quotients. The denominator is positive and the absolute sum of normalized signed weights is at most $\gamma<2$. Both radial and angular errors strictly decrease with projection magnitude $t=|x|$ at fixed input norm. For the kernel reconstructions this follows from $A(t)=|a_m(x)|$, $0<A'(t)\le A(t)/t<2$ and the derivative of $\|z\|^2+A(t)^2-2tA(t)$; for affine reconstruction it follows from $\|z\|^2-t^2$. Hence the score of the correct regime is $T_+$ and the other score is $T_->T_+$, for all three reconstructions.

Define $\rho=[(1-T_-+\tau)/(1-T_++\tau)]^\omega\in(0,1)$ and $g=\phi_1$. The gate $g$ equals $1/(1+\rho)$ in regime 1 and $\rho/(1+\rho)$ in regime 2. Local $(g,1-g,gZ_1,(1-g)Z_2)$ and the baseline $(g,1-g,Z_1,Z_2)$ are injective on this support and both have affine dimension four, including the intercept. Consider $h=\alpha Z_1+\beta Z_2$ in regime 1 and $h=\beta Z_1+\alpha Z_2$ in regime 2. Optimal excess population MSEs are

$$ A_L=\frac{u^2v^2(\beta-\rho\alpha)^2}{u^2+\rho^2v^2},\qquad A_B=\frac{u^2v^2(\alpha-\beta)^2}{u^2+v^2}. $$

The two regime indicators and two slope functions are independent; the target has mean zero in each regime and the slopes are orthogonal by sign symmetry. One slope contributes $\frac12\inf_k[u^2(\alpha-k)^2+v^2(\beta-\rho k)^2]$; the second contributes the same. For the baseline replace the weighting ratio by one. Local improves for $(\alpha,\beta)=(1,0)$ and deteriorates for $(0,1)$. If $0<\beta<\alpha$, the unique positive contrast with $\rho=\beta/\alpha$ makes $A_L=0$. The example uses prescribed references and unrestricted affine coefficients, not learned-regime recovery or a matched-ridge guarantee.

### Exact inverse stability in the same example

The recovery matrix is diagonal with entries $g,1-g$, and $s(\phi)=\rho/(1+\rho)$. The two histories $(u,v)$ and $(u,-v)$ have identical gates and differ only in the weak weighted coordinate. The least Lipschitz constant $C_{\rm inv}$ of any exact inverse on the encoded support is at least $(1+\rho)/\rho$. Equality holds for sufficiently small $\rho$: within a regime this is the maximum amplification, while the gate-vector distances between regimes approach $\sqrt2$ and the original distances are bounded. For the aligned target,

$$\lim_{\rho\downarrow0} A_L C_{\rm inv}^2=v^2.$$

Prediction accessibility can improve even as exact input recovery becomes increasingly unstable. This is a clean, unstandardized, finite-support result, not a claim of robustness under arbitrary numerical corruption.

## 5. Relationship to established methods

Reconstruction by local kernel affine hull models follows Kumar et al. (2024, 2025). Local prediction from delay coordinates, local coordinate coding, local tangent coding, mixtures of experts and usable-information theory distinguish predictive accessibility from information retention (Farmer and Sidorowich, 1987; Jacobs et al., 1991; Yu et al., 2009; Yu and Zhang, 2010; Xu et al., 2020). The positive-kernel control is normalized vector-valued kernel smoothing, while normalized inverse-kernel constructions have an antecedent in rescaled RBF interpolation. Here the gates are determined by reconstruction rather than jointly trained on prediction targets. The contribution is the analysis of the reconstruction-affinity encodings, information-equivalent controls, exact task-dependent comparisons, and their recovery properties. Clean Lipschitz-constrained risk certification and autonomous Koopman closure are different scientific questions and are not required by this framework.

See [Experimental protocols](PROTOCOLS.md) for the implemented numerical conventions and [Reproduction](../REPRODUCIBILITY.md) for executable checks of the finite-law example and recovery inequalities. Numerical checks are not formal proof verification.

## References

- M. Kumar, B. A. Moser and L. Fischer (2024). On mitigating the utility-loss in differentially private learning: A new perspective by a geometrically inspired kernel approach. *Journal of Artificial Intelligence Research*, 79, 515–567. [DOI](https://doi.org/10.1613/jair.1.15071).
- M. Kumar et al. (2025). Geometrically inspired kernel machines for collaborative learning beyond gradient descent. *Journal of Artificial Intelligence Research*, 83, Article 16. [DOI](https://doi.org/10.1613/jair.1.16821).
- J. D. Farmer and J. J. Sidorowich (1987). Predicting chaotic time series. *Physical Review Letters*, 59, 845–848. [DOI](https://doi.org/10.1103/PhysRevLett.59.845).
- Y. Xu, S. Zhao, J. Song, R. Stewart and S. Ermon (2020). A theory of usable information under computational constraints. *International Conference on Learning Representations*. [Record](https://openreview.net/forum?id=r1eBeyHFDH).
- E. A. Nadaraya (1964). On estimating regression. *Theory of Probability and Its Applications*, 9, 141–142. [DOI](https://doi.org/10.1137/1109020).
- S. De Marchi, A. Idda and G. Santin (2017). A rescaled method for RBF approximation. *Approximation Theory XV*, 39–59. [DOI](https://doi.org/10.1007/978-3-319-59912-0_3).
- K. Yu, T. Zhang and Y. Gong (2009). Nonlinear learning using local coordinate coding. *Advances in Neural Information Processing Systems*, 22.
- K. Yu and T. Zhang (2010). Improved local coordinate coding using local tangents. *Proceedings of the International Conference on Machine Learning*.
- R. A. Jacobs, M. I. Jordan, S. J. Nowlan and G. E. Hinton (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79–87.
