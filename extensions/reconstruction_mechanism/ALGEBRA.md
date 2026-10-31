# Reconstruction-to-prediction algebra and bounded correction

## Decision and scope

**Algebra decision: GO to one matched forecasting study.** The normalized
KAHM reconstruction departure is not universally redundant with the common
gated affine coordinates or with the matched normalized RBF departure. An
explicit three-reference construction satisfies the inherited kernel and
regularizer conventions and proves this statement. The decision establishes
an identifiable candidate mechanism, **not forecasting value**. Symmetric
banks can instead yield exactly the same raw augmented feature span. Neither
case implies that a fitted decoder will predict future observations well.

The executable companion, `algebra.py`, uses fabricated reference banks,
query coordinates, and algebraic target vectors only. It accesses no reserved
forecasting outcomes and fits no forecasting models. Its completed results
are in `algebra_results.json`. The fixed reference factor is computed by the
actual `SharedBank._fit_kahm_factor` implementation in
`extensions/direct_reconstruction/model.py`; the kernel is the Gaussian
kernel in `extensions/predictive_geometry/study/vendor/upstream/kxx_matrix.py`.
The mathematical statements below are derived here. The script provides
independent arithmetic checks and numerical witnesses, rather than replacing
the proofs with empirical ranks.

## 1. The actual maps and the proposed features

For one fixed local bank, let the rows of
\(Q\in\mathbb R^{M\times p}\) be reference histories, let \(P\) be its
fixed principal-coordinate basis, let \(W\) be its fixed projected covariance
precision, and let \(r\) be the projected dimension. The inherited kernel is

\[
k_i(x)=\exp\!\left[-\frac{1}{2r}
 (P^\top(x-q_i))^\top W(P^\top(x-q_i))\right].
\]

Write \(k(x)\) as a row vector, let \(K\) be the corresponding reference
Gram matrix, and set \(C=(K+\lambda I)^{-1}\), with the positive regularizer
returned by the inherited reference-energy iteration. The two reconstructions
are

\[
A_K(x)=\frac{k(x)CQ}{k(x)C\mathbf1},\qquad
A_R(x)=\frac{k(x)Q}{k(x)\mathbf1}.
\tag{1}
\]

The first uses signed normalized KAHM weights. The second is the matched
normalized RBF reconstruction with the **same** references, projection,
covariance, and Gaussian bandwidth. A common positive rescaling of a row
\(k(x)\), including subtraction of the maximum log kernel before
exponentiation, cancels from both ratios.

Let
\(A_A(x)=\mu+(x-\mu)PP^\top\) be the local affine reconstruction, using
row-vector notation. With a common fixed gate \(g_j(x)\) for bank \(j\),
the candidate added features are

\[
d_{K,j}(x)=g_j(x)\{A_{K,j}(x)-A_{A,j}(x)\},\qquad
d_{R,j}(x)=g_j(x)\{A_{R,j}(x)-A_{A,j}(x)\}.
\tag{2}
\]

Both are appended to the identical common gate and gated-principal-coordinate
features. The affine departure is zero. This specifies where reconstruction
can enter prediction: through additional functions of the observed history
available to a restricted residual decoder. It does not require the
reconstruction itself to be a prediction of the future.

## 2. Identities that do not establish a new forecasting mechanism

### Proposition 1: normalized weights, residuals, and direct target transfer

Whenever \(s(x)=k(x)C\mathbf1\ne0\), define
\(\alpha(x)=k(x)C/s(x)\). Then

\[
\alpha(x)\mathbf1=1,\qquad A_K(x)=\alpha(x)Q,
\qquad A(x)+\{x-A(x)\}=x.
\tag{3}
\]

The last identity holds for every reconstruction map. Adding a reconstruction
and its residual to unrestricted raw features can create linear redundancy;
it is not evidence that reconstruction adds predictive information.

For any fabricated or observed reference target vector \(y\),

\[
\alpha(x)y=
\frac{k(x)(K+\lambda I)^{-1}y}
     {k(x)(K+\lambda I)^{-1}\mathbf1}.
\tag{4}
\]

Equation (4) is exactly a ratio of two kernel-ridge predictions, one fitted
to \(y\) and one fitted to the constant vector. Here kernel ridge means the
dual formula shown, with no additional unpenalized intercept. Calling this
same calculation direct future-weight transfer does not create an independent
predictor. It is an identity check, not a comparator that KAHM can meaningfully
outperform under identical arithmetic and settings.

Similarly, a linear readout \(A_K(x)b\) is normalized kernel ridge on the
restricted pseudo-target vector \(Qb\). The feature construction is therefore
a particular constrained basis and normalization, rather than a new general
regression principle.

For an affine scalar target \(f(q)=q b+c\),

\[
\sum_i\alpha_i(x)f(q_i)-f(x)
   =\{A_K(x)-x\}b.
\tag{5}
\]

This follows from \(\sum_i\alpha_i=1\) and is an affine-weight identity.
It does not apply to an arbitrary nonlinear future-generating function, and
does not imply that minimizing history reconstruction error minimizes
forecasting error.

## 3. Exact conditions for raw feature redundancy

### Proposition 2: a sufficient KAHM/RBF reparameterization condition

Suppose the reference Gram matrix is row regular,
\(K\mathbf1=\kappa\mathbf1\), and suppose its inverse-regularized action
preserves the augmented reference-coordinate span:

\[
C\mathbf1=a\mathbf1,
\quad a=(\kappa+\lambda)^{-1},
\qquad CQ=QB+\mathbf1 c^\top.
\tag{6}
\]

Then, at every query,

\[
A_K(x)=\{A_R(x)B+c^\top\}/a.
\tag{7}
\]

**Proof.** Substitute (6) into the numerator and denominator of (1).
The common factor \(k(x)\mathbf1\) cancels. If the augmented span is
invariant under the invertible operator \(C\), its restriction to that
finite-dimensional span is invertible. Thus, after removal of any redundant
coordinate columns, (7) is an invertible affine change of the active
coordinate functions. The augmented raw function spans are equal. \(\square\)

For the departure feature,

\[
g(A_K-A_A)
=g(A_R-A_A)B/a
 +gA_A(B/a-I)+g c^\top/a.
\tag{8}
\]

The final two terms lie in the common gate and gated-PC span, because
\(gA_A=g\mu+g(x-\mu)PP^\top\). Consequently the two departures can add
the same raw augmented function span even though their coordinates differ.
Condition (6) is sufficient; its failure alone is not a proof of
nonredundancy. That requires a separate argument, given in Proposition 4.

### Proposition 3: exact symmetric degeneracy with two references

Take one one-dimensional bank, \(q=(-1,1)^\top\), \(P=1\), \(g=1\),
and \(A_A(x)=x\). Write

\[
K=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix},\qquad
c=\frac{1+\rho+\lambda}{1-\rho+\lambda}.
\]

The constant and contrast vectors are eigenvectors of \(K\), so

\[
A_K(x)=cA_R(x),\qquad
d_K(x)=c\,d_R(x)+(c-1)x.
\tag{9}
\]

Thus
\(\operatorname{span}(1,x,d_K)=\operatorname{span}(1,x,d_R)\) exactly.
The executable source-compatible fixture uses sample-variance precision
\(1/2\), hence \(\rho=e^{-1}\), and inherited
\(\lambda=2.6519665429866075\). It gives
\(c=1.224037566462905\). The maximum errors in (9) are below
\(6\times10^{-16}\) on the declared grid.

**Regularization and normalization qualification.** Equal raw function spans
do not guarantee identical fitted predictions. An isotropic coefficient
penalty is generally not invariant under a nonorthogonal change of feature
coordinates. Appending duplicated linear features can change the induced
penalty without changing the function span. Moreover, applying separate
nonlinear bounded transforms to the common block and departure block need
not preserve (8) or (9). Indeed, the transformed two-reference fixture has
a nonzero numerical residual to the transformed RBF/common span despite
the exact raw identity. Forecast comparisons must therefore concern the
**complete declared reconstruction, normalization, and decoder procedure**;
a difference cannot automatically be attributed to greater raw basis capacity.

## 4. An analytic nonredundancy result for irregular references

### Proposition 4: three unequally spaced scalar references

Let \(q_1<q_2<q_3\) be scalar references with
\(q_1+q_3\ne2q_2\), let the Gaussian precision be \(t>0\), let
\(C=(K+\lambda I)^{-1}\), and suppose \(v=C\mathbf1\) is componentwise
positive. Use one bank with gate one and affine reconstruction \(x\).
Then

\[
A_K\notin\operatorname{span}(1,x,A_R),\qquad
d_K\notin\operatorname{span}(1,x,d_R).
\tag{10}
\]

**Proof.** Set \(u=Cq\). The six unordered pair sums \(q_i+q_j\),
\(i\le j\), are distinct under the stated spacing condition. Products of
Gaussian translates satisfy

\[
k_i(x)k_j(x)=
e^{-t x^2}\,
e^{-t(q_i^2+q_j^2)/2}\,
e^{t(q_i+q_j)x}.
\tag{11}
\]

After removing the common factor, the six distinct real exponentials are
linearly independent. For example, differentiating a putative zero linear
combination at zero gives a nonsingular Vandermonde system.

Suppose first that \(A_K=cA_R+d\). Cross multiplication gives

\[
(k(u-dv))(k\mathbf1)=c(kq)(kv).
\tag{12}
\]

Let \(w=u-dv\). Matching the independent diagonal and off-diagonal
terms from (11) yields

\[
w_i=cq_i v_i,\qquad
w_i+w_j=c(q_i v_j+q_j v_i).
\tag{13}
\]

If \(c=0\), then \(u=dv\); invertibility of \(C\) gives
\(q=d\mathbf1\), contradicting distinct references. If \(c\ne0\),
substitution in the second identity gives
\((q_i-q_j)(v_i-v_j)=0\), hence all \(v_i\) are equal. A constant
\(C\mathbf1\) would require constant row sums of \(K\). Yet the middle
reference has larger row sum than either endpoint: its two distances are
strictly shorter than the corresponding endpoint's far distance, and the
Gaussian is strictly decreasing with distance. This is a contradiction.
Thus \(A_K\) is not an affine function of \(A_R\).

Both reconstructions are bounded on the entire real line. For \(A_R\)
this follows from positive normalized kernel weights. For \(A_K\), write

\[
A_K(x)=\sum_i\frac{k_i(x)v_i}{\sum_j k_j(x)v_j}\,
                   \frac{u_i}{v_i}.
\tag{14}
\]

The coefficients in (14) are positive and sum to one, so this is a bounded
combination of three finite constants. A putative identity
\(A_K=a+bx+cA_R\) must therefore have \(b=0\), reducing to the
already excluded case. Subtracting \(x\) proves the departure statement.
\(\square\)

This result holds for a nondegenerate, open class of three-reference banks
satisfying the positivity condition. It is not an almost-everywhere claim
for every higher-dimensional or multiple-gate bank produced by training.
In particular, adding other common nonlinear gate functions can alter the
span comparison. It is enough to reject a universal claim that the proposed
KAHM departure is always an affine or RBF reparameterization.

### Source-compatible irregular fixture and a positivity certificate

Take \(q=(-1,0,2)^\top\). Its unbiased sample variance is \(7/3\),
so the inherited projected precision is \(t=3/7\). Its six pair sums are
\(-2,-1,0,1,2,4\). The exact Gaussian Gram matrix has unequal row sums,
and the actual source iteration gives \(\lambda=4.498274743339644\).

A useful sufficient positivity condition is available without checking
individual inverse entries. Write \(K=I+E\),
\(r_E=\|E\|_\infty\), and \(a=\lambda+1\). If \(a>2r_E\),
the Neumann expansion gives

\[
\left\|C\mathbf1-a^{-1}\mathbf1\right\|_\infty
\le\frac{r_E}{a(a-r_E)}<a^{-1}.
\tag{15}
\]

Thus \(C\mathbf1>0\). For the irregular fixture,
\(r_E=1.2314905926823394\) and \(a=5.498274743339644\), so (15)
applies with a substantial margin. This is also guaranteed directly by the
source regularizer convention: \(\nu=\|q\|^2/3=5/3\), initialization
uses \(2.5\nu\), and subsequent iterations use \(2\nu\) plus a
nonnegative reconstruction residual mean square. Hence every used regularizer
is at least \(10/3\), while \(r_E<2\).

The separate equally spaced fixture \(q=(0,1,2)\) illustrates that equal
spacing with three references is not itself the two-reference degeneracy.
Here \(C\mathbf1\) is nonconstant and the contrast
\((-1,0,1)^\top\) is a Gram eigenvector. With
\(\rho=e^{-1/2}\),

\[
\frac{A_K(x)-1}{A_R(x)-1}
=\frac{1}{\lambda+1-\rho^4}
  \frac{k(x)\mathbf1}{k(x)C\mathbf1},\quad x\ne1.
\tag{16}
\]

The right-hand ratio is nonconstant by independence of the three Gaussian
translates and the nonconstant vector \(C\mathbf1\). Reflection about
one would force any affine relation between the maps to preserve their
center, contradicting (16). Positivity from (15) again makes both maps
bounded and excludes an additional linear \(x\) term.

### Completed numerical witnesses

The query grid contains 81 equally spaced points on \([-3,4]\).
Residuals below are maximum absolute residuals after least-squares projection
onto the stated columns. They are numerical observations on this fabricated
grid, not independent proofs of global function-space equality or inequality.

| Fabricated reference rows | Inherited regularizer | Raw KAHM departure residual to \(1,x,d_R\) | Rank of \([1,x,d_R,d_K]\) | KAHM departure second difference at \(0,0.5,1\) |
|---|---:|---:|---:|---:|
| \((-1,1)\) | 2.651966543 | \(8.88\times10^{-16}\) | 3 | -0.0339305264 |
| \((0,1,2)\) | 4.276973776 | 0.0078485492 | 4 | 0.0500545779 |
| \((-1,0,2)\) | 4.498274743 | 0.0535567100 | 4 | 0.0366395272 |

The irregular fixture also has raw residual 0.374514697 outside
\(\operatorname{span}(1,x)\). Its minimum relative signed denominator on
the grid is 0.949853739. Across all fixtures, direct normalized-KRR and affine
target identities agree to below \(9\times10^{-16}\). The nonzero finite
second differences provide additional curvature witnesses; finite differences
and ranks do not measure predictive usefulness.

For a diagnostic of the normalization caveat only, each fabricated grid also
supplies its own normalization constants. After separately bounding the common
and departure blocks, residuals to the transformed common/RBF span are
0.0218322995, 0.0240379327, and 0.0308656511 respectively. The first is
nonzero despite the exact raw identity (9). This diagnostic is not a
forecasting train/test protocol.

## 5. Bounded features and a deterministic correction bound

### Proposition 5: blockwise radial normalization

For finite \(v\in\mathbb R^d\) and \(B>0\), define

\[
T_B(v)=\frac{v}{\sqrt{1+\|v\|^2/B^2}}.
\tag{17}
\]

Then \(\|T_B(v)\|<B\). With
\(a_v=(1+\|v\|^2/B^2)^{-1/2}\), its Jacobian is

\[
D T_B(v)=a_v I-a_v^3 vv^\top/B^2.
\tag{18}
\]

Its radial eigenvalue is \(a_v^3\), its tangential eigenvalues are
\(a_v\), and its operator norm is at most one. It is nonexpansive as a
map of its finite input vector. Preceding it with A-only coordinate
standardization does not give the original history-to-feature map a global
Lipschitz constant: inverse scales and derivatives of the preceding
reconstruction and gate maps still matter.

The declared procedure fixes A-only centers and standard deviations, with
standard-deviation floor \(10^{-6}\), then applies (17) separately to
the common block and the departure block, using \(B_b=\sqrt{d_b}\).
It performs **no second coordinate standardization inside ridge**. If all
blocks are transformed this way, their stacked feature vector \(\psi(x)\)
satisfies

\[
\|\psi(x)\|\le R,\qquad
R=\sqrt{\sum_b B_b^2}=\sqrt{\sum_b d_b}.
\tag{19}
\]

The non-strict inequality accommodates limiting values and floating-point
rounding. A small logged numerical tolerance is appropriate for verification.

### Proposition 6: centered ridge coefficient and prediction bounds

For one horizon, fit a residual head on \(n\) A rows by minimizing

\[
\frac1n\sum_{i=1}^n(e_i-a-\psi_i^\top w)^2
  +\alpha\|w\|^2,\qquad \alpha>0,
\tag{20}
\]

with an unpenalized intercept. Write
\(\bar\psi=n^{-1}\sum_i\psi_i\),
\(\bar e=n^{-1}\sum_i e_i\),
\(X_c=(\psi_i-\bar\psi)^\top_{i=1}^n\), and
\(e_c=e-\bar e\mathbf1\). The equivalent penalty in the sum-of-squares
normal equations is **\(n\alpha\)**. Thus

\[
w=(X_c^\top X_c+n\alpha I)^{-1}X_c^\top e_c,
\quad q(x)=\bar e+(\psi(x)-\bar\psi)^\top w.
\tag{21}
\]

In the singular-value decomposition, each nonzero singular value contributes
the factor \(s/(s^2+n\alpha)\), bounded above by
\(1/(2\sqrt{n\alpha})\). Consequently

\[
\|w\|\le L_e
:=\frac{\|e_c\|}{2\sqrt{n\alpha}}
=\frac{\operatorname{RMS}(e_c)}{2\sqrt\alpha},
\tag{22}
\]

and

\[
|q(x)-\bar e|\le\|\psi(x)-\bar\psi\|L_e,
\qquad
|q(x)|\le|\bar e|+(R+\|\bar\psi\|)L_e
\le|\bar e|+2RL_e.
\tag{23}
\]

For a separately selected correction multiplier \(\gamma\), the bound
on \(|\gamma q(x)|\) is multiplied by \(|\gamma|\). Horizon-specific
penalties, residuals, and multipliers use their respective horizon-specific
constants. These bounds are generic properties of the stated bounded
features and ridge objective; they do not favor KAHM over the matched RBF
or common-feature head.

The bound controls the **residual correction**. It does not bound an
unbounded raw baseline, establish forecasting accuracy, or ensure a single
constant across refits with changing targets or penalties. In particular,
removing the departure cap in the declared secondary ablation removes
the uniform feature bound (19), although the pointwise inequality in (23)
still holds for every finite feature vector.

The fabricated diagnostic uses 31 fitting rows, blocks of widths 3 and 4,
\(\alpha=0.01\), and 115 finite query rows including components of
magnitude \(10^{300}\). It returns:

| Quantity | Value |
|---|---:|
| Maximum query feature norm | 2.6457513111 |
| Feature norm bound \(R=\sqrt7\) | 2.6457513111 |
| Actual coefficient norm | 0.8212715682 |
| Coefficient norm bound (22) | 5.8240371451 |
| Maximum absolute query correction | 1.6423729792 |
| Uniform correction bound (23), using the fitted feature mean | 15.7077403466 |

All features and corrections are finite, and the pointwise and uniform
inequalities pass. The transform is evaluated using scaled vectors for
large magnitudes rather than squaring \(10^{300}\). This proves nothing
about forecasting performance and is not evidence that every upstream
history-to-feature computation can represent an arbitrarily large finite
history without numerical fallback.

## 6. Signed-denominator safety and interpretation limits

Let \(w_0(x)=k(x)C\). The inherited relative guard accepts a reconstruction
only if all relevant calculations are finite and

\[
\frac{|w_0(x)\mathbf1|}{\|w_0(x)\|_1}>\tau.
\tag{24}
\]

Then the normalized weights satisfy
\(\|\alpha(x)\|_1<1/\tau\), implying
\(\|A_K(x)\|<(\max_i\|q_i\|)/\tau\). At the declared
\(\tau=10^{-10}\), this is a weak safety bound; it does not itself make
the decoder correction practically small. Normalized RBF reconstruction is
a convex combination, whereas the signed KAHM reconstruction need not lie
inside the reference convex hull.

The finite-computation and signed-denominator checks precede the bounded
transform. If the declared deployment procedure substitutes zero departure
when a reconstruction cannot be formed safely, that substitution is part of
the estimator and must be counted and reported with matched handling across
comparators. An unsafe computation is not permission to hide an observation
or silently select a better method. Both the zero vector before normalization
and any explicitly specified zero transformed departure remain within the
bounded feature construction, but these two fallback conventions must not
be confused when reporting the implemented estimator.

These guards are prospective safety measures. They are **not** a diagnosis
of the earlier adaptive-study failures: the previously examined KAHM burst
had a safe denominator, and affine extrapolation also produced large errors.
The generic bounded correction specifically addresses decoder magnitude and
should not be presented as a retrospective proof of the old failure cause.

## 7. Consequences for the completed forecasting study

The algebra permits the reserved evaluation only after freezing the full
construction and comparisons. The same fixed reference banks, covariance
metrics, Gaussian kernel settings, gates, common features, A-only
standardization, residual targets, decoder convention, and selection budgets
must be used for the matched KAHM and RBF arms. The algebra introduces no
forecasting-value threshold and must not be used to change one after seeing
reserved outcomes.

Useful evidence would be a reproducible advantage of the complete KAHM
construction in the predefined setting under those comparisons. Equal
performance, failure, or isolated improvement over a weaker control would
not establish KAHM-specific forecasting value. The algebra warrants one
bounded empirical test; it neither requires a positive result nor licenses
further performance-method extensions after a negative result. Broader
claims that KAHM is an effective general time-series forecasting solution
remain unsupported by these algebraic results alone.

## Reproduce

From the repository root, run:

```bash
python3 extensions/reconstruction_mechanism/algebra.py
```

The script overwrites only its declared result JSON by default. It checks
the exact identities, symmetric degeneracy, irregular-bank witness conditions,
finite numerical behavior, and bounded-ridge inequalities, then emits the
algebra decision. Source SHA-256 values and all unrounded diagnostic values
are recorded in `algebra_results.json`.
