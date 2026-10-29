# Feature construction and numerical methods

This guide defines the representation and decoder implemented in
`src/certified_forecasting/`. The [manuscript](paper/) gives the statistical
results and proofs; the [reproduction guide](docs/REPRODUCIBILITY.md) specifies
the study configurations and commands. Training-objective accuracy and
population prediction risk are different quantities, and the software records
a separate diagnostic for each.

## KAHM features and stable normalization

`NestedSoftFeatures(kind="kahm")` calls the parent repository's
`train_kahm_state_regime_abstraction` and `kahm_folding_distance_matrix`.
These functions fit local Kernel Affine Hull Machine (KAHM) reconstruction
models and evaluate their geometric discrepancy scores. They require the
extension to remain inside the complete parent repository.

Let $x$ be a causal observation history. In dictionary block $b$, let $C_b$ be
the number of reference groups and let $T_{bc}(x)\in[0,1]$ be the folding
score for group $c\in\{1,\ldots,C_b\}$. A smaller score indicates a closer
reconstruction match. For an exponent $\omega>0$ and offset $\tau>0$, the
soft association with group $c$ is

$$
\Phi_{bc}(x)=
\frac{(1-T_{bc}(x)+\tau)^\omega}
     {\sum_{r=1}^{C_b}(1-T_{br}(x)+\tau)^\omega}.
$$

The vector $\Phi_b(x)=(\Phi_{b1}(x),\ldots,\Phi_{bC_b}(x))^\top$ has
nonnegative entries summing to one. These entries are relative reconstruction
affinities; they are not calibrated class probabilities. The default exponent
is 16 and the default offset is $10^{-6}$.

For numerical evaluation, `stable_associations` clips the supplied scores to
$[0,1]$, computes
$\ell_{bc}=\omega\log(1-T_{bc}(x)+\tau)$, and returns
$\exp(\ell_{bc}-\operatorname{logsumexp}_r\ell_{br})$.
Here $\operatorname{logsumexp}_r\ell_{br}=\log\sum_r\exp(\ell_{br})$,
evaluated by subtracting the largest log weight before exponentiation.
This expression is algebraically equal to the normalized affinities above.
Replacing a small positive denominator by uniform weights would define a
different feature map even when the relative weights are numerically
resolvable. Fitting and calibration must use the same implemented map.

By default, histories are standardized using training-input means and standard
deviations. Optional preprocessing uses training extrema or leaves histories
unscaled, as specified in the [scaling overview](README.md#optional-kahm-input-scaling).
Each block has its own seeded clustering. The default geometry fit uses at most
2,500 training histories, at most 48 KAHM reference points per cluster, and at
most two principal directions. Decoder fitting can use all training histories.
The configuration records these computational limits; they are not assumptions
of the statistical certificates.

The `affine` comparator fits local principal-component affine subspaces to
k-means clusters. Its score combines a normalized reconstruction residual with
a local distance penalty. The `rbf` comparator forms normalized Gaussian
associations from distances to k-means centers. Both are separate geometric
feature maps with their own model definitions.

## Retained blocks and the decoder class

At stage $s$, the representation retains the first $s$ fitted feature blocks:

$$
\Gamma_s(x)=(\Phi_1(x)^\top,\ldots,\Phi_s(x)^\top)^\top,
\qquad p_s=\sum_{b=1}^s C_b.
$$

Thus $p_s$ is the total number of feature coordinates. Each block separately
sums to one; the stack belongs to a product of simplices. Appending a block
leaves all earlier coordinate functions unchanged, even when the blocks were
constructed from different cluster partitions.

A decoder with intercept $a\in\mathbb R^q$ and coefficient blocks
$W_b\in\mathbb R^{C_b\times q}$ predicts
$a+\sum_{b=1}^s W_b^\top\Phi_b(x)$ before clipping. The number $q$ is the
number of jointly predicted target coordinates, including all forecast
horizons. The constraint is $\sum_b\|W_b\|_{\rm F}\leq\kappa$, where
$\|\cdot\|_{\rm F}$ is the square root of the sum of squared matrix entries
and $\kappa\geq0$ is the coefficient budget.

An earlier decoder is feasible after a block is appended: assign the new block
zero coefficients and retain its intercept and other coefficients. Its
predictions and budget usage are unchanged. If a block is duplicated exactly,
its two coefficient matrices can instead be merged by addition. The triangle
inequality shows that merging cannot increase the sum of block norms.
Consequently, exact duplication leaves the feasible predictor class unchanged.

The feature-space metric and its induced distance on histories are

$$
\rho_s(u,v)=\max_{1\leq b\leq s}\|u_b-v_b\|_2,
\qquad d_s(x,x')=\rho_s(\Gamma_s(x),\Gamma_s(x')).
$$

Here $u=(u_1,\ldots,u_s)$ and $v=(v_1,\ldots,v_s)$ are stacked feature
vectors, with $u_b,v_b\in\mathbb R^{C_b}$, and $\|\cdot\|_2$ is the
Euclidean norm. The distance between two unclipped predictions is at most
$\kappa d_s(x,x')$ by the
triangle inequality and the block-norm constraint. Clipping to a known output
box cannot increase this distance.

Separately refitting an entire higher-resolution map or changing the history
length does not preserve the same coordinate functions and therefore does not
establish this nesting property. Nesting also concerns the feasible class,
not the risks of separately fitted estimators. A squared Frobenius ridge
penalty can decrease when coefficients are spread over duplicates, so the
penalized fit need not be duplication invariant.

## Convex decoder fitting

Let $N$ be the number of training histories, let
$F\in\mathbb R^{N\times p_s}$ contain the feature vectors as rows, and let
$\mathsf Y\in\mathbb R^{N\times q}$ contain the joint targets as rows.
Write $W\in\mathbb R^{p_s\times q}$ for the stack of the row blocks $W_b$,
$a\in\mathbb R^q$ for the intercept, and $\mathbf1_N$ for the length-$N$
vector of ones. These are the training conventions used in the manuscript.
`BlockBoundedRidge` solves

$$
\min_{W,a}\;
\frac1N\|\mathsf Y-FW-\mathbf1_Na^\top\|_{\rm F}^2
+\lambda\|W\|_{\rm F}^2
\quad\text{subject to}\quad
\sum_{b=1}^s\|W_b\|_{\rm F}\leq\kappa.
$$

The ridge coefficient $\lambda\geq0$ is named `alpha` in the implementation.
The loss sums over target coordinates and horizons and averages over training
rows; it is not additionally divided by $q$. The intercept is unconstrained
and unpenalized. Training rows may be dependent because minimizing this finite
objective does not require a statistical independence assumption.

Centering the feature and target matrices eliminates the intercept. Let
$F_c$ and $\mathsf Y_c$ be the centered matrices. The reduced objective is
$J(W)=N^{-1}\|\mathsf Y_c-F_cW\|_{\rm F}^2+\lambda\|W\|_{\rm F}^2$,
and its gradient is
$G(W)=2F_c^\top(F_cW-\mathsf Y_c)/N+2\lambda W$.
The solver uses accelerated projected gradient steps with monotone restarts,
based on the fast iterative shrinkage-thresholding algorithm (FISTA).

Projection onto the feasible set first computes each coefficient block's
Frobenius norm. If their sum exceeds $\kappa$, it finds a nonnegative threshold
$\theta$ for which
$\sum_b(\|W_b\|_{\rm F}-\theta)_+=\kappa$, where
$(v)_+=\max\{v,0\}$. A nonzero block is then multiplied by
$(1-\theta/\|W_b\|_{\rm F})_+$; a zero block remains zero.
Feasible inputs require no change, and a zero budget gives the zero coefficient
matrix.

For a feasible iterate, the recorded Frank--Wolfe gap is

$$
\mathcal D(W)=\langle G(W),W\rangle_{\rm F}
+\kappa\max_b\|G_b(W)\|_{\rm F}.
$$

Here $G_b(W)$ is the gradient block associated with $W_b$, and
$\langle A,B\rangle_{\rm F}=\sum_{i,j}A_{ij}B_{ij}$ is the Frobenius inner
product. Convexity gives $0\leq J(W)-J^*\leq\mathcal D(W)$, where $J^*$ is
the minimum feasible objective value. Iteration count, objective value, gap,
and convergence status are recorded. These are floating-point diagnostics,
not interval-arithmetic guarantees.

The implementation stores a sensitivity bound for each scalar target in
`lipschitz_[j]`: it is the sum of the Euclidean norms of the corresponding
coefficient column across blocks. Here `j` is the zero-based column position
in the ordered target vector, rather than the numerical forecast horizon. The certificate for a returned predictor
requires its actual sensitivity to satisfy the budget used for calibration;
it does not require the training optimizer to reach its exact minimum.

The final forecast projects the affine prediction onto the task's known
output interval or box. Euclidean projection onto a box is coordinatewise
clipping. It is nonexpansive and cannot increase squared error when the target
lies in that box. The training objective uses the unclipped affine predictor;
all reported forecast losses use the clipped prediction.

## Alternative decoders

`BoundedRidge` replaces the sum-of-block-norms constraint by
$\|W\|_{\rm F}\leq\kappa$ while retaining the same loss, intercept, and
ridge term. After centering, an eigendecomposition of the feature covariance
produces the unconstrained ridge solution. If it exceeds the norm budget, a
nonnegative additional Lagrange multiplier is found by bisection. The budget
is joint across all target coordinates and horizons; `lipschitz_[j]` is the
Euclidean norm of scalar target column `j`.

The overall Frobenius constraint does not have the same duplication invariance
as the block-sum constraint. Spreading one coefficient block equally across
identical copies can reduce the overall Frobenius norm and enlarge the
feasible predictor class.

`KoopmanRidge` fits an affine feature transition and a separate decoder of the
current observation. It composes the transition for each requested horizon
and clips only the final observed-output prediction. The software exposes the
effective horizon coefficient matrices and their Euclidean sensitivity
bounds. The fitted transition does not establish exact Koopman closure.
Direct quadratic regression and polynomial-feature rollout are separate
comparison methods.

## Synthetic episodes and causal histories

The four main generators have scalar observations in the known interval
$[-1,1]$. The `harmonic` and `quasiperiodic` generators have the tighter absolute
bound 0.88; using $[-1,1]$ is still valid. Innovations and observation noise
are uniformly distributed on the configured bounded intervals. No observed
sample maximum serves as a population bound. Initial conditions, parameters,
phases, and innovations are sampled independently across episodes. Windows
within an episode can overlap and be dependent.

Histories are ordered from oldest to newest. All horizons for one candidate
use a common valid origin set. To compare history lengths under the same risk
definition, use a common `first_origin` at least as large as the maximum
history length minus one. The calibration unit is an independent episode,
not an overlapping window. Stationarity is unnecessary when new episodes
follow the same specified finite-episode law and origin protocol.

## Numerical and implementation checks

The tests cover KAHM fitting and evaluation, simplex normalization, retention
of earlier coordinates, numerical underflow in affinity normalization,
decoder feasibility and optimality diagnostics, output clipping, causal
history indexing, bounded episode generation, and persistence forecasts.
Additional tests check projection onto the block-norm constraint, agreement
with the one-block constrained ridge solution, the optimization gap,
duplicate-block merging, the block-maximum sensitivity bound, and equivalence
between composed Koopman forecasts and explicit affine rollout.

The suite also checks preprocessing, model selection, score-archive integrity,
and calibration formulas. The [verification archive](docs/ARCHIVE_INDEX.md)
identifies recorded executions and the versions to which their results apply.
Simulation-based coverage assessment and reconstruction of experimental
results provide separate checks; neither a unit-test pass nor numerical
reconstruction establishes the statistical sampling assumptions.
