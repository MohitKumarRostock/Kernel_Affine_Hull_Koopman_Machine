# Forecasting implementation notes

## Relationship to the original implementation

The primary `NestedSoftFeatures(kind="kahm")` feature bank calls the parent
repository's `train_kahm_state_regime_abstraction` and
`kahm_folding_distance_matrix`. It does not replace the KAHM reconstruction
mechanism with nearest-centroid or PCA features. Inputs are standardized using
development-input means and standard deviations. Each retained block is trained
with its own seeded state-space clustering. At most 2,500 development histories
are used to construct the geometry by default; decoder training can use all
development histories. Each cluster uses at most 48 KAHM training prototypes by
default, with at most two retained principal directions. These limits are
configurable, deterministic computational choices, not statistical assumptions.

Associations are evaluated by subtracting `logsumexp` from
`omega * log(1 - clip(folding_score, 0, 1) + tau)`. This implements the paper's
normalization in real arithmetic. It deliberately removes the legacy rule that
replaces a small positive denominator with a uniform vector. Models must be
fitted and certified using this implemented map; old fitted coefficients and
old certificates cannot be transferred to it unchanged.

The `affine` comparator fits local PCA affine subspaces to k-means clusters and
combines a normalized reconstruction residual with a local distance penalty.
The `rbf` comparator uses normalized Gaussian distances to k-means centers.
These comparators are distinct geometric feature maps and are not called KAHM.

## What is nested

For fitted simplex blocks Phi_1,...,Phi_J, the stage-j feature vector is their
concatenation `(Phi_1,...,Phi_j)`. Every old coordinate remains unchanged; the
concatenation is a product of simplices, not a single simplex. An old decoder
embeds into the next stage by appending zero coefficient rows. Its predictions,
intercept, and sum of block Frobenius norms are unchanged. Thus fixed-budget
decoder classes are nested. The primary `BlockBoundedRidge` constrains the sum
of block norms. Duplicating an existing block does not enlarge this predictor
class: coefficients on identical blocks can be merged without increasing their
sum of norms. An ordinary overall Frobenius constraint lacks this protection,
because spreading coefficients over repeated blocks lowers that norm.

Separately refitting a higher-resolution map or using a different history length
does not have this nesting guarantee. Appending blocks does not promise better
out-of-sample prediction. A squared-Frobenius ridge penalty can still favor
spreading coefficients over duplicates, even under the block constraint;
invariance is asserted for the feasible predictor class, not the fitted
penalized estimator.

## Decoder and numerical optimization

The primary `BlockBoundedRidge` solves

    min_{W,b: sum_s ||W_s||_F <= kappa}
        (1/n) ||Y - X W - 1 b^T||_F^2 + alpha ||W||_F^2.

After eliminating the unpenalized intercept by centering, monotone-restarted
FISTA minimizes this convex quadratic. Its projection first projects the
vector of block norms onto an l1 ball and then rescales each coefficient block.
The returned Frank-Wolfe gap upper-bounds regularized objective suboptimality.
Iteration count, objective, gap, and convergence status are retained. The
matching input metric is the maximum Euclidean distance over feature blocks.
The per-horizon Lipschitz bound is the sum of the relevant block column norms.

The `BoundedRidge` ablation solves

    min_{W,b: ||W||_F <= kappa}
        (1/n) ||Y - X W - 1 b^T||_F^2 + alpha ||W||_F^2.

The intercept is eliminated by centering the development feature and target
matrices. Eigendecomposition of the feature covariance produces ridge
solutions. If the unconstrained ridge solution exceeds the norm budget, a
scalar additional Lagrange multiplier is found by bisection. The norm budget
is joint across all forecast horizons; each individual column has Euclidean
norm at most that budget. `lipschitz_[h]` stores the actual coefficient norm of
the fitted horizon-h decoder. The final forecast clips to task-defined target
bounds. Euclidean projection onto an interval or box is nonexpansive and cannot
increase squared error for an in-box target.

The primary direct multi-horizon forecast uses the observed history only and
predicts each raw future target. It is not an iterated learned Koopman matrix.
`KoopmanRidge` supplies an actual iterated baseline: it separately fits an affine
feature transition and a current-observation decoder, composes the transition
for each requested horizon, and clips only the final raw output. Effective
horizon coefficient matrices and their Euclidean Lipschitz bounds are exposed.
No exact closure claim follows from this empirical transition fit. The direct
polynomial comparator and iterated polynomial comparator must be named
separately in experimental reporting.

## Synthetic episode laws

The four generators have scalar outputs in the known interval [-1,1].
`harmonic` and `quasiperiodic` actually have the tighter absolute bound 0.88;
using [-1,1] remains valid. Innovation and observation noise are uniform and
bounded. No empirical target maximum is used as a population bound. Random
initial conditions, parameters, phases, and innovations are independent across
episodes. Within-episode windows may overlap and are dependent.

Histories are ordered oldest to newest. All target horizons for a candidate
share the same valid origin set. To compare history lengths on an identical
risk target, supply a common `first_origin` at least as large as the maximum
history length minus one. The certification unit is an independent episode,
not an individual overlapping window. Stationarity is not required when new
episodes use exactly the same finite-episode sampling law and origin protocol.

## Validation performed

The unit-test suite checks: actual original-KAHM fitting and evaluation;
simplex normalization and retained coordinates; a numerical underflow case
that formerly gave a spurious uniform vector; decoder norm feasibility and
first-order optimality; nonexpansive output clipping; zero-padding of a decoder;
causal history indexing and common origin support; bounded seeded episode
generation; and persistence using only the latest observation. Additional tests
verify a known block-ball projection, agreement of the one-block solver with
the exact constrained ridge solution, the reported optimization gap, duplicate
block merging, the block-maximum Lipschitz bound, and agreement of composed
Koopman forecasts with explicit affine rollout. All thirteen model/protocol
tests passed on the available Linux/Python environment. Experiment-level
validation, coverage evaluation, and result generation are handled separately.

