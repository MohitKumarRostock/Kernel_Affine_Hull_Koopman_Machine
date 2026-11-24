# Kernel affine hull reconstruction and linear forecasting

## 1. Problem and notation

Let $(x_i,y_i)_{i=1}^N$ be observed transitions in $\mathbb R^D$. The objective is to construct a representation $\phi:\mathbb R^D\to\Delta_C$, where

$$
\Delta_C=\{u\in\mathbb R^C:u_j\geq0,\ \mathbf1^\top u=1\},
$$

and to approximate $\phi(y_i)$ from $\phi(x_i)$. The forecast concerns representation coordinates, not necessarily the original physical state. No inverse state decoder is required or supplied by the core forecasting interface.

## 2. Local reconstruction

K-means clustering of the training current states defines groups of reference samples. Each group has one or more local reconstruction models; the effective number of groups may differ from the requested number under the configured treatment of singleton groups.

For a reference matrix $Y=[y_1,\ldots,y_n]$, the local model uses leading eigenvectors of the sample covariance to construct an encoding $P^\top x$. The retained dimension $d$ is bounded by the requested dimension, sample rank and $n-1$, and is reduced when encoded coordinate ranges are insufficient. The encoded reference covariance $\Sigma$ must be invertible. For two encoded vectors $u,v$ the Gaussian kernel is

$$
k(u,v)=\exp\left[-\frac{(u-v)^\top\Sigma^{-1}(u-v)}{2d}\right].
$$

With $K$ the reference Gram matrix, the regularized kernel regression matrix is of the form $(K+\lambda I)^{-1}$. The implementation determines $\lambda$ from the reference data through the residual-based iteration in [kernel_regularized_least_squares.py](../kernel_regularized_least_squares.py). For a query $x$, form the row vector $w(x)=k_x^\top(K+\lambda I)^{-1}$, where $k_x$ contains its kernel evaluations against the reference samples. Provided the sum of the weights is nonzero, the affine reconstruction is

$$
\widehat x=Y\frac{w(x)^\top}{w(x)\mathbf1}.
$$

The weights sum to one but need not be nonnegative. When the weight sum vanishes, the implementation leaves the unnormalized weights rather than dividing by zero. Nondegenerate reference samples and invertible retained covariance are prerequisites; singular or otherwise inadmissible inputs must not be interpreted as valid fitted models.

For a single reconstruction define

$$
d_E=1-\exp(-\|x-\widehat x\|_2),\qquad
d_A=\frac{1}{\pi}\arccos\left(\frac{x^\top\widehat x}{\|x\|_2\|\widehat x\|_2}\right).
$$

The angular calculation uses cosine zero when the product of norms vanishes and clips numerical cosine values to $[-1,1]$. The folding score is $\sqrt{(d_E^2+d_A^2)/2}$. The class score $T_c(x)$ is the minimum folding score over the local models of class $c$.

Implementation: [autoencoder.py](../autoencoder.py), [dim_reduce.py](../dim_reduce.py), [kxx_matrix.py](../kxx_matrix.py), [kxa_matrix.py](../kxa_matrix.py), [autoencoder_filtering_extended.py](../autoencoder_filtering_extended.py), [parallel_autoencoders.py](../parallel_autoencoders.py), and [combine_multiple_autoencoders_extended.py](../combine_multiple_autoencoders_extended.py).

## 3. Soft state coordinates

For positive $\omega$ and $\tau$, set

$$
a_c(x)=1-T_c(x)+\tau,\qquad
\phi_c(x)=\frac{a_c(x)^\omega}{\sum_{j=1}^C a_j(x)^\omega}.
$$

For a reference class $c$, $\phi_c(x)=\left[1+\sum_{j\ne c}(a_j/a_c)^\omega\right]^{-1}$. Thus label agreement depends on relative affinities, rather than on the absolute reconstruction quality of the reference class alone. A larger $\omega$ concentrates mass on the largest affinities but may discard within-class predictive information.

The implementation clips computed folding scores to $[0,1]$. If the sum of the powered affinities is at most $10^{-12}$, it returns uniform coordinates for that column. Extreme parameter values therefore require numerical diagnostics; the mathematical strict positivity of the affinities does not guarantee strict positivity of every floating-point coordinate.

A hard reference map, such as nearest stored cluster center, is conceptually distinct from $\operatorname{argmax}_c\phi_c(x)$.

## 4. Linear transition fitting

Define $\Phi=[\phi(x_1),\ldots,\phi(x_N)]$ and $\Chi=[\phi(y_1),\ldots,\phi(y_N)]$. The implemented normalized least-mean-squares (NLMS) recursion updates the matrix $B$ according to

$$
B_{i+1}=B_i+
\frac{\beta}{1+\beta\|\phi(x_i)\|_2^2}
\phi(x_i)\big(\phi(y_i)-B_i^\top\phi(x_i)\big)^\top,
\qquad 0<\beta<1.
$$

The default initial matrix is the identity. Multiple epochs replay the update over the training data. The forecast is $B^\top\phi(x)$; an unconstrained $h$-step forecast is $(B^\top)^h\phi(x)$. Identity initialization and simplex-valued training pairs preserve $B\mathbf1=\mathbf1$ in exact arithmetic but do not ensure nonnegative predicted coordinates. Optional projection of each row of $B$ onto $\Delta_C$ yields a row-stochastic matrix. This is distinct from projecting each forecast vector onto the simplex, which defines a nonlinear recurrence.

For deterministic dynamics $F$, exact closure means $\phi(F(x))=B^\top\phi(x)$ for every state in the specified domain. Only under this identity is the function span of the coordinates necessarily invariant under the Koopman composition operator; redundant coordinates must be accounted for. A fitted matrix and finite held-out errors are not evidence of that exact identity.

## 5. Python interface

The implementation is in [kernel_affine_hull_koopman_machines.py](../kernel_affine_hull_koopman_machines.py). Run from the repository root or include the root on the Python module search path.

| Function | Inputs | Output |
| --- | --- | --- |
| `train_kahm_state_regime_abstraction` | Current states $(D,N)$ and reconstruction settings | Fitted class model |
| `kahm_folding_distance_matrix` | Class model and states $(D,N)$ | Scores $(C,N)$ |
| `kahm_associations` | Class model, states, $\omega,\tau$ | Coordinates $(C,N)$ |
| `nlms_koopman_closure` | Current and successor coordinates $(C,N)$ | Matrix $(C,C)$ and per-epoch error history |
| `fit_kahkm` | Paired states $(D,N)$ and model settings | `KAHKMFitResult` |
| `predict_kahkm_association` | Fitted model and current states $(D,N)$ | Predicted coordinates $(C,N)$ |
| `evaluate_kahkm` | Fitted model and held-out pairs $(D,N)$ | `KAHKMEvaluationResult` |

**Orientation:** samples are columns in this library. Certificate routines accept samples as rows, so transpose both coordinate arrays before passing them to the certificate statistics interface.

### Example: prediction in the Duffing oscillator

```python
import numpy as np
from kernel_affine_hull_koopman_machines import (
    make_duffing_snapshots,
    fit_kahkm,
    kahm_associations,
    predict_kahkm_association,
    evaluate_kahkm,
)

X0, X1 = make_duffing_snapshots(n_steps=1200, dt=0.03, seed=1)
n_train = int(0.75 * X0.shape[1])

fit = fit_kahkm(
    X0[:, :n_train], X1[:, :n_train],
    n_clusters=25, subspace_dim=4, Nb=100,
    omega=8.0, tau=1e-6, beta=0.1, nlms_epochs=20,
    random_state=0, n_jobs=1, batch_size=256,
    project_stochastic=True, verbose=False,
)

X_test, Y_test = X0[:, n_train:], X1[:, n_train:]
pred = predict_kahkm_association(fit, X_test, n_jobs=1)
target = kahm_associations(
    fit.abstraction_model, Y_test,
    omega=fit.omega, tau=fit.tau, n_jobs=1,
    show_progress=False,
)

assert pred.shape == target.shape
rmse = np.sqrt(np.mean(np.sum((pred - target)**2, axis=0)))
diagnostics = evaluate_kahkm(fit, X_test, Y_test, n_jobs=1)
print("Vector RMSE:", rmse)
print("Relative squared error:", diagnostics.closure_error)
print("Association R^2:", diagnostics.association_r2)
print("Operator spectral norm:", np.linalg.norm(fit.B, ord=2))
```

Use `use_stochastic=True` in the prediction call to select the row-projected matrix when that matrix was requested at fitting. The principal fitted object contains the local model, unprojected matrix, optional row-projected matrix, training error, update history, $\omega$, and $\tau$.

## 6. Evaluation

The vector RMSE averages squared Euclidean errors per transition, not per coordinate. Relative squared error divides the total squared prediction error by the total squared target norm (with the implementation's numerical safeguard). Association $R^2$ instead uses the total squared deviation of targets from their sample mean. High apparent accuracy in nearly constant coordinates does not alone demonstrate retention of physical-state information; inspect target variation, effective rank and simplex diagnostics as well.

Fit the representation and transition model on training data; choose hyperparameters using separate selection data. A later segment of the same trajectory is a temporally held-out evaluation, not a sample of independent transitions. For population confidence guarantees use the independent sampling design and assumptions in [Prediction-error certificates](SOFT_STATE_CERTIFICATES.md).

The numerical library has no general guarantee of state-space invertibility, long-horizon stability, or finite-sample exact Koopman closure.
