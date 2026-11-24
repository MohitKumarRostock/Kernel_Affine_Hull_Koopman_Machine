# KAHM feature computation and numerical conventions

This reference documents the **preserved Kernel Affine Hull Machine (KAHM)
implementation** used to construct the features in the controlled-calibration
study. It identifies the mathematical conventions, source files, numerical
guarantees, and reproducibility limits. It requires **no research manuscript,
historical manuscript PDF, or older manuscript revision**.

## Prediction setting and nested representations

A prediction-time history $X$ is mapped to learned affinity coordinates.
The controlled study compares two nested representations. The first has two
coordinates; the augmented representation retains those coordinates and
appends four more:

```math
\Gamma_1(X)=\Phi_1(X)\in\mathbb R^2,
\qquad
\Gamma_2(X)=(\Phi_1(X),\Phi_2(X))\in\mathbb R^6.
```

Each group is normalized separately. Concatenating two normalized groups
does **not** yield one six-coordinate probability simplex. Feature
development and decoder fitting are separate from iid calibration.
Future target values are not inputs to feature computation.

The implementation can construct additional blocks; the controlled
comparison uses the selected two-group prefix. See the
[controlled-calibration description](../experiments/controlled_calibration/README.md)
and [frozen experimental protocol](../experiments/controlled_calibration/protocol/PROTOCOL.md)
for the independent episode design and recorded study.

## Native geometry and PCA dimension policy

For a local reference set, let $d_x$ be the ambient input dimension,
$N_{\mathrm{loc}}$ the number of local reference columns after sampling or
augmentation, and $r_{\max}$ the prescribed subspace cap. The initial native
PCA rank request is:

```math
r_0=\min\{r_{\max},d_x,N_{\mathrm{loc}}-1\}.
```

Native dimension reduction uses the centered reference covariance and
retains leading eigenvectors with computed eigenvalues greater than
binary64 machine epsilon $2^{-52}$. The rank request is reduced while
the minimum projected coordinate range is smaller than $10^{-3}$, stopping
when that criterion is met or the requested rank reaches one. The rank-one
range criterion can remain unmet and is recorded as such.

An absence of positive empirical variance is a fitting failure. A
zero-rank fallback is available only when explicitly authorized by an
outer wrapper; it is **not** silently treated as native fitting success.
The original implementation retains its reference-storage and
floating-point conventions, including its documented float32 conversion
before the native PCA stage.

Local autoencoders produce kernel-based reconstructions from fitted
reference vectors. Reconstruction discrepancies determine soft
association scores. For a score $s_j(x)$ and its normalization block
$\mathcal B$, the preserved normalized-power evaluation has the form:

```math
w_j(x)=(1-s_j(x)+\eta)^\omega,
\qquad
\phi_j(x)=\frac{w_j(x)}{\sum_{h\in\mathcal B}w_h(x)},
\qquad
\eta=10^{-6}.
```

The fixed model determines the power parameter $\omega$ and the block
layout. Normalization is performed **within each block**.

## Interval arithmetic and numerical interpretation

The preserved forward evaluator treats retained binary32 and binary64
model inputs as exact binary rationals and encloses the corresponding
real-arithmetic forward scores using MPFI/MPFR. Interval endpoints are
converted to binary64 with outward rounding. The adaptive wrapper can
increase working precision until its stated score and feature-error
budgets are met; failed or unresolved evaluations do not become valid
enclosures by omission.

The separate normalized-soft-state evaluator propagates input-score
intervals to lower and upper feature-coordinate bounds. Its nominal
feature values use score midpoints with 160-bit MPFR arithmetic before
binary64 output. The resulting intervals concern **fixed stored inputs
and model parameters**. They are not an end-to-end error certificate
for model training, dataset acquisition, or hypothetical ideal KAHM
parameters.

## Preserved source map

The links below point to code **inside this repository**, without passing
through any previous manuscript or article PDF.

| Component | Preserved implementation |
| --- | --- |
| Original KAHM construction and folding | [Native KAHM](../experiments/controlled_calibration/source/step1_geometry_pilot/vendor/dimension_certificates/runtime/risk/kernel_affine_hull_koopman_machines.py) |
| PCA eigenvectors and eigenvalue threshold | [Native dimension reduction](../experiments/controlled_calibration/source/step1_geometry_pilot/vendor/dimension_certificates/runtime/risk/dim_reduce.py) |
| Requested ranks, range reductions, and rank audit | [Dimension policy](../experiments/controlled_calibration/source/step1_geometry_pilot/vendor/dimension_certificates/runtime/policy/kahm_dimension_policy.py) |
| Kernel reconstruction with retained references | [Local autoencoder](../experiments/controlled_calibration/source/step1_geometry_pilot/vendor/dimension_certificates/runtime/risk/autoencoder.py) |
| Development-only scaling and nested features | [Feature-map code](../experiments/controlled_calibration/source/step1_geometry_pilot/vendor/dimension_certificates/runtime/risk/extensions/certified_forecasting/src/certified_forecasting/features.py) |
| Geometry-bank preparation | [Geometry construction](../experiments/controlled_calibration/source/step6_geometry_regimes/src/geometry.py) |
| Forward score and coordinate intervals | [Python interval wrapper](../experiments/controlled_calibration/source/step11_output_accuracy/engine/certified.py) and [MPFI engine](../experiments/controlled_calibration/source/step11_output_accuracy/engine/interval.cpp) |
| Nominal soft states and feature enclosures | [Soft-state wrapper](../experiments/controlled_calibration/source/step11_output_accuracy/soft_state.py) and [160-bit MPFR evaluator](../experiments/controlled_calibration/source/step11_output_accuracy/softsrc/power_box.cpp) |

The six principal feature and interval implementation files were
independently checked against their historical source counterparts and
found byte-identical. This source map documents the **unchanged
implementation**, not a rewritten or approximate replacement.

## Reproduce the archived numerical study

From this repository's root, with the environment specified in
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md), run:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce.py --only controlled_calibration --output reproduction_controlled
```

The driver checks [MANIFEST.json](../MANIFEST.json), then recomputes the
controlled experiment's retained numerical records, predictions, and
risk-certificate decisions. It does **not** retrain native KAHM models,
regenerate all native feature vectors, or repeat the entire original
development campaign. Its population-risk interpretation retains the
stated iid calibration and independence assumptions.

This source reference is self-contained for implementation identification
and numerical conventions, while the separate experiment protocol gives
the complete retained experimental design. Historical validation reports
continue to describe their original source manifests; a new documentation
revision does not retroactively revalidate those reports.
