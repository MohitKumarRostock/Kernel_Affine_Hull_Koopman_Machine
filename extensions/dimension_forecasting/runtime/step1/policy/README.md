# Shared KAHM construction dimension policy

The default cap is 20. For each actual local reference matrix with `d` rows and
`N` columns, the native initial request is `min(cap, d, N-1)`. No normal direction
is reserved. An explicit integer cap, including 2, is an intentional override.
The separately named `legacy_normal_departure` policy uses `d-1` only for a
deliberate replay of the historical forecasting construction.

`fit_native_pca` accepts the unchanged pinned `dim_reduce` callback. The callback
computes its original centered, sample-covariance PCA in float64 and keeps only
eigenvalues above float64 machine epsilon. The caller supplies references after
its existing storage conversion; native autoencoder references are float32.
Projected-coordinate ranges are then checked against `1e-3`, decreasing the
requested rank until the range check passes or the requested rank reaches one.
The original rank-one stopping rule is preserved even when the threshold still
fails. This condition is explicit in the audit.

The original low-level encoder does not support singleton or constant reference
matrices. The default policy therefore raises a clear error with a
`dimension_audit` attribute for these cases. `allow_rank_zero=True` is reserved
for the forecasting wrapper's inherited fallback. It returns empty projected
coordinates/basis and labels the result `inherited_zero_rank_fallback`; this is
not called a successful native fit. No generic numerical fallback is added.

## API

```python
from kahm_dimension_policy import dimension_request, fit_native_pca

request = dimension_request(d, N, cap=20, policy="native")
projected, basis, audit = fit_native_pca(
    reference_columns, dim_reduce=pinned_dim_reduce,
    cap=20, policy="native", min_range=1e-3,
)
assert audit["effective_rank"] == basis.shape[1]
```

Every audit separates the supplied cap, ambient dimension, actual local sample
count, initial admissible rank and actual basis-column count. It records binding
initial limits and later eigenvalue/range reductions. A `pca_trace` stores each
requested and returned rank and its smallest projected coordinate range.
The `policy_source_sha256` field identifies the exact shared source file loaded
for the request; it is computed once at import without numerical work.

No diagnostic eigensolve is performed. If the requested cap truncates the
returned eigensystem, the total positive-eigenvalue count may be unknown. In that
case `empirical_positive_eigenvalue_count` is null, its `exact` flag is false,
and a lower bound is recorded. A returned rank below its request or a full
ambient return identifies the exact total. `native_pca_calls` counts calls to
the original PCA callback; `diagnostic_eigendecomposition_count` remains zero.

This policy performs no outcome-based dimension selection. Construction-only
fixtures establish numerical parity with the pinned native encoder; they do not
establish forecasting performance or comparable execution times. Clipping the
initial request to the ambient dimension avoids redundant native calls that
could otherwise occur when a raw over-ambient request encounters a small range.
The resulting projector, kernel factors and reconstruction remain unchanged on
the tested parity fixtures.
