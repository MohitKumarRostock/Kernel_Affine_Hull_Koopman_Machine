"""Native local factors with one shared, audited PCA call sequence.

The factor arithmetic after PCA is copied from the pinned native autoencoder
at d6d74d1953a4310b73065a556df816e5ab5e3ebc. Only the dimension request and
its audit are new. A rank-zero return is the forecasting wrapper's existing
explicit fallback, not a successful native autoencoder.
"""
from __future__ import annotations

import numpy as np

from extensions.dimension_policy_bridge import DEFAULT_KAHM_CAP, fit_native_pca


def fit_local_autoencoder(reference_columns, *, upstream_loader,
                          cap=DEFAULT_KAHM_CAP, dimension_policy="native"):
    y_data = np.asarray(reference_columns, dtype=np.float32)
    projected, basis, audit = fit_native_pca(
        y_data, dim_reduce=upstream_loader("dim_reduce").dim_reduce,
        cap=cap, policy=dimension_policy, allow_rank_zero=True)
    if basis.shape[1] == 0:
        return None, audit
    covariance = np.atleast_2d(np.cov(projected.T, rowvar=False, bias=False))
    precision = np.linalg.inv(covariance)
    kernel = upstream_loader("kxx_matrix").kxx_matrix(projected, precision, "Gaussian")
    inverse = upstream_loader("kernel_regularized_least_squares").kernel_regularized_least_squares(
        kernel, y_data)
    values, vectors = np.linalg.eigh(precision)
    clipped = np.clip(values, a_min=0.0, a_max=None)
    scales = np.sqrt(clipped).astype(vectors.dtype, copy=False)
    whitening = (vectors * scales[None, :]) @ vectors.T
    return {"y_data_n_subspace": projected, "PC": basis, "kerneltype": "Gaussian",
            "B": inverse, "y_data": y_data, "sqrt_weights_matrix": whitening}, audit
