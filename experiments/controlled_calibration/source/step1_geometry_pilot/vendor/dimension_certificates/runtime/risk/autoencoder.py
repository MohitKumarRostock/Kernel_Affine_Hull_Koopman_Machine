import numpy as np
from dimension_policy_bridge import DEFAULT_KAHM_CAP, fit_native_pca

def autoencoder(y_data: np.ndarray, subspace_dim: int = DEFAULT_KAHM_CAP,
                min_range: float = 1e-3):
    y_data = np.asarray(y_data, dtype=np.float32)
    D, N = y_data.shape

    # The declared native request is min(cap, D, N-1); no normal direction
    # is reserved. Keep original float32 storage and original PCA callback.
    from dim_reduce import dim_reduce
    y_data_n_subspace, PC, dimension_audit = fit_native_pca(
        y_data, dim_reduce=dim_reduce, cap=subspace_dim, min_range=min_range,
    )

    # Estimate covariance across the projected feature coordinates.
    cov_matrix = np.cov(y_data_n_subspace.T, rowvar=False, bias=False)
    cov_matrix = np.atleast_2d(cov_matrix)   # ensure 2D even when 1D subspace

    # The Gaussian metric uses the inverse projected covariance.
    weights_matrix = np.linalg.inv(cov_matrix)

    kerneltype = "Gaussian"

    from kxx_matrix import kxx_matrix
    from kernel_regularized_least_squares import kernel_regularized_least_squares

    Kxx = kxx_matrix(y_data_n_subspace, weights_matrix, kerneltype)
    B = kernel_regularized_least_squares(Kxx, y_data)

    # matrix square root of weights_matrix
    eigvals, eigvecs = np.linalg.eigh(weights_matrix)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    s = np.sqrt(eigvals_clipped).astype(eigvecs.dtype, copy=False)
    sqrt_weights_matrix = (eigvecs * s[None, :]) @ eigvecs.T


    AE = {
        "y_data_n_subspace": y_data_n_subspace,
        "PC": PC,
        "kerneltype": kerneltype,
        "B": B,
        "y_data": y_data,
        "sqrt_weights_matrix": sqrt_weights_matrix,
        "dimension_audit": dimension_audit,
    }

    return AE
