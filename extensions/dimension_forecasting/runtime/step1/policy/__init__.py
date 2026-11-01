"""Shared construction policy for the isolated KAHM dimension correction."""
from .kahm_dimension_policy import (
    DEFAULT_KAHM_CAP, NATIVE_POLICY, LEGACY_NORMAL_DEPARTURE_POLICY,
    dimension_request, finalize_dimension_audit, fit_native_pca,
    validate_dimension_cap,
)
