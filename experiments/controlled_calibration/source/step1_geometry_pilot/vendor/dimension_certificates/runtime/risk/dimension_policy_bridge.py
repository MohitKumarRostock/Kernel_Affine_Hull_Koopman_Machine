"""Load the one shared construction policy packaged beside this source tree."""
from pathlib import Path
import sys

_policy_dir = Path(__file__).resolve().parents[1] / 'policy'
_module_path = _policy_dir / 'kahm_dimension_policy.py'
_loaded = sys.modules.get('kahm_dimension_policy')
if _loaded is not None and Path(_loaded.__file__).resolve() != _module_path:
    raise ImportError('A different KAHM dimension policy is already imported; use an isolated process.')
if str(_policy_dir) not in sys.path:
    sys.path.insert(0, str(_policy_dir))
from kahm_dimension_policy import (  # noqa: E402,F401
    DEFAULT_KAHM_CAP, NATIVE_POLICY, POLICY_SOURCE_SHA256, SCHEMA, dimension_request,
    finalize_dimension_audit, fit_native_pca, validate_dimension_cap,
)
