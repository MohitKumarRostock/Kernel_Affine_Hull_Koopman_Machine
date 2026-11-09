# Forecasting construction correction — Step 1

This is an isolated implementation derivative. It changes the local subspace
dimension policy and records its effect on each actual reference set. It does
not rerun a forecasting study, score the archived evaluation outcomes, revise a
manuscript, or establish that a larger rank improves accuracy.

## Policy and retained numerical rules

The default requested local rank is now

`min(20, ambient_dimension, local_reference_count - 1)`.

The cap is configurable. The configuration key `rank` denotes that requested
cap, not the effective rank of every local model. The extra ambient-minus-one
restriction is absent from the native policy. A well-spread scalar input can
therefore use rank one; inputs of dimension 12 or 16 can use their full ambient
dimension when the references support it.

All dimension requests and PCA/range decisions use the shared sibling module
`../policy/kahm_dimension_policy.py`. References retain the original float32
quantization before the pinned float64 PCA calculation. The native positive
eigenvalue threshold and projected-range reduction remain unchanged. Reduction
stops at rank one even if the range threshold is not met, and the audit records
that condition. The forecasting wrappers retain their existing explicitly
logged zero-rank behavior for singleton or constant reference sets. Other
failures are not relabeled as successful fits.

There is no rank selection based on outcomes. Training, selection, calibration,
residual chronology, feature transformations, penalty grids and query-only
prediction interfaces retain their existing roles. A changed local rank does
change the representation and derived feature dimensions/radii; it is a changed
forecasting procedure whose accuracy and cost remain to be evaluated.

## Explicit historical replay

Historical rank-two replay requires both settings:

```python
{"rank": 2, "dimension_policy": "legacy_normal_departure"}
```

This retains `min(2, d-1, Nc-1)` and the old rejection of scalar inputs.
`{"rank": 2, "dimension_policy": "native"}` is a deliberate native cap-two
ablation, which differs from the historical policy for a two-dimensional input.
Do not describe those two policies as interchangeable.

The immutable `reference_current/` directory contains the previously used
runtime, adapters, manifests, source differences and licenses. It is used only
by explicit parity checks. It is never imported by the corrected fit path.
Because the source modules use common import names, corrected and historical
trees must be run in separate Python processes; the integrity checks reject a
mixed import state.

## Practical entry points

Add `forecasting_corrected/code` to Python's module path. The adapters locate
their vendored runtime and shared policy inside this standalone Step 1 package.

**Main manuscript's principal bounded experiment**

```python
from principal_adapter import fit_principal_suite

fitted = fit_principal_suite(
    A_X, A_Y, B_X, B_Y, C_X, C_Y,
    seed=seed, horizons=(1, 4, 12),
    rank_cap=20, dimension_policy="native",
)
forecasts = fitted.predict(query_histories)
```

This calls the actual ten-method
`extensions.reconstruction_mechanism.models.fit_suite` runtime, with standard
geometry scaling. The lower-level `geometry_adapter.fit_scaling_suite` retains
the six explicit scaler choices and accepts `config={"rank": ..., "dimension_policy": ...}`.

**Current centered forecasting pipeline**

```python
from centered_adapter import fit_centered_suite

fitted = fit_centered_suite(
    stage_inputs, stage_targets, stage_origins, residual_fold_ids,
    seed=seed, arm="standard", horizons=(1, 4, 12),
    config={"rank": 20, "dimension_policy": "native"},
)
forecasts = fitted.predict(query_histories)
```

Its existing stage-validation rules remain active. The fixed-common adapter
also accepts the same explicit configuration. Neither prediction API accepts
query targets.

The policy is propagated through `LocalFeatureBank`, `RobustLocalFeatureBank`,
`SharedBank`, the instrumented principal bank and the adapters. The controlled
study's `GeometryBank` uses the same helper. The direct-reconstruction module
inherits the corrected bank and default configuration. The existing natural
`ForecastRegressor` forwards the corrected geometry defaults.

The supplemental native local factor implementation calls the shared PCA
helper once, then retains the pinned autoencoder's kernel and regularization
arithmetic. It avoids a second diagnostic PCA pass. Its factor arrays are
checked bitwise against the unmodified native autoencoder.

## Auditable local ranks

Every local model records `dimension_audit`, containing the configured cap,
ambient dimension, actual retained reference count, initial admissible rank,
limiting constraints, effective rank, reduction reasons and native PCA trace.
The audit distinguishes a fitted rank from an inherited rank-zero fallback.
The positive-eigenvalue total is left unknown when the returned basis only
identifies a lower bound; no extra eigendecomposition is added for that log.

These records appear in local bank metadata, `SharedBank.diagnostics`, and
serialized bank state. The effective `rank` and `requested_rank` fields remain
available for existing consumers. KAHM, affine and matched RBF constructions
share the actual reference rows and PCA bases.

## Verification and dependencies

From the Step 1 package root:

```bash
python3 tests/test_forecasting_policy.py --output verification/forecasting_policy_checks.json
```

The test uses ordinary Python assertions, NumPy, SciPy, scikit-learn and
threadpoolctl; pytest is not required. The original numerical version pins are
retained at
`code/vendor_geometry/kahm_geometry/extensions/reconstruction_mechanism/requirements.txt`.
The checks were run with Python 3.12.14 and set numerical-library thread counts
to one. Matplotlib is listed by the historical study but is not imported by
these construction checks.

Checks cover dimensions 1, 2, 12, 16, 24 and 48; full ambient rank;
reference-limited and explicitly capped fits; scalar range floors; constant
references; all matched local controls; actual principal and centered entry
points; nondefault cap propagation; and strict rejection of invalid caps.
Both principal and centered historical paths are run in fresh subprocesses
against the bundled reference. The receipt records exact array comparisons and
selection equality. All inputs and fitting/selection targets are fabricated;
no terminal target is generated or supplied.

## Provenance and coverage

The forecasting runtime derives from commit
`7dda356138675934aab5e327ec667bde0a5eda79`, with the retained tanh-scaling
derivative as its immediate base. Its nested native numerical source derives
from `d6d74d1953a4310b73065a556df816e5ab5e3ebc`.

`provenance/` contains the source-copy hashes, native source manifest and
corrected runtime inventory. The principal reproduction implementation is
located under `extensions/reconstruction_mechanism`, and the historical
reference uses local rank two. The historical inventories also identify two
manuscript passages whose contents are outside the computational distribution.
Their original hashes remain available as provenance identifiers. Source
verification accounts for these two omissions explicitly and does not claim
to verify the manuscript contents. The extension's
[source-distribution description](../../../README.md#source-distribution-and-verification)
specifies the available-source check and its scope.

The runtime manifest verifies the corrected code, shared policy identity and
bundled historical reference. It is not regenerated by verification. After an
intentional code change, the explicit maintenance command is
`python3 forecasting_corrected/provenance/build_runtime_manifest.py`; doing so
creates a new source identity and requires new verification.
The older `DERIVATIVE_MANIFEST.json` and scaling differences retained under
`code/vendor_geometry/` describe the parent tanh derivative. The authoritative
inventory for this corrected implementation is
`provenance/FORECASTING_RUNTIME_MANIFEST.json`.

This package supplies the complete principal fitting/query implementation and
the current centered pipeline. It does not replace historical task generators,
saved forecasts, scoring analyses, or every separate supporting-study runner.
In particular, the older online experiment's driver is outside this runtime
subset. Those experiment-specific entry points must be reviewed when a later
step authorizes their revision or rerun.
