# Independent verification of zero-rank local banks

The native admissible dimension is `min(cap, d, N_local - 1)`. A local
bank with one reference therefore has dimension zero. Its covariance
normalization by `N_local - 1` is undefined and is unnecessary for verifying
the required empty basis.

The separately recorded [verifier](verify_results.py) handles this
zero-admissible-rank case before calculating a covariance. For banks with
positive admissible dimension, it retains the frozen covariance calculation,
eigenvalue threshold, range reduction, and cached eigendecomposition. Both
zero-rank paths retain the original checks of the fitted basis shape,
projector, reference membership, transformed reference values, and dimension
metadata. Two counters distinguish zero admissible dimension from zero
positive covariance rank.

The original `../verify_results.py`, `../verify_crossfit.py`, experiment
configuration, analysis, and source freeze remain unchanged. The correction
is restricted to `selected_native_bank`; [the exact diff](verifier_change.patch)
and [provenance](correction_provenance.json) identify the versions. The frozen
manufactured-data oracle's `native_basis` function also calculates covariance
before checking zero admissible rank. That function is unchanged and is not
called by the production verifier's selected-bank checks.

## Regression evidence

[check_zero_rank_verifier.py](check_zero_rank_verifier.py) uses manufactured
inputs and the frozen preprocessing function. Its independent reference basis
uses SVD. Across standard, min-max, and tanh preprocessing, the checks cover:

- The original singleton failure and the corrected empty-basis verification.
- Rejection of a nonempty singleton basis and inconsistent dimension metadata.
- Constant banks containing several references.
- Positive-rank banks at caps 1, 2, 4, and 12, including cached reuse.

The [regression receipt](pca_zero_rank_regression.json) contains 27 successful
checks: 18 accepted valid cases and 9 expected failures or rejections. It
also verifies all 17 frozen file hashes and confirms by AST comparison that
all pre-existing selected-bank assertions, and all code outside the corrected
function, remain unchanged. Original and corrected positive-rank cases have
identical audit counts, ranks, and cached PCA arrays.

This receipt verifies the implementation correction only. No production
data, fitted state, forecast, or result table is read. It does not establish
the validity of any experimental performance result.

From the repository root, a new regression receipt can be produced with:

```bash
python extensions/crossfit_residual_v2/audit/post_release/check_zero_rank_verifier.py \
  --output /path/to/new_pca_regression_receipt.json
```

## Production verification after record restoration

Restore the original experiment records, including the sibling
`crossfit_residual` provenance, before running the corrected verifier. Keep
the frozen source in its original relative directory structure. Then run:

```bash
python extensions/crossfit_residual_v2/audit/post_release/verify_results.py \
  --root /path/to/restored/extensions/crossfit_residual_v2 \
  --source-root extensions/crossfit_residual_v2 \
  --output /path/to/new_full_production_audit_receipt.json
```

The verifier requires the existing release record, complete sealed fits,
predictions, targets, and completed frozen analysis. It preserves the
original release, provenance, and source-hash gates. Missing evidence causes
failure. Existing audit receipts are never overwritten. The new production
receipt records the corrected verifier's own SHA-256.
