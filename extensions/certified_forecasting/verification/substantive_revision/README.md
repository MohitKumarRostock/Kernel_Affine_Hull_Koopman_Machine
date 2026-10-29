# Verification of the scientific extension

This directory records checks of the revision relative to commit
`c15a7526f64b4b47ac09998548a6beee1f763c22`. Earlier verification directories
retain their historical scope and are not overwritten.

- `core_tests.log`: 54 standard tests, including the four new causal speech,
  grouping, and KL checks; all pass.
- `witness_tests.log`: seven analytic and independently formulated benchmark
  tests; all pass.
- `retained_statistics.json`: reconstruction of the original studies' saved
  moments, 5,388 lower/upper certificate pairs, 792 improvement bounds,
  original tables, and the 28,800-trial archive.
- `revision_tables.json`: independent reaggregation of absolute MSE,
  fitting variation, relative effects and seed counts in the combined matched
  table, and all speech table entries.
- `speech_portable_verification.json`: a successful run of the portable
  independent raw-audio, frozen-selection, loss, endpoint, and replay verifier.
- `layout.json` and `pdf_fonts.txt`: final PDF hash, page/abstract counts,
  source hashes, build checks, and rendered-page review.

The new benchmark and decision-study numerical reports are retained with their
results in `results/witness_comparison/analysis/audit.json` and
`results/decision_value/verification.json`. Their protocols, sources, raw
records, and statistical scopes are described beside those reports.

All 61 tests passed. Original moment reconstruction does not recreate missing
historical episode arrays. Floating-point primal/dual checks are not formal
interval arithmetic. Numerical audits do not establish population sampling
assumptions, an exhaustive novelty assessment, or editorial acceptance.
