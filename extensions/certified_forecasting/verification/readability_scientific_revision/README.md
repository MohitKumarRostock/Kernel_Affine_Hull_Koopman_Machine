# Readability and scientific-consistency verification

These records accompany the 29 October 2026 revision of the forecasting
manuscript and repository documentation. The
[scientific verification report](../../docs/SCIENTIFIC_VERIFICATION.md)
describes the findings, methods, reproduction commands, and limitations.

| File | Contents |
|---|---|
| `retained_statistics.json` | Independent reconstruction of original tables, confidence arithmetic, exact-risk values, and writer aggregation |
| `scaling_statistics.json` | Independent reconstruction of original-seed and confirmation scaling selections and mean effects |
| `matched_statistics.json` | Independent reconstruction of all primary matched-geometry selections and contrasts |
| `mathematical_checks.json` | Finite-law and numerical checks supporting the direct mathematical review |
| `integrity.json` | Manuscript, numerical-evidence preservation, build, and test checks |
| `unit_tests.txt` | Complete successful run of 50 tests, with no skips |
| `manifest.json` | Environment, baseline revision, and SHA-256 hashes of the verification records |

The mathematical checks supplement the proofs; finite examples do not prove
universal coverage. Original-synthetic endpoint reconstruction uses saved
sufficient statistics, while writer and scalar-record checks use the retained
individual records identified by their hashes. Full production model fitting
and all full episode-score audits were not rerun.

Paths beginning with `records.tar.gz::` in the matched verification record
identify members of the retained record archive. Other input paths are relative
to the forecasting extension. Historical study records and source snapshots
remain unchanged.
