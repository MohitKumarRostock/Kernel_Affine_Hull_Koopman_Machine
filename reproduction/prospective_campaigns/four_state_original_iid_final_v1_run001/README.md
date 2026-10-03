# Final four-state certificate campaign: retained evidence

This directory archives the final controlled certificate-validation campaign.

- Run: `four_state_original_iid_final_v1_run001`
- Campaign role: `final`
- Planned and completed evaluation datasets: `117000`
- Execution source commit: `cde430dbcca09bf911439bdb5bc3f85350e232cd`
- Numerical-verifier source commit: `2ff6d61708d17907f9908208a29abe6113143412`
- Configuration SHA-256: `630b6c30f5dcbda9034b85b2a3416d92ef264953cd05133f3e53fa14e7aae182`
- Original evidence-inventory SHA-256: `851025653f42a15bac4f2f17c2c1f792562aaccf461d94b8232ca5ac06023024`
- Compressed archive SHA-256: `4776942edbd659f7a1c02ec554507f7c0259ca410449e2e5fd66f5353bf638d5`

## Contents

`evidence.tar.gz` contains the original final campaign directory and the
separate numerical-verification directory, preserving all 32 retained files
byte-for-byte.

The archive includes the full schedule, attempt accounting, sampled four-state
counts, all 117000 per-replicate certificate results, analytical references,
environment and dependency checks, source provenance, original SHA256SUMS
files, and the later independent numerical-verification report.

`ARCHIVE.json` records each archived member's byte size and SHA-256 digest.
This directory's `SHA256SUMS` covers `evidence.tar.gz`, `ARCHIVE.json`, and
this README.

## Verification status

The independent numerical verifier reported:

- 117000 / 117000 replicates verified;
- 13 analytical references verified;
- 13 final design cases verified;
- 9 sample sizes verified;
- 0 strict coverage failures;
- 0 tolerance-aware coverage failures;
- 0 raw erroneous exclusions;
- 49271 zero certificates.

The verifier recomputed empirical statistics from retained counts and
recomputed certificate quantities independently using rational and
high-precision decimal arithmetic. It did not regenerate random samples.

Exact rational design metadata was additionally used to identify threshold
boundary cases separately from raw binary floating-point comparisons.

## Reproduction

The experimental execution commit and later verifier/archive commits describe
different operations and must not be conflated.

To inspect the evidence, first verify this directory's `SHA256SUMS`, then
unpack `evidence.tar.gz` into a new directory outside the source tree. Both
restored source directories contain their original `SHA256SUMS` inventories.

The independent numerical audit entry point is:

`scripts/verify_four_state_final_numerics.py --evidence <restored-run-directory>`

Use a clean checkout containing the verifier source and its tests.

## Preservation

All 32 archived file contents were compared byte-for-byte with the retained
external originals during packaging. Neither the final experiment nor the
numerical verifier was rerun during packaging, and the external originals were
not modified.

This archive is retained evidence for the final controlled four-state study.
It does not by itself establish the theorem or independence of the random
streams.
