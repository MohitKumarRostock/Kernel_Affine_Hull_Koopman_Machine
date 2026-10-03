# Four-state certificate pilot: retained evidence

This directory archives pilot evidence, not a final experimental campaign.

- Run: `four_state_original_iid_pilot_v1_run001`
- Execution source commit: `a6408039a33311092f78fa94bb90032071e509da`
- Numerical-verifier source commit: `d98ccf4e0b655e69f489c62338207357347636ea`
- Original evidence-inventory SHA-256: `2e956771d05d425a99a9d4805cd96a51d5928c334562128d4317715c437bbd76`
- Compressed archive SHA-256: `813d68f8eeccba80b5cfa8d58f4524d88d16eb4389c593a6af49d6cbab0174a1`

## Contents

`evidence.tar.gz` contains the original pilot directory (27 files) and the
separate numerical-verification directory (5 files), with their original
names and file contents. `ARCHIVE.json` records every member's size and hash.
This directory's `SHA256SUMS` covers the archive, this README, and ARCHIVE.json.

The archive retains 10,240 evaluation records, their samples and seed metadata,
the full schedule and attempt log, analytical references, configuration,
source provenance, preflight logs, and the later numerical-verification report.

## Verification and interpretation

The retained independent numerical report passed for 10,240 replicates and
8 references. It recomputes from saved counts; it does not regenerate samples.
The original campaign summary still says `not_yet_performed` for independent
verification because it predates that report. Do not rewrite that summary.

Check the archive checksum, then unpack into a new directory outside the source
working tree. Both restored directories retain their original SHA256SUMS files.
The numerical audit entry point is `scripts/verify_four_state_pilot_numerics.py`
with `--evidence` pointing to the restored pilot directory. Use a clean checkout
containing the verifier. The experimental execution commit and the later
verification/archive commits describe different operations.

## Preservation

All 32 archived file contents were compared byte-for-byte after decompression.
The external original directories were left unchanged. Only archive metadata
was normalized (file order, ownership, permissions, and timestamps); timestamps
inside the retained records were not changed. These settings do not claim
identical compressed bytes across different Python or compression-library versions.

This is an archival copy of one pilot run, not a new run or a replacement for
historical results. Its numerical outcomes are not proof of statistical coverage.
