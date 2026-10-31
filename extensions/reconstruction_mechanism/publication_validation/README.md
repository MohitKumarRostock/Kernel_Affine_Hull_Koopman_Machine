# Verification records for the experimental evidence

This directory contains the checks performed when the reconstruction-departure results were first prepared for public release. Each record identifies the files and environment that it checked. A record for a previous manuscript archive does not verify a later document edition.

| Record | Verification scope |
| --- | --- |
| `independent_recheck.json` | Independent-code evaluation of the saved evidence without refitting a forecasting model |
| `contract_tests.log` | The 38 numerical, data-partition, and statistical-analysis tests |
| `transport_verify.json` | Hashes of all 77 lossless parts and 1,631 archived members |
| `release_check.json` | Compilation and extraction checks for the archives identified by the hashes in that record; exact replay of all 1,920 selected prediction arrays from the extracted reproduction package |
| `LOCAL_EXECUTION_PROVENANCE.json` | Original local Git objects and identities, including the prospective scientific checkpoint |

The original release-check record reports a 99-page source build with no unresolved references or overflow warnings. That page count describes the checked historical document, not subsequent manuscript editions. Archive hashes identify the exact package to which a verification claim applies.

The local Git objects record execution provenance. They do not establish public preregistration or the ancestry of a subsequently published branch. The original execution, scientific checkpoint, predictions, statistical decision, audit, and replay remain identified by their respective source and output hashes.
