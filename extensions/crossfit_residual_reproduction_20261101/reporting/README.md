# Archiving the verified computational reproduction

`build_archives.py` creates the complete retained numerical evidence package
after the 192-task execution, scientific analysis, and independent reproduction
audit have finished. It uses the Python standard library and does not alter
scientific files. Reporting utilities are separate from the scientific freeze.

The tool requires `verification/reproduction_audit.json` with status `passed`,
mode `computational_reproduction`, and coverage of all 192 tasks, 8,064 forecast
arrays, and 24,192 horizon losses. It checks the source/configuration identities,
both task seals, target release, complete task receipts, analysis table hashes,
audit bindings, and execution chronology before producing archives. Archive
members must also agree with every applicable retained file hash. Source code
is obtained from the pinned published checkout and is not duplicated in these
numerical packages.

`PUBLISHED_SOURCE.json` must contain a full 40-character
`published_source_commit` and `freeze_sha256`. The latter must equal the current
source-freeze hash. The publication receipt is itself included and hashed; the
offline archiver makes no remote GitHub request and does not independently
authenticate publication.

## Command

Run only after production and audit completion, choosing an output directory
outside the Git checkout:

```bash
python reporting/build_archives.py \
  --output-dir /workspace/scratch/2332edd06992/crossfit_reproduction_archives
```

The default limit is 400 MiB of input per standalone records archive. A single
file larger than the selected limit, or a compact analysis inventory larger
than the limit, causes a clear failure before writing archives. The utility
never splits an individual numerical file into operating-system fragments.

## Retained inventory

| Deliverable | Inventory |
| --- | --- |
| `KAHM_Crossfit_Reproduction_Records_NN.zip` | Every regular numerical/evidence file under `prepared/`, `results/`, `predictions/`, and `execution/`, grouped in deterministic path order. |
| `KAHM_Crossfit_Reproduction_Analysis.zip` | Analysis outputs, verification/readiness evidence, configuration, source freeze, both stage seals, target release, original-source manifest, and publication receipt. |
| `KAHM_Crossfit_Reproduction_Archive_Manifest.json` | Every archived file's relative path, size, SHA-256, CRC32, archive membership and compression; every archive's size, SHA-256 and verification outcome; source pins and scope. |

Repository source and documentation extensions encountered in these scopes are
explicitly listed as exclusions in the manifest. Numerical archives use stored
ZIP entries for already compressed NPZ/GZ and image files; other entries use
deflate. Each member path retains the checkout-relative
`extensions/crossfit_residual_reproduction_20261101/` prefix, allowing all
standalone ZIPs to be extracted over the pinned published checkout.

## Completeness and failure handling

Every ZIP member is read once for independent SHA-256, size and CRC32
verification after writing. The ZIP reader checks its internal CRC as well.
The complete archive set must cover the declared inventory exactly without
duplicate relative files. File identities, sizes and modification times are
checked for changes during construction, and every applicable seal hash must
match the bytes actually archived.

Outputs are assembled in a clearly named staging directory. Individual
deliverables are promoted using atomic no-overwrite links only after all
checks pass. The manifest is promoted last and is the completed-set commit
marker. Every final destination ZIP is checked again for exact size and SHA-256
before that manifest is promoted. A failed build preserves staged files with `ARCHIVE_BUILD_FAILED.json`
and removes only final links created by that failed build. Existing artifacts
are never overwritten. Interrupted or failed staging directories are not
deliverables and must not be uploaded as completed evidence.

The archive retains the reproduction's explicit limitations: the same declared
simulation cohort is reused, no independent repetitions are added, and the
unavailable original completed-run records are not authenticated.

## Delivery verification

The [archive delivery receipt](archive_delivery_verification.json) records a
post-build size/hash discrepancy in the first records archive, preservation of
the invalid bytes, reconstruction against the unchanged original manifest, and
size/SHA-256 checks for all final destination ZIPs. The cause of the discrepancy
has not been established. The original manifest-pinned builder is preserved in
[history/build_archives_initial.py](history/build_archives_initial.py); the
current utility adds the final-destination guard described above. This reporting
provenance does not change scientific evidence or the original archive inventory.
