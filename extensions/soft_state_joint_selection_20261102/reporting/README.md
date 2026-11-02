# Complete numerical archives and compact result records

These utilities form a reporting layer outside the prospective scientific
freeze. They preserve the approved source and numerical bytes and perform no
fitting, selection, forecast scoring, or statistical recomputation. Run them
only after the complete frozen analysis and independent audit have passed.

The scientific reference is public commit
`8e0250279074aefa39079fa33205d3c74f835848`, with repository tree
`1946da072f230244ad92795f9517ef89d1f06487`. The publication receipt and prepared
manifests bind that reference to this execution. The earlier local source commit
is not substituted for the actual public reference.

## Completion and identity requirements

The numerical packager requires all 256 tasks and 32 independent repetitions,
the complete six-contrast primary family, 4,096 terminal forecasts, 4,096
diagnostic C forecasts, and 76,800 independently replayed B penalty-path arrays.
The passed full audit must identify the exact completed analysis. Source,
configuration, publication, preparation, fit, prediction and target-release
hashes must agree. Every sealed numerical file and every original analysis
table must be retained. An incomplete cohort or a missing record stops
packaging; no unsuccessful configuration is silently omitted.

## Create complete numerical archives

From the repository root, choose a **new directory outside the repository**:

```bash
python3 extensions/soft_state_joint_selection_20261102/reporting/package_numerical_records.py \
  --output-dir /tmp/KAHM_Joint_Selection_Numerical_Records
```

The default layout contains five ZIP files:

| Archive | Retained records |
| --- | --- |
| `Data_and_Forecasts.zip` | All prepared physical episodes, noise and retained windows; D queries and targets; terminal forecasts and diagnostics; original analysis and full audit; readiness and source/stage metadata; prediction and phase-driver logs |
| `Fitted_Models_00_07.zip` | Every candidate state, B penalty path, C diagnostic, selection/numerical record and fit log for repetitions 00–07 |
| `Fitted_Models_08_15.zip` | The same complete records for repetitions 08–15 |
| `Fitted_Models_16_23.zip` | The same complete records for repetitions 16–23 |
| `Fitted_Models_24_31.zip` | The same complete records for repetitions 24–31 |

Each model archive holds 64 complete task suites. Fit-lock and fit-stage
metadata are intentionally duplicated between model archives. Fitted task
states are not duplicated. The source implementation is retrieved from its
pinned public commit; numerical ZIPs include exact source hashes and non-code
contracts but do not duplicate scientific code.

Each ZIP must be **strictly smaller than 500,000,000 bytes**. The limit concerns
the completed file, including its ZIP directory and generated metadata. A size
failure leaves no accepted partial output directory. If measured sizes require
finer partitions, repeat into a new destination using, for example:

```bash
python3 extensions/soft_state_joint_selection_20261102/reporting/package_numerical_records.py \
  --output-dir /tmp/KAHM_Joint_Selection_Numerical_Records_Smaller \
  --model-repetitions-per-archive 4 --data-parts 2
```

Model shard sizes may be 8, 4, 2 or 1 complete repetitions. Multiple data parts
use a deterministic size-balanced partition of whole files; no numerical array
is converted, filtered or divided. The archive index gives the exact final
layout. Every listed archive is required for a complete numerical replay.

NPZ files are stored without another compression pass. Text records are
losslessly deflated. All ZIP member timestamps and file modes are fixed. After
closing every archive, the packager reopens **every member**, checks its length
and CRC, and verifies SHA-256 against the original source snapshot. The final
`numerical_archive_index.json` contains archive and member hashes, canonical
task allocation, complete file coverage, intentional duplicate metadata and
the origin of each sealed hash. Ancillary logs outside scientific seals are
identified as packaging snapshots.

## Create compact Git records

After the numerical archives have been verified:

```bash
python3 extensions/soft_state_joint_selection_20261102/reporting/build_compact_release.py \
  --archive-index /tmp/KAHM_Joint_Selection_Numerical_Records/numerical_archive_index.json
```

The builder writes lossless gzip copies of:

| Original record | Compact copy |
| --- | --- |
| `analysis/B_candidate_scores.csv` | `analysis/B_candidate_scores.csv.gz` |
| `analysis/basis_dimensions.csv` | `analysis/basis_dimensions.csv.gz` |
| `verification/independent_audit.json` | `verification/independent_audit.json.gz` |

Gzip timestamps are zero and the original-filename header is empty. Every
decompressed byte, including CSV line endings, matches its original. No rows,
columns, precision or JSON fields are removed from these full records. The
originals remain unchanged in the completed evidence and numerical archives.

The builder also writes:

- `verification/independent_audit_summary.json`: all original audit top-level
  fields except `tasks_detail`, with a distinct summary schema and links by
  size/hash to the complete original and compressed receipt. The original audit
  timestamp is retained and the summary has a separate creation timestamp.
- `reporting/compact_release_manifest.json`: original/compressed identities,
  complete analysis-table identities, source and publication bindings, and the
  restoration contract.
- `reporting/compact_release_validation.json`: exact-byte compression checks,
  unchanged source/table checks, and a successful check-only invocation of the
  restoration helper on the actual completed records.
- `reporting/numerical_archive_index.json.gz` and
  `reporting/numerical_archive_summary.json`: the full member index and a compact
  record of archive identities and verification scope.

The audit summary is neither another execution of the audit nor a replacement
input for the frozen analyzer/verifier. Existing compact outputs are never
overwritten. The distribution status of the separate ZIPs is recorded only
after their persistence succeeds; these utilities do not invent download URLs.

## Restore compact records

```bash
python3 extensions/soft_state_joint_selection_20261102/reporting/restore_compact_records.py --check-only
python3 extensions/soft_state_joint_selection_20261102/reporting/restore_compact_records.py
```

The first command checks gzip and original byte identities without writing
originals. The second restores only the three declared records. Existing
identical originals are accepted. Any conflicting original or corrupt gzip is
rejected before restoration; no original is overwritten. This operation does
not import scientific modules or recompute forecasts or statistics.

## Three reproduction scopes

1. **Compile the manuscript.** Use the supplied generated tables/figures and the
   manuscript build guide. Numerical archives and fitting are unnecessary.
2. **Regenerate manuscript displays.** Restore the compressed records, then run
   the paper's joint-selection display builder into a fresh output directory.
   Its data inputs are the completed tables and full audit.
3. **Replay the complete computation.** Restore every numerical ZIP into a
   complete checkout containing the pinned scientific source, verify archive
   hashes against the supplied member index, and run the frozen analyzer and
   independent verifier with new output paths:

```bash
python3 extensions/soft_state_joint_selection_20261102/analysis_code/analyze_soft_state.py \
  --output /tmp/KAHM_Joint_Selection_Reanalysis
python3 extensions/soft_state_joint_selection_20261102/audit/verify_soft_state.py \
  --analysis /tmp/KAHM_Joint_Selection_Reanalysis \
  --output /tmp/KAHM_Joint_Selection_Replayed_Audit.json
```

The archive paths begin with
`extensions/soft_state_joint_selection_20261102/`. Restore only into an empty
record tree or after checking existing files for exact identity. Recorded
workflow order and hash agreement are procedural evidence, not independent
oversight or cryptographic target isolation. Packaging verification does not
replace the completed scientific audit or constitute an independent scientific
replication.
