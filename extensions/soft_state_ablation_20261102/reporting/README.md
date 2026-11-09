# Compact completed-result records

The scientific run and independent audit are complete. This directory contains
reporting utilities added after that run; it is outside the prospectively
published scientific freeze. The procedures, predictions, outcomes, original
analysis tables, and full audit receipt are unchanged.

## Published compact records

Three large records are published as lossless gzip files:

| Original record | Published compressed copy |
| --- | --- |
| `analysis/B_candidate_scores.csv` | `analysis/B_candidate_scores.csv.gz` |
| `analysis/basis_dimensions.csv` | `analysis/basis_dimensions.csv.gz` |
| `verification/independent_audit.json` | `verification/independent_audit.json.gz` |

The gzip timestamp is zero and its original-filename header is empty. Gzip
decompression yields the original bytes, including the CSV line endings; no
rounding, row selection, column removal, or JSON reformatting is applied.
[compact_release_manifest.json](compact_release_manifest.json) gives compressed
and uncompressed sizes and SHA256 hashes. Other analysis CSVs and the analysis
summary remain plain files. The full originals are also retained in the
separately indexed numerical archives.

[independent_audit_summary.json](../verification/independent_audit_summary.json)
keeps every full-audit top-level field except `tasks_detail`. Its distinct schema
identifies the summary, `source_audit_schema` preserves the original schema,
and `full_audit` binds the full receipt and its compressed copy. The source
audit timestamp is retained; the summary has a separate creation timestamp.
This summary is neither a second audit execution nor a substitute input for
the frozen verifier or complete display builder.

## Restore and check

From the repository root:

```bash
python extensions/soft_state_ablation_20261102/reporting/restore_compact_records.py --check-only
python extensions/soft_state_ablation_20261102/reporting/restore_compact_records.py
```

The first command verifies compressed identities and decompressed bytes without
writing originals. The second restores only the three named paths. An existing
file is accepted only when its size and SHA256 are identical. A conflicting
file is rejected before restoration; it is never overwritten. Temporary
restoration files are removed on completion or failure. No scientific modules
are imported and no forecasts or statistics are recomputed.

The restored files match the original hashes in `analysis/analysis_summary.json`
and the original full-audit identity. They can therefore be used with the
unchanged reporting builder and inspected with ordinary CSV/JSON tools.

## Statistical checks and computational replay

The independent primary checker recomputes all four paired comparisons from the compact analysis tables:

```bash
python reproduction/audits/local_position_forecasting/verify_primary.py \
  --evidence extensions/soft_state_ablation_20261102/analysis \
  --output /tmp/local_position_checks.json
```

After restoring the compact records, the reporting source can verify the full audit binding and rebuild numerical summaries and displays in a new directory:

```bash
python reproduction/reporting/soft_state_ablation/build_soft_state_assets.py \
  --study-root extensions/soft_state_ablation_20261102 \
  --output /tmp/soft_state_reporting
```

These commands use saved results; they do not fit models. The complete audit and analysis tables are required by the reporting source. Its outputs include CSV/JSON statistics and table and figure formats.

A complete independent numerical replay additionally needs the three numerical
archives: `Data_and_Forecasts.zip`, `Fitted_Models_00_15.zip`, and
`Fitted_Models_16_31.zip`, together with their member-hash index and the exact
frozen repository source. Extract all three into a separate complete checkout.
Use a new audit output path, for example:

```bash
python extensions/soft_state_ablation_20261102/audit/verify_soft_state.py --output /tmp/soft_state_audit_replayed.json
```

The numerical archives preserve original full records, all B/C paths, fitted
states, data and predictions. They were completed before this compact reporting
layer was added. Their snapshot index remains authoritative for their contents.
The archive packager performs no scientific recomputation and identifies any
retained ancillary files separately from scientifically sealed records.

## Verification record

[compact_release_validation.json](compact_release_validation.json) records
deterministic compression and byte-exact round trips, restoration in an isolated
empty tree, repeat acceptance of identical originals, rejection of conflicting
or corrupted inputs, documentary source checks, and preservation of the original
scientific source and numerical records. It hashes the final reporting files
and compact manifest. These checks validate the release transformation; the
scientific audit remains the separately identified full receipt.
