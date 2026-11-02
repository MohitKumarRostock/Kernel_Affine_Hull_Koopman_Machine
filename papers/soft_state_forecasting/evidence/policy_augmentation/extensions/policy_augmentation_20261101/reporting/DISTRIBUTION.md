# Research report and numerical evidence

The [research report](KAHM_Policy_Augmentation_Research_Report.pdf) summarizes
the completed synthetic and separate natural evaluations. The
[archive index](ARCHIVE_INDEX.json) identifies all 24 delivered files by exact
byte length and SHA-256: the report, one analysis ZIP, eighteen independently
readable records ZIPs, and four supporting files. The complete ZIP set contains
4,005 members and 4,338,048,776 compressed bytes. Every member was verified
against its source by size, SHA-256 and ZIP CRC.

Numerical archive binaries accompany the research-report delivery and are
available from the author; they are not stored in Git. The public repository
contains the scientific code, full analysis tables, seals, audit receipts,
reporting code and this exact distribution index. Download every records
part for a complete restoration. Parts contain complete files and should be
extracted individually, not concatenated.

## Restore and verify the computation

Use the [archive README](ARCHIVE_README.txt) for archive and member verification,
safe extraction, synthetic reanalysis, and natural reanalysis. The immutable
combined scientific source is commit
`b1915813e98a59572a4bbb800adc2bd1f3fccc1b`; the subsequent results release adds
reporting material. Source-publication chronology is recorded in
[SOURCE_REFERENCES.json](SOURCE_REFERENCES.json). A local freeze is not public
preregistration.

The external `KAHM_Policy_Augmentation_Archive_Manifest.json` gives the complete
member inventory. Its final hash is recorded in the index. The manifest's
explanatory source-provenance amendment leaves all archive bytes and member
identities unchanged, and records both the executed packager hash and the
later reporting-source hash.

## Rebuild the report

The combined `KAHM_Policy_Augmentation_Metrics.json` is a root-level member of
`KAHM_Policy_Augmentation_Analysis.zip` and is also delivered separately. A
source-only checkout does not contain this external metrics file. After
extracting the analysis archive at the repository root, use the reporting
code from the results release and run:

```bash
python extensions/policy_augmentation_20261101/reporting/build_report.py \
  --metrics KAHM_Policy_Augmentation_Metrics.json \
  --output KAHM_Policy_Augmentation_Research_Report_Rebuilt.pdf
```

The builder uses ReportLab, Matplotlib and the numerical dependencies and
requires both studies' passed audit records. It does not fit or select models.
Its delivered layout was rendered and visually checked on all nine pages;
the [QA receipt](report_qa.json) records the exact delivered PDF and metrics
hashes. Different rendering environments may change PDF bytes while retaining
the reported numerical content.

To regenerate the combined metrics after restoring all records, run the
collector with a new output name:

```bash
python extensions/policy_augmentation_20261101/reporting/collect_metrics.py \
  --output KAHM_Policy_Augmentation_Metrics_Recomputed.json
```

The collector validates both audit receipts and their analysis identities.
Preserve the original delivered files and their index while reproducing them.
