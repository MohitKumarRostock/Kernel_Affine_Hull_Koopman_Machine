# Numerical evidence and archive identities

The [archive index](ARCHIVE_INDEX.json) records the delivered file identities. The numerical ZIP set contains 4,005 members and 4,338,048,776 compressed bytes. Its verification record compares every member with its source by byte length, SHA-256, and ZIP CRC.

The numerical archive binaries are not stored in Git. The repository contains scientific code, analysis tables, source identities, audit records, and the distribution index. The index does not establish a public download service. Complete restoration requires all record parts; each is independently readable and must be extracted separately.

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

## Recompute combined metrics

To regenerate the combined metrics after restoring all records, run the
collector with a new output name:

```bash
python extensions/policy_augmentation_20261101/reporting/collect_metrics.py \
  --output KAHM_Policy_Augmentation_Metrics_Recomputed.json
```

The collector validates both audit receipts and their analysis identities.
Preserve the original delivered files and their index while reproducing them.
