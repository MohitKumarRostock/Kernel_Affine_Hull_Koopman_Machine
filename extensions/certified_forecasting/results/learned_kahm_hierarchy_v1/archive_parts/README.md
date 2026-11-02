# Exact byte parts of the frozen evidence payloads

These eight files preserve the original `samples.jsonl.gz` and
`qp_certificates.jsonl.gz` byte streams in parts of at most 8 MiB. The split is
archive packaging only: neither gzip stream was decompressed or recompressed,
and no observation, optimization certificate, or confidence endpoint changed.

From `extensions/certified_forecasting`, restore both original paths before
running the independent audit or rebuilding result assets:

```bash
python scripts/restore_hierarchy_payloads.py
```

To verify restoration in a separate directory:

```bash
python scripts/restore_hierarchy_payloads.py --output-dir /tmp/kahm_hierarchy_payloads
```

`manifest.json` lists the ordered parts, byte counts, SHA-256 hashes, and Git
blob SHA-1 identifiers for each part and each complete original. The script
checks the entire batch before installing any file, installs each complete
file atomically, leaves existing matching files intact, and refuses to
overwrite an existing file whose bytes differ. The original frozen result
inventory and its hashes continue to refer to the reconstructed originals.

`restoration_verification.json` records the successful byte-for-byte round
trip, unchanged-file check, corrupt-part rejection, temporary-file cleanup,
and conflicting-destination rejection. It is a packaging check, separate from
the study's independent numerical and provenance audits.
