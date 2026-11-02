# Raw scientific inputs

This directory contains the raw scientific inputs captured from source commit
`ac531aefc66066c3fc320d94c0b362fb88f94967`. The
[dataset card](DATASET_CARD.md) describes seven compressed family archives and
553 logical subdatasets in `canonical/`.

Run-level metric files and underlying scientific inputs serve different
purposes. The pre-campaign inventory found that state-transition and trajectory
arrays were not systematically persisted for every retained result. The
canonical capture supplies these inputs with explicit provenance; the earlier
metric files remain separately identifiable.

The capture uses `scripts/raw_data_capture.py`.
Each dataset consists of:

- a compressed `.npz` archive containing the actual numerical input arrays;
- a JSON sidecar recording dataset ID, role, source script, generation seed,
  generation parameters, array shapes and dtypes, Git commit, working-tree
  state, Python/NumPy/platform information, and the reference lockfile hash;
- an archive SHA-256 checksum;
- a canonical array-content SHA-256 checksum independent of ZIP-container
  metadata.

`canonical/manifest.json` enumerates the captured families and records their
verification. `canonical/dataset_index.tsv` identifies the logical subdatasets,
and `canonical/SHA256SUMS` supplies distributed-file checksums.

The pre-campaign evidence audit is stored under `reproduction/data_inventory/`.
