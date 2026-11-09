# Raw scientific inputs

The `canonical/` directory contains the trajectory, state-transition and training-noise arrays used by the reported empirical results. The [dataset card](DATASET_CARD.md) describes the seven archive families and their 553 logical datasets. Per-run metric CSV files are analysis outputs; they do not replace these input arrays.

The data-capture utility is `scripts/raw_data_capture.py`. Each captured dataset has:

- a compressed `.npz` archive containing the numerical arrays;
- a JSON sidecar recording the dataset ID, role, generation parameters, random seed, array shapes and dtypes, source commit, working-tree state and software environment;
- an archive SHA-256 checksum;
- an array-content SHA-256 checksum independent of ZIP-container metadata.

`canonical/manifest.json` lists the distributed archives and their metadata. `canonical/dataset_index.tsv` provides a dataset index, and `canonical/SHA256SUMS` provides file checksums. The input-coverage audit is under `reproduction/data_inventory/`.
