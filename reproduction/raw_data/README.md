# Raw scientific inputs

This directory is reserved for raw scientific input datasets captured by the
final frozen reproduction campaign.

The repository already contains extensive unaggregated run-level result files,
but the pre-campaign inventory shows that underlying state-transition or
trajectory arrays were not systematically persisted for every retained result.
Those metric-level files are therefore not relabeled as raw scientific data.

Final-campaign raw datasets are written with `scripts/raw_data_capture.py`.
Each dataset consists of:

- a compressed `.npz` archive containing the actual numerical input arrays;
- a JSON sidecar recording dataset ID, role, source script, generation seed,
  generation parameters, array shapes and dtypes, Git commit, working-tree
  state, Python/NumPy/platform information, and the reference lockfile hash;
- an archive SHA-256 checksum;
- a canonical array-content SHA-256 checksum independent of ZIP-container
  metadata.

A later campaign manifest will enumerate every captured dataset used by the
retained reported results and will verify all sidecars and checksums.

The pre-campaign evidence audit is stored under `reproduction/data_inventory/`.
