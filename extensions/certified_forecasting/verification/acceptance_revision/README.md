# Hierarchy and learned-representation revision

This verification accompanies the substantive revision after commit
`90e56f2fb7227f3a14f347b586676c62698f915a`.

## New mathematical and empirical content

- The scalar centered block hierarchy and all its proofs are in
  `paper/sections/block_witness.tex`. Independent reviews checked the pair
  identity, pathwise subset inequality, range bound, compact-support
  consistency, exact rare-conflict and occupancy formulas, weak duality,
  calibration, and conditional-oracle limitation.
- `experiments/block_hierarchy/verify.py` independently checks the complete
  new eight-law numerical study. Its report is retained in
  `results/block_hierarchy_v1/independent_verification.json`.
- `learned_decision_independent_audit.py` imports no project implementation
  module. It independently reconstructs the 40 learned KAHM banks, 80 KMeans
  fits, 240 local reference models, 240 validation candidates, 80 selected
  decoders, 320 endpoint cells, and 160 decisions from portable numeric JSON.
  The largest endpoint discrepancy is 1.11e-16.

## Software and publication checks

The standard test suite has 65 passing tests, including nine new mathematical
block-witness tests. The earlier witness suite adds seven passing tests.
Logs are retained here. The revised PDF keeps all current theorem proofs and
retains adverse empirical findings. Older repetitive experiment exposition is
consolidated in the main text, while complete earlier source and numerical
records remain available.

The learned-study provenance record preserves an initial source-archive
preflight failure and an explicit repair before calibration. All actual
execution sources and frozen models match the recorded hashes. The audit
checks consistency of the retained records; it is not an independent
timestamp attestation. The exact executed runner is retained despite later
archive-writing maintenance.

The main manuscript does not assert universal KAHM accuracy superiority,
natural-task success frequency, a larger-block benefit on the learned study,
or a new general concentration inequality. Its empirical scope separates
the block benchmark from the learned KAHM stratified-pair experiment.
