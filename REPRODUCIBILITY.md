# Reproducing the forecasting study

Reproduction has several distinct levels. Compiling a document checks neither its data nor its models. Recomputing a summary from saved losses checks the analysis conditional on those losses. Replaying fitted states requires the original arrays. Regenerating the simulation additionally requires the complete scientific dependency chain.

## Source identities

The central experiment is `extensions/soft_state_ablation_20261102/`. Its preproduction scientific source is commit `2b7d19b5b47f46b5b21b28d24960688f7969d96f`. Completed results were published at `65ecf679796cc38b2342e2a4c12fa0226dc9ac78`; the subsequent comprehensive manuscript edition is at `6c39a003e61ace058851ec6badca3e8dd13ef183`. These identities refer to different objects and must not be substituted for one another.

The original freeze identifies inherited generators and native geometry in `extensions/dimension_forecasting/` and the checked numerical solver in `extensions/policy_augmentation_20261101/code/`. The root `requirements.txt` belongs to an earlier architecture and is not the central study environment. Install the versions in the central study's `requirements.txt` in a dedicated environment. The recorded environment uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and threadpoolctl 3.6.0.

## Document compilation and saved-loss arithmetic

From `papers/local_position_forecasting/`:

```bash
python scripts/build.py
python scripts/verify_primary.py --output primary_check.json
```

The first command builds `main.pdf` and `supplement.pdf` using the supplied tables. The second requires NumPy and SciPy and checks 128 paired repetition differences, four adjusted upper endpoints, and 24 panel summaries against the compact records. Neither command refits a forecasting model. A successful saved-loss check is not a new run of the full audit described below.

## Compact-record restoration

The public compact release stores B candidate scores, basis-dimension records, and the full audit receipt as lossless gzip files. From the repository root:

```bash
python extensions/soft_state_ablation_20261102/reporting/restore_compact_records.py --check-only
python extensions/soft_state_ablation_20261102/reporting/restore_compact_records.py
```

The manifest binds compressed and uncompressed sizes and SHA256 identities. An existing identical file is accepted; a conflicting original is rejected. Restoration performs no forecasting computation.

## Full simulation regeneration

Start in a complete checkout containing the published results and genuine publication receipt. Create a separate worktree at the preproduction source; choose a path that does not already exist:

```bash
git fetch origin research/kahm-soft-state-ablation-2026-11-02
git worktree add --detach ../kahm_soft_state_replay 2b7d19b5b47f46b5b21b28d24960688f7969d96f
cp extensions/soft_state_ablation_20261102/PUBLICATION.json ../kahm_soft_state_replay/extensions/soft_state_ablation_20261102/PUBLICATION.json
cd ../kahm_soft_state_replay
```

After activating the dedicated study environment, execute the following stages in order:

```bash
python extensions/soft_state_ablation_20261102/code/pipeline.py abc
python extensions/soft_state_ablation_20261102/code/pipeline.py fit --workers 3
python extensions/soft_state_ablation_20261102/code/pipeline.py seal-fits
python extensions/soft_state_ablation_20261102/code/pipeline.py d
python extensions/soft_state_ablation_20261102/code/pipeline.py predict --workers 3
python extensions/soft_state_ablation_20261102/code/pipeline.py seal-predictions
python extensions/soft_state_ablation_20261102/code/pipeline.py release
python extensions/soft_state_ablation_20261102/analysis_code/analyze_soft_state.py
python extensions/soft_state_ablation_20261102/audit/verify_soft_state.py
```

The source-only worktree contains no completed production seals. Do not run generation in the preserved completed-results tree, delete selected seals to bypass ordering, or create a replacement freeze or publication receipt. Each production worker uses one numerical thread. The implementation checks occupied outputs rather than silently overwriting conflicting evidence. Compare substantive arrays and tolerance-based numerical results across reruns; timestamps and archive bytes can differ.

The stage records establish execution order but not cryptographic isolation of D targets. Earlier research informed the design. The source freeze is not a claim of public preregistration or external oversight.

## Replay from retained numerical records

Audit replay requires prepared episodes, fitted JSON/NPZ states, predictions, and execution records from all three separately indexed numerical archives. Their inventory is `extensions/soft_state_ablation_20261102/reporting/numerical_archive_summary.json`. No public archive download URL is asserted by that inventory. Full source regeneration is available independently of an author-delivered archive.

With the complete arrays restored in a separate checkout, write a new audit receipt rather than replacing the released one:

```bash
python extensions/soft_state_ablation_20261102/audit/verify_soft_state.py --output /tmp/soft_state_audit_replayed.json
```

The retained audit reconstructs features, refits B penalty paths using Cholesky calculations rather than production SVD, replays C/D forecasts, and recomputes aggregation and inference. It reports 2,563,814 checks, 25,600 B penalty-path arrays, and 2,048 arrays for each of C and D. Maximum absolute D forecast discrepancy is 5.59e-12 under relative/absolute tolerances 2e-7/2e-9. It does not independently reimplement the physical-law integrators or partition clustering, and is not an independent scientific replication by another group.

## Related studies

Each study in [docs/RESEARCH_INDEX.md](docs/RESEARCH_INDEX.md) retains its own requirements, source identity, sampling units, and inference family. Reproducing the same simulation units does not create new independent observations. The natural-data policy analysis is descriptive and uses M4 training data only. Its unfavorable result is part of the retained evidence, not a failed run to exclude.
