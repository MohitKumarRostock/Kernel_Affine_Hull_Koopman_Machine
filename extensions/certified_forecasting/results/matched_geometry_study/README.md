# Matched KAHM geometry and forecasting-head study

This prospective follow-up supplies the missing RBF and affine-hull Frobenius
and rollout controls. All three geometries receive the same head, history,
retained-size, and preprocessing choices. Validation selects the models;
test results do not determine any model choice.

The complete frozen schedule finished successfully: **124 jobs and 13,440
candidate fits**, with zero failures. It covers all seven original tasks,
five new fitting seeds (5701--5705), all four preprocessing modes, three
heads, and retained dimensions 2, 6, 14, and 30. The
[protocol](protocol.md) states the design and confidence allocation.

## What was learned

KAHM has lower mean MSE in 28 of the 42 task/head/alternative comparisons,
and higher mean MSE in 14. These are related descriptive comparisons,
not 42 independent replications.

On nonlinear AR, all six comparisons favor KAHM in all five repetitions.
On PenDigits, the reduction against RBF is 15.6--17.9% across the three
heads, with improvements in every repetition. The block-direct comparison
against affine geometry improves by 12.7% on average. The new affine heads
perform substantially better than KAHM on the oscillator tasks. Quadratic
and AR controls remain more accurate than KAHM on PenDigits.

Only one of 180 prespecified calibration comparisons certifies a positive
KAHM improvement: Van der Pol, seed 5703, rollout versus RBF, with MSE
improvement lower endpoint 0.00107100848. All other improvement statements
remain unresolved. The confidence event is joint within one synthetic
task/seed; it is not joint across all tasks or seeds. PenDigits comparisons
reuse the same held-out writers and remain descriptive.

See the [revision account](../../docs/MANUSCRIPT_REVISION.md) for the full
primary comparison table and its interpretation.

## Retained files

- `analysis/all_candidates.csv`: every candidate and horizon.
- `analysis/selected_candidates.csv` and `selected_pooled.csv`: exact
  validation-selected models, settings, and pooled outcomes.
- `analysis/taskwise_models.csv`: absolute MSEs, including ordinary controls.
- `analysis/taskwise_comparisons.csv` and `paired_effects.csv`: all primary,
  fixed-dimension, and head-selected effects, including regressions.
- `analysis/calibration_comparisons.csv`: all 180 primary comparisons.
- `analysis/pendigits_writer_differences.csv`: paired writer-level effects.
- `analysis/audit.json`: archive, score, certificate, grid, scaler, and control
  invariance checks and exact hashes of analyzed inputs.
- `records.tar.gz`: complete scalar results, frozen candidates, all writer
  losses, configurations, manifests, logs, and job accounting. The archive
  includes an index with a hash for every member.
- `source_snapshot.tar.gz`: exact frozen fitting/analysis source in the
  repository ancestry required by the KAHM imports, with the reference
  configurations and restoration instructions.
- `episode_archive_index.json`: hashes and sizes of all 120 full synthetic
  score archives. The large full archives are regenerated with the runner.
- `evidence_package.json`: identity and integrity checks of the separate
  compact evidence ZIP, retaining 270 selected synthetic models, 810 arrays,
  and 6,635,520 float64 values exactly as stored in the full source archives.

The full analysis checked 30,240 arrays containing 247,726,080 values and
131,040 PenDigits writer-loss values. A separate standard-library script
reconstructed all 360 primary choices and all 42 effects from the raw
records; identities matched exactly and numerical differences were below
4.3e-14. Development smoke runs are recorded only as implementation checks.

## Reproduction and independent checking

Run the commands in the extension's
[reproduction guide](../../docs/REPRODUCIBILITY.md). To independently check
the scalar primary analysis without rerunning model fits, extract the complete
record archive to a new directory and run, from the extension directory:

```bash
mkdir -p results/matched_records_for_check
tar -xzf results/matched_geometry_study/records.tar.gz \
  -C results/matched_records_for_check
python scripts/verify_matched_summary.py \
  --study results/matched_records_for_check \
  --analysis results/matched_geometry_study/analysis \
  --output results/matched_records_for_check/independent_recheck.json
```

This independent script imports neither the runner nor the main analyzer.
It verifies selection, pooling, writer means, and the primary comparison
table; the full per-episode audit additionally requires the full score files.
All original experiments and the earlier input-scaling study remain retained
separately with their original provenance.
