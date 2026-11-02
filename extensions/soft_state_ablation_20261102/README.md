# Predictive utility of geometry-induced soft-state abstractions

This study separates three contributions to a geometric forecasting
representation: soft memberships, position within their reference regions,
and nonlinear reconstruction departures. Every arm predicts the same future
observation with a direct affine ridge head. The scientific design is in
[PROTOCOL.md](PROTOCOL.md); the exact settings are in [config.json](config.json).

| Arm | Readout input |
| --- | --- |
| G | Separately transformed membership gates |
| P | G plus separately transformed gate-weighted local principal coordinates |
| K | P plus separately transformed KAHM departures |
| R | P plus separately transformed positive-kernel RBF departures |

The raw gates form a product of simplices. The transformed inputs and signed
enrichments need not be simplex-valued. KAHM and RBF share the memberships and
reference geometry within each matched bank. No arm retains a raw-history
forecast, residual target, or calibrated correction multiplier.

The fixed production cohort has 32 independent repetitions, four inherited
laws and two added-noise levels, yielding 256 tasks. History width is 12 and
horizons are 1, 4 and 12. A fits geometry and coefficients; B chooses penalties
and dimensions; C supplies diagnostics only; D supplies final evaluation.
The primary view uses P's B-selected cap for all arms. Independently selected
caps are secondary. There is one four-contrast family; neither diagnostic
subsets nor the secondary view can replace an unsuccessful primary result.

## Status and evidence

**Completed:** all 256 tasks, the fixed four-contrast analysis, and the full
independent computational audit completed on 2 November 2026. Three of four
primary criteria were established. In the nonlinear population, P improved G
by 69.20% and K improved P by 10.46%, establishing their prescribed 3% margins.
K improved R by only 0.03% in point loss and did not establish that margin.
K had 1.11% higher linear loss than P while meeting the 5% noninferiority
criterion. These results support useful enrichment under the declared
representation and selection procedure; they do not establish a general
KAHM advantage over matched RBF enrichment.

Read [RESULTS.md](RESULTS.md) for every primary comparison, condition and
horizon heterogeneity, selection diagnostics, costs, and interpretation.
The [analysis summary](analysis/analysis_summary.json) retains exact numerical
values and table hashes. The
[independent audit summary](verification/independent_audit_summary.json)
retains all full-receipt top-level information except individual task records
and identifies the lossless compressed full audit.

The scientific source was published and verified before production preparation
at commit
[`2b7d19b5b47f46b5b21b28d24960688f7969d96f`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2b7d19b5b47f46b5b21b28d24960688f7969d96f).
The containing results-release commit has a separate identity; it does not
replace this prospective scientific source identity. An earlier rejected
publication attempt remains a dated historical record.

A source freeze records the implemented procedure;
`PUBLICATION.json` binds that freeze to its verified GitHub commit before
production preparation. Stage seals document ordering, not access control or
external scientific replication.

The study was designed with knowledge of earlier results. Its fresh seeds add
independent repetitions under known laws, not new application domains. No
earlier observations are pooled. See
[analysis_code/HISTORICAL_PRECISION.md](analysis_code/HISTORICAL_PRECISION.md)
for the budget rationale and its limitations.

## Read, compile, or regenerate released evidence

The supplied paper tables and figures are ready to compile. From
`papers/soft_state_forecasting/`, run `latexmk -pdf -interaction=nonstopmode
-halt-on-error main.tex` and the same command for `supplement.tex`. The
[paper build instructions](../../papers/soft_state_forecasting/README_BUILD.md)
give the document requirements. Compilation needs no numerical archive,
model fitting, or restoration of compressed records.

The public compact release stores exactly three large records as lossless
gzip files: `analysis/B_candidate_scores.csv.gz`,
`analysis/basis_dimensions.csv.gz`, and
`verification/independent_audit.json.gz`. Other analysis tables and the analysis
summary remain plain files. The original bytes are retained locally and in the
separate numerical archives. Their uncompressed sizes and SHA256 identities
are unchanged; the analyzer's original table hashes still apply.

To restore the three original records in a compact checkout, run from the
repository root:

```bash
python extensions/soft_state_ablation_20261102/reporting/restore_compact_records.py
```

The utility checks compressed and uncompressed sizes and SHA256 hashes, permits
an existing identical original, and rejects a conflicting file. It performs
no scientific computation. Add `--check-only` to verify without restoring.
The [compact release manifest](reporting/compact_release_manifest.json) binds
the exact bytes, and [reporting instructions](reporting/README.md) explain
restoration and evidence regeneration.

After restoration, the reporting builder can regenerate the supplied display
assets into a new directory; the large numerical archives are unnecessary
for this step:

```bash
python papers/soft_state_forecasting/build_soft_state_assets.py --output /tmp/soft_state_assets_rebuilt
```

Replaying the independent computational audit additionally requires the
retained prepared data, fitted states, forecasts, and execution records from
all three separately indexed numerical archives. Extract those into a separate
complete repository checkout with the exact frozen scientific source. Keep the
released analysis and audit receipt, and choose a new output path:

```bash
python extensions/soft_state_ablation_20261102/audit/verify_soft_state.py --output /tmp/soft_state_audit_replayed.json
```

Existing output paths are not overwritten. The replay receipt gets a new
timestamp; the released receipt remains the reference artifact.

## Reproduce the simulation from published source

Use a complete repository checkout containing the exact inherited sources
recorded in `freeze.json`. Install the versions in
[requirements.txt](requirements.txt) in a dedicated environment. Production
workers each use one numerical thread; the prescribed run uses three workers.
The checked inherited ridge solver records any NumPy-SVD to SciPy-gesvd
recovery, without changing the ridge objective or silently excluding tasks.

Start from a checkout of the completed results branch and create a separate,
source-only worktree at the exact preproduction commit. Choose a worktree path
that does not already exist. Copy the genuine publication receipt from the
completed checkout; it identifies the same source and remains valid evidence
that publication preceded the new generation time:

```bash
git fetch origin research/kahm-soft-state-ablation-2026-11-02
git worktree add --detach ../kahm_soft_state_replay 2b7d19b5b47f46b5b21b28d24960688f7969d96f
cp extensions/soft_state_ablation_20261102/PUBLICATION.json ../kahm_soft_state_replay/extensions/soft_state_ablation_20261102/PUBLICATION.json
cd ../kahm_soft_state_replay
```

This source commit contains the original freeze but no production outputs or
stage seals. A results-commit checkout already contains completed seals, so
removing only its analysis or verification directory is insufficient. Do not
create a replacement freeze or invent a publication receipt. From the new
worktree's repository root, run the stages sequentially:

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

The run is append-only: an occupied output is checked or rejected. A separate
simulation reproduction needs a clean output tree in its own checkout,
containing the exact scientific source, original freeze and genuine publication
receipt. Do not run these generation commands in the preserved completed data
tree. Preserve the released outputs separately for comparison. Execution
timestamps and byte-level archive hashes can change across runs even where
numerical values agree; compare substantive arrays and reported tolerances.

The production order is enforced: D generation requires the complete fit
seal, and terminal analysis requires the full prediction seal and release.
Prediction workers receive query arrays only. D targets are physically present
after D preparation; these procedural checks are not cryptographic isolation.

## Retained records and audit scope

`prepared/` holds episode, noise, row and target records; `results/` contains
portable JSON/NPZ fitted states, all B candidate predictions, C diagnostics,
and numerical/resource logs. `predictions/` contains the eight logical D
forecast arrays per task and query diagnostics. Large numerical records are
kept in the separately indexed data archive rather than copied into Git.
The [numerical archive summary](reporting/numerical_archive_summary.json)
identifies the three author-deliverable archives and their compressed full
member index; it does not assert a public download URL.
Compact analysis tables, source identities, a full audit summary and its
lossless compressed complete receipt remain with this study. Public source is
sufficient to regenerate the cohort. Compression, restoration, and the numerical
archive packager are reporting utilities outside the original scientific freeze.

The independent audit reconstructs histories and targets from saved base
episodes and noise, reconstructs the four feature maps from portable state,
checks native rank rules and exact shared-block identities, and refits every
B penalty path using a Cholesky calculation distinct from the production SVD.
It replays selected C and D forecasts and independently recomputes losses,
repetition aggregates and all four adjusted upper bounds. The passed receipt
records 2,563,814 checks, 25,600 B penalty paths, 2,048 C forecasts and 2,048 D
forecasts. Maximum absolute D replay discrepancy was 5.59e-12 under frozen
relative/absolute tolerances 2e-7/2e-9. There were no solver recoveries. Partition
k-means and physical-law integration were not independently reimplemented.
This computational audit is not an independent replication by another group.

## Connection to the manuscript

The current paper is in [../../papers/soft_state_forecasting/](../../papers/soft_state_forecasting/).
This experiment evaluates external-target predictive usefulness of the
representation hierarchy. It does not fit autonomous soft-coordinate
propagation or establish Markov sufficiency, Koopman invariance, or universal
forecasting superiority. Earlier dimension, residual and policy studies keep
their original procedures and comparison families; this study does not
relabel their results.
