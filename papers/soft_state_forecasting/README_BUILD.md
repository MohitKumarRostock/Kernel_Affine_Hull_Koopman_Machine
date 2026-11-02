# Editable KAHM forecasting revision: 2 November 2026

## Completed joint scaling and dimension study

This edition integrates the completed, separately frozen experiment on joint
input-scaling and native local-dimension selection: **32 fresh independent
repetitions, 256 tasks, six primary comparisons, and a passed independent
computational audit with 7,422,883 checks**. Three input transformations and
five native caps supply fifteen geometry candidates per task. All transforms
and coefficients are fitted on A; B selects; C is diagnostic; D is terminal.
There is no A+B refit.

The shared-bank comparison selects a common geometry with P. On the nonlinear
population, P improves G by 68.69% and K improves P by 8.86%; both establish
their prescribed 3% superiority margins. K has 0.9186% higher loss than the
matched RBF arm, leaving the K/R margin unestablished. Its linear loss is
1.26% higher than P's while meeting the 5% noninferiority criterion.

The direct tuning comparison independently selects K among all fifteen
candidates and compares it with independently dimension-selected,
standard-only K on the **same fresh cohort**. The nonlinear point gain is
only **0.0727%**, and the 3% improvement criterion is not established. Joint
K has 0.2491% higher linear loss while meeting the 5% noninferiority criterion.
Thus four of six primary criteria are established. These findings do not
establish equivalence, a general KAHM-over-RBF advantage, global optimization
of all geometry parameters, or useful autonomous coordinate dynamics.

Scientific source was published before production preparation at commit
`8e0250279074aefa39079fa33205d3c74f835848`. Completed compact results are
identified by commit `06df3ed4e43c6f58d0f887a1459ee9613b6a9760`. The repository branch is
`research/kahm-soft-state-joint-selection-2026-11-02`. The actual public
source commit differs from the reviewed local checkpoint because browser
publication used directory batches; the nineteen extension files and complete
repository root tree were verified exactly. `PUBLICATION.json` retains both identities.
Scientific source, completed results, and the manuscript edition are distinct.

## Build the supplied documents

From this directory, with TeX Live, pdfLaTeX, BibTeX and latexmk installed,
run these commands sequentially:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

All generated tables, figures, bibliography entries and styles are supplied.
Compilation requires no network, numerical archives, or model fitting. Main
appendices retain the principal proofs and the earlier decoder and dimension
protocol. The Supplementary Studies provides full representation-study
protocols and descriptive decompositions, followed by the earlier evidence.
Template files and licenses remain unchanged. TMLR review formatting does
not imply submission or acceptance. This is an author working package with
provenance and public repository links, not a separately anonymized archive.

Editable sources are tracked individually. The standalone source ZIP retains
the root directory `KAHM_Reconstruction_Prediction/` and is supplied separately
as a document download rather than duplicated in Git. It includes both PDFs,
current source and display manifests, and dated verification records.

## Interpretation and comparison families

G contains transformed memberships; P adds weighted local principal
coordinates; K and R respectively add KAHM and positive-kernel RBF departures.
Each arm predicts the same observed response through a direct affine ridge
head using only its declared representation. Separate block transformations
preserve the exact G/P prefixes within each candidate geometry.

The preceding fixed-standardization experiment selects P's cap for its four
primary comparisons; independent per-arm cap selection is secondary there.
The new joint-selection experiment has a separate six-comparison family:
four P-selected shared-bank comparisons and two primary comparisons of
independently selected K against the same-cohort standard-only procedure.
Independent K/R and law, noise, horizon, and selection-frequency breakdowns
remain descriptive. Repetition aggregates, not individual origins or task
cells, are the units of inference. The cohorts and confidence families are
not pooled.

The raw memberships lie in a product of simplices. Transformed or signed
enriched inputs need not. The conditional Koopman discussion concerns exact
raw observable-space invariance under specified evolution. No autonomous
coordinate propagation or rollout has been fitted or claimed. The known laws,
finite candidate grids, allocation, and affine-readout family constrain the
scope of the forecasting evidence.

## Joint-study evidence and regeneration

The repository study is under
`extensions/soft_state_joint_selection_20261102/`. Its `RESULTS.md`, complete
analysis, source-publication and phase receipts, audit summary, and compressed
full audit are published there. The largest analysis tables and full audit
are losslessly compressed; `reporting/restore_compact_records.py` verifies
original sizes and SHA256 hashes before restoring them.

Five indexed numerical ZIPs retain all underlying data and forecasts plus
four shards of portable fitted states and candidate paths. Every archive
member is checked by SHA256 and CRC after packaging. The public frozen source
can regenerate these records; the large numerical files are not copied into
this paper folder. Compact paper-facing evidence is under
`evidence/joint_selection/`.

`build_joint_selection_assets.py` regenerates the new displays from the
completed analysis and its passed audit in a repository checkout. It checks
exact table hashes and row counts, independently recomputes all six
primary bounds, and verifies full replay coverage before writing an output
directory. `generated/joint_selection/asset_manifest.json` binds the displays
to their evidence. Compilation uses the supplied assets and does not rerun
this reporting builder.

The production audit independently reconstructs all 76,800 B penalty-path
forecast arrays and 4,096 selected arrays on each of C and D. Their maximum
absolute prediction discrepancies are 1.13e-10, 2.42e-12, and 6.28e-12,
respectively. It checks A-fitted scalers, candidate geometries, native
dimensions, selections, standard-only comparators, data histories, phase
ordering, exported statistics, and all primary decisions. There are zero
numerical solver recoveries. The audit uses saved fitted geometry and
physical episodes; it does not independently refit partition k-means or
reintegrate the physical laws. Procedural seals do not provide cryptographic
isolation, external oversight, or independent scientific replication.

## Preceding fixed-standardization study

The completed predecessor contains 32 independent repetitions and 256 tasks,
with three of four primary criteria established. P improves nonlinear G
loss by 69.20%, and K improves P by 10.46%, both exceeding their 3% margins.
The K/R gain is only 0.03%, leaving that margin unestablished. K has 1.11%
higher linear loss than P while meeting the 5% noninferiority criterion.

Its scientific source is commit
`2b7d19b5b47f46b5b21b28d24960688f7969d96f`; its completed results are commit
`65ecf679796cc38b2342e2a4c12fa0226dc9ac78`. Records remain under
`extensions/soft_state_ablation_20261102/`. Its independent audit passed
2,563,814 checks over 25,600 B path arrays and 2,048 selected arrays on each
of C and D. Its three indexed numerical archives, compact paper evidence,
display assets, and four-comparison family retain their original identities.

## Earlier evidence and current document verification

- Dimension selection: three of six primary criteria established.
- Preprocessing: one of four; no general tanh benefit.
- Residual targets: two of four, through a same-cohort computational reproduction
  with prior outcome knowledge and no added independent repetitions.
- Fresh policy augmentation: two of four; both linear criteria and neither
  prescribed nonlinear superiority criterion established.
- Natural panel: 48 M4 training series, descriptive; augmentation has 2.27%
  higher aggregate loss than the base policy, not an official M4 Test result.

All **121 generated display files and 76 compact evidence files** in the
completed predecessor manuscript remain byte-for-byte unchanged. Earlier
protocols, comparison families and receipts retain their original scope.
Prior and preproduction metadata are archived under
`revision_review/prior_metadata/`; their dated pending statuses do not describe
this completed edition.

Current compilation, text checks, preservation checks, a clean extracted-source
build, and visual-review receipts are under
`build_evidence/joint_selection_completed_20261102/`. `REVISION_STATUS.json`
records the completed scientific and editorial scope.
`SOURCE_FILE_MANIFEST.json` inventories the delivered files and hashes.
Historical receipts are retained under dated directories and do not
substitute for current document verification.
