# Editable KAHM forecasting revision: 2 November 2026

## Joint-selection review checkpoint

The current working revision adds a **frozen joint input-scaling and
native-dimension follow-up protocol**. Its implementation, manufactured
checks, and staged independent audit are complete. Source publication and
production evaluation are pending. This checkpoint contains no new scientific
results from joint selection and does not assert that tuning improves held-out
performance.

The proposed source checkpoint is local commit
`b68cc9e96b3f5357390934c737e7648861dfb6ab`, based on public manuscript commit
`6c39a003e61ace058851ec6badca3e8dd13ef183`. Its intended branch is
`research/kahm-soft-state-joint-selection-2026-11-02`. That new branch has not
been published. The existing result and audit identifiers below describe the
completed predecessor study only.

The new main-paper section and supplementary protocol distinguish P-selected
shared geometry, individually selected procedures, and paired standard-only
references. The fixed design has three scalers, five native dimension caps,
32 fresh repetitions, 256 tasks, and one six-comparison primary family.
The complete protocol, implementation, analysis, independent audit, and
manufactured preflight are in
`extensions/soft_state_joint_selection_20261102/` in the companion repository
source checkpoint.

## Completed predecessor revision

The fixed four-arm study is complete: **32 independent repetitions, 256 task
suites, all four primary comparisons, and a passed independent computational
audit**. The main paper centers geometry-induced soft-state abstractions and
tests membership, local position and reconstruction as distinct observables.

In the P-selected shared-bank comparison, local position improves nonlinear
loss by 69.20% relative to memberships, and KAHM departures improve P by 10.46%.
Both establish their prescribed 3% margins. The KAHM-over-RBF gain is only
0.03%, leaving that margin unestablished. K has 1.11% higher linear loss than P
while meeting the 5% noninferiority criterion. Three of four criteria are
established. This is not an equivalence result, a general KAHM-superiority
result, or an autonomous-dynamics experiment.

Scientific source was published before production preparation at commit
`2b7d19b5b47f46b5b21b28d24960688f7969d96f`. Completed compact results are
identified by commit `65ecf679796cc38b2342e2a4c12fa0226dc9ac78`. The active repository branch is
`research/kahm-soft-state-ablation-2026-11-02`. Scientific source, result release
and manuscript edition have distinct identities.

## Build the supplied documents

From this directory, with TeX Live, pdfLaTeX, BibTeX and latexmk installed,
run these commands sequentially:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

All generated tables, figures, bibliography entries and styles are supplied.
Compilation requires no network, numerical archives or model fitting. Main
appendices contain principal proofs and the earlier decoder and dimension
protocol. The Supplementary Studies gives the full new protocol and descriptive
decompositions, followed by earlier evidence. Template files and licenses
remain unchanged. TMLR review formatting does not imply submission or acceptance.
This is an author working package with provenance and public repository links,
not a separately anonymized review archive.

Editable sources are tracked individually. A standalone source ZIP is supplied
as a document download rather than duplicated in Git.

## Interpretation

G contains transformed memberships; P adds weighted local principal coordinates;
K and R respectively add KAHM and positive-kernel RBF departures. Every arm
predicts the same observed target through a direct affine ridge head using only
its declared representation. Separate block transformations preserve exact G/P
prefixes. A fits; B selects; C supplies diagnostics only. P selects the primary
bank; independent per-arm cap selection is secondary.

The raw memberships lie in a product of simplices. Transformed or signed
enriched inputs need not. The conditional Koopman discussion concerns exact
raw observable-space invariance under specified evolution. No autonomous
coordinate propagation or rollout has been fitted or claimed.

K improves P on the two oscillator laws but worsens it on nonlinear
autoregression. Its aggregate nonlinear gain is concentrated at horizon twelve.
Independent cap selection leaves K slightly worse than R in point loss. These
secondary findings cannot replace primary comparisons. The four known laws,
selected geometry and affine-readout family constrain generalization.

## Evidence and regeneration

The repository study is under `extensions/soft_state_ablation_20261102/`.
Its `RESULTS.md`, analysis, source and phase seals, audit summary and compressed
full audit are published there. The largest tables and full audit are stored
losslessly compressed; `reporting/restore_compact_records.py` restores them
only after checking their original sizes and SHA256 hashes.

Full data, fitted states, candidate paths and predictions are retained in three
separately indexed numerical archives. Public source can regenerate them;
these large records are not copied into this paper folder. Compact paper-facing
evidence is under `evidence/soft_state_ablation/`.

`build_soft_state_assets.py` regenerates the new displays in a repository
checkout from the completed analysis and passed audit. It verifies input
hashes, counts and the four repetition-level bounds before writing a new output
directory. The supplied `generated/soft_state_ablation/asset_manifest.json`
binds the current displays to their inputs. Compilation does not require
rerunning this builder.

The independent audit passes 2,563,814 checks over all 256 tasks, 25,600 B
penalty-path arrays, 2,048 C arrays and 2,048 D arrays. It refits ridge paths
using Cholesky normal equations, independently reconstructs features and data
windows, and recomputes losses and comparison bounds. There are zero solver
recoveries or query-denominator fallback calls. It does not independently refit
partition k-means or reintegrate physical laws. Ordered source publication and
seals document procedure, not cryptographic target isolation or external
scientific replication.

## Earlier evidence and verification

- Dimension selection: three of six primary criteria established.
- Preprocessing: one of four; no general tanh benefit.
- Residual targets: two of four, through a same-cohort computational reproduction
  with prior outcome knowledge and no additional independent repetitions.
- Fresh policy augmentation: two of four; both linear criteria and neither
  prescribed nonlinear superiority criterion established.
- Natural panel: 48 M4 training series, descriptive; augmentation has 2.27%
  higher aggregate loss than the base policy, not an official M4 Test result.

All 106 historical generated display files and 48 prior compact evidence files
remain byte-for-byte unchanged. Earlier protocols, comparison families and
receipts retain their original scope. Preproduction metadata is archived under
`revision_review/prior_metadata/`; it does not describe the completed edition.

Current compilation, text checks, preservation checks, clean extracted-source
build and visual-review receipts are under
`build_evidence/ablation_completed_20261102/`. `REVISION_STATUS.json` records
completed scientific and editorial scope; `SOURCE_FILE_MANIFEST.json` inventories
the delivered editable files and hashes. Historical receipts are retained under
dated directories and do not substitute for current verification.
