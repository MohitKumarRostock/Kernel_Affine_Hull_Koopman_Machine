# Editable KAHM forecasting revision: 2 November 2026

## Status

**Preproduction revision. The new four-arm study has not been executed.**
Its source and analysis are frozen and checked on artificial data. GitHub source
publication was blocked by a rejected write. The protocol requires verified
publication before generating production data. No new forecasting performance,
primary decision, or completed production audit is claimed.

The main paper centers geometry-induced soft memberships, within-region
position and reconstruction enrichment. It specifies the missing direct G/P/K/R
comparison and condenses the completed studies without changing their outcomes.
The supplement gives the prospective protocol and preserves full prior evidence.
Both PDFs prominently identify this edition's preproduction status.

## Build

From this directory, with TeX Live, pdfLaTeX, BibTeX and latexmk installed, run
these commands sequentially:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

All tables, figures, bibliography entries and styles are supplied. Compilation
requires no network, numerical archives or model fitting. The TMLR review style
is formatting only; no submission or acceptance is implied. Template files and
licenses remain unchanged. Current verification records are under
`build_evidence/ablation_preproduction_20261102/`. Earlier receipts retain their
historical scope; the preceding build guide is archived under
`revision_review/prior_metadata/softstate_20261102_README_BUILD.md`.

## New design and files

`core_soft_state_study.tex` specifies the main design;
`supporting/section_soft_state_ablation.tex` gives details. Compact frozen design
and artificial-validation evidence are in
`evidence/soft_state_ablation_preproduction/`. `REVISION_STATUS.json` records the
publication block and absence of production execution.

G contains transformed memberships; P adds weighted local principal coordinates;
K and R respectively add KAHM or positive-kernel RBF departures to P. Every arm
predicts the same observed target through its own direct affine ridge head.
There is no retained raw-history forecast, residual target or calibration.
Separate block transformations preserve exact G/P prefixes. A fits; B selects;
C supplies diagnostics only. P selects the common primary bank; independent
per-arm cap selection is secondary.

The fixed design has 32 repetitions and 256 tasks. One primary family tests
nonlinear P/G, K/P and K/R at a 3% superiority margin, and linear K/P at a 5%
noninferiority margin. One-sided Bonferroni paired-t bounds use repetition
aggregates. Neither outcomes nor interim inspection may change the sample size,
candidate menu or reporting family. Fresh repetitions do not add new domains.
The prepared scientific code is in the local repository under
`extensions/soft_state_ablation_20261102/`.

The raw membership map takes values in a product of simplices; transformed or
enriched inputs need not. The conditional Koopman discussion concerns exact raw
observable-space invariance. No autonomous coordinate propagation or rollout
study has been executed or claimed.

## Completed evidence preserved

- Dimension selection: three of six primary criteria established.
- Preprocessing: one of four; no general tanh benefit.
- Residual targets: two of four, from a same-cohort computational reproduction
  with prior outcome knowledge and no new independent repetitions.
- Fresh policy augmentation: two of four; both linear criteria and neither
  prescribed nonlinear superiority criterion established.
- Natural panel: 48 M4 training series, descriptive; augmentation has 2.27% higher
  aggregate loss than the base policy. This is not an official M4 Test result.

All 106 historical generated display files and 48 prior compact evidence files
remain byte-for-byte unchanged. Original reporting builders and generated assets
retain their study-specific scope. The compact main presentation does not
promote secondary results or combine separate comparison families.

## Validation and outstanding work

Artificial checks cover native full and zero ranks, cap aliases, exact shared
blocks, independent ridge solutions, portable state, checked numerical recovery
and stage-order rejection. The independent artificial audit checks eight tasks
with 14,130 checks and maximum selected-D prediction difference about 9.9e-13.
A separate 32-repetition artificial-loss fixture passes 17 statistical checks.
These findings validate software behavior, not production predictive utility.

The remaining sequence is source publication and verification, a genuine
publication receipt, every fixed production task, ordered fit/prediction seals,
target release, analysis, independent audit, integration of all results, and
final document review. The preproduction draft is not a completed submission.
