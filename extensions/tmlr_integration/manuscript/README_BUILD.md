# Building the KAHM forecasting documents

**Kernel Affine Hull Features for Time-Series Forecasting: Feature Spans,
Bounded Corrections, and Matched Evaluation**

Scientific source dated **2026-11-01**. The verified builds contain **28 pages
in the main paper and 72 pages in the Supplementary Studies**. The two entry points are:

- `main.tex`: the main paper, central proofs, and complete principal experiment.
- `supplement.tex`: the self-contained *Supplementary Studies*, including their
  methods, results, technical appendices, and references.

Both use the supplied anonymous TMLR review template. The template headers
indicate formatting mode; neither document has been submitted to a journal.

## Compile the supplied source

Install TeX Live with `latexmk`, pdfLaTeX, BibTeX, Latin Modern fonts, and the
packages requested by the two entry points. From this directory, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The outputs are `main.pdf` and `supplement.pdf`. All referenced tables, figures,
bibliography entries, and style files are supplied. Compilation requires no
network access, numerical archives, or forecasting-model fitting.
`TEMPLATE_PROVENANCE.json` records the original official style-file hashes;
those files and their license are retained unchanged.

## Principal experiment and generated displays

The principal experiment is the sealed study in
`extensions/dimension_forecasting`. It compares ten complete procedures on
192 task suites, with independent B selection of each geometric procedure's
cap, C calibration, and fresh D evaluation. Fixed-cap-two, native-cap-twenty,
and KAHM-selected-bank comparisons are secondary. The primary six-comparison
family establishes the nonlinear margins over common features and raw RBF
and linear-control noninferiority. It does not establish the complete joint
criterion. The paper reports every method and specified condition.

`generated/dimension_forecasting/` contains the principal manuscript displays.
Its dedicated builder reads the released analysis, verifies every input table
against the study's analysis summary, and records source/output hashes and
formatted numerical cells in `asset_manifest.json`. To regenerate these
assets from a repository checkout:

```bash
python build_dimension_assets.py --study-root /path/to/repository/extensions/dimension_forecasting
```

The builder needs Python, NumPy, and Matplotlib. When this source is inside the
repository, the study directory can also be detected automatically. It does
not fit models, choose settings, or read unreleased evaluation targets. The
legacy asset generators and `generated/reconstruction_mechanism/` belong to
the historical fixed-cap experiment and are not inputs to the present main
paper's principal tables. They cannot overwrite the new asset namespace.

The stage records identify the principal evidence:

| Record | SHA-256 |
|---|---|
| Source freeze | `1b80460fe2d33ede50bd085c026f7cdb6219e4f54057ff01d67c0e026fbcd60e` |
| Fitted-state lock | `6f8188e841bdf4b330f05f16912e428f74ad957cde3373ad9b80241d296d3e81` |
| Prediction lock | `074e7bd7186adb4f816daca3b46e2f8ab3bd8e7c331058aba20c0d8d10f29e7d` |
| Target release | `bfd871d11ff66c60847ba66766817071f3ba1eddb87d1c8b09bdd7e661c1c4f5` |

The study README gives separate commands for generating/fitting the study,
replaying saved predictions, reconstructing numerical analysis, and running
the independent audit. The manuscript revision performs none of the numerical
experiments again. Its tables summarize the completed and audited records.

## Numerical records and audit scope

The six separately supplied files
`KAHM_Forecasting_Dimension_Records_01.zip` through
`KAHM_Forecasting_Dimension_Records_06.zip` contain the complete numerical
records in disjoint repetition groups (0–3, 4–7, 8–11, 12–15, 16–19, 20–23).
The first also contains the global inventories and stage seals. Their member
paths restore under `extensions/dimension_forecasting`, and the accompanying
`KAHM_Forecasting_Dimension_Records_manifest.json` records all member and
archive checksums. They contain prepared arrays, selected and candidate
states, predictions, and execution records; code and analysis tables are
provided with the accompanying computational source.

The study retains 31,104 forecast arrays and 155,271,168 forecast values across
all candidate and reporting views. The ten calibrated primary procedures
account for 1,920 arrays and 9,584,640 values. All 192 tasks completed. The
independent audit regenerated episodes from the pinned generators and checked
seeds, noise, windows, target construction, B/C choices,
18,816 local bases, and 27 analysis tables containing 261,450 rows. These
checks establish computational consistency, not an independent scientific
replication or a general forecasting advantage.

## Supporting studies and historical provenance

The *Supplementary Studies* preserves its original complete comparison
families and numerical results. Its local-rank constraints are explicit
properties of those procedures, usually a cap of two with the specified
ambient-dimension restriction. They are not evaluations of the native default
or the dimension-selected principal procedure. Their different targets,
sampling units, and loss normalizations remain separate.

The public source commit
`7dda356138675934aab5e327ec667bde0a5eda79` identifies the historical
reconstruction-mechanism implementation and archived experiment. It does not
identify the current principal dimension experiment or these manuscripts.
The current extension and manuscript are supplied as accompanying source;
their local availability is not a claim of remote publication.

The original `KAHM_TMLR_Saved_Forecasts.zip` (80,805,506 bytes, SHA-256
`d149ae648b0b380eccf108d8d217566be173182a0ac047982afad4836fb92edd`)
remains a historical fixed-cap numeric supplement. Its original eight-table
verifier does not reconstruct the current dimension-selected primary study.
The standalone author source archive retains the original `revision_review/`
records with their historical scope. The
previous source inventory is preserved byte-for-byte at
`revision_review/prior_metadata/SOURCE_FILE_MANIFEST.json`. The root
`SOURCE_FILE_MANIFEST.json` inventories the current delivered manuscript source files;
`build_evidence/` contains its scientific, numerical-display, and visual
review receipts.

The editable source is an author package, not an anonymous review upload.
It preserves `LICENSE`, `DATA_LICENSE`, `LICENSE_TMLR.txt`, and identifying
provenance records. Anonymous PDF formatting does not anonymize those records.
