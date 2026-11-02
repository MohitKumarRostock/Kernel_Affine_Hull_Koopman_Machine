# Building the KAHM forecasting documents

**Geometry-Induced Soft-State Abstractions for Forecasting:
Predictive Enrichment with Kernel Affine Hull Machines**

Soft-state manuscript revision dated **2026-11-02**. This edition centers
the representation question: what a geometric soft membership retains,
which observables KAHM adds, and how their external predictive utility
relates to conditional dynamical closure. Current build, preservation,
and visual-review receipts are in `build_evidence/softstate_20261102/`.
The preceding `policy_augmentation_20261101` receipts describe the
36-page main paper and 86-page supplement supplied for this revision;
the earlier `coordinated_20261101` records retain their historical scope.
The two entry points are:

- `main.tex`: the main paper, central proofs, complete principal experiment,
  and the preprocessing and residual-target follow-up synthesis in
  `core_followup_experiments.tex`, followed by the fresh policy-augmentation
  study in `core_policy_augmentation.tex`.
- `supplement.tex`: the self-contained *Supplementary Studies*, including
  historical and follow-up methods, results, technical appendices and references.

Both use the supplied anonymous TMLR review template. Its headers indicate
formatting mode, not submission or acceptance. This edition reports completed
studies. This manuscript revision adds a conditional observable-space
proposition and performs document and preservation checks; it does not fit
new models, rerun the original forecasting campaigns, or generate additional
evaluation observations. All 106 files under `generated/` are retained
byte-for-byte from the supplied edition.

## Conceptual changes in this edition

The shared locality gates are explicitly identified as a soft membership
map in a product of simplices, normalized within each partition. Common
features retain within-state principal coordinates; KAHM and matched RBF
departures enrich those observables. Their raw vectors are denoted by
`psi_K` and `psi_R`; bounded decoder inputs are denoted by `chi_K` and
`chi_R`. This notation change does not alter a fitted estimator.

The new `core_state_dynamics.tex`, included at the end of `core_theory.tex`,
proves the equivalence between invariant augmented raw observable spaces
and exact finite-dimensional propagation under a specified deterministic
history evolution. Equal raw spans share exact invariance status; distinct
spans do not establish closure or better forecasting. Existing experiments
fit direct observed-target forecasts, not autonomous coordinate rollouts.
The companion reconstruction-score affinity map is explicitly distinguished
from the shared center-distance gates evaluated here.

The abstract, introduction, related work, methods, discussion and supporting
scope adopt this framework. Empirical comparison families, mathematical
estimator definitions, protocols, generated displays, numerical evidence,
reporting builders and research-transparency records are preserved.

## Compile the supplied source

Install TeX Live with `latexmk`, pdfLaTeX, BibTeX, Latin Modern fonts and the
packages requested by the two entry points. From this directory, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The outputs are `main.pdf` and `supplement.pdf`. All referenced tables, figures,
bibliography entries and style files are supplied. Compilation of the source
ZIP requires no network access, numerical archives, or forecasting-model fitting.
`TEMPLATE_PROVENANCE.json` records the original official style-file hashes;
those files and their license are retained unchanged.

## Study identities and generated displays

| Study source under `extensions/` | Display namespace under `generated/` | Inferential scope |
| --- | --- | --- |
| `dimension_forecasting/` | `dimension_forecasting/` | Principal six-comparison family; three criteria established |
| `preprocessing_dimension/` | `preprocessing_dimension/` | Separate four-comparison family; one criterion established; fixed-common residual comparisons are secondary |
| `crossfit_residual_reproduction_20261101/` | `crossfit_residual/` | Original four-comparison family; two criteria established; 24 original cross-fit repetition units and zero additional independent repetitions |
| `policy_augmentation_20261101/` | `policy_augmentation/` | Fresh 24-repetition, four-comparison family; neither 3% nonlinear superiority criterion established; both 5% linear noninferiority criteria established |
| `policy_augmentation_natural_20261101/` | `policy_augmentation/` | Separate 48-series M4 Train panel; descriptive only, with no independent-series inference or official M4 Test comparison |

The namespaces keep each study's values and provenance separate. The cross-fit
display namespace reads the completed computational reproduction, not an
unavailable original completed run. Its prior reported outcomes were known
before regeneration. New execution records cannot authenticate the missing
original bytes or historical seals.

### Principal dimension-selected experiment

The principal experiment is the sealed study in
`extensions/dimension_forecasting`. It compares ten complete procedures on
192 task suites, with independent B selection of each geometric procedure's
cap, C calibration and fresh D evaluation. Fixed-cap-two, native-cap-twenty
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

The builder needs Python, NumPy and Matplotlib. When this source is inside the
repository, the study directory can also be detected automatically. It does
not fit models, choose settings or read unreleased evaluation targets.

### Preprocessing and residual-target follow-ups

From a complete repository checkout, regenerate the follow-up display assets
with:

```bash
python build_followup_assets.py
```

The builder detects the study directories automatically. Explicit locations
can be supplied with:

```bash
python build_followup_assets.py \
  --preprocessing-root /path/to/repository/extensions/preprocessing_dimension \
  --crossfit-root /path/to/repository/extensions/crossfit_residual_reproduction_20261101
```

The optional `--output-root` selects a destination other than the default
`generated/`. Each follow-up namespace contains `numbers.tex`, `metrics.json`,
`asset_manifest.json` and compact generated tables. These are reporting
outputs derived from completed evidence. Regeneration requires the study
inputs identified by the builder; it does not rerun fitting or terminal
forecast scoring. The supplied generated files are sufficient to compile
both PDFs independently of those numerical records.

The preprocessing study establishes only linear-control noninferiority in its
four-comparison family. Positive fixed-common correction results remain
secondary. The cross-fit study establishes the shared-baseline nonlinear gain
and linear noninferiority, while the independent-RBF margin and expanded-policy
superiority are not established. In-sample KAHM's 12.4259% baseline gain and
3.9643% independent in-sample RBF gain are secondary; its 1.3862% higher mean
error than the expanded policy is a post hoc descriptive comparison. None of
these findings replaces an unmet primary claim or establishes that adding
KAHM improves a forecasting portfolio.

The legacy asset generators and `generated/reconstruction_mechanism/` belong
to the historical fixed-cap experiment. They are not the source of the
current principal or follow-up tables and cannot substitute for their
separate namespaces.

### Fresh policy augmentation and the natural panel

The new section evaluates the incremental effect of adding one centered
in-sample KAHM residual family to the retained non-KAHM menu. A second
augmentation adds a raw-history Nystrom RBF residual family with the same
number of additional B scores and C family slots. Search counts are matched;
feature widths, statistical capacity and computation are not equated.

The fresh simulation family establishes two of four criteria: both linear
noninferiority criteria and neither nonlinear superiority criterion. Point
nonlinear gains are 0.8162% over the base policy and 0.6582% over the extra-RBF
policy, below the prescribed 3% margin. This failure does not establish
equivalence or absence of a smaller improvement. The separate natural panel
has 2.2725% higher aggregate SMSE than the base policy, with KAHM selected in
three of 48 series. Standalone improvements against shared baselines remain
secondary mechanism evidence.

The source archive contains the compact published evidence used for these
displays under `evidence/policy_augmentation/`. Run:

```bash
python build_policy_assets.py
```

The builder requires Python and SciPy. It checks exact published Git blob
identities and the analysis-summary SHA-256 hashes; recomputes all four
paired endpoints from 24 saved repetition records and natural means from
all 48 per-series records; and writes tables, numerical macros and an asset
manifest under `generated/policy_augmentation/`. It does not refit models or
replay the original 11,616 raw forecast matrices. Synthetic cap/scaler
frequencies are read from the pinned published results, and natural
frequencies from the pinned metrics record; this is not an independent
reconstruction of every fitted bank. LaTeX compilation requires neither
SciPy nor the numerical archives because the generated displays are supplied.

The frozen scientific source is commit
`b1915813e98a59572a4bbb800adc2bd1f3fccc1b`; the completed result tables and
reporting records are from `b59410407a7cd34155aa1955fe0c801f62381364`, on
`research/kahm-policy-augmentation-2026-11-01`. The M4 Train snapshot is
`6c1067e5a57161249b17289a565178dc7a3fb3ca`. These are experiment-source and
results identities; they are not a claim that this new manuscript edition
has already been committed to that repository.

The 18 complete numerical-record archives are separate from this author
source. Their filenames and checksums are retained in the pinned
`reporting/ARCHIVE_INDEX.json`, with restoration instructions in
`reporting/DISTRIBUTION.md` under the bundled synthetic evidence. The author
source includes compact tables and provenance, not those full fitted states
and forecast archives. The original synthetic/natural audit receipts are
preserved with their stated scopes; the current revision rechecks numerical
reporting and document construction, not the entire original experiment.

## Frozen records and document changes

The stage records identify the principal dimension evidence:

| Record | SHA-256 |
| --- | --- |
| Source freeze | `1b80460fe2d33ede50bd085c026f7cdb6219e4f54057ff01d67c0e026fbcd60e` |
| Fitted-state lock | `6f8188e841bdf4b330f05f16912e428f74ad957cde3373ad9b80241d296d3e81` |
| Prediction lock | `074e7bd7186adb4f816daca3b46e2f8ab3bd8e7c331058aba20c0d8d10f29e7d` |
| Target release | `bfd871d11ff66c60847ba66766817071f3ba1eddb87d1c8b09bdd7e661c1c4f5` |

Each follow-up retains its own source freeze, fit lock, prediction lock,
target-release record and independent-audit receipt. Their study guides give
separate commands for regenerating data, fitting, replaying saved predictions,
reconstructing analysis and auditing the evidence.

This edition changes manuscript text, notation, references and
reporting guides. It preserves generated displays and leaves the frozen scientific source, configurations,
protocols, numerical analyzers, retained outputs and phase seals unchanged.
Document-release identities are recorded separately from experiment identities.
Compilation, numerical verification and independent scientific replication
are different operations.

## Numerical records and audit scope

The six separately supplied files
`KAHM_Forecasting_Dimension_Records_01.zip` through
`KAHM_Forecasting_Dimension_Records_06.zip` contain the complete principal
numerical records in disjoint repetition groups (0–3, 4–7, 8–11, 12–15,
16–19, 20–23). The first also contains the global inventories and stage seals.
Their member paths restore under `extensions/dimension_forecasting`, and the
accompanying `KAHM_Forecasting_Dimension_Records_manifest.json` records all
member and archive checksums. They contain prepared arrays, selected and
candidate states, predictions and execution records; code and analysis tables
are provided with the accompanying computational source.

The principal study retains 31,104 forecast arrays and 155,271,168 forecast
values across all candidate and reporting views. The ten calibrated primary
procedures account for 1,920 arrays and 9,584,640 values. All 192 tasks completed.
The independent audit regenerated episodes from the pinned generators and
checked seeds, noise, windows, target construction, B/C choices, 18,816 local
bases and 27 analysis tables containing 261,450 rows. These checks establish
computational consistency, not an independent scientific replication or a
general forecasting advantage.

The preprocessing package has a separate analysis ZIP and twelve records ZIPs,
as described in `extensions/preprocessing_dimension/RESULTS_PACKAGE.md`. Its
audit checks all aggregate arithmetic and primary contrasts, with raw
forecast-to-loss replay restricted to the prespecified eight repetition-zero
tasks. The cross-fit package contains one analysis ZIP and sixteen standalone
records ZIPs, with 3,311 retained files. Its audit covers all 192 tasks, 8,064
forecast arrays and 24,192 horizon losses. The cross-fit report links the exact
archive manifest and final delivery-verification receipt. These packages
support their respective studies and are not dependencies of LaTeX compilation.

## Supporting studies and historical provenance

The *Supplementary Studies* preserves the original complete comparison
families and numerical results while adding the completed preprocessing,
residual-target and policy-augmentation studies. Local-rank constraints in earlier procedures are
explicit properties of those experiments, usually a cap of two with the
specified ambient-dimension restriction. They are not evaluations of the
native default or the dimension-selected principal procedure. Different
targets, sampling units, training allocations and loss normalizations remain
separate.

The public source commit
`7dda356138675934aab5e327ec667bde0a5eda79` identifies the historical
reconstruction-mechanism implementation and archived experiment. It does not
identify the current principal dimension experiment, the follow-ups or this
coordinated manuscript edition. Use the study-specific source records and
the coordinated document-release record for those identities.

The original `KAHM_TMLR_Saved_Forecasts.zip` (80,805,506 bytes, SHA-256
`d149ae648b0b380eccf108d8d217566be173182a0ac047982afad4836fb92edd`)
remains a historical fixed-cap numeric supplement. Its original eight-table
verifier does not reconstruct the current dimension-selected primary study
or either follow-up. `package_tmlr_review.py`, `verify_saved_forecasts.py`
and the original `package_submission.py` retain their historical schema and
scope; they are not coordinated-evidence builders.

The standalone author source archive retains the original `revision_review/`
records with their historical scope. The previous source inventory is
preserved byte-for-byte at
`revision_review/prior_metadata/SOURCE_FILE_MANIFEST.json`; the immediately
preceding dimension-study edition's inventory is retained at
`revision_review/prior_metadata/dimension_SOURCE_FILE_MANIFEST.json`. The
preceding coordinated inventory is retained at
`revision_review/prior_metadata/coordinated_SOURCE_FILE_MANIFEST.json`. The root
`SOURCE_FILE_MANIFEST.json` inventories the delivered manuscript source files;
the older top-level `build_evidence/` receipts describe the preceding
dimension-study edition. The preceding coordinated document verification, independent
display check, visual reviews and isolated compilation receipt are in
`build_evidence/coordinated_20261101/` in the archive and
`../revision_evidence/coordinated_20261101/` in the repository. The new build
evidence under `build_evidence/softstate_20261102/` identifies the
current delivered edition. The immediately preceding policy-edition inventory
is preserved at
`revision_review/prior_metadata/policy_augmentation_SOURCE_FILE_MANIFEST.json`.

The editable source is an author package, not an anonymous review upload.
It preserves `LICENSE`, `DATA_LICENSE`, `LICENSE_TMLR.txt` and identifying
provenance records. Anonymous PDF formatting does not anonymize those records.
