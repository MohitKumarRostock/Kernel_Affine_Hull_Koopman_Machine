A Hierarchy of Risk Certificates for Forecasting Representations

PACKAGE CONTENTS

This package contains the editable main manuscript and experimental supplement,
their compiled PDFs, bibliography, all included figures and tables, and the
retained detailed study sources. main.tex and supplement.tex are the two
compilation entry points. The sections directory contains the article,
appendices, and detailed study text; supplement_sections contains generated
copies for the supplement. The compiled files are main.pdf and supplement.pdf.

BUILD

Inside the complete repository, from extensions/certified_forecasting, run:

    python scripts/build_manuscript.py

The standalone manuscript source package places the same script beside
main.tex. From that directory, run:

    python build_manuscript.py

The script deterministically regenerates supplement_sections from the retained
original study sources, expands table inputs, qualifies cross-document
references, and compiles the main paper before the supplement. It checks for
undefined references or citations, duplicate labels, and overfull boxes.
Use --prepare-only to regenerate the supplement sources without compiling.

For a manual build of the already generated sources, run in this directory:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

The build requires Python 3.9 or later, latexmk, pdfLaTeX, BibTeX, amsmath, mathtools,
enumitem, microtype, booktabs, lastpage, xr, and the standard PSNFSS times
package. The supplement reads main.aux for cross-references, so preserve that
file between the two compilation steps. The bundled
jmlr2e.sty is the JMLR formatting dependency. The document uses its preprint
option, 11-point text, and the standard JMLR text dimensions.

MANUSCRIPT

The main manuscript has 50 pages, including appendices and references; the
experimental supplement has 23 pages. The main abstract has 186
whitespace-delimited words. Its central result is a normalized scalar
Lipschitz block hierarchy: pairs are its first member, population expectations
increase, and compactness and bounded targets give convergence to the optimal
class risk. Exact examples state the limits of fixed block sizes. Independent
block calibration, stratification, and fitted-risk upper bounds support
representation-class decisions.

The revised introduction centers the representation-assessment question,
identifies the inherited KAHM ingredients, and states the hierarchy's
relationship to earlier risk bounds. A complete eight-law comparison and
learned-KAHM decision study are complemented by fresh calibration of all forty
archived KAHM banks. That new study compares four block sizes, two learned
partitions, and a common-sample empirical-minimum comparator under one 0.05
failure allocation over 1,760 endpoints. It supplies new evidence conditional
on fixed banks, not forty new representation-learning replications. It uses
KL lower calibration, while the earlier learned pair study used empirical
Bernstein lower calibration; block-size effects are compared within the new
study. An independent numerical audit reconstructs all 143,440 optimization
programs, 1,760 endpoints, and 880 decisions. All optimization stopping-limit
cases and the disclosed auxiliary-log provenance exception are retained.

The main text consolidates secondary forecasting and explicit-feature decision
studies. Their complete earlier text, tables, and figures are compiled in
supplement.pdf together with inherited KAHM reconstruction details and the
full experimental protocols. The broader scaling, matched-geometry, and speech
studies retain their adverse and inconclusive outcomes. All theorem,
proposition, and corollary statements and their proofs remain in main.pdf.
The result archives and exact reproduction commands remain public. No font
sizes or text dimensions were changed to reduce page count.

RESEARCH SOFTWARE AND DATA

Repository:
https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine

Research branch and extension:
https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/certified-soft-state-forecasting/extensions/certified_forecasting

Start with the extension README.md and docs/STUDY_GUIDE.md. The extension's
docs/REPRODUCIBILITY.md specifies environments, reproduction commands,
experiment configurations, data sources, and provenance. Separate study
directories document the certificate-comparison, controlled-decision, and
speech experiments. Frozen experiment records retain their original source
hashes; exact historical source verification uses the recorded source snapshot
or historical checkout identified by that guide.

The new learned-hierarchy study distributes its two largest gzip evidence
files as exact byte parts in results/learned_kahm_hierarchy_v1/archive_parts.
Before auditing these retained records, run from the extension directory:

    python scripts/restore_hierarchy_payloads.py

The helper verifies each part and each reconstructed original against its
SHA-256 hash. The restored files preserve the exact frozen bytes; this
packaging step does not modify the study. Manuscript compilation uses the
included tables and figures and does not require this restoration.

The public audio and trajectory archives remain at their attributed sources.
The repository provides acquisition instructions, checksums, and eligibility
rules. Statistical conclusions depend on the sampling units and assumptions
stated in the manuscript; per-window counts do not replace independent groups.

TABLES AND FIGURES

The extension contains scripts/make_paper_assets.py, make_scaling_assets.py,
and make_matched_assets.py for the forecasting, scaling, and matched-geometry
assets. Experiment-specific analyzers provide the other tables and figures.
Their commands and required result directories are specified in
extensions/certified_forecasting/docs/REPRODUCIBILITY.md. Asset generation uses
stored complete result records. It does not select configurations using test
performance. All numerical table entries and figure data in this package
correspond to those experiment records.
