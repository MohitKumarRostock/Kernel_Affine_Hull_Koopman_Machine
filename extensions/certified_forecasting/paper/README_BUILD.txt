A Hierarchy of Risk Certificates for Forecasting Representations

PACKAGE CONTENTS

This package contains the editable manuscript, compiled PDF, bibliography,
all included figures and tables, and additional numerical tables and figures
that accompany the reported studies. main.tex is the compilation entry point.
The sections directory contains the article and appendices. The compiled PDF
is main.pdf. Editorial review reports are distributed separately from this
submission package.

BUILD

From the directory containing main.tex, run:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The build requires pdfLaTeX, BibTeX, amsmath, mathtools, enumitem, microtype,
booktabs, lastpage, and the standard PSNFSS times package. The bundled
jmlr2e.sty is the JMLR formatting dependency. The document uses its preprint
option, 11-point text, and the standard JMLR text dimensions.

MANUSCRIPT

The manuscript has 57 pages, including appendices and references. Its abstract
has 181 whitespace-delimited words. Its central result is a normalized scalar
Lipschitz block hierarchy: pairs are its first member, population expectations
increase, and compactness and bounded targets give convergence to the optimal
class risk. Exact examples state the limits of fixed block sizes. Independent
block calibration, stratification, and fitted-risk upper bounds support
representation-class decisions.

New evidence includes a complete eight-law block comparison and a fresh
learned-KAHM decision experiment with 20 independent trials per regime,
training-only min-max preprocessing, learned partitions, and one joint failure
allocation. The broader scaling, matched-geometry, and speech studies retain
their adverse and inconclusive outcomes. All proofs of the stated results
are included in the same PDF.

The main text consolidates the earlier fixed-grid forecasting and explicit-
feature decision studies. Their complete earlier text, tables, and figures
remain in this package: sections/experiments.tex, oscillators.tex,
sensitivity.tex, real_data.tex, and archive/decision_value_detailed.tex.
They are supplementary source material, not additional sections of main.pdf.
The result archives and exact reproduction commands remain public.

JMLR's author guidance requires a cover-letter length justification for
manuscripts above 50 pages. The separate revision report supplies a proposed
justification. No formatting dimensions or font sizes were changed to reduce
page count.

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
