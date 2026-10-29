Certifying Prediction Limits and Representation Refinement
with Kernel Affine Hull Machines

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

The manuscript has 53 pages, including appendices and references. Its abstract
has 189 whitespace-delimited words. The paper studies pairwise and stratified
risk lower bounds, upper bounds for fitted forecasts, representation refinement,
and minimum feasible representation cost. Numerical evidence includes
finite-support certificate comparisons, a controlled decision experiment,
synthetic forecasting, pen trajectories, and speech energy.

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
