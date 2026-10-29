Forecasting with Kernel Affine Hull Machines:
Representation Refinement and Risk Certification

This package contains the editable LaTeX manuscript, compiled PDF, bibliography,
tables, figures, and preprint formatting dependencies. The manuscript and the
accompanying repository use the same mathematical notation and experimental
definitions.

BUILD

From the directory containing main.tex, run:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The build requires pdflatex, BibTeX, amsmath, mathtools, enumitem, microtype,
booktabs, lastpage, and the standard PSNFSS times package. The bundled
jmlr2e.sty is an unmodified third-party LaTeX dependency from
https://github.com/JmlrOrg/jmlr-style-file.
These dependencies apply to manuscript compilation; the software and
experiments use the Python environment documented in the extension's
docs/REPRODUCIBILITY.md.

MANUSCRIPT LAYOUT

The current PDF has 49 pages, including appendices and references. Its abstract
contains 187 whitespace-delimited words, or 194 when hyphenated terms are split.
The current layout uses the article class with twoside,11pt, Times Roman text,
a six-inch text width, and an 8.5-inch text height. The preprint package option
suppresses publisher-specific metadata. Hyperlinks remain active with their
colors and borders hidden. This layout is a presentation choice that can be
adapted separately from the research software, experimental protocols, and
results when a publication venue is selected.

The shorter version consolidates repeated exposition while preserving all
proofs, notation, experimental values, citations, and scientific limitations.
All four data figures use labels of at least nine points at their printed
size, and the dense matched-MSE table uses nine-point text.

IMPLEMENTATION AND EXPERIMENTS

Repository branch:
https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/certified-soft-state-forecasting/extensions/certified_forecasting

Original experimental code baseline:
a8bcf6c8016499e1a4b64f7dd4196af43c60bd78

The repository's docs/REPRODUCIBILITY.md gives the environment, complete
experiment commands, source provenance, and archive layout. Original studies,
input-scaling experiments, and matched geometry--decoder comparisons retain
separate records. Synthetic confirmation uses new episodes. PenDigits
follow-ups use the same evaluation writers and provide descriptive comparisons.

To reproduce the input-scaling study from the extension directory:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src
    python experiments/prepare_pendigits.py
    python experiments/run_input_scaling.py --output results/reproduced_input_scaling --workers 4

To reproduce the matched geometry--decoder study:

    python experiments/run_matched_geometry.py --output results/reproduced_matched_geometry --workers 6 --freeze-only
    python experiments/run_matched_geometry.py --output results/reproduced_matched_geometry --workers 6 --resume
    python experiments/analyze_matched_geometry.py --study results/reproduced_matched_geometry --output results/reproduced_matched_geometry_analysis
    python scripts/make_matched_assets.py --analysis results/reproduced_matched_geometry_analysis

All reproduction commands use new output directories. The frozen configurations
and manifests identify the parameters and source revisions of the retained runs.

TABLES AND FIGURES

The following commands regenerate the scaling and matched-study assets from
complete retained analyses, without fitting models:

    python scripts/make_scaling_assets.py
    python scripts/make_matched_assets.py --analysis results/matched_geometry_study/analysis

Their command-line options accept alternative analysis directories. Numerical
table entries remain unchanged. Figure typography and dimensions were revised
for readability; all 85 plotted line series and their axis scales and limits
agree exactly with the preceding version. The oscillator table explicitly
labels the mean of the square-root upper endpoints; the matched-dimension
figure describes percentage changes in MSE.

REVISION AND VERIFICATION RECORD

The downloadable source archive includes review/REVISION_NOTES.txt, describing
the readability and scientific-consistency changes, and
review/readability_changes.diff, comparing the text sources with the supplied
manuscript. The repository's docs/SCIENTIFIC_VERIFICATION.md identifies
the checked proofs, statistical formulas, numerical records, tests, and limits
of the verification. The verification reconstructs retained evidence and does
not claim that every production experiment was rerun.
The shorter edition adds review/LENGTH_AND_FORMAT_REVISION.txt and associated
integrity and plot-series records. The earlier verification records describe
the preceding scientific revision and remain unchanged.
