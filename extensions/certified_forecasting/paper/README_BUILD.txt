Forecasting with Kernel Affine Hull Machines:
Representation Refinement and Risk Certification

This package contains the editable LaTeX manuscript, compiled PDF, bibliography,
tables, figures, and JMLR preprint style. The manuscript and the accompanying
repository use the same mathematical notation and experimental definitions.

BUILD

From the directory containing main.tex, run:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The build requires pdflatex, BibTeX, amsmath, mathtools, enumitem, microtype,
booktabs, and lastpage. The included JMLR style is unchanged from
https://github.com/JmlrOrg/jmlr-style-file.

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

All reproduction commands use new output directories. The frozen configurations
and manifests identify the parameters and source revisions of the retained runs.

TABLES AND FIGURES

The following commands regenerate the scaling and matched-study assets from
complete retained analyses, without fitting models:

    python scripts/make_scaling_assets.py
    python scripts/make_matched_assets.py --analysis results/matched_geometry_study/analysis

Their command-line options accept alternative analysis directories. Numerical
table entries and supplied figure assets are unchanged by this revision. The
oscillator table explicitly labels the mean of the square-root upper endpoints;
the matched-dimension figure describes percentage changes in MSE.

REVISION AND VERIFICATION RECORD

The downloadable source archive includes review/REVISION_NOTES.txt, describing
the readability and scientific-consistency changes, and
review/readability_changes.diff, comparing the text sources with the supplied
manuscript. The repository's docs/SCIENTIFIC_VERIFICATION.md identifies
the checked proofs, statistical formulas, numerical records, tests, and limits
of the verification. The verification reconstructs retained evidence and does
not claim that every production experiment was rerun.
