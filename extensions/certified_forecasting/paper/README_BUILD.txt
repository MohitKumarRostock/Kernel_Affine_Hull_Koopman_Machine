Certifying Prediction Limits and Representation Refinement
with Kernel Affine Hull Machines

This package contains the editable manuscript, compiled PDF, bibliography,
tables, figures, and preprint formatting dependencies. The research software,
protocols, and supporting evidence are maintained independently of the eventual
publication venue.

BUILD

From the directory containing main.tex, run:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The build requires pdfLaTeX, BibTeX, amsmath, mathtools, enumitem, microtype,
booktabs, lastpage, and the standard PSNFSS times package. The bundled
jmlr2e.sty is an unmodified third-party dependency from
https://github.com/JmlrOrg/jmlr-style-file.

LAYOUT AND SCOPE

The complete manuscript has 50 pages, including all appendices and references. The abstract contains 185 whitespace-delimited
words, or 195 when hyphenated terms are separated. The article class uses
twoside,11pt, Times Roman text, a six-inch text width, and an 8.5-inch text
height. The preprint package option suppresses publisher-specific metadata.
These presentation choices are separate from the software and experimental
protocols.

The scientific revision adds witness-slack analysis, stratified calibration
with estimated masses, minimum-feasible-cost certificates, stronger lower-bound
comparators, a controlled positive-tolerance study, and speech-energy forecasting.
Earlier experiment descriptions and repeated reproduction instructions are
consolidated. All proof arguments, primary comparison rows, adverse results,
and sampling qualifications remain. Detailed secondary scaling tables and the
complete matched fixed-dimension figure are retained as supplementary assets
in this package and the repository.

EXPERIMENTS AND EVIDENCE

Repository branch:
https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/certified-soft-state-forecasting/extensions/certified_forecasting

The extension's docs/REPRODUCIBILITY.md gives the Python environment, all
experiment commands, source provenance, and archive layout. The independent
studies have separate protocols and evidence directories:

    experiments/witness_comparison
    experiments/decision_value
    experiments/speech_forecasting

The first two compare the general representation certificates; their explicit
representations are not additional KAHM accuracy benchmarks. The speech study
uses a documented group split and a fixed bounded observable. Its original
scoring-order deviation and identical strict-order replay are both recorded.
The two executions form one study.

The source archive contains the final manuscript and verification account.
The complete experiment code, compressed trial records, derived arrays,
selected predictors, and numerical audits are in the repository. The provider's
audio archive is downloaded from its official source, with checksum and license
attribution in the speech study instructions.

TABLES AND FIGURES

From the extension root, regenerate the retained paper assets with:

    python scripts/make_paper_assets.py
    python scripts/make_scaling_assets.py
    python scripts/make_matched_assets.py --analysis results/matched_geometry_study/analysis
    python experiments/decision_value/make_assets.py
    python experiments/speech_forecasting/analyze.py --results results/speech_forecasting_v1_replay --output results/reproduced_speech_analysis --paper paper

The witness analyzer reconstructs all endpoints and writes assets while
requiring a new analysis directory; its README gives the reproduction sequence.
These commands read fixed evidence. They do not select favorable results or
refit forecasting models. The revised matched table reports absolute MSE,
fitting SD, relative effect, and paired seed counts together. The speech table
reports maximum upper endpoints across fits under one simultaneous allocation.

VERIFICATION

The extension's docs/SUBSTANTIVE_REVISION.md describes the scientific changes,
checks, outcomes, and limits. Machine-readable records are in
verification/substantive_revision and the three study directories. The release
includes 61 passing tests, independent endpoint reconstruction, numerical
optimization checks, table reaggregation, and rendered-PDF inspection.
The checks are not a formal proof system or an exhaustive novelty assessment.

Earlier readability and length revisions retain their own historical records;
their page counts and byte-identity statements describe those earlier editions.
The current final PDF hash and layout measurements identify this edition.
