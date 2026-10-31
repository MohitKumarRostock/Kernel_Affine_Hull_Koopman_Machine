# Building the KAHM forecasting manuscript

The entry point is `main.tex`. This directory contains the complete LaTeX manuscript, bibliography, official TMLR style, and all figures and tables required for compilation.

Install TeX Live with `latexmk`, pdfLaTeX, BibTeX, and Latin Modern fonts. From this directory, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The output is `main.pdf`. Compilation does not fit forecasting models or access an online service. The default source uses anonymous TMLR review mode.

The paper studies when KAHM reconstruction yields useful functions for a finite forecasting decoder. The mathematical analysis establishes conditional feature-span distinctness and a bound on residual correction magnitude. The empirical results support improvement over the common local-coordinate basis without establishing the prescribed advantage over all matched and competitive controls.

Generated displays are derived from the saved study evidence. The public repository provides the experimental source, inputs, selected model states, and verification tools through the study-specific reproduction guides. Each scientific checkpoint identifies the original experiment; the manuscript source archive identifies the document edition built from that evidence.
