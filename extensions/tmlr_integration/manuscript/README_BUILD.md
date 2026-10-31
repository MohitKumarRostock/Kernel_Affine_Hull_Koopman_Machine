# Build the completed KAHM research preprint

Entry point: `main.tex`.

This source package contains the complete LaTeX manuscript, bibliography, TMLR preprint style, and all generated figures/tables. It reports the completed reconstruction-mechanism study and earlier preserved results. No forecasting model or online service is required to compile it.

Install TeX Live, `latexmk`, and Latin Modern fonts, then run in this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript uses `\usepackage[preprint]{tmlr}` and a Research preprint header. It has not been submitted for peer review. Anonymous author metadata is retained from the supplied manuscript. The first-page AI-assistance disclosure describes support for study design, implementation, verification, analysis and writing.

The final mechanism outcome is STOP_FORECASTING_PERFORMANCE_BRANCH. Useful performance over the common basis is established; the required advantage over matched reconstruction and practical controls is not. All generated numbers come from completed saved evidence. The separate reproduction package and public repository contain the scientific code and audits.
