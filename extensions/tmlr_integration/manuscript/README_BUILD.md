# Compile the integrated anonymous manuscript

This is the editable source of *Kernel Affine Hull Features for Prediction:
Conditional Gains, Costs, and Limits of Transfer*. It consolidates two unpublished
research studies and a separate bounded scaling sensitivity in the official
TMLR anonymous review format.

From this directory, using TeX Live, latexmk, pdfLaTeX, BibTeX, and Latin Modern:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The source archive places these files at its root. It contains all required
generated tables, number macros, and vector figures; forecasting experiments
are not required to compile it. The official `tmlr.sty`, `tmlr.bst`, and
`fancyhdr.sty` remain unmodified. Their exact upstream commit and hashes are
recorded in `TEMPLATE_PROVENANCE.json`. Do not add `accepted` or `preprint`
options to the review build.

The main paper explains the controlled mechanism, all primary capacity
contrasts, the independent natural-source comparison, measured computational
costs, and the unsuccessful frozen diagnostic. Proofs, exact generators,
method specifications, full source and policy tables, fitting warnings, and
other diagnostics follow the references. The scaling sensitivity reuses the
original draws and natural sources and is explicitly post-hoc. No result is
pooled across distinct inference units or statistical comparison families.

`ASSET_PROVENANCE.json` records the source hashes for the copied study assets
and newly generated confirmation/scaling tables and number macros. In a full research
checkout with the saved natural run restored, regenerate these assets using
`python extensions/tmlr_integration/build_assets.py` from the repository root.
The complete review supplement also includes that script and its inputs.
Verification of an extracted archive should precede rebuilding because
compilation creates auxiliary files.

The review source omits the author block, contact information, and identifying
current-study links. Published references and required third-party attribution
remain. The separate review supplement includes the anonymously cited related
manuscript and its supplement. The AI-assistance disclosure is on the first
page and records the actual scope of assistance. Scientific author review and
approval remain necessary before journal submission.
