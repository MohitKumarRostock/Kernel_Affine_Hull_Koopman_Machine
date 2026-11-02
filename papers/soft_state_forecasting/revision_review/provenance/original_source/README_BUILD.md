# Building the KAHM forecasting documents

**Kernel Affine Hull Features for Time-Series Forecasting: Distinctness, Bounded Corrections, and Predictive Limits** has two document entry points:

- `main.tex` contains the focused manuscript, including the core mathematical proofs and the principal experimental protocol.
- `supplement.tex` contains the self-contained *Supporting Studies* document, with the complementary methods, results, technical appendices, and references.

Both documents use anonymous TMLR review mode. The main manuscript summarizes the complementary findings and their limitations; the supporting document provides their detailed evidence.

## Compile

Install TeX Live with `latexmk`, pdfLaTeX, BibTeX, and Latin Modern fonts. From this directory, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The outputs are `main.pdf` and `supplement.pdf`. The source contains the bibliography, official TMLR style files, and every figure and table needed by the two entry points. Compilation does not fit forecasting models or require an online service.

In a distributed source ZIP, these files are under `manuscript/`. `SOURCE_FILE_MANIFEST.json` identifies the included file bytes. `TEMPLATE_PROVENANCE.json` records the original style-file hashes; the style and its license are retained without modification.

## Interpretation and numerical reproduction

The main paper characterizes normalized target transfer, conditional feature-span distinctness, and a bound on forecast-correction magnitude. The principal experiment establishes improvement over the common local-coordinate basis but does not establish the specified advantage over every comparator. Complete proofs and the principal design accompany the main text. The supporting document retains the conditional gains, unfavorable comparisons, and incomplete evaluations from the complementary studies.

The separate saved-forecast supplement contains the principal experiment's numeric forecasts, normalization targets, reported tables, and standalone verifier. Its numerical checks recompute losses and statistical comparisons from saved arrays. They do not refit models, repeat model selection, rerun feature-bound checks, or measure computation time.

## Attribution and review files

The editable source archive retains the required `LICENSE`, `DATA_LICENSE`, and `LICENSE_TMLR.txt` attribution. Some attribution identifies the original authors. Consequently, the source archive is **not an anonymous review upload**, even though its default PDFs use anonymous review mode.

The intended review files are the separately compiled main PDF and `KAHM_TMLR_Submission_Supplement.zip`. The latter contains the supporting-study PDF and the unchanged saved-forecast supplement. Document compilation and archive integrity checks do not establish scientific correctness or a journal decision.
