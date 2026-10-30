# Build the anonymous TMLR manuscript

This directory contains the anonymous review version of *When Kernel Affine Hull Geometry Helps Prediction: Local Coordinates, Decoder Capacity, and Conditional Gains*. It uses the official TMLR template in its default review mode. Do not add the `accepted` or `preprint` option for the review submission.

## Compile

From this directory, with a TeX distribution containing `latexmk`, pdfLaTeX, and BibTeX:

```sh
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The resulting PDF is `build/main.pdf`. The source archive `TMLR_Manuscript_Source.zip` places this directory's contents at its root, so the same command works after extraction. The manuscript source includes its generated numbers, tables, and figures; experiment execution is not required to compile it.

Required LaTeX packages, in addition to the bundled template, include `amsmath`, `amssymb`, `amsthm`, `mathtools`, `booktabs`, `microtype`, `enumitem`, `longtable`, `array`, `graphicx`, `hyperref`, `url`, `fontenc`, `lmodern`, `natbib`, and `eso-pic`. These are available in common TeX Live installations. The template supplies its own unmodified `fancyhdr.sty`.

## Template provenance

The following files are copied without modification from the official [TMLR style repository](https://github.com/JmlrOrg/tmlr-style-file) at commit `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`:

| File | SHA-256 |
| --- | --- |
| `tmlr.sty` | `816214ff5919aa457b6b443bee52b15d9561421417b7f8a50cc84651519f0002` |
| `tmlr.bst` | `306fd454cf40771bee01293eeb98d2c1cd5f4e11ed0cd7296b335f354fc45206` |
| `fancyhdr.sty` | `3d2922548e0e5f1a6c5676eda6ebb6dc20d7d305b4d8c2be5f1c833fb1084e6d` |

`LICENSE_TMLR.txt` is the repository's unmodified license. Copyright and attribution notices in the template files are preserved. The manuscript does not override the template's page size, margins, header, type size, or font selection. Tables use captions above their contents, and the bibliography precedes the appendix.

## Submission files and anonymity

Upload the compiled anonymous PDF as the manuscript. `TMLR_Manuscript_Source.zip` contains its source, and `TMLR_Reproducibility_Supplement.zip` contains the accompanying research artifact. Follow the reproduction guide in that supplement to audit historical results or execute the experiments.

The review source contains no named author block, affiliations, email addresses, identifying current-study repository link, or named PDF author metadata. Published work and the inherited third-party software remain cited in the third person; their attribution and licenses are retained. A related unpublished manuscript is cited anonymously; its anonymized manuscript and supplement accompany the research artifact as `Related_Manuscript.pdf` and `Related_Manuscript_Supplement.pdf`. No claim is made here that the author list or publication status of any prior work has changed.

`generated/review_diagnostics.tex` adds retrospective sensitivity and conditioning checks and is a required input to this revision. These checks are explicitly distinguished from the frozen primary analyses. The original experimental configurations and archived outcomes are not modified by the manuscript build.

## After acceptance

Prepare a separate camera-ready version using `\usepackage[accepted]{tmlr}`, restore the actual author details, and supply the publication month, year, and OpenReview link required by the official template. Do not use that version for the anonymous review submission.
