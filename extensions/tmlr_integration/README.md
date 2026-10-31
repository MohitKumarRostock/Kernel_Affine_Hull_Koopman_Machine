# KAHM reconstruction and prediction: manuscript and reproducibility

**Kernel Affine Hull Representations for Time-Series Forecasting: Distinctness, Stability, and Predictive Limits** connects the algebra of local reconstruction with the performance of finite forecasting decoders. It presents theoretical conditions, a matched reconstruction-departure experiment, and the relevant controlled, natural-data, and adaptive studies.

- [Manuscript PDF](review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Editable LaTeX source](review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [LaTeX entry point](manuscript/main.tex)
- [Principal experimental findings](../reconstruction_mechanism/FINDINGS.md)
- [Principal study reproduction guide](../reconstruction_mechanism/README.md)
- [Guide to all forecasting studies](../../TIME_SERIES_RESEARCH.md)

## Build the paper

The source includes the bibliography, official TMLR style, figures, and tables. Install TeX Live with `latexmk`, pdfLaTeX, BibTeX, and Latin Modern fonts. From the repository root:

```bash
cd extensions/tmlr_integration/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Compilation produces `main.pdf` without fitting a forecasting model. The manuscript uses anonymous TMLR review mode. The public repository remains author-identifying and is not an anonymous review artifact.

The source states the scope of AI assistance and separates mathematical results from empirical conclusions. A successful build establishes that the document compiles; the scientific claims depend on the assumptions and evidence described in the paper.

## Generate displays from saved evidence

The source package already includes every display needed for compilation. To reconstruct the adaptive and reconstruction-departure displays, restore the repository evidence using the [principal study guide](../reconstruction_mechanism/README.md), then run from the repository root:

```bash
python extensions/tmlr_integration/build_adaptive_assets.py
python extensions/tmlr_integration/build_mechanism_assets.py
python -m extensions.tmlr_integration.format_manuscript_assets
```

The builders compute summaries from the saved evidence and record input and output hashes. The formatting command applies the manuscript's caption and notation conventions without changing numerical table entries, and records the presentation transformation separately. These commands do not fit new models. Earlier displays have their own source studies and generation scripts. The manuscript's provenance files map generated assets to those inputs.

## Verify and package the principal study

The reconstruction-departure packager checks the original scientific checkpoint, saved analysis, internal audit, and selected-state replay. After the evidence has been restored:

```bash
python -m extensions.reconstruction_mechanism.package_results audit
python -m extensions.reconstruction_mechanism.package_results build \
  --pdf /absolute/path/to/main.pdf --output /absolute/path/to/release
python -m extensions.reconstruction_mechanism.package_results verify /absolute/path/to/release
```

| Artifact | Contents |
| --- | --- |
| `KAHM_Reconstruction_Prediction_Manuscript.pdf` | Compiled manuscript |
| `KAHM_Reconstruction_Prediction_Manuscript_Source.zip` | Editable LaTeX, bibliography, style files, and generated displays |
| `KAHM_Reconstruction_Mechanism_Reproduction.zip` | Scientific code, exact inputs, predictions, fitted states, and verification records for the principal experiment |
| `RELEASE_MANIFEST.json` | Byte sizes, SHA-256 hashes, and provenance of the packaged files |

The reproduction archive covers the reconstruction-departure experiment. It is not a complete collection of every historical dataset and experiment in the paper. The public repository provides the earlier studies through their separate guides. The repository's lossless evidence transport also includes the adaptive study, whereas the principal reproduction archive contains the mechanism study and its numerical dependencies.

The study-specific audit does not invoke older CPU-sensitive forecast replays. Those retain the environments and limitations reported in their original studies. Recomputing an analysis or replaying a saved state is a computational consistency check, not a new independent experimental observation.

## Anonymous saved-forecast supplement

The [review-supplement builder](package_tmlr_review.py) creates `KAHM_TMLR_Saved_Forecasts.zip`, an approximately 80.8 MB anonymous archive for numerical verification of the principal experiment. Restore the mechanism evidence first, then run from the repository root:

```bash
python extensions/tmlr_integration/package_tmlr_review.py --output-dir /tmp/kahm-tmlr-review
```

The output directory must be new. The builder writes an extracted supplement, the ZIP archive, and a package summary containing its size and SHA-256 digest. The large ZIP is distributed as a downloadable submission artifact; the repository contains the builder and [standalone verifier](verify_saved_forecasts.py).

After unpacking the ZIP, enter the extracted supplement directory, install its numerical dependencies, and run:

```bash
python -m pip install -r requirements.txt
python verify_saved_forecasts.py
```

The supplement contains every held-out forecast from all 192 tasks and ten procedures, the training targets and variance scales needed for normalization, and eight reported numerical tables. The verifier checks file and array hashes, complete task and method coverage, normalized losses, repetition aggregates, and all six primary contrasts. It recomputes the eight tables without importing forecasting models or deserializing fitted model objects.

Its scope is saved-forecast and statistical reproduction. It does not refit models, repeat feature-bound checks, remeasure timing, or rerun the earlier studies. Full fitting and verification use the separate public study code and evidence. This scope is stated in the supplement itself so that its numerical checks can be interpreted separately from complete experimental replication.

## Earlier studies and archival verification

The original integrated-paper exporter, `package.py`, covers an earlier subset of the evidence. Its whole-checkpoint checks and anonymous exports are meaningful only for that historical scope. They do not produce a complete reproduction bundle for the present manuscript. In particular, they predate the additive, target-geometry, direct-training, adaptive, and reconstruction-departure experiments.

Historical checkpoints include their original documentation and retain their original checksums. Whole-checkpoint verification requires the matching historical checkout or archive. The current reader documentation is maintained separately from these immutable records. Study-specific scientific freezes continue to identify the source and inputs that produced each reported result.

The principal paper is supported by several forms of evidence: complete prediction arrays where retained, compact historical summaries where full arrays are unavailable, independently evaluated fitted states, and prospective source checkpoints with their stated public or local status. The manuscript distinguishes these forms rather than treating them as interchangeable guarantees.

## Review materials

The review PDF and saved-forecast supplement are prepared for anonymous review. Public repository URLs, author metadata, Git history, and identifying provenance remain separate from those review materials. Necessary citations, data attribution, and software licenses retain their accurate scope.

The repository does not assert a journal submission or decision. Submission metadata, authorship, funding, conflicts of interest, and approval of the scientific interpretation are supplied by the authors through the journal's submission process.
