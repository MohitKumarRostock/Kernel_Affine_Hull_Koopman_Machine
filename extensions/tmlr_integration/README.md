# KAHM forecasting: manuscript, supporting studies, and reproduction

**Kernel Affine Hull Features for Time-Series Forecasting: Distinctness, Bounded Corrections, and Predictive Limits** studies the functions supplied by local reconstruction, the magnitude of a residual forecast correction, and the predictive performance of a matched forecasting procedure.

- [Main manuscript](review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Supporting Studies](review/KAHM_Supplementary_Studies.pdf)
- [Editable source for both documents](review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Main LaTeX entry point](manuscript/main.tex) and [supporting-study entry point](manuscript/supplement.tex)
- [Principal experimental findings](../reconstruction_mechanism/FINDINGS.md)
- [Principal study reproduction guide](../reconstruction_mechanism/README.md)
- [Guide to all forecasting studies](../../TIME_SERIES_RESEARCH.md)

## Scientific scope

The main manuscript presents the core representation, its algebraic relation to matched kernel methods, a uniform correction-magnitude bound, and the principal reconstruction-departure experiment. Its appendices contain complete core proofs and the principal experimental protocol. The main text also summarizes the complementary findings and their limitations.

The self-contained *Supporting Studies* document preserves the controlled, natural-data, scaling, additive-correction, target-geometry, direct-training, and online adaptation investigations. Each study retains its own methods, sampling units, normalization, results, and qualifications. Their results are not pooled as independent observations of the principal comparison.

The empirical evidence establishes an improvement over the shared local-coordinate basis without establishing the prespecified practical advantage over all comparators. Feature-span distinctness, bounded corrections, and forecasting accuracy answer separate questions.

## Build the documents

Install TeX Live with `latexmk`, pdfLaTeX, BibTeX, and Latin Modern fonts. From the manuscript directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The outputs are `main.pdf` and `supplement.pdf`. Both use anonymous TMLR review mode. The [build guide](manuscript/README_BUILD.md) describes the source archive and attribution. All required display assets are included; compilation does not fit a forecasting model.

The public repository and the editable source archive retain author-identifying attribution. They are not anonymous review uploads. Scientific licenses and attribution are preserved in the editable source.

## Prepare the document package

The [document-package builder](package_submission.py) combines existing PDFs, the editable source dependencies, and an existing saved-forecast ZIP. It requires only Python's standard library and writes to a new output directory. From the repository root, supply the actual saved-forecast archive path:

```bash
python extensions/tmlr_integration/package_submission.py \
  --main-pdf extensions/tmlr_integration/manuscript/main.pdf \
  --supplement-pdf extensions/tmlr_integration/manuscript/supplement.pdf \
  --source-dir extensions/tmlr_integration/manuscript \
  --saved-forecast-zip /absolute/path/to/KAHM_TMLR_Saved_Forecasts.zip \
  --output-dir /absolute/path/to/new-release
```

| Artifact | Contents and intended use |
| --- | --- |
| `KAHM_Reconstruction_Prediction_Manuscript.pdf` | Main PDF, including core proofs and principal protocol; supplied separately for review |
| `KAHM_Supplementary_Studies.pdf` | Complete supporting-study document; also included in the review supplement |
| `KAHM_TMLR_Submission_Supplement.zip` | Supporting-study PDF, review README, and unchanged `saved_forecast_supplement/` files; checked to be below 100 MB |
| `KAHM_Reconstruction_Prediction_Manuscript_Source.zip` | Exact explicit dependencies for both entry points, build instructions, template provenance, and required licenses; not an anonymous upload |
| `RELEASE_MANIFEST.json` | Artifact sizes and hashes, source/template checks, evidence-integrity scope, and intended submission files |

The source archive excludes build auxiliaries, unused historical source fragments, obsolete PDF outputs, Git metadata, and machine-specific logs. Required license attribution remains intact. Its member list is determined by the two entry points and their explicit TeX, bibliography, and figure dependencies, together with an allowlist of build and attribution files.

The builder verifies the original saved-forecast manifest before copying all 400 member payloads unchanged. It checks archive readback and the review-size limit, but does not run a statistical verifier, refit or replay models, compile documents, inspect PDF appearance, or establish content anonymity. Those tasks have their own checks. Scientific checkpoint files are read without modification.

## Saved-forecast reproduction

The [saved-forecast builder](package_tmlr_review.py) creates `KAHM_TMLR_Saved_Forecasts.zip` from the restored principal-study evidence. To create that input archive, follow the [evidence-restoration guide](../reconstruction_mechanism/README.md), then run:

```bash
python extensions/tmlr_integration/package_tmlr_review.py --output-dir /tmp/kahm-saved-forecasts
```

The output directory must be new. The archive contains all 192 task suites and ten procedures, training targets and variance scales, eight reported numerical tables, and the [standalone verifier](verify_saved_forecasts.py). Its file size is approximately 81 MB; the combined document-package builder checks the final review archive's exact size.

After unpacking the review archive, enter `saved_forecast_supplement/` and run:

```bash
python -m pip install -r requirements.txt
python verify_saved_forecasts.py
```

The verifier checks file and array hashes, complete task and method coverage, normalized losses, repetition aggregates, selected-policy forecast identities, and all six primary contrasts. It recomputes the eight tables without importing forecasting models or deserializing fitted model objects. Its scope is saved-forecast and statistical reproduction. It does not refit models, repeat feature-bound checks, remeasure timing, or rerun the complementary studies.

## Full experimental reproduction and display generation

Full fitting, evidence restoration, selected-model replay, and independent-code checks are described in the [principal study guide](../reconstruction_mechanism/README.md). They retain their distinct scopes and numerical-portability limitations. Recomputing saved statistics or replaying a selected state does not add independent experimental observations.

After restoring the evidence, the principal displays can be regenerated from the repository root:

```bash
python extensions/tmlr_integration/build_mechanism_assets.py
python -m extensions.tmlr_integration.format_manuscript_assets
```

These commands reconstruct displays from saved evidence rather than fitting new models. Complementary displays have their own source studies and generation scripts. The scientific source and input checkpoints continue to identify the experiments that produced the numerical evidence; a document release manifest identifies the separately compiled edition.

## Historical records

The original integrated-paper exporter, `package.py`, covers an earlier subset of evidence. Its whole-checkpoint checks and anonymous exports apply only to that historical scope. They predate several complementary experiments and do not produce the present document package.

The principal study's `package_results.py` preserves the original scientific reproduction workflow, including its selected-state replay requirements. The new document-package builder does not replace or modify that scientific checkpoint. Historical verification records retain the hashes of the exact files they checked; they do not establish verification of a later document edition.

The repository does not assert that the documents have been submitted or accepted. Authors supply submission metadata, the final author list, funding information, conflicts, and approval of the scientific interpretation through the journal's submission process.
