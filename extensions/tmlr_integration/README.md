# KAHM time-series forecasting: forecast-trained local geometry

This is an **unsubmitted research draft** with anonymous author metadata
retained pending author publication preparation. The integrated manuscript
preserves all previous studies and adds the separate exploratory gate pilot
and directly forecast-trained reconstruction experiment.

- [Current manuscript PDF](review/KAHM_Forecast_Trained_Manuscript.pdf)
- [Editable manuscript](manuscript/main.tex)
- [Direct reconstruction study](../direct_reconstruction/README.md)
- [Frozen protocol](../direct_reconstruction/PROTOCOL.md)
- [Previous proxy-metric findings](../target_geometry/FINDINGS.md)

The method fits a fixed baseline and geometric residual head, trains selected
KAHM reconstruction parameters against forecast error on B, selects on C,
and evaluates on D. Matched local controls and practical raw-input models
receive explicitly stated training and optimization opportunities. The
planned roster has 36 simulation and ten natural tasks. Three rideshare tasks
lack B windows and remain feasibility failures, making the complete
four-collection accuracy aggregate undefined.

## Audit and rebuild the current draft

From the repository root:

```bash
python -m pip install -r extensions/direct_reconstruction/requirements.txt
python extensions/direct_reconstruction/evidence.py unpack \
  --run-dir extensions/direct_reconstruction/results
python extensions/direct_reconstruction/analyze.py \
  extensions/direct_reconstruction/results --verify-live-source
python -m extensions.tmlr_integration.package_direct audit
cd extensions/tmlr_integration/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

All generated historical and current displays are included. The frozen
analyzer recomputes the five new TeX tables/macros from saved predictions;
no forecasting refits are required. The manuscript uses the bundled TMLR
preprint style, an **Unsubmitted research draft** header, and an accurate
AI-assistance disclosure. No journal submission has been made.

## Current deliverables

```bash
python -m extensions.tmlr_integration.package_direct build \
  --pdf /absolute/path/to/main.pdf --output /absolute/path/to/release
python -m extensions.tmlr_integration.package_direct verify /absolute/path/to/release
```

| Deliverable | Scope |
|---|---|
| `KAHM_Forecast_Trained_Manuscript.pdf` | Complete integrated manuscript |
| `KAHM_Forecast_Trained_Manuscript_Source.zip` | Editable LaTeX, bibliography, styles and all generated displays |
| `KAHM_Direct_Reconstruction_Reproduction.zip` | New frozen experiment, exact numerical dependencies, inputs, predictions, selected models, traces, failure records and separate preserved pilot |
| `RELEASE_MANIFEST.json` | Byte sizes, SHA-256 hashes and evidence accounting |

The packager runs the frozen analyzer on a disposable copy, replays selected
models, compares regenerated reports and tables exactly, and verifies byte
inventories. It does not refit candidate forecasters. The reproduction ZIP
contains unpacked evidence so its audit is directly runnable; the repository
uses lossless task archives to keep publication manageable. The prospective
scientific freeze is distinct from post-execution packaging inventories.

The [evidence-only CI workflow](../../.github/workflows/direct-reconstruction.yml)
checks numerical contracts, replays saved evidence, compiles the manuscript
and packages review files. Its tests use fabricated/development data; it does
not rerun the reserved forecast fits. Earlier packagers retain their stated
historical scope and must not be used to claim a complete reproduction bundle
for the newest study.

## Historical study documentation

The remaining sections document the original integrated artifact and its
preserved evidence. The original anonymous exporter `package.py` predates the
forecasting-utility extension and must not be used to present a complete
reproducibility bundle for the revised paper. Its source checks and historical
audits remain valid for their stated original scope.

The paper separates reconstruction accuracy, the information supplied to a
predictor, and the finite decoder's ability to use that information. Conditional
synthetic gains are presented alongside unfavorable natural-source comparisons.
The natural confirmation has four original source families; 80 confirmation
task configurations are not 80 independent forecasting domains. The new scaling comparisons reuse those sources and the original controlled
draws, so they are explicitly post-hoc sensitivity. Measured pipeline costs
include implementation overhead and do not establish an intrinsic
hardware-independent complexity ratio.

## Build the paper

Install TeX Live, `latexmk`, and the Latin Modern fonts. From this directory:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The original archived version used anonymous submission mode. Existing
source-derived tables and vector figures are bundled. Building the PDF does not
fit forecasting models. The companion `build_assets.py` prepares the integrated
paper assets from the unchanged source studies, archived natural reports,
and the independently audited scaling-sensitivity summary.

## Verify the public research checkout

From the repository root, first restore the committed reference artifacts if
needed. The original reference command is needed only when its run is absent;
the sensitivity restoration accepts already present identical files:

```bash
python extensions/natural_forecasting/package.py restore-reference \
  extensions/natural_forecasting/reference/natural_v1.zip.parts.json
python -m extensions.scaling_sensitivity.reference restore
python extensions/tmlr_integration/package.py check-source
python -m unittest extensions.tmlr_integration.package_tests -v
```

The source check verifies all 217 files in the original public checkpoint,
20 registered natural-study scientific dependencies, the controlled
scientific snapshot/compact archive, and 609 files in the new sensitivity
execution freeze. The original scientific sources and results remain intact.
The separate [sensitivity guide](../scaling_sensitivity/README.md) documents
1,664 new natural method runs and 1,440 controlled head fits on reused data,
including the public execution checkpoint, fixed two-arm design, and scope.

Restoration refuses to overwrite an existing run. The pinned scientific
dependencies are in `extensions/natural_forecasting/requirements.txt`; use
Python 3.12 and install them in an environment outside an extracted review
artifact. Then audit or replay both studies:

```bash
python extensions/tmlr_integration/package.py audit .
python extensions/tmlr_integration/package.py replay . --output ../tmlr-replay
```

The audit independently recomputes 2,496 task-method errors from saved natural
forecasts, verifies targets/origins and complete rosters, replays source-balanced
analyses, and refits the small diagnostic regressions to check frozen
coefficients. It also recomputes the archived controlled paired statistics.
The same audit reconstructs all 1,664 new natural sensitivity losses and all
1,440 controlled episode-loss summaries, validates the retained full-prediction
audit, and regenerates the sensitivity tables. Compact controlled replay is
distinguished from independently recreating its omitted full prediction arrays.
The audit fits no forecasting model or geometry bank. Replay writes to a
separate empty directory and preserves every measurement record.

## Original anonymous review exporter (historical scope)

After the final PDF has been compiled and visually checked:

```bash
python extensions/tmlr_integration/package.py build \
  --pdf /absolute/path/to/main.pdf \
  --output /absolute/path/to/tmlr-release
```

The output consists of:

| File | Purpose |
|---|---|
| `TMLR_Integrated_Manuscript.pdf` | Anonymous integrated review manuscript |
| `TMLR_Integrated_Manuscript_Source.zip` | Complete editable LaTeX, official style, figures, and tables |
| `TMLR_Integrated_Review_Supplement.zip` | Anonymous executable review artifact with source data, scientific code, frozen records, saved forecasts, and related unpublished manuscripts |
| `RELEASE_MANIFEST.json` | Byte sizes and SHA-256 hashes of the delivered files |

ZIP entries have deterministic timestamps and ordering. Both ZIPs together must
be below 100 MB. The PDF must be newer than all of its build inputs. Existing
named outputs require explicit `--overwrite`; unrelated files are preserved.

From a source checkout, verify the ZIPs directly:

```bash
python extensions/tmlr_integration/package.py verify /path/to/TMLR_Integrated_Manuscript_Source.zip
python extensions/tmlr_integration/package.py verify /path/to/TMLR_Integrated_Review_Supplement.zip
python extensions/tmlr_integration/package.py audit /path/to/TMLR_Integrated_Review_Supplement.zip
```

The supplement also contains these commands and a top-level
`REVIEWER_README.md`. It is a repository-shaped standalone review view, not a
copy of the complete public Git history.

## Exact provenance and anonymous review

The public original checkpoint remains immutable. The review ZIP omits
identifying historical overviews, first-party author/contact/citation metadata,
current-study repository links, old manuscript versions, Git history, and
unrelated files. One optional project-author metadata field is removed. In the
review view only, the identifying full-repository `CHECKPOINT.json` is replaced
by an explicitly post-study inventory of included unchanged source/data files.
The export manifest records the original and packaged hashes and describes
this transformation. It never presents the new inventory as an original
freeze or changes the prior scientific hashes.

The exact scientific sources, datasets, forecasts, analyses, initial and
confirmation freezes remain byte-identical. The original `VALIDATION.json`
retains its historical 217-file verification claim; the review checker states
its own selected-source scope. Original generic runtime paths are retained in
frozen records. Required licenses, published references, dataset metadata, and
immutable upstream-source provenance are declared attribution exceptions.
Mechanical identity scans do not guarantee anonymity against inference from
scientific content. The PDF and related-work PDFs also require visual review.

The older natural-study packager checks the complete public checkpoint and
must not be used as the entry point for the anonymous subset. Use this
integration's `package.py` for review exports. Fresh natural-study full runs
remain possible using unchanged scientific commands and the review inventory;
they must use a new run directory and create their own freeze.

## Author decisions before journal submission

This is an unpublished research draft. Computational checks and export
validation do not substitute for substantive author review. The authors must
verify the scientific interpretation and citations, confirm their intellectual
contributions and author list, settle funding and competing-interest
declarations, and approve the submission. The first-page AI-assistance
disclosure describes actual assistance and must not be reduced to language
editing alone. No submission, acceptance, publication, or author approval is
asserted by the source repository or package command.
