# Preparing the TMLR submission

The manuscript uses the official unmodified TMLR style in anonymous submission
mode. The packaging command creates an editable manuscript bundle and an
executable reproducibility supplement. It does not upload or submit either file.

## Build and verify

From this extension directory, after installing `requirements.txt`:

```bash
python -m reproduce audit
python -m reproduce review-diagnostics --output runs/review-diagnostics
python -m reproduce paper
python -m reproduce submission --output-dir dist/tmlr
```

The package command writes:

| File | Contents |
|---|---|
| `TMLR_Manuscript_Source.zip` | Anonymous LaTeX source, exact official TMLR styles/license, bibliography, figures, tables and build instructions; `main.tex` is at the ZIP root |
| `TMLR_Reproducibility_Supplement.zip` | Complete scientific snapshot, all compact historical records, executable runners/auditors, pinned dependencies/environment, Airfoil data, manuscript sources and any provided anonymized related manuscripts |
| `TMLR_PACKAGE_MANIFEST.json` | ZIP byte sizes, ZIP SHA-256 hashes and per-bundle content digests |

The diagnostic command above checks archived paired outcomes only and leaves
the manuscript's committed full diagnostic assets intact. To regenerate both
the paired analysis and saved-bank rank/gate checks, first complete a full run:

```bash
python -m reproduce full --workers 4 --run-dir runs/full-review
python -m reproduce review-diagnostics --run-dir runs/full-review --output manuscript/generated
python -m reproduce paper
python -m reproduce submission --output-dir dist/tmlr
```

Saved-bank checks are never promised by an invocation without `--run-dir`.
The normal paper build preserves the committed `review_*` assets.

`manuscript/main.pdf`, built separately by the paper command, is the submission
PDF. The prior named PDF is never copied into either ZIP. A source ZIP is a
convenience for rebuilding; TMLR's principal manuscript upload is the PDF.
The combined ZIP size is required to be below 100,000,000 bytes. Each ZIP uses
sorted names, fixed timestamps, stable permissions and deterministic content
manifests. ZIP byte equality is expected for unchanged inputs and the same
compression implementation; content digests are independent of ZIP metadata.

Extract the supplement into an empty directory and verify it before regenerating
assets:

```bash
python scripts/package_tmlr_submission.py --verify-dir .
python -m reproduce audit
python -m reproduce smoke
python -m reproduce full --workers 4 --run-dir runs/full-review
```

See the generated `REVIEWER_README.md` and [REPRODUCING.md](REPRODUCING.md)
for environment installation, resource needs, full job counts and audit scope.
The full route regenerates the prescribed seeds; it does not add independent
statistical confirmation. No network access is needed after dependencies are
installed. The review diagnostic route is descriptive reuse of archived
outcomes and does not alter either prespecified comparison family.

## Exact evidence and the review view

The package command first verifies `study/SNAPSHOT.json` and
`archive/MANIFEST.json`. It preserves every distributed `study/` and `archive/`
file byte-for-byte, including protocol freezes, margins, statistical outcomes,
source code and generic historical scratch paths. All nine inputs required by
the paper-asset generator are included. Historical hash records are evidence
of recorded internal plans, not external preregistrations or independent
timestamp attestations. New executions retain separate provenance.

Each `SUBMISSION_MANIFEST.json` contains a complete file inventory, SHA-256
hashes, a content-tree digest and a source-to-package mapping. The selected
source-tree digest identifies all included original files before optional
review metadata transformations; it does not claim to identify unrelated
repository contents. No link to a mutable current-study branch is required.

The review view excludes Git history, `CITATION.cff`, prior named PDFs, local
caches, environments and generated run directories. It removes the optional
author field from the staged `pyproject.toml`; the manifest records both the
original source hash and transformed package hash. That metadata view does not
rewrite the repository's authorship records or any scientific file.

Required license/copyright notices, ordinary published bibliography entries,
official style-file attribution and unchanged upstream provenance are retained.
The manifest explicitly lists these narrow exceptions. They document software
and literature sources, and are not an assertion about who submitted the paper.
The mechanical scan rejects unexpected first-party author/contact metadata and
current-study branch links. It does not promise zero names or make identity
inference impossible. Reviewers can inspect the published attributions without
assuming that the software's rights holders are the submitting authors.

Any related manuscripts distributed in `submission/related_work/` must be
prepared as anonymous PDFs before packaging. Their inclusion does not assert
an archival publication status. Inspect text, metadata, hyperlinks, bookmarks
and embedded files as part of preparing those PDFs.

## Submission checks

The final PDF should show anonymous authors, the TMLR review heading, readable
figures/tables, resolved references and an appendix after the references.
Use the official style without custom margin, font or layout modifications.
Read the current [TMLR author guide](https://jmlr.org/tmlr/author-guide.html)
and [submission instructions](https://jmlr.org/tmlr/submissions.html).

OpenReview requires the actual author list and other author-specific declarations
outside the anonymous PDF. The repository does not fabricate authorship,
conflicts, funding, submission status or overlap declarations. The scientific
scope remains the declared synthetic laws and secondary static-data check;
formatting, a successful build and computational reproduction do not by
themselves establish acceptance.
