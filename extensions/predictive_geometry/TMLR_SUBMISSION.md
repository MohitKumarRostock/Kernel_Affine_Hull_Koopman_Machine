# Controlled-study archives and review materials

This guide describes the archival packaging tools for the controlled representation study. The current paper combines this study with later forecasting evidence and is available through the [integrated manuscript guide](../tmlr_integration/README.md). The exporter in this directory covers the controlled study alone.

## Build and verify the controlled-study package

From this extension directory, after installing `requirements.txt`:

```bash
python -m reproduce audit
python -m reproduce review-diagnostics --output runs/review-diagnostics
python -m reproduce paper
python -m reproduce submission --output-dir dist/tmlr
```

The commands verify archived results, generate paired-outcome diagnostics, compile the standalone controlled-study paper, and create its source and reproduction archives. They do not submit a paper to a journal.

| File | Contents |
| --- | --- |
| `TMLR_Manuscript_Source.zip` | LaTeX source, official TMLR style, bibliography, figures, tables, and build instructions for the standalone controlled study |
| `TMLR_Reproducibility_Supplement.zip` | Scientific source snapshot, compact historical records, runners, auditors, pinned dependencies, and supplied data |
| `TMLR_PACKAGE_MANIFEST.json` | Archive sizes, SHA-256 hashes, and content digests |

The first diagnostic command checks saved paired outcomes. Reconstructing rank and gate diagnostics from fitted banks additionally requires a complete run:

```bash
python -m reproduce full --workers 4 --run-dir runs/full-review
python -m reproduce review-diagnostics --run-dir runs/full-review --output manuscript/generated
python -m reproduce paper
```

The full route repeats the published seeds and retains fitted banks and predictions. It provides computational reproduction, not independent confirmation. Its environment, job counts, and audit scope are described in [REPRODUCING.md](REPRODUCING.md).

## Verify an extracted archive

Extract the reproduction ZIP into an empty directory. From its root:

```bash
python scripts/package_tmlr_submission.py --verify-dir .
python -m reproduce audit
python -m reproduce smoke
```

The packager verifies `study/SNAPSHOT.json` and `archive/MANIFEST.json`, then retains the included scientific source and archive bytes. Each archive manifest records its full file inventory, content hashes, and any metadata transformation used for an anonymous review view. Historical protocol records retain their original identities.

The source package and reproduction package have different purposes. The source package compiles the document; the reproduction package supplies the controlled experiment and its archived numerical evidence. Neither is a complete reproduction package for all studies in the integrated manuscript.

## Document and attribution scope

The anonymous export omits Git history, first-party citation metadata, prior named PDFs, caches, environments, and generated run directories. Its recorded transformations concern review metadata, not scientific source or experimental outcomes. Software licenses, ordinary published references, official template attribution, and immutable upstream-source provenance retain their accurate scope.

A mechanical metadata scan does not guarantee that authorship cannot be inferred from scientific content. The public repository is author-identifying. The integrated manuscript and any associated supplement require their own document and anonymity checks; the historical exporter does not certify later materials.
