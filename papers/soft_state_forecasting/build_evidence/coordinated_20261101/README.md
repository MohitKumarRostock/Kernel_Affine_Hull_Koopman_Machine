# Coordinated manuscript verification — 1 November 2026

This directory records verification of the coordinated manuscript and repository
presentation. It is separate from the scientific source freezes and production
audits of the three experiments. No models were fitted, forecasts changed, or
new primary tests introduced during this revision.

The principal dimension study retains six primary comparisons (three criteria
established). The separate preprocessing study retains four (one established).
The residual-target study retains four (two established); its computational
reproduction contributes zero additional independent repetitions. Favorable
fixed-common and in-sample comparisons remain secondary. The in-sample KAHM
versus expanded-policy ratio is a point description without a new test.

`input_snapshot.json` records the repository publication parent, supplied author
archive, original document hashes, and the scientific evidence protected from
editing. `verify_revision.py` checks preservation of those scientific inputs,
the original propositions and proofs, the reporting assets and local TeX inputs.
`principal_display_verification.json` is a fresh run of the existing independent
principal-study display verifier.

Run `python extensions/tmlr_integration/revision_evidence/coordinated_20261101/verify_revision.py`
from the repository root. The default mathematical baseline is the pinned
historical commit `f36e7c797f37d7d073dae42603b3770543d9fb43`, so publication of
this revision does not change the comparison. That commit must be present in
the local checkout; a shallow checkout may need its history fetched first.

`main_visual_review.json` and `supplement_visual_review.json` identify the PDFs
actually inspected. The reviewed PDFs are delivered unchanged. Visual review
is not an automated guarantee of every rendering condition.

`package_revision.py` retains the prior source archive's historical provenance,
overlays the revised source, and compiles both entry points in an isolated
directory. It checks their extracted text, page counts, local input closure,
and unresolved-reference/overflow warnings. The generated
`package_build_verification.json` records those checks. The archive's
`SOURCE_FILE_MANIFEST.json` inventories the exact delivered members; the
previous inventory is preserved in its historical metadata directory.

From the repository root, package after building and reviewing both PDFs:

```bash
python extensions/tmlr_integration/revision_evidence/coordinated_20261101/package_revision.py \
  --source-archive /path/to/prior/KAHM_Reconstruction_Prediction_Manuscript_Source.zip \
  --output-dir /path/to/deliverables
```

The prior archive is identified by SHA-256, not by its filename alone. The
script requires Python, pypdf, latexmk, pdfLaTeX, BibTeX, and pdftotext. It does
not download data or rerun scientific fitting, analysis, or production audits.

The final artifact hashes and page counts are recorded in
`../../review/COORDINATED_MANUSCRIPT_RELEASE.json`. Older top-level manuscript
`build_evidence/` receipts concern the preceding dimension-study edition and
remain historical records.
