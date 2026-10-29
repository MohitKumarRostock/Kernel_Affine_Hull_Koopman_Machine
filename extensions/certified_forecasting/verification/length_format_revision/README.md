# Manuscript length and JMLR formatting verification

Date: 29 October 2026. Comparison revision:
`5300c4833ed849e712f30609eb257e570d064d3b`.

The revised manuscript contains a 187-word abstract and 49 pages, including
appendices and references. Counting hyphenated abstract terms separately gives
194 words. The official `article` class options remain `twoside,11pt`, and the
JMLR style file is unchanged. Times Roman text follows the journal's formatting
guide; text dimensions and line spacing retain their previous settings.
The official `preprint` option suppresses publication metadata that has not
been assigned, without changing the body font size or margins.

The revision consolidates repeated prose while retaining all definitions,
proofs, empirical findings, adverse results, and statistical qualifications.
The matched-study reproduction commands are consolidated in
[`paper/README_BUILD.txt`](../../paper/README_BUILD.txt); the appendix retains
the protocol and reproduction sequence. Four data figures use larger labels,
and the dense matched-MSE table uses nine-point text. No production fit or
experimental implementation changed for this editorial revision.

## Records

- `integrity.json` records the abstract counts, PDF page count and hash,
  source-preservation checks, and printed figure-font sizes.
- `plot_series_audit.json` compares the baseline and revised asset generators
  on identical retained inputs. All four figures, 28 axes, 85 lines, and 302
  coordinate pairs agree exactly, including axis scales and limits.
- `plot_series_values.json` retains the captured baseline and current plotted
  arrays and axis metadata supporting that comparison.

The theory, notation, method, optimization appendix, and exact-risk appendix
are byte-identical to the preceding scientifically reviewed sources. All
labels, cross-references, citation keys, and numerical table entries are
preserved. The figure files change because their typography and dimensions
change; their numerical content is invariant. The complete PDF was rendered
for visual review and builds without undefined references/citations, overfull
boxes, or oversized floats.

The earlier records in
[`readability_scientific_revision`](../readability_scientific_revision/)
document the preceding 53-page edition and its scientific/software checks.
They remain unchanged as historical evidence. The broader verification scope
and limitations are stated in
[`docs/SCIENTIFIC_VERIFICATION.md`](../../docs/SCIENTIFIC_VERIFICATION.md).

## Official formatting sources

- [JMLR submission instructions](https://jmlr.org/author-info.html)
- [JMLR formatting guide](https://www.jmlr.org/format/format.html)
- [Official sample and package options](https://github.com/JmlrOrg/jmlr-style-file/blob/master/sample.tex)

The length and abstract requirements are addressed. The authors remain
responsible for funding and competing-interest declarations, corresponding
author details, coauthor approval, and the required cover-letter statements.
