# Reproduction scope

## Build the current documents

From the repository root:

```bash
python papers/score_enrichment/scripts/build.py
```

The command builds the article and supplement from the supplied tables. It requires Python and a TeX installation with pdfLaTeX and the imported packages. BibTeX is used when available; supplied `.bbl` files also support compilation. On Ubuntu, the document environment includes `texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-publishers`, and `lmodern`. No historical experiment directory or numerical archive is required.

The [publication manifest](papers/score_enrichment/PUBLICATION_MANIFEST.json) identifies 41 approved document-build inputs. The [original build receipt](papers/score_enrichment/DOCUMENT_BUILD.json) identifies the committed PDFs. A fresh build may differ in binary bytes across TeX environments. Document compilation is not a model audit.

## Check the published summary arithmetic

With SciPy installed:

```bash
python extensions/score_enrichment_20261102/verify_primary_summary.py
```

The checker verifies the exact primary CSV identity, all 40 comparison identities, sample sizes, critical value, standard errors, confidence endpoints, effect ratios, and decisions. It uses reported means and standard deviations. It does not reconstruct repetition-level losses or refit any model.

## Verify publication preservation

In a Git checkout containing the archived predecessor commit:

```bash
git fetch origin archive/pre-manuscript-cleanup-2026-11-02
python scripts/verify_publication_tree.py --output-dir /tmp/kahm-publication-check
```

Choose an output directory that does not already exist. The verifier checks the unchanged paper and study subtrees, all committed working files, local Markdown links, pinned historical paths, and TeX input dependencies. It writes a JSON receipt and a path/object inventory for files absent from the current checkout. The corresponding GitHub workflow builds the documents in a temporary directory and checks summary arithmetic without running experiments or modifying tracked PDFs.

## Complete experimental execution

The numerical implementation and complete compact results release are supplied separately in `KAHM_Step8_Repository_Update.zip`, inside `extensions/score_enrichment_20261102/release/`. They were not mirrored in the publication branch before cleanup and are not mirrored by this cleanup. The scientific-source manifest is `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`; the study identifier is `KAHM-SCORE-20261102-v1`.

That package supplies the frozen implementation, cohort roster, repetition-level loss tables, analysis and audit scripts, and stage records. Its names and hashes are provenance identifiers, not public download URLs. The original guarded overlay utility must not be forced over this paper-focused checkout or conflicting changes; use a separate compatible working tree and follow the package instructions.

The recorded experimental environment uses Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, and pytest 9.0.2. The platform signature is checked by the experimental preflight. These requirements belong to the complete scientific package, not the older root requirements retained in the historical archive. No simulation regeneration is performed by the cleanup or its workflow.

## Numerical archives and historical studies

The release inventory describes sixteen numerical archives totaling 9.024 GiB. The complete set has no established public endpoint. The only supplied copy inspected during editorial checking was incomplete and unreadable; it is not included here. This delivery limitation does not establish an error in the original calculation, but it prevents replay from those supplied copies. See [AVAILABILITY.md](extensions/score_enrichment_20261102/AVAILABILITY.md).

The [historical index](docs/RESEARCH_INDEX.md) links earlier studies at immutable commit `5123365917ae630b684f8a3bebb0294ee695ce7a`. To inspect or reproduce an earlier study, use a separate worktree at that commit, keeping its original paths, manifests, and environment:

```bash
git fetch origin archive/pre-manuscript-cleanup-2026-11-02
git worktree add --detach ../kahm-research-archive 5123365917ae630b684f8a3bebb0294ee695ce7a
```

Git history has not been rewritten. A paper-only checkout is smaller and easier to navigate, but a full-history clone still contains historical objects. Source identity, saved-summary checks, model replay, refitting, and external scientific replication remain different verification tasks.
