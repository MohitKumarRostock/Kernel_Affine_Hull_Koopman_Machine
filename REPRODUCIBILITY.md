# Reproduction scope

## Documents

The approved manuscript is **Predictive Enrichment of Geometry-Induced Soft-State Abstractions: Information Retention and Decoder Access**. From the repository root, run:

```bash
python papers/score_enrichment/scripts/build.py
```

This builds `main.pdf` and `supplement.pdf` in the paper directory using the supplied static tables. It requires Python and a TeX installation with pdfLaTeX, BibTeX, and the imported packages. Supplied `.bbl` files permit a build without BibTeX. No simulation or numerical archive is needed. A branch-restricted GitHub workflow performs the same document build and publishes only the two PDFs and a document-build receipt. Its output is a document build, not a forecasting audit.

The publication manifest records the 41 approved document-build inputs. Their Git subtree before publication documentation is `2906ac70db2806f480fc4e3de1683c1859b848b1`. Matching these inputs establishes source identity, not scientific correctness. Recompiled PDF bytes may differ with TeX versions, fonts, timestamps, or document identifiers.

## Primary-summary arithmetic

The exact released `primary_contrasts.csv` is included under `extensions/score_enrichment_20261102/`. With SciPy installed:

```bash
python extensions/score_enrichment_20261102/verify_primary_summary.py
```

The checker verifies the original CSV hash, all 40 comparison identities, sample sizes, Student critical value, standard errors, confidence endpoints, effect ratios, and decision counts. It operates on reported means and standard deviations. It does not reconstruct the 2,560 paired repetition differences, authenticate predictions, or rerun a model. The distinction from a repetition-level saved-loss check is deliberate.

## Complete scientific execution package

The separately delivered author package contains the unchanged compact source/results release under `extensions/score_enrichment_20261102/release/`. That full release is not mirrored in this publication branch. It contains the frozen numerical implementation, cohort roster, selected losses, paired repetitions, analysis scripts, computational audit, and stage records. The recorded protocol is `KAHM-SCORE-20261102-v1`; its source-manifest SHA256 is `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`.

The historical freeze retains its original title and local-publication status. Publishing these identifiers after the experiment does not turn the local prospective freeze into public preregistration. The results receipt has a separate identity and cites files in the complete release; its references are not a claim that every file is included here.

The exact repository-overlay package is `KAHM_Step8_Repository_Update.zip`, 6,348,062 bytes, SHA256 `8ef7bca05707c86f87922de1698719e971d0ce2b182c2ce264cd6a3c6c414637`. It contains the complete author-delivered compact release and its file manifests. Its guarded application utility is intended for the original base checkout; it must not be forced over this publication edition or conflicting local changes. The complete author ZIP is separately identified in the study availability statement. Neither package name is a public download URL.

Full regeneration uses the complete source package in a new output tree, not a preserved completed-results tree. Its recorded environment uses Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, and pytest 9.0.2, with three workers and one numerical thread each. The environment preflight checks the recorded platform and library signature. The root requirements belong to older research and are not a replacement environment. No full regeneration was performed as part of repository publication.

## Model replay and availability

The original inventory describes sixteen large numerical archives totaling 9.024 GiB. The complete archives are not distributed here and have no established public endpoint. The only supplied copy available during editorial checking was incomplete and unreadable; it is not included in this branch. These delivery limitations do not by themselves establish an error in the original computation, but they prevent full replay from the supplied copies.

The retained computational audit reports distinct ridge calculations and forecast replay, while sharing the feature encoder and native primitives. It does not independently reimplement clustering or physical-law integration and is not external scientific replication. Archive hashes, saved-loss checks, model replay, refitting, and external replication have different scopes.

Related studies retain their own environments, source identities, and sampling units. Their reproduction instructions are linked from [the research index](docs/RESEARCH_INDEX.md).
