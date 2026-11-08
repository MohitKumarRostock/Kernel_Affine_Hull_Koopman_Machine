# Reproduction tasks and available materials

This guide describes the files and commands in the enrichment publication branch. Different checks answer different questions: a document build checks presentation, a summary calculation checks arithmetic conditional on reported statistics, and a model replay checks predictions from saved numerical inputs.

For a guided comparison, start with the [worked examples and experiment guide](docs/EXPERIMENT_GUIDE.md). The explanation script uses the original primary table and only the Python standard library.

## Choose a task

| Task | Supported by this branch? | Inputs or requirements |
| --- | --- | --- |
| Read or compile the article and supplement | Yes | Supplied LaTeX, bibliography, styles, and static tables; Python and a compatible TeX installation |
| Check the 40 primary comparisons from summary statistics | Yes | Supplied `primary_contrasts.csv`, the summary checker, and SciPy |
| Recalculate the statistics from repetition losses | No | Repetition-level selected losses and the original saved-loss analysis code, or an independent implementation |
| Recompute fitted predictions | No | Original feature and prediction code, prepared episodes, fitted model states, and the required saved arrays |
| Regenerate the complete experiment | No | Generator and experiment-running source, exact task roster and stream labels, computational environment, and the full analysis and audit procedures |

The central protocol is `KAHM-SCORE-20261102-v1`. It specifies 64 repetitions under each of four laws and a family of 40 primary comparisons. Each procedure makes a separate validation choice using the same validation data within a task.

## Check summary arithmetic

Run from the repository root with Python 3 and SciPy installed:

```bash
python3 extensions/score_enrichment_20261102/verify_primary_summary.py
```

The checker verifies the supplied CSV's recorded identity and all 40 comparison identities. It checks sample sizes, the Student critical value, standard errors, interval endpoints, relative gains, and decisions. Its inputs are the reported means and standard deviations. It does not recover those statistics from repetition losses, construct features, refit models, or generate predictions.

A successful check means that these stored summaries are arithmetically consistent. Statistical coverage still depends on the sampling assumptions and the adequacy of the paired-Student approximation.

## Build the documents

From the repository root:

```bash
python3 papers/score_enrichment/scripts/build.py
```

The command compiles `papers/score_enrichment/main.pdf` and `supplement.pdf` from the supplied sources and static tables. The TeX installation must provide pdfLaTeX and the packages loaded by the preamble and bundled TMLR style. BibTeX is used when available; supplied `.bbl` files also support compilation. No experiment data or numerical archive is required.

The [publication manifest](papers/score_enrichment/PUBLICATION_MANIFEST.json) records the document-build inputs, and the [build receipt](papers/score_enrichment/DOCUMENT_BUILD.json) identifies the distributed PDFs. PDF bytes can differ between TeX environments. The document build does not verify forecasting calculations.

## Check the publication files and links

In a Git checkout containing the recorded predecessor commit:

```bash
git fetch origin archive/pre-manuscript-cleanup-2026-11-02
python3 scripts/verify_publication_tree.py --output-dir /tmp/kahm-publication-check
```

Use an output directory that does not already exist. This check compares the paper and study subtrees with the Git identities in [PUBLICATION_SCOPE.json](docs/PUBLICATION_SCOPE.json). It also checks committed working files, local Markdown links, pinned archive paths, and TeX input dependencies. The output receipt records the scope of these file checks. They establish file identity and document dependencies, not the correctness of numerical experiments.

## Requirements for full experimental reproduction

The availability records identify a separate author package, `KAHM_Step8_Repository_Update.zip`, with the expected experiment path `extensions/score_enrichment_20261102/release/`. That directory is absent from this publication branch. The package name and recorded source-manifest hash `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28` are provenance identifiers; they do not provide a download endpoint or establish that the package is accessible to a reader.

Full reproduction requires the original numerical implementation, the exact cohort roster, the relevant losses and predictions, and the analysis and audit code. Replaying saved fitted models requires the corresponding numerical arrays. Regenerating those arrays requires the original source, task specification, random streams, and environment.

The recorded experimental software versions are Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, and pytest 9.0.2. Those version strings belong to the original experiment; they are not the requirements for the lightweight summary checker or document build. A complete environment also includes platform and numerical-library details, which are absent here.

The experiment records describe sixteen numerical archives totaling 9.024 GiB. The archives and a public download endpoint are absent from this branch. The [availability record](extensions/score_enrichment_20261102/AVAILABILITY.md) documents the distribution status. Neither rebuilding the PDFs nor passing the summary checker supplies these missing model-reproduction steps.

## Related studies

The [research index](docs/RESEARCH_INDEX.md) links earlier studies at immutable commit `5123365917ae630b684f8a3bebb0294ee695ce7a`. Their targets, populations, settings, and comparison families differ from the central experiment.

To inspect an archived study, retain its original paths, manifests, and environment. A detached working tree provides access without creating a new branch:

```bash
git fetch origin archive/pre-manuscript-cleanup-2026-11-02
git worktree add --detach ../kahm-research-archive 5123365917ae630b684f8a3bebb0294ee695ce7a
```

Repeated execution of the same stored observations checks the recorded calculation. It does not provide new independent experimental evidence.
