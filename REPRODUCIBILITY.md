# Reproduction tasks and available materials

The central enrichment experiment and the supplementary predictor-capacity study have separate reproduction bundles. They use different observations, loss scales and comparison families. This guide identifies the inputs and computation behind each command.

Start with the [experiment guide](docs/EXPERIMENT_GUIDE.md) for the scientific questions and an interpreted comparison, or the [reproduction directory](reproduction/README.md) for a short map of the two bundles.

## Choose a task

| Task | Available route | What it establishes |
| --- | --- | --- |
| Interpret a central primary comparison | `scripts/explain_primary_comparison.py` | Explains the original summary table; Python standard library only |
| Rebuild central statistics from repetition losses | `reproduction/score_enrichment/check_saved_losses.py` | Reconstructs procedure summaries, paired differences and all 40 primary comparisons from saved selected losses |
| Check the supplementary capacity source and archives | `python3 -m reproduce verify` in `reproduction/capacity_followup/` | Checks the original scientific snapshot and archive identities; standard library only |
| Recalculate supplementary capacity and supporting first-stage statistics | `python3 -m reproduce audit` in that bundle | Reconstructs statistics from saved repetition records using the original analysis source |
| Regenerate the supplementary experiments | `python3 -m reproduce full` in that bundle | Generates observations, fits models and compares regenerated outputs with the archive |
| Recompute central features or predictions, or regenerate its models | Original generator, runner, roster and numerical states are still required | Not supported by the central saved-loss bundle |
| Compile the repository article and supplement | `papers/score_enrichment/scripts/build.py` | Builds the supplied documents and static tables; no numerical experiment is run |

## Central study: rebuild the saved-loss analysis

Protocol `KAHM-SCORE-20261102-v1` specifies 64 independent repetitions under each of four data-generating distributions and 40 primary comparisons. Procedures make separate validation choices using the same observations within a task. The primary analysis uses terminal split D and the independently selected setting for each procedure.

From the repository root, using Python 3.12:

```bash
python3 -m pip install -r reproduction/score_enrichment/requirements.txt
python3 reproduction/score_enrichment/check_saved_losses.py
```

The checker reads `evidence/selected_losses.csv.gz` directly. The bundle contains 17 original evidence records; the selected-loss CSV is losslessly compressed and the other 16 records retain their original bytes. Both the distributed file identity and the exact decompressed CSV identity are checked.

A successful run prints JSON with `status: PASS`, 26,624 selected-loss rows, 416 rebuilt procedure summaries, 2,560 paired differences and 40 primary comparisons. The decisions are 16 lower candidate losses, 7 higher candidate losses and 17 unresolved comparisons. To save the rebuilt tables, choose a new directory outside the bundle:

```bash
python3 reproduction/score_enrichment/check_saved_losses.py --output /tmp/kahm-score-analysis
```

The output contains `rebuilt_all_procedure_summary.csv`, `rebuilt_primary_paired_repetitions.csv`, `rebuilt_primary_contrasts.csv` and `verification.json`. An existing output directory is rejected. The supplied records are not overwritten.

This independently written checker reconstructs statistics from the original saved losses. It also checks record completeness, allowed selected settings and consistency across assessment splits and selection rules. It does not recover the original analysis program, establish that every recorded setting minimized the complete validation search, or repeat the recorded computational audit. Training targets and prediction arrays are absent, so the checker cannot recalculate the saved standardized losses from raw squared errors. The [bundle guide](reproduction/score_enrichment/README.md) specifies the checks, tolerances and original record identities.

Statistical agreement is conditional on the recorded observations and selected procedures. The paired-Student intervals retain their stated approximation and sampling assumptions. Neither hash agreement nor successful arithmetic checks establish distribution-free coverage or an external scientific replication.

### Original summary-only check

The original publication directory also supports its narrower check:

```bash
python3 extensions/score_enrichment_20261102/verify_primary_summary.py
```

This command uses the reported means and standard deviations and SciPy's Student quantile. It checks all 40 summary rows without reconstructing their repetition-level inputs. The independent saved-loss checker above supports the additional reconstruction from individual repetition records.

## Supplementary study: predictor capacity

The [capacity bundle](reproduction/capacity_followup/) contains the original experiment source and its archived records. The capacity comparison uses 24 repetitions under three data-generating distributions, MSE rather than the central study's SMSE, and a family of 12 comparisons. Supporting first-stage records determine its practical margins and retain their own 29-comparison analysis.

From the repository root:

```bash
cd reproduction/capacity_followup
python3 -m reproduce verify
```

`verify` uses only the standard library and checks all 51 scientific snapshot entries and 17 archive entries, including decompressed gzip identities. It does not fit a model or recalculate a statistical comparison.

For statistical analysis, use Python 3.12 and the bundle's pinned dependencies. From that same directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m reproduce audit --output runs/archive-audit
```

The audit checks protocol and margin provenance and rebuilds all 29 first-stage and 12 capacity comparisons from the saved records. Outputs are under `runs/archive-audit/stage1/` and `runs/archive-audit/capacity/`, with a summary in `runs/archive-audit/archive_audit.json`. It performs no observation generation or model fitting. The [bundle guide](reproduction/capacity_followup/README.md) explains the specification and analyzer hash differences reported by the audit.

To regenerate the first-stage, capacity and numerical-sensitivity experiments, still from the capacity directory:

```bash
.venv/bin/python -m reproduce full --workers 4 --run-dir runs/full
```

This mode fits the original models and checks regenerated predictions and losses against the archive. It is substantially more work than the saved-record audit. Add `--resume` to continue a compatible incomplete run. Reusing published random streams checks computational reproduction and does not add independent repetitions. This command concerns the supplementary study; it cannot generate the central 64-repetition evaluation dataset.

## Build and check the document files

The document directory contains LaTeX sources, bibliography entries and static tables. The commands in this section compile and check these files; they do not execute the numerical experiment.

From the repository root, with a compatible TeX installation:

```bash
python3 papers/score_enrichment/scripts/build.py
```

The command compiles the supplied article and supplement from their LaTeX, bibliography and static tables. The [publication manifest](papers/score_enrichment/PUBLICATION_MANIFEST.json) records document inputs, and the [build record](papers/score_enrichment/DOCUMENT_BUILD.json) identifies the distributed PDFs. PDF bytes can differ between TeX environments.

In a Git checkout containing the recorded archive commit, check the publication and reproduction file scopes with:

```bash
git fetch origin archive/pre-manuscript-cleanup-2026-11-02
python3 scripts/verify_publication_tree.py --output-dir /tmp/kahm-publication-check
```

Use a new output directory. The checker uses [PUBLICATION_SCOPE.json](docs/PUBLICATION_SCOPE.json) to verify recorded file identities and publication dependencies. It writes the file, link and dependency checks to `verification.json`. These checks are separate from the numerical analyses above.

## Central materials still required for model reproduction

The original central generator and experiment runner, exact `cohort.json` roster and stream labels, complete source snapshot, fitted models and forecast arrays remain unavailable here. The records identify `KAHM_Step8_Repository_Update.zip` and the path `extensions/score_enrichment_20261102/release/`; these execution files are unavailable.

The recorded original source-manifest SHA256 is `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`. The saved-loss bundle's manifest records its own distributed files and original evidence identities; it is not that original source manifest. The experiment records refer to sixteen numerical archives totaling 9.024 GiB. The 9.024 GiB figure comes from the experiment records; the complete archive set is unavailable for independent size and integrity checks. Those archives and a public download endpoint are not supplied. Archive volume is a practical distribution constraint, separate from the unavailable original implementation, complete task roster and environment fingerprint. Omitting stored numerical outputs would not by itself prevent regeneration if the complete execution materials were available.

The [availability guide](extensions/score_enrichment_20261102/AVAILABILITY.md) summarizes the supplied files and missing execution materials.

Original central software versions are recorded as Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0 and pytest 9.0.2. The independent saved-loss checker has a separately validated Python 3.12 environment and pinned dependencies. They do not by themselves supply the original platform or numerical-library fingerprint.

## Other studies

The [research index](docs/RESEARCH_INDEX.md) links other studies at immutable commits. Their protocols, populations and comparison families remain separate. The archive at `5123365917ae630b684f8a3bebb0294ee695ce7a` contains their code and records. Results from those studies are not substitutes for the central experiment's missing inputs.
