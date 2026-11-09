# Reproduction and repository verification

The central forecasting experiment and the separate predictor-capacity study use different observations, loss scales, and comparison families. This guide identifies the inputs and computation behind each available command.

Start with the [experiment guide](docs/EXPERIMENT_GUIDE.md) for the scientific questions and worked comparisons, or the [reproduction directory](reproduction/README.md) for a map of the two bundles.

## Choose a task

| Task | Available route | What it establishes |
| --- | --- | --- |
| Interpret a central primary comparison | `scripts/explain_primary_comparison.py` | Explains the saved summary table using the Python standard library |
| Recalculate central statistics from per-run losses | `reproduction/score_enrichment/check_saved_losses.py` | Reconstructs procedure summaries, paired differences, and all 40 primary comparisons |
| Check the predictor-capacity source and archives | `python3 -m reproduce verify` in `reproduction/capacity_followup/` | Checks scientific source and archive identities using the standard library |
| Recalculate capacity and supporting first-stage statistics | `python3 -m reproduce audit` in that bundle | Computes statistics from saved per-repetition records using the supplied analysis source |
| Regenerate the capacity experiments | `python3 -m reproduce full` in that bundle | Generates observations, fits models, and compares regenerated outputs with the archive |
| Recompute central features or predictions, or regenerate its models | Requires unavailable simulation and fitting code, task list, and numerical inputs | Not supported by the central saved-loss bundle |
| Verify repository scope, scientific sources, and links | `scripts/verify_publication_tree.py` | Checks pinned code/evidence identities, documentation-manifest entries, tracked-file checksums, and documentation links |

## Central study: analysis of saved per-run losses

Protocol `KAHM-SCORE-20261102-v1` specifies 64 independent repetitions under each of four data-generating processes and 40 primary comparisons. Procedures make separate validation choices using the same observations within a task. The primary analysis uses test split D and the separately selected setting for each procedure.

From the repository root, using Python 3.12:

```bash
python3 -m pip install -r reproduction/score_enrichment/requirements.txt
python3 reproduction/score_enrichment/check_saved_losses.py
```

The bundle contains 17 experiment records. The script reads `evidence/selected_losses.csv.gz` directly and verifies its compressed and decompressed identities. It also checks the identities of the other 16 records.

A successful run prints JSON with `status: PASS`, 26,624 selected-loss rows, 416 procedure summaries, 2,560 paired differences, and 40 primary comparisons. The decisions are 16 lower candidate losses, 7 higher candidate losses, and 17 unresolved comparisons. To save the calculated tables, choose a new directory outside the bundle:

```bash
python3 reproduction/score_enrichment/check_saved_losses.py --output /tmp/kahm-score-analysis
```

The output contains `rebuilt_all_procedure_summary.csv`, `rebuilt_primary_paired_repetitions.csv`, `rebuilt_primary_contrasts.csv`, and `verification.json`. An existing output directory is rejected, and supplied records are not overwritten.

The script independently implements the statistical calculations from saved losses. It checks record completeness, allowed selected settings, and consistency across evaluation splits and selection rules. It does not establish that each recorded setting minimizes the complete validation grid or repeat the fitting and prediction audit. Training targets and prediction arrays are unavailable, so the script cannot reconstruct the standardized losses from squared forecast errors. The [bundle guide](reproduction/score_enrichment/README.md) specifies its checks and tolerances.

Statistical agreement is conditional on the saved losses and selected procedures. The paired-Student intervals retain their sampling and approximation assumptions. File hashes and agreement between calculations do not establish distribution-free coverage or replication with independent observations.

### Summary-statistic arithmetic

The protocol directory also supplies a narrower calculation:

```bash
python3 extensions/score_enrichment_20261102/verify_primary_summary.py
```

This command uses reported means and standard deviations and SciPy's Student quantile to check all 40 summary rows. The saved-loss script additionally reconstructs those inputs from per-repetition losses.

## Predictor-capacity study

The [capacity bundle](reproduction/capacity_followup/) contains experiment source and archived records. Its comparison uses 24 repetitions under three data-generating processes, MSE rather than the central study's SMSE, and a family of 12 comparisons. Supporting first-stage records determine its practical margins and have their own 29-comparison analysis.

From the repository root:

```bash
cd reproduction/capacity_followup
python3 -m reproduce verify
```

`verify` uses only the standard library and checks all 51 scientific snapshot entries and 17 archive entries, including decompressed gzip identities. It does not fit models or recalculate statistical comparisons.

For statistical analysis, use Python 3.12 and the bundle's pinned dependencies. From the same directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m reproduce audit --output runs/archive-audit
```

The audit checks protocol and margin provenance and recalculates all 29 first-stage and 12 capacity comparisons from saved records. Outputs are under `runs/archive-audit/stage1/` and `runs/archive-audit/capacity/`, with a summary in `runs/archive-audit/archive_audit.json`. It generates no observations and fits no models. The [bundle guide](reproduction/capacity_followup/README.md) explains the specification and analyzer hash comparisons reported by the audit.

To regenerate the first-stage, capacity, and numerical-sensitivity experiments, still from the capacity directory:

```bash
.venv/bin/python -m reproduce full --workers 4 --run-dir runs/full
```

This mode fits the specified models and checks regenerated predictions and losses against the archive. Add `--resume` to continue a compatible incomplete run. Reusing the specified random streams checks computational reproducibility and does not supply independent repetitions. These commands concern the capacity study; they do not regenerate the central 64-repetition evaluation dataset.

## Repository integrity

The repository verifier uses Git and the Python standard library. In a Git checkout, make the declared source commits available and run the check from the repository root:

```bash
python3 scripts/verify_publication_tree.py --list-source-commits
git fetch --filter=blob:none origin \
  00d7741f1feaf4693630f039d8d37b9db6d9f14f \
  5123365917ae630b684f8a3bebb0294ee695ce7a \
  a67c40190458bab7b3e56b4fac76770798d074b3 \
  f039e74a248e874c08530bab4617d5ad011d6d21
python3 scripts/verify_publication_tree.py --output-dir /tmp/kahm-repository-check
python3 -B -m unittest discover -s tests -v
```

Use a new output directory. [PUBLICATION_SCOPE.json](docs/PUBLICATION_SCOPE.json) identifies the protected scientific sources and allowed reader-documentation paths. The verifier checks the complete scientific file sets and their Git identities in the protocol directory and both reproduction bundles. It permits the central bundle manifest to update only the byte count and SHA256 for its documented README entry. Every scientific manifest field and all code/evidence identities remain checked. Local links must resolve to tracked files or directories, and pinned links to this repository must resolve within the declared source commits.

The report is written to `verification.json`. These integrity checks are separate from statistical analysis and model fitting. The GitHub Actions workflow runs the repository checks, the central summary and saved-loss analyses, and the capacity source/archive identity check. Its numerical dependencies are those in the central bundle's requirements file.

## Central execution materials

The complete central simulation and fitting source, exact `cohort.json` task list and random-stream labels, complete source snapshot, fitted model states, and forecast arrays are unavailable. Execution records identify `KAHM_Step8_Repository_Update.zip` and the path `extensions/score_enrichment_20261102/release/`; those materials are not supplied.

The recorded experiment-source manifest SHA256 is `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`. The saved-loss bundle's manifest identifies its own distributed files and evidence; it does not provide that experiment-source manifest. Experiment records list sixteen numerical archives totaling 9.024 GiB. The complete archive set is unavailable for independent size and integrity checks, and no public download endpoint is supplied. Archive volume is a distribution constraint, separate from the unavailable implementation, exact task list, and full numerical environment. Stored numerical outputs would not be essential to regeneration if the complete implementation and required inputs were available.

The [availability guide](extensions/score_enrichment_20261102/AVAILABILITY.md) summarizes the supplied files and required execution materials.

The recorded central environment uses Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, and pytest 9.0.2. The independent statistical-analysis script uses a separately validated Python 3.12 environment with pinned dependencies. These package versions do not supply the complete platform and numerical-library configuration.

## Other studies

The [research index](docs/RESEARCH_INDEX.md) links source code and records at immutable commits. Each study has its own data, protocol, and comparison family. Its implementation does not substitute for unavailable inputs to the central experiment.
