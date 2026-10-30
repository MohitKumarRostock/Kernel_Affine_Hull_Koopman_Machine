# Reproducing the experiments and paper assets

## Supported environment

The reference execution used Python 3.12.14 on Linux x86_64, NumPy 2.3.5,
SciPy 1.17.0, scikit-learn 1.8.0, pandas 2.2.3, and Matplotlib 3.10.8.
`study/environment.json` preserves the original machine record. Each new run
also records its actual Python, platform, and dependency versions.

Install `requirements.txt` in an isolated Python 3.12 environment. The CLI
forces one BLAS/OpenMP thread per worker before starting scientific subprocesses.
Use `--workers 1` on machines with limited memory. Four workers match the
reference setup. Full generated states and predictions occupy roughly 0.4 GB;
allow at least 1 GB of free space for the run, source copies, summaries, and
temporary files. Runtime depends on CPU and linear-algebra libraries. Validation
results from the tested implementation are recorded separately in
`VALIDATION.json` when the release check is complete.

CPU architecture, BLAS, and library versions can affect floating-point results,
eigenspaces, and clustering. Exact byte equality is required for published
inputs and source snapshots. Reported MSE reproduction is checked numerically;
timestamps, runtime measurements, and new execution provenance are expected to
differ.

## Routes

Run all commands from `extensions/predictive_geometry`.

| Command | Work performed |
|---|---|
| `python -m reproduce smoke` | Unit and numerical checks, small execution examples, archived statistics, and asset generation |
| `python -m reproduce full --workers 4` | All declared experiments from their published random seeds, independent audits, numerical sensitivity, and new paper assets |
| `python -m reproduce audit` | Verify all committed snapshot/archive hashes and recompute both primary statistical families |
| `python -m reproduce audit --run-dir runs/NAME` | Audit generated predictions/losses and rebuild summaries for a completed full run |
| `python -m reproduce tables` | Generate manuscript assets from the compact committed archive |
| `python -m reproduce tables --input runs/NAME/results --output runs/NAME/paper_assets` | Generate assets from a fresh full computational reproduction |
| `python -m reproduce paper` | Rebuild archival assets and compile `manuscript/main.pdf` with `latexmk` |

A chosen output directory makes the full command easy to resume:

```bash
python -m reproduce full --workers 4 --run-dir runs/reproduction
python -m reproduce full --workers 4 --run-dir runs/reproduction --resume
```

Resume requires matching execution mode and orchestration/source manifests.
Completed scientific jobs additionally check their original source signatures.
An incomplete or corrupted job is not silently accepted. Use a new run directory
after intentional code changes. Output locations cannot overlap the source,
historical archive, or manuscript directories.

## Full execution order and counts

1. Verify the compact archive and exact scientific snapshot. Materialize two
   separate working directories without modifying any committed file.
2. Run two pilot repetitions of seven settings: 14 fitted banks. These verify
   replay; the already declared practical margins remain unchanged.
3. Run 24 first-stage repetitions of the three forecasting systems and four
   branch settings: 168 fitted banks. Three branch targets share each branch
   bank. The independent inference unit is a complete replication, not an
   origin, horizon, target variant, or method row.
4. Run 24 Airfoil model seeds on the fixed grouped cohort. These are descriptive
   fitting variations, not 24 independent observed datasets.
5. Run the capacity follow-up: 24 repetitions × three systems, four feature
   paths × two widths, plus one validation selector per bank. This yields 72
   banks and 648 method records. All data and fitted banks are generated in
   this execution using the original phase-4 seed rule.
6. Bind the newly generated 24 Van der Pol capacity banks, split signatures,
   and prediction archives to `sensitivity_execution_manifest.json`. Run all
   four stabilized feature/width pipelines for every bank: 96 refits.
7. Recompute the two statistical families, independently reconstruct saved
   episode/row losses from predictions, compare MSEs with the archived records,
   export compact results, and generate LaTeX tables and figures.

The same outer input scaler, clustering and reference selection, PCA bases,
geometric settings, clipping rules, data generators, and head grids are used
through the unchanged study functions. No fitting or selection uses test
targets. The sensitivity changes only the query-kernel row rescaling and then
refits/validates heads consistently with that feature map.

## Historical protocol evidence and current execution provenance

The original `protocol_freeze.json`, `followup_freeze.json`, and
`numerical_sensitivity_freeze.json` are immutable historical records of internal
freezes. They were not external preregistrations. The packaging layer never
rewrites these files to make new output hashes look historical.

The capacity freeze hashes the original stage1 `records.json`, including its
timings, because that record set supplied the capacity practical margins. New
stage1 execution necessarily has new timing metadata. Consequently:

- `stage1_work/` runs the first stage and writes newly generated stage1 records.
- `capacity_work/` retains the exact historical stage1 records solely as the
  evidence for the already frozen margins. Its capacity outcomes are generated
  anew in `results_capacity/` and `capacity_records.json`.
- The unchanged sensitivity runner is directed to the **newly named current
  execution manifest**, which hashes the newly generated phase4 banks and
  results. The original numerical freeze remains unchanged alongside it and
  is linked by its SHA-256 hash.
- `execution_manifest.json` versions the new orchestration code independently
  and records actual environment, completion steps, and output paths.

The first-stage statistical auditor also hashes full historical pilot records,
including timings. Its rerun therefore uses the unchanged pilot evidence for
margin provenance and newly generated confirmation/natural outcomes for
analysis. The rerun pilots' MSEs are compared with the archive separately.
Historical absolute filenames in the capacity margin evidence are resolved to
the exact-byte local copies by hash; the frozen margin file is not changed.
Fresh result exports also contain `margin_provenance_pilots.json.gz` and
`margin_provenance_records.json.gz`. The asset builder verifies their
decompressed hashes against the original margin evidence and uses them only
to check the frozen margins. All new outcomes, contrasts, and figures continue
to use the fresh result records.

The capacity auditor also displays two historical source changes between the
earlier numerical-margin freeze and the final pre-data amendment that added the
secondary validation selector. These are the analysis specification and its
auditor. The unchanged final `followup_freeze.json` binds both final versions;
the original margin values were not changed. This recorded amendment is retained
in audit output rather than hidden by rewriting an earlier freeze.

## Outputs and audits

A completed full run contains:

```text
runs/NAME/
  execution_manifest.json
  reproduction_comparison.json
  audit_summary.json
  stage1_work/                 exact source, generated banks and predictions
  capacity_work/               exact source, generated banks and predictions
    sensitivity_execution_manifest.json
  results/
    stage1/                   records, mechanism diagnostics, 29 contrasts
    capacity/                 648 records and 12 contrasts
    numerical_sensitivity/    96 records, summaries, prediction/numerical audit
  paper_assets/               generated LaTeX, figures, claim manifest
```

`reproduction_comparison.json` matches records by phase, replicate, setting,
feature, head, and decoder width where relevant. It reports the maximum MSE
drift and counts within `rtol=1e-6, atol=1e-10`. If any matched MSE exceeds that
tolerance, the full command and fresh-run audit return a nonzero validation
result, retain the execution outputs, and do not label the run a verified
reproduction. Platform-dependent differences therefore remain visible; they
are not silently replaced by archived results.

The prediction audit computes squared residuals and group averages independently
of the model's own loss function. First-stage primary/selector predictions,
every capacity prediction, and every sensitivity prediction are checked. The
required coverage is explicit: 2,304 first-stage records across 192 files,
648 capacity records, and 96 sensitivity records across 24 files. Missing
required prediction metadata causes an audit failure. The
normalizer audit re-evaluates original and stabilized test query routines rather
than accepting stored counts alone. Every reported comparison retains its
original independent-unit definition.

The committed archive is sufficient to verify all reported aggregate means,
both paired-statistical families, and the descriptive mechanism tables. Auditing
the predictions themselves requires generated full-run files; no large model
pickle is distributed through Git.

## Continuous integration and licensing

The parent repository installs the source workflow
`ci/kahm-predictive-geometry.yml` at
`.github/workflows/kahm-predictive-geometry.yml`. It runs the smoke route on the
research branch and uploads the execution manifest and generated assets. CI
does not pretend that a reduced smoke fit reproduces the complete study.

Source/data licenses and citations are in `LICENSE`, `THIRD_PARTY_NOTICES.md`,
and `CITATION.cff`. The paper metadata are provisional until author approval.
