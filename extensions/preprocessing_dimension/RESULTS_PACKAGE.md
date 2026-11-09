# Completed study and reproducibility package

The 24-repetition, 192-task study is complete. See [FINDINGS.md](FINDINGS.md)
for the scientific report, [REPORT_METRICS.json](REPORT_METRICS.json) for exact
headline values, and [the independent audit](audit/final_results_audit.json)
for its verification scope. The analysis established one of four prespecified
primary claims: linear-control noninferiority. The nonlinear superiority
claims were not established. The separate fixed-common mechanism experiment
yielded positive secondary evidence for centered KAHM residual corrections.

The numerical source and analysis definitions remain identical to the
[frozen source commit](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5cfd7a1745a52459b7b5045fe84803b9bbb2b56a/extensions/preprocessing_dimension).
This results addition does not change the frozen experiment. The local source
freeze preceded development-data generation; its GitHub publication followed
execution and is not a public preregistration.

## Evidence in this branch

- `analysis/` contains aggregate tables, primary paired contrasts, selection
  counts, integrity records, and numerical stability summaries.
- `reporting/figures/` contains four PNG/PDF figure pairs and exact plotting
  metadata. Descriptive secondary intervals remain unadjusted.
- `audit/verify_results.py` independently reconstructs primary geometry,
  selection decisions, all aggregate arithmetic, and the prespecified raw-loss
  replay subset. Its receipt states what was and was not checked.
- `FIT_LOCK.json`, `PREDICTION_LOCK.json`, and `TARGET_RELEASE.json` bind the
  completed stages. The source/configuration seal is `freeze.json`.

## Complete numerical records

The separately delivered `KAHM_Preprocessing_Dimension_Analysis.zip` contains
all 33 exported CSV tables, their checksums, the analysis summary, the audit
receipt, the figure manifest, and reproduction instructions. The large
per-forecast tables are included there rather than in Git.

Twelve disjoint `KAHM_Preprocessing_Dimension_Records_01.zip` through
`KAHM_Preprocessing_Dimension_Records_12.zip` preserve all prepared arrays,
fitted states, B/C predictions, terminal forecasts, diagnostics, and execution
receipts. Each contains two independent repetitions; the first also contains
global manifests and seals. Every archive member was read back and matched
to its original SHA-256 digest. Archive sizes, repetition ranges, and digests
are in [reporting/archive_index.json](reporting/archive_index.json). The full
member manifest accompanies the delivered archives and is also in the
analysis ZIP.

Restore all twelve record ZIPs at the repository root, preserving the
`extensions/dimension_forecasting` runtime and this extension. Extract the
analysis ZIP into a separate inspection directory; the saved analysis tables
can be compared with a fresh analysis of the restored records.

From this extension:

```bash
python code/pipeline.py verify
python analysis_code/analyze.py --root . --output verification/reanalysis
python audit/verify_results.py --root . --output verification/reproduced_audit.json
```

The independent auditor reads `analysis/`; retain this branch's analysis
tables and supply the complete table set from the analysis ZIP when running
the full audit. Existing result files are never overwritten. Use new output
paths for every replay. Reusing the same seeds reproduces this experiment;
it does not create a new independent research sample.

## Rebuild figures and documents

With the full original `analysis/` directory restored and the numerical
requirements installed, reporting additionally uses pandas, matplotlib,
ReportLab, Pillow, and the DejaVu font family. Run:

```bash
python reporting/build_figures.py
python reporting/build_findings.py
python reporting/render_documents.py --output /absolute/path/to/output
```

The [README](README.md) specifies numerical execution from the recorded
configuration, and the [protocol](PROTOCOL.md) fixes all split roles and
decision rules.
