# Coefficient-budget sensitivity follow-up

This is a follow-up study designed after inspecting the main experiment. It uses
new seeds 2701, 2702, and 2703, with independent development, calibration, and
test streams under the unchanged main runner's seed-domain convention. The same
streams are reused across the three coefficient budgets as a common-random-number
comparison. The wrapper checks that they agree across budgets.

Budgets: 1, 4, 16. History: 12. Other main-v1 settings are unchanged. There are
21 candidate models per run, 12 system/seed replicates per budget, 36 runs total,
and 756 fitted candidates. Every scheduled run completed successfully.

The probability budget is .05/3 for each budget-specific full family (including
its paired gates). It gives joint confidence .95 over the three budget families
within a system/seed replicate, not over all twelve replicates. The summary
selects a stage within a family using validation only. It never selects a budget
using test performance.

All complete results and per-episode scores are retained under
`results/sensitivity_v1_k1`, `results/sensitivity_v1_k4`, and
`results/sensitivity_v1_k16`. The generated comparison files are
`results/sensitivity_summary.json`, `results/sensitivity_selected_results.csv`,
and `paper/tables/sensitivity_results.tex`.

Independent formula/provenance audit results:

| Budget | Runs | Candidate-horizon certificates | Paired comparisons | Status |
|---|---:|---:|---:|---|
| 1 | 12 | 756 | 108 | PASS |
| 4 | 12 | 756 | 108 | PASS |
| 16 | 12 | 756 | 108 | PASS |

No model reported nonconvergence. The maximum recorded optimization gaps were
9.33572e-8, 9.99774e-8, and 9.97032e-8 for budgets 1, 4, and 16 respectively.

The three campaigns ran concurrently with one BLAS/OpenMP thread per process.
Recorded campaign wall times were approximately 98, 101, and 103 seconds;
runtime is hardware dependent.
