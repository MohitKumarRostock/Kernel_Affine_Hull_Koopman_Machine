# Local dimension and risk certification

This extension evaluates local subspace dimension in retained KAHM feature
banks. It compares cap two, the native cap twenty, and validation selection
over six caps on six independent-episode synthetic laws. All four prefixes of
each selected bank remain fixed before calibration, preserving their nested
class relation. The complete experimental design and confidence allocation are
in [PROTOCOL.md](PROTOCOL.md).

The implementation contains an isolated copy of the verified construction
policy and required native runtime. `runtime_sources.json` records the source
and SHA-256 digest of every copied file. Original experiments and manuscript
assets are separate from this extension.

## Reproduction

Run commands from this directory with Python 3.12 and the versions recorded in
the campaign manifest. NumPy, SciPy, scikit-learn, and Matplotlib are required.

For a new execution of the same declared design, choose an output directory
that does not yet exist. The tracked `runs/20261101` directory records the
completed study and must be preserved. A repeated execution uses the same
seeds and is not an independent research sample.

```bash
python study.py freeze --output runs/reproduction_20261101 --preflight audit/prefreeze_review.json
python study.py develop --output runs/reproduction_20261101 --workers 2
python study.py calibrate --output runs/reproduction_20261101 --workers 2
python study.py test --output runs/reproduction_20261101 --workers 2
python analyze.py --run runs/reproduction_20261101
```

Each stage checks source hashes and the preceding global seal. Completed or
failed attempts are retained; the runner does not overwrite them. Development
precedes all calibration draws, and the entire calibration stage precedes test
generation. Array analysis reads numerical archives and metadata without
unpickling the fitted models.

The 24 evaluated model aliases per repetition yield 4680 designated one-sided
endpoints across the campaign, jointly allocated failure probability 0.05.
This scope concerns this targeted synthetic study; it is not a reproduction of
the entire earlier forty-candidate comparison or a natural-data evaluation.

## Empirical results

[Scientific findings](FINDINGS.md) report the complete six-law comparison. Validation selection reduced mean KAHM test error on five laws and reproduced cap two on quasiperiodic; the native cap twenty had mixed effects. Selected-forecast certificate decisions were unchanged, and none of the simultaneous policy-improvement endpoints was positive. [Machine-readable metrics](REPORT_METRICS.json) and [analysis tables](runs/20261101/analysis) provide the exact values.

The complete numerical records are distributed separately as `KAHM_Risk_Dimension_Numerical_Records.zip`. Extract that archive at the repository root to restore `extensions/dimension_certificates/runs/20261101`. The archive includes the frozen schedule, fitted models, scores, retained full forecasts, generation hashes and global seals. The generated analysis tables, summaries and figures remain in this extension.

After extracting the records, the independent audit runs with:

```bash
python audit/post_release/final_results_audit.py --root . --run runs/20261101
```

The audit does not unpickle models. Its recorded result is [audit/post_release/final_results_audit.json](audit/post_release/final_results_audit.json).
