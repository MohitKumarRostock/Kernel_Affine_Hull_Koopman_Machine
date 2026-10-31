# Focused development study: bounded adaptive KAHM

This extension implements the small development study requested after the
forecast-trained manuscript. It compares six online procedures on eight paired
synthetic streams and four previously inspected natural series, at two capacity
tiers. It is not independent source confirmation. The frozen protocol and
automatically reconstructed findings, including unfavorable outcomes, govern
the recommendation about a larger manuscript extension.

Read `PROTOCOL.md`, `DATA_PROVENANCE.md`, `config.json`, and the completed
`analysis/decision.json`. `DEVELOPMENT_FREEZE.json` fixes scientific source and
inputs before the reported runs. Original manuscript sources and archived
studies are unchanged.

The current source incorporates numerical amendment A1 after the original
complete attempt returned TECHNICAL_BLOCK. Read AMENDMENT_A1.md. The full
original source and evidence are preserved in archives/initial_attempt.zip.
The repeat uses the same data, models' mathematical objectives, calibration
grid and decision rules, with a stable correction-head solve applied to all
geometric arms. Both attempts are development evidence; neither is independent
confirmation. FINDINGS.md and the PDF explain the resulting manuscript decision.

## Reproduce

From the repository root, install the versions in this extension's
`requirements.txt`, then:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest extensions.adaptive_development.tests_models extensions.adaptive_development.tests_protocol -v
python -m extensions.adaptive_development.run --all --output /tmp/kahm_adaptive_replay
python -m extensions.adaptive_development.analyze --results /tmp/kahm_adaptive_replay --output /tmp/kahm_adaptive_analysis
python -m extensions.adaptive_development.audit --results /tmp/kahm_adaptive_replay --analysis /tmp/kahm_adaptive_analysis --output /tmp/kahm_adaptive_analysis/audit.json
```

Timing values vary with hardware, scheduling and environment. Numerical results
are generated from retained raw inputs and the exact fixed update rules; small
floating-point differences can depend on numerical libraries and CPU dispatch.
The saved evidence permits independent recomputation of targets, losses,
calibration weights, coverage, state caps and decision criteria without model
refitting.

`python -m extensions.adaptive_development.data --prepare` recreates input
archives. That preparation command additionally needs the four original
natural NPZs and source manifest from `extensions/direct_reconstruction/data`.
Ordinary forecasting replay requires only this extension's prepared inputs.

## What the comparison means

The raw baseline and all geometric heads fit current bounded-window objectives.
Adaptive KAHM, affine and RBF-local arms share reference contents and refresh
opportunities. KAHM uses the actual normalized signed reconstruction and folding
score. The kernel comparator is exact recursive sliding-window Gaussian kernel
ridge with a free intercept; it is not claimed to be published KRLS-T.

The two tiers cap geometry references and replay rows, not equal byte footprints.
The kernel comparator has M support rows per horizon, whereas geometry shares
M references across horizons. Model-owned NumPy bytes and whole-worker RSS are
reported separately. The 20 ms observation allowance is monitored rather than
enforced. Current adaptive state remains warm even if calibration chooses a
zero correction. No claim of minimal deployment cost or hard real-time safety
follows from this prototype.

## Attribution

Software follows the repository MIT license. Reused natural observations retain
their original CC BY 4.0 attribution and notices in DATA_PROVENANCE.md and the
prepared input manifest. AI tools assisted implementation, numerical checks,
analysis and reporting; this development package is not a submitted paper or
an external independent replication.
