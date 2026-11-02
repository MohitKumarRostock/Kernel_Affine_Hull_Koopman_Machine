# Input scaling on the speech-energy cohort

This study compares training standardization, raw inputs, training min--max
scaling to [-1,+1], and clipped min--max scaling for KAHM, RBF, and affine-hull
forecasts. The complete scientific specification is in [protocol.md](protocol.md).

The same recording groups were used in the earlier speech study, and that
study's results were already available. These comparisons are descriptive.
No new confidence bounds are computed; the earlier speech certificates apply
only to their original frozen predictors and allocation.

## Reproduction

Install the extension's documented dependencies and run from its root:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/speech_scaling/run_study.py --archive /path/to/mini_speech_commands.zip --output results/speech_scaling_reproduction
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/speech_scaling/verify_study.py --results results/speech_scaling_reproduction
python experiments/speech_scaling/make_assets.py --results results/speech_scaling_reproduction
```

The archive source and checksum are specified in the protocol. Every execution
requires a new output directory. Development uses all three seeds and all four
scaling arms, then freezes all 36 selected models before computing any test
loss. The 2,688 development candidates and all 26,460 selected-model test scores
are retained. The same 4,096 training rows per seed determine preprocessing;
geometry is subsequently fitted on a seeded 1,600-row subsample.

To audit the committed evidence without downloading audio or loading models:

```bash
PYTHONPATH=src python experiments/speech_scaling/verify_study.py --results results/speech_scaling_v1 --portable
python experiments/speech_scaling/make_assets.py --results results/speech_scaling_v1
```

`group_losses.csv.gz` is the primary portable test evidence. `development_*.json`
contains every validation outcome, all selected candidates, frozen scaler
parameters, and geometry subsample indices. `all_horizon_selected.csv` and
`all_horizon_summary.csv` average horizon-specific MSE with equal weights.
The sample standard deviations measure fitting variation on one fixed cohort.
`paired_differences.csv` reports each fit's paired group-level changes and
conditional standard errors. The latter are not simultaneous confidence bounds.

The model replay verifier independently rebuilds group losses and training
extrema, checks every validation selection, and compares standard/raw candidates
with the original study. The derived energy arrays, sampled training rows, and
selected models are reproducible local outputs; their hashes identify the
execution. They are omitted from Git under the existing binary-artifact policy.
The committed source and portable evidence suffice to regenerate them.
`source_snapshot.tar.gz` also preserves the exact runner, protocol, verifier,
feature and decoder modules, and inherited repository modules identified by
the experiment hashes. `source_snapshot.json` records every member hash.
