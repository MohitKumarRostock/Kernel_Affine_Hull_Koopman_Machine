# Finite-support comparison of lower risk certificates

The protocol in `protocol.md` fixes the laws, sample sizes, repetitions,
confidence allocation and reported outcomes before calibration draws. The
benchmark targets one full scalar Lipschitz class; an affine-only diagnostic
is deliberately kept separate. It requires only the repository's NumPy,
SciPy and Matplotlib dependencies.

From `extensions/certified_forecasting`, run:

```bash
python -m unittest discover -s experiments/witness_comparison -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/witness_comparison/run.py --output results/witness_comparison_reproduced
python experiments/witness_comparison/analyze.py --results results/witness_comparison_reproduced --paper paper
```

The runner refuses an existing output directory. The analyzer refuses an
existing `analysis` directory; its `--paper` directory may be a fresh temporary
directory when checking reproducibility. Neither command refits forecasting
models or changes the historical experiments.

`trials.jsonl.gz` retains every trial's counts, unconditional and stratified
pair matrices, primal/dual optimization certificate, lower endpoints and
certificate-only timings. A separate deterministic child seed reconstructs
the full ordered iid sample. `population.json` contains the population
quantities and numerical oracle brackets. `manifest.json` records the protocol,
source hashes, configuration and runtime environment before sampling;
`SHA256SUMS.json` records immutable output hashes after completion.

The analyzer reconstructs all endpoints without solving another QP, verifies
schedule completeness and input hashes, and creates `summary.csv`,
`exclusion_power.csv`, `audit.json` and manuscript assets. Its reconstruction
is a numerical consistency check, not an independent proof of probability
coverage. The test module instead uses analytic oracle solutions, arbitrary
dual-feasible points, exact binomial coverage calculations and an independent
linear-program solution of the mass minimization problem.

Each method separately receives one-sided failure probability 0.05. Coverage
and exclusion frequencies across 300 simulated calibration samples have their
own Monte Carlo uncertainty. Neither the pooled frequency nor the simulation
itself proves distribution-free coverage. The primal/dual brackets include
explicit floating-point guards and feasibility checks; they are not formal
interval-arithmetic certificates. Time measurements exclude feature learning
and data generation and should not be interpreted as deployment benchmarks.
