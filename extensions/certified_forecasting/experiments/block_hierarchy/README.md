# From pair witnesses to calibrated block minima

This experiment accompanies the manuscript's scalar block hierarchy. For a
block of `b` independent observations, minimize squared residual sum subject
to all pairwise Lipschitz constraints, then divide by `b-1`. The size-two
score is the original pair witness. Expected exact scores increase with block
size and converge to the class optimum for bounded scalar targets on a
compact metric representation space. These population properties do not
order finite-sample confidence endpoints.

The [protocol](protocol.md) fixes all eight earlier finite-support laws, 100
new independent streams per law, sample sizes 256/1024/4096, block sizes
2/4/8/16/32, and a full-sample empirical-minimum comparator. The complete
18-endpoint family has total failure allocation .05 within each law and
repetition. The observed zero violations across 800 streams do not establish
a new confidence theorem or a campaign-wide 95% event.

## Reproduce

From the extension root, choose a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python experiments/block_hierarchy/run.py \
  --output results/reproduced_block_hierarchy
python experiments/block_hierarchy/verify.py \
  --results results/reproduced_block_hierarchy \
  --output verification/reproduced_block_hierarchy.json
MPLCONFIGDIR=/tmp/kahm_mpl python experiments/block_hierarchy/make_assets.py \
  --results results/reproduced_block_hierarchy
```

The runner refuses to overwrite an existing directory. It freezes the
protocol and source snapshot before sampling. The asset command reproduces
the paper table and figure from every retained result; it does not select
successful laws or repetitions.

For the existing portable record, use the independent verifier described in
[VERIFICATION.md](VERIFICATION.md). It imports none of the experiment runner,
the legacy QP implementation, or the new general-feature solver.

## What the comparison shows

At n=4096, partial collapse has mean lower endpoint 0.0590 for pairs,
0.1373 at block size 16, and 0.1244 at size 32. Full-sample empirical minimum
is stronger at 0.2044. The population hierarchy and the uncalibrated mean
scores increase even where confidence endpoints decrease. On a two-atom law
whose pair witness is already exact, increasing block size weakens the
endpoint. Every law and every method is retained in `endpoints.csv` and
`summary.csv`.

The deterministic analytic curves cover rare conflicts and independent noise
on many separated representation points. They demonstrate both convergence
for each fixed distribution and the absence of a universal fixed block size.

The benchmark uses a finite-support solver with repeated-count caching, so
its wall-clock time is not a general continuous-support scalability claim.
`src/certified_forecasting/block_witness.py` separately supplies an audited
solver for arbitrary finite feature rows, including the maximum-block metric.
It returns a feasible dual lower score, primal upper score, gap, feasibility
residual, and numerical certificate. A primal objective alone is not a valid
lower certificate. Computation uses floating-point guards, not interval
arithmetic.

## Retained evidence and numerical scope

`results/block_hierarchy_v1/` contains all 800 raw categorical streams,
45,579 empirical QP certificates, 14,400 endpoints, eight population oracles,
3,066 exact-curve rows, a source snapshot, and independent verification.
The largest empirical primal-dual gap is 4.00e-11. The independent audit
reconstructs 4,166,400 block scores and all endpoints.

After the frozen run, the general-feature module received only defensive
input-validation and empty-dual handling changes. The recorded valid-range
score bound is unchanged. The exact executed module is retained inside the
source snapshot; the benchmark's finite-support QP solver was not changed.

The generic ideas of empirical-minimum lower bounds, batching, and
U-statistic optimization have established antecedents. The manuscript
identifies its specific centered Lipschitz hierarchy and its limitations;
it does not claim a new general concentration inequality.
