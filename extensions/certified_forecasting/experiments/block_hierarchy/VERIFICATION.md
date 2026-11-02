# Independent verification of the block hierarchy study

The verifier imports neither `run.py` nor either Lipschitz optimization
implementation. It reads the frozen laws, retained categorical observations,
optimization certificates, and endpoint table. Run it from the extension root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/block_hierarchy/verify.py \
  --results results/block_hierarchy_v1 \
  --output verification/block_hierarchy_recheck.json
```

The output path must not already exist. The archived completed check is
`results/block_hierarchy_v1/independent_verification.json`.

## What is checked

The audit verifies the checksums of every original result file and every
execution source inside the frozen source archive. It replays all 800
categorical streams from the frozen seed and compares all 3,276,800 retained
observations exactly. It checks the complete eight-law, 100-repetition,
three-sample-size, six-method grid; no selection is made by endpoint quality.

For each of the 45,579 cached empirical optimization problems and eight
population problems, it independently reconstructs the scalar Lipschitz
constraint system, aggregates repeated representation points, and evaluates
the Lagrangian dual in extended floating-point precision. Nonnegative dual
multipliers give lower values. Primal predictions are checked against every
pairwise Lipschitz constraint and the target box, and their objective gives
an upper value. Closed-form constant and rare-conflict optima provide
additional checks. The audit verifies the recorded numerical guards and
primal–dual gaps; it does not accept a primal objective as a lower bound.

All 4,166,400 block scores and their means, variances, minima, and maxima are
reconstructed. An independent bisection routine computes each binary-KL
endpoint, including the full-sample empirical-minimum comparator. All 14,400
endpoints are checked against the table and the known class risk. The
reported simultaneous event is per law and repetition, covering all 18
method/sample-size statements; it is not a single 95% event across the 800
streams.

The 3,066 exact hierarchy table rows are checked against their formulas.
The formulas are independently verified in 33 small-block examples: multinomial
count enumeration for rare conflicts and exhaustive observation-sequence
enumeration for independent binary noise at separated representation points.
The nine tests in `tests/test_block_witness.py` separately check the reusable
block solver using analytic optima, the pathwise average-subset inequality,
translation and permutation invariance, duplicated blocks, and a differential
comparison with the earlier one-dimensional solver.

## Completed check

The archived independent audit reports zero score-summary discrepancy,
maximum independently evaluated dual discrepancy `2.22e-16`, no positive
primal constraint violation, and maximum empirical primal–dual gap
`4.00e-11`. Endpoint reconstruction differs by at most `1.01e-12`, reflecting
the runner's explicit downward numerical guard. The exact enumeration error
is at most `6.14e-14`. There are zero observed endpoint violations in the 800
complete streams.

These are numerical reproducibility and implementation checks. Floating-point
guards are not interval arithmetic, and zero observed violations in a finite
experiment are not a proof of the theorem's coverage guarantee. That guarantee
comes from the stated boundedness, independence, and confidence allocation.
The finite-support cache also makes runtime unsuitable as a generic timing
claim for continuously varying inputs.
