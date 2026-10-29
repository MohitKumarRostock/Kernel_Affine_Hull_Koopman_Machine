# Source provenance and verification

## Frozen simulation

The protocol was finalized before any benchmark calibration draws. Its SHA-256
is `03609bea294099ac1fc169383e139667b21a78f375369fc58b371cc7f5c1cb63`.
A wording correction before sampling clarified that a lower bound on the
class minimum also lower-bounds risks of class members, but does not supply
an upper-risk or excess-risk guarantee for the empirical fitted predictor.
It did not change any law, confidence formula, allocation or schedule.

The pre-draw manifest records the exact configuration, environment and source
hashes. These computational files are unchanged since that manifest:

| File | SHA-256 |
|---|---|
| `core.py` | `d371981afa931067439ee3f249c931992f0eaf7a10e56b71076ff421f0b6c32b` |
| `run.py` | `e5254ab7dc243c92724077eae51e71aab696d265640bd4f573a68357294f45e1` |
| `test_core.py` | `a33b9cd75750fbbb615b0d2665d555caa3142581220ebf196ee7782cf00833d1` |

The compressed trial record SHA-256 is
`b93ac3f936c07a0deebca1c022fc2d6872f6edef4855ecdd5cab781d7b47ac4d`.
`SHA256SUMS.json` verifies the raw run files. The 9,600 scheduled trials and
76,800 method evaluations all completed. There were no solver failures or
fallbacks; the largest empirical primal–dual gap was
4.000000330961484e-11. The full simulation took 10.61 seconds in the recorded
environment; this elapsed time is not a per-method benchmark.

## Derived-analysis revisions

The initial analyzer was fingerprinted before sampling as
`2310cfadbfe4f1cbd2fc1268ab85a36a48d1405006150da2c2f0aad197c94a12`.
Three reporting changes were made after the raw simulation completed:

1. The noisy-affine law has exact minimum MSE 0.25, equal to the squared
   RMSE tolerance 0.50. Its guarded numerical oracle bracket therefore
   contains the threshold. The initial analyzer required strict separation.
   The revised analyzer retains its exclusion frequency and labels it
   `threshold_inside_numerical_oracle_bracket`, without assigning a false
   strict class-exclusion conclusion. The initial stopped analysis is
   documented in `analysis_attempt_01/failure.json`.
2. The first figure's legend intersected its bottom-axis labels. Increasing
   only the bottom margin corrected the overlap. The preceding verified
   analysis, analyzer source and figure are retained in `analysis_revision_01`.
   Its `summary.csv` and `exclusion_power.csv` are byte-identical to the final
   versions. The zero-risk figure uses display precision coarser than the
   numerical oracle guard, avoiding an artificial 1e-12 vertical scale.
3. The standalone table caption now expands empirical risk minimization
   (ERM) and Kullback–Leibler (KL). The preceding analysis and generator are
   retained in `analysis_revision_02`. Both numerical CSVs are again
   byte-identical; no formula, endpoint, curve or timing changed.

The final analyzer SHA-256 is
`510a1b198f3e9229677bf841998a612581c4cc9a54e18576580aff63bcabe0ef`.
No simulation was rerun and no raw endpoint, sample, law or selection rule
changed during these reporting revisions. `analysis/audit.json` records the
final analysis source hashes; `analysis/SHA256SUMS.json` verifies its derived
outputs. The raw run manifest intentionally retains the pre-draw source
identities rather than being rewritten after analysis.

## Verification and limits

The exact substantive test command, from the extension directory, is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s experiments/witness_comparison -p 'test_*.py' -v
```

All seven tests passed before the simulation. They check all eight known
population optima against feasible primal/dual brackets, the rare-pairing
calculation, the affine-versus-Lipschitz mismatch, omitted empirical support
and global extension, arbitrary nonnegative dual multipliers, exact binomial
coverage of ERM and centered endpoints, and the mass minimization against an
independent linear-program solver. They also reproduce the historical
0.9030575149728274 expectation lower bound under known zero input variance.

The final analyzer reconstructs every endpoint from the retained counts,
pair matrices and dual multipliers, with maximum absolute difference
1.1102230246251565e-16. It verifies the complete trial schedule and every raw
input hash. No endpoint lies in an ambiguous numerical oracle bracket.
All four primary methods have zero observed coverage violations on this grid.
The CSV summaries contain method/law/size binomial Monte Carlo intervals and
all three positive-tolerance exclusion frequencies, including unsuccessful
and false-exclusion cases. The analytic tests and the reconstruction audit
serve different purposes; numerical reconstruction alone does not establish
probability validity.

The numerical dual evaluations use explicit floating-point guards, not formal
interval arithmetic. Finite support permits small convex programs and known
geometric bounds; the measured costs do not establish computational behavior
for continuous, high-dimensional representations. Timings exclude data
sampling and representation lookup, use a precomputed finite-support
pair-score table, and are descriptive measurements of this prototype on one
machine. Each confidence statement is one-sided at 95% for one method and
one fixed trial; there is no joint 95% guarantee over methods or Monte Carlo
conditions.
