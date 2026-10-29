# Decision-value study: retained results

The experiment evaluates class exclusion at root-mean-square error (RMSE)
tolerance 0.30 for three predefined representation costs, 1, 3, and 5, measured
by feature-coordinate count. In the repairable regime, the last feature block
reveals predictive information absent from the earlier blocks. In the
irreducible regime, the remaining target uncertainty cannot be removed by any
of the three representations. The exact class-optimal MSEs are `(0.138,0.138,0.01)`
and `(0.138,0.138,0.138)`. This is evidence about the general representation
certificate under a controlled mechanism, with no KAHM-specific performance
claim.

## Complete outcomes

Every prescribed sample size and all 100 independent repetitions per regime
were completed. Counts below are out of 100 repetitions.

| Calibration episodes | Repairable upper acceptance | Repairable minimum cost: ordinary lower | Repairable minimum cost: stratified lower | Irreducible, all classes excluded: ordinary lower | Irreducible, all classes excluded: stratified lower |
|---:|---:|---:|---:|---:|---:|
| 256 | 0 | 0 | 0 | 0 | 0 |
| 1,024 | 100 | 0 | 0 | 0 | 0 |
| 4,096 | 100 | 0 | 0 | 0 | 0 |
| 16,384 | 100 | 0 | 100 | 0 | 1 |
| 65,536 | 100 | 0 | 100 | 0 | 100 |

Validation selects the cost-5 predictor in every repairable repetition.
Upper calibration certifies that same predictor from 1,024 episodes onward. Lower certification
adds a class-level conclusion: both cheaper classes are inadequate, so cost
5 is minimal within the prescribed family. All irreducible upper decisions
remain unresolved. At 4,096 episodes, ordinary pairing excludes the constant
class in all repetitions of both regimes while stratified pairing excludes
none; the conditional score and probability intervals have a finite-sample
cost. No run or sample-size point is omitted.

One failure budget 0.05 covers all upper, ordinary lower, and both stratified
lower families, all three prefixes, and all five sample sizes **within a
regime and repetition**. It is not a single event across the 200 repetitions.
Exact finite-population evaluation found no interval-event failure among
those 200 instances. All 600 fits converged; their largest Frank–Wolfe gap
was `9.911812667750958e-11` and every coefficient budget was feasible.

## Evidence files

- `fresh/protocol.json`, `precalibration_manifest.json`, and
  `frozen_models_manifest.json`: the design and source/model hashes, frozen
  before generation of calibration observations.
- `fresh/development_states.npz`, `repairable_calibration_states.npz`, and
  `irreducible_calibration_states.npz`: all training, validation, and calibration
  atom indices. These compact arrays reconstruct every history and target.
- `fresh/frozen_models.json`: all fitted coefficients, population and
  development risks, optimization gaps, and seeds.
- `fresh/population_oracle.json`: complete finite-support enumeration,
  class-optimal risks, and ordinary and conditional population witnesses.
- `fresh/interval_details.jsonl.gz`: every upper interval, ordinary lower
  interval, stratum count, conditional interval, mass interval, and minimizing
  probability vector.
- `fresh/endpoints.csv`, `decisions.csv`, `decision_summary.csv`, and
  `coverage.csv`: all candidate endpoints and complete decision summaries.
- `fresh/summary.json`: completion statistics and SHA-256 hashes of retained
  experiment outputs. Empty cost cells in CSV, or null costs in JSON, mean
  positive infinity because no candidate meets the relevant condition.
- `verification.json`: independent reconstruction of all 600 fitted models,
  3,000 endpoint records, 6,000 mass linear programs, and 600 distinct data
  streams. The verifier checks the LP with vertex enumeration rather than the
  runner's greedy algorithm.

Each simulated state contains a Bernoulli variable `S` with probability 0.2
of value one, and three mutually independent uniform signs `Z`, `R`, and `N`,
independent of `S`. A uniform sign takes values -1 and 1 with equal probability.
The observed history is `(Z,S)`. The future target is `0.8*S*Z + 0.1*N` in the
repairable regime and `0.8*S*R + 0.1*N` in the irreducible regime. The unsigned
8-bit integer (`uint8`) state index stored in the archives is
`8*S + 4*((Z+1)/2) + 2*((R+1)/2) + (N+1)/2`. Calibration archive rows are the
100 repetitions in increasing seed order. Each smaller calibration sample
is the declared initial segment of that row. Known population masses and
risks enter the oracle evaluation only; empirical lower certificates use
simultaneous estimated-mass intervals.

## Original-campaign audit

`existing_audit/` contains all 1,796 candidate records from 62 original main,
oscillator, and coefficient-budget runs, reconstructed at three positive
tolerances and every horizon. All retained upper acceptance summaries were
reproduced. All 186 lower endpoints of the full-family validation winners
are zero; KAHM-only winners have 184 zero endpoints out of 186. Candidate
exclusions nevertheless occur: at tolerance 0.35, the main campaign has
301 accepted, 24 excluded, and 475 unresolved candidates, and the oscillator
campaign has 55, 30, and 155. These are descriptive counts over original
per-repetition confidence events. The retained records do not support an
independent reconstruction of their sample-size curves.

Reproduction commands and the exact protocol are in
`../../experiments/decision_value/`.
